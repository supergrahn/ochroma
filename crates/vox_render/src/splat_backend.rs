//! Wraps `spectra_renderer::Renderer` in a dedicated OS thread so Bevy's render
//! schedule never blocks waiting for GPU work to complete.
//!
//! One frame of latency: `submit_frame()` sends work; `read_last_output()` returns
//! the result of the PREVIOUS submission. Acceptable for realtime.
//!
//! The pixel buffer is shared as `Arc<Vec<u8>>`: readers get an O(1) Arc clone with
//! no data copy. The write side uses `Arc::make_mut` to reuse the allocation when
//! no reader is holding the previous Arc.

#[cfg(feature = "spectra-native")]
use std::sync::mpsc::{Sender, channel};
#[cfg(feature = "spectra-native")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "spectra-native")]
use spectra_gpu::{CudarcSlangBackend, GpuBackend, VulkanSlangBackend};
#[cfg(feature = "spectra-native")]
use spectra_renderer::{RenderConfig, Renderer};
#[cfg(feature = "spectra-native")]
use spectra_scene_state::{CameraLayer, SceneState};

#[cfg(feature = "spectra-native")]
enum RtCommand {
    /// Update scene (new tessellated splat geometry), camera, and render one frame.
    Render {
        scene: Option<SceneState>,
        camera: CameraLayer,
    },
    /// Terminate the render thread.
    Shutdown,
}

/// Non-blocking frontend to `Renderer` running on a dedicated OS thread.
///
/// Submit frames via [`submit_frame`]; read the last completed frame via
/// [`read_last_output`] (O(1) Arc clone, no data copy).
#[cfg(feature = "spectra-native")]
pub struct SpectraRenderBackend {
    tx: Sender<RtCommand>,
    last_output: Arc<Mutex<Arc<Vec<u8>>>>,
    fail_count: u32,
    width: u32,
    height: u32,
}

/// Locate the Spectra `.slang` kernel directory: `SPECTRA_SLANG_DIR` if set and
/// valid, else the `spectra/slang` dir beside the engine repo (matching the
/// `../../../spectra/...` path deps in Cargo.toml). Returns `None` if neither
/// exists (the renderer then falls back to temp_dir and produces blank frames).
#[cfg(feature = "spectra-native")]
fn resolve_slang_kernel_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(d) = std::env::var("SPECTRA_SLANG_DIR") {
        let p = PathBuf::from(d);
        if p.is_dir() {
            return Some(p);
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spectra/slang");
    p.is_dir().then_some(p)
}

/// One-shot still: **path-trace** `splats` from a camera at `eye` looking at
/// `target` (`fov_y` radians) at `width`×`height`, accumulating `spp` samples on
/// the Vulkan backend, lit by a directional sun along `sun_dir` (direction toward
/// the sun). Returns RGBA8 (`w*h*4`). Synchronous — for cinematic stills; the
/// wgpu `TiledSplatRenderer` stays the interactive path. Builds a coloured,
/// camera-facing scene via [`crate::splat_convert::splats_to_lit_scene`].
#[cfg(feature = "spectra-native")]
pub fn pathtrace_splats_to_rgba(
    splats: &[vox_core::types::GaussianSplat],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    sun_dir: [f32; 3],
) -> Result<Vec<u8>, String> {
    use crate::splat_convert::{camera_layer, splats_to_lit_scene};
    use spectra_scene_state::LightLayer;

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    config.target_spp = spp;
    let mut renderer = Renderer::new(gpu, config);

    let mut scene = splats_to_lit_scene(splats, width, height, eye);

    // Spectra expects the direction from the shaded point toward the light.
    scene.lights = LightLayer {
        light_data: pack_vulkan_directional_light(sun_dir, [1.0, 0.97, 0.92], 5.0).to_vec(),
        light_count: 1,
    };

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::from(eye),
        glam::Vec3::from(target),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);

    let frame = renderer.render().map_err(|e| format!("render: {e:?}"))?;
    let n = (frame.width * frame.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..4 {
            out.push((frame.beauty[i * 4 + ch].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    Ok(out)
}

/// A flat PBR material for a path-traced triangle mesh (e.g. a Forge building).
///
/// The `*_tex` fields index into the `textures` slice passed to
/// [`pathtrace_mesh_textured_to_rgba`] (-1 = no texture). When `albedo_tex`
/// is set, the sampled texel REPLACES `base_color`; `roughness_tex` takes its
/// R channel; `normal_tex` is a tangent-space normal map. `uv_scale`
/// multiplies mesh UVs before sampling (textures wrap).
///
/// `transmission > 0.0` switches the material from opaque Lambert to REAL
/// transmissive glass (`MAT_GLASS`): the megakernel's `dispatch_sample`
/// already routes `MAT_GLASS` through the same `sample_glass` BSDF the proven
/// SDF window path calls directly (`megakernel.slang`:
/// `sample_glass(wo, n, ior=1.5, rough, absorption=float3(0), depth=1, ...)`)
/// — no shader change, only the packer's material-type slot. `ior` is the
/// glass index of refraction (1.5 = architectural glass; ignored while
/// `transmission == 0.0`). `thin_walled = false` (the default) is the full
/// refractive path; `thin_walled = true` is the thin-pane branch
/// (Schlick-Fresnel reflect, else straight-through tinted by `base_color`).
/// CAVEAT: the megakernel multiplies every bounce by `|n·wi|`; the refractive
/// branch's pdf cancellation absorbs that at near-normal incidence, but the
/// thin-walled branch (`f = albedo`, `pdf = 1`) does not, so thin panes lose
/// ~cos²θ per face at oblique view angles — prefer refractive.
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct PbrMaterial {
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission_strength: f32,
    pub albedo_tex: i32,
    pub roughness_tex: i32,
    pub normal_tex: i32,
    pub uv_scale: [f32; 2],
    /// 0.0 = opaque Lambert (the historical behavior); > 0.0 = transmissive
    /// `MAT_GLASS`.
    pub transmission: f32,
    /// Glass index of refraction; only read when `transmission > 0.0`.
    pub ior: f32,
    /// Thin-pane glass vs refractive solid glass; only read when
    /// `transmission > 0.0`.
    pub thin_walled: bool,
}

#[cfg(feature = "spectra-native")]
impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color: [0.5, 0.5, 0.5],
            roughness: 0.85,
            metallic: 0.0,
            emission_strength: 0.0,
            albedo_tex: -1,
            roughness_tex: -1,
            normal_tex: -1,
            uv_scale: [1.0, 1.0],
            transmission: 0.0,
            ior: 1.5,
            thin_walled: false,
        }
    }
}

/// A texture image for the path tracer's flat atlas.
///
/// Row-major, channels interleaved (`data[(y*width + x)*channels + c]`),
/// values LINEAR in [0,1] — sRGB decoding (e.g. PolyHaven diffuse JPEGs) is
/// the CALLER's job; the path tracer works in linear and applies no transfer
/// function. `channels` is 1–4 (3 = RGB, 4 = RGBA; missing channels read as
/// 0 with alpha 1 on the GPU). `data.len()` must equal
/// `width * height * channels`.
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub data: Vec<f32>,
}

/// The light rig the mesh path tracer uses: four directional lights (sun +
/// sky + camera fill + rim fill — a cheap stand-in for HDRI/physical-sky
/// ambient so surfaces not facing the sun stay readable in asset-review
/// stills) plus the renderer's ambient sky-dome gradient (`sky_dome_*`).
///
/// `Default` reproduces the rig that was hardcoded in
/// [`pathtrace_mesh_textured_to_rgba`] EXACTLY (same colors and intensities),
/// so rendering with `LightRig { sun_dir, ..Default::default() }` is
/// byte-identical to the legacy entry point. The fill-light DIRECTIONS are
/// derived from `eye`/`target` at render time (camera fill points from the
/// camera toward the scene, lifted 0.35 in Y; rim fill opposes it); the rig
/// only scales their intensities. Sky color is fixed at the legacy
/// `[0.58, 0.62, 0.72]`, camera fill at `[0.72, 0.74, 0.78]`, rim at
/// `[0.45, 0.47, 0.52]`.
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct LightRig {
    /// Direction TOWARD the sun (normalized at render time).
    pub sun_dir: [f32; 3],
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    /// Intensity of the fixed-color sky light shining straight down (+Y dir).
    pub sky_intensity: f32,
    /// Intensity of the camera-direction fill light.
    pub camera_fill: f32,
    /// Intensity of the rim light opposing the camera fill.
    pub rim_fill: f32,
    /// Ambient sky-dome intensity: the renderer's miss-ray gradient lights
    /// every surface (direct hemisphere visibility + indirect bounces), on
    /// top of the four directional lights. The legacy default (0.5 on a
    /// blue-grey gradient) is bright enough to wash chroma and flatten
    /// contrast in asset stills; review rigs typically lower it.
    pub sky_dome_intensity: f32,
    /// Sky-dome gradient color at the zenith (linear RGB).
    pub sky_dome_zenith: [f32; 3],
    /// Sky-dome gradient color at the horizon (linear RGB).
    pub sky_dome_horizon: [f32; 3],
}

#[cfg(feature = "spectra-native")]
impl Default for LightRig {
    fn default() -> Self {
        Self {
            // The conventional asset-still sun used by the GPU tests; callers
            // almost always override this per view.
            sun_dir: [0.3, 0.5, 0.8],
            sun_color: [1.0, 0.96, 0.90],
            sun_intensity: 4.0,
            sky_intensity: 0.9,
            camera_fill: 1.0,
            rim_fill: 0.45,
            // The spectra-renderer RenderState defaults — Default must stay
            // byte-identical to the legacy hardcoded rig.
            sky_dome_intensity: 0.5,
            sky_dome_zenith: [0.15, 0.25, 0.45],
            sky_dome_horizon: [0.7, 0.6, 0.5],
        }
    }
}

/// One-shot still: **path-trace a triangle mesh** (positions/normals/uvs/indices
/// + per-triangle `material_ids` indexing into `materials`) from a camera at
/// `eye` looking at `target`, lit by a directional sun. For detailed Forge
/// building meshes — the path tracer renders triangles natively (no splat
/// bridge). Returns RGBA8 (`w*h*4`). Additive.
///
/// Untextured: delegates to [`pathtrace_mesh_textured_to_rgba`] with no
/// textures; any `*_tex` indices in `materials` are ignored (no atlas bound).
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_mesh_to_rgba(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[[u32; 3]],
    material_ids: &[u8],
    materials: &[PbrMaterial],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    sun_dir: [f32; 3],
) -> Result<Vec<u8>, String> {
    pathtrace_mesh_textured_to_rgba(
        positions,
        normals,
        uvs,
        indices,
        material_ids,
        materials,
        &[],
        eye,
        target,
        fov_y,
        width,
        height,
        spp,
        sun_dir,
    )
}

/// One-shot still: **path-trace a textured triangle mesh**. Same as
/// [`pathtrace_mesh_to_rgba`] plus `textures`: the images materials reference
/// via `albedo_tex` / `roughness_tex` / `normal_tex` (indices into this
/// slice). Uploads them as the Spectra legacy flat atlas
/// (`Renderer::set_texture_atlas`), which the megakernel samples with EWA
/// filtering. Texture data must be LINEAR (see [`TextureImage`]). Returns
/// RGBA8 (`w*h*4`). Additive.
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_mesh_textured_to_rgba(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[[u32; 3]],
    material_ids: &[u8],
    materials: &[PbrMaterial],
    textures: &[TextureImage],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    sun_dir: [f32; 3],
) -> Result<Vec<u8>, String> {
    pathtrace_mesh_lit_to_rgba(
        positions,
        normals,
        uvs,
        indices,
        material_ids,
        materials,
        textures,
        eye,
        target,
        fov_y,
        width,
        height,
        spp,
        &LightRig {
            sun_dir,
            ..Default::default()
        },
    )
}

/// One-shot still: **path-trace a textured triangle mesh with an explicit
/// light rig**. Same as [`pathtrace_mesh_textured_to_rgba`] but the whole
/// 4-light rig (sun color/intensity, sky, camera fill, rim fill) comes from
/// `rig` instead of being hardcoded — see [`LightRig`]. With
/// `LightRig { sun_dir, ..Default::default() }` the output is byte-identical
/// to the legacy entry point. Returns RGBA8 (`w*h*4`). Additive.
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_mesh_lit_to_rgba(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[[u32; 3]],
    material_ids: &[u8],
    materials: &[PbrMaterial],
    textures: &[TextureImage],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    rig: &LightRig,
) -> Result<Vec<u8>, String> {
    use crate::splat_convert::camera_layer;
    use spectra_scene_state::{LightLayer, MaterialLayer, SceneState};

    // The kernel indexes `materials[material_id]` directly, so a short (or
    // slot-ordered-instead-of-forge-id-ordered) material table silently reads
    // the wrong material — the historical "brick on the window frames" zoning
    // bug. Trip immediately instead.
    debug_assert!(
        materials.len() > material_ids.iter().copied().max().unwrap_or(0) as usize,
        "material table too short: max material_id {} needs {} materials, got {} \
         (materials must be indexed by forge channel id — \
         see load_building_mesh_pbr_by_forge_id)",
        material_ids.iter().copied().max().unwrap_or(0),
        material_ids.iter().copied().max().unwrap_or(0) as usize + 1,
        materials.len()
    );

    let (tex_descs, tex_data) = build_texture_atlas(textures)?;

    let mut scene = SceneState::new(width, height);
    scene.geometry.vertex_count = positions.len();
    scene.geometry.triangle_count = indices.len();
    scene.geometry.positions = positions.iter().flat_map(|p| *p).collect();
    scene.geometry.normals = normals.iter().flat_map(|n| *n).collect();
    scene.geometry.uvs = uvs.iter().flat_map(|t| *t).collect();
    scene.geometry.indices = indices.iter().flat_map(|t| *t).collect();
    scene.geometry.material_ids = material_ids.iter().map(|&m| m as u32).collect();

    let mut params = Vec::with_capacity(materials.len() * VULKAN_MATERIAL_FLOATS);
    for m in materials {
        params.extend_from_slice(&pack_vulkan_mesh_material(*m));
    }
    scene.materials = MaterialLayer {
        params,
        spectral_spd: Default::default(),
        material_count: materials.len(),
    };

    // Sun + sky + camera fill: a cheap stand-in for the HDRI/physical-sky
    // ambient the real Forge+Crucible renders use, so surfaces not facing the
    // sun are still readable in asset review stills. Colors/intensities come
    // from `rig`; fill DIRECTIONS stay derived from eye/target (see LightRig).
    let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
    let camera_fill = (glam::Vec3::from(eye) - glam::Vec3::from(target) + glam::Vec3::Y * 0.35)
        .normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::new();
    for (dir, color, intensity) in [
        (sun.to_array(), rig.sun_color, rig.sun_intensity),
        // Sky, from straight above; legacy fixed color.
        (glam::Vec3::Y.to_array(), [0.58, 0.62, 0.72], rig.sky_intensity),
        (camera_fill.to_array(), [0.72, 0.74, 0.78], rig.camera_fill),
        (rim_fill.to_array(), [0.45, 0.47, 0.52], rig.rim_fill),
    ] {
        light_data.extend_from_slice(&pack_vulkan_directional_light(dir, color, intensity));
    }
    scene.lights = LightLayer {
        light_data,
        light_count: 4,
    };

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::from(eye),
        glam::Vec3::from(target),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();
    scene.mark_geometry_changed();
    scene.mark_materials_changed();
    scene.mark_lights_changed();

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    config.target_spp = spp;
    // Transmissive glass needs path DEPTH: a two-faced pane costs two bounces
    // before the ray even reaches the content behind it. near_realtime's
    // 3-bounce budget plus the NRC query-at-bounce-3 early exit would render
    // panes black-by-config. Only scenes that actually contain a transmissive
    // material pay for the deeper budget — opaque scenes keep the historical
    // config (and their renders) byte-identical.
    if materials.iter().any(|m| m.transmission > 0.0) {
        config.max_bounces = 8;
        config.use_nrc = false;
    }
    let mut renderer = Renderer::new(gpu, config);
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    // The ambient dome from the rig (set_sky_gradient needs the render state,
    // so it must come after load_scene_state). With Default rig values this
    // re-applies the RenderState defaults — byte-identical to the legacy rig.
    renderer.set_sky_gradient(
        rig.sky_dome_zenith,
        rig.sky_dome_horizon,
        rig.sky_dome_intensity,
    );
    // ORDER MATTERS: set_texture_atlas silently no-ops before scene state
    // exists, so it must come after load_scene_state.
    if !textures.is_empty() {
        renderer
            .set_texture_atlas(&tex_descs, &tex_data, textures.len() as u32)
            .map_err(|e| format!("set_texture_atlas: {e:?}"))?;
    }
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);
    let frame = renderer.render().map_err(|e| format!("render: {e:?}"))?;
    let n = (frame.width * frame.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..4 {
            out.push((frame.beauty[i * 4 + ch].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    Ok(out)
}

/// One cooked signed-distance volume to hand to the native SDF-volume primitive.
///
/// `distances` are DECODED metres in x-fastest grid order
/// (`idx = x + nx*(y + ny*z)`), exactly the convention the engine's
/// `vox_physics::sdf::SdfVolume` and Spectra's `sdf_probe.slang` agree on.
/// `resolution` is `[nx, ny, nz]`, `origin` the grid corner in the asset's local
/// frame, `voxel_size` the metres per cell. `narrow_band` is informational here
/// (the distances are already decoded).
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone)]
pub struct SdfVolumeInput {
    pub resolution: [u32; 3],
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub narrow_band: f32,
    /// Decoded signed distances in metres, x-fastest, length = nx*ny*nz.
    pub distances: Vec<f32>,
}

/// One-shot still: **path-trace ONE building's SDF as a SOLID surface** through
/// the engine-owned Spectra path tracer's NATIVE SDF-volume primitive (M0).
///
/// This is the confetti→wall proof: instead of tessellating splats to flat
/// camera-facing quads (the lossy `splat_convert`/`splat_backend` bridge), the
/// path tracer SPHERE-TRACES the instanced SDF atlas in `megakernel.slang` and
/// returns a continuous filled façade. The volume is uploaded as one SDF
/// instance with an identity rotation, translated by `position` and uniformly
/// scaled by `scale`. Shaded with a flat Lambert `albedo` lit by the four-light
/// rig (full per-atom colour is a later milestone). Returns RGBA8 (`w*h*4`).
///
/// Seconds/frame is acceptable — this is a correctness spike, not realtime.
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_to_rgba(
    volume: &SdfVolumeInput,
    position: [f32; 3],
    scale: f32,
    albedo: [f32; 3],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    rig: &LightRig,
) -> Result<Vec<u8>, String> {
    use crate::splat_convert::camera_layer;
    use spectra_scene_state::{
        LightLayer, SceneState, SdfInstanceHeader, SdfLayer, SdfVolumeHeader,
    };

    let expected = (volume.resolution[0] * volume.resolution[1] * volume.resolution[2]) as usize;
    if volume.distances.len() != expected {
        return Err(format!(
            "SDF distances len {} != nx*ny*nz {}",
            volume.distances.len(),
            expected
        ));
    }
    if scale <= 0.0 {
        return Err(format!("SDF instance scale must be > 0, got {scale}"));
    }

    // World-space AABB of the single instance: the grid's local bounds, scaled +
    // translated (identity rotation), then padded by the narrow band so the
    // sphere-trace starts cleanly outside the band.
    let local_min = volume.origin;
    let local_max = [
        volume.origin[0] + (volume.resolution[0] - 1) as f32 * volume.voxel_size,
        volume.origin[1] + (volume.resolution[1] - 1) as f32 * volume.voxel_size,
        volume.origin[2] + (volume.resolution[2] - 1) as f32 * volume.voxel_size,
    ];
    let pad = volume.narrow_band.max(volume.voxel_size) * scale;
    let world_aabb_min = [
        position[0] + local_min[0] * scale - pad,
        position[1] + local_min[1] * scale - pad,
        position[2] + local_min[2] * scale - pad,
    ];
    let world_aabb_max = [
        position[0] + local_max[0] * scale + pad,
        position[1] + local_max[1] * scale + pad,
        position[2] + local_max[2] * scale + pad,
    ];

    let sdf = SdfLayer::from_parts(
        vec![SdfVolumeHeader {
            asset_id: 1,
            resolution: volume.resolution,
            origin: volume.origin,
            voxel_size: volume.voxel_size,
            narrow_band: volume.narrow_band,
            distance_offset: 0,
            distance_count: volume.distances.len() as u32,
        }],
        vec![SdfInstanceHeader {
            instance_id: 1,
            asset_id: 1,
            volume_index: 0,
            position,
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0], // identity
            uniform_scale: scale,
            world_aabb_min,
            world_aabb_max,
        }],
        volume.distances.clone(),
    );

    let mut scene = SceneState::new(width, height);
    scene.sdf = sdf;
    scene.mark_sdf_changed();

    // Sentinel triangle. The renderer's software BVH/shadow traversal reads a
    // zeroed dummy `g_bvh_nodes` when a scene has NO triangles, which makes
    // `trace_shadow` loop forever (node 0 pushes children 0,0 repeatedly) and
    // hangs the GPU (device lost). One tiny degenerate triangle far below the
    // building gives the traversal a valid single-leaf tree so it terminates;
    // it is invisible (out of frame, ~1mm, faces away) and is NOT the rendered
    // surface — the SDF sphere-trace fills 100% of the visible façade.
    {
        use spectra_scene_state::MaterialLayer;
        let far = [position[0], world_aabb_min[1] - 1000.0, position[2]];
        scene.geometry.vertex_count = 3;
        scene.geometry.triangle_count = 1;
        scene.geometry.positions = vec![
            far[0],
            far[1],
            far[2],
            far[0] + 0.001,
            far[1],
            far[2],
            far[0],
            far[1],
            far[2] + 0.001,
        ];
        scene.geometry.normals = vec![0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0];
        scene.geometry.uvs = vec![0.0; 6];
        scene.geometry.indices = vec![0, 1, 2];
        scene.geometry.material_ids = vec![0];
        scene.materials = MaterialLayer {
            params: vec![0.0; VULKAN_MATERIAL_FLOATS],
            spectral_spd: Default::default(),
            material_count: 1,
        };
        scene.mark_geometry_changed();
        scene.mark_materials_changed();
    }

    // Four-light rig (sun + sky + camera fill + rim fill), identical structure to
    // the mesh path so SDF stills match building-review lighting.
    let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
    let camera_fill = (glam::Vec3::from(eye) - glam::Vec3::from(target) + glam::Vec3::Y * 0.35)
        .normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::new();
    for (dir, color, intensity) in [
        (sun.to_array(), rig.sun_color, rig.sun_intensity),
        (glam::Vec3::Y.to_array(), [0.58, 0.62, 0.72], rig.sky_intensity),
        (camera_fill.to_array(), [0.72, 0.74, 0.78], rig.camera_fill),
        (rim_fill.to_array(), [0.45, 0.47, 0.52], rig.rim_fill),
    ] {
        light_data.extend_from_slice(&pack_vulkan_directional_light(dir, color, intensity));
    }
    scene.lights = LightLayer {
        light_data,
        light_count: 4,
    };
    scene.mark_lights_changed();

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::from(eye),
        glam::Vec3::from(target),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    config.target_spp = spp;
    let mut renderer = Renderer::new(gpu, config);
    renderer.set_sdf_albedo(albedo);
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_sky_gradient(
        rig.sky_dome_zenith,
        rig.sky_dome_horizon,
        rig.sky_dome_intensity,
    );
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);

    let frame = renderer.render().map_err(|e| format!("render: {e:?}"))?;
    let n = (frame.width * frame.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..4 {
            out.push((frame.beauty[i * 4 + ch].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    Ok(out)
}

/// One world-space placement of an SDF volume in a multi-instance scene (M1).
///
/// `volume_index` references the `volumes` slice passed to
/// [`pathtrace_sdf_scene_to_rgba`] (several instances may share one cooked
/// field). `position` is the world translation of the volume's local origin,
/// `uniform_scale` scales the whole field about that origin (rotation is
/// `rotation_xyzw` as a unit quaternion, identity = `[0,0,0,1]`), and `albedo`
/// is the instance's flat Lambert colour so a block of buildings is visually
/// distinct (full per-atom colour is a later milestone).
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct SdfSceneInstance {
    pub volume_index: u32,
    pub position: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub uniform_scale: f32,
    pub albedo: [f32; 3],
}

/// One cooked atom's material contribution for the M2 per-surface gather. The
/// SDF carries massing only; these atoms carry the per-surface channel + colour
/// AND the window/glass identity. `position` is in the instance's LOCAL space
/// (the same space as the cooked SDF grid) — [`pathtrace_sdf_scene_with_atoms_to_rgba`]
/// transforms it to world with the instance's transform. `channel` is the
/// canonical channel id (see [`SDF_ATOM_CH_GLASS`] etc.) the megakernel
/// dispatches on; `color` is linear RGB.
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct SdfSceneAtom {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub channel: u32,
}

/// Canonical atom-channel ids. MUST match the `ATOM_CH_*` constants in
/// `slang/megakernel.slang`. Glass routes to the path-traced glass BSDF;
/// everything else is opaque Lambert shaded with the atom colour.
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_FACADE: u32 = 0;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_ROOF: u32 = 1;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_TRIM: u32 = 2;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_DOOR: u32 = 3;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_STRUCTURE: u32 = 4;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_GLASS: u32 = 5;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_DETAIL: u32 = 6;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_GROUND: u32 = 7;
#[cfg(feature = "spectra-native")]
pub const SDF_ATOM_CH_OTHER: u32 = 8;

/// Map a cooked atom `channel` string (from atoms.json) to a canonical
/// [`SDF_ATOM_CH_*`] id. Unknown channels fall back to [`SDF_ATOM_CH_OTHER`].
#[cfg(feature = "spectra-native")]
pub fn sdf_atom_channel_from_str(s: &str) -> u32 {
    match s.to_ascii_lowercase().as_str() {
        "facade" | "wall" => SDF_ATOM_CH_FACADE,
        "roof" => SDF_ATOM_CH_ROOF,
        "trim" => SDF_ATOM_CH_TRIM,
        "door" => SDF_ATOM_CH_DOOR,
        "structure" => SDF_ATOM_CH_STRUCTURE,
        "glass" | "window" => SDF_ATOM_CH_GLASS,
        "detail" => SDF_ATOM_CH_DETAIL,
        "ground" => SDF_ATOM_CH_GROUND,
        _ => SDF_ATOM_CH_OTHER,
    }
}

/// Generic optical aperture on an SDF instance (engine-side mirror of a cooked
/// glass rect; instance-LOCAL space — the same space as the atoms + SDF grid).
/// Wave-1 Task 6 wires these into rect-anchored glass classification, muntins
/// and interior mapping; until then [`pathtrace_sdf_scene_textured_to_rgba`]
/// accepts only EMPTY per-instance lists (non-empty errors explicitly rather
/// than silently dropping authored windows).
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct SdfApertureRect {
    pub center: [f32; 3],
    /// Unit in-plane axis (rect "width" direction).
    pub right: [f32; 3],
    /// Unit in-plane axis (rect "height" direction).
    pub up: [f32; 3],
    /// Half extents in metres along `right` / `up`.
    pub half_extents: [f32; 2],
    /// Forge WindowStyle id (DoubleHung=0, Casement=1, Bay=2, Lancet=3, Sash=4).
    pub style_id: u32,
}

/// Per-instance material dispatch for the textured SDF path: channel id
/// (0..=8, the [`SDF_ATOM_CH_FACADE`]..[`SDF_ATOM_CH_OTHER`] order) -> index
/// into the `materials` slice passed to
/// [`pathtrace_sdf_scene_textured_to_rgba`]. `-1` = no PBR material for that
/// channel: the kernel keeps the flat k-NN atom colour (M2 behaviour), which
/// is also how glass stays on its BSDF route.
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct SdfChannelMaterials {
    pub material_for_channel: [i32; 9],
}

/// Per-volume UV params replicating the cook's `forge_box_uv` convention
/// (game_asset_cook.rs: `p = [local.x + w/2, local.y, d/2 - local.z]`,
/// dominant-|normal|-axis plane pick, 1 texture tile per 2.5 m). The megakernel
/// reproduces that projection EXACTLY at the SDF hit so the texture layer lands
/// where the cooked atom colours were sampled from (the M1 consistency gate).
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy)]
pub struct SdfUvParams {
    /// `forge_width * 0.5` (metres).
    pub offset_x: f32,
    /// `forge_depth * 0.5` (metres).
    pub offset_z: f32,
    /// `1.0 / 2.5` — texture tiles per metre.
    pub tile_recip: f32,
}

/// Pack the per-instance channel->material table for `g_sdf_channel_material`:
/// 9 floats per instance, ids verbatim as f32 (-1.0 = keep the atom colour).
#[cfg(feature = "spectra-native")]
fn pack_sdf_channel_material_table(channel_materials: &[SdfChannelMaterials]) -> Vec<f32> {
    let mut out = Vec::with_capacity(channel_materials.len() * 9);
    for cm in channel_materials {
        out.extend(cm.material_for_channel.iter().map(|&id| id as f32));
    }
    out
}

/// Pack the per-volume box-UV params for `g_sdf_volume_uv_params`: 4 floats
/// per volume — `[offset_x, offset_z, tile_recip, 0.0 pad]`.
#[cfg(feature = "spectra-native")]
fn pack_sdf_uv_params(uv_params: &[SdfUvParams]) -> Vec<f32> {
    let mut out = Vec::with_capacity(uv_params.len() * 4);
    for p in uv_params {
        out.extend_from_slice(&[p.offset_x, p.offset_z, p.tile_recip, 0.0]);
    }
    out
}

/// One-shot still: **path-trace a CLUSTER of buildings as native SDF instances**
/// through the engine-owned Spectra path tracer's SDF-volume primitive (M1 —
/// the multi-instance proof). Extends [`pathtrace_sdf_to_rgba`]'s single
/// volume/instance to a MULTI-VOLUME atlas (`volumes`: distinct cooked fields,
/// concatenated with per-volume distance offsets exactly like the SDF pillar's
/// atlas) and MANY instances (`instances`: each a `(volume_index, world
/// transform, albedo)`). The megakernel sphere-traces the union of all instance
/// AABBs and, at every step, takes the min signed distance over all instances —
/// so the front-most building's surface is the hit and buildings occlude each
/// other correctly. Returns RGBA8 (`w*h*4`).
///
/// SDF-only: no Gaussian intersection, no splat bridge — sphere-traced SDFs are
/// the single geometry primitive. The one degenerate sentinel triangle (far
/// below the scene, invisible) only exists to give the software BVH/shadow
/// traversal a valid single-leaf tree (a zero-triangle scene hangs the GPU);
/// it is never the rendered surface.
///
/// Seconds/frame is acceptable — this is a correctness spike, not realtime.
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_to_rgba(
    volumes: &[SdfVolumeInput],
    instances: &[SdfSceneInstance],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    rig: &LightRig,
) -> Result<Vec<u8>, String> {
    let (rgba, _report) = pathtrace_sdf_scene_perf(
        volumes,
        instances,
        eye,
        target,
        fov_y,
        width,
        height,
        spp,
        rig,
        &SdfScenePerfKnobs::default(),
    )?;
    Ok(rgba)
}

/// Perf-harness knobs for [`pathtrace_sdf_scene_perf`]. `Default` reproduces
/// [`pathtrace_sdf_scene_to_rgba`] exactly (no instrumentation, no overrides).
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Default)]
pub struct SdfScenePerfKnobs {
    /// `u_sdf_debug_mode`: 0 = normal, 1 = march-stats→film (steps/empty/evals
    /// in RGB, hit flag in A; skips normal+shading), 2 = skip SDF march.
    pub debug_mode: i32,
    /// Disable the post-render denoiser (isolates GPU kernel time).
    pub disable_denoiser: bool,
    /// Override `RenderConfig::near_realtime`'s `max_bounces` (default 3).
    pub max_bounces: Option<u32>,
    /// Collect per-dispatch kernel timings via `Renderer::timing_sink`.
    pub collect_kernel_timing: bool,
    /// Number of `render()` calls on ONE persistent renderer (default/0 = 1).
    /// Frame 0 pays runtime kernel compilation + lazy GPU allocation; frames
    /// 1.. are the steady-state cost. `per_kernel_ms` and the march counters
    /// are reported for the LAST frame only; `frame_times_ms` has every frame.
    pub frames: u32,
    /// Force the exhaustive M1 union-AABB march (`u_sdf_march_legacy = 1`)
    /// instead of the interval-culled march — the A/B baseline.
    pub legacy_march: bool,
    /// Strip the pipeline to what an SDF-primary frame actually uses:
    /// no ReSTIR DI/PT chains, no NRC query, max_bounces = 1.
    pub lean_pipeline: bool,
}

/// Per-render measurement report from [`pathtrace_sdf_scene_perf`].
#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Default)]
pub struct SdfScenePerfReport {
    /// Wall-clock `Renderer::render()` time of the LAST frame (includes
    /// readback + denoiser; frame 0 additionally pays kernel compilation).
    pub render_time_ms: f64,
    /// Wall-clock ms of EVERY `render()` call (frame 0 = cold/compile).
    pub frame_times_ms: Vec<f64>,
    pub samples_done: u32,
    /// Raw `(kernel_label, ms)` per dispatch of the LAST frame, in dispatch
    /// order (empty unless `collect_kernel_timing`).
    pub per_kernel_ms: Vec<(String, f32)>,
    /// Tonemapped beauty of the last frame (RGBA f32, clamped [0,1]).
    pub beauty: Vec<f32>,
    /// RAW film accumulators (film_r/g/b, divided by samples_done) — the
    /// unclamped per-ray march counters when `debug_mode == 1`. RGB triplets.
    pub raw_film: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Instrumented variant of [`pathtrace_sdf_scene_to_rgba`]: same scene build,
/// same renderer, plus perf knobs (march-stats debug mode, per-kernel timing,
/// denoiser/bounce overrides) and a [`SdfScenePerfReport`].
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_perf(
    volumes: &[SdfVolumeInput],
    instances: &[SdfSceneInstance],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    rig: &LightRig,
    knobs: &SdfScenePerfKnobs,
) -> Result<(Vec<u8>, SdfScenePerfReport), String> {
    use crate::splat_convert::camera_layer;
    use spectra_scene_state::{
        LightLayer, SceneState, SdfInstanceHeader, SdfLayer, SdfVolumeHeader,
    };

    if volumes.is_empty() {
        return Err("pathtrace_sdf_scene_to_rgba: no volumes".into());
    }
    if instances.is_empty() {
        return Err("pathtrace_sdf_scene_to_rgba: no instances".into());
    }

    // --- Multi-volume atlas: validate each field and lay out the concatenated
    // distance buffer with per-volume offsets (the SdfAtlasGpu / uploader shape).
    let mut all_distances: Vec<f32> = Vec::new();
    let mut volume_headers: Vec<SdfVolumeHeader> = Vec::with_capacity(volumes.len());
    let mut local_bounds: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(volumes.len());
    for (vi, v) in volumes.iter().enumerate() {
        let expected =
            (v.resolution[0] * v.resolution[1] * v.resolution[2]) as usize;
        if v.distances.len() != expected {
            return Err(format!(
                "volume {vi}: distances len {} != nx*ny*nz {}",
                v.distances.len(),
                expected
            ));
        }
        let distance_offset = all_distances.len() as u32;
        all_distances.extend_from_slice(&v.distances);
        volume_headers.push(SdfVolumeHeader {
            asset_id: (vi as u64) + 1,
            resolution: v.resolution,
            origin: v.origin,
            voxel_size: v.voxel_size,
            narrow_band: v.narrow_band,
            distance_offset,
            distance_count: v.distances.len() as u32,
        });
        let lmin = v.origin;
        let lmax = [
            v.origin[0] + (v.resolution[0] - 1) as f32 * v.voxel_size,
            v.origin[1] + (v.resolution[1] - 1) as f32 * v.voxel_size,
            v.origin[2] + (v.resolution[2] - 1) as f32 * v.voxel_size,
        ];
        local_bounds.push((lmin, lmax));
    }

    // --- Instances: per-instance world AABB (rotated local box corners), header,
    // and flat albedo. World AABB must enclose the transformed grid so the
    // megakernel's union-AABB sphere-trace covers every instance.
    let mut instance_headers: Vec<SdfInstanceHeader> = Vec::with_capacity(instances.len());
    let mut instance_albedo: Vec<f32> = Vec::with_capacity(instances.len() * 3);
    let mut scene_min = [f32::INFINITY; 3];
    let mut scene_max = [f32::NEG_INFINITY; 3];
    for (ii, inst) in instances.iter().enumerate() {
        let vol = inst.volume_index as usize;
        if vol >= volumes.len() {
            return Err(format!(
                "instance {ii}: volume_index {vol} out of range (have {} volumes)",
                volumes.len()
            ));
        }
        if inst.uniform_scale <= 0.0 {
            return Err(format!(
                "instance {ii}: uniform_scale must be > 0, got {}",
                inst.uniform_scale
            ));
        }
        let (lmin, lmax) = local_bounds[vol];
        let q = glam::Quat::from_array(inst.rotation_xyzw).normalize();
        let pos = glam::Vec3::from(inst.position);
        let s = inst.uniform_scale;
        // Transform all 8 corners (rot * scale * corner + pos) → world AABB.
        let mut wmin = [f32::INFINITY; 3];
        let mut wmax = [f32::NEG_INFINITY; 3];
        for cx in [lmin[0], lmax[0]] {
            for cy in [lmin[1], lmax[1]] {
                for cz in [lmin[2], lmax[2]] {
                    let world = pos + q * (glam::Vec3::new(cx, cy, cz) * s);
                    for k in 0..3 {
                        wmin[k] = wmin[k].min(world[k]);
                        wmax[k] = wmax[k].max(world[k]);
                    }
                }
            }
        }
        // Pad by the (scaled) narrow band so the sphere-trace starts cleanly
        // outside the field's band on entry.
        let pad = volumes[vol].narrow_band.max(volumes[vol].voxel_size) * s;
        for k in 0..3 {
            wmin[k] -= pad;
            wmax[k] += pad;
            scene_min[k] = scene_min[k].min(wmin[k]);
            scene_max[k] = scene_max[k].max(wmax[k]);
        }
        instance_headers.push(SdfInstanceHeader {
            instance_id: (ii as u64) + 1,
            asset_id: (vol as u64) + 1,
            volume_index: inst.volume_index,
            position: inst.position,
            rotation_xyzw: q.to_array(),
            uniform_scale: s,
            world_aabb_min: wmin,
            world_aabb_max: wmax,
        });
        instance_albedo.extend_from_slice(&inst.albedo);
    }

    let sdf = SdfLayer::from_parts_with_albedo(
        volume_headers,
        instance_headers,
        all_distances,
        instance_albedo,
    );

    let mut scene = SceneState::new(width, height);
    scene.sdf = sdf;
    scene.mark_sdf_changed();

    // Sentinel triangle (see pathtrace_sdf_to_rgba): one tiny degenerate tri far
    // below the whole cluster gives the software BVH/shadow traversal a valid
    // single-leaf tree so it terminates instead of hanging the GPU. Invisible —
    // the SDF sphere-trace fills the visible façades.
    {
        use spectra_scene_state::MaterialLayer;
        let cx = 0.5 * (scene_min[0] + scene_max[0]);
        let cz = 0.5 * (scene_min[2] + scene_max[2]);
        let far = [cx, scene_min[1] - 1000.0, cz];
        scene.geometry.vertex_count = 3;
        scene.geometry.triangle_count = 1;
        scene.geometry.positions = vec![
            far[0],
            far[1],
            far[2],
            far[0] + 0.001,
            far[1],
            far[2],
            far[0],
            far[1],
            far[2] + 0.001,
        ];
        scene.geometry.normals = vec![0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0];
        scene.geometry.uvs = vec![0.0; 6];
        scene.geometry.indices = vec![0, 1, 2];
        scene.geometry.material_ids = vec![0];
        scene.materials = MaterialLayer {
            params: vec![0.0; VULKAN_MATERIAL_FLOATS],
            spectral_spd: Default::default(),
            material_count: 1,
        };
        scene.mark_geometry_changed();
        scene.mark_materials_changed();
    }

    // Four-light rig, identical structure to pathtrace_sdf_to_rgba / the mesh
    // path so cluster stills match building-review lighting.
    let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
    let camera_fill = (glam::Vec3::from(eye) - glam::Vec3::from(target) + glam::Vec3::Y * 0.35)
        .normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::new();
    for (dir, color, intensity) in [
        (sun.to_array(), rig.sun_color, rig.sun_intensity),
        (glam::Vec3::Y.to_array(), [0.58, 0.62, 0.72], rig.sky_intensity),
        (camera_fill.to_array(), [0.72, 0.74, 0.78], rig.camera_fill),
        (rim_fill.to_array(), [0.45, 0.47, 0.52], rig.rim_fill),
    ] {
        light_data.extend_from_slice(&pack_vulkan_directional_light(dir, color, intensity));
    }
    scene.lights = LightLayer {
        light_data,
        light_count: 4,
    };
    scene.mark_lights_changed();

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::from(eye),
        glam::Vec3::from(target),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    config.target_spp = spp;
    if knobs.disable_denoiser {
        config.denoiser_mode = spectra_renderer::DenoiserMode::None;
    }
    if let Some(mb) = knobs.max_bounces {
        config.max_bounces = mb;
    }
    if knobs.lean_pipeline {
        config.use_restir = false;
        config.use_nrc = false;
        config.resample_mode = spectra_renderer::ResampleMode::None;
        config.max_bounces = knobs.max_bounces.unwrap_or(1);
    }
    let mut renderer = Renderer::new(gpu, config);
    renderer.set_sdf_debug_mode(knobs.debug_mode);
    renderer.sdf_march_legacy = if knobs.legacy_march { 1 } else { 0 };
    if knobs.collect_kernel_timing {
        renderer.timing_sink = Some(Vec::new());
    }
    // Fallback single albedo (used only if the per-instance buffer is absent).
    renderer.set_sdf_albedo(instances[0].albedo);
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_sky_gradient(
        rig.sky_dome_zenith,
        rig.sky_dome_horizon,
        rig.sky_dome_intensity,
    );
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);

    // Render `frames` times on the SAME renderer: frame 0 pays runtime slangc
    // kernel compilation + lazy GPU allocation; frames 1.. are steady state.
    let n_frames = knobs.frames.max(1);
    let mut frame_times_ms: Vec<f64> = Vec::with_capacity(n_frames as usize);
    let mut frame = None;
    for fi in 0..n_frames {
        // Per-frame timing: only keep the last frame's per-kernel entries.
        if knobs.collect_kernel_timing {
            renderer.timing_sink = Some(Vec::new());
        }
        let t0 = std::time::Instant::now();
        let f = renderer.render().map_err(|e| format!("render (frame {fi}): {e:?}"))?;
        frame_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        frame = Some(f);
    }
    let frame = frame.unwrap();

    // RAW film readback (unclamped counters for debug_mode == 1): beauty goes
    // through the CPU tonemap which clamps to [0,1], so the march counters can
    // only be harvested from the film accumulators directly.
    let n = (frame.width * frame.height) as usize;
    let mut raw_film = vec![0.0f32; n * 3];
    if let Some(state) = renderer.state.as_ref() {
        if let (Some(rh), Some(gh), Some(bh)) = (state.film_r, state.film_g, state.film_b) {
            let mut r = vec![0.0f32; n];
            let mut g = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            if renderer.gpu.download_f32(&rh, &mut r).is_ok()
                && renderer.gpu.download_f32(&gh, &mut g).is_ok()
                && renderer.gpu.download_f32(&bh, &mut b).is_ok()
            {
                let inv = 1.0 / frame.samples_done.max(1) as f32;
                for i in 0..n {
                    raw_film[i * 3] = r[i] * inv;
                    raw_film[i * 3 + 1] = g[i] * inv;
                    raw_film[i * 3 + 2] = b[i] * inv;
                }
            }
        }
    }

    let report = SdfScenePerfReport {
        render_time_ms: *frame_times_ms.last().unwrap(),
        frame_times_ms,
        samples_done: frame.samples_done,
        per_kernel_ms: renderer.timing_sink.take().unwrap_or_default(),
        beauty: frame.beauty.clone(),
        raw_film,
        width: frame.width,
        height: frame.height,
    };
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..4 {
            out.push((frame.beauty[i * 4 + ch].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    Ok((out, report))
}

/// One-shot still: **path-trace a CLUSTER of SDF buildings WITH per-surface
/// material from the cooked atoms** (M2 — the appearance milestone). Extends
/// [`pathtrace_sdf_scene_to_rgba`] (which gives each building one flat albedo)
/// by feeding the cooked ATOMS alongside the geometry-only SDF: at a surface hit
/// the megakernel k-NN-gathers the winning instance's nearest atoms and reads
/// their channel + colour, so walls/roof/trim show real per-surface COLOUR and
/// — crucially — **windows render as real path-traced GLASS** (an SDF glass hit
/// transmits/refracts and continues the path instead of terminating opaque, so
/// the window is see-through/reflective and the tracer shows what is behind it).
///
/// `atoms_per_instance[i]` is the cooked atom list for `instances[i]` in that
/// instance's LOCAL space (same space as the cooked SDF grid); they are
/// transformed to world here with the instance's transform. Still SDF-only — no
/// Gaussian/splat bridge for surfaces. Returns RGBA8 (`w*h*4`).
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_with_atoms_to_rgba(
    volumes: &[SdfVolumeInput],
    instances: &[SdfSceneInstance],
    atoms_per_instance: &[Vec<SdfSceneAtom>],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    rig: &LightRig,
) -> Result<Vec<u8>, String> {
    use crate::splat_convert::camera_layer;
    use spectra_scene_state::{
        LightLayer, SceneState, SdfInstanceHeader, SdfLayer, SdfVolumeHeader,
    };

    if volumes.is_empty() {
        return Err("pathtrace_sdf_scene_with_atoms_to_rgba: no volumes".into());
    }
    if instances.is_empty() {
        return Err("pathtrace_sdf_scene_with_atoms_to_rgba: no instances".into());
    }
    if atoms_per_instance.len() != instances.len() {
        return Err(format!(
            "atoms_per_instance len {} != instances len {}",
            atoms_per_instance.len(),
            instances.len()
        ));
    }

    // --- Multi-volume atlas (identical to pathtrace_sdf_scene_to_rgba). -------
    let mut all_distances: Vec<f32> = Vec::new();
    let mut volume_headers: Vec<SdfVolumeHeader> = Vec::with_capacity(volumes.len());
    let mut local_bounds: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(volumes.len());
    for (vi, v) in volumes.iter().enumerate() {
        let expected = (v.resolution[0] * v.resolution[1] * v.resolution[2]) as usize;
        if v.distances.len() != expected {
            return Err(format!(
                "volume {vi}: distances len {} != nx*ny*nz {}",
                v.distances.len(),
                expected
            ));
        }
        let distance_offset = all_distances.len() as u32;
        all_distances.extend_from_slice(&v.distances);
        volume_headers.push(SdfVolumeHeader {
            asset_id: (vi as u64) + 1,
            resolution: v.resolution,
            origin: v.origin,
            voxel_size: v.voxel_size,
            narrow_band: v.narrow_band,
            distance_offset,
            distance_count: v.distances.len() as u32,
        });
        let lmin = v.origin;
        let lmax = [
            v.origin[0] + (v.resolution[0] - 1) as f32 * v.voxel_size,
            v.origin[1] + (v.resolution[1] - 1) as f32 * v.voxel_size,
            v.origin[2] + (v.resolution[2] - 1) as f32 * v.voxel_size,
        ];
        local_bounds.push((lmin, lmax));
    }

    // --- Instances + per-atom WORLD-space material buffers. ------------------
    let mut instance_headers: Vec<SdfInstanceHeader> = Vec::with_capacity(instances.len());
    let mut instance_albedo: Vec<f32> = Vec::with_capacity(instances.len() * 3);
    // Flat SoA atom buffers + per-instance [offset, count] ranges (M2).
    let mut atom_positions: Vec<f32> = Vec::new();
    let mut atom_colors: Vec<f32> = Vec::new();
    let mut atom_channels: Vec<f32> = Vec::new();
    let mut instance_atom_range: Vec<f32> = Vec::with_capacity(instances.len() * 2);
    let mut scene_min = [f32::INFINITY; 3];
    let mut scene_max = [f32::NEG_INFINITY; 3];
    for (ii, inst) in instances.iter().enumerate() {
        let vol = inst.volume_index as usize;
        if vol >= volumes.len() {
            return Err(format!(
                "instance {ii}: volume_index {vol} out of range (have {} volumes)",
                volumes.len()
            ));
        }
        if inst.uniform_scale <= 0.0 {
            return Err(format!(
                "instance {ii}: uniform_scale must be > 0, got {}",
                inst.uniform_scale
            ));
        }
        let (lmin, lmax) = local_bounds[vol];
        let q = glam::Quat::from_array(inst.rotation_xyzw).normalize();
        let pos = glam::Vec3::from(inst.position);
        let s = inst.uniform_scale;
        let mut wmin = [f32::INFINITY; 3];
        let mut wmax = [f32::NEG_INFINITY; 3];
        for cx in [lmin[0], lmax[0]] {
            for cy in [lmin[1], lmax[1]] {
                for cz in [lmin[2], lmax[2]] {
                    let world = pos + q * (glam::Vec3::new(cx, cy, cz) * s);
                    for k in 0..3 {
                        wmin[k] = wmin[k].min(world[k]);
                        wmax[k] = wmax[k].max(world[k]);
                    }
                }
            }
        }
        let pad = volumes[vol].narrow_band.max(volumes[vol].voxel_size) * s;
        for k in 0..3 {
            wmin[k] -= pad;
            wmax[k] += pad;
            scene_min[k] = scene_min[k].min(wmin[k]);
            scene_max[k] = scene_max[k].max(wmax[k]);
        }
        instance_headers.push(SdfInstanceHeader {
            instance_id: (ii as u64) + 1,
            asset_id: (vol as u64) + 1,
            volume_index: inst.volume_index,
            position: inst.position,
            rotation_xyzw: q.to_array(),
            uniform_scale: s,
            world_aabb_min: wmin,
            world_aabb_max: wmax,
        });
        instance_albedo.extend_from_slice(&inst.albedo);

        // Transform this instance's atoms to world space (rot*scale*local + pos),
        // matching the SDF grid transform, and append to the flat SoA buffers.
        let atom_offset = atom_channels.len() as u32;
        for atom in &atoms_per_instance[ii] {
            let lp = glam::Vec3::from(atom.position);
            let wp = pos + q * (lp * s);
            atom_positions.extend_from_slice(&[wp.x, wp.y, wp.z]);
            atom_colors.extend_from_slice(&atom.color);
            atom_channels.push(atom.channel as f32);
        }
        let atom_count = atom_channels.len() as u32 - atom_offset;
        instance_atom_range.push(atom_offset as f32);
        instance_atom_range.push(atom_count as f32);
    }

    let sdf = SdfLayer::from_parts_with_atoms(
        volume_headers,
        instance_headers,
        all_distances,
        instance_albedo,
        atom_positions,
        atom_colors,
        atom_channels,
        instance_atom_range,
    );

    let mut scene = SceneState::new(width, height);
    scene.sdf = sdf;
    scene.mark_sdf_changed();

    // Sentinel triangle (BVH validity — invisible, see pathtrace_sdf_to_rgba).
    {
        use spectra_scene_state::MaterialLayer;
        let cx = 0.5 * (scene_min[0] + scene_max[0]);
        let cz = 0.5 * (scene_min[2] + scene_max[2]);
        let far = [cx, scene_min[1] - 1000.0, cz];
        scene.geometry.vertex_count = 3;
        scene.geometry.triangle_count = 1;
        scene.geometry.positions = vec![
            far[0],
            far[1],
            far[2],
            far[0] + 0.001,
            far[1],
            far[2],
            far[0],
            far[1],
            far[2] + 0.001,
        ];
        scene.geometry.normals = vec![0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0];
        scene.geometry.uvs = vec![0.0; 6];
        scene.geometry.indices = vec![0, 1, 2];
        scene.geometry.material_ids = vec![0];
        scene.materials = MaterialLayer {
            params: vec![0.0; VULKAN_MATERIAL_FLOATS],
            spectral_spd: Default::default(),
            material_count: 1,
        };
        scene.mark_geometry_changed();
        scene.mark_materials_changed();
    }

    // Four-light rig (identical to the M1 cluster path).
    let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
    let camera_fill = (glam::Vec3::from(eye) - glam::Vec3::from(target) + glam::Vec3::Y * 0.35)
        .normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::new();
    for (dir, color, intensity) in [
        (sun.to_array(), rig.sun_color, rig.sun_intensity),
        (glam::Vec3::Y.to_array(), [0.58, 0.62, 0.72], rig.sky_intensity),
        (camera_fill.to_array(), [0.72, 0.74, 0.78], rig.camera_fill),
        (rim_fill.to_array(), [0.45, 0.47, 0.52], rig.rim_fill),
    ] {
        light_data.extend_from_slice(&pack_vulkan_directional_light(dir, color, intensity));
    }
    scene.lights = LightLayer {
        light_data,
        light_count: 4,
    };
    scene.mark_lights_changed();

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::from(eye),
        glam::Vec3::from(target),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    config.target_spp = spp;
    // Glass needs continuation bounces; disable NRC so the refracted ray never
    // gets short-circuited into the cache at a deep bounce. Keep >= 3 bounces
    // (camera→glass→behind→...).
    config.use_nrc = false;
    if config.max_bounces < 3 {
        config.max_bounces = 3;
    }
    let mut renderer = Renderer::new(gpu, config);
    renderer.set_sdf_albedo(instances[0].albedo);
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_sky_gradient(
        rig.sky_dome_zenith,
        rig.sky_dome_horizon,
        rig.sky_dome_intensity,
    );
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);

    let frame = renderer.render().map_err(|e| format!("render: {e:?}"))?;
    let n = (frame.width * frame.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..4 {
            out.push((frame.beauty[i * 4 + ch].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    Ok(out)
}

/// Padded world-space AABB of one SDF instance: the rotated/scaled/translated
/// grid corners, padded by the (scaled) narrow band — the EXACT min/max packed
/// into the instance header (`SdfInstanceHeader::world_aabb_min/max`), which
/// the megakernel also reads back as the atom cell grid's anchor. Keeping this
/// in one place guarantees the grid build, the header, and the kernel's cell
/// math all use bit-identical values.
#[cfg(feature = "spectra-native")]
fn sdf_instance_world_aabb(
    volume: &SdfVolumeInput,
    inst: &SdfSceneInstance,
) -> ([f32; 3], [f32; 3]) {
    let lmin = volume.origin;
    let lmax = [
        volume.origin[0] + (volume.resolution[0] - 1) as f32 * volume.voxel_size,
        volume.origin[1] + (volume.resolution[1] - 1) as f32 * volume.voxel_size,
        volume.origin[2] + (volume.resolution[2] - 1) as f32 * volume.voxel_size,
    ];
    let q = glam::Quat::from_array(inst.rotation_xyzw).normalize();
    let pos = glam::Vec3::from(inst.position);
    let s = inst.uniform_scale;
    let mut wmin = [f32::INFINITY; 3];
    let mut wmax = [f32::NEG_INFINITY; 3];
    for cx in [lmin[0], lmax[0]] {
        for cy in [lmin[1], lmax[1]] {
            for cz in [lmin[2], lmax[2]] {
                let world = pos + q * (glam::Vec3::new(cx, cy, cz) * s);
                for k in 0..3 {
                    wmin[k] = wmin[k].min(world[k]);
                    wmax[k] = wmax[k].max(world[k]);
                }
            }
        }
    }
    let pad = volume.narrow_band.max(volume.voxel_size) * s;
    for k in 0..3 {
        wmin[k] -= pad;
        wmax[k] += pad;
    }
    (wmin, wmax)
}

/// One instance's uniform cell grid over its atoms' WORLD positions — the
/// wave-1 gather accelerator. The megakernel visits the 3×3×3 cell
/// neighbourhood of a surface hit instead of the instance's full atom range
/// (~80× fewer atom reads on the cooked craftsman). Built host-side by
/// [`build_sdf_atom_cell_grid`]; uploaded via `SdfLayer::from_parts_textured`
/// as `g_sdf_atom_grid_headers` (6 floats/instance) + `g_sdf_atom_cell_table`
/// (2 floats/cell), with the atom SoA re-ordered cell-sorted so each cell's
/// atoms are one contiguous `(start, count)` run.
#[cfg(feature = "spectra-native")]
struct SdfAtomCellGrid {
    /// Cells per axis (4..=32 each).
    dims: [u32; 3],
    /// World-space cell edge length (metres).
    cell_size: f32,
    /// Grid anchor = the instance's padded `world_aabb_min` (the kernel reads
    /// the same value from the instance header).
    grid_min: [f32; 3],
    /// `(start, count)` per cell, x-fastest cell order; `start` is LOCAL to
    /// the instance's atom range (kernel reads atom `atom_base + start + k`).
    cell_table: Vec<(u32, u32)>,
    /// Permutation into cell-sorted order: `order[k]` = original atom index
    /// placed at sorted slot `k` (stable within a cell).
    order: Vec<u32>,
}

#[cfg(feature = "spectra-native")]
impl SdfAtomCellGrid {
    /// Clamped cell coordinate of a world-space point (the kernel's cell math).
    fn cell_coord(&self, p: [f32; 3]) -> [i32; 3] {
        let mut c = [0i32; 3];
        for k in 0..3 {
            let f = ((p[k] - self.grid_min[k]) / self.cell_size).floor() as i32;
            c[k] = f.clamp(0, self.dims[k] as i32 - 1);
        }
        c
    }

    /// X-fastest linear cell index.
    fn cell_index(&self, c: [i32; 3]) -> usize {
        c[0] as usize
            + self.dims[0] as usize * (c[1] as usize + self.dims[1] as usize * c[2] as usize)
    }

    /// Number of atoms the kernel's gather visits at world-space point `p`:
    /// the sum of the 3×3×3 cell-neighbourhood counts (clamped to grid dims).
    fn neighbourhood_atom_count(&self, p: [f32; 3]) -> u32 {
        let c = self.cell_coord(p);
        let mut total = 0u32;
        for dz in -1i32..=1 {
            let cz = c[2] + dz;
            if cz < 0 || cz >= self.dims[2] as i32 {
                continue;
            }
            for dy in -1i32..=1 {
                let cy = c[1] + dy;
                if cy < 0 || cy >= self.dims[1] as i32 {
                    continue;
                }
                for dx in -1i32..=1 {
                    let cx = c[0] + dx;
                    if cx < 0 || cx >= self.dims[0] as i32 {
                        continue;
                    }
                    total += self.cell_table[self.cell_index([cx, cy, cz])].1;
                }
            }
        }
        total
    }
}

/// Build one instance's atom cell grid over WORLD-space positions.
///
/// Cell size = `max(0.5 m, 2 × median nearest-neighbour atom spacing,
/// min_cell)` — large enough that every atom within the glass detect radius
/// (`min_cell`, the kernel's `max(voxel_max * 0.6, 0.18)`) AND the K-NN blend
/// span of a surface hit lies inside the hit's 3×3×3 cell neighbourhood, so
/// the gridded gather reproduces the linear scan's K-nearest set exactly (the
/// kernel additionally falls back to the full scan for the rare sparse hits
/// whose 6-NN span exceeds one cell).
/// Dims derive from the instance's padded world AABB (`wmin`/`wmax`, the same
/// values in the instance header the kernel anchors on), clamped to 4³..32³;
/// the cell grows if 32 cells can't span an axis. The median spacing comes
/// from a deterministic even-stride sample (≤ 256 atoms, O(sample·n)).
/// Counting-sort is stable: atoms keep their original relative order within a
/// cell, so the build is fully deterministic.
#[cfg(feature = "spectra-native")]
fn build_sdf_atom_cell_grid(
    world_positions: &[[f32; 3]],
    wmin: [f32; 3],
    wmax: [f32; 3],
    min_cell: f32,
) -> SdfAtomCellGrid {
    let n = world_positions.len();
    let extent = [
        (wmax[0] - wmin[0]).max(1.0e-4),
        (wmax[1] - wmin[1]).max(1.0e-4),
        (wmax[2] - wmin[2]).max(1.0e-4),
    ];

    // Median nearest-neighbour spacing from a deterministic sample.
    let mut cell = 0.5f32.max(min_cell);
    if n >= 2 {
        let sample = n.min(256);
        let stride = (n / sample).max(1);
        let mut nn: Vec<f32> = Vec::with_capacity(sample);
        let mut i = 0usize;
        while i < n && nn.len() < sample {
            let p = world_positions[i];
            let mut best = f32::INFINITY;
            for (j, q) in world_positions.iter().enumerate() {
                if j == i {
                    continue;
                }
                let d2 = (q[0] - p[0]) * (q[0] - p[0])
                    + (q[1] - p[1]) * (q[1] - p[1])
                    + (q[2] - p[2]) * (q[2] - p[2]);
                if d2 < best {
                    best = d2;
                }
            }
            nn.push(best.sqrt());
            i += stride;
        }
        nn.sort_by(|a, b| a.total_cmp(b));
        cell = (2.0 * nn[nn.len() / 2]).max(0.5).max(min_cell);
    }

    // Cap the grid at 32 cells per axis: grow the cell until every axis fits.
    for e in extent {
        if e / cell > 32.0 {
            cell = e / 32.0;
        }
    }

    let mut dims = [0u32; 3];
    for k in 0..3 {
        dims[k] = ((extent[k] / cell).ceil() as u32).clamp(4, 32);
    }

    let n_cells = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    let mut grid = SdfAtomCellGrid {
        dims,
        cell_size: cell,
        grid_min: wmin,
        cell_table: vec![(0u32, 0u32); n_cells],
        order: vec![0u32; n],
    };

    // Counting sort: count per cell, prefix-sum starts, stable scatter.
    let mut atom_cell = vec![0usize; n];
    for (i, p) in world_positions.iter().enumerate() {
        let ci = grid.cell_index(grid.cell_coord(*p));
        atom_cell[i] = ci;
        grid.cell_table[ci].1 += 1;
    }
    let mut acc = 0u32;
    for entry in grid.cell_table.iter_mut() {
        entry.0 = acc;
        acc += entry.1;
    }
    let mut next: Vec<u32> = grid.cell_table.iter().map(|e| e.0).collect();
    for (i, &ci) in atom_cell.iter().enumerate() {
        grid.order[next[ci] as usize] = i as u32;
        next[ci] += 1;
    }
    grid
}

/// One-shot still: **path-trace SDF buildings with the CELL-GRID atom gather
/// and per-channel PBR materials** — the wave-1 textured-SDF entry point
/// (Tasks 1+2; later wave tasks grow it with normal maps and aperture rects).
/// A thin extension of [`pathtrace_sdf_scene_with_atoms_to_rgba`]: the SAME
/// scene build, plus
///   - a per-instance uniform cell grid over the atoms' WORLD positions so the
///     megakernel's k-NN gather visits only the 3×3×3 cell neighbourhood of a
///     hit (`u_sdf_textured = 1`) — identical K-nearest + glass logic over
///     ~80× fewer atoms, byte-identical frames when no materials are supplied
///     (the parity gate `sdf_gather_grid_matches_linear` proves it);
///   - per-channel PBR dispatch: `channel_materials[i]` maps each instance's
///     gather channel (0..=8) to an index into `materials` (-1 = keep the M2
///     flat atom colour). `materials`/`textures` ride the SAME packing as
///     [`pathtrace_mesh_textured_to_rgba`] (`set_texture_atlas` AFTER
///     `load_scene_state` — order matters);
///   - `uv_params[v]` carries the cook's `forge_box_uv` constants per volume
///     so the kernel box-projects texture UVs exactly where the cooked atom
///     colours were sampled (`g_sdf_volume_uv_params`).
///
/// `apertures_per_instance` must currently be all-empty (rect-anchored glass
/// is wave-1 Task 6); `lighting_tier` must be 0 (M2 headlight rig — NEE/GI
/// tiers are a later task). `aov_roughness` is a debug flag for the quality
/// gates: when set, the returned RGBA's ALPHA channel carries the primary-hit
/// roughness AOV (`g_aov_roughness`) instead of coverage. Returns RGBA8
/// (`w*h*4`).
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_textured_to_rgba(
    volumes: &[SdfVolumeInput],
    uv_params: &[SdfUvParams],
    instances: &[SdfSceneInstance],
    atoms_per_instance: &[Vec<SdfSceneAtom>],
    apertures_per_instance: &[Vec<SdfApertureRect>],
    channel_materials: &[SdfChannelMaterials],
    materials: &[PbrMaterial],
    textures: &[TextureImage],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    width: u32,
    height: u32,
    spp: u32,
    lighting_tier: u32,
    rig: &LightRig,
    aov_roughness: bool,
) -> Result<Vec<u8>, String> {
    use crate::splat_convert::camera_layer;
    use spectra_scene_state::{
        LightLayer, SceneState, SdfInstanceHeader, SdfLayer, SdfVolumeHeader,
    };

    if volumes.is_empty() {
        return Err("pathtrace_sdf_scene_textured_to_rgba: no volumes".into());
    }
    if instances.is_empty() {
        return Err("pathtrace_sdf_scene_textured_to_rgba: no instances".into());
    }
    if atoms_per_instance.len() != instances.len() {
        return Err(format!(
            "atoms_per_instance len {} != instances len {}",
            atoms_per_instance.len(),
            instances.len()
        ));
    }
    if uv_params.len() != volumes.len() {
        return Err(format!(
            "uv_params len {} != volumes len {} (one forge_box_uv param set per volume)",
            uv_params.len(),
            volumes.len()
        ));
    }
    if channel_materials.len() != instances.len() {
        return Err(format!(
            "channel_materials len {} != instances len {}",
            channel_materials.len(),
            instances.len()
        ));
    }
    if apertures_per_instance.len() != instances.len() {
        return Err(format!(
            "apertures_per_instance len {} != instances len {}",
            apertures_per_instance.len(),
            instances.len()
        ));
    }
    if apertures_per_instance.iter().any(|a| !a.is_empty()) {
        return Err(
            "aperture rects are not wired yet (wave-1 Task 6) — pass empty per-instance \
             lists; refusing to silently drop authored windows"
                .into(),
        );
    }
    if lighting_tier != 0 {
        return Err(format!(
            "lighting_tier {lighting_tier} not wired yet — only tier 0 (the M2 headlight \
             rig) is implemented in wave 1 Tasks 1-3"
        ));
    }
    // Every channel-material id must index into `materials`, and every material
    // texture reference must index into `textures` — the kernel trusts these.
    for (ii, cm) in channel_materials.iter().enumerate() {
        for (ch, &id) in cm.material_for_channel.iter().enumerate() {
            if id >= 0 && id as usize >= materials.len() {
                return Err(format!(
                    "instance {ii} channel {ch}: material id {id} out of range \
                     (have {} materials)",
                    materials.len()
                ));
            }
        }
    }
    for (mi, m) in materials.iter().enumerate() {
        for (label, tex) in [
            ("albedo_tex", m.albedo_tex),
            ("roughness_tex", m.roughness_tex),
            ("normal_tex", m.normal_tex),
        ] {
            if tex >= 0 && tex as usize >= textures.len() {
                return Err(format!(
                    "material {mi}: {label} {tex} out of range (have {} textures)",
                    textures.len()
                ));
            }
        }
    }

    // Texture atlas packing through the SAME helper the mesh path uses
    // (build first so a malformed TextureImage errors before any GPU work).
    let (tex_descs, tex_data) = build_texture_atlas(textures)?;

    // --- Multi-volume atlas (identical to the M2 entry point). ---------------
    let mut all_distances: Vec<f32> = Vec::new();
    let mut volume_headers: Vec<SdfVolumeHeader> = Vec::with_capacity(volumes.len());
    for (vi, v) in volumes.iter().enumerate() {
        let expected = (v.resolution[0] * v.resolution[1] * v.resolution[2]) as usize;
        if v.distances.len() != expected {
            return Err(format!(
                "volume {vi}: distances len {} != nx*ny*nz {}",
                v.distances.len(),
                expected
            ));
        }
        let distance_offset = all_distances.len() as u32;
        all_distances.extend_from_slice(&v.distances);
        volume_headers.push(SdfVolumeHeader {
            asset_id: (vi as u64) + 1,
            resolution: v.resolution,
            origin: v.origin,
            voxel_size: v.voxel_size,
            narrow_band: v.narrow_band,
            distance_offset,
            distance_count: v.distances.len() as u32,
        });
    }

    // The kernel's glass window-detect radius is scene-wide:
    // max(coarsest world voxel * 0.6, 0.18 m) — see sdf_voxel_extents +
    // the detect_r site in megakernel.slang. Floor every instance's cell size
    // at it so a glass atom inside the detect radius is ALWAYS within the
    // 3x3x3 cell neighbourhood of the hit (exact glass classification).
    let mut voxel_max_world = 0.0f32;
    for inst in instances {
        // Out-of-range volume indices error properly in the main loop below.
        if let Some(v) = volumes.get(inst.volume_index as usize) {
            voxel_max_world = voxel_max_world.max(v.voxel_size * inst.uniform_scale);
        }
    }
    let glass_detect_r = (voxel_max_world * 0.6).max(0.18);

    // --- Instances + CELL-SORTED world-space atom SoA + per-instance grid. ---
    let mut instance_headers: Vec<SdfInstanceHeader> = Vec::with_capacity(instances.len());
    let mut instance_albedo: Vec<f32> = Vec::with_capacity(instances.len() * 3);
    let mut atom_positions: Vec<f32> = Vec::new();
    let mut atom_colors: Vec<f32> = Vec::new();
    let mut atom_channels: Vec<f32> = Vec::new();
    let mut instance_atom_range: Vec<f32> = Vec::with_capacity(instances.len() * 2);
    let mut atom_grid_headers: Vec<f32> = Vec::with_capacity(instances.len() * 6);
    let mut atom_cell_table: Vec<f32> = Vec::new();
    let mut scene_min = [f32::INFINITY; 3];
    let mut scene_max = [f32::NEG_INFINITY; 3];
    for (ii, inst) in instances.iter().enumerate() {
        let vol = inst.volume_index as usize;
        if vol >= volumes.len() {
            return Err(format!(
                "instance {ii}: volume_index {vol} out of range (have {} volumes)",
                volumes.len()
            ));
        }
        if inst.uniform_scale <= 0.0 {
            return Err(format!(
                "instance {ii}: uniform_scale must be > 0, got {}",
                inst.uniform_scale
            ));
        }
        let q = glam::Quat::from_array(inst.rotation_xyzw).normalize();
        let pos = glam::Vec3::from(inst.position);
        let s = inst.uniform_scale;
        let (wmin, wmax) = sdf_instance_world_aabb(&volumes[vol], inst);
        for k in 0..3 {
            scene_min[k] = scene_min[k].min(wmin[k]);
            scene_max[k] = scene_max[k].max(wmax[k]);
        }
        instance_headers.push(SdfInstanceHeader {
            instance_id: (ii as u64) + 1,
            asset_id: (vol as u64) + 1,
            volume_index: inst.volume_index,
            position: inst.position,
            rotation_xyzw: q.to_array(),
            uniform_scale: s,
            world_aabb_min: wmin,
            world_aabb_max: wmax,
        });
        instance_albedo.extend_from_slice(&inst.albedo);

        // World-space atoms (rot*scale*local + pos — the SDF grid transform).
        let world_pos: Vec<[f32; 3]> = atoms_per_instance[ii]
            .iter()
            .map(|a| (pos + q * (glam::Vec3::from(a.position) * s)).to_array())
            .collect();

        // Per-instance cell grid, anchored at the SAME padded world AABB min
        // the kernel reads back from the instance header. The atom SoA is
        // appended in CELL-SORTED order so each cell is one (start,count) run.
        let grid = build_sdf_atom_cell_grid(&world_pos, wmin, wmax, glass_detect_r);
        let atom_offset = atom_channels.len() as u32;
        for &orig in &grid.order {
            let a = &atoms_per_instance[ii][orig as usize];
            let wp = world_pos[orig as usize];
            atom_positions.extend_from_slice(&wp);
            atom_colors.extend_from_slice(&a.color);
            atom_channels.push(a.channel as f32);
        }
        let atom_count = atom_channels.len() as u32 - atom_offset;
        instance_atom_range.push(atom_offset as f32);
        instance_atom_range.push(atom_count as f32);

        // 6-float grid header + this instance's (start,count) cell table.
        atom_grid_headers.extend_from_slice(&[
            grid.dims[0] as f32,
            grid.dims[1] as f32,
            grid.dims[2] as f32,
            grid.cell_size,
            atom_cell_table.len() as f32, // table offset in FLOATS
            atom_offset as f32,           // atom_base into the SoA
        ]);
        for &(start, count) in &grid.cell_table {
            atom_cell_table.push(start as f32);
            atom_cell_table.push(count as f32);
        }
    }

    // Per-channel material table (9 floats/instance) + per-volume box-UV
    // params (4 floats/volume) — the buffers the megakernel's textured shade
    // reads as g_sdf_channel_material / g_sdf_volume_uv_params.
    let channel_material_table = pack_sdf_channel_material_table(channel_materials);
    let volume_uv_params = pack_sdf_uv_params(uv_params);

    let sdf = SdfLayer::from_parts_textured(
        volume_headers,
        instance_headers,
        all_distances,
        instance_albedo,
        atom_positions,
        atom_colors,
        atom_channels,
        instance_atom_range,
        atom_grid_headers,
        atom_cell_table,
        channel_material_table,
        volume_uv_params,
    );

    let mut scene = SceneState::new(width, height);
    scene.sdf = sdf;
    scene.mark_sdf_changed();

    // Sentinel triangle (BVH validity — invisible, see pathtrace_sdf_to_rgba).
    {
        use spectra_scene_state::MaterialLayer;
        let cx = 0.5 * (scene_min[0] + scene_max[0]);
        let cz = 0.5 * (scene_min[2] + scene_max[2]);
        let far = [cx, scene_min[1] - 1000.0, cz];
        scene.geometry.vertex_count = 3;
        scene.geometry.triangle_count = 1;
        scene.geometry.positions = vec![
            far[0],
            far[1],
            far[2],
            far[0] + 0.001,
            far[1],
            far[2],
            far[0],
            far[1],
            far[2] + 0.001,
        ];
        scene.geometry.normals = vec![0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0];
        scene.geometry.uvs = vec![0.0; 6];
        scene.geometry.indices = vec![0, 1, 2];
        scene.geometry.material_ids = vec![0];
        // The PBR materials the channel table indexes (g_materials), packed
        // exactly like the mesh path's (pack_vulkan_mesh_material, 156-float
        // Vulkan layout). When the caller supplies none, keep the M2 zeroed
        // sentinel material so frames stay byte-identical to the atoms path.
        // The off-frame sentinel triangle references slot 0 either way — it is
        // invisible, so its material never shades a pixel.
        scene.materials = if materials.is_empty() {
            MaterialLayer {
                params: vec![0.0; VULKAN_MATERIAL_FLOATS],
                spectral_spd: Default::default(),
                material_count: 1,
            }
        } else {
            let mut params = Vec::with_capacity(materials.len() * VULKAN_MATERIAL_FLOATS);
            for m in materials {
                params.extend_from_slice(&pack_vulkan_mesh_material(*m));
            }
            MaterialLayer {
                params,
                spectral_spd: Default::default(),
                material_count: materials.len(),
            }
        };
        scene.mark_geometry_changed();
        scene.mark_materials_changed();
    }

    // Four-light rig (identical to the M2 atoms path).
    let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
    let camera_fill = (glam::Vec3::from(eye) - glam::Vec3::from(target) + glam::Vec3::Y * 0.35)
        .normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::new();
    for (dir, color, intensity) in [
        (sun.to_array(), rig.sun_color, rig.sun_intensity),
        (glam::Vec3::Y.to_array(), [0.58, 0.62, 0.72], rig.sky_intensity),
        (camera_fill.to_array(), [0.72, 0.74, 0.78], rig.camera_fill),
        (rim_fill.to_array(), [0.45, 0.47, 0.52], rig.rim_fill),
    ] {
        light_data.extend_from_slice(&pack_vulkan_directional_light(dir, color, intensity));
    }
    scene.lights = LightLayer {
        light_data,
        light_count: 4,
    };
    scene.mark_lights_changed();

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::from(eye),
        glam::Vec3::from(target),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    config.target_spp = spp;
    // Glass needs continuation bounces; disable NRC so the refracted ray never
    // gets short-circuited into the cache at a deep bounce. Keep >= 3 bounces
    // (camera→glass→behind→...). IDENTICAL to the M2 entry point — the parity
    // gate compares this path's frames byte-wise against it.
    config.use_nrc = false;
    if config.max_bounces < 3 {
        config.max_bounces = 3;
    }
    let mut renderer = Renderer::new(gpu, config);
    renderer.set_sdf_albedo(instances[0].albedo);
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_sky_gradient(
        rig.sky_dome_zenith,
        rig.sky_dome_horizon,
        rig.sky_dome_intensity,
    );
    // ORDER MATTERS: set_texture_atlas silently no-ops before scene state
    // exists, so it must come AFTER load_scene_state (the verified-order
    // landmine — same wiring as pathtrace_mesh_lit_to_rgba).
    if !textures.is_empty() {
        renderer
            .set_texture_atlas(&tex_descs, &tex_data, textures.len() as u32)
            .map_err(|e| format!("set_texture_atlas: {e:?}"))?;
    }
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);

    let frame = renderer.render().map_err(|e| format!("render: {e:?}"))?;
    let n = (frame.width * frame.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..4 {
            out.push((frame.beauty[i * 4 + ch].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    // Debug AOV swap for the quality gates: replace the alpha channel with the
    // primary-hit roughness AOV (the megakernel writes g_aov_roughness at
    // u_bounce == 0 for every SDF hit), read straight off the GPU.
    if aov_roughness {
        let state = renderer
            .state
            .as_ref()
            .ok_or("aov_roughness: renderer has no state after render")?;
        let aov = state
            .aov_gpu
            .as_ref()
            .ok_or("aov_roughness: renderer allocated no AOV buffers")?;
        let handle = aov
            .get("roughness")
            .ok_or("aov_roughness: no 'roughness' AOV channel")?;
        let mut rough = vec![0.0f32; n];
        renderer
            .gpu
            .download_f32(handle, &mut rough)
            .map_err(|e| format!("aov_roughness readback: {e:?}"))?;
        for i in 0..n {
            out[i * 4 + 3] = (rough[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    Ok(out)
}

/// Build the flat texture atlas arrays for `Renderer::set_texture_atlas`.
///
/// Spectra's `slang/texture_atlas.slang` `TextureDesc` is
/// `[offset, width, height, channels]` per texture, where `offset` is an
/// index in FLOATS (f32 elements, not bytes or texels) into the flat
/// `g_tex_data` buffer; texels are row-major with interleaved channels
/// (`base = offset + (y*width + x)*channels`). `channels` 3 = RGB / 4 = RGBA;
/// `fetch_texel` pads missing channels (alpha defaults to 1). UVs wrap,
/// texel coordinates clamp at tile edges.
#[cfg(feature = "spectra-native")]
fn build_texture_atlas(textures: &[TextureImage]) -> Result<(Vec<u32>, Vec<f32>), String> {
    let mut descs = Vec::with_capacity(textures.len() * 4);
    let mut data: Vec<f32> = Vec::with_capacity(textures.iter().map(|t| t.data.len()).sum());
    for (i, t) in textures.iter().enumerate() {
        if t.channels < 1 || t.channels > 4 {
            return Err(format!(
                "texture {i}: channels must be 1..=4, got {}",
                t.channels
            ));
        }
        let expected = (t.width as usize) * (t.height as usize) * (t.channels as usize);
        if t.data.len() != expected {
            return Err(format!(
                "texture {i}: data.len()={} but width*height*channels={expected}",
                t.data.len()
            ));
        }
        descs.extend_from_slice(&[data.len() as u32, t.width, t.height, t.channels]);
        data.extend_from_slice(&t.data);
    }
    Ok((descs, data))
}

#[cfg(feature = "spectra-native")]
const VULKAN_MATERIAL_FLOATS: usize = 156;
#[cfg(feature = "spectra-native")]
const VULKAN_LIGHT_FLOATS: usize = 36;

#[cfg(feature = "spectra-native")]
fn pack_u32(x: u32) -> f32 {
    f32::from_bits(x)
}

#[cfg(feature = "spectra-native")]
fn pack_i32(x: i32) -> f32 {
    f32::from_bits(x as u32)
}

/// Pack `LightData` using the SPIR-V reflection layout for
/// `slang/light_types.slang`.
///
/// The shared Spectra constants describe the CUDA/tight layout (28 floats).
/// Vulkan pads `float3` fields, so `StructuredBuffer<LightData>` reads
/// 36 floats / 144 bytes per light. The tight layout shifts direction, color,
/// and intensity into the wrong fields.
#[cfg(feature = "spectra-native")]
fn pack_vulkan_directional_light(
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
) -> [f32; VULKAN_LIGHT_FLOATS] {
    let mut a = [0.0f32; VULKAN_LIGHT_FLOATS];

    a[0] = pack_u32(3); // LIGHT_DIRECTIONAL
    a[4] = direction[0];
    a[5] = direction[1];
    a[6] = direction[2];
    a[8] = color[0];
    a[9] = color[1];
    a[10] = color[2];
    a[11] = intensity;
    a[31] = pack_u32(0xFFFF_FFFF); // group_mask
    a[32] = pack_i32(0); // num_filters
    a[33] = pack_i32(0); // filter_offset

    a
}

/// Pack `MaterialData` using the SPIR-V reflection layout for
/// `slang/material_types.slang`.
///
/// The old CPU packer is the CUDA/tight layout: 132 floats / 528 bytes.
/// Vulkan reflection pads several `float3` fields and arrays, so
/// `StructuredBuffer<MaterialData>` uses a 156-float / 624-byte array stride.
/// Using the tight layout puts albedo/emission in padding slots and renders
/// later material IDs as black silhouettes.
#[cfg(feature = "spectra-native")]
fn pack_vulkan_mesh_material(m: PbrMaterial) -> [f32; VULKAN_MATERIAL_FLOATS] {
    let mut a = [0.0f32; VULKAN_MATERIAL_FLOATS];

    // `transmission > 0` selects MAT_GLASS (3): dispatch_sample's existing
    // `case MAT_GLASS` (material_dispatch.slang) refracts/transmits via the
    // same `sample_glass` BSDF the proven SDF window path calls — no shader
    // change. Opaque materials keep the historical MAT_LAMBERT (1) packing
    // byte-for-byte. Slot indices are the Vulkan SPIR-V reflection layout of
    // `MaterialData` (std430: float3 aligned to 16 bytes), cross-checked
    // against the slots this packer already proves out on screen (albedo 4-6,
    // eye colors 44/52, opacity_tex 72, uv_scale 94-95, substrate 144-147).
    let glass = m.transmission > 0.0;
    a[0] = pack_u32(if glass { 3 } else { 1 }); // MAT_GLASS : MAT_LAMBERT
    a[4] = m.base_color[0];
    a[5] = m.base_color[1];
    a[6] = m.base_color[2];
    a[7] = m.roughness;
    a[9] = m.metallic;
    a[10] = m.ior; // Default is 1.5 — identical to the old hardcoded constant

    a[20] = m.base_color[0];
    a[21] = m.base_color[1];
    a[22] = m.base_color[2];
    a[23] = m.emission_strength;

    if glass {
        // absorption_color (24-26) + absorption_depth (27): the EXACT values
        // the working SDF glass path passes to sample_glass in
        // megakernel.slang — `sample_glass(..., float3(0.0f), 1.0f, ...)`:
        // clear glass, no Beer-Lambert absorption. Replicated field-for-field
        // so the mesh MAT_GLASS route is parameter-identical to the proven
        // SDF window route.
        a[24] = 0.0;
        a[25] = 0.0;
        a[26] = 0.0;
        a[27] = 1.0;
    }

    a[28] = pack_i32(m.albedo_tex);
    a[29] = pack_i32(m.roughness_tex);
    a[30] = pack_i32(m.normal_tex);
    a[31] = pack_i32(-1); // displacement_tex
    a[33] = 0.5; // displacement_midlevel
    a[41] = 1.0; // hair_tangent.y

    a[44] = 0.3;
    a[45] = 0.2;
    a[46] = 0.1;
    a[47] = 0.005;
    a[48] = 1.376;
    a[52] = 0.9;
    a[53] = 0.88;
    a[54] = 0.85;
    a[55] = 0.35;
    a[57] = 1.0; // thin_film_ior

    a[58] = pack_u32(0); // is_holdout
    a[59] = pack_u32(0xFFFF_FFFF); // light_inclusion_mask
    a[60] = pack_u32(0); // light_exclusion_mask
    a[61] = pack_u32(0xFF); // visibility_mask
    a[72] = pack_i32(-1); // opacity_tex
    a[73] = pack_i32(if glass && m.thin_walled { 1 } else { 0 }); // thin_walled

    a[74] = 1.0; // diffuse_weight
    a[76] = 0.3; // specular_weight
    a[84] = 0.5;
    a[85] = 0.5;
    a[86] = 0.5;
    a[88] = 1.0;
    a[89] = 1.0;
    a[90] = 1.0;
    a[91] = 0.3;
    a[92] = 1.0; // energy_conservation
    a[94] = m.uv_scale[0];
    a[95] = m.uv_scale[1];
    a[99] = pack_u32(0); // layer_count

    for layer in 0..6 {
        let color = 100 + layer * 4;
        a[color] = 0.5;
        a[color + 1] = 0.5;
        a[color + 2] = 0.5;
        a[124 + layer] = 0.5;
        a[130 + layer] = 1.0;
        a[136 + layer] = pack_u32(0);
    }

    a[144] = 0.5;
    a[145] = 0.5;
    a[146] = 0.5;
    a[147] = 0.5;
    a[148] = pack_i32(0);
    a[149] = pack_i32(0);
    a[150] = pack_i32(-1);
    a[151] = 0.5;

    a
}

#[cfg(feature = "spectra-native")]
impl SpectraRenderBackend {
    /// Spawn render thread with near-realtime config (4 spp, DLSS, NRC, ReSTIR PT).
    pub fn realtime(width: u32, height: u32) -> Result<Self, String> {
        let config = RenderConfig::near_realtime(width, height);
        Self::spawn(config, width, height)
    }

    /// Spawn render thread with full offline config (128 spp).
    pub fn cinematic(width: u32, height: u32) -> Result<Self, String> {
        let mut config = RenderConfig::default();
        config.width = width;
        config.height = height;
        Self::spawn(config, width, height)
    }

    fn spawn(mut config: RenderConfig, width: u32, height: u32) -> Result<Self, String> {
        // Without a Slang kernel dir, the renderer falls back to temp_dir() and
        // finds no `.slang` files -> every dispatch is a no-op -> blank frames.
        // Resolve it: SPECTRA_SLANG_DIR override, else the spectra repo's slang/.
        if config.slang_kernel_dir.is_none() {
            config.slang_kernel_dir = resolve_slang_kernel_dir();
        }
        let (tx, rx) = channel::<RtCommand>();
        let last_output: Arc<Mutex<Arc<Vec<u8>>>> = Arc::new(Mutex::new(Arc::new(Vec::new())));
        let last_output_clone = Arc::clone(&last_output);

        std::thread::Builder::new()
            .name("spectra-render".into())
            .spawn(move || {
                // Backend selection: prefer Vulkan (Slang->SPIR-V + ash), which runs
                // on any Vulkan ICD including Mesa/lavapipe (CPU) on boxes with no
                // NVIDIA GPU. Fall back to the CUDA backend when Vulkan init fails
                // (e.g. no Vulkan loader, or a CUDA-only deployment where the CUDA
                // path is the intended one). The `SPECTRA_BACKEND` env var forces a
                // choice: `vulkan`/`vk` or `cuda`.
                //
                // Each backend is a distinct `Renderer<G>` instantiation, so the
                // generic `run_render_loop` is monomorphised once per backend and we
                // dispatch into it after selection.
                let forced = std::env::var("SPECTRA_BACKEND").ok();
                let prefer = forced.as_deref().map(str::to_ascii_lowercase);

                let try_vulkan = !matches!(prefer.as_deref(), Some("cuda"));
                let try_cuda = !matches!(prefer.as_deref(), Some("vulkan") | Some("vk"));

                if try_vulkan {
                    match VulkanSlangBackend::new(0) {
                        Ok(gpu) => {
                            eprintln!(
                                "[spectra-render] using Vulkan backend: {}",
                                gpu.device_name()
                            );
                            run_render_loop(gpu, config, rx, last_output_clone);
                            return;
                        }
                        Err(e) => {
                            if !try_cuda {
                                eprintln!("[spectra-render] Vulkan init failed: {e}");
                                return;
                            }
                            eprintln!(
                                "[spectra-render] Vulkan init failed ({e}); falling back to CUDA"
                            );
                        }
                    }
                }

                if try_cuda {
                    match CudarcSlangBackend::new(0) {
                        Ok(gpu) => {
                            eprintln!("[spectra-render] using CUDA backend: {}", gpu.device_name());
                            run_render_loop(gpu, config, rx, last_output_clone);
                        }
                        Err(e) => {
                            eprintln!(
                                "[spectra-render] GPU init failed (no backend available): {e}"
                            );
                        }
                    }
                }
            })
            .map_err(|e| format!("thread spawn failed: {e}"))?;

        Ok(Self {
            tx,
            last_output,
            fail_count: 0,
            width,
            height,
        })
    }

    /// Submit a frame request (non-blocking).
    ///
    /// Pass `new_scene = Some(...)` to upload a new scene; `None` to reuse the
    /// previously loaded scene.
    pub fn submit_frame(
        &mut self,
        new_scene: Option<SceneState>,
        camera: CameraLayer,
    ) -> Result<(), String> {
        self.tx
            .send(RtCommand::Render {
                scene: new_scene,
                camera,
            })
            .map_err(|e| {
                self.fail_count += 1;
                format!("render thread channel closed: {e}")
            })?;
        self.fail_count = 0;
        Ok(())
    }

    /// Read the last completed frame as a shared `Arc` — O(1), no data copy.
    ///
    /// Returns an empty `Arc<Vec<u8>>` before the first frame completes.
    pub fn read_last_output(&self) -> Arc<Vec<u8>> {
        self.last_output
            .lock()
            .map(|g| Arc::clone(&*g))
            .unwrap_or_else(|_| Arc::new(Vec::new()))
    }

    /// Number of consecutive `submit_frame()` failures. Reset to 0 on success.
    pub fn fail_count(&self) -> u32 {
        self.fail_count
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(feature = "spectra-native")]
impl Drop for SpectraRenderBackend {
    fn drop(&mut self) {
        let _ = self.tx.send(RtCommand::Shutdown);
    }
}

/// Generic render loop, monomorphised once per concrete backend `G`.
///
/// Owns the `Renderer<G>` and services `RtCommand`s until the channel closes or
/// `Shutdown` arrives. The published frame is written into `last_output` as a
/// `u8` RGBA buffer via `Arc::make_mut` allocation reuse.
#[cfg(feature = "spectra-native")]
fn run_render_loop<G: GpuBackend>(
    gpu: G,
    config: RenderConfig,
    rx: std::sync::mpsc::Receiver<RtCommand>,
    last_output: Arc<Mutex<Arc<Vec<u8>>>>,
) {
    let mut renderer = Renderer::new(gpu, config);
    let mut render_buf: Vec<u8> = Vec::new();

    loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => break,
        };
        match cmd {
            RtCommand::Shutdown => break,
            RtCommand::Render { scene, camera } => {
                // New scene geometry: upload the tessellated splat mesh.
                // `load_scene_state` replaces the old `load_splat_scene`.
                if let Some(s) = scene {
                    if let Err(e) = renderer.load_scene_state(s) {
                        eprintln!("[spectra-render] load_scene_state: {e}");
                    }
                }
                // Camera: the new renderer takes a column-major view matrix.
                // `set_camera_view_matrix` writes it into the scene's CameraLayer
                // (and keeps u_view_proj in sync); `set_view_proj` makes the
                // current-frame matrix explicit for temporal reprojection.
                renderer.set_camera_view_matrix(camera.view_matrix);
                renderer.set_view_proj(camera.view_matrix);

                // Render one full frame. `render()` replaces `render_splat_frame()`
                // and returns the tonemapped FrameOutput (no separate readback call).
                let frame = match renderer.render() {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("[spectra-render] render: {e}");
                        continue;
                    }
                };

                // Convert FrameOutput.beauty (tonemapped linear RGBA f32 in [0,1])
                // into the published u8 RGBA buffer. This is the readback step that
                // the old `read_splat_output_into` performed internally.
                let px = (frame.width * frame.height) as usize;
                render_buf.clear();
                render_buf.reserve(px * 4);
                for i in 0..px {
                    let r = frame.beauty[i * 4];
                    let g = frame.beauty[i * 4 + 1];
                    let b = frame.beauty[i * 4 + 2];
                    let a = frame.beauty[i * 4 + 3];
                    render_buf.push((r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                    render_buf.push((g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                    render_buf.push((b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                    render_buf.push((a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
                // Readback succeeded — swap into shared slot.
                // Arc::make_mut reuses the Vec allocation when no reader holds the Arc
                // (common case); the old frame data lands in render_buf for next frame's
                // resize-in-place reuse.
                if let Ok(mut guard) = last_output.lock() {
                    std::mem::swap(Arc::make_mut(&mut *guard), &mut render_buf);
                }
            }
        }
    }
}

#[cfg(all(test, feature = "spectra-native"))]
#[path = "splat_backend_tests/mod.rs"]
mod tests;

