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
mod tests {
    use super::SpectraRenderBackend;

    #[test]
    fn spectra_render_backend_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SpectraRenderBackend>();
    }

    #[test]
    fn read_last_output_before_first_frame_returns_empty() {
        // Construction requires GPU — just verify the type structure compiles correctly.
        // For GPU smoke test, run manually with: cargo test -p vox_render --features spectra-native
    }

    #[test]
    fn read_last_output_returns_arc() {
        // Type-level check: read_last_output() must return Arc<Vec<u8>>, not Vec<u8>.
        let _: fn(&SpectraRenderBackend) -> std::sync::Arc<Vec<u8>> =
            SpectraRenderBackend::read_last_output;
    }

    /// Real GPU render through the Vulkan (Slang->SPIR-V + ash) backend.
    ///
    /// Constructs a `VulkanSlangBackend` (runs on Mesa/lavapipe CPU Vulkan when no
    /// discrete GPU is present), builds a `Renderer`, loads a small quad scene
    /// directly in front of the camera, renders one frame, and asserts the beauty
    /// buffer has the expected size and real (non-zero-variance) pixel content.
    ///
    /// Run with:
    ///   LD_LIBRARY_PATH=$HOME/.local/slang/lib \
    ///     cargo test -p vox_render --features spectra-native \
    ///     vulkan_backend_renders_quad -- --nocapture
    #[test]
    fn vulkan_backend_renders_quad() {
        use crate::splat_convert::{camera_layer, splats_to_scene};
        use spectra_gpu::VulkanSlangBackend;
        use spectra_renderer::{RenderConfig, Renderer};
        use vox_core::types::GaussianSplat;

        let (w, h) = (64u32, 64u32);

        // 1. Bring up the Vulkan backend. If no Vulkan loader/ICD is reachable
        //    this is an environment limitation, not a code defect — report and skip.
        let gpu = match VulkanSlangBackend::new(0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[vulkan_backend_renders_quad] backend init failed: {e}");
                panic!("VulkanSlangBackend::new(0) failed: {e}");
            }
        };
        eprintln!(
            "[vulkan_backend_renders_quad] backend up: {}",
            spectra_gpu::GpuBackend::device_name(&gpu)
        );

        // 2. A near-realtime config at our target resolution (low spp for speed).
        let mut config = RenderConfig::near_realtime(w, h);
        config.slang_kernel_dir = super::resolve_slang_kernel_dir();
        assert!(
            config.slang_kernel_dir.is_some(),
            "Slang kernel dir not found (set SPECTRA_SLANG_DIR)"
        );
        let mut renderer = Renderer::new(gpu, config);

        // 3. One bright surface splat at the origin, facing the camera (+Z normal).
        let splat = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            std::array::from_fn(|_| half::f16::from_f32(0.9).to_bits()),
        );
        let scene = splats_to_scene(&[splat], w, h);
        renderer
            .load_scene_state(scene)
            .expect("load_scene_state should succeed");

        // 4. Camera looking down -Z at the quad from +Z (column-major view).
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        )
        .to_cols_array();
        let cam = camera_layer(view, std::f32::consts::FRAC_PI_4, w, h);
        renderer.set_camera_view_matrix(cam.view_matrix);
        renderer.set_view_proj(cam.view_matrix);

        // 5. Render one frame.
        let frame = renderer.render().expect("render() should succeed");

        // 6. Assert real pixel content: correct size + non-zero variance.
        assert_eq!(frame.width, w, "output width must match target");
        assert_eq!(frame.height, h, "output height must match target");
        assert_eq!(
            frame.beauty.len(),
            (w * h * 4) as usize,
            "beauty buffer must be width*height*4 floats"
        );

        let n = frame.beauty.len() as f64;
        let mean = frame.beauty.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = frame
            .beauty
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        eprintln!(
            "[vulkan_backend_renders_quad] mean={mean:.6} variance={variance:.9} \
             samples_done={}",
            frame.samples_done
        );
        assert!(
            variance > 1e-9,
            "rendered image must have real per-pixel variation (variance={variance})"
        );
    }

    /// First-frame correctness (plan Task 3): the FIRST `render()` of a fresh
    /// process must produce the same image as the second render of the same
    /// scene. Today the first frame races the runtime Slang compile — late
    /// kernels are skipped and it shades through a mismatched fallback path,
    /// so the first mean luma diverges from the second.
    ///
    /// Renders the textured checker quad twice in-process (each call builds a
    /// fresh backend + renderer, so the only shared state is process-wide
    /// kernel-compile state) and asserts the mean lumas agree within 2%.
    ///
    /// NOTE: must be the first GPU work in the process — run it alone:
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib first_frame_matches_second -- --nocapture
    #[test]
    fn first_frame_matches_second() {
        use super::{PbrMaterial, TextureImage, pathtrace_mesh_textured_to_rgba};

        let (w, h) = (128u32, 128u32);

        // Same checker quad as vulkan_pathtrace_textured_quad_checkerboard:
        // two CCW triangles facing +Z, UVs [0,1]², 64x64 2x2 checker albedo.
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = [[0u32, 1, 2], [0, 2, 3]];
        let material_ids = [0u8, 0];

        let (tw, th) = (64u32, 64u32);
        let mut data = Vec::with_capacity((tw * th * 3) as usize);
        for y in 0..th {
            for x in 0..tw {
                let v = if (x >= tw / 2) != (y >= th / 2) {
                    0.9f32
                } else {
                    0.05f32
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let checker = TextureImage {
            width: tw,
            height: th,
            channels: 3,
            data,
        };
        let materials = [PbrMaterial {
            albedo_tex: 0,
            ..Default::default()
        }];

        let render = || {
            pathtrace_mesh_textured_to_rgba(
                &positions,
                &normals,
                &uvs,
                &indices,
                &material_ids,
                &materials,
                std::slice::from_ref(&checker),
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
                std::f32::consts::FRAC_PI_4,
                w,
                h,
                8,
                [0.3, 0.5, 0.8],
            )
            .expect("pathtrace_mesh_textured_to_rgba should succeed")
        };

        let mean_luma = |img: &[u8]| -> f64 {
            let mut sum = 0.0f64;
            let n = (w * h) as usize;
            for i in 0..n {
                sum += 0.2126 * img[i * 4] as f64
                    + 0.7152 * img[i * 4 + 1] as f64
                    + 0.0722 * img[i * 4 + 2] as f64;
            }
            sum / n as f64
        };

        let first = render();
        let second = render();
        assert_eq!(first.len(), (w * h * 4) as usize);
        assert_eq!(second.len(), (w * h * 4) as usize);

        let l1 = mean_luma(&first);
        let l2 = mean_luma(&second);
        let rel = (l1 - l2).abs() / l2.max(1e-9);
        eprintln!(
            "[first_frame_matches_second] first mean luma={l1:.3} second mean luma={l2:.3} \
             rel delta={:.2}%",
            rel * 100.0
        );
        assert!(
            l2 > 1.0,
            "second render must show the lit quad (mean luma={l2:.3}) — \
             scene/kernels are broken if this is black"
        );
        assert!(
            rel <= 0.02,
            "first frame of a fresh process must match the second within 2% \
             (first={l1:.3}, second={l2:.3}, rel delta={:.2}%)",
            rel * 100.0
        );
    }

    /// LightRig parameterization (plan Task 4): the legacy entry point
    /// `pathtrace_mesh_textured_to_rgba` and the new
    /// `pathtrace_mesh_lit_to_rgba` with `LightRig { sun_dir, ..Default }`
    /// must produce BYTE-IDENTICAL images on the same scene (the default rig
    /// reproduces the previously hardcoded sun/sky/fill/rim exactly), and
    /// halving `sun_intensity` must measurably darken the image (>10% lower
    /// mean luma).
    ///
    /// Run with:
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib light_rig -- --nocapture
    #[test]
    fn light_rig_default_byte_identical_half_sun_darker() {
        use super::{
            LightRig, PbrMaterial, TextureImage, pathtrace_mesh_lit_to_rgba,
            pathtrace_mesh_textured_to_rgba,
        };

        let (w, h) = (128u32, 128u32);

        // Textured quad facing +Z (same shape as the checkerboard test) so the
        // byte-identity comparison also covers the texture-atlas path. Checker
        // albedo 0.55/0.05 keeps the sunlit half below tonemap saturation so
        // halving the sun shows up linearly in the mean luma.
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = [[0u32, 1, 2], [0, 2, 3]];
        let material_ids = [0u8, 0];

        let (tw, th) = (64u32, 64u32);
        let mut data = Vec::with_capacity((tw * th * 3) as usize);
        for y in 0..th {
            for x in 0..tw {
                let v = if (x >= tw / 2) != (y >= th / 2) {
                    0.55f32
                } else {
                    0.05f32
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let checker = TextureImage {
            width: tw,
            height: th,
            channels: 3,
            data,
        };
        let materials = [PbrMaterial {
            albedo_tex: 0,
            ..Default::default()
        }];

        let eye = [0.0f32, 0.0, 3.0];
        let target = [0.0f32, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let sun_dir = [0.3f32, 0.5, 0.8];
        let spp = 8;

        let legacy = pathtrace_mesh_textured_to_rgba(
            &positions,
            &normals,
            &uvs,
            &indices,
            &material_ids,
            &materials,
            std::slice::from_ref(&checker),
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            sun_dir,
        )
        .expect("legacy pathtrace_mesh_textured_to_rgba should succeed");

        let render_rig = |rig: &LightRig| {
            pathtrace_mesh_lit_to_rgba(
                &positions,
                &normals,
                &uvs,
                &indices,
                &material_ids,
                &materials,
                std::slice::from_ref(&checker),
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                rig,
            )
            .expect("pathtrace_mesh_lit_to_rgba should succeed")
        };

        let default_rig = render_rig(&LightRig {
            sun_dir,
            ..Default::default()
        });
        let half_sun = render_rig(&LightRig {
            sun_dir,
            sun_intensity: LightRig::default().sun_intensity * 0.5,
            ..Default::default()
        });

        assert_eq!(legacy.len(), (w * h * 4) as usize);
        assert_eq!(default_rig.len(), (w * h * 4) as usize);
        assert_eq!(half_sun.len(), (w * h * 4) as usize);

        let mean_luma = |img: &[u8]| -> f64 {
            let mut sum = 0.0f64;
            let n = (w * h) as usize;
            for i in 0..n {
                sum += 0.2126 * img[i * 4] as f64
                    + 0.7152 * img[i * 4 + 1] as f64
                    + 0.0722 * img[i * 4 + 2] as f64;
            }
            sum / n as f64
        };

        let l_legacy = mean_luma(&legacy);
        let l_default = mean_luma(&default_rig);
        let l_half = mean_luma(&half_sun);
        let diff_bytes = legacy
            .iter()
            .zip(default_rig.iter())
            .filter(|(a, b)| a != b)
            .count();
        let darker_pct = (1.0 - l_half / l_default.max(1e-9)) * 100.0;
        eprintln!(
            "[light_rig] legacy mean luma={l_legacy:.3} default-rig mean luma={l_default:.3} \
             differing bytes={diff_bytes} | half-sun mean luma={l_half:.3} \
             ({darker_pct:.1}% darker than default)"
        );

        assert!(
            l_default > 10.0,
            "default-rig render must show the lit quad (mean luma={l_default:.3}) — \
             scene/kernels are broken if this is black"
        );
        assert_eq!(
            legacy, default_rig,
            "default LightRig must reproduce the legacy hardcoded rig byte-identically \
             ({diff_bytes} bytes differ; legacy luma={l_legacy:.3}, default luma={l_default:.3})"
        );
        assert!(
            l_half < 0.9 * l_default,
            "halving sun_intensity must darken the image by >10% \
             (default={l_default:.3}, half-sun={l_half:.3}, only {darker_pct:.1}% darker)"
        );
    }

    /// Real GPU render: path-trace a quad with a 2x2 checkerboard albedo
    /// texture and assert the checker actually shows up in the pixels.
    ///
    /// The quad spans [-1,1]² at z=0 with UVs [0,1]², viewed head-on from
    /// (0,0,3). The 64x64 RGB texture is bright (0.9) where the texel's
    /// half-x and half-y quadrant parities differ, dark (0.05) where they
    /// match — so in image space the bright checker quadrants land on one
    /// image diagonal and the dark on the other, regardless of axis flips.
    /// Asserts bright-diagonal mean luminance >= 2x dark-diagonal mean, and
    /// that the SAME quad rendered without a texture shows no such asymmetry
    /// (min/max diagonal ratio > 0.8).
    ///
    /// Run with:
    ///   SPECTRA_BACKEND=vulkan scripts/build-spectra-native.sh test -p vox_render \
    ///     --features spectra-native --lib vulkan_pathtrace_textured_quad_checkerboard \
    ///     -- --nocapture
    #[test]
    fn vulkan_pathtrace_textured_quad_checkerboard() {
        use super::{PbrMaterial, TextureImage, pathtrace_mesh_textured_to_rgba};

        let (w, h) = (128u32, 128u32);

        // Quad: two CCW triangles facing +Z, UVs spanning [0,1]².
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = [[0u32, 1, 2], [0, 2, 3]];
        let material_ids = [0u8, 0];

        // 64x64 RGB checkerboard: 2x2 quadrants, bright where the quadrant
        // parities differ (diagonal pattern), dark where they match.
        let (tw, th) = (64u32, 64u32);
        let mut data = Vec::with_capacity((tw * th * 3) as usize);
        for y in 0..th {
            for x in 0..tw {
                let v = if (x >= tw / 2) != (y >= th / 2) {
                    0.9f32
                } else {
                    0.05f32
                };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let checker = TextureImage {
            width: tw,
            height: th,
            channels: 3,
            data,
        };

        let eye = [0.0f32, 0.0, 3.0];
        let target = [0.0f32, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let sun_dir = [0.3f32, 0.5, 0.8];
        let spp = 8;

        let render = |materials: &[PbrMaterial], textures: &[TextureImage]| {
            pathtrace_mesh_textured_to_rgba(
                &positions,
                &normals,
                &uvs,
                &indices,
                &material_ids,
                materials,
                textures,
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                sun_dir,
            )
            .expect("pathtrace_mesh_textured_to_rgba should succeed")
        };

        let textured = render(
            &[PbrMaterial {
                albedo_tex: 0,
                ..Default::default()
            }],
            std::slice::from_ref(&checker),
        );
        let untextured = render(
            &[PbrMaterial {
                base_color: [0.8, 0.8, 0.8],
                ..Default::default()
            }],
            &[],
        );
        assert_eq!(textured.len(), (w * h * 4) as usize);
        assert_eq!(untextured.len(), (w * h * 4) as usize);

        // Mean RGB luminance over a central window (half-size w/8) of each
        // image quadrant. Window centers at 1/4 and 3/4 of the frame project
        // well inside the quad (NDC 0.25..0.75 => world 0.31..0.93 at z=0)
        // and never touch the UV-0.5 checker boundary at world 0.
        let quadrant_mean = |img: &[u8], cx: u32, cy: u32| -> f64 {
            let r = w / 8;
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for y in (cy - r)..(cy + r) {
                for x in (cx - r)..(cx + r) {
                    let i = ((y * w + x) * 4) as usize;
                    sum += (img[i] as f64 + img[i + 1] as f64 + img[i + 2] as f64) / 3.0;
                    n += 1;
                }
            }
            sum / n as f64
        };
        let quads = |img: &[u8]| -> [f64; 4] {
            [
                quadrant_mean(img, w / 4, h / 4),     // image TL
                quadrant_mean(img, 3 * w / 4, h / 4), // image TR
                quadrant_mean(img, w / 4, 3 * h / 4), // image BL
                quadrant_mean(img, 3 * w / 4, 3 * h / 4), // image BR
            ]
        };

        // Diagonal grouping: one diagonal holds the bright checker quadrants,
        // the other the dark — whichever way the image axes map to UV.
        let diag = |q: [f64; 4]| -> (f64, f64) {
            let a = (q[0] + q[3]) / 2.0; // TL + BR
            let b = (q[1] + q[2]) / 2.0; // TR + BL
            (a.max(b), a.min(b))
        };

        let tq = quads(&textured);
        let uq = quads(&untextured);
        let (t_bright, t_dark) = diag(tq);
        let (u_bright, u_dark) = diag(uq);
        eprintln!(
            "[vulkan_pathtrace_textured_quad_checkerboard] textured quadrants \
             TL={:.1} TR={:.1} BL={:.1} BR={:.1} -> bright={t_bright:.1} dark={t_dark:.1}",
            tq[0], tq[1], tq[2], tq[3]
        );
        eprintln!(
            "[vulkan_pathtrace_textured_quad_checkerboard] untextured quadrants \
             TL={:.1} TR={:.1} BL={:.1} BR={:.1} -> max={u_bright:.1} min={u_dark:.1}",
            uq[0], uq[1], uq[2], uq[3]
        );

        assert!(
            t_bright >= 2.0 * t_dark,
            "bright checker quadrants must be >= 2x dark ones \
             (bright={t_bright:.2}, dark={t_dark:.2})"
        );
        assert!(
            u_dark > 0.8 * u_bright,
            "untextured quad must NOT show the checker asymmetry \
             (max={u_bright:.2}, min={u_dark:.2})"
        );
    }

    // ---- M0: native SDF-volume primitive (confetti → wall) ----------------

    /// Minimal dependency-free PNG writer (RGBA8, zlib stored blocks). Mirrors
    /// vox_app::shell::cpu_render::write_png so the SDF still can be inspected.
    #[cfg(feature = "spectra-native")]
    fn write_png_rgba(path: &str, rgba: &[u8], w: u32, h: u32) {
        use std::io::Write;
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = 0xFFFF_FFFFu32;
            for &b in bytes {
                crc ^= b as u32;
                for _ in 0..8 {
                    let mask = (crc & 1).wrapping_neg();
                    crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
                }
            }
            !crc
        }
        fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(kind);
            out.extend_from_slice(data);
            let mut crc_in = Vec::with_capacity(4 + data.len());
            crc_in.extend_from_slice(kind);
            crc_in.extend_from_slice(data);
            out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
        }
        let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
        let stride = (w * 4) as usize;
        for y in 0..h as usize {
            raw.push(0u8);
            raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
        }
        // zlib stored blocks (no deflate dependency).
        let mut comp = vec![0x78u8, 0x01u8];
        let mut i = 0;
        while i < raw.len() {
            let block = (raw.len() - i).min(0xFFFF);
            let is_last = i + block >= raw.len();
            comp.push(if is_last { 1 } else { 0 });
            comp.extend_from_slice(&(block as u16).to_le_bytes());
            comp.extend_from_slice(&(!(block as u16)).to_le_bytes());
            comp.extend_from_slice(&raw[i..i + block]);
            i += block;
        }
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in &raw {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        comp.extend_from_slice(&((b << 16) | a).to_be_bytes());

        let mut png = Vec::new();
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &comp);
        chunk(&mut png, b"IEND", &[]);
        let mut f = std::fs::File::create(path).expect("create png");
        f.write_all(&png).expect("write png");
    }

    /// A pixel is "covered" (part of the rendered building surface) when its
    /// luma clearly exceeds the dark sky-gradient background. The SDF is lit by
    /// the sun + fills; the gradient sky is dark-ish at the horizon, so a simple
    /// luma threshold separates solid surface from sky.
    #[cfg(feature = "spectra-native")]
    fn luma(px: &[u8]) -> f32 {
        0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32
    }

    /// Solidity = how SOLID (vs confetti) the rendered surface is, independent of
    /// the building's (non-rectangular) silhouette. For every scanline that
    /// contains surface pixels, measure the filled fraction within that row's
    /// span (first→last surface pixel). A continuous wall fills its span ≈ 1.0;
    /// sparse splat confetti leaves gaps inside the span and scores far lower.
    /// Returns (mean_span_fill, total_surface_pixels). Rows whose span is shorter
    /// than `min_span` (slivers / antialias fringe) are ignored.
    #[cfg(feature = "spectra-native")]
    fn surface_solidity(rgba: &[u8], w: u32, h: u32, thr: f32, min_span: u32) -> (f64, u64) {
        let mut sum_fill = 0.0f64;
        let mut rows = 0u64;
        let mut total = 0u64;
        for y in 0..h {
            let mut first = None;
            let mut last = 0u32;
            let mut count = 0u32;
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                if luma(&rgba[i..i + 4]) > thr {
                    first.get_or_insert(x);
                    last = x;
                    count += 1;
                }
            }
            total += count as u64;
            if let Some(f) = first {
                let span = last - f + 1;
                if span >= min_span {
                    sum_fill += count as f64 / span as f64;
                    rows += 1;
                }
            }
        }
        let mean = if rows > 0 { sum_fill / rows as f64 } else { 0.0 };
        (mean, total)
    }

    /// M0 ACCEPTANCE: the engine-owned Spectra path tracer renders ONE real
    /// building's cooked SDF as a SOLID, continuous surface via the NATIVE
    /// SDF-volume primitive (sphere trace in megakernel.slang) — NOT the lossy
    /// triangle-quad splat bridge. Loads forge.house.craftsman's cooked 64³-class
    /// GWN-signed field from atoms.json, renders it, writes a PNG, and asserts
    /// the façade fills >= 97% of its projected silhouette bounding box (confetti
    /// would be sparse). Prints the OLD splat-bridge coverage for the same camera
    /// as the confetti→wall delta.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///   SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SLANG_DIR=$HOME/slang-sdk \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_building_renders_solid_surface -- --nocapture --test-threads=1
    #[test]
    fn sdf_building_renders_solid_surface() {
        use super::{LightRig, SdfVolumeInput, pathtrace_sdf_to_rgba};

        // --- Load the cooked SDF from the civitas craftsman asset ----------
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms")
            .join("forge.house.craftsman.atoms.json");
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
        let sdf = &json["sdf"];
        let res: Vec<u64> = sdf["resolution"]
            .as_array()
            .expect("resolution array")
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let origin: Vec<f32> = sdf["origin"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let voxel_size = sdf["voxel_size"].as_f64().unwrap() as f32;
        let narrow_band = sdf["narrow_band"].as_f64().unwrap() as f32;
        let snorm: Vec<i64> = sdf["distances_snorm16"]
            .as_array()
            .expect("distances_snorm16 array")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        // Decode snorm16 -> metres, exactly as vox_physics::sdf::from_snorm16_grid.
        let distances: Vec<f32> = snorm
            .iter()
            .map(|&v| {
                let n = if v as i32 == i16::MIN as i32 {
                    -1.0
                } else {
                    v as f32 / i16::MAX as f32
                };
                n.clamp(-1.0, 1.0) * narrow_band
            })
            .collect();
        let resolution = [res[0] as u32, res[1] as u32, res[2] as u32];
        eprintln!(
            "[sdf_building] craftsman SDF: res={:?} voxel={voxel_size:.4} band={narrow_band:.4} \
             n_dist={} negatives(inside)={}",
            resolution,
            distances.len(),
            distances.iter().filter(|d| **d < 0.0).count()
        );
        assert!(
            distances.iter().any(|d| *d < 0.0),
            "SDF must contain interior (negative) samples — else it is not a solid"
        );

        let volume = SdfVolumeInput {
            resolution,
            origin: [origin[0], origin[1], origin[2]],
            voxel_size,
            narrow_band,
            distances,
        };

        // --- Camera framing the SDF grid (identity instance transform) -----
        // Grid world extent (scale 1, position 0): origin .. origin+(res-1)*voxel.
        let gmin = volume.origin;
        let gmax = [
            volume.origin[0] + (resolution[0] - 1) as f32 * voxel_size,
            volume.origin[1] + (resolution[1] - 1) as f32 * voxel_size,
            volume.origin[2] + (resolution[2] - 1) as f32 * voxel_size,
        ];
        let center = [
            0.5 * (gmin[0] + gmax[0]),
            0.5 * (gmin[1] + gmax[1]),
            0.5 * (gmin[2] + gmax[2]),
        ];
        let radius = {
            let dx = gmax[0] - gmin[0];
            let dy = gmax[1] - gmin[1];
            let dz = gmax[2] - gmin[2];
            0.5 * (dx * dx + dy * dy + dz * dz).sqrt()
        };
        // Front-quarter view, eye pulled back ~2.2 radii so the whole building
        // projects inside the frame.
        let dist = radius * 2.2;
        let eye = [
            center[0] + dist * 0.55,
            center[1] + dist * 0.35,
            center[2] + dist * 0.75,
        ];
        let (w, h) = (192u32, 192u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        // Dark sky / dark fills so the only lit pixels are the SDF surface — the
        // coverage gate then cleanly separates solid façade from background. The
        // SDF's view-facing fill (in megakernel.slang) lights every hit pixel.
        let rig = LightRig {
            sun_dir: [0.4, 0.7, 0.55],
            sun_intensity: 2.0,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 0.0,
            sky_dome_zenith: [0.0, 0.0, 0.0],
            sky_dome_horizon: [0.0, 0.0, 0.0],
            ..Default::default()
        };

        let rgba = pathtrace_sdf_to_rgba(
            &volume,
            [0.0, 0.0, 0.0],
            1.0,
            [0.72, 0.68, 0.60],
            eye,
            center,
            fov_y,
            w,
            h,
            4,
            &rig,
        )
        .expect("SDF render should succeed");

        let out_dir = std::env::temp_dir();
        let png_path = out_dir.join("ochroma_sdf_craftsman.png");
        write_png_rgba(png_path.to_str().unwrap(), &rgba, w, h);
        eprintln!("[sdf_building] wrote {}", png_path.display());

        // --- Solidity: per-row span-fill (handles the non-rectangular house
        // silhouette). Background is the dark sky; lit pixels are the surface.
        let thr = 30.0f32; // out of 255 luma
        let (sdf_coverage, covered) = surface_solidity(&rgba, w, h, thr, 4);
        assert!(covered > 0, "render is entirely background — SDF never hit");

        // --- OLD splat-bridge coverage for the SAME asset/camera -----------
        // Build camera-facing quad splats over the SDF surface (one per inside
        // voxel boundary) and path-trace them through the lossy bridge, so the
        // confetti→wall delta is printed side by side. Skippable via
        // OCHROMA_SKIP_SPLAT_COMPARE=1 to iterate on the SDF path alone.
        let splat_coverage = if std::env::var("OCHROMA_SKIP_SPLAT_COMPARE").is_ok() {
            f64::NAN
        } else {
            use vox_core::types::GaussianSplat;
            let [nx, ny, nz] = resolution;
            let mut splats: Vec<GaussianSplat> = Vec::new();
            let dist_at = |x: u32, y: u32, z: u32| -> f32 {
                volume.distances[(x + nx * (y + ny * z)) as usize]
            };
            // Surface voxels: inside cell adjacent to an outside cell → a splat
            // at the cell centre. This is the splat sampling the bridge gets.
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        if dist_at(x, y, z) >= 0.0 {
                            continue;
                        }
                        let neighbour_outside = [
                            (x + 1 < nx).then(|| dist_at(x + 1, y, z)),
                            (x > 0).then(|| dist_at(x - 1, y, z)),
                            (y + 1 < ny).then(|| dist_at(x, y + 1, z)),
                            (y > 0).then(|| dist_at(x, y - 1, z)),
                            (z + 1 < nz).then(|| dist_at(x, y, z + 1)),
                            (z > 0).then(|| dist_at(x, y, z - 1)),
                        ]
                        .iter()
                        .flatten()
                        .any(|d| *d >= 0.0);
                        if !neighbour_outside {
                            continue;
                        }
                        let p = [
                            volume.origin[0] + x as f32 * voxel_size,
                            volume.origin[1] + y as f32 * voxel_size,
                            volume.origin[2] + z as f32 * voxel_size,
                        ];
                        splats.push(GaussianSplat::surface(
                            p,
                            [voxel_size, 0.0, 0.0],
                            [0.0, voxel_size, 0.0],
                            voxel_size,
                            1.0,
                            255,
                            std::array::from_fn(|_| half::f16::from_f32(0.7).to_bits()),
                        ));
                    }
                }
            }
            match super::pathtrace_splats_to_rgba(
                &splats, eye, center, fov_y, w, h, 16, rig.sun_dir,
            ) {
                Ok(srgba) => {
                    let (solidity, sc) = surface_solidity(&srgba, w, h, thr, 4);
                    let png2 = out_dir.join("ochroma_splat_craftsman.png");
                    write_png_rgba(png2.to_str().unwrap(), &srgba, w, h);
                    eprintln!(
                        "[sdf_building] wrote {} (splat bridge, {sc} surface px)",
                        png2.display()
                    );
                    solidity
                }
                Err(e) => {
                    eprintln!("[sdf_building] splat-bridge render failed: {e}");
                    f64::NAN
                }
            }
        };

        eprintln!(
            "[sdf_building] SOLID COVERAGE (per-row span fill — continuous wall ≈ 1.0):\n  \
             SDF native primitive : {:.4} ({covered} surface px)\n  \
             OLD splat bridge     : {:.4}   <-- confetti",
            sdf_coverage, splat_coverage
        );

        assert!(
            sdf_coverage >= 0.97,
            "SDF façade must be a continuous filled surface (span-fill solidity \
             {:.4} < 0.97). A solid building fills each scanline span; confetti \
             leaves gaps.",
            sdf_coverage
        );
    }

    /// Load one cooked atoms.json SDF into a `SdfVolumeInput`, decoding snorm16
    /// distances to metres exactly as `vox_physics::sdf::from_snorm16_grid`.
    #[cfg(feature = "spectra-native")]
    fn load_atoms_sdf(path: &std::path::Path) -> super::SdfVolumeInput {
        let bytes =
            std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse atoms.json");
        let sdf = &json["sdf"];
        let res: Vec<u64> = sdf["resolution"]
            .as_array()
            .expect("resolution")
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let origin: Vec<f32> = sdf["origin"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let voxel_size = sdf["voxel_size"].as_f64().unwrap() as f32;
        let narrow_band = sdf["narrow_band"].as_f64().unwrap() as f32;
        let snorm: Vec<i64> = sdf["distances_snorm16"]
            .as_array()
            .expect("distances_snorm16")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let distances: Vec<f32> = snorm
            .iter()
            .map(|&v| {
                let n = if v as i32 == i16::MIN as i32 {
                    -1.0
                } else {
                    v as f32 / i16::MAX as f32
                };
                n.clamp(-1.0, 1.0) * narrow_band
            })
            .collect();
        super::SdfVolumeInput {
            resolution: [res[0] as u32, res[1] as u32, res[2] as u32],
            origin: [origin[0], origin[1], origin[2]],
            voxel_size,
            narrow_band,
            distances,
        }
    }

    /// Load the cooked ATOMS (position + channel + colour) from an atoms.json
    /// into `SdfSceneAtom`s in the asset's LOCAL space (same space as the SDF
    /// grid). `force_glass_opaque` reclassifies glass→facade (the control render
    /// that proves windows differ from walls). Returns (atoms, raw_glass_count).
    #[cfg(feature = "spectra-native")]
    fn load_atoms_material(
        path: &std::path::Path,
        force_glass_opaque: bool,
    ) -> (Vec<super::SdfSceneAtom>, usize) {
        let bytes =
            std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse atoms.json");
        let arr = json["atoms"].as_array().expect("atoms array");
        let mut atoms = Vec::with_capacity(arr.len());
        let mut glass = 0usize;
        for a in arr {
            let p = a["position"].as_array().expect("position");
            let c = a["color"].as_array().expect("color");
            let ch_str = a["channel"].as_str().unwrap_or("other");
            let mut channel = super::sdf_atom_channel_from_str(ch_str);
            if channel == super::SDF_ATOM_CH_GLASS {
                glass += 1;
                if force_glass_opaque {
                    channel = super::SDF_ATOM_CH_FACADE;
                }
            }
            atoms.push(super::SdfSceneAtom {
                position: [
                    p[0].as_f64().unwrap() as f32,
                    p[1].as_f64().unwrap() as f32,
                    p[2].as_f64().unwrap() as f32,
                ],
                color: [
                    c[0].as_f64().unwrap() as f32,
                    c[1].as_f64().unwrap() as f32,
                    c[2].as_f64().unwrap() as f32,
                ],
                channel,
            });
        }
        (atoms, glass)
    }

    /// M2 ACCEPTANCE: per-surface MATERIAL from the cooked atoms + real GLASS
    /// windows. The engine-owned Spectra path tracer renders one cooked craftsman
    /// as a native SDF (geometry only), and at each surface hit k-NN-gathers the
    /// nearest atoms to recover the per-surface channel + colour — so walls/roof/
    /// trim show real COLOUR (not one flat albedo), and WINDOWS render as real
    /// path-traced GLASS (the SDF glass hit transmits/refracts and continues the
    /// path instead of terminating opaque, so the window is see-through and shows
    /// the bright sky backdrop behind it). Proven by an A/B render: the same
    /// scene with glass reclassified as opaque facade. The pixels that DIFFER are
    /// the glass-classified window pixels, and they get BRIGHTER (sky shows
    /// through) — a wall cannot. Writes both PNGs, prints the glass-pixel count
    /// and the colour-variety + transparency verdict.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///   SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SLANG_DIR=$HOME/slang-sdk \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_craftsman_atom_material_and_glass -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_craftsman_atom_material_and_glass() {
        use super::{
            pathtrace_sdf_scene_with_atoms_to_rgba, LightRig, SdfSceneInstance,
        };

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

        let volume = load_atoms_sdf(&asset_path);
        let (atoms, raw_glass) = load_atoms_material(&asset_path, false);
        let (atoms_opaque, _) = load_atoms_material(&asset_path, true);
        eprintln!(
            "[sdf_glass] craftsman: res={:?} voxel={:.4} band={:.4} atoms={} (glass={})",
            volume.resolution,
            volume.voxel_size,
            volume.narrow_band,
            atoms.len(),
            raw_glass
        );
        assert!(raw_glass > 50, "expected many cooked glass atoms, got {raw_glass}");

        // Place the single building so its base sits on y=0, centred at origin.
        let ground_y = -volume.origin[1];
        let instance = SdfSceneInstance {
            volume_index: 0,
            position: [
                -(volume.origin[0]
                    + (volume.resolution[0] - 1) as f32 * volume.voxel_size * 0.5),
                ground_y,
                -(volume.origin[2]
                    + (volume.resolution[2] - 1) as f32 * volume.voxel_size * 0.5),
            ],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            uniform_scale: 1.0,
            albedo: [0.7, 0.7, 0.7],
        };
        let instances = [instance];

        // Camera: look at the FRONT facade (the cooked windows are on +Z, the
        // front wall) from slightly above, close enough that windows are large.
        let (w, h) = (320u32, 320u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let center = [0.0f32, 4.5, 0.0];
        let eye = [3.0f32, 6.0, 22.0]; // in front (+Z), slight oblique
        // Bright distinctive sky so transparent glass (which transmits to the
        // sky behind) reads visibly brighter than opaque facade.
        let rig = LightRig {
            sun_dir: [0.3, 0.6, 0.7],
            sun_intensity: 2.2,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 1.0,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };

        let t0 = std::time::Instant::now();
        let rgba = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&volume),
            &instances,
            &[atoms],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            &rig,
        )
        .expect("SDF atom-material + glass render should succeed");
        let secs = t0.elapsed().as_secs_f64();

        let rgba_opaque = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&volume),
            &instances,
            &[atoms_opaque],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            &rig,
        )
        .expect("control (glass→opaque) render should succeed");

        let out_dir = std::env::temp_dir();
        let glass_png = out_dir.join("ochroma_sdf_craftsman_glass.png");
        let opaque_png = out_dir.join("ochroma_sdf_craftsman_opaque.png");
        write_png_rgba(glass_png.to_str().unwrap(), &rgba, w, h);
        write_png_rgba(opaque_png.to_str().unwrap(), &rgba_opaque, w, h);
        eprintln!("[sdf_glass] wrote {}", glass_png.display());
        eprintln!("[sdf_glass] wrote {} (control)", opaque_png.display());

        // The background is the smooth blue SKY DOME gradient (b clearly the max
        // channel, bright). A pixel is on the BUILDING if it is NOT that bright
        // blue sky. Glass windows can themselves be dark/blue, but they sit
        // inside the building silhouette and (critically) DIFFER between the two
        // renders — the sky is byte-identical between them.
        let is_sky = |px: &[u8]| -> bool {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            b > r + 14 && b > g + 8 && (r + g + b) > 330
        };

        let mut building_px = 0usize;
        // --- Colour variety across the building (proves per-surface material,
        // not one flat albedo): bucket each building pixel's dominant hue and
        // require several distinct buckets to be well-populated. -------------
        let mut hue_buckets = [0usize; 12];
        // --- Glass detection via A/B diff: a WINDOW pixel is a building pixel
        // whose colour changes meaningfully when glass is enabled vs. when the
        // glass atoms are reclassified to opaque facade. A wall is byte-identical
        // between the two renders; only the path-traced glass (transmit/refract/
        // reflect, blue tint) differs. Here the front windows transmit into the
        // building's darker interior so they go DARKER than the opaque facade. --
        let mut glass_pixels = 0usize;
        let mut glass_diff_sum = 0.0f64;
        for p in 0..(w * h) as usize {
            let g = &rgba[p * 4..p * 4 + 4];
            let o = &rgba_opaque[p * 4..p * 4 + 4];
            // On the building when the OPAQUE control (no glass holes) is not sky.
            let on_building = !is_sky(o);
            if on_building {
                building_px += 1;
                let (r, gg, b) = (g[0] as f32, g[1] as f32, g[2] as f32);
                if r.max(gg).max(b) - r.min(gg).min(b) > 12.0 {
                    let hue = rgb_hue(r, gg, b); // 0..360
                    let bucket = ((hue / 30.0) as usize).min(11);
                    hue_buckets[bucket] += 1;
                }
                // Per-channel absolute difference between the two renders.
                let diff = (g[0] as f32 - o[0] as f32).abs()
                    + (g[1] as f32 - o[1] as f32).abs()
                    + (g[2] as f32 - o[2] as f32).abs();
                if diff > 36.0 {
                    glass_pixels += 1;
                    glass_diff_sum += diff as f64;
                }
            }
        }

        let populated_hues = hue_buckets.iter().filter(|&&n| n >= 8).count();
        let mean_diff = if glass_pixels > 0 {
            glass_diff_sum / glass_pixels as f64
        } else {
            0.0
        };
        eprintln!(
            "[sdf_glass] M2 ATOM MATERIAL + GLASS (all native SDF):\n  \
             building pixels    : {building_px}\n  \
             glass-classified px: {glass_pixels} (window pixels that CHANGE when \
             glass is enabled — path-traced through the pane)\n  \
             mean glass A/B diff: {mean_diff:.1} (sum |ΔRGB| over the pane pixels)\n  \
             colour hue buckets : {populated_hues}/12 populated (per-surface colour \
             variety)\n  \
             hue histogram      : {hue_buckets:?}\n  \
             seconds/frame      : {secs:.2}s (x2 renders)",
        );

        // --- Acceptance gates. ----------------------------------------------
        assert!(
            building_px > 5000,
            "building barely visible ({building_px} px) — camera/placement wrong"
        );
        // Walls/roof/trim must show REAL per-surface colour, not one flat albedo:
        // multiple distinct hue buckets must be populated.
        assert!(
            populated_hues >= 3,
            "expected per-surface colour variety (>= 3 populated hue buckets), got \
             {populated_hues}. A single flat albedo would fill one bucket."
        );
        // WINDOWS must render as GLASS: a real, non-trivial set of building pixels
        // must be path-traced through the pane (differ from the opaque control).
        // A purely opaque building would have ZERO such pixels (the two renders
        // would be byte-identical everywhere on the building).
        assert!(
            glass_pixels >= 200,
            "expected >= 200 glass-classified window pixels (path-traced through \
             the pane, differing from the opaque-glass control), got {glass_pixels}. \
             Either windows are smoothed out of the SDF and the atom gather can't \
             locate them, or the glass hit is terminating opaque instead of \
             transmitting."
        );
    }

    /// Wave-1 Task 1 ACCEPTANCE: the cell-grid atom gather is a pure
    /// acceleration. The SAME craftsman frame rendered through the grid path
    /// (`pathtrace_sdf_scene_textured_to_rgba`, `u_sdf_textured = 1`,
    /// gather-only mode: no materials/textures yet, identical flat-blend
    /// shading) must match the linear-gather oracle
    /// (`pathtrace_sdf_scene_with_atoms_to_rgba`, `u_sdf_textured = 0`)
    /// byte-for-byte within 1 LSB, while visiting ~80x fewer atoms per hit.
    /// The gather cost is host-computed: avg atoms tested per probe = sum of
    /// the 3x3x3 cell-neighbourhood counts at 1,000 surface points (the atoms
    /// themselves ARE surface samples).
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_gather_grid_matches_linear -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_gather_grid_matches_linear() {
        use super::{
            pathtrace_sdf_scene_textured_to_rgba, pathtrace_sdf_scene_with_atoms_to_rgba,
        };

        // Shared craftsman harness: the M2 glass test's placement + camera +
        // rig, so the frame exercises facade, roof, trim AND glass-detect
        // gather paths.
        let hz = craftsman_harness();
        let CraftsmanHarness {
            ref asset_path,
            ref volume,
            ref atoms,
            raw_glass,
            instance,
            eye,
            center,
            fov_y,
            w,
            h,
            rig,
            ..
        } = hz;
        let atoms = atoms.clone();
        let n_atoms = atoms.len();
        eprintln!(
            "[sdf_gather_grid] craftsman: res={:?} voxel={:.4} atoms={n_atoms} (glass={raw_glass})",
            volume.resolution, volume.voxel_size
        );
        let instances = [instance];

        // Linear oracle (M2 entry point, u_sdf_textured = 0: full atom scan).
        let rgba_linear = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(volume),
            &instances,
            &[atoms.clone()],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            &rig,
        )
        .expect("linear-gather oracle render should succeed");

        // Grid path (new entry point, u_sdf_textured = 1: 3x3x3 cell gather,
        // materials empty + channel table all -1 -> identical flat-blend
        // shading, gather-only mode).
        let rgba_grid = pathtrace_sdf_scene_textured_to_rgba(
            &volume_slice(volume),
            &[load_forge_uv_params(asset_path)],
            &instances,
            &[atoms.clone()],
            &[Vec::new()],
            &[super::SdfChannelMaterials {
                material_for_channel: [-1i32; 9],
            }],
            &[],
            &[],
            eye,
            center,
            fov_y,
            w,
            h,
            6,
            0,
            &rig,
            false,
        )
        .expect("grid-gather render should succeed");

        let out_dir = std::env::temp_dir();
        let linear_png = out_dir.join("ochroma_sdf_gather_linear.png");
        let grid_png = out_dir.join("ochroma_sdf_gather_grid.png");
        write_png_rgba(linear_png.to_str().unwrap(), &rgba_linear, w, h);
        write_png_rgba(grid_png.to_str().unwrap(), &rgba_grid, w, h);
        eprintln!("[sdf_gather_grid] wrote {} (linear oracle)", linear_png.display());
        eprintln!("[sdf_gather_grid] wrote {} (cell grid)", grid_png.display());

        // --- Parity: byte-wise max |Δrgb| over the whole RGBA frame. ---------
        assert_eq!(rgba_linear.len(), rgba_grid.len());
        let mut max_d_bytes = 0u8;
        let mut diff_px = 0usize;
        for (i, (a, b)) in rgba_linear.iter().zip(rgba_grid.iter()).enumerate() {
            let d = a.abs_diff(*b);
            if d > 0 && i % 4 != 3 {
                diff_px += 1;
            }
            if d > max_d_bytes {
                max_d_bytes = d;
            }
        }
        let max_drgb = max_d_bytes as f64 / 255.0;

        // --- Gather cost, host-computed from the SAME grid the entry point
        // uploads: rebuild it over the instance's world-space atoms and sum the
        // 3x3x3 neighbourhood counts at 1,000 deterministic surface probes. ---
        let (wmin, wmax) = super::sdf_instance_world_aabb(volume, &instance);
        let q = glam::Quat::from_array(instance.rotation_xyzw).normalize();
        let pos = glam::Vec3::from(instance.position);
        let s = instance.uniform_scale;
        let world_pos: Vec<[f32; 3]> = atoms
            .iter()
            .map(|a| (pos + q * (glam::Vec3::from(a.position) * s)).to_array())
            .collect();
        // Same glass-detect-radius cell floor the entry point applies.
        let glass_detect_r = (volume.voxel_size * instance.uniform_scale * 0.6).max(0.18);
        let grid = super::build_sdf_atom_cell_grid(&world_pos, wmin, wmax, glass_detect_r);
        eprintln!(
            "[sdf_gather_grid] grid: dims={:?} cell={:.3} m cells={} table={} floats",
            grid.dims,
            grid.cell_size,
            grid.cell_table.len(),
            grid.cell_table.len() * 2
        );

        let n_probes = 1000usize;
        let mut seed: u64 = 0x243F_6A88_85A3_08D3; // fixed -> deterministic probes
        let mut total_tested: u64 = 0;
        for _ in 0..n_probes {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = ((seed >> 33) as usize) % n_atoms;
            total_tested += grid.neighbourhood_atom_count(world_pos[idx]) as u64;
        }
        let g_avg = total_tested as f64 / n_probes as f64;
        let speedup = n_atoms as f64 / g_avg;

        // The acceptance line (real measured values, no constants).
        eprintln!(
            "gather: grid={g_avg:.1} atoms/hit (avg) vs linear={n_atoms}, \
             speedup={speedup:.0}x, max|\u{394}rgb|={max_drgb:.4}"
        );
        eprintln!(
            "[sdf_gather_grid] differing RGB bytes: {diff_px} of {}",
            (w * h * 3) as usize
        );

        assert!(
            speedup >= 20.0,
            "cell-grid gather must visit >= 20x fewer atoms than the linear scan \
             (got {speedup:.1}x at {g_avg:.1} atoms/hit vs {n_atoms})"
        );
        assert!(
            max_drgb < 1.0 / 255.0,
            "grid gather must reproduce the linear-gather frame within 1 LSB \
             (max|\u{394}rgb| = {max_drgb:.4}, {diff_px} RGB bytes differ). The grid \
             neighbourhood or cell math is wrong — fix it, don't relax this gate."
        );
    }

    /// Wave-1 Task 2 ACCEPTANCE (CPU-only, no GPU): per-channel material
    /// packing for the textured SDF path. Builds `SdfChannelMaterials` from a
    /// hand-rolled cooked-material list (facade -> material 0 with a real
    /// albedo texture, roof -> material 1, glass -> -1 = keep the M2 BSDF
    /// route) plus the matching `SdfUvParams`, packs both through the SAME
    /// host packers `pathtrace_sdf_scene_textured_to_rgba` uploads with, and
    /// asserts the exact buffer layout and values the megakernel will read:
    /// `g_sdf_channel_material` is 9 floats/instance with the craftsman's
    /// facade slot resolving to a material whose `albedo_tex >= 0` and whose
    /// roughness equals the cooked `roughness_factor`;
    /// `g_sdf_volume_uv_params` is 4 floats/volume carrying the cook's
    /// forge_box_uv constants (w/2, d/2, 1/2.5).
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_channel_material_packing() {
        use super::{
            pack_sdf_channel_material_table, pack_sdf_uv_params, PbrMaterial,
            SdfChannelMaterials, SdfUvParams, SDF_ATOM_CH_FACADE, SDF_ATOM_CH_GLASS,
            SDF_ATOM_CH_ROOF,
        };

        // Hand-rolled cooked-material list (the craftsman shape): facade is
        // material 0 (clapboard, albedo texture 0, cooked roughness 0.6),
        // roof is material 1 (slate, albedo texture 1, cooked roughness 0.6),
        // glass stays -1 so the kernel keeps the M2 glass BSDF route.
        let cooked_facade_roughness = 0.6f32;
        let cooked_roof_roughness = 0.6f32;
        let materials = [
            PbrMaterial {
                base_color: [0.82, 0.75, 0.60],
                roughness: cooked_facade_roughness,
                albedo_tex: 0,
                roughness_tex: 2,
                ..Default::default()
            },
            PbrMaterial {
                base_color: [0.25, 0.25, 0.28],
                roughness: cooked_roof_roughness,
                albedo_tex: 1,
                roughness_tex: 3,
                ..Default::default()
            },
        ];
        let mut table = SdfChannelMaterials {
            material_for_channel: [-1i32; 9],
        };
        table.material_for_channel[SDF_ATOM_CH_FACADE as usize] = 0;
        table.material_for_channel[SDF_ATOM_CH_ROOF as usize] = 1;
        // Two instances sharing the table (a block of two craftsman).
        let per_instance = [table, table];

        let packed = pack_sdf_channel_material_table(&per_instance);
        eprintln!(
            "[sdf_channel_packing] packed channel table ({} floats / {} instances):",
            packed.len(),
            per_instance.len()
        );
        for (ii, inst) in packed.chunks_exact(9).enumerate() {
            eprintln!("  instance {ii}: {inst:?}");
        }

        // Layout: exactly 9 floats per instance, ids verbatim (as f32).
        assert_eq!(
            packed.len(),
            9 * per_instance.len(),
            "channel-material table must be 9 floats per instance"
        );
        for inst in packed.chunks_exact(9) {
            assert_eq!(inst[SDF_ATOM_CH_FACADE as usize], 0.0, "facade slot -> material 0");
            assert_eq!(inst[SDF_ATOM_CH_ROOF as usize], 1.0, "roof slot -> material 1");
            assert_eq!(inst[SDF_ATOM_CH_GLASS as usize], -1.0, "glass slot stays -1 (M2 BSDF)");
        }

        // The facade slot must resolve to a REAL textured material: the id in
        // the packed table indexes the materials slice, whose entry carries an
        // atlas albedo texture and the cooked roughness_factor.
        let facade_id = packed[SDF_ATOM_CH_FACADE as usize] as usize;
        let facade_mat = &materials[facade_id];
        eprintln!(
            "[sdf_channel_packing] facade slot -> material {facade_id}: albedo_tex={} \
             roughness={} (cooked roughness_factor={cooked_facade_roughness})",
            facade_mat.albedo_tex, facade_mat.roughness
        );
        assert!(
            facade_mat.albedo_tex >= 0,
            "facade material must carry a real atlas albedo texture (albedo_tex >= 0)"
        );
        assert_eq!(
            facade_mat.roughness, cooked_facade_roughness,
            "facade material roughness must equal the cooked roughness_factor"
        );
        let roof_id = packed[9 + SDF_ATOM_CH_ROOF as usize] as usize;
        assert_eq!(
            materials[roof_id].roughness, cooked_roof_roughness,
            "roof material roughness must equal the cooked roughness_factor"
        );

        // UV params: 4 floats per volume, the cook's forge_box_uv constants.
        let uvp = pack_sdf_uv_params(&[SdfUvParams {
            offset_x: 9.5 * 0.5,
            offset_z: 10.5 * 0.5,
            tile_recip: 1.0 / 2.5,
        }]);
        eprintln!("[sdf_channel_packing] packed uv params: {uvp:?}");
        assert_eq!(uvp.len(), 4, "uv params must be 4 floats per volume");
        assert_eq!(uvp[0], 4.75, "offset_x = forge_width * 0.5");
        assert_eq!(uvp[1], 5.25, "offset_z = forge_depth * 0.5");
        assert_eq!(uvp[2], 0.4, "tile_recip = 1 / 2.5 m per texture tile");
        assert_eq!(uvp[3], 0.0, "pad slot must be zero");
    }

    /// Wrap a single SdfVolumeInput in a 1-element slice for the scene API.
    #[cfg(feature = "spectra-native")]
    fn volume_slice(v: &super::SdfVolumeInput) -> Vec<super::SdfVolumeInput> {
        vec![v.clone()]
    }

    /// The cook's forge_box_uv constants for one cooked asset, read from the
    /// payload's forge_description footprint (`width`/`depth` in metres) —
    /// `offset_x = w/2`, `offset_z = d/2`, 1 texture tile per 2.5 m. These are
    /// the SAME constants `game_asset_cook.rs::forge_box_uv` used to sample
    /// the atom colours, so the kernel's box projection lands texel-exact on
    /// the cooked appearance.
    #[cfg(feature = "spectra-native")]
    fn load_forge_uv_params(path: &std::path::Path) -> super::SdfUvParams {
        let bytes =
            std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse atoms.json");
        let fp = &json["forge_description"]["footprint"];
        let width = fp["width"].as_f64().expect("forge footprint width") as f32;
        let depth = fp["depth"].as_f64().expect("forge footprint depth") as f32;
        super::SdfUvParams {
            offset_x: width * 0.5,
            offset_z: depth * 0.5,
            tile_recip: 1.0 / 2.5,
        }
    }

    /// Shared wave-1 craftsman harness: the cooked SDF + atoms, the M2 glass
    /// test's placement (base on y=0, centred at the origin) and its exact
    /// camera/rig, so the parity gate, the textured quality gates and the M2
    /// glass test all measure the SAME frame.
    #[cfg(feature = "spectra-native")]
    struct CraftsmanHarness {
        asset_path: std::path::PathBuf,
        volume: super::SdfVolumeInput,
        atoms: Vec<super::SdfSceneAtom>,
        raw_glass: usize,
        instance: super::SdfSceneInstance,
        eye: [f32; 3],
        center: [f32; 3],
        fov_y: f32,
        w: u32,
        h: u32,
        rig: super::LightRig,
    }

    #[cfg(feature = "spectra-native")]
    fn craftsman_harness() -> CraftsmanHarness {
        use super::{LightRig, SdfSceneInstance};

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

        let volume = load_atoms_sdf(&asset_path);
        let (atoms, raw_glass) = load_atoms_material(&asset_path, false);

        // Place the single building so its base sits on y=0, centred at origin
        // (identical to the M2 glass test).
        let ground_y = -volume.origin[1];
        let instance = SdfSceneInstance {
            volume_index: 0,
            position: [
                -(volume.origin[0]
                    + (volume.resolution[0] - 1) as f32 * volume.voxel_size * 0.5),
                ground_y,
                -(volume.origin[2]
                    + (volume.resolution[2] - 1) as f32 * volume.voxel_size * 0.5),
            ],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            uniform_scale: 1.0,
            albedo: [0.7, 0.7, 0.7],
        };

        // The M2 front-facade camera + bright-sky rig.
        let rig = LightRig {
            sun_dir: [0.3, 0.6, 0.7],
            sun_intensity: 2.2,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 1.0,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };

        CraftsmanHarness {
            asset_path,
            volume,
            atoms,
            raw_glass,
            instance,
            eye: [3.0, 6.0, 22.0],
            center: [0.0, 4.5, 0.0],
            fov_y: std::f32::consts::FRAC_PI_4,
            w: 320,
            h: 320,
            rig,
        }
    }

    // --- Test-local copy of the game's TextureCache resolver pattern --------
    // (civitas_care/src/asset/textures.rs — the engine cannot depend on the
    // game crate, so the quality test replicates the exact load semantics:
    // logical `polyhaven://<stem>_<kind>` URIs resolve through the pinned
    // stem->set table; diffuse texels are sRGB-decoded to linear and
    // mean-normalized toward the cooked base_color_factor tint; normal and
    // roughness maps load raw; everything box-downsamples to <=256 (diffuse/
    // rough) / <=128 (normal) in linear space.)

    /// Logical URI stem -> on-disk PolyHaven set (the game's pinned table).
    #[cfg(feature = "spectra-native")]
    const POLYHAVEN_STEM_SETS: &[(&str, &str)] = &[
        ("clapboard", "brown_planks_05"),
        ("painted_trim_primary", "beige_wall_001"),
        ("painted_trim_shadow", "beige_wall_001"),
        ("wood_door", "brown_planks_03"),
        ("slate_roof", "roof_slates_02"),
        ("stucco", "concrete_wall_003"),
        ("brick", "brick_wall_001"),
        ("painted_siding", "blue_painted_planks"),
        ("painted_siding_worn", "distressed_painted_planks"),
        ("roof_tiles", "clay_roof_tiles"),
        ("roof_shingles", "red_slate_roof_tiles_01"),
        ("metal_roof", "corrugated_iron_02"),
        ("asphalt", "asphalt_02"),
        ("lawn", "leafy_grass"),
        ("concrete_pavers", "concrete_pavers"),
        ("brick_common", "brick_wall_001"),
        ("brick_painted", "painted_worn_brick"),
        ("plaster", "painted_plaster_wall"),
    ];

    #[cfg(feature = "spectra-native")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TexKind {
        Diffuse,
        Normal,
        Roughness,
    }

    /// Resolve a cooked texture URI to (on-disk set, kind). Handles both the
    /// logical `polyhaven://<stem>_<kind>` form and the direct relative
    /// `textures/polyhaven/<set>/1k/<kind>.jpg` form the cook also emits.
    #[cfg(feature = "spectra-native")]
    fn resolve_cooked_texture_uri(uri: &str) -> Option<(String, TexKind)> {
        if let Some(rest) = uri.strip_prefix("polyhaven://") {
            let (stem, kind) = if let Some(s) = rest.strip_suffix("_diff") {
                (s, TexKind::Diffuse)
            } else if let Some(s) = rest.strip_suffix("_nor") {
                (s, TexKind::Normal)
            } else if let Some(s) = rest.strip_suffix("_rough") {
                (s, TexKind::Roughness)
            } else {
                return None;
            };
            let set = POLYHAVEN_STEM_SETS
                .iter()
                .find(|(s, _)| *s == stem)
                .map(|(_, set)| *set)?;
            return Some((set.to_string(), kind));
        }
        // Direct path: textures/polyhaven/<set>/1k/{diffuse,normal,roughness}.jpg
        let rest = uri.strip_prefix("textures/polyhaven/")?;
        let mut parts = rest.split('/');
        let set = parts.next()?;
        let _res = parts.next()?; // "1k"
        let kind = match parts.next()? {
            "diffuse.jpg" => TexKind::Diffuse,
            "normal.jpg" => TexKind::Normal,
            "roughness.jpg" => TexKind::Roughness,
            _ => return None,
        };
        Some((set.to_string(), kind))
    }

    /// Exact piecewise sRGB EOTF (encoded -> linear) — same as textures.rs.
    #[cfg(feature = "spectra-native")]
    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Integer box-filter reduction to `<= max_size` per side (linear space).
    #[cfg(feature = "spectra-native")]
    fn box_downsample(
        data: Vec<f32>,
        width: u32,
        height: u32,
        channels: u32,
        max_size: u32,
    ) -> (Vec<f32>, u32, u32) {
        let factor = (width.max(height)).div_ceil(max_size).max(1);
        if factor <= 1 {
            return (data, width, height);
        }
        let (ow, oh) = (width / factor, height / factor);
        let c = channels as usize;
        let mut out = vec![0.0f32; (ow * oh) as usize * c];
        let inv = 1.0 / (factor * factor) as f32;
        for oy in 0..oh {
            for ox in 0..ow {
                let base = ((oy * ow + ox) as usize) * c;
                for sy in 0..factor {
                    for sx in 0..factor {
                        let src =
                            (((oy * factor + sy) * width + ox * factor + sx) as usize) * c;
                        for ch in 0..c {
                            out[base + ch] += data[src + ch];
                        }
                    }
                }
                for ch in 0..c {
                    out[base + ch] *= inv;
                }
            }
        }
        (out, ow, oh)
    }

    /// Load one PolyHaven map exactly like the game's TextureCache: diffuse is
    /// sRGB->linear, mean-normalized toward `tint` (`clamp(tint/mean, 0.25,
    /// 4.0)` per channel, the ALL-P1-1 no-double-darkening rule), <=256;
    /// normal raw 3ch <=128; roughness raw 1ch <=256.
    #[cfg(feature = "spectra-native")]
    fn load_polyhaven_map(
        polyhaven_root: &std::path::Path,
        set: &str,
        kind: TexKind,
        tint: [f32; 3],
    ) -> super::TextureImage {
        let file = match kind {
            TexKind::Diffuse => "diffuse.jpg",
            TexKind::Normal => "normal.jpg",
            TexKind::Roughness => "roughness.jpg",
        };
        let path = polyhaven_root.join(set).join("1k").join(file);
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
            .to_rgb8();
        let (width, height) = img.dimensions();
        let (channels, data, max_size): (u32, Vec<f32>, u32) = match kind {
            TexKind::Diffuse => {
                let linear: Vec<f32> = img
                    .pixels()
                    .flat_map(|p| [0, 1, 2].map(|c| srgb_to_linear(p.0[c] as f32 / 255.0)))
                    .collect();
                // Set mean (untinted) -> per-channel normalization factor.
                let mut mean = [0.0f64; 3];
                for texel in linear.chunks_exact(3) {
                    for (m, &v) in mean.iter_mut().zip(texel) {
                        *m += v as f64;
                    }
                }
                let n = (linear.len() / 3).max(1) as f64;
                let factor = [0, 1, 2].map(|c| {
                    (tint[c] / ((mean[c] / n) as f32).max(1e-4)).clamp(0.25, 4.0)
                });
                (
                    3,
                    linear
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (v * factor[i % 3]).clamp(0.0, 1.0))
                        .collect(),
                    256,
                )
            }
            TexKind::Normal => (
                3,
                img.pixels()
                    .flat_map(|p| [0, 1, 2].map(|c| p.0[c] as f32 / 255.0))
                    .collect(),
                128,
            ),
            TexKind::Roughness => (
                1,
                img.pixels().map(|p| p.0[0] as f32 / 255.0).collect(),
                256,
            ),
        };
        let (data, width, height) = box_downsample(data, width, height, channels, max_size);
        super::TextureImage {
            width,
            height,
            channels,
            data,
        }
    }

    /// Parse the cooked payload's per-channel `ReadyAssetPbrMaterial` list
    /// (serde, the SAME atoms.json the M2 test reads) and build the engine
    /// inputs: `PbrMaterial`s + deduped `TextureImage`s through the resolver
    /// above, plus the per-instance channel->material table. Glass keeps -1 so
    /// windows stay on the M2 glass BSDF route.
    #[cfg(feature = "spectra-native")]
    #[allow(clippy::type_complexity)]
    fn load_craftsman_pbr(
        asset_path: &std::path::Path,
    ) -> (
        Vec<super::PbrMaterial>,
        Vec<super::TextureImage>,
        super::SdfChannelMaterials,
    ) {
        #[derive(serde::Deserialize, Default)]
        struct CookedTextureSet {
            #[serde(default)]
            base_color: Option<String>,
            #[serde(default)]
            normal: Option<String>,
            #[serde(default)]
            roughness: Option<String>,
        }
        #[derive(serde::Deserialize)]
        struct CookedPbrMaterial {
            id: String,
            channel: String,
            base_color_factor: [f32; 4],
            metallic_factor: f32,
            roughness_factor: f32,
            #[serde(default)]
            textures: CookedTextureSet,
        }

        let bytes = std::fs::read(asset_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", asset_path.display()));
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse atoms.json");
        let cooked: Vec<CookedPbrMaterial> =
            serde_json::from_value(json["materials"].clone())
                .expect("parse cooked ReadyAssetPbrMaterial list");
        assert!(
            !cooked.is_empty(),
            "cooked payload carries no per-channel PBR materials"
        );

        // textures/polyhaven root sits beside the atoms dir in the pack.
        let polyhaven_root = asset_path
            .parent()
            .and_then(|p| p.parent())
            .expect("atoms dir parent")
            .join("textures/polyhaven");

        let mut materials: Vec<super::PbrMaterial> = Vec::new();
        let mut textures: Vec<super::TextureImage> = Vec::new();
        // Dedupe by (set, kind, quantized tint) like the game's cache.
        let mut tex_index: std::collections::HashMap<(String, TexKind, [u32; 3]), i32> =
            std::collections::HashMap::new();
        let mut table = super::SdfChannelMaterials {
            material_for_channel: [-1i32; 9],
        };

        for cm in &cooked {
            let slot = super::sdf_atom_channel_from_str(&cm.channel) as usize;
            if slot == super::SDF_ATOM_CH_GLASS as usize {
                continue; // glass stays -1: the M2 BSDF route, never textured
            }
            let tint = [
                cm.base_color_factor[0],
                cm.base_color_factor[1],
                cm.base_color_factor[2],
            ];
            let mut resolve = |uri: &Option<String>, expect: TexKind| -> i32 {
                let Some(uri) = uri.as_deref() else { return -1 };
                let Some((set, kind)) = resolve_cooked_texture_uri(uri) else {
                    eprintln!("[sdf_textured] unmapped texture uri '{uri}' (flat colour)");
                    return -1;
                };
                assert_eq!(kind, expect, "uri '{uri}' resolved to the wrong map kind");
                // Tint only keys diffuse entries (normal/rough are tint-free).
                let tkey = match kind {
                    TexKind::Diffuse => [
                        (tint[0].max(0.0) * 1024.0).round() as u32,
                        (tint[1].max(0.0) * 1024.0).round() as u32,
                        (tint[2].max(0.0) * 1024.0).round() as u32,
                    ],
                    _ => [0; 3],
                };
                let key = (set.clone(), kind, tkey);
                if let Some(&idx) = tex_index.get(&key) {
                    return idx;
                }
                let img = load_polyhaven_map(&polyhaven_root, &set, kind, tint);
                let idx = textures.len() as i32;
                textures.push(img);
                tex_index.insert(key, idx);
                idx
            };
            let albedo_tex = resolve(&cm.textures.base_color, TexKind::Diffuse);
            let roughness_tex = resolve(&cm.textures.roughness, TexKind::Roughness);
            let normal_tex = resolve(&cm.textures.normal, TexKind::Normal);

            let mat_id = materials.len() as i32;
            materials.push(super::PbrMaterial {
                base_color: tint,
                roughness: cm.roughness_factor,
                metallic: cm.metallic_factor,
                emission_strength: 0.0,
                albedo_tex,
                roughness_tex,
                normal_tex,
                uv_scale: [1.0, 1.0],
                // SDF path: glass stays opaque here — its windows route
                // through the dedicated SDF glass branch in the megakernel,
                // not through MAT_GLASS mesh materials.
                ..super::PbrMaterial::default()
            });
            if table.material_for_channel[slot] < 0 {
                table.material_for_channel[slot] = mat_id;
            }
            eprintln!(
                "[sdf_textured] cooked material '{}' channel '{}' -> slot {slot} mat {mat_id} \
                 (albedo_tex={albedo_tex} rough_tex={roughness_tex} normal_tex={normal_tex} \
                 roughness={})",
                cm.id, cm.channel, cm.roughness_factor
            );
        }
        (materials, textures, table)
    }

    /// Wave-1 Task 3 ACCEPTANCE — the M1 textured-building gate. Renders the
    /// cooked craftsman through the textured SDF path (per-channel PolyHaven
    /// PBR materials box-projected at the hit with the cook's forge_box_uv
    /// convention) and through the M2 flat path (same camera), writes both
    /// PNGs, and measures three gates over the front facade:
    ///   (a) DETAIL: mean |∇luminance| over the facade mask, textured vs
    ///       flat-blend — texture must add >= 3.0x the gradient energy the
    ///       k-NN atom blend has (clapboard courses actually land on the wall);
    ///   (b) CONSISTENCY: mean |textured - flat| RGB over >= 2,000 facade px
    ///       < 0.12 — the texture layer sits ON the cooked appearance (same
    ///       box projection + tint normalization the cook used), it does not
    ///       repaint the building;
    ///   (c) PER-CHANNEL ROUGHNESS: the roughness AOV (returned in alpha via
    ///       aov_roughness) differs between the slate porch roof and the
    ///       clapboard wall by > 0.05 — real per-channel PBR dispatch, not one
    ///       material everywhere.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_craftsman_textured_quality -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_craftsman_textured_quality() {
        use super::{
            pathtrace_sdf_scene_textured_to_rgba, pathtrace_sdf_scene_with_atoms_to_rgba,
        };

        let hz = craftsman_harness();
        let (materials, textures, channel_mats) = load_craftsman_pbr(&hz.asset_path);
        let uv_params = load_forge_uv_params(&hz.asset_path);
        eprintln!(
            "[sdf_textured] craftsman: atoms={} (glass={}) materials={} textures={} \
             uv_params=(w/2={}, d/2={}, 1/tile={})",
            hz.atoms.len(),
            hz.raw_glass,
            materials.len(),
            textures.len(),
            uv_params.offset_x,
            uv_params.offset_z,
            uv_params.tile_recip
        );
        assert!(
            channel_mats.material_for_channel[super::SDF_ATOM_CH_FACADE as usize] >= 0,
            "cooked payload must map the facade channel to a textured material"
        );
        assert!(
            channel_mats.material_for_channel[super::SDF_ATOM_CH_ROOF as usize] >= 0,
            "cooked payload must map the roof channel to a textured material"
        );

        let (w, h) = (hz.w, hz.h);

        // Textured render; alpha carries the primary-hit roughness AOV.
        let t0 = std::time::Instant::now();
        let rgba_tex = pathtrace_sdf_scene_textured_to_rgba(
            &volume_slice(&hz.volume),
            &[uv_params],
            &[hz.instance],
            &[hz.atoms.clone()],
            &[Vec::new()],
            &[channel_mats],
            &materials,
            &textures,
            hz.eye,
            hz.center,
            hz.fov_y,
            w,
            h,
            6,
            0,
            &hz.rig,
            true,
        )
        .expect("textured SDF render should succeed");
        let secs_tex = t0.elapsed().as_secs_f64();

        // Flat control: the M2 path (k-NN atom blend), same camera + rig.
        let rgba_flat = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&hz.volume),
            &[hz.instance],
            &[hz.atoms.clone()],
            hz.eye,
            hz.center,
            hz.fov_y,
            w,
            h,
            6,
            &hz.rig,
        )
        .expect("flat-control (M2) render should succeed");

        // PNGs for the human eyeball line (textured alpha holds the roughness
        // AOV — force it opaque for viewing).
        let out_dir = std::env::temp_dir();
        let tex_png = out_dir.join("sdf_craftsman_textured.png");
        let flat_png = out_dir.join("sdf_craftsman_flat_control.png");
        let mut rgba_tex_view = rgba_tex.clone();
        for px in rgba_tex_view.chunks_exact_mut(4) {
            px[3] = 255;
        }
        write_png_rgba(tex_png.to_str().unwrap(), &rgba_tex_view, w, h);
        write_png_rgba(flat_png.to_str().unwrap(), &rgba_flat, w, h);
        eprintln!("[sdf_textured] wrote {} (textured)", tex_png.display());
        eprintln!("[sdf_textured] wrote {} (M2 flat control)", flat_png.display());

        let idx = |x: u32, y: u32| -> usize { ((y * w + x) * 4) as usize };
        let luma =
            |p: &[u8]| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32)
                / 255.0;

        // --- Facade mask, classified on the FLAT control so the same pixels
        // are compared in both renders: inside the front-wall super-rect,
        // bright-but-not-blown, and WARM (r >= b rejects the slate porch roof,
        // the blue sky and the cyan glass panes). -----------------------------
        let (rx0, rx1, ry0, ry1) = (55u32, 262u32, 112u32, 232u32);
        let mut mask = vec![false; (w * h) as usize];
        let mut facade_px = 0usize;
        for y in ry0..ry1 {
            for x in rx0..rx1 {
                let p = &rgba_flat[idx(x, y)..idx(x, y) + 4];
                let l = luma(p);
                if l >= 0.45 && l <= 0.97 && p[0] >= p[2] {
                    mask[(y * w + x) as usize] = true;
                    facade_px += 1;
                }
            }
        }
        eprintln!("[sdf_textured] facade-classified pixels: {facade_px}");
        assert!(
            facade_px >= 2000,
            "facade mask too small ({facade_px} px < 2000) — camera/mask drifted"
        );

        // --- Gate (a): facade detail energy (mean |∇luminance| over in-mask
        // neighbour pairs — window/roof edges never enter the metric). --------
        let grad_energy = |rgba: &[u8]| -> f64 {
            let mut sum = 0.0f64;
            let mut pairs = 0usize;
            for y in ry0..ry1 {
                for x in rx0..rx1 {
                    if !mask[(y * w + x) as usize] {
                        continue;
                    }
                    let l = luma(&rgba[idx(x, y)..idx(x, y) + 4]);
                    if x + 1 < rx1 && mask[(y * w + x + 1) as usize] {
                        let r = luma(&rgba[idx(x + 1, y)..idx(x + 1, y) + 4]);
                        sum += (l - r).abs() as f64;
                        pairs += 1;
                    }
                    if y + 1 < ry1 && mask[((y + 1) * w + x) as usize] {
                        let d = luma(&rgba[idx(x, y + 1)..idx(x, y + 1) + 4]);
                        sum += (l - d).abs() as f64;
                        pairs += 1;
                    }
                }
            }
            sum / pairs.max(1) as f64
        };
        let energy_tex = grad_energy(&rgba_tex);
        let energy_flat = grad_energy(&rgba_flat);
        let energy_ratio = energy_tex / energy_flat.max(1e-9);
        let pass_a = energy_ratio >= 3.0;
        eprintln!(
            "M1 texture: facade detail energy {energy_ratio:.2}x flat-blend (gate >= 3.0x) \
             -> {} (textured {energy_tex:.5}, flat {energy_flat:.5})",
            if pass_a { "PASS" } else { "FAIL" }
        );

        // --- Gate (b): consistency — the texture layer must sit ON the cooked
        // appearance (same box projection + tint normalization the cook used
        // to colour the atoms), not repaint the building. ---------------------
        let mut dev_sum = 0.0f64;
        for y in ry0..ry1 {
            for x in rx0..rx1 {
                if !mask[(y * w + x) as usize] {
                    continue;
                }
                let t = &rgba_tex[idx(x, y)..idx(x, y) + 4];
                let f = &rgba_flat[idx(x, y)..idx(x, y) + 4];
                let d = (t[0] as f32 - f[0] as f32).abs()
                    + (t[1] as f32 - f[1] as f32).abs()
                    + (t[2] as f32 - f[2] as f32).abs();
                dev_sum += (d / (3.0 * 255.0)) as f64;
            }
        }
        let mean_dev = dev_sum / facade_px as f64;
        let pass_b = mean_dev < 0.12;
        eprintln!(
            "M1 consistency: mean |hit_rgb - atom_gather_rgb| = {mean_dev:.4} (gate < 0.12) \
             -> {} ({facade_px} facade px)",
            if pass_b { "PASS" } else { "FAIL" }
        );

        // --- Gate (c): per-channel roughness from the AOV (alpha channel of
        // the textured render). Roof sample = SLATE-classified pixels (cool
        // blue-grey, mid luma in the flat control) inside the porch-roof rect
        // — the gable end facing the camera is wall, the porch roof is the
        // visible slate face; facade sample = warm-wall pixels on the clean
        // clapboard right wall, off the silhouette edge. Classifying on the
        // flat control keeps both samples identical regardless of what the
        // textured shade draws.
        let mean_alpha_classified =
            |x0: u32, x1: u32, y0: u32, y1: u32, want_slate: bool| -> (f64, usize) {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = &rgba_flat[idx(x, y)..idx(x, y) + 4];
                        let l = luma(p);
                        let is_slate = p[2] >= p[0] && l > 0.25 && l < 0.75;
                        let is_wall = l >= 0.45 && l <= 0.97 && p[0] >= p[2];
                        if (want_slate && is_slate) || (!want_slate && is_wall) {
                            sum += rgba_tex[idx(x, y) + 3] as f64 / 255.0;
                            count += 1;
                        }
                    }
                }
                (sum / count.max(1) as f64, count)
            };
        // Porch-roof rect (the slate face below the upper windows) and the
        // window-free right-wall strip (x < 254 stays off the sky edge).
        let (rough_roof, n_roof) = mean_alpha_classified(100, 200, 152, 185, true);
        let (rough_wall, n_wall) = mean_alpha_classified(242, 254, 152, 212, false);
        assert!(
            n_roof >= 150,
            "too few slate-classified porch-roof pixels ({n_roof}) — crop/classifier drifted"
        );
        assert!(
            n_wall >= 300,
            "too few wall-classified right-wall pixels ({n_wall}) — crop/classifier drifted"
        );
        let rough_delta = (rough_roof - rough_wall).abs();
        let pass_c = rough_delta > 0.05;
        eprintln!(
            "roughness roof={rough_roof:.3} facade={rough_wall:.3} (|delta|={rough_delta:.3}, \
             gate > 0.05, {n_roof}/{n_wall} px) -> {}",
            if pass_c { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "[sdf_textured] eyeball: clapboard courses + slate porch roof + real roughness \
             split — verify {} against {} ({secs_tex:.2}s textured frame)",
            tex_png.display(),
            flat_png.display()
        );

        // --- The gates. Fix the UV/material binding, never weaken these. -----
        assert!(
            pass_a,
            "texture detail is not landing on the facade (energy ratio {energy_ratio:.2}x \
             < 3.0x; textured {energy_tex:.5} vs flat {energy_flat:.5}). The box-UV \
             projection or the channel->material binding is wrong."
        );
        assert!(
            pass_b,
            "textured facade diverges from the cooked atom appearance (mean dev \
             {mean_dev:.4} >= 0.12) — the kernel's box UV does not match the cook's \
             forge_box_uv convention."
        );
        assert!(
            pass_c,
            "roof and facade roughness do not differ (roof {rough_roof:.3} vs facade \
             {rough_wall:.3}) — per-channel material dispatch is not landing."
        );
    }

    // ===== Mesh M0: the cooked craftsman as a REAL textured, lit triangle =====
    // ===== mesh through the proven single-mesh path tracer ====================

    /// The cooked craftsman's triangle mesh exactly as the pack carries it
    /// (`ReadyAssetPayload.mesh`): per-vertex positions/normals/uvs (the
    /// cook's box-projected world-scale UVs, multiply convention already
    /// applied — used as-is with `uv_scale = [1,1]`, textures wrap) and
    /// PER-TRIANGLE material ids indexing the payload's cooked `materials`
    /// list in cooked order.
    #[cfg(feature = "spectra-native")]
    struct CookedMesh {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<[u32; 3]>,
        material_ids: Vec<u8>,
    }

    /// Load `ReadyAssetPayload.mesh` from a cooked atoms.json — the SAME pack
    /// file the SDF craftsman tests read; the triangle mesh lives beside the
    /// SDF/atoms in that payload. Validates shape invariants (parallel vertex
    /// arrays, per-triangle material ids, in-range indices) so a drifted cook
    /// fails loudly here, not as GPU garbage.
    #[cfg(feature = "spectra-native")]
    fn load_craftsman_mesh(asset_path: &std::path::Path) -> CookedMesh {
        let bytes = std::fs::read(asset_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", asset_path.display()));
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse atoms.json");
        let mesh = &json["mesh"];
        assert!(
            mesh.is_object(),
            "cooked payload {} carries no `mesh` object",
            asset_path.display()
        );
        let f32s_n = |key: &str, n: usize| -> Vec<Vec<f32>> {
            mesh[key]
                .as_array()
                .unwrap_or_else(|| panic!("mesh.{key} missing"))
                .iter()
                .map(|v| {
                    let a = v.as_array().unwrap_or_else(|| panic!("mesh.{key} row"));
                    assert_eq!(a.len(), n, "mesh.{key} row arity");
                    a.iter().map(|x| x.as_f64().unwrap() as f32).collect()
                })
                .collect()
        };
        let positions: Vec<[f32; 3]> = f32s_n("positions", 3)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let normals: Vec<[f32; 3]> = f32s_n("normals", 3)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let uvs: Vec<[f32; 2]> = f32s_n("uvs", 2)
            .into_iter()
            .map(|v| [v[0], v[1]])
            .collect();
        let indices: Vec<[u32; 3]> = mesh["indices"]
            .as_array()
            .expect("mesh.indices")
            .iter()
            .map(|v| {
                let a = v.as_array().expect("triangle");
                assert_eq!(a.len(), 3, "triangle arity");
                [
                    a[0].as_u64().unwrap() as u32,
                    a[1].as_u64().unwrap() as u32,
                    a[2].as_u64().unwrap() as u32,
                ]
            })
            .collect();
        let material_ids: Vec<u8> = mesh["material_ids"]
            .as_array()
            .expect("mesh.material_ids")
            .iter()
            .map(|v| {
                let m = v.as_u64().expect("material id");
                assert!(m < 256, "material id {m} exceeds u8");
                m as u8
            })
            .collect();
        assert_eq!(positions.len(), normals.len(), "positions/normals parallel");
        assert_eq!(positions.len(), uvs.len(), "positions/uvs parallel");
        assert_eq!(
            indices.len(),
            material_ids.len(),
            "material_ids must be PER-TRIANGLE"
        );
        assert!(
            indices
                .iter()
                .all(|t| t.iter().all(|&i| (i as usize) < positions.len())),
            "triangle index out of range"
        );
        CookedMesh {
            positions,
            normals,
            uvs,
            indices,
            material_ids,
        }
    }

    /// Shared cooked-material loader: parse the payload's
    /// `ReadyAssetPbrMaterial` list IN COOKED ORDER, loading every referenced
    /// PolyHaven map through the game's TextureCache pattern above (diffuse
    /// sRGB→linear + tint-normalized, normal/roughness raw, box-downsampled,
    /// deduped). HONORS the cooked `transmission`/`ior`/`thin_walled` fields
    /// (the cook sets them on curtain-wall vision glass; absent fields default
    /// to the historical opaque values). Returns
    /// (materials, textures, per-material channel names).
    #[cfg(feature = "spectra-native")]
    fn load_cooked_pbr_materials(
        asset_path: &std::path::Path,
    ) -> (
        Vec<super::PbrMaterial>,
        Vec<super::TextureImage>,
        Vec<String>,
    ) {
        #[derive(serde::Deserialize, Default)]
        struct CookedTextureSet {
            #[serde(default)]
            base_color: Option<String>,
            #[serde(default)]
            normal: Option<String>,
            #[serde(default)]
            roughness: Option<String>,
        }
        fn default_cooked_ior() -> f32 {
            1.5
        }
        #[derive(serde::Deserialize)]
        struct CookedPbrMaterial {
            id: String,
            channel: String,
            base_color_factor: [f32; 4],
            metallic_factor: f32,
            roughness_factor: f32,
            #[serde(default)]
            textures: CookedTextureSet,
            /// > 0 = the cook tagged this material REAL transmissive glass.
            #[serde(default)]
            transmission: f32,
            #[serde(default = "default_cooked_ior")]
            ior: f32,
            #[serde(default)]
            thin_walled: bool,
        }

        let bytes = std::fs::read(asset_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", asset_path.display()));
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse atoms.json");
        let cooked: Vec<CookedPbrMaterial> =
            serde_json::from_value(json["materials"].clone())
                .expect("parse cooked ReadyAssetPbrMaterial list");
        assert!(
            !cooked.is_empty(),
            "cooked payload carries no per-channel PBR materials"
        );

        // textures/polyhaven root sits beside the atoms dir in the pack.
        let polyhaven_root = asset_path
            .parent()
            .and_then(|p| p.parent())
            .expect("atoms dir parent")
            .join("textures/polyhaven");

        let mut materials: Vec<super::PbrMaterial> = Vec::new();
        let mut textures: Vec<super::TextureImage> = Vec::new();
        let mut channels: Vec<String> = Vec::new();
        // Dedupe by (set, kind, quantized tint) like the game's cache.
        let mut tex_index: std::collections::HashMap<(String, TexKind, [u32; 3]), i32> =
            std::collections::HashMap::new();

        for (i, cm) in cooked.iter().enumerate() {
            let tint = [
                cm.base_color_factor[0],
                cm.base_color_factor[1],
                cm.base_color_factor[2],
            ];
            let mut resolve = |uri: &Option<String>, expect: TexKind| -> i32 {
                let Some(uri) = uri.as_deref() else { return -1 };
                let Some((set, kind)) = resolve_cooked_texture_uri(uri) else {
                    eprintln!("[mesh_m0] unmapped texture uri '{uri}' (flat colour)");
                    return -1;
                };
                assert_eq!(kind, expect, "uri '{uri}' resolved to the wrong map kind");
                let tkey = match kind {
                    TexKind::Diffuse => [
                        (tint[0].max(0.0) * 1024.0).round() as u32,
                        (tint[1].max(0.0) * 1024.0).round() as u32,
                        (tint[2].max(0.0) * 1024.0).round() as u32,
                    ],
                    _ => [0; 3],
                };
                let key = (set.clone(), kind, tkey);
                if let Some(&idx) = tex_index.get(&key) {
                    return idx;
                }
                let img = load_polyhaven_map(&polyhaven_root, &set, kind, tint);
                let idx = textures.len() as i32;
                textures.push(img);
                tex_index.insert(key, idx);
                idx
            };
            let albedo_tex = resolve(&cm.textures.base_color, TexKind::Diffuse);
            let roughness_tex = resolve(&cm.textures.roughness, TexKind::Roughness);
            let normal_tex = resolve(&cm.textures.normal, TexKind::Normal);

            materials.push(super::PbrMaterial {
                base_color: tint,
                roughness: cm.roughness_factor,
                metallic: cm.metallic_factor,
                emission_strength: 0.0,
                albedo_tex,
                roughness_tex,
                normal_tex,
                uv_scale: [1.0, 1.0],
                // The cooked glass tag travels with the material: curtain-wall
                // vision glass arrives transmissive FROM THE COOK, everything
                // else keeps the historical opaque defaults.
                transmission: cm.transmission,
                ior: cm.ior,
                thin_walled: cm.thin_walled,
            });
            channels.push(cm.channel.clone());
            eprintln!(
                "[mesh_m0] cooked material {i} '{}' channel '{}' \
                 (albedo_tex={albedo_tex} rough_tex={roughness_tex} normal_tex={normal_tex} \
                 roughness={} metallic={} transmission={})",
                cm.id, cm.channel, cm.roughness_factor, cm.metallic_factor, cm.transmission
            );
        }
        (materials, textures, channels)
    }

    /// GENERALIZED cooked-building material loader: a material table indexed
    /// by the FORGE MESH MATERIAL IDS the payload's `mesh.material_ids`
    /// actually carry (forge `facade/extrude.rs`: 0 wall, 1 roof, 2 glass,
    /// 3 reveal, 4 trim, 5 cornice, 6 door), each id bound to the cooked
    /// material of its CHANNEL — the same forge-id → channel collapse the
    /// cook's `forge_material_channel` uses for atoms. Cooked
    /// `transmission`/`ior`/`thin_walled` tags are KEPT, so a curtain-wall
    /// payload's vision glass (forge id 2 → the cooked "glass" channel
    /// material) renders transmissive exactly as cooked. Returns
    /// (materials[forge_id], textures, channel name per forge_id).
    #[cfg(feature = "spectra-native")]
    fn load_building_mesh_pbr_by_forge_id(
        asset_path: &std::path::Path,
    ) -> (
        Vec<super::PbrMaterial>,
        Vec<super::TextureImage>,
        Vec<String>,
    ) {
        let (materials, textures, channels) = load_cooked_pbr_materials(asset_path);
        let by_channel = |name: &str| -> super::PbrMaterial {
            let idx = channels
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("cooked payload has no '{name}' channel material"));
            materials[idx]
        };
        // Forge mesh material id -> cooked channel (the cook's
        // forge_material_channel mapping: reveal/trim/cornice all collapse
        // onto the trim channel).
        let forge_channels =
            ["facade", "roof", "glass", "trim", "trim", "trim", "door", "glass_lit"];
        let table: Vec<super::PbrMaterial> = forge_channels
            .iter()
            .map(|ch| {
                if *ch == "glass_lit" {
                    // Forge id 7 (MAT_GLASS_LIT): the cooked payload carries
                    // one glass material; lit panes are the glass clone with
                    // a warm interior glow so a deterministic subset of
                    // windows reads inhabited instead of dead.
                    let mut lit = by_channel("glass");
                    lit.base_color = [1.0, 0.82, 0.58];
                    lit.emission_strength = 1.6;
                    lit.transmission = 0.0;
                    lit
                } else {
                    by_channel(ch)
                }
            })
            .collect();
        (
            table,
            textures,
            forge_channels.iter().map(|ch| ch.to_string()).collect(),
        )
    }

    /// CPU-side depth-buffered material-id rasterization of the cooked mesh
    /// with the EXACT pinhole the Spectra camera kernel uses
    /// (`camera.slang`: `ndc_x = (2(x+0.5)/w − 1)·aspect`,
    /// `ndc_y = 1 − 2(y+0.5)/h`, `ray = fwd + ndc·tanHalf·basis`), so the
    /// per-material pixel masks line up with the GPU render — the gates then
    /// measure exactly the wall/roof/trim pixels, no hand-tuned crops. The
    /// test cross-checks this projection against the GPU image (coverage
    /// agreement) before any gate uses it. Returns per pixel: (cooked
    /// material id, triangle index) of the nearest triangle, or (−1, −1)
    /// where nothing covers the pixel. The triangle index lets the
    /// detail-energy gate restrict itself to SAME-FACE pixel pairs — the
    /// forge cook models each clapboard course as real geometry, so
    /// face-boundary shading steps would otherwise dominate the flat
    /// control's gradient energy and hide the texture signal the gate
    /// measures. Depth is perspective-correct (interpolated 1/z).
    #[cfg(feature = "spectra-native")]
    fn rasterize_material_masks(
        mesh: &CookedMesh,
        eye: [f32; 3],
        target: [f32; 3],
        fov_y: f32,
        w: u32,
        h: u32,
    ) -> (Vec<i32>, Vec<i32>) {
        use glam::Vec3;
        let eye_v = Vec3::from(eye);
        let fwd = (Vec3::from(target) - eye_v).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = (fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        // World point -> (pixel x, pixel y, camera-space depth). Integer pixel
        // coordinates are pixel CENTERS (the -0.5 below mirrors the kernel's
        // gx + 0.5 sampling).
        let project = |p: Vec3| -> Option<(f32, f32, f32)> {
            let d = p - eye_v;
            let cz = d.dot(fwd);
            if cz <= 1e-3 {
                return None;
            }
            let ndc_x = d.dot(right) / (cz * tan_half);
            let ndc_y = d.dot(up) / (cz * tan_half);
            let sx = (ndc_x / aspect + 1.0) * 0.5 * w as f32 - 0.5;
            let sy = (1.0 - ndc_y) * 0.5 * h as f32 - 0.5;
            Some((sx, sy, cz))
        };
        let edge = |a: (f32, f32), b: (f32, f32), px: f32, py: f32| -> f32 {
            (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
        };
        let mut mat = vec![-1i32; (w * h) as usize];
        let mut tri_id = vec![-1i32; (w * h) as usize];
        let mut depth = vec![f32::INFINITY; (w * h) as usize];
        for (t_idx, (tri, &mid)) in mesh.indices.iter().zip(&mesh.material_ids).enumerate() {
            let p0 = Vec3::from(mesh.positions[tri[0] as usize]);
            let p1 = Vec3::from(mesh.positions[tri[1] as usize]);
            let p2 = Vec3::from(mesh.positions[tri[2] as usize]);
            let (Some(a), Some(b), Some(c)) = (project(p0), project(p1), project(p2))
            else {
                continue; // behind the camera — the building never is
            };
            let area = edge((a.0, a.1), (b.0, b.1), c.0, c.1);
            if area.abs() < 1e-6 {
                continue; // degenerate in screen space
            }
            let x0 = a.0.min(b.0).min(c.0).floor().max(0.0) as u32;
            let x1 = (a.0.max(b.0).max(c.0).ceil() as i64).clamp(0, w as i64 - 1) as u32;
            let y0 = a.1.min(b.1).min(c.1).floor().max(0.0) as u32;
            let y1 = (a.1.max(b.1).max(c.1).ceil() as i64).clamp(0, h as i64 - 1) as u32;
            if x0 > x1 || y0 > y1 {
                continue;
            }
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32, y as f32);
                    let wa = edge((b.0, b.1), (c.0, c.1), px, py) / area;
                    let wb = edge((c.0, c.1), (a.0, a.1), px, py) / area;
                    let wc = edge((a.0, a.1), (b.0, b.1), px, py) / area;
                    if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                        continue; // outside (normalizing by signed area makes
                                  // inside-test winding-independent)
                    }
                    let inv_z = wa / a.2 + wb / b.2 + wc / c.2;
                    if inv_z <= 0.0 {
                        continue;
                    }
                    let z = 1.0 / inv_z;
                    let idx = (y * w + x) as usize;
                    if z < depth[idx] {
                        depth[idx] = z;
                        mat[idx] = mid as i32;
                        tri_id[idx] = t_idx as i32;
                    }
                }
            }
        }
        (mat, tri_id)
    }

    /// Mesh M0 ACCEPTANCE: render ONE cooked building — the craftsman — as a
    /// REAL, textured, lit TRIANGLE MESH through the proven single-mesh path
    /// tracer (`pathtrace_mesh_lit_to_rgba`): the pack's actual mesh
    /// (positions/normals/box-UVs/indices + per-triangle material ids), the
    /// cooked per-channel PolyHaven PBR materials loaded via the TextureCache
    /// pattern, a sun+sky+bounce rig, a 3/4-front inspection camera. Writes
    /// `mesh_craftsman_textured.png` plus a flat control (same camera/rig,
    /// textures stripped to flat albedo) and gates on measured outcomes:
    ///   (a) facade DETAIL ENERGY: the per-pixel RGB texture residual
    ///       (|Δrgb| textured-vs-flat over the CPU-rasterized facade mask)
    ///       >= 2.5x the flat control's matched same-face per-pixel RGB noise
    ///       floor — the texture paints real colour onto the wall, well above
    ///       the render-noise floor;
    ///   (b) per-part materials: wall-vs-roof AND wall-vs-door mean |Δrgb|
    ///       > 0.12, plus a wall-vs-trim split self-calibrated against the
    ///       flat control (the cooked trim albedo is near-wall by design) —
    ///       distinct per-channel materials landed;
    ///   (c) lit coverage >= 0.25 of the frame, after the CPU mask and the GPU
    ///       silhouette are shown to agree (the projection cross-check that
    ///       makes every mask-based gate trustworthy).
    /// MATERIALS: loaded via `load_building_mesh_pbr_by_forge_id` (the single
    /// canonical mesh-material loader) — the table is indexed by the FORGE
    /// channel ids the mesh's `material_ids` actually carry, so the window/
    /// door frames (forge id 4) render the TRIM material, not whatever sat at
    /// cooked slot 4 (historically ground_concrete with a red-brick diffuse —
    /// the "brick on the window frames" zoning bug). The wall-vs-trim /
    /// wall-vs-door thresholds below were RE-BASELINED 2026-06-12 on the
    /// post-zoning-fix render (trim = stucco, door = wood; they were first
    /// measured on the cross-wired render where trim was secretly brick and
    /// door was metal).
    /// GLASS: the craftsman's cooked glass channel is opaque (transmission 0
    /// from the cook), so panes render opaque here and the gates stay stable.
    /// Real mesh transmission is gated in `mesh_glass_transmits_checkerboard`,
    /// which also renders the craftsman glass demo.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib mesh_craftsman_textured_lit -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn mesh_craftsman_textured_lit() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

        // 1+2: the cooked payload's REAL mesh + per-FORGE-ID PBR materials
        // (table index == forge channel id == mesh.material_ids meaning).
        let mesh = load_craftsman_mesh(&asset_path);
        let (materials, textures, channels) = load_building_mesh_pbr_by_forge_id(&asset_path);
        eprintln!(
            "[mesh_m0] cooked mesh: {} verts, {} tris, {} materials, {} textures",
            mesh.positions.len(),
            mesh.indices.len(),
            materials.len(),
            textures.len()
        );
        let max_id = *mesh.material_ids.iter().max().unwrap() as usize;
        assert!(
            max_id < materials.len(),
            "mesh material id {max_id} out of range of {} cooked materials",
            materials.len()
        );
        // The parts the gates measure, located by cooked channel name.
        let chan_id = |name: &str| -> i32 {
            channels
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("cooked payload has no '{name}' channel"))
                as i32
        };
        let (wall_id, roof_id, trim_id, door_id) = (
            chan_id("facade"),
            chan_id("roof"),
            chan_id("trim"),
            chan_id("door"),
        );
        for (label, id) in [("wall", wall_id), ("roof", roof_id), ("trim", trim_id)] {
            let m = &materials[id as usize];
            assert!(
                m.albedo_tex >= 0 && m.roughness_tex >= 0 && m.normal_tex >= 0,
                "{label} material must carry diffuse+roughness+normal maps \
                 (got {} {} {})",
                m.albedo_tex,
                m.roughness_tex,
                m.normal_tex
            );
        }

        // 4: 3/4-front inspection camera (front facade is +Z, windows on +Z;
        // right wall +X) from slightly above so facade, right wall, porch,
        // trim AND both roof planes are all in frame — and a sun+sky+bounce
        // rig: warm sun from high front-right, blue sky dome, soft sky/camera
        // bounce fills so shadowed faces stay readable.
        let (w, h) = (512u32, 512u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        // 3/4-front inspection framing: oblique from the front-right and
        // slightly above so facade, right wall, porch, trim, door AND both roof
        // planes are all in frame, the building filling most of the frame.
        let target = [0.0f32, 4.2, 0.6];
        let eye = [8.4f32, 5.6, 12.4];
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            // Lower dome than the SDF rigs: the full-blue 1.0 dome washes the
            // beige/stucco albedo split into neutral grey.
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        // High spp: gate (a) divides the texture residual by the flat
        // control's render-noise floor, which falls as 1/√spp — at low spp the
        // floor is comparable to the (EWA-softened) broad-face texture and
        // starves the ratio. 256 spp drives the floor well under the texture
        // residual so the ratio reflects texture, not variance.
        let spp = 256u32;

        // 5: the textured render through the EXISTING entry point.
        let t0 = std::time::Instant::now();
        let rgba_tex = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &materials,
            &textures,
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("textured craftsman mesh render should succeed");
        let secs = t0.elapsed().as_secs_f64();

        // 6: FLAT control — same camera/rig/mesh, textures stripped to the
        // cooked flat albedo (the detail-energy denominator).
        let flat_materials: Vec<PbrMaterial> = materials
            .iter()
            .map(|m| PbrMaterial {
                albedo_tex: -1,
                roughness_tex: -1,
                normal_tex: -1,
                ..*m
            })
            .collect();
        let rgba_flat = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &flat_materials,
            &[],
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("flat-control craftsman mesh render should succeed");

        let out_dir = std::env::temp_dir();
        let tex_png = out_dir.join("mesh_craftsman_textured.png");
        let flat_png = out_dir.join("mesh_craftsman_flat_control.png");
        let opaque = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        write_png_rgba(tex_png.to_str().unwrap(), &opaque(rgba_tex.clone()), w, h);
        write_png_rgba(flat_png.to_str().unwrap(), &opaque(rgba_flat.clone()), w, h);

        // --- Per-material pixel masks from the CPU rasterizer (same pinhole
        // as the GPU), eroded one pixel so silhouette/part edges never enter
        // the part metrics. ---------------------------------------------------
        let (mats_px, tris_px) = rasterize_material_masks(&mesh, eye, target, fov_y, w, h);
        let eroded = |id: i32| -> Vec<bool> {
            let mut m = vec![false; (w * h) as usize];
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = (y * w + x) as usize;
                    if mats_px[i] == id
                        && mats_px[i - 1] == id
                        && mats_px[i + 1] == id
                        && mats_px[i - w as usize] == id
                        && mats_px[i + w as usize] == id
                    {
                        m[i] = true;
                    }
                }
            }
            m
        };

        // --- Projection cross-check: the GPU silhouette (non-sky pixels) and
        // the CPU raster coverage must agree, otherwise every mask-based gate
        // below would measure the wrong pixels. The background is the smooth
        // bright-blue sky-dome gradient; building/ground pixels are not it. --
        let is_sky = |px: &[u8]| -> bool {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            b > r + 14 && b > g + 8 && (r + g + b) > 330
        };
        let mut raster_cov = 0usize;
        let mut lit_px = 0usize;
        let mut overlap = 0usize;
        for p in 0..(w * h) as usize {
            let on_raster = mats_px[p] >= 0;
            let on_render = !is_sky(&rgba_tex[p * 4..p * 4 + 4]);
            raster_cov += on_raster as usize;
            lit_px += on_render as usize;
            overlap += (on_raster && on_render) as usize;
        }
        let agree_raster = overlap as f64 / raster_cov.max(1) as f64;
        let agree_render = overlap as f64 / lit_px.max(1) as f64;
        eprintln!(
            "[mesh_m0] projection cross-check: raster {raster_cov} px, render {lit_px} px, \
             overlap/raster = {agree_raster:.3}, overlap/render = {agree_render:.3}"
        );
        assert!(
            agree_raster >= 0.70 && agree_render >= 0.70,
            "CPU mask and GPU silhouette disagree (overlap/raster {agree_raster:.3}, \
             overlap/render {agree_render:.3}) — the mask projection does not match \
             the render; the gates below would be meaningless"
        );

        let luma = |p: &[u8]| {
            (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0
        };
        let idx = |x: u32, y: u32| -> usize { ((y * w + x) * 4) as usize };

        // --- Gate (a): facade detail energy — the TEXTURE RESIDUAL the texture
        // layer adds to the surface, as a multiple of the flat control's own
        // within-face noise floor. ---------------------------------------------
        //
        // Why not a raw |∇luminance| ratio: the forge cook models every
        // clapboard COURSE as real geometry (a stack of beveled course prisms,
        // visible in the renders), so the dominant luminance gradients are
        // course-EDGE shading steps that appear IDENTICALLY in the textured and
        // the flat render — they cancel in a ratio and starve it toward ~1.3x
        // no matter how strong the texture is (measured across cameras/spp).
        //
        // The honest, geometry-immune signal is the per-pixel TEXTURE RESIDUAL
        // — mean |Δrgb| between the textured render and the flat control over
        // the wall mask. The flat control IS the same geometry, lighting and
        // camera with ONLY the texture removed, so this residual is exactly
        // what the texture layer paints onto the surface and nothing else (the
        // shared course-edge geometry cancels). To prove that residual is
        // texture and not Monte-Carlo noise we divide by a MATCHED noise floor:
        // the flat control's own per-pixel |Δrgb| between SAME-FACE horizontal
        // neighbours, ÷√2 (a flat Lambert face is constant shade, so its only
        // neighbour variation is uncorrelated render noise; ÷√2 converts a
        // two-sample difference to a one-sample deviation, matching the
        // residual's one-sample form). Both terms are per-pixel RGB colour
        // deviations — an apples-to-apples ratio. Texture on the surface ⇒
        // residual ≫ floor.
        let wall_mask = eroded(wall_id);
        let wall_px = wall_mask.iter().filter(|&&b| b).count();
        assert!(
            wall_px >= 3000,
            "facade mask too small ({wall_px} px < 3000) — camera drifted off the wall"
        );
        let rgb_dev = |a: &[u8], b: &[u8]| -> f64 {
            ((a[0] as f32 - b[0] as f32).abs()
                + (a[1] as f32 - b[1] as f32).abs()
                + (a[2] as f32 - b[2] as f32).abs()) as f64
                / (3.0 * 255.0)
        };
        // Texture residual: mean |Δrgb| textured-vs-flat over the wall.
        let residual = {
            let mut sum = 0.0f64;
            for (p, &on) in wall_mask.iter().enumerate() {
                if on {
                    sum += rgb_dev(&rgba_tex[p * 4..p * 4 + 4], &rgba_flat[p * 4..p * 4 + 4]);
                }
            }
            sum / wall_px as f64
        };
        // Matched per-pixel RGB noise floor: flat control, same-face horizontal
        // neighbour |Δrgb| ÷ √2.
        let noise_floor = {
            let mut sum = 0.0f64;
            let mut pairs = 0usize;
            for y in 0..h {
                for x in 0..w - 1 {
                    let p = (y * w + x) as usize;
                    if wall_mask[p] && wall_mask[p + 1] && tris_px[p + 1] == tris_px[p] {
                        sum += rgb_dev(
                            &rgba_flat[p * 4..p * 4 + 4],
                            &rgba_flat[(p + 1) * 4..(p + 1) * 4 + 4],
                        );
                        pairs += 1;
                    }
                }
            }
            (sum / pairs.max(1) as f64) / std::f64::consts::SQRT_2
        };
        let energy_ratio = residual / noise_floor.max(1e-9);
        let pass_a = energy_ratio >= 2.5;
        eprintln!(
            "[mesh_m0] facade detail energy {energy_ratio:.2}x flat-control \
             (per-pixel RGB texture residual / matched flat noise floor, gate >= 2.5x) \
             -> {} (residual {residual:.5}, noise floor {noise_floor:.5}, {wall_px} \
             facade px)",
            if pass_a { "PASS" } else { "FAIL" }
        );

        // --- Gate (b): per-part materials — wall vs roof and wall vs trim
        // mean-colour splits in the TEXTURED render, over the eroded
        // per-material masks. Distinct cooked materials must land as distinct
        // rendered parts. -----------------------------------------------------
        let mean_rgb = |rgba: &[u8], mask: &[bool]| -> ([f64; 3], usize) {
            let mut sum = [0.0f64; 3];
            let mut n = 0usize;
            for (p, &on) in mask.iter().enumerate() {
                if on {
                    for (s, &v) in sum.iter_mut().zip(&rgba[p * 4..p * 4 + 3]) {
                        *s += v as f64;
                    }
                    n += 1;
                }
            }
            ([sum[0] / n.max(1) as f64, sum[1] / n.max(1) as f64, sum[2] / n.max(1) as f64], n)
        };
        let split = |a: [f64; 3], b: [f64; 3]| -> f64 {
            ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / (3.0 * 255.0)
        };
        let trim_mask = eroded(trim_id);
        let (wall_rgb, n_wall) = mean_rgb(&rgba_tex, &wall_mask);
        let (roof_rgb, n_roof) = mean_rgb(&rgba_tex, &eroded(roof_id));
        let (trim_rgb, n_trim) = mean_rgb(&rgba_tex, &trim_mask);
        let (door_rgb, n_door) = mean_rgb(&rgba_tex, &eroded(door_id));
        assert!(
            n_roof >= 1500,
            "roof mask too small ({n_roof} px < 1500) — roof planes not in frame"
        );
        assert!(
            n_trim >= 200,
            "trim mask too small ({n_trim} px < 200) — trim not visible"
        );
        assert!(
            n_door >= 200,
            "door mask too small ({n_door} px < 200) — door not visible"
        );
        let d_roof = split(wall_rgb, roof_rgb);
        let d_trim = split(wall_rgb, trim_rgb);
        let d_door = split(wall_rgb, door_rgb);
        // The cooked trim albedo [0.85,0.82,0.78] is INTENTIONALLY close to
        // the wall's [0.82,0.75,0.6] (painted stucco trim on a beige
        // clapboard house), so wall-vs-trim stays a self-calibrated check:
        // the FLAT control (pure cooked albedos, same lighting) shows what
        // the maximal trim/wall split looks like here, and the textured
        // render must reproduce at least 60% of it above an absolute floor.
        // Wall-vs-roof and wall-vs-door carry the hard 0.12 distinctness
        // gates. RE-BASELINED 2026-06-12 on the post-zoning-fix render
        // (materials now indexed by FORGE id: trim mask = reveal stucco, door
        // = wood — they were first measured on the cross-wired render where
        // "trim" pixels were glass geometry shaded stucco and the door was
        // detail_metal): measured d_trim 0.057 (flat-control split 0.060),
        // d_door 0.176 — trim floor pinned at 0.03 (just inside the
        // measurement, same ~2x margin as the original 0.008 floor), door
        // keeps the hard 0.12.
        let (wall_rgb_flat, _) = mean_rgb(&rgba_flat, &wall_mask);
        let (trim_rgb_flat, _) = mean_rgb(&rgba_flat, &trim_mask);
        let d_trim_flat = split(wall_rgb_flat, trim_rgb_flat);
        let pass_b1 = d_roof > 0.12;
        let pass_b2 = d_trim > 0.03 && d_trim >= 0.6 * d_trim_flat;
        let pass_b3 = d_door > 0.12;
        eprintln!(
            "[mesh_m0] per-part materials: wall vs roof |Δrgb| = {d_roof:.3} (gate > 0.12) \
             -> {} (wall rgb [{:.0},{:.0},{:.0}] {n_wall} px, roof rgb [{:.0},{:.0},{:.0}] \
             {n_roof} px)",
            if pass_b1 { "PASS" } else { "FAIL" },
            wall_rgb[0],
            wall_rgb[1],
            wall_rgb[2],
            roof_rgb[0],
            roof_rgb[1],
            roof_rgb[2]
        );
        eprintln!(
            "[mesh_m0] per-part materials: wall vs trim |Δrgb| = {d_trim:.3} \
             (gate > 0.03 and >= 0.6x the flat-control split {d_trim_flat:.3}; the \
             cooked trim albedo is near-wall by design) -> {} \
             (trim rgb [{:.0},{:.0},{:.0}] {n_trim} px)",
            if pass_b2 { "PASS" } else { "FAIL" },
            trim_rgb[0],
            trim_rgb[1],
            trim_rgb[2]
        );
        eprintln!(
            "[mesh_m0] per-part materials: wall vs door |Δrgb| = {d_door:.3} (gate > 0.12) \
             -> {} (door rgb [{:.0},{:.0},{:.0}] {n_door} px)",
            if pass_b3 { "PASS" } else { "FAIL" },
            door_rgb[0],
            door_rgb[1],
            door_rgb[2]
        );

        // --- Gate (c): lit coverage of the frame + the building is actually
        // LIT (mean luminance over its pixels well above black). --------------
        let coverage = lit_px as f64 / (w * h) as f64;
        let mut lum_sum = 0.0f64;
        for p in 0..(w * h) as usize {
            if !is_sky(&rgba_tex[p * 4..p * 4 + 4]) {
                lum_sum += luma(&rgba_tex[p * 4..p * 4 + 4]) as f64;
            }
        }
        let mean_lit_luma = lum_sum / lit_px.max(1) as f64;
        let pass_c = coverage >= 0.25;
        eprintln!(
            "[mesh_m0] lit coverage {coverage:.3} of frame (gate >= 0.25) -> {}, \
             seconds/frame {secs:.2}s (mean lit luma {mean_lit_luma:.3})",
            if pass_c { "PASS" } else { "FAIL" }
        );
        eprintln!("[mesh_m0] wrote {}", tex_png.display());
        eprintln!("[mesh_m0] wrote {} (flat control)", flat_png.display());
        eprintln!(
            "[mesh_m0] eyeball: a real textured craftsman — clapboard siding courses, \
             slate roof, stucco trim, pale-blue glass panes (opaque BY CHOICE here so \
             this test's gates stay stable; real transmission is gated in \
             mesh_glass_transmits_checkerboard) — vs the flat-colour control"
        );

        // --- The gates. Fix the UV/material/texture binding, never weaken. ---
        assert!(
            pass_a,
            "texture detail is not landing on the mesh facade (residual/floor \
             {energy_ratio:.2}x < 2.5x; per-pixel RGB texture residual {residual:.5} vs \
             matched flat noise floor {noise_floor:.5}) — the UV or atlas binding is wrong"
        );
        assert!(
            pass_b1,
            "wall and roof do not render as distinct materials \
             (|Δrgb| {d_roof:.3} <= 0.12) — per-triangle material ids are not landing"
        );
        assert!(
            pass_b2,
            "wall and trim do not render as distinct materials \
             (|Δrgb| {d_trim:.3}, floor 0.03, flat-control split {d_trim_flat:.3}) \
             — per-triangle material ids are not landing"
        );
        assert!(
            pass_b3,
            "wall and door do not render as distinct materials \
             (|Δrgb| {d_door:.3} <= 0.12) — per-triangle material ids are not landing"
        );
        assert!(
            pass_c,
            "building covers too little of the inspection frame \
             (coverage {coverage:.3} < 0.25) — camera framing drifted"
        );
        assert!(
            mean_lit_luma > 0.15,
            "building renders nearly black (mean lit luma {mean_lit_luma:.3}) — \
             the light rig is not landing"
        );
    }

    /// Mesh GLASS ISOLATION GATE: prove REAL transmission through the mesh
    /// path tracer on a TRIVIAL, CHEAP scene — a 16x16 emissive checkerboard
    /// wall at z=0 and a two-faced glass slab (front face z=2.0, back face
    /// z=1.96) in front of it, camera at z=6 looking straight through the
    /// slab at the wall. No directional lights, no sky dome: the checker's
    /// white cells are the ONLY emitters, so any pattern the glass-region
    /// pixels show is, by construction, content from BEHIND the glass.
    ///
    /// Gates (each render 256x256 @ 32 spp — SECONDS, printed):
    ///   (packing) the glass material packs a[0]=MAT_GLASS(3), a[10]=ior,
    ///       a[24..27]=absorption_color (0,0,0), a[27]=absorption_depth 1.0,
    ///       a[73]=thin_walled — the Vulkan MaterialData reflection slots,
    ///       carrying the EXACT parameter set the proven SDF window path
    ///       passes to sample_glass in megakernel.slang
    ///       (`sample_glass(wo, n, 1.5, rough, float3(0.0f), 1.0f, ...)`);
    ///       `transmission = 0` still packs MAT_LAMBERT with the historical
    ///       slots untouched;
    ///   (transmission) Pearson correlation between glass-region pixel
    ///       luminance and the KNOWN checker parity projected through each
    ///       pixel's camera ray onto the wall plane — scored against the
    ///       better of two optical models (ideal-slab straight-through, and
    ///       the engine's flipped-normal double-"entering" refraction that
    ///       magnifies the pattern; see the inline comment): transmissive
    ///       r > 0.5 AND r >= 3x the opaque control's r (identical scene, the
    ///       glass material's transmission flipped to 0 — its unlit Lambert
    ///       panes cannot show the wall pattern);
    ///   (sanity) the directly-visible wall outside the glass correlates
    ///       r > 0.8 in BOTH renders, so a transmission failure is
    ///       unambiguously the glass route, not the emissive wall.
    /// Writes `mesh_glass_checker.png` + `mesh_glass_checker_opaque.png`,
    /// then ONE craftsman demo render with the cooked glass channel flipped
    /// transmissive — `mesh_craftsman_glass.png`, eyeball-only, no gate.
    ///
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --release --lib mesh_glass_transmits_checkerboard -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn mesh_glass_transmits_checkerboard() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        // --- Packer layout: the MAT_GLASS slots (CPU-only, instant). --------
        let glass_mat = PbrMaterial {
            base_color: [1.0, 1.0, 1.0],
            roughness: 0.0, // delta glass: clean refraction, f/pdf = 1
            transmission: 1.0,
            ior: 1.5,
            thin_walled: false,
            ..Default::default()
        };
        let packed = super::pack_vulkan_mesh_material(glass_mat);
        assert_eq!(packed[0].to_bits(), 3, "a[0] must pack MAT_GLASS (3)");
        assert_eq!(packed[10], 1.5, "a[10] must carry the glass ior");
        assert_eq!(
            &packed[24..28],
            &[0.0, 0.0, 0.0, 1.0],
            "a[24..28] must carry absorption_color (0,0,0) + absorption_depth \
             1.0 — the SDF window reference's sample_glass arguments"
        );
        assert_eq!(packed[73].to_bits(), 0, "a[73] thin_walled = 0 (refractive)");
        let packed_thin =
            super::pack_vulkan_mesh_material(PbrMaterial { thin_walled: true, ..glass_mat });
        assert_eq!(
            packed_thin[73].to_bits(),
            1,
            "thin_walled = true must land in a[73] (the MaterialData thin_walled slot)"
        );
        let packed_opaque = super::pack_vulkan_mesh_material(PbrMaterial::default());
        assert_eq!(
            packed_opaque[0].to_bits(),
            1,
            "transmission = 0 must still pack MAT_LAMBERT (1)"
        );
        assert_eq!(
            packed_opaque[10], 1.5,
            "default ior must reproduce the old hardcoded 1.5 byte-for-byte"
        );
        assert_eq!(
            &packed_opaque[24..28],
            &[0.0; 4],
            "opaque materials must leave the absorption slots zeroed (historical)"
        );
        eprintln!(
            "[glass] packer: type {} in a[0], ior {} in a[10], absorption \
             [{},{},{}]/{} in a[24..28], thin_walled {} in a[73]",
            packed[0].to_bits(),
            packed[10],
            packed[24],
            packed[25],
            packed[26],
            packed[27],
            packed[73].to_bits()
        );

        // --- The trivial scene: emissive checker wall + glass slab. ---------
        fn push_quad(
            mesh: &mut CookedMesh,
            corners: [[f32; 3]; 4],
            normal: [f32; 3],
            mat: u8,
        ) {
            let v0 = mesh.positions.len() as u32;
            for c in corners {
                mesh.positions.push(c);
                mesh.normals.push(normal);
                mesh.uvs.push([0.0, 0.0]);
            }
            mesh.indices.push([v0, v0 + 1, v0 + 2]);
            mesh.indices.push([v0, v0 + 2, v0 + 3]);
            mesh.material_ids.push(mat);
            mesh.material_ids.push(mat);
        }
        const CELL: f32 = 0.5; // checker cell size (m)
        const HALF: f32 = 4.0; // wall half-extent (m) -> 16x16 cells
        let mut scene_mesh = CookedMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_ids: Vec::new(),
        };
        let ncell = (2.0 * HALF / CELL) as i32;
        for cy in 0..ncell {
            for cx in 0..ncell {
                let x0 = -HALF + cx as f32 * CELL;
                let y0 = -HALF + cy as f32 * CELL;
                push_quad(
                    &mut scene_mesh,
                    [
                        [x0, y0, 0.0],
                        [x0 + CELL, y0, 0.0],
                        [x0 + CELL, y0 + CELL, 0.0],
                        [x0, y0 + CELL, 0.0],
                    ],
                    [0.0, 0.0, 1.0],
                    ((cx + cy) % 2) as u8, // 0 = white EMITTER, 1 = black
                );
            }
        }
        // Two-faced glass slab: the craftsman panes' topology. Front face
        // normal +Z (toward camera), back face normal -Z (outward of slab).
        let (gh, zf, zb) = (0.9f32, 2.0f32, 1.96f32);
        const GLASS_ID: u8 = 2;
        push_quad(
            &mut scene_mesh,
            [[-gh, -gh, zf], [gh, -gh, zf], [gh, gh, zf], [-gh, gh, zf]],
            [0.0, 0.0, 1.0],
            GLASS_ID,
        );
        push_quad(
            &mut scene_mesh,
            [[-gh, -gh, zb], [-gh, gh, zb], [gh, gh, zb], [gh, -gh, zb]],
            [0.0, 0.0, -1.0],
            GLASS_ID,
        );
        let materials = [
            // White checker cell: the scene's ONLY light source. emission
            // packs base_color * strength (a[20..24]).
            PbrMaterial {
                base_color: [1.0, 1.0, 1.0],
                emission_strength: 3.0,
                ..Default::default()
            },
            // Black checker cell: near-zero albedo, no emission.
            PbrMaterial {
                base_color: [0.02, 0.02, 0.02],
                ..Default::default()
            },
            glass_mat,
        ];

        // --- Camera straight through the slab; lights all OFF. --------------
        let (w, h) = (256u32, 256u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [0.0f32, 0.0, 6.0];
        let target = [0.0f32, 0.0, 0.0];
        let rig = LightRig {
            sun_dir: [0.0, 1.0, 0.0],
            sun_intensity: 0.0,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 0.0,
            ..Default::default()
        };
        let spp = 32u32;

        let render = |mats: &[PbrMaterial]| -> (Vec<u8>, f64) {
            let t0 = std::time::Instant::now();
            let rgba = pathtrace_mesh_lit_to_rgba(
                &scene_mesh.positions,
                &scene_mesh.normals,
                &scene_mesh.uvs,
                &scene_mesh.indices,
                &scene_mesh.material_ids,
                mats,
                &[],
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                &rig,
            )
            .expect("checker-through-glass render should succeed");
            (rgba, t0.elapsed().as_secs_f64())
        };
        let (rgba_t, secs_t) = render(&materials);
        eprintln!("[glass] render time {secs_t:.2}s (each frame, expect < 10s)");
        let mut opaque_materials = materials;
        opaque_materials[GLASS_ID as usize].transmission = 0.0;
        let (rgba_o, secs_o) = render(&opaque_materials);
        eprintln!("[glass] render time {secs_o:.2}s (each frame, expect < 10s)");

        let out_dir = std::env::temp_dir();
        let force_alpha = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        let t_png = out_dir.join("mesh_glass_checker.png");
        let o_png = out_dir.join("mesh_glass_checker_opaque.png");
        write_png_rgba(t_png.to_str().unwrap(), &force_alpha(rgba_t.clone()), w, h);
        write_png_rgba(o_png.to_str().unwrap(), &force_alpha(rgba_o.clone()), w, h);
        eprintln!("[glass] wrote {} (transmissive)", t_png.display());
        eprintln!("[glass] wrote {} (opaque control)", o_png.display());

        // --- Per-pixel ground truth: the same pinhole as the GPU (proven by
        // the M0 projection cross-check), inverted to a camera ray, hit the
        // wall plane z=0, read the checker parity. TWO optical models, both
        // "the pattern behind the glass, transmitted":
        //   (straight) ideal slab physics — enter+exit refractions cancel,
        //       the ray continues straight (lateral offset sub-pixel);
        //   (refracted) the engine's CURRENT model — the megakernel flips
        //       shading normals toward the ray and `dispatch_sample` passes
        //       `mat.ior` (not the IOR-stack eta), so BOTH slab faces refract
        //       as "entering" with eta = 1/ior: the slab is a weak lens that
        //       MAGNIFIES the pattern (visible in the PNG).
        // The gate scores against the better-matching model so it certifies
        // transmission, not one refraction convention. Pixels whose first hit
        // is the glass slab (CPU rasterizer, eroded 1px) are the glass
        // region; pixels whose first hit is the wall itself are the
        // direct-view sanity region. Pixels within 0.04 m of a predicted
        // checker boundary are excluded (AA tolerance). ----------------------
        let (mats_px, _tris_px) = rasterize_material_masks(&scene_mesh, eye, target, fov_y, w, h);
        let eye_v = glam::Vec3::from(eye);
        let fwd = (glam::Vec3::from(target) - eye_v).normalize();
        let right = fwd.cross(glam::Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = (fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        // Snell refraction of unit `d` through a plane with normal +Z,
        // ratio eta = n_i/n_t (the sample_glass convention).
        let refract_z = |d: glam::Vec3, eta: f32| -> glam::Vec3 {
            let n = glam::Vec3::Z;
            let cos_i = -n.dot(d); // d points INTO the surface (d.z < 0)
            let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
            debug_assert!(sin2_t < 1.0, "no TIR at these angles");
            let cos_t = (1.0 - sin2_t).sqrt();
            (d * eta + n * (eta * cos_i - cos_t)).normalize()
        };
        // Checker parity at the wall-plane point, None if outside the wall or
        // within 0.04 m of a cell boundary.
        let parity_at = |wx: f32, wy: f32| -> Option<f64> {
            if wx <= -HALF || wx >= HALF || wy <= -HALF || wy >= HALF {
                return None;
            }
            let fx = (wx + HALF) / CELL;
            let fy = (wy + HALF) / CELL;
            let frx = fx - fx.floor();
            let fry = fy - fy.floor();
            if frx.min(1.0 - frx).min(fry).min(1.0 - fry) * CELL < 0.04 {
                return None;
            }
            let white = ((fx.floor() as i32 + fy.floor() as i32) % 2) == 0;
            Some(if white { 1.0 } else { 0.0 })
        };
        // (pixel, expected) per model.
        let mut glass_straight: Vec<(usize, f64)> = Vec::new();
        let mut glass_refracted: Vec<(usize, f64)> = Vec::new();
        let mut wall_sel: Vec<(usize, f64)> = Vec::new();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = (y * w + x) as usize;
                let ndc_x = aspect * ((x as f32 + 0.5) * 2.0 / w as f32 - 1.0);
                let ndc_y = 1.0 - (y as f32 + 0.5) * 2.0 / h as f32;
                let dir = (fwd + right * (ndc_x * tan_half) + up * (ndc_y * tan_half))
                    .normalize();
                if dir.z >= -1e-6 {
                    continue;
                }
                let m = mats_px[i];
                let eroded = |id: i32| {
                    mats_px[i - 1] == id
                        && mats_px[i + 1] == id
                        && mats_px[i - w as usize] == id
                        && mats_px[i + w as usize] == id
                };
                let at_plane = |origin: glam::Vec3, d: glam::Vec3, z: f32| -> glam::Vec3 {
                    origin + d * ((z - origin.z) / d.z)
                };
                if m == GLASS_ID as i32 && eroded(GLASS_ID as i32) {
                    // Straight-through model.
                    let p = at_plane(eye_v, dir, 0.0);
                    if let Some(e) = parity_at(p.x, p.y) {
                        glass_straight.push((i, e));
                    }
                    // Double-"entering" refraction model (the engine's).
                    let eta = 1.0 / 1.5;
                    let p1 = at_plane(eye_v, dir, zf);
                    let d1 = refract_z(dir, eta);
                    let p2 = at_plane(p1, d1, zb);
                    let d2 = refract_z(d1, eta);
                    let p3 = at_plane(p2, d2, 0.0);
                    if let Some(e) = parity_at(p3.x, p3.y) {
                        glass_refracted.push((i, e));
                    }
                } else if (m == 0 || m == 1) && eroded(m) {
                    let p = at_plane(eye_v, dir, 0.0);
                    if let Some(e) = parity_at(p.x, p.y) {
                        wall_sel.push((i, e));
                    }
                }
            }
        }
        eprintln!(
            "[glass] regions: {} glass px (straight model) / {} (refracted \
             model), {} direct-wall px (eroded, boundary-excluded)",
            glass_straight.len(),
            glass_refracted.len(),
            wall_sel.len()
        );
        assert!(
            glass_straight.len() >= 2000 && glass_refracted.len() >= 2000,
            "glass region too small ({} / {} px) — slab not covering the frame center",
            glass_straight.len(),
            glass_refracted.len()
        );
        assert!(
            wall_sel.len() >= 2000,
            "direct-wall region too small ({} px)",
            wall_sel.len()
        );

        // Pearson correlation: measured luminance vs the known checker parity.
        let pearson = |img: &[u8], sel: &[(usize, f64)]| -> f64 {
            let n = sel.len() as f64;
            let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &(i, xv) in sel {
                let yv = 0.2126 * img[i * 4] as f64
                    + 0.7152 * img[i * 4 + 1] as f64
                    + 0.0722 * img[i * 4 + 2] as f64;
                sx += xv;
                sy += yv;
                sxx += xv * xv;
                syy += yv * yv;
                sxy += xv * yv;
            }
            let cov = sxy / n - (sx / n) * (sy / n);
            let vx = sxx / n - (sx / n) * (sx / n);
            let vy = syy / n - (sy / n) * (sy / n);
            if vx <= 1e-12 || vy <= 1e-9 {
                return 0.0; // constant image: no pattern at all
            }
            cov / (vx.sqrt() * vy.sqrt())
        };

        // Sanity: the emissive wall itself renders the pattern in BOTH
        // frames — failures below are then unambiguously the glass route.
        let rw_t = pearson(&rgba_t, &wall_sel);
        let rw_o = pearson(&rgba_o, &wall_sel);
        eprintln!(
            "[glass] direct-wall sanity: r={rw_t:.3} (transmissive frame), \
             r={rw_o:.3} (opaque frame) (gate both > 0.8)"
        );
        assert!(
            rw_t > 0.8 && rw_o > 0.8,
            "the emissive checker wall itself does not render (direct-view \
             r {rw_t:.3}/{rw_o:.3} <= 0.8) — scene/emission problem, not glass"
        );

        // --- THE gate: the pattern BEHIND the glass shows through ONLY when
        // the material transmits. r = the better-matching optical model. -----
        let model_r = |img: &[u8]| -> (f64, f64) {
            (pearson(img, &glass_straight), pearson(img, &glass_refracted))
        };
        let (rt_s, rt_r) = model_r(&rgba_t);
        let (ro_s, ro_r) = model_r(&rgba_o);
        let rt = rt_s.max(rt_r);
        let ro = ro_s.max(ro_r);
        eprintln!(
            "[glass] model correlations: transmissive straight={rt_s:.3} \
             refracted={rt_r:.3}; opaque straight={ro_s:.3} refracted={ro_r:.3}"
        );
        let pass = rt > 0.5 && rt >= 3.0 * ro;
        eprintln!(
            "[glass] checker-through-glass: transmissive correlation r={rt:.3} vs \
             opaque r={ro:.3} (gate rt > 0.5 and rt >= 3x ro) -> {}",
            if pass { "PASS" } else { "FAIL" }
        );
        assert!(
            pass,
            "checkerboard behind the glass does not show through (transmissive \
             r {rt:.3}, opaque control r {ro:.3}; need rt > 0.5 and rt >= 3x ro) \
             — check the MAT_GLASS slot layout in pack_vulkan_mesh_material \
             (type a[0], ior a[10], absorption a[24..28], thin_walled a[73]) \
             against material_types.slang's Vulkan reflection layout"
        );

        // --- ONE craftsman demo render (eyeball-only, no gate): the cooked
        // glass channel flipped transmissive — the same MAT_GLASS parameters
        // the gate above just proved. The single allowed slow render. --------
        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");
        let mesh = load_craftsman_mesh(&asset_path);
        let (mut cmats, ctex, channels) = load_building_mesh_pbr_by_forge_id(&asset_path);
        let glass_id = channels
            .iter()
            .position(|c| c == "glass")
            .expect("cooked payload has no 'glass' channel");
        cmats[glass_id].transmission = 1.0;
        cmats[glass_id].ior = 1.5;
        cmats[glass_id].thin_walled = false;
        // Frontal-low framing: the porch parapet's glazing and the facade
        // windows fill the view, sky and porch cavity behind them.
        let (cw, ch) = (512u32, 512u32);
        let c_target = [0.0f32, 2.6, 0.0];
        let c_eye = [0.5f32, 2.4, 14.0];
        let c_rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let rgba_demo = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &cmats,
            &ctex,
            c_eye,
            c_target,
            fov_y,
            cw,
            ch,
            128,
            &c_rig,
        )
        .expect("craftsman glass demo render should succeed");
        let demo_png = out_dir.join("mesh_craftsman_glass.png");
        write_png_rgba(demo_png.to_str().unwrap(), &force_alpha(rgba_demo), cw, ch);
        eprintln!(
            "[glass] wrote {} (craftsman demo, glass channel transmissive, \
             {:.2}s — the one allowed slow render)",
            demo_png.display(),
            t0.elapsed().as_secs_f64()
        );
    }

    /// Connected components of one forge material id over 0.1 mm-welded
    /// vertices — the real measured geometry the curtain-wall grid gates
    /// count. Returns each component's AABB.
    #[cfg(feature = "spectra-native")]
    fn mesh_material_components(mesh: &CookedMesh, mat: u8) -> Vec<([f32; 3], [f32; 3])> {
        use std::collections::HashMap;
        let weld: Vec<u64> = {
            let mut ids: HashMap<[i64; 3], u64> = HashMap::new();
            mesh.positions
                .iter()
                .map(|p| {
                    let key = [
                        (p[0] as f64 * 1.0e4).round() as i64,
                        (p[1] as f64 * 1.0e4).round() as i64,
                        (p[2] as f64 * 1.0e4).round() as i64,
                    ];
                    let next = ids.len() as u64;
                    *ids.entry(key).or_insert(next)
                })
                .collect()
        };
        let tris: Vec<usize> = (0..mesh.indices.len())
            .filter(|&t| mesh.material_ids[t] == mat)
            .collect();
        let mut parent: Vec<usize> = (0..tris.len()).collect();
        fn find(parent: &mut Vec<usize>, mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        let mut owner: HashMap<u64, usize> = HashMap::new();
        for (ti, &t) in tris.iter().enumerate() {
            for &v in &mesh.indices[t] {
                match owner.get(&weld[v as usize]) {
                    Some(&other) => {
                        let (ra, rb) = (find(&mut parent, ti), find(&mut parent, other));
                        if ra != rb {
                            parent[ra] = rb;
                        }
                    }
                    None => {
                        owner.insert(weld[v as usize], ti);
                    }
                }
            }
        }
        let mut boxes: HashMap<usize, ([f32; 3], [f32; 3])> = HashMap::new();
        for (ti, &t) in tris.iter().enumerate() {
            let root = find(&mut parent, ti);
            let entry = boxes
                .entry(root)
                .or_insert(([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]));
            for &v in &mesh.indices[t] {
                let p = mesh.positions[v as usize];
                for a in 0..3 {
                    entry.0[a] = entry.0[a].min(p[a]);
                    entry.1[a] = entry.1[a].max(p[a]);
                }
            }
        }
        boxes.into_values().collect()
    }

    /// Mean Sobel gradient-magnitude of luminance over mask pixels whose full
    /// 3x3 neighbourhood is inside the mask — the facade detail-energy
    /// measure (edges from mullions/transoms/spandrel-glass alternation),
    /// silhouette-immune by construction.
    #[cfg(feature = "spectra-native")]
    fn masked_sobel_energy(rgba: &[u8], mask: &[bool], w: u32, h: u32) -> f64 {
        let luma = |i: usize| -> f64 {
            0.2126 * rgba[i * 4] as f64
                + 0.7152 * rgba[i * 4 + 1] as f64
                + 0.0722 * rgba[i * 4 + 2] as f64
        };
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for y in 1..h - 1 {
            'px: for x in 1..w - 1 {
                let i = (y * w + x) as usize;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let j = ((y as i32 + dy) * w as i32 + (x as i32 + dx)) as usize;
                        if !mask[j] {
                            continue 'px;
                        }
                    }
                }
                let l = |dx: i32, dy: i32| {
                    luma(((y as i32 + dy) * w as i32 + (x as i32 + dx)) as usize)
                };
                let gx = (l(1, -1) + 2.0 * l(1, 0) + l(1, 1))
                    - (l(-1, -1) + 2.0 * l(-1, 0) + l(-1, 1));
                let gy = (l(-1, 1) + 2.0 * l(0, 1) + l(1, 1))
                    - (l(-1, -1) + 2.0 * l(0, -1) + l(1, -1));
                sum += (gx * gx + gy * gy).sqrt();
                n += 1;
            }
        }
        sum / n.max(1) as f64
    }

    /// CURTAIN-WALL FACADE GATE (forge facade-axis wave: the glass office).
    /// Loads the COOKED curtain-wall office tower
    /// (`city.office.l5.3x3.glass_office_tower_01`, cooked with
    /// `forge.facade_system: "curtain_wall"` into
    /// `assets/buildings/curtain_wall`) and verifies the facade family by
    /// GEOMETRY and by a CHEAP render (256² @ 32 spp — seconds):
    ///
    ///   (grid) mullion/transom counts measured as welded connected
    ///       components of the trim material, cells = Σ_walls
    ///       (mullions−1)·(transoms−1) — must equal the parametric
    ///       expectation bays·floors·3 derived from the payload's own
    ///       forge_description (width/depth/floors/floor_height) and the
    ///       forge CurtainWallParams default bay width (1.8 m);
    ///   (glass) vision panes counted from MAT_GLASS triangles == 2·bays·
    ///       floors, AND the cooked glass material arrives transmissive
    ///       (transmission > 0, thin_walled) and packs MAT_GLASS(3) —
    ///       spandrels counted as wall-plane MAT_WALL panels == bays·floors,
    ///       their material opaque (packs MAT_LAMBERT(1));
    ///   (energy) facade detail energy — masked Sobel gradient energy of the
    ///       rendered tower vs a flat Lambert box of the same dimensions,
    ///       same camera/rig/spp — ratio >= 3.0;
    ///   (time) each gate render < 10 s.
    /// Writes `curtain_wall_office.png` + `curtain_wall_flatbox.png`.
    ///
    /// Cook first (civitas repo, ~/Ochroma/projects/civitas_care):
    ///   mkdir -p /tmp/curtain_src/office && cp \
    ///     assets/source/buildings/office/glass_office_tower_01.asset.json \
    ///     /tmp/curtain_src/office/ && mkdir -p assets/buildings/curtain_wall/textures \
    ///     && cp -r assets/buildings/forge_starter/textures/polyhaven \
    ///     assets/buildings/curtain_wall/textures/
    ///   cargo build --release --bin game_asset_cook && \
    ///   GAME_FORGE_BIN=$HOME/src/forge/target/release/aetherspectra-forge \
    ///     ./target/release/game_asset_cook --no-starters \
    ///     --source /tmp/curtain_src --output assets/buildings/curtain_wall
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib forge_curtain_wall_office -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_curtain_wall_office() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        let atoms_path = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            "Ochroma/projects/civitas_care/assets/buildings/curtain_wall/atoms/\
             city.office.l5.3x3.glass_office_tower_01.atoms.json",
        );
        let mesh = load_craftsman_mesh(&atoms_path);
        let (materials, textures, channels) = load_building_mesh_pbr_by_forge_id(&atoms_path);
        assert_eq!(channels[2], "glass", "forge id 2 must bind the glass channel");

        // ── The parametric expectation, from the payload's own cooked
        //    forge_description — never hardcoded counts. ─────────────────────
        let json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&atoms_path)
                .unwrap_or_else(|e| panic!("read {}: {e}", atoms_path.display())),
        )
        .expect("parse atoms.json");
        let desc = &json["forge_description"];
        assert_eq!(
            desc["facade_system"].as_str(),
            Some("curtain_wall"),
            "payload was not cooked with facade_system curtain_wall"
        );
        let width = desc["footprint"]["width"].as_f64().unwrap() as f32;
        let depth = desc["footprint"]["depth"].as_f64().unwrap() as f32;
        let floors = desc["floors"].as_u64().unwrap() as usize;
        let fh = desc["floor_height"].as_f64().unwrap() as f32;
        // Forge CurtainWallParams::default().bay_width — the one cross-repo
        // parametric constant (mullion spacing target).
        const BAY_WIDTH: f32 = 1.8;
        let bays_w = (width / BAY_WIDTH).round().max(1.0) as usize;
        let bays_d = (depth / BAY_WIDTH).round().max(1.0) as usize;
        let bays_total = 2 * (bays_w + bays_d);
        let exp_mullions = bays_total + 4; // bays_e + 1 per edge
        let exp_transoms = 4 * (3 * floors + 1);
        let exp_cells = 3 * bays_total * floors;
        let exp_glass = 2 * bays_total * floors;
        let exp_spandrels = bays_total * floors;
        let total_h = floors as f32 * fh;
        let (half_w, half_d) = (width * 0.5, depth * 0.5);

        // ── GATE 1: the mullion/transom grid, measured from welded
        //    components of the trim material (forge id 4). ───────────────────
        let trim = mesh_material_components(&mesh, 4);
        // Per wall: components beyond each footprint plane (members stand
        // proud of the skin, so their centroids sit outside the plane).
        let mut per_wall: [(usize, usize); 4] = [(0, 0); 4]; // (mullions, transoms)
        for (mn, mx) in &trim {
            let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
            let wall = if c[0] > half_w {
                0
            } else if c[0] < -half_w {
                1
            } else if c[2] > half_d {
                2
            } else if c[2] < -half_d {
                3
            } else {
                panic!("trim component centroid {c:?} is not on any facade plane");
            };
            if mx[1] - mn[1] > 1.0 {
                per_wall[wall].0 += 1; // full-height vertical = mullion
            } else {
                per_wall[wall].1 += 1; // thin horizontal run = transom
            }
        }
        let mullions: usize = per_wall.iter().map(|w| w.0).sum();
        let transoms: usize = per_wall.iter().map(|w| w.1).sum();
        let cells: usize = per_wall
            .iter()
            .map(|&(nv, nt)| nv.saturating_sub(1) * nt.saturating_sub(1))
            .sum();
        let grid_pass = mullions == exp_mullions && transoms == exp_transoms && cells == exp_cells;
        eprintln!(
            "[curtain] grid: {mullions} mullions x {transoms} transoms = {cells} cells \
             (expect {exp_cells} == bays*floors*3: {bays_total} bays x {floors} floors x \
             3 panes_per) -> {}",
            if grid_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            grid_pass,
            "curtain-wall grid is not the parametric expectation: \
             {mullions}/{exp_mullions} mullions, {transoms}/{exp_transoms} transoms, \
             {cells}/{exp_cells} cells (per-wall {per_wall:?})"
        );

        // ── GATE 2: vision glass transmissive + spandrels opaque. ────────────
        let glass_tris = mesh.material_ids.iter().filter(|&&m| m == 2).count();
        let glass_panels = glass_tris / 2;
        let glass_mat = materials[2];
        let packed_glass = super::pack_vulkan_mesh_material(glass_mat);
        let glass_pass = glass_panels == exp_glass
            && glass_mat.transmission > 0.0
            && glass_mat.thin_walled
            && packed_glass[0].to_bits() == 3;
        eprintln!(
            "[curtain] vision glass panels transmissive: {glass_panels} panels, MAT_GLASS -> {}",
            if glass_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            glass_pass,
            "vision glass is not cooked transmissive: {glass_panels}/{exp_glass} panels, \
             transmission {}, thin_walled {}, packed type {} (the cook must tag the \
             curtain-wall glass channel transmission > 0)",
            glass_mat.transmission,
            glass_mat.thin_walled,
            packed_glass[0].to_bits()
        );
        // Spandrels: MAT_WALL (forge id 0) panels lying IN a facade plane
        // (every vertex within 2 cm of one wall plane, above grade) — the
        // underside seal and interior proxies live elsewhere.
        let mut spandrel_tris = 0usize;
        for (t, tri) in mesh.indices.iter().enumerate() {
            if mesh.material_ids[t] != 0 {
                continue;
            }
            let pts: Vec<[f32; 3]> = tri.iter().map(|&i| mesh.positions[i as usize]).collect();
            let in_plane = |f: &dyn Fn(&[f32; 3]) -> f32| pts.iter().all(|p| f(p).abs() < 0.02);
            let on_facade = in_plane(&|p: &[f32; 3]| p[0] - half_w)
                || in_plane(&|p: &[f32; 3]| p[0] + half_w)
                || in_plane(&|p: &[f32; 3]| p[2] - half_d)
                || in_plane(&|p: &[f32; 3]| p[2] + half_d);
            let above_grade = pts.iter().map(|p| p[1]).fold(f32::MIN, f32::max) > 0.01;
            if on_facade && above_grade {
                spandrel_tris += 1;
            }
        }
        let spandrels = spandrel_tris / 2;
        let wall_mat = materials[0];
        let packed_wall = super::pack_vulkan_mesh_material(wall_mat);
        let spandrel_pass = spandrels == exp_spandrels
            && wall_mat.transmission == 0.0
            && packed_wall[0].to_bits() == 1;
        eprintln!(
            "[curtain] spandrel panels opaque: {spandrels} -> {}",
            if spandrel_pass { "PASS" } else { "FAIL" }
        );
        assert!(
            spandrel_pass,
            "spandrels wrong: {spandrels}/{exp_spandrels} panels, wall transmission {}, \
             packed type {}",
            wall_mat.transmission,
            packed_wall[0].to_bits()
        );

        // ── GATE 3: facade detail energy vs a flat Lambert box of the same
        //    dimensions (same camera, rig, resolution, spp). CHEAP renders. ──
        let (w, h) = (256u32, 256u32);
        let spp = 32u32;
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [26.0f32, 16.0, 34.0];
        let target = [0.0f32, total_h * 0.45, 0.0];
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };

        let t0 = std::time::Instant::now();
        let rgba_tower = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &materials,
            &textures,
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("curtain-wall tower render should succeed");
        let secs_tower = t0.elapsed().as_secs_f64();

        // Flat-box control: the same envelope as ONE Lambert box (the cooked
        // facade albedo, no textures, no grid, no glass).
        let mut box_mesh = CookedMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_ids: Vec::new(),
        };
        let mut push_quad = |corners: [[f32; 3]; 4], normal: [f32; 3]| {
            let v0 = box_mesh.positions.len() as u32;
            for c in corners {
                box_mesh.positions.push(c);
                box_mesh.normals.push(normal);
                box_mesh.uvs.push([0.0, 0.0]);
            }
            box_mesh.indices.push([v0, v0 + 1, v0 + 2]);
            box_mesh.indices.push([v0, v0 + 2, v0 + 3]);
            box_mesh.material_ids.extend_from_slice(&[0, 0]);
        };
        let (hw, hd, th) = (half_w, half_d, total_h);
        push_quad(
            [[-hw, 0.0, hd], [hw, 0.0, hd], [hw, th, hd], [-hw, th, hd]],
            [0.0, 0.0, 1.0],
        );
        push_quad(
            [[hw, 0.0, -hd], [-hw, 0.0, -hd], [-hw, th, -hd], [hw, th, -hd]],
            [0.0, 0.0, -1.0],
        );
        push_quad(
            [[hw, 0.0, hd], [hw, 0.0, -hd], [hw, th, -hd], [hw, th, hd]],
            [1.0, 0.0, 0.0],
        );
        push_quad(
            [[-hw, 0.0, -hd], [-hw, 0.0, hd], [-hw, th, hd], [-hw, th, -hd]],
            [-1.0, 0.0, 0.0],
        );
        push_quad(
            [[-hw, th, hd], [hw, th, hd], [hw, th, -hd], [-hw, th, -hd]],
            [0.0, 1.0, 0.0],
        );
        let box_materials = [PbrMaterial {
            base_color: wall_mat.base_color,
            roughness: wall_mat.roughness,
            ..Default::default()
        }];
        let t1 = std::time::Instant::now();
        let rgba_box = pathtrace_mesh_lit_to_rgba(
            &box_mesh.positions,
            &box_mesh.normals,
            &box_mesh.uvs,
            &box_mesh.indices,
            &box_mesh.material_ids,
            &box_materials,
            &[],
            eye,
            target,
            fov_y,
            w,
            h,
            spp,
            &rig,
        )
        .expect("flat-box control render should succeed");
        let secs_box = t1.elapsed().as_secs_f64();

        let coverage_mask = |m: &CookedMesh| -> (Vec<bool>, usize) {
            let (mats_px, _) = rasterize_material_masks(m, eye, target, fov_y, w, h);
            let mask: Vec<bool> = mats_px.iter().map(|&v| v >= 0).collect();
            let n = mask.iter().filter(|&&b| b).count();
            (mask, n)
        };
        let (tower_mask, tower_px) = coverage_mask(&mesh);
        let (box_mask, box_px) = coverage_mask(&box_mesh);
        assert!(
            tower_px > 5000 && box_px > 5000,
            "camera framing drifted: tower {tower_px} px, box {box_px} px coverage"
        );
        let e_tower = masked_sobel_energy(&rgba_tower, &tower_mask, w, h);
        let e_box = masked_sobel_energy(&rgba_box, &box_mask, w, h);
        let energy_ratio = e_tower / e_box.max(1e-9);
        let energy_pass = energy_ratio >= 3.0;
        eprintln!(
            "[curtain] facade detail energy {energy_ratio:.2}x a flat-box control \
             (gate >= 3.0x) -> {} (tower sobel {e_tower:.3}, box sobel {e_box:.3}, \
             {tower_px}/{box_px} px)",
            if energy_pass { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "[curtain] render time {secs_tower:.2}s per frame (expect < 10s) \
             [control {secs_box:.2}s]"
        );

        let out_dir = std::env::temp_dir();
        let force_alpha = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        let tower_png = out_dir.join("curtain_wall_office.png");
        let box_png = out_dir.join("curtain_wall_flatbox.png");
        write_png_rgba(tower_png.to_str().unwrap(), &force_alpha(rgba_tower), w, h);
        write_png_rgba(box_png.to_str().unwrap(), &force_alpha(rgba_box), w, h);
        eprintln!("[curtain] wrote {} (curtain-wall office)", tower_png.display());
        eprintln!("[curtain] wrote {} (flat-box control)", box_png.display());
        eprintln!(
            "[curtain] eyeball: a glass office tower — vertical mullions + floor-line/\
             mid-floor transoms gridding every facade, opaque spandrel bands at the \
             slabs, transmissive vision glass showing the interior plates — vs a \
             featureless Lambert box"
        );

        assert!(
            energy_pass,
            "the curtain-wall facade does not read as structured vs a plain box \
             (sobel ratio {energy_ratio:.2}x < 3.0x)"
        );
        assert!(
            secs_tower.min(secs_box) < 10.0,
            "gate renders are not cheap: {secs_tower:.2}s / {secs_box:.2}s per frame"
        );
    }

    /// Signed solid angle of triangle (p0,p1,p2) from `q` (van Oosterom–
    /// Strackee) — the same oracle the forge seal tests and the cook's GWN
    /// sign gate use.
    #[cfg(feature = "spectra-native")]
    fn tri_solid_angle(q: glam::DVec3, p0: glam::DVec3, p1: glam::DVec3, p2: glam::DVec3) -> f64 {
        let (a, b, c) = (p0 - q, p1 - q, p2 - q);
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        if la < 1e-12 || lb < 1e-12 || lc < 1e-12 {
            return 0.0;
        }
        let det = a.dot(b.cross(c));
        let denom = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        2.0 * det.atan2(denom)
    }

    /// Generalized winding number of a cooked mesh at `q`.
    #[cfg(feature = "spectra-native")]
    fn cooked_gwn(mesh: &CookedMesh, q: glam::DVec3) -> f64 {
        let mut sum = 0.0;
        for tri in &mesh.indices {
            let p0 = glam::DVec3::from(mesh.positions[tri[0] as usize].map(f64::from));
            let p1 = glam::DVec3::from(mesh.positions[tri[1] as usize].map(f64::from));
            let p2 = glam::DVec3::from(mesh.positions[tri[2] as usize].map(f64::from));
            sum += tri_solid_angle(q, p0, p1, p2);
        }
        sum / (4.0 * std::f64::consts::PI)
    }

    /// Replica of the cook's GWN closed gate (`SdfBaker::bake`): 4x4x4
    /// stratified jittered probes (SplitMix stream 0xC001_BA5E, bit-for-bit),
    /// |w| in (0.15, 0.85) anywhere => OPEN (Err carries the min fractional
    /// |w|); otherwise Ok(interior probe count), which must be >= 1 for the
    /// Closed sign to be provable. This is the gate that decides
    /// `sign=gwn closed=yes` at recook time.
    #[cfg(feature = "spectra-native")]
    fn cooked_probe_gate(mesh: &CookedMesh) -> Result<usize, f32> {
        struct SplitMix(u64);
        impl SplitMix {
            fn unit(&mut self) -> f32 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u64 << 24) as f32
            }
        }
        let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &mesh.positions {
            for a in 0..3 {
                mn[a] = mn[a].min(p[a]);
                mx[a] = mx[a].max(p[a]);
            }
        }
        let extent = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
        let mut rng = SplitMix(0xC001_BA5E);
        let (mut interior, mut min_abs, mut fractional) = (0usize, f32::INFINITY, false);
        for cz in 0..4u32 {
            for cy in 0..4u32 {
                for cx in 0..4u32 {
                    let u = (cx as f32 + rng.unit()) / 4.0;
                    let v = (cy as f32 + rng.unit()) / 4.0;
                    let s = (cz as f32 + rng.unit()) / 4.0;
                    let q = glam::DVec3::new(
                        (mn[0] + u * extent[0]) as f64,
                        (mn[1] + v * extent[1]) as f64,
                        (mn[2] + s * extent[2]) as f64,
                    );
                    let w = cooked_gwn(mesh, q).abs() as f32;
                    if w < 0.15 {
                        continue;
                    }
                    min_abs = min_abs.min(w);
                    if w <= 0.85 {
                        fractional = true;
                    } else {
                        interior += 1;
                    }
                }
            }
        }
        if fractional {
            return Err(min_abs);
        }
        if interior == 0 {
            return Err(0.0);
        }
        Ok(interior)
    }

    /// Per-floor-slice skin width profile of a cooked building: max X extent
    /// of vertical MAT_WALL(0) / MAT_GLASS(2) skin triangles per slice.
    /// Mullions/cornices/roofs are excluded by material, horizontal sheets by
    /// normal — what remains IS the massing silhouette the shape gates
    /// measure. Returns (slice widths, building height).
    #[cfg(feature = "spectra-native")]
    fn skin_width_profile(mesh: &CookedMesh, slices: usize) -> (Vec<f32>, f32) {
        use glam::Vec3;
        let height = mesh
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut min_x = vec![f32::INFINITY; slices];
        let mut max_x = vec![f32::NEG_INFINITY; slices];
        for (t, tri) in mesh.indices.iter().enumerate() {
            let m = mesh.material_ids[t];
            if m != 0 && m != 2 {
                continue;
            }
            let p0 = Vec3::from(mesh.positions[tri[0] as usize]);
            let p1 = Vec3::from(mesh.positions[tri[1] as usize]);
            let p2 = Vec3::from(mesh.positions[tri[2] as usize]);
            let g = (p2 - p0).cross(p1 - p0);
            if g.length() < 1e-9 || g.normalize().y.abs() > 0.5 {
                continue; // horizontal sheet (roof cap, underside, plate)
            }
            let cy = (p0.y + p1.y + p2.y) / 3.0;
            let s = ((cy / height) * slices as f32).floor() as usize;
            if s >= slices {
                continue;
            }
            for p in [p0, p1, p2] {
                min_x[s] = min_x[s].min(p.x);
                max_x[s] = max_x[s].max(p.x);
            }
        }
        let widths = (0..slices)
            .map(|s| {
                if max_x[s] > min_x[s] {
                    max_x[s] - min_x[s]
                } else {
                    0.0
                }
            })
            .collect();
        (widths, height)
    }

    /// Count massing volumes from a width profile: a new volume starts where
    /// the slice width steps by more than 1 m (empty seam slices are skipped).
    #[cfg(feature = "spectra-native")]
    fn count_width_plateaus(widths: &[f32]) -> usize {
        let mut plateaus = 0usize;
        let mut last: Option<f32> = None;
        for &w in widths {
            if w <= 0.0 {
                continue;
            }
            if last.is_none_or(|l| (w - l).abs() > 1.0) {
                plateaus += 1;
            }
            last = Some(w);
        }
        plateaus
    }

    /// Normalized front-silhouette mask of a cooked building: the CPU
    /// rasterizer (the GPU-cross-checked projection) with a camera fitted so
    /// every building fills the frame the same way — pairwise mask deltas
    /// then measure SHAPE, not size.
    #[cfg(feature = "spectra-native")]
    fn fitted_silhouette(mesh: &CookedMesh, res: u32) -> Vec<bool> {
        let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &mesh.positions {
            for a in 0..3 {
                mn[a] = mn[a].min(p[a]);
                mx[a] = mx[a].max(p[a]);
            }
        }
        let centre = [
            (mn[0] + mx[0]) * 0.5,
            (mn[1] + mx[1]) * 0.5,
            (mn[2] + mx[2]) * 0.5,
        ];
        let half = ((mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2])) * 0.5;
        let fov_y = std::f32::consts::FRAC_PI_4;
        let dist = half / (fov_y * 0.5).tan() * 1.25;
        let eye = [centre[0], centre[1], centre[2] + dist];
        let (mats_px, _) = rasterize_material_masks(mesh, eye, centre, fov_y, res, res);
        mats_px.iter().map(|&v| v >= 0).collect()
    }

    /// MASSING-AXIS GATES (grammar §4.2 wave): podium+tower + setback +
    /// box, all loaded from COOKED payloads (`assets/buildings/massing`,
    /// cooked by game_asset_cook through the massing-aware forge) and
    /// measured from geometry, then rendered CHEAP (256² @ 32 spp).
    ///
    ///   (volumes) podium_tower stacks exactly 2 skin-width plateaus —
    ///       podium floors / tower floors / inset / total height measured
    ///       from the mesh against the payload's own forge_description;
    ///   (watertight) podium_tower + setback + box all pass the cook's GWN
    ///       64-probe closed gate (bit-for-bit replica), and the box payload
    ///       mesh is BYTE-IDENTICAL to the pre-massing cook of the same
    ///       directive (assets/buildings/curtain_wall, cooked on master);
    ///   (facade) the tower band is glass-dominant curtain wall with the
    ///       cooked glass material transmissive (packs MAT_GLASS=3); the
    ///       podium band is wall-dominant punched window;
    ///   (shape) pairwise fitted-silhouette deltas between podium_tower,
    ///       setback and box all exceed 8% — the axis spans 3 distinct
    ///       shapes;
    ///   (render) three PNGs, each frame < 10 s.
    ///
    /// Cook first (civitas repo, ~/Ochroma/projects/civitas_care):
    ///   mkdir -p /tmp/massing_src/office && cp assets/source/buildings/office/\
    ///     {podium_tower_01,setback_tower_01,glass_office_tower_01}.asset.json \
    ///     /tmp/massing_src/office/ && mkdir -p assets/buildings/massing/textures \
    ///     && cp -r assets/buildings/forge_starter/textures/polyhaven \
    ///     assets/buildings/massing/textures/
    ///   cargo build --release --bin game_asset_cook && \
    ///   GAME_FORGE_BIN=$HOME/src/forge/target/release/aetherspectra-forge \
    ///     ./target/release/game_asset_cook --no-starters \
    ///     --source /tmp/massing_src --output assets/buildings/massing
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib forge_massing_podium_tower -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_massing_podium_tower() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};

        let home = std::path::PathBuf::from(std::env::var("HOME").unwrap());
        let massing_atoms = home.join("Ochroma/projects/civitas_care/assets/buildings/massing/atoms");
        let pt_path = massing_atoms.join("city.office.l8.4x4.podium_tower_01.atoms.json");
        let sb_path = massing_atoms.join("city.office.l6.3x3.setback_tower_01.atoms.json");
        let box_path = massing_atoms.join("city.office.l5.3x3.glass_office_tower_01.atoms.json");
        let box_pre_path = home.join(
            "Ochroma/projects/civitas_care/assets/buildings/curtain_wall/atoms/\
             city.office.l5.3x3.glass_office_tower_01.atoms.json",
        );

        let pt = load_craftsman_mesh(&pt_path);
        let sb = load_craftsman_mesh(&sb_path);
        let bx = load_craftsman_mesh(&box_path);
        let bx_pre = load_craftsman_mesh(&box_pre_path);

        // ── GATE 1: podium_tower volumes / floors / inset / height, measured
        //    from the cooked mesh against its own forge_description. ─────────
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&pt_path).unwrap()).expect("parse atoms.json");
        let desc = &json["forge_description"];
        assert_eq!(desc["massing_mode"].as_str(), Some("podium_tower"));
        let pf = desc["podium_floors"].as_u64().unwrap() as u32;
        let tf = desc["tower_floors"].as_u64().unwrap() as u32;
        let fh = desc["floor_height"].as_f64().unwrap() as f32;
        let plot_w = desc["footprint"]["width"].as_f64().unwrap() as f32;
        let floors = desc["floors"].as_u64().unwrap() as u32;
        assert_eq!(pf + tf, floors, "floor split must tile the height");

        let slices = floors as usize;
        let (pt_widths, pt_height) = skin_width_profile(&pt, slices);
        let pt_volumes = count_width_plateaus(&pt_widths);
        let podium_w: f32 = pt_widths[..pf as usize]
            .iter()
            .fold(0.0f32, |m, &w| m.max(w));
        // Tower width away from the seam slice.
        let tower_w: f32 = pt_widths[(pf as usize + 1)..]
            .iter()
            .filter(|w| **w > 0.0)
            .fold(0.0f32, |m, &w| m.max(w));
        let inset_m = (podium_w - tower_w) * 0.5;
        let inset_pct = 100.0 * inset_m / podium_w;
        let gate1 = pt_volumes == 2
            && inset_m > 0.5
            && tf > pf
            && (podium_w - plot_w).abs() < 0.1
            && pt_height >= floors as f32 * fh
            && pt_height < floors as f32 * fh + 1.0;
        eprintln!(
            "[massing] podium_tower volumes: {pt_volumes} (podium {pf} floors, tower {tf} \
             floors); tower footprint inset = {inset_pct:.1}% of podium ({inset_m:.2} m, \
             podium {podium_w:.2} m -> tower {tower_w:.2} m) (expect inset>0, tf>pf, total \
             height {pt_height:.1}m) -> {}",
            if gate1 { "PASS" } else { "FAIL" }
        );
        assert!(
            gate1,
            "podium_tower massing wrong: {pt_volumes} volumes, inset {inset_m} m, \
             pf {pf} tf {tf}, height {pt_height} vs {} floors x {fh} m",
            floors
        );

        // ── GATE 2: watertight — the cook's GWN closed gate on all three,
        //    plus box byte-identity against the pre-massing cook. ────────────
        let gwn_label = |r: &Result<usize, f32>| match r {
            Ok(n) => format!("gwn yes ({n} interior probes)"),
            Err(w) => format!("NO (fractional |w|={w:.3})"),
        };
        let pt_gate = cooked_probe_gate(&pt);
        let sb_gate = cooked_probe_gate(&sb);
        let bx_gate = cooked_probe_gate(&bx);
        let identical = pt_eq_mesh(&bx, &bx_pre);
        let gate2 =
            pt_gate.is_ok() && sb_gate.is_ok() && bx_gate.is_ok() && identical;
        eprintln!(
            "[massing] watertight: podium_tower closed={}, setback closed={}, box-regression \
             closed={} & byte-identical to pre-massing ({} tris) = {identical} -> {}",
            gwn_label(&pt_gate),
            gwn_label(&sb_gate),
            gwn_label(&bx_gate),
            bx.indices.len(),
            if gate2 { "PASS" } else { "FAIL" }
        );
        assert!(gate2, "watertight/regression gate failed");

        // ── GATE 3: per-volume facade — transmissive curtain-wall tower over
        //    a punched podium, measured from band areas + the packed
        //    material. ─────────────────────────────────────────────────────────
        let (materials, textures, channels) = load_building_mesh_pbr_by_forge_id(&pt_path);
        assert_eq!(channels[2], "glass", "forge id 2 must bind the glass channel");
        let podium_top = pf as f32 * fh;
        let total_h = floors as f32 * fh;
        let band_area = |y0: f32, y1: f32| -> (f64, f64) {
            use glam::Vec3;
            let (mut glass, mut wall) = (0.0f64, 0.0f64);
            for (t, tri) in pt.indices.iter().enumerate() {
                let m = pt.material_ids[t];
                if m != 0 && m != 2 {
                    continue;
                }
                let p0 = Vec3::from(pt.positions[tri[0] as usize]);
                let p1 = Vec3::from(pt.positions[tri[1] as usize]);
                let p2 = Vec3::from(pt.positions[tri[2] as usize]);
                let g = (p2 - p0).cross(p1 - p0);
                if g.length() < 1e-9 || g.normalize().y.abs() > 0.5 {
                    continue;
                }
                let cy = (p0.y + p1.y + p2.y) / 3.0;
                if cy < y0 || cy > y1 {
                    continue;
                }
                let area = (g.length() * 0.5) as f64;
                if m == 2 {
                    glass += area;
                } else {
                    wall += area;
                }
            }
            (glass, wall)
        };
        let (tower_glass, tower_wall) = band_area(podium_top + 1.0, total_h - 1.0);
        let (podium_glass, podium_wall) = band_area(0.5, podium_top - 0.5);
        let tower_glass_tris = pt
            .indices
            .iter()
            .zip(&pt.material_ids)
            .filter(|(tri, m)| {
                **m == 2 && {
                    let cy = tri
                        .iter()
                        .map(|&i| pt.positions[i as usize][1])
                        .sum::<f32>()
                        / 3.0;
                    cy > podium_top
                }
            })
            .count();
        let glass_mat = materials[2];
        let packed = super::pack_vulkan_mesh_material(glass_mat);
        let gate3 = tower_glass_tris > 0
            && glass_mat.transmission > 0.0
            && glass_mat.thin_walled
            && packed[0].to_bits() == 3
            && tower_glass > tower_wall
            && podium_glass < podium_wall * 0.3;
        eprintln!(
            "[massing] tower facade = curtain_wall (transmissive glass panels \
             {} > 0, MAT_GLASS packed, band glass {tower_glass:.0} m^2 > wall \
             {tower_wall:.0} m^2), podium facade = punched (glass {podium_glass:.0} m^2 << \
             wall {podium_wall:.0} m^2) -> {}",
            tower_glass_tris / 2,
            if gate3 { "PASS" } else { "FAIL" }
        );
        assert!(
            gate3,
            "per-volume facade wrong: tower glass tris {tower_glass_tris}, transmission {}, \
             packed {}, tower {tower_glass}/{tower_wall}, podium {podium_glass}/{podium_wall}",
            glass_mat.transmission,
            packed[0].to_bits()
        );

        // ── GATE 4: the axis spans 3 distinct shapes — pairwise fitted-
        //    silhouette deltas. ───────────────────────────────────────────────
        let res = 160u32;
        let masks = [
            ("podium_tower", fitted_silhouette(&pt, res)),
            ("setback", fitted_silhouette(&sb, res)),
            ("box", fitted_silhouette(&bx, res)),
        ];
        let mut min_delta = f64::INFINITY;
        let mut deltas = Vec::new();
        for i in 0..masks.len() {
            for j in (i + 1)..masks.len() {
                let (xor, union) = masks[i]
                    .1
                    .iter()
                    .zip(&masks[j].1)
                    .fold((0usize, 0usize), |(x, u), (&a, &b)| {
                        (x + (a != b) as usize, u + (a || b) as usize)
                    });
                let delta = xor as f64 / union.max(1) as f64;
                min_delta = min_delta.min(delta);
                deltas.push(format!("{} vs {} = {:.1}%", masks[i].0, masks[j].0, delta * 100.0));
            }
        }
        let gate4 = min_delta > 0.08;
        eprintln!(
            "[massing] shape distinct: pairwise silhouette delta {} (gate > 8%) -> {}",
            deltas.join(", "),
            if gate4 { "PASS" } else { "FAIL" }
        );
        assert!(gate4, "silhouettes are not distinct: min delta {min_delta:.3}");

        // ── GATE 5: cheap renders — a human eyeballs glass-on-a-base, the
        //    ziggurat, and the plain box. ─────────────────────────────────────
        let (w, h) = (256u32, 256u32);
        let spp = 32u32;
        let fov_y = std::f32::consts::FRAC_PI_4;
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let out_dir = std::env::temp_dir();
        let mut worst_secs = 0.0f64;
        for (label, mesh, path) in [
            ("massing_podium_tower", &pt, &pt_path),
            ("massing_setback", &sb, &sb_path),
            ("massing_box", &bx, &box_path),
        ] {
            let (mats, texs, _) = load_building_mesh_pbr_by_forge_id(path);
            let max_y = mesh
                .positions
                .iter()
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max);
            let max_x = mesh
                .positions
                .iter()
                .map(|p| p[0].abs())
                .fold(0.0f32, f32::max);
            let half = (max_y * 0.5).max(max_x);
            let dist = half / (fov_y * 0.5).tan() * 1.35;
            let dir = glam::Vec3::new(0.8, 0.42, 1.0).normalize();
            let target = [0.0f32, max_y * 0.48, 0.0];
            let eye = [
                target[0] + dir.x * dist,
                target[1] + dir.y * dist,
                target[2] + dir.z * dist,
            ];
            let t0 = std::time::Instant::now();
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                &mats,
                &texs,
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                &rig,
            )
            .unwrap_or_else(|e| panic!("{label} render failed: {e}"));
            let secs = t0.elapsed().as_secs_f64();
            worst_secs = worst_secs.max(secs);
            let mut rgba = rgba;
            for px in rgba.chunks_exact_mut(4) {
                px[3] = 255;
            }
            let png = out_dir.join(format!("{label}.png"));
            write_png_rgba(png.to_str().unwrap(), &rgba, w, h);
            eprintln!("[massing] wrote {} ({secs:.2}s)", png.display());
        }
        eprintln!(
            "[massing] render time {worst_secs:.2}s/frame (< 10s) -> {}",
            if worst_secs < 10.0 { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "[massing] eyeball: a glass curtain-wall tower seated on a wider punched-window \
             podium (parapet crown), a three-band stepped ziggurat, and the plain 8-floor box"
        );
        assert!(
            worst_secs < 10.0,
            "gate renders are not cheap: {worst_secs:.2}s/frame"
        );
    }

    /// FACADE MATERIALS ACCEPTANCE — part 1: the per-surface zoning fix,
    /// render-proven on the craftsman.
    ///
    /// The historical bug: the mesh's `material_ids` are FORGE channel ids
    /// (0 wall, 1 roof, 2 glass, 3 reveal, 4 trim, 5 cornice, 6 door), but the
    /// deleted positional loader fed the cooked SLOT-ordered material list to
    /// the kernel, so the 1304 window/door FRAME triangles (forge id 4)
    /// indexed cooked slot 4 = `ground_concrete`, whose diffuse map is
    /// `castle_brick_02_red` — red-brick mortar lines on every frame. The fix
    /// is `load_building_mesh_pbr_by_forge_id` (table indexed by forge id).
    ///
    /// Gates (each render 256x256 @ 32 spp — CHEAP, seconds):
    ///   (loader) the forge-id table binds [2]=glass, [3]=[4]=[5]=trim,
    ///       [6]=door, with one shared trim texture distinct from the facade's;
    ///       the cooked slot list really does carry "ground" at slot 4 (the
    ///       cross-wire this fix removes);
    ///   (was present) the cross-wired control render's frame pixels carry the
    ///       dark brick-tinted ground material: frame mean |Δrgb| cross-wired
    ///       vs fixed > 0.10 and the fixed frame is brighter by > 0.10 luma
    ///       (stucco trim vs the [0.34,0.33,0.30]-tinted brick; measured
    ///       0.146 / 0.144 against a 0.007 wall noise floor);
    ///   (absent) the FIXED render's frame mean lands on the trim material's
    ///       own flat-control mean (|Δrgb| < 0.06 — the mean-normalizing tint
    ///       makes textured mean == flat mean on the same material) and is
    ///       closer to the flat TRIM tone than to the flat FACADE tone —
    ///       the frame is trim-material, not facade-material;
    ///   (time) every gate render < 10 s.
    /// Writes `materials_zoning_craftsman.png` (the fixed render) plus the
    /// cross-wired control for eyeballing.
    ///
    /// Run alone (GPU):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --profile release-fast --lib forge_facade_materials -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_facade_materials() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig, PbrMaterial};

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");
        let mesh = load_craftsman_mesh(&asset_path);

        // FIXED: the canonical forge-id-indexed table. CONTROL: the cooked
        // slot-ordered list exactly as the deleted positional loader returned
        // it (transmission stripped) — the cross-wire being proven gone.
        let (fixed, ftex, fchan) = load_building_mesh_pbr_by_forge_id(&asset_path);
        let (mut wired, wtex, wchan) = load_cooked_pbr_materials(&asset_path);
        for m in &mut wired {
            m.transmission = 0.0;
            m.ior = 1.5;
            m.thin_walled = false;
        }

        // ── Loader-truth gate: forge ids bind their channels; slot 4 of the
        //    cooked list is the GROUND material the frames used to index. ────
        assert_eq!(fchan[0], "facade");
        assert_eq!(fchan[2], "glass", "forge id 2 must bind the glass channel");
        assert_eq!(fchan[3], "trim", "forge id 3 (reveal) must bind trim");
        assert_eq!(fchan[4], "trim", "forge id 4 (frames) must bind trim");
        assert_eq!(fchan[6], "door", "forge id 6 must bind the door channel");
        assert_eq!(
            wchan[4], "ground",
            "cooked slot 4 is no longer the ground material — the historical \
             cross-wire this test documents has changed shape; re-derive the control"
        );
        assert_eq!(
            fixed[3].albedo_tex, fixed[4].albedo_tex,
            "reveal and frame must share the one trim texture"
        );
        assert_ne!(
            fixed[4].albedo_tex, fixed[0].albedo_tex,
            "frame (trim) texture must be distinct from the facade texture"
        );
        assert_ne!(
            fixed[4].albedo_tex, -1,
            "trim material must carry a real albedo map"
        );
        eprintln!(
            "[materials] loader truth: forge-id table [2]=glass [3]=[4]=[5]=trim [6]=door; \
             cooked slot 4 = '{}' (the old frames' material, diffuse '{}') -> PASS",
            wchan[4], "castle_brick_02_red"
        );

        // ── Frontal window framing (the glass-demo camera): facade windows +
        //    porch fill the view so the frame strips get real pixel counts. ──
        let (w, h) = (256u32, 256u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let target = [0.0f32, 2.6, 0.0];
        let eye = [0.5f32, 2.4, 14.0];
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let spp = 32u32;
        let mut worst_secs = 0.0f64;
        let mut render = |mats: &[PbrMaterial], texs: &[super::TextureImage]| -> Vec<u8> {
            let t0 = std::time::Instant::now();
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                mats,
                texs,
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                &rig,
            )
            .expect("craftsman zoning render should succeed");
            worst_secs = worst_secs.max(t0.elapsed().as_secs_f64());
            rgba
        };
        let rgba_fix = render(&fixed, &ftex);
        let rgba_bug = render(&wired, &wtex);
        // Flat control on the FIXED table: pure per-forge-id albedos under the
        // same lighting — the "what does the trim material look like here"
        // reference the absent-gate compares against.
        let flat: Vec<PbrMaterial> = fixed
            .iter()
            .map(|m| PbrMaterial {
                albedo_tex: -1,
                roughness_tex: -1,
                normal_tex: -1,
                ..*m
            })
            .collect();
        let rgba_flat = render(&flat, &[]);

        // ── Per-forge-id pixel masks (same pinhole as the GPU). Frame strips
        //    are narrow at 256², so the frame mask is NOT eroded (the means
        //    below average hundreds of pixels; both renders share the exact
        //    same mask, so edge pixels cancel in the comparison). ─────────────
        let (mats_px, _tris_px) = rasterize_material_masks(&mesh, eye, target, fov_y, w, h);
        let mask_of = |id: i32| -> Vec<bool> { mats_px.iter().map(|&m| m == id).collect() };
        let frame_mask = mask_of(4);
        let wall_mask = mask_of(0);
        let n_frame = frame_mask.iter().filter(|&&b| b).count();
        let n_wall = wall_mask.iter().filter(|&&b| b).count();
        assert!(
            n_frame >= 150,
            "frame mask too small ({n_frame} px < 150) — camera drifted off the windows"
        );
        assert!(n_wall >= 2000, "wall mask too small ({n_wall} px < 2000)");

        let mean_rgb = |rgba: &[u8], mask: &[bool]| -> [f64; 3] {
            let mut sum = [0.0f64; 3];
            let mut n = 0usize;
            for (p, &on) in mask.iter().enumerate() {
                if on {
                    for (s, &v) in sum.iter_mut().zip(&rgba[p * 4..p * 4 + 3]) {
                        *s += v as f64;
                    }
                    n += 1;
                }
            }
            sum.map(|s| s / n.max(1) as f64)
        };
        let split = |a: [f64; 3], b: [f64; 3]| -> f64 {
            ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / (3.0 * 255.0)
        };
        let luma =
            |c: [f64; 3]| -> f64 { (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0 };

        let f_fix = mean_rgb(&rgba_fix, &frame_mask);
        let f_bug = mean_rgb(&rgba_bug, &frame_mask);
        let f_flat = mean_rgb(&rgba_flat, &frame_mask);
        let w_fix = mean_rgb(&rgba_fix, &wall_mask);
        let w_bug = mean_rgb(&rgba_bug, &wall_mask);
        let w_flat = mean_rgb(&rgba_flat, &wall_mask);

        // (was present): the cross-wired frames really were a different,
        // darker (brick-tinted-ground) material. Measured 2026-06-12 on the
        // 780M/RADV: |Δrgb| 0.146, Δluma 0.144 against a 0.007 wall noise
        // floor — gates pinned at 0.10 (>14x the floor).
        let d_bug_fix = split(f_bug, f_fix);
        let d_luma = luma(f_fix) - luma(f_bug);
        let pass_present = d_bug_fix > 0.10 && d_luma > 0.10;
        // Walls are the SAME facade material in both renders — the fix must
        // not touch them (render noise only).
        let d_wall = split(w_bug, w_fix);
        let pass_wall_same = d_wall < 0.03;
        // (absent): the fixed frame lands on the trim material's own flat
        // mean and sits closer to flat-trim than to flat-facade.
        let d_fix_flat = split(f_fix, f_flat);
        let d_to_wall = split(f_fix, w_flat);
        let pass_absent = d_fix_flat < 0.06 && d_fix_flat < d_to_wall;
        // The number the gate line reports: how far the frame now sits from
        // the wall in the FIXED render (frames are their own material).
        let d_wall_frame = split(w_fix, f_fix);

        eprintln!(
            "[materials] cross-wired control: frame mean rgb [{:.0},{:.0},{:.0}] \
             (brick-tinted ground) vs fixed [{:.0},{:.0},{:.0}] (trim): \
             |Δrgb| {d_bug_fix:.3} (gate > 0.10), Δluma {d_luma:.3} (gate > 0.10); \
             wall unchanged |Δrgb| {d_wall:.3} (gate < 0.03)",
            f_bug[0], f_bug[1], f_bug[2], f_fix[0], f_fix[1], f_fix[2]
        );
        eprintln!(
            "[materials] fixed frame vs its trim flat-control |Δrgb| {d_fix_flat:.3} \
             (gate < 0.06); frame-to-flat-trim {d_fix_flat:.3} < frame-to-flat-facade \
             {d_to_wall:.3} ({n_frame} frame px, {n_wall} wall px)"
        );
        let pass_zoning = pass_present && pass_wall_same && pass_absent;
        eprintln!(
            "[materials] window/frame region brick-signature ABSENT (was present): \
             wall-vs-frame |Δrgb| now {d_wall_frame:.3}, frame is trim-material \
             not facade-material -> {}",
            if pass_zoning { "PASS" } else { "FAIL" }
        );

        let out_dir = std::env::temp_dir();
        let opaque = |mut v: Vec<u8>| -> Vec<u8> {
            for px in v.chunks_exact_mut(4) {
                px[3] = 255;
            }
            v
        };
        let fix_png = out_dir.join("materials_zoning_craftsman.png");
        let bug_png = out_dir.join("materials_zoning_craftsman_crosswired.png");
        write_png_rgba(fix_png.to_str().unwrap(), &opaque(rgba_fix.clone()), w, h);
        write_png_rgba(bug_png.to_str().unwrap(), &opaque(rgba_bug.clone()), w, h);
        eprintln!("[materials] wrote {} (fixed zoning)", fix_png.display());
        eprintln!(
            "[materials] wrote {} (cross-wired control — brick frames)",
            bug_png.display()
        );
        eprintln!(
            "[materials] render time {worst_secs:.2}s/frame (< 10s) -> {}",
            if worst_secs < 10.0 { "PASS" } else { "FAIL" }
        );

        assert!(
            pass_present,
            "the cross-wired control does not show the historical brick-on-frames \
             signature (|Δrgb| {d_bug_fix:.3}, Δluma {d_luma:.3}) — the control \
             no longer reproduces the bug; re-derive it before trusting the fix"
        );
        assert!(
            pass_wall_same,
            "the zoning fix changed the WALL ({d_wall:.3} >= 0.03) — it must only \
             re-route non-wall forge ids"
        );
        assert!(
            pass_absent,
            "the fixed render's frames do not carry the trim material \
             (frame-vs-flat-trim |Δrgb| {d_fix_flat:.3}, frame-vs-flat-facade \
             {d_to_wall:.3}) — the loader repoint is wrong; fix it, \
             do not relax this gate"
        );
        assert!(worst_secs < 10.0, "gate renders are not cheap: {worst_secs:.2}s");
    }

    /// Facade elevations: straight-on front views so the facade composition
    /// (window stacks, floor hierarchy, casings, trim-zoned roof edges) is
    /// actually inspectable. Denoised eyeball output, sanity-gated non-empty.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn facade_elevations() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};
        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let rig = LightRig {
            sun_dir: [0.35, 0.55, 0.75],
            sun_intensity: 2.4,
            sky_intensity: 0.35,
            camera_fill: 0.15,
            rim_fill: 0.0,
            sky_dome_intensity: 0.7,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let shots = [
            ("forge.house.craftsman", "facade_craftsman.png"),
            ("city.res_low.l1.2x2.tudor_cottage_01", "facade_tudor.png"),
            ("city.res_high.l5.2x2.highrise_point_tower_01", "facade_highrise.png"),
            ("city.com_reg.l4.3x4.modern_hotel_block_01", "facade_hotel.png"),
        ];
        for (id, png) in shots {
            let path = atoms_dir.join(format!("{id}.atoms.json"));
            let mesh = load_craftsman_mesh(&path);
            let (mats, texs, _) = load_building_mesh_pbr_by_forge_id(&path);
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for v in &mesh.positions {
                for a in 0..3 {
                    lo[a] = lo[a].min(v[a]);
                    hi[a] = hi[a].max(v[a]);
                }
            }
            // Straight-on elevation of the +Z (front) facade: pull back far
            // enough that the larger of width/height fits a 45-degree fov.
            let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
            let span = (hi[0] - lo[0]).max(hi[1] - lo[1]);
            let dist = span * 1.25 + (hi[2] - lo[2]) * 0.5;
            let eye = [c[0], c[1], hi[2] + dist];
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions, &mesh.normals, &mesh.uvs, &mesh.indices,
                &mesh.material_ids, &mats, &texs,
                eye, c, std::f32::consts::FRAC_PI_4, 512, 512, 32, &rig,
            )
            .expect("facade render");
            let lit = rgba
                .chunks_exact(4)
                .filter(|px| px[0] as u32 + px[1] as u32 + px[2] as u32 > 30)
                .count();
            assert!(lit > 20_000, "{id}: facade render nearly empty ({lit})");
            let mut pxv: Vec<[u8; 4]> =
                rgba.chunks_exact(4).map(|p| [p[0], p[1], p[2], 255]).collect();
            crate::denoiser::SpectralDenoiser::new(0.7).denoise(&mut pxv, 512, 512);
            let flat: Vec<u8> = pxv.into_iter().flatten().collect();
            let out = std::env::temp_dir().join(png);
            write_png_rgba(out.to_str().unwrap(), &flat, 512, 512);
            eprintln!("[facade] wrote {}", out.display());
        }
    }

    /// Showcase: render a roster of cooked buildings (each a different style
    /// family + kind) through the mesh path — visual proof of the F7 family
    /// routing (tudor/industrial textures) and the catalog breadth. No hard
    /// quality gates; sanity = each render is non-empty and distinct.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn showcase_building_roster() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};
        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let shots = [
            ("city.res_low.l1.2x2.tudor_cottage_01", "showcase_tudor.png"),
            ("city.ind_heavy.l2.5x5.heavy_factory_01", "showcase_factory.png"),
            ("city.res_high.l5.2x2.highrise_point_tower_01", "showcase_highrise.png"),
        ];
        for (id, png) in shots {
            let path = atoms_dir.join(format!("{id}.atoms.json"));
            if !path.exists() {
                panic!("showcase asset missing: {}", path.display());
            }
            let mesh = load_craftsman_mesh(&path);
            let (mats, texs, _) = load_building_mesh_pbr_by_forge_id(&path);
            // 3/4 aerial framing scaled to the building's bounds.
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for v in &mesh.positions {
                for a in 0..3 {
                    lo[a] = lo[a].min(v[a]);
                    hi[a] = hi[a].max(v[a]);
                }
            }
            let c = [
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ];
            let ext = ((hi[0] - lo[0]).powi(2)
                + (hi[1] - lo[1]).powi(2)
                + (hi[2] - lo[2]).powi(2))
            .sqrt();
            let eye = [c[0] + ext * 0.75, c[1] + ext * 0.45, c[2] + ext * 0.75];
            let rgba = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                &mats,
                &texs,
                eye,
                c,
                std::f32::consts::FRAC_PI_4,
                384,
                384,
                32,
                &rig,
            )
            .expect("showcase render");
            let lit = rgba
                .chunks_exact(4)
                .filter(|px| px[0] as u32 + px[1] as u32 + px[2] as u32 > 30)
                .count();
            assert!(lit > 10_000, "{id}: render nearly empty ({lit} lit px)");
            let mut pxv: Vec<[u8; 4]> = rgba
                .chunks_exact(4)
                .map(|p| [p[0], p[1], p[2], 255])
                .collect();
            crate::denoiser::SpectralDenoiser::new(0.7).denoise(&mut pxv, 384, 384);
            let flat: Vec<u8> = pxv.into_iter().flatten().collect();
            let out = std::env::temp_dir().join(png);
            write_png_rgba(out.to_str().unwrap(), &flat, 384, 384);
            eprintln!("[showcase] wrote {} ({lit} lit px)", out.display());
        }
    }

    /// Close-up visual proof for the zoning fix: the full-building gate frames
    /// are only a few pixels wide at 14 m, so the (measured, 14x-noise-floor)
    /// material change is invisible in those PNGs. This renders ONE window band
    /// up close with both material tables so a human can actually see the
    /// brick-tinted frames vs clean trim. No new gates — the measured gates
    /// live in `forge_facade_materials`; this writes the eyeball evidence.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_facade_zoning_closeup() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");
        let mesh = load_craftsman_mesh(&asset_path);

        let (fixed, ftex, _) = load_building_mesh_pbr_by_forge_id(&asset_path);
        let (mut wired, wtex, wchan) = load_cooked_pbr_materials(&asset_path);
        assert_eq!(
            wchan[4], "ground",
            "control derivation changed — see forge_facade_materials"
        );
        for m in &mut wired {
            m.transmission = 0.0;
            m.ior = 1.5;
            m.thin_walled = false;
        }

        // Self-aiming close-up: centroid of the FRONT-face window glass (forge
        // id 2, verts near max-Z), camera pulled straight back from it. No
        // hand-guessed coordinates — the sanity assert below stays the proof
        // the window band really is in frame.
        // Side-wall window (max-X glass): the front facade's windows hide
        // behind the porch gable, so shoot the +X wall straight-on instead.
        let mut xmax = f32::NEG_INFINITY;
        for (t, tri) in mesh.indices.iter().enumerate() {
            if mesh.material_ids[t] != 2 {
                continue;
            }
            for &vi in tri {
                xmax = xmax.max(mesh.positions[vi as usize][0]);
            }
        }
        let mut c = [0.0f64; 3];
        let mut nv = 0usize;
        for (t, tri) in mesh.indices.iter().enumerate() {
            if mesh.material_ids[t] != 2 {
                continue;
            }
            for &vi in tri {
                let p = mesh.positions[vi as usize];
                if p[0] >= xmax - 0.3 {
                    for (a, &b) in c.iter_mut().zip(&p) {
                        *a += b as f64;
                    }
                    nv += 1;
                }
            }
        }
        assert!(nv > 0, "no side-wall glass verts found (id 2 near x={xmax})");
        let target = [
            (c[0] / nv as f64) as f32,
            (c[1] / nv as f64) as f32,
            (c[2] / nv as f64) as f32,
        ];
        let (w, h) = (448u32, 448u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [target[0] + 4.5, target[1], target[2]];
        let rig = LightRig {
            sun_dir: [0.45, 0.65, 0.55],
            sun_intensity: 2.6,
            sky_intensity: 0.3,
            camera_fill: 0.2,
            rim_fill: 0.0,
            sky_dome_intensity: 0.65,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.80, 0.86, 0.95],
            ..Default::default()
        };
        let spp = 48u32;
        let mut render = |mats: &[super::PbrMaterial], texs: &[super::TextureImage]| -> Vec<u8> {
            pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                mats,
                texs,
                eye,
                target,
                fov_y,
                w,
                h,
                spp,
                &rig,
            )
            .expect("closeup render should succeed")
        };
        let rgba_fix = render(&fixed, &ftex);
        let rgba_bug = render(&wired, &wtex);

        // Sanity: the window band must actually be in frame (frame material
        // id 4 present) — otherwise the camera drifted and the PNGs prove
        // nothing again.
        let (mats_px, _) = rasterize_material_masks(&mesh, eye, target, fov_y, w, h);
        let n_frame = mats_px.iter().filter(|&&m| m == 4).count();
        let n_glass = mats_px.iter().filter(|&&m| m == 2).count();
        assert!(
            n_frame >= 800 && n_glass >= 400,
            "closeup camera missed the window band (frame px {n_frame}, glass px {n_glass})"
        );

        let out_dir = std::env::temp_dir();
        // Edge-aware bilateral cleanup so the eyeball PNGs show materials, not
        // residual sample noise (the measured gates upstream stay raw).
        let opaque = |v: Vec<u8>| -> Vec<u8> {
            let mut px: Vec<[u8; 4]> = v
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], 255])
                .collect();
            crate::denoiser::SpectralDenoiser::new(0.7).denoise(&mut px, w, h);
            px.into_iter().flatten().collect()
        };
        let fix_png = out_dir.join("zoning_closeup_fixed.png");
        let bug_png = out_dir.join("zoning_closeup_crosswired.png");
        write_png_rgba(fix_png.to_str().unwrap(), &opaque(rgba_fix), w, h);
        write_png_rgba(bug_png.to_str().unwrap(), &opaque(rgba_bug), w, h);
        eprintln!(
            "[zoning-closeup] wrote {} and {} ({n_frame} frame px, {n_glass} glass px in view)",
            fix_png.display(),
            bug_png.display()
        );
    }


    /// Forge-facade wave 1, Task 1 gate (L/U/T cook unlock): render the cooked
    /// L-SHAPED walkup (`city.res_med.l3.3x4.lshape_walkup`) through the proven
    /// mesh path (`pathtrace_mesh_lit_to_rgba` + the cooked-mesh/PBR loaders)
    /// from a top-down inspection camera and verify the building is REALLY
    /// non-rectangular:
    ///   1. GROUND-CAP L PROOF — the cook seals every footprint with an
    ///      underside cap at y = 0 that follows the footprint polygon, so the
    ///      cap rasterized into an XZ grid must fill three bbox quadrants and
    ///      leave one (the L's removed corner) empty. This is measured from
    ///      the cooked payload itself, never assumed.
    ///   2. GPU SILHOUETTE GATE — pixels whose camera ray stays inside the
    ///      void quadrant for every height of the building must read as
    ///      BACKGROUND in the path-traced image (`void_cov < 0.02`) while the
    ///      building still fills the frame (`body_cov > 0.25`). Prints
    ///      `void_corner_empty: <bool>` and writes
    ///      `lut_lshape_nonrectangular.png` for human inspection.
    /// Background pixels are classified against the mean of the four frame
    /// corners (always sky-dome — the building never reaches them), and the
    /// classifier is cross-checked against the CPU rasterization of the same
    /// mesh with the same pinhole before any gate uses it.
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --release --lib forge_facade_lut_lshape -- --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    fn forge_facade_lut_lshape() {
        use super::{pathtrace_mesh_lit_to_rgba, LightRig};

        let atoms = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            "Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms/city.res_med.l3.3x4.lshape_walkup.atoms.json",
        );
        let mesh = load_craftsman_mesh(&atoms);
        let (materials, textures, _channels) = load_building_mesh_pbr_by_forge_id(&atoms);
        eprintln!(
            "[lut_lshape] cooked mesh: {} verts, {} tris, {} materials, {} textures",
            mesh.positions.len(),
            mesh.indices.len(),
            materials.len(),
            textures.len()
        );

        // ---- 1. Ground-cap L proof: rasterize the y≈0 underside cap into an
        // XZ grid and measure per-quadrant fill. The cap is the cook's seal of
        // the REAL footprint polygon, so an L cook leaves exactly one bbox
        // quadrant empty; a rectangular cook fills all four. -----------------
        let cap_tris: Vec<&[u32; 3]> = mesh
            .indices
            .iter()
            .filter(|tri| {
                tri.iter()
                    .all(|&i| mesh.positions[i as usize][1] <= 0.02)
            })
            .collect();
        assert!(
            !cap_tris.is_empty(),
            "cooked mesh has no ground cap at y<=0.02 — the seal pass drifted"
        );
        let (mut fx0, mut fx1, mut fz0, mut fz1) =
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for tri in &cap_tris {
            for &i in tri.iter() {
                let p = mesh.positions[i as usize];
                fx0 = fx0.min(p[0]);
                fx1 = fx1.max(p[0]);
                fz0 = fz0.min(p[2]);
                fz1 = fz1.max(p[2]);
            }
        }
        const GRID: usize = 128;
        let mut cap_grid = vec![false; GRID * GRID];
        for tri in &cap_tris {
            let p: Vec<[f32; 2]> = tri
                .iter()
                .map(|&i| {
                    let q = mesh.positions[i as usize];
                    [q[0], q[2]]
                })
                .collect();
            let den = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0])
                + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
            if den.abs() < 1e-9 {
                continue;
            }
            for gx in 0..GRID {
                let qx = fx0 + (gx as f32 + 0.5) / GRID as f32 * (fx1 - fx0);
                for gz in 0..GRID {
                    let qz = fz0 + (gz as f32 + 0.5) / GRID as f32 * (fz1 - fz0);
                    let l1 = ((p[1][1] - p[2][1]) * (qx - p[2][0])
                        + (p[2][0] - p[1][0]) * (qz - p[2][1]))
                        / den;
                    let l2 = ((p[2][1] - p[0][1]) * (qx - p[2][0])
                        + (p[0][0] - p[2][0]) * (qz - p[2][1]))
                        / den;
                    let l3 = 1.0 - l1 - l2;
                    if l1 >= -1e-6 && l2 >= -1e-6 && l3 >= -1e-6 {
                        cap_grid[gx * GRID + gz] = true;
                    }
                }
            }
        }
        let (cx, cz) = ((fx0 + fx1) * 0.5, (fz0 + fz1) * 0.5);
        // Quadrant fill fractions, cells within 0.4 m of the quadrant border
        // excluded so shared walls never leak across.
        let mut quad_fill = [0.0f64; 4];
        for (q, fill) in quad_fill.iter_mut().enumerate() {
            let (want_xp, want_zp) = (q & 1 == 1, q & 2 == 2);
            let (mut inside, mut filled) = (0usize, 0usize);
            for gx in 0..GRID {
                let x = fx0 + (gx as f32 + 0.5) / GRID as f32 * (fx1 - fx0);
                if (x > cx + 0.4) != want_xp || (x - cx).abs() <= 0.4 {
                    continue;
                }
                for gz in 0..GRID {
                    let z = fz0 + (gz as f32 + 0.5) / GRID as f32 * (fz1 - fz0);
                    if (z > cz + 0.4) != want_zp || (z - cz).abs() <= 0.4 {
                        continue;
                    }
                    inside += 1;
                    filled += cap_grid[gx * GRID + gz] as usize;
                }
            }
            *fill = filled as f64 / inside.max(1) as f64;
        }
        let void_q = (0..4)
            .min_by(|&a, &b| quad_fill[a].total_cmp(&quad_fill[b]))
            .unwrap();
        let filled_min = (0..4)
            .filter(|&q| q != void_q)
            .map(|q| quad_fill[q])
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "[lut_lshape] ground-cap quadrant fill (x-,z-)/(x+,z-)/(x-,z+)/(x+,z+): \
             {:.3} {:.3} {:.3} {:.3} -> void quadrant {} (fill {:.3}), other min {:.3}",
            quad_fill[0], quad_fill[1], quad_fill[2], quad_fill[3], void_q,
            quad_fill[void_q], filled_min
        );
        assert!(
            quad_fill[void_q] < 0.05 && filled_min > 0.85,
            "cooked footprint does not read as an L: quadrant fills {quad_fill:?} \
             (need one ~empty quadrant and three ~full)"
        );

        // ---- 2. Path-trace top-down and gate the GPU silhouette over the
        // void quadrant. -------------------------------------------------------
        let (w, h) = (512u32, 512u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let eye = [0.0f32, 40.0, 0.1];
        let target = [0.0f32, 0.0, 0.0];
        // Bright high dome so the below-horizon background reads clearly
        // brighter than the dark slate roof — the legacy dome's near-black
        // brown sat within the classifier's distance of the roof texture.
        let rig = LightRig {
            sun_dir: [0.4, 0.8, 0.4],
            sun_intensity: 2.6,
            sky_dome_intensity: 1.0,
            sky_dome_zenith: [0.45, 0.62, 0.95],
            sky_dome_horizon: [0.85, 0.90, 0.98],
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let rgba = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &materials,
            &textures,
            eye,
            target,
            fov_y,
            w,
            h,
            64,
            &rig,
        )
        .expect("L-shape walkup mesh render should succeed");
        let secs = t0.elapsed().as_secs_f64();
        let png = std::env::temp_dir().join("lut_lshape_nonrectangular.png");
        let mut opaque = rgba.clone();
        for px in opaque.chunks_exact_mut(4) {
            px[3] = 255;
        }
        write_png_rgba(png.to_str().unwrap(), &opaque, w, h);
        eprintln!(
            "[lut_lshape] wrote {} ({secs:.2}s, 64 spp top-down)",
            png.display()
        );

        // Diagnostic companion view (NOT gated): a 3/4 view into the removed
        // corner so a human can inspect how the notch is actually built —
        // walls, soffit, roof. Written beside the gate PNG.
        {
            let (vx, vz) = (
                if void_q & 1 == 1 { 1.0f32 } else { -1.0 },
                if void_q & 2 == 2 { 1.0f32 } else { -1.0 },
            );
            let eye34 = [vx * 16.0, 12.0, vz * 18.0];
            let rgba34 = pathtrace_mesh_lit_to_rgba(
                &mesh.positions,
                &mesh.normals,
                &mesh.uvs,
                &mesh.indices,
                &mesh.material_ids,
                &materials,
                &textures,
                eye34,
                [0.0, 4.0, 0.0],
                fov_y,
                w,
                h,
                64,
                &rig,
            )
            .expect("3/4 diagnostic render should succeed");
            let mut o = rgba34;
            for px in o.chunks_exact_mut(4) {
                px[3] = 255;
            }
            let png34 = std::env::temp_dir().join("lut_lshape_three_quarter.png");
            write_png_rgba(png34.to_str().unwrap(), &o, w, h);
            eprintln!("[lut_lshape] wrote diagnostic 3/4 view {}", png34.display());
        }

        // GPU silhouette via a MATTE pass: the same mesh/camera rendered with
        // every material flat WHITE + emissive, no textures — geometry then
        // reads near-white regardless of sun/shadow while the background
        // stays the dome, so a luma threshold is an exact GPU silhouette.
        // (The below-horizon dome cannot be re-coloured through the rig, and
        // the slate roof texture overlaps it in both luma and hue, so
        // colour-matching the beauty frame against the frame corners is not
        // a reliable classifier — measured, not assumed.)
        let matte_materials: Vec<super::PbrMaterial> = materials
            .iter()
            .map(|_| super::PbrMaterial {
                base_color: [1.0, 1.0, 1.0],
                roughness: 1.0,
                metallic: 0.0,
                emission_strength: 4.0,
                albedo_tex: -1,
                roughness_tex: -1,
                normal_tex: -1,
                uv_scale: [1.0, 1.0],
                ..Default::default()
            })
            .collect();
        let rgba_matte = pathtrace_mesh_lit_to_rgba(
            &mesh.positions,
            &mesh.normals,
            &mesh.uvs,
            &mesh.indices,
            &mesh.material_ids,
            &matte_materials,
            &[],
            eye,
            target,
            fov_y,
            w,
            h,
            64,
            &rig,
        )
        .expect("matte silhouette render should succeed");
        let matte_luma = |p: usize| -> f64 {
            0.2126 * rgba_matte[p * 4] as f64
                + 0.7152 * rgba_matte[p * 4 + 1] as f64
                + 0.0722 * rgba_matte[p * 4 + 2] as f64
        };
        let corners = [
            0usize,
            (w - 1) as usize,
            ((h - 1) * w) as usize,
            ((h - 1) * w + (w - 1)) as usize,
        ];
        let bg_luma = corners.iter().map(|&p| matte_luma(p)).sum::<f64>() / 4.0;
        assert!(
            bg_luma < 140.0,
            "matte background luma {bg_luma:.0} too bright for a clean threshold"
        );
        let thr = bg_luma + 60.0;
        let is_background = |p: usize| -> bool { matte_luma(p) < thr };

        // Cross-check: CPU raster of the same mesh with the same pinhole must
        // agree with the GPU non-background silhouette.
        let (mats_px, _tris_px) = rasterize_material_masks(&mesh, eye, target, fov_y, w, h);
        let (mut raster_cov, mut lit_px, mut overlap) = (0usize, 0usize, 0usize);
        for p in 0..(w * h) as usize {
            let on_raster = mats_px[p] >= 0;
            let on_render = !is_background(p);
            raster_cov += on_raster as usize;
            lit_px += on_render as usize;
            overlap += (on_raster && on_render) as usize;
        }
        let agree_raster = overlap as f64 / raster_cov.max(1) as f64;
        let agree_render = overlap as f64 / lit_px.max(1) as f64;
        eprintln!(
            "[lut_lshape] silhouette cross-check: raster {raster_cov} px, render {lit_px} px, \
             overlap/raster {agree_raster:.3}, overlap/render {agree_render:.3}"
        );
        assert!(
            agree_raster > 0.9 && agree_render > 0.9,
            "GPU silhouette and CPU raster disagree (agree_raster {agree_raster:.3}, \
             agree_render {agree_render:.3}) — the background classifier or camera drifted"
        );

        // The void screen block: pixels whose ray stays inside the void
        // quadrant (inset 0.45 m, clear of walls/overhang) from the ground to
        // above the roof — geometry can cover them ONLY by occupying the
        // removed corner. The ray's XZ track is linear in height, so inside
        // at y=0 and y=ymax+0.6 means inside throughout.
        let ymax = mesh
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let (qx0, qx1) = if void_q & 1 == 1 { (cx, fx1) } else { (fx0, cx) };
        let (qz0, qz1) = if void_q & 2 == 2 { (cz, fz1) } else { (fz0, cz) };
        let inset = 0.45f32;
        let in_void = |x: f32, z: f32| -> bool {
            x > qx0 + inset && x < qx1 - inset && z > qz0 + inset && z < qz1 - inset
        };
        // Same pinhole as rasterize_material_masks / the camera kernel.
        let eye_v = glam::Vec3::from(eye);
        let fwd = (glam::Vec3::from(target) - eye_v).normalize();
        let right = fwd.cross(glam::Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = (fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        let mut void_px = 0usize;
        let mut void_lit = 0usize;
        for y in 0..h {
            for x in 0..w {
                let ndc_x = (2.0 * (x as f32 + 0.5) / w as f32 - 1.0) * aspect;
                let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / h as f32;
                let dir = (fwd + (right * ndc_x + up * ndc_y) * tan_half).normalize();
                if dir.y.abs() < 1e-4 {
                    continue;
                }
                let at = |wy: f32| -> (f32, f32) {
                    let t = (wy - eye_v.y) / dir.y;
                    (eye_v.x + dir.x * t, eye_v.z + dir.z * t)
                };
                let (x0, z0) = at(0.0);
                let (x1, z1) = at(ymax + 0.6);
                if in_void(x0, z0) && in_void(x1, z1) {
                    void_px += 1;
                    void_lit += !is_background((y * w + x) as usize) as usize;
                }
            }
        }
        assert!(
            void_px >= 500,
            "void screen block too small ({void_px} px) — camera framing drifted"
        );
        let void_cov = void_lit as f64 / void_px as f64;
        let body_cov = lit_px as f64 / (w * h) as f64;
        let void_corner_empty = void_cov < 0.02 && body_cov > 0.25;
        eprintln!(
            "void_corner_empty: {void_corner_empty}  (void_cov={void_cov:.3} body_cov={body_cov:.3} \
             void_px={void_px})"
        );
        assert!(
            void_corner_empty,
            "L void corner not empty in the GPU render: void_cov={void_cov:.3} (need < 0.02), \
             body_cov={body_cov:.3} (need > 0.25) — see {}",
            png.display()
        );
    }

    /// Exact mesh equality (f32 bit patterns) — the box byte-identity oracle.
    #[cfg(feature = "spectra-native")]
    fn pt_eq_mesh(a: &CookedMesh, b: &CookedMesh) -> bool {
        a.positions.len() == b.positions.len()
            && a.indices.len() == b.indices.len()
            && a.material_ids == b.material_ids
            && a.indices == b.indices
            && a.positions
                .iter()
                .zip(&b.positions)
                .all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
            && a.normals
                .iter()
                .zip(&b.normals)
                .all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
            && a.uvs
                .iter()
                .zip(&b.uvs)
                .all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
    }

    /// RGB → hue in degrees [0,360). Used to measure per-surface colour variety.
    #[cfg(feature = "spectra-native")]
    fn rgb_hue(r: f32, g: f32, b: f32) -> f32 {
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let d = mx - mn;
        if d <= 0.0 {
            return 0.0;
        }
        let h = if mx == r {
            60.0 * (((g - b) / d) % 6.0)
        } else if mx == g {
            60.0 * ((b - r) / d + 2.0)
        } else {
            60.0 * ((r - g) / d + 4.0)
        };
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }

    /// M1 ACCEPTANCE: the engine-owned Spectra path tracer natively renders a
    /// CLUSTER of cooked civitas buildings as a small city block — MANY SDF
    /// instances referencing a MULTI-VOLUME atlas, sphere-traced as solid
    /// surfaces in megakernel.slang, occluding each other correctly, on flat
    /// ground. Per-instance flat albedo so buildings are visually distinct.
    /// Writes a PNG, prints instance count + coverage + seconds, and asserts
    /// >= 8 instances and that each building's projected footprint is solidly
    /// filled (high coverage over the footprints — not confetti). All native
    /// SDF: no quad/triangle fallback for surfaces (only the off-frame sentinel
    /// triangle for BVH validity).
    ///
    /// Run alone (GPU, seconds/frame is fine):
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///   SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SLANG_DIR=$HOME/slang-sdk \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
    ///     --lib sdf_city_block_renders_multiple_solid_buildings -- --nocapture --test-threads=1
    /// The shared M1 city-block scene: 12 instances over 6 cooked volumes on a
    /// 3x4 grid, the tight city-builder oblique camera, dark-sky sun rig and
    /// 320x240 framing. Single source of truth for the M1 milestone test AND
    /// the perf-breakdown harness so both always measure the same workload.
    #[cfg(feature = "spectra-native")]
    #[allow(clippy::type_complexity)]
    fn city_block_scene() -> (
        Vec<super::SdfVolumeInput>,
        Vec<super::SdfSceneInstance>,
        [f32; 3],      // eye
        [f32; 3],      // center / look target
        f32,           // fov_y
        (u32, u32),    // (w, h)
        super::LightRig,
    ) {
        use super::{LightRig, SdfSceneInstance};

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");

        // --- Multi-volume atlas: distinct cooked civitas fields. ------------
        let asset_names = [
            "forge.house.craftsman",
            "forge.house.victorian",
            "city.res_med.l3.3x4.rowhouse_02",
            "civic.school",
            "civic.police",
            "city.res_low.l1.2x2.cottage_01",
        ];
        let volumes: Vec<super::SdfVolumeInput> = asset_names
            .iter()
            .map(|n| load_atoms_sdf(&atoms_dir.join(format!("{n}.atoms.json"))))
            .collect();
        assert!(
            volumes.iter().all(|v| v.distances.iter().any(|d| *d < 0.0)),
            "every cooked field must contain interior (negative) samples"
        );
        for (n, v) in asset_names.iter().zip(&volumes) {
            eprintln!(
                "[sdf_block] volume {n}: res={:?} voxel={:.4} band={:.4}",
                v.resolution, v.voxel_size, v.narrow_band
            );
        }

        // Per-volume horizontal footprint (metres) so we can lay instances out
        // on a non-overlapping grid (the X/Z extent of the cooked grid).
        let footprint = |v: &super::SdfVolumeInput| -> (f32, f32) {
            (
                (v.resolution[0] - 1) as f32 * v.voxel_size,
                (v.resolution[2] - 1) as f32 * v.voxel_size,
            )
        };
        // Ground placement: each cooked field's local Y origin is at v.origin[1];
        // translate so the field's base sits on y=0 (flat ground).
        let ground_y = |v: &super::SdfVolumeInput| -> f32 { -v.origin[1] };

        // Distinct per-instance albedos (warm/cool variety so buildings read as
        // separate masses in the PNG).
        let palette = [
            [0.78, 0.40, 0.34], // brick red
            [0.46, 0.58, 0.74], // slate blue
            [0.80, 0.74, 0.52], // sand
            [0.55, 0.70, 0.52], // sage
            [0.74, 0.62, 0.46], // tan
            [0.62, 0.52, 0.66], // muted violet
            [0.84, 0.78, 0.66], // cream
            [0.50, 0.64, 0.68], // teal grey
        ];

        // --- Block layout: 3 rows x 4 columns = 12 instances on a grid. ------
        let cols = 4usize;
        let rows = 3usize;
        let gap = 6.0f32; // metres between footprints
        // Column/row pitch from the widest footprint so nothing overlaps.
        let max_w = volumes
            .iter()
            .map(|v| footprint(v).0)
            .fold(0.0f32, f32::max);
        let max_d = volumes
            .iter()
            .map(|v| footprint(v).1)
            .fold(0.0f32, f32::max);
        let pitch_x = max_w + gap;
        let pitch_z = max_d + gap;

        let mut instances: Vec<SdfSceneInstance> = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let vol = idx % volumes.len();
                let v = &volumes[vol];
                let (fw, fd) = footprint(v);
                // Cell centre on the grid, then offset so the field's own grid
                // origin lands such that the building is centred in its cell.
                let cell_x = (c as f32 - (cols as f32 - 1.0) * 0.5) * pitch_x;
                let cell_z = (r as f32 - (rows as f32 - 1.0) * 0.5) * pitch_z;
                let position = [
                    cell_x - (v.origin[0] + fw * 0.5),
                    ground_y(v),
                    cell_z - (v.origin[2] + fd * 0.5),
                ];
                instances.push(SdfSceneInstance {
                    volume_index: vol as u32,
                    position,
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    uniform_scale: 1.0,
                    albedo: palette[idx % palette.len()],
                });
            }
        }
        let n_instances = instances.len();
        assert!(
            n_instances >= 8,
            "need >= 8 instances for the cluster proof, have {n_instances}"
        );

        // --- Camera: city-builder oblique looking down at the block. ---------
        // Scene horizontal centre is the origin (grid is symmetric about 0). The
        // block spans ~(cols-1)*pitch in X and ~(rows-1)*pitch in Z plus one
        // footprint; frame it tightly so the buildings fill the view.
        let block_w = (cols as f32 - 1.0) * pitch_x + max_w;
        let block_d = (rows as f32 - 1.0) * pitch_z + max_d;
        let center = [0.0f32, 5.0, 0.0];
        let span = block_w.max(block_d);
        // Pull in close for a filled frame; a flatter oblique (eye height ~0.5
        // span) reads as the classic city-builder 3/4 view.
        let dist = span * 0.62;
        let eye = [
            center[0] + dist * 0.55,
            center[1] + dist * 0.55,
            center[2] + dist * 0.95,
        ];
        let (w, h) = (320u32, 240u32);
        let fov_y = std::f32::consts::FRAC_PI_4;

        // Dark sky / dark fills so only the lit SDF surfaces register; the SDF's
        // view-facing fill (megakernel) lights every hit pixel above the sky.
        let rig = LightRig {
            sun_dir: [0.4, 0.75, 0.5],
            sun_intensity: 2.4,
            sky_intensity: 0.0,
            camera_fill: 0.0,
            rim_fill: 0.0,
            sky_dome_intensity: 0.0,
            sky_dome_zenith: [0.0, 0.0, 0.0],
            sky_dome_horizon: [0.0, 0.0, 0.0],
            ..Default::default()
        };

        (volumes, instances, eye, center, fov_y, (w, h), rig)
    }

    /// M1: the multi-instance proof — a CITY BLOCK of buildings, every surface a
    /// native sphere-traced SDF instance, each building SOLID and per-instance
    /// coloured, buildings occluding each other correctly.
    #[cfg(feature = "spectra-native")]
    #[test]
    fn sdf_city_block_renders_multiple_solid_buildings() {
        use super::pathtrace_sdf_scene_to_rgba;

        let (volumes, instances, eye, center, fov_y, (w, h), rig) = city_block_scene();
        let n_instances = instances.len();

        let t0 = std::time::Instant::now();
        let rgba = pathtrace_sdf_scene_to_rgba(
            &volumes, &instances, eye, center, fov_y, w, h, 4, &rig,
        )
        .expect("SDF cluster render should succeed");
        let secs = t0.elapsed().as_secs_f64();

        let out_dir = std::env::temp_dir();
        let png_path = out_dir.join("ochroma_sdf_city_block.png");
        write_png_rgba(png_path.to_str().unwrap(), &rgba, w, h);
        eprintln!("[sdf_block] wrote {}", png_path.display());

        // --- Total lit coverage (fraction of frame that is solid building). --
        let thr = 30.0f32;
        let lit = (0..(w * h))
            .filter(|&p| luma(&rgba[(p * 4) as usize..(p * 4 + 4) as usize]) > thr)
            .count();
        let total_coverage = lit as f64 / (w * h) as f64;

        // --- Per-instance solidity: project each instance's world AABB to a
        // screen box, then measure SPAN-FILL within that box (the M0 confetti-vs-
        // solid metric, localized): for each row in the box, the lit pixels must
        // fill their span (first→last lit pixel). A SOLID building fills each
        // scanline span ≈ 1.0; confetti leaves gaps; an absent instance has no
        // lit pixels at all. This is robust to the oblique view making the
        // projected 3D-AABB box much larger than the building's silhouette (the
        // box is mostly corner air, which span-fill correctly ignores).
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::from(eye),
            glam::Vec3::from(center),
            glam::Vec3::Y,
        );
        let aspect = w as f32 / h as f32;
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, 0.05, 10_000.0);
        let view_proj = proj * view;
        let project = |p: glam::Vec3| -> Option<(f32, f32)> {
            let clip = view_proj * p.extend(1.0);
            if clip.w <= 1e-4 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            // NDC x in [-1,1] → pixel; y flipped (NDC +y up, pixel +y down).
            let px = (ndc.x * 0.5 + 0.5) * w as f32;
            let py = (1.0 - (ndc.y * 0.5 + 0.5)) * h as f32;
            Some((px, py))
        };

        let mut solid_instances = 0usize;
        let mut per_inst_fill: Vec<f64> = Vec::new();
        for inst in &instances {
            let v = &volumes[inst.volume_index as usize];
            let lmin = v.origin;
            let lmax = [
                v.origin[0] + (v.resolution[0] - 1) as f32 * v.voxel_size,
                v.origin[1] + (v.resolution[1] - 1) as f32 * v.voxel_size,
                v.origin[2] + (v.resolution[2] - 1) as f32 * v.voxel_size,
            ];
            let q = glam::Quat::from_array(inst.rotation_xyzw).normalize();
            let pos = glam::Vec3::from(inst.position);
            let (mut bx0, mut by0, mut bx1, mut by1) =
                (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            let mut any = false;
            for cx in [lmin[0], lmax[0]] {
                for cy in [lmin[1], lmax[1]] {
                    for cz in [lmin[2], lmax[2]] {
                        let world =
                            pos + q * (glam::Vec3::new(cx, cy, cz) * inst.uniform_scale);
                        if let Some((px, py)) = project(world) {
                            bx0 = bx0.min(px);
                            by0 = by0.min(py);
                            bx1 = bx1.max(px);
                            by1 = by1.max(py);
                            any = true;
                        }
                    }
                }
            }
            if !any {
                per_inst_fill.push(0.0);
                continue;
            }
            // Clamp the screen box to the frame; span-fill within it per row.
            let x0 = bx0.floor().clamp(0.0, (w - 1) as f32) as u32;
            let y0 = by0.floor().clamp(0.0, (h - 1) as f32) as u32;
            let x1 = bx1.ceil().clamp(0.0, (w - 1) as f32) as u32;
            let y1 = by1.ceil().clamp(0.0, (h - 1) as f32) as u32;
            let (mut sum_fill, mut span_rows, mut box_lit) = (0.0f64, 0u64, 0u64);
            for y in y0..=y1 {
                let (mut first, mut last, mut count) = (None, 0u32, 0u32);
                for x in x0..=x1 {
                    let i = ((y * w + x) * 4) as usize;
                    if luma(&rgba[i..i + 4]) > thr {
                        first.get_or_insert(x);
                        last = x;
                        count += 1;
                    }
                }
                box_lit += count as u64;
                if let Some(f) = first {
                    let span = last - f + 1;
                    if span >= 3 {
                        sum_fill += count as f64 / span as f64;
                        span_rows += 1;
                    }
                }
            }
            // A building absent from the frame has no lit pixels → fill 0.
            let fill = if span_rows > 0 && box_lit > 0 {
                sum_fill / span_rows as f64
            } else {
                0.0
            };
            per_inst_fill.push(fill);
            // Solid: the building's scanline spans are well-filled (continuous
            // walls/roof) AND it actually has surface pixels in the frame.
            if fill >= 0.80 && box_lit > 20 {
                solid_instances += 1;
            }
        }

        let mean_fill = per_inst_fill.iter().sum::<f64>() / per_inst_fill.len() as f64;
        eprintln!(
            "[sdf_block] MULTI-INSTANCE SDF CLUSTER (all native sphere-traced SDF):\n  \
             instances placed   : {n_instances}\n  \
             instances solid    : {solid_instances} (per-instance span-fill >= 0.80)\n  \
             total lit coverage : {:.4} of frame\n  \
             mean span-fill     : {:.4}\n  \
             seconds/frame      : {:.2}s\n  \
             per-instance fill  : {:?}",
            total_coverage,
            mean_fill,
            secs,
            per_inst_fill
                .iter()
                .map(|f| (f * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
        );

        // --- Acceptance gates. ----------------------------------------------
        assert!(
            n_instances >= 8,
            "must render >= 8 SDF instances, placed {n_instances}"
        );
        assert!(
            lit > 0,
            "frame is entirely background — no SDF surface was hit"
        );
        // The cluster must be SOLID: the large majority of placed buildings must
        // each fill their projected footprint (multiple distinct solid masses,
        // not one building and not confetti). Allow a couple of edge instances
        // to be partly clipped by the frame.
        assert!(
            solid_instances >= 8,
            "expected >= 8 buildings to render as SOLID surfaces (per-instance \
             span-fill >= 0.80), got {solid_instances} (mean span-fill {:.3}). A \
             solid SDF cluster fills each building's scanline spans; confetti / \
             missing instances do not.",
            mean_fill
        );
        // And the block must occupy a real chunk of the frame (not a single
        // building lost in a sea of sky).
        assert!(
            total_coverage >= 0.15,
            "building cluster should cover a substantial part of the frame, got \
             {total_coverage:.4}"
        );
    }

    /// Aggregate raw per-dispatch timings into (label, count, total_ms) rows,
    /// sorted by total descending.
    #[cfg(feature = "spectra-native")]
    fn aggregate_kernel_ms(raw: &[(String, f32)]) -> Vec<(String, u32, f32)> {
        let mut map: std::collections::HashMap<String, (u32, f32)> = Default::default();
        for (label, ms) in raw {
            let e = map.entry(label.clone()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += *ms;
        }
        let mut rows: Vec<(String, u32, f32)> = map
            .into_iter()
            .map(|(label, (n, total))| (label, n, total))
            .collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        rows
    }

    /// PERF BREAKDOWN (run explicitly with --ignored): per-kernel + per-cost
    /// breakdown of the M1 city-block frame on this box. Prints the hard table
    /// that proves where the seconds go — march vs normal+shade vs everything
    /// else — plus per-ray march counters (steps, empty-space steps, instance
    /// evaluations) harvested from the kernel via u_sdf_debug_mode=1.
    ///
    /// Run:
    ///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    ///   SPECTRA_SLANG_DIR=$HOME/src/spectra-perf/slang \
    ///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release \
    ///     --lib sdf_city_block_perf_breakdown -- --ignored --nocapture --test-threads=1
    #[cfg(feature = "spectra-native")]
    #[test]
    #[ignore = "perf harness — run explicitly; seconds-per-frame on the iGPU"]
    fn sdf_city_block_perf_breakdown() {
        use super::{SdfScenePerfKnobs, pathtrace_sdf_scene_perf};

        let (volumes, instances, eye, center, fov_y, (w, h), rig) = city_block_scene();
        let _n_inst = instances.len() as f64;
        let n_px = (w * h) as usize;

        #[derive(Clone, Copy)]
        struct V {
            spp: u32,
            debug: i32,
            legacy: bool,
            lean: bool,
        }
        let run = |v: V, label: &str| {
            let knobs = SdfScenePerfKnobs {
                debug_mode: v.debug,
                disable_denoiser: true,
                max_bounces: None,
                collect_kernel_timing: true,
                frames: 3,
                legacy_march: v.legacy,
                lean_pipeline: v.lean,
            };
            let t0 = std::time::Instant::now();
            let (rgba, report) = pathtrace_sdf_scene_perf(
                &volumes, &instances, eye, center, fov_y, w, h, v.spp, &rig, &knobs,
            )
            .expect("perf render should succeed");
            let wall_s = t0.elapsed().as_secs_f64();
            eprintln!(
                "\n[perf] ===== {label} (spp={} debug={} legacy_march={} lean={}) =====\n\
                 [perf] frame times ms (f0=cold compile+alloc): {:?}\n\
                 [perf] steady-state frame = {:.1} ms (wall incl. setup {:.1}s) samples_done={}",
                v.spp, v.debug, v.legacy, v.lean,
                report
                    .frame_times_ms
                    .iter()
                    .map(|t| (t * 10.0).round() / 10.0)
                    .collect::<Vec<_>>(),
                report.render_time_ms, wall_s, report.samples_done
            );
            let rows = aggregate_kernel_ms(&report.per_kernel_ms);
            let total_kernel_ms: f32 = rows.iter().map(|r| r.2).sum();
            eprintln!(
                "[perf] {:<28} {:>5} {:>12} {:>8}",
                "kernel", "n", "total ms", "share"
            );
            for (klabel, n, total) in rows.iter().take(8) {
                eprintln!(
                    "[perf] {:<28} {:>5} {:>12.1} {:>7.1}%",
                    klabel,
                    n,
                    total,
                    100.0 * total / total_kernel_ms
                );
            }
            eprintln!(
                "[perf] {:<28} {:>5} {:>12.1} {:>7.1}%",
                "ALL KERNELS",
                report.per_kernel_ms.len(),
                total_kernel_ms,
                100.0
            );
            (rgba, report)
        };

        // March counter harvest from a debug_mode=1 run's raw film:
        // (marching_rays, steps mean/p99/max, empty mean, evals mean, totals)
        let harvest = |r: &super::SdfScenePerfReport| {
            let mut steps: Vec<f32> = Vec::with_capacity(n_px);
            let mut empty: Vec<f32> = Vec::with_capacity(n_px);
            let mut evals: Vec<f32> = Vec::with_capacity(n_px);
            for i in 0..n_px {
                steps.push(r.raw_film[i * 3]);
                empty.push(r.raw_film[i * 3 + 1]);
                evals.push(r.raw_film[i * 3 + 2]);
            }
            let sum = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
            let marching = steps.iter().filter(|s| **s > 0.0).count();
            let mut sorted = steps.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p99 = sorted[((sorted.len() - 1) as f64 * 0.99) as usize];
            let max = *sorted.last().unwrap();
            (
                marching,
                sum(&steps) / n_px as f64,         // steps mean (all px)
                sum(&empty) / n_px as f64,         // empty mean
                sum(&evals) / n_px as f64,         // evals mean
                p99 as f64,
                max as f64,
                sum(&steps),
                sum(&evals),
            )
        };
        let shade_ms = |r: &super::SdfScenePerfReport| -> f32 {
            r.per_kernel_ms
                .iter()
                .filter(|(l, _)| l == "shade_megakernel")
                .map(|(_, ms)| *ms)
                .sum()
        };
        let kernels_total = |r: &super::SdfScenePerfReport| -> f32 {
            r.per_kernel_ms.iter().map(|(_, ms)| *ms).sum()
        };

        // ---- The variant matrix (all steady-state, frames=3, denoiser off) --
        let (_, w2) = run(V { spp: 4, debug: 2, legacy: true,  lean: false }, "W2 no-SDF floor");
        let (_, w1) = run(V { spp: 4, debug: 1, legacy: true,  lean: false }, "W1 LEGACY march-only");
        let (w0_rgba, w0) = run(V { spp: 4, debug: 0, legacy: true,  lean: false }, "W0 LEGACY full (M1 baseline)");
        let (_, a1) = run(V { spp: 4, debug: 1, legacy: false, lean: false }, "A1 FIXA march-only");
        let (a0_rgba, a0) = run(V { spp: 4, debug: 0, legacy: false, lean: false }, "A0 FIXA full");
        let (l0_rgba, l0) = run(V { spp: 1, debug: 0, legacy: false, lean: true }, "L0 FIXA+LEAN 1spp (realtime-shaped)");
        let (_, l1) = run(V { spp: 1, debug: 0, legacy: true,  lean: true }, "L1 LEGACY+LEAN 1spp");
        // Lean march isolation (single shade dispatch, no ReSTIR noise):
        let (_, l2) = run(V { spp: 1, debug: 2, legacy: false, lean: true }, "L2 LEAN no-SDF floor");
        let (_, l1d) = run(V { spp: 1, debug: 1, legacy: true,  lean: true }, "L1d LEGACY+LEAN march-only");
        let (_, l0d) = run(V { spp: 1, debug: 1, legacy: false, lean: true }, "L0d FIXA+LEAN march-only");

        // ---- Correctness proof: Fix A must render the same city block. ------
        let out_dir = std::env::temp_dir();
        write_png_rgba(out_dir.join("ochroma_sdf_perf_legacy.png").to_str().unwrap(), &w0_rgba, w, h);
        write_png_rgba(out_dir.join("ochroma_sdf_perf_fixa.png").to_str().unwrap(), &a0_rgba, w, h);
        write_png_rgba(out_dir.join("ochroma_sdf_perf_lean.png").to_str().unwrap(), &l0_rgba, w, h);
        let mut diff_sum = 0u64;
        let mut diff_max = 0u8;
        let mut diff_cnt = 0usize;
        for i in 0..(n_px * 4) {
            let d = w0_rgba[i].abs_diff(a0_rgba[i]);
            diff_sum += d as u64;
            diff_max = diff_max.max(d);
            if d > 8 {
                diff_cnt += 1;
            }
        }
        let thr = 30.0f32;
        let lit_of = |rgba: &[u8]| {
            (0..n_px)
                .filter(|&p| luma(&rgba[p * 4..p * 4 + 4]) > thr)
                .count()
        };
        let (lit_w0, lit_a0, lit_l0) = (lit_of(&w0_rgba), lit_of(&a0_rgba), lit_of(&l0_rgba));
        eprintln!(
            "\n[perf] image diff legacy vs fixA: mean={:.3}/255 max={} px>8={}/{} | lit px: legacy={} fixA={} lean1spp={}",
            diff_sum as f64 / (n_px * 4) as f64, diff_max, diff_cnt, n_px * 4, lit_w0, lit_a0, lit_l0
        );

        // ---- Counters before/after. -----------------------------------------
        let (mar_b, st_b, em_b, ev_b, p99_b, max_b, tot_st_b, tot_ev_b) = harvest(&w1);
        let (mar_a, st_a, em_a, ev_a, p99_a, max_a, tot_st_a, tot_ev_a) = harvest(&a1);

        // March-only isolation in the LEAN pipeline (1 shade dispatch, no
        // ReSTIR/NRC dispatch noise): debug1 (march, no normal/shade) minus
        // debug2 (no SDF work at all).
        let march_b = shade_ms(&l1d) - shade_ms(&l2);
        let march_a = shade_ms(&l0d) - shade_ms(&l2);
        let _ = (&w1, &a1); // full-pipeline debug variants (counters only)

        eprintln!("\n[perf] ========== HARD TABLE (320x240, 12 instances; steady-state frame) ==========");
        eprintln!("[perf] cold frame0 (compile+alloc) ms  : {:.0} (runtime slangc — every fresh Renderer)", w0.frame_times_ms[0]);
        eprintln!("[perf] steady frame ms                 : LEGACY={:.1}  FIXA={:.1}  FIXA+LEAN(1spp)={:.1}  LEGACY+LEAN={:.1}", w0.render_time_ms, a0.render_time_ms, l0.render_time_ms, l1.render_time_ms);
        eprintln!("[perf] GPU kernels total ms            : LEGACY={:.1}  FIXA={:.1}  FIXA+LEAN={:.1}  no-SDF floor={:.1}", kernels_total(&w0), kernels_total(&a0), kernels_total(&l0), kernels_total(&w2));
        eprintln!("[perf] shade_megakernel ms             : LEGACY={:.1}  FIXA={:.1}  LEAN={:.1}  no-SDF={:.1}", shade_ms(&w0), shade_ms(&a0), shade_ms(&l0), shade_ms(&w2));
        eprintln!("[perf] march-only ms (lean, 1 dispatch) : LEGACY={march_b:.1}  FIXA={march_a:.1}  speedup x{:.1}", march_b / march_a.max(1e-3));
        eprintln!("[perf] CPU+readback overhead ms        : LEGACY={:.1}  FIXA+LEAN={:.1}", w0.render_time_ms - kernels_total(&w0) as f64, l0.render_time_ms - kernels_total(&l0) as f64);
        eprintln!("[perf] --- march counters (mean over all px; 55.9% of rays enter the AABB) ---");
        eprintln!("[perf] steps/ray mean|p99|max          : LEGACY {st_b:.1}|{p99_b:.0}|{max_b:.0}   FIXA {st_a:.1}|{p99_a:.0}|{max_a:.0}");
        eprintln!("[perf] empty steps/ray mean            : LEGACY {em_b:.1} ({:.1}%)   FIXA {em_a:.1} ({:.1}%)", 100.0 * em_b / st_b.max(1e-9), 100.0 * em_a / st_a.max(1e-9));
        eprintln!("[perf] instance evals/ray mean         : LEGACY {ev_b:.1}   FIXA {ev_a:.1}   reduction x{:.1}", ev_b / ev_a.max(1e-9));
        eprintln!("[perf] totals/frame@1spp  steps|evals  : LEGACY {tot_st_b:.2e}|{tot_ev_b:.2e}   FIXA {tot_st_a:.2e}|{tot_ev_a:.2e}");
        eprintln!("[perf] marching rays                   : LEGACY {mar_b} (union-AABB entrants)  FIXA {mar_a} (instance-AABB entrants)");
        eprintln!("[perf] ==============================================================================\n");

        // ---- Real-outcome gates. --------------------------------------------
        assert!(w0.render_time_ms > 0.0 && shade_ms(&w0) > 0.0, "timing sink empty");
        assert!(mar_b > 0 && tot_st_b > 0.0, "march counters empty — debug mode not wired");
        assert!(max_b > 16.0, "legacy max steps implausibly low — counters clamped?");
        // Fix A must visit the same surfaces — the IMAGE is the invariant.
        // (The marching-ray count legitimately drops: legacy marches every
        // union-AABB entrant; interval culling only marches rays that enter
        // at least one instance AABB.)
        assert!(
            mar_a > 0 && mar_a <= mar_b,
            "interval culling should march a subset of legacy rays: {mar_b} -> {mar_a}"
        );
        assert!(
            lit_a0 as f64 >= lit_w0 as f64 * 0.98 && lit_a0 as f64 <= lit_w0 as f64 * 1.02,
            "Fix A changed lit coverage: {lit_w0} -> {lit_a0}"
        );
        assert!(
            diff_sum as f64 / ((n_px * 4) as f64) < 1.0,
            "Fix A image diverged from legacy: mean diff {:.3}/255",
            diff_sum as f64 / (n_px * 4) as f64
        );
        // Fix A must actually reduce the work, not just match the image.
        assert!(
            ev_a < ev_b * 0.5,
            "Fix A failed to cut instance evals: {ev_b:.1} -> {ev_a:.1}"
        );
    }

    /// 720p PROBE (run explicitly with --ignored): the realtime-shaped frame
    /// (interval-culled march + lean pipeline, 1 spp) at the design doc's
    /// contract-point internal resolution, measured on THIS box — the number
    /// the 4070 Ti projection scales from (no assumed pixel-scaling).
    #[cfg(feature = "spectra-native")]
    #[test]
    #[ignore = "perf harness — run explicitly"]
    fn sdf_city_block_perf_720p() {
        use super::{SdfScenePerfKnobs, pathtrace_sdf_scene_perf};

        let (volumes, instances, eye, center, fov_y, _wh, rig) = city_block_scene();
        let (w, h) = (1280u32, 720u32);
        let knobs = SdfScenePerfKnobs {
            debug_mode: 0,
            disable_denoiser: true,
            max_bounces: None,
            collect_kernel_timing: true,
            frames: 4,
            legacy_march: false,
            lean_pipeline: true,
        };
        let (rgba, report) = pathtrace_sdf_scene_perf(
            &volumes, &instances, eye, center, fov_y, w, h, 1, &rig, &knobs,
        )
        .expect("720p perf render should succeed");
        let knobs_legacy = SdfScenePerfKnobs {
            legacy_march: true,
            ..knobs.clone()
        };
        let (_, report_legacy) = pathtrace_sdf_scene_perf(
            &volumes, &instances, eye, center, fov_y, w, h, 1, &rig, &knobs_legacy,
        )
        .expect("720p legacy perf render should succeed");

        let rows = aggregate_kernel_ms(&report.per_kernel_ms);
        eprintln!("\n[perf720] ===== 1280x720, 1spp, lean, interval-culled march =====");
        eprintln!("[perf720] frame times ms: {:?}", report.frame_times_ms.iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>());
        for (label, n, total) in rows.iter() {
            eprintln!("[perf720] {label:<24} n={n} {total:>8.1} ms");
        }
        let kernels: f32 = report.per_kernel_ms.iter().map(|(_, ms)| ms).sum();
        let kernels_legacy: f32 = report_legacy.per_kernel_ms.iter().map(|(_, ms)| ms).sum();
        let n_px = (w * h) as usize;
        let thr = 30.0f32;
        let lit = (0..n_px)
            .filter(|&p| luma(&rgba[p * 4..p * 4 + 4]) > thr)
            .count();
        eprintln!(
            "[perf720] steady frame: FIXA+LEAN={:.1} ms (kernels {:.1}) | LEGACY+LEAN={:.1} ms (kernels {:.1}) | lit coverage {:.3}",
            report.render_time_ms,
            kernels,
            report_legacy.render_time_ms,
            kernels_legacy,
            lit as f64 / n_px as f64
        );
        let png = std::env::temp_dir().join("ochroma_sdf_perf_720p.png");
        write_png_rgba(png.to_str().unwrap(), &rgba, w, h);
        eprintln!("[perf720] wrote {}", png.display());
        assert!(report.render_time_ms > 0.0 && kernels > 0.0);
        assert!(
            lit as f64 / n_px as f64 > 0.15,
            "720p frame lost the city block (lit {:.3})",
            lit as f64 / n_px as f64
        );
    }
}
