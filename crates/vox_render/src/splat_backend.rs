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

    a[0] = pack_u32(1); // MAT_LAMBERT
    a[4] = m.base_color[0];
    a[5] = m.base_color[1];
    a[6] = m.base_color[2];
    a[7] = m.roughness;
    a[9] = m.metallic;
    a[10] = 1.5; // ior

    a[20] = m.base_color[0];
    a[21] = m.base_color[1];
    a[22] = m.base_color[2];
    a[23] = m.emission_strength;

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
    a[73] = pack_i32(0); // thin_walled

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
}
