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

/// One-shot still: **path-trace SDF buildings with the CELL-GRID atom gather**
/// — the wave-1 textured-SDF entry point (Task 1 scaffold; later wave tasks
/// grow it with per-channel PBR materials, box-UV textures, normal maps and
/// aperture rects). A thin extension of
/// [`pathtrace_sdf_scene_with_atoms_to_rgba`]: the SAME scene build, plus a
/// per-instance uniform cell grid over the atoms' WORLD positions so the
/// megakernel's k-NN gather visits only the 3×3×3 cell neighbourhood of a hit
/// (`u_sdf_textured = 1`) instead of scanning the instance's full atom range —
/// identical K-nearest + glass logic over ~80× fewer atoms, byte-identical
/// frames (the parity gate `sdf_gather_grid_matches_linear` proves it).
/// Returns RGBA8 (`w*h*4`).
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_textured_to_rgba(
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
            LightRig, SdfSceneInstance, pathtrace_sdf_scene_textured_to_rgba,
            pathtrace_sdf_scene_with_atoms_to_rgba,
        };

        let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Ochroma/projects/civitas_care/assets/buildings/forge_starter/atoms");
        let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

        let volume = load_atoms_sdf(&asset_path);
        let (atoms, raw_glass) = load_atoms_material(&asset_path, false);
        let n_atoms = atoms.len();
        eprintln!(
            "[sdf_gather_grid] craftsman: res={:?} voxel={:.4} atoms={n_atoms} (glass={raw_glass})",
            volume.resolution, volume.voxel_size
        );

        // Same placement + camera + rig as the M2 glass test so the frame
        // exercises facade, roof, trim AND glass-detect gather paths.
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

        let (w, h) = (320u32, 320u32);
        let fov_y = std::f32::consts::FRAC_PI_4;
        let center = [0.0f32, 4.5, 0.0];
        let eye = [3.0f32, 6.0, 22.0];
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

        // Linear oracle (M2 entry point, u_sdf_textured = 0: full atom scan).
        let rgba_linear = pathtrace_sdf_scene_with_atoms_to_rgba(
            &volume_slice(&volume),
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
        // materials empty -> identical flat-blend shading, gather-only mode).
        let rgba_grid = pathtrace_sdf_scene_textured_to_rgba(
            &volume_slice(&volume),
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
        let (wmin, wmax) = super::sdf_instance_world_aabb(&volume, &instance);
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

    /// Wrap a single SdfVolumeInput in a 1-element slice for the scene API.
    #[cfg(feature = "spectra-native")]
    fn volume_slice(v: &super::SdfVolumeInput) -> Vec<super::SdfVolumeInput> {
        vec![v.clone()]
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
