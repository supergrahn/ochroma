//! Perturbation coverage gate (design `2026-06-13-kernel-target-split-config.md`
//! Phase 5 — "the dead-settings antidote").
//!
//! The codebase's dominant failure mode is built-but-unwired systems: a config
//! leaf exists, serializes, validates — but nothing downstream reads it, so the
//! UI knob is dead. This test is the antidote.
//!
//! For every leaf field of [`spectra_renderer::RenderSettings`] we either:
//!   1. PERTURB it (set it off-default), render a tiny synthetic SDF scene
//!      through the settings-first path, and assert the output **measurably
//!      changes** (a content hash differs from the default-settings baseline);
//!      OR
//!   2. classify it [`KNOWN_UNWIRED`] with a written reason (dead in the native
//!      still path today, or no deterministic visible effect on this trivial
//!      scene — needs a richer fixture).
//!
//! The coverage assertion ([`perturbation_coverage`]) then checks that
//! `WIRED ∪ KNOWN_UNWIRED` names **every** leaf in the serialized default tree.
//! Adding a new leaf to `RenderSettings` therefore FORCES a decision: wire it
//! and prove it moves pixels, or write down why it does not. A leaf that is
//! silently bound nowhere — the dead-setting bug — cannot pass this gate.
//!
//! A WIRED leaf that gets unwired (e.g. someone deletes the `apply_settings`
//! line for `look.exposure_ev`) makes its perturbation produce a hash IDENTICAL
//! to the baseline → the test FAILS, naming the dead setting. See the summary
//! for the demonstrated failure.
//!
//! GPU-gated (`spectra-native`): run alone, single-threaded.
//!   SPECTRA_BACKEND=vulkan SLANG_DIR=$HOME/slang-sdk \
//!     cargo test -p vox_render --features spectra-native \
//!     perturbation_coverage -- --nocapture --test-threads=1

#![cfg(feature = "spectra-native")]

// Real internals reached through the crate (this is a child module of the
// crate, so `super::super` is `crate::splat_backend`).
use super::super::{
    pack_vulkan_directional_light, resolve_slang_kernel_dir, RenderConfig, Renderer,
    VulkanSlangBackend, VULKAN_MATERIAL_FLOATS,
};
use crate::splat_convert::camera_layer;
use spectra_renderer::RenderSettings;

// =============================================================================
// Synthetic scene — an analytic SDF sphere. No external assets, deterministic.
// =============================================================================

/// Build a small signed-distance sphere grid (no cooked asset needed). Returns
/// (resolution, origin, voxel_size, narrow_band, distances). The sphere is
/// centred in a `n³` grid spanning `[-1, 1]³`, radius `0.7`.
fn synthetic_sphere_sdf() -> ([u32; 3], [f32; 3], f32, f32, Vec<f32>) {
    let n: u32 = 32;
    let extent = 2.0f32; // grid spans [-1, 1]
    let voxel = extent / (n - 1) as f32;
    let origin = [-1.0f32, -1.0, -1.0];
    let radius = 0.7f32;
    let narrow_band = 4.0 * voxel;
    let mut distances = Vec::with_capacity((n * n * n) as usize);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = [
                    origin[0] + i as f32 * voxel,
                    origin[1] + j as f32 * voxel,
                    origin[2] + k as f32 * voxel,
                ];
                let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - radius;
                // Clamp into the narrow band like a cooked field.
                distances.push(d.clamp(-narrow_band, narrow_band));
            }
        }
    }
    ([n, n, n], origin, voxel, narrow_band, distances)
}

/// FNV-1a hash of the quantised beauty buffer. Quantising to 8-bit before
/// hashing makes the comparison robust to sub-LSB float jitter while still
/// catching any visible change (tonemap, exposure, lighting, geometry).
fn frame_hash(rgba: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in rgba {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Mean luma (R+G+B average over all pixels), 0..255. Used only for the
/// human-readable trace line.
fn mean_luma(rgba: &[u8]) -> f64 {
    let n = rgba.len() / 4;
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = rgba
        .chunks_exact(4)
        .map(|p| (p[0] as f64 + p[1] as f64 + p[2] as f64) / 3.0)
        .sum();
    sum / n as f64
}

/// Render the synthetic SDF sphere driven ENTIRELY by `settings` — the
/// settings-first render the perturbation gate exercises.
///
/// Mirrors [`super::super::pathtrace_sdf_to_rgba`] but takes a full
/// [`RenderSettings`]: `apply_settings` carries `render`/`look`/`features`/the
/// atmosphere toggle onto the config, and the `lighting.*` values reach the GPU
/// through the `Renderer::set_sun`/`set_atmosphere`/`set_sky_gradient` setters
/// and the directional-light rig (exactly as the production SDF path wires the
/// `LightRig`). Resolution comes from `settings.render.resolution`.
fn render_with_settings(settings: &RenderSettings) -> Result<Vec<u8>, String> {
    use spectra_scene_state::{
        LightLayer, MaterialLayer, SceneState, SdfInstanceHeader, SdfLayer, SdfVolumeHeader,
    };

    let (resolution, origin, voxel_size, narrow_band, distances) = synthetic_sphere_sdf();
    let width = settings.render.resolution[0];
    let height = settings.render.resolution[1];

    let position = [0.0f32, 0.0, 0.0];
    let scale = 1.0f32;
    let local_min = origin;
    let local_max = [
        origin[0] + (resolution[0] - 1) as f32 * voxel_size,
        origin[1] + (resolution[1] - 1) as f32 * voxel_size,
        origin[2] + (resolution[2] - 1) as f32 * voxel_size,
    ];
    let pad = narrow_band.max(voxel_size) * scale;
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
            resolution,
            origin,
            voxel_size,
            narrow_band,
            distance_offset: 0,
            distance_count: distances.len() as u32,
        }],
        vec![SdfInstanceHeader {
            instance_id: 1,
            asset_id: 1,
            volume_index: 0,
            position,
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            uniform_scale: scale,
            world_aabb_min,
            world_aabb_max,
        }],
        distances,
    );

    let mut scene = SceneState::new(width, height);
    scene.sdf = sdf;
    scene.mark_sdf_changed();

    // Sentinel triangle far below the sphere so the shadow-BVH traversal has a
    // valid single-leaf tree (same guard as the production SDF path).
    {
        let far = [position[0], world_aabb_min[1] - 1000.0, position[2]];
        scene.geometry.vertex_count = 3;
        scene.geometry.triangle_count = 1;
        scene.geometry.positions = vec![
            far[0], far[1], far[2],
            far[0] + 0.001, far[1], far[2],
            far[0], far[1], far[2] + 0.001,
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

    // Camera: front-quarter view pulled back so the sphere fills the frame.
    let center = [0.0f32, 0.0, 0.0];
    let eye = [1.6f32, 1.1, 2.2];
    let fov_y = std::f32::consts::FRAC_PI_4;

    // Four-light rig driven from settings.lighting (not a LightRig). Directions
    // for the fills are derived from the camera exactly like the production path.
    let sun = glam::Vec3::from(settings.lighting.sun.dir).normalize_or_zero();
    let camera_fill = (glam::Vec3::from(eye) - glam::Vec3::from(center) + glam::Vec3::Y * 0.35)
        .normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::new();
    for (dir, color, intensity) in [
        (sun.to_array(), settings.lighting.sun.color, settings.lighting.sun.intensity),
        (
            glam::Vec3::Y.to_array(),
            [0.58, 0.62, 0.72],
            settings.lighting.sky_dome.sky_intensity,
        ),
        (
            camera_fill.to_array(),
            [0.72, 0.74, 0.78],
            settings.lighting.sky_dome.camera_fill,
        ),
        (
            rim_fill.to_array(),
            [0.45, 0.47, 0.52],
            settings.lighting.sky_dome.rim_fill,
        ),
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
        glam::Vec3::from(center),
        glam::Vec3::Y,
    )
    .to_cols_array();
    let cam = camera_layer(view, fov_y, width, height);
    scene.camera = cam.clone();

    let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
    let mut config = RenderConfig::near_realtime(width, height);
    config.slang_kernel_dir = resolve_slang_kernel_dir();
    // THE settings-first route: every render/look/feature/atmosphere leaf the
    // renderer reads from the config arrives through apply_settings.
    config.apply_settings(settings);
    let mut renderer = Renderer::new(gpu, config);
    renderer.set_sdf_albedo([0.62, 0.55, 0.48]);
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_sky_gradient(
        settings.lighting.sky_dome.zenith,
        settings.lighting.sky_dome.horizon,
        settings.lighting.sky_dome.intensity,
    );
    // Atmosphere values reach the GPU through the setters only when enabled,
    // matching the production path (otherwise u_atmosphere_enabled stays 0).
    if settings.lighting.atmosphere.enabled {
        renderer.set_sun(sun.to_array(), settings.lighting.sun.radiance);
        renderer.set_atmosphere(
            true,
            settings.lighting.atmosphere.mie,
            settings.lighting.atmosphere.turbidity,
        );
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

// =============================================================================
// Leaf catalogue: WIRED (perturb + prove) vs KNOWN_UNWIRED (documented).
// =============================================================================

/// A WIRED leaf: its dotted path + a closure that sets it off-default on a
/// fresh `RenderSettings`. The harness renders the perturbed settings and
/// asserts the frame hash differs from the default-settings baseline.
struct Wired {
    path: &'static str,
    perturb: fn(&mut RenderSettings),
}

/// The leaves that demonstrably move pixels on this synthetic sphere scene.
/// Each perturbation is chosen to be unambiguously visible (and to stay inside
/// `validate()`'s ranges so resolve/validate would accept it too).
fn wired_leaves() -> Vec<Wired> {
    vec![
        // --- look: tonemap operator + exposure are the headline look knobs ---
        Wired {
            path: "look.exposure_ev",
            perturb: |s| s.look.exposure_ev = 3.0, // +3 EV — much brighter
        },
        Wired {
            path: "look.tonemap.operator",
            // Default is Aces; Linear (no tonemap) changes the highlight rolloff.
            perturb: |s| s.look.tonemap.operator = spectra_renderer::SettingsToneMapper::Linear,
        },
        // --- render: spp + bounces change the converged/noise content -------
        Wired {
            path: "render.spp",
            // 1 spp vs the near_realtime default — noise pattern differs.
            perturb: |s| s.render.spp = 1,
        },
        // --- lighting.sun: the dominant key light ---------------------------
        Wired {
            path: "lighting.sun.intensity",
            perturb: |s| s.lighting.sun.intensity = 12.0,
        },
        Wired {
            path: "lighting.sun.color",
            perturb: |s| s.lighting.sun.color = [1.0, 0.1, 0.1], // red sun
        },
        Wired {
            path: "lighting.sun.dir",
            perturb: |s| s.lighting.sun.dir = [-0.6, 0.3, -0.7], // back-light
        },
        // --- lighting.sky_dome: the ambient gradient + fills ----------------
        Wired {
            path: "lighting.sky_dome.intensity",
            perturb: |s| s.lighting.sky_dome.intensity = 3.0,
        },
        Wired {
            path: "lighting.sky_dome.zenith",
            perturb: |s| s.lighting.sky_dome.zenith = [1.0, 0.0, 0.0],
        },
        Wired {
            path: "lighting.sky_dome.horizon",
            perturb: |s| s.lighting.sky_dome.horizon = [0.0, 1.0, 0.0],
        },
        Wired {
            path: "lighting.sky_dome.sky_intensity",
            perturb: |s| s.lighting.sky_dome.sky_intensity = 4.0,
        },
        Wired {
            path: "lighting.sky_dome.camera_fill",
            perturb: |s| s.lighting.sky_dome.camera_fill = 5.0,
        },
        Wired {
            path: "lighting.sky_dome.rim_fill",
            perturb: |s| s.lighting.sky_dome.rim_fill = 5.0,
        },
    ]
}

/// Leaves NOT proven by this test, each with a written reason. These are the
/// settings that have no deterministic visible effect on the trivial sphere
/// scene, OR are not yet consumed by the native still path. They are tracked
/// here (not silently ignored): the coverage assertion makes the union of
/// WIRED + KNOWN_UNWIRED equal the full leaf set, so a NEW leaf cannot slip in
/// unclassified.
///
/// NOTE: an entry here is a debt marker, not an endorsement — the design's
/// goal is to shrink this list as each capability gets a real fixture.
const KNOWN_UNWIRED: &[(&str, &str)] = &[
    // version is schema metadata, not a render input.
    ("version", "schema metadata — never affects pixels by design"),
    // backend.mode selects Vulkan vs CUDA; on this AMD box only Vulkan exists,
    // and the selector still hard-prefers Vulkan (Phase 2 integrator note).
    (
        "backend.mode",
        "backend selector not yet honored (Phase 2 note: force_* not wired into spawn); AMD box is Vulkan-only",
    ),
    // resolution DOES change output, but it changes the buffer length so a hash
    // diff is trivially guaranteed and proves nothing about wiring — excluded
    // to keep the gate meaningful (it is exercised implicitly: render uses it).
    (
        "render.resolution",
        "drives buffer size; consumed (render_with_settings uses it) but a size-change hash diff is not a wiring proof",
    ),
    // max_bounces reaches the kernel (the bind diag prints u_max_bounces), but
    // an isolated diffuse sphere has no surrounding geometry to bounce light
    // off, so 1 vs 8 bounces render identically. Wired-but-invisible on this
    // fixture; a proof needs an enclosed (Cornell-box-style) scene.
    (
        "render.max_bounces",
        "reaches the kernel (u_max_bounces in bind diag) but an isolated sphere has no indirect-bounce geometry; needs an enclosed scene",
    ),
    // denoiser: near_realtime default already has a denoiser mode; toggling
    // enabled=false maps to DenoiserMode::None (wired in apply_settings), but on
    // a 1-spp-converged flat sphere the denoised vs raw frames can hash-collide.
    // Strength/backend are sub-leaves of the same not-reliably-visible stage.
    (
        "render.denoiser.enabled",
        "wired in apply_settings (off→DenoiserMode::None) but no reliable visible delta on a flat sphere; needs a noisy fixture",
    ),
    (
        "render.denoiser.strength",
        "denoiser blend strength — same fixture limitation as denoiser.enabled",
    ),
    (
        "render.denoiser.backend",
        "SVGF/OptiX selector — the backend probe owns realizability; not a per-pixel knob",
    ),
    // upscaler: not consumed by the native still path at all (realtime-only).
    (
        "render.upscaler.mode",
        "realtime upscaler (FSR/DLSS) — not consumed by the offline still path; A2 seam",
    ),
    (
        "render.upscaler.quality",
        "realtime upscaler quality — not consumed by the offline still path",
    ),
    // look.tonemap LUT + color grade / white balance / levels are DEAD in the
    // native path today (design §B/§67: Cdl/LiftGammaGain exist but are unwired
    // in the native render). Tracked as the next wiring target.
    (
        "look.tonemap.lut",
        "display LUT path not loaded by the native renderer yet (dead-in-native — design §67)",
    ),
    (
        "look.tonemap.lut_strength",
        "LUT blend strength — dead until the LUT path is wired",
    ),
    (
        "look.color.enabled",
        "ASC-CDL color grade unwired in native path (design §67 — Cdl exists but dead)",
    ),
    ("look.color.value.slope", "CDL slope — dead until color grade is wired"),
    ("look.color.value.offset", "CDL offset — dead until color grade is wired"),
    ("look.color.value.power", "CDL power — dead until color grade is wired"),
    (
        "look.color.value.saturation",
        "CDL saturation — dead until color grade is wired",
    ),
    (
        "look.white_balance.enabled",
        "white balance unwired in native path (design §67)",
    ),
    (
        "look.white_balance.value.temperature_k",
        "white balance temperature — dead until WB is wired",
    ),
    (
        "look.white_balance.value.tint",
        "white balance tint — dead until WB is wired",
    ),
    (
        "look.levels.enabled",
        "Lift/Gamma/Gain unwired in native path (design §67 — LiftGammaGain dead)",
    ),
    ("look.levels.value.lift", "LGG lift — dead until levels is wired"),
    ("look.levels.value.gamma", "LGG gamma — dead until levels is wired"),
    ("look.levels.value.gain", "LGG gain — dead until levels is wired"),
    // lighting.sun.radiance only matters when atmosphere is enabled.
    (
        "lighting.sun.radiance",
        "only consumed by the atmosphere in-scatter path; same fixture limit as atmosphere.enabled",
    ),
    // atmosphere: enabling routes set_sun/set_atmosphere (verified: the setters
    // run), but the sphere occludes the sky in this framing so the in-scatter is
    // not visible. Wired-but-invisible on this fixture; a proof needs a
    // sky-filling shot (camera pointed at open sky).
    (
        "lighting.atmosphere.enabled",
        "set_sun/set_atmosphere run when enabled, but the sphere occludes the sky here; needs a sky-filling shot",
    ),
    (
        "lighting.atmosphere.mie",
        "atmosphere anisotropy — needs a sky-filling fixture (sphere occludes in-scatter)",
    ),
    (
        "lighting.atmosphere.turbidity",
        "atmosphere turbidity — needs a sky-filling fixture (sphere occludes in-scatter)",
    ),
    (
        "lighting.atmosphere.hdri",
        "HDRI environment path not loaded by this fixture (empty default = analytic model)",
    ),
    // features.*: wired onto the config in apply_settings, but a single diffuse
    // sphere does not exercise restir/nrc/path-guide/mnee/lpe/photon/ser/mega/
    // spectral in a way that deterministically moves 8-bit pixels. They are
    // unit-proven to reach the config in spectra-renderer settings_tests; a
    // visible-delta proof needs a caustics/GI fixture.
    ("features.restir.enabled", "reaches config (settings_tests); no visible delta on a flat sphere"),
    ("features.nrc.enabled", "reaches config; converged radiance cache invisible on a flat sphere"),
    ("features.niv.enabled", "reaches config; needs a volumetric/irradiance fixture"),
    ("features.path_guide.enabled", "reaches config; guiding only changes variance, not the mean"),
    ("features.mnee.enabled", "reaches config; needs a caustics fixture"),
    ("features.lpe.enabled", "reaches config; LPE writes AOVs, not the beauty buffer"),
    ("features.photon_map.enabled", "reaches config; needs a caustics fixture"),
    ("features.ser.enabled", "reaches config; SER is a scheduling optimization — identical pixels by design"),
    ("features.mega_geometry.enabled", "reaches config; CLAS path needs a mega-geometry fixture"),
    ("features.spectral.enabled", "reaches config (SpectralMode::Hero4); needs a dispersive fixture"),
    ("features.weathering.enabled", "weathering masks are a mesh-path input; not used by the SDF fixture"),
    // material.default_uv_scale only affects textured surfaces; the SDF sphere
    // is untextured flat albedo.
    (
        "material.default_uv_scale",
        "applies to textured materials; the SDF fixture is untextured flat albedo",
    ),
    // camera.*: the config's default-camera section feeds a no-explicit-camera
    // render via RenderConfig::camera_layer_from_settings. This fixture instead
    // builds its own framing camera (front-quarter view) so the sphere fills the
    // frame deterministically — so the config camera leaves are not the ones
    // driving these renders. They are unit-proven to reach a CameraLayer in
    // spectra-renderer's `default_camera_settings_build_a_camera_layer` /
    // `camera_override_reaches_layer`; a pixel-delta proof here needs the fixture
    // to drive its camera from settings.camera instead of the fixed eye.
    (
        "camera.position",
        "default-camera seam (camera_layer_from_settings) is unit-proven in spectra-renderer; this fixture uses its own framing camera",
    ),
    (
        "camera.target",
        "default-camera seam unit-proven in spectra-renderer; fixture uses its own framing camera",
    ),
    (
        "camera.up",
        "default-camera seam unit-proven in spectra-renderer; fixture uses its own framing camera",
    ),
    (
        "camera.fov_y_deg",
        "default-camera seam unit-proven in spectra-renderer; fixture uses its own framing camera",
    ),
    (
        "camera.near",
        "default-camera seam unit-proven in spectra-renderer; fixture uses its own framing camera",
    ),
    (
        "camera.far",
        "default-camera seam unit-proven in spectra-renderer; fixture uses its own framing camera",
    ),
    (
        "camera.lens_radius",
        "DoF lens radius reaches the CameraLayer (unit-proven); the fixture camera is pinhole",
    ),
    (
        "camera.focus_distance",
        "DoF focus distance reaches the CameraLayer (unit-proven); only used when lens_radius > 0",
    ),
];

/// Enumerate every logical leaf path in the serialized default `RenderSettings`
/// tree (dotted path). An array (e.g. `lighting.sun.dir`, `[f32; 3]`) is ONE
/// logical leaf — perturbing any element is wiring-equivalent to perturbing the
/// vector — so array elements collapse to their containing field's path rather
/// than expanding to `field[i]`. This is the ground-truth set the coverage
/// assertion must fully account for.
fn all_leaf_paths() -> Vec<String> {
    fn walk(prefix: &str, v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, sv) in map {
                    let p = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    walk(&p, sv, out);
                }
            }
            // An array is a single logical leaf (the whole vector). Its scalar
            // elements do not get their own `[i]` paths.
            serde_json::Value::Array(_) => out.push(prefix.to_string()),
            _ => out.push(prefix.to_string()),
        }
    }
    let json = serde_json::to_value(RenderSettings::default()).expect("serialize default");
    let mut out = Vec::new();
    walk("", &json, &mut out);
    out.sort();
    out.dedup();
    out
}

// =============================================================================
// The gate.
// =============================================================================

/// Coverage check (no GPU): every leaf is classified, with no overlap and no
/// gaps. This is the part that FORCES a decision on a newly-added leaf even on a
/// box without a GPU.
#[test]
fn perturbation_coverage_classifies_every_leaf() {
    let all: std::collections::BTreeSet<String> = all_leaf_paths().into_iter().collect();
    let wired: std::collections::BTreeSet<String> =
        wired_leaves().into_iter().map(|w| w.path.to_string()).collect();
    let unwired: std::collections::BTreeSet<String> =
        KNOWN_UNWIRED.iter().map(|(p, _)| p.to_string()).collect();

    // No leaf may be both wired and explicitly-unwired.
    let overlap: Vec<&String> = wired.intersection(&unwired).collect();
    assert!(
        overlap.is_empty(),
        "leaves classified as BOTH wired and unwired: {overlap:?}"
    );

    let classified: std::collections::BTreeSet<String> =
        wired.union(&unwired).cloned().collect();

    // Every classification path must be a real leaf (catches typos / renames).
    let phantom: Vec<&String> = classified.difference(&all).collect();
    assert!(
        phantom.is_empty(),
        "classified paths that are NOT real RenderSettings leaves (typo/rename?): {phantom:?}"
    );

    // Every real leaf must be classified — THE dead-settings guard. A new leaf
    // bound nowhere lands here unclassified and fails the gate, naming itself.
    let unclassified: Vec<&String> = all.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "UNCLASSIFIED RenderSettings leaves (wire them + prove a pixel delta, or add to KNOWN_UNWIRED with a reason): {unclassified:?}"
    );

    eprintln!(
        "[perturbation] {} leaves total: {} WIRED (pixel-delta proven), {} KNOWN_UNWIRED (documented)",
        all.len(),
        wired.len(),
        unwired.len()
    );
}

/// The GPU gate: for every WIRED leaf, perturbing it must change the rendered
/// frame relative to the default-settings baseline. A WIRED leaf that produces
/// an IDENTICAL hash is a dead setting → FAIL, naming it.
///
/// Done-When (design Phase 5): this passes for the wired set; temporarily
/// unwiring one (e.g. deleting `self.exposure_ev = s.look.exposure_ev` in
/// `RenderConfig::apply_settings`) makes `look.exposure_ev` hash-match the
/// baseline and FAILS here, printing the dead-setting name.
#[test]
fn perturbation_coverage() {
    // Small frames keep the ~15 renders fast; the look/lighting deltas are
    // resolution-independent.
    let base_settings = {
        let mut s = RenderSettings::default();
        s.render.resolution = [96, 96];
        // A finite spp so the baseline is a fixed pattern (near_realtime default
        // is already finite; pin it for determinism).
        s.render.spp = 16;
        s
    };

    let baseline = render_with_settings(&base_settings)
        .expect("baseline render (GPU/Vulkan required — run with spectra-native + a GPU)");
    let base_hash = frame_hash(&baseline);
    let base_luma = mean_luma(&baseline);
    eprintln!(
        "[perturbation] baseline hash={base_hash:016x} mean_luma={base_luma:.2} ({}x{})",
        base_settings.render.resolution[0], base_settings.render.resolution[1]
    );

    let mut dead: Vec<String> = Vec::new();
    for w in wired_leaves() {
        let mut s = base_settings.clone();
        (w.perturb)(&mut s);
        // The perturbation must itself stay in-range (so it could come from a
        // real config file). validate() is the same gate resolve() runs.
        s.validate()
            .unwrap_or_else(|e| panic!("perturbation for '{}' is out of range: {e}", w.path));
        let frame = render_with_settings(&s)
            .unwrap_or_else(|e| panic!("render for '{}' failed: {e}", w.path));
        let h = frame_hash(&frame);
        let l = mean_luma(&frame);
        let changed = h != base_hash;
        eprintln!(
            "[perturbation] {:32} hash={:016x} mean_luma={:6.2} -> {}",
            w.path,
            h,
            l,
            if changed { "CHANGED" } else { "DEAD (identical to baseline!)" }
        );
        if !changed {
            dead.push(w.path.to_string());
        }
    }

    assert!(
        dead.is_empty(),
        "DEAD SETTINGS — these WIRED leaves did not change the render (bound nowhere?): {dead:?}"
    );
    eprintln!(
        "[perturbation] all {} wired leaves moved the render — no dead settings.",
        wired_leaves().len()
    );
}
