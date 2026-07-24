//! Resident Spectra path-tracer for the live window (Render Keystone T1).
//!
//! [`ResidentSceneRenderer`] constructs the Vulkan backend + `Renderer` +
//! `KernelSet` **ONCE** (paying the device init + ~9.3s slangc compile + initial
//! BLAS/TLAS build), then streams camera + dirty scene deltas per frame. This is
//! the opposite of the legacy still path (`pathtrace_mesh_lit_*`), which rebuilt
//! the entire renderer on every call — structurally unable to hit the 30fps
//! real-time bar on the 780M.
//!
//! It generalizes [`crate::splat_backend::spectra_resident_bench`]'s construct-once
//! setup into a reusable object: `new` does the one-time build, `set_scene` does a
//! dirty-layer-aware upload (reporting rebuilt/reused via [`SceneSyncReport`]), and
//! `render_camera` is a pure camera-stream frame (`set_camera_view_matrix` +
//! `set_view_proj` + `render()` — NO backend/renderer reconstruction).
//!
//! Owned single-threaded by the game's `App`/`GameView` alongside `present`
//! (decision §8 of the design): no `Arc<Mutex<>>`, deterministic frames.

#![cfg(feature = "spectra-native")]

#[cfg(all(target_os = "windows", feature = "spectra-native-optix"))]
use spectra_gpu::CudarcSlangBackend;
use spectra_gpu::GpuBufferHandle;
#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
use spectra_gpu::SharedMetalBackend;
#[cfg(not(any(
    all(target_os = "windows", feature = "spectra-native-optix"),
    all(target_os = "macos", feature = "spectra-native-metal")
)))]
use spectra_gpu::SharedVulkanBackend;
pub use spectra_renderer::FrameOutput;
use spectra_renderer::{RenderConfig, RenderSettings, Renderer, RendererTexture2D};

/// The LIVE path-tracer compute backend. The standing rule: use CUDA on NVIDIA
/// when available. On the Windows CUDA box built WITH `spectra-native-optix`
/// this is the cudarc CUDA backend (RT cores + DLSS + OptiX); elsewhere — the
/// Linux/AMD dev box, OR a Windows build WITHOUT the optix feature (the
/// KHR-water witness build) — it is the Vulkan backend. Gating on the feature
/// (not just the OS) lets the box exercise the VK_KHR animated-water path. The
/// window present is a separate, thin swapchain (display only), independent of this.
#[cfg(all(target_os = "windows", feature = "spectra-native-optix"))]
type ResidentBackend = CudarcSlangBackend;
#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
type ResidentBackend = SharedMetalBackend;
#[cfg(not(any(
    all(target_os = "windows", feature = "spectra-native-optix"),
    all(target_os = "macos", feature = "spectra-native-metal")
)))]
type ResidentBackend = SharedVulkanBackend;
use spectra_scene_state::{GpuSceneCmd, LightLayer, OP_SET_TRANSFORM, SceneDeltaRing, SceneState};

/// Re-export the R31 fidelity tier and tier-table types so the game layer can
/// select a tier and load `render.ron` fidelity overrides through `vox_render`
/// without depending on `spectra-renderer` or `spectra-types` directly.
pub use spectra_renderer::FidelityTier;
pub use spectra_renderer::RenderTarget;
pub use spectra_renderer::{TierEntry, TierTable, UpscalerMode, UpscalerQuality};

use crate::mega_geometry::{
    ForbiddenGeometryCpuActivity, GeometryCpuOwnershipCounters, GeometryCpuOwnershipSnapshot,
    GeometryHostServiceKind,
};
use crate::scene_delta_adapter::{RetainedDeltaError, RetainedDeltaPlan, RetainedRenderMirror};
use crate::splat_backend::{
    LIGHT_FLOATS, LightRig, pack_directional_light, pack_point_light, pack_spot_light,
    pack_sun_disk_light, resolve_slang_kernel_dir, rig_to_settings, seed_features_from_config,
    sun_solid_angle,
};

/// Result of a scene-delta upload — the reuse-vs-rebuild proof.
///
/// `layers_rebuilt == 0 && layers_reused > 0` means the GPU scene was reused
/// (no per-frame BLAS rebuild) — the witness that the resident path is alive.
#[derive(Debug, Clone, Copy)]
pub struct SceneSyncReport {
    layers_rebuilt: u32,
    layers_reused: u32,
}

impl SceneSyncReport {
    /// Number of geometry layers whose BLAS was rebuilt on this upload.
    pub fn layers_rebuilt(&self) -> u32 {
        self.layers_rebuilt
    }
    /// Number of geometry layers whose BLAS was reused (unchanged).
    pub fn layers_reused(&self) -> u32 {
        self.layers_reused
    }
}

/// Resident path-tracer for the live window. Constructed ONCE; streams camera +
/// dirty deltas. Single-threaded with the redraw loop (owned by the game).
pub struct ResidentSceneRenderer {
    /// The renderer, constructed once: KernelSet, SceneUploader, GpuStage (BLAS
    /// cache + per-layer fingerprints), RenderState, AtmosphereManager. Reused
    /// across every frame.
    renderer: Renderer<ResidentBackend>,
    /// Internal render resolution.
    width: u32,
    height: u32,
    /// Light rig — drives the four-light scene rig + sky/atmosphere setters.
    rig: LightRig,
    /// Live multiplier for authored emissive materials. Mirrored into Spectra as
    /// `u_emissive_scale` and used while deriving emissive point lights so
    /// direct-hit emission and NEE lights stay in lockstep.
    emissive_scale: f32,
    /// Aurora Step 1: the id-sorted, coalesced (latest-wins-per-slot) command
    /// ring for resident transform deltas. Backed by a `BTreeMap<slot, …>` so
    /// the drained stream is ALWAYS ascending-slot with one command per slot —
    /// never HashMap/RNG/allocator order. Flushed on `render_camera` through the
    /// GPU `apply_scene_delta` indexed scatter (ONE buffer upload + ONE
    /// dispatch), REPLACING the old per-frame CPU `pending_refits` Vec push+sort
    /// +drain. The indexed scatter is order-independent, so the resident buffer
    /// is a pure function of this id-sorted command log (the determinism moat).
    delta_ring: SceneDeltaRing,
    /// Engine-owned mirror for the NodeId→instance_index mapping of the scene
    /// currently uploaded. Populated by `reset_retained_mirror` after every
    /// `set_scene`; drives `drain_scene_deltas` so the game only emits
    /// `Vec<vox_scene::SceneDelta>`.
    retained_mirror: RetainedRenderMirror,
    /// Last combined view-projection rendered by the live camera stream. Static
    /// cameras may progressively accumulate; camera motion must clear the
    /// progressive film and temporal histories without rebuilding the renderer.
    last_view_proj: Option<[f32; 16]>,
    /// Last projection matrix, tracked separately so tiny view-matrix float drift
    /// does not look like a projection/camera cut.
    last_projection: Option<[f32; 16]>,
    /// Last camera view matrix, used to distinguish a real cut/teleport from
    /// continuous orbit/pan motion.
    last_camera_view: Option<[f32; 16]>,
    /// Previous camera transform retained for external temporal consumers
    /// (DLSS-G); unlike internal accumulation this survives until the following
    /// rendered frame has consumed it.
    previous_view_proj: Option<[f32; 16]>,
    /// Reset decision associated with the most recently rendered camera frame.
    /// External temporal consumers must use the same cut decision as Spectra.
    last_camera_reset: bool,
    /// Monotonic traced-frame identity exported to present backends.
    present_frame_index: u32,
    /// Fail-closed ownership evidence shared by every game using the resident
    /// renderer. Allowed native submissions are measured separately from
    /// forbidden CPU geometry decisions. Fed by [`Self::bridge_geometry_cpu_ownership`]
    /// from Spectra's live probe (see `geometry_cpu_bridge`), so a MegaGeometry
    /// forbidden CPU op the renderer records is reflected in this witness rather
    /// than sitting vacuously at zero.
    geometry_cpu_ownership: GeometryCpuOwnershipCounters,
    /// Last-seen cumulative values of Spectra's `geometry_cpu_contract` probe.
    /// The bridge folds the positive delta since this baseline into
    /// `geometry_cpu_ownership`, so double-counting across frames is impossible.
    geometry_cpu_bridge: GeometryCpuOwnershipBridge,
}

/// Monotonic baseline for the Spectra→engine CPU-ownership counter bridge.
#[derive(Default)]
struct GeometryCpuOwnershipBridge {
    /// Indexed by `ForbiddenGeometryCpuActivity` discriminant (engine and Spectra
    /// contracts share the discriminant layout, so the index maps 1:1).
    forbidden: [u64; ForbiddenGeometryCpuActivity::ALL.len()],
    native_calls: u64,
    native_ns: u64,
    page_calls: u64,
    page_ns: u64,
}

impl ResidentSceneRenderer {
    /// Construct ONCE. Pays device init + the ~9.3s `KernelSet` slangc compile +
    /// the initial BLAS/TLAS build for `initial`. Mirrors the construct-once
    /// setup of `spectra_resident_bench` (`splat_backend.rs:1007-1037`) as a
    /// reusable object.
    ///
    /// `spp`/`max_bounces` are the real-time levers (forced into the config);
    /// `width`/`height` is the internal render resolution.
    pub fn new(
        width: u32,
        height: u32,
        rig: LightRig,
        spp: u32,
        max_bounces: u32,
        initial: SceneState,
    ) -> Result<Self, String> {
        // The legacy positional spp/bounces seam, expressed on top of the tier
        // path: start from Balanced, then force the caller's explicit
        // spp/bounces. Keeps every existing caller/test byte-compatible while
        // routing through the single tier-aware construction body.
        let mut settings = RenderSettings::for_tier(FidelityTier::Balanced);
        settings.render.spp = spp;
        settings.render.max_bounces = max_bounces;
        Self::new_from_settings(width, height, rig, settings, max_bounces, initial)
    }

    /// Construct ONCE for a [`FidelityTier`] (R31 fidelity ladder). The tier
    /// deterministically sets the path tracer's cost knobs (spp, bounces,
    /// spectral, GI/ReSTIR, denoiser) via [`RenderSettings::for_tier`]; the
    /// `rig`'s lighting/look values are layered in. The path tracer is ALWAYS
    /// the renderer — the tier only scales its per-frame cost. `width`/`height`
    /// is the internal render resolution (the caller derives it from
    /// `tier.internal_max_width()`; FSR upscales to the display on the
    /// real-time tiers).
    pub fn new_with_tier(
        width: u32,
        height: u32,
        rig: LightRig,
        tier: FidelityTier,
        tier_table: Option<&spectra_renderer::TierTable>,
        initial: SceneState,
    ) -> Result<Self, String> {
        let settings = match tier_table {
            Some(table) => RenderSettings::for_tier_from_ron(tier, table),
            None => RenderSettings::for_tier(tier),
        };
        let max_bounces = settings.render.max_bounces;
        Self::new_from_settings(width, height, rig, settings, max_bounces, initial)
    }

    /// Shared construction body. `tier_settings` carries the cost knobs (spp,
    /// bounces, spectral, denoiser, restir) which WIN over the rig; the rig's
    /// lighting/look values are merged in on top. `max_bounces` is forced into
    /// the config last (it lives outside `apply_settings`'s settings mapping for
    /// some paths).
    fn new_from_settings(
        width: u32,
        height: u32,
        rig: LightRig,
        tier_settings: RenderSettings,
        max_bounces: u32,
        initial: SceneState,
    ) -> Result<Self, String> {
        let gpu = ResidentBackend::new(0).map_err(|e| format!("gpu backend init: {e:?}"))?;
        // CONFIG-FIRST: the live present-path gates default from `config/ochroma.ron`
        // (`vox_config`); a still-present env var (the documented A/B "sweep lever")
        // wins over the config value, which wins over the former hardcoded literal.
        // Every config default equals the old literal, so with no env + no ron the
        // frame is byte-identical.
        let rcfg = &vox_config::config().resident_renderer;
        let mut config = RenderConfig::near_realtime(width, height);
        // CONFIG-FIRST (spectra.ron): layer the engine-side single source of truth
        // UNDER the game's authoritative render.ron + realtime preset. `apply_to`
        // maps every spectra-owned RenderConfig field (relief/ground/terrain/
        // denoise/feature-flags/dlss/…) from spectra.ron; the realtime budget + the
        // authoritative LOOK are then re-asserted ON TOP — the tier via
        // `apply_settings` (spp/bounces/tonemap/exposure/grade/denoiser), the
        // explicit overrides below (optix/ground/temporal/restir/lean), so this
        // NEVER clobbers them. With the shipped spectra.ron (defaults == prior
        // literals) the mapped values EQUAL RenderConfig::default(), so the resident
        // frame is byte-identical until spectra.ron is edited. The only fields
        // near_realtime sets that apply_to would change AND nothing downstream
        // re-touches are `mode` + `convergence_threshold` — preserved here.
        {
            use spectra_renderer::spectra_config::ApplyToRenderConfig;
            let rt_mode = config.mode;
            let rt_convergence = config.convergence_threshold;
            // PRESERVE the caller's internal render resolution. `near_realtime` set
            // width/height from the (iw, ih) the game passed (res_in_override /
            // render.ron resolution_in / tier cap). `apply_to` maps spectra.ron
            // `sampling.{width,height}` (default 1920x1080 = the OFFLINE film size,
            // NOT the live internal res) OVER them, silently forcing the live
            // path-trace to 1080p regardless of the requested res — the ~5fps-at-4K
            // bug. Re-assert the caller's value, like mode + convergence below.
            let rt_width = config.width;
            let rt_height = config.height;
            spectra_renderer::spectra_config::config().apply_to(&mut config);
            config.mode = rt_mode;
            config.convergence_threshold = rt_convergence;
            config.width = rt_width;
            config.height = rt_height;
        }
        config.slang_kernel_dir = resolve_slang_kernel_dir();
        // Lean shade kernel: byte-identical for pure triangle-mesh city scenes (the
        // heavy SSS/volume/polarization/SDF/Gaussian paths are off on the resident
        // config) and it skips the NRC-coupled heavy path. With NRC off on the
        // real-time tiers, make lean unconditional for the resident render.
        config.prefer_lean_shade = rcfg.prefer_lean_shade;
        let _ = std::env::var("OCHROMA_SHADE_LEAN"); // (legacy opt-in, now default-on)

        // Start from the rig (lighting, look, weathering toggle), seed the
        // near_realtime feature parity, THEN overlay the tier's cost knobs so
        // the tier — not the rig and not the base preset — owns spp / bounces /
        // spectral / ReSTIR / denoiser. This ordering is load-bearing:
        // rig_to_settings writes spp/bounces, so the tier overwrite must follow.
        let mut settings = rig_to_settings(&rig, tier_settings.render.spp, max_bounces);
        seed_features_from_config(&mut settings, &config);
        // Tier wins on the cost knobs.
        settings.render.spp = tier_settings.render.spp;
        settings.render.max_bounces = tier_settings.render.max_bounces;
        settings.render.denoiser = tier_settings.render.denoiser.clone();
        settings.render.upscaler = tier_settings.render.upscaler.clone();
        settings.features.spectral = tier_settings.features.spectral.clone();
        settings.features.restir = tier_settings.features.restir.clone();
        config.apply_settings(&settings);
        config.max_bounces = max_bounces;

        // NRC world-space radiance cache (online-trained; the inline shade hook
        // short-circuits deep bounces once cells warm up). Env-gated A/B toggle —
        // SPECTRA_NRC=1 forces ON (compiles the FEATURE_NRC shade variant + binds the
        // cache/stash + dispatches nrc_harvest), =0 forces OFF (byte-identical). Default
        // = whatever the tier/config set (currently false). SPECTRA_NRC_BOUNCE sets the
        // query/short-circuit depth (paths shorter than this stay full unbiased PT).
        if let Ok(v) = std::env::var("SPECTRA_NRC") {
            let t = v.trim();
            config.use_nrc = t == "1" || t.eq_ignore_ascii_case("true");
        }
        if let Some(b) = std::env::var("SPECTRA_NRC_BOUNCE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
        {
            config.nrc_query_bounce = b;
        }

        // GLASS BOUNCE FLOOR (SOTA glass fix, now TIER-AWARE — the #1 perf knob).
        // The still path (`splat_backend` `pathtrace_mesh_lit_weathered_to_rgba`,
        // ~line 934) bumps `max_bounces` to 8 whenever the scene contains a
        // transmissive (MAT_GLASS) material, because a glass pane needs: enter
        // front face (refract) → exit back face (refract) → travel to a lit
        // surface → bounce off it → back through the pane. The real-time tiers
        // only give 2 (Performance) / 3 (Balanced) / 6 (Beauty) bounces, so
        // transmitted rays die INSIDE the glass before reaching any light and the
        // pane reads dark/flat-matte instead of transmissive+reflective. But a
        // FLAT floor of 8 forced every glassy city onto the slowest path
        // regardless of tier (8->2 is ~-58% sample_loop), so the RESIDENT path
        // now derives the floor from the TIER (see `glass_floor_for_tier`):
        // Performance gets 4, Balanced 5, Beauty keeps 8 — the lowest depth each
        // tier needs to still refract through both faces and reach light. The
        // RESIDENT path (the live game + the STYLE_PROBE/SHOT_HERO witness) never
        // applied any bump, so curtain-wall glass looked opaque. Mirror the still
        // path here but with the tier-aware floor:
        // scan the INITIAL scene's material table for any MAT_GLASS material and,
        // if present, raise the bounce floor. Determinism-safe: a pure `any()`
        // over the flat param array (no HashMap/RNG iteration); changes only the
        // path-length budget, never the sample order. `SPECTRA_GLASS_BOUNCES`
        // overrides the floor (config-first); 0 disables the bump entirely.
        // TRANSMISSIVE BOUNCE FLOORS — glass and water are now floored SEPARATELY.
        // Architectural glass needs depth to refract through both faces and reach
        // light (~4-8); the SEA surface needs far less (a surface refraction + one
        // lit hop, ~3-4) yet used to be lumped with glass and forced max_bounces->8
        // even on a near-empty map with only water — the dominant trace cost. We
        // now raise the floor to the MAX of the applicable per-material floors, so
        // a glassy city keeps full glass depth while a water-only map pays only the
        // (much cheaper) water floor. Each has an env override (SPECTRA_GLASS_BOUNCES
        // / SPECTRA_WATER_BOUNCES) > config override (>=0) > per-tier floor; a config
        // default of -1 = use the tier floor, 0 = disable that material's bump.
        let has_glass = scene_has_mat_type(&initial, MAT_GLASS_TYPE);
        let has_water = scene_has_mat_type(&initial, MAT_WATER_TYPE);
        if has_glass || has_water {
            let mut floor = config.max_bounces;
            if has_glass {
                let g = std::env::var("SPECTRA_GLASS_BOUNCES")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .or_else(|| {
                        let ov = rcfg.glass_bounces_override;
                        if ov >= 0 { Some(ov as u32) } else { None }
                    })
                    .unwrap_or_else(|| glass_floor_for_tier(config.max_bounces));
                floor = floor.max(g);
            }
            if has_water {
                let wf = std::env::var("SPECTRA_WATER_BOUNCES")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .or_else(|| {
                        let ov = rcfg.water_bounces_override;
                        if ov >= 0 { Some(ov as u32) } else { None }
                    })
                    .unwrap_or_else(|| water_floor_for_tier(config.max_bounces));
                floor = floor.max(wf);
            }
            if floor > config.max_bounces {
                eprintln!(
                    "[glass-bounce-floor] scene has {}; max_bounces {} -> {} \
                     (transmissive panes + the sea surface need enough depth to refract \
                     and reach light; glass and water floored separately)",
                    match (has_glass, has_water) {
                        (true, true) => "MAT_GLASS+MAT_WATER",
                        (true, false) => "MAT_GLASS",
                        _ => "MAT_WATER",
                    },
                    config.max_bounces,
                    floor
                );
                config.max_bounces = floor;
            }
        }

        // NVIDIA-stack foundation: enable OptiX HW-RT for the LIVE game by default
        // (config-first, not env-gated). ResidentSceneRenderer historically never
        // set this, so the shipped game ran the SOFTWARE BVH and the whole RT-core
        // stack (CLAS/Mega-Geometry, HW TLAS, ReSTIR-PT quality, DLSS-RR guides)
        // was dead code. The construct gate in spectra renderer/mod.rs
        // (`want_optix && raw_context()!=0 && should_use_optix_rt()`) safely
        // rejects this on AMD/Vulkan/no-CUDA → software fallback, so it's correct
        // on the dev 780M too. SPECTRA_USE_OPTIX_RT remains an override.
        config.use_optix_rt = rcfg.use_optix_rt;

        // GROUND ANTI-TILING (FIX 1) — the stochastic texture-bombing knobs the
        // megakernel ground path reads (`u_ground_antitile_*`). The spectra
        // RenderConfig defaults (strength 0.85, cell 18 m) ship the working look;
        // SPECTRA_GROUND_ANTITILE_STRENGTH / _CELL_M override them on the box for a
        // no-rebuild witness sweep (config-first; mirrors the OCHROMA_GROUND_UV /
        // SPECTRA_GLASS_BOUNCES override pattern). The game render.ron `ground`
        // block is the authored source-of-truth; these env vars are the sweep lever.
        if let Some(s) = std::env::var("SPECTRA_GROUND_ANTITILE_STRENGTH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
        {
            config.ground_antitile_strength = s;
        }
        if let Some(c) = std::env::var("SPECTRA_GROUND_ANTITILE_CELL_M")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            config.ground_antitile_cell_m = c;
        }

        // GROUND DE-FACET (FIX 1) — blend the faceted ground shading normal toward
        // world-up (`u_ground_smooth_normal`). The spectra RenderConfig default
        // (1.0 = smooth) ships the fix; SPECTRA_GROUND_SMOOTH_NORMAL is the runtime
        // override the game sets from its render.ron `ground.smooth_normal` (config-
        // first; same bridge pattern as the antitile knobs above).
        if let Some(s) = std::env::var("SPECTRA_GROUND_SMOOTH_NORMAL")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        {
            config.ground_smooth_normal = s;
        }

        // R1 TEMPORAL ACCUMULATION (the keystone real-time-quality fix). At 1 spp
        // a single-frame spatial denoise is blotchy/flickery under motion; the
        // temporal_reproject kernel reprojects the previous displayed frame's
        // accumulated color through prev_view_proj and EMA-blends it, giving an
        // effective ~10 spp without raising sample count. Cross-vendor (pure
        // compute), and disocclusion-rejected so static surfaces converge while
        // moving/newly-revealed pixels fall through to the current frame.
        //
        // Default ON for the live resident render (this is what makes 1 spp
        // shippable). OCHROMA_TEMPORAL=off disables it for an A/B witness; the
        // alpha/threshold knobs stay config-first on RenderConfig.temporal.
        config.temporal.enabled = rcfg.temporal_enabled;
        match std::env::var("OCHROMA_TEMPORAL").as_deref() {
            Ok("off") | Ok("0") => {
                config.temporal.enabled = false;
                eprintln!(
                    "[temporal-override] OCHROMA_TEMPORAL=off -> temporal accumulation disabled"
                );
            }
            Ok("on") | Ok("1") => {
                config.temporal.enabled = true;
                eprintln!(
                    "[temporal-override] OCHROMA_TEMPORAL=on -> temporal accumulation enabled"
                );
            }
            _ => {}
        }
        // Optional config-first tuning overrides. Precedence: env (the A/B sweep
        // lever) > config value > spectra RenderConfig.temporal default. The config
        // sentinel -1.0 means "leave the spectra default" (identical to the old
        // behavior when the env var was unset).
        // The config sentinel -1.0 means "no override → keep the spectra default";
        // `(x != -1.0).then_some(x)` maps it to None (the normal-threshold valid
        // range includes -1.0, so a plain range filter could NOT distinguish the
        // sentinel — this guard does).
        if let Some(v) = std::env::var("OCHROMA_TEMPORAL_ALPHA_STATIC")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or((rcfg.temporal_alpha_static != -1.0).then_some(rcfg.temporal_alpha_static))
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        {
            config.temporal.alpha_static = v;
        }
        if let Some(v) = std::env::var("OCHROMA_TEMPORAL_ALPHA_MOVING")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or((rcfg.temporal_alpha_moving != -1.0).then_some(rcfg.temporal_alpha_moving))
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        {
            config.temporal.alpha_moving = v;
        }
        if let Some(v) = std::env::var("OCHROMA_TEMPORAL_DEPTH_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or((rcfg.temporal_depth_threshold != -1.0).then_some(rcfg.temporal_depth_threshold))
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            config.temporal.depth_threshold = v;
        }
        if let Some(v) = std::env::var("OCHROMA_TEMPORAL_NORMAL_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or((rcfg.temporal_normal_threshold != -1.0).then_some(rcfg.temporal_normal_threshold))
            .filter(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        {
            config.temporal.normal_threshold = v;
        }

        // PHASE R0 measurement override: OCHROMA_RESTIR=on|off forces the ReSTIR
        // DI + PT path on/off at runtime WITHOUT permanently changing the tier,
        // so the `--compare-map` A/B harness can render the same scene both ways.
        // `apply_settings` maps features.restir -> use_restir but leaves
        // resample_mode untouched (a wiring gap the audit flagged); the PT path
        // only runs when resample_mode != None, so we set BOTH here. Temporal-only
        // is the R0 mode (spatial is the R1 follow-up). Unset => leave as resolved.
        // env OCHROMA_RESTIR > config restir_override. "auto" (the default) leaves
        // the tier-resolved value untouched (identical to the old unset behavior).
        // Clean per-variant toggles: SPECTRA_RESTIR_{DI,GI,PT}=1 select WHICH single
        // variant the OCHROMA_RESTIR=on A/B pass enables (so DI/GI/PT are measured in
        // isolation, not stacked). When none is named the default variant is PT (the
        // GRIS full-path reuse). All default OFF; the "off" branch force-clears every
        // flag so the OFF pass is byte-identical to the pre-ReSTIR path regardless of
        // which SPECTRA_RESTIR_* env the sweep recipe leaves set.
        let want_di = std::env::var("SPECTRA_RESTIR_DI").as_deref() == Ok("1");
        let want_gi = std::env::var("SPECTRA_RESTIR_GI").as_deref() == Ok("1");
        let want_pt = std::env::var("SPECTRA_RESTIR_PT").as_deref() == Ok("1");
        let default_pt = !want_di && !want_gi && !want_pt;
        // PT ON-pass resample mode is config-first via SPECTRA_PT_COMPARE_MODE
        // (temporal default | spatial => temporal+spatial).
        let pt_mode = match std::env::var("SPECTRA_PT_COMPARE_MODE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .replace('_', "")
            .as_str()
        {
            "spatial" | "temporalspatial" => spectra_renderer::ResampleMode::TemporalSpatial,
            _ => spectra_renderer::ResampleMode::Temporal,
        };
        let restir_sel = std::env::var("OCHROMA_RESTIR")
            .ok()
            .unwrap_or_else(|| rcfg.restir_override.clone());
        match restir_sel.as_str() {
            "on" | "1" => {
                config.use_restir = want_di;
                config.use_restir_gi = want_gi;
                config.resample_mode = if want_pt || default_pt {
                    pt_mode
                } else {
                    spectra_renderer::ResampleMode::None
                };
                eprintln!(
                    "[restir-override] OCHROMA_RESTIR=on -> DI={} GI={} PT={:?} (di={} gi={} pt={} default_pt={})",
                    config.use_restir,
                    config.use_restir_gi,
                    config.resample_mode,
                    want_di,
                    want_gi,
                    want_pt,
                    default_pt
                );
            }
            "off" | "0" => {
                config.use_restir = false;
                config.use_restir_gi = false;
                config.resample_mode = spectra_renderer::ResampleMode::None;
                eprintln!(
                    "[restir-override] OCHROMA_RESTIR=off -> ALL ReSTIR variants off (DI/GI/PT)"
                );
            }
            _ => {}
        }
        // PHASE R0 measurement override: SPECTRA_SHOT_SPP forces the per-frame
        // target spp (the `--compare-map` equal-time loop uses this to match the
        // ON/OFF passes to equal wall-clock). Unset => the tier's spp stands.
        // env SPECTRA_SHOT_SPP > config shot_spp_override. 0 (default) = use tier spp.
        let spp_sel = std::env::var("SPECTRA_SHOT_SPP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(rcfg.shot_spp_override);
        if spp_sel > 0 {
            config.target_spp = spp_sel;
            eprintln!("[spp-override] SPECTRA_SHOT_SPP={spp_sel} -> target_spp={spp_sel}");
        }
        // TDR GUARD (Windows WDDM 2s GPU watchdog): bound every path-trace
        // dispatch so no single launch trips the watchdog and kills the first
        // heavy city frame with DXGI_ERROR_DEVICE_HUNG (0x887A0007). The renderer
        // bands the camera→bounce pipeline into horizontal strips of
        // ceil(max_pixels_per_dispatch / width) rows; bands are an exact partition
        // (deterministic, top-to-bottom), so this changes timing only, never the
        // image. Config-first: SPECTRA_MAX_PIXELS_PER_DISPATCH overrides; default
        // 65_536 (~256×256 worth of threads) keeps each city megakernel launch far
        // under 2s while staying coarse enough to avoid per-band launch overhead.
        // Set to 0 to disable banding (e.g. on non-WDDM / hardware-TDR-disabled).
        config.max_pixels_per_dispatch = std::env::var("SPECTRA_MAX_PIXELS_PER_DISPATCH")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(rcfg.max_pixels_per_dispatch);

        // CUDA GRAPH POLICY: OFF for the live resident game path.
        //
        // Graph capture is a batch/multi-spp launch-amortization tool. In the live
        // city it captures the first displayed frame's full megakernel sequence and
        // can stall for seconds or minutes before the player sees anything. Spectra
        // now hard-disables graph capture in `RenderMode::Interactive`; this config
        // value stays false so the resident log and policy agree. The env override is
        // kept only as an A/B switch for non-interactive experiments.
        // env SPECTRA_CUDA_GRAPHS (0|1) > config use_cuda_graphs (default OFF).
        config.use_cuda_graphs = match std::env::var("SPECTRA_CUDA_GRAPHS").as_deref() {
            Ok("0") => false,
            Ok(_) => true,
            Err(_) => rcfg.use_cuda_graphs,
        };
        eprintln!("[cuda-graphs] use_cuda_graphs={}", config.use_cuda_graphs);

        // SER (Shader Execution Reordering): the 4-pass software reorder
        // (`ser_reorder.slang`) is wired in spectra but never fired on the live
        // resident path because `RenderConfig::near_realtime` leaves
        // `ser_enabled = false`. The reorder coheres divergent material/closest-
        // hit shading work across a warp before the shade megakernel, which is a
        // deterministic gather/scatter on a stable sort key and leaves the image
        // bit-identical, but it is not automatically a throughput win: the live
        // Meridian 1440p A/B measured 66.49 ms ON versus 50.76 ms OFF at 1 spp / four
        // bounces. Default OFF until a representative scene proves otherwise.
        // `SPECTRA_SER=1` remains the explicit A/B override. This mirrors
        // spectra's own SPECTRA_SER override (render_config.rs:1348) so both
        // layers agree.
        // SER is now a RUNTIME config field (was `#[cfg(target_os="windows")]`-
        // gated). Config-first: the default (false) applies on every OS; the SER
        // reorder kernel only actually fires on the CUDA box. It remains
        // image-identical, but its performance direction is scene-dependent.
        // env SPECTRA_SER (0|1) still overrides below.
        config.ser_enabled = rcfg.ser_enabled;
        match std::env::var("SPECTRA_SER").as_deref() {
            Ok("1") => {
                config.ser_enabled = true;
                eprintln!("[ser-override] SPECTRA_SER=1 -> ser_enabled=true");
            }
            Ok("0") => {
                config.ser_enabled = false;
                eprintln!("[ser-override] SPECTRA_SER=0 -> ser_enabled=false");
            }
            _ => {}
        }
        eprintln!("[ser] requested={}", config.ser_enabled);

        let renderer = Renderer::new(gpu, config);

        let mut me = Self {
            renderer,
            width,
            height,
            rig,
            emissive_scale: 1.0,
            delta_ring: SceneDeltaRing::new(),
            retained_mirror: RetainedRenderMirror::new(),
            last_view_proj: None,
            last_projection: None,
            last_camera_view: None,
            previous_view_proj: None,
            last_camera_reset: true,
            present_frame_index: 0,
            geometry_cpu_ownership: GeometryCpuOwnershipCounters::default(),
            geometry_cpu_bridge: GeometryCpuOwnershipBridge::default(),
        };
        // Upload the initial scene (with the rig's lights + sky/atmosphere) so the
        // first render_camera has geometry/materials/lights resident.
        me.upload_scene(initial)?;
        Ok(me)
    }

    /// Upload a scene delta. Dirty-layer aware via `gpu_stage.sync` — unchanged
    /// layers are NOT re-uploaded. Returns the rebuilt/reused counts (the reuse
    /// proof). The held rig's lights + sky/atmosphere are injected so lighting is
    /// consistent across scene edits.
    pub fn set_scene(&mut self, scene: SceneState) -> Result<SceneSyncReport, String> {
        self.upload_scene(scene)
    }

    /// Update the render-only light rig without rebuilding geometry. Directional
    /// sky/sun/moon/fog values are uniforms; the NEE light list is a light-layer
    /// upload. Returns `false` because day/night changes no longer require a
    /// resident scene rebuild.
    pub fn set_light_rig(&mut self, rig: LightRig) -> bool {
        self.set_lighting_state(rig, self.emissive_scale)
    }

    /// Select the production main-NEE sampler without enabling ReSTIR or
    /// changing the one-shadow-ray-per-path budget.
    pub fn set_nee_importance_sampling(&mut self, enabled: bool) {
        self.renderer.set_nee_importance_sampling(enabled);
        eprintln!(
            "[nee-light-selection] mode={}",
            if enabled {
                "importance-reservoir"
            } else {
                "uniform-diagnostic"
            }
        );
    }

    /// Combined light-rig + emissive scale update. Use this when the game clock
    /// changes both sun/moon state and lit-window strength; it refreshes the
    /// resident light layer once.
    pub fn set_lighting_state(&mut self, rig: LightRig, emissive_scale: f32) -> bool {
        let emissive_scale = emissive_scale.max(0.0);
        // LIGHTING-ONLY history cut (gap-dossier D2, 2026-07-10): the film
        // buffers progressively accumulate under a static camera, so after N
        // accumulated samples a sun/hour change contributes only ~1/N per
        // frame — the live image visually NEVER follows the clock (dev.ron
        // `time_of_day` 22.0 rendered pixel-identical daylight to 16.5 while
        // the fresh-start offline path honored it). A lighting change makes
        // the accumulated radiance semantically STALE, exactly like a scene
        // rebuild — hard-cut the film + temporal history. This is a film
        // clear only: NO scene/BLAS/TLAS rebuild, allocations and residency
        // preserved (`reset_camera_accumulation`).
        let lighting_changed = self.rig != rig || self.emissive_scale != emissive_scale;
        self.rig = rig;
        self.emissive_scale = emissive_scale;
        self.apply_rig_uniforms();
        self.renderer.set_emissive_scale(self.emissive_scale);
        if let Err(e) = self.refresh_light_layer() {
            eprintln!("[light-rig] light-layer refresh skipped: {e}");
        }
        if lighting_changed {
            self.renderer.reset_camera_accumulation();
            self.renderer.reset_sharc_cache();
            eprintln!(
                "[light-rig] lighting changed (sun_dir={:?} radiance={} night={}) -> film/temporal history reset (lighting-only, no scene rebuild)",
                self.rig.sun_dir, self.rig.sun_radiance, self.rig.is_night
            );
        }
        false
    }

    fn refresh_light_layer(&mut self) -> Result<(), String> {
        let lights = {
            let state = self
                .renderer
                .state
                .as_ref()
                .ok_or_else(|| "renderer state is not initialized".to_string())?;
            let scene = state
                .scene_state
                .as_ref()
                .ok_or_else(|| "scene state is not resident yet".to_string())?;
            build_light_layer_for_scene(scene, &self.rig, self.emissive_scale)
        };
        self.renderer
            .replace_light_layer(lights)
            .map_err(|e| format!("replace_light_layer: {e}"))
    }

    /// Bind an equirectangular HDRI environment map (linear RGB, `channels`
    /// usually 3). Drives the megakernel's miss-ray environment lookup + HDRI
    /// importance-sampled NEE (both already in the kernel) — replacing the
    /// procedural gradient sky with real image-based lighting. Empty `data`
    /// clears it (back to procedural sky). Call after the scene exists.
    pub fn set_hdri(&mut self, data: &[f32], width: u32, height: u32, channels: u32) {
        self.renderer.set_hdri(data, width, height, channels);
    }

    /// Bind resident material texture objects. Call after scene upload and before
    /// terrain/spray slots that reference the same texture-slot ordering.
    pub fn set_material_textures(&mut self, textures: &[RendererTexture2D]) -> Result<(), String> {
        self.renderer
            .set_native_textures(textures)
            .map_err(|e| format!("set_native_textures: {e:?}"))
    }

    /// Bind resident material textures and keep the host descriptor/f32 mirror
    /// available for OptiX OMM baking while that builder is being moved to cooked
    /// opacity payloads. Shader sampling still uses native texture objects.
    pub fn set_material_textures_with_mirror(
        &mut self,
        textures: &[RendererTexture2D],
        texture_descs: &[u32],
        texture_data: &[f32],
    ) -> Result<(), String> {
        self.renderer
            .set_native_textures_with_host_mirror(textures, texture_descs, texture_data)
            .map_err(|e| format!("set_native_textures_with_host_mirror: {e:?}"))
    }

    /// Bind the GPU-resident sparse virtual texture table. Slot ids are the same
    /// integer material slots used by resident material textures; SVT changes the
    /// backing store to paged tiles.
    pub fn set_svt_residency(
        &mut self,
        texture_descs: &[i32],
        page_table: &[i32],
        tile_cache_rgba: &[f32],
        texture_count: u32,
        tile_size: u32,
        cache_tiles: u32,
        max_tiles_per_axis: u32,
        texture_size: u32,
    ) -> Result<(), String> {
        self.renderer
            .set_svt_residency(
                texture_descs,
                page_table,
                tile_cache_rgba,
                texture_count,
                tile_size,
                cache_tiles,
                max_tiles_per_axis,
                texture_size,
            )
            .map_err(|e| format!("set_svt_residency: {e:?}"))
    }

    pub fn clear_svt_residency(&mut self) -> Result<(), String> {
        self.renderer
            .clear_svt_residency()
            .map_err(|e| format!("clear_svt_residency: {e:?}"))
    }

    /// Upload the per-vertex weathering PATTERN (the cook/geometry-anchored
    /// masks: `7 * vertex_count` floats, `[moss, water_stain, paint_chip, rust,
    /// soot, efflorescence, edge_wear]`) and enable the megakernel's dormant
    /// 7-channel weathering engine (`apply_weathering_full`). The masks MUST be
    /// parallel to the uploaded scene's MERGED vertex buffer
    /// (`SceneState.geometry.positions` order). `upload_scene` calls this from
    /// `scene.geometry.weathering_masks` automatically; expose it so a caller can
    /// re-drive it. An empty slice disables weathering. MUST be called AFTER scene
    /// upload (it needs the renderer `state` to exist). See
    /// [[dynamic-weathering-directive]] — the cook bakes the PATTERN, the sim
    /// scales the INTENSITY via [`set_weathering_intensity`].
    pub fn set_weathering_masks(&mut self, masks: &[f32]) -> Result<(), String> {
        self.renderer
            .set_weathering_masks(masks)
            .map_err(|e| format!("set_weathering_masks: {e:?}"))
    }

    /// Upload the SPRAY FIELD (world-space ground-cover weight grid) + its
    /// ChannelTable → texture-slot mapping. Drives the megakernel's top-2 convex
    /// triplanar ground blend (kills biome grid seams + shows painted strokes).
    /// MUST be called AFTER `set_material_textures` (the channel slots reference
    /// resident texture entries). An empty `packed` disables the field (the ground
    /// falls back to its single-slot triplanar path). See
    /// `vox_core::spray::SprayField`.
    ///
    /// - `packed`: 2 u32/cell (4 u8 channels/word), row-major — `pack_u32()`.
    /// - `res`: `[res_x, res_z]`; `origin`/`cell_size`: world placement.
    /// - `channel_slots`: 8 i32/channel — MUST match the megakernel
    ///   `SPRAY_CH_STRIDE`. Layout `[base_albedo, base_normal, base_rough,
    ///   base_disp, overlay_albedo, overlay_alpha, overlay_normal, overlay_disp]`
    ///   (-1 = none). Uploaded verbatim; the shader interprets the stride.
    /// - `channel_uv_scales`: 2 f32/channel `[base_uv_per_m, overlay_uv_per_m]`.
    ///   Empty keeps compatibility by falling back to the mesh-wide ground UV.
    pub fn set_spray_field(
        &mut self,
        packed: &[u32],
        res: [u32; 2],
        origin: [f32; 2],
        cell_size: f32,
        channel_slots: &[i32],
        channel_uv_scales: &[f32],
    ) -> Result<(), String> {
        self.renderer
            .set_spray_field(
                packed,
                res,
                origin,
                cell_size,
                channel_slots,
                channel_uv_scales,
            )
            .map_err(|e| format!("set_spray_field: {e:?}"))
    }

    /// GROUND MACRO VARIATION LAYER. Forward ONE global aerial-scale albedo (an
    /// texture slot) the ground albedo lerps toward with distance — keeps meso-scale
    /// patchiness resolving at aerial range where the ~1 m detail tile mips to a
    /// flat average ("textures are just colors"). `slot < 0` or `blend <= 0`
    /// disables (byte-identical).
    pub fn set_ground_macro(
        &mut self,
        slot: i32,
        tile_m: f32,
        luma_reference: f32,
        blend: f32,
    ) -> Result<(), String> {
        self.renderer
            .set_ground_macro(slot, tile_m, luma_reference, blend)
            .map_err(|e| format!("set_ground_macro: {e:?}"))
    }

    /// PER-PIXEL CURVATURE FIELD. Forward a world-space mean-curvature grid (1 f32/
    /// cell, row-major `iz*res_x+ix`; convex ridge > 0, concave hollow < 0) baked
    /// from the heightmap. The megakernel terrain blend samples it BILINEARLY at the
    /// hit XZ for a SMOOTH per-pixel curvature — no per-triangle facet step, hence no
    /// angular scree "squares" when `curvature_amp`/`cavity_amp` are on. Empty values
    /// disable the field (kernel falls back to the legacy per-triangle κ).
    pub fn set_curvature_field(
        &mut self,
        values: &[f32],
        res: [u32; 2],
        origin: [f32; 2],
        cell_size: f32,
    ) -> Result<(), String> {
        self.renderer
            .set_curvature_field(values, res, origin, cell_size)
            .map_err(|e| format!("set_curvature_field: {e:?}"))
    }

    /// WATER-DEPTH FIELD (P3, ROOT 4). Forward a world-space per-cell water-column
    /// depth grid (1 f32/cell, row-major `iz*res_x+ix`; `0` = dry) baked from the
    /// heightmap + sea plane. The megakernel ground path samples it BILINEARLY at the
    /// hit XZ so submerged ground reads as a depth-blended BED (wet sand → mud → silt)
    /// and the intertidal band above `sea_level` darkens/roughens — fixing the
    /// flat-Coastal-sand / grass-under-water look. `sea_level` is the world-Y of the
    /// static sea plane (drives the wet band). Empty values disable the field.
    pub fn set_depth_field(
        &mut self,
        values: &[f32],
        res: [u32; 2],
        origin: [f32; 2],
        cell_size: f32,
        sea_level: f32,
    ) -> Result<(), String> {
        self.renderer
            .set_depth_field(values, res, origin, cell_size, sea_level)
            .map_err(|e| format!("set_depth_field: {e:?}"))
    }

    /// WATER FLOW FIELD: per-cell `(vel_x, vel_z)` m/s interleaved (2 f32/cell) on
    /// the SAME grid as the depth field. Drives flow-aligned wave scroll +
    /// white-water on MAT_WATER. Empty/`enabled=false` → flow OFF.
    pub fn set_water_flow_field(&mut self, values: &[f32], enabled: bool) -> Result<(), String> {
        self.renderer
            .set_water_flow_field(values, enabled)
            .map_err(|e| format!("set_water_flow_field: {e:?}"))
    }

    /// K5 (animated water): per-frame refit of the water surface's vertices. `node`
    /// is the retained water HybridMesh's scene node; `verts` are its new displaced
    /// positions (proto-local order, exactly the water proto's vertex count). Resolves
    /// node → resident instance → prototype, then rebuilds ONLY that proto's GAS +
    /// the IAS (no scene rebuild). Returns `Ok(false)` when the CLAS IAS path is not
    /// active (AMD/Vulkan) or nothing was refit; `Err` on a bad node / count mismatch.
    /// Call BEFORE `render_camera`.
    pub fn refit_water_geometry(
        &mut self,
        node: vox_scene::NodeId,
        verts: &[[f32; 3]],
    ) -> Result<bool, String> {
        let inst = self.retained_instance_index(node).ok_or_else(|| {
            format!("refit_water_geometry: no resident instance for node {node:?}")
        })?;
        let proto = {
            let st = self
                .renderer
                .state
                .as_ref()
                .ok_or("refit_water_geometry: renderer state uninitialized")?;
            let ss = st
                .scene_state
                .as_ref()
                .ok_or("refit_water_geometry: no resident scene")?;
            *ss.geometry
                .instance_proto_index
                .get(inst)
                .ok_or_else(|| format!("refit_water_geometry: no proto for instance {inst}"))?
        };
        self.renderer
            .refit_water_geometry(proto, verts)
            .map_err(|e| format!("refit_water_geometry: {e:?}"))
    }

    /// UNDERWATER BED atlas slots (P3). Bind the submerged bed materials the
    /// megakernel blends by water depth: `wet_*` (~0 m), `mud_*` (~2 m), `silt_*`
    /// (~8 m+), and `bed_*` (riverbed, flow-driven follow-up). Each is an
    /// (albedo, normal, roughness) texture-slot triple; pass `-1` for an absent layer
    /// (it rolls back to the next-shallower bed → byte-identical when all `-1`).
    /// Resolve against the SAME resident material texture set as the meshes.
    #[allow(clippy::too_many_arguments)]
    pub fn set_underwater_layers(
        &mut self,
        wet_albedo: i32,
        wet_normal: i32,
        wet_rough: i32,
        mud_albedo: i32,
        mud_normal: i32,
        mud_rough: i32,
        silt_albedo: i32,
        silt_normal: i32,
        silt_rough: i32,
        bed_albedo: i32,
        bed_normal: i32,
        bed_rough: i32,
    ) {
        self.renderer.set_underwater_layers(
            wet_albedo,
            wet_normal,
            wet_rough,
            mud_albedo,
            mud_normal,
            mud_rough,
            silt_albedo,
            silt_normal,
            silt_rough,
            bed_albedo,
            bed_normal,
            bed_rough,
        );
    }

    /// SLOPE / HEIGHT LAYERED TERRAIN MATERIAL (#33). Bind the steep-face ROCK +
    /// transition DIRT texture slots the megakernel ground path blends over the biome
    /// ground by surface slope (geometric up-cosine) + height. Slots reference
    /// resident material textures. Pass `-1` for any
    /// absent layer (its weight rolls back into the biome ground → byte-identical).
    /// The slope/height THRESHOLDS are config-first (`RenderConfig.slope_layer`).
    pub fn set_slope_layers(
        &mut self,
        rock_albedo: i32,
        rock_normal: i32,
        dirt_albedo: i32,
        dirt_normal: i32,
        rock_disp: i32,
        dirt_disp: i32,
    ) {
        self.renderer.set_slope_layers(
            rock_albedo,
            rock_normal,
            dirt_albedo,
            dirt_normal,
            rock_disp,
            dirt_disp,
        );
    }

    /// High-altitude SNOW cap: a top-of-stack ground layer eased in above
    /// `height_snow` with a slope falloff (caps sit on ledges/peaks, not vertical
    /// faces). albedo = -1 / huge height → off (byte-identical lowland ground).
    pub fn set_slope_snow(
        &mut self,
        snow_albedo: i32,
        snow_normal: i32,
        snow_disp: i32,
        height_snow: f32,
        height_snow_band: f32,
        snow_slope_cos: f32,
    ) {
        self.renderer.set_slope_snow(
            snow_albedo,
            snow_normal,
            snow_disp,
            height_snow,
            height_snow_band,
            snow_slope_cos,
        );
    }

    /// Set the per-channel DYNAMIC weathering intensity (how worn THIS render's
    /// geometry is — sim-driven from building age + missed maintenance). Channel
    /// order matches the masks; values clamped to `[0,1]`. Multiplies the baked
    /// pattern in the megakernel (`u_weathering_intensity_0..6`). Defaults to
    /// `[1.0; 7]` (full reference pattern) until called.
    pub fn set_weathering_intensity(&mut self, intensity: [f32; 7]) {
        self.renderer.set_weathering_intensity(intensity);
    }

    fn apply_rig_uniforms(&mut self) {
        let rig = self.rig;
        let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
        let e_sun = rig.sun_radiance;
        self.renderer
            .set_weathering_intensity(rig.weathering_intensity);
        self.renderer.set_sky_gradient(
            rig.sky_dome_zenith,
            rig.sky_dome_horizon,
            rig.sky_dome_intensity,
        );
        if std::env::var("SPECTRA_FILM_DIAG").is_ok() {
            eprintln!(
                "[atm_diag] resident update: rig.atmosphere_enabled={} mie={} turbidity={} sun_dir={:?} e_sun={} fog_enabled={} fog_density={} fog_aniso={}",
                rig.atmosphere_enabled,
                rig.atmosphere_mie,
                rig.atmosphere_turbidity,
                rig.sun_dir,
                e_sun,
                rig.fog_enabled,
                rig.fog_density,
                rig.fog_anisotropy
            );
        }
        // THE SUN LIGHTS SURFACES REGARDLESS OF THE ATMOSPHERE FLAG (2026-07-02).
        // set_sun used to live INSIDE `if rig.atmosphere_enabled` — so disabling the
        // Bruneton haze in render.ron silently left u_sun_radiance at 0. The
        // atmosphere flag gates only the haze/scatter integrals.
        self.renderer.set_sun(sun.to_array(), e_sun);
        self.renderer.set_moon(
            rig.moon_dir,
            rig.moon_color,
            rig.moon_radiance,
            rig.moon_phase,
            rig.moon_bright_limb_angle,
        );
        self.renderer.set_atmosphere(
            rig.atmosphere_enabled,
            rig.atmosphere_mie,
            rig.atmosphere_turbidity,
        );
        self.renderer.set_fog(
            rig.fog_enabled,
            rig.fog_density,
            rig.fog_color,
            rig.fog_height_falloff,
            rig.fog_anisotropy,
        );
    }

    /// Shared upload body used by both `new` and `set_scene`.
    fn upload_scene(&mut self, mut scene: SceneState) -> Result<SceneSyncReport, String> {
        // Force the internal render resolution onto the scene's camera so the
        // film matches the renderer's framebuffer.
        scene.camera.width = self.width;
        scene.camera.height = self.height;

        // UPLOAD DIAG (OCHROMA_SCENE_UPLOAD_DIAG=1): the Y-extent of the geometry
        // ACTUALLY handed to the GPU. Splits the flat-terrain search space in one
        // run: CPU mesh has the relief (terrain-mesh-diag) — if THIS prints flat,
        // the game's scene build dropped it; if it prints the relief, the loss is
        // downstream in the BLAS/CLAS build.
        if std::env::var("OCHROMA_SCENE_UPLOAD_DIAG").is_ok() {
            let p = &scene.geometry.positions;
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for c in p.chunks_exact(3) {
                lo = lo.min(c[1]);
                hi = hi.max(c[1]);
            }
            eprintln!(
                "[scene-upload-diag] verts={} y_range=[{lo:.1},{hi:.1}] instances={} protos={}",
                p.len() / 3,
                scene.geometry.instance_transforms.len() / 16,
                scene.geometry.proto_ranges.len(),
            );
            // Count-vs-content: a declared count SMALLER than the arrays silently
            // truncates the traced geometry to the first N triangles.
            eprintln!(
                "[scene-upload-diag] declared vertex_count={} triangle_count={} vs arrays: positions/3={} indices/3={}",
                scene.geometry.vertex_count,
                scene.geometry.triangle_count,
                p.len() / 3,
                scene.geometry.indices.len() / 3,
            );
            // The TERRAIN proto = the largest by vertex count. Print its range,
            // its OWN y-extent, and the transform of its first instance (row-major
            // 4x4; y-scale = m[5], y-translate = m[7]) — a squashed/offset terrain
            // instance paints the real map's materials with no relief.
            if let Some((pi, r)) = scene
                .geometry
                .proto_ranges
                .iter()
                .enumerate()
                .max_by_key(|(_, r)| r.1)
            {
                let (v0, vc, _, _) = *r;
                let (mut plo, mut phi) = (f32::MAX, f32::MIN);
                for i in (v0 as usize)..((v0 + vc) as usize) {
                    let y = p[i * 3 + 1];
                    plo = plo.min(y);
                    phi = phi.max(y);
                }
                let inst = scene
                    .geometry
                    .instance_proto_index
                    .iter()
                    .position(|&x| x == pi as u32);
                let tf: Vec<f32> = inst
                    .map(|ii| scene.geometry.instance_transforms[ii * 16..ii * 16 + 16].to_vec())
                    .unwrap_or_default();
                eprintln!(
                    "[scene-upload-diag] biggest proto #{pi}: verts={vc} y=[{plo:.1},{phi:.1}] first_inst={inst:?} tf_row_y={:?}",
                    tf.get(4..8),
                );
                // The TLAS culls ray-vs-BLAS tests by the per-proto AABB: a stale
                // FLAT aabb here makes everything above its ceiling untraceable
                // while the low ground renders perfectly.
                eprintln!(
                    "[scene-upload-diag] proto_aabbs[{pi}]={:?} (of {})",
                    scene.geometry.proto_aabbs.get(pi),
                    scene.geometry.proto_aabbs.len(),
                );
            }
        }

        scene.lights = build_light_layer_for_scene(&scene, &self.rig, self.emissive_scale);
        scene.mark_lights_changed();

        // WEATHERING: the per-vertex pattern travels on the scene
        // (`geometry.weathering_masks`, built by `meshes_to_instanced_scene`).
        // The megakernel binds the SEPARATE `g_weathering_masks` StructuredBuffer
        // (set via `set_weathering_masks`), NOT the interleaved vertex buffer the
        // uploader packs, so we MUST re-drive the setter after every scene upload
        // or the live/resident path renders clean (the historical gap: no setter
        // on ResidentSceneRenderer). Taken out before `scene` is moved into
        // `load_scene_state` (the megakernel weathering path reads ONLY the
        // separate buffer, so the interleaved copy is redundant here). Empty
        // masks → the setter disables weathering.
        let weathering_masks = std::mem::take(&mut scene.geometry.weathering_masks);

        // TDR DIAGNOSIS: the one-time scene/BVH/TLAS upload is a prime suspect for
        // a single >2s GPU op. Wall-time it when SPECTRA_DISPATCH_TIMING=1.
        let _scene_t = std::time::Instant::now();
        self.renderer
            .load_scene_state(scene)
            .map_err(|e| format!("load_scene_state: {e:?}"))?;
        // Bind the weathering pattern to the megakernel's g_weathering_masks +
        // flip u_weathering_enabled. After load_scene_state so `state` exists.
        self.renderer
            .set_weathering_masks(&weathering_masks)
            .map_err(|e| format!("set_weathering_masks: {e:?}"))?;
        self.renderer.reset_sharc_cache();
        self.renderer.reset_camera_accumulation();
        self.last_view_proj = None;
        self.last_projection = None;
        self.last_camera_view = None;
        self.previous_view_proj = None;
        self.last_camera_reset = true;
        if std::env::var("SPECTRA_DISPATCH_TIMING").as_deref() == Ok("1") {
            eprintln!(
                "[dispatch_timing] load_scene_state (BVH/TLAS upload): {:.1} ms",
                _scene_t.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Sky-dome + atmosphere from the rig — set after the scene so the state
        // exists. These setters are idempotent and cheap (no GPU rebuild).
        self.apply_rig_uniforms();

        let (rebuilt, reused) = self.renderer.last_scene_sync();
        Ok(SceneSyncReport {
            layers_rebuilt: rebuilt,
            layers_reused: reused,
        })
    }

    /// Record one instance's transform delta (movers, articulated rigid
    /// sub-parts). Appended to the id-sorted, coalesced [`SceneDeltaRing`] and
    /// applied on the next [`render_camera`] via the GPU `apply_scene_delta`
    /// indexed scatter into the persistent resident buffer (copying current →
    /// prev_transform before overwrite, for motion vectors / DLSS-RR).
    ///
    /// Determinism: the ring is a `BTreeMap<slot, …>`, so repeated writes to the
    /// same slot coalesce latest-wins in place (no push+sort, no HashMap/RNG
    /// order), and the drained stream is ascending-slot — the indexed scatter is
    /// order-independent. This REPLACES the old CPU `pending_refits` Vec.
    ///
    /// `transform` is a row-major 3×4 (upper rows of a 4×4 world transform).
    pub fn update_instance_transform(&mut self, instance_index: usize, transform: [[f32; 4]; 3]) {
        self.delta_ring
            .set_transform(instance_index as u32, transform);
    }

    /// Record one instance's material-table base delta. This is the generic
    /// retained primitive used by animated material variants such as emissive
    /// indicators; it does not rebuild geometry, prototypes, or the scene.
    pub fn update_instance_material_base(&mut self, instance_index: usize, material_base: u32) {
        self.delta_ring
            .set_material_base(instance_index as u32, material_base);
    }

    /// Stamp the NodeId→instance_index mapping after a full scene upload.
    /// Call once after every `set_scene` that changes the instance order.
    pub fn reset_retained_mirror<I>(&mut self, nodes: I) -> Result<(), RetainedDeltaError>
    where
        I: IntoIterator<Item = vox_scene::NodeId>,
    {
        self.retained_mirror.replace_instances_in_node_order(nodes)
    }

    /// Drain and apply transform-only `deltas` as TLAS refits. Returns
    /// `Err` (without clearing `deltas`) for any structural delta — caller
    /// must then trigger a full-scene rebuild via `set_scene`.
    /// Clears `deltas` on success.
    #[cfg(feature = "spectra-native")]
    pub fn drain_scene_deltas(
        &mut self,
        deltas: &mut Vec<vox_scene::SceneDelta>,
    ) -> Result<RetainedDeltaPlan, String> {
        let plan = self
            .retained_mirror
            .plan_deltas(deltas.as_slice())
            .map_err(|e| format!("retained scene delta plan: {e}"))?;
        if plan.requires_scene_rebuild() {
            return Err(format!(
                "retained scene structural rebuild required \
                 (structural_deltas={})",
                plan.stats.structural_deltas
            ));
        }
        plan.queue_resident_refits(self)
            .map_err(|e| format!("retained scene delta apply: {e}"))?;
        deltas.clear();
        Ok(plan)
    }

    /// Thin forwarding accessors — let the game inspect mirror state without
    /// holding a RetainedRenderMirror field.
    pub fn retained_instance_index(&self, node: vox_scene::NodeId) -> Option<usize> {
        self.retained_mirror.instance_index(node)
    }

    pub fn retained_node_for_instance(&self, instance_index: usize) -> Option<vox_scene::NodeId> {
        self.retained_mirror.node_for_instance(instance_index)
    }

    pub fn retained_node_count(&self) -> usize {
        self.retained_mirror.len()
    }

    /// Flush the queued delta ring into the resident `g_instances` buffer
    /// WITHOUT tracing or presenting a frame. This is the EXACT same ring-drain +
    /// `apply_scene_delta` (g_instances patch) + IAS-refit / legacy-fallback that
    /// [`render_camera`] runs BEFORE the megakernel — minus the trace/present.
    ///
    /// Use it to land queued [`update_instance_transform`] / `drain_scene_deltas`
    /// deltas into the buffer that [`download_resident_instances`] reads, on a
    /// minimal scene where a full `render_camera` trace would CUDA-fault. The
    /// determinism witness (replay the same delta log twice → byte-identical
    /// resident buffer) only needs the `g_instances` patch, which
    /// `apply_scene_delta` performs; the IAS refit reads that patched buffer back
    /// to rebuild the traversable and does not change `g_instances`. No-op when the
    /// ring is empty.
    pub fn flush_pending_deltas(&mut self) -> Result<(), String> {
        if !self.delta_ring.is_empty() {
            let cmds = self.delta_ring.drain_commands();
            let ias_refit = self
                .renderer
                .apply_scene_delta_and_refit_ias(&cmds)
                .map_err(|e| format!("apply_scene_delta_and_refit_ias: {e:?}"))?;
            if !ias_refit {
                // Legacy fallback (no CLAS IAS active): drive the traversed TLAS via
                // the per-backend MODE_UPDATE refit from the same drained commands.
                let dirty: Vec<(usize, [[f32; 4]; 3])> = cmds
                    .iter()
                    .filter(|c| c.op == OP_SET_TRANSFORM)
                    .map(|c| (c.slot as usize, c.transform_3x4()))
                    .collect();
                self.renderer
                    .refit_instance_transforms(&dirty)
                    .map_err(|e| format!("refit_instance_transforms: {e:?}"))?;
            }
        }
        Ok(())
    }

    /// Per-frame camera stream. Pure state mutation (`set_camera_view_matrix` +
    /// `set_view_proj`) then `render()`. Does NOT reconstruct the backend or the
    /// renderer, and does NOT re-upload the scene.
    ///
    /// `view`/`proj` are column-major `Mat4::to_cols_array()` arrays.
    pub fn render_camera(
        &mut self,
        view: [f32; 16],
        proj: [f32; 16],
    ) -> Result<FrameOutput, String> {
        // Aurora Steps 3-5 (K4): per-instance transform deltas now flow through the
        // resident `g_instances` buffer and into the TRAVERSED OptiX CLAS IAS via an
        // IAS REFIT — the zero-CPU-per-frame mover path. We drain the id-sorted,
        // coalesced ring into ONE `apply_scene_delta` indexed scatter (patching
        // `g_instances`, the determinism artifact) and then refit the IAS over the
        // SAME retained per-prototype GASes (NO BLAS/proto/material rebuild, NO
        // `build_instanced_scene`). The trace reads the moved instances from
        // `g_instances`.
        //
        // On a backend with no CLAS IAS (the AMD/Vulkan dev box, or OptiX not yet
        // built) the refit reports `false` after still patching `g_instances`; we
        // then fall back to the legacy `MODE_UPDATE` refit so movers don't freeze on
        // that path. On the NVIDIA/CUDA ship target the CLAS IAS refit runs and the
        // legacy refit is RETIRED (never reached).
        if !self.delta_ring.is_empty() {
            let cmds = self.delta_ring.drain_commands();
            let ias_refit = self
                .renderer
                .apply_scene_delta_and_refit_ias(&cmds)
                .map_err(|e| format!("apply_scene_delta_and_refit_ias: {e:?}"))?;
            if !ias_refit {
                // Legacy fallback (no CLAS IAS active): drive the traversed TLAS via
                // the per-backend MODE_UPDATE refit from the same drained commands.
                let dirty: Vec<(usize, [[f32; 4]; 3])> = cmds
                    .iter()
                    .filter(|c| c.op == OP_SET_TRANSFORM)
                    .map(|c| (c.slot as usize, c.transform_3x4()))
                    .collect();
                self.renderer
                    .refit_instance_transforms(&dirty)
                    .map_err(|e| format!("refit_instance_transforms: {e:?}"))?;
            } else if std::env::var("OCHROMA_AURORA_TRACE").as_deref() == Ok("1") {
                eprintln!(
                    "[aurora-k4] {} delta(s) → g_instances → OptiX IAS refit (no CPU scene rebuild)",
                    cmds.len()
                );
            }
        }

        // The megakernel's `u_view_proj` (and the R1 temporal-reprojection kernel)
        // multiply a WORLD-space position by this matrix to get clip space, so it
        // MUST be the COMBINED view-projection (proj * view), not the projection
        // alone. `set_camera_view_matrix` sets `current_view_proj = view` and
        // `set_view_proj` overwrites it — historically with `proj` ALONE, which
        // made `u_view_proj` projection-only. That was dormant (its only consumer,
        // g_velocity, feeds ReSTIR-GI which ships OFF), but R1 temporal
        // accumulation reprojects world hit positions through it, so projection-
        // only rejected ~100% of pixels (reprojection landed out of bounds).
        // Combine here (glam column-major: clip = proj * view * world).
        let view_m = glam::Mat4::from_cols_array(&view);
        let proj_m = glam::Mat4::from_cols_array(&proj);
        let view_proj = (proj_m * view_m).to_cols_array();
        let projection_changed = self
            .last_projection
            .is_some_and(|prev| matrix_changed(prev, proj));
        let camera_moved = self
            .last_camera_view
            .is_some_and(|prev| camera_motion_changed(prev, view));
        let camera_cut_now = self
            .last_camera_view
            .is_some_and(|prev| camera_cut(prev, view));
        let temporal_reset =
            self.last_view_proj.is_none() || projection_changed || camera_cut_now;
        if projection_changed || camera_moved {
            if projection_changed || camera_cut_now {
                self.renderer.reset_camera_accumulation();
            } else {
                self.renderer.reset_camera_color_accumulation();
            }
        }
        self.previous_view_proj = self.last_view_proj;
        self.last_view_proj = Some(view_proj);
        self.last_projection = Some(proj);
        self.last_camera_view = Some(view);
        self.last_camera_reset = temporal_reset;
        // Keep the camera frame (eye/forward) from the view matrix; then set the
        // COMBINED view-projection as u_view_proj.
        self.renderer.set_camera_view_matrix(view);
        self.renderer.set_view_proj(view_proj);
        self.renderer.set_projection_matrix(proj);
        let frame = self.renderer.render().map_err(|e| format!("render: {e:?}"))?;
        self.present_frame_index = self.present_frame_index.wrapping_add(1);
        // Mirror any MegaGeometry CPU-ownership activity Spectra recorded during
        // this frame into the engine-side witness (lock-free; no GPU sync).
        self.bridge_geometry_cpu_ownership();
        Ok(frame)
    }

    /// Camera state for vendor-neutral temporal presentation. The matrices are
    /// captured at the same resident render boundary as the guide textures.
    pub fn present_camera_state(
        &self,
    ) -> Option<([f32; 16], [f32; 16], [f32; 16], [f32; 16])> {
        Some((
            self.last_camera_view?,
            self.last_projection?,
            self.last_view_proj?,
            self.previous_view_proj.unwrap_or(self.last_view_proj?),
        ))
    }

    pub fn frame_index(&self) -> u32 {
        self.present_frame_index
    }

    pub fn temporal_reset_pending(&self) -> bool {
        self.last_camera_reset
    }

    /// Apply a batch of per-instance transform refits IMMEDIATELY through the
    /// GPU `MODE_UPDATE` refit (`refit_instanced_tlas` on the retained hardware
    /// TLAS — BLASes untouched, no scene rebuild), returning whether a drift
    /// rebuild fired. This is the same call [`render_camera`] flushes
    /// internally; exposing it lets a caller TIME the refit phase separately
    /// from the render phase (the perf-spike mover loop). `dirty` is
    /// `(instance_index, row-major 3×4 transform)`, id-ordered by the caller
    /// for determinism. Prefer [`update_instance_transform`] +
    /// [`render_camera`] for the normal game loop.
    pub fn refit_instances(&mut self, dirty: &[(usize, [[f32; 4]; 3])]) -> Result<bool, String> {
        self.renderer
            .refit_instance_transforms(dirty)
            .map_err(|e| format!("refit_instance_transforms: {e:?}"))
    }

    /// Aurora Step 1 witness hook: flush the id-sorted, coalesced delta ring
    /// through the GPU `apply_scene_delta` indexed scatter IMMEDIATELY (ONE
    /// buffer upload + ONE dispatch), WITHOUT rendering — so a bench can TIME the
    /// resident transform path in isolation (vs the old Vec sort+drain). Drains
    /// the ring. No-op when the ring is empty.
    pub fn flush_scene_delta(&mut self) -> Result<(), String> {
        if self.delta_ring.is_empty() {
            return Ok(());
        }
        let cmds = self.delta_ring.drain_commands();
        self.renderer
            .apply_scene_delta(&cmds)
            .map_err(|e| format!("apply_scene_delta: {e:?}"))
    }

    /// Aurora Step 1: apply an explicit id-sorted command batch directly,
    /// bypassing the ring (the bench builds the batch once and times the GPU
    /// apply). The caller guarantees ascending-slot + one command per slot.
    pub fn apply_scene_delta(&mut self, cmds: &[GpuSceneCmd]) -> Result<(), String> {
        self.renderer
            .apply_scene_delta(cmds)
            .map_err(|e| format!("apply_scene_delta: {e:?}"))
    }

    /// Aurora Step 1 witness: download the persistent resident `g_instances`
    /// buffer as raw u32 words (`count * 32`). The determinism artifact — hash
    /// this and compare across two runs of the same delta sequence. Also lets
    /// the witness inspect `prev_transform` (words 12..23 per instance).
    pub fn download_resident_instances(&self) -> Result<Vec<u32>, String> {
        self.renderer
            .download_resident_instances()
            .map_err(|e| format!("download_resident_instances: {e:?}"))
    }

    /// Number of resident instances the persistent buffer is sized for.
    pub fn resident_instance_count(&self) -> usize {
        self.renderer.resident_instance_count()
    }

    /// Number of distinct slots currently pending in the delta ring (the CPU
    /// queue depth — the witness asserts this is O(deltas), not a growing Vec).
    pub fn pending_delta_count(&self) -> usize {
        self.delta_ring.len()
    }

    /// Pure camera-stream render (`set_camera_view_matrix` + `set_view_proj` +
    /// `render()`) that does NOT flush `pending_refits` — the render phase in
    /// isolation, for benches that drive refits via [`refit_instances`] and
    /// want the render timing uncontaminated by refit. The normal game loop
    /// uses [`render_camera`], which flushes pending refits first.
    pub fn render_only(&mut self, view: [f32; 16], proj: [f32; 16]) -> Result<FrameOutput, String> {
        self.renderer.set_camera_view_matrix(view);
        self.renderer.set_view_proj(proj);
        self.renderer.set_projection_matrix(proj);
        let frame = self.renderer.render().map_err(|e| format!("render: {e:?}"))?;
        self.bridge_geometry_cpu_ownership();
        Ok(frame)
    }

    /// Invariant-checked facts for the active native acceleration scene.
    pub fn geometry_accel_stats(
        &self,
    ) -> Result<
        spectra_renderer::renderer::geometry_backend::GeometryAccelStats,
        spectra_renderer::renderer::geometry_backend::GeometryBackendError,
    > {
        self.renderer.geometry_accel_stats()
    }

    /// Number of TLAS instances registered after the last scene upload — the
    /// registered instance count (not an acceleration-kind substitute).
    pub fn tlas_instance_count(&self) -> usize {
        self.renderer.tlas_instance_count()
    }

    /// Snapshot of the engine-wide MegaGeometry CPU ownership contract. This is
    /// diagnostic state and never forces a GPU synchronization or stats readback.
    /// Reflects Spectra's live `geometry_cpu_contract` probe via
    /// [`Self::bridge_geometry_cpu_ownership`], which the render paths run each
    /// frame — so this witness is only vacuously zero when no MegaGeometry CPU op
    /// actually occurred, never merely because nothing was wired to record one.
    pub fn geometry_cpu_ownership(&self) -> GeometryCpuOwnershipSnapshot {
        self.geometry_cpu_ownership.snapshot()
    }

    /// Bridge Spectra's probed `geometry_cpu_contract` counters (owned by the
    /// inner `Renderer`, incremented by the GPU-owned MegaGeometry LOD/page/IAS
    /// pipelines) into this engine-side witness. Reads the inner cumulative
    /// snapshot, folds the positive per-activity / per-service delta since the
    /// last bridge into `geometry_cpu_ownership`, and advances the baseline. Never
    /// synchronizes the GPU or reads back device stats — the inner snapshot is a
    /// lock-free atomic load. Idempotent across frames (delta == 0 is a no-op).
    ///
    /// Only the two MegaGeometry host services are mirrored; Spectra's
    /// `FixedRenderPassSubmission` is the ordinary render-pass submit, not a
    /// geometry host service, and is intentionally excluded from the witness.
    fn bridge_geometry_cpu_ownership(&mut self) {
        use spectra_renderer::renderer::geometry_cpu_contract::{
            ForbiddenGeometryCpuActivity as SpectraForbidden, GeometryHostService as SpectraService,
        };
        let snap = self.renderer.geometry_cpu_ownership.snapshot();
        for activity in SpectraForbidden::ALL {
            let idx = activity as usize;
            let cur = snap.forbidden_count(activity);
            if cur > self.geometry_cpu_bridge.forbidden[idx] {
                let delta = cur - self.geometry_cpu_bridge.forbidden[idx];
                // Discriminants match 1:1, so `ALL[idx]` is the mirror variant.
                self.geometry_cpu_ownership
                    .add_forbidden(ForbiddenGeometryCpuActivity::ALL[idx], delta);
                self.geometry_cpu_bridge.forbidden[idx] = cur;
            }
        }
        let native_calls = snap.service_calls(SpectraService::NativeAccelerationSubmission);
        if native_calls > self.geometry_cpu_bridge.native_calls {
            let native_ns = snap.service_ns(SpectraService::NativeAccelerationSubmission);
            self.geometry_cpu_ownership.add_host_service(
                GeometryHostServiceKind::NativeAccelerationSubmission,
                native_calls - self.geometry_cpu_bridge.native_calls,
                native_ns.saturating_sub(self.geometry_cpu_bridge.native_ns),
            );
            self.geometry_cpu_bridge.native_calls = native_calls;
            self.geometry_cpu_bridge.native_ns = native_ns;
        }
        let page_calls = snap.service_calls(SpectraService::OpaquePageTransport);
        if page_calls > self.geometry_cpu_bridge.page_calls {
            let page_ns = snap.service_ns(SpectraService::OpaquePageTransport);
            self.geometry_cpu_ownership.add_host_service(
                GeometryHostServiceKind::OpaquePageTransport,
                page_calls - self.geometry_cpu_bridge.page_calls,
                page_ns.saturating_sub(self.geometry_cpu_bridge.page_ns),
            );
            self.geometry_cpu_bridge.page_calls = page_calls;
            self.geometry_cpu_bridge.page_ns = page_ns;
        }
    }

    /// A fully-reused [`SceneSyncReport`] (`rebuilt == 0`, `reused == 1`): the report
    /// for a frame where the caller determined the scene was unchanged and chose
    /// NOT to re-upload it. Lets the live seam prove camera-only moves rebuild no
    /// BLASes without forcing a redundant `set_scene`.
    pub fn reused_delta(&self) -> SceneSyncReport {
        SceneSyncReport {
            layers_rebuilt: 0,
            layers_reused: 1,
        }
    }

    /// The light rig driving this renderer.
    pub fn rig(&self) -> &LightRig {
        &self.rig
    }

    /// Internal render resolution.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Route the renderer's final frame to a CUDA interop device pointer (the
    /// zero-copy DLSS present path) or back to host beauty. When `Interop`, the
    /// `PACK_RGBA` dispatch DtoD-copies the packed RGBA f32 straight into
    /// `color_ptr` and NO host beauty download happens.
    pub fn set_render_target(&mut self, target: spectra_renderer::RenderTarget) {
        self.renderer.set_render_target(target);
    }

    /// Enable present-side temporal guidance (jitter + motion vectors) on the
    /// Interop path when a temporal present backend is actually active.
    pub fn set_present_temporal_upscale(&mut self, on: bool) {
        self.renderer.set_present_temporal_upscale(on);
    }

    /// Tell Spectra whether the active present backend is actually doing external
    /// ray reconstruction denoise+upscale. This is separate from temporal
    /// upscaler guide production so SR fallback does not disable internal denoise.
    pub fn set_present_ray_reconstruction(&mut self, on: bool) {
        self.renderer.set_present_ray_reconstruction(on);
    }

    /// Tell Spectra that present owns temporal radiance reconstruction (SNR or
    /// RR), so its internal temporal accumulator must not run as a second owner.
    pub fn set_present_reconstruction_owns_temporal(&mut self, on: bool) {
        self.renderer
            .set_present_reconstruction_owns_temporal(on);
    }

    /// WATER (MAT_WATER) procedural-wave + shoreline-foam controls, forwarded to
    /// the spectra `Renderer` (`u_water_*` in the megakernel). `[amp, scale, speed,
    /// time, foam_strength, foam_rough]`. The game advances `time` by wall-clock
    /// every present frame so the sea ripples (it has had ZERO callers, so
    /// `u_water_time` was pinned at 0 = frozen water). Only the MAT_WATER surface
    /// reads these; every other material is byte-identical. A no-op when no scene
    /// is loaded.
    pub fn set_water_params(&mut self, params: [f32; 6]) {
        self.renderer.set_water_params(params);
    }

    /// WATER (MAT_WATER) non-animated optical look, forwarded to the spectra
    /// `Renderer`. Channel order:
    /// `[clarity, roughness_floor, inscatter_r, inscatter_g, inscatter_b,
    /// scatter_sigma, foam_depth_m]`.
    ///
    /// This is resident state, not process env, so render.ron/dev hot reload can
    /// update water colour/depth response without reconstructing the renderer.
    pub fn set_water_look_params(&mut self, params: [f32; 7]) {
        self.renderer.set_water_look_params(params);
    }

    /// World-planar terrain-ground look, forwarded live to spectra state.config.
    /// Channel order:
    /// `[smooth_normal, antitile_strength, antitile_cell_m, normal_strength,
    /// normal_graze, ao_strength, macro_scale, macro_strength, heightblend_k,
    /// curv_albedo, blend_sharpen]`.
    pub fn set_ground_look_params(&mut self, params: [f32; 11]) {
        self.renderer.set_ground_look_params(params);
    }

    /// Underwater-bed blend look, forwarded live to spectra state.config.
    /// Channel order:
    /// `[shallow_m, deep_m, wetband_m, wetband_darken, wetband_rough_drop,
    /// silt_rough_boost]`.
    pub fn set_underwater_params(&mut self, params: [f32; 6]) {
        self.renderer.set_underwater_params(params);
    }

    /// Live authored-emission multiplier. This is a uniform on direct emissive
    /// hits and also regenerates the resident light layer so emissive NEE lights
    /// match the same scale without a material or geometry rebuild.
    pub fn set_emissive_scale(&mut self, scale: f32) {
        self.emissive_scale = scale.max(0.0);
        self.renderer.set_emissive_scale(self.emissive_scale);
        if let Err(e) = self.refresh_light_layer() {
            eprintln!("[light-rig] emissive light-layer refresh skipped: {e}");
        }
    }

    /// LIVE COLOR-GRADE — forward the game's Look-panel grade to the spectra
    /// `Renderer`, mutating the resident `state.config` in place so the change
    /// reaches the post-tonemap grade uniforms (`u_exposure` / `u_grade_*` /
    /// `u_cdl_*`) next frame WITHOUT a `reconfigure` (buffer/graph rebuild) —
    /// exactly like `set_water_params`. Scalars (not a spectra struct) so the
    /// engine/game boundary stays decoupled from spectra's `GradeParams` type.
    /// `temperature_k` is white-balance Kelvin; the three CDL arrays are
    /// per-channel slope(gain)/offset(lift)/power(gamma). A no-op when no scene
    /// is loaded.
    #[allow(clippy::too_many_arguments)]
    pub fn set_grade_params(
        &mut self,
        exposure_ev: f32,
        temperature_k: f32,
        tint: f32,
        saturation: f32,
        contrast: f32,
        cdl_slope: [f32; 3],
        cdl_offset: [f32; 3],
        cdl_power: [f32; 3],
    ) {
        self.renderer.set_grade_params(
            exposure_ev,
            temperature_k,
            tint,
            saturation,
            contrast,
            cdl_slope,
            cdl_offset,
            cdl_power,
        );
    }

    /// The currently-configured render target.
    pub fn render_target(&self) -> spectra_renderer::RenderTarget {
        self.renderer.render_target()
    }

    /// The device pointer the last realtime frame routed its packed RGBA into
    /// (`Some` after an `Interop` frame; `None` on the host-beauty path).
    pub fn last_pack_output_ptr(&self) -> Option<u64> {
        self.renderer.last_pack_output_ptr()
    }

    #[cfg(not(any(
        all(target_os = "windows", feature = "spectra-native-optix"),
        all(target_os = "macos", feature = "spectra-native-metal")
    )))]
    pub fn shared_vulkan_backend(&self) -> SharedVulkanBackend {
        self.renderer.gpu.clone()
    }

    #[cfg(not(any(
        all(target_os = "windows", feature = "spectra-native-optix"),
        all(target_os = "macos", feature = "spectra-native-metal")
    )))]
    pub fn vulkan_backend_identity(&self) -> usize {
        self.renderer.gpu.identity()
    }

    #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
    pub fn shared_metal_backend(&self) -> SharedMetalBackend {
        self.renderer.gpu.clone()
    }

    #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
    pub fn metal_backend_identity(&self) -> usize {
        self.renderer.gpu.identity()
    }

    #[cfg(not(all(target_os = "windows", feature = "spectra-native-optix")))]
    pub fn last_device_ldr_buffer(&self) -> Option<GpuBufferHandle> {
        self.renderer.last_device_ldr_buffer()
    }

    pub fn last_vulkan_reconstruction(
        &self,
    ) -> Option<spectra_renderer::renderer::VulkanReconstructionBuffers> {
        self.renderer.last_vulkan_reconstruction()
    }

    /// Enable/disable production of the full SNR guide set on the Vulkan device
    /// present path (see `Renderer::set_snr_guides_enabled`).
    pub fn set_snr_guides_enabled(&mut self, on: bool) {
        self.renderer.set_snr_guides_enabled(on);
    }

    /// Request that the next SNR guide frame discards reprojected history
    /// (camera cut / resize / teleport). Consumed once by the validity-mask pass.
    pub fn request_snr_history_reset(&mut self) {
        self.renderer.request_snr_history_reset();
    }

    pub fn last_denoise_execution(&self) -> spectra_renderer::DenoiseExecution {
        self.renderer.last_denoise_execution()
    }

    /// RR (DLSS Ray Reconstruction) guide device ptrs for the last frame, in the
    /// order `(diffuse_albedo, specular_albedo, normals, roughness, depth,
    /// motion)`. See `Renderer::rr_guide_ptrs`.
    pub fn rr_guide_ptrs(&mut self) -> [u64; 7] {
        self.renderer.rr_guide_ptrs()
    }

    /// CUDA event recorded behind the latest interop payload writes. CUDA present
    /// waits it on its own stream before consuming guide/color pointers.
    pub fn rr_payload_ready_event(&mut self) -> u64 {
        self.renderer.rr_payload_ready_event()
    }

    /// The sub-pixel camera jitter (PIXELS, Halton(2,3)−0.5 ∈ [-0.5, 0.5]) the
    /// last frame applied to the primary ray when an upscaler is active. The
    /// RR/DLSS present feeds this IDENTICAL offset to `Jitter.Offset.X/Y` to
    /// un-jitter the temporal reprojection. See `Renderer::rr_jitter`.
    pub fn rr_jitter(&self) -> (f32, f32) {
        self.renderer.rr_jitter()
    }

    /// Internal resolution of the RR/DLSS guide buffers produced by the last
    /// frame. Present uses this to scale pixel-space motion vectors.
    pub fn rr_render_size(&self) -> (u32, u32) {
        self.renderer.rr_render_size()
    }
}

/// Extract the camera forward axis (world -Z of the view) from a scene's
/// column-major view matrix. Used to orient the camera-relative fill lights.
/// TIER-AWARE glass bounce floor. The flat floor of 8 (matching the still
/// path's `has_glass` bump) was the #1 real-time perf knob: a glassy city
/// forced max_bounces->8 regardless of tier, and 8->2 is ~-58% sample_loop.
/// At 854x480 Performance the flat floor read glass8 15.3fps; capping it
/// glass3 24.3fps / glass2 32.0fps clears the >=30fps SHIP GATE; Balanced
/// 1706x960 glass8 4.4fps -> glass3 6.6fps (+60%).
///
/// The quality caveat: geometric glass needs ~4-5 bounces (enter front face ->
/// through interior -> exit back face -> reach a lit surface -> back), so a
/// pane at 1-2 bounces can read BLACK in close shots. The floor is therefore
/// keyed off the tier's OWN base bounce count (Performance <=2, Balanced 3-5,
/// Beauty >=6) so each tier gets its configured target:
///   - Performance (base <=2): floor 1  — emergency 780M path stays close to 30 FPS
///   - Balanced    (base 3..=5): floor 4
///   - Beauty      (base >=6): floor 8  — unchanged, full geometric glass
/// `SPECTRA_GLASS_BOUNCES` still overrides this (config-first); 0 disables.
/// Deterministic: a pure function of the tier's config value, never reordered.
fn glass_floor_for_tier(base_bounces: u32) -> u32 {
    // CONFIG-FIRST: the per-tier glass bounce floors are `config/ochroma.ron`
    // (`resident_renderer.glass_floor_{performance,balanced,beauty}`). Defaults
    // 2 / 4 / 8 match the measured Urban Horizon performance path.
    let rcfg = &vox_config::config().resident_renderer;
    match base_bounces {
        0..=2 => rcfg.glass_floor_performance,
        3..=5 => rcfg.glass_floor_balanced,
        _ => rcfg.glass_floor_beauty,
    }
}

/// TIER-AWARE water bounce floor — the SEA (MAT_WATER) surface's own, LOWER floor.
/// CONFIG-FIRST: `config/ochroma.ron`
/// (`resident_renderer.water_floor_{performance,balanced,beauty}`, defaults 2/4/4).
/// Water reads correctly with a single surface refraction + one lit hop, far less
/// depth than architectural glass, so flooring it separately is the #1 trace-fps
/// lever on water-heavy / near-empty maps without touching glass quality.
fn water_floor_for_tier(base_bounces: u32) -> u32 {
    let rcfg = &vox_config::config().resident_renderer;
    match base_bounces {
        0..=2 => rcfg.water_floor_performance,
        3..=5 => rcfg.water_floor_balanced,
        _ => rcfg.water_floor_beauty,
    }
}

/// The Beauty-tier glass floor — full geometric glass depth. Matches the still
/// path's glass bump (`splat_backend::pathtrace_mesh_lit_weathered_to_rgba`,
/// which sets 8 when `has_glass`): a glass pane needs enough depth to refract
/// through both faces and reach a lit surface, or it terminates dark and reads
/// opaque/matte.
#[cfg(test)]
const GLASS_MIN_BOUNCES: u32 = 8;

/// The material-type tag value for transmissive glass. MUST match the
/// `MAT_GLASS` id both material packers write into element 0 of every material
/// (`splat_backend::pack_cuda_mesh_material` / `pack_vulkan_mesh_material`) and
/// the megakernel's `case MAT_GLASS` dispatch.
const MAT_GLASS_TYPE: u32 = 3;

/// The material-type tag for the sea/water surface (`MAT_WATER`). Water shades
/// through the SAME transmissive Fresnel/Beer-Lambert glass BSDF (plus wave
/// normals + foam), so it wants the same multi-bounce floor as glass — a ray must
/// refract through the surface and reach a lit point, or the sea reads opaque/dark.
const MAT_WATER_TYPE: u32 = 21;

/// True when the scene's material table holds at least one material whose type
/// tag equals `want_ty` (e.g. [`MAT_GLASS_TYPE`] or [`MAT_WATER_TYPE`]) — used to
/// floor the glass and water bounce budgets INDEPENDENTLY.
///
/// Element 0 of every packed material is the material-type tag (a `u32` stored
/// in an `f32` via `from_bits`/`pack_u32`) in the canonical 132-float layout, so the per-material stride is derived from
/// `params.len() / material_count` and element 0 of each material is decoded with
/// `to_bits()`. Pure scan — no HashMap/RNG iteration, so it is deterministic.
fn scene_has_mat_type(scene: &SceneState, want_ty: u32) -> bool {
    let mats = &scene.materials;
    if mats.material_count == 0 || mats.params.is_empty() {
        return false;
    }
    let stride = mats.params.len() / mats.material_count;
    if stride == 0 {
        return false;
    }
    (0..mats.material_count).any(|i| {
        mats.params
            .get(i * stride)
            .map(|t| t.to_bits() == want_ty)
            .unwrap_or(false)
    })
}

const MAT_EMISSION_SLOT: usize = 18;
const MAT_EMISSION_COLOR_SLOT: usize = 15;
const MAT_FLAGS_SLOT: usize = 49;
const MAT_FLAG_SUPPRESS_NEE_EMITTER: u32 = 1 << 3;
const MAT_FLAG_NEE_DOWNLIGHT: u32 = 1 << 4;

// CONFIG-FIRST: the night-NEE emitter cap (`instancing.max_emissive_point_lights`,
// default 4096) and the per-emitter radiant-power scale (`resident_renderer.
// emissive_light_scale`, default 200, env OCHROMA_EMISSIVE_LIGHT_SCALE) live in
// `config/ochroma.ron` and are read at their call sites. NEE selects ONE light per
// pixel uniformly so cost is O(1)/sample; the cap keeps the buffer upload + ReSTIR
// candidate quality bounded, and the id-sorted slice keeps selection deterministic.

/// Derive NEE point lights from the scene's emissive instances.
///
/// MegaLights night path: any instance whose material has `emission_strength > 0`
/// (lit windows = MAT_GLASS_LIT-equivalent, street lights, signage) becomes a
/// world-space POINT light at the instance's world centroid. Without this, emissive
/// SURFACES only contribute when a camera/bounce ray DIRECTLY hits them — they do
/// not LIGHT the surrounding city. Turning them into NEE lights is what makes a
/// night frame carry: every pixel can next-event-sample the nearby windows/lamps.
///
/// Deterministic: instances scanned in id (index) order, capped slice, no
/// HashMap/RNG iteration. Returns packed Vulkan `LightData` floats appended to the
/// directional rig.
fn build_light_layer_for_scene(
    scene: &SceneState,
    rig: &LightRig,
    emissive_surface_scale: f32,
) -> LightLayer {
    // Inject the rig's four-light directional rig (sun/sky/camera-fill/rim),
    // mirroring spectra_resident_bench's light setup so lighting is identical
    // to the proven bench path.
    let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
    let cam_fwd = scene_camera_forward(scene);
    let camera_fill = (-cam_fwd + glam::Vec3::Y * 0.35).normalize_or_zero();
    let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
    let mut light_data: Vec<f32> = Vec::with_capacity(4 * LIGHT_FLOATS);

    // P1 — ONE PHYSICAL SUN. E_sun drives both the visible disk and NEE disk.
    let l_sun = rig.sun_radiance / sun_solid_angle();
    let day_light_count = if vox_config::config().resident_renderer.nee_day_lights {
        light_data.extend_from_slice(&pack_sun_disk_light(sun.to_array(), rig.sun_color, l_sun));
        for (dir, color, intensity) in [
            (
                glam::Vec3::Y.to_array(),
                rig.analytic_sky_fill_color,
                rig.sky_intensity,
            ),
            (
                camera_fill.to_array(),
                rig.analytic_camera_fill_color,
                rig.camera_fill,
            ),
            (
                rim_fill.to_array(),
                rig.analytic_rim_fill_color,
                rig.rim_fill,
            ),
        ] {
            light_data.extend_from_slice(&pack_directional_light(dir, color, intensity));
        }
        4
    } else {
        0
    };

    // Config-first: night lit windows default ON (config.lit_windows_enabled);
    // env OCHROMA_LIT_WINDOWS=0|off force-disables for an A/B witness.
    let lit_window_disabled = !vox_config::config().resident_renderer.lit_windows_enabled
        || matches!(
            std::env::var("OCHROMA_LIT_WINDOWS").as_deref(),
            Ok("0") | Ok("off")
        );
    let (emissive_lights, emissive_count) = if rig.is_night && !lit_window_disabled {
        let point_scale = std::env::var("OCHROMA_EMISSIVE_LIGHT_SCALE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(vox_config::config().resident_renderer.emissive_light_scale);
        emissive_point_lights(scene, point_scale * emissive_surface_scale.max(0.0))
    } else {
        (Vec::new(), 0)
    };
    light_data.extend_from_slice(&emissive_lights);
    let light_count = day_light_count + emissive_count;
    if emissive_count > 0 {
        eprintln!(
            "[night-lights] derived {emissive_count} emissive point lights \
             -> {light_count} total lights"
        );
    }
    LightLayer {
        light_data,
        light_count,
    }
}

fn emissive_point_lights(scene: &SceneState, scale: f32) -> (Vec<f32>, usize) {
    let geo = &scene.geometry;
    let mats = &scene.materials;
    let inst_count = geo.instance_count;
    if inst_count == 0 || mats.material_count == 0 || mats.params.is_empty() {
        return (Vec::new(), 0);
    }
    let mstride = mats.params.len() / mats.material_count;
    if mstride <= MAT_EMISSION_SLOT {
        return (Vec::new(), 0);
    }
    // Need per-instance transform (16 floats, row-major) to place the light.
    if geo.instance_transforms.len() < inst_count * 16 {
        return (Vec::new(), 0);
    }

    // Precompute per-material emission (strength + color). An instance is a light
    // when ANY triangle of its prototype resolves to an emissive material.
    let mat_emission: Vec<(f32, [f32; 3], bool, bool)> = (0..mats.material_count)
        .map(|mi| {
            let b = mi * mstride;
            let em = mats
                .params
                .get(b + MAT_EMISSION_SLOT)
                .copied()
                .unwrap_or(0.0);
            let col = [
                mats.params
                    .get(b + MAT_EMISSION_COLOR_SLOT)
                    .copied()
                    .unwrap_or(1.0),
                mats.params
                    .get(b + MAT_EMISSION_COLOR_SLOT + 1)
                    .copied()
                    .unwrap_or(1.0),
                mats.params
                    .get(b + MAT_EMISSION_COLOR_SLOT + 2)
                    .copied()
                    .unwrap_or(1.0),
            ];
            let nee_emitter = mats
                .params
                .get(b + MAT_FLAGS_SLOT)
                .map(|flags| flags.to_bits() & MAT_FLAG_SUPPRESS_NEE_EMITTER == 0)
                // Legacy packed materials predate the flag and remain eligible.
                .unwrap_or(true);
            let nee_downlight = mats
                .params
                .get(b + MAT_FLAGS_SLOT)
                .is_some_and(|flags| flags.to_bits() & MAT_FLAG_NEE_DOWNLIGHT != 0);
            (em, col, nee_emitter, nee_downlight)
        })
        .collect();
    if !mat_emission
        .iter()
        .any(|(em, _, nee_emitter, _)| *em > 0.0 && *nee_emitter)
    {
        return (Vec::new(), 0);
    }

    // Per-instance material BASE is ADDED to each triangle's relative material id
    // (`final = base + tri.material_id`) — it is NOT the absolute material index
    // (the earlier bug). Empty ⇒ base 0. Proto index selects the geometry range.
    let bases = &geo.instance_material_base;
    let protos = &geo.instance_proto_index;
    let tri_mat = &geo.material_ids; // per-triangle relative material id
    let ranges = &geo.proto_ranges; // (v_off, v_cnt, t_off, t_cnt) per proto

    let max_emissive_point_lights =
        vox_config::config().instancing.max_emissive_point_lights as usize;
    let mut out: Vec<f32> = Vec::new();
    let mut count = 0usize;
    let mut downlight_probe_count = 0usize;
    let mut downlight_wet_min_lux = f32::INFINITY;
    let mut downlight_dry_max_lux = 0.0f32;
    let mut downlight_max_glare_cd = 0.0f32;
    for i in 0..inst_count {
        if count >= max_emissive_point_lights {
            break;
        }
        let base = bases.get(i).copied().unwrap_or(0) as usize;
        let proto = protos.get(i).copied().unwrap_or(i as u32) as usize;

        // Integrate the actual emissive surface. `emission_strength` is
        // radiance, not radiant power: a tiny status LED must not illuminate a
        // street as strongly as a large luminaire just because both materials
        // carry the same scalar. The old strongest-material shortcut also put
        // the point at the whole prototype AABB centre. That made traffic-light
        // optics into huge coloured street lights and could place a streetlamp
        // emitter halfway down its support pole.
        //
        // Accumulate emitted RGB flux (radiance * world-space triangle area)
        // and its flux-weighted centroid. The point-light approximation then
        // preserves the source's integrated power and location while keeping
        // the existing O(1) ReSTIR light sample per pixel.
        let mut flux_rgb = [0.0f32; 3];
        let mut flux_weight = 0.0f32;
        let mut emitter_power = 0.0f32;
        let mut emitter_center = glam::Vec3::ZERO;
        let mut emitter_direction = glam::Vec3::ZERO;
        let mut strongest_emission = (0.0f32, [0.0f32; 3]);
        let mut downlight = false;
        if let Some(&(_, _, t_off, t_cnt)) = ranges.get(proto) {
            let t0 = t_off as usize;
            let t1 = t0 + t_cnt as usize;
            for t in t0..t1.min(tri_mat.len()) {
                let rel = tri_mat[t] as usize;
                let abs = base + rel;
                if let Some(&(em, col, nee_emitter, nee_downlight)) = mat_emission.get(abs) {
                    if !(em > 0.0) || !nee_emitter {
                        continue;
                    }
                    if em > strongest_emission.0 {
                        strongest_emission = (em, col);
                    }
                    downlight |= nee_downlight;
                    let ib = t * 3;
                    let Some((&i0, rest)) =
                        geo.indices.get(ib).zip(geo.indices.get(ib + 1..ib + 3))
                    else {
                        continue;
                    };
                    let [i1, i2] = rest else { continue };
                    let read_world = |index: u32| -> Option<glam::Vec3> {
                        let pb = index as usize * 3;
                        let p = glam::Vec3::new(
                            *geo.positions.get(pb)?,
                            *geo.positions.get(pb + 1)?,
                            *geo.positions.get(pb + 2)?,
                        );
                        let m = &geo.instance_transforms[i * 16..i * 16 + 16];
                        Some(glam::Vec3::new(
                            m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                            m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11],
                        ))
                    };
                    let (Some(p0), Some(p1), Some(p2)) =
                        (read_world(i0), read_world(*i1), read_world(*i2))
                    else {
                        continue;
                    };
                    let area = 0.5 * (p1 - p0).cross(p2 - p0).length();
                    if !(area > 0.0) || !area.is_finite() {
                        continue;
                    }
                    let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                    // A shielded luminaire emits from the authored lower panel,
                    // not its sealed housing top and sides. Keeping only
                    // downward-facing triangles preserves the fixture's inward
                    // tilt and prevents cancelling normals.
                    if nee_downlight && face_normal.y >= -0.10 {
                        continue;
                    }
                    let tri_flux = em * area;
                    emitter_power += tri_flux;
                    for channel in 0..3 {
                        flux_rgb[channel] += col[channel].max(0.0) * tri_flux;
                    }
                    let weight = tri_flux
                        * (0.2126 * col[0].max(0.0)
                            + 0.7152 * col[1].max(0.0)
                            + 0.0722 * col[2].max(0.0));
                    emitter_center += ((p0 + p1 + p2) / 3.0) * weight;
                    emitter_direction += face_normal * weight;
                    flux_weight += weight;
                }
            }
            // Legacy/developer SceneState fixtures may provide material ranges
            // without host geometry. Keep their old eligibility semantics; all
            // product scenes carry the exact positions/indices and therefore
            // take the area-integrated path above.
            if emitter_power == 0.0 && (geo.positions.is_empty() || geo.indices.is_empty()) {
                let (em, col) = strongest_emission;
                if em > 0.0 {
                    flux_rgb = col.map(|value| value.max(0.0) * em);
                    flux_weight = em;
                    emitter_power = em;
                }
            }
        } else {
            // No proto ranges (single-soup legacy): treat base as the absolute id.
            if let Some(&(em, col, nee_emitter, nee_downlight)) = mat_emission.get(base) {
                if em > 0.0 && nee_emitter {
                    flux_rgb = col.map(|value| value.max(0.0) * em);
                    flux_weight = em;
                    emitter_power = em;
                    downlight = nee_downlight;
                }
            }
        }
        let peak_flux = flux_rgb.into_iter().fold(0.0f32, f32::max);
        if !(emitter_power > 0.0) {
            continue;
        }
        let color = if peak_flux > 0.0 {
            flux_rgb.map(|value| value / peak_flux)
        } else {
            [0.0; 3]
        };

        // World centroid: instance transform applied to the proto's object-space
        // AABB center when available, else the transform translation.
        let m = &geo.instance_transforms[i * 16..i * 16 + 16];
        let local_center = geo
            .proto_aabbs
            .get(proto)
            .map(|(lo, hi)| {
                [
                    0.5 * (lo[0] + hi[0]),
                    0.5 * (lo[1] + hi[1]),
                    0.5 * (lo[2] + hi[2]),
                ]
            })
            .unwrap_or([0.0, 0.0, 0.0]);
        // Row-major 4x4: world = M * [local,1]. Row r is m[r*4..r*4+4].
        let fallback_center = glam::Vec3::new(
            m[0] * local_center[0] + m[1] * local_center[1] + m[2] * local_center[2] + m[3],
            m[4] * local_center[0] + m[5] * local_center[1] + m[6] * local_center[2] + m[7],
            m[8] * local_center[0] + m[9] * local_center[1] + m[10] * local_center[2] + m[11],
        );
        let center = if flux_weight > 0.0 {
            emitter_center / flux_weight
        } else {
            fallback_center
        };
        let downlight_power_scale = std::env::var("OCHROMA_DOWNLIGHT_POWER_SCALE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(vox_config::config().resident_renderer.downlight_power_scale)
            .clamp(0.01, 1000.0);
        let mut intensity = peak_flux.max(emitter_power * f32::EPSILON)
            * scale
            * if downlight {
                downlight_power_scale
            } else {
                1.0
            };
        if downlight {
            let min_intensity = vox_config::config()
                .resident_renderer
                .downlight_min_intensity
                .max(0.0);
            let max_intensity = vox_config::config()
                .resident_renderer
                .downlight_max_intensity
                .max(min_intensity);
            intensity = intensity.clamp(min_intensity, max_intensity);
            // The authored road-light ledger uses a conservative far-edge probe
            // 7.5 m below and 12 m inward, plus the directly-below dry maximum.
            let far_wet = estimate_downlight_probe(intensity, 7.5, 12.0, true);
            let near_dry = estimate_downlight_probe(intensity, 7.5, 0.0, false);
            downlight_probe_count += 1;
            downlight_wet_min_lux = downlight_wet_min_lux.min(far_wet.lux);
            downlight_dry_max_lux = downlight_dry_max_lux.max(near_dry.lux);
            downlight_max_glare_cd = downlight_max_glare_cd.max(far_wet.glare_candela);
        }
        let spot_direction = emitter_direction.normalize_or_zero();
        out.extend_from_slice(&if downlight {
            pack_spot_light(
                center.to_array(),
                color,
                intensity,
                spot_direction.to_array(),
            )
        } else {
            pack_point_light(center.to_array(), color, intensity)
        });
        count += 1;
    }
    if downlight_probe_count > 0 {
        eprintln!(
            "[night-downlight-probes] count={downlight_probe_count} wet_far_min_lux={downlight_wet_min_lux:.2} dry_axis_max_lux={downlight_dry_max_lux:.2} glare_max_cd={downlight_max_glare_cd:.1} contract_lux=8.00..50.00 glare_cap_cd=2000"
        );
    }
    (out, count)
}

#[derive(Clone, Copy, Debug)]
struct DownlightProbeEstimate {
    lux: f32,
    glare_candela: f32,
}

/// Deterministic engineering probe for the same broad 60/75 degree downlight
/// packed by `pack_downlight`. Intensity is treated as luminous intensity;
/// inverse-square falloff gives incident lux and the wet factor is the authored
/// conservative road-surface acceptance condition.
fn estimate_downlight_probe(
    intensity: f32,
    vertical_m: f32,
    horizontal_m: f32,
    wet: bool,
) -> DownlightProbeEstimate {
    let vertical_m = vertical_m.max(0.01);
    let horizontal_m = horizontal_m.max(0.0);
    let distance_sq = vertical_m * vertical_m + horizontal_m * horizontal_m;
    let angle_deg = horizontal_m.atan2(vertical_m).to_degrees();
    let angular = if angle_deg <= 60.0 {
        1.0
    } else if angle_deg >= 75.0 {
        0.0
    } else {
        let t = (angle_deg - 60.0) / 15.0;
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    let condition = if wet { 0.72 } else { 1.0 };
    DownlightProbeEstimate {
        lux: intensity.max(0.0) * angular * condition / distance_sq,
        glare_candela: intensity.max(0.0) * angular / std::f32::consts::PI,
    }
}

fn scene_camera_forward(scene: &SceneState) -> glam::Vec3 {
    let m = glam::Mat4::from_cols_array(&scene.camera.view_matrix);
    // Forward in world space is the inverse-rotation of view -Z. For an
    // orthonormal view basis, the world forward is the negated third row of the
    // rotation part (column-major: row 2 = elements [2], [6], [10]).
    let fwd = glam::Vec3::new(-m.x_axis.z, -m.y_axis.z, -m.z_axis.z);
    fwd.normalize_or_zero()
}

fn matrix_changed(a: [f32; 16], b: [f32; 16]) -> bool {
    a.iter()
        .zip(b.iter())
        .any(|(x, y)| (*x - *y).abs() > 1.0e-5)
}

fn camera_motion_changed(prev_view: [f32; 16], next_view: [f32; 16]) -> bool {
    let moved_m = camera_eye(prev_view).distance(camera_eye(next_view));
    let prev_fwd = camera_forward(prev_view);
    let next_fwd = camera_forward(next_view);
    let rotated_rad = prev_fwd.dot(next_fwd).clamp(-1.0, 1.0).acos();
    moved_m > 0.02 || rotated_rad > 1.0e-4
}

fn camera_cut(prev_view: [f32; 16], next_view: [f32; 16]) -> bool {
    let prev_eye = camera_eye(prev_view);
    let next_eye = camera_eye(next_view);
    let moved_m = prev_eye.distance(next_eye);
    let prev_fwd = camera_forward(prev_view);
    let next_fwd = camera_forward(next_view);
    let dot = prev_fwd.dot(next_fwd).clamp(-1.0, 1.0);
    let rotated_rad = dot.acos();

    // Treat ordinary orbit/pan as continuous motion. Full reset is reserved for
    // save-load framing jumps, dev-camera teleports, and scripted cut cameras.
    moved_m > 500.0 || rotated_rad > 0.50
}

fn camera_eye(view: [f32; 16]) -> glam::Vec3 {
    glam::Mat4::from_cols_array(&view)
        .inverse()
        .transform_point3(glam::Vec3::ZERO)
}

fn camera_forward(view: [f32; 16]) -> glam::Vec3 {
    let m = glam::Mat4::from_cols_array(&view);
    glam::Vec3::new(-m.x_axis.z, -m.y_axis.z, -m.z_axis.z).normalize_or_zero()
}

#[cfg(test)]
mod glass_floor_tests {
    use super::{
        GLASS_MIN_BOUNCES, emissive_point_lights, estimate_downlight_probe, glass_floor_for_tier,
        water_floor_for_tier,
    };
    use crate::splat_backend::{PbrMaterial, pack_mesh_material, pack_spot_light};
    use spectra_scene_state::{MaterialLayer, SceneState};

    fn reservoir_hash(mut value: u32) -> u32 {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        value ^ (value >> 16)
    }

    fn select_weighted(weights: &[f32], seed: u32) -> (usize, f32) {
        let total: f32 = weights
            .iter()
            .copied()
            .filter(|w| w.is_finite() && *w > 0.0)
            .sum();
        if !(total > 1.0e-20 && total < 1.0e30) {
            let index = seed as usize % weights.len();
            return (index, 1.0 / weights.len() as f32);
        }
        let mut running = 0.0;
        let mut selected = 0usize;
        for (index, weight) in weights.iter().copied().enumerate() {
            let weight = if weight.is_finite() && weight > 0.0 {
                weight
            } else {
                0.0
            };
            let next = running + weight;
            let bits = reservoir_hash(seed ^ (index as u32).wrapping_mul(0x9e37_79b9));
            let random = (bits >> 8) as f32 / 16_777_216.0;
            if weight > 0.0 && random * next < weight {
                selected = index;
            }
            running = next;
        }
        (selected, weights[selected] / total)
    }

    #[test]
    fn nee_importance_reservoir_is_deterministic_and_uses_exact_pdf() {
        let weights = [1.0, 3.0, 6.0];
        for seed in [0, 1, 7, 19, 2_048, u32::MAX] {
            let a = select_weighted(&weights, seed);
            let b = select_weighted(&weights, seed);
            assert_eq!(a, b);
            assert_eq!(a.1, weights[a.0] / 10.0);
        }
    }

    #[test]
    fn nee_importance_reservoir_matches_weighted_distribution() {
        let weights = [1.0, 3.0, 6.0];
        let mut counts = [0u32; 3];
        for seed in 0..100_000u32 {
            counts[select_weighted(&weights, seed).0] += 1;
        }
        for (count, expected) in counts.into_iter().zip([0.1, 0.3, 0.6]) {
            let observed = count as f32 / 100_000.0;
            assert!(
                (observed - expected).abs() < 0.01,
                "{observed} vs {expected}"
            );
        }
    }

    #[test]
    fn nee_importance_estimator_is_unbiased_for_discrete_lights() {
        let weights = [1.0, 3.0, 6.0];
        let contributions = [2.0, 5.0, 11.0];
        let expected = contributions.iter().sum::<f32>();
        let weighted_expectation = weights
            .iter()
            .zip(contributions)
            .map(|(weight, contribution)| {
                let pdf = *weight / weights.iter().sum::<f32>();
                pdf * contribution / pdf
            })
            .sum::<f32>();
        assert!((weighted_expectation - expected).abs() < 1.0e-6);
    }

    #[test]
    fn nee_zero_or_nonfinite_weights_fall_back_to_uniform() {
        for weights in [[0.0, 0.0, 0.0], [f32::NAN, -1.0, 0.0]] {
            assert_eq!(select_weighted(&weights, 1), (1, 1.0 / 3.0));
        }
    }

    fn one_instance_scene(material: PbrMaterial) -> SceneState {
        let mut scene = SceneState::new(1, 1);
        scene.geometry.instance_count = 1;
        scene.geometry.instance_transforms = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        scene.geometry.instance_material_base = vec![0];
        scene.geometry.instance_proto_index = vec![0];
        scene.geometry.proto_ranges = vec![(0, 3, 0, 1)];
        scene.geometry.proto_aabbs = vec![([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5])];
        scene.geometry.material_ids = vec![0];
        scene.materials = MaterialLayer {
            params: pack_mesh_material(material).to_vec(),
            spectral_spd: Default::default(),
            material_count: 1,
        };
        scene
    }

    #[test]
    fn night_nee_requires_authored_emission_and_never_invents_lit_glass() {
        let plain_glass = PbrMaterial {
            transmission: 1.0,
            ..PbrMaterial::default()
        };
        let (_, plain_count) = emissive_point_lights(&one_instance_scene(plain_glass), 1.0);
        assert_eq!(
            plain_count, 0,
            "plain Glass must not become a hashed/material-index night emitter"
        );

        let authored_lit_glass = PbrMaterial {
            transmission: 1.0,
            emission_strength: 5.0,
            ..PbrMaterial::default()
        };
        let (_, authored_count) =
            emissive_point_lights(&one_instance_scene(authored_lit_glass), 1.0);
        assert_eq!(
            authored_count, 1,
            "an exact authored emissive material must remain eligible for night NEE"
        );

        let visible_indicator = PbrMaterial {
            emission_strength: 8.0,
            nee_emitter: false,
            ..PbrMaterial::default()
        };
        let (_, indicator_count) =
            emissive_point_lights(&one_instance_scene(visible_indicator), 1.0);
        assert_eq!(
            indicator_count, 0,
            "a visible indicator must not be promoted to environmental NEE"
        );

        let authored_downlight = PbrMaterial {
            emission_strength: 5.0,
            nee_downlight: true,
            ..PbrMaterial::default()
        };
        let (downlight, downlight_count) =
            emissive_point_lights(&one_instance_scene(authored_downlight), 1.0);
        assert_eq!(downlight_count, 1);
        assert_eq!(downlight[0].to_bits(), 5, "luminaire must pack LIGHT_SPOT");
        assert_eq!(&downlight[18..21], &[0.0, -1.0, 0.0]);

        let authored_black_emitter = PbrMaterial {
            base_color: [0.0; 3],
            emission_strength: 5.0,
            ..PbrMaterial::default()
        };
        let (black_light, black_count) =
            emissive_point_lights(&one_instance_scene(authored_black_emitter), 1.0);
        assert_eq!(black_count, 1);
        assert_eq!(
            &black_light[4..7],
            &[0.0, 0.0, 0.0],
            "runtime must preserve authored black emission rather than inventing a warm tint"
        );
    }

    #[test]
    fn authored_road_downlight_meets_wet_dry_lux_and_glare_contract() {
        const PANEL_EMISSION: f32 = 0.18;
        const PANEL_AREA_M2: f32 = 0.60 * 0.24;
        const EMISSIVE_SCALE: f32 = 200.0;
        const DOWNLIGHT_SCALE: f32 = 500.0;
        let intensity = PANEL_EMISSION * PANEL_AREA_M2 * EMISSIVE_SCALE * DOWNLIGHT_SCALE;
        let far_wet = estimate_downlight_probe(intensity, 7.5, 12.0, true);
        let far_dry = estimate_downlight_probe(intensity, 7.5, 12.0, false);
        let near_dry = estimate_downlight_probe(intensity, 7.5, 0.0, false);
        eprintln!(
            "[road-downlight-contract] wet_far_lux={:.2} dry_far_lux={:.2} dry_axis_lux={:.2} glare_cd={:.1}",
            far_wet.lux, far_dry.lux, near_dry.lux, far_wet.glare_candela,
        );
        assert!(
            far_wet.lux >= 8.0,
            "wet far-edge probe under 8 lux: {far_wet:?}"
        );
        assert!(
            far_dry.lux <= 50.0,
            "dry far-edge probe over 50 lux: {far_dry:?}"
        );
        assert!(
            near_dry.lux <= 50.0,
            "dry axis probe over 50 lux: {near_dry:?}"
        );
        assert!(
            far_wet.glare_candela <= 2_000.0,
            "glare cap exceeded: {far_wet:?}"
        );
    }

    #[test]
    fn authored_spot_axis_is_not_replaced_by_canonical_down() {
        let packed = pack_spot_light(
            [0.0, 7.5, 0.0],
            [1.0, 0.9, 0.75],
            2400.0,
            [0.45, -0.89, 0.0],
        );
        let axis = glam::Vec3::from_slice(&packed[18..21]);
        let expected = glam::Vec3::new(0.45, -0.89, 0.0).normalize();
        assert!(
            axis.dot(expected) > 0.9999,
            "packed={axis:?} expected={expected:?}"
        );
        assert!(axis.x > 0.4, "inward aim was discarded: {axis:?}");
    }

    /// Locks the per-tier glass bounce caps to the exact values witnessed on the
    /// box. Performance stays at its own base depth for the AMD 780M FSR
    /// wide-city path; Balanced/Beauty keep deeper geometric glass.
    #[test]
    fn per_tier_caps_are_the_witnessed_values() {
        // Performance tier (base bounces 1-2).
        assert_eq!(glass_floor_for_tier(2), 1, "Performance glass floor");
        assert_eq!(glass_floor_for_tier(1), 1, "sub-Performance floors to 1");
        assert_eq!(glass_floor_for_tier(0), 1, "zero-bounce floors to 1");
        // Balanced tier (base bounces 3).
        assert_eq!(glass_floor_for_tier(3), 4, "Balanced glass floor");
        assert_eq!(
            glass_floor_for_tier(5),
            4,
            "upper Balanced band floors to 4"
        );
        // Beauty tier (base bounces 6) — unchanged full geometric glass.
        assert_eq!(
            glass_floor_for_tier(6),
            GLASS_MIN_BOUNCES,
            "Beauty glass floor"
        );
        assert_eq!(
            glass_floor_for_tier(8),
            GLASS_MIN_BOUNCES,
            "high tiers keep 8"
        );
        assert_eq!(GLASS_MIN_BOUNCES, 8, "Beauty constant is the still-path 8");
    }

    /// Locks the water-only Performance path to the witnessed 1280x720 -> 1440p
    /// FSR result: water must not raise Performance above its base depth.
    #[test]
    fn water_performance_floor_keeps_legal_rr_at_one_bounce() {
        assert_eq!(water_floor_for_tier(2), 1, "Performance water floor");
        assert_eq!(water_floor_for_tier(1), 1, "sub-Performance water floor");
        assert_eq!(water_floor_for_tier(0), 1, "zero-bounce water floor");
        assert_eq!(water_floor_for_tier(3), 4, "Balanced water floor");
        assert_eq!(water_floor_for_tier(6), 4, "Beauty water floor");
    }

    /// The configured floor is a per-tier target. Performance is intentionally
    /// allowed to stay at two bounces for the 25+ FPS wide-city RR path; deeper
    /// tiers keep the higher glass targets.
    #[test]
    fn floor_matches_the_tier_target() {
        for base in [0u32, 1, 2, 3, 4, 5, 6, 7, 8] {
            let floor = glass_floor_for_tier(base);
            assert!(
                floor == 2 || floor == 4 || floor == GLASS_MIN_BOUNCES,
                "base {base} -> unexpected floor {floor}"
            );
        }
    }
}
