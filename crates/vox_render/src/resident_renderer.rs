//! Resident Spectra path-tracer for the live window (Render Keystone T1).
//!
//! [`ResidentCityRenderer`] constructs the Vulkan backend + `Renderer` +
//! `KernelSet` **ONCE** (paying the device init + ~9.3s slangc compile + initial
//! BLAS/TLAS build), then streams camera + dirty scene deltas per frame. This is
//! the opposite of the legacy still path (`pathtrace_mesh_lit_*`), which rebuilt
//! the entire renderer on every call — structurally unable to hit the 30fps
//! real-time bar on the 780M.
//!
//! It generalizes [`crate::splat_backend::spectra_resident_bench`]'s construct-once
//! setup into a reusable object: `new` does the one-time build, `set_scene` does a
//! dirty-layer-aware upload (reporting rebuilt/reused via [`SceneDelta`]), and
//! `render_camera` is a pure camera-stream frame (`set_camera_view_matrix` +
//! `set_view_proj` + `render()` — NO backend/renderer reconstruction).
//!
//! Owned single-threaded by the game's `App`/`GameView` alongside `present`
//! (decision §8 of the design): no `Arc<Mutex<>>`, deterministic frames.

#![cfg(feature = "spectra-native")]

#[cfg(not(target_os = "windows"))]
use spectra_gpu::VulkanSlangBackend;
#[cfg(target_os = "windows")]
use spectra_gpu::CudarcSlangBackend;
use spectra_renderer::{FrameOutput, RenderConfig, RenderSettings, Renderer};

/// The LIVE path-tracer compute backend. The standing rule: use CUDA on NVIDIA
/// when available. On Windows (NVIDIA/CUDA box) this is the cudarc CUDA backend;
/// elsewhere (e.g. the Linux/AMD dev box) it falls back to Vulkan. The window
/// present is a separate, thin swapchain (display only), independent of this.
#[cfg(target_os = "windows")]
type ResidentBackend = CudarcSlangBackend;
#[cfg(not(target_os = "windows"))]
type ResidentBackend = VulkanSlangBackend;
use spectra_scene_state::{GpuSceneCmd, LightLayer, SceneDeltaRing, SceneState};

/// Re-export the R31 fidelity tier so the game layer can select a tier through
/// `vox_render` without depending on `spectra-renderer` directly.
pub use spectra_renderer::FidelityTier;
pub use spectra_renderer::RenderTarget;

use crate::splat_backend::{
    LightRig, VULKAN_LIGHT_FLOATS, pack_vulkan_directional_light, resolve_slang_kernel_dir,
    rig_to_settings, seed_features_from_config,
};

/// Result of a scene-delta upload — the reuse-vs-rebuild proof.
///
/// `layers_rebuilt == 0 && layers_reused > 0` means the GPU scene was reused
/// (no per-frame BLAS rebuild) — the witness that the resident path is alive.
#[derive(Debug, Clone, Copy)]
pub struct SceneDelta {
    layers_rebuilt: u32,
    layers_reused: u32,
}

impl SceneDelta {
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
pub struct ResidentCityRenderer {
    /// The renderer, constructed once: KernelSet, SceneUploader, GpuStage (BLAS
    /// cache + per-layer fingerprints), RenderState, AtmosphereManager. Reused
    /// across every frame.
    renderer: Renderer<ResidentBackend>,
    /// Internal render resolution.
    width: u32,
    height: u32,
    /// Light rig — drives the four-light scene rig + sky/atmosphere setters.
    rig: LightRig,
    /// Aurora Step 1: the id-sorted, coalesced (latest-wins-per-slot) command
    /// ring for resident transform deltas. Backed by a `BTreeMap<slot, …>` so
    /// the drained stream is ALWAYS ascending-slot with one command per slot —
    /// never HashMap/RNG/allocator order. Flushed on `render_camera` through the
    /// GPU `apply_scene_delta` indexed scatter (ONE buffer upload + ONE
    /// dispatch), REPLACING the old per-frame CPU `pending_refits` Vec push+sort
    /// +drain. The indexed scatter is order-independent, so the resident buffer
    /// is a pure function of this id-sorted command log (the determinism moat).
    delta_ring: SceneDeltaRing,
}

impl ResidentCityRenderer {
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
        initial: SceneState,
    ) -> Result<Self, String> {
        let settings = RenderSettings::for_tier(tier);
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
        let mut config = RenderConfig::near_realtime(width, height);
        config.slang_kernel_dir = resolve_slang_kernel_dir();
        // Lean shade kernel: byte-identical for pure triangle-mesh city scenes (the
        // heavy SSS/volume/polarization/SDF/Gaussian paths are off on the resident
        // config) and it skips the NRC-coupled heavy path. With NRC off on the
        // real-time tiers, make lean unconditional for the resident render.
        config.prefer_lean_shade = true;
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

        // NVIDIA-stack foundation: enable OptiX HW-RT for the LIVE game by default
        // (config-first, not env-gated). ResidentCityRenderer historically never
        // set this, so the shipped game ran the SOFTWARE BVH and the whole RT-core
        // stack (CLAS/Mega-Geometry, HW TLAS, ReSTIR-PT quality, DLSS-RR guides)
        // was dead code. The construct gate in spectra renderer/mod.rs
        // (`want_optix && raw_context()!=0 && should_use_optix_rt()`) safely
        // rejects this on AMD/Vulkan/no-CUDA → software fallback, so it's correct
        // on the dev 780M too. SPECTRA_USE_OPTIX_RT remains an override.
        config.use_optix_rt = true;

        // PHASE R0 measurement override: OCHROMA_RESTIR=on|off forces the ReSTIR
        // DI + PT path on/off at runtime WITHOUT permanently changing the tier,
        // so the `--compare-map` A/B harness can render the same scene both ways.
        // `apply_settings` maps features.restir -> use_restir but leaves
        // resample_mode untouched (a wiring gap the audit flagged); the PT path
        // only runs when resample_mode != None, so we set BOTH here. Temporal-only
        // is the R0 mode (spatial is the R1 follow-up). Unset => leave as resolved.
        match std::env::var("OCHROMA_RESTIR").as_deref() {
            Ok("on") | Ok("1") => {
                config.use_restir = true;
                config.resample_mode = spectra_renderer::ResampleMode::Temporal;
                eprintln!("[restir-override] OCHROMA_RESTIR=on -> use_restir=true resample_mode=Temporal");
            }
            Ok("off") | Ok("0") => {
                config.use_restir = false;
                config.resample_mode = spectra_renderer::ResampleMode::None;
                eprintln!("[restir-override] OCHROMA_RESTIR=off -> use_restir=false resample_mode=None");
            }
            _ => {}
        }
        // PHASE R0 measurement override: SPECTRA_SHOT_SPP forces the per-frame
        // target spp (the `--compare-map` equal-time loop uses this to match the
        // ON/OFF passes to equal wall-clock). Unset => the tier's spp stands.
        if let Some(spp) = std::env::var("SPECTRA_SHOT_SPP").ok().and_then(|v| v.parse::<u32>().ok()) {
            if spp > 0 {
                config.target_spp = spp;
                eprintln!("[spp-override] SPECTRA_SHOT_SPP={spp} -> target_spp={spp}");
            }
        }
        // SPECTRAL MODE (16-band Hero4 by default). The old dodge here forced
        // SpectralMode::Single to avoid "black buildings" under Hero4. That
        // premise is now STALE: the original black-buildings cause was the
        // CORE-SUN RGB-lighting fix (spectra feabf3a), which routes building
        // light unconditionally into the RGB film in BOTH spectral modes. The
        // remaining issue was that the spectral XYZ film was WRITE-ONLY on every
        // shipping path (never resolved), so Hero4 was plumbed-but-inert. That is
        // now fixed in spectra: the present + host-beauty resolves drive
        // u_spectral_blend from spectral_mode and fold the spectral CHROMA over
        // the RGB MAGNITUDE (luminance-preserving — see film.slang), so Hero4 can
        // never collapse a lit surface to black. Default to Hero4 (richer
        // metameric chroma); keep Single reachable for A/B via OCHROMA_SPECTRAL.
        //   OCHROMA_SPECTRAL=single|0|off  -> scalar single-wavelength path
        //   OCHROMA_SPECTRAL=multi|hero4|1 -> 16-band Hero4 (default)
        config.spectral_mode = match std::env::var("OCHROMA_SPECTRAL").as_deref() {
            Ok("single") | Ok("0") | Ok("off") => spectra_renderer::SpectralMode::Single,
            // multi / hero4 / 1 / unset all select the spectral path
            _ => spectra_renderer::SpectralMode::Hero4,
        };

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
            .unwrap_or(65_536);

        // CUDA-GRAPH ↔ D3D12-INTEROP FIX: the resident renderer feeds the DLSS
        // present through a D3D12-shared external-memory texture (RenderTarget::
        // Interop). CUDA-graph capture/replay of the per-sample pipeline records
        // the interop memcpy_dtod into that SHARED resource into the graph; on
        // replay the captured cross-context/external-memory dependency is not
        // re-resolved against the D3D12 queue, which hangs the GPU past the 2s
        // WDDM watchdog (DXGI_ERROR_DEVICE_HUNG, 0x887A0007) on the first city
        // present — the exact city-only failure (the menu never runs the renderer
        // or graphs). Disable graphs by default on the resident/interop path; the
        // per-sample dispatch cost dwarfs the graph launch-overhead saving at
        // these resolutions. SPECTRA_CUDA_GRAPHS=1 re-enables for benchmarking.
        config.use_cuda_graphs = std::env::var("SPECTRA_CUDA_GRAPHS").as_deref() == Ok("1");

        let renderer = Renderer::new(gpu, config);

        let mut me = Self {
            renderer,
            width,
            height,
            rig,
            delta_ring: SceneDeltaRing::new(),
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
    pub fn set_scene(&mut self, scene: SceneState) -> Result<SceneDelta, String> {
        self.upload_scene(scene)
    }

    /// Bind the path tracer's flat texture atlas (texture keystone). The
    /// megakernel samples it (EWA) for any material whose `albedo_tex >= 0`.
    /// MUST be called AFTER scene upload (`new_with_tier` / `set_scene`) because
    /// the renderer's `set_texture_atlas` needs `state` to exist. `texture_descs`
    /// + `texture_data` come from `splat_backend::build_texture_atlas`; an empty
    /// atlas disables sampling (every material falls back to its flat colour).
    /// Bind an equirectangular HDRI environment map (linear RGB, `channels`
    /// usually 3). Drives the megakernel's miss-ray environment lookup + HDRI
    /// importance-sampled NEE (both already in the kernel) — replacing the
    /// procedural gradient sky with real image-based lighting. Empty `data`
    /// clears it (back to procedural sky). Call after the scene exists.
    pub fn set_hdri(&mut self, data: &[f32], width: u32, height: u32, channels: u32) {
        self.renderer.set_hdri(data, width, height, channels);
    }

    pub fn set_texture_atlas(
        &mut self,
        texture_descs: &[u32],
        texture_data: &[f32],
        num_textures: u32,
    ) -> Result<(), String> {
        self.renderer
            .set_texture_atlas(texture_descs, texture_data, num_textures)
            .map_err(|e| format!("set_texture_atlas: {e:?}"))
    }

    /// Shared upload body used by both `new` and `set_scene`.
    fn upload_scene(&mut self, mut scene: SceneState) -> Result<SceneDelta, String> {
        // Force the internal render resolution onto the scene's camera so the
        // film matches the renderer's framebuffer.
        scene.camera.width = self.width;
        scene.camera.height = self.height;

        // Inject the rig's four-light directional rig (sun/sky/camera-fill/rim),
        // mirroring spectra_resident_bench's light setup so lighting is identical
        // to the proven bench path.
        let rig = &self.rig;
        let sun = glam::Vec3::from(rig.sun_dir).normalize_or_zero();
        // Camera-relative fill: derive from the scene camera's forward axis.
        let cam_fwd = scene_camera_forward(&scene);
        let camera_fill = (-cam_fwd + glam::Vec3::Y * 0.35).normalize_or_zero();
        let rim_fill = glam::Vec3::new(-camera_fill.x, 0.55, -camera_fill.z).normalize_or_zero();
        let mut light_data: Vec<f32> = Vec::with_capacity(4 * VULKAN_LIGHT_FLOATS);
        // Fill COLORS now ride the rig (config-driven via render.ron
        // `lighting_rig.analytic_fills`) — the prime blue-cast culprit is one
        // config edit, not a buried literal. Defaults equal the historical triples.
        for (dir, color, intensity) in [
            (sun.to_array(), rig.sun_color, rig.sun_intensity),
            (glam::Vec3::Y.to_array(), rig.analytic_sky_fill_color, rig.sky_intensity),
            (camera_fill.to_array(), rig.analytic_camera_fill_color, rig.camera_fill),
            (rim_fill.to_array(), rig.analytic_rim_fill_color, rig.rim_fill),
        ] {
            light_data.extend_from_slice(&pack_vulkan_directional_light(dir, color, intensity));
        }
        scene.lights = LightLayer {
            light_data,
            light_count: 4,
        };
        scene.mark_lights_changed();

        // TDR DIAGNOSIS: the one-time scene/BVH/TLAS upload is a prime suspect for
        // a single >2s GPU op. Wall-time it when SPECTRA_DISPATCH_TIMING=1.
        let _scene_t = std::time::Instant::now();
        self.renderer
            .load_scene_state(scene)
            .map_err(|e| format!("load_scene_state: {e:?}"))?;
        if std::env::var("SPECTRA_DISPATCH_TIMING").as_deref() == Ok("1") {
            eprintln!(
                "[dispatch_timing] load_scene_state (BVH/TLAS upload): {:.1} ms",
                _scene_t.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Sky-dome + atmosphere from the rig — set after the scene so the state
        // exists. These setters are idempotent and cheap (no GPU rebuild).
        self.renderer.set_sky_gradient(
            rig.sky_dome_zenith,
            rig.sky_dome_horizon,
            rig.sky_dome_intensity,
        );
        if rig.atmosphere_enabled {
            self.renderer.set_sun(sun.to_array(), rig.sun_radiance);
            self.renderer
                .set_atmosphere(true, rig.atmosphere_mie, rig.atmosphere_turbidity);
        }

        let (rebuilt, reused) = self.renderer.last_scene_sync();
        Ok(SceneDelta {
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
        self.delta_ring.set_transform(instance_index as u32, transform);
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
        // Aurora Step 1: the id-sorted, coalesced delta ring is the new plumbing
        // for per-instance transforms (replacing the per-frame CPU Vec push+sort).
        // The persistent resident `g_instances` buffer + `apply_scene_delta` indexed
        // scatter is a PROVEN, determinism-witnessed FOUNDATION — but it is NOT yet
        // read by the trace, so the LIVE render must still drive the *traversed*
        // TLAS via the legacy `MODE_UPDATE` refit (otherwise movers freeze on
        // screen). We therefore drain the ring and feed the legacy refit here; the
        // resident buffer is exercised by the witness + wired into traversal at
        // Steps 2/5, which then retire this legacy refit. No render regression.
        if !self.delta_ring.is_empty() {
            let cmds = self.delta_ring.drain_commands();
            let dirty: Vec<(usize, [[f32; 4]; 3])> = cmds
                .iter()
                .map(|c| (c.slot as usize, c.transform_3x4()))
                .collect();
            self.renderer
                .refit_instance_transforms(&dirty)
                .map_err(|e| format!("refit_instance_transforms: {e:?}"))?;
        }

        self.renderer.set_camera_view_matrix(view);
        self.renderer.set_view_proj(proj);
        self.renderer
            .render()
            .map_err(|e| format!("render: {e:?}"))
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
    pub fn refit_instances(
        &mut self,
        dirty: &[(usize, [[f32; 4]; 3])],
    ) -> Result<bool, String> {
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
    pub fn render_only(
        &mut self,
        view: [f32; 16],
        proj: [f32; 16],
    ) -> Result<FrameOutput, String> {
        self.renderer.set_camera_view_matrix(view);
        self.renderer.set_view_proj(proj);
        self.renderer.render().map_err(|e| format!("render: {e:?}"))
    }

    /// `(instances, clusters)` of the active RTX-Mega-Geometry CLAS scene
    /// (per-prototype CLAS → GAS-over-CLAS → IAS), or `None` when the CLAS path
    /// is not active (single-GAS / software fallback / built below
    /// `SPECTRA_CLAS_THRESHOLD` / OptiX SDK not compiled). The witness that the
    /// Mega-Geometry path actually engaged at scale.
    pub fn clas_stats(&self) -> Option<(usize, usize)> {
        self.renderer.device_clas_stats()
    }

    /// Number of TLAS instances registered after the last scene upload — the
    /// fallback instance count when [`clas_stats`] is `None`.
    pub fn tlas_instance_count(&self) -> usize {
        self.renderer.tlas_instance_count()
    }

    /// A fully-reused [`SceneDelta`] (`rebuilt == 0`, `reused == 1`): the report
    /// for a frame where the caller determined the scene was unchanged and chose
    /// NOT to re-upload it. Lets the live seam prove camera-only moves rebuild no
    /// BLASes without forcing a redundant `set_scene`.
    pub fn reused_delta(&self) -> SceneDelta {
        SceneDelta {
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

    /// The currently-configured render target.
    pub fn render_target(&self) -> spectra_renderer::RenderTarget {
        self.renderer.render_target()
    }

    /// The device pointer the last realtime frame routed its packed RGBA into
    /// (`Some` after an `Interop` frame; `None` on the host-beauty path).
    pub fn last_pack_output_ptr(&self) -> Option<u64> {
        self.renderer.last_pack_output_ptr()
    }

    /// RR (DLSS Ray Reconstruction) guide device ptrs for the last frame, in the
    /// order `(diffuse_albedo, specular_albedo, normals, roughness, depth,
    /// motion)`. See `Renderer::rr_guide_ptrs`.
    pub fn rr_guide_ptrs(&mut self) -> [u64; 6] {
        self.renderer.rr_guide_ptrs()
    }
}

/// Extract the camera forward axis (world -Z of the view) from a scene's
/// column-major view matrix. Used to orient the camera-relative fill lights.
fn scene_camera_forward(scene: &SceneState) -> glam::Vec3 {
    let m = glam::Mat4::from_cols_array(&scene.camera.view_matrix);
    // Forward in world space is the inverse-rotation of view -Z. For an
    // orthonormal view basis, the world forward is the negated third row of the
    // rotation part (column-major: row 2 = elements [2], [6], [10]).
    let fwd = glam::Vec3::new(-m.x_axis.z, -m.y_axis.z, -m.z_axis.z);
    fwd.normalize_or_zero()
}
