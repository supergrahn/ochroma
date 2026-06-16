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
use spectra_scene_state::{LightLayer, SceneState};

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
    /// Pending per-instance transform refits, recorded by
    /// [`update_instance_transform`] and applied on the next `render_camera`.
    /// (instance_index, row-major 3×4 transform). Full GPU refit lands in T4;
    /// here the record is held so the live seam can call the method today.
    pending_refits: Vec<(usize, [[f32; 4]; 3])>,
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
        if std::env::var("OCHROMA_SHADE_LEAN").as_deref() == Ok("1") {
            config.prefer_lean_shade = true;
        }
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
        // ROOT-CAUSE FIX (black buildings on CUDA): the Hero4 spectral path routes
        // primary-hit NEE direct lighting into the spectral shadow buffers
        // (megakernel hwss branch) instead of g_shadow_contrib_r → apply_shadows →
        // film. On the CUDA precompiled-PTX path that spectral contribution never
        // lands on the final RGB film, so every lit surface (the whole merged city
        // soup) integrates to ~0 and renders as a black silhouette under a lit sky
        // — exactly the "only sky, no buildings" symptom. Force the scalar
        // (SpectralMode::Single) NEE path, whose g_shadow_contrib_r → apply_shadows
        // resolve is the proven-working lighting route, until the spectral CUDA
        // resolve is wired. Overridable via OCHROMA_SPECTRAL=1 for the spectral work.
        if std::env::var("OCHROMA_SPECTRAL").as_deref() != Ok("1") {
            config.spectral_mode = spectra_renderer::SpectralMode::Single;
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
            pending_refits: Vec::new(),
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

    /// Record one instance's transform refit (movers, articulated rigid
    /// sub-parts). Applied on the next [`render_camera`] via the GPU
    /// `MODE_UPDATE` refit (`refit_instanced_tlas`) — BLASes untouched, no scene
    /// rebuild. Records are coalesced (latest-wins per instance) and id-sorted
    /// for deterministic apply order.
    ///
    /// `transform` is a row-major 3×4 (upper rows of a 4×4 world transform).
    pub fn update_instance_transform(&mut self, instance_index: usize, transform: [[f32; 4]; 3]) {
        // Coalesce: keep only the latest transform per instance this frame
        // (id-ordered, no HashMap — deterministic).
        if let Some(slot) = self
            .pending_refits
            .iter_mut()
            .find(|(i, _)| *i == instance_index)
        {
            slot.1 = transform;
        } else {
            self.pending_refits.push((instance_index, transform));
            // Keep id-sorted for deterministic apply order (T4 wiring).
            self.pending_refits.sort_by_key(|(i, _)| *i);
        }
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
        // Flush coalesced, id-sorted mover transforms through the GPU
        // `MODE_UPDATE` refit (BLASes untouched, no scene rebuild). Draining
        // keeps determinism: the same sequence of frames yields the same state.
        if !self.pending_refits.is_empty() {
            let dirty: Vec<(usize, [[f32; 4]; 3])> = std::mem::take(&mut self.pending_refits);
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
