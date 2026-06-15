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

use spectra_gpu::VulkanSlangBackend;
use spectra_renderer::{FrameOutput, RenderConfig, Renderer};
use spectra_scene_state::{LightLayer, SceneState};

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
    renderer: Renderer<VulkanSlangBackend>,
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
        let gpu = VulkanSlangBackend::new(0).map_err(|e| format!("vulkan backend init: {e:?}"))?;
        let mut config = RenderConfig::near_realtime(width, height);
        config.slang_kernel_dir = resolve_slang_kernel_dir();
        if std::env::var("OCHROMA_SHADE_LEAN").as_deref() == Ok("1") {
            config.prefer_lean_shade = true;
        }
        let mut settings = rig_to_settings(&rig, spp, max_bounces);
        seed_features_from_config(&mut settings, &config);
        config.apply_settings(&settings);
        config.max_bounces = max_bounces;

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

        self.renderer
            .load_scene_state(scene)
            .map_err(|e| format!("load_scene_state: {e:?}"))?;

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

    /// The light rig driving this renderer.
    pub fn rig(&self) -> &LightRig {
        &self.rig
    }

    /// Internal render resolution.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
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
