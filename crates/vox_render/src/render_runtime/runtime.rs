//! `RenderRuntime` — the construct-once, engine-side render service.
//!
//! FIRST SLICE: wraps a `ResidentSceneRenderer` and owns the game-agnostic
//! *delta-only present loop* (drain queued `SceneDelta` refits → stream camera →
//! trace → convert to RGBA8). A new game gets this present/optimization path for
//! free instead of re-implementing it.
//!
//! This slice is intentionally a transition layer: the wrapper is constructed
//! FROM an already-built `ResidentSceneRenderer` ([`RenderRuntime::new_wrapping`])
//! and exposes a transition accessor ([`RenderRuntime::renderer_mut`]) so the
//! game's not-yet-moved setup/scene methods keep driving the inner renderer.
//! Decoupling full construction from `GameState` is a LATER slice.

use crate::resident_renderer::{ResidentSceneRenderer, SceneSyncReport};
use crate::resident_renderer::FidelityTier;
use crate::scene_delta_adapter::RetainedDeltaPlan;
use crate::spectral::RenderCamera;
use crate::render_runtime::env::elapsed_ms;
use crate::render_runtime::frame::{beauty_to_rgba8, interop_dims};
use crate::render_runtime::terrain::TerrainUpload;
use vox_scene::NodeId;

// CONFIG-FIRST: the snow-cap slope cutoff (geometric up-cosine — caps sit on
// ledges/peaks, not vertical faces) is `config/ochroma.ron` `terrain.snow_slope_cos`
// (default 0.78 == the old literal). Read at the `set_slope_snow` call site below.

/// Result of a host-readback present (`render_and_present`).
///
/// Carries the converted frame plus the timing/scene proofs the caller needs to
/// drive its own per-frame breakdown logging (which references project-specific
/// env gates + counters that stay game-side in this slice).
pub struct PresentResult {
    /// RGBA8 bytes + dimensions of the presented frame.
    pub rgba8: (Vec<u8>, u32, u32),
    /// Scene-sync proof for this frame (refit/reuse witness).
    pub sync: SceneSyncReport,
    /// The drained retained-delta plan (refit vs structural-rebuild proof).
    pub plan: RetainedDeltaPlan,
    /// Milliseconds spent draining + applying queued deltas.
    pub delta_apply_ms: f64,
    /// Milliseconds spent in `render_camera`.
    pub render_ms: f64,
    /// True when this present applied at least one TLAS refit.
    pub refit: bool,
}

/// Result of a zero-copy GPU present (`render_and_present_gpu`).
///
/// The GPU twin of [`PresentResult`]: the present routes the packed RGBA f32
/// beauty into the CUDA interop `color_ptr` (no host readback), so instead of
/// RGBA8 bytes this carries the resolved interop output pointer + dims plus the
/// same timing/scene proofs the caller needs for its per-frame breakdown logging
/// (which references project-specific env gates + counters that stay game-side).
pub struct GpuPresentResult {
    /// Device pointer the frame's packed RGBA actually landed in (the renderer's
    /// `last_pack_output_ptr`, falling back to the requested `color_ptr`).
    pub out_ptr: u64,
    /// Render-resolution width of the presented interop frame.
    pub rw: u32,
    /// Render-resolution height of the presented interop frame.
    pub rh: u32,
    /// Scene-sync proof for this frame (refit/reuse witness).
    pub sync: SceneSyncReport,
    /// The drained retained-delta plan (refit vs structural-rebuild proof).
    pub plan: RetainedDeltaPlan,
    /// Milliseconds spent draining + applying queued deltas.
    pub delta_apply_ms: f64,
    /// Milliseconds spent in `render_camera`.
    pub render_ms: f64,
    /// True when this present applied at least one TLAS refit.
    pub refit: bool,
}

/// Construct-once render service wrapping a resident renderer.
pub struct RenderRuntime {
    renderer: ResidentSceneRenderer,
    iw: u32,
    ih: u32,
    tier: FidelityTier,
}

impl RenderRuntime {
    /// Wrap an ALREADY-BUILT `ResidentSceneRenderer`.
    ///
    /// The game keeps performing the `GameState`-coupled construction (scene
    /// assembly, atlas/terrain upload, retained-mirror stamp) for now; full
    /// construction decoupling is a later slice. `iw`/`ih` are the internal
    /// render resolution the renderer was built at; `tier` its fidelity tier.
    pub fn new_wrapping(
        renderer: ResidentSceneRenderer,
        iw: u32,
        ih: u32,
        tier: FidelityTier,
    ) -> Self {
        Self { renderer, iw, ih, tier }
    }

    /// Internal render resolution this runtime produces (`(iw, ih)`).
    pub fn internal_size(&self) -> (u32, u32) {
        (self.iw, self.ih)
    }

    /// The fidelity tier this runtime renders at.
    pub fn tier(&self) -> FidelityTier {
        self.tier
    }

    /// Transition accessor — shared reference to the inner renderer so the game
    /// can keep calling its not-yet-moved setup/scene methods.
    ///
    /// `pub` (not `pub(crate)`) because the game's `LiveFrameSource` lives in a
    /// DIFFERENT crate and must reach the inner renderer during the migration.
    /// This is a deliberate transition seam: it shrinks as setup/scene methods
    /// migrate into `RenderRuntime` in later slices, and is removed once empty.
    #[doc(hidden)]
    pub fn renderer(&self) -> &ResidentSceneRenderer {
        &self.renderer
    }

    /// Transition accessor — mutable reference to the inner renderer (see
    /// [`Self::renderer`]).
    #[doc(hidden)]
    pub fn renderer_mut(&mut self) -> &mut ResidentSceneRenderer {
        &mut self.renderer
    }

    /// Drain and apply queued retained scene deltas as TLAS refits. Returns
    /// `Err` (without clearing `deltas`) for any structural delta — the caller
    /// must then trigger a full-scene rebuild. Clears `deltas` on success.
    pub fn drain_scene_deltas(
        &mut self,
        deltas: &mut Vec<vox_scene::SceneDelta>,
    ) -> Result<RetainedDeltaPlan, String> {
        self.renderer.drain_scene_deltas(deltas)
    }

    /// Retained host-readback present loop: drain scene deltas, stream the
    /// camera, trace, and convert to RGBA8.
    ///
    /// This is the body of the game's `LiveFrameSource::frame_with_scene_deltas`
    /// ported verbatim (minus the project-specific breakdown logging, which the
    /// game wraps around this call). `deltas`: queued `vox_scene::SceneDelta`
    /// batches since the last frame, cleared on success. `camera`: the engine
    /// `RenderCamera` (view + proj).
    pub fn render_and_present(
        &mut self,
        deltas: &mut Vec<vox_scene::SceneDelta>,
        camera: &RenderCamera,
    ) -> Result<PresentResult, String> {
        let delta_t = std::time::Instant::now();
        let plan = self.renderer.drain_scene_deltas(deltas)?;
        let delta_apply_ms = elapsed_ms(delta_t);
        let upload_delta = self.renderer.reused_delta();
        let refit = plan.stats.refits > 0;
        let render_t = std::time::Instant::now();
        let frame = self.renderer.render_camera(
            camera.view.to_cols_array(),
            camera.proj.to_cols_array(),
        )?;
        let render_ms = elapsed_ms(render_t);
        Ok(PresentResult {
            rgba8: beauty_to_rgba8(&frame),
            sync: upload_delta,
            plan,
            delta_apply_ms,
            render_ms,
            refit,
        })
    }

    /// ZERO-COPY GPU present path (M1 DLSS): drain scene deltas as TLAS refits,
    /// stream the camera, trace, and route the packed RGBA f32 beauty into the
    /// CUDA interop pointer `color_ptr` (no host readback).
    ///
    /// This is the drain + render core of the game's
    /// `LiveFrameSource::frame_gpu_with_scene_deltas` ported verbatim (`self.runtime
    /// .renderer_mut().` → `self.renderer.`), minus the project-specific breakdown
    /// logging (`scene_assembly_count` / `print_scene_frame_breakdown`), which the
    /// game wraps around this call using the returned [`GpuPresentResult`] proofs.
    /// `deltas`: queued `vox_scene::SceneDelta` batches since the last frame,
    /// cleared on success. `camera`: the engine `RenderCamera` (view + proj).
    pub fn render_and_present_gpu(
        &mut self,
        deltas: &mut Vec<vox_scene::SceneDelta>,
        camera: &RenderCamera,
        color_ptr: u64,
    ) -> Result<GpuPresentResult, String> {
        use crate::resident_renderer::RenderTarget;
        let want = RenderTarget::Interop { color_ptr };
        if self.renderer.render_target() != want {
            self.renderer.set_render_target(want);
        }
        let delta_t = std::time::Instant::now();
        let plan = self.renderer.drain_scene_deltas(deltas)?;
        let delta_apply_ms = elapsed_ms(delta_t);
        let upload_delta = self.renderer.reused_delta();
        let refit = plan.stats.refits > 0;
        let render_t = std::time::Instant::now();
        let frame = self.renderer.render_camera(
            camera.view.to_cols_array(), camera.proj.to_cols_array())?;
        let render_ms = elapsed_ms(render_t);
        let out_ptr = self.renderer.last_pack_output_ptr().unwrap_or(color_ptr);
        let (rw, rh) = interop_dims(&frame, self.iw, self.ih);
        Ok(GpuPresentResult {
            out_ptr,
            rw,
            rh,
            sync: upload_delta,
            plan,
            delta_apply_ms,
            render_ms,
            refit,
        })
    }

    /// RR (DLSS Ray Reconstruction) guide device pointers for the frame the last
    /// GPU present produced, packed into the present's `RrGuides`.
    ///
    /// Body of the game's `LiveFrameSource::rr_guides` ported verbatim
    /// (`self.runtime.renderer_mut().` → `self.renderer.`). `&mut self` because
    /// `rr_guide_ptrs` takes `&mut self` on the inner renderer.
    pub fn rr_guides(&mut self) -> spectra_present::RrGuides {
        let g = self.renderer.rr_guide_ptrs();
        let (jx, jy) = self.renderer.rr_jitter();
        spectra_present::RrGuides {
            diffuse_albedo: g[0],
            specular_albedo: g[1],
            normals: g[2],
            roughness: g[3],
            depth: g[4],
            motion: g[5],
            jitter_x: jx,
            jitter_y: jy,
            mv_scale_x: 1.0,
            mv_scale_y: 1.0,
        }
    }

    // ---- setup methods (the clean runtime API; games never touch the inner renderer) ----

    /// Upload the path tracer's flat texture atlas (descs = `[offset,w,h,channels]`
    /// per texture; data = concatenated linear samples; `num_textures` entries).
    /// Pass-through to the inner renderer. Call before [`Self::set_terrain`] (the
    /// terrain slope/spray slots reference resident atlas entries).
    pub fn set_atlas(
        &mut self,
        texture_descs: &[u32],
        texture_data: &[f32],
        num_textures: u32,
    ) -> Result<(), String> {
        self.renderer
            .set_texture_atlas(texture_descs, texture_data, num_textures)
    }

    /// Bind an equirectangular HDRI for image-based sky + lighting (`channels`
    /// usually 3). Empty `data` clears it (back to the procedural gradient sky).
    /// Call after the scene exists. Pass-through to the inner renderer.
    pub fn set_hdri(&mut self, data: &[f32], width: u32, height: u32, channels: u32) {
        self.renderer.set_hdri(data, width, height, channels);
    }

    /// Upload the full terrain material payload — spray field + slope/height
    /// layered material + per-pixel curvature — from a GPU-free [`TerrainUpload`].
    /// MUST be called AFTER [`Self::set_atlas`] (slope/spray slots reference
    /// resident atlas entries). This is the engine home of the game's former
    /// `upload_spray_and_terrain` (ported verbatim, `renderer.` → `self.renderer.`,
    /// `spray_upload.` → `upload.`).
    pub fn set_terrain(&mut self, upload: &TerrainUpload) -> Result<(), String> {
        if !upload.packed.is_empty() {
            self.renderer.set_spray_field(
                &upload.packed, upload.res,
                upload.origin, upload.cell_size,
                &upload.channel_slots,
            )?;
            eprintln!("[spray] field uploaded: {}x{} cells @ {:.2}m, {} channels",
                upload.res[0], upload.res[1], upload.cell_size,
                upload.channel_slots.len() / 4);
        }
        self.renderer.set_slope_layers(
            upload.slope_rock_albedo, upload.slope_rock_normal,
            upload.slope_dirt_albedo, upload.slope_dirt_normal,
            upload.slope_rock_disp, upload.slope_dirt_disp,
        );
        eprintln!("[slope-layer] rock_albedo={} rock_normal={} dirt_albedo={} dirt_normal={} (>=0 = layered terrain on)",
            upload.slope_rock_albedo, upload.slope_rock_normal,
            upload.slope_dirt_albedo, upload.slope_dirt_normal);
        if !upload.curvature_values.is_empty() {
            self.renderer.set_curvature_field(
                &upload.curvature_values,
                upload.curvature_res,
                upload.curvature_origin,
                upload.curvature_cell_size,
            )?;
            eprintln!("[curvature] per-pixel field uploaded: {}x{} cells @ {:.2}m",
                upload.curvature_res[0], upload.curvature_res[1],
                upload.curvature_cell_size);
        }
        self.renderer.set_slope_snow(
            upload.slope_snow_albedo, upload.slope_snow_normal,
            upload.slope_snow_disp, upload.slope_height_snow,
            upload.slope_height_snow_band,
            vox_config::config().terrain.snow_slope_cos,
        );
        eprintln!("[snow-cap] snow_albedo={} snow_line={:.0}m band={:.0}m (>=0 + finite = snow on)",
            upload.slope_snow_albedo, upload.slope_height_snow,
            upload.slope_height_snow_band);
        Ok(())
    }

    // ---- retained mirror accessors (thin delegation to ResidentSceneRenderer) ----

    /// Renderer instance index currently mirrored for `node`, if present.
    pub fn retained_instance_index(&self, node: NodeId) -> Option<usize> {
        self.renderer.retained_instance_index(node)
    }

    /// Retained node currently mirrored at `instance_index`, if any.
    pub fn retained_node_for_instance(&self, instance_index: usize) -> Option<NodeId> {
        self.renderer.retained_node_for_instance(instance_index)
    }

    /// Number of retained nodes currently mapped to renderer instances.
    pub fn retained_node_count(&self) -> usize {
        self.renderer.retained_node_count()
    }
}
