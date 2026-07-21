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

use crate::render_runtime::env::elapsed_ms;
use crate::render_runtime::frame::{beauty_to_rgba8, interop_dims};
#[cfg(not(all(target_os = "windows", feature = "spectra-native-optix")))]
use crate::render_runtime::present::{
    DeviceFrame, DevicePixelOrder, DeviceTemporalFrame, VulkanPixelOrder,
};
use crate::render_runtime::terrain::TerrainUpload;
use crate::resident_renderer::FidelityTier;
use crate::resident_renderer::{ResidentSceneRenderer, SceneSyncReport};
use crate::scene_delta_adapter::RetainedDeltaPlan;
use crate::spectral::RenderCamera;
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

#[cfg(not(all(target_os = "windows", feature = "spectra-native-optix")))]
pub struct DevicePresentResult {
    pub frame: DeviceFrame,
    pub sync: SceneSyncReport,
    pub plan: RetainedDeltaPlan,
    pub delta_apply_ms: f64,
    pub render_ms: f64,
    pub refit: bool,
}

#[cfg(not(all(target_os = "windows", feature = "spectra-native-optix")))]
pub type VulkanPresentResult = DevicePresentResult;

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
        Self {
            renderer,
            iw,
            ih,
            tier,
        }
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
        let frame = self
            .renderer
            .render_camera(camera.view.to_cols_array(), camera.proj.to_cols_array())?;
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
        let frame = self
            .renderer
            .render_camera(camera.view.to_cols_array(), camera.proj.to_cols_array())?;
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

    /// Same-device Vulkan/Metal present path. Spectra resolves the frame into a
    /// renderer-owned RGBA8/BGRA8 GPU buffer; the presenter consumes that handle
    /// directly without host beauty conversion or staging upload.
    #[cfg(not(all(target_os = "windows", feature = "spectra-native-optix")))]
    pub fn render_and_present_device(
        &mut self,
        deltas: &mut Vec<vox_scene::SceneDelta>,
        camera: &RenderCamera,
        pixel_order: DevicePixelOrder,
        display: (u32, u32),
        reset_history: bool,
    ) -> Result<DevicePresentResult, String> {
        use crate::resident_renderer::RenderTarget;
        let target = RenderTarget::DeviceBuffer {
            bgra: pixel_order.is_bgra(),
        };
        if self.renderer.render_target() != target {
            self.renderer.set_render_target(target);
        }

        // SNR: a camera cut / resize / teleport must discard reprojected history.
        // The validity masks are generated inside `render_camera` below, so the
        // reset must be requested BEFORE it runs.
        if reset_history {
            self.renderer.request_snr_history_reset();
        }

        let delta_t = std::time::Instant::now();
        let plan = self.renderer.drain_scene_deltas(deltas)?;
        let delta_apply_ms = elapsed_ms(delta_t);
        let sync = self.renderer.reused_delta();
        let refit = plan.stats.refits > 0;
        let render_t = std::time::Instant::now();
        let rendered = self
            .renderer
            .render_camera(camera.view.to_cols_array(), camera.proj.to_cols_array())?;
        let render_ms = elapsed_ms(render_t);
        let buffer = self
            .renderer
            .last_device_ldr_buffer()
            .ok_or_else(|| "renderer produced no device frame".to_string())?;
        let (render_width, render_height) = interop_dims(&rendered, self.iw, self.ih);
        #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
        let backend_identity = self.renderer.metal_backend_identity();
        #[cfg(not(all(target_os = "macos", feature = "spectra-native-metal")))]
        let backend_identity = self.renderer.vulkan_backend_identity();
        let mut frame = DeviceFrame::new(
            buffer,
            backend_identity,
            pixel_order,
            (render_width, render_height),
            display,
            reset_history,
        );
        if let Some(temporal) = self.renderer.last_vulkan_reconstruction() {
            frame = frame.with_temporal(
                DeviceTemporalFrame::new(
                    temporal.color,
                    temporal.depth,
                    temporal.motion,
                    temporal.jitter,
                    temporal.projection,
                )
                .with_reactive(temporal.reactive),
            );
            // SNR: attach the FULL validating `ReconstructionFrameV1` when the
            // renderer produced the complete guide set this frame. The presenter
            // runs SNR only when a Spectra mode is REQUESTED
            // (`set_reconstruction_request`); when Off/DLSS it ignores this and
            // uses the temporal/direct path above, so attaching is safe. The
            // validity masks were generated GPU-side by the renderer (which owns
            // the kernel + the persistent depth/normal history), so they are
            // attached directly — no host readback, no cross-crate kernel compile.
            if let Some(snr) = temporal.snr {
                use crate::render_runtime::present::{
                    ReconstructionCamera, ReconstructionFrameV1, ReconstructionGuides,
                };
                let guides = ReconstructionGuides {
                    color: snr.color,
                    albedo: snr.albedo,
                    normal: snr.normal,
                    depth: snr.depth,
                    motion: snr.motion,
                    roughness: snr.roughness,
                    metallic: snr.metallic,
                    transmission: snr.transmission,
                    emission: snr.emission,
                    reflection_distance: snr.reflection_distance,
                    reflection_hit_class: snr.reflection_hit_class,
                };
                let camera = ReconstructionCamera {
                    exposure_ev: snr.exposure_ev,
                    camera_near: snr.camera_near,
                    camera_far: snr.camera_far,
                    camera_fov_y: snr.camera_fov_y,
                    jitter: snr.jitter,
                    frame_index: snr.frame_index,
                    reset_history: reset_history || snr.reset_history,
                };
                let recon =
                    ReconstructionFrameV1::new(backend_identity, snr.internal, guides, camera)
                        .with_validity_masks(snr.reactive, snr.disocclusion, snr.history_valid);
                frame = frame.with_reconstruction(recon);
            }
        }
        Ok(DevicePresentResult {
            frame,
            sync,
            plan,
            delta_apply_ms,
            render_ms,
            refit,
        })
    }

    #[cfg(not(all(target_os = "windows", feature = "spectra-native-optix")))]
    pub fn render_and_present_vulkan(
        &mut self,
        deltas: &mut Vec<vox_scene::SceneDelta>,
        camera: &RenderCamera,
        pixel_order: VulkanPixelOrder,
        display: (u32, u32),
        reset_history: bool,
    ) -> Result<VulkanPresentResult, String> {
        self.render_and_present_device(deltas, camera, pixel_order, display, reset_history)
    }

    /// RR (DLSS Ray Reconstruction) guide device pointers for the frame the last
    /// GPU present produced, packed into the present's `RrGuides`.
    ///
    /// Body of the game's `LiveFrameSource::rr_guides` ported verbatim
    /// (`self.runtime.renderer_mut().` → `self.renderer.`). `&mut self` because
    /// `rr_guide_ptrs` takes `&mut self` on the inner renderer.
    /// Enable present-side temporal guidance (jitter + MV) for an active
    /// temporal present backend.
    pub fn set_present_temporal_upscale(&mut self, on: bool) {
        self.renderer.set_present_temporal_upscale(on);
    }

    /// True only when the active present backend is actually doing external RR
    /// denoise+upscale. SR fallback still gets temporal guidance, but internal
    /// denoise/temporal must remain available.
    pub fn set_present_ray_reconstruction(&mut self, on: bool) {
        self.renderer.set_present_ray_reconstruction(on);
    }

    /// Enable/disable production of the full SNR `ReconstructionFrameV1` guide set
    /// on the Vulkan device present path. Default on (so a settings-driven SNR
    /// request activates without extra wiring); a game can gate this by
    /// reconstruction mode to skip the guide cost on Off/DLSS tiers.
    pub fn set_snr_guides_enabled(&mut self, on: bool) {
        self.renderer.set_snr_guides_enabled(on);
    }

    pub fn rr_guides(&mut self) -> crate::render_runtime::present::RrGuides {
        let g = self.renderer.rr_guide_ptrs();
        let (jx, jy) = self.renderer.rr_jitter();
        let ready_event = self.renderer.rr_payload_ready_event();
        // PACK_MOTION_VECTORS already converts NDC reprojection into pixel-space
        // current->previous motion. NGX MV.Scale is only for inputs that still
        // need scaling into pixels, so the correct scale here is identity.
        let mv_scale_x = 1.0;
        let mv_scale_y = 1.0;
        let guides = crate::render_runtime::present::RrGuides {
            diffuse_albedo: g[0],
            specular_albedo: g[1],
            normals: g[2],
            roughness: g[3],
            depth: g[4],
            motion: g[5],
            ready_event,
            jitter_x: jx,
            jitter_y: jy,
            mv_scale_x,
            mv_scale_y,
        };
        if std::env::var("SPECTRA_RR_GUIDE_DIAG").as_deref() == Ok("1") {
            use std::sync::atomic::{AtomicU32, Ordering};
            static GUIDE_DIAG_FRAME: AtomicU32 = AtomicU32::new(0);
            let n = GUIDE_DIAG_FRAME.fetch_add(1, Ordering::Relaxed);
            if n % 30 == 0 {
                eprintln!(
                    "[rr-guides] f{n} diffuse={} specular={} normals={} roughness={} depth={} motion={} \
                     jitter=({:.3},{:.3}) mv_scale=({:.3},{:.3})",
                    guides.diffuse_albedo != 0,
                    guides.specular_albedo != 0,
                    guides.normals != 0,
                    guides.roughness != 0,
                    guides.depth != 0,
                    guides.motion != 0,
                    guides.jitter_x,
                    guides.jitter_y,
                    guides.mv_scale_x,
                    guides.mv_scale_y,
                );
            }
        }
        guides
    }

    // ---- setup methods (the clean runtime API; games never touch the inner renderer) ----

    /// Upload native material texture objects. Call before [`Self::set_terrain`]
    /// because terrain slope/spray slots reference the same resident slot order.
    pub fn set_material_textures(
        &mut self,
        textures: &[spectra_renderer::RendererTexture2D],
    ) -> Result<(), String> {
        self.renderer.set_material_textures(textures)
    }

    /// Upload native material texture objects and keep the host descriptor/f32
    /// mirror available for OMM baking/cache. The render shaders sample native
    /// texture objects; the mirror is not a GPU texel buffer.
    pub fn set_material_textures_with_mirror(
        &mut self,
        textures: &[spectra_renderer::RendererTexture2D],
        texture_descs: &[u32],
        texture_data: &[f32],
    ) -> Result<(), String> {
        self.renderer
            .set_material_textures_with_mirror(textures, texture_descs, texture_data)
    }

    /// Bind sparse virtual texture residency for material texture slots.
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
        self.renderer.set_svt_residency(
            texture_descs,
            page_table,
            tile_cache_rgba,
            texture_count,
            tile_size,
            cache_tiles,
            max_tiles_per_axis,
            texture_size,
        )
    }

    pub fn clear_svt_residency(&mut self) -> Result<(), String> {
        self.renderer.clear_svt_residency()
    }

    /// Bind an equirectangular HDRI for image-based sky + lighting (`channels`
    /// usually 3). Empty `data` clears it (back to the procedural gradient sky).
    /// Call after the scene exists. Pass-through to the inner renderer.
    pub fn set_hdri(&mut self, data: &[f32], width: u32, height: u32, channels: u32) {
        self.renderer.set_hdri(data, width, height, channels);
    }

    /// Upload the full terrain material payload — spray field + slope/height
    /// layered material + per-pixel curvature — from a GPU-free [`TerrainUpload`].
    /// MUST be called AFTER [`Self::set_material_textures`] (slope/spray slots
    /// reference resident texture entries). This is the engine home of the game's former
    /// `upload_spray_and_terrain` (ported verbatim, `renderer.` → `self.renderer.`,
    /// `spray_upload.` → `upload.`).
    pub fn set_terrain(&mut self, upload: &TerrainUpload) -> Result<(), String> {
        if !upload.packed.is_empty() {
            let channel_count = upload.channel_slots.len() / vox_core::spray::CHANNEL_SLOT_INTS;
            if !upload.channel_uv_scales.is_empty()
                && upload.channel_uv_scales.len()
                    != channel_count * vox_core::spray::CHANNEL_UV_FLOATS
            {
                return Err(format!(
                    "terrain channel UV contract mismatch: {} floats for {channel_count} channels (expected {})",
                    upload.channel_uv_scales.len(),
                    channel_count * vox_core::spray::CHANNEL_UV_FLOATS
                ));
            }
            self.renderer.set_spray_field(
                &upload.packed,
                upload.res,
                upload.origin,
                upload.cell_size,
                &upload.channel_slots,
                &upload.channel_uv_scales,
            )?;
            eprintln!(
                "[spray] field uploaded: {}x{} cells @ {:.2}m, {} channels, {} UV-scale pairs",
                upload.res[0],
                upload.res[1],
                upload.cell_size,
                channel_count,
                upload.channel_uv_scales.len() / vox_core::spray::CHANNEL_UV_FLOATS
            );
        }
        self.renderer.set_ground_macro(
            upload.ground_macro_slot,
            upload.ground_macro_tile_m,
            upload.ground_macro_luma_reference,
            upload.ground_macro_blend,
        )?;
        eprintln!(
            "[ground-macro] slot={} tile_m={:.1} luma_ref={:.4} blend={:.2}",
            upload.ground_macro_slot,
            upload.ground_macro_tile_m,
            upload.ground_macro_luma_reference,
            upload.ground_macro_blend
        );
        self.renderer.set_slope_layers(
            upload.slope_rock_albedo,
            upload.slope_rock_normal,
            upload.slope_dirt_albedo,
            upload.slope_dirt_normal,
            upload.slope_rock_disp,
            upload.slope_dirt_disp,
        );
        eprintln!(
            "[slope-layer] rock_albedo={} rock_normal={} dirt_albedo={} dirt_normal={} (>=0 = layered terrain on)",
            upload.slope_rock_albedo,
            upload.slope_rock_normal,
            upload.slope_dirt_albedo,
            upload.slope_dirt_normal
        );
        if !upload.curvature_values.is_empty() {
            self.renderer.set_curvature_field(
                &upload.curvature_values,
                upload.curvature_res,
                upload.curvature_origin,
                upload.curvature_cell_size,
            )?;
            eprintln!(
                "[curvature] per-pixel field uploaded: {}x{} cells @ {:.2}m",
                upload.curvature_res[0], upload.curvature_res[1], upload.curvature_cell_size
            );
        }
        // WATER-DEPTH FIELD + UNDERWATER BED layers (P3, ROOT 4). Upload the static
        // per-cell water column so the megakernel can blend the bed by depth (and
        // darken/roughen the intertidal wet band) instead of painting submerged
        // ground as flat Coastal grass/sand. Empty `depth_values` leaves both OFF
        // (byte-identical). Order does not matter; both are static scene data.
        self.renderer.set_underwater_layers(
            upload.uw_wet_albedo,
            upload.uw_wet_normal,
            upload.uw_wet_rough,
            upload.uw_mud_albedo,
            upload.uw_mud_normal,
            upload.uw_mud_rough,
            upload.uw_silt_albedo,
            upload.uw_silt_normal,
            upload.uw_silt_rough,
            upload.uw_bed_albedo,
            upload.uw_bed_normal,
            upload.uw_bed_rough,
        );
        if !upload.depth_values.is_empty() {
            self.renderer.set_depth_field(
                &upload.depth_values,
                upload.depth_res,
                upload.depth_origin,
                upload.depth_cell_size,
                upload.depth_sea_level,
            )?;
            eprintln!(
                "[water-depth] per-cell field uploaded: {}x{} cells @ {:.2}m, sea_y={:.2} (uw_wet={} uw_mud={} uw_silt={})",
                upload.depth_res[0],
                upload.depth_res[1],
                upload.depth_cell_size,
                upload.depth_sea_level,
                upload.uw_wet_albedo,
                upload.uw_mud_albedo,
                upload.uw_silt_albedo
            );
        }
        if !upload.flow_values.is_empty() {
            self.renderer
                .set_water_flow_field(&upload.flow_values, true)?;
            eprintln!(
                "[water-flow] per-cell flow field uploaded: {} cells (vx,vz) on the {}x{} depth grid",
                upload.flow_values.len() / 2,
                upload.depth_res[0],
                upload.depth_res[1]
            );
        }
        self.renderer.set_slope_snow(
            upload.slope_snow_albedo,
            upload.slope_snow_normal,
            upload.slope_snow_disp,
            upload.slope_height_snow,
            upload.slope_height_snow_band,
            vox_config::config().terrain.snow_slope_cos,
        );
        eprintln!(
            "[snow-cap] snow_albedo={} snow_line={:.0}m band={:.0}m (>=0 + finite = snow on)",
            upload.slope_snow_albedo, upload.slope_height_snow, upload.slope_height_snow_band
        );
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

    /// K5 (animated water): per-frame refit of the water surface node's vertices
    /// (see [`ResidentSceneRenderer::refit_water_geometry`]). No-op `Ok(false)` off
    /// the CLAS IAS path (AMD/Vulkan) or when the node isn't resident.
    pub fn refit_water_geometry(
        &mut self,
        node: NodeId,
        verts: &[[f32; 3]],
    ) -> Result<bool, String> {
        self.renderer.refit_water_geometry(node, verts)
    }
}
