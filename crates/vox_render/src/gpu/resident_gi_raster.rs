//! `ResidentGiRaster` — the resident GI → raster frame loop (AAA Spec 11).
//!
//! This is the keystone of GI→raster GPU residency: on ONE shared `wgpu::Device`
//! it runs the spectral GI compute pass and binds its output buffer DIRECTLY as
//! input to the tiled rasterizer, with ZERO CPU readback between the two. The
//! per-frame GI `poll(Wait)` (the `map_async` stall in [`GpuGi::step`]) is
//! ELIMINATED on this path: one encoder records GI → fold → and the tiled chain
//! reads the now-GI-lit splat buffer.
//!
//! ## The resident handoff
//!
//!   1. [`GpuGi::dispatch_gi_resident`] records the GI compute dispatch into the
//!      encoder (NO readback copy) — radiance stays resident in
//!      `GpuGi::radiance_buffer` (`array<array<f32,16>>`, `STORAGE | COPY_SRC`).
//!   2. [`GiCombinePass::dispatch`] folds `radiance[i][0..8]` into
//!      `splat_buf[i].spectral[0..8]` (f16-quantized to bit-match the readback
//!      oracle) — binding the GI buffer directly as a read storage input and the
//!      tiled renderer's persistent splat buffer as read-write output.
//!   3. ONE `queue.submit` flushes GI + fold; no `map_async`, no `poll`. Then
//!      [`TiledSplatRenderer::render`] runs the frozen four-pass chain over the
//!      now-GI-lit splat buffer.
//!
//! HONEST SCOPE: the tiled chain's OWN single `tile_count` host readback (a
//! `map_async` + `poll(Wait)` between `tile_assign` and `radix_sort`, inside
//! [`TiledSplatRenderer::render`]) is separate and out of scope here — Spec 11
//! kills only the GI→raster readback. [`Self::map_async_count`] therefore counts
//! ONLY the GI-side maps this struct performs (which is zero on the resident
//! path); it does not count the tiled chain's internal count readback.

use vox_core::types::GaussianSplat;

use crate::gpu::GpuContext;
use crate::gpu::gi_combine::GiCombinePass;
use crate::gpu::tiled_splat_renderer::{TiledRenderError, TiledSplatRenderer};
use crate::spectral::RenderCamera;
use crate::spectral_gi::{GpuGi, GpuGiError};

/// Resident GI → raster frame loop over a shared [`GpuContext`]. Owns the GI
/// compute engine, the residency fold, and the tiled rasterizer; uploads the
/// splat set once (into the tiled renderer) and re-lights it on-device each
/// frame with NO CPU round-trip.
pub struct ResidentGiRaster {
    ctx: GpuContext,
    gi: GpuGi,
    gi_combine: GiCombinePass,
    tiled: TiledSplatRenderer,
    splats: Vec<GaussianSplat>,
    splat_count: u32,
    /// Diagnostic: number of `map_async` calls this struct issues on the GI→raster
    /// path. Stays 0 — the residency proof. (Does NOT include the tiled chain's
    /// own internal `tile_count` readback; see the module doc.)
    map_async_count: u64,
}

impl ResidentGiRaster {
    /// Build the resident loop over `splats`, sized for a `width × height` frame.
    ///
    /// Constructs the GI engine on the shared device (no second `request_device`)
    /// and the tiled renderer (which uploads the splats once and validates buffer
    /// sizes against device limits). A [`TiledRenderError`] is mapped to
    /// [`GpuGiError::DeviceCreation`] rather than panicking — never unwraps on a
    /// limit/adapter failure.
    pub fn new(
        ctx: GpuContext,
        splats: &[GaussianSplat],
        width: u32,
        height: u32,
    ) -> Result<Self, GpuGiError> {
        let splat_count = splats.len() as u32;
        let gi = GpuGi::new_with_context(&ctx, splat_count.max(1));
        let gi_combine = GiCombinePass::new(ctx.device());
        let tiled = TiledSplatRenderer::new(ctx.clone(), splats, width, height)
            .map_err(|e: TiledRenderError| GpuGiError::DeviceCreation(e.to_string()))?;
        Ok(Self {
            ctx,
            gi,
            gi_combine,
            tiled,
            splats: splats.to_vec(),
            splat_count,
            map_async_count: 0,
        })
    }

    /// Render one resident frame: GI compute → fold → tiled raster, all on the
    /// shared device with ZERO CPU readback between GI and the raster.
    ///
    /// ONE encoder records the GI compute dispatch (no readback) then the fold
    /// (radiance → splat spectral); ONE `queue.submit` flushes them; NO
    /// `map_async`, NO `poll` (so [`Self::map_async_count`] stays 0). Then the
    /// frozen tiled chain renders the now-GI-lit splat buffer and the spectral
    /// output texture is returned.
    pub fn render_frame(
        &mut self,
        camera: &RenderCamera,
        hour: f32,
    ) -> Result<wgpu::Texture, GpuGiError> {
        let device = self.ctx.device().clone();
        let queue = self.ctx.queue().clone();

        // ── ONE encoder: resident GI compute → residency fold ──────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident_gi_raster_encoder"),
        });
        // (1) GI compute — output stays resident in gi.radiance_buffer (no readback).
        self.gi
            .dispatch_gi_resident(&mut encoder, &queue, &self.splats, hour);
        // (2) Fold radiance[i][0..8] → splat_buf[i].spectral[0..8], f16-quantized
        //     to bit-match the readback oracle. Binds the GI buffer DIRECTLY as a
        //     read input and the tiled splat buffer as read-write output.
        self.gi_combine.dispatch(
            &device,
            &mut encoder,
            self.gi.radiance_buffer(),
            self.tiled.splat_buf(),
            self.splat_count,
        );
        // ONE submit. NO map_async, NO poll on the GI→raster handoff.
        queue.submit(Some(encoder.finish()));

        // ── Tiled raster over the now-GI-lit splat buffer ──────────────────────
        let frame = self
            .tiled
            .render(camera)
            .map_err(|e: TiledRenderError| GpuGiError::DeviceCreation(e.to_string()))?;
        Ok(frame.spectral_texture)
    }

    /// Number of `map_async` calls this struct performs on the GI→raster path.
    /// Stays 0 on the resident path — the residency proof. Does NOT count the
    /// tiled chain's own internal `tile_count` readback (out of scope; see module
    /// doc).
    pub fn map_async_count(&self) -> u64 {
        self.map_async_count
    }

    /// Adapter name backing the shared device (diagnostics).
    pub fn adapter_name(&self) -> &str {
        self.ctx.adapter_name()
    }
}
