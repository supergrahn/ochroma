//! `TiledSplatRenderer` — the real-time on-device tiled splat chain.
//!
//! This is the renderer slice of AAA Spec 05: it uploads a splat set ONCE into
//! persistent GPU buffers and then, per frame, runs the validated four-pass
//! tiled chain entirely on the device —
//!
//!   `tile_assign` → `radix_sort` → `tile_range_build` → `splat_raster`
//!
//! — drawing 8 spectral bands + transmittance into a reused 4-layer
//! `rgba32float` texture array, and reports a measured GPU-millisecond reading
//! from the shared [`GpuTimers`] harness. The four passes are the FROZEN,
//! bit-exact-validated foundation ([`TileAssignPass`], [`RadixSortPass`],
//! [`TileRangeBuildPass`], [`SplatRasterPass`]); this renderer only wires them
//! together with persistent buffers, two camera uniforms, and a single
//! host-count readback between assign and sort.
//!
//! ## The Y-flip fix (the load-bearing wiring detail)
//!
//! The two projection shaders disagree on screen-Y orientation:
//!   * `tile_assign.wgsl` projects with a Y-FLIP:
//!     `py = (1 - (ndc.y*0.5+0.5)) * vh`   (NDC +Y up → pixel +Y down)
//!   * `splat_raster.wgsl` projects with NO Y-flip:
//!     `screen = (ndc*0.5+0.5) * (w,h)`     (uses ndc.y directly)
//!
//! Left unreconciled, a splat is ASSIGNED to the vertically-mirrored tile
//! relative to where the raster EVALUATES it → coverage collapses to near zero.
//! The shaders are frozen, so the fix lives in the camera uniforms: the raster
//! camera's `view_proj` has its **second row negated** (`view_proj.y_row *= -1`),
//! which makes the raster see `ndc.y' = -ndc.y`, so its
//! `screen.y = (-ndc.y*0.5+0.5)*h = (1-(ndc.y*0.5+0.5))*h` — byte-for-byte the
//! same pixel row `tile_assign` used to pick the tile. The `tile_assign` uniform
//! keeps the canonical (un-negated) `view_proj`. This was confirmed empirically
//! in `tiled_y_orientation_matches_cpu`: the GPU lit region's centroid
//! co-locates with the CPU `spectra_render` reference (dy ≈ 4px / 256), which a
//! mirrored binding would not — coverage MAGNITUDE legitimately differs (the GPU
//! EWA footprint is tighter than the CPU path), so orientation, not area, is the
//! invariant asserted.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use vox_core::spectral::{Illuminant, SpectralBands, spectral_to_xyz, xyz_to_srgb};
use vox_core::types::GaussianSplat;

use crate::gpu::GpuContext;
use crate::gpu::gpu_timing::GpuTimers;
use crate::gpu::radix_sort_pass::{RadixSortPass, RadixSortPrepared};
use crate::gpu::splat_buffer::{
    GpuSplatFull, gaussian_splat_to_gpu_full, gaussian_splats_to_transforms,
};
use crate::gpu::splat_raster::{RasterParams, SplatRasterPass, SplatRasterPrepared};
use crate::gpu::tile_assign::{TileAssignBuffers, TileAssignPass};
use crate::gpu::tile_range_build::{TileRangeBuildPass, TileRangePrepared};
use crate::spectral::RenderCamera;

/// Tile edge length in pixels (must match `tile_assign.wgsl` / `splat_raster.wgsl`
/// and the CPU `spectra_render` `TILE_SIZE`).
const TILE_SIZE: u32 = 16;
/// Maximum tile entries emitted per splat by `tile_assign` (`MAX_TILES_PER_SPLAT`).
/// Solid city atoms can legitimately cover more than a 4x4 tile block when the
/// camera is close, so keep enough headroom to avoid chopped rectangular gaps.
const MAX_TILES_PER_SPLAT: u32 = 256;
/// Tile entries allocated PER SPLAT SLOT by [`TiledSplatRenderer::new_with_capacity`]
/// — budget-derived sizing instead of [`TiledSplatRenderer::new`]'s worst-case
/// `MAX_TILES_PER_SPLAT`. 16 entries/splat lets a 1M-splat budget fit the default
/// 128 MiB storage-binding limit (1M × 16 × 4 B = 64 MB per entry-class buffer);
/// overshoot is discarded by `render()`'s existing host clamp (plus naga's
/// bounds-checked storage writes inside the frozen `tile_assign` shader) and
/// reported per frame via [`TiledSplatRenderer::tile_entry_overflow`].
const ENTRY_HEADROOM: u32 = 16;

/// Error returned when the tiled renderer cannot be created or run. Never panics
/// on a missing/inadequate GPU — mirrors [`crate::gpu::splat_rt_gpu::SplatRtGpuError`].
#[derive(Debug, Clone)]
pub enum TiledRenderError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed.
    DeviceCreation(String),
    /// A required GPU resource would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`). Returned instead
    /// of letting wgpu raise an uncaptured Validation Error that aborts the
    /// process. `what` names the offending buffer, `requested` is the byte size
    /// we'd need, `limit` is the device's cap.
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for TiledRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TiledRenderError::NoAdapter => write!(f, "no GPU adapter available"),
            TiledRenderError::DeviceCreation(e) => write!(f, "GPU device creation failed: {e}"),
            TiledRenderError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => write!(
                f,
                "GPU resource '{what}' requires {requested} bytes, exceeding device limit {limit}"
            ),
        }
    }
}

impl std::error::Error for TiledRenderError {}

// ── Camera uniforms — replicated EXACTLY from the two frozen shaders ──────────

/// `tile_assign.wgsl` binding-0 `CameraUniform` (std140): two mat4, a vec4
/// viewport, then `tiles_xy: vec2<u32>`, `splat_count: u32`, `_pad: u32`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AssignCameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    /// xy = (width, height); zw = (1/width, 1/height).
    viewport_size: [f32; 4],
    tiles_xy: [u32; 2],
    splat_count: u32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<AssignCameraUniform>() == 64 + 64 + 16 + 16);

/// `splat_raster.wgsl` binding-0 `CameraUniform` (std140): three mat4, then
/// `viewport_size: vec2<f32>` + `_pad: vec2<f32>` (208 bytes total).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RasterCameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    viewport_size: [f32; 2],
    _pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<RasterCameraUniform>() == 64 * 3 + 16);

/// The output of one [`TiledSplatRenderer::render`] call.
pub struct TiledFrame {
    /// The reused 4-layer `rgba32float` texture array (layers 0–1 = 8 spectral
    /// bands, layer 3 = transmittance). Owned by the renderer and overwritten
    /// each frame; this is a clone of the handle (wgpu `Texture` is `Arc`-backed).
    pub spectral_texture: wgpu::Texture,
    /// Measured GPU milliseconds for the raster-dominated tail of the compute
    /// span, from the hardware `TIMESTAMP_QUERY` harness. `None` when the device
    /// lacks timestamps or the reading was sub-tick (caller falls back to
    /// [`Self::wall_ms`]).
    pub raster_gpu_ms: Option<f32>,
    /// Always-available wall-clock milliseconds of the blocking GPU span
    /// (encoder-B submit → `poll(Wait)`). Holds only GPU work, so it is a
    /// faithful upper-bound fallback when `raster_gpu_ms` is `None`.
    pub wall_ms: f32,
    /// Host wall-clock span of stage A (M3.1 Task 1): from just before the
    /// encoder-A submit (`tile_assign` + count copy) to after the — possibly
    /// grow-retried — entry-count readback returned. Honest caveat: the poll
    /// drains the whole queue, so this span contains the same-queue expand
    /// execution submitted just before render.
    pub assign_ms: f32,
    /// Host wall-clock span of the per-frame 4-layer output zero-fill.
    /// Always 0.0 since M3.1 Task 3 moved the fill to construction time
    /// (`alloc`): the per-frame fill was provably redundant —
    /// `splat_raster.wgsl:124–135` unconditionally `textureStore`s layers
    /// 0/1/3 for every in-bounds pixel every frame (the raster dispatch
    /// covers the full tile grid and `tile_range_build` re-clears all tile
    /// ranges), and layer 2 is written by no pass and read by no path
    /// (`resolve_to_srgb` reads layers 0–1 only). Kept as a field so the
    /// harness stage table keeps its pinned shape (`clear=0.00`).
    pub clear_ms: f32,
    width: u32,
    height: u32,
}

impl TiledFrame {
    /// PROOF-ONLY readback: copy the spectral texture (layers 0–1) to the host,
    /// reconstruct the 8-band SPD per pixel, run `spectral_to_xyz → xyz_to_srgb`,
    /// and return `(srgb_pixels, non_black_count)` where a pixel counts as
    /// non-black when any of r/g/b is non-zero.
    ///
    /// This is NOT on the render hot path — it exists to validate that the GPU
    /// chain actually drew something. The 8 GPU bands are pair bins covering the
    /// 16-band [`SpectralBands`]: bin 0 expands to bands 0 and 1, bin 1 to bands
    /// 2 and 3, and so on. This matches [`gaussian_splat_to_gpu_full`].
    pub fn resolve_to_srgb(
        &self,
        ctx: &GpuContext,
        illuminant: &Illuminant,
    ) -> (Vec<[u8; 4]>, usize) {
        let device = ctx.device();
        let queue = ctx.queue();
        let width = self.width;
        let height = self.height;
        let total = (width * height) as usize;

        // Row stride must be 256-byte aligned for copy_texture_to_buffer.
        // rgba32float = 16 bytes/texel.
        let unpadded_bpr = width * 16;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;

        // Read back layers 0 and 1 (the 8 spectral bands).
        let layer_bytes = (padded_bpr * height) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_resolve_readback"),
            size: layer_bytes * 2,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tiled_resolve_encoder"),
        });
        for layer in 0u32..2 {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.spectral_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: layer as u64 * layer_bytes,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if !matches!(rx.recv(), Ok(Ok(()))) {
            // Map failed: return all-black rather than panic (no-panic contract).
            return (vec![[0u8, 0, 0, 255]; total], 0);
        }

        let pixels;
        let non_black;
        {
            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            // Each layer is `height` rows of `padded_bpr/4` f32; the first
            // `width*4` f32 of each row are the live texels.
            let row_floats = (padded_bpr / 4) as usize;
            let layer0_off = 0usize;
            let layer1_off = (layer_bytes / 4) as usize;

            let mut out = Vec::with_capacity(total);
            let mut nb = 0usize;
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let b0 = layer0_off + y * row_floats + x * 4;
                    let b1 = layer1_off + y * row_floats + x * 4;
                    let mut spd = [0.0f32; 16];
                    let packed = [
                        floats[b0],
                        floats[b0 + 1],
                        floats[b0 + 2],
                        floats[b0 + 3],
                        floats[b1],
                        floats[b1 + 1],
                        floats[b1 + 2],
                        floats[b1 + 3],
                    ];
                    for (bin, &v) in packed.iter().enumerate() {
                        spd[bin * 2] = v;
                        spd[bin * 2 + 1] = v;
                    }

                    let xyz = spectral_to_xyz(&SpectralBands(spd), illuminant);
                    let rgb = xyz_to_srgb(xyz);
                    let r = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
                    if r != 0 || g != 0 || b != 0 {
                        nb += 1;
                    }
                    out.push([r, g, b, 255]);
                }
            }
            pixels = out;
            non_black = nb;
        }
        readback.unmap();
        (pixels, non_black)
    }
}

/// On-device tiled spectral splat renderer. Owns the shared [`GpuContext`], the
/// four frozen passes, and all persistent GPU resources; uploads the splat set
/// exactly once in [`Self::new`] and reuses everything across frames.
pub struct TiledSplatRenderer {
    ctx: GpuContext,

    tile_assign: TileAssignPass,
    radix: RadixSortPass,
    tile_range: TileRangeBuildPass,
    raster: SplatRasterPass,

    // Persistent inputs (written ONCE in `new`).
    splat_buf: wgpu::Buffer,
    transform_buf: wgpu::Buffer,

    // Persistent per-tile ranges (num_tiles * 8 bytes).
    tile_ranges: wgpu::Buffer,

    // Persistent sort scratch (max_tile_entries * 4 bytes each).
    tmp_lo: wgpu::Buffer,
    tmp_hi: wgpu::Buffer,
    tmp_vals: wgpu::Buffer,

    /// Persistent tile-assign entry scratch (M3.1 Task 2): the 3 entry-sized
    /// buffers + count + indirect args that `TileAssignPass::dispatch` used to
    /// allocate (and the next frame's `poll(Wait)` used to destroy) EVERY
    /// frame — 192–311 MB/frame of churn at city scale. Allocated once in
    /// [`Self::alloc`] at `max_tile_entries`; the grow block recreates it at
    /// the grown size alongside `tmp_lo/hi/vals`.
    assign_bufs: TileAssignBuffers,

    /// Cached per-frame encode handles (M3.1 Task 5): the tile-assign main +
    /// indirect bind groups and the prepared radix/range/raster
    /// params/bind-group handles. Built once at the end of [`Self::alloc`]
    /// and REBUILT in the grow block — every one of them binds the recreated
    /// `assign_bufs` entry buffers and/or the sort tmps. With these, the
    /// per-frame encode creates ZERO buffers, views, or bind groups.
    assign_bg: wgpu::BindGroup,
    assign_indirect_bg: wgpu::BindGroup,
    radix_prepared: RadixSortPrepared,
    range_prepared: TileRangePrepared,
    raster_prepared: SplatRasterPrepared,

    // Reused 4-layer rgba32float output.
    output_texture: wgpu::Texture,

    // Two camera uniforms (assign layout + raster layout).
    assign_camera_buf: wgpu::Buffer,
    raster_camera_buf: wgpu::Buffer,

    // 4-byte MAP_READ buffer for the single host-count readback.
    count_readback: wgpu::Buffer,

    timers: GpuTimers,

    width: u32,
    height: u32,
    splat_count: u32,
    /// Allocated splat slots: `splats.len()` for [`Self::new`], `max_splats` for
    /// [`Self::new_with_capacity`] — the clamp ceiling of
    /// [`Self::set_active_splat_count`].
    capacity: u32,
    /// Last frame's tile-entry overshoot (`count − max_tile_entries`, clamped
    /// at 0). See [`Self::tile_entry_overflow`].
    tile_entry_overflow: u32,
    /// Entry-scratch grow-and-retry enabled? `true` ONLY on the
    /// [`Self::new_with_capacity`] path — [`Self::new`] sizes its scratch at
    /// the worst-case `splat_count × MAX_TILES_PER_SPLAT`, which the frozen
    /// `tile_assign` shader can never exceed, so growth is unreachable there
    /// and the flag keeps `new`'s observable behavior byte-identical.
    grow_entries: bool,
    /// Cumulative grow events. See [`Self::tile_entry_regrows`].
    tile_entry_regrows: u32,
    /// The REAL (post-grow) tile-entry count the last [`Self::render`] read
    /// back from `tile_assign`'s atomic — debug/test visibility for the
    /// persistent-scratch reuse proof (M3.1 Task 2).
    last_entry_count: u32,
    /// Hard per-buffer ceiling for `max_tile_entries`, computed once from the
    /// captured device limits: `min(max_storage_buffer_binding_size,
    /// max_buffer_size) / 4` (4 B per entry). 33,554,432 entries under wgpu
    /// `Limits::default()` (128 MiB bindings).
    entry_ceiling: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_tiles: u32,
    max_tile_entries: u32,
}

/// Build the five cached per-frame encode handles (M3.1 Task 5): the
/// tile-assign main + indirect bind groups and the prepared
/// radix/range/raster handles. Shared by [`TiledSplatRenderer::alloc`]
/// (construction) and the in-`render` grow block (rebuild — the handles bind
/// the recreated `assign_bufs` entry buffers and sort tmps, so reuse after a
/// grow would encode against the DROPPED buffers).
#[allow(clippy::too_many_arguments)]
fn build_frame_handles(
    device: &wgpu::Device,
    tile_assign: &TileAssignPass,
    radix: &RadixSortPass,
    tile_range: &TileRangeBuildPass,
    raster: &SplatRasterPass,
    assign_camera_buf: &wgpu::Buffer,
    raster_camera_buf: &wgpu::Buffer,
    splat_buf: &wgpu::Buffer,
    transform_buf: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tmp_lo: &wgpu::Buffer,
    tmp_hi: &wgpu::Buffer,
    tmp_vals: &wgpu::Buffer,
    assign_bufs: &TileAssignBuffers,
    output_texture: &wgpu::Texture,
    width: u32,
    height: u32,
    tiles_x: u32,
) -> (
    wgpu::BindGroup,
    wgpu::BindGroup,
    RadixSortPrepared,
    TileRangePrepared,
    SplatRasterPrepared,
) {
    let assign_bg =
        tile_assign.create_bind_group(device, assign_camera_buf, splat_buf, transform_buf, assign_bufs);
    let assign_indirect_bg = tile_assign.create_indirect_bind_group(device, assign_bufs);
    let radix_prepared = radix.prepare(
        device,
        &assign_bufs.tile_keys_lo,
        &assign_bufs.tile_keys_hi,
        &assign_bufs.tile_vals,
        tmp_lo,
        tmp_hi,
        tmp_vals,
    );
    let range_prepared = tile_range.prepare(device, &assign_bufs.tile_keys_hi, tile_ranges);
    let raster_prepared = raster.prepare(
        device,
        raster_camera_buf,
        splat_buf,
        &assign_bufs.tile_vals,
        tile_ranges,
        output_texture,
        RasterParams {
            width,
            height,
            num_tiles_x: tiles_x,
            _pad: 0,
        },
    );
    (
        assign_bg,
        assign_indirect_bg,
        radix_prepared,
        range_prepared,
        raster_prepared,
    )
}

impl TiledSplatRenderer {
    /// Build the renderer over `splats`, sized for a `width × height` frame.
    ///
    /// Validates every persistent buffer against `max_storage_buffer_binding_size`
    /// AND `max_buffer_size` BEFORE allocating — returning
    /// [`TiledRenderError::ExceedsDeviceLimits`] instead of letting wgpu abort the
    /// process. Uploads `splat_buf` + `transform_buf` exactly once. The
    /// [`GpuTimers`] degrade to disabled (wall-clock fallback) when the borrowed
    /// device lacks `TIMESTAMP_QUERY`.
    pub fn new(
        ctx: GpuContext,
        splats: &[GaussianSplat],
        width: u32,
        height: u32,
    ) -> Result<Self, TiledRenderError> {
        let splat_count = splats.len() as u32;
        // tile_assign emits up to MAX_TILES_PER_SPLAT entries per splat.
        let max_tile_entries = splat_count.saturating_mul(MAX_TILES_PER_SPLAT).max(1);
        let mut this = Self::alloc(
            ctx,
            splat_count,
            max_tile_entries,
            width,
            height,
            wgpu::BufferUsages::empty(),
        )?;

        // ── Upload splats + transforms ONCE ─────────────────────────────────────
        if splat_count > 0 {
            let gpu_splats: Vec<GpuSplatFull> =
                splats.iter().map(gaussian_splat_to_gpu_full).collect();
            this.ctx
                .queue()
                .write_buffer(&this.splat_buf, 0, bytemuck::cast_slice(&gpu_splats));

            let transforms = gaussian_splats_to_transforms(splats);
            this.ctx
                .queue()
                .write_buffer(&this.transform_buf, 0, bytemuck::cast_slice(&transforms));
        }
        this.splat_count = splat_count;
        Ok(this)
    }

    /// Build the renderer with `max_splats` persistent splat slots and NO upload
    /// (virtualized splat rendering M1/M2): the input buffers start zeroed and
    /// `splat_count` starts 0, so a `render` before any upload draws nothing.
    /// Callers (the GPU expand pass, or host `write_buffer`) fill
    /// [`Self::splat_buf`] / [`Self::transform_buf`] and call
    /// [`Self::set_active_splat_count`] per frame — the renderer is constructed
    /// ONCE and never rebuilt.
    ///
    /// Entry scratch is budget-derived — `max_splats × ENTRY_HEADROOM (16)`
    /// entries instead of `new`'s worst-case `MAX_TILES_PER_SPLAT` (256) — which
    /// is what lets six-figure capacities fit `Limits::default()`. When a frame's
    /// real demand overshoots this scratch, `render()` GROWS it to
    /// `demand.max(2×current)` (clamped at the device ceiling, ~33.55M entries
    /// under `Limits::default()`) and retries the idempotent `tile_assign`
    /// in-frame — see [`Self::tile_entry_regrows`]. Honest memory bound at the
    /// ceiling: 6 entry-class buffers (3 per-frame assign + 3 persistent sort
    /// tmps) × ≤128 MiB ≈ 768 MiB. Only past the ceiling does the old behavior
    /// remain: host clamp + a nonzero [`Self::tile_entry_overflow`]. Both input
    /// buffers additionally carry `COPY_SRC` so proof readbacks can inspect what
    /// the expand pass wrote. Validated against
    /// `max_storage_buffer_binding_size` / `max_buffer_size` exactly like
    /// [`Self::new`], returning [`TiledRenderError::ExceedsDeviceLimits`]
    /// instead of aborting.
    pub fn new_with_capacity(
        ctx: GpuContext,
        max_splats: u32,
        width: u32,
        height: u32,
    ) -> Result<Self, TiledRenderError> {
        let max_tile_entries = max_splats.saturating_mul(ENTRY_HEADROOM).max(1);
        let mut this = Self::alloc(
            ctx,
            max_splats,
            max_tile_entries,
            width,
            height,
            wgpu::BufferUsages::COPY_SRC,
        )?;
        // Budget-derived scratch can legitimately undershoot real demand
        // (measured: 25,857,730 entries vs a 16M scratch at 1k buildings) —
        // enable the in-frame grow-and-retry ONLY on this path.
        this.grow_entries = true;
        Ok(this)
    }

    /// Shared allocation/validation core of [`Self::new`] and
    /// [`Self::new_with_capacity`]: validates every persistent buffer against
    /// `max_storage_buffer_binding_size` AND `max_buffer_size` BEFORE
    /// allocating, then builds the four frozen passes and all persistent
    /// resources. Performs NO upload and leaves `splat_count` at 0 (callers set
    /// it). `extra_input_usage` is OR-ed into the splat/transform buffer usages:
    /// `new` passes `empty()` (its buffers stay byte-identical to before this
    /// refactor), the capacity path passes `COPY_SRC`.
    fn alloc(
        ctx: GpuContext,
        capacity: u32,
        max_tile_entries: u32,
        width: u32,
        height: u32,
        extra_input_usage: wgpu::BufferUsages,
    ) -> Result<Self, TiledRenderError> {
        let device = ctx.device().clone();
        let queue = ctx.queue().clone();

        let width = width.max(1);
        let height = height.max(1);

        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        let num_tiles = tiles_x * tiles_y;

        // ── Validate sizes against device limits BEFORE allocating ────────────
        let limits = device.limits();
        let max_storage = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        // Hard ceiling for any later entry-scratch growth (4 B per entry):
        // 134,217,728 B / 4 = 33,554,432 entries under `Limits::default()`.
        let entry_ceiling = (max_storage.min(max_buffer) / 4).min(u32::MAX as u64) as u32;

        let splat_bytes = (capacity as u64).max(1) * std::mem::size_of::<GpuSplatFull>() as u64;
        let transform_bytes = (capacity as u64).max(1) * 2 * 16; // two vec4 per splat
        let ranges_bytes = num_tiles as u64 * 8;
        let entries_bytes = max_tile_entries as u64 * 4;

        for (what, bytes) in [
            ("splat_buf", splat_bytes),
            ("transform_buf", transform_bytes),
            ("tile_ranges", ranges_bytes),
            ("tile_entries", entries_bytes),
        ] {
            if bytes > max_storage {
                return Err(TiledRenderError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage,
                });
            }
            if bytes > max_buffer {
                return Err(TiledRenderError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        // ── Passes ────────────────────────────────────────────────────────────
        let tile_assign = TileAssignPass::new(&device);
        let radix = RadixSortPass::new(&device);
        let tile_range = TileRangeBuildPass::new(&device);
        let raster = SplatRasterPass::new(&device);

        // ── Persistent buffers ─────────────────────────────────────────────────
        let splat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_splat_buf"),
            // tile_assign binds this read_write; needs STORAGE. Min 80 to never bind 0.
            size: splat_bytes.max(80),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_input_usage,
            mapped_at_creation: false,
        });
        let transform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_transform_buf"),
            size: transform_bytes.max(32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_input_usage,
            mapped_at_creation: false,
        });
        let tile_ranges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_tile_ranges"),
            size: ranges_bytes.max(8),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tmp_lo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_sort_tmp_lo"),
            size: entries_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tmp_hi = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_sort_tmp_hi"),
            size: entries_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tmp_vals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_sort_tmp_vals"),
            size: entries_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Persistent tile-assign entry scratch (M3.1 Task 2) — allocated ONCE
        // here, reused by every `dispatch_into` in `render()`. Size validated
        // above ("tile_entries"). wgpu zero-initializes it on first use; after
        // that the per-encode `tile_count` clear is the only re-arming needed.
        let assign_bufs = TileAssignBuffers::allocate(&device, max_tile_entries);
        let assign_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_assign_camera"),
            size: std::mem::size_of::<AssignCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_raster_camera"),
            size: std::mem::size_of::<RasterCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_count_readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Reused output texture (4 layers). COPY_DST for the ONE-TIME zero-fill
        // below; COPY_SRC so the proof readback can read it.
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tiled_output_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // ONE-TIME 4-layer zero-fill (M3.1 Task 3) — this used to run EVERY
        // frame in `render()` (4 × queue.write_texture of a padded zero row:
        // 59 MB/frame of staging at 1280×720). It is needed at most once:
        // every frame the raster dispatch covers the full tile grid and the
        // frozen `splat_raster.wgsl:124–135` unconditionally `textureStore`s
        // layers 0/1/3 for every in-bounds pixel (zeros where nothing
        // accumulated, since `tile_range_build` re-clears all tile ranges each
        // frame), and layer 2 is written by no pass and read by no path
        // (`resolve_to_srgb` reads layers 0–1 only). Guarded by
        // `tiled_no_stale_pixels_without_per_frame_fill`.
        let padded_bpr = (width * 16).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let zero_row = vec![0u8; (padded_bpr * height) as usize];
        for layer in 0u32..4 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &zero_row,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // ── Cached per-frame encode handles (M3.1 Task 5) ────────────────────
        let (assign_bg, assign_indirect_bg, radix_prepared, range_prepared, raster_prepared) =
            build_frame_handles(
                &device,
                &tile_assign,
                &radix,
                &tile_range,
                &raster,
                &assign_camera_buf,
                &raster_camera_buf,
                &splat_buf,
                &transform_buf,
                &tile_ranges,
                &tmp_lo,
                &tmp_hi,
                &tmp_vals,
                &assign_bufs,
                &output_texture,
                width,
                height,
                tiles_x,
            );

        // ── GPU timers: 1 pair, slot 0 brackets the whole compute span ──────────
        let timers = GpuTimers::new(&device, &queue, device.features(), 1);

        Ok(Self {
            ctx,
            tile_assign,
            radix,
            tile_range,
            raster,
            splat_buf,
            transform_buf,
            tile_ranges,
            tmp_lo,
            tmp_hi,
            tmp_vals,
            assign_bufs,
            assign_bg,
            assign_indirect_bg,
            radix_prepared,
            range_prepared,
            raster_prepared,
            output_texture,
            assign_camera_buf,
            raster_camera_buf,
            count_readback,
            timers,
            width,
            height,
            splat_count: 0,
            capacity,
            tile_entry_overflow: 0,
            grow_entries: false,
            tile_entry_regrows: 0,
            last_entry_count: 0,
            entry_ceiling,
            tiles_x,
            tiles_y,
            num_tiles,
            max_tile_entries,
        })
    }

    /// Adapter name backing the shared device (diagnostics).
    pub fn adapter_name(&self) -> &str {
        self.ctx.adapter_name()
    }

    /// The persistent on-device splat buffer (`array<GpuSplatFull>`, 80 B/splat).
    /// Exposed so the resident GI→raster path ([`crate::gpu::gi_combine`]) can
    /// fold GI radiance into `spectral[0..8]` in place before [`Self::render`]
    /// reads it — additive accessor, no change to existing logic.
    pub fn splat_buf(&self) -> &wgpu::Buffer {
        &self.splat_buf
    }

    /// The persistent on-device transform buffer (two vec4 per splat:
    /// `[scale.xyz, 0]` then `[quat x,y,z,w]`, matching
    /// [`gaussian_splats_to_transforms`]). Exposed for the same reason as
    /// [`Self::splat_buf`]: the GPU expand pass writes instance-transformed
    /// atoms directly into it before [`Self::render`] reads it — additive
    /// accessor, no change to existing logic.
    pub fn transform_buf(&self) -> &wgpu::Buffer {
        &self.transform_buf
    }

    /// Set how many leading splat slots the next [`Self::render`] draws, clamped
    /// to the allocated capacity (never panics). This is just a field write:
    /// `render()` already rebuilds the assign camera uniform (which carries
    /// `splat_count`) and sizes the assign dispatch from this field every
    /// frame, so no extra GPU work happens here.
    pub fn set_active_splat_count(&mut self, n: u32) {
        self.splat_count = n.min(self.capacity);
    }

    /// How many tile entries the LAST [`Self::render`] overshot the allocated
    /// entry scratch by (`count.saturating_sub(max_tile_entries)`; 0 = no
    /// overflow). Overshot entries were discarded — by naga's bounds-checked
    /// storage writes inside the frozen `tile_assign` shader and by `render()`'s
    /// existing host clamp — so a nonzero value means the frame dropped tile
    /// work; this is the honest counter a budget controller (M3) watches.
    pub fn tile_entry_overflow(&self) -> u32 {
        self.tile_entry_overflow
    }

    /// Current tile-entry scratch capacity (entries, 4 B each). Starts at the
    /// constructor-derived size (`splats × MAX_TILES_PER_SPLAT` for
    /// [`Self::new`], `max_splats × ENTRY_HEADROOM` for
    /// [`Self::new_with_capacity`]) and only ever grows — see
    /// [`Self::tile_entry_regrows`].
    pub fn tile_entry_capacity(&self) -> u32 {
        self.max_tile_entries
    }

    /// Cumulative count of in-frame entry-scratch grow events (capacity path
    /// only; always 0 for [`Self::new`]). Each event re-sized the scratch to
    /// `demand.max(2×previous)` clamped at the device ceiling and re-ran the
    /// idempotent `tile_assign` in the same frame, so the frame that grew
    /// still drew complete tile work. A steadily climbing value under a
    /// changing camera is normal; growth stops once the scratch fits the
    /// scene's worst frame.
    pub fn tile_entry_regrows(&self) -> u32 {
        self.tile_entry_regrows
    }

    /// The REAL tile-entry count the last [`Self::render`] read back from the
    /// `tile_assign` atomic (the post-grow value when a grow-retry happened).
    /// Debug/test accessor backing the M3.1 Task-2 persistent-scratch reuse
    /// proof: two renders over different splat sets must report two different
    /// nonzero counts through the ONE `assign_bufs` allocation.
    pub(crate) fn last_entry_count(&self) -> u32 {
        self.last_entry_count
    }

    /// Are GPU timestamps live (vs. wall-clock fallback)?
    pub fn timers_enabled(&self) -> bool {
        self.timers.is_enabled()
    }

    /// Build the `tile_assign` camera uniform with the CANONICAL (un-negated)
    /// `view_proj`. This is the projection that decides which tile a splat lands
    /// in (with the shader's own Y-flip applied internally).
    fn assign_camera(&self, camera: &RenderCamera) -> AssignCameraUniform {
        let view_proj = camera.view_proj();
        AssignCameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            viewport_size: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            tiles_xy: [self.tiles_x, self.tiles_y],
            splat_count: self.splat_count,
            _pad: 0,
        }
    }

    /// Build the `splat_raster` camera uniform with the Y-NEGATED `view_proj`
    /// (second row of the column-major matrix flipped). See the module doc: this
    /// is the empirically-determined fix that makes the no-Y-flip raster shader
    /// evaluate each splat at the SAME pixel row `tile_assign` used to bin it.
    fn raster_camera(&self, camera: &RenderCamera) -> RasterCameraUniform {
        let mut view_proj = camera.view_proj();
        // Negate the Y output row of the clip-space transform: row 1 of the
        // mathematical matrix = element [.][1] of each column (column-major glam).
        // Flipping it sends ndc.y → -ndc.y in the raster, cancelling the
        // tile_assign Y-flip so both passes agree on screen-Y.
        let mut cols = view_proj.to_cols_array();
        cols[1] = -cols[1]; // col0.y
        cols[5] = -cols[5]; // col1.y
        cols[9] = -cols[9]; // col2.y
        cols[13] = -cols[13]; // col3.y
        view_proj = Mat4::from_cols_array(&cols);

        let inv_view = camera.view.inverse();
        RasterCameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            inv_view: inv_view.to_cols_array_2d(),
            viewport_size: [self.width as f32, self.height as f32],
            _pad: [0.0, 0.0],
        }
    }

    /// ONE blocking `poll(Wait)` readback of the real tile-entry count that the
    /// just-submitted `tile_assign` accumulated in its `tile_count` atomic (the
    /// TRUE demand — the frozen shader counts every desired entry even when the
    /// bounds-checked store discarded it). `None` when the map fails — callers
    /// treat that as zero entries (black frame, no panic).
    fn read_entry_count(&self, device: &wgpu::Device) -> Option<u32> {
        let slice = self.count_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return None;
        }
        let data = slice.get_mapped_range();
        let v = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.count_readback.unmap();
        Some(v)
    }

    /// Run the full tiled chain for `camera` and return a [`TiledFrame`].
    ///
    /// Stage layout:
    ///   A) `tile_assign` into the PERSISTENT `assign_bufs` (allocated once in
    ///      `alloc`, recreated only on entry-scratch growth — M3.1 Task 2) +
    ///      copy `tile_count` to a MAP_READ buffer; submit; ONE blocking poll
    ///      to read the real entry count `n` (the only host readback between
    ///      stages).
    ///   B) timed (slot 0): `radix_sort(n)` → `tile_range_build(n)` →
    ///      `splat_raster`; resolve timers; submit. `raster_gpu_ms` is read
    ///      back after. (The output clear is NOT in the frame — M3.1 Task 3:
    ///      the texture is zero-filled once in `alloc`; the raster overwrites
    ///      layers 0/1/3 full-grid every frame and layer 2 is never read.)
    pub fn render(&mut self, camera: &RenderCamera) -> Result<TiledFrame, TiledRenderError> {
        let device = self.ctx.device().clone();
        let queue = self.ctx.queue().clone();

        // (a) Write BOTH camera uniforms.
        let assign_cam = self.assign_camera(camera);
        let raster_cam = self.raster_camera(camera);
        queue.write_buffer(&self.assign_camera_buf, 0, bytemuck::bytes_of(&assign_cam));
        queue.write_buffer(&self.raster_camera_buf, 0, bytemuck::bytes_of(&raster_cam));

        // (b) Encoder A: tile_assign into the PERSISTENT `assign_bufs` (M3.1
        // Task 2 — no per-frame entry-buffer allocation) through the CACHED
        // bind groups (M3.1 Task 5 — no per-frame bind-group creation); copy
        // tile_count to readback.
        let mut enc_a = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tiled_encoder_a"),
        });
        self.tile_assign.dispatch_bound(
            &mut enc_a,
            self.splat_count,
            &self.assign_bg,
            &self.assign_indirect_bg,
            &self.assign_bufs,
        );
        enc_a.copy_buffer_to_buffer(&self.assign_bufs.tile_count, 0, &self.count_readback, 0, 4);
        // Stage-A span (M3.1 Task 1): submit → entry-count read complete,
        // INCLUDING any grow-retry below. The poll drains the whole queue, so
        // this span contains the same-queue expand execution submitted just
        // before render — see the `TiledFrame::assign_ms` rustdoc.
        let t_assign = std::time::Instant::now();
        queue.submit(Some(enc_a.finish()));

        // ONE poll(Wait) to read the real entry count (readback failure = zero
        // entries: frame stays black, no panic — same contract as before).
        let mut v = self.read_entry_count(&device).unwrap_or(0);

        // ── M3 grow-and-retry: self-heal a budget-derived scratch undershoot ──
        // Capacity path ONLY (`grow_entries`): `new` sizes the scratch at the
        // worst case its own `splat_count.saturating_mul(MAX_TILES_PER_SPLAT)`
        // line allows, so `v > max_tile_entries` is unreachable there and this
        // block can never change `new`'s observable behavior. `tile_assign` is
        // idempotent (it recomputes per-splat depth/conic from the unchanged
        // splat/transform/camera inputs and zero-clears its own `tile_count`),
        // so the retry needs only the re-sized buffers: our three persistent
        // sort tmps AND (since M3.1 Task 2 made them persistent too) the
        // `assign_bufs` entry scratch, recreated at `grown` BEFORE the retry.
        // The on-grow memory ceiling math still holds: 6 entry-class buffers
        // (3 in `assign_bufs` + 3 sort tmps) × ≤128 MiB ≈ 768 MiB.
        if self.grow_entries && v > self.max_tile_entries {
            let grown = v
                .max(self.max_tile_entries.saturating_mul(2))
                .min(self.entry_ceiling);
            if grown > self.max_tile_entries {
                let entries_bytes = grown as u64 * 4;
                for (buf, label) in [
                    (&mut self.tmp_lo, "tiled_sort_tmp_lo"),
                    (&mut self.tmp_hi, "tiled_sort_tmp_hi"),
                    (&mut self.tmp_vals, "tiled_sort_tmp_vals"),
                ] {
                    *buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: entries_bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                }
                self.assign_bufs = TileAssignBuffers::allocate(&device, grown);
                self.max_tile_entries = grown;
                self.tile_entry_regrows += 1;

                // REBUILD the cached encode handles (M3.1 Task 5): every one
                // of them binds the just-dropped entry buffers / sort tmps.
                let (assign_bg, assign_indirect_bg, radix_prepared, range_prepared, raster_prepared) =
                    build_frame_handles(
                        &device,
                        &self.tile_assign,
                        &self.radix,
                        &self.tile_range,
                        &self.raster,
                        &self.assign_camera_buf,
                        &self.raster_camera_buf,
                        &self.splat_buf,
                        &self.transform_buf,
                        &self.tile_ranges,
                        &self.tmp_lo,
                        &self.tmp_hi,
                        &self.tmp_vals,
                        &self.assign_bufs,
                        &self.output_texture,
                        self.width,
                        self.height,
                        self.tiles_x,
                    );
                self.assign_bg = assign_bg;
                self.assign_indirect_bg = assign_indirect_bg;
                self.radix_prepared = radix_prepared;
                self.range_prepared = range_prepared;
                self.raster_prepared = raster_prepared;

                // RETRY ONCE in-frame: re-encode the idempotent tile_assign at
                // the grown capacity + the count copy, submit, re-read.
                let mut enc_retry =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("tiled_encoder_a_retry"),
                    });
                self.tile_assign.dispatch_bound(
                    &mut enc_retry,
                    self.splat_count,
                    &self.assign_bg,
                    &self.assign_indirect_bg,
                    &self.assign_bufs,
                );
                enc_retry.copy_buffer_to_buffer(
                    &self.assign_bufs.tile_count,
                    0,
                    &self.count_readback,
                    0,
                    4,
                );
                queue.submit(Some(enc_retry.finish()));
                v = self.read_entry_count(&device).unwrap_or(0);
            }
            // At the ceiling (`grown == max_tile_entries`) growth is impossible:
            // fall through to the honest clamp + overflow report below.
        }
        let assign_ms = t_assign.elapsed().as_secs_f64() as f32 * 1.0e3;
        // The real (post-grow) demand this frame — see `last_entry_count()`.
        self.last_entry_count = v;

        // Record how far past the allocated entries this frame went (0 in the
        // worst-case-sized `new` path) — see `tile_entry_overflow()`.
        self.tile_entry_overflow = v.saturating_sub(self.max_tile_entries);
        // Defensive clamp: never sort/range-build past the allocated entries.
        let n = v.min(self.max_tile_entries);

        // NO per-frame output clear (M3.1 Task 3): the 4-layer zero-fill that
        // lived here (59 MB/frame of `queue.write_texture` staging at 720p)
        // moved to construction time in `alloc` — the frozen raster
        // (`splat_raster.wgsl:124–135`) unconditionally stores layers 0/1/3
        // for every in-bounds pixel every frame, and layer 2 is never read,
        // so no stale pixel can survive. `clear_ms` reports 0.0 from here on.

        // (c) Encoder B: the compute span (sort → ranges → raster).
        //
        // GPU TIMING NOTE — why slot-0 wraps ONLY the raster dispatch:
        // `GpuTimers` measures one begin→end pair per *compute pass*. The radix
        // sort and tile_range_build each open their OWN (frozen, untimed) passes,
        // so a single timestamp pair cannot straddle the whole chain. The
        // dominant on-GPU cost at this scale is the raster, so we attach slot 0
        // to a measured raster pass: we open the timed pass OURSELVES (carrying
        // the timestamp writes) and the raster records its dispatch into the same
        // encoder right after. wgpu serializes passes in submission order, so the
        // begin marker precedes and the end marker follows the raster work that
        // dominates the span. When timestamps are unavailable the writes are
        // `None` and the caller falls back to wall-clock. The honest caveat: this
        // measures the raster-dominated tail of the chain, not the sort/ranges
        // prologue — documented here rather than faked into a whole-chain number.
        let mut enc_b = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tiled_encoder_b"),
        });

        // radix sort the persistent tile_assign buffers in place (8 passes)
        // through the PREPARED handles (M3.1 Task 5): per-frame host work is
        // 8 × 16 B `queue.write_buffer` instead of 8 × (params staging
        // `create_buffer_init` + fresh 9-entry bind group). Bit-exact with
        // `sort` — proven by `radix_prepared_matches_sort_bitexact`.
        self.radix
            .sort_prepared(&queue, &mut enc_b, n, &self.radix_prepared);

        // Build per-tile [start,end) ranges from the sorted hi keys (tile ids),
        // prepared handles likewise (one 16 B params write per frame).
        self.tile_range
            .dispatch_prepared(&queue, &mut enc_b, n, self.num_tiles, &self.range_prepared);

        // NOTE on GPU timing: pass-boundary TIMESTAMP_QUERY can only bracket a
        // single compute pass's begin→end, but each stage here (radix_sort,
        // tile_range_build, splat_raster) opens its OWN pass with
        // `timestamp_writes: None` — and those shaders/passes are frozen. So the
        // chain cannot be hardware-timestamped without restructuring (Spec 11
        // does exactly that via encoder-level timestamps / a resident loop).
        // Until then `raster_gpu_ms` is honestly `None` and the wall-clock of the
        // GPU-only `enc_b` submit→poll below is the faithful chain measure.
        // Prepared raster (M3.1 Task 5): params/view/bind group were built in
        // `alloc` (raster params are static per renderer — width/height/tiles
        // never change), so the per-frame cost is exactly one compute pass.
        // `tiles_y == height.div_ceil(16)` — the same dispatch grid `dispatch`
        // computed from `params.height`.
        self.raster
            .dispatch_prepared(&mut enc_b, self.tiles_x, self.tiles_y, &self.raster_prepared);

        // Wall-clock the blocking GPU span: this encoder holds ONLY GPU work and
        // we poll to completion, so the delta is a faithful (submit-overhead-
        // inflated) measure of the whole chain's GPU time.
        let t_gpu = std::time::Instant::now();
        queue.submit(Some(enc_b.finish()));
        device.poll(wgpu::Maintain::Wait);
        let wall_ms = t_gpu.elapsed().as_secs_f64() as f32 * 1.0e3;

        // Honest: the chain's frozen passes can't be pass-boundary-timestamped
        // (see the NOTE above), so there is no hardware GPU number for this slice;
        // the caller reports `wall_ms` labelled "wall". Spec 11 restructures to a
        // resident loop where a real GPU timestamp becomes available.
        let raster_gpu_ms: Option<f32> = None;

        Ok(TiledFrame {
            spectral_texture: self.output_texture.clone(),
            raster_gpu_ms,
            wall_ms,
            assign_ms,
            clear_ms: 0.0,
            width: self.width,
            height: self.height,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// In-module tests — the capacity-API contract (virtualized splat rendering M1/M2
// plan, Task 3). The frozen four-pass chain itself is validated by
// `tests/tiled_splat_renderer_test.rs`; these tests prove ONLY the additive
// capacity surface: construct once, upload twice, render twice, persistent
// buffers untouched outside the active range.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use half::f16;

    const W: u32 = 320;
    const H: u32 = 180;

    /// Build a headless [`GpuContext`] over the local hardware GPU, mirroring
    /// the house `try_gpu_context` pattern (`spectral_gi.rs`): print the skip
    /// line and return `None` on no adapter / a software rasteriser so a
    /// GPU-less box stays green.
    fn try_gpu_context(label: &str) -> Option<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("[{label}] no adapter — skipping GPU test");
                return None;
            }
        };
        let info = adapter.get_info();
        if crate::gpu::adapter::ensure_hardware(&info).is_err() {
            eprintln!(
                "[{label}] software adapter ({}) — skipping GPU test",
                info.name
            );
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tiled_capacity_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(GpuContext::from_parts(&device, &queue, &info))
    }

    /// A deterministic `nx × ny` block of opaque overlapping splats spanning
    /// `x ∈ [cx−ex, cx+ex]`, `y ∈ [cy−ey, cy+ey]` at depth `z` — solid
    /// coverage, no RNG.
    fn splat_block(
        nx: i32,
        ny: i32,
        cx: f32,
        cy: f32,
        z: f32,
        ex: f32,
        ey: f32,
    ) -> Vec<GaussianSplat> {
        let spd: [u16; 16] = std::array::from_fn(|_| f16::from_f32(0.9).to_bits());
        let mut v = Vec::with_capacity((nx * ny) as usize);
        for gy in 0..ny {
            for gx in 0..nx {
                let x = cx - ex + 2.0 * ex * gx as f32 / (nx - 1).max(1) as f32;
                let y = cy - ey + 2.0 * ey * gy as f32 / (ny - 1).max(1) as f32;
                v.push(GaussianSplat::volume(
                    [x, y, z],
                    [0.2, 0.2, 0.2],
                    Quat::IDENTITY,
                    255,
                    spd,
                ));
            }
        }
        v
    }

    fn camera_looking_at(eye: Vec3, target: Vec3) -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_3, // 60°
                W as f32 / H as f32,
                0.1,
                200.0,
            ),
        }
    }

    /// Host-side upload of a splat set into the renderer's persistent buffers —
    /// the stand-in for the GPU expand pass (Task 4), per the plan's Task 3
    /// test recipe.
    fn upload(ctx: &GpuContext, r: &TiledSplatRenderer, splats: &[GaussianSplat]) {
        let gpu: Vec<GpuSplatFull> = splats.iter().map(gaussian_splat_to_gpu_full).collect();
        ctx.queue()
            .write_buffer(r.splat_buf(), 0, bytemuck::cast_slice(&gpu));
        let transforms = gaussian_splats_to_transforms(splats);
        ctx.queue()
            .write_buffer(r.transform_buf(), 0, bytemuck::cast_slice(&transforms));
    }

    /// The no-rebuild contract: ONE `new_with_capacity` renderer serves two
    /// different splat sets from two different cameras, and a sentinel bit
    /// pattern written into the LAST capacity slot before frame 1 survives both
    /// frames bit-exactly — proving the SAME persistent `splat_buf` carried both
    /// frames and nothing outside the active range was touched.
    #[test]
    fn tiled_capacity_two_frames_one_construction_sentinel_survives() {
        let Some(ctx) = try_gpu_context("tiled_capacity") else {
            return;
        };

        const CAPACITY: u32 = 200_000;
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), CAPACITY, W, H)
            .expect("new_with_capacity(200k, 320x180) fits default device limits");

        // Sentinel: a known 80-byte bit pattern in the LAST capacity slot,
        // written BEFORE frame 1 and read back AFTER frame 2. tile_assign only
        // writes slots < the active splat count, so a rebuilt-or-clobbered
        // buffer is the only thing that can change it.
        let sentinel: [f32; 20] = std::array::from_fn(|i| 1000.5 + 7.25 * i as f32);
        let slot_bytes = std::mem::size_of::<GpuSplatFull>() as u64;
        let tail_off = (CAPACITY as u64 - 1) * slot_bytes;
        ctx.queue().write_buffer(
            renderer.splat_buf(),
            tail_off,
            bytemuck::cast_slice(&sentinel),
        );

        // Frame 1: a 30×20 block around (-2, 2.5, 0), close camera.
        let set1 = splat_block(30, 20, -2.0, 2.5, 0.0, 2.0, 1.5);
        upload(&ctx, &renderer, &set1);
        renderer.set_active_splat_count(set1.len() as u32);
        let cam1 = camera_looking_at(Vec3::new(-2.0, 2.5, 5.0), Vec3::new(-2.0, 2.5, 0.0));
        let frame1 = renderer.render(&cam1).expect("frame 1");
        let (_, nb1) = frame1.resolve_to_srgb(&ctx, &Illuminant::d65());
        let overflow1 = renderer.tile_entry_overflow();

        // Frame 2: a DISJOINT, larger 56×32 block around (14, 1, -3), farther
        // camera — a genuinely different frame with a different coverage count.
        let set2 = splat_block(56, 32, 14.0, 1.0, -3.0, 4.5, 2.5);
        assert_ne!(set1.len(), set2.len());
        upload(&ctx, &renderer, &set2);
        renderer.set_active_splat_count(set2.len() as u32);
        let cam2 = camera_looking_at(Vec3::new(14.0, 1.0, 9.0), Vec3::new(14.0, 1.0, -3.0));
        let frame2 = renderer.render(&cam2).expect("frame 2");
        let (_, nb2) = frame2.resolve_to_srgb(&ctx, &Illuminant::d65());
        let overflow2 = renderer.tile_entry_overflow();

        // Read the tail slot back and demand the EXACT bit pattern.
        let device = ctx.device();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_capacity_sentinel_readback"),
            size: slot_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tiled_capacity_sentinel_copy"),
        });
        enc.copy_buffer_to_buffer(renderer.splat_buf(), tail_off, &readback, 0, slot_bytes);
        ctx.queue().submit(Some(enc.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        assert!(
            matches!(rx.recv(), Ok(Ok(()))),
            "sentinel readback map failed"
        );
        let intact = {
            let data = slice.get_mapped_range();
            let got: &[f32] = bytemuck::cast_slice(&data);
            got.len() == sentinel.len()
                && got
                    .iter()
                    .zip(sentinel.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        readback.unmap();

        // Real computed outcomes: both frames lit, differently; sentinel
        // bit-identical; zero entry overflow at this scale.
        assert!(nb1 > 0, "frame 1 must light pixels (got {nb1})");
        assert!(nb2 > 0, "frame 2 must light pixels (got {nb2})");
        assert_ne!(
            nb1, nb2,
            "two different scenes/cameras must produce different coverage"
        );
        assert!(
            intact,
            "tail-slot sentinel must survive both frames bit-exactly"
        );
        let overflow = overflow1.max(overflow2);
        assert_eq!(
            overflow, 0,
            "no tile-entry overflow at this scale (f1={overflow1} f2={overflow2})"
        );

        println!(
            "[tiled_capacity] capacity={CAPACITY} frame1 non_black={nb1} frame2 non_black={nb2} sentinel=intact overflow={overflow}"
        );
    }

    /// M3 Task 1 — the measured capacity bug (9,857,730 dropped entries at
    /// 1,000 buildings) self-heals: a `new_with_capacity` renderer whose real
    /// tile-entry demand overflows the budget-derived scratch must GROW the
    /// scratch to `demand.max(2×capacity)` (clamped at the device ceiling) and
    /// re-run the idempotent `tile_assign` in the SAME frame, then never regrow
    /// again once sized for the scene.
    #[test]
    fn tiled_entry_growth_self_heals() {
        let Some(ctx) = try_gpu_context("tiled_entry") else {
            return;
        };

        // 2,048 splat slots → initial entry scratch 2,048 × ENTRY_HEADROOM(16)
        // = 32,768 entries.
        const CAPACITY: u32 = 2_048;
        const INITIAL_ENTRIES: u32 = CAPACITY * ENTRY_HEADROOM;
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), CAPACITY, W, H)
            .expect("new_with_capacity(2048, 320x180) fits default device limits");
        assert_eq!(
            renderer.tile_entry_capacity(),
            INITIAL_ENTRIES,
            "initial entry capacity must be the budget-derived 32,768"
        );
        assert_eq!(renderer.tile_entry_regrows(), 0, "no regrow before render");

        // 2,000 LARGE opaque splats 1–3 m in front of the camera: each projects
        // far past the 320×180 screen, so its tile bbox clamps to the full
        // 20×12 = 240-tile grid (< MAX_TILES_PER_SPLAT 256) → real demand
        // ≈ 2,000 × 240 = 480,000 entries ≫ the 32,768 scratch.
        let spd: [u16; 16] = std::array::from_fn(|_| f16::from_f32(0.9).to_bits());
        let splats: Vec<GaussianSplat> = (0..2000)
            .map(|i| {
                let t = i as f32 / 1999.0;
                GaussianSplat::volume(
                    [
                        -0.5 + (i % 17) as f32 / 16.0,
                        -0.5 + (i % 13) as f32 / 12.0,
                        -(1.0 + 2.0 * t),
                    ],
                    [2.0, 2.0, 2.0],
                    Quat::IDENTITY,
                    255,
                    spd,
                )
            })
            .collect();
        upload(&ctx, &renderer, &splats);
        renderer.set_active_splat_count(splats.len() as u32);

        let cam = camera_looking_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -2.0));
        let frame = renderer.render(&cam).expect("frame 1 (the growth frame)");
        let (_, nb) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());

        let grown = renderer.tile_entry_capacity();
        assert_eq!(
            renderer.tile_entry_regrows(),
            1,
            "the overflowing first frame must grow the entry scratch exactly once"
        );
        assert_eq!(
            renderer.tile_entry_overflow(),
            0,
            "after the in-frame grow-and-retry no tile entries may be dropped"
        );
        assert!(
            grown > INITIAL_ENTRIES,
            "entry capacity must exceed the initial 32,768 (got {grown})"
        );
        // grown = demand.max(2 × 32,768); full-screen splats put the demand far
        // beyond 65,536, so the settled capacity IS the real readback demand.
        assert!(
            grown > 2 * INITIAL_ENTRIES,
            "this scene's demand must exceed even the doubled scratch (got {grown})"
        );
        assert!(nb > 0, "the grown frame must light pixels (got {nb})");

        // Second render, same camera/scene: already sized — must NOT regrow.
        let _frame2 = renderer.render(&cam).expect("frame 2 (already sized)");
        assert_eq!(
            renderer.tile_entry_regrows(),
            1,
            "no second regrow once the scratch fits the scene"
        );
        assert_eq!(
            renderer.tile_entry_overflow(),
            0,
            "second render must not overflow the grown scratch"
        );

        println!(
            "[tiled_entry] demand={grown} (>32768) grew {INITIAL_ENTRIES}->{grown} regrows=1 \
             overflow=0 non_black={nb} (>0) second_render regrows=1 overflow=0"
        );
    }

    /// M3.1 Task 2 — persistent tile-assign entry scratch: ONE renderer (whose
    /// `assign_bufs` is allocated exactly once in `alloc`) renders two
    /// DIFFERENT splat sets through `dispatch_into` over that one allocation,
    /// and both entry-count readbacks return the real (different, nonzero)
    /// counts — proving the per-frame `tile_count` zero-clear + buffer reuse
    /// is sound. The persistence is by construction: after Task 2 there is no
    /// per-frame allocation path left in `render()`.
    #[test]
    fn tile_assign_reuse_two_dispatches_one_allocation() {
        let Some(ctx) = try_gpu_context("tile_assign_reuse") else {
            return;
        };
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), 8_192, W, H)
            .expect("new_with_capacity(8192, 320x180) fits default device limits");

        // Set A: the tiled_capacity frame-1 recipe (30×20 block, close camera).
        let set_a = splat_block(30, 20, -2.0, 2.5, 0.0, 2.0, 1.5);
        upload(&ctx, &renderer, &set_a);
        renderer.set_active_splat_count(set_a.len() as u32);
        let cam_a = camera_looking_at(Vec3::new(-2.0, 2.5, 5.0), Vec3::new(-2.0, 2.5, 0.0));
        let frame_a = renderer.render(&cam_a).expect("frame A");
        let (_, nb_a) = frame_a.resolve_to_srgb(&ctx, &Illuminant::d65());
        let count_a = renderer.last_entry_count();

        // Set B: the tiled_capacity frame-2 recipe (56×32 block, farther
        // camera) — a different count by construction.
        let set_b = splat_block(56, 32, 14.0, 1.0, -3.0, 4.5, 2.5);
        assert_ne!(set_a.len(), set_b.len());
        upload(&ctx, &renderer, &set_b);
        renderer.set_active_splat_count(set_b.len() as u32);
        let cam_b = camera_looking_at(Vec3::new(14.0, 1.0, 9.0), Vec3::new(14.0, 1.0, -3.0));
        let frame_b = renderer.render(&cam_b).expect("frame B");
        let (_, nb_b) = frame_b.resolve_to_srgb(&ctx, &Illuminant::d65());
        let count_b = renderer.last_entry_count();

        // Real computed outcomes: two real, different, nonzero entry counts
        // through ONE TileAssignBuffers allocation, both frames lit.
        assert!(count_a > 0, "set A must emit tile entries (got {count_a})");
        assert!(count_b > 0, "set B must emit tile entries (got {count_b})");
        assert_ne!(
            count_a, count_b,
            "two different splat sets must produce different entry counts \
             through the reused allocation"
        );
        assert!(nb_a > 0, "frame A must light pixels (got {nb_a})");
        assert!(nb_b > 0, "frame B must light pixels (got {nb_b})");

        println!(
            "[tile_assign_reuse] dispatch_into x2 over one allocation: \
             count1={count_a} count2={count_b} n1!=n2"
        );
    }

    /// M3.1 Task 1 — per-frame stage spans on the renderer: one render must
    /// fill `assign_ms` (encoder-A submit → entry-count read complete) with a
    /// real nonzero wall span. Since Task 3 removed the per-frame 4-layer
    /// zero-fill (it runs once in `alloc`), `clear_ms` must report EXACTLY
    /// 0.0 — a nonzero value means staging writes crept back into the frame.
    #[test]
    fn tiled_frame_stage_spans() {
        let Some(ctx) = try_gpu_context("tiled_stage") else {
            return;
        };
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), 8_192, W, H)
            .expect("new_with_capacity(8192, 320x180) fits default device limits");
        let set = splat_block(30, 20, -2.0, 2.5, 0.0, 2.0, 1.5);
        upload(&ctx, &renderer, &set);
        renderer.set_active_splat_count(set.len() as u32);
        let cam = camera_looking_at(Vec3::new(-2.0, 2.5, 5.0), Vec3::new(-2.0, 2.5, 0.0));
        let frame = renderer.render(&cam).expect("render");
        assert!(
            frame.assign_ms > 0.0,
            "assign_ms must be a real submit->readback span (got {})",
            frame.assign_ms
        );
        assert_eq!(
            frame.clear_ms, 0.0,
            "clear_ms must be exactly 0.0 — the per-frame fill was removed \
             in Task 3 (got {})",
            frame.clear_ms
        );
        println!(
            "[tiled_stage] assign_ms={:.3} clear_ms={:.3} wall_ms={:.3}",
            frame.assign_ms, frame.clear_ms, frame.wall_ms
        );
    }

    /// M3.1 Task 3 — the stale-pixel guard for dropping the per-frame 59 MB
    /// output zero-fill: a bright frame followed by a 0-splat frame from the
    /// SAME camera must resolve fully black. This holds WITHOUT any per-frame
    /// fill because `tile_range_build` clears all tile ranges each frame and
    /// the frozen raster (`splat_raster.wgsl:124–135`) unconditionally
    /// `textureStore`s layers 0/1/3 for every in-bounds pixel — zeros when no
    /// splats survive. Written green BEFORE the fill removal (behavior-
    /// preserving optimization); it must stay green after.
    #[test]
    fn tiled_no_stale_pixels_without_per_frame_fill() {
        let Some(ctx) = try_gpu_context("tiled_no_stale") else {
            return;
        };
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), 8_192, W, H)
            .expect("new_with_capacity(8192, 320x180) fits default device limits");

        // Bright frame: the house 30×20 block, close camera.
        let set = splat_block(30, 20, -2.0, 2.5, 0.0, 2.0, 1.5);
        upload(&ctx, &renderer, &set);
        renderer.set_active_splat_count(set.len() as u32);
        let cam = camera_looking_at(Vec3::new(-2.0, 2.5, 5.0), Vec3::new(-2.0, 2.5, 0.0));
        let bright = renderer.render(&cam).expect("bright frame");
        let (_, nb1) = bright.resolve_to_srgb(&ctx, &Illuminant::d65());
        assert!(nb1 > 0, "bright frame must light pixels (got {nb1})");

        // Empty frame, SAME camera: not one stale pixel may survive.
        renderer.set_active_splat_count(0);
        let empty = renderer.render(&cam).expect("empty frame");
        let (_, nb2) = empty.resolve_to_srgb(&ctx, &Illuminant::d65());
        assert_eq!(
            nb2, 0,
            "the 0-splat frame must not inherit one stale pixel from the \
             bright frame (got {nb2} non-black)"
        );

        println!(
            "[tiled_no_stale] bright non_black={nb1} then empty non_black=0 \
             (per-frame fill removed)"
        );
    }
}
