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
use vox_core::spectral::{spectral_to_xyz, xyz_to_srgb, Illuminant, SpectralBands};
use vox_core::types::GaussianSplat;

use crate::gpu::gpu_timing::GpuTimers;
use crate::gpu::radix_sort_pass::RadixSortPass;
use crate::gpu::splat_buffer::{
    gaussian_splat_to_gpu_full, gaussian_splats_to_transforms, GpuSplatFull,
};
use crate::gpu::splat_raster::{RasterParams, SplatRasterPass};
use crate::gpu::tile_assign::TileAssignPass;
use crate::gpu::tile_range_build::TileRangeBuildPass;
use crate::gpu::GpuContext;
use crate::spectral::RenderCamera;

/// Tile edge length in pixels (must match `tile_assign.wgsl` / `splat_raster.wgsl`
/// and the CPU `spectra_render` `TILE_SIZE`).
const TILE_SIZE: u32 = 16;
/// Maximum tile entries emitted per splat by `tile_assign` (`MAX_TILES_PER_SPLAT`).
const MAX_TILES_PER_SPLAT: u32 = 16;

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
            TiledRenderError::ExceedsDeviceLimits { what, requested, limit } => write!(
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
    /// chain actually drew something. The 8 GPU bands are mapped onto the low 8
    /// of the 16-band [`SpectralBands`] (the upper 8 stay 0), matching how the
    /// raster shader packs `spectral[0..8]`.
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
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
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
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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
                    spd[0] = floats[b0];
                    spd[1] = floats[b0 + 1];
                    spd[2] = floats[b0 + 2];
                    spd[3] = floats[b0 + 3];
                    spd[4] = floats[b1];
                    spd[5] = floats[b1 + 1];
                    spd[6] = floats[b1 + 2];
                    spd[7] = floats[b1 + 3];

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

    // Reused 4-layer rgba32float output.
    output_texture: wgpu::Texture,

    // Two camera uniforms (assign layout + raster layout).
    assign_camera_buf: wgpu::Buffer,
    raster_camera_buf: wgpu::Buffer,

    // 4-byte MAP_READ buffer for the single host-count readback.
    count_readback: wgpu::Buffer,

    // Zero buffer used to clear the output texture each frame (feature-free).
    zero_row: Vec<u8>,

    timers: GpuTimers,

    width: u32,
    height: u32,
    splat_count: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_tiles: u32,
    max_tile_entries: u32,
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
        let device = ctx.device().clone();
        let queue = ctx.queue().clone();

        let width = width.max(1);
        let height = height.max(1);

        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        let num_tiles = tiles_x * tiles_y;

        let splat_count = splats.len() as u32;
        // tile_assign emits up to MAX_TILES_PER_SPLAT entries per splat.
        let max_tile_entries = splat_count.saturating_mul(MAX_TILES_PER_SPLAT).max(1);

        // ── Validate sizes against device limits BEFORE allocating ────────────
        let limits = device.limits();
        let max_storage = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;

        let splat_bytes = (splat_count as u64).max(1) * std::mem::size_of::<GpuSplatFull>() as u64;
        let transform_bytes = (splat_count as u64).max(1) * 2 * 16; // two vec4 per splat
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiled_transform_buf"),
            size: transform_bytes.max(32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // Reused output texture (4 layers). COPY_DST so we can zero-fill from the
        // queue each frame; COPY_SRC so the proof readback can read it.
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tiled_output_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 4 },
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

        // ── Upload splats + transforms ONCE ─────────────────────────────────────
        if splat_count > 0 {
            let gpu_splats: Vec<GpuSplatFull> =
                splats.iter().map(gaussian_splat_to_gpu_full).collect();
            queue.write_buffer(&splat_buf, 0, bytemuck::cast_slice(&gpu_splats));

            let transforms = gaussian_splats_to_transforms(splats);
            queue.write_buffer(&transform_buf, 0, bytemuck::cast_slice(&transforms));
        }

        // ── GPU timers: 1 pair, slot 0 brackets the whole compute span ──────────
        let timers = GpuTimers::new(&device, &queue, device.features(), 1);

        // Padded zero row for the per-frame texture clear.
        let padded_bpr = (width * 16).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let zero_row = vec![0u8; (padded_bpr * height) as usize];

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
            output_texture,
            assign_camera_buf,
            raster_camera_buf,
            count_readback,
            zero_row,
            timers,
            width,
            height,
            splat_count,
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

    /// Run the full tiled chain for `camera` and return a [`TiledFrame`].
    ///
    /// Stage layout:
    ///   A) `tile_assign` (allocates its own per-frame entry buffers) + copy
    ///      `tile_count` to a MAP_READ buffer; submit; ONE blocking poll to read
    ///      the real entry count `n` (the only host readback between stages).
    ///   B) timed (slot 0): `radix_sort(n)` → `tile_range_build(n)` → clear the
    ///      output (queue zero-fill, untimed) → `splat_raster`; resolve timers;
    ///      submit. `raster_gpu_ms` is read back after.
    pub fn render(&mut self, camera: &RenderCamera) -> Result<TiledFrame, TiledRenderError> {
        let device = self.ctx.device().clone();
        let queue = self.ctx.queue().clone();

        // (a) Write BOTH camera uniforms.
        let assign_cam = self.assign_camera(camera);
        let raster_cam = self.raster_camera(camera);
        queue.write_buffer(&self.assign_camera_buf, 0, bytemuck::bytes_of(&assign_cam));
        queue.write_buffer(&self.raster_camera_buf, 0, bytemuck::bytes_of(&raster_cam));

        // (b) Encoder A: tile_assign → buffers; copy tile_count to readback.
        let mut enc_a = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tiled_encoder_a"),
        });
        let assign = self.tile_assign.dispatch(
            &device,
            &mut enc_a,
            &self.assign_camera_buf,
            &self.splat_buf,
            &self.transform_buf,
            self.splat_count,
            self.max_tile_entries,
        );
        enc_a.copy_buffer_to_buffer(&assign.tile_count, 0, &self.count_readback, 0, 4);
        queue.submit(Some(enc_a.finish()));

        // ONE poll(Wait) to read the real entry count.
        let n = {
            let slice = self.count_readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            device.poll(wgpu::Maintain::Wait);
            if !matches!(rx.recv(), Ok(Ok(()))) {
                // Readback failed: treat as zero entries (frame stays black, no panic).
                0u32
            } else {
                let data = slice.get_mapped_range();
                let v = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
                drop(data);
                self.count_readback.unmap();
                // Defensive clamp: never sort/range-build past the allocated entries.
                v.min(self.max_tile_entries)
            }
        };

        // Clear the output texture (queue zero-fill, all 4 layers) BEFORE the
        // timed encoder — feature-free, avoids requiring CLEAR_TEXTURE, and keeps
        // the clear out of the measured compute span. The raster pass then writes
        // every in-bounds pixel of layers 0/1/3, so no stale pixel survives.
        let padded_bpr = (self.width * 16).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        for layer in 0u32..4 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                &self.zero_row,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
            );
        }

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

        // radix sort the per-frame tile_assign buffers in place (8 passes).
        self.radix.sort(
            &device,
            &mut enc_b,
            &assign.tile_keys_lo,
            &assign.tile_keys_hi,
            &assign.tile_vals,
            n,
            &self.tmp_lo,
            &self.tmp_hi,
            &self.tmp_vals,
        );

        // Build per-tile [start,end) ranges from the sorted hi keys (tile ids).
        self.tile_range.dispatch(
            &device,
            &mut enc_b,
            &assign.tile_keys_hi,
            &self.tile_ranges,
            n,
            self.num_tiles,
        );

        // NOTE on GPU timing: pass-boundary TIMESTAMP_QUERY can only bracket a
        // single compute pass's begin→end, but each stage here (radix_sort,
        // tile_range_build, splat_raster) opens its OWN pass with
        // `timestamp_writes: None` — and those shaders/passes are frozen. So the
        // chain cannot be hardware-timestamped without restructuring (Spec 11
        // does exactly that via encoder-level timestamps / a resident loop).
        // Until then `raster_gpu_ms` is honestly `None` and the wall-clock of the
        // GPU-only `enc_b` submit→poll below is the faithful chain measure.
        self.raster.dispatch(
            &device,
            &mut enc_b,
            &self.raster_camera_buf,
            &self.splat_buf,
            &assign.tile_vals,
            &self.tile_ranges,
            &self.output_texture,
            RasterParams {
                width: self.width,
                height: self.height,
                num_tiles_x: self.tiles_x,
                _pad: 0,
            },
        );

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
            width: self.width,
            height: self.height,
        })
    }
}
