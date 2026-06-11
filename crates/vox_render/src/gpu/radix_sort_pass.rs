//! 4-pass 8-bit GPU radix sort.
//!
//! Sorts pairs of u32 key buffers (lo/hi) plus a u32 value buffer in ascending
//! 64-bit key order: `full_key = (keys_hi << 32) | keys_lo`.
//!
//! Algorithm: "Fast 4-way parallel radix sorting on GPUs" — each pass processes
//! 8 bits, giving 8 total passes (4 for lo-word, 4 for hi-word).
//!
//! Per-pass stages:
//!   1. `radix_histogram` — count 256 buckets per workgroup into a global histogram.
//!   2. `radix_prefix_sum` — exclusive scan of the 256-entry histogram.
//!   3. `radix_scatter` — scatter keys/vals to output using the prefix cursors.
//!
//! Ping-pong between the live buffers and the provided temporaries so the result
//! always lands back in the original buffers after an even number of passes (8).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Params struct — must match the WGSL `RadixParams` struct exactly (16 bytes).
// ---------------------------------------------------------------------------

/// Per-pass parameters uploaded as a uniform before each histogram/scatter dispatch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RadixParams {
    /// Number of valid elements to sort.
    pub count: u32,
    /// Bit offset for the 8-bit digit extracted this pass (0, 8, 16, or 24).
    pub bit_shift: u32,
    /// 0 = use `keys_lo` as the key source; 1 = use `keys_hi`.
    pub use_hi_key: u32,
    /// Pass index (0-7), informational — not consumed by the shader.
    pub pass_idx: u32,
}

// ---------------------------------------------------------------------------
// RadixSortPrepared
// ---------------------------------------------------------------------------

/// Persistent encoder-B handles for [`RadixSortPass::sort_prepared`] (M3.1
/// Task 5): one 16 B `UNIFORM|COPY_DST` params buffer and one bind group per
/// radix pass, built ONCE against caller-owned key/val/tmp buffers by
/// [`RadixSortPass::prepare`]. Kills the per-frame churn the [`RadixSortPass::sort`]
/// path pays — 8 × (`create_buffer_init` params staging + a fresh 9-entry
/// bind group) EVERY call. Rebuild whenever any bound buffer is recreated
/// (the renderer's entry-scratch grow path).
pub struct RadixSortPrepared {
    /// One params uniform per pass — 8 DISTINCT buffers, so the pre-submit
    /// `queue.write_buffer` calls in `sort_prepared` can never alias each
    /// other within a frame.
    params_bufs: [wgpu::Buffer; 8],
    /// Per-pass bind groups with the exact src/dst ping-pong parity of
    /// [`RadixSortPass::sort`]: even passes read the originals and write the
    /// tmps, odd passes the reverse.
    bind_groups: [wgpu::BindGroup; 8],
}

// ---------------------------------------------------------------------------
// RadixSortPass
// ---------------------------------------------------------------------------

/// 4-pass 8-bit GPU radix sort over pairs of u32 key buffers + a value buffer.
pub struct RadixSortPass {
    /// Counts 256 per-bucket occurrences across all keys.
    histogram_pipeline: wgpu::ComputePipeline,
    /// Computes an exclusive prefix sum of the 256-entry histogram in place.
    prefix_pipeline: wgpu::ComputePipeline,
    /// Scatters keys and values to output positions determined by the prefix.
    scatter_pipeline: wgpu::ComputePipeline,
    /// Shared bind group layout used by all three pipelines.
    bgl: wgpu::BindGroupLayout,
    /// GPU-side histogram buffer (256 × u32, atomic).
    histogram_buf: wgpu::Buffer,
    /// GPU-side prefix buffer (256 × u32, atomic cursor during scatter).
    prefix_buf: wgpu::Buffer,
    /// Params uniform buffer (16 bytes, written once per pass).
    params_buf: wgpu::Buffer,
}

impl RadixSortPass {
    /// Create all three compute pipelines and allocate persistent GPU buffers.
    pub fn new(device: &wgpu::Device) -> Self {
        // ---- shader module ------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_pass_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("radix_sort_pass.wgsl").into()),
        });

        // ---- bind group layout -------------------------------------------
        // One layout covers all three entry points.  Unused bindings in each
        // stage are bound but not accessed by the shader.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_sort_bgl"),
            entries: &[
                // 0: RadixParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: keys_lo input (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: keys_hi input (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: vals input (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: histogram (read_write storage, atomics)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: prefix (read_write storage, atomics)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: keys_lo output
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: keys_hi output
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: vals output
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ---- pipeline layout ---------------------------------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radix_sort_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // ---- pipelines ---------------------------------------------------
        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_histogram_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("radix_histogram"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let prefix_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_prefix_sum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("radix_prefix_sum"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_scatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("radix_scatter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ---- persistent GPU buffers --------------------------------------
        let histogram_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_histogram_buf"),
            // 256 u32 buckets
            size: 256 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prefix_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_prefix_buf"),
            size: 256 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params are written fresh before each pass.
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_params_buf"),
            size: std::mem::size_of::<RadixParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            histogram_pipeline,
            prefix_pipeline,
            scatter_pipeline,
            bgl,
            histogram_buf,
            prefix_buf,
            params_buf,
        }
    }

    /// Sort `count` entries in-place across 8 passes (4 × lo-word + 4 × hi-word).
    ///
    /// After the call `keys_lo`, `keys_hi`, and `vals` contain the sorted data —
    /// the ping-pong uses `tmp_*` internally and the final result is copied back
    /// to the originals when needed.  With 8 (even) passes the data naturally
    /// ends in the original buffers.
    ///
    /// # Arguments
    /// * `keys_lo` / `keys_hi` — 64-bit key split into two `u32` buffers.
    ///   `keys_lo` holds the low 32 bits (e.g. `depth.to_bits()`);
    ///   `keys_hi` holds the high 32 bits (e.g. tile index).
    /// * `vals` — one `u32` per entry (splat index).
    /// * `tmp_lo`, `tmp_hi`, `tmp_vals` — scratch buffers of the same size.
    ///
    /// # Panics
    /// Does not panic; if `count == 0` all dispatches are skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn sort(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        keys_lo: &wgpu::Buffer,
        keys_hi: &wgpu::Buffer,
        vals: &wgpu::Buffer,
        count: u32,
        tmp_lo: &wgpu::Buffer,
        tmp_hi: &wgpu::Buffer,
        tmp_vals: &wgpu::Buffer,
    ) {
        if count == 0 {
            return;
        }

        // Workgroup count for per-element dispatches.
        let wg_count = count.div_ceil(256);

        // The two buffer pairs that we ping-pong between.
        // Pass 0 reads from (keys_lo, keys_hi, vals) and writes to (tmp_*).
        // Pass 1 reads from (tmp_*) and writes back, etc.
        // With 8 passes the final result ends up in the original buffers.
        let pairs: [(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer); 2] =
            [(keys_lo, keys_hi, vals), (tmp_lo, tmp_hi, tmp_vals)];

        // 8 passes total: bits 0-7, 8-15, 16-23, 24-31 of lo, then same of hi.
        for pass in 0u32..8 {
            let use_hi_key: u32 = if pass < 4 { 0 } else { 1 };
            let bit_shift: u32 = (pass % 4) * 8;

            let src_pair = &pairs[(pass % 2) as usize];
            let dst_pair = &pairs[((pass + 1) % 2) as usize];

            let (src_lo, src_hi, src_vals) = (src_pair.0, src_pair.1, src_pair.2);
            let (dst_lo, dst_hi, dst_vals) = (dst_pair.0, dst_pair.1, dst_pair.2);

            let params = RadixParams {
                count,
                bit_shift,
                use_hi_key,
                pass_idx: pass,
            };

            // Upload params.
            encoder.clear_buffer(&self.histogram_buf, 0, None);
            encoder.clear_buffer(&self.prefix_buf, 0, None);

            // We need a tiny staging upload for params.  wgpu's
            // `write_buffer_with` is encoder-based; we use a one-shot init
            // buffer and copy it in.
            let params_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("radix_params_staging"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(
                &params_staging,
                0,
                &self.params_buf,
                0,
                std::mem::size_of::<RadixParams>() as u64,
            );

            // ---- stage 1: histogram --------------------------------------
            let hist_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_hist_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_lo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src_hi.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: src_vals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.prefix_buf.as_entire_binding(),
                    },
                    // output slots — bound but not written during histogram
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: dst_lo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: dst_hi.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: dst_vals.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_histogram_pass"),
                    timestamp_writes: None,
                });
                pass_enc.set_pipeline(&self.histogram_pipeline);
                pass_enc.set_bind_group(0, &hist_bg, &[]);
                pass_enc.dispatch_workgroups(wg_count, 1, 1);
            }

            // ---- stage 2: prefix sum (1 workgroup of 256 threads) --------
            // Reuse the same bind group — prefix_sum only reads histogram[] and
            // writes prefix[]; it ignores all other bindings.
            {
                let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_prefix_pass"),
                    timestamp_writes: None,
                });
                pass_enc.set_pipeline(&self.prefix_pipeline);
                pass_enc.set_bind_group(0, &hist_bg, &[]);
                pass_enc.dispatch_workgroups(1, 1, 1);
            }

            // ---- stage 3: scatter ----------------------------------------
            {
                let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_scatter_pass"),
                    timestamp_writes: None,
                });
                pass_enc.set_pipeline(&self.scatter_pipeline);
                pass_enc.set_bind_group(0, &hist_bg, &[]);
                pass_enc.dispatch_workgroups(wg_count, 1, 1);
            }
        }
        // After 8 passes (even), data is back in the original (keys_lo, keys_hi, vals).
    }

    /// Build the persistent per-pass params buffers + bind groups for
    /// [`Self::sort_prepared`] (M3.1 Task 5 — additive; [`Self::sort`] is
    /// untouched). Each of the 8 passes gets its OWN 16 B `UNIFORM|COPY_DST`
    /// params buffer bound at binding 0 and a bind group with the exact
    /// src/dst ping-pong parity of `sort`. The handles borrow nothing — they
    /// hold their own resource references — but they BIND the passed buffers,
    /// so rebuild them whenever any of those buffers is recreated.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        keys_lo: &wgpu::Buffer,
        keys_hi: &wgpu::Buffer,
        vals: &wgpu::Buffer,
        tmp_lo: &wgpu::Buffer,
        tmp_hi: &wgpu::Buffer,
        tmp_vals: &wgpu::Buffer,
    ) -> RadixSortPrepared {
        // Same ping-pong table as `sort`: pass 0 reads the originals and
        // writes the tmps; pass 1 the reverse; etc.
        let pairs: [(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer); 2] =
            [(keys_lo, keys_hi, vals), (tmp_lo, tmp_hi, tmp_vals)];

        let params_bufs: [wgpu::Buffer; 8] = std::array::from_fn(|pass| {
            let label = format!("radix_prepared_params_{pass}");
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: std::mem::size_of::<RadixParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let bind_groups: [wgpu::BindGroup; 8] = std::array::from_fn(|pass| {
            let (src_lo, src_hi, src_vals) = pairs[pass % 2];
            let (dst_lo, dst_hi, dst_vals) = pairs[(pass + 1) % 2];
            let label = format!("radix_prepared_bg_{pass}");
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&label),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_bufs[pass].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_lo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src_hi.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: src_vals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.prefix_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: dst_lo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: dst_hi.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: dst_vals.as_entire_binding(),
                    },
                ],
            })
        });

        RadixSortPrepared {
            params_bufs,
            bind_groups,
        }
    }

    /// [`Self::sort`] against pre-built handles (M3.1 Task 5): identical 8-pass
    /// encode — per pass clear histogram/prefix then the same histogram →
    /// prefix → scatter sub-passes — but the per-frame host work shrinks to
    /// 8 × `queue.write_buffer` of 16 B. Bit-exact with `sort` (proven by
    /// `radix_prepared_matches_sort_bitexact`).
    ///
    /// The params writes are PRE-SUBMIT: they execute at the next
    /// `queue.submit`, before the encoder's command buffer — safe because each
    /// pass owns a distinct buffer written exactly once per call. (Two
    /// `sort_prepared` calls on the SAME prepared handles into one
    /// un-submitted encoder would alias the params — submit in between, as
    /// `render()` does.)
    pub fn sort_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
        prepared: &RadixSortPrepared,
    ) {
        if count == 0 {
            return;
        }

        let wg_count = count.div_ceil(256);

        for pass in 0u32..8 {
            let use_hi_key: u32 = if pass < 4 { 0 } else { 1 };
            let bit_shift: u32 = (pass % 4) * 8;

            let params = RadixParams {
                count,
                bit_shift,
                use_hi_key,
                pass_idx: pass,
            };
            queue.write_buffer(
                &prepared.params_bufs[pass as usize],
                0,
                bytemuck::bytes_of(&params),
            );

            encoder.clear_buffer(&self.histogram_buf, 0, None);
            encoder.clear_buffer(&self.prefix_buf, 0, None);

            let bg = &prepared.bind_groups[pass as usize];

            // ---- stage 1: histogram --------------------------------------
            {
                let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_histogram_pass"),
                    timestamp_writes: None,
                });
                pass_enc.set_pipeline(&self.histogram_pipeline);
                pass_enc.set_bind_group(0, bg, &[]);
                pass_enc.dispatch_workgroups(wg_count, 1, 1);
            }

            // ---- stage 2: prefix sum (1 workgroup of 256 threads) --------
            {
                let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_prefix_pass"),
                    timestamp_writes: None,
                });
                pass_enc.set_pipeline(&self.prefix_pipeline);
                pass_enc.set_bind_group(0, bg, &[]);
                pass_enc.dispatch_workgroups(1, 1, 1);
            }

            // ---- stage 3: scatter ----------------------------------------
            {
                let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_scatter_pass"),
                    timestamp_writes: None,
                });
                pass_enc.set_pipeline(&self.scatter_pipeline);
                pass_enc.set_bind_group(0, bg, &[]);
                pass_enc.dispatch_workgroups(wg_count, 1, 1);
            }
        }
        // After 8 passes (even), data is back in the original (keys_lo, keys_hi, vals).
    }
}

// ---------------------------------------------------------------------------
// CPU reference sort (for testing / validation)
// ---------------------------------------------------------------------------

/// Sort `(lo, hi, val)` triples by the combined 64-bit key `(hi << 32) | lo`
/// in ascending order.  Used as a CPU reference to validate GPU output.
pub fn cpu_reference_sort(lo: &mut [u32], hi: &mut [u32], vals: &mut [u32]) {
    assert_eq!(lo.len(), hi.len());
    assert_eq!(lo.len(), vals.len());

    let n = lo.len();
    let mut triples: Vec<(u64, u32)> = (0..n)
        .map(|i| ((hi[i] as u64) << 32 | lo[i] as u64, vals[i]))
        .collect();

    triples.sort_unstable_by_key(|&(k, _)| k);

    for (i, (k, v)) in triples.into_iter().enumerate() {
        lo[i] = (k & 0xFFFF_FFFF) as u32;
        hi[i] = (k >> 32) as u32;
        vals[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// Headless device/queue over the local hardware GPU — the house skip
    /// pattern (`tiled_splat_renderer.rs::try_gpu_context`): print the skip
    /// line and return `None` on no adapter / a software rasteriser so a
    /// GPU-less box stays green.
    fn try_gpu(label: &str) -> Option<(wgpu::Device, wgpu::Queue)> {
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
                label: Some("radix_prepared_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some((device, queue))
    }

    /// Blocking readback of a whole `n × u32` STORAGE|COPY_SRC buffer as raw bytes.
    fn read_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf: &wgpu::Buffer,
        n: u32,
    ) -> Vec<u8> {
        let bytes = n as u64 * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_prepared_test_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("radix_prepared_test_readback"),
        });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map callback").expect("map ok");
        let out = slice.get_mapped_range().to_vec();
        staging.unmap();
        out
    }

    /// M3.1 Task 5: `prepare` + `sort_prepared` must be BIT-EXACT with `sort`.
    ///
    /// ~50k deterministic entries are uploaded twice into two identical
    /// buffer sets; one is sorted via the per-frame-churn `sort`, the other
    /// via `prepare` + `sort_prepared`; all three result buffers from each
    /// are read back and byte-compared (plus both against the CPU reference,
    /// proving a real sort happened — not equal garbage).
    ///
    /// INPUT CONSTRUCTION (race-free by design): the frozen scatter claims
    /// output slots with `atomicAdd(&prefix[bucket], 1u)` in ARRIVAL order,
    /// so ANY input where two distinct triples share a per-pass digit is
    /// run-to-run nondeterministic — verified here: `sort` vs `sort` on
    /// identical 50k hash-keyed inputs fails a byte-compare (the documented
    /// radix nondeterminism landmine). The oracle therefore uses 256 distinct
    /// triples — key byte `b` of group `g` is `(g * odd_b) & 0xFF`, an
    /// odd-multiplier bijection mod 256, so in EVERY pass all 256 bucket
    /// digits are distinct across groups — repeated 195× as IDENTICAL
    /// triples (val = g too). Each bucket's slot range is then filled
    /// entirely with identical bytes: the scatter race cannot change the
    /// output, both paths are fully deterministic, and any wrapper
    /// divergence (ping-pong parity, per-pass bit_shift/use_hi_key routing,
    /// bindings) still produces different bytes.
    #[test]
    fn radix_prepared_matches_sort_bitexact() {
        let Some((device, queue)) = try_gpu("radix_prepared") else {
            return;
        };
        let pass = RadixSortPass::new(&device);

        // 256 groups × 195 identical copies = 49,920 entries.
        const GROUPS: u32 = 256;
        const COPIES: u32 = 195;
        const N: u32 = GROUPS * COPIES;
        // One odd multiplier per key byte (passes 0-3 = lo bytes, 4-7 = hi).
        const M: [u32; 8] = [1, 3, 5, 7, 9, 11, 13, 15];
        let byte = |g: u32, b: usize| -> u32 { (g.wrapping_mul(M[b])) & 0xFF };
        let mut lo = Vec::with_capacity(N as usize);
        let mut hi = Vec::with_capacity(N as usize);
        let mut vals = Vec::with_capacity(N as usize);
        for g in 0..GROUPS {
            let glo = byte(g, 0) | byte(g, 1) << 8 | byte(g, 2) << 16 | byte(g, 3) << 24;
            let ghi = byte(g, 4) | byte(g, 5) << 8 | byte(g, 6) << 16 | byte(g, 7) << 24;
            for _ in 0..COPIES {
                lo.push(glo);
                hi.push(ghi);
                vals.push(g);
            }
        }

        let mk = |label: &str, data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mk_tmp = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: N as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        // Two identical buffer sets.
        let (a_lo, a_hi, a_vals) = (mk("a_lo", &lo), mk("a_hi", &hi), mk("a_vals", &vals));
        let (a_tlo, a_thi, a_tvals) = (mk_tmp("a_tlo"), mk_tmp("a_thi"), mk_tmp("a_tvals"));
        let (b_lo, b_hi, b_vals) = (mk("b_lo", &lo), mk("b_hi", &hi), mk("b_vals", &vals));
        let (b_tlo, b_thi, b_tvals) = (mk_tmp("b_tlo"), mk_tmp("b_thi"), mk_tmp("b_tvals"));

        // Set A through the existing per-frame-churn `sort`.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("radix_prepared_test_sort"),
        });
        pass.sort(
            &device, &mut enc, &a_lo, &a_hi, &a_vals, N, &a_tlo, &a_thi, &a_tvals,
        );
        queue.submit(Some(enc.finish()));

        // Set B through prepare + sort_prepared (the M3.1 Task 5 path).
        let prepared = pass.prepare(&device, &b_lo, &b_hi, &b_vals, &b_tlo, &b_thi, &b_tvals);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("radix_prepared_test_sort_prepared"),
        });
        pass.sort_prepared(&queue, &mut enc, N, &prepared);
        queue.submit(Some(enc.finish()));

        let (ra_lo, ra_hi, ra_vals) = (
            read_bytes(&device, &queue, &a_lo, N),
            read_bytes(&device, &queue, &a_hi, N),
            read_bytes(&device, &queue, &a_vals, N),
        );
        let (rb_lo, rb_hi, rb_vals) = (
            read_bytes(&device, &queue, &b_lo, N),
            read_bytes(&device, &queue, &b_hi, N),
            read_bytes(&device, &queue, &b_vals, N),
        );

        assert_eq!(ra_lo, rb_lo, "keys_lo: sort vs sort_prepared must be byte-equal");
        assert_eq!(ra_hi, rb_hi, "keys_hi: sort vs sort_prepared must be byte-equal");
        assert_eq!(ra_vals, rb_vals, "vals: sort vs sort_prepared must be byte-equal");

        // Both must equal the CPU reference — a real sort, not equal garbage.
        let (mut c_lo, mut c_hi, mut c_vals) = (lo.clone(), hi.clone(), vals.clone());
        cpu_reference_sort(&mut c_lo, &mut c_hi, &mut c_vals);
        assert_eq!(ra_lo, bytemuck::cast_slice::<u32, u8>(&c_lo), "keys_lo vs CPU reference");
        assert_eq!(ra_hi, bytemuck::cast_slice::<u32, u8>(&c_hi), "keys_hi vs CPU reference");
        assert_eq!(ra_vals, bytemuck::cast_slice::<u32, u8>(&c_vals), "vals vs CPU reference");

        println!("[radix_prepared] sorted={N} entries match RadixSortPass::sort bit-exactly");
    }

    /// The `RadixParams` struct must be exactly 16 bytes so it maps 1:1 onto
    /// the WGSL uniform without any hidden padding.
    #[test]
    fn radix_sort_params_size() {
        assert_eq!(std::mem::size_of::<RadixParams>(), 16);
    }

    /// The struct must satisfy Pod/Zeroable (compile-time check via zeroed()).
    #[test]
    fn radix_params_pod() {
        let _ = RadixParams::zeroed();
    }

    /// CPU reference sort: verify basic ascending order on a small synthetic input.
    #[test]
    fn cpu_reference_sort_basic() {
        let mut lo = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut hi = vec![0u32; 8]; // all hi == 0, sort by lo only
        let mut vals = vec![0u32, 1, 2, 3, 4, 5, 6, 7];

        cpu_reference_sort(&mut lo, &mut hi, &mut vals);

        // Keys should be in ascending order.
        for w in lo.windows(2) {
            assert!(w[0] <= w[1], "keys out of order: {} > {}", w[0], w[1]);
        }
    }

    /// CPU reference sort: keys spanning both lo and hi words.
    #[test]
    fn cpu_reference_sort_full_64bit() {
        let n = 64usize;
        let mut lo: Vec<u32> = (0..n as u32).rev().collect();
        let mut hi: Vec<u32> = (0..n as u32).rev().collect();
        let mut vals: Vec<u32> = (0..n as u32).collect();

        cpu_reference_sort(&mut lo, &mut hi, &mut vals);

        for i in 1..n {
            let prev = (hi[i - 1] as u64) << 32 | lo[i - 1] as u64;
            let curr = (hi[i] as u64) << 32 | lo[i] as u64;
            assert!(prev <= curr, "64-bit keys out of order at index {}", i);
        }
    }

    // GPU correctness is tested via integration tests (a wgpu device is required).
    // Run with: cargo test --test gpu_radix_sort -- --ignored
}
