//! `GiCombinePass` — the GI → raster residency fold (AAA Spec 11).
//!
//! Binds the GI radiance storage buffer DIRECTLY as input and the tiled
//! renderer's persistent splat buffer as read-write output, on ONE shared
//! `wgpu::Device`, and folds the GI-lit radiance's first 8 spectral bands into
//! each splat's `spectral[0..8]` field — entirely on-device, with NO CPU
//! readback between the GI compute pass and the rasterizer. This is the wedge
//! that kills the per-frame GI `poll(Wait)` on the resident path.
//!
//! The fold f16-quantizes each band to be BIT-IDENTICAL to the readback oracle
//! (`GpuGi::step` → `gaussian_splat_to_gpu_full`, which stores
//! `half::f16::from_f32(v)` and decodes it back). See `gi_combine.wgsl`: the
//! WGSL `quantize_f16` copies the validated round-to-nearest-even `f32_to_f16_rne`
//! from `relight_gpu.wgsl` rather than the truncating `pack2x16float` builtin.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// `gi_combine.wgsl` binding-2 `CombineParams` (std140 uniform): the receiver
/// count followed by three u32 of padding (16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CombineParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<CombineParams>() == 16);

/// On-device GI radiance → splat-spectral fold. Owns one compute pipeline and
/// its bind-group layout; per `dispatch` it allocates a transient params uniform
/// and a per-call bind group (the GI radiance + splat buffers vary per call), and
/// dispatches `count.div_ceil(64)` workgroups of 64.
pub struct GiCombinePass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GiCombinePass {
    /// Build the fold pipeline on the shared device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi_combine_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gi_combine.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_combine_bgl"),
            entries: &[
                // 0 — GI radiance (read storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1 — splat buffer (read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2 — params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi_combine_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi_combine_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }

    /// Record the fold into `encoder`: bind `gi_radiance_buf` (binding 0) and
    /// `splat_buf` (binding 1, read-write), upload a transient `{count}` uniform,
    /// and dispatch `splat_count.div_ceil(64)` workgroups. The caller submits.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gi_radiance_buf: &wgpu::Buffer,
        splat_buf: &wgpu::Buffer,
        splat_count: u32,
    ) {
        if splat_count == 0 {
            return;
        }
        let params = CombineParams { count: splat_count, _pad0: 0, _pad1: 0, _pad2: 0 };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_combine_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_combine_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gi_radiance_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gi_combine_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(splat_count.div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::splat_buffer::GpuSplatFull;

    /// Build a headless hardware device, or `None` on no-GPU / software adapter
    /// (the test then SKIPs — never fails — per the house no-panic contract).
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let info = adapter.get_info();
        if crate::gpu::adapter::ensure_hardware(&info).is_err() {
            eprintln!("[gi_combine test] software adapter ({}) — skipping", info.name);
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gi_combine_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;
        Some((device, queue))
    }

    /// The fold must write `splat.spectral[0..8]` EXACTLY equal to the readback
    /// oracle's f16 round-trip of `radiance[i][0..8]`, and leave the rest of the
    /// splat (bands beyond 8 — here the opacity/position fields) untouched.
    #[test]
    fn gi_combine_writes_first_8_bands_f16_quantized() {
        let Some((device, queue)) = try_device() else { return };

        const N: u32 = 4;
        // Distinct, non-power-of-two radiance values per (splat, band) so the f16
        // round-trip is non-trivial (exercises round-to-nearest-even, not just
        // exactly-representable values). Bands 8..15 are deliberately bright so a
        // bug that folded them into spectral[0..8] would show.
        let mut radiance = vec![[0.0f32; 16]; N as usize];
        for i in 0..N as usize {
            for b in 0..16 {
                radiance[i][b] = 0.01 + (i as f32) * 0.137 + (b as f32) * 0.0231;
            }
        }
        let radiance_flat: Vec<f32> =
            radiance.iter().flat_map(|r| r.iter().copied()).collect();

        // Splats pre-seeded with a recognizable sentinel in spectral + a known
        // opacity, so we can assert the fold touches ONLY spectral[0..8].
        let mut splats = vec![GpuSplatFull::zeroed(); N as usize];
        for (i, s) in splats.iter_mut().enumerate() {
            s.opacity_color = [0.0, 0.0, 0.0, 0.5 + 0.1 * i as f32];
            s.position_depth = [i as f32, 0.0, 0.0, 0.0];
            s.spectral = [-9.0; 8]; // sentinel: must be overwritten by the fold
        }

        let radiance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_combine_test_radiance"),
            contents: bytemuck::cast_slice(&radiance_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let splat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_combine_test_splats"),
            contents: bytemuck::cast_slice(&splats),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_combine_test_readback"),
            size: (splats.len() * std::mem::size_of::<GpuSplatFull>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pass = GiCombinePass::new(&device);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gi_combine_test") });
        pass.dispatch(&device, &mut encoder, &radiance_buf, &splat_buf, N);
        encoder.copy_buffer_to_buffer(
            &splat_buf,
            0,
            &readback,
            0,
            (splats.len() * std::mem::size_of::<GpuSplatFull>()) as u64,
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        assert!(matches!(rx.recv(), Ok(Ok(()))), "readback map must succeed on real hardware");

        let out: Vec<GpuSplatFull> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, GpuSplatFull>(&data).to_vec()
        };
        readback.unmap();

        // Assert EXACT bit-equality against the oracle f16 round-trip for bands
        // 0..8, and that the fold left opacity/position untouched.
        let mut max_abs = 0.0f32;
        for i in 0..N as usize {
            for b in 0..8 {
                let expected =
                    half::f16::from_bits(half::f16::from_f32(radiance[i][b]).to_bits()).to_f32();
                let got = out[i].spectral[b];
                let d = (got - expected).abs();
                if d > max_abs {
                    max_abs = d;
                }
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "splat {i} band {b}: fold f16 quant must be bit-identical to oracle: \
                     got={got} expected={expected}"
                );
            }
            // Untouched fields: opacity preserved, position preserved.
            assert_eq!(
                out[i].opacity_color[3], 0.5 + 0.1 * i as f32,
                "fold must not touch opacity (splat {i})"
            );
            assert_eq!(
                out[i].position_depth[0], i as f32,
                "fold must not touch position (splat {i})"
            );
        }

        eprintln!(
            "[gi_combine] N={N} max_abs(spectral - f16_oracle)={max_abs:e} (bit-identical required)"
        );
        assert_eq!(max_abs, 0.0, "fold must be bit-identical to the f16 readback oracle");
    }
}
