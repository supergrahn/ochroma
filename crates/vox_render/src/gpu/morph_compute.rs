use bytemuck::{Pod, Zeroable};
use wgpu;

/// GPU-layout packed splat delta (std430-compatible, 64 bytes).
/// Matches PackedSplatDelta in morph_targets.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSplatDelta {
    pub splat_index: u32,    // 4
    pub d_position: [f32; 3], // 12
    pub d_scale: [f32; 3],   // 12
    pub _pad0: f32,           // 4
    pub d_spectral: [u16; 8], // 16
    pub _pad1: [u32; 4],      // 16
    // Total: 4 + 12 + 12 + 4 + 16 + 16 = 64 bytes
}

/// Target offset range in the delta buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TargetOffsets {
    pub start: u32,
    pub end: u32,
    pub _pad: [u32; 2],
}

pub struct MorphComputePass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl MorphComputePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_src = include_str!("morph_targets.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morph_targets"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("morph_compute_bgl"),
            entries: &[
                // @binding(0) base_splats: storage read
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
                // @binding(1) delta_buffer: storage read
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
                // @binding(2) morph_params: uniform
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
                // @binding(3) target_offsets: storage read
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
                // @binding(4) output_splats: storage read_write
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morph_compute_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morph_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_morph_targets"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }

    /// Dispatch the morph compute pass.
    /// `delta_buf`: all active targets' deltas concatenated.
    /// `weights`: up to 16 f32 weights (padded to 16 if fewer).
    /// `offsets`: TargetOffsets per target (up to 16).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        base_splat_buf: &wgpu::Buffer,
        delta_buf: &wgpu::Buffer,
        weights: &[f32; 16],
        offsets: &[TargetOffsets; 16],
        output_buf: &wgpu::Buffer,
        splat_count: u32,
    ) {
        // Pack uniform: weights[16] + target_count u32 + splat_count u32 + pad[2] u32
        // = 64 + 16 = 80 bytes
        let active_targets = offsets.iter().filter(|o| o.end > o.start).count() as u32;

        let mut uniform_data = Vec::<u8>::with_capacity(80);
        for &w in weights.iter() {
            uniform_data.extend_from_slice(&w.to_le_bytes());
        }
        uniform_data.extend_from_slice(&active_targets.to_le_bytes());
        uniform_data.extend_from_slice(&splat_count.to_le_bytes());
        uniform_data.extend_from_slice(&0u32.to_le_bytes()); // _pad[0]
        uniform_data.extend_from_slice(&0u32.to_le_bytes()); // _pad[1]

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morph_params_uniform"),
            size: uniform_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(&uniform_data);
        params_buf.unmap();

        let offsets_bytes: &[u8] = bytemuck::cast_slice(offsets.as_slice());
        let offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morph_target_offsets"),
            size: offsets_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        offsets_buf.slice(..).get_mapped_range_mut().copy_from_slice(offsets_bytes);
        offsets_buf.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morph_compute_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: base_splat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
            ],
        });

        let workgroups = splat_count.div_ceil(256);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("morph_compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_splat_delta_size() {
        assert_eq!(std::mem::size_of::<GpuSplatDelta>(), 64);
    }
}
