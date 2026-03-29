//! GPU compute pass for Gaussian splat skinning.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSkinSplat {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub scale: [f32; 3],
    pub opacity: f32,
    pub rotation: [f32; 4],
    pub spectral: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuJointTransform {
    pub skin_matrix: [[f32; 4]; 4],
}

pub struct SkinningCompute {
    pipeline: wgpu::ComputePipeline,
    _base_splat_buffer: wgpu::Buffer,
    _joint_binding_buffer: wgpu::Buffer,
    joint_transform_buffer: wgpu::Buffer,
    pub skinned_splat_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub splat_count: u32,
}

impl SkinningCompute {
    pub fn new(
        device: &wgpu::Device,
        base_splats: &[GpuSkinSplat],
        joint_bindings: &[u32],
        joint_count: usize,
    ) -> Self {
        assert_eq!(base_splats.len(), joint_bindings.len());
        let splat_count = base_splats.len() as u32;

        let base_splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinning_base_splats"),
            contents: bytemuck::cast_slice(base_splats),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let joint_binding_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinning_joint_bindings"),
            contents: bytemuck::cast_slice(joint_bindings),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let joint_transform_size = (joint_count * std::mem::size_of::<GpuJointTransform>()) as u64;
        let joint_transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_joint_transforms"),
            size: joint_transform_size.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skinned_splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_output_splats"),
            size: ((base_splats.len() * std::mem::size_of::<GpuSkinSplat>()) as u64).max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_src = include_str!("skinning.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinning_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: base_splat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: joint_binding_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: joint_transform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: skinned_splat_buffer.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinning_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skinning_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_skin"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            _base_splat_buffer: base_splat_buffer,
            _joint_binding_buffer: joint_binding_buffer,
            joint_transform_buffer,
            skinned_splat_buffer,
            bind_group,
            splat_count,
        }
    }

    pub fn update_joints(&self, queue: &wgpu::Queue, joint_transforms: &[GpuJointTransform]) {
        queue.write_buffer(&self.joint_transform_buffer, 0, bytemuck::cast_slice(joint_transforms));
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = self.splat_count.div_ceil(64);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skinning_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn skinning_wgsl_shader_compiles() {
        let source = include_str!("skinning.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .expect("WGSL parse error");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator.validate(&module).expect("WGSL validation error");
    }
}
