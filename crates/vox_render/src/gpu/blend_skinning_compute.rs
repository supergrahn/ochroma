//! GPU blend tree skinning — blends up to 4 animation poses before skinning.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::gpu::skinning_compute::{GpuSkinSplat, GpuJointTransform};

const MAX_POSES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlendUniform {
    pub weights: [f32; 4],
    pub joint_count: u32,
    pub _pad: [u32; 3],
}

pub struct BlendSkinningCompute {
    pipeline: wgpu::ComputePipeline,
    _base_splat_buffer: wgpu::Buffer,
    _joint_binding_buffer: wgpu::Buffer,
    pose_buffers: [wgpu::Buffer; MAX_POSES],
    blend_uniform_buffer: wgpu::Buffer,
    pub skinned_splat_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub splat_count: u32,
    pub joint_count: u32,
}

impl BlendSkinningCompute {
    pub fn new(
        device: &wgpu::Device,
        base_splats: &[GpuSkinSplat],
        joint_bindings: &[u32],
        joint_count: usize,
    ) -> Self {
        assert_eq!(base_splats.len(), joint_bindings.len());
        let splat_count = base_splats.len() as u32;
        let jc = joint_count.max(1);
        let identity = GpuJointTransform { skin_matrix: glam::Mat4::IDENTITY.to_cols_array_2d() };
        let identity_data: Vec<GpuJointTransform> = vec![identity; jc];

        let default_splat = [GpuSkinSplat::zeroed()];
        let base_splat_contents: &[GpuSkinSplat] = if base_splats.is_empty() { &default_splat } else { base_splats };
        let base_splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_base_splats"),
            contents: bytemuck::cast_slice(base_splat_contents),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let default_binding = [0u32];
        let joint_binding_contents: &[u32] = if joint_bindings.is_empty() { &default_binding } else { joint_bindings };
        let joint_binding_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_joint_bindings"),
            contents: bytemuck::cast_slice(joint_binding_contents),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pose_buffers: [wgpu::Buffer; MAX_POSES] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("blend_pose_{}", i)),
                contents: bytemuck::cast_slice(&identity_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });

        let blend_uniform = BlendUniform {
            weights: [1.0, 0.0, 0.0, 0.0],
            joint_count: jc as u32,
            _pad: [0; 3],
        };
        let blend_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_uniform"),
            contents: bytemuck::bytes_of(&blend_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let skinned_splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blend_skinned_output"),
            size: ((base_splats.len().max(1) * std::mem::size_of::<GpuSkinSplat>()) as u64).max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend_bgl"),
            entries: &[
                Self::storage_entry(0, true),
                Self::storage_entry(1, true),
                Self::storage_entry(2, true),
                Self::storage_entry(3, true),
                Self::storage_entry(4, true),
                Self::storage_entry(5, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                Self::storage_entry(7, false),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: base_splat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: joint_binding_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pose_buffers[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pose_buffers[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pose_buffers[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pose_buffers[3].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: blend_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: skinned_splat_buffer.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blend_skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blend_skinning.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blend_skinning_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blend_skinning_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_blend_skin"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            _base_splat_buffer: base_splat_buffer,
            _joint_binding_buffer: joint_binding_buffer,
            pose_buffers,
            blend_uniform_buffer,
            skinned_splat_buffer,
            bind_group,
            splat_count,
            joint_count: jc as u32,
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn update_pose(&self, queue: &wgpu::Queue, pose_idx: usize, joints: &[GpuJointTransform]) {
        assert!(pose_idx < MAX_POSES);
        queue.write_buffer(&self.pose_buffers[pose_idx], 0, bytemuck::cast_slice(joints));
    }

    pub fn update_weights(&self, queue: &wgpu::Queue, weights: [f32; 4]) {
        let u = BlendUniform { weights, joint_count: self.joint_count, _pad: [0; 3] };
        queue.write_buffer(&self.blend_uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wgs = self.splat_count.div_ceil(64);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blend_skinning_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wgs, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn blend_skinning_wgsl_parses_and_validates() {
        let src = include_str!("blend_skinning.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        v.validate(&module).expect("WGSL validation error");
    }

    #[test]
    fn blend_uniform_is_pod() {
        use bytemuck::Zeroable;
        let _ = super::BlendUniform::zeroed();
    }
}
