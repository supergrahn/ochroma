//! SDF soft shadow compute pass.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SdfUniform {
    pub origin: [f32; 3],
    pub _pad0: f32,
    pub voxel_size: f32,
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    pub light_dir: [f32; 3],
    pub penumbra_k: f32,
    pub max_dist: f32,
    pub _pad1: [f32; 3],
}

pub struct SdfShadowPass {
    pipeline: wgpu::ComputePipeline,
    _sdf_buffer: wgpu::Buffer,
    sdf_uniform_buffer: wgpu::Buffer,
    pub shadow_buffer: wgpu::Buffer,
    _sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
}

impl SdfShadowPass {
    pub fn new(
        device: &wgpu::Device,
        sdf_data: &[f32],
        sdf_uniform: SdfUniform,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let sdf_data_safe = if sdf_data.is_empty() { &[0.0f32] as &[f32] } else { sdf_data };

        let sdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_shadow_sdf"),
            contents: bytemuck::cast_slice(sdf_data_safe),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sdf_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_shadow_uniform"),
            contents: bytemuck::bytes_of(&sdf_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pixel_count = ((width * height) as u64).max(1);
        let shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_shadow_output"),
            size: pixel_count * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sdf_shadow_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_shadow_bgl"),
            entries: &[
                // binding 0: sdf_params uniform
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
                // binding 1: sdf_data storage
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
                // binding 2: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 4: shadow output
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_shadow_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sdf_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sdf_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: shadow_buffer.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sdf_shadow.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_shadow_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_sdf_shadow"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            _sdf_buffer: sdf_buffer,
            sdf_uniform_buffer,
            shadow_buffer,
            _sampler: sampler,
            bind_group,
            width,
            height,
        }
    }

    pub fn update_sdf(&self, queue: &wgpu::Queue, sdf_data: &[f32]) {
        queue.write_buffer(&self._sdf_buffer, 0, bytemuck::cast_slice(sdf_data));
    }

    pub fn update_uniform(&self, queue: &wgpu::Queue, uniform: &SdfUniform) {
        queue.write_buffer(&self.sdf_uniform_buffer, 0, bytemuck::bytes_of(uniform));
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sdf_shadow_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sdf_shadow_wgsl_parses() {
        let src = include_str!("sdf_shadow.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let _ = v.validate(&module);
    }

    #[test]
    fn sdf_uniform_is_pod() {
        use bytemuck::Zeroable;
        let _ = super::SdfUniform::zeroed();
    }
}
