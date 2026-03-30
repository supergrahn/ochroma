use bytemuck::{Pod, Zeroable};
use wgpu;

/// Spectral depth-of-field with per-band chromatic bokeh.
/// Band 0 (violet) defocuses more than band 7 (red) — Abbe dispersion.
pub struct DofPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    pub focus_distance: f32,
    pub aperture: f32,
    pub focal_length: f32,
    pub abbe_coeff: f32, // default 0.08
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DofParams {
    pub focus_distance: f32,
    pub aperture: f32,
    pub focal_length: f32,
    pub abbe_coeff: f32,
    pub width: u32,
    pub height: u32,
    pub _pad: [u32; 2],
}

impl DofPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dof.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_bgl"),
            entries: &[
                // binding 0: spectral_lo (bands 0-3, read-only)
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
                // binding 1: spectral_hi (bands 4-7, read-only)
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
                // binding 2: depth_buf (read-only)
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
                // binding 3: out_lo (read-write)
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
                // binding 4: out_hi (read-write)
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
                // binding 5: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dof_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dof_compute"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            focus_distance: 10.0,
            aperture: 2.8,
            focal_length: 50.0,
            abbe_coeff: 0.08,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        spectral_input: &wgpu::Buffer, // flat bands 0-3 + 4-7 per pixel
        depth_tex: &wgpu::Texture,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        use wgpu::util::DeviceExt;

        let params = DofParams {
            focus_distance: self.focus_distance,
            aperture: self.aperture,
            focal_length: self.focal_length,
            abbe_coeff: self.abbe_coeff,
            width,
            height,
            _pad: [0; 2],
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // The caller is expected to split spectral_input into lo/hi halves.
        // For simplicity this stub binds the same buffer for both lo and hi
        // as a placeholder; real integration should supply split buffers.
        let pixel_count = (width * height) as u64;
        let half_size   = pixel_count * std::mem::size_of::<[f32; 4]>() as u64;

        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Depth is passed as a texture; we create a temporary storage buffer
        // view by treating depth_view as a texture binding — full integration
        // requires a separate depth linearisation pass.  Here we bind the
        // output buffer for the depth slot as a stub to keep bind group valid.
        let _ = depth_view;

        fn buf_binding(buffer: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset,
                size: Some(std::num::NonZeroU64::new(size).unwrap()),
            })
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_bind_group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_binding(spectral_input, 0, half_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_binding(spectral_input, half_size, half_size),
                },
                // binding 2 (depth_buf) — stub: bind output lo as placeholder
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_binding(output, 0, half_size),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_binding(output, 0, half_size),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_binding(output, half_size, half_size),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let wg_x = width.div_ceil(16);
        let wg_y = height.div_ceil(16);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dof_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::DofParams;

    #[test]
    fn dof_params_size() {
        assert_eq!(std::mem::size_of::<DofParams>(), 32);
    }
}
