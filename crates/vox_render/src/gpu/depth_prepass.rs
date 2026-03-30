// depth_prepass.rs — Rust host-side driver for the depth / velocity prepass.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Parameters uploaded to the depth prepass as a uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DepthPrepassParams {
    pub width:       u32,
    pub height:      u32,
    pub splat_count: u32,
    pub _pad:        u32,
}

/// Per-splat depth and screen-space velocity compute pass.
///
/// Each thread processes one splat:
///   1. Projects the splat center through the current-frame view-projection matrix.
///   2. Writes `-clip.w` (view-space depth) to a `r32float` texture.
///   3. Projects the same point through the previous-frame matrix and computes
///      the pixel-space velocity (Δpixels), written to a `rg32float` texture.
pub struct DepthPrepass {
    pipeline: wgpu::ComputePipeline,
    bgl:      wgpu::BindGroupLayout,
}

impl DepthPrepass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_src = include_str!("depth_prepass.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("depth_prepass_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("depth_prepass_bgl"),
            entries: &[
                // binding 0 — current-frame CameraUniform
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 1 — previous-frame CameraUniform
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 2 — array<GpuSplatFull> (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 3 — depth_out: texture_storage_2d<r32float, write>
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 4 — velocity_out: texture_storage_2d<rg32float, write>
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 5 — DepthPrepassParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("depth_prepass_pipeline_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("depth_prepass_pipeline"),
            layout:              Some(&layout),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache:               None,
        });

        Self { pipeline, bgl }
    }

    /// Create the `r32float` depth texture (width × height).
    pub fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label:              Some("depth_prepass_depth_tex"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count:    1,
            sample_count:       1,
            dimension:          wgpu::TextureDimension::D2,
            format:             wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:       &[],
        })
    }

    /// Create the `rg32float` velocity texture (width × height).
    pub fn create_velocity_texture(
        device: &wgpu::Device,
        width:  u32,
        height: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label:              Some("depth_prepass_velocity_tex"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count:    1,
            sample_count:       1,
            dimension:          wgpu::TextureDimension::D2,
            format:             wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:       &[],
        })
    }

    /// Dispatch the depth + velocity prepass.
    ///
    /// # Parameters
    /// - `camera_buf`      — current-frame `CameraUniform` buffer (uniform)
    /// - `prev_camera_buf` — previous-frame `CameraUniform` buffer (uniform)
    /// - `splat_buf`       — `array<GpuSplatFull>` storage buffer
    /// - `depth_tex`       — `r32float` texture from `create_depth_texture`
    /// - `velocity_tex`    — `rg32float` texture from `create_velocity_texture`
    /// - `params`          — width, height, and splat_count
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device:          &wgpu::Device,
        encoder:         &mut wgpu::CommandEncoder,
        camera_buf:      &wgpu::Buffer,
        prev_camera_buf: &wgpu::Buffer,
        splat_buf:       &wgpu::Buffer,
        depth_tex:       &wgpu::Texture,
        velocity_tex:    &wgpu::Texture,
        params:          DepthPrepassParams,
    ) {
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("depth_prepass_params_buf"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor {
            label:     Some("depth_prepass_depth_view"),
            format:    Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let velocity_view = velocity_tex.create_view(&wgpu::TextureViewDescriptor {
            label:     Some("depth_prepass_velocity_view"),
            format:    Some(wgpu::TextureFormat::Rg32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("depth_prepass_bind_group"),
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prev_camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: splat_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: wgpu::BindingResource::TextureView(&velocity_view),
                },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        // Workgroup size is 64; dispatch enough groups to cover all splats.
        let num_groups = params.splat_count.div_ceil(64);

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("depth_prepass_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }
}
