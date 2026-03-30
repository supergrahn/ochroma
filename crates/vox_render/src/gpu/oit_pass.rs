//! Moment-based Order-Independent Transparency (OIT).
//!
//! Implements Hamburger 2018 moment-based OIT:
//! - `OitAccumulatePass` — accumulates power moments of depth distribution.
//! - `OitResolvePass`    — composites transparent layer over opaque using moments.
//!
//! Moment texture format:     `Rgba32Float` (b0, b1, b2, b3).
//! Transmittance texture format: `R32Float`.
//! Both require `STORAGE_BINDING | TEXTURE_BINDING`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Shared uniform
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OitParams {
    pub width: u32,
    pub height: u32,
    pub transparent_count: u32,
    pub _pad: u32,
}

// ---------------------------------------------------------------------------
// Texture helpers
// ---------------------------------------------------------------------------

const MOMENTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const TRANSMITTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

fn oit_texture_usage() -> wgpu::TextureUsages {
    wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING
}

// ---------------------------------------------------------------------------
// OitAccumulatePass
// ---------------------------------------------------------------------------

pub struct OitAccumulatePass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl OitAccumulatePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_accumulate_bgl"),
            entries: &[
                // 0: camera uniform
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
                // 1: transparent splat buffer (read_only storage)
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
                // 2: sorted transparent index values (read_only storage)
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
                // 3: moments texture (rgba32float, read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: MOMENTS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: transmittance texture (r32float, read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: TRANSMITTANCE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 5: OIT params uniform
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_accumulate_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("oit_accumulate.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_accumulate_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oit_accumulate_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("oit_accumulate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }

    /// Create the OIT moment textures for width×height.
    /// Returns (moments_tex, transmittance_tex).
    pub fn create_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };

        let moments_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oit_moments_tex"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MOMENTS_FORMAT,
            usage: oit_texture_usage(),
            view_formats: &[],
        });

        let transmittance_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oit_transmittance_tex"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TRANSMITTANCE_FORMAT,
            usage: oit_texture_usage(),
            view_formats: &[],
        });

        (moments_tex, transmittance_tex)
    }

    /// Dispatch OIT accumulation for transparent splats.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        camera_buf: &wgpu::Buffer,
        transparent_splat_buf: &wgpu::Buffer,
        sorted_transparent_vals: &wgpu::Buffer,
        moments_tex: &wgpu::Texture,
        transmittance_tex: &wgpu::Texture,
        params: OitParams,
    ) {
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("oit_params_tmp"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let moments_view = moments_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let transmittance_view =
            transmittance_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_accumulate_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: transparent_splat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sorted_transparent_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&moments_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        let count = params.transparent_count.max(1);
        let wg_x = count.div_ceil(256);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("oit_accumulate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg_x, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// OitResolvePass
// ---------------------------------------------------------------------------

pub struct OitResolvePass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl OitResolvePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_resolve_bgl"),
            entries: &[
                // 0: opaque spectral texture (rgba32float, read_only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: MOMENTS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 1: moments texture (rgba32float, read_only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: MOMENTS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 2: transmittance texture (r32float, read_only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TRANSMITTANCE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 3: output texture (rgba32float, write_only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: MOMENTS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: OIT params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("oit_resolve.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_resolve_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oit_resolve_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("oit_resolve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }

    /// Dispatch OIT resolve: composite transparent over opaque using moments.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        opaque_texture: &wgpu::Texture,
        moments_tex: &wgpu::Texture,
        transmittance_tex: &wgpu::Texture,
        output_texture: &wgpu::Texture,
        params: OitParams,
    ) {
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("oit_resolve_params_tmp"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let opaque_view = opaque_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let moments_view = moments_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let transmittance_view =
            transmittance_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_resolve_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&moments_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let pixel_count = params.width * params.height;
        let wg_x = pixel_count.max(1).div_ceil(256);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("oit_resolve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg_x, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oit_params_size() {
        assert_eq!(std::mem::size_of::<OitParams>(), 16);
    }
}
