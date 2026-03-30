// splat_raster.rs — Rust host-side driver for the EWA tile rasterizer.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Parameters uploaded to the raster pass as a uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RasterParams {
    pub width:       u32,
    pub height:      u32,
    pub num_tiles_x: u32,
    pub _pad:        u32,
}

/// EWA tile rasterizer compute pass.
///
/// One workgroup per 16×16 pixel tile; each thread handles one pixel.
/// Reads front-to-back sorted splat indices from `sorted_vals` and writes
/// accumulated spectral radiance + transmittance to a 4-layer `rgba32float`
/// texture array:
///   - layer 0: spectral bands 0–3
///   - layer 1: spectral bands 4–7
///   - layer 2: reserved (OIT moments)
///   - layer 3: transmittance in r
pub struct SplatRasterPass {
    pipeline: wgpu::ComputePipeline,
    bgl:      wgpu::BindGroupLayout,
}

impl SplatRasterPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_src = include_str!("splat_raster.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("splat_raster_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("splat_raster_bgl"),
            entries: &[
                // binding 0 — CameraUniform
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
                // binding 1 — array<GpuSplatFull> (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 2 — sorted_vals: array<u32>
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
                // binding 3 — TileRangesBuffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 4 — output texture_storage_2d_array<rgba32float, write>
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:       wgpu::StorageTextureAccess::WriteOnly,
                        format:       wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                // binding 5 — RasterParams uniform
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
            label:                Some("splat_raster_pipeline_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("splat_raster_pipeline"),
            layout:              Some(&layout),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache:               None,
        });

        Self { pipeline, bgl }
    }

    /// Create the 4-layer `rgba32float` output texture array (width × height).
    ///
    /// Layer assignment:
    ///   0 — spectral bands 0–3
    ///   1 — spectral bands 4–7
    ///   2 — reserved (OIT moments)
    ///   3 — transmittance (r channel)
    pub fn create_output_texture(
        device: &wgpu::Device,
        width:  u32,
        height: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label:              Some("splat_raster_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count:    1,
            sample_count:       1,
            dimension:          wgpu::TextureDimension::D2,
            format:             wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats:       &[],
        })
    }

    /// Dispatch the EWA rasterize pass.
    ///
    /// # Parameters
    /// - `camera_buf`    — `CameraUniform` buffer (uniform, ≥192 bytes)
    /// - `splat_buf`     — `array<GpuSplatFull>` storage buffer
    /// - `sorted_vals`   — `array<u32>` splat indices sorted front-to-back per tile
    /// - `tile_ranges`   — `array<vec2<u32>>` start/end per tile; 8 bytes × num_tiles
    /// - `output_texture`— 4-layer `rgba32float` texture from `create_output_texture`
    /// - `params`        — width, height, and num_tiles_x
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device:         &wgpu::Device,
        encoder:        &mut wgpu::CommandEncoder,
        camera_buf:     &wgpu::Buffer,
        splat_buf:      &wgpu::Buffer,
        sorted_vals:    &wgpu::Buffer,
        tile_ranges:    &wgpu::Buffer,
        output_texture: &wgpu::Texture,
        params:         RasterParams,
    ) {
        // Upload RasterParams to a transient uniform buffer.
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("splat_raster_params_buf"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        // Create a full-array view across all 4 layers.
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor {
            label:             Some("splat_raster_output_view"),
            format:            Some(wgpu::TextureFormat::Rgba32Float),
            dimension:         Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer:  0,
            array_layer_count: Some(4),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("splat_raster_bind_group"),
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_vals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tile_ranges.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        // Dispatch: one workgroup per tile.
        let num_tiles_x = params.num_tiles_x;
        let num_tiles_y = params.height.div_ceil(16);

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:              Some("splat_raster_pass"),
            timestamp_writes:   None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(num_tiles_x, num_tiles_y, 1);
    }
}
