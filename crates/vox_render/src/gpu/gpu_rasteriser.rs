use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;

use crate::spectral::RenderCamera;

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });
    (texture, view, sampler)
}

// ---------------------------------------------------------------------------
// GPU-side data structures (std430-compatible)
// ---------------------------------------------------------------------------

/// GPU-side splat data matching the WGSL `SplatData` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSplatData {
    pub position: [f32; 3],
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub opacity: f32,
    pub _pad: f32,
    pub spectral: [f32; 8],
}

/// Camera uniform matching the WGSL `CameraUniform` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub _pad: [f32; 2],
}

// ---------------------------------------------------------------------------
// Helper: convert GaussianSplat slice to GPU format (no sorting)
// ---------------------------------------------------------------------------

/// Convert a slice of [`GaussianSplat`] to [`GpuSplatData`] without depth sorting.
pub fn splats_to_gpu(splats: &[GaussianSplat]) -> Vec<GpuSplatData> {
    splats
        .iter()
        .map(|s| {
            let mut spectral = [0.0f32; 8];
            for b in 0..8 {
                spectral[b] = half::f16::from_bits(s.spectral[b]).to_f32();
            }
            GpuSplatData {
                position: s.position,
                scale_x: s.scale[0],
                scale_y: s.scale[1],
                scale_z: s.scale[2],
                opacity: s.opacity as f32 / 255.0,
                _pad: 0.0,
                spectral,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GpuRasteriser
// ---------------------------------------------------------------------------

/// Rasterises Gaussian splats on the GPU via a wgpu render pipeline.
pub struct GpuRasteriser {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_sampler: wgpu::Sampler,

    // Shadow pass fields
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    light_buffer: Option<wgpu::Buffer>,
    shadow_depth_texture: Option<wgpu::Texture>,
    shadow_depth_view: Option<wgpu::TextureView>,
    shadow_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl GpuRasteriser {
    /// Create a new rasteriser.
    ///
    /// `device`         - wgpu device
    /// `surface_format` - texture format of the render target
    /// `width`/`height` - viewport dimensions in pixels
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // Load WGSL shader source compiled into the binary.
        let shader_src = include_str!("splat_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout: binding 0 = camera uniform, binding 1 = splat storage
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // billboards are always camera-facing
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // back-to-front sorted, no depth buffer
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Camera uniform buffer (will be overwritten each frame)
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform_buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (depth_texture, depth_view, depth_sampler) = create_depth_texture(device, width, height);

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group_layout: bind_group_layout,
            width,
            height,
            depth_texture,
            depth_view,
            depth_sampler,
            shadow_pipeline: None,
            light_buffer: None,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            shadow_bind_group_layout: None,
        }
    }

    /// Initialise the shadow depth prepass pipeline and resources.
    ///
    /// Must be called once after `new()` before using `render_with_shadow`.
    pub fn init_shadow_pass(&mut self, device: &wgpu::Device) {
        // Create 512×512 shadow depth texture
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_depth_texture"),
            size: wgpu::Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Light uniform buffer (4×4 f32 matrix = 64 bytes)
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_uniform_buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout: binding 0 = light uniform, binding 1 = splat storage
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader_src = include_str!("shadow_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_shadow"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.shadow_pipeline = Some(shadow_pipeline);
        self.light_buffer = Some(light_buf);
        self.shadow_depth_texture = Some(shadow_texture);
        self.shadow_depth_view = Some(shadow_view);
        self.shadow_bind_group_layout = Some(bgl);
    }

    /// Run the shadow depth prepass for the given splats and light matrix.
    pub fn render_shadow_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        splats_gpu: &[GpuSplatData],
        light_view_proj: &glam::Mat4,
    ) {
        let (Some(pipeline), Some(light_buf), Some(shadow_depth_view), Some(bgl)) = (
            self.shadow_pipeline.as_ref(),
            self.light_buffer.as_ref(),
            self.shadow_depth_view.as_ref(),
            self.shadow_bind_group_layout.as_ref(),
        ) else {
            return;
        };

        if splats_gpu.is_empty() {
            return;
        }

        // Write light matrix to uniform buffer
        let mat_data: [[f32; 4]; 4] = light_view_proj.to_cols_array_2d();
        queue.write_buffer(light_buf, 0, bytemuck::cast_slice(&mat_data));

        // Per-frame splat storage buffer
        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow_splat_storage_buffer"),
            contents: bytemuck::cast_slice(splats_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shadow_encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_render_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..(splats_gpu.len() as u32 * 6), 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render with an optional shadow pass before the main render.
    ///
    /// If `light_view_proj` is `Some`, the shadow depth prepass runs first.
    pub fn render_with_shadow(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        illuminant: &Illuminant,
        light_view_proj: Option<&glam::Mat4>,
    ) {
        if splats.is_empty() {
            return;
        }

        // Convert splats once for both passes
        let gpu_splats = splats_to_gpu(splats);

        if let Some(lvp) = light_view_proj {
            self.render_shadow_pass(device, queue, &gpu_splats, lvp);
        }

        self.render(device, queue, target_view, splats, camera, illuminant);
    }

    /// Render the given splats to `target_view`.
    ///
    /// Splats are depth-sorted on the CPU (back-to-front) and uploaded each frame.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        _illuminant: &Illuminant,
    ) {
        if splats.is_empty() {
            return;
        }

        // --- 1. Write camera uniform ---
        let view = camera.view;
        let view_proj = camera.view_proj();
        let inv_view = view.inverse();

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            inv_view: inv_view.to_cols_array_2d(),
            viewport_size: [self.width as f32, self.height as f32],
            _pad: [0.0; 2],
        };

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // --- 2. CPU depth sort (back-to-front) ---
        // Compute view-space Z for each splat and sort most negative first (farthest).
        let mut indexed: Vec<(usize, f32)> = splats
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let pos = glam::Vec4::new(s.position[0], s.position[1], s.position[2], 1.0);
                let view_pos = view * pos;
                (i, view_pos.z)
            })
            .collect();

        // Sort by view-space Z ascending (most negative = farthest first for RH)
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- 3. Convert sorted splats to GpuSplatData ---
        let gpu_splats: Vec<GpuSplatData> = indexed
            .iter()
            .map(|&(i, _)| {
                let s = &splats[i];
                let mut spectral = [0.0f32; 8];
                for b in 0..8 {
                    spectral[b] = f16::from_bits(s.spectral[b]).to_f32();
                }
                GpuSplatData {
                    position: s.position,
                    scale_x: s.scale[0],
                    scale_y: s.scale[1],
                    scale_z: s.scale[2],
                    opacity: s.opacity as f32 / 255.0,
                    _pad: 0.0,
                    spectral,
                }
            })
            .collect();

        // --- 4. Create storage buffer with splat data ---
        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat_storage_buffer"),
            contents: bytemuck::cast_slice(&gpu_splats),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // --- 5. Create bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_bind_group"),
            layout: &self.camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_buffer.as_entire_binding(),
                },
            ],
        });

        // --- 6. Create render pass ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("splat_render_encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);

            // --- 7. Draw: 6 vertices per quad, one instance per splat ---
            let splat_count = gpu_splats.len() as u32;
            render_pass.draw(0..6, 0..splat_count);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Update stored viewport dimensions (e.g. after a window resize).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (tex, view, sampler) = create_depth_texture(device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
        self.depth_sampler = sampler;
    }

    /// Return a reference to the depth texture view.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_texture_descriptor_format_is_depth32float() {
        assert_eq!(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureFormat::Depth32Float,
        );
    }
}
