use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;

use crate::spectral::RenderCamera;

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
// GpuRasteriser
// ---------------------------------------------------------------------------

/// Rasterises Gaussian splats on the GPU via a wgpu render pipeline.
pub struct GpuRasteriser {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
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

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group_layout: bind_group_layout,
            width,
            height,
        }
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
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
