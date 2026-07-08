//! Froxel volumetric lighting pass — scatter + resolve compute stages.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// CPU-side froxel data structures
// ---------------------------------------------------------------------------

/// One froxel voxel: per-spectral-band in-scatter + transmittance.
/// Layout: 8×f32 scatter + f32 transmittance + 3×f32 pad = 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FroxelVoxel {
    pub scatter: [f32; 8], // per-band in-scatter (8 spectral bands)
    pub transmittance: f32,
    pub _pad: [f32; 3],
}

/// 3-D grid of froxel voxels (exponential depth distribution).
pub struct FroxelVolume {
    pub width: u32, // typically screen_width / 12
    pub height: u32,
    pub depth: u32, // 64 z-slices
    pub voxels: Vec<FroxelVoxel>,
}

impl FroxelVolume {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
            voxels: vec![
                FroxelVoxel {
                    scatter: [0.0; 8],
                    transmittance: 1.0,
                    _pad: [0.0; 3],
                };
                (width * height * depth) as usize
            ],
        }
    }

    /// Flat index for voxel (x, y, z).
    pub fn idx(&self, x: u32, y: u32, z: u32) -> usize {
        (x + self.width * (y + self.height * z)) as usize
    }

    /// World-space depth for z-slice k using exponential distribution.
    pub fn slice_z(&self, k: u32, z_near: f32, z_far: f32) -> f32 {
        z_near * (z_far / z_near).powf(k as f32 / self.depth as f32)
    }
}

// ---------------------------------------------------------------------------
// GPU uniform
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VolumetricParams {
    pub sun_direction: [f32; 3],
    pub mie_coeff: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub froxel_width: u32,
    pub froxel_height: u32,
    pub froxel_depth: u32,
    pub _pad: [u32; 3],
}

// ---------------------------------------------------------------------------
// GPU pass
// ---------------------------------------------------------------------------

pub struct VolumetricPass {
    scatter_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    scatter_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    froxel_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    pub volume: FroxelVolume,
}

impl VolumetricPass {
    pub fn new(
        device: &wgpu::Device,
        froxel_width: u32,
        froxel_height: u32,
        froxel_depth: u32,
    ) -> Self {
        let volume = FroxelVolume::new(froxel_width, froxel_height, froxel_depth);

        let froxel_byte_size = (volume.voxels.len() * std::mem::size_of::<FroxelVoxel>()) as u64;
        let froxel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volumetric_froxel_buffer"),
            size: froxel_byte_size.max(std::mem::size_of::<FroxelVoxel>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_params = VolumetricParams {
            sun_direction: [0.0, 1.0, 0.0],
            mie_coeff: 0.01,
            z_near: 0.1,
            z_far: 1000.0,
            froxel_width,
            froxel_height,
            froxel_depth,
            _pad: [0u32; 3],
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("volumetric_params_buffer"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- scatter BGL -------------------------------------------------
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_scatter_bgl"),
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
                // 1: volumetric params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: froxel buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: sdf volume (read_only)
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
            ],
        });

        // ---- resolve BGL -------------------------------------------------
        // Binding 2 is WRITE-only: baseline wgpu rejects read-write
        // rgba32float storage textures (`create_bind_group_layout` validation,
        // the error the regression test pinned), so the pre-resolve colour is
        // read through binding 4 — a copy `dispatch_resolve` records before
        // the compute pass. Binding 5 is the pass's own params uniform (froxel
        // dims + z range), which the resolve march needs.
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_resolve_bgl"),
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
                // 1: froxel buffer (read-only)
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
                // 2: splat output texture (storage, write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 3: depth texture (sample)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: pre-resolve colour copy (sampled, textureLoad only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: volumetric params uniform
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

        // ---- scatter pipeline --------------------------------------------
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scatter_compute.wgsl").into()),
        });
        let scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_scatter_layout"),
            bind_group_layouts: &[&scatter_bgl],
            push_constant_ranges: &[],
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("volumetric_scatter_pipeline"),
            layout: Some(&scatter_layout),
            module: &scatter_shader,
            entry_point: Some("scatter_compute"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ---- resolve pipeline --------------------------------------------
        // Built from its own shader module: the resolve entry cannot share
        // scatter_compute.wgsl — that module's globals at bindings 1–3
        // (uniform / storage-rw / storage-ro) conflict with resolve_bgl's
        // types, which is exactly the latent wrong-entry-point bug this
        // replaces (the old code compiled scatter_compute.wgsl with entry
        // point "scatter_compute" against resolve_bgl).
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("volumetric_resolve.wgsl").into()),
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_resolve_layout"),
            bind_group_layouts: &[&resolve_bgl],
            push_constant_ranges: &[],
        });
        let resolve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("volumetric_resolve_pipeline"),
            layout: Some(&resolve_layout),
            module: &resolve_shader,
            entry_point: Some("volumetric_resolve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            scatter_pipeline,
            resolve_pipeline,
            scatter_bgl,
            resolve_bgl,
            froxel_buffer,
            params_buffer,
            volume,
        }
    }

    /// Upload current params to GPU.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &VolumetricParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the scatter compute pass (fills froxel buffer).
    pub fn dispatch_scatter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        sdf_buffer: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_scatter_bg"),
            layout: &self.scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.froxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sdf_buffer.as_entire_binding(),
                },
            ],
        });

        let wg_x = self.volume.width.div_ceil(8);
        let wg_y = self.volume.height.div_ceil(8);
        let wg_z = self.volume.depth.div_ceil(4);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("volumetric_scatter_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.scatter_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    /// Dispatch the resolve pass (applies froxel attenuation to spectral framebuffer).
    ///
    /// `splat_output_texture` must be Rgba32Float with
    /// `STORAGE_BINDING | COPY_SRC` usage: baseline wgpu forbids read-write
    /// rgba32float storage textures, so the pre-resolve colour is copied to a
    /// scratch texture first and the compute pass reads the copy while writing
    /// the original in place. The scratch texture is created per call — this
    /// pass is not frame-coupled yet (zero render-loop callers); the first
    /// real frame loop should cache it.
    pub fn dispatch_resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        splat_output_texture: &wgpu::Texture,
        depth_texture: &wgpu::Texture,
    ) {
        let size = splat_output_texture.size();
        let scene_copy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volumetric_resolve_scene_copy"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        encoder.copy_texture_to_texture(
            splat_output_texture.as_image_copy(),
            scene_copy.as_image_copy(),
            size,
        );

        let splat_view = splat_output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let scene_copy_view = scene_copy.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_resolve_bg"),
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.froxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&splat_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&scene_copy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let wg_x = self.volume.width.div_ceil(8);
        let wg_y = self.volume.height.div_ceil(8);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("volumetric_resolve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu_context(label: &str) -> Option<crate::gpu::GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("[{label}] no adapter — skipping GPU test");
                return None;
            }
        };
        let info = adapter.get_info();
        if crate::gpu::adapter::ensure_hardware(&info).is_err() {
            eprintln!(
                "[{label}] software adapter ({}) — skipping GPU test",
                info.name
            );
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("volumetric_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(crate::gpu::GpuContext::from_parts(&device, &queue, &info))
    }

    /// Regression test for the wrong-entry-point resolve pipeline: the resolve
    /// pipeline used to be built from `scatter_compute.wgsl` with entry point
    /// `scatter_compute` against `resolve_bgl`, whose bindings 1/2/3
    /// (storage-ro / Rgba32Float storage texture / depth texture) do not match
    /// that shader's globals (uniform / storage-rw / storage-ro) — so
    /// `create_compute_pipeline` raised a wgpu validation error on any real
    /// device. Latent until now because `VolumetricPass` had zero constructors
    /// anywhere in the workspace.
    #[test]
    fn volumetric_pass_constructs_on_device() {
        let Some(ctx) = try_gpu_context("volumetric") else {
            return;
        };
        let device = ctx.device();

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _pass = VolumetricPass::new(device, 16, 9, 64);
        let err = pollster::block_on(device.pop_error_scope());
        assert!(
            err.is_none(),
            "VolumetricPass::new raised a wgpu validation error: {}",
            err.map(|e| e.to_string()).unwrap_or_default()
        );
        println!(
            "[volumetric] constructed scatter+resolve pipelines on {}",
            ctx.adapter_name()
        );
    }

    /// Resolve readback gate: a 1.0-seeded Rgba32Float target marched through
    /// a fog-filled froxel volume must come back darker, and monotonically
    /// darker as `mie_coeff` rises (more extinction → lower transmittance).
    #[test]
    fn volumetric_resolve_attenuates() {
        let Some(ctx) = try_gpu_context("volumetric") else {
            return;
        };
        let device = ctx.device();
        let queue = ctx.queue();

        let (fw, fh, fd) = (16u32, 9u32, 8u32);
        let (tex_w, tex_h) = (64u32, 36u32);
        let pass = VolumetricPass::new(device, fw, fh, fd);

        // Zeroed CameraUniform (3×mat4 + 2×vec2 = 208 B): the scatter pass's
        // world position only feeds the dummy-SDF miss path, and the resolve
        // marches in froxel space — neither needs a real view matrix here.
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volumetric_test_camera"),
            size: 208,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // Dummy 1-float SDF buffer: the scatter binding-3 producer (the
        // Tier-2 clipmap) is a later wave — scatter hard-codes dims=(1,1,1),
        // so this binding contributes nothing real yet.
        let sdf_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("volumetric_test_dummy_sdf"),
            contents: bytemuck::bytes_of(&1.0f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let extent = wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        };
        let splat_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volumetric_test_splat_target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volumetric_test_depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let row_bytes = tex_w * 16; // 1024 — already COPY_BYTES_PER_ROW aligned
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volumetric_test_readback"),
            size: (row_bytes * tex_h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let seed: Vec<f32> = vec![1.0; (tex_w * tex_h * 4) as usize];

        let mean_luma = |mie_coeff: f32| -> f32 {
            pass.update_params(
                queue,
                &VolumetricParams {
                    sun_direction: [0.0, 1.0, 0.0],
                    mie_coeff,
                    z_near: 0.1,
                    z_far: 1000.0,
                    froxel_width: fw,
                    froxel_height: fh,
                    froxel_depth: fd,
                    _pad: [0u32; 3],
                },
            );
            queue.write_texture(
                splat_tex.as_image_copy(),
                bytemuck::cast_slice(&seed),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(tex_h),
                },
                extent,
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("volumetric_test_encoder"),
            });
            // Clear depth to 1.0 (far plane) — the march runs the full column.
            {
                let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("volumetric_test_depth_clear"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            pass.dispatch_scatter(&mut encoder, device, &camera_buf, &sdf_buf);
            pass.dispatch_resolve(&mut encoder, device, &camera_buf, &splat_tex, &depth_tex);
            encoder.copy_texture_to_buffer(
                splat_tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_bytes),
                        rows_per_image: Some(tex_h),
                    },
                },
                extent,
            );
            queue.submit([encoder.finish()]);

            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).expect("map_async result channel");
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .expect("map_async callback ran")
                .expect("readback buffer mapped");
            let mean = {
                let data = slice.get_mapped_range();
                let texels: &[f32] = bytemuck::cast_slice(&data);
                let mut sum = 0.0f64;
                let mut n = 0usize;
                for px in texels.chunks_exact(4) {
                    sum += ((px[0] + px[1] + px[2]) / 3.0) as f64;
                    n += 1;
                }
                (sum / n as f64) as f32
            };
            readback.unmap();
            mean
        };

        let luma_low = mean_luma(0.01);
        let luma_high = mean_luma(0.2);
        println!("[volumetric] luma mie0.01={luma_low:.4} mie0.2={luma_high:.4}");
        assert!(
            0.0 < luma_high && luma_high < luma_low && luma_low < 1.0,
            "expected 0 < mie0.2 luma < mie0.01 luma < 1 (a luma of 1.0 means \
             the resolve wrote nothing): low={luma_low} high={luma_high}"
        );
    }

    #[test]
    fn froxel_voxel_size() {
        assert_eq!(std::mem::size_of::<FroxelVoxel>(), 48);
    }

    #[test]
    fn froxel_slice_z_near() {
        let vol = FroxelVolume::new(16, 9, 64);
        let z = vol.slice_z(0, 0.1, 1000.0);
        assert!((z - 0.1).abs() < 1e-5, "expected ~0.1, got {z}");
    }

    #[test]
    fn froxel_slice_z_far() {
        let vol = FroxelVolume::new(16, 9, 64);
        let z = vol.slice_z(64, 0.1, 1000.0);
        assert!((z - 1000.0).abs() < 1e-3, "expected ~1000.0, got {z}");
    }
}
