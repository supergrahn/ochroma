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
    pub scatter: [f32; 8],   // per-band in-scatter (8 spectral bands)
    pub transmittance: f32,
    pub _pad: [f32; 3],
}

/// 3-D grid of froxel voxels (exponential depth distribution).
pub struct FroxelVolume {
    pub width: u32,   // typically screen_width / 12
    pub height: u32,
    pub depth: u32,   // 64 z-slices
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

        let froxel_byte_size =
            (volume.voxels.len() * std::mem::size_of::<FroxelVoxel>()) as u64;
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
                // 1: froxel buffer (read_only)
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
                // 2: splat output texture (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
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
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scatter_compute.wgsl").into()),
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
            entry_point: Some("scatter_compute"),
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
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.froxel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sdf_buffer.as_entire_binding() },
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
    pub fn dispatch_resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        splat_output_texture: &wgpu::Texture,
        depth_texture: &wgpu::Texture,
    ) {
        let splat_view = splat_output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_resolve_bg"),
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.froxel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&splat_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&depth_view) },
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
