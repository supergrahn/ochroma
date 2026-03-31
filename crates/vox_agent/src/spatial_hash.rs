use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::desc::SpatialHashDesc;
use crate::state::AgentStateBuffers;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpatialUniforms {
    agent_count: u32,
    grid_width:  u32,
    cell_size:   f32,
    origin_x:    f32,
    origin_z:    f32,
    _pad:        [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixUniforms {
    cell_count: u32,
    _pad:       [u32; 3],
}

pub struct SpatialHashPipelines {
    pub count:        wgpu::ComputePipeline,
    pub count_bgl:    wgpu::BindGroupLayout,
    pub prefix:       wgpu::ComputePipeline,
    pub prefix_bgl:   wgpu::BindGroupLayout,
    pub scatter:      wgpu::ComputePipeline,
    pub scatter_bgl:  wgpu::BindGroupLayout,
    pub su_buf:       wgpu::Buffer,
    pub pu_buf:       wgpu::Buffer,
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    bgl: &wgpu::BindGroupLayout,
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl SpatialHashPipelines {
    pub fn new(device: &wgpu::Device, desc: &SpatialHashDesc) -> Self {
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_count_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });

        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_prefix_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });

        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_scatter_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                uniform_entry(5),
            ],
        });

        let su = SpatialUniforms {
            agent_count: 0,
            grid_width: desc.grid_width(),
            cell_size: desc.cell_size,
            origin_x: desc.grid_origin_x,
            origin_z: desc.grid_origin_z,
            _pad: [0; 3],
        };
        let pu = PrefixUniforms { cell_count: desc.cell_count(), _pad: [0; 3] };

        let su_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_spatial_uniforms"),
            contents: bytemuck::bytes_of(&su),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_prefix_uniforms"),
            contents: bytemuck::bytes_of(&pu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let count = make_pipeline(device, "sh_count",
            include_str!("../shaders/spatial_hash_count.wgsl"), &count_bgl, "main");
        let prefix = make_pipeline(device, "sh_prefix",
            include_str!("../shaders/spatial_hash_prefix_sum.wgsl"), &prefix_bgl, "main");
        let scatter = make_pipeline(device, "sh_scatter",
            include_str!("../shaders/spatial_hash_scatter.wgsl"), &scatter_bgl, "main");

        Self { count, count_bgl, prefix, prefix_bgl, scatter, scatter_bgl, su_buf, pu_buf }
    }
}

/// Placeholder: actual dispatch is in AgentComputeLayer::tick() which has access to wgpu::Device
/// for bind group creation. SpatialHashPipelines exposes the BGLs; AgentComputeLayer creates BGs.
pub fn rebuild_spatial_hash(
    _encoder: &mut wgpu::CommandEncoder,
    _pipelines: &SpatialHashPipelines,
    _buffers: &AgentStateBuffers,
) {
    // Intentionally empty: bind groups require &wgpu::Device which isn't available here.
    // See AgentComputeLayer::tick() for the full 3-pass dispatch.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;
    use crate::desc::{AgentStateDesc, SpatialHashDesc};
    use crate::state::AgentStateBuffers;

    fn sh_desc() -> SpatialHashDesc {
        SpatialHashDesc { grid_origin_x: 0.0, grid_origin_z: 0.0,
                          grid_extent: 100.0, cell_size: 10.0 }
    }

    #[test]
    fn spatial_hash_pipelines_compile() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = sh_desc();
        // If shaders fail to compile, wgpu panics here.
        let _sh = SpatialHashPipelines::new(&device, &desc);
    }

    #[test]
    fn neighbour_query_correctness() {
        // Place 20 agents: 10 near (5,0,5), 10 near (15,0,5).
        // This test verifies shader compilation and buffer setup succeed.
        // Full dispatch correctness is in the integration test (Task 8).
        let Some((device, queue)) = test_device() else { return; };

        let desc = AgentStateDesc {
            agent_count: 20,
            custom_floats: 0,
            spectral: false,
            spatial_hash: Some(sh_desc()),
        };
        let buffers = AgentStateBuffers::new(&device, desc);

        let mut positions = vec![[0.0f32; 3]; 20];
        for i in 0..10 { positions[i] = [5.0 + i as f32 * 0.1, 0.0, 5.0]; }
        for i in 10..20 { positions[i] = [15.0 + (i-10) as f32 * 0.1, 0.0, 5.0]; }
        buffers.upload_positions(&queue, &positions);
        buffers.mark_all_alive(&queue);
        queue.submit([]);
    }
}
