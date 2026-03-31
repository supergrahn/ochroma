use bytemuck::bytes_of;
use crate::desc::AgentStateDesc;
use crate::state::AgentStateBuffers;
use crate::uniforms::AgentUniforms;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("shader compilation failed: {0}")]
    Compilation(String),
    #[error("bind group creation failed: {0}")]
    BindGroup(String),
}

/// Behavior shader source.
pub enum ShaderSource {
    Wgsl(String),
}

pub struct AgentComputePipeline {
    pipeline:      wgpu::ComputePipeline,
    bgl:           wgpu::BindGroupLayout,
    uniform_buf:   wgpu::Buffer,
    desc_snapshot: AgentStateDesc,
}

fn base_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
    fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
    vec![
        storage(0, true),   // positions_in
        storage(1, false),  // positions_out
        storage(2, true),   // velocities_in
        storage(3, false),  // velocities_out
        storage(4, false),  // agent_flags
        uniform(5),         // uniforms
    ]
}

/// Returns the WGSL binding declarations for a given descriptor.
/// Game shaders can paste this at the top of their source.
pub fn layout_source(desc: &AgentStateDesc) -> String {
    let mut out = String::from(
r#"struct AgentUniforms {
    agent_count:  u32,
    custom_floats: u32,
    dt:            f32,
    time:          f32,
    grid_width:    u32,
    cell_size:     f32,
    _pad0:         f32,
    _pad1:         f32,
}
@group(0) @binding(0) var<storage, read>       positions_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> positions_out: array<f32>;
@group(0) @binding(2) var<storage, read>       velocities_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocities_out:array<f32>;
@group(0) @binding(4) var<storage, read_write> agent_flags:   array<u32>;
@group(0) @binding(5) var<uniform>             uniforms:      AgentUniforms;
"#);
    let mut next_binding = 6u32;
    if desc.spatial_hash.is_some() {
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> spatial_cells:  array<u32>;\n",
            next_binding));
        next_binding += 1;
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> cell_offsets:   array<u32>;\n",
            next_binding));
        next_binding += 1;
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> cell_data:      array<u32>;\n",
            next_binding));
        next_binding += 1;
    }
    if desc.custom_floats > 0 {
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read_write> custom: array<f32>;\n",
            next_binding));
        next_binding += 1;
    }
    if desc.spectral {
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> spectral_samples: array<f32>;\n",
            next_binding));
    }
    out
}

impl AgentComputePipeline {
    pub fn new(
        device: &wgpu::Device,
        source: ShaderSource,
        desc: &AgentStateDesc,
    ) -> Result<Self, PipelineError> {
        let ShaderSource::Wgsl(wgsl) = source;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agent_behavior"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let mut bgl_entries = base_bgl_entries();
        let mut next = 6u32;
        let storage = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        if desc.spatial_hash.is_some() {
            bgl_entries.push(storage(next,     true));
            bgl_entries.push(storage(next + 1, true));
            bgl_entries.push(storage(next + 2, true));
            next += 3;
        }
        if desc.custom_floats > 0 {
            bgl_entries.push(storage(next, false));
            next += 1;
        }
        if desc.spectral {
            bgl_entries.push(storage(next, true));
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_behavior_bgl"),
            entries: &bgl_entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_behavior_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("agent_behavior_pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("agent_update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_uniforms"),
            size: std::mem::size_of::<AgentUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self { pipeline, bgl, uniform_buf, desc_snapshot: desc.clone() })
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        buffers: &AgentStateBuffers,
        spectral_samples: Option<&wgpu::Buffer>,
        uniforms: AgentUniforms,
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytes_of(&uniforms));

        let mut entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry { binding: 0, resource: buffers.read_positions().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffers.write_positions().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffers.read_velocities().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buffers.write_velocities().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buffers.flags().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.uniform_buf.as_entire_binding() },
        ];
        let mut next = 6u32;
        if self.desc_snapshot.spatial_hash.is_some() {
            entries.push(wgpu::BindGroupEntry { binding: next,
                resource: buffers.spatial_cells().unwrap().as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: next + 1,
                resource: buffers.cell_offsets().unwrap().as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: next + 2,
                resource: buffers.cell_data().unwrap().as_entire_binding() });
            next += 3;
        }
        if self.desc_snapshot.custom_floats > 0 {
            entries.push(wgpu::BindGroupEntry { binding: next,
                resource: buffers.custom().unwrap().as_entire_binding() });
            next += 1;
        }
        if self.desc_snapshot.spectral {
            if let Some(s) = spectral_samples {
                entries.push(wgpu::BindGroupEntry { binding: next,
                    resource: s.as_entire_binding() });
            }
        }

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_behavior_bg"),
            layout: &self.bgl,
            entries: &entries,
        });

        let n = uniforms.agent_count;
        let workgroups = (n + 63) / 64;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("agent_update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;
    use crate::desc::AgentStateDesc;
    use crate::state::AgentStateBuffers;
    use crate::uniforms::AgentUniforms;

    fn minimal_desc(n: u32) -> AgentStateDesc {
        AgentStateDesc { agent_count: n, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn default_shader_loads_successfully() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = minimal_desc(100);
        let wgsl = include_str!("../shaders/default_agent.wgsl").to_string();
        let result = AgentComputePipeline::new(&device, ShaderSource::Wgsl(wgsl), &desc);
        assert!(result.is_ok(), "default shader must load: {:?}", result.err());
    }

    #[test]
    fn layout_source_contains_correct_bindings() {
        let desc = minimal_desc(10);
        let src = layout_source(&desc);
        assert!(src.contains("@binding(0)"), "must have binding 0 (positions_in)");
        assert!(src.contains("@binding(5)"), "must have binding 5 (uniforms)");
        assert!(!src.contains("@binding(6)"), "no binding 6 without spatial hash");
    }

    #[test]
    fn layout_source_includes_spatial_hash_bindings_when_desc_has_it() {
        use crate::desc::SpatialHashDesc;
        let desc = AgentStateDesc {
            agent_count: 10, custom_floats: 0, spectral: false,
            spatial_hash: Some(SpatialHashDesc {
                grid_origin_x: 0.0, grid_origin_z: 0.0,
                grid_extent: 100.0, cell_size: 10.0,
            }),
        };
        let src = layout_source(&desc);
        assert!(src.contains("spatial_cells"),  "must declare spatial_cells");
        assert!(src.contains("cell_offsets"),   "must declare cell_offsets");
        assert!(src.contains("cell_data"),      "must declare cell_data");
    }

    #[test]
    fn default_shader_dispatches_and_integrates_velocity() {
        let Some((device, queue)) = test_device() else { return; };
        let desc = minimal_desc(4);
        let buffers = AgentStateBuffers::new(&device, desc.clone());

        buffers.upload_positions(&queue, &[[0.0, 0.0, 0.0]; 4]);
        buffers.upload_velocities(&queue, &[[1.0, 0.0, 0.0]; 4]);
        buffers.mark_all_alive(&queue);

        let wgsl = include_str!("../shaders/default_agent.wgsl").to_string();
        let pipeline = AgentComputePipeline::new(&device, ShaderSource::Wgsl(wgsl), &desc)
            .expect("pipeline");

        let uniforms = AgentUniforms {
            agent_count: 4, custom_floats: 0,
            dt: 1.0, time: 0.0,
            grid_width: 0, cell_size: 1.0,
            _pad: [0.0; 2],
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        pipeline.dispatch(&device, &mut encoder, &queue, &buffers, None, uniforms);

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 4 * 3 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffers.write_positions(), 0, &readback, 0, 4 * 3 * 4);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        assert!((data[0] - 1.0).abs() < 1e-5,
            "x position after dt=1: expected 1.0, got {}", data[0]);
        assert!((data[1]).abs() < 1e-5, "y position unchanged: got {}", data[1]);
        assert!((data[2]).abs() < 1e-5, "z position unchanged: got {}", data[2]);
    }
}
