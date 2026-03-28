use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// A key-value pair for GPU sorting.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SortEntry {
    pub depth: f32,
    pub index: u32,
}

/// GPU-based bitonic sort for depth-sorting splats.
pub struct GpuSorter {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuSorter {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("radix_sort.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sort_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bitonic_sort"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bitonic_sort_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout }
    }

    /// Sort entries on the GPU. Returns sorted buffer.
    pub fn sort(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entries: &[SortEntry],
    ) -> Vec<SortEntry> {
        if entries.len() <= 1 { return entries.to_vec(); }

        // Pad to next power of 2
        let n = entries.len().next_power_of_two();
        let mut padded = entries.to_vec();
        while padded.len() < n {
            padded.push(SortEntry { depth: f32::MAX, index: u32::MAX });
        }

        // Create GPU buffer
        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sort_data"),
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Bitonic sort passes
        let total_stages = (n as f32).log2() as u32;
        for stage in 0..total_stages {
            for step in (0..=stage).rev() {
                let params = [n as u32, stage, step, 0]; // [count, stage, step, pad]
                let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sort_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_bg"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: data_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: param_buffer.as_entire_binding() },
                    ],
                });

                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
                }
                queue.submit(std::iter::once(encoder.finish()));
            }
        }

        // Read back (synchronous for now)
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_staging"),
            size: (n * std::mem::size_of::<SortEntry>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging, 0, staging.size());
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let result: Vec<SortEntry> = bytemuck::cast_slice(&data)[..entries.len()].to_vec();
        drop(data);
        staging.unmap();

        result
    }
}
