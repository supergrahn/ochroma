//! Rust wrappers for the tile_assign and write_indirect_args compute passes.

/// Buffers produced by a single `TileAssignPass::dispatch` call.
pub struct TileAssignBuffers {
    /// Low 32 bits of each sort key  (reinterpreted depth bits).
    pub tile_keys_lo: wgpu::Buffer,
    /// High 32 bits of each sort key (tile index).
    pub tile_keys_hi: wgpu::Buffer,
    /// Splat index for each emitted tile entry.
    pub tile_vals: wgpu::Buffer,
    /// Atomic counter — total tile entries written.
    pub tile_count: wgpu::Buffer,
    /// Indirect dispatch args `[wg_x, 1, 1]` for the next pass.
    pub indirect_args: wgpu::Buffer,
}

// ── TileAssignPass ────────────────────────────────────────────────────────────

pub struct TileAssignPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl TileAssignPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_assign_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tile_assign.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_assign_bgl"),
            entries: &[
                // 0 — CameraUniform
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
                // 1 — GpuSplatFull SSBO (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2 — tile_keys_lo
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
                // 3 — tile_keys_hi
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4 — tile_vals
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5 — tile_count (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6 — splat_transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile_assign_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile_assign_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("tile_assign"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout: bgl }
    }

    /// Dispatch tile assignment for `splat_count` splats.
    ///
    /// Newly allocated output buffers are returned in `TileAssignBuffers`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        camera_buf: &wgpu::Buffer,
        splat_buf: &wgpu::Buffer,
        transform_buf: &wgpu::Buffer,
        splat_count: u32,
        max_tile_entries: u32,
    ) -> TileAssignBuffers {
        let entry_bytes = max_tile_entries as u64 * std::mem::size_of::<u32>() as u64;
        let entry_bytes = entry_bytes.max(4); // never allocate 0-byte buffers

        let tile_keys_lo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_keys_lo"),
            size: entry_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tile_keys_hi = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_keys_hi"),
            size: entry_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tile_vals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_vals"),
            size: entry_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tile_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_count"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // indirect_args: 3 × u32 for DispatchIndirectArgs
        let indirect_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_assign_indirect_args"),
            size: 3 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero the tile_count so it can be used as an atomic.
        encoder.clear_buffer(&tile_count, 0, None);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_assign_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tile_keys_lo.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tile_keys_hi.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tile_vals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: tile_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: transform_buf.as_entire_binding() },
            ],
        });

        let wg_x = splat_count.div_ceil(256);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_assign_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }

        // ── write_indirect_args sub-pass ──────────────────────────────────────
        Self::dispatch_write_indirect_args(device, encoder, &tile_count, &indirect_args);

        TileAssignBuffers {
            tile_keys_lo,
            tile_keys_hi,
            tile_vals,
            tile_count,
            indirect_args,
        }
    }

    fn dispatch_write_indirect_args(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        tile_count_buf: &wgpu::Buffer,
        indirect_args_buf: &wgpu::Buffer,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("write_indirect_args_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("write_indirect_args.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("write_indirect_args_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("write_indirect_args_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("write_indirect_args_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("write_indirect_args"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("write_indirect_args_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tile_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: indirect_args_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("write_indirect_args_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    /// Verify that both WGSL sources can be parsed by naga without errors.
    #[test]
    fn tile_assign_wgsl_parses() {
        let src = include_str!("tile_assign.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("tile_assign.wgsl parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let _ = v.validate(&module);
    }

    #[test]
    fn write_indirect_args_wgsl_parses() {
        let src = include_str!("write_indirect_args.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .expect("write_indirect_args.wgsl parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let _ = v.validate(&module);
    }
}
