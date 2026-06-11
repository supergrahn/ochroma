//! Rust wrappers for the tile_assign and write_indirect_args compute passes.

/// Buffers produced by a single `TileAssignPass::dispatch` call — or allocated
/// ONCE via [`TileAssignBuffers::allocate`] and reused across frames through
/// [`TileAssignPass::dispatch_into`] (M3.1 Task 2: kills the per-frame
/// `3 × max_tile_entries × 4 B` alloc/free churn — 192–311 MB/frame at city
/// scale).
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

impl TileAssignBuffers {
    /// Allocate the five buffers a `tile_assign` dispatch writes — exactly the
    /// set (labels, usages, `entry_bytes.max(4)` sizing) the per-frame path
    /// used to create inside [`TileAssignPass::dispatch`]. Callers own the
    /// allocation and pass it to [`TileAssignPass::dispatch_into`] every frame;
    /// reuse is sound because `dispatch_into` zero-clears `tile_count` per
    /// encode and downstream passes only read the first `tile_count` entries.
    pub fn allocate(device: &wgpu::Device, max_tile_entries: u32) -> Self {
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

        Self {
            tile_keys_lo,
            tile_keys_hi,
            tile_vals,
            tile_count,
            indirect_args,
        }
    }
}

// ── TileAssignPass ────────────────────────────────────────────────────────────

pub struct TileAssignPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// `write_indirect_args` pipeline, built ONCE here (M3.1 Task 2 — it used
    /// to be a full shader-module → BGL → layout → pipeline build per frame).
    indirect_pipeline: wgpu::ComputePipeline,
    indirect_bgl: wgpu::BindGroupLayout,
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

        // ── write_indirect_args pipeline — built ONCE (M3.1 Task 2) ──────────
        // This entire block used to run inside `dispatch_write_indirect_args`
        // on EVERY frame (naga parse/validate + RADV pipeline create on the
        // hot path). The per-dispatch remainder is just a bind group + the
        // 1-workgroup pass.
        let indirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("write_indirect_args_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("write_indirect_args.wgsl").into()),
        });

        let indirect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let indirect_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("write_indirect_args_layout"),
            bind_group_layouts: &[&indirect_bgl],
            push_constant_ranges: &[],
        });

        let indirect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("write_indirect_args_pipeline"),
                layout: Some(&indirect_layout),
                module: &indirect_shader,
                entry_point: Some("write_indirect_args"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout: bgl,
            indirect_pipeline,
            indirect_bgl,
        }
    }

    /// Dispatch tile assignment for `splat_count` splats.
    ///
    /// Newly allocated output buffers are returned in `TileAssignBuffers`.
    /// Per-frame callers should allocate ONCE via [`TileAssignBuffers::allocate`]
    /// and use [`Self::dispatch_into`] instead — this convenience entry point
    /// pays the full 3 × `max_tile_entries × 4 B` allocation on every call.
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
        let bufs = TileAssignBuffers::allocate(device, max_tile_entries);
        self.dispatch_into(
            device,
            encoder,
            camera_buf,
            splat_buf,
            transform_buf,
            splat_count,
            &bufs,
        );
        bufs
    }

    /// Encode tile assignment against CALLER-OWNED buffers (M3.1 Task 2): the
    /// exact encode path of [`Self::dispatch`] minus the allocation — zero-clear
    /// `bufs.tile_count`, bind, the main `tile_assign` pass, then the cached
    /// `write_indirect_args` pipeline's 1-workgroup sub-pass. Reuse across
    /// frames is sound because the clear re-arms the atomic counter each encode
    /// and downstream passes only read the first `tile_count` entries of the
    /// key/val buffers.
    ///
    /// Still creates both bind groups per call; per-frame callers should cache
    /// them via [`Self::create_bind_group`] / [`Self::create_indirect_bind_group`]
    /// and use [`Self::dispatch_bound`] (M3.1 Task 5) — this method delegates
    /// to it, so the encode is byte-identical either way.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        camera_buf: &wgpu::Buffer,
        splat_buf: &wgpu::Buffer,
        transform_buf: &wgpu::Buffer,
        splat_count: u32,
        bufs: &TileAssignBuffers,
    ) {
        let bind_group = self.create_bind_group(device, camera_buf, splat_buf, transform_buf, bufs);
        let indirect_bg = self.create_indirect_bind_group(device, bufs);
        self.dispatch_bound(encoder, splat_count, &bind_group, &indirect_bg, bufs);
    }

    /// Build the main `tile_assign` bind group over caller-owned buffers
    /// (M3.1 Task 5 — additive). Per-frame callers build this ONCE (and again
    /// whenever `bufs` or the camera/splat/transform buffers are recreated)
    /// and pass it to [`Self::dispatch_bound`] so the hot path creates zero
    /// bind groups.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        splat_buf: &wgpu::Buffer,
        transform_buf: &wgpu::Buffer,
        bufs: &TileAssignBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_assign_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.tile_keys_lo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.tile_keys_hi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.tile_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.tile_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: transform_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Build the 2-entry `write_indirect_args` bind group over `bufs`
    /// (M3.1 Task 5 — additive). It uses a DIFFERENT layout than the main
    /// bind group, which is why [`Self::dispatch_bound`] takes both cached
    /// handles: bind-group creation needs the device, and the bound path
    /// deliberately has no device parameter.
    pub fn create_indirect_bind_group(
        &self,
        device: &wgpu::Device,
        bufs: &TileAssignBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("write_indirect_args_bg"),
            layout: &self.indirect_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.tile_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.indirect_args.as_entire_binding(),
                },
            ],
        })
    }

    /// [`Self::dispatch_into`] against CALLER-CACHED bind groups (M3.1
    /// Task 5): zero-clear `bufs.tile_count`, the main `tile_assign` pass,
    /// then the cached `write_indirect_args` pipeline's 1-workgroup
    /// sub-pass — identical command stream, zero per-call resource creation.
    /// `bind_group` / `indirect_bind_group` must come from
    /// [`Self::create_bind_group`] / [`Self::create_indirect_bind_group`]
    /// over the SAME `bufs`.
    pub fn dispatch_bound(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        splat_count: u32,
        bind_group: &wgpu::BindGroup,
        indirect_bind_group: &wgpu::BindGroup,
        bufs: &TileAssignBuffers,
    ) {
        // Zero the tile_count so it can be used as an atomic.
        encoder.clear_buffer(&bufs.tile_count, 0, None);

        let wg_x = splat_count.div_ceil(256);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_assign_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }

        // ── write_indirect_args sub-pass (cached pipeline) ────────────────────
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("write_indirect_args_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.indirect_pipeline);
        pass.set_bind_group(0, indirect_bind_group, &[]);
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
        // MUST validate, not just parse: a dimension-invalid expression (e.g.
        // mat3x3 * vec2) parses fine but fails validation, and discarding the
        // result let exactly that bug ship undriven. Assert it succeeds.
        v.validate(&module)
            .expect("tile_assign.wgsl must pass naga validation, not just parse");
    }

    #[test]
    fn write_indirect_args_wgsl_parses() {
        let src = include_str!("write_indirect_args.wgsl");
        let module =
            naga::front::wgsl::parse_str(src).expect("write_indirect_args.wgsl parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .expect("write_indirect_args.wgsl must pass naga validation");
    }
}
