use wgpu;

/// Spectral dual-Kawase bloom.
/// Per-band extraction with per-band thresholds (bands 0-1 have lower threshold).
/// 6 levels of downsample + upsample per band group (0-3 and 4-7).
pub struct BloomPass {
    downsample_pipeline: wgpu::ComputePipeline,
    upsample_pipeline: wgpu::ComputePipeline,
    combine_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    pub strength: f32,
    pub threshold_low: f32,  // for bands 0-1 (high-freq, UV-range)
    pub threshold_high: f32, // for bands 2-7
    num_levels: u32,
}

impl BloomPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bloom.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bgl"),
            entries: &[
                // binding 0: input_lo (bands 0-3, read-only storage)
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
                // binding 1: input_hi (bands 4-7, read-only storage)
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
                // binding 2: output_lo (read-write storage)
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
                // binding 3: output_hi (read-write storage)
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
                // binding 4: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("bloom_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bloom_downsample_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bloom_downsample"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let upsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bloom_upsample_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bloom_upsample"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let combine_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bloom_combine_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bloom_extract"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            downsample_pipeline,
            upsample_pipeline,
            combine_pipeline,
            bgl,
            strength: 0.3,
            threshold_low: 0.5,
            threshold_high: 1.0,
            num_levels: 6,
        }
    }

    /// Dispatch bloom on the spectral framebuffer.
    /// `input_texture` is the Rgba32Float texture array (4 layers, 8 bands).
    /// The bloom is added back into `input_texture` in-place.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        _device: &wgpu::Device,
        _input_texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        // Full dispatch implementation would allocate mip-level buffers and
        // iterate downsample -> upsample per level.  Stubbed for now; the
        // pipeline objects are created and wgpu validates the shader on
        // device creation.
        let _ = (encoder, width, height, self.num_levels);
        let _ = &self.downsample_pipeline;
        let _ = &self.upsample_pipeline;
        let _ = &self.combine_pipeline;
        let _ = &self.bgl;
    }
}

impl super::super::postprocess::PostProcessPass for BloomPass {
    fn name(&self) -> &'static str {
        "Bloom"
    }

    fn execute(&self, ctx: &mut super::super::postprocess::PostProcessContext) {
        // Bloom operates on a spectral texture array; here we forward to
        // dispatch once a concrete spectral texture handle is plumbed through
        // PostProcessContext.  For now the pass is wired up correctly but the
        // actual dispatch is a no-op pending that plumbing.
        let _ = ctx;
    }
}

#[cfg(test)]
mod tests {
    /// The pass name must be the string literal "Bloom" so that
    /// GpuPostProcessPipeline::set_enabled("Bloom", …) can address it.
    #[test]
    fn bloom_pass_name() {
        // We cannot call BloomPass::new() without a wgpu::Device, so we
        // verify the name constant directly from the impl.
        assert_eq!(
            <super::BloomPass as crate::postprocess::PostProcessPass>::name(
                // SAFETY: we are only calling a &'static str method that does
                // not dereference any fields — the struct is never constructed.
                // We use a transmuted zero-sized reference trick so that Rust
                // doesn't require an actual BloomPass value.
                unsafe { &*(std::ptr::NonNull::dangling().as_ptr() as *const super::BloomPass) }
            ),
            "Bloom"
        );
    }
}
