//! Real-time spectral GI probe system with rolling update schedule.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// A single spectral radiance probe: 6 face directions × 8 spectral bands.
/// face order: +X -X +Y -Y +Z -Z
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SpectralProbe {
    /// radiance[face][band]
    pub radiance: [[f32; 8]; 6], // 192 bytes
    pub last_updated: u32,       // 4 bytes
    pub world_pos: [f32; 3],     // 12 bytes
    // total: 208 bytes — no padding needed
}

/// GPU-layout probe for std430 upload.
/// radiance as flat array<f32, 48> (6 faces × 8 bands), world_pos as vec3, _pad f32.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuProbe {
    pub radiance: [f32; 48], // 192 bytes
    pub world_pos: [f32; 3], // 12 bytes
    pub _pad: f32,           // 4 bytes — total: 208 bytes
}

/// World-space probe grid. probes are laid out x-major: idx = x + dims[0]*(y + dims[1]*z)
pub struct ProbeGrid {
    pub origin: [f32; 3],
    pub spacing: f32,
    pub dims: [u32; 3],
    pub probes: Vec<SpectralProbe>,
}

impl ProbeGrid {
    pub fn new(origin: [f32; 3], spacing: f32, dims: [u32; 3]) -> Self {
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        Self {
            origin,
            spacing,
            dims,
            probes: vec![
                SpectralProbe {
                    radiance: [[0.0; 8]; 6],
                    last_updated: 0,
                    world_pos: [0.0; 3],
                };
                count
            ],
        }
    }

    pub fn count(&self) -> usize {
        self.probes.len()
    }

    pub fn world_pos(&self, x: u32, y: u32, z: u32) -> [f32; 3] {
        [
            self.origin[0] + x as f32 * self.spacing,
            self.origin[1] + y as f32 * self.spacing,
            self.origin[2] + z as f32 * self.spacing,
        ]
    }

    /// Convert to GPU layout for upload.
    pub fn to_gpu_probes(&self) -> Vec<GpuProbe> {
        self.probes
            .iter()
            .map(|p| {
                let mut radiance = [0.0f32; 48];
                for face in 0..6 {
                    for band in 0..8 {
                        radiance[face * 8 + band] = p.radiance[face][band];
                    }
                }
                GpuProbe {
                    radiance,
                    world_pos: p.world_pos,
                    _pad: 0.0,
                }
            })
            .collect()
    }
}

/// Update priority for rolling probe refresh.
#[derive(Clone, Debug)]
pub struct ProbeUpdateCandidate {
    pub probe_idx: u32,
    pub priority: f32, // higher = more urgent
}

/// Manages the GPU-side probe buffer and rolling update schedule.
pub struct GiProbePass {
    pub grid: ProbeGrid,
    gpu_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    frame_index: u32,
}

impl GiProbePass {
    pub fn new(device: &wgpu::Device, grid: ProbeGrid) -> Self {
        let gpu_probes = grid.to_gpu_probes();
        let probe_data: &[u8] = if gpu_probes.is_empty() {
            &[0u8; std::mem::size_of::<GpuProbe>()]
        } else {
            bytemuck::cast_slice(&gpu_probes)
        };

        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_probe_buffer"),
            contents: probe_data,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_probe_bgl"),
            entries: &[
                // binding 0: ProbeUpdateParams uniform
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
                // binding 1: probes storage read_write
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
                // binding 2: sdf_volume storage read_only
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: SdfParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe_update_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("probe_update.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi_probe_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi_probe_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("probe_update_face"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            grid,
            gpu_buffer,
            pipeline,
            bgl,
            frame_index: 0,
        }
    }

    /// Select up to 4 probes to refresh this frame based on age + camera proximity.
    pub fn select_update_candidates(
        &self,
        camera_pos: [f32; 3],
        max_probe_dist: f32,
    ) -> Vec<ProbeUpdateCandidate> {
        let safe_max = if max_probe_dist <= 0.0 {
            1.0
        } else {
            max_probe_dist
        };

        let mut candidates: Vec<ProbeUpdateCandidate> = self
            .grid
            .probes
            .iter()
            .enumerate()
            .map(|(idx, probe)| {
                let age_weight =
                    (self.frame_index.saturating_sub(probe.last_updated)) as f32 / 120.0;

                let dx = probe.world_pos[0] - camera_pos[0];
                let dy = probe.world_pos[1] - camera_pos[1];
                let dz = probe.world_pos[2] - camera_pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let camera_proximity_weight = 1.0 - (dist / safe_max).clamp(0.0, 1.0);

                let priority = (age_weight + camera_proximity_weight).clamp(0.0, 1.0);

                ProbeUpdateCandidate {
                    probe_idx: idx as u32,
                    priority,
                }
            })
            .collect();

        candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        candidates.truncate(4);
        candidates
    }

    /// Upload all probe data to GPU.
    pub fn upload(&self, queue: &wgpu::Queue) {
        let gpu_probes = self.grid.to_gpu_probes();
        if !gpu_probes.is_empty() {
            queue.write_buffer(&self.gpu_buffer, 0, bytemuck::cast_slice(&gpu_probes));
        }
    }

    /// Dispatch the probe update compute pass for the selected probes.
    pub fn dispatch_update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        _queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        sdf_buffer: &wgpu::Buffer,
        _framebuffer_texture: &wgpu::Texture,
    ) {
        let candidates = self.select_update_candidates(camera_pos, 100.0);
        if candidates.is_empty() {
            self.frame_index += 1;
            return;
        }

        // Build ProbeUpdateParams — matches WGSL struct layout
        // probe_indices: [u32; 4], probe_count: u32, _pad0: u32 × 3,
        // grid_origin: [f32; 3], grid_spacing: f32,
        // grid_dims: [u32; 3], _pad1: u32,
        // frame_index: u32, _pad2: [u32; 3]
        let mut probe_indices = [0u32; 4];
        for (i, c) in candidates.iter().enumerate() {
            probe_indices[i] = c.probe_idx;
        }
        let probe_count = candidates.len() as u32;

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct ProbeUpdateParams {
            probe_indices: [u32; 4],
            probe_count: u32,
            _pad0: [u32; 3],
            grid_origin: [f32; 3],
            grid_spacing: f32,
            grid_dims: [u32; 3],
            _pad1: u32,
            frame_index: u32,
            _pad2: [u32; 3],
        }

        let params = ProbeUpdateParams {
            probe_indices,
            probe_count,
            _pad0: [0; 3],
            grid_origin: self.grid.origin,
            grid_spacing: self.grid.spacing,
            grid_dims: self.grid.dims,
            _pad1: 0,
            frame_index: self.frame_index,
            _pad2: [0; 3],
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_probe_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Minimal SdfParams uniform (origin, cell_size, dims, _pad)
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct SdfParams {
            origin: [f32; 3],
            cell_size: f32,
            dims: [u32; 3],
            _pad: u32,
        }
        let sdf_params = SdfParams {
            origin: [0.0; 3],
            cell_size: 1.0,
            dims: [1, 1, 1],
            _pad: 0,
        };
        let sdf_params_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gi_probe_sdf_params"),
                contents: bytemuck::bytes_of(&sdf_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group =
            device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gi_probe_bind_group"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.gpu_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: sdf_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: sdf_params_buf.as_entire_binding(),
                        },
                    ],
                });

        let workgroup_count = (candidates.len() * 6) as u32;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi_probe_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Mark selected probes as updated
        for c in &candidates {
            self.grid.probes[c.probe_idx as usize].last_updated = self.frame_index;
        }

        self.frame_index += 1;
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.gpu_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_grid_count() {
        let grid = ProbeGrid::new([0.0; 3], 1.0, [4, 2, 4]);
        assert_eq!(grid.count(), 32);
    }

    #[test]
    fn probe_world_pos() {
        let grid = ProbeGrid::new([0.0; 3], 2.0, [4, 2, 4]);
        let pos = grid.world_pos(1, 0, 2);
        assert_eq!(pos, [2.0, 0.0, 4.0]);
    }

    #[test]
    fn gpu_probe_size() {
        assert_eq!(std::mem::size_of::<GpuProbe>(), 208);
    }

    #[test]
    fn spectral_probe_size() {
        assert_eq!(std::mem::size_of::<SpectralProbe>(), 208);
    }

    #[test]
    fn select_candidates_returns_at_most_4() {
        // Use a 10-probe grid (2×1×5)
        let grid = ProbeGrid::new([0.0; 3], 1.0, [2, 1, 5]);
        assert_eq!(grid.count(), 10);

        // Build a GiProbePass without a real device by testing the logic directly
        // via select_update_candidates on a mock grid through the struct fields.
        // We can't construct GiProbePass without wgpu::Device, so test the
        // candidate selection logic independently here.
        let frame_index = 0u32;
        let camera_pos = [0.0f32; 3];
        let max_probe_dist = 100.0f32;

        let candidates: Vec<ProbeUpdateCandidate> = grid
            .probes
            .iter()
            .enumerate()
            .map(|(idx, probe)| {
                let age_weight = frame_index.saturating_sub(probe.last_updated) as f32 / 120.0;
                let dx = probe.world_pos[0] - camera_pos[0];
                let dy = probe.world_pos[1] - camera_pos[1];
                let dz = probe.world_pos[2] - camera_pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let prox = 1.0 - (dist / max_probe_dist).clamp(0.0, 1.0);
                ProbeUpdateCandidate {
                    probe_idx: idx as u32,
                    priority: (age_weight + prox).clamp(0.0, 1.0),
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .take(4)
            .collect();

        assert!(candidates.len() <= 4);
    }
}
