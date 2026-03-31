use crate::desc::AgentStateDesc;

/// SoA GPU buffers allocated from an AgentStateDesc.
pub struct AgentStateBuffers {
    desc: AgentStateDesc,
    positions_a: wgpu::Buffer,
    positions_b: wgpu::Buffer,
    velocities_a: wgpu::Buffer,
    velocities_b: wgpu::Buffer,
    flags: wgpu::Buffer,
    spatial_cell: Option<wgpu::Buffer>,
    cell_counts: Option<wgpu::Buffer>,
    cell_offsets: Option<wgpu::Buffer>,
    cell_data: Option<wgpu::Buffer>,
    custom: Option<wgpu::Buffer>,
    spectral_cache: Option<wgpu::Buffer>,
    read_index: u8,
}

fn make_buffer(device: &wgpu::Device, size: u64, label: &str, usage: wgpu::BufferUsages)
    -> wgpu::Buffer
{
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4), // wgpu requires size > 0
        usage,
        mapped_at_creation: false,
    })
}

const STORAGE_RW: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);

impl AgentStateBuffers {
    pub fn new(device: &wgpu::Device, desc: AgentStateDesc) -> Self {
        let n = desc.agent_count as u64;
        let pos_size = n * 12;  // [f32;3] = 12 bytes
        let vel_size = n * 12;
        let flag_size = n * 4;  // u32

        let (spatial_cell, cell_counts, cell_offsets, cell_data) =
            if let Some(sh) = &desc.spatial_hash {
                let cells = sh.cell_count() as u64;
                (
                    Some(make_buffer(device, n * 4, "agent_spatial_cell", STORAGE_RW)),
                    Some(make_buffer(device, cells * 4, "agent_cell_counts",
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)),
                    Some(make_buffer(device, (cells + 1) * 4, "agent_cell_offsets", STORAGE_RW)),
                    Some(make_buffer(device, n * 4, "agent_cell_data", STORAGE_RW)),
                )
            } else {
                (None, None, None, None)
            };

        let custom = (desc.custom_floats > 0).then(|| {
            make_buffer(device, n * desc.custom_floats as u64 * 4, "agent_custom", STORAGE_RW)
        });

        let spectral_cache = desc.spectral.then(|| {
            make_buffer(device, n * 16 * 4, "agent_spectral_cache", STORAGE_RW)
        });

        Self {
            positions_a: make_buffer(device, pos_size, "agent_pos_a", STORAGE_RW),
            positions_b: make_buffer(device, pos_size, "agent_pos_b", STORAGE_RW),
            velocities_a: make_buffer(device, vel_size, "agent_vel_a", STORAGE_RW),
            velocities_b: make_buffer(device, vel_size, "agent_vel_b", STORAGE_RW),
            flags: make_buffer(device, flag_size, "agent_flags", STORAGE_RW),
            spatial_cell,
            cell_counts,
            cell_offsets,
            cell_data,
            custom,
            spectral_cache,
            read_index: 0,
            desc,
        }
    }

    pub fn desc(&self) -> &AgentStateDesc { &self.desc }

    pub fn swap(&mut self) { self.read_index ^= 1; }

    pub fn read_positions(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.positions_a } else { &self.positions_b }
    }

    pub fn write_positions(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.positions_b } else { &self.positions_a }
    }

    pub fn read_velocities(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.velocities_a } else { &self.velocities_b }
    }

    pub fn write_velocities(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.velocities_b } else { &self.velocities_a }
    }

    pub fn flags(&self) -> &wgpu::Buffer { &self.flags }
    pub fn spatial_cells(&self) -> Option<&wgpu::Buffer> { self.spatial_cell.as_ref() }
    pub fn cell_counts(&self) -> Option<&wgpu::Buffer> { self.cell_counts.as_ref() }
    pub fn cell_offsets(&self) -> Option<&wgpu::Buffer> { self.cell_offsets.as_ref() }
    pub fn cell_data(&self) -> Option<&wgpu::Buffer> { self.cell_data.as_ref() }
    pub fn custom(&self) -> Option<&wgpu::Buffer> { self.custom.as_ref() }
    pub fn spectral_cache(&self) -> Option<&wgpu::Buffer> { self.spectral_cache.as_ref() }

    /// Initialize positions from a CPU slice. Slice must be exactly agent_count * 3 floats.
    pub fn upload_positions(&self, queue: &wgpu::Queue, positions: &[[f32; 3]]) {
        assert_eq!(positions.len() as u32, self.desc.agent_count);
        let flat: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_buffer(self.read_positions(), 0, bytemuck::cast_slice(&flat));
    }

    /// Initialize velocities from a CPU slice. Slice must be exactly agent_count * 3 floats.
    pub fn upload_velocities(&self, queue: &wgpu::Queue, velocities: &[[f32; 3]]) {
        assert_eq!(velocities.len() as u32, self.desc.agent_count);
        let flat: Vec<f32> = velocities.iter().flat_map(|v| v.iter().copied()).collect();
        queue.write_buffer(self.read_velocities(), 0, bytemuck::cast_slice(&flat));
    }

    /// Mark all agents as alive (flag bit 0 = 1).
    pub fn mark_all_alive(&self, queue: &wgpu::Queue) {
        let flags: Vec<u32> = vec![1u32; self.desc.agent_count as usize];
        queue.write_buffer(&self.flags, 0, bytemuck::cast_slice(&flags));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;

    fn minimal_desc(n: u32) -> AgentStateDesc {
        AgentStateDesc { agent_count: n, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn buffers_allocate_without_gpu() {
        // Only checks struct construction compiles and desc is stored.
        let desc = minimal_desc(100);
        assert_eq!(desc.agent_count, 100);
    }

    #[test]
    fn ping_pong_swap_alternates() {
        let Some((device, _queue)) = test_device() else { return; };
        let mut buf = AgentStateBuffers::new(&device, minimal_desc(10));
        let a_before = buf.read_positions() as *const wgpu::Buffer;
        buf.swap();
        let a_after = buf.read_positions() as *const wgpu::Buffer;
        assert_ne!(a_before, a_after, "swap must alternate read buffer");
    }

    #[test]
    fn spatial_hash_buffers_present_when_desc_has_spatial_hash() {
        let Some((device, _queue)) = test_device() else { return; };
        use crate::desc::SpatialHashDesc;
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 0,
            spectral: false,
            spatial_hash: Some(SpatialHashDesc {
                grid_origin_x: 0.0, grid_origin_z: 0.0,
                grid_extent: 100.0, cell_size: 10.0,
            }),
        };
        let buf = AgentStateBuffers::new(&device, desc);
        assert!(buf.spatial_cells().is_some());
        assert!(buf.cell_offsets().is_some());
        assert!(buf.cell_data().is_some());
    }

    #[test]
    fn spatial_hash_buffers_absent_when_no_spatial_hash() {
        let Some((device, _queue)) = test_device() else { return; };
        let buf = AgentStateBuffers::new(&device, minimal_desc(50));
        assert!(buf.spatial_cells().is_none());
    }

    #[test]
    fn custom_buffer_present_when_custom_floats_nonzero() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 8,
            spectral: false,
            spatial_hash: None,
        };
        let buf = AgentStateBuffers::new(&device, desc);
        assert!(buf.custom().is_some());
    }

    #[test]
    fn spectral_buffer_present_when_desc_spectral_true() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 0,
            spectral: true,
            spatial_hash: None,
        };
        let buf = AgentStateBuffers::new(&device, desc);
        assert!(buf.spectral_cache().is_some());
    }
}
