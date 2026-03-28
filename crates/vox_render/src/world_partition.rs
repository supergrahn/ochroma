use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkState {
    Unloaded,
    Loading,
    Loaded,
    Active,
    Streaming,
}

#[derive(Debug, Clone)]
pub struct WorldPartition {
    pub chunks: Vec<Vec<ChunkInfo>>,
    pub chunk_size: f32,
    pub grid_width: usize,
    pub grid_height: usize,
    pub active_center: [i32; 2],
    pub load_radius: i32,
}

#[derive(Debug, Clone)]
pub struct ChunkInfo {
    pub grid_pos: [i32; 2],
    pub state: ChunkState,
    pub entity_count: u32,
    pub splat_count: u32,
    pub memory_mb: f32,
}

impl WorldPartition {
    pub fn new(grid_w: usize, grid_h: usize, chunk_size: f32) -> Self {
        let mut chunks = Vec::with_capacity(grid_h);
        for gz in 0..grid_h {
            let mut row = Vec::with_capacity(grid_w);
            for gx in 0..grid_w {
                row.push(ChunkInfo {
                    grid_pos: [gx as i32, gz as i32],
                    state: ChunkState::Unloaded,
                    entity_count: 0,
                    splat_count: 0,
                    memory_mb: 0.0,
                });
            }
            chunks.push(row);
        }
        Self {
            chunks,
            chunk_size,
            grid_width: grid_w,
            grid_height: grid_h,
            active_center: [0, 0],
            load_radius: 2,
        }
    }

    pub fn update_camera(&mut self, camera_x: f32, camera_z: f32) {
        let cx = (camera_x / self.chunk_size).floor() as i32;
        let cz = (camera_z / self.chunk_size).floor() as i32;
        self.active_center = [cx, cz];

        for row in &mut self.chunks {
            for chunk in row {
                let dx = (chunk.grid_pos[0] - cx).abs();
                let dz = (chunk.grid_pos[1] - cz).abs();
                let dist = dx.max(dz);
                chunk.state = if dist == 0 {
                    ChunkState::Active
                } else if dist <= self.load_radius {
                    ChunkState::Loaded
                } else {
                    ChunkState::Unloaded
                };
            }
        }
    }

    pub fn get_chunk(&self, gx: usize, gz: usize) -> Option<&ChunkInfo> {
        self.chunks.get(gz).and_then(|row| row.get(gx))
    }

    pub fn active_chunks(&self) -> Vec<&ChunkInfo> {
        self.chunks
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| c.state == ChunkState::Active)
            .collect()
    }

    pub fn loaded_chunks(&self) -> Vec<&ChunkInfo> {
        self.chunks
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| c.state == ChunkState::Loaded || c.state == ChunkState::Active)
            .collect()
    }

    pub fn total_loaded_memory_mb(&self) -> f32 {
        self.chunks
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| c.state == ChunkState::Loaded || c.state == ChunkState::Active)
            .map(|c| c.memory_mb)
            .sum()
    }

    pub fn chunk_at_world(&self, world_x: f32, world_z: f32) -> Option<[i32; 2]> {
        let gx = (world_x / self.chunk_size).floor() as i32;
        let gz = (world_z / self.chunk_size).floor() as i32;
        if gx >= 0 && gx < self.grid_width as i32 && gz >= 0 && gz < self.grid_height as i32 {
            Some([gx, gz])
        } else {
            None
        }
    }

    /// Generate a 2D color map for the editor overlay.
    /// Returns a flat array of RGB colors, one per chunk.
    pub fn generate_minimap(&self) -> Vec<[u8; 3]> {
        self.chunks
            .iter()
            .flat_map(|row| {
                row.iter().map(|chunk| match chunk.state {
                    ChunkState::Active => [0, 200, 0],
                    ChunkState::Loaded => [0, 100, 200],
                    ChunkState::Loading => [200, 200, 0],
                    ChunkState::Streaming => [200, 100, 0],
                    ChunkState::Unloaded => [50, 50, 50],
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_partition() {
        let wp = WorldPartition::new(8, 8, 64.0);
        assert_eq!(wp.grid_width, 8);
        assert_eq!(wp.grid_height, 8);
        assert_eq!(wp.chunks.len(), 8);
        assert_eq!(wp.chunks[0].len(), 8);
        // All chunks start unloaded
        for row in &wp.chunks {
            for chunk in row {
                assert_eq!(chunk.state, ChunkState::Unloaded);
            }
        }
    }

    #[test]
    fn camera_update_activates_nearby() {
        let mut wp = WorldPartition::new(8, 8, 64.0);
        wp.load_radius = 1;
        // Place camera in chunk [3, 3]
        wp.update_camera(3.0 * 64.0 + 10.0, 3.0 * 64.0 + 10.0);
        assert_eq!(wp.active_center, [3, 3]);
        // Center chunk should be Active
        assert_eq!(wp.get_chunk(3, 3).unwrap().state, ChunkState::Active);
        // Adjacent chunks should be Loaded
        assert_eq!(wp.get_chunk(2, 3).unwrap().state, ChunkState::Loaded);
        assert_eq!(wp.get_chunk(4, 3).unwrap().state, ChunkState::Loaded);
        assert_eq!(wp.get_chunk(3, 2).unwrap().state, ChunkState::Loaded);
    }

    #[test]
    fn far_chunks_unloaded() {
        let mut wp = WorldPartition::new(8, 8, 64.0);
        wp.load_radius = 1;
        wp.update_camera(0.0, 0.0);
        // Far corner should be Unloaded
        assert_eq!(wp.get_chunk(7, 7).unwrap().state, ChunkState::Unloaded);
        assert_eq!(wp.get_chunk(5, 5).unwrap().state, ChunkState::Unloaded);
    }

    #[test]
    fn minimap_colors_correct() {
        let mut wp = WorldPartition::new(4, 4, 64.0);
        wp.load_radius = 0;
        wp.update_camera(64.0, 64.0); // center at [1,1]
        let minimap = wp.generate_minimap();
        assert_eq!(minimap.len(), 16);
        // Chunk [1,1] = row 1, col 1 = index 5 should be green (Active)
        assert_eq!(minimap[5], [0, 200, 0]);
        // Chunk [0,0] = index 0 should be grey (Unloaded)
        assert_eq!(minimap[0], [50, 50, 50]);
    }

    #[test]
    fn memory_tracking_works() {
        let mut wp = WorldPartition::new(4, 4, 64.0);
        wp.load_radius = 0;
        // Set memory on a couple chunks
        wp.chunks[1][1].memory_mb = 32.0;
        wp.chunks[0][0].memory_mb = 16.0;
        // Before camera update, all unloaded -> 0 memory
        assert_eq!(wp.total_loaded_memory_mb(), 0.0);
        // Activate chunk [1,1]
        wp.update_camera(64.0 + 10.0, 64.0 + 10.0);
        // Now [1,1] is Active
        assert!((wp.total_loaded_memory_mb() - 32.0).abs() < 0.001);
    }
}
