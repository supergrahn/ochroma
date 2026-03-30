//! Cell-based spatial hash for O(k) neighbour queries in crowd simulation.

use glam::Vec3;
use std::collections::HashMap;

pub struct SpatialHash {
    pub cell_size: f32,
    buckets: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self { cell_size, buckets: HashMap::new() }
    }

    fn cell(&self, pos: Vec3) -> (i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    pub fn clear(&mut self) {
        for v in self.buckets.values_mut() {
            v.clear();
        }
    }

    pub fn insert(&mut self, idx: usize, pos: Vec3) {
        self.buckets.entry(self.cell(pos)).or_default().push(idx);
    }

    pub fn neighbours(&self, pos: Vec3, _radius: f32) -> Vec<usize> {
        let (cx, cz) = self.cell(pos);
        let mut result = Vec::new();
        for dx in -1..=1i32 {
            for dz in -1..=1i32 {
                if let Some(bucket) = self.buckets.get(&(cx + dx, cz + dz)) {
                    result.extend_from_slice(bucket);
                }
            }
        }
        result
    }
}
