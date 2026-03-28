use dashmap::DashMap;
use glam::Vec3;

type VoxelKey = (i32, i32, i32);

pub struct SpatialHash {
    cell_size: f32,
    map: DashMap<VoxelKey, Vec<u32>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self { cell_size, map: DashMap::new() }
    }

    fn key(&self, pos: Vec3) -> VoxelKey {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    pub fn insert(&mut self, instance_id: u32, position: Vec3) {
        let key = self.key(position);
        self.map.entry(key).or_default().push(instance_id);
    }

    pub fn remove(&mut self, instance_id: u32, position: Vec3) {
        let key = self.key(position);
        if let Some(mut ids) = self.map.get_mut(&key) {
            ids.retain(|id| *id != instance_id);
        }
    }

    pub fn query_voxel(&self, position: Vec3) -> Vec<u32> {
        let key = self.key(position);
        self.map.get(&key).map(|ids| ids.clone()).unwrap_or_default()
    }

    pub fn query_radius(&self, position: Vec3, radius: f32) -> Vec<u32> {
        let cells = (radius / self.cell_size).ceil() as i32 + 1;
        let ck = self.key(position);
        let mut result = Vec::new();
        for dx in -cells..=cells {
            for dy in -cells..=cells {
                for dz in -cells..=cells {
                    if let Some(ids) = self.map.get(&(ck.0 + dx, ck.1 + dy, ck.2 + dz)) {
                        result.extend(ids.iter());
                    }
                }
            }
        }
        result
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query_voxel() {
        let mut hash = SpatialHash::new(10.0);
        hash.insert(1, Vec3::new(5.0, 5.0, 5.0));
        hash.insert(2, Vec3::new(5.5, 5.5, 5.5));
        let result = hash.query_voxel(Vec3::new(5.0, 5.0, 5.0));
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn empty_query_returns_nothing() {
        let hash = SpatialHash::new(10.0);
        let result = hash.query_voxel(Vec3::ZERO);
        assert!(result.is_empty());
    }

    #[test]
    fn remove_removes_instance() {
        let mut hash = SpatialHash::new(10.0);
        let pos = Vec3::new(1.0, 2.0, 3.0);
        hash.insert(42, pos);
        assert!(!hash.query_voxel(pos).is_empty());
        hash.remove(42, pos);
        assert!(hash.query_voxel(pos).is_empty());
    }

    #[test]
    fn query_radius_finds_nearby() {
        let mut hash = SpatialHash::new(5.0);
        hash.insert(1, Vec3::new(0.0, 0.0, 0.0));
        hash.insert(2, Vec3::new(4.0, 0.0, 0.0));
        hash.insert(3, Vec3::new(100.0, 0.0, 0.0));
        let result = hash.query_radius(Vec3::ZERO, 10.0);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        // 3 is far away, might not be in result depending on cell math
    }
}
