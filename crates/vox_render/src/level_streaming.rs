use std::path::PathBuf;

/// State of a streaming level.
#[derive(Debug, Clone, PartialEq)]
pub enum LevelState {
    Unloaded,
    Loading,
    Loaded { splat_count: usize },
    Visible,
    Unloading,
}

/// A level that can be streamed in/out.
pub struct StreamingLevel {
    pub name: String,
    pub bounds: ([f32; 3], [f32; 3]), // AABB min, max
    pub asset_path: PathBuf,
    pub state: LevelState,
    pub priority: u32,
}

impl StreamingLevel {
    /// Center of this level's AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.bounds.0[0] + self.bounds.1[0]) * 0.5,
            (self.bounds.0[1] + self.bounds.1[1]) * 0.5,
            (self.bounds.0[2] + self.bounds.1[2]) * 0.5,
        ]
    }

    /// Distance from a point to the closest point on this level's AABB.
    pub fn distance_to(&self, pos: [f32; 3]) -> f32 {
        let mut dist_sq = 0.0f32;
        for i in 0..3 {
            let v = pos[i];
            if v < self.bounds.0[i] {
                let d = self.bounds.0[i] - v;
                dist_sq += d * d;
            } else if v > self.bounds.1[i] {
                let d = v - self.bounds.1[i];
                dist_sq += d * d;
            }
        }
        dist_sq.sqrt()
    }
}

/// Manages async level loading/unloading based on camera distance.
pub struct LevelStreamingManager {
    pub levels: Vec<StreamingLevel>,
    pub load_distance: f32,
    pub unload_distance: f32,
    pub max_loaded: usize,
}

impl LevelStreamingManager {
    pub fn new(load_dist: f32, unload_dist: f32) -> Self {
        Self {
            levels: Vec::new(),
            load_distance: load_dist,
            unload_distance: unload_dist,
            max_loaded: 16,
        }
    }

    pub fn add_level(&mut self, name: &str, bounds: ([f32; 3], [f32; 3]), path: PathBuf) {
        self.levels.push(StreamingLevel {
            name: name.to_string(),
            bounds,
            asset_path: path,
            state: LevelState::Unloaded,
            priority: 0,
        });
    }

    /// Update level states based on camera position.
    pub fn update(&mut self, camera_pos: [f32; 3]) {
        // Compute distances and update priorities
        let mut distances: Vec<(usize, f32)> = self
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| (i, l.distance_to(camera_pos)))
            .collect();

        // Sort by distance for priority
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, &(idx, dist)) in distances.iter().enumerate() {
            self.levels[idx].priority = rank as u32;

            match &self.levels[idx].state {
                LevelState::Unloaded if dist <= self.load_distance => {
                    if self.loaded_count() < self.max_loaded {
                        self.levels[idx].state = LevelState::Loading;
                    }
                }
                LevelState::Loading => {
                    // Simulate instant load for CPU-side logic
                    self.levels[idx].state = LevelState::Loaded { splat_count: 0 };
                }
                LevelState::Loaded { .. } if dist <= self.load_distance => {
                    self.levels[idx].state = LevelState::Visible;
                }
                LevelState::Visible if dist > self.unload_distance => {
                    self.levels[idx].state = LevelState::Unloading;
                }
                LevelState::Loaded { .. } if dist > self.unload_distance => {
                    self.levels[idx].state = LevelState::Unloading;
                }
                LevelState::Unloading => {
                    self.levels[idx].state = LevelState::Unloaded;
                }
                _ => {}
            }
        }
    }

    /// Levels that need to be loaded.
    pub fn levels_to_load(&self) -> Vec<&StreamingLevel> {
        self.levels
            .iter()
            .filter(|l| matches!(l.state, LevelState::Loading))
            .collect()
    }

    /// Levels that need to be unloaded.
    pub fn levels_to_unload(&self) -> Vec<&StreamingLevel> {
        self.levels
            .iter()
            .filter(|l| matches!(l.state, LevelState::Unloading))
            .collect()
    }

    /// Count of currently loaded or visible levels.
    pub fn loaded_count(&self) -> usize {
        self.levels
            .iter()
            .filter(|l| {
                matches!(
                    l.state,
                    LevelState::Loaded { .. } | LevelState::Visible | LevelState::Loading
                )
            })
            .count()
    }

    /// Count of currently visible levels.
    pub fn visible_count(&self) -> usize {
        self.levels
            .iter()
            .filter(|l| matches!(l.state, LevelState::Visible))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> LevelStreamingManager {
        let mut mgr = LevelStreamingManager::new(100.0, 200.0);
        mgr.add_level(
            "level_a",
            ([0.0, 0.0, 0.0], [50.0, 50.0, 50.0]),
            PathBuf::from("levels/a.vxm"),
        );
        mgr.add_level(
            "level_b",
            ([500.0, 0.0, 0.0], [550.0, 50.0, 550.0]),
            PathBuf::from("levels/b.vxm"),
        );
        mgr.add_level(
            "level_c",
            ([1000.0, 0.0, 0.0], [1050.0, 50.0, 1050.0]),
            PathBuf::from("levels/c.vxm"),
        );
        mgr
    }

    #[test]
    fn near_camera_loads() {
        let mut mgr = make_manager();
        // Camera near level_a
        mgr.update([25.0, 25.0, 25.0]); // triggers Loading
        mgr.update([25.0, 25.0, 25.0]); // Loading -> Loaded
        mgr.update([25.0, 25.0, 25.0]); // Loaded -> Visible
        assert!(mgr.visible_count() >= 1);
        assert!(matches!(mgr.levels[0].state, LevelState::Visible));
    }

    #[test]
    fn far_unloads() {
        let mut mgr = make_manager();
        // Load level_a
        mgr.update([25.0, 25.0, 25.0]);
        mgr.update([25.0, 25.0, 25.0]);
        mgr.update([25.0, 25.0, 25.0]);
        assert!(matches!(mgr.levels[0].state, LevelState::Visible));

        // Move camera far away
        mgr.update([5000.0, 0.0, 0.0]); // Visible -> Unloading
        mgr.update([5000.0, 0.0, 0.0]); // Unloading -> Unloaded
        assert!(matches!(mgr.levels[0].state, LevelState::Unloaded));
    }

    #[test]
    fn max_loaded_respected() {
        let mut mgr = LevelStreamingManager::new(10000.0, 20000.0);
        mgr.max_loaded = 2;
        for i in 0..5 {
            let x = i as f32 * 10.0;
            mgr.add_level(
                &format!("level_{}", i),
                ([x, 0.0, 0.0], [x + 5.0, 5.0, 5.0]),
                PathBuf::from(format!("levels/{}.vxm", i)),
            );
        }
        // All levels within load distance
        mgr.update([25.0, 2.5, 2.5]);
        // Should not exceed max_loaded
        assert!(mgr.loaded_count() <= 2);
    }

    #[test]
    fn priority_reflects_distance() {
        let mut mgr = make_manager();
        mgr.update([25.0, 25.0, 25.0]);
        // level_a is closest, should have lowest priority number
        assert!(mgr.levels[0].priority < mgr.levels[2].priority);
    }
}
