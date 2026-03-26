use std::collections::HashMap;
use vox_core::lwc::{TileCoord, TileState};

pub struct TileManager {
    tiles: HashMap<TileCoord, TileState>,
    active_radius: i32,
}

impl TileManager {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            active_radius: 1,
        }
    }

    pub fn tile_state(&self, tile: TileCoord) -> TileState {
        self.tiles.get(&tile).copied().unwrap_or(TileState::Cold)
    }

    pub fn update_camera(&mut self, camera_tile: TileCoord) {
        // Mark far tiles as Evicting then remove
        let r = self.active_radius;
        let to_evict: Vec<TileCoord> = self.tiles.keys().copied().filter(|t| {
            (t.x - camera_tile.x).abs() > r || (t.z - camera_tile.z).abs() > r
        }).collect();
        for t in to_evict {
            self.tiles.remove(&t);
        }

        // Activate tiles within radius
        for dx in -r..=r {
            for dz in -r..=r {
                let tile = TileCoord {
                    x: camera_tile.x + dx,
                    z: camera_tile.z + dz,
                };
                self.tiles.entry(tile).or_insert(TileState::Active);
                if let Some(state) = self.tiles.get_mut(&tile) {
                    if *state == TileState::Cold {
                        *state = TileState::Active;
                    }
                }
            }
        }
    }

    pub fn active_tiles(&self) -> Vec<TileCoord> {
        self.tiles
            .iter()
            .filter(|(_, s)| **s == TileState::Active)
            .map(|(t, _)| *t)
            .collect()
    }
}

impl Default for TileManager {
    fn default() -> Self {
        Self::new()
    }
}
