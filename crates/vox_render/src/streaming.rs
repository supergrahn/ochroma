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

    pub fn with_radius(active_radius: i32) -> Self {
        Self {
            tiles: HashMap::new(),
            active_radius,
        }
    }

    pub fn tile_state(&self, tile: TileCoord) -> TileState {
        self.tiles.get(&tile).copied().unwrap_or(TileState::Cold)
    }

    pub fn update_camera(&mut self, camera_tile: TileCoord) -> Vec<TileCoord> {
        let mut newly_active = Vec::new();

        // Mark far tiles as Evicting then remove
        let r = self.active_radius;
        let to_evict: Vec<TileCoord> = self.tiles.keys().copied().filter(|t| {
            (t.x - camera_tile.x).abs() > r || (t.z - camera_tile.z).abs() > r
        }).collect();
        for t in to_evict {
            self.tiles.remove(&t);
        }

        // Activate tiles within radius; track newly activated ones
        for dx in -r..=r {
            for dz in -r..=r {
                let tile = TileCoord {
                    x: camera_tile.x + dx,
                    z: camera_tile.z + dz,
                };
                let was_present = self.tiles.contains_key(&tile);
                self.tiles.entry(tile).or_insert(TileState::Active);
                if !was_present {
                    newly_active.push(tile);
                }
            }
        }

        newly_active
    }

    pub fn active_tiles(&self) -> Vec<TileCoord> {
        self.tiles.keys().copied().collect()
    }
}

impl Default for TileManager {
    fn default() -> Self {
        Self::new()
    }
}

use vox_data::vxm::{VxmFile, VxmError};

pub struct AsyncAssetLoader;

impl AsyncAssetLoader {
    pub fn new() -> Self { Self }

    pub async fn load_from_bytes(&self, bytes: &[u8]) -> Result<VxmFile, VxmError> {
        let bytes = bytes.to_vec();
        tokio::task::spawn_blocking(move || VxmFile::read(&bytes[..]))
            .await
            .map_err(|e| VxmError::Io(std::io::Error::other(e)))?
    }

    pub async fn load_from_path(&self, path: &std::path::Path) -> Result<VxmFile, VxmError> {
        let bytes = tokio::fs::read(path).await?;
        self.load_from_bytes(&bytes).await
    }
}

impl Default for AsyncAssetLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::lwc::TileCoord;

    #[test]
    fn tile_manager_returns_newly_active_tiles_on_first_update() {
        let mut tm = TileManager::with_radius(1);
        let newly = tm.update_camera(TileCoord { x: 0, z: 0 });
        assert_eq!(newly.len(), 9, "first update should activate 9 tiles (3x3 grid)");
    }

    #[test]
    fn tile_manager_no_newly_active_on_same_position() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 0, z: 0 });
        let newly = tm.update_camera(TileCoord { x: 0, z: 0 });
        assert!(newly.is_empty(), "second call at same position should yield no new tiles");
    }

    #[test]
    fn tile_manager_evicts_distant_tiles() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 0, z: 0 });
        tm.update_camera(TileCoord { x: 100, z: 100 });
        let active = tm.active_tiles();
        assert!(
            !active.contains(&TileCoord { x: 0, z: 0 }),
            "tile (0,0) should be evicted after camera moves to (100,100)"
        );
    }

    #[test]
    fn tile_manager_active_tiles_within_radius() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 5, z: 5 });
        let active = tm.active_tiles();
        assert_eq!(active.len(), 9);
        for t in &active {
            assert!((t.x - 5).abs() <= 1 && (t.z - 5).abs() <= 1);
        }
    }
}
