use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const TILE_SIZE: f64 = 1000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileCoord { pub x: i32, pub z: i32 }

impl TileCoord {
    pub fn anchor(&self) -> (f64, f64) {
        (self.x as f64 * TILE_SIZE, self.z as f64 * TILE_SIZE)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorldCoord {
    pub tile: TileCoord,
    pub local: Vec3,
}

impl WorldCoord {
    pub fn from_absolute(x: f64, y: f64, z: f64) -> Self {
        let tile_x = (x / TILE_SIZE).floor() as i32;
        let tile_z = (z / TILE_SIZE).floor() as i32;
        Self {
            tile: TileCoord { x: tile_x, z: tile_z },
            local: Vec3::new(
                (x - tile_x as f64 * TILE_SIZE) as f32,
                y as f32,
                (z - tile_z as f64 * TILE_SIZE) as f32,
            ),
        }
    }

    pub fn to_absolute(&self) -> (f64, f64, f64) {
        let (ax, az) = self.tile.anchor();
        (ax + self.local.x as f64, self.local.y as f64, az + self.local.z as f64)
    }

    pub fn local_relative_to(&self, camera_tile: TileCoord) -> Vec3 {
        let dx = (self.tile.x - camera_tile.x) as f32 * TILE_SIZE as f32;
        let dz = (self.tile.z - camera_tile.z) as f32 * TILE_SIZE as f32;
        Vec3::new(self.local.x + dx, self.local.y, self.local.z + dz)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileState { Cold, Warming, Warm, Active, Evicting }
