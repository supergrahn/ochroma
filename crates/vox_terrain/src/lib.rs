pub mod brushes;
pub mod deform;
pub mod foliage;
pub mod heightmap;
pub mod navmesh;
pub mod navmesh_bridge;
pub mod scene;
pub mod texture_paint;
pub mod volume;
pub mod water;

pub use deform::{apply_explosion, carve_sphere, carve_tunnel, fill_sphere};
pub use scene::TerrainScene;
pub use water::WaterField;

use serde::{Deserialize, Serialize};

// CONFIG-FIRST: these `pub const`s are retained as the public-API DEFAULT
// mirror, but the runtime source of truth is `vox_config` (`config/ochroma.ron`
// `terrain.heightmap_size` / `terrain.heightmap_resolution`). The runtime read
// sites below pull from config so the values are tunable without a rebuild; the
// const values equal the config defaults, so behavior is unchanged when no
// `ochroma.ron` is present.
pub const HEIGHTMAP_SIZE: usize = 4096;
pub const HEIGHTMAP_RESOLUTION: f32 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceType {
    Grass,
    Dirt,
    Rock,
    Sand,
    Snow,
    WaterBed,
}

#[derive(Debug, Clone)]
pub struct TerrainTile {
    pub heights: Vec<f32>,
    pub surfaces: Vec<SurfaceType>,
    pub size: usize,
}

impl TerrainTile {
    pub fn flat(height: f32) -> Self {
        let size = vox_config::config().terrain.heightmap_size as usize;
        let count = size * size;
        Self {
            heights: vec![height; count],
            surfaces: vec![SurfaceType::Grass; count],
            size,
        }
    }

    pub fn height_at(&self, x: usize, z: usize) -> f32 {
        let x = x.min(self.size - 1);
        let z = z.min(self.size - 1);
        self.heights[z * self.size + x]
    }

    pub fn set_height(&mut self, x: usize, z: usize, h: f32) {
        let x = x.min(self.size - 1);
        let z = z.min(self.size - 1);
        self.heights[z * self.size + x] = h;
    }

    /// Bilinear interpolation for sub-cell sampling.
    /// local_x and local_z are in world units (multiplied by HEIGHTMAP_RESOLUTION per cell).
    pub fn sample(&self, local_x: f32, local_z: f32) -> f32 {
        let res = vox_config::config().terrain.heightmap_resolution;
        let fx = (local_x / res).max(0.0);
        let fz = (local_z / res).max(0.0);
        let x0 = (fx as usize).min(self.size - 1);
        let z0 = (fz as usize).min(self.size - 1);
        let x1 = (x0 + 1).min(self.size - 1);
        let z1 = (z0 + 1).min(self.size - 1);
        let tx = fx - fx.floor();
        let tz = fz - fz.floor();

        let h00 = self.heights[z0 * self.size + x0];
        let h10 = self.heights[z0 * self.size + x1];
        let h01 = self.heights[z1 * self.size + x0];
        let h11 = self.heights[z1 * self.size + x1];

        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;
        h0 + (h1 - h0) * tz
    }
}
