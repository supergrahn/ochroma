/// Manages streaming splat data to GPU in tiles.
pub struct MegaGeometryDispatch {
    /// Maximum splats in GPU memory at once.
    pub gpu_budget: usize,
    /// Tile size in pixels.
    pub tile_size: u32,
    /// Screen width/height for tile calculation.
    pub screen_width: u32,
    pub screen_height: u32,
    /// Stats.
    pub last_frame_stats: FrameStats,
}

#[derive(Debug, Default, Clone)]
pub struct FrameStats {
    pub total_splats_in_scene: usize,
    pub splats_uploaded: usize,
    pub splats_culled: usize,
    pub splats_rendered: usize,
    pub tiles_processed: u32,
    pub clusters_tested: u32,
    pub clusters_hit: u32,
}

/// A screen-space tile for tile-based rendering.
#[derive(Debug)]
pub struct ScreenTile {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub splat_indices: Vec<u32>,  // splats visible in this tile
}

impl MegaGeometryDispatch {
    pub fn new(screen_width: u32, screen_height: u32, gpu_budget: usize) -> Self {
        Self {
            gpu_budget,
            tile_size: 16,
            screen_width,
            screen_height,
            last_frame_stats: FrameStats::default(),
        }
    }

    /// Compute screen tiles.
    pub fn compute_tiles(&self) -> Vec<ScreenTile> {
        let nx = self.screen_width.div_ceil(self.tile_size);
        let ny = self.screen_height.div_ceil(self.tile_size);
        let mut tiles = Vec::with_capacity((nx * ny) as usize);
        for ty in 0..ny {
            for tx in 0..nx {
                tiles.push(ScreenTile {
                    x: tx * self.tile_size,
                    y: ty * self.tile_size,
                    width: self.tile_size.min(self.screen_width - tx * self.tile_size),
                    height: self.tile_size.min(self.screen_height - ty * self.tile_size),
                    splat_indices: Vec::new(),
                });
            }
        }
        tiles
    }

    /// Assign splats to tiles based on their screen-space position.
    /// This is the core of tile-based rasterisation.
    pub fn assign_splats_to_tiles(
        &mut self,
        tiles: &mut [ScreenTile],
        screen_positions: &[(f32, f32, f32)], // (screen_x, screen_y, radius)
    ) {
        let mut stats = FrameStats {
            total_splats_in_scene: screen_positions.len(),
            ..Default::default()
        };

        for (i, &(sx, sy, radius)) in screen_positions.iter().enumerate() {
            if i >= self.gpu_budget {
                stats.splats_culled += 1;
                continue;
            }

            let x_min = ((sx - radius).max(0.0) / self.tile_size as f32).floor() as u32;
            let x_max = ((sx + radius).min(self.screen_width as f32 - 1.0) / self.tile_size as f32).ceil() as u32;
            let y_min = ((sy - radius).max(0.0) / self.tile_size as f32).floor() as u32;
            let y_max = ((sy + radius).min(self.screen_height as f32 - 1.0) / self.tile_size as f32).ceil() as u32;

            let tiles_x = self.screen_width.div_ceil(self.tile_size);

            let mut assigned = false;
            for ty in y_min..=y_max {
                for tx in x_min..=x_max {
                    let tile_idx = (ty * tiles_x + tx) as usize;
                    if tile_idx < tiles.len() {
                        tiles[tile_idx].splat_indices.push(i as u32);
                        assigned = true;
                    }
                }
            }

            if assigned {
                stats.splats_rendered += 1;
            } else {
                stats.splats_culled += 1;
            }
        }

        stats.splats_uploaded = stats.splats_rendered;
        stats.tiles_processed = tiles.len() as u32;
        self.last_frame_stats = stats;
    }

    pub fn tile_count(&self) -> u32 {
        let nx = self.screen_width.div_ceil(self.tile_size);
        let ny = self.screen_height.div_ceil(self.tile_size);
        nx * ny
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }
}
