use serde::{Deserialize, Serialize};

/// Pollution grid — tracks air and ground pollution levels across the city.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollutionGrid {
    width: usize,
    height: usize,
    cell_size: f32,
    /// Air pollution per cell (0.0 = clean, 1.0 = heavily polluted).
    pub air: Vec<f32>,
    /// Ground pollution per cell.
    pub ground: Vec<f32>,
}

impl PollutionGrid {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let count = width * height;
        Self {
            width,
            height,
            cell_size,
            air: vec![0.0; count],
            ground: vec![0.0; count],
        }
    }

    fn idx(&self, x: usize, z: usize) -> usize {
        z.min(self.height - 1) * self.width + x.min(self.width - 1)
    }

    pub fn world_to_cell(&self, wx: f32, wz: f32) -> (usize, usize) {
        let cx = ((wx / self.cell_size) + self.width as f32 * 0.5).max(0.0) as usize;
        let cz = ((wz / self.cell_size) + self.height as f32 * 0.5).max(0.0) as usize;
        (cx.min(self.width - 1), cz.min(self.height - 1))
    }

    /// Add a pollution source at world position.
    pub fn add_source(&mut self, wx: f32, wz: f32, radius: f32, intensity: f32, is_air: bool) {
        let (cx, cz) = self.world_to_cell(wx, wz);
        let r_cells = (radius / self.cell_size).ceil() as i32;

        for dz in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let nx = cx as i32 + dx;
                let nz = cz as i32 + dz;
                if nx < 0 || nz < 0 || nx >= self.width as i32 || nz >= self.height as i32 {
                    continue;
                }
                let dist = ((dx * dx + dz * dz) as f32).sqrt() * self.cell_size;
                if dist > radius {
                    continue;
                }
                let falloff = 1.0 - dist / radius;
                let idx = self.idx(nx as usize, nz as usize);
                if is_air {
                    self.air[idx] = (self.air[idx] + intensity * falloff).min(1.0);
                } else {
                    self.ground[idx] = (self.ground[idx] + intensity * falloff).min(1.0);
                }
            }
        }
    }

    /// Diffuse pollution (spreads to neighbors each tick).
    pub fn diffuse(&mut self, rate: f32) {
        let mut new_air = self.air.clone();
        for z in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                let idx = self.idx(x, z);
                let neighbors = [
                    self.air[self.idx(x - 1, z)],
                    self.air[self.idx(x + 1, z)],
                    self.air[self.idx(x, z - 1)],
                    self.air[self.idx(x, z + 1)],
                ];
                let avg: f32 = neighbors.iter().sum::<f32>() / 4.0;
                new_air[idx] = self.air[idx] + (avg - self.air[idx]) * rate;
            }
        }
        self.air = new_air;
    }

    /// Decay pollution naturally.
    pub fn decay(&mut self, rate: f32) {
        for v in &mut self.air {
            *v = (*v * (1.0 - rate)).max(0.0);
        }
        for v in &mut self.ground {
            *v = (*v * (1.0 - rate * 0.1)).max(0.0); // ground decays slower
        }
    }

    pub fn air_at(&self, wx: f32, wz: f32) -> f32 {
        let (cx, cz) = self.world_to_cell(wx, wz);
        self.air[self.idx(cx, cz)]
    }

    pub fn average_air_pollution(&self) -> f32 {
        self.air.iter().sum::<f32>() / self.air.len() as f32
    }
}
