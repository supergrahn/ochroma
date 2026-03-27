/// 2D grid of land values, updated when buildings/services change.
pub struct LandValueGrid {
    width: usize,
    height: usize,
    cell_size: f32, // metres per cell
    values: Vec<f32>,
}

impl LandValueGrid {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        Self { width, height, cell_size, values: vec![0.0; width * height] }
    }

    pub fn get(&self, x: usize, z: usize) -> f32 {
        if x < self.width && z < self.height { self.values[z * self.width + x] } else { 0.0 }
    }

    pub fn set(&mut self, x: usize, z: usize, value: f32) {
        if x < self.width && z < self.height { self.values[z * self.width + x] = value; }
    }

    /// Convert world position to grid cell.
    pub fn world_to_cell(&self, x: f32, z: f32) -> (usize, usize) {
        let cx = ((x / self.cell_size) + self.width as f32 * 0.5).max(0.0) as usize;
        let cz = ((z / self.cell_size) + self.height as f32 * 0.5).max(0.0) as usize;
        (cx.min(self.width - 1), cz.min(self.height - 1))
    }

    /// Sample land value at world position (bilinear).
    pub fn sample(&self, x: f32, z: f32) -> f32 {
        let (cx, cz) = self.world_to_cell(x, z);
        self.get(cx, cz)
    }

    /// Recalculate land values based on service positions and parks.
    pub fn recalculate(&mut self, services: &[(f32, f32, f32)], parks: &[(f32, f32)]) {
        for z in 0..self.height {
            for x in 0..self.width {
                let wx = (x as f32 - self.width as f32 * 0.5) * self.cell_size;
                let wz = (z as f32 - self.height as f32 * 0.5) * self.cell_size;

                let mut value = 0.5; // base

                // Services increase nearby land value
                for (sx, sz, radius) in services {
                    let dist = ((wx - sx).powi(2) + (wz - sz).powi(2)).sqrt();
                    if dist < *radius {
                        value += 0.3 * (1.0 - dist / radius);
                    }
                }

                // Parks increase nearby land value
                for (px, pz) in parks {
                    let dist = ((wx - px).powi(2) + (wz - pz).powi(2)).sqrt();
                    if dist < 200.0 {
                        value += 0.2 * (1.0 - dist / 200.0);
                    }
                }

                self.set(x, z, value.min(1.0));
            }
        }
    }
}
