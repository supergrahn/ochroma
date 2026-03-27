use vox_core::spectral::SpectralBands;
use glam::Vec3;
use std::collections::HashMap;

/// A cell in the irradiance cache storing spectral radiance.
#[derive(Debug, Clone)]
pub struct IrradianceCell {
    pub position: Vec3,
    pub incoming: SpectralBands,    // total incoming spectral irradiance
    pub bounce_count: u32,          // how many bounces contributed
    pub last_updated: u32,          // frame number
}

/// Spatial grid for GI cache.
pub struct GICache {
    cells: HashMap<(i32, i32, i32), IrradianceCell>,
    cell_size: f32,
    max_bounces: u32,
}

impl GICache {
    pub fn new(cell_size: f32, max_bounces: u32) -> Self {
        Self { cells: HashMap::new(), cell_size, max_bounces }
    }

    fn key(&self, pos: Vec3) -> (i32, i32, i32) {
        ((pos.x / self.cell_size).floor() as i32,
         (pos.y / self.cell_size).floor() as i32,
         (pos.z / self.cell_size).floor() as i32)
    }

    /// Add a spectral bounce contribution to a cell.
    pub fn add_bounce(&mut self, position: Vec3, spectral: SpectralBands, frame: u32) {
        let key = self.key(position);
        let cell = self.cells.entry(key).or_insert_with(|| IrradianceCell {
            position,
            incoming: SpectralBands([0.0; 8]),
            bounce_count: 0,
            last_updated: frame,
        });
        // Accumulate: average incoming radiance
        for i in 0..8 {
            cell.incoming.0[i] = (cell.incoming.0[i] * cell.bounce_count as f32 + spectral.0[i])
                / (cell.bounce_count + 1) as f32;
        }
        cell.bounce_count += 1;
        cell.last_updated = frame;
    }

    /// Query the cached irradiance at a position.
    pub fn query(&self, position: Vec3) -> Option<&IrradianceCell> {
        self.cells.get(&self.key(position))
    }

    /// Compute first bounce: direct light hits surface, reflects with surface SPD.
    pub fn compute_first_bounce(
        &mut self,
        surface_pos: Vec3,
        surface_spd: &SpectralBands,
        light_spd: &SpectralBands,
        frame: u32,
    ) {
        // Reflected spectrum = surface_spd x light_spd (element-wise)
        let reflected = SpectralBands(std::array::from_fn(|i| surface_spd.0[i] * light_spd.0[i]));
        self.add_bounce(surface_pos, reflected, frame);
    }

    /// Compute second bounce: reflected light from nearby cached cells.
    pub fn compute_second_bounce(
        &mut self,
        position: Vec3,
        surface_spd: &SpectralBands,
        search_radius: f32,
        frame: u32,
    ) {
        let r_cells = (search_radius / self.cell_size).ceil() as i32;
        let center_key = self.key(position);
        let mut accumulated = SpectralBands([0.0; 8]);
        let mut count = 0u32;

        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                for dz in -r_cells..=r_cells {
                    let key = (center_key.0 + dx, center_key.1 + dy, center_key.2 + dz);
                    if let Some(cell) = self.cells.get(&key) {
                        let dist = cell.position.distance(position);
                        if dist < search_radius && dist > 0.01 {
                            let falloff = 1.0 / (1.0 + dist * dist);
                            for i in 0..8 {
                                accumulated.0[i] += cell.incoming.0[i] * falloff;
                            }
                            count += 1;
                        }
                    }
                }
            }
        }

        if count > 0 {
            // Modulate by surface reflectance
            let bounced = SpectralBands(std::array::from_fn(|i| {
                (accumulated.0[i] / count as f32) * surface_spd.0[i]
            }));
            self.add_bounce(position, bounced, frame);
        }
    }

    pub fn cell_count(&self) -> usize { self.cells.len() }

    /// Maximum configured bounces.
    pub fn max_bounces(&self) -> u32 { self.max_bounces }

    /// Evict stale cells not updated in N frames.
    pub fn evict_stale(&mut self, current_frame: u32, max_age: u32) {
        self.cells.retain(|_, cell| current_frame - cell.last_updated < max_age);
    }
}
