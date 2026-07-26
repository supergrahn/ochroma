//! Shore interaction — wave shoaling, breaking, beach friction, erosion and foam.
//!
//! Ported from `forge-water`'s `sim::shore::ShoreInteraction`. Given a wave-height
//! field and a depth map it produces the shoaled/broken wave field, a breaking
//! mask, a swash-zone foam field, a per-cell erosion rate and the interpolated
//! shoreline polyline.
//!
//! Physics:
//! * **Shoaling** — Green's law, `H = H_deep · (d_deep / d)^¼`: a wave entering
//!   shallow water slows, shortens and grows.
//! * **Breaking** — the McCowan depth-limited criterion `H / d > 0.78`.
//! * **Beach friction** — bed drag rising as depth falls.
//! * **Erosion** — proportional to wave height at breaking cells only.
//!
//! Grids are row-major `[row * cols + col]`. `depth_map` is **positive in water**;
//! zero or negative is land (its magnitude being height above the waterline).
//!
//! # Determinism
//!
//! Fully deterministic and **gameplay-safe**: every pass is a fixed `0..n`
//! traversal writing a freshly allocated output (never reading a value written in
//! the same pass), there is no RNG, no `HashMap` and no clock. Fold
//! [`ShorelineResult`] into the replay hash if you drive gameplay (erosion,
//! flood damage) with it.

/// Configuration for shore/beach interaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShoreConfig {
    pub shoaling_enabled: bool,
    /// `H/d` ratio at which waves break (McCowan: 0.78).
    pub breaking_threshold: f32,
    /// Energy-loss fraction from beach friction at the waterline.
    pub beach_friction: f32,
    /// Beach erosion rate per wave cycle.
    pub erosion_rate: f32,
    pub foam_at_shore: bool,
    /// Minimum water depth (m) treated as wet.
    pub min_depth: f32,
    /// Wave run-up distance on the beach (m).
    pub swash_length: f32,
}

impl Default for ShoreConfig {
    fn default() -> Self {
        Self {
            shoaling_enabled: true,
            breaking_threshold: 0.78,
            beach_friction: 0.3,
            erosion_rate: 0.01,
            foam_at_shore: true,
            min_depth: 0.1,
            swash_length: 5.0,
        }
    }
}

/// Result of a shore interaction pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ShorelineResult {
    pub rows: usize,
    pub cols: usize,
    /// Wave height after shoaling and friction, row-major.
    pub wave_heights: Vec<f32>,
    /// True where the wave is breaking.
    pub breaking_mask: Vec<bool>,
    /// Foam coverage 0..1 (1 at breaking cells, falling off through the swash).
    pub foam_density: Vec<f32>,
    /// Erosion rate per cell (only nonzero where breaking).
    pub erosion_map: Vec<f32>,
    /// Interpolated land/water crossings in world `(x, z)`.
    pub shoreline_positions: Vec<(f32, f32)>,
    pub num_breaking_cells: usize,
}

/// Simulates wave interaction with shoreline terrain.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShoreInteraction {
    pub config: ShoreConfig,
}

impl ShoreInteraction {
    #[must_use]
    pub fn new(config: ShoreConfig) -> Self {
        Self { config }
    }

    /// Run one shore interaction pass over `rows × cols` cells of `spacing` metres.
    ///
    /// Returns an empty result (never panics) on a size mismatch or empty grid.
    #[must_use]
    pub fn simulate(
        &self,
        wave_heights: &[f32],
        depth_map: &[f32],
        rows: usize,
        cols: usize,
        spacing: f32,
    ) -> ShorelineResult {
        let n = rows * cols;
        if n == 0 || wave_heights.len() != n || depth_map.len() != n {
            return ShorelineResult {
                rows,
                cols,
                wave_heights: Vec::new(),
                breaking_mask: Vec::new(),
                foam_density: Vec::new(),
                erosion_map: Vec::new(),
                shoreline_positions: Vec::new(),
                num_breaking_cells: 0,
            };
        }

        let shoreline_positions = self.detect_shoreline(depth_map, rows, cols, spacing);

        let mut heights = if self.config.shoaling_enabled {
            self.apply_shoaling(wave_heights, depth_map)
        } else {
            wave_heights.to_vec()
        };

        let breaking_mask = self.detect_breaking(&heights, depth_map);
        heights = self.apply_friction(&heights, depth_map);
        let erosion_map = self.compute_erosion(&breaking_mask, &heights);

        let foam_density = if self.config.foam_at_shore {
            self.generate_shore_foam(&breaking_mask, depth_map, spacing)
        } else {
            vec![0.0f32; n]
        };

        let num_breaking_cells = breaking_mask.iter().filter(|&&b| b).count();

        ShorelineResult {
            rows,
            cols,
            wave_heights: heights,
            breaking_mask,
            foam_density,
            erosion_map,
            shoreline_positions,
            num_breaking_cells,
        }
    }

    /// Green's law shoaling: `H = H_deep · (d_deep / d)^¼` where `d > min_depth`.
    #[must_use]
    pub fn apply_shoaling(&self, wave_heights: &[f32], depth_map: &[f32]) -> Vec<f32> {
        let mut shoaled = wave_heights.to_vec();
        let d_deep = depth_map.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !(d_deep > 0.0) {
            return shoaled;
        }
        for k in 0..depth_map.len() {
            if depth_map[k] > self.config.min_depth {
                let d = depth_map[k].max(self.config.min_depth);
                shoaled[k] = wave_heights[k] * (d_deep / d).powf(0.25);
            }
        }
        shoaled
    }

    /// Depth-limited breaking: `H / d > breaking_threshold` in water.
    #[must_use]
    pub fn detect_breaking(&self, shoaled_heights: &[f32], depth_map: &[f32]) -> Vec<bool> {
        let mut breaking = vec![false; depth_map.len()];
        for k in 0..depth_map.len() {
            if depth_map[k] > self.config.min_depth {
                breaking[k] = shoaled_heights[k] / depth_map[k] > self.config.breaking_threshold;
            }
        }
        breaking
    }

    /// Beach friction: `loss = friction · (1 − d/d_max)`, clamped to `[0, 1]`.
    #[must_use]
    pub fn apply_friction(&self, wave_heights: &[f32], depth_map: &[f32]) -> Vec<f32> {
        let mut heights = wave_heights.to_vec();
        let d_max = depth_map.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !(d_max > 0.0) {
            return heights;
        }
        for k in 0..depth_map.len() {
            if depth_map[k] > 0.0 {
                let loss =
                    (self.config.beach_friction * (1.0 - depth_map[k] / d_max)).clamp(0.0, 1.0);
                heights[k] *= 1.0 - loss;
            }
        }
        heights
    }

    /// `erosion = erosion_rate · H` at breaking cells, zero elsewhere.
    #[must_use]
    pub fn compute_erosion(&self, breaking_mask: &[bool], wave_heights: &[f32]) -> Vec<f32> {
        let mut erosion = vec![0.0f32; wave_heights.len()];
        for k in 0..wave_heights.len() {
            if breaking_mask[k] {
                erosion[k] = self.config.erosion_rate * wave_heights[k];
            }
        }
        erosion
    }

    /// Shore foam: 1.0 at breaking cells, with a linear falloff across the swash
    /// zone on both sides of the waterline.
    #[must_use]
    pub fn generate_shore_foam(
        &self,
        breaking_mask: &[bool],
        depth_map: &[f32],
        spacing: f32,
    ) -> Vec<f32> {
        let mut foam = vec![0.0f32; depth_map.len()];
        if !(spacing > 0.0) || !(self.config.swash_length > 0.0) {
            for k in 0..depth_map.len() {
                if breaking_mask[k] {
                    foam[k] = 1.0;
                }
            }
            return foam;
        }
        let near_thresh = self.config.swash_length / spacing * self.config.min_depth;
        let beach_thresh = -self.config.swash_length / spacing * 0.5;
        for k in 0..depth_map.len() {
            if breaking_mask[k] {
                foam[k] = 1.0;
                continue;
            }
            let d = depth_map[k];
            let near_shore = d >= 0.0 && d < near_thresh;
            let on_beach = d < 0.0 && d > beach_thresh;
            if near_shore || on_beach {
                let normalized = (d.abs() * spacing / self.config.swash_length).clamp(0.0, 1.0);
                foam[k] = foam[k].max(1.0 - normalized);
            }
        }
        foam
    }

    /// Shoreline crossings: cells where depth changes sign between 4-neighbours,
    /// with the crossing point linearly interpolated. Returns world `(x, z)`.
    #[must_use]
    pub fn detect_shoreline(
        &self,
        depth_map: &[f32],
        rows: usize,
        cols: usize,
        spacing: f32,
    ) -> Vec<(f32, f32)> {
        let at = |i: usize, j: usize| depth_map[i * cols + j];
        let crossing = |d1: f32, d2: f32| -> Option<f32> {
            let wet1 = d1 > 0.0;
            let wet2 = d2 > 0.0;
            if wet1 == wet2 {
                return None;
            }
            Some(if (d1 - d2).abs() > 1e-6 {
                d1.abs() / (d1 - d2).abs()
            } else {
                0.5
            })
        };
        let mut positions = Vec::new();
        for i in 0..rows {
            for j in 0..cols.saturating_sub(1) {
                if let Some(t) = crossing(at(i, j), at(i, j + 1)) {
                    positions.push(((j as f32 + t) * spacing, i as f32 * spacing));
                }
            }
        }
        for i in 0..rows.saturating_sub(1) {
            for j in 0..cols {
                if let Some(t) = crossing(at(i, j), at(i + 1, j)) {
                    positions.push((j as f32 * spacing, (i as f32 + t) * spacing));
                }
            }
        }
        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greens_law_amplifies_waves_in_the_shallows() {
        let shore = ShoreInteraction::default();
        let shoaled = shore.apply_shoaling(&[1.0, 1.0], &[4.0, 1.0]);
        // deep cell (4/4)^0.25 = 1 -> unchanged; shallow (4/1)^0.25 = sqrt(2).
        assert!((shoaled[0] - 1.0).abs() < 1e-6);
        assert!((shoaled[1] - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn breaking_uses_the_mccowan_078_criterion() {
        let shore = ShoreInteraction::default();
        let mask = shore.detect_breaking(&[0.9, 0.5], &[1.0, 1.0]);
        assert!(mask[0], "H/d = 0.90 must break");
        assert!(!mask[1], "H/d = 0.50 must not break");
    }

    #[test]
    fn friction_dissipates_more_energy_in_shallow_water() {
        let shore = ShoreInteraction::default();
        let out = shore.apply_friction(&[1.0, 1.0], &[10.0, 1.0]);
        // deep: loss = 0.3*(1-1) = 0 -> unchanged.
        assert!((out[0] - 1.0).abs() < 1e-6);
        // shallow: loss = 0.3*(1-0.1) = 0.27 -> 0.73.
        assert!((out[1] - 0.73).abs() < 1e-5, "got {}", out[1]);
    }

    #[test]
    fn erosion_only_happens_where_waves_break() {
        let shore = ShoreInteraction::default();
        let er = shore.compute_erosion(&[true, false], &[2.0, 2.0]);
        assert!((er[0] - 0.02).abs() < 1e-6);
        assert_eq!(er[1], 0.0);
    }

    #[test]
    fn shoreline_crossing_is_interpolated_not_snapped() {
        let shore = ShoreInteraction::default();
        // Row [2.0, -2.0]: sign change at t = 2/4 = 0.5 -> x = 0.5 * 10 = 5.0.
        let pos = shore.detect_shoreline(&[2.0, -2.0], 1, 2, 10.0);
        assert_eq!(pos.len(), 1);
        assert!((pos[0].0 - 5.0).abs() < 1e-5, "x = {}", pos[0].0);
        assert!((pos[0].1 - 0.0).abs() < 1e-5);
        // An asymmetric crossing must land off-centre: [3, -1] -> t = 3/4.
        let skewed = shore.detect_shoreline(&[3.0, -1.0], 1, 2, 10.0);
        assert!((skewed[0].0 - 7.5).abs() < 1e-5, "x = {}", skewed[0].0);
    }

    #[test]
    fn foam_peaks_at_breakers_and_fades_up_the_beach() {
        let shore = ShoreInteraction::default();
        // spacing 1 m, swash 5 m: beach_thresh = -2.5, so d = -1 is in the swash.
        let depth = vec![1.0, -1.0, -4.0];
        let foam = shore.generate_shore_foam(&[true, false, false], &depth, 1.0);
        assert_eq!(foam[0], 1.0, "breaking cell is full foam");
        // d=-1: normalized = 1*1/5 = 0.2 -> foam 0.8.
        assert!((foam[1] - 0.8).abs() < 1e-6, "swash foam = {}", foam[1]);
        assert_eq!(foam[2], 0.0, "dry land beyond the swash has no foam");
    }

    #[test]
    fn full_pass_reports_breaking_erosion_and_foam_together() {
        let shore = ShoreInteraction::default();
        let res = shore.simulate(&[0.9; 4], &[1.0; 4], 2, 2, 1.0);
        assert_eq!(res.num_breaking_cells, 4);
        assert_eq!(res.erosion_map.iter().filter(|&&e| e > 0.0).count(), 4);
        assert_eq!(res.foam_density.iter().filter(|&&f| f == 1.0).count(), 4);
        assert_eq!(res.wave_heights.len(), 4);
    }

    #[test]
    fn a_beach_profile_breaks_inshore_but_not_offshore() {
        // The behavioural test that matters: a real sloping beach.
        let shore = ShoreInteraction::default();
        let cols = 8;
        // Depth falling 8 m -> -1 m across the profile (last cell is dry beach).
        let depth: Vec<f32> = (0..cols).map(|i| 8.0 - i as f32 * 1.2).collect();
        let waves = vec![1.2f32; cols];
        let res = shore.simulate(&waves, &depth, 1, cols, 5.0);
        // Deepest cell must not break; the shallowest wet cell must.
        assert!(!res.breaking_mask[0], "offshore should not break");
        assert!(
            res.breaking_mask.iter().any(|&b| b),
            "an inshore cell must break"
        );
        // Exactly one shoreline crossing on a monotone profile.
        assert_eq!(res.shoreline_positions.len(), 1);
    }

    #[test]
    fn mismatched_input_sizes_are_refused_not_panicked() {
        let shore = ShoreInteraction::default();
        let res = shore.simulate(&[1.0, 1.0], &[1.0], 1, 2, 1.0);
        assert_eq!(res.num_breaking_cells, 0);
        assert!(res.wave_heights.is_empty());
    }

    #[test]
    fn pass_is_reproducible() {
        let shore = ShoreInteraction::default();
        let depth: Vec<f32> = (0..16).map(|i| 6.0 - i as f32 * 0.5).collect();
        let waves: Vec<f32> = (0..16).map(|i| 0.8 + (i % 3) as f32 * 0.1).collect();
        assert_eq!(
            shore.simulate(&waves, &depth, 4, 4, 2.0),
            shore.simulate(&waves, &depth, 4, 4, 2.0)
        );
    }
}
