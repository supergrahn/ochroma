//! Hair strand rendering and simulation.
//! HairStrand: 8 control points, spectral melanin model, mass-spring physics.
//! HairSplatGenerator: converts strands to GaussianSplats for EWA rendering.

use glam::Vec3;
use vox_core::types::GaussianSplat;
use half::f16;

/// A single hair strand with physics state and spectral colour.
pub struct HairStrand {
    pub root_pos: Vec3,
    /// 8 control points along the strand from root to tip.
    pub control_points: Vec<Vec3>,
    /// Width (radius) at each control point — tapers toward tip.
    pub width_curve: Vec<f32>,
    /// Spectral reflectance per band (from melanin model).
    pub spectral_melanin: [f32; 16],
    // Physics state
    prev_positions: Vec<Vec3>,
    #[allow(dead_code)]
    velocities: Vec<Vec3>,
    is_pinned: Vec<bool>,
}

impl HairStrand {
    /// Create a strand with 8 control points hanging straight down from root.
    pub fn new(root_pos: Vec3, length: f32, eumelanin: f32, pheomelanin: f32) -> Self {
        let n = 8;
        let control_points: Vec<Vec3> = (0..n)
            .map(|i| root_pos + Vec3::new(0.0, -length * i as f32 / (n - 1) as f32, 0.0))
            .collect();
        let width_curve: Vec<f32> = (0..n)
            .map(|i| 0.003 * (1.0 - i as f32 / n as f32)) // 3mm root → 0 tip
            .collect();
        let spectral_melanin = HairStrand::compute_spectral_melanin(eumelanin, pheomelanin);
        let prev_positions = control_points.clone();
        let velocities = vec![Vec3::ZERO; n];
        let mut is_pinned = vec![false; n];
        is_pinned[0] = true; // root is pinned

        Self {
            root_pos,
            control_points,
            width_curve,
            spectral_melanin,
            prev_positions,
            velocities,
            is_pinned,
        }
    }

    /// Compute spectral reflectance from eumelanin and pheomelanin densities.
    /// Based on d'Eon et al. absorption model.
    pub fn compute_spectral_melanin(eumelanin_density: f32, pheomelanin_density: f32) -> [f32; 16] {
        // Absorption coefficient spectra per unit density
        const A_EU: [f32; 16] = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.38, 0.35, 0.33, 0.31, 0.30];
        const A_PH: [f32; 16] = [0.80, 0.75, 0.70, 0.60, 0.50, 0.35, 0.20, 0.12, 0.05, 0.035, 0.02, 0.015, 0.010, 0.005, 0.002, 0.0];

        let mut result = [0.0f32; 16];
        for b in 0..16 {
            let absorption = eumelanin_density * A_EU[b] + pheomelanin_density * A_PH[b];
            result[b] = (-absorption).exp(); // Beer-Lambert reflectance
        }
        result
    }

    /// Pin the root to a new world position (call each frame from skeleton attachment).
    pub fn set_root(&mut self, new_root: Vec3) {
        let delta = new_root - self.control_points[0];
        self.control_points[0] = new_root;
        self.prev_positions[0] = new_root;
        self.root_pos = new_root;
        // Translate all control points by the same delta to avoid spring explosion on teleport.
        // Only if the movement is large.
        if delta.length() > 0.5 {
            for cp in &mut self.control_points {
                *cp += delta;
            }
            for pp in &mut self.prev_positions {
                *pp += delta;
            }
        }
    }

    /// Simulate one physics step.
    pub fn step(&mut self, dt: f32, gravity: Vec3, wind: Vec3) {
        let h = dt;
        let damping = 0.98f32;
        let n = self.control_points.len();

        // Compute rest lengths (segment lengths from current configuration).
        // Using current length as rest length (strands are inextensible by damped spring).
        let rest_lengths: Vec<f32> = (0..n - 1)
            .map(|i| {
                (self.control_points[i + 1] - self.control_points[i])
                    .length()
                    .max(1e-4)
            })
            .collect();

        // Verlet integration with wind + gravity.
        let external = gravity + wind;
        for i in 0..n {
            if self.is_pinned[i] {
                continue;
            }
            let vel = (self.control_points[i] - self.prev_positions[i]) * damping;
            let new_pos = self.control_points[i] + vel + external * h * h;
            self.prev_positions[i] = self.control_points[i];
            self.control_points[i] = new_pos;
        }

        // Distance constraints (4 iterations for stability).
        for _ in 0..4 {
            for (i, &rl) in rest_lengths[..n - 1].iter().enumerate() {
                if self.is_pinned[i] && self.is_pinned[i + 1] {
                    continue;
                }
                let delta = self.control_points[i + 1] - self.control_points[i];
                let dist = delta.length();
                if dist < 1e-6 {
                    continue;
                }
                let error = dist - rl;
                let correction = delta.normalize() * error * 0.5;
                if !self.is_pinned[i] {
                    self.control_points[i] += correction;
                }
                if !self.is_pinned[i + 1] {
                    self.control_points[i + 1] -= correction;
                }
            }
        }
    }
}

/// Converts `HairStrand` instances into `GaussianSplat` lists for EWA splatting.
pub struct HairSplatGenerator {
    /// Number of GaussianSplats to generate per strand (default 4 — every 2 control points).
    pub splats_per_strand: usize,
}

impl HairSplatGenerator {
    pub fn new() -> Self {
        Self { splats_per_strand: 4 }
    }

    /// Convert a HairStrand to GaussianSplats.
    /// Each splat spans one segment of the strand, positioned at midpoint,
    /// oriented along the strand direction.
    pub fn generate(&self, strand: &HairStrand) -> Vec<GaussianSplat> {
        let n = strand.control_points.len();
        let step = (n - 1).max(1) / self.splats_per_strand.max(1);
        let mut splats = Vec::with_capacity(self.splats_per_strand);

        for s in 0..self.splats_per_strand {
            let i = (s * step).min(n - 2);
            let p0 = strand.control_points[i];
            let p1 = strand.control_points[(i + 1).min(n - 1)];
            let mid = (p0 + p1) * 0.5;
            let seg_dir = p1 - p0;
            let seg_len = seg_dir.length().max(1e-4);

            // Width at this segment.
            let width = strand.width_curve.get(i).copied().unwrap_or(0.001);

            // Elongate splat along strand direction.
            let scale_along = seg_len * 0.5;
            let scale_perp = width;

            // Encode orientation as quaternion (rotate Y-up to seg_dir).
            let up = Vec3::Y;
            let rotation = if seg_dir.normalize().dot(up).abs() > 0.99 {
                glam::Quat::IDENTITY
            } else {
                glam::Quat::from_rotation_arc(up, seg_dir.normalize())
            };
            // Spectral: use strand's melanin reflectance, packed as f16.
            let spectral_f16: [u16; 16] =
                std::array::from_fn(|b| f16::from_f32(strand.spectral_melanin[b]).to_bits());

            splats.push(GaussianSplat::volume(
                mid.to_array(),
                [scale_perp, scale_along, scale_perp],
                rotation,
                200, // ~78% opacity for hair
                spectral_f16,
            ));
        }

        splats
    }
}

impl Default for HairSplatGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// LOD radii controlling simulation and rendering cutoffs for a hair groom.
#[derive(Debug, Clone)]
pub struct HairLodRadius {
    pub full_sim: f32, // default 5.0m — full physics simulation
    pub rigid: f32,    // default 20.0m — rigid (no physics, just render)
}

impl Default for HairLodRadius {
    fn default() -> Self {
        Self { full_sim: 5.0, rigid: 20.0 }
    }
}

/// A collection of hair strands with LOD-aware simulation and splat generation.
pub struct HairGroom {
    pub strands: Vec<HairStrand>,
    pub lod_radius: HairLodRadius,
}

impl HairGroom {
    pub fn new() -> Self {
        Self { strands: Vec::new(), lod_radius: HairLodRadius::default() }
    }

    pub fn add_strand(&mut self, strand: HairStrand) {
        self.strands.push(strand);
    }

    /// Simulate all strands within full_sim distance of camera.
    pub fn simulate(&mut self, dt: f32, gravity: Vec3, wind: Vec3, camera_pos: Vec3) {
        for strand in &mut self.strands {
            let dist = (strand.root_pos - camera_pos).length();
            if dist < self.lod_radius.full_sim {
                strand.step(dt, gravity, wind);
            }
            // Beyond full_sim: skip simulation (rigid).
            // Beyond rigid: culled by renderer.
        }
    }

    /// Generate splats for all strands within rigid distance of camera.
    pub fn generate_splats(&self, camera_pos: Vec3) -> Vec<GaussianSplat> {
        let splat_gen = HairSplatGenerator::new();
        let mut result = Vec::new();
        for strand in &self.strands {
            let dist = (strand.root_pos - camera_pos).length();
            if dist < self.lod_radius.rigid {
                result.extend(splat_gen.generate(strand));
            }
        }
        result
    }
}

impl Default for HairGroom {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn melanin_jet_black() {
        let bands = HairStrand::compute_spectral_melanin(3.0, 0.0);
        for (b, &v) in bands.iter().enumerate() {
            assert!(v < 0.5, "band {b} = {v}, expected < 0.5 for jet black");
        }
    }

    #[test]
    fn melanin_blonde() {
        let bands = HairStrand::compute_spectral_melanin(0.1, 0.1);
        for (b, &v) in bands.iter().enumerate() {
            assert!(v > 0.7, "band {b} = {v}, expected > 0.7 for blonde");
        }
    }

    #[test]
    fn melanin_red_high_in_red_bands() {
        // High pheomelanin: absorbs UV (band 0), reflects red (band 7).
        let bands = HairStrand::compute_spectral_melanin(0.3, 1.5);
        assert!(
            bands[7] > bands[0],
            "band 7 ({}) should be brighter than band 0 ({}) for red hair",
            bands[7],
            bands[0]
        );
    }

    #[test]
    fn hair_splat_generator_produces_splats() {
        let strand = HairStrand::new(Vec3::ZERO, 0.3, 0.5, 0.0);
        let splats = HairSplatGenerator::new().generate(&strand);
        assert_eq!(splats.len(), 4);
    }

    #[test]
    fn hair_strand_root_pinned() {
        let root = Vec3::new(1.0, 2.0, 3.0);
        let mut strand = HairStrand::new(root, 0.3, 0.5, 0.0);
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let wind = Vec3::new(0.1, 0.0, 0.0);
        strand.step(0.1, gravity, wind);
        let cp0 = strand.control_points[0];
        assert!(
            (cp0 - root).length() < 1e-5,
            "root control point moved: {:?}",
            cp0
        );
    }

    #[test]
    fn hair_groom_simulate_moves_strands() {
        let camera = Vec3::ZERO;
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let wind = Vec3::ZERO;

        let mut groom = HairGroom::new();

        // Near strand — within full_sim radius.
        let near_root = Vec3::new(0.0, 0.0, 1.0);
        let strand_near = HairStrand::new(near_root, 0.3, 0.5, 0.0);
        let near_tip_initial = strand_near.control_points[7];
        groom.add_strand(strand_near);

        // Far strand — well outside full_sim radius.
        let far_root = Vec3::new(0.0, 0.0, 100.0);
        let strand_far = HairStrand::new(far_root, 0.3, 0.5, 0.0);
        let far_tip_initial = strand_far.control_points[7];
        groom.add_strand(strand_far);

        groom.simulate(0.1, gravity, wind, camera);

        let near_tip_after = groom.strands[0].control_points[7];
        let far_tip_after = groom.strands[1].control_points[7];

        assert!(
            (near_tip_after - near_tip_initial).length() > 1e-6,
            "near strand tip should have moved"
        );
        assert!(
            (far_tip_after - far_tip_initial).length() < 1e-6,
            "far strand tip should not have moved"
        );
    }

    #[test]
    fn hair_strand_falls_under_gravity() {
        let root = Vec3::new(0.0, 1.0, 0.0);
        let mut strand = HairStrand::new(root, 0.3, 0.5, 0.0);
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let wind = Vec3::ZERO;
        let tip_initial_y = strand.control_points[7].y;

        for _ in 0..10 {
            strand.step(0.016, gravity, wind);
        }

        let tip_after_y = strand.control_points[7].y;
        assert!(
            tip_after_y < tip_initial_y,
            "tip y ({}) should be lower than initial y ({}) after gravity",
            tip_after_y,
            tip_initial_y
        );
    }
}
