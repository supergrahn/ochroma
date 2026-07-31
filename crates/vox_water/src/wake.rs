//! Kelvin wake generation for objects moving across a water surface.
//!
//! Ported from `forge-water`'s `sim::wake::WakeGenerator`. A displacement hull
//! trails a V-shaped wake confined to the **Kelvin half-angle**
//! `arcsin(1/3) ≈ 19.47°` — a constant of deep-water gravity-wave physics,
//! independent of hull speed (Kelvin, 1887). Inside that wedge the model sums a
//! transverse and a divergent component with a `1/√distance` amplitude decay.
//!
//! # Determinism
//!
//! Pure function of `(position, velocity, width, grid, )` — fixed row-major
//! iteration, no clock, no RNG. As with [`crate::ripples`], the caller decides
//! whether it is cosmetic (render clock) or replay truth (sim tick).

use glam::Vec3;

/// Below this speed (m/s) an object leaves no wake.
pub const MIN_WAKE_SPEED: f32 = 0.1;

/// Generates Kelvin wake height fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WakeGenerator {
    /// Kelvin half-angle (radians). `arcsin(1/3)` unless deliberately stylised.
    pub kelvin_angle: f32,
}

impl Default for WakeGenerator {
    fn default() -> Self {
        Self {
            kelvin_angle: (1.0f32 / 3.0).asin(),
        }
    }
}

impl WakeGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a Kelvin wake height field for one moving object.
    ///
    /// Returns a `grid_resolution²` row-major array spanning
    /// `[-grid_size/2, grid_size/2]` (endpoints inclusive) **relative to the
    /// object**, i.e. sample `[row * res + col]` sits at world
    /// `(object_position.x + coord(col), object_position.z + coord(row))`.
    /// Objects slower than [`MIN_WAKE_SPEED`] produce an all-zero field.
    #[must_use]
    pub fn generate_kelvin_wake(
        &self,
        object_position: Vec3,
        object_velocity: Vec3,
        object_width: f32,
        grid_resolution: usize,
        grid_size: f32,
    ) -> Vec<f32> {
        let res = grid_resolution;
        let mut wake = vec![0.0f32; res * res];
        if res == 0 || !(object_width > 0.0) {
            return wake;
        }

        let speed = object_velocity.length();
        if speed < MIN_WAKE_SPEED {
            return wake;
        }
        let vel_dir = object_velocity / speed;
        let (cos_theta, sin_theta) = (vel_dir.x, vel_dir.z);

        let base_amplitude = 0.1 * (speed / 10.0).powi(2) * object_width;
        let k_transverse = std::f32::consts::TAU / (object_width * 2.0);
        let k_divergent = std::f32::consts::TAU / object_width;

        let coord = |n: usize| -> f32 {
            if res == 1 {
                -grid_size / 2.0
            } else {
                -grid_size / 2.0 + grid_size * n as f32 / (res - 1) as f32
            }
        };

        for row in 0..res {
            let z = coord(row) + object_position.z;
            for col in 0..res {
                let x = coord(col) + object_position.x;

                // Rotate into the hull frame: +x_rot is ahead of the bow.
                let x_rot = x * cos_theta + z * sin_theta;
                let z_rot = -x * sin_theta + z * cos_theta;

                // Only astern of the hull carries a wake.
                if x_rot >= 0.0 {
                    continue;
                }
                let lateral = z_rot.abs();
                let longitudinal = x_rot.abs();
                if lateral.atan2(longitudinal) >= self.kelvin_angle {
                    continue;
                }

                let dist = (longitudinal * longitudinal + lateral * lateral)
                    .sqrt()
                    .max(1.0);
                let amplitude = base_amplitude / dist.sqrt();
                let fade = (-0.05 * longitudinal).exp();

                let transverse = amplitude * (k_transverse * z_rot).sin() * fade;
                let divergent = amplitude * 0.5 * (k_divergent * (x_rot + z_rot)).sin() * fade;
                wake[row * res + col] = transverse + divergent;
            }
        }
        wake
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kelvin_half_angle_is_19_47_degrees() {
        assert!((WakeGenerator::new().kelvin_angle.to_degrees() - 19.4712).abs() < 1e-3);
    }

    #[test]
    fn stationary_object_leaves_no_wake() {
        let g = WakeGenerator::new();
        let h = g.generate_kelvin_wake(Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), 5.0, 32, 100.0);
        assert!(h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn wake_lies_astern_and_inside_the_kelvin_wedge() {
        let g = WakeGenerator::new();
        let (res, size) = (65usize, 100.0f32);
        // Moving +X from the origin: the wake must be at negative X only, and every
        // wet cell must be inside the 19.47 degree wedge.
        let h = g.generate_kelvin_wake(Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0), 5.0, res, size);
        let coord = |n: usize| -size / 2.0 + size * n as f32 / (res - 1) as f32;
        let mut wet = 0usize;
        for row in 0..res {
            for col in 0..res {
                if h[row * res + col].abs() <= 1e-9 {
                    continue;
                }
                wet += 1;
                let (x, z) = (coord(col), coord(row));
                assert!(x < 0.0, "wake ahead of the bow at x={x}");
                let angle = z.abs().atan2(x.abs());
                assert!(
                    angle < g.kelvin_angle + 1e-4,
                    "cell outside the Kelvin wedge: {} deg",
                    angle.to_degrees()
                );
            }
        }
        assert!(
            wet > 20,
            "expected a substantial wake region, got {wet} cells"
        );
    }

    #[test]
    fn faster_hull_makes_a_bigger_wake() {
        let g = WakeGenerator::new();
        let peak = |speed: f32| {
            g.generate_kelvin_wake(Vec3::ZERO, Vec3::new(speed, 0.0, 0.0), 5.0, 64, 100.0)
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()))
        };
        let (slow, fast) = (peak(4.0), peak(12.0));
        // base_amplitude goes as speed^2 -> 3x speed is ~9x amplitude.
        assert!(fast > slow * 5.0, "fast={fast} slow={slow}");
    }

    #[test]
    fn wake_rotates_with_the_heading() {
        let g = WakeGenerator::new();
        let along_x = g.generate_kelvin_wake(Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0), 5.0, 33, 80.0);
        let along_z = g.generate_kelvin_wake(Vec3::ZERO, Vec3::new(0.0, 0.0, 8.0), 5.0, 33, 80.0);
        assert!(along_x != along_z, "wake did not follow the heading");
        // Same amount of water disturbed, just pointed elsewhere.
        let count = |f: &[f32]| f.iter().filter(|v| v.abs() > 1e-9).count();
        let (a, b) = (count(&along_x), count(&along_z));
        assert!(
            (a as i64 - b as i64).abs() <= 4,
            "wedge area changed with heading: {a} vs {b}"
        );
    }

    #[test]
    fn zero_width_hull_is_rejected_without_nan() {
        let g = WakeGenerator::new();
        let h = g.generate_kelvin_wake(Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0), 0.0, 16, 50.0);
        assert!(h.iter().all(|v| *v == 0.0 && v.is_finite()));
    }
}
