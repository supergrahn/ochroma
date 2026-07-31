//! Concentric ripple propagation from point disturbances (rain impacts, dropped
//! objects, oars, pier pilings).
//!
//! Ported from `forge-water`'s `sim::ripples::RippleSimulator`, which was itself a
//! port of the deprecated Python `hydroprime.interaction.ripples`. The model is a
//! sum of damped circular gravity waves:
//!
//! ```text
//! h(r, t) = Σ  A · sin(k·r − ω·t) · exp(−r / L)
//! ```
//!
//! with the deep-water dispersion relation `k = ω² / g` (`ω = 2π f`) and an
//! exponential decay length `L`.
//!
//! # Determinism
//!
//! [`RippleSimulator::simulate`] is a pure function of `(ripples, time, grid)` —
//! it allocates one output buffer, iterates a fixed row-major order and reads no
//! clock. The **caller** picks the side of the line: pass a render wall-clock
//! `time` and it is cosmetic; pass sim-tick seconds and it is replay-exact.
//! [`RippleSimulator::ripples`] is a `Vec`, so source iteration order is insertion
//! order — never a `HashMap`.

use glam::Vec3;

use crate::GRAVITY;

/// Default e-folding decay length (m) — a ripple has faded to ~2 % of its
/// amplitude by `4 · DEFAULT_DECAY_LENGTH`.
pub const DEFAULT_DECAY_LENGTH: f32 = 20.0;

/// A single ripple source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ripple {
    /// World-space origin. Only `x` and `z` are used; `y` is carried for callers.
    pub position: Vec3,
    /// Crest amplitude (m) at the origin, before decay.
    pub amplitude: f32,
    /// Temporal frequency (Hz).
    pub frequency: f32,
    /// e-folding decay length (m).
    pub decay_length: f32,
}

impl Ripple {
    /// A ripple with the default decay length.
    #[must_use]
    pub fn new(position: Vec3, amplitude: f32, frequency: f32) -> Self {
        Self {
            position,
            amplitude,
            frequency,
            decay_length: DEFAULT_DECAY_LENGTH,
        }
    }

    /// Surface height contribution of this ripple alone at world `(x, z)`, `time`.
    #[must_use]
    pub fn height_at(&self, x: f32, z: f32, time: f32) -> f32 {
        let omega = std::f32::consts::TAU * self.frequency;
        let k = omega * omega / GRAVITY;
        let dx = x - self.position.x;
        let dz = z - self.position.z;
        let r = (dx * dx + dz * dz).sqrt();
        let decay = if self.decay_length > 1e-6 {
            (-r / self.decay_length).exp()
        } else {
            0.0
        };
        self.amplitude * (k * r - omega * time).sin() * decay
    }
}

/// Sums circular ripples onto a square height grid.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RippleSimulator {
    ripples: Vec<Ripple>,
}

impl RippleSimulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a ripple source with the default decay length.
    pub fn add_ripple(&mut self, position: Vec3, amplitude: f32, frequency: f32) {
        self.ripples
            .push(Ripple::new(position, amplitude, frequency));
    }

    /// Add a fully specified ripple source.
    pub fn push(&mut self, ripple: Ripple) {
        self.ripples.push(ripple);
    }

    /// Drop every source older than the caller's policy allows. Retains insertion
    /// order, so determinism is preserved.
    pub fn retain(&mut self, keep: impl FnMut(&Ripple) -> bool) {
        self.ripples.retain(keep);
    }

    pub fn clear(&mut self) {
        self.ripples.clear();
    }

    #[must_use]
    pub fn ripples(&self) -> &[Ripple] {
        &self.ripples
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ripples.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ripples.is_empty()
    }

    /// Summed ripple height at one world point.
    #[must_use]
    pub fn height_at(&self, x: f32, z: f32, time: f32) -> f32 {
        let mut sum = 0.0;
        for ripple in &self.ripples {
            sum += ripple.height_at(x, z, time);
        }
        sum
    }

    /// Evaluate the summed ripple height field at `time`.
    ///
    /// Returns a `grid_resolution²` row-major array covering
    /// `[-grid_size/2, grid_size/2]` on both axes with endpoints inclusive.
    /// Index `[row * res + col]` is world `(x = coord(col), z = coord(row))`.
    #[must_use]
    pub fn simulate(&self, time: f32, grid_resolution: usize, grid_size: f32) -> Vec<f32> {
        let res = grid_resolution;
        let mut heights = vec![0.0f32; res * res];
        if res == 0 {
            return heights;
        }
        let coord = |n: usize| -> f32 {
            if res == 1 {
                -grid_size / 2.0
            } else {
                -grid_size / 2.0 + grid_size * n as f32 / (res - 1) as f32
            }
        };
        for row in 0..res {
            let z = coord(row);
            for col in 0..res {
                heights[row * res + col] = self.height_at(coord(col), z, time);
            }
        }
        heights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_field_is_zero() {
        let sim = RippleSimulator::new();
        let h = sim.simulate(0.0, 16, 100.0);
        assert_eq!(h.len(), 256);
        assert!(h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ripple_height_matches_the_analytic_wave_equation() {
        let mut sim = RippleSimulator::new();
        sim.add_ripple(Vec3::ZERO, 0.5, 2.0);
        // res=3, size=100 -> coords [-50, 0, 50]; cell (row=1,col=2) is x=50,z=0,r=50.
        let h = sim.simulate(0.0, 3, 100.0);
        let omega = std::f32::consts::TAU * 2.0;
        let k = omega * omega / GRAVITY;
        let expected = 0.5 * (k * 50.0f32).sin() * (-50.0f32 / DEFAULT_DECAY_LENGTH).exp();
        assert!(
            (h[1 * 3 + 2] - expected).abs() < 1e-5,
            "got {}",
            h[1 * 3 + 2]
        );
    }

    #[test]
    fn amplitude_decays_with_distance() {
        let r = Ripple::new(Vec3::ZERO, 1.0, 1.0);
        // Envelope only: compare the peak envelope, not the oscillating sample.
        let near = (-1.0f32 / DEFAULT_DECAY_LENGTH).exp();
        let far = (-40.0f32 / DEFAULT_DECAY_LENGTH).exp();
        assert!(far < near * 0.2, "decay too weak: {far} vs {near}");
        // And the field really is bounded by that envelope.
        assert!(r.height_at(40.0, 0.0, 0.0).abs() <= far + 1e-6);
    }

    #[test]
    fn ripples_superpose_additively() {
        let mut a = RippleSimulator::new();
        a.add_ripple(Vec3::new(-10.0, 0.0, 0.0), 0.4, 1.5);
        let mut b = RippleSimulator::new();
        b.add_ripple(Vec3::new(12.0, 0.0, 3.0), 0.7, 2.5);
        let mut both = RippleSimulator::new();
        both.add_ripple(Vec3::new(-10.0, 0.0, 0.0), 0.4, 1.5);
        both.add_ripple(Vec3::new(12.0, 0.0, 3.0), 0.7, 2.5);

        let (ha, hb, hab) = (
            a.simulate(0.7, 8, 60.0),
            b.simulate(0.7, 8, 60.0),
            both.simulate(0.7, 8, 60.0),
        );
        for i in 0..hab.len() {
            assert!(
                (hab[i] - (ha[i] + hb[i])).abs() < 1e-5,
                "cell {i}: {} != {} + {}",
                hab[i],
                ha[i],
                hb[i]
            );
        }
    }

    #[test]
    fn time_advances_the_phase() {
        let mut sim = RippleSimulator::new();
        sim.add_ripple(Vec3::ZERO, 0.5, 2.0);
        let a = sim.simulate(0.0, 8, 50.0);
        let b = sim.simulate(0.3, 8, 50.0);
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    #[test]
    fn field_is_reproducible_for_the_same_inputs() {
        let mut sim = RippleSimulator::new();
        sim.add_ripple(Vec3::new(3.0, 0.0, -4.0), 0.25, 3.0);
        sim.add_ripple(Vec3::new(-7.0, 0.0, 1.0), 0.5, 1.0);
        assert_eq!(sim.simulate(1.25, 12, 40.0), sim.simulate(1.25, 12, 40.0));
    }
}
