//! Buoyancy and water-surface contact.
//!
//! Ported from `forge-water`'s `sim::collision::CollisionHandler`, and **fixed on
//! the way across**. The original treated submersion as *binary* — an object was
//! either fully displacing its whole volume or none of it, switching the instant
//! its centre crossed the waterline. That makes Archimedes' law a step function:
//! a floating hull cannot find an equilibrium draft, it can only oscillate between
//! zero force and a force many times its weight. Here submersion is the
//! **continuous** fraction of the object's vertical extent below the surface, so
//! a body settles smoothly at the draft where buoyancy balances weight.
//!
//! `F_b = ρ_water · g · V_submerged`, upward (+Y).
//!
//! # Determinism
//!
//! Pure arithmetic on `f32` — no state, no clock, no RNG. **Gameplay-safe**: fold
//! the resulting forces into the replay hash if they drive sim state.

use glam::Vec3;

use crate::GRAVITY;

/// Fresh water density (kg/m³).
pub const FRESH_WATER_DENSITY: f32 = 1000.0;
/// Sea water density (kg/m³) at typical salinity/temperature.
pub const SEA_WATER_DENSITY: f32 = 1025.0;

/// Computes buoyancy forces and water-surface contact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Buoyancy {
    /// Water density (kg/m³).
    pub water_density: f32,
    pub gravity: f32,
}

impl Default for Buoyancy {
    fn default() -> Self {
        Self {
            water_density: FRESH_WATER_DENSITY,
            gravity: GRAVITY,
        }
    }
}

impl Buoyancy {
    #[must_use]
    pub fn new(water_density: f32) -> Self {
        Self {
            water_density,
            gravity: GRAVITY,
        }
    }

    /// Sea water preset.
    #[must_use]
    pub fn sea() -> Self {
        Self::new(SEA_WATER_DENSITY)
    }

    /// The fraction `0..1` of an object's vertical extent that is under water.
    ///
    /// `center_y` is the object's centre, `height` its full vertical extent. A
    /// zero-height object degenerates to the binary test at its centre.
    #[must_use]
    pub fn submerged_fraction(&self, center_y: f32, height: f32, water_height: f32) -> f32 {
        if !(height > 0.0) {
            return if center_y < water_height { 1.0 } else { 0.0 };
        }
        let bottom = center_y - height * 0.5;
        ((water_height - bottom) / height).clamp(0.0, 1.0)
    }

    /// Buoyant force on an object of `volume` m³ whose vertical extent is
    /// `height` m, centred at `position`, under a surface at `water_height`.
    ///
    /// The displaced volume is `volume × submerged_fraction`, i.e. the object is
    /// assumed to have uniform cross-section over its height — the standard
    /// approximation for a hull or a crate and continuous everywhere.
    #[must_use]
    pub fn force(&self, position: Vec3, volume: f32, height: f32, water_height: f32) -> Vec3 {
        let fraction = self.submerged_fraction(position.y, height, water_height);
        Vec3::new(
            0.0,
            self.water_density * self.gravity * volume.max(0.0) * fraction,
            0.0,
        )
    }

    /// The equilibrium draft: the fraction of an object's height that sits below
    /// the surface when buoyancy exactly balances its weight, for an object of
    /// mean density `object_density` (kg/m³). Returns `1.0` (it sinks) when the
    /// object is denser than the water.
    #[must_use]
    pub fn equilibrium_draft_fraction(&self, object_density: f32) -> f32 {
        if !(self.water_density > 0.0) {
            return 0.0;
        }
        (object_density / self.water_density).clamp(0.0, 1.0)
    }

    /// True when the object is touching or below the water surface.
    #[must_use]
    pub fn touches_surface(&self, position: Vec3, water_height: f32) -> bool {
        position.y <= water_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_submerged_force_is_archimedes() {
        let b = Buoyancy::default();
        // V = 2 m³ fully under: F = 1000 * 9.81 * 2 = 19 620 N up.
        let f = b.force(Vec3::new(0.0, -5.0, 0.0), 2.0, 1.0, 0.0);
        assert!((f.y - 19620.0).abs() < 1e-2, "F = {}", f.y);
        assert_eq!((f.x, f.z), (0.0, 0.0));
    }

    #[test]
    fn object_clear_of_the_water_has_no_buoyancy() {
        let b = Buoyancy::default();
        assert_eq!(b.force(Vec3::new(0.0, 5.0, 0.0), 2.0, 1.0, 1.0).y, 0.0);
    }

    #[test]
    fn submersion_is_continuous_not_a_step() {
        // THE FIX over the forge original: half-submerged means half the force.
        let b = Buoyancy::default();
        let half = b.force(Vec3::new(0.0, 0.0, 0.0), 2.0, 2.0, 0.0);
        let full = b.force(Vec3::new(0.0, -10.0, 0.0), 2.0, 2.0, 0.0);
        assert!((half.y - full.y * 0.5).abs() < 1e-2, "half = {}", half.y);
        // And it varies smoothly across the waterline rather than jumping.
        let mut previous = 0.0f32;
        for step in 0..=10 {
            let center_y = 1.0 - step as f32 * 0.2; // +1.0 (clear) down to -1.0 (under)
            let f = b.force(Vec3::new(0.0, center_y, 0.0), 2.0, 2.0, 0.0).y;
            assert!(f >= previous - 1e-3, "force must rise as the object sinks");
            assert!(
                f - previous <= full.y * 0.11 + 1e-3,
                "force jumped {} N in one 10% step — not continuous",
                f - previous
            );
            previous = f;
        }
        assert!((previous - full.y).abs() < 1e-2);
    }

    #[test]
    fn sea_water_lifts_harder_than_fresh() {
        let fresh = Buoyancy::default().force(Vec3::new(0.0, -1.0, 0.0), 1.0, 0.5, 0.0);
        let sea = Buoyancy::sea().force(Vec3::new(0.0, -1.0, 0.0), 1.0, 0.5, 0.0);
        assert!((sea.y - 10055.25).abs() < 1e-2, "sea F = {}", sea.y);
        assert!(sea.y > fresh.y);
    }

    #[test]
    fn equilibrium_draft_matches_the_density_ratio() {
        let b = Buoyancy::sea();
        // Ice (917 kg/m³) in sea water: ~89.5 % submerged — the "tip of the iceberg".
        let ice = b.equilibrium_draft_fraction(917.0);
        assert!((ice - 0.894634).abs() < 1e-4, "ice draft = {ice}");
        // Denser than water -> sinks (fully submerged).
        assert_eq!(b.equilibrium_draft_fraction(7800.0), 1.0);
    }

    #[test]
    fn a_body_at_its_equilibrium_draft_is_in_balance() {
        // Buoyant force must equal weight when the draft matches the density ratio.
        let b = Buoyancy::sea();
        let (volume, height, density) = (3.0f32, 1.5f32, 600.0f32);
        let draft = b.equilibrium_draft_fraction(density);
        // Place the centre so exactly `draft` of the height is under y = 0.
        let center_y = height * 0.5 - draft * height;
        let lift = b
            .force(Vec3::new(0.0, center_y, 0.0), volume, height, 0.0)
            .y;
        let weight = density * volume * GRAVITY;
        assert!(
            (lift - weight).abs() < 1e-1,
            "lift {lift} vs weight {weight}"
        );
    }

    #[test]
    fn surface_contact_at_and_below_the_waterline() {
        let b = Buoyancy::default();
        assert!(b.touches_surface(Vec3::ZERO, 0.0));
        assert!(b.touches_surface(Vec3::new(0.0, -1.0, 0.0), 0.0));
        assert!(!b.touches_surface(Vec3::new(0.0, 0.5, 0.0), 0.0));
    }

    #[test]
    fn degenerate_inputs_stay_finite() {
        let b = Buoyancy::default();
        assert_eq!(b.force(Vec3::ZERO, -5.0, 1.0, 0.0).y, 0.0);
        assert!(b.force(Vec3::ZERO, 1.0, 0.0, 1.0).y.is_finite());
        assert_eq!(b.submerged_fraction(0.0, 0.0, 1.0), 1.0);
        assert_eq!(b.submerged_fraction(2.0, 0.0, 1.0), 0.0);
    }
}
