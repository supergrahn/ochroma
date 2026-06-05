//! `SpectralPhysics` — a single, clean, reachable facade that lets a game apply
//! an impact to a spectral material and get back the fracture / damage result,
//! driven entirely by the material's `[u16; 16]` spectral profile.
//!
//! Background
//! ----------
//! The crate already contains a fully-implemented spectral fracture / damage
//! layer (`spectral_fracture`, `spectral_resonance`, `spectral_damage`,
//! `destruction`) but it had no single callable entry point: a game had to
//! manually decode the profile, derive a resonance profile, call the plane
//! generator, count fragments and apply the spectral shift itself. This module
//! wires those pieces together behind one call:
//!
//! ```ignore
//! let result = SpectralPhysics::apply_impact(impact_pos, impulse_ns, &spectral);
//! ```
//!
//! Physics intuition
//! -----------------
//! A *brittle* material (glass: sharp UV/violet absorption, high spectral
//! variance, low total energy) shatters into many fragments under a given
//! impulse. A *ductile* material (metal: broad / flat absorption, high total
//! energy) absorbs the same impulse with little or no fragmentation. The
//! `brittleness` term below captures this: it is high when spectral variance is
//! high (sharp resonance) and total energy is low (little plastic capacity),
//! and low for flat high-energy profiles.

use glam::Vec3;
use vox_core::types::GaussianSplat;

use crate::destruction::spectral_shift_on_break;
use crate::spectral_fracture::{FracturePlane, SpectralResonanceFracture};
use crate::spectral_resonance::SpectralResonanceProfile;

/// Result of applying one impact to a spectral material.
#[derive(Debug, Clone)]
pub struct SpectralImpactResult {
    /// Whether the impulse exceeded the material's fracture threshold.
    pub fractured: bool,
    /// Brittleness of the material in `[0, 1]`. Glass ≈ 1, ductile metal ≈ 0.
    pub brittleness: f32,
    /// Fracture threshold (Ns-equivalent) derived from the spectral profile.
    pub threshold: f32,
    /// Resonance frequency (Hz) of the material — high for glass, low for wood/stone.
    pub resonance_hz: f32,
    /// Geometric fracture planes generated at the impact site. These are the
    /// *primary* crack surfaces (their count is clamped by the underlying
    /// resonance generator).
    pub planes: Vec<FracturePlane>,
    /// Total number of crack surfaces, including brittleness-driven secondary
    /// micro-fractures: a brittle material develops more cracks than a ductile
    /// one from the same set of primary planes. Always `>= planes.len()`.
    pub crack_surfaces: u32,
    /// Number of discrete fragments the object breaks into.
    /// `1` means "intact" (no fragmentation), larger means more shattering.
    pub fragment_count: u32,
}

impl SpectralImpactResult {
    /// Total number of distinct crack surfaces produced (primary planes plus
    /// brittleness-driven secondary micro-fractures).
    pub fn crack_count(&self) -> usize {
        self.crack_surfaces as usize
    }

    /// Number of *primary* geometric fracture planes (the renderable surfaces).
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }
}

/// Stateless facade over the spectral fracture / damage layer.
pub struct SpectralPhysics;

impl SpectralPhysics {
    /// Decode a `[u16; 16]` spectral profile to normalised `[f32; 16]` (0..1 per band).
    fn decode(spectral: &[u16; 16]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = spectral[i] as f32 / 65535.0;
        }
        out
    }

    /// Compute the brittleness of a material in `[0, 1]` from its spectral profile.
    ///
    /// Brittle (glass): sharp short-λ absorption → high spectral variance, and
    /// low total energy → little plastic / ductile capacity → brittleness near 1.
    /// Ductile (metal): broad, flat, high-energy absorption → low variance, high
    /// total energy → brittleness near 0.
    pub fn brittleness(spectral: &[u16; 16]) -> f32 {
        let p = Self::decode(spectral);
        let total: f32 = p.iter().sum();
        let mean = total / 16.0;
        let variance: f32 = p.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 16.0;

        // Sharpness in [0,1]: high spectral variance → sharp resonance → brittle.
        // Variance of a 0/1 alternating profile is ~0.25, so scale by 4.
        let sharpness = (variance * 4.0).clamp(0.0, 1.0);

        // Ductility in [0,1]: high average energy → more plastic capacity (metal).
        // mean is already in [0,1].
        let ductility = mean.clamp(0.0, 1.0);

        // Brittleness rewards sharpness and penalises ductility.
        (sharpness * (1.0 - ductility)).clamp(0.0, 1.0)
    }

    /// Apply a single impact to a spectral material and return the fracture /
    /// damage result, driven entirely by the `[u16; 16]` spectral profile.
    ///
    /// * `impact_pos` — world-space impact point.
    /// * `impulse_ns` — impulse magnitude in Newton-seconds.
    /// * `spectral`   — the material's 16-band spectral profile.
    pub fn apply_impact(
        impact_pos: Vec3,
        impulse_ns: f32,
        spectral: &[u16; 16],
    ) -> SpectralImpactResult {
        let brittleness = Self::brittleness(spectral);
        let threshold = SpectralResonanceFracture::fracture_threshold(spectral);
        let resonance_hz =
            SpectralResonanceProfile::from_spectral(&Self::decode(spectral)).resonance_hz;

        let planes =
            SpectralResonanceFracture::compute_planes(impact_pos, impulse_ns, spectral);
        let fractured = !planes.is_empty();

        // Crack surfaces: the primary planes, plus brittleness-driven secondary
        // micro-fractures. A brittle material (glass) develops extra cracks
        // radiating from each primary plane; a ductile one (metal) does not, so
        // its crack count stays at the primary plane count.
        let primary = planes.len() as f32;
        let crack_surfaces = if !fractured {
            0
        } else {
            // Each primary plane spawns up to ~3 secondary cracks scaled by
            // brittleness. Metal (brittleness 0) → secondaries 0 → crack_surfaces
            // == primary. Glass (brittleness ≈ 1) → ~3 extra per plane.
            let secondary = (primary * brittleness * 3.0).round();
            (primary + secondary).max(1.0) as u32
        };

        // Fragment count: an intact body is 1 fragment. Each crack surface can
        // split the body, and a brittle material multiplies the fragmentation
        // (shatter) while a ductile one resists it (stays near the primary count).
        let fragment_count = if !fractured {
            1
        } else {
            let shatter = 1.0 + crack_surfaces as f32 * (0.5 + brittleness * 4.0);
            shatter.round().max(1.0) as u32
        };

        SpectralImpactResult {
            fractured,
            brittleness,
            threshold,
            resonance_hz,
            planes,
            crack_surfaces,
            fragment_count,
        }
    }

    /// Build a normalised `[u16; 16]` material profile from a splat's spectral
    /// bands. A `GaussianSplat` stores spectral bands as f16 *bits*; this decodes
    /// each band to `[0,1]` and re-encodes it as the normalised `u16` profile the
    /// fracture model consumes (`value / 65535`).
    pub fn profile_from_splat(splat: &GaussianSplat) -> [u16; 16] {
        let mut out = [0u16; 16];
        for (b, slot) in out.iter_mut().enumerate() {
            let v = splat.spectral_f32(b).clamp(0.0, 1.0);
            *slot = (v * 65535.0).round() as u16;
        }
        out
    }

    /// Apply an impact to a concrete renderable splat in place: compute the
    /// fracture result from the splat's own spectral profile, and — if the
    /// material fractured — apply the on-break spectral shift to the splat so
    /// the visible material darkens / cracks. Returns the impact result.
    pub fn apply_impact_to_splat(
        splat: &mut GaussianSplat,
        impact_pos: Vec3,
        impulse_ns: f32,
    ) -> SpectralImpactResult {
        let profile = Self::profile_from_splat(splat);
        let result = Self::apply_impact(impact_pos, impulse_ns, &profile);
        if result.fractured {
            spectral_shift_on_break(splat);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// Ductile metal: broad, flat, high-energy absorption across all bands.
    fn metal_spectral() -> [u16; 16] {
        [60000u16; 16]
    }

    /// Brittle glass: sharp short-λ absorption — alternating high/low bands,
    /// low total energy. Characteristic of a transparent crystalline solid.
    fn glass_spectral() -> [u16; 16] {
        [
            60000, 100, 60000, 100, 60000, 100, 60000, 100, 60000, 100, 60000,
            100, 60000, 100, 60000, 100,
        ]
    }

    /// A glass splat whose stored f16 bands decode to a sharp, alternating
    /// high/low profile (the renderable analogue of `glass_spectral`).
    fn glass_splat() -> GaussianSplat {
        let mut s = [0u16; 16];
        for b in 0..16 {
            let v = if b % 2 == 0 { 0.9f32 } else { 0.002f32 };
            s[b] = half::f16::from_f32(v).to_bits();
        }
        GaussianSplat::volume([0.0; 3], [1.0; 3], Quat::IDENTITY, 255, s)
    }

    #[test]
    fn brittle_glass_fractures_more_than_ductile_metal_same_impulse() {
        // Same world impact, same impulse, only the spectral profile differs.
        let impact = Vec3::new(1.0, 2.0, 3.0);
        let impulse = 200_000.0;

        let glass = SpectralPhysics::apply_impact(impact, impulse, &glass_spectral());
        let metal = SpectralPhysics::apply_impact(impact, impulse, &metal_spectral());

        // Both must actually fracture under this large impulse — otherwise the
        // comparison is vacuous.
        assert!(glass.fractured, "glass must fracture under {impulse} Ns");
        assert!(metal.fractured, "metal must fracture under {impulse} Ns");

        // Glass is the brittle one.
        assert!(
            glass.brittleness > metal.brittleness,
            "glass brittleness {} must exceed metal brittleness {}",
            glass.brittleness,
            metal.brittleness
        );

        // Glass produces strictly more crack surfaces than metal under the
        // identical impulse (real ordered crack counts — its lower total energy
        // means a lower threshold, hence a larger impulse/threshold ratio).
        assert!(
            glass.crack_count() > metal.crack_count(),
            "glass crack_count {} must exceed metal crack_count {}",
            glass.crack_count(),
            metal.crack_count()
        );

        // And strictly more fragments (the headline outcome).
        assert!(
            glass.fragment_count > metal.fragment_count,
            "glass fragment_count {} must exceed metal fragment_count {}",
            glass.fragment_count,
            metal.fragment_count
        );
    }

    #[test]
    fn fragment_count_is_one_when_below_threshold() {
        // Tiny impulse, well below either material's fracture threshold.
        let result = SpectralPhysics::apply_impact(Vec3::ZERO, 0.0001, &metal_spectral());
        assert!(!result.fractured, "no fracture below threshold");
        assert_eq!(result.crack_count(), 0, "no crack planes below threshold");
        assert_eq!(
            result.fragment_count, 1,
            "an intact body is a single fragment"
        );
    }

    #[test]
    fn multi_step_increasing_impulse_grows_glass_fragments_monotonically() {
        // Drive the same glass material with three increasing impulses and
        // confirm the fragment count is non-decreasing and ultimately grows.
        let glass = glass_spectral();
        let threshold = SpectralResonanceFracture::fracture_threshold(&glass);

        let r_low = SpectralPhysics::apply_impact(Vec3::ZERO, threshold * 1.1, &glass);
        let r_mid = SpectralPhysics::apply_impact(Vec3::ZERO, threshold * 4.0, &glass);
        let r_high = SpectralPhysics::apply_impact(Vec3::ZERO, threshold * 16.0, &glass);

        assert!(r_low.fractured && r_mid.fractured && r_high.fractured);

        // Crack-plane count is non-decreasing with impulse and the strongest hit
        // produces strictly more cracks than the weakest.
        assert!(
            r_low.crack_count() <= r_mid.crack_count()
                && r_mid.crack_count() <= r_high.crack_count(),
            "crack counts must be non-decreasing: {} {} {}",
            r_low.crack_count(),
            r_mid.crack_count(),
            r_high.crack_count()
        );
        assert!(
            r_high.fragment_count > r_low.fragment_count,
            "strongest impact ({} frags) must shatter more than weakest ({} frags)",
            r_high.fragment_count,
            r_low.fragment_count
        );
    }

    #[test]
    fn apply_impact_to_splat_shifts_spectral_on_fracture() {
        // A glass splat hit hard: it fractures and the on-break spectral shift
        // reduces its UV/violet band 0 (factor 0.6 in spectral_shift_on_break).
        let mut splat = glass_splat();
        let band0_before = splat.spectral_f32(0);
        assert!(band0_before > 0.1, "glass band 0 should start bright");

        let result =
            SpectralPhysics::apply_impact_to_splat(&mut splat, Vec3::ZERO, 200_000.0);
        assert!(result.fractured, "hard hit on glass must fracture");

        let band0_after = splat.spectral_f32(0);
        // spectral_shift_on_break multiplies bands 0..3 by 0.6.
        assert!(
            band0_after < band0_before * 0.7,
            "fractured glass band 0 must drop: before={band0_before} after={band0_after}"
        );
    }

    #[test]
    fn sub_threshold_impact_does_not_shift_splat_spectral() {
        // Below threshold: no fracture, splat spectral unchanged.
        let mut splat = glass_splat();
        let band0_before = splat.spectral_f32(0);

        let result =
            SpectralPhysics::apply_impact_to_splat(&mut splat, Vec3::ZERO, 0.00001);
        assert!(!result.fractured, "tiny impulse must not fracture");

        let band0_after = splat.spectral_f32(0);
        assert!(
            (band0_after - band0_before).abs() < 1e-6,
            "unfractured splat spectral must be unchanged: before={band0_before} after={band0_after}"
        );
    }
}
