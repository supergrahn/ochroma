//! Spectral resonance physics: fracture and acoustic emission from
//! optical-acoustic material coupling.
//!
//! Optical absorption spectra determine a material's internal resonance
//! frequency. Short-wavelength absorption → sharp resonance (glass);
//! broad absorption → low resonance (stone/wood).

use glam::Vec3;

// No per-band frequency table needed — resonance_hz is computed analytically below.

/// Material resonance profile derived from spectral reflectance data.
#[derive(Clone, Debug)]
pub struct SpectralResonanceProfile {
    /// Dominant resonance frequency in Hz.
    /// Glass (sharp short-λ absorption) → > 1000 Hz.
    /// Wood/stone (broad absorption) → < 800 Hz.
    pub resonance_hz: f32,
    /// Crystalline regularity in [0, 1]: 1.0 = perfectly crystalline (axis-aligned fractures).
    pub regularity: f32,
    /// Material stiffness in [0, 1]: proportion of energy in short-wavelength bands.
    pub stiffness: f32,
}

impl SpectralResonanceProfile {
    /// Derive resonance profile from normalised spectral values (0..1 per band).
    ///
    /// Resonance frequency is driven by the **ratio** of UV/violet absorption (bands 0-3)
    /// to total absorption. Glass: most absorption at short λ → high ratio → high resonance.
    /// Wood/stone: absorption spread across visible range → low ratio → low resonance.
    pub fn from_spectral(spectral: &[f32; 16]) -> Self {
        let total_weight: f32 = spectral.iter().sum::<f32>().max(1e-6);

        // Short-wavelength fraction: bands 0-3 (380-455nm, UV/violet)
        let uv_weight: f32 = spectral[..4].iter().sum::<f32>();
        let uv_fraction = uv_weight / total_weight;

        // Resonance: pure UV absorption (glass) → 8000 Hz; broad/mid absorption → 200 Hz
        // Linear interpolation scaled by sharpness
        let mean = total_weight / 16.0;
        let variance = spectral.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / 16.0;
        // Sharpness: UV fraction must be high AND variance high for glass-like resonance.
        // Use uv_fraction^2 to strongly penalise materials with spread absorption.
        // Glass: uv_fraction ≈ 0.89, variance ≈ 0.14 → sharpness ≈ min(0.89² × 0.14 × 100, 1) ≈ 1.0
        // Wood:  uv_fraction ≈ 0.18, variance ≈ 0.018 → sharpness ≈ 0.18² × 0.018 × 100 ≈ 0.0006
        let sharpness = (uv_fraction * uv_fraction * variance * 100.0).clamp(0.0, 1.0);
        let resonance_hz = 200.0 + sharpness * 7800.0;

        // Regularity: low spectral variance → crystalline → axis-aligned fractures
        let regularity = 1.0 / (1.0 + variance * 10.0);

        // Stiffness from short-wavelength (UV/violet) band energy
        let stiffness = (spectral[0] + spectral[1] + spectral[2] + spectral[3]) / 4.0;

        Self {
            resonance_hz,
            regularity,
            stiffness,
        }
    }
}

/// A single fracture plane in local object space.
#[derive(Clone, Debug)]
pub struct FractureResonancePlane {
    pub origin: Vec3,
    pub normal: Vec3,
}

/// Spectral-resonance-driven fracture plane generator.
pub struct SpectralFracture;

impl SpectralFracture {
    /// Compute fracture planes driven by spectral resonance profile.
    ///
    /// - `impact_local`: impact point in object-local space
    /// - `impulse_ns`:  impulse magnitude in Newton-seconds
    /// - `profile`:     resonance profile from `SpectralResonanceProfile::from_spectral()`
    /// - `count`:       number of planes to generate
    pub fn compute_planes(
        impact_local: Vec3,
        impulse_ns: f32,
        profile: &SpectralResonanceProfile,
        count: usize,
    ) -> Vec<FractureResonancePlane> {
        use std::f32::consts::TAU;
        let mut planes = Vec::with_capacity(count);
        let spread = (1.0 - profile.regularity) * std::f32::consts::FRAC_PI_2;

        for i in 0..count {
            let t = i as f32 / count as f32;
            let base_normal =
                Vec3::new((t * TAU).cos(), 0.0, (t * TAU).sin()).normalize();

            let normal = if profile.regularity > 0.7 {
                snap_to_axis(base_normal)
            } else {
                let perturb = Vec3::new(
                    (t * 7.3 + 1.1).sin() * spread,
                    (t * 4.7 + 0.5).cos() * spread,
                    (t * 11.1 + 2.3).sin() * spread * 0.5,
                );
                (base_normal + perturb).normalize()
            };

            let dist = (impulse_ns / (profile.stiffness.max(0.1) * 1000.0))
                .clamp(0.05, 0.5);
            let origin = impact_local + normal * dist * (t + 0.5);
            planes.push(FractureResonancePlane { origin, normal });
        }
        planes
    }
}

fn snap_to_axis(v: Vec3) -> Vec3 {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    if ax >= ay && ax >= az {
        Vec3::new(v.x.signum(), 0.0, 0.0)
    } else if ay >= ax && ay >= az {
        Vec3::new(0.0, v.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, v.z.signum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glass_profile_has_high_resonance_frequency() {
        // Glass: sharp absorption at short wavelengths (UV/violet), transparent at long
        let glass_spectral = [
            0.9f32, 0.9, 0.85, 0.80, 0.1, 0.05, 0.05, 0.05,
            0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
        ];
        let profile = SpectralResonanceProfile::from_spectral(&glass_spectral);
        println!("glass resonance_hz = {:.1} Hz", profile.resonance_hz);
        assert!(
            profile.resonance_hz > 1000.0,
            "glass should have high resonance freq, got {}Hz",
            profile.resonance_hz
        );
    }

    #[test]
    fn wood_profile_has_low_resonance_frequency() {
        // Wood: broad absorption across mid-range, peak in orange/green
        let wood_spectral = [
            0.1f32, 0.12, 0.15, 0.25, 0.45, 0.55, 0.40, 0.30,
            0.25, 0.20, 0.17, 0.15, 0.13, 0.12, 0.11, 0.10,
        ];
        let profile = SpectralResonanceProfile::from_spectral(&wood_spectral);
        println!("wood resonance_hz = {:.1} Hz", profile.resonance_hz);
        assert!(
            profile.resonance_hz < 800.0,
            "wood should have low resonance freq, got {}Hz",
            profile.resonance_hz
        );
    }

    #[test]
    fn fracture_planes_respect_crystalline_regularity() {
        // Crystal: uniform absorption → high regularity → axis-aligned planes
        let crystal = [0.8f32; 16];
        let profile = SpectralResonanceProfile::from_spectral(&crystal);
        let planes =
            SpectralFracture::compute_planes(Vec3::ZERO, 100.0, &profile, 8);
        for plane in &planes {
            let aligned = plane.normal.x.abs() > 0.9
                || plane.normal.y.abs() > 0.9
                || plane.normal.z.abs() > 0.9;
            assert!(
                aligned,
                "crystalline material should fracture in axis-aligned planes, got {:?}",
                plane.normal
            );
        }
    }
}
