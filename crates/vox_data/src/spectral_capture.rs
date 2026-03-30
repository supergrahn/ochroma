//! 3-photo spectral material capture.
//!
//! Estimates per-surface spectral reflectance from three photographs taken
//! under different known illuminants (daylight, tungsten, cool-LED).
//!
//! For each pixel the system solves: measured_rgb ≈ light_spd × reflectance
//! The result is a SpectralMaterialProfile with per-band mean and variance.

use crate::spectral_upsampler::SpectralUpsampler;

/// Spectral power distribution of a light source — energy in each of 16 bands.
#[derive(Debug, Clone, Copy)]
pub struct LightSpd(pub [f32; 16]);

impl LightSpd {
    /// Daylight D65 approximation (normalised).
    pub fn daylight() -> Self {
        Self([0.82, 0.84, 0.86, 0.88, 0.91, 0.94, 0.97, 0.98, 1.00, 0.99, 0.99, 0.98, 0.97, 0.96, 0.95, 0.95])
    }

    /// Tungsten / incandescent approximation (red-heavy).
    pub fn tungsten() -> Self {
        Self([0.15, 0.17, 0.20, 0.24, 0.28, 0.34, 0.40, 0.50, 0.60, 0.70, 0.80, 0.87, 0.93, 0.97, 1.00, 1.00])
    }

    /// Cool LED approximation (blue-heavy).
    pub fn cool_led() -> Self {
        Self([0.55, 0.65, 0.80, 0.95, 1.00, 0.95, 0.90, 0.80, 0.70, 0.65, 0.55, 0.47, 0.40, 0.35, 0.30, 0.28])
    }
}

/// Measured spectral reflectance profile for a material.
#[derive(Debug, Clone)]
pub struct SpectralMaterialProfile {
    /// Mean per-band reflectance across all sampled pixels.
    pub reflectance: [f32; 16],
    /// Per-band variance (confidence indicator).
    pub variance: [f32; 16],
}

impl SpectralMaterialProfile {
    /// Estimate spectral reflectance from three RGB photographs under known lights.
    ///
    /// Each photo is represented by its mean sRGB value over the material region.
    /// This is sufficient for the unit-test approximation; production uses per-pixel crops.
    pub fn from_three_photos(photos: [&[f32; 3]; 3], lights: [LightSpd; 3]) -> Self {
        // Upsample each photo's mean RGB to 16-band spectral measurement
        let measured: [[f32; 16]; 3] = [
            SpectralUpsampler::from_rgb(photos[0][0], photos[0][1], photos[0][2]),
            SpectralUpsampler::from_rgb(photos[1][0], photos[1][1], photos[1][2]),
            SpectralUpsampler::from_rgb(photos[2][0], photos[2][1], photos[2][2]),
        ];

        // For each band, estimate reflectance by weighted average: r[b] = mean(measured[i][b] / light[i][b])
        let mut reflectance = [0.0f32; 16];
        let mut variance = [0.0f32; 16];

        for b in 0..16 {
            let estimates: [f32; 3] = [
                (measured[0][b] / lights[0].0[b].max(1e-4)).clamp(0.0, 1.0),
                (measured[1][b] / lights[1].0[b].max(1e-4)).clamp(0.0, 1.0),
                (measured[2][b] / lights[2].0[b].max(1e-4)).clamp(0.0, 1.0),
            ];
            let mean = (estimates[0] + estimates[1] + estimates[2]) / 3.0;
            let var = estimates.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / 3.0;
            reflectance[b] = mean;
            variance[b] = var;
        }

        Self { reflectance, variance }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daylight_spd_peaks_at_green() {
        let d = LightSpd::daylight();
        let peak = d.0.iter().copied().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        println!("D65 peak band index = {}", peak);
        assert_eq!(peak, 8, "D65 peak should be band 8 (580nm), got band {}", peak);
    }

    #[test]
    fn tungsten_spd_peaks_at_red() {
        let t = LightSpd::tungsten();
        let peak = t.0.iter().copied().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        assert!(peak >= 14, "tungsten peak should be in the red/NIR bands (14-15), got band {}", peak);
    }

    #[test]
    fn three_photo_profile_in_unit_range() {
        let lights = [LightSpd::daylight(), LightSpd::tungsten(), LightSpd::cool_led()];
        let photos = [[0.5f32, 0.5, 0.5], [0.5f32, 0.45, 0.4], [0.45f32, 0.5, 0.55]];
        let profile = SpectralMaterialProfile::from_three_photos(
            [&photos[0], &photos[1], &photos[2]],
            lights,
        );
        for (i, &v) in profile.reflectance.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "reflectance[{}] = {} must be in [0,1]", i, v);
        }
    }

    #[test]
    fn three_photo_variance_is_nonneg() {
        let lights = [LightSpd::daylight(), LightSpd::tungsten(), LightSpd::cool_led()];
        let photos = [[1.0f32, 0.0, 0.0], [0.8f32, 0.1, 0.05], [0.7f32, 0.05, 0.1]];
        let profile = SpectralMaterialProfile::from_three_photos(
            [&photos[0], &photos[1], &photos[2]],
            lights,
        );
        for (i, &v) in profile.variance.iter().enumerate() {
            assert!(v >= 0.0, "variance[{}] = {} must be non-negative", i, v);
        }
    }

    #[test]
    fn gray_surface_has_flat_reflectance() {
        let lights = [LightSpd::daylight(), LightSpd::tungsten(), LightSpd::cool_led()];
        let gray = [0.5f32, 0.5, 0.5];
        let profile = SpectralMaterialProfile::from_three_photos([&gray, &gray, &gray], lights);
        let min = profile.reflectance.iter().copied().fold(f32::MAX, f32::min);
        let max = profile.reflectance.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            max - min < 0.4,
            "gray surface should have relatively flat reflectance, range was {:.3}",
            max - min
        );
    }
}
