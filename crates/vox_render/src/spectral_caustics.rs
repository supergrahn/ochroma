//! Spectral caustics via per-band Snell's law and Cauchy glass dispersion.
//!
//! Each of the 16 spectral bands refracts at a slightly different angle because
//! glass IOR varies with wavelength (dispersion). Chromatic aberration in the
//! caustic pattern emerges from first principles — no post-process required.

use glam::Vec3;

/// Centre wavelength of each spectral band in micrometres (µm).
pub const BAND_UM: [f32; 16] = [
    0.380, 0.405, 0.430, 0.455, 0.480, 0.505, 0.530, 0.555,
    0.580, 0.605, 0.630, 0.655, 0.680, 0.705, 0.730, 0.755,
];

/// Cauchy dispersion coefficients for borosilicate glass (N-BK7).
/// n(λ) = A + B/λ² + C/λ⁴  (λ in µm)
pub struct CauchyGlass {
    pub a: f32, // 1.5046 for N-BK7
    pub b: f32, // 0.00420 µm²
    pub c: f32, // 0.0000 µm⁴ (negligible for visible range)
}

impl CauchyGlass {
    /// N-BK7 borosilicate glass (standard optical glass).
    pub fn n_bk7() -> Self {
        Self { a: 1.5046, b: 0.00420, c: 0.0 }
    }

    /// Custom glass approximated from Abbe number.
    /// `nd`: IOR at 587nm (d-line). `vd`: Abbe number.
    pub fn from_abbe(nd: f32, vd: f32) -> Self {
        let b = (nd - 1.0) / (vd.max(10.0)) * 0.015;
        Self { a: nd - b / (0.587 * 0.587), b, c: 0.0 }
    }

    /// Compute IOR for a single wavelength in µm.
    pub fn ior(&self, lambda_um: f32) -> f32 {
        self.a + self.b / (lambda_um * lambda_um) + self.c / (lambda_um.powi(4))
    }

    /// Compute IOR for all 16 spectral bands.
    pub fn ior_bands(&self) -> [f32; 16] {
        std::array::from_fn(|i| self.ior(BAND_UM[i]))
    }
}

/// Spectral refraction — applies Snell's law per band.
pub struct SpectralCaustics;

impl SpectralCaustics {
    /// Refract a 16-band spectral ray through a glass interface.
    ///
    /// # Arguments
    /// * `incident_dir` — unit direction of incoming ray (pointing INTO surface)
    /// * `normal` — unit surface normal (pointing OUT of surface, towards incident medium)
    /// * `incident_spectral` — spectral intensity of the incoming ray per band
    /// * `glass` — Cauchy glass dispersion coefficients
    ///
    /// # Returns
    /// `SpectralRefraction` with per-band refracted directions and transmitted intensities.
    /// Bands undergoing total internal reflection have zero transmitted intensity.
    pub fn refract(
        incident_dir: Vec3,
        normal: Vec3,
        incident_spectral: [f32; 16],
        glass: &CauchyGlass,
    ) -> SpectralRefraction {
        let n_air = 1.0003_f32;
        let cos_i = (-incident_dir).dot(normal).clamp(-1.0, 1.0);
        let sin_i_sq = (1.0 - cos_i * cos_i).max(0.0);
        let sin_i = sin_i_sq.sqrt();

        let ior_bands = glass.ior_bands();
        let mut directions = [Vec3::ZERO; 16];
        let mut transmitted = [0.0f32; 16];

        for b in 0..16 {
            let n_ratio = n_air / ior_bands[b];
            let sin_t_sq = (n_ratio * sin_i).powi(2);

            if sin_t_sq > 1.0 {
                // Total internal reflection — no transmission in this band
                transmitted[b] = 0.0;
                directions[b] = Vec3::ZERO;
            } else {
                let cos_t = (1.0 - sin_t_sq).sqrt();
                // Refracted direction: Snell's law vector form
                let dir_t = n_ratio * incident_dir + (n_ratio * cos_i - cos_t) * normal;
                directions[b] = dir_t.normalize_or_zero();
                transmitted[b] = incident_spectral[b];

                // Fresnel transmittance (simplified, unpolarised)
                let r_s = ((n_air * cos_i - ior_bands[b] * cos_t)
                    / (n_air * cos_i + ior_bands[b] * cos_t)).powi(2);
                let r_p = ((ior_bands[b] * cos_i - n_air * cos_t)
                    / (ior_bands[b] * cos_i + n_air * cos_t)).powi(2);
                let reflectance = (r_s + r_p) * 0.5;
                transmitted[b] *= 1.0 - reflectance;
            }
        }

        SpectralRefraction { directions, transmitted }
    }

    /// Compute the angular spread between the shortest and longest wavelength
    /// refracted directions. This is the chromatic aberration angle in radians.
    pub fn chromatic_spread(refraction: &SpectralRefraction) -> f32 {
        let d0 = refraction.directions[0];
        let d15 = refraction.directions[15];
        if d0.length_squared() < 1e-6 || d15.length_squared() < 1e-6 {
            return 0.0;
        }
        d0.dot(d15).clamp(-1.0, 1.0).acos()
    }
}

/// Output of `SpectralCaustics::refract()`.
pub struct SpectralRefraction {
    /// Refracted direction per spectral band.
    pub directions: [Vec3; 16],
    /// Transmitted intensity per band (0 if total internal reflection).
    pub transmitted: [f32; 16],
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn n_bk7_violet_ior_exceeds_red() {
        let glass = CauchyGlass::n_bk7();
        let bands = glass.ior_bands();
        assert!(
            bands[0] > bands[15],
            "violet IOR ({:.4}) should exceed red IOR ({:.4}) — normal dispersion",
            bands[0], bands[15]
        );
    }

    #[test]
    fn n_bk7_violet_ior_approximately_1_530() {
        let glass = CauchyGlass::n_bk7();
        let n_violet = glass.ior(0.380);
        println!("N-BK7 at 380nm: {:.4}", n_violet);
        assert!(
            approx_eq(n_violet, 1.530, 0.005),
            "N-BK7 at 380nm should be ~1.530, got {:.4}", n_violet
        );
    }

    #[test]
    fn n_bk7_red_ior_approximately_1_513() {
        let glass = CauchyGlass::n_bk7();
        let n_red = glass.ior(0.660);
        println!("N-BK7 at 660nm: {:.4}", n_red);
        assert!(
            approx_eq(n_red, 1.513, 0.005),
            "N-BK7 at 660nm should be ~1.513, got {:.4}", n_red
        );
    }

    #[test]
    fn normal_incidence_preserves_direction() {
        let glass = CauchyGlass::n_bk7();
        let incident = Vec3::new(0.0, -1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let refraction = SpectralCaustics::refract(incident, normal, [1.0; 16], &glass);
        for b in 0..16 {
            let dir = refraction.directions[b];
            if dir.length_squared() > 0.5 {
                assert!(
                    dir.y < -0.99,
                    "band {} at normal incidence should go straight through, got {:?}", b, dir
                );
            }
        }
    }

    #[test]
    fn oblique_incidence_produces_chromatic_spread() {
        let glass = CauchyGlass::n_bk7();
        let angle = PI / 4.0;
        let incident = Vec3::new(angle.sin(), -angle.cos(), 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let refraction = SpectralCaustics::refract(incident, normal, [1.0; 16], &glass);
        let spread = SpectralCaustics::chromatic_spread(&refraction);
        println!("chromatic_spread at 45°: {}", spread);
        assert!(
            spread > 0.0,
            "oblique incidence should produce chromatic spread > 0, got {}", spread
        );
    }

    #[test]
    fn transmitted_values_in_unit_range() {
        let glass = CauchyGlass::n_bk7();
        let angle = 80.0_f32.to_radians();
        let incident = Vec3::new(angle.sin(), -angle.cos(), 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let refraction = SpectralCaustics::refract(incident, normal, [1.0; 16], &glass);
        for b in 0..16 {
            assert!(
                refraction.transmitted[b] >= 0.0 && refraction.transmitted[b] <= 1.0,
                "band {} transmitted {} out of [0,1]", b, refraction.transmitted[b]
            );
        }
    }

    #[test]
    fn violet_refracts_more_than_red_at_oblique_angle() {
        let glass = CauchyGlass::n_bk7();
        let angle = 45.0_f32.to_radians();
        let incident = Vec3::new(angle.sin(), -angle.cos(), 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let refraction = SpectralCaustics::refract(incident, normal, [1.0; 16], &glass);
        let x0 = refraction.directions[0].x;
        let x15 = refraction.directions[15].x;
        println!("violet x={:.5}, red x={:.5}", x0, x15);
        assert!(
            x0 < x15,
            "violet (x={:.5}) should refract more than red (x={:.5})", x0, x15
        );
    }
}
