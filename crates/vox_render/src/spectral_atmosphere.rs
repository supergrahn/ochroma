//! Physically based spectral sky model.
//! Rayleigh scattering scales as λ⁻⁴: shorter wavelengths scatter more.
//!
//! Convention: `view_zenith_rad` is the **elevation angle** above the horizon.
//! `PI/2` = straight up (shortest atmospheric path, bluest sky).
//! `0.0`  = horizon (longest path, reddest sky).

pub const BAND_WAVELENGTHS_NM: [f32; 16] = [
    380.0, 405.0, 430.0, 455.0, 480.0, 505.0, 530.0, 555.0,
    580.0, 605.0, 630.0, 655.0, 680.0, 705.0, 730.0, 755.0,
];

/// Reference Rayleigh scattering cross-section at 550nm [km⁻¹]
const BETA_R_REF_KM: f32 = 0.0128; // per km at sea level
/// Mie scattering coefficient [km⁻¹]
const BETA_M_KM: f32 = 0.005;
/// Rayleigh scale height [km]
const H_R_KM: f32 = 8.0;
/// Mie scale height [km]
const H_M_KM: f32 = 1.2;
/// Atmosphere effective thickness [km]
const ATMO_THICKNESS_KM: f32 = 60.0;

pub struct AerosolProfile {
    pub haze_factor: f32,
}

pub struct SpectralAtmosphere {
    pub aerosol: AerosolProfile,
    /// Sun elevation above horizon in radians (0 = horizon, PI/2 = overhead)
    pub sun_zenith: f32,
    pub sun_azimuth: f32,
    /// Alias for sun_zenith used by Domain 12a wiring
    pub sun_elevation: f32,
    pub turbidity: f32,
}

impl SpectralAtmosphere {
    pub fn earth() -> Self {
        Self {
            aerosol: AerosolProfile { haze_factor: 1.0 },
            sun_zenith: std::f32::consts::FRAC_PI_4,
            sun_azimuth: 0.0,
            sun_elevation: std::f32::consts::FRAC_PI_4,
            turbidity: 2.0,
        }
    }

    /// Rayleigh scattering coefficient at sea level [km⁻¹]: scales as λ⁻⁴.
    pub fn beta_rayleigh(lambda_nm: f32) -> f32 {
        BETA_R_REF_KM * (550.0_f32 / lambda_nm).powi(4)
    }

    /// Optical depth along a path at `elevation_rad` (elevation above horizon).
    /// Integrates through exponentially-decreasing atmosphere column.
    fn optical_depth(elevation_rad: f32, lambda_nm: f32, haze: f32) -> f32 {
        // Air-mass factor: ratio of actual path length to vertical path
        // Clamp elevation to ~1° minimum to avoid infinity at the horizon
        let sin_elev = elevation_rad.sin().clamp(0.017, 1.0); // 1° minimum
        let air_mass = 1.0 / sin_elev;

        // Slant path length through atmosphere [km]
        let path_km = ATMO_THICKNESS_KM * air_mass;

        let beta_r = Self::beta_rayleigh(lambda_nm);
        let beta_m = BETA_M_KM * haze;

        // Numerical integration over 20 layers
        let steps = 20_u32;
        let ds = path_km / steps as f32; // [km per step]
        let mut tau = 0.0f32;
        for i in 0..steps {
            // Height increases as we travel up the slant path
            let h_km = (i as f32 + 0.5) * ds * sin_elev;
            let density_r = (-h_km / H_R_KM).exp();
            let density_m = (-h_km / H_M_KM).exp();
            tau += (beta_r * density_r + beta_m * density_m) * ds;
        }
        tau
    }

    /// Compute per-band sky radiance.
    ///
    /// `view_zenith_rad`: elevation above horizon in radians.
    /// `PI/2` → overhead (blue dominant); `~0` → near horizon (red dominant).
    pub fn sky_radiance(&self, view_zenith_rad: f32, _view_azimuth_rad: f32) -> [f32; 16] {
        let haze = self.aerosol.haze_factor;
        let mut radiance = [0.0_f32; 16];
        let mut max_val = f32::EPSILON;

        for (b, &lambda) in BAND_WAVELENGTHS_NM.iter().enumerate() {
            let tau_view = Self::optical_depth(view_zenith_rad, lambda, haze);
            let tau_sun = Self::optical_depth(self.sun_zenith, lambda, haze);

            // Transmittance: attenuation along view ray AND sun ray
            let transmittance = (-(tau_view + tau_sun)).exp();

            // In-scatter: Rayleigh term dominates at short wavelengths
            let beta_r = Self::beta_rayleigh(lambda);
            let beta_m_val = BETA_M_KM * haze;
            let in_scatter = (beta_r + beta_m_val * 0.5) * transmittance;
            radiance[b] = in_scatter;

            if radiance[b] > max_val {
                max_val = radiance[b];
            }
        }

        // Normalise to [0, 1]
        for v in &mut radiance {
            *v /= max_val;
        }
        radiance
    }

    /// Solar irradiance reaching the surface — normalised per-band.
    pub fn solar_irradiance(&self) -> [f32; 16] {
        let haze = self.aerosol.haze_factor;
        let mut irr = [0.0_f32; 16];
        let mut max_val = f32::EPSILON;
        for (b, &lambda) in BAND_WAVELENGTHS_NM.iter().enumerate() {
            let tau = Self::optical_depth(self.sun_zenith, lambda, haze);
            irr[b] = (-tau).exp();
            if irr[b] > max_val {
                max_val = irr[b];
            }
        }
        for v in &mut irr {
            *v /= max_val;
        }
        irr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_sky_has_more_short_wavelength_radiance() {
        let atmo = SpectralAtmosphere::earth();
        // PI/2 = elevation straight up — shortest path through atmosphere
        let zenith = atmo.sky_radiance(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(
            zenith[0] > zenith[15],
            "violet (band 0={}) should exceed NIR (band 15={}) at zenith",
            zenith[0],
            zenith[15]
        );
    }

    #[test]
    fn sunset_has_more_long_wavelength_radiance() {
        let atmo = SpectralAtmosphere::earth();
        // 0.02 ≈ near horizon — very long atmospheric path, Rayleigh scatters out short λ
        let horizon = atmo.sky_radiance(0.02, 0.0);
        assert!(
            horizon[12] > horizon[0],
            "red (band 12={}) should exceed violet (band 0={}) at sunset",
            horizon[12],
            horizon[0]
        );
    }

    #[test]
    fn radiance_is_normalised_to_unit_range() {
        let atmo = SpectralAtmosphere::earth();
        let r = atmo.sky_radiance(std::f32::consts::FRAC_PI_4, 0.0);
        for (i, &v) in r.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "band {} radiance {} out of [0,1]", i, v);
        }
    }

    // Domain 12a additional tests
    #[test]
    fn blue_sky_violet_exceeds_red() {
        let atmo = SpectralAtmosphere::earth();
        let radiance = atmo.sky_radiance(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(
            radiance[0] > radiance[15],
            "violet band 0 ({}) should exceed NIR band 15 ({}) — Rayleigh λ⁻⁴",
            radiance[0],
            radiance[15]
        );
    }

    #[test]
    fn horizon_is_redder_than_zenith() {
        let atmo = SpectralAtmosphere::earth();
        let zenith = atmo.sky_radiance(std::f32::consts::FRAC_PI_2, 0.0);
        let horizon = atmo.sky_radiance(0.05, 0.0);
        let zenith_ratio = zenith[0] / (zenith[15] + 1e-6);
        let horizon_ratio = horizon[0] / (horizon[15] + 1e-6);
        assert!(
            zenith_ratio > horizon_ratio,
            "zenith is bluer (ratio {:.2}) than horizon (ratio {:.2})",
            zenith_ratio,
            horizon_ratio
        );
    }

    #[test]
    fn solar_irradiance_in_unit_range() {
        let atmo = SpectralAtmosphere::earth();
        let irr = atmo.solar_irradiance();
        for (i, &v) in irr.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "band {} irradiance {} out of [0,1]", i, v);
        }
    }
}
