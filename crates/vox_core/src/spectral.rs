use serde::{Serialize, Deserialize};

pub const BAND_WAVELENGTHS: [f32; 8] = [380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0];
pub const BAND_SPACING: f32 = 40.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralBands(pub [f32; 8]);

#[derive(Debug, Clone)]
pub struct Illuminant {
    pub bands: [f32; 8],
}

// CIE 1931 2° observer at our 8 bands
const CIE_X: [f32; 8] = [0.0014, 0.0434, 0.3362, 0.0049, 0.2904, 0.9163, 0.5419, 0.0874];
const CIE_Y: [f32; 8] = [0.0000, 0.0116, 0.0600, 0.3230, 0.9540, 0.8700, 0.3810, 0.0468];
const CIE_Z: [f32; 8] = [0.0065, 0.2074, 1.7721, 0.2720, 0.0633, 0.0017, 0.0017, 0.0000];

impl Illuminant {
    pub fn d65() -> Self {
        Self { bands: [49.98, 68.70, 100.15, 109.35, 104.05, 97.74, 86.56, 74.35] }
    }
    pub fn d50() -> Self {
        Self { bands: [25.83, 52.93, 86.68, 100.00, 100.76, 97.74, 84.34, 70.06] }
    }
    pub fn a() -> Self {
        Self { bands: [9.80, 17.68, 29.49, 45.78, 66.29, 90.01, 115.92, 142.08] }
    }
    pub fn f11() -> Self {
        Self { bands: [3.00, 15.00, 60.00, 40.00, 80.00, 120.00, 55.00, 15.00] }
    }
}

pub fn spectral_to_xyz(spd: &SpectralBands, illuminant: &Illuminant) -> [f32; 3] {
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut z = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    let mut norm_z = 0.0f32;

    for i in 0..8 {
        let power = spd.0[i] * illuminant.bands[i];
        x += power * CIE_X[i] * BAND_SPACING;
        y += power * CIE_Y[i] * BAND_SPACING;
        z += power * CIE_Z[i] * BAND_SPACING;
        norm_x += illuminant.bands[i] * CIE_X[i] * BAND_SPACING;
        norm_y += illuminant.bands[i] * CIE_Y[i] * BAND_SPACING;
        norm_z += illuminant.bands[i] * CIE_Z[i] * BAND_SPACING;
    }

    if norm_y > 0.0 {
        let nx = if norm_x > 0.0 { norm_x } else { norm_y };
        let nz = if norm_z > 0.0 { norm_z } else { norm_y };
        // Normalize each channel by illuminant's own XYZ integral, then
        // scale to the CIE D65 white point so the sRGB matrix maps white to [1,1,1].
        let xn = (x / nx) * 0.9505;
        let yn = y / norm_y;
        let zn = (z / nz) * 1.0888;
        [xn, yn, zn]
    } else {
        [0.0, 0.0, 0.0]
    }
}

pub fn xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz;
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

pub fn linear_to_srgb_gamma(c: f32) -> f32 {
    if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Convert linear RGB to approximate 8-band spectral reflectance.
///
/// The 8 bands span 380-720nm at ~40nm intervals. This is a rough approximation
/// using simple primary decomposition: R peaks at 620nm, G at 540nm, B at 460nm.
pub fn rgb_to_spectral(r: f32, g: f32, b: f32) -> [u16; 8] {
    use half::f16;
    let bands = [
        b * 0.3,                // 380nm — violet, mostly blue
        b * 0.7,                // 420nm — blue
        b * 1.0,                // 460nm — peak blue
        g * 0.4 + b * 0.2,     // 500nm — cyan/green transition
        g * 1.0,                // 540nm — peak green
        r * 0.4 + g * 0.3,     // 580nm — yellow
        r * 1.0,                // 620nm — peak red
        r * 0.6,                // 660nm — deep red falloff
    ];
    std::array::from_fn(|i| f16::from_f32(bands[i].clamp(0.0, 1.0)).to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_to_xyz_white_illuminant_produces_nonzero() {
        let white = SpectralBands([1.0; 8]);
        let xyz = spectral_to_xyz(&white, &Illuminant::d65());
        assert!(xyz[0] > 0.0, "X should be positive for white SPD");
        assert!(xyz[1] > 0.0, "Y should be positive for white SPD");
        assert!(xyz[2] > 0.0, "Z should be positive for white SPD");
    }

    #[test]
    fn spectral_to_xyz_black_is_zero() {
        let black = SpectralBands([0.0; 8]);
        let xyz = spectral_to_xyz(&black, &Illuminant::d65());
        assert_eq!(xyz, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn spectral_to_xyz_zero_illuminant_is_zero() {
        let white = SpectralBands([1.0; 8]);
        let dark = Illuminant { bands: [0.0; 8] };
        let xyz = spectral_to_xyz(&white, &dark);
        assert_eq!(xyz, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgb_to_spectral_red_peaks_at_band_6() {
        use half::f16;
        let s = rgb_to_spectral(1.0, 0.0, 0.0);
        let band6 = f16::from_bits(s[6]).to_f32();
        assert!(band6 > 0.9, "pure red should peak at band 6 (620nm), got {}", band6);
        let band2 = f16::from_bits(s[2]).to_f32();
        assert!(band2 < 0.01, "pure red should have ~zero at band 2 (460nm), got {}", band2);
    }

    #[test]
    fn rgb_to_spectral_black_is_all_zero() {
        let s = rgb_to_spectral(0.0, 0.0, 0.0);
        for (i, &v) in s.iter().enumerate() {
            assert_eq!(v, 0, "black should produce zero at band {}", i);
        }
    }

    #[test]
    fn xyz_to_srgb_black_is_zero() {
        let rgb = xyz_to_srgb([0.0, 0.0, 0.0]);
        assert_eq!(rgb, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn linear_to_srgb_gamma_zero_is_zero() {
        assert_eq!(linear_to_srgb_gamma(0.0), 0.0);
    }

    #[test]
    fn linear_to_srgb_gamma_one_is_one() {
        let g = linear_to_srgb_gamma(1.0);
        assert!((g - 1.0).abs() < 0.001, "gamma(1.0) should be ~1.0, got {}", g);
    }
}
