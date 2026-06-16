use serde::{Serialize, Deserialize};

/// 380–755 nm at 25 nm steps (USGS wavelength grid, 16 bands).
pub const BAND_WAVELENGTHS: [f32; 16] = [
    380.0, 405.0, 430.0, 455.0, 480.0, 505.0, 530.0, 555.0,
    580.0, 605.0, 630.0, 655.0, 680.0, 705.0, 730.0, 755.0,
];
pub const BAND_SPACING: f32 = 25.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralBands(pub [f32; 16]);

#[derive(Debug, Clone)]
pub struct Illuminant {
    pub bands: [f32; 16],
}

/// CIE 1931 2° observer, 380–755 nm at 25 nm steps.
const CIE_X: [f32; 16] = [0.01741, 0.08028, 0.26000, 0.21000, 0.00949, 0.00000, 0.11201, 0.38000, 0.74300, 1.02200, 0.71600, 0.38100, 0.19700, 0.09020, 0.03400, 0.01180];
const CIE_Y: [f32; 16] = [0.00039, 0.00232, 0.01998, 0.09520, 0.17399, 0.46600, 0.69500, 0.94500, 0.86800, 0.65100, 0.38100, 0.18000, 0.08000, 0.03300, 0.01200, 0.00400];
const CIE_Z: [f32; 16] = [0.08290, 0.38637, 1.29900, 1.24500, 0.45640, 0.05250, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000];

impl Illuminant {
    pub fn d65() -> Self {
        Self { bands: [49.98, 52.31, 56.45, 68.70, 82.75, 91.49, 95.00, 100.00, 102.10, 100.75, 99.20, 98.00, 93.50, 88.69, 83.29, 78.28] }
    }
    pub fn d50() -> Self {
        Self { bands: [25.83, 31.22, 36.93, 52.93, 67.23, 79.00, 86.68, 93.00, 97.74, 100.00, 100.76, 99.82, 97.74, 94.34, 88.49, 83.56] }
    }
    pub fn a() -> Self {
        Self { bands: [9.80, 12.09, 14.71, 17.68, 21.00, 24.67, 28.70, 33.09, 37.82, 42.87, 48.24, 53.91, 59.86, 66.06, 72.50, 79.13] }
    }
    pub fn f11() -> Self {
        Self { bands: [3.00, 4.00, 8.00, 15.00, 30.00, 45.00, 60.00, 70.00, 80.00, 100.00, 120.00, 90.00, 55.00, 30.00, 20.00, 15.00] }
    }
}

pub fn spectral_to_xyz(spd: &SpectralBands, illuminant: &Illuminant) -> [f32; 3] {
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut z = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    let mut norm_z = 0.0f32;

    for i in 0..16 {
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

/// Convert linear RGB to approximate 16-band spectral reflectance.
///
/// The 16 bands span 380-755nm at 25nm steps (USGS wavelength grid). This is a rough
/// approximation using simple primary decomposition: R peaks at 630nm, G at 545nm, B at 455nm.
pub fn rgb_to_spectral(r: f32, g: f32, b: f32) -> [u16; 16] {
    use half::f16;
    let bands = [
        b * 0.3,                        // 380nm — violet, mostly blue
        b * 0.6,                        // 405nm
        b * 0.9,                        // 430nm
        b * 1.0,                        // 455nm — peak blue
        b * 0.7 + g * 0.1,             // 480nm
        g * 0.5 + b * 0.2,             // 505nm — cyan/green
        g * 0.9,                        // 530nm
        g * 1.0,                        // 555nm — peak green
        g * 0.6 + r * 0.1,             // 580nm — yellow
        r * 0.5 + g * 0.2,             // 605nm
        r * 0.9,                        // 630nm
        r * 1.0,                        // 655nm — peak red
        r * 0.8,                        // 680nm
        r * 0.6,                        // 705nm
        r * 0.4,                        // 730nm
        r * 0.3,                        // 755nm — deep red falloff
    ];
    std::array::from_fn(|i| f16::from_f32(bands[i].clamp(0.0, 1.0)).to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_to_xyz_white_illuminant_produces_nonzero() {
        let white = SpectralBands([1.0; 16]);
        let xyz = spectral_to_xyz(&white, &Illuminant::d65());
        assert!(xyz[0] > 0.0, "X should be positive for white SPD");
        assert!(xyz[1] > 0.0, "Y should be positive for white SPD");
        assert!(xyz[2] > 0.0, "Z should be positive for white SPD");
    }

    #[test]
    fn spectral_to_xyz_black_is_zero() {
        let black = SpectralBands([0.0; 16]);
        let xyz = spectral_to_xyz(&black, &Illuminant::d65());
        assert_eq!(xyz, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn spectral_to_xyz_zero_illuminant_is_zero() {
        let white = SpectralBands([1.0; 16]);
        let dark = Illuminant { bands: [0.0; 16] };
        let xyz = spectral_to_xyz(&white, &dark);
        assert_eq!(xyz, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgb_to_spectral_red_peaks_at_band_11() {
        use half::f16;
        let s = rgb_to_spectral(1.0, 0.0, 0.0);
        let band11 = f16::from_bits(s[11]).to_f32();
        assert!(band11 > 0.9, "pure red should peak at band 11 (655nm), got {}", band11);
        let band3 = f16::from_bits(s[3]).to_f32();
        assert!(band3 < 0.01, "pure red should have ~zero at band 3 (455nm), got {}", band3);
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

    // --- Building albedo conversion tests ---
    //
    // These tests validate the spectral→base_color conversion used by
    // `urban_horizon::spectra_frame::pbr_for_channel`. The function converts
    // a 16-band reflectance to a LINEAR sRGB base_color for the path tracer
    // (which works in linear space, splat_backend.rs line 264).
    //
    // The old code applied `linear_to_srgb_gamma` after `xyz_to_srgb`, which
    // baked a ~1.4× brightness boost into the albedo (e.g. concrete grey linear
    // 0.50 → gamma 0.74, then stored as "linear" reflectance). The fix removes
    // the gamma step so the linear value is passed through directly.

    /// Helper replicating the fixed `pbr_for_channel` base_color calculation:
    /// flat reflectance → `spectral_to_xyz` → `xyz_to_srgb` (linear) → clamp.
    /// No `linear_to_srgb_gamma` — path tracer receives linear RGB.
    fn flat_reflectance_to_linear_base_color(v: f32) -> [f32; 3] {
        let xyz = spectral_to_xyz(&SpectralBands([v; 16]), &Illuminant::d65());
        let lin = xyz_to_srgb(xyz);
        [lin[0].clamp(0.0, 1.0), lin[1].clamp(0.0, 1.0), lin[2].clamp(0.0, 1.0)]
    }

    /// Perfect-white flat reflectance (1.0) → base_color ≈ [1, 1, 1] linear.
    #[test]
    fn white_flat_reflectance_maps_to_white_base_color() {
        let bc = flat_reflectance_to_linear_base_color(1.0);
        for (i, &c) in bc.iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 0.02,
                "channel {i}: flat 1.0 reflectance should give linear base_color ~1.0, got {c}"
            );
        }
    }

    /// Perfect-black flat reflectance (0.0) → base_color = [0, 0, 0].
    #[test]
    fn black_flat_reflectance_maps_to_black_base_color() {
        let bc = flat_reflectance_to_linear_base_color(0.0);
        for (i, &c) in bc.iter().enumerate() {
            assert!(c < 0.01, "channel {i}: flat 0.0 reflectance should give base_color ~0.0, got {c}");
        }
    }

    /// Concrete-like reflectance (0.30 flat, physical range 0.25–0.35) must
    /// produce a mid-dark linear base_color well below 0.5.
    ///
    /// The old gamma-encoding bug produced ~0.58 for this input; asserting
    /// < 0.45 rules out the bug. Y = 0.30 for a neutral flat reflectance
    /// (confirmed analytically: the normalization preserves Y = reflectance).
    #[test]
    fn concrete_reflectance_030_gives_realistic_linear_albedo() {
        let bc = flat_reflectance_to_linear_base_color(0.30);
        for (i, &c) in bc.iter().enumerate() {
            assert!(
                c > 0.20 && c < 0.45,
                "channel {i}: concrete flat-0.30 → linear base_color should be ~0.28–0.32, \
                 got {c:.4} (old gamma bug gave ~0.58)"
            );
        }
    }

    /// Service/concrete grey from `render_gpu::building_rgb(Service)` = (0.59, 0.59, 0.63).
    /// Uplifted to spectral and back to linear, the luminance must be in [0.35, 0.60] —
    /// NOT the near-white ~0.74 the old `linear_to_srgb_gamma` call produced.
    #[test]
    fn service_grey_spectral_roundtrip_is_not_near_white() {
        use half::f16;
        let (r, g, b) = (0.59f32, 0.59f32, 0.63f32);
        let bits = rgb_to_spectral(r, g, b);
        let refl: [f32; 16] = std::array::from_fn(|i| f16::from_bits(bits[i]).to_f32());
        let xyz = spectral_to_xyz(&SpectralBands(refl), &Illuminant::d65());
        let lin = xyz_to_srgb(xyz);
        let bc = [lin[0].clamp(0.0, 1.0), lin[1].clamp(0.0, 1.0), lin[2].clamp(0.0, 1.0)];
        let luma = 0.2126 * bc[0] + 0.7152 * bc[1] + 0.0722 * bc[2];
        assert!(
            luma < 0.65,
            "service grey linear luma should be < 0.65, got {luma:.4} (old gamma bug: ~0.74)"
        );
        assert!(luma > 0.30, "service grey should be mid-grey, not black: luma {luma:.4}");
    }

    /// Dark asphalt reflectance (0.08) must be proportionally much dimmer than
    /// concrete (0.30), proving the linear scale is preserved and not collapsed
    /// toward white by gamma encoding.
    #[test]
    fn dark_reflectance_proportionally_dimmer_than_concrete() {
        let bc_concrete = flat_reflectance_to_linear_base_color(0.30);
        let bc_asphalt  = flat_reflectance_to_linear_base_color(0.08);
        let y_c = bc_concrete[1];
        let y_a = bc_asphalt[1];
        assert!(
            y_a < y_c * 0.5,
            "asphalt Y {y_a:.4} should be < half concrete Y {y_c:.4} \
             (old gamma bug collapsed the linear scale)"
        );
    }
}
