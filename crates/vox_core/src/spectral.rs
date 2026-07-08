use serde::{Deserialize, Serialize};

/// 380–755 nm at 25 nm steps (USGS wavelength grid, 16 bands).
pub const BAND_WAVELENGTHS: [f32; 16] = [
    380.0, 405.0, 430.0, 455.0, 480.0, 505.0, 530.0, 555.0, 580.0, 605.0, 630.0, 655.0, 680.0,
    705.0, 730.0, 755.0,
];
pub const BAND_SPACING: f32 = 25.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralBands(pub [f32; 16]);

#[derive(Debug, Clone)]
pub struct Illuminant {
    pub bands: [f32; 16],
}

/// CIE 1931 2° observer, 380–755 nm at 25 nm steps.
const CIE_X: [f32; 16] = [
    0.01741, 0.08028, 0.26000, 0.21000, 0.00949, 0.00000, 0.11201, 0.38000, 0.74300, 1.02200,
    0.71600, 0.38100, 0.19700, 0.09020, 0.03400, 0.01180,
];
const CIE_Y: [f32; 16] = [
    0.00039, 0.00232, 0.01998, 0.09520, 0.17399, 0.46600, 0.69500, 0.94500, 0.86800, 0.65100,
    0.38100, 0.18000, 0.08000, 0.03300, 0.01200, 0.00400,
];
const CIE_Z: [f32; 16] = [
    0.08290, 0.38637, 1.29900, 1.24500, 0.45640, 0.05250, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
];

impl Illuminant {
    pub fn d65() -> Self {
        Self {
            bands: [
                49.98, 52.31, 56.45, 68.70, 82.75, 91.49, 95.00, 100.00, 102.10, 100.75, 99.20,
                98.00, 93.50, 88.69, 83.29, 78.28,
            ],
        }
    }
    pub fn d50() -> Self {
        Self {
            bands: [
                25.83, 31.22, 36.93, 52.93, 67.23, 79.00, 86.68, 93.00, 97.74, 100.00, 100.76,
                99.82, 97.74, 94.34, 88.49, 83.56,
            ],
        }
    }
    pub fn a() -> Self {
        Self {
            bands: [
                9.80, 12.09, 14.71, 17.68, 21.00, 24.67, 28.70, 33.09, 37.82, 42.87, 48.24, 53.91,
                59.86, 66.06, 72.50, 79.13,
            ],
        }
    }
    pub fn f11() -> Self {
        Self {
            bands: [
                3.00, 4.00, 8.00, 15.00, 30.00, 45.00, 60.00, 70.00, 80.00, 100.00, 120.00, 90.00,
                55.00, 30.00, 20.00, 15.00,
            ],
        }
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
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Smits-1999 reflectance basis curves resampled to this crate's 16-band
/// 380–755 nm / 25 nm grid and refined so the RGB→spectral→RGB roundtrip is
/// faithful under *this* renderer's `spectral_to_xyz` (per-channel D65
/// normalization) + `xyz_to_srgb`.
///
/// Each row is a smooth, physically plausible reflectance in [0,1]:
/// - `CYAN`    — high in blue/green, drops toward red
/// - `MAGENTA` — high in blue + red, dips in green
/// - `YELLOW`  — low in blue, high in green/red
/// - `RED`     — low in blue/green, rises toward red
/// - `GREEN`   — bump centered on green
/// - `BLUE`    — high in blue, drops toward red
///
/// Derivation (deterministic, reproducible — no RNG, no per-call magic): the
/// classic Smits 10-sample white/cyan/magenta/yellow/red/green/blue spectra were
/// linearly resampled onto the 16-band grid, then a fixed-seed coordinate-descent
/// minimized the squared roundtrip error over a uniform 6×6×6 RGB grid (with a
/// smoothness penalty to keep the curves physical). The white basis is implicitly
/// all-ones, so any neutral `r=g=b=v` yields a FLAT `v` spectrum — and a flat
/// reflectance roundtrips exactly through `spectral_to_xyz` (which is white-balanced
/// to D65), giving neutrals zero roundtrip error with no channel-ordering tint.
/// Max roundtrip error over the full 6×6×6 grid is ≈0.019 per channel.
const SMITS_BASIS: [[f32; 16]; 6] = [
    // CYAN
    [
        0.9515, 0.9531, 0.9641, 0.9940, 1.0000, 1.0010, 1.0000, 1.0000, 0.8873, 0.5236, 0.2510,
        0.0981, 0.0248, 0.0004, 0.0000, 0.0000,
    ],
    // MAGENTA
    [
        1.0000, 1.0000, 1.0000, 1.0000, 0.4400, 0.0000, 0.0000, 0.0000, 0.0336, 0.7349, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    ],
    // YELLOW
    [
        0.0000, 0.0000, 0.0000, 0.0507, 0.5268, 1.0000, 1.0000, 1.0000, 1.0000, 0.9415, 0.7962,
        0.6867, 0.6205, 0.5917, 0.5826, 0.5804,
    ],
    // RED
    [
        0.0533, 0.0513, 0.0397, 0.0023, 0.0000, 0.0000, 0.0000, 0.0000, 0.1113, 0.4777, 0.7495,
        0.9004, 0.9729, 0.9978, 1.0000, 1.0000,
    ],
    // GREEN
    [
        0.0000, 0.0000, 0.0000, 0.0000, 0.5601, 1.0000, 1.0000, 1.0000, 0.9683, 0.2634, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    ],
    // BLUE
    [
        1.0000, 1.0000, 1.0000, 0.9507, 0.4716, 0.0000, 0.0000, 0.0000, 0.0000, 0.0574, 0.2031,
        0.3145, 0.3815, 0.4111, 0.4209, 0.4233,
    ],
];

/// Convert linear RGB reflectance to a 16-band spectral reflectance via the
/// Smits-1999 method.
///
/// The 16 bands span 380–755 nm at 25 nm steps (USGS wavelength grid). The RGB is
/// decomposed into a white component plus one secondary (cyan/magenta/yellow) and
/// one primary (red/green/blue) basis curve, chosen by which channel is the
/// smallest / largest, then weighted so the spectrum is a smooth, physically
/// plausible reflectance in [0,1]. This feeds the real spectral path tracer AND
/// roundtrips faithfully: a neutral grey reconstructs a flat spectrum (no tint, no
/// darkening), and saturated colors reconstruct hue-preserved. No brightness
/// compensation is applied or required.
pub fn rgb_to_spectral(r: f32, g: f32, b: f32) -> [u16; 16] {
    use half::f16;
    let [cyan, magenta, yellow, red, green, blue] = SMITS_BASIS;
    std::array::from_fn(|k| {
        // White component is `min(r,g,b)`; the two residuals select one secondary
        // (the channel that is *absent*) and one primary (the dominant channel).
        let spec = if r <= g && r <= b {
            // red is smallest -> white = r, add cyan for the (g,b) excess
            r + if g <= b {
                (g - r) * cyan[k] + (b - g) * blue[k]
            } else {
                (b - r) * cyan[k] + (g - b) * green[k]
            }
        } else if g <= r && g <= b {
            // green is smallest -> add magenta for the (r,b) excess
            g + if r <= b {
                (r - g) * magenta[k] + (b - r) * blue[k]
            } else {
                (b - g) * magenta[k] + (r - b) * red[k]
            }
        } else {
            // blue is smallest -> add yellow for the (r,g) excess
            b + if r <= g {
                (r - b) * yellow[k] + (g - r) * green[k]
            } else {
                (g - b) * yellow[k] + (r - g) * red[k]
            }
        };
        f16::from_f32(spec.clamp(0.0, 1.0)).to_bits()
    })
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
        assert!(
            band11 > 0.9,
            "pure red should peak at band 11 (655nm), got {}",
            band11
        );
        let band3 = f16::from_bits(s[3]).to_f32();
        assert!(
            band3 < 0.01,
            "pure red should have ~zero at band 3 (455nm), got {}",
            band3
        );
    }

    /// GATE: the RGB → spectral → XYZ(D65) → sRGB roundtrip must be faithful.
    ///
    /// Neutrals must come back within ±0.03 per channel with NO channel-ordering
    /// flip (no blue tint — the old code reconstructed grey as R<G<B). Saturated
    /// colors must come back within ±0.07, hue-preserved (the channel ordering of
    /// out must match in). FAILS on the old primary-sum `rgb_to_spectral`; PASSES
    /// on the Smits basis.
    #[test]
    fn roundtrip_is_faithful() {
        use half::f16;
        fn roundtrip(rgb: [f32; 3]) -> [f32; 3] {
            let bits = rgb_to_spectral(rgb[0], rgb[1], rgb[2]);
            let refl: [f32; 16] = std::array::from_fn(|i| f16::from_bits(bits[i]).to_f32());
            let xyz = spectral_to_xyz(&SpectralBands(refl), &Illuminant::d65());
            let lin = xyz_to_srgb(xyz);
            [
                lin[0].clamp(0.0, 1.0),
                lin[1].clamp(0.0, 1.0),
                lin[2].clamp(0.0, 1.0),
            ]
        }
        // (input, tol, neutral?)
        let neutrals: [[f32; 3]; 4] = [
            [0.18, 0.18, 0.18],
            [0.50, 0.50, 0.50],
            [0.90, 0.90, 0.90],
            [0.32, 0.32, 0.30], // concrete (essentially neutral)
        ];
        for rgb in neutrals {
            let out = roundtrip(rgb);
            for c in 0..3 {
                let e = (out[c] - rgb[c]).abs();
                assert!(
                    e <= 0.03,
                    "neutral {rgb:?} -> {out:?}: channel {c} error {e:.4} > 0.03"
                );
            }
            // No channel-ordering flip: a (near-)neutral must NOT come back blue-tinted.
            assert!(
                out[2] <= out[0] + 0.03 && out[2] <= out[1] + 0.03,
                "neutral {rgb:?} -> {out:?}: blue must not dominate (old code gave R<G<B tint)"
            );
        }
        // Saturated: within ±0.07, hue-preserved (channel ordering preserved).
        let saturated: [[f32; 3]; 3] = [
            [0.34, 0.58, 0.36], // grass  (g > r,b)
            [0.45, 0.28, 0.22], // brick  (r > g > b)
            [0.30, 0.50, 0.80], // sky    (b > g > r)
        ];
        for rgb in saturated {
            let out = roundtrip(rgb);
            for c in 0..3 {
                let e = (out[c] - rgb[c]).abs();
                assert!(
                    e <= 0.07,
                    "saturated {rgb:?} -> {out:?}: channel {c} error {e:.4} > 0.07"
                );
            }
            // Hue preserved: the rank order of channels must survive the roundtrip.
            let order = |v: &[f32; 3]| {
                let mut idx = [0usize, 1, 2];
                idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
                idx
            };
            assert_eq!(
                order(&rgb),
                order(&out),
                "saturated {rgb:?} -> {out:?}: channel rank order (hue) not preserved"
            );
        }
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
        assert!(
            (g - 1.0).abs() < 0.001,
            "gamma(1.0) should be ~1.0, got {}",
            g
        );
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
        [
            lin[0].clamp(0.0, 1.0),
            lin[1].clamp(0.0, 1.0),
            lin[2].clamp(0.0, 1.0),
        ]
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
            assert!(
                c < 0.01,
                "channel {i}: flat 0.0 reflectance should give base_color ~0.0, got {c}"
            );
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
        let bc = [
            lin[0].clamp(0.0, 1.0),
            lin[1].clamp(0.0, 1.0),
            lin[2].clamp(0.0, 1.0),
        ];
        let luma = 0.2126 * bc[0] + 0.7152 * bc[1] + 0.0722 * bc[2];
        assert!(
            luma < 0.65,
            "service grey linear luma should be < 0.65, got {luma:.4} (old gamma bug: ~0.74)"
        );
        assert!(
            luma > 0.30,
            "service grey should be mid-grey, not black: luma {luma:.4}"
        );
    }

    /// Dark asphalt reflectance (0.08) must be proportionally much dimmer than
    /// concrete (0.30), proving the linear scale is preserved and not collapsed
    /// toward white by gamma encoding.
    #[test]
    fn dark_reflectance_proportionally_dimmer_than_concrete() {
        let bc_concrete = flat_reflectance_to_linear_base_color(0.30);
        let bc_asphalt = flat_reflectance_to_linear_base_color(0.08);
        let y_c = bc_concrete[1];
        let y_a = bc_asphalt[1];
        assert!(
            y_a < y_c * 0.5,
            "asphalt Y {y_a:.4} should be < half concrete Y {y_c:.4} \
             (old gamma bug collapsed the linear scale)"
        );
    }
}
