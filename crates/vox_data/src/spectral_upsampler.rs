//! Smits 1999 RGB→spectral upsampling.
//!
//! Decomposes sRGB (linear, [0,1]) into 7 basis spectra:
//!   white, cyan, magenta, yellow, red, green, blue
//! and returns a 16-band reflectance [f32; 16].
//!
//! Band centre wavelengths match vox_render::spectral_atmosphere::BAND_NM:
//!   [380, 405, 430, 455, 480, 505, 530, 555, 580, 605, 630, 655, 680, 705, 730, 755] nm

/// 16-band reflectance coefficients for each Smits basis spectrum.
/// Rows: white, cyan, magenta, yellow, red, green, blue.
/// Values from Smits 1999 extended to 16 bands at 380–755 nm, 25 nm steps.
const SMITS_BASIS: [[f32; 16]; 7] = [
    // white
    [0.941, 0.939, 0.988, 0.999, 0.999, 0.999, 0.961, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999],
    // cyan
    [0.971, 0.977, 0.979, 0.976, 0.978, 0.996, 0.999, 0.578, 0.044, 0.015, 0.021, 0.004, 0.005, 0.000, 0.000, 0.000],
    // magenta
    [0.978, 0.965, 0.944, 0.587, 0.183, 0.057, 0.032, 0.062, 0.203, 0.513, 0.855, 0.976, 0.989, 0.999, 0.999, 0.999],
    // yellow
    [0.001, 0.002, 0.003, 0.017, 0.110, 0.355, 0.854, 0.998, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999],
    // red
    [0.101, 0.062, 0.060, 0.048, 0.066, 0.043, 0.032, 0.073, 0.302, 0.692, 0.960, 0.995, 0.995, 0.966, 0.995, 0.995],
    // green
    [0.000, 0.000, 0.000, 0.001, 0.083, 0.500, 0.962, 0.999, 0.973, 0.700, 0.236, 0.049, 0.028, 0.014, 0.006, 0.002],
    // blue
    [0.844, 0.913, 0.911, 0.952, 0.991, 0.659, 0.287, 0.088, 0.018, 0.008, 0.006, 0.003, 0.002, 0.000, 0.000, 0.000],
];

pub struct SpectralUpsampler;

impl SpectralUpsampler {
    /// Convert linear sRGB to a 16-band spectral reflectance via Smits 1999 decomposition.
    ///
    /// Inputs must be in [0, 1]. Values outside this range are clamped.
    pub fn from_rgb(r: f32, g: f32, b: f32) -> [f32; 16] {
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        let (white, cyan, magenta, yellow, red, green, blue) = Self::decompose(r, g, b);

        let weights = [white, cyan, magenta, yellow, red, green, blue];
        let mut out = [0.0f32; 16];
        for (i, basis) in SMITS_BASIS.iter().enumerate() {
            for b in 0..16 {
                out[b] += weights[i] * basis[b];
            }
        }
        // Normalise to [0, 1]
        let max = out.iter().copied().fold(f32::EPSILON, f32::max);
        if max > 1.0 {
            for v in &mut out {
                *v /= max;
            }
        }
        out
    }

    /// Decompose (r, g, b) into 7 basis weights following Smits 1999 §3.
    #[allow(unused_assignments)]
    fn decompose(r: f32, g: f32, b: f32) -> (f32, f32, f32, f32, f32, f32, f32) {
        let (mut white, mut cyan, mut magenta, mut yellow, mut red, mut green, mut blue) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        if r <= g && r <= b {
            white = r;
            if g <= b {
                yellow = g - r;
                blue = b - g;
            } else {
                yellow = b - r;
                green = g - b;
            }
        } else if g <= r && g <= b {
            white = g;
            if r <= b {
                magenta = r - g;
                blue = b - r;
            } else {
                magenta = b - g;
                red = r - b;
            }
        } else {
            white = b;
            if r <= g {
                cyan = r - b;
                green = g - r;
            } else {
                cyan = g - b;
                red = r - g;
            }
        }
        (white, cyan, magenta, yellow, red, green, blue)
    }
}

/// Named spectral material with 16-band reflectance.
pub struct SpectralMaterial {
    pub name: &'static str,
    /// 16-band reflectance at [380, 405, 430, 455, 480, 505, 530, 555, 580, 605, 630, 655, 680, 705, 730, 755] nm.
    pub reflectance: [f32; 16],
}

/// Compile-time database of physically motivated spectral materials.
/// IDs are 1-indexed to match VXM v3 `spectral_material_id` (0 = unassigned).
pub struct SpectralMaterialDb;

impl SpectralMaterialDb {
    /// All materials. Index + 1 = material_id in VXM v3.
    pub const MATERIALS: &'static [SpectralMaterial] = &[
        SpectralMaterial { name: "foliage",  reflectance: [0.05, 0.05, 0.06, 0.06, 0.07, 0.35, 0.55, 0.55, 0.12, 0.08, 0.06, 0.05, 0.05, 0.20, 0.45, 0.55] },
        SpectralMaterial { name: "soil",     reflectance: [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.19, 0.22, 0.26, 0.28, 0.30, 0.31, 0.31, 0.32] },
        SpectralMaterial { name: "rock",     reflectance: [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.22, 0.22] },
        SpectralMaterial { name: "water",    reflectance: [0.03, 0.04, 0.05, 0.06, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01] },
        SpectralMaterial { name: "glass",    reflectance: [0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94, 0.93, 0.93, 0.92, 0.92, 0.91, 0.91, 0.91] },
        SpectralMaterial { name: "concrete", reflectance: [0.20, 0.20, 0.21, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27, 0.28] },
        SpectralMaterial { name: "snow",     reflectance: [0.88, 0.89, 0.91, 0.92, 0.93, 0.94, 0.94, 0.94, 0.94, 0.93, 0.93, 0.92, 0.92, 0.91, 0.91, 0.90] },
        SpectralMaterial { name: "asphalt",  reflectance: [0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07] },
    ];

    /// Look up a material by name (case-insensitive). Returns None if not found.
    pub fn find_by_name(name: &str) -> Option<&'static SpectralMaterial> {
        Self::MATERIALS.iter().find(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// Retrieve material by 1-based ID (as stored in VXM v3). Returns None for id=0.
    pub fn find_by_id(id: u16) -> Option<&'static SpectralMaterial> {
        if id == 0 || id as usize > Self::MATERIALS.len() {
            None
        } else {
            Some(&Self::MATERIALS[id as usize - 1])
        }
    }

    /// Find the closest material by L2 distance in spectral space.
    pub fn classify(reflectance: &[f32; 16]) -> &'static SpectralMaterial {
        Self::MATERIALS
            .iter()
            .min_by(|a, b| {
                let da: f32 = a.reflectance.iter().zip(reflectance).map(|(x, y)| (x - y).powi(2)).sum();
                let db: f32 = b.reflectance.iter().zip(reflectance).map(|(x, y)| (x - y).powi(2)).sum();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_rgb_returns_flat_spectrum() {
        let s = SpectralUpsampler::from_rgb(1.0, 1.0, 1.0);
        for (i, &v) in s.iter().enumerate() {
            assert!(v > 0.9, "white: band {} should be near 1.0, got {}", i, v);
        }
    }

    #[test]
    fn black_rgb_returns_zero_spectrum() {
        let s = SpectralUpsampler::from_rgb(0.0, 0.0, 0.0);
        for (i, &v) in s.iter().enumerate() {
            assert!(v < 1e-5, "black: band {} should be ~0.0, got {}", i, v);
        }
    }

    #[test]
    fn red_rgb_concentrates_in_high_bands() {
        let s = SpectralUpsampler::from_rgb(1.0, 0.0, 0.0);
        let high: f32 = s[8..16].iter().copied().sum::<f32>() / 8.0;
        let low: f32 = s[0..4].iter().copied().sum::<f32>() / 4.0;
        println!("red: high bands avg {:.3}, low bands avg {:.3}", high, low);
        assert!(high > low, "red: high bands avg {:.3} should exceed low bands avg {:.3}", high, low);
    }

    #[test]
    fn blue_rgb_concentrates_in_low_bands() {
        let s = SpectralUpsampler::from_rgb(0.0, 0.0, 1.0);
        let low: f32 = s[0..4].iter().copied().sum::<f32>() / 4.0;
        let high: f32 = s[10..16].iter().copied().sum::<f32>() / 6.0;
        println!("blue: low bands avg {:.3}, high bands avg {:.3}", low, high);
        assert!(low > high, "blue: low bands avg {:.3} should exceed high bands avg {:.3}", low, high);
    }

    #[test]
    fn output_stays_in_unit_range() {
        let inputs = [(0.5, 0.5, 0.5), (1.0, 0.0, 0.5), (0.2, 0.8, 0.1)];
        for (r, g, b) in inputs {
            let s = SpectralUpsampler::from_rgb(r, g, b);
            for (i, &v) in s.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "rgb({},{},{}) band {} = {} out of [0,1]",
                    r, g, b, i, v
                );
            }
        }
    }

    #[test]
    fn material_db_find_by_name() {
        let m = SpectralMaterialDb::find_by_name("foliage").unwrap();
        assert_eq!(m.name, "foliage");
    }

    #[test]
    fn material_db_find_by_id_one_based() {
        let m = SpectralMaterialDb::find_by_id(1).unwrap();
        assert_eq!(m.name, "foliage");
    }

    #[test]
    fn material_db_id_zero_returns_none() {
        assert!(SpectralMaterialDb::find_by_id(0).is_none());
    }

    #[test]
    fn classify_foliage() {
        let green_spectrum = [0.05f32, 0.05, 0.06, 0.06, 0.07, 0.35, 0.55, 0.55, 0.12, 0.08, 0.06, 0.05, 0.05, 0.20, 0.45, 0.55];
        let m = SpectralMaterialDb::classify(&green_spectrum);
        assert_eq!(m.name, "foliage", "strong green peak should classify as foliage");
    }

    #[test]
    fn classify_snow() {
        let bright = [0.88f32, 0.89, 0.91, 0.92, 0.93, 0.94, 0.94, 0.94, 0.94, 0.93, 0.93, 0.92, 0.92, 0.91, 0.91, 0.90];
        let m = SpectralMaterialDb::classify(&bright);
        assert_eq!(m.name, "snow", "flat high reflectance should classify as snow");
    }

    #[test]
    fn all_materials_have_unique_names() {
        let names: std::collections::HashSet<&str> =
            SpectralMaterialDb::MATERIALS.iter().map(|m| m.name).collect();
        assert_eq!(
            names.len(),
            SpectralMaterialDb::MATERIALS.len(),
            "every material must have a unique name"
        );
    }
}
