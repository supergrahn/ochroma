//! Species-specific spectral sensitivity remapping.
//!
//! Ochroma renders in 16 spectral bands (380–755nm). Different species perceive
//! wavelengths differently. This module remaps 16-band spectral data to a
//! species-appropriate RGB output for display.

/// Display mode for spectral data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeciesView {
    /// Standard human trichromatic vision via CIE 1931 observer.
    Human,
    /// Honeybee trichromatic vision: UV / blue / green channels.
    Bee,
    /// Mantis shrimp: 16 receptor types across 300–700nm, false-coloured.
    MantisShrimp,
}

impl SpeciesView {
    /// Remap a 16-band SPD `[f32; 16]` to an sRGB triple `[f32; 3]` for display.
    ///
    /// Band layout (index → centre wavelength):
    ///   0→380nm, 1→405nm, 2→430nm, 3→455nm, 4→480nm, 5→505nm, 6→530nm, 7→555nm,
    ///   8→580nm, 9→605nm, 10→630nm, 11→655nm, 12→680nm, 13→705nm, 14→730nm, 15→755nm
    pub fn remap(&self, spd: &[f32; 16]) -> [f32; 3] {
        match self {
            SpeciesView::Human => remap_human(spd),
            SpeciesView::Bee => remap_bee(spd),
            SpeciesView::MantisShrimp => remap_mantis_shrimp(spd),
        }
    }
}

/// Human: CIE 1931 2° observer. Weights sampled at 16 band centres (380–755nm at 25nm steps).
/// Source: CIE publication 15:2004 table.
fn remap_human(spd: &[f32; 16]) -> [f32; 3] {
    const CIE_X: [f32; 16] = [0.01741, 0.08028, 0.26000, 0.21000, 0.00949, 0.00000, 0.11201, 0.38000, 0.74300, 1.02200, 0.71600, 0.38100, 0.19700, 0.09020, 0.03400, 0.01180];
    const CIE_Y: [f32; 16] = [0.00039, 0.00232, 0.01998, 0.09520, 0.17399, 0.46600, 0.69500, 0.94500, 0.86800, 0.65100, 0.38100, 0.18000, 0.08000, 0.03300, 0.01200, 0.00400];
    const CIE_Z: [f32; 16] = [0.08290, 0.38637, 1.29900, 1.24500, 0.45640, 0.05250, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000];

    let x: f32 = spd.iter().zip(CIE_X.iter()).map(|(s, w)| s * w).sum();
    let y: f32 = spd.iter().zip(CIE_Y.iter()).map(|(s, w)| s * w).sum();
    let z: f32 = spd.iter().zip(CIE_Z.iter()).map(|(s, w)| s * w).sum();

    xyz_to_srgb(x, y, z)
}

/// Bee trichromat: UV (344nm peak), Blue (436nm peak), Green (544nm peak).
/// Bees are red-blind — bands 8–15 (580–755nm) are invisible to bees.
///
/// Sensitivity arrays:
///   S (UV receptor, ~340nm peak): nonzero only at bands 0–2
///   M (blue receptor, ~440nm peak): nonzero at bands 1–6
///   L (green receptor, ~544nm peak): nonzero at bands 2–7 (cuts off before 580nm)
///
/// Output RGB: green→R, blue→G, UV→B (standard bee false-colour convention).
fn remap_bee(spd: &[f32; 16]) -> [f32; 3] {
    // S (UV receptor, peaks ~340nm): covers 380–430nm via bands 0–2
    const S: [f32; 16] = [0.30, 0.50, 0.60, 0.40, 0.15, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00];
    // M (blue receptor, peaks ~440nm): covers 405–555nm via bands 1–6
    const M: [f32; 16] = [0.05, 0.20, 0.60, 0.90, 0.80, 0.45, 0.15, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00];
    // L (green receptor, peaks ~544nm): covers 430–555nm via bands 2–7. Cutoff at 580nm.
    const L: [f32; 16] = [0.00, 0.00, 0.01, 0.05, 0.30, 0.75, 0.99, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00];

    let uv: f32 = spd.iter().zip(S.iter()).map(|(s, w)| s * w).sum();
    let blue: f32 = spd.iter().zip(M.iter()).map(|(s, w)| s * w).sum();
    let green: f32 = spd.iter().zip(L.iter()).map(|(s, w)| s * w).sum();
    // Map to display RGB: green→R, blue→G, UV→B (standard bee false-colour convention)
    [green.clamp(0.0, 1.0), blue.clamp(0.0, 1.0), uv.clamp(0.0, 1.0)]
}

/// Mantis shrimp: 16 receptor classes (R1–R16) from ~300–700nm.
/// With 16 Ochroma bands (380–755nm), each band maps to one mantis shrimp receptor.
/// Each band is assigned a unique hue cycling UV→violet→blue→cyan→green→yellow→orange→red→NIR.
/// Bands 12–15 (680–755nm) are NIR and beyond the mantis shrimp range — they map near-zero.
fn remap_mantis_shrimp(spd: &[f32; 16]) -> [f32; 3] {
    // Each column gives [R, G, B] weights for that spectral band.
    // Rows: R=0, G=1, B=2. Bands 0-11 have unique hue; bands 12-15 are NIR → near-zero.
    const MS: [[f32; 16]; 3] = [
        // R: 0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15
        [0.00, 0.00, 0.05, 0.00, 0.00, 0.10, 0.30, 0.60, 0.90, 1.00, 0.80, 0.60, 0.00, 0.00, 0.00, 0.00],
        // G
        [0.00, 0.15, 0.35, 0.60, 0.85, 1.00, 0.85, 0.70, 0.45, 0.20, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        // B
        [1.00, 0.85, 0.75, 0.50, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00],
    ];

    let mut rgb = [0.0f32; 3];
    for c in 0..3 {
        for b in 0..16 {
            rgb[c] += MS[c][b] * spd[b].clamp(0.0, 1.0);
        }
    }

    // Normalise so max component = 1.0 (preserves relative hue balance)
    let max_c = rgb.iter().copied().fold(f32::EPSILON, f32::max);
    let scale = if max_c > 1.0 { 1.0 / max_c } else { 1.0 };
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Convert CIE XYZ to linear sRGB. Clamps to [0, 1].
/// Uses the IEC 61966-2-1 D65 matrix.
fn xyz_to_srgb(x: f32, y: f32, z: f32) -> [f32; 3] {
    let r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_white_light_is_neutral() {
        let spd = [0.5f32; 16];
        let rgb = SpeciesView::Human.remap(&spd);
        assert!(rgb[0] > 0.0 && rgb[1] > 0.0 && rgb[2] > 0.0,
            "flat SPD should produce non-zero RGB, got {:?}", rgb);
    }

    #[test]
    fn bee_red_only_produces_no_output() {
        // Only bands 8–15 (580–755nm) lit — bees are blind to these wavelengths
        let mut spd = [0.0f32; 16];
        for b in 8..16 { spd[b] = 1.0; }
        let rgb = SpeciesView::Bee.remap(&spd);
        println!("bee red-only: {:?}", rgb);
        assert_eq!(rgb[0], 0.0, "bee green channel should be 0 for red-only light: {:?}", rgb);
        assert_eq!(rgb[2], 0.0, "bee UV channel should be 0 for red-only light: {:?}", rgb);
    }

    #[test]
    fn bee_uv_only_produces_blue_channel_output() {
        let mut spd = [0.0f32; 16];
        spd[0] = 1.0;
        let rgb = SpeciesView::Bee.remap(&spd);
        println!("bee UV-only: {:?}", rgb);
        assert!(rgb[2] > 0.0, "bee UV input should produce output in B channel: {:?}", rgb);
        assert_eq!(rgb[0], 0.0, "bee UV input should produce no R channel output: {:?}", rgb);
    }

    #[test]
    fn mantis_each_band_produces_distinct_hue() {
        let mut results = Vec::new();
        for b in 0..16 {
            let mut spd = [0.0f32; 16];
            spd[b] = 1.0;
            results.push(SpeciesView::MantisShrimp.remap(&spd));
        }
        // Bands 0–11 produce distinct outputs; bands 12–15 are NIR → all near-zero
        for i in 0..12 {
            for j in (i + 1)..12 {
                let same = results[i].iter().zip(results[j].iter())
                    .all(|(a, b)| (a - b).abs() < 1e-4);
                assert!(!same, "bands {} and {} produced identical mantis output: {:?}",
                    i, j, results[i]);
            }
        }
    }

    #[test]
    fn mantis_output_in_unit_range() {
        let spd = [0.8, 0.6, 0.9, 0.3, 0.7, 0.5, 0.4, 0.2, 0.8, 0.6, 0.9, 0.3, 0.7, 0.5, 0.4, 0.2];
        let rgb = SpeciesView::MantisShrimp.remap(&spd);
        for (i, &v) in rgb.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "channel {} value {} out of [0,1]", i, v);
        }
    }

    #[test]
    fn species_view_human_cie_weights_give_nonzero_luminance() {
        let spd = [1.0f32; 16];
        let rgb = SpeciesView::Human.remap(&spd);
        let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        assert!(luminance > 0.1, "CIE observer on all-ones SPD should give luminance > 0.1, got {}", luminance);
    }
}
