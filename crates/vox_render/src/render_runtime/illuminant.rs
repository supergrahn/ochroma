//! Asset-editor preview illuminant + its CIE RGB tint table.
//!
//! Moved verbatim from the game's `spectra_frame::witnesses` so a new game never
//! re-implements the spectral preview tint. No game-type dependencies — uses only
//! the engine's `vox_core::spectral::Illuminant`.

/// Which illuminant the in-game asset-editor preview lights the draft under (R14 spectral).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewIlluminant {
    /// CIE D65 noon daylight (neutral).
    Daylight,
    /// CIE D50 horizon daylight (slightly warm).
    Horizon,
    /// CIE A incandescent (~2700K, warm).
    Warm,
    /// CIE F11 fluorescent.
    Fluorescent,
}

impl PreviewIlluminant {
    fn to_illuminant(self) -> vox_core::spectral::Illuminant {
        use vox_core::spectral::Illuminant;
        match self {
            PreviewIlluminant::Daylight => Illuminant::d65(),
            PreviewIlluminant::Horizon => Illuminant::d50(),
            PreviewIlluminant::Warm => Illuminant::a(),
            PreviewIlluminant::Fluorescent => Illuminant::f11(),
        }
    }

    /// The illuminant's relative RGB tint, normalized so D65 reads neutral.
    pub fn rgb_tint(self) -> [f32; 3] {
        use vox_core::spectral::Illuminant;
        const CIE_X: [f32; 16] = [
            0.01741, 0.08028, 0.26000, 0.21000, 0.00949, 0.00000, 0.11201, 0.38000, 0.74300,
            1.02200, 0.71600, 0.38100, 0.19700, 0.09020, 0.03400, 0.01180,
        ];
        const CIE_Y: [f32; 16] = [
            0.00039, 0.00232, 0.01998, 0.09520, 0.17399, 0.46600, 0.69500, 0.94500, 0.86800,
            0.65100, 0.38100, 0.18000, 0.08000, 0.03300, 0.01200, 0.00400,
        ];
        const CIE_Z: [f32; 16] = [
            0.08290, 0.38637, 1.29900, 1.24500, 0.45640, 0.05250, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
        ];
        fn integrate(bands: &[f32; 16]) -> [f32; 3] {
            let mut xyz = [0.0f32; 3];
            for i in 0..16 {
                xyz[0] += bands[i] * CIE_X[i];
                xyz[1] += bands[i] * CIE_Y[i];
                xyz[2] += bands[i] * CIE_Z[i];
            }
            xyz
        }
        fn to_srgb(xyz: [f32; 3]) -> [f32; 3] {
            let [x, y, z] = xyz;
            [
                (3.2406 * x - 1.5372 * y - 0.4986 * z).max(0.0),
                (-0.9689 * x + 1.8758 * y + 0.0415 * z).max(0.0),
                (0.0557 * x - 0.2040 * y + 1.0570 * z).max(0.0),
            ]
        }
        let this = to_srgb(integrate(&self.to_illuminant().bands));
        let ref_d65 = to_srgb(integrate(&Illuminant::d65().bands));
        let ratio = [
            this[0] / ref_d65[0].max(1e-4),
            this[1] / ref_d65[1].max(1e-4),
            this[2] / ref_d65[2].max(1e-4),
        ];
        let g = ratio[1].max(1e-4);
        [
            (ratio[0] / g).clamp(0.2, 2.0),
            1.0,
            (ratio[2] / g).clamp(0.2, 2.0),
        ]
    }
}
