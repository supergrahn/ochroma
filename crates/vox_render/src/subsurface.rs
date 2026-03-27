use vox_core::spectral::SpectralBands;

/// Subsurface scattering parameters for translucent materials.
#[derive(Debug, Clone)]
pub struct SubsurfaceProfile {
    pub name: String,
    /// Mean free path per spectral band (how far light travels inside).
    /// Longer MFP = more translucent at that wavelength.
    pub mean_free_path: SpectralBands,
    /// Absorption per band.
    pub absorption: SpectralBands,
    /// Overall translucency (0 = opaque, 1 = fully translucent).
    pub translucency: f32,
}

impl SubsurfaceProfile {
    /// Leaf/vegetation profile: red and near-IR pass through, green partially, blue absorbed.
    pub fn vegetation() -> Self {
        Self {
            name: "vegetation".to_string(),
            mean_free_path: SpectralBands([0.1, 0.15, 0.2, 0.5, 1.0, 0.8, 2.0, 3.0]),
            absorption: SpectralBands([0.8, 0.7, 0.5, 0.3, 0.1, 0.2, 0.05, 0.03]),
            translucency: 0.4,
        }
    }

    /// Skin profile: red penetrates deep, blue stays near surface.
    pub fn skin() -> Self {
        Self {
            name: "skin".to_string(),
            mean_free_path: SpectralBands([0.05, 0.08, 0.12, 0.3, 0.5, 0.8, 1.5, 2.0]),
            absorption: SpectralBands([0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.08]),
            translucency: 0.3,
        }
    }

    /// Wax/marble profile: uniform translucency.
    pub fn wax() -> Self {
        Self {
            name: "wax".to_string(),
            mean_free_path: SpectralBands([1.0; 8]),
            absorption: SpectralBands([0.1; 8]),
            translucency: 0.6,
        }
    }

    /// Compute the transmitted spectral radiance through a material of given thickness.
    pub fn transmit(&self, incoming: &SpectralBands, thickness: f32) -> SpectralBands {
        SpectralBands(std::array::from_fn(|i| {
            let mfp = self.mean_free_path.0[i];
            let abs = self.absorption.0[i];
            if mfp <= 0.0 { return 0.0; }
            // Beer-Lambert: transmitted = incoming * exp(-thickness * absorption / mfp)
            let attenuation = (-thickness * abs / mfp).exp();
            incoming.0[i] * attenuation * self.translucency
        }))
    }

    /// Compute colour shift: how much each band is affected relative to others.
    pub fn spectral_shift(&self, thickness: f32) -> SpectralBands {
        let white = SpectralBands([1.0; 8]);
        self.transmit(&white, thickness)
    }
}
