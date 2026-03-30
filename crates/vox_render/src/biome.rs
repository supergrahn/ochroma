//! Biome system — spectral material definitions for terrain regions.
//! BiomeDef specifies what spectral Gaussian Splats look like in a given biome.

use vox_core::types::GaussianSplat;
use half::f16;

/// Climate zone for biome classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClimateZone {
    Tropical,
    Temperate,
    Boreal,
    Arctic,
    Desert,
    Mediterranean,
}

/// Spectral splat profile for a biome — defines what splats look like in this biome.
#[derive(Debug, Clone)]
pub struct SpectralBiomeProfile {
    /// Base spectral emission per band for ground splats (f32, will be packed to f16).
    pub ground_spectral: [f32; 8],
    /// Base spectral emission for vegetation splats.
    pub vegetation_spectral: [f32; 8],
    /// Scale of ground splats (radius in metres).
    pub ground_splat_scale: f32,
    /// Spectral variance — random deviation per band to add visual noise.
    pub spectral_variance: f32,
}

impl SpectralBiomeProfile {
    /// Apply this profile's spectral values to a splat, with optional variance.
    pub fn apply_to_splat(&self, splat: &mut GaussianSplat, is_vegetation: bool, variation_seed: u32) {
        let base = if is_vegetation { &self.vegetation_spectral } else { &self.ground_spectral };
        // Deterministic per-splat variation using LCG hash
        let hash = variation_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (hash as f32 / u32::MAX as f32 - 0.5) * self.spectral_variance;
        for (b, &base_val) in base.iter().enumerate() {
            let val = (base_val + noise).clamp(0.0, 1.0);
            splat.spectral_mut()[b] = f16::from_f32(val).to_bits();
        }
    }
}

/// A single biome definition.
#[derive(Debug, Clone)]
pub struct BiomeDef {
    pub name: String,
    pub climate: ClimateZone,
    pub profile: SpectralBiomeProfile,
    /// Temperature range [min, max] in Celsius.
    pub temperature_range: [f32; 2],
    /// Rainfall range [min, max] in mm/year.
    pub rainfall_range: [f32; 2],
}

impl BiomeDef {
    /// Sample name for testing — a temperate forest biome.
    pub fn temperate_forest() -> Self {
        Self {
            name: "Temperate Forest".into(),
            climate: ClimateZone::Temperate,
            profile: SpectralBiomeProfile {
                // Ground: brown-green (high mid bands, low UV/blue)
                ground_spectral: [0.05, 0.08, 0.12, 0.25, 0.35, 0.30, 0.22, 0.18],
                // Vegetation: green (high 550nm region = bands 2-3)
                vegetation_spectral: [0.02, 0.04, 0.35, 0.55, 0.40, 0.15, 0.08, 0.05],
                ground_splat_scale: 0.5,
                spectral_variance: 0.05,
            },
            temperature_range: [5.0, 20.0],
            rainfall_range: [600.0, 1500.0],
        }
    }

    pub fn desert() -> Self {
        Self {
            name: "Desert".into(),
            climate: ClimateZone::Desert,
            profile: SpectralBiomeProfile {
                // Sand: warm ochre (high mid-to-low bands)
                ground_spectral: [0.08, 0.10, 0.18, 0.35, 0.55, 0.65, 0.60, 0.55],
                vegetation_spectral: [0.02, 0.04, 0.15, 0.25, 0.20, 0.12, 0.08, 0.05],
                ground_splat_scale: 0.8,
                spectral_variance: 0.08,
            },
            temperature_range: [20.0, 50.0],
            rainfall_range: [0.0, 250.0],
        }
    }

    pub fn arctic() -> Self {
        Self {
            name: "Arctic".into(),
            climate: ClimateZone::Arctic,
            profile: SpectralBiomeProfile {
                // Snow: high UV, flat across all bands (white/blue-white)
                ground_spectral: [0.85, 0.88, 0.90, 0.90, 0.89, 0.88, 0.87, 0.85],
                vegetation_spectral: [0.50, 0.55, 0.60, 0.65, 0.60, 0.55, 0.50, 0.45],
                ground_splat_scale: 0.6,
                spectral_variance: 0.03,
            },
            temperature_range: [-40.0, -5.0],
            rainfall_range: [100.0, 400.0],
        }
    }
}

/// Assigns a biome to a world position based on temperature and rainfall maps.
pub struct BiomeClassifier {
    pub biomes: Vec<BiomeDef>,
}

impl BiomeClassifier {
    pub fn new(biomes: Vec<BiomeDef>) -> Self { Self { biomes } }

    /// Classify a world position into a biome index.
    /// `temperature` in Celsius, `rainfall` in mm/year.
    pub fn classify(&self, temperature: f32, rainfall: f32) -> Option<usize> {
        // Find biome whose temperature+rainfall ranges best contain this position.
        // Score = sum of how centered the values are in the range.
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, biome) in self.biomes.iter().enumerate() {
            let temp_center = (biome.temperature_range[0] + biome.temperature_range[1]) * 0.5;
            let temp_span = (biome.temperature_range[1] - biome.temperature_range[0]).max(1.0);
            let rain_center = (biome.rainfall_range[0] + biome.rainfall_range[1]) * 0.5;
            let rain_span = (biome.rainfall_range[1] - biome.rainfall_range[0]).max(1.0);

            let temp_score = 1.0 - ((temperature - temp_center) / temp_span).abs();
            let rain_score = 1.0 - ((rainfall - rain_center) / rain_span).abs();
            let score = temp_score + rain_score;

            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        best_idx
    }

    /// Default biome classifier with 3 biomes.
    pub fn default_world() -> Self {
        Self::new(vec![
            BiomeDef::temperate_forest(),
            BiomeDef::desert(),
            BiomeDef::arctic(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desert_biome_has_warm_spectral() {
        let desert = BiomeDef::desert();
        assert!(desert.profile.ground_spectral[5] > 0.5);
    }

    #[test]
    fn arctic_biome_has_bright_spectral() {
        let arctic = BiomeDef::arctic();
        assert!(arctic.profile.ground_spectral[0] > 0.8);
    }

    #[test]
    fn biome_classifier_classifies_desert() {
        let classifier = BiomeClassifier::default_world();
        let idx = classifier.classify(35.0, 100.0).expect("should classify");
        assert_eq!(classifier.biomes[idx].climate, ClimateZone::Desert);
    }

    #[test]
    fn biome_classifier_classifies_arctic() {
        let classifier = BiomeClassifier::default_world();
        let idx = classifier.classify(-20.0, 200.0).expect("should classify");
        assert_eq!(classifier.biomes[idx].climate, ClimateZone::Arctic);
    }
}
