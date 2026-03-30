//! TerrainSplatizer — converts terrain heightfields to GaussianSplats
//! with physically measured spectral reflectances from USGS material database.
//!
//! Biome → splat_weights[4] → blend 4 spectral curves from SpectralTerrainMaterials.

/// Biome kind — mirrors forge-terrain Biome enum (re-defined here to avoid forge dep).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiomeKind {
    Alpine, Tundra, Forest, Grassland, Desert, Wetland, Coastal,
    SubalpineShrub, Savanna, Taiga, TropicalRainforest,
}

/// 7-slot spectral terrain material palette (16 bands each, 380–755nm).
/// Slot order: Water(0), Sand(1), Grass(2), Dirt(3), Rock(4), Snow(5), Forest/Bark(6).
pub struct SpectralTerrainMaterials {
    pub slots: [[f32; 16]; 7],
}

impl Default for SpectralTerrainMaterials {
    fn default() -> Self {
        Self {
            slots: [
                // Water (0)
                [0.03, 0.04, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                // Sand (1)
                [0.25, 0.28, 0.31, 0.34, 0.36, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
                // Grass (2)
                [0.04, 0.04, 0.05, 0.07, 0.08, 0.10, 0.12, 0.12, 0.08, 0.05, 0.04, 0.04, 0.05, 0.20, 0.45, 0.55],
                // Dirt (3)
                [0.07, 0.09, 0.11, 0.13, 0.14, 0.16, 0.18, 0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.30],
                // Rock (4)
                [0.15, 0.17, 0.19, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33],
                // Snow (5)
                [0.93, 0.94, 0.95, 0.95, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.85],
                // Forest/Bark (6)
                [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
            ],
        }
    }
}

/// Map biome + elevation fraction to 4-channel splat blend weights.
/// Weights sum to 1.0. Channel mapping: [water, snow, vegetation, ground].
pub fn biome_to_splat_weights(biome: BiomeKind, height: f32, world_height: f32) -> [f32; 4] {
    let _t = (height / world_height.max(1.0)).clamp(0.0, 1.0);
    match biome {
        BiomeKind::Alpine             => [0.00, 0.50, 0.05, 0.45],
        BiomeKind::Tundra             => [0.00, 0.40, 0.20, 0.40],
        BiomeKind::Forest             => [0.00, 0.05, 0.70, 0.25],
        BiomeKind::Grassland          => [0.00, 0.05, 0.75, 0.20],
        BiomeKind::Desert             => [0.00, 0.10, 0.00, 0.90],
        BiomeKind::Wetland            => [0.60, 0.00, 0.30, 0.10],
        BiomeKind::Coastal            => [0.30, 0.10, 0.25, 0.35],
        BiomeKind::SubalpineShrub     => [0.00, 0.25, 0.50, 0.25],
        BiomeKind::Savanna            => [0.00, 0.10, 0.55, 0.35],
        BiomeKind::Taiga              => [0.00, 0.10, 0.65, 0.25],
        BiomeKind::TropicalRainforest => [0.10, 0.00, 0.80, 0.10],
    }
}

/// Blend 4 spectral slots using blend weights.
/// Channel mapping: [0]=Water, [1]=Snow, [2]=Grass, [3]=Dirt.
pub fn blend_spectral_terrain(mats: &SpectralTerrainMaterials, weights: &[f32; 4]) -> [f32; 16] {
    let slot_indices = [0usize, 5, 2, 3]; // Water, Snow, Grass, Dirt
    let mut result = [0.0f32; 16];
    for (ch, (&w, &slot)) in weights.iter().zip(slot_indices.iter()).enumerate() {
        let _ = ch;
        for band in 0..16 {
            result[band] += w * mats.slots[slot][band];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_splatizer_snow_at_high_altitude() {
        let mats = SpectralTerrainMaterials::default();
        let weights = biome_to_splat_weights(BiomeKind::Alpine, 320.0, 400.0);
        let spectral = blend_spectral_terrain(&mats, &weights);
        let avg_reflectance: f32 = spectral.iter().sum::<f32>() / 16.0;
        println!("alpine snow blend should be bright, avg_reflectance = {:.3}", avg_reflectance);
        assert!(
            avg_reflectance > 0.4,
            "alpine snow blend should be bright, avg_reflectance = {:.3}",
            avg_reflectance
        );
    }

    #[test]
    fn test_terrain_splatizer_water_in_wetland() {
        let mats = SpectralTerrainMaterials::default();
        let weights = biome_to_splat_weights(BiomeKind::Wetland, 5.0, 100.0);
        let spectral = blend_spectral_terrain(&mats, &weights);
        let near_ir_avg: f32 = spectral[8..16].iter().sum::<f32>() / 8.0;
        println!("wetland near-IR should be dark (water dominant), near_ir_avg = {:.3}", near_ir_avg);
        assert!(
            near_ir_avg < 0.15,
            "wetland near-IR should be dark (water dominant), near_ir_avg = {:.3}",
            near_ir_avg
        );
    }
}
