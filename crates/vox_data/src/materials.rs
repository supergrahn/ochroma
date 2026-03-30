use std::collections::HashMap;
use vox_core::spectral::SpectralBands;

#[derive(Debug, Clone)]
pub struct SpectralMaterial {
    pub tag: String,
    pub description: String,
    pub spd: SpectralBands,
    pub spd_worn: SpectralBands,
}

pub struct MaterialLibrary {
    materials: HashMap<String, SpectralMaterial>,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        Self { materials: HashMap::new() }
    }

    pub fn register(&mut self, material: SpectralMaterial) {
        self.materials.insert(material.tag.clone(), material);
    }

    pub fn get(&self, tag: &str) -> Option<&SpectralMaterial> {
        self.materials.get(tag)
    }

    pub fn all(&self) -> impl Iterator<Item = &SpectralMaterial> {
        self.materials.values()
    }
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        let mut lib = MaterialLibrary::new();

        // Bands: [380, 405, 430, 455, 480, 505, 530, 555, 580, 605, 630, 655, 680, 705, 730, 755 nm]

        lib.register(SpectralMaterial {
            tag: "concrete_raw".to_string(),
            description: "Raw concrete surface, diffuse grey".to_string(),
            spd: SpectralBands([0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.38, 0.38, 0.38, 0.37, 0.37, 0.37, 0.36]),
            spd_worn: SpectralBands([0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.34, 0.36, 0.36, 0.36, 0.36, 0.35, 0.35, 0.34, 0.34]),
        });

        lib.register(SpectralMaterial {
            tag: "brick_red".to_string(),
            description: "Red clay brick".to_string(),
            // Red brick: low at 380-455nm (blue/violet), rises toward 580-660nm (orange/red), plateau into NIR
            spd: SpectralBands([0.05, 0.05, 0.06, 0.07, 0.08, 0.10, 0.14, 0.25, 0.45, 0.60, 0.68, 0.70, 0.71, 0.72, 0.72, 0.73]),
            spd_worn: SpectralBands([0.04, 0.04, 0.05, 0.06, 0.07, 0.09, 0.12, 0.21, 0.40, 0.53, 0.60, 0.62, 0.63, 0.64, 0.64, 0.65]),
        });

        lib.register(SpectralMaterial {
            tag: "glass_clear".to_string(),
            description: "Clear float glass, highly transmissive".to_string(),
            spd: SpectralBands([0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]),
            spd_worn: SpectralBands([0.10, 0.10, 0.10, 0.11, 0.11, 0.11, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]),
        });

        lib.register(SpectralMaterial {
            tag: "vegetation_leaf".to_string(),
            description: "Green vegetation, chlorophyll absorption".to_string(),
            // Vegetation: absorbs blue/red (photosynthesis), reflects strongly at 530-555nm green, NIR plateau
            spd: SpectralBands([0.04, 0.04, 0.05, 0.06, 0.08, 0.20, 0.38, 0.42, 0.15, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05]),
            spd_worn: SpectralBands([0.05, 0.05, 0.06, 0.07, 0.09, 0.18, 0.30, 0.35, 0.18, 0.10, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07]),
        });

        lib.register(SpectralMaterial {
            tag: "metal_steel".to_string(),
            description: "Polished steel, broad reflectance".to_string(),
            spd: SpectralBands([0.55, 0.56, 0.57, 0.58, 0.60, 0.61, 0.62, 0.63, 0.65, 0.65, 0.66, 0.67, 0.67, 0.68, 0.68, 0.69]),
            spd_worn: SpectralBands([0.45, 0.46, 0.47, 0.48, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.55, 0.56, 0.56, 0.57, 0.57, 0.58]),
        });

        lib.register(SpectralMaterial {
            tag: "metal_oxidized".to_string(),
            description: "Oxidized/rusted metal surface".to_string(),
            // Rust: reddish-orange, higher at 580-660nm, extends into NIR
            spd: SpectralBands([0.06, 0.06, 0.07, 0.08, 0.10, 0.13, 0.18, 0.30, 0.42, 0.52, 0.56, 0.58, 0.59, 0.60, 0.60, 0.61]),
            spd_worn: SpectralBands([0.07, 0.07, 0.08, 0.09, 0.11, 0.15, 0.20, 0.33, 0.45, 0.55, 0.59, 0.61, 0.62, 0.63, 0.63, 0.64]),
        });

        lib.register(SpectralMaterial {
            tag: "asphalt_dry".to_string(),
            description: "Dry asphalt road surface".to_string(),
            spd: SpectralBands([0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09]),
            spd_worn: SpectralBands([0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.09, 0.10, 0.10, 0.10, 0.10, 0.10, 0.11, 0.11]),
        });

        lib.register(SpectralMaterial {
            tag: "slate_grey".to_string(),
            description: "Slate grey roofing material".to_string(),
            spd: SpectralBands([0.18, 0.19, 0.19, 0.20, 0.20, 0.21, 0.21, 0.21, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.23, 0.23]),
            spd_worn: SpectralBands([0.16, 0.17, 0.17, 0.18, 0.18, 0.19, 0.19, 0.20, 0.20, 0.20, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21]),
        });

        lib.register(SpectralMaterial {
            tag: "water_still".to_string(),
            description: "Still water surface, high blue reflectance".to_string(),
            // Water: reflects more blue/cyan, absorbs red and NIR
            spd: SpectralBands([0.06, 0.07, 0.08, 0.10, 0.12, 0.12, 0.11, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02]),
            spd_worn: SpectralBands([0.06, 0.07, 0.08, 0.10, 0.12, 0.12, 0.11, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02]),
        });

        lib.register(SpectralMaterial {
            tag: "soil_dry".to_string(),
            description: "Dry soil, brownish earth tone".to_string(),
            spd: SpectralBands([0.08, 0.09, 0.09, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32, 0.33, 0.34, 0.35, 0.35]),
            spd_worn: SpectralBands([0.07, 0.08, 0.08, 0.09, 0.11, 0.13, 0.17, 0.20, 0.23, 0.26, 0.28, 0.30, 0.31, 0.32, 0.33, 0.33]),
        });

        lib.register(SpectralMaterial {
            tag: "wood_painted_green".to_string(),
            description: "Wood painted with green paint".to_string(),
            // Green paint: reflects around 505-555nm
            spd: SpectralBands([0.04, 0.04, 0.05, 0.07, 0.12, 0.28, 0.38, 0.38, 0.25, 0.12, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06]),
            spd_worn: SpectralBands([0.05, 0.05, 0.06, 0.08, 0.13, 0.24, 0.32, 0.32, 0.22, 0.12, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07]),
        });

        lib
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_library_has_base_materials() {
        let lib = MaterialLibrary::default();
        assert!(lib.get("concrete_raw").is_some());
        assert!(lib.get("brick_red").is_some());
        assert!(lib.get("glass_clear").is_some());
        assert!(lib.get("vegetation_leaf").is_some());
    }

    #[test]
    fn brick_red_has_red_dominant_spectrum() {
        let lib = MaterialLibrary::default();
        let brick = lib.get("brick_red").unwrap();
        // Band 6 (620nm, red) should be much higher than band 2 (460nm, blue)
        assert!(brick.spd.0[6] > brick.spd.0[2],
            "brick red band 6 ({}) should exceed band 2 ({})", brick.spd.0[6], brick.spd.0[2]);
    }

    #[test]
    fn register_and_retrieve_custom_material() {
        let mut lib = MaterialLibrary::new();
        lib.register(SpectralMaterial {
            tag: "test_mat".to_string(),
            description: "test".to_string(),
            spd: SpectralBands([0.5; 16]),
            spd_worn: SpectralBands([0.4; 16]),
        });
        let mat = lib.get("test_mat").expect("custom material should exist");
        assert_eq!(mat.spd.0[0], 0.5);
    }

    #[test]
    fn nonexistent_material_returns_none() {
        let lib = MaterialLibrary::default();
        assert!(lib.get("unobtainium").is_none());
    }
}
