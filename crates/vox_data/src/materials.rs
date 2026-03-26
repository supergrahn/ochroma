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

        // Bands: [380nm, 420nm, 460nm, 500nm, 540nm, 580nm, 620nm, 660nm]

        lib.register(SpectralMaterial {
            tag: "concrete_raw".to_string(),
            description: "Raw concrete surface, diffuse grey".to_string(),
            spd: SpectralBands([0.30, 0.32, 0.34, 0.36, 0.37, 0.38, 0.38, 0.37]),
            spd_worn: SpectralBands([0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.36, 0.35]),
        });

        lib.register(SpectralMaterial {
            tag: "brick_red".to_string(),
            description: "Red clay brick".to_string(),
            // Red brick: low at 380-460nm (blue/violet), rises toward 580-660nm (orange/red)
            spd: SpectralBands([0.05, 0.06, 0.07, 0.09, 0.15, 0.45, 0.65, 0.70]),
            spd_worn: SpectralBands([0.04, 0.05, 0.06, 0.08, 0.13, 0.40, 0.58, 0.62]),
        });

        lib.register(SpectralMaterial {
            tag: "glass_clear".to_string(),
            description: "Clear float glass, highly transmissive".to_string(),
            spd: SpectralBands([0.08, 0.08, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07]),
            spd_worn: SpectralBands([0.10, 0.10, 0.11, 0.11, 0.10, 0.10, 0.09, 0.09]),
        });

        lib.register(SpectralMaterial {
            tag: "vegetation_leaf".to_string(),
            description: "Green vegetation, chlorophyll absorption".to_string(),
            // Vegetation: absorbs blue/red (photosynthesis), reflects strongly at 540nm green
            spd: SpectralBands([0.04, 0.05, 0.06, 0.10, 0.42, 0.15, 0.06, 0.05]),
            spd_worn: SpectralBands([0.05, 0.06, 0.07, 0.11, 0.35, 0.18, 0.08, 0.07]),
        });

        lib.register(SpectralMaterial {
            tag: "metal_steel".to_string(),
            description: "Polished steel, broad reflectance".to_string(),
            spd: SpectralBands([0.55, 0.58, 0.60, 0.62, 0.63, 0.65, 0.66, 0.67]),
            spd_worn: SpectralBands([0.45, 0.48, 0.50, 0.52, 0.53, 0.54, 0.55, 0.56]),
        });

        lib.register(SpectralMaterial {
            tag: "metal_oxidized".to_string(),
            description: "Oxidized/rusted metal surface".to_string(),
            // Rust: reddish-orange, higher at 580-660nm
            spd: SpectralBands([0.06, 0.07, 0.08, 0.10, 0.18, 0.42, 0.55, 0.58]),
            spd_worn: SpectralBands([0.07, 0.08, 0.09, 0.11, 0.20, 0.45, 0.58, 0.61]),
        });

        lib.register(SpectralMaterial {
            tag: "asphalt_dry".to_string(),
            description: "Dry asphalt road surface".to_string(),
            spd: SpectralBands([0.04, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08]),
            spd_worn: SpectralBands([0.06, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10]),
        });

        lib.register(SpectralMaterial {
            tag: "slate_grey".to_string(),
            description: "Slate grey roofing material".to_string(),
            spd: SpectralBands([0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.22, 0.22]),
            spd_worn: SpectralBands([0.16, 0.17, 0.18, 0.19, 0.20, 0.20, 0.21, 0.21]),
        });

        lib.register(SpectralMaterial {
            tag: "water_still".to_string(),
            description: "Still water surface, high blue reflectance".to_string(),
            // Water: reflects more blue/cyan, absorbs red
            spd: SpectralBands([0.06, 0.08, 0.10, 0.12, 0.10, 0.07, 0.04, 0.03]),
            spd_worn: SpectralBands([0.06, 0.08, 0.10, 0.12, 0.10, 0.07, 0.04, 0.03]),
        });

        lib.register(SpectralMaterial {
            tag: "soil_dry".to_string(),
            description: "Dry soil, brownish earth tone".to_string(),
            spd: SpectralBands([0.08, 0.09, 0.10, 0.13, 0.18, 0.25, 0.30, 0.32]),
            spd_worn: SpectralBands([0.07, 0.08, 0.09, 0.12, 0.17, 0.23, 0.28, 0.30]),
        });

        lib.register(SpectralMaterial {
            tag: "wood_painted_green".to_string(),
            description: "Wood painted with green paint".to_string(),
            // Green paint: reflects around 500-580nm
            spd: SpectralBands([0.04, 0.05, 0.08, 0.20, 0.38, 0.25, 0.08, 0.06]),
            spd_worn: SpectralBands([0.05, 0.06, 0.09, 0.18, 0.32, 0.22, 0.09, 0.07]),
        });

        lib
    }
}
