use crate::proc_gs_advanced::*;
use half::f16;
use vox_core::types::GaussianSplat;

/// A catalog entry describing a procedural asset.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub name: String,
    pub category: AssetCategory,
    pub generator: AssetGenerator,
    pub splat_count_estimate: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetCategory {
    ResidentialBuilding,
    CommercialBuilding,
    IndustrialBuilding,
    ServiceBuilding,
    Tree,
    Prop,
    Vehicle,
    TerrainTile,
}

#[derive(Debug, Clone)]
pub enum AssetGenerator {
    /// Building with given width, depth, floor range.
    Building {
        width_range: (f32, f32),
        depth: f32,
        floors_range: (u32, u32),
        style: BuildingStyle,
    },
    /// Tree with height and canopy radius ranges.
    Tree {
        height_range: (f32, f32),
        canopy_range: (f32, f32),
    },
    /// Bench prop.
    Bench,
    /// Lamp post with height range.
    LampPost { height_range: (f32, f32) },
    /// Grass patch.
    GrassPatch { size: f32, density: f32 },
    /// Custom splat generator function (uses a fixed SPD pattern).
    Custom { description: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingStyle {
    Victorian,
    Modern,
    Suburban,
    Industrial,
    Commercial,
}

impl CatalogEntry {
    /// Generate splats for this entry with a given seed.
    pub fn generate(&self, seed: u64) -> Vec<GaussianSplat> {
        match &self.generator {
            AssetGenerator::Building {
                width_range,
                depth,
                floors_range,
                style,
            } => {
                let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
                use rand::Rng;
                let width = rng.random_range(width_range.0..=width_range.1);
                let floors = rng.random_range(floors_range.0..=floors_range.1);
                generate_building_with_style(seed, width, *depth, floors, *style)
            }
            AssetGenerator::Tree {
                height_range,
                canopy_range,
            } => {
                let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
                use rand::Rng;
                let height = rng.random_range(height_range.0..=height_range.1);
                let canopy = rng.random_range(canopy_range.0..=canopy_range.1);
                generate_tree(seed, height, canopy)
            }
            AssetGenerator::Bench => generate_bench(seed),
            AssetGenerator::LampPost { height_range } => {
                let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
                use rand::Rng;
                let h = rng.random_range(height_range.0..=height_range.1);
                generate_lamp_post(seed, h)
            }
            AssetGenerator::GrassPatch { size, density } => {
                generate_grass_patch(seed, *size, *density)
            }
            AssetGenerator::Custom { .. } => generate_placeholder_cube(seed),
        }
    }
}

/// Generate a building with style-specific SPD values.
fn generate_building_with_style(
    seed: u64,
    width: f32,
    depth: f32,
    floors: u32,
    style: BuildingStyle,
) -> Vec<GaussianSplat> {
    let (wall_spd, roof_spd) = match style {
        BuildingStyle::Victorian => (
            // Red brick
            [0.08, 0.08, 0.10, 0.15, 0.25, 0.55, 0.65, 0.60],
            // Slate grey
            [0.12, 0.13, 0.15, 0.17, 0.18, 0.18, 0.17, 0.16],
        ),
        BuildingStyle::Modern => (
            // Glass/steel (bright, high reflectance)
            [0.40, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45],
            // Dark flat roof
            [0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05],
        ),
        BuildingStyle::Suburban => (
            // Painted wood (warm beige)
            [0.15, 0.18, 0.22, 0.28, 0.32, 0.35, 0.33, 0.30],
            // Terracotta
            [0.10, 0.10, 0.12, 0.18, 0.30, 0.50, 0.55, 0.50],
        ),
        BuildingStyle::Industrial => (
            // Corrugated metal
            [0.20, 0.22, 0.25, 0.28, 0.30, 0.30, 0.28, 0.25],
            // Same metal roof
            [0.20, 0.22, 0.25, 0.28, 0.30, 0.30, 0.28, 0.25],
        ),
        BuildingStyle::Commercial => (
            // Concrete with glass
            [0.25, 0.27, 0.30, 0.32, 0.33, 0.33, 0.31, 0.28],
            // Flat grey
            [0.10, 0.10, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10],
        ),
    };

    generate_styled_building(seed, width, depth, floors, wall_spd, roof_spd)
}

fn generate_styled_building(
    seed: u64,
    width: f32,
    depth: f32,
    floors: u32,
    wall_spd: [f32; 8],
    roof_spd: [f32; 8],
) -> Vec<GaussianSplat> {
    use rand::prelude::*;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let total_height = floors as f32 * 3.5;
    let wall_spectral: [u16; 8] = std::array::from_fn(|i| f16::from_f32(wall_spd[i]).to_bits());
    let roof_spectral: [u16; 8] = std::array::from_fn(|i| f16::from_f32(roof_spd[i]).to_bits());

    let mut splats = Vec::new();
    let density = 400.0; // splats per m² reduced for performance

    // Front and back walls
    for &z in &[0.0f32, -depth] {
        let area = width * total_height;
        let count = (area * density / 100.0) as usize;
        for _ in 0..count {
            let x = rng.random_range(0.0..width);
            let y = rng.random_range(0.0..total_height);
            let s = 0.04 + rng.random::<f32>() * 0.04;
            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [s, s, s * 0.3],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: wall_spectral,
            });
        }
    }

    // Left and right walls
    for &x in &[0.0f32, width] {
        let area = depth * total_height;
        let count = (area * density / 100.0) as usize;
        for _ in 0..count {
            let z = -rng.random_range(0.0..depth);
            let y = rng.random_range(0.0..total_height);
            let s = 0.04 + rng.random::<f32>() * 0.04;
            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [s * 0.3, s, s],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: wall_spectral,
            });
        }
    }

    // Roof
    let roof_count = (width * depth * density / 100.0) as usize;
    for _ in 0..roof_count {
        splats.push(GaussianSplat {
            position: [
                rng.random_range(0.0..width),
                total_height,
                -rng.random_range(0.0..depth),
            ],
            scale: [0.06, 0.02, 0.06],
            rotation: [0, 0, 0, 32767],
            opacity: 245,
            _pad: [0; 3],
            spectral: roof_spectral,
        });
    }

    splats
}

fn generate_placeholder_cube(seed: u64) -> Vec<GaussianSplat> {
    let _ = seed;
    let grey_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.4).to_bits());
    let mut splats = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                splats.push(GaussianSplat {
                    position: [x as f32 * 0.5, y as f32 * 0.5, z as f32 * 0.5],
                    scale: [0.2, 0.2, 0.2],
                    rotation: [0, 0, 0, 32767],
                    opacity: 200,
                    _pad: [0; 3],
                    spectral: grey_spd,
                });
            }
        }
    }
    splats
}

/// Create the default asset catalog with all built-in asset types.
pub fn default_catalog() -> Vec<CatalogEntry> {
    vec![
        // Residential
        CatalogEntry {
            name: "Victorian Terraced House".into(),
            category: AssetCategory::ResidentialBuilding,
            generator: AssetGenerator::Building {
                width_range: (4.5, 6.0),
                depth: 12.0,
                floors_range: (2, 4),
                style: BuildingStyle::Victorian,
            },
            splat_count_estimate: 2000,
        },
        CatalogEntry {
            name: "Modern Apartment".into(),
            category: AssetCategory::ResidentialBuilding,
            generator: AssetGenerator::Building {
                width_range: (8.0, 12.0),
                depth: 15.0,
                floors_range: (4, 8),
                style: BuildingStyle::Modern,
            },
            splat_count_estimate: 5000,
        },
        CatalogEntry {
            name: "Suburban House".into(),
            category: AssetCategory::ResidentialBuilding,
            generator: AssetGenerator::Building {
                width_range: (8.0, 12.0),
                depth: 10.0,
                floors_range: (1, 2),
                style: BuildingStyle::Suburban,
            },
            splat_count_estimate: 1500,
        },
        // Commercial
        CatalogEntry {
            name: "Corner Shop".into(),
            category: AssetCategory::CommercialBuilding,
            generator: AssetGenerator::Building {
                width_range: (6.0, 10.0),
                depth: 8.0,
                floors_range: (1, 2),
                style: BuildingStyle::Commercial,
            },
            splat_count_estimate: 1200,
        },
        CatalogEntry {
            name: "Office Block".into(),
            category: AssetCategory::CommercialBuilding,
            generator: AssetGenerator::Building {
                width_range: (15.0, 25.0),
                depth: 20.0,
                floors_range: (5, 12),
                style: BuildingStyle::Modern,
            },
            splat_count_estimate: 8000,
        },
        // Industrial
        CatalogEntry {
            name: "Warehouse".into(),
            category: AssetCategory::IndustrialBuilding,
            generator: AssetGenerator::Building {
                width_range: (20.0, 30.0),
                depth: 25.0,
                floors_range: (1, 2),
                style: BuildingStyle::Industrial,
            },
            splat_count_estimate: 3000,
        },
        CatalogEntry {
            name: "Factory".into(),
            category: AssetCategory::IndustrialBuilding,
            generator: AssetGenerator::Building {
                width_range: (25.0, 40.0),
                depth: 30.0,
                floors_range: (2, 3),
                style: BuildingStyle::Industrial,
            },
            splat_count_estimate: 5000,
        },
        // Service
        CatalogEntry {
            name: "School".into(),
            category: AssetCategory::ServiceBuilding,
            generator: AssetGenerator::Building {
                width_range: (20.0, 25.0),
                depth: 15.0,
                floors_range: (2, 3),
                style: BuildingStyle::Suburban,
            },
            splat_count_estimate: 3000,
        },
        CatalogEntry {
            name: "Hospital".into(),
            category: AssetCategory::ServiceBuilding,
            generator: AssetGenerator::Building {
                width_range: (30.0, 40.0),
                depth: 25.0,
                floors_range: (3, 6),
                style: BuildingStyle::Modern,
            },
            splat_count_estimate: 6000,
        },
        CatalogEntry {
            name: "Fire Station".into(),
            category: AssetCategory::ServiceBuilding,
            generator: AssetGenerator::Building {
                width_range: (15.0, 20.0),
                depth: 12.0,
                floors_range: (1, 2),
                style: BuildingStyle::Industrial,
            },
            splat_count_estimate: 1500,
        },
        // Trees
        CatalogEntry {
            name: "Oak Tree".into(),
            category: AssetCategory::Tree,
            generator: AssetGenerator::Tree {
                height_range: (6.0, 12.0),
                canopy_range: (3.0, 5.0),
            },
            splat_count_estimate: 3000,
        },
        CatalogEntry {
            name: "Pine Tree".into(),
            category: AssetCategory::Tree,
            generator: AssetGenerator::Tree {
                height_range: (8.0, 15.0),
                canopy_range: (2.0, 3.5),
            },
            splat_count_estimate: 2500,
        },
        CatalogEntry {
            name: "Small Shrub".into(),
            category: AssetCategory::Tree,
            generator: AssetGenerator::Tree {
                height_range: (1.0, 2.0),
                canopy_range: (1.0, 1.5),
            },
            splat_count_estimate: 500,
        },
        // Props
        CatalogEntry {
            name: "Park Bench".into(),
            category: AssetCategory::Prop,
            generator: AssetGenerator::Bench,
            splat_count_estimate: 150,
        },
        CatalogEntry {
            name: "Street Lamp".into(),
            category: AssetCategory::Prop,
            generator: AssetGenerator::LampPost {
                height_range: (4.0, 5.5),
            },
            splat_count_estimate: 80,
        },
        CatalogEntry {
            name: "Grass Patch".into(),
            category: AssetCategory::TerrainTile,
            generator: AssetGenerator::GrassPatch {
                size: 10.0,
                density: 50.0,
            },
            splat_count_estimate: 5000,
        },
        // Vehicles
        CatalogEntry {
            name: "Car".into(),
            category: AssetCategory::Vehicle,
            generator: AssetGenerator::Custom {
                description: "Sedan car".into(),
            },
            splat_count_estimate: 500,
        },
        CatalogEntry {
            name: "Bus".into(),
            category: AssetCategory::Vehicle,
            generator: AssetGenerator::Custom {
                description: "City bus".into(),
            },
            splat_count_estimate: 800,
        },
    ]
}
