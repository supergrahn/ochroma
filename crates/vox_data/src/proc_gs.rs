use glam::Vec3;
use half::f16;
use rand::SeedableRng;
use rand::Rng;
use rand::rngs::StdRng;
use serde::Deserialize;
use vox_core::types::GaussianSplat;

use crate::materials::MaterialLibrary;

#[derive(Debug, Deserialize, Clone)]
pub struct RuleHeader {
    pub asset_type: String,
    pub style: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub enum GeometryStrategy {
    StructuredPlacement,
    GrowthAlgorithm,
    ComponentAssembly,
    SurfaceScattering,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GeometryConfig {
    pub strategy: GeometryStrategy,
    pub floor_count_min: u32,
    pub floor_count_max: u32,
    pub height_min: f32,
    pub height_max: f32,
    pub width_min: f32,
    pub width_max: f32,
    pub depth_min: f32,
    pub depth_max: f32,
    pub splats_per_sqm: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MaterialZoneConfig {
    pub name: String,
    pub material_tag: String,
    pub zone_type: String, // "wall", "roof", "floor", etc.
    pub coverage: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct VariationConfig {
    pub scale_min: f32,
    pub scale_max: f32,
    pub opacity_min: f32,
    pub opacity_max: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SplatRule {
    pub header: RuleHeader,
    pub geometry: GeometryConfig,
    pub material_zones: Vec<MaterialZoneConfig>,
    pub variation: VariationConfig,
}

fn make_splat(
    rng: &mut StdRng,
    position: Vec3,
    scale: f32,
    opacity: f32,
    spd: &[f32; 8],
) -> GaussianSplat {
    let mut splat = GaussianSplat {
        position: position.into(),
        scale: [scale, scale, scale],
        rotation: [0i16, 0, 0, 32767],
        opacity: (opacity.clamp(0.0, 1.0) * 255.0) as u8,
        _pad: [0u8; 3],
        spectral: [0u16; 8],
    };

    // Random small rotation perturbation
    let rx = rng.random_range(-0.1f32..0.1);
    let ry = rng.random_range(-0.1f32..0.1);
    splat.rotation[0] = (rx * 32767.0) as i16;
    splat.rotation[1] = (ry * 32767.0) as i16;

    for (i, &v) in spd.iter().enumerate() {
        splat.spectral[i] = f16::from_f32(v).to_bits();
    }

    splat
}

/// Simple building generation without TOML rules — uses direct parameters.
pub fn emit_splats_simple(seed: u64, width: f32, depth: f32) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let floors = 2 + rng.random_range(0..3u32);
    let floor_height = 3.0 + rng.random::<f32>() * 0.5;
    let total_height = floors as f32 * floor_height;

    let brick_spd: [u16; 8] = [
        f16::from_f32(0.08).to_bits(),
        f16::from_f32(0.08).to_bits(),
        f16::from_f32(0.10).to_bits(),
        f16::from_f32(0.15).to_bits(),
        f16::from_f32(0.25).to_bits(),
        f16::from_f32(0.55).to_bits(),
        f16::from_f32(0.65).to_bits(),
        f16::from_f32(0.60).to_bits(),
    ];
    let roof_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.15).to_bits());

    let mut splats = Vec::new();

    // Four walls
    let density = 500.0; // splats per m² of wall
    for wall in 0..4 {
        let (_wall_w, _wall_d, wx, wz, flip) = match wall {
            0 => (width, 0.0, 0.0, 0.0, false),    // front
            1 => (width, 0.0, 0.0, -depth, false),  // back
            2 => (0.0, depth, 0.0, 0.0, true),      // left
            _ => (0.0, depth, width, 0.0, true),     // right
        };

        let area = if flip {
            depth * total_height
        } else {
            width * total_height
        };
        let count = (area * density / 100.0) as usize; // reduced density for perf
        for _ in 0..count {
            let (x, z) = if flip {
                (wx, wz - rng.random::<f32>() * depth)
            } else {
                (rng.random::<f32>() * width, wz)
            };
            let y = rng.random::<f32>() * total_height;
            let s = 0.04 + rng.random::<f32>() * 0.04;
            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [s, s, s * 0.3],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: brick_spd,
            });
        }
    }

    // Roof
    let roof_count = (width * depth * density / 100.0) as usize;
    for _ in 0..roof_count {
        splats.push(GaussianSplat {
            position: [
                rng.random::<f32>() * width,
                total_height,
                -rng.random::<f32>() * depth,
            ],
            scale: [0.06, 0.02, 0.06],
            rotation: [0, 0, 0, 32767],
            opacity: 245,
            _pad: [0; 3],
            spectral: roof_spd,
        });
    }

    splats
}

pub fn emit_splats(rule: &SplatRule, seed: u64) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let lib = MaterialLibrary::default();

    let floor_count = rng.random_range(rule.geometry.floor_count_min..=rule.geometry.floor_count_max);
    let height = rng.random_range(rule.geometry.height_min..rule.geometry.height_max);
    let width = rng.random_range(rule.geometry.width_min..rule.geometry.width_max);
    let depth = rng.random_range(rule.geometry.depth_min..rule.geometry.depth_max);
    let floor_height = height / floor_count as f32;

    let mut splats = Vec::new();

    for zone in &rule.material_zones {
        // Look up SPD from material library, fall back to neutral grey
        let spd: [f32; 8] = if let Some(mat) = lib.get(&zone.material_tag) {
            mat.spd.0
        } else {
            [0.5f32; 8]
        };

        let zone_type = zone.zone_type.as_str();

        match zone_type {
            "roof" => {
                let area = width * depth;
                let count = (area * rule.geometry.splats_per_sqm * zone.coverage) as u32;
                for _ in 0..count {
                    let x = rng.random_range(-width / 2.0..width / 2.0);
                    let z = rng.random_range(-depth / 2.0..depth / 2.0);
                    let y = height;
                    let scale = rng.random_range(rule.variation.scale_min..rule.variation.scale_max);
                    let opacity = rng.random_range(rule.variation.opacity_min..rule.variation.opacity_max);
                    splats.push(make_splat(&mut rng, Vec3::new(x, y, z), scale, opacity, &spd));
                }
            }
            "wall" | _ => {
                // 4 walls, each floor
                for floor in 0..floor_count {
                    let base_y = floor as f32 * floor_height;
                    // 4 faces: front, back, left, right
                    let faces: [(Vec3, Vec3, f32, f32); 4] = [
                        // front (+z)
                        (Vec3::new(0.0, base_y + floor_height / 2.0, depth / 2.0),
                         Vec3::new(1.0, 0.0, 0.0), width, floor_height),
                        // back (-z)
                        (Vec3::new(0.0, base_y + floor_height / 2.0, -depth / 2.0),
                         Vec3::new(1.0, 0.0, 0.0), width, floor_height),
                        // left (-x)
                        (Vec3::new(-width / 2.0, base_y + floor_height / 2.0, 0.0),
                         Vec3::new(0.0, 0.0, 1.0), depth, floor_height),
                        // right (+x)
                        (Vec3::new(width / 2.0, base_y + floor_height / 2.0, 0.0),
                         Vec3::new(0.0, 0.0, 1.0), depth, floor_height),
                    ];

                    for (center, tangent, face_w, face_h) in &faces {
                        let area = face_w * face_h;
                        let count = (area * rule.geometry.splats_per_sqm * zone.coverage) as u32;
                        for _ in 0..count {
                            let u = rng.random_range(-face_w / 2.0..face_w / 2.0);
                            let v = rng.random_range(-face_h / 2.0..face_h / 2.0);
                            let pos = *center + *tangent * u + Vec3::Y * v;
                            let scale = rng.random_range(rule.variation.scale_min..rule.variation.scale_max);
                            let opacity = rng.random_range(rule.variation.opacity_min..rule.variation.opacity_max);
                            splats.push(make_splat(&mut rng, pos, scale, opacity, &spd));
                        }
                    }
                }
            }
        }
    }

    splats
}
