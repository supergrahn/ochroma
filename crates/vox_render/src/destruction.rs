use vox_core::types::GaussianSplat;
use glam::Vec3;

/// A destruction mask applied to an instance at render time.
#[derive(Debug, Clone)]
pub struct DestructionMask {
    pub instance_id: u32,
    pub impact_point: Vec3,
    pub radius: f32,
    pub progression: f32, // 0.0 = intact, 1.0 = fully destroyed
}

/// Apply destruction masks to a set of splats.
/// Returns the modified splats with reduced opacity in destruction zones.
pub fn apply_destruction_masks(
    splats: &[GaussianSplat],
    masks: &[DestructionMask],
) -> Vec<GaussianSplat> {
    let mut result = splats.to_vec();

    for mask in masks {
        for splat in &mut result {
            let pos = Vec3::new(splat.position[0], splat.position[1], splat.position[2]);
            let dist = pos.distance(mask.impact_point);

            if dist < mask.radius {
                // Scale opacity based on distance from impact and progression
                let falloff = (dist / mask.radius).max(0.0);
                let destruction_factor = (1.0 - mask.progression * (1.0 - falloff)).max(0.0);
                splat.opacity = (splat.opacity as f32 * destruction_factor) as u8;
            }
        }
    }

    result
}

/// Generate debris splats from a destruction event.
pub fn generate_debris(
    impact_point: Vec3,
    radius: f32,
    debris_count: usize,
    seed: u64,
) -> Vec<GaussianSplat> {
    use rand::prelude::*;
    use rand::SeedableRng;
    use half::f16;

    let mut rng = StdRng::seed_from_u64(seed);
    let debris_spd = [
        f16::from_f32(0.15).to_bits(), f16::from_f32(0.15).to_bits(),
        f16::from_f32(0.18).to_bits(), f16::from_f32(0.20).to_bits(),
        f16::from_f32(0.20).to_bits(), f16::from_f32(0.20).to_bits(),
        f16::from_f32(0.18).to_bits(), f16::from_f32(0.16).to_bits(),
    ];

    (0..debris_count)
        .map(|_| {
            let angle = rng.random::<f32>() * std::f32::consts::TAU;
            let dist = rng.random::<f32>() * radius;
            let height = rng.random::<f32>() * radius * 0.5;
            GaussianSplat {
                position: [
                    impact_point.x + angle.cos() * dist,
                    impact_point.y + height,
                    impact_point.z + angle.sin() * dist,
                ],
                scale: [0.03 + rng.random::<f32>() * 0.05, 0.03, 0.03],
                rotation: [0, 0, 0, 32767],
                opacity: (150.0 + rng.random::<f32>() * 100.0) as u8,
                _pad: [0; 3],
                spectral: debris_spd,
            }
        })
        .collect()
}
