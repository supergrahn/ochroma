use half::f16;
use rand::prelude::*;
use rand::SeedableRng;
use vox_core::types::GaussianSplat;

/// Growth algorithm for trees and organic structures.
pub fn generate_tree(seed: u64, height: f32, canopy_radius: f32) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut splats = Vec::new();

    // Trunk SPD (brown bark)
    let trunk_spd: [u16; 8] = [
        f16::from_f32(0.06).to_bits(),
        f16::from_f32(0.07).to_bits(),
        f16::from_f32(0.08).to_bits(),
        f16::from_f32(0.10).to_bits(),
        f16::from_f32(0.12).to_bits(),
        f16::from_f32(0.15).to_bits(),
        f16::from_f32(0.13).to_bits(),
        f16::from_f32(0.10).to_bits(),
    ];

    // Leaf SPD (green)
    let leaf_spd: [u16; 8] = [
        f16::from_f32(0.03).to_bits(),
        f16::from_f32(0.04).to_bits(),
        f16::from_f32(0.06).to_bits(),
        f16::from_f32(0.10).to_bits(),
        f16::from_f32(0.45).to_bits(),
        f16::from_f32(0.30).to_bits(),
        f16::from_f32(0.08).to_bits(),
        f16::from_f32(0.04).to_bits(),
    ];

    // Trunk: cylinder of splats
    let trunk_radius = 0.15 + height * 0.02;
    let trunk_segments = (height * 5.0) as usize;
    for i in 0..trunk_segments {
        let y = i as f32 / trunk_segments as f32 * height;
        let radius = trunk_radius * (1.0 - y / height * 0.6); // taper
        let circumference_splats = (radius * 20.0).max(4.0) as usize;
        for j in 0..circumference_splats {
            let angle = (j as f32 / circumference_splats as f32) * std::f32::consts::TAU;
            let x = angle.cos() * radius + (rng.random::<f32>() - 0.5) * 0.02;
            let z = angle.sin() * radius + (rng.random::<f32>() - 0.5) * 0.02;
            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [0.04, 0.06, 0.04],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: trunk_spd,
            });
        }
    }

    // Branches: L-system inspired
    let num_branches = 3 + rng.random_range(0..4u32);
    for b in 0..num_branches {
        let branch_height = height * (0.4 + rng.random::<f32>() * 0.4);
        let branch_angle = (b as f32 / num_branches as f32) * std::f32::consts::TAU
            + rng.random::<f32>() * 0.5;
        let branch_length = canopy_radius * (0.3 + rng.random::<f32>() * 0.5);

        let segments = (branch_length * 8.0) as usize;
        for i in 0..segments {
            let t = i as f32 / segments as f32;
            let x = branch_angle.cos() * t * branch_length;
            let z = branch_angle.sin() * t * branch_length;
            let y = branch_height + t * branch_length * 0.3; // slight upward curve
            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [0.03 * (1.0 - t * 0.7), 0.03, 0.03 * (1.0 - t * 0.7)],
                rotation: [0, 0, 0, 32767],
                opacity: 230,
                _pad: [0; 3],
                spectral: trunk_spd,
            });
        }
    }

    // Canopy: cluster of leaf splats
    let canopy_center_y = height * 0.7;
    let canopy_splats = (canopy_radius * canopy_radius * 100.0) as usize;
    for _ in 0..canopy_splats {
        // Spherical distribution with more density at the top
        let theta = rng.random::<f32>() * std::f32::consts::TAU;
        let phi = rng.random::<f32>() * std::f32::consts::PI * 0.7; // bias upward
        let r = canopy_radius * rng.random::<f32>().sqrt();
        let x = r * phi.sin() * theta.cos();
        let z = r * phi.sin() * theta.sin();
        let y = canopy_center_y + r * phi.cos();

        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [
                0.08 + rng.random::<f32>() * 0.06,
                0.04,
                0.08 + rng.random::<f32>() * 0.06,
            ],
            rotation: [0, 0, 0, 32767],
            opacity: (180.0 + rng.random::<f32>() * 60.0) as u8,
            _pad: [0; 3],
            spectral: leaf_spd,
        });
    }

    splats
}

/// Component assembly for props (benches, lamp posts, etc.).
pub fn generate_bench(seed: u64) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut splats = Vec::new();

    let iron_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.15).to_bits());
    let wood_spd: [u16; 8] = [
        f16::from_f32(0.10).to_bits(),
        f16::from_f32(0.12).to_bits(),
        f16::from_f32(0.15).to_bits(),
        f16::from_f32(0.20).to_bits(),
        f16::from_f32(0.22).to_bits(),
        f16::from_f32(0.20).to_bits(),
        f16::from_f32(0.18).to_bits(),
        f16::from_f32(0.15).to_bits(),
    ];

    let length = 1.8 + rng.random::<f32>() * 0.4;
    let height = 0.45;

    // Iron frame (two side supports)
    for side in [-1.0f32, 1.0] {
        let x = side * length * 0.45;
        for iy in 0..10 {
            let y = iy as f32 / 10.0 * height;
            splats.push(GaussianSplat {
                position: [x, y, 0.0],
                scale: [0.02, 0.03, 0.015],
                rotation: [0, 0, 0, 32767],
                opacity: 250,
                _pad: [0; 3],
                spectral: iron_spd,
            });
        }
    }

    // Wooden slats (seat)
    for slat in 0..5 {
        let z = (slat as f32 - 2.0) * 0.08;
        for ix in 0..20 {
            let x = (ix as f32 / 20.0 - 0.5) * length;
            splats.push(GaussianSplat {
                position: [x, height, z],
                scale: [0.05, 0.01, 0.035],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: wood_spd,
            });
        }
    }

    // Back rest (3 slats)
    for slat in 0..3 {
        let y = height + 0.1 + slat as f32 * 0.08;
        for ix in 0..20 {
            let x = (ix as f32 / 20.0 - 0.5) * length;
            splats.push(GaussianSplat {
                position: [x, y, -0.15],
                scale: [0.05, 0.035, 0.01],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: wood_spd,
            });
        }
    }

    splats
}

/// Surface scattering for terrain patches (grass, gravel, flowers).
pub fn generate_grass_patch(seed: u64, size: f32, density: f32) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let count = (size * size * density) as usize;

    let grass_spd: [u16; 8] = [
        f16::from_f32(0.03).to_bits(),
        f16::from_f32(0.04).to_bits(),
        f16::from_f32(0.06).to_bits(),
        f16::from_f32(0.10).to_bits(),
        f16::from_f32(0.40).to_bits(),
        f16::from_f32(0.25).to_bits(),
        f16::from_f32(0.08).to_bits(),
        f16::from_f32(0.04).to_bits(),
    ];

    (0..count)
        .map(|_| {
            let x = (rng.random::<f32>() - 0.5) * size;
            let z = (rng.random::<f32>() - 0.5) * size;
            let blade_height = 0.05 + rng.random::<f32>() * 0.15;
            GaussianSplat {
                position: [x, blade_height * 0.5, z],
                scale: [0.01, blade_height, 0.01],
                rotation: [0, 0, 0, 32767],
                opacity: (200.0 + rng.random::<f32>() * 40.0) as u8,
                _pad: [0; 3],
                spectral: grass_spd,
            }
        })
        .collect()
}

/// Generate a lamp post.
pub fn generate_lamp_post(_seed: u64, height: f32) -> Vec<GaussianSplat> {
    let mut splats = Vec::new();

    let iron_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.12).to_bits());
    let glass_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.8).to_bits());

    // Pole
    let pole_segments = (height * 8.0) as usize;
    for i in 0..pole_segments {
        let y = i as f32 / pole_segments as f32 * height;
        splats.push(GaussianSplat {
            position: [0.0, y, 0.0],
            scale: [0.03, 0.08, 0.03],
            rotation: [0, 0, 0, 32767],
            opacity: 250,
            _pad: [0; 3],
            spectral: iron_spd,
        });
    }

    // Lamp housing (top)
    for dx in -2..=2 {
        for dz in -2..=2 {
            splats.push(GaussianSplat {
                position: [dx as f32 * 0.04, height, dz as f32 * 0.04],
                scale: [0.05, 0.04, 0.05],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: glass_spd,
            });
        }
    }

    splats
}
