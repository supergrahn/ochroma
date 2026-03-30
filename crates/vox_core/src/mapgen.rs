use crate::types::GaussianSplat;
use half::f16;

/// Simple value noise (not Perlin — just interpolated random values).
fn value_noise(x: f32, z: f32, seed: u64) -> f32 {
    let ix = x.floor() as i64;
    let iz = z.floor() as i64;
    let fx = x - x.floor();
    let fz = z - z.floor();

    let hash = |x: i64, z: i64| -> f32 {
        let n = (x.wrapping_mul(374761393) ^ z.wrapping_mul(668265263) ^ (seed as i64)) as u64;
        let n = n.wrapping_mul(1103515245).wrapping_add(12345);
        (n as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    let v00 = hash(ix, iz);
    let v10 = hash(ix + 1, iz);
    let v01 = hash(ix, iz + 1);
    let v11 = hash(ix + 1, iz + 1);

    let fx = fx * fx * (3.0 - 2.0 * fx); // smoothstep
    let fz = fz * fz * (3.0 - 2.0 * fz);

    let v0 = v00 + (v10 - v00) * fx;
    let v1 = v01 + (v11 - v01) * fx;
    v0 + (v1 - v0) * fz
}

/// Multi-octave noise for terrain.
fn terrain_noise(x: f32, z: f32, seed: u64) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 0.01;

    for octave in 0..4 {
        total += value_noise(x * frequency, z * frequency, seed + octave) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    total
}

/// Generate a procedural terrain map with hills, flat areas, and a river.
pub fn generate_map(seed: u64, size: f32, density: f32) -> Vec<GaussianSplat> {
    let spacing = 1.0 / density.sqrt();
    let nx = (size / spacing).ceil() as i32;
    let nz = (size / spacing).ceil() as i32;
    let half = size * 0.5;
    let mut splats = Vec::with_capacity((nx * nz) as usize);

    // 16-band SPD: 380–755 nm at 25 nm steps (USGS wavelength grid)
    let spd_to_u16 = |vals: [f32; 16]| -> [u16; 16] {
        std::array::from_fn(|i| f16::from_f32(vals[i]).to_bits())
    };
    let grass_spd = spd_to_u16([0.03, 0.03, 0.04, 0.04, 0.05, 0.06, 0.10, 0.38, 0.36, 0.22, 0.14, 0.08, 0.06, 0.42, 0.46, 0.48]);
    let rock_spd  = spd_to_u16([0.12, 0.13, 0.15, 0.17, 0.18, 0.18, 0.17, 0.16, 0.16, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.13]);
    let water_spd = spd_to_u16([0.02, 0.03, 0.05, 0.08, 0.10, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]);
    let sand_spd  = spd_to_u16([0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47]);

    for ix in 0..nx {
        for iz in 0..nz {
            let x = ix as f32 * spacing - half;
            let z = iz as f32 * spacing - half;

            // Height from noise
            let base_height = terrain_noise(x, z, seed) * 15.0;

            // River: carve a channel along a sine curve through the middle
            let river_center_z = (x * 0.02).sin() * 30.0;
            let dist_to_river = (z - river_center_z).abs();
            let river_width = 15.0;
            let river_bank = 25.0;

            let (height, spd) = if dist_to_river < river_width {
                // Water
                (-1.0, water_spd)
            } else if dist_to_river < river_bank {
                // Sandy bank
                let t = (dist_to_river - river_width) / (river_bank - river_width);
                (base_height * t * 0.3, sand_spd)
            } else if base_height > 8.0 {
                // Rocky hills
                (base_height, rock_spd)
            } else {
                // Grass
                (base_height.max(0.0), grass_spd)
            };

            splats.push(GaussianSplat::surface(
                [x, height, z],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                spacing * 0.5,
                spacing * 0.5,
                250,
                spd,
            ));
        }
    }

    splats
}
