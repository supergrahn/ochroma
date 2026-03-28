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

// ---------------------------------------------------------------------------
// Detailed building generation with visible architectural elements
// ---------------------------------------------------------------------------

/// Return (wall_spd, window_spd, door_spd, roof_spd) for a named building style.
fn style_spds(style: &str) -> ([u16; 8], [u16; 8], [u16; 8], [u16; 8]) {
    match style {
        "victorian" => (
            // Red brick wall
            [
                f16::from_f32(0.08).to_bits(),
                f16::from_f32(0.08).to_bits(),
                f16::from_f32(0.10).to_bits(),
                f16::from_f32(0.14).to_bits(),
                f16::from_f32(0.22).to_bits(),
                f16::from_f32(0.52).to_bits(),
                f16::from_f32(0.62).to_bits(),
                f16::from_f32(0.55).to_bits(),
            ],
            // Clear glass window (high transmittance across visible spectrum)
            [
                f16::from_f32(0.55).to_bits(),
                f16::from_f32(0.60).to_bits(),
                f16::from_f32(0.65).to_bits(),
                f16::from_f32(0.70).to_bits(),
                f16::from_f32(0.72).to_bits(),
                f16::from_f32(0.70).to_bits(),
                f16::from_f32(0.65).to_bits(),
                f16::from_f32(0.58).to_bits(),
            ],
            // Oak door (warm brown)
            [
                f16::from_f32(0.06).to_bits(),
                f16::from_f32(0.07).to_bits(),
                f16::from_f32(0.10).to_bits(),
                f16::from_f32(0.16).to_bits(),
                f16::from_f32(0.24).to_bits(),
                f16::from_f32(0.30).to_bits(),
                f16::from_f32(0.28).to_bits(),
                f16::from_f32(0.22).to_bits(),
            ],
            // Slate roof (dark grey)
            [
                f16::from_f32(0.08).to_bits(),
                f16::from_f32(0.09).to_bits(),
                f16::from_f32(0.10).to_bits(),
                f16::from_f32(0.11).to_bits(),
                f16::from_f32(0.12).to_bits(),
                f16::from_f32(0.12).to_bits(),
                f16::from_f32(0.11).to_bits(),
                f16::from_f32(0.10).to_bits(),
            ],
        ),
        "modern" => (
            // Concrete wall (neutral grey)
            [
                f16::from_f32(0.25).to_bits(),
                f16::from_f32(0.26).to_bits(),
                f16::from_f32(0.27).to_bits(),
                f16::from_f32(0.28).to_bits(),
                f16::from_f32(0.28).to_bits(),
                f16::from_f32(0.27).to_bits(),
                f16::from_f32(0.26).to_bits(),
                f16::from_f32(0.25).to_bits(),
            ],
            // Tinted glass (blue-green tint)
            [
                f16::from_f32(0.15).to_bits(),
                f16::from_f32(0.20).to_bits(),
                f16::from_f32(0.35).to_bits(),
                f16::from_f32(0.45).to_bits(),
                f16::from_f32(0.50).to_bits(),
                f16::from_f32(0.40).to_bits(),
                f16::from_f32(0.25).to_bits(),
                f16::from_f32(0.15).to_bits(),
            ],
            // Metal door (dark metallic)
            [
                f16::from_f32(0.12).to_bits(),
                f16::from_f32(0.13).to_bits(),
                f16::from_f32(0.14).to_bits(),
                f16::from_f32(0.15).to_bits(),
                f16::from_f32(0.16).to_bits(),
                f16::from_f32(0.16).to_bits(),
                f16::from_f32(0.15).to_bits(),
                f16::from_f32(0.14).to_bits(),
            ],
            // Flat dark roof
            [
                f16::from_f32(0.05).to_bits(),
                f16::from_f32(0.06).to_bits(),
                f16::from_f32(0.06).to_bits(),
                f16::from_f32(0.07).to_bits(),
                f16::from_f32(0.07).to_bits(),
                f16::from_f32(0.07).to_bits(),
                f16::from_f32(0.06).to_bits(),
                f16::from_f32(0.06).to_bits(),
            ],
        ),
        _ => (
            // Painted wood wall (off-white/cream)
            [
                f16::from_f32(0.40).to_bits(),
                f16::from_f32(0.42).to_bits(),
                f16::from_f32(0.45).to_bits(),
                f16::from_f32(0.50).to_bits(),
                f16::from_f32(0.55).to_bits(),
                f16::from_f32(0.58).to_bits(),
                f16::from_f32(0.56).to_bits(),
                f16::from_f32(0.50).to_bits(),
            ],
            // Clear glass window
            [
                f16::from_f32(0.50).to_bits(),
                f16::from_f32(0.55).to_bits(),
                f16::from_f32(0.60).to_bits(),
                f16::from_f32(0.65).to_bits(),
                f16::from_f32(0.68).to_bits(),
                f16::from_f32(0.65).to_bits(),
                f16::from_f32(0.60).to_bits(),
                f16::from_f32(0.55).to_bits(),
            ],
            // Wood door (warm brown)
            [
                f16::from_f32(0.08).to_bits(),
                f16::from_f32(0.09).to_bits(),
                f16::from_f32(0.12).to_bits(),
                f16::from_f32(0.18).to_bits(),
                f16::from_f32(0.26).to_bits(),
                f16::from_f32(0.32).to_bits(),
                f16::from_f32(0.30).to_bits(),
                f16::from_f32(0.24).to_bits(),
            ],
            // Terracotta roof (warm orange-brown)
            [
                f16::from_f32(0.08).to_bits(),
                f16::from_f32(0.09).to_bits(),
                f16::from_f32(0.12).to_bits(),
                f16::from_f32(0.18).to_bits(),
                f16::from_f32(0.28).to_bits(),
                f16::from_f32(0.42).to_bits(),
                f16::from_f32(0.48).to_bits(),
                f16::from_f32(0.40).to_bits(),
            ],
        ),
    }
}

/// Encode a rotation from a normal vector as i16 quaternion [x, y, z, w].
fn normal_to_rotation_i16(normal: [f32; 3]) -> [i16; 4] {
    // Align the splat's local Z with the surface normal.
    // Default orientation is Z = [0, 0, 1], so find quaternion from [0,0,1] to normal.
    let from = [0.0f32, 0.0, 1.0];
    let dot = from[0] * normal[0] + from[1] * normal[1] + from[2] * normal[2];

    if dot > 0.9999 {
        // Already aligned
        return [0, 0, 0, 32767];
    }
    if dot < -0.9999 {
        // Opposite: 180-degree rotation around Y
        return [0, 32767, 0, 0];
    }

    // Cross product
    let cx = from[1] * normal[2] - from[2] * normal[1];
    let cy = from[2] * normal[0] - from[0] * normal[2];
    let cz = from[0] * normal[1] - from[1] * normal[0];
    let w = 1.0 + dot;
    let len = (cx * cx + cy * cy + cz * cz + w * w).sqrt();
    let inv = 32767.0 / len;

    [
        (cx * inv) as i16,
        (cy * inv) as i16,
        (cz * inv) as i16,
        (w * inv) as i16,
    ]
}

/// Scatter splats on a wall surface in the XY plane at fixed Z.
fn scatter_wall_surface(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    min: [f32; 3],
    max: [f32; 3],
    spd: [u16; 8],
    density_per_sqm: f32,
    normal: [f32; 3],
) {
    let w = (max[0] - min[0]).abs();
    let h = (max[1] - min[1]).abs();
    let area = w * h;
    let count = (area * density_per_sqm).max(1.0) as usize;
    let rot = normal_to_rotation_i16(normal);

    for _ in 0..count {
        let x = min[0] + rng.random::<f32>() * (max[0] - min[0]);
        let y = min[1] + rng.random::<f32>() * (max[1] - min[1]);
        // Z is same for both min and max (planar)
        let z = min[2];

        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [0.05, 0.05, 0.015], // flat against wall
            rotation: rot,
            opacity: 240,
            _pad: [0; 3],
            spectral: spd,
        });
    }
}

/// Scatter splats on a wall surface in the YZ plane at fixed X.
fn scatter_wall_surface_xz(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    min: [f32; 3],
    max: [f32; 3],
    spd: [u16; 8],
    density_per_sqm: f32,
    normal: [f32; 3],
) {
    let d = (max[2] - min[2]).abs();
    let h = (max[1] - min[1]).abs();
    let area = d * h;
    let count = (area * density_per_sqm).max(1.0) as usize;
    let rot = normal_to_rotation_i16(normal);
    let x = min[0]; // fixed X for side walls

    for _ in 0..count {
        let z = min[2] + rng.random::<f32>() * (max[2] - min[2]);
        let y = min[1] + rng.random::<f32>() * (max[1] - min[1]);

        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [0.015, 0.05, 0.05], // flat against side wall
            rotation: rot,
            opacity: 240,
            _pad: [0; 3],
            spectral: spd,
        });
    }
}

/// Place a rectangular window as a cluster of glass-SPD splats.
fn place_window(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    center: [f32; 3],
    w: f32,
    h: f32,
    spd: [u16; 8],
) {
    let cols = (w / 0.12).ceil() as usize;
    let rows = (h / 0.12).ceil() as usize;

    for iy in 0..rows {
        for ix in 0..cols {
            let x = center[0] - w * 0.5 + (ix as f32 + 0.5) * w / cols as f32
                + (rng.random::<f32>() - 0.5) * 0.02;
            let y = center[1] + (iy as f32 + 0.5) * h / rows as f32
                + (rng.random::<f32>() - 0.5) * 0.02;
            let z = center[2];

            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [0.06, 0.06, 0.01], // very flat glass pane
                rotation: [0, 0, 0, 32767],
                opacity: 200, // slightly transparent
                _pad: [0; 3],
                spectral: spd,
            });
        }
    }
}

/// Place a door as a cluster of splats at ground level.
fn place_door(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    center: [f32; 3],
    w: f32,
    h: f32,
    spd: [u16; 8],
) {
    let cols = (w / 0.10).ceil() as usize;
    let rows = (h / 0.10).ceil() as usize;

    for iy in 0..rows {
        for ix in 0..cols {
            let x = center[0] - w * 0.5 + (ix as f32 + 0.5) * w / cols as f32
                + (rng.random::<f32>() - 0.5) * 0.01;
            let y = center[1] + (iy as f32 + 0.5) * h / rows as f32;
            let z = center[2];

            splats.push(GaussianSplat {
                position: [x, y, z],
                scale: [0.05, 0.05, 0.02],
                rotation: [0, 0, 0, 32767],
                opacity: 250,
                _pad: [0; 3],
                spectral: spd,
            });
        }
    }
}

/// Scatter roof splats over a flat (or slightly pitched) rectangular area.
fn scatter_roof(
    splats: &mut Vec<GaussianSplat>,
    rng: &mut StdRng,
    min: [f32; 3],
    max: [f32; 3],
    spd: [u16; 8],
    density_per_sqm: f32,
) {
    let w = (max[0] - min[0]).abs();
    let d = (max[2] - min[2]).abs();
    let area = w * d;
    let count = (area * density_per_sqm).max(1.0) as usize;

    for _ in 0..count {
        let x = min[0] + rng.random::<f32>() * (max[0] - min[0]);
        let z = min[2] + rng.random::<f32>() * (max[2] - min[2]);
        // Slight pitch: y varies slightly with z
        let pitch_offset = (z - min[2]) / d.max(0.001) * (max[1] - min[1]);
        let y = min[1] + pitch_offset + (rng.random::<f32>() - 0.5) * 0.02;

        splats.push(GaussianSplat {
            position: [x, y, z],
            scale: [0.06, 0.015, 0.06], // flat horizontal
            rotation: [0, 0, 0, 32767],
            opacity: 245,
            _pad: [0; 3],
            spectral: spd,
        });
    }
}

/// Generate a detailed building with visible windows, doors, walls, and roof.
///
/// Each architectural element uses distinct spectral properties so they
/// render as visually different materials (brick vs. glass vs. wood, etc.).
pub fn generate_detailed_building(
    seed: u64,
    width: f32,
    depth: f32,
    floors: u32,
    style: &str,
) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut splats = Vec::new();
    let floor_height = 3.2;
    let total_height = floors as f32 * floor_height;

    let (wall_spd, window_spd, door_spd, roof_spd) = style_spds(style);

    // === WALLS: Dense surface scattering with consistent normals ===

    // Front wall (+Z face)
    scatter_wall_surface(
        &mut splats, &mut rng,
        [0.0, 0.0, 0.0], [width, total_height, 0.0],
        wall_spd, 200.0, [0.0, 0.0, 1.0],
    );

    // Back wall (-Z face)
    scatter_wall_surface(
        &mut splats, &mut rng,
        [0.0, 0.0, -depth], [width, total_height, -depth],
        wall_spd, 200.0, [0.0, 0.0, -1.0],
    );

    // Left wall (-X face)
    scatter_wall_surface_xz(
        &mut splats, &mut rng,
        [0.0, 0.0, -depth], [0.0, total_height, 0.0],
        wall_spd, 200.0, [-1.0, 0.0, 0.0],
    );

    // Right wall (+X face)
    scatter_wall_surface_xz(
        &mut splats, &mut rng,
        [width, 0.0, -depth], [width, total_height, 0.0],
        wall_spd, 200.0, [1.0, 0.0, 0.0],
    );

    // === WINDOWS: Rectangular clusters at regular positions ===
    let window_width = 0.8;
    let window_height = 1.2;
    let window_spacing = width / (width / 2.5).floor().max(1.0);

    for floor in 0..floors {
        let base_y = floor as f32 * floor_height + floor_height * 0.4;
        let num_windows = (width / window_spacing).floor() as i32;

        for w_idx in 0..num_windows {
            let cx = (w_idx as f32 + 0.5) * window_spacing;
            // Skip if too close to edges
            if cx < 1.0 || cx > width - 1.0 {
                continue;
            }

            // Front windows
            place_window(
                &mut splats, &mut rng,
                [cx, base_y, 0.01],
                window_width, window_height,
                window_spd,
            );

            // Back windows (some floors, randomly)
            if rng.random::<f32>() > 0.3 {
                place_window(
                    &mut splats, &mut rng,
                    [cx, base_y, -depth - 0.01],
                    window_width, window_height,
                    window_spd,
                );
            }
        }
    }

    // === DOOR: Ground floor, center of front wall ===
    let door_cx = width * 0.5;
    let door_width = 1.0;
    let door_height = 2.2;
    place_door(
        &mut splats, &mut rng,
        [door_cx, 0.0, 0.02],
        door_width, door_height,
        door_spd,
    );

    // === ROOF: Flat with slight pitch ===
    scatter_roof(
        &mut splats, &mut rng,
        [0.0, total_height, -depth],
        [width, total_height + 0.3, 0.0],
        roof_spd, 150.0,
    );

    splats
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
