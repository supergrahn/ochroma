use vox_core::types::GaussianSplat;
use half::f16;
use rand::prelude::*;
use rand::SeedableRng;

/// A 3D signed distance field for volumetric terrain.
pub struct TerrainVolume {
    /// Dimensions in voxels.
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    /// Voxel size in metres.
    pub voxel_size: f32,
    /// Origin in world space.
    pub origin: [f32; 3],
    /// SDF values: negative = solid, positive = air, zero = surface.
    pub data: Vec<f32>,
    /// Material index per voxel (which spectral material to use).
    pub materials: Vec<u8>,
}

impl TerrainVolume {
    pub fn new(size_x: usize, size_y: usize, size_z: usize, voxel_size: f32) -> Self {
        let count = size_x * size_y * size_z;
        Self {
            size_x, size_y, size_z, voxel_size,
            origin: [
                -(size_x as f32 * voxel_size) / 2.0,
                -(size_y as f32 * voxel_size) / 4.0, // terrain center is lower
                -(size_z as f32 * voxel_size) / 2.0,
            ],
            data: vec![1.0; count], // start as all air
            materials: vec![0; count],
        }
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size_y * self.size_x + y * self.size_x + x
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        if x < self.size_x && y < self.size_y && z < self.size_z {
            self.data[self.index(x, y, z)]
        } else {
            1.0 // outside = air
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f32) {
        if x < self.size_x && y < self.size_y && z < self.size_z {
            let idx = self.index(x, y, z);
            self.data[idx] = value;
        }
    }

    pub fn set_material(&mut self, x: usize, y: usize, z: usize, material: u8) {
        if x < self.size_x && y < self.size_y && z < self.size_z {
            let idx = self.index(x, y, z);
            self.materials[idx] = material;
        }
    }

    pub fn get_material(&self, x: usize, y: usize, z: usize) -> u8 {
        if x < self.size_x && y < self.size_y && z < self.size_z {
            self.materials[self.index(x, y, z)]
        } else {
            0
        }
    }

    /// World-space position of a voxel.
    pub fn voxel_to_world(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        [
            self.origin[0] + x as f32 * self.voxel_size,
            self.origin[1] + y as f32 * self.voxel_size,
            self.origin[2] + z as f32 * self.voxel_size,
        ]
    }

    /// World position to voxel coordinates.
    pub fn world_to_voxel(&self, wx: f32, wy: f32, wz: f32) -> (usize, usize, usize) {
        let x = ((wx - self.origin[0]) / self.voxel_size).max(0.0) as usize;
        let y = ((wy - self.origin[1]) / self.voxel_size).max(0.0) as usize;
        let z = ((wz - self.origin[2]) / self.voxel_size).max(0.0) as usize;
        (x.min(self.size_x - 1), y.min(self.size_y - 1), z.min(self.size_z - 1))
    }

    /// Sample the SDF at a world position (trilinear interpolation).
    pub fn sample_world(&self, wx: f32, wy: f32, wz: f32) -> f32 {
        let lx = (wx - self.origin[0]) / self.voxel_size;
        let ly = (wy - self.origin[1]) / self.voxel_size;
        let lz = (wz - self.origin[2]) / self.voxel_size;

        let ix = lx.floor() as usize;
        let iy = ly.floor() as usize;
        let iz = lz.floor() as usize;
        let fx = lx.fract();
        let fy = ly.fract();
        let fz = lz.fract();

        // Trilinear interpolation
        let c000 = self.get(ix, iy, iz);
        let c100 = self.get(ix + 1, iy, iz);
        let c010 = self.get(ix, iy + 1, iz);
        let c110 = self.get(ix + 1, iy + 1, iz);
        let c001 = self.get(ix, iy, iz + 1);
        let c101 = self.get(ix + 1, iy, iz + 1);
        let c011 = self.get(ix, iy + 1, iz + 1);
        let c111 = self.get(ix + 1, iy + 1, iz + 1);

        let c00 = c000 + (c100 - c000) * fx;
        let c10 = c010 + (c110 - c010) * fx;
        let c01 = c001 + (c101 - c001) * fx;
        let c11 = c011 + (c111 - c011) * fx;

        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;

        c0 + (c1 - c0) * fz
    }

    /// Compute gradient (surface normal) at a point using central differences.
    pub fn gradient(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        let dx = self.get(x.wrapping_add(1).min(self.size_x - 1), y, z)
               - self.get(x.saturating_sub(1), y, z);
        let dy = self.get(x, y.wrapping_add(1).min(self.size_y - 1), z)
               - self.get(x, y.saturating_sub(1), z);
        let dz = self.get(x, y, z.wrapping_add(1).min(self.size_z - 1))
               - self.get(x, y, z.saturating_sub(1));
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-8);
        [dx / len, dy / len, dz / len]
    }

    /// Is this voxel on the surface (SDF crosses zero)?
    pub fn is_surface(&self, x: usize, y: usize, z: usize) -> bool {
        let v = self.get(x, y, z);
        if v > 0.0 { return false; } // in air

        // Check if any neighbour is air
        let neighbours = [
            (x.wrapping_sub(1), y, z), (x + 1, y, z),
            (x, y.wrapping_sub(1), z), (x, y + 1, z),
            (x, y, z.wrapping_sub(1)), (x, y, z + 1),
        ];

        neighbours.iter().any(|&(nx, ny, nz)| self.get(nx, ny, nz) > 0.0)
    }

    /// Count solid voxels.
    pub fn solid_count(&self) -> usize {
        self.data.iter().filter(|&&v| v <= 0.0).count()
    }

    /// Count surface voxels.
    pub fn surface_count(&self) -> usize {
        let mut count = 0;
        for z in 0..self.size_z {
            for y in 0..self.size_y {
                for x in 0..self.size_x {
                    if self.is_surface(x, y, z) { count += 1; }
                }
            }
        }
        count
    }
}

/// Terrain sculpting operations on the SDF.
pub mod sculpt {
    use super::*;

    /// Add a sphere of solid terrain (dig out = positive radius makes solid).
    pub fn add_sphere(vol: &mut TerrainVolume, center: [f32; 3], radius: f32, material: u8) {
        let r_voxels = (radius / vol.voxel_size).ceil() as i32 + 1;
        let (cx, cy, cz) = vol.world_to_voxel(center[0], center[1], center[2]);

        for dz in -r_voxels..=r_voxels {
            for dy in -r_voxels..=r_voxels {
                for dx in -r_voxels..=r_voxels {
                    let x = (cx as i32 + dx).max(0) as usize;
                    let y = (cy as i32 + dy).max(0) as usize;
                    let z = (cz as i32 + dz).max(0) as usize;
                    if x >= vol.size_x || y >= vol.size_y || z >= vol.size_z { continue; }

                    let wp = vol.voxel_to_world(x, y, z);
                    let dist = ((wp[0] - center[0]).powi(2) + (wp[1] - center[1]).powi(2) + (wp[2] - center[2]).powi(2)).sqrt();
                    let sdf = dist - radius;

                    // Smooth min: blend with existing value
                    let current = vol.get(x, y, z);
                    vol.set(x, y, z, current.min(sdf));
                    if sdf <= 0.0 {
                        vol.set_material(x, y, z, material);
                    }
                }
            }
        }
    }

    /// Remove a sphere (carve a cave/hole).
    pub fn remove_sphere(vol: &mut TerrainVolume, center: [f32; 3], radius: f32) {
        let r_voxels = (radius / vol.voxel_size).ceil() as i32 + 1;
        let (cx, cy, cz) = vol.world_to_voxel(center[0], center[1], center[2]);

        for dz in -r_voxels..=r_voxels {
            for dy in -r_voxels..=r_voxels {
                for dx in -r_voxels..=r_voxels {
                    let x = (cx as i32 + dx).max(0) as usize;
                    let y = (cy as i32 + dy).max(0) as usize;
                    let z = (cz as i32 + dz).max(0) as usize;
                    if x >= vol.size_x || y >= vol.size_y || z >= vol.size_z { continue; }

                    let wp = vol.voxel_to_world(x, y, z);
                    let dist = ((wp[0] - center[0]).powi(2) + (wp[1] - center[1]).powi(2) + (wp[2] - center[2]).powi(2)).sqrt();
                    let sdf = -(dist - radius); // inverted: inside sphere becomes air

                    let current = vol.get(x, y, z);
                    vol.set(x, y, z, current.max(sdf));
                }
            }
        }
    }

    /// Create a flat ground plane at a given height.
    pub fn add_ground_plane(vol: &mut TerrainVolume, height: f32, material: u8) {
        for z in 0..vol.size_z {
            for y in 0..vol.size_y {
                for x in 0..vol.size_x {
                    let wp = vol.voxel_to_world(x, y, z);
                    let sdf = wp[1] - height; // below height = solid
                    let current = vol.get(x, y, z);
                    vol.set(x, y, z, current.min(sdf));
                    if sdf <= 0.0 {
                        vol.set_material(x, y, z, material);
                    }
                }
            }
        }
    }

    /// Add an overhanging cliff.
    pub fn add_cliff(vol: &mut TerrainVolume, base: [f32; 3], height: f32, overhang: f32, width: f32, material: u8) {
        // Cliff is a box that curves outward at the top
        let r = (height.max(width).max(overhang) / vol.voxel_size).ceil() as i32 + 2;
        let (cx, cy, cz) = vol.world_to_voxel(base[0], base[1], base[2]);

        for dz in -r..=r {
            for dy in 0..=(height / vol.voxel_size).ceil() as i32 {
                for dx in -r..=r {
                    let x = (cx as i32 + dx).max(0) as usize;
                    let y = (cy as i32 + dy).max(0) as usize;
                    let z = (cz as i32 + dz).max(0) as usize;
                    if x >= vol.size_x || y >= vol.size_y || z >= vol.size_z { continue; }

                    let wp = vol.voxel_to_world(x, y, z);
                    let local_y = (wp[1] - base[1]) / height; // 0 at base, 1 at top
                    let local_y = local_y.clamp(0.0, 1.0);

                    // Overhang: the solid region extends further in X at the top
                    let local_overhang = overhang * local_y * local_y; // quadratic overhang
                    let effective_x = wp[0] - base[0] - local_overhang;
                    let effective_z = wp[2] - base[2];

                    // Box SDF with the overhang offset
                    let half_w = width * 0.5;
                    let dx_dist = effective_x.abs() - half_w;
                    let dz_dist = effective_z.abs() - half_w;
                    let dy_dist = wp[1] - (base[1] + height);

                    let sdf = dx_dist.max(dz_dist).max(dy_dist);

                    let current = vol.get(x, y, z);
                    vol.set(x, y, z, current.min(sdf));
                    if sdf <= 0.0 {
                        vol.set_material(x, y, z, material);
                    }
                }
            }
        }
    }

    /// Add a cave tunnel between two points.
    pub fn add_cave(vol: &mut TerrainVolume, start: [f32; 3], end: [f32; 3], radius: f32) {
        let dir = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
        let length = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if length < 0.01 { return; }

        let steps = (length / (vol.voxel_size * 0.5)).ceil() as usize;
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let point = [
                start[0] + dir[0] * t,
                start[1] + dir[1] * t,
                start[2] + dir[2] * t,
            ];
            remove_sphere(vol, point, radius);
        }
    }

    /// Add a natural arch.
    pub fn add_arch(vol: &mut TerrainVolume, center: [f32; 3], span: f32, height: f32, thickness: f32, material: u8) {
        let steps = 20;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let angle = t * std::f32::consts::PI;
            let x = center[0] + angle.cos() * span * 0.5;
            let y = center[1] + angle.sin() * height;
            let z = center[2];
            add_sphere(vol, [x, y, z], thickness, material);
        }
    }
}

/// Material spectral properties for terrain volumes.
pub struct VolumeMaterial {
    pub id: u8,
    pub name: String,
    pub spectral: [f32; 8],
}

pub fn default_volume_materials() -> Vec<VolumeMaterial> {
    vec![
        VolumeMaterial { id: 0, name: "rock".into(), spectral: [0.12, 0.13, 0.15, 0.17, 0.18, 0.18, 0.17, 0.16] },
        VolumeMaterial { id: 1, name: "grass".into(), spectral: [0.03, 0.04, 0.06, 0.10, 0.40, 0.25, 0.08, 0.04] },
        VolumeMaterial { id: 2, name: "dirt".into(), spectral: [0.10, 0.12, 0.15, 0.20, 0.22, 0.20, 0.18, 0.15] },
        VolumeMaterial { id: 3, name: "sand".into(), spectral: [0.20, 0.22, 0.25, 0.30, 0.35, 0.38, 0.36, 0.32] },
        VolumeMaterial { id: 4, name: "snow".into(), spectral: [0.85, 0.87, 0.89, 0.90, 0.90, 0.89, 0.87, 0.85] },
        VolumeMaterial { id: 5, name: "clay".into(), spectral: [0.15, 0.16, 0.18, 0.22, 0.28, 0.35, 0.32, 0.28] },
        VolumeMaterial { id: 6, name: "moss".into(), spectral: [0.02, 0.03, 0.05, 0.08, 0.30, 0.18, 0.06, 0.03] },
        VolumeMaterial { id: 7, name: "ice".into(), spectral: [0.70, 0.75, 0.80, 0.82, 0.80, 0.75, 0.70, 0.65] },
    ]
}

/// Convert surface voxels of a TerrainVolume to Gaussian splats.
pub fn volume_to_splats(vol: &TerrainVolume, materials: &[VolumeMaterial], seed: u64) -> Vec<GaussianSplat> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut splats = Vec::new();

    for z in 1..vol.size_z - 1 {
        for y in 1..vol.size_y - 1 {
            for x in 1..vol.size_x - 1 {
                if !vol.is_surface(x, y, z) { continue; }

                let wp = vol.voxel_to_world(x, y, z);
                let _normal = vol.gradient(x, y, z);
                let mat_id = vol.get_material(x, y, z);

                let mat = materials.iter().find(|m| m.id == mat_id);
                let spectral: [u16; 8] = match mat {
                    Some(m) => std::array::from_fn(|i| f16::from_f32(m.spectral[i]).to_bits()),
                    None => std::array::from_fn(|_| f16::from_f32(0.3).to_bits()),
                };

                // Slight random offset for organic look
                let jitter = vol.voxel_size * 0.2;
                let jx = (rng.random::<f32>() - 0.5) * jitter;
                let jy = (rng.random::<f32>() - 0.5) * jitter;
                let jz = (rng.random::<f32>() - 0.5) * jitter;

                // Orient splat along surface normal
                let scale = vol.voxel_size * 0.5;

                splats.push(GaussianSplat {
                    position: [wp[0] + jx, wp[1] + jy, wp[2] + jz],
                    scale: [scale, scale * 0.3, scale], // flatten along normal direction
                    rotation: [0, 0, 0, 32767], // simplified -- proper orientation would use normal
                    opacity: 245,
                    _pad: [0; 3],
                    spectral,
                });
            }
        }
    }

    splats
}

/// Generate a demo terrain volume with ground, hills, a cliff, a cave, and an arch.
pub fn generate_demo_volume(_seed: u64) -> TerrainVolume {
    let mut vol = TerrainVolume::new(64, 32, 64, 1.0);

    // Ground plane
    sculpt::add_ground_plane(&mut vol, 0.0, 1); // grass

    // Hills
    sculpt::add_sphere(&mut vol, [10.0, -2.0, 10.0], 8.0, 1);  // grass hill
    sculpt::add_sphere(&mut vol, [-15.0, -3.0, -10.0], 10.0, 0); // rock hill
    sculpt::add_sphere(&mut vol, [20.0, -1.0, -15.0], 6.0, 2);  // dirt mound

    // Overhanging cliff!
    sculpt::add_cliff(&mut vol, [-10.0, 0.0, 0.0], 12.0, 5.0, 6.0, 0);

    // Cave through the rock hill
    sculpt::add_cave(&mut vol, [-20.0, 2.0, -10.0], [-10.0, 2.0, -10.0], 3.0);

    // Natural arch
    sculpt::add_arch(&mut vol, [0.0, 0.0, -20.0], 10.0, 8.0, 2.0, 0);

    vol
}
