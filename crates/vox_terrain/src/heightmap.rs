use half::f16;
use vox_core::types::GaussianSplat;

/// A heightmap loaded from data (not generated procedurally).
pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,    // height values, row-major
    pub cell_size: f32,    // metres per cell
    pub origin: [f32; 2],  // world-space origin [x, z]
}

/// Material zones by height.
#[derive(Debug, Clone)]
pub struct TerrainMaterialZone {
    pub max_height: f32,       // up to this height, use this material
    pub surface_type: String,  // material name
    pub spectral: [f32; 8],    // SPD values
}

impl Heightmap {
    /// Create from a flat array of heights.
    pub fn from_data(width: usize, height: usize, data: Vec<f32>, cell_size: f32) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            width,
            height,
            data,
            cell_size,
            origin: [0.0, 0.0],
        }
    }

    /// Create a flat terrain at a given height.
    pub fn flat(width: usize, height: usize, cell_size: f32, terrain_height: f32) -> Self {
        Self::from_data(width, height, vec![terrain_height; width * height], cell_size)
    }

    /// Sample height at world position (bilinear interpolation).
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        let local_x = (world_x - self.origin[0]) / self.cell_size;
        let local_z = (world_z - self.origin[1]) / self.cell_size;

        let ix = local_x.floor() as i32;
        let iz = local_z.floor() as i32;
        let fx = local_x - local_x.floor();
        let fz = local_z - local_z.floor();

        let h = |x: i32, z: i32| -> f32 {
            let x = x.clamp(0, self.width as i32 - 1) as usize;
            let z = z.clamp(0, self.height as i32 - 1) as usize;
            self.data[z * self.width + x]
        };

        let h00 = h(ix, iz);
        let h10 = h(ix + 1, iz);
        let h01 = h(ix, iz + 1);
        let h11 = h(ix + 1, iz + 1);

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    /// Compute surface normal at a point (from surrounding heights).
    pub fn normal_at(&self, world_x: f32, world_z: f32) -> [f32; 3] {
        let dx = self.sample(world_x + self.cell_size, world_z)
            - self.sample(world_x - self.cell_size, world_z);
        let dz = self.sample(world_x, world_z + self.cell_size)
            - self.sample(world_x, world_z - self.cell_size);
        let nx = -dx;
        let ny = 2.0 * self.cell_size;
        let nz = -dz;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / len, ny / len, nz / len]
    }

    /// Get slope angle in degrees at a point.
    pub fn slope_at(&self, world_x: f32, world_z: f32) -> f32 {
        let n = self.normal_at(world_x, world_z);
        n[1].acos().to_degrees() // angle from vertical
    }

    /// World-space bounds.
    pub fn bounds(&self) -> ([f32; 2], [f32; 2]) {
        let min = self.origin;
        let max = [
            self.origin[0] + self.width as f32 * self.cell_size,
            self.origin[1] + self.height as f32 * self.cell_size,
        ];
        (min, max)
    }

    /// Total world area in square metres.
    pub fn area(&self) -> f32 {
        (self.width as f32 * self.cell_size) * (self.height as f32 * self.cell_size)
    }

    /// Generate Gaussian splats for this terrain with material zones.
    pub fn to_splats(
        &self,
        zones: &[TerrainMaterialZone],
        splats_per_cell: u32,
    ) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        let sub = splats_per_cell.max(1);

        for iz in 0..self.height {
            for ix in 0..self.width {
                let base_x = self.origin[0] + ix as f32 * self.cell_size;
                let base_z = self.origin[1] + iz as f32 * self.cell_size;

                for si in 0..sub {
                    for sj in 0..sub {
                        let frac_x = (si as f32 + 0.5) / sub as f32;
                        let frac_z = (sj as f32 + 0.5) / sub as f32;
                        let wx = base_x + frac_x * self.cell_size;
                        let wz = base_z + frac_z * self.cell_size;
                        let wy = self.sample(wx, wz);

                        // Pick material based on height
                        let zone = zones
                            .iter()
                            .find(|z| wy <= z.max_height)
                            .or_else(|| zones.last());

                        let spectral: [u16; 8] = match zone {
                            Some(z) => {
                                std::array::from_fn(|i| f16::from_f32(z.spectral[i]).to_bits())
                            }
                            None => std::array::from_fn(|_| f16::from_f32(0.3).to_bits()),
                        };

                        let scale = self.cell_size / sub as f32 * 0.5;
                        splats.push(GaussianSplat {
                            position: [wx, wy, wz],
                            scale: [scale, 0.02, scale],
                            rotation: [0, 0, 0, 32767],
                            opacity: 250,
                            _pad: [0; 3],
                            spectral,
                        });
                    }
                }
            }
        }

        splats
    }
}

/// Default terrain material zones.
pub fn default_zones() -> Vec<TerrainMaterialZone> {
    vec![
        TerrainMaterialZone {
            max_height: -0.5,
            surface_type: "water".into(),
            spectral: [0.01, 0.03, 0.08, 0.12, 0.10, 0.06, 0.03, 0.01],
        },
        TerrainMaterialZone {
            max_height: 0.5,
            surface_type: "sand".into(),
            spectral: [0.20, 0.22, 0.25, 0.30, 0.35, 0.38, 0.36, 0.32],
        },
        TerrainMaterialZone {
            max_height: 8.0,
            surface_type: "grass".into(),
            spectral: [0.03, 0.04, 0.06, 0.10, 0.40, 0.25, 0.08, 0.04],
        },
        TerrainMaterialZone {
            max_height: 15.0,
            surface_type: "rock".into(),
            spectral: [0.12, 0.13, 0.15, 0.17, 0.18, 0.18, 0.17, 0.16],
        },
        TerrainMaterialZone {
            max_height: f32::MAX,
            surface_type: "snow".into(),
            spectral: [0.85, 0.87, 0.89, 0.90, 0.90, 0.89, 0.87, 0.85],
        },
    ]
}

/// Load a heightmap from raw bytes (f32 little-endian grid).
pub fn load_heightmap_raw(
    data: &[u8],
    width: usize,
    height: usize,
    cell_size: f32,
) -> Result<Heightmap, String> {
    let expected = width * height * 4;
    if data.len() != expected {
        return Err(format!(
            "Expected {} bytes for {}x{} f32 grid, got {}",
            expected,
            width,
            height,
            data.len()
        ));
    }

    let heights: Vec<f32> = data
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(Heightmap::from_data(width, height, heights, cell_size))
}

/// Create a heightmap from a simple noise function (for testing/demos).
pub fn generate_test_heightmap(
    width: usize,
    height: usize,
    cell_size: f32,
    seed: u64,
) -> Heightmap {
    let mut data = vec![0.0f32; width * height];

    for z in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fz = z as f32 / height as f32;

            // Simple multi-octave noise
            let h1 = ((fx * 3.0 + seed as f32 * 0.1).sin() * (fz * 4.0).cos()) * 5.0;
            let h2 = ((fx * 7.0 + 1.0).sin() * (fz * 6.0 + 2.0).cos()) * 2.0;
            let h3 = ((fx * 13.0).sin() * (fz * 11.0).cos()) * 1.0;

            // River valley through the middle
            let dist_to_center = ((fz - 0.5).abs() * 2.0).min(1.0);
            let valley = (1.0 - (dist_to_center * dist_to_center)) * -3.0;

            data[z * width + x] = h1 + h2 + h3 + valley;
        }
    }

    Heightmap::from_data(width, height, data, cell_size)
}
