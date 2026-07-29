use half::f16;
use vox_core::types::GaussianSplat;

/// A heightmap loaded from data (not generated procedurally).
pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,   // height values, row-major
    pub cell_size: f32,   // metres per cell
    pub origin: [f32; 2], // world-space origin [x, z]
}

/// Material zones by height.
#[derive(Debug, Clone)]
pub struct TerrainMaterialZone {
    pub max_height: f32,      // up to this height, use this material
    pub surface_type: String, // material name
    pub spectral: [f32; 8],   // SPD values
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
        Self::from_data(
            width,
            height,
            vec![terrain_height; width * height],
            cell_size,
        )
    }

    /// Bilinear height sample at a *local* (cell-space) coordinate. Shared core
    /// of [`sample`](Heightmap::sample) so callers that already have the
    /// local coordinate (the normal/slope taps) don't repeat the
    /// `origin`/`cell_size` divide. Arithmetic is identical to the world-space
    /// path, so results are bit-for-bit unchanged.
    #[inline(always)]
    fn sample_local(&self, local_x: f32, local_z: f32) -> f32 {
        let ix = local_x.floor() as i32;
        let iz = local_z.floor() as i32;
        let fx = local_x - local_x.floor();
        let fz = local_z - local_z.floor();

        let w = self.width as i32;
        let h_clamp = self.height as i32;
        let h = |x: i32, z: i32| -> f32 {
            let x = x.clamp(0, w - 1) as usize;
            let z = z.clamp(0, h_clamp - 1) as usize;
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

    /// Sample height at world position (bilinear interpolation).
    #[inline]
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        let local_x = (world_x - self.origin[0]) / self.cell_size;
        let local_z = (world_z - self.origin[1]) / self.cell_size;
        self.sample_local(local_x, local_z)
    }

    /// Sample the exact piecewise-planar surface produced by the canonical
    /// heightmap triangle topology.
    ///
    /// [`sample`](Self::sample) is intentionally bilinear and remains the
    /// simulation/terrain-authoring contract. A rendered heightfield is a
    /// triangle mesh, however: each cell is split from its top-right corner to
    /// its bottom-left corner. On a saddle cell that planar surface can differ
    /// materially from the bilinear patch. Geometry that must sit on the
    /// *visible* ground (roads, foundations, props) uses this sampler so the
    /// terrain cannot pass through it between heightmap vertices.
    #[inline]
    pub fn sample_triangulated(&self, world_x: f32, world_z: f32) -> f32 {
        if self.width < 2 || self.height < 2 {
            return self.sample(world_x, world_z);
        }

        let local_x = ((world_x - self.origin[0]) / self.cell_size)
            .clamp(0.0, (self.width - 1) as f32);
        let local_z = ((world_z - self.origin[1]) / self.cell_size)
            .clamp(0.0, (self.height - 1) as f32);
        let ix = (local_x.floor() as usize).min(self.width - 2);
        let iz = (local_z.floor() as usize).min(self.height - 2);
        let fx = local_x - ix as f32;
        let fz = local_z - iz as f32;

        let h00 = self.data[iz * self.width + ix];
        let h10 = self.data[iz * self.width + ix + 1];
        let h01 = self.data[(iz + 1) * self.width + ix];
        let h11 = self.data[(iz + 1) * self.width + ix + 1];

        if fx + fz <= 1.0 {
            // Canonical first triangle: top-left, bottom-left, top-right.
            h00 + (h10 - h00) * fx + (h01 - h00) * fz
        } else {
            // Canonical second triangle: top-right, bottom-left, bottom-right.
            h11 + (h10 - h11) * (1.0 - fz) + (h01 - h11) * (1.0 - fx)
        }
    }

    /// Compute surface normal at a point (from surrounding heights).
    pub fn normal_at(&self, world_x: f32, world_z: f32) -> [f32; 3] {
        // Four taps one cell apart. Each local coordinate is computed with the
        // SAME `(world ± cell_size - origin) / cell_size` expression the old
        // four `self.sample(...)` calls used, so the float result is
        // bit-identical; `sample_local` only shares the clamp/index setup.
        let cs = self.cell_size;
        let lxp = (world_x + cs - self.origin[0]) / cs;
        let lxm = (world_x - cs - self.origin[0]) / cs;
        let lz0 = (world_z - self.origin[1]) / cs;
        let lx0 = (world_x - self.origin[0]) / cs;
        let lzp = (world_z + cs - self.origin[1]) / cs;
        let lzm = (world_z - cs - self.origin[1]) / cs;
        let dx = self.sample_local(lxp, lz0) - self.sample_local(lxm, lz0);
        let dz = self.sample_local(lx0, lzp) - self.sample_local(lx0, lzm);
        let nx = -dx;
        let ny = 2.0 * cs;
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
        use rayon::prelude::*;
        let sub = splats_per_cell.max(1);
        let per_row = self.width * (sub * sub) as usize;

        // Rows are independent and each emits a fixed `per_row` block of splats
        // in ix-major / (si,sj) order. Building rows in parallel and flattening
        // in `iz` order reproduces the exact serial ix-major ordering, and each
        // splat's arithmetic (sample, zone find, f16 bits) is unchanged ->
        // byte-identical output (deterministic by construction: no shared
        // state, fixed concat order).
        (0..self.height)
            .into_par_iter()
            .flat_map_iter(|iz| {
                let mut row = Vec::with_capacity(per_row);
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

                            let spectral: [u16; 16] = match zone {
                                Some(z) => std::array::from_fn(|i| {
                                    f16::from_f32(z.spectral[i % 8]).to_bits()
                                }),
                                None => std::array::from_fn(|_| f16::from_f32(0.3).to_bits()),
                            };

                            let scale = self.cell_size / sub as f32 * 0.5;
                            row.push(GaussianSplat::surface(
                                [wx, wy, wz],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0],
                                scale,
                                scale,
                                250,
                                spectral,
                            ));
                        }
                    }
                }
                row
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Heightmap;

    #[test]
    fn triangulated_sample_matches_the_rendered_cell_diagonal() {
        // Saddle cell: bilinear interpolation lands at 0.5 in the centre, but
        // the canonical top-right -> bottom-left render diagonal is y=1.
        let hm = Heightmap::from_data(2, 2, vec![0.0, 1.0, 1.0, 0.0], 10.0);
        assert_eq!(hm.sample(5.0, 5.0), 0.5);
        assert_eq!(hm.sample_triangulated(5.0, 5.0), 1.0);

        // Both samplers retain the exact authored samples at grid vertices.
        assert_eq!(hm.sample_triangulated(0.0, 0.0), 0.0);
        assert_eq!(hm.sample_triangulated(10.0, 0.0), 1.0);
        assert_eq!(hm.sample_triangulated(0.0, 10.0), 1.0);
        assert_eq!(hm.sample_triangulated(10.0, 10.0), 0.0);
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

    // Each cell is a pure function of (x, z, seed) written to its own slot, so
    // rows are independent: `par_chunks_mut(width)` computes the identical
    // value in the identical slot on every thread -> the buffer is
    // bit-for-bit identical to the serial loop (deterministic by construction;
    // no cross-row reduction, no shared mutable state).
    use rayon::prelude::*;
    data.par_chunks_mut(width).enumerate().for_each(|(z, row)| {
        let fz = z as f32 / height as f32;
        // River valley through the middle (per-row constant).
        let dist_to_center = ((fz - 0.5).abs() * 2.0).min(1.0);
        let valley = (1.0 - (dist_to_center * dist_to_center)) * -3.0;
        let fz4 = (fz * 4.0).cos();
        let fz6 = (fz * 6.0 + 2.0).cos();
        let fz11 = (fz * 11.0).cos();

        for (x, cell) in row.iter_mut().enumerate() {
            let fx = x as f32 / width as f32;

            // Simple multi-octave noise — SAME ops/order as the scalar loop.
            let h1 = ((fx * 3.0 + seed as f32 * 0.1).sin() * fz4) * 5.0;
            let h2 = ((fx * 7.0 + 1.0).sin() * fz6) * 2.0;
            let h3 = ((fx * 13.0).sin() * fz11) * 1.0;

            *cell = h1 + h2 + h3 + valley;
        }
    });

    Heightmap::from_data(width, height, data, cell_size)
}
