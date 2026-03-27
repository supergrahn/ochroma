use vox_core::types::GaussianSplat;

/// Number of LOD levels in a chain.
pub const LOD_LEVEL_COUNT: usize = 4;

/// A single LOD level within a chain.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// Level index: 0 = full detail, 3 = billboard.
    pub level: u32,
    /// Number of splats at this level.
    pub splat_count: usize,
    /// Bounding sphere radius for this level's geometry.
    pub bounding_sphere_radius: f32,
    /// Distance at which this level is selected.
    pub transition_distance: f32,
    /// The splats at this level.
    pub splats: Vec<GaussianSplat>,
}

/// Complete LOD hierarchy for an asset: [LOD0: full, LOD1: 40%, LOD2: 10%, LOD3: 1 billboard].
#[derive(Debug, Clone)]
pub struct LodChain {
    pub levels: [LodLevel; LOD_LEVEL_COUNT],
}

/// Fractions of original splat count for each LOD level.
const LOD_FRACTIONS: [f32; LOD_LEVEL_COUNT] = [1.0, 0.4, 0.1, 0.0]; // 0.0 means 1 billboard

/// Default transition distances for each level.
const LOD_DISTANCES: [f32; LOD_LEVEL_COUNT] = [0.0, 50.0, 150.0, 400.0];

/// Generate a complete 4-level LOD chain from a set of splats by stride-sampling.
pub fn generate_lod_chain(splats: &[GaussianSplat]) -> LodChain {
    let bounding_radius = compute_bounding_radius(splats);

    let mut levels: Vec<LodLevel> = Vec::with_capacity(LOD_LEVEL_COUNT);

    for i in 0..LOD_LEVEL_COUNT {
        let level_splats = if i == 0 {
            // LOD 0: full detail.
            splats.to_vec()
        } else if i == LOD_LEVEL_COUNT - 1 {
            // LOD 3: single billboard splat.
            if splats.is_empty() {
                vec![]
            } else {
                vec![create_billboard_splat(splats)]
            }
        } else {
            // LOD 1-2: stride-sample.
            stride_sample(splats, LOD_FRACTIONS[i])
        };

        levels.push(LodLevel {
            level: i as u32,
            splat_count: level_splats.len(),
            bounding_sphere_radius: bounding_radius,
            transition_distance: LOD_DISTANCES[i],
            splats: level_splats,
        });
    }

    LodChain {
        levels: levels.try_into().unwrap_or_else(|v: Vec<LodLevel>| {
            panic!("Expected {} levels, got {}", LOD_LEVEL_COUNT, v.len())
        }),
    }
}

/// Stride-sample splats at a given fraction (e.g., 0.4 = keep 40%).
fn stride_sample(splats: &[GaussianSplat], fraction: f32) -> Vec<GaussianSplat> {
    let target = ((splats.len() as f32 * fraction) as usize).max(1);
    if target >= splats.len() {
        return splats.to_vec();
    }
    let step = splats.len() as f32 / target as f32;
    (0..target)
        .map(|i| splats[(i as f32 * step) as usize])
        .collect()
}

/// Create a single billboard splat that represents the entire asset.
fn create_billboard_splat(splats: &[GaussianSplat]) -> GaussianSplat {
    if splats.is_empty() {
        return GaussianSplat {
            position: [0.0; 3],
            scale: [1.0; 3],
            rotation: [0, 0, 0, 32767],
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        };
    }

    // Average position.
    let mut avg_pos = [0.0f32; 3];
    for s in splats {
        for j in 0..3 {
            avg_pos[j] += s.position[j];
        }
    }
    let n = splats.len() as f32;
    for p in &mut avg_pos {
        *p /= n;
    }

    // Scale to encompass all splats.
    let radius = compute_bounding_radius(splats);
    let scale = [radius, radius, radius];

    // Average spectral values.
    let mut avg_spectral = [0u32; 8];
    for s in splats {
        for j in 0..8 {
            avg_spectral[j] += s.spectral[j] as u32;
        }
    }
    let mut spectral = [0u16; 8];
    for j in 0..8 {
        spectral[j] = (avg_spectral[j] / splats.len() as u32) as u16;
    }

    GaussianSplat {
        position: avg_pos,
        scale,
        rotation: [0, 0, 0, 32767], // identity quaternion
        opacity: 255,
        _pad: [0; 3],
        spectral,
    }
}

/// Compute bounding sphere radius from a set of splats.
fn compute_bounding_radius(splats: &[GaussianSplat]) -> f32 {
    if splats.is_empty() {
        return 0.0;
    }

    let mut center = [0.0f32; 3];
    for s in splats {
        for j in 0..3 {
            center[j] += s.position[j];
        }
    }
    let n = splats.len() as f32;
    for c in &mut center {
        *c /= n;
    }

    let mut max_dist_sq = 0.0f32;
    for s in splats {
        let dx = s.position[0] - center[0];
        let dy = s.position[1] - center[1];
        let dz = s.position[2] - center[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        max_dist_sq = max_dist_sq.max(dist_sq);
    }

    max_dist_sq.sqrt()
}

/// Select the appropriate LOD level based on distance and screen size.
///
/// `distance` is the distance from the camera to the object.
/// `screen_size` is the projected screen-space size in pixels (larger = closer/bigger).
pub fn select_lod_level(distance: f32, screen_size: f32) -> u32 {
    // Use a combination of distance and screen size.
    // Screen size < 10px = billboard, < 50px = LOD2, < 200px = LOD1, else LOD0.
    if screen_size < 10.0 || distance > LOD_DISTANCES[3] {
        3
    } else if screen_size < 50.0 || distance > LOD_DISTANCES[2] {
        2
    } else if screen_size < 200.0 || distance > LOD_DISTANCES[1] {
        1
    } else {
        0
    }
}

/// Compute a crossfade factor for smooth transitions between LOD levels.
///
/// Returns 0.0 when firmly within the current level, 1.0 at the boundary
/// where the next level takes over.
pub fn crossfade_factor(distance: f32, level: u32) -> f32 {
    let level_idx = level as usize;
    if level_idx >= LOD_LEVEL_COUNT - 1 {
        return 0.0; // No transition beyond last level.
    }

    let current_dist = LOD_DISTANCES[level_idx];
    let next_dist = LOD_DISTANCES[level_idx + 1];
    let range = next_dist - current_dist;

    if range <= 0.0 {
        return 0.0;
    }

    // Transition band is the last 20% of each level's distance range.
    let transition_start = current_dist + range * 0.8;

    if distance <= transition_start {
        0.0
    } else if distance >= next_dist {
        1.0
    } else {
        let transition_range = next_dist - transition_start;
        ((distance - transition_start) / transition_range).clamp(0.0, 1.0)
    }
}

/// Generates sub-splat micro-detail for extreme zoom levels.
pub struct MicroDetailGenerator;

impl MicroDetailGenerator {
    /// Generate brick wall detail with mortar lines.
    /// Returns a cluster of fine splats with height variation for mortar grooves.
    pub fn generate_brick_detail(seed: u32) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        let brick_w = 0.2;
        let brick_h = 0.1;
        let mortar = 0.01;

        // Simple deterministic pseudo-random from seed.
        let mut rng = seed;

        for row in 0..4 {
            let offset = if row % 2 == 0 { 0.0 } else { brick_w * 0.5 };
            for col in 0..5 {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let variation = (rng % 100) as f32 / 1000.0;

                let x = col as f32 * (brick_w + mortar) + offset;
                let y = row as f32 * (brick_h + mortar);

                // Brick face splat (raised).
                splats.push(make_micro_splat(
                    [x, y, 0.01 + variation],
                    [brick_w * 0.45, brick_h * 0.45, 0.005],
                ));

                // Mortar groove splat (recessed) — height variation.
                if col < 4 {
                    splats.push(make_micro_splat(
                        [x + brick_w, y, -0.005],
                        [mortar * 0.5, brick_h * 0.45, 0.002],
                    ));
                }
            }

            // Horizontal mortar line.
            splats.push(make_micro_splat(
                [0.5, y_mortar(row, brick_h, mortar), -0.005],
                [0.5, mortar * 0.5, 0.002],
            ));
        }

        splats
    }

    /// Generate wood grain detail.
    pub fn generate_wood_grain(seed: u32) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        let mut rng = seed;

        for i in 0..12 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let offset = (rng % 100) as f32 / 500.0;

            let y = i as f32 * 0.08 + offset;
            let thickness = 0.003 + (rng % 50) as f32 / 10000.0;

            splats.push(make_micro_splat(
                [0.5, y, 0.0],
                [0.5, thickness, 0.001],
            ));
        }

        splats
    }

    /// Generate metal scratch detail.
    pub fn generate_metal_scratches(seed: u32) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        let mut rng = seed;

        for _ in 0..8 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let x = (rng % 1000) as f32 / 1000.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let y = (rng % 1000) as f32 / 1000.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let angle = (rng % 628) as f32 / 100.0; // 0..2pi

            let length = 0.05 + (rng % 100) as f32 / 1000.0;
            let sx = length * angle.cos();
            let sy = length * angle.sin();

            splats.push(make_micro_splat(
                [x, y, -0.001],
                [sx.abs().max(0.002), sy.abs().max(0.002), 0.0005],
            ));
        }

        splats
    }
}

fn y_mortar(row: i32, brick_h: f32, mortar: f32) -> f32 {
    row as f32 * (brick_h + mortar) + brick_h
}

fn make_micro_splat(position: [f32; 3], scale: [f32; 3]) -> GaussianSplat {
    GaussianSplat {
        position,
        scale,
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        _pad: [0; 3],
        spectral: [0; 8],
    }
}

/// Temporal Anti-Aliasing accumulator for sub-pixel jitter sampling.
#[derive(Debug, Clone)]
pub struct TemporalAccumulator {
    /// Accumulated pixel buffer (flattened RGBA f32).
    pub accumulated: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// Number of accumulated samples.
    pub sample_count: u32,
    /// Previous frame's motion vectors for history rejection.
    pub motion_vectors: Vec<[f32; 2]>,
    /// Motion threshold for history rejection.
    pub rejection_threshold: f32,
}

impl TemporalAccumulator {
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            accumulated: vec![0.0; pixel_count * 4],
            width,
            height,
            sample_count: 0,
            motion_vectors: vec![[0.0, 0.0]; pixel_count],
            rejection_threshold: 2.0, // pixels
        }
    }

    /// Add a jittered sample to the accumulation buffer.
    ///
    /// `pixels` is RGBA f32 data, `jitter_offset` is the sub-pixel offset used.
    pub fn add_sample(&mut self, pixels: &[f32], _jitter_offset: [f32; 2]) {
        let expected = (self.width * self.height * 4) as usize;
        assert_eq!(pixels.len(), expected, "pixel buffer size mismatch");

        let pixel_count = (self.width * self.height) as usize;

        for i in 0..pixel_count {
            let base = i * 4;
            let mv_len =
                (self.motion_vectors[i][0].powi(2) + self.motion_vectors[i][1].powi(2)).sqrt();

            if mv_len > self.rejection_threshold {
                // History rejection: replace with current sample.
                for c in 0..4 {
                    self.accumulated[base + c] = pixels[base + c];
                }
            } else {
                // Exponential moving average blend.
                let weight = if self.sample_count == 0 {
                    1.0
                } else {
                    1.0 / (self.sample_count as f32 + 1.0)
                };
                for c in 0..4 {
                    self.accumulated[base + c] =
                        self.accumulated[base + c] * (1.0 - weight) + pixels[base + c] * weight;
                }
            }
        }

        self.sample_count += 1;
    }

    /// Set motion vectors for history rejection.
    pub fn set_motion_vectors(&mut self, vectors: &[[f32; 2]]) {
        let expected = (self.width * self.height) as usize;
        assert_eq!(vectors.len(), expected);
        self.motion_vectors.copy_from_slice(vectors);
    }

    /// Resolve the accumulated buffer to final pixel values.
    pub fn resolve(&self) -> Vec<f32> {
        self.accumulated.clone()
    }

    /// Reset the accumulator for a new sequence.
    pub fn reset(&mut self) {
        self.accumulated.fill(0.0);
        self.sample_count = 0;
        self.motion_vectors.iter_mut().for_each(|v| *v = [0.0, 0.0]);
    }

    /// Compute variance of the accumulated buffer (measure of convergence).
    /// Lower variance = better quality from accumulation.
    pub fn compute_variance(&self, current_pixels: &[f32]) -> f32 {
        let expected = (self.width * self.height * 4) as usize;
        if current_pixels.len() != expected || self.accumulated.len() != expected {
            return f32::MAX;
        }

        let mut total_diff = 0.0f32;
        let pixel_count = (self.width * self.height) as usize;
        for i in 0..pixel_count {
            let base = i * 4;
            for c in 0..3 {
                // RGB only
                let diff = self.accumulated[base + c] - current_pixels[base + c];
                total_diff += diff * diff;
            }
        }

        total_diff / (pixel_count as f32 * 3.0)
    }
}
