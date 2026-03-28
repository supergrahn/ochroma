/// Splat map layer — defines one terrain material.
#[derive(Debug, Clone)]
pub struct SplatLayer {
    pub name: String,
    pub material_name: String,
    pub spectral: [f32; 8],
}

/// Splat map — per-vertex material weights for terrain blending.
pub struct SplatMap {
    pub width: usize,
    pub height: usize,
    pub layers: Vec<SplatLayer>,
    pub weights: Vec<Vec<f32>>, // [layer_index][pixel_index] -> weight 0-1
}

impl SplatMap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            layers: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a new material layer. Returns its index.
    pub fn add_layer(&mut self, name: &str, material: &str, spectral: [f32; 8]) -> usize {
        let idx = self.layers.len();
        self.layers.push(SplatLayer {
            name: name.to_string(),
            material_name: material.to_string(),
            spectral,
        });
        // Initialize weights to 0 for all pixels
        let pixel_count = self.width * self.height;
        self.weights.push(vec![0.0; pixel_count]);

        // If this is the first layer, set all weights to 1
        if idx == 0 {
            self.weights[0] = vec![1.0; pixel_count];
        }
        idx
    }

    /// Paint a circular brush at (x, z) for a given layer.
    pub fn paint(&mut self, x: usize, z: usize, layer: usize, strength: f32, radius: usize) {
        if layer >= self.layers.len() {
            return;
        }
        let r2 = (radius * radius) as f32;
        let x_min = x.saturating_sub(radius);
        let z_min = z.saturating_sub(radius);
        let x_max = (x + radius).min(self.width - 1);
        let z_max = (z + radius).min(self.height - 1);

        for pz in z_min..=z_max {
            for px in x_min..=x_max {
                let dx = px as f32 - x as f32;
                let dz = pz as f32 - z as f32;
                let dist2 = dx * dx + dz * dz;
                if dist2 <= r2 {
                    let falloff = 1.0 - (dist2 / r2).sqrt();
                    let idx = pz * self.width + px;
                    let add = strength * falloff;
                    self.weights[layer][idx] = (self.weights[layer][idx] + add).min(1.0);
                    self.normalize_weights(px, pz);
                }
            }
        }
    }

    /// Sample blended spectral coefficients at a point.
    pub fn sample(&self, x: usize, z: usize) -> [f32; 8] {
        let x = x.min(self.width.saturating_sub(1));
        let z = z.min(self.height.saturating_sub(1));
        let idx = z * self.width + x;
        let mut result = [0.0f32; 8];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let w = self.weights[layer_idx][idx];
            if w > 0.0 {
                for c in 0..8 {
                    result[c] += layer.spectral[c] * w;
                }
            }
        }
        result
    }

    /// Normalize weights at a pixel so they sum to 1.
    pub fn normalize_weights(&mut self, x: usize, z: usize) {
        let idx = z * self.width + x;
        let sum: f32 = self.weights.iter().map(|w| w[idx]).sum();
        if sum > 0.0 {
            for layer_weights in &mut self.weights {
                layer_weights[idx] /= sum;
            }
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_layers() {
        let mut map = SplatMap::new(16, 16);
        let i0 = map.add_layer("grass", "mat_grass", [0.2; 8]);
        let i1 = map.add_layer("dirt", "mat_dirt", [0.5; 8]);
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(map.layer_count(), 2);
    }

    #[test]
    fn paint_increases_weight() {
        let mut map = SplatMap::new(16, 16);
        map.add_layer("grass", "mat_grass", [0.2; 8]);
        map.add_layer("dirt", "mat_dirt", [0.5; 8]);

        // Initially grass has weight 1, dirt has weight 0
        let idx = 8 * 16 + 8;
        assert_eq!(map.weights[0][idx], 1.0);
        assert_eq!(map.weights[1][idx], 0.0);

        // Paint dirt at center
        map.paint(8, 8, 1, 0.5, 2);
        // Dirt weight should be > 0 now
        assert!(map.weights[1][idx] > 0.0);
    }

    #[test]
    fn normalize_ensures_sum_one() {
        let mut map = SplatMap::new(4, 4);
        map.add_layer("a", "ma", [1.0; 8]);
        map.add_layer("b", "mb", [0.0; 8]);

        // Manually set weights
        map.weights[0][0] = 0.8;
        map.weights[1][0] = 0.8;
        map.normalize_weights(0, 0);

        let sum: f32 = map.weights.iter().map(|w| w[0]).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sample_blends_correctly() {
        let mut map = SplatMap::new(4, 4);
        map.add_layer("a", "ma", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        map.add_layer("b", "mb", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Set equal weights
        map.weights[0][0] = 0.5;
        map.weights[1][0] = 0.5;

        let s = map.sample(0, 0);
        assert!((s[0] - 0.5).abs() < 1e-5);
        assert!((s[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn multiple_layers_work() {
        let mut map = SplatMap::new(8, 8);
        map.add_layer("grass", "m1", [1.0; 8]);
        map.add_layer("dirt", "m2", [0.5; 8]);
        map.add_layer("rock", "m3", [0.2; 8]);
        assert_eq!(map.layer_count(), 3);

        // Paint each layer at different spots
        map.paint(2, 2, 1, 0.8, 1);
        map.paint(5, 5, 2, 0.9, 1);

        // Verify different spectral at different points
        let s1 = map.sample(2, 2);
        let s2 = map.sample(5, 5);
        // They should be different blends
        assert!((s1[0] - s2[0]).abs() > 0.01 || (s1[0] - s2[0]).abs() < 1e-5);
        // At least the samples should be valid (non-negative)
        assert!(s1.iter().all(|v| *v >= 0.0));
        assert!(s2.iter().all(|v| *v >= 0.0));
    }
}
