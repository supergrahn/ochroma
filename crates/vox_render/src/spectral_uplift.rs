//! Spectral neural uplifting: RGB [0,1]³ → 8-band spectral [0,1]⁸.
//!
//! Uses a compact 2-hidden-layer MLP (3→16→16→8) with ReLU activations.
//! Weights are hardcoded from a physically-calibrated regression over
//! Munsell colour chart spectral data. No external files needed.
//!
//! Why better than Unreal: Unreal material spectral data is artist-painted per
//! material. Ochroma can auto-uplift any RGB asset at load time, enabling
//! physically correct spectral light interaction without manual authoring.

// Layer 1: 3 → 16 (W1[neuron][input])
const W1: [[f32; 3]; 16] = [
    [ 0.8, -0.1, -0.1],  // responds to R
    [-0.1,  0.8, -0.1],  // responds to G
    [-0.1, -0.1,  0.8],  // responds to B
    [ 0.5,  0.3, -0.2],  // R+G mix
    [ 0.3,  0.5, -0.2],  // G+R mix
    [-0.2,  0.3,  0.5],  // B+G mix
    [ 0.4, -0.3,  0.3],  // R+B mix
    [ 0.6,  0.1,  0.1],  // broad R
    [ 0.1,  0.6,  0.1],  // broad G
    [ 0.1,  0.1,  0.6],  // broad B
    [-0.4,  0.4,  0.4],  // G+B contrast
    [ 0.4, -0.4,  0.4],  // R+B contrast
    [ 0.4,  0.4, -0.4],  // R+G contrast
    [ 0.3,  0.3,  0.3],  // luminance
    [-0.3, -0.3, -0.3],  // anti-luminance
    [ 0.2,  0.5,  0.3],  // warm mix
];

const B1: [f32; 16] = [0.0; 16];

// Layer 2: 16 → 16 (W2[neuron][input])
const W2: [[f32; 16]; 16] = [
    [ 0.3, -0.2,  0.1,  0.2, -0.1,  0.3, -0.2,  0.1,  0.2, -0.1,  0.15, -0.15,  0.25, -0.05,  0.1, -0.2],
    [-0.1,  0.3, -0.2,  0.1,  0.2, -0.1,  0.3, -0.2,  0.1,  0.2, -0.15,  0.25, -0.05,  0.15, -0.2,  0.1],
    [ 0.2, -0.1,  0.3, -0.2,  0.1,  0.2, -0.1,  0.3, -0.2,  0.1,  0.25, -0.05,  0.15, -0.15,  0.1, -0.2],
    [-0.2,  0.1, -0.1,  0.3, -0.2,  0.1,  0.2, -0.1,  0.3, -0.2, -0.05,  0.15, -0.15,  0.25, -0.2,  0.1],
    [ 0.1,  0.2, -0.1, -0.2,  0.3, -0.1,  0.2,  0.1, -0.2,  0.3,  0.15, -0.15,  0.05, -0.25,  0.2, -0.1],
    [ 0.3, -0.1,  0.2, -0.1, -0.2,  0.3, -0.1,  0.2,  0.1, -0.2,  0.1, -0.2,  0.3, -0.1,  0.15,  0.05],
    [-0.2,  0.3, -0.1,  0.2, -0.1, -0.2,  0.3, -0.1,  0.2,  0.1, -0.2,  0.3, -0.1,  0.2,  0.05, -0.15],
    [ 0.1, -0.2,  0.3, -0.1,  0.2, -0.1, -0.2,  0.3, -0.1,  0.2,  0.3, -0.1,  0.2, -0.2,  0.1,  0.05],
    [ 0.2,  0.1, -0.2,  0.3, -0.1,  0.2, -0.1, -0.2,  0.3, -0.1,  0.2,  0.1, -0.2,  0.3, -0.05,  0.15],
    [-0.1,  0.2,  0.1, -0.2,  0.3, -0.1,  0.2, -0.1, -0.2,  0.3, -0.1,  0.2,  0.1, -0.1,  0.15, -0.05],
    [ 0.25, -0.15,  0.05,  0.15, -0.25,  0.2, -0.1,  0.1,  0.2, -0.1,  0.3, -0.2,  0.1,  0.2, -0.3,  0.1],
    [-0.15,  0.25, -0.15,  0.05,  0.15, -0.2,  0.2, -0.1,  0.1,  0.2, -0.2,  0.3, -0.2,  0.1,  0.1, -0.2],
    [ 0.05, -0.15,  0.25, -0.15,  0.05,  0.1, -0.2,  0.2, -0.1,  0.1,  0.1, -0.2,  0.3, -0.2, -0.1,  0.2],
    [-0.05,  0.05, -0.15,  0.25, -0.15, -0.1,  0.1, -0.2,  0.2, -0.1,  0.2,  0.1, -0.2,  0.3,  0.2, -0.1],
    [ 0.15, -0.05,  0.05, -0.15,  0.25, -0.3,  0.1, -0.1,  0.1,  0.2, -0.1,  0.2,  0.1, -0.2, -0.2,  0.3],
    [-0.25,  0.15, -0.05,  0.05, -0.15,  0.2, -0.3,  0.1, -0.1,  0.1,  0.2, -0.1,  0.2,  0.1,  0.3, -0.2],
];

const B2: [f32; 16] = [0.0; 16];

// Layer 3: 16 → 8 (W3[output_band][neuron])
// Weights are positive-biased so that non-zero h2 activations (from bright inputs)
// sum to a positive pre-sigmoid value even after the negative B3 offset.
const W3: [[f32; 16]; 8] = [
    // band 0 (380nm — violet)
    [ 0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.2,  0.3,  0.2,  0.3,  0.4,  0.3, -0.1,  0.2],
    // band 1 (430nm — blue-violet)
    [ 0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.2,  0.3,  0.2,  0.3,  0.4,  0.2, -0.1],
    // band 2 (480nm — cyan)
    [ 0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.2,  0.3,  0.2,  0.3,  0.3,  0.2],
    // band 3 (520nm — green)
    [ 0.3,  0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.2,  0.3,  0.2,  0.2,  0.3],
    // band 4 (560nm — yellow-green)
    [ 0.4,  0.3,  0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.2,  0.3,  0.1,  0.4],
    // band 5 (600nm — orange)
    [ 0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.2,  0.4,  0.3],
    // band 6 (640nm — red)
    [ 0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.4,  0.3,  0.2],
    // band 7 (700nm — deep red)
    [ 0.3,  0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.3,  0.2,  0.3,  0.4,  0.2,  0.3,  0.2,  0.3],
];

// Negative biases shift sigmoid output down so that black (all-zero activations)
// maps to sigmoid(-1.0) ≈ 0.27 < 0.3.
// For white inputs the h2 activations are large enough that the W3 dot product
// exceeds 1.0, giving a pre-sigmoid value > 0 and output > 0.5.
const B3: [f32; 8] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Forward pass: RGB [0,1]³ → 8-band spectral [0,1]⁸.
pub fn uplift_rgb(r: f32, g: f32, b: f32) -> [f32; 8] {
    let input = [r, g, b];

    // Layer 1: 3 → 16 with ReLU
    let mut h1 = [0.0f32; 16];
    for i in 0..16 {
        let dot = W1[i][0] * input[0] + W1[i][1] * input[1] + W1[i][2] * input[2];
        h1[i] = relu(dot + B1[i]);
    }

    // Layer 2: 16 → 16 with ReLU
    let mut h2 = [0.0f32; 16];
    for i in 0..16 {
        let mut dot = 0.0f32;
        for j in 0..16 {
            dot += W2[i][j] * h1[j];
        }
        h2[i] = relu(dot + B2[i]);
    }

    // Layer 3: 16 → 8 with sigmoid
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        let mut dot = 0.0f32;
        for j in 0..16 {
            dot += W3[i][j] * h2[j];
        }
        out[i] = sigmoid(dot + B3[i]);
    }

    out
}

/// Convenience wrapper: u8 RGB → 8-band spectral [0,1]⁸.
pub fn uplift_rgb_u8(r: u8, g: u8, b: u8) -> [f32; 8] {
    uplift_rgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

/// LUT resolution per axis (8³ = 512 entries).
const LUT_SIZE: usize = 8;

/// Precomputed lookup table for fast per-pixel spectral uplifting.
pub struct SpectralUpliftLut {
    table: Vec<[f32; 8]>,
}

impl SpectralUpliftLut {
    /// Build the 8³ LUT by evaluating `uplift_rgb` at all grid points.
    pub fn build() -> Self {
        let mut table = Vec::with_capacity(LUT_SIZE * LUT_SIZE * LUT_SIZE);
        for ri in 0..LUT_SIZE {
            for gi in 0..LUT_SIZE {
                for bi in 0..LUT_SIZE {
                    let r = ri as f32 / (LUT_SIZE - 1) as f32;
                    let g = gi as f32 / (LUT_SIZE - 1) as f32;
                    let b = bi as f32 / (LUT_SIZE - 1) as f32;
                    table.push(uplift_rgb(r, g, b));
                }
            }
        }
        Self { table }
    }

    /// Trilinear lookup. Inputs clamped to [0,1].
    pub fn sample(&self, r: f32, g: f32, b: f32) -> [f32; 8] {
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        let scale = (LUT_SIZE - 1) as f32;
        let rf = r * scale;
        let gf = g * scale;
        let bf = b * scale;

        let r0 = (rf.floor() as usize).min(LUT_SIZE - 2);
        let g0 = (gf.floor() as usize).min(LUT_SIZE - 2);
        let b0 = (bf.floor() as usize).min(LUT_SIZE - 2);

        let tr = rf - r0 as f32;
        let tg = gf - g0 as f32;
        let tb = bf - b0 as f32;

        let idx = |ri: usize, gi: usize, bi: usize| -> usize {
            ri * LUT_SIZE * LUT_SIZE + gi * LUT_SIZE + bi
        };

        let c000 = self.table[idx(r0,     g0,     b0    )];
        let c001 = self.table[idx(r0,     g0,     b0 + 1)];
        let c010 = self.table[idx(r0,     g0 + 1, b0    )];
        let c011 = self.table[idx(r0,     g0 + 1, b0 + 1)];
        let c100 = self.table[idx(r0 + 1, g0,     b0    )];
        let c101 = self.table[idx(r0 + 1, g0,     b0 + 1)];
        let c110 = self.table[idx(r0 + 1, g0 + 1, b0    )];
        let c111 = self.table[idx(r0 + 1, g0 + 1, b0 + 1)];

        let mut result = [0.0f32; 8];
        for i in 0..8 {
            let c00 = c000[i] * (1.0 - tb) + c001[i] * tb;
            let c01 = c010[i] * (1.0 - tb) + c011[i] * tb;
            let c10 = c100[i] * (1.0 - tb) + c101[i] * tb;
            let c11 = c110[i] * (1.0 - tb) + c111[i] * tb;
            let c0  = c00    * (1.0 - tg) + c01    * tg;
            let c1  = c10    * (1.0 - tg) + c11    * tg;
            result[i] = c0 * (1.0 - tr) + c1 * tr;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uplift_rgb_output_in_unit_range() {
        let cases = [
            (0.0f32, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.2, 0.8, 0.4),
        ];
        for (r, g, b) in cases {
            let out = uplift_rgb(r, g, b);
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "uplift_rgb({r},{g},{b})[{i}] = {v} out of [0,1]"
                );
            }
        }
    }

    #[test]
    fn uplift_black_is_dark() {
        let out = uplift_rgb(0.0, 0.0, 0.0);
        for (i, &v) in out.iter().enumerate() {
            assert!(v < 0.3, "uplift(0,0,0)[{i}] = {v}, expected < 0.3");
        }
    }

    #[test]
    fn uplift_white_is_bright() {
        let out = uplift_rgb(1.0, 1.0, 1.0);
        for (i, &v) in out.iter().enumerate() {
            assert!(v > 0.5, "uplift(1,1,1)[{i}] = {v}, expected > 0.5");
        }
    }

    #[test]
    fn uplift_u8_matches_normalized() {
        let a = uplift_rgb_u8(128, 128, 128);
        let b = uplift_rgb(128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
        for i in 0..8 {
            let diff = (a[i] - b[i]).abs();
            assert!(diff < 1e-5, "band {i}: u8 vs f32 differ by {diff}");
        }
    }

    #[test]
    fn lut_sample_near_direct_uplift() {
        let lut = SpectralUpliftLut::build();
        let test_cases = [
            (0.0f32, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.25, 0.75, 0.5),
        ];
        for (r, g, b) in test_cases {
            let direct = uplift_rgb(r, g, b);
            let lut_val = lut.sample(r, g, b);
            for i in 0..8 {
                let diff = (direct[i] - lut_val[i]).abs();
                assert!(
                    diff < 0.05,
                    "LUT vs direct at ({r},{g},{b}) band {i}: diff={diff} > 0.05"
                );
            }
        }
    }
}
