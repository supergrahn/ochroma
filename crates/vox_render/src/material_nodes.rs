use serde::{Deserialize, Serialize};

/// Extended material node system — 30+ nodes matching Unreal's key capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialNodeV2 {
    // ── Input nodes ──────────────────────────────────────────────
    /// Constant scalar value.
    Constant { value: f32 },
    /// Constant spectral (8-band) value.
    ConstantSpectral { bands: [f32; 8] },
    /// UV texture coordinates.
    TextureCoord,
    /// World-space position.
    WorldPosition,
    /// Camera-to-surface view direction.
    ViewDirection,
    /// Animated time value.
    Time,

    // ── Math ─────────────────────────────────────────────────────
    Add(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    Subtract(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    Multiply(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    Divide(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    Power(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    Sqrt(Box<MaterialNodeV2>),
    Abs(Box<MaterialNodeV2>),
    Clamp {
        input: Box<MaterialNodeV2>,
        min: f32,
        max: f32,
    },
    Lerp {
        a: Box<MaterialNodeV2>,
        b: Box<MaterialNodeV2>,
        t: Box<MaterialNodeV2>,
    },

    // ── Spectral operations ──────────────────────────────────────
    SpectralMultiply(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    SpectralAdd(Box<MaterialNodeV2>, Box<MaterialNodeV2>),
    SpectralLerp {
        a: Box<MaterialNodeV2>,
        b: Box<MaterialNodeV2>,
        t: f32,
    },

    // ── Surface properties ───────────────────────────────────────
    Fresnel {
        base: Box<MaterialNodeV2>,
        exponent: f32,
    },
    Metallic { value: f32 },
    Roughness { value: f32 },
    Emission {
        input: Box<MaterialNodeV2>,
        intensity: f32,
    },
    Opacity { value: f32 },
    SubsurfaceScattering {
        color: Box<MaterialNodeV2>,
        radius: f32,
    },

    // ── Noise / procedural ───────────────────────────────────────
    PerlinNoise { scale: f32, octaves: u32 },
    VoronoiNoise { scale: f32 },
    Checkerboard {
        scale: f32,
        color_a: Box<MaterialNodeV2>,
        color_b: Box<MaterialNodeV2>,
    },
    Gradient { direction: [f32; 3] },

    // ── Utility ──────────────────────────────────────────────────
    Remap {
        input: Box<MaterialNodeV2>,
        from_min: f32,
        from_max: f32,
        to_min: f32,
        to_max: f32,
    },
    SmoothStep {
        edge0: f32,
        edge1: f32,
        input: Box<MaterialNodeV2>,
    },
    OneMinus(Box<MaterialNodeV2>),
    /// Saturate — clamp to [0, 1].
    Saturate(Box<MaterialNodeV2>),
}

/// Evaluation context — supplies runtime values (time, position, UV, etc.).
pub struct MaterialEvalContext {
    pub time: f32,
    pub world_position: [f32; 3],
    pub uv: [f32; 2],
    pub view_direction: [f32; 3],
}

impl Default for MaterialEvalContext {
    fn default() -> Self {
        Self {
            time: 0.0,
            world_position: [0.0; 3],
            uv: [0.0; 2],
            view_direction: [0.0, 0.0, -1.0],
        }
    }
}

// ── Hash-based procedural noise helpers ──────────────────────────────────────

/// Simple hash-based pseudo-random in [0, 1].
fn hash_f32(x: f32, y: f32) -> f32 {
    let n = (x * 127.1 + y * 311.7).sin() * 43_758.547;
    n.fract().abs()
}

/// Value noise — smooth interpolated hash noise.
fn value_noise(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;
    // Smoothstep
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);

    let a = hash_f32(ix, iy);
    let b = hash_f32(ix + 1.0, iy);
    let c = hash_f32(ix, iy + 1.0);
    let d = hash_f32(ix + 1.0, iy + 1.0);

    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uy
}

/// Fractional Brownian Motion (fBm) layered noise.
fn fbm(x: f32, y: f32, octaves: u32) -> f32 {
    let mut value = 0.0_f32;
    let mut amplitude = 0.5_f32;
    let mut cx = x;
    let mut cy = y;
    for _ in 0..octaves {
        value += amplitude * value_noise(cx, cy);
        cx *= 2.0;
        cy *= 2.0;
        amplitude *= 0.5;
    }
    value
}

/// Simple 2-D Voronoi / cellular noise returning distance to nearest cell centre.
fn voronoi(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;
    let mut min_dist = 1.0_f32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let nx = dx as f32;
            let ny = dy as f32;
            let px = hash_f32(ix + nx, iy + ny);
            let py = hash_f32(iy + ny, ix + nx);
            let diffx = nx + px - fx;
            let diffy = ny + py - fy;
            let d = (diffx * diffx + diffy * diffy).sqrt();
            if d < min_dist {
                min_dist = d;
            }
        }
    }
    min_dist.clamp(0.0, 1.0)
}

impl MaterialNodeV2 {
    /// Evaluate the node tree, returning a single scalar (`f32`).
    /// For spectral nodes the first band (index 0) is returned.
    pub fn evaluate_f32(&self, ctx: &MaterialEvalContext) -> f32 {
        match self {
            // ── Inputs ───────────────────────────────────────────
            Self::Constant { value } => *value,
            Self::ConstantSpectral { bands } => bands[0],
            Self::TextureCoord => ctx.uv[0], // return U
            Self::WorldPosition => ctx.world_position[0],
            Self::ViewDirection => ctx.view_direction[0],
            Self::Time => ctx.time,

            // ── Math ─────────────────────────────────────────────
            Self::Add(a, b) => a.evaluate_f32(ctx) + b.evaluate_f32(ctx),
            Self::Subtract(a, b) => a.evaluate_f32(ctx) - b.evaluate_f32(ctx),
            Self::Multiply(a, b) => a.evaluate_f32(ctx) * b.evaluate_f32(ctx),
            Self::Divide(a, b) => {
                let d = b.evaluate_f32(ctx);
                if d.abs() < 1e-10 {
                    0.0
                } else {
                    a.evaluate_f32(ctx) / d
                }
            }
            Self::Power(base, exp) => a_pow(base.evaluate_f32(ctx), exp.evaluate_f32(ctx)),
            Self::Sqrt(v) => v.evaluate_f32(ctx).max(0.0).sqrt(),
            Self::Abs(v) => v.evaluate_f32(ctx).abs(),
            Self::Clamp { input, min, max } => input.evaluate_f32(ctx).clamp(*min, *max),
            Self::Lerp { a, b, t } => {
                let va = a.evaluate_f32(ctx);
                let vb = b.evaluate_f32(ctx);
                let vt = t.evaluate_f32(ctx).clamp(0.0, 1.0);
                va + (vb - va) * vt
            }

            // ── Spectral (scalar path returns band 0) ────────────
            Self::SpectralMultiply(a, b) => {
                let sa = a.evaluate_spectral(ctx);
                let sb = b.evaluate_spectral(ctx);
                sa[0] * sb[0]
            }
            Self::SpectralAdd(a, b) => {
                let sa = a.evaluate_spectral(ctx);
                let sb = b.evaluate_spectral(ctx);
                sa[0] + sb[0]
            }
            Self::SpectralLerp { a, b, t } => {
                let sa = a.evaluate_spectral(ctx);
                let sb = b.evaluate_spectral(ctx);
                sa[0] + (sb[0] - sa[0]) * t
            }

            // ── Surface properties ───────────────────────────────
            Self::Fresnel { base, exponent } => {
                let base_val = base.evaluate_f32(ctx);
                let vd = ctx.view_direction;
                // Approximate: use view_direction.z as cos(theta) between normal (assumed [0,0,1]) and view.
                let cos_theta = vd[2].abs().clamp(0.0, 1.0);
                base_val + (1.0 - base_val) * (1.0 - cos_theta).powf(*exponent)
            }
            Self::Metallic { value } => *value,
            Self::Roughness { value } => *value,
            Self::Emission { input, intensity } => input.evaluate_f32(ctx) * intensity,
            Self::Opacity { value } => *value,
            Self::SubsurfaceScattering { color, radius } => {
                color.evaluate_f32(ctx) * radius.min(1.0)
            }

            // ── Noise / procedural ───────────────────────────────
            Self::PerlinNoise { scale, octaves } => {
                let x = ctx.uv[0] * scale;
                let y = ctx.uv[1] * scale;
                fbm(x, y, *octaves)
            }
            Self::VoronoiNoise { scale } => {
                let x = ctx.uv[0] * scale;
                let y = ctx.uv[1] * scale;
                voronoi(x, y)
            }
            Self::Checkerboard {
                scale,
                color_a,
                color_b,
            } => {
                let cx = (ctx.uv[0] * scale).floor() as i32;
                let cy = (ctx.uv[1] * scale).floor() as i32;
                if (cx + cy) % 2 == 0 {
                    color_a.evaluate_f32(ctx)
                } else {
                    color_b.evaluate_f32(ctx)
                }
            }
            Self::Gradient { direction } => {
                let p = ctx.world_position;
                let len_sq =
                    direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
                if len_sq < 1e-10 {
                    0.0
                } else {
                    let dot = p[0] * direction[0] + p[1] * direction[1] + p[2] * direction[2];
                    (dot / len_sq.sqrt()).clamp(0.0, 1.0)
                }
            }

            // ── Utility ──────────────────────────────────────────
            Self::Remap {
                input,
                from_min,
                from_max,
                to_min,
                to_max,
            } => {
                let v = input.evaluate_f32(ctx);
                let range = from_max - from_min;
                if range.abs() < 1e-10 {
                    *to_min
                } else {
                    let t = (v - from_min) / range;
                    to_min + t * (to_max - to_min)
                }
            }
            Self::SmoothStep { edge0, edge1, input } => {
                let x = input.evaluate_f32(ctx);
                let range = edge1 - edge0;
                if range.abs() < 1e-10 {
                    if x >= *edge1 { 1.0 } else { 0.0 }
                } else {
                    let t = ((x - edge0) / range).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                }
            }
            Self::OneMinus(v) => 1.0 - v.evaluate_f32(ctx),
            Self::Saturate(v) => v.evaluate_f32(ctx).clamp(0.0, 1.0),
        }
    }

    /// Evaluate the node tree, returning 8 spectral bands.
    pub fn evaluate_spectral(&self, ctx: &MaterialEvalContext) -> [f32; 8] {
        match self {
            Self::ConstantSpectral { bands } => *bands,
            Self::Constant { value } => [*value; 8],

            Self::SpectralMultiply(a, b) => {
                let sa = a.evaluate_spectral(ctx);
                let sb = b.evaluate_spectral(ctx);
                std::array::from_fn(|i| sa[i] * sb[i])
            }
            Self::SpectralAdd(a, b) => {
                let sa = a.evaluate_spectral(ctx);
                let sb = b.evaluate_spectral(ctx);
                std::array::from_fn(|i| sa[i] + sb[i])
            }
            Self::SpectralLerp { a, b, t } => {
                let sa = a.evaluate_spectral(ctx);
                let sb = b.evaluate_spectral(ctx);
                std::array::from_fn(|i| sa[i] + (sb[i] - sa[i]) * t)
            }

            // For all other nodes, broadcast the scalar result to 8 bands.
            _ => {
                let v = self.evaluate_f32(ctx);
                [v; 8]
            }
        }
    }
}

/// Safe power function (handles negative base gracefully).
fn a_pow(base: f32, exp: f32) -> f32 {
    if base < 0.0 {
        -((-base).powf(exp))
    } else {
        base.powf(exp)
    }
}

/// Complete material definition using the V2 node graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMaterial {
    pub name: String,
    pub albedo: MaterialNodeV2,
    pub metallic: MaterialNodeV2,
    pub roughness: MaterialNodeV2,
    pub emission: Option<MaterialNodeV2>,
    pub opacity: MaterialNodeV2,
    pub normal_strength: f32,
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MaterialEvalContext {
        MaterialEvalContext::default()
    }

    #[test]
    fn constant_returns_value() {
        let n = MaterialNodeV2::Constant { value: 0.42 };
        assert!((n.evaluate_f32(&ctx()) - 0.42).abs() < 1e-6);
    }

    #[test]
    fn add_works() {
        let n = MaterialNodeV2::Add(
            Box::new(MaterialNodeV2::Constant { value: 0.3 }),
            Box::new(MaterialNodeV2::Constant { value: 0.5 }),
        );
        assert!((n.evaluate_f32(&ctx()) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn multiply_works() {
        let n = MaterialNodeV2::Multiply(
            Box::new(MaterialNodeV2::Constant { value: 0.4 }),
            Box::new(MaterialNodeV2::Constant { value: 0.5 }),
        );
        assert!((n.evaluate_f32(&ctx()) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn lerp_interpolates() {
        let n = MaterialNodeV2::Lerp {
            a: Box::new(MaterialNodeV2::Constant { value: 0.0 }),
            b: Box::new(MaterialNodeV2::Constant { value: 1.0 }),
            t: Box::new(MaterialNodeV2::Constant { value: 0.25 }),
        };
        assert!((n.evaluate_f32(&ctx()) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn fresnel_increases_at_grazing_angles() {
        let base = MaterialNodeV2::Constant { value: 0.04 };
        let fresnel_node = MaterialNodeV2::Fresnel {
            base: Box::new(base),
            exponent: 5.0,
        };

        // Head-on: view_direction.z = -1 => cos_theta = 1 => (1-1)^5 = 0 => result ~ base
        let ctx_head_on = MaterialEvalContext {
            view_direction: [0.0, 0.0, -1.0],
            ..Default::default()
        };
        let head_on = fresnel_node.evaluate_f32(&ctx_head_on);

        // Grazing: view_direction nearly perpendicular => cos_theta ~ 0 => large fresnel
        let ctx_grazing = MaterialEvalContext {
            view_direction: [1.0, 0.0, -0.05],
            ..Default::default()
        };
        let grazing = fresnel_node.evaluate_f32(&ctx_grazing);

        assert!(
            grazing > head_on + 0.3,
            "Fresnel should be much higher at grazing angles: head_on={head_on}, grazing={grazing}"
        );
    }

    #[test]
    fn perlin_noise_in_range() {
        let n = MaterialNodeV2::PerlinNoise {
            scale: 4.0,
            octaves: 3,
        };
        // Sample at many UV positions
        for i in 0..20 {
            let u = i as f32 / 20.0;
            for j in 0..20 {
                let v = j as f32 / 20.0;
                let c = MaterialEvalContext {
                    uv: [u, v],
                    ..Default::default()
                };
                let val = n.evaluate_f32(&c);
                assert!(
                    val >= -0.01 && val <= 1.01,
                    "Noise out of range: {val} at uv=({u},{v})"
                );
            }
        }
    }

    #[test]
    fn checkerboard_alternates() {
        let n = MaterialNodeV2::Checkerboard {
            scale: 2.0,
            color_a: Box::new(MaterialNodeV2::Constant { value: 0.0 }),
            color_b: Box::new(MaterialNodeV2::Constant { value: 1.0 }),
        };
        // (0.1, 0.1) => floor(0.2)=0, floor(0.2)=0 => (0+0)%2=0 => color_a=0
        let c1 = MaterialEvalContext {
            uv: [0.1, 0.1],
            ..Default::default()
        };
        assert!((n.evaluate_f32(&c1) - 0.0).abs() < 1e-6);

        // (0.3, 0.1) => floor(0.6)=0, floor(0.2)=0 => still 0 => color_a
        // (0.6, 0.1) => floor(1.2)=1, floor(0.2)=0 => (1+0)%2=1 => color_b=1
        let c2 = MaterialEvalContext {
            uv: [0.6, 0.1],
            ..Default::default()
        };
        assert!((n.evaluate_f32(&c2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn remap_maps_correctly() {
        let n = MaterialNodeV2::Remap {
            input: Box::new(MaterialNodeV2::Constant { value: 0.5 }),
            from_min: 0.0,
            from_max: 1.0,
            to_min: 10.0,
            to_max: 20.0,
        };
        assert!((n.evaluate_f32(&ctx()) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn smoothstep_correct_curve() {
        let n = MaterialNodeV2::SmoothStep {
            edge0: 0.0,
            edge1: 1.0,
            input: Box::new(MaterialNodeV2::Constant { value: 0.5 }),
        };
        // smoothstep(0.5) = 0.5^2 * (3 - 2*0.5) = 0.25 * 2 = 0.5
        assert!((n.evaluate_f32(&ctx()) - 0.5).abs() < 1e-6);

        // At edges
        let n0 = MaterialNodeV2::SmoothStep {
            edge0: 0.0,
            edge1: 1.0,
            input: Box::new(MaterialNodeV2::Constant { value: 0.0 }),
        };
        assert!((n0.evaluate_f32(&ctx()) - 0.0).abs() < 1e-6);

        let n1 = MaterialNodeV2::SmoothStep {
            edge0: 0.0,
            edge1: 1.0,
            input: Box::new(MaterialNodeV2::Constant { value: 1.0 }),
        };
        assert!((n1.evaluate_f32(&ctx()) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nested_graph_evaluates() {
        // (0.3 + 0.2) * 0.5 = 0.25
        let n = MaterialNodeV2::Multiply(
            Box::new(MaterialNodeV2::Add(
                Box::new(MaterialNodeV2::Constant { value: 0.3 }),
                Box::new(MaterialNodeV2::Constant { value: 0.2 }),
            )),
            Box::new(MaterialNodeV2::Constant { value: 0.5 }),
        );
        assert!((n.evaluate_f32(&ctx()) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn spectral_multiply_works() {
        let a = MaterialNodeV2::ConstantSpectral {
            bands: [0.5; 8],
        };
        let b = MaterialNodeV2::ConstantSpectral {
            bands: [0.4, 0.6, 0.8, 1.0, 0.2, 0.3, 0.7, 0.9],
        };
        let n = MaterialNodeV2::SpectralMultiply(Box::new(a), Box::new(b));
        let result = n.evaluate_spectral(&ctx());
        assert!((result[0] - 0.2).abs() < 1e-6);
        assert!((result[1] - 0.3).abs() < 1e-6);
        assert!((result[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn one_minus_and_saturate() {
        let n = MaterialNodeV2::OneMinus(Box::new(MaterialNodeV2::Constant { value: 0.3 }));
        assert!((n.evaluate_f32(&ctx()) - 0.7).abs() < 1e-6);

        let s = MaterialNodeV2::Saturate(Box::new(MaterialNodeV2::Constant { value: 1.5 }));
        assert!((s.evaluate_f32(&ctx()) - 1.0).abs() < 1e-6);

        let s2 = MaterialNodeV2::Saturate(Box::new(MaterialNodeV2::Constant { value: -0.5 }));
        assert!((s2.evaluate_f32(&ctx()) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn node_material_struct() {
        let mat = NodeMaterial {
            name: "test_mat".to_string(),
            albedo: MaterialNodeV2::Constant { value: 0.8 },
            metallic: MaterialNodeV2::Metallic { value: 0.0 },
            roughness: MaterialNodeV2::Roughness { value: 0.5 },
            emission: None,
            opacity: MaterialNodeV2::Opacity { value: 1.0 },
            normal_strength: 1.0,
        };
        assert_eq!(mat.name, "test_mat");
        assert!((mat.albedo.evaluate_f32(&ctx()) - 0.8).abs() < 1e-6);
        assert!((mat.roughness.evaluate_f32(&ctx()) - 0.5).abs() < 1e-6);
    }
}
