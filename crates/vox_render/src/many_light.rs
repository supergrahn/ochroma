//! Many-light stochastic sampler — importance-driven light selection for
//! scenes with thousands of lights, where shading every point against every
//! light is O(lights).
//!
//! # The RIS / WRS math
//!
//! We pick ONE light to shade a point from, in proportion to its (unshadowed)
//! contribution, using **Weighted Reservoir Sampling** (WRS, Chao's algorithm
//! / the ReSTIR streaming form of Resampled Importance Sampling, RIS).
//!
//! For a shade point `x` and the `M` candidate lights `i`, define the target
//! function `p̂_i = contribution(light_i, x)` (intensity × attenuation × color
//! luminance). We stream the lights into a reservoir, keeping candidate `i`
//! with probability `p̂_i / (Σ_{j≤i} p̂_j)`. After the stream the reservoir
//! holds light `s` with probability `p̂_s / Σ_j p̂_j`.
//!
//! The RIS estimator is unbiased when each chosen sample carries the
//! reciprocal-PDF weight
//!
//! ```text
//!     W = (1 / p̂_s) · ( (Σ_j p̂_j) / M )
//! ```
//!
//! so that `E[ p̂_s · W ] = (1/M) Σ_j p̂_j = mean contribution`, and summing
//! `f(s)·W` over `N` reservoirs converges to the brute-force Σ_i f(i).
//! (`Σ_j p̂_j` is the running reservoir weight, `M` the candidate count.)

use crate::lighting::PointLight;
use glam::Vec3;

/// Number of spectral bands in the engine's spectral representation.
pub const SPECTRAL_BANDS: usize = 16;

/// Deterministic LCG matching the engine idiom (`vfx_graph.rs`). Rollback
/// determinism requires this over the `rand` crate. The `[0,1)` contract is
/// honoured by keeping only 24 mantissa-exact bits (the `>> 40` form) so the
/// top draw is `(2^24 - 1)/2^24 < 1.0`, never exactly `1.0`.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Seed-mix to avoid a degenerate zero state, same constants as the
        // engine's vfx LCG seeding.
        Self {
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }

    /// Next f32 in `[0, 1)`. Mantissa-exact, strictly below 1.0.
    fn next_unit(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Rec. 709 luminance of a linear RGB colour.
#[inline]
fn luminance(color: [f32; 3]) -> f32 {
    0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
}

/// A spectral wrapper over a [`PointLight`]: carries 16-band radiant power.
///
/// The sampler's spectral target function uses the summed band power in place
/// of RGB luminance, so brighter (more total power) lights are preferred.
#[derive(Debug, Clone)]
pub struct SpectralPointLight {
    /// The geometric/attenuation light this wraps.
    pub light: PointLight,
    /// Per-band radiant power, 16 spectral bands.
    pub power: [f32; SPECTRAL_BANDS],
}

impl SpectralPointLight {
    /// Summed band power — the spectral analogue of RGB luminance.
    #[inline]
    pub fn total_power(&self) -> f32 {
        self.power.iter().sum()
    }

    /// Spectral target function `p̂` at `shade_point`:
    /// `attenuation(distance) × Σ_band power`.
    #[inline]
    pub fn target(&self, shade_point: Vec3) -> f32 {
        let d = self.light.position.distance(shade_point);
        let att = self.light.attenuation(d);
        att * self.total_power()
    }
}

/// One selected light plus its unbiased RIS weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSample {
    /// Index into the light slice the sampler was built over.
    pub light_index: usize,
    /// Reciprocal-PDF RIS weight `W = (1/p̂_s) · (Σp̂ / M)`.
    pub weight: f32,
    /// The target value `p̂_s` of the chosen light (handy for estimators).
    pub target: f32,
    /// Candidate count `M` this reservoir streamed (lights inspected).
    ///
    /// `W` carries a `1/M` factor, so `weight · M` is the canonical
    /// sum-estimator `(1/p̂_s)·Σp̂` used to reconstruct the full Σ over lights.
    pub m: usize,
}

/// Target function `p̂` for an RGB point light at a shade point:
/// `intensity` is folded into `attenuation`, so this is
/// `attenuation(distance) × luminance(color)`.
#[inline]
fn rgb_target(light: &PointLight, shade_point: Vec3) -> f32 {
    let d = light.position.distance(shade_point);
    let att = light.attenuation(d);
    att * luminance(light.color)
}

/// Weighted reservoir sampler over a slice of [`PointLight`]s.
///
/// Borrows the lights; build is O(1). Sampling is O(lights) for the `full`
/// path or O(cell) when driven by a [`LightGrid`].
pub struct LightSampler<'a> {
    lights: &'a [PointLight],
}

impl<'a> LightSampler<'a> {
    /// Build a sampler over all lights.
    pub fn new(lights: &'a [PointLight]) -> Self {
        Self { lights }
    }

    /// Number of lights this sampler covers.
    pub fn len(&self) -> usize {
        self.lights.len()
    }

    /// Whether there are no lights.
    pub fn is_empty(&self) -> bool {
        self.lights.is_empty()
    }

    /// Single weighted-reservoir sample over **all** lights (the `full` path).
    ///
    /// Returns `None` only if every candidate has zero target (no light
    /// reaches the point) or there are no lights.
    pub fn sample(&self, shade_point: Vec3, rng_seed: u64) -> Option<LightSample> {
        let mut rng = Lcg::new(rng_seed);
        self.reservoir(shade_point, 0..self.lights.len(), &mut rng)
    }

    /// `N` independent reservoir samples (the per-pixel candidate-count knob).
    /// Each reservoir uses a decorrelated sub-seed.
    pub fn sample_n(&self, shade_point: Vec3, rng_seed: u64, n: usize) -> Vec<LightSample> {
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let sub_seed = rng_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(k as u64);
            if let Some(s) = self.sample(shade_point, sub_seed) {
                out.push(s);
            }
        }
        out
    }

    /// Reservoir sample restricted to the candidate indices yielded by `iter`
    /// (the `grid` path drives this with grid-culled candidates).
    pub fn sample_indices<I>(
        &self,
        shade_point: Vec3,
        candidates: I,
        rng_seed: u64,
    ) -> Option<LightSample>
    where
        I: Iterator<Item = usize>,
    {
        let mut rng = Lcg::new(rng_seed);
        self.reservoir(shade_point, candidates, &mut rng)
    }

    /// Core WRS / RIS reservoir over the supplied candidate indices.
    fn reservoir<I>(
        &self,
        shade_point: Vec3,
        candidates: I,
        rng: &mut Lcg,
    ) -> Option<LightSample>
    where
        I: Iterator<Item = usize>,
    {
        let mut w_sum = 0.0f32; // running Σ p̂_j
        let mut m = 0usize; // candidate count M
        let mut chosen: Option<usize> = None;
        let mut chosen_target = 0.0f32;

        for i in candidates {
            let p_hat = rgb_target(&self.lights[i], shade_point);
            m += 1;
            if p_hat <= 0.0 {
                continue; // zero-contribution lights can never be selected
            }
            w_sum += p_hat;
            // Keep candidate i with probability p̂_i / w_sum.
            if rng.next_unit() * w_sum <= p_hat {
                chosen = Some(i);
                chosen_target = p_hat;
            }
        }

        let s = chosen?;
        // RIS weight W = (1/p̂_s) · (w_sum / M).
        let weight = (1.0 / chosen_target) * (w_sum / m as f32);
        Some(LightSample {
            light_index: s,
            weight,
            target: chosen_target,
            m,
        })
    }
}

/// Spectral weighted reservoir sampler — identical WRS/RIS machinery, but the
/// target function uses summed 16-band power instead of RGB luminance.
pub struct SpectralLightSampler<'a> {
    lights: &'a [SpectralPointLight],
}

impl<'a> SpectralLightSampler<'a> {
    pub fn new(lights: &'a [SpectralPointLight]) -> Self {
        Self { lights }
    }

    pub fn len(&self) -> usize {
        self.lights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lights.is_empty()
    }

    /// Single spectral reservoir sample over all lights.
    pub fn sample(&self, shade_point: Vec3, rng_seed: u64) -> Option<LightSample> {
        let mut rng = Lcg::new(rng_seed);
        let mut w_sum = 0.0f32;
        let mut m = 0usize;
        let mut chosen: Option<usize> = None;
        let mut chosen_target = 0.0f32;

        for (i, l) in self.lights.iter().enumerate() {
            let p_hat = l.target(shade_point);
            m += 1;
            if p_hat <= 0.0 {
                continue;
            }
            w_sum += p_hat;
            if rng.next_unit() * w_sum <= p_hat {
                chosen = Some(i);
                chosen_target = p_hat;
            }
        }

        let s = chosen?;
        let weight = (1.0 / chosen_target) * (w_sum / m as f32);
        Some(LightSample {
            light_index: s,
            weight,
            target: chosen_target,
            m,
        })
    }
}

/// Uniform spatial grid over light positions, so candidate gathering for a
/// shade point is O(cell + neighbourhood) instead of O(lights).
///
/// Cell size is the average light radius; a light is inserted into every cell
/// its `radius`-sphere overlaps, so [`relevant_lights`](Self::relevant_lights)
/// returns a superset of all lights that can possibly reach a point in a cell.
pub struct LightGrid {
    cell_size: f32,
    /// Hash map from integer cell coord → light indices touching that cell.
    cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
}

impl LightGrid {
    /// Build a grid over `lights`. Cell size is the mean radius (clamped to a
    /// small positive floor so a scene of zero-radius lights still builds).
    pub fn build(lights: &[PointLight]) -> Self {
        let cell_size = if lights.is_empty() {
            1.0
        } else {
            let mean_r: f32 =
                lights.iter().map(|l| l.radius).sum::<f32>() / lights.len() as f32;
            mean_r.max(1e-3)
        };

        let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::new();

        for (i, l) in lights.iter().enumerate() {
            // Cells overlapping the light's bounding box [pos - r, pos + r].
            let min = l.position - Vec3::splat(l.radius);
            let max = l.position + Vec3::splat(l.radius);
            let (x0, y0, z0) = Self::cell_of(min, cell_size);
            let (x1, y1, z1) = Self::cell_of(max, cell_size);
            for cx in x0..=x1 {
                for cy in y0..=y1 {
                    for cz in z0..=z1 {
                        cells.entry((cx, cy, cz)).or_default().push(i);
                    }
                }
            }
        }

        Self { cell_size, cells }
    }

    #[inline]
    fn cell_of(p: Vec3, cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x / cell_size).floor() as i32,
            (p.y / cell_size).floor() as i32,
            (p.z / cell_size).floor() as i32,
        )
    }

    /// Cell size in world units.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Light indices whose radius reaches the cell containing `shade_point`.
    ///
    /// This is a (possibly strict) superset of the lights with nonzero
    /// attenuation at the point: every light with nonzero attenuation is
    /// guaranteed to be present, because such a light's sphere overlaps the
    /// point's cell.
    pub fn relevant_lights(&self, shade_point: Vec3) -> impl Iterator<Item = usize> + '_ {
        let cell = Self::cell_of(shade_point, self.cell_size);
        self.cells
            .get(&cell)
            .into_iter()
            .flat_map(|v| v.iter().copied())
    }
}

/// Monte-Carlo radiance estimate via stochastic light selection.
///
/// Sums `N` reservoir samples, each contributing
/// `f(s) · W` where `f` is the diffuse `N·L` shading of the chosen light and
/// `W` its RIS weight. Converges to the brute-force Σ over all lights.
///
/// `normal` is the surface normal; pass `Vec3::ZERO` to disable the cosine
/// term (pure radiance, used by the unbiasedness reference).
pub fn estimate_radiance(
    shade_point: Vec3,
    normal: Vec3,
    lights: &[PointLight],
    samples: usize,
    seed: u64,
) -> [f32; 3] {
    if lights.is_empty() || samples == 0 {
        return [0.0; 3];
    }
    let sampler = LightSampler::new(lights);
    let mut acc = [0.0f32; 3];

    for k in 0..samples {
        let sub_seed = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(k as u64);
        if let Some(s) = sampler.sample(shade_point, sub_seed) {
            // weight·M = (1/p̂_s)·Σp̂ — the canonical RIS sum-estimator, so the
            // running average over N samples converges to Σ_i f_i (brute force).
            let w = s.weight * s.m as f32;
            let f = shade_contribution(&lights[s.light_index], shade_point, normal);
            acc[0] += f[0] * w;
            acc[1] += f[1] * w;
            acc[2] += f[2] * w;
        }
    }

    let inv_n = 1.0 / samples as f32;
    [acc[0] * inv_n, acc[1] * inv_n, acc[2] * inv_n]
}

/// Full RGB shading contribution `f(light)` of one light at a shade point:
/// `attenuation(distance) × color × max(N·L, 0)`. With `normal == ZERO` the
/// cosine term is 1 (pure radiance), matching the sampler's scalar target so
/// the brute-force reference is `Σ_i f_i`.
fn shade_contribution(light: &PointLight, shade_point: Vec3, normal: Vec3) -> [f32; 3] {
    let d = light.position.distance(shade_point);
    let att = light.attenuation(d);
    let cos = if normal == Vec3::ZERO {
        1.0
    } else {
        let to_light = (light.position - shade_point).normalize_or_zero();
        normal.dot(to_light).max(0.0)
    };
    let s = att * cos;
    [light.color[0] * s, light.color[1] * s, light.color[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic scene of `n` lights, positions/colours from a seeded LCG.
    fn make_scene(n: usize, seed: u64) -> Vec<PointLight> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let px = (rng.next_unit() - 0.5) * 20.0;
                let py = (rng.next_unit() - 0.5) * 20.0;
                let pz = (rng.next_unit() - 0.5) * 20.0;
                PointLight {
                    position: Vec3::new(px, py, pz),
                    color: [
                        0.2 + rng.next_unit(),
                        0.2 + rng.next_unit(),
                        0.2 + rng.next_unit(),
                    ],
                    intensity: 0.5 + rng.next_unit() * 2.0,
                    radius: 8.0 + rng.next_unit() * 12.0,
                }
            })
            .collect()
    }

    /// Brute-force reference radiance: Σ over all lights of full contribution.
    fn brute_force(shade_point: Vec3, normal: Vec3, lights: &[PointLight]) -> [f32; 3] {
        let mut acc = [0.0f32; 3];
        for l in lights {
            let f = super::shade_contribution(l, shade_point, normal);
            acc[0] += f[0];
            acc[1] += f[1];
            acc[2] += f[2];
        }
        acc
    }

    /// LOAD-BEARING: 4096-sample reservoir estimate matches brute force within
    /// 2% relative error per RGB channel over a 64-light scene.
    #[test]
    fn unbiasedness_matches_brute_force_within_2pct() {
        let lights = make_scene(64, 0xABCD_1234);
        let shade_point = Vec3::new(1.0, 2.0, -3.0);
        let normal = Vec3::ZERO; // pure radiance reference

        let reference = brute_force(shade_point, normal, &lights);
        let estimate = estimate_radiance(shade_point, normal, &lights, 4096, 0x5151_5151);

        for c in 0..3 {
            let rel = (estimate[c] - reference[c]).abs() / reference[c].max(1e-6);
            assert!(
                rel < 0.02,
                "channel {c}: estimate {} vs reference {} rel err {:.4} >= 2%",
                estimate[c],
                reference[c],
                rel
            );
        }
    }

    /// A light beyond its attenuation radius must never be selected.
    #[test]
    fn zero_contribution_light_never_selected() {
        // Two reachable lights near the point, one far beyond its radius.
        let lights = vec![
            PointLight {
                position: Vec3::new(1.0, 0.0, 0.0),
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
                radius: 5.0,
            },
            PointLight {
                position: Vec3::new(-1.0, 0.0, 0.0),
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
                radius: 5.0,
            },
            // 100 units away with radius 5 → attenuation 0 at origin.
            PointLight {
                position: Vec3::new(100.0, 0.0, 0.0),
                color: [1.0, 1.0, 1.0],
                intensity: 10.0,
                radius: 5.0,
            },
        ];
        let far_index = 2;
        let sampler = LightSampler::new(&lights);
        for k in 0..1000u64 {
            let s = sampler
                .sample(Vec3::ZERO, k.wrapping_mul(2654435761).wrapping_add(7))
                .expect("a reachable light should always be chosen");
            assert_ne!(
                s.light_index, far_index,
                "zero-attenuation light {far_index} selected on draw {k}"
            );
        }
    }

    /// Grid path estimate matches full path within 3%; relevant_lights is a
    /// strict subset that contains every nonzero-attenuation light.
    #[test]
    fn grid_culling_matches_full_and_is_correct_subset() {
        let lights = make_scene(64, 0x7777_0001);
        let shade_point = Vec3::new(2.0, -1.0, 4.0);
        let normal = Vec3::ZERO;
        let grid = LightGrid::build(&lights);

        // --- subset / completeness check ---
        let relevant: std::collections::HashSet<usize> =
            grid.relevant_lights(shade_point).collect();
        // Every light with nonzero attenuation must be present.
        let mut nonzero = std::collections::HashSet::new();
        for (i, l) in lights.iter().enumerate() {
            if l.attenuation(l.position.distance(shade_point)) > 0.0 {
                nonzero.insert(i);
                assert!(
                    relevant.contains(&i),
                    "light {i} has nonzero attenuation but was culled by grid"
                );
            }
        }
        // Subset must be strict (the scene spans 20 units, lights radius < 20,
        // so some lights cannot reach this point).
        assert!(
            relevant.len() < lights.len(),
            "grid relevant set ({}) is not a strict subset of all {} lights",
            relevant.len(),
            lights.len()
        );
        assert!(
            !nonzero.is_empty(),
            "test scene degenerate: no light reaches the shade point"
        );

        // --- grid vs full estimate agreement ---
        let sampler = LightSampler::new(&lights);
        let samples = 8192usize;
        let mut full_acc = [0.0f32; 3];
        let mut grid_acc = [0.0f32; 3];
        for k in 0..samples {
            let sub = (k as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(11);
            if let Some(s) = sampler.sample(shade_point, sub) {
                let w = s.weight * s.m as f32;
                let f = super::shade_contribution(&lights[s.light_index], shade_point, normal);
                for c in 0..3 {
                    full_acc[c] += f[c] * w;
                }
            }
            if let Some(s) =
                sampler.sample_indices(shade_point, grid.relevant_lights(shade_point), sub)
            {
                let w = s.weight * s.m as f32;
                let f = super::shade_contribution(&lights[s.light_index], shade_point, normal);
                for c in 0..3 {
                    grid_acc[c] += f[c] * w;
                }
            }
        }
        let inv = 1.0 / samples as f32;
        for c in 0..3 {
            let full = full_acc[c] * inv;
            let grid_v = grid_acc[c] * inv;
            let rel = (grid_v - full).abs() / full.max(1e-6);
            assert!(
                rel < 0.03,
                "channel {c}: grid {grid_v} vs full {full} rel err {:.4} >= 3%",
                rel
            );
        }
    }

    /// Same seed → identical sample sequence (indices AND weights bit-equal).
    #[test]
    fn determinism_same_seed_identical_sequence() {
        let lights = make_scene(48, 0x1357_9BDF);
        let shade_point = Vec3::new(0.5, 0.5, 0.5);
        let sampler = LightSampler::new(&lights);
        let a = sampler.sample_n(shade_point, 0xDEAD_BEEF, 200);
        let b = sampler.sample_n(shade_point, 0xDEAD_BEEF, 200);
        assert_eq!(a.len(), b.len(), "sample counts differ");
        for (i, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(sa.light_index, sb.light_index, "index mismatch at {i}");
            assert_eq!(
                sa.weight.to_bits(),
                sb.weight.to_bits(),
                "weight not bit-equal at {i}"
            );
        }
    }

    /// Spectral path: a 10× total-power light is chosen in >=70% of draws over
    /// an equal-distance 1× light.
    #[test]
    fn spectral_prefers_higher_band_power() {
        let bright: [f32; SPECTRAL_BANDS] = [1.0; SPECTRAL_BANDS]; // Σ = 16
        let dim: [f32; SPECTRAL_BANDS] = [0.1; SPECTRAL_BANDS]; // Σ = 1.6 (10× less)
        let lights = vec![
            SpectralPointLight {
                light: PointLight {
                    position: Vec3::new(3.0, 0.0, 0.0),
                    color: [1.0, 1.0, 1.0],
                    intensity: 1.0,
                    radius: 10.0,
                },
                power: bright,
            },
            SpectralPointLight {
                light: PointLight {
                    position: Vec3::new(-3.0, 0.0, 0.0), // same distance from origin
                    color: [1.0, 1.0, 1.0],
                    intensity: 1.0,
                    radius: 10.0,
                },
                power: dim,
            },
        ];
        let sampler = SpectralLightSampler::new(&lights);
        let mut bright_count = 0;
        for k in 0..1000u64 {
            let s = sampler
                .sample(Vec3::ZERO, k.wrapping_mul(2862933555777941757).wrapping_add(3))
                .expect("a light should be chosen");
            if s.light_index == 0 {
                bright_count += 1;
            }
        }
        assert!(
            bright_count >= 700,
            "10x-power light chosen only {bright_count}/1000 draws (< 70%)"
        );
    }

    /// Single-light degenerate case: one sample equals the analytic
    /// contribution exactly (within f32 epsilon), because W·p̂ = full
    /// contribution and only one light exists.
    #[test]
    fn single_light_one_sample_equals_brute_force() {
        let lights = vec![PointLight {
            position: Vec3::new(2.0, 3.0, 1.0),
            color: [0.7, 0.5, 0.9],
            intensity: 1.3,
            radius: 9.0,
        }];
        let shade_point = Vec3::new(-1.0, 0.5, 0.0);
        let normal = Vec3::ZERO;

        let reference = brute_force(shade_point, normal, &lights);
        let estimate = estimate_radiance(shade_point, normal, &lights, 1, 0x9999);

        for c in 0..3 {
            assert!(
                (estimate[c] - reference[c]).abs() <= 1e-5 * reference[c].max(1.0),
                "channel {c}: 1-sample estimate {} != analytic {}",
                estimate[c],
                reference[c]
            );
        }
    }
}
