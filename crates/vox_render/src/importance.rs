//! Importance-pruning for Gaussian splat assets (HGSC/HPC-inspired).
//!
//! This is an **offline / asset-time** optimization: it scores each splat by its
//! visual contribution, then drops the lowest-importance splats to shrink an
//! asset while preserving as much perceptual fidelity (and spectral energy) as
//! possible. It is the complement of [`crate::atom_budget`], which selects an
//! LOD *at frame time* against a per-frame atom budget. Pruning happens once,
//! when an asset is authored/exported; atom-budget selection happens every frame
//! on whatever the asset shipped. They compose: prune offline to a smaller asset,
//! then let the runtime budget further trim per frame.
//!
//! # Scoring formula
//!
//! Inspired by HGSC's per-Gaussian importance scores (drop low-importance
//! Gaussians, octree/KD-tree compress the rest). For each splat `i` we compute
//!
//! ```text
//!   score_i = opacity_i * size_i * energy_i * redundancy_i
//! ```
//!
//! where each factor is a real, splat-native quantity:
//!
//! - `opacity_i`   = `opacity / 255` in `[0, 1]` — a fully transparent splat
//!   contributes nothing and scores 0.
//! - `size_i`      = a projected-extent proxy. For 3DGS volumes it is the
//!   geometric-mean cross-section `(s_u*s_v + s_v*s_w + s_u*s_w)/3` (proportional
//!   to the silhouette area of the ellipsoid); for 2DGS surfaces it is the disk
//!   area `s_u*s_v`. Bigger splats cover more pixels, so they matter more.
//! - `energy_i`    = sum of the 16 spectral bands (decoded from f16). This is the
//!   radiant energy the splat injects into the framebuffer; a black splat
//!   (all-zero spectrum) scores 0 regardless of size.
//! - `redundancy_i` = a local-redundancy *down-weight* in `(0, 1]`. A splat that
//!   sits in a dense cluster of similarly-colored neighbors is largely redundant
//!   (its neighbors already paint that region), so it is cheaper to drop. We
//!   measure it with a uniform spatial grid (cell size = a multiple of the
//!   median splat size) so neighbor queries are O(n) in practice. For splat `i`
//!   with `k` color-similar neighbors in its 27-cell neighborhood,
//!   `redundancy_i = 1 / (1 + REDUNDANCY_WEIGHT * k_similar)`. An isolated splat
//!   (`k_similar = 0`) keeps full weight 1.0; a splat buried in a uniform cluster
//!   is strongly suppressed.
//!
//! "Color-similar" uses the cosine similarity of the two splats' 16-band spectra
//! (≥ [`COLOR_SIMILARITY_THRESHOLD`]); position proximity uses a world-space
//! radius derived from the two splats' sizes. Both conditions must hold for a
//! neighbor to count as redundant.
//!
//! All arithmetic is deterministic (fixed iteration order, no RNG, no parallel
//! reduction over floats), so the same input + params yields byte-identical
//! output.

use vox_core::types::GaussianSplat;

/// Down-weight strength for redundant neighbors. Each color-similar neighbor
/// divides the splat's score by roughly this much.
pub const REDUNDANCY_WEIGHT: f32 = 0.5;

/// Cosine-similarity threshold (on the 16-band spectrum) above which two splats
/// are considered "the same color" for redundancy purposes.
pub const COLOR_SIMILARITY_THRESHOLD: f32 = 0.98;

/// Result of a prune operation.
#[derive(Debug, Clone)]
pub struct PruneResult {
    /// The splats that survived pruning, in their original relative order.
    pub kept: Vec<GaussianSplat>,
    /// How many splats were removed.
    pub removed: usize,
    /// Spectral energy retained: `sum(energy of kept) / sum(energy of original)`
    /// in `[0, 1]`. A real fidelity proxy — 1.0 means no radiant energy was lost.
    pub energy_retained: f32,
}

/// Decode the summed spectral energy (Σ of the 16 f16 bands) of a splat.
#[inline]
fn spectral_energy(splat: &GaussianSplat) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..GaussianSplat::BANDS {
        sum += splat.spectral_f32(b).max(0.0);
    }
    sum
}

/// Projected-size proxy: silhouette-ish cross-section area.
#[inline]
fn size_proxy(splat: &GaussianSplat) -> f32 {
    let s = splat.scales();
    let (su, sv, sw) = (s[0].abs(), s[1].abs(), s[2].abs());
    if splat.is_surface() {
        // 2DGS disk: area ~ su * sv (sw is unused / zero).
        su * sv
    } else {
        // 3DGS ellipsoid: mean of the three principal cross-section areas.
        (su * sv + sv * sw + su * sw) / 3.0
    }
}

/// Cosine similarity of two splats' 16-band spectra in `[-1, 1]` (≈1 = same hue).
#[inline]
fn spectral_cosine(a: &GaussianSplat, b: &GaussianSplat) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..GaussianSplat::BANDS {
        let va = a.spectral_f32(i);
        let vb = b.spectral_f32(i);
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    let denom = (na.sqrt()) * (nb.sqrt());
    if denom <= 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Uniform spatial hash grid over splat centroids for O(1)-amortized neighbor
/// queries. Cell size is chosen from the median splat size so neighborhoods stay
/// bounded.
struct SpatialGrid {
    inv_cell: f32,
    min: [f32; 3],
    // cell key -> list of splat indices
    buckets: std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn cell_of(&self, p: [f32; 3]) -> (i32, i32, i32) {
        (
            ((p[0] - self.min[0]) * self.inv_cell).floor() as i32,
            ((p[1] - self.min[1]) * self.inv_cell).floor() as i32,
            ((p[2] - self.min[2]) * self.inv_cell).floor() as i32,
        )
    }

    fn build(splats: &[GaussianSplat], cell: f32) -> Self {
        let mut min = [f32::INFINITY; 3];
        for s in splats {
            let p = s.position();
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
            }
        }
        if !min[0].is_finite() {
            min = [0.0; 3];
        }
        let cell = cell.max(1e-4);
        let mut grid = SpatialGrid {
            inv_cell: 1.0 / cell,
            min,
            buckets: std::collections::HashMap::new(),
        };
        for (i, s) in splats.iter().enumerate() {
            let key = grid.cell_of(s.position());
            grid.buckets.entry(key).or_default().push(i);
        }
        grid
    }

    /// Indices in the 27-cell neighborhood of `p` (excluding nothing; caller
    /// filters self). Returned in deterministic (cell-sorted) order.
    fn neighbors(&self, p: [f32; 3]) -> Vec<usize> {
        let (cx, cy, cz) = self.cell_of(p);
        let mut out = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(b) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        out.extend_from_slice(b);
                    }
                }
            }
        }
        out
    }
}

/// Per-splat importance scores (HGSC-style), one per input splat, same order.
///
/// `score = opacity * size * energy * redundancy_down_weight`. See the module
/// docs for the precise definition of each factor. Returns an empty vec for an
/// empty input.
pub fn importance_scores(splats: &[GaussianSplat]) -> Vec<f32> {
    let n = splats.len();
    if n == 0 {
        return Vec::new();
    }

    // Cell size = 2x the median splat size proxy (in linear units), floored.
    let mut sizes: Vec<f32> = splats
        .iter()
        .map(|s| {
            let p = size_proxy(s);
            // size_proxy is an area; use its sqrt as a linear extent for spacing.
            p.max(0.0).sqrt()
        })
        .collect();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_extent = sizes[n / 2].max(1e-3);
    let cell = (median_extent * 4.0).max(1e-3);

    let grid = SpatialGrid::build(splats, cell);

    let mut scores = Vec::with_capacity(n);
    for (i, s) in splats.iter().enumerate() {
        let opacity = s.opacity() as f32 / 255.0;
        let size = size_proxy(s);
        let energy = spectral_energy(s);

        // Count color-similar, spatially-close neighbors.
        let pi = s.position();
        let ri = size_proxy(s).sqrt().max(1e-3);
        let mut similar = 0u32;
        for &j in grid.neighbors(pi).iter() {
            if j == i {
                continue;
            }
            let sj = &splats[j];
            let pj = sj.position();
            let dist2 = (pi[0] - pj[0]).powi(2) + (pi[1] - pj[1]).powi(2) + (pi[2] - pj[2]).powi(2);
            let rj = size_proxy(sj).sqrt().max(1e-3);
            let reach = ri + rj;
            if dist2 > reach * reach {
                continue;
            }
            if spectral_cosine(s, sj) >= COLOR_SIMILARITY_THRESHOLD {
                similar += 1;
            }
        }
        let redundancy = 1.0 / (1.0 + REDUNDANCY_WEIGHT * similar as f32);

        scores.push(opacity * size * energy * redundancy);
    }
    scores
}

/// What to prune against.
#[derive(Debug, Clone, Copy)]
pub enum PruneTarget {
    /// Keep this fraction of splats (e.g. 0.5 = keep the most-important half).
    KeepFraction(f32),
    /// Keep every splat whose importance score is at least this absolute floor.
    QualityFloor(f32),
}

/// Total positive spectral energy of a splat set.
fn total_energy(splats: &[GaussianSplat]) -> f32 {
    splats.iter().map(spectral_energy).sum()
}

/// Prune the lowest-importance splats.
///
/// Determinism: scores are computed deterministically, and ties are broken by
/// original index (stable), so identical input + target produces byte-identical
/// `kept`.
pub fn prune(splats: &[GaussianSplat], target: PruneTarget) -> PruneResult {
    let n = splats.len();
    if n == 0 {
        return PruneResult {
            kept: Vec::new(),
            removed: 0,
            energy_retained: 1.0,
        };
    }
    let scores = importance_scores(splats);
    let orig_energy = total_energy(splats);

    // Indices sorted by (score desc, index asc) — deterministic.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let keep_set: std::collections::BTreeSet<usize> = match target {
        PruneTarget::KeepFraction(f) => {
            let f = f.clamp(0.0, 1.0);
            // Round to nearest; keep at least 1 if fraction > 0 and we have splats.
            let mut k = (f * n as f32).round() as usize;
            if f > 0.0 {
                k = k.max(1);
            }
            k = k.min(n);
            order.iter().take(k).copied().collect()
        }
        PruneTarget::QualityFloor(floor) => order
            .iter()
            .copied()
            .filter(|&i| scores[i] >= floor)
            .collect(),
    };

    // Emit kept splats in their ORIGINAL order (stable, deterministic).
    let kept: Vec<GaussianSplat> = (0..n)
        .filter(|i| keep_set.contains(i))
        .map(|i| splats[i])
        .collect();

    let kept_energy = total_energy(&kept);
    let energy_retained = if orig_energy > 0.0 {
        (kept_energy / orig_energy).clamp(0.0, 1.0)
    } else {
        1.0
    };

    PruneResult {
        removed: n - kept.len(),
        kept,
        energy_retained,
    }
}

/// Mean per-pixel absolute RGB difference between two equally-sized framebuffers,
/// normalized to `[0, 1]` (averaged over all pixels and the 3 color channels).
pub fn mean_pixel_diff(
    a: &crate::gpu::software_rasteriser::Framebuffer,
    b: &crate::gpu::software_rasteriser::Framebuffer,
) -> f32 {
    debug_assert_eq!(a.pixels.len(), b.pixels.len());
    if a.pixels.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for (pa, pb) in a.pixels.iter().zip(b.pixels.iter()) {
        for c in 0..3 {
            acc += (pa[c] as f64 - pb[c] as f64).abs();
        }
    }
    let n = (a.pixels.len() * 3) as f64;
    (acc / n / 255.0) as f32
}

/// Render-guarded prune.
///
/// Prunes toward `target_fraction`, but verifies the result against the original
/// by rendering both at `camera` through the [`SoftwareRasteriser`] and measuring
/// [`mean_pixel_diff`]. If the diff exceeds `max_mean_pixel_diff` (i.e. pruning
/// visibly hollowed the scene), the guard **backs off**: it keeps progressively
/// more splats until the diff falls under the bound.
///
/// # Back-off design
///
/// We do a deterministic ascending search on the keep-fraction. Starting from
/// `target_fraction`, if the render diff is over budget we step the fraction up
/// by [`BACKOFF_STEP`] (0.1) and re-evaluate, until either the diff is within
/// bound or the fraction reaches 1.0 (keep everything — diff is then 0 by
/// construction, since pruning nothing cannot change the image). The first
/// fraction that satisfies the bound wins. This is monotone in practice (keeping
/// more importance-ranked splats only reduces the diff) and always terminates in
/// at most `ceil((1 - target)/step) + 1` renders.
///
/// The returned [`PruneResult`] is the pruned set at the accepted fraction; the
/// accepted fraction is recoverable from `kept.len()`.
pub fn prune_with_render_guard(
    splats: &[GaussianSplat],
    target_fraction: f32,
    camera: &crate::spectral::RenderCamera,
    max_mean_pixel_diff: f32,
) -> PruneResult {
    use crate::gpu::software_rasteriser::SoftwareRasteriser;
    use vox_core::spectral::Illuminant;

    const BACKOFF_STEP: f32 = 0.1;
    // Fixed guard render resolution. Small enough to be cheap, large enough to
    // catch hollowing. Deterministic.
    const GUARD_W: u32 = 96;
    const GUARD_H: u32 = 96;

    if splats.is_empty() {
        return prune(splats, PruneTarget::KeepFraction(target_fraction));
    }

    let illum = Illuminant::d65();
    let mut ras = SoftwareRasteriser::new(GUARD_W, GUARD_H);
    let reference = ras.render_gaussian(splats, camera, &illum, None);

    let mut frac = target_fraction.clamp(0.0, 1.0);
    loop {
        let result = prune(splats, PruneTarget::KeepFraction(frac));
        let rendered = ras.render_gaussian(&result.kept, camera, &illum, None);
        let diff = mean_pixel_diff(&reference, &rendered);
        if diff <= max_mean_pixel_diff || frac >= 1.0 {
            return result;
        }
        frac = (frac + BACKOFF_STEP).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::RenderCamera;
    use glam::{Mat4, Quat, Vec3};
    use half::f16;

    fn flat_spectral(value: f32) -> [u16; 16] {
        [f16::from_f32(value).to_bits(); 16]
    }

    /// A scene of 1000 redundant overlapping low-opacity splats clustered tightly
    /// at the origin, plus 100 distinct high-opacity splats spread far apart.
    /// Returns (splats, distinct_positions).
    fn redundant_plus_distinct_scene() -> (Vec<GaussianSplat>, Vec<[f32; 3]>) {
        let mut splats = Vec::new();
        // 1000 redundant: tightly clustered near origin, same dim color, low opacity.
        // Deterministic pseudo-jitter via index arithmetic (no RNG).
        for i in 0..1000u32 {
            let fx = ((i.wrapping_mul(2654435761)) % 1000) as f32 / 1000.0 - 0.5;
            let fy = ((i.wrapping_mul(40503)) % 1000) as f32 / 1000.0 - 0.5;
            let fz = ((i.wrapping_mul(2246822519)) % 1000) as f32 / 1000.0 - 0.5;
            let pos = [fx * 0.2, fy * 0.2, fz * 0.2];
            splats.push(GaussianSplat::volume(
                pos,
                [0.05, 0.05, 0.05],
                Quat::IDENTITY,
                20, // low opacity
                flat_spectral(0.1),
            ));
        }
        // 100 distinct: spread on a wide grid, opaque, bright.
        let mut distinct_positions = Vec::new();
        for i in 0..100u32 {
            let gx = (i % 10) as f32 * 5.0 - 22.5;
            let gy = (i / 10) as f32 * 5.0 - 22.5;
            let pos = [gx, gy, 10.0 + (i as f32) * 0.01];
            distinct_positions.push(pos);
            splats.push(GaussianSplat::volume(
                pos,
                [0.5, 0.5, 0.5],
                Quat::IDENTITY,
                255, // opaque
                flat_spectral(1.0),
            ));
        }
        (splats, distinct_positions)
    }

    #[test]
    fn prune_keeps_distinct_drops_redundant() {
        let (splats, distinct) = redundant_plus_distinct_scene();
        let result = prune(&splats, PruneTarget::KeepFraction(0.2));

        // Identify how many of the 100 distinct positions survived.
        let kept_positions: std::collections::HashSet<[u32; 3]> = result
            .kept
            .iter()
            .map(|s| {
                let p = s.position();
                [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
            })
            .collect();
        let distinct_kept = distinct
            .iter()
            .filter(|p| kept_positions.contains(&[p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]))
            .count();

        // Keep fraction 0.2 of 1100 = 220 kept, 880 removed.
        assert_eq!(result.kept.len(), 220, "keep 0.2 of 1100");
        assert_eq!(result.removed, 880);
        assert!(
            distinct_kept >= 90,
            "should keep >=90 of the 100 distinct splats, kept {distinct_kept}"
        );
        // The removed splats should be dominated by the redundant cluster:
        // at most 10 distinct were dropped, so >=870 of the 880 removed are redundant.
        let distinct_dropped = 100 - distinct_kept;
        assert!(
            distinct_dropped <= 10,
            "at most 10 distinct dropped, dropped {distinct_dropped}"
        );
        println!(
            "[prune_keeps_distinct_drops_redundant] kept={} removed={} distinct_kept={}/100 distinct_dropped={} energy_retained={:.4}",
            result.kept.len(),
            result.removed,
            distinct_kept,
            distinct_dropped,
            result.energy_retained
        );
    }

    #[test]
    fn energy_retained_exceeds_half_on_uniform_scene() {
        // Uniform scene: all splats identical except spectral brightness varies,
        // so importance ranks the brightest. Keeping the top half must retain
        // STRICTLY more than half the energy.
        let mut splats = Vec::new();
        for i in 0..200u32 {
            // Spread positions so redundancy doesn't dominate; brightness ramps.
            let pos = [i as f32 * 2.0, 0.0, 0.0];
            let brightness = 0.1 + (i as f32 / 200.0); // 0.1 .. 1.1
            splats.push(GaussianSplat::volume(
                pos,
                [0.3, 0.3, 0.3],
                Quat::IDENTITY,
                255,
                flat_spectral(brightness),
            ));
        }
        let result = prune(&splats, PruneTarget::KeepFraction(0.5));
        assert_eq!(result.kept.len(), 100);
        assert!(
            result.energy_retained > 0.5 && result.energy_retained <= 1.0,
            "energy_retained must be in (0.5, 1.0], got {}",
            result.energy_retained
        );
        println!(
            "[energy_retained_exceeds_half_on_uniform_scene] energy_retained={:.6} (kept brightest 100/200)",
            result.energy_retained
        );
    }

    fn wall_camera() -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 500.0),
        }
    }

    /// A flat "wall" of overlapping splats filling the view. Naive 0.1 pruning
    /// hollows it (mean pixel diff over bound); the guard backs off.
    fn wall_scene() -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        // 20x20 grid of overlapping opaque white splats across the view plane.
        for gy in 0..20 {
            for gx in 0..20 {
                let x = gx as f32 * 0.25 - 2.375;
                let y = gy as f32 * 0.25 - 2.375;
                splats.push(GaussianSplat::volume(
                    [x, y, 0.0],
                    [0.22, 0.22, 0.22],
                    Quat::IDENTITY,
                    255,
                    flat_spectral(1.0),
                ));
            }
        }
        splats
    }

    #[test]
    fn render_guard_backs_off_when_hollowing() {
        let splats = wall_scene();
        let cam = wall_camera();
        let bound = 0.05;

        // Naive aggressive prune to 0.1 — measure its diff to confirm it hollows.
        use crate::gpu::software_rasteriser::SoftwareRasteriser;
        use vox_core::spectral::Illuminant;
        let illum = Illuminant::d65();
        let mut ras = SoftwareRasteriser::new(96, 96);
        let reference = ras.render_gaussian(&splats, &cam, &illum, None);
        let naive = prune(&splats, PruneTarget::KeepFraction(0.1));
        let naive_fb = ras.render_gaussian(&naive.kept, &cam, &illum, None);
        let naive_diff = mean_pixel_diff(&reference, &naive_fb);

        // Guarded prune backs off.
        let guarded = prune_with_render_guard(&splats, 0.1, &cam, bound);
        let guarded_fb = ras.render_gaussian(&guarded.kept, &cam, &illum, None);
        let guarded_diff = mean_pixel_diff(&reference, &guarded_fb);
        let final_fraction = guarded.kept.len() as f32 / splats.len() as f32;

        assert!(
            naive_diff > bound,
            "naive 0.1 prune should hollow the wall: diff {naive_diff} must exceed bound {bound}"
        );
        assert!(
            final_fraction > 0.1,
            "guard should back off above 0.1, final fraction {final_fraction}"
        );
        assert!(
            guarded_diff <= bound,
            "guarded diff {guarded_diff} must be within bound {bound}"
        );
        println!(
            "[render_guard_backs_off_when_hollowing] naive(0.1) diff={naive_diff:.5} > bound={bound} | guarded final_fraction={final_fraction:.3} diff={guarded_diff:.5} <= bound"
        );
    }

    #[test]
    fn determinism_byte_identical() {
        let (splats, _) = redundant_plus_distinct_scene();
        let a = prune(&splats, PruneTarget::KeepFraction(0.37));
        let b = prune(&splats, PruneTarget::KeepFraction(0.37));
        assert_eq!(a.kept.len(), b.kept.len());
        // Byte-compare the kept splats.
        let ab: &[u8] = bytemuck::cast_slice(&a.kept);
        let bb: &[u8] = bytemuck::cast_slice(&b.kept);
        assert_eq!(ab, bb, "two prune runs must be byte-identical");
        assert_eq!(a.removed, b.removed);
        assert_eq!(a.energy_retained.to_bits(), b.energy_retained.to_bits());
        println!(
            "[determinism_byte_identical] two runs byte-identical: {} kept, {} bytes",
            a.kept.len(),
            ab.len()
        );
    }

    #[test]
    fn empty_input_is_safe() {
        let result = prune(&[], PruneTarget::KeepFraction(0.5));
        assert_eq!(result.kept.len(), 0);
        assert_eq!(result.removed, 0);
        assert_eq!(result.energy_retained, 1.0);
    }
}
