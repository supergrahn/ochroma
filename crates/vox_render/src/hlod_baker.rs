//! HLOD baker — K-Means++ clustering to produce multi-level splat LOD representations.
//! HLODBaker::bake_cell() reduces a cell's full splat set to hierarchical levels.

use glam::{self, Vec3};
use half::f16;
use vox_core::types::GaussianSplat;

/// Specification for one HLOD level.
pub struct HLODSpec {
    /// k = cell_splat_count / reduction_factor
    pub reduction_factor: u32,
}

/// One level of HLOD output.
pub struct HLODLevel {
    pub level: u32,
    pub splats: Vec<GaussianSplat>,
    pub coverage_error: f32, // max screen-space error at reference distance (1000m)
}

pub struct HLODBaker;

impl HLODBaker {
    /// Bake multiple HLOD levels from a cell's full splat set.
    pub fn bake_cell(cell_splats: &[GaussianSplat], levels: &[HLODSpec]) -> Vec<HLODLevel> {
        levels
            .iter()
            .enumerate()
            .map(|(level_idx, spec)| {
                let k = (cell_splats.len() / spec.reduction_factor as usize).max(1);
                let clustered = Self::kmeans_cluster(cell_splats, k, 50);
                let coverage_error =
                    Self::compute_coverage_error(&clustered, cell_splats, 1000.0);
                HLODLevel {
                    level: level_idx as u32,
                    splats: clustered,
                    coverage_error,
                }
            })
            .collect()
    }

    /// K-Means++ clustering of splats into k representative splats.
    fn kmeans_cluster(splats: &[GaussianSplat], k: usize, max_iter: u32) -> Vec<GaussianSplat> {
        if splats.is_empty() {
            return vec![];
        }
        // Cap k to available splats.
        let k = k.min(splats.len());

        // --- K-Means++ initialization ---
        let mut centroids: Vec<GaussianSplat> = Vec::with_capacity(k);

        // First centroid: deterministic LCG seeded from k.
        let mut lcg = k as u64;
        lcg = lcg_next(lcg);
        let first_idx = (lcg % splats.len() as u64) as usize;
        centroids.push(splats[first_idx]);

        for i in 1..k {
            // Compute D² distances.
            let mut d2: Vec<(f32, usize)> = splats
                .iter()
                .enumerate()
                .map(|(idx, s)| {
                    let pos = Vec3::from(s.position());
                    let min_dist2 = centroids
                        .iter()
                        .map(|c| (pos - Vec3::from(c.position())).length_squared())
                        .fold(f32::MAX, f32::min);
                    (min_dist2, idx)
                })
                .collect();

            // Sort descending by D² for deterministic selection.
            d2.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let candidate_count = d2.len() as u64;
            lcg = lcg_next(lcg);
            let pick = ((i as u64).wrapping_mul(1_664_525).wrapping_add(1_013_904_223) ^ lcg)
                % candidate_count;
            centroids.push(splats[d2[pick as usize].1]);
        }

        // --- Assignment + update ---
        let n = splats.len();
        let mut assignments: Vec<usize> = vec![0; n];
        // Initialize assignments.
        for (idx, s) in splats.iter().enumerate() {
            assignments[idx] = nearest_centroid(s, &centroids);
        }

        for _iter in 0..max_iter {
            // Update centroids from current assignments.
            let new_centroids = update_centroids(splats, &assignments, &centroids, k);

            // Reassign.
            let mut changed = false;
            for (idx, s) in splats.iter().enumerate() {
                let new_assign = nearest_centroid(s, &new_centroids);
                if new_assign != assignments[idx] {
                    assignments[idx] = new_assign;
                    changed = true;
                }
            }

            centroids = new_centroids;

            if !changed {
                break;
            }
        }

        centroids
    }

    /// For each original splat, find nearest HLOD splat and compute max screen-space error.
    fn compute_coverage_error(
        hlod: &[GaussianSplat],
        original: &[GaussianSplat],
        ref_dist: f32,
    ) -> f32 {
        if hlod.is_empty() || original.is_empty() {
            return 0.0;
        }

        original
            .iter()
            .map(|orig| {
                let orig_pos = Vec3::from(orig.position());
                let nearest_dist = hlod
                    .iter()
                    .map(|h| (orig_pos - Vec3::from(h.position())).length())
                    .fold(f32::MAX, f32::min);
                nearest_dist / ref_dist
            })
            .fold(0.0f32, f32::max)
    }
}

// --- Helpers ---

fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407)
}

fn nearest_centroid(s: &GaussianSplat, centroids: &[GaussianSplat]) -> usize {
    let pos = Vec3::from(s.position());
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, (pos - Vec3::from(c.position())).length_squared()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn update_centroids(
    splats: &[GaussianSplat],
    assignments: &[usize],
    prev_centroids: &[GaussianSplat],
    k: usize,
) -> Vec<GaussianSplat> {
    let mut new_centroids: Vec<GaussianSplat> = prev_centroids.to_vec();
    let mut cluster_members: Vec<Vec<usize>> = vec![vec![]; k];

    for (idx, &assign) in assignments.iter().enumerate() {
        cluster_members[assign].push(idx);
    }

    // Track which splats have been assigned to at least one cluster.
    // We'll handle empty clusters after the main pass.
    let mut assigned_splats: Vec<bool> = vec![false; splats.len()];
    for (ci, members) in cluster_members.iter().enumerate() {
        for &m in members {
            assigned_splats[m] = true;
            let _ = ci;
        }
    }

    for (ci, members) in cluster_members.iter().enumerate() {
        if members.is_empty() {
            // Find the nearest unassigned splat (or just any splat not in a singleton cluster).
            // Nearest to previous centroid among all splats.
            let prev_pos = Vec3::from(prev_centroids[ci].position());
            let nearest = splats
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (Vec3::from(a.position()) - prev_pos).length_squared();
                    let db = (Vec3::from(b.position()) - prev_pos).length_squared();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            new_centroids[ci] = splats[nearest];
            continue;
        }

        // Mean position.
        let count = members.len() as f32;
        let mut sum_pos = Vec3::ZERO;
        for &m in members {
            sum_pos += Vec3::from(splats[m].position());
        }
        let new_pos = sum_pos / count;

        // Scale = bounding sphere radius (max dist from new centroid to any member).
        let radius = members
            .iter()
            .map(|&m| (Vec3::from(splats[m].position()) - new_pos).length())
            .fold(0.0f32, f32::max);

        // Spectral: weighted average by opacity.
        let mut spectral_sum = [0.0f32; 16];
        let mut opacity_weight_sum = 0.0f32;
        let mut opacity_sum = 0u32;
        for &m in members {
            let s = &splats[m];
            let w = s.opacity() as f32 / 255.0;
            let spec = unpack_spectral(s);
            for b in 0..16 {
                spectral_sum[b] += w * spec[b];
            }
            opacity_weight_sum += w;
            opacity_sum += s.opacity() as u32;
        }

        if opacity_weight_sum > 0.0 {
            for val in spectral_sum.iter_mut() {
                *val /= opacity_weight_sum;
            }
        }

        // Opacity: mean, clamped to 255.
        let new_opacity = ((opacity_sum / members.len() as u32).min(255)) as u8;

        new_centroids[ci] = GaussianSplat::volume(
            new_pos.into(),
            [radius; 3],
            glam::Quat::IDENTITY,
            new_opacity,
            pack_spectral(spectral_sum),
        );
    }

    new_centroids
}

fn unpack_spectral(s: &GaussianSplat) -> [f32; 16] {
    std::array::from_fn(|b| f16::from_bits(s.spectral()[b]).to_f32())
}

fn pack_spectral(vals: [f32; 16]) -> [u16; 16] {
    std::array::from_fn(|b| f16::from_f32(vals[b].clamp(0.0, 1.0)).to_bits())
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn make_splat(x: f32, y: f32, z: f32, spectral_val: f32, opacity: u8) -> GaussianSplat {
        let spec_bits = f16::from_f32(spectral_val.clamp(0.0, 1.0)).to_bits();
        GaussianSplat::volume([x, y, z], [0.1; 3], glam::Quat::IDENTITY, opacity, [spec_bits; 16])
    }

    fn make_splats_grid(n: usize) -> Vec<GaussianSplat> {
        (0..n)
            .map(|i| {
                let x = (i % 32) as f32;
                let y = (i / 32) as f32;
                make_splat(x, y, 0.0, 0.5, 200)
            })
            .collect()
    }

    #[test]
    fn hlod_baker_bake_reduces_splats() {
        let splats = make_splats_grid(1000);
        let levels = vec![HLODSpec { reduction_factor: 64 }];
        let result = HLODBaker::bake_cell(&splats, &levels);
        assert_eq!(result.len(), 1);
        // k = 1000 / 64 = 15, .max(1) = 15; so ≤ 16 splats
        assert!(result[0].splats.len() <= 16, "got {} splats", result[0].splats.len());
    }

    #[test]
    fn hlod_baker_preserves_spectral() {
        // All splats have the same spectral value.
        let target = 0.75f32;
        let splats: Vec<GaussianSplat> = (0..50)
            .map(|i| make_splat(i as f32, 0.0, 0.0, target, 200))
            .collect();
        let levels = vec![HLODSpec { reduction_factor: 5 }];
        let result = HLODBaker::bake_cell(&splats, &levels);
        for s in &result[0].splats {
            let spec = unpack_spectral(s);
            for b in 0..16 {
                let diff = (spec[b] - target).abs();
                assert!(diff < 0.02, "spectral[{}] = {} expected ~{}", b, spec[b], target);
            }
        }
    }

    #[test]
    fn hlod_baker_single_splat() {
        let splats = vec![make_splat(1.0, 2.0, 3.0, 0.5, 128)];
        let levels = vec![HLODSpec { reduction_factor: 1 }];
        let result = HLODBaker::bake_cell(&splats, &levels);
        assert_eq!(result[0].splats.len(), 1);
        let s = &result[0].splats[0];
        assert!((s.position()[0] - 1.0).abs() < 1e-4);
        assert!((s.position()[1] - 2.0).abs() < 1e-4);
        assert!((s.position()[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn hlod_baker_two_levels() {
        let splats = make_splats_grid(500);
        let levels = vec![
            HLODSpec { reduction_factor: 10 },
            HLODSpec { reduction_factor: 100 },
        ];
        let result = HLODBaker::bake_cell(&splats, &levels);
        assert_eq!(result.len(), 2);
        assert!(
            result[0].splats.len() > result[1].splats.len(),
            "level[0] ({}) should have more splats than level[1] ({})",
            result[0].splats.len(),
            result[1].splats.len()
        );
    }

    #[test]
    fn coverage_error_same_splats() {
        let splats = make_splats_grid(20);
        let err = HLODBaker::compute_coverage_error(&splats, &splats, 1000.0);
        assert_eq!(err, 0.0, "coverage error of same splats should be 0");
    }

    #[test]
    fn kmeans_does_not_panic_on_k_greater_than_splats() {
        // k > splat count: bake with reduction_factor=1 on a tiny set.
        let splats = vec![
            make_splat(0.0, 0.0, 0.0, 0.3, 100),
            make_splat(1.0, 0.0, 0.0, 0.6, 150),
        ];
        // reduction_factor=1 → k=2 for 2 splats; also try factor=1 on 1 splat.
        let levels = vec![HLODSpec { reduction_factor: 1 }];
        let result = HLODBaker::bake_cell(&splats, &levels);
        assert!(!result[0].splats.is_empty());

        // Explicitly request more k than splats via bake_cell with a 1-splat input.
        let one = vec![make_splat(0.0, 0.0, 0.0, 0.5, 200)];
        let result2 = HLODBaker::bake_cell(&one, &[HLODSpec { reduction_factor: 1 }]);
        assert_eq!(result2[0].splats.len(), 1);

        // Direct call with k > splats.
        let clustered = HLODBaker::kmeans_cluster(&splats, 1000, 10);
        assert_eq!(clustered.len(), splats.len()); // capped to len
    }
}
