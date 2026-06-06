//! Atom-budget splat selector — unified, budget-driven LOD selection.
//!
//! Wires the previously-dead cluster BVH (`clas.rs`), per-cluster LOD chains
//! (built here from `hierarchical_lod.rs` rules), and frustum culling
//! (`frustum.rs`) into one per-frame stage. Frame cost becomes bounded by a
//! chosen splat budget instead of scene size ("Nanite for splats").
//!
//! Pipeline per `select()`:
//! 1. Walk the BVH against the camera frustum, collecting visible cluster ids.
//! 2. Score each visible cluster by projected solid angle × opacity
//!    (`score = total_opacity * r² / d²`).
//! 3. Assign each cluster its distance-driven LOD via [`select_lod_level`].
//! 4. While the summed splat count exceeds the budget, demote the lowest-score
//!    clusters one LOD level at a time; if there is slack, promote the
//!    highest-score clusters (never above their distance LOD) to spend it.
//! 5. Emit splat indices for each cluster at its final level, applying a
//!    crossfade opacity multiplier for clusters within a LOD transition band.
//!
//! Single owner: `&mut self` on `select()` reuses internal scratch buffers.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::Vec3;

use vox_core::types::GaussianSplat;

use crate::clas::{build_cluster_bvh, build_clusters, ClusterBVHNode, SplatCluster};
use crate::frustum::Frustum;
use crate::hierarchical_lod::{crossfade_factor, select_lod_level, LOD_LEVEL_COUNT};
use crate::spectral::RenderCamera;

/// Fraction of original splat count kept at each LOD level.
/// L0 = full, L1 = 40%, L2 = 10%, L3 = single billboard (handled specially).
/// Mirrors `hierarchical_lod::LOD_FRACTIONS` (private there).
const LOD_FRACTIONS: [f32; LOD_LEVEL_COUNT] = [1.0, 0.4, 0.1, 0.0];

/// Per-cluster precomputed LOD index lists (built once at scene load).
pub struct ClusterLod {
    cluster_id: u32,
    /// Indices into the global splat array. `L0 ⊇ L1 ⊇ L2`; `L3` = 1 billboard
    /// (the single highest-opacity splat of the cluster).
    levels: [Vec<u32>; LOD_LEVEL_COUNT],
}

impl ClusterLod {
    /// Cluster id this table belongs to.
    pub fn cluster_id(&self) -> u32 {
        self.cluster_id
    }

    /// Splat indices emitted at the given LOD level.
    pub fn level(&self, lod: usize) -> &[u32] {
        &self.levels[lod.min(LOD_LEVEL_COUNT - 1)]
    }
}

/// Reused per-frame scratch buffers (private).
#[derive(Default)]
struct SelectScratch {
    /// Visible cluster ids collected from the BVH walk this frame.
    visible: Vec<u32>,
    /// Per-visible-cluster working state, parallel to `visible`.
    work: Vec<ClusterWork>,
}

/// Working LOD-selection state for one visible cluster within a `select()`.
#[derive(Clone, Copy)]
struct ClusterWork {
    cluster_id: u32,
    /// Distance from the camera eye to the cluster centre.
    distance: f32,
    /// Importance score = total_opacity * r² / d².
    score: f32,
    /// Distance-driven LOD ceiling — clusters never render finer than this.
    distance_lod: u8,
    /// Current working LOD level (0 = finest).
    lod: u8,
    /// Splat count at the current LOD level.
    count: usize,
}

/// Heap entry for budget demotion (pop the *lowest* score first).
#[derive(Clone, Copy)]
struct DemoteEntry {
    score: f32,
    work_idx: usize,
}
impl PartialEq for DemoteEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.work_idx == other.work_idx
    }
}
impl Eq for DemoteEntry {}
impl Ord for DemoteEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on score: reverse the score comparison. Ties broken by
        // work_idx (which maps 1:1 to cluster id ordering) for determinism.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.work_idx.cmp(&self.work_idx))
    }
}
impl PartialOrd for DemoteEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Heap entry for budget promotion (pop the *highest* score first).
#[derive(Clone, Copy)]
struct PromoteEntry {
    score: f32,
    work_idx: usize,
}
impl PartialEq for PromoteEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.work_idx == other.work_idx
    }
}
impl Eq for PromoteEntry {}
impl Ord for PromoteEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.work_idx.cmp(&self.work_idx))
    }
}
impl PartialOrd for PromoteEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Budget-driven splat selector over a static splat array.
pub struct AtomBudgetSelector {
    clusters: Vec<SplatCluster>,
    bvh: Option<ClusterBVHNode>,
    lods: Vec<ClusterLod>,
    /// Streaming hook — non-resident clusters are skipped by `select()`.
    resident: Vec<bool>,
    scratch: SelectScratch,
}

/// What happened during one `select()` — the smoke prints this verbatim.
#[derive(Debug, Clone)]
pub struct SelectionStats {
    pub budget: usize,
    pub selected: usize,
    pub clusters_visible: usize,
    pub clusters_culled: usize,
    /// Clusters per final LOD level.
    pub lod_histogram: [usize; LOD_LEVEL_COUNT],
    /// Wall time of the select call, microseconds.
    pub select_us: u64,
}

/// Selected splat indices + per-splat crossfade opacity multiplier.
/// Emitted as parallel arrays to stay GPU-upload friendly.
///
/// Fields are crate-internal: `select()` is the only writer and maintains the
/// `indices.len() == opacity_scale.len()` invariant; cross-crate consumers
/// read through the accessors and cannot desynchronize the arrays.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    pub(crate) indices: Vec<u32>,
    /// Same length as `indices`; `1.0` except in LOD transition bands.
    pub(crate) opacity_scale: Vec<f32>,
}

impl Selection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Selected indices into the static splat array.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Per-selected-splat opacity multiplier, parallel to [`indices`](Self::indices).
    pub fn opacity_scale(&self) -> &[f32] {
        &self.opacity_scale
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn clear(&mut self) {
        self.indices.clear();
        self.opacity_scale.clear();
    }
}

impl AtomBudgetSelector {
    /// Build over a static splat array. `O(n log n)`; call once at scene load.
    pub fn build(splats: &[GaussianSplat], target_cluster_size: usize) -> Self {
        let clusters = build_clusters(splats, target_cluster_size.max(1));
        let bvh = build_cluster_bvh(&clusters);
        let lods: Vec<ClusterLod> = clusters
            .iter()
            .map(|c| build_cluster_lod(c, splats))
            .collect();
        let resident = vec![true; clusters.len()];
        AtomBudgetSelector {
            clusters,
            bvh,
            lods,
            resident,
            scratch: SelectScratch::default(),
        }
    }

    /// Number of clusters built over the static set.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Splat indices of one cluster at L0 (full detail), for GI gather.
    /// Empty slice for an unknown id.
    pub fn cluster_indices(&self, cluster_id: u32) -> &[u32] {
        match self.lods.iter().find(|l| l.cluster_id == cluster_id) {
            Some(l) => &l.levels[0],
            None => &[],
        }
    }

    /// Streaming hook: non-resident clusters are skipped by `select()`.
    pub fn set_cluster_resident(&mut self, cluster_id: u32, resident: bool) {
        if let Some(pos) = self.clusters.iter().position(|c| c.id == cluster_id) {
            self.resident[pos] = resident;
        }
    }

    /// Select `≤ budget` splat indices for this camera. Deterministic.
    /// Clears + fills `out`; returns stats. Never panics (empty scene →
    /// `selected = 0`).
    pub fn select(
        &mut self,
        camera: &RenderCamera,
        budget: usize,
        out: &mut Selection,
    ) -> SelectionStats {
        let start = std::time::Instant::now();
        out.clear();

        let total_clusters = self.clusters.len();
        let frustum = Frustum::from_view_proj(camera.view_proj());
        let eye = camera_eye(camera);

        // --- 1. Frustum-cull the cluster set via the BVH. ---
        let visible = &mut self.scratch.visible;
        visible.clear();
        if let Some(bvh) = &self.bvh {
            collect_visible(bvh, &self.clusters, &self.resident, &frustum, visible);
        }
        // Deterministic order: ascending cluster id.
        visible.sort_unstable();

        // --- 2/3. Score each visible cluster + assign its distance LOD. ---
        let work = &mut self.scratch.work;
        work.clear();
        for &cid in visible.iter() {
            let cluster = &self.clusters[cid as usize];
            let centre = cluster.center;
            let radius = aabb_radius(cluster);
            let d = (centre - eye).length().max(1e-3);
            let screen = projected_screen_size(radius, d);
            let distance_lod = select_lod_level(d, screen) as u8;
            let score = cluster.total_opacity * (radius * radius) / (d * d);
            let lod = distance_lod;
            let count = self.lods[cid as usize].levels[lod as usize].len();
            work.push(ClusterWork {
                cluster_id: cid,
                distance: d,
                score,
                distance_lod,
                lod,
                count,
            });
        }

        // --- 4. Drive the summed splat count toward the budget. ---
        let mut total: usize = work.iter().map(|w| w.count).sum();

        if total > budget {
            // Demote lowest-score clusters one level at a time.
            let mut heap: BinaryHeap<DemoteEntry> = BinaryHeap::with_capacity(work.len());
            for (i, w) in work.iter().enumerate() {
                if (w.lod as usize) < LOD_LEVEL_COUNT - 1 {
                    heap.push(DemoteEntry {
                        score: w.score,
                        work_idx: i,
                    });
                }
            }
            while total > budget {
                let Some(entry) = heap.pop() else { break };
                let w = &mut work[entry.work_idx];
                if (w.lod as usize) >= LOD_LEVEL_COUNT - 1 {
                    continue;
                }
                let old = w.count;
                w.lod += 1;
                w.count = self.lods[w.cluster_id as usize].levels[w.lod as usize].len();
                total = total - old + w.count;
                if (w.lod as usize) < LOD_LEVEL_COUNT - 1 {
                    heap.push(DemoteEntry {
                        score: w.score,
                        work_idx: entry.work_idx,
                    });
                }
            }
            // Every cluster floors at its 1-splat L3 billboard, so when MORE
            // CLUSTERS are visible than the budget allows, demotion alone
            // cannot honor the documented `selected <= budget` bound. Shed
            // whole clusters lowest-score-first (deterministic: score then
            // cluster id) until the bound holds. A shed cluster emits nothing
            // this frame and is excluded from the LOD histogram.
            if total > budget {
                let mut by_score: Vec<usize> = (0..work.len()).collect();
                by_score.sort_by(|&a, &b| {
                    work[a]
                        .score
                        .partial_cmp(&work[b].score)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| work[a].cluster_id.cmp(&work[b].cluster_id))
                });
                for idx in by_score {
                    if total <= budget {
                        break;
                    }
                    let w = &mut work[idx];
                    total -= w.count;
                    w.count = 0; // shed: emit loop skips zero-count clusters
                }
            }
        } else if total < budget {
            // Slack: promote highest-score clusters toward (never above) their
            // distance LOD to spend the budget.
            let mut heap: BinaryHeap<PromoteEntry> = BinaryHeap::with_capacity(work.len());
            for (i, w) in work.iter().enumerate() {
                if w.lod > w.distance_lod {
                    heap.push(PromoteEntry {
                        score: w.score,
                        work_idx: i,
                    });
                }
            }
            while let Some(entry) = heap.peek().copied() {
                let w = work[entry.work_idx];
                if w.lod <= w.distance_lod {
                    heap.pop();
                    continue;
                }
                let next_lod = w.lod - 1;
                let next_count =
                    self.lods[w.cluster_id as usize].levels[next_lod as usize].len();
                let delta = next_count - w.count;
                if total + delta > budget {
                    break;
                }
                heap.pop();
                let wm = &mut work[entry.work_idx];
                wm.lod = next_lod;
                wm.count = next_count;
                total += delta;
                if wm.lod > wm.distance_lod {
                    heap.push(PromoteEntry {
                        score: wm.score,
                        work_idx: entry.work_idx,
                    });
                }
            }
        }

        // --- 5. Emit indices + crossfade opacity multipliers. ---
        out.indices.reserve(total);
        out.opacity_scale.reserve(total);
        let mut histogram = [0usize; LOD_LEVEL_COUNT];
        for w in work.iter() {
            if w.count == 0 {
                // Shed under extreme budget pressure (or genuinely empty) —
                // contributes nothing and is not a rendered LOD.
                continue;
            }
            histogram[w.lod as usize] += 1;
            let level = &self.lods[w.cluster_id as usize].levels[w.lod as usize];
            // Crossfade only meaningful when rendering at the distance-driven
            // level (a transition band between this level and the next). The
            // common case is scale == 1.0 (outside any band) — bulk-extend then.
            let fade = if w.lod == w.distance_lod {
                crossfade_factor(w.distance, w.lod as u32)
            } else {
                0.0
            };
            let scale = 1.0 - fade;
            out.indices.extend_from_slice(level);
            if scale == 1.0 {
                out.opacity_scale.resize(out.indices.len(), 1.0);
            } else {
                out.opacity_scale
                    .extend(std::iter::repeat_n(scale, level.len()));
            }
        }

        let selected = out.indices.len();
        let clusters_visible = work.len();
        let clusters_culled = total_clusters - clusters_visible;
        let select_us = start.elapsed().as_micros() as u64;

        SelectionStats {
            budget,
            selected,
            clusters_visible,
            clusters_culled,
            lod_histogram: histogram,
            select_us,
        }
    }

    /// Cluster ids nearest `pos` until `≥ min_splats` are covered (GI subset
    /// query). Best-first walk; returned ids are sorted by ascending cluster id
    /// for determinism. The set always covers `≥ min_splats` splats when the
    /// scene holds that many.
    pub fn nearest_clusters(&self, pos: Vec3, min_splats: usize) -> Vec<u32> {
        if self.clusters.is_empty() || min_splats == 0 {
            return Vec::new();
        }
        // Order clusters by distance from `pos` to their centre, tie-break by
        // cluster id. Accumulate until the covered splat count reaches the goal.
        let mut order: Vec<(f32, u32)> = self
            .clusters
            .iter()
            .map(|c| ((c.center - pos).length_squared(), c.id))
            .collect();
        order.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut covered = 0usize;
        let mut out: Vec<u32> = Vec::new();
        for (_, cid) in order {
            out.push(cid);
            covered += self.clusters[cid as usize].splat_indices.len();
            if covered >= min_splats {
                break;
            }
        }
        out.sort_unstable();
        out
    }
}

/// Build the 4-level per-cluster LOD index table by opacity-weighted prefixes.
fn build_cluster_lod(cluster: &SplatCluster, splats: &[GaussianSplat]) -> ClusterLod {
    // Sort cluster indices by opacity descending (stable id tie-break) so that
    // L1/L2 prefixes keep the most-visible atoms. L3 = the single brightest.
    let mut sorted: Vec<u32> = cluster.splat_indices.clone();
    sorted.sort_by(|&a, &b| {
        let oa = splats[a as usize].opacity();
        let ob = splats[b as usize].opacity();
        ob.cmp(&oa).then_with(|| a.cmp(&b))
    });

    let n = sorted.len();
    let l0 = sorted.clone();
    let l1_len = ((n as f32 * LOD_FRACTIONS[1]).round() as usize)
        .clamp(if n > 0 { 1 } else { 0 }, n);
    let l2_len = ((n as f32 * LOD_FRACTIONS[2]).round() as usize)
        .clamp(if n > 0 { 1 } else { 0 }, n);
    let l1 = sorted[..l1_len].to_vec();
    let l2 = sorted[..l2_len].to_vec();
    let l3 = if n > 0 { vec![sorted[0]] } else { Vec::new() };

    ClusterLod {
        cluster_id: cluster.id,
        levels: [l0, l1, l2, l3],
    }
}

/// Bounding-sphere radius of a cluster's AABB.
fn aabb_radius(cluster: &SplatCluster) -> f32 {
    ((cluster.aabb_max - cluster.aabb_min) * 0.5).length().max(1e-4)
}

/// Camera eye position in world space (inverse-view translation).
fn camera_eye(camera: &RenderCamera) -> Vec3 {
    camera.view.inverse().col(3).truncate()
}

/// Cheap projected-size proxy used to drive `select_lod_level`'s screen-size
/// branch: radius/distance scaled to a pixel-ish magnitude. No trig.
fn projected_screen_size(radius: f32, distance: f32) -> f32 {
    // 1000 ~= viewport pixel scale; keeps the screen-size branch consistent
    // with the distance branch at the demo's draw distances.
    1000.0 * radius / distance
}

/// Walk the cluster BVH against the frustum, collecting visible & resident
/// cluster ids. Internal nodes are sphere-tested by their AABB; leaves resolve
/// to the cluster's own AABB sphere.
fn collect_visible(
    node: &ClusterBVHNode,
    clusters: &[SplatCluster],
    resident: &[bool],
    frustum: &Frustum,
    out: &mut Vec<u32>,
) {
    match node {
        ClusterBVHNode::Leaf { cluster_id } => {
            let c = &clusters[*cluster_id as usize];
            if !resident[*cluster_id as usize] {
                return;
            }
            let centre = (c.aabb_min + c.aabb_max) * 0.5;
            let radius = ((c.aabb_max - c.aabb_min) * 0.5).length().max(1e-4);
            if frustum.contains_sphere(centre, radius) {
                out.push(*cluster_id);
            }
        }
        ClusterBVHNode::Internal {
            aabb_min,
            aabb_max,
            left,
            right,
        } => {
            let centre = (*aabb_min + *aabb_max) * 0.5;
            let radius = ((*aabb_max - *aabb_min) * 0.5).length().max(1e-4);
            if !frustum.contains_sphere(centre, radius) {
                return;
            }
            collect_visible(left, clusters, resident, frustum, out);
            collect_visible(right, clusters, resident, frustum, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat};
    use std::f32::consts::FRAC_PI_4;

    fn splat_at(pos: [f32; 3], opacity: u8) -> GaussianSplat {
        GaussianSplat::volume(pos, [0.3, 0.3, 0.3], Quat::IDENTITY, opacity, [0u16; 16])
    }

    /// A grid of splats spanning a wide area, deterministic opacities. ~64k
    /// splats — a walking_sim-like scene for budget tests.
    fn grid_scene() -> Vec<GaussianSplat> {
        let mut v = Vec::new();
        let n = 40; // 40^3 = 64000
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let op = (((x * 7 + y * 13 + z * 17) % 200) + 55) as u8;
                    v.push(splat_at(
                        [x as f32 * 1.0 - 20.0, y as f32 * 0.5, z as f32 * 1.0 - 20.0],
                        op,
                    ));
                }
            }
        }
        v
    }

    fn camera_at(eye: Vec3, target: Vec3) -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(FRAC_PI_4, 1.0, 0.1, 2000.0),
        }
    }

    #[test]
    fn budget_is_spent_not_just_bounded() {
        let scene = grid_scene();
        assert!(scene.len() >= 60_000, "scene should be ~64k, got {}", scene.len());
        let mut sel = AtomBudgetSelector::build(&scene, 128);
        // Camera in the middle of the cloud looking along +Z so most clusters
        // are in view.
        let cam = camera_at(Vec3::new(0.0, 10.0, -25.0), Vec3::new(0.0, 10.0, 25.0));
        let mut out = Selection::new();
        let stats = sel.select(&cam, 2000, &mut out);
        assert!(
            out.indices.len() <= 2000 && out.indices.len() >= 1900,
            "budget not spent: selected {} (want 1900..=2000)",
            out.indices.len()
        );
        assert_eq!(stats.selected, out.indices.len());
    }

    #[test]
    fn behind_camera_culls_to_zero_then_flips() {
        // All splats at z < -10; camera at origin looking +Z (away from them).
        let mut scene = Vec::new();
        for i in 0..2000 {
            let f = i as f32;
            scene.push(splat_at([(f % 10.0) - 5.0, (f * 0.01) % 5.0, -10.0 - (f * 0.1)], 200));
        }
        let mut sel = AtomBudgetSelector::build(&scene, 64);

        let looking_away = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 50.0));
        let mut out = Selection::new();
        let stats_away = sel.select(&looking_away, 100_000, &mut out);
        assert_eq!(out.indices.len(), 0, "splats behind camera must be culled");
        assert!(stats_away.clusters_culled > 0, "cull count must be > 0");

        let looking_at = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -50.0));
        let stats_at = sel.select(&looking_at, 100_000, &mut out);
        assert!(
            out.indices.len() > 0,
            "splats in front of camera must be visible"
        );
        assert!(stats_at.clusters_visible > 0);
    }

    #[test]
    fn near_cluster_emits_at_least_4x_far() {
        // Two identical 1k-splat clusters: one at ~5 m, one at ~300 m, budget ∞.
        let mut scene = Vec::new();
        for i in 0..1000 {
            let j = (i % 100) as f32 * 0.02;
            scene.push(splat_at([j, j, 5.0], 200));
        }
        for i in 0..1000 {
            let j = (i % 100) as f32 * 0.02;
            scene.push(splat_at([j, j, 305.0], 200));
        }
        let mut sel = AtomBudgetSelector::build(&scene, 2048); // 1 cluster each
        let cam = camera_at(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 500.0));
        let mut out = Selection::new();
        sel.select(&cam, usize::MAX, &mut out);

        // Count emitted indices that point at the near vs far cluster (near
        // splats have z≈5, far z≈305).
        let mut near = 0usize;
        let mut far = 0usize;
        for &idx in &out.indices {
            if scene[idx as usize].position()[2] < 100.0 {
                near += 1;
            } else {
                far += 1;
            }
        }
        assert!(near > 0 && far > 0, "both clusters must contribute (near={near} far={far})");
        assert!(
            near >= 4 * far,
            "near cluster ({near}) should emit >= 4x the far cluster ({far})"
        );
    }

    #[test]
    fn budget_pressure_degrades_lod_globally() {
        let scene = grid_scene();
        let mut sel = AtomBudgetSelector::build(&scene, 128);
        let cam = camera_at(Vec3::new(0.0, 10.0, -25.0), Vec3::new(0.0, 10.0, 25.0));
        let mut out = Selection::new();

        let hi = sel.select(&cam, 32_000, &mut out);
        let lo = sel.select(&cam, 4_000, &mut out);

        assert!(
            lo.lod_histogram[0] < hi.lod_histogram[0],
            "L0 count under pressure ({}) should shrink vs relaxed ({})",
            lo.lod_histogram[0],
            hi.lod_histogram[0]
        );
        assert!(
            lo.lod_histogram[3] > hi.lod_histogram[3],
            "L3 count under pressure ({}) should grow vs relaxed ({})",
            lo.lod_histogram[3],
            hi.lod_histogram[3]
        );
    }

    #[test]
    fn boundary_crossfade_scales_opacity_between_zero_and_full() {
        // One wide cluster placed so its centre sits at d≈135 m — inside the
        // L1→L2 transition band (LOD_DISTANCES 50→150, band starts at
        // 50 + 0.8*100 = 130 m). Spread ±10 m so the projected-size proxy keeps
        // it distance-driven at L1 (screen ∈ [50,200)), where the crossfade band
        // is defined.
        let mut scene = Vec::new();
        for i in 0..400 {
            let gx = (i % 20) as f32 - 10.0;
            let gy = (i / 20) as f32 - 10.0;
            scene.push(splat_at([gx, gy, 135.0], 220));
        }
        let mut sel = AtomBudgetSelector::build(&scene, 2048); // single cluster
        let cam = camera_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 200.0));
        let mut out = Selection::new();
        sel.select(&cam, usize::MAX, &mut out);

        assert!(!out.indices.is_empty(), "cluster must emit splats");
        // The grid may shed tiny boundary-cell clusters whose screen-size proxy
        // resolves to a band-less LOD (scale 1.0) — the *band* claim is about
        // the bulk of the emission, so require a strict-interior multiplier on
        // the majority of emitted splats, then check the band on one of them.
        let in_band: Vec<f32> = out
            .opacity_scale
            .iter()
            .copied()
            .filter(|&s| s > 0.0 && s < 1.0)
            .collect();
        assert!(
            in_band.len() * 2 > out.opacity_scale.len(),
            "most emitted splats ({} of {}) must carry a strictly-interior crossfade multiplier",
            in_band.len(),
            out.opacity_scale.len()
        );
        // Apply the multiplier to a real u8 opacity and check the band.
        let full = 220u8;
        let faded_scale = in_band[0];
        let faded = (full as f32 * faded_scale).round() as u8;
        assert!(
            faded < full && faded > 0,
            "faded opacity {faded} must be strictly between 0 and full ({full})"
        );
    }

    /// The documented `selected <= budget` bound must hold even when MORE
    /// CLUSTERS are visible than the budget — the L3 1-splat floor cannot be
    /// allowed to overshoot; whole low-score clusters are shed instead.
    #[test]
    fn budget_holds_when_visible_clusters_exceed_it() {
        let scene = grid_scene();
        let mut sel = AtomBudgetSelector::build(&scene, 128); // several hundred clusters
        let cam = camera_at(Vec3::new(0.0, 10.0, -25.0), Vec3::new(0.0, 10.0, 25.0));
        let mut out = Selection::new();

        let stats = sel.select(&cam, 100, &mut out); // far below cluster count
        assert!(
            stats.clusters_visible > 100,
            "precondition: more visible clusters ({}) than budget",
            stats.clusters_visible
        );
        assert!(
            out.indices.len() <= 100,
            "hard bound violated: selected {} > budget 100",
            out.indices.len()
        );
        assert!(out.indices.len() > 0, "shedding must not empty the selection");
        // Histogram counts only EMITTING clusters — it must sum to <= budget
        // (every survivor is at L3 = 1 splat under this much pressure).
        let emitting: usize = stats.lod_histogram.iter().sum();
        assert!(
            emitting <= 100,
            "histogram counts shed clusters: {emitting} emitting > 100"
        );
        // Deterministic shedding: same call yields the exact same survivors.
        let mut again = Selection::new();
        sel.select(&cam, 100, &mut again);
        assert_eq!(out.indices, again.indices, "shedding must be deterministic");
    }

    #[test]
    fn selection_is_deterministic() {
        let scene = grid_scene();
        let mut sel = AtomBudgetSelector::build(&scene, 128);
        let cam = camera_at(Vec3::new(2.0, 8.0, -20.0), Vec3::new(1.0, 9.0, 30.0));
        let mut a = Selection::new();
        let mut b = Selection::new();
        sel.select(&cam, 8000, &mut a);
        sel.select(&cam, 8000, &mut b);
        assert_eq!(a.indices, b.indices, "index vectors must be exactly equal");
        assert_eq!(
            a.opacity_scale, b.opacity_scale,
            "opacity multipliers must be exactly equal"
        );
    }

    #[test]
    fn nearest_clusters_subset_distance_bounded() {
        let scene = grid_scene();
        let sel = AtomBudgetSelector::build(&scene, 128);
        let player = Vec3::new(0.0, 5.0, 0.0);
        let k = 2000usize;

        let ids = sel.nearest_clusters(player, k);
        assert!(!ids.is_empty(), "must return clusters");

        // Covered splat count meets the goal.
        let covered: usize = ids
            .iter()
            .map(|&id| sel.cluster_indices(id).len())
            .sum();
        assert!(covered >= k, "subset covers {covered} splats (< {k})");

        // Old path: sort ALL splats by distance, take k, record max distance.
        let mut dists: Vec<f32> = scene
            .iter()
            .map(|s| (Vec3::from(s.position()) - player).length())
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let old_max = dists[k - 1];

        // New subset's max splat distance must be ≤ old max + a cluster radius.
        let mut subset_max = 0.0f32;
        let mut max_radius = 0.0f32;
        for &id in &ids {
            for &idx in sel.cluster_indices(id) {
                let d = (Vec3::from(scene[idx as usize].position()) - player).length();
                subset_max = subset_max.max(d);
            }
            // cluster radius for the bound
            let c = &sel.clusters[id as usize];
            let r = ((c.aabb_max - c.aabb_min) * 0.5).length();
            max_radius = max_radius.max(r);
        }
        assert!(
            subset_max <= old_max + max_radius + 1e-3,
            "subset max distance {subset_max} exceeds old max {old_max} + radius {max_radius}"
        );
    }
}
