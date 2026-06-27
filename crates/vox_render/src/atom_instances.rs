//! Instance-aware atom budget selection — "Nanite for splat *cities*".
//!
//! `AssetAtomLibrary` cooks a handful of splat assets ONCE (clusters, per-
//! cluster LOD index tables, two asset-level imposter LODs, flat GPU tables);
//! `InstancedSelector` then runs a two-level budget cut over thousands of
//! placed instances: an instance pass (frustum + distance classifies each
//! placement as culled / far-imposter / near-cluster-granular) feeding the
//! proven `atom_budget` host heap walk, extended with the I0→I1 imposter
//! demotion. The emitted `Vec<ClusterDraw>` (prefix-summed `atom_offset`s)
//! drives the GPU expand pass.
//!
//! Engine-level only: this module speaks assets / instances / atoms.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use half::f16;

use vox_core::types::GaussianSplat;

use crate::clas::{SplatCluster, build_clusters};
use crate::frustum::Frustum;
use crate::hierarchical_lod::{LOD_LEVEL_COUNT, crossfade_factor, select_lod_level};
use crate::spectral::RenderCamera;

// CONFIG-FIRST (de-duplicated): the LOD fraction ladder + the far/imposter
// switch distances were copy-pasted here. They now have ONE source — `vox_config`
// (`config/ochroma.ron` `scatter.lod_fractions` / `scatter.far_instance_m` =
// `LOD_DISTANCES[2]` / `scatter.imposter_i1_m` = `LOD_DISTANCES[3]`) — read at the
// call sites below. Defaults equal the old literals (`[1.0,0.4,0.1,0.0]` / 150 / 400).

/// I0 imposter voxel resolution per axis — 4³ = 64 atoms maximum.
const IMPOSTER0_RES: usize = 4;

/// One placed asset instance — 48 B POD, GPU-upload friendly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct AtomInstance {
    /// World-space translation.
    pub position: [f32; 3],
    /// Index into the library's asset list.
    pub asset: u32,
    /// World-space rotation quaternion, XYZW. Expected unit length.
    pub rotation: [f32; 4],
    /// Caller-chosen stable id, carried through to draws.
    pub instance_id: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<AtomInstance>() == 48);

impl AtomInstance {
    pub fn new(asset: u32, instance_id: u32, position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            position,
            asset,
            rotation,
            instance_id,
            _pad: [0; 3],
        }
    }

    pub fn asset(&self) -> u32 {
        self.asset
    }

    pub fn instance_id(&self) -> u32 {
        self.instance_id
    }
}

/// Packed asset-local library atom — 64 B, std430 friendly. The pinned layout:
/// vec4 `pos.xyz + opacity (0..1)`, vec4 `scale.xyz + 0`, vec4 `quat xyzw`,
/// 4×u32 holding the 8 pair-averaged spectral bins as f16 (lo half = bin 2j,
/// hi half = bin 2j+1) — the same `(band 2i + band 2i+1)/2` binning as
/// `gaussian_splat_to_gpu_full`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct PackedLibraryAtom {
    pub(crate) pos_opacity: [f32; 4],
    pub(crate) scale: [f32; 4],
    pub(crate) quat: [f32; 4],
    pub(crate) spectral: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<PackedLibraryAtom>() == 64);

fn pack_atom(s: &GaussianSplat) -> PackedLibraryAtom {
    let p = s.position();
    let sc = s.scales();
    let q = s.decoded_rotation();
    let bin = |i: usize| -> u16 {
        let a = f16::from_bits(s.spectral()[i * 2]).to_f32();
        let b = f16::from_bits(s.spectral()[i * 2 + 1]).to_f32();
        f16::from_f32((a + b) * 0.5).to_bits()
    };
    let spectral = std::array::from_fn(|j| {
        let lo = bin(j * 2) as u32;
        let hi = bin(j * 2 + 1) as u32;
        lo | (hi << 16)
    });
    PackedLibraryAtom {
        pos_opacity: [p[0], p[1], p[2], s.opacity() as f32 / 255.0],
        scale: [sc[0], sc[1], sc[2], 0.0],
        quat: [q.x, q.y, q.z, q.w],
        spectral,
    }
}

/// What one `ClusterDraw` renders: a cluster at a LOD level, or an asset-level
/// imposter (`level` 0 = I0 voxel imposter ≤ 64 atoms, 1 = I1 single atom).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawUnit {
    Cluster { id: u32, lod: u8 },
    Imposter { level: u8 },
}

/// One selected work unit of one instance, with its slot range in the
/// expanded output buffers (`atom_offset` = running prefix sum in emission
/// order → disjoint GPU write slots).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterDraw {
    pub instance: u32,
    pub asset: u32,
    pub unit: DrawUnit,
    pub opacity_scale: f32,
    pub atom_offset: u32,
    pub atom_count: u32,
}

/// Selected draws. `InstancedSelector::select` is the only writer.
#[derive(Debug, Clone, Default)]
pub struct InstancedSelection {
    draws: Vec<ClusterDraw>,
}

impl InstancedSelection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn draws(&self) -> &[ClusterDraw] {
        &self.draws
    }

    /// Total atoms across all draws (== the expanded splat count).
    pub fn atom_count(&self) -> usize {
        self.draws.iter().map(|d| d.atom_count as usize).sum()
    }
}

/// What happened during one `InstancedSelector::select`.
#[derive(Debug, Clone)]
pub struct InstancedStats {
    pub budget: usize,
    pub selected: usize,
    pub instances_visible: usize,
    pub instances_culled: usize,
    pub instances_far: usize,
    /// Emitting units per final level: `[L0, L1, L2, L3, I0, I1]`.
    pub lod_histogram: [usize; 6],
    /// Wall time of the select call, microseconds.
    pub select_us: u64,
}

/// Per-cluster metadata kept for the instance-pass classification. Mirrors
/// exactly what the `atom_budget` oracle reads per cluster: the AABB-midpoint
/// sphere for the frustum test, the centroid for distance/score. `pub(crate)`
/// so the same-crate GPU scorer
/// ([`crate::gpu::instanced_select_gpu::InstancedSelectGpu`]) uploads the SAME
/// table instead of re-deriving it.
pub(crate) struct ClusterMeta {
    pub(crate) id: u32,
    /// AABB midpoint (asset-local) — the oracle's BVH-leaf frustum centre.
    pub(crate) frustum_center: Vec3,
    /// Splat centroid (asset-local) — the oracle's distance/score centre.
    pub(crate) centroid: Vec3,
    /// AABB half-diagonal length, `.max(1e-4)`.
    pub(crate) radius: f32,
    pub(crate) total_opacity: f32,
}

struct AssetEntry {
    /// Offset of this asset's atoms in the packed atom table.
    /// (Read via the `pub(crate)` GPU-table accessors — expand pass, Task 4.)
    #[allow(dead_code)]
    atom_base: u32,
    /// Number of full (non-imposter) atoms.
    #[allow(dead_code)]
    full_atom_count: u32,
    /// Number of I0 voxel-imposter atoms (1..=64; 0 only for an empty asset).
    #[allow(dead_code)]
    imposter0_count: u32,
    /// Local AABB midpoint.
    bounds_center: Vec3,
    /// Local AABB half-diagonal, `.max(1e-4)`.
    bounds_radius: f32,
    /// Sum of cluster opacities — the asset-level score weight.
    total_opacity: f32,
    clusters: Vec<ClusterMeta>,
    /// Per cluster, per LOD: `(offset, len)` into the concatenated index list.
    /// Parallel to `clusters` (cluster id == position, as `build_clusters`
    /// assigns sequential ids).
    cluster_ranges: Vec<[(u32, u32); LOD_LEVEL_COUNT]>,
    /// `(offset, len)` for the I0 and I1 imposter units.
    imposter_ranges: [(u32, u32); 2],
}

/// Cook-once, immutable (`Send + Sync`, share via `Arc`) library of splat
/// assets: real `clas` clusters, the proven opacity-prefix LOD tables, two
/// asset-level imposter LODs, and the flat GPU tables for the expand pass.
pub struct AssetAtomLibrary {
    assets: Vec<AssetEntry>,
    /// All library atoms packed 64 B — per asset: full atoms, then I0, then I1.
    packed_atoms: Vec<PackedLibraryAtom>,
    /// Concatenated ASSET-LOCAL atom index lists; units address it via
    /// `(offset, len)`. The GPU pass adds `asset_atom_base` per draw.
    index_list: Vec<u32>,
}

impl AssetAtomLibrary {
    /// Cook the library. `assets[i]` becomes asset id `i` (asset-local
    /// coordinates). `O(n log n)` per asset; call once at load.
    pub fn build(assets: &[Vec<GaussianSplat>], target_cluster_size: usize) -> Self {
        let mut entries = Vec::with_capacity(assets.len());
        let mut packed_atoms = Vec::new();
        let mut index_list = Vec::new();

        for atoms in assets {
            let clusters = build_clusters(atoms, target_cluster_size.max(1));

            // Local bounds over the full atoms.
            let mut mn = Vec3::splat(f32::MAX);
            let mut mx = Vec3::splat(f32::MIN);
            for s in atoms {
                let p = Vec3::from(s.position());
                mn = mn.min(p);
                mx = mx.max(p);
            }
            let (bounds_center, bounds_radius) = if atoms.is_empty() {
                (Vec3::ZERO, 1e-4)
            } else {
                ((mn + mx) * 0.5, ((mx - mn) * 0.5).length().max(1e-4))
            };

            let atom_base = packed_atoms.len() as u32;
            let full_atom_count = atoms.len() as u32;
            packed_atoms.extend(atoms.iter().map(pack_atom));

            // Per-cluster LOD index tables + metadata, ascending cluster id.
            let mut metas = Vec::with_capacity(clusters.len());
            let mut cluster_ranges = Vec::with_capacity(clusters.len());
            let mut total_opacity = 0.0f32;
            for c in &clusters {
                let levels = build_cluster_levels(c, atoms);
                let mut ranges = [(0u32, 0u32); LOD_LEVEL_COUNT];
                for (lod, level) in levels.iter().enumerate() {
                    ranges[lod] = (index_list.len() as u32, level.len() as u32);
                    index_list.extend_from_slice(level);
                }
                cluster_ranges.push(ranges);
                total_opacity += c.total_opacity;
                metas.push(ClusterMeta {
                    id: c.id,
                    frustum_center: (c.aabb_min + c.aabb_max) * 0.5,
                    centroid: c.center,
                    radius: aabb_radius(c),
                    total_opacity: c.total_opacity,
                });
            }

            // Asset-level imposters, appended after the full atoms.
            let mut imposter_ranges = [(index_list.len() as u32, 0u32); 2];
            let mut imposter0_count = 0u32;
            if !atoms.is_empty() {
                let (i0, i1) = build_imposters(atoms, mn, mx);
                imposter0_count = i0.len() as u32;
                imposter_ranges[0] = (index_list.len() as u32, imposter0_count);
                index_list
                    .extend((0..imposter0_count).map(|k| full_atom_count + k));
                imposter_ranges[1] = (index_list.len() as u32, 1);
                index_list.push(full_atom_count + imposter0_count);
                packed_atoms.extend(i0.iter().map(pack_atom));
                packed_atoms.push(pack_atom(&i1));
            }

            entries.push(AssetEntry {
                atom_base,
                full_atom_count,
                imposter0_count,
                bounds_center,
                bounds_radius,
                total_opacity,
                clusters: metas,
                cluster_ranges,
                imposter_ranges,
            });
        }

        AssetAtomLibrary {
            assets: entries,
            packed_atoms,
            index_list,
        }
    }

    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Library atoms INCLUDING imposters (NOT the virtual instanced count).
    pub fn total_atoms(&self) -> usize {
        self.packed_atoms.len()
    }

    /// Asset-local bounding sphere `(centre, radius)` — AABB midpoint and
    /// half-diagonal over the asset's full atoms.
    pub fn asset_bounds(&self, asset: u32) -> (Vec3, f32) {
        let a = &self.assets[asset as usize];
        (a.bounds_center, a.bounds_radius)
    }

    /// Bytes held resident by the cooked tables.
    pub fn resident_bytes(&self) -> usize {
        let mut bytes = self.packed_atoms.len() * std::mem::size_of::<PackedLibraryAtom>()
            + self.index_list.len() * std::mem::size_of::<u32>();
        for a in &self.assets {
            bytes += std::mem::size_of::<AssetEntry>()
                + a.clusters.len() * std::mem::size_of::<ClusterMeta>()
                + a.cluster_ranges.len() * std::mem::size_of::<[(u32, u32); LOD_LEVEL_COUNT]>();
        }
        bytes
    }

    // Flat GPU-table accessors below feed the same-crate expand pass
    // (`ExpandDrawsPass`, plan Task 4) and the in-module tests; `dead_code`
    // allowed until that pass lands.

    /// Flat packed atom table (per asset: full atoms, I0 atoms, I1 atom) for
    /// the GPU expand pass.
    #[allow(dead_code)]
    pub(crate) fn packed_atoms(&self) -> &[PackedLibraryAtom] {
        &self.packed_atoms
    }

    /// Concatenated asset-LOCAL atom index lists (see `unit_range`).
    #[allow(dead_code)]
    pub(crate) fn atom_index_list(&self) -> &[u32] {
        &self.index_list
    }

    /// Offset of `asset`'s atoms in [`packed_atoms`](Self::packed_atoms) —
    /// the GPU pass adds this to the asset-local indices.
    #[allow(dead_code)]
    pub(crate) fn asset_atom_base(&self, asset: u32) -> u32 {
        self.assets[asset as usize].atom_base
    }

    /// Number of cooked clusters for `asset` (`build_clusters` assigns the
    /// sequential ids `0..count`). Lets the same-crate GPU expand pass
    /// snapshot every per-unit `(offset, len)` range at construction.
    pub(crate) fn cluster_count(&self, asset: u32) -> usize {
        self.assets[asset as usize].clusters.len()
    }

    /// `(offset, len)` of one draw unit's slice of the concatenated index list.
    #[allow(dead_code)]
    pub(crate) fn unit_range(&self, asset: u32, unit: DrawUnit) -> (u32, u32) {
        let a = &self.assets[asset as usize];
        match unit {
            DrawUnit::Cluster { id, lod } => a.cluster_ranges[id as usize][lod as usize],
            DrawUnit::Imposter { level } => a.imposter_ranges[level as usize],
        }
    }

    /// Asset-local atom indices of one draw unit.
    #[allow(dead_code)]
    pub(crate) fn unit_indices(&self, asset: u32, unit: DrawUnit) -> &[u32] {
        let (off, len) = self.unit_range(asset, unit);
        &self.index_list[off as usize..(off + len) as usize]
    }

    /// Sum of cluster opacities for `asset` — the far-imposter score weight
    /// the GPU instance kernel uploads per asset.
    pub(crate) fn asset_total_opacity(&self, asset: u32) -> f32 {
        self.assets[asset as usize].total_opacity
    }

    /// Per-cluster metadata for `asset`, ascending cluster id
    /// (`build_clusters` assigns sequential ids, so position == id). Feeds the
    /// GPU scorer's flattened global cluster-meta table.
    pub(crate) fn cluster_metas(&self, asset: u32) -> &[ClusterMeta] {
        &self.assets[asset as usize].clusters
    }
}

/// Build the 4-level per-cluster LOD index table by opacity prefixes —
/// verbatim `atom_budget::build_cluster_lod` semantics (opacity-DESCENDING
/// sort, ascending-index tie-break; L1/L2 rounded prefixes clamped ≥ 1 when
/// non-empty; L3 = the single brightest atom).
fn build_cluster_levels(
    cluster: &SplatCluster,
    splats: &[GaussianSplat],
) -> [Vec<u32>; LOD_LEVEL_COUNT] {
    let mut sorted: Vec<u32> = cluster.splat_indices.clone();
    sorted.sort_by(|&a, &b| {
        let oa = splats[a as usize].opacity();
        let ob = splats[b as usize].opacity();
        ob.cmp(&oa).then_with(|| a.cmp(&b))
    });

    let n = sorted.len();
    let lod_fractions = &vox_config::config().scatter.lod_fractions;
    let l0 = sorted.clone();
    let l1_len =
        ((n as f32 * lod_fractions[1]).round() as usize).clamp(if n > 0 { 1 } else { 0 }, n);
    let l2_len =
        ((n as f32 * lod_fractions[2]).round() as usize).clamp(if n > 0 { 1 } else { 0 }, n);
    let l1 = sorted[..l1_len].to_vec();
    let l2 = sorted[..l2_len].to_vec();
    let l3 = if n > 0 { vec![sorted[0]] } else { Vec::new() };

    [l0, l1, l2, l3]
}

/// Bounding-sphere radius of a cluster's AABB (the oracle's `aabb_radius`).
fn aabb_radius(cluster: &SplatCluster) -> f32 {
    ((cluster.aabb_max - cluster.aabb_min) * 0.5)
        .length()
        .max(1e-4)
}

/// Per-cell accumulator for the I0 voxel downsample.
#[derive(Clone, Copy)]
struct CellAccum {
    w_sum: f32,
    pos_w: Vec3,
    pos_plain: Vec3,
    count: u32,
    opacity_sum: u32,
    spectral_w: [f32; 16],
    spectral_plain: [f32; 16],
}

impl CellAccum {
    fn zero() -> Self {
        CellAccum {
            w_sum: 0.0,
            pos_w: Vec3::ZERO,
            pos_plain: Vec3::ZERO,
            count: 0,
            opacity_sum: 0,
            spectral_w: [0.0; 16],
            spectral_plain: [0.0; 16],
        }
    }

    fn add(&mut self, p: Vec3, s: &GaussianSplat) {
        let w = s.opacity() as f32 / 255.0;
        self.w_sum += w;
        self.pos_w += p * w;
        self.pos_plain += p;
        self.count += 1;
        self.opacity_sum += s.opacity() as u32;
        for b in 0..16 {
            let v = f16::from_bits(s.spectral()[b]).to_f32();
            self.spectral_w[b] += v * w;
            self.spectral_plain[b] += v;
        }
    }

    fn mean_position(&self) -> Vec3 {
        if self.w_sum > 0.0 {
            self.pos_w / self.w_sum
        } else {
            self.pos_plain / self.count as f32
        }
    }

    fn mean_spectral(&self) -> [u16; 16] {
        std::array::from_fn(|b| {
            let v = if self.w_sum > 0.0 {
                self.spectral_w[b] / self.w_sum
            } else {
                self.spectral_plain[b] / self.count as f32
            };
            f16::from_f32(v).to_bits()
        })
    }
}

/// Heap entry for budget demotion (pop the *lowest* score first) — verbatim
/// `atom_budget::DemoteEntry` (the tie-break IS the determinism contract).
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
        // work_idx for determinism.
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

/// Heap entry for budget promotion (pop the *highest* score first) — verbatim
/// `atom_budget::PromoteEntry`.
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

/// Working state for one selectable unit (one cluster of one near instance,
/// or one far instance's imposter chain) within a `select()`. `pub(crate)` so
/// the same-crate GPU scorer builds the identical units for the shared
/// [`budget_walk_and_emit`].
#[derive(Clone, Copy)]
pub(crate) struct WorkUnit {
    pub(crate) instance: u32,
    pub(crate) asset: u32,
    /// Cluster id for cluster units; `u32::MAX` for imposter units (also the
    /// shed tie-break key after `(score, instance)`).
    pub(crate) unit_key: u32,
    pub(crate) is_imposter: bool,
    /// Eye distance to the unit's score centre (transformed centroid for
    /// clusters, instance bounds centre for imposters).
    pub(crate) distance: f32,
    /// Importance score = total_opacity * r² / d² — the one formula both
    /// unit kinds compete under.
    pub(crate) score: f32,
    /// Distance-driven ceiling: cluster LOD 0..3, or initial imposter level.
    pub(crate) distance_lod: u8,
    /// Current working level along the unit's chain (clusters L0..L3,
    /// imposters I0..I1).
    pub(crate) lod: u8,
    /// Atom count at the current level.
    pub(crate) count: usize,
}

/// Two-level instance-then-cluster budget cut over an [`AssetAtomLibrary`].
/// Single owner: `&mut self` on `select()` reuses internal scratch buffers.
/// Deterministic; never panics (no instances → `selected = 0`).
pub struct InstancedSelector {
    library: Arc<AssetAtomLibrary>,
    instances: Vec<AtomInstance>,
    /// Reused per-frame scratch.
    work: Vec<WorkUnit>,
    shed_order: Vec<usize>,
}

impl InstancedSelector {
    pub fn new(library: Arc<AssetAtomLibrary>) -> Self {
        InstancedSelector {
            library,
            instances: Vec::new(),
            work: Vec::new(),
            shed_order: Vec::new(),
        }
    }

    /// Replace the placed-instance set (copied; cheap, 48 B POD each).
    /// Instances referencing an unknown asset are skipped at select time.
    pub fn set_instances(&mut self, instances: &[AtomInstance]) {
        self.instances.clear();
        self.instances.extend_from_slice(instances);
    }

    /// Select `≤ budget` atoms worth of draw units for this camera.
    /// Clears + fills `out` (via [`budget_walk_and_emit`], the ONE shared
    /// walk); returns stats. The budget walk is the `AtomBudgetSelector` walk
    /// verbatim (same heap `Ord`s, same op order) extended with the I0→I1
    /// imposter demotion.
    pub fn select(
        &mut self,
        camera: &RenderCamera,
        budget: usize,
        out: &mut InstancedSelection,
    ) -> InstancedStats {
        let start = std::time::Instant::now();

        let library = self.library.clone();
        let lib = library.as_ref();
        let frustum = Frustum::from_view_proj(camera.view_proj());
        let eye = camera.view.inverse().col(3).truncate();

        // --- 1. Instance pass: cull / far-imposter / near-cluster-granular.
        // Output ordered ascending instance index; near-cluster work units
        // ordered (instance index, cluster id).
        let work = &mut self.work;
        work.clear();
        let mut instances_visible = 0usize;
        let mut instances_culled = 0usize;
        let mut instances_far = 0usize;
        for (inst_idx, inst) in self.instances.iter().enumerate() {
            let Some(entry) = lib.assets.get(inst.asset as usize) else {
                instances_culled += 1;
                continue;
            };
            let q = Quat::from_xyzw(
                inst.rotation[0],
                inst.rotation[1],
                inst.rotation[2],
                inst.rotation[3],
            );
            let t = Vec3::from(inst.position);
            let world_center = q.mul_vec3(entry.bounds_center) + t;
            if !frustum.contains_sphere(world_center, entry.bounds_radius) {
                instances_culled += 1;
                continue;
            }
            instances_visible += 1;

            let inst_distance = (world_center - eye).length();
            if inst_distance >= vox_config::config().scatter.far_instance_m {
                // Far: ONE work unit walking the [I0, I1] imposter chain.
                instances_far += 1;
                let d = inst_distance.max(1e-3);
                let radius = entry.bounds_radius;
                let score = entry.total_opacity * (radius * radius) / (d * d);
                let level: u8 = if inst_distance >= vox_config::config().scatter.imposter_i1_m { 1 } else { 0 };
                let count = entry.imposter_ranges[level as usize].1 as usize;
                work.push(WorkUnit {
                    instance: inst_idx as u32,
                    asset: inst.asset,
                    unit_key: u32::MAX,
                    is_imposter: true,
                    distance: d,
                    score,
                    distance_lod: level,
                    lod: level,
                    count,
                });
            } else {
                // Near: per-cluster work units, each individually frustum-
                // tested on its transformed cluster sphere (the oracle's
                // BVH-leaf test: AABB midpoint + half-diagonal radius).
                for (ci, cm) in entry.clusters.iter().enumerate() {
                    let frustum_centre = q.mul_vec3(cm.frustum_center) + t;
                    if !frustum.contains_sphere(frustum_centre, cm.radius) {
                        continue;
                    }
                    // Score/LOD off the transformed centroid — the oracle's
                    // exact formulas.
                    let centre = q.mul_vec3(cm.centroid) + t;
                    let radius = cm.radius;
                    let d = (centre - eye).length().max(1e-3);
                    let screen = projected_screen_size(radius, d);
                    let distance_lod = select_lod_level(d, screen) as u8;
                    let score = cm.total_opacity * (radius * radius) / (d * d);
                    let lod = distance_lod;
                    let count = entry.cluster_ranges[ci][lod as usize].1 as usize;
                    work.push(WorkUnit {
                        instance: inst_idx as u32,
                        asset: inst.asset,
                        unit_key: cm.id,
                        is_imposter: false,
                        distance: d,
                        score,
                        distance_lod,
                        lod,
                        count,
                    });
                }
            }
        }

        // --- 2/3. The SHARED budget walk + emit (also the GPU scorer's
        // tail): demote/promote/shed, then prefix-summed draw emission.
        let (lod_histogram, selected) =
            budget_walk_and_emit(lib, &mut self.work, &mut self.shed_order, budget, out);

        InstancedStats {
            budget,
            selected,
            instances_visible,
            instances_culled,
            instances_far,
            lod_histogram,
            select_us: start.elapsed().as_micros() as u64,
        }
    }
}

/// The shared post-instance-pass budget walk: drive the summed atom count of
/// `work` toward `budget` (the `atom_budget` oracle's demote/promote/shed
/// sequence, verbatim, extended with the I0→I1 imposter demotion) and emit
/// the surviving units into `out` ascending (instance, unit) with
/// prefix-summed `atom_offset`s. Returns `(lod_histogram, selected)`.
///
/// ONE walk, TWO scorers: [`InstancedSelector::select`] (the CPU instance
/// pass) and [`crate::gpu::instanced_select_gpu::InstancedSelectGpu::select`]
/// (the GPU two-kernel scorer) both end HERE — exact CPU/GPU draw equality
/// holds by construction, proven bit-level by the untouched M1 oracle tests.
pub(crate) fn budget_walk_and_emit(
    lib: &AssetAtomLibrary,
    work: &mut [WorkUnit],
    shed_scratch: &mut Vec<usize>,
    budget: usize,
    out: &mut InstancedSelection,
) -> ([usize; 6], usize) {
    out.draws.clear();

    // Atom count of `w` at chain level `lod`.
    let unit_len = |w: &WorkUnit, lod: u8| -> usize {
        let a = &lib.assets[w.asset as usize];
        if w.is_imposter {
            a.imposter_ranges[lod as usize].1 as usize
        } else {
            a.cluster_ranges[w.unit_key as usize][lod as usize].1 as usize
        }
    };
    // Coarsest chain level: I1 for imposters, L3 for clusters.
    let max_level = |w: &WorkUnit| -> u8 {
        if w.is_imposter { 1 } else { (LOD_LEVEL_COUNT - 1) as u8 }
    };

    // --- 2. Drive the summed atom count toward the budget (the oracle's
    // demote/promote/shed sequence, verbatim).
    let mut total: usize = work.iter().map(|w| w.count).sum();

    if total > budget {
        // Demote lowest-score units one level at a time.
        let mut heap: BinaryHeap<DemoteEntry> = BinaryHeap::with_capacity(work.len());
        for (i, w) in work.iter().enumerate() {
            if w.lod < max_level(w) {
                heap.push(DemoteEntry {
                    score: w.score,
                    work_idx: i,
                });
            }
        }
        while total > budget {
            let Some(entry) = heap.pop() else { break };
            let w = work[entry.work_idx];
            if w.lod >= max_level(&w) {
                continue;
            }
            let old = w.count;
            let new_lod = w.lod + 1;
            let new_count = unit_len(&w, new_lod);
            let wm = &mut work[entry.work_idx];
            wm.lod = new_lod;
            wm.count = new_count;
            total = total - old + new_count;
            if new_lod < max_level(&w) {
                heap.push(DemoteEntry {
                    score: w.score,
                    work_idx: entry.work_idx,
                });
            }
        }
        // Every unit floors at 1 atom (L3 / I1), so when MORE UNITS are
        // visible than the budget allows, shed whole lowest-score units
        // — deterministic order (score, instance, unit) ascending.
        if total > budget {
            let shed = shed_scratch;
            shed.clear();
            shed.extend(0..work.len());
            shed.sort_by(|&a, &b| {
                work[a]
                    .score
                    .partial_cmp(&work[b].score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| work[a].instance.cmp(&work[b].instance))
                    .then_with(|| work[a].unit_key.cmp(&work[b].unit_key))
            });
            for &idx in shed.iter() {
                if total <= budget {
                    break;
                }
                let w = &mut work[idx];
                total -= w.count;
                w.count = 0; // shed: emit loop skips zero-count units
            }
        }
    } else if total < budget {
        // Slack: promote highest-score units toward (never above) their
        // distance level to spend the budget.
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
            let next_count = unit_len(&w, next_lod);
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

    // --- 3. Emit draws ascending (instance, unit) with prefix-summed
    // offsets; crossfade exactly as the oracle (cluster units only — far
    // imposters carry scale 1.0 in M1).
    let mut histogram = [0usize; 6];
    let mut offset = 0u32;
    out.draws.reserve(work.len());
    for w in work.iter() {
        if w.count == 0 {
            continue;
        }
        let (unit, opacity_scale) = if w.is_imposter {
            histogram[4 + w.lod as usize] += 1;
            (DrawUnit::Imposter { level: w.lod }, 1.0)
        } else {
            histogram[w.lod as usize] += 1;
            let fade = if w.lod == w.distance_lod {
                crossfade_factor(w.distance, w.lod as u32)
            } else {
                0.0
            };
            (
                DrawUnit::Cluster {
                    id: w.unit_key,
                    lod: w.lod,
                },
                1.0 - fade,
            )
        };
        out.draws.push(ClusterDraw {
            instance: w.instance,
            asset: w.asset,
            unit,
            opacity_scale,
            atom_offset: offset,
            atom_count: w.count as u32,
        });
        offset += w.count as u32;
    }

    (histogram, offset as usize)
}

/// The oracle's projected-size proxy: radius/distance scaled to a pixel-ish
/// magnitude (verbatim `atom_budget::projected_screen_size`).
fn projected_screen_size(radius: f32, distance: f32) -> f32 {
    1000.0 * radius / distance
}

/// Deterministic asset-level imposters. I0 = 4×4×4 voxel downsample over the
/// asset AABB (per occupied cell: opacity-weighted mean position, cell-fitted
/// scale, summed-clamped opacity, opacity-weighted spectral mean; cells
/// emitted in ascending linear cell-key order). I1 = one bounds-fitting atom
/// with the asset's mean colour.
fn build_imposters(
    atoms: &[GaussianSplat],
    aabb_min: Vec3,
    aabb_max: Vec3,
) -> (Vec<GaussianSplat>, GaussianSplat) {
    let size = aabb_max - aabb_min;
    let res = IMPOSTER0_RES;
    let cell_of = |v: f32, mn: f32, sz: f32| -> usize {
        if sz <= 1e-6 {
            0
        } else {
            ((((v - mn) / sz) * res as f32) as usize).min(res - 1)
        }
    };

    let mut cells: Vec<Option<CellAccum>> = vec![None; res * res * res];
    let mut whole = CellAccum::zero();
    for s in atoms {
        let p = Vec3::from(s.position());
        let key = (cell_of(p.x, aabb_min.x, size.x) * res + cell_of(p.y, aabb_min.y, size.y)) * res
            + cell_of(p.z, aabb_min.z, size.z);
        cells[key].get_or_insert_with(CellAccum::zero).add(p, s);
        whole.add(p, s);
    }

    let cell_half = Vec3::new(
        (size.x / res as f32 * 0.5).max(1e-3),
        (size.y / res as f32 * 0.5).max(1e-3),
        (size.z / res as f32 * 0.5).max(1e-3),
    );
    // Ascending linear key = sorted cell-key order (deterministic).
    let i0: Vec<GaussianSplat> = cells
        .iter()
        .flatten()
        .map(|c| {
            GaussianSplat::volume(
                c.mean_position().into(),
                cell_half.into(),
                Quat::IDENTITY,
                c.opacity_sum.min(255) as u8,
                c.mean_spectral(),
            )
        })
        .collect();

    let centre = (aabb_min + aabb_max) * 0.5;
    let half = ((aabb_max - aabb_min) * 0.5).max(Vec3::splat(1e-3));
    let i1 = GaussianSplat::volume(
        centre.into(),
        half.into(),
        Quat::IDENTITY,
        255,
        whole.mean_spectral(),
    );
    (i0, i1)
}

/// Closed-loop frame-budget controller (design §4.7): holds a measured frame
/// time at a setpoint by modulating the per-frame atom budget.
///
/// Dynamics, applied on every [`update`](Self::update) (pinned by the M3
/// plan; design §4.7 verbatim with EMA α = 0.2):
///
/// 1. `ema ← frame_ms` on the first update, `ema ← 0.8·ema + 0.2·frame_ms`
///    after that (`frame_ms` is guarded with `.max(1e-3)` — a zero/negative
///    caller value can never divide-by-zero or invert the loop).
/// 2. `budget ← clamp(round(budget · √(target_ms / ema)), min, max)`.
///
/// The square-root exponent is the design's damping choice: half-strength
/// proportional correction in log space, so a frame measured at 4× the
/// setpoint halves the budget rather than quartering it. Pure control loop —
/// engine-level, nothing game-specific. Never panics.
pub struct FrameBudgetGovernor {
    target_ms: f32,
    budget: usize,
    min: usize,
    max: usize,
    /// Exponential moving average of the measured frame ms; `None` until the
    /// first `update` seeds it.
    ema_ms: Option<f32>,
}

impl FrameBudgetGovernor {
    /// `target_ms` is the setpoint (the M3 gate uses 14.5 ms — 60 fps with
    /// 2.1 ms of headroom); `initial` is the starting budget, clamped into
    /// `[min, max]` like every subsequent value.
    pub fn new(target_ms: f32, initial: usize, min: usize, max: usize) -> Self {
        FrameBudgetGovernor {
            target_ms,
            budget: initial.clamp(min, max),
            min,
            max,
            ema_ms: None,
        }
    }

    /// The budget to use for the NEXT frame's select.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Current smoothed frame time, ms (0.0 until the first `update`).
    pub fn ema_ms(&self) -> f32 {
        self.ema_ms.unwrap_or(0.0)
    }

    /// Feed one measured frame time (the honest full select+expand+render
    /// wall ms) and recompute the budget per the pinned dynamics above.
    pub fn update(&mut self, frame_ms: f32) {
        let frame_ms = frame_ms.max(1e-3);
        let ema = match self.ema_ms {
            None => frame_ms,
            Some(prev) => 0.8 * prev + 0.2 * frame_ms,
        };
        self.ema_ms = Some(ema);
        let next = (self.budget as f32 * (self.target_ms / ema).sqrt()).round() as usize;
        self.budget = next.clamp(self.min, self.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_budget::{AtomBudgetSelector, Selection};
    use crate::spectral::RenderCamera;
    use glam::{Mat4, Quat, Vec3};
    use half::f16;
    use std::f32::consts::FRAC_PI_4;
    use std::sync::Arc;
    use vox_core::types::GaussianSplat;

    /// splitmix64-style scramble (house pattern from `scale_trial`).
    fn hash_u64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    /// Hash a pair of seeds into a uniform `f32` in `[0, 1)`.
    fn hash01(a: u64, b: u64) -> f32 {
        let h = hash_u64(a ^ hash_u64(b.wrapping_mul(0x100_0000_01B3)));
        (h >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Flat-ish 16-band spectrum at `level`, f16 bits.
    fn spectral_level(level: f32) -> [u16; 16] {
        std::array::from_fn(|i| {
            let v = if i < 8 { level } else { level * 0.55 };
            f16::from_f32(v).to_bits()
        })
    }

    /// Deterministic hash-jittered blob asset: `n` atoms in a cylinder of
    /// `radius` × `height` rooted at the local origin. No RNG crate.
    fn synthetic_asset(seed: u64, n: usize, radius: f32, height: f32) -> Vec<GaussianSplat> {
        (0..n)
            .map(|i| {
                let ii = i as u64;
                let ang = hash01(seed ^ ii, 10) * std::f32::consts::TAU;
                let rr = hash01(seed ^ ii, 11).sqrt() * radius;
                let hh = hash01(seed ^ ii, 12) * height;
                let op = 160 + (hash_u64(seed ^ ii.wrapping_mul(2_654_435_761)) % 90) as u8;
                let level = 0.35 + hash01(seed ^ ii, 13) * 0.5;
                GaussianSplat::volume(
                    [rr * ang.cos(), hh, rr * ang.sin()],
                    [0.25, 0.25, 0.25],
                    Quat::IDENTITY,
                    op,
                    spectral_level(level),
                )
            })
            .collect()
    }

    fn camera(eye: Vec3, target: Vec3, fovy: f32, far: f32) -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(fovy, 1.0, 0.1, far),
        }
    }

    // ---------------------------------------------------------------- Task 1

    #[test]
    fn atom_library_dedups_and_bounds_memory() {
        let assets: Vec<Vec<GaussianSplat>> = (0..5)
            .map(|a| synthetic_asset(0xA55E7 + a as u64, 5000, 4.0, 12.0))
            .collect();
        let lib = AssetAtomLibrary::build(&assets, 128);
        assert_eq!(lib.asset_count(), 5);

        // 10k instances would share these atoms — the dedup is the point.
        assert_eq!(std::mem::size_of::<AtomInstance>(), 48, "AtomInstance must stay 48 B");

        let total = lib.total_atoms();
        assert!(
            total > 25_000 && total < 30_000,
            "total_atoms {total} outside (25000, 30000) — imposters missing or duplicated"
        );
        let bytes = lib.resident_bytes();
        assert!(bytes < 64 << 20, "resident_bytes {bytes} >= 64 MiB");
        assert!(bytes > 0, "resident_bytes must be a real measurement");

        let i0: Vec<usize> = (0..5)
            .map(|a| lib.assets[a].imposter0_count as usize)
            .collect();
        for (a, &c) in i0.iter().enumerate() {
            assert!((1..=64).contains(&c), "asset {a} imposter0 count {c} outside 1..=64");
            let (_, len) = lib.unit_range(a as u32, DrawUnit::Imposter { level: 1 });
            assert_eq!(len, 1, "asset {a} must have exactly one I1 atom");
        }
        // The concatenated index list must reference only asset-local atoms:
        // imposter units point at the appended imposter atoms, cluster units at
        // the full atoms.
        for a in 0..5u32 {
            let full = lib.assets[a as usize].full_atom_count;
            let i0c = lib.assets[a as usize].imposter0_count;
            let idx0 = lib.unit_indices(a, DrawUnit::Imposter { level: 0 });
            assert!(
                idx0.iter().all(|&i| i >= full && i < full + i0c),
                "asset {a} I0 indices must point at the appended imposter atoms"
            );
            let l0 = lib.unit_indices(a, DrawUnit::Cluster { id: 0, lod: 0 });
            assert!(
                !l0.is_empty() && l0.iter().all(|&i| i < full),
                "asset {a} cluster L0 indices must point at full atoms"
            );
        }
        assert!(
            lib.atom_index_list().len() > 5 * 5000,
            "index list must hold every cluster LOD prefix"
        );
        println!(
            "[atom_library] assets=5 total_atoms={total} imposter0={i0:?} imposter1=1 resident_bytes={bytes}"
        );
    }

    #[test]
    fn atom_library_imposters_deterministic() {
        let assets: Vec<Vec<GaussianSplat>> = (0..5)
            .map(|a| synthetic_asset(0xD00D + a as u64, 5000, 4.0, 12.0))
            .collect();
        let lib_a = AssetAtomLibrary::build(&assets, 128);
        let lib_b = AssetAtomLibrary::build(&assets, 128);
        let pa = lib_a.packed_atoms();
        let pb = lib_b.packed_atoms();
        assert_eq!(pa.len(), pb.len());
        assert_eq!(
            bytemuck::cast_slice::<PackedLibraryAtom, u8>(pa),
            bytemuck::cast_slice::<PackedLibraryAtom, u8>(pb),
            "two builds must produce byte-identical packed atom tables"
        );

        let mut i0_total = 0usize;
        for a in 0..5u32 {
            // Recompute the asset AABB with the builder's fold.
            let mut mn = Vec3::splat(f32::MAX);
            let mut mx = Vec3::splat(f32::MIN);
            for s in &assets[a as usize] {
                let p = Vec3::from(s.position());
                mn = mn.min(p);
                mx = mx.max(p);
            }
            let base = lib_a.asset_atom_base(a) as usize;
            let full = lib_a.assets[a as usize].full_atom_count as usize;
            let i0 = lib_a.assets[a as usize].imposter0_count as usize;
            assert!((1..=64).contains(&i0));
            i0_total += i0;
            for k in 0..i0 {
                let x = &pa[base + full + k];
                let y = &pb[base + full + k];
                assert_eq!(
                    x.pos_opacity.map(f32::to_bits),
                    y.pos_opacity.map(f32::to_bits),
                    "I0 atom positions must be bit-identical across builds"
                );
                let p = Vec3::new(x.pos_opacity[0], x.pos_opacity[1], x.pos_opacity[2]);
                assert!(
                    p.cmpge(mn - Vec3::splat(1e-4)).all() && p.cmple(mx + Vec3::splat(1e-4)).all(),
                    "asset {a} I0 atom {k} at {p:?} escapes the asset AABB [{mn:?}, {mx:?}]"
                );
            }
            // I1 = single bounds-fitting atom at the AABB centre.
            let i1 = &pa[base + full + i0];
            let centre = (mn + mx) * 0.5;
            assert_eq!(
                [i1.pos_opacity[0], i1.pos_opacity[1], i1.pos_opacity[2]].map(f32::to_bits),
                [centre.x, centre.y, centre.z].map(f32::to_bits),
                "asset {a} I1 must sit exactly at the bounds centre"
            );
        }
        println!(
            "[atom_library] determinism: 2 builds byte-identical; {i0_total} imposter0 positions bit-equal; i1 at bounds centre x5"
        );
    }

    // ---------------------------------------------------------------- Task 2

    const IDENTITY_ROT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    /// ONE identity-transform instance must reproduce the shipped
    /// `AtomBudgetSelector` walk bit-for-bit across budgets, including one
    /// under shed pressure.
    #[test]
    fn instanced_selector_matches_atom_budget_oracle() {
        let atoms = synthetic_asset(0xBEEF, 5000, 4.0, 12.0);
        let lib = Arc::new(AssetAtomLibrary::build(std::slice::from_ref(&atoms), 128));
        let mut isel = InstancedSelector::new(lib.clone());
        isel.set_instances(&[AtomInstance::new(0, 0, [0.0; 3], IDENTITY_ROT)]);
        let mut oracle = AtomBudgetSelector::build(&atoms, 128);

        // Whole asset (blob ±4 m, 0..12 m tall) comfortably inside the
        // frustum at ~30 m so BVH-walk and per-cluster culling agree.
        let cam = camera(
            Vec3::new(0.0, 6.0, -30.0),
            Vec3::new(0.0, 6.0, 0.0),
            FRAC_PI_4,
            2000.0,
        );

        let mut total_k = 0usize;
        for &budget in &[6000usize, 1200, 300, 25] {
            let mut o = Selection::new();
            let so = oracle.select(&cam, budget, &mut o);
            let mut s = InstancedSelection::new();
            let ss = isel.select(&cam, budget, &mut s);

            if budget == 25 {
                assert!(
                    so.clusters_visible > budget,
                    "precondition: budget 25 must be under shed pressure (visible {})",
                    so.clusters_visible
                );
            }

            // Flatten the draws to asset-local atom indices + opacity scales.
            let mut flat_idx: Vec<u32> = Vec::new();
            let mut flat_scale: Vec<f32> = Vec::new();
            for d in s.draws() {
                assert!(
                    matches!(d.unit, DrawUnit::Cluster { .. }),
                    "near identity instance must emit cluster units only, got {:?}",
                    d.unit
                );
                let idx = lib.unit_indices(0, d.unit);
                assert_eq!(idx.len() as u32, d.atom_count);
                flat_idx.extend_from_slice(idx);
                flat_scale.extend(std::iter::repeat_n(d.opacity_scale, idx.len()));
            }

            assert!(!flat_idx.is_empty(), "budget {budget}: selection must be non-trivial");
            assert_eq!(
                flat_idx,
                o.indices(),
                "budget {budget}: atom indices diverge from the oracle"
            );
            assert_eq!(flat_scale.len(), o.opacity_scale().len());
            for (i, (a, b)) in flat_scale.iter().zip(o.opacity_scale()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "budget {budget}: opacity_scale[{i}] not bit-equal ({a} vs {b})"
                );
            }
            assert_eq!(ss.selected, flat_idx.len());
            total_k += flat_idx.len();
        }
        println!("[instanced] oracle match: {total_k} indices bit-equal");
    }

    /// 10,000 instances × 5,000-atom assets = 50M virtual atoms; budget 1M
    /// must be spent (0.9·B..=B), not just bounded.
    #[test]
    fn instanced_selector_budget_bound_at_city_scale() {
        let assets: Vec<Vec<GaussianSplat>> = (0..4)
            .map(|a| synthetic_asset(0xC17 + a as u64, 5000, 4.0, 12.0))
            .collect();
        let lib = Arc::new(AssetAtomLibrary::build(&assets, 128));
        let mut sel = InstancedSelector::new(lib);

        // 100×100 grid, 4 m spacing (±198 m), deterministic yaw, assets
        // round-robin so the virtual count is exactly 10,000 × 5,000.
        let instances: Vec<AtomInstance> = (0..10_000u32)
            .map(|i| {
                let gx = (i % 100) as f32 - 49.5;
                let gz = (i / 100) as f32 - 49.5;
                let q = Quat::from_rotation_y((i % 4) as f32 * std::f32::consts::FRAC_PI_2);
                AtomInstance::new(i % 4, i, [gx * 4.0, 0.0, gz * 4.0], [q.x, q.y, q.z, q.w])
            })
            .collect();
        sel.set_instances(&instances);

        // 100 m straight overhead, wide-angle: the whole grid is in view —
        // instances under the camera stay near (< 150 m), the rest go I0.
        let cam = RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 100.0, 0.0), Vec3::ZERO, Vec3::Z),
            proj: Mat4::perspective_rh(2.5, 1.0, 0.1, 2000.0),
        };

        let budget = 1_000_000usize;
        let mut out = InstancedSelection::new();
        let stats = sel.select(&cam, budget, &mut out);

        assert_eq!(stats.selected, out.atom_count());
        assert!(
            stats.selected >= 900_000 && stats.selected <= 1_000_000,
            "budget not spent: selected {} (want 900000..=1000000)",
            stats.selected
        );
        assert!(stats.instances_visible > 0, "real visible count required");
        assert!(stats.select_us > 0, "select_us must be a real measurement");
        println!(
            "[instanced] 10000 instances x 5000-atom asset = 50000000 virtual atoms | budget={} selected={} visible_instances={} select_ms={:.3}",
            budget,
            stats.selected,
            stats.instances_visible,
            stats.select_us as f64 / 1000.0
        );
    }

    /// The same asset at 30 m renders cluster-granular (nearest clusters L0);
    /// at 600 m it collapses to an imposter of 1..=64 atoms.
    #[test]
    fn instanced_selector_far_imposter_near_detail() {
        let atoms = synthetic_asset(0xFA12, 5000, 15.0, 10.0);
        let lib = Arc::new(AssetAtomLibrary::build(std::slice::from_ref(&atoms), 128));
        let mut sel = InstancedSelector::new(lib);
        sel.set_instances(&[
            AtomInstance::new(0, 0, [0.0, 0.0, 30.0], IDENTITY_ROT),
            AtomInstance::new(0, 1, [0.0, 0.0, 600.0], IDENTITY_ROT),
        ]);
        let cam = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, 100.0), FRAC_PI_4, 2000.0);

        let mut out = InstancedSelection::new();
        let stats = sel.select(&cam, usize::MAX, &mut out);
        assert_eq!(stats.instances_far, 1, "the 600 m instance must classify far");

        let mut near = 0usize;
        let mut far = 0usize;
        let mut near_l0_units = 0usize;
        for d in out.draws() {
            if d.instance == 0 {
                near += d.atom_count as usize;
                assert!(matches!(d.unit, DrawUnit::Cluster { .. }));
                if matches!(d.unit, DrawUnit::Cluster { lod: 0, .. }) {
                    near_l0_units += 1;
                }
            } else {
                far += d.atom_count as usize;
                assert!(
                    matches!(d.unit, DrawUnit::Imposter { .. }),
                    "far instance must emit imposter units, got {:?}",
                    d.unit
                );
            }
        }
        println!("[instanced] near_atoms={near} far_atoms={far}");
        assert!((1..=64).contains(&far), "far atoms {far} outside 1..=64");
        assert!(near >= 50 * far, "near {near} must be >= 50x far {far}");
        assert!(
            near_l0_units > 0,
            "the near instance's nearest clusters must emit at L0"
        );
    }

    #[test]
    fn instanced_selector_deterministic() {
        let assets: Vec<Vec<GaussianSplat>> = (0..2)
            .map(|a| synthetic_asset(0xDE7 + a as u64, 5000, 4.0, 12.0))
            .collect();
        let lib = Arc::new(AssetAtomLibrary::build(&assets, 128));
        let mut sel = InstancedSelector::new(lib);
        let instances: Vec<AtomInstance> = (0..2500u32)
            .map(|i| {
                let gx = (i % 50) as f32 - 24.5;
                let gz = (i / 50) as f32 - 24.5;
                let q = Quat::from_rotation_y((i % 8) as f32 * 0.7853982);
                AtomInstance::new(i % 2, i, [gx * 4.0, 0.0, gz * 4.0], [q.x, q.y, q.z, q.w])
            })
            .collect();
        sel.set_instances(&instances);
        let cam = RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 80.0, 0.0), Vec3::ZERO, Vec3::Z),
            proj: Mat4::perspective_rh(2.2, 1.0, 0.1, 2000.0),
        };

        let mut a = InstancedSelection::new();
        let mut b = InstancedSelection::new();
        sel.select(&cam, 200_000, &mut a);
        sel.select(&cam, 200_000, &mut b);
        assert!(!a.draws().is_empty(), "scene must emit draws");
        assert_eq!(a.draws(), b.draws(), "draw vectors must be exactly equal");
        println!(
            "[instanced] deterministic: {} draws identical across 2 selects",
            a.draws().len()
        );
    }

    #[test]
    fn instanced_selector_no_instances_selects_zero() {
        let atoms = synthetic_asset(0x0, 500, 4.0, 12.0);
        let lib = Arc::new(AssetAtomLibrary::build(std::slice::from_ref(&atoms), 128));
        let mut sel = InstancedSelector::new(lib);
        let cam = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, 100.0), FRAC_PI_4, 2000.0);

        let mut out = InstancedSelection::new();
        // Never called set_instances at all.
        let stats = sel.select(&cam, 1_000_000, &mut out);
        assert_eq!(stats.selected, 0);
        assert!(out.draws().is_empty());
        assert_eq!(out.atom_count(), 0);

        // And explicitly empty.
        sel.set_instances(&[]);
        let stats = sel.select(&cam, 1_000_000, &mut out);
        assert_eq!(stats.selected, 0);
        assert_eq!(stats.instances_visible, 0);
        println!("[instanced] no instances -> selected=0 draws=0 (no panic)");
    }

    /// Design §4.7 convergence: against the linear cost model
    /// `ms = 6 + budget/50_000` the analytic settle point is
    /// `(14.5 − 6) · 50_000 = 425_000` atoms at exactly 14.5 ms. After 100
    /// closed-loop updates the governor must sit within ±10% of both.
    #[test]
    fn governor_converges_on_linear_cost() {
        let cost = |b: usize| 6.0 + b as f32 / 50_000.0;
        let mut g = FrameBudgetGovernor::new(14.5, 1_000_000, 100_000, 1_000_000);
        for _ in 0..100 {
            let ms = cost(g.budget());
            g.update(ms);
        }
        assert!(
            (g.ema_ms() - 14.5).abs() < 1.45,
            "ema must settle at the 14.5 ms setpoint, got {}",
            g.ema_ms()
        );
        assert!(
            (382_500..=467_500).contains(&g.budget()),
            "budget must settle near the analytic 425_000, got {}",
            g.budget()
        );
        println!(
            "[governor] linear-cost settle: budget={} ema_ms={:.2} after 100 updates",
            g.budget(),
            g.ema_ms()
        );
    }

    /// Design §4.7 rails: a flat-cheap frame (1 ms regardless of budget) must
    /// rail the budget to `max`; a flat-expensive frame (100 ms) to `min`.
    #[test]
    fn governor_clamps_at_rails() {
        let mut cheap = FrameBudgetGovernor::new(14.5, 300_000, 100_000, 1_000_000);
        for _ in 0..100 {
            cheap.update(1.0);
        }
        assert_eq!(
            cheap.budget(),
            1_000_000,
            "flat-cheap cost must rail the budget to max"
        );

        let mut expensive = FrameBudgetGovernor::new(14.5, 300_000, 100_000, 1_000_000);
        for _ in 0..100 {
            expensive.update(100.0);
        }
        assert_eq!(
            expensive.budget(),
            100_000,
            "flat-expensive cost must rail the budget to min"
        );
        println!(
            "[governor] clamps: cheap-> {} expensive-> {}",
            cheap.budget(),
            expensive.budget()
        );
    }
}
