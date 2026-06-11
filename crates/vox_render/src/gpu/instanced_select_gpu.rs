//! `InstancedSelectGpu` — instance+cluster scoring on the shared context (M3
//! of virtualized splat rendering).
//!
//! A shared-[`GpuContext`] twin of
//! [`InstancedSelector`](crate::atom_instances::InstancedSelector) following
//! the proven [`crate::gpu::atom_budget_gpu::AtomBudgetGpu`] doctrine: the
//! embarrassingly parallel, rounding-sensitive SCORING runs on device in the
//! oracle's exact op order (no fast-math — mirroring the op order is the
//! determinism mechanism), while the tiny sequential BUDGET WALK stays on the
//! host. Unlike `AtomBudgetGpu`, the walk is not mirrored: both selectors call
//! the SAME [`budget_walk_and_emit`] — one walk, two scorers — so GPU draws
//! equal CPU draws exactly, by construction.
//!
//! ## The fused chain (`instanced_select_gpu.wgsl`, M3.1 Task 4)
//!
//! ONE submit, ONE readback. Four passes in one command encoder:
//!
//! 1. `score_instances` — one thread per placed instance: transform the asset
//!    bounds sphere, frustum-classify cull / far / near, and score far
//!    imposters (`total_opacity · r²/d²`, I0/I1 level at the 150 m / 400 m
//!    thresholds).
//! 2. `scan_pairs` — ONE thread, a deterministic serial prefix scan over the
//!    kernel-1 results in ascending instance order: per-near-instance pair
//!    bases, the total pair demand into `PairMeta`, and the kernel-2 indirect
//!    args (capped to ZERO workgroups when demand overflows `max_near_units`).
//! 3. `emit_pairs` — one thread per instance, writing the near `(instance,
//!    cluster)` pairs at the scanned bases — ascending (instance, cluster),
//!    EXACTLY the order the deleted host pair loop built.
//! 4. `score_pairs` — one thread per near pair (dispatched INDIRECT off the
//!    scan's args, guarded by `PairMeta.pair_count`): transformed
//!    cluster-sphere frustum flag, distance LOD and score off the transformed
//!    centroid.
//!
//! The encoder then copies `PairMeta` (16 B) + the instance scores (n × 16 B)
//! + the FULL pair-score region (`max_near_units` × 16 B — `pair_count` is
//! unknown host-side before the map; the ≤4 MB device→readback copy is the
//! price of one sync) into one fused readback buffer: one submit, one
//! blocking `map_read`. The host then assembles the identical `WorkUnit`
//! stream (far units interleaved with near units in instance order) and runs
//! the shared walk.
//!
//! ## The GPU-resident walk (`select_resident` — M3.2, the M3.1-named escalation)
//!
//! [`select_resident`](InstancedSelectGpu::select_resident) moves the budget
//! walk itself onto the device: ONE submit runs the fused scoring chain, the
//! unit build, a sort-free **radix-bucketed threshold refinement**, a
//! two-level deterministic u32 exclusive scan, and device-side `ExpandDraw`
//! emission; ONE 64 B stats readback is the only sync. The CPU
//! [`budget_walk_and_emit`] is UNTOUCHED and remains the bit-equality oracle
//! in tests only (the `resident_walk_` suite) — the host walk `select()` and
//! its suite stay as the regression bed and the fallback bridge.
//!
//! **The equivalence theorem the port rests on** (verified against
//! `atom_instances.rs`):
//!
//! 1. `DemoteEntry::cmp` (`atom_instances.rs:514–524`) reverses BOTH score
//!    and `work_idx` under the max-`BinaryHeap` ⇒ pop order is ascending
//!    `(score, work_idx)`. A demoted unit re-enters at the SAME unique key ⇒
//!    each unit demotes FULLY before its successor starts, except the single
//!    boundary unit where the freed total crosses `total0 − budget`
//!    (`:777–796`). The demote end state is therefore a pure THRESHOLD over
//!    the key: below the boundary fully demoted, the boundary unit partially
//!    (minimal levels closing the gap), above untouched.
//! 2. The shed order `(score, instance, unit_key)` (`:804–811`) equals
//!    ascending `(score, work_idx)` because the work stream ascends
//!    `(instance, cluster id)`. Shed end state: the minimal key-prefix whose
//!    summed `count(max_level)` reaches the residual need, boundary unit
//!    INCLUSIVE (`:812–819`).
//! 3. The promote branch (`:821–857`) is **structurally dead** for both
//!    scorers: every unit enters with `lod == distance_lod`
//!    (`atom_instances.rs:695–696` and this module's k1/k2 mirrors), so the
//!    promote heap starts empty. The GPU walk implements demote+shed+emit
//!    only; the oracle suite is the runtime check.
//!
//! The refinement finds the boundary KEY without sorting: 8 passes over the
//! 64-bit `(score-bits, tie)` key, 8 bits per pass — every live matching unit
//! `atomicAdd`s its u32 weight into 256 buckets (deterministic by integer
//! associativity/commutativity regardless of arrival order), a serial pick
//! fixes the next 8 prefix bits where the ascending bucket accumulation
//! crosses the need. The frozen `RadixSortPass` is NOT reused — its scatter
//! is proven run-to-run nondeterministic (M3.1 Task 5), and the walk never
//! needed a sorted array, only the boundary key and the sums below it.
//!
//! ## The CPU fallback
//!
//! A near-pair demand above `max_near_units` (read from the scanned
//! `PairMeta.pair_count` AFTER the single readback; kernel 2 already
//! dispatched zero workgroups) falls back to the internal, always-in-sync CPU
//! [`InstancedSelector`] — bit-identical output by the shared-walk contract —
//! and increments [`fallback_count`](InstancedSelectGpu::fallback_count).
//! Buffer limits (including the fused readback size) are validated UP FRONT
//! at construction ([`InstancedSelectGpuError::ExceedsDeviceLimits`]), never
//! as an uncaptured wgpu validation abort.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::atom_instances::{
    AssetAtomLibrary, AtomInstance, DrawUnit, InstancedSelection, InstancedSelector,
    InstancedStats, WorkUnit, budget_walk_and_emit,
};
use crate::gpu::GpuContext;
use crate::gpu::atom_budget_gpu::{GpuPlane, camera_eye, frustum_planes};
use crate::gpu::gpu_timing::GpuTimers;
use crate::spectral::RenderCamera;

/// Bytes per uploaded instance (`AtomInstance` POD — mirrored 48 B in WGSL).
const INSTANCE_BYTES: u64 = std::mem::size_of::<AtomInstance>() as u64;
/// Bytes per near (instance, cluster) pair (`2 × u32`).
const PAIR_BYTES: u64 = 8;
/// Bytes per kernel output record (both kernels emit 16 B PODs).
const SCORE_BYTES: u64 = 16;
/// Bytes of the scanned [`PairMeta`] readback header.
const PAIR_META_BYTES: u64 = 16;

/// Bytes per GPU walk unit (`WalkUnit` in the shader).
const WALK_UNIT_BYTES: u64 = 32;
/// Bytes per device-emitted draw (`ExpandDraw` — `expand_draws.rs`'s layout).
const DRAW_BYTES: u64 = 32;
/// Bytes per `counts` entry (vec2<u32>: atom count / draw flag, scanned to
/// atom offset / draw slot).
const COUNT_BYTES: u64 = 8;
/// Size of the walk-state buffer (`WalkState` in the shader: 256 u32 buckets
/// + scalars + the constant tail), padded to 16.
const WALK_STATE_BYTES: u64 = 1104;
/// The cleared prefix of the walk state — everything BEFORE the constant
/// `blocks_base` field at offset 1096 (written once at construction).
const WALK_STATE_CLEAR_BYTES: u64 = 1096;
/// Byte offset of `WalkState.blocks_base` (the constant tail).
const WALK_STATE_BLOCKS_BASE_OFFSET: usize = 1096;
/// Bytes of the [`GpuWalkStats`] block — THE one `select_resident` readback.
const STATS_BYTES: u64 = 64;

/// Dispatch params uniform (mirrors `Params` in the shader). Same 32 B layout
/// as pre-Task-4; field 2 repurposed `pair_count` → `max_near_units` (the
/// scan computes the pair count GPU-side); field 4 repurposed `_pad` →
/// `budget` (M3.2 — read only by the resident-walk kernels).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    instance_count: u32,
    max_near_units: u32,
    asset_count: u32,
    budget: u32,
    eye: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Params>() == 32);

/// Per-asset static upload (mirrors `AssetInfo` in the shader): local bounds
/// sphere, summed cluster opacity, and the asset's slice of the flattened
/// global cluster-meta table for the GPU pair scan/emit. Repacked at the SAME
/// 32 B as pre-Task-4 — `opacity` is the old `opacity[0]` (the only lane ever
/// read), so kernel 1's score math is bit-identical. M3.2 repurposed the pad
/// lane as `atom_base` (the packed-atom offset `emit_draws` writes into
/// `ExpandDraw.atom_base`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAssetInfo {
    center_radius: [f32; 4],
    opacity: f32,
    cluster_base: u32,
    cluster_count: u32,
    atom_base: u32,
}

const _: () = assert!(std::mem::size_of::<GpuAssetInfo>() == 32);

/// The scanned pair-demand header (mirrors `PairMeta` in the shader): first
/// 16 B of the fused readback. `pair_count > max_near_units` ⇒ kernel 2 ran
/// zero workgroups and the host takes the bit-identical CPU fallback.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct PairMeta {
    pair_count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

const _: () = assert!(std::mem::size_of::<PairMeta>() as u64 == PAIR_META_BYTES);

/// One flattened cluster meta (mirrors `ClusterMeta` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuClusterMeta {
    frustum_radius: [f32; 4],
    centroid_opacity: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuClusterMeta>() == 32);

/// Kernel-1 readback POD (mirrors `InstScore`): `kind` 0 cull / 1 far / 2
/// near. For far instances `level` is the imposter level (0/1); for near
/// instances it carries the asset's cluster_count for the GPU serial scan and
/// is never read host-side.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GpuInstScore {
    kind: u32,
    level: u32,
    score: f32,
    distance: f32,
}

const _: () = assert!(std::mem::size_of::<GpuInstScore>() as u64 == SCORE_BYTES);

/// Kernel-2 readback POD (mirrors `PairScore`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GpuPairScore {
    passes: u32,
    distance_lod: u32,
    score: f32,
    distance: f32,
}

const _: () = assert!(std::mem::size_of::<GpuPairScore>() as u64 == SCORE_BYTES);

/// The 64 B stats block — `select_resident`'s ONE readback (mirrors
/// `WalkStats` in the shader). `pair_count` is the fallback decision;
/// everything else fills [`InstancedStats`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GpuWalkStats {
    pair_count: u32,
    selected: u32,
    draw_count: u32,
    visible: u32,
    culled: u32,
    far_count: u32,
    lod_histogram: [u32; 6],
    _pad: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuWalkStats>() as u64 == STATS_BYTES);

/// Which path a [`select_resident`](InstancedSelectGpu::select_resident) call
/// took: draws GPU-resident (consume via
/// [`ExpandDrawsPass::encode_indirect`](crate::gpu::expand_draws::ExpandDrawsPass)),
/// or the bit-identical CPU twin after a near-pair overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectPath {
    GpuResident,
    CpuFallback,
}

/// Error returned when the GPU selector cannot be created or run. Same shape
/// class as [`crate::gpu::atom_budget_gpu::AtomBudgetGpuError`] — never an
/// uncaptured wgpu validation abort.
#[derive(Debug, Clone)]
pub enum InstancedSelectGpuError {
    /// A required GPU buffer would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`), or the
    /// instance set exceeds the constructed `max_instances`.
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
    /// Mapping a readback buffer failed.
    Readback(String),
}

impl std::fmt::Display for InstancedSelectGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstancedSelectGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => write!(
                f,
                "GPU buffer '{what}' requires {requested} bytes, exceeding device limit {limit}"
            ),
            InstancedSelectGpuError::Readback(e) => write!(f, "GPU readback failed: {e}"),
        }
    }
}

impl std::error::Error for InstancedSelectGpuError {}

/// Per-[`select`](InstancedSelectGpu::select) stage breakdown (M3.1 Task 1):
/// host wall spans plus optional hardware-timestamp readings for the two
/// scoring kernels. All spans are milliseconds. `Default` (all zero / `None`)
/// until the first select.
#[derive(Clone, Copy, Debug, Default)]
pub struct SelectGpuBreakdown {
    /// Sum of the blocking GPU spans this select performed (each measured
    /// submit → map-complete).
    pub gpu_span_ms: f32,
    /// Kernel-1 (`score_instances`) hardware timestamp. `None` when the device
    /// lacks `TIMESTAMP_QUERY` (test devices request `Features::empty()`).
    pub k1_gpu_ms: Option<f32>,
    /// Kernel-2 (`score_pairs`) hardware timestamp — the fused encoder always
    /// runs the k2 pass (zero indirect workgroups when there are no pairs), so
    /// this is `None` only when timers are disabled.
    pub k2_gpu_ms: Option<f32>,
    /// Host near-pair build + upload span. 0.0 since Task 4 moved pair
    /// emission GPU-side (`scan_pairs` + `emit_pairs`) — kept so the harness
    /// stage table proves the host loop stays deleted.
    pub pair_build_ms: f32,
    /// `WorkUnit` stream assembly span (the kernel-readback → walk bridge).
    pub assemble_ms: f32,
    /// The shared [`budget_walk_and_emit`] span. Exactly 0.0 on the
    /// GPU-resident path — the walk runs on device (see `walk_gpu_ms`).
    pub walk_ms: f32,
    /// Blocking readback polls this select performed: 1 on the fused GPU path
    /// (including the CPU-fallback path — the fallback decision reads the same
    /// single fused readback), 0 for an empty instance set.
    pub syncs: u32,
    /// The GPU walk span (build units → emit_finalize) hardware timestamp,
    /// `select_resident` only. `None` when the device lacks `TIMESTAMP_QUERY`
    /// or on the host-walk `select()` path. Constructed internally only.
    pub walk_gpu_ms: Option<f32>,
}

/// GPU instance+cluster scorer feeding the shared host budget walk. Owns
/// persistent buffers sized at construction; per
/// [`select`](Self::select) only the planes/params (and the near-pair list)
/// are re-uploaded. Single owner: `&mut self` reuses internal scratch.
pub struct InstancedSelectGpu {
    ctx: GpuContext,
    pipeline_instances: wgpu::ComputePipeline,
    pipeline_scan: wgpu::ComputePipeline,
    pipeline_emit: wgpu::ComputePipeline,
    pipeline_pairs: wgpu::ComputePipeline,
    /// Per-kernel persistent bind groups, each carrying exactly the bindings
    /// its entry point uses (the union exceeds the default
    /// `max_storage_buffers_per_shader_stage` limit of 8). `bg_pairs` has no
    /// `k2_args` binding, so the indirect dispatch's usage scope never sees
    /// the indirect buffer as writable storage (wgpu usage-conflict rule).
    bg_instances: wgpu::BindGroup,
    bg_scan: wgpu::BindGroup,
    bg_emit: wgpu::BindGroup,
    bg_pairs: wgpu::BindGroup,

    // ── The M3.2 resident-walk kernels (one module, per-kernel layouts). ──
    pipeline_build_far: wgpu::ComputePipeline,
    pipeline_build_near: wgpu::ComputePipeline,
    pipeline_spread: wgpu::ComputePipeline,
    pipeline_pick: wgpu::ComputePipeline,
    pipeline_walk_finalize: wgpu::ComputePipeline,
    pipeline_final_counts: wgpu::ComputePipeline,
    pipeline_scan_blocks: wgpu::ComputePipeline,
    pipeline_scan_totals: wgpu::ComputePipeline,
    pipeline_add_back: wgpu::ComputePipeline,
    pipeline_emit_draws: wgpu::ComputePipeline,
    pipeline_emit_finalize: wgpu::ComputePipeline,
    bg_build_far: wgpu::BindGroup,
    bg_build_near: wgpu::BindGroup,
    bg_spread: wgpu::BindGroup,
    bg_pick: wgpu::BindGroup,
    bg_walk_finalize: wgpu::BindGroup,
    bg_final_counts: wgpu::BindGroup,
    bg_scan_blocks: wgpu::BindGroup,
    bg_scan_totals: wgpu::BindGroup,
    bg_add_back: wgpu::BindGroup,
    bg_emit_draws: wgpu::BindGroup,
    bg_emit_finalize: wgpu::BindGroup,

    instances_buf: wgpu::Buffer,
    planes_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    inst_out: wgpu::Buffer,
    pair_out: wgpu::Buffer,
    /// The scanned `PairMeta` (16 B, STORAGE|COPY_SRC) — copied into the head
    /// of the fused readback.
    pair_meta_buf: wgpu::Buffer,
    /// Kernel-2 indirect dispatch args (12 B, STORAGE|INDIRECT), written by
    /// `scan_pairs`.
    k2_args_buf: wgpu::Buffer,
    /// THE one readback: `PairMeta` + instance scores + the full pair-score
    /// region, mapped once per select.
    fused_readback: wgpu::Buffer,

    // ── M3.2 resident-walk buffers. ──
    /// `WalkState`: buckets + scalars (cleared per resident select) + the
    /// constant `blocks_base` tail.
    walk_state_buf: wgpu::Buffer,
    /// 3 × 12 B indirect dispatch-args slots, written by `scan_pairs`.
    walk_args_buf: wgpu::Buffer,
    /// Device-emitted `ExpandDraw`s — bound UNMODIFIED by the indirect expand.
    draws_buf: wgpu::Buffer,
    /// GPU-written `ExpandParams` (16 B, UNIFORM|STORAGE — storage here,
    /// uniform at binding 0 of the indirect expand).
    expand_params_buf: wgpu::Buffer,
    /// Expand indirect dispatch args (12 B, STORAGE|INDIRECT).
    expand_args_buf: wgpu::Buffer,
    /// The 64 B `GpuWalkStats` block (STORAGE|COPY_SRC) + its MAP_READ twin.
    stats_buf: wgpu::Buffer,
    stats_readback: wgpu::Buffer,

    library: Arc<AssetAtomLibrary>,
    max_instances: u32,
    max_near_units: u32,

    /// Host copy of the placed instances (pass-B assembly).
    instances: Vec<AtomInstance>,
    /// The always-in-sync CPU twin — the deterministic near-overflow fallback.
    cpu: InstancedSelector,
    fallback_count: u64,

    // Reused per-select scratch.
    work: Vec<WorkUnit>,
    shed_scratch: Vec<usize>,

    /// 3 timestamp pairs: slot 0 = `score_instances`, slot 1 = `score_pairs`,
    /// slot 2 = the resident walk span (build units → emit_finalize, split
    /// begin/end writes). Inert (all-`None` readings) when the device lacks
    /// `TIMESTAMP_QUERY`.
    timers: GpuTimers,
    /// Stage breakdown of the previous select — see [`Self::last_breakdown`].
    last_breakdown: SelectGpuBreakdown,
}

impl InstancedSelectGpu {
    /// Build the scorer on the shared context: validate every buffer
    /// (including the fused readback) against
    /// `max_storage_buffer_binding_size` / `max_buffer_size` UP FRONT, upload
    /// the per-asset bounds+cluster-slice table and the flattened global
    /// cluster-meta table ONCE, and size the persistent
    /// instance/pair/score/scan buffers from `max_instances` /
    /// `max_near_units`.
    pub fn new_with_context(
        ctx: &GpuContext,
        library: Arc<AssetAtomLibrary>,
        max_instances: u32,
        max_near_units: u32,
    ) -> Result<Self, InstancedSelectGpuError> {
        let device = ctx.device();

        // Flatten the library's cluster metas (ascending asset, ascending
        // cluster id); each asset's base+count into the flattened table ride
        // in `GpuAssetInfo` for the GPU-side pair scan/emit.
        let asset_count = library.asset_count() as u32;
        let mut metas: Vec<GpuClusterMeta> = Vec::new();
        let mut infos: Vec<GpuAssetInfo> = Vec::with_capacity(asset_count as usize);
        for a in 0..asset_count {
            let cluster_base = metas.len() as u32;
            for cm in library.cluster_metas(a) {
                metas.push(GpuClusterMeta {
                    frustum_radius: [
                        cm.frustum_center.x,
                        cm.frustum_center.y,
                        cm.frustum_center.z,
                        cm.radius,
                    ],
                    centroid_opacity: [cm.centroid.x, cm.centroid.y, cm.centroid.z, cm.total_opacity],
                });
            }
            let (centre, radius) = library.asset_bounds(a);
            infos.push(GpuAssetInfo {
                center_radius: [centre.x, centre.y, centre.z, radius],
                opacity: library.asset_total_opacity(a),
                cluster_base,
                cluster_count: library.cluster_count(a) as u32,
                atom_base: library.asset_atom_base(a),
            });
        }

        // The flat (index_offset, len) range table for the resident walk:
        // 4 LOD slots per GLOBAL cluster (the metas' flattening order), then
        // 2 imposter slots per asset at imposter_base = total_clusters * 4.
        let mut range_table: Vec<[u32; 2]> = Vec::with_capacity(metas.len() * 4 + infos.len() * 2);
        for a in 0..asset_count {
            for id in 0..library.cluster_count(a) as u32 {
                for lod in 0..crate::hierarchical_lod::LOD_LEVEL_COUNT {
                    let (off, len) =
                        library.unit_range(a, DrawUnit::Cluster { id, lod: lod as u8 });
                    range_table.push([off, len]);
                }
            }
        }
        for a in 0..asset_count {
            for level in 0..2u8 {
                let (off, len) = library.unit_range(a, DrawUnit::Imposter { level });
                range_table.push([off, len]);
            }
        }

        let inst_bytes = (max_instances as u64 * INSTANCE_BYTES).max(INSTANCE_BYTES);
        let inst_out_bytes = (max_instances as u64 * SCORE_BYTES).max(SCORE_BYTES);
        let pair_bytes = (max_near_units as u64 * PAIR_BYTES).max(PAIR_BYTES);
        let pair_out_bytes = (max_near_units as u64 * SCORE_BYTES).max(SCORE_BYTES);
        let pair_base_bytes = (max_instances as u64 * 4).max(4);
        let assets_bytes = (infos.len() as u64 * std::mem::size_of::<GpuAssetInfo>() as u64)
            .max(std::mem::size_of::<GpuAssetInfo>() as u64);
        let metas_bytes = (metas.len() as u64 * std::mem::size_of::<GpuClusterMeta>() as u64)
            .max(std::mem::size_of::<GpuClusterMeta>() as u64);
        let plane_bytes = 6 * std::mem::size_of::<GpuPlane>() as u64;
        // THE one readback: PairMeta + instance scores + the FULL pair-score
        // region (pair_count is unknown host-side before the map).
        let fused_readback_bytes = PAIR_META_BYTES + inst_out_bytes + pair_out_bytes;

        // The walk's unit domain: one slot per far instance + one per near
        // pair, padded to a whole number of 256-wide scan blocks so the block
        // totals start at a clean `blocks_base` slot inside `counts`.
        let units_cap = max_instances as u64 + max_near_units as u64;
        let units_padded = units_cap.div_ceil(256).max(1) * 256;
        let blocks_cap = units_padded / 256;
        let unit_base_bytes = (max_instances as u64 * 4).max(4);
        let units_bytes = units_padded * WALK_UNIT_BYTES;
        let counts_bytes = (units_padded + blocks_cap) * COUNT_BYTES;
        let walk_draws_bytes = units_padded * DRAW_BYTES;
        let ranges_bytes = (range_table.len() as u64 * 8).max(8);
        if units_padded > u32::MAX as u64 {
            return Err(InstancedSelectGpuError::ExceedsDeviceLimits {
                what: "instanced_select_units (cap exceeds u32)",
                requested: units_padded,
                limit: u32::MAX as u64,
            });
        }

        // Validate every STORAGE buffer BEFORE creating + binding it (the
        // wave-13 / atom_budget_gpu no-panic contract).
        let limits = device.limits();
        let max_storage = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        if fused_readback_bytes > max_buffer {
            return Err(InstancedSelectGpuError::ExceedsDeviceLimits {
                what: "instanced_select_fused_readback",
                requested: fused_readback_bytes,
                limit: max_buffer,
            });
        }
        for (what, bytes) in [
            ("instanced_select_instances", inst_bytes),
            ("instanced_select_inst_scores", inst_out_bytes),
            ("instanced_select_pairs", pair_bytes),
            ("instanced_select_pair_scores", pair_out_bytes),
            ("instanced_select_pair_base", pair_base_bytes),
            ("instanced_select_asset_infos", assets_bytes),
            ("instanced_select_cluster_metas", metas_bytes),
            ("instanced_select_planes", plane_bytes),
            ("instanced_select_unit_base", unit_base_bytes),
            ("instanced_select_units", units_bytes),
            ("instanced_select_counts", counts_bytes),
            ("instanced_select_walk_draws", walk_draws_bytes),
            ("instanced_select_ranges", ranges_bytes),
            ("instanced_select_walk_state", WALK_STATE_BYTES),
            ("instanced_select_stats", STATS_BYTES),
        ] {
            if bytes > max_storage {
                return Err(InstancedSelectGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage,
                });
            }
            if bytes > max_buffer {
                return Err(InstancedSelectGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        // ── Library tables — uploaded ONCE (kept alive by the bind group). ──
        let assets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instanced_select_asset_infos"),
            contents: if infos.is_empty() {
                bytemuck::bytes_of(&GpuAssetInfo::zeroed()).to_vec()
            } else {
                bytemuck::cast_slice(&infos).to_vec()
            }
            .as_slice(),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let metas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instanced_select_cluster_metas"),
            contents: if metas.is_empty() {
                bytemuck::bytes_of(&GpuClusterMeta::zeroed()).to_vec()
            } else {
                bytemuck::cast_slice(&metas).to_vec()
            }
            .as_slice(),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── Persistent per-select buffers. ──────────────────────────────────
        let instances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_instances"),
            size: inst_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // GPU-emitted (scan_pairs/emit_pairs) — the host never writes pairs.
        let pairs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_pairs"),
            size: pair_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pair_base_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_pair_base"),
            size: pair_base_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pair_meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_pair_meta"),
            size: PAIR_META_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let k2_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_k2_args"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let planes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_planes"),
            size: plane_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let inst_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_inst_scores"),
            size: inst_out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pair_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_pair_scores"),
            size: pair_out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fused_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_fused_readback"),
            size: fused_readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── M3.2 resident-walk buffers. ─────────────────────────────────────
        let ranges_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instanced_select_ranges"),
            contents: if range_table.is_empty() {
                bytemuck::bytes_of(&[0u32; 2]).to_vec()
            } else {
                bytemuck::cast_slice(&range_table).to_vec()
            }
            .as_slice(),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let unit_base_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_unit_base"),
            size: unit_base_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let units_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_units"),
            size: units_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_counts"),
            size: counts_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // WalkState: zeroed at creation, the constant `blocks_base` tail
        // written once HERE; the encoder clears only the mutable prefix.
        let walk_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_walk_state"),
            size: WALK_STATE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = walk_state_buf.slice(..).get_mapped_range_mut();
            view.fill(0);
            view[WALK_STATE_BLOCKS_BASE_OFFSET..WALK_STATE_BLOCKS_BASE_OFFSET + 4]
                .copy_from_slice(&(units_padded as u32).to_le_bytes());
        }
        walk_state_buf.unmap();
        let walk_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_walk_args"),
            size: 36,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let draws_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_walk_draws"),
            size: walk_draws_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let expand_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_expand_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let expand_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_expand_args"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_stats"),
            size: STATS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instanced_select_stats_readback"),
            size: STATS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Pipelines (one module, four entry points) + FOUR persistent bind
        //    groups, one per kernel — every binding a persistent buffer. Each
        //    kernel's layout carries EXACTLY the bindings its entry point
        //    statically uses: the union is 10 storage buffers, but the
        //    default `max_storage_buffers_per_shader_stage` limit is 8, so a
        //    shared all-bindings layout cannot exist on default-limit devices
        //    (per-kernel: k1=4, scan=4, emit=5, k2=6 — all within the limit).
        //    A welcome consequence: `score_pairs`' layout has no `k2_args`
        //    binding, so the indirect dispatch's usage scope never sees the
        //    indirect buffer as writable storage (wgpu usage-conflict rule).──
        let shader = device.create_shader_module(wgpu::include_wgsl!("instanced_select_gpu.wgsl"));

        // The global binding map (mirrors the WGSL @binding numbers):
        // 0=params 1=instances 2=assets 3=cluster_metas 4=planes 5=pairs
        // 6=out_instances 7=out_pairs 8=pair_base 9=pair_meta 10=k2_args
        // 11=ubase 12=units 13=counts 14=walk_state 15=walk_args 16=ranges
        // 17=draws 18=expand_params 19=expand_args 20=stats.
        let binding_buf: [&wgpu::Buffer; 21] = [
            &params_buf,
            &instances_buf,
            &assets_buf,
            &metas_buf,
            &planes_buf,
            &pairs_buf,
            &inst_out,
            &pair_out,
            &pair_base_buf,
            &pair_meta_buf,
            &k2_args_buf,
            &unit_base_buf,
            &units_buf,
            &counts_buf,
            &walk_state_buf,
            &walk_args_buf,
            &ranges_buf,
            &draws_buf,
            &expand_params_buf,
            &expand_args_buf,
            &stats_buf,
        ];
        let layout_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                // 0 = the params uniform; 1–4 + 16 read-only tables; the rest
                // written by at least one kernel (read_write in the WGSL).
                ty: if binding == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding <= 4 || binding == 16,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let make_kernel = |entry: &str, label: &'static str, bindings: &[u32]| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> =
                bindings.iter().map(|&b| layout_entry(b)).collect();
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let resources: Vec<wgpu::BindGroupEntry> = bindings
                .iter()
                .map(|&b| wgpu::BindGroupEntry {
                    binding: b,
                    resource: binding_buf[b as usize].as_entire_binding(),
                })
                .collect();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &resources,
            });
            (pipeline, bind_group)
        };
        let (pipeline_instances, bg_instances) =
            make_kernel("score_instances", "instanced_select_k1", &[0, 1, 2, 4, 6]);
        let (pipeline_scan, bg_scan) = make_kernel(
            "scan_pairs",
            "instanced_select_scan",
            // Exactly 8 storage bindings — the default per-stage limit.
            &[0, 6, 8, 9, 10, 11, 14, 15, 20],
        );
        let (pipeline_emit, bg_emit) =
            make_kernel("emit_pairs", "instanced_select_emit", &[0, 1, 2, 5, 8, 9]);
        let (pipeline_pairs, bg_pairs) = make_kernel(
            "score_pairs",
            "instanced_select_k2",
            &[0, 1, 3, 4, 5, 7, 9],
        );
        // The M3.2 resident-walk kernels. No layout binds an indirect buffer
        // its own dispatch consumes (walk_args only in scan, k2_args only in
        // scan — the house usage rule).
        let (pipeline_build_far, bg_build_far) = make_kernel(
            "build_far_units",
            "instanced_select_build_far",
            &[0, 1, 2, 6, 11, 12, 14, 16],
        );
        let (pipeline_build_near, bg_build_near) = make_kernel(
            "build_near_units",
            "instanced_select_build_near",
            &[1, 5, 7, 8, 11, 12, 14, 16],
        );
        let (pipeline_spread, bg_spread) = make_kernel(
            "walk_bucket_spread",
            "instanced_select_spread",
            &[0, 2, 12, 14, 16],
        );
        let (pipeline_pick, bg_pick) =
            make_kernel("walk_bucket_pick", "instanced_select_pick", &[0, 14]);
        let (pipeline_walk_finalize, bg_walk_finalize) =
            make_kernel("walk_finalize", "instanced_select_walk_finalize", &[0, 14]);
        let (pipeline_final_counts, bg_final_counts) = make_kernel(
            "final_counts",
            "instanced_select_final_counts",
            &[0, 2, 12, 13, 14, 16],
        );
        let (pipeline_scan_blocks, bg_scan_blocks) =
            make_kernel("scan_blocks", "instanced_select_scan_blocks", &[13, 14]);
        let (pipeline_scan_totals, bg_scan_totals) =
            make_kernel("scan_totals", "instanced_select_scan_totals", &[13, 14]);
        let (pipeline_add_back, bg_add_back) =
            make_kernel("scan_add_back", "instanced_select_add_back", &[13, 14]);
        let (pipeline_emit_draws, bg_emit_draws) = make_kernel(
            "emit_draws",
            "instanced_select_emit_draws",
            &[0, 2, 12, 13, 14, 16, 17, 20],
        );
        let (pipeline_emit_finalize, bg_emit_finalize) = make_kernel(
            "emit_finalize",
            "instanced_select_emit_finalize",
            &[14, 18, 19, 20],
        );

        // ── Timestamp harness: slot 0 = kernel 1, slot 1 = kernel 2, slot 2 =
        //    the resident walk span (split begin/end writes). Degrades to
        //    inert (None readings) when TIMESTAMP_QUERY was not granted. ──
        let timers = GpuTimers::new(device, ctx.queue(), device.features(), 3);

        let cpu = InstancedSelector::new(library.clone());
        Ok(Self {
            ctx: ctx.clone(),
            pipeline_instances,
            pipeline_scan,
            pipeline_emit,
            pipeline_pairs,
            bg_instances,
            bg_scan,
            bg_emit,
            bg_pairs,
            pipeline_build_far,
            pipeline_build_near,
            pipeline_spread,
            pipeline_pick,
            pipeline_walk_finalize,
            pipeline_final_counts,
            pipeline_scan_blocks,
            pipeline_scan_totals,
            pipeline_add_back,
            pipeline_emit_draws,
            pipeline_emit_finalize,
            bg_build_far,
            bg_build_near,
            bg_spread,
            bg_pick,
            bg_walk_finalize,
            bg_final_counts,
            bg_scan_blocks,
            bg_scan_totals,
            bg_add_back,
            bg_emit_draws,
            bg_emit_finalize,
            instances_buf,
            planes_buf,
            params_buf,
            inst_out,
            pair_out,
            pair_meta_buf,
            k2_args_buf,
            fused_readback,
            walk_state_buf,
            walk_args_buf,
            draws_buf,
            expand_params_buf,
            expand_args_buf,
            stats_buf,
            stats_readback,
            library,
            max_instances,
            max_near_units,
            instances: Vec::new(),
            cpu,
            fallback_count: 0,
            work: Vec::new(),
            shed_scratch: Vec::new(),
            timers,
            last_breakdown: SelectGpuBreakdown::default(),
        })
    }

    /// Replace the placed-instance set: upload the 48 B PODs directly (they
    /// fit the device buffer iff `len ≤ max_instances` — an oversized set is
    /// reported by the next [`select`](Self::select)), keep a host copy, and
    /// keep the internal CPU fallback selector in sync.
    pub fn set_instances(&mut self, instances: &[AtomInstance]) {
        self.instances.clear();
        self.instances.extend_from_slice(instances);
        self.cpu.set_instances(instances);
        if !instances.is_empty() && instances.len() as u32 <= self.max_instances {
            self.ctx
                .queue()
                .write_buffer(&self.instances_buf, 0, bytemuck::cast_slice(instances));
        }
    }

    /// How many selects fell back to the internal CPU selector (near-pair
    /// demand above `max_near_units`).
    pub fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    /// The previous [`select`](Self::select)'s per-stage breakdown
    /// ([`SelectGpuBreakdown::default`] until the first select). Filled on
    /// EVERY select, including the CPU-fallback path (whose own pair/assembly/
    /// walk work runs inside the CPU twin and is honestly reported as 0.0,
    /// with `gpu_span_ms` covering the k1 span that preceded the fallback).
    pub fn last_breakdown(&self) -> SelectGpuBreakdown {
        self.last_breakdown
    }

    /// GPU-scored equivalent of
    /// [`InstancedSelector::select`](crate::atom_instances::InstancedSelector::select):
    /// ONE submit runs kernel 1 (classify+score instances), the serial pair
    /// scan, the pair emission and the indirect kernel 2 (score pairs), then
    /// ONE readback returns `PairMeta` + both score streams; the host
    /// assembles the identical `WorkUnit` stream and ends in the SAME
    /// [`budget_walk_and_emit`] — so `out` equals the CPU selector's output
    /// exactly. `select_us` covers the whole call. Never panics: limit
    /// violations return [`InstancedSelectGpuError::ExceedsDeviceLimits`],
    /// near-pair overflow (read from the scanned `pair_count`) falls back to
    /// the CPU twin.
    pub fn select(
        &mut self,
        camera: &RenderCamera,
        budget: usize,
        out: &mut InstancedSelection,
    ) -> Result<InstancedStats, InstancedSelectGpuError> {
        let start = std::time::Instant::now();
        let span_ms = |d: std::time::Duration| (d.as_secs_f64() * 1.0e3) as f32;
        let mut bd = SelectGpuBreakdown::default();
        let n = self.instances.len();
        if n as u64 > self.max_instances as u64 {
            return Err(InstancedSelectGpuError::ExceedsDeviceLimits {
                what: "instanced_select_instances (set exceeds max_instances)",
                requested: n as u64 * INSTANCE_BYTES,
                limit: self.max_instances as u64 * INSTANCE_BYTES,
            });
        }

        let library = self.library.clone();
        let lib = library.as_ref();
        self.work.clear();

        let mut instances_visible = 0usize;
        let mut instances_culled = 0usize;
        let mut instances_far = 0usize;

        if n > 0 {
            let queue = self.ctx.queue();
            let device = self.ctx.device();

            // Host-normalized planes + eye — the SAME helpers (now
            // `pub(crate)`) the proven atom_budget_gpu port uploads, so the
            // plane floats are bit-equal to the CPU `Frustum`'s.
            let planes: [GpuPlane; 6] = frustum_planes(camera.view_proj());
            queue.write_buffer(&self.planes_buf, 0, bytemuck::cast_slice(&planes));
            let eye = camera_eye(camera);
            let params = Params {
                instance_count: n as u32,
                max_near_units: self.max_near_units,
                asset_count: lib.asset_count() as u32,
                budget: budget.min(u32::MAX as usize) as u32,
                eye: [eye.x, eye.y, eye.z, 0.0],
            };
            queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            // ── The fused chain: k1 → scan → emit → k2(indirect) in ONE
            //    encoder — ONE submit, ONE readback. ──────────────────────────
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("instanced_select_fused_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("instanced_select_score_instances"),
                    timestamp_writes: self.timers.compute_writes(0),
                });
                pass.set_pipeline(&self.pipeline_instances);
                pass.set_bind_group(0, &self.bg_instances, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            {
                // ONE serial thread: the deterministic prefix scan over k1's
                // results — per-near-instance pair bases, the total demand
                // into PairMeta, and k2's indirect args (ZERO workgroups when
                // the demand overflows the max_near_units band).
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("instanced_select_scan_pairs"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_scan);
                pass.set_bind_group(0, &self.bg_scan, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("instanced_select_emit_pairs"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_emit);
                pass.set_bind_group(0, &self.bg_emit, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            {
                // Indirect dispatch off the scan's args — the host does not
                // know the pair count pre-map. `bg_pairs` has no k2_args
                // binding: wgpu forbids the indirect buffer doubling as
                // writable storage in the dispatch's own usage scope.
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("instanced_select_score_pairs"),
                    timestamp_writes: self.timers.compute_writes(1),
                });
                pass.set_pipeline(&self.pipeline_pairs);
                pass.set_bind_group(0, &self.bg_pairs, &[]);
                pass.dispatch_workgroups_indirect(&self.k2_args_buf, 0);
            }
            // Resolve both timestamp slots (k2 always runs as a pass — slot 1
            // is written even at zero workgroups), then the fused copies:
            // PairMeta (16 B) + instance scores (n × 16 B) + the FULL
            // pair-score region (max_near_units × 16 B — pair_count is
            // unknown host-side before the map; the ≤4 MB device→readback
            // copy is the price of one sync).
            self.timers.resolve(&mut encoder);
            let inst_copy = n as u64 * SCORE_BYTES;
            let pair_copy = (self.max_near_units as u64 * SCORE_BYTES).max(SCORE_BYTES);
            encoder.copy_buffer_to_buffer(
                &self.pair_meta_buf,
                0,
                &self.fused_readback,
                0,
                PAIR_META_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &self.inst_out,
                0,
                &self.fused_readback,
                PAIR_META_BYTES,
                inst_copy,
            );
            encoder.copy_buffer_to_buffer(
                &self.pair_out,
                0,
                &self.fused_readback,
                PAIR_META_BYTES + inst_copy,
                pair_copy,
            );
            let t_gpu = std::time::Instant::now();
            queue.submit(Some(encoder.finish()));
            // THE one sync: map the fused readback and parse IN PLACE (no
            // whole-buffer `to_vec` — only the bytes actually needed are
            // copied into typed storage; `Vec<u8>` would carry no POD
            // alignment guarantee anyway).
            let map_bytes = PAIR_META_BYTES + inst_copy + pair_copy;
            let slice = self.fused_readback.slice(..map_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            device.poll(wgpu::Maintain::Wait);
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(InstancedSelectGpuError::Readback(e.to_string())),
                Err(e) => return Err(InstancedSelectGpuError::Readback(e.to_string())),
            }
            bd.gpu_span_ms = span_ms(t_gpu.elapsed());
            bd.syncs = 1;
            // Timestamp readings AFTER the map: the GPU work is already
            // drained, so resolve_ms's poll is effectively free. None on
            // devices without TIMESTAMP_QUERY.
            bd.k1_gpu_ms = self.timers.resolve_ms(device, 0);
            bd.k2_gpu_ms = self.timers.resolve_ms(device, 1);

            let data = slice.get_mapped_range();
            let mut meta = PairMeta::zeroed();
            bytemuck::bytes_of_mut(&mut meta).copy_from_slice(&data[..PAIR_META_BYTES as usize]);

            // ── Pair demand, scanned GPU-side: overflow → the bit-identical
            //    CPU fallback. Kernel 2 already dispatched ZERO workgroups, so
            //    the pair-score region is stale and never read. ───────────────
            let pair_count = meta.pair_count as usize;
            if pair_count > self.max_near_units as usize {
                drop(data);
                self.fused_readback.unmap();
                self.fallback_count += 1;
                // Honest fallback breakdown: the one fused readback (and its
                // blocking poll) preceded the fallback decision; the
                // fallback's own pair/assembly/walk work happens inside the
                // CPU twin, so those spans are 0.0 here.
                self.last_breakdown = bd;
                let mut stats = self.cpu.select(camera, budget, out);
                stats.select_us = start.elapsed().as_micros() as u64;
                return Ok(stats);
            }
            // Typed copies out of the mapped range (assemble span: the
            // kernel-readback → walk bridge starts here). The GPU emission
            // order (ascending instance, ascending cluster via the
            // serial-scan bases) equals the deleted host pair loop's order by
            // construction — the 9-combo equality suite is the proof.
            let t_assemble = std::time::Instant::now();
            let mut inst_scores = vec![GpuInstScore::zeroed(); n];
            bytemuck::cast_slice_mut::<GpuInstScore, u8>(&mut inst_scores)
                .copy_from_slice(&data[PAIR_META_BYTES as usize..(PAIR_META_BYTES + inst_copy) as usize]);
            let mut pair_scores = vec![GpuPairScore::zeroed(); pair_count];
            let pair_off = (PAIR_META_BYTES + inst_copy) as usize;
            bytemuck::cast_slice_mut::<GpuPairScore, u8>(&mut pair_scores)
                .copy_from_slice(&data[pair_off..pair_off + pair_count * SCORE_BYTES as usize]);
            drop(data);
            self.fused_readback.unmap();

            // ── Assemble the CPU instance pass's exact WorkUnit stream:
            //    ascending instance order, far units interleaved with the
            //    frustum-passing near cluster units. ───────────────────────────
            let mut cursor = 0usize;
            for (i, s) in inst_scores.iter().enumerate() {
                match s.kind {
                    0 => instances_culled += 1,
                    1 => {
                        instances_visible += 1;
                        instances_far += 1;
                        let asset = self.instances[i].asset;
                        let level = s.level as u8;
                        let count = lib.unit_range(asset, DrawUnit::Imposter { level }).1 as usize;
                        self.work.push(WorkUnit {
                            instance: i as u32,
                            asset,
                            unit_key: u32::MAX,
                            is_imposter: true,
                            distance: s.distance,
                            score: s.score,
                            distance_lod: level,
                            lod: level,
                            count,
                        });
                    }
                    _ => {
                        instances_visible += 1;
                        let asset = self.instances[i].asset;
                        for (ci, cm) in lib.cluster_metas(asset).iter().enumerate() {
                            let p = &pair_scores[cursor];
                            cursor += 1;
                            if p.passes == 0 {
                                continue;
                            }
                            let lod = p.distance_lod as u8;
                            let count = lib
                                .unit_range(asset, DrawUnit::Cluster { id: ci as u32, lod })
                                .1 as usize;
                            self.work.push(WorkUnit {
                                instance: i as u32,
                                asset,
                                unit_key: cm.id,
                                is_imposter: false,
                                distance: p.distance,
                                score: p.score,
                                distance_lod: lod,
                                lod,
                                count,
                            });
                        }
                    }
                }
            }
            bd.assemble_ms = span_ms(t_assemble.elapsed());
        }

        // ── THE shared walk — the same function `InstancedSelector::select`
        //    ends with (one walk, two scorers). ────────────────────────────────
        let t_walk = std::time::Instant::now();
        let (lod_histogram, selected) =
            budget_walk_and_emit(lib, &mut self.work, &mut self.shed_scratch, budget, out);
        bd.walk_ms = span_ms(t_walk.elapsed());
        self.last_breakdown = bd;

        Ok(InstancedStats {
            budget,
            selected,
            instances_visible,
            instances_culled,
            instances_far,
            lod_histogram,
            select_us: start.elapsed().as_micros() as u64,
        })
    }

    /// The fully GPU-resident select (M3.2 — the M3.1-named escalation): ONE
    /// submit runs k1 → scan_pairs → emit_pairs → k2(indirect) →
    /// build_far/near_units → 8×(spread, pick) demote → 8×(spread, pick) shed
    /// → walk_finalize → final_counts → scan ×3 → emit_draws → emit_finalize,
    /// then ONE 64 B map of the stats block — the host never sees scores,
    /// WorkUnits, or draws. On [`SelectPath::GpuResident`] the draws / params
    /// / indirect args are GPU-resident (see [`Self::draws_buf`],
    /// [`Self::expand_params_buf`], [`Self::expand_args_buf`] — consumed by
    /// `ExpandDrawsPass::encode_indirect`) and `fallback_out` is cleared. On
    /// [`SelectPath::CpuFallback`] (scanned `pair_count > max_near_units`,
    /// the GPU tail zeroed itself) `fallback_out` holds the CPU twin's draws
    /// bit-identically and [`Self::fallback_count`] increments. `select_us`
    /// covers the whole call. Never panics: limit violations return
    /// [`InstancedSelectGpuError::ExceedsDeviceLimits`].
    pub fn select_resident(
        &mut self,
        camera: &RenderCamera,
        budget: usize,
        fallback_out: &mut InstancedSelection,
    ) -> Result<(InstancedStats, SelectPath), InstancedSelectGpuError> {
        let start = std::time::Instant::now();
        let span_ms = |d: std::time::Duration| (d.as_secs_f64() * 1.0e3) as f32;
        let mut bd = SelectGpuBreakdown::default();
        let n = self.instances.len();
        if n as u64 > self.max_instances as u64 {
            return Err(InstancedSelectGpuError::ExceedsDeviceLimits {
                what: "instanced_select_instances (set exceeds max_instances)",
                requested: n as u64 * INSTANCE_BYTES,
                limit: self.max_instances as u64 * INSTANCE_BYTES,
            });
        }
        *fallback_out = InstancedSelection::new();

        let queue = self.ctx.queue();
        let device = self.ctx.device();

        if n == 0 {
            // No GPU work — but zero the resident outputs so a following
            // `encode_indirect` honestly expands nothing (no stale frame).
            queue.write_buffer(&self.expand_params_buf, 0, &[0u8; 16]);
            queue.write_buffer(&self.expand_args_buf, 0, bytemuck::cast_slice(&[0u32, 1, 1]));
            self.last_breakdown = bd;
            return Ok((
                InstancedStats {
                    budget,
                    selected: 0,
                    instances_visible: 0,
                    instances_culled: 0,
                    instances_far: 0,
                    lod_histogram: [0; 6],
                    select_us: start.elapsed().as_micros() as u64,
                },
                SelectPath::GpuResident,
            ));
        }

        // Host-normalized planes + eye — bit-equal to the CPU `Frustum`'s.
        let planes: [GpuPlane; 6] = frustum_planes(camera.view_proj());
        queue.write_buffer(&self.planes_buf, 0, bytemuck::cast_slice(&planes));
        let eye = camera_eye(camera);
        let params = Params {
            instance_count: n as u32,
            max_near_units: self.max_near_units,
            asset_count: self.library.asset_count() as u32,
            budget: budget.min(u32::MAX as usize) as u32,
            eye: [eye.x, eye.y, eye.z, 0.0],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // ── THE one encoder: the full pinned chain, ending in emit_finalize
        //    + the 64 B stats copy. ────────────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("instanced_select_resident_encoder"),
        });
        // Clear the walk's mutable state (buckets + scalars; the constant
        // blocks_base tail survives) and the stats block. Both are tiny —
        // this is NOT a per-frame tile clear.
        encoder.clear_buffer(&self.walk_state_buf, 0, Some(WALK_STATE_CLEAR_BYTES));
        encoder.clear_buffer(&self.stats_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_score_instances"),
                timestamp_writes: self.timers.compute_writes(0),
            });
            pass.set_pipeline(&self.pipeline_instances);
            pass.set_bind_group(0, &self.bg_instances, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_scan_pairs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scan);
            pass.set_bind_group(0, &self.bg_scan, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_emit_pairs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_emit);
            pass.set_bind_group(0, &self.bg_emit, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_score_pairs"),
                timestamp_writes: self.timers.compute_writes(1),
            });
            pass.set_pipeline(&self.pipeline_pairs);
            pass.set_bind_group(0, &self.bg_pairs, &[]);
            pass.dispatch_workgroups_indirect(&self.k2_args_buf, 0);
        }
        {
            // First walk pass — opens the slot-2 multi-pass span.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_build_far_units"),
                timestamp_writes: self.timers.compute_writes_begin(2),
            });
            pass.set_pipeline(&self.pipeline_build_far);
            pass.set_bind_group(0, &self.bg_build_far, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_build_near_units"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build_near);
            pass.set_bind_group(0, &self.bg_build_near, &[]);
            pass.dispatch_workgroups_indirect(&self.k2_args_buf, 0);
        }
        // 8 demote + 8 shed refinement passes — the shed ones no-op via the
        // walk_state regime checks when the demote boundary closed the gap.
        for _ in 0..16 {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("instanced_select_walk_spread"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_spread);
                pass.set_bind_group(0, &self.bg_spread, &[]);
                pass.dispatch_workgroups_indirect(&self.walk_args_buf, 0);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("instanced_select_walk_pick"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pick);
                pass.set_bind_group(0, &self.bg_pick, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_walk_finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_walk_finalize);
            pass.set_bind_group(0, &self.bg_walk_finalize, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_final_counts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_final_counts);
            pass.set_bind_group(0, &self.bg_final_counts, &[]);
            pass.dispatch_workgroups_indirect(&self.walk_args_buf, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_scan_blocks"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scan_blocks);
            pass.set_bind_group(0, &self.bg_scan_blocks, &[]);
            pass.dispatch_workgroups_indirect(&self.walk_args_buf, 12);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_scan_totals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scan_totals);
            pass.set_bind_group(0, &self.bg_scan_totals, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_scan_add_back"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_add_back);
            pass.set_bind_group(0, &self.bg_add_back, &[]);
            pass.dispatch_workgroups_indirect(&self.walk_args_buf, 12);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_emit_draws"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_emit_draws);
            pass.set_bind_group(0, &self.bg_emit_draws, &[]);
            pass.dispatch_workgroups_indirect(&self.walk_args_buf, 0);
        }
        {
            // Last walk pass — closes the slot-2 span.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("instanced_select_emit_finalize"),
                timestamp_writes: self.timers.compute_writes_end(2),
            });
            pass.set_pipeline(&self.pipeline_emit_finalize);
            pass.set_bind_group(0, &self.bg_emit_finalize, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.timers.resolve(&mut encoder);
        encoder.copy_buffer_to_buffer(&self.stats_buf, 0, &self.stats_readback, 0, STATS_BYTES);

        let t_gpu = std::time::Instant::now();
        queue.submit(Some(encoder.finish()));
        // THE one sync: the 64 B stats map.
        let slice = self.stats_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(InstancedSelectGpuError::Readback(e.to_string())),
            Err(e) => return Err(InstancedSelectGpuError::Readback(e.to_string())),
        }
        bd.gpu_span_ms = span_ms(t_gpu.elapsed());
        bd.syncs = 1;
        bd.k1_gpu_ms = self.timers.resolve_ms(device, 0);
        bd.k2_gpu_ms = self.timers.resolve_ms(device, 1);
        bd.walk_gpu_ms = self.timers.resolve_ms(device, 2);

        let mut gs = GpuWalkStats::zeroed();
        {
            let data = slice.get_mapped_range();
            bytemuck::bytes_of_mut(&mut gs).copy_from_slice(&data[..STATS_BYTES as usize]);
        }
        self.stats_readback.unmap();

        if gs.pair_count as u64 > self.max_near_units as u64 {
            // The GPU tail already no-opped (walk args zeroed, ExpandParams
            // 0/0) — take the bit-identical CPU twin.
            self.fallback_count += 1;
            self.last_breakdown = bd;
            let mut stats = self.cpu.select(camera, budget, fallback_out);
            stats.select_us = start.elapsed().as_micros() as u64;
            return Ok((stats, SelectPath::CpuFallback));
        }

        self.last_breakdown = bd;
        Ok((
            InstancedStats {
                budget,
                selected: gs.selected as usize,
                instances_visible: gs.visible as usize,
                instances_culled: gs.culled as usize,
                instances_far: gs.far_count as usize,
                lod_histogram: gs.lod_histogram.map(|v| v as usize),
                select_us: start.elapsed().as_micros() as u64,
            },
            SelectPath::GpuResident,
        ))
    }

    // The expand-seam accessors below feed `ExpandDrawsPass::encode_indirect`
    // (M3.2 Task 2) and the in-module `resident_walk_` tests; `dead_code`
    // allowed until that seam lands (the atom_instances house pattern).

    /// The device-emitted `ExpandDraw` buffer (`units_cap × 32 B`, STORAGE) —
    /// bound at binding 3 by `ExpandDrawsPass::encode_indirect`.
    #[allow(dead_code)]
    pub(crate) fn draws_buf(&self) -> &wgpu::Buffer {
        &self.draws_buf
    }

    /// The GPU-written `ExpandParams` (16 B, UNIFORM|STORAGE) — binding 0 of
    /// the indirect expand.
    #[allow(dead_code)]
    pub(crate) fn expand_params_buf(&self) -> &wgpu::Buffer {
        &self.expand_params_buf
    }

    /// The expand indirect dispatch args (12 B, STORAGE|INDIRECT) — consumed
    /// by `dispatch_workgroups_indirect`.
    #[allow(dead_code)]
    pub(crate) fn expand_args_buf(&self) -> &wgpu::Buffer {
        &self.expand_args_buf
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// In-module tests — M3 Task 3: the GPU two-kernel scorer feeding the SHARED
// `budget_walk_and_emit` must reproduce the CPU `InstancedSelector` draws
// EXACTLY (vec equality) across 9 (camera, budget) combos, beat it at city
// scale, fall back bit-identically past the near-unit cap, and be
// deterministic. The equality test also drives the full seam:
// select → ExpandDrawsPass::encode → set_active_splat_count → render.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_instances::{
        AssetAtomLibrary, AtomInstance, InstancedSelection, InstancedSelector,
    };
    use crate::gpu::GpuContext;
    use crate::gpu::expand_draws::ExpandDrawsPass;
    use crate::gpu::tiled_splat_renderer::TiledSplatRenderer;
    use crate::spectral::RenderCamera;
    use glam::{Mat4, Quat, Vec3};
    use half::f16;
    use std::f32::consts::FRAC_PI_4;
    use std::sync::Arc;
    use vox_core::spectral::Illuminant;
    use vox_core::types::GaussianSplat;

    const W: u32 = 320;
    const H: u32 = 180;

    /// House `try_gpu` pattern (shape from `expand_draws.rs::tests`):
    /// hardware adapter or printed skip + green exit.
    fn try_gpu_context(label: &str) -> Option<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("[{label}] no adapter — skipping GPU test");
                return None;
            }
        };
        let info = adapter.get_info();
        if crate::gpu::adapter::ensure_hardware(&info).is_err() {
            eprintln!(
                "[{label}] software adapter ({}) — skipping GPU test",
                info.name
            );
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("instanced_select_gpu_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(GpuContext::from_parts(&device, &queue, &info))
    }

    /// splitmix64-style scramble (house pattern, same as `atom_instances`).
    fn hash_u64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn hash01(a: u64, b: u64) -> f32 {
        let h = hash_u64(a ^ hash_u64(b.wrapping_mul(0x100_0000_01B3)));
        (h >> 40) as f32 / (1u64 << 24) as f32
    }

    fn spectral_level(level: f32) -> [u16; 16] {
        std::array::from_fn(|i| {
            let v = if i < 8 { level } else { level * 0.55 };
            f16::from_f32(v).to_bits()
        })
    }

    /// Deterministic hash-jittered blob asset — the M1 city generator copied
    /// VERBATIM from `atom_instances.rs::tests::synthetic_asset`.
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

    /// The M1 city geometry: 4 × 5,000-atom synthetic assets, 10,000 instances
    /// on the 100×100 × 4 m grid with quarter-turn yaws (copied from
    /// `instanced_selector_budget_bound_at_city_scale`).
    fn city_scene() -> (Arc<AssetAtomLibrary>, Vec<AtomInstance>) {
        let assets: Vec<Vec<GaussianSplat>> = (0..4)
            .map(|a| synthetic_asset(0xC17 + a as u64, 5000, 4.0, 12.0))
            .collect();
        let lib = Arc::new(AssetAtomLibrary::build(&assets, 128));
        let instances: Vec<AtomInstance> = (0..10_000u32)
            .map(|i| {
                let gx = (i % 100) as f32 - 49.5;
                let gz = (i / 100) as f32 - 49.5;
                let q = Quat::from_rotation_y((i % 4) as f32 * std::f32::consts::FRAC_PI_2);
                AtomInstance::new(i % 4, i, [gx * 4.0, 0.0, gz * 4.0], [q.x, q.y, q.z, q.w])
            })
            .collect();
        (lib, instances)
    }

    /// The plan's 3 cameras: 100 m overhead (fovy 2.5), oblique 45° aerial,
    /// street-level 8 m altitude (the near-pair-heavy view).
    fn city_cameras() -> [RenderCamera; 3] {
        [
            RenderCamera {
                view: Mat4::look_at_rh(Vec3::new(0.0, 100.0, 0.0), Vec3::ZERO, Vec3::Z),
                proj: Mat4::perspective_rh(2.5, 1.0, 0.1, 2000.0),
            },
            RenderCamera {
                view: Mat4::look_at_rh(Vec3::new(80.0, 80.0, 80.0), Vec3::ZERO, Vec3::Y),
                proj: Mat4::perspective_rh(1.2, 1.0, 0.1, 2000.0),
            },
            RenderCamera {
                view: Mat4::look_at_rh(
                    Vec3::new(0.0, 8.0, -190.0),
                    Vec3::new(0.0, 6.0, 0.0),
                    Vec3::Y,
                ),
                proj: Mat4::perspective_rh(FRAC_PI_4, 1.0, 0.1, 2000.0),
            },
        ]
    }

    /// THE CONTRACT: across 3 cameras × 3 budgets (incl. shed pressure at
    /// 30k), the GPU draws must EXACTLY equal the CPU selector's (vec
    /// equality, f32 fields bit-meaningful) with matching stats — then one
    /// combo drives the full expand+render seam on the SAME shared context.
    #[test]
    fn instanced_gpu_select_equals_cpu_selector() {
        let Some(ctx) = try_gpu_context("instanced_gpu") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut cpu = InstancedSelector::new(lib.clone());
        cpu.set_instances(&instances);
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);

        let cameras = city_cameras();
        let budgets = [1_000_000usize, 250_000, 30_000];
        let mut total_draws = 0usize;
        for (ci, cam) in cameras.iter().enumerate() {
            for &budget in &budgets {
                let mut cpu_out = InstancedSelection::new();
                let cpu_stats = cpu.select(cam, budget, &mut cpu_out);
                let mut gpu_out = InstancedSelection::new();
                let gpu_stats = gpu.select(cam, budget, &mut gpu_out).expect("gpu select");

                assert!(
                    !cpu_out.draws().is_empty(),
                    "camera {ci} budget {budget}: combo must be non-trivial"
                );
                assert_eq!(
                    gpu_out.draws(),
                    cpu_out.draws(),
                    "camera {ci} budget {budget}: GPU draws must EXACTLY equal the CPU selector"
                );
                assert_eq!(gpu_stats.selected, cpu_stats.selected, "camera {ci} budget {budget}");
                assert_eq!(
                    gpu_stats.instances_visible, cpu_stats.instances_visible,
                    "camera {ci} budget {budget}: instances_visible"
                );
                assert_eq!(
                    gpu_stats.instances_culled, cpu_stats.instances_culled,
                    "camera {ci} budget {budget}: instances_culled"
                );
                assert_eq!(
                    gpu_stats.instances_far, cpu_stats.instances_far,
                    "camera {ci} budget {budget}: instances_far"
                );
                assert_eq!(
                    gpu_stats.lod_histogram, cpu_stats.lod_histogram,
                    "camera {ci} budget {budget}: lod_histogram"
                );
                total_draws += gpu_out.draws().len();
            }
        }
        assert_eq!(
            gpu.fallback_count(),
            0,
            "equality must be proven on the GPU path, not the CPU fallback"
        );
        assert!(total_draws > 0);
        println!(
            "[instanced_gpu] draws bit-equal across 9 (camera,budget) combos (total_draws={total_draws})"
        );

        // Full seam on the SAME shared context: GPU select → expand → render.
        const SEAM_BUDGET: u32 = 250_000;
        let mut renderer = TiledSplatRenderer::new_with_capacity(ctx.clone(), SEAM_BUDGET, W, H)
            .expect("250k capacity fits default device limits");
        let mut pass =
            ExpandDrawsPass::new_with_context(&ctx, &lib, instances.len() as u32, SEAM_BUDGET)
                .expect("expand pass fits default device limits");
        let mut sel = InstancedSelection::new();
        gpu.select(&cameras[0], SEAM_BUDGET as usize, &mut sel)
            .expect("seam select");
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("instanced_gpu_seam_encode"),
            });
        let n = pass.encode(
            &mut enc,
            &sel,
            &instances,
            renderer.splat_buf(),
            renderer.transform_buf(),
        );
        ctx.queue().submit(Some(enc.finish()));
        assert_eq!(n as usize, sel.atom_count(), "expand must take the whole selection");
        renderer.set_active_splat_count(n);
        let frame = renderer.render(&cameras[0]).expect("render after GPU select");
        let (_, non_black) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());
        assert!(
            non_black > 0,
            "GPU-selected city must rasterize ({non_black} non-black pixels)"
        );
        println!("[instanced_gpu] seam: expanded={n} non_black={non_black}");
    }

    /// ≥10 alternating selects on the 10k-instance city: the GPU median must
    /// beat the CPU median (debug-build numbers, labelled — the release gate
    /// numbers come from the Task-4 scale_trial harness).
    #[test]
    fn instanced_gpu_faster_at_city_scale() {
        let Some(ctx) = try_gpu_context("instanced_gpu") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut cpu = InstancedSelector::new(lib.clone());
        cpu.set_instances(&instances);
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[0];
        const BUDGET: usize = 1_000_000;
        let mut out = InstancedSelection::new();
        // One unmeasured warmup each (pipeline compile / page-in).
        cpu.select(cam, BUDGET, &mut out);
        gpu.select(cam, BUDGET, &mut out).expect("gpu warmup select");

        let mut cpu_us: Vec<u64> = Vec::new();
        let mut gpu_us: Vec<u64> = Vec::new();
        for _ in 0..10 {
            let s = cpu.select(cam, BUDGET, &mut out);
            cpu_us.push(s.select_us);
            let g = gpu.select(cam, BUDGET, &mut out).expect("gpu select");
            gpu_us.push(g.select_us);
        }
        cpu_us.sort_unstable();
        gpu_us.sort_unstable();
        let c = cpu_us[cpu_us.len() / 2] as f64 / 1000.0;
        let g = gpu_us[gpu_us.len() / 2] as f64 / 1000.0;
        assert_eq!(gpu.fallback_count(), 0, "speed must be measured on the GPU path");
        println!(
            "[instanced_gpu] city-scale select: cpu_ms={c:.2} gpu_ms={g:.2} (medians of 10, debug build)"
        );
        assert!(
            g < c,
            "GPU select ({g:.2} ms) must beat CPU select ({c:.2} ms) at city scale"
        );
    }

    /// `max_near_units = 64` + the street camera (hundreds of near instances ×
    /// dozens of clusters) overflows the pair budget: the selector must fall
    /// back to its internal CPU twin, bit-identically, and count it.
    #[test]
    fn instanced_gpu_near_overflow_falls_back() {
        let Some(ctx) = try_gpu_context("instanced_gpu") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut cpu = InstancedSelector::new(lib.clone());
        cpu.set_instances(&instances);
        let mut gpu =
            InstancedSelectGpu::new_with_context(&ctx, lib.clone(), instances.len() as u32, 64)
                .expect("64-unit near cap fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[2]; // street level — near-pair heavy
        const BUDGET: usize = 250_000;
        let mut cpu_out = InstancedSelection::new();
        let cpu_stats = cpu.select(cam, BUDGET, &mut cpu_out);
        let mut gpu_out = InstancedSelection::new();
        let gpu_stats = gpu
            .select(cam, BUDGET, &mut gpu_out)
            .expect("near overflow must fall back, not error");

        assert_eq!(
            gpu.fallback_count(),
            1,
            "street camera must overflow the 64-unit near cap exactly once"
        );
        assert!(!cpu_out.draws().is_empty(), "street view must emit draws");
        assert_eq!(
            gpu_out.draws(),
            cpu_out.draws(),
            "fallback draws must EXACTLY equal the CPU selector"
        );
        assert_eq!(gpu_stats.selected, cpu_stats.selected);
        assert_eq!(gpu_stats.lod_histogram, cpu_stats.lod_histogram);
        println!("[instanced_gpu] fallback: pairs>64 -> cpu path, draws equal, fallback_count=1");
    }

    /// Two GPU selects of the same scene/camera/budget produce exactly equal
    /// draw vectors (run-to-run bit stability of the scoring kernels + the
    /// deterministic shared walk).
    #[test]
    fn instanced_gpu_deterministic() {
        let Some(ctx) = try_gpu_context("instanced_gpu") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[0];
        let mut a = InstancedSelection::new();
        let mut b = InstancedSelection::new();
        gpu.select(cam, 250_000, &mut a).expect("select a");
        gpu.select(cam, 250_000, &mut b).expect("select b");
        assert!(!a.draws().is_empty(), "scene must emit draws");
        assert_eq!(a.draws(), b.draws(), "draw vectors must be exactly equal");
        assert_eq!(gpu.fallback_count(), 0);
        println!(
            "[instanced_gpu] deterministic: {} draws identical x2",
            a.draws().len()
        );
    }

    /// Copy `bytes` out of `buf` at `offset` and return them (blocking, proof-
    /// only readback — the house helper from `expand_draws.rs::tests`).
    fn read_back(ctx: &GpuContext, buf: &wgpu::Buffer, offset: u64, bytes: u64) -> Vec<u8> {
        let device = ctx.device();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident_walk_test_readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident_walk_test_readback_copy"),
        });
        enc.copy_buffer_to_buffer(buf, offset, &readback, 0, bytes);
        ctx.queue().submit(Some(enc.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        assert!(matches!(rx.recv(), Ok(Ok(()))), "readback map failed");
        let data = slice.get_mapped_range().to_vec();
        readback.unmap();
        data
    }

    /// THE M3.2 CONTRACT: across the same 9 (camera, budget) combos — incl.
    /// the 30k street combo that exercises shed — the GPU-resident walk's
    /// device-emitted `ExpandDraw` bytes must EXACTLY equal
    /// `ExpandDrawsPass::lower` over the CPU selector's draws (the oracle),
    /// with the 64 B stats block matching the CPU stats.
    #[test]
    fn resident_walk_draws_equal_cpu_oracle() {
        let Some(ctx) = try_gpu_context("resident_walk") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut cpu = InstancedSelector::new(lib.clone());
        cpu.set_instances(&instances);
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);
        // The CPU draw-byte oracle: `lower` is the EXACT loop `encode` uploads.
        let mut lower_pass =
            ExpandDrawsPass::new_with_context(&ctx, &lib, instances.len() as u32, 1_000_000)
                .expect("expand pass fits default device limits");

        let cameras = city_cameras();
        let budgets = [1_000_000usize, 250_000, 30_000];
        let mut total_draws = 0usize;
        let mut total_bytes = 0usize;
        let mut selected_sum = 0usize;
        for (ci, cam) in cameras.iter().enumerate() {
            for &budget in &budgets {
                let mut cpu_out = InstancedSelection::new();
                let cpu_stats = cpu.select(cam, budget, &mut cpu_out);
                assert!(
                    !cpu_out.draws().is_empty(),
                    "camera {ci} budget {budget}: combo must be non-trivial"
                );
                let (expected, cpu_total) = {
                    let (draws, total) = lower_pass.lower(&cpu_out, instances.len());
                    (bytemuck::cast_slice::<_, u8>(draws).to_vec(), total)
                };
                assert_eq!(
                    cpu_total as usize,
                    cpu_out.atom_count(),
                    "camera {ci} budget {budget}: lower must take the whole selection"
                );

                let mut fb = InstancedSelection::new();
                let (stats, path) = gpu
                    .select_resident(cam, budget, &mut fb)
                    .expect("resident select");
                assert_eq!(
                    path,
                    SelectPath::GpuResident,
                    "camera {ci} budget {budget}: must stay on the GPU path"
                );
                assert!(
                    fb.draws().is_empty(),
                    "camera {ci} budget {budget}: fallback_out must be cleared on the GPU path"
                );

                // The GPU-written ExpandParams: (total, draw_count) head.
                let pbytes = read_back(&ctx, gpu.expand_params_buf(), 0, 16);
                let p: &[u32] = bytemuck::cast_slice(&pbytes);
                assert_eq!(
                    p[0] as usize, cpu_stats.selected,
                    "camera {ci} budget {budget}: ExpandParams.total"
                );
                let cpu_draw_count = expected.len() / 32;
                assert_eq!(
                    p[1] as usize, cpu_draw_count,
                    "camera {ci} budget {budget}: ExpandParams.draw_count"
                );

                // THE REAL draws_buf bytes — the buffer Task 2's indirect
                // expand binds unmodified.
                let got = read_back(&ctx, gpu.draws_buf(), 0, expected.len() as u64);
                assert!(
                    got == expected,
                    "camera {ci} budget {budget}: GPU draw bytes diverge from lower(cpu draws)"
                );

                assert_eq!(
                    stats.selected, cpu_stats.selected,
                    "camera {ci} budget {budget}: selected"
                );
                assert_eq!(
                    stats.instances_visible, cpu_stats.instances_visible,
                    "camera {ci} budget {budget}: instances_visible"
                );
                assert_eq!(
                    stats.instances_culled, cpu_stats.instances_culled,
                    "camera {ci} budget {budget}: instances_culled"
                );
                assert_eq!(
                    stats.instances_far, cpu_stats.instances_far,
                    "camera {ci} budget {budget}: instances_far"
                );
                assert_eq!(
                    stats.lod_histogram, cpu_stats.lod_histogram,
                    "camera {ci} budget {budget}: lod_histogram"
                );

                total_draws += cpu_draw_count;
                total_bytes += expected.len();
                selected_sum += stats.selected;
            }
        }
        assert_eq!(
            gpu.fallback_count(),
            0,
            "equality must be proven on the GPU-resident path, not the fallback"
        );
        assert!(total_draws > 0 && total_bytes > 0);
        println!(
            "[resident_walk] draws bit-equal across 9 (camera,budget) combos (total_draws={total_draws}, bytes={total_bytes})"
        );
        println!("[resident_walk] stats equal across 9 combos (selected sum={selected_sum})");
    }

    /// Two `select_resident` calls of the same scene/camera/budget must leave
    /// byte-identical draw buffers (the u32-atomic bucket sums + serial picks
    /// + fixed-op-order f32 scoring are the determinism mechanisms).
    #[test]
    fn resident_walk_deterministic() {
        let Some(ctx) = try_gpu_context("resident_walk") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[0];
        let mut fb = InstancedSelection::new();

        gpu.select_resident(cam, 250_000, &mut fb).expect("select a");
        let pa = read_back(&ctx, gpu.expand_params_buf(), 0, 16);
        let draws_a = bytemuck::cast_slice::<u8, u32>(&pa)[1] as u64;
        let a = read_back(&ctx, gpu.draws_buf(), 0, draws_a * 32);

        gpu.select_resident(cam, 250_000, &mut fb).expect("select b");
        let pb = read_back(&ctx, gpu.expand_params_buf(), 0, 16);
        let draws_b = bytemuck::cast_slice::<u8, u32>(&pb)[1] as u64;
        let b = read_back(&ctx, gpu.draws_buf(), 0, draws_b * 32);

        assert!(draws_a > 0, "scene must emit draws");
        assert_eq!(draws_a, draws_b, "draw counts must match across runs");
        assert!(a == b, "draw buffers must be byte-identical across two selects");
        assert_eq!(gpu.fallback_count(), 0);
        println!(
            "[resident_walk] deterministic: {} draw bytes identical x2",
            a.len()
        );
    }

    /// `max_near_units = 64` + the street camera overflows the pair band: the
    /// scanned overflow zeroes the walk/emit/expand tail GPU-side (ExpandParams
    /// reads back 0) and the host takes the bit-identical CPU twin.
    #[test]
    fn resident_walk_fallback_band() {
        let Some(ctx) = try_gpu_context("resident_walk") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut cpu = InstancedSelector::new(lib.clone());
        cpu.set_instances(&instances);
        let mut gpu =
            InstancedSelectGpu::new_with_context(&ctx, lib.clone(), instances.len() as u32, 64)
                .expect("64-unit near cap fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[2]; // street level — near-pair heavy
        const BUDGET: usize = 250_000;
        let mut cpu_out = InstancedSelection::new();
        let cpu_stats = cpu.select(cam, BUDGET, &mut cpu_out);
        let mut fb = InstancedSelection::new();
        let (stats, path) = gpu
            .select_resident(cam, BUDGET, &mut fb)
            .expect("near overflow must fall back, not error");

        assert_eq!(path, SelectPath::CpuFallback, "street view must overflow the 64-unit band");
        assert_eq!(gpu.fallback_count(), 1, "exactly one fallback");
        assert!(!cpu_out.draws().is_empty(), "street view must emit draws");
        assert_eq!(
            fb.draws(),
            cpu_out.draws(),
            "fallback draws must EXACTLY equal the CPU selector"
        );
        assert_eq!(stats.selected, cpu_stats.selected);
        assert_eq!(stats.lod_histogram, cpu_stats.lod_histogram);

        // The zeroed tail: an indirect expand off this select expands 0 atoms.
        let pbytes = read_back(&ctx, gpu.expand_params_buf(), 0, 16);
        let p: &[u32] = bytemuck::cast_slice(&pbytes);
        assert_eq!(p[0], 0, "overflow must zero ExpandParams.total");
        assert_eq!(p[1], 0, "overflow must zero ExpandParams.draw_count");
        println!(
            "[resident_walk] fallback: pairs>64 -> CpuFallback, draws equal cpu, fallback_count=1, expand_total=0"
        );
    }

    /// The resident breakdown: NO host walk, NO host assembly, exactly ONE
    /// blocking readback (the 64 B stats map). `walk_gpu` is `n/a` on this
    /// device by construction (`Features::empty()` — no TIMESTAMP_QUERY).
    #[test]
    fn resident_walk_breakdown() {
        let Some(ctx) = try_gpu_context("resident_walk") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[0];
        let mut fb = InstancedSelection::new();
        let (stats, path) = gpu
            .select_resident(cam, 250_000, &mut fb)
            .expect("resident select");
        assert_eq!(path, SelectPath::GpuResident);
        assert!(stats.selected > 0, "scene must select atoms");

        let b = gpu.last_breakdown();
        assert_eq!(
            b.walk_ms, 0.0,
            "the host walk span must be exactly 0.0 — the walk is GPU-resident"
        );
        assert_eq!(
            b.assemble_ms, 0.0,
            "no host WorkUnit assembly on the resident path"
        );
        assert_eq!(b.pair_build_ms, 0.0, "the host pair loop stays deleted");
        assert_eq!(b.syncs, 1, "ONE 64 B stats readback is the only sync");
        assert!(
            b.gpu_span_ms > 0.0,
            "gpu_span_ms must be a real submit->map span (got {})",
            b.gpu_span_ms
        );
        let wg = match b.walk_gpu_ms {
            Some(ms) => format!("{ms:.3}"),
            None => "n/a".to_string(),
        };
        println!(
            "[resident_walk] breakdown: walk={:.3} assemble={:.3} syncs={} walk_gpu={}",
            b.walk_ms, b.assemble_ms, b.syncs, wg
        );
    }

    /// M3.1 Task 1 instrument, tightened by Task 4 — the per-stage select
    /// breakdown: one warmup + one measured select must fill real host spans
    /// (gpu_span, walk), do exactly ONE blocking readback poll (the fused
    /// k1→scan→emit→k2 submit), and report `pair_build_ms == 0.0` (pair
    /// emission is GPU-side; the host pair loop is deleted). The timestamp
    /// columns are `None` here by construction — the test device requests
    /// `Features::empty()`, so `n/a` is the correct print on this device.
    #[test]
    fn instanced_gpu_breakdown_reports_stages() {
        let Some(ctx) = try_gpu_context("instanced_gpu") else {
            return;
        };
        let (lib, instances) = city_scene();
        let mut gpu = InstancedSelectGpu::new_with_context(
            &ctx,
            lib.clone(),
            instances.len() as u32,
            262_144,
        )
        .expect("city scene fits default device limits");
        gpu.set_instances(&instances);

        let cam = &city_cameras()[0];
        let mut out = InstancedSelection::new();
        gpu.select(cam, 250_000, &mut out).expect("warmup select");
        gpu.select(cam, 250_000, &mut out).expect("measured select");

        let b = gpu.last_breakdown();
        assert!(
            b.gpu_span_ms > 0.0,
            "gpu_span_ms must be a real submit->map span (got {})",
            b.gpu_span_ms
        );
        assert!(
            b.walk_ms > 0.0,
            "walk_ms must be a real budget-walk span (got {})",
            b.walk_ms
        );
        assert_eq!(
            b.syncs, 1,
            "the fused select does exactly 1 blocking readback poll"
        );
        assert_eq!(
            b.pair_build_ms, 0.0,
            "pair emission is GPU-side — the host pair-build span must be 0.0"
        );
        assert!(b.assemble_ms >= 0.0, "assemble span must be non-negative");
        let k1 = match b.k1_gpu_ms {
            Some(ms) => format!("{ms:.3}"),
            None => "n/a".to_string(),
        };
        println!(
            "[instanced_gpu] breakdown: gpu_span={:.3} pair_build={:.3} assemble={:.3} walk={:.3} syncs={} k1_gpu={}",
            b.gpu_span_ms, b.pair_build_ms, b.assemble_ms, b.walk_ms, b.syncs, k1
        );
    }
}
