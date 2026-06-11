// instanced_select_gpu.wgsl
// Fused instance+cluster scoring for `InstancedSelectGpu` — the GPU port
// of the per-instance / per-(instance,cluster) floating-point work inside
// `InstancedSelector::select`'s instance pass (M3 of virtualized splat
// rendering, the `atom_budget_gpu.wgsl` doctrine: parallel scoring on device,
// the tiny sequential budget walk on host). M3.1 Task 4 fused the chain into
// ONE submit: k1 → scan_pairs → emit_pairs → k2 (indirect), so the host does
// a single readback and never builds the pair list.
//
// KERNEL 1 — `score_instances`, ONE THREAD PER INSTANCE. The CPU op order,
// mirrored exactly:
//
//   world_center = quat_rotate(rot, bounds_center) + pos
//   cull (kind 0) iff asset >= asset_count OR !contains_sphere(world_center,
//                                                              bounds_radius)
//   inst_distance = length(world_center - eye)            (NO max here)
//   far (kind 1) iff inst_distance >= 150.0:
//       d     = max(inst_distance, 1e-3)
//       score = total_opacity * radius² / d²
//       level = 1 iff inst_distance >= 400.0 else 0
//   near (kind 2) otherwise — scan/emit build the (instance, cluster) pairs.
//
// `scan_pairs` — ONE thread, a deterministic SERIAL prefix scan over the
// kernel-1 results in ascending instance order: near instances get their
// exclusive pair-base offset, everything else 0xffffffff; writes the total to
// `pair_meta.pair_count` and the kernel-2 indirect dispatch args (ZERO
// workgroups when the total overflows `params.max_near_units` — the host then
// takes the bit-identical CPU fallback).
//
// `emit_pairs` — one thread per instance: near instances write their
// `(instance, cluster_base + ci)` pairs at `pair_base[i]` — ascending
// (instance, cluster), EXACTLY the order the deleted host loop built.
//
// KERNEL 2 — `score_pairs`, ONE THREAD PER NEAR (instance, cluster) PAIR
// (guarded by `pair_meta.pair_count`, dispatched indirect off `k2_args`):
//
//   frustum_centre = quat_rotate(rot, cm.frustum_center) + pos
//   passes         = contains_sphere(frustum_centre, cm.radius)
//   centre         = quat_rotate(rot, cm.centroid) + pos
//   d              = max(length(centre - eye), 1e-3)
//   screen         = 1000.0 * cm.radius / d
//   distance_lod   = select_lod_level(d, screen)
//   score          = cm.total_opacity * cm.radius² / d²
//
// The 150.0 / 400.0 thresholds hardcode the CPU consts `FAR_INSTANCE_M` /
// `IMPOSTER_I1_M`. `select_lod_level` + `contains_sphere` are VERBATIM from
// `atom_budget_gpu.wgsl`; `quat_rotate` is VERBATIM from `expand_draws.wgsl`
// (glam `Quat::mul_vec3`). The 6 frustum planes are host-normalized
// (`atom_budget_gpu::frustum_planes`) so the plane floats are bit-equal to
// the CPU oracle's. No fast-math (naga/wgpu emit strict IEEE): mirroring the
// op order is the determinism mechanism — the BARY_EPS lesson.

struct Params {
    instance_count: u32,
    // The pair-capacity band: a scanned total above this caps kernel 2 to
    // zero workgroups and routes the host to the CPU fallback.
    max_near_units: u32,
    asset_count: u32,
    // The atom budget for the GPU-resident walk (M3.2 — the old `_pad` lane;
    // same 32 B layout, the host-walk `select()` path never reads it).
    budget: u32,
    // Camera eye (inverse-view translation), padded to vec4 alignment.
    eye: vec4<f32>,
}

// Mirror of the 48 B `AtomInstance` POD (std430: vec3 @0, asset @12, rot @16,
// instance_id @32, pads — stride 48, byte-identical to the Rust layout).
struct Instance {
    pos: vec3<f32>,
    asset: u32,
    rot: vec4<f32>,
    instance_id: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// Per-asset static data: local bounds sphere, summed cluster opacity, and the
// asset's slice of the flattened global cluster-meta table (base + count) for
// the GPU-side pair scan/emit. Same 32 B as the pre-Task-4 layout — the old
// `opacity: vec4` is repacked as one f32 (the only lane ever read) + the two
// cluster fields + pad, so kernel 1's score math is bit-identical.
struct AssetInfo {
    // bounds_center.xyz, bounds_radius in .w
    center_radius: vec4<f32>,
    // asset total_opacity (the old opacity.x — same float, same bits)
    opacity: f32,
    cluster_base: u32,
    cluster_count: u32,
    // Offset of the asset's atoms in the packed library table (M3.2 — the old
    // `_pad` lane; `emit_draws` writes it into `ExpandDraw.atom_base`).
    atom_base: u32,
}

// One asset-local cluster of the flattened global meta table (the library's
// `ClusterMeta`, ascending asset then ascending cluster id).
struct ClusterMeta {
    // AABB-midpoint frustum centre .xyz, AABB half-diagonal radius in .w
    frustum_radius: vec4<f32>,
    // splat centroid .xyz, cluster total_opacity in .w
    centroid_opacity: vec4<f32>,
}

// One frustum plane: normalized normal in .xyz, plane d in .w. Host-normalized.
struct Plane {
    n_d: vec4<f32>,
}

// One near (instance, global cluster) pair, emitted by `emit_pairs` in
// ascending (instance, cluster) order — the deleted host loop's exact order.
struct Pair {
    instance: u32,
    cluster: u32,
}

// The scan result the host reads back alongside the scores: the total near
// (instance, cluster) pair demand this select. 16 B.
struct PairMeta {
    pair_count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// Kernel-1 output, 16 B. kind: 0 = culled, 1 = far, 2 = near.
struct InstScore {
    kind: u32,
    // Far (kind 1): initial imposter level (0 = I0, 1 = I1). Near (kind 2):
    // the asset's cluster_count, stashed so the serial scan reads ONLY this
    // sequential stream (no dependent instances→assets loads per iteration).
    // The host reads it for far instances only.
    level: u32,
    score: f32,
    // d = max(inst_distance, 1e-3) for far instances.
    distance: f32,
}

// Kernel-2 output, 16 B.
struct PairScore {
    // 0 or 1 — the transformed cluster sphere's frustum result.
    passes: u32,
    // Distance-driven LOD ceiling (0..=3).
    distance_lod: u32,
    score: f32,
    // d = max(length(centre - eye), 1e-3).
    distance: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> assets: array<AssetInfo>;
@group(0) @binding(3) var<storage, read> cluster_metas: array<ClusterMeta>;
@group(0) @binding(4) var<storage, read> planes: array<Plane>;   // exactly 6
@group(0) @binding(5) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(6) var<storage, read_write> out_instances: array<InstScore>;
@group(0) @binding(7) var<storage, read_write> out_pairs: array<PairScore>;
// Exclusive per-instance pair-base offsets (0xffffffff for non-near), written
// by `scan_pairs`, consumed by `emit_pairs`.
@group(0) @binding(8) var<storage, read_write> pair_base: array<u32>;
@group(0) @binding(9) var<storage, read_write> pair_meta: PairMeta;
// Kernel-2 indirect dispatch args (x, y, z workgroups). Bound ONLY in the
// scan pipeline's layout — the indirect dispatch itself must not see this
// buffer as writable storage in its own usage scope (wgpu usage rules).
@group(0) @binding(10) var<storage, read_write> k2_args: array<u32, 3>;

// ─────────────────────────────────────────────────────────────────────────────
// M3.2 — the GPU-resident budget walk (sort-free radix-bucketed threshold
// refinement over the unique 64-bit (score-bits, tie) key; see the module doc
// of `instanced_select_gpu.rs` for the equivalence theorem).
// ─────────────────────────────────────────────────────────────────────────────

// One selectable walk unit — 32 B, slot index == `key_lo` (the tie): far
// units and near pairs land at their unit-base rank, which is contiguous
// 0..unit_total and order-isomorphic to the CPU `work_idx`.
struct WalkUnit {
    // bitcast<u32>(score) — scores are finite and >= 0, where non-negative
    // IEEE f32 bit patterns are monotonic as u32.
    key_hi: u32,
    // The tie: the unit's rank in the per-instance unit-base scan.
    key_lo: u32,
    instance: u32,
    asset: u32,
    // Asset-local cluster id; 0xffffffff = imposter unit.
    unit_key: u32,
    distance_lod: u32,
    distance: f32,
    // 1 = live; 0 = dead slot (frustum-failing near pair) — contributes 0
    // everywhere and is never emitted.
    flags: u32,
}

// The walk's mutable state. Buckets + every scalar up to (and including)
// `_pad1` are cleared by the encoder each `select_resident`; the trailing
// constant region (`blocks_base`) is written once at construction.
struct WalkState {
    // 256 u32 weight buckets per refinement pass — deterministic by integer
    // associativity/commutativity regardless of atomicAdd arrival order.
    buckets: array<atomic<u32>, 256>,
    // Σ count(distance_lod) over live units (the CPU walk's starting total).
    total0: atomic<u32>,
    // Σ (count(distance_lod) − count(max_level)) — the total demotable weight.
    w_total: atomic<u32>,
    // far_count + pair_count — the unit-domain size (0 on overflow).
    unit_total: u32,
    // 1 when the scanned pair demand overflows max_near_units — every fixed
    // dispatch early-outs; the indirect tail already has zero workgroups.
    overflow: u32,
    // The scanned pair demand (build_near_units' thread guard).
    pair_count: u32,
    // 0..15 — which (spread, pick) refinement pass is running (0–7 demote,
    // 8–15 shed). Advanced by every pick, active or not.
    pass_idx: u32,
    // The demote boundary key, resolved 8 bits per pass, plus the cumulative
    // weight strictly below it.
    d_prefix_hi: u32,
    d_prefix_lo: u32,
    d_below: u32,
    // The shed boundary key + cumulative weight below it.
    s_prefix_hi: u32,
    s_prefix_lo: u32,
    s_below: u32,
    // walk_finalize: 0 = no walk (total0 <= budget), 1 = demote boundary,
    // 2 = full demotion + shed.
    mode: u32,
    // total0 − budget (mode != 0).
    need: u32,
    // Grand totals of the two-level emission scan (scan_totals).
    emit_atoms: u32,
    emit_draw_count: u32,
    _pad0: u32,
    _pad1: u32,
    // CONSTANT (never cleared): first block-totals slot in `counts`.
    blocks_base: u32,
}

// `ExpandDraw` — the EXACT 32 B lowered form `ExpandDrawsPass::encode`
// uploads (`expand_draws.rs`); Task 2's indirect expand binds this buffer
// unmodified.
struct ExpandDrawOut {
    atom_offset: u32,
    atom_count: u32,
    index_offset: u32,
    atom_base: u32,
    instance: u32,
    opacity_scale: f32,
    pad0: u32,
    pad1: u32,
}

// `expand_draws.wgsl` binding-0 `ExpandParams` (16 B) — written by
// emit_finalize as storage, bound as uniform by the indirect expand.
struct ExpandParamsOut {
    total: u32,
    draw_count: u32,
    pad0: u32,
    pad1: u32,
}

// The ONE 64 B readback (`GpuWalkStats` POD in Rust).
struct WalkStats {
    pair_count: u32,
    selected: u32,
    draw_count: u32,
    visible: u32,
    culled: u32,
    far_count: u32,
    lod_histogram: array<atomic<u32>, 6>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

// Exclusive per-instance unit-base offsets (the tie ranks), written by
// `scan_pairs` for EVERY instance (culled instances advance nothing).
@group(0) @binding(11) var<storage, read_write> ubase: array<u32>;
@group(0) @binding(12) var<storage, read_write> units: array<WalkUnit>;
// Per-unit (atom_count, draw_flag) pairs, scanned in place to exclusive
// (atom_offset, draw_slot); block totals live at walk_state.blocks_base.
@group(0) @binding(13) var<storage, read_write> counts: array<vec2<u32>>;
@group(0) @binding(14) var<storage, read_write> walk_state: WalkState;
// Walk indirect dispatch args: slot 0 (offset 0) = 64-wide unit domain,
// slot 1 (offset 12) = 256-wide scan blocks, slot 2 reserved. Bound ONLY in
// the scan pipeline's layout (the k2_args usage rule).
@group(0) @binding(15) var<storage, read_write> walk_args: array<u32, 9>;
// The once-uploaded flat (index_offset, len) range table: 4 slots per global
// cluster, then 2 per asset at imposter_base = total_clusters * 4.
@group(0) @binding(16) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(17) var<storage, read_write> draws: array<ExpandDrawOut>;
@group(0) @binding(18) var<storage, read_write> expand_params: ExpandParamsOut;
// Expand indirect dispatch args — never dispatched indirect inside THIS
// chain, so emit_finalize may bind it writable.
@group(0) @binding(19) var<storage, read_write> expand_args: array<u32, 3>;
@group(0) @binding(20) var<storage, read_write> stats: WalkStats;

// Mirror of `hierarchical_lod::select_lod_level` — VERBATIM from
// `atom_budget_gpu.wgsl`. LOD_DISTANCES = [0,50,150,400].
fn select_lod_level(distance: f32, screen_size: f32) -> u32 {
    if (screen_size < 10.0 || distance > 400.0) {
        return 3u;
    } else if (screen_size < 50.0 || distance > 150.0) {
        return 2u;
    } else if (screen_size < 200.0 || distance > 50.0) {
        return 1u;
    } else {
        return 0u;
    }
}

// Mirror of `Frustum::contains_sphere` — VERBATIM from `atom_budget_gpu.wgsl`:
// cull (return 0) iff for any plane `dot(normal, centre) + d < -radius`.
fn contains_sphere(centre: vec3<f32>, radius: f32) -> u32 {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let p = planes[i].n_d;
        let dist = p.x * centre.x + p.y * centre.y + p.z * centre.z + p.w;
        if (dist < -radius) {
            return 0u;
        }
    }
    return 1u;
}

// glam Quat::mul_vec3 — VERBATIM from `expand_draws.wgsl`:
// v' = v(w² − b·b) + b(2 v·b) + (b × v)(2w).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    return v * (q.w * q.w - dot(b, b)) + b * (dot(v, b) * 2.0) + cross(b, v) * (q.w * 2.0);
}

@compute @workgroup_size(64)
fn score_instances(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.instance_count) {
        return;
    }
    let inst = instances[i];

    var s: InstScore;
    s.kind = 0u;
    s.level = 0u;
    s.score = 0.0;
    s.distance = 0.0;

    // Instances referencing an unknown asset count as culled (the CPU's
    // `lib.assets.get(..) else { culled }` arm).
    if (inst.asset >= params.asset_count) {
        out_instances[i] = s;
        return;
    }
    let a = assets[inst.asset];
    let radius = a.center_radius.w;

    // world_center = q.mul_vec3(bounds_center) + t   (CPU op order)
    let world_center = quat_rotate(inst.rot, a.center_radius.xyz) + inst.pos;
    if (contains_sphere(world_center, radius) == 0u) {
        out_instances[i] = s;
        return;
    }

    // inst_distance = (world_center - eye).length()   (NO max — the CPU's
    // raw classification distance).
    let inst_distance = length(world_center - params.eye.xyz);
    if (inst_distance >= 150.0) {
        // Far: the [I0, I1] imposter chain unit, scored on the asset bounds.
        let d = max(inst_distance, 1e-3);
        s.kind = 1u;
        if (inst_distance >= 400.0) {
            s.level = 1u;
        } else {
            s.level = 0u;
        }
        s.score = a.opacity * (radius * radius) / (d * d);
        s.distance = d;
    } else {
        // Near: per-cluster granularity — kernel 2 scores the pairs. The
        // asset's cluster_count rides in `level` (unused for near, never read
        // by the host for kind 2) so `scan_pairs` needs no asset indirection.
        s.kind = 2u;
        s.level = a.cluster_count;
        s.distance = inst_distance;
    }
    out_instances[i] = s;
}

// One serial-scan step (M3.2 factored the per-instance body out so the walk's
// unit-base scan, the stats tallies, and the pair scan stay ONE in-order
// accumulation): near instances claim `level` (= cluster_count) pair slots
// AND unit slots; far instances claim one unit slot; culled instances claim
// nothing. Called on PRELOADED scores — the callers keep the 8-wide batch.
fn scan_step(
    j: u32,
    sc: InstScore,
    pair_total: ptr<function, u32>,
    unit_total: ptr<function, u32>,
    vis: ptr<function, u32>,
    cul: ptr<function, u32>,
    far_n: ptr<function, u32>,
) {
    if (sc.kind == 2u) {
        pair_base[j] = *pair_total;
        *pair_total = *pair_total + sc.level;
        ubase[j] = *unit_total;
        *unit_total = *unit_total + sc.level;
        *vis = *vis + 1u;
    } else if (sc.kind == 1u) {
        pair_base[j] = 0xffffffffu;
        ubase[j] = *unit_total;
        *unit_total = *unit_total + 1u;
        *vis = *vis + 1u;
        *far_n = *far_n + 1u;
    } else {
        pair_base[j] = 0xffffffffu;
        ubase[j] = *unit_total;
        *cul = *cul + 1u;
    }
}

// ONE thread: the deterministic serial prefix scan over kernel 1's results in
// ascending instance order — the exclusive pair-base per near instance, the
// total pair demand, and kernel 2's indirect args; since M3.2 also the walk's
// unit-base scan (`ubase`, tie ranks), the unit-domain total + indirect args,
// and the visible/culled/far stats. Overflow of the `max_near_units` band
// caps the indirect dispatch to ZERO workgroups AND zeroes the walk args (the
// whole resident tail no-ops); the host reads the pair count and takes the
// bit-identical CPU fallback.
@compute @workgroup_size(1)
fn scan_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var total = 0u;
    var ub = 0u;
    var vis = 0u;
    var cul = 0u;
    var far_n = 0u;
    let n = params.instance_count;
    var i = 0u;
    // Serial walk, batched 8 wide: the 8 loads per batch are independent
    // (k1 stashed each near asset's cluster_count in `level`, so no
    // instances→assets indirection), letting the single wave issue them
    // before the in-order accumulation consumes them — a pure pipelining
    // unroll of the same serial walk: same thread, same order, same sums.
    // One dependent 16 B load per iteration measured 2.15 ms at 10k
    // instances on the 780M; the batch hides most of that latency.
    loop {
        if (i + 8u > n) {
            break;
        }
        let s0 = out_instances[i];
        let s1 = out_instances[i + 1u];
        let s2 = out_instances[i + 2u];
        let s3 = out_instances[i + 3u];
        let s4 = out_instances[i + 4u];
        let s5 = out_instances[i + 5u];
        let s6 = out_instances[i + 6u];
        let s7 = out_instances[i + 7u];
        scan_step(i, s0, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 1u, s1, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 2u, s2, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 3u, s3, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 4u, s4, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 5u, s5, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 6u, s6, &total, &ub, &vis, &cul, &far_n);
        scan_step(i + 7u, s7, &total, &ub, &vis, &cul, &far_n);
        i = i + 8u;
    }
    for (var j = i; j < n; j = j + 1u) {
        scan_step(j, out_instances[j], &total, &ub, &vis, &cul, &far_n);
    }
    pair_meta.pair_count = total;
    let overflow = total > params.max_near_units;
    let capped = select(total, 0u, overflow);
    k2_args[0] = (capped + 63u) / 64u;
    k2_args[1] = 1u;
    k2_args[2] = 1u;
    // ── The M3.2 walk extension: unit domain, args, stats, overflow flag. ──
    walk_state.pair_count = total;
    walk_state.overflow = select(0u, 1u, overflow);
    let units_n = select(ub, 0u, overflow);
    walk_state.unit_total = units_n;
    stats.pair_count = total;
    stats.visible = vis;
    stats.culled = cul;
    stats.far_count = far_n;
    walk_args[0] = (units_n + 63u) / 64u;   // slot 0: 64-wide unit domain
    walk_args[1] = 1u;
    walk_args[2] = 1u;
    walk_args[3] = (units_n + 255u) / 256u; // slot 1: 256-wide scan blocks
    walk_args[4] = 1u;
    walk_args[5] = 1u;
    walk_args[6] = 0u;                      // slot 2: reserved
    walk_args[7] = 0u;
    walk_args[8] = 0u;
}

// One thread per instance: near instances write their (instance, cluster)
// pairs at the scanned base — ascending (instance, cluster), EXACTLY the
// order the deleted host loop built. On overflow (pair_count above the band)
// nothing is emitted: kernel 2 dispatches zero workgroups and the pair buffer
// (sized max_near_units) must not be written past its end.
@compute @workgroup_size(64)
fn emit_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.instance_count) {
        return;
    }
    if (pair_meta.pair_count > params.max_near_units) {
        return;
    }
    let base = pair_base[i];
    if (base == 0xffffffffu) {
        return;
    }
    let a = assets[instances[i].asset];
    for (var ci = 0u; ci < a.cluster_count; ci = ci + 1u) {
        pairs[base + ci] = Pair(i, a.cluster_base + ci);
    }
}

@compute @workgroup_size(64)
fn score_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pair_meta.pair_count) {
        return;
    }
    let pr = pairs[i];
    let inst = instances[pr.instance];
    let cm = cluster_metas[pr.cluster];
    let radius = cm.frustum_radius.w;

    var s: PairScore;

    // frustum_centre = q.mul_vec3(cm.frustum_center) + t — the oracle's
    // BVH-leaf test on the transformed cluster sphere.
    let frustum_centre = quat_rotate(inst.rot, cm.frustum_radius.xyz) + inst.pos;
    s.passes = contains_sphere(frustum_centre, radius);

    // Score/LOD off the transformed centroid — the oracle's exact formulas.
    let centre = quat_rotate(inst.rot, cm.centroid_opacity.xyz) + inst.pos;
    let d = max(length(centre - params.eye.xyz), 1e-3);
    let screen = 1000.0 * radius / d;
    s.distance_lod = select_lod_level(d, screen);
    s.score = cm.centroid_opacity.w * (radius * radius) / (d * d);
    s.distance = d;

    out_pairs[i] = s;
}

// ─────────────────────────────────────────────────────────────────────────────
// M3.2 — the GPU budget walk kernels.
// ─────────────────────────────────────────────────────────────────────────────

const IMPOSTER_KEY: u32 = 0xffffffffu;

// First imposter range slot: the flat table holds total_clusters * 4 cluster
// slots, then asset_count * 2 imposter slots — recovered from the array
// length so the 32 B Params layout stays untouched.
fn ranges_imposter_base() -> u32 {
    return arrayLength(&ranges) - params.asset_count * 2u;
}

// Atom count of `(asset, unit_key)` at chain level `lod` — the CPU walk's
// `unit_len`, looked up from the once-uploaded flat range table.
fn unit_count_at(asset: u32, unit_key: u32, lod: u32) -> u32 {
    if (unit_key == IMPOSTER_KEY) {
        return ranges[ranges_imposter_base() + asset * 2u + lod].y;
    }
    return ranges[(assets[asset].cluster_base + unit_key) * 4u + lod].y;
}

// The 8-bit digit of the 64-bit key for refinement pass k (0..7, top-down).
fn key_digit(key_hi: u32, key_lo: u32, k: u32) -> u32 {
    if (k < 4u) {
        return (key_hi >> ((3u - k) * 8u)) & 0xffu;
    }
    return (key_lo >> ((7u - k) * 8u)) & 0xffu;
}

// Does the key's top 8*k bits equal the resolved prefix (pass k competes on
// digit k among keys matching the k already-resolved digits)?
fn prefix_match(key_hi: u32, key_lo: u32, p_hi: u32, p_lo: u32, k: u32) -> bool {
    if (k == 0u) {
        return true;
    }
    if (k <= 4u) {
        let sh = (4u - k) * 8u;
        return (key_hi >> sh) == (p_hi >> sh);
    }
    if (key_hi != p_hi) {
        return false;
    }
    let sh = (8u - k) * 8u;
    return (key_lo >> sh) == (p_lo >> sh);
}

// Lexicographic 64-bit key compare: -1 / 0 / +1.
fn key_cmp(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> i32 {
    if (a_hi < b_hi) { return -1; }
    if (a_hi > b_hi) { return 1; }
    if (a_lo < b_lo) { return -1; }
    if (a_lo > b_lo) { return 1; }
    return 0;
}

// Mirror of `hierarchical_lod::crossfade_factor` — VERBATIM op order
// (LOD_DISTANCES = [0, 50, 150, 400]; the transition band is the last 20% of
// each level's range; level >= 3 never fades). Same fixed-op-order f32
// bit-equality mechanism as the scores.
fn crossfade_factor(distance: f32, level: u32) -> f32 {
    if (level >= 3u) {
        return 0.0; // No transition beyond last level.
    }
    var current_dist = 0.0;
    var next_dist = 50.0;
    if (level == 1u) {
        current_dist = 50.0;
        next_dist = 150.0;
    } else if (level == 2u) {
        current_dist = 150.0;
        next_dist = 400.0;
    }
    let range = next_dist - current_dist;
    if (range <= 0.0) {
        return 0.0;
    }
    let transition_start = current_dist + range * 0.8;
    if (distance <= transition_start) {
        return 0.0;
    } else if (distance >= next_dist) {
        return 1.0;
    }
    let transition_range = next_dist - transition_start;
    return clamp((distance - transition_start) / transition_range, 0.0, 1.0);
}

// One thread per instance: far (kind 1) instances write their imposter-chain
// walk unit at slot ubase[i] (== the tie) and accumulate the deterministic
// u32 totals. Near/culled instances are handled elsewhere/not at all.
@compute @workgroup_size(64)
fn build_far_units(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.instance_count || walk_state.overflow != 0u) {
        return;
    }
    let s = out_instances[i];
    if (s.kind != 1u) {
        return;
    }
    let tie = ubase[i];
    let asset = instances[i].asset;
    var u: WalkUnit;
    u.key_hi = bitcast<u32>(s.score);
    u.key_lo = tie;
    u.instance = i;
    u.asset = asset;
    u.unit_key = IMPOSTER_KEY;
    u.distance_lod = s.level;
    u.distance = s.distance;
    u.flags = 1u;
    units[tie] = u;
    let c0 = unit_count_at(asset, IMPOSTER_KEY, s.level);
    let cmax = unit_count_at(asset, IMPOSTER_KEY, 1u);
    atomicAdd(&walk_state.total0, c0);
    atomicAdd(&walk_state.w_total, c0 - cmax);
}

// One thread per near (instance, cluster) pair (indirect off k2_args — zero
// workgroups on overflow): writes the cluster walk unit at ubase[i] + ci.
// Frustum-failing pairs become DEAD slots (flags = passes = 0) contributing
// 0 everywhere — the tie map stays order-isomorphic to the CPU work_idx.
@compute @workgroup_size(64)
fn build_near_units(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= walk_state.pair_count || walk_state.overflow != 0u) {
        return;
    }
    let pr = pairs[p];
    let ps = out_pairs[p];
    // ci = position within the instance's pair block == the asset-local
    // cluster id (build_clusters assigns sequential ids; position == id).
    let ci = p - pair_base[pr.instance];
    let tie = ubase[pr.instance] + ci;
    var u: WalkUnit;
    u.key_hi = bitcast<u32>(ps.score);
    u.key_lo = tie;
    u.instance = pr.instance;
    u.asset = instances[pr.instance].asset;
    u.unit_key = ci;
    u.distance_lod = ps.distance_lod;
    u.distance = ps.distance;
    u.flags = ps.passes;
    units[tie] = u;
    if (ps.passes != 0u) {
        // pr.cluster IS the global flattened cluster index — the range table
        // is laid out in the same order (no assets indirection needed here).
        let c0 = ranges[pr.cluster * 4u + ps.distance_lod].y;
        let cmax = ranges[pr.cluster * 4u + 3u].y;
        atomicAdd(&walk_state.total0, c0);
        atomicAdd(&walk_state.w_total, c0 - cmax);
    }
}

// One thread per unit (indirect, walk_args slot 0), 16 dispatches: every live
// unit whose key matches the resolved prefix atomicAdds its weight (demote
// passes 0–7: count(distance_lod) − count(max_level); shed passes 8–15:
// count(max_level)) into the 256 u32 buckets — deterministic by integer
// associativity/commutativity regardless of arrival order.
@compute @workgroup_size(64)
fn walk_bucket_spread(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tie = gid.x;
    if (walk_state.overflow != 0u || tie >= walk_state.unit_total) {
        return;
    }
    let total0 = atomicLoad(&walk_state.total0);
    if (total0 <= params.budget) {
        return; // no walk — every unit stays at distance_lod
    }
    let need = total0 - params.budget;
    let wt = atomicLoad(&walk_state.w_total);
    let pi = walk_state.pass_idx;
    let demote_phase = pi < 8u;
    if (demote_phase && need > wt) {
        return; // regime 3: full demotion needs no boundary — shed resolves it
    }
    if (!demote_phase && need <= wt) {
        return; // regime 2: the demote boundary closed the gap — no shed
    }
    let u = units[tie];
    if (u.flags == 0u) {
        return;
    }
    let maxl = select(3u, 1u, u.unit_key == IMPOSTER_KEY);
    var weight: u32;
    if (demote_phase) {
        weight = unit_count_at(u.asset, u.unit_key, u.distance_lod)
            - unit_count_at(u.asset, u.unit_key, maxl);
        if (weight == 0u) {
            return; // zero-weight units never decide the boundary
        }
    } else {
        weight = unit_count_at(u.asset, u.unit_key, maxl);
    }
    let k = pi & 7u;
    var p_hi = walk_state.d_prefix_hi;
    var p_lo = walk_state.d_prefix_lo;
    if (!demote_phase) {
        p_hi = walk_state.s_prefix_hi;
        p_lo = walk_state.s_prefix_lo;
    }
    if (!prefix_match(u.key_hi, u.key_lo, p_hi, p_lo, k)) {
        return;
    }
    atomicAdd(&walk_state.buckets[key_digit(u.key_hi, u.key_lo, k)], weight);
}

var<workgroup> pick_buckets: array<u32, 256>;

// ONE 256-thread workgroup: snapshot + zero the buckets, then thread 0
// serially accumulates them ascending, fixes the next 8 prefix bits where the
// running sum crosses the target, and carries the below-prefix accumulation
// forward. Always advances pass_idx (shed passes must see pi 8..15 even when
// the demote passes were inactive).
@compute @workgroup_size(256)
fn walk_bucket_pick(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    pick_buckets[t] = atomicLoad(&walk_state.buckets[t]);
    atomicStore(&walk_state.buckets[t], 0u); // clean for the next pass
    workgroupBarrier();
    if (t != 0u) {
        return;
    }
    let pi = walk_state.pass_idx;
    walk_state.pass_idx = pi + 1u;
    if (walk_state.overflow != 0u || walk_state.unit_total == 0u) {
        return;
    }
    let total0 = atomicLoad(&walk_state.total0);
    if (total0 <= params.budget) {
        return;
    }
    let need = total0 - params.budget;
    let wt = atomicLoad(&walk_state.w_total);
    let demote_phase = pi < 8u;
    if (demote_phase && need > wt) {
        return;
    }
    if (!demote_phase && need <= wt) {
        return;
    }
    // `target` is WGSL-reserved — `goal` is the crossing target.
    var goal: u32;
    var below: u32;
    if (demote_phase) {
        goal = need;
        below = walk_state.d_below;
    } else {
        goal = need - wt; // the shed need: what full demotion left over
        below = walk_state.s_below;
    }
    var run = 0u;
    var digit = 255u;
    for (var d = 0u; d < 256u; d = d + 1u) {
        let b = pick_buckets[d];
        if (below + run + b >= goal) {
            digit = d;
            break;
        }
        run = run + b;
    }
    let k = pi & 7u;
    if (demote_phase) {
        if (k < 4u) {
            walk_state.d_prefix_hi = walk_state.d_prefix_hi | (digit << ((3u - k) * 8u));
        } else {
            walk_state.d_prefix_lo = walk_state.d_prefix_lo | (digit << ((7u - k) * 8u));
        }
        walk_state.d_below = below + run;
    } else {
        if (k < 4u) {
            walk_state.s_prefix_hi = walk_state.s_prefix_hi | (digit << ((3u - k) * 8u));
        } else {
            walk_state.s_prefix_lo = walk_state.s_prefix_lo | (digit << ((7u - k) * 8u));
        }
        walk_state.s_below = below + run;
    }
}

// ONE thread: consolidate the walk regime for final_counts/emit_draws.
@compute @workgroup_size(1)
fn walk_finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    if (walk_state.overflow != 0u) {
        walk_state.mode = 0u;
        walk_state.need = 0u;
        return;
    }
    let total0 = atomicLoad(&walk_state.total0);
    if (total0 <= params.budget) {
        walk_state.mode = 0u;
        walk_state.need = 0u;
        return;
    }
    let need = total0 - params.budget;
    walk_state.need = need;
    let wt = atomicLoad(&walk_state.w_total);
    walk_state.mode = select(2u, 1u, need <= wt);
}

// The per-unit end state — the pure function of (key vs demote boundary,
// key vs shed boundary, distance_lod) the equivalence theorem reduces the
// CPU heap walk to. Returns (final lod, final count); count 0 = shed.
fn final_lod_count(u: WalkUnit) -> vec2<u32> {
    let maxl = select(3u, 1u, u.unit_key == IMPOSTER_KEY);
    var lod = u.distance_lod;
    let mode = walk_state.mode;
    if (mode == 1u) {
        let c = key_cmp(u.key_hi, u.key_lo, walk_state.d_prefix_hi, walk_state.d_prefix_lo);
        if (c < 0) {
            lod = maxl; // below the boundary: fully demoted
        } else if (c == 0) {
            // THE boundary unit (keys unique): the minimal level whose freed
            // atoms close the remaining gap — the CPU loop's mid-chain stop.
            let base = unit_count_at(u.asset, u.unit_key, u.distance_lod);
            let still = walk_state.need - walk_state.d_below;
            lod = maxl;
            for (var l = u.distance_lod + 1u; l <= maxl; l = l + 1u) {
                if (base - unit_count_at(u.asset, u.unit_key, l) >= still) {
                    lod = l;
                    break;
                }
            }
        }
        // c > 0: above the boundary — untouched at distance_lod.
    } else if (mode == 2u) {
        lod = maxl; // full demotion everywhere…
        if (key_cmp(u.key_hi, u.key_lo, walk_state.s_prefix_hi, walk_state.s_prefix_lo) <= 0) {
            return vec2<u32>(lod, 0u); // …and shed at or below the boundary
        }
    }
    return vec2<u32>(lod, unit_count_at(u.asset, u.unit_key, lod));
}

// One thread per unit (indirect, walk_args slot 0): the (atom_count,
// draw_flag) pair feeding the two-level deterministic exclusive scan.
@compute @workgroup_size(64)
fn final_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tie = gid.x;
    if (walk_state.overflow != 0u || tie >= walk_state.unit_total) {
        return;
    }
    let u = units[tie];
    if (u.flags == 0u) {
        counts[tie] = vec2<u32>(0u, 0u);
        return;
    }
    let lc = final_lod_count(u);
    counts[tie] = vec2<u32>(lc.y, select(0u, 1u, lc.y > 0u));
}

var<workgroup> scan_sh: array<vec2<u32>, 256>;

// Scan pass A (one workgroup per 256 units, indirect walk_args slot 1):
// workgroup-exclusive Hillis–Steele scan of (atoms, draws) in place; block
// totals written at walk_state.blocks_base + block. Deterministic: fixed
// barrier-ordered op order, u32 adds.
@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = lid.x;
    let e = wid.x * 256u + t;
    var v = vec2<u32>(0u, 0u);
    if (walk_state.overflow == 0u && e < walk_state.unit_total) {
        v = counts[e];
    }
    scan_sh[t] = v;
    for (var off = 1u; off < 256u; off = off << 1u) {
        workgroupBarrier();
        var add = vec2<u32>(0u, 0u);
        if (t >= off) {
            add = scan_sh[t - off];
        }
        workgroupBarrier();
        scan_sh[t] = scan_sh[t] + add;
    }
    workgroupBarrier();
    if (e < walk_state.unit_total) {
        counts[e] = scan_sh[t] - v; // inclusive → exclusive
    }
    if (t == 255u) {
        counts[walk_state.blocks_base + wid.x] = scan_sh[255u];
    }
}

// Scan pass B (ONE workgroup): serial exclusive scan over the block totals;
// the grand totals are the emission's (selected atoms, draw count).
@compute @workgroup_size(1)
fn scan_totals(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    if (walk_state.overflow != 0u) {
        walk_state.emit_atoms = 0u;
        walk_state.emit_draw_count = 0u;
        return;
    }
    let nb = (walk_state.unit_total + 255u) / 256u;
    var running = vec2<u32>(0u, 0u);
    for (var b = 0u; b < nb; b = b + 1u) {
        let bt = counts[walk_state.blocks_base + b];
        counts[walk_state.blocks_base + b] = running;
        running = running + bt;
    }
    walk_state.emit_atoms = running.x;
    walk_state.emit_draw_count = running.y;
}

// Scan pass C (indirect walk_args slot 1): add each block's exclusive prefix
// back onto its elements — counts[tie] becomes (atom_offset, draw_slot).
@compute @workgroup_size(256)
fn scan_add_back(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let e = wid.x * 256u + lid.x;
    if (walk_state.overflow != 0u || e >= walk_state.unit_total) {
        return;
    }
    counts[e] = counts[e] + counts[walk_state.blocks_base + wid.x];
}

// One thread per unit (indirect, walk_args slot 0): surviving units write
// their ExpandDraw at the compacted draw slot — in tie order, which IS the
// CPU emission order — and tally the lod histogram (u32 atomics).
@compute @workgroup_size(64)
fn emit_draws(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tie = gid.x;
    if (walk_state.overflow != 0u || tie >= walk_state.unit_total) {
        return;
    }
    let u = units[tie];
    if (u.flags == 0u) {
        return;
    }
    let lc = final_lod_count(u);
    if (lc.y == 0u) {
        return; // shed — the CPU emit loop's `count == 0` skip
    }
    let pre = counts[tie]; // (atom_offset, draw slot)
    var r: vec2<u32>;
    var opacity_scale = 1.0;
    if (u.unit_key == IMPOSTER_KEY) {
        r = ranges[ranges_imposter_base() + u.asset * 2u + lc.x];
        atomicAdd(&stats.lod_histogram[4u + lc.x], 1u);
    } else {
        r = ranges[(assets[u.asset].cluster_base + u.unit_key) * 4u + lc.x];
        // Crossfade only at the distance level — the CPU emit's exact rule.
        var fade = 0.0;
        if (lc.x == u.distance_lod) {
            fade = crossfade_factor(u.distance, lc.x);
        }
        opacity_scale = 1.0 - fade;
        atomicAdd(&stats.lod_histogram[lc.x], 1u);
    }
    var d: ExpandDrawOut;
    d.atom_offset = pre.x;
    d.atom_count = lc.y;
    d.index_offset = r.x;
    d.atom_base = assets[u.asset].atom_base;
    d.instance = u.instance;
    d.opacity_scale = opacity_scale;
    d.pad0 = 0u;
    d.pad1 = 0u;
    draws[pre.y] = d;
}

// ONE thread, the chain's last word: ExpandParams + the expand indirect args
// + the stats scalars. On overflow everything zeroes — an indirect expand off
// this select honestly expands nothing.
@compute @workgroup_size(1)
fn emit_finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var total = walk_state.emit_atoms;
    var draw_count = walk_state.emit_draw_count;
    if (walk_state.overflow != 0u) {
        total = 0u;
        draw_count = 0u;
    }
    expand_params.total = total;
    expand_params.draw_count = draw_count;
    expand_params.pad0 = 0u;
    expand_params.pad1 = 0u;
    expand_args[0] = (total + 255u) / 256u; // expand_draws.wgsl main is @256
    expand_args[1] = 1u;
    expand_args[2] = 1u;
    stats.selected = total;
    stats.draw_count = draw_count;
}
