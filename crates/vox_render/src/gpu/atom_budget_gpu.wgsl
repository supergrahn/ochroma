// Atom-budget LOD scorer — GPU compute port of the per-cluster scoring core of
// the CPU oracle `crate::atom_budget::AtomBudgetSelector::select`.
//
// ONE THREAD PER CLUSTER. Each thread reproduces, in the EXACT operation order
// of the CPU oracle, the per-cluster floating-point work that the oracle does
// inside its visible-cluster loop:
//
//   radius   = length((aabb_max - aabb_min) * 0.5)            , then max(_, 1e-4)
//   d        = length(center - eye)                           , then max(_, 1e-3)
//   screen   = 1000.0 * radius / d
//   dlod     = select_lod_level(d, screen)        (branch ladder, see below)
//   score    = total_opacity * (radius * radius) / (d * d)
//   pass     = frustum.contains_sphere(aabb_mid, radius)   (leaf-level test)
//
// `aabb_mid` and `radius` here are EXACTLY the values the oracle's BVH *leaf*
// test uses (`collect_visible`): centre = (aabb_min + aabb_max) * 0.5, radius =
// max(length((aabb_max - aabb_min) * 0.5), 1e-4). Note the scoring distance `d`
// instead uses the cluster CENTROID (`center`), not the AABB midpoint — the
// oracle does too (`cluster.center`). Both are uploaded per cluster.
//
// The 6 frustum planes are normalized HOST-side (identically to `frustum.rs`'s
// `Plane::from_vec4`) and uploaded, so plane floats are bit-identical to the
// oracle's; the GPU only evaluates `dot(normal, p) + d < -radius`.
//
// No fast-math (naga/wgpu emit strict IEEE), so mirroring the op order yields
// the same RADV rounding as the CPU (the BARY_EPS lesson, commit dae84d8).
//
// The HOST owns everything after this: the BVH structural walk (internal-node
// pruning), the budget demote/promote/shed sequence, and index emission. That
// part is a tiny sequential pass over a few hundred clusters where GPU sorting
// buys nothing and risks divergence. See the module doc for the split rationale.

// One cluster's static, pre-uploaded geometry. std430 layout: three vec4 = 48 B.
struct Cluster {
    // aabb_min.xyz, total_opacity in .w
    aabb_min_op: vec4<f32>,
    // aabb_max.xyz, _pad in .w
    aabb_max: vec4<f32>,
    // centroid.xyz (cluster.center), _pad in .w
    center: vec4<f32>,
};

// One frustum plane: normalized normal in .xyz, plane d in .w. Host-normalized.
struct Plane {
    n_d: vec4<f32>,
};

struct Params {
    cluster_count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    // Camera eye (inverse-view translation), padded to vec4 alignment.
    eye: vec4<f32>,
};

// Per-cluster scoring output. std430: 16 bytes.
struct ClusterScore {
    score: f32,
    distance: f32,
    // 0 or 1 — leaf-level frustum.contains_sphere result.
    passes_frustum: u32,
    // Distance-driven LOD ceiling (0..=3).
    distance_lod: u32,
};

@group(0) @binding(0) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(1) var<storage, read> planes: array<Plane>;   // exactly 6
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> out_scores: array<ClusterScore>;

// Mirror of `hierarchical_lod::select_lod_level`. LOD_DISTANCES = [0,50,150,400].
//   screen < 10 || d > 400 -> 3
//   screen < 50 || d > 150 -> 2
//   screen < 200 || d > 50 -> 1
//   else                   -> 0
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

// Mirror of `Frustum::contains_sphere`: cull (return false / 0) iff for any
// plane `dot(normal, centre) + d < -radius`. Planes are pre-normalized.
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cluster_count) {
        return;
    }
    let c = clusters[i];
    let aabb_min = c.aabb_min_op.xyz;
    let total_opacity = c.aabb_min_op.w;
    let aabb_max = c.aabb_max.xyz;
    let center = c.center.xyz;
    let eye = params.eye.xyz;

    // radius = ((aabb_max - aabb_min) * 0.5).length().max(1e-4)
    let half = (aabb_max - aabb_min) * 0.5;
    let radius = max(length(half), 1e-4);

    // d = (center - eye).length().max(1e-3)   [scoring uses the CENTROID]
    let d = max(length(center - eye), 1e-3);

    // screen = 1000.0 * radius / d
    let screen = 1000.0 * radius / d;

    let distance_lod = select_lod_level(d, screen);

    // score = total_opacity * (radius * radius) / (d * d)
    let score = total_opacity * (radius * radius) / (d * d);

    // Leaf-level frustum test uses the AABB MIDPOINT and the same radius — this
    // matches the oracle's BVH leaf accept in `collect_visible`.
    let aabb_mid = (aabb_min + aabb_max) * 0.5;
    let passes = contains_sphere(aabb_mid, radius);

    out_scores[i] = ClusterScore(score, d, passes, distance_lod);
}
