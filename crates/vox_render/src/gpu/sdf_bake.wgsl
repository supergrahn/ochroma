// GPU mesh→SDF cook — SDF-pillar design §4.3.
//
// Port-and-upgrade of spectra's physics_sdf_gen.slang three-pass structure,
// generalized from cube-only u_resolution to anisotropic [nx, ny, nz], with
// the parity sign replaced by the generalized winding number:
//
//   init          per voxel      dist² = f32::MAX bits, seed invalid
//   seed_min      per triangle   atomicMin(ordered f32 bits of d²) into the
//                                triangle's ±2-voxel AABB (Ericson closest pt)
//   seed_payload  per triangle   equality-guarded closest-point write — the
//                                benign race among equal distances the Slang
//                                source documents
//   jfa           per voxel      26-neighbor jump flood carrying closest-point
//                                xyz; distance re-derived from the carried
//                                point, so in-band voxels end exact
//   sign_resolve  per voxel      f32 solid-angle sum over all triangles
//                                (Van Oosterom–Strackee); |w| ≥ 0.5 ⇒ negate
//   encode        per 4 voxels   snorm8 saturating at ±band, packed into u32
//
// Storage atomics are u32/i32 only in WGSL — the scatter-min therefore runs
// on bitcast<u32>(d²), which is order-preserving for all non-negative f32.

struct Params {
    res: vec3<u32>,
    num_tris: u32,
    origin: vec3<f32>,
    voxel: f32,
    band: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct StepParams {
    step: i32,
    _a: i32,
    _b: i32,
    _c: i32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> dist_bits: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> signed_dist: array<f32>;
@group(0) @binding(5) var<storage, read_write> snorm_out: array<u32>;
@group(0) @binding(6) var<uniform> S: StepParams;

// JFA ping-pong: xyz = carried closest point, w = validity (1 valid, -1 not).
@group(1) @binding(0) var<storage, read> seed_src: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> seed_dst: array<vec4<f32>>;

const DIST_INIT_BITS: u32 = 0x7F7FFFFFu; // f32::MAX — max ordered bits for d² ≥ 0
const LARGE: f32 = 1e20;
const FOUR_PI: f32 = 12.566370614359172;

fn voxel_count() -> u32 {
    return P.res.x * P.res.y * P.res.z;
}

fn voxel_index(v: vec3<i32>) -> u32 {
    return (u32(v.z) * P.res.y + u32(v.y)) * P.res.x + u32(v.x);
}

fn voxel_pos(v: vec3<i32>) -> vec3<f32> {
    return P.origin + vec3<f32>(v) * P.voxel;
}

fn unflatten(i: u32) -> vec3<i32> {
    let x = i % P.res.x;
    let y = (i / P.res.x) % P.res.y;
    let z = i / (P.res.x * P.res.y);
    return vec3<i32>(i32(x), i32(y), i32(z));
}

fn vert(i: u32) -> vec3<f32> {
    return vec3<f32>(positions[3u * i], positions[3u * i + 1u], positions[3u * i + 2u]);
}

// Closest point on triangle abc to p — Ericson, RTCD §5.1.5 (full Voronoi
// region version: exact on vertices, edges and face, no epsilon division).
fn closest_point_on_tri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= voxel_count()) {
        return;
    }
    atomicStore(&dist_bits[i], DIST_INIT_BITS);
    seed_dst[i] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

// Shared per-triangle voxel sweep. write_payload=false: scatter-min the d²
// bits. write_payload=true: `<=`-guarded payload write (the second pass) —
// `<=` rather than `==` so a last-ulp recompute difference can only admit a
// payload within one ulp of the true min, never drop it.
fn seed_triangle(fi: u32, write_payload: bool) {
    let ia = indices[3u * fi];
    let ib = indices[3u * fi + 1u];
    let ic = indices[3u * fi + 2u];
    let a = vert(ia);
    let b = vert(ib);
    let c = vert(ic);

    let inv = 1.0 / P.voxel;
    let tri_min = min(a, min(b, c));
    let tri_max = max(a, max(b, c));
    var vmin = vec3<i32>(floor((tri_min - P.origin) * inv)) - vec3<i32>(2);
    var vmax = vec3<i32>(floor((tri_max - P.origin) * inv)) + vec3<i32>(3);
    vmin = max(vmin, vec3<i32>(0));
    vmax = min(vmax, vec3<i32>(P.res)); // exclusive upper bound

    for (var vz = vmin.z; vz < vmax.z; vz++) {
        for (var vy = vmin.y; vy < vmax.y; vy++) {
            for (var vx = vmin.x; vx < vmax.x; vx++) {
                let v = vec3<i32>(vx, vy, vz);
                let p = voxel_pos(v);
                let cp = closest_point_on_tri(p, a, b, c);
                let diff = p - cp;
                let d2 = dot(diff, diff);
                let idx = voxel_index(v);
                if (write_payload) {
                    if (bitcast<u32>(d2) <= atomicLoad(&dist_bits[idx])) {
                        seed_dst[idx] = vec4<f32>(cp, 1.0);
                    }
                } else {
                    atomicMin(&dist_bits[idx], bitcast<u32>(d2));
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn seed_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.num_tris) {
        return;
    }
    seed_triangle(gid.x, false);
}

@compute @workgroup_size(256)
fn seed_payload(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.num_tris) {
        return;
    }
    seed_triangle(gid.x, true);
}

@compute @workgroup_size(256)
fn jfa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= voxel_count()) {
        return;
    }
    let v = unflatten(i);
    let p = voxel_pos(v);

    var best = seed_src[i];
    var best_d2 = LARGE;
    if (best.w > 0.0) {
        let diff = p - best.xyz;
        best_d2 = dot(diff, diff);
    }

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                let n = v + vec3<i32>(dx, dy, dz) * S.step;
                if (any(n < vec3<i32>(0)) || any(n >= vec3<i32>(P.res))) {
                    continue;
                }
                let s = seed_src[voxel_index(n)];
                if (s.w <= 0.0) {
                    continue;
                }
                let diff = p - s.xyz;
                let d2 = dot(diff, diff);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best = s;
                }
            }
        }
    }
    seed_dst[i] = best;
}

// Signed solid angle of the triangle (a, b, c) as seen from the origin —
// Van Oosterom & Strackee. The (0,0)-guard covers a query point coincident
// with a vertex (degenerate; contributes nothing).
fn solid_angle(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let la = length(a);
    let lb = length(b);
    let lc = length(c);
    let num = dot(a, cross(b, c));
    let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    if (num == 0.0 && den == 0.0) {
        return 0.0;
    }
    return 2.0 * atan2(num, den);
}

@compute @workgroup_size(256)
fn sign_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= voxel_count()) {
        return;
    }
    let v = unflatten(i);
    let p = voxel_pos(v);

    let s = seed_src[i];
    var d = P.band; // no carried point (empty seed set) ⇒ saturate positive
    if (s.w > 0.0) {
        d = distance(p, s.xyz);
    }

    var total = 0.0;
    for (var t = 0u; t < P.num_tris; t++) {
        let a = vert(indices[3u * t]) - p;
        let b = vert(indices[3u * t + 1u]) - p;
        let c = vert(indices[3u * t + 2u]) - p;
        total += solid_angle(a, b, c);
    }
    let w = total / FOUR_PI;
    if (abs(w) >= 0.5) {
        d = -d;
    }
    signed_dist[i] = d;
}

@compute @workgroup_size(256)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let n = voxel_count();
    let words = (n + 3u) / 4u;
    if (o >= words) {
        return;
    }
    var word = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = 4u * o + j;
        var q: i32 = 0;
        if (i < n) {
            let nd = clamp(signed_dist[i] / P.band, -1.0, 1.0);
            q = i32(round(nd * 127.0));
        }
        word |= (u32(q) & 0xFFu) << (8u * j);
    }
    snorm_out[o] = word;
}
