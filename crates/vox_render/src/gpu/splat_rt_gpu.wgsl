// Spectral splat ray tracer — GPU compute mirror of the CPU oracle in
// `splat_rt.rs` (`render_orthographic` / `trace_ray`, the `bvh = None`
// brute-force path). ONE THREAD PER PIXEL. Bit-equivalence target: the CPU
// `f32` math, reproduced here in `f32` with identical operation order.
//
// =====================  RAW GaussianSplat BYTE LAYOUT  =======================
// We upload the 96-byte `#[repr(C)] GaussianSplat` POD verbatim (bytemuck) and
// decode it in-shader. Viewed as 24 little-endian u32 words (`array<u32,24>`):
//
//   word  0..2  position : [f32;3]               (world centroid)
//   word  3     kind     : u32                    (0 = 2DGS surface, 1 = 3DGS)
//   word  4..6  tangent_u: [f32;3]
//   word  7     scale_u  : f32
//   word  8..10 tangent_v: [f32;3]
//   word 11     scale_v  : f32
//   word 12..13 rotation : [i16;4] quat XYZW /32767
//                 word12 = (x_i16 & 0xFFFF) | (y_i16 << 16)
//                 word13 = (z_i16 & 0xFFFF) | (w_i16 << 16)
//                 each i16 is SIGN-extended: i32(w << 16) >> 16 (low half),
//                 i32(w) >> 16 (high half) — arithmetic shift recovers sign.
//   word 14     scale_w  : f32
//   word 15     opacity u8 + _pad[3]  -> opacity = word15 & 0xFF
//   word 16..23 spectral : [u16;16] as f16, 2 per word.
//                 band 2*j   = unpack2x16float(word(16+j)).x
//                 band 2*j+1 = unpack2x16float(word(16+j)).y
// =============================================================================
//
// The per-thread hit gather is bounded to BUDGET (64 — the CPU's hard budget).
// A pixel-ray that pierces MORE than 64 splats within the 3σ cutoff keeps the
// 64 nearest by t_peak (the CPU composites the first 64 in sorted order, which
// is exactly the 64 nearest); scenes with >64 hits/ray therefore differ from an
// unbounded gather, EXACTLY mirroring the CPU `budget` semantics.

const BANDS: u32 = 16u;
const BUDGET: u32 = 64u;
const SIGMA_CUTOFF: f32 = 3.0;
const POWER_CUTOFF: f32 = 4.5;            // 0.5 * 3^2
const TRANSMITTANCE_THRESHOLD: f32 = 0.001;
const ALPHA_THRESHOLD: f32 = 0.003921569; // 1.0 / 255.0
const SCALE_FLOOR: f32 = 1e-4;

struct Camera {
    eye:     vec3<f32>,
    _p0:     f32,
    forward: vec3<f32>,
    _p1:     f32,
    right:   vec3<f32>,
    _p2:     f32,
    up:      vec3<f32>,
    _p3:     f32,
    width:   f32,
    height:  f32,
    px_w:    u32,
    px_h:    u32,
    splat_count: u32,
    _p4: u32,
    _p5: u32,
    _p6: u32,
};

// One raw splat = 24 u32 words. WGSL has no u16/i16; we decode from words.
struct RawSplat {
    w: array<u32, 24>,
};

@group(0) @binding(0) var<storage, read>       splats: array<RawSplat>;
@group(0) @binding(1) var<uniform>             cam:    Camera;
// Output: px_w*px_h pixels, each 17 floats (16 bands + alpha), row-major.
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;

fn splat_position(i: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(splats[i].w[0]),
        bitcast<f32>(splats[i].w[1]),
        bitcast<f32>(splats[i].w[2]),
    );
}
fn splat_kind(i: u32) -> u32 { return splats[i].w[3]; }
fn splat_tangent_u(i: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(splats[i].w[4]), bitcast<f32>(splats[i].w[5]), bitcast<f32>(splats[i].w[6]));
}
fn splat_scale_u(i: u32) -> f32 { return bitcast<f32>(splats[i].w[7]); }
fn splat_tangent_v(i: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(splats[i].w[8]), bitcast<f32>(splats[i].w[9]), bitcast<f32>(splats[i].w[10]));
}
fn splat_scale_v(i: u32) -> f32 { return bitcast<f32>(splats[i].w[11]); }
fn splat_scale_w(i: u32) -> f32 { return bitcast<f32>(splats[i].w[14]); }
fn splat_opacity(i: u32) -> f32 { return f32(splats[i].w[15] & 0xFFu) / 255.0; }

// Decode the i16 quaternion XYZW. word12 = (x | y<<16), word13 = (z | w<<16).
// Arithmetic right shift on i32 sign-extends the 16-bit halves.
fn splat_quat(i: u32) -> vec4<f32> {
    let lo = splats[i].w[12];
    let hi = splats[i].w[13];
    let xi = (i32(lo) << 16u) >> 16u;   // low  16 bits, sign-extended
    let yi = i32(lo) >> 16u;            // high 16 bits, sign-extended
    let zi = (i32(hi) << 16u) >> 16u;
    let wi = i32(hi) >> 16u;
    return vec4<f32>(f32(xi), f32(yi), f32(zi), f32(wi)) / 32767.0;
}

fn splat_band(i: u32, b: u32) -> f32 {
    let word = splats[i].w[16u + (b >> 1u)];
    let pair = unpack2x16float(word);
    if (b & 1u) == 0u { return pair.x; }
    return pair.y;
}

// glam Quat::normalize then Mat3::from_quat. glam normalizes by 1/length.
// Mat3::from_quat for a UNIT quaternion (x,y,z,w):
//   col0 = (1-2(y²+z²),   2(x y + w z),   2(x z - w y))
//   col1 = (2(x y - w z),  1-2(x²+z²),    2(y z + w x))
//   col2 = (2(x z + w y),  2(y z - w x),  1-2(x²+y²))
// We build it as 3 columns (matching glam's column-major Mat3).
fn quat_to_mat3(q_in: vec4<f32>) -> mat3x3<f32> {
    let len = length(q_in);
    let q = q_in / len;            // glam normalize
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    let col0 = vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy);
    let col1 = vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx);
    let col2 = vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy));
    return mat3x3<f32>(col0, col1, col2);
}

// Mirror of `finish_hit`: returns alpha (>0) on a hit, or -1.0 on miss.
fn finish_hit(d2: f32, opacity: f32) -> f32 {
    let power = 0.5 * d2;
    if power > POWER_CUTOFF { return -1.0; }
    let alpha = min(opacity * exp(-power), 0.99);
    if alpha < ALPHA_THRESHOLD { return -1.0; }
    return alpha;
}

// Per-splat ray/Gaussian peak. Returns vec2(t_peak, alpha); alpha < 0 == miss.
fn ray_gaussian_hit(origin: vec3<f32>, dir: vec3<f32>, i: u32) -> vec2<f32> {
    let opacity = splat_opacity(i);
    let center = splat_position(i);

    if splat_kind(i) == 0u {
        // ---- 2DGS surface disk ----
        let u = splat_tangent_u(i);
        let v = splat_tangent_v(i);
        let nrm = cross(u, v);
        let n_len = length(nrm);
        if n_len < 1e-8 { return vec2<f32>(0.0, -1.0); }
        let normal = nrm / n_len;
        let denom = dot(dir, normal);
        if abs(denom) < 1e-8 { return vec2<f32>(0.0, -1.0); }
        let t = dot(center - origin, normal) / denom;
        if t <= 0.0 { return vec2<f32>(0.0, -1.0); }
        let hit = origin + dir * t;
        let rel = hit - center;
        let su = max(sqrt(dot(u, u)), SCALE_FLOOR);
        let sv = max(sqrt(dot(v, v)), SCALE_FLOOR);
        let ru = max(splat_scale_u(i), SCALE_FLOOR);
        let rv = max(splat_scale_v(i), SCALE_FLOOR);
        let du = dot(rel, u / su) / ru;
        let dv = dot(rel, v / sv) / rv;
        let d2 = du * du + dv * dv;
        let alpha = finish_hit(d2, opacity);
        return vec2<f32>(t, alpha);
    } else {
        // ---- 3DGS ellipsoid: transform ray into unit-sphere space ----
        let r_mat = quat_to_mat3(splat_quat(i));
        let inv_s = vec3<f32>(
            1.0 / max(splat_scale_u(i), SCALE_FLOOR),
            1.0 / max(splat_scale_v(i), SCALE_FLOOR),
            1.0 / max(splat_scale_w(i), SCALE_FLOOR),
        );
        // World->local is Rᵀ. glam `r_t * v` with column-major Mat3 transpose:
        // (Rᵀ v).k = dot(column_k(R), v). transpose() then multiply == this.
        let r_t = transpose(r_mat);
        let o_local = (r_t * (origin - center)) * inv_s;
        let d_local = (r_t * dir) * inv_s;
        let dd = dot(d_local, d_local);
        if dd < 1e-12 { return vec2<f32>(0.0, -1.0); }
        let t_peak = -dot(o_local, d_local) / dd;
        if t_peak <= 0.0 { return vec2<f32>(0.0, -1.0); }
        let closest = o_local + d_local * t_peak;
        let d2 = dot(closest, closest);
        let alpha = finish_hit(d2, opacity);
        return vec2<f32>(t_peak, alpha);
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= cam.px_w || py >= cam.px_h { return; }

    // Orthographic ray construction (mirror render_orthographic). The host has
    // already normalized forward/right/up.
    let fwd = cam.forward;
    let u = ((f32(px) + 0.5) / f32(cam.px_w) - 0.5) * cam.width;
    let vv = ((f32(py) + 0.5) / f32(cam.px_h) - 0.5) * cam.height;
    let origin = cam.eye + cam.right * u + cam.up * vv;
    let dir = fwd; // already unit; matches dir.normalize_or_zero() of a unit fwd

    // Per-thread bounded hit gather, insertion-sorted by t_peak (tie-break by
    // splat index, matching the CPU's stable depth-keyed sort). We keep at most
    // BUDGET hits — the 64 nearest, which is what the CPU composites.
    var hit_t: array<f32, 64>;
    var hit_a: array<f32, 64>;
    var hit_i: array<u32, 64>;
    var count: u32 = 0u;

    let n = cam.splat_count;
    for (var s: u32 = 0u; s < n; s++) {
        let ta = ray_gaussian_hit(origin, dir, s);
        if ta.y < 0.0 { continue; } // miss
        let t = ta.x;
        let a = ta.y;

        // Find insertion position (sorted ascending by t, tie-break by index).
        // If the array is full and this hit is not nearer than the last kept
        // one, drop it (it would never be composited within BUDGET).
        if count == BUDGET {
            let lt = hit_t[BUDGET - 1u];
            let li = hit_i[BUDGET - 1u];
            if (t > lt) || (t == lt && s >= li) { continue; }
        }

        // Locate slot via linear scan from the back (insertion sort).
        var pos = min(count, BUDGET - 1u);
        // Shift larger elements right to make room.
        loop {
            if pos == 0u { break; }
            let pt = hit_t[pos - 1u];
            let pi = hit_i[pos - 1u];
            // element (pos-1) is "after" the new one if it sorts later.
            if (pt > t) || (pt == t && pi > s) {
                hit_t[pos] = pt;
                hit_a[pos] = hit_a[pos - 1u];
                hit_i[pos] = pi;
                pos = pos - 1u;
            } else {
                break;
            }
        }
        hit_t[pos] = t;
        hit_a[pos] = a;
        hit_i[pos] = s;
        if count < BUDGET { count = count + 1u; }
    }

    // Front-to-back over-composite (mirror trace_ray's compositing loop).
    var bands: array<f32, 16>;
    for (var b = 0u; b < BANDS; b++) { bands[b] = 0.0; }
    var transmittance: f32 = 1.0;

    for (var h: u32 = 0u; h < count; h++) {
        if h >= BUDGET { break; }
        if transmittance < TRANSMITTANCE_THRESHOLD { break; }
        let a = hit_a[h];
        let si = hit_i[h];
        let weight = a * transmittance;
        for (var b = 0u; b < BANDS; b++) {
            bands[b] = bands[b] + weight * splat_band(si, b);
        }
        transmittance = transmittance * (1.0 - a);
    }

    let base = (py * cam.px_w + px) * 17u;
    for (var b = 0u; b < BANDS; b++) {
        out[base + b] = bands[b];
    }
    out[base + 16u] = 1.0 - transmittance;
}
