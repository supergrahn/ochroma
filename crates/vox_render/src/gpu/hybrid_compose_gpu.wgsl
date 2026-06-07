// Hybrid mesh + Gaussian-splat compositor — GPU compute mirror of the CPU
// oracle in `hybrid_compose.rs` (`render_hybrid_lit` -> `rasterise_meshes` then
// `composite_splats`). ONE THREAD PER PIXEL, TWO DISPATCHES. Bit-equivalence
// target: the CPU `f32` rasterizer/compositor reproduced here with IDENTICAL
// operation order.
//
// =============================  PASS STRUCTURE  ==============================
// The CPU oracle runs two ordered passes sharing one depth buffer:
//   1. rasterise_meshes : nearest-wins depth + shaded-spectral write.
//   2. composite_splats : front-to-back over-composite, each splat fragment
//      depth-tested against the *mesh* depth from pass 1, then resolved
//      `out = accum + transmittance * mesh_background`.
// We mirror this as TWO SEPARATE COMPUTE DISPATCHES over the same storage
// buffers. Two dispatches (not one kernel) because pass 2 reads the COMPLETE
// mesh depth/spectral of pass 1 — the CPU finishes ALL of pass 1 before ANY of
// pass 2. A single kernel cannot guarantee that global ordering across pixels
// without a barrier we don't need; two dispatches give it for free and map
// 1:1 onto the oracle's `render_hybrid_lit` call order.
//
// HOST-SIDE PREP (identical-input principle, like the splat_rt_gpu raw upload):
//   * Triangles are clipped (Sutherland-Hodgman near+far) and fanned to screen-
//     space sub-triangles ON THE HOST in Rust, using code byte-identical to the
//     oracle's `clip_triangle_near` / `clip_polygon_far` / `project_cam` /
//     `edge`. We upload post-clip `MeshTri` (3 screen verts + shaded spectral).
//   * Splats are projected via the oracle's OWN `project_gaussian` ON THE HOST,
//     filtered + sorted front-to-back exactly as `composite_splats` does, and
//     uploaded as pre-projected `SplatRec`. The WGSL never reprojects — so the
//     EWA covariance / conic / radius math cannot diverge by a single ULP.
// The GPU therefore reproduces ONLY the per-pixel rasterization + compositing
// inner loops, which is where a GPU port earns its keep.
// =============================================================================

const BANDS: u32 = 16u;
const ALPHA_THRESHOLD: f32 = 0.003921569;       // 1.0 / 255.0
const TRANSMITTANCE_THRESHOLD: f32 = 0.001;
const F32_MAX: f32 = 3.4028235e38;

struct Params {
    px_w: u32,
    px_h: u32,
    tri_count: u32,
    splat_count: u32,
};

// One post-clip screen-space triangle. a/b/c are (screen_x, screen_y, cam_z);
// `inv_area2` is the host-computed 1/edge(a,b,c) (the CPU's `inv_area2`).
// `spectral` is the already-shaded 16-band reflectance for the whole tri.
struct MeshTri {
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    inv_area2: f32,
    _p0: f32, _p1: f32,
    spectral: array<vec4<f32>, 4>,  // 16 bands packed as 4 vec4
};

// One pre-projected splat record (mirrors the CPU `SplatRecord`), in the SAME
// front-to-back (ascending depth) order the CPU sorts into.
struct SplatRec {
    cx: f32, cy: f32,            // screen_pos
    k0: f32, k1: f32, k2: f32,  // conic
    radius: f32,
    depth: f32,
    opacity: f32,
    spectral: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform>             params:    Params;
@group(0) @binding(1) var<storage, read>       tris:      array<MeshTri>;
@group(0) @binding(2) var<storage, read>       splats:    array<SplatRec>;
// Shared mesh G-buffer: depth (cam_z, nearest-wins) + 16-band shaded spectral.
@group(0) @binding(3) var<storage, read_write> mesh_depth:    array<f32>;
@group(0) @binding(4) var<storage, read_write> mesh_spectral: array<f32>; // px*16
// Final outputs: composited 16-band spectral + resolved depth.
@group(0) @binding(5) var<storage, read_write> out_spectral:  array<f32>; // px*16
@group(0) @binding(6) var<storage, read_write> out_depth:     array<f32>;

// Edge function (signed area x2) for segment (a,b) and point p — the CPU `edge`,
// SAME operation order: `(b.0-a.0)*(py-a.1) - (b.1-a.1)*(px-a.0)`.
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Inside-test tolerance on the NORMALIZED barycentric. We accept `w >= -BARY_EPS`.
//
// WHAT GOES WRONG WITHOUT IT. The CPU oracle's separate-op f32 edge evaluates a
// shared-diagonal pixel to EXACTLY 0 (its `edge` returns `-0.0`, and `-0.0 >= 0.0`
// so the CPU fills it). RADV FMA-contracts `a*b - c*d`, so the GPU rounds that
// same exact-zero edge to a tiny NEGATIVE and the strict `w < 0` test rejects it
// on BOTH adjacent triangles — a 1px hole along every shared diagonal.
//
// WHY NOT THE OLD 1e-4. A fixed `1e-4` fills the hole but DILATES every
// silhouette: on a thin/high-aspect triangle the per-pixel barycentric step on
// the slim edges shrinks toward 1e-4, so 1e-4 of slack approaches a whole pixel
// and the GPU grabs a strictly-outside pixel the CPU rejects (probe-proven 0.579
// full-pixel spectral divergence).
//
// DERIVATION OF BARY_EPS (measured on RADV PHOENIX via `diag_bary_magnitudes`):
//   * shared-diagonal hole: the FMA rounding of the exact-zero edge is
//     -2.13e-8 normalized (worst |sep-fma| over the diagonal = 1.19e-7).
//   * thin-sliver over-coverage: the CPU's NEAREST strictly-outside pixel sits at
//     -1.13e-3 normalized (its min weight); anything the CPU rejects on that
//     sliver is at least this far below zero.
// So the divergent band is roughly [-1.2e-7, 0) (must accept) and the
// must-reject band starts near -1.1e-3 — ~four orders of magnitude apart. We pick
// BARY_EPS = 1e-5: ~80x ABOVE the measured FMA hole-rounding (and above the
// finding's conservative ~1e-5 hole estimate) so every diagonal pixel is filled,
// yet ~100x BELOW the thin-sliver reject threshold so no genuinely-outside pixel
// is ever grabbed, on any aspect ratio exercised by the suite.
const BARY_EPS: f32 = 1e-5;

fn band_of(s: array<vec4<f32>, 4>, b: u32) -> f32 {
    let v = s[b >> 2u];
    let lane = b & 3u;
    if lane == 0u { return v.x; }
    if lane == 1u { return v.y; }
    if lane == 2u { return v.z; }
    return v.w;
}

// -- Pass 1: mesh rasterization. One thread per pixel, brute force over every
// post-clip screen-space triangle, mirroring `rasterise_meshes`' inner loop. --
@compute @workgroup_size(8, 8, 1)
fn mesh_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= params.px_w || py >= params.px_h { return; }

    let pidx = py * params.px_w + px;
    let pxf = f32(px) + 0.5;
    let pyf = f32(py) + 0.5;

    var best_depth: f32 = F32_MAX;       // matches fb.depth init (f32::MAX)
    var best_spec: array<vec4<f32>, 4>;
    var hit: bool = false;

    let n = params.tri_count;
    for (var t: u32 = 0u; t < n; t = t + 1u) {
        let tri = tris[t];
        // The CPU bbox-clamps before the inside test; an out-of-bbox pixel never
        // enters the loop body. We replicate the inside test (w>=0), which is a
        // superset of the bbox — identical accept/reject per pixel.
        let w0 = edge_fn(tri.bx, tri.by, tri.cx, tri.cy, pxf, pyf) * tri.inv_area2;
        let w1 = edge_fn(tri.cx, tri.cy, tri.ax, tri.ay, pxf, pyf) * tri.inv_area2;
        let w2 = edge_fn(tri.ax, tri.ay, tri.bx, tri.by, pxf, pyf) * tri.inv_area2;
        // Inside test with the derived BARY_EPS: accept `w >= -BARY_EPS` to fill
        // the shared-diagonal exact-zero edge (which RADV's FMA rounds slightly
        // negative) WITHOUT dilating thin silhouettes. Mirrors the CPU's
        // `w < 0.0` reject up to the measured FMA rounding only.
        if w0 < -BARY_EPS || w1 < -BARY_EPS || w2 < -BARY_EPS { continue; }

        // Perspective-correct depth: interpolate INVERSE depth, then invert.
        let inv_z = w0 / tri.az + w1 / tri.bz + w2 / tri.cz;
        if inv_z <= 0.0 { continue; }
        let depth = 1.0 / inv_z;
        if depth >= best_depth { continue; }   // nearest wins (z-test)
        best_depth = depth;
        best_spec = tri.spectral;
        hit = true;
    }

    mesh_depth[pidx] = best_depth;
    let base = pidx * BANDS;
    if hit {
        for (var b: u32 = 0u; b < BANDS; b = b + 1u) {
            mesh_spectral[base + b] = band_of(best_spec, b);
        }
    } else {
        for (var b: u32 = 0u; b < BANDS; b = b + 1u) {
            mesh_spectral[base + b] = 0.0;
        }
    }
}

// -- Pass 2: splat composite. One thread per pixel, front-to-back over the
// host-sorted splat records, depth-tested against the mesh depth, then resolved
// OVER the mesh background. Mirrors `composite_splats` exactly. -------------
@compute @workgroup_size(8, 8, 1)
fn splat_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= params.px_w || py >= params.px_h { return; }

    let pidx = py * params.px_w + px;
    let base = pidx * BANDS;
    let m_depth = mesh_depth[pidx];

    var accum: array<f32, 16>;
    for (var b: u32 = 0u; b < BANDS; b = b + 1u) { accum[b] = 0.0; }
    var transmittance: f32 = 1.0;
    var splat_depth: f32 = F32_MAX;     // CPU `splat_depth` init = f32::MAX

    let pxf = f32(px) + 0.5;
    let pyf = f32(py) + 0.5;

    let n = params.splat_count;
    for (var s: u32 = 0u; s < n; s = s + 1u) {
        if transmittance < TRANSMITTANCE_THRESHOLD { break; } // T monotone-decreasing
        let rec = splats[s];

        // CPU bbox-rejects a pixel outside [cx-r, cx+r] x [cy-r, cy+r]; the
        // Gaussian `power` test below subsumes coverage, but we keep the SAME
        // bbox gate so accept/reject and accumulation order match bit-for-bit.
        let y_min = floor(rec.cy - rec.radius);
        let y_max = ceil(rec.cy + rec.radius);
        let x_min = floor(rec.cx - rec.radius);
        let x_max = ceil(rec.cx + rec.radius);
        if f32(py) < y_min || f32(py) > y_max || f32(px) < x_min || f32(px) > x_max {
            continue;
        }

        let t = transmittance;
        if t < TRANSMITTANCE_THRESHOLD { continue; }
        // DEPTH-CORRECT OCCLUSION: reject a fragment behind the mesh depth.
        if rec.depth >= m_depth { continue; }

        let dx = pxf - rec.cx;
        let dy = pyf - rec.cy;
        let power = -0.5 * (rec.k0 * dx * dx + 2.0 * rec.k1 * dx * dy + rec.k2 * dy * dy);
        if power > 0.0 { continue; }
        let alpha = min(rec.opacity * exp(power), 0.99);
        if alpha < ALPHA_THRESHOLD { continue; }
        let weight = alpha * t;
        for (var b: u32 = 0u; b < BANDS; b = b + 1u) {
            accum[b] = accum[b] + weight * band_of(rec.spectral, b);
        }
        transmittance = t * (1.0 - alpha);
        if rec.depth < splat_depth { splat_depth = rec.depth; }
    }

    // Resolve: composite the accumulated splat spectrum OVER the mesh spectrum
    // using residual transmittance (mirrors `composite_splats`' resolve loop).
    let cov = 1.0 - transmittance;
    if cov <= 0.0 {
        // No splat contribution — mesh (or empty) stands.
        for (var b: u32 = 0u; b < BANDS; b = b + 1u) {
            out_spectral[base + b] = mesh_spectral[base + b];
        }
        out_depth[pidx] = m_depth;
        return;
    }
    for (var b: u32 = 0u; b < BANDS; b = b + 1u) {
        let bg = mesh_spectral[base + b];
        out_spectral[base + b] = accum[b] + transmittance * bg;
    }
    // A nearer covering splat updates the visible depth.
    if splat_depth < m_depth {
        out_depth[pidx] = splat_depth;
    } else {
        out_depth[pidx] = m_depth;
    }
}
