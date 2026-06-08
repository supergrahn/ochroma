// tile_assign.wgsl
// Compute shader: projects Gaussian splats to screen space, computes 2D EWA
// conic coefficients via the Zwicker 2002 Jacobian projection, and emits
// (tile_index, depth_bits) key / splat_index value pairs for every tile that
// the splat's 3-sigma screen-space bounding box overlaps.

const TILE_SIZE: u32 = 16u;
const MAX_TILES_PER_SPLAT: u32 = 16u;

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view:      mat4x4<f32>,
    // viewport_size.xy = width, height in pixels
    // viewport_size.zw = 1/width, 1/height
    viewport_size: vec4<f32>,
    // tiles_x = ceil(width / TILE_SIZE), tiles_y = ceil(height / TILE_SIZE)
    tiles_xy: vec2<u32>,
    // Total number of splats in the current draw.
    splat_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// ── Storage buffers ───────────────────────────────────────────────────────────

struct GpuSplatFull {
    // xyz = world position, w = view-space depth (written here)
    position_depth: vec4<f32>,
    // 2D EWA conic coefficients (written here); _pad unused
    conic: vec3<f32>,
    _pad0: f32,
    // w = opacity [0..1], xyz reserved
    opacity_color: vec4<f32>,
    // 8 spectral bands
    spectral0: vec4<f32>,
    spectral1: vec4<f32>,
}

@group(0) @binding(1) var<storage, read_write> splats: array<GpuSplatFull>;

// Low 32 bits of the packed u64 sort key (reinterpreted depth bits).
@group(0) @binding(2) var<storage, read_write> tile_keys_lo: array<u32>;
// High 32 bits of the packed u64 sort key (tile index).
@group(0) @binding(3) var<storage, read_write> tile_keys_hi: array<u32>;
// Splat index for each emitted tile entry.
@group(0) @binding(4) var<storage, read_write> tile_vals:    array<u32>;
// Atomic counter of total emitted entries.
@group(0) @binding(5) var<storage, read_write> tile_count:   atomic<u32>;

// Per-splat transform: two vec4<f32> packed as:
//   transforms[2*i+0] = vec4(scale.x, scale.y, scale.z, 0)
//   transforms[2*i+1] = vec4(quat.x,  quat.y,  quat.z,  quat.w)
@group(0) @binding(6) var<storage, read>       splat_transforms: array<vec4<f32>>;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build rotation matrix from a unit quaternion (xyzw layout).
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y + z*z),       2.0*(x*y + z*w),       2.0*(x*z - y*w)),
        vec3<f32>(      2.0*(x*y - z*w), 1.0 - 2.0*(x*x + z*z),       2.0*(y*z + x*w)),
        vec3<f32>(      2.0*(x*z + y*w),       2.0*(y*z - x*w), 1.0 - 2.0*(x*x + y*y)),
    );
}

/// Build the 3×3 world-space covariance Σ_world from scale and quaternion.
/// Σ = R · S · S^T · R^T   (S = diag(scale))
fn build_cov3(scale: vec3<f32>, quat: vec4<f32>) -> mat3x3<f32> {
    let R = quat_to_mat3(quat);
    let s0 = scale.x; let s1 = scale.y; let s2 = scale.z;
    // M = R * S
    let col0 = R[0] * s0;
    let col1 = R[1] * s1;
    let col2 = R[2] * s2;
    // Σ = M * M^T
    return mat3x3<f32>(
        vec3<f32>(dot(col0, col0), dot(col0, col1), dot(col0, col2)),
        vec3<f32>(dot(col1, col0), dot(col1, col1), dot(col1, col2)),
        vec3<f32>(dot(col2, col0), dot(col2, col1), dot(col2, col2)),
    );
}

/// Zwicker 2002 Jacobian for the local affine approximation of the perspective
/// projection.  Returns the 2×3 Jacobian J such that
///   Σ_2D = J · Σ_world · J^T
///
/// `p_view` is the view-space position.
/// `focal` = (viewport_width/2, viewport_height/2).
fn ewa_jacobian(p_view: vec3<f32>, focal: vec2<f32>) -> mat2x3<f32> {
    let iz = 1.0 / p_view.z;
    let iz2 = iz * iz;
    return mat2x3<f32>(
        vec3<f32>(focal.x * iz,         0.0, -focal.x * p_view.x * iz2),
        vec3<f32>(0.0,         focal.y * iz, -focal.y * p_view.y * iz2),
    );
}

/// Project world-space covariance to 2D screen-space covariance and return
/// the EWA conic coefficients [a, b, c] of the ellipse
///   a*(dx)^2 + 2*b*(dx*dy) + c*(dy)^2 = 1
/// from Σ_2D^{-1}.
fn project_cov3_to_conic(cov3: mat3x3<f32>, p_view: vec3<f32>, focal: vec2<f32>) -> vec3<f32> {
    let J = ewa_jacobian(p_view, focal);

    // W = view rotation part (upper-left 3x3 of view matrix).
    // We need Σ_view = W * Σ_world * W^T before applying J.
    // For simplicity we embed W in the Jacobian via the view matrix rows.
    // Here we use J directly on Σ_world (common practical approximation).
    // Σ_2D = J * Σ_world * J^T
    //
    // J is a WGSL `mat2x3` — 2 columns, each a vec3 — and those columns are the
    // ROWS of the 2×3 math Jacobian, i.e. exactly the COLUMNS of J^T. So
    // Σ_world · J^T has columns `cov3 * J[i]` (mat3x3 * vec3 → vec3). The previous
    // `transpose(J)` produced a `mat3x2` whose columns are vec2, making
    // `cov3 * JT[i]` a dimension-invalid `mat3x3 * vec2` that failed naga
    // validation (latent: no GPU path drove this shader until the tiled renderer).
    let SJT_col0 = cov3 * J[0]; // Σ_world * (J^T col 0) → vec3
    let SJT_col1 = cov3 * J[1]; // Σ_world * (J^T col 1) → vec3

    // J * (Σ_world * J^T)  → 2×2
    let cov2_00 = dot(J[0], SJT_col0);
    let cov2_01 = dot(J[0], SJT_col1);
    let cov2_11 = dot(J[1], SJT_col1);

    // Add a small low-pass filter (0.3 px^2) for numerical stability.
    let c00 = cov2_00 + 0.3;
    let c01 = cov2_01;
    let c11 = cov2_11 + 0.3;

    // Inverse of 2×2 symmetric matrix: [a b; b c]^-1 = 1/det * [c -b; -b a]
    let det = c00 * c11 - c01 * c01;
    if abs(det) < 1e-8 { return vec3<f32>(0.0, 0.0, 0.0); }
    let inv_det = 1.0 / det;
    return vec3<f32>(c11 * inv_det, -c01 * inv_det, c00 * inv_det);
}

/// Reinterpret a float's bit pattern as u32 (for depth sorting).
fn float_to_bits(f: f32) -> u32 {
    return bitcast<u32>(f);
}

// ── Entry point ───────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn tile_assign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_idx = gid.x;
    if splat_idx >= camera.splat_count { return; }

    let world_pos = splats[splat_idx].position_depth.xyz;

    // ── View-space position ───────────────────────────────────────────────────
    let view_pos4 = camera.view * vec4<f32>(world_pos, 1.0);
    let p_view = view_pos4.xyz;

    // Cull splats behind the near plane.
    if p_view.z >= -1e-4 { return; }

    // ── Clip / NDC / pixel projection ────────────────────────────────────────
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0 { return; }
    let ndc = clip.xyz / clip.w;
    // Cull outside the view frustum (with a small slack for partially visible splats).
    if ndc.x < -1.4 || ndc.x > 1.4 || ndc.y < -1.4 || ndc.y > 1.4 { return; }

    let vw = camera.viewport_size.x;
    let vh = camera.viewport_size.y;
    // NDC (-1..1) → pixel centre
    let px = (ndc.x * 0.5 + 0.5) * vw;
    let py = (1.0 - (ndc.y * 0.5 + 0.5)) * vh;   // flip Y: NDC +Y = up, pixel +Y = down

    // ── 3D covariance → 2D conic ──────────────────────────────────────────────
    let t_base = splat_idx * 2u;
    let scale_raw = splat_transforms[t_base].xyz;
    let quat_raw  = splat_transforms[t_base + 1u];

    let focal = vec2<f32>(vw * 0.5, vh * 0.5);
    let cov3  = build_cov3(scale_raw, quat_raw);
    let conic = project_cov3_to_conic(cov3, p_view, focal);

    // Write depth and conic back to the splat buffer.
    splats[splat_idx].position_depth.w = -p_view.z;   // positive view-space depth
    splats[splat_idx].conic = conic;

    // ── 3-sigma screen bounding box ───────────────────────────────────────────
    // Largest eigenvalue of Σ_2D (approximate via conic inverse).
    // conic = Σ_2D^{-1} [a, b, c] → Σ_2D [a', b', c'] via the same formula.
    let ca = conic.x; let cb = conic.y; let cc = conic.z;
    let det_conic = ca * cc - cb * cb;
    if abs(det_conic) < 1e-10 { return; }
    let inv_det = 1.0 / det_conic;
    let s2d_a = cc * inv_det;
    let s2d_c = ca * inv_det;
    // Eigenvalues of [[a', b'], [b', c']]:
    let trace_half = (s2d_a + s2d_c) * 0.5;
    let diff_half  = (s2d_a - s2d_c) * 0.5;
    let lambda_max = trace_half + sqrt(max(0.0, diff_half * diff_half + cb * cb * inv_det * inv_det));
    let sigma_max  = sqrt(max(0.0, lambda_max));

    let radius = 3.0 * sigma_max;
    if radius < 0.5 { return; }     // sub-pixel splat, skip

    let bb_x0 = max(0.0,  px - radius);
    let bb_y0 = max(0.0,  py - radius);
    let bb_x1 = min(vw,   px + radius);
    let bb_y1 = min(vh,   py + radius);

    // Tile range covered by bbox.
    let tx0 = u32(bb_x0) / TILE_SIZE;
    let ty0 = u32(bb_y0) / TILE_SIZE;
    let tx1 = min(camera.tiles_xy.x - 1u, u32(bb_x1) / TILE_SIZE);
    let ty1 = min(camera.tiles_xy.y - 1u, u32(bb_y1) / TILE_SIZE);

    let depth_bits = float_to_bits(-p_view.z);

    // ── Emit tile entries (up to MAX_TILES_PER_SPLAT) ─────────────────────────
    var emitted: u32 = 0u;
    var ty: u32 = ty0;
    loop {
        if ty > ty1 || emitted >= MAX_TILES_PER_SPLAT { break; }
        var tx: u32 = tx0;
        loop {
            if tx > tx1 || emitted >= MAX_TILES_PER_SPLAT { break; }
            let tile_index = ty * camera.tiles_xy.x + tx;
            let slot = atomicAdd(&tile_count, 1u);
            tile_keys_hi[slot] = tile_index;
            tile_keys_lo[slot] = depth_bits;
            tile_vals[slot]    = splat_idx;
            emitted += 1u;
            tx += 1u;
        }
        ty += 1u;
    }
}
