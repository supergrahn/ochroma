//! Pure-math render-runtime helpers — deterministic hashing, RGB color, and
//! geometry utilities.
//!
//! Moved verbatim from the game's `render_gpu::buildings` / `render_gpu::materials`
//! / `render_gpu::mesh_utils` / `spectra_frame::helpers` so a new game never
//! re-implements these. No game-type dependencies (f32/u64/[f32; 3]/glam only).

use glam::{Mat4, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Deterministic hash helpers (from render_gpu::buildings)
// ---------------------------------------------------------------------------

/// splitmix64 mix — the repo's standard inline deterministic hash (no `rand`).
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One independent 64-bit channel of an instance seed, keyed by `salt`.
pub fn variation_bits(seed: u64, salt: u64) -> u64 {
    splitmix64(seed ^ splitmix64(salt))
}

/// Deterministic uniform in `[0, 1)` for `(seed, salt)`.
pub fn variation_unit(seed: u64, salt: u64) -> f32 {
    (variation_bits(seed, salt) >> 40) as f32 / (1u64 << 24) as f32
}

/// Deterministic uniform in `[-1, 1)` for `(seed, salt)`.
pub fn variation_signed(seed: u64, salt: u64) -> f32 {
    variation_unit(seed, salt) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Colour utilities (from render_gpu::materials)
// ---------------------------------------------------------------------------

/// Relative luminance of a linear RGB triple (Rec. 709 weights).
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

/// Linear interpolation between `a` and `b` by `t`.
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// Lift each channel toward white by `amount`, clamped into the display range.
pub fn lift_rgb(rgb: [f32; 3], amount: f32) -> [f32; 3] {
    [
        mix(rgb[0], 1.0, amount).clamp(0.02, 1.0),
        mix(rgb[1], 1.0, amount).clamp(0.02, 1.0),
        mix(rgb[2], 1.0, amount).clamp(0.02, 1.0),
    ]
}

/// Scale each channel by `scale`, clamped into the display range.
pub fn scale_rgb(rgb: [f32; 3], scale: f32) -> [f32; 3] {
    [
        (rgb[0] * scale).clamp(0.02, 1.0),
        (rgb[1] * scale).clamp(0.02, 1.0),
        (rgb[2] * scale).clamp(0.02, 1.0),
    ]
}

/// Tint a base colour toward a texture's local luminance detail (from
/// `render_gpu::materials`). `strength` blends between the flat base and the
/// per-texel detail lift.
pub fn texture_luma_tint_rgb(
    base: [f32; 3],
    texture_rgb: [f32; 3],
    texture_average: [f32; 3],
    strength: f32,
) -> [f32; 3] {
    let detail = (luminance(texture_rgb) / luminance(texture_average).max(0.02)).clamp(0.68, 1.38);
    let lift = mix(1.0, detail, strength.clamp(0.0, 1.0));
    [
        (base[0] * lift).clamp(0.02, 1.0),
        (base[1] * lift).clamp(0.02, 1.0),
        (base[2] * lift).clamp(0.02, 1.0),
    ]
}

/// Glass display tint: blends `base` toward a glass tint and adds a sky-sheen
/// term (from `render_gpu::materials`).
pub fn glass_display_rgb(base: [f32; 3], roughness: f32, upwardness: f32) -> [f32; 3] {
    let tint = [0.14, 0.68, 0.82];
    let sheen = (1.0 - roughness).clamp(0.0, 1.0);
    let sky = (0.18 + upwardness * 0.10 + sheen * 0.18).clamp(0.0, 0.42);
    [
        mix(base[0], tint[0], 0.72) + sky,
        mix(base[1], tint[1], 0.72) + sky,
        mix(base[2], tint[2], 0.72) + sky,
    ]
}

// ---------------------------------------------------------------------------
// Geometry math (from render_gpu::materials / render_gpu::mesh_utils)
// ---------------------------------------------------------------------------

/// Box-project a world-space `point` onto a planar UV using the dominant axis of
/// `normal` (from `render_gpu::materials`).
pub fn box_project_uv(point: Vec3, normal: Vec3) -> [f32; 2] {
    let n = normal.abs();
    let texel_scale = 0.42;
    if n.y >= n.x && n.y >= n.z {
        [point.x * texel_scale, point.z * texel_scale]
    } else if n.x >= n.z {
        [point.z * texel_scale, point.y * texel_scale]
    } else {
        [point.x * texel_scale, point.y * texel_scale]
    }
}

/// Orthonormal (tangent, bitangent) basis for a surface `normal` (from
/// `render_gpu::materials`).
pub fn tangent_basis_for_normal(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.86 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent).normalize_or_zero();
    (tangent, bitangent)
}

/// Recursively subdivide `points` into `subdiv` levels of uniform 4-split (from
/// `render_gpu::mesh_utils`).
pub fn subdivide_triangle(points: [Vec3; 3], subdiv: usize) -> Vec<[Vec3; 3]> {
    if subdiv <= 1 {
        return vec![points];
    }
    let [a, b, c] = points;
    let ab = (a + b) * 0.5;
    let bc = (b + c) * 0.5;
    let ca = (c + a) * 0.5;
    let next = subdiv - 1;
    let mut out = Vec::with_capacity(4usize.pow(next as u32));
    out.extend(subdivide_triangle([a, ab, ca], next));
    out.extend(subdivide_triangle([ab, b, bc], next));
    out.extend(subdivide_triangle([ca, bc, c], next));
    out.extend(subdivide_triangle([ab, bc, ca], next));
    out
}

/// Area of a triangle from its three corner points (from `render_gpu::mesh_utils`).
pub fn asset_triangle_area(points: [Vec3; 3]) -> f32 {
    let [a, b, c] = points;
    (b - a).cross(c - a).length() * 0.5
}

/// Highest triangle vertex Y that covers world XZ point `q` (from
/// `render_gpu::mesh_utils`).
pub fn local_roof_top(body_tris: &[[[f32; 3]; 3]], q: [f32; 2]) -> Option<f32> {
    let mut best: Option<f32> = None;
    for tri in body_tris {
        let [a, b, c] = *tri;
        let ax = a[0] - q[0];
        let az = a[2] - q[1];
        let bx = b[0] - q[0];
        let bz = b[2] - q[1];
        let cx = c[0] - q[0];
        let cz = c[2] - q[1];
        let d0 = ax * bz - az * bx;
        let d1 = bx * cz - bz * cx;
        let d2 = cx * az - cz * ax;
        if (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0) {
            let y = a[1].max(b[1]).max(c[1]);
            best = Some(best.map_or(y, |prev: f32| prev.max(y)));
        }
    }
    best
}

/// Unproject an NDC point through `inv_view_proj` to world space (from
/// `render_gpu::mesh_utils`).
pub fn unproject(inv_view_proj: Mat4, ndc_x: f32, ndc_y: f32, ndc_z: f32) -> Vec3 {
    let p = inv_view_proj * Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
    p.truncate() / p.w
}

/// Choose a modest internal render resolution that caps at `max_w` pixels wide
/// while preserving the display aspect ratio. Both dimensions are rounded to
/// multiples of 2 so upscalers and tile renderers stay aligned (from
/// `spectra_frame::helpers`).
pub fn internal_resolution(display_w: u32, display_h: u32, max_w: u32) -> (u32, u32) {
    let scale = (max_w as f32 / display_w as f32).min(1.0);
    let w = ((display_w as f32 * scale).round() as u32).max(2) & !1;
    let h = ((display_h as f32 * scale).round() as u32).max(2) & !1;
    (w, h)
}
