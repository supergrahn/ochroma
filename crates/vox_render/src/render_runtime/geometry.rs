//! UV/mesh geometry + render resolution helpers (from render_gpu::materials / render_gpu::mesh_utils / spectra_frame::helpers).

use glam::{Mat4, Vec3, Vec4};

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
