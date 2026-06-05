//! Conversion from Ochroma's `GaussianSplat` to a Spectra [`SceneState`].
//!
//! The current `spectra-renderer` is a triangle-mesh spectral path tracer driven
//! by `spectra_scene_state::SceneState` (it no longer ingests Gaussian primitives
//! directly — the old `SplatScene` / `GaussianDisc` / `GaussianBlob` API was
//! removed and the Gaussian rasteriser now lives in the separate
//! `spectra-gaussian-render` crate, which vox_render does not depend on).
//!
//! To render Ochroma splats through the native path tracer we tessellate each
//! splat into a small camera-agnostic quad (2 triangles) placed in world space:
//!
//! * `kind == 0` (surface / 2DGS) → a flat quad spanned by `tangent_u`/`tangent_v`,
//!   scaled by `scale_u`/`scale_v`, with the quad normal = `cross(tu, tv)`.
//! * `kind == 1` (volume / 3DGS) → an axis-aligned quad on the XY plane sized by
//!   the splat's `scale_u`/`scale_v`, facing +Z. (A full ellipsoid tessellation is
//!   future work; a single quad gives a renderable proxy footprint.)
//!
//! Spectral: Ochroma carries 16 bands (380–755 nm). The path tracer's per-material
//! spectral SPD is 16 bands as well, but wiring a unique material per splat requires
//! packing the 132-float `MaterialData` struct, which is out of scope here. All
//! tessellated triangles therefore use the uploader's default material (index 0);
//! `material_ids` is left empty so the uploader assigns material 0 to every triangle.
//! Per-splat spectral colour is preserved on this struct's API surface (see the
//! returned vertex normals / positions) and can be promoted to per-material SPDs
//! once the material packing helper lands.

use spectra_scene_state::{CameraLayer, SceneState};
use vox_core::types::GaussianSplat;

/// One quad = 4 vertices, 2 triangles (6 indices).
const VERTS_PER_SPLAT: usize = 4;
const INDICES_PER_SPLAT: usize = 6;

/// Compute the four corner positions and the normal of a single splat's quad.
///
/// Returns `([p0, p1, p2, p3], normal)` where the corners wind CCW:
/// `p0 = c - u - v`, `p1 = c + u - v`, `p2 = c + u + v`, `p3 = c - u + v`,
/// with `u = half_u * tangent_u` and `v = half_v * tangent_v`.
fn splat_quad(s: &GaussianSplat) -> ([[f32; 3]; 4], [f32; 3]) {
    let c = s.position();

    // Choose the in-plane axes. Surface splats carry an authored tangent frame;
    // volume splats have no surface, so we fall back to an XY billboard.
    let (tu, tv) = if s.is_surface() {
        (s.tangent_u(), s.tangent_v())
    } else {
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    };

    let hu = s.scale_u().max(1e-6);
    let hv = s.scale_v().max(1e-6);

    let u = [tu[0] * hu, tu[1] * hu, tu[2] * hu];
    let v = [tv[0] * hv, tv[1] * hv, tv[2] * hv];

    let p0 = [c[0] - u[0] - v[0], c[1] - u[1] - v[1], c[2] - u[2] - v[2]];
    let p1 = [c[0] + u[0] - v[0], c[1] + u[1] - v[1], c[2] + u[2] - v[2]];
    let p2 = [c[0] + u[0] + v[0], c[1] + u[1] + v[1], c[2] + u[2] + v[2]];
    let p3 = [c[0] - u[0] + v[0], c[1] - u[1] + v[1], c[2] - u[2] + v[2]];

    // Normal = normalize(cross(tu, tv)).
    let nx = tu[1] * tv[2] - tu[2] * tv[1];
    let ny = tu[2] * tv[0] - tu[0] * tv[2];
    let nz = tu[0] * tv[1] - tu[1] * tv[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-8);
    let normal = [nx / len, ny / len, nz / len];

    ([p0, p1, p2, p3], normal)
}

/// Append a single splat's quad (4 verts, 2 tris) to the flat geometry arrays.
///
/// `base` is the current vertex count (== positions.len() / 3) before appending,
/// used to offset the triangle indices.
fn push_splat_quad(
    s: &GaussianSplat,
    base: u32,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) {
    let (corners, n) = splat_quad(s);

    for p in &corners {
        positions.extend_from_slice(p);
        normals.extend_from_slice(&n);
    }
    // UVs for the four corners (unit square).
    uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

    // Two triangles: (0,1,2) and (0,2,3), offset by `base`.
    indices.extend_from_slice(&[
        base,
        base + 1,
        base + 2,
        base,
        base + 2,
        base + 3,
    ]);
}

/// Convert a slice of [`GaussianSplat`] into a renderable [`SceneState`].
///
/// Every splat is tessellated into a quad and packed into the scene's
/// [`spectra_scene_state::GeometryLayer`]. `width`/`height` set the render target
/// and camera resolution; the caller fills in the real camera via
/// [`SceneState::camera`] before rendering.
///
/// Splats with an unrecognised `kind` are skipped.
pub fn splats_to_scene(splats: &[GaussianSplat], width: u32, height: u32) -> SceneState {
    let renderable: Vec<&GaussianSplat> = splats
        .iter()
        .filter(|s| s.is_surface() || s.is_volume())
        .collect();

    let mut positions: Vec<f32> = Vec::with_capacity(renderable.len() * VERTS_PER_SPLAT * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(renderable.len() * VERTS_PER_SPLAT * 3);
    let mut uvs: Vec<f32> = Vec::with_capacity(renderable.len() * VERTS_PER_SPLAT * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(renderable.len() * INDICES_PER_SPLAT);

    for (i, s) in renderable.iter().enumerate() {
        let base = (i * VERTS_PER_SPLAT) as u32;
        push_splat_quad(s, base, &mut positions, &mut normals, &mut uvs, &mut indices);
    }

    let mut scene = SceneState::new(width, height);
    scene.geometry.vertex_count = positions.len() / 3;
    scene.geometry.triangle_count = indices.len() / 3;
    scene.geometry.positions = positions;
    scene.geometry.normals = normals;
    scene.geometry.uvs = uvs;
    scene.geometry.indices = indices;
    // material_ids left empty → uploader assigns the default material (index 0).
    scene.mark_geometry_changed();
    scene
}

/// Build a [`CameraLayer`] from a column-major view matrix and vertical FOV.
///
/// This is the camera type the native renderer consumes (`Renderer::set_camera_view_matrix`
/// reads `CameraLayer::view_matrix`). Width/height set the target resolution and aspect.
pub fn camera_layer(view_matrix: [f32; 16], fov_y_radians: f32, width: u32, height: u32) -> CameraLayer {
    let mut cam = CameraLayer::new_default();
    cam.view_matrix = view_matrix;
    cam.fov_y_radians = fov_y_radians;
    cam.width = width;
    cam.height = height;
    cam
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn zero_spectral() -> [u16; 16] {
        [0u16; 16]
    }

    #[test]
    fn surface_quad_normal_is_z_axis() {
        // tangent_u = X, tangent_v = Y  →  normal = +Z.
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let (_corners, n) = splat_quad(&s);
        assert!(n[2] > 0.99, "nz = {} expected ~1.0", n[2]);
        assert!(n[0].abs() < 1e-5);
        assert!(n[1].abs() < 1e-5);
    }

    #[test]
    fn surface_quad_corner_spread_matches_scale() {
        // scale_u = 2, scale_v = 0.5, axes = X/Y, centre at origin.
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            2.0,
            0.5,
            255,
            zero_spectral(),
        );
        let (corners, _n) = splat_quad(&s);
        // p2 = c + u + v = (+2, +0.5, 0); p0 = (-2, -0.5, 0).
        assert_eq!(corners[2], [2.0, 0.5, 0.0]);
        assert_eq!(corners[0], [-2.0, -0.5, 0.0]);
    }

    #[test]
    fn scene_has_two_triangles_per_splat() {
        let surf = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let vol = GaussianSplat::volume(
            [5.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            glam::Quat::IDENTITY,
            128,
            zero_spectral(),
        );
        let scene = splats_to_scene(&[surf, vol], 64, 48);
        // 2 splats → 8 vertices, 4 triangles, 12 indices.
        assert_eq!(scene.geometry.vertex_count, 8);
        assert_eq!(scene.geometry.triangle_count, 4);
        assert_eq!(scene.geometry.indices.len(), 12);
        assert_eq!(scene.geometry.positions.len(), 24);
        assert_eq!(scene.width, 64);
        assert_eq!(scene.height, 48);
    }

    #[test]
    fn scene_indices_offset_per_splat() {
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let scene = splats_to_scene(&[s.clone(), s], 32, 32);
        // Second splat's triangles must reference vertices 4..7, not 0..3.
        // Last index is the last corner of the second quad → base(4) + 3 = 7.
        assert_eq!(*scene.geometry.indices.last().unwrap(), 7);
        // Max index must equal vertex_count - 1.
        let max_idx = *scene.geometry.indices.iter().max().unwrap();
        assert_eq!(max_idx as usize, scene.geometry.vertex_count - 1);
    }

    #[test]
    fn camera_layer_carries_view_and_fov() {
        let view = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -5.0, 1.0,
        ];
        let cam = camera_layer(view, 0.8, 320, 240);
        assert_eq!(cam.view_matrix, view);
        assert_eq!(cam.fov_y_radians, 0.8);
        assert_eq!(cam.width, 320);
        assert_eq!(cam.height, 240);
    }
}
