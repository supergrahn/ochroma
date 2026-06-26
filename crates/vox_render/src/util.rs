//! Pure-engine leaf utilities: screen/camera math, geometry bounds helpers.
//!
//! All items here are game-agnostic — no building types, no ECS components,
//! no game-layer deps. The game re-exports the public surface from
//! `render_gpu::mod` so call sites in `play.rs` and `hud.rs` compile unchanged.

use glam::{Mat4, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Camera math — screen↔world projections
// ---------------------------------------------------------------------------

/// Cast a ray from screen pixel `(px, py)` and intersect the ground plane
/// `y = 0`; returns the world `[x, z]` hit (or `None` if the ray misses, e.g.
/// pointing at the sky). Used to turn a click into a place-here position.
pub fn screen_to_ground(
    view: Mat4,
    proj: Mat4,
    px: f32,
    py: f32,
    width: u32,
    height: u32,
) -> Option<[f32; 2]> {
    let inv = (proj * view).inverse();
    let ndc_x = 2.0 * (px + 0.5) / width as f32 - 1.0;
    let ndc_y = 1.0 - 2.0 * (py + 0.5) / height as f32;
    let unproject = |z: f32| -> Vec3 {
        let p = inv * Vec4::new(ndc_x, ndc_y, z, 1.0);
        p.truncate() / p.w
    };
    let near = unproject(0.0);
    let dir = (unproject(1.0) - near).normalize();
    if dir.y.abs() < 1e-5 {
        return None;
    }
    let t = -near.y / dir.y;
    if t < 0.0 {
        return None;
    }
    let hit = near + dir * t;
    Some([hit.x, hit.z])
}

/// Project a ground point `(x, z)` (y = 0) to screen pixel coordinates — the
/// inverse of [`screen_to_ground`], used to draw in-progress selection/region
/// outlines over the rendered scene. Returns `None` if the point is behind the camera.
pub fn world_to_screen(
    view: Mat4,
    proj: Mat4,
    x: f32,
    z: f32,
    width: u32,
    height: u32,
) -> Option<(f32, f32)> {
    let clip = proj * view * Vec4::new(x, 0.0, z, 1.0);
    if clip.w <= 1e-5 {
        return None; // behind / on the camera plane
    }
    let ndc = clip.truncate() / clip.w;
    let px = (ndc.x + 1.0) * 0.5 * width as f32 - 0.5;
    let py = (1.0 - ndc.y) * 0.5 * height as f32 - 0.5;
    Some((px, py))
}

// ---------------------------------------------------------------------------
// Camera-driven atom filter — pure-engine, no game deps
// ---------------------------------------------------------------------------

/// Camera-driven resident atom selector.
///
/// The city view keeps the game assets as authored instances, then admits only
/// the atoms whose owning instance is near enough and inside the current camera
/// frame. This is asset atom streaming, not a mesh-level LOD table.
#[derive(Debug, Clone, Copy)]
pub struct CameraAtomFilter {
    pub eye: Vec3,
    pub view_proj: Mat4,
    pub max_distance: f32,
    pub frame_pad: f32,
}

impl CameraAtomFilter {
    pub fn from_camera(camera: &crate::spectral::RenderCamera, max_distance: f32) -> Self {
        let eye = camera.view.inverse().transform_point3(Vec3::ZERO);
        Self {
            eye,
            view_proj: camera.proj * camera.view,
            max_distance,
            frame_pad: 0.22,
        }
    }

    pub fn accepts_asset(&self, center: Vec3, radius: f32) -> bool {
        if center.distance(self.eye) - radius > self.max_distance {
            return false;
        }

        let clip = self.view_proj * center.extend(1.0);
        if clip.w <= 0.01 {
            return false;
        }
        let ndc = clip.truncate() / clip.w;
        let distance = center.distance(self.eye).max(1.0);
        let radius_pad = (radius / distance).clamp(0.04, 0.65);
        let pad = self.frame_pad + radius_pad;

        ndc.x >= -1.0 - pad
            && ndc.x <= 1.0 + pad
            && ndc.y >= -1.0 - pad
            && ndc.y <= 1.0 + pad
            && ndc.z >= -0.35
            && ndc.z <= 1.35
    }
}

// ---------------------------------------------------------------------------
// Scene bounds — used by game-side resident_asset_bounds / visual_bounds
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box over a rendered scene or sub-scene. Used to
/// auto-frame cameras and compute culling budgets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneBounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl SceneBounds {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    pub fn into_option(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }

    pub fn include_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn include_sphere(&mut self, center: Vec3, radius: f32) {
        let r = Vec3::splat(radius.max(0.0));
        self.include_point(center - r);
        self.include_point(center + r);
    }

    pub fn padded(mut self, padding: f32) -> Self {
        let p = Vec3::splat(padding.max(0.0));
        self.min -= p;
        self.max += p;
        self
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn radius(&self) -> f32 {
        ((self.max - self.min).length() * 0.5).max(1.0)
    }
}

// ---------------------------------------------------------------------------
// Generic bounds accumulators — game-agnostic (take &[HybridMesh] etc.)
// ---------------------------------------------------------------------------

/// Expand `bounds` to include all vertex positions in `meshes`.
pub fn include_mesh_bounds(
    bounds: &mut SceneBounds,
    meshes: &[crate::hybrid_compose::HybridMesh],
) {
    for mesh in meshes {
        for position in &mesh.positions {
            bounds.include_point(Vec3::from(*position));
        }
    }
}

/// Expand `bounds` to include all world-space AABBs given as `(min, max)` pairs.
/// Use this for SDF volumes after calling `.world_aabb()` on each volume — the
/// game precomputes these since `ResidentSdfVolume` has game-layer deps.
pub fn include_aabb_bounds(bounds: &mut SceneBounds, aabbs: &[([f32; 3], [f32; 3])]) {
    for (min, max) in aabbs {
        bounds.include_point(Vec3::from(*min));
        bounds.include_point(Vec3::from(*max));
    }
}

/// Expand `bounds` to include all Gaussian splat volumes in `splats`.
pub fn include_splat_bounds(
    bounds: &mut SceneBounds,
    splats: &[vox_core::types::GaussianSplat],
) {
    for splat in splats {
        let radius = splat
            .scales()
            .into_iter()
            .fold(0.0_f32, |acc, scale| acc.max(scale.abs()))
            .max(0.1);
        bounds.include_sphere(Vec3::from(splat.position()), radius);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    #[test]
    fn screen_to_ground_projects_centre_pixel() {
        // Orthographic-ish camera looking straight down from y=100
        let view = Mat4::look_at_rh(
            glam::vec3(0.0, 100.0, 0.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 0.0, -1.0),
        );
        let proj =
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 500.0);
        // Centre pixel of a 100x100 viewport — ray hits y=0 plane somewhere near origin
        let hit = screen_to_ground(view, proj, 50.0, 50.0, 100, 100);
        assert!(hit.is_some(), "centre ray must intersect ground plane");
        let [x, z] = hit.unwrap();
        assert!(x.abs() < 5.0, "expected near origin x, got {x}");
        assert!(z.abs() < 5.0, "expected near origin z, got {z}");
    }

    #[test]
    fn scene_bounds_empty_then_grow() {
        let mut b = SceneBounds::empty();
        assert!(b.is_empty());
        b.include_point(Vec3::new(1.0, 2.0, 3.0));
        assert!(!b.is_empty());
        assert_eq!(b.min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(b.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn include_aabb_bounds_two_boxes() {
        let mut b = SceneBounds::empty();
        include_aabb_bounds(
            &mut b,
            &[([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])],
        );
        assert!(!b.is_empty());
        assert_eq!(b.min, Vec3::splat(0.0));
        assert_eq!(b.max, Vec3::splat(3.0));
    }
}
