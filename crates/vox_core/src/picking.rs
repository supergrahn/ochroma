//! Screen-to-world ray casting for mouse picking and building placement.

use glam::{Mat4, Vec3, Vec4};

/// A world-space ray originating from a screen-space mouse position.
#[derive(Clone, Debug)]
pub struct ScreenRay {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// A pickable entity represented as a world-space sphere.
#[derive(Clone, Debug)]
pub struct SplatPickEntry {
    pub position: [f32; 3],
    pub radius: f32,
}

impl ScreenRay {
    pub fn from_screen(sx: f32, sy: f32, width: f32, height: f32, view_proj_inv: Mat4) -> Self {
        let ndc_x = (sx / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (sy / height) * 2.0;

        // Use depth 0.0 for the near plane and 1.0 for the far plane,
        // matching the wgpu / Vulkan / Metal depth range used by glam's
        // orthographic_rh and perspective_rh projection matrices.
        let near_ndc = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_ndc  = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near_world = view_proj_inv * near_ndc;
        let far_world  = view_proj_inv * far_ndc;

        let near_w = Vec3::new(near_world.x, near_world.y, near_world.z) / near_world.w;
        let far_w  = Vec3::new(far_world.x,  far_world.y,  far_world.z)  / far_world.w;

        let direction = (far_w - near_w).normalize();
        Self { origin: near_w, direction }
    }

    pub fn terrain_hit(&self, height_fn: &dyn Fn(f32, f32) -> f32, max_dist: f32) -> Option<Vec3> {
        let steps = 64usize;
        let mut t_lo = 0.0f32;
        let mut t_hi = max_dist;

        let sample = |t: f32| -> f32 {
            let p = self.origin + self.direction * t;
            p.y - height_fn(p.x, p.z)
        };

        if sample(0.0) < 0.0 {
            return Some(self.origin);
        }
        if sample(t_hi) >= 0.0 {
            return None;
        }

        for _ in 0..steps {
            let t_mid = (t_lo + t_hi) * 0.5;
            if sample(t_mid) >= 0.0 {
                t_lo = t_mid;
            } else {
                t_hi = t_mid;
            }
        }

        let t = (t_lo + t_hi) * 0.5;
        let p = self.origin + self.direction * t;
        Some(Vec3::new(p.x, height_fn(p.x, p.z), p.z))
    }

    pub fn nearest_splat(&self, splats: &[SplatPickEntry], max_dist: f32) -> Option<usize> {
        let mut best_idx = None;
        let mut best_t = f32::MAX;

        for (i, entry) in splats.iter().enumerate() {
            let center = Vec3::from(entry.position);
            let oc = self.origin - center;
            let b = oc.dot(self.direction);
            let c = oc.dot(oc) - entry.radius * entry.radius;
            let discriminant = b * b - c;
            if discriminant < 0.0 {
                continue;
            }
            let t = -b - discriminant.sqrt();
            if t < 0.0 || t > max_dist {
                continue;
            }
            if t < best_t {
                best_t = t;
                best_idx = Some(i);
            }
        }
        best_idx
    }
}
