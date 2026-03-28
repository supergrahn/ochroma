use glam::{Vec3, Vec4, Mat4, Vec4Swizzles};

struct Plane {
    normal: Vec3,
    d: f32,
}

impl Plane {
    fn from_vec4(v: Vec4) -> Self {
        let len = v.xyz().length();
        Self {
            normal: v.xyz() / len,
            d: v.w / len,
        }
    }

    fn distance_to(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    pub fn from_view_proj(vp: Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        let planes = [
            Plane::from_vec4(row3 + row0), // Left
            Plane::from_vec4(row3 - row0), // Right
            Plane::from_vec4(row3 + row1), // Bottom
            Plane::from_vec4(row3 - row1), // Top
            Plane::from_vec4(row3 + row2), // Near
            Plane::from_vec4(row3 - row2), // Far
        ];

        Self { planes }
    }

    pub fn contains_sphere(&self, centre: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.distance_to(centre) < -radius {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frustum() -> Frustum {
        // Simple perspective looking down -Z
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        Frustum::from_view_proj(proj * view)
    }

    #[test]
    fn origin_is_visible() {
        let f = test_frustum();
        assert!(f.contains_sphere(Vec3::ZERO, 0.5), "origin should be inside frustum");
    }

    #[test]
    fn point_behind_camera_is_not_visible() {
        let f = test_frustum();
        assert!(!f.contains_sphere(Vec3::new(0.0, 0.0, 10.0), 0.1),
            "point behind camera should be outside frustum");
    }

    #[test]
    fn far_away_point_is_not_visible() {
        let f = test_frustum();
        assert!(!f.contains_sphere(Vec3::new(0.0, 0.0, -200.0), 0.1),
            "point beyond far plane should be culled");
    }
}
