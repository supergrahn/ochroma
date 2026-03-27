use glam::{Vec3, Mat4};
use vox_render::frustum::Frustum;

#[test]
fn point_inside_frustum_is_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);
    assert!(frustum.contains_sphere(Vec3::new(0.0, 0.0, -10.0), 1.0));
}

#[test]
fn point_behind_camera_is_not_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);
    assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, 10.0), 1.0));
}

#[test]
fn point_far_right_is_not_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);
    assert!(!frustum.contains_sphere(Vec3::new(200.0, 0.0, -10.0), 1.0));
}

#[test]
fn sphere_partially_inside_is_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);
    assert!(frustum.contains_sphere(Vec3::new(50.0, 0.0, -10.0), 100.0));
}
