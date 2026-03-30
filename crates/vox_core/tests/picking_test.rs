use glam::{Mat4, Vec3};
use vox_core::picking::{ScreenRay, SplatPickEntry};

fn top_down_view_proj_inv() -> Mat4 {
    let view = Mat4::look_at_rh(
        Vec3::new(5.0, 10.0, 5.0),
        Vec3::new(5.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    let proj = Mat4::orthographic_rh(-5.0, 5.0, -5.0, 5.0, 0.1, 20.0);
    (proj * view).inverse()
}

#[test]
fn mouse_ray_hits_terrain_at_correct_world_position() {
    let vp_inv = top_down_view_proj_inv();
    let ray = ScreenRay::from_screen(50.0, 50.0, 100.0, 100.0, vp_inv);
    let hit = ray.terrain_hit(&|_x, _z| 0.0, 20.0);
    assert!(hit.is_some(), "ray must hit flat terrain");
    let h = hit.unwrap();
    println!("terrain hit at [{:.2}, {:.2}, {:.2}]", h.x, h.y, h.z);
    assert!((h.x - 5.0).abs() < 0.1, "x must be ~5.0, got {}", h.x);
    assert!((h.z - 5.0).abs() < 0.1, "z must be ~5.0, got {}", h.z);
    assert!(h.y.abs() < 0.1, "y must be ~0.0, got {}", h.y);
}

#[test]
fn mouse_ray_no_hit_beyond_max_dist() {
    let vp_inv = top_down_view_proj_inv();
    let ray = ScreenRay::from_screen(50.0, 50.0, 100.0, 100.0, vp_inv);
    let hit = ray.terrain_hit(&|_x, _z| 0.0, 0.5);
    assert!(hit.is_none(), "ray must not hit terrain beyond max_dist");
}

#[test]
fn mouse_ray_selects_nearest_splat() {
    let vp_inv = top_down_view_proj_inv();
    let ray = ScreenRay::from_screen(75.0, 50.0, 100.0, 100.0, vp_inv);

    let splats = vec![
        SplatPickEntry { position: [3.0, 0.0, 5.0], radius: 0.5 },
        SplatPickEntry { position: [7.5, 0.0, 5.0], radius: 0.5 },
    ];
    let result = ray.nearest_splat(&splats, 20.0);
    println!("selected splat index: {:?}", result);
    assert_eq!(result, Some(1), "ray aimed right must select splat at index 1");
}

#[test]
fn mouse_ray_returns_none_when_no_splat_in_range() {
    let vp_inv = top_down_view_proj_inv();
    let ray = ScreenRay::from_screen(50.0, 50.0, 100.0, 100.0, vp_inv);
    let splats = vec![
        SplatPickEntry { position: [0.0, 0.0, 0.0], radius: 0.1 },
    ];
    let result = ray.nearest_splat(&splats, 0.05);
    assert!(result.is_none(), "no splat within max_dist must return None");
}
