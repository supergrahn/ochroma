use glam::Vec3;
use vox_render::cinematic::*;

#[test]
fn dof_coc_increases_with_distance() {
    let dof = DepthOfField {
        focal_distance: 10.0,
        aperture: 2.8,
        bokeh_shape: BokehShape::Circle,
    };
    let near = dof.coc_radius(10.0, 36.0); // at focus
    let far = dof.coc_radius(50.0, 36.0); // far from focus
    assert!(far > near, "CoC should increase with distance from focus");
}

#[test]
fn camera_path_interpolation() {
    let mut cam = CinematicCamera::new();
    cam.add_keyframe(0.0, Vec3::ZERO, Vec3::NEG_Z, 60.0);
    cam.add_keyframe(2.0, Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Z, 45.0);
    cam.playback_time = 1.0; // midpoint
    let (pos, _, fov) = cam.evaluate().unwrap();
    assert!(
        pos.x > 0.0 && pos.x < 10.0,
        "Should interpolate position"
    );
    assert!(fov > 45.0 && fov < 60.0, "Should interpolate FOV");
}

#[test]
fn looping_wraps_time() {
    let mut cam = CinematicCamera::new();
    cam.add_keyframe(0.0, Vec3::ZERO, Vec3::ZERO, 60.0);
    cam.add_keyframe(1.0, Vec3::X, Vec3::ZERO, 60.0);
    cam.looping = true;
    cam.playback_time = 1.5; // past end
    let (pos, _, _) = cam.evaluate().unwrap();
    assert!(pos.x < 1.0, "Looping should wrap time");
}

#[test]
fn finished_detection() {
    let mut cam = CinematicCamera::new();
    cam.add_keyframe(0.0, Vec3::ZERO, Vec3::ZERO, 60.0);
    cam.add_keyframe(1.0, Vec3::X, Vec3::ZERO, 60.0);
    assert!(!cam.is_finished());
    cam.playback_time = 2.0;
    assert!(cam.is_finished());
}

#[test]
fn lens_effects_defaults() {
    let lens = LensEffects::default();
    assert!(!lens.flare_enabled);
    assert_eq!(lens.chromatic_aberration, 0.0);
}
