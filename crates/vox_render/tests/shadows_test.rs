use glam::Vec3;
use vox_render::shadows::ShadowMapper;

const RESOLUTION: usize = 512;
const BIAS: f32 = 0.005;

/// Helper: create a shadow mapper with sun shining straight down (-Y).
fn setup_sun_down() -> ShadowMapper {
    let mut sm = ShadowMapper::new(RESOLUTION);
    let camera_pos = Vec3::new(0.0, 10.0, 0.0);
    let camera_fwd = Vec3::new(0.0, 0.0, -1.0);
    let sun_dir = Vec3::new(0.0, -1.0, 0.0); // straight down
    sm.update(camera_pos, camera_fwd, sun_dir);
    sm
}

#[test]
fn point_directly_under_sun_is_not_in_shadow() {
    let mut sm = setup_sun_down();

    // Place a single splat high up (a "roof") at y=20.
    let positions = vec![Vec3::new(0.0, 20.0, -10.0)];
    let radii = vec![3.0];
    sm.render_shadow_map(&positions, &radii);

    // A point on the ground far away from the occluder should be lit.
    let ground_point = Vec3::new(50.0, 0.0, -10.0);
    assert!(
        !sm.is_in_shadow(ground_point, BIAS),
        "Point far from occluder should NOT be in shadow"
    );
}

#[test]
fn point_behind_tall_building_is_in_shadow() {
    let mut sm = ShadowMapper::new(RESOLUTION);
    // Camera looking toward -Z, building is 10m in front of camera (within cascade 0: 0-20m).
    let camera_pos = Vec3::new(0.0, 10.0, 10.0);
    let camera_fwd = Vec3::new(0.0, 0.0, -1.0);
    // Sun shining straight down.
    let sun_dir = Vec3::new(0.0, -1.0, 0.0);
    sm.update(camera_pos, camera_fwd, sun_dir);

    // Build a "tall building" as a column of wide splats at the cascade centre.
    let building_z = 0.0; // ~10m in front of camera
    let mut positions = Vec::new();
    let mut radii = Vec::new();
    for y in 0..30 {
        positions.push(Vec3::new(0.0, y as f32 * 1.0 + 5.0, building_z));
        radii.push(8.0); // wide splats to ensure good coverage
    }
    sm.render_shadow_map(&positions, &radii);

    // A point on the ground directly below the building should be in shadow.
    let shadowed_point = Vec3::new(0.0, 0.5, building_z);
    assert!(
        sm.is_in_shadow(shadowed_point, BIAS),
        "Point directly below building column should be in shadow"
    );
}

#[test]
fn shadow_map_update_does_not_panic() {
    let mut sm = ShadowMapper::new(256);
    let camera_pos = Vec3::new(100.0, 50.0, 100.0);
    let camera_fwd = Vec3::new(-1.0, -0.3, -1.0).normalize();
    let sun_dir = Vec3::new(0.3, -0.9, 0.1).normalize();

    // Should not panic with any reasonable inputs.
    sm.update(camera_pos, camera_fwd, sun_dir);

    // Verify matrices are not identity (they were computed).
    for cascade in &sm.cascades {
        assert_ne!(
            cascade.light_view_proj,
            glam::Mat4::IDENTITY,
            "Light VP should be computed, not identity"
        );
    }
}

#[test]
fn cascade_splits_cover_correct_ranges() {
    let sm = ShadowMapper::new(64);

    assert_eq!(sm.cascade_count(), 3);

    let (near0, far0) = sm.cascade_range(0);
    let (near1, far1) = sm.cascade_range(1);
    let (near2, far2) = sm.cascade_range(2);

    assert!((near0 - 0.0).abs() < f32::EPSILON);
    assert!((far0 - 20.0).abs() < f32::EPSILON);

    assert!((near1 - 20.0).abs() < f32::EPSILON);
    assert!((far1 - 100.0).abs() < f32::EPSILON);

    assert!((near2 - 100.0).abs() < f32::EPSILON);
    assert!((far2 - 500.0).abs() < f32::EPSILON);

    // Cascades are contiguous.
    assert!((far0 - near1).abs() < f32::EPSILON, "Cascade 0 far should equal cascade 1 near");
    assert!((far1 - near2).abs() < f32::EPSILON, "Cascade 1 far should equal cascade 2 near");
}

#[test]
fn light_view_projection_produces_valid_matrices() {
    let mut sm = ShadowMapper::new(128);
    let camera_pos = Vec3::new(0.0, 5.0, 0.0);
    let camera_fwd = Vec3::new(0.0, 0.0, -1.0);
    let sun_dir = Vec3::new(-0.5, -0.8, 0.2).normalize();

    sm.update(camera_pos, camera_fwd, sun_dir);

    for (i, cascade) in sm.cascades.iter().enumerate() {
        let vp = cascade.light_view_proj;

        // The determinant should be non-zero (invertible matrix).
        let det = vp.determinant();
        assert!(
            det.abs() > 1e-10,
            "Cascade {} light VP determinant too small: {}",
            i,
            det
        );

        // The matrix should not contain NaN or Inf.
        let cols = [vp.x_axis, vp.y_axis, vp.z_axis, vp.w_axis];
        for col in &cols {
            assert!(col.x.is_finite(), "Cascade {} has non-finite value", i);
            assert!(col.y.is_finite(), "Cascade {} has non-finite value", i);
            assert!(col.z.is_finite(), "Cascade {} has non-finite value", i);
            assert!(col.w.is_finite(), "Cascade {} has non-finite value", i);
        }
    }
}

#[test]
fn empty_scene_produces_no_shadows() {
    let mut sm = setup_sun_down();
    sm.render_shadow_map(&[], &[]);

    let test_point = Vec3::new(0.0, 0.0, -10.0);
    assert!(
        !sm.is_in_shadow(test_point, BIAS),
        "Empty scene should produce no shadows"
    );
}
