use vox_render::camera::{CameraController, CameraMode};

#[test]
fn new_camera_at_default_position() {
    let cam = CameraController::new(16.0 / 9.0);
    assert!(cam.position.y > 0.0);
    assert_eq!(cam.mode, CameraMode::CityOverview);
}

#[test]
fn orbit_changes_position() {
    let mut cam = CameraController::new(1.0);
    let initial = cam.position;
    cam.orbit(0.5);
    assert_ne!(cam.position.x, initial.x);
}

#[test]
fn zoom_changes_distance() {
    let mut cam = CameraController::new(1.0);
    let initial_dist = cam.orbit_distance;
    cam.zoom(-20.0);
    assert!(cam.orbit_distance < initial_dist);
}

#[test]
fn altitude_clamped_above_zero() {
    let mut cam = CameraController::new(1.0);
    cam.set_altitude(-100.0);
    assert!(cam.altitude >= 1.0);
}

#[test]
fn view_proj_produces_valid_matrix() {
    let cam = CameraController::new(16.0 / 9.0);
    let vp = cam.view_proj();
    assert!(vp.x_axis.length() > 0.0);
}
