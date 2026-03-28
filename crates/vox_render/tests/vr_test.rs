use vox_render::vr::*;
use glam::Quat;

#[test]
fn default_headset_at_standing_height() {
    let state = HeadsetState::default();
    assert!((state.head_position.y - 1.7).abs() < 0.01);
}

#[test]
fn stereo_views_offset_by_ipd() {
    let state = HeadsetState::default();
    let (left, right) = state.compute_eye_views(0.1, 1000.0, 1.5, 0.9);
    // Left eye should be offset to the left of right eye
    let left_pos = left.view_matrix.inverse().col(3).truncate();
    let right_pos = right.view_matrix.inverse().col(3).truncate();
    assert!(left_pos.x < right_pos.x, "Left eye should be left of right");
}

#[test]
fn desktop_mode_no_stereo() {
    let session = VrSession::new_desktop();
    assert!(!session.needs_stereo());
}

#[test]
fn simulated_mode_needs_stereo() {
    let session = VrSession::new_simulated();
    assert!(session.needs_stereo());
}

#[test]
fn simulate_head_turn() {
    let mut session = VrSession::new_simulated();
    session.simulate_head_turn(1.0, 0.0);
    assert_ne!(session.headset.head_rotation, Quat::IDENTITY);
}

#[test]
fn target_framerate_72fps() {
    let session = VrSession::new_simulated();
    assert!((session.target_frame_ms() - 13.89).abs() < 0.1);
}
