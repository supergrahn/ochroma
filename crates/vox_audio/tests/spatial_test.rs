//! Tests for the spatial audio manager — pure math, no speakers needed.

use glam::Vec3;
use vox_audio::spatial::{compute_spatial, Listener, SpatialAudioManager};

// ── Distance attenuation ──────────────────────────────────────────────────

#[test]
fn closer_source_has_higher_volume() {
    let listener = Listener {
        position: Vec3::ZERO,
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    let base_vol = 1.0;
    let atten = 0.1;

    let (vol_near, _) = compute_spatial(Vec3::new(0.0, 0.0, -5.0), &listener, base_vol, atten);
    let (vol_far, _) = compute_spatial(Vec3::new(0.0, 0.0, -50.0), &listener, base_vol, atten);

    assert!(
        vol_near > vol_far,
        "near volume ({vol_near}) should be louder than far volume ({vol_far})"
    );
}

#[test]
fn source_at_listener_has_full_volume() {
    let listener = Listener::default();
    let (vol, pan) = compute_spatial(listener.position, &listener, 1.0, 0.1);
    assert!((vol - 1.0).abs() < 1e-5, "volume at listener should be ~1.0, got {vol}");
    assert!(pan.abs() < 1e-5, "pan at listener should be ~0, got {pan}");
}

#[test]
fn attenuation_formula_matches_spec() {
    let listener = Listener::default();
    let distance = 10.0;
    let atten = 0.1;
    let expected = 1.0 / (1.0 + distance * atten);
    let pos = Vec3::new(0.0, 0.0, -distance);
    let (vol, _) = compute_spatial(pos, &listener, 1.0, atten);
    assert!(
        (vol - expected).abs() < 1e-5,
        "expected {expected}, got {vol}"
    );
}

// ── Stereo panning ────────────────────────────────────────────────────────

#[test]
fn source_to_right_has_positive_pan() {
    let listener = Listener {
        position: Vec3::ZERO,
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    // Right vector = forward x up = (-Z) x Y = -X ... wait, let's verify:
    // cross(-Z, Y) = (-Z).cross(Y)
    // = (0, 0, -1) x (0, 1, 0) = (0*0 - (-1)*1, (-1)*0 - 0*0, 0*1 - 0*0) = (1, 0, 0)
    // So right = +X. Source at +X should give positive pan.
    let (_, pan) = compute_spatial(Vec3::new(10.0, 0.0, 0.0), &listener, 1.0, 0.1);
    assert!(pan > 0.0, "source to the right should have positive pan, got {pan}");
}

#[test]
fn source_to_left_has_negative_pan() {
    let listener = Listener {
        position: Vec3::ZERO,
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    let (_, pan) = compute_spatial(Vec3::new(-10.0, 0.0, 0.0), &listener, 1.0, 0.1);
    assert!(pan < 0.0, "source to the left should have negative pan, got {pan}");
}

#[test]
fn source_directly_ahead_has_zero_pan() {
    let listener = Listener {
        position: Vec3::ZERO,
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    let (_, pan) = compute_spatial(Vec3::new(0.0, 0.0, -10.0), &listener, 1.0, 0.1);
    assert!(
        pan.abs() < 1e-5,
        "source directly ahead should have ~0 pan, got {pan}"
    );
}

// ── Listener update changes volumes ───────────────────────────────────────

#[test]
fn moving_listener_closer_increases_volume() {
    let source_pos = Vec3::new(0.0, 0.0, -20.0);
    let base_vol = 1.0;
    let atten = 0.1;

    let listener_far = Listener {
        position: Vec3::ZERO,
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    let listener_near = Listener {
        position: Vec3::new(0.0, 0.0, -15.0),
        forward: -Vec3::Z,
        up: Vec3::Y,
    };

    let (vol_far, _) = compute_spatial(source_pos, &listener_far, base_vol, atten);
    let (vol_near, _) = compute_spatial(source_pos, &listener_near, base_vol, atten);

    assert!(
        vol_near > vol_far,
        "moving listener closer should increase volume: near={vol_near} far={vol_far}"
    );
}

// ── SpatialAudioManager integration ───────────────────────────────────────

#[test]
fn manager_play_and_stop_multiple_sources() {
    let mut mgr = SpatialAudioManager::new_silent();

    let h1 = mgr.play_3d(
        std::path::Path::new("nonexistent.wav"),
        Vec3::new(5.0, 0.0, 0.0),
        0.8,
        false,
    );
    let h2 = mgr.play_3d(
        std::path::Path::new("also_nonexistent.wav"),
        Vec3::new(-5.0, 0.0, 0.0),
        0.6,
        false,
    );
    let h3 = mgr.play_2d(std::path::Path::new("music.wav"), 1.0);

    assert_eq!(mgr.active_count(), 3);
    assert!(mgr.is_playing(h1));
    assert!(mgr.is_playing(h2));
    assert!(mgr.is_playing(h3));

    mgr.stop(h1);
    assert_eq!(mgr.active_count(), 2);
    assert!(!mgr.is_playing(h1));
    assert!(mgr.is_playing(h2));

    mgr.stop(h2);
    mgr.stop(h3);
    assert_eq!(mgr.active_count(), 0);
}

#[test]
fn manager_set_source_position_updates_spatial() {
    let mut mgr = SpatialAudioManager::new_silent();
    mgr.set_listener(Vec3::ZERO, -Vec3::Z, Vec3::Y);

    let h = mgr.play_3d(
        std::path::Path::new("test.wav"),
        Vec3::new(100.0, 0.0, 0.0),
        1.0,
        false,
    );

    // Far away — should have lower volume.
    let (vol_far, _) = mgr.compute_spatial_for(Vec3::new(100.0, 0.0, 0.0), 1.0);

    // Move it closer.
    mgr.set_source_position(h, Vec3::new(1.0, 0.0, 0.0));
    let (vol_near, _) = mgr.compute_spatial_for(Vec3::new(1.0, 0.0, 0.0), 1.0);

    assert!(vol_near > vol_far);
    assert!(mgr.is_playing(h));
}

#[test]
fn play_tone_creates_source() {
    let mut mgr = SpatialAudioManager::new_silent();
    assert_eq!(mgr.active_count(), 0);

    let h = mgr.play_tone(440.0, 1.0, 0.5);
    assert_eq!(mgr.active_count(), 1);
    assert!(mgr.is_playing(h));
}

#[test]
fn tone_auto_finishes_after_duration() {
    let mut mgr = SpatialAudioManager::new_silent();
    let h = mgr.play_tone(440.0, 0.5, 1.0);
    assert!(mgr.is_playing(h));

    // Tick past the duration.
    mgr.tick(0.6);
    assert!(!mgr.is_playing(h));
    assert_eq!(mgr.active_count(), 0);
}

#[test]
fn silent_manager_reports_not_available() {
    let mgr = SpatialAudioManager::new_silent();
    assert!(!mgr.is_available());
}

#[test]
fn tick_updates_spatial_sources() {
    let mut mgr = SpatialAudioManager::new_silent();
    mgr.set_listener(Vec3::ZERO, -Vec3::Z, Vec3::Y);

    let _h = mgr.play_3d(
        std::path::Path::new("test.wav"),
        Vec3::new(10.0, 0.0, 0.0),
        1.0,
        false,
    );

    // Should not panic and source should remain active.
    mgr.tick(0.016);
    assert_eq!(mgr.active_count(), 1);
}

#[test]
fn listener_right_vector_is_correct() {
    let listener = Listener {
        position: Vec3::ZERO,
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    let right = listener.right();
    // forward(-Z) x up(Y) = +X
    assert!((right - Vec3::X).length() < 1e-5, "right should be +X, got {right}");
}
