use vox_render::lod_crossfade::*;

#[test]
fn transition_progresses_over_time() {
    let mut t = LodTransition::new(0, 0, 1, 0.5); // 0.5 second transition
    assert_eq!(t.progress, 0.0);
    t.tick(0.25);
    assert!((t.progress - 0.5).abs() < 0.01);
    t.tick(0.25);
    assert!(t.is_complete());
}

#[test]
fn opacity_crossfade() {
    let mut t = LodTransition::new(0, 0, 1, 1.0);
    t.tick(0.5);
    assert!((t.from_opacity() - 0.5).abs() < 0.01);
    assert!((t.to_opacity() - 0.5).abs() < 0.01);
}

#[test]
fn manager_removes_completed_transitions() {
    let mut mgr = LodCrossfadeManager::new(0.1);
    mgr.request_lod_change(0, 0, 1);
    assert_eq!(mgr.active_count(), 1);
    mgr.tick(0.2); // exceeds duration
    assert_eq!(mgr.active_count(), 0);
}

#[test]
fn same_lod_no_transition() {
    let mut mgr = LodCrossfadeManager::new(0.5);
    mgr.request_lod_change(0, 1, 1); // same LOD
    assert_eq!(mgr.active_count(), 0);
}
