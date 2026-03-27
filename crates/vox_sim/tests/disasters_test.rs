use vox_sim::disasters::{DisasterManager, DisasterType};

#[test]
fn trigger_and_resolve_fire() {
    let mut mgr = DisasterManager::new();
    mgr.trigger(DisasterType::Fire, [50.0, 50.0], 0.5);
    assert_eq!(mgr.active_count(), 1);
    // Tick many times to resolve
    for _ in 0..2000 {
        mgr.tick(0.1);
    }
    assert_eq!(mgr.active_count(), 0);
}

#[test]
fn responding_services_help() {
    let mut mgr = DisasterManager::new();
    let id = mgr.trigger(DisasterType::Fire, [0.0, 0.0], 0.8);
    let initial_time = mgr.active[0].time_remaining;
    mgr.respond(id);
    assert!(mgr.active[0].time_remaining < initial_time);
}

#[test]
fn affected_area_check() {
    let mut mgr = DisasterManager::new();
    mgr.trigger(DisasterType::Flood, [0.0, 0.0], 0.5);
    assert!(mgr.is_affected([10.0, 10.0]).is_some());
    assert!(mgr.is_affected([500.0, 500.0]).is_none());
}
