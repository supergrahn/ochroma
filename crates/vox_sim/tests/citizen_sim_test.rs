use vox_sim::citizen::{CitizenManager, LifecycleStage};

#[test]
fn spawn_citizen_correct_lifecycle() {
    let mut mgr = CitizenManager::new();
    let id = mgr.spawn(25.0, None);
    let c = mgr.get(id).unwrap();
    assert_eq!(c.lifecycle, LifecycleStage::Worker);
}

#[test]
fn tick_ages_citizens() {
    let mut mgr = CitizenManager::new();
    mgr.spawn(17.0, None);
    mgr.tick(2.0); // age 17 -> 19
    let c = mgr.all().first().unwrap();
    assert_eq!(c.lifecycle, LifecycleStage::Worker); // was Student at 17, now Worker at 19
}

#[test]
fn old_citizens_die() {
    let mut mgr = CitizenManager::new();
    mgr.spawn(95.0, None); // very old
    mgr.tick(5.0); // age -> 100, should die
    assert_eq!(mgr.count(), 0);
}

#[test]
fn young_citizens_survive() {
    let mut mgr = CitizenManager::new();
    for _ in 0..10 {
        mgr.spawn(20.0, None);
    }
    mgr.tick(1.0);
    assert_eq!(mgr.count(), 10);
}
