use vox_sim::citizen::{CitizenManager, DailyState, LifecycleStage};
use vox_sim::routines::update_routines;

#[test]
fn workers_commute_in_morning() {
    let mut mgr = CitizenManager::new();
    let id = mgr.spawn(30.0, Some(1));
    if let Some(c) = mgr.all_mut().iter_mut().find(|c| c.id == id) {
        c.workplace = Some(10);
    }

    let _movements = update_routines(mgr.all_mut(), 7.5);
    let citizen = mgr.get(id).unwrap();
    assert_eq!(citizen.daily_state, DailyState::Commuting);
}

#[test]
fn workers_return_in_evening() {
    let mut mgr = CitizenManager::new();
    let id = mgr.spawn(30.0, Some(1));
    if let Some(c) = mgr.all_mut().iter_mut().find(|c| c.id == id) {
        c.workplace = Some(10);
        c.daily_state = DailyState::AtWork;
    }

    let _ = update_routines(mgr.all_mut(), 17.5);
    let citizen = mgr.get(id).unwrap();
    assert_eq!(citizen.daily_state, DailyState::Returning);
}

#[test]
fn children_stay_home() {
    let mut mgr = CitizenManager::new();
    mgr.spawn(5.0, Some(1)); // child
    let _ = update_routines(mgr.all_mut(), 8.0);
    let citizen = mgr.all().first().unwrap();
    assert_eq!(citizen.daily_state, DailyState::AtHome);
}
