use vox_sim::buildings::{BuildingManager, BuildingType};

#[test]
fn add_and_find_building() {
    let mut mgr = BuildingManager::new();
    mgr.add_building(BuildingType::Residential, [0.0, 0.0], 10);
    mgr.add_building(BuildingType::Commercial, [100.0, 0.0], 5);
    assert_eq!(mgr.count(), 2);
    assert_eq!(mgr.total_housing(), 10);
    assert_eq!(mgr.total_jobs(), 5);
}

#[test]
fn find_nearest_with_vacancy() {
    let mut mgr = BuildingManager::new();
    mgr.add_building(BuildingType::Residential, [0.0, 0.0], 2);
    mgr.add_building(BuildingType::Residential, [100.0, 0.0], 2);
    let nearest = mgr.find_nearest_with_vacancy([10.0, 0.0], BuildingType::Residential);
    assert_eq!(nearest, Some(0));
}

#[test]
fn assign_occupant_respects_capacity() {
    let mut mgr = BuildingManager::new();
    mgr.add_building(BuildingType::Residential, [0.0, 0.0], 1);
    assert!(mgr.assign_occupant(0));
    assert!(!mgr.assign_occupant(0)); // full
}
