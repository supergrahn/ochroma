use vox_sim::services::{ServiceManager, ServiceType};

#[test]
fn place_school_covers_area() {
    let mut mgr = ServiceManager::new();
    mgr.place_service(ServiceType::PrimarySchool, [0.0, 0.0]);
    assert!(mgr.is_covered([500.0, 0.0], ServiceType::PrimarySchool));
    assert!(!mgr.is_covered([2000.0, 0.0], ServiceType::PrimarySchool));
}

#[test]
fn total_cost_sums_all() {
    let mut mgr = ServiceManager::new();
    mgr.place_service(ServiceType::Clinic, [0.0, 0.0]);
    mgr.place_service(ServiceType::FireStation, [100.0, 0.0]);
    assert!(mgr.total_cost() > 0.0);
}
