use vox_sim::districts::{DistrictManager, DistrictPolicy};

#[test]
fn create_and_find_district() {
    let mut mgr = DistrictManager::new();
    mgr.create_district("Downtown", [0.0, 0.0], [100.0, 100.0]);
    assert!(mgr.district_at([50.0, 50.0]).is_some());
    assert!(mgr.district_at([200.0, 200.0]).is_none());
}

#[test]
fn policy_affects_tax() {
    let mut mgr = DistrictManager::new();
    let id = mgr.create_district("High Tax", [0.0, 0.0], [100.0, 100.0]);
    mgr.set_policy(id, DistrictPolicy { tax_modifier: 0.05, ..Default::default() });
    assert!((mgr.tax_modifier_at([50.0, 50.0]) - 0.05).abs() < 0.001);
}
