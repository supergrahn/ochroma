use vox_sim::zoning::{ZoneType, ZoningManager};

#[test]
fn zone_plot_creates_undeveloped() {
    let mut mgr = ZoningManager::new();
    mgr.zone_plot(ZoneType::ResidentialLow, [0.0, 0.0], [10.0, 10.0]);
    assert_eq!(mgr.plot_count(), 1);
    assert_eq!(mgr.undeveloped_plots().len(), 1);
}

#[test]
fn develop_plot_removes_from_undeveloped() {
    let mut mgr = ZoningManager::new();
    let id = mgr.zone_plot(ZoneType::CommercialLocal, [0.0, 0.0], [10.0, 10.0]);
    mgr.develop_plot(id, 100);
    assert_eq!(mgr.undeveloped_plots().len(), 0);
}

#[test]
fn demand_increases_with_citizens() {
    let mut mgr = ZoningManager::new();
    mgr.zone_plot(ZoneType::ResidentialLow, [0.0, 0.0], [10.0, 10.0]);
    mgr.update_demand(1000);
    assert!(mgr.demand.residential > 0.0);
    assert!(mgr.demand.commercial > 0.0);
}
