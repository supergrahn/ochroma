use vox_sim::transport::{TransportManager, TransportType};

#[test]
fn create_bus_route_with_stops() {
    let mut mgr = TransportManager::new();
    let id = mgr.create_route(TransportType::Bus, 10.0, 3);
    mgr.add_stop(id, [0.0, 0.0], "Central");
    mgr.add_stop(id, [1000.0, 0.0], "East");
    assert_eq!(mgr.routes[0].stops.len(), 2);
}

#[test]
fn route_length_calculation() {
    let mut mgr = TransportManager::new();
    let id = mgr.create_route(TransportType::Bus, 10.0, 1);
    mgr.add_stop(id, [0.0, 0.0], "A");
    mgr.add_stop(id, [1000.0, 0.0], "B");
    assert!((mgr.routes[0].route_length() - 1000.0).abs() < 1.0);
}

#[test]
fn travel_time_positive() {
    let mut mgr = TransportManager::new();
    let id = mgr.create_route(TransportType::Metro, 5.0, 2);
    mgr.add_stop(id, [0.0, 0.0], "A");
    mgr.add_stop(id, [5000.0, 0.0], "B");
    assert!(mgr.routes[0].travel_time_minutes() > 0.0);
}

#[test]
fn hourly_revenue_scales_with_vehicles() {
    let mut mgr = TransportManager::new();
    let id1 = mgr.create_route(TransportType::Bus, 10.0, 1);
    mgr.add_stop(id1, [0.0, 0.0], "A");
    let id2 = mgr.create_route(TransportType::Bus, 10.0, 3);
    mgr.add_stop(id2, [0.0, 0.0], "A");
    let rev1 = mgr.routes[0].hourly_revenue(2.0, 0.5);
    let rev3 = mgr.routes[1].hourly_revenue(2.0, 0.5);
    assert!((rev3 / rev1 - 3.0).abs() < 0.1);
}
