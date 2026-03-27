use glam::Vec3;
use vox_sim::vehicles::*;

#[test]
fn spawn_and_move_vehicle() {
    let mut mgr = VehicleManager::new(100);
    let _id = mgr.spawn(VehicleType::Car, Vec3::ZERO, vec![0, 1, 2]).unwrap();
    assert_eq!(mgr.count(), 1);
    mgr.tick(1.0);
    assert!(mgr.vehicles[0].speed > 0.0);
}

#[test]
fn vehicle_follows_route() {
    let mut mgr = VehicleManager::new(100);
    mgr.spawn(VehicleType::Car, Vec3::ZERO, vec![0, 1]).unwrap();
    // Tick enough to advance past first segment
    for _ in 0..200 {
        mgr.tick(0.1);
    }
    // Should have advanced to segment 1 or parked
    let v = &mgr.vehicles[0];
    assert!(v.route_index > 0 || v.parked);
}

#[test]
fn max_vehicles_respected() {
    let mut mgr = VehicleManager::new(2);
    assert!(mgr.spawn(VehicleType::Car, Vec3::ZERO, vec![0]).is_some());
    assert!(mgr.spawn(VehicleType::Bus, Vec3::ZERO, vec![0]).is_some());
    assert!(mgr.spawn(VehicleType::Truck, Vec3::ZERO, vec![0]).is_none());
}

#[test]
fn emergency_vehicle_faster() {
    let mut mgr = VehicleManager::new(100);
    mgr.spawn(VehicleType::Car, Vec3::ZERO, vec![0]);
    mgr.spawn(VehicleType::EmergencyVehicle, Vec3::ZERO, vec![0]);
    mgr.tick(2.0);
    assert!(mgr.vehicles[1].max_speed > mgr.vehicles[0].max_speed);
}
