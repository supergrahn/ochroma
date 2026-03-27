use glam::Vec3;
use vox_sim::roads::{RoadNetwork, RoadSegment, RoadType};

#[test]
fn straight_road_length() {
    let seg = RoadSegment { id: 0, road_type: RoadType::LocalStreet, start: Vec3::ZERO, end: Vec3::new(100.0, 0.0, 0.0), control_point: None };
    assert!((seg.length() - 100.0).abs() < 0.5);
}

#[test]
fn bezier_road_sample_midpoint() {
    let seg = RoadSegment { id: 0, road_type: RoadType::Avenue, start: Vec3::ZERO, end: Vec3::new(100.0, 0.0, 0.0), control_point: Some(Vec3::new(50.0, 0.0, 30.0)) };
    let mid = seg.sample(0.5);
    assert!(mid.z > 0.0, "Bezier midpoint should be offset");
}

#[test]
fn add_roads_creates_segments() {
    let mut net = RoadNetwork::new();
    net.add_straight(RoadType::LocalStreet, Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0));
    net.add_straight(RoadType::LocalStreet, Vec3::ZERO, Vec3::new(0.0, 0.0, 100.0));
    assert_eq!(net.segment_count(), 2);
}

#[test]
fn intersecting_roads_create_intersection() {
    let mut net = RoadNetwork::new();
    net.add_straight(RoadType::LocalStreet, Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0));
    net.add_straight(RoadType::LocalStreet, Vec3::ZERO, Vec3::new(0.0, 0.0, 100.0));
    assert!(net.intersection_count() > 0, "Roads sharing an endpoint should create intersection");
}

#[test]
fn road_type_properties() {
    assert_eq!(RoadType::Highway.lanes(), 6);
    assert!(RoadType::Highway.width() > RoadType::LocalStreet.width());
}
