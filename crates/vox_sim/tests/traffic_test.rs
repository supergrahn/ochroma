use vox_sim::traffic::{RoadSegmentTraffic, TrafficNetwork};

#[test]
fn greenshields_velocity_at_zero_density() {
    let seg = RoadSegmentTraffic::new(0, 1.0, 100.0, 60.0);
    assert!((seg.velocity() - 60.0).abs() < 0.01);
}

#[test]
fn greenshields_velocity_at_max_density() {
    let mut seg = RoadSegmentTraffic::new(0, 1.0, 100.0, 60.0);
    seg.density = 100.0;
    assert!((seg.velocity()).abs() < 0.01);
}

#[test]
fn greenshields_velocity_at_half_density() {
    let mut seg = RoadSegmentTraffic::new(0, 1.0, 100.0, 60.0);
    seg.density = 50.0;
    assert!((seg.velocity() - 30.0).abs() < 0.01);
}

#[test]
fn flow_is_density_times_velocity() {
    let mut seg = RoadSegmentTraffic::new(0, 1.0, 100.0, 60.0);
    seg.density = 50.0;
    assert!((seg.compute_flow() - 1500.0).abs() < 0.01);
}

#[test]
fn inject_vehicles_increases_density() {
    let mut net = TrafficNetwork::new();
    net.add_segment(RoadSegmentTraffic::new(0, 1.0, 200.0, 60.0));
    net.inject_vehicles(0, 50.0);
    assert!(net.segments[0].density > 0.0);
}

#[test]
fn tick_updates_flow() {
    let mut net = TrafficNetwork::new();
    net.add_segment(RoadSegmentTraffic::new(0, 1.0, 200.0, 60.0));
    net.inject_vehicles(0, 50.0);
    net.tick(1.0);
    assert!(net.segments[0].flow >= 0.0);
}
