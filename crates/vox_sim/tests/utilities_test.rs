use vox_sim::utilities::{UtilityNetwork, UtilityType};

#[test]
fn consumer_connected_to_source_is_served() {
    let mut net = UtilityNetwork::new(UtilityType::Water);
    let source = net.add_source([0.0, 0.0], 1000.0);
    let consumer = net.add_consumer([100.0, 0.0], 50.0);
    net.connect(source, consumer, 100.0);
    assert!(net.is_served(consumer));
}

#[test]
fn disconnected_consumer_not_served() {
    let mut net = UtilityNetwork::new(UtilityType::Power);
    net.add_source([0.0, 0.0], 1000.0);
    let consumer = net.add_consumer([100.0, 0.0], 50.0);
    // No connection
    assert!(!net.is_served(consumer));
}

#[test]
fn deficit_detection() {
    let mut net = UtilityNetwork::new(UtilityType::Sewage);
    net.add_source([0.0, 0.0], 100.0);
    net.add_consumer([10.0, 0.0], 200.0);
    assert!(net.has_deficit());
}

#[test]
fn no_deficit_when_capacity_exceeds_demand() {
    let mut net = UtilityNetwork::new(UtilityType::Water);
    net.add_source([0.0, 0.0], 1000.0);
    net.add_consumer([10.0, 0.0], 50.0);
    assert!(!net.has_deficit());
}
