use vox_sim::agent::AgentManager;
use vox_core::lwc::WorldCoord;

#[test]
fn test_agent_moves_toward_destination() {
    let mut mgr = AgentManager::new();
    let start = WorldCoord::from_absolute(0.0, 0.0, 0.0);
    let dest = WorldCoord::from_absolute(10.0, 0.0, 0.0);

    let id = mgr.spawn(start, 5.0); // speed = 5 m/s
    mgr.get_mut(id).unwrap().destination = Some(dest);

    // tick 1 second — agent should move 5m toward destination
    mgr.tick(1.0);

    let agent = mgr.get(id).unwrap();
    let (ax, _, _) = agent.position.to_absolute();
    assert!((ax - 5.0).abs() < 1e-3, "Agent should have moved to x=5, got {}", ax);
    // Destination still set since we haven't arrived yet
    assert!(agent.destination.is_some(), "Destination should still be set");
}

#[test]
fn test_agent_stops_at_destination() {
    let mut mgr = AgentManager::new();
    let start = WorldCoord::from_absolute(0.0, 0.0, 0.0);
    let dest = WorldCoord::from_absolute(3.0, 0.0, 0.0);

    let id = mgr.spawn(start, 5.0); // speed = 5 m/s
    mgr.get_mut(id).unwrap().destination = Some(dest);

    // tick 2 seconds — agent would overshoot (10m) but should clamp to dest (3m)
    mgr.tick(2.0);

    let agent = mgr.get(id).unwrap();
    let (ax, ay, az) = agent.position.to_absolute();
    assert!((ax - 3.0).abs() < 1e-3, "Agent x should be 3.0 at destination, got {}", ax);
    assert!((ay - 0.0).abs() < 1e-3, "Agent y should be 0.0, got {}", ay);
    assert!((az - 0.0).abs() < 1e-3, "Agent z should be 0.0, got {}", az);
    assert!(agent.destination.is_none(), "Destination should be cleared on arrival");
    assert_eq!(agent.velocity.length(), 0.0, "Velocity should be zero after arrival");
}

#[test]
fn test_agent_count() {
    let mut mgr = AgentManager::new();
    assert_eq!(mgr.count(), 0);
    let pos = WorldCoord::from_absolute(0.0, 0.0, 0.0);
    mgr.spawn(pos, 1.0);
    mgr.spawn(pos, 2.0);
    assert_eq!(mgr.count(), 2);
}

#[test]
fn test_agent_no_destination_stays_put() {
    let mut mgr = AgentManager::new();
    let pos = WorldCoord::from_absolute(100.0, 50.0, 200.0);
    let id = mgr.spawn(pos, 5.0);

    mgr.tick(10.0);

    let agent = mgr.get(id).unwrap();
    let (ax, ay, az) = agent.position.to_absolute();
    assert!((ax - 100.0).abs() < 1e-3);
    assert!((ay - 50.0).abs() < 1e-3);
    assert!((az - 200.0).abs() < 1e-3);
}
