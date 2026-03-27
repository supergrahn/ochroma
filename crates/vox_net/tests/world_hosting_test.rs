use vox_net::world_hosting::*;

#[test]
fn test_create_world() {
    let mut host = WorldHost::new();
    let id = host.create_world("MyWorld", "alice", 10).unwrap();
    let world = host.get_world(id).unwrap();
    assert_eq!(world.name, "MyWorld");
    assert_eq!(world.owner, "alice");
    assert_eq!(world.max_players, 10);
    assert_eq!(world.state, WorldState::Running);
    assert_eq!(world.player_count(), 0);
}

#[test]
fn test_duplicate_world_name_rejected() {
    let mut host = WorldHost::new();
    host.create_world("UniqueWorld", "alice", 10).unwrap();
    let result = host.create_world("UniqueWorld", "bob", 5);
    assert_eq!(result, Err(WorldHostError::NameAlreadyTaken));
}

#[test]
fn test_join_and_leave_world() {
    let mut host = WorldHost::new();
    let id = host.create_world("TestWorld", "owner", 4).unwrap();

    host.join_world(id, "player1").unwrap();
    host.join_world(id, "player2").unwrap();
    assert_eq!(host.get_world(id).unwrap().player_count(), 2);

    host.leave_world(id, "player1").unwrap();
    assert_eq!(host.get_world(id).unwrap().player_count(), 1);
}

#[test]
fn test_capacity_limits() {
    let mut host = WorldHost::new();
    let id = host.create_world("SmallWorld", "owner", 2).unwrap();

    host.join_world(id, "p1").unwrap();
    host.join_world(id, "p2").unwrap();
    assert_eq!(host.join_world(id, "p3"), Err(WorldHostError::WorldFull));
}

#[test]
fn test_player_already_in_world() {
    let mut host = WorldHost::new();
    let id = host.create_world("W", "owner", 10).unwrap();
    host.join_world(id, "p1").unwrap();
    assert_eq!(
        host.join_world(id, "p1"),
        Err(WorldHostError::PlayerAlreadyInWorld)
    );
}

#[test]
fn test_leave_nonexistent_player() {
    let mut host = WorldHost::new();
    let id = host.create_world("W", "owner", 10).unwrap();
    assert_eq!(
        host.leave_world(id, "ghost"),
        Err(WorldHostError::PlayerNotInWorld)
    );
}

#[test]
fn test_portal_creation() {
    let mut host = WorldHost::new();
    let w1 = host.create_world("World1", "alice", 10).unwrap();
    let w2 = host.create_world("World2", "bob", 10).unwrap();

    let portal_id = host
        .create_portal(w1, [0.0, 0.0, 0.0], w2, [100.0, 0.0, 50.0], "Gateway")
        .unwrap();

    let portals = host.portals_for_world(w1);
    assert_eq!(portals.len(), 1);
    assert_eq!(portals[0].id, portal_id);
    assert_eq!(portals[0].label, "Gateway");
    assert_eq!(portals[0].destination_position, [100.0, 0.0, 50.0]);

    // Portal also visible from destination world
    let portals_w2 = host.portals_for_world(w2);
    assert_eq!(portals_w2.len(), 1);
}

#[test]
fn test_portal_to_nonexistent_world() {
    let mut host = WorldHost::new();
    let w1 = host.create_world("World1", "alice", 10).unwrap();
    let fake_id = uuid::Uuid::new_v4();

    assert_eq!(
        host.create_portal(w1, [0.0; 3], fake_id, [0.0; 3], "Bad"),
        Err(WorldHostError::WorldNotFound)
    );
}

#[test]
fn test_list_worlds() {
    let mut host = WorldHost::new();
    host.create_world("A", "owner", 5).unwrap();
    host.create_world("B", "owner", 10).unwrap();
    host.create_world("C", "owner", 15).unwrap();

    let worlds = host.list_worlds();
    assert_eq!(worlds.len(), 3);
}

#[test]
fn test_world_stats() {
    let mut host = WorldHost::new();
    let w1 = host.create_world("StatsWorld", "alice", 8).unwrap();
    let w2 = host.create_world("Other", "bob", 4).unwrap();

    host.join_world(w1, "p1").unwrap();
    host.join_world(w1, "p2").unwrap();
    host.create_portal(w1, [0.0; 3], w2, [0.0; 3], "Portal1")
        .unwrap();

    let stats = host.world_stats(w1).unwrap();
    assert_eq!(stats.player_count, 2);
    assert_eq!(stats.max_players, 8);
    assert_eq!(stats.state, WorldState::Running);
    assert_eq!(stats.portal_count, 1);
}

#[test]
fn test_pause_and_stop_world() {
    let mut host = WorldHost::new();
    let id = host.create_world("PauseMe", "owner", 4).unwrap();

    host.pause_world(id).unwrap();
    assert_eq!(host.get_world(id).unwrap().state, WorldState::Paused);

    // Cannot join a paused world
    assert!(host.join_world(id, "player").is_err());

    host.stop_world(id).unwrap();
    assert_eq!(host.get_world(id).unwrap().state, WorldState::Stopped);
}

#[test]
fn test_join_nonexistent_world() {
    let mut host = WorldHost::new();
    let fake_id = uuid::Uuid::new_v4();
    assert_eq!(
        host.join_world(fake_id, "player"),
        Err(WorldHostError::WorldNotFound)
    );
}
