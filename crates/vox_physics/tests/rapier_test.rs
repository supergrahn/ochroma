use vox_physics::RapierPhysicsWorld;

#[test]
fn create_world() {
    let world = RapierPhysicsWorld::new();
    assert_eq!(world.body_count(), 0);
    assert_eq!(world.collider_count(), 0);
}

#[test]
fn ball_falls_to_ground() {
    let mut world = RapierPhysicsWorld::new();
    world.add_static_collider([0.0, -1.0, 0.0], [100.0, 1.0, 100.0]);
    let (ball, _) = world.add_dynamic_sphere([0.0, 10.0, 0.0], 0.5, 1.0);
    for _ in 0..120 {
        world.step();
    }
    let pos = world.body_position(ball).unwrap();
    assert!(pos[1] < 5.0, "Ball should have fallen: y={}", pos[1]);
    assert!(pos[1] > -1.5, "Ball should be above ground: y={}", pos[1]);
}

#[test]
fn dynamic_box_falls() {
    let mut world = RapierPhysicsWorld::new();
    world.add_static_collider([0.0, -1.0, 0.0], [50.0, 1.0, 50.0]);
    let (bx, _) = world.add_dynamic_box([0.0, 5.0, 0.0], [0.5, 0.5, 0.5], 2.0);
    for _ in 0..60 {
        world.step();
    }
    let pos = world.body_position(bx).unwrap();
    assert!(pos[1] < 5.0, "Box should have fallen: y={}", pos[1]);
}

#[test]
fn character_controller() {
    let mut world = RapierPhysicsWorld::new();
    let (char_handle, _) = world.add_character_controller([0.0, 1.0, 0.0], 0.3, 1.8);
    world.set_kinematic_position(char_handle, [5.0, 1.0, 0.0]);
    world.step();
    let pos = world.body_position(char_handle).unwrap();
    assert!((pos[0] - 5.0).abs() < 0.1, "Character x={}", pos[0]);
}

#[test]
fn apply_force_changes_velocity() {
    let mut world = RapierPhysicsWorld::new();
    let (body, _) = world.add_dynamic_box([0.0, 5.0, 0.0], [0.5, 0.5, 0.5], 1.0);
    world.apply_force(body, [100.0, 0.0, 0.0]);
    world.step();
    let vel = world.body_velocity(body).unwrap();
    assert!(vel[0] > 0.0, "Body should have positive x velocity");
}

#[test]
fn raycast_hits_static() {
    let mut world = RapierPhysicsWorld::new();
    world.add_static_collider([0.0, 0.0, 0.0], [10.0, 0.5, 10.0]);
    world.step();
    let result = world.raycast([0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0);
    assert!(result.is_some(), "Ray should hit the ground collider");
    let (hit, dist) = result.unwrap();
    assert!(dist < 10.5, "Hit distance should be ~9.5, got {}", dist);
    assert!(hit[1].abs() < 1.0, "Hit y should be near 0, got {}", hit[1]);
}
