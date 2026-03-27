use glam::Vec3;
use vox_physics::{Aabb, PhysicsWorld, RigidBody};

#[test]
fn aabb_intersection() {
    let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
    let b = Aabb::from_center_half_extents(Vec3::new(1.5, 0.0, 0.0), Vec3::splat(1.0));
    assert!(a.intersects(&b));
}

#[test]
fn aabb_no_intersection() {
    let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
    let b = Aabb::from_center_half_extents(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0));
    assert!(!a.intersects(&b));
}

#[test]
fn falling_body_hits_ground() {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBody {
        id: 0,
        position: Vec3::new(0.0, 10.0, 0.0),
        velocity: Vec3::ZERO,
        mass: 1.0,
        is_static: false,
    });
    for _ in 0..100 {
        world.step(0.016);
    }
    let body = world.get_body(1).unwrap();
    assert!(body.position.y <= 0.1, "Body should hit ground, y={}", body.position.y);
}

#[test]
fn static_body_doesnt_move() {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBody {
        id: 0,
        position: Vec3::new(0.0, 5.0, 0.0),
        velocity: Vec3::ZERO,
        mass: 1.0,
        is_static: true,
    });
    world.step(1.0);
    assert_eq!(world.get_body(1).unwrap().position.y, 5.0);
}

#[test]
fn collision_detection_finds_overlapping_bodies() {
    let mut world = PhysicsWorld::new();
    world.add_body_with_collider(
        RigidBody { id: 0, position: Vec3::ZERO, velocity: Vec3::ZERO, mass: 1.0, is_static: false },
        Vec3::splat(1.0),
    );
    world.add_body_with_collider(
        RigidBody { id: 0, position: Vec3::new(1.5, 0.0, 0.0), velocity: Vec3::ZERO, mass: 1.0, is_static: false },
        Vec3::splat(1.0),
    );
    let pairs = world.check_collisions();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn collision_response_separates_bodies() {
    let mut world = PhysicsWorld::new();
    world.add_body_with_collider(
        RigidBody { id: 0, position: Vec3::new(-0.5, 5.0, 0.0), velocity: Vec3::ZERO, mass: 1.0, is_static: true },
        Vec3::splat(1.0),
    );
    world.add_body_with_collider(
        RigidBody { id: 0, position: Vec3::new(0.5, 5.0, 0.0), velocity: Vec3::ZERO, mass: 1.0, is_static: false },
        Vec3::splat(1.0),
    );
    // Run enough steps so collision resolution pushes them apart
    for _ in 0..5 {
        world.step(0.016);
    }
    // After several steps the bodies should no longer deeply overlap (separation converges)
    // Verify body 2 has been pushed away from body 1
    let b1 = world.get_body(1).unwrap();
    let b2 = world.get_body(2).unwrap();
    let dist = (b2.position.x - b1.position.x).abs();
    assert!(dist >= 1.9, "Bodies should be separated, dist={}", dist);
}
