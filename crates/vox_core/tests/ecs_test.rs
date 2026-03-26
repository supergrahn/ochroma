use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use uuid::Uuid;
use vox_core::ecs::{LodLevel, SplatInstanceComponent};

#[test]
fn can_spawn_splat_instance() {
    let mut world = World::new();
    let entity = world
        .spawn(SplatInstanceComponent {
            asset_uuid: Uuid::new_v4(),
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            instance_id: 42,
            lod: LodLevel::Full,
        })
        .id();
    let inst = world.get::<SplatInstanceComponent>(entity).unwrap();
    assert_eq!(inst.instance_id, 42);
}

#[test]
fn can_query_instances() {
    let mut world = World::new();
    for i in 0..100 {
        world.spawn(SplatInstanceComponent {
            asset_uuid: Uuid::new_v4(),
            position: Vec3::new(i as f32, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            instance_id: i,
            lod: LodLevel::Full,
        });
    }
    let mut query = world.query::<&SplatInstanceComponent>();
    assert_eq!(query.iter(&world).count(), 100);
}
