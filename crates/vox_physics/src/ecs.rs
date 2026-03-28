//! bevy_ecs integration for vox_physics.

use bevy_ecs::prelude::*;
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};

use vox_core::ecs::{ColliderComponent, ColliderShape, TransformComponent};
use crate::rapier::RapierPhysicsWorld;

// RapierPhysicsWorld doesn't derive Resource — implement it manually.
// All fields are Send + Sync + 'static.
impl Resource for RapierPhysicsWorld {}

// ─── Components ─────────────────────────────────────────────────────────────

/// Body type intent — read by spawn_physics_bodies_system.
/// Dynamic = full rigid body physics. Static = immovable collider. Kinematic = position-driven.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysicsBodyTypeComponent {
    #[default]
    Dynamic,
    Static,
    Kinematic,
}

/// Output component written by spawn_physics_bodies_system once an entity is
/// registered in RapierPhysicsWorld. Presence signals registration is complete.
/// Static entities (collider-only) do NOT get this component.
#[derive(Component, Debug, Clone)]
pub struct PhysicsBodyComponent {
    pub body_handle:     RigidBodyHandle,
    pub collider_handle: ColliderHandle,
}

/// Marker inserted on static entities after their collider is registered,
/// so spawn_physics_bodies_system does not process them again each frame.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct StaticColliderRegistered;

// ─── Systems ─────────────────────────────────────────────────────────────────

/// Registers new entities (ColliderComponent + TransformComponent, no PhysicsBodyComponent yet)
/// into Rapier and attaches a PhysicsBodyComponent.
pub fn spawn_physics_bodies_system(
    mut commands: Commands,
    mut physics: ResMut<RapierPhysicsWorld>,
    query: Query<
        (Entity, &ColliderComponent, &TransformComponent, Option<&PhysicsBodyTypeComponent>),
        (Without<PhysicsBodyComponent>, Without<StaticColliderRegistered>),
    >,
) {
    for (entity, collider, transform, body_type) in query.iter() {
        let pos = [
            transform.position.x,
            transform.position.y,
            transform.position.z,
        ];

        let body_type = body_type.copied().unwrap_or_default();

        match body_type {
            PhysicsBodyTypeComponent::Static => {
                match &collider.shape {
                    ColliderShape::Box { half_extents } => {
                        physics.add_static_collider(pos, *half_extents);
                    }
                    ColliderShape::Sphere { radius } => {
                        physics.add_static_collider(pos, [*radius, *radius, *radius]);
                    }
                    ColliderShape::Capsule { radius, height: _ } => {
                        physics.add_static_collider(pos, [*radius, *radius, *radius]);
                    }
                }
                // Mark static entities so they're skipped on subsequent frames.
                commands.entity(entity).insert(StaticColliderRegistered);
            }

            PhysicsBodyTypeComponent::Dynamic | PhysicsBodyTypeComponent::Kinematic => {
                let (body_handle, collider_handle) = match &collider.shape {
                    ColliderShape::Box { half_extents } => {
                        physics.add_dynamic_box(pos, *half_extents, 1.0)
                    }
                    ColliderShape::Sphere { radius } => {
                        physics.add_dynamic_sphere(pos, *radius, 1.0)
                    }
                    ColliderShape::Capsule { radius, height } => {
                        physics.add_character_controller(pos, *radius, *height)
                    }
                };
                commands.entity(entity).insert(PhysicsBodyComponent {
                    body_handle,
                    collider_handle,
                });
            }
        }
    }
}

/// Advances the Rapier simulation by one fixed step.
pub fn physics_step_system(mut physics: ResMut<RapierPhysicsWorld>) {
    physics.step();
}

/// Reads Rapier body positions and writes them back to TransformComponent.
pub fn sync_transforms_system(
    physics: Res<RapierPhysicsWorld>,
    mut query: Query<(&PhysicsBodyComponent, &mut TransformComponent)>,
) {
    for (body, mut transform) in query.iter_mut() {
        if let Some([x, y, z]) = physics.body_position(body.body_handle) {
            transform.position.x = x;
            transform.position.y = y;
            transform.position.z = z;
        }
    }
}

// ─── Plugin ──────────────────────────────────────────────────────────────────

/// Bevy plugin that inserts the Rapier physics world and registers the three
/// physics systems in `Update`, chained in order.
pub struct PhysicsPlugin;

impl bevy_app::Plugin for PhysicsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(RapierPhysicsWorld::new());
        app.add_systems(
            bevy_app::Update,
            (
                spawn_physics_bodies_system,
                physics_step_system,
                sync_transforms_system,
            )
                .chain(),
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::schedule::Schedule;
    use vox_core::ecs::{ColliderComponent, ColliderShape, TransformComponent};

    fn make_world() -> (World, Schedule) {
        let mut world = World::new();
        world.insert_resource(RapierPhysicsWorld::new());
        let schedule = Schedule::default();
        (world, schedule)
    }

    #[test]
    fn body_type_default_is_dynamic() {
        assert_eq!(PhysicsBodyTypeComponent::default(), PhysicsBodyTypeComponent::Dynamic);
    }

    #[test]
    fn physics_body_component_is_component() {
        fn _assert_component<T: Component>() {}
        _assert_component::<PhysicsBodyComponent>();
    }

    #[test]
    fn spawn_system_registers_body_in_rapier() {
        let (mut world, mut schedule) = make_world();
        schedule.add_systems(spawn_physics_bodies_system);

        let mut transform = TransformComponent::default();
        transform.position.y = 5.0;

        world.spawn((
            ColliderComponent { shape: ColliderShape::Box { half_extents: [0.5, 0.5, 0.5] } },
            transform,
            PhysicsBodyTypeComponent::Dynamic,
        ));

        schedule.run(&mut world);

        // Entity should now have a PhysicsBodyComponent.
        let mut query = world.query::<&PhysicsBodyComponent>();
        assert!(query.iter(&world).count() >= 1, "entity missing PhysicsBodyComponent");

        // Rapier should have at least one body.
        let physics = world.resource::<RapierPhysicsWorld>();
        assert!(physics.body_count() >= 1, "rapier body_count should be >= 1");
    }

    #[test]
    fn step_and_sync_moves_transform() {
        let (mut world, mut schedule) = make_world();
        schedule.add_systems((
            spawn_physics_bodies_system,
            physics_step_system,
            sync_transforms_system,
        ).chain());

        // Spawn a dynamic sphere at y=10.
        let mut transform = TransformComponent::default();
        transform.position.y = 10.0;

        world.spawn((
            ColliderComponent { shape: ColliderShape::Sphere { radius: 0.5 } },
            transform,
            PhysicsBodyTypeComponent::Dynamic,
        ));

        // Run 60 steps (~1 second at 60 Hz); sphere should fall under gravity.
        for _ in 0..60 {
            schedule.run(&mut world);
        }

        let mut query = world.query::<&TransformComponent>();
        for t in query.iter(&world) {
            assert!(
                t.position.y < 8.0,
                "expected sphere to fall below y=8.0, got y={}",
                t.position.y
            );
        }
    }

    #[test]
    fn spawn_runs_once_per_entity() {
        let (mut world, mut schedule) = make_world();
        schedule.add_systems(spawn_physics_bodies_system);

        let mut transform = TransformComponent::default();
        transform.position.y = 0.0;

        world.spawn((
            ColliderComponent { shape: ColliderShape::Box { half_extents: [0.5, 0.5, 0.5] } },
            transform,
            PhysicsBodyTypeComponent::Dynamic,
        ));

        schedule.run(&mut world);
        let count_after_first = world.resource::<RapierPhysicsWorld>().body_count();

        schedule.run(&mut world);
        let count_after_second = world.resource::<RapierPhysicsWorld>().body_count();

        assert_eq!(
            count_after_first, count_after_second,
            "body_count changed between runs — spawn system ran twice for the same entity"
        );
    }
}
