//! bevy_ecs integration for vox_physics.

use bevy_ecs::prelude::*;
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_type_default_is_dynamic() {
        assert_eq!(PhysicsBodyTypeComponent::default(), PhysicsBodyTypeComponent::Dynamic);
    }

    #[test]
    fn physics_body_component_is_component() {
        fn _assert_component<T: Component>() {}
        _assert_component::<PhysicsBodyComponent>();
    }
}
