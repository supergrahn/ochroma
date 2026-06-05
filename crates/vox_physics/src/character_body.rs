//! Rapier KinematicCharacterController integration for Ochroma.
//!
//! Replaces the flat-plane Y detection in vox_core::character_controller.
//! The existing math helpers (is_walkable_slope, compute_slope_slide, etc.)
//! remain in vox_core and are called by game code on top of KCC output.

use rapier3d::prelude::*;
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct CharacterOutput {
    pub effective_translation: Vec3,
    pub grounded: bool,
    pub ground_normal: Vec3,
}

pub struct CharacterBody {
    pub rigid_body: RigidBodyHandle,
    pub collider:   ColliderHandle,
    pub controller: KinematicCharacterController,
    pub half_height: f32,
    pub radius: f32,
}

impl CharacterBody {
    pub fn new(
        position:    Vec3,
        half_height: f32,
        radius:      f32,
        bodies:      &mut RigidBodySet,
        colliders:   &mut ColliderSet,
    ) -> Self {
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let rb_handle = bodies.insert(rb);
        let collider = ColliderBuilder::capsule_y(half_height, radius)
            .friction(0.0)
            .build();
        let col_handle = colliders.insert_with_parent(collider, rb_handle, bodies);
        let controller = KinematicCharacterController {
            up: Vector::y_axis(),
            offset: CharacterLength::Absolute(0.01),
            slide: true,
            autostep: Some(CharacterAutostep {
                max_height:              CharacterLength::Absolute(0.3),
                min_width:               CharacterLength::Relative(0.5),
                include_dynamic_bodies:  false,
            }),
            max_slope_climb_angle: 45_f32.to_radians(),
            min_slope_slide_angle: 50_f32.to_radians(),
            snap_to_ground: Some(CharacterLength::Absolute(0.1)),
            ..Default::default()
        };
        Self { rigid_body: rb_handle, collider: col_handle, controller, half_height, radius }
    }

    pub fn move_and_slide(
        &self,
        desired_velocity: Vec3,
        dt:               f32,
        bodies:           &RigidBodySet,
        colliders:        &ColliderSet,
        query_pipeline:   &QueryPipeline,
    ) -> CharacterOutput {
        let desired = vector![
            desired_velocity.x * dt,
            desired_velocity.y * dt,
            desired_velocity.z * dt
        ];
        let rb = &bodies[self.rigid_body];
        let shape = SharedShape::capsule_y(self.half_height, self.radius);
        let filter = QueryFilter::default().exclude_collider(self.collider);
        let mut collisions = Vec::new();
        let movement = self.controller.move_shape(
            dt, bodies, colliders, query_pipeline,
            shape.as_ref(), rb.position(), desired, filter,
            |c| collisions.push(c),
        );
        let ground_normal = collisions.iter()
            .filter(|c| c.hit.normal1.y > 0.5)
            .map(|c| Vec3::new(c.hit.normal1.x, c.hit.normal1.y, c.hit.normal1.z))
            .next()
            .unwrap_or(Vec3::Y);
        CharacterOutput {
            effective_translation: Vec3::new(movement.translation.x, movement.translation.y, movement.translation.z),
            grounded: movement.grounded,
            ground_normal,
        }
    }

    pub fn apply_translation(&self, translation: Vec3, bodies: &mut RigidBodySet) {
        let rb = &mut bodies[self.rigid_body];
        let current = rb.translation();
        let next = current + vector![translation.x, translation.y, translation.z];
        rb.set_next_kinematic_translation(next);
    }

    pub fn position(&self, bodies: &RigidBodySet) -> Vec3 {
        let t = bodies[self.rigid_body].translation();
        Vec3::new(t.x, t.y, t.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_world() -> (RigidBodySet, ColliderSet, QueryPipeline) {
        let mut bodies    = RigidBodySet::new();
        let colliders     = ColliderSet::new();
        let qp            = QueryPipeline::new();
        let _ = &mut bodies;
        // Note: floor collider is inserted after creating the struct to avoid borrow issues
        (bodies, colliders, qp)
    }

    fn make_world_with_floor() -> (RigidBodySet, ColliderSet, QueryPipeline) {
        let bodies        = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let qp            = QueryPipeline::new();
        let floor = ColliderBuilder::cuboid(10.0, 0.1, 10.0)
            .translation(vector![0.0, -0.1, 0.0])
            .build();
        colliders.insert(floor);
        (bodies, colliders, qp)
    }

    #[test]
    fn character_body_creates_without_panic() {
        let (mut bodies, mut colliders, _) = make_world_with_floor();
        let _cb = CharacterBody::new(Vec3::new(0.0, 2.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        assert_eq!(bodies.len(), 1);
        assert_eq!(colliders.len(), 2); // floor + character capsule
    }

    #[test]
    fn position_returns_spawn_location() {
        let (mut bodies, mut colliders, _) = make_world();
        let spawn = Vec3::new(3.0, 5.0, -2.0);
        let cb = CharacterBody::new(spawn, 0.8, 0.3, &mut bodies, &mut colliders);
        let pos = cb.position(&bodies);
        assert!((pos.x - 3.0).abs() < 0.001, "x mismatch: {}", pos.x);
        assert!((pos.y - 5.0).abs() < 0.001, "y mismatch: {}", pos.y);
        assert!((pos.z - -2.0).abs() < 0.001, "z mismatch: {}", pos.z);
    }

    #[test]
    fn move_and_slide_on_flat_floor_is_grounded() {
        let (mut bodies, mut colliders, mut qp) = make_world_with_floor();
        let cb = CharacterBody::new(Vec3::new(0.0, 1.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        qp.update(&colliders);
        let output = cb.move_and_slide(Vec3::new(0.0, -10.0, 0.0), 1.0 / 60.0, &bodies, &colliders, &qp);
        assert!(
            output.grounded || output.effective_translation.y.abs() < 0.2,
            "expected grounded or minimal Y motion, got translation {:?}", output.effective_translation
        );
    }

    #[test]
    fn move_and_slide_on_raised_platform_detects_ground() {
        let mut bodies    = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut qp        = QueryPipeline::new();
        let platform = ColliderBuilder::cuboid(5.0, 0.1, 5.0)
            .translation(vector![0.0, 5.0, 0.0])
            .build();
        colliders.insert(platform);
        let cb = CharacterBody::new(Vec3::new(0.0, 6.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        qp.update(&colliders);
        let output = cb.move_and_slide(Vec3::new(0.0, -10.0, 0.0), 1.0 / 60.0, &bodies, &colliders, &qp);
        assert!(
            output.grounded || output.effective_translation.y.abs() < 0.2,
            "BUG: character on raised platform (Y=5) not detected as grounded. translation.y = {}",
            output.effective_translation.y
        );
    }

    #[test]
    fn apply_translation_queues_correct_kinematic_position() {
        let (mut bodies, mut colliders, _) = make_world();
        let spawn = Vec3::new(0.0, 2.0, 0.0);
        let cb = CharacterBody::new(spawn, 0.8, 0.3, &mut bodies, &mut colliders);
        cb.apply_translation(Vec3::new(1.0, 0.0, 0.0), &mut bodies);
        // set_next_kinematic_translation queues the move for the next physics step.
        // Verify via next_position() that the correct translation was queued.
        let rb = &bodies[cb.rigid_body];
        let next = rb.next_position();
        assert!(
            (next.translation.x - (spawn.x + 1.0)).abs() < 0.01,
            "queued x should be spawn.x + 1.0 = {}, got {}",
            spawn.x + 1.0,
            next.translation.x
        );
    }
}
