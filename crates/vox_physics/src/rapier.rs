//! Rapier3D physics integration for Ochroma.
//!
//! Provides a full rigid-body physics world with collision detection,
//! raycasting, character controllers, and force application.
//! Enable with: `cargo build --features rapier`

use rapier3d::prelude::*;

/// Real physics world using Rapier3D.
pub struct RapierPhysicsWorld {
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
}

impl RapierPhysicsWorld {
    pub fn new() -> Self {
        Self {
            gravity: vector![0.0, -9.81, 0.0],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
        }
    }

    /// Set custom gravity vector.
    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.gravity = vector![gravity[0], gravity[1], gravity[2]];
    }

    /// Add a static collider (floor, wall, etc.).
    pub fn add_static_collider(
        &mut self,
        position: [f32; 3],
        half_extents: [f32; 3],
    ) -> ColliderHandle {
        let collider = ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
            .translation(vector![position[0], position[1], position[2]])
            .build();
        self.collider_set.insert(collider)
    }

    /// Add a dynamic rigid body with a box collider.
    pub fn add_dynamic_box(
        &mut self,
        position: [f32; 3],
        half_extents: [f32; 3],
        mass: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![position[0], position[1], position[2]])
            .additional_mass(mass)
            .build();
        let body_handle = self.rigid_body_set.insert(body);

        let collider =
            ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2]).build();
        let collider_handle =
            self.collider_set
                .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);

        (body_handle, collider_handle)
    }

    /// Add a dynamic sphere.
    pub fn add_dynamic_sphere(
        &mut self,
        position: [f32; 3],
        radius: f32,
        mass: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![position[0], position[1], position[2]])
            .additional_mass(mass)
            .build();
        let body_handle = self.rigid_body_set.insert(body);

        let collider = ColliderBuilder::ball(radius).build();
        let collider_handle =
            self.collider_set
                .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);

        (body_handle, collider_handle)
    }

    /// Add a kinematic character controller body with a capsule collider.
    pub fn add_character_controller(
        &mut self,
        position: [f32; 3],
        radius: f32,
        height: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let body = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![position[0], position[1], position[2]])
            .build();
        let body_handle = self.rigid_body_set.insert(body);

        let collider = ColliderBuilder::capsule_y(height * 0.5, radius).build();
        let collider_handle =
            self.collider_set
                .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);

        (body_handle, collider_handle)
    }

    /// Step the physics simulation by one tick.
    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
        // Update query pipeline after each step for raycasting.
        self.query_pipeline.update(&self.collider_set);
    }

    /// Get body position as `[x, y, z]`.
    pub fn body_position(&self, handle: RigidBodyHandle) -> Option<[f32; 3]> {
        self.rigid_body_set.get(handle).map(|b| {
            let t = b.translation();
            [t.x, t.y, t.z]
        })
    }

    /// Get body linear velocity as `[x, y, z]`.
    pub fn body_velocity(&self, handle: RigidBodyHandle) -> Option<[f32; 3]> {
        self.rigid_body_set.get(handle).map(|b| {
            let v = b.linvel();
            [v.x, v.y, v.z]
        })
    }

    /// Set kinematic body next position (for character controllers).
    pub fn set_kinematic_position(&mut self, handle: RigidBodyHandle, position: [f32; 3]) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_next_kinematic_translation(vector![position[0], position[1], position[2]]);
        }
    }

    /// Apply a force to a dynamic body (accumulated until next step).
    pub fn apply_force(&mut self, handle: RigidBodyHandle, force: [f32; 3]) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.add_force(vector![force[0], force[1], force[2]], true);
        }
    }

    /// Apply an impulse to a dynamic body (instantaneous velocity change).
    pub fn apply_impulse(&mut self, handle: RigidBodyHandle, impulse: [f32; 3]) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.apply_impulse(vector![impulse[0], impulse[1], impulse[2]], true);
        }
    }

    /// Cast a ray and return the hit point and distance of the first intersection.
    pub fn raycast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_dist: f32,
    ) -> Option<([f32; 3], f32)> {
        let ray = Ray::new(
            point![origin[0], origin[1], origin[2]],
            vector![direction[0], direction[1], direction[2]],
        );

        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_dist,
                true,
                QueryFilter::default(),
            )
            .map(|(_handle, toi)| {
                let hit = ray.point_at(toi);
                ([hit.x, hit.y, hit.z], toi)
            })
    }

    /// Number of rigid bodies in the world.
    pub fn body_count(&self) -> usize {
        self.rigid_body_set.len()
    }

    /// Number of colliders in the world.
    pub fn collider_count(&self) -> usize {
        self.collider_set.len()
    }
}

impl Default for RapierPhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}
