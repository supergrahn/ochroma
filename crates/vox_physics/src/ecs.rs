//! bevy_ecs integration for vox_physics.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};
use std::sync::Arc;

use crate::rapier::RapierPhysicsWorld;
use crate::sdf::{
    SdfAssetId, SdfColliderSet, SdfVolume, SdfVolumeCache, SdfVolumeDesc, SdfVolumeError,
};
use vox_core::ecs::{ColliderComponent, ColliderShape, TransformComponent};

// RapierPhysicsWorld doesn't derive Resource — implement it manually.
// All fields are Send + Sync + 'static.
impl Resource for RapierPhysicsWorld {}

const SDF_UNIFORM_SCALE_EPSILON: f32 = 1.0e-4;

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
    pub body_handle: RigidBodyHandle,
    pub collider_handle: ColliderHandle,
}

/// Marker inserted on static entities after their collider is registered,
/// so spawn_physics_bodies_system does not process them again each frame.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct StaticColliderRegistered;

/// Sampled SDF collider intent. The decoded asset-local SDF volume lives in
/// [`SdfColliderRuntime`]'s cache; entities only carry the asset id and local
/// offset from their [`TransformComponent`].
#[derive(Component, Debug, Clone)]
pub struct SdfColliderComponent {
    pub asset_id: SdfAssetId,
    pub local_position: Vec3,
    pub local_rotation: Quat,
    pub uniform_scale: f32,
}

impl SdfColliderComponent {
    pub fn new(asset_id: SdfAssetId) -> Self {
        Self {
            asset_id,
            local_position: Vec3::ZERO,
            local_rotation: Quat::IDENTITY,
            uniform_scale: 1.0,
        }
    }

    pub fn with_local_transform(
        mut self,
        local_position: Vec3,
        local_rotation: Quat,
        uniform_scale: f32,
    ) -> Self {
        self.local_position = local_position;
        self.local_rotation = local_rotation;
        self.uniform_scale = uniform_scale.max(SDF_UNIFORM_SCALE_EPSILON);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SdfColliderDiagnostics {
    pub registered_instances: usize,
    pub missing_asset_ids: Vec<SdfAssetId>,
    pub skipped_non_uniform_scale: usize,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct SdfColliderRuntime {
    volume_cache: SdfVolumeCache,
    colliders: SdfColliderSet,
    diagnostics: SdfColliderDiagnostics,
}

impl SdfColliderRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn volume_cache(&self) -> &SdfVolumeCache {
        &self.volume_cache
    }

    pub fn volume_cache_mut(&mut self) -> &mut SdfVolumeCache {
        &mut self.volume_cache
    }

    pub fn colliders(&self) -> &SdfColliderSet {
        &self.colliders
    }

    pub fn diagnostics(&self) -> &SdfColliderDiagnostics {
        &self.diagnostics
    }

    pub fn insert_volume(
        &mut self,
        asset_id: SdfAssetId,
        volume: Arc<SdfVolume>,
    ) -> Option<Arc<SdfVolume>> {
        self.volume_cache.insert(asset_id, volume)
    }

    pub fn insert_snorm16_volume(
        &mut self,
        asset_id: SdfAssetId,
        desc: SdfVolumeDesc,
        distances_snorm16: &[i16],
    ) -> Result<Arc<SdfVolume>, SdfVolumeError> {
        self.volume_cache
            .get_or_insert_snorm16_grid(asset_id, desc, distances_snorm16)
    }
}

// ─── Systems ─────────────────────────────────────────────────────────────────

/// Registers new entities (ColliderComponent + TransformComponent, no PhysicsBodyComponent yet)
/// into Rapier and attaches a PhysicsBodyComponent.
#[allow(clippy::type_complexity)]
pub fn spawn_physics_bodies_system(
    mut commands: Commands,
    mut physics: ResMut<RapierPhysicsWorld>,
    query: Query<
        (
            Entity,
            &ColliderComponent,
            &TransformComponent,
            Option<&PhysicsBodyTypeComponent>,
        ),
        (
            Without<PhysicsBodyComponent>,
            Without<StaticColliderRegistered>,
        ),
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
                        physics.add_static_sphere(pos, *radius);
                    }
                    ColliderShape::Capsule { radius, height } => {
                        physics.add_static_capsule(pos, *radius, *height);
                    }
                }
                // Mark static entities so they're skipped on subsequent frames.
                commands.entity(entity).insert(StaticColliderRegistered);
            }

            PhysicsBodyTypeComponent::Dynamic => {
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

            PhysicsBodyTypeComponent::Kinematic => {
                let (body_handle, collider_handle) = match &collider.shape {
                    ColliderShape::Box { half_extents } => {
                        physics.add_kinematic_box(pos, *half_extents, 1.0)
                    }
                    ColliderShape::Sphere { radius } => {
                        physics.add_kinematic_sphere(pos, *radius, 1.0)
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
        if let Some(r) = physics.body_rotation(body.body_handle) {
            transform.rotation = glam::Quat::from_xyzw(r[0], r[1], r[2], r[3]);
        }
    }
}

/// Rebuilds the sampled resident SDF set from ECS transforms. This does not
/// talk to Rapier: it maintains the shared SDF query/projection set that
/// placement, particles, soft bodies, and debug render paths can sample.
pub fn sync_sdf_colliders_system(
    mut sdf_runtime: ResMut<SdfColliderRuntime>,
    query: Query<(&SdfColliderComponent, &TransformComponent)>,
) {
    let mut colliders = SdfColliderSet::new();
    let mut diagnostics = SdfColliderDiagnostics::default();

    for (collider, transform) in query.iter() {
        let Some(transform_scale) = uniform_transform_scale(transform.scale) else {
            diagnostics.skipped_non_uniform_scale += 1;
            continue;
        };
        let Some(volume) = sdf_runtime.volume_cache.get(collider.asset_id) else {
            diagnostics.missing_asset_ids.push(collider.asset_id);
            continue;
        };

        let scale = transform_scale * collider.uniform_scale.max(SDF_UNIFORM_SCALE_EPSILON);
        let position =
            transform.position + transform.rotation * (collider.local_position * transform_scale);
        let rotation = transform.rotation * collider.local_rotation;
        colliders.add_instance(collider.asset_id, volume, position, rotation, scale);
        diagnostics.registered_instances += 1;
    }

    sdf_runtime.colliders = colliders;
    sdf_runtime.diagnostics = diagnostics;
}

fn uniform_transform_scale(scale: Vec3) -> Option<f32> {
    let scale = scale.abs();
    let uniform = scale.x;
    ((scale.x - scale.y).abs() <= SDF_UNIFORM_SCALE_EPSILON
        && (scale.x - scale.z).abs() <= SDF_UNIFORM_SCALE_EPSILON
        && uniform.is_finite()
        && uniform > SDF_UNIFORM_SCALE_EPSILON)
        .then_some(uniform)
}

// ─── Plugin ──────────────────────────────────────────────────────────────────

/// Bevy plugin that inserts the Rapier physics world and registers the three
/// physics systems in `Update`, chained in order.
pub struct PhysicsPlugin;

impl bevy_app::Plugin for PhysicsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(RapierPhysicsWorld::new());
        app.insert_resource(SdfColliderRuntime::new());
        app.add_systems(
            bevy_app::Update,
            (
                spawn_physics_bodies_system,
                sync_sdf_colliders_system,
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
    use crate::sdf::SdfVolume;
    use bevy_ecs::schedule::Schedule;
    use std::sync::Arc;
    use vox_core::ecs::{ColliderComponent, ColliderShape, TransformComponent};

    fn make_world() -> (World, Schedule) {
        let mut world = World::new();
        world.insert_resource(RapierPhysicsWorld::new());
        let schedule = Schedule::default();
        (world, schedule)
    }

    #[test]
    fn body_type_default_is_dynamic() {
        assert_eq!(
            PhysicsBodyTypeComponent::default(),
            PhysicsBodyTypeComponent::Dynamic
        );
    }

    #[test]
    fn physics_body_component_is_component() {
        fn _assert_component<T: Component>() {}
        _assert_component::<PhysicsBodyComponent>();
    }

    #[test]
    fn sdf_collider_component_is_component() {
        fn _assert_component<T: Component>() {}
        _assert_component::<SdfColliderComponent>();
    }

    #[test]
    fn spawn_system_registers_body_in_rapier() {
        let (mut world, mut schedule) = make_world();
        schedule.add_systems(spawn_physics_bodies_system);

        let mut transform = TransformComponent::default();
        transform.position.y = 5.0;

        world.spawn((
            ColliderComponent {
                shape: ColliderShape::Box {
                    half_extents: [0.5, 0.5, 0.5],
                },
            },
            transform,
            PhysicsBodyTypeComponent::Dynamic,
        ));

        schedule.run(&mut world);

        // Entity should now have a PhysicsBodyComponent.
        let mut query = world.query::<&PhysicsBodyComponent>();
        assert!(
            query.iter(&world).count() >= 1,
            "entity missing PhysicsBodyComponent"
        );

        // Rapier should have at least one body.
        let physics = world.resource::<RapierPhysicsWorld>();
        assert!(
            physics.body_count() >= 1,
            "rapier body_count should be >= 1"
        );
    }

    #[test]
    fn step_and_sync_moves_transform() {
        let (mut world, mut schedule) = make_world();
        schedule.add_systems(
            (
                spawn_physics_bodies_system,
                physics_step_system,
                sync_transforms_system,
            )
                .chain(),
        );

        // Spawn a dynamic sphere at y=10.
        let mut transform = TransformComponent::default();
        transform.position.y = 10.0;

        world.spawn((
            ColliderComponent {
                shape: ColliderShape::Sphere { radius: 0.5 },
            },
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
            ColliderComponent {
                shape: ColliderShape::Box {
                    half_extents: [0.5, 0.5, 0.5],
                },
            },
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

    #[test]
    fn sync_sdf_colliders_registers_cached_volume() {
        let mut world = World::new();
        let mut runtime = SdfColliderRuntime::new();
        let asset_id = SdfAssetId(11);
        runtime.insert_volume(asset_id, test_box_volume());
        world.insert_resource(runtime);

        let mut transform = TransformComponent::default();
        transform.position = Vec3::new(5.0, 0.0, -2.0);
        world.spawn((SdfColliderComponent::new(asset_id), transform));

        let mut schedule = Schedule::default();
        schedule.add_systems(sync_sdf_colliders_system);
        schedule.run(&mut world);

        let runtime = world.resource::<SdfColliderRuntime>();
        assert_eq!(runtime.colliders().len(), 1);
        assert_eq!(runtime.diagnostics().registered_instances, 1);
        assert!(runtime.diagnostics().missing_asset_ids.is_empty());
        assert!(
            runtime
                .colliders()
                .sample_world(Vec3::new(5.0, 0.0, -2.0))
                .is_some(),
            "registered SDF should answer world queries"
        );
    }

    #[test]
    fn sync_sdf_colliders_reuses_cached_volume_for_multiple_entities() {
        let mut world = World::new();
        let mut runtime = SdfColliderRuntime::new();
        let asset_id = SdfAssetId(12);
        runtime.insert_volume(asset_id, test_box_volume());
        world.insert_resource(runtime);

        let first = TransformComponent::default();
        let mut second = TransformComponent::default();
        second.position.x = 4.0;
        world.spawn((SdfColliderComponent::new(asset_id), first));
        world.spawn((SdfColliderComponent::new(asset_id), second));

        let mut schedule = Schedule::default();
        schedule.add_systems(sync_sdf_colliders_system);
        schedule.run(&mut world);

        let runtime = world.resource::<SdfColliderRuntime>();
        assert_eq!(runtime.volume_cache().len(), 1);
        assert_eq!(runtime.colliders().len(), 2);
        assert!(Arc::ptr_eq(
            &runtime.colliders().instances()[0].volume,
            &runtime.colliders().instances()[1].volume
        ));
    }

    #[test]
    fn sync_sdf_colliders_reports_missing_cached_volume() {
        let mut world = World::new();
        let asset_id = SdfAssetId(13);
        world.insert_resource(SdfColliderRuntime::new());
        world.spawn((
            SdfColliderComponent::new(asset_id),
            TransformComponent::default(),
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(sync_sdf_colliders_system);
        schedule.run(&mut world);

        let runtime = world.resource::<SdfColliderRuntime>();
        assert_eq!(runtime.colliders().len(), 0);
        assert_eq!(runtime.diagnostics().missing_asset_ids, vec![asset_id]);
    }

    #[test]
    fn sync_sdf_colliders_skips_non_uniform_scale() {
        let mut world = World::new();
        let mut runtime = SdfColliderRuntime::new();
        let asset_id = SdfAssetId(14);
        runtime.insert_volume(asset_id, test_box_volume());
        world.insert_resource(runtime);

        let mut transform = TransformComponent::default();
        transform.scale = Vec3::new(1.0, 2.0, 1.0);
        world.spawn((SdfColliderComponent::new(asset_id), transform));

        let mut schedule = Schedule::default();
        schedule.add_systems(sync_sdf_colliders_system);
        schedule.run(&mut world);

        let runtime = world.resource::<SdfColliderRuntime>();
        assert_eq!(runtime.colliders().len(), 0);
        assert_eq!(runtime.diagnostics().skipped_non_uniform_scale, 1);
    }

    fn test_box_volume() -> Arc<SdfVolume> {
        let resolution = [3, 3, 3];
        let origin = Vec3::splat(-1.0);
        let voxel_size = 1.0;
        let narrow_band = 2.0;
        let mut distances = Vec::new();
        for z in 0..resolution[2] {
            for y in 0..resolution[1] {
                for x in 0..resolution[0] {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                    let d = p.x.abs().max(p.y.abs()).max(p.z.abs()) - 0.5;
                    distances.push((d / narrow_band * i16::MAX as f32).round() as i16);
                }
            }
        }

        Arc::new(
            SdfVolume::from_snorm16_grid(
                SdfVolumeDesc {
                    resolution,
                    origin,
                    voxel_size,
                    narrow_band,
                },
                &distances,
            )
            .unwrap(),
        )
    }
}
