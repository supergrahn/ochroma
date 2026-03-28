//! ECS integration for LOD streaming.
//!
//! Provides `LodStreamingPlugin` which wires the existing HierarchicalLOD,
//! LodCrossfadeManager, and TileManager systems into bevy_ecs.

use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::lod_crossfade::LodCrossfadeManager;
use crate::streaming::TileManager;

// Allow these types to be stored as bevy_ecs Resources.
impl Resource for LodCrossfadeManager {}
impl Resource for TileManager {}

// ── Resources ──────────────────────────────────────────────────────────────

/// Camera state used by lod_select_system to compute screen-space sizes.
/// Update this each frame from your camera transform before running systems.
#[derive(Resource, Debug, Clone)]
pub struct CameraSettings {
    /// Camera world position.
    pub position: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Render target height in pixels.
    pub screen_height: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            fov_y: std::f32::consts::FRAC_PI_4, // 45°
            screen_height: 1080.0,
        }
    }
}

/// Per-frame delta-time used by lod_crossfade_system.
/// Update this each frame with your frame duration in seconds.
#[derive(Resource, Debug, Clone, Copy)]
pub struct TimeStep(pub f32);

impl Default for TimeStep {
    fn default() -> Self { Self(1.0 / 60.0) }
}

// ── Systems ────────────────────────────────────────────────────────────────

/// Compute screen-space projected size in pixels for a unit-radius (1 m) bounding sphere.
fn projected_pixels(distance: f32, fov_y: f32, screen_height: f32) -> f32 {
    if distance < 0.001 {
        return screen_height;
    }
    let half_fov_tan = (fov_y * 0.5).tan();
    (1.0 / (distance * half_fov_tan)) * (screen_height * 0.5)
}

/// For each entity with TransformComponent + LodStateComponent, select the appropriate
/// LOD level based on camera distance and projected screen size. Requests a crossfade
/// transition when the level changes.
pub fn lod_select_system(
    camera: Res<CameraSettings>,
    mut crossfade: ResMut<LodCrossfadeManager>,
    mut query: Query<(Entity, &vox_core::ecs::TransformComponent, &mut vox_core::ecs::LodStateComponent)>,
) {
    for (entity, transform, mut lod_state) in query.iter_mut() {
        let distance = (transform.position - camera.position).length();
        let screen_size = projected_pixels(distance, camera.fov_y, camera.screen_height);
        let new_level = crate::hierarchical_lod::select_lod_level(distance, screen_size) as u8;

        if new_level != lod_state.current_level {
            crossfade.request_lod_change(
                entity.index(),
                lod_state.current_level as u32,
                new_level as u32,
            );
            lod_state.current_level = new_level;
        }
    }
}

/// Advance all active LOD crossfade transitions by `TimeStep.0` seconds,
/// then write the resulting crossfade weight back to each entity's LodStateComponent.
pub fn lod_crossfade_system(
    dt: Res<TimeStep>,
    mut crossfade: ResMut<LodCrossfadeManager>,
    mut query: Query<(Entity, &mut vox_core::ecs::LodStateComponent)>,
) {
    crossfade.tick(dt.0);

    for (entity, mut lod_state) in query.iter_mut() {
        if let Some(transition) = crossfade.get_transition(entity.index()) {
            lod_state.crossfade = transition.progress;
        } else {
            lod_state.crossfade = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_settings_default() {
        let s = CameraSettings::default();
        assert_eq!(s.position, Vec3::ZERO);
        assert!(s.fov_y > 0.0);
        assert!(s.screen_height > 0.0);
    }

    #[test]
    fn timestep_default_is_60hz() {
        let dt = TimeStep::default();
        assert!((dt.0 - 1.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn close_entity_selects_lod0() {
        let mut world = World::new();
        let mut cam = CameraSettings::default();
        cam.position = Vec3::ZERO;
        cam.screen_height = 2160.0; // 4K: projected_pixels(10m, 45°, 2160) ≈ 260px → LOD 0
        world.insert_resource(cam);
        world.insert_resource(LodCrossfadeManager { transitions: vec![], transition_duration: 0.5 });

        let entity = world.spawn((
            vox_core::ecs::TransformComponent {
                position: Vec3::new(0.0, 0.0, 10.0),
                ..Default::default()
            },
            vox_core::ecs::LodStateComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(lod_select_system);
        schedule.run(&mut world);

        let lod = world.entity(entity).get::<vox_core::ecs::LodStateComponent>().unwrap();
        assert_eq!(lod.current_level, 0, "10 m away should be LOD 0, got {}", lod.current_level);
    }

    #[test]
    fn distant_entity_selects_lod3() {
        let mut world = World::new();
        let mut cam = CameraSettings::default();
        cam.position = Vec3::ZERO;
        world.insert_resource(cam);
        world.insert_resource(LodCrossfadeManager { transitions: vec![], transition_duration: 0.5 });

        let entity = world.spawn((
            vox_core::ecs::TransformComponent {
                position: Vec3::new(0.0, 0.0, 500.0),
                ..Default::default()
            },
            vox_core::ecs::LodStateComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(lod_select_system);
        schedule.run(&mut world);

        let lod = world.entity(entity).get::<vox_core::ecs::LodStateComponent>().unwrap();
        assert_eq!(lod.current_level, 3, "500 m away should be LOD 3, got {}", lod.current_level);
    }

    #[test]
    fn crossfade_progresses_over_ticks() {
        let mut world = World::new();
        world.insert_resource(LodCrossfadeManager { transitions: vec![], transition_duration: 1.0 });
        world.insert_resource(TimeStep(0.1));
        world.insert_resource(CameraSettings {
            position: Vec3::ZERO,
            screen_height: 2160.0,
            ..Default::default()
        });

        // Spawn a far entity so lod_select picks LOD 3
        let entity = world.spawn((
            vox_core::ecs::TransformComponent {
                position: Vec3::new(0.0, 0.0, 500.0),
                ..Default::default()
            },
            vox_core::ecs::LodStateComponent { current_level: 0, crossfade: 0.0 },
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems((lod_select_system, lod_crossfade_system).chain());

        // Run once — lod_select requests 0→3 transition; lod_crossfade advances it by 0.1 s
        schedule.run(&mut world);

        let lod = world.entity(entity).get::<vox_core::ecs::LodStateComponent>().unwrap();
        assert!(
            lod.crossfade > 0.0 && lod.crossfade <= 1.0,
            "crossfade should be in (0, 1], got {}",
            lod.crossfade
        );
    }
}
