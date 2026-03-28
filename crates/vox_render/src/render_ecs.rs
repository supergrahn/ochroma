//! Standalone ECS integration for splat rendering.
//!
//! `SplatRenderPlugin` provides a `splat_gather_system` that reads
//! `SplatAssetComponent + Visible` entities and writes world-space splats
//! into the `RenderBuffer` resource each frame.
//!
//! Usage:
//! ```rust,ignore
//! app.insert_resource(RenderBuffer::default());
//! app.add_plugins(SplatRenderPlugin);
//! ```

use bevy_ecs::prelude::*;
use vox_core::ecs::{SplatAssetComponent, TransformComponent, Visible};
use vox_core::engine_runtime::{transform_splat, RenderBuffer};

// ── System ─────────────────────────────────────────────────────────────────

/// Gather world-space splats from all visible `SplatAssetComponent` entities
/// into `RenderBuffer`.
///
/// Clears `RenderBuffer.splats` each frame before re-gathering.
pub fn splat_gather_system(
    mut buffer: ResMut<RenderBuffer>,
    query: Query<(&SplatAssetComponent, &TransformComponent), With<Visible>>,
) {
    buffer.splats.clear();
    for (asset, transform) in query.iter() {
        for &splat in &asset.splats {
            buffer.splats.push(transform_splat(splat, transform));
        }
    }
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that inserts `RenderBuffer` and registers `splat_gather_system`
/// in `Update`.
pub struct SplatRenderPlugin;

impl bevy_app::Plugin for SplatRenderPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(RenderBuffer::default());
        app.add_systems(bevy_app::Update, splat_gather_system);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;
    use bevy_ecs::schedule::Schedule;
    use bevy_ecs::world::World;
    use glam::{Quat, Vec3};
    use uuid::Uuid;
    use vox_core::types::GaussianSplat;

    fn zero_splat() -> GaussianSplat {
        GaussianSplat {
            position: [0.0, 0.0, 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        }
    }

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(SplatRenderPlugin);
    }

    #[test]
    fn splat_gather_system_collects_visible_splats() {
        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());

        world.spawn((
            SplatAssetComponent {
                uuid: Uuid::nil(),
                splat_count: 3,
                splats: vec![zero_splat(), zero_splat(), zero_splat()],
            },
            TransformComponent {
                position: Vec3::new(5.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            Visible,
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(splat_gather_system);
        schedule.run(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 3, "all 3 splats should be gathered");
        assert!(
            (buffer.splats[0].position[0] - 5.0).abs() < 1e-5,
            "splat x should match entity world x"
        );
    }

    #[test]
    fn splat_gather_system_ignores_invisible() {
        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());

        world.spawn((
            SplatAssetComponent {
                uuid: Uuid::nil(),
                splat_count: 1,
                splats: vec![zero_splat()],
            },
            TransformComponent {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            // No Visible
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(splat_gather_system);
        schedule.run(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 0, "invisible entity should be skipped");
    }
}
