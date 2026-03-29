//! Procedural walk-cycle animation system.
//!
//! Provides `ProceduralWalkComponent` — an ECS component that stores a set of
//! base splat positions and an accumulated time. `animation_system` advances
//! time each frame, computes a sinusoidal vertical bob, and pushes the
//! resulting `GaussianSplat`s into the `RenderBuffer`.
//!
//! No skeleton or GLTF data required — suitable for demos and placeholder NPCs.

use bevy_ecs::prelude::*;
use glam::Vec3;
use vox_core::engine_runtime::{FrameTime, RenderBuffer};
use vox_core::types::GaussianSplat;

/// Procedural walk-cycle component.
#[derive(Component, Debug, Clone)]
pub struct ProceduralWalkComponent {
    pub base_positions: Vec<[f32; 3]>,
    pub time: f32,
    pub bob_amplitude: f32,
    pub bob_frequency: f32,
}

impl ProceduralWalkComponent {
    pub fn humanoid_blob(center: Vec3) -> Self {
        let offsets: &[[f32; 3]] = &[
            [0.0,   0.0,  0.0],
            [0.0,   0.4,  0.0],
            [0.0,   0.8,  0.0],
            [-0.2,  0.2,  0.0],
            [0.2,   0.2,  0.0],
            [-0.15,-0.4,  0.0],
            [0.15, -0.4,  0.0],
            [0.0,  -0.7,  0.0],
        ];
        let base_positions = offsets
            .iter()
            .map(|o| [center.x + o[0], center.y + o[1], center.z + o[2]])
            .collect();
        Self {
            base_positions,
            time: 0.0,
            bob_amplitude: 0.05,
            bob_frequency: 2.0,
        }
    }
}

pub fn animation_system(
    time: Res<FrameTime>,
    mut render_buffer: ResMut<RenderBuffer>,
    mut query: Query<&mut ProceduralWalkComponent>,
) {
    let dt = time.dt;
    for mut npc in query.iter_mut() {
        npc.time += dt;
        let bob = (npc.time * npc.bob_frequency * std::f32::consts::TAU).sin()
            * npc.bob_amplitude;
        for base in &npc.base_positions {
            render_buffer.splats.push(GaussianSplat {
                position: [base[0], base[1] + bob, base[2]],
                scale: [0.12, 0.12, 0.12],
                rotation: [0i16, 0, 0, 32767],
                opacity: 200,
                _pad: [0; 3],
                spectral: [15000u16; 8],
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;
    use glam::Vec3;

    fn make_world_with_resources() -> World {
        let mut world = World::new();
        world.insert_resource(FrameTime { dt: 0.016, total: 0.0, frame: 1 });
        world.insert_resource(RenderBuffer::default());
        world
    }

    fn run_animation_system(world: &mut World) {
        let mut system = IntoSystem::into_system(animation_system);
        system.initialize(world);
        system.run((), world);
        system.apply_deferred(world);
    }

    #[test]
    fn walk_component_humanoid_blob_has_8_splats() {
        let comp = ProceduralWalkComponent::humanoid_blob(Vec3::ZERO);
        assert_eq!(comp.base_positions.len(), 8);
    }

    #[test]
    fn walk_component_humanoid_blob_centers_on_given_position() {
        let center = Vec3::new(5.0, 10.0, -3.0);
        let comp = ProceduralWalkComponent::humanoid_blob(center);
        // All positions should be near the center (within max offset ~1 unit)
        for pos in &comp.base_positions {
            assert!((pos[0] - center.x).abs() <= 0.3,
                "x offset too large: {}", pos[0] - center.x);
            assert!((pos[2] - center.z).abs() <= 0.3,
                "z offset too large: {}", pos[2] - center.z);
        }
    }

    #[test]
    fn animation_system_pushes_splats_to_render_buffer() {
        let mut world = make_world_with_resources();
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::ZERO));

        run_animation_system(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 8, "8 splats should be pushed for one NPC");
    }

    #[test]
    fn animation_system_splat_opacity_and_spectral_are_set() {
        let mut world = make_world_with_resources();
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::ZERO));

        run_animation_system(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        for splat in &buffer.splats {
            assert_eq!(splat.opacity, 200);
            assert_eq!(splat.spectral, [15000u16; 8]);
        }
    }

    #[test]
    fn bob_offset_changes_with_time() {
        let mut world = make_world_with_resources();
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::ZERO));

        // First tick
        run_animation_system(&mut world);
        let y_first = {
            let buffer = world.resource::<RenderBuffer>();
            buffer.splats[0].position[1]
        };

        // Clear buffer and tick again with different dt
        {
            let mut buffer = world.resource_mut::<RenderBuffer>();
            buffer.splats.clear();
        }
        {
            let mut ft = world.resource_mut::<FrameTime>();
            ft.dt = 0.1; // larger dt -> different phase
        }

        run_animation_system(&mut world);
        let y_second = {
            let buffer = world.resource::<RenderBuffer>();
            buffer.splats[0].position[1]
        };

        // The y positions should differ (different bob phase)
        assert!(
            (y_first - y_second).abs() > 1e-6,
            "bob should change with time: y_first={y_first}, y_second={y_second}"
        );
    }

    #[test]
    fn animation_system_accumulates_time_on_component() {
        let mut world = make_world_with_resources();
        let entity = world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::ZERO)).id();

        run_animation_system(&mut world);

        let comp = world.get::<ProceduralWalkComponent>(entity).unwrap();
        assert!(
            (comp.time - 0.016).abs() < 1e-5,
            "time should be accumulated: got {}",
            comp.time
        );
    }

    #[test]
    fn multiple_npc_entities_all_push_splats() {
        let mut world = make_world_with_resources();
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::new(0.0, 0.0, 0.0)));
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::new(5.0, 0.0, 0.0)));
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::new(-5.0, 0.0, 0.0)));

        run_animation_system(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(
            buffer.splats.len(), 24,
            "3 NPCs * 8 splats = 24 total"
        );
    }
}
