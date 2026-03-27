use bevy_ecs::prelude::*;
use vox_core::ecs::{SplatAssetComponent, SplatInstanceComponent, LodLevel};
use vox_core::mapgen::generate_map;
use glam::{Quat, Vec3};
use uuid::Uuid;

/// Spawn terrain as an ECS entity in the world.
/// Uses procedural map generation with hills, a river, and terrain materials.
pub fn spawn_terrain(world: &mut World, width: f32, _depth: f32, _material: &str) -> Uuid {
    // Use seed 42 and a density of 1 splat per m²
    let splats = generate_map(42, width, 1.0);
    let uuid = Uuid::new_v4();
    let splat_count = splats.len() as u32;

    // Spawn asset
    world.spawn(SplatAssetComponent {
        uuid,
        splat_count,
        splats,
    });

    // Spawn instance at world origin
    world.spawn(SplatInstanceComponent {
        asset_uuid: uuid,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
        instance_id: 1000, // reserved for terrain
        lod: LodLevel::Full,
    });

    uuid
}
