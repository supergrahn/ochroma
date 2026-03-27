use bevy_ecs::prelude::*;
use vox_core::ecs::{SplatAssetComponent, SplatInstanceComponent, LodLevel};
use vox_core::terrain::{TerrainPlane, generate_terrain_splats};
use glam::{Quat, Vec3};
use uuid::Uuid;

/// Spawn terrain as an ECS entity in the world.
pub fn spawn_terrain(world: &mut World, width: f32, depth: f32, material: &str) -> Uuid {
    let terrain = TerrainPlane::new(width, depth, 4.0); // 4 splats per m²
    let splats = generate_terrain_splats(&terrain, material);
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
