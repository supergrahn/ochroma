use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use uuid::Uuid;
use half::f16;
use vox_core::ecs::{SplatAssetComponent, SplatInstanceComponent, LodLevel};
use vox_core::types::GaussianSplat;
use vox_sim::roads::RoadSegment;

/// Generate surface splats for a road segment.
pub fn generate_road_splats(segment: &RoadSegment) -> Vec<GaussianSplat> {
    let asphalt_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.05).to_bits());
    let mut splats = Vec::new();
    let steps = (segment.length() / 0.5).ceil() as usize; // One splat every 0.5m
    let half_width = segment.road_type.width() * 0.5;

    for i in 0..=steps {
        let t = i as f32 / steps.max(1) as f32;
        let center = segment.sample(t);
        // Get road direction for perpendicular offset
        let t2 = ((i as f32 + 0.5) / steps.max(1) as f32).min(1.0);
        let dir = (segment.sample(t2) - center).normalize_or_zero();
        let perp = Vec3::new(-dir.z, 0.0, dir.x);

        // Place splats across the width
        let width_steps = ((half_width * 2.0 / 0.5).ceil() as i32).max(1);
        for w in 0..width_steps {
            let offset = (w as f32 / width_steps as f32 - 0.5) * half_width * 2.0;
            let pos = center + perp * offset;
            splats.push(GaussianSplat {
                position: [pos.x, 0.01, pos.z], // slightly above terrain
                scale: [0.25, 0.01, 0.25],
                rotation: [0, 0, 0, 32767],
                opacity: 240,
                _pad: [0; 3],
                spectral: asphalt_spd,
            });
        }
    }
    splats
}

/// Spawn a road segment's visual representation as an ECS entity.
pub fn spawn_road_visual(world: &mut World, segment: &RoadSegment) -> Uuid {
    let splats = generate_road_splats(segment);
    let uuid = Uuid::new_v4();
    let count = splats.len() as u32;

    world.spawn(SplatAssetComponent { uuid, splat_count: count, splats });
    world.spawn(SplatInstanceComponent {
        asset_uuid: uuid,
        position: Vec3::ZERO, // splats are already in world space
        rotation: Quat::IDENTITY,
        scale: 1.0,
        instance_id: segment.id + 10000, // offset to avoid collision
        lod: LodLevel::Full,
    });

    uuid
}
