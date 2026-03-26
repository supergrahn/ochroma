use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    Full,
    Reduced,
}

#[derive(Component, Debug, Clone)]
pub struct SplatInstanceComponent {
    pub asset_uuid: Uuid,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
    pub instance_id: u32,
    pub lod: LodLevel,
}

#[derive(Component, Debug)]
pub struct SplatAssetComponent {
    pub uuid: Uuid,
    pub splat_count: u32,
    pub splats: Vec<crate::types::GaussianSplat>,
}
