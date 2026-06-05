use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use serde::{Serialize, Deserialize};
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

// ---------------------------------------------------------------------------
// Engine Runtime v2 — Bevy ECS components
// ---------------------------------------------------------------------------

/// Core transform (replaces SplatInstanceComponent's position/rotation/scale).
#[derive(Component, Debug, Clone)]
pub struct TransformComponent {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for TransformComponent {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Entity name.
#[derive(Component, Debug, Clone)]
pub struct NameComponent(pub String);

/// Tags for searching.
#[derive(Component, Debug, Clone, Default)]
pub struct TagsComponent(pub Vec<String>);

/// Reference to a loaded asset.
#[derive(Component, Debug, Clone)]
pub struct AssetRefComponent {
    pub path: String,
    pub handle: u64,
}

/// Script attachment — list of script names to run on this entity.
#[derive(Component, Debug, Clone, Default)]
pub struct ScriptComponent {
    pub scripts: Vec<String>,
}

/// Linear velocity for physics-driven entities.
///
/// Scripts can mutate this via `ScriptCommand::ApplyForce`, which integrates
/// the requested force into the velocity (treating mass as 1.0 unit). The
/// physics integration step reads this and advances `TransformComponent`.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
pub struct VelocityComponent {
    pub linear: Vec3,
}

/// Accumulated force for the current fixed step.
///
/// `ScriptCommand::ApplyForce` adds into `accumulated`; the physics
/// integration step consumes it into `VelocityComponent` and clears it.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
pub struct ForceComponent {
    pub accumulated: Vec3,
}

/// Collider shape for physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColliderShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { radius: f32, height: f32 },
}

/// Collider for physics.
#[derive(Component, Debug, Clone)]
pub struct ColliderComponent {
    pub shape: ColliderShape,
}

/// Audio emitter.
#[derive(Component, Debug, Clone)]
pub struct AudioEmitterComponent {
    pub clip_path: String,
    pub volume: f32,
    pub looping: bool,
    pub playing: bool,
    pub spatial: bool,
}

/// Per-entity LOD state written by lod_select_system.
/// Attach this component to any entity that should receive automatic LOD selection.
/// `current_level` maps to the 4-level hierarchy in `vox_render::hierarchical_lod`:
///   0 = full detail, 1 = 40% splats, 2 = 10% splats, 3 = billboard.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct LodStateComponent {
    /// Current LOD level index (0–3).
    pub current_level: u8,
    /// Crossfade blending weight towards the next level (0.0 = stable, 1.0 = fully transitioned).
    pub crossfade: f32,
}

/// Point light.
#[derive(Component, Debug, Clone)]
pub struct PointLightComponent {
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

/// Directional light.
#[derive(Component, Debug, Clone)]
pub struct DirectionalLightComponent {
    pub color: [f32; 3],
    pub intensity: f32,
    pub direction: Vec3,
}

/// Custom game data (key-value store per entity).
#[derive(Component, Debug, Clone, Default)]
pub struct CustomDataComponent {
    pub data: std::collections::HashMap<String, String>,
}

/// Marker: this entity is visible after frustum culling.
#[derive(Component)]
pub struct Visible;

/// Marker: this entity is the player's camera target.
#[derive(Component)]
pub struct CameraTarget;
