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
}
