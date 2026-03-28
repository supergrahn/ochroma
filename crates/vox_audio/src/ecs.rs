//! bevy_ecs integration for vox_audio.
//!
//! # Feature gate
//! `AudioEngineResource` is only available without the `audio-backend` feature because
//! `rodio::OutputStream` is `!Send` and cannot satisfy `bevy_ecs::Resource`.
//! Build and test with `--no-default-features`.
//!
//! Other types like `AudioListenerSettings`, `AudioTimeStep`, and `AudioPlaybackComponent`
//! are always available since they are plain Send+Sync types.

use bevy_ecs::prelude::*;
use glam::Vec3;

#[cfg(not(feature = "audio-backend"))]
use crate::AudioEngine;

// ── Resources ──────────────────────────────────────────────────────────────

/// Wraps AudioEngine as a bevy_ecs Resource.
/// Available only without the `audio-backend` feature.
#[cfg(not(feature = "audio-backend"))]
#[derive(Resource)]
pub struct AudioEngineResource {
    pub engine: AudioEngine,
}

#[cfg(not(feature = "audio-backend"))]
impl Default for AudioEngineResource {
    fn default() -> Self {
        Self { engine: AudioEngine::new(64) }
    }
}

/// Listener position used by audio_tick_system to set AudioEngine's
/// listener before each tick (affects distance attenuation).
#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioListenerSettings {
    pub position: Vec3,
}

impl Default for AudioListenerSettings {
    fn default() -> Self {
        Self { position: Vec3::ZERO }
    }
}

/// Per-frame delta-time for audio_tick_system.
#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioTimeStep(pub f32);

impl Default for AudioTimeStep {
    fn default() -> Self { Self(1.0 / 60.0) }
}

// ── Output component ───────────────────────────────────────────────────────

/// Inserted on an entity by audio_emitter_system when playback starts.
/// Removed when playback stops. Presence signals the source is registered
/// in AudioEngine.
#[derive(Component, Debug, Clone, Copy)]
pub struct AudioPlaybackComponent {
    pub source_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "audio-backend"))]
    #[test]
    fn audio_engine_resource_default_capacity() {
        let r = AudioEngineResource::default();
        assert_eq!(r.engine.active_count(), 0);
    }

    #[test]
    fn audio_listener_default_at_origin() {
        let l = AudioListenerSettings::default();
        assert_eq!(l.position, Vec3::ZERO);
    }

    #[test]
    fn audio_timestep_default_is_60hz() {
        let dt = AudioTimeStep::default();
        assert!((dt.0 - 1.0 / 60.0).abs() < 1e-6);
    }
}
