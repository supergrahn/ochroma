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

// ── Systems (no-backend only) ──────────────────────────────────────────────

/// Start or stop audio sources based on AudioEmitterComponent.playing.
///
/// - playing=true  + no AudioPlaybackComponent → registers source, inserts component
/// - playing=false + AudioPlaybackComponent    → stops source, removes component
#[cfg(not(feature = "audio-backend"))]
pub fn audio_emitter_system(
    mut commands: Commands,
    mut engine: ResMut<AudioEngineResource>,
    start_query: Query<
        (Entity, &vox_core::ecs::AudioEmitterComponent, &vox_core::ecs::TransformComponent),
        Without<AudioPlaybackComponent>,
    >,
    stop_query: Query<
        (Entity, &vox_core::ecs::AudioEmitterComponent, &AudioPlaybackComponent),
    >,
) {
    for (entity, emitter, transform) in start_query.iter() {
        if !emitter.playing {
            continue;
        }
        let source = crate::AudioSource {
            id: 0,
            position: transform.position,
            volume: emitter.volume,
            looping: emitter.looping,
            clip: emitter.clip_path.clone(),
        };
        let source_id = engine.engine.play(source);
        commands.entity(entity).insert(AudioPlaybackComponent { source_id });
    }

    for (entity, emitter, playback) in stop_query.iter() {
        if emitter.playing {
            continue;
        }
        engine.engine.stop(playback.source_id);
        commands.entity(entity).remove::<AudioPlaybackComponent>();
    }
}

/// Update listener position and advance AudioEngine by one timestep.
/// Evicts lowest-priority sources over the engine's max_sources budget.
#[cfg(not(feature = "audio-backend"))]
pub fn audio_tick_system(
    dt: Res<AudioTimeStep>,
    listener: Res<AudioListenerSettings>,
    mut engine: ResMut<AudioEngineResource>,
) {
    engine.engine.set_listener(listener.position);
    engine.engine.tick(dt.0);
}

// ── Plugin (no-backend only) ───────────────────────────────────────────────

/// Bevy plugin that registers audio ECS resources and systems.
///
/// Usage: `app.add_plugins(AudioPlugin::default())`
///
/// Update `AudioListenerSettings.position` each frame from the camera transform.
/// Set `AudioEmitterComponent.playing = true` to start a sound, `false` to stop it.
#[cfg(not(feature = "audio-backend"))]
#[derive(Default)]
pub struct AudioPlugin;

#[cfg(not(feature = "audio-backend"))]
impl bevy_app::Plugin for AudioPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(AudioEngineResource::default());
        app.insert_resource(AudioListenerSettings::default());
        app.insert_resource(AudioTimeStep::default());
        app.add_systems(
            bevy_app::Update,
            (audio_emitter_system, audio_tick_system).chain(),
        );
    }
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

    #[cfg(not(feature = "audio-backend"))]
    #[test]
    fn playing_true_inserts_playback_component() {
        use bevy_ecs::schedule::Schedule;
        let mut world = World::new();
        world.insert_resource(AudioEngineResource::default());

        let entity = world.spawn((
            vox_core::ecs::AudioEmitterComponent {
                clip_path: "test.wav".into(),
                volume: 1.0,
                looping: false,
                playing: true,
                spatial: false,
            },
            vox_core::ecs::TransformComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);

        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_some(),
            "entity should have AudioPlaybackComponent after playing=true"
        );
        let res = world.resource::<AudioEngineResource>();
        assert_eq!(res.engine.active_count(), 1);
    }

    #[cfg(not(feature = "audio-backend"))]
    #[test]
    fn playing_false_removes_playback_component() {
        use bevy_ecs::schedule::Schedule;
        let mut world = World::new();
        let mut engine_res = AudioEngineResource::default();
        let source = crate::AudioSource {
            id: 0,
            position: glam::Vec3::ZERO,
            volume: 1.0,
            looping: false,
            clip: "test.wav".into(),
        };
        let source_id = engine_res.engine.play(source);
        world.insert_resource(engine_res);

        let entity = world.spawn((
            vox_core::ecs::AudioEmitterComponent {
                clip_path: "test.wav".into(),
                volume: 1.0,
                looping: false,
                playing: false,
                spatial: false,
            },
            vox_core::ecs::TransformComponent::default(),
            AudioPlaybackComponent { source_id },
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);
        world.flush();

        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_none(),
            "AudioPlaybackComponent should be removed after playing=false"
        );
        let res = world.resource::<AudioEngineResource>();
        assert_eq!(res.engine.active_count(), 0);
    }

    #[cfg(not(feature = "audio-backend"))]
    #[test]
    fn tick_culls_over_budget_sources() {
        use bevy_ecs::schedule::Schedule;
        let mut world = World::new();
        let mut res = AudioEngineResource { engine: crate::AudioEngine::new(1) };
        res.engine.play(crate::AudioSource { id: 0, position: glam::Vec3::ZERO, volume: 1.0, looping: false, clip: "a.wav".into() });
        res.engine.play(crate::AudioSource { id: 0, position: glam::Vec3::ZERO, volume: 0.5, looping: false, clip: "b.wav".into() });
        assert_eq!(res.engine.active_count(), 2);
        world.insert_resource(res);
        world.insert_resource(AudioListenerSettings::default());
        world.insert_resource(AudioTimeStep(0.016));

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_tick_system);
        schedule.run(&mut world);

        let res = world.resource::<AudioEngineResource>();
        assert_eq!(res.engine.active_count(), 1, "tick should cull to max_sources=1");
    }

    #[cfg(not(feature = "audio-backend"))]
    #[test]
    fn plugin_inserts_resources() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(AudioPlugin::default());
        assert!(app.world().contains_resource::<AudioEngineResource>());
        assert!(app.world().contains_resource::<AudioListenerSettings>());
        assert!(app.world().contains_resource::<AudioTimeStep>());
    }
}
