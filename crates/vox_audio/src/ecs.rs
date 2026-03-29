//! bevy_ecs integration for vox_audio.
//!
//! `AudioHandleResource` wraps a `Sender<AudioCommand>` which is always `Send`,
//! so it can be used as a Bevy ECS Resource regardless of the `audio-backend`
//! feature flag. This resolves the previous `!Send` conflict with
//! `rodio::OutputStream`.

use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::{AudioEngine, AudioHandle};

// ── Resources ──────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct AudioHandleResource(pub Option<AudioHandle>);

#[derive(Resource)]
pub struct AudioEngineResource {
    pub engine: AudioEngine,
}

impl Default for AudioEngineResource {
    fn default() -> Self {
        Self { engine: AudioEngine::new(64) }
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioListenerSettings {
    pub position: Vec3,
}

impl Default for AudioListenerSettings {
    fn default() -> Self {
        Self { position: Vec3::ZERO }
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioTimeStep(pub f32);

impl Default for AudioTimeStep {
    fn default() -> Self { Self(1.0 / 60.0) }
}

// ── Components ─────────────────────────────────────────────────────────────

#[derive(Component, Debug, Clone, Copy)]
pub struct AudioPlaybackComponent {
    pub source_id: u32,
}

// ── Systems ────────────────────────────────────────────────────────────────

pub fn audio_emitter_system(
    mut commands: Commands,
    handle: Res<AudioHandleResource>,
    start_query: Query<
        (Entity, &vox_core::ecs::AudioEmitterComponent),
        (Without<AudioPlaybackComponent>, Changed<vox_core::ecs::AudioEmitterComponent>),
    >,
    stop_query: Query<
        (Entity, &AudioPlaybackComponent, &vox_core::ecs::AudioEmitterComponent),
    >,
) {
    let Some(ref audio) = handle.0 else { return };

    for (entity, emitter) in start_query.iter() {
        if emitter.playing {
            let id = audio.play(&emitter.clip_path, emitter.volume, emitter.looping);
            commands.entity(entity).insert(AudioPlaybackComponent { source_id: id });
        }
    }

    for (entity, playback, emitter) in stop_query.iter() {
        if !emitter.playing {
            audio.stop(playback.source_id);
            commands.entity(entity).remove::<AudioPlaybackComponent>();
        }
    }
}

pub fn audio_tick_system(
    dt: Res<AudioTimeStep>,
    listener: Res<AudioListenerSettings>,
    mut engine: ResMut<AudioEngineResource>,
) {
    engine.engine.set_listener(listener.position);
    engine.engine.tick(dt.0);
}

// ── Plugin ─────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct AudioPlugin;

impl bevy_app::Plugin for AudioPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        let handle = AudioHandle::spawn();
        app.insert_resource(AudioHandleResource(handle));
        app.insert_resource(AudioEngineResource::default());
        app.insert_resource(AudioListenerSettings::default());
        app.insert_resource(AudioTimeStep::default());
        app.add_systems(
            bevy_app::Update,
            (audio_emitter_system, audio_tick_system).chain(),
        );
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::schedule::Schedule;

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

    #[test]
    fn audio_plugin_builds_without_panic() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        assert!(app.world().contains_resource::<AudioHandleResource>());
        assert!(app.world().contains_resource::<AudioEngineResource>());
        assert!(app.world().contains_resource::<AudioListenerSettings>());
        assert!(app.world().contains_resource::<AudioTimeStep>());
    }

    #[test]
    fn audio_emitter_system_noop_when_handle_is_none() {
        let mut world = World::new();
        world.insert_resource(AudioHandleResource(None));

        let entity = world.spawn(vox_core::ecs::AudioEmitterComponent {
            clip_path: "x.ogg".into(),
            volume: 1.0,
            looping: false,
            playing: true,
            spatial: false,
        }).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);
        world.flush();

        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_none(),
        );
    }

    #[test]
    fn audio_emitter_system_noop_when_not_playing() {
        let mut world = World::new();
        world.insert_resource(AudioHandleResource(None));

        let entity = world.spawn(vox_core::ecs::AudioEmitterComponent {
            clip_path: "x.ogg".into(),
            volume: 1.0,
            looping: false,
            playing: false,
            spatial: false,
        }).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);
        world.flush();

        assert!(world.entity(entity).get::<AudioPlaybackComponent>().is_none());
    }

    #[test]
    fn audio_emitter_system_inserts_component_with_real_handle() {
        use crate::AudioCommand;
        use std::sync::{Arc, atomic::AtomicU32};

        let (tx, _rx) = std::sync::mpsc::channel::<AudioCommand>();
        let handle = AudioHandle::new_test(tx, Arc::new(AtomicU32::new(1)));

        let mut world = World::new();
        world.insert_resource(AudioHandleResource(Some(handle)));

        let entity = world.spawn(vox_core::ecs::AudioEmitterComponent {
            clip_path: "boom.wav".into(),
            volume: 0.8,
            looping: false,
            playing: true,
            spatial: false,
        }).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);
        world.flush();

        assert!(world.entity(entity).get::<AudioPlaybackComponent>().is_some());
    }

    #[test]
    fn audio_emitter_system_removes_component_when_stopped() {
        use crate::AudioCommand;
        use std::sync::{Arc, atomic::AtomicU32};

        let (tx, _rx) = std::sync::mpsc::channel::<AudioCommand>();
        let handle = AudioHandle::new_test(tx, Arc::new(AtomicU32::new(1)));

        let mut world = World::new();
        world.insert_resource(AudioHandleResource(Some(handle)));

        let entity = world.spawn((
            vox_core::ecs::AudioEmitterComponent {
                clip_path: "boom.wav".into(),
                volume: 0.8,
                looping: false,
                playing: false,
                spatial: false,
            },
            AudioPlaybackComponent { source_id: 42 },
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);
        world.flush();

        assert!(world.entity(entity).get::<AudioPlaybackComponent>().is_none());
    }

    #[test]
    fn tick_culls_over_budget_sources() {
        let mut world = World::new();
        let mut res = AudioEngineResource { engine: crate::AudioEngine::new(1) };
        res.engine.play(crate::AudioSource { id: 0, position: Vec3::ZERO, volume: 1.0, looping: false, clip: "a.wav".into() });
        res.engine.play(crate::AudioSource { id: 0, position: Vec3::ZERO, volume: 0.5, looping: false, clip: "b.wav".into() });
        world.insert_resource(res);
        world.insert_resource(AudioListenerSettings::default());
        world.insert_resource(AudioTimeStep(0.016));

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_tick_system);
        schedule.run(&mut world);

        let res = world.resource::<AudioEngineResource>();
        assert_eq!(res.engine.active_count(), 1);
    }
}
