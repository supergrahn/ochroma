# Audio File Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire rodio file decoding so `AudioEmitterComponent.clip_path` plays actual `.wav`/`.ogg` audio files.

**Architecture:** Replace the `!Send` `AudioBackend` design with a channel-based `AudioThread` that owns rodio on a background thread. The ECS side holds only a `Sender<AudioCommand>` which is `Send`, eliminating the feature-gate conflict. `AudioPlugin` inserts `AudioHandleResource` and `audio_emitter_system` sends play/stop commands over the channel.

**Tech Stack:** `rodio 0.19` (`Decoder` + `Source`), `std::sync::mpsc`, `bevy_ecs 0.16`, `bevy_app 0.16`

---

## Key Files

- `crates/vox_audio/src/lib.rs` — `AudioCommand`, `AudioThread`, `AudioHandle`
- `crates/vox_audio/src/ecs.rs` — `AudioHandleResource`, `audio_emitter_system`, `AudioPlugin`
- `crates/vox_app/src/bin/engine_runner.rs` — wire `AudioHandle` into `EngineApp`

---

### Task 1: `AudioCommand` + `AudioThread` + `AudioHandle` in `lib.rs`

- [ ] Replace `AudioBackend` with `AudioCommand` enum, `AudioThread` struct, and `AudioHandle` struct in `crates/vox_audio/src/lib.rs`
- [ ] Keep `AudioEngine`, `AudioSource`, `SpatialAudioManager`, `SoundEffect` etc. unchanged
- [ ] Add unit tests

**Files:**
- Modify: `crates/vox_audio/src/lib.rs`

Replace the `#[cfg(feature = "audio-backend")]` block that defines `AudioBackend` (lines 12–73) with the following. Leave everything from `pub struct AudioEngine` onward intact, except remove the `backend: Option<AudioBackend>` field and its related `#[cfg]` blocks from `AudioEngine` — replace them with a note that `AudioHandle` is the new playback path.

```rust
pub mod acoustic_raytracer;
pub mod audio_graph;
pub mod spatial;
pub mod synth;
pub mod ecs;

pub use spatial::{compute_spatial, Listener, SpatialAudioManager};
pub use synth::{generate_click, generate_collect_sound, generate_place_sound, generate_tone, save_wav};

use glam::Vec3;

// ── AudioCommand ────────────────────────────────────────────────────────────

/// Commands sent from the ECS/main thread to the audio background thread.
#[derive(Debug)]
pub enum AudioCommand {
    /// Play a file. `id` is caller-assigned for later Stop.
    Play { id: u32, path: String, volume: f32, looping: bool },
    /// Stop a single playing sound.
    Stop { id: u32 },
    /// Stop all playing sounds immediately.
    StopAll,
}

// ── AudioThread (audio-backend only) ────────────────────────────────────────

/// Owns `rodio::OutputStream` and all `Sink`s. Lives exclusively on the
/// spawned audio background thread — never crosses thread boundaries.
#[cfg(feature = "audio-backend")]
struct AudioThread {
    receiver: std::sync::mpsc::Receiver<AudioCommand>,
    stream_handle: rodio::OutputStreamHandle,
    sinks: std::collections::HashMap<u32, rodio::Sink>,
}

#[cfg(feature = "audio-backend")]
impl AudioThread {
    fn run(mut self) {
        while let Ok(cmd) = self.receiver.recv() {
            match cmd {
                AudioCommand::Play { id, path, volume, looping } => {
                    match std::fs::File::open(&path) {
                        Ok(file) => {
                            match rodio::Decoder::new(std::io::BufReader::new(file)) {
                                Ok(source) => {
                                    match rodio::Sink::try_new(&self.stream_handle) {
                                        Ok(sink) => {
                                            sink.set_volume(volume);
                                            if looping {
                                                sink.append(source.repeat_infinite());
                                            } else {
                                                sink.append(source);
                                            }
                                            self.sinks.insert(id, sink);
                                        }
                                        Err(e) => eprintln!("[ochroma-audio] Sink error for {}: {}", path, e),
                                    }
                                }
                                Err(e) => eprintln!("[ochroma-audio] Decode error for {}: {}", path, e),
                            }
                        }
                        Err(e) => eprintln!("[ochroma-audio] File open error for {}: {}", path, e),
                    }
                }
                AudioCommand::Stop { id } => {
                    if let Some(sink) = self.sinks.remove(&id) {
                        sink.stop();
                    }
                }
                AudioCommand::StopAll => {
                    for (_, sink) in self.sinks.drain() {
                        sink.stop();
                    }
                }
            }
        }
    }
}

// ── AudioHandle ──────────────────────────────────────────────────────────────

/// Send half of the audio command channel.
///
/// This type IS `Send` + `Sync` — it contains only a `Sender` and an `Arc<AtomicU32>`.
/// It lives on the ECS/main thread and communicates with `AudioThread` via mpsc.
///
/// Obtain one via `AudioHandle::spawn()`. If the `audio-backend` feature is
/// disabled, `spawn()` returns `None` and all methods are silent no-ops.
#[derive(Clone)]
pub struct AudioHandle {
    sender: std::sync::mpsc::Sender<AudioCommand>,
    next_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl AudioHandle {
    /// Spawn the audio background thread and return a handle to it.
    ///
    /// Returns `None` if rodio cannot open the default output device.
    /// The background thread exits automatically when all `AudioHandle` clones
    /// are dropped (channel closes, `recv()` returns `Err`).
    #[cfg(feature = "audio-backend")]
    pub fn spawn() -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<AudioCommand>();
        let (_stream, stream_handle) = match rodio::OutputStream::try_default() {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("[ochroma-audio] Failed to open audio device: {}", e);
                return None;
            }
        };
        let audio_thread = AudioThread {
            receiver: rx,
            stream_handle,
            sinks: std::collections::HashMap::new(),
        };
        std::thread::Builder::new()
            .name("ochroma-audio".into())
            .spawn(move || {
                // OutputStream must stay alive on this thread for as long as sinks play.
                let _keep_stream_alive = _stream;
                audio_thread.run();
            })
            .expect("failed to spawn audio thread");
        Some(Self {
            sender: tx,
            next_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1)),
        })
    }

    /// No-op stub when `audio-backend` feature is disabled.
    #[cfg(not(feature = "audio-backend"))]
    pub fn spawn() -> Option<Self> {
        None
    }

    /// Play a file at `path`. Returns a `source_id` that can be passed to `stop()`.
    /// If the handle was constructed from a `None` spawn, this is a silent no-op.
    pub fn play(&self, path: &str, volume: f32, looping: bool) -> u32 {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = self.sender.send(AudioCommand::Play {
            id,
            path: path.to_string(),
            volume,
            looping,
        });
        id
    }

    /// Stop a specific playing sound by its `source_id`.
    pub fn stop(&self, id: u32) {
        let _ = self.sender.send(AudioCommand::Stop { id });
    }

    /// Stop all currently playing sounds.
    pub fn stop_all(&self) {
        let _ = self.sender.send(AudioCommand::StopAll);
    }
}

// ── AudioSource / AudioEngine (kept for priority-budget management) ──────────

#[derive(Debug, Clone)]
pub struct AudioSource {
    pub id: u32,
    pub position: Vec3,
    pub volume: f32,
    pub looping: bool,
    pub clip: String,
}

/// Logical audio engine: tracks active sources for priority/budget management.
/// Actual hardware playback is delegated to `AudioHandle` / `AudioThread`.
pub struct AudioEngine {
    max_sources: usize,
    pub sources: Vec<AudioSource>,
    next_id: u32,
    pub listener_position: Vec3,
}

impl AudioEngine {
    pub fn new(max_sources: usize) -> Self {
        Self {
            max_sources,
            sources: Vec::new(),
            next_id: 1,
            listener_position: Vec3::ZERO,
        }
    }

    pub fn init_backend(&mut self) {
        // Backend is now owned by AudioHandle on a separate thread.
        // Call AudioHandle::spawn() separately and store it alongside AudioEngine.
    }

    pub fn set_listener(&mut self, pos: Vec3) {
        self.listener_position = pos;
    }

    pub fn play(&mut self, mut source: AudioSource) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        source.id = id;
        self.sources.push(source);
        id
    }

    pub fn stop(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
    }

    pub fn active_count(&self) -> usize {
        self.sources.len()
    }

    pub fn effective_volume(&self, source: &AudioSource) -> f32 {
        Self::effective_volume_at(source, self.listener_position)
    }

    fn effective_volume_at(source: &AudioSource, listener: Vec3) -> f32 {
        let dist = source.position.distance(listener);
        let attenuation = 1.0 / (1.0 + dist * 0.1);
        source.volume * attenuation
    }

    pub fn active_sources_by_priority(&self) -> Vec<&AudioSource> {
        let mut sources: Vec<&AudioSource> = self.sources.iter().collect();
        sources.sort_by(|a, b| {
            self.effective_volume(b)
                .partial_cmp(&self.effective_volume(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sources
    }

    /// Legacy sine-wave playback stub — kept for call-site compatibility.
    /// Actual sine generation is handled by SpatialAudioManager or AudioHandle.
    pub fn play_sine_backend(&mut self, _id: u32, _frequency: f32, _duration_secs: f32, _volume: f32) {
        // No-op: migrate call sites to AudioHandle::play() with a pre-generated wav, or SpatialAudioManager.
    }

    /// Tick: evict lowest-priority sources if over budget.
    pub fn tick(&mut self, _dt: f32) {
        let listener = self.listener_position;
        while self.sources.len() > self.max_sources {
            if let Some(idx) = self.sources
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    Self::effective_volume_at(a, listener)
                        .partial_cmp(&Self::effective_volume_at(b, listener))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.sources.remove(idx);
            } else {
                break;
            }
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(64)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// AudioHandle::spawn() should return Some when the audio-backend feature is on
    /// and a default output device is available (CI may skip via env).
    #[cfg(feature = "audio-backend")]
    #[test]
    fn audio_handle_spawn_returns_some_with_feature() {
        // On headless CI this may return None — skip rather than fail.
        if std::env::var("CI").is_ok() {
            return;
        }
        let handle = AudioHandle::spawn();
        assert!(handle.is_some(), "AudioHandle::spawn() should return Some on a machine with audio");
    }

    /// Sending a play command for a nonexistent file must not panic the ECS thread.
    /// The audio thread logs an error and continues.
    #[test]
    fn audio_handle_play_nonexistent_file_does_not_panic() {
        // Build a handle with a dummy sender (no real thread needed for this test).
        let (tx, _rx) = std::sync::mpsc::channel::<AudioCommand>();
        let handle = AudioHandle {
            sender: tx,
            next_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1)),
        };
        // play() sends over the channel — _rx dropped so send returns Err, which is swallowed.
        let id = handle.play("nonexistent.wav", 1.0, false);
        assert!(id >= 1, "play() should return a nonzero id");
        handle.stop(id); // must not panic
    }

    #[test]
    fn audio_engine_active_count_starts_zero() {
        let engine = AudioEngine::new(64);
        assert_eq!(engine.active_count(), 0);
    }

    #[test]
    fn audio_engine_tick_culls_over_budget() {
        let mut engine = AudioEngine::new(1);
        engine.play(AudioSource { id: 0, position: Vec3::ZERO, volume: 1.0, looping: false, clip: "a.wav".into() });
        engine.play(AudioSource { id: 0, position: Vec3::ZERO, volume: 0.5, looping: false, clip: "b.wav".into() });
        assert_eq!(engine.active_count(), 2);
        engine.tick(0.016);
        assert_eq!(engine.active_count(), 1, "tick should cull to max_sources=1");
    }
}
```

---

### Task 2: `AudioHandleResource` + updated ECS in `ecs.rs`

- [ ] Replace `AudioEngineResource` (currently gated `#[cfg(not(feature = "audio-backend"))]`) with `AudioHandleResource` (always `Send`, no feature gate)
- [ ] Rewrite `audio_emitter_system` to send `AudioCommand` via `AudioHandle`
- [ ] Remove the feature gate from `AudioPlugin`
- [ ] Keep `AudioListenerSettings`, `AudioTimeStep`, `AudioPlaybackComponent`, and `audio_tick_system`
- [ ] Add unit tests

**Files:**
- Modify: `crates/vox_audio/src/ecs.rs`

Complete replacement for `crates/vox_audio/src/ecs.rs`:

```rust
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

/// ECS Resource wrapping the channel-based audio handle.
///
/// Replaces the old `AudioEngineResource`. Always `Send` + `Sync` because
/// `AudioHandle` contains only a `Sender` and an `Arc<AtomicU32>`.
///
/// `None` when the `audio-backend` feature is disabled or the device
/// failed to open — systems check `handle.0.is_some()` before sending.
#[derive(Resource)]
pub struct AudioHandleResource(pub Option<AudioHandle>);

/// Wraps `AudioEngine` for priority/budget tracking in ECS.
/// Does NOT own hardware playback — that is `AudioHandleResource`.
#[derive(Resource)]
pub struct AudioEngineResource {
    pub engine: AudioEngine,
}

impl Default for AudioEngineResource {
    fn default() -> Self {
        Self { engine: AudioEngine::new(64) }
    }
}

/// Listener position used by `audio_tick_system` (affects distance attenuation).
#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioListenerSettings {
    pub position: Vec3,
}

impl Default for AudioListenerSettings {
    fn default() -> Self {
        Self { position: Vec3::ZERO }
    }
}

/// Per-frame delta-time for `audio_tick_system`.
#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioTimeStep(pub f32);

impl Default for AudioTimeStep {
    fn default() -> Self { Self(1.0 / 60.0) }
}

// ── Components ─────────────────────────────────────────────────────────────

/// Inserted on an entity by `audio_emitter_system` when playback starts.
/// Removed when `AudioEmitterComponent.playing` becomes false.
/// Presence signals the source is actively playing in `AudioThread`.
#[derive(Component, Debug, Clone, Copy)]
pub struct AudioPlaybackComponent {
    pub source_id: u32,
}

// ── Systems ────────────────────────────────────────────────────────────────

/// Start or stop audio file playback based on `AudioEmitterComponent.playing`.
///
/// - `playing = true`  + no `AudioPlaybackComponent` → sends `Play` command, inserts component
/// - `playing = false` + has `AudioPlaybackComponent` → sends `Stop` command, removes component
///
/// Uses `Changed<AudioEmitterComponent>` on the start query to avoid re-sending
/// every frame for already-playing emitters.
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

/// Update listener position and advance `AudioEngine` budget tick.
/// Does NOT interact with hardware — just evicts lowest-priority logical sources.
pub fn audio_tick_system(
    dt: Res<AudioTimeStep>,
    listener: Res<AudioListenerSettings>,
    mut engine: ResMut<AudioEngineResource>,
) {
    engine.engine.set_listener(listener.position);
    engine.engine.tick(dt.0);
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that wires audio ECS resources and systems.
///
/// No feature gate required — `AudioHandleResource` is always `Send`.
/// On systems without `audio-backend`, the handle is `None` and emitter
/// system becomes a no-op (silent mode).
///
/// Usage:
/// ```rust
/// app.add_plugins(AudioPlugin);
/// ```
///
/// To play a sound: set `AudioEmitterComponent { clip_path: "assets/boom.wav".into(), playing: true, .. }`
/// on any entity. To stop: set `playing = false`.
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

    /// AudioPlugin registers all required resources.
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

    /// With a None handle, audio_emitter_system must not panic even when
    /// emitter.playing is true and no AudioPlaybackComponent is present.
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

        // None handle → no AudioPlaybackComponent inserted
        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_none(),
            "None handle should leave no AudioPlaybackComponent"
        );
    }

    /// playing=false with None handle must not panic and must not insert component.
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

        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_none(),
            "playing=false should never insert AudioPlaybackComponent"
        );
    }

    /// audio_emitter_system with a real AudioHandle (dummy sender, no real thread)
    /// inserts AudioPlaybackComponent when playing=true.
    #[test]
    fn audio_emitter_system_inserts_component_with_real_handle() {
        use crate::AudioCommand;
        use std::sync::{Arc, atomic::AtomicU32};

        let (tx, _rx) = std::sync::mpsc::channel::<AudioCommand>();
        let handle = AudioHandle {
            sender: tx,
            next_id: Arc::new(AtomicU32::new(1)),
        };

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

        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_some(),
            "AudioPlaybackComponent should be inserted when playing=true and handle is Some"
        );
    }

    /// audio_emitter_system removes AudioPlaybackComponent when playing becomes false.
    #[test]
    fn audio_emitter_system_removes_component_when_stopped() {
        use crate::AudioCommand;
        use std::sync::{Arc, atomic::AtomicU32};

        let (tx, _rx) = std::sync::mpsc::channel::<AudioCommand>();
        let handle = AudioHandle {
            sender: tx,
            next_id: Arc::new(AtomicU32::new(1)),
        };

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

        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_none(),
            "AudioPlaybackComponent should be removed when playing=false"
        );
    }

    #[test]
    fn tick_culls_over_budget_sources() {
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
}
```

---

### Task 3: Wire `AudioHandle` into `engine_runner.rs`

- [ ] Add `audio_handle: Option<vox_audio::AudioHandle>` field to `EngineApp`
- [ ] Initialize it in `EngineApp::new()` (or equivalent constructor) alongside `AudioEngine`
- [ ] Verify `cargo check -p vox_app` passes

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Reading notes:** In `engine_runner.rs`, `EngineApp` (lines 71–181) already has `audio: AudioEngine` (line 152). The `AudioEngine` is constructed at line 323 with `init_backend()`. The `AudioPlugin`/ECS path is not used in `EngineApp` — it uses `AudioEngine` directly for legacy sine playback. The correct wiring is:

1. Add `audio_handle` field to `EngineApp`:

```rust
// In EngineApp struct, after `audio: AudioEngine,`:
audio_handle: Option<vox_audio::AudioHandle>,
```

2. Initialize in the constructor block (around line 323), after `audio: { ... }`:

```rust
audio_handle: vox_audio::AudioHandle::spawn(),
```

3. To play a file via `AudioEmitterComponent` from the ECS path, pass `audio_handle.as_ref()` to any system that needs it, or use it directly:

```rust
// Example: play an asset file on a keypress
if let Some(ref handle) = self.audio_handle {
    handle.play("assets/audio/click.wav", 0.5, false);
}
```

4. Keep the existing `self.audio.play_sine_backend(...)` calls at line 1124 — `play_sine_backend` is now a no-op stub, so migrate those call sites to `SpatialAudioManager` or generate a wav with `generate_click()` and play via `audio_handle`.

**Verification command:**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

Expected output: no errors (warnings are acceptable).

---

## Integration Notes

### `AudioHandle` field visibility in tests

The `AudioHandle` struct fields (`sender`, `next_id`) are private. Tests that need to construct a dummy handle directly (without spawning a thread) must be `#[cfg(test)]` inside `vox_audio` and use `pub(crate)` visibility, **or** the test helpers construct the handle via a `pub fn new_test(tx, next_id)` constructor added inside a `#[cfg(test)]` block:

```rust
#[cfg(test)]
impl AudioHandle {
    pub fn new_test(
        sender: std::sync::mpsc::Sender<AudioCommand>,
        next_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    ) -> Self {
        Self { sender, next_id }
    }
}
```

Add this block to `lib.rs` so the `ecs.rs` tests can build `AudioHandle` without touching private fields.

### Feature flag

The `audio-backend` feature in `crates/vox_audio/Cargo.toml` remains unchanged. `AudioHandle::spawn()` returns `None` when the feature is off. All ECS systems gracefully skip when the handle is `None`.

### Looping `.ogg` files

`rodio::Decoder` supports `.ogg` (via the `vorbis` feature in rodio, enabled by default in 0.19). No extra dependencies are needed.

### Thread lifetime

`AudioThread::run()` blocks on `receiver.recv()`. When all `AudioHandle` clones are dropped, the `Sender` closes, `recv()` returns `Err`, and the thread exits cleanly. No explicit shutdown signal is needed.
