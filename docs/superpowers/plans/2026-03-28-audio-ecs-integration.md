# Audio ECS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `AudioEngine` and `AudioEmitterComponent` into a bevy_ecs `AudioPlugin` so any entity with `AudioEmitterComponent + TransformComponent` automatically starts/stops audio and receives distance-based volume culling each frame.

**Architecture:** A new `vox_audio::ecs` module provides `AudioPlugin`, two ordered systems (`audio_emitter_system` → `audio_tick_system`), and three resources (`AudioEngineResource`, `AudioListenerSettings`, `AudioTimeStep`). `AudioEngineResource` wraps the existing `AudioEngine` (no rodio backend — ECS manages source state only; backend init stays on the caller's main thread). `AudioPlaybackComponent` is an output component inserted on entities when they start playing, removed when they stop.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `vox_audio::AudioEngine`, `vox_audio::AudioSource`, `vox_core::ecs::{AudioEmitterComponent, TransformComponent}`

---

## Key Files (read before editing)

- `crates/vox_audio/Cargo.toml` — add `bevy_ecs`, `bevy_app` deps
- `crates/vox_audio/src/lib.rs` — add `pub mod ecs;`
- `crates/vox_audio/src/ecs.rs` — **CREATE**: all resources, components, systems, plugin
- `crates/vox_core/src/ecs.rs` — `AudioEmitterComponent { clip_path, volume, looping, playing, spatial }` + `TransformComponent { position: Vec3, ... }`
- `crates/vox_audio/src/lib.rs` — `AudioEngine::play(AudioSource) -> u32`, `stop(id)`, `tick(dt)`, `set_listener(pos)`, `active_count()`

## AudioEngine Send + Sync note

`AudioEngine` with the `audio-backend` feature enabled contains `Option<AudioBackend>`, and `AudioBackend` holds `rodio::OutputStream` which is `!Send`. **ECS integration always runs without the audio-backend feature** — no `libasound2-dev` required for tests. Backend init is the caller's responsibility on the main thread. The `impl Resource` below is safe because `AudioEngine` without the backend feature contains only `Vec<AudioSource>` and `Vec3`, both `Send + Sync`.

## File Structure

**Create:**
- `crates/vox_audio/src/ecs.rs` — resources + output component + systems + plugin

**Modify:**
- `crates/vox_audio/Cargo.toml` — add bevy_ecs + bevy_app
- `crates/vox_audio/src/lib.rs` — add `pub mod ecs;`

---

### Task 1: bevy deps + ecs.rs skeleton

**Files:**
- Modify: `crates/vox_audio/Cargo.toml`
- Modify: `crates/vox_audio/src/lib.rs`
- Create: `crates/vox_audio/src/ecs.rs`

- [ ] **Step 1: Add deps to vox_audio/Cargo.toml**

Open `crates/vox_audio/Cargo.toml`. In `[dependencies]` add:

```toml
bevy_ecs = { workspace = true }
bevy_app = { workspace = true }
```

- [ ] **Step 2: Add pub mod ecs to lib.rs**

Open `crates/vox_audio/src/lib.rs`. After the last `pub mod` line (e.g. `pub mod synth;`) add:

```rust
pub mod ecs;
```

- [ ] **Step 3: Create crates/vox_audio/src/ecs.rs**

```rust
//! bevy_ecs integration for vox_audio.
//!
//! Provides `AudioPlugin` which drives `AudioEngine` from ECS.
//!
//! # Backend note
//! Rodio backend (audio-backend feature) uses `OutputStream` which is `!Send`.
//! This ECS layer manages source state only. Backend init is the caller's
//! responsibility on the main thread before the ECS schedule runs.

use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::{AudioEngine, AudioSource};

// AudioEngine without the audio-backend feature contains only Vec<AudioSource>
// and Vec3, both Send + Sync. Safe to use as a Resource in that configuration.
#[cfg(not(feature = "audio-backend"))]
impl Resource for AudioEngine {}

// ── Resources ──────────────────────────────────────────────────────────────

/// Wraps AudioEngine as a bevy_ecs Resource.
/// Use this as the primary audio state in an ECS world.
#[derive(Resource)]
pub struct AudioEngineResource {
    pub engine: AudioEngine,
}

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
```

- [ ] **Step 4: Verify compile**

```bash
cargo check -p vox_audio 2>&1 | tail -5
```
Expected: no errors (there may be an unused-import warning for `AudioSource` — that's fine, it'll be used in Task 2).

- [ ] **Step 5: Run tests**

```bash
cargo test -p vox_audio --lib -- ecs 2>&1 | tail -5
```
Expected: `3 passed; 0 failed`

- [ ] **Step 6: Commit**

```bash
git add crates/vox_audio/Cargo.toml crates/vox_audio/src/lib.rs crates/vox_audio/src/ecs.rs
git commit -m "feat(audio): ecs skeleton — AudioEngineResource, AudioListenerSettings, AudioPlaybackComponent"
```

---

### Task 2: audio_emitter_system

**Files:**
- Modify: `crates/vox_audio/src/ecs.rs`

`audio_emitter_system` has two query branches:
1. **Start**: entities with `AudioEmitterComponent { playing: true }` and WITHOUT `AudioPlaybackComponent` → register in AudioEngine, insert `AudioPlaybackComponent`
2. **Stop**: entities with `AudioEmitterComponent { playing: false }` and WITH `AudioPlaybackComponent` → stop in AudioEngine, remove `AudioPlaybackComponent`

- [ ] **Step 1: Add failing tests** — append inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn playing_true_inserts_playback_component() {
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

    #[test]
    fn playing_false_removes_playback_component() {
        let mut world = World::new();
        let mut engine_res = AudioEngineResource::default();
        // Manually register a source to simulate prior playback
        let source = AudioSource {
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
                playing: false,  // wants to stop
                spatial: false,
            },
            vox_core::ecs::TransformComponent::default(),
            AudioPlaybackComponent { source_id },
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_emitter_system);
        schedule.run(&mut world);

        // Deferred remove may not be visible until world.flush()
        world.flush();
        assert!(
            world.entity(entity).get::<AudioPlaybackComponent>().is_none(),
            "AudioPlaybackComponent should be removed after playing=false"
        );
        let res = world.resource::<AudioEngineResource>();
        assert_eq!(res.engine.active_count(), 0);
    }
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test -p vox_audio --lib -- ecs::tests::playing_true_inserts_playback_component 2>&1 | tail -5
```
Expected: FAIL — `audio_emitter_system` not defined

- [ ] **Step 3: Implement audio_emitter_system** — insert BEFORE `#[cfg(test)]` in `ecs.rs`:

```rust
// ── Systems ────────────────────────────────────────────────────────────────

/// Start or stop audio sources based on AudioEmitterComponent.playing.
///
/// - playing=true  + no AudioPlaybackComponent → registers source, inserts component
/// - playing=false + AudioPlaybackComponent    → stops source, removes component
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
    // Start new sources
    for (entity, emitter, transform) in start_query.iter() {
        if !emitter.playing {
            continue;
        }
        let source = AudioSource {
            id: 0, // AudioEngine assigns the actual id
            position: transform.position,
            volume: emitter.volume,
            looping: emitter.looping,
            clip: emitter.clip_path.clone(),
        };
        let source_id = engine.engine.play(source);
        commands.entity(entity).insert(AudioPlaybackComponent { source_id });
    }

    // Stop removed sources
    for (entity, emitter, playback) in stop_query.iter() {
        if emitter.playing {
            continue;
        }
        engine.engine.stop(playback.source_id);
        commands.entity(entity).remove::<AudioPlaybackComponent>();
    }
}
```

- [ ] **Step 4: Run all ecs tests**

```bash
cargo test -p vox_audio --lib -- ecs 2>&1 | tail -8
```
Expected: `5 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
git add crates/vox_audio/src/ecs.rs
git commit -m "feat(audio): audio_emitter_system — start/stop sources from AudioEmitterComponent.playing"
```

---

### Task 3: audio_tick_system + AudioPlugin

**Files:**
- Modify: `crates/vox_audio/src/ecs.rs`

`audio_tick_system` sets the listener position on `AudioEngine` then calls `tick(dt)`, which applies distance attenuation and evicts lowest-priority sources over budget. `AudioPlugin` wires everything together.

- [ ] **Step 1: Add failing tests** — append inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn tick_culls_over_budget_sources() {
        let mut world = World::new();
        // Budget of 1 — second source should be evicted on tick
        let mut res = AudioEngineResource { engine: AudioEngine::new(1) };
        res.engine.play(AudioSource { id: 0, position: Vec3::ZERO, volume: 1.0, looping: false, clip: "a.wav".into() });
        res.engine.play(AudioSource { id: 0, position: Vec3::ZERO, volume: 0.5, looping: false, clip: "b.wav".into() });
        assert_eq!(res.engine.active_count(), 2, "pre-condition: 2 sources before tick");
        world.insert_resource(res);
        world.insert_resource(AudioListenerSettings::default());
        world.insert_resource(AudioTimeStep(0.016));

        let mut schedule = Schedule::default();
        schedule.add_systems(audio_tick_system);
        schedule.run(&mut world);

        let res = world.resource::<AudioEngineResource>();
        assert_eq!(res.engine.active_count(), 1, "tick should cull to max_sources=1");
    }

    #[test]
    fn plugin_inserts_resources() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(AudioPlugin::default());
        assert!(app.world().contains_resource::<AudioEngineResource>());
        assert!(app.world().contains_resource::<AudioListenerSettings>());
        assert!(app.world().contains_resource::<AudioTimeStep>());
    }
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_audio --lib -- ecs::tests::tick_culls_over_budget_sources 2>&1 | tail -5
```
Expected: FAIL

- [ ] **Step 3: Implement audio_tick_system and AudioPlugin** — insert after `audio_emitter_system` and BEFORE `#[cfg(test)]`:

```rust
/// Update listener position and advance AudioEngine by one timestep.
/// Evicts lowest-priority sources over the engine's max_sources budget.
pub fn audio_tick_system(
    dt: Res<AudioTimeStep>,
    listener: Res<AudioListenerSettings>,
    mut engine: ResMut<AudioEngineResource>,
) {
    engine.engine.set_listener(listener.position);
    engine.engine.tick(dt.0);
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that registers audio ECS resources and systems.
///
/// Usage: `app.add_plugins(AudioPlugin::default())`
///
/// Update `AudioListenerSettings.position` each frame from the camera transform.
/// Set `AudioEmitterComponent.playing = true` to start a sound,
/// `false` to stop it.
#[derive(Default)]
pub struct AudioPlugin;

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
```

- [ ] **Step 4: Run all ecs tests**

```bash
cargo test -p vox_audio --lib -- ecs 2>&1 | tail -10
```
Expected: `7 passed; 0 failed`

- [ ] **Step 5: Run full vox_audio suite to check no regressions**

```bash
cargo test -p vox_audio 2>&1 | grep -E "FAILED|^test result"
```
Expected: `0 failed` (8 existing + 7 new = 15 total)

- [ ] **Step 6: Commit**

```bash
git add crates/vox_audio/src/ecs.rs
git commit -m "feat(audio): audio_tick_system + AudioPlugin — complete audio ECS integration"
```

---

## Self-Review

**Spec coverage:**
- ✅ `AudioEngineResource` wraps `AudioEngine` → Task 1
- ✅ `AudioListenerSettings { position: Vec3 }` → Task 1
- ✅ `AudioTimeStep(f32)` → Task 1
- ✅ `AudioPlaybackComponent { source_id: u32 }` → Task 1
- ✅ `audio_emitter_system` — start when `playing=true`, stop when `playing=false` → Task 2
- ✅ `audio_tick_system` — sets listener, calls `engine.tick()` → Task 3
- ✅ `AudioPlugin` — inserts 3 resources, chains 2 systems in `Update` → Task 3
- ✅ Test: `playing_true_inserts_playback_component` → Task 2
- ✅ Test: `playing_false_removes_playback_component` → Task 2
- ✅ Test: `tick_culls_over_budget_sources` → Task 3
- ✅ Test: `plugin_inserts_resources` → Task 3

**Placeholder scan:** No TBDs. All function bodies shown in full.

**Type consistency:**
- `AudioEngineResource { engine: AudioEngine }` — defined Task 1, used Tasks 2/3 ✅
- `AudioPlaybackComponent { source_id: u32 }` — defined Task 1, inserted/removed Task 2 ✅
- `AudioListenerSettings { position: Vec3 }` — defined Task 1, read Task 3 ✅
- `AudioTimeStep(f32)` — defined Task 1, read Task 3 ✅
- `AudioEngine::play(AudioSource) -> u32` — from lib.rs, called Task 2 ✅
- `AudioEngine::stop(id: u32)` — from lib.rs, called Task 2 ✅
- `AudioEngine::set_listener(pos: Vec3)` — from lib.rs, called Task 3 ✅
- `AudioEngine::tick(dt: f32)` — from lib.rs, called Task 3 ✅
- `world.flush()` in `playing_false_removes_playback_component` — needed because `commands.remove` is deferred ✅
