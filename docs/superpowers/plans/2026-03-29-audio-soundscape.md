# Audio Soundscape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Soundscape` ECS resource to `vox_app` that manages ambient and positional sound layers using the existing `vox_audio` ECS infrastructure, with a `SoundscapePlugin` that spawns emitter entities and toggles playback.

**Architecture:** The existing `crates/vox_app/src/soundscape.rs` has an `AmbientLayer`/`Soundscape` system oriented around city-builder concepts (population, weather, construction). This plan replaces it with a generic engine-layer `SoundLayer` + `Soundscape` model that any game type can use, plus a `SoundscapePlugin` that bridges to `vox_audio::ecs::AudioEmitterComponent`. The existing `vox_audio::ecs` module provides `AudioEmitterComponent`, `AudioPlaybackComponent`, `AudioPlugin`, and the emitter/tick systems. The soundscape plugin spawns one entity per `SoundLayer` with the appropriate components and manages the `playing` flag.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `vox_audio::ecs::{AudioEmitterComponent, AudioPlugin}`, `vox_core::ecs::TransformComponent`

---

## Key Files (read before editing)

- `crates/vox_app/src/soundscape.rs` — existing `Soundscape` with city-builder layers
- `crates/vox_audio/src/ecs.rs` — `AudioEmitterComponent`, `AudioPlaybackComponent`, `AudioPlugin`, `audio_emitter_system`, `audio_tick_system`
- `crates/vox_audio/src/lib.rs` — `AudioEngine`, `AudioSource`, feature-gated `AudioBackend`
- `crates/vox_core/src/ecs.rs` — `AudioEmitterComponent { clip_path, volume, looping, playing, spatial }`, `TransformComponent`
- `crates/vox_app/Cargo.toml` — already depends on `vox_audio`, `bevy_ecs`, `bevy_app`
- `crates/vox_app/src/lib.rs` — already has `pub mod soundscape;`

## File Structure

**Modify:**
- `crates/vox_app/src/soundscape.rs` — replace existing city-specific soundscape with generic `SoundLayer` + `Soundscape` + `SoundscapePlugin`

**No new files required.**

---

### Task 1: Generic `SoundLayer` and `Soundscape` structs

**Files:**
- Modify: `crates/vox_app/src/soundscape.rs`

Replace the existing city-builder-oriented `Soundscape` with a generic version.

- [ ] **Step 1: Write the complete new module with tests** — replace the entire contents of `crates/vox_app/src/soundscape.rs`:

```rust
//! Generic ambient soundscape system.
//!
//! `SoundLayer` defines an individual audio layer (wind, birds, music, etc.).
//! `Soundscape` manages a collection of layers and an active toggle.
//! `SoundscapePlugin` bridges to `vox_audio`'s ECS emitter system.

use bevy_ecs::prelude::*;
use glam::Vec3;

// ── Data types ────────────────────────────────────────────────────────────

/// A single sound layer in the soundscape.
#[derive(Debug, Clone)]
pub struct SoundLayer {
    /// Human-readable name (e.g. "wind", "birds").
    pub name: String,
    /// Path to the audio clip asset.
    pub clip_path: String,
    /// Volume in [0, 1].
    pub volume: f32,
    /// Whether this layer loops continuously.
    pub looping: bool,
    /// Whether this layer uses spatial (3D) audio.
    pub spatial: bool,
}

/// Manages ambient sound layers with an active toggle.
#[derive(Debug, Clone)]
pub struct Soundscape {
    pub layers: Vec<SoundLayer>,
    pub active: bool,
}

impl Soundscape {
    /// Create an empty soundscape.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            active: true,
        }
    }

    /// Add a layer to the soundscape.
    pub fn add_layer(&mut self, layer: SoundLayer) {
        self.layers.push(layer);
    }

    /// Remove a layer by name. Returns true if found and removed.
    pub fn remove_layer(&mut self, name: &str) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.name != name);
        self.layers.len() < before
    }

    /// Create a default outdoor soundscape with 3 layers: wind, distant_traffic, birds.
    pub fn outdoor_default() -> Self {
        Self {
            layers: vec![
                SoundLayer {
                    name: "wind".into(),
                    clip_path: "audio/ambient/wind_loop.ogg".into(),
                    volume: 0.3,
                    looping: true,
                    spatial: false,
                },
                SoundLayer {
                    name: "distant_traffic".into(),
                    clip_path: "audio/ambient/traffic_distant.ogg".into(),
                    volume: 0.15,
                    looping: true,
                    spatial: false,
                },
                SoundLayer {
                    name: "birds".into(),
                    clip_path: "audio/ambient/birds_morning.ogg".into(),
                    volume: 0.2,
                    looping: true,
                    spatial: true,
                },
            ],
            active: true,
        }
    }
}

impl Default for Soundscape {
    fn default() -> Self {
        Self::new()
    }
}

// ── ECS Integration ───────────────────────────────────────────────────────

/// bevy_ecs Resource wrapping `Soundscape`.
#[derive(Resource)]
pub struct SoundscapeResource(pub Soundscape);

/// Marker component on entities spawned by the soundscape system.
#[derive(Component, Debug)]
pub struct SoundscapeLayerMarker {
    pub layer_name: String,
}

/// System that spawns/despawns emitter entities to match `SoundscapeResource.layers`
/// and sets `AudioEmitterComponent.playing` based on `Soundscape.active`.
pub fn soundscape_sync_system(
    mut commands: Commands,
    soundscape: Res<SoundscapeResource>,
    existing: Query<(Entity, &SoundscapeLayerMarker)>,
) {
    let active = soundscape.0.active;

    // Collect existing layer names
    let existing_names: Vec<(Entity, String)> = existing
        .iter()
        .map(|(e, m)| (e, m.layer_name.clone()))
        .collect();

    // Despawn entities whose layer was removed
    for (entity, name) in &existing_names {
        if !soundscape.0.layers.iter().any(|l| l.name == *name) {
            commands.entity(*entity).despawn();
        }
    }

    // Spawn missing layers, update playing state on existing ones
    for layer in &soundscape.0.layers {
        let already_spawned = existing_names.iter().any(|(_, n)| *n == layer.name);
        if !already_spawned {
            commands.spawn((
                SoundscapeLayerMarker {
                    layer_name: layer.name.clone(),
                },
                vox_core::ecs::AudioEmitterComponent {
                    clip_path: layer.clip_path.clone(),
                    volume: layer.volume,
                    looping: layer.looping,
                    playing: active,
                    spatial: layer.spatial,
                },
                vox_core::ecs::TransformComponent::default(),
            ));
        }
    }
}

/// System that updates `AudioEmitterComponent.playing` on all soundscape entities
/// when `Soundscape.active` changes.
pub fn soundscape_toggle_system(
    soundscape: Res<SoundscapeResource>,
    mut query: Query<&mut vox_core::ecs::AudioEmitterComponent, With<SoundscapeLayerMarker>>,
) {
    let active = soundscape.0.active;
    for mut emitter in query.iter_mut() {
        emitter.playing = active;
    }
}

/// Bevy plugin that registers the soundscape ECS resources and systems.
///
/// Usage:
/// ```rust,ignore
/// app.add_plugins(SoundscapePlugin::new(Soundscape::outdoor_default()));
/// ```
pub struct SoundscapePlugin {
    pub initial: Soundscape,
}

impl SoundscapePlugin {
    pub fn new(initial: Soundscape) -> Self {
        Self { initial }
    }

    pub fn with_outdoor_default() -> Self {
        Self::new(Soundscape::outdoor_default())
    }
}

impl bevy_app::Plugin for SoundscapePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(SoundscapeResource(self.initial.clone()));
        app.add_systems(
            bevy_app::Update,
            (soundscape_sync_system, soundscape_toggle_system).chain(),
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outdoor_default_has_3_layers() {
        let s = Soundscape::outdoor_default();
        assert_eq!(s.layers.len(), 3, "outdoor default should have 3 layers");
        assert!(s.active, "should be active by default");
    }

    #[test]
    fn add_layer_increases_count() {
        let mut s = Soundscape::new();
        assert_eq!(s.layers.len(), 0);
        s.add_layer(SoundLayer {
            name: "test".into(),
            clip_path: "test.ogg".into(),
            volume: 0.5,
            looping: false,
            spatial: false,
        });
        assert_eq!(s.layers.len(), 1);
    }

    #[test]
    fn remove_layer_decreases_count() {
        let mut s = Soundscape::outdoor_default();
        assert_eq!(s.layers.len(), 3);
        let removed = s.remove_layer("wind");
        assert!(removed, "should find and remove 'wind'");
        assert_eq!(s.layers.len(), 2);
    }

    #[test]
    fn remove_nonexistent_layer_returns_false() {
        let mut s = Soundscape::outdoor_default();
        let removed = s.remove_layer("nonexistent");
        assert!(!removed);
        assert_eq!(s.layers.len(), 3);
    }

    #[test]
    fn soundscape_plugin_builds_without_panic() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(SoundscapePlugin::with_outdoor_default());
        assert!(app.world().contains_resource::<SoundscapeResource>());
    }

    #[test]
    fn soundscape_sync_spawns_entities() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        world.insert_resource(SoundscapeResource(Soundscape::outdoor_default()));

        let mut schedule = Schedule::default();
        schedule.add_systems(soundscape_sync_system);
        schedule.run(&mut world);
        // Flush deferred commands
        world.flush();

        let count = world
            .query::<&SoundscapeLayerMarker>()
            .iter(&world)
            .count();
        assert_eq!(count, 3, "should spawn 3 entities for 3 layers");
    }

    #[test]
    fn soundscape_toggle_sets_playing() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        let mut ss = Soundscape::outdoor_default();
        ss.active = false;
        world.insert_resource(SoundscapeResource(ss));

        // Spawn a layer entity manually
        world.spawn((
            SoundscapeLayerMarker { layer_name: "wind".into() },
            vox_core::ecs::AudioEmitterComponent {
                clip_path: "wind.ogg".into(),
                volume: 0.3,
                looping: true,
                playing: true, // start as playing
                spatial: false,
            },
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(soundscape_toggle_system);
        schedule.run(&mut world);

        let emitter = world
            .query::<&vox_core::ecs::AudioEmitterComponent>()
            .iter(&world)
            .next()
            .unwrap();
        assert!(!emitter.playing, "playing should be false since soundscape.active=false");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_app --lib -- soundscape::tests 2>&1 | tail -15
```

Expected: all 7 tests pass.

- [ ] **Step 3: Verify full crate compiles**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

Expected: successful check. The `#[cfg(feature = "audio-backend")]` gate on `AudioPlugin` does not affect `SoundscapePlugin` since it only uses `AudioEmitterComponent` (always available).

- [ ] **Step 4: Commit**

```bash
git add crates/vox_app/src/soundscape.rs
git commit -m "feat(soundscape): generic SoundLayer + Soundscape + SoundscapePlugin with ECS sync"
```

---

### Task 2: Wire soundscape toggle to M key

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs` (or the main app initialization)

Add the `SoundscapePlugin` to the app and toggle `SoundscapeResource.active` on M key press.

- [ ] **Step 1: Add plugin registration**

In the app builder / plugin registration section:

```rust
use crate::soundscape::{Soundscape, SoundscapePlugin, SoundscapeResource};

// In plugin setup:
app.add_plugins(SoundscapePlugin::with_outdoor_default());
```

- [ ] **Step 2: Add keyboard toggle system**

```rust
/// Toggle soundscape active state. Called from the input handling section.
fn toggle_soundscape(soundscape: &mut SoundscapeResource) {
    soundscape.0.active = !soundscape.0.active;
    let state = if soundscape.0.active { "ON" } else { "OFF" };
    println!("[ochroma] Soundscape: {}", state);
}
```

In the input handling section, on `KeyCode::M` press, call `toggle_soundscape`.

- [ ] **Step 3: Verify compile**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

Expected: successful check.

- [ ] **Step 4: Commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(soundscape): wire SoundscapePlugin + M-key toggle in engine runner"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All tasks covered (SoundLayer/Soundscape, SoundscapePlugin, M-key toggle)
- [x] **No placeholders:** All code blocks are complete
- [x] **Type consistency:** Uses `vox_core::ecs::AudioEmitterComponent` directly, `SoundLayer` maps 1:1 to emitter fields
- [x] **TDD:** 7 tests covering data structures, plugin init, entity spawning, toggle behavior
- [x] **Feature gate safety:** No dependency on `audio-backend` feature; uses only always-available ECS components
- [x] **Engine generality:** Replaced city-builder soundscape with generic layers; `outdoor_default()` is game-agnostic
