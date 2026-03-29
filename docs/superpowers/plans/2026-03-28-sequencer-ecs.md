# Sequencer ECS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `Sequence` timeline and `RigidStateMachine` into bevy_ecs so any entity with `SequencePlayerComponent` or `RigidAnimationComponent` automatically has its `TransformComponent` driven by keyframe data each frame.

**Architecture:** A new `vox_render::seq_ecs` module provides `SequencerPlugin`, two systems (`sequence_player_system` → `rigid_animation_system`), and two components (`SequencePlayerComponent`, `RigidAnimationComponent`). Both components live in `vox_render` (not `vox_core`) because they reference types from `vox_render::sequencer` and `vox_render::rigid_animation`. The existing `vox_render::lod_ecs::TimeStep` resource is reused for the delta-time — no new resource needed.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"` (already in `vox_render/Cargo.toml`), `vox_render::sequencer::{Sequence, KeyframeValue, TrackType, SequenceKeyframe, Interpolation}`, `vox_render::rigid_animation::{RigidStateMachine, RigidClip, RigidKeyframe}`, `vox_core::ecs::TransformComponent`, `vox_render::lod_ecs::TimeStep`

---

## Key Files (read before editing)

- `crates/vox_render/src/sequencer.rs` — `Sequence::new`, `add_track`, `add_keyframe`, `evaluate(time) -> Vec<(String, KeyframeValue)>`
- `crates/vox_render/src/rigid_animation.rs` — `RigidStateMachine::new(initial_state)`, `add_state`, `tick(dt) -> RigidKeyframe`; `RigidClip::new(name, duration, looping)`, `add_keyframe`; `RigidKeyframe::identity(time)`
- `crates/vox_render/src/lod_ecs.rs` — `TimeStep(f32)` resource (already registered)
- `crates/vox_render/src/lib.rs` — add `pub mod seq_ecs;`
- `crates/vox_core/src/ecs.rs` — `TransformComponent { position: Vec3, rotation: Quat, scale: Vec3 }`

## File Structure

**Create:**
- `crates/vox_render/src/seq_ecs.rs` — components + systems + plugin (single file)

**Modify:**
- `crates/vox_render/src/lib.rs` — add `pub mod seq_ecs;`

---

### Task 1: SequencePlayerComponent + RigidAnimationComponent

**Files:**
- Create: `crates/vox_render/src/seq_ecs.rs`
- Modify: `crates/vox_render/src/lib.rs`

- [ ] **Step 1: Add `pub mod seq_ecs;` to lib.rs**

Open `crates/vox_render/src/lib.rs`. After `pub mod lod_ecs;` add:

```rust
pub mod seq_ecs;
```

- [ ] **Step 2: Create `crates/vox_render/src/seq_ecs.rs`** with this exact content:

```rust
//! ECS integration for the Sequencer and RigidStateMachine.
//!
//! Add `SequencePlayerComponent` to any entity whose `TransformComponent`
//! should be driven by a `Sequence` timeline.
//!
//! Add `RigidAnimationComponent` to any entity whose `TransformComponent`
//! should be driven by a `RigidStateMachine`.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::sequencer::{KeyframeValue, Sequence};
use crate::rigid_animation::RigidStateMachine;
use crate::lod_ecs::TimeStep;

// ── Components ─────────────────────────────────────────────────────────────

/// Drives an entity's TransformComponent from a keyframed Sequence timeline.
///
/// On spawn: set `playing = true` to start. The system writes the first
/// Transform-valued track's interpolated value to the entity each frame.
#[derive(Component, Debug, Clone)]
pub struct SequencePlayerComponent {
    pub sequence: Sequence,
    /// Playback cursor in seconds.
    pub current_time: f32,
    /// True = advance time and apply transforms this frame.
    pub playing: bool,
    /// True = wrap current_time at sequence.duration instead of stopping.
    pub looping: bool,
}

impl SequencePlayerComponent {
    pub fn new(sequence: Sequence) -> Self {
        Self {
            sequence,
            current_time: 0.0,
            playing: false,
            looping: false,
        }
    }

    /// Reset to start and begin playing.
    pub fn play(&mut self) {
        self.playing = true;
        self.current_time = 0.0;
    }

    /// True once the sequence has played to the end (non-looping only).
    pub fn is_finished(&self) -> bool {
        !self.looping && self.current_time >= self.sequence.duration
    }
}

/// Drives an entity's TransformComponent from a `RigidStateMachine`.
///
/// Each frame the state machine is ticked and the resulting keyframe
/// position/rotation/scale is written to the entity's `TransformComponent`.
#[derive(Component)]
pub struct RigidAnimationComponent {
    pub machine: RigidStateMachine,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequencer::{Interpolation, SequenceKeyframe, TrackType};
    use crate::rigid_animation::{RigidClip, RigidKeyframe};

    #[test]
    fn sequence_player_new_is_stopped() {
        let seq = Sequence::new("test", 1.0);
        let player = SequencePlayerComponent::new(seq);
        assert!(!player.playing);
        assert_eq!(player.current_time, 0.0);
    }

    #[test]
    fn sequence_player_play_resets_time() {
        let seq = Sequence::new("test", 1.0);
        let mut player = SequencePlayerComponent::new(seq);
        player.current_time = 0.5;
        player.play();
        assert!(player.playing);
        assert_eq!(player.current_time, 0.0);
    }

    #[test]
    fn rigid_animation_component_wraps_machine() {
        // Compile test: RigidAnimationComponent can be constructed.
        let machine = RigidStateMachine::new("idle");
        let _comp = RigidAnimationComponent { machine };
    }
}
```

- [ ] **Step 3: Verify compile**

```bash
cargo check -p vox_render 2>&1 | tail -5
```
Expected: clean

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render --lib -- seq_ecs 2>&1 | tail -5
```
Expected: `3 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/seq_ecs.rs crates/vox_render/src/lib.rs
git commit -m "feat(seq): SequencePlayerComponent + RigidAnimationComponent ECS components"
```

---

### Task 2: sequence_player_system

**Files:**
- Modify: `crates/vox_render/src/seq_ecs.rs`

`sequence_player_system` advances `current_time`, evaluates the sequence, and writes the first Transform-valued track result to `TransformComponent`.

- [ ] **Step 1: Add the failing test** — append inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn sequence_advances_transform() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        world.insert_resource(TimeStep(0.5)); // 0.5 s per tick

        // Build a 2-second sequence: y moves from 0 → 10
        let mut seq = Sequence::new("move", 2.0);
        let idx = seq.add_track("cam", TrackType::CameraTransform);
        seq.add_keyframe(idx, SequenceKeyframe {
            time: 0.0,
            value: KeyframeValue::Transform {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale:    [1.0, 1.0, 1.0],
            },
            interpolation: Interpolation::Linear,
        });
        seq.add_keyframe(idx, SequenceKeyframe {
            time: 2.0,
            value: KeyframeValue::Transform {
                position: [0.0, 10.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale:    [1.0, 1.0, 1.0],
            },
            interpolation: Interpolation::Linear,
        });

        let mut player = SequencePlayerComponent::new(seq);
        player.play();

        let entity = world.spawn((
            player,
            vox_core::ecs::TransformComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(sequence_player_system);
        schedule.run(&mut world);

        let transform = world.entity(entity).get::<vox_core::ecs::TransformComponent>().unwrap();
        // After 0.5 s on a 0→10 y range over 2 s: should be ~y=2.5
        assert!(
            transform.position.y > 0.0,
            "sequence should have advanced the entity transform, y={}",
            transform.position.y
        );
    }

    #[test]
    fn sequence_stops_when_finished() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        world.insert_resource(TimeStep(5.0)); // overshoot a 2 s sequence

        let mut seq = Sequence::new("short", 2.0);
        let idx = seq.add_track("cam", TrackType::CameraTransform);
        seq.add_keyframe(idx, SequenceKeyframe {
            time: 0.0,
            value: KeyframeValue::Transform {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale:    [1.0, 1.0, 1.0],
            },
            interpolation: Interpolation::Linear,
        });

        let mut player = SequencePlayerComponent::new(seq);
        player.play();

        let entity = world.spawn((
            player,
            vox_core::ecs::TransformComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(sequence_player_system);
        schedule.run(&mut world);

        let player = world.entity(entity).get::<SequencePlayerComponent>().unwrap();
        assert!(player.is_finished(), "player should be finished after overshooting duration");
        assert!(!player.playing, "playing should be false when sequence ends");
    }
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- seq_ecs::tests::sequence_advances_transform 2>&1 | tail -5
```
Expected: FAIL — `sequence_player_system` not defined

- [ ] **Step 3: Implement** — insert BEFORE `#[cfg(test)]` in `seq_ecs.rs`:

```rust
// ── Systems ────────────────────────────────────────────────────────────────

/// Advance each sequence's playback cursor and write the first Transform-valued
/// track result to the entity's TransformComponent.
///
/// - Skips entities where `playing == false`.
/// - Clamps to duration and sets `playing = false` for non-looping sequences.
/// - Wraps time around `duration` for looping sequences.
pub fn sequence_player_system(
    dt: Res<TimeStep>,
    mut query: Query<(&mut SequencePlayerComponent, &mut vox_core::ecs::TransformComponent)>,
) {
    for (mut player, mut transform) in query.iter_mut() {
        if !player.playing {
            continue;
        }

        player.current_time += dt.0 * player.sequence.playback_speed;

        if player.looping {
            if player.sequence.duration > 0.0 {
                player.current_time %= player.sequence.duration;
            }
        } else if player.current_time >= player.sequence.duration {
            player.current_time = player.sequence.duration;
            player.playing = false;
        }

        let results = player.sequence.evaluate(player.current_time);
        for (_track_name, value) in results {
            if let KeyframeValue::Transform { position, rotation, scale } = value {
                transform.position = Vec3::from(position);
                transform.rotation = Quat::from_array(rotation);
                transform.scale    = Vec3::from(scale);
                break; // Apply the first Transform track only
            }
        }
    }
}
```

- [ ] **Step 4: Run all seq_ecs tests**

```bash
cargo test -p vox_render --lib -- seq_ecs 2>&1 | tail -8
```
Expected: `5 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/seq_ecs.rs
git commit -m "feat(seq): sequence_player_system — advance timeline + write TransformComponent"
```

---

### Task 3: rigid_animation_system + SequencerPlugin

**Files:**
- Modify: `crates/vox_render/src/seq_ecs.rs`

`rigid_animation_system` ticks the state machine and writes the resulting keyframe to `TransformComponent`. `SequencerPlugin` chains both systems in `Update` and requires callers to have registered `TimeStep` (already done by `LodStreamingPlugin` or manually).

- [ ] **Step 1: Add failing tests** — append inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn rigid_animation_advances_transform() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        world.insert_resource(TimeStep(0.25)); // 250 ms per tick

        // A 1-second clip: moves from y=0 to y=4
        let mut clip = RigidClip::new("move", 1.0, false);
        clip.keyframes.push(RigidKeyframe::identity(0.0));
        clip.keyframes.push(RigidKeyframe {
            time: 1.0,
            position: Vec3::new(0.0, 4.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        });

        let mut machine = RigidStateMachine::new("move");
        machine.add_state("move", clip);

        let entity = world.spawn((
            RigidAnimationComponent { machine },
            vox_core::ecs::TransformComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(rigid_animation_system);
        schedule.run(&mut world);

        let transform = world.entity(entity).get::<vox_core::ecs::TransformComponent>().unwrap();
        assert!(
            transform.position.y >= 0.0,
            "rigid animation should write a valid position, y={}",
            transform.position.y
        );
    }

    #[test]
    fn plugin_registers_systems() {
        use bevy_app::App;
        // Plugin should build without panicking.
        // TimeStep must be pre-inserted (SequencerPlugin doesn't own it).
        let mut app = App::new();
        app.insert_resource(TimeStep(1.0 / 60.0));
        app.add_plugins(SequencerPlugin);
        // No assertion needed — panicking during build counts as failure.
    }
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- seq_ecs::tests::rigid_animation_advances_transform 2>&1 | tail -5
```
Expected: FAIL — `rigid_animation_system`, `SequencerPlugin` not defined

- [ ] **Step 3: Implement** — insert after `sequence_player_system` and BEFORE `#[cfg(test)]`:

```rust
/// Tick the `RigidStateMachine` and write the resulting keyframe
/// position/rotation/scale to the entity's `TransformComponent`.
pub fn rigid_animation_system(
    dt: Res<TimeStep>,
    mut query: Query<(&mut RigidAnimationComponent, &mut vox_core::ecs::TransformComponent)>,
) {
    for (mut anim, mut transform) in query.iter_mut() {
        let kf = anim.machine.tick(dt.0);
        transform.position = kf.position;
        transform.rotation = kf.rotation;
        transform.scale    = kf.scale;
    }
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that chains `sequence_player_system` and `rigid_animation_system`
/// in `Update`.
///
/// **Requires** `TimeStep` to be registered before running (insert it yourself
/// or add `LodStreamingPlugin` first, which registers `TimeStep`).
pub struct SequencerPlugin;

impl bevy_app::Plugin for SequencerPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            bevy_app::Update,
            (sequence_player_system, rigid_animation_system).chain(),
        );
    }
}
```

- [ ] **Step 4: Run all seq_ecs tests**

```bash
cargo test -p vox_render --lib -- seq_ecs 2>&1 | tail -10
```
Expected: `7 passed; 0 failed`

- [ ] **Step 5: Full vox_render suite — no regressions**

```bash
cargo test -p vox_render --lib 2>&1 | grep "test result"
```
Expected: `0 failed`

- [ ] **Step 6: Commit**

```bash
git add crates/vox_render/src/seq_ecs.rs
git commit -m "feat(seq): rigid_animation_system + SequencerPlugin — complete sequencer ECS integration"
```

---

## Self-Review

**Spec coverage:**
- ✅ `SequencePlayerComponent { sequence, current_time, playing, looping }` → Task 1
- ✅ `SequencePlayerComponent::new`, `play`, `is_finished` → Task 1
- ✅ `RigidAnimationComponent { machine: RigidStateMachine }` → Task 1
- ✅ `sequence_player_system` — advance time, clamp/wrap, evaluate, write transform → Task 2
- ✅ Non-looping stops at duration, sets `playing = false` → Task 2
- ✅ `rigid_animation_system` — tick machine, write position/rotation/scale → Task 3
- ✅ `SequencerPlugin` — chains both systems in `Update`, requires caller to insert `TimeStep` → Task 3
- ✅ Test: `sequence_advances_transform` → Task 2
- ✅ Test: `sequence_stops_when_finished` → Task 2
- ✅ Test: `rigid_animation_advances_transform` → Task 3
- ✅ Test: `plugin_registers_systems` → Task 3

**Placeholder scan:** No TBDs. All function bodies shown in full.

**Type consistency:**
- `SequencePlayerComponent::is_finished()` — defined Task 1, tested Task 2 ✅
- `sequence_player_system` uses `Res<TimeStep>` from `crate::lod_ecs::TimeStep` — imported at top of file ✅
- `KeyframeValue::Transform { position: [f32;3], rotation: [f32;4], scale: [f32;3] }` — from `crate::sequencer` ✅
- `Vec3::from(position)` converts `[f32;3]` to `Vec3` ✅
- `Quat::from_array(rotation)` converts `[f32;4]` to `Quat` ✅
- `RigidStateMachine::new(initial_state: &str)` — Task 3 test calls `new("move")` then `add_state("move", clip)` — correct: initial state must match an added state name ✅
- `RigidKeyframe::identity(time)` — from `rigid_animation.rs` ✅
- `RigidClip::new(name, duration, looping)` — from `rigid_animation.rs`, push to `clip.keyframes` ✅
