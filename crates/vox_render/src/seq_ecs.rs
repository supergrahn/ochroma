//! ECS integration for the Sequencer and RigidStateMachine.
//!
//! Add `SequencePlayerComponent` to any entity whose `TransformComponent`
//! should be driven by a `Sequence` timeline.
//!
//! Add `RigidAnimationComponent` to any entity whose `TransformComponent`
//! should be driven by a `RigidStateMachine`.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::lod_ecs::TimeStep;
use crate::sequencer::{KeyframeValue, Sequence};
use crate::rigid_animation::RigidStateMachine;

// ── Components ─────────────────────────────────────────────────────────────

/// Drives an entity's TransformComponent from a keyframed Sequence timeline.
///
/// On spawn: set `playing = true` to start. The system writes the first
/// Transform-valued track's interpolated value to the entity each frame.
/// The driving system multiplies delta-time by `sequence.playback_speed` when advancing `current_time`.
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
    /// Only returns true after the system has set `playing = false` on completion,
    /// so a freshly constructed (never-played) player does not appear finished.
    pub fn is_finished(&self) -> bool {
        !self.looping && !self.playing && self.current_time >= self.sequence.duration
    }
}

/// Drives an entity's TransformComponent from a `RigidStateMachine`.
///
/// Each frame the state machine is ticked and the resulting keyframe
/// position/rotation/scale is written to the entity's `TransformComponent`.
#[derive(Component, Debug, Clone)]
pub struct RigidAnimationComponent {
    pub machine: RigidStateMachine,
}

// ── Systems ────────────────────────────────────────────────────────────────

/// Advance each sequence's playback cursor and write the first Transform-valued
/// track result to the entity's TransformComponent.
///
/// - Skips entities where `playing == false`.
/// - Clamps to duration and sets `playing = false` for non-looping sequences.
/// - Wraps time around `duration` for looping sequences.
/// - Multiplies delta-time by `sequence.playback_speed`.
///
/// # Panics / Limitations
/// `sequence.playback_speed` must be ≥ 0.0. Negative values are not supported.
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
                break; // Apply the first Transform track (tracks are ordered by insertion)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn sequence_advances_transform() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;
        use crate::sequencer::{TrackType, SequenceKeyframe, Interpolation};

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
        use crate::sequencer::{TrackType, SequenceKeyframe, Interpolation};

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

    #[test]
    fn sequence_loops_correctly() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;
        use crate::sequencer::{TrackType, SequenceKeyframe, Interpolation};

        let mut world = World::new();
        world.insert_resource(TimeStep(1.5)); // tick past the 1-second duration

        let mut seq = Sequence::new("loop", 1.0);
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
            time: 1.0,
            value: KeyframeValue::Transform {
                position: [0.0, 5.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale:    [1.0, 1.0, 1.0],
            },
            interpolation: Interpolation::Linear,
        });

        let mut player = SequencePlayerComponent::new(seq);
        player.looping = true;
        player.play();

        let entity = world.spawn((
            player,
            vox_core::ecs::TransformComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(sequence_player_system);
        schedule.run(&mut world);

        let player = world.entity(entity).get::<SequencePlayerComponent>().unwrap();
        // After 1.5s tick on a 1s sequence, current_time should wrap to 0.5s
        assert!(player.playing, "looping sequence should still be playing");
        assert!(
            player.current_time < 1.0,
            "time should have wrapped, current_time={}",
            player.current_time
        );
    }
}
