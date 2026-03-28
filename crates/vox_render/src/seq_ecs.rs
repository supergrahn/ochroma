//! ECS integration for the Sequencer and RigidStateMachine.
//!
//! Add `SequencePlayerComponent` to any entity whose `TransformComponent`
//! should be driven by a `Sequence` timeline.
//!
//! Add `RigidAnimationComponent` to any entity whose `TransformComponent`
//! should be driven by a `RigidStateMachine`.

use bevy_ecs::prelude::*;

use crate::sequencer::Sequence;
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
}
