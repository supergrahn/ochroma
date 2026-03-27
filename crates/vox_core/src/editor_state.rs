/// Editor state machine for Play/Pause/Stop workflow.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    Editing,
    Playing,
    Paused,
}

/// Manages editor state transitions and world state save/restore for play mode.
pub struct EditorStateMachine {
    pub mode: EditorMode,
    saved_state: Option<Vec<u8>>,
}

impl EditorStateMachine {
    pub fn new() -> Self {
        Self {
            mode: EditorMode::Editing,
            saved_state: None,
        }
    }

    /// Enter play mode. Saves current state for later restore.
    pub fn play(&mut self, world_state: Vec<u8>) {
        if self.mode == EditorMode::Editing {
            self.saved_state = Some(world_state);
            self.mode = EditorMode::Playing;
        }
    }

    /// Pause play mode (toggle).
    pub fn pause(&mut self) {
        if self.mode == EditorMode::Playing {
            self.mode = EditorMode::Paused;
        } else if self.mode == EditorMode::Paused {
            self.mode = EditorMode::Playing;
        }
    }

    /// Stop play mode. Returns the saved state to restore.
    pub fn stop(&mut self) -> Option<Vec<u8>> {
        if self.mode == EditorMode::Playing || self.mode == EditorMode::Paused {
            self.mode = EditorMode::Editing;
            self.saved_state.take()
        } else {
            None
        }
    }

    pub fn is_editing(&self) -> bool {
        self.mode == EditorMode::Editing
    }
    pub fn is_playing(&self) -> bool {
        self.mode == EditorMode::Playing
    }
    pub fn is_paused(&self) -> bool {
        self.mode == EditorMode::Paused
    }
    pub fn should_tick_game(&self) -> bool {
        self.mode == EditorMode::Playing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_transition_cycle() {
        let mut sm = EditorStateMachine::new();
        assert!(sm.is_editing());

        // editing -> playing
        sm.play(vec![1, 2, 3]);
        assert!(sm.is_playing());
        assert!(sm.should_tick_game());

        // playing -> paused
        sm.pause();
        assert!(sm.is_paused());
        assert!(!sm.should_tick_game());

        // paused -> playing (toggle)
        sm.pause();
        assert!(sm.is_playing());

        // playing -> stop -> editing
        let restored = sm.stop();
        assert!(sm.is_editing());
        assert_eq!(restored, Some(vec![1, 2, 3]));
    }

    #[test]
    fn saved_state_returned_on_stop() {
        let mut sm = EditorStateMachine::new();
        let data = vec![10, 20, 30, 40];
        sm.play(data.clone());
        let restored = sm.stop();
        assert_eq!(restored, Some(data));
    }

    #[test]
    fn cant_play_while_already_playing() {
        let mut sm = EditorStateMachine::new();
        sm.play(vec![1]);
        // Try to play again — should be ignored, saved state unchanged
        sm.play(vec![99]);
        let restored = sm.stop();
        assert_eq!(restored, Some(vec![1]));
    }

    #[test]
    fn should_tick_game_only_in_playing() {
        let mut sm = EditorStateMachine::new();
        assert!(!sm.should_tick_game()); // editing

        sm.play(vec![]);
        assert!(sm.should_tick_game()); // playing

        sm.pause();
        assert!(!sm.should_tick_game()); // paused

        sm.stop();
        assert!(!sm.should_tick_game()); // editing again
    }

    #[test]
    fn stop_while_editing_returns_none() {
        let mut sm = EditorStateMachine::new();
        assert_eq!(sm.stop(), None);
    }

    #[test]
    fn stop_from_paused() {
        let mut sm = EditorStateMachine::new();
        sm.play(vec![5, 6]);
        sm.pause();
        let restored = sm.stop();
        assert!(sm.is_editing());
        assert_eq!(restored, Some(vec![5, 6]));
    }
}
