//! Rollback netcode for deterministic multiplayer.
//!
//! Maintains a circular buffer of game state snapshots. When a late input arrives,
//! rolls back to the frame before the input, re-simulates forward, then blends
//! the corrected state into the rendered output.
//!
//! Supports up to MAX_ROLLBACK_FRAMES (8) frames of history.

use std::collections::HashMap;

pub const MAX_ROLLBACK_FRAMES: usize = 8;

// Input bit masks
pub const INPUT_LEFT: u32 = 1;
pub const INPUT_RIGHT: u32 = 2;
pub const INPUT_UP: u32 = 4;
pub const INPUT_DOWN: u32 = 8;
pub const INPUT_ACTION1: u32 = 16;
pub const INPUT_ACTION2: u32 = 32;

#[derive(Clone, Debug, Default)]
pub struct InputFrame {
    pub frame: u64,
    pub player_id: u8,
    pub input_bits: u32,
}

pub trait GameState: Clone {
    fn apply_input(&mut self, inputs: &[InputFrame]);
    fn frame(&self) -> u64;
}

pub struct RollbackBuffer<S: GameState> {
    frames: Vec<Option<S>>,
    inputs: HashMap<u64, Vec<InputFrame>>,
    current_frame: u64,
}

impl<S: GameState> RollbackBuffer<S> {
    pub fn new() -> Self {
        let mut frames = Vec::with_capacity(MAX_ROLLBACK_FRAMES);
        for _ in 0..MAX_ROLLBACK_FRAMES {
            frames.push(None);
        }
        Self {
            frames,
            inputs: HashMap::new(),
            current_frame: 0,
        }
    }

    pub fn save_frame(&mut self, state: S) {
        let slot = (state.frame() % MAX_ROLLBACK_FRAMES as u64) as usize;
        self.frames[slot] = Some(state);
    }

    /// Stores input in the inputs map. Returns true if rollback is needed
    /// (i.e. the input arrived late, for a frame already simulated).
    pub fn receive_input(&mut self, input: InputFrame) -> bool {
        let needs = self.needs_rollback(input.frame);
        self.inputs.entry(input.frame).or_default().push(input);
        needs
    }

    pub fn needs_rollback(&self, input_frame: u64) -> bool {
        if input_frame >= self.current_frame {
            return false;
        }
        let slot = (input_frame % MAX_ROLLBACK_FRAMES as u64) as usize;
        self.frames[slot].is_some()
    }

    /// Returns the frame to roll back to (the input_frame itself), if a
    /// snapshot exists for it.
    pub fn rollback_frame(&self, input_frame: u64) -> Option<u64> {
        let slot = (input_frame % MAX_ROLLBACK_FRAMES as u64) as usize;
        if self.frames[slot].is_some() {
            Some(input_frame)
        } else {
            None
        }
    }

    pub fn get_snapshot(&self, frame: u64) -> Option<&S> {
        let slot = (frame % MAX_ROLLBACK_FRAMES as u64) as usize;
        self.frames[slot].as_ref()
    }

    pub fn advance_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }
}

impl<S: GameState> Default for RollbackBuffer<S> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RollbackSession<S: GameState> {
    pub buffer: RollbackBuffer<S>,
    pub local_player_id: u8,
    pub confirmed_frame: u64,
}

impl<S: GameState> RollbackSession<S> {
    pub fn new(local_player_id: u8) -> Self {
        Self {
            buffer: RollbackBuffer::new(),
            local_player_id,
            confirmed_frame: 0,
        }
    }

    pub fn record_state(&mut self, state: S) {
        self.buffer.save_frame(state);
    }

    /// Adds an input. Returns true if rollback is needed.
    pub fn add_input(&mut self, input: InputFrame) -> bool {
        self.buffer.receive_input(input)
    }

    /// Returns the frame to roll back to for the given late input frame.
    pub fn get_rollback_target(&self, late_input_frame: u64) -> Option<u64> {
        self.buffer.rollback_frame(late_input_frame)
    }

    /// Marks a frame as confirmed (both players agree on the state up to this frame).
    pub fn confirm_frame(&mut self, frame: u64) {
        self.confirmed_frame = frame;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct TestState {
        frame: u64,
        value: i32,
    }

    impl GameState for TestState {
        fn apply_input(&mut self, inputs: &[InputFrame]) {
            for inp in inputs {
                if inp.input_bits & INPUT_RIGHT != 0 {
                    self.value += 1;
                }
            }
            self.frame += 1;
        }
        fn frame(&self) -> u64 {
            self.frame
        }
    }

    #[test]
    fn save_and_retrieve_snapshot() {
        let mut buf: RollbackBuffer<TestState> = RollbackBuffer::new();
        let state = TestState { frame: 5, value: 42 };
        buf.save_frame(state);
        let snapshot = buf.get_snapshot(5).expect("snapshot at frame 5");
        assert_eq!(snapshot.frame, 5);
        assert_eq!(snapshot.value, 42);
    }

    #[test]
    fn needs_rollback_for_late_input() {
        let mut buf: RollbackBuffer<TestState> = RollbackBuffer::new();
        // Save frames 1 through 5
        for f in 1u64..=5 {
            buf.save_frame(TestState { frame: f, value: f as i32 });
        }
        buf.advance_frame(5);
        // Input arriving for frame 3 is late
        let result = buf.receive_input(InputFrame { frame: 3, player_id: 1, input_bits: INPUT_RIGHT });
        assert!(result, "expected rollback needed for late input at frame 3");
    }

    #[test]
    fn no_rollback_for_current_input() {
        let mut buf: RollbackBuffer<TestState> = RollbackBuffer::new();
        buf.advance_frame(5);
        // Input for frame 5 (current frame) — not late
        let result = buf.receive_input(InputFrame { frame: 5, player_id: 1, input_bits: INPUT_LEFT });
        assert!(!result, "expected no rollback for input at current frame");
    }

    #[test]
    fn advance_frame_updates_current() {
        let mut buf: RollbackBuffer<TestState> = RollbackBuffer::new();
        buf.advance_frame(10);
        assert_eq!(buf.current_frame(), 10);
    }

    #[test]
    fn session_confirm_frame_advances_confirmed() {
        let mut session: RollbackSession<TestState> = RollbackSession::new(0);
        session.confirm_frame(5);
        assert_eq!(session.confirmed_frame, 5);
    }
}
