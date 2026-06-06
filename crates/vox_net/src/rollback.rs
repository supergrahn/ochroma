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

// ---------------------------------------------------------------------------
// Deterministic two-player kinematic simulation + predict/rollback/resimulate.
//
// The pieces above (`RollbackBuffer`, `RollbackSession`) are generic snapshot
// machinery: a ring buffer of `GameState` snapshots plus an input map, with
// helpers to detect a late input and find the snapshot to roll back to. What
// they did NOT have is an actual simulation that exercises predict ->
// roll-back -> re-simulate, nor a concrete deterministic game state to prove
// it converges. Everything below adds that working loop.
// ---------------------------------------------------------------------------

/// Number of players in the rollback world. Player 0 is local, player 1 is the
/// predicted remote peer (matches the 2-player net_walk demo).
pub const NUM_PLAYERS: usize = 2;

/// Discrete movement velocities (m/s) selected by the directional input bits.
/// Fixed integer-ish constants keep the float math reproducible across runs.
const MOVE_SPEED: f32 = 3.0;

/// Fixed simulation timestep in seconds. Constant => deterministic integration.
pub const SIM_DT: f32 = 0.016;

/// A single player's kinematic state: world position and current velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerKinematics {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
}

impl Default for PlayerKinematics {
    fn default() -> Self {
        Self { position: [0.0; 3], velocity: [0.0; 3] }
    }
}

/// Translate input bits into a velocity vector (XZ plane, deterministic).
fn velocity_from_input(input_bits: u32) -> [f32; 3] {
    let mut v = [0.0f32; 3];
    if input_bits & INPUT_LEFT != 0 {
        v[0] -= MOVE_SPEED;
    }
    if input_bits & INPUT_RIGHT != 0 {
        v[0] += MOVE_SPEED;
    }
    if input_bits & INPUT_UP != 0 {
        v[2] += MOVE_SPEED;
    }
    if input_bits & INPUT_DOWN != 0 {
        v[2] -= MOVE_SPEED;
    }
    v
}

/// Deterministic kinematic state for the whole world (all players).
///
/// One `apply_input` call advances exactly one tick: each player's velocity is
/// set from its input (if any input was supplied for this player this tick),
/// otherwise the previous velocity is *retained* — that retention is precisely
/// the prediction model ("remote keeps doing what it was doing"). Then every
/// player integrates `position += velocity * SIM_DT`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSim {
    pub players: [PlayerKinematics; NUM_PLAYERS],
    tick: u64,
}

impl Default for WorldSim {
    fn default() -> Self {
        Self { players: [PlayerKinematics::default(); NUM_PLAYERS], tick: 0 }
    }
}

impl WorldSim {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash the *bit patterns* of every f32 in the world plus the tick. Used by
    /// the determinism guard: identical inputs must yield a bit-identical hash.
    pub fn state_hash(&self) -> u64 {
        // FNV-1a over the raw f32 bits (NOT the float values) and the tick.
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |bits: u32| {
            for byte in bits.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for p in &self.players {
            for c in p.position {
                mix(c.to_bits());
            }
            for c in p.velocity {
                mix(c.to_bits());
            }
        }
        for byte in self.tick.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn position_of(&self, player: usize) -> [f32; 3] {
        self.players[player].position
    }
}

impl GameState for WorldSim {
    fn apply_input(&mut self, inputs: &[InputFrame]) {
        // Set velocity for any player that supplied an input this tick; players
        // without an input keep their previous velocity (the prediction model).
        for inp in inputs {
            let p = inp.player_id as usize;
            if p < NUM_PLAYERS {
                self.players[p].velocity = velocity_from_input(inp.input_bits);
            }
        }
        // Integrate every player one fixed step.
        for p in &mut self.players {
            for axis in 0..3 {
                p.position[axis] += p.velocity[axis] * SIM_DT;
            }
        }
        self.tick += 1;
    }

    fn frame(&self) -> u64 {
        self.tick
    }
}

/// A self-contained predict / roll-back / re-simulate driver around a
/// [`WorldSim`]. It owns the authoritative-so-far input timeline per tick and a
/// rollback buffer of `WorldSim` snapshots, and exposes:
///
/// * [`Predictor::tick`] — advance one tick using known/predicted inputs.
/// * [`Predictor::receive_remote_input`] — feed a (possibly late) remote input;
///   if it corrects a tick already simulated, the next [`Predictor::tick`] (or
///   an explicit [`Predictor::resimulate_if_needed`]) rolls back to that tick,
///   replaces the predicted input, and re-runs forward to the present.
///
/// `resim_count` records how many full re-simulations actually executed, so a
/// test can assert that a rollback truly happened (not just that numbers lined
/// up by luck).
pub struct Predictor {
    sim: WorldSim,
    /// Snapshots indexed by tick (ring buffer lives in `buffer`).
    buffer: RollbackBuffer<WorldSim>,
    /// Per-tick list of inputs that should actually be *applied* that tick.
    ///
    /// Only inputs that should change a player's velocity live here: the local
    /// player's input every tick, and a remote player's input ONLY at the ticks
    /// it is known for (authoritative). A tick with no remote input entry is a
    /// pure prediction for the remote — `apply_input` is fed only the known
    /// inputs, so the remote's velocity is *retained* (the prediction model).
    timeline: HashMap<u64, Vec<InputFrame>>,
    /// Earliest tick whose inputs changed and needs re-simulation, if any.
    dirty_from: Option<u64>,
    /// How many times a full rollback re-simulation has run.
    pub resim_count: u32,
}

impl Predictor {
    pub fn new(initial: WorldSim) -> Self {
        let mut buffer = RollbackBuffer::new();
        // Snapshot the initial state at tick 0 so a tick-0 correction can roll
        // back to it.
        buffer.save_frame(initial);
        buffer.advance_frame(initial.frame());
        Self {
            sim: initial,
            buffer,
            timeline: HashMap::new(),
            dirty_from: None,
            resim_count: 0,
        }
    }

    pub fn current_tick(&self) -> u64 {
        self.sim.frame()
    }

    pub fn world(&self) -> &WorldSim {
        &self.sim
    }

    pub fn position_of(&self, player: usize) -> [f32; 3] {
        self.sim.position_of(player)
    }

    /// Replace the local player's input for `tick` in the timeline, preserving
    /// any already-known remote inputs for that tick.
    fn set_local_input(&mut self, tick: u64, local_player: u8, local_input: u32) {
        let list = self.timeline.entry(tick).or_default();
        list.retain(|f| f.player_id != local_player);
        list.push(InputFrame { frame: tick, player_id: local_player, input_bits: local_input });
    }

    /// Advance exactly one tick. `local_input` is this client's input bits for
    /// the new tick. Before stepping, any pending rollback (from a late remote
    /// input) is resolved so the present state is always reconciled.
    pub fn tick(&mut self, local_player: u8, local_input: u32) {
        self.resimulate_if_needed();

        let next_tick = self.sim.frame() + 1;
        self.set_local_input(next_tick, local_player, local_input);
        let set = self.timeline.get(&next_tick).cloned().unwrap_or_default();
        self.sim.apply_input(&set);
        self.buffer.save_frame(self.sim);
        self.buffer.advance_frame(self.sim.frame());
    }

    /// Feed a remote player's authoritative input for `tick`. If that tick was
    /// already simulated *without* this input (it was predicted), mark the
    /// timeline dirty from that tick so the next `tick`/`resimulate_if_needed`
    /// rolls back. Returns true if this input caused a correction (a divergence
    /// that must be reconciled).
    pub fn receive_remote_input(&mut self, input: InputFrame) -> bool {
        let tick = input.frame;
        let pid = input.player_id;
        if pid as usize >= NUM_PLAYERS {
            return false;
        }

        let list = self.timeline.entry(tick).or_default();
        let previously = list.iter().find(|f| f.player_id == pid).map(|f| f.input_bits);
        let changed = previously != Some(input.input_bits);
        list.retain(|f| f.player_id != pid);
        list.push(input);

        // A correction only forces a rollback if that tick has already been
        // simulated (tick <= current) and the input differs from what we used.
        let needs_rollback = changed && tick <= self.sim.frame();
        if needs_rollback {
            self.dirty_from = Some(match self.dirty_from {
                Some(d) => d.min(tick),
                None => tick,
            });
        }
        needs_rollback
    }

    /// If a late correction marked the timeline dirty, roll back to the snapshot
    /// *before* the corrected tick and re-simulate forward to the present using
    /// the corrected timeline. Increments `resim_count` when it runs.
    pub fn resimulate_if_needed(&mut self) {
        let Some(dirty_tick) = self.dirty_from.take() else {
            return;
        };
        let present = self.sim.frame();
        // Roll back to the snapshot at the tick *before* the correction.
        let restore_tick = dirty_tick.saturating_sub(1);
        let snapshot = match self.buffer.get_snapshot(restore_tick) {
            Some(s) if s.frame() == restore_tick => *s,
            // Snapshot evicted from the ring buffer; can't roll back this far.
            _ => return,
        };
        self.sim = snapshot;

        // Re-simulate each tick from restore_tick+1..=present feeding only the
        // recorded inputs for that tick. Ticks with no recorded remote input
        // retain the remote's velocity (the same prediction), now seeded from
        // the corrected state — so once an authoritative turn is applied, every
        // later predicted tick rides the corrected velocity forward.
        for t in (restore_tick + 1)..=present {
            let set = self.timeline.get(&t).cloned().unwrap_or_default();
            self.sim.apply_input(&set);
            self.buffer.save_frame(self.sim);
        }
        self.buffer.advance_frame(self.sim.frame());
        self.resim_count += 1;
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

    // -----------------------------------------------------------------------
    // Deterministic predict / roll-back / re-simulate proof.
    // -----------------------------------------------------------------------

    /// Ground-truth remote (player B = id 1) simulation: applies B's scripted
    /// inputs the instant they happen, no delay. This is "what B actually did".
    fn ground_truth_b(script: &[(u64, u32)], total_ticks: u64) -> WorldSim {
        let mut sim = WorldSim::new();
        for t in 1..=total_ticks {
            let mut inputs = Vec::new();
            for &(when, bits) in script {
                if when == t {
                    inputs.push(InputFrame { frame: t, player_id: 1, input_bits: bits });
                }
            }
            sim.apply_input(&inputs);
        }
        sim
    }

    /// B's TRUE position at every tick (index t = position after tick t).
    fn ground_truth_b_track(script: &[(u64, u32)], total_ticks: u64) -> Vec<[f32; 3]> {
        let mut sim = WorldSim::new();
        let mut track = vec![sim.position_of(1)];
        for t in 1..=total_ticks {
            let inputs: Vec<InputFrame> = script
                .iter()
                .filter(|&&(when, _)| when == t)
                .map(|&(_, bits)| InputFrame { frame: t, player_id: 1, input_bits: bits })
                .collect();
            sim.apply_input(&inputs);
            track.push(sim.position_of(1));
        }
        track
    }

    /// THE acceptance test: A predicts B with a 3-tick input delay; B turns at
    /// tick 10. A's prediction DIVERGES at ticks 11-12, then rollback RECONCILES
    /// it bit-exactly once the late turn input arrives at tick 13.
    #[test]
    fn rollback_converges() {
        const DELAY: u64 = 3;
        const TURN_TICK: u64 = 10;
        const TOTAL: u64 = 14;

        // B's scripted inputs: starts moving +Z at tick 1, turns to +X at tick 10.
        let script: [(u64, u32); 2] = [(1, INPUT_UP), (TURN_TICK, INPUT_RIGHT)];
        let b_true = ground_truth_b_track(&script, TOTAL);

        // A is player 0 (stationary here for clarity) and predicts player 1 (B).
        let mut a = Predictor::new(WorldSim::new());

        // Record A's predicted view of B at each tick and the divergence.
        let mut predicted_b: Vec<[f32; 3]> = vec![a.position_of(1)];
        let mut div_at_11 = 0.0f32;
        let mut div_at_12 = 0.0f32;

        for t in 1..=TOTAL {
            // A advances its own (stationary) player; B is predicted (no input
            // supplied for player 1 => velocity retained).
            a.tick(0, 0);

            // Deliver any of B's inputs whose 3-tick delay has now elapsed.
            for &(when, bits) in &script {
                if when + DELAY == t {
                    a.receive_remote_input(InputFrame { frame: when, player_id: 1, input_bits: bits });
                    // Reconcile immediately so the recorded view reflects the
                    // post-rollback truth at this tick.
                    a.resimulate_if_needed();
                }
            }

            let pb = a.position_of(1);
            predicted_b.push(pb);

            // Measure divergence from B's true position at the SAME tick, in the
            // window AFTER the turn but BEFORE the corrected input arrives.
            let d = dist(pb, b_true[t as usize]);
            if t == 11 {
                div_at_11 = d;
            }
            if t == 12 {
                div_at_12 = d;
            }
        }

        // (a) DIVERGENCE: at ticks 11 and 12, A's predicted B (still going +Z)
        // differs measurably from B's true position (now going +X).
        println!("[rollback] divergence at tick 11 = {div_at_11:.6} m");
        println!("[rollback] divergence at tick 12 = {div_at_12:.6} m");
        assert!(
            div_at_11 > 1e-3,
            "expected A's prediction of B to DIVERGE at tick 11, got {div_at_11}"
        );
        assert!(
            div_at_12 > div_at_11,
            "divergence must grow tick over tick: 12 ({div_at_12}) > 11 ({div_at_11})"
        );

        // (b) RECONCILIATION: after the late turn input (tick 10) arrives at
        // tick 13 and rollback re-simulates, A's view of B matches B's true
        // position bit-exactly at the present tick.
        let final_div = dist(a.position_of(1), b_true[TOTAL as usize]);
        println!("[rollback] reconciled divergence at tick {TOTAL} = {final_div:.9} m");
        assert!(
            final_div < 1e-5,
            "after rollback, A's view of B must MATCH truth (<1e-5), got {final_div}"
        );
        // Bit-exact check on each axis (determinism of identical input streams).
        let av = a.position_of(1);
        let tv = b_true[TOTAL as usize];
        for axis in 0..3 {
            assert_eq!(
                av[axis].to_bits(),
                tv[axis].to_bits(),
                "axis {axis}: reconciled {} must be bit-identical to truth {}",
                av[axis], tv[axis]
            );
        }

        // (c) The rollback ACTUALLY executed (not a coincidence): at least one
        // re-simulation ran. Both the tick-1 and tick-10 inputs arrive late.
        println!("[rollback] resim_count = {}", a.resim_count);
        assert!(a.resim_count > 0, "rollback must have re-simulated at least once");
    }

    /// Determinism guard: identical input streams produce a bit-identical state
    /// hash after 100 ticks (hashes the f32 bits, not the values).
    #[test]
    fn determinism_identical_inputs_bit_identical_hash() {
        let run = || {
            let mut sim = WorldSim::new();
            for t in 1..=100u64 {
                // A pseudo-deterministic but non-trivial input pattern per tick.
                let bits = match t % 4 {
                    0 => INPUT_UP,
                    1 => INPUT_RIGHT,
                    2 => INPUT_DOWN,
                    _ => INPUT_LEFT,
                };
                sim.apply_input(&[
                    InputFrame { frame: t, player_id: 0, input_bits: bits },
                    InputFrame { frame: t, player_id: 1, input_bits: bits ^ INPUT_UP },
                ]);
            }
            sim.state_hash()
        };
        let h1 = run();
        let h2 = run();
        println!("[determinism] hash run 1 = {h1:#018x}");
        println!("[determinism] hash run 2 = {h2:#018x}");
        assert_eq!(h1, h2, "identical inputs must yield a bit-identical state hash");

        // A different input stream must yield a different hash (the guard has teeth).
        let mut other = WorldSim::new();
        for t in 1..=100u64 {
            other.apply_input(&[InputFrame { frame: t, player_id: 0, input_bits: INPUT_UP }]);
        }
        assert_ne!(other.state_hash(), h1, "different inputs must differ in hash");
    }

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Cross-check: the ground-truth helper and a Predictor that receives every
    /// input ON TIME agree exactly (sanity for the rollback-free path).
    #[test]
    fn predictor_with_no_delay_matches_ground_truth() {
        let script: [(u64, u32); 2] = [(1, INPUT_UP), (10, INPUT_RIGHT)];
        let truth = ground_truth_b(&script, 14);

        let mut a = Predictor::new(WorldSim::new());
        for t in 1..=14u64 {
            // Deliver B's input for this tick BEFORE advancing it (truly on-time,
            // so the tick already incorporates B's authoritative input and no
            // rollback is ever needed).
            for &(when, bits) in &script {
                if when == t {
                    a.receive_remote_input(InputFrame { frame: when, player_id: 1, input_bits: bits });
                }
            }
            a.tick(0, 0);
        }
        assert_eq!(a.resim_count, 0, "on-time inputs need no rollback");
        let av = a.position_of(1);
        let tv = truth.position_of(1);
        for axis in 0..3 {
            assert_eq!(av[axis].to_bits(), tv[axis].to_bits(), "axis {axis} must match truth");
        }
    }
}
