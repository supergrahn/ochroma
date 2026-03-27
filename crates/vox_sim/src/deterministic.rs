use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A deterministic PRNG wrapper that tracks how many values have been drawn.
#[derive(Debug)]
pub struct DeterministicRng {
    rng: StdRng,
    seed: u64,
    draws: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
            draws: 0,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// Generate a random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.draws += 1;
        self.rng.random::<u64>()
    }

    /// Generate a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.draws += 1;
        self.rng.random::<f64>()
    }

    /// Generate a random u32 in the given range.
    pub fn next_range(&mut self, min: u32, max: u32) -> u32 {
        self.draws += 1;
        self.rng.random_range(min..max)
    }

    /// Reset to the original seed, clearing the draw counter.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
        self.draws = 0;
    }
}

/// A serialisable player action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlayerAction {
    pub action_type: String,
    pub payload: Vec<u8>,
}

/// A record of a single tick's inputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputRecord {
    pub tick: u64,
    pub actions: Vec<PlayerAction>,
    pub rng_seed: u64,
}

/// Records and replays simulation inputs for deterministic replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationRecorder {
    records: HashMap<u64, InputRecord>,
}

impl SimulationRecorder {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record inputs for a given tick.
    pub fn record_tick(&mut self, tick_number: u64, actions: Vec<PlayerAction>, seed: u64) {
        self.records.insert(
            tick_number,
            InputRecord {
                tick: tick_number,
                actions,
                rng_seed: seed,
            },
        );
    }

    /// Replay a tick, returning its recorded actions and seed.
    pub fn replay_tick(&self, tick_number: u64) -> Option<(Vec<PlayerAction>, u64)> {
        self.records
            .get(&tick_number)
            .map(|r| (r.actions.clone(), r.rng_seed))
    }

    /// Number of recorded ticks.
    pub fn tick_count(&self) -> usize {
        self.records.len()
    }

    /// Save the recording to a file (JSON).
    pub fn save_recording(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.records)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a recording from a file (JSON).
    pub fn load_recording(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let records: HashMap<u64, InputRecord> = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Self { records })
    }
}

impl Default for SimulationRecorder {
    fn default() -> Self {
        Self::new()
    }
}
