use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDelta {
    pub entity_id: u32,
    pub component: String,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

pub struct ReplicationServer {
    tick_counter: u64,
    pending_deltas: Vec<EntityDelta>,
}

impl ReplicationServer {
    pub fn new() -> Self {
        Self {
            tick_counter: 0,
            pending_deltas: Vec::new(),
        }
    }

    pub fn tick(&mut self) -> u64 {
        self.tick_counter += 1;
        self.tick_counter
    }

    pub fn apply_input(&mut self, _input: &[u8]) -> Vec<EntityDelta> {
        self.pending_deltas.drain(..).collect()
    }
}

impl Default for ReplicationServer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ReplicationClient {
    applied_count: u64,
}

impl ReplicationClient {
    pub fn new() -> Self {
        Self { applied_count: 0 }
    }

    pub fn apply_deltas(&mut self, deltas: &[EntityDelta]) {
        self.applied_count += deltas.len() as u64;
    }
}

impl Default for ReplicationClient {
    fn default() -> Self {
        Self::new()
    }
}
