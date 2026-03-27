use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDelta {
    pub entity_id: u32,
    pub component: String,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

/// A network message between client and server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetMessage {
    /// Client → Server: player input
    PlayerInput { player_id: u32, action: PlayerAction },
    /// Server → Client: state delta
    StateDelta { tick: u64, deltas: Vec<EntityDelta> },
    /// Server → Client: full state snapshot
    FullSnapshot { tick: u64, data: Vec<u8> },
    /// Ping/pong for latency measurement
    Ping { timestamp: u64 },
    Pong { timestamp: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlayerAction {
    PlaceRoad { start: [f32; 3], end: [f32; 3] },
    Zone { position: [f32; 2], zone_type: String },
    PlaceBuilding { position: [f32; 3], asset_id: String },
    AdjustBudget { tax_type: String, rate: f32 },
}

impl NetMessage {
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }
}

pub struct ReplicationServer {
    pub tick: u64,
    pending_deltas: Vec<EntityDelta>,
}

impl ReplicationServer {
    pub fn new() -> Self {
        Self {
            tick: 0,
            pending_deltas: Vec::new(),
        }
    }

    pub fn tick_counter(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    pub fn apply_input(&mut self, _input: &[u8]) -> Vec<EntityDelta> {
        self.pending_deltas.drain(..).collect()
    }

    pub fn process_message(&mut self, msg: &NetMessage) -> Vec<NetMessage> {
        match msg {
            NetMessage::PlayerInput { player_id, action } => {
                self.tick += 1;
                let delta = EntityDelta {
                    entity_id: *player_id,
                    component: format!("{:?}", action),
                    data: serde_json::to_vec(action).unwrap_or_default(),
                    timestamp: self.tick,
                };
                vec![NetMessage::StateDelta { tick: self.tick, deltas: vec![delta] }]
            }
            NetMessage::Ping { timestamp } => vec![NetMessage::Pong { timestamp: *timestamp }],
            _ => vec![],
        }
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

    pub fn applied_count(&self) -> u64 {
        self.applied_count
    }
}

impl Default for ReplicationClient {
    fn default() -> Self {
        Self::new()
    }
}
