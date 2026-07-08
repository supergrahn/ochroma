use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDelta {
    pub entity_id: u32,
    pub component: String,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

/// Opaque serialized game command.  The engine does not interpret the bytes —
/// the game layer serializes its own command type (e.g. via `serde_json` or
/// `bincode`) and deserializes on receipt.
///
/// ```rust,ignore
/// // game side (not in the engine crate):
/// let cmd = MyGameCommand::DoSomething { value: 42 };
/// let payload = CommandPayload::encode(&serde_json::to_vec(&cmd).unwrap());
/// let msg = NetMessage::PlayerInput { player_id: 1, command: payload };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandPayload(pub Vec<u8>);

impl CommandPayload {
    /// Wrap raw bytes produced by the game's serializer.
    pub fn encode(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    /// Return the raw bytes for the game's deserializer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// A network message between client and server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetMessage {
    /// Client → Server: player input (opaque command bytes; the engine does
    /// not interpret the payload).
    PlayerInput {
        player_id: u32,
        command: CommandPayload,
    },
    /// Server → Client: state delta
    StateDelta {
        tick: u64,
        deltas: Vec<EntityDelta>,
    },
    /// Server → Client: full state snapshot
    FullSnapshot {
        tick: u64,
        data: Vec<u8>,
    },
    /// Ping/pong for latency measurement
    Ping {
        timestamp: u64,
    },
    Pong {
        timestamp: u64,
    },
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
            NetMessage::PlayerInput { player_id, command } => {
                self.tick += 1;
                let delta = EntityDelta {
                    entity_id: *player_id,
                    component: "command".to_string(),
                    data: command.as_bytes().to_vec(),
                    timestamp: self.tick,
                };
                vec![NetMessage::StateDelta {
                    tick: self.tick,
                    deltas: vec![delta],
                }]
            }
            NetMessage::Ping { timestamp } => vec![NetMessage::Pong {
                timestamp: *timestamp,
            }],
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
