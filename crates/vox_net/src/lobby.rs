use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayerRole {
    Mayor,      // full control
    Councillor, // limited control (can't adjust budget)
    Spectator,  // view only
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerInfo {
    pub id: u32,
    pub name: String,
    pub role: PlayerRole,
    pub connected: bool,
    pub latency_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobbyState {
    pub game_id: String,
    pub host_name: String,
    pub city_name: String,
    pub max_players: u32,
    pub players: Vec<PlayerInfo>,
    pub in_progress: bool,
}

impl LobbyState {
    pub fn new(host_name: &str, city_name: &str, max_players: u32) -> Self {
        Self {
            game_id: uuid::Uuid::new_v4().to_string(),
            host_name: host_name.to_string(),
            city_name: city_name.to_string(),
            max_players,
            players: vec![PlayerInfo {
                id: 0,
                name: host_name.to_string(),
                role: PlayerRole::Mayor,
                connected: true,
                latency_ms: 0,
            }],
            in_progress: false,
        }
    }

    pub fn add_player(&mut self, name: &str, role: PlayerRole) -> Option<u32> {
        if self.players.len() as u32 >= self.max_players {
            return None;
        }
        let id = self.players.len() as u32;
        self.players.push(PlayerInfo {
            id,
            name: name.to_string(),
            role,
            connected: true,
            latency_ms: 0,
        });
        Some(id)
    }

    pub fn remove_player(&mut self, id: u32) {
        self.players.retain(|p| p.id != id);
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn is_full(&self) -> bool {
        self.players.len() as u32 >= self.max_players
    }

    pub fn can_player_act(&self, player_id: u32, action: &str) -> bool {
        if let Some(player) = self.players.iter().find(|p| p.id == player_id) {
            match player.role {
                PlayerRole::Mayor => true,
                PlayerRole::Councillor => action != "budget" && action != "policy",
                PlayerRole::Spectator => false,
            }
        } else {
            false
        }
    }
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub sender_id: u32,
    pub sender_name: String,
    pub text: String,
    pub timestamp: f64,
}

/// Chat history.
pub struct ChatHistory {
    pub messages: Vec<ChatMessage>,
    pub max_messages: usize,
}

impl ChatHistory {
    pub fn new(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
        }
    }

    pub fn add(&mut self, sender_id: u32, sender_name: &str, text: &str, timestamp: f64) {
        self.messages.push(ChatMessage {
            sender_id,
            sender_name: sender_name.to_string(),
            text: text.to_string(),
            timestamp,
        });
        if self.messages.len() > self.max_messages {
            self.messages.remove(0);
        }
    }

    pub fn recent(&self, count: usize) -> &[ChatMessage] {
        let start = self.messages.len().saturating_sub(count);
        &self.messages[start..]
    }
}
