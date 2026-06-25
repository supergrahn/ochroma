use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;

/// A serializable snapshot of an entity's components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub id: u32,
    pub components: BTreeMap<String, serde_json::Value>,
}

/// A complete world snapshot that can be saved/loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub version: u32,
    pub timestamp: f64,
    pub entities: Vec<EntitySnapshot>,
    pub simulation: SimulationSnapshot,
    pub metadata: BTreeMap<String, String>,
}

/// Engine-level snapshot of simulation time state — game-agnostic.
///
/// Game-specific counters (population, funds, tax rates, zone counts, …) belong
/// in `WorldSnapshot::metadata` (keyed strings) or in the game's own serialised
/// payload layered on top of this snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    pub game_time_hours: f64,
    pub season_day: u32,
}

impl WorldSnapshot {
    pub fn new(timestamp: f64) -> Self {
        Self {
            version: 1,
            timestamp,
            entities: Vec::new(),
            simulation: SimulationSnapshot {
                game_time_hours: 0.0,
                season_day: 0,
            },
            metadata: BTreeMap::new(),
        }
    }

    pub fn add_entity(&mut self, id: u32) -> &mut EntitySnapshot {
        self.entities.push(EntitySnapshot {
            id,
            components: BTreeMap::new(),
        });
        self.entities.last_mut().unwrap()
    }

    /// Serialize to compressed binary (zstd JSON).
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let json = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        zstd::bulk::compress(&json, 3).map_err(|e| e.to_string())
    }

    /// Deserialize from compressed binary.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let json = zstd::bulk::decompress(data, 50 * 1024 * 1024).map_err(|e| e.to_string())?;
        serde_json::from_slice(&json).map_err(|e| e.to_string())
    }

    /// Save to file.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), String> {
        let data = self.to_bytes()?;
        std::fs::write(path, data).map_err(|e| e.to_string())
    }

    /// Load from file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        Self::from_bytes(&data)
    }
}
