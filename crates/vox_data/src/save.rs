use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;

const SAVE_MAGIC: &[u8; 4] = b"OCHS";
const SAVE_VERSION: u16 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveHeader {
    pub version: u16,
    pub city_name: String,
    pub game_time_hours: f64,
    pub citizen_count: u32,
    pub funds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GameState {
    pub header: SaveHeader,
    pub data: Vec<u8>, // Serialized game state (opaque to the save system)
}

#[derive(Debug, thiserror::Error)]
pub enum SaveError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid save file magic")]
    InvalidMagic,
    #[error("incompatible save version: {0}")]
    IncompatibleVersion(u16),
    #[error("serialization error: {0}")]
    Serialize(String),
    #[error("deserialization error: {0}")]
    Deserialize(String),
}

pub fn save_game(state: &GameState, path: &Path) -> Result<(), SaveError> {
    let json =
        serde_json::to_vec(state).map_err(|e| SaveError::Serialize(e.to_string()))?;
    let compressed = zstd::bulk::compress(&json, 3)
        .map_err(|e| SaveError::Serialize(e.to_string()))?;

    let mut file = std::fs::File::create(path)?;
    file.write_all(SAVE_MAGIC)?;
    file.write_all(&SAVE_VERSION.to_le_bytes())?;
    file.write_all(&(compressed.len() as u64).to_le_bytes())?;
    file.write_all(&compressed)?;
    Ok(())
}

pub fn load_game(path: &Path) -> Result<GameState, SaveError> {
    let mut file = std::fs::File::open(path)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != SAVE_MAGIC {
        return Err(SaveError::InvalidMagic);
    }

    let mut ver = [0u8; 2];
    file.read_exact(&mut ver)?;
    let version = u16::from_le_bytes(ver);
    if version != SAVE_VERSION {
        return Err(SaveError::IncompatibleVersion(version));
    }

    let mut size_bytes = [0u8; 8];
    file.read_exact(&mut size_bytes)?;
    let compressed_size = u64::from_le_bytes(size_bytes) as usize;

    let mut compressed = vec![0u8; compressed_size];
    file.read_exact(&mut compressed)?;

    let json = zstd::bulk::decompress(&compressed, 10 * 1024 * 1024)
        .map_err(|e| SaveError::Deserialize(e.to_string()))?;
    serde_json::from_slice(&json).map_err(|e| SaveError::Deserialize(e.to_string()))
}
