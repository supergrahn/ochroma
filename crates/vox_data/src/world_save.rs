use serde::{Serialize, Deserialize};
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSave {
    pub version: u32,
    pub engine_version: String,
    pub timestamp: String,
    pub scene_name: String,
    pub entities: Vec<SavedEntity>,
    pub resources: SavedResources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedEntity {
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],     // quaternion xyzw
    pub scale: [f32; 3],
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub tags: Vec<String>,
    pub custom_data: HashMap<String, serde_json::Value>,
    pub collider: Option<SavedCollider>,
    pub audio: Option<SavedAudio>,
    pub light: Option<SavedLight>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedCollider {
    pub shape_type: String,  // "box", "sphere", "capsule"
    pub dimensions: Vec<f32>, // half_extents for box, [radius] for sphere, [radius, height] for capsule
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedAudio {
    pub clip_path: String,
    pub volume: f32,
    pub looping: bool,
    pub spatial: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedLight {
    pub light_type: String,  // "point", "directional"
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedResources {
    pub time_of_day: f32,
    pub camera_position: [f32; 3],
    pub camera_rotation: [f32; 4],
    pub game_state: String,
}

impl WorldSave {
    pub fn new(scene_name: &str) -> Self {
        Self {
            version: 1,
            engine_version: "0.1.0".to_string(),
            timestamp: chrono_lite_timestamp(),
            scene_name: scene_name.to_string(),
            entities: Vec::new(),
            resources: SavedResources {
                time_of_day: 12.0,
                camera_position: [0.0, 10.0, 30.0],
                camera_rotation: [0.0, 0.0, 0.0, 1.0],
                game_state: "playing".to_string(),
            },
        }
    }

    pub fn add_entity(&mut self, entity: SavedEntity) {
        self.entities.push(entity);
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, &json).map_err(|e| e.to_string())?;
        println!("[save] Saved {} entities to {}", self.entities.len(), path.display());
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let save: Self = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        println!("[save] Loaded {} entities from {}", save.entities.len(), path.display());
        Ok(save)
    }

    pub fn entity_count(&self) -> usize { self.entities.len() }

    /// Quick save to default location
    pub fn quick_save_path() -> std::path::PathBuf {
        let dir = dirs_next::data_dir().unwrap_or_else(|| std::path::PathBuf::from(".")).join("ochroma/saves");
        std::fs::create_dir_all(&dir).ok();
        dir.join("quicksave.ochroma_save")
    }

    /// Auto save path with timestamp
    pub fn auto_save_path() -> std::path::PathBuf {
        let dir = dirs_next::data_dir().unwrap_or_else(|| std::path::PathBuf::from(".")).join("ochroma/saves");
        std::fs::create_dir_all(&dir).ok();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        dir.join(format!("autosave_{}.ochroma_save", timestamp))
    }
}

fn chrono_lite_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}
