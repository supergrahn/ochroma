use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A scene file that describes a game level/world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    pub version: u32,
    pub name: String,
    pub description: String,
    pub entities: Vec<SceneEntity>,
    pub settings: SceneSettings,
}

/// An entity in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: u32,
    pub name: String,
    pub parent: Option<u32>,
    pub transform: Transform,
    pub components: HashMap<String, serde_json::Value>,
}

/// Transform (position, rotation, scale).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // quaternion xyzw
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

/// Scene-level settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSettings {
    pub ambient_light: [f32; 3],
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_color: [f32; 3],
    pub gravity: [f32; 3],
    pub skybox: Option<String>,
}

impl Default for SceneSettings {
    fn default() -> Self {
        Self {
            ambient_light: [0.1, 0.1, 0.12],
            fog_enabled: false,
            fog_density: 0.001,
            fog_color: [0.7, 0.8, 0.9],
            gravity: [0.0, -9.81, 0.0],
            skybox: None,
        }
    }
}

impl SceneFile {
    pub fn new(name: &str) -> Self {
        Self {
            version: 1,
            name: name.to_string(),
            description: String::new(),
            entities: Vec::new(),
            settings: SceneSettings::default(),
        }
    }

    pub fn add_entity(&mut self, name: &str, transform: Transform) -> u32 {
        let id = self.entities.len() as u32;
        self.entities.push(SceneEntity {
            id, name: name.to_string(), parent: None,
            transform, components: HashMap::new(),
        });
        id
    }

    /// Save to JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load from JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }

    pub fn entity_count(&self) -> usize { self.entities.len() }
}
