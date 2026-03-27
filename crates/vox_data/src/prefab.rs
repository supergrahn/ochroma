use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A prefab -- a reusable entity template with hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefab {
    pub name: String,
    pub entities: Vec<PrefabEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefabEntity {
    pub name: String,
    pub local_position: [f32; 3],
    pub local_rotation: [f32; 4],
    pub local_scale: [f32; 3],
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub tags: Vec<String>,
    pub children_indices: Vec<usize>,
    pub components: HashMap<String, serde_json::Value>,
}

impl Prefab {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            entities: Vec::new(),
        }
    }

    pub fn add_entity(&mut self, entity: PrefabEntity) -> usize {
        let idx = self.entities.len();
        self.entities.push(entity);
        idx
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&data).map_err(|e| e.to_string())
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Instantiate this prefab at a world position.
    /// Returns a list of entities to spawn with their world-space transforms.
    pub fn instantiate(&self, world_position: [f32; 3]) -> Vec<PrefabInstance> {
        self.entities
            .iter()
            .map(|e| PrefabInstance {
                name: e.name.clone(),
                world_position: [
                    e.local_position[0] + world_position[0],
                    e.local_position[1] + world_position[1],
                    e.local_position[2] + world_position[2],
                ],
                world_rotation: e.local_rotation,
                world_scale: e.local_scale,
                asset_path: e.asset_path.clone(),
                scripts: e.scripts.clone(),
                tags: e.tags.clone(),
            })
            .collect()
    }
}

pub struct PrefabInstance {
    pub name: String,
    pub world_position: [f32; 3],
    pub world_rotation: [f32; 4],
    pub world_scale: [f32; 3],
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub tags: Vec<String>,
}
