use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetType {
    Building,
    Prop,
    Vegetation,
    Terrain,
    Character,
    Component,
    Vehicle,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetPipeline {
    ProcGS,
    Turnaround,
    NeuralInfill,
    LyraCapture,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssetEntry {
    pub uuid: Uuid,
    pub name: String,
    pub path: String,
    pub style: String,
    pub asset_type: AssetType,
    pub pipeline: AssetPipeline,
    pub tags: Vec<String>,
    pub description: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct IndexWrapper {
    assets: Vec<AssetEntry>,
}

pub struct AssetLibrary {
    entries: HashMap<Uuid, AssetEntry>,
}

impl AssetLibrary {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    pub fn register(&mut self, entry: AssetEntry) {
        self.entries.insert(entry.uuid, entry);
    }

    pub fn get(&self, uuid: Uuid) -> Option<&AssetEntry> {
        self.entries.get(&uuid)
    }

    pub fn search_by_tag(&self, tag: &str) -> Vec<&AssetEntry> {
        self.entries
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    pub fn search_by_type(&self, asset_type: &AssetType) -> Vec<&AssetEntry> {
        self.entries
            .values()
            .filter(|e| &e.asset_type == asset_type)
            .collect()
    }

    pub fn all(&self) -> impl Iterator<Item = &AssetEntry> {
        self.entries.values()
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    pub fn save_index(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let entries: Vec<AssetEntry> = self.entries.values().cloned().collect();
        let wrapper = IndexWrapper { assets: entries };
        let toml_str = toml::to_string_pretty(&wrapper)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, toml_str)
    }

    pub fn load_index(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let wrapper: IndexWrapper = toml::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut lib = Self::new();
        for entry in wrapper.assets {
            lib.register(entry);
        }
        Ok(lib)
    }
}

impl Default for AssetLibrary {
    fn default() -> Self {
        Self::new()
    }
}
