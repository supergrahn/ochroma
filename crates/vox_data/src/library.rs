use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetType {
    Building,
    Prop,
    Vegetation,
    Terrain,
    Character,
    Component,
    Vehicle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetPipeline {
    ProcGS,
    Turnaround,
    NeuralInfill,
    LyraCapture,
}

#[derive(Debug, Clone)]
pub struct AssetEntry {
    pub uuid: Uuid,
    pub name: String,
    pub asset_type: AssetType,
    pub pipeline: AssetPipeline,
    pub tags: Vec<String>,
    pub description: String,
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

    pub fn get(&self, uuid: &Uuid) -> Option<&AssetEntry> {
        self.entries.get(uuid)
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
}

impl Default for AssetLibrary {
    fn default() -> Self {
        Self::new()
    }
}
