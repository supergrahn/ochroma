use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModManifest {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub entry_point: String, // wasm file name
}

#[derive(Debug, Clone)]
pub struct LoadedMod {
    pub manifest: ModManifest,
    pub path: PathBuf,
    pub enabled: bool,
    pub load_order: u32,
}

pub struct ModManager {
    pub mods: Vec<LoadedMod>,
    pub mod_directory: PathBuf,
}

impl ModManager {
    pub fn new(mod_directory: PathBuf) -> Self {
        Self {
            mods: Vec::new(),
            mod_directory,
        }
    }

    /// Scan the mod directory for .ochroma_mod packages (directories with manifest.toml).
    pub fn scan(&mut self) {
        self.mods.clear();
        if let Ok(entries) = std::fs::read_dir(&self.mod_directory) {
            let mut order = 0u32;
            for entry in entries.flatten() {
                let manifest_path = entry.path().join("manifest.toml");
                if manifest_path.exists() {
                    if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                        if let Ok(manifest) = toml::from_str::<ModManifest>(&content) {
                            self.mods.push(LoadedMod {
                                manifest,
                                path: entry.path(),
                                enabled: true,
                                load_order: order,
                            });
                            order += 1;
                        }
                    }
                }
            }
        }
    }

    pub fn enable(&mut self, name: &str) {
        if let Some(m) = self.mods.iter_mut().find(|m| m.manifest.name == name) {
            m.enabled = true;
        }
    }

    pub fn disable(&mut self, name: &str) {
        if let Some(m) = self.mods.iter_mut().find(|m| m.manifest.name == name) {
            m.enabled = false;
        }
    }

    pub fn enabled_mods(&self) -> Vec<&LoadedMod> {
        let mut mods: Vec<&LoadedMod> = self.mods.iter().filter(|m| m.enabled).collect();
        mods.sort_by_key(|m| m.load_order);
        mods
    }

    /// Check for dependency conflicts.
    pub fn check_dependencies(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let enabled_names: Vec<&str> = self
            .enabled_mods()
            .iter()
            .map(|m| m.manifest.name.as_str())
            .collect();
        for m in self.enabled_mods() {
            for dep in &m.manifest.dependencies {
                if !enabled_names.contains(&dep.as_str()) {
                    errors.push(format!(
                        "{} requires {} but it's not enabled",
                        m.manifest.name, dep
                    ));
                }
            }
        }
        errors
    }

    pub fn mod_count(&self) -> usize {
        self.mods.len()
    }
}
