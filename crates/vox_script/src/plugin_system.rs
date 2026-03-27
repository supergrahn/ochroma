use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Metadata describing a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub engine_version_min: String,
    pub engine_version_max: String,
    pub entry_point: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

/// Current state of a plugin.
#[derive(Debug, Clone, PartialEq)]
pub enum PluginState {
    Discovered,
    Loading,
    Active,
    Error(String),
    Disabled,
}

/// Internal plugin record.
#[derive(Debug, Clone)]
struct PluginRecord {
    manifest: PluginManifest,
    state: PluginState,
    dir: PathBuf,
    last_modified: Option<SystemTime>,
}

/// Manages hot-loadable plugins.
#[derive(Debug)]
pub struct PluginManager {
    plugins: HashMap<String, PluginRecord>,
    engine_version: String,
}

impl PluginManager {
    pub fn new(engine_version: &str) -> Self {
        Self {
            plugins: HashMap::new(),
            engine_version: engine_version.to_string(),
        }
    }

    /// Discover plugins in a directory. Each subdirectory should contain a `manifest.toml`.
    pub fn discover(&mut self, dir: &Path) -> Vec<String> {
        let mut discovered = Vec::new();

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return discovered,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let manifest_path = path.join("manifest.toml");
            if !manifest_path.exists() {
                continue;
            }

            let content = match std::fs::read_to_string(&manifest_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let manifest: PluginManifest = match toml::from_str(&content) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let name = manifest.name.clone();

            // Check that entry point file exists.
            let entry_path = path.join(&manifest.entry_point);
            if !entry_path.exists() {
                self.plugins.insert(
                    name.clone(),
                    PluginRecord {
                        manifest,
                        state: PluginState::Error("Missing entry point file".to_string()),
                        dir: path,
                        last_modified: None,
                    },
                );
                discovered.push(name);
                continue;
            }

            let modified = std::fs::metadata(&entry_path)
                .and_then(|m| m.modified())
                .ok();

            self.plugins.insert(
                name.clone(),
                PluginRecord {
                    manifest,
                    state: PluginState::Discovered,
                    dir: path,
                    last_modified: modified,
                },
            );
            discovered.push(name);
        }

        discovered
    }

    /// Check if a manifest is compatible with the current engine version.
    pub fn is_compatible(&self, manifest: &PluginManifest) -> bool {
        version_in_range(
            &self.engine_version,
            &manifest.engine_version_min,
            &manifest.engine_version_max,
        )
    }

    /// Load a plugin by name. Returns an error string if it fails.
    pub fn load(&mut self, name: &str) -> Result<(), String> {
        let record = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| format!("Plugin '{}' not found", name))?;

        if matches!(record.state, PluginState::Error(_)) {
            return Err(format!(
                "Plugin '{}' is in error state: {:?}",
                name, record.state
            ));
        }

        let compatible = version_in_range(
            &self.engine_version,
            &record.manifest.engine_version_min,
            &record.manifest.engine_version_max,
        );
        if !compatible {
            record.state = PluginState::Error("Incompatible engine version".to_string());
            return Err("Incompatible engine version".to_string());
        }

        record.state = PluginState::Loading;
        // In a real implementation, we would dynamically load the plugin here.
        record.state = PluginState::Active;
        Ok(())
    }

    /// Unload a plugin by name.
    pub fn unload(&mut self, name: &str) -> Result<(), String> {
        let record = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| format!("Plugin '{}' not found", name))?;
        record.state = PluginState::Disabled;
        Ok(())
    }

    /// Return names of all active plugins.
    pub fn active_plugins(&self) -> Vec<String> {
        self.plugins
            .iter()
            .filter(|(_, r)| r.state == PluginState::Active)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Return total number of known plugins.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Get the state of a plugin by name.
    pub fn plugin_state(&self, name: &str) -> Option<&PluginState> {
        self.plugins.get(name).map(|r| &r.state)
    }

    /// Check for plugins whose entry-point file has been modified since last check.
    /// Returns names of updated plugins.
    pub fn check_for_updates(&mut self) -> Vec<String> {
        let mut updated = Vec::new();

        for (name, record) in &mut self.plugins {
            let entry_path = record.dir.join(&record.manifest.entry_point);
            let current_modified = std::fs::metadata(&entry_path)
                .and_then(|m| m.modified())
                .ok();

            if let (Some(current), Some(last)) = (current_modified, record.last_modified) {
                if current > last {
                    updated.push(name.clone());
                    record.last_modified = Some(current);
                }
            }
        }

        updated
    }

}

/// Simple semver comparison: check if version is in [min, max] range.
fn version_in_range(version: &str, min: &str, max: &str) -> bool {
    let v = parse_version(version);
    let lo = parse_version(min);
    let hi = parse_version(max);
    v >= lo && v <= hi
}

fn parse_version(s: &str) -> (u32, u32, u32) {
    let parts: Vec<u32> = s
        .split('.')
        .filter_map(|p| p.parse().ok())
        .collect();
    (
        parts.first().copied().unwrap_or(0),
        parts.get(1).copied().unwrap_or(0),
        parts.get(2).copied().unwrap_or(0),
    )
}
