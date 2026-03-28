//! VFX editor data model.
//!
//! Extends the existing VFX system with editor metadata, categories,
//! and a library for managing editable VFX assets.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::vfx::{self, VfxEffect};

/// An editable VFX definition with editor metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfxAsset {
    pub name: String,
    pub effect: VfxEffect,
    pub thumbnail_path: Option<String>,
    pub category: VfxCategory,
    pub description: String,
    pub tags: Vec<String>,
    pub preview_camera_distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VfxCategory {
    Fire,
    Smoke,
    Explosion,
    Weather,
    Magic,
    Environment,
    UI,
    Custom,
}

/// A curve editor point (for editing CurveF32 in the UI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveEditorPoint {
    pub time: f32,
    pub value: f32,
    pub tangent_in: f32,
    pub tangent_out: f32,
    pub interpolation: CurveInterpolation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CurveInterpolation {
    Linear,
    Bezier,
    Step,
}

/// A VFX library — collection of editable effects.
pub struct VfxLibrary {
    pub effects: Vec<VfxAsset>,
}

impl VfxLibrary {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    pub fn add(&mut self, asset: VfxAsset) {
        self.effects.push(asset);
    }

    pub fn find(&self, name: &str) -> Option<&VfxAsset> {
        self.effects.iter().find(|e| e.name == name)
    }

    pub fn by_category(&self, cat: VfxCategory) -> Vec<&VfxAsset> {
        self.effects.iter().filter(|e| e.category == cat).collect()
    }

    pub fn save_to_dir(&self, dir: &Path) -> Result<usize, String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut count = 0;
        for asset in &self.effects {
            let filename = format!("{}.vfx.json", asset.name.replace(' ', "_"));
            let path = dir.join(filename);
            let json = serde_json::to_string_pretty(asset).map_err(|e| e.to_string())?;
            std::fs::write(path, json).map_err(|e| e.to_string())?;
            count += 1;
        }
        Ok(count)
    }

    pub fn load_from_dir(&mut self, dir: &Path) -> Result<usize, String> {
        if !dir.exists() {
            return Err(format!("Directory does not exist: {}", dir.display()));
        }
        let mut count = 0;
        let entries = std::fs::read_dir(dir).map_err(|e| e.to_string())?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let json = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                if let Ok(asset) = serde_json::from_str::<VfxAsset>(&json) {
                    self.effects.push(asset);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    pub fn count(&self) -> usize {
        self.effects.len()
    }

    /// Create default library with pre-built effects.
    pub fn with_defaults() -> Self {
        let mut lib = Self::new();

        lib.add(VfxAsset {
            name: "campfire".into(),
            effect: vfx::effect_fire(),
            thumbnail_path: None,
            category: VfxCategory::Fire,
            description: "A warm campfire with flickering flames".into(),
            tags: vec!["fire".into(), "warm".into(), "ambient".into()],
            preview_camera_distance: 5.0,
        });

        lib.add(VfxAsset {
            name: "chimney_smoke".into(),
            effect: vfx::effect_smoke(),
            thumbnail_path: None,
            category: VfxCategory::Smoke,
            description: "Gentle smoke rising from a chimney".into(),
            tags: vec!["smoke".into(), "ambient".into()],
            preview_camera_distance: 8.0,
        });

        lib.add(VfxAsset {
            name: "barrel_explosion".into(),
            effect: vfx::effect_explosion(),
            thumbnail_path: None,
            category: VfxCategory::Explosion,
            description: "Explosive barrel burst with debris".into(),
            tags: vec!["explosion".into(), "destruction".into()],
            preview_camera_distance: 12.0,
        });

        lib.add(VfxAsset {
            name: "rain_heavy".into(),
            effect: vfx::effect_rain(),
            thumbnail_path: None,
            category: VfxCategory::Weather,
            description: "Heavy rainfall over a large area".into(),
            tags: vec!["rain".into(), "weather".into()],
            preview_camera_distance: 15.0,
        });

        lib.add(VfxAsset {
            name: "magic_sparkle".into(),
            effect: vfx::effect_sparkle(),
            thumbnail_path: None,
            category: VfxCategory::Magic,
            description: "Magical sparkle effect for enchantments".into(),
            tags: vec!["magic".into(), "sparkle".into(), "enchant".into()],
            preview_camera_distance: 3.0,
        });

        lib.add(VfxAsset {
            name: "ground_dust".into(),
            effect: vfx::effect_dust(),
            thumbnail_path: None,
            category: VfxCategory::Environment,
            description: "Dust particles drifting near the ground".into(),
            tags: vec!["dust".into(), "environment".into(), "ambient".into()],
            preview_camera_distance: 5.0,
        });

        lib
    }
}

impl Default for VfxLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_library() {
        let lib = VfxLibrary::new();
        assert_eq!(lib.count(), 0);
    }

    #[test]
    fn test_add_effects() {
        let mut lib = VfxLibrary::new();
        lib.add(VfxAsset {
            name: "test_fire".into(),
            effect: vfx::effect_fire(),
            thumbnail_path: None,
            category: VfxCategory::Fire,
            description: "Test fire".into(),
            tags: vec!["fire".into()],
            preview_camera_distance: 5.0,
        });
        assert_eq!(lib.count(), 1);
    }

    #[test]
    fn test_find_by_name() {
        let lib = VfxLibrary::with_defaults();
        let found = lib.find("campfire");
        assert!(found.is_some());
        assert_eq!(found.unwrap().category, VfxCategory::Fire);

        let not_found = lib.find("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_filter_by_category() {
        let lib = VfxLibrary::with_defaults();
        let fires = lib.by_category(VfxCategory::Fire);
        assert!(!fires.is_empty());
        for f in &fires {
            assert_eq!(f.category, VfxCategory::Fire);
        }

        let weather = lib.by_category(VfxCategory::Weather);
        assert!(!weather.is_empty());

        let ui = lib.by_category(VfxCategory::UI);
        assert!(ui.is_empty());
    }

    #[test]
    fn test_save_load_round_trip() {
        let lib = VfxLibrary::with_defaults();
        let original_count = lib.count();
        assert!(original_count > 0);

        let dir = std::env::temp_dir().join("ochroma_vfx_test");
        let _ = std::fs::remove_dir_all(&dir);

        let saved = lib.save_to_dir(&dir).unwrap();
        assert_eq!(saved, original_count);

        let mut loaded_lib = VfxLibrary::new();
        let loaded = loaded_lib.load_from_dir(&dir).unwrap();
        assert_eq!(loaded, original_count);
        assert_eq!(loaded_lib.count(), original_count);

        // Verify a specific effect survived the round-trip
        let campfire = loaded_lib.find("campfire");
        assert!(campfire.is_some());
        assert_eq!(campfire.unwrap().category, VfxCategory::Fire);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
