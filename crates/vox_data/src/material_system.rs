use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialDefinition {
    pub material: MaterialHeader,
    pub spectral: SpectralConfig,
    pub properties: MaterialProperties,
    pub worn: Option<WornConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialHeader {
    pub name: String,
    #[serde(rename = "type")]
    pub mat_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    pub bands: [f32; 8],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub roughness: f32,
    pub metallic: f32,
    pub opacity: f32,
    #[serde(default)]
    pub emission: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WornConfig {
    pub bands: [f32; 8],
    pub wear_factor: f32,
}

impl MaterialDefinition {
    /// Load from a TOML file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_toml_str(&content, path.display().to_string())
    }

    /// Parse from a TOML string.
    pub fn from_toml_str(content: &str, source: String) -> Result<Self, String> {
        toml::from_str(content).map_err(|e| format!("Parse error in {}: {}", source, e))
    }

    /// Save to a TOML file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let content = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, content).map_err(|e| e.to_string())
    }

    /// Get the effective spectral bands (interpolated by wear).
    pub fn effective_bands(&self) -> [f32; 8] {
        match &self.worn {
            Some(worn) => {
                let f = worn.wear_factor.clamp(0.0, 1.0);
                std::array::from_fn(|i| self.spectral.bands[i] * (1.0 - f) + worn.bands[i] * f)
            }
            None => self.spectral.bands,
        }
    }

    /// Convert to u16 spectral array for GaussianSplat.
    pub fn to_splat_spectral(&self) -> [u16; 8] {
        let bands = self.effective_bands();
        std::array::from_fn(|i| f16::from_f32(bands[i]).to_bits())
    }
}

/// Material library — loads and caches materials from a directory.
pub struct MaterialSystem {
    materials: HashMap<String, MaterialDefinition>,
    file_paths: HashMap<String, std::path::PathBuf>,
}

impl Default for MaterialSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialSystem {
    pub fn new() -> Self {
        Self {
            materials: HashMap::new(),
            file_paths: HashMap::new(),
        }
    }

    /// Load all .toml material files from a directory.
    pub fn load_directory(&mut self, dir: &Path) -> Result<usize, String> {
        let mut count = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "toml").unwrap_or(false) {
                    match MaterialDefinition::load(&path) {
                        Ok(mat) => {
                            let name = mat.material.name.clone();
                            self.file_paths.insert(name.clone(), path);
                            self.materials.insert(name, mat);
                            count += 1;
                        }
                        Err(e) => {
                            eprintln!("[materials] Failed to load {}: {}", path.display(), e)
                        }
                    }
                }
            }
        }
        Ok(count)
    }

    /// Get a material by name.
    pub fn get(&self, name: &str) -> Option<&MaterialDefinition> {
        self.materials.get(name)
    }

    /// Reload a material from disk (for hot-reload).
    pub fn reload(&mut self, name: &str) -> Result<(), String> {
        let path = self
            .file_paths
            .get(name)
            .ok_or("Material not found")?
            .clone();
        let mat = MaterialDefinition::load(&path)?;
        self.materials.insert(name.to_string(), mat);
        Ok(())
    }

    /// Reload all materials.
    pub fn reload_all(&mut self) -> Vec<String> {
        let names: Vec<String> = self.materials.keys().cloned().collect();
        let mut errors = Vec::new();
        for name in names {
            if let Err(e) = self.reload(&name) {
                errors.push(format!("{}: {}", name, e));
            }
        }
        errors
    }

    pub fn count(&self) -> usize {
        self.materials.len()
    }

    pub fn names(&self) -> Vec<&str> {
        self.materials.keys().map(|s| s.as_str()).collect()
    }
}

/// Create some default material TOML files in a directory.
pub fn create_default_materials(dir: &Path) -> Result<usize, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    let defaults = vec![
        MaterialDefinition {
            material: MaterialHeader {
                name: "Brick Red".into(),
                mat_type: "spectral".into(),
            },
            spectral: SpectralConfig {
                bands: [0.08, 0.08, 0.10, 0.15, 0.25, 0.55, 0.65, 0.60],
            },
            properties: MaterialProperties {
                roughness: 0.8,
                metallic: 0.0,
                opacity: 1.0,
                emission: 0.0,
            },
            worn: Some(WornConfig {
                bands: [0.06, 0.06, 0.08, 0.12, 0.18, 0.38, 0.45, 0.40],
                wear_factor: 0.0,
            }),
        },
        MaterialDefinition {
            material: MaterialHeader {
                name: "Glass Clear".into(),
                mat_type: "spectral".into(),
            },
            spectral: SpectralConfig {
                bands: [0.85, 0.88, 0.90, 0.91, 0.91, 0.90, 0.89, 0.87],
            },
            properties: MaterialProperties {
                roughness: 0.1,
                metallic: 0.0,
                opacity: 0.5,
                emission: 0.0,
            },
            worn: None,
        },
        MaterialDefinition {
            material: MaterialHeader {
                name: "Steel".into(),
                mat_type: "spectral".into(),
            },
            spectral: SpectralConfig {
                bands: [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.63, 0.63],
            },
            properties: MaterialProperties {
                roughness: 0.3,
                metallic: 1.0,
                opacity: 1.0,
                emission: 0.0,
            },
            worn: Some(WornConfig {
                bands: [0.25, 0.22, 0.20, 0.22, 0.30, 0.45, 0.50, 0.48],
                wear_factor: 0.0,
            }),
        },
    ];

    for mat in &defaults {
        let filename = mat.material.name.to_lowercase().replace(' ', "_") + ".toml";
        mat.save(&dir.join(filename))?;
    }

    Ok(defaults.len())
}
