use serde::{Deserialize, Serialize};
use std::path::Path;

/// A complete game map/level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapFile {
    pub version: u32,
    pub name: String,
    pub description: String,
    pub terrain: TerrainConfig,
    pub foliage_rules: Vec<FoliageConfig>,
    pub placed_objects: Vec<PlacedObject>,
    pub lights: Vec<LightConfig>,
    pub settings: MapSettings,
    pub spawn_points: Vec<SpawnPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConfig {
    pub heightmap_path: Option<String>, // path to raw heightmap file
    pub width: u32,                     // cells
    pub height: u32,                    // cells
    pub cell_size: f32,                 // metres
    pub max_elevation: f32,             // metres
    pub material_zones: Vec<MaterialZoneConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialZoneConfig {
    pub max_height: f32,
    pub material_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoliageConfig {
    pub name: String,
    pub asset_path: String,
    pub density: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub max_slope: f32,
    pub scale_range: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacedObject {
    pub name: String,
    pub asset_path: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4], // quaternion xyzw
    pub scale: [f32; 3],
    pub scripts: Vec<String>,                              // attached script names
    pub properties: std::collections::HashMap<String, String>, // custom key-value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightConfig {
    pub light_type: String, // "point", "directional", "spot"
    pub position: [f32; 3],
    pub direction: Option<[f32; 3]>,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapSettings {
    pub ambient_light: [f32; 3],
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_color: [f32; 3],
    pub gravity: f32,
    pub time_of_day: f32,   // starting hour
    pub weather: String,     // starting weather
    pub skybox: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnPoint {
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub is_default: bool,
}

impl Default for MapSettings {
    fn default() -> Self {
        Self {
            ambient_light: [0.1, 0.1, 0.12],
            fog_enabled: false,
            fog_density: 0.001,
            fog_color: [0.7, 0.8, 0.9],
            gravity: 9.81,
            time_of_day: 12.0,
            weather: "clear".into(),
            skybox: None,
        }
    }
}

impl MapFile {
    pub fn new(name: &str) -> Self {
        Self {
            version: 1,
            name: name.to_string(),
            description: String::new(),
            terrain: TerrainConfig {
                heightmap_path: None,
                width: 256,
                height: 256,
                cell_size: 1.0,
                max_elevation: 50.0,
                material_zones: Vec::new(),
            },
            foliage_rules: Vec::new(),
            placed_objects: Vec::new(),
            lights: Vec::new(),
            settings: MapSettings::default(),
            spawn_points: vec![SpawnPoint {
                name: "Default".into(),
                position: [0.0, 10.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                is_default: true,
            }],
        }
    }

    /// Add a placed object to the map.
    pub fn place_object(&mut self, name: &str, asset: &str, position: [f32; 3]) {
        self.placed_objects.push(PlacedObject {
            name: name.to_string(),
            asset_path: asset.to_string(),
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            scripts: Vec::new(),
            properties: std::collections::HashMap::new(),
        });
    }

    /// Add a light to the map.
    pub fn add_light(
        &mut self,
        light_type: &str,
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    ) {
        self.lights.push(LightConfig {
            light_type: light_type.to_string(),
            position,
            direction: None,
            color,
            intensity,
            radius: Some(50.0),
        });
    }

    /// Save to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load from JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }

    pub fn object_count(&self) -> usize {
        self.placed_objects.len()
    }

    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Get the default spawn point.
    pub fn default_spawn(&self) -> Option<&SpawnPoint> {
        self.spawn_points.iter().find(|s| s.is_default)
    }
}
