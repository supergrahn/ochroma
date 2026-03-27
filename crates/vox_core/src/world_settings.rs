use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSettings {
    // Time
    pub time_of_day: f32,
    pub day_length_seconds: f32,
    pub time_paused: bool,

    // Physics
    pub gravity: f32,
    pub physics_timestep: f32,

    // Atmosphere
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_color: [f32; 3],
    pub fog_start_distance: f32,

    // Lighting
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub sun_direction: [f32; 3],

    // Sky
    pub sky_enabled: bool,
    pub sky_color_zenith: [f32; 3],
    pub sky_color_horizon: [f32; 3],

    // Post-processing
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub exposure: f32,
    pub tone_mapping: String,
    pub vignette_strength: f32,
    pub ssao_enabled: bool,
    pub ssao_strength: f32,
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self {
            time_of_day: 12.0,
            day_length_seconds: 600.0,
            time_paused: false,
            gravity: 9.81,
            physics_timestep: 1.0 / 60.0,
            fog_enabled: false,
            fog_density: 0.002,
            fog_color: [0.7, 0.8, 0.9],
            fog_start_distance: 50.0,
            ambient_color: [0.1, 0.1, 0.12],
            ambient_intensity: 1.0,
            sun_color: [1.0, 0.95, 0.9],
            sun_intensity: 1.0,
            sun_direction: [0.3, -1.0, 0.2],
            sky_enabled: true,
            sky_color_zenith: [0.2, 0.4, 0.8],
            sky_color_horizon: [0.7, 0.8, 0.9],
            bloom_enabled: true,
            bloom_intensity: 0.3,
            exposure: 1.0,
            tone_mapping: "ACES".to_string(),
            vignette_strength: 0.15,
            ssao_enabled: true,
            ssao_strength: 0.5,
        }
    }
}

impl WorldSettings {
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&data).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_sensible() {
        let ws = WorldSettings::default();
        assert!((ws.time_of_day - 12.0).abs() < f32::EPSILON);
        assert!((ws.gravity - 9.81).abs() < 0.01);
        assert!(ws.sky_enabled);
        assert!(ws.bloom_enabled);
        assert!(ws.ssao_enabled);
        assert!(!ws.fog_enabled);
        assert!(!ws.time_paused);
        assert_eq!(ws.tone_mapping, "ACES");
    }

    #[test]
    fn save_load_round_trip() {
        let dir = std::env::temp_dir().join("vox_world_settings_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut ws = WorldSettings::default();
        ws.time_of_day = 18.5;
        ws.fog_enabled = true;
        ws.tone_mapping = "Filmic".to_string();
        ws.gravity = 3.72; // Mars

        ws.save(&path).unwrap();
        let loaded = WorldSettings::load(&path).unwrap();

        assert!((loaded.time_of_day - 18.5).abs() < f32::EPSILON);
        assert!(loaded.fog_enabled);
        assert_eq!(loaded.tone_mapping, "Filmic");
        assert!((loaded.gravity - 3.72).abs() < 0.01);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn all_fields_preserved() {
        let dir = std::env::temp_dir().join("vox_world_settings_fields_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings2.json");

        let mut ws = WorldSettings::default();
        ws.day_length_seconds = 1200.0;
        ws.physics_timestep = 1.0 / 120.0;
        ws.fog_density = 0.05;
        ws.fog_color = [1.0, 0.0, 0.0];
        ws.fog_start_distance = 100.0;
        ws.ambient_color = [0.5, 0.5, 0.5];
        ws.ambient_intensity = 2.0;
        ws.sun_color = [0.8, 0.8, 0.5];
        ws.sun_intensity = 3.0;
        ws.sun_direction = [0.0, -1.0, 0.0];
        ws.sky_color_zenith = [0.0, 0.0, 1.0];
        ws.sky_color_horizon = [1.0, 1.0, 1.0];
        ws.bloom_intensity = 0.8;
        ws.exposure = 2.5;
        ws.vignette_strength = 0.5;
        ws.ssao_strength = 1.0;

        ws.save(&path).unwrap();
        let loaded = WorldSettings::load(&path).unwrap();

        assert!((loaded.day_length_seconds - 1200.0).abs() < f32::EPSILON);
        assert!((loaded.fog_density - 0.05).abs() < 0.001);
        assert!((loaded.ambient_intensity - 2.0).abs() < f32::EPSILON);
        assert!((loaded.sun_intensity - 3.0).abs() < f32::EPSILON);
        assert!((loaded.bloom_intensity - 0.8).abs() < 0.001);
        assert!((loaded.exposure - 2.5).abs() < 0.001);
        assert!((loaded.ssao_strength - 1.0).abs() < f32::EPSILON);

        std::fs::remove_dir_all(&dir).ok();
    }
}
