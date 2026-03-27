use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameSettings {
    // Graphics
    pub resolution: (u32, u32),
    pub quality: String, // "Low", "Medium", "High", "Ultra"
    pub vsync: bool,
    pub fullscreen: bool,
    pub render_distance: f32,

    // Audio
    pub master_volume: f32,
    pub music_volume: f32,
    pub sfx_volume: f32,
    pub ambient_volume: f32,

    // Gameplay
    pub auto_save_interval_secs: f32,
    pub edge_scroll_enabled: bool,
    pub camera_speed: f32,
    pub scroll_speed: f32,

    // Accessibility
    pub ui_scale: f32,
    pub colorblind_mode: ColorblindMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorblindMode {
    None,
    Protanopia,
    Deuteranopia,
    Tritanopia,
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            quality: "High".to_string(),
            vsync: true,
            fullscreen: false,
            render_distance: 5000.0,
            master_volume: 0.8,
            music_volume: 0.5,
            sfx_volume: 0.7,
            ambient_volume: 0.6,
            auto_save_interval_secs: 300.0,
            edge_scroll_enabled: true,
            camera_speed: 1.0,
            scroll_speed: 1.0,
            ui_scale: 1.0,
            colorblind_mode: ColorblindMode::None,
        }
    }
}

impl GameSettings {
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let toml = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, toml).map_err(|e| e.to_string())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        toml::from_str(&content).map_err(|e| e.to_string())
    }
}
