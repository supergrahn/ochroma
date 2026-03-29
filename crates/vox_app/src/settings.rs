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

// ── AppSettings (engine-level) ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub resolution: (u32, u32),
    pub vsync: bool,
    pub master_volume: f32,
    pub fullscreen: bool,
    pub render_quality: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            vsync: true,
            master_volume: 0.8,
            fullscreen: false,
            render_quality: "High".to_string(),
        }
    }
}

pub fn load_settings(path: &std::path::Path) -> AppSettings {
    match std::fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content).unwrap_or_default(),
        Err(_) => AppSettings::default(),
    }
}

pub fn save_settings(settings: &AppSettings, path: &std::path::Path) -> Result<(), String> {
    let toml_str = toml::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, toml_str).map_err(|e| e.to_string())
}

pub fn show_settings_panel(ctx: &egui::Context, settings: &mut AppSettings, open: &mut bool) -> bool {
    let mut changed = false;
    egui::Window::new("Settings")
        .open(open)
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading("Graphics");
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                let mut w = settings.resolution.0 as f32;
                let mut h = settings.resolution.1 as f32;
                if ui.add(egui::DragValue::new(&mut w).range(640.0..=3840.0).prefix("W: ")).changed() {
                    settings.resolution.0 = w as u32; changed = true;
                }
                if ui.add(egui::DragValue::new(&mut h).range(480.0..=2160.0).prefix("H: ")).changed() {
                    settings.resolution.1 = h as u32; changed = true;
                }
            });
            if ui.checkbox(&mut settings.vsync, "VSync").changed() { changed = true; }
            if ui.checkbox(&mut settings.fullscreen, "Fullscreen").changed() { changed = true; }
            ui.horizontal(|ui| {
                ui.label("Quality:");
                egui::ComboBox::from_id_salt("quality_combo")
                    .selected_text(&settings.render_quality)
                    .show_ui(ui, |ui| {
                        for q in &["Low", "Medium", "High", "Ultra"] {
                            if ui.selectable_label(settings.render_quality == *q, *q).clicked() {
                                settings.render_quality = q.to_string(); changed = true;
                            }
                        }
                    });
            });
            ui.separator();
            ui.heading("Audio");
            ui.horizontal(|ui| {
                ui.label("Master Volume:");
                if ui.add(egui::Slider::new(&mut settings.master_volume, 0.0..=1.0)).changed() {
                    changed = true;
                }
            });
        });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_settings_default() {
        let s = AppSettings::default();
        assert_eq!(s.resolution, (1920, 1080));
        assert!(s.vsync);
        assert!((s.master_volume - 0.8).abs() < 1e-5);
    }

    #[test]
    fn load_settings_falls_back_to_default_on_missing_file() {
        let path = std::path::Path::new("/tmp/nonexistent_ochroma_settings_test.toml");
        let s = load_settings(path);
        assert_eq!(s.resolution, (1920, 1080));
    }

    #[test]
    fn save_and_load_settings_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("app_settings.toml");
        let mut s = AppSettings::default();
        s.resolution = (3840, 2160);
        s.vsync = false;
        s.master_volume = 0.5;
        save_settings(&s, &path).unwrap();
        let loaded = load_settings(&path);
        assert_eq!(loaded.resolution, (3840, 2160));
        assert!(!loaded.vsync);
        assert!((loaded.master_volume - 0.5).abs() < 1e-5);
    }

    #[test]
    fn load_settings_falls_back_on_malformed_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_settings.toml");
        std::fs::write(&path, "this is not valid toml {{{{").unwrap();
        let s = load_settings(&path);
        assert_eq!(s.resolution, (1920, 1080));
    }
}
