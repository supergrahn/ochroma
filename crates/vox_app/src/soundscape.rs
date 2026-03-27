use glam::Vec3;

/// An ambient audio layer that plays based on conditions.
#[derive(Debug, Clone)]
pub struct AmbientLayer {
    pub name: String,
    pub base_volume: f32,
    pub current_volume: f32,
    pub category: SoundCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundCategory {
    UrbanHum,      // Traffic, city noise
    Nature,        // Birds, wind, water
    Weather,       // Rain, thunder, wind
    Construction,  // Building sites
    Emergency,     // Sirens
    Music,         // Background music
}

/// Manages the city's audio atmosphere.
pub struct Soundscape {
    pub layers: Vec<AmbientLayer>,
    pub listener_position: Vec3,
    pub time_of_day: f32,
    pub weather_intensity: f32, // 0 = clear, 1 = heavy weather
    pub construction_active: bool,
    pub emergency_active: bool,
}

impl Soundscape {
    pub fn new() -> Self {
        Self {
            layers: vec![
                AmbientLayer { name: "urban_hum".into(), base_volume: 0.3, current_volume: 0.0, category: SoundCategory::UrbanHum },
                AmbientLayer { name: "birds".into(), base_volume: 0.2, current_volume: 0.0, category: SoundCategory::Nature },
                AmbientLayer { name: "wind".into(), base_volume: 0.15, current_volume: 0.0, category: SoundCategory::Nature },
                AmbientLayer { name: "rain".into(), base_volume: 0.0, current_volume: 0.0, category: SoundCategory::Weather },
                AmbientLayer { name: "construction".into(), base_volume: 0.0, current_volume: 0.0, category: SoundCategory::Construction },
                AmbientLayer { name: "sirens".into(), base_volume: 0.0, current_volume: 0.0, category: SoundCategory::Emergency },
                AmbientLayer { name: "music".into(), base_volume: 0.2, current_volume: 0.2, category: SoundCategory::Music },
            ],
            listener_position: Vec3::ZERO,
            time_of_day: 12.0,
            weather_intensity: 0.0,
            construction_active: false,
            emergency_active: false,
        }
    }

    /// Update soundscape based on current game state.
    pub fn update(&mut self, camera_pos: Vec3, hour: f32, population: u32, weather: f32, constructing: bool, emergency: bool) {
        self.listener_position = camera_pos;
        self.time_of_day = hour;
        self.weather_intensity = weather;
        self.construction_active = constructing;
        self.emergency_active = emergency;

        for layer in &mut self.layers {
            layer.current_volume = match layer.category {
                SoundCategory::UrbanHum => {
                    // Louder with more population, quieter at night
                    let pop_factor = (population as f32 / 10000.0).min(1.0);
                    let time_factor = if (7.0..22.0).contains(&hour) { 1.0 } else { 0.3 };
                    layer.base_volume * pop_factor * time_factor
                }
                SoundCategory::Nature => {
                    // Louder in parks/outskirts, louder at dawn/dusk
                    let dawn_dusk = if (5.0..8.0).contains(&hour) || (17.0..20.0).contains(&hour) { 1.5 } else { 0.8 };
                    // Altitude factor: higher camera = more wind, less birds
                    let alt_factor = if layer.name == "wind" {
                        (camera_pos.y / 100.0).min(1.0)
                    } else {
                        (1.0 - camera_pos.y / 200.0).max(0.2)
                    };
                    layer.base_volume * dawn_dusk * alt_factor
                }
                SoundCategory::Weather => {
                    weather * 0.8 // rain volume proportional to weather intensity
                }
                SoundCategory::Construction => {
                    if constructing && (8.0..18.0).contains(&hour) { 0.4 } else { 0.0 }
                }
                SoundCategory::Emergency => {
                    if emergency { 0.6 } else { 0.0 }
                }
                SoundCategory::Music => {
                    layer.base_volume // constant background music
                }
            };
        }
    }

    /// Get the volume of a specific layer.
    pub fn layer_volume(&self, name: &str) -> f32 {
        self.layers.iter().find(|l| l.name == name).map(|l| l.current_volume).unwrap_or(0.0)
    }

    /// Get all currently audible layers.
    pub fn audible_layers(&self) -> Vec<&AmbientLayer> {
        self.layers.iter().filter(|l| l.current_volume > 0.01).collect()
    }
}

impl Default for Soundscape {
    fn default() -> Self {
        Self::new()
    }
}
