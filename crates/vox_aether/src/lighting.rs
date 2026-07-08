//! Lighting designer: time-of-day, weather modifiers, mood color presets.
//!
//! Ported from AetherSpectra `engines/aether/weather/lighting_designer.py`.
//!
//! # NOTE on sun direction
//! `SkySettings::sun_direction` here is a preset/authoring hint.
//! At runtime, consume `vox_core::celestial::celestial_key_light()` for the
//! accurate physically-based direction.
//!
//! # RECONCILED
//! Sun/moon direction from `vox_core::celestial`; this crate produces the
//! atmospheric spec that wraps it.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeOfDay {
    Dawn,
    Morning,
    Noon,
    Afternoon,
    Dusk,
    Evening,
    Night,
    Midnight,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Weather {
    Clear,
    PartlyCloudy,
    Overcast,
    Rainy,
    Stormy,
    Foggy,
    Snowy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mood {
    Neutral,
    Romantic,
    Tense,
    Joyful,
    Melancholic,
    Mysterious,
    Dramatic,
    Peaceful,
    Eerie,
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightSource {
    pub name: String,
    /// `"sun"`, `"area"`, `"point"`, `"spot"`, `"hdri"`.
    pub light_type: String,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub angle: f32,
    pub cast_shadows: bool,
    pub shadow_softness: f32,
    pub use_contact_shadows: bool,
}

impl Default for LightSource {
    fn default() -> Self {
        Self {
            name: String::new(),
            light_type: "area".to_string(),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            radius: 0.0,
            angle: 45.0,
            cast_shadows: true,
            shadow_softness: 0.5,
            use_contact_shadows: false,
        }
    }
}

/// Sky and atmosphere settings.
///
/// NOTE: `sun_direction` is a preset/authoring hint; at runtime, consume
/// `vox_core::celestial::celestial_key_light()` for accurate direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkySettings {
    /// `"nishita"`, `"hdri"`, `"gradient"`.
    pub sky_type: String,
    /// Preset/authoring hint — see module-level note on reconciliation.
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_size: f32,
    pub turbidity: f32,
    pub air_density: f32,
    pub dust_density: f32,
    pub ozone_density: f32,
    pub horizon_color: [f32; 3],
    pub zenith_color: [f32; 3],
    pub hdri_path: Option<String>,
    pub hdri_rotation: f32,
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            sky_type: "nishita".to_string(),
            sun_direction: [0.5, 0.5, 1.0],
            sun_intensity: 1.0,
            sun_size: 0.01,
            turbidity: 2.0,
            air_density: 1.0,
            dust_density: 0.0,
            ozone_density: 1.0,
            horizon_color: [0.8, 0.85, 1.0],
            zenith_color: [0.1, 0.2, 0.5],
            hdri_path: None,
            hdri_rotation: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumetricSettings {
    pub enabled: bool,
    pub density: f32,
    pub anisotropy: f32,
    pub absorption_color: [f32; 3],
    pub emission_color: [f32; 3],
}

impl Default for VolumetricSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 0.01,
            anisotropy: 0.3,
            absorption_color: [0.0, 0.0, 0.0],
            emission_color: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightingSetup {
    pub name: String,
    pub description: String,
    pub time_of_day: TimeOfDay,
    pub weather: Weather,
    pub mood: Mood,
    pub key_light: Option<LightSource>,
    pub fill_lights: Vec<LightSource>,
    pub rim_lights: Vec<LightSource>,
    pub practical_lights: Vec<LightSource>,
    pub ambient_lights: Vec<LightSource>,
    pub sky: SkySettings,
    pub volumetrics: VolumetricSettings,
    pub exposure: f32,
    pub gamma: f32,
    pub color_temperature_shift: f32,
    pub saturation: f32,
    pub contrast: f32,
}

// ---------------------------------------------------------------------------
// Preset data structs
// ---------------------------------------------------------------------------

/// Time-of-day preset data (from Python TIME_PRESETS).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimePreset {
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub exposure: f32,
    pub turbidity: f32,
}

/// Weather modifier data (from Python WEATHER_MODIFIERS).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeatherModifier {
    pub sun_intensity_mult: f32,
    pub shadow_softness: f32,
    pub sky_saturation: f32,
    pub volumetric_density: f32,
}

/// Mood color adjustment data (from Python MOOD_COLORS).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoodColor {
    pub color_shift: [f32; 3],
    pub saturation: f32,
    pub contrast: f32,
}

// ---------------------------------------------------------------------------
// Preset accessor functions
// ---------------------------------------------------------------------------

/// Return the time-of-day preset for the given time.
pub fn time_preset(t: TimeOfDay) -> TimePreset {
    match t {
        TimeOfDay::Dawn => TimePreset {
            sun_direction: [0.3, 0.1, 0.2],
            sun_intensity: 0.4,
            sun_color: [1.0, 0.7, 0.5],
            sky_horizon: [1.0, 0.6, 0.4],
            sky_zenith: [0.4, 0.5, 0.7],
            exposure: 0.5,
            turbidity: 3.0,
        },
        TimeOfDay::Morning => TimePreset {
            sun_direction: [0.5, 0.3, 0.5],
            sun_intensity: 0.7,
            sun_color: [1.0, 0.95, 0.9],
            sky_horizon: [0.9, 0.9, 1.0],
            sky_zenith: [0.3, 0.5, 0.8],
            exposure: 0.0,
            turbidity: 2.0,
        },
        TimeOfDay::Noon => TimePreset {
            sun_direction: [0.0, 0.0, 1.0],
            sun_intensity: 1.0,
            sun_color: [1.0, 1.0, 1.0],
            sky_horizon: [0.7, 0.8, 1.0],
            sky_zenith: [0.2, 0.4, 0.8],
            exposure: -0.3,
            turbidity: 2.0,
        },
        TimeOfDay::Afternoon => TimePreset {
            sun_direction: [-0.5, 0.3, 0.5],
            sun_intensity: 0.8,
            sun_color: [1.0, 0.98, 0.95],
            sky_horizon: [0.85, 0.85, 0.95],
            sky_zenith: [0.3, 0.5, 0.8],
            exposure: 0.0,
            turbidity: 2.5,
        },
        TimeOfDay::Dusk => TimePreset {
            sun_direction: [-0.3, 0.1, 0.15],
            sun_intensity: 0.3,
            sun_color: [1.0, 0.5, 0.3],
            sky_horizon: [1.0, 0.4, 0.3],
            sky_zenith: [0.3, 0.2, 0.5],
            exposure: 0.8,
            turbidity: 4.0,
        },
        TimeOfDay::Evening => TimePreset {
            sun_direction: [-0.1, 0.05, 0.1],
            sun_intensity: 0.1,
            sun_color: [0.9, 0.4, 0.2],
            sky_horizon: [0.3, 0.2, 0.3],
            sky_zenith: [0.1, 0.1, 0.2],
            exposure: 1.5,
            turbidity: 5.0,
        },
        TimeOfDay::Night => TimePreset {
            sun_direction: [0.0, 0.0, -0.3],
            sun_intensity: 0.02,
            sun_color: [0.7, 0.8, 1.0],
            sky_horizon: [0.05, 0.05, 0.1],
            sky_zenith: [0.02, 0.02, 0.05],
            exposure: 3.0,
            turbidity: 2.0,
        },
        TimeOfDay::Midnight => TimePreset {
            sun_direction: [0.0, 0.0, -0.5],
            sun_intensity: 0.01,
            sun_color: [0.6, 0.7, 1.0],
            sky_horizon: [0.02, 0.02, 0.05],
            sky_zenith: [0.01, 0.01, 0.02],
            exposure: 4.0,
            turbidity: 1.5,
        },
    }
}

/// Return the weather modifier for the given weather condition.
pub fn weather_modifier(w: Weather) -> WeatherModifier {
    match w {
        Weather::Clear => WeatherModifier {
            sun_intensity_mult: 1.0,
            shadow_softness: 0.1,
            sky_saturation: 1.0,
            volumetric_density: 0.0,
        },
        Weather::PartlyCloudy => WeatherModifier {
            sun_intensity_mult: 0.8,
            shadow_softness: 0.3,
            sky_saturation: 0.9,
            volumetric_density: 0.005,
        },
        Weather::Overcast => WeatherModifier {
            sun_intensity_mult: 0.3,
            shadow_softness: 0.8,
            sky_saturation: 0.6,
            volumetric_density: 0.01,
        },
        Weather::Rainy => WeatherModifier {
            sun_intensity_mult: 0.2,
            shadow_softness: 0.9,
            sky_saturation: 0.5,
            volumetric_density: 0.02,
        },
        Weather::Stormy => WeatherModifier {
            sun_intensity_mult: 0.1,
            shadow_softness: 1.0,
            sky_saturation: 0.3,
            volumetric_density: 0.03,
        },
        Weather::Foggy => WeatherModifier {
            sun_intensity_mult: 0.4,
            shadow_softness: 1.0,
            sky_saturation: 0.4,
            volumetric_density: 0.1,
        },
        Weather::Snowy => WeatherModifier {
            sun_intensity_mult: 0.6,
            shadow_softness: 0.5,
            sky_saturation: 0.7,
            volumetric_density: 0.02,
        },
    }
}

/// Return the mood color adjustment for the given mood.
pub fn mood_color(m: Mood) -> MoodColor {
    match m {
        Mood::Neutral => MoodColor {
            color_shift: [0.0, 0.0, 0.0],
            saturation: 1.0,
            contrast: 1.0,
        },
        Mood::Romantic => MoodColor {
            color_shift: [0.1, 0.0, -0.05],
            saturation: 0.9,
            contrast: 0.9,
        },
        Mood::Tense => MoodColor {
            color_shift: [0.0, -0.05, 0.05],
            saturation: 0.8,
            contrast: 1.2,
        },
        Mood::Joyful => MoodColor {
            color_shift: [0.05, 0.05, 0.0],
            saturation: 1.2,
            contrast: 1.0,
        },
        Mood::Melancholic => MoodColor {
            color_shift: [0.0, 0.0, 0.1],
            saturation: 0.7,
            contrast: 0.9,
        },
        Mood::Mysterious => MoodColor {
            color_shift: [-0.05, 0.0, 0.1],
            saturation: 0.8,
            contrast: 1.1,
        },
        Mood::Dramatic => MoodColor {
            color_shift: [0.0, 0.0, 0.0],
            saturation: 1.0,
            contrast: 1.4,
        },
        Mood::Peaceful => MoodColor {
            color_shift: [0.02, 0.02, 0.0],
            saturation: 0.85,
            contrast: 0.85,
        },
        Mood::Eerie => MoodColor {
            color_shift: [0.0, 0.1, 0.0],
            saturation: 0.6,
            contrast: 1.2,
        },
    }
}

// ---------------------------------------------------------------------------
// LightingDesigner
// ---------------------------------------------------------------------------

/// Automated lighting designer for 3-D environments.
///
/// Creates complete lighting setups based on time of day, weather conditions,
/// and emotional mood.
pub struct LightingDesigner;

impl LightingDesigner {
    pub fn new() -> Self {
        Self
    }

    /// Create complete lighting setup.
    pub fn create_lighting_setup(
        &self,
        environment_type: &str,
        time_of_day: TimeOfDay,
        weather: Weather,
        mood: Mood,
        include_practicals: bool,
        character_lighting: bool,
    ) -> LightingSetup {
        let tp = time_preset(time_of_day.clone());
        let wm = weather_modifier(weather.clone());
        let mc = mood_color(mood.clone());

        let sky = SkySettings {
            sky_type: "nishita".to_string(),
            sun_direction: tp.sun_direction,
            sun_intensity: tp.sun_intensity * wm.sun_intensity_mult,
            turbidity: tp.turbidity,
            horizon_color: tp.sky_horizon,
            zenith_color: tp.sky_zenith,
            ..Default::default()
        };

        let key_light = Some(self.create_key_light(&tp, &wm, environment_type));

        let fill_lights = if character_lighting {
            self.create_fill_lights(&tp)
        } else {
            vec![]
        };

        let rim_lights = if character_lighting {
            self.create_rim_lights(&tp)
        } else {
            vec![]
        };

        let practical_lights = if include_practicals && environment_type == "interior" {
            self.create_practical_lights(&time_of_day)
        } else {
            vec![]
        };

        let ambient_lights = self.create_ambient_lights(&tp);

        let volumetrics = VolumetricSettings {
            enabled: wm.volumetric_density > 0.0,
            density: wm.volumetric_density,
            anisotropy: 0.3,
            ..Default::default()
        };

        LightingSetup {
            name: format!("{time_of_day:?}_{weather:?}_{mood:?}").to_lowercase(),
            description: format!("{time_of_day:?} lighting, {weather:?} weather, {mood:?} mood")
                .to_lowercase(),
            time_of_day,
            weather,
            mood,
            key_light,
            fill_lights,
            rim_lights,
            practical_lights,
            ambient_lights,
            sky,
            volumetrics,
            exposure: tp.exposure,
            gamma: 1.0,
            color_temperature_shift: 0.0,
            saturation: mc.saturation,
            contrast: mc.contrast,
        }
    }

    // Private helpers

    fn create_key_light(
        &self,
        tp: &TimePreset,
        wm: &WeatherModifier,
        environment_type: &str,
    ) -> LightSource {
        let intensity = tp.sun_intensity * wm.sun_intensity_mult;
        if environment_type == "exterior" {
            LightSource {
                name: "sun".to_string(),
                light_type: "sun".to_string(),
                position: [
                    tp.sun_direction[0] * 100.0,
                    tp.sun_direction[1] * 100.0,
                    tp.sun_direction[2] * 100.0,
                ],
                color: tp.sun_color,
                intensity,
                shadow_softness: wm.shadow_softness,
                ..Default::default()
            }
        } else {
            LightSource {
                name: "key_light".to_string(),
                light_type: "area".to_string(),
                position: [3.0, 2.5, 0.0],
                rotation: [45.0, 0.0, 45.0],
                color: tp.sun_color,
                intensity: intensity * 0.5,
                radius: 2.0,
                shadow_softness: wm.shadow_softness,
                ..Default::default()
            }
        }
    }

    fn create_fill_lights(&self, tp: &TimePreset) -> Vec<LightSource> {
        let fill_intensity = tp.sun_intensity * 0.3;
        vec![
            LightSource {
                name: "fill_left".to_string(),
                light_type: "area".to_string(),
                position: [-2.0, 1.5, 2.0],
                color: [0.9, 0.95, 1.0],
                intensity: fill_intensity,
                radius: 1.5,
                cast_shadows: false,
                ..Default::default()
            },
            LightSource {
                name: "fill_right".to_string(),
                light_type: "area".to_string(),
                position: [2.0, 1.5, 2.0],
                color: [1.0, 0.98, 0.95],
                intensity: fill_intensity * 0.7,
                radius: 1.5,
                cast_shadows: false,
                ..Default::default()
            },
        ]
    }

    fn create_rim_lights(&self, tp: &TimePreset) -> Vec<LightSource> {
        let rim_intensity = tp.sun_intensity * 0.5;
        vec![LightSource {
            name: "rim_back".to_string(),
            light_type: "spot".to_string(),
            position: [0.0, 2.5, -3.0],
            rotation: [30.0, 0.0, 0.0],
            color: [1.0, 0.95, 0.9],
            intensity: rim_intensity,
            angle: 60.0,
            cast_shadows: false,
            ..Default::default()
        }]
    }

    fn create_practical_lights(&self, time_of_day: &TimeOfDay) -> Vec<LightSource> {
        let base_intensity = if matches!(time_of_day, TimeOfDay::Night | TimeOfDay::Midnight) {
            0.3
        } else {
            0.1
        };
        vec![LightSource {
            name: "lamp_a".to_string(),
            light_type: "point".to_string(),
            position: [-2.0, 1.2, 2.0],
            color: [1.0, 0.85, 0.6],
            intensity: base_intensity,
            radius: 0.2,
            shadow_softness: 0.5,
            ..Default::default()
        }]
    }

    fn create_ambient_lights(&self, tp: &TimePreset) -> Vec<LightSource> {
        let ambient_intensity = tp.sun_intensity * 0.1;
        vec![LightSource {
            name: "ambient_fill".to_string(),
            light_type: "area".to_string(),
            position: [0.0, 3.0, 0.0],
            rotation: [90.0, 0.0, 0.0],
            color: [0.9, 0.95, 1.0],
            intensity: ambient_intensity,
            radius: 5.0,
            cast_shadows: false,
            ..Default::default()
        }]
    }
}

impl Default for LightingDesigner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dawn_time_preset_values() {
        let p = time_preset(TimeOfDay::Dawn);
        assert_eq!(p.sun_intensity, 0.4);
        assert_eq!(p.turbidity, 3.0);
        assert_eq!(p.sun_color, [1.0, 0.7, 0.5]);
    }

    #[test]
    fn foggy_weather_volumetric_density() {
        let p = weather_modifier(Weather::Foggy);
        assert_eq!(p.volumetric_density, 0.1);
        assert_eq!(p.sun_intensity_mult, 0.4);
    }

    #[test]
    fn tense_mood_contrast() {
        let p = mood_color(Mood::Tense);
        assert_eq!(p.contrast, 1.2);
        assert_eq!(p.saturation, 0.8);
    }

    #[test]
    fn midnight_exposure() {
        let p = time_preset(TimeOfDay::Midnight);
        assert_eq!(p.exposure, 4.0);
        assert_eq!(p.sun_intensity, 0.01);
    }
}
