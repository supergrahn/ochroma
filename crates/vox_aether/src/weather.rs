//! Weather effects: particle configs, fog configs, weather presets and blending.
//!
//! Ported from AetherSpectra `engines/aether/weather/weather_effects.py`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeatherType {
    Clear,
    Overcast,
    Rainy,
    Stormy,
    Foggy,
    Snowy,
    Drizzle,
    Misty,
    Windy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindPreset {
    Calm,
    GentleBreeze,
    Moderate,
    Strong,
    Stormy,
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleConfig {
    pub count: u32,
    pub lifetime: f32,
    pub emission_rate: f32,
    pub emission_shape: String,
    pub emission_size: [f32; 3],
    pub emission_offset: [f32; 3],
    pub velocity_min: [f32; 3],
    pub velocity_max: [f32; 3],
    pub gravity: f32,
    pub drag: f32,
    pub turbulence: f32,
    pub size: f32,
    pub size_random: f32,
    pub material: String,
    pub color: [f32; 4],
    pub motion_blur_factor: f32,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            count: 1000,
            lifetime: 2.0,
            emission_rate: 500.0,
            emission_shape: "box".to_string(),
            emission_size: [50.0, 50.0, 20.0],
            emission_offset: [0.0, 0.0, 10.0],
            velocity_min: [0.0, 0.0, -5.0],
            velocity_max: [0.0, 0.0, -10.0],
            gravity: 9.81,
            drag: 0.1,
            turbulence: 0.0,
            size: 0.02,
            size_random: 0.5,
            material: "rain_drop".to_string(),
            color: [0.8, 0.9, 1.0, 0.5],
            motion_blur_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FogConfig {
    pub density: f32,
    /// Fog height falloff.
    pub height: f32,
    /// Exponential falloff rate.
    pub falloff: f32,
    pub color: [f32; 3],
    /// Light scattering direction (−1 to 1).
    pub anisotropy: f32,
    pub volume_samples: u32,
    pub volume_step_rate: f32,
}

impl Default for FogConfig {
    fn default() -> Self {
        Self {
            density: 0.1,
            height: 10.0,
            falloff: 2.0,
            color: [0.8, 0.85, 0.9],
            anisotropy: 0.0,
            volume_samples: 64,
            volume_step_rate: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeatherEffect {
    pub weather_type: WeatherType,
    pub wind_preset: WindPreset,
    /// Weather DNA (wetness, fog_density, precipitation, wind_strength).
    pub weather_dna: [f32; 4],
    pub particles: Vec<ParticleConfig>,
    pub fog: Option<FogConfig>,
    pub wind_direction: [f32; 3],
    pub wind_strength: f32,
    pub wind_turbulence: f32,
    pub gust_frequency: f32,
    pub gust_strength: f32,
    pub bloom_intensity: f32,
    pub chromatic_aberration: f32,
    pub ambient_sound: String,
    pub ambient_volume: f32,
}

// ---------------------------------------------------------------------------
// WeatherEffectsLibrary
// ---------------------------------------------------------------------------

/// Library of pre-configured weather effects, calibrated for production with
/// appropriate particle counts, fog density, and wind parameters.
pub struct WeatherEffectsLibrary;

fn wind_strength_clear(preset: &WindPreset) -> f32 {
    match preset {
        WindPreset::Calm => 0.1,
        WindPreset::GentleBreeze => 0.25,
        WindPreset::Moderate => 0.5,
        WindPreset::Strong => 0.75,
        WindPreset::Stormy => 1.0,
    }
}

impl WeatherEffectsLibrary {
    /// Clear sky with optional breeze.
    pub fn get_clear(wind_preset: WindPreset) -> WeatherEffect {
        let ws = wind_strength_clear(&wind_preset);
        WeatherEffect {
            weather_type: WeatherType::Clear,
            wind_preset,
            weather_dna: [0.0, 0.0, 0.0, ws],
            particles: vec![],
            fog: None,
            wind_direction: [1.0, 0.2, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.1,
            gust_frequency: 0.3,
            gust_strength: ws * 0.2,
            bloom_intensity: 0.0,
            chromatic_aberration: 0.0,
            ambient_sound: String::new(),
            ambient_volume: 0.0,
        }
    }

    /// Overcast sky, no precipitation.
    pub fn get_overcast(wind_preset: WindPreset) -> WeatherEffect {
        let ws = match &wind_preset {
            WindPreset::Calm => 0.1,
            WindPreset::GentleBreeze => 0.15,
            WindPreset::Moderate => 0.35,
            WindPreset::Strong => 0.6,
            WindPreset::Stormy => 0.8,
        };
        WeatherEffect {
            weather_type: WeatherType::Overcast,
            wind_preset,
            weather_dna: [0.1, 0.2, 0.0, ws],
            particles: vec![],
            fog: Some(FogConfig {
                density: 0.02,
                height: 50.0,
                falloff: 1.0,
                color: [0.7, 0.75, 0.8],
                ..Default::default()
            }),
            wind_direction: [1.0, 0.1, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.15,
            gust_frequency: 0.4,
            gust_strength: ws * 0.25,
            bloom_intensity: 0.0,
            chromatic_aberration: 0.0,
            ambient_sound: "wind_light".to_string(),
            ambient_volume: 0.2,
        }
    }

    /// Moderate rain with wind.
    pub fn get_rainy(wind_preset: WindPreset) -> WeatherEffect {
        let ws = f32::max(
            0.3,
            match &wind_preset {
                WindPreset::Calm => 0.2,
                WindPreset::GentleBreeze => 0.3,
                WindPreset::Moderate => 0.5,
                WindPreset::Strong => 0.7,
                WindPreset::Stormy => 0.9,
            },
        );

        let rain_particles = ParticleConfig {
            count: 5000,
            lifetime: 1.5,
            emission_rate: 3000.0,
            emission_shape: "box".to_string(),
            emission_size: [60.0, 60.0, 5.0],
            emission_offset: [0.0, 0.0, 25.0],
            velocity_min: [-ws * 2.0, 0.0, -12.0],
            velocity_max: [ws * 2.0, 0.5, -18.0],
            gravity: 9.81,
            drag: 0.05,
            turbulence: ws * 0.2,
            size: 0.02,
            size_random: 0.3,
            material: "rain_drop".to_string(),
            color: [0.8, 0.9, 1.0, 0.4],
            motion_blur_factor: 1.5,
        };

        let splash_particles = ParticleConfig {
            count: 2000,
            lifetime: 0.3,
            emission_rate: 1500.0,
            emission_shape: "plane".to_string(),
            emission_size: [60.0, 60.0, 0.1],
            emission_offset: [0.0, 0.0, 0.05],
            velocity_min: [-1.0, -1.0, 0.5],
            velocity_max: [1.0, 1.0, 1.5],
            gravity: 9.81,
            drag: 0.3,
            turbulence: 0.1,
            size: 0.01,
            size_random: 0.5,
            material: "rain_splash".to_string(),
            color: [0.9, 0.95, 1.0, 0.3],
            motion_blur_factor: 0.5,
        };

        WeatherEffect {
            weather_type: WeatherType::Rainy,
            wind_preset,
            weather_dna: [0.7, 0.3, 0.6, ws],
            particles: vec![rain_particles, splash_particles],
            fog: Some(FogConfig {
                density: 0.04,
                height: 30.0,
                falloff: 1.5,
                color: [0.6, 0.65, 0.7],
                ..Default::default()
            }),
            wind_direction: [1.0, 0.0, -0.1],
            wind_strength: ws,
            wind_turbulence: 0.25,
            gust_frequency: 0.6,
            gust_strength: ws * 0.3,
            bloom_intensity: 0.1,
            chromatic_aberration: 0.002,
            ambient_sound: "rain_moderate".to_string(),
            ambient_volume: 0.7,
        }
    }

    /// Heavy storm with intense rain and wind.
    pub fn get_stormy(wind_preset: WindPreset) -> WeatherEffect {
        let ws = f32::max(
            0.8,
            match &wind_preset {
                WindPreset::Calm => 0.6,
                WindPreset::GentleBreeze => 0.7,
                WindPreset::Moderate => 0.8,
                WindPreset::Strong => 0.9,
                WindPreset::Stormy => 1.0,
            },
        );

        let heavy_rain = ParticleConfig {
            count: 10000,
            lifetime: 1.0,
            emission_rate: 8000.0,
            emission_shape: "box".to_string(),
            emission_size: [80.0, 80.0, 8.0],
            emission_offset: [0.0, 0.0, 30.0],
            velocity_min: [-ws * 5.0, -1.0, -20.0],
            velocity_max: [ws * 3.0, 1.0, -28.0],
            gravity: 10.0,
            drag: 0.03,
            turbulence: ws * 0.4,
            size: 0.025,
            size_random: 0.4,
            material: "rain_drop_storm".to_string(),
            color: [0.7, 0.8, 0.95, 0.5],
            motion_blur_factor: 2.0,
        };

        let spray = ParticleConfig {
            count: 3000,
            lifetime: 2.0,
            emission_rate: 1500.0,
            emission_shape: "box".to_string(),
            emission_size: [80.0, 80.0, 5.0],
            emission_offset: [0.0, 0.0, 2.0],
            velocity_min: [ws * 3.0, -0.5, 0.5],
            velocity_max: [ws * 6.0, 0.5, 2.0],
            gravity: 0.5,
            drag: 0.2,
            turbulence: 0.5,
            size: 0.05,
            size_random: 0.6,
            material: "mist_spray".to_string(),
            color: [0.8, 0.85, 0.9, 0.2],
            motion_blur_factor: 0.3,
        };

        let splash_storm = ParticleConfig {
            count: 4000,
            lifetime: 0.4,
            emission_rate: 3000.0,
            emission_shape: "plane".to_string(),
            emission_size: [80.0, 80.0, 0.1],
            emission_offset: [0.0, 0.0, 0.05],
            velocity_min: [-2.0, -2.0, 1.0],
            velocity_max: [2.0, 2.0, 3.0],
            gravity: 9.81,
            drag: 0.2,
            turbulence: 0.2,
            size: 0.015,
            size_random: 0.5,
            material: "rain_splash_storm".to_string(),
            color: [0.85, 0.9, 1.0, 0.4],
            motion_blur_factor: 0.8,
        };

        WeatherEffect {
            weather_type: WeatherType::Stormy,
            wind_preset,
            weather_dna: [0.9, 0.4, 0.9, ws],
            particles: vec![heavy_rain, spray, splash_storm],
            fog: Some(FogConfig {
                density: 0.08,
                height: 20.0,
                falloff: 2.0,
                color: [0.5, 0.55, 0.6],
                ..Default::default()
            }),
            wind_direction: [1.0, 0.15, -0.15],
            wind_strength: ws,
            wind_turbulence: 0.5,
            gust_frequency: 1.0,
            gust_strength: ws * 0.5,
            bloom_intensity: 0.15,
            chromatic_aberration: 0.003,
            ambient_sound: "storm_heavy".to_string(),
            ambient_volume: 1.0,
        }
    }

    /// Heavy fog with reduced visibility.
    pub fn get_foggy(wind_preset: WindPreset) -> WeatherEffect {
        let ws = f32::min(
            0.15,
            match &wind_preset {
                WindPreset::Calm => 0.05,
                WindPreset::GentleBreeze => 0.1,
                WindPreset::Moderate => 0.15,
                WindPreset::Strong => 0.2,
                WindPreset::Stormy => 0.25,
            },
        );

        let fog_drift = ParticleConfig {
            count: 500,
            lifetime: 10.0,
            emission_rate: 50.0,
            emission_shape: "box".to_string(),
            emission_size: [100.0, 100.0, 15.0],
            emission_offset: [0.0, 0.0, 5.0],
            velocity_min: [ws * 0.5, -0.1, -0.1],
            velocity_max: [ws * 1.5, 0.1, 0.1],
            gravity: 0.0,
            drag: 0.1,
            turbulence: 0.1,
            size: 2.0,
            size_random: 0.8,
            material: "fog_wisp".to_string(),
            color: [0.85, 0.88, 0.92, 0.1],
            motion_blur_factor: 0.1,
        };

        WeatherEffect {
            weather_type: WeatherType::Foggy,
            wind_preset,
            weather_dna: [0.3, 0.8, 0.0, ws],
            particles: vec![fog_drift],
            fog: Some(FogConfig {
                density: 0.15,
                height: 15.0,
                falloff: 3.0,
                color: [0.85, 0.88, 0.92],
                anisotropy: 0.2,
                volume_samples: 128,
                volume_step_rate: 1.0,
            }),
            wind_direction: [1.0, 0.0, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.05,
            gust_frequency: 0.1,
            gust_strength: ws * 0.1,
            bloom_intensity: 0.2,
            chromatic_aberration: 0.001,
            ambient_sound: "fog_ambient".to_string(),
            ambient_volume: 0.3,
        }
    }

    /// Snowfall with light accumulation.
    pub fn get_snowy(wind_preset: WindPreset) -> WeatherEffect {
        let ws = match &wind_preset {
            WindPreset::Calm => 0.1,
            WindPreset::GentleBreeze => 0.2,
            WindPreset::Moderate => 0.4,
            WindPreset::Strong => 0.6,
            WindPreset::Stormy => 0.8,
        };

        let snow = ParticleConfig {
            count: 3000,
            lifetime: 5.0,
            emission_rate: 600.0,
            emission_shape: "box".to_string(),
            emission_size: [60.0, 60.0, 10.0],
            emission_offset: [0.0, 0.0, 25.0],
            velocity_min: [-ws * 2.0, -0.5, -1.5],
            velocity_max: [ws * 2.0, 0.5, -3.0],
            gravity: 1.0,
            drag: 0.4,
            turbulence: ws * 0.3,
            size: 0.03,
            size_random: 0.6,
            material: "snowflake".to_string(),
            color: [1.0, 1.0, 1.0, 0.8],
            motion_blur_factor: 0.3,
        };

        WeatherEffect {
            weather_type: WeatherType::Snowy,
            wind_preset,
            weather_dna: [0.2, 0.2, 0.8, ws],
            particles: vec![snow],
            fog: Some(FogConfig {
                density: 0.03,
                height: 40.0,
                falloff: 1.5,
                color: [0.9, 0.92, 0.95],
                ..Default::default()
            }),
            wind_direction: [1.0, 0.1, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.2,
            gust_frequency: 0.3,
            gust_strength: ws * 0.2,
            bloom_intensity: 0.1,
            chromatic_aberration: 0.0,
            ambient_sound: "snow_wind".to_string(),
            ambient_volume: 0.4,
        }
    }

    /// Light drizzle with minimal wind.
    pub fn get_drizzle(wind_preset: WindPreset) -> WeatherEffect {
        let ws = f32::min(
            0.35,
            match &wind_preset {
                WindPreset::Calm => 0.1,
                WindPreset::GentleBreeze => 0.2,
                WindPreset::Moderate => 0.35,
                WindPreset::Strong => 0.5,
                WindPreset::Stormy => 0.6,
            },
        );

        let drizzle = ParticleConfig {
            count: 2000,
            lifetime: 2.0,
            emission_rate: 1000.0,
            emission_shape: "box".to_string(),
            emission_size: [50.0, 50.0, 5.0],
            emission_offset: [0.0, 0.0, 20.0],
            velocity_min: [-ws, -0.2, -6.0],
            velocity_max: [ws, 0.2, -10.0],
            gravity: 8.0,
            drag: 0.08,
            turbulence: 0.1,
            size: 0.015,
            size_random: 0.3,
            material: "rain_drop_light".to_string(),
            color: [0.85, 0.92, 1.0, 0.3],
            motion_blur_factor: 1.0,
        };

        WeatherEffect {
            weather_type: WeatherType::Drizzle,
            wind_preset,
            weather_dna: [0.4, 0.4, 0.3, ws],
            particles: vec![drizzle],
            fog: Some(FogConfig {
                density: 0.03,
                height: 35.0,
                falloff: 1.8,
                color: [0.7, 0.75, 0.8],
                ..Default::default()
            }),
            wind_direction: [1.0, 0.05, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.15,
            gust_frequency: 0.35,
            gust_strength: ws * 0.2,
            bloom_intensity: 0.05,
            chromatic_aberration: 0.0,
            ambient_sound: "rain_light".to_string(),
            ambient_volume: 0.4,
        }
    }

    /// Light mist with soft atmosphere.
    pub fn get_misty(wind_preset: WindPreset) -> WeatherEffect {
        let ws = f32::min(
            0.1,
            match &wind_preset {
                WindPreset::Calm => 0.03,
                WindPreset::GentleBreeze => 0.08,
                WindPreset::Moderate => 0.1,
                WindPreset::Strong => 0.15,
                WindPreset::Stormy => 0.2,
            },
        );

        WeatherEffect {
            weather_type: WeatherType::Misty,
            wind_preset,
            weather_dna: [0.2, 0.5, 0.0, ws],
            particles: vec![],
            fog: Some(FogConfig {
                density: 0.06,
                height: 25.0,
                falloff: 2.0,
                color: [0.82, 0.85, 0.9],
                anisotropy: 0.1,
                ..Default::default()
            }),
            wind_direction: [1.0, 0.0, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.03,
            gust_frequency: 0.1,
            gust_strength: 0.01,
            bloom_intensity: 0.15,
            chromatic_aberration: 0.0,
            ambient_sound: "mist_ambient".to_string(),
            ambient_volume: 0.2,
        }
    }

    /// Strong wind with debris/leaves.
    pub fn get_windy(wind_preset: WindPreset) -> WeatherEffect {
        let ws = f32::max(
            0.5,
            match &wind_preset {
                WindPreset::Calm => 0.3,
                WindPreset::GentleBreeze => 0.4,
                WindPreset::Moderate => 0.6,
                WindPreset::Strong => 0.8,
                WindPreset::Stormy => 1.0,
            },
        );

        let debris = ParticleConfig {
            count: 500,
            lifetime: 4.0,
            emission_rate: 125.0,
            emission_shape: "box".to_string(),
            emission_size: [80.0, 80.0, 5.0],
            emission_offset: [-40.0, 0.0, 2.0],
            velocity_min: [ws * 8.0, -2.0, -0.5],
            velocity_max: [ws * 15.0, 2.0, 2.0],
            gravity: 0.5,
            drag: 0.15,
            turbulence: ws * 0.5,
            size: 0.05,
            size_random: 0.7,
            material: "leaf_debris".to_string(),
            color: [0.4, 0.35, 0.2, 1.0],
            motion_blur_factor: 0.8,
        };

        let dust = ParticleConfig {
            count: 1000,
            lifetime: 3.0,
            emission_rate: 333.0,
            emission_shape: "box".to_string(),
            emission_size: [80.0, 80.0, 3.0],
            emission_offset: [0.0, 0.0, 0.5],
            velocity_min: [ws * 3.0, -1.0, 0.2],
            velocity_max: [ws * 8.0, 1.0, 1.5],
            gravity: 0.1,
            drag: 0.2,
            turbulence: ws * 0.4,
            size: 0.02,
            size_random: 0.5,
            material: "dust".to_string(),
            color: [0.6, 0.55, 0.4, 0.3],
            motion_blur_factor: 0.5,
        };

        WeatherEffect {
            weather_type: WeatherType::Windy,
            wind_preset,
            weather_dna: [0.0, 0.0, 0.0, ws],
            particles: vec![debris, dust],
            fog: Some(FogConfig {
                density: 0.01,
                height: 50.0,
                falloff: 1.0,
                color: [0.75, 0.72, 0.65],
                ..Default::default()
            }),
            wind_direction: [1.0, 0.2, 0.0],
            wind_strength: ws,
            wind_turbulence: 0.4,
            gust_frequency: 0.8,
            gust_strength: ws * 0.4,
            bloom_intensity: 0.05,
            chromatic_aberration: 0.0,
            ambient_sound: "wind_strong".to_string(),
            ambient_volume: 0.8,
        }
    }

    /// Get weather effect by type and wind preset.
    pub fn get_effect(weather_type: WeatherType, wind_preset: WindPreset) -> WeatherEffect {
        match weather_type {
            WeatherType::Clear => Self::get_clear(wind_preset),
            WeatherType::Overcast => Self::get_overcast(wind_preset),
            WeatherType::Rainy => Self::get_rainy(wind_preset),
            WeatherType::Stormy => Self::get_stormy(wind_preset),
            WeatherType::Foggy => Self::get_foggy(wind_preset),
            WeatherType::Snowy => Self::get_snowy(wind_preset),
            WeatherType::Drizzle => Self::get_drizzle(wind_preset),
            WeatherType::Misty => Self::get_misty(wind_preset),
            WeatherType::Windy => Self::get_windy(wind_preset),
        }
    }

    /// Blend between two weather effects for smooth transitions.
    ///
    /// `factor` = 0.0 → pure `a`, 1.0 → pure `b`.
    pub fn blend_effects(a: &WeatherEffect, b: &WeatherEffect, factor: f32) -> WeatherEffect {
        let factor = factor.clamp(0.0, 1.0);
        let inv = 1.0 - factor;

        let blended_dna = [
            a.weather_dna[0] * inv + b.weather_dna[0] * factor,
            a.weather_dna[1] * inv + b.weather_dna[1] * factor,
            a.weather_dna[2] * inv + b.weather_dna[2] * factor,
            a.weather_dna[3] * inv + b.weather_dna[3] * factor,
        ];

        let dominant = if factor >= 0.5 { b } else { a };

        let wind_dir_raw = [
            a.wind_direction[0] * inv + b.wind_direction[0] * factor,
            a.wind_direction[1] * inv + b.wind_direction[1] * factor,
            a.wind_direction[2] * inv + b.wind_direction[2] * factor,
        ];
        let mag = (wind_dir_raw[0] * wind_dir_raw[0]
            + wind_dir_raw[1] * wind_dir_raw[1]
            + wind_dir_raw[2] * wind_dir_raw[2])
            .sqrt();
        let wind_direction = if mag > 0.0 {
            [
                wind_dir_raw[0] / mag,
                wind_dir_raw[1] / mag,
                wind_dir_raw[2] / mag,
            ]
        } else {
            wind_dir_raw
        };

        let blended_fog = match (&a.fog, &b.fog) {
            (Some(fa), Some(fb)) => Some(FogConfig {
                density: fa.density * inv + fb.density * factor,
                height: fa.height * inv + fb.height * factor,
                falloff: fa.falloff * inv + fb.falloff * factor,
                color: [
                    fa.color[0] * inv + fb.color[0] * factor,
                    fa.color[1] * inv + fb.color[1] * factor,
                    fa.color[2] * inv + fb.color[2] * factor,
                ],
                anisotropy: fa.anisotropy * inv + fb.anisotropy * factor,
                volume_samples: fb.volume_samples,
                volume_step_rate: fa.volume_step_rate * inv + fb.volume_step_rate * factor,
            }),
            (None, Some(fb)) => Some(FogConfig {
                density: fb.density * factor,
                height: fb.height,
                falloff: fb.falloff,
                color: fb.color,
                anisotropy: fb.anisotropy,
                volume_samples: fb.volume_samples,
                volume_step_rate: fb.volume_step_rate,
            }),
            (Some(fa), None) => Some(FogConfig {
                density: fa.density * inv,
                height: fa.height,
                falloff: fa.falloff,
                color: fa.color,
                anisotropy: fa.anisotropy,
                volume_samples: fa.volume_samples,
                volume_step_rate: fa.volume_step_rate,
            }),
            (None, None) => None,
        };

        WeatherEffect {
            weather_type: dominant.weather_type.clone(),
            wind_preset: dominant.wind_preset.clone(),
            weather_dna: blended_dna,
            particles: dominant.particles.clone(),
            fog: blended_fog,
            wind_direction,
            wind_strength: a.wind_strength * inv + b.wind_strength * factor,
            wind_turbulence: a.wind_turbulence * inv + b.wind_turbulence * factor,
            gust_frequency: a.gust_frequency * inv + b.gust_frequency * factor,
            gust_strength: a.gust_strength * inv + b.gust_strength * factor,
            bloom_intensity: a.bloom_intensity * inv + b.bloom_intensity * factor,
            chromatic_aberration: a.chromatic_aberration * inv + b.chromatic_aberration * factor,
            ambient_sound: dominant.ambient_sound.clone(),
            ambient_volume: a.ambient_volume * inv + b.ambient_volume * factor,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rainy_weather_dna() {
        let effect = WeatherEffectsLibrary::get_rainy(WindPreset::Moderate);
        // weather_dna = (0.7, 0.3, 0.6, wind_strength)
        assert_eq!(effect.weather_dna[0], 0.7);
        assert_eq!(effect.weather_dna[1], 0.3);
        assert_eq!(effect.weather_dna[2], 0.6);
    }

    #[test]
    fn stormy_rain_particle_count() {
        let effect = WeatherEffectsLibrary::get_stormy(WindPreset::Stormy);
        // heavy_rain count = 10000
        assert_eq!(effect.particles[0].count, 10000);
    }

    #[test]
    fn foggy_fog_density() {
        let effect = WeatherEffectsLibrary::get_foggy(WindPreset::Calm);
        assert_eq!(effect.fog.as_ref().unwrap().density, 0.15);
        assert_eq!(effect.fog.as_ref().unwrap().volume_samples, 128);
    }

    #[test]
    fn blend_effects_fog_interpolation() {
        let a = WeatherEffectsLibrary::get_clear(WindPreset::Calm);
        let b = WeatherEffectsLibrary::get_foggy(WindPreset::Calm);
        let blended = WeatherEffectsLibrary::blend_effects(&a, &b, 1.0);
        // At factor=1.0 blended fog density = b's density
        assert!((blended.fog.as_ref().unwrap().density - 0.15).abs() < 1e-5);
    }
}
