//! Atmospheric effects: fog, volumetric clouds, particles, god rays, depth haze.
//!
//! Ported from AetherSpectra `engines/aether/weather/atmosphere_generator.py`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleType {
    Dust,
    Rain,
    Snow,
    Leaves,
    Fireflies,
    Embers,
    Pollen,
    Ash,
    Bubbles,
    Sparks,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FogType {
    /// Even density throughout.
    Uniform,
    /// Hugging the ground.
    Ground,
    /// Horizontal layers.
    Layered,
    /// Full 3-D volumetric.
    Volumetric,
    /// Depth-based haze.
    Distance,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudType {
    /// Puffy white clouds.
    Cumulus,
    /// Flat layered clouds.
    Stratus,
    /// Wispy high-altitude.
    Cirrus,
    /// Storm clouds.
    Cumulonimbus,
    /// Rain clouds.
    Nimbostratus,
    /// Mid-level puffy.
    Altocumulus,
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FogSettings {
    pub fog_type: FogType,
    pub density: f32,
    pub start_distance: f32,
    pub end_distance: f32,
    /// For ground fog.
    pub height: f32,
    pub falloff: f32,
    pub color: [f32; 3],
    pub scattering: f32,
    pub absorption: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            fog_type: FogType::Uniform,
            density: 0.01,
            start_distance: 5.0,
            end_distance: 100.0,
            height: 2.0,
            falloff: 1.0,
            color: [0.8, 0.85, 0.9],
            scattering: 0.5,
            absorption: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumetricCloudSettings {
    pub cloud_type: CloudType,
    /// Cloud base height in metres.
    pub altitude: f32,
    /// Vertical thickness.
    pub thickness: f32,
    /// 0–1 sky coverage.
    pub coverage: f32,
    pub density: f32,
    pub scale: f32,
    pub detail_scale: f32,
    pub color: [f32; 3],
    pub absorption: f32,
    pub wind_offset: [f32; 3],
}

impl Default for VolumetricCloudSettings {
    fn default() -> Self {
        Self {
            cloud_type: CloudType::Cumulus,
            altitude: 500.0,
            thickness: 200.0,
            coverage: 0.5,
            density: 0.02,
            scale: 500.0,
            detail_scale: 50.0,
            color: [1.0, 1.0, 1.0],
            absorption: 0.1,
            wind_offset: [0.0, 0.0, 0.0],
        }
    }
}

/// Flattened cloud preset data (from Python CLOUD_PRESETS dict).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CloudPreset {
    pub altitude: f32,
    pub thickness: f32,
    pub coverage: f32,
    pub density: f32,
    pub scale: f32,
    pub absorption: f32,
}

/// Return the cloud preset for the given type.
pub fn cloud_preset(t: CloudType) -> CloudPreset {
    match t {
        CloudType::Cumulus => CloudPreset {
            altitude: 500.0,
            thickness: 200.0,
            coverage: 0.4,
            density: 0.015,
            scale: 400.0,
            absorption: 0.1,
        },
        CloudType::Stratus => CloudPreset {
            altitude: 300.0,
            thickness: 400.0,
            coverage: 0.9,
            density: 0.025,
            scale: 600.0,
            absorption: 0.15,
        },
        CloudType::Cirrus => CloudPreset {
            altitude: 800.0,
            thickness: 100.0,
            coverage: 0.3,
            density: 0.008,
            scale: 800.0,
            absorption: 0.05,
        },
        CloudType::Cumulonimbus => CloudPreset {
            altitude: 200.0,
            thickness: 500.0,
            coverage: 0.85,
            density: 0.04,
            scale: 250.0,
            absorption: 0.3,
        },
        CloudType::Nimbostratus => CloudPreset {
            altitude: 150.0,
            thickness: 350.0,
            coverage: 0.95,
            density: 0.035,
            scale: 300.0,
            absorption: 0.25,
        },
        CloudType::Altocumulus => CloudPreset {
            altitude: 400.0,
            thickness: 150.0,
            coverage: 0.6,
            density: 0.018,
            scale: 350.0,
            absorption: 0.12,
        },
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleSystem {
    pub name: String,
    pub particle_type: ParticleType,
    pub count: u32,
    pub size: f32,
    pub size_variation: f32,
    pub lifetime: f32,
    pub velocity: [f32; 3],
    pub velocity_variation: f32,
    pub gravity: f32,
    pub emission_area: [f32; 3],
    pub emission_position: [f32; 3],
    pub color: [f32; 3],
    pub color_variation: f32,
    pub opacity: f32,
    /// For glowing particles.
    pub emission: f32,
    pub rotation_speed: f32,
    pub turbulence: f32,
}

/// Flattened particle preset data (from Python PARTICLE_PRESETS dict).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticlePreset {
    pub count: u32,
    pub size: f32,
    pub lifetime: f32,
    pub velocity: [f32; 3],
    pub gravity: f32,
    pub color: [f32; 3],
    pub opacity: f32,
    pub emission: f32,
    pub rotation_speed: f32,
    pub turbulence: f32,
}

/// Return the particle preset for the given type.
pub fn particle_preset(t: ParticleType) -> ParticlePreset {
    match t {
        ParticleType::Dust => ParticlePreset {
            count: 500,
            size: 0.005,
            lifetime: 8.0,
            velocity: [0.0, 0.01, 0.0],
            gravity: -0.001,
            color: [0.9, 0.85, 0.8],
            opacity: 0.3,
            emission: 0.0,
            rotation_speed: 0.0,
            turbulence: 0.2,
        },
        ParticleType::Rain => ParticlePreset {
            count: 5000,
            size: 0.02,
            lifetime: 2.0,
            velocity: [0.0, -5.0, 0.0],
            gravity: 2.0,
            color: [0.7, 0.75, 0.8],
            opacity: 0.5,
            emission: 0.0,
            rotation_speed: 0.0,
            turbulence: 0.05,
        },
        ParticleType::Snow => ParticlePreset {
            count: 2000,
            size: 0.015,
            lifetime: 10.0,
            velocity: [0.0, -0.3, 0.0],
            gravity: 0.1,
            color: [1.0, 1.0, 1.0],
            opacity: 0.8,
            emission: 0.0,
            rotation_speed: 0.5,
            turbulence: 0.3,
        },
        ParticleType::Leaves => ParticlePreset {
            count: 100,
            size: 0.05,
            lifetime: 15.0,
            velocity: [0.5, -0.2, 0.0],
            gravity: 0.05,
            color: [0.6, 0.5, 0.2],
            opacity: 1.0,
            emission: 0.0,
            rotation_speed: 1.0,
            turbulence: 0.4,
        },
        ParticleType::Fireflies => ParticlePreset {
            count: 50,
            size: 0.01,
            lifetime: 5.0,
            velocity: [0.0, 0.1, 0.0],
            gravity: 0.0,
            color: [1.0, 0.9, 0.3],
            opacity: 0.9,
            emission: 2.0,
            rotation_speed: 0.0,
            turbulence: 0.5,
        },
        ParticleType::Embers => ParticlePreset {
            count: 200,
            size: 0.008,
            lifetime: 4.0,
            velocity: [0.0, 0.5, 0.0],
            gravity: -0.1,
            color: [1.0, 0.5, 0.1],
            opacity: 0.8,
            emission: 3.0,
            rotation_speed: 0.0,
            turbulence: 0.3,
        },
        ParticleType::Pollen => ParticlePreset {
            count: 300,
            size: 0.003,
            lifetime: 12.0,
            velocity: [0.1, 0.02, 0.0],
            gravity: -0.005,
            color: [1.0, 0.95, 0.7],
            opacity: 0.4,
            emission: 0.0,
            rotation_speed: 0.0,
            turbulence: 0.4,
        },
        ParticleType::Ash => ParticlePreset {
            count: 400,
            size: 0.01,
            lifetime: 8.0,
            velocity: [0.2, -0.1, 0.0],
            gravity: 0.02,
            color: [0.3, 0.3, 0.3],
            opacity: 0.6,
            emission: 0.0,
            rotation_speed: 0.0,
            turbulence: 0.25,
        },
        ParticleType::Bubbles => ParticlePreset {
            count: 200,
            size: 0.02,
            lifetime: 6.0,
            velocity: [0.0, 0.2, 0.0],
            gravity: -0.05,
            color: [0.9, 0.95, 1.0],
            opacity: 0.4,
            emission: 0.0,
            rotation_speed: 0.0,
            turbulence: 0.15,
        },
        ParticleType::Sparks => ParticlePreset {
            count: 100,
            size: 0.005,
            lifetime: 1.5,
            velocity: [0.0, 2.0, 0.0],
            gravity: 1.5,
            color: [1.0, 0.7, 0.2],
            opacity: 1.0,
            emission: 5.0,
            rotation_speed: 0.0,
            turbulence: 0.1,
        },
    }
}

/// Flattened fog preset data (from Python FOG_PRESETS dict).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FogPreset {
    pub fog_type: FogType,
    pub density: f32,
    /// Height override (only for ground/layered fog presets).
    pub height: Option<f32>,
    pub color: [f32; 3],
}

/// Return a named fog preset.
///
/// Known keys: `"light_mist"`, `"morning_fog"`, `"dense_fog"`,
/// `"forest_haze"`, `"smoke"`. Returns `None` for unknown names.
pub fn fog_preset(name: &str) -> Option<FogPreset> {
    match name {
        "light_mist" => Some(FogPreset {
            fog_type: FogType::Uniform,
            density: 0.005,
            height: None,
            color: [0.9, 0.92, 0.95],
        }),
        "morning_fog" => Some(FogPreset {
            fog_type: FogType::Ground,
            density: 0.02,
            height: Some(1.5),
            color: [0.95, 0.95, 1.0],
        }),
        "dense_fog" => Some(FogPreset {
            fog_type: FogType::Volumetric,
            density: 0.05,
            height: None,
            color: [0.85, 0.88, 0.9],
        }),
        "forest_haze" => Some(FogPreset {
            fog_type: FogType::Layered,
            density: 0.015,
            height: Some(3.0),
            color: [0.8, 0.85, 0.75],
        }),
        "smoke" => Some(FogPreset {
            fog_type: FogType::Volumetric,
            density: 0.03,
            height: None,
            color: [0.4, 0.4, 0.45],
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GodRaySettings {
    pub enabled: bool,
    pub intensity: f32,
    pub decay: f32,
    pub samples: u32,
    pub density: f32,
    pub weight: f32,
    /// Which light source creates the rays.
    pub light_source: String,
}

impl Default for GodRaySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.5,
            decay: 0.95,
            samples: 50,
            density: 1.0,
            weight: 0.5,
            light_source: "sun".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthHazeSettings {
    pub enabled: bool,
    pub start_distance: f32,
    pub intensity: f32,
    pub color: [f32; 3],
    pub height_falloff: f32,
}

impl Default for DepthHazeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            start_distance: 20.0,
            intensity: 0.3,
            color: [0.7, 0.8, 0.9],
            height_falloff: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtmosphereConfig {
    pub name: String,
    pub description: String,
    pub fog: Option<FogSettings>,
    pub particles: Vec<ParticleSystem>,
    pub god_rays: GodRaySettings,
    pub depth_haze: DepthHazeSettings,
    pub ambient_sound_hints: Vec<String>,
}

// ---------------------------------------------------------------------------
// AtmosphereGenerator
// ---------------------------------------------------------------------------

/// Generator for atmospheric effects in 3-D environments.
///
/// Creates fog, particles, and other atmospheric elements based on
/// environment type, weather condition, and mood string inputs.
pub struct AtmosphereGenerator;

impl AtmosphereGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Create atmospheric effects for an environment.
    pub fn create_atmosphere(
        &self,
        environment_type: &str,
        weather: &str,
        time_of_day: &str,
        mood: &str,
        include_particles: bool,
    ) -> AtmosphereConfig {
        let fog = self.create_fog_for_weather(weather, environment_type);
        let particles = if include_particles {
            self.create_particles(environment_type, weather, time_of_day)
        } else {
            vec![]
        };
        let god_rays = self.create_god_rays(weather, time_of_day, environment_type);
        let depth_haze = self.create_depth_haze(weather, environment_type);
        let ambient_sounds = self.suggest_ambient_sounds(environment_type, weather, time_of_day);

        AtmosphereConfig {
            name: format!("{environment_type}_{weather}_{time_of_day}"),
            description: format!(
                "Atmospheric effects for {environment_type} in {weather} {time_of_day} ({mood})"
            ),
            fog,
            particles,
            god_rays,
            depth_haze,
            ambient_sound_hints: ambient_sounds,
        }
    }

    /// Create a specific particle system from a preset.
    pub fn create_particle_system(
        &self,
        particle_type: ParticleType,
        intensity: f32,
        area_size: [f32; 3],
        position: [f32; 3],
    ) -> ParticleSystem {
        let preset = particle_preset(particle_type.clone());
        ParticleSystem {
            name: format!("{particle_type:?}_system").to_lowercase(),
            particle_type,
            count: (preset.count as f32 * intensity) as u32,
            size: preset.size,
            size_variation: 0.5,
            lifetime: preset.lifetime,
            velocity: preset.velocity,
            velocity_variation: 0.3,
            gravity: preset.gravity,
            emission_area: area_size,
            emission_position: position,
            color: preset.color,
            color_variation: 0.1,
            opacity: preset.opacity,
            emission: preset.emission,
            rotation_speed: preset.rotation_speed,
            turbulence: preset.turbulence,
        }
    }

    // Private helpers

    fn create_fog_for_weather(&self, weather: &str, environment_type: &str) -> Option<FogSettings> {
        let preset_name: Option<&str> = match weather {
            "foggy" => Some("dense_fog"),
            "rainy" | "overcast" | "stormy" => Some("light_mist"),
            _ => None,
        };

        let preset = if let Some(name) = preset_name {
            fog_preset(name)?
        } else if matches!(environment_type, "forest" | "swamp") {
            let mut p = fog_preset("forest_haze")?;
            p.density *= 0.5;
            p
        } else {
            return None;
        };

        Some(FogSettings {
            fog_type: preset.fog_type,
            density: preset.density,
            start_distance: 5.0,
            end_distance: 100.0,
            height: preset.height.unwrap_or(2.0),
            falloff: 1.0,
            color: preset.color,
            scattering: 0.5,
            absorption: 0.1,
        })
    }

    fn create_particles(
        &self,
        environment_type: &str,
        weather: &str,
        time_of_day: &str,
    ) -> Vec<ParticleSystem> {
        let mut particles = Vec::new();

        match weather {
            "rainy" => {
                particles.push(self.create_particle_system(
                    ParticleType::Rain,
                    1.0,
                    [20.0, 20.0, 20.0],
                    [0.0, 15.0, 0.0],
                ));
            }
            "snowy" => {
                particles.push(self.create_particle_system(
                    ParticleType::Snow,
                    1.0,
                    [20.0, 20.0, 20.0],
                    [0.0, 10.0, 0.0],
                ));
            }
            "stormy" => {
                particles.push(self.create_particle_system(
                    ParticleType::Rain,
                    2.0,
                    [20.0, 20.0, 20.0],
                    [0.0, 15.0, 0.0],
                ));
            }
            _ => {}
        }

        if matches!(environment_type, "forest" | "garden" | "meadow") {
            if matches!(time_of_day, "day" | "morning" | "afternoon") {
                particles.push(self.create_particle_system(
                    ParticleType::Dust,
                    0.5,
                    [10.0, 5.0, 10.0],
                    [0.0, 2.0, 0.0],
                ));
            }
            if matches!(time_of_day, "night" | "evening" | "dusk") {
                particles.push(self.create_particle_system(
                    ParticleType::Fireflies,
                    1.0,
                    [15.0, 3.0, 15.0],
                    [0.0, 1.5, 0.0],
                ));
            }
        }

        if matches!(environment_type, "autumn_forest" | "autumn") {
            particles.push(self.create_particle_system(
                ParticleType::Leaves,
                1.0,
                [15.0, 10.0, 15.0],
                [0.0, 8.0, 0.0],
            ));
        } else if matches!(environment_type, "volcanic" | "fire") {
            particles.push(self.create_particle_system(
                ParticleType::Ash,
                1.0,
                [20.0, 15.0, 20.0],
                [0.0, 10.0, 0.0],
            ));
            particles.push(self.create_particle_system(
                ParticleType::Embers,
                0.5,
                [5.0, 3.0, 5.0],
                [0.0, 1.0, 0.0],
            ));
        } else if matches!(environment_type, "interior" | "room") && weather == "clear" {
            particles.push(self.create_particle_system(
                ParticleType::Dust,
                0.3,
                [3.0, 3.0, 3.0],
                [2.0, 2.0, 0.0],
            ));
        }

        particles
    }

    fn create_god_rays(
        &self,
        weather: &str,
        time_of_day: &str,
        environment_type: &str,
    ) -> GodRaySettings {
        if !matches!(weather, "clear" | "partly_cloudy") {
            return GodRaySettings::default();
        }
        if !matches!(time_of_day, "dawn" | "morning" | "dusk" | "afternoon") {
            return GodRaySettings::default();
        }
        let intensity = if matches!(environment_type, "forest" | "interior") {
            0.6
        } else {
            0.3
        };
        GodRaySettings {
            enabled: true,
            intensity,
            decay: 0.95,
            samples: 50,
            density: 1.0,
            weight: 0.5,
            light_source: "sun".to_string(),
        }
    }

    fn create_depth_haze(&self, weather: &str, environment_type: &str) -> DepthHazeSettings {
        let mut haze_intensity = 0.2_f32;
        if matches!(weather, "foggy" | "overcast") {
            haze_intensity = 0.5;
        } else if matches!(weather, "rainy" | "stormy") {
            haze_intensity = 0.4;
        }
        if matches!(environment_type, "desert" | "beach") {
            haze_intensity += 0.1;
        }
        DepthHazeSettings {
            enabled: true,
            start_distance: 15.0,
            intensity: haze_intensity,
            color: [0.7, 0.8, 0.9],
            height_falloff: 0.1,
        }
    }

    fn suggest_ambient_sounds(
        &self,
        environment_type: &str,
        weather: &str,
        time_of_day: &str,
    ) -> Vec<String> {
        let mut sounds = Vec::new();
        match weather {
            "rainy" => sounds.push("rain_medium".to_string()),
            "stormy" => {
                sounds.push("rain_heavy".to_string());
                sounds.push("thunder_distant".to_string());
            }
            "snowy" => {
                sounds.push("wind_gentle".to_string());
                sounds.push("snow_crunch".to_string());
            }
            _ => {}
        }
        match environment_type {
            "forest" => {
                if matches!(time_of_day, "day" | "morning" | "afternoon") {
                    sounds.push("birds_forest".to_string());
                    sounds.push("wind_leaves".to_string());
                } else if matches!(time_of_day, "night" | "evening") {
                    sounds.push("crickets".to_string());
                    sounds.push("owl_distant".to_string());
                }
            }
            "urban" => {
                if matches!(time_of_day, "day" | "morning" | "afternoon") {
                    sounds.push("city_ambient".to_string());
                    sounds.push("traffic_distant".to_string());
                } else {
                    sounds.push("city_night".to_string());
                }
            }
            "beach" => {
                sounds.push("waves".to_string());
                sounds.push("seagulls".to_string());
            }
            "interior" => sounds.push("room_tone".to_string()),
            _ => {}
        }
        sounds
    }
}

impl Default for AtmosphereGenerator {
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
    fn cloud_preset_cumulus_values() {
        let p = cloud_preset(CloudType::Cumulus);
        assert_eq!(p.altitude, 500.0);
        assert_eq!(p.coverage, 0.4);
        assert_eq!(p.density, 0.015);
        assert_eq!(p.absorption, 0.1);
    }

    #[test]
    fn particle_preset_rain_count() {
        let p = particle_preset(ParticleType::Rain);
        assert_eq!(p.count, 5000);
        assert_eq!(p.gravity, 2.0);
    }

    #[test]
    fn fog_preset_dense_fog_density() {
        let p = fog_preset("dense_fog").expect("dense_fog must exist");
        assert_eq!(p.density, 0.05);
    }

    #[test]
    fn particle_preset_snow_opacity() {
        let p = particle_preset(ParticleType::Snow);
        assert_eq!(p.opacity, 0.8);
        assert_eq!(p.rotation_speed, 0.5);
    }
}
