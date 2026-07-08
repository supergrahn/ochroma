//! Seasonal parameters: atmospheric conditions, vegetation states, precipitation.
//!
//! Ported from AetherSpectra `engines/aether/weather/season_schema.py`.
//!
//! `Season` is defined here because the original Python imported it from
//! `motionfactory` which is not available in this codebase.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecipitationType {
    None,
    Rain,
    Snow,
    Sleet,
    Mist,
    Fog,
}

// ---------------------------------------------------------------------------
// SeasonalParameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalParameters {
    pub season: Season,

    // Sun parameters
    pub sun_angle_min: f32,
    pub sun_angle_max: f32,
    pub day_length_hours: f32,

    // Sky colors
    pub sky_color_day: [f32; 3],
    pub sky_color_horizon: [f32; 3],
    pub sky_color_night: [f32; 3],

    // Sun
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,

    // Atmospheric effects
    pub fog_density_multiplier: f32,
    pub haze_intensity: f32,

    // Precipitation
    pub precipitation_type: PrecipitationType,
    pub precipitation_probability: f32,

    // Temperature
    pub temperature_celsius: f32,

    // Ground coverage
    pub snow_coverage: f32,
    pub frost_intensity: f32,

    // Vegetation
    pub leaf_coverage: f32,
    pub flower_bloom_intensity: f32,

    // Color grading
    pub saturation_multiplier: f32,
    pub warmth_shift: f32,
}

/// Return the seasonal atmospheric parameters for the given season.
pub fn get_seasonal_parameters(season: Season) -> SeasonalParameters {
    match season {
        Season::Spring => SeasonalParameters {
            season: Season::Spring,
            sun_angle_min: 30.0,
            sun_angle_max: 65.0,
            day_length_hours: 12.5,
            sky_color_day: [0.55, 0.70, 0.85],
            sky_color_horizon: [0.85, 0.75, 0.65],
            sky_color_night: [0.05, 0.08, 0.15],
            sun_color: [1.0, 0.95, 0.85],
            sun_intensity: 1.2,
            fog_density_multiplier: 0.8,
            haze_intensity: 0.2,
            precipitation_type: PrecipitationType::Rain,
            precipitation_probability: 0.3,
            temperature_celsius: 15.0,
            snow_coverage: 0.0,
            frost_intensity: 0.1,
            leaf_coverage: 0.6,
            flower_bloom_intensity: 0.7,
            saturation_multiplier: 1.1,
            warmth_shift: 0.1,
        },
        Season::Summer => SeasonalParameters {
            season: Season::Summer,
            sun_angle_min: 40.0,
            sun_angle_max: 75.0,
            day_length_hours: 15.0,
            sky_color_day: [0.40, 0.65, 0.90],
            sky_color_horizon: [0.75, 0.80, 0.90],
            sky_color_night: [0.02, 0.04, 0.10],
            sun_color: [1.0, 1.0, 0.95],
            sun_intensity: 1.5,
            fog_density_multiplier: 0.5,
            haze_intensity: 0.3,
            precipitation_type: PrecipitationType::Rain,
            precipitation_probability: 0.2,
            temperature_celsius: 25.0,
            snow_coverage: 0.0,
            frost_intensity: 0.0,
            leaf_coverage: 1.0,
            flower_bloom_intensity: 0.5,
            saturation_multiplier: 1.3,
            warmth_shift: 0.0,
        },
        Season::Autumn => SeasonalParameters {
            season: Season::Autumn,
            sun_angle_min: 25.0,
            sun_angle_max: 55.0,
            day_length_hours: 11.0,
            sky_color_day: [0.60, 0.65, 0.75],
            sky_color_horizon: [0.90, 0.70, 0.50],
            sky_color_night: [0.08, 0.08, 0.12],
            sun_color: [1.0, 0.85, 0.65],
            sun_intensity: 1.0,
            fog_density_multiplier: 1.5,
            haze_intensity: 0.5,
            precipitation_type: PrecipitationType::Mist,
            precipitation_probability: 0.4,
            temperature_celsius: 10.0,
            snow_coverage: 0.0,
            frost_intensity: 0.2,
            leaf_coverage: 0.7,
            flower_bloom_intensity: 0.1,
            saturation_multiplier: 1.0,
            warmth_shift: 0.3,
        },
        Season::Winter => SeasonalParameters {
            season: Season::Winter,
            sun_angle_min: 15.0,
            sun_angle_max: 40.0,
            day_length_hours: 9.0,
            sky_color_day: [0.70, 0.75, 0.85],
            sky_color_horizon: [0.85, 0.85, 0.90],
            sky_color_night: [0.10, 0.12, 0.18],
            sun_color: [0.95, 0.95, 1.0],
            sun_intensity: 0.8,
            fog_density_multiplier: 1.2,
            haze_intensity: 0.4,
            precipitation_type: PrecipitationType::Snow,
            precipitation_probability: 0.3,
            temperature_celsius: -2.0,
            snow_coverage: 0.7,
            frost_intensity: 0.8,
            leaf_coverage: 0.0,
            flower_bloom_intensity: 0.0,
            saturation_multiplier: 0.8,
            warmth_shift: -0.2,
        },
    }
}

// ---------------------------------------------------------------------------
// VegetationSeasonalState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VegetationSeasonalState {
    pub season: Season,
    pub has_leaves: bool,
    pub leaf_density: f32,
    pub leaf_color_primary: Option<[f32; 3]>,
    pub leaf_color_secondary: Option<[f32; 3]>,
    pub leaf_color_variation: f32,
    pub has_flowers: bool,
    pub flower_density: f32,
    pub flower_color: Option<[f32; 3]>,
    pub has_ground_litter: bool,
    pub ground_litter_density: f32,
    pub snow_accumulation: f32,
    pub has_buds: bool,
    pub bud_density: f32,
}

// ---------------------------------------------------------------------------
// Built-in vegetation state presets
// ---------------------------------------------------------------------------

fn deciduous_spring() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Spring,
        has_leaves: true,
        leaf_density: 0.6,
        leaf_color_primary: Some([0.4, 0.7, 0.2]),
        leaf_color_secondary: Some([0.5, 0.8, 0.3]),
        leaf_color_variation: 0.3,
        has_flowers: true,
        flower_density: 0.5,
        flower_color: Some([1.0, 0.9, 0.95]),
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.0,
        has_buds: true,
        bud_density: 0.4,
    }
}

fn deciduous_summer() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Summer,
        has_leaves: true,
        leaf_density: 1.0,
        leaf_color_primary: Some([0.13, 0.54, 0.13]),
        leaf_color_secondary: Some([0.2, 0.6, 0.2]),
        leaf_color_variation: 0.2,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn deciduous_autumn_yellow() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Autumn,
        has_leaves: true,
        leaf_density: 0.7,
        leaf_color_primary: Some([1.0, 0.84, 0.0]),
        leaf_color_secondary: Some([1.0, 0.65, 0.0]),
        leaf_color_variation: 0.5,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: true,
        ground_litter_density: 0.6,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn deciduous_autumn_red() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Autumn,
        has_leaves: true,
        leaf_density: 0.7,
        leaf_color_primary: Some([0.8, 0.2, 0.1]),
        leaf_color_secondary: Some([0.9, 0.4, 0.1]),
        leaf_color_variation: 0.4,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: true,
        ground_litter_density: 0.6,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn deciduous_autumn_brown() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Autumn,
        has_leaves: true,
        leaf_density: 0.6,
        leaf_color_primary: Some([0.6, 0.4, 0.2]),
        leaf_color_secondary: Some([0.5, 0.35, 0.15]),
        leaf_color_variation: 0.3,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: true,
        ground_litter_density: 0.7,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn deciduous_winter() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Winter,
        has_leaves: false,
        leaf_density: 0.0,
        leaf_color_primary: None,
        leaf_color_secondary: None,
        leaf_color_variation: 0.2,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.5,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn evergreen_spring() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Spring,
        has_leaves: true,
        leaf_density: 1.0,
        leaf_color_primary: Some([0.1, 0.4, 0.1]),
        leaf_color_secondary: None,
        leaf_color_variation: 0.2,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn evergreen_summer() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Summer,
        has_leaves: true,
        leaf_density: 1.0,
        leaf_color_primary: Some([0.1, 0.45, 0.1]),
        leaf_color_secondary: None,
        leaf_color_variation: 0.2,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn evergreen_autumn() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Autumn,
        has_leaves: true,
        leaf_density: 1.0,
        leaf_color_primary: Some([0.08, 0.4, 0.08]),
        leaf_color_secondary: None,
        leaf_color_variation: 0.2,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.0,
        has_buds: false,
        bud_density: 0.0,
    }
}

fn evergreen_winter() -> VegetationSeasonalState {
    VegetationSeasonalState {
        season: Season::Winter,
        has_leaves: true,
        leaf_density: 1.0,
        leaf_color_primary: Some([0.05, 0.35, 0.05]),
        leaf_color_secondary: None,
        leaf_color_variation: 0.2,
        has_flowers: false,
        flower_density: 0.0,
        flower_color: None,
        has_ground_litter: false,
        ground_litter_density: 0.0,
        snow_accumulation: 0.7,
        has_buds: false,
        bud_density: 0.0,
    }
}

/// Known evergreen species (lower-case).
const EVERGREEN_SPECIES: &[&str] = &["pine", "spruce", "fir", "cedar", "cypress", "redwood"];

/// Return the autumn color variant for a deciduous tree family.
fn autumn_color_state(tree_family: &str) -> VegetationSeasonalState {
    match tree_family.to_lowercase().as_str() {
        "maple" | "oak" | "cherry" | "apple" => deciduous_autumn_red(),
        "birch" | "aspen" | "ash" | "poplar" | "willow" => deciduous_autumn_yellow(),
        "beech" | "elm" => deciduous_autumn_brown(),
        _ => deciduous_autumn_yellow(),
    }
}

/// Get the vegetation seasonal state for a tree.
///
/// `tree_family` examples: `"oak"`, `"pine"`, `"birch"`, etc.
/// `is_evergreen` overrides species detection when set to `true`.
pub fn get_vegetation_seasonal_state(
    tree_family: &str,
    season: Season,
    is_evergreen: bool,
) -> VegetationSeasonalState {
    let is_ev = is_evergreen || EVERGREEN_SPECIES.contains(&tree_family.to_lowercase().as_str());

    if is_ev {
        match season {
            Season::Spring => evergreen_spring(),
            Season::Summer => evergreen_summer(),
            Season::Autumn => evergreen_autumn(),
            Season::Winter => evergreen_winter(),
        }
    } else {
        match season {
            Season::Spring => deciduous_spring(),
            Season::Summer => deciduous_summer(),
            Season::Autumn => autumn_color_state(tree_family),
            Season::Winter => deciduous_winter(),
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
    fn spring_params_exact_values() {
        let p = get_seasonal_parameters(Season::Spring);
        assert_eq!(p.sun_angle_min, 30.0);
        assert_eq!(p.day_length_hours, 12.5);
        assert_eq!(p.sky_color_day, [0.55, 0.70, 0.85]);
        assert_eq!(p.sun_intensity, 1.2);
        assert_eq!(p.fog_density_multiplier, 0.8);
        assert_eq!(p.precipitation_type, PrecipitationType::Rain);
    }

    #[test]
    fn winter_params_snow_coverage() {
        let p = get_seasonal_parameters(Season::Winter);
        assert_eq!(p.snow_coverage, 0.7);
        assert_eq!(p.frost_intensity, 0.8);
        assert_eq!(p.temperature_celsius, -2.0);
        assert_eq!(p.precipitation_type, PrecipitationType::Snow);
    }

    #[test]
    fn autumn_maple_gets_red_leaves() {
        let state = get_vegetation_seasonal_state("maple", Season::Autumn, false);
        // DECIDUOUS_AUTUMN_RED: leaf_color_primary = (0.8, 0.2, 0.1)
        assert_eq!(state.leaf_color_primary, Some([0.8, 0.2, 0.1]));
    }

    #[test]
    fn birch_autumn_yellow() {
        let state = get_vegetation_seasonal_state("birch", Season::Autumn, false);
        assert_eq!(state.leaf_color_primary, Some([1.0, 0.84, 0.0]));
    }

    #[test]
    fn evergreen_winter_snow() {
        let state = get_vegetation_seasonal_state("pine", Season::Winter, false);
        assert_eq!(state.snow_accumulation, 0.7);
    }
}
