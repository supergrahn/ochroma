//! vox_aether — Atmosphere, weather, season, and lighting data model layer.
//!
//! Ported from AetherSpectra's Python atmosphere/weather engine. This crate is
//! pure Rust data/model with NO GPU dependencies, no wgpu, no CUDA. It produces
//! atmospheric specifications consumed by the render layer.
//!
//! # Reconciliation with vox_core::celestial
//! Sun and moon direction are NOT computed here — that lives in
//! `vox_core::celestial` (`compute_sun_position`, `compute_moon_position`,
//! `celestial_key_light`). This crate produces the atmospheric spec (fog, clouds,
//! particles, mood, season) that WRAPS the accurate celestial directions at
//! runtime. Compose them: get direction from vox_core::celestial, then apply the
//! atmospheric/weather spec from vox_aether on top.

pub mod atmosphere;
pub mod lighting;
pub mod season;
pub mod weather;

pub use atmosphere::{
    AtmosphereConfig, AtmosphereGenerator, CloudPreset, CloudType, DepthHazeSettings, FogPreset,
    FogSettings, FogType, GodRaySettings, ParticlePreset, ParticleSystem, ParticleType,
    VolumetricCloudSettings, cloud_preset, fog_preset, particle_preset,
};
pub use lighting::{
    LightSource, LightingDesigner, LightingSetup, Mood, MoodColor, SkySettings, TimeOfDay,
    TimePreset, VolumetricSettings, Weather, WeatherModifier, mood_color, time_preset,
    weather_modifier,
};
pub use season::{
    PrecipitationType, Season, SeasonalParameters, VegetationSeasonalState,
    get_seasonal_parameters, get_vegetation_seasonal_state,
};
pub use weather::{
    FogConfig, ParticleConfig, WeatherEffect, WeatherEffectsLibrary, WeatherType, WindPreset,
};
