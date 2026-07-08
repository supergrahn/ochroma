//! Celestial body models for the Ochroma engine.
//!
//! Engine-level, game-agnostic. The game layer (urban_horizon) owns map-specific
//! parameters (latitude, clock cadence) and calls into these to obtain GPU-ready
//! light directions and colors.
//!
//! # Modules
//!
//! - [`sun`] — NOAA Solar Position Algorithm (accurate altitude/azimuth, equation of time).
//! - [`moon`] — Meeus low-precision lunar position + phase (20 longitude + 10 latitude terms).
//! - [`key_light`] — Unified day/night key-light blending sun and moon with civil-twilight
//!   smoothstep and sky-dome gradient.

pub mod key_light;
pub mod moon;
pub mod sun;

pub use key_light::{KeyLight, KeyLightPalette, celestial_key_light};
pub use moon::{MoonConfig, MoonPosition, compute_moon_position};
pub use sun::{
    SunConfig, SunPosition, compute_sun_position, hour_from_tick_of_day, solar_day_from_civic_day,
    sun_color_radiance_from_altitude,
};
