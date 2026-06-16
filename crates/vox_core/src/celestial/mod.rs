//! Celestial body models for the Ochroma engine.
//!
//! Engine-level, game-agnostic. The game layer (urban_horizon) owns map-specific
//! parameters (latitude, clock cadence) and calls into these to obtain GPU-ready
//! light directions and colors.

pub mod sun;
pub use sun::{SunConfig, hour_from_tick_of_day, solar_day_from_civic_day};
