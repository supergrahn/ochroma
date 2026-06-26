//! Environment-variable + timing helpers shared across the render runtime.
//!
//! Moved verbatim from the game's `spectra_frame::helpers` so a new game never
//! re-implements these. No game-type dependencies (std only). These read generic
//! env-var NAMES passed by the caller — the project-specific `OCHROMA_*` wrappers
//! (`retained_scene_enabled`/`scene_breakdown_enabled`) stay game-side.

/// Milliseconds elapsed since `t`.
pub fn elapsed_ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// True when env var `name` is set to a non-empty, non-`"0"` value.
pub fn env_truthy(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Parse env var `name` as a `usize`, returning `None` if unset/unparseable.
pub fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// Parse env var `name` as an `f32`, returning `None` if unset/unparseable.
pub fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.trim().parse().ok()
}
