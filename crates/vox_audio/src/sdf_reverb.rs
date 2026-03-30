//! SDF-driven reverb estimation.
//!
//! Estimates RT60 (reverb time), pre-delay, and room size from either
//! a sampled SDF or a simple room descriptor. Uses Sabine's formula.
//!
//! For each spectral band, computes a ReverbBand with RT60 and room gain.
//! The 8 Ochroma spectral bands map to 3 acoustic frequency groups:
//!   Bands 0-2 → high freq (2kHz-8kHz): absorbed more by air
//!   Bands 3-5 → mid freq (500Hz-2kHz): absorbed by surfaces
//!   Bands 6-7 → low freq (80Hz-250Hz): least absorption

/// Room acoustic descriptor — 3 absorption values for [high, mid, low] frequency groups.
#[derive(Debug, Clone)]
pub struct RoomEstimate {
    pub volume_m3: f32,
    pub surface_area_m2: f32,
    /// Average absorption coefficients per frequency group: [high, mid, low].
    pub avg_absorption: [f32; 3],
}

/// Reverb parameters for a single spectral band.
#[derive(Debug, Clone, Copy)]
pub struct ReverbBand {
    /// Time in seconds for sound to decay by 60 dB (Sabine's formula).
    pub rt60: f32,
    /// Early reflection pre-delay in milliseconds.
    pub pre_delay_ms: f32,
    /// Late reverb tail gain (based on surface reflectivity).
    pub room_gain: f32,
}

/// Full reverb profile across all 8 Ochroma spectral bands.
#[derive(Debug, Clone)]
pub struct ReverbProfile {
    pub bands: [ReverbBand; 8],
}

/// Compute RT60 using Sabine's formula:  RT60 = 0.161 * V / (absorption * surface_area)
fn sabine_rt60(volume_m3: f32, avg_absorption: f32, surface_area_m2: f32) -> f32 {
    let a = avg_absorption.max(1e-6);
    let s = surface_area_m2.max(1e-6);
    (0.161 * volume_m3) / (a * s)
}

/// Estimate the reverb profile from a room descriptor.
///
/// Band mapping:
///   Bands 0-2 → high freq  → avg_absorption[0]
///   Bands 3-5 → mid freq   → avg_absorption[1]
///   Bands 6-7 → low freq   → avg_absorption[2]
pub fn estimate_reverb(room: &RoomEstimate) -> ReverbProfile {
    let pre_delay_ms =
        1000.0 * room.volume_m3.powf(1.0 / 3.0) / (2.0 * 343.0);

    // One ReverbBand per absorption group.
    let make_band = |abs: f32| -> ReverbBand {
        let rt60 = sabine_rt60(room.volume_m3, abs, room.surface_area_m2);
        let room_gain = (1.0 - abs).clamp(0.0, 0.99);
        ReverbBand { rt60, pre_delay_ms, room_gain }
    };

    let high = make_band(room.avg_absorption[0]);
    let mid  = make_band(room.avg_absorption[1]);
    let low  = make_band(room.avg_absorption[2]);

    ReverbProfile {
        bands: [
            high, high, high, // bands 0-2 → high freq
            mid,  mid,  mid,  // bands 3-5 → mid freq
            low,  low,        // bands 6-7 → low freq
        ],
    }
}

/// Build a `RoomEstimate` from AABB half-extents and per-group absorption.
///
/// Full box dimensions: l = 2*hx, w = 2*hy, h = 2*hz
///   volume        = l * w * h = 8 * hx * hy * hz
///   surface area  = 2 * (l*w + w*h + h*l)
pub fn room_from_aabb(half_extents: glam::Vec3, avg_absorption: [f32; 3]) -> RoomEstimate {
    let (hx, hy, hz) = (half_extents.x, half_extents.y, half_extents.z);
    let l = 2.0 * hx;
    let w = 2.0 * hy;
    let h = 2.0 * hz;
    let volume_m3 = l * w * h;
    let surface_area_m2 = 2.0 * (l * w + w * h + h * l);
    RoomEstimate { volume_m3, surface_area_m2, avg_absorption }
}

/// Open-space reverb preset — very short RT60, minimal pre-delay.
///
/// Modeled as a small, highly-absorptive volume to represent sound dispersing
/// into open air with almost no reflective boundary energy.
pub fn outdoor_reverb() -> ReverbProfile {
    // Use a small representative volume with near-maximum absorption so that
    // Sabine's formula yields RT60 well below 0.3 s for all bands.
    let room = room_from_aabb(
        glam::Vec3::new(5.0, 2.5, 5.0),
        [0.97, 0.95, 0.90],
    );
    estimate_reverb(&room)
}

/// Cave reverb preset — long RT60 due to highly reflective stone surfaces.
pub fn cave_reverb() -> ReverbProfile {
    let room = room_from_aabb(
        glam::Vec3::new(5.0, 2.5, 4.0),
        [0.02, 0.03, 0.01],
    );
    estimate_reverb(&room)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sabine_rt60_large_room_longer_than_small() {
        let small = room_from_aabb(glam::Vec3::new(2.0, 2.0, 2.0), [0.3, 0.3, 0.3]);
        let large = room_from_aabb(glam::Vec3::new(10.0, 5.0, 10.0), [0.3, 0.3, 0.3]);
        let small_profile = estimate_reverb(&small);
        let large_profile = estimate_reverb(&large);
        // All bands should have longer RT60 for the larger room.
        for i in 0..8 {
            assert!(
                large_profile.bands[i].rt60 > small_profile.bands[i].rt60,
                "band {}: large rt60 {} <= small rt60 {}",
                i,
                large_profile.bands[i].rt60,
                small_profile.bands[i].rt60
            );
        }
    }

    #[test]
    fn outdoor_reverb_low_rt60() {
        let profile = outdoor_reverb();
        for (i, band) in profile.bands.iter().enumerate() {
            assert!(
                band.rt60 < 0.3,
                "band {}: outdoor rt60 {} >= 0.3s",
                i,
                band.rt60
            );
        }
    }

    #[test]
    fn cave_reverb_high_rt60() {
        let profile = cave_reverb();
        // Bands 6-7 → low freq (least absorption) — should have the highest RT60.
        for i in 6..8 {
            assert!(
                profile.bands[i].rt60 > 1.0,
                "band {}: cave low-freq rt60 {} <= 1.0s",
                i,
                profile.bands[i].rt60
            );
        }
    }

    #[test]
    fn pre_delay_scales_with_room_size() {
        let small = room_from_aabb(glam::Vec3::new(2.0, 2.0, 2.0), [0.3, 0.3, 0.3]);
        let large = room_from_aabb(glam::Vec3::new(20.0, 10.0, 20.0), [0.3, 0.3, 0.3]);
        let small_profile = estimate_reverb(&small);
        let large_profile = estimate_reverb(&large);
        // pre_delay is the same within a profile (derived from volume only), but
        // larger room → larger volume → longer pre_delay.
        assert!(
            large_profile.bands[0].pre_delay_ms > small_profile.bands[0].pre_delay_ms,
            "large pre_delay {} <= small pre_delay {}",
            large_profile.bands[0].pre_delay_ms,
            small_profile.bands[0].pre_delay_ms
        );
    }
}
