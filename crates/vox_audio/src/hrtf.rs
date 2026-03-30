//! Simplified analytical HRTF for binaural audio.
//!
//! Computes per-ear gain and delay from azimuth/elevation using the Woodworth
//! spherical-head model. No external dataset required — pure geometry.
//!
//! ITD (interaural time difference): Woodworth formula.
//! ILD (interaural level difference): frequency-dependent shadowing.

const SPEED_OF_SOUND: f32 = 343.0; // m/s
const SAMPLE_RATE: f32 = 44100.0;

/// Parameters for the analytical HRTF model.
pub struct HrtfParams {
    /// Head radius in metres. Human average is ~0.0875 m.
    pub head_radius_m: f32,
}

impl Default for HrtfParams {
    fn default() -> Self {
        Self { head_radius_m: 0.0875 }
    }
}

/// Per-sample binaural panning result.
pub struct BinauralSample {
    pub left_gain: f32,
    pub right_gain: f32,
    /// ITD in samples at 44100 Hz. Positive = right ear leads (source to the right).
    pub itd_samples: i32,
}

/// Compute HRTF gains and ITD from azimuth/elevation.
///
/// `azimuth_rad` — horizontal angle; positive = source to the right of the listener.
/// `elevation_rad` — vertical angle; positive = above the listener.
pub fn compute_hrtf(azimuth_rad: f32, elevation_rad: f32, params: &HrtfParams) -> BinauralSample {
    // --- ITD via Woodworth formula ---
    // itd_seconds = (r / c) * (sin(θ) + θ)  where θ = azimuth
    let itd_seconds = (params.head_radius_m / SPEED_OF_SOUND)
        * (azimuth_rad.sin() + azimuth_rad);
    let itd_samples = (itd_seconds * SAMPLE_RATE).round() as i32;

    // --- ILD ---
    let ild_db = 6.0 * azimuth_rad.sin().abs() * (1.0 + elevation_rad.abs() * 0.3);
    let ild_linear = 10f32.powf(ild_db / 20.0);

    // Positive azimuth → source is to the right → left ear is shadowed
    let (left_gain, right_gain) = if azimuth_rad >= 0.0 {
        (1.0 / ild_linear, 1.0_f32)
    } else {
        (1.0_f32, 1.0 / ild_linear)
    };

    let left_gain = left_gain.clamp(0.1, 1.0);
    let right_gain = right_gain.clamp(0.1, 1.0);

    BinauralSample { left_gain, right_gain, itd_samples }
}

/// Convert a world-space source position to HRTF azimuth and elevation
/// relative to the listener's coordinate frame.
///
/// Returns `(azimuth_rad, elevation_rad)`.
pub fn world_to_hrtf(
    source_pos: glam::Vec3,
    listener_pos: glam::Vec3,
    listener_forward: glam::Vec3,
    listener_up: glam::Vec3,
) -> (f32, f32) {
    let rel = source_pos - listener_pos;
    let distance = rel.length();

    if distance < 1e-6 {
        return (0.0, 0.0);
    }

    let forward = listener_forward.normalize();
    let up = listener_up.normalize();
    let right = forward.cross(up).normalize();

    let fwd_comp = rel.dot(forward);
    let right_comp = rel.dot(right);
    let up_comp = rel.dot(up);

    let azimuth = right_comp.atan2(fwd_comp);
    // Clamp argument to [-1, 1] to guard against floating-point overshoot
    let elevation = (up_comp / distance).clamp(-1.0, 1.0).asin();

    (azimuth, elevation)
}

/// Manages spatial HRTF panning for a set of audio sources.
pub struct SpatialHrtfMixer {
    pub sample_rate: u32,
    params: HrtfParams,
}

impl SpatialHrtfMixer {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            params: HrtfParams::default(),
        }
    }

    /// Compute binaural panning for a single source.
    pub fn compute_pan(
        &self,
        source_pos: [f32; 3],
        listener_pos: [f32; 3],
        listener_forward: [f32; 3],
        listener_up: [f32; 3],
    ) -> BinauralSample {
        let (az, el) = world_to_hrtf(
            glam::Vec3::from(source_pos),
            glam::Vec3::from(listener_pos),
            glam::Vec3::from(listener_forward),
            glam::Vec3::from(listener_up),
        );
        compute_hrtf(az, el, &self.params)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn hrtf_source_directly_ahead_is_centered() {
        let params = HrtfParams::default();
        let s = compute_hrtf(0.0, 0.0, &params);
        let diff = (s.left_gain - s.right_gain).abs();
        assert!(diff < 1e-4, "gains should be equal ahead: L={} R={}", s.left_gain, s.right_gain);
    }

    #[test]
    fn hrtf_source_right_has_higher_right_gain() {
        let params = HrtfParams::default();
        let s = compute_hrtf(PI / 2.0, 0.0, &params);
        assert!(
            s.right_gain > s.left_gain,
            "right gain should exceed left for source at PI/2: L={} R={}",
            s.left_gain,
            s.right_gain
        );
    }

    #[test]
    fn hrtf_itd_is_zero_at_center() {
        let params = HrtfParams::default();
        let s = compute_hrtf(0.0, 0.0, &params);
        assert_eq!(s.itd_samples, 0);
    }

    #[test]
    fn hrtf_itd_positive_for_right_source() {
        let params = HrtfParams::default();
        let s = compute_hrtf(PI / 4.0, 0.0, &params);
        assert!(s.itd_samples > 0, "ITD should be positive for source at PI/4, got {}", s.itd_samples);
    }

    #[test]
    fn world_to_hrtf_source_ahead() {
        // Source directly ahead of listener
        let source = glam::Vec3::new(0.0, 0.0, -5.0);
        let listener = glam::Vec3::ZERO;
        let forward = glam::Vec3::new(0.0, 0.0, -1.0);
        let up = glam::Vec3::new(0.0, 1.0, 0.0);
        let (az, _el) = world_to_hrtf(source, listener, forward, up);
        assert!(az.abs() < 1e-4, "azimuth should be ~0 for source ahead, got {}", az);
    }
}
