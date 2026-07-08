//! Unified day/night key-light for the Ochroma engine.
//!
//! `celestial_key_light` blends the sun (daytime) and moon (nighttime) into a
//! single key-light that drives the path tracer's sun slot throughout the full
//! 24-hour cycle. The moon uses the same GPU uniform as the sun at night — no new
//! uniforms, no kernel changes.
//!
//! # Transition
//!
//! Civil twilight spans solar altitude −6° … +6°. We use a `smoothstep` blend
//! factor `t` over that range:
//! - `t = 1.0`: full day (sun above +6°).
//! - `t = 0.0`: full night (sun below −6°).
//! - `0 < t < 1`: dusk/dawn blend.
//!
//! The **direction** is never interpolated through zero: whichever body is dominant
//! (t > 0.5 → sun, t ≤ 0.5 → moon) supplies the direction. This prevents the
//! "false-zenith bug" where zeroing the night direction made the shader fall back
//! to (0,1,0) and render a bright noon sky at midnight.
//!
//! # Sky dome
//!
//! `sky_zenith`, `sky_horizon`, and `sky_intensity` are also blended between day,
//! dusk, and night palettes and returned for the renderer's miss-ray gradient.

use crate::celestial::moon::MoonPosition;
use crate::celestial::sun::SunPosition;

/// DATA-only look palette for the unified celestial key-light.
///
/// The ENGINE owns the *math* (`celestial_key_light` — the smoothstep/lerp
/// civil-twilight blend, the two-stage night→dusk→day interpolation, the
/// altitude-driven sun ramp). This struct carries the *numbers* the math
/// interpolates: the day/dusk/night sky-dome anchor palettes + the daytime sun
/// color/radiance ramp coefficients.
///
/// Per CLAUDE.md, `vox_core` is an ENGINE crate and must NOT bake game look
/// policy. So `celestial_key_light` takes a `&KeyLightPalette` and the GAME
/// supplies the values from `assets/config/render.ron` (the ONE render config).
/// [`KeyLightPalette::default`] carries the historical working consts as a
/// fallback ONLY; the game's `render.ron` is the source of truth, pinned equal
/// to this default by the game's `default_mirrors_render_ron` test.
///
/// This is the same pattern [`crate::celestial::moon::MoonConfig`] already uses
/// (a data struct threaded into the position computation), extended to the
/// celestial key-light look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyLightPalette {
    // ── Sky-dome anchor palettes (linear RGB + intensity) ────────────────────
    /// Daytime sky-dome zenith color (sun above +6°).
    pub day_zenith: [f32; 3],
    /// Daytime sky-dome horizon color.
    pub day_horizon: [f32; 3],
    /// Daytime sky-dome intensity multiplier.
    pub day_intensity: f32,
    /// Dusk/dawn sky-dome zenith color (sun near the horizon).
    pub dusk_zenith: [f32; 3],
    /// Dusk/dawn sky-dome horizon color.
    pub dusk_horizon: [f32; 3],
    /// Dusk/dawn sky-dome intensity multiplier.
    pub dusk_intensity: f32,
    /// Nighttime sky-dome zenith color (sun below −6°).
    pub night_zenith: [f32; 3],
    /// Nighttime sky-dome horizon color.
    pub night_horizon: [f32; 3],
    /// Nighttime sky-dome intensity multiplier.
    pub night_intensity: f32,
    // ── Daytime sun color + radiance ramp (driven by sin(altitude)) ──────────
    /// Sun green-channel base at horizon (`G = base + slope * sin_alt`).
    pub sun_color_g_base: f32,
    /// Sun green-channel slope toward zenith.
    pub sun_color_g_slope: f32,
    /// Sun blue-channel base at horizon (`B = base + slope * sin_alt`).
    pub sun_color_b_base: f32,
    /// Sun blue-channel slope toward zenith.
    pub sun_color_b_slope: f32,
    /// Sun radiance base at horizon (`radiance = base + scale * sin_alt^exp`).
    pub sun_radiance_base: f32,
    /// Sun radiance scale toward zenith.
    pub sun_radiance_scale: f32,
    /// Sun radiance altitude exponent.
    pub sun_radiance_exp: f32,
}

impl Default for KeyLightPalette {
    /// Historical WORKING consts (pre-session). These are the engine fallback;
    /// the game's `render.ron` mirrors them and is the source of truth.
    fn default() -> Self {
        Self {
            // Sky-dome AMBIENT fill (not the visible atmosphere sky). A
            // desaturated soft-blue at roughly HALF strength so the warm
            // directional sun dominates and shadows read with contrast.
            day_zenith: [0.40, 0.52, 0.72], // soft desaturated sky-blue
            day_horizon: [0.80, 0.83, 0.88], // near-neutral warm horizon
            day_intensity: 0.45,
            dusk_zenith: [0.12, 0.16, 0.45],  // deep dusk blue
            dusk_horizon: [1.00, 0.45, 0.18], // orange-red horizon
            dusk_intensity: 0.35,
            night_zenith: [0.008, 0.015, 0.060], // dark deep blue
            night_horizon: [0.015, 0.022, 0.090],
            night_intensity: 0.018,
            // Color: warm orange-red at horizon → neutral white at noon.
            sun_color_g_base: 0.55, // 0.55 near horizon → 0.98 at noon
            sun_color_g_slope: 0.43,
            sun_color_b_base: 0.10, // 0.10 near horizon → 0.95 at noon
            sun_color_b_slope: 0.85,
            // Radiance: dim at horizon, bright at noon.
            sun_radiance_base: 8.0,
            sun_radiance_scale: 22.0,
            sun_radiance_exp: 0.4,
        }
    }
}

/// The unified key-light returned by `celestial_key_light`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyLight {
    /// World-space direction FROM scene TOWARD the light source (Y-up, never zero).
    pub direction: [f32; 3],
    /// Linear RGB color.
    pub color: [f32; 3],
    /// Scalar radiance / intensity (maps to `LightRig::sun_radiance`).
    pub radiance: f32,
    /// Sky-dome zenith color (linear RGB).
    pub sky_zenith: [f32; 3],
    /// Sky-dome horizon color (linear RGB).
    pub sky_horizon: [f32; 3],
    /// Sky-dome overall intensity multiplier.
    pub sky_intensity: f32,
}

/// Compute the unified celestial key-light from a pre-computed sun and moon
/// position and a DATA-only look [`KeyLightPalette`].
///
/// `sun` and `moon` must have been computed for the SAME observer and time.
/// The function does not re-compute positions — it only blends the results.
///
/// The engine owns the *math* (the smoothstep/lerp blend below); the `palette`
/// supplies every look *number* (the day/dusk/night sky anchors + the sun
/// color/radiance ramp coefficients). Pass [`KeyLightPalette::default`] for the
/// historical working look; the GAME passes a palette built from `render.ron`.
pub fn celestial_key_light(
    sun: &SunPosition,
    moon: &MoonPosition,
    palette: &KeyLightPalette,
) -> KeyLight {
    // Civil twilight blend factor: smoothstep over −6° … +6° solar altitude.
    // (−6° … +6° is the standard civil-twilight convention — physics-ish, stays.)
    let sun_alt_deg = sun.altitude_rad.to_degrees() as f32;
    let t = smoothstep(-6.0, 6.0, sun_alt_deg);

    // ── Day key-light (sun) ──────────────────────────────────────────────────
    // Altitude factor: 0 at horizon, 1 at zenith; drives color + radiance.
    let sin_alt = sun.altitude_rad.sin().max(0.0) as f32;
    // Color: warm orange-red at horizon → neutral white at noon (ramp from palette).
    let day_color = [
        1.0_f32,
        palette.sun_color_g_base + palette.sun_color_g_slope * sin_alt,
        palette.sun_color_b_base + palette.sun_color_b_slope * sin_alt,
    ];
    // Radiance: dim at horizon, bright at noon (ramp from palette).
    let day_radiance = palette.sun_radiance_base
        + palette.sun_radiance_scale * sin_alt.powf(palette.sun_radiance_exp);

    // ── Night key-light (moon) ───────────────────────────────────────────────
    let night_color = moon.color;
    let night_radiance = moon.radiance; // already phase-scaled, 0 if below horizon

    // ── Direction ────────────────────────────────────────────────────────────
    // Never lerp direction through zero: pick whichever body dominates.
    let direction = if t > 0.5 {
        sun.direction
    } else {
        moon.direction
    };

    // ── Radiance and color blend ──────────────────────────────────────────────
    let radiance = lerp(night_radiance, day_radiance, t);
    let color = lerp3(night_color, day_color, t);

    // ── Sky dome ──────────────────────────────────────────────────────────────
    // Three anchor palettes (day, dusk, night) come from the DATA `palette`; the
    // engine owns only the two-stage blend MATH below. The values were tuned so
    // the warm directional sun dominates and shadows read with contrast (a
    // saturated pure-blue dome at high intensity floods every surface with blue
    // and washes the brick grey).
    //
    // Two-stage blend: night→dusk at t=0..0.5, dusk→day at t=0.5..1.
    let (sky_zenith, sky_horizon, sky_intensity) = if t <= 0.5 {
        let s = t * 2.0; // 0..1 over night→dusk
        (
            lerp3(palette.night_zenith, palette.dusk_zenith, s),
            lerp3(palette.night_horizon, palette.dusk_horizon, s),
            lerp(palette.night_intensity, palette.dusk_intensity, s),
        )
    } else {
        let s = (t - 0.5) * 2.0; // 0..1 over dusk→day
        (
            lerp3(palette.dusk_zenith, palette.day_zenith, s),
            lerp3(palette.dusk_horizon, palette.day_horizon, s),
            lerp(palette.dusk_intensity, palette.day_intensity, s),
        )
    };

    KeyLight {
        direction,
        color,
        radiance,
        sky_zenith,
        sky_horizon,
        sky_intensity,
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celestial::moon::{MoonConfig, MoonPosition};
    use crate::celestial::sun::{SunConfig, compute_sun_position};

    const EARTH_CONFIG: SunConfig = SunConfig {
        latitude_deg: 40.71,
        north_azimuth_offset_deg: 0.0,
        axial_tilt_deg: 23.44,
    };

    const MOON_CONFIG: MoonConfig = MoonConfig {
        max_radiance: 0.08,
        color: [0.72, 0.82, 1.0],
    };

    fn nyc_sun(day: u32, hour: f64) -> SunPosition {
        compute_sun_position(day, hour, 40.7128, -74.006, &EARTH_CONFIG)
    }

    fn full_moon_position(above_horizon: bool) -> MoonPosition {
        // Normalized direction: [0, sin(50°), cos(50°)] ≈ [0, 0.766, 0.643]
        MoonPosition {
            altitude_rad: if above_horizon { 0.8 } else { -0.3 },
            azimuth_rad: 3.14,
            direction: [0.0, 0.766_044_4, 0.642_787_6], // sin/cos(50°), unit length
            phase: 1.0,
            phase_angle_rad: 0.0, // full moon → phase angle 0
            bright_limb_angle_rad: 0.0,
            radiance: if above_horizon {
                MOON_CONFIG.max_radiance
            } else {
                0.0
            },
            color: MOON_CONFIG.color,
        }
    }

    fn new_moon_position() -> MoonPosition {
        MoonPosition {
            altitude_rad: 0.5,
            azimuth_rad: 1.0,
            direction: [0.5, 0.7, 0.5],
            phase: 0.0,
            phase_angle_rad: std::f64::consts::PI, // new moon → phase angle π
            bright_limb_angle_rad: 0.0,
            radiance: 0.0,
            color: MOON_CONFIG.color,
        }
    }

    /// Noon (equinox, NYC): bright warm-white key + bright blue sky.
    #[test]
    fn noon_bright_warm_white_sky() {
        let sun = nyc_sun(80, 12.0); // local noon ≈ solar noon at equinox
        let moon = full_moon_position(false); // moon below horizon at noon
        let key = celestial_key_light(&sun, &moon, &KeyLightPalette::default());

        assert!(
            key.radiance > 15.0,
            "noon key-light should be bright; radiance={:.2}",
            key.radiance
        );
        // Color should be warm white (R > 0.9, G > 0.8, B > 0.6).
        assert!(
            key.color[0] > 0.9 && key.color[1] > 0.7,
            "noon color should be warm white; got {:?}",
            key.color
        );
        // Sky should be at the daytime intensity anchor (the WORKING value: the
        // desaturated soft-blue dome runs at ~HALF strength so the warm sun
        // dominates and shadows keep contrast — DAY_INTENSITY 0.45, NOT the old
        // washed-out 0.9). Pin against the palette default so a render.ron edit
        // to day_intensity is caught here too.
        assert!(
            (key.sky_intensity - KeyLightPalette::default().day_intensity).abs() < 1e-6,
            "noon sky intensity should equal the day anchor ({:.2}); got {:.2}",
            KeyLightPalette::default().day_intensity,
            key.sky_intensity
        );
        assert!(
            key.sky_zenith[2] > key.sky_zenith[0],
            "noon sky zenith should have more blue than red; got {:?}",
            key.sky_zenith
        );
    }

    /// Midnight (equinox, NYC) + full moon above horizon → dim cool key + dark sky.
    #[test]
    fn midnight_full_moon_dim_cool_dark_sky() {
        let sun = nyc_sun(80, 0.0); // midnight
        let moon = full_moon_position(true);
        let key = celestial_key_light(&sun, &moon, &KeyLightPalette::default());

        // Key light should be dim (moon radiance << noon sun).
        assert!(
            key.radiance < 0.2,
            "midnight full-moon key should be very dim; radiance={:.4}",
            key.radiance
        );
        // Color should be cool (blue > red).
        assert!(
            key.color[2] > key.color[0],
            "midnight moon color should be cool (B > R); got {:?}",
            key.color
        );
        // Sky should be dark.
        assert!(
            key.sky_intensity < 0.10,
            "midnight sky intensity should be very low; got {:.4}",
            key.sky_intensity
        );
    }

    /// Midnight + new moon → near-zero key radiance + dark sky.
    #[test]
    fn midnight_new_moon_near_zero_key() {
        let sun = nyc_sun(80, 0.0);
        let moon = new_moon_position();
        let key = celestial_key_light(&sun, &moon, &KeyLightPalette::default());

        assert!(
            key.radiance < 0.01,
            "new-moon midnight key radiance should be near zero; got {:.6}",
            key.radiance
        );
        assert!(
            key.sky_intensity < 0.10,
            "new-moon midnight sky should be dark; got {:.4}",
            key.sky_intensity
        );
    }

    /// Direction is always unit-length (never zero or NaN).
    #[test]
    fn direction_always_unit() {
        for (day, hour) in [
            (1u32, 0.0_f64),
            (80, 6.0),
            (80, 12.0),
            (172, 0.0),
            (355, 18.0),
        ] {
            let sun = nyc_sun(day, hour);
            let moon = full_moon_position(true);
            let key = celestial_key_light(&sun, &moon, &KeyLightPalette::default());
            let d = key.direction;
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "direction must be unit; day={day} hour={hour} len={len:.6}"
            );
            assert!(
                d.iter().all(|x| x.is_finite()),
                "direction must be finite; day={day} hour={hour} dir={d:?}"
            );
        }
    }

    /// Below-horizon moon drives zero key radiance.
    #[test]
    fn below_horizon_moon_zero_radiance() {
        let sun = nyc_sun(80, 0.0); // midnight
        let moon = full_moon_position(false); // explicitly below horizon
        let key = celestial_key_light(&sun, &moon, &KeyLightPalette::default());
        assert_eq!(
            key.radiance, 0.0,
            "below-horizon moon should give zero key radiance"
        );
    }

    /// Sky intensity increases from midnight to civil-twilight start to noon.
    ///
    /// The civil-twilight blend runs over sun altitude −6° … +6°. We compare
    /// three points: midnight (below horizon), civil-twilight onset (~6:20 local
    /// at NYC equinox, altitude ≈ −3°), and solar noon.
    #[test]
    fn sky_intensity_increases_day_to_night() {
        let moon = full_moon_position(false);
        // NYC equinox: sunrise ≈ 06:00 local; civil twilight starts before 06:00.
        // At 06:10 the sun is around −2°…+2° (in the twilight blend zone).
        let midnight = celestial_key_light(&nyc_sun(80, 0.0), &moon, &KeyLightPalette::default());
        let civil_twilight =
            celestial_key_light(&nyc_sun(80, 6.1), &moon, &KeyLightPalette::default());
        let noon = celestial_key_light(&nyc_sun(80, 12.0), &moon, &KeyLightPalette::default());

        assert!(
            midnight.sky_intensity < civil_twilight.sky_intensity,
            "sky_intensity: midnight ({:.3}) < civil_twilight ({:.3})",
            midnight.sky_intensity,
            civil_twilight.sky_intensity
        );
        assert!(
            civil_twilight.sky_intensity < noon.sky_intensity,
            "sky_intensity: civil_twilight ({:.3}) < noon ({:.3})",
            civil_twilight.sky_intensity,
            noon.sky_intensity
        );
    }
}
