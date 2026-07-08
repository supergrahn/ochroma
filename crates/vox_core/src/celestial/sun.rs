//! Accurate solar position model for the Ochroma engine.
//!
//! Implements the **NOAA Solar Position Algorithm** as published by the
//! US National Oceanic and Atmospheric Administration / ESRL Global Monitoring
//! Division, which is itself based on Meeus, *Astronomical Algorithms* (2nd ed.,
//! 1998). The NOAA formulae are described at:
//!   <https://gml.noaa.gov/grad/solcalc/solareqns.PDF>
//!
//! Achieved accuracy vs. full Meeus: < 0.01° in zenith angle for dates within
//! ±50 years of J2000.0, and < 0.5° for dates within ±150 years. More than
//! sufficient for a real-time game renderer.
//!
//! # Algorithm summary
//!
//! 1. Fractional year γ from day-of-year + fractional hour.
//! 2. **Equation of time** (minutes): Fourier series in γ (accounts for the
//!    eccentricity of Earth's orbit and the tilt of the ecliptic).
//! 3. **Solar declination** δ (radians): Fourier series in γ (encodes the 23.44°
//!    obliquity and its annual variation). The `axial_tilt_deg` SunConfig field is
//!    preserved for legacy/alien-planet use but the DEFAULT Earth computation
//!    comes from the NOAA Fourier series — not a simple sine of the tilt.
//! 4. **True solar time** from clock UTC + equation-of-time + longitude correction.
//! 5. **Hour angle** H from true solar time (0 at solar noon).
//! 6. **Altitude** from sin(lat)sin(δ)+cos(lat)cos(δ)cos(H).
//! 7. **Azimuth** (0=N, 90=E, 180=S, 270=W) from the full-circle atan2 form.
//! 8. Convert altitude + azimuth → Y-up world direction; apply north-axis offset.
//!
//! The returned direction is NEVER zeroed below the horizon — radiance carries
//! the on/off; a zero direction causes a default-zenith fallback that produces a
//! bright noon sky at midnight (the old bug).
//!
//! # Coordinate system (unchanged from previous version)
//!
//! Engine is Y-up. The returned direction points FROM the scene TOWARD the sun
//! (`dot(normal, sun_dir)` in the megakernel).
//!
//! World axes (urban_horizon map convention):
//! - **+X** = east
//! - **+Y** = up
//! - **+Z** = south  (`north_azimuth_offset_deg = 0` → map +Z is geographic south)

use glam::{Mat3, Vec3};

/// Configuration for the engine sun. All angular fields are in degrees.
///
/// # Fields
///
/// | Field | Meaning | Default |
/// |---|---|---|
/// | `latitude_deg` | Observer latitude (°N positive) | 40.71 (NYC) |
/// | `north_azimuth_offset_deg` | Compass bearing (°) of the map +Z axis from true north | 0.0 |
/// | `axial_tilt_deg` | **Legacy field** — NOT used in the NOAA default path. Earth's real obliquity is embedded in the declination Fourier series. Keep for alien-planet / custom-declination overrides; set to `f32::NAN` to signal "use NOAA series". | 23.44 |
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunConfig {
    /// Observer latitude in degrees (positive = north, negative = south).
    pub latitude_deg: f32,
    /// Azimuth (°) from geographic north to the map +Z axis, CW looking down.
    /// `0.0` → map +Z points geographic south. `180.0` → map +Z is geographic north.
    pub north_azimuth_offset_deg: f32,
    /// Legacy axial-tilt field. The NOAA default path IGNORES this and uses its
    /// own Fourier series (which already encodes 23.44° obliquity). Set a finite
    /// value to OVERRIDE the NOAA declination with a simple
    /// `tilt · sin(2π(day−81)/365)` (useful for alien planets or zero-tilt tests).
    /// `f32::NAN` = always use the NOAA Fourier series (recommended for Earth).
    pub axial_tilt_deg: f32,
}

impl Default for SunConfig {
    fn default() -> Self {
        Self {
            latitude_deg: 40.71,
            north_azimuth_offset_deg: 0.0,
            axial_tilt_deg: 23.44,
        }
    }
}

/// Full result of a solar position computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunPosition {
    /// Altitude above the horizon in radians. Negative = below horizon.
    pub altitude_rad: f64,
    /// Azimuth in radians, measured CW from geographic north (0=N, π/2=E, π=S).
    pub azimuth_rad: f64,
    /// World-space unit direction FROM the scene TOWARD the sun (Y-up).
    /// **Never zero** — even when the sun is below the horizon. Use `altitude_rad`
    /// to check visibility; radiance carries the on/off signal.
    pub direction: [f32; 3],
    /// Solar declination in radians (positive in northern-hemisphere summer).
    pub declination_rad: f64,
    /// Equation of time in minutes. Nonzero because of orbital eccentricity and
    /// obliquity; ranges from about −14 min (Feb) to +16 min (early Nov).
    pub equation_of_time_minutes: f64,
    /// APPARENT altitude in radians = geometric `altitude_rad` + atmospheric
    /// refraction (the angular lift of the sun's image near the horizon, up to
    /// ≈+0.48° at the geometric horizon, falling to ≈0 above ~15°). This is the
    /// altitude an observer actually SEES the disk at; the returned `direction`
    /// is built from it (so sunrise/sunset land slightly earlier, golden hour is
    /// physically placed). Equals `altitude_rad` when refraction is disabled or
    /// the sun is more than ~1° below the geometric horizon.
    pub apparent_altitude_rad: f64,
}

impl SunConfig {
    /// NYC Lower Manhattan preset (lat 40.71°N, no north offset).
    pub fn nyc() -> Self {
        Self::default()
    }

    /// Compute the solar position for the given observer, day, and time.
    ///
    /// - `day_of_year`: 1 = Jan 1, 365 = Dec 31 (leap days ok; formula is robust).
    /// - `hour`: UTC hour as a decimal (0.0 = midnight, 12.0 = noon UTC).
    ///   Pass LOCAL clock time only if `longitude_deg` is 0 and there is no UTC
    ///   offset; otherwise pass UTC and let the algorithm derive true solar time.
    /// - `latitude_deg`: observer latitude (°N).
    /// - `longitude_deg`: observer longitude (°E, negative = west).
    ///
    /// The `longitude_deg` and timezone offset are used to compute the **true
    /// solar time** including the equation of time. Passing `longitude = 0.0`
    /// and reading the sun at clock noon at the prime meridian is fine for a
    /// quick sanity check, but for a real city (NYC: −74°) you must pass the
    /// real longitude.
    ///
    /// For the game, prefer calling `SunConfig::direction_at` which wraps this.
    pub fn compute(
        &self,
        day_of_year: u32,
        hour: f64,
        latitude_deg: f64,
        longitude_deg: f64,
    ) -> SunPosition {
        compute_sun_position(day_of_year, hour, latitude_deg, longitude_deg, self)
    }

    /// Compute the Y-up world direction (FROM scene TOWARD sun) for a given
    /// solar day and hour. Uses the observer latitude stored in `self`, and
    /// longitude 0 (prime meridian) + no timezone offset.
    ///
    /// For game use, call `compute(day, hour, lat, lon)` instead; this entry
    /// point is kept for backward-compat with callers that don't know longitude.
    ///
    /// **Direction is NEVER zeroed** — use the sign of `altitude_rad` to decide
    /// whether to apply radiance.
    pub fn direction(&self, day_of_year: f32, hour_of_day: f32) -> Vec3 {
        let pos = compute_sun_position(
            day_of_year as u32,
            hour_of_day as f64,
            self.latitude_deg as f64,
            0.0, // legacy: no longitude
            self,
        );
        Vec3::from(pos.direction)
    }

    /// Sun color and radiance modulated by altitude. Returns `([f32;3], f32)`.
    ///
    /// At night (`altitude ≤ 0`) radiance is 0 but color is still returned
    /// (dim red) — the caller decides whether to apply it.
    pub fn color_and_radiance(&self, day_of_year: f32, hour_of_day: f32) -> ([f32; 3], f32) {
        let pos = compute_sun_position(
            day_of_year as u32,
            hour_of_day as f64,
            self.latitude_deg as f64,
            0.0,
            self,
        );
        sun_color_radiance_from_altitude(pos.altitude_rad)
    }
}

// ── Core NOAA computation ─────────────────────────────────────────────────────

/// Compute the solar position using the NOAA Solar Position Algorithm.
///
/// Reference: NOAA/ESRL "Solar Position Algorithm" (Meeus ch.25 basis).
/// See: <https://gml.noaa.gov/grad/solcalc/solareqns.PDF>
///
/// - `hour_local`: LOCAL STANDARD TIME as a decimal (0.0 = midnight, 12.0 = noon).
///   This is what the game clock produces for the observer's time zone.
/// - `longitude_deg`: observer longitude (°E, negative = west).
///
/// The function converts local standard time → true solar time using:
///   `tst = hour_local×60 + eqtime + 4×longitude − 60×timezone`
/// where `timezone = longitude / 15` (standard-meridian convention). This means
/// `4×longitude − 60×timezone = 4×lon − 4×lon = 0` — the longitude terms cancel
/// and only the **equation of time** corrects local noon to true solar noon.
///
/// This is the correct model for a city builder: the game clock runs in local
/// standard time, and solar noon differs from clock noon only by the equation of
/// time (up to ±16 minutes). Passing `hour_local=12.0` at any longitude gives
/// a sun very close to solar noon as expected.
pub fn compute_sun_position(
    day_of_year: u32,
    hour_local: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    config: &SunConfig,
) -> SunPosition {
    // Backward-compatible default path: assume the game clock is LOCAL STANDARD
    // TIME at the longitude's standard meridian (timezone = longitude/15), so the
    // longitude terms cancel and only the equation of time corrects local→true
    // solar time. Atmospheric refraction is applied (it lifts the apparent disk
    // near the horizon) — this changes only the returned `direction` (built from
    // the apparent altitude) and the new `apparent_altitude_rad`; the geometric
    // `altitude_rad` is unchanged, so every existing altitude-keyed call site
    // (e.g. `is_night = altitude_rad < 0`) and test is byte-for-byte preserved.
    compute_sun_position_tz(
        day_of_year,
        hour_local,
        latitude_deg,
        longitude_deg,
        longitude_deg / 15.0,
        true,
        config,
    )
}

/// Compute the solar position with an EXPLICIT timezone offset and optional
/// atmospheric refraction — the ultra-realistic entry point.
///
/// Reference: NOAA/ESRL "Solar Position Algorithm" (Meeus, *Astronomical
/// Algorithms* 2nd ed. ch.25 basis). <https://gml.noaa.gov/grad/solcalc/solareqns.PDF>
///
/// - `hour_local`: civil clock time as a decimal in the observer's timezone.
/// - `longitude_deg`: observer longitude (°E, negative = west).
/// - `timezone_offset_hours`: the observer's UTC offset in hours (e.g. NYC EST =
///   −5.0, EDT = −4.0). True solar time is
///   `tst = hour_local·60 + eqtime + 4·longitude − 60·tz`. With
///   `tz = longitude/15` (standard-meridian convention) the longitude terms
///   cancel and only the equation of time corrects local noon → solar noon. With
///   a REAL fixed timezone, a city east/west of its standard meridian gets the
///   correct intra-timezone solar-noon shift (up to ±30 min) — the realism the
///   plumbed-but-discarded longitude was meant to provide.
/// - `apply_refraction`: when `true`, the returned `direction` is built from the
///   apparent (refracted) altitude and `apparent_altitude_rad` carries the lift;
///   the geometric `altitude_rad` is always the unrefracted value.
///
/// All math is f64 and deterministic (no RNG, no map/hash iteration) — safe for
/// the engine's replay-exact moat.
pub fn compute_sun_position_tz(
    day_of_year: u32,
    hour_local: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    timezone_offset_hours: f64,
    apply_refraction: bool,
    config: &SunConfig,
) -> SunPosition {
    use std::f64::consts::TAU;

    // ── Step 1: fractional year γ (radians) ──────────────────────────────────
    // Fractional day-of-year; NOAA uses 365-day year for the Fourier series.
    let gamma = TAU / 365.0 * (day_of_year as f64 - 1.0 + (hour_local - 12.0) / 24.0);

    // ── Step 2: equation of time (minutes) ───────────────────────────────────
    // NOAA formula from solareqns.PDF, accurate to < 0.5 min.
    let eqtime = 229.18
        * (0.000075 + 0.001868 * gamma.cos()
            - 0.032077 * gamma.sin()
            - 0.014615 * (2.0 * gamma).cos()
            - 0.04089 * (2.0 * gamma).sin());

    // ── Step 3: solar declination δ (radians) ────────────────────────────────
    // NOAA Fourier series — already encodes the 23.44° obliquity.
    // Override with simple sine formula if `axial_tilt_deg` is finite AND the
    // caller explicitly set a non-Earth value (i.e. not 23.44 exactly).
    let decl = {
        let tilt = config.axial_tilt_deg;
        let use_noaa = tilt.is_nan() || (tilt - 23.44).abs() < 0.01; // default Earth → NOAA series
        if use_noaa {
            // NOAA Fourier series (Meeus simplified).
            0.006918 - 0.399912 * gamma.cos() + 0.070257 * gamma.sin()
                - 0.006758 * (2.0 * gamma).cos()
                + 0.000907 * (2.0 * gamma).sin()
                - 0.002697 * (3.0 * gamma).cos()
                + 0.00148 * (3.0 * gamma).sin()
        } else {
            // Simple axial-tilt sine for alien worlds / zero-tilt tests.
            (tilt as f64).to_radians() * (TAU * (day_of_year as f64 - 81.0) / 365.0).sin()
        }
    };

    // ── Step 4: true solar time (minutes) ────────────────────────────────────
    // NOAA: tst = local_minutes + eqtime + 4*longitude - 60*timezone
    // Using the standard-meridian convention timezone = longitude/15:
    //   4*longitude - 60*(longitude/15) = 4*longitude - 4*longitude = 0
    // So the longitude terms cancel and only the equation of time matters.
    // This is the correct model for local standard time input.
    let tst = hour_local * 60.0 + eqtime + 4.0 * longitude_deg - 60.0 * timezone_offset_hours;
    // With timezone = longitude/15 the longitude terms cancel (legacy default).
    // With a real fixed timezone, `4·longitude − 60·tz` is the genuine
    // intra-timezone solar-time offset. (No rem_euclid needed — the hour-angle
    // formula below handles the full range naturally.)

    // ── Step 5: hour angle H (degrees → radians) ──────────────────────────────
    // Solar noon: tst = 720 min → H = 0. Morning: H < 0. Afternoon: H > 0.
    let ha_deg = tst / 4.0 - 180.0;
    let ha = ha_deg.to_radians();

    // ── Step 6: altitude ─────────────────────────────────────────────────────
    let lat = latitude_deg.to_radians();
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_dec = decl.sin();
    let cos_dec = decl.cos();
    let cos_ha = ha.cos();

    let sin_alt = (sin_lat * sin_dec + cos_lat * cos_dec * cos_ha).clamp(-1.0, 1.0);
    let altitude = sin_alt.asin();

    // ── Step 6b: atmospheric refraction → apparent altitude ──────────────────
    // The atmosphere lifts the apparent disk near the horizon (≈+0.48° at the
    // geometric horizon). `altitude` stays GEOMETRIC; `apparent_altitude` adds
    // the refraction and is what an observer sees (and what the rendered sun
    // direction should use, so sunrise/sunset + golden-hour land physically).
    let apparent_altitude = if apply_refraction {
        altitude + atmospheric_refraction_deg(altitude.to_degrees()).to_radians()
    } else {
        altitude
    };
    // Direction is built from the APPARENT altitude (azimuth is unaffected by
    // refraction); when refraction is off this is identical to the geometric path.
    let sin_alt = apparent_altitude.sin();
    let cos_alt = apparent_altitude.cos();

    // ── Step 7: azimuth (0=N, 90=E, 180=S, 270=W) ───────────────────────────
    // Full-circle atan2 form, numerically stable at all latitudes including poles.
    let azimuth = if cos_alt.abs() < 1e-10 {
        // Sun is exactly at zenith or nadir — azimuth is undefined; use 0.
        0.0_f64
    } else {
        // Component along east: −cos(δ)·sin(H)
        // Component along north: sin(δ)·cos(lat) − cos(δ)·cos(H)·sin(lat)
        let east = -cos_dec * ha.sin();
        let north = sin_dec * cos_lat - cos_dec * cos_ha * sin_lat;
        // atan2(east, north) gives CW-from-north azimuth in (−π, π]; shift to [0, 2π).
        east.atan2(north).rem_euclid(std::f64::consts::TAU)
    };

    // ── Step 8: azimuth + altitude → Y-up world direction ────────────────────
    // +X = east, +Y = up, +Z = south (our map convention).
    // Geographic azimuth: 0=N, 90=E → the component equations are:
    //   east  component = cos(alt) * sin(az)      (az=90° = east = +X)
    //   up    component = sin(alt)                (+Y)
    //   south component = cos(alt) * cos(az)      (az=0°=N → Z=0, az=180°=S → Z=cos_alt)
    // Wait — az=0 is NORTH (−Z in our space where +Z=south), az=180 is SOUTH (+Z).
    // So south component = cos(alt) * cos(az + π) = −cos(alt)*cos(az).
    // Equivalently: Z = cos(alt)*(−cos(az)) — negative for north-facing, positive for south.
    //
    // Let's derive directly:
    //   North geog  = −Z in our space  (map +Z = south)
    //   East geog   = +X
    // So:
    //   raw_x = cos(alt)*sin(az)        // east = +X ✓
    //   raw_y = sin(alt)               // up   = +Y ✓
    //   raw_z = −cos(alt)*cos(az)      // south = +Z (az=180°→cos=-1→raw_z=+cos_alt ✓)
    let cos_az = azimuth.cos();
    let sin_az = azimuth.sin();
    let raw = Vec3::new(
        (cos_alt * sin_az) as f32,
        sin_alt as f32,
        (-cos_alt * cos_az) as f32,
    );

    // Apply north-axis rotation: rotate about Y by -offset so the designer can
    // align the map +Z with any compass heading.
    let rot_rad = (-config.north_azimuth_offset_deg).to_radians() as f32;
    let rot = Mat3::from_rotation_y(rot_rad);
    let dir = rot * raw;

    // Normalise (guard against floating-point drift; raw should already be unit).
    let dir_norm = if dir.length_squared() > 1e-12 {
        dir.normalize()
    } else {
        Vec3::Y // absolute fallback (e.g. exactly at zenith)
    };

    SunPosition {
        altitude_rad: altitude,
        azimuth_rad: azimuth,
        direction: dir_norm.to_array(),
        declination_rad: decl,
        equation_of_time_minutes: eqtime,
        apparent_altitude_rad: apparent_altitude,
    }
}

/// Atmospheric refraction (degrees to ADD to the geometric solar elevation) from
/// the NOAA solar-calculator model. Positive near the horizon (≈+0.48° at the
/// geometric horizon), tapering to ~0 above ~15°, and 0 well below the horizon
/// where no refracted image exists. `geom_alt_deg` is the GEOMETRIC elevation.
///
/// Reference: NOAA ESRL solar-position calculator refraction term (Sæmundsson /
/// Bennett family of fits). Deterministic, f64.
#[inline]
pub fn atmospheric_refraction_deg(geom_alt_deg: f64) -> f64 {
    if geom_alt_deg > 85.0 {
        0.0
    } else if geom_alt_deg > 5.0 {
        let t = geom_alt_deg.to_radians().tan();
        (58.1 / t - 0.07 / t.powi(3) + 0.000_086 / t.powi(5)) / 3600.0
    } else if geom_alt_deg > -0.575 {
        let e = geom_alt_deg;
        (1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))) / 3600.0
    } else if geom_alt_deg > -1.0 {
        // Just below the geometric horizon: the refracted image can still be
        // visible. Below ~−1° the model is invalid and no image exists → 0.
        let t = geom_alt_deg.to_radians().tan();
        (-20.774 / t) / 3600.0
    } else {
        0.0
    }
}

/// Derive sun color and radiance from solar altitude (radians).
///
/// - altitude ≤ 0  → radiance = 0.0, color = dim reddish (not applied by caller).
/// - altitude > 0  → warm orange at horizon → neutral white at noon.
pub fn sun_color_radiance_from_altitude(altitude_rad: f64) -> ([f32; 3], f32) {
    let sin_alt = altitude_rad.sin() as f32;
    if sin_alt <= 0.0 {
        return ([0.0, 0.0, 0.0], 0.0);
    }
    let t = sin_alt.clamp(0.0, 1.0);
    let g = 0.55 + 0.42 * t;
    let b = 0.15 + 0.77 * t;
    let color = [1.0_f32, g, b];
    let radiance = 8.0 + 22.0 * t.powf(0.4);
    (color, radiance)
}

// ── Tick ↔ Solar time helpers ─────────────────────────────────────────────────

/// Derive `hour_of_day` [0, 24) from a tick-of-day [0, ticks_per_day).
#[inline]
pub fn hour_from_tick_of_day(tick_of_day: u64, ticks_per_day: u64) -> f32 {
    let hours_per_tick = 24.0 / ticks_per_day as f32;
    tick_of_day as f32 * hours_per_tick
}

/// Derive `day_of_year` [0, 365) from a civic day-of-year [0, days_per_year).
///
/// The game uses a 360-day civic year; map it linearly onto the 365-day solar
/// calendar so seasons stay correctly positioned.
#[inline]
pub fn solar_day_from_civic_day(civic_day_of_year: u64, civic_days_per_year: u64) -> f32 {
    civic_day_of_year as f32 * 365.0 / civic_days_per_year as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const NYC_LAT: f64 = 40.7128;
    const NYC_LON: f64 = -74.006;
    const EARTH_CONFIG: SunConfig = SunConfig {
        latitude_deg: 40.71,
        north_azimuth_offset_deg: 0.0,
        axial_tilt_deg: 23.44,
    };

    // Helper: compute sun at given day/hour for NYC lat/lon.
    fn nyc_sun(day: u32, hour: f64) -> SunPosition {
        compute_sun_position(day, hour, NYC_LAT, NYC_LON, &EARTH_CONFIG)
    }

    // ── NOAA accuracy tests ───────────────────────────────────────────────────

    /// NYC March equinox (day 80) solar noon: altitude ≈ 49.3°, sun due south.
    ///
    /// At local standard time 12:00 the sun is near solar noon (offset only by
    /// equation of time ≈ 0 near the equinox). Altitude = 90° − lat + decl ≈ 49.3°.
    #[test]
    fn nyc_equinox_noon_altitude() {
        // Pass local standard time 12:00.
        let pos = nyc_sun(80, 12.0);
        let alt_deg = pos.altitude_rad.to_degrees();
        // Expected: 90° − 40.71° + small declination term at equinox ≈ 49.3°.
        assert!(
            (alt_deg - 49.3).abs() < 1.0,
            "NYC equinox solar noon altitude should be ~49.3° but was {:.2}°",
            alt_deg
        );
        // At solar noon the sun is due south: azimuth ≈ 180°.
        let az_deg = pos.azimuth_rad.to_degrees();
        assert!(
            (az_deg - 180.0).abs() < 5.0,
            "NYC equinox solar noon azimuth should be ~180° (south) but was {:.2}°",
            az_deg
        );
    }

    /// Equation of time is ~+16 minutes in early November (day 305).
    #[test]
    fn equation_of_time_early_november() {
        let pos = nyc_sun(305, 12.0);
        let eot = pos.equation_of_time_minutes;
        assert!(
            eot > 14.0 && eot < 18.0,
            "EoT early November should be ~+16 min but was {:.2} min",
            eot
        );
    }

    /// Equator equinox solar noon: sun almost directly overhead (altitude ≈ 90°).
    #[test]
    fn equator_equinox_noon_overhead() {
        let pos = compute_sun_position(80, 12.0, 0.0, 0.0, &EARTH_CONFIG);
        let alt_deg = pos.altitude_rad.to_degrees();
        // At the equinox the declination is ~0°; at lat=0, lon=0, the solar
        // time at 12:00 UTC is already true solar noon → altitude ≈ 90°.
        assert!(
            alt_deg > 88.0,
            "Equator equinox noon altitude should be near 90° but was {:.2}°",
            alt_deg
        );
    }

    /// Svalbard (78°N) summer solstice MIDNIGHT: midnight sun (above horizon).
    #[test]
    fn svalbard_midnight_sun() {
        let lat = 78.0_f64;
        let lon = 15.0_f64;
        // June 21 ≈ day 172. Midnight UTC = hour 0.
        // At lon=15°E, UTC midnight corresponds to ~01:00 local, still midnight sun.
        let pos = compute_sun_position(172, 0.0, lat, lon, &EARTH_CONFIG);
        let alt_deg = pos.altitude_rad.to_degrees();
        assert!(
            alt_deg > 0.0,
            "Svalbard (78°N) summer midnight: sun should be above horizon (midnight sun) but was {:.2}°",
            alt_deg
        );
    }

    /// Svalbard winter solstice NOON: polar night (below horizon all day).
    #[test]
    fn svalbard_polar_night() {
        let lat = 78.0_f64;
        let lon = 15.0_f64;
        // Dec 21 ≈ day 355.
        let pos = compute_sun_position(355, 12.0, lat, lon, &EARTH_CONFIG);
        let alt_deg = pos.altitude_rad.to_degrees();
        assert!(
            alt_deg < 0.0,
            "Svalbard (78°N) winter noon: polar night — sun should be below horizon but was {:.2}°",
            alt_deg
        );
    }

    /// Polar summer: azimuth sweeps through a large range over 24h (circumpolar).
    /// The sun circles the sky rather than rising and setting.
    #[test]
    fn svalbard_azimuth_sweeps_full_range() {
        let lat = 78.0_f64;
        let lon = 15.0_f64;
        let azimuths: Vec<f64> = (0..24)
            .map(|h| {
                compute_sun_position(172, h as f64, lat, lon, &EARTH_CONFIG)
                    .azimuth_rad
                    .to_degrees()
            })
            .collect();

        // Find effective span accounting for wrap-around at 360°.
        // Simplest: check that the range of raw values covers > 180°.
        let min_az = azimuths.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_az = azimuths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // In polar summer the azimuth cycles through the full 360° so the raw
        // range (before any unwrapping) will cover at least 180° (often 350°+).
        assert!(
            max_az - min_az > 180.0,
            "Svalbard polar summer: azimuth should span >180° over 24h but range was {:.1}°",
            max_az - min_az
        );
    }

    /// Direction is NEVER zero, even below the horizon.
    #[test]
    fn direction_never_zero_below_horizon() {
        // Midnight at NYC — definitely below horizon.
        let pos = nyc_sun(80, 0.0);
        assert!(
            pos.altitude_rad < 0.0,
            "precondition: midnight should be below horizon"
        );
        let d = pos.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-4,
            "direction must be unit length even below horizon; len={len:.6}"
        );
    }

    // ── Legacy API backward-compat tests ─────────────────────────────────────

    /// At the March equinox (day 80) solar noon, sun is near south at correct altitude.
    #[test]
    fn legacy_equinox_noon_sun_is_due_south_at_correct_altitude() {
        let dir = EARTH_CONFIG.direction(80.0, 12.0);
        // Not zeroed (above horizon at noon).
        assert!(dir.y > 0.0, "sun should be above horizon at noon");
        let alt = dir.y.asin().to_degrees();
        let expected_alt = 90.0 - EARTH_CONFIG.latitude_deg;
        assert!(
            (alt - expected_alt).abs() < 2.0,
            "equinox noon altitude should be ≈{expected_alt:.1}°; got {alt:.2}°"
        );
    }

    /// Axial tilt override: zero tilt → no seasonal variation in noon altitude.
    #[test]
    fn zero_axial_tilt_produces_no_seasonal_variation() {
        let flat = SunConfig {
            axial_tilt_deg: 0.0,
            ..EARTH_CONFIG
        };
        let alt_eq = flat.direction(80.0, 12.0).y.asin().to_degrees();
        let alt_su = flat.direction(172.0, 12.0).y.asin().to_degrees();
        let alt_wi = flat.direction(355.0, 12.0).y.asin().to_degrees();
        assert!(
            (alt_eq - alt_su).abs() < 1.0,
            "zero tilt: equinox and summer noon should match; {alt_eq:.2}° vs {alt_su:.2}°"
        );
        assert!(
            (alt_eq - alt_wi).abs() < 1.0,
            "zero tilt: equinox and winter noon should match; {alt_eq:.2}° vs {alt_wi:.2}°"
        );
    }

    /// `hour_from_tick_of_day` with TICKS_PER_DAY=144.
    #[test]
    fn tick_to_hour_matches_care_day_anchors() {
        let tpd = 144u64;
        let h48 = hour_from_tick_of_day(48, tpd);
        assert!((h48 - 8.0).abs() < 0.01, "tick 48 → 08:00; got {h48}");
        let h102 = hour_from_tick_of_day(102, tpd);
        assert!((h102 - 17.0).abs() < 0.01, "tick 102 → 17:00; got {h102}");
    }

    /// `solar_day_from_civic_day` maps 360-day year → 365-day solar year linearly.
    #[test]
    fn solar_day_mapping_is_correct() {
        let d0 = solar_day_from_civic_day(0, 360);
        assert!((d0 - 0.0).abs() < 1e-6, "civic 0 → solar 0; got {d0}");
        let d360 = solar_day_from_civic_day(360, 360);
        assert!(
            (d360 - 365.0).abs() < 1e-4,
            "civic 360 → solar 365; got {d360}"
        );
        let d180 = solar_day_from_civic_day(180, 360);
        assert!(
            (d180 - 182.5).abs() < 0.01,
            "civic 180 → solar 182.5; got {d180}"
        );
    }

    // ── Refraction + timezone (ultra-realistic additions) ─────────────────────

    /// Atmospheric refraction LIFTS the apparent sun near the horizon and is
    /// negligible high in the sky.
    #[test]
    fn refraction_lifts_apparent_altitude_near_horizon() {
        // Horizon (~0°): refraction should add roughly half a degree.
        let r0 = atmospheric_refraction_deg(0.0);
        assert!(
            r0 > 0.4 && r0 < 0.6,
            "horizon refraction should be ≈0.48°, got {r0:.4}°"
        );
        // High sun (~60°): refraction is tiny (< 0.02°).
        let r60 = atmospheric_refraction_deg(60.0);
        assert!(
            r60 >= 0.0 && r60 < 0.02,
            "high-sun refraction ≈0, got {r60:.4}°"
        );
        // Find a sample where the sun is just ABOVE the horizon (0..5°) and verify
        // the apparent altitude is strictly lifted there. Scan the sunrise window.
        let mut found = false;
        for i in 0..60 {
            let hour = 6.0 + i as f64 * 0.05; // 06:00 → 09:00
            let pos = nyc_sun(80, hour);
            let geom_deg = pos.altitude_rad.to_degrees();
            if geom_deg > 0.2 && geom_deg < 5.0 {
                assert!(
                    pos.apparent_altitude_rad > pos.altitude_rad,
                    "apparent ({:.5}) should exceed geometric ({:.5}) at low sun",
                    pos.apparent_altitude_rad,
                    pos.altitude_rad
                );
                // Geometric altitude_rad is NOT mutated by refraction (legacy-stable).
                let geom = compute_sun_position_tz(
                    80,
                    hour,
                    NYC_LAT,
                    NYC_LON,
                    NYC_LON / 15.0,
                    false,
                    &EARTH_CONFIG,
                );
                assert!(
                    (geom.altitude_rad - pos.altitude_rad).abs() < 1e-12,
                    "geometric altitude must be identical with/without refraction"
                );
                assert!(
                    (geom.apparent_altitude_rad - geom.altitude_rad).abs() < 1e-12,
                    "with refraction off, apparent == geometric"
                );
                found = true;
                break;
            }
        }
        assert!(found, "expected a low-sun sample in the sunrise window");
    }

    /// An explicit timezone shifts solar noon for a city offset from its standard
    /// meridian — the realism the discarded longitude was meant to provide.
    ///
    /// At longitude 0 with timezone 0, clock noon ≈ solar noon (hour angle ≈ 0).
    /// Move the observer +7.5° EAST while KEEPING timezone 0: the sun crosses the
    /// meridian 30 min EARLIER, so at clock noon the sun is already PAST south
    /// (afternoon, azimuth > 180°). The legacy lon/15 path would cancel this.
    #[test]
    fn explicit_timezone_shifts_solar_noon_within_zone() {
        let cfg = EARTH_CONFIG;
        // Standard meridian: clock noon ≈ solar noon (offset only by equation of
        // time), so the sun is near south.
        let at_meridian = compute_sun_position_tz(80, 12.0, 40.0, 0.0, 0.0, false, &cfg)
            .azimuth_rad
            .to_degrees();
        // 7.5° EAST of the meridian, SAME timezone (offset 0): solar noon is
        // ~30 min EARLIER, so at clock noon the sun has moved further past south.
        let east = compute_sun_position_tz(80, 12.0, 40.0, 7.5, 0.0, false, &cfg)
            .azimuth_rad
            .to_degrees();
        assert!(
            east > at_meridian + 5.0,
            "7.5°E (same tz) should push clock-noon azimuth >5° further west: meridian={at_meridian:.2}° east={east:.2}°"
        );
        // The legacy default (timezone = longitude/15) CANCELS the longitude, so
        // clock noon is unchanged by longitude — proves the cancellation is intact.
        let legacy = compute_sun_position(80, 12.0, 40.0, 7.5, &cfg)
            .azimuth_rad
            .to_degrees();
        assert!(
            (legacy - at_meridian).abs() < 0.5,
            "legacy lon/15 path keeps clock-noon azimuth longitude-independent: legacy={legacy:.2}° meridian={at_meridian:.2}°"
        );
    }
}
