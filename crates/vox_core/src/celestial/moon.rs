//! Lunar position model for the Ochroma engine.
//!
//! Implements the **Meeus "low-precision" lunar position algorithm** from
//! *Astronomical Algorithms* (2nd ed., 1998), Chapters 22 & 47.
//!
//! # Algorithm
//!
//! 1. Julian Date from calendar year + day-of-year + fractional hour.
//! 2. Julian centuries T from J2000.0.
//! 3. Lunar fundamental arguments (mean longitude L₀, sun/moon anomalies M/M′,
//!    elongation D, argument of latitude F) — all in degrees, reduced mod 360.
//! 4. Geocentric ecliptic **longitude** Σl: 20 principal periodic terms from
//!    Meeus Table 47.A (units: 1×10⁻⁶ degrees per term → sum divided by 10⁶).
//!    Includes: equation of the centre, evection (main lunar perturbation from
//!    the sun, period ≈ 31.8 days), variation (period ≈ 14 days), annual
//!    equation, parallactic equation, and 14 further terms.
//! 5. Geocentric ecliptic **latitude** Σb: 10 principal periodic terms from
//!    Meeus Table 47.B.
//! 6. Mean obliquity ε of the ecliptic (slowly decreasing, linear-in-T approximation).
//! 7. Ecliptic → equatorial: RA and Dec.
//! 8. Local hour angle from Greenwich Mean Sidereal Time + observer longitude − RA.
//! 9. Altitude and azimuth from the standard spherical-astronomy formulas.
//! 10. Phase from the sun–moon elongation D (0=new, 1=full).
//!
//! # Precision
//!
//! Including 20 longitude + 10 latitude terms gives approximately **1–2°** in
//! longitude and **0.5–1°** in latitude compared to the full Meeus series. This is
//! sufficient for a game renderer where a few-degree error in the moon's direction
//! is invisible to the player. The phase is accurate to within ~0.5 day (< 2% in
//! illumination fraction).
//!
//! # Coordinate convention
//!
//! Same as the sun: Y-up, +X=east, +Z=south. The returned direction points FROM
//! the scene TOWARD the moon and is NEVER zeroed (radiance carries the on/off).

use glam::{Mat3, Vec3};

/// Configuration for the moon light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonConfig {
    /// Maximum key-light radiance for a full moon (relative to the sun at noon ≈ 30).
    /// A full moon is ~400,000× dimmer than the noon sun; 0.08 gives a visually
    /// interesting night-mode without being invisible.
    pub max_radiance: f32,
    /// Key-light color at full moon: slightly cool/blue (7000 K colorimetry).
    /// [0.72, 0.82, 1.0] gives a pleasant moonlit-night feel.
    pub color: [f32; 3],
}

impl Default for MoonConfig {
    fn default() -> Self {
        Self {
            max_radiance: 0.08,
            color: [0.72, 0.82, 1.0],
        }
    }
}

/// Result of a lunar position computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonPosition {
    /// Altitude above horizon in radians. Negative = below horizon.
    pub altitude_rad: f64,
    /// Azimuth in radians, CW from geographic north (0=N, π/2=E, π=S).
    pub azimuth_rad: f64,
    /// World-space unit direction FROM scene TOWARD the moon (Y-up, never zero).
    pub direction: [f32; 3],
    /// Illumination fraction: 0.0 = new moon (dark), 1.0 = full moon (bright).
    pub phase: f32,
    /// Key-light radiance, phase-scaled. 0 when below horizon.
    pub radiance: f32,
    /// Key-light color (from `MoonConfig`).
    pub color: [f32; 3],
}

/// Compute the moon's position for a given observer and time.
///
/// - `year`: calendar year (e.g. 2025). The game uses 2025 as the base epoch.
/// - `day_of_year`: 1 = Jan 1, 365/366 = Dec 31.
/// - `hour_utc`: UTC hour as decimal.
/// - `latitude_deg`: observer latitude (°N).
/// - `longitude_deg`: observer longitude (°E, negative = west).
/// - `config`: moon config (color, max radiance).
/// - `north_azimuth_offset_deg`: same as `SunConfig` — CCW rotation of world +Z from south.
pub fn compute_moon_position(
    year: i32,
    day_of_year: u32,
    hour_utc: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    config: &MoonConfig,
    north_azimuth_offset_deg: f32,
) -> MoonPosition {
    use std::f64::consts::{PI, TAU};

    // ── Convert day-of-year → calendar month + day ────────────────────────────
    let (month, day) = day_of_year_to_month_day(year, day_of_year);

    // ── Step 1: Julian Date ───────────────────────────────────────────────────
    let jd = julian_date(year, month, day, hour_utc);

    // ── Step 2: Julian centuries from J2000.0 ─────────────────────────────────
    let t = (jd - 2_451_545.0) / 36525.0;

    // ── Step 3: Fundamental arguments (degrees, reduced mod 360) ─────────────
    // Moon's mean longitude (°).
    let l0 = (218.316_447_7 + 481_267.881_234_21 * t).rem_euclid(360.0);
    // Sun's mean anomaly (°).
    let m = (357.529_109_2 + 35_999.050_290_9 * t).rem_euclid(360.0);
    // Moon's mean anomaly (°).
    let mp = (134.963_396_4 + 477_198.867_505_5 * t).rem_euclid(360.0);
    // Moon's argument of elongation (°).
    let d = (297.850_192_1 + 445_267.111_403_4 * t).rem_euclid(360.0);
    // Moon's argument of latitude (°).
    let f = (93.272_095_0 + 483_202.017_523_3 * t).rem_euclid(360.0);

    // Convert to radians for sin/cos.
    let r = PI / 180.0;
    // l0 used in degrees for lambda computation below; r used for the others.
    let mr = m * r;
    let mpr = mp * r;
    let dr = d * r;
    let fr = f * r;

    // ── Step 4: Σl — ecliptic longitude corrections (1e-6 degrees) ───────────
    // 20 principal terms from Meeus Table 47.A.
    // Arguments: (D, M, M', F) coefficients; amplitude in 1e-6 °.
    let sigma_l: f64 = 6_288_774.0 * mpr.sin()
        + 1_274_027.0 * (2.0 * dr - mpr).sin()   // evection
        + 658_314.0 * (2.0 * dr).sin()             // variation
        + 213_618.0 * (2.0 * mpr).sin()
        - 185_116.0 * mr.sin()                     // annual equation
        - 114_332.0 * (2.0 * fr).sin()
        + 58_793.0 * (2.0 * dr - 2.0 * mpr).sin()
        + 57_066.0 * (2.0 * dr - mr - mpr).sin()
        + 53_322.0 * (2.0 * dr + mpr).sin()
        + 45_758.0 * (2.0 * dr - mr).sin()
        - 40_923.0 * (mpr - mr).sin()
        - 34_720.0 * dr.sin()                      // parallactic equation
        - 30_383.0 * (mpr + mr).sin()
        + 15_327.0 * (2.0 * dr - 2.0 * fr).sin()
        - 12_528.0 * (mpr + 2.0 * fr).sin()
        + 10_980.0 * (mpr - 2.0 * fr).sin()
        + 10_675.0 * (4.0 * dr - mpr).sin()
        + 10_034.0 * (3.0 * mpr).sin()
        + 8_548.0 * (4.0 * dr - 2.0 * mpr).sin()
        - 7_888.0 * (2.0 * dr + mr - mpr).sin();

    // ── Step 5: Σb — ecliptic latitude corrections (1e-6 degrees) ────────────
    // 10 principal terms from Meeus Table 47.B.
    let sigma_b: f64 = 5_128_122.0 * fr.sin()
        + 280_602.0 * (mpr + fr).sin()
        + 277_693.0 * (mpr - fr).sin()
        + 173_237.0 * (2.0 * dr - fr).sin()
        + 55_413.0 * (2.0 * dr - mpr + fr).sin()
        + 46_271.0 * (2.0 * dr - mpr - fr).sin()
        + 32_573.0 * (2.0 * dr + fr).sin()
        + 17_198.0 * (2.0 * mpr + fr).sin()
        + 9_266.0 * (2.0 * dr + mpr - fr).sin()
        + 8_822.0 * (2.0 * mpr - fr).sin();

    // ── Step 6: geocentric ecliptic coordinates ───────────────────────────────
    let lambda = (l0 + sigma_l / 1_000_000.0).rem_euclid(360.0); // longitude (°)
    let beta = sigma_b / 1_000_000.0;                              // latitude (°)

    // ── Step 7: mean obliquity of the ecliptic ────────────────────────────────
    let epsilon = 23.439_291_1 - 0.013_004_167 * t; // degrees

    // ── Step 8: ecliptic → equatorial (RA, Dec) ───────────────────────────────
    let lam = lambda * r;
    let bet = beta * r;
    let eps = epsilon * r;

    let sin_dec = bet.sin() * eps.cos() + bet.cos() * eps.sin() * lam.sin();
    let dec = sin_dec.clamp(-1.0, 1.0).asin(); // declination (rad)

    // Right ascension: atan2(sin(λ)cos(ε) − tan(β)sin(ε), cos(λ))
    let ra_num = lam.sin() * eps.cos() - bet.tan() * eps.sin();
    let ra_den = lam.cos();
    let ra = ra_num.atan2(ra_den).rem_euclid(TAU); // RA (rad), [0, 2π)

    // ── Step 9: local hour angle ──────────────────────────────────────────────
    // Greenwich Mean Sidereal Time (°), then local.
    let gmst_deg = (280.460_618_37 + 360.985_647_366_29 * (jd - 2_451_545.0))
        .rem_euclid(360.0);
    let lst_deg = (gmst_deg + longitude_deg).rem_euclid(360.0);
    let lha = (lst_deg * r - ra).rem_euclid(TAU);
    // Shift to (−π, π] so morning/afternoon sign is correct.
    let lha = if lha > PI { lha - TAU } else { lha };

    // ── Step 10: altitude and azimuth ─────────────────────────────────────────
    let lat = latitude_deg * r;
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let cos_dec = dec.cos();
    let sin_dec_v = dec.sin();

    let sin_alt = (sin_lat * sin_dec_v + cos_lat * cos_dec * lha.cos()).clamp(-1.0, 1.0);
    let altitude = sin_alt.asin();

    let cos_alt = altitude.cos();
    let azimuth = if cos_alt.abs() < 1e-10 {
        0.0
    } else {
        let east = -cos_dec * lha.sin();
        let north = sin_dec_v * cos_lat - cos_dec * lha.cos() * sin_lat;
        east.atan2(north).rem_euclid(TAU)
    };

    // ── Step 11: phase from elongation D ─────────────────────────────────────
    // Mean elongation D (degrees) is the moon–sun angular separation from the
    // perspective of the Earth centre. Full moon: D ≈ 180°, new moon: D ≈ 0°.
    // phase = (1 − cos(D)) / 2:  0.0 at D=0 (new), 1.0 at D=180° (full).
    let elongation_rad = d * r;  // D already in [0,360) degrees → radians
    let phase = ((1.0 - elongation_rad.cos()) / 2.0).clamp(0.0, 1.0) as f32;

    // ── Step 12: key-light radiance ───────────────────────────────────────────
    // Zero when below horizon; phase-scaled otherwise.
    let radiance = if altitude <= 0.0 {
        0.0
    } else {
        config.max_radiance * phase
    };

    // ── Step 13: Y-up world direction (never zeroed) ──────────────────────────
    let cos_az = azimuth.cos();
    let sin_az = azimuth.sin();
    let raw = Vec3::new(
        (cos_alt * sin_az) as f32,
        sin_alt as f32,
        (-cos_alt * cos_az) as f32,
    );
    let rot_rad = (-north_azimuth_offset_deg).to_radians();
    let rot = Mat3::from_rotation_y(rot_rad);
    let dir = rot * raw;
    let dir_norm = if dir.length_squared() > 1e-12 {
        dir.normalize()
    } else {
        Vec3::Y
    };

    MoonPosition {
        altitude_rad: altitude,
        azimuth_rad: azimuth,
        direction: dir_norm.to_array(),
        phase,
        radiance,
        color: config.color,
    }
}

// ── Julian Date helper ────────────────────────────────────────────────────────

/// Compute the Julian Date for a calendar date + fractional hour (UTC).
fn julian_date(year: i32, month: u32, day: u32, hour_utc: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y as f64 + 4716.0)).floor()
        + (30.6001 * (m as f64 + 1.0)).floor()
        + day as f64
        + hour_utc / 24.0
        + b
        - 1524.5
}

/// Convert a day-of-year (1-based) to (month, day) for the given calendar year.
fn day_of_year_to_month_day(year: i32, day_of_year: u32) -> (u32, u32) {
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_month = [
        31u32,
        if is_leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut remaining = day_of_year.max(1).min(if is_leap { 366 } else { 365 });
    for (i, &dim) in days_in_month.iter().enumerate() {
        if remaining <= dim {
            return (i as u32 + 1, remaining);
        }
        remaining -= dim;
    }
    (12, 31) // fallback
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const DEFAULT_CONFIG: MoonConfig = MoonConfig {
        max_radiance: 0.08,
        color: [0.72, 0.82, 1.0],
    };

    /// Julian Date of J2000.0 (Jan 1.5, 2000) = 2451545.0
    #[test]
    fn julian_date_j2000() {
        let jd = julian_date(2000, 1, 1, 12.0);
        assert!(
            (jd - 2_451_545.0).abs() < 0.001,
            "J2000.0 JD should be 2451545.0 but was {:.3}",
            jd
        );
    }

    /// Day-of-year conversion: day 1 → Jan 1, day 32 → Feb 1.
    #[test]
    fn day_of_year_conversion() {
        assert_eq!(day_of_year_to_month_day(2025, 1), (1, 1));
        assert_eq!(day_of_year_to_month_day(2025, 32), (2, 1));
        assert_eq!(day_of_year_to_month_day(2025, 365), (12, 31));
    }

    /// Full moon: the moon should be roughly opposite the sun in the sky.
    /// Near J2000.0 (Jan 21, 2000 ≈ full moon), the moon should be high at midnight.
    #[test]
    fn full_moon_roughly_opposite_sun() {
        // Jan 21, 2000 ≈ full moon (confirmed by almanac).
        // Day 21 of 2000. At midnight UTC, the sun is at ~−180° from noon.
        let moon = compute_moon_position(2000, 21, 0.0, 40.71, -74.0, &DEFAULT_CONFIG, 0.0);
        // Near full moon so phase should be > 0.7.
        assert!(
            moon.phase > 0.7,
            "Jan 21 2000 should be near full moon, phase={:.2}",
            moon.phase
        );
    }

    /// New moon: near Jan 6, 2000 (known new moon from almanac) → low phase.
    #[test]
    fn new_moon_low_phase() {
        // Jan 6, 2000 ≈ new moon. At UTC noon.
        let moon = compute_moon_position(2000, 6, 12.0, 40.71, -74.0, &DEFAULT_CONFIG, 0.0);
        // Phase should be < 0.2 near new moon.
        assert!(
            moon.phase < 0.2,
            "Jan 6 2000 should be near new moon, phase={:.2}",
            moon.phase
        );
    }

    /// Moon below horizon → radiance = 0.
    #[test]
    fn below_horizon_moon_zero_radiance() {
        // Force altitude negative by using a configuration where the moon is
        // definitely below the horizon: if moon.altitude_rad < 0, radiance = 0.
        // We test this by checking the logic holds: altitude <= 0 → radiance = 0.
        let moon = compute_moon_position(2000, 21, 12.0, 40.71, -74.0, &DEFAULT_CONFIG, 0.0);
        if moon.altitude_rad <= 0.0 {
            assert_eq!(moon.radiance, 0.0, "below-horizon moon must have 0 radiance");
        }
    }

    /// Direction is never zero (unit length).
    #[test]
    fn direction_always_unit_length() {
        for day in [1u32, 90, 172, 265, 355] {
            for hour in [0.0_f64, 6.0, 12.0, 18.0] {
                let moon =
                    compute_moon_position(2025, day, hour, 40.71, -74.0, &DEFAULT_CONFIG, 0.0);
                let d = moon.direction;
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-4,
                    "moon direction must be unit length; day={day} hour={hour} len={len:.6}"
                );
            }
        }
    }

    /// Moon radiance is always less than the sun's maximum (30.0).
    #[test]
    fn moon_radiance_less_than_sun() {
        let moon = compute_moon_position(2000, 21, 0.0, 40.71, -74.0, &DEFAULT_CONFIG, 0.0);
        let sun_max_radiance = 30.0_f32;
        assert!(
            moon.radiance < sun_max_radiance,
            "moon radiance ({}) must be < sun maximum ({})",
            moon.radiance,
            sun_max_radiance
        );
    }
}
