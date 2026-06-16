//! Real celestial Sun model for the Ochroma engine.
//!
//! Models the sun's diurnal arc across the sky using standard solar geometry:
//! declination (day-of-year + axial tilt), hour angle (solar time), and the
//! observer's latitude + north-axis orientation. The sun traces a circular arc
//! tilted by (90° − latitude) from horizontal, offset by solar declination.
//!
//! # Coordinate system
//!
//! Engine is Y-up. The returned direction vector points FROM the scene TOWARD
//! the sun (the convention the megakernel uses: `dot(normal, u_sun_direction)`).
//!
//! World axes (matches urban_horizon map convention):
//! - **+X** = east
//! - **+Y** = up
//! - **+Z** = south  (so −Z = north by default; rotated by `north_azimuth_offset_deg`)
//!
//! # Altitude/azimuth → Y-up world direction
//!
//! ```text
//! az  = azimuth measured CW from geographic north (standard solar convention)
//! alt = altitude above horizon
//!
//! raw_x =  cos(alt) * sin(az)       // east component  (az=90° = east)
//! raw_y =  sin(alt)                 // up component
//! raw_z =  cos(alt) * cos(az)       // south component (az=0 = geographic south = +Z)
//! ```
//!
//! Then the whole vector is rotated about Y by `north_azimuth_offset_deg` so the
//! designer can align the map's +Z with any compass direction.

use glam::{Vec3, Mat3};

/// Configuration for the engine sun. All angular fields are in degrees for
/// readability; internally converted to radians on use.
///
/// # Adjustable fields
///
/// | Field | Meaning | Default |
/// |---|---|---|
/// | `latitude_deg` | Observer latitude (°N positive) | 40.71 (NYC) |
/// | `north_azimuth_offset_deg` | Compass bearing (°) of the map's +Z axis from true geographic north. 0 = map +Z is south (standard), 90 = map +Z is west, −90 = map +Z is east | 0.0 |
/// | `axial_tilt_deg` | Earth's axial tilt used for declination (23.44° Earth, adjustable for alien skies or exaggerated seasons) | 23.44 |
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunConfig {
    /// Observer latitude in degrees (positive = north, negative = south).
    /// NYC Lower Manhattan ≈ 40.71°N.
    pub latitude_deg: f32,
    /// Azimuth (°) from geographic north to the map's +Z axis, measured CW
    /// looking down (standard map bearing convention). `0.0` means the map's
    /// +Z axis points geographic south (so −Z = geographic north). Set to 180°
    /// if the map's +Z IS geographic north (e.g., typical "north-up" maps).
    pub north_azimuth_offset_deg: f32,
    /// Earth's axial tilt for declination computation. Earth = 23.44°.
    /// Increase for more extreme seasons, decrease or zero for no seasons,
    /// use exotic values for alien planets.
    pub axial_tilt_deg: f32,
}

impl Default for SunConfig {
    fn default() -> Self {
        Self {
            // NYC Lower Manhattan — game map location.
            // TODO: move into map metadata when map files carry geo-coordinates.
            latitude_deg: 40.71,
            north_azimuth_offset_deg: 0.0,
            axial_tilt_deg: 23.44,
        }
    }
}

impl SunConfig {
    /// NYC Lower Manhattan preset — the urban_horizon map location.
    /// Latitude 40.71°N, no north offset (map +Z = geographic south).
    pub fn nyc() -> Self {
        Self::default()
    }

    /// Compute the sun's world-space direction (pointing FROM scene TOWARD sun,
    /// i.e. the light source direction) for a given `day_of_year` [0, 365) and
    /// `hour_of_day` [0, 24).
    ///
    /// Returns `Vec3::ZERO` when the sun is below the horizon (night).
    ///
    /// # Solar math (IAU / Meeus simplified)
    ///
    /// ```text
    /// declination δ  = axial_tilt · sin(2π · (day_of_year − 81) / 365)
    /// hour angle  H  = (solar_hour − 12) · 15°          [degrees → radians]
    ///
    /// altitude  = asin( sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(H) )
    /// azimuth   = atan2( −cos(δ)·sin(H),
    ///                     sin(lat)·cos(δ)·cos(H) − cos(lat)·sin(δ) )
    ///   (CW from north, 0 = N, 90 = E, 180 = S, 270 = W)
    /// ```
    pub fn direction(&self, day_of_year: f32, hour_of_day: f32) -> Vec3 {
        let lat = self.latitude_deg.to_radians();
        let tilt = self.axial_tilt_deg.to_radians();

        // Solar declination: angle between equator and ecliptic plane at this
        // day. Peak +δ at summer solstice (~Jun 21 = day 172).
        let declination = tilt * (std::f32::consts::TAU * (day_of_year - 81.0) / 365.0).sin();

        // Hour angle: 0 at solar noon, negative in the morning, positive in the
        // afternoon. Each hour = 15°.
        let hour_angle = ((hour_of_day - 12.0) * 15.0f32).to_radians();

        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let sin_dec = declination.sin();
        let cos_dec = declination.cos();
        let cos_h = hour_angle.cos();
        let sin_h = hour_angle.sin();

        // Altitude above horizon.
        let sin_alt = sin_lat * sin_dec + cos_lat * cos_dec * cos_h;
        let altitude = sin_alt.clamp(-1.0, 1.0).asin();

        // Night: sun below horizon → no contribution.
        if altitude <= 0.0 {
            return Vec3::ZERO;
        }

        let cos_alt = altitude.cos();

        // Solar azimuth: CW from geographic north (standard convention).
        // atan2(-cos_dec * sin_h,  sin_lat * cos_dec * cos_h - cos_lat * sin_dec)
        let az_y = -cos_dec * sin_h;
        let az_x = sin_lat * cos_dec * cos_h - cos_lat * sin_dec;
        let azimuth = az_y.atan2(az_x);

        // Convert altitude/azimuth → Y-up world-space direction.
        // Map convention: +X = east, +Y = up, +Z = south (geographic south).
        //
        // The Meeus solar azimuth formula uses az=0 = geographic SOUTH (CW
        // from south), so:
        //   east component  = cos(alt) * sin(az)   (az=90° → east, az=270° → west)
        //   up component    = sin(alt)
        //   south component = cos(alt) * cos(az)   (az=0 → south = +Z ✓)
        let raw = Vec3::new(
            cos_alt * azimuth.sin(),  // +X = east
            sin_alt,                  // +Y = up
            cos_alt * azimuth.cos(),  // +Z = south (az=0 is south)
        );

        // Apply the north-axis rotation: rotate about Y by -offset so that if
        // the map's +Z is e.g. geographic east (offset=−90°), the sun path is
        // correctly rotated into map space.
        let rot_rad = (-self.north_azimuth_offset_deg).to_radians();
        let rot = Mat3::from_rotation_y(rot_rad);
        let dir = rot * raw;

        // Normalise to guard against any floating-point drift.
        dir.normalize()
    }

    /// Sun color and intensity modulated by altitude above the horizon.
    ///
    /// Returns `(color: [f32; 3], radiance: f32)`:
    /// - **Near horizon** (altitude ≈ 0°): warm orange/red `[1.0, 0.55, 0.15]`,
    ///   low radiance (~8.0).
    /// - **Midday** (altitude ≈ 90°): neutral white `[1.0, 0.97, 0.92]`, full
    ///   radiance (~30.0).
    /// - Night (altitude ≤ 0°): black, zero radiance.
    ///
    /// The altitude is derived by calling `direction()` and reading the Y
    /// component (= sin(altitude)).
    pub fn color_and_radiance(&self, day_of_year: f32, hour_of_day: f32) -> ([f32; 3], f32) {
        let dir = self.direction(day_of_year, hour_of_day);
        let sin_alt = dir.y; // equals sin(altitude) for a normalised direction
        if sin_alt <= 0.0 {
            return ([0.0, 0.0, 0.0], 0.0);
        }

        // Blend factor: 0 = near horizon, 1 = zenith.
        let t = sin_alt.clamp(0.0, 1.0);

        // Color: warm orange-red at low sun → neutral cool-white at noon.
        let r = 1.0f32;
        let g = 0.55 + 0.42 * t;                // 0.55 at horizon → 0.97 at noon
        let b = 0.15 + 0.77 * t;                // 0.15 at horizon → 0.92 at noon
        let color = [r, g, b];

        // Radiance: physically the optical path through the atmosphere scales as
        // 1/sin(alt) (air mass), so radiance ∝ sin(alt). We clamp the transition
        // near the horizon to avoid a discontinuity right at alt=0.
        let radiance = 8.0 + 22.0 * t.powf(0.4); // 8 at horizon → 30 at noon

        (color, radiance)
    }
}

// ── Tick ↔ Solar time helpers ─────────────────────────────────────────────

/// Derive `hour_of_day` [0, 24) from a tick-of-day [0, ticks_per_day).
///
/// Uses the urban_horizon cadence: `TICKS_PER_DAY = 144` (1 tick = 10 min).
/// Passes through the calendar's `hour` field for the coarse value and adds
/// fractional minutes from `minute` so the sun moves smoothly between ticks.
#[inline]
pub fn hour_from_tick_of_day(tick_of_day: u64, ticks_per_day: u64) -> f32 {
    // Each tick = 24/ticks_per_day hours.
    let hours_per_tick = 24.0 / ticks_per_day as f32;
    tick_of_day as f32 * hours_per_tick
}

/// Derive `day_of_year` [0, 365) from a civic day-of-year [0, days_per_year).
///
/// The game uses a 360-day civic year; map it linearly onto the 365-day solar
/// calendar so seasons stay correctly positioned.
#[inline]
pub fn solar_day_from_civic_day(civic_day_of_year: u64, civic_days_per_year: u64) -> f32 {
    // Linear remap: 0 → 0, civic_days_per_year → 365.
    civic_day_of_year as f32 * 365.0 / civic_days_per_year as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const NYC: SunConfig = SunConfig {
        latitude_deg: 40.71,
        north_azimuth_offset_deg: 0.0,
        axial_tilt_deg: 23.44,
    };

    // ── Helper ──────────────────────────────────────────────────────────────

    fn alt_deg(dir: Vec3) -> f32 {
        dir.y.asin().to_degrees()
    }

    // Azimuth of `dir` measured CW from +Z (geographic south in our convention
    // after the north_azimuth_offset=0 rotation).  We compare against known
    // solar azimuths.
    fn az_from_south_deg(dir: Vec3) -> f32 {
        // In our convention +X=east, +Z=south. Azimuth CW from south = atan2(+X, +Z).
        dir.x.atan2(dir.z).to_degrees()
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// At the March equinox (day 80), at solar noon (12:00), the sun is due south
    /// (az ≈ 180° from N = 0° from S in our map) at altitude ≈ 90° − latitude.
    #[test]
    fn equinox_noon_sun_is_due_south_at_correct_altitude() {
        let dir = NYC.direction(80.0, 12.0);
        assert!(dir != Vec3::ZERO, "sun should be above horizon at noon");

        let alt = alt_deg(dir);
        let expected_alt = 90.0 - NYC.latitude_deg; // ≈ 49.29°
        assert!(
            (alt - expected_alt).abs() < 1.0,
            "equinox noon altitude should be ≈{expected_alt:.1}°; got {alt:.2}°"
        );

        // Due south: map +Z = south, az from south ≈ 0°.
        let az = az_from_south_deg(dir);
        assert!(
            az.abs() < 2.0,
            "equinox noon sun should be due south (az≈0° from S); got {az:.2}°"
        );

        // Must be unit vector.
        assert!(
            (dir.length() - 1.0).abs() < 1e-5,
            "direction must be normalized; length={}",
            dir.length()
        );
    }

    /// At the equinox (day 80), sunrise is near 6:00 and the sun rises due east.
    #[test]
    fn equinox_sunrise_direction_is_near_east() {
        // Just after sunrise: the sun should be near the horizon and east.
        let dir = NYC.direction(80.0, 6.1);
        assert!(
            dir != Vec3::ZERO,
            "sun should be above horizon just after equinox sunrise"
        );
        let alt = alt_deg(dir);
        assert!(
            alt < 8.0,
            "just after sunrise altitude should be low; got {alt:.2}°"
        );
        // East = +X positive, small Z.
        assert!(
            dir.x > 0.5,
            "sunrise should point east (dir.x > 0.5); got x={}",
            dir.x
        );
    }

    /// Midnight (0:00) is always below the horizon.
    #[test]
    fn midnight_is_below_horizon() {
        // Use summer solstice (day 172) where the day is longest — if midnight
        // is ever above horizon, it would be here.
        let dir = NYC.direction(172.0, 0.0);
        assert_eq!(
            dir,
            Vec3::ZERO,
            "midnight at NYC in summer should be below horizon; got {dir:?}"
        );
    }

    /// Winter solstice noon: declination ≈ −23.44°, so altitude is much lower.
    #[test]
    fn winter_solstice_noon_is_lower_than_equinox() {
        let equinox_dir = NYC.direction(80.0, 12.0);
        let solstice_dir = NYC.direction(355.0, 12.0); // ≈ Dec 21
        assert!(
            solstice_dir != Vec3::ZERO,
            "sun above horizon at noon in winter"
        );
        let eq_alt = alt_deg(equinox_dir);
        let sol_alt = alt_deg(solstice_dir);
        assert!(
            sol_alt < eq_alt - 15.0,
            "winter solstice noon altitude ({sol_alt:.1}°) should be >15° below equinox ({eq_alt:.1}°)"
        );
    }

    /// Summer solstice noon: declination ≈ +23.44°, altitude is higher.
    #[test]
    fn summer_solstice_noon_is_higher_than_equinox() {
        let equinox_dir = NYC.direction(80.0, 12.0);
        let solstice_dir = NYC.direction(172.0, 12.0); // ≈ Jun 21
        let eq_alt = alt_deg(equinox_dir);
        let sol_alt = alt_deg(solstice_dir);
        assert!(
            sol_alt > eq_alt + 15.0,
            "summer solstice noon altitude ({sol_alt:.1}°) should be >15° above equinox ({eq_alt:.1}°)"
        );
    }

    /// Axial tilt 0 means no seasonal variation: noon altitude is constant
    /// across the year at exactly (90° − latitude).
    #[test]
    fn zero_axial_tilt_produces_no_seasonal_variation() {
        let flat = SunConfig {
            axial_tilt_deg: 0.0,
            ..NYC
        };
        let dir_equinox = flat.direction(80.0, 12.0);
        let dir_summer = flat.direction(172.0, 12.0);
        let dir_winter = flat.direction(355.0, 12.0);
        let alt_e = alt_deg(dir_equinox);
        let alt_s = alt_deg(dir_summer);
        let alt_w = alt_deg(dir_winter);
        assert!(
            (alt_e - alt_s).abs() < 0.5,
            "zero tilt: equinox and summer noon should match; {alt_e:.2}° vs {alt_s:.2}°"
        );
        assert!(
            (alt_e - alt_w).abs() < 0.5,
            "zero tilt: equinox and winter noon should match; {alt_e:.2}° vs {alt_w:.2}°"
        );
    }

    /// North azimuth offset of 180° should mirror the sun east↔west.
    #[test]
    fn north_azimuth_offset_180_mirrors_east_west() {
        let flipped = SunConfig {
            north_azimuth_offset_deg: 180.0,
            ..NYC
        };
        let default_dir = NYC.direction(80.0, 14.0); // afternoon: sun west of south
        let flipped_dir = flipped.direction(80.0, 14.0);
        // X (east) component should be negated.
        assert!(
            (default_dir.x + flipped_dir.x).abs() < 0.05,
            "180° offset should negate east component; got {:.3} and {:.3}",
            default_dir.x,
            flipped_dir.x
        );
    }

    /// `color_and_radiance` returns zero at night.
    #[test]
    fn color_and_radiance_zero_at_night() {
        let (color, radiance) = NYC.color_and_radiance(80.0, 0.0);
        assert_eq!(color, [0.0, 0.0, 0.0], "color should be zero at night");
        assert_eq!(radiance, 0.0, "radiance should be zero at night");
    }

    /// At noon, radiance is close to 30 and color is near neutral white.
    #[test]
    fn color_and_radiance_high_at_noon() {
        let (color, radiance) = NYC.color_and_radiance(80.0, 12.0);
        assert!(
            radiance > 25.0,
            "noon radiance should be close to 30; got {radiance:.2}"
        );
        // Neutral white: R≈1, G≈0.97, B≈0.92 — all channels above 0.8.
        assert!(
            color[0] > 0.9 && color[1] > 0.8 && color[2] > 0.7,
            "noon color should be near white; got {:?}",
            color
        );
    }

    /// `hour_from_tick_of_day` with TICKS_PER_DAY=144:
    /// tick 48 → hour 8.0, tick 102 → hour 17.0.
    #[test]
    fn tick_to_hour_matches_care_day_anchors() {
        let tpd = 144u64;
        let h48 = hour_from_tick_of_day(48, tpd);
        assert!(
            (h48 - 8.0).abs() < 0.01,
            "tick 48 should be 08:00; got {h48}"
        );
        let h102 = hour_from_tick_of_day(102, tpd);
        assert!(
            (h102 - 17.0).abs() < 0.01,
            "tick 102 should be 17:00; got {h102}"
        );
    }

    /// `solar_day_from_civic_day` maps 360-day year onto 365-day solar year linearly.
    #[test]
    fn solar_day_mapping_is_correct() {
        // Day 0 → solar day 0.
        let d0 = solar_day_from_civic_day(0, 360);
        assert!((d0 - 0.0).abs() < 1e-6, "civic day 0 → solar day 0; got {d0}");
        // Day 360 → solar day 365.
        let d360 = solar_day_from_civic_day(360, 360);
        assert!(
            (d360 - 365.0).abs() < 1e-4,
            "civic day 360 → solar day 365; got {d360}"
        );
        // Day 180 → solar day 182.5.
        let d180 = solar_day_from_civic_day(180, 360);
        assert!(
            (d180 - 182.5).abs() < 0.01,
            "civic day 180 → solar day 182.5; got {d180}"
        );
    }
}
