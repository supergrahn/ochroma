use glam::Vec3;

/// Sun position based on time of day and latitude.
pub struct SunModel {
    pub latitude: f32, // degrees
}

impl SunModel {
    pub fn new(latitude: f32) -> Self {
        Self { latitude }
    }

    /// Calculate sun direction for a given hour (0-24) and day of year (0-365).
    pub fn sun_direction(&self, hour: f32, day_of_year: u32) -> Vec3 {
        let declination = 23.45_f32.to_radians()
            * ((360.0 / 365.0 * (day_of_year as f32 + 284.0)).to_radians()).sin();
        let hour_angle = (hour - 12.0) * 15.0_f32.to_radians();
        let lat = self.latitude.to_radians();

        let altitude = (lat.sin() * declination.sin()
            + lat.cos() * declination.cos() * hour_angle.cos())
        .asin();
        let azimuth = (declination.sin() * lat.cos()
            - declination.cos() * lat.sin() * hour_angle.cos())
        .atan2(-declination.cos() * hour_angle.sin());

        Vec3::new(
            azimuth.cos() * altitude.cos(),
            altitude.sin(),
            azimuth.sin() * altitude.cos(),
        )
        .normalize()
    }

    /// Sun intensity based on altitude (0 at horizon, max at zenith).
    pub fn sun_intensity(&self, hour: f32, day_of_year: u32) -> f32 {
        let dir = self.sun_direction(hour, day_of_year);
        dir.y.max(0.0) // intensity proportional to altitude above horizon
    }

    /// Is the sun above the horizon?
    pub fn is_daytime(&self, hour: f32, day_of_year: u32) -> bool {
        self.sun_direction(hour, day_of_year).y > 0.0
    }
}

/// Sky colour model (simplified Preetham).
pub fn sky_color(sun_direction: Vec3, view_direction: Vec3) -> [f32; 3] {
    let sun_alt = sun_direction.y.max(0.0);

    // Base sky blue, modified by sun position
    let zenith_r = 0.1 + 0.3 * (1.0 - sun_alt); // redder at low sun
    let zenith_g = 0.2 + 0.2 * sun_alt;
    let zenith_b = 0.4 + 0.4 * sun_alt;

    // Horizon is brighter and more orange near sunset
    let view_alt = view_direction.y.max(0.0);
    let horizon_blend = 1.0 - view_alt;

    let horizon_r = 0.8 * (1.0 - sun_alt) + 0.3 * sun_alt;
    let horizon_g = 0.4 * (1.0 - sun_alt) + 0.3 * sun_alt;
    let horizon_b = 0.2 * (1.0 - sun_alt) + 0.4 * sun_alt;

    [
        zenith_r * (1.0 - horizon_blend) + horizon_r * horizon_blend,
        zenith_g * (1.0 - horizon_blend) + horizon_g * horizon_blend,
        zenith_b * (1.0 - horizon_blend) + horizon_b * horizon_blend,
    ]
}

/// A point light in the scene.
#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32, // attenuation radius
}

impl PointLight {
    /// Attenuation at a given distance.
    pub fn attenuation(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.0;
        }
        let d = distance / self.radius;
        self.intensity * (1.0 - d * d).max(0.0)
    }
}

/// Manages all lights in the scene.
pub struct LightManager {
    pub sun: SunModel,
    pub point_lights: Vec<PointLight>,
    pub ambient_intensity: f32,
}

impl LightManager {
    pub fn new(latitude: f32) -> Self {
        Self {
            sun: SunModel::new(latitude),
            point_lights: Vec::new(),
            ambient_intensity: 0.1,
        }
    }

    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }

    /// Get total light contribution at a point.
    pub fn light_at(&self, position: Vec3, hour: f32, day: u32) -> f32 {
        let sun = self.sun.sun_intensity(hour, day);
        let point: f32 = self
            .point_lights
            .iter()
            .map(|l| l.attenuation(l.position.distance(position)))
            .sum();
        (sun + point + self.ambient_intensity).min(2.0)
    }

    pub fn point_light_count(&self) -> usize {
        self.point_lights.len()
    }
}

// ── Simplified Sun Direction ──────────────────────────────────────────────

/// Compute sun direction from hour (0-24) and latitude (degrees).
///
/// Uses a simplified model with declination = 0 (equinox approximation).
/// Returns a normalized Vec3 where +Y is up, +X is east, +Z is south.
pub fn sun_direction(hour: f32, latitude_deg: f32) -> Vec3 {
    let lat = latitude_deg.to_radians();
    // Hour angle: 0 at noon, negative morning, positive afternoon
    let hour_angle = (hour - 12.0) * 15.0_f32.to_radians();
    // Declination = 0 (equinox)
    let sin_alt = lat.sin() * 0.0 + lat.cos() * 1.0 * hour_angle.cos();
    let altitude = sin_alt.asin();
    let cos_az = (0.0 - lat.sin() * sin_alt) / (lat.cos() * altitude.cos() + 1e-10);
    let azimuth = if hour_angle.sin() > 0.0 {
        std::f32::consts::PI - cos_az.clamp(-1.0, 1.0).acos()
    } else {
        std::f32::consts::PI + cos_az.clamp(-1.0, 1.0).acos()
    };

    Vec3::new(
        -azimuth.sin() * altitude.cos(),
        altitude.sin(),
        -azimuth.cos() * altitude.cos(),
    )
    .normalize()
}

// ── Sky Colors ────────────────────────────────────────────────────────────

/// Zenith, horizon, and sun disk colors from a Preetham-inspired sky model.
#[derive(Debug, Clone, Copy)]
pub struct SkyColors {
    /// RGB color at the zenith (straight up).
    pub zenith: [f32; 3],
    /// RGB color at the horizon.
    pub horizon: [f32; 3],
    /// RGB color of the sun disk.
    pub sun: [f32; 3],
}

/// Compute sky colors using a simplified Preetham model with turbidity = 2.5.
pub fn preetham_sky(sun_dir: Vec3) -> SkyColors {
    let turbidity: f32 = 2.5;
    let sun_alt = sun_dir.y.max(0.0);
    let sun_below = sun_dir.y < 0.0;

    if sun_below {
        let night = 0.02;
        // Twilight blend: sun just below horizon still has warm afterglow
        let twilight = ((-sun_dir.y) / 0.3).clamp(0.0, 1.0); // 0 = just set, 1 = deep night
        let horizon_r = (0.4 * (1.0 - twilight) + night * twilight).clamp(0.0, 1.0);
        let horizon_g = (0.15 * (1.0 - twilight) + night * twilight).clamp(0.0, 1.0);
        let horizon_b = (0.1 * (1.0 - twilight) + night * 1.2 * twilight).clamp(0.0, 1.0);
        return SkyColors {
            zenith: [night * 0.5, night * 0.5, night * 0.8],
            horizon: [horizon_r, horizon_g, horizon_b],
            sun: [0.0, 0.0, 0.0],
        };
    }

    let chi = (4.0 / 9.0 - turbidity / 120.0)
        * (std::f32::consts::PI - 2.0 * sun_alt.acos()).max(0.0);
    let zenith_y = ((4.0453 * turbidity - 4.971) * chi.tan()
        - 0.2155 * turbidity + 2.4192)
        .max(0.0)
        / 20.0;

    let zenith_r = (0.15 + 0.05 * (turbidity - 2.0)).clamp(0.0, 1.0) * zenith_y;
    let zenith_g = (0.2 + 0.1 * sun_alt).clamp(0.0, 1.0) * zenith_y;
    let zenith_b = (0.45 + 0.3 * sun_alt - 0.05 * turbidity).clamp(0.0, 1.0) * zenith_y;

    let sunset_factor = 1.0 - sun_alt;
    let horizon_r = (0.7 * sunset_factor + 0.3 * sun_alt).clamp(0.0, 1.0);
    let horizon_g = (0.35 * sunset_factor + 0.3 * sun_alt).clamp(0.0, 1.0);
    let horizon_b = (0.15 * sunset_factor + 0.4 * sun_alt).clamp(0.0, 1.0);

    let sun_r = (1.0 - 0.3 * sun_alt).clamp(0.0, 1.0);
    let sun_g = (0.85 - 0.3 * sunset_factor).clamp(0.0, 1.0);
    let sun_b = (0.6 * sun_alt).clamp(0.0, 1.0);

    SkyColors {
        zenith: [zenith_r.clamp(0.0, 1.0), zenith_g.clamp(0.0, 1.0), zenith_b.clamp(0.0, 1.0)],
        horizon: [horizon_r, horizon_g, horizon_b],
        sun: [sun_r, sun_g, sun_b],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn sun_direction_noon_is_high() {
        let dir = sun_direction(12.0, 45.0);
        assert!(dir.y > 0.5, "sun at noon should be high; got y={}", dir.y);
        assert!((dir.length() - 1.0).abs() < 1e-4, "direction should be normalized");
    }

    #[test]
    fn sun_direction_midnight_is_below_horizon() {
        let dir = sun_direction(0.0, 45.0);
        assert!(dir.y < 0.0, "sun at midnight should be below horizon; got y={}", dir.y);
    }

    #[test]
    fn sun_direction_6am_is_near_horizon() {
        let dir = sun_direction(6.0, 0.0);
        assert!(dir.y.abs() < 0.3, "sun at 6am equator should be near horizon; got y={}", dir.y);
    }

    #[test]
    fn sky_colors_values_in_range() {
        let dir = sun_direction(12.0, 45.0);
        let colors = preetham_sky(dir);
        for c in &colors.zenith {
            assert!(*c >= 0.0 && *c <= 1.0, "zenith component {} out of [0,1]", c);
        }
        for c in &colors.horizon {
            assert!(*c >= 0.0 && *c <= 1.0, "horizon component {} out of [0,1]", c);
        }
        for c in &colors.sun {
            assert!(*c >= 0.0 && *c <= 1.0, "sun component {} out of [0,1]", c);
        }
    }

    #[test]
    fn sky_colors_sunset_has_warm_horizon() {
        let dir = sun_direction(18.5, 45.0);
        let colors = preetham_sky(dir);
        assert!(
            colors.horizon[0] > colors.horizon[2],
            "sunset horizon should be redder than blue: r={} b={}",
            colors.horizon[0], colors.horizon[2]
        );
    }
}
