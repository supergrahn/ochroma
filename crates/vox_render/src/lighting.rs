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
