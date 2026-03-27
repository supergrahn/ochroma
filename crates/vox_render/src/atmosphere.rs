use glam::Vec3;

/// Atmospheric scattering parameters.
pub struct AtmosphereParams {
    pub rayleigh_scale_height: f32,  // km
    pub mie_scale_height: f32,       // km
    pub rayleigh_coefficient: Vec3,  // RGB scattering coefficients
    pub mie_coefficient: f32,
    pub mie_g: f32,                  // anisotropy (-1 to 1)
    pub sun_intensity: f32,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            rayleigh_scale_height: 8.0,
            mie_scale_height: 1.2,
            rayleigh_coefficient: Vec3::new(5.8e-6, 13.5e-6, 33.1e-6), // Earth atmosphere
            mie_coefficient: 21e-6,
            mie_g: 0.76,
            sun_intensity: 22.0,
        }
    }
}

/// Compute sky colour at a given view direction using single-scattering approximation.
pub fn compute_sky_color(
    view_dir: Vec3,
    sun_dir: Vec3,
    params: &AtmosphereParams,
) -> [f32; 3] {
    let cos_theta = view_dir.dot(sun_dir).clamp(-1.0, 1.0);

    // Rayleigh phase function: 3/(16pi) * (1 + cos^2 theta)
    let rayleigh_phase = 0.0596831 * (1.0 + cos_theta * cos_theta);

    // Mie phase function (Henyey-Greenstein): (1-g^2) / (4pi * (1+g^2-2g*cos theta)^1.5)
    let g = params.mie_g;
    let mie_denom = 1.0 + g * g - 2.0 * g * cos_theta;
    let mie_phase = 0.07958 * (1.0 - g * g) / (mie_denom * mie_denom.sqrt());

    // View altitude approximation (ground level)
    let altitude_factor = (view_dir.y.max(0.0) * 2.0).min(1.0);

    // Optical depth approximation
    let rayleigh_depth = (-altitude_factor / params.rayleigh_scale_height).exp();
    let mie_depth = (-altitude_factor / params.mie_scale_height).exp();

    // Scattering
    let rayleigh = params.rayleigh_coefficient * rayleigh_phase * rayleigh_depth;
    let mie = Vec3::splat(params.mie_coefficient * mie_phase * mie_depth);

    let color = (rayleigh + mie) * params.sun_intensity;

    [color.x.min(1.0), color.y.min(1.0), color.z.min(1.0)]
}

/// Compute fog attenuation at a given distance.
pub fn compute_fog(distance: f32, fog_density: f32, fog_color: [f32; 3]) -> ([f32; 3], f32) {
    let fog_factor = (-distance * fog_density).exp();
    let blend = 1.0 - fog_factor;
    (fog_color, blend)
}

/// Compute god ray intensity along a view ray.
/// Simple approximation: sample along ray, check if each sample is in shadow.
pub fn compute_god_ray_intensity(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sun_dir: Vec3,
    max_distance: f32,
    samples: u32,
) -> f32 {
    let step = max_distance / samples as f32;
    let mut intensity = 0.0f32;

    for i in 0..samples {
        let t = (i as f32 + 0.5) * step;
        let sample_pos = ray_origin + ray_dir * t;

        // Simple: intensity based on alignment with sun direction
        let alignment = ray_dir.dot(sun_dir).max(0.0);
        let distance_falloff = 1.0 / (1.0 + t * 0.1);

        // Height-based density (more particles near ground)
        let height_density = (-sample_pos.y.max(0.0) * 0.05).exp();

        intensity += alignment * distance_falloff * height_density;
    }

    (intensity / samples as f32).min(1.0)
}
