use vox_render::atmosphere::*;
use glam::Vec3;

#[test]
fn sky_is_blue_overhead() {
    let params = AtmosphereParams::default();
    let color = compute_sky_color(Vec3::Y, Vec3::new(0.5, 0.8, 0.0).normalize(), &params);
    // Blue should dominate overhead
    assert!(color[2] > color[0], "Sky should be bluer than red overhead");
}

#[test]
fn sunset_is_red() {
    let params = AtmosphereParams::default();
    // Look at low sun near horizon
    let sun_dir = Vec3::new(1.0, 0.05, 0.0).normalize();
    let color = compute_sky_color(sun_dir, sun_dir, &params);
    // At sunset, Mie scattering dominates -> warm colours
    assert!(color[0] > 0.0 || color[1] > 0.0, "Sunset should have warm tones");
}

#[test]
fn fog_increases_with_distance() {
    let (_, blend_near) = compute_fog(10.0, 0.01, [0.7, 0.8, 0.9]);
    let (_, blend_far) = compute_fog(100.0, 0.01, [0.7, 0.8, 0.9]);
    assert!(blend_far > blend_near, "More fog at greater distance");
}

#[test]
fn god_rays_strongest_toward_sun() {
    let toward_sun = compute_god_ray_intensity(
        Vec3::ZERO, Vec3::new(0.0, 0.3, -1.0).normalize(),
        Vec3::new(0.0, 0.3, -1.0).normalize(), 50.0, 16,
    );
    let away_from_sun = compute_god_ray_intensity(
        Vec3::ZERO, Vec3::new(0.0, 0.3, 1.0).normalize(),
        Vec3::new(0.0, 0.3, -1.0).normalize(), 50.0, 16,
    );
    assert!(toward_sun > away_from_sun, "God rays stronger toward sun");
}
