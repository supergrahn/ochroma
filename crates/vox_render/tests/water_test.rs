use vox_render::water::*;
use glam::Vec3;

#[test]
fn river_generates_splats() {
    let river = WaterSurface::river(Vec3::ZERO, 10.0, 50.0, 1.0);
    let splats = river.generate_splats(0.0);
    assert!(!splats.is_empty());
    assert!(splats.len() > 100);
}

#[test]
fn wave_animation_changes_height() {
    let river = WaterSurface::river(Vec3::ZERO, 10.0, 10.0, 1.0);
    let splats_t0 = river.generate_splats(0.0);
    let splats_t1 = river.generate_splats(1.0);
    // Heights should differ due to wave animation
    let differs = splats_t0.iter().zip(splats_t1.iter())
        .any(|(a, b)| (a.position[1] - b.position[1]).abs() > 0.001);
    assert!(differs, "Wave animation should change heights over time");
}

#[test]
fn fresnel_higher_at_grazing_angle() {
    let perpendicular = WaterSurface::fresnel(Vec3::Y, 1.33);
    let grazing = WaterSurface::fresnel(Vec3::new(1.0, 0.05, 0.0).normalize(), 1.33);
    assert!(grazing > perpendicular, "Fresnel should be higher at grazing angles");
}

#[test]
fn reflection_direction_correct() {
    let incoming = Vec3::new(0.5, -0.5, 0.0).normalize();
    let reflected = WaterSurface::reflect(incoming);
    assert!(reflected.y > 0.0, "Reflected direction should point upward");
}

#[test]
fn lake_generates_circular_area() {
    let lake = WaterSurface::lake(Vec3::ZERO, 20.0);
    let splats = lake.generate_splats(0.0);
    assert!(splats.len() > 1000, "Lake should have many surface splats");
}
