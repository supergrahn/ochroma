use glam::Vec3;
use vox_render::lighting::*;

#[test]
fn sun_above_horizon_at_noon() {
    let model = SunModel::new(45.0);
    assert!(model.is_daytime(12.0, 172)); // June 21
    assert!(model.sun_direction(12.0, 172).y > 0.0);
}

#[test]
fn sun_below_horizon_at_midnight() {
    let model = SunModel::new(45.0);
    assert!(!model.is_daytime(0.0, 172));
}

#[test]
fn sun_intensity_peaks_at_noon() {
    let model = SunModel::new(45.0);
    let noon = model.sun_intensity(12.0, 172);
    let morning = model.sun_intensity(8.0, 172);
    assert!(noon > morning);
}

#[test]
fn point_light_attenuation() {
    let light = PointLight {
        position: Vec3::ZERO,
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        radius: 10.0,
    };
    assert!(light.attenuation(0.0) > light.attenuation(5.0));
    assert_eq!(light.attenuation(10.0), 0.0);
    assert_eq!(light.attenuation(20.0), 0.0);
}

#[test]
fn sky_color_is_valid_rgb() {
    let color = sky_color(Vec3::new(0.0, 0.8, 0.0), Vec3::new(0.0, 0.5, -1.0));
    for c in &color {
        assert!(*c >= 0.0 && *c <= 1.0);
    }
}

#[test]
fn light_manager_combines_sources() {
    let mut mgr = LightManager::new(45.0);
    mgr.add_point_light(PointLight {
        position: Vec3::ZERO,
        color: [1.0, 1.0, 1.0],
        intensity: 0.5,
        radius: 10.0,
    });
    let light = mgr.light_at(Vec3::new(1.0, 0.0, 0.0), 12.0, 172);
    assert!(light > 0.0);
}
