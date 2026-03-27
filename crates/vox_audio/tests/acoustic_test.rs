use glam::Vec3;
use vox_audio::acoustic_raytracer::*;

fn small_room_scene() -> AcousticScene {
    AcousticScene {
        surfaces: vec![
            // Floor
            AcousticSurface {
                position: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::Y,
                material: AcousticMaterial::CONCRETE,
                radius: 5.0,
            },
            // Ceiling
            AcousticSurface {
                position: Vec3::new(0.0, 3.0, 0.0),
                normal: -Vec3::Y,
                material: AcousticMaterial::CONCRETE,
                radius: 5.0,
            },
            // Back wall (brick)
            AcousticSurface {
                position: Vec3::new(0.0, 1.5, -5.0),
                normal: Vec3::Z,
                material: AcousticMaterial::BRICK,
                radius: 5.0,
            },
            // Front wall (glass)
            AcousticSurface {
                position: Vec3::new(0.0, 1.5, 5.0),
                normal: -Vec3::Z,
                material: AcousticMaterial::GLASS,
                radius: 5.0,
            },
        ],
    }
}

#[test]
fn brick_absorbs_high_frequencies() {
    let material = AcousticMaterial::BRICK;
    let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::Z, 3);
    ray.absorb(&material);

    // High frequency should be most absorbed.
    assert!(ray.energy[2] < ray.energy[0], "brick should absorb high more than low");
    assert!(ray.energy[2] < ray.energy[1], "brick should absorb high more than mid");
    // Specifically: low should retain 95%, high should retain 40%.
    assert!((ray.energy[0] - 0.95).abs() < 0.01);
    assert!((ray.energy[2] - 0.40).abs() < 0.01);
}

#[test]
fn glass_transmits_high_frequencies() {
    let material = AcousticMaterial::GLASS;
    let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::Z, 3);
    ray.absorb(&material);

    // High frequency should be least absorbed (most transmitted).
    assert!(ray.energy[2] > ray.energy[0], "glass should transmit high more than low");
    assert!((ray.energy[2] - 0.95).abs() < 0.01);
    assert!((ray.energy[0] - 0.70).abs() < 0.01);
}

#[test]
fn obstruction_reduces_volume() {
    let source = Vec3::new(0.0, 1.0, 0.0);
    let listener = Vec3::new(0.0, 1.0, 10.0);

    let wall = AcousticSurface {
        position: Vec3::new(0.0, 1.0, 5.0),
        normal: -Vec3::Z,
        material: AcousticMaterial::CONCRETE,
        radius: 10.0,
    };

    let attenuation = compute_obstruction(source, listener, &[wall]);

    // All bands should have some attenuation.
    assert!(attenuation[0] > 0.0, "low freq should be attenuated");
    assert!(attenuation[1] > 0.0, "mid freq should be attenuated");
    assert!(attenuation[2] > 0.0, "high freq should be attenuated");

    // High frequencies should be attenuated more.
    assert!(attenuation[2] > attenuation[0], "high freq blocked more than low");
}

#[test]
fn doppler_shift_approaching_source() {
    let source_pos = Vec3::new(0.0, 0.0, -10.0);
    let listener_pos = Vec3::ZERO;

    // Source moving toward listener.
    let source_vel = Vec3::new(0.0, 0.0, 30.0); // 30 m/s toward listener
    let listener_vel = Vec3::ZERO;

    let shift = doppler_shift(source_vel, listener_vel, source_pos, listener_pos, SPEED_OF_SOUND);

    // Approaching source = higher pitch = multiplier > 1.0.
    assert!(shift > 1.0, "approaching source should increase frequency, got {}", shift);
    // Expected: 343 / (343 - 30) = ~1.096
    assert!((shift - 1.096).abs() < 0.01, "shift should be ~1.096, got {}", shift);
}

#[test]
fn doppler_shift_receding_source() {
    let source_pos = Vec3::new(0.0, 0.0, -10.0);
    let listener_pos = Vec3::ZERO;

    // Source moving away from listener.
    let source_vel = Vec3::new(0.0, 0.0, -30.0); // 30 m/s away
    let listener_vel = Vec3::ZERO;

    let shift = doppler_shift(source_vel, listener_vel, source_pos, listener_pos, SPEED_OF_SOUND);

    // Receding source = lower pitch = multiplier < 1.0.
    assert!(shift < 1.0, "receding source should decrease frequency, got {}", shift);
}

#[test]
fn rt60_longer_in_open_space() {
    let open_scene = AcousticScene { surfaces: vec![] };
    let closed_scene = small_room_scene();

    let source = Vec3::new(0.0, 1.5, 0.0);
    let listener = Vec3::new(2.0, 1.5, 0.0);

    let open_result = trace_sound(source, listener, &open_scene, 3);
    let closed_result = trace_sound(source, listener, &closed_scene, 3);

    assert!(
        open_result.rt60 > closed_result.rt60,
        "open space RT60 ({}) should be > closed room RT60 ({})",
        open_result.rt60,
        closed_result.rt60
    );
}

#[test]
fn procedural_rain_scales_with_intensity() {
    let light = ProceduralSound::Rain { intensity: 0.2 };
    let heavy = ProceduralSound::Rain { intensity: 0.8 };

    let light_spec = light.generate_frequency_spectrum();
    let heavy_spec = heavy.generate_frequency_spectrum();

    for i in 0..3 {
        assert!(
            heavy_spec[i] > light_spec[i],
            "heavy rain band {} ({}) should be louder than light ({})",
            i,
            heavy_spec[i],
            light_spec[i]
        );
    }

    // Rain should have most energy in high frequencies.
    let heavy_high = heavy_spec[2];
    let heavy_low = heavy_spec[0];
    assert!(heavy_high > heavy_low, "rain should have more high freq than low");
}

#[test]
fn direct_path_inverse_square() {
    let scene = AcousticScene { surfaces: vec![] };

    // Source at distance 2.
    let result_near = trace_sound(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), &scene, 0);
    // Source at distance 4.
    let result_far = trace_sound(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), &scene, 0);

    // Inverse square: doubling distance should quarter the energy.
    let ratio = result_near.direct_attenuation[0] / result_far.direct_attenuation[0];
    assert!(
        (ratio - 4.0).abs() < 0.1,
        "doubling distance should quarter energy, ratio was {}",
        ratio
    );
}

#[test]
fn reflection_adds_delay() {
    let scene = small_room_scene();
    let source = Vec3::new(0.0, 1.5, 0.0);
    let listener = Vec3::new(1.0, 1.5, 0.0);

    let result = trace_sound(source, listener, &scene, 3);

    // Direct path distance.
    let direct_dist = source.distance(listener);
    let direct_delay = direct_dist / SPEED_OF_SOUND;

    // All reflections should have longer delay than direct path.
    for reflection in &result.early_reflections {
        assert!(
            reflection.delay > direct_delay,
            "reflection delay ({}) should exceed direct delay ({})",
            reflection.delay,
            direct_delay
        );
    }

    // There should be at least one reflection in a room.
    assert!(
        !result.early_reflections.is_empty(),
        "room should produce reflections"
    );
}

#[test]
fn wood_absorbs_low_frequencies() {
    let material = AcousticMaterial::WOOD;
    let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::Z, 3);
    ray.absorb(&material);

    assert!(ray.energy[0] < ray.energy[2], "wood should absorb low more than high");
    assert!((ray.energy[0] - 0.45).abs() < 0.01);
}

#[test]
fn procedural_traffic_scales_with_vehicles() {
    let few = ProceduralSound::Traffic { vehicle_count: 4, avg_speed: 40.0 };
    let many = ProceduralSound::Traffic { vehicle_count: 100, avg_speed: 40.0 };

    let few_spec = few.generate_frequency_spectrum();
    let many_spec = many.generate_frequency_spectrum();

    for i in 0..3 {
        assert!(
            many_spec[i] > few_spec[i],
            "more vehicles should be louder in band {}",
            i
        );
    }
}
