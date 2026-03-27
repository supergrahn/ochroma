use glam::Vec3;
use vox_render::vfx::*;

#[test]
fn fire_effect_produces_particles_after_tick() {
    let effect = effect_fire();
    let mut inst = VfxInstance::new(effect, Vec3::ZERO);
    assert_eq!(inst.particle_count(), 0);
    // Tick enough to emit at 40/s
    inst.tick(0.1);
    assert!(inst.particle_count() > 0, "fire should emit particles after tick");
}

#[test]
fn particles_die_after_lifetime() {
    let effect = VfxEffect {
        name: "short_lived".into(),
        emitters: vec![VfxEmitter {
            shape: EmitterShape::Point,
            rate: 100.0,
            burst: None,
            lifetime: RangeF32::new(0.1, 0.1),
            velocity: VelocityConfig {
                direction: [0.0, 1.0, 0.0],
                speed: RangeF32::new(1.0, 1.0),
                randomness: 0.0,
            },
            size: CurveF32::constant(0.1),
            opacity: CurveF32::constant(1.0),
            color: ColorConfig {
                start_spectral: [0.5; 8],
                end_spectral: [0.5; 8],
            },
            gravity_scale: 0.0,
            max_particles: 1000,
        }],
    };

    let mut inst = VfxInstance::new(effect, Vec3::ZERO);
    // Spawn particles
    inst.tick(0.05);
    let count_after_spawn = inst.particle_count();
    assert!(count_after_spawn > 0);

    // Deactivate so no new particles spawn, then tick past lifetime
    inst.active = false;
    // Reactivate but with zero rate to let existing ones die
    inst.active = true;
    // Manually set rate to 0 so no new particles spawn while we wait for death
    inst.effect.emitters[0].rate = 0.0;
    inst.tick(0.2);
    inst.tick(0.2);
    assert_eq!(inst.particle_count(), 0, "particles should die after lifetime");
}

#[test]
fn explosion_burst_spawns_immediately() {
    let effect = effect_explosion();
    let mut inst = VfxInstance::new(effect, Vec3::ZERO);
    // Even a tiny tick should trigger the burst
    inst.tick(0.001);
    // Explosion has burst of 50 + 30 = 80
    assert!(
        inst.particle_count() >= 50,
        "explosion burst should spawn particles immediately, got {}",
        inst.particle_count()
    );
}

#[test]
fn fade_out_curve_evaluates_correctly() {
    let c = CurveF32::fade_out(1.0);
    assert!((c.evaluate(0.0) - 1.0).abs() < 1e-6);
    assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
    assert!((c.evaluate(1.0) - 0.0).abs() < 1e-6);

    // Midpoint interpolation
    assert!((c.evaluate(0.25) - 0.75).abs() < 1e-6);
}

#[test]
fn cone_emitter_produces_directional_particles() {
    let effect = effect_fire(); // uses Cone emitter
    let mut inst = VfxInstance::new(effect, Vec3::ZERO);
    inst.tick(0.5);
    let splats = inst.to_splats();
    assert!(!splats.is_empty());

    // All fire particles should have moved upward (positive y) from origin
    // after enough time, since direction is [0, 1, 0] with gravity_scale -0.3
    for splat in &splats {
        // They start at y=0, velocity is upward, gravity is inverted (buoyant)
        // so y should be positive or at least near zero
        assert!(
            splat.position[1] >= -1.0,
            "fire particle y={} should not have fallen far below origin",
            splat.position[1]
        );
    }
}

#[test]
fn effect_with_no_emitters_produces_no_particles() {
    let effect = VfxEffect {
        name: "empty".into(),
        emitters: vec![],
    };
    let mut inst = VfxInstance::new(effect, Vec3::ZERO);
    inst.tick(1.0);
    assert_eq!(inst.particle_count(), 0);
    assert!(inst.to_splats().is_empty());
    assert!(inst.is_finished());
}

#[test]
fn to_splats_count_matches_particle_count() {
    let effect = effect_smoke();
    let mut inst = VfxInstance::new(effect, Vec3::ZERO);
    inst.tick(0.5);
    assert_eq!(inst.to_splats().len(), inst.particle_count());
}

#[test]
fn prebuilt_effects_have_reasonable_defaults() {
    let effects = vec![
        effect_fire(),
        effect_smoke(),
        effect_explosion(),
        effect_sparkle(),
        effect_rain(),
        effect_dust(),
    ];

    for effect in &effects {
        assert!(!effect.name.is_empty(), "effect must have a name");
        assert!(!effect.emitters.is_empty(), "effect {} must have emitters", effect.name);

        for emitter in &effect.emitters {
            assert!(emitter.max_particles > 0, "{}: max_particles must be > 0", effect.name);
            assert!(emitter.lifetime.min > 0.0, "{}: lifetime.min must be > 0", effect.name);
            assert!(
                emitter.lifetime.max >= emitter.lifetime.min,
                "{}: lifetime.max must be >= min",
                effect.name
            );
            assert!(
                !emitter.size.keys.is_empty(),
                "{}: size curve must have keys",
                effect.name
            );
            assert!(
                !emitter.opacity.keys.is_empty(),
                "{}: opacity curve must have keys",
                effect.name
            );
        }
    }
}

#[test]
fn serialization_roundtrip() {
    let effect = effect_fire();
    let json = serde_json::to_string(&effect).expect("serialize");
    let deserialized: VfxEffect = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.name, "fire");
    assert_eq!(deserialized.emitters.len(), 1);
}

#[test]
fn rain_effect_particles_move_downward() {
    let effect = effect_rain();
    let mut inst = VfxInstance::new(effect, Vec3::new(0.0, 50.0, 0.0));
    inst.tick(0.1);
    inst.tick(0.1);

    let splats = inst.to_splats();
    assert!(!splats.is_empty());

    // Rain particles should have moved below spawn height
    let below_spawn = splats.iter().filter(|s| s.position[1] < 50.0).count();
    assert!(
        below_spawn > 0,
        "rain particles should move downward"
    );
}
