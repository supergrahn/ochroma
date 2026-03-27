use vox_render::particles::{ParticleSystem, ParticleEmitter};
use glam::Vec3;

#[test]
fn emitter_produces_particles() {
    let mut sys = ParticleSystem::new(1000);
    sys.add_emitter(ParticleEmitter::smoke(Vec3::new(0.0, 5.0, 0.0)));
    sys.tick(1.0);
    assert!(sys.particle_count() > 0);
}

#[test]
fn particles_die_after_lifetime() {
    let mut sys = ParticleSystem::new(1000);
    sys.add_emitter(ParticleEmitter::dust(Vec3::ZERO));
    sys.tick(0.1); // spawn some
    let count_after_spawn = sys.particle_count();
    assert!(count_after_spawn > 0);

    // Remove emitter so no new ones spawn
    sys.emitters.clear();
    for _ in 0..100 { sys.tick(0.1); } // 10 seconds
    assert_eq!(sys.particle_count(), 0, "All particles should have expired");
}

#[test]
fn max_particles_respected() {
    let mut sys = ParticleSystem::new(10);
    sys.add_emitter(ParticleEmitter::smoke(Vec3::ZERO));
    for _ in 0..100 { sys.tick(0.1); }
    assert!(sys.particle_count() <= 10);
}

#[test]
fn to_splats_matches_count() {
    let mut sys = ParticleSystem::new(1000);
    sys.add_emitter(ParticleEmitter::smoke(Vec3::ZERO));
    sys.tick(1.0);
    let splats = sys.to_splats();
    assert_eq!(splats.len(), sys.particle_count());
}
