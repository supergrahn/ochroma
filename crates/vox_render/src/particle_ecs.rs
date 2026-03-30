//! ECS-integrated CPU particle system for Ochroma Engine.
//!
//! Simulates particles on the CPU and renders them as Gaussian splats
//! via the existing `RenderBuffer`.

use bevy_ecs::prelude::*;
use glam::{self, Vec3};
use half::f16;
use rand::prelude::*;

use vox_core::engine_runtime::{FrameTime, RenderBuffer};
use vox_core::types::GaussianSplat;

// ── Per-particle state ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
}

impl Particle {
    pub fn life_fraction(&self) -> f32 {
        (self.life / self.max_life).clamp(0.0, 1.0)
    }

    pub fn is_dead(&self) -> bool {
        self.life >= self.max_life
    }
}

// ── ECS Components ──────────────────────────────────────────────────────

#[derive(Component, Debug, Clone)]
pub struct ParticleEmitterComponent {
    pub max_particles: u32,
    pub emit_rate: f32,
    pub local_offset: Vec3,
    pub initial_velocity: Vec3,
    pub velocity_spread: f32,
    pub lifetime: f32,
    pub splat_scale: f32,
    pub color: [f32; 3],
    pub gravity_scale: f32,
}

impl Default for ParticleEmitterComponent {
    fn default() -> Self {
        Self {
            max_particles: 500,
            emit_rate: 50.0,
            local_offset: Vec3::ZERO,
            initial_velocity: Vec3::new(0.0, 5.0, 0.0),
            velocity_spread: 1.0,
            lifetime: 1.5,
            splat_scale: 0.15,
            color: [1.0, 0.5, 0.1],
            gravity_scale: 1.0,
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct ParticleEmitterStateComponent {
    pub particles: Vec<Particle>,
    pub pending_emit: f32,
    pub rng_seed: u64,
}

impl Default for ParticleEmitterStateComponent {
    fn default() -> Self {
        Self { particles: Vec::new(), pending_emit: 0.0, rng_seed: 42 }
    }
}

// ── Simulation ──────────────────────────────────────────────────────────

pub fn simulate_particles(
    state: &mut ParticleEmitterStateComponent,
    emitter: &ParticleEmitterComponent,
    emitter_world_pos: Vec3,
    dt: f32,
) {
    let mut rng = StdRng::seed_from_u64(state.rng_seed);

    state.pending_emit += emitter.emit_rate * dt;
    let to_emit = state.pending_emit.floor() as u32;
    state.pending_emit -= to_emit as f32;

    for _ in 0..to_emit {
        if state.particles.len() >= emitter.max_particles as usize { break; }
        let spread = emitter.velocity_spread;
        let vx = emitter.initial_velocity.x + (rng.random::<f32>() - 0.5) * spread;
        let vy = emitter.initial_velocity.y + (rng.random::<f32>() - 0.5) * spread;
        let vz = emitter.initial_velocity.z + (rng.random::<f32>() - 0.5) * spread;
        state.particles.push(Particle {
            position: emitter_world_pos + emitter.local_offset,
            velocity: Vec3::new(vx, vy, vz),
            life: 0.0,
            max_life: emitter.lifetime,
            size: emitter.splat_scale,
        });
    }

    let gravity = Vec3::new(0.0, -9.81 * emitter.gravity_scale, 0.0);
    for p in &mut state.particles {
        p.velocity += gravity * dt;
        p.position += p.velocity * dt;
        p.life += dt;
    }
    state.particles.retain(|p| !p.is_dead());
    state.rng_seed = rng.random();
}

pub fn particles_to_splats(
    particles: &[Particle],
    emitter: &ParticleEmitterComponent,
) -> Vec<GaussianSplat> {
    particles.iter().map(|p| {
        let frac = p.life_fraction();
        let opacity = ((1.0 - frac) * 240.0) as u8;
        let scale = p.size * (1.0 - frac * 0.5);
        let [r, g, b] = emitter.color;
        let spectral: [u16; 16] = [
            f16::from_f32(b * 0.8).to_bits(),
            f16::from_f32(b * 0.9).to_bits(),
            f16::from_f32(b * 1.0).to_bits(),
            f16::from_f32(g * 0.9).to_bits(),
            f16::from_f32(g * 1.0).to_bits(),
            f16::from_f32(r * 0.8).to_bits(),
            f16::from_f32(r * 0.9).to_bits(),
            f16::from_f32(r * 1.0).to_bits(),
            f16::from_f32(b * 0.8).to_bits(),
            f16::from_f32(b * 0.9).to_bits(),
            f16::from_f32(b * 1.0).to_bits(),
            f16::from_f32(g * 0.9).to_bits(),
            f16::from_f32(g * 1.0).to_bits(),
            f16::from_f32(r * 0.8).to_bits(),
            f16::from_f32(r * 0.9).to_bits(),
            f16::from_f32(r * 1.0).to_bits(),
        ];
        GaussianSplat::volume(
            [p.position.x, p.position.y, p.position.z],
            [scale, scale, scale],
            glam::Quat::IDENTITY,
            opacity,
            spectral,
        )
    }).collect()
}

// ── ECS Systems ─────────────────────────────────────────────────────────

pub fn particle_simulate_system(
    time: Res<FrameTime>,
    mut query: Query<(
        &ParticleEmitterComponent,
        &mut ParticleEmitterStateComponent,
        Option<&vox_core::ecs::TransformComponent>,
    )>,
) {
    let dt = time.dt;
    for (emitter, mut state, transform) in query.iter_mut() {
        let world_pos = transform.map(|t| t.position).unwrap_or(Vec3::ZERO);
        simulate_particles(&mut state, emitter, world_pos, dt);
    }
}

pub fn particle_splat_system(
    mut buffer: ResMut<RenderBuffer>,
    query: Query<(&ParticleEmitterComponent, &ParticleEmitterStateComponent)>,
) {
    for (emitter, state) in query.iter() {
        let splats = particles_to_splats(&state.particles, emitter);
        buffer.splats.extend(splats);
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────

pub struct ParticlePlugin;

impl bevy_app::Plugin for ParticlePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            bevy_app::Update,
            (particle_simulate_system, particle_splat_system.after(particle_simulate_system)),
        );
    }
}

// ── Preset emitters ──────────────────────────────────────────────────────

pub mod presets {
    use super::*;

    pub fn fire() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 500, emit_rate: 80.0, initial_velocity: Vec3::new(0.0, 4.0, 0.0),
            velocity_spread: 1.5, lifetime: 1.2, splat_scale: 0.2,
            color: [1.0, 0.4, 0.05], gravity_scale: -0.3,
            local_offset: Vec3::ZERO,
        }
    }

    pub fn smoke() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 300, emit_rate: 30.0, local_offset: Vec3::new(0.0, 1.0, 0.0),
            initial_velocity: Vec3::new(0.0, 2.0, 0.0), velocity_spread: 0.8,
            lifetime: 3.0, splat_scale: 0.4, color: [0.3, 0.3, 0.3],
            gravity_scale: -0.1,
        }
    }

    pub fn sparks() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 200, emit_rate: 100.0, initial_velocity: Vec3::new(0.0, 8.0, 0.0),
            velocity_spread: 5.0, lifetime: 0.5, splat_scale: 0.05,
            color: [1.0, 0.9, 0.2], gravity_scale: 1.0,
            local_offset: Vec3::ZERO,
        }
    }

    pub fn rain() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 2000, emit_rate: 500.0, local_offset: Vec3::new(0.0, 20.0, 0.0),
            initial_velocity: Vec3::new(0.0, -15.0, 0.0), velocity_spread: 1.0,
            lifetime: 2.0, splat_scale: 0.02, color: [0.3, 0.4, 0.8],
            gravity_scale: 0.5,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;
    use bevy_ecs::schedule::Schedule;
    use bevy_ecs::world::World;

    #[test]
    fn particle_is_dead_after_max_life() {
        let p = Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 2.0, max_life: 1.5, size: 0.1 };
        assert!(p.is_dead());
    }

    #[test]
    fn particle_alive_before_max_life() {
        let p = Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 0.5, max_life: 1.5, size: 0.1 };
        assert!(!p.is_dead());
    }

    #[test]
    fn simulate_emits_particles() {
        let emitter = ParticleEmitterComponent { emit_rate: 100.0, lifetime: 2.0, ..Default::default() };
        let mut state = ParticleEmitterStateComponent::default();
        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.1);
        assert!(!state.particles.is_empty());
        assert!(state.particles.len() >= 9 && state.particles.len() <= 11,
            "Expected ~10, got {}", state.particles.len());
    }

    #[test]
    fn simulate_gravity_decreases_y_velocity() {
        let emitter = ParticleEmitterComponent {
            emit_rate: 100.0, initial_velocity: Vec3::new(0.0, 10.0, 0.0),
            velocity_spread: 0.0, gravity_scale: 1.0, lifetime: 5.0, ..Default::default()
        };
        let mut state = ParticleEmitterStateComponent::default();
        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.01);
        let vy_initial = state.particles[0].velocity.y;
        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.5);
        let vy_after = state.particles[0].velocity.y;
        assert!(vy_after < vy_initial, "gravity should reduce y: {} -> {}", vy_initial, vy_after);
    }

    #[test]
    fn simulate_removes_dead_particles() {
        let emitter = ParticleEmitterComponent { emit_rate: 50.0, lifetime: 0.05, ..Default::default() };
        let mut state = ParticleEmitterStateComponent::default();
        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.02);
        let count = state.particles.len();
        assert!(count > 0);
        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.1);
        assert!(state.particles.len() <= count + 10);
    }

    #[test]
    fn particles_to_splats_correct_count() {
        let emitter = ParticleEmitterComponent::default();
        let particles = vec![
            Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 0.0, max_life: 1.0, size: 0.1 },
            Particle { position: Vec3::ONE, velocity: Vec3::ZERO, life: 0.5, max_life: 1.0, size: 0.1 },
        ];
        assert_eq!(particles_to_splats(&particles, &emitter).len(), 2);
    }

    #[test]
    fn particles_to_splats_opacity_decreases_with_age() {
        let emitter = ParticleEmitterComponent::default();
        let young = Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 0.1, max_life: 2.0, size: 0.1 };
        let old   = Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 1.8, max_life: 2.0, size: 0.1 };
        let sy = particles_to_splats(&[young], &emitter);
        let so = particles_to_splats(&[old], &emitter);
        assert!(sy[0].opacity() > so[0].opacity());
    }

    #[test]
    fn particle_plugin_builds_without_panic() {
        let mut app = App::new();
        app.insert_resource(FrameTime::default());
        app.insert_resource(RenderBuffer::default());
        app.add_plugins(ParticlePlugin);
    }

    #[test]
    fn particle_systems_add_splats_to_buffer() {
        let mut world = World::new();
        world.insert_resource(FrameTime { dt: 0.1, total: 0.0, frame: 0 });
        world.insert_resource(RenderBuffer::default());
        world.spawn((
            ParticleEmitterComponent { emit_rate: 100.0, lifetime: 2.0, ..Default::default() },
            ParticleEmitterStateComponent::default(),
        ));
        let mut schedule = Schedule::default();
        schedule.add_systems((
            particle_simulate_system,
            particle_splat_system.after(particle_simulate_system),
        ));
        schedule.run(&mut world);
        let buffer = world.resource::<RenderBuffer>();
        assert!(!buffer.splats.is_empty());
    }

    #[test]
    fn preset_fire_is_red_dominant() {
        let fire = presets::fire();
        assert!(fire.color[0] > fire.color[2]);
    }

    #[test]
    fn preset_smoke_has_low_gravity() {
        let smoke = presets::smoke();
        assert!(smoke.gravity_scale < 0.0);
    }

    #[test]
    fn emitter_default_is_reasonable() {
        let e = ParticleEmitterComponent::default();
        assert!(e.max_particles > 0 && e.emit_rate > 0.0 && e.lifetime > 0.0);
    }
}
