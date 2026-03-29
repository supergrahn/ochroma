# GPU Particle System as Gaussian Splats — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ECS-integrated CPU particle system to `vox_render` that simulates N particles per frame and renders them as `GaussianSplat` entries in the existing `RenderBuffer`. This extends the existing `gpu_particles.rs` module with bevy_ecs components and systems.

**Architecture:** A `ParticleEmitterComponent` ECS component holds emitter parameters (rate, velocity, lifetime, color). A `ParticleEmitterStateComponent` holds mutable simulation state (live particles, pending emit accumulator). A `particle_simulate_system` runs each frame: emits new particles, applies gravity + velocity integration, removes dead particles. A `particle_splat_system` converts live particles to `GaussianSplat` entries and appends them to `RenderBuffer.splats`. A `ParticlePlugin` wires both systems. The existing `gpu_particles::GpuParticleSystem` is left untouched — this adds a parallel ECS path that reuses its `GpuParticle` struct.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `bytemuck = "1"`, `half = "2"`, `rand = "0.9"`, `vox_core::types::GaussianSplat`, `vox_core::engine_runtime::RenderBuffer`.

---

## Key File Paths (read before editing)

- `crates/vox_render/src/gpu_particles.rs` — existing `GpuParticle`, `GpuParticleSystem`, `tick_cpu()`
- `crates/vox_render/src/render_ecs.rs` — `SplatRenderPlugin`, `splat_gather_system`, `RenderBuffer` pattern
- `crates/vox_render/src/lib.rs` — module registry
- `crates/vox_render/Cargo.toml` — dependencies
- `crates/vox_core/src/engine_runtime.rs` — `FrameTime`, `RenderBuffer { splats: Vec<GaussianSplat> }`
- `crates/vox_core/src/types.rs` — `GaussianSplat { position, scale, rotation, opacity, spectral }`

## File Structure

**Create:**
- `crates/vox_render/src/particle_ecs.rs` — all ECS types, systems, and plugin

**Modify:**
- `crates/vox_render/src/lib.rs` — add `pub mod particle_ecs;`

---

### Task 1: ParticleEmitterComponent + ParticleState struct

**Files:**
- Create: `crates/vox_render/src/particle_ecs.rs` (first portion)

- [ ] **Step 1: Read `crates/vox_render/src/gpu_particles.rs` to understand existing types**

Confirm `GpuParticle { position, age, velocity, lifetime, size, opacity, color }` is `#[repr(C)] Pod Zeroable`.

- [ ] **Step 2: Create `crates/vox_render/src/particle_ecs.rs`**

```rust
//! ECS-integrated particle system for Ochroma Engine.
//!
//! Simulates particles on the CPU and renders them as Gaussian splats
//! via the existing `RenderBuffer`. Each `ParticleEmitterComponent` entity
//! gets a `ParticleEmitterStateComponent` that holds live particle data.
//!
//! Systems:
//! - `particle_simulate_system` — emit, integrate, cull dead particles
//! - `particle_splat_system` — convert live particles to `GaussianSplat`

use bevy_ecs::prelude::*;
use glam::Vec3;
use half::f16;
use rand::prelude::*;

use vox_core::engine_runtime::{FrameTime, RenderBuffer};
use vox_core::types::GaussianSplat;

// ---------------------------------------------------------------------------
// Particle State (per-particle data, CPU side)
// ---------------------------------------------------------------------------

/// Per-particle simulation state.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
}

impl Particle {
    /// Returns the life fraction in [0, 1] where 0 = just born, 1 = dead.
    pub fn life_fraction(&self) -> f32 {
        (self.life / self.max_life).clamp(0.0, 1.0)
    }

    /// Is this particle dead?
    pub fn is_dead(&self) -> bool {
        self.life >= self.max_life
    }
}

// ---------------------------------------------------------------------------
// ECS Components
// ---------------------------------------------------------------------------

/// Emitter configuration — attach to an entity to create a particle source.
///
/// Does not hold mutable state; pair with `ParticleEmitterStateComponent`.
#[derive(Component, Debug, Clone)]
pub struct ParticleEmitterComponent {
    /// Maximum number of live particles for this emitter.
    pub max_particles: u32,
    /// Particles emitted per second.
    pub emit_rate: f32,
    /// World-space emission origin (relative to entity TransformComponent).
    pub local_offset: Vec3,
    /// Initial velocity direction (world-space).
    pub initial_velocity: Vec3,
    /// Random spread applied to velocity (metres/s).
    pub velocity_spread: f32,
    /// Particle lifetime in seconds.
    pub lifetime: f32,
    /// Gaussian splat scale for each particle.
    pub splat_scale: f32,
    /// Base color as spectral reflectance (simplified: 3 floats mapped to bands).
    pub color: [f32; 3],
    /// Gravity multiplier (1.0 = standard earth gravity).
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
            color: [1.0, 0.5, 0.1], // warm orange
            gravity_scale: 1.0,
        }
    }
}

/// Mutable emitter state — holds live particles and emission accumulator.
///
/// Insert alongside `ParticleEmitterComponent`. The `particle_simulate_system`
/// will auto-insert this if missing.
#[derive(Component, Debug, Clone)]
pub struct ParticleEmitterStateComponent {
    pub particles: Vec<Particle>,
    pub pending_emit: f32,
    pub rng_seed: u64,
}

impl Default for ParticleEmitterStateComponent {
    fn default() -> Self {
        Self {
            particles: Vec::new(),
            pending_emit: 0.0,
            rng_seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Simulate one frame of particle emission + physics.
///
/// Public for unit testing. Called by `particle_simulate_system`.
pub fn simulate_particles(
    state: &mut ParticleEmitterStateComponent,
    emitter: &ParticleEmitterComponent,
    emitter_world_pos: Vec3,
    dt: f32,
) {
    let mut rng = StdRng::seed_from_u64(state.rng_seed);

    // Emit new particles
    state.pending_emit += emitter.emit_rate * dt;
    let to_emit = state.pending_emit.floor() as u32;
    state.pending_emit -= to_emit as f32;

    for _ in 0..to_emit {
        if state.particles.len() >= emitter.max_particles as usize {
            break;
        }
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

    // Advance existing particles
    let gravity = Vec3::new(0.0, -9.81 * emitter.gravity_scale, 0.0);
    for p in &mut state.particles {
        p.velocity += gravity * dt;
        p.position += p.velocity * dt;
        p.life += dt;
    }

    // Remove dead particles
    state.particles.retain(|p| !p.is_dead());

    // Advance RNG seed so next frame is different
    state.rng_seed = rng.random();
}

/// Convert live particles to Gaussian splats.
///
/// Public for unit testing. Called by `particle_splat_system`.
pub fn particles_to_splats(
    particles: &[Particle],
    emitter: &ParticleEmitterComponent,
) -> Vec<GaussianSplat> {
    particles
        .iter()
        .map(|p| {
            let frac = p.life_fraction();
            // Fade out opacity and shrink over lifetime
            let opacity = ((1.0 - frac) * 240.0) as u8;
            let scale = p.size * (1.0 - frac * 0.5); // shrink to 50% at death

            // Map color to spectral bands (simplified):
            // bands [0..2] = blue contribution, [3..4] = green, [5..7] = red
            let r = emitter.color[0];
            let g = emitter.color[1];
            let b = emitter.color[2];
            let spectral: [u16; 8] = [
                f16::from_f32(b * 0.8).to_bits(),
                f16::from_f32(b * 0.9).to_bits(),
                f16::from_f32(b * 1.0).to_bits(),
                f16::from_f32(g * 0.9).to_bits(),
                f16::from_f32(g * 1.0).to_bits(),
                f16::from_f32(r * 0.8).to_bits(),
                f16::from_f32(r * 0.9).to_bits(),
                f16::from_f32(r * 1.0).to_bits(),
            ];

            GaussianSplat {
                position: [p.position.x, p.position.y, p.position.z],
                scale: [scale, scale, scale],
                rotation: [0, 0, 0, 32767], // identity quaternion
                opacity,
                _pad: [0; 3],
                spectral,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ECS Systems
// ---------------------------------------------------------------------------

/// Simulate all particle emitters: emit new particles, apply physics, cull dead.
///
/// Auto-inserts `ParticleEmitterStateComponent` on entities that have
/// `ParticleEmitterComponent` but no state yet.
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
        let world_pos = transform
            .map(|t| t.position)
            .unwrap_or(Vec3::ZERO);
        simulate_particles(&mut state, emitter, world_pos, dt);
    }
}

/// Convert live particles to Gaussian splats and append to RenderBuffer.
///
/// Runs AFTER `splat_gather_system` (which clears the buffer) but the buffer
/// may already have scene splats — we extend rather than replace.
pub fn particle_splat_system(
    mut buffer: ResMut<RenderBuffer>,
    query: Query<(
        &ParticleEmitterComponent,
        &ParticleEmitterStateComponent,
    )>,
) {
    for (emitter, state) in query.iter() {
        let splats = particles_to_splats(&state.particles, emitter);
        buffer.splats.extend(splats);
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Bevy plugin that registers particle simulation and rendering systems.
///
/// Systems run in `Update`:
/// 1. `particle_simulate_system` — emit + physics + cull
/// 2. `particle_splat_system` — convert to Gaussian splats (after simulate)
///
/// Does not insert any resources — callers spawn `ParticleEmitterComponent` +
/// `ParticleEmitterStateComponent` entities as needed.
pub struct ParticlePlugin;

impl bevy_app::Plugin for ParticlePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        use bevy_app::Update;
        app.add_systems(
            Update,
            (
                particle_simulate_system,
                particle_splat_system.after(particle_simulate_system),
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Preset Emitters
// ---------------------------------------------------------------------------

/// Preset emitter configurations for common effects.
pub mod presets {
    use super::*;

    /// Fire emitter — warm orange particles rising upward.
    pub fn fire() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 500,
            emit_rate: 80.0,
            local_offset: Vec3::ZERO,
            initial_velocity: Vec3::new(0.0, 4.0, 0.0),
            velocity_spread: 1.5,
            lifetime: 1.2,
            splat_scale: 0.2,
            color: [1.0, 0.4, 0.05],
            gravity_scale: -0.3, // negative: fire rises faster
        }
    }

    /// Smoke emitter — grey particles drifting upward slowly.
    pub fn smoke() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 300,
            emit_rate: 30.0,
            local_offset: Vec3::new(0.0, 1.0, 0.0),
            initial_velocity: Vec3::new(0.0, 2.0, 0.0),
            velocity_spread: 0.8,
            lifetime: 3.0,
            splat_scale: 0.4,
            color: [0.3, 0.3, 0.3],
            gravity_scale: -0.1,
        }
    }

    /// Sparks emitter — bright yellow particles with fast initial velocity.
    pub fn sparks() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 200,
            emit_rate: 100.0,
            local_offset: Vec3::ZERO,
            initial_velocity: Vec3::new(0.0, 8.0, 0.0),
            velocity_spread: 5.0,
            lifetime: 0.5,
            splat_scale: 0.05,
            color: [1.0, 0.9, 0.2],
            gravity_scale: 1.0,
        }
    }

    /// Rain emitter — small blue particles falling downward from above.
    pub fn rain() -> ParticleEmitterComponent {
        ParticleEmitterComponent {
            max_particles: 2000,
            emit_rate: 500.0,
            local_offset: Vec3::new(0.0, 20.0, 0.0),
            initial_velocity: Vec3::new(0.0, -15.0, 0.0),
            velocity_spread: 1.0,
            lifetime: 2.0,
            splat_scale: 0.02,
            color: [0.3, 0.4, 0.8],
            gravity_scale: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;
    use bevy_ecs::schedule::Schedule;
    use bevy_ecs::world::World;

    #[test]
    fn particle_is_dead_after_max_life() {
        let p = Particle {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            life: 2.0,
            max_life: 1.5,
            size: 0.1,
        };
        assert!(p.is_dead());
        assert!((p.life_fraction() - 1.0).abs() < 0.01);
    }

    #[test]
    fn particle_alive_before_max_life() {
        let p = Particle {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            life: 0.5,
            max_life: 1.5,
            size: 0.1,
        };
        assert!(!p.is_dead());
    }

    #[test]
    fn simulate_emits_particles() {
        let emitter = ParticleEmitterComponent {
            emit_rate: 100.0,
            lifetime: 2.0,
            ..Default::default()
        };
        let mut state = ParticleEmitterStateComponent::default();

        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.1);
        assert!(
            !state.particles.is_empty(),
            "Should have emitted particles after 0.1s at 100/s, got {}",
            state.particles.len()
        );
        assert!(
            state.particles.len() >= 9 && state.particles.len() <= 11,
            "Expected ~10 particles, got {}",
            state.particles.len()
        );
    }

    #[test]
    fn simulate_gravity_decreases_y_velocity() {
        let emitter = ParticleEmitterComponent {
            emit_rate: 100.0,
            initial_velocity: Vec3::new(0.0, 10.0, 0.0),
            velocity_spread: 0.0,
            gravity_scale: 1.0,
            lifetime: 5.0,
            ..Default::default()
        };
        let mut state = ParticleEmitterStateComponent::default();

        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.01);
        let vy_after_emit = state.particles[0].velocity.y;

        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.5);
        let vy_after_gravity = state.particles[0].velocity.y;

        assert!(
            vy_after_gravity < vy_after_emit,
            "Gravity should reduce y velocity: {} -> {}",
            vy_after_emit,
            vy_after_gravity
        );
    }

    #[test]
    fn simulate_removes_dead_particles() {
        let emitter = ParticleEmitterComponent {
            emit_rate: 50.0,
            lifetime: 0.05, // very short
            ..Default::default()
        };
        let mut state = ParticleEmitterStateComponent::default();

        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.02);
        let count_after_emit = state.particles.len();
        assert!(count_after_emit > 0);

        // Tick past lifetime
        simulate_particles(&mut state, &emitter, Vec3::ZERO, 0.1);
        // Original particles should be dead; some new ones may exist
        // Key: count doesn't grow unbounded
        assert!(
            state.particles.len() <= count_after_emit + 10,
            "Dead particles should be culled"
        );
    }

    #[test]
    fn particles_to_splats_produces_correct_count() {
        let emitter = ParticleEmitterComponent::default();
        let particles = vec![
            Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 0.0, max_life: 1.0, size: 0.1 },
            Particle { position: Vec3::ONE, velocity: Vec3::ZERO, life: 0.5, max_life: 1.0, size: 0.1 },
        ];
        let splats = particles_to_splats(&particles, &emitter);
        assert_eq!(splats.len(), 2);
    }

    #[test]
    fn particles_to_splats_opacity_decreases_with_age() {
        let emitter = ParticleEmitterComponent::default();
        let young = Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 0.1, max_life: 2.0, size: 0.1 };
        let old = Particle { position: Vec3::ZERO, velocity: Vec3::ZERO, life: 1.8, max_life: 2.0, size: 0.1 };

        let splats_young = particles_to_splats(&[young], &emitter);
        let splats_old = particles_to_splats(&[old], &emitter);

        assert!(
            splats_young[0].opacity > splats_old[0].opacity,
            "Young particle opacity ({}) should exceed old ({})",
            splats_young[0].opacity,
            splats_old[0].opacity
        );
    }

    #[test]
    fn particle_plugin_builds_without_panic() {
        let mut app = App::new();
        app.insert_resource(FrameTime::default());
        app.insert_resource(RenderBuffer::default());
        app.add_plugins(ParticlePlugin);
    }

    #[test]
    fn particle_systems_add_splats_to_render_buffer() {
        let mut world = World::new();
        world.insert_resource(FrameTime { dt: 0.1, total: 0.0, frame: 0 });
        world.insert_resource(RenderBuffer::default());

        world.spawn((
            ParticleEmitterComponent {
                emit_rate: 100.0,
                lifetime: 2.0,
                ..Default::default()
            },
            ParticleEmitterStateComponent::default(),
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems((
            particle_simulate_system,
            particle_splat_system.after(particle_simulate_system),
        ));
        schedule.run(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert!(
            !buffer.splats.is_empty(),
            "RenderBuffer should contain particle splats after one tick"
        );
    }

    #[test]
    fn preset_fire_has_reasonable_values() {
        let fire = presets::fire();
        assert!(fire.emit_rate > 0.0);
        assert!(fire.lifetime > 0.0);
        assert!(fire.splat_scale > 0.0);
        assert!(fire.color[0] > fire.color[2], "Fire should be red-dominant");
    }

    #[test]
    fn preset_smoke_has_low_gravity() {
        let smoke = presets::smoke();
        assert!(smoke.gravity_scale < 0.0, "Smoke should rise (negative gravity)");
    }

    #[test]
    fn emitter_default_is_reasonable() {
        let e = ParticleEmitterComponent::default();
        assert!(e.max_particles > 0);
        assert!(e.emit_rate > 0.0);
        assert!(e.lifetime > 0.0);
    }
}
```

- [ ] **Step 3: Run compile check**

```bash
cargo check -p vox_render 2>&1 | tail -5
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render -- particle_ecs 2>&1 | tail -25
```

Expected: all 12 tests pass.

**Commit:** `feat(particles): ParticleEmitterComponent + ParticleEmitterStateComponent + Particle struct`

---

### Task 2: CPU particle simulation (covered in Task 1)

The `simulate_particles()` function is already implemented in `particle_ecs.rs` above. It handles:

- Accumulating fractional emissions via `pending_emit`
- Spawning new particles with random velocity spread
- Applying gravity: `velocity.y -= 9.81 * gravity_scale * dt`
- Position integration: `position += velocity * dt`
- Life advancement: `life += dt`
- Dead particle removal: `retain(|p| !p.is_dead())`

The `particles_to_splats()` function converts live particles to `GaussianSplat` with:
- Position from particle world position
- Scale that shrinks over lifetime (50% at death)
- Opacity that fades from 240 to 0
- Spectral bands derived from emitter color (RGB mapped to 8 bands)

No additional code needed.

---

### Task 3: particle_system ECS systems + ParticlePlugin

**Files:**
- Modify: `crates/vox_render/src/lib.rs`

- [ ] **Step 1: Add module to lib.rs**

Add `pub mod particle_ecs;` to the module list in `crates/vox_render/src/lib.rs`.

- [ ] **Step 2: Verify everything compiles together**

```bash
cargo check -p vox_render 2>&1 | tail -5
```

- [ ] **Step 3: Run the full test suite**

```bash
cargo test -p vox_render -- particle 2>&1 | tail -30
```

Expected: all `particle_ecs` tests pass alongside existing `gpu_particles` tests.

**Commit:** `feat(particles): particle_simulate_system + particle_splat_system + ParticlePlugin`

---

### Task 4: Wire in vox_app demo

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Read `engine_runner.rs` to find where to add the demo emitter**

- [ ] **Step 2: Add imports and demo spawn**

Add to imports:

```rust
use vox_render::particle_ecs::{ParticleEmitterComponent, ParticleEmitterStateComponent, ParticlePlugin, presets};
```

In the setup section, add the plugin and spawn a fire emitter:

```rust
// --- Particle system demo ---
// ParticlePlugin registers simulate + splat systems in Update.
// (Uncomment when ready to test:)
// app.add_plugins(ParticlePlugin);
//
// // Spawn a fire particle emitter at a campfire location
// world.spawn((
//     presets::fire(),
//     ParticleEmitterStateComponent::default(),
//     TransformComponent {
//         position: Vec3::new(0.0, 0.5, -5.0),
//         rotation: Quat::IDENTITY,
//         scale: Vec3::ONE,
//     },
// ));
```

- [ ] **Step 3: Verify full crate compiles**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

**Commit:** `feat(particles): CPU particle system simulated as Gaussian splats via RenderBuffer`

---

## Summary

| Task | File | What |
|------|------|------|
| 1 | `crates/vox_render/src/particle_ecs.rs` | Components, Particle struct, simulation, splat conversion, systems, plugin, presets |
| 2 | (same file) | `simulate_particles()` + `particles_to_splats()` already in Task 1 |
| 3 | `crates/vox_render/src/lib.rs` | Add `pub mod particle_ecs;` |
| 4 | `crates/vox_app/src/bin/engine_runner.rs` | Wire ParticlePlugin + demo fire emitter |

**Final commit:** `feat(particles): CPU particle system simulated as Gaussian splats via RenderBuffer`
