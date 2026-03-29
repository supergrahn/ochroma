# Splat Particle System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A particle system where each particle IS a Gaussian splat — rendered through the existing EWA pipeline with no separate pass, spectral emission drives audio synthesis on impact.

**Architecture:** `SplatEmitter` owns a pool of `SplatParticle` structs (position, velocity, lifetime, base spectral). Each frame, `SplatEmitter::tick(dt)` integrates physics and returns a `Vec<GaussianSplat>` for injection into the scene splat list before the EWA render. Opacity decays with lifetime: `splat.opacity = base_opacity * (remaining / lifetime)`. Spectral emission is per-emitter — fire emitters have high red/orange bands, sparks have high blue/white bands. On collision (lifetime end), `vox_audio::synthesize_impact` is called using the particle's spectral bands to produce the correct material-resonant impact sound. The existing `gpu_particles` module stub is augmented with the CPU-side emitter.

**Why better than Unreal:** Unreal Niagara is a separate system rendered as billboarded sprites (or mesh particles). Ochroma splat particles composite identically to scene geometry — no z-fighting, no billboarding artifacts, correct EWA soft-edge blending, and the spectral bands physically connect visual appearance to audio. A glass particle sounds like glass; a rock particle sounds like rock.

**Tech Stack:** Rust, `vox_core::types::GaussianSplat`, `vox_audio::synthesize_impact` (existing), `half::f16` (existing workspace dep).

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_render/src/splat_particles.rs` | Create | `SplatEmitter`, `SplatParticle`, `EmitterConfig` |
| `crates/vox_render/src/lib.rs` | Modify | `pub mod splat_particles;` |
| `crates/vox_app/src/bin/engine_runner.rs` | Modify | Wire emitters, inject particles before render |

---

## Task 1: SplatEmitter core

**Files:**
- Create: `crates/vox_render/src/splat_particles.rs`

- [ ] Create `crates/vox_render/src/splat_particles.rs`:

```rust
//! Gaussian splat particle system.
//!
//! Each particle is a `GaussianSplat` rendered through the standard EWA pipeline.
//! Particles are Gaussian splats — no separate render pass, no z-fighting.
//!
//! Spectral emission drives audio: on particle death, the caller can call
//! `vox_audio::synthesize_impact` with `particle.spectral` to get impact audio.

use vox_core::types::GaussianSplat;
use half::f16;

/// Configuration for a single emitter instance.
#[derive(Debug, Clone)]
pub struct EmitterConfig {
    /// World-space emission origin.
    pub origin: [f32; 3],
    /// Initial velocity range: particles are given velocity in
    /// `[base_velocity ± spread]` in each axis.
    pub base_velocity: [f32; 3],
    pub velocity_spread: f32,
    /// Particle lifetime range in seconds.
    pub min_lifetime: f32,
    pub max_lifetime: f32,
    /// Starting opacity [0, 255].
    pub base_opacity: u8,
    /// Scale of each particle splat.
    pub scale: [f32; 3],
    /// 8-band spectral profile for this emitter. Values in [0, 1].
    /// High blue (band 0) = glassy/electric, high red (band 7) = fire/rock.
    pub spectral: [f32; 8],
    /// Particles emitted per second.
    pub emit_rate: f32,
    /// Maximum live particles at once (pool size).
    pub max_particles: usize,
    /// Gravity acceleration (m/s² downward). Typical: -9.8.
    pub gravity: f32,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            base_velocity: [0.0, 3.0, 0.0],
            velocity_spread: 1.0,
            min_lifetime: 0.5,
            max_lifetime: 2.0,
            base_opacity: 200,
            scale: [0.05, 0.05, 0.05],
            spectral: [0.0, 0.0, 0.0, 0.0, 0.5, 0.8, 0.9, 0.3], // fire-ish
            emit_rate: 30.0,
            max_particles: 256,
            gravity: -9.8,
        }
    }
}

/// A single live particle.
#[derive(Debug, Clone)]
pub struct SplatParticle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub remaining: f32,
    pub lifetime: f32,
    pub spectral: [f32; 8],
    pub base_opacity: u8,
    pub scale: [f32; 3],
}

impl SplatParticle {
    /// Convert to `GaussianSplat` with opacity modulated by remaining lifetime.
    pub fn to_splat(&self) -> GaussianSplat {
        let t = (self.remaining / self.lifetime).clamp(0.0, 1.0);
        let opacity = (self.base_opacity as f32 * t) as u8;
        let spectral: [u16; 8] = std::array::from_fn(|i| {
            f16::from_f32(self.spectral[i].clamp(0.0, 1.0)).to_bits()
        });
        GaussianSplat {
            position: self.position,
            scale: self.scale,
            rotation: [0, 0, 0, 32767], // identity quaternion
            opacity,
            _pad: [0; 3],
            spectral,
        }
    }
}

/// Controls emission and updates live particles.
pub struct SplatEmitter {
    pub config: EmitterConfig,
    particles: Vec<SplatParticle>,
    /// Accumulated fractional particles to emit.
    emit_accum: f32,
    /// Simple deterministic "random" state (xorshift).
    rng: u64,
    /// Spectral bands of particles that died this frame (for audio).
    pub died_this_frame: Vec<[f32; 8]>,
}

impl SplatEmitter {
    pub fn new(config: EmitterConfig) -> Self {
        Self {
            particles: Vec::with_capacity(config.max_particles),
            emit_accum: 0.0,
            rng: 0xdeadbeefcafe1234,
            died_this_frame: Vec::new(),
            config,
        }
    }

    /// Advance simulation by `dt` seconds.
    /// Returns `&[SplatParticle]` — call `to_splat()` on each for rendering.
    pub fn tick(&mut self, dt: f32) {
        self.died_this_frame.clear();

        // Integrate existing particles
        self.particles.retain_mut(|p| {
            p.remaining -= dt;
            if p.remaining <= 0.0 {
                self.died_this_frame.push(p.spectral);
                return false;
            }
            p.velocity[1] += self.config.gravity * dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
            true
        });

        // Emit new particles
        self.emit_accum += self.config.emit_rate * dt;
        while self.emit_accum >= 1.0 && self.particles.len() < self.config.max_particles {
            self.emit_accum -= 1.0;
            self.particles.push(self.spawn_particle());
        }
        if self.emit_accum > self.config.emit_rate { self.emit_accum = 0.0; }
    }

    /// Collect current particles as `GaussianSplat` for injection into scene.
    pub fn splats(&self) -> Vec<GaussianSplat> {
        self.particles.iter().map(|p| p.to_splat()).collect()
    }

    pub fn live_count(&self) -> usize { self.particles.len() }

    fn spawn_particle(&mut self) -> SplatParticle {
        let spread = self.config.velocity_spread;
        let vx = self.config.base_velocity[0] + self.rand_f32(-spread, spread);
        let vy = self.config.base_velocity[1] + self.rand_f32(-spread, spread);
        let vz = self.config.base_velocity[2] + self.rand_f32(-spread, spread);
        let lifetime = self.rand_f32(self.config.min_lifetime, self.config.max_lifetime);
        SplatParticle {
            position: self.config.origin,
            velocity: [vx, vy, vz],
            remaining: lifetime,
            lifetime,
            spectral: self.config.spectral,
            base_opacity: self.config.base_opacity,
            scale: self.config.scale,
        }
    }

    fn rand_f32(&mut self, min: f32, max: f32) -> f32 {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let t = (self.rng as f32) / (u64::MAX as f32);
        min + t * (max - min)
    }
}

/// Preset emitter configurations.
impl EmitterConfig {
    /// Orange fire: high red/orange spectral bands.
    pub fn fire(origin: [f32; 3]) -> Self {
        Self {
            origin,
            base_velocity: [0.0, 2.0, 0.0],
            velocity_spread: 0.5,
            min_lifetime: 0.8,
            max_lifetime: 2.5,
            base_opacity: 180,
            scale: [0.08, 0.08, 0.08],
            spectral: [0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 1.0, 0.6],
            emit_rate: 40.0,
            max_particles: 300,
            gravity: -2.0, // fire rises slowly
        }
    }

    /// Blue electric sparks: high blue/violet bands.
    pub fn sparks(origin: [f32; 3]) -> Self {
        Self {
            origin,
            base_velocity: [0.0, 1.0, 0.0],
            velocity_spread: 3.0,
            min_lifetime: 0.1,
            max_lifetime: 0.4,
            base_opacity: 230,
            scale: [0.02, 0.02, 0.02],
            spectral: [1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0],
            emit_rate: 80.0,
            max_particles: 150,
            gravity: -9.8,
        }
    }

    /// Rocky debris: high red/brown bands, matching rock material audio.
    pub fn debris(origin: [f32; 3]) -> Self {
        Self {
            origin,
            base_velocity: [0.0, 4.0, 0.0],
            velocity_spread: 2.5,
            min_lifetime: 0.5,
            max_lifetime: 1.5,
            base_opacity: 200,
            scale: [0.1, 0.1, 0.1],
            spectral: [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.9],
            emit_rate: 15.0,
            max_particles: 80,
            gravity: -9.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_spawns_particles_over_time() {
        let mut emitter = SplatEmitter::new(EmitterConfig {
            emit_rate: 100.0,
            max_particles: 50,
            ..Default::default()
        });
        emitter.tick(0.2); // expect ~20 particles
        assert!(emitter.live_count() > 0, "should have particles after tick");
        assert!(emitter.live_count() <= 50, "should not exceed max_particles");
    }

    #[test]
    fn particles_die_after_lifetime() {
        let mut emitter = SplatEmitter::new(EmitterConfig {
            emit_rate: 100.0,
            max_particles: 10,
            min_lifetime: 0.1,
            max_lifetime: 0.1, // all die at exactly 0.1s
            ..Default::default()
        });
        emitter.tick(0.05);
        let count_before = emitter.live_count();
        emitter.tick(0.2); // advance past lifetime
        assert_eq!(emitter.live_count(), 0, "all particles should be dead");
        assert!(emitter.died_this_frame.len() <= count_before);
    }

    #[test]
    fn to_splat_opacity_decreases_with_age() {
        let p = SplatParticle {
            position: [0.0; 3],
            velocity: [0.0; 3],
            remaining: 0.5,
            lifetime: 1.0,
            spectral: [0.5; 8],
            base_opacity: 200,
            scale: [0.1; 3],
        };
        let splat = p.to_splat();
        assert!(splat.opacity < 200, "opacity should decrease with age");
        assert!(splat.opacity >= 95, "at half lifetime, opacity should be ~100");
    }

    #[test]
    fn to_splat_opacity_is_zero_at_death() {
        let p = SplatParticle {
            position: [0.0; 3],
            velocity: [0.0; 3],
            remaining: 0.0,
            lifetime: 1.0,
            spectral: [0.5; 8],
            base_opacity: 200,
            scale: [0.1; 3],
        };
        assert_eq!(p.to_splat().opacity, 0);
    }

    #[test]
    fn splats_returns_one_per_particle() {
        let mut emitter = SplatEmitter::new(EmitterConfig {
            emit_rate: 10.0,
            max_particles: 5,
            ..Default::default()
        });
        emitter.tick(1.0);
        let splats = emitter.splats();
        assert_eq!(splats.len(), emitter.live_count());
    }

    #[test]
    fn died_this_frame_has_correct_spectral() {
        let spectral = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut emitter = SplatEmitter::new(EmitterConfig {
            emit_rate: 100.0,
            max_particles: 10,
            min_lifetime: 0.01,
            max_lifetime: 0.01,
            spectral,
            ..Default::default()
        });
        emitter.tick(0.005);
        emitter.tick(0.02); // kill all
        assert!(!emitter.died_this_frame.is_empty(), "particles should have died");
        for dead_spectral in &emitter.died_this_frame {
            assert_eq!(*dead_spectral, spectral, "spectral should match emitter config");
        }
    }
}
```

- [ ] Add `pub mod splat_particles;` to `crates/vox_render/src/lib.rs`

- [ ] Run:
```bash
cargo test -p vox_render splat_particles
```
Expected: 6 tests pass.

- [ ] Commit:
```bash
git commit -m "feat(render): SplatEmitter — Gaussian splat particles with spectral emission"
```

---

## Task 2: Wire into engine_runner — emit particles, play impact audio

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Add `particle_emitters: Vec<vox_render::splat_particles::SplatEmitter>` field to `EngineApp` (all construction sites; initialize as `Vec::new()`).

- [ ] In the per-frame update loop, tick all emitters, collect splats, inject into scene, play audio for deaths:

```rust
// Tick particle emitters
let mut particle_splats: Vec<vox_core::types::GaussianSplat> = Vec::new();
for emitter in &mut self.particle_emitters {
    emitter.tick(self.frame_dt);
    particle_splats.extend(emitter.splats());

    // Play impact audio for each particle that died this frame
    for dead_spectral in &emitter.died_this_frame {
        let wav_path = vox_audio::create_impact_wav(dead_spectral, 0.08);
        if let Some(audio) = &self.audio_handle {
            audio.play(wav_path.to_str().unwrap_or(""), 0.4, false);
        }
    }
}

// Inject particles into render splat list (after scene_splats, before render call)
let mut render_splats = self.scene_splats.clone();
render_splats.extend(particle_splats);
```

- [ ] Wire `KeyE` to spawn a fire emitter at camera position:

```rust
if self.input_state.just_pressed(vox_core::input::Key::KeyE) {
    use vox_render::splat_particles::{SplatEmitter, EmitterConfig};
    let pos = self.camera.position().to_array();
    self.particle_emitters.push(SplatEmitter::new(EmitterConfig::fire(pos)));
    println!("[ochroma] Spawned fire emitter at {:?}", pos);
}
```

- [ ] Verify compile:
```bash
cargo check --bin ochroma
```

- [ ] Commit:
```bash
git commit -m "feat(app): wire SplatEmitter into engine_runner with KeyE spawn + impact audio"
```

---

## Acceptance Criteria

| # | Test | Command |
|---|------|---------|
| 1 | Emitter spawns, ages, kills particles | `cargo test -p vox_render splat_particles` |
| 2 | Engine compiles with emitters wired | `cargo check --bin ochroma` |
| 3 | Full workspace green | `cargo test` |
