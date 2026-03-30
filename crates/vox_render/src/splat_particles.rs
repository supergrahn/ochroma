//! Gaussian splat particle system.
//!
//! Each particle is a `GaussianSplat` rendered through the standard EWA pipeline.
//! Particles are Gaussian splats — no separate render pass, no z-fighting.
//!
//! Spectral emission drives audio: on particle death, the caller can call
//! `vox_audio::synthesize_impact` with `particle.spectral` to get impact audio.

use glam;
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
    /// 16-band spectral profile for this emitter. Values in [0, 1].
    /// High blue (band 0) = glassy/electric, high red (band 15) = fire/rock.
    pub spectral: [f32; 16],
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
            spectral: [0.0, 0.0, 0.0, 0.0, 0.5, 0.8, 0.9, 0.3, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02], // fire-ish
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
    pub spectral: [f32; 16],
    pub base_opacity: u8,
    pub scale: [f32; 3],
}

impl SplatParticle {
    /// Convert to `GaussianSplat` with opacity modulated by remaining lifetime.
    pub fn to_splat(&self) -> GaussianSplat {
        let t = (self.remaining / self.lifetime).clamp(0.0, 1.0);
        let opacity = (self.base_opacity as f32 * t) as u8;
        let spectral: [u16; 16] = std::array::from_fn(|i| {
            f16::from_f32(self.spectral[i].clamp(0.0, 1.0)).to_bits()
        });
        GaussianSplat::volume(
            self.position,
            self.scale,
            glam::Quat::IDENTITY,
            opacity,
            spectral,
        )
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
    pub died_this_frame: Vec<[f32; 16]>,
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
        let mut newly_dead: Vec<[f32; 16]> = Vec::new();
        self.particles.retain_mut(|p| {
            p.remaining -= dt;
            if p.remaining <= 0.0 {
                newly_dead.push(p.spectral);
                return false;
            }
            p.velocity[1] += self.config.gravity * dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
            true
        });
        self.died_this_frame.extend(newly_dead);

        // Emit new particles
        self.emit_accum += self.config.emit_rate * dt;
        while self.emit_accum >= 1.0 && self.particles.len() < self.config.max_particles {
            self.emit_accum -= 1.0;
            let particle = self.spawn_particle();
            self.particles.push(particle);
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
            spectral: [0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.02],
            emit_rate: 40.0,
            max_particles: 300,
            gravity: -2.0,
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
            spectral: [1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            spectral: [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
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
            max_lifetime: 0.1,
            ..Default::default()
        });
        emitter.tick(0.05);
        let count_before = emitter.live_count();
        assert!(count_before > 0, "should have spawned particles");
        emitter.tick(0.2);
        assert!(emitter.died_this_frame.len() > 0, "particles should have died");
        assert!(emitter.died_this_frame.len() <= count_before);
    }

    #[test]
    fn to_splat_opacity_decreases_with_age() {
        let p = SplatParticle {
            position: [0.0; 3],
            velocity: [0.0; 3],
            remaining: 0.5,
            lifetime: 1.0,
            spectral: [0.5; 16],
            base_opacity: 200,
            scale: [0.1; 3],
        };
        let splat = p.to_splat();
        assert!(splat.opacity() < 200, "opacity should decrease with age");
        assert!(splat.opacity() >= 95, "at half lifetime, opacity should be ~100");
    }

    #[test]
    fn to_splat_opacity_is_zero_at_death() {
        let p = SplatParticle {
            position: [0.0; 3],
            velocity: [0.0; 3],
            remaining: 0.0,
            lifetime: 1.0,
            spectral: [0.5; 16],
            base_opacity: 200,
            scale: [0.1; 3],
        };
        assert_eq!(p.to_splat().opacity(), 0);
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
        let spectral = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0f32];
        let mut emitter = SplatEmitter::new(EmitterConfig {
            emit_rate: 100.0,
            max_particles: 10,
            min_lifetime: 0.01,
            max_lifetime: 0.01,
            spectral,
            ..Default::default()
        });
        emitter.tick(0.02);  // spawn particles with remaining = 0.01
        emitter.tick(0.02);  // integrate: remaining = -0.01, particles die
        assert!(!emitter.died_this_frame.is_empty(), "particles should have died");
        for dead_spectral in &emitter.died_this_frame {
            assert_eq!(*dead_spectral, spectral, "spectral should match emitter config");
        }
    }
}
