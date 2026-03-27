//! Data-driven VFX system — Ochroma's answer to Niagara.
//!
//! Effects are defined declaratively via [`VfxEffect`] (serialisable to/from
//! JSON/TOML), instantiated at runtime as [`VfxInstance`], ticked each frame,
//! and converted to [`GaussianSplat`]s for spectral rendering.

use glam::{Quat, Vec3};
use half::f16;
use serde::{Deserialize, Serialize};
use vox_core::types::GaussianSplat;

// ---------------------------------------------------------------------------
// Data types (serialisable effect definitions)
// ---------------------------------------------------------------------------

/// A VFX effect definition — loaded from config, not hardcoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfxEffect {
    pub name: String,
    pub emitters: Vec<VfxEmitter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfxEmitter {
    pub shape: EmitterShape,
    pub rate: f32,
    pub burst: Option<u32>,
    pub lifetime: RangeF32,
    pub velocity: VelocityConfig,
    pub size: CurveF32,
    pub opacity: CurveF32,
    pub color: ColorConfig,
    pub gravity_scale: f32,
    pub max_particles: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmitterShape {
    Point,
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    Cone { angle: f32, radius: f32 },
    Ring { radius: f32, width: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityConfig {
    pub direction: [f32; 3],
    pub speed: RangeF32,
    pub randomness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeF32 {
    pub min: f32,
    pub max: f32,
}

impl RangeF32 {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    pub fn sample(&self, t: f32) -> f32 {
        self.min + (self.max - self.min) * t
    }
}

/// A curve defined by keypoints (for size/opacity over lifetime).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveF32 {
    pub keys: Vec<(f32, f32)>,
}

impl CurveF32 {
    pub fn constant(value: f32) -> Self {
        Self {
            keys: vec![(0.0, value), (1.0, value)],
        }
    }

    pub fn fade_out(start: f32) -> Self {
        Self {
            keys: vec![(0.0, start), (1.0, 0.0)],
        }
    }

    pub fn fade_in_out(peak: f32) -> Self {
        Self {
            keys: vec![(0.0, 0.0), (0.2, peak), (0.8, peak), (1.0, 0.0)],
        }
    }

    pub fn evaluate(&self, t: f32) -> f32 {
        if self.keys.is_empty() {
            return 0.0;
        }
        if t <= self.keys[0].0 {
            return self.keys[0].1;
        }
        if t >= self.keys.last().unwrap().0 {
            return self.keys.last().unwrap().1;
        }

        for window in self.keys.windows(2) {
            if t >= window[0].0 && t <= window[1].0 {
                let frac = (t - window[0].0) / (window[1].0 - window[0].0);
                return window[0].1 + (window[1].1 - window[0].1) * frac;
            }
        }
        self.keys.last().unwrap().1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorConfig {
    pub start_spectral: [f32; 8],
    pub end_spectral: [f32; 8],
}

impl ColorConfig {
    /// Linearly interpolate between start and end spectral values.
    pub fn evaluate(&self, t: f32) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = self.start_spectral[i] + (self.end_spectral[i] - self.start_spectral[i]) * t;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

/// Runtime state for a VFX effect instance.
pub struct VfxInstance {
    pub effect: VfxEffect,
    pub position: Vec3,
    pub rotation: Quat,
    pub active: bool,
    pub time: f32,
    emitter_states: Vec<EmitterState>,
}

struct EmitterState {
    particles: Vec<VfxParticle>,
    accumulator: f32,
    rng_state: u64,
    burst_fired: bool,
}

struct VfxParticle {
    position: Vec3,
    velocity: Vec3,
    age: f32,
    lifetime: f32,
    #[allow(dead_code)]
    size_start: f32,
}

impl VfxInstance {
    pub fn new(effect: VfxEffect, position: Vec3) -> Self {
        let emitter_states = effect
            .emitters
            .iter()
            .enumerate()
            .map(|(i, _)| EmitterState {
                particles: Vec::new(),
                accumulator: 0.0,
                rng_state: 42u64.wrapping_add(i as u64 * 7919),
                burst_fired: false,
            })
            .collect();

        Self {
            effect,
            position,
            rotation: Quat::IDENTITY,
            active: true,
            time: 0.0,
            emitter_states,
        }
    }

    pub fn tick(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.time += dt;

        let gravity = Vec3::new(0.0, -9.81, 0.0);

        for (emitter_idx, emitter) in self.effect.emitters.iter().enumerate() {
            let state = &mut self.emitter_states[emitter_idx];

            // --- Update existing particles ---
            state.particles.retain_mut(|p| {
                p.age += dt;
                if p.age >= p.lifetime {
                    return false;
                }
                p.velocity += gravity * emitter.gravity_scale * dt;
                p.position += p.velocity * dt;
                true
            });

            // --- Burst spawn (once) ---
            if !state.burst_fired {
                if let Some(count) = emitter.burst {
                    for _ in 0..count {
                        if state.particles.len() >= emitter.max_particles {
                            break;
                        }
                        let p = spawn_particle(emitter, &mut state.rng_state, self.position);
                        state.particles.push(p);
                    }
                }
                state.burst_fired = true;
            }

            // --- Rate-based emission ---
            state.accumulator += emitter.rate * dt;
            while state.accumulator >= 1.0
                && state.particles.len() < emitter.max_particles
            {
                state.accumulator -= 1.0;
                let p = spawn_particle(emitter, &mut state.rng_state, self.position);
                state.particles.push(p);
            }
        }
    }

    /// Convert live particles to GaussianSplats for rendering.
    pub fn to_splats(&self) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        for (emitter_idx, emitter) in self.effect.emitters.iter().enumerate() {
            let state = &self.emitter_states[emitter_idx];
            for p in &state.particles {
                let t = (p.age / p.lifetime).clamp(0.0, 1.0);
                let size = emitter.size.evaluate(t);
                let opacity = emitter.opacity.evaluate(t);
                let spectral_f32 = emitter.color.evaluate(t);
                let spectral: [u16; 8] =
                    std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits());

                splats.push(GaussianSplat {
                    position: [p.position.x, p.position.y, p.position.z],
                    scale: [size, size, size],
                    rotation: [0, 0, 0, 32767],
                    opacity: (opacity * 255.0).clamp(0.0, 255.0) as u8,
                    _pad: [0; 3],
                    spectral,
                });
            }
        }
        splats
    }

    pub fn particle_count(&self) -> usize {
        self.emitter_states.iter().map(|s| s.particles.len()).sum()
    }

    /// Returns true when all emitters have exhausted their burst and all
    /// particles have died.
    pub fn is_finished(&self) -> bool {
        if !self.active {
            return true;
        }
        for (i, emitter) in self.effect.emitters.iter().enumerate() {
            let state = &self.emitter_states[i];
            // If the emitter still has a continuous rate, it is never "finished"
            // unless deactivated.
            if emitter.rate > 0.0 {
                return false;
            }
            if !state.particles.is_empty() {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Spawn helpers
// ---------------------------------------------------------------------------

fn next_random(rng: &mut u64) -> f32 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*rng >> 33) as f32 / (1u64 << 31) as f32
}

fn spawn_particle(emitter: &VfxEmitter, rng: &mut u64, origin: Vec3) -> VfxParticle {
    let t = next_random(rng);
    let lifetime = emitter.lifetime.sample(t);
    let speed_t = next_random(rng);
    let speed = emitter.velocity.speed.sample(speed_t);

    // Base direction
    let dir = Vec3::new(
        emitter.velocity.direction[0],
        emitter.velocity.direction[1],
        emitter.velocity.direction[2],
    )
    .normalize_or_zero();

    // Random offset based on randomness factor
    let rx = (next_random(rng) - 0.5) * 2.0 * emitter.velocity.randomness;
    let ry = (next_random(rng) - 0.5) * 2.0 * emitter.velocity.randomness;
    let rz = (next_random(rng) - 0.5) * 2.0 * emitter.velocity.randomness;
    let random_dir = Vec3::new(rx, ry, rz);

    let final_dir = (dir + random_dir).normalize_or_zero();
    let velocity = final_dir * speed;

    // Spawn position based on emitter shape
    let offset = match &emitter.shape {
        EmitterShape::Point => Vec3::ZERO,
        EmitterShape::Sphere { radius } => {
            let x = (next_random(rng) - 0.5) * 2.0;
            let y = (next_random(rng) - 0.5) * 2.0;
            let z = (next_random(rng) - 0.5) * 2.0;
            let v = Vec3::new(x, y, z).normalize_or_zero();
            v * next_random(rng) * radius
        }
        EmitterShape::Box { half_extents } => {
            let x = (next_random(rng) - 0.5) * 2.0 * half_extents[0];
            let y = (next_random(rng) - 0.5) * 2.0 * half_extents[1];
            let z = (next_random(rng) - 0.5) * 2.0 * half_extents[2];
            Vec3::new(x, y, z)
        }
        EmitterShape::Cone { angle, radius } => {
            let a = next_random(rng) * std::f32::consts::TAU;
            let r = next_random(rng) * radius;
            let spread = angle.to_radians().sin() * r;
            Vec3::new(a.cos() * spread, 0.0, a.sin() * spread)
        }
        EmitterShape::Ring {
            radius,
            width,
        } => {
            let a = next_random(rng) * std::f32::consts::TAU;
            let r = radius + (next_random(rng) - 0.5) * width;
            Vec3::new(a.cos() * r, 0.0, a.sin() * r)
        }
    };

    let size_start = emitter.size.evaluate(0.0);

    VfxParticle {
        position: origin + offset,
        velocity,
        age: 0.0,
        lifetime,
        size_start,
    }
}

// ---------------------------------------------------------------------------
// Pre-built VFX effects
// ---------------------------------------------------------------------------

/// Fire effect — upward flickering particles with warm spectral tones.
pub fn effect_fire() -> VfxEffect {
    VfxEffect {
        name: "fire".into(),
        emitters: vec![VfxEmitter {
            shape: EmitterShape::Cone {
                angle: 15.0,
                radius: 0.3,
            },
            rate: 40.0,
            burst: None,
            lifetime: RangeF32::new(0.5, 1.5),
            velocity: VelocityConfig {
                direction: [0.0, 1.0, 0.0],
                speed: RangeF32::new(2.0, 4.0),
                randomness: 0.3,
            },
            size: CurveF32 {
                keys: vec![(0.0, 0.15), (0.3, 0.25), (1.0, 0.05)],
            },
            opacity: CurveF32::fade_out(1.0),
            color: ColorConfig {
                start_spectral: [0.05, 0.10, 0.20, 0.50, 0.80, 0.95, 0.90, 0.60],
                end_spectral: [0.02, 0.05, 0.10, 0.25, 0.40, 0.30, 0.20, 0.10],
            },
            gravity_scale: -0.3,
            max_particles: 500,
        }],
    }
}

/// Smoke effect — slow rising, expanding, fading particles.
pub fn effect_smoke() -> VfxEffect {
    VfxEffect {
        name: "smoke".into(),
        emitters: vec![VfxEmitter {
            shape: EmitterShape::Sphere { radius: 0.2 },
            rate: 8.0,
            burst: None,
            lifetime: RangeF32::new(2.0, 4.0),
            velocity: VelocityConfig {
                direction: [0.0, 1.0, 0.0],
                speed: RangeF32::new(0.5, 1.5),
                randomness: 0.4,
            },
            size: CurveF32 {
                keys: vec![(0.0, 0.2), (0.5, 0.5), (1.0, 0.8)],
            },
            opacity: CurveF32 {
                keys: vec![(0.0, 0.6), (0.3, 0.5), (1.0, 0.0)],
            },
            color: ColorConfig {
                start_spectral: [0.30; 8],
                end_spectral: [0.15; 8],
            },
            gravity_scale: -0.2,
            max_particles: 300,
        }],
    }
}

/// Explosion — large burst, outward spherical expansion.
pub fn effect_explosion() -> VfxEffect {
    VfxEffect {
        name: "explosion".into(),
        emitters: vec![
            // Core flash
            VfxEmitter {
                shape: EmitterShape::Point,
                rate: 0.0,
                burst: Some(50),
                lifetime: RangeF32::new(0.3, 0.8),
                velocity: VelocityConfig {
                    direction: [0.0, 1.0, 0.0],
                    speed: RangeF32::new(5.0, 12.0),
                    randomness: 1.0,
                },
                size: CurveF32 {
                    keys: vec![(0.0, 0.4), (0.2, 0.6), (1.0, 0.1)],
                },
                opacity: CurveF32::fade_out(1.0),
                color: ColorConfig {
                    start_spectral: [0.10, 0.20, 0.40, 0.70, 0.95, 1.00, 0.95, 0.70],
                    end_spectral: [0.05, 0.08, 0.15, 0.30, 0.45, 0.35, 0.25, 0.12],
                },
                gravity_scale: 0.3,
                max_particles: 200,
            },
            // Debris
            VfxEmitter {
                shape: EmitterShape::Sphere { radius: 0.5 },
                rate: 0.0,
                burst: Some(30),
                lifetime: RangeF32::new(0.8, 2.0),
                velocity: VelocityConfig {
                    direction: [0.0, 0.5, 0.0],
                    speed: RangeF32::new(3.0, 8.0),
                    randomness: 0.9,
                },
                size: CurveF32::constant(0.1),
                opacity: CurveF32::fade_out(0.8),
                color: ColorConfig {
                    start_spectral: [0.15, 0.12, 0.10, 0.20, 0.25, 0.22, 0.18, 0.12],
                    end_spectral: [0.08, 0.06, 0.05, 0.10, 0.12, 0.10, 0.08, 0.06],
                },
                gravity_scale: 1.0,
                max_particles: 200,
            },
        ],
    }
}

/// Sparkle — small bright flashes.
pub fn effect_sparkle() -> VfxEffect {
    VfxEffect {
        name: "sparkle".into(),
        emitters: vec![VfxEmitter {
            shape: EmitterShape::Sphere { radius: 0.5 },
            rate: 15.0,
            burst: None,
            lifetime: RangeF32::new(0.2, 0.6),
            velocity: VelocityConfig {
                direction: [0.0, 0.5, 0.0],
                speed: RangeF32::new(0.5, 2.0),
                randomness: 0.8,
            },
            size: CurveF32::fade_out(0.08),
            opacity: CurveF32::fade_in_out(1.0),
            color: ColorConfig {
                start_spectral: [0.80, 0.85, 0.90, 0.95, 1.00, 1.00, 0.95, 0.90],
                end_spectral: [0.60, 0.65, 0.70, 0.75, 0.80, 0.80, 0.75, 0.70],
            },
            gravity_scale: -0.1,
            max_particles: 200,
        }],
    }
}

/// Rain — downward streaks.
pub fn effect_rain() -> VfxEffect {
    VfxEffect {
        name: "rain".into(),
        emitters: vec![VfxEmitter {
            shape: EmitterShape::Box {
                half_extents: [10.0, 0.0, 10.0],
            },
            rate: 200.0,
            burst: None,
            lifetime: RangeF32::new(0.8, 1.2),
            velocity: VelocityConfig {
                direction: [0.0, -1.0, 0.0],
                speed: RangeF32::new(8.0, 12.0),
                randomness: 0.05,
            },
            size: CurveF32::constant(0.02),
            opacity: CurveF32::constant(0.5),
            color: ColorConfig {
                start_spectral: [0.30, 0.35, 0.40, 0.45, 0.50, 0.48, 0.42, 0.35],
                end_spectral: [0.30, 0.35, 0.40, 0.45, 0.50, 0.48, 0.42, 0.35],
            },
            gravity_scale: 0.5,
            max_particles: 2000,
        }],
    }
}

/// Dust — light particles drifting near ground.
pub fn effect_dust() -> VfxEffect {
    VfxEffect {
        name: "dust".into(),
        emitters: vec![VfxEmitter {
            shape: EmitterShape::Box {
                half_extents: [2.0, 0.3, 2.0],
            },
            rate: 12.0,
            burst: None,
            lifetime: RangeF32::new(1.5, 3.0),
            velocity: VelocityConfig {
                direction: [0.2, 0.3, 0.1],
                speed: RangeF32::new(0.2, 0.8),
                randomness: 0.6,
            },
            size: CurveF32 {
                keys: vec![(0.0, 0.03), (0.5, 0.06), (1.0, 0.04)],
            },
            opacity: CurveF32::fade_in_out(0.4),
            color: ColorConfig {
                start_spectral: [0.10, 0.12, 0.15, 0.20, 0.22, 0.20, 0.18, 0.15],
                end_spectral: [0.08, 0.10, 0.12, 0.16, 0.18, 0.16, 0.14, 0.12],
            },
            gravity_scale: -0.05,
            max_particles: 300,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_constant_evaluates() {
        let c = CurveF32::constant(5.0);
        assert!((c.evaluate(0.0) - 5.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 5.0).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn curve_fade_out_evaluates() {
        let c = CurveF32::fade_out(1.0);
        assert!((c.evaluate(0.0) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn range_sample() {
        let r = RangeF32::new(2.0, 6.0);
        assert!((r.sample(0.0) - 2.0).abs() < 1e-6);
        assert!((r.sample(0.5) - 4.0).abs() < 1e-6);
        assert!((r.sample(1.0) - 6.0).abs() < 1e-6);
    }
}
