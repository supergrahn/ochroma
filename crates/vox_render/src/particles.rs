use glam::Vec3;
use half::f16;
use vox_core::types::GaussianSplat;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub opacity: f32,
    pub scale: f32,
    pub spectral: [u16; 8],
}

#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    pub position: Vec3,
    pub emission_rate: f32, // particles per second
    pub particle_lifetime: f32,
    pub initial_velocity: Vec3,
    pub velocity_randomness: f32,
    pub particle_scale: f32,
    pub spectral: [u16; 8], // spectral appearance
    pub gravity: Vec3,
    pub accumulator: f32,
}

impl ParticleEmitter {
    pub fn smoke(position: Vec3) -> Self {
        let grey_spd = std::array::from_fn(|_| f16::from_f32(0.3).to_bits());
        Self {
            position,
            emission_rate: 5.0,
            particle_lifetime: 3.0,
            initial_velocity: Vec3::new(0.0, 2.0, 0.0),
            velocity_randomness: 0.5,
            particle_scale: 0.3,
            spectral: grey_spd,
            gravity: Vec3::new(0.0, 0.5, 0.0), // smoke rises
            accumulator: 0.0,
        }
    }

    pub fn dust(position: Vec3) -> Self {
        let brown_spd: [u16; 8] = [
            f16::from_f32(0.10).to_bits(), f16::from_f32(0.12).to_bits(),
            f16::from_f32(0.15).to_bits(), f16::from_f32(0.20).to_bits(),
            f16::from_f32(0.22).to_bits(), f16::from_f32(0.20).to_bits(),
            f16::from_f32(0.18).to_bits(), f16::from_f32(0.15).to_bits(),
        ];
        Self {
            position,
            emission_rate: 20.0,
            particle_lifetime: 1.5,
            initial_velocity: Vec3::new(0.0, 1.0, 0.0),
            velocity_randomness: 2.0,
            particle_scale: 0.15,
            spectral: brown_spd,
            gravity: Vec3::new(0.0, -3.0, 0.0),
            accumulator: 0.0,
        }
    }
}

pub struct ParticleSystem {
    pub emitters: Vec<ParticleEmitter>,
    pub particles: Vec<Particle>,
    pub max_particles: usize,
    rng_state: u64,
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            emitters: Vec::new(),
            particles: Vec::new(),
            max_particles,
            rng_state: 42,
        }
    }

    #[allow(dead_code)]
    fn next_random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 33) as f32 / (1u64 << 31) as f32
    }

    pub fn add_emitter(&mut self, emitter: ParticleEmitter) {
        self.emitters.push(emitter);
    }

    pub fn tick(&mut self, dt: f32) {
        // Update existing particles
        self.particles.retain_mut(|p| {
            p.lifetime -= dt;
            if p.lifetime <= 0.0 { return false; }
            p.velocity += Vec3::new(0.0, -9.81, 0.0) * dt; // gravity
            p.position += p.velocity * dt;
            p.opacity = (p.lifetime / p.max_lifetime).clamp(0.0, 1.0);
            true
        });

        // Emit new particles
        for emitter in &mut self.emitters {
            emitter.accumulator += emitter.emission_rate * dt;
            while emitter.accumulator >= 1.0 && self.particles.len() < self.max_particles {
                emitter.accumulator -= 1.0;
                let rand_x = (self.rng_state as f32 / u64::MAX as f32 - 0.5) * emitter.velocity_randomness;
                self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_z = (self.rng_state as f32 / u64::MAX as f32 - 0.5) * emitter.velocity_randomness;
                self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                self.particles.push(Particle {
                    position: emitter.position,
                    velocity: emitter.initial_velocity + Vec3::new(rand_x, 0.0, rand_z),
                    lifetime: emitter.particle_lifetime,
                    max_lifetime: emitter.particle_lifetime,
                    opacity: 1.0,
                    scale: emitter.particle_scale,
                    spectral: emitter.spectral,
                });
            }
        }
    }

    /// Convert active particles to GaussianSplats for rendering.
    pub fn to_splats(&self) -> Vec<GaussianSplat> {
        self.particles.iter().map(|p| {
            GaussianSplat {
                position: [p.position.x, p.position.y, p.position.z],
                scale: [p.scale, p.scale, p.scale],
                rotation: [0, 0, 0, 32767],
                opacity: (p.opacity * 200.0) as u8,
                _pad: [0; 3],
                spectral: p.spectral,
            }
        }).collect()
    }

    pub fn particle_count(&self) -> usize { self.particles.len() }
}
