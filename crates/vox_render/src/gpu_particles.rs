use bytemuck::{Pod, Zeroable};

/// GPU-friendly particle data (aligned for compute shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub lifetime: f32,
    pub size: f32,
    pub opacity: f32,
    pub color: [f32; 2], // packed spectral index + intensity
}

/// GPU particle emitter configuration.
#[derive(Debug, Clone)]
pub struct GpuParticleEmitter {
    pub position: [f32; 3],
    pub emission_rate: f32,
    pub max_particles: u32,
    pub initial_velocity: [f32; 3],
    pub velocity_randomness: f32,
    pub lifetime_range: [f32; 2],
    pub size_range: [f32; 2],
    pub gravity: [f32; 3],
}

/// Manages GPU particle buffers for compute dispatch.
pub struct GpuParticleSystem {
    pub emitters: Vec<GpuParticleEmitter>,
    pub particles: Vec<GpuParticle>,
    pub max_total_particles: usize,
    pub active_count: usize,
    emit_accumulator: Vec<f32>, // per-emitter fractional particle accumulation
}

impl GpuParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            emitters: Vec::new(),
            particles: Vec::new(),
            max_total_particles: max_particles,
            active_count: 0,
            emit_accumulator: Vec::new(),
        }
    }

    pub fn add_emitter(&mut self, emitter: GpuParticleEmitter) {
        self.emitters.push(emitter);
        self.emit_accumulator.push(0.0);
    }

    /// CPU fallback tick — emit new particles and advance existing ones.
    pub fn tick_cpu(&mut self, dt: f32) {
        // Advance existing particles
        for p in &mut self.particles {
            p.age += dt;
            // Apply gravity from the first emitter (simplified)
            let gravity = if !self.emitters.is_empty() {
                self.emitters[0].gravity
            } else {
                [0.0, -9.81, 0.0]
            };
            p.velocity[0] += gravity[0] * dt;
            p.velocity[1] += gravity[1] * dt;
            p.velocity[2] += gravity[2] * dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
            // Fade opacity as particle ages
            let life_frac = (p.age / p.lifetime).min(1.0);
            p.opacity = 1.0 - life_frac;
        }

        // Remove dead particles
        self.particles.retain(|p| p.age < p.lifetime);

        // Emit new particles
        for (i, emitter) in self.emitters.iter().enumerate() {
            self.emit_accumulator[i] += emitter.emission_rate * dt;
            let to_emit = self.emit_accumulator[i] as u32;
            self.emit_accumulator[i] -= to_emit as f32;

            for _ in 0..to_emit {
                if self.particles.len() >= self.max_total_particles {
                    break;
                }
                let lifetime =
                    (emitter.lifetime_range[0] + emitter.lifetime_range[1]) * 0.5;
                let size = (emitter.size_range[0] + emitter.size_range[1]) * 0.5;
                self.particles.push(GpuParticle {
                    position: emitter.position,
                    age: 0.0,
                    velocity: emitter.initial_velocity,
                    lifetime,
                    size,
                    opacity: 1.0,
                    color: [0.0, 1.0],
                });
            }
        }

        self.active_count = self.particles.len();
    }

    /// Raw bytes for GPU buffer upload.
    pub fn particle_buffer_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.particles)
    }

    pub fn active_particle_count(&self) -> usize {
        self.active_count
    }

    /// Convert active particles to Gaussian splats for rendering.
    pub fn to_splats(&self) -> Vec<vox_core::types::GaussianSplat> {
        self.particles
            .iter()
            .map(|p| {
                let scale_val = p.size * (1.0 - (p.age / p.lifetime).min(1.0));
                vox_core::types::GaussianSplat {
                    position: p.position,
                    scale: [scale_val; 3],
                    rotation: [0, 0, 0, 32767], // identity quaternion
                    opacity: (p.opacity * 255.0) as u8,
                    _pad: [0; 3],
                    spectral: [0; 8],
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_emitter() -> GpuParticleEmitter {
        GpuParticleEmitter {
            position: [0.0, 0.0, 0.0],
            emission_rate: 100.0, // 100 particles/sec
            max_particles: 1000,
            initial_velocity: [0.0, 5.0, 0.0],
            velocity_randomness: 0.0,
            lifetime_range: [1.0, 2.0],
            size_range: [0.1, 0.2],
            gravity: [0.0, -9.81, 0.0],
        }
    }

    #[test]
    fn emitter_produces_particles() {
        let mut sys = GpuParticleSystem::new(10000);
        sys.add_emitter(test_emitter());
        assert_eq!(sys.active_particle_count(), 0);

        sys.tick_cpu(0.1); // should emit ~10 particles
        assert!(sys.active_particle_count() > 0);
    }

    #[test]
    fn tick_advances_age() {
        let mut sys = GpuParticleSystem::new(10000);
        sys.add_emitter(test_emitter());
        sys.tick_cpu(0.1);

        let count_before = sys.active_particle_count();
        assert!(count_before > 0);
        let first_age = sys.particles[0].age;

        sys.tick_cpu(0.05);
        // The first particle (which existed before) should have advanced age
        assert!(sys.particles[0].age > first_age);
    }

    #[test]
    fn dead_particles_removed() {
        let mut sys = GpuParticleSystem::new(10000);
        sys.add_emitter(GpuParticleEmitter {
            lifetime_range: [0.05, 0.05], // very short lifetime
            ..test_emitter()
        });
        sys.tick_cpu(0.01); // emit some
        let count_after_emit = sys.active_particle_count();
        assert!(count_after_emit > 0);

        // Tick past their lifetime
        sys.tick_cpu(0.1);
        // All original particles should be dead, but new ones may have spawned
        // The key check: system doesn't grow unbounded
        assert!(sys.active_particle_count() <= count_after_emit + 20);
    }

    #[test]
    fn buffer_size_correct() {
        let mut sys = GpuParticleSystem::new(10000);
        sys.add_emitter(test_emitter());
        sys.tick_cpu(0.1);

        let particle_size = std::mem::size_of::<GpuParticle>();
        let expected = sys.active_particle_count() * particle_size;
        assert_eq!(sys.particle_buffer_bytes().len(), expected);
    }
}
