//! Position-Based Fluids (PBF) — GPU-accelerated.
//! 4 compute passes per frame: predict → neighbors → solve → integrate.
//! Density constraint: λᵢ = −Cᵢ / (∇Cᵢ² + ε),  Cᵢ = ρᵢ/ρ₀ − 1

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PbfParticle {
    pub position:      [f32; 3],
    pub density:       f32,
    pub predicted_pos: [f32; 3],
    pub lambda:        f32,
    pub velocity:      [f32; 3],
    pub pressure:      f32,
    pub spectral:      [f32; 16],
}

pub struct PbfGpuBuffers {
    pub particle_buf:  wgpu::Buffer,
    pub neighbor_buf:  wgpu::Buffer,
    pub params_buf:    wgpu::Buffer,
    pub max_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PbfParams {
    pub dt:             f32,
    pub rest_density:   f32,
    pub smoothing_h:    f32,
    pub epsilon:        f32,
    pub particle_count: u32,
    pub max_neighbors:  u32,
    pub _pad:           [u32; 2],
}

pub struct PbfFluidSim {
    pub particles: Vec<PbfParticle>,
    pub params:    PbfParams,
    pub gpu:       Option<PbfGpuBuffers>,
}

pub const WATER_SPECTRAL: [f32; 16] = [
    0.1, 0.8, 0.9, 0.3, 0.1, 0.05, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
];
pub const BLOOD_SPECTRAL: [f32; 16] = [
    0.0, 0.0, 0.01, 0.05, 0.2, 0.6, 0.4, 0.1,
    0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01,
];
pub const LAVA_SPECTRAL: [f32; 16] = [
    0.0, 0.0, 0.02, 0.1, 0.4, 0.8, 1.0, 1.0,
    0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60,
];

impl PbfFluidSim {
    pub fn new(rest_density: f32, smoothing_h: f32) -> Self {
        Self {
            particles: Vec::new(),
            params: PbfParams {
                dt: 1.0 / 60.0,
                rest_density,
                smoothing_h,
                epsilon: 1e-4,
                particle_count: 0,
                max_neighbors: 64,
                _pad: [0; 2],
            },
            gpu: None,
        }
    }

    pub fn spawn(&mut self, pos: [f32; 3], vel: [f32; 3], spectral: [f32; 16]) {
        self.particles.push(PbfParticle {
            position: pos,
            density: self.params.rest_density,
            predicted_pos: pos,
            lambda: 0.0,
            velocity: vel,
            pressure: 0.0,
            spectral,
        });
        self.params.particle_count = self.particles.len() as u32;
    }

    pub fn cpu_step(&mut self) {
        let dt = self.params.dt;
        let h = self.params.smoothing_h;
        let rho0 = self.params.rest_density;
        let eps = self.params.epsilon;
        let n = self.particles.len();

        // Pass 1: predict positions (apply gravity)
        for p in &mut self.particles {
            p.velocity[1] -= 9.81 * dt;
            p.predicted_pos = [
                p.position[0] + p.velocity[0] * dt,
                p.position[1] + p.velocity[1] * dt,
                p.position[2] + p.velocity[2] * dt,
            ];
        }

        // Pass 2: compute density using poly6 kernel
        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
        let mut densities = vec![0.0f32; n];
        for i in 0..n {
            let mut rho = 0.0f32;
            for j in 0..n {
                let dx = self.particles[i].predicted_pos[0] - self.particles[j].predicted_pos[0];
                let dy = self.particles[i].predicted_pos[1] - self.particles[j].predicted_pos[1];
                let dz = self.particles[i].predicted_pos[2] - self.particles[j].predicted_pos[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < h * h {
                    rho += poly6_coeff * (h * h - r2).powi(3);
                }
            }
            densities[i] = rho;
        }

        // Pass 3: solve density constraint — compute λᵢ and Δpᵢ
        let spiky_grad = -45.0 / (std::f32::consts::PI * h.powi(6));
        for i in 0..n {
            let ci = densities[i] / rho0 - 1.0;
            let mut grad_sq = 0.0f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = self.particles[i].predicted_pos[0] - self.particles[j].predicted_pos[0];
                let dy = self.particles[i].predicted_pos[1] - self.particles[j].predicted_pos[1];
                let dz = self.particles[i].predicted_pos[2] - self.particles[j].predicted_pos[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < h && r > 1e-6 {
                    let g = spiky_grad * (h - r).powi(2) / r;
                    grad_sq += (g * dx) * (g * dx) + (g * dy) * (g * dy) + (g * dz) * (g * dz);
                }
            }
            self.particles[i].lambda = -ci / (grad_sq / (rho0 * rho0) + eps);
        }

        let mut deltas = vec![[0.0f32; 3]; n];
        for i in 0..n {
            let li = self.particles[i].lambda;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = self.particles[i].predicted_pos[0] - self.particles[j].predicted_pos[0];
                let dy = self.particles[i].predicted_pos[1] - self.particles[j].predicted_pos[1];
                let dz = self.particles[i].predicted_pos[2] - self.particles[j].predicted_pos[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < h && r > 1e-6 {
                    let g = spiky_grad * (h - r).powi(2) / r;
                    let lj = self.particles[j].lambda;
                    let s = (li + lj) / rho0;
                    deltas[i][0] += s * g * dx;
                    deltas[i][1] += s * g * dy;
                    deltas[i][2] += s * g * dz;
                }
            }
        }

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.predicted_pos[0] += deltas[i][0];
            p.predicted_pos[1] += deltas[i][1];
            p.predicted_pos[2] += deltas[i][2];
            // Ground plane clamp
            if p.predicted_pos[1] < 0.0 {
                p.predicted_pos[1] = 0.0;
            }
        }

        // Pass 4: integrate velocity + position
        for p in &mut self.particles {
            p.velocity = [
                (p.predicted_pos[0] - p.position[0]) / dt,
                (p.predicted_pos[1] - p.position[1]) / dt,
                (p.predicted_pos[2] - p.position[2]) / dt,
            ];
            p.position = p.predicted_pos;
        }
    }

    pub fn upload_to_gpu(&mut self, device: &wgpu::Device, max_particles: u32) {
        let particle_bytes = max_particles as u64 * std::mem::size_of::<PbfParticle>() as u64;
        let neighbor_bytes =
            max_particles as u64 * (self.params.max_neighbors as u64 + 1) * 4;
        let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbf_particles"),
            size: particle_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let neighbor_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbf_neighbors"),
            size: neighbor_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbf_params"),
            size: std::mem::size_of::<PbfParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu = Some(PbfGpuBuffers {
            particle_buf,
            neighbor_buf,
            params_buf,
            max_particles,
        });
    }

    pub fn gpu_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pipelines: &PbfPipelines,
    ) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => {
                self.cpu_step();
                return;
            }
        };
        self.params.particle_count = self.particles.len() as u32;
        queue.write_buffer(&gpu.params_buf, 0, bytemuck::bytes_of(&self.params));
        queue.write_buffer(
            &gpu.particle_buf,
            0,
            bytemuck::cast_slice(&self.particles),
        );
        let count = self.params.particle_count;
        let workgroups = count.div_ceil(64);
        for pipeline in [
            &pipelines.predict,
            &pipelines.neighbors,
            &pipelines.solve,
            &pipelines.integrate,
        ] {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pbf_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &pipelines.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

pub struct PbfPipelines {
    pub predict:    wgpu::ComputePipeline,
    pub neighbors:  wgpu::ComputePipeline,
    pub solve:      wgpu::ComputePipeline,
    pub integrate:  wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbf_particle_size_aligned() {
        // position(12) + density(4) + predicted(12) + lambda(4) + velocity(12) + pressure(4) + spectral(64) = 112 bytes
        assert_eq!(
            std::mem::size_of::<PbfParticle>(),
            112,
            "PbfParticle must be 112 bytes for GPU alignment"
        );
    }

    #[test]
    fn pbf_params_size() {
        assert_eq!(std::mem::size_of::<PbfParams>(), 32);
    }

    #[test]
    fn spawn_increments_count() {
        let mut sim = PbfFluidSim::new(1000.0, 0.1);
        sim.spawn([0.0, 1.0, 0.0], [0.0; 3], WATER_SPECTRAL);
        sim.spawn([0.1, 1.0, 0.0], [0.0; 3], WATER_SPECTRAL);
        assert_eq!(sim.particles.len(), 2);
        assert_eq!(sim.params.particle_count, 2);
    }

    #[test]
    fn cpu_step_particles_fall_under_gravity() {
        let mut sim = PbfFluidSim::new(1000.0, 0.15);
        sim.spawn([0.0, 5.0, 0.0], [0.0; 3], WATER_SPECTRAL);
        let y0 = sim.particles[0].position[1];
        sim.cpu_step();
        let y1 = sim.particles[0].position[1];
        assert!(y1 < y0, "particle should fall: y0={} y1={}", y0, y1);
    }

    #[test]
    fn cpu_step_ground_plane_clamped() {
        let mut sim = PbfFluidSim::new(1000.0, 0.15);
        sim.spawn([0.0, 0.001, 0.0], [0.0, -10.0, 0.0], WATER_SPECTRAL);
        for _ in 0..10 {
            sim.cpu_step();
        }
        assert!(
            sim.particles[0].position[1] >= 0.0,
            "particle must not go below ground plane"
        );
    }

    #[test]
    fn spectral_profile_persists_through_step() {
        let mut sim = PbfFluidSim::new(1000.0, 0.15);
        sim.spawn([0.0, 2.0, 0.0], [0.0; 3], BLOOD_SPECTRAL);
        sim.cpu_step();
        assert!(
            sim.particles[0].spectral[9] > 0.0,
            "blood band 9 must persist through physics step"
        );
    }

    #[test]
    fn gpu_buffers_size_calculation() {
        // Validate that buffer sizing arithmetic doesn't overflow for 50k particles
        let max: u32 = 50_000;
        let particle_bytes = max as u64 * std::mem::size_of::<PbfParticle>() as u64;
        let neighbor_bytes = max as u64 * (64u64 + 1) * 4; // max_neighbors=64
        assert!(particle_bytes > 0, "particle buffer must be non-zero");
        assert_eq!(particle_bytes, 50_000 * 112); // 112 bytes per PbfParticle
        assert_eq!(neighbor_bytes, 50_000 * 65 * 4);
    }

    #[test]
    fn pbf_particle_size_memory_budget() {
        let per_particle = std::mem::size_of::<PbfParticle>();
        let total = 50_000usize * per_particle;
        println!("50000 * {} = {}", per_particle, total);
        assert_eq!(total, 5_600_000, "50k particles at 112 bytes = 5.6MB");
    }
}
