//! Simplified SPH (Smoothed Particle Hydrodynamics) fluid simulation.

use glam::Vec3;

/// A single SPH particle.
#[derive(Debug, Clone)]
pub struct FluidParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub density: f32,
    pub pressure: f32,
}

/// SPH fluid simulation.
pub struct FluidSimulation {
    pub particles: Vec<FluidParticle>,
    pub gravity: Vec3,
    pub rest_density: f32,
    pub stiffness: f32,
    pub viscosity: f32,
    pub particle_radius: f32,
}

impl FluidSimulation {
    pub fn new(rest_density: f32, stiffness: f32) -> Self {
        Self {
            particles: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            rest_density,
            stiffness,
            viscosity: 0.1,
            particle_radius: 0.5,
        }
    }

    pub fn add_particle(&mut self, position: Vec3) {
        self.particles.push(FluidParticle {
            position,
            velocity: Vec3::ZERO,
            density: 0.0,
            pressure: 0.0,
        });
    }

    /// Add a block of particles in a grid pattern.
    pub fn add_block(&mut self, center: Vec3, size: Vec3, spacing: f32) {
        let half = size * 0.5;
        let mut x = -half.x;
        while x <= half.x {
            let mut y = -half.y;
            while y <= half.y {
                let mut z = -half.z;
                while z <= half.z {
                    self.add_particle(center + Vec3::new(x, y, z));
                    z += spacing;
                }
                y += spacing;
            }
            x += spacing;
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn to_positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// SPH kernel (poly6-like, simplified).
    fn kernel(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        let x = 1.0 - (r / h) * (r / h);
        // Simplified normalization
        (315.0 / (64.0 * std::f32::consts::PI * h.powi(3))) * x * x * x
    }

    /// Gradient of spiky kernel for pressure.
    fn kernel_gradient(&self, r_vec: Vec3, r: f32, h: f32) -> Vec3 {
        if r >= h || r < 1e-6 {
            return Vec3::ZERO;
        }
        let x = h - r;
        let coeff = -45.0 / (std::f32::consts::PI * h.powi(6)) * x * x;
        (r_vec / r) * coeff
    }

    /// Laplacian of viscosity kernel.
    fn kernel_laplacian(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        45.0 / (std::f32::consts::PI * h.powi(6)) * (h - r)
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let h = self.particle_radius * 2.0; // smoothing radius
        let n = self.particles.len();
        if n == 0 {
            return;
        }

        // Step 1: Compute density for each particle
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        let velocities: Vec<Vec3> = self.particles.iter().map(|p| p.velocity).collect();

        let mut densities = vec![0.0f32; n];
        for i in 0..n {
            let mut density = 0.0f32;
            for j in 0..n {
                let r = positions[i].distance(positions[j]);
                density += self.kernel(r, h);
            }
            // Each particle has unit mass
            densities[i] = density.max(self.rest_density * 0.1);
        }

        // Step 2: Compute pressure from density
        let mut pressures = vec![0.0f32; n];
        for i in 0..n {
            pressures[i] = self.stiffness * (densities[i] - self.rest_density);
        }

        // Store density/pressure
        for i in 0..n {
            self.particles[i].density = densities[i];
            self.particles[i].pressure = pressures[i];
        }

        // Step 3: Compute forces (pressure + viscosity + gravity)
        let mut forces = vec![Vec3::ZERO; n];
        for i in 0..n {
            let mut pressure_force = Vec3::ZERO;
            let mut viscosity_force = Vec3::ZERO;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let r_vec = positions[i] - positions[j];
                let r = r_vec.length();

                // Pressure force
                let avg_pressure = (pressures[i] + pressures[j]) * 0.5;
                let grad = self.kernel_gradient(r_vec, r, h);
                if densities[j] > 1e-6 {
                    pressure_force -= grad * (avg_pressure / densities[j]);
                }

                // Viscosity force
                let lap = self.kernel_laplacian(r, h);
                if densities[j] > 1e-6 {
                    viscosity_force += (velocities[j] - velocities[i]) * (lap / densities[j]);
                }
            }

            viscosity_force *= self.viscosity;
            forces[i] = pressure_force + viscosity_force + self.gravity;
        }

        // Step 4: Integrate
        for i in 0..n {
            self.particles[i].velocity += forces[i] * dt;
            let vel = self.particles[i].velocity;
            self.particles[i].position += vel * dt;

            // Simple ground plane at y=0
            if self.particles[i].position.y < 0.0 {
                self.particles[i].position.y = 0.0;
                self.particles[i].velocity.y *= -0.3; // damped bounce
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particles_fall_under_gravity() {
        let mut sim = FluidSimulation::new(1000.0, 50.0);
        sim.add_particle(Vec3::new(0.0, 5.0, 0.0));
        let start_y = sim.particles[0].position.y;

        for _ in 0..10 {
            sim.step(0.01);
        }

        let end_y = sim.particles[0].position.y;
        assert!(end_y < start_y, "particle should fall: {start_y} -> {end_y}");
    }

    #[test]
    fn density_computed_for_uniform_grid() {
        let mut sim = FluidSimulation::new(1000.0, 50.0);
        sim.add_block(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0.5);
        assert!(sim.particle_count() > 0);

        sim.step(0.001); // single tiny step to compute densities

        // All particles should have non-zero density
        for p in &sim.particles {
            assert!(p.density > 0.0, "density should be positive: {}", p.density);
        }
    }

    #[test]
    fn block_of_particles_settles() {
        let mut sim = FluidSimulation::new(100.0, 5.0);
        sim.viscosity = 0.5;
        sim.particle_radius = 0.3;
        sim.add_block(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.8, 0.8, 0.8), 0.4);
        let count = sim.particle_count();

        // Run with small timesteps for stability
        for _ in 0..300 {
            sim.step(0.002);
        }

        // Particles should have moved down toward ground
        let avg_y: f32 = sim.particles.iter().map(|p| p.position.y).sum::<f32>() / count as f32;
        assert!(avg_y < 3.0, "particles should settle downward, avg_y: {avg_y}");
    }

    #[test]
    fn particle_count_preserved() {
        let mut sim = FluidSimulation::new(1000.0, 50.0);
        sim.add_block(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0.5);
        let count = sim.particle_count();
        assert!(count > 0);

        for _ in 0..50 {
            sim.step(0.01);
        }

        assert_eq!(sim.particle_count(), count);
        assert_eq!(sim.to_positions().len(), count);
    }
}
