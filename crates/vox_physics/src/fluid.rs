//! Simplified SPH (Smoothed Particle Hydrodynamics) fluid simulation.
//! Includes spectral emission profiles, surface foam particles, and SDF ground-plane collision.

use glam::Vec3;

// ---------------------------------------------------------------------------
// Spectral profiles
// ---------------------------------------------------------------------------

/// Spectral emission profile for fluid materials.
#[derive(Debug, Clone)]
pub struct FluidSpectralProfile {
    pub name: &'static str,
    /// Base spectral radiance per band at unit density (8 bands).
    pub emission: [f32; 8],
}

/// Predefined spectral profiles.
pub const WATER_PROFILE: FluidSpectralProfile = FluidSpectralProfile {
    name: "water",
    // Deep blue-cyan: high band 1-2, low elsewhere
    emission: [0.1, 0.8, 0.9, 0.3, 0.1, 0.05, 0.02, 0.01],
};

pub const LAVA_PROFILE: FluidSpectralProfile = FluidSpectralProfile {
    name: "lava",
    // Red-orange thermal: high bands 5-7, low high-freq
    emission: [0.0, 0.0, 0.02, 0.1, 0.4, 0.8, 1.0, 1.0],
};

pub const MUD_PROFILE: FluidSpectralProfile = FluidSpectralProfile {
    name: "mud",
    emission: [0.05, 0.05, 0.08, 0.1, 0.12, 0.1, 0.08, 0.05],
};

pub const BLOOD_PROFILE: FluidSpectralProfile = FluidSpectralProfile {
    name: "blood",
    emission: [0.0, 0.0, 0.01, 0.05, 0.2, 0.6, 0.4, 0.1],
};

// ---------------------------------------------------------------------------
// Foam particles
// ---------------------------------------------------------------------------

/// Surface foam particle spawned when fluid pressure exceeds threshold.
#[derive(Debug, Clone)]
pub struct FoamParticle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub lifetime: f32,
    pub age: f32,
}

// ---------------------------------------------------------------------------
// SPH core types
// ---------------------------------------------------------------------------

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
    /// Spectral emission profile for this fluid.
    pub profile: FluidSpectralProfile,
    /// Surface foam particles.
    pub foam_particles: Vec<FoamParticle>,
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
            profile: WATER_PROFILE.clone(),
            foam_particles: Vec::new(),
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

    /// Returns the spectral emission at a particle, scaled by its density.
    pub fn get_spectral_at(&self, particle: &FluidParticle) -> [f32; 8] {
        let scale = particle.density;
        let e = &self.profile.emission;
        [
            e[0] * scale,
            e[1] * scale,
            e[2] * scale,
            e[3] * scale,
            e[4] * scale,
            e[5] * scale,
            e[6] * scale,
            e[7] * scale,
        ]
    }

    /// Step foam particles: spawn new foam at high-pressure fluid particles,
    /// integrate existing foam, and remove expired foam.
    pub fn step_foam(&mut self, dt: f32) {
        const MAX_FOAM: usize = 500;
        const FOAM_LIFETIME: f32 = 2.0;
        const FOAM_PRESSURE_THRESHOLD_MULTIPLIER: f32 = 5.0;
        const FOAM_GRAVITY_Y: f32 = -9.81;
        const FOAM_DAMPING: f32 = 0.95;

        let rest_pressure = self.rest_density * self.stiffness.max(0.001);
        let threshold = rest_pressure * FOAM_PRESSURE_THRESHOLD_MULTIPLIER;

        // Spawn foam at high-pressure particles
        for particle in &self.particles {
            if self.foam_particles.len() >= MAX_FOAM {
                break;
            }
            if particle.pressure > threshold {
                // Spawn a foam particle at this location with a small upward kick
                self.foam_particles.push(FoamParticle {
                    position: particle.position.to_array(),
                    velocity: [
                        particle.velocity.x * 0.1,
                        particle.velocity.y.abs() * 0.2 + 0.5,
                        particle.velocity.z * 0.1,
                    ],
                    lifetime: FOAM_LIFETIME,
                    age: 0.0,
                });
            }
        }

        // Integrate and age foam particles
        for foam in &mut self.foam_particles {
            foam.velocity[1] += FOAM_GRAVITY_Y * dt;
            foam.velocity[0] *= FOAM_DAMPING;
            foam.velocity[1] *= FOAM_DAMPING;
            foam.velocity[2] *= FOAM_DAMPING;
            foam.position[0] += foam.velocity[0] * dt;
            foam.position[1] += foam.velocity[1] * dt;
            foam.position[2] += foam.velocity[2] * dt;

            // Ground plane SDF collision
            if foam.position[1] < 0.0 {
                foam.position[1] = 0.0;
                foam.velocity[1] = foam.velocity[1].abs() * 0.2;
            }

            foam.age += dt;
        }

        // Remove expired foam
        self.foam_particles.retain(|f| f.age < f.lifetime);
    }

    // ---------------------------------------------------------------------------
    // SPH kernels
    // ---------------------------------------------------------------------------

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

    // ---------------------------------------------------------------------------
    // Tait equation of state for pressure
    // ---------------------------------------------------------------------------

    /// Tait equation of state: p = B * ((rho/rho0)^gamma - 1).
    /// Uses gamma=7 (water-like). B is derived from stiffness.
    fn tait_pressure(&self, density: f32) -> f32 {
        const GAMMA: f32 = 7.0;
        let b = self.stiffness;
        let ratio = density / self.rest_density;
        b * (ratio.powf(GAMMA) - 1.0)
    }

    // ---------------------------------------------------------------------------
    // Simulation step
    // ---------------------------------------------------------------------------

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

        // Step 2: Compute pressure from density — Tait equation of state
        let mut pressures = vec![0.0f32; n];
        for i in 0..n {
            pressures[i] = self.tait_pressure(densities[i]);
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
        for (i, force) in forces.iter().enumerate().take(n) {
            self.particles[i].velocity += *force * dt;
            let vel = self.particles[i].velocity;
            self.particles[i].position += vel * dt;

            // SDF ground plane at y=0
            if self.particles[i].position.y < 0.0 {
                self.particles[i].position.y = 0.0;
                self.particles[i].velocity.y *= -0.3; // damped bounce
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralFluid — PBF-backed fluid with per-particle spectral[16]
// ---------------------------------------------------------------------------

use crate::pbf::{PbfFluidSim, BLOOD_SPECTRAL, LAVA_SPECTRAL, WATER_SPECTRAL};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralFluidKind {
    Water,
    Blood,
    Lava,
    Custom([f32; 16]),
}

pub struct SpectralFluid {
    pub kind: SpectralFluidKind,
    pub sim: PbfFluidSim,
}

impl SpectralFluid {
    pub fn new(kind: SpectralFluidKind) -> Self {
        Self { kind, sim: PbfFluidSim::new(1000.0, 0.1) }
    }

    pub fn spectral_for_kind(kind: SpectralFluidKind) -> [f32; 16] {
        match kind {
            SpectralFluidKind::Water => WATER_SPECTRAL,
            SpectralFluidKind::Blood => BLOOD_SPECTRAL,
            SpectralFluidKind::Lava => LAVA_SPECTRAL,
            SpectralFluidKind::Custom(s) => s,
        }
    }

    /// Spawn a particle with the kind's canonical spectral profile.
    pub fn spawn(&mut self, pos: [f32; 3], vel: [f32; 3]) {
        let spectral = Self::spectral_for_kind(self.kind);
        self.sim.spawn(pos, vel, spectral);
    }

    /// Spawn a particle whose spectral profile is linearly mixed between
    /// the kind's profile and `mix` at weight `mix_weight ∈ [0,1]`.
    pub fn spawn_mixed(&mut self, pos: [f32; 3], vel: [f32; 3], mix: &[f32; 16], mix_weight: f32) {
        let base = Self::spectral_for_kind(self.kind);
        let w = mix_weight.clamp(0.0, 1.0);
        let mut spectral = [0.0f32; 16];
        for b in 0..16 {
            spectral[b] = base[b] * (1.0 - w) + mix[b] * w;
        }
        self.sim.spawn(pos, vel, spectral);
    }

    /// Advance simulation by one CPU step.
    pub fn step(&mut self) {
        self.sim.cpu_step();
    }

    pub fn particle_count(&self) -> usize {
        self.sim.particles.len()
    }

    /// Return the mean spectral radiance across all particles.
    pub fn mean_spectral(&self) -> [f32; 16] {
        let n = self.sim.particles.len();
        if n == 0 {
            return [0.0f32; 16];
        }
        let mut acc = [0.0f32; 16];
        for p in &self.sim.particles {
            for (a, &s) in acc.iter_mut().zip(p.spectral.iter()) {
                *a += s;
            }
        }
        for v in &mut acc {
            *v /= n as f32;
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// SpectralFluid tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod spectral_fluid_tests {
    use super::*;

    #[test]
    fn water_particles_have_blue_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
        fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
        let s = &fluid.sim.particles[0].spectral;
        assert!(s[1] > s[12], "water band 1 (blue) must exceed band 12 (red)");
    }

    #[test]
    fn blood_particles_have_red_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Blood);
        fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
        let s = &fluid.sim.particles[0].spectral;
        assert!(s[9] > s[0], "blood band 9 (red) must exceed band 0 (violet)");
    }

    #[test]
    fn mixed_spawn_blends_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
        fluid.spawn_mixed([0.0, 1.0, 0.0], [0.0; 3], &LAVA_SPECTRAL, 0.5);
        let s = fluid.sim.particles[0].spectral;
        assert!(
            s[14] > WATER_SPECTRAL[14],
            "mixed particle band 14 should exceed pure water"
        );
        assert!(
            s[14] < LAVA_SPECTRAL[14],
            "mixed particle band 14 should not fully reach lava"
        );
    }

    #[test]
    fn step_does_not_zero_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
        for i in 0..5 {
            fluid.spawn([i as f32 * 0.1, 2.0, 0.0], [0.0; 3]);
        }
        fluid.step();
        let mean = fluid.mean_spectral();
        assert!(mean[2] > 0.0, "water blue band must persist after physics step");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Existing tests (must keep passing) ---

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

    // --- New spectral / foam tests ---

    #[test]
    fn water_profile_has_high_blue() {
        // emission[1] (cyan-blue) + emission[2] (blue-green) > 1.0
        let sum = WATER_PROFILE.emission[1] + WATER_PROFILE.emission[2];
        assert!(
            sum > 1.0,
            "water should have high blue-cyan bands: emission[1]+emission[2] = {sum}"
        );
    }

    #[test]
    fn lava_profile_has_high_red() {
        // emission[6] (red) + emission[7] (far red) > 1.5
        let sum = LAVA_PROFILE.emission[6] + LAVA_PROFILE.emission[7];
        assert!(
            sum > 1.5,
            "lava should have high red bands: emission[6]+emission[7] = {sum}"
        );
    }

    #[test]
    fn foam_spawns_at_high_pressure() {
        let mut sim = FluidSimulation::new(1000.0, 50.0);
        sim.add_particle(Vec3::new(0.0, 1.0, 0.0));

        // Force a high pressure value directly
        sim.particles[0].pressure = 1_000_000.0;
        sim.particles[0].density = 5000.0;

        assert_eq!(sim.foam_particles.len(), 0);
        sim.step_foam(0.016);
        assert!(
            sim.foam_particles.len() > 0,
            "foam should spawn when pressure is very high"
        );
    }

    #[test]
    fn spectral_at_returns_scaled_emission() {
        let sim = FluidSimulation::new(1000.0, 50.0);
        let particle = FluidParticle {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            density: 2.0,
            pressure: 0.0,
        };

        let spectral = sim.get_spectral_at(&particle);

        // All bands should be scaled by density=2.0
        for (band_idx, (s, e)) in spectral.iter().zip(sim.profile.emission.iter()).enumerate() {
            let expected = e * 2.0;
            assert!(
                (s - expected).abs() < 1e-6,
                "band {band_idx}: expected {expected}, got {s}"
            );
        }
    }
}
