use glam::Vec3;

/// Position-based dynamics cloth simulation.
pub struct ClothSimulation {
    pub particles: Vec<ClothParticle>,
    pub constraints: Vec<DistanceConstraint>,
    pub gravity: Vec3,
    pub damping: f32,
    pub iterations: u32,
}

/// A single cloth particle.
pub struct ClothParticle {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub acceleration: Vec3,
    pub mass: f32,
    pub pinned: bool,
}

/// Distance constraint between two particles.
pub struct DistanceConstraint {
    pub particle_a: usize,
    pub particle_b: usize,
    pub rest_length: f32,
    pub stiffness: f32,
}

impl ClothSimulation {
    /// Create a grid of particles connected by distance constraints.
    /// Top row (y=0 row, i.e. first row) is pinned.
    pub fn new_grid(width: usize, height: usize, spacing: f32) -> Self {
        let mut particles = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let pos = Vec3::new(x as f32 * spacing, -(y as f32) * spacing, 0.0);
                particles.push(ClothParticle {
                    position: pos,
                    prev_position: pos,
                    acceleration: Vec3::ZERO,
                    mass: 1.0,
                    pinned: y == 0, // pin top row
                });
            }
        }

        let mut constraints = Vec::new();
        let diag_len = spacing * std::f32::consts::SQRT_2;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                // Horizontal
                if x + 1 < width {
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + 1,
                        rest_length: spacing,
                        stiffness: 1.0,
                    });
                }
                // Vertical
                if y + 1 < height {
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + width,
                        rest_length: spacing,
                        stiffness: 1.0,
                    });
                }
                // Diagonal (down-right)
                if x + 1 < width && y + 1 < height {
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + width + 1,
                        rest_length: diag_len,
                        stiffness: 0.5,
                    });
                }
                // Diagonal (down-left)
                if x > 0 && y + 1 < height {
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + width - 1,
                        rest_length: diag_len,
                        stiffness: 0.5,
                    });
                }
            }
        }

        Self {
            particles,
            constraints,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.01,
            iterations: 5,
        }
    }

    /// Step the simulation forward by dt seconds.
    pub fn step(&mut self, dt: f32) {
        // Verlet integration
        for particle in &mut self.particles {
            if particle.pinned {
                continue;
            }
            let velocity = particle.position - particle.prev_position;
            particle.prev_position = particle.position;
            particle.position += velocity * (1.0 - self.damping)
                + (self.gravity + particle.acceleration) * dt * dt;
            particle.acceleration = Vec3::ZERO;
        }

        // Constraint solving (multiple iterations for stability)
        for _ in 0..self.iterations {
            for ci in 0..self.constraints.len() {
                let pa_idx = self.constraints[ci].particle_a;
                let pb_idx = self.constraints[ci].particle_b;
                let rest_length = self.constraints[ci].rest_length;
                let stiffness = self.constraints[ci].stiffness;

                let a = self.particles[pa_idx].position;
                let b = self.particles[pb_idx].position;
                let diff = b - a;
                let dist = diff.length();
                if dist < 0.0001 {
                    continue;
                }
                let correction = diff * (1.0 - rest_length / dist) * 0.5 * stiffness;

                let a_pinned = self.particles[pa_idx].pinned;
                let b_pinned = self.particles[pb_idx].pinned;

                if !a_pinned {
                    self.particles[pa_idx].position += correction;
                }
                if !b_pinned {
                    self.particles[pb_idx].position -= correction;
                }
            }
        }
    }

    /// Convert cloth particles to splat positions.
    pub fn to_positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Apply wind force to all unpinned particles.
    pub fn apply_wind(&mut self, force: Vec3) {
        for particle in &mut self.particles {
            if !particle.pinned {
                particle.acceleration += force / particle.mass;
            }
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_creates_correct_particle_count() {
        let cloth = ClothSimulation::new_grid(4, 3, 1.0);
        assert_eq!(cloth.particle_count(), 12);
    }

    #[test]
    fn pinned_particles_dont_move() {
        let mut cloth = ClothSimulation::new_grid(3, 3, 1.0);
        let pinned_positions: Vec<Vec3> = cloth
            .particles
            .iter()
            .filter(|p| p.pinned)
            .map(|p| p.position)
            .collect();

        for _ in 0..10 {
            cloth.step(1.0 / 60.0);
        }

        let pinned_after: Vec<Vec3> = cloth
            .particles
            .iter()
            .filter(|p| p.pinned)
            .map(|p| p.position)
            .collect();

        for (before, after) in pinned_positions.iter().zip(&pinned_after) {
            assert!(
                (*before - *after).length() < 0.0001,
                "Pinned particle moved: {before} -> {after}"
            );
        }
    }

    #[test]
    fn gravity_pulls_cloth_down() {
        let mut cloth = ClothSimulation::new_grid(3, 3, 1.0);
        let initial_y: f32 = cloth
            .particles
            .iter()
            .filter(|p| !p.pinned)
            .map(|p| p.position.y)
            .sum();

        for _ in 0..10 {
            cloth.step(1.0 / 60.0);
        }

        let final_y: f32 = cloth
            .particles
            .iter()
            .filter(|p| !p.pinned)
            .map(|p| p.position.y)
            .sum();

        assert!(
            final_y < initial_y,
            "Gravity should pull cloth down: initial_y={initial_y}, final_y={final_y}"
        );
    }

    #[test]
    fn constraint_maintains_approximate_distance() {
        let mut cloth = ClothSimulation::new_grid(2, 2, 1.0);
        cloth.iterations = 20; // more iterations for tighter constraints

        for _ in 0..30 {
            cloth.step(1.0 / 60.0);
        }

        // Check that constraint distances are roughly maintained
        for constraint in &cloth.constraints {
            let a = cloth.particles[constraint.particle_a].position;
            let b = cloth.particles[constraint.particle_b].position;
            let dist = a.distance(b);
            let ratio = dist / constraint.rest_length;
            assert!(
                ratio > 0.5 && ratio < 2.0,
                "Constraint distance diverged too much: dist={dist}, rest={}",
                constraint.rest_length
            );
        }
    }

    #[test]
    fn wind_applies_force() {
        let mut cloth = ClothSimulation::new_grid(3, 3, 1.0);
        cloth.gravity = Vec3::ZERO; // disable gravity for this test

        cloth.apply_wind(Vec3::new(0.0, 0.0, 10.0));
        for _ in 0..10 {
            cloth.step(1.0 / 60.0);
        }

        // Unpinned particles should have moved in Z direction
        let has_z_movement = cloth
            .particles
            .iter()
            .filter(|p| !p.pinned)
            .any(|p| p.position.z > 0.001);
        assert!(has_z_movement, "Wind should push particles in Z direction");
    }
}
