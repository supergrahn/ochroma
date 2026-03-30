//! XPBD soft body simulation — Extended Position-Based Dynamics.
//! 8 substeps per physics frame at dt=1/480s for stability.

use glam::Vec3;
use vox_core::types::GaussianSplat;

const SUBSTEPS: u32 = 8;
const PHYSICS_DT: f32 = 1.0 / 480.0;

#[derive(Debug, Clone)]
pub struct SoftParticle {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub velocity: Vec3,
    pub inv_mass: f32, // 0.0 = pinned (infinite mass)
}

#[derive(Debug, Clone)]
pub enum SoftConstraint {
    Distance {
        a: usize,
        b: usize,
        rest_length: f32,
        compliance: f32, // α — lower = stiffer. 0.0 = rigid
    },
    Volume {
        particles: [usize; 4], // tetrahedron
        rest_volume: f32,
        compliance: f32,
    },
    ShapeMatch {
        particle_indices: Vec<usize>,
        rest_positions: Vec<Vec3>,
        stiffness: f32,
    },
}

pub struct SoftBody {
    pub particles: Vec<SoftParticle>,
    pub constraints: Vec<SoftConstraint>,
    /// Maps each GaussianSplat to the nearest particle (by index).
    pub splat_bindings: Vec<usize>,
    pub gravity: Vec3,
    /// Accumulated substep time (handle fractional frames)
    time_accumulator: f32,
}

impl SoftBody {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            particles: Vec::new(),
            constraints: Vec::new(),
            splat_bindings: Vec::new(),
            gravity,
            time_accumulator: 0.0,
        }
    }

    pub fn add_particle(&mut self, pos: Vec3, mass: f32) -> usize {
        let idx = self.particles.len();
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.particles.push(SoftParticle {
            position: pos,
            prev_position: pos,
            velocity: Vec3::ZERO,
            inv_mass,
        });
        idx
    }

    pub fn pin_particle(&mut self, idx: usize) {
        self.particles[idx].inv_mass = 0.0;
    }

    pub fn add_distance_constraint(&mut self, a: usize, b: usize, compliance: f32) {
        let rest = (self.particles[a].position - self.particles[b].position).length();
        self.constraints.push(SoftConstraint::Distance {
            a,
            b,
            rest_length: rest,
            compliance,
        });
    }

    /// Advance by `dt` seconds, running SUBSTEPS substeps per PHYSICS_DT interval.
    pub fn step(&mut self, dt: f32) {
        self.time_accumulator += dt;
        let h = PHYSICS_DT / SUBSTEPS as f32;

        while self.time_accumulator >= PHYSICS_DT {
            for _ in 0..SUBSTEPS {
                self.substep(h);
            }
            self.time_accumulator -= PHYSICS_DT;
        }
    }

    fn substep(&mut self, h: f32) {
        // (a) Integrate: store prev, apply gravity, advance position
        for p in &mut self.particles {
            if p.inv_mass == 0.0 {
                continue; // pinned
            }
            p.prev_position = p.position;
            p.velocity += self.gravity * h;
            p.position += p.velocity * h;
        }

        // (b) Solve constraints (Gauss-Seidel, 1 iteration per substep for XPBD)
        let h2 = h * h;
        for constraint in &self.constraints {
            match constraint {
                SoftConstraint::Distance {
                    a,
                    b,
                    rest_length,
                    compliance,
                } => {
                    let a = *a;
                    let b = *b;
                    let rest_length = *rest_length;
                    let compliance = *compliance;

                    let pos_a = self.particles[a].position;
                    let pos_b = self.particles[b].position;
                    let delta = pos_a - pos_b;
                    let dist = delta.length();

                    if dist < 1e-10 {
                        continue;
                    }

                    let dir = delta / dist;
                    let c = dist - rest_length;

                    let inv_ma = self.particles[a].inv_mass;
                    let inv_mb = self.particles[b].inv_mass;
                    let w_sum = inv_ma + inv_mb;

                    if w_sum < 1e-10 {
                        continue; // both pinned
                    }

                    // XPBD compliance correction: compliance/h^2 softens the constraint
                    let correction = c / (w_sum + compliance / h2);

                    self.particles[a].position -= dir * (inv_ma * correction);
                    self.particles[b].position += dir * (inv_mb * correction);
                }
                SoftConstraint::Volume {
                    particles,
                    rest_volume,
                    compliance,
                } => {
                    // Tetrahedral volume constraint
                    let [i0, i1, i2, i3] = *particles;
                    let p0 = self.particles[i0].position;
                    let p1 = self.particles[i1].position;
                    let p2 = self.particles[i2].position;
                    let p3 = self.particles[i3].position;

                    let v1 = p1 - p0;
                    let v2 = p2 - p0;
                    let v3 = p3 - p0;
                    let vol = v1.dot(v2.cross(v3)) / 6.0;
                    let c = vol - rest_volume;

                    // Gradients of volume w.r.t. each particle position
                    let grad0 = (v2.cross(v3) + v3.cross(v1) + v1.cross(v2)) / 6.0;
                    let grad1 = (p2 - p0).cross(p3 - p0) / 6.0;
                    let grad2 = (p3 - p0).cross(p1 - p0) / 6.0;
                    let grad3 = (p1 - p0).cross(p2 - p0) / 6.0;

                    let inv_masses = [
                        self.particles[i0].inv_mass,
                        self.particles[i1].inv_mass,
                        self.particles[i2].inv_mass,
                        self.particles[i3].inv_mass,
                    ];
                    let grads = [grad0, grad1, grad2, grad3];
                    let w_sum: f32 = inv_masses
                        .iter()
                        .zip(grads.iter())
                        .map(|(w, g)| w * g.length_squared())
                        .sum();

                    if w_sum < 1e-10 {
                        continue;
                    }

                    let lambda = -c / (w_sum + compliance / h2);

                    self.particles[i0].position += inv_masses[0] * lambda * grads[0];
                    self.particles[i1].position += inv_masses[1] * lambda * grads[1];
                    self.particles[i2].position += inv_masses[2] * lambda * grads[2];
                    self.particles[i3].position += inv_masses[3] * lambda * grads[3];
                }
                SoftConstraint::ShapeMatch {
                    particle_indices,
                    rest_positions,
                    stiffness,
                } => {
                    // Simple shape matching: pull each particle toward its rest
                    // position (relative to the group centroid).
                    if particle_indices.len() != rest_positions.len() || particle_indices.is_empty() {
                        continue;
                    }

                    // Current centroid
                    let centroid: Vec3 = particle_indices
                        .iter()
                        .map(|&i| self.particles[i].position)
                        .sum::<Vec3>()
                        / particle_indices.len() as f32;

                    // Rest centroid
                    let rest_centroid: Vec3 =
                        rest_positions.iter().copied().sum::<Vec3>() / rest_positions.len() as f32;

                    for (local_idx, &global_idx) in particle_indices.iter().enumerate() {
                        if self.particles[global_idx].inv_mass == 0.0 {
                            continue;
                        }
                        let rest_rel = rest_positions[local_idx] - rest_centroid;
                        let goal = centroid + rest_rel;
                        let diff = goal - self.particles[global_idx].position;
                        self.particles[global_idx].position += diff * stiffness;
                    }
                }
            }
        }

        // (c) Update velocities from position change
        for p in &mut self.particles {
            if p.inv_mass == 0.0 {
                continue;
            }
            p.velocity = (p.position - p.prev_position) / h;
        }
    }

    /// Apply the current particle positions to the bound splats.
    pub fn apply_to_splats(&self, splats: &mut [GaussianSplat]) {
        for (splat_idx, &particle_idx) in self.splat_bindings.iter().enumerate() {
            if splat_idx >= splats.len() || particle_idx >= self.particles.len() {
                continue;
            }
            let pos = self.particles[particle_idx].position;
            splats[splat_idx].set_position([pos.x, pos.y, pos.z]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_particle_body(compliance: f32) -> SoftBody {
        let mut body = SoftBody::new(Vec3::new(0.0, -9.81, 0.0));
        let a = body.add_particle(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let b = body.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        body.add_distance_constraint(a, b, compliance);
        body
    }

    #[test]
    fn soft_body_distance_constraint_maintains_length() {
        let mut body = make_two_particle_body(0.0);

        // Run for 60 full physics steps (each 1/60s)
        for _ in 0..60 {
            body.step(1.0 / 60.0);
        }

        let pa = body.particles[0].position;
        let pb = body.particles[1].position;
        let dist = pa.distance(pb);

        assert!(
            (dist - 1.0).abs() < 0.1,
            "Distance constraint should maintain ~1.0 rest length after 60 steps, got {dist}"
        );
    }

    #[test]
    fn pinned_particle_doesnt_move() {
        let mut body = SoftBody::new(Vec3::new(0.0, -9.81, 0.0));
        let pinned = body.add_particle(Vec3::ZERO, 1.0);
        body.pin_particle(pinned);

        for _ in 0..60 {
            body.step(1.0 / 60.0);
        }

        let pos = body.particles[pinned].position;
        assert!(
            pos.length() < 1e-5,
            "Pinned particle should stay at origin, got {pos}"
        );
    }

    #[test]
    fn soft_body_falls_under_gravity() {
        let mut body = SoftBody::new(Vec3::new(0.0, -9.81, 0.0));
        body.add_particle(Vec3::new(0.0, 1.0, 0.0), 1.0);

        // Simulate 0.5 seconds — free fall: y = 1 - 0.5*9.81*0.5^2 ≈ -0.226
        body.step(0.5);

        let y = body.particles[0].position.y;
        assert!(
            y < 0.5,
            "Particle should have fallen below y=0.5 after 0.5s under gravity, got y={y}"
        );
    }
}
