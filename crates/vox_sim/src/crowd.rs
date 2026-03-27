//! Crowd simulation — many agents moving simultaneously with basic avoidance.

use glam::Vec3;

/// A single crowd agent.
pub struct CrowdAgent {
    pub position: Vec3,
    pub velocity: Vec3,
    pub target: Vec3,
    pub speed: f32,
    pub radius: f32,
}

/// Manages a crowd of agents with simple steering and avoidance.
pub struct CrowdSimulation {
    pub agents: Vec<CrowdAgent>,
    pub avoidance_weight: f32,
    pub separation_distance: f32,
}

impl CrowdSimulation {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            avoidance_weight: 2.0,
            separation_distance: 1.5,
        }
    }

    /// Add an agent and return its index.
    pub fn add_agent(&mut self, position: Vec3, target: Vec3, speed: f32) -> usize {
        let idx = self.agents.len();
        self.agents.push(CrowdAgent {
            position,
            velocity: Vec3::ZERO,
            target,
            speed,
            radius: 0.5,
        });
        idx
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Advance the simulation by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        let count = self.agents.len();
        // Compute desired velocities and avoidance forces
        let mut forces: Vec<Vec3> = Vec::with_capacity(count);

        for i in 0..count {
            let agent = &self.agents[i];
            let to_target = agent.target - agent.position;
            let dist_to_target = to_target.length();

            // Desired velocity toward target
            let desired = if dist_to_target > 0.01 {
                (to_target / dist_to_target) * agent.speed
            } else {
                Vec3::ZERO
            };

            // Avoidance force from nearby agents
            let mut avoidance = Vec3::ZERO;
            for j in 0..count {
                if i == j {
                    continue;
                }
                let other = &self.agents[j];
                let diff = agent.position - other.position;
                let dist = diff.length();
                let min_sep = self.separation_distance;
                if dist < min_sep && dist > 0.001 {
                    // Repulsion inversely proportional to distance
                    let strength = self.avoidance_weight * (min_sep - dist) / min_sep;
                    avoidance += (diff / dist) * strength;
                }
            }

            forces.push(desired + avoidance);
        }

        // Apply forces
        for (i, agent) in self.agents.iter_mut().enumerate() {
            let force = forces[i];
            let speed_limit = agent.speed * 1.5; // allow slight overshoot for avoidance
            agent.velocity = force;
            if agent.velocity.length() > speed_limit {
                agent.velocity = agent.velocity.normalize() * speed_limit;
            }
            agent.position += agent.velocity * dt;
        }
    }
}

impl Default for CrowdSimulation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agents_move_toward_targets() {
        let mut sim = CrowdSimulation::new();
        sim.add_agent(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 2.0);
        let start = sim.agents[0].position;
        for _ in 0..10 {
            sim.tick(0.1);
        }
        let end = sim.agents[0].position;
        assert!(end.x > start.x, "agent should move toward target: {start} -> {end}");
    }

    #[test]
    fn agents_dont_overlap() {
        let mut sim = CrowdSimulation::new();
        // Place two agents very close, heading toward same target
        sim.add_agent(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 2.0);
        sim.add_agent(Vec3::new(0.1, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 2.0);

        for _ in 0..100 {
            sim.tick(0.05);
        }

        let dist = sim.agents[0].position.distance(sim.agents[1].position);
        // They should maintain some separation (at least their radii)
        assert!(
            dist > 0.3,
            "agents should maintain separation, got distance {dist}"
        );
    }

    #[test]
    fn hundred_agents_no_panic() {
        let mut sim = CrowdSimulation::new();
        for i in 0..100 {
            let x = (i % 10) as f32 * 2.0;
            let z = (i / 10) as f32 * 2.0;
            sim.add_agent(
                Vec3::new(x, 0.0, z),
                Vec3::new(50.0, 0.0, 50.0),
                1.5,
            );
        }
        assert_eq!(sim.agent_count(), 100);
        for _ in 0..50 {
            sim.tick(0.05);
        }
        assert_eq!(sim.agent_count(), 100);
    }

    #[test]
    fn stationary_target_reached() {
        let mut sim = CrowdSimulation::new();
        let target = Vec3::new(3.0, 0.0, 0.0);
        sim.add_agent(Vec3::ZERO, target, 5.0);

        for _ in 0..200 {
            sim.tick(0.05);
        }

        let dist = sim.agents[0].position.distance(target);
        assert!(dist < 0.5, "agent should reach target, distance: {dist}");
    }
}
