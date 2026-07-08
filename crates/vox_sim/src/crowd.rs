//! Crowd simulation — many agents moving simultaneously with basic avoidance.

use glam::Vec3;
use rayon::prelude::*;

use crate::spatial_hash::SpatialHash;

/// A single crowd agent.
pub struct CrowdAgent {
    pub position: Vec3,
    pub velocity: Vec3,
    pub target: Vec3,
    pub speed: f32,
    pub radius: f32,
    pub path: Vec<[f32; 3]>,
    pub path_index: usize,
}

/// Minimum agent count before the per-agent avoidance pass is parallelized.
/// Below this the rayon fork/join overhead outweighs the work; the result is
/// identical either way (each agent writes only its own delta slot).
const PARALLEL_THRESHOLD: usize = 2048;

/// Manages a crowd of agents with simple steering and avoidance.
pub struct CrowdSimulation {
    pub agents: Vec<CrowdAgent>,
    pub avoidance_weight: f32,
    pub separation_distance: f32,
    /// Spatial hash reused across ticks (cleared + refilled), instead of a
    /// fresh allocation every `tick` — removes the per-tick map/bucket churn.
    hash: SpatialHash,
    /// Per-agent steering output, reused across ticks to avoid reallocation.
    deltas: Vec<Vec3>,
}

impl CrowdSimulation {
    pub fn new() -> Self {
        let separation_distance = 1.5;
        Self {
            agents: Vec::new(),
            avoidance_weight: 2.0,
            separation_distance,
            hash: SpatialHash::new(separation_distance * 1.5),
            deltas: Vec::new(),
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
            path: Vec::new(),
            path_index: 0,
        });
        idx
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Compute one agent's steering velocity from read-only pre-tick state.
    ///
    /// Pure function of `i`, the immutable `agents` slice, and the immutable
    /// spatial `hash`. It writes nothing shared, and iterates neighbours in the
    /// hash's fixed scan order, so calling it for each `i` serially or in
    /// parallel (each writing its own `deltas[i]`) yields bit-identical results.
    #[inline]
    fn steer(
        agents: &[CrowdAgent],
        hash: &SpatialHash,
        avoidance_weight: f32,
        separation_distance: f32,
        i: usize,
        neighbour_scratch: &mut Vec<usize>,
    ) -> Vec3 {
        let agent = &agents[i];

        let effective_target = if !agent.path.is_empty() && agent.path_index < agent.path.len() {
            let wp = agent.path[agent.path_index];
            Vec3::new(wp[0], wp[1], wp[2])
        } else {
            agent.target
        };

        let to_target = effective_target - agent.position;
        let dist_to_target = to_target.length();
        let desired = if dist_to_target > 0.01 {
            to_target / dist_to_target * agent.speed
        } else {
            Vec3::ZERO
        };

        let mut avoidance = Vec3::ZERO;
        hash.neighbours_into(agent.position, separation_distance, neighbour_scratch);
        for &j in neighbour_scratch.iter() {
            if j == i {
                continue;
            }
            let other = &agents[j];
            let diff = agent.position - other.position;
            let dist = diff.length();
            if dist < separation_distance && dist > 1e-4 {
                let strength =
                    avoidance_weight * (separation_distance - dist) / separation_distance;
                avoidance += (diff / dist) * strength;
            }
        }

        (desired + avoidance).clamp_length_max(agent.speed * 1.5)
    }

    /// Advance the simulation by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        self.tick_impl(dt, None);
    }

    /// Advance one tick, FORCING the serial steering path regardless of the
    /// agent count. Test-only hook used to prove the parallel path produces
    /// byte-identical state to the serial reference on the same input.
    #[doc(hidden)]
    pub fn tick_force_serial(&mut self, dt: f32) {
        self.tick_impl(dt, Some(false));
    }

    /// Advance one tick, FORCING the parallel steering path. Test-only hook.
    #[doc(hidden)]
    pub fn tick_force_parallel(&mut self, dt: f32) {
        self.tick_impl(dt, Some(true));
    }

    /// One tick. `force` overrides the auto parallel/serial selection
    /// (`Some(true)` = parallel, `Some(false)` = serial, `None` = by threshold).
    fn tick_impl(&mut self, dt: f32, force: Option<bool>) {
        let n = self.agents.len();
        if n == 0 {
            return;
        }

        // Reuse the spatial hash across ticks: clear keeps the allocated bucket
        // Vecs, and reinsertion is in agent-index order, so each bucket's
        // contents are index-ascending — identical to a freshly built hash.
        self.hash.clear();
        for (i, agent) in self.agents.iter().enumerate() {
            self.hash.insert(i, agent.position);
        }

        self.deltas.clear();
        self.deltas.resize(n, Vec3::ZERO);

        let agents = &self.agents;
        let hash = &self.hash;
        let avoidance_weight = self.avoidance_weight;
        let separation_distance = self.separation_distance;

        let parallel = force.unwrap_or(n >= PARALLEL_THRESHOLD);
        if parallel {
            // Determinism-safe parallelism: each closure writes ONLY its own
            // `deltas[i]` and reads only the immutable pre-tick `agents`/`hash`.
            // The result is independent of task scheduling, so it is bit-identical
            // to the serial path (each agent's internal avoidance sum keeps its
            // fixed neighbour-scan order). No cross-task float reduction occurs.
            self.deltas
                .par_iter_mut()
                .enumerate()
                .for_each_init(Vec::new, |scratch, (i, out)| {
                    *out = Self::steer(
                        agents,
                        hash,
                        avoidance_weight,
                        separation_distance,
                        i,
                        scratch,
                    );
                });
        } else {
            let mut scratch: Vec<usize> = Vec::new();
            for (i, out) in self.deltas.iter_mut().enumerate() {
                *out = Self::steer(
                    agents,
                    hash,
                    avoidance_weight,
                    separation_distance,
                    i,
                    &mut scratch,
                );
            }
        }

        for (agent, vel) in self.agents.iter_mut().zip(self.deltas.iter()) {
            agent.velocity = *vel;
            agent.position += *vel * dt;

            const WAYPOINT_ARRIVAL_DIST: f32 = 0.4;
            if !agent.path.is_empty() && agent.path_index < agent.path.len() {
                let wp = agent.path[agent.path_index];
                let wp_pos = Vec3::new(wp[0], wp[1], wp[2]);
                if (agent.position - wp_pos).length() < WAYPOINT_ARRIVAL_DIST {
                    agent.path_index += 1;
                    if agent.path_index < agent.path.len() {
                        let next = agent.path[agent.path_index];
                        agent.target = Vec3::new(next[0], next[1], next[2]);
                    }
                }
            }
        }
    }
}

impl CrowdAgent {
    pub fn set_navmesh_destination(&mut self, dest: Vec3, navmesh: &vox_core::navmesh::NavMesh) {
        self.path.clear();
        self.path_index = 0;
        let start_pos = [self.position.x, self.position.y, self.position.z];
        let goal_pos = [dest.x, dest.y, dest.z];
        let Some(start_id) = navmesh.nearest_node(start_pos) else {
            self.target = dest;
            return;
        };
        let Some(goal_id) = navmesh.nearest_node(goal_pos) else {
            self.target = dest;
            return;
        };
        if let Some(waypoints) = navmesh.find_path(start_id, goal_id) {
            self.path = waypoints;
            if let Some(wp) = self.path.first() {
                self.target = Vec3::new(wp[0], wp[1], wp[2]);
            }
        } else {
            self.target = dest;
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
        assert!(
            end.x > start.x,
            "agent should move toward target: {start} -> {end}"
        );
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
            sim.add_agent(Vec3::new(x, 0.0, z), Vec3::new(50.0, 0.0, 50.0), 1.5);
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
