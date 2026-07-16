use glam::Vec3;
use std::collections::BTreeMap;
use vox_core::lwc::WorldCoord;

#[derive(Debug, Clone)]
pub struct Agent {
    pub id: u32,
    pub position: WorldCoord,
    pub velocity: Vec3,
    pub destination: Option<WorldCoord>,
    pub speed: f32,
}

impl Agent {
    pub fn new(id: u32, position: WorldCoord, speed: f32) -> Self {
        Self {
            id,
            position,
            velocity: Vec3::ZERO,
            destination: None,
            speed,
        }
    }
}

/// Spatial agents keyed by a monotonic `u32` id in a `BTreeMap`, so both storage
/// and iteration order are DETERMINISTIC (id-ordered). This is a prerequisite for
/// folding agent state into the replay hash: previously the keys were `Uuid::new_v4()`
/// (OS entropy) in a `HashMap`, so iteration order was non-reproducible across
/// processes — tolerated only because movement is order-independent and the key was
/// never hashed. Keying by a monotonic id removes that latent determinism hazard.
pub struct AgentManager {
    agents: BTreeMap<u32, Agent>,
    next_id: u32,
}

impl AgentManager {
    pub fn new() -> Self {
        Self {
            agents: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Spawn an agent and return its stable monotonic id. Ids are never reused, so a
    /// spawn/despawn churn cannot collide with a live agent.
    pub fn spawn(&mut self, position: WorldCoord, speed: f32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.agents.insert(id, Agent::new(id, position, speed));
        id
    }

    pub fn get(&self, id: u32) -> Option<&Agent> {
        self.agents.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Agent> {
        self.agents.get_mut(&id)
    }

    pub fn count(&self) -> usize {
        self.agents.len()
    }

    /// Remove an agent by id. Returns the removed agent if it existed.
    pub fn remove(&mut self, id: u32) -> Option<Agent> {
        self.agents.remove(&id)
    }

    /// Iterate over all agents in deterministic id order.
    pub fn iter(&self) -> impl Iterator<Item = &Agent> {
        self.agents.values()
    }

    /// Advance simulation by `dt` seconds.
    /// Agents move toward their destination at their speed.
    /// On arrival the destination is cleared.
    pub fn tick(&mut self, dt: f32) {
        for agent in self.agents.values_mut() {
            let Some(dest) = agent.destination else {
                continue;
            };

            // Work in a common reference frame: use the agent's own tile
            let agent_local = agent.position.local;
            let dest_local = dest.local_relative_to(agent.position.tile);

            let diff = dest_local - agent_local;
            let dist = diff.length();

            let step = agent.speed * dt;
            if dist <= step {
                // Arrived — snap to destination and clear it
                agent.position = dest;
                agent.velocity = Vec3::ZERO;
                agent.destination = None;
            } else {
                let dir = diff / dist;
                agent.velocity = dir * agent.speed;
                let new_local = agent_local + dir * step;
                // Reconstruct absolute position and re-wrap into tile coords
                let (ax, az) = agent.position.tile.anchor();
                let abs_x = ax + new_local.x as f64;
                let abs_y = new_local.y as f64;
                let abs_z = az + new_local.z as f64;
                agent.position = WorldCoord::from_absolute(abs_x, abs_y, abs_z);
            }
        }
    }
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}
