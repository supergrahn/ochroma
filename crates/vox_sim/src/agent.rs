use glam::Vec3;
use std::collections::HashMap;
use uuid::Uuid;
use vox_core::lwc::WorldCoord;

#[derive(Debug, Clone)]
pub struct Agent {
    pub id: Uuid,
    pub position: WorldCoord,
    pub velocity: Vec3,
    pub destination: Option<WorldCoord>,
    pub speed: f32,
}

impl Agent {
    pub fn new(position: WorldCoord, speed: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            position,
            velocity: Vec3::ZERO,
            destination: None,
            speed,
        }
    }
}

pub struct AgentManager {
    agents: HashMap<Uuid, Agent>,
}

impl AgentManager {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn spawn(&mut self, position: WorldCoord, speed: f32) -> Uuid {
        let agent = Agent::new(position, speed);
        let id = agent.id;
        self.agents.insert(id, agent);
        id
    }

    pub fn get(&self, id: Uuid) -> Option<&Agent> {
        self.agents.get(&id)
    }

    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut Agent> {
        self.agents.get_mut(&id)
    }

    pub fn count(&self) -> usize {
        self.agents.len()
    }

    /// Advance simulation by `dt` seconds.
    /// Agents move toward their destination at their speed.
    /// On arrival the destination is cleared.
    pub fn tick(&mut self, dt: f32) {
        for agent in self.agents.values_mut() {
            let Some(dest) = agent.destination else { continue };

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
