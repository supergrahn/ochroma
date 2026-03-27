use std::collections::HashMap;
use uuid::Uuid;

/// State of a world instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldState {
    Starting,
    Running,
    Paused,
    Stopped,
}

/// A persistent world instance.
#[derive(Debug, Clone)]
pub struct WorldInstance {
    pub id: Uuid,
    pub name: String,
    pub owner: String,
    pub max_players: u32,
    pub current_players: Vec<String>,
    pub state: WorldState,
}

impl WorldInstance {
    pub fn player_count(&self) -> u32 {
        self.current_players.len() as u32
    }

    pub fn is_full(&self) -> bool {
        self.player_count() >= self.max_players
    }
}

/// A portal connecting two worlds.
#[derive(Debug, Clone)]
pub struct Portal {
    pub id: Uuid,
    pub source_world: Uuid,
    pub source_position: [f32; 3],
    pub destination_world: Uuid,
    pub destination_position: [f32; 3],
    pub label: String,
}

/// Statistics for a world instance.
#[derive(Debug, Clone)]
pub struct WorldStats {
    pub world_id: Uuid,
    pub player_count: u32,
    pub max_players: u32,
    pub state: WorldState,
    pub portal_count: usize,
}

/// Error type for world hosting operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldHostError {
    WorldNotFound,
    WorldFull,
    PlayerAlreadyInWorld,
    PlayerNotInWorld,
    NameAlreadyTaken,
    InvalidState(String),
}

impl std::fmt::Display for WorldHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorldNotFound => write!(f, "world not found"),
            Self::WorldFull => write!(f, "world is full"),
            Self::PlayerAlreadyInWorld => write!(f, "player already in world"),
            Self::PlayerNotInWorld => write!(f, "player not in world"),
            Self::NameAlreadyTaken => write!(f, "world name already taken"),
            Self::InvalidState(s) => write!(f, "invalid state: {s}"),
        }
    }
}

impl std::error::Error for WorldHostError {}

/// Manages multiple persistent world instances.
pub struct WorldHost {
    worlds: HashMap<Uuid, WorldInstance>,
    portals: Vec<Portal>,
}

impl WorldHost {
    pub fn new() -> Self {
        Self {
            worlds: HashMap::new(),
            portals: Vec::new(),
        }
    }

    /// Create a new world instance. Returns the world ID.
    pub fn create_world(
        &mut self,
        name: &str,
        owner: &str,
        max_players: u32,
    ) -> Result<Uuid, WorldHostError> {
        // Check for duplicate names
        if self.worlds.values().any(|w| w.name == name) {
            return Err(WorldHostError::NameAlreadyTaken);
        }

        let id = Uuid::new_v4();
        let world = WorldInstance {
            id,
            name: name.to_string(),
            owner: owner.to_string(),
            max_players,
            current_players: Vec::new(),
            state: WorldState::Starting,
        };
        self.worlds.insert(id, world);
        // Auto-transition to Running
        if let Some(w) = self.worlds.get_mut(&id) {
            w.state = WorldState::Running;
        }
        Ok(id)
    }

    /// A player joins a world.
    pub fn join_world(&mut self, world_id: Uuid, player: &str) -> Result<(), WorldHostError> {
        let world = self
            .worlds
            .get_mut(&world_id)
            .ok_or(WorldHostError::WorldNotFound)?;

        if world.state != WorldState::Running {
            return Err(WorldHostError::InvalidState(
                "world is not running".to_string(),
            ));
        }

        if world.is_full() {
            return Err(WorldHostError::WorldFull);
        }

        if world.current_players.contains(&player.to_string()) {
            return Err(WorldHostError::PlayerAlreadyInWorld);
        }

        world.current_players.push(player.to_string());
        Ok(())
    }

    /// A player leaves a world.
    pub fn leave_world(&mut self, world_id: Uuid, player: &str) -> Result<(), WorldHostError> {
        let world = self
            .worlds
            .get_mut(&world_id)
            .ok_or(WorldHostError::WorldNotFound)?;

        let pos = world
            .current_players
            .iter()
            .position(|p| p == player)
            .ok_or(WorldHostError::PlayerNotInWorld)?;

        world.current_players.remove(pos);
        Ok(())
    }

    /// List all worlds.
    pub fn list_worlds(&self) -> Vec<&WorldInstance> {
        self.worlds.values().collect()
    }

    /// Get a specific world by ID.
    pub fn get_world(&self, world_id: Uuid) -> Option<&WorldInstance> {
        self.worlds.get(&world_id)
    }

    /// Get statistics for a world.
    pub fn world_stats(&self, world_id: Uuid) -> Result<WorldStats, WorldHostError> {
        let world = self.worlds.get(&world_id).ok_or(WorldHostError::WorldNotFound)?;
        let portal_count = self
            .portals
            .iter()
            .filter(|p| p.source_world == world_id || p.destination_world == world_id)
            .count();

        Ok(WorldStats {
            world_id,
            player_count: world.player_count(),
            max_players: world.max_players,
            state: world.state.clone(),
            portal_count,
        })
    }

    /// Create a portal linking two worlds.
    pub fn create_portal(
        &mut self,
        source_world: Uuid,
        source_position: [f32; 3],
        destination_world: Uuid,
        destination_position: [f32; 3],
        label: &str,
    ) -> Result<Uuid, WorldHostError> {
        if !self.worlds.contains_key(&source_world) {
            return Err(WorldHostError::WorldNotFound);
        }
        if !self.worlds.contains_key(&destination_world) {
            return Err(WorldHostError::WorldNotFound);
        }

        let id = Uuid::new_v4();
        self.portals.push(Portal {
            id,
            source_world,
            source_position,
            destination_world,
            destination_position,
            label: label.to_string(),
        });
        Ok(id)
    }

    /// List all portals for a given world.
    pub fn portals_for_world(&self, world_id: Uuid) -> Vec<&Portal> {
        self.portals
            .iter()
            .filter(|p| p.source_world == world_id || p.destination_world == world_id)
            .collect()
    }

    /// Pause a running world.
    pub fn pause_world(&mut self, world_id: Uuid) -> Result<(), WorldHostError> {
        let world = self
            .worlds
            .get_mut(&world_id)
            .ok_or(WorldHostError::WorldNotFound)?;
        if world.state != WorldState::Running {
            return Err(WorldHostError::InvalidState(
                "can only pause a running world".to_string(),
            ));
        }
        world.state = WorldState::Paused;
        Ok(())
    }

    /// Stop a world.
    pub fn stop_world(&mut self, world_id: Uuid) -> Result<(), WorldHostError> {
        let world = self
            .worlds
            .get_mut(&world_id)
            .ok_or(WorldHostError::WorldNotFound)?;
        world.state = WorldState::Stopped;
        Ok(())
    }
}

impl Default for WorldHost {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_join() {
        let mut host = WorldHost::new();
        let id = host.create_world("TestWorld", "alice", 4).unwrap();
        host.join_world(id, "bob").unwrap();
        let world = host.get_world(id).unwrap();
        assert_eq!(world.player_count(), 1);
        assert_eq!(world.state, WorldState::Running);
    }

    #[test]
    fn test_capacity_limit() {
        let mut host = WorldHost::new();
        let id = host.create_world("Small", "owner", 2).unwrap();
        host.join_world(id, "p1").unwrap();
        host.join_world(id, "p2").unwrap();
        assert_eq!(host.join_world(id, "p3"), Err(WorldHostError::WorldFull));
    }
}
