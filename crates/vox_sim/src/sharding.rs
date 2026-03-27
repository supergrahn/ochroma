use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a simulation shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u32);

/// A tile coordinate in the simulation world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

/// Message types exchanged between shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardMessage {
    EntityMigration(MigrationRecord),
    TickSync { shard_id: ShardId, tick: u64 },
    StateQuery { entity_id: u64 },
    StateResponse { entity_id: u64, state: Vec<u8> },
}

/// Record of an entity migration between shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRecord {
    pub entity_id: u64,
    pub from_shard: ShardId,
    pub to_shard: ShardId,
    pub timestamp: u64,
    pub serialised_state: Vec<u8>,
}

/// An independently-simulatable shard of the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationShard {
    pub id: ShardId,
    tiles: HashSet<TileCoord>,
    entities: HashSet<u64>,
    tick: u64,
}

impl SimulationShard {
    pub fn new(id: ShardId, tiles: HashSet<TileCoord>) -> Self {
        Self {
            id,
            tiles,
            entities: HashSet::new(),
            tick: 0,
        }
    }

    /// Advance local simulation by one tick.
    pub fn tick(&mut self) {
        self.tick += 1;
    }

    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn add_entity(&mut self, id: u64) {
        self.entities.insert(id);
    }

    pub fn remove_entity(&mut self, id: u64) -> bool {
        self.entities.remove(&id)
    }

    pub fn contains_entity(&self, id: u64) -> bool {
        self.entities.contains(&id)
    }

    pub fn contains_tile(&self, coord: &TileCoord) -> bool {
        self.tiles.contains(coord)
    }

    pub fn tiles(&self) -> &HashSet<TileCoord> {
        &self.tiles
    }
}

/// Manages multiple simulation shards with load balancing and entity migration.
#[derive(Debug)]
pub struct ShardManager {
    shards: HashMap<ShardId, SimulationShard>,
    next_shard_id: u32,
    global_tick: u64,
}

impl ShardManager {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
            next_shard_id: 0,
            global_tick: 0,
        }
    }

    /// Create a new shard owning the given tiles.
    pub fn create_shard(&mut self, tiles: HashSet<TileCoord>) -> ShardId {
        let id = ShardId(self.next_shard_id);
        self.next_shard_id += 1;
        self.shards.insert(id, SimulationShard::new(id, tiles));
        id
    }

    /// Assign an entity to a specific shard.
    pub fn assign_entity(&mut self, entity_id: u64, shard_id: ShardId) -> bool {
        if let Some(shard) = self.shards.get_mut(&shard_id) {
            shard.add_entity(entity_id);
            true
        } else {
            false
        }
    }

    /// Migrate an entity from one shard to another, returning a migration record.
    pub fn migrate_entity(
        &mut self,
        entity_id: u64,
        from_shard: ShardId,
        to_shard: ShardId,
    ) -> Option<MigrationRecord> {
        // Verify both shards exist and entity is in source
        if !self.shards.contains_key(&from_shard) || !self.shards.contains_key(&to_shard) {
            return None;
        }
        if !self.shards[&from_shard].contains_entity(entity_id) {
            return None;
        }

        // Serialise a placeholder state (in a real engine this would be the entity component data)
        let serialised_state = entity_id.to_le_bytes().to_vec();

        self.shards.get_mut(&from_shard).unwrap().remove_entity(entity_id);
        self.shards.get_mut(&to_shard).unwrap().add_entity(entity_id);

        let record = MigrationRecord {
            entity_id,
            from_shard,
            to_shard,
            timestamp: self.global_tick,
            serialised_state,
        };

        Some(record)
    }

    /// Look up which shard owns a given tile coordinate.
    pub fn shard_for_tile(&self, coord: TileCoord) -> Option<ShardId> {
        for shard in self.shards.values() {
            if shard.contains_tile(&coord) {
                return Some(shard.id);
            }
        }
        None
    }

    /// Rebalance shards: split those with >10k entities, merge those with <100.
    pub fn rebalance(&mut self) -> Vec<MigrationRecord> {
        let mut records = Vec::new();

        // Phase 1: split large shards (>10_000 entities)
        let large_shards: Vec<ShardId> = self
            .shards
            .values()
            .filter(|s| s.entity_count() > 10_000)
            .map(|s| s.id)
            .collect();

        for shard_id in large_shards {
            let shard = self.shards.get(&shard_id).unwrap();
            let entities: Vec<u64> = shard.entities.iter().copied().collect();
            let tiles: Vec<TileCoord> = shard.tiles.iter().copied().collect();
            let half = entities.len() / 2;
            let tile_half = tiles.len() / 2;

            // Create new shard with half the tiles
            let new_tiles: HashSet<TileCoord> = tiles[tile_half..].iter().copied().collect();
            let new_id = self.create_shard(new_tiles);

            // Migrate second half of entities to the new shard
            for &eid in &entities[half..] {
                self.shards.get_mut(&shard_id).unwrap().remove_entity(eid);
                self.shards.get_mut(&new_id).unwrap().add_entity(eid);
                records.push(MigrationRecord {
                    entity_id: eid,
                    from_shard: shard_id,
                    to_shard: new_id,
                    timestamp: self.global_tick,
                    serialised_state: eid.to_le_bytes().to_vec(),
                });
            }
        }

        // Phase 2: merge small shards (<100 entities)
        loop {
            let small_shards: Vec<ShardId> = self
                .shards
                .values()
                .filter(|s| s.entity_count() < 100)
                .map(|s| s.id)
                .collect();

            if small_shards.len() < 2 {
                break;
            }

            let a = small_shards[0];
            let b = small_shards[1];

            let entities_b: Vec<u64> = self.shards[&b].entities.iter().copied().collect();
            let tiles_b: HashSet<TileCoord> = self.shards[&b].tiles.clone();

            // Move all entities from b to a
            for eid in &entities_b {
                records.push(MigrationRecord {
                    entity_id: *eid,
                    from_shard: b,
                    to_shard: a,
                    timestamp: self.global_tick,
                    serialised_state: eid.to_le_bytes().to_vec(),
                });
                self.shards.get_mut(&a).unwrap().add_entity(*eid);
            }

            // Transfer tiles from b to a
            for tile in tiles_b {
                self.shards.get_mut(&a).unwrap().tiles.insert(tile);
            }

            self.shards.remove(&b);
        }

        records
    }

    /// Total number of entities across all shards.
    pub fn total_entities(&self) -> usize {
        self.shards.values().map(|s| s.entity_count()).sum()
    }

    /// Number of active shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Get a reference to a shard by ID.
    pub fn shard(&self, id: ShardId) -> Option<&SimulationShard> {
        self.shards.get(&id)
    }

    /// Tick the global clock.
    pub fn tick(&mut self) {
        self.global_tick += 1;
        for shard in self.shards.values_mut() {
            shard.tick();
        }
    }
}

impl Default for ShardManager {
    fn default() -> Self {
        Self::new()
    }
}
