use std::collections::{HashMap, VecDeque};

/// A virtual texture tile identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    pub x: u32,
    pub y: u32,
    pub mip_level: u32,
}

/// State of a virtual texture tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLoadState {
    NotLoaded,
    Loading,
    Loaded,
    Evicting,
}

/// LRU cache for virtual texture tiles.
pub struct TileCache {
    max_tiles: usize,
    tiles: HashMap<TileId, TileLoadState>,
    access_order: VecDeque<TileId>, // front = least recently used
}

impl TileCache {
    pub fn new(max_tiles: usize) -> Self {
        Self { max_tiles, tiles: HashMap::new(), access_order: VecDeque::new() }
    }

    /// Mark a tile as requested/accessed.
    pub fn touch(&mut self, id: TileId) {
        // Move to back (most recently used)
        self.access_order.retain(|t| t != &id);
        self.access_order.push_back(id);

        self.tiles.entry(id).or_insert(TileLoadState::Loading);
    }

    /// Mark a tile as fully loaded.
    pub fn mark_loaded(&mut self, id: TileId) {
        self.tiles.insert(id, TileLoadState::Loaded);
    }

    /// Evict least recently used tiles to stay within budget.
    pub fn evict_to_budget(&mut self) -> Vec<TileId> {
        let mut evicted = Vec::new();
        while self.tiles.len() > self.max_tiles {
            if let Some(lru) = self.access_order.pop_front() {
                self.tiles.remove(&lru);
                evicted.push(lru);
            } else {
                break;
            }
        }
        evicted
    }

    pub fn is_loaded(&self, id: TileId) -> bool {
        self.tiles.get(&id) == Some(&TileLoadState::Loaded)
    }

    pub fn tile_count(&self) -> usize { self.tiles.len() }
    pub fn loaded_count(&self) -> usize {
        self.tiles.values().filter(|s| **s == TileLoadState::Loaded).count()
    }
}
