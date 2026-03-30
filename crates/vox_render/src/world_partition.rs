//! World partition system — 3D cell-based streaming for open worlds.
//! CellCoord is 3D (x, y, z) where y is vertical slab (0=surface, negative=underground, positive=sky).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;
use glam::{self, Vec3};
use vox_core::types::GaussianSplat;

// ---------------------------------------------------------------------------
// CellCoord
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CellCoord(pub i32, pub i32, pub i32);

impl CellCoord {
    pub fn from_world_pos(pos: Vec3, cell_size: f32) -> Self {
        CellCoord(
            (pos.x / cell_size).floor() as i32,
            (pos.y / cell_size).floor() as i32,
            (pos.z / cell_size).floor() as i32,
        )
    }

    pub fn center(&self, cell_size: f32) -> Vec3 {
        Vec3::new(
            (self.0 as f32 + 0.5) * cell_size,
            (self.1 as f32 + 0.5) * cell_size,
            (self.2 as f32 + 0.5) * cell_size,
        )
    }
}

// ---------------------------------------------------------------------------
// Aabb
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_coord(coord: CellCoord, cell_size: f32) -> Self {
        let min = Vec3::new(
            coord.0 as f32 * cell_size,
            coord.1 as f32 * cell_size,
            coord.2 as f32 * cell_size,
        );
        Self { min, max: min + Vec3::splat(cell_size) }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        let closest = center.clamp(self.min, self.max);
        (closest - center).length_squared() <= radius * radius
    }
}

// ---------------------------------------------------------------------------
// SplatCompressed — HLOD representation
// ---------------------------------------------------------------------------

/// Compressed splat for HLOD (always-resident, small representation).
/// Uses f16 spectral and reduced precision scale to save memory.
#[derive(Debug, Clone)]
pub struct SplatCompressed {
    pub position: [f32; 3],
    pub scale:    f32,       // uniform scale (HLOD splats are spherical)
    pub opacity:  u8,
    pub spectral: [u16; 16], // f16 bits
}

impl SplatCompressed {
    pub fn from_splat(splat: &GaussianSplat) -> Self {
        Self {
            position: splat.position(),
            scale: (splat.scale_u() + splat.scale_v() + splat.scale_w()) / 3.0,
            opacity: splat.opacity(),
            spectral: *splat.spectral(),
        }
    }

    pub fn to_splat(&self) -> GaussianSplat {
        GaussianSplat::volume(
            self.position,
            [self.scale; 3],
            glam::Quat::IDENTITY,
            self.opacity,
            self.spectral,
        )
    }
}

// ---------------------------------------------------------------------------
// SpectralProbe
// ---------------------------------------------------------------------------

/// Spectral GI probe — 6 faces × 8 bands, radiance cache entry.
#[derive(Debug, Clone, Default)]
pub struct SpectralProbe {
    pub world_pos: Vec3,
    pub radiance: [[f32; 8]; 6],
}

// ---------------------------------------------------------------------------
// CellLoadState
// ---------------------------------------------------------------------------

pub enum CellLoadState {
    Unloaded,
    /// Async load in progress.
    /// In production, replace with tokio::task::JoinHandle<Result<LoadedCellData, CellLoadError>>.
    Loading { queued_frame: u64 },
    Loaded,
    Evicting,
}

// ---------------------------------------------------------------------------
// WorldCell
// ---------------------------------------------------------------------------

pub struct WorldCell {
    pub coord: CellCoord,
    pub bounds: Aabb,
    pub splat_tile_path: PathBuf,
    pub hlod_splats: Vec<SplatCompressed>,
    pub gi_probes: Vec<SpectralProbe>,
    pub load_state: CellLoadState,
    pub splat_count: u32,
    pub last_accessed: Instant,
}

impl WorldCell {
    pub fn new(coord: CellCoord, cell_size: f32, splat_tile_path: PathBuf) -> Self {
        Self {
            coord,
            bounds: Aabb::from_coord(coord, cell_size),
            splat_tile_path,
            hlod_splats: Vec::new(),
            gi_probes: Vec::new(),
            load_state: CellLoadState::Unloaded,
            splat_count: 0,
            last_accessed: Instant::now(),
        }
    }

    pub fn is_loaded(&self) -> bool {
        matches!(self.load_state, CellLoadState::Loaded)
    }

    pub fn memory_estimate_mb(&self) -> f32 {
        // Each full splat ~80 bytes; each compressed HLOD splat ~32 bytes
        let full_mb  = self.splat_count as f32 * 80.0 / (1024.0 * 1024.0);
        let hlod_mb  = self.hlod_splats.len() as f32 * 32.0 / (1024.0 * 1024.0);
        full_mb + hlod_mb
    }
}

// ---------------------------------------------------------------------------
// LoadRadius / SplatBudget
// ---------------------------------------------------------------------------

pub struct LoadRadius {
    pub inner_r: f32,   // full splats loaded inside this radius
    pub outer_r: f32,   // unload outside this radius
    pub hlod_r:  f32,   // HLOD available from inner_r to hlod_r
}

impl Default for LoadRadius {
    fn default() -> Self {
        Self { inner_r: 256.0, outer_r: 512.0, hlod_r: 400.0 }
    }
}

pub struct SplatBudget {
    pub max_gpu_mb: u32,
}

impl Default for SplatBudget {
    fn default() -> Self {
        Self { max_gpu_mb: 2048 }
    }
}

// ---------------------------------------------------------------------------
// WorldPartition
// ---------------------------------------------------------------------------

pub struct WorldPartition {
    pub cell_size: f32,
    pub cells: HashMap<CellCoord, WorldCell>,
    pub loaded_cells: HashSet<CellCoord>,
    pub load_radius: LoadRadius,
    pub splat_budget: SplatBudget,
    current_frame: u64,
}

impl WorldPartition {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
            loaded_cells: HashSet::new(),
            load_radius: LoadRadius::default(),
            splat_budget: SplatBudget::default(),
            current_frame: 0,
        }
    }

    pub fn register_cell(&mut self, coord: CellCoord, splat_tile_path: PathBuf) {
        let cell = WorldCell::new(coord, self.cell_size, splat_tile_path);
        self.cells.insert(coord, cell);
    }

    /// Compute cells that should be loading, HLOD-only, or unloaded based on camera position.
    /// Returns (to_load, to_hlod, to_unload).
    pub fn compute_streaming_state(
        &self,
        camera_pos: Vec3,
    ) -> (Vec<CellCoord>, Vec<CellCoord>, Vec<CellCoord>) {
        let mut to_load   = Vec::new();
        let mut to_hlod   = Vec::new();
        let mut to_unload = Vec::new();

        for (coord, cell) in &self.cells {
            let dist = (cell.bounds.center() - camera_pos).length();

            if dist < self.load_radius.inner_r {
                if !matches!(cell.load_state, CellLoadState::Loaded | CellLoadState::Loading { .. }) {
                    to_load.push(*coord);
                }
            } else if dist < self.load_radius.outer_r {
                if !matches!(cell.load_state, CellLoadState::Loaded | CellLoadState::Loading { .. }) {
                    to_hlod.push(*coord);
                }
            } else if matches!(cell.load_state, CellLoadState::Loaded) {
                to_unload.push(*coord);
            }
        }

        // Sort to_load by distance (closest first = highest priority)
        let cell_size = self.cell_size;
        to_load.sort_by(|a, b| {
            let da = (a.center(cell_size) - camera_pos).length();
            let db = (b.center(cell_size) - camera_pos).length();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        (to_load, to_hlod, to_unload)
    }

    /// Mark a cell as loading (queued for async IO).
    pub fn begin_load(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.load_state = CellLoadState::Loading { queued_frame: self.current_frame };
        }
    }

    /// Mark a cell as loaded (called when async IO completes).
    pub fn complete_load(&mut self, coord: CellCoord, splat_count: u32) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.load_state   = CellLoadState::Loaded;
            cell.splat_count  = splat_count;
            cell.last_accessed = Instant::now();
            self.loaded_cells.insert(coord);
        }
    }

    /// Begin eviction of a loaded cell.
    pub fn begin_evict(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.load_state = CellLoadState::Evicting;
            self.loaded_cells.remove(&coord);
        }
    }

    /// Complete eviction.
    pub fn complete_evict(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.load_state  = CellLoadState::Unloaded;
            cell.splat_count = 0;
        }
    }

    /// Total GPU memory used by loaded cells.
    pub fn gpu_splat_mb(&self) -> f32 {
        self.cells
            .values()
            .filter(|c| matches!(c.load_state, CellLoadState::Loaded))
            .map(|c| c.memory_estimate_mb())
            .sum()
    }

    /// Find the best eviction candidate: loaded, outside outer_r, LRU.
    pub fn find_eviction_candidate(&self, camera_pos: Vec3) -> Option<CellCoord> {
        self.cells
            .iter()
            .filter(|(_, c)| {
                matches!(c.load_state, CellLoadState::Loaded)
                    && (c.bounds.center() - camera_pos).length() >= self.load_radius.outer_r
            })
            .min_by_key(|(_, c)| c.last_accessed)
            .map(|(coord, _)| *coord)
    }

    /// Advance frame counter.
    pub fn tick(&mut self) {
        self.current_frame += 1;
    }

    /// Number of currently loaded cells.
    pub fn loaded_count(&self) -> usize {
        self.loaded_cells.len()
    }

    /// Cell count registered (regardless of load state).
    pub fn registered_count(&self) -> usize {
        self.cells.len()
    }
}

// ---------------------------------------------------------------------------
// CellEventHandler trait
// ---------------------------------------------------------------------------

/// Trait for reacting to cell load/unload events (e.g. script hooks, NPC spawning).
pub trait CellEventHandler: Send + Sync {
    fn on_cell_loaded(&mut self, coord: CellCoord);
    fn on_cell_unloaded(&mut self, coord: CellCoord);
}

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------

/// Streaming event sent from async loading tasks back to the main thread.
pub enum StreamEvent {
    CellReady { coord: CellCoord, splat_count: u32 },
    CellLoadFailed { coord: CellCoord, error: String },
}

// ---------------------------------------------------------------------------
// CellLoadRequest — priority queue entry using f32::to_bits() ordering
// ---------------------------------------------------------------------------

/// Priority entry for the cell load queue.
/// Uses f32::to_bits() so the value can be stored in a BinaryHeap without
/// pulling in the `ordered_float` crate.  Higher priority_bits = higher priority.
pub struct CellLoadRequest {
    pub coord:         CellCoord,
    pub priority_bits: u32, // f32::to_bits() of priority score
}

impl CellLoadRequest {
    pub fn new(coord: CellCoord, priority: f32) -> Self {
        Self { coord, priority_bits: priority.to_bits() }
    }

    pub fn priority(&self) -> f32 {
        f32::from_bits(self.priority_bits)
    }
}

impl PartialEq for CellLoadRequest {
    fn eq(&self, o: &Self) -> bool { self.priority_bits == o.priority_bits }
}
impl Eq for CellLoadRequest {}
impl PartialOrd for CellLoadRequest {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for CellLoadRequest {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.priority_bits.cmp(&o.priority_bits) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    // --- CellCoord -----------------------------------------------------------

    #[test]
    fn cell_coord_from_world_pos() {
        let pos = Vec3::new(128.0, 0.0, 0.0);
        let coord = CellCoord::from_world_pos(pos, 128.0);
        assert_eq!(coord, CellCoord(1, 0, 0));
    }

    #[test]
    fn cell_coord_center() {
        let coord = CellCoord(0, 0, 0);
        let c = coord.center(128.0);
        assert!((c - Vec3::new(64.0, 64.0, 64.0)).length() < 1e-4);
    }

    // --- WorldPartition lifecycle --------------------------------------------

    #[test]
    fn world_partition_register_and_load() {
        let mut wp = WorldPartition::new(128.0);
        let coord  = CellCoord(0, 0, 0);
        wp.register_cell(coord, PathBuf::from("tile_0_0_0.vxm"));
        assert_eq!(wp.registered_count(), 1);
        assert_eq!(wp.loaded_count(), 0);

        wp.begin_load(coord);
        wp.complete_load(coord, 5000);
        assert_eq!(wp.loaded_count(), 1);
        assert!(wp.cells[&coord].is_loaded());
    }

    #[test]
    fn world_partition_eviction() {
        let mut wp = WorldPartition::new(128.0);
        let coord  = CellCoord(0, 0, 0);
        wp.register_cell(coord, PathBuf::from("tile_0_0_0.vxm"));
        wp.complete_load(coord, 1000);
        assert_eq!(wp.loaded_count(), 1);

        wp.begin_evict(coord);
        assert_eq!(wp.loaded_count(), 0);

        wp.complete_evict(coord);
        assert_eq!(wp.cells[&coord].splat_count, 0);
        assert!(matches!(wp.cells[&coord].load_state, CellLoadState::Unloaded));
    }

    // --- Streaming state -----------------------------------------------------

    #[test]
    fn world_partition_compute_streaming_nearby() {
        let mut wp = WorldPartition::new(128.0);
        let coord  = CellCoord(0, 0, 0);
        wp.register_cell(coord, PathBuf::from("tile.vxm"));

        // Camera at the cell's center — well within inner_r (256.0 default)
        let camera_pos = coord.center(128.0);
        let (to_load, _to_hlod, _to_unload) = wp.compute_streaming_state(camera_pos);
        assert!(to_load.contains(&coord), "nearby cell should be in to_load");
    }

    #[test]
    fn world_partition_compute_streaming_far() {
        let mut wp = WorldPartition::new(128.0);
        let coord  = CellCoord(0, 0, 0);
        wp.register_cell(coord, PathBuf::from("tile.vxm"));
        // Mark as already loaded
        wp.complete_load(coord, 500);

        // Camera very far away — beyond outer_r (512.0 default)
        let camera_pos = Vec3::new(1000.0, 0.0, 0.0);
        let (_to_load, _to_hlod, to_unload) = wp.compute_streaming_state(camera_pos);
        assert!(to_unload.contains(&coord), "far loaded cell should be in to_unload");
    }

    // --- Aabb ----------------------------------------------------------------

    #[test]
    fn aabb_intersects_sphere() {
        // Cell at origin (0..128 on each axis)
        let aabb = Aabb::from_coord(CellCoord(0, 0, 0), 128.0);
        // Sphere centered at (100, 0, 0) with radius 200: closest point on aabb is (100,0,0)
        // distance = 0, so definitely intersects
        assert!(aabb.intersects_sphere(Vec3::new(100.0, 0.0, 0.0), 200.0));
        // Sphere centered very far away, small radius — should NOT intersect
        assert!(!aabb.intersects_sphere(Vec3::new(1000.0, 0.0, 0.0), 10.0));
    }

    // --- SplatCompressed -----------------------------------------------------

    #[test]
    fn splat_compressed_roundtrip() {
        let original = GaussianSplat::volume(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            glam::Quat::IDENTITY,
            200,
            [0u16; 16],
        );
        let compressed = SplatCompressed::from_splat(&original);
        let recovered  = compressed.to_splat();

        assert_eq!(recovered.position(), original.position());
        assert_eq!(recovered.opacity(),  original.opacity());
    }

    // --- GPU memory ----------------------------------------------------------

    #[test]
    fn gpu_splat_mb_accumulates() {
        let mut wp    = WorldPartition::new(128.0);
        let coord     = CellCoord(0, 0, 0);
        wp.register_cell(coord, PathBuf::from("tile.vxm"));
        wp.complete_load(coord, 10_000);
        assert!(wp.gpu_splat_mb() > 0.0, "gpu_splat_mb should be positive after loading");
    }

    // --- CellLoadRequest / BinaryHeap ----------------------------------------

    #[test]
    fn cell_load_request_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(CellLoadRequest::new(CellCoord(0, 0, 0), 1.0));
        heap.push(CellLoadRequest::new(CellCoord(1, 0, 0), 5.0));
        heap.push(CellLoadRequest::new(CellCoord(2, 0, 0), 3.0));

        // Max-heap: highest priority_bits first
        let top = heap.pop().unwrap();
        assert_eq!(top.coord, CellCoord(1, 0, 0));
    }
}
