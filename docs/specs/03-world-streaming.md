# Domain 3 — World Streaming & Open World

**Status:** Spec v1.0 — 2026-03-29
**Crate scope:** `vox_core`, `vox_data`, `vox_render`, `vox_terrain`, `vox_app`
**Dependencies:** TerrainVolume (SDF), NavMesh + A*, GaussianSplat pipeline, wgpu 24, tokio, rayon, snappy/zstd

---

## Goals

Ochroma must support arbitrarily large open worlds composed of spectral Gaussian Splat tiles. The engine must stream cells in and out based on camera proximity without hitching the render thread. Hierarchical LOD must keep GPU splat counts bounded. Procedural Content Generation must populate cells deterministically from biome and terrain data. The system must feel seamless to players: no pop-in, no framerate dips on cell transitions, no visible seams between biomes.

Specifically:
- Maintain 70 FPS at 4K during cell transitions on a mid-range GPU (RTX 4070 class).
- Support world extents of at least 64 km × 64 km with vertical layering for underground caves and sky structures.
- Bound GPU splat memory to a configurable `SplatBudget`; never exceed it.
- Allow scripts (Rhai) to react to cell load/unload events for NPC spawning and ambient audio triggering.
- HLOD must eliminate visual pop when transitioning from distant representation to full splats.

---

## 3.1 World Partition System

### Core Types

```rust
// vox_core::world_partition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord(pub i32, pub i32, pub i32);

pub struct WorldPartition {
    pub cell_size: f32,                                     // e.g. 128.0 metres
    pub cells: HashMap<CellCoord, WorldCell>,
    pub loaded_cells: HashSet<CellCoord>,
    pub load_radius: LoadRadius,
    pub splat_budget: SplatBudget,
    lru_tracker: LruCache<CellCoord, Instant>,              // for eviction ordering
}

pub struct LoadRadius {
    pub inner_r: f32,   // full splat loading radius
    pub outer_r: f32,   // HLOD-only radius; beyond = unloaded
    pub hlod_r: f32,    // HLOD representation available from inner_r to hlod_r
}

pub struct SplatBudget {
    pub max_gpu_mb: u32,
}

pub enum CellLoadState {
    Unloaded,
    Loading(tokio::task::JoinHandle<Result<LoadedCellData, CellLoadError>>),
    Loaded,
    Evicting,
}

pub struct WorldCell {
    pub coord: CellCoord,
    pub bounds: Aabb,
    pub splat_tile_path: PathBuf,
    pub entity_data_path: PathBuf,
    pub hlod_splats: Vec<SplatCompressed>,      // always resident; small
    pub gi_probes: Vec<SpectralProbe>,
    pub navmesh_chunk: Option<NavMesh>,
    pub load_state: CellLoadState,
    pub splat_count: u32,                        // known from cell metadata, before load
    pub last_accessed: Instant,
}
```

`CellCoord` is three-dimensional because the engine supports multi-level worlds: a cell at `(x, y, z)` where `y` represents vertical slab index. This allows caves (negative y slabs), surface (y = 0), and sky structures (positive y slabs) to be independent streaming units sharing the same partition machinery.

`Aabb` is `vox_core::math::Aabb { min: Vec3, max: Vec3 }` using `glam::Vec3`.

### Streaming Radius Logic

Each frame, the `WorldPartitionSystem` computes the set of cells that should be in each state:

```
desired_full   = { coord | dist(camera, cell_center) < inner_r }
desired_hlod   = { coord | inner_r <= dist < outer_r }
desired_unload = { coord | dist >= outer_r }
```

Cells in `desired_full` that are `Unloaded` or `HLOD`-only are queued for full load.
Cells in `desired_hlod` that are `Unloaded` get their HLOD data from the always-resident `hlod_splats` field immediately — no async required.
Cells in `desired_unload` that are `Loaded` are queued for eviction.

### Streaming Thread Architecture

A dedicated `tokio::spawn`ed task, `CellStreamingTask`, owns an `mpsc::Sender<StreamEvent>` channel back to the main thread. The main thread owns the `Receiver<StreamEvent>` and drains it each frame inside `WorldPartitionSystem::process_stream_events()`.

```rust
pub enum StreamEvent {
    CellReady { coord: CellCoord, data: Box<LoadedCellData> },
    CellLoadFailed { coord: CellCoord, error: CellLoadError },
}

pub struct LoadedCellData {
    pub splats: Vec<GaussianSplat>,
    pub entity_data: EntityData,
    pub navmesh_chunk: Option<NavMesh>,
}
```

The main thread never blocks on disk I/O. `LoadedCellData` is heap-allocated and sent through the channel as a `Box<_>` to keep the channel message small.

### Priority Queue

Cells awaiting load are sorted in a `BinaryHeap<CellLoadRequest>`. Priority score:

```
priority = (1.0 / camera_distance) * frustum_weight
```

Where `frustum_weight = 2.0` if the cell's `Aabb` intersects the camera frustum, `1.0` otherwise. This ensures that cells visible to the player load before cells behind the camera. The streaming task pulls from this queue FIFO at its configured parallelism ceiling (`max_concurrent_loads: usize`, default 4).

### Memory Budget Enforcement

After each `CellReady` event is processed and the GPU buffer allocated, `WorldPartitionSystem::check_budget()` evaluates total GPU splat memory:

```rust
fn check_budget(&mut self, pool: &mut SplatBufferPool) {
    while self.gpu_splat_mb() > self.splat_budget.max_gpu_mb {
        // find LRU loaded cell outside outer_r
        if let Some(coord) = self.find_eviction_candidate() {
            self.begin_evict(coord, pool);
        } else {
            break; // no evictable cells; accept overrun, log warning
        }
    }
}
```

Eviction candidates are `Loaded` cells with `dist(camera, cell_center) >= outer_r`, sorted by `last_accessed` ascending. The `LruCache<CellCoord, Instant>` is updated on every frame that a cell contributes visible splats.

### Cell Events

```rust
pub trait CellEventHandler {
    fn on_cell_loaded(&mut self, coord: CellCoord, world: &WorldPartition);
    fn on_cell_unloaded(&mut self, coord: CellCoord);
}
```

The Rhai scripting bridge (`vox_script`) implements `CellEventHandler`. Scripts register closures:

```rhai
on_cell_loaded(|coord| {
    if coord == cell(4, 0, 2) { spawn_npc("orc_patrol", coord); }
});
```

### Cell Authoring: WorldPartitionEditor

`WorldPartitionEditor` runs at bake time (not at runtime) in `vox_app`. It takes a scene's full splat list and partitions splats into cells by spatial hash: `cell_coord = floor(position / cell_size)` for each axis. Splats on cell boundaries are assigned to the cell containing their centroid.

Incremental bake: each cell tracks a `content_hash: u64` computed from the source asset hashes of all splat assemblies contributing to that cell. On re-bake, if the content hash matches the stored hash, the cell file is skipped. This makes iterative world editing fast even for large worlds.

---

## 3.2 Hierarchical LOD (HLOD)

### HLODBuilder

```rust
pub struct HLODBaker;

impl HLODBaker {
    pub fn bake_cell(
        cell_splats: &[GaussianSplat],
        levels: &[HLODSpec],
    ) -> Vec<HLODLevel> { ... }
}

pub struct HLODSpec {
    pub reduction_factor: u32,   // k = cell_splat_count / reduction_factor
}

pub struct HLODLevel {
    pub level: u32,
    pub splats: Vec<SplatCompressed>,
    pub coverage_error: f32,     // max screen-space error in pixels at reference distance
}
```

#### K-Means Clustering Algorithm

For level 0 HLOD (most aggressive reduction): `k = cell_splat_count / 64`.

1. Initialize `k` centroids using K-Means++ initialization: first centroid chosen uniformly at random; each subsequent centroid chosen with probability proportional to squared distance from the nearest existing centroid. This avoids degenerate clusterings.
2. Iterate until convergence (max 50 iterations):
   a. Assignment step: for each splat, assign to nearest centroid by Euclidean distance on `position`.
   b. Update step: recompute centroid position as mean of assigned splat positions.
3. For each cluster, produce one HLOD splat:
   - `position`: cluster centroid
   - `scale`: bounding sphere radius of cluster positions (max distance from centroid)
   - `spectral`: weighted average of member splat spectrals, weights = member `opacity` values cast to f32
   - `opacity`: `min(255, sum_of_member_opacities)` to preserve apparent density
   - `rotation`: identity quaternion (HLOD splats are axis-aligned)

`coverage_error` is computed as the maximum angular size (in screen pixels at 1080p, 60° FoV, 100m reference distance) of the displacement between an original splat and its HLOD representative.

#### LOD Selection

At runtime, `HlodSelector::select_for_cell(cell, camera_transform, viewport_size)` returns `LodSelection::Full` or `LodSelection::Hlod(level)`. Switch condition:

```
use full splats when: coverage_error_at_current_distance < 1.0 screen pixel
```

`coverage_error_at_current_distance = coverage_error * reference_distance / camera_distance`

#### HLOD Crossfade

`CrossfadeHLOD` manages a per-cell blend factor `t: f32` in `[0, 1]`. When a full cell load completes, `t` begins ramping from `1.0` (full HLOD) to `0.0` (full splats) over `0.5` seconds. During the ramp, both HLOD and full splats are submitted to the renderer with alpha modulated by `t` and `1.0 - t` respectively. At `t = 0.0`, HLOD splats are removed from the render list. This eliminates hard pop at the cost of one extra draw call for 0.5s per cell load.

---

## 3.3 Async Cell Loading Pipeline

### CellLoader

```rust
pub struct CellLoader {
    thread_pool: rayon::ThreadPool,  // separate from render threadpool
    active_loads: HashMap<CellCoord, AbortHandle>,
    tx: mpsc::Sender<StreamEvent>,
}
```

`CellLoader` uses a dedicated rayon `ThreadPool` (default 4 threads, configurable via `CellLoaderConfig::io_threads`) to avoid contending with rendering work on the shared rayon global pool.

Loading stages per cell:
1. **Disk read**: `std::fs::read(splat_tile_path)` — synchronous, on the I/O thread.
2. **Decompress**: detect format from header magic byte: `0xCA` = snappy, `0xCB` = zstd. Decompress to `Vec<u8>` containing packed `SplatCompressed` records.
3. **Expand**: `SplatCompressed::expand()` → `GaussianSplat`. `SplatCompressed` stores `f16` spectrals and `i16` rotation quaternion components. Expansion unpacks these to full `GaussianSplat` layout.
4. **Send**: wrap in `LoadedCellData`, send via `mpsc::Sender<StreamEvent>`.
5. **GPU upload** (main thread, on receipt): `SplatBufferPool::upload(coord, splats)`.

### GPU Buffer Pool

```rust
pub struct SplatBufferPool {
    free: Vec<wgpu::Buffer>,
    used: HashMap<CellCoord, wgpu::Buffer>,
    buffer_size_splats: u32,   // each buffer holds this many splats
}
```

At startup, `SplatBufferPool::new()` pre-allocates `N = max_gpu_mb * 1024 * 1024 / (bytes_per_splat * buffer_size_splats)` buffers using `device.create_buffer()` with `BufferUsages::VERTEX | BufferUsages::COPY_DST`. When a cell load completes, `pool.allocate(coord)` pops from `free`; when a cell is evicted, `pool.release(coord)` pushes back to `free`. Pre-allocation eliminates device-side allocation latency on cell transitions.

`bytes_per_splat = size_of::<GaussianSplat>()` = 3×4 + 3×4 + 4×2 + 1 + 8×2 = 12 + 12 + 8 + 1 + 16 = 49 bytes, padded to 52 bytes for alignment.

### Double-Buffering

The render thread reads `pool.used` each frame to collect active cell buffers. Cell buffer swaps (new allocation becoming active) happen only at the frame boundary, inside a `std::sync::Mutex<ActiveCellSet>` that is locked briefly once per frame. The `CellStreamingTask` writes to a staging `HashMap<CellCoord, wgpu::Buffer>` which is swapped into `ActiveCellSet` atomically.

### Streaming Telemetry

```rust
pub struct StreamingTelemetry {
    pub cells_loading: u32,
    pub cells_loaded: u32,
    pub cells_evicted: u32,
    pub bytes_loaded_per_sec: f64,
    pub gpu_splat_mb: f32,
}
```

Updated each frame. Written to a lock-free `AtomicU64`-backed ring buffer; read by the profiler overlay without blocking the streaming thread.

---

## 3.4 Procedural Content Generation (PCG) Graph

### Graph Types

```rust
pub struct PcgGraph {
    pub nodes: HashMap<NodeId, PcgNode>,
    pub edges: Vec<PcgEdge>,      // PcgEdge { from: NodeId, to: NodeId, slot: u8 }
}

pub type PointCloud = Vec<PcgPoint>;

pub struct PcgPoint {
    pub position: Vec3,
    pub normal: Vec3,
    pub spectral_sample: [f32; 8],  // sampled from terrain at this point
}

pub enum PcgNode {
    SampleTerrain       { resolution: f32 },
    FilterBySlope       { min_deg: f32, max_deg: f32 },
    FilterByHeight      { min_y: f32, max_y: f32 },
    FilterBySpectralBand { band: u8, min: f32, max: f32 },
    ScatterSplatAssembly { asset_path: PathBuf, density: f32, jitter: f32, align_to_normal: bool },
    AddNoise            { frequency: f32, amplitude: f32 },
    Merge,
    Split               { condition: SplitCondition },
    Debug               { label: String },
}
```

`FilterBySpectralBand` is an Ochroma-unique node: it filters placement points by the spectral signature of the underlying terrain material at that point. For example, a node configured with `band = 2, min = 0.4` retains only positions where the terrain's near-infrared band (band 2) exceeds 0.4 — naturally selecting vegetated ground over bare rock without requiring a separate vegetation mask texture.

### Evaluation

`PcgGraph::evaluate(cell_bounds: Aabb, terrain: &TerrainVolume) -> Vec<SplatInstance>`:

1. Topological sort of nodes (Kahn's algorithm on `edges`).
2. Execute nodes in topological order. Each node is a pure function: `fn execute(inputs: &[PointCloud], node: &PcgNode) -> PointCloud`.
3. `SampleTerrain` generates a uniform grid of points at `resolution` metre spacing within `cell_bounds`, samples SDF and normals from `TerrainVolume::sample_normal_at()`, samples spectral from `TerrainVolume::sample_spectral_at()`.
4. `ScatterSplatAssembly` loads the referenced assembly asset, instantiates one copy per input point with position jittered by `jitter` metres (uniform random), optionally rotated to align `up` axis to the point's normal.
5. Terminal nodes produce `Vec<SplatInstance>` which are baked into the cell's splat tile.

### Bake Granularity and Caching

PCG bakes at cell granularity. The bake result for a cell is stored alongside the splat tile. The content hash includes the PCG graph definition hash and the terrain data hash for the cell's bounds. If neither changes, the PCG output is not recomputed.

### Live Preview

In editor mode (`vox_app`), `PcgPreviewRenderer` runs the graph at reduced resolution (`resolution *= 4.0` in `SampleTerrain`) and renders the result as translucent ghost splats (opacity multiplied by 0.3) in the editor viewport. Runs on a background thread; result pushed to render thread via `mpsc`. Refreshes 250ms after any graph node edit.

---

## 3.5 Biome System

### Types

```rust
pub struct BiomeMap {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<BiomeId>,      // world_x maps to [0..width], world_z to [0..height]
    pub world_scale: f32,          // metres per texel
}

pub type BiomeId = u8;

pub struct BiomeDef {
    pub id: BiomeId,
    pub name: String,
    pub terrain_material_palette: Vec<MaterialBlendRule>,
    pub foliage_sets: Vec<FoliageSet>,
    pub ambient_spectral: [f32; 8],
}

pub struct MaterialBlendRule {
    pub material_index: u8,
    pub height_range: RangeF32,
    pub slope_range: RangeF32,
    pub weight_curve: AnimationCurve,  // vox_core curve type; input = blend factor, output = weight
}
```

### Biome Blending

`BiomeSampler::sample(world_x: f32, world_z: f32) -> [BiomeWeight; 3]` returns the three dominant biomes at a world position and their blend weights. The sampler:

1. Samples the `BiomeMap` with bilinear interpolation.
2. Finds the three highest-weight biomes in the 2×2 neighbourhood.
3. Normalises weights to sum to 1.0.

Terrain material weights at a given point are computed by evaluating each contributing biome's `MaterialBlendRule` set at the current height and slope, then blending by the biome weights.

### BiomeMap Generation

`BiomeMapGenerator::generate(world_width: f32, world_depth: f32, seed: u64) -> BiomeMap`:

1. Generate a Worley noise field at low frequency (cell spacing ≈ 2 km) using seed. Each Worley cell is assigned a `BiomeId` from a weighted distribution specified in `BiomeDistribution`.
2. Add medium-frequency Perlin noise (octaves = 3, persistence = 0.5, frequency = 0.001) to the Worley distance field to break up straight biome boundaries.
3. Threshold and quantise to produce the `texels` array.

### SpectralBiome

Each `BiomeDef::ambient_spectral` is an 8-band spectral irradiance value in W/m²/nm for the dominant ambient light in that biome. This value is used to initialise `SpectralProbe::ambient` for GI probes within cells belonging to that biome. A forest biome with high near-infrared ambient will correctly colour all objects within it with a subtle green-shift via the GI system, without requiring per-object tint parameters. A desert biome's high red-band ambient causes warm tinting of sand and rocks.

---

## 3.6 Spline Road / River System

### Types

```rust
pub struct SplineRoad {
    pub control_points: Vec<Vec3>,
    pub width: f32,
    pub carve_depth: f32,
    pub material_path: PathBuf,
}

pub struct SplineRiver {
    pub control_points: Vec<Vec3>,
    pub width: f32,
    pub carve_depth: f32,
    pub upstream_end: Vec3,
}
```

Splines are evaluated as Catmull-Rom curves. Given control points `P0..Pn`, a point at parameter `t ∈ [i, i+1]` is evaluated using the four surrounding control points `P_{i-1}, P_i, P_{i+1}, P_{i+2}` with tension `0.5`.

### Road Carving

`SplineRoad::carve(terrain: &mut TerrainVolume)`:

1. Sample the spline at `step = 0.5` metre intervals.
2. At each sample point, call `terrain.carve_sphere(center, radius)` where `radius = width / 2.0`. The sphere profile is elongated by calling `carve_sphere` at three offsets: `±normal * carve_depth * 0.5` and at the surface. This creates a flat-bottomed channel.
3. After carving, re-generate terrain SDF normals for the affected region via `TerrainVolume::recompute_normals_region(aabb)`.

### Road Surface Splats

`RoadSplatGenerator::generate(road: &SplineRoad, material: &SplatMaterial) -> Vec<GaussianSplat>`:

Sample the spline at `step = 0.25m`. At each sample, generate a row of splats across the road width: `ceil(width / 0.25)` splats per cross-section row, offset by `(-width/2 + i * 0.25)` along the road's local right vector. Each splat is positioned at the SDF surface below the road centre, with scale matching road texture density.

### River

`SplineRiver` uses the same carving approach as roads. After carving, `FluidSimSource { world_pos: upstream_end }` is registered in the physics/fluid system to emit water particles. The water surface is generated as a procedural splat strip using a water material with high opacity in near-infrared bands (simulating real water spectra: strong absorption above 750 nm).

### Editor Interaction

In the editor, `SplineEditor` manages control points as draggable handles. Left-click on empty terrain adds a control point at the cursor's terrain intersection. Drag moves the selected handle. Pressing `S` auto-smooths by adjusting tangents. The spline preview is drawn as a 3D curve gizmo. Road/river splats are regenerated live on handle drag using the reduced-resolution preview pipeline.

---

## File Map

```
vox_core/src/
  world_partition.rs        — WorldPartition, CellCoord, WorldCell, CellLoadState, LoadRadius, SplatBudget
  biome.rs                  — BiomeDef, BiomeMap, BiomeSampler, BiomeMapGenerator, MaterialBlendRule
  pcg.rs                    — PcgGraph, PcgNode, PcgEdge, PcgPoint, PointCloud

vox_data/src/
  cell_file.rs              — cell .vxc file format: header, SplatCompressed block, entity block, navmesh block
  cell_loader.rs            — CellLoader, LoadedCellData, StreamEvent, CellLoadError
  world_bake.rs             — WorldPartitionEditor, incremental bake, content hash
  hlod_baker.rs             — HLODBaker, HLODSpec, HLODLevel, k-means implementation
  pcg_bake.rs               — PCG bake pipeline, per-cell caching

vox_render/src/
  splat_buffer_pool.rs      — SplatBufferPool, pre-allocation, allocate/release
  streaming_telemetry.rs    — StreamingTelemetry, atomic ring buffer
  crossfade_hlod.rs         — CrossfadeHLOD, blend factor ramp

vox_terrain/src/
  spline.rs                 — SplineRoad, SplineRiver, CatmullRomSpline, road carve, splat generation
  pcg_executor.rs           — PcgGraph::evaluate, node implementations

vox_app/src/
  world_partition_system.rs — WorldPartitionSystem, per-frame update, budget check, event dispatch
  spline_editor.rs          — SplineEditor, control point handles
  pcg_preview.rs            — PcgPreviewRenderer, background thread, live refresh
```

---

## Milestones

### M1 — Basic Cell Streaming (3 days)
- `CellCoord`, `WorldCell`, `WorldPartition` types defined in `vox_core`.
- `.vxc` file format specified and implemented in `vox_data`.
- `CellLoader` async pipeline: disk read → decompress → expand → `StreamEvent`.
- `SplatBufferPool` pre-allocation; cell buffers integrated into `SpectraEwaTileRenderer`.
- `WorldPartitionSystem` per-frame update: load/evict based on `LoadRadius`.
- Streaming telemetry displayed in profiler overlay.
- **Acceptance:** 16 cells loaded/unloaded without framerate dip; GPU memory stays within `SplatBudget`.

### M2 — HLOD (2 days)
- `HLODBaker` K-means++ clustering; `HLODLevel` with `coverage_error`.
- HLOD splats serialised into `.vxc` metadata block (always resident).
- LOD selection integrated into render submission.
- `CrossfadeHLOD` 0.5s alpha blend on full-cell load.
- **Acceptance:** no visible pop on cell transition at any camera speed; splat count at outer radius ≤ 1.5% of full count.

### M3 — PCG Graph (3 days)
- All `PcgNode` variants implemented; `PcgGraph::evaluate` with topological sort.
- `FilterBySpectralBand` working against `TerrainVolume::sample_spectral_at()`.
- PCG bake integrated into `WorldPartitionEditor`; per-cell cache.
- Live preview in editor viewport.
- **Acceptance:** PCG-populated forest cell matches hand-authored reference within 15% splat density; incremental rebake skips unchanged cells.

### M4 — Biome System (2 days)
- `BiomeDef`, `BiomeMap`, Worley+Perlin generation.
- `BiomeSampler` 3-biome blending; terrain material weights derived.
- `SpectralBiome` ambient initialises GI probes.
- **Acceptance:** biome boundary renders without hard seam; forest/desert ambient spectral difference visible in rendered output.

### M5 — Spline Roads and Rivers (2 days)
- `SplineRoad` carving into `TerrainVolume` via Catmull-Rom sampling.
- Road surface splat generation from material asset.
- `SplineRiver` fluid source integration.
- Editor spline handle drag with live preview.
- **Acceptance:** 1 km road carved and surfaced without terrain Z-fighting; spline editable in real time at 60 FPS.

**Total estimated effort:** 12 engineering-days.

---

## Acceptance Criteria (System-Level)

1. A 4 km × 4 km world (1024 cells at 128m cell size) streams without exceeding `SplatBudget::max_gpu_mb` at any camera position.
2. Camera moving at 60 m/s across cell boundaries: zero frame budget spikes above 16ms.
3. HLOD transition: no visible luminance discontinuity measurable by frame diff (< 2% mean pixel delta).
4. PCG: foliage distribution is deterministic across separate bake runs with the same seed.
5. Cell load/unload events fire reliably; Rhai callbacks execute within the same frame as the event.
6. Incremental bake: editing one cell's source assets triggers rebake of only that cell (verified by bake log).
7. Biome blending: no single-texel biome boundary visible at 1m inspection distance.
8. Spline road: road surface splats have no gap or overlap at spline sample junctions.

---

## Effort Summary

| Area | Days |
|------|------|
| M1 Cell Streaming | 3 |
| M2 HLOD | 2 |
| M3 PCG Graph | 3 |
| M4 Biome System | 2 |
| M5 Spline Roads/Rivers | 2 |
| **Total** | **12** |
