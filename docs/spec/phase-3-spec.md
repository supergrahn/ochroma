# Phase 3 — Scale

**Goal:** A living, 100km city. LWC prevents floating-point jitter at world scale. VRAM
virtualization streams Gaussian data on demand from NVMe. Basic agents navigate the road
graph. Lyra enables real-world video capture into `.vxm` assets. The city builder is
playable as a product: prompt-driven generation, interactive plopping, real-time simulation
of pedestrians and traffic, and a 100km explorable world.

Builds on Phase 2's Neural Layout Interpreter, Proc-GS, compositional infill, and spectral
runtime shifts.

---

## Scope Boundary

### In scope (Phase 3)
- LWC: `f64` tile-anchor coordinates, `f32` local offsets within tiles
- Multi-tile SVO: 100km world divided into 1km × 1km tiles
- VRAM virtualization: NVMe-to-GPU Gaussian paging based on camera frustum
- Async tile streaming: `tokio` + io_uring predictive warm-up
- Basic agent simulation: pedestrians on road graph
- Neural traffic flow: LWR shockwave model for vehicles
- Lyra video capture pipeline (offline tool in `vox_tools`)
- Destruction masking: negative Gaussians for damage simulation
- Procedural zoning: growth model for AI-driven district expansion
- Dynamic time-of-day cycle (full 24-hour)
- Multi-scale rendering: satellite view (billboard splats) → street level (micro-Gaussians)
- Headless mode: run simulation without rendering (server-side)
- Telemetry dashboard: Rust web server for VRAM/FPS/agent-count monitoring

### Explicitly out of scope for Phase 3
- Multiplayer / CRDT sync (Phase 4)
- AR/VR output (Phase 4)
- Advanced economy (full supply chain, dynamic market) (Phase 4)
- Neural denoiser (Phase 4)
- Mod support / Wasm sandboxing (Phase 4)

---

## 1. Large World Coordinates (LWC)

At 100km scale, `f32` coordinates produce visible jitter (precision loss beyond ~10km from
origin). Phase 3 introduces a tile-anchor coordinate system.

```
World coordinate = tile_anchor (f64, absolute) + local_offset (f32, relative to anchor)

tile_anchor: (i32, i32) tile index × 1000.0 metres
  - stored as f64 for full precision at any world position
  - example: tile (47, 83) → anchor = (47000.0, 83000.0) metres

local_offset: (f32, f32, f32) position within the 1km tile
  - always within [-500m, +500m] range
  - f32 precision is sufficient within this range (sub-mm accuracy)
```

The CUDA rasterizer receives only `local_offset` — the camera's tile anchor is subtracted
CPU-side before GPU submission. No jitter at any world scale.

All Phase 0–2 `Vec3` positions are local offsets. The only change is adding the
`tile_anchor` to `SplatInstance`.

---

## 2. Multi-Tile SVO and Streaming

### World Structure

```
100km × 100km world = 100 × 100 = 10,000 tiles
Each tile = 1km × 1km × SVO
Active tiles (in VRAM): camera frustum + 1-tile buffer = ~9–25 tiles
Warm tiles (decompressing): next predicted frustum = ~9 tiles
Cold tiles (on NVMe): everything else
```

### VRAM Virtualization

```
Camera moves → frustum recalculated
    ↓
Tile visibility set updated
    ↓
New tiles predicted (based on camera velocity vector, 5s lookahead)
    ↓
io_uring async read: NVMe → system RAM (zstd decompress)
    ↓
CUDA async copy: system RAM → VRAM
    ↓
Tile registered in active SVO
    ↓
Old tiles outside extended frustum evicted from VRAM
```

Tile load budget: ≤ 200ms per tile from NVMe to VRAM (non-blocking, overlapped with
render). The predictive warm-up ensures tiles are ready before the camera reaches them.

### Tile File Format

Each tile is a `.vxt` (Vox Tile) file:

```
[Header: 32 bytes]
  magic:        [u8; 4]  = b"VXTT"
  version:      u16      = 1
  tile_x:       i32
  tile_z:       i32
  instance_count: u32
  _pad:         [u8; 14]

[Instance array: instance_count × 64 bytes, zstd-compressed]
  asset_uuid:   [u8; 16]
  position:     [f32; 3]   (local offset within tile)
  rotation:     [f32; 4]   (quaternion)
  scale:        f32
  wear_level:   f32
  spectral_shift: [f16; 8]
  instance_id:  u32
  _pad:         [u8; 4]
```

---

## 3. Multi-Scale Rendering

Seamless zoom from satellite view to street level without loading screens.

```
Altitude > 2000m:   Billboard splats (single oriented quad per building)
Altitude 200–2000m: Reduced-density LOD (LOD 2, ~10% of full splat count)
Altitude 50–200m:   Standard LOD (LOD 1, ~40% of full splat count)
Altitude < 50m:     Full detail (LOD 0, 100% splat count)
```

Billboard splats are pre-baked per asset — a single large Gaussian approximating the
asset's silhouette and spectral average. Transitions between LODs are opacity-crossfaded
over 2 frames to prevent popping.

---

## 4. Agent Simulation

### Pedestrian Agents

Basic pedestrians navigate the road graph generated in Phase 2.

```rust
struct Agent {
    position: Vec3,           // local offset in current tile
    tile: (i32, i32),         // current tile anchor
    velocity: Vec3,
    destination: WorldCoord,
    path: Vec<RoadEdge>,      // pre-computed path on road graph
    asset_uuid: Uuid,         // which pedestrian .vxm
    animation_phase: f32,     // walk cycle position 0.0–1.0
}
```

Agent count: up to 10,000 active agents per 4km² zone. Agents outside the active zone are
simulated at low frequency (1Hz) using aggregate flow tensors instead of individual paths.

### Neural Traffic Flow (LWR Model)

Vehicle traffic uses the Lighthill-Whitham-Richards (LWR) shockwave model — a continuous
density-flow PDE solved on the road graph. No individual vehicle agents at city scale.

```
ρ_t + (ρ × v(ρ))_x = 0

where:
  ρ = traffic density (vehicles/km)
  v(ρ) = Greenshields velocity: v_max × (1 - ρ/ρ_max)
```

Solved per road segment at 10Hz. Vehicles rendered as instanced `.vxm` assets at density
implied by ρ — not simulated individually at distance.

---

## 5. Lyra Video Capture Pipeline

Offline tooling in `vox_tools`. Converts a short phone video of a real object or building
into a `.vxm` asset.

```
vox_tools lyra-capture \
  --input ./footage/my_building.mp4 \
  --output ./library/buildings/captured_building_01.vxm \
  --spectral-map ./maps/building_material_zones.toml
```

Internal pipeline (Lyra, arXiv:2509.19296, NVIDIA ICLR 2026):
1. Extract frames from video
2. Video diffusion model infers geometry from monocular sequence
3. Self-distillation: RGB decoder supervises 3DGS decoder
4. 3DGS output covers full object including unseen sides
5. Spectral albedo reconstruction: strip real-world lighting → assign SPDs
6. Pack to `.vxm` v0.2 with entity_id tagging

Minimum video requirements: 5+ seconds, object visible from at least 2 sides, consistent
lighting (avoid harsh direct sun).

---

## 6. Destruction Masking

Negative Gaussians subtract opacity from the scene at their position. Used for damage,
demolition, and destruction effects.

```rust
struct DestructionMask {
    instance_id: u32,
    negative_splats: Vec<GaussianSplat>,  // opacity values are negative
    progression: f32,                      // 0.0 = intact, 1.0 = fully destroyed
}
```

The CUDA rasterizer processes negative Gaussians in the same depth-sorted pass. A
Gaussian with negative opacity reduces the accumulated alpha of Gaussians behind it,
creating holes in the surface. No geometry is deleted — the mask is applied at render time.

---

## 7. Procedural Zoning and Growth

The city grows procedurally when the player zones land. The growth model decides what
Proc-GS rules to apply per zone based on adjacency, land value, and district type.

```
Player zones a block as "Residential"
    ↓
Growth model selects rule variant based on:
  - Adjacent zone types (Commercial → higher density residential)
  - Land value tensor (derived from proximity to parks, transport)
  - Era parameter (set by player or auto-derived from surrounding style)
    ↓
Proc-GS assembles buildings into plots
    ↓
Over game-time: buildings gain wear, vegetation grows, style drifts
```

Land value is a 2D tensor over the world, updated when new assets are plopped or zones
change.

---

## 8. Headless Mode

Run the city simulation without rendering — for server-side multiplayer state or offline
batch generation.

```
vox_app --headless --tile-range 0,0,99,99 --sim-speed 10x
```

In headless mode:
- No CUDA rasterizer, no swapchain, no window
- ECS simulation systems run at full speed
- Agent simulation, weather, wear, and zoning all update
- Tile state serialised to `.vxt` files periodically
- HTTP API exposed for querying city state (telemetry, agent counts, zone data)

---

## 9. Telemetry Dashboard

Rust-based web server (`axum`) embedded in `vox_app`. Accessible from any browser on
the local network while the engine runs.

Metrics exposed:
- VRAM usage (total, per-tile breakdown)
- Active / warm / cold tile counts
- Splat count (rendered, culled, streamed)
- Frame time breakdown (sort, rasterize, shadow, UI)
- Agent count (active, aggregate)
- Async I/O queue depth

---

## Performance Budget (Phase 3)

| Metric | Target |
|---|---|
| GPU | RTX 4080 (16GB VRAM) |
| Resolution | 2560×1440 |
| World size | 100km × 100km |
| Active tiles | Up to 25 in VRAM simultaneously |
| Splat count (visible) | ≤ 50,000,000 |
| Frame time | ≤ 6.9ms (144fps) at street level |
| Tile stream time | ≤ 200ms NVMe → VRAM |
| Agent count (active) | ≤ 10,000 |

---

## Phase 3 Exit Criteria

- [ ] Camera can travel from 0,0 to 50km,50km without jitter or loading screens
- [ ] Tile streaming loads new tiles before camera reaches them (predictive warm-up working)
- [ ] Satellite view → street view zoom produces seamless LOD transition
- [ ] 10,000 pedestrian agents navigate road graph without frame time regression
- [ ] LWR traffic flow model produces visible congestion propagation on busy roads
- [ ] Lyra pipeline produces a clean `.vxm` from a 10-second phone video
- [ ] Destruction mask creates visible hole in a building without geometry deletion
- [ ] Procedural zoning generates a coherent residential block from a zone placement
- [ ] Headless mode runs simulation at 10× speed for 60 minutes without crash
- [ ] Telemetry dashboard shows live VRAM and frame breakdown in browser

---

## Phase 4 Preview

Once Phase 3 exits:
- CRDT-based multiplayer: 1,000+ players building the same city
- Neural denoiser for spectral noise at high Gaussian density
- Wasm sandboxing for mod support
- AR/VR output (Apple Vision Pro, Quest)
- Advanced economy: full supply chain, dynamic market, agent social graphs
- Neural Undo buffer: rewind city state to any previous point

---

## Mapping to Master Requirements List

| Phase 3 item | Master list ref |
|---|---|
| LWC | #2 |
| Multi-tile SVO | #4, #5 |
| VRAM virtualization | #8 |
| Async tile streaming | #9 |
| Predictive warm-up | #132 |
| Multi-scale rendering | #148 |
| Pedestrian agents | #31, #112 |
| LWR traffic flow | #116 |
| Lyra capture | #42 (extended) |
| Destruction masking | #39 |
| Procedural zoning | #36 |
| Headless mode | #49 |
| Telemetry dashboard | #140 |
| Dynamic time-of-day (full cycle) | #40 |
