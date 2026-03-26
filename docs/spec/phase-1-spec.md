# Phase 1 — Foundation

**Goal:** A functional city block. Multiple asset types (buildings, props, vegetation,
terrain) generated via Proc-GS rules, loaded from the asset library, instanced across a
1km tile, rendered spectrally at 60fps. The turnaround pipeline produces real `.vxm` assets
from Flux-generated images. The engine is playable in the sense that assets can be plopped
interactively via a minimal UI.

Builds directly on Phase 0's validated Rust↔CUDA↔Spectral foundation.

---

## Scope Boundary

### In scope (Phase 1)
- Bevy ECS integration for entity/component management
- SVO + DashMap spatial hash for 1km tile occupancy
- Full 8-band spectral pipeline with CIE observer integration
- Async asset loading via `tokio` + io_uring
- Proc-GS rule system: surface scattering + structured placement
- Spectral materials library (10 base materials, physically derived SPDs)
- Deterministic seed-driven variation across all asset types
- Turnaround pipeline tooling: Flux → 3DGS → spectral albedo reconstruction → `.vxm`
- Entity ID (16-bit) in `.vxm` format v0.2 + CUDA entity buffer
- Shadow Catcher generation on asset plop
- Asset library: on-disk structure, cataloguing, UUID indexing
- Basic terrain ground plane with tileable surface panels
- Minimal egui plop UI: select asset from library, place in world
- LOD system: two levels (full-detail ≤ 200m, reduced-density > 200m)

### Explicitly out of scope for Phase 1
- Neural Layout Interpreter (Phase 2)
- Compositional neural infill (Phase 2)
- LLM rule authoring loop (Phase 2)
- Generative model for micro-detail (Phase 2)
- Weather / wear runtime SPD shifts (Phase 2)
- Agent simulation, pathfinding, city economy
- Multi-tile streaming / LWC (Phase 3)
- Lyra video capture (Phase 3)
- Multiplayer, headless mode

---

## Crate Updates

```
ochroma/
├── crates/
│   ├── vox_core/       + ECS component definitions, SVO types, LWC types (reserved)
│   ├── vox_render/     + Full 8-band spectral kernel, entity buffer, LOD dispatch,
│   │                     Shadow Catcher mesh renderer
│   ├── vox_data/       + .vxm v0.2 format, Proc-GS emitter, spectral materials library,
│   │                     asset library index, async loader
│   ├── vox_app/        + egui plop UI, tile editor, asset browser
│   └── vox_tools/      NEW — turnaround pipeline CLI tool (offline, not in engine runtime)
```

`vox_sim` and `vox_nn` remain reserved but not created.

---

## `.vxm` Format v0.2

Adds `entity_id` to each splat. Replaces the `_pad` byte with a semantic zone tag.

```
[Header: 64 bytes]  — unchanged from v0.1

[Splat array: splat_count × 54 bytes each, zstd-compressed]
  position:      [f32; 3]   (x, y, z) relative to asset origin
  scale:         [f32; 3]   (half-axes of Gaussian ellipsoid)
  rotation:      [i16; 4]   (quantized quaternion)
  opacity:       u8         (0–255, linear)
  semantic_zone: u8         (index into asset's MaterialZone list)
  entity_id:     u16        (component identity — shared across a cluster)
  spectral:      [f16; 8]   (8 spectral band coefficients, 380–720nm)
```

**Total per-splat:** 54 bytes uncompressed.

`entity_id` is assigned at generation time. All Gaussians belonging to the same semantic
component (a window, a door, a chimney) share an `entity_id`. Instances in the world each
receive a unique world-level `instance_id` at runtime — the `.vxm` `entity_id` identifies
the component type, the runtime `instance_id` identifies the specific placed object.

---

## ECS Architecture (Bevy)

Phase 1 introduces Bevy ECS for managing asset instances in the world.

```rust
// Core components
struct SplatInstance {
    asset_uuid: Uuid,
    transform: Transform,         // world position, rotation, scale
    spectral_shift: SpectralShift, // per-instance SPD modifier
    opacity_mask: Option<u64>,    // bitmask of masked entity_ids
    instance_id: u32,             // unique world-level ID
}

struct SplatAsset {
    uuid: Uuid,
    splat_buffer: GpuBuffer,      // Gaussians in VRAM
    shadow_catcher: GpuMesh,      // convex hull mesh for shadow casting
    lod_levels: [LodDescriptor; 2],
}
```

Systems:
- `asset_stream_system` — async load/unload assets based on camera proximity
- `instance_cull_system` — frustum cull instances before GPU submission
- `lod_select_system` — choose LOD level per instance based on screen-space size
- `shadow_catcher_system` — submit shadow catcher meshes to shadow pass

---

## SVO + Spatial Hash

```
SVO (Sparse Voxel Octree):
  - 1km tile divided into 8m³ voxels
  - Each voxel stores a list of SplatInstance references
  - Used for: frustum culling, occlusion queries, collision proxy

DashMap spatial hash:
  - Key: (tile_x, tile_z, voxel_index)
  - Value: Vec<InstanceId>
  - Used for: fast neighbour lookup during plop snapping
```

Phase 1 covers a single 1km tile. Multi-tile SVO (Phase 3) extends this to 100km.

---

## Spectral Pipeline (Full 8-Band)

Replaces the Phase 0 4-band approximation.

```
8 bands: 380nm, 420nm, 460nm, 500nm, 540nm, 580nm, 620nm, 660nm

CIE 1931 observer integration:
  X = Σ (SPD[λ] × illuminant[λ] × x̄[λ]) × Δλ
  Y = Σ (SPD[λ] × illuminant[λ] × ȳ[λ]) × Δλ
  Z = Σ (SPD[λ] × illuminant[λ] × z̄[λ]) × Δλ

XYZ → linear sRGB → gamma correction → display
```

Illuminants supported in Phase 1:
- D65 (daylight, 6500K)
- D50 (warm daylight, 5000K)
- A (incandescent, 2856K)
- F11 (fluorescent)

Time-of-day spectral shift and weather effects deferred to Phase 2.

---

## Proc-GS Rule System

### Rule File Format

Rules are stored as `.splat_rule` files (TOML) in the asset library. LLM generates these;
the engine loads and executes them.

```toml
[rule]
asset_type = "House"
style = "victorian_terraced"

[geometry]
strategy = "structured_placement"
floor_count = { min = 2, max = 4 }
floor_height = { min = 3.2, max = 3.8 }
base_width = { min = 4.5, max = 6.0 }
depth = 12.0

[[materials]]
tag = "brick_facade"
spd = "brick_red"
density = 800.0
scale_range = [0.04, 0.08]
entity_id_zone = 1

[[materials]]
tag = "sash_window"
spd = "glass_clear"
density = 1200.0
scale_range = [0.02, 0.04]
entity_id_zone = 2
component_ref = "sash_window_canonical.vxm"

[[materials]]
tag = "slate_roof"
spd = "slate_grey"
density = 600.0
scale_range = [0.05, 0.10]
entity_id_zone = 3

[variation]
facade_color_shift = 0.15
wear_level = { min = 0.0, max = 0.4 }
bay_window_probability = 0.35
```

### Rule Examples Across Asset Types

**Tree:**
```toml
[rule]
asset_type = "Tree"
style = "oak_deciduous"

[geometry]
strategy = "growth_algorithm"
trunk_height = { min = 3.0, max = 8.0 }
canopy_radius = { min = 2.0, max = 5.0 }
branch_density = { min = 0.4, max = 0.8 }

[[materials]]
tag = "trunk"
spd = "bark_rough"
density = 400.0
entity_id_zone = 1

[[materials]]
tag = "canopy"
spd = "vegetation_leaf"
density = 2000.0
entity_id_zone = 2

[variation]
seasonal_state = "summer"
wear_level = { min = 0.0, max = 0.2 }
```

**Prop (bench):**
```toml
[rule]
asset_type = "Prop"
style = "bench_victorian"

[geometry]
strategy = "component_assembly"
length = { min = 1.6, max = 2.0 }

[[materials]]
tag = "frame"
spd = "metal_cast_iron"
density = 600.0
entity_id_zone = 1

[[materials]]
tag = "slats"
spd = "wood_painted_green"
density = 800.0
entity_id_zone = 2

[variation]
wear_level = { min = 0.1, max = 0.6 }
paint_color_shift = 0.2
```

---

## Turnaround Pipeline (vox_tools CLI)

Offline tooling — not part of the engine runtime. Runs as a separate process to produce
`.vxm` assets from Flux-generated images.

```
vox_tools turnaround \
  --views ./views/bench_front.png ./views/bench_left.png ... \
  --output ./library/props/bench_victorian_01.vxm \
  --material-map ./maps/bench_material_zones.toml
```

Steps executed internally:
1. Run 3DGS reconstruction on provided view images
2. Extract per-Gaussian spectral albedo (strip baked lighting)
3. Map Gaussians to material zones from `--material-map`
4. Assign SPD curves from the spectral materials library
5. Assign `entity_id` per zone
6. Pack and compress to `.vxm` v0.2

The material map is a TOML file defining which parts of the asset map to which spectral
material. The LLM generates this map from the same reference images.

---

## Asset Library Structure

```
library/
├── INDEX.toml                    # UUID → file path index
├── buildings/
│   ├── house_victorian_terraced_01.vxm
│   ├── house_victorian_terraced_02.vxm
│   └── ...
├── props/
│   ├── bench_victorian_01.vxm
│   ├── lamp_post_gas_era.vxm
│   └── ...
├── vegetation/
│   ├── oak_summer.vxm
│   └── ...
├── terrain/
│   ├── cobblestone_panel.vxm
│   ├── pavement_victorian.vxm
│   └── ...
├── characters/
│   └── ...
└── components/                   # sub-assets used by Proc-GS rules
    ├── sash_window_canonical.vxm
    ├── brick_panel_london_stock.vxm
    └── ...
```

`INDEX.toml` stores metadata per asset:
```toml
[[asset]]
uuid = "550e8400-e29b-41d4-a716-446655440000"
path = "buildings/house_victorian_terraced_01.vxm"
type = "Building"
style = "victorian_terraced"
tags = ["victorian", "residential", "london", "brick"]
seed = 0
rule = "rules/house_victorian_terraced.splat_rule"
pipeline = "ProcGS"
created = "2026-03-16"
```

---

## Shadow Catchers

Generated automatically when any asset is plopped. The engine computes a convex hull from
the asset's Gaussian positions and stores it as a low-poly mesh in VRAM alongside the
Gaussian buffer. Shadow Catchers are:

- Never rendered to the colour buffer
- Submitted to the shadow depth pass each frame
- Regenerated if the asset instance is scaled non-uniformly

---

## Plop UI (egui)

Minimal editor for Phase 1. No simulation, no zoning, no economy.

- Asset browser: search library by tag or type
- Preview pane: renders selected asset in isolation
- Place mode: click terrain to plop selected asset
- Select mode: click placed asset to select, shows entity_id breakdown
- Transform gizmo: move, rotate placed instance
- Spectral shift panel: adjust per-instance wear and colour shift

---

## Performance Budget (Phase 1)

| Metric | Target |
|---|---|
| GPU | RTX 3070 (8GB VRAM) |
| Resolution | 1920×1080 |
| Tile size | 1km × 1km |
| Splat count | ≤ 5,000,000 (mixed assets) |
| Frame time | ≤ 16.7ms (60fps) |
| VRAM usage | ≤ 4GB |
| Asset load time | ≤ 500ms per .vxm from NVMe (async, non-blocking) |

---

## Phase 1 Exit Criteria

- [ ] Bevy ECS manages 10,000+ instances without frame time regression
- [ ] SVO correctly culls instances outside the camera frustum
- [ ] Full 8-band spectral pipeline produces visually correct output under 4 illuminants
- [ ] Proc-GS emits deterministic buildings, trees, and props from `.splat_rule` files
- [ ] Turnaround pipeline produces a clean `.vxm` from a Flux turnaround image set
- [ ] Entity ID buffer correctly identifies clicked components in the plop UI
- [ ] Shadow Catchers produce correct shadow shapes on terrain
- [ ] Asset library INDEX.toml correctly catalogues and retrieves all assets by UUID and tag
- [ ] 5M splats on a 1km tile at 1080p hits ≥ 60fps on target GPU
- [ ] Async asset loading does not stall the render thread

---

## Phase 2 Preview

Once Phase 1 exits:
- Neural Layout Interpreter: LLM → Latent Scene Graph → auto-populate tile
- Compositional neural infill + Latent Refinement Pass for unique details
- LLM rule authoring loop (forward, reverse, edit modes)
- Weather / wear runtime SPD shifts
- Expand tile to multiple 1km blocks

---

## Mapping to Master Requirements List

| Phase 1 item | Master list ref |
|---|---|
| Bevy ECS | #3 |
| SVO | #5 |
| DashMap spatial hash | #4 |
| Async I/O (tokio) | #9 |
| Full spectral pipeline | #11 |
| CUDA kernels (updated) | #12 |
| Spectral materials library | #17 |
| .vxm v0.2 + entity ID | #21, #22, #30, #44 |
| Proc-GS rule system | #121 |
| Turnaround pipeline | #42 |
| Spectral albedo reconstruction | #208 |
| Asset library structure | #30 |
| Shadow Catchers | #33 (partial) |
| LOD system | #23 |
| egui plop UI | #41 |
