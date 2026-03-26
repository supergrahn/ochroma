# Phase 2 — Intelligence

**Goal:** A full city district generated from a single text prompt. The Neural Layout
Interpreter plans the layout, Proc-GS assembles the skeleton, compositional neural infill
adds unique details, and runtime spectral shifts handle weather and wear. A Victorian
street with 50 buildings, props, vegetation, and characters can be generated in one session
without manual asset placement.

Builds on Phase 1's asset library, Proc-GS rule system, and spectral pipeline.

---

## Scope Boundary

### In scope (Phase 2)
- Neural Layout Interpreter: LLM → Latent Scene Graph → SVO placement
- Compositional neural infill + Latent Refinement Pass
- LLM rule authoring loop (forward, reverse, edit modes)
- Generative model for micro-detail augmentation on Proc-GS scaffolds
- Weather runtime SPD shifts (rain, overcast, fog, snow)
- Wear runtime SPD shifts (age parameter per instance)
- Time-of-day illuminant shifting (D65 → sunset → night)
- Multi-block tile: expand from 1km to 4km × 4km (4 tiles)
- Procedural street layout (road graph generation from prompt)
- Auto-population: given a street layout, fill with appropriate assets
- Style consistency enforcement across a district

### Explicitly out of scope for Phase 2
- Lyra video capture (Phase 3)
- LWC / 100km scale (Phase 3)
- Agent simulation, pathfinding (Phase 3)
- City economy, zoning (Phase 3)
- Multiplayer

---

## New Crates

```
vox_nn/     NEW — neural model inference runtime
              - local model loading (GGUF / safetensors)
              - async inference job queue
              - Layout Interpreter LLM client
              - micro-detail diffusion model runner
```

---

## 1. Neural Layout Interpreter

The "Architect Brain." Converts a text prompt into a structured `SceneGraph` that the
Proc-GS assembler consumes. Does not generate pixels or Gaussians — only spatial structure.

### Input / Output

```
Input:  "A narrow Victorian street, slightly run-down, one corner shop"

Output: SceneGraph {
    street: StreetLayout {
        width: 6.5,
        length: 120.0,
        orientation: 47.0deg,
        surface: "cobblestone_panel",
    },
    slots: [
        BuildingSlot { position: (0, 0, 0),   rule: "house_victorian_terraced", seed: 42,  wear: 0.6 },
        BuildingSlot { position: (6, 0, 0),   rule: "house_victorian_terraced", seed: 7,   wear: 0.4 },
        BuildingSlot { position: (12, 0, 0),  rule: "house_victorian_corner_shop", seed: 1, wear: 0.5 },
        ...
    ],
    props: [
        PropSlot { position: (3, 0, 2),  asset: "lamp_post_gas_era",    rotation: 0.0 },
        PropSlot { position: (9, 0, 2),  asset: "bench_victorian_01",   rotation: 0.0 },
    ],
    vegetation: [
        VegetationSlot { position: (15, 0, 3), rule: "oak_summer", seed: 99 },
    ],
    atmosphere: AtmosphereState {
        weather: Weather::LightRain,
        time_of_day: 19.5,      // 7:30pm
        season: Season::Autumn,
    },
}
```

### Spatial Constraint Enforcement

The LLM tends to make spatial errors (overlapping buildings, assets floating above ground,
doors facing away from the street). The Layout Interpreter runs a constraint pass after
LLM output:

- All buildings face the street (orientation corrected to road normal)
- No asset bounding boxes overlap
- All assets snap to terrain height
- Building spacing obeys minimum setback rules per style
- Corner buildings use corner-specific rule variants

Research basis: GALA3D (arXiv:2402.07207) and DreamScape (arXiv:2404.09227) both
demonstrate LLM layout + correction loop for coherent 3DGS scene generation.

---

## 2. Compositional Neural Infill

Adds unique organic detail on top of Proc-GS scaffolds. The Proc-GS skeleton is structurally
correct but visually repetitive. Infill makes each instance feel unique.

### What It Adds

- Unique weathering patterns (specific crack locations, moss distribution)
- One-off details (a specific poster on the wall, a broken drainpipe)
- Items flagged as "unique" in the Scene Graph ("one has a blue door")

### Workflow

```
Proc-GS scaffold .vxm
    ↓
Identify junction points (locations where unique detail is requested)
    ↓
Submit infill jobs to vox_nn async queue:
  "weathered blue door, victorian style, worn paint"
    ↓
Diffusion model generates Gaussian cluster for each junction
    ↓
Stitch clusters into scaffold at junction coordinates
    ↓
Latent Refinement Pass: CUDA alpha-blend at boundaries
    ↓
Final .vxm with seamless procedural + neural surface
```

### Latent Refinement Pass

CUDA kernel running after infill stitching:

1. Find all Gaussians within 0.3m of a procedural/neural boundary
2. Procedural Gaussians in the zone: opacity *= `distance_to_boundary / 0.3`
3. Neural Gaussians in the zone: opacity *= `1.0 - (distance_to_boundary / 0.3)`
4. Net result: smooth opacity crossfade, no hard edge

Research basis: CompGS (arXiv:2410.20723), CG3D (arXiv:2311.17907).

---

## 3. LLM Rule Authoring Loop

Three modes, all operating on `.splat_rule` TOML files:

### Forward (knowledge → rule)
```
Prompt: "Victorian terraced house, London 1880s"
  → LLM outputs complete .splat_rule TOML
  → vox_tools validates rule (parameter ranges, SPD references)
  → Proc-GS emits 5 test variants
  → Render and review
  → Iterate if needed
  → Commit rule to library
```

### Reverse (asset → rule)
```
Hero .vxm asset rendered from 6–8 angles
  → Images sent to vision LLM
  → LLM extracts: proportions, material zones, variation ranges
  → Outputs .splat_rule TOML
  → Validated via test emission
  → Committed to library
```

### Edit (rule → variation)
```
Existing .splat_rule + instruction:
  "make a wealthier version of this"
  → LLM modifies: floor_height, bay_window_probability,
    SPD shift toward stone, reduced wear_level
  → New .splat_rule saved as variant
```

All three modes are accessible from a `vox_tools llm-rule` CLI command and from
the egui editor in Phase 2.

---

## 4. Runtime Spectral Shifts

Spectral properties change at runtime without reloading assets. All shifts are GPU-side
uniform updates to the CUDA rasterizer — zero CPU cost.

### Weather States

| Weather | SPD Effect |
|---|---|
| Clear | No shift (D65 baseline) |
| Overcast | Reduced irradiance, blue-shift (higher CCT) |
| Light rain | Wet material specular spike on outdoor surfaces |
| Heavy rain | Strong specular, reduced diffuse, fog scattering |
| Fog | Gaussian density field attenuates distant SPDs |
| Snow | Near-white spectral shift on horizontal surfaces |

Weather state is a global tensor updated per-frame. Each Gaussian's effective SPD is
`base_spd × weather_modifier[surface_type][weather_state]`.

### Wear / Age

Per-instance `wear_level` (0.0–1.0) modifies SPD curves at runtime:

```
effective_spd = lerp(
    material_library[tag].spd_new,
    material_library[tag].spd_worn,
    instance.wear_level
)
```

`spd_worn` is a physically derived version of the base SPD shifted toward:
- Darker, less saturated (most materials)
- More specular roughness (metals, painted surfaces)
- Biological growth (concrete, brick) — slight green shift at 520–560nm

### Time-of-Day Illuminant

```
dawn    →  D50 + low irradiance + warm horizon scatter
day     →  D65
sunset  →  D50 → A (2856K) + red-orange scatter
dusk    →  A + low irradiance + blue ambient
night   →  Artificial illuminants only (sodium, LED, gas lamp SPDs)
midnight → Moonlight (D65 × 0.001) + artificial
```

---

## 5. Street Layout Generation

The Layout Interpreter generates a road graph from a prompt before placing buildings.

```
"A Victorian residential district with a main high street and side roads"
    ↓
Road graph: nodes (intersections) + edges (road segments)
    ↓
Building plots generated along road edges
    ↓
Plot sizes determine which building rules apply (terraced vs. detached)
    ↓
Proc-GS assembles buildings into plots
    ↓
Props and vegetation auto-placed by density rules per road type
```

Road surface panels are Proc-GS tileables (cobblestone, tarmac, pavement). Road graph
stored in the SVO as navigable edges for Phase 3 agent pathfinding.

---

## Performance Budget (Phase 2)

| Metric | Target |
|---|---|
| GPU | RTX 3070 (8GB VRAM) |
| Resolution | 1920×1080 |
| Tile size | 4km × 4km (4 tiles) |
| Splat count | ≤ 20,000,000 |
| Frame time | ≤ 16.7ms (60fps) |
| VRAM usage | ≤ 6GB |
| Full district generation time | ≤ 60 seconds from prompt to rendered scene |

---

## Phase 2 Exit Criteria

- [ ] "A narrow Victorian street" prompt generates a coherent, populated street in ≤ 60s
- [ ] All buildings face the street (constraint enforcement working)
- [ ] No asset overlap in generated layouts
- [ ] Compositional infill produces a unique detail on at least one building per street
- [ ] Latent Refinement Pass produces no visible seam at procedural/neural boundaries
- [ ] Weather state change (clear → rain) visually updates all asset SPDs within one frame
- [ ] Wear parameter change on a selected instance updates SPD in real time
- [ ] Time-of-day cycle runs continuously without frame time regression
- [ ] LLM rule authoring loop produces a validated new rule in a single session
- [ ] 20M splats on 4km tile at 1080p hits ≥ 60fps

---

## Phase 3 Preview

Once Phase 2 exits:
- LWC (Large World Coordinates) for 100km scale
- Multi-tile streaming: VRAM virtualization, NVMe-to-GPU paging
- Basic agent simulation: pedestrians navigate road graph
- Lyra video capture pipeline
- City simulation fundamentals: zoning, land value, basic economy

---

## Mapping to Master Requirements List

| Phase 2 item | Master list ref |
|---|---|
| Neural Layout Interpreter | #36, #44 |
| Latent Scene Graph | #36 |
| Compositional infill | #193 |
| Latent Refinement Pass | #14 (partial) |
| LLM rule authoring | #43 (partial) |
| Weather SPD shifts | #35 |
| Wear runtime shifts | #126 |
| Time-of-day illuminant | #40 |
| Street layout generation | #36, #37 |
| Fog (Gaussian density field) | #105 |
| Multi-tile expansion | #4, #5 |
