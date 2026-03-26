# Asset Generation Pipeline

All `.vxm` assets are generated, never captured or converted. This document specifies
the pipelines, how they compose, the full prompt-to-render chain, and how variation is
managed sustainably at scale.

For the research backing of each technology, see `technology-foundations.md`.

---

## Full Engine Pipeline: Prompt to Rendered Frame

```
User prompt: "A rainy evening on a 19th-century London street"
    ↓
┌─────────────────────────────────────────────────────────────┐
│  1. NEURAL LAYOUT INTERPRETER (GALA3D / DreamScape pattern) │
│     LLM generates a Latent Scene Graph:                     │
│     - street length, width, orientation                     │
│     - building positions, spacing, facing direction         │
│     - style tags per building slot ("victorian_terraced")   │
│     - special elements ("puddles", "wet_road", "fog")       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  2. PROC-GS ASSEMBLER (Proc-GS, arXiv:2412.07660)          │
│     For each building slot in the Scene Graph:              │
│     - pull canonical Gaussian components from library       │
│       (brick_panel, sash_window, slate_roof, cornice...)    │
│     - assemble per SplatRule + seed                         │
│     - inject into SVO at scene graph coordinates            │
│     - tag all Gaussians with entity_id per component        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  3. COMPOSITIONAL NEURAL INFILL (CompGS / CG3D pattern)     │
│     For "unique" elements flagged in the Scene Graph:       │
│     - generate unique detail clusters via diffusion model   │
│       (weathered door, cracked chimney, specific graffiti)  │
│     - stitch neural clusters into Proc-GS frame             │
│     - run Latent Refinement Pass to blend boundaries        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  4. SPECTRAL RENDER (SpectralGaussians pattern)             │
│     CUDA kernels:                                           │
│     - depth radix sort (tile-based, 16×16px)                │
│     - spectral_rasterize: SPD × illuminant → RGB per tile   │
│     - shadow catcher pass: mesh-based shadow casting        │
│     - alpha-gaussian blend at procedural/neural boundaries  │
│     Output: 144Hz spectral frame at target resolution       │
└─────────────────────────────────────────────────────────────┘
```

---

## Asset Creation Pipelines

### Pipeline A: Turnaround Reconstruction (Hero Assets)

For complex assets where geometry is too specific for procedural rules — landmarks,
vehicles, characters.

```
LLM generates style prompt
    ↓
Flux 2 Pro: generate hero reference image
    ↓
Flux 2 Pro: generate N views from reference
  (front, front-left 45°, left, back-left 45°,
   back, back-right 45°, right, front-right 45°, top)
  — same pose, no drift, reference-anchored
    ↓
3DGS reconstruction: fit Gaussians to all views
    ↓
Spectral albedo reconstruction (#208):
  strip baked lighting → assign real SPDs from library
    ↓
.vxm asset with entity_id tagging
```

### Pipeline B: Procedural Generation (Proc-GS — Mass Fill)

For all architectural and natural assets. Infinite variation from finite rules.

```
LLM generates SplatRule definition
  — from prompt ("Victorian terraced house, London 1880s")
  — or from reverse analysis of a hero asset (images → LLM → SplatRule)
  — or by editing an existing rule ("make this poorer", "add 30 years of wear")
    ↓
Gaussian Emitter (Rust)
  - assembles canonical components per rule
  - places Gaussians along analytically defined surfaces
  - assigns physically defined SPD per material zone
  - tags all Gaussians with entity_id per semantic component
  - deterministic: same seed = same asset, always
    ↓
.vxm asset
```

### Pipeline C: Compositional Neural Infill (Phase 2+)

Adds unique detail on top of a Proc-GS scaffold. Never replaces structure.

```
Proc-GS scaffold .vxm + unique element prompts + seed
    ↓
Diffusion model generates unique detail clusters
  ("weathered blue door", "cracked chimney pot", "1960s repair patch")
    ↓
Stitched into scaffold at tagged junction points
    ↓
Latent Refinement Pass: CUDA alpha-blend at boundaries
    ↓
Refined .vxm with seamless procedural/neural surface
```

### Pipeline D: Video Capture (Lyra — Phase 3+)

For real-world assets captured by users with a phone.

```
User records short video of real object or building
    ↓
Lyra (NVIDIA, ICLR 2026, arXiv:2509.19296)
  - video diffusion model infers unseen geometry
  - self-distillation: RGB decoder supervises 3DGS decoder
  - outputs full Gaussian scene from monocular video
    ↓
Spectral albedo reconstruction:
  strip real-world baked lighting → assign SPDs from library
    ↓
.vxm asset
```

---

## Which Pipeline for Which Asset Type

The pipeline is fully asset-type agnostic. A `SplatRule` for a tree is structurally
identical to one for a building — the emitter does not know or care what it is generating.
Every asset type, regardless of pipeline, produces a `.vxm` file and enters the same
library.

| Asset Type | Primary Pipeline | Rule / Geometry Notes |
|---|---|---|
| Buildings — background filler | B (Proc-GS) | Walls, windows, roofs defined by rules |
| Buildings — hero, landmarks | A (Turnaround) | Too specific for rules |
| Terrain, roads, pavements | B (Proc-GS) | Surface scattering, tileable panels |
| Trees, shrubs | B (Proc-GS) | Parametric branching, leaf cluster SPD |
| Grass, ground cover | B (Proc-GS) | Density field, wind variation parameter |
| Vehicles, machinery | A (Turnaround) | Precise engineered geometry |
| Props — benches, bins, signs | A (Turnaround) | Small, specific, one-off |
| Props — tileable (cobblestones) | B (Proc-GS) | Regular, rule-definable |
| Characters, creatures | A (Turnaround) | Organic form, Flux handles well |
| Unique details (worn door, crack) | C (Infill) on B | Stitched into scaffold |
| Real-world user captures | D (Lyra) | Phone video → .vxm |

---

## LLM as Rule Author, Extractor, and Editor

The LLM participates at every stage of rule authoring. Three modes:

**Forward — knowledge to rule:**
```
"Victorian terraced house, London 1880s"
  ↓ LLM draws on absorbed architectural knowledge
  → SplatRule
```

**Reverse — asset to rule:**
```
Hero asset rendered from 6–8 angles
  ↓ LLM analyses images, extracts proportions, materials, structure
  → SplatRule that produces buildings in that style
```

**Edit — rule to variation:**
```
Existing SplatRule + instruction
  ↓ LLM adjusts parameters
  → Modified SplatRule
```

| Instruction | What the LLM changes |
|---|---|
| "make this neighbourhood poorer" | increase `wear_level`, add broken window probability, reduce floor heights |
| "make this the wealthy end" | raise floor heights, add bay windows, shift SPD toward stone |
| "post-war damage, 30 years later" | high `wear_level`, destruction masking on windows, soot SPD shift |
| "add a corner shop" | ground-floor material zone override, wider street-facing windows |
| "colder climate version" | swap vegetation for bare, increase roof pitch, add oxidised drainage |

---

## Proc-GS: Rule Anatomy

```rust
pub struct SplatRule {
    pub asset_type: AssetType,          // House, Office, Tree, Road, etc.
    pub geometry: GeometryRule,         // how to place Gaussians
    pub materials: Vec<MaterialZone>,   // which surfaces get which SPD
    pub variation: VariationParams,     // seed-driven parameter ranges
}

pub struct MaterialZone {
    pub tag: &'static str,             // "brick_facade", "glass_window", "roof_tile"
    pub spd: SpectralMaterial,         // physically defined reflectance curve
    pub density: f32,                  // Gaussians per m²
    pub scale_range: (f32, f32),       // min/max Gaussian size
    pub entity_id: u16,               // semantic identity for this zone
}

pub struct VariationParams {
    pub facade_color_shift: f32,       // ±spectral shift on base material
    pub window_count: (u32, u32),      // min/max windows per floor
    pub floor_count: (u32, u32),       // min/max floors
    pub wear_level: f32,               // 0.0 = new, 1.0 = heavily weathered
    pub era: BuildingEra,              // affects proportions and detail density
}
```

### ProceduralSplat Trait

The trait that allows the engine to treat rule-based and neural assets uniformly.
Note: `apply_neural_infill` submits an async job — it does not block.

```rust
pub trait ProceduralSplat {
    /// Generate the deterministic structural skeleton from rules + seed.
    fn generate_skeleton(&self, seed: u64) -> Vec<GaussianSplat>;

    /// Submit a neural infill job for unique detail elements.
    /// Non-blocking: returns a job handle. Result arrives via callback.
    fn request_neural_infill(&self, prompt: &str, tx: InfillSender) -> InfillJobHandle;

    /// Apply a completed neural infill cluster to this asset.
    fn apply_infill(&mut self, cluster: NeuralCluster, junction: JunctionPoint);
}
```

---

## Spectral Materials Library

Physically measured SPD values — not artist-tuned RGB.

| Material Tag | Description | Notes |
|---|---|---|
| `concrete_raw` | Unpainted concrete | ~0.3 reflectance, flat curve |
| `brick_red` | Red clay brick | Absorption peak ~450nm |
| `glass_clear` | Clear float glass | High transmission, low reflectance |
| `glass_tinted` | Tinted commercial glass | Shifted transmission curve |
| `metal_steel` | Bare steel | High reflectance, metallic |
| `metal_oxidized` | Rusted steel | Shifted toward red, lower overall |
| `asphalt_dry` | Dry road surface | Very low reflectance (~0.05) |
| `asphalt_wet` | Wet road surface | Specular spike, same base albedo |
| `vegetation_leaf` | Broadleaf foliage | Chlorophyll absorption at 670nm |
| `soil_dry` | Bare earth | Warm curve, moderate reflectance |
| `water_still` | Still water surface | Near-zero reflectance, high transmission |

New materials require a physically derived SPD — not an RGB approximation.

---

## Semantic Splat Segmentation

Every Gaussian carries a 16-bit `entity_id`. Assigned at generation time by the rule or
neural model that placed it. Component clusters (a window, a door, a chimney) share an ID.

The CUDA rasterizer writes entity IDs to a parallel entity buffer alongside the colour
framebuffer. This gives the engine a full semantic map of every rendered frame at zero
extra cost beyond the buffer allocation.

Uses:
- Click-to-select in the editor (which entity did the cursor hit?)
- Agent interaction (find the door, find the window, find the roof)
- Destruction masking (remove all Gaussians with entity_id = this window)
- Debug visualisation (colour by entity_id to see component boundaries)

---

## Shadow Catchers

Gaussian splats are volumetric — they do not cast hard shadows natively. When any asset
is plopped, the engine generates an invisible convex hull mesh (the Shadow Catcher) that
wraps the asset's Gaussian bounds. The spectral sun casts shadows from the Shadow Catcher
onto terrain and neighbouring assets.

This is a one-way system: the Shadow Catcher casts shadows but is never rendered itself.
Only the Gaussians are visible.

---

## Latent Refinement Pass (Alpha-Gaussian Blending)

At boundaries between Proc-GS structural Gaussians and neural infill clusters, a CUDA
blending pass runs after assembly:

1. Identify junction Gaussians (within N metres of a procedural/neural boundary)
2. Procedural Gaussians within the zone taper opacity toward the boundary
3. Neural Gaussians fade in from the boundary outward
4. Result: seamless surface, no visible seam between rule-generated and AI-generated
   geometry

---

## The Asset Library

Every asset created — via any pipeline — lands in a permanent, reusable library of `.vxm`
files. Assets are never remade. They are instanced.

### Instancing

One `.vxm` file is loaded into VRAM once. The engine instances it anywhere in the world
using a transform (position + rotation + scale). The Gaussian data itself is never
duplicated — only the transform is stored per instance.

```
bench_victorian_01.vxm   — 1 copy in VRAM
    ↓
Instanced 50,000 times across the city
Each instance carries:
  - world transform (position, rotation)
  - optional spectral shift (faded paint, rust, bird droppings)
  - optional opacity mask (broken slat, missing piece)
  - unique entity_id for this instance
```

### Library Growth

The library accumulates over time. Every session that creates assets adds to it permanently:

```
Buildings:
  house_victorian_terraced_01.vxm    ← Proc-GS, seed 0
  house_victorian_terraced_02.vxm    ← Proc-GS, seed 1
  house_victorian_terraced_worn.vxm  ← Proc-GS, wear_level 0.8
  town_hall_whitehall.vxm            ← Turnaround, landmark

Vegetation:
  oak_summer.vxm
  oak_winter.vxm
  grass_patch_dense.vxm

Props:
  bench_victorian_01.vxm
  lamp_post_gas_era.vxm
  letter_box_victorian.vxm
  cobblestone_panel.vxm              ← tileable terrain panel

Characters:
  pedestrian_victorian_male_01.vxm
  pedestrian_victorian_female_01.vxm

Proc-GS Components (sub-assets, not standalone):
  sash_window_canonical.vxm          ← referenced by building rules
  brick_panel_london_stock.vxm       ← referenced by building rules
  slate_roof_section.vxm             ← referenced by building rules
```

### Cross-Game Reuse

The library is engine-level, not game-level. Any game built on this engine uses the same
library. A Victorian city builder and a Victorian RPG share every asset. An asset made for
one game is available to all.

```
bench_victorian_01.vxm
  → used in city builder (park bench, 50,000 instances)
  → used in RPG (street prop, 200 instances)
  → used in horror game (same bench, spectral shift to night/decay)
```

Nothing is remade. Spectral shifts and wear parameters handle visual variation at runtime
without creating new files.

---

## Sustainability Model

| Engine | Method | Scales with |
|---|---|---|
| CityEngine | CGA grammar rules | Rule definitions (one per style) |
| SpeedTree | Parametric tree rules | Rule definitions (one per species) |
| Cities: Skylines 2 | Pre-made mesh assets | Artist hours per building |
| **This engine** | Proc-GS rules + neural infill | Rule definitions (one per style) |

Cities: Skylines 2 is the failure mode. Every building required an artist. This engine
avoids that — one rule set generates a style's entire building stock.

---

## Asset Lifecycle

```
Generation (Pipeline A, B, C, or D)
    ↓
Validation (vox_data: format, spectral range, density, entity_id integrity)
    ↓
.vxm file on disk (zstd compressed)
    ↓
Runtime instancing (transform applied, no re-generation)
    ↓
Runtime surface variation (spectral shifts, opacity masking)
    ↓
Optional: re-bake modified instance to new .vxm for persistence
```

---

## Phase Roadmap

| Phase | Deliverable |
|---|---|
| 0 | Synthetic procedural test asset (hardcoded, no rule system) |
| 1 | Proc-GS rule system: surface scattering + structured placement |
| 1 | Spectral materials library (10 base materials) |
| 1 | Deterministic seed-driven variation |
| 1 | Turnaround pipeline: Flux → 3DGS → spectral albedo reconstruction |
| 1 | Entity ID tagging in .vxm format and CUDA entity buffer |
| 1 | Shadow Catcher generation on asset plop |
| 2 | Neural Layout Interpreter: LLM → Latent Scene Graph → SVO placement |
| 2 | Compositional neural infill + Latent Refinement Pass |
| 2 | LLM rule authoring loop (forward, reverse, edit) |
| 3 | Lyra video capture pipeline |
| 3 | Dynamic wear / weather as runtime SPD parameter shifts |

---

## Mapping to Master Requirements List

| This document | Master list ref |
|---|---|
| Neural Layout Interpreter | #36, #44 |
| Proc-GS rules | #121 |
| Spectral materials library | #17, #11 |
| Seed-driven variation | #29 |
| Turnaround pipeline | #42 |
| Spectral albedo reconstruction | #208 |
| Compositional infill | #193 |
| Latent Refinement Pass | #14 (partial) |
| Entity ID / semantic segmentation | #44 |
| Shadow Catchers | #33 (partial) |
| Style transfer / LLM rule editing | #237 |
| Wear / weather SPD shift | #126, #35 |
| Lyra video capture | #42 (extended) |
| Validation passes | #277 (partial) |
| Asset lifecycle / instancing | #30, #255 |
