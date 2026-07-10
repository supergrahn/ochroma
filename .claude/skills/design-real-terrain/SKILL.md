---
name: design-real-terrain
description: Author photoreal layered terrain materials — base layer (grass/soil/rock per biome) plus physically-placed overlay layers (rock, dirt, mud, mulch, sand, snow) with per-layer relief and honest transitions. Use whenever authoring or tuning terrain surfaces, biome records, spray channels, or ground-material witnesses for Urban Horizon. Counterpart of design-real-building; same realism floor, applied to the ground.
---

# Design real terrain (the layered-materials standard)

The ground is a STACK, never a single texture: an opaque BASE layer per biome
(grass, soil, bedrock, sand) with alpha OVERLAY layers deposited on top — rock
outcrop, dirt, mud, mulch/litter, snow — each placed by a PHYSICAL driver, each
with its own relief. The authoring model already exists (`urban_horizon/src/map/
surface.rs`: `Vec<GroundLayer>`, layer 0 opaque base, later layers alpha
overlays, blended bottom→top; every ground set ships displacement). This skill
is the standard for WHAT to author and how to judge it.

## Step 1 — The square-metre prompt (photograph-first, before any texture picking)

For each biome/zone, write the prose description of ONE square metre as a macro
photo, answering all five:
1. **Geology & soil** — what is the base when everything above is scraped off?
   (loam under meadow, sandy till under pine, bedrock at ridges)
2. **Moisture regime** — where does water come from and go? (drives mud, gloss,
   moss, reed litter; wet ground is DARKER and SMOOTHER — albedo −30-50%,
   roughness down)
3. **What falls on it** — canopy above → needle/leaf mulch; cliffs above →
   scree; floods → silt bands; nothing → clean base.
4. **Disturbance** — paths, grazing, construction scars, fire — where and how
   worn (wear exposes the layer BELOW, never a new color).
5. **Season/climate state** — dry-summer tan vs spring green (the tint lives in
   the biome record, never baked into shared textures — the green-terrain law).

## Step 2 — The layer stack (order is deposition order, bottom → top)

| # | Layer | Role | Physical placement driver |
|---|---|---|---|
| 0 | BASE (opaque) | biome soil/grass/sand/bedrock | biome classification only |
| 1 | Rock/outcrop | parent material poking through | slope > ~30°, ridge curvature, thin-soil noise |
| 2 | Dirt/erosion | exposed subsoil | flow-accumulation scars, wear/paths, construction aprons |
| 3 | Mud/silt | wet fines | flow-accum hollows, shore bands, drainage lines; high moisture |
| 4 | Mulch/litter | organic fall | under canopy radius (FOREST_FLOOR), density ∝ tree density |
| 5 | Sand drift / snow | climate deposits | coastal bands / altitude+facing (snow: ledges not cliff faces) |

Rules:
- **Every overlay is JUSTIFIED by its driver** — a mud patch on a dry ridge or
  mulch in an open field is an authoring bug. Drivers come from the map's real
  fields (slope, curvature, flow, canopy, moisture), never freehand noise.
- **Transitions are height-aware, not fade-outs**: at a grass/rock boundary the
  rock's displacement must win in its own pixels (grass disappears INTO the
  cracks); a 50% alpha cross-fade band wider than ~0.5m reads as painted.
- **Wear reveals the stack**: a path is base-with-dirt-overlay at the center,
  bruised grass at the edges — the layers below must be authored as if exposed.
- **Gameplay coupling is SOFT**: layers modulate (build cost, growth, walk
  speed) — never hard-gate placement (the layering-and-impact decision).

## Step 3 — Per-layer detail floor (same relief law as buildings)

Every layer ships the FULL 2k map set (albedo + normal + roughness +
displacement) with an AUTHORED displacement scale — never a shared default:

| Layer class | displacement scale | roughness logic |
|---|---|---|
| Grass/meadow base | 15-30mm (blade clumps) | 0.8-0.95, brighter dry = rougher |
| Rock/outcrop/scree | 40-80mm | 0.75-0.9, wet faces darker+smoother |
| Dirt/subsoil | 8-20mm (clods, prints) | 0.85-0.95 dry / 0.5-0.7 wet |
| Mud/silt | 5-15mm (ruts, cracks) | 0.25-0.5 (wet gloss mandatory) |
| Mulch/needle litter | 10-25mm | 0.85-0.95 |
| Sand | 10-30mm (ripples) | 0.7-0.85 dry / 0.45 wet band |

- Tint discipline: dry = tan/straw, dirt = brown, mud = grey-brown — NEVER
  olive-wash everything green (the green-terrain root cause; retints live in
  `biome_record.rs`/`config` `TerrainMaterial`, not in shared textures).
- Tiling breakup at three scales: texture (2k set), meso (overlay masks),
  macro (biome-driven low-frequency modulation — close-the-gaps Task 8).
  A distant third of the map must differ from another third by ΔE > 6.

## Step 4 — Where it lives (engine truth, 2026-07-10)

- Author stacks per biome in `src/map/surface.rs` (`GroundLayer` — `alpha:
  Some` = overlay) + `src/map/biome_record.rs` (per-biome layers, splat_role,
  tints, dry/moist branches).
- The live wire is `src/map/spray.rs`: 12 channels, base + overlay per channel;
  the megakernel composites via `terrain_surface.overlay_composite`.
- **KNOWN LIMIT (the gap to close before deep stacks):** the runtime carries
  base + ONE overlay per channel, wired for 7 channels (`spray.rs`
  `biome_overlay` uses `layers.get(1)` only). Deeper authored stacks (rock+
  mud+mulch on one cell) silently truncate. Extending the composite to N
  layers (or 2 overlays: geological + organic) is a named engine task — do not
  author >2-deep stacks expecting them to render until it lands.
- Scatter is part of the material read: FOREST_FLOOR drives tree density and
  vice versa — a mulch overlay without canopy scatter above it is half-authored.

## Step 5 — Witness (all present-path, gate camera rules)

1. **Close-up raking light** (sun ≤25°): every layer's relief visible — grass
   clumps shadow, rock cracks deepen, mud ruts glint. Flat = fail.
2. **Transition crop**: grass→rock and grass→mud boundaries show height-aware
   interpenetration, no soft airbrush band.
3. **Driver sanity flyover**: rock only on slopes/ridges, mud only in hollows/
   shores, mulch only under canopy — one wrong-place patch fails the pass.
4. **Macro thirds**: three distant crops from different map thirds, pairwise
   ΔE > 6, no visible channel tiling.
5. **Wet state** (rain/shore): moisture darkening + gloss on mud/sand bands.
Record verdicts in the gap dossier ledger like every other witness.
