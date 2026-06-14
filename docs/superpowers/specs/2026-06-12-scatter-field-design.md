# Design: The Scatter Field — minimal-cost mass instancing of grass, then trees / props / debris (2026-06-12)

**Status:** Draft
**Scope:** Make the spectral splat engine render real grass at world scale (a billion-blade FIELD) at a fixed frame budget by GENERATING blade splats on the GPU per visible terrain tile every frame — never storing them — hash-seeded by world-tile coordinate, masked by the SDF/CellClass scatter mask, distance-LOD'd into the pinned T0/T1/T2 tier contract, and capped by the one `FrameBudgetGovernor`. Per-frame cost is bounded by SCREEN COVERAGE, never by world population. The same machine generalizes to trees/props/debris with a prototype atom-set per scatter-class. Engine crates stay game-agnostic: the engine knows "density field × mask × scatter-class", the game names the biomes.
**Related:** [Hybrid Atom+SDF LOD](./2026-06-12-hybrid-atom-sdf-lod-design.md) (PINNED tier contract — T0/T1/T2 and the shared `FrameBudgetGovernor`; this design is a SIBLING PRODUCER feeding the same tiled chain, it does NOT redefine the tiers), [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (§4.5 `ExpandDrawsPass` + §4.6 budget-derived capacity + §4.7 governor — the scatter generator reuses the indirect/budget machinery), the M3.2 plan [Virtualized Splat M3.2](../plans/2026-06-11-virtualized-splat-rendering-m3-2.md) (`encode_indirect` + device-emitted draws + one-submit/one-64B-readback — the shape the scatter dispatch copies), [SDF Pillar](./2026-06-10-sdf-pillar-design.md) (§4.6 `DistanceTransform2d` / `gpu/distance_field_2d.rs` Tier-3 2D field — the substrate for the per-tile DENSITY field, and the Tier-2 clipmap clearance-SDF the mask multiplies).

---

## 1. Problem Statement

Concrete, observable symptoms (verified 2026-06-12 against the current engine + the `2x2 km 0.5 m grass = 16M instances ≈ 1 GB transforms` industry datum from GPU procedural-placement practice):

- **There is no grass at all, and the only "mass instancing" path stores every instance.** The shipped scatter-class machinery is `atom_instances.rs::AtomInstance` (48 B/instance, uploaded to `expand_instances` storage) driven by `InstancedSelector` — built for ~10k BUILDINGS. A grass field at world scale is `2,048×2,048 m at 0.25 m pitch = 67 million blades`; at 48 B/`AtomInstance` that is **3.2 GB of host instance data before a single splat exists** — the same wall the building path already refused for atoms (`5.6 GB GPU` rejected in the virtualized-splat design §1). Storing blades is a non-starter; the building path's *dedup-by-asset* trick does not help because every blade is a unique placement.
- **Triangle grass cards cannot do what this engine's primitive does for free.** The conventional answer (AMD work-graph DetailedTile → 16×16 mesh-shader patches, Unity grass-compute blade cards) rasterizes sub-pixel triangle quads in HW, wasting 2×2 quads on distant blades and POPPING when a card's LOD swaps to a billboard. The published anti-aliased-splat line (Mip-Splatting / Multi-Scale 3DGS / LOD-GS) shows the splat primitive shrinks a blade to a sub-pixel anisotropic footprint that the EWA raster antialiases natively — the engine already owns that raster (`tiled_splat_renderer.rs`, frozen four-pass chain), and nothing produces grass into it.
- **The frame cost of a large field, if naively instanced, scales with POPULATION not coverage.** A 1 km grass field seen from a hill fills maybe 30% of the screen; the overwhelming majority of its blades are beyond 50 m and project sub-pixel. An instance-stored path pays O(blades-in-frustum) to cull and transform them all every frame. There is no representation between "individual blade" and "invisible", so the far field — where MOST of the pixels are — costs almost as much as the near field. The cost must COLLAPSE into the far tier, and today it cannot.
- **Determinism discipline forbids stored procedural state.** The engine's determinism contract (`distance_field_2d.rs` module doc: "bit-identical across runs and across GPUs … the determinism surface replay and the employment gate consume") means grass must look identical every frame AND across save/load AND across machines. A streamed/cached grass buffer would have to be persisted and replayed; a pure hash of world-tile coordinate has zero state to persist.
- **The mask substrate exists but feeds nothing.** `DistanceTransform2d` (SDF Pillar §4.6, shipped in `gpu/distance_field_2d.rs`) produces exactly the per-cell field a density mask needs, and the game already derives `CellClass::{Water, Slope, Buildable}` (`civitas_care/src/map/layers.rs`) + the clearance-SDF clipmap (SDF Pillar §4.2 Tier-2). No consumer multiplies them into a scatter density. Grass through buildings, roads, and water is the default failure mode of every naive scatter.

---

## 2. Done When

**Headline (all milestones landed), on the AMD 780M (RADV) floor and recorded on the RTX 4070 Ti target:**

running `cd ~/src/ochroma && cargo run --release -p vox_app --bin scatter_trial -- --grass --field-m 1024 --gate-ms 33` prints

```
[scatter_trial] grass field 1024x1024 m | population≈4.3e9 blades | tiles_visible=<V> tiles_T1=<n1> tiles_T2=<n2>
[scatter_trial] generated: blades_T1=<B≤budget> shell_tiles=<n2> | gen p50=<G> ms expand=<E> ms chain=<C> ms
[scatter_trial] frame p50=<X> ms p99=<Y> ms @ 1280x720 | budget=<BUD> settled
[scatter_trial] gate: p50=<X> ms vs gate=33.0 ms (tier=30fps-floor) adapter=<name> -> PASS
[scatter_trial] determinism: blade buffer bit-identical across 2 runs (sha256=<H>)
wrote scatter_grass.png   (green near field, yellow-tipped, blades shrink to sub-pixel dots with distance, far field a flat green ground-tint shell)
```

with **X ≤ 33.0** on the 780M floor, **B ≤ the budget** (printed, ≤ a few hundred thousand individual blades — never population), the far-field shell tiles' pixels costing O(1) ground-tint per tile (the cost-collapse, measured in M2), and the determinism line proving the same hash makes the same blades. The PNG passes eyeball: green near grass with yellow tips, blades visibly antialiasing to dots at range, the 1 km far field rendered as a single SDF-displaced ground-tint shell — and a human at the keyboard verifies all of it without reading code.

**Phased Done When per milestone:**

- **M1 — A procedurally-generated, hash-deterministic, budget-capped grass tile rendered as splats.** `cargo test -p vox_render --lib scatter_generate -- --nocapture` prints `[scatter] tile (17,42) seed=<s> generated <N> blades, budget=<B>, capped=<true|false> | gen <G> ms | rendered non_black=<nb>` with `N ≤ B` always, `N > 0`, every blade a `kind==1` anisotropic `GaussianSplat` whose y-extent ≥ 4× its x/z-extent (printed min/mean aspect), the green→yellow spectral gradient asserted (tip bin > root bin in the long-wavelength half, printed), AND a second run over the SAME tile coordinate producing a **byte-identical** blade buffer (`assert_eq!` on the readback, sha printed). A stub that returns `Vec::new()` or ignores the budget fails the `N > 0 && N ≤ B` assert.
- **M2 — Distance LOD into the tier contract + the cost-collapse measurement.** `cargo test -p vox_render --lib scatter_lod -- --nocapture` prints `[scatter] LOD: tile@<d1>m=T1(<a> blades) tile@<d2>m=T1(<b><a> blades) tile@<d3>m=T2(shell, 0 blades, 1 ground-tint draw)` proving blade count strictly DECREASES with distance and collapses to a shell past the pinned T2 distance, AND `[scatter] cost-collapse: 1024 m field, 30% screen coverage → <p>% of field PIXELS in T2 shell, T2 cost=<ct> blade-equivalents vs T1 cost=<c1>` with **p ≥ 80%** (most pixels in the cheap tier) and `ct < 0.01 · c1` (the far field is two-to-three orders cheaper than if it were blades), both numbers computed from real tile classification over a real camera, not asserted constants.
- **M3 — SDF/CellClass mask over real terrain + wind.** `cargo test -p vox_render --lib scatter_mask -- --nocapture` prints `[scatter] mask: density×clearance-SDF×cellclass → 0 blades over water/road cells (<wr> sampled, 0 leaked), <gf>% density on open grass | wind: phase spread <ps> rad, tip displacement <td> m at t=1.0s` — zero blades generated where the mask reads impassable (asserted over a real water/road mask), a believable wind tip-sway computed from the per-blade hash phase + a wind field (printed displacement non-zero and bounded), and the mask sample cost `<ms> ns/blade` printed (≤ the budget stated in §4.4). A human runs `cargo run --release -p vox_app --bin scatter_trial -- --grass --mask demo_city` and sees grass on lawns, bare ground on roads/water, swaying.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Hash placement is deterministic | generate tile `(17,42)` twice, read back both blade buffers: `assert_eq!(buf_a, buf_b)` byte-for-byte, sha printed | `assert!(!blades.is_empty())` |
| Population independence (coverage-bound cost) | generate a 4-tile near patch and a 4096-tile far field at the same budget: `assert!(near_blades <= budget && far_field_blades <= budget)` and `assert!(gen_ms_far < 2.0 * gen_ms_near)` — 1000× more population, < 2× cost, both printed | `assert!(blades <= budget)` on one tile (trivially true) |
| Budget cap is real | budget = 10_000, a tile that would place 50_000: `assert_eq!(generated, 10_000)` (the indirect `atomicAdd` clamp), printed `capped=true` | timing one dispatch without checking the count |
| Blade is an anisotropic vertical splat | over 1000 generated blades: `assert!(min_aspect_y_over_x >= 4.0)` and `assert!(kind == 1)` (3DGS volume), printed | `assert!(scale_u > 0.0)` |
| Green→yellow spectral gradient | tip splat vs root splat of a curved blade: `assert!(tip.spectral[long_band] > root.spectral[long_band] && root.spectral[green_band] > tip.spectral[green_band])`, both printed | `assert!(spectral[0] > 0.0)` |
| Distance LOD shrinks then collapses | classify one tile at 20/120/600 m: `assert!(blades_20 > blades_120 && blades_600 == 0 && shell_draws_600 == 1)`, printed | `assert!(lod(600.0) != lod(20.0))` without counting |
| Cost collapses to T2 | real camera over a 1024 m field: `assert!(t2_pixel_fraction >= 0.80 && t2_cost < 0.01 * t1_cost)`, both computed and printed | asserting a hard-coded percentage |
| Mask kills grass on water/road | real water+road `CellClass` mask: `assert_eq!(blades_on_impassable, 0)` over ≥ 1000 sampled impassable cells, printed | `assert!(density >= 0.0)` |
| Wind perturbs the tangent frame | t=0 vs t=1.0 s on one blade: `assert!(tip_xz_displacement > 0.0 && tip_xz_displacement < blade_height)`, printed; root unmoved | `assert!(wind != 0.0)` |
| Generalizes to a prototype scatter-class | a 5-atom "rock" prototype scatter-class at lower density: `assert!(rock_blades_per_tile < grass_blades_per_tile && rock_atom_count == 5 * rock_instances)`, printed | `assert!(class_count > 0)` |

---

## 4. Architecture

### 4.1 Honest framing — the headline claim, stated correctly

A grass field at 0.25 m pitch over 1024×1024 m is **≈ 4.3 billion notional blades** (≈ 16.8 M at 0.25 m over a 2 km² map scaled up; the exact "billion" is a function of pitch × area and is reported, not invented). The engine NEVER materializes them. What it pays for, per frame, is bounded by what the camera can SEE: a budget-capped count of individual blade splats in the near/mid tiers (T0/T1) plus one ground-tint draw per far tile (T2). The correct sentence — used everywhere in this design and its prints — is:

> **"a billion-blade FIELD rendered at a fixed frame budget."** Never "a billion blades per frame."

The per-frame INDIVIDUAL-blade ceiling is whatever fits the splat budget within coverage — empirically **tens-to-low-hundreds of thousands** of blades (a 150k–250k slice of the same `FrameBudgetGovernor` cap the building path uses), with everything beyond the T1 distance aggregated to shells. That ceiling is a hard number printed every run (§4.9). Where does the "billion-blade cost" actually go? **Into a per-tile constant.** A far tile is one ground-tint shell draw regardless of how many notional blades it contains — a 32×32 m tile holding ~16k blades at 0.25 m collapses to ONE textured-surface contribution. The cost is `O(visible tiles)`, and visible-far-tile count is `O(√coverage)`, not `O(population)`.

This is the splat-native version of the industry pattern (AMD work-graph: DetailedTile 16×16 patches near, single sparse quad far; procedural-placement-on-terrain: allocate memory only for in-range instances) — but with the splat advantages the tier contract already banks: free sub-pixel AA, continuous (crossfaded, never popping) LOD, and a far representation (`ground-tint shell`) that is a primitive of the engine, not separately-authored imposter art.

### 4.2 The pipeline at a glance — a SIBLING PRODUCER into the frozen chain

The scatter field reuses, beat for beat, the producer→tiled-chain seam the building path proved (virtualized-splat §4.5–4.7; M3.2 `encode_indirect`):

```
 per frame, render thread, ONE shared GpuContext:
   ScatterTileCull (compute)  — frustum + distance cull the world-tile grid →
        visible-tile list split into T1 (blade-gen) and T2 (shell) buckets,
        indirect dispatch args, all DEVICE-side (no host tile list)
   ScatterGenerate (compute)  — one workgroup per T1 tile: hash-place blades,
        mask-test each, atomicAdd survivors into splat_buf/transform_buf at a
        prefix-summed offset, CLAMPED at the budget tail (the §4.6 cap)
   ScatterShell   (compute)   — one thread per T2 tile: write ONE ground-tint
        surface splat (2DGS disk) into splat_buf — the cost-collapsed far field
   [building ExpandDrawsPass writes its atoms into the SAME splat_buf, disjoint
    slot range — scatter and buildings are co-producers under one budget]
   TiledSplatRenderer::render  — the FROZEN four-pass chain, unchanged, over the
        combined active splat count
```

The frozen `tile_assign → radix_sort → tile_range_build → splat_raster` chain is **not touched** — exactly the contract `ExpandDrawsPass` and `GiCombinePass` honor (write into `splat_buf`, let `tile_assign` fill `conic`/depth in-device). Scatter generation writes `GpuSplatFull` (80 B) + the transform pair (32 B) into the renderer's persistent buffers at slots disjoint from the building expand's, using the same `new_with_capacity` / `set_active_splat_count` discipline. The shared `FrameBudgetGovernor` (`atom_instances.rs:984`) hands the scatter generator a sub-budget each frame; over-budget tiles clamp (drop blades), never abort.

### 4.3 Generation — the per-visible-tile compute pass (M1)

**The world-tile grid.** The field is partitioned into fixed `TILE_M`-metre square tiles (pinned `TILE_M = 8.0` near, see §4.5 for why far tiles aggregate 4×4 of these). A tile's identity is its integer world-tile coordinate `(tx, tz) = (floor(world_x / TILE_M), floor(world_z / TILE_M))` — the ONLY input to its hash. No tile stores anything; `(tx, tz)` regenerates it identically forever.

**Tile cull first (`ScatterTileCull`).** Before any blade exists, one compute thread per candidate tile (the grid cells whose AABB intersects the camera frustum's ground footprint — a bounded ring around the camera, radius = the T2 cutoff) does: frustum-test the tile's ground AABB (the proven `contains_sphere` / plane test reused from `instanced_select_gpu.wgsl`), compute eye distance, classify **culled / T1-near / T2-far** by the pinned tier distances (§4.5), and `atomicAdd` the tile index into the matching device-side bucket + bump the indirect dispatch args. This is the `scan_pairs → k2_args` shape from M3.2: the host never sees the tile list; the next dispatch runs indirect off the bucket counts. Cost is `O(candidate tiles)` ≈ a few thousand `worst-case` (a 1 km ring at 8 m tiles ≈ 49k cells, frustum-pruned to the ~6k in front of the camera) — trivial.

**Hash placement (`ScatterGenerate`).** One workgroup per T1 tile (indirect dispatch off the T1 bucket). The tile seeds a counter-based hash from `(tx, tz)`; the chosen hash is **`pcg3d`** (the integer counter-based hash, `hash(uvec3) -> uvec3`), keyed `pcg3d(tx, tz, blade_index)` — counter-based (not a stateful RNG) so it is trivially parallel, identical on every GPU, and replay-safe; this is the integer-hash production practice the procedural-placement literature pins, and it matches the engine's existing `splitmix64`/`pcg`-style determinism (the `hash_u64` house helper in `atom_instances.rs`/`distance_field_2d.rs`). Each of the tile's `BLADES_PER_TILE` candidate slots:
1. `h = pcg3d(tx, tz, k)` → two uniform `f32`s for the in-tile `(u, v)` offset (jittered grid: blade `k` lives in sub-cell `k` with a hash jitter, so density is even, not clumpy), more lanes of `h` for height/lean/phase/colour jitter.
2. Sample the mask (§4.4) at the blade's world position: `density × clearance_sdf_gate × cellclass_gate`. If a hash lane `> density`, or either gate is 0, the slot is REJECTED — no blade.
3. Survivors `atomicAdd(1)` a tile-shared then global counter to claim a prefix-summed output slot in `splat_buf`. If the global counter is `≥ budget_tail`, the blade is dropped (the clamp — the honest ceiling, §4.6). Otherwise emit the blade primitive (§4.4-blade) directly into `splat_buf[slot]` + `transform_buf[2·slot..]`, `conic`/depth zeroed for `tile_assign`.

The output is byte-deterministic: the hash is a pure function of `(tx, tz, k)`, the mask is a pure function of world position, and slot assignment is by `atomicAdd` — which, **for the COUNT and the per-blade CONTENT is deterministic**, but blade-to-slot ORDER is arrival-dependent. To keep the buffer bit-identical (M1's determinism gate) the generator uses the M3.2 lesson: claim slots by a **deterministic prefix sum over `(tile, k)`**, not raw `atomicAdd` arrival order — each tile reserves a contiguous range sized by a first-pass survivor count (a 2-pass count-then-emit per tile, the same shape `distance_field_2d` uses to stay replay-safe), so blade `(tile, k)` always lands at the same slot. (Raw `atomicAdd` would still be CORRECT to render and population-deterministic, but not byte-identical across runs — the prefix-sum buys the stronger save/load/replay guarantee §4.7 needs.)

### 4.4 The blade primitive and the mask

**One blade = one (or 2–3 stacked) anisotropic `GaussianSplat`.** A grass blade IS a vertically-stretched, slightly-curved, anisotropic Gaussian — the engine's `GaussianSplat::volume` primitive (`vox_core/src/types.rs:96`) already carries everything needed: `position`, anisotropic `scale = [x, y, z]` (the half-axes), a `rotation` quat (the tangent frame), `opacity`, and the 16-band `spectral` (the colour). The pinned blade:
- **Scale (anisotropy):** `scale = [w, h, w]` with `h ≈ 0.12–0.30 m` (hash-jittered blade height) and `w ≈ 0.006–0.012 m` (blade half-width). The y/x aspect ≥ 4:1 (M1 asserts ≥ 4) — this is what makes it read as a blade, not a dot, near; and what makes it shrink to a sub-pixel dot far (the free-AA property).
- **Tangent frame (rotation):** the blade's up-axis is the local terrain normal tilted by a hash-jittered lean (a few degrees) plus the wind perturbation (§4.6). The quat is `from_rotation_arc(Y, lean_dir)` composed with a hash yaw — so blades point believably up-and-leaning, not all-identical.
- **Curve (2–3 stacked splats):** a single splat is a straight blade; a believable curved blade is **3 stacked volume splats** along the lean arc (root, mid, tip), each shorter, the tip leaned furthest — `BLADE_SPLATS ∈ {1, 3}`, hash-or-distance-chosen (near blades curve with 3, mid blades straighten to 1 — a free micro-LOD inside T1). Each stacked splat is an independent budget slot.
- **Spectral gradient (green→yellow):** the root splat's spectrum peaks in the green bands (≈ 520–560 nm), the tip splat's peak shifts toward yellow (≈ 560–600 nm, the long-wavelength half) with lower saturation — the classic grass tip-dry gradient. With 3 stacked splats the gradient is across the stack; with 1 it is baked into the single spectrum biased by the blade's hash (some blades greener, some yellower — spectral colour variation, §4.6). M1 asserts `tip.long_band > root.long_band` and `root.green_band > tip.green_band`.

*Why this beats triangle blade cards (the published-splat-AA argument, made concrete):*
- **Free anti-aliasing.** A distant blade splat's projected footprint shrinks below a pixel and the EWA raster integrates it as a sub-pixel anisotropic Gaussian — exactly the Mip-Splatting / Multi-Scale-3DGS result. A triangle blade card at sub-pixel size shimmers and requires MSAA/TAA the engine would otherwise not pay for. Grass is *near-ideal splat content* precisely because almost all of a field's blades are sub-pixel.
- **Continuous LOD, no popping.** Blade count, blade height, and the 1↔3-splat curve all fade with distance through the same crossfade-opacity mechanism the tier contract already uses (`hierarchical_lod::crossfade_factor`); a triangle field pops when a card LOD or a billboard swaps.
- **No separate imposter authoring.** The far representation (T2 ground-tint shell, §4.5) is the same `GaussianSplat` primitive (a flat 2DGS surface disk), not a hand-made billboard atlas. One primitive, three tiers.

**The mask — `density × clearance-SDF × CellClass` (M3).** No grass through buildings/roads/water. Three multiplicative gates, sampled at the blade's world `(x, z)`:
1. **Density field** — a per-tile (or per-region) scalar field giving the local grass density `[0, 1]` (lawns dense, scrubland sparse, bare ground 0). Stored in the **`DistanceTransform2d` / 2D-field substrate** (SDF Pillar §4.6, `gpu/distance_field_2d.rs`) — a `1024²` f16/u8 field is **1–2 MB** for a 2 km map (the design's whole "tiny memory" claim: the field is the only stored thing, and it is the mask, not the blades). The engine type is semantic-free; the game writes "grass density" into it exactly as it writes "childcare cost" into the coverage field.
2. **Clearance-SDF gate** — sample the Tier-2 clipmap clearance SDF (SDF Pillar §4.2): `gate = smoothstep(0, clearance_margin, sdf_clearance(x,z))` → 0 inside/near any building footprint, 1 in the open. This is what keeps grass from growing through walls and under foundations, for free, off a field the engine already maintains.
3. **CellClass gate** — `gate = is_grassable(cellclass(x,z)) ? 1 : 0`, where the game maps `CellClass::{Water, Slope-cliff, Road}` → 0 and `Buildable`/lawn → 1 (`civitas_care/src/map/layers.rs`). Engine-side this is one `u8` mask lookup, identical machinery to the density field; the game owns the mapping.

**Mask sample cost.** All three are **one texture/buffer fetch each** (the density and CellClass are 2D fields, hardware-or-buffer fetch; the clearance is one trilinear clipmap fetch — the SDF Pillar's "1 fetch/sample" property §4.4). Three fetches per candidate blade slot, on the GPU, fully coherent within a tile's workgroup → the budgeted cost is **≤ ~10 ns/candidate** on the 780M (memory-bound, three coherent fetches), and rejected candidates cost only the fetches (no emit). M3 prints the measured `ns/blade`.

### 4.5 Distance LOD into the pinned tier contract (M2)

The tier contract is PINNED by the sibling hybrid-LOD design; grass slots into it — it does NOT redefine it. The mapping (grass's reading of T0/T1/T2):

| Tier | Pinned distance | Buildings (sibling design) | **Grass (this design)** |
|---|---|---|---|
| **T0 near** | `0 – 50 m` | SDF surface | **dense individual blade splats, 3-splat curve, full height, full density** |
| **T1 mid** | `50 – 150 m` | atom splats (virtualized chain) | **individual blade splats, fewer (density falls off), 1-splat straight, crossfaded** |
| **T2 far** | `≥ 150 m` | aggregate imposters (boxes) | **SDF-displaced / textured ground-tint SHELL — one surface splat per far tile, the field's colour baked into the terrain surface** |

The distances are `hierarchical_lod::LOD_DISTANCES = [0, 50, 150, 400]` (verified in `atom_instances.rs`), shared verbatim. Grass uses `T0/T1` for individual blades (the T0/T1 split is a *within-T1* density/curve falloff for grass — both are "individual blade splats", the engine emits blades in both, just fewer and simpler with distance) and the **150 m boundary** as the blade→shell transition, crossfaded across `150 m → transition_end` so blades dissolve into the shell with no pop (the `crossfade_factor` mechanism).

**The cost-collapse argument, with real numbers (the M2 measurement, computed not asserted):**
Take the headline 1024×1024 m field, the M2 camera a person standing on a low rise (eye 2 m, looking out), screen coverage ≈ 30% field. Partition the visible field by tier:
- **T1 blade ring (0–150 m):** a 150 m-radius half-disc in front of the camera ≈ `π·150²/2 ≈ 35,000 m²` of ground. At the governor's grass sub-budget (say **150,000 blade-splats**), that is the entire individual-blade cost — `150k splats`, the honest ceiling.
- **T2 shell field (150 m – field edge):** the rest of the visible 1 km field — `≈ 300,000+ m²` of ground, ≈ **90% of the visible field AREA and, because far ground compresses in screen space, ≈ 80–88% of the field's screen PIXELS**. Its cost: one ground-tint surface splat per 32×32 m far tile → `300,000 / 1024 ≈ 290 shell draws`. **290 splats** for the 80%+ of pixels that would, as blades, have been `300,000 m² × 16 blades/m² ≈ 4.8 MILLION blades`.

So the cost ratio is `290 shell-splats : 4.8M notional-blades ≈ 1 : 16,000` — the far field is **~4 orders of magnitude** cheaper than blades, and **M2's `ct < 0.01 · c1`** (shell cost under 1% of the blade-ring cost) is the conservatively-stated, measured form of that. **This is where the "billion-blade cost" goes: it collapses into a few hundred per-tile shell draws.** The per-frame individual-blade count is pinned at the budget (150k), independent of whether the field is 1 km or 100 km — only the shell-tile count grows with visible area, and that grows as `O(√coverage)`, sub-linearly, never as population.

### 4.6 Determinism, animation, spectral variation

**Determinism (M1 gate, §4.7 cross-save).** The blade buffer is a pure function of `(tx, tz, k)` through the hash, the static density/mask fields, and the wind TIME (which is a sim input, replayed identically). No blade state is stored. Same frame → same blades; same save/load → same blades (the fields are part of the saved map, the hash is code); same machine or another → same blades (counter-based `pcg3d` is integer-exact, the prefix-sum slot assignment is order-free). This is the `distance_field_2d` determinism discipline applied to scatter: *the determinism surface is the hash + the fields, both replay-safe, neither streamed.*

**Animation — wind as a tangent-frame perturbation in the generation pass.** Wind is a small analytic 2D field `wind(x, z, t)` (a sum of 2–3 scrolling sine waves — the standard cheap grass-wind field; engine-generic, the game can drive its amplitude/direction from weather). In `ScatterGenerate`, AFTER the base tangent frame is built, each blade's tip lean is perturbed by `wind(x,z,t) · per_blade_phase`, where `per_blade_phase = hash_lane(h)` so neighbouring blades sway out of phase (no rigid "whole field tilts as one"). With the 3-stacked-splat curve, the perturbation increases root→tip (root fixed, tip sways most) — a believable blade bend, computed as `quat` tweaks, zero extra primitives. Because wind is applied AT generation and `t` is a replayed sim input, animated grass stays fully deterministic. M3 asserts non-zero, bounded tip displacement with the root unmoved.

**Spectral colour variation.** Each blade's hash also drives a small spectral jitter: a per-blade hue offset (greener ↔ yellower), a brightness jitter, and the root→tip green→yellow gradient (§4.4-blade). This is `spectral[16]` math at generation — the engine's native colour space — giving a field that varies blade-to-blade and dries toward the tips, all from the same free hash lanes, no texture.

### 4.7 Save / load / replay safety

There is nothing to save. A grass field is `(the density field + the mask + the scatter-class params + the hash)`. The density/mask fields ride the existing map save (they are `DistanceTransform2d`-class 2D fields, part of the map, ≈ 1–2 MB); the scatter-class params (blade height range, density curve, colour) are static asset data; the hash is code. On load, the identical fields + identical hash reproduce the identical field, frame-for-frame, including animation given the replayed wind `t`. This is strictly stronger than a streamed/cached grass system, which would have to persist and version a multi-gigabyte instance buffer and prove it reloads bit-identically — the scatter field sidesteps that entire class of bug by storing none of it.

### 4.8 Generalization — trees / props / debris are the same machine (the AssetAtomLibrary seam)

Grass's "procedural blade" is the only grass-specific piece. Replace it with a **prototype atom-set per scatter-class** and the identical pipeline scatters trees, rocks, props, debris:
- A **scatter-class** = `{ prototype: AssetAtomLibrary asset (the existing cooked atom set), density (lower — trees are sparse), scale jitter, rotation policy, mask gates }`. The generator, instead of synthesizing a blade, **instances the prototype's atoms** at the hash-placed position — which is EXACTLY what `ExpandDrawsPass` already does for buildings (`expand_draws.wgsl`: fetch asset-local atom, `pos' = iq·p + it`, write to `splat_buf`). So trees/props are literally `scatter placement → AtomInstance → the existing building expand path`, with the placements GENERATED per-tile by the hash instead of stored. The seam to the `AssetAtomLibrary` is clean: grass is the special-case "1–3 procedural atoms" prototype; everything else is "cooked-prototype atoms", same budget, same tiers (a far tree → its I0/I1 imposter or, with the sibling design, its SDF box; a far debris field → a ground-tint/rubble shell).
- Density/clearance/CellClass masking, frame-budget capping, distance-LOD into T0/T1/T2, determinism, and wind (trees sway too, via the same per-instance phase applied to the prototype's tangent frame) all carry over unchanged. Debris from destruction (SDF Pillar §4.8) is a scatter-class whose density field is the carve's dirty region — the destruction system writes a density field, the scatter machine renders it.

The engine stays game-agnostic throughout: it knows `scatter-class = (prototype | procedural-blade) × density-field × mask × tier-policy`. "Grass", "oak", "rubble", "biome" are game words written into the density field and the class params.

### 4.9 The honest cost ceiling — stated, never exceeded

The per-frame INDIVIDUAL primitive count from scatter is `min(survivors-within-coverage, grass_sub_budget)` — a **hard, printed** number, a slice of the one `FrameBudgetGovernor` cap (the building path's `budget_cap`, shared). Empirically **tens-to-low-hundreds of thousands** of individual blade-splats (the M2 ceiling pins it at ~150k on the 780M floor, more on the 4070 Ti). Everything beyond T1 is shells: `O(visible far tiles)`, a few hundred. The design NEVER claims per-frame billions; every print says "field" for population and "budget"/"selected" for the per-frame count, and the M1 determinism + M2 cost-collapse + §4.9 ceiling are the three numbers that keep the claim honest.

**Cost arithmetic, both machines (the §2 headline, justified):**

| Stage | 780M floor (RADV, UMA) | 4070 Ti target |
|---|---|---|
| `ScatterTileCull` (~6k candidate tiles) | ~0.1 ms | < 0.02 ms |
| `ScatterGenerate` (150k survivors, 3 mask fetches each, ~2× candidates tested) | ~1.5–2.5 ms (memory-bound, the `distance_field_2d` 1024² JFA is ~5–6 ms for 10× the fetches, so 150k×3 fetches budgets here) | ~0.3 ms |
| `ScatterShell` (~290 tiles) | < 0.05 ms | negligible |
| chain raster of +150k blade-splats (inside the shared budget) | folded into the frozen chain's `chain ≈ 10–21 ms` at the 100–150k budget (M3.2 measured) | < 5 ms (M3.2 target-hardware arithmetic) |
| **Scatter's own added cost** | **~2–3 ms gen+shell** on top of the shared chain | **< 0.4 ms** |

The 780M holds the **33 ms floor** (M3.2's proven envelope: chain 10–21 ms + scatter gen 2–3 ms + the building path, governed to the budget); the 4070 Ti holds **16.6 ms** (the order-of-magnitude headroom M3.2 pins). The grass sub-budget is the lever — at the floor the governor settles it where 33 ms holds; on target it opens it for denser near grass under 16.6 ms. **No number is quoted without its tier and adapter** (the M3.2 gate-line discipline, reused by `scatter_trial`).

### 4.10 Threading / device / ownership

Engine-only, `vox_render`. One new module `crates/vox_render/src/gpu/scatter_field.rs` (+ `scatter_field.wgsl`) owns the three compute passes; render-thread-owned (`&mut`, no locks), shared `GpuContext` (`new_with_context`, the `GpuGi`/`ExpandDrawsPass` precedent). It writes into the renderer's persistent `splat_buf`/`transform_buf` (the `ExpandDrawsPass` contract) at a slot range disjoint from the building expand's; the harness (`scatter_trial`) and later the game's `SceneRenderer` allocate the combined capacity and call `set_active_splat_count(building_atoms + scatter_splats)`. The density/mask fields are `DistanceTransform2d`-class engine types the game fills. The frozen four-pass chain and the building expand path are untouched — scatter composes purely through buffers and additive APIs.

---

## 5. Data Models

```rust
// crates/vox_render/src/gpu/scatter_field.rs (new) — engine-generic.

/// A scatter-class: what to scatter and how densely. Engine knows
/// `procedural blade` vs `cooked prototype`; the game names the biome.
#[derive(Debug, Clone)]
pub struct ScatterClass {
    kind: ScatterKind,          // private — Blade | Prototype(asset)
    base_density: f32,          // blades/instances per m² at density-field = 1.0
    height_range: [f32; 2],     // blade height jitter (m); prototype: scale jitter
    blade_splats: u8,           // 1 or 3 (curve) — Blade kind only
    tier_distances: [f32; 4],   // pinned LOD_DISTANCES; not redefined, carried
}
impl ScatterClass {
    pub fn grass(base_density: f32, height_range: [f32; 2]) -> Self;
    pub fn prototype(asset: u32, base_density: f32) -> Self;
    pub fn base_density(&self) -> f32;
}

#[derive(Debug, Clone)]
enum ScatterKind { Blade, Prototype { asset: u32 } }  // private

/// GPU per-tile generation params (16 B uniform, std140). Pinned hash inputs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterParams {
    tile_m: f32,                // metres per near-tile (8.0)
    blades_per_tile: u32,       // candidate slots per T1 tile
    budget_tail: u32,           // global splat-buffer slot ceiling (the clamp)
    wind_t: f32,                // sim time (replayed) — determinism input
}

/// One emitted scatter result. `selected` is the per-frame INDIVIDUAL ceiling
/// (≤ the grass sub-budget); `shell_draws` is the cost-collapsed far field.
pub struct ScatterStats {
    pub tiles_visible: u32, pub tiles_t1: u32, pub tiles_t2: u32,
    pub blade_splats: u32,      // individual blades emitted (≤ budget) — the ceiling
    pub shell_draws: u32,       // far-tile ground-tint shells
    pub clamped: bool,          // hit budget_tail this frame
    pub gen_ms: f32,
}
```

The blade primitive is the existing `vox_core::types::GaussianSplat` (`volume` for blades, `surface` for shells) and the existing `gpu::splat_buffer::GpuSplatFull` (80 B) — scatter introduces NO new splat layout, only the generator.

---

## 6. API

```rust
// crates/vox_render/src/gpu/scatter_field.rs (new, GPU; shared-context house pattern)
impl ScatterField {
    /// Build the three passes on the shared context. Validates the generation
    /// buffers against device limits UP FRONT (the ExpandDrawsError house
    /// pattern). `classes` are the resident scatter-classes (grass + any
    /// prototypes); prototypes reference the same AssetAtomLibrary the building
    /// expand uses (the §4.8 seam). Errors mirror ExpandDrawsError.
    pub fn new_with_context(
        ctx: &GpuContext,
        classes: &[ScatterClass],
        budget_cap: u32,
    ) -> Result<Self, ScatterFieldError>;

    /// Set/replace the density + CellClass mask fields for a class (the game
    /// writes grass density; engine never learns "grass"). 2D-field substrate.
    pub fn set_density_field(&mut self, class: u32, field: &DistanceTransform2d);
    pub fn set_cellclass_mask(&mut self, class: u32, mask: &wgpu::Texture);

    /// Record the full per-frame scatter into `encoder`: tile-cull → generate
    /// → shell, writing GpuSplatFull + transforms into `splat_buf`/`transform_buf`
    /// at `[base_slot, base_slot + budget)` (disjoint from the building expand's
    /// range), dispatched INDIRECT off device-side tile buckets (M3.2 shape).
    /// `clearance` is the Tier-2 clipmap SDF (mask gate 2). Returns ScatterStats
    /// (one 64 B readback, the M3.2 discipline). Caller then adds blade_splats to
    /// the active count and renders the frozen chain.
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &RenderCamera,
        clearance: &SdfClipmap,
        wind_t: f32,
        budget: u32,
        base_slot: u32,
        splat_buf: &wgpu::Buffer,
        transform_buf: &wgpu::Buffer,
    ) -> ScatterStats;
}

// Existing signatures relied on verbatim (do not re-derive):
//   GaussianSplat::volume(position:[f32;3], scale:[f32;3], rotation: Quat, opacity: u8, spectral:[u16;16]) -> Self
//   GaussianSplat::surface(position, tangent_u, tangent_v, scale_u, scale_v, opacity, spectral) -> Self  // the T2 shell disk
//   TiledSplatRenderer::new_with_capacity(ctx, max_splats, w, h) / set_active_splat_count(n) / splat_buf() / transform_buf()
//   ExpandDrawsPass::encode_indirect(encoder, src, instances, splat_buf, transform_buf)  // the prototype-scatter path reuses this
//   DistanceTransform2d::distance_texture()/seed_id_texture()  // the density-field substrate (gpu/distance_field_2d.rs)
//   SdfClipmap::texture()/header_buf()  // the clearance gate (SDF Pillar §4.2)
//   FrameBudgetGovernor::new(target_ms, initial, min, max) / budget() / update(frame_ms)  // atom_instances.rs:984
// Threading: ScatterField is render-thread-owned (&mut), shared GpuContext, no locks.
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `ScatterField::new_with_context` | `scatter_trial` setup; game `SceneRenderer::new_city` | `crates/vox_app/src/bin/scatter_trial.rs` (new); `civitas_care/src/render_gpu/mod.rs` | once; shares the building `AssetAtomLibrary` for prototype classes |
| `ScatterField::set_density_field` / `set_cellclass_mask` | game map build / on terrain edit | `civitas_care/src/render_gpu/mod.rs` | game writes grass density into the 2D field; engine semantic-free |
| `ScatterField::encode` | every frame, one encoder, before `TiledSplatRenderer::render`, after the building `ExpandDrawsPass::encode_indirect` | `scatter_trial` frame loop; `SceneRenderer::render` | writes scatter splats into `splat_buf` at `base_slot = building_atoms` |
| `set_active_splat_count(building_atoms + scatter.blade_splats + scatter.shell_draws)` | same frame, after `encode` | both | the combined producer count; chain renders both |
| Shared `FrameBudgetGovernor` | wraps the `budget` arg split between building + scatter sub-budgets | `SceneRenderer::render`; `scatter_trial` loop | one governor over all tiers (the pinned contract) |
| Density-field substrate | `ScatterField` binds `DistanceTransform2d` outputs | `gpu/distance_field_2d.rs` | the only stored thing; ≈ 1–2 MB |
| Clearance gate | `ScatterField::encode(clearance = SdfClipmap)` | SDF Pillar `gpu/sdf_clipmap.rs` | mask gate 2 (1 trilinear fetch) |
| Prototype-scatter (trees/props) | `ScatterField` emits `AtomInstance`s → building `ExpandDrawsPass::encode_indirect` | `gpu/expand_draws.rs` | §4.8 seam — same expand path, generated placements |
| M1/M2/M3 tests | `cargo test -p vox_render --lib scatter_*` | `scatter_field.rs` tests | synthetic tile/camera/mask; the GPU house pattern + `try_gpu_context` |

---

## 8. Open Questions

- [ ] **Grass sub-budget split.** How is the one `FrameBudgetGovernor` cap divided between buildings (T1 atoms) and scatter (T1 blades) per frame? Static split (e.g. 60/40), or a second proportional loop on each producer's measured cost? Decide from the M2 numbers — measure both producers' ms at the floor before pinning a split; a static split is the v1 default unless M2 shows starvation.
- [ ] **Two-pass count-then-emit vs raw atomicAdd for the determinism gate.** §4.3 pins the prefix-sum (2-pass) slot assignment for byte-identical buffers. Measure the 2-pass cost in M1; if it doubles `gen_ms` and the save/replay gate can be met by re-deriving (the field is a pure function regardless of slot order, so a stable SORT of the survivor slice by `(tile,k)` after a raw atomicAdd may be cheaper). Decide from M1's printed `gen_ms` both ways.
- [ ] **Tile size `TILE_M` and `BLADES_PER_TILE`.** Pinned 8 m / (8 m × 8 m × 16 blades/m² ≈ 1024 candidates) for M1; the workgroup-occupancy sweet spot wants `BLADES_PER_TILE` a multiple of the workgroup size. Sweep on the 780M like the `distance_field_2d` WG-shape sweep before pinning.
- [ ] **T2 shell appearance — flat tint vs SDF-displaced.** v1 ships the flat ground-tint surface splat (the cheap, correct cost-collapse). The "SDF-displaced shell" (a height-perturbed surface reading the terrain/density field for a grass-texture micro-relief) is the quality upgrade — sequence it after M2 proves the flat shell's cost, with the displacement read from the same density field.
- [ ] **Wind field source.** v1 uses an engine-internal analytic sine-sum wind. Whether the game drives amplitude/direction (weather) through a uniform, or the engine exposes a `set_wind(dir, strength)` — decide at M3 (engine stays generic either way; it is a uniform).
- [ ] **Near-field overdraw at street level.** A camera lying in the grass sees deep blade overdraw (the chain's known cost driver). Does the near-blade density need a coverage-aware throttle (fewer, fatter blades when the camera is in the field)? Measure at M2's street camera; the governor already sheds, but blade overdraw may need its own near-density curve.

---

## 9. Out of Scope

- **Changing the frozen four-pass tiled chain** (`tile_assign`/`radix_sort`/`tile_range_build`/`splat_raster`) — scatter composes via `splat_buf` writes exactly like `ExpandDrawsPass`/`GiCombinePass`. The radix-raster equal-key nondeterminism landmine (memory `aaa-phase2-resident-frame`) is inherited, not introduced: scatter's determinism gate asserts at the BUFFER seam (the generated `splat_buf` bytes), never on framebuffer pixels.
- **The Spectra path tracer** — scatter feeds the interactive wgpu chain only; routing a grass field through Spectra stills is a later bridge (the `spectra-pathtracer-runs` staged plan), not this design.
- **Authoring tools for the density field** — the engine consumes a 2D field; how the game paints/derives grass density (biome rules, lawn zoning) is game work.
- **Collision / gameplay interaction with blades** (mowing, trampling, hiding in grass) — the field is render-only here; an interactive density-field edit is a future scatter-class feature.
- **GI from the scatter set** — `ResidentGiRaster` keeps working on whatever is in `splat_buf`; tuning GI for a 150k-blade field is not addressed.
- **Snow/sand/water-surface scatter** — same machine in principle, but their primitives (not vertical blades, not cooked prototypes) are their own follow-ups.

---

## 10. Related Plans / Designs

- Depends on: [Hybrid Atom+SDF LOD](./2026-06-12-hybrid-atom-sdf-lod-design.md) (the PINNED T0/T1/T2 tier contract + the one `FrameBudgetGovernor` — this design references, never redefines, them), [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (§4.5 `ExpandDrawsPass`, §4.6 `new_with_capacity` capacity, §4.7 governor — the producer→chain seam scatter reuses), the [M3.2 plan](../plans/2026-06-11-virtualized-splat-rendering-m3-2.md) (`encode_indirect` + device-emitted draws + one-submit/one-64B-readback — the indirect-dispatch shape `ScatterField::encode` copies), [SDF Pillar](./2026-06-10-sdf-pillar-design.md) (§4.6 `DistanceTransform2d` density-field substrate + §4.2 Tier-2 clearance-SDF clipmap — the two mask gates).
- Required before: the Scatter Field implementation plan (M1 generation → M2 LOD/cost-collapse → M3 mask+wind, one plan per milestone), which slots after the hybrid-LOD tier contract and the SDF clearance clipmap land.
- Related: `hybrid-atom-sdf-lod-direction` memory (the 2026-06-12 directive this implements: "minimize the cost of scattering", "billion-blade field at fixed budget, never billion per frame"); the AMD work-graph / procedural-placement-on-terrain industry patterns (cited in §1/§4.1 as the triangle-mesh counterpart this splat-native design beats on AA + LOD + imposter-authoring); Mip-Splatting / Multi-Scale-3DGS / LOD-GS (the published basis for the free sub-pixel splat AA claim in §4.4).
```
