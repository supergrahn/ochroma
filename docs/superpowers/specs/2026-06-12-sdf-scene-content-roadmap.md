# Design: SDF Scene-Content Roadmap — housing, water, vegetation in ONE path tracer under ONE ray budget (2026-06-12)

**Status:** Draft
**Scope:** Reconcile the three SDF-content tracks — [SDF Housing Quality](./2026-06-12-sdf-housing-quality-design.md) (Track A), [SDF Water](./2026-06-12-sdf-water-design.md) (Track B), and the [FloraPrime audit + foliage decision](./2026-06-12-floraprime-audit.md) (Track C) — into one roadmap: one scene model that a single Spectra megakernel ray resolves, one per-frame ray/sample budget **B** split across the three content classes, one build sequence, and one honest statement of what degrades first when B cannot afford all three at full quality. **B is TBD** — a separate perf agent ([Spectra Realtime](./2026-06-10-spectra-realtime-design.md)) is determining it on the megakernel right now; every number here is parameterized on B and nothing in this doc may restate or presume the realtime millisecond ladder.
**Related:** [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (the one-path-tracer contract all three tracks render into), [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) (`SdfBaker`/`SdfAtlasGpu`, the 64 MB atlas quota), [Scatter Field](./2026-06-12-scatter-field-design.md) (the placement/governor machine Track C reuses), [Terraform Tool](./2026-06-11-terraform-tool-design.md) (the `water_flips` signal Track B couples to), wave-1 plans: [housing](../plans/2026-06-12-sdf-housing-quality-wave1.md), [water](../plans/2026-06-12-sdf-water-wave1.md), [flora](../plans/2026-06-12-floraprime-vegetation-plan.md).

---

## 1. Problem Statement

- Three tracks were designed in parallel and each is individually budget-parameterized on B, but **no document states how they share one B**. Each track's worst case alone can eat the whole frame: housing's L2 GI tier is ×3.4 frame rays, water at grazing-over-shallows is k_w ≈ 1.0 extra traversals per water sample, foliage's alpha-test anyhit was 3.58 ms on an RTX 5080 *with* Opacity Micromaps (Indiana Jones anchor). Summed naively they need an effective ~5 traversal-equivalents/pixel — likely several times what the perf verdict will grant.
- Two of the three classes are **transparency cost amplifiers**: a water hit and a leaf-card crossing both turn one ray into a chain (Fresnel continuation; N alpha tests + translucent continuation). Without a shared cap and shared per-class counters, the first integrated scene (buildings + shore + trees) will blow the budget in a way no single track's gates can detect.
- The three tracks **share unstated infrastructure**: the k-NN atom material gather (Track A Task 1's cell grid) is also the resolver for water-refracted bottom hits and SDF wood hits; the 64 MB `SdfAtlasGpu` quota is consumed by buildings AND tree wood AND the water-harness bottom; the terrain heightfield's CSG zero-set is simultaneously the water shoreline, the housing foundation, and the vegetation ground mask; one cook must deterministically produce all three. Built independently, these collide.
- Four parties edit `megakernel.slang` concurrently (perf agent, Track A textured-hit work, Track B water branch, Track C foliage/shell work). Without an explicit serialization order the merges collide on the same hit-dispatch code.

---

## 2. Done When

Running `cargo run -p vox_app -- --shot-waterfront` on the `tidewater_flats` map (after Waves 1–2 below) produces `shot_waterfront.png` showing, in one frame from one path tracer: a **textured** craftsman block (visible roof-vs-facade material split, muntined windows), **mirrored in Gerstner-perturbed water** with see-through shallows and a foam band, with **at least one backlit tree** showing sky-gaps through its canopy — and prints one budget line of the form
`frame_budget: B=<verdict> opaque=<n> water=<n> (cap 0.25B) foliage=<n> (cap βf·B) total=<n> ≤ B  tier=L<k>`
where every counter is a real measured traversal count and `total ≤ B` holds. A human at the keyboard verifies the mirror, the shallows, and the lacy canopy without reading code.

(Until B is set this command runs with `B=∞` and the line prints `B=UNSET (perf verdict pending)` — the counters are still real, which is exactly what hands the perf agent its ground truth.)

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| One closest-hit across all three classes | Ray aimed at building-behind-water returns water t (analytic) < building t (sphere-trace); `assert!(hit.t_water < hit.t_sdf && hit.class == WATER)` with real scene data | `assert!(hit.is_some())` |
| Per-class traversal counters | `g_class_stats` printed each frame satisfies the identity `total == primary + opaque_light + water_cont + foliage_cont` and `water_cont <= 0.25 * B` with water enabled at f_w ≈ 0.3 | counters exist, all zero |
| Shared material gather | Same cell-grid gather invoked from opaque-building hit, water-refracted bottom hit, and SDF-wood hit returns identical material for identical world position: `assert!(max_delta < 1.0/255.0)` | three separate gather functions each `return default_material()` |
| Budget degradation ladder | Forcing `B` to ¼ of measured full-quality spend flips, in order: `tier L2→L1`, `d_v` inward (printed), water `mode=reduced_tlas` — and the frame still completes under the forced B | a `degrade()` fn that is never called |
| Unified cook | One cook invocation on `tidewater_flats` emits buildings (SDF+atoms+materials+rects), water fields (3× R32F 256² + heights), and flora (SDF wood + card sets + PROSPECT table) with a printed sha256 per artifact, byte-identical across two runs | cook exits 0 |

---

## 4. Architecture

### 4.1 The unified scene model — what one ray resolves

One megakernel ray performs a single closest-hit query across four primitive kinds, taking the minimum t. There is exactly **one stochastic continuation per path vertex** regardless of class — Fresnel choice for dielectrics, transmittance roulette for leaves — never ray splitting, so cost grows bounce-multiplicatively, not exponentially.

| Class | Geometry primitive | Intersection cost | Hit shading | Transparency behavior |
|---|---|---|---|---|
| **Housing** (Track A) | Instanced SDF volumes in `SdfAtlasGpu` (snorm8 bricks) | Sphere-trace, 100–384 steps × 8 trilinear loads (C_T ≈ 0.8k–3k loads) | Cell-grid k-NN atom gather → channel→`MaterialData` dispatch → box-UV PBR (forge_box_uv convention, 2.5 m/tile) + procedural muntins/bevels | **Opaque**, except cook-derived aperture rects → M2 glass: ONE continuation ray, interior-mapped (cheaper than re-trace) |
| **Water** (Track B) | Analytic mask-clipped plane: `t=(h−ro.y)/rd.y` + ≤4 texel reads — **no march** | ~10 ALU (detection is free) | Gerstner 4-octave **normal** perturbation (~80 ALU, zero rays) + shore fields (foam/wet/depth, 0 rays) | **Dielectric**: `sample_glass` (brdf_glass.slang:227, IOR 1.333) verbatim — ONE Fresnel-chosen continuation + `beer_lambert` (:77); deep cells skip refraction entirely |
| **Vegetation wood** (Track C) | QSM cylinders baked to SDF via `SdfBaker` — **same atlas, same sphere-trace as buildings** (~183 KB/asset) | Same as housing | Same cell-grid atom gather as housing | Opaque |
| **Vegetation foliage** (Track C) | Alpha-tested leaf cards — `MAT_VEGETATION` (megakernel.slang:2291–2371), **the ONE justified non-SDF exception** | BVH anyhit: leaf-shape opacity test, 30–60 ALU each, L̄ ≈ 8–30 overlaps/ray near a hero tree | PROSPECT-PRO 8-band double-sided translucent BSDF (`leaf_transmittance`, LEAF_* constants byte-identical to FloraPrime's `LeafGeometry::shader_code()`) | **Translucent thin surface**: transmittance-rouletted continuation; far canopy collapses to a **volumetric green shell** (one march/instance, O(1) per far tree) |
| **Terrain** | Heightfield SDF; the CSG `max(water_level, −terrain)` zero-set | Sphere-trace / analytic | Atom gather + biome | Opaque; shared substrate for all three (see §4.5) |

Resolution rule: analytic water t is computed in closed form and min'd against the SDF-instance sphere-trace and the leaf-card BVH hit; the water mask clip and the deep-water cutoff are applied at detection so deep water never spawns a refraction. The leaf-card anyhit runs on **every** segment that crosses canopy — including building-lighting rays and water-reflection rays — which is why foliage is accounted as a traversal *tax*, not just its own rays (§4.2).

Bounce floor is a **scene-level** decision, not per-track: water on ⇒ `max_bounces ≥ 3`; water reflecting M2 glass windows ⇒ ≥ 4 on those paths (the M2 host already forces ≥3).

### 4.2 The shared budget split — one B, three claimants

Let B = the perf verdict's total scene traversals/frame, r = B/(P·S) traversals/pixel/frame. Accounting convention: a traversal is charged to the surface class **that spawned it** (anyhit alpha-test work is charged to foliage as traversal-cost-equivalents even though the ray belongs to another class).

Default split at the perf tier (RTX 4070 Ti; the 780M floor runs the same code at seconds/frame, correctness-first):

| Class | Default share of B | Mechanism | Why this share |
|---|---|---|---|
| Opaque housing: primary visibility + lighting tier | **~50%** (the residual after caps) | Lighting tier = the dial: L1 NEE sun ×1.8, L2 +1 GI bounce ×3.4, L3 ×5 frame rays; pick the **largest tier whose base R fits the residual**. All C_hit shading (textures, normals, muntins, interior mapping) is per-hit ALU/loads, ~0 rays, runs unconditionally | This is the baseline image at every pixel; its lighting rays also serve water bottoms and foliage NEE |
| Water continuations | **≤ 25% hard cap** (W_max = β_w·B, β_w = 0.25); typical spend 5–15% | Unbiased Russian roulette on the continuation (throughput /= p); ΔW ≈ f_w·P·S·k_w·C_T with k_w = F̄ + 0.3·(1−F̄)·shallow_share ∈ [~0.1 … ~1.0]; the deep-cutoff is a net saver | The cap bounds the grazing-over-shallows worst case; the headline (city mirrored in water) lives inside this share |
| Foliage: anyhit tax + translucent continuations | **≤ 25%** (β_f·B) | `FrameBudgetGovernor` sets the card→volumetric-shell distance d_v so integrated L̄·(c_at_eff + c_bvh) ≤ β_f·B; c_at_eff ≈ 0.45·c_at only if OMM exists (perf agent's call — on pure software BVH assume 2–5× worse) | Foliage is the only class whose cost is a multiplier on OTHER classes' rays; it must be governed continuously, not capped stochastically |

**The two amplifiers, named:** water and foliage are the transparency cost amplifiers. Water turns every shallow/grazing sample into a second full scene traversal; foliage turns every canopy-crossing segment into 8–30 alpha tests plus a possible translucent continuation. Both concentrate cost exactly where the image lives (grazing reflections, backlit canopy edges) — so neither can be budgeted by average-case math; both ship with mandatory per-frame printed counters (the water `g_water_stats` pattern generalized to `g_class_stats`).

**The honest sum.** Full quality on all three simultaneously ≈ L2 housing GI (×3.4) + full water (its own ladder wants r ≥ 3 on water pixels, +0.3–0.5 r scene-wide at f_w ≈ 0.3) + hero foliage (×1.2–1.5 tax on canopy rays + continuations) ⇒ **effective r ≈ 4.5–6**. If the verdict lands near r ≈ 1–2 (the plausible 1–2 ms HW-RT outcome), **the budget affords roughly two of the three at full quality**, with temporal reuse (ReSTIR/denoiser — owned entirely by the realtime design, not this roadmap) closing part of the remaining gap. This doc does not pretend otherwise.

### 4.3 The degradation ladder — what degrades first

Per-hit shading **never degrades** (textures, normal maps, muntins, interior mapping, Gerstner normals, foam/wet fields, PROSPECT spectral response — all ~0 rays). Ray spenders degrade in this order, first to last:

1. **Housing GI depth** (L3→L2→L1): least visible loss; the image still reads with NEE sun only. This is the first dial turned because it frees the most rays (×5→×1.8) for the least perceptual damage.
2. **Foliage d_v pulled inward**: continuous and graceful — hero card density survives near camera, mid-distance canopies coarsen to the volumetric shell earlier; grass collapses to shell first. Loses lacy mid-distance silhouettes, keeps near realism and far O(1).
3. **Water reflections step down the ladder**: full TLAS → selector-reduced TLAS reflections (bottoms direct-light-only) → rouletted continuations clamped to the cap. Defended longer than foliage mid-field because the mirrored city is the headline differentiator vs CS2's screen-space fakes.
4. **Water analytic fallback** (Fresnel·sky + Beer-Lambert tint + foam, 0 extra traversals): the r ≈ 1 row. This is **not a silent degrade — it is an explicit renegotiation trigger**; if the verdict forces this row, the water headline is OFF and that fact goes back to the perf/product conversation as a printed number, not a discovered-in-production surprise.
5. **NEE sun** (L1→L0): effectively never — one shadow ray per vertex is the cheapest, highest-value ray in the frame and the last thing cut.

### 4.4 Build sequencing — everything gates on the perf verdict

**Gate G0 (hard, external):** the perf agent (a) frees `megakernel.slang` (it is editing it now — all kernel work below rebases onto its head) and (b) sets B. Two sub-gates inside G0: Track A's kernel tasks land behind the `u_sdf_textured` uniform (default-off, existing renders byte-identical) and may rebase early; Track C's megakernel edit (Task 3, shell + governor) must additionally wait for **Universal Renderer M0** to land the `MAT_VEGETATION`/`g_splat_buffer` bind, or it collides.

Given G0, order by cheap + high-value first:

**Wave 1 — zero or negative ray cost (start immediately on the freed kernel; B's value not yet needed):**

| # | Item | Ray cost | Why this position |
|---|---|---|---|
| 1 | **A-Task1: cell-grid k-NN gather** | **NET NEGATIVE** — replaces the ~49k-load linear scan (~450× cut, 16–60× one C_T per hit) | Must land first, full stop: every track's hits (buildings, water bottoms, wood) ride this resolver. Gate: parity <1/255, ≥20× printed speedup |
| 2 | A-M1: textured building (box-UV PBR via existing `set_texture_atlas`/EWA) | 0 extra rays (~12 loads/hit bilinear) | Biggest visible win per ray spent: zero |
| 3 | A-M2: detail crispness (TBN normals, aperture rects, muntins, interior-mapped glass) | 0 rays; transmitted glass rays get **cheaper** | Interior mapping saves a full C_T per glass continuation |
| 4 | C-M0: FloraPrime wood → SDF via `SdfBaker` | 0 runtime (cook-side) | Same baker/atlas as buildings; unblocks everything in Track C |
| 5 | **W0–W2: dielectric water + Gerstner + Beer-Lambert/shore fields + instrumentation** | W0 spends +1 continuation per water sample (k_w ≈ 1.0); W2's deep skip drops k_w to ~0.1–0.5 — a net saver vs W0 | The mechanism is proven (M2 glass verbatim); W2's Task-4 counters and the printed r ∈ {1, 1.5, 2, 3} ladder are **the data the perf agent's B-setting needs** — building the instrumentation early feeds the verdict rather than waiting on it |

**Wave 2 — the ray spenders (require B's value to pick tiers/levers):**

| # | Item | Gate on B |
|---|---|---|
| 6 | A-M3: lighting tiers (NEE + tiered GI replacing the headlight/matcap terminate) | Pick largest tier with base R ≤ residual share; serves all three classes' light |
| 7 | C-M1: PROSPECT leaf cards through `MAT_VEGETATION` | First real anyhit spend; needs Universal Renderer M0 bind landed |
| 8 | C-M2: volumetric shell + d_v governor | **Must land before any dense scatter** — it is the foliage budget-enforcement valve |
| 9 | W3: game-water unification (`--shot-water` on tidewater_flats, 65536/65536 render==sim cell agreement) | Needs A-M1/M2 buildings for the mirrored-city headline; measures real f_w |
| 10 | **Integration shot `--shot-waterfront` (§2)** | First frame where all three classes share one measured B |

**Wave 3 — scale + budget-gated stretch:**

| # | Item | Notes |
|---|---|---|
| 11 | A-M4: 100 archetypes, one atlas | Cook-time only; unified ledger gate (§4.5) |
| 12 | C-M3: Scatter Field reconciliation (blade → strap-leaf card) + C-M4: FloraPrime replaces forge in `cook_flora` (sha256-deterministic, forge demoted to `--preview-forge`) | Scatter machine unchanged in structure |
| 13 | A-M5 detail-SDF slabs; W-v1.5 terraform flood/drain coupling (<1 ms field re-cook on `water_flips`) | A-M5 built only if B affords +10–20 steps on 2–5% of pixels; killed without ceremony if M2 bevels read |
| 14 | v2 seams (named, not built): spectral σ(λ) water, FFT ocean, geometric wave displacement, `fluid.rs`/`pbf.rs` → SDF surfacing, caustics via photon/cache tier only, trained flora ML | Separate designs |

### 4.5 Cross-track dependencies — the shared infrastructure, owned once

| Shared thing | Owner | Consumers | Collision risk if unowned |
|---|---|---|---|
| **Cell-grid k-NN atom material gather** (A-Task1) | Track A | Building hits, water-refracted bottom/terrain hits, SDF-wood hits | Three gathers with three parities; also the single point where the HW-RT mitigation (primary-hit G-buffer material resolve, atom-colour-only on secondary bounces) would be retrofitted **for all three at once** if the verdict makes per-vertex gather unaffordable |
| **`SdfAtlasGpu` 64 MB quota + one resident ledger** | SDF Pillar | Buildings 19–29 MB, wood ~183 KB/asset (21 species ≈ 4 MB), water-harness bottom ~1 MB, terrain | Each track's individual ledger passes while the union busts the quota. One printed ledger, all classes, every cook. Total new resident across tracks ≈ 100–120 MB on the 780M UMA (vs the ~370 MB splat chain this path replaces) |
| **Terrain CSG zero-set** `max(water_level, −terrain)` + `CellClass::Water` mask | Terraform/terrain | Water shoreline + clip field, housing foundations, vegetation ground/scatter mask, sim coverage geodesics | Render-water vs sim-water vs scatter-exclusion disagreeing on the same cell. The mask is THE single field by construction (W3's 65536/65536 gate is the proof) |
| **One cook** | `game_asset_cook` | Buildings (`.asset.json` directives → SDF+atoms+materials+rects), water fields (3× R32F 256² + heights, sub-ms re-cook), flora (FloraPrime QSM → SDF wood + canonical card sets + PROSPECT table) | Texture/material conventions drift (the box-UV convention at the SDF hit MUST be the cook's `forge_box_uv` exactly); determinism gate is sha256 per artifact |
| **Dielectric machinery** `sample_glass`/`beer_lambert` | Spectra (proven M2) | Building window apertures AND water — verbatim, one implementation | Bounce-floor interaction: water reflecting glass needs ≥4 bounces; a per-track floor would silently truncate the other's paths |
| **Budget instrumentation** `g_class_stats` (generalizes W-Task4's `g_water_stats`) | This roadmap | All three + the perf agent | Without per-class counters the §4.2 split is an assumption, not a measurement |
| **`megakernel.slang` serialization** | Perf agent (now) → A (uniform-gated) → Universal-Renderer M0 → C-Task3 | All kernel edits | Concurrent edits to the same hit-dispatch; the order above is mandatory |
| **`FrameBudgetGovernor`** | Virtualized-splat/Scatter lineage | Foliage d_v lever (C-M2); the same mechanism family as water's roulette cap | Two governors with two notions of "budget"; both must read the same B and the same counters |

---

## 5. Data Models

```rust
/// One per-frame budget object, set once from the perf verdict, read by every class.
/// Lives host-side; uploaded as uniforms. No pub fields.
pub struct FrameRayBudget {
    total_b: u64,          // B: traversals/frame — perf agent's verdict (u64::MAX until set)
    beta_water: f32,       // hard water continuation cap share (default 0.25)
    beta_foliage: f32,     // foliage anyhit-tax + continuation share (default 0.25)
    lighting_tier: LightingTier, // L0..L3 — largest tier whose base R fits the residual
    d_v_m: f32,            // card→volumetric-shell distance, governor output
}

impl FrameRayBudget {
    pub fn residual_opaque(&self) -> u64; // B − caps actually spent last frame (opaque may borrow unspent)
    pub fn from_verdict(b: u64) -> Self;  // applies §4.3 ladder to pick tier + d_v
}

/// GPU-side per-frame counters (atomics), printed every frame. Identity:
/// total == primary + opaque_light + water_cont + foliage_cont (+ skipped_deep is informational).
#[repr(C)]
pub struct ClassStats {
    primary: u32,
    opaque_light: u32,   // NEE + GI traversals charged to the lighting tier
    water_cont: u32,     // must satisfy water_cont <= beta_water * total_b
    foliage_cont: u32,   // continuations; anyhit ALU charged as traversal-equivalents
    foliage_anyhit: u32, // raw alpha tests (for the L̄ measurement)
    skipped_deep: u32,   // deep-water refraction skips (net-saver evidence)
}
```

---

## 6. API

The roadmap introduces no new public engine API beyond `FrameRayBudget`; it binds the tracks to APIs that already exist and that the wave-1 plans carry in their IMPORTANT NOTES:

```text
// Reused verbatim — implementations must NOT fork these:
sample_glass(...)            // ~/src/spectra/slang/brdf_glass.slang:227 — windows AND water
beer_lambert(...)            // brdf_glass.slang:77 — water depth absorption, σ=(0.45,0.12,0.06)/m
MAT_VEGETATION hit branch    // megakernel.slang:2291–2371 — leaf cards, opacity test, PROSPECT BSDF
set_texture_atlas(...)       // Spectra flat texture atlas + EWA (spectra-texture-bridge memory)
SdfBaker / SdfAtlasGpu       // SDF Pillar — buildings AND QSM wood bake through the same baker
forge_box_uv (2.5 m/tile)    // game_asset_cook.rs:1711 — THE UV convention; the SDF hit replicates it exactly
CellClass::Water / WaterMask // the one clip field: render water == sim water == scatter exclusion
FrameBudgetGovernor          // scatter lineage — drives d_v in C-M2
// Threading: FrameRayBudget is set on the render thread before encode; ClassStats read back next frame.
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `FrameRayBudget::from_verdict` | render-loop init + on perf-verdict change | host frame driver (the perf agent's megakernel host) | once; re-derived if B changes |
| `ClassStats` atomics | every megakernel path-vertex branch (opaque/water/foliage) | `~/src/spectra/slang/megakernel.slang` | behind the same uniforms as each feature; zero cost when class disabled |
| Budget line print (§2) | end of frame, after stats readback | shot harness / frame driver | the human-visible contract |
| Water roulette cap | water continuation spawn | megakernel water branch (W-Task4) | unbiased: throughput /= p |
| d_v governor | scatter tile build, per frame | `FrameBudgetGovernor` call site (C-M2) | reads `foliage_anyhit` from last frame |
| Unified cook manifest + sha256 | end of cook | `game_asset_cook` / `cook_flora` / water field cook | one manifest, three sections |
| Unified resident ledger | asset upload | atlas/library upload path | one printed ledger across all three classes |

---

## 8. Open Questions

- [ ] **B itself** — the perf verdict (value, HW-RT vs software BVH, OMM availability). Everything in §4.2–§4.4 is parameterized on it; Wave 2 cannot pick tiers without it.
- [ ] If the verdict is HW-RT at ~1–2 ms: does the per-vertex material gather move to the primary-hit G-buffer resolve (Track A names the mitigation; wave 1 does not build it)? Decision needed before Wave-2 item 6.
- [ ] rgba8 quantization of the PolyHaven texture sets (4× memory saving vs f32) — Track A open question; affects the unified ledger.
- [ ] The empirical card density that reads photoreal at d_v (Track C's "only the cost milestone resolves this") — sets whether β_f = 0.25 is generous or tight.
- [ ] Choppy-water roughness clamp until the denoiser lands (Track B secondary risk) — owned by realtime, but the clamp value is set here.
- [ ] Default split 50/25/25 — revisit after the first `--shot-waterfront` counters; opaque borrowing unspent water budget is allowed, the reverse is not (caps are hard).

---

## 9. Out of Scope

- The realtime millisecond ladder, ReSTIR, radiance caching, neural reconstruction, OMM implementation — owned by [Spectra Realtime](./2026-06-10-spectra-realtime-design.md); this doc only consumes its verdict B.
- Caustics (deferred to the photon/cache tier in Track B's own words), FFT ocean, geometric wave displacement, spectral σ(λ) through the 16-band film, `fluid.rs`/`pbf.rs` SDF surfacing — v2 seams, named in §4.4 item 14, not designed here.
- The untrained FloraPrime ML scaffold (`floraprime_gen`) — not on the production path.
- Re-litigating any single track's internal design decisions (leaf-card vs SDF foliage, analytic plane vs marched water, interior mapping vs hollow glass) — those are decided in the track docs; this doc only reconciles them.

---

## 10. Related Plans / Designs

- Depends on: [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (Gate G0: B + the freed megakernel), [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (M0 bind before C-Task3).
- Reconciles: [SDF Housing Quality](./2026-06-12-sdf-housing-quality-design.md), [SDF Water](./2026-06-12-sdf-water-design.md), [FloraPrime Audit](./2026-06-12-floraprime-audit.md).
- Drives ordering of: [housing wave 1](../plans/2026-06-12-sdf-housing-quality-wave1.md), [water wave 1](../plans/2026-06-12-sdf-water-wave1.md), [flora plan](../plans/2026-06-12-floraprime-vegetation-plan.md).
- Related: [SDF Pillar](./2026-06-10-sdf-pillar-design.md), [Scatter Field](./2026-06-12-scatter-field-design.md), [Terraform Tool](./2026-06-11-terraform-tool-design.md), [AI Asset Factory](./2026-06-10-ai-asset-factory-design.md).
