# Design: SDF Water — water as an analytic SDF primitive in the Spectra path tracer, dielectric by construction, budget-parameterized (2026-06-12)

**Status:** Draft
**Scope:** Give the SDF-only Spectra runtime a WATER primitive: a near-flat, mask-clipped, analytically-intersected water-level SDF whose hits run the SAME dielectric mechanism as the proven M2 glass windows (Fresnel reflect / Snell refract / continuation ray / Beer-Lambert absorption), with procedural wave NORMALS (no remeshing, no simulation in v1), CSG shoreline + shore-distance foam/wet-edge from the game's water mask, and a HARD, perf-tier-parameterized per-frame water-ray budget — because water is the path tracer's most expensive mode and this design refuses to be budget-blind about it.
**Related:**
- Builds directly on: [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (SDF-only direction; the M0–M2 ladder this extends), the `hybrid-atom-sdf-lod-direction` memory (SDF is the single runtime geometry primitive).
- Reuses verbatim: the M2 glass mechanism — `~/src/spectra/slang/megakernel.slang:1266–1338` (SDF glass hit → `sample_glass` → continuation ray that transmits instead of terminating) and `~/src/spectra/slang/brdf_glass.slang` (`sample_glass` :227, `beer_lambert` :77, `fresnel_dielectric`, TIR handling :361–372). PROVEN on the AMD 780M (spectra `1480e2c`, ochroma `160469b`: 1063 see-through window pixels, mean |ΔRGB| 395 vs the opaque control).
- Couples to: [Terraform Tool Wave 1](../plans/2026-06-11-terraform-tool-wave1.md) (**landed** — `MapTerrain::apply_stamp` → `rederive_region` → `water_flips` → `childcare_field_dirty`, verified in `~/Ochroma/projects/urban_horizon/src/map/terrain.rs:122–150` and `derive.rs:43–72`), [Terraform Tool Design](./2026-06-11-terraform-tool-design.md).
- Budget contract: the per-frame ray/sample budget **B** is set by the Spectra perf agent's verdict (in flight on the megakernel right now). Every affordance in §4.6 is parameterized on B — nothing here assumes a number that agent hasn't measured. Floor truth: ~9.3 s/frame full-scene on the 780M software sphere-trace; target 1–2 ms on the RTX 4070 Ti ([Spectra Realtime](./2026-06-10-spectra-realtime-design.md)).

---

## 1. Problem Statement

Concrete, observable symptoms (every claim verified against code 2026-06-12):

- **Water today is a flat dark-blue splat carpet — no reflection, no transparency, no waves, no shore.** The game's only water rendering is the terrain splat layer's class tint: `~/Ochroma/projects/urban_horizon/src/render_gpu/mod.rs:334/3811` — `let boost = if z.surface_type == "water" { 3.0 } else { 5.0 }`, a darker-blue Gaussian per cell. CS2-class water (even with its screen-space fakes) is strictly ahead of this.
- **The sim's water is load-bearing and the picture ignores it.** `CellClass::Water` / `WaterMask { bodies, sea_level }` (`src/map/layers.rs`) gate buildability (`derive.rs::classify_cell` — Water if `hm.sample ≤ sea_level` OR inside a `WaterBody.outline`), sever childcare coverage geodesics (`coverage_field.rs:111` — water cells are 255/impassable), and flood/drain under the landed terraform tool (`apply_stamp` → `rederive_region` returns `water_flips`; `end_stroke` dirties the childcare field). The render reads NONE of this — sim water and picture water are two unrelated artifacts that can silently diverge.
- **The Spectra SDF runtime has a proven dielectric but no water primitive.** The M2 glass branch (`megakernel.slang:1266–1338`) is exactly water's light transport — a hit that scatters (Fresnel reflect / Snell refract via `sample_glass`) and CONTINUES the path instead of terminating — and `beer_lambert` (`brdf_glass.slang:77`, `T = exp(-σ·d)`) is water's depth absorption, already implemented. But the only SDF geometry the megakernel traces is the baked snorm-volume atlas (`sdf_render_intersect`, up to `MAX_STEPS = 384` trilinear steps per ray, min over ALL instances per step — `megakernel.slang:687–744`). Baking a dynamic water plane into that machinery would be absurd: a flat surface needs ZERO march steps, and a flood edit would re-bake a volume.
- **Nobody has stated water's ray cost, and it is the budget crux.** Water is the path tracer's most expensive mode: every water sample spawns a continuation ray (transparency does not terminate the path) and raises the required bounce floor from 2 to ≥3 (camera → water → scene → light; M2's host already forces `max_bounces >= 3` for exactly this reason, `splat_backend.rs:1205–1206`). A design that promises "correct reflections of the dynamic city" without per-ray arithmetic against the perf agent's budget B is vapor. §4.6 is that arithmetic.

The throughline: **water is the SDF/path-trace sweet spot — a dynamic implicit surface (no remeshing for waves/ripples/flood), CSG'd against the terrain field the game already owns (free shore distance), with physically correct light transport the tracer already proved on glass — IF and only IF its continuation-ray cost is hard-budgeted per perf tier.**

---

## 2. Done When

**Engine (v1, the 780M floor — correctness at seconds/frame, NO millisecond gate):**

Running

```bash
cd ~/src/ochroma && cargo test --release -p vox_render --features spectra-native \
  sdf_water -- --nocapture
```

prints (real measured values, this shape):

```
[sdf-water] scene: bottom_sdf=1 buildings=2 water_planes=1 grid=128x128 @ 2.0 m
[sdf-water] water px: 29841 (f_w=0.29)  reflection-evidence px: 2417 (gate > 800)
[sdf-water] shallow bottom visible: 0.71 (gate > 0.50)  deep mean transmittance: 0.034 (gate < 0.05)
[sdf-water] deep B/R: 5.8 vs shallow B/R: 1.9 (gate deep > 2x shallow — Beer-Lambert is spectral-shaped, not a grey fade)
[sdf-water] foam px: 1862, max |shore_dist| = 4.7 m (gate: all inside the 6.0 m band, count > 200)
[sdf-water] waves: t=0.0 vs t=1.5 differ on 18233 water px (gate > 0.3 * water_px)
[sdf-water] rays: continuations=29841 reflect=3110 refract=11850 skipped_deep=14881
[sdf-water] cost: water frame 8.1 s vs dry frame 4.3 s = 1.88x (gate <= 2.5x)
wrote sdf_water_calm.png / sdf_water_choppy.png / sdf_water_opaque_control.png
```

and a human looking at `sdf_water_calm.png` sees: a shoreline scene where the water **mirrors the red-roofed building** (absent in `sdf_water_opaque_control.png`), the **bottom shows through in the shallows** and fades to deep blue-green with depth, and a **white foam band hugs the shore**. The opaque control is today's flat dark-blue look, side by side.

**Game (W3 — the unification still on a real map):**

```bash
cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play -- --shot-water water_shots
```

prints `[water] mask unification: render==sim on 65536/65536 cells (CellClass::Water is THE clip field)`, `[water] tidewater_flats: planes=<N> water cells=<W> shore cells=<S>`, the same `[water] rays: …` accounting line, and writes `water_shots/water_tidewater.png` — the game's actual map water, path-traced, reflecting the game's actual shore buildings. A human verifies all of it without reading code.

**Realtime (1–2 ms) is explicitly NOT in this doc's Done When.** §4.6 parameterizes every quality affordance on the perf agent's budget verdict; the realtime gate itself is owned by [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) and validatable only on `tomespensin`.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Water is a real dielectric, not a tint | render the shore scene twice (dielectric vs opaque-Lambert control): `assert!(reflection_evidence_px > 800)` where evidence = water pixels in the shore building's hue band that are absent in the control; both counts printed | `assert!(water_pixels > 0)` |
| Refraction shows the real bottom | shallow band (depth < 2 m): `assert!(bottom_visible_fraction > 0.5)` — bottom-material hue detected through the surface; printed | `assert!(image_not_black)` |
| Beer-Lambert is depth- and wavelength-correct | `assert!(deep_transmittance < 0.05)` at depth > d_cut AND `assert!(deep_b_over_r > 2.0 * shallow_b_over_r)` — red dies first, exactly like real water | asserting deep merely darker |
| Waves animate without remeshing | frames at `u_water_time` 0.0 and 1.5: `assert!(changed_water_px > 0.3 * water_px)`; zero geometry re-upload between frames (`assert_eq!(scene_uploads, 1)`) | comparing two identical frames |
| Shore foam from the SDF shore distance | every foam-classified pixel maps to `|shore_dist| < foam_width`: `assert!(max_foam_shore_dist < 6.0 && foam_px > 200)` | drawing a fixed white ring |
| The water-ray budget is hard and printed | `WaterFrameReport`: `assert!(skipped_deep > 0)` (the deep cutoff fires) AND `assert!(water_frame_s <= 2.5 * dry_frame_s)`; counters printed every run | no instrumentation |
| Mask unification (game) | per-cell: `(water_cell_height > NEG_INF) == terrain.is_water(centre)` over all 256² cells: `assert_eq!(agree, 65536)` | sampling 3 hand-picked cells |
| Off = byte-identical | `u_water_enabled = 0` (default): M0/M1/M2 acceptance tests (`sdf_building_renders_solid_surface` 0.976, `sdf_city_block` 12/12, `sdf_craftsman_atom_material_and_glass`) stay green unmodified | only testing the new path |

---

## 4. Architecture

### 4.1 Why water is THE SDF/path-trace sweet spot (stated, with the code that proves each leg)

1. **Dynamic implicit surface — no remeshing, ever.** A water surface is `d(p) = p.y − h_w` clipped by a 2D mask. Waves perturb the *normal* (or later the height function), never a mesh; a flood edit changes a 2D field, never a BLAS. Raster water pays per-vertex update or FFT-bake every frame; an implicit surface evaluated only where rays land pays only at hits. The acceleration-structure rebuild problem that makes dynamic water hostile to triangle ray tracing (the caustics literature names "dynamically computed, tessellated or frequently topologically changed (e.g. water)" geometry as the AS-rebuild pain) simply does not exist for an analytic SDF.
2. **CSG with the terrain field the game already owns.** Water volume = (below the surface plane) ∩ (not inside terrain): `d_water = max(p.y − h_w, −d_terrain)` — the shoreline is the zero-crossing of that max, and the **shore distance falls out for free** as the 2D signed distance to the mask boundary (cooked from `CellClass::Water`, the SAME grid `derive.rs` maintains and the coverage geodesic routes around). Wet edges, foam bands and depth fades are all *reads of that field*, zero extra rays.
3. **Physical light transport, already proven on this stack.** Fresnel reflect + Snell refract + Beer-Lambert absorption is exactly the M2 glass mechanism — the SDF hit samples `sample_glass`, the ray **transmits and continues** instead of terminating (`megakernel.slang:1266–1338`), and the path tracer composes the rest: the reflection of the dynamic city is **correct by construction** because the reflected ray traces the same scene the camera does. No screen-space fake (CS2/raster SSR fails off-screen and at grazing edges — the path-traced games line replaced SSR with the unified trace for exactly this reason; NVIDIA's path-traced-games guidance treats water reflections as first-class traced rays). This is a headline win the engine gets for one continuation ray.

### 4.2 The water SDF — analytic plane(s) clipped by the mask; intersection is O(1), not a march

**The primitive.** Per water body, a flat plane at its surface height (`WaterMask.sea_level` for the sea; `WaterBody.surface_height` per lake), clipped by the 2D water-cell field. The whole map's water is a piecewise-constant height field `water_cell_height(x,z)` (one value per 4 m cell; `−INF` = dry), cooked game-side from the SAME source as `CellClass::Water`.

**The intersection is analytic — this is the cheap half of water.** For each of the ≤ `MAX_WATER_PLANES = 4` distinct surface heights `h_k`: `t_k = (h_k − ro.y) / rd.y`; accept the nearest `t_k ∈ (0, t_max)` whose hit cell has `water_cell_height == h_k` (one texel fetch). Total cost: ~10 ALU + ≤4 texel fetches per ray — **versus up to 384 trilinear-sampled sphere-trace steps × N instances for the baked SDF buildings** (`sdf_render_intersect`). Water *detection* is ~free; §4.6 shows the continuation ray is the entire cost. The water test runs in the megakernel's SDF block (before the sky-miss fallthrough, every bounce — the M2 gating precedent at `megakernel.slang:1194–1208`), competing on nearest-`t` with `sdf_render_intersect`'s result and the triangle `g_hit_t`, gated by `u_water_enabled` (0 default = byte-identical existing renders).

**Waves are NORMAL displacement on the flat plane, not geometry (v1).** At a water hit, perturb the shading normal with a 4-octave Gerstner sum (per octave: direction, wavelength, amplitude, speed; `u_water_time` uniform): ~60–80 ALU per *hit* (sin/cos per octave), evaluated lazily exactly where rays land — no per-frame FFT pass, no displacement bake, no vertex update, deterministic in `t` (replay-friendly, matching the game's determinism rules). Amplitude is capped (≤ 0.3 m) so the bump-vs-silhouette error stays sub-pixel at city camera distances; the flat-intersection approximation is stated, not hidden. **Why Gerstner and not FFT for v1:** FFT ocean (Tessendorf / NVIDIA OceanCS / GodotOceanWaves) buys open-ocean spectrum fidelity for a per-frame 256²–512² FFT compute pass + displacement/normal/foam map memory; per-thread it wins only at large wave counts (N Gerstner evaluations vs log N FFT — and the BTH comparative analysis measures Gerstner *winning* overall at small N). City lakes, rivers and harbors want a handful of octaves of local ripple, not a Phillips spectrum. FFT is the v2 upgrade for true ocean maps; the seam (§4.8) is one function: `wave_normal(xz, t)`.
**Geometric wave displacement (v2+):** sphere-trace the displaced height function with a Lipschitz bound — 8–24 bounded steps per water ray instead of 1 analytic step. That arithmetic is why it is NOT v1.

### 4.3 The dielectric material — the M2 glass branch with water constants

The water hit runs the proven glass continuation verbatim with different parameters:

- **IOR 1.333** (water) instead of 1.5 — `sample_glass(wo, n, ior, roughness, absorption_color, absorption_depth, distance, u1, u2, u3)` (`brdf_glass.slang:227`) already takes it. Normal-incidence Fresnel R₀ = ((1.333−1)/(1.333+1))² ≈ **0.020** — water reflects ~2 % head-on and approaches 100 % at grazing, all from `fresnel_dielectric`. One stochastic continuation per sample: reflect with probability F, else refract (the M2 branch spawns exactly ONE next ray — `megakernel.slang:1301–1330`; the Fresnel split converges over spp/temporal accumulation, the standard real-time-PT treatment).
- **Roughness from sea state:** `roughness = lerp(0.02, 0.18, choppiness)` where choppiness also scales the Gerstner amplitudes — calm water is a near-mirror (cheap, sharp, low-noise reflections), choppy water is rough-specular (noisier; the denoiser/temporal tier's problem, flagged in §4.6).
- **Beer-Lambert depth absorption, wavelength-correct.** `beer_lambert(absorption_color, absorption_depth, distance)` = `exp(-σ·d)` exists (`brdf_glass.slang:77`). Default game-water σ (coastal/lake water, per the published attenuation data — red is absorbed within ~5 m; clear-water K ranges ~0.02–0.2+ m⁻¹ blue→coastal): **σ = (0.45, 0.12, 0.06) m⁻¹** for (R, G, B) → red gone by ~6 m, green by ~25 m, blue persisting — the blue-green deepening is *physics*, not an artist gradient. Exposed as a per-map material parameter (murky harbor vs alpine lake).
- **In-water path length (v1 estimate, stated honestly):** at the refraction event, attenuate throughput by `beer_lambert(σ_norm, 1, d_est)` with `d_est = water_depth(xz) / max(|wi.y|, 0.2)` from the cooked depth field — zero per-ray state. The refracted ray then sphere-traces to the real bottom (the SDF block already runs on every bounce — the M2 change at `megakernel.slang:1205–1207`); the bottom's sun term is additionally attenuated by the vertical column `exp(−σ·depth)` (light had to come down through the water too). Error: steep-refraction path lengths are underestimated near |wi.y| → 0.2; the exact per-segment attenuation via the existing nested-dielectric IOR stack (`g_ray_ior_stack_0..7` + `g_ray_ior_depth`, `megakernel.slang:262–271` — built for this, currently unused by the SDF glass branch) is the v2 upgrade. v1 paths never travel water→air upward (the bottom shade terminates the chain at the 3-bounce floor), so TIR is rare; `sample_glass` handles it anyway (`brdf_glass.slang:361–372`).
- **The deep-water cutoff — the single biggest ray saver.** Where `water_depth(xz) > d_cut` with `d_cut = ln(1/ε)/σ_min` (ε = 0.05, σ_min = 0.06 → **d_cut ≈ 50 m**, or per-map ~12 m for murky σ_min 0.25): the refraction branch is *pointless* (transmittance < 5 %). Skip it: with probability (1−F) terminate immediately adding the deep-water body color; with probability F trace the reflection. Deep water pixels then cost ~F ≈ 0.02–0.2 continuations instead of 1.0 — **deep water is nearly free; the cost concentrates in the shallows and at grazing reflections**, which is where the image quality lives anyway. Counted and printed (`skipped_deep`).

### 4.4 CSG shoreline + shore-distance fields — foam and wet edges for zero rays

Cooked once per mask change (sub-millisecond CPU: two-pass chamfer/jump-flood over the 256² = 65,536-cell grid), three co-sized 2D fields:

| Field | Definition | Consumers |
|---|---|---|
| `water_cell_height` | controlling surface height per cell, `−INF` dry | the §4.2 mask clip (the intersection test) |
| `shore_distance` | signed metres to the water-mask boundary (+ inside water) | foam band: water-side shading where `shore_dist < foam_width` (default 4 m), broken up by the wave noise — pure shading, **0 extra rays**; wet-edge darkening of terrain/SDF hits where `−wet_width < shore_dist < 0` (albedo × 0.6, roughness × 0.5) — also 0 rays |
| `water_depth` | `h_w − terrain_height` per wet cell (≥ 0) | Beer-Lambert `d_est` (§4.3), the deep cutoff, shallow-tint ramp |

This is the prompt-level CSG payoff made concrete: `max(water_level, −terrain)`'s zero-set IS the shoreline, and every shoreline effect becomes a texel read of fields the game's own water mask generates. No screen-space depth tricks, no edge meshes.

### 4.5 The water mask UNIFICATION — one field, sim and render

`CellClass::Water` (derived by `classify_cell`: `sample ≤ sea_level` OR inside a `WaterBody.outline`) is already authoritative for buildability, coverage severing, and terraform flips. This design makes it authoritative for the PICTURE too: the game cooks `water_cell_height` from **exactly** `MapTerrain`'s mask + `WaterMask` heights, so a cell is rendered wet **iff** the sim says it is wet. The W3 gate is literal: per-cell agreement `(water_cell_height > −INF) == terrain.is_water(centre)` over all 65,536 cells, printed as `render==sim on 65536/65536 cells`. The engine stays game-agnostic: it receives a generic *fluid surface layer* (planes + three f32 grids + material params) — `CellClass`, `WaterBody`, and every game noun stay on the game side of the seam (the CLAUDE.md engine rule; the Primary Bridge ownership test).

### 4.6 Ray-cost arithmetic and the HARD per-frame water budget (the crux — parameterized on the perf verdict B)

**Definitions.** P = internal pixel count; S = samples/pixel/frame; f_w = water screen fraction; F̄ = mean Fresnel over water pixels (camera-dependent: ~0.04 looking down, 0.2–0.4 at grazing sunset shots); C_T = one full scene traversal (the budget's unit — on this stack the 384-step × N-instance SDF march dominates it); C_U = the bounded underwater march to the bottom (depth-capped; ~0.2–0.4 C_T in the shallows); **B = the perf agent's per-frame traversal budget**, equivalently **r = B/(P·S)** traversals per pixel per frame.

**What a water sample costs.** Detection is ~free (§4.2: ~10 ALU + ≤4 texels vs the ~10³–10⁵-ALU building march). The continuation is the cost:

| Branch (prob.) | Extra traversals | Notes |
|---|---|---|
| Reflect (F̄) | **+1.0 C_T** | re-enters the full city scene — the expensive, headline ray |
| Refract, shallow (1−F̄, depth ≤ d_cut) | **+C_U ≈ 0.3 C_T** | bounded march to the bottom + attenuated rig shade (no shadow trace — the M2 SDF Lambert precedent, `megakernel.slang:1340–1350`) |
| Refract, deep (1−F̄, depth > d_cut) | **+0** | skipped: deep-water color, terminate |

Per-frame water surcharge: **ΔW ≈ f_w · P · S · k_w · C_T**, with `k_w = F̄ + (1−F̄)·0.3·shallow_share` ∈ **[~0.1 (deep lake, high camera) … ~1.0 (grazing shot over shallows)]**. Plus the bounce-floor fact: water pixels REQUIRE `max_bounces ≥ 3` (≥ 4 where water reflects the M2 glass windows — a water→window→interior chain); the host already forces 3 on the with-atoms path.

**The hard budget.** `W_max = β · B` (default β = 0.25 — water may spend a quarter of the frame's traversal budget, a knob). Enforcement is *unbiased Russian roulette on the continuation*: spawn with `p = min(1, W_max / (f_w·P·S·k̂_w))` and divide throughput by `p` (k̂_w from last frame's counters). Noise rises as the budget shrinks; temporal accumulation/denoiser recovers it — the standard stochastic-transparency treatment in shipped path-traced games. Counters (`g_water_stats`: water px, continuations, reflect/refract/skipped/rouletted) are mandatory and printed every harness run — the budget is *observable*, never assumed.

**The affordance ladder — "given budget B, here is what we afford":**

| Perf-tier verdict r = B/(P·S) | Water affordance | Surcharge |
|---|---|---|
| **Floor (780M, seconds/frame — no r)** | everything on, S = 32–64; gate: water frame ≤ 2.5× dry frame (measured: k_w ≤ 1.0 ⇒ ≤ ~1.3× at f_w 0.3; 2.5× is the generous failure alarm) | seconds, acceptable |
| **r ≥ 3** | full water: reflect into the full TLAS, refract to the real bottom, exact ladder of §4.3 | ΔW ≈ f_w·k_w·P·S·C_T fits inside B−2·P·S·C_T |
| **2 ≤ r < 3** | continuation kept; reflected rays trace the *selector-reduced* TLAS (the universal design's LOD lever 1 — fewer instances/coarser SDF mips for water-reflected rays); refracted bottoms direct-light-only | reflection C_T → ~0.5 C_T |
| **1 < r < 2** | rouletted continuations (`p < 1`), ReSTIR/temporal reuse carries convergence; foam/wet/deep-color unaffected (0-ray effects) | ΔW clamped to (r−1)·P·S·C_T exactly |
| **r ≈ 1** | **honest fallback, headline OFF:** Fresnel-weighted sky/env + Beer-Lambert depth tint + foam — zero extra traversals. City reflections in water do not exist at this tier; if the perf verdict lands here, that is a renegotiation trigger, not a quiet downgrade | +0 |

Memory cost (all tiers): three R32F 256² fields + one height table ≈ **0.8 MB** (vs the 36 MB SDF atlas); Gerstner params are uniforms; the harness bottom SDF ≈ 1 MB. Per-hit ALU: ~80 (waves) + ~60 (Fresnel/sample) — noise off the critical path.

### 4.7 Terraform coupling — flood/drain is a field re-cook, not a scene rebuild (v1.5)

The landed terraform chain already produces the exact dirty signal: `apply_stamp` → `rederive_region(aabb+1)` → `water_flips > 0` → `MapTerrain.version` bump (`terrain.rs:122–150`). v1.5 couples the render: on `water_flips > 0`, re-cook the three §4.4 fields **for the dirty AABB only** (chamfer re-relaxation over the stamp's cells + a ring; worst case the full 65k-cell grid re-cooks in well under a millisecond) and re-upload as a dirty range — no geometry, no BLAS, no re-bake. The player lowers a shore, the cell flips Water in the SAME tick that severs the childcare geodesic, and the SAME flip makes the path-traced water appear there. Flood-severs-coverage (the terraform cascade) and flood-fills-the-picture become one event with two consumers. Drain is the symmetric flip. **v1 ships static bodies; v1.5 is this paragraph; v2 is simulated fluid (§4.8).**

### 4.8 Deferred honestly: caustics, FFT ocean, simulated fluid

- **Caustics — DEFERRED to the photon/cache path, and costly even offline.** Camera-first path tracing finds water caustics only through specular-diffuse-specular chains that NEE cannot sample — the literature is blunt ("rendering caustics by regular path tracing is possible in theory, [but] the cost is unacceptable in practice"; even biased real-time approaches go to cascaded caustic maps or screen-space photon mapping on DXR hardware). Real caustics on this stack belong to the realtime design's radiance-cache/photon tier, on `tomespensin`, measured. v1 renders NO caustics. (A non-physical procedural bottom-light modulation is a labeled fake we may add later, default OFF — never silently sold as caustics.)
- **FFT ocean** — v2 for genuine ocean maps; the seam is `wave_normal(xz, t)` (§4.2). Costs a per-frame FFT compute pass + map memory; not paid for lakes and rivers.
- **Simulated fluid (v2) — the seam, named now.** `vox_physics` already has `fluid.rs` / `pbf.rs` (position-based fluids). The v2 path is particles → an SDF surface (anisotropic-kernel / Zhu-Bridson surfacing) → the SAME water-hit branch with the analytic plane swapped for a sampled fluid field. The render mechanism (dielectric hit + continuation + absorption) is **unchanged** — that is the payoff of doing v1 as an SDF primitive instead of a special-cased plane shader. The surfacing pass itself (sim-paced, regional) is v2's design, not this one.
- **Rivers with gradient** — `WaterKind::River` exists; v1 renders each body at its flat `surface_height` (a sloping river renders as terraces at body boundaries). Honest limitation; a per-cell height (the field already supports it) + flow-aligned waves is the v1.5+ fix.

### 4.9 SOTA grounding (searched 2026-06-12)

- **Path-traced games treat water reflections/refractions as traced rays, not SSR:** NVIDIA's path-traced-games guidance (reflection motion vectors *for water*, refraction motion vectors for glass — water is the named use case) and Cyberpunk 2077 RT Overdrive replacing SSR with the unified full-resolution trace. The stochastic single-continuation + temporal/denoiser pattern in §4.6 is that shipping pattern.
- **Beer-Lambert/Fresnel/TIR in tracers** is textbook and cheap (demofox's canonical walkthrough; THREE.js-PathTracing-Renderer ships Beer-Lambert water tanks).
- **Wave models:** Tessendorf-line FFT (NVIDIA OceanCS, GodotOceanWaves) vs Gerstner (per-thread N vs log N; Gerstner cheaper at small N per the BTH comparative thesis; Crest documents Gerstner's lower per-component update cost). v1's lazy per-hit Gerstner evaluation is the path-tracer-native variant: no precompute at all.
- **Real water optics:** I = I₀e^(−κz); red extinct by ~5 m, blue K as low as 0.02 m⁻¹ (clear) to 0.2+ m⁻¹ (coastal); Jerlov water classes — §4.3's σ defaults are inside the measured coastal envelope.
- **Caustics cost:** cascaded caustic maps (Computers & Graphics 2021) and screen-space photon mapping (DXR) exist precisely because path-traced caustics are impractical — grounding §4.8's deferral.
- **SDF raymarching canon** (Inigo Quilez) — though v1 water deliberately needs no march at all (§4.2): the analytic plane is the degenerate, cheapest SDF.

Sources: [NVIDIA: Rendering Perfect Reflections and Refractions in Path-Traced Games](https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/), [Cyberpunk 2077 RT Overdrive interview](https://www.nvidia.com/en-us/geforce/news/cyberpunk-2077-ray-tracing-overdrive-mode-interview/), [demofox: Raytracing Reflection, Refraction, Fresnel, TIR, and Beer's Law](https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/), [THREE.js-PathTracing-Renderer](https://erichlof.github.io/THREE.js-PathTracing-Renderer/), [NVIDIA OceanCS (FFT ocean)](https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/OceanCS_Slides.pdf), [GodotOceanWaves](https://github.com/2Retr0/GodotOceanWaves), [BTH: comparative performance analysis of FFT vs Gerstner](https://bth.diva-portal.org/smash/get/diva2:1778248/FULLTEXT02.pdf), [Crest wave conditions](https://crest.readthedocs.io/en/4.11/user/wave-conditions.html), [Large-scale ray-traced water caustics via cascaded caustic maps](https://www.sciencedirect.com/science/article/pii/S0097849321001230), [Screen-Space Photon Mapping caustics (DXR)](https://www.researchgate.net/publication/331336521_Caustics_Using_Screen-Space_Photon_Mapping_High-Quality_and_Real-Time_Rendering_with_DXR_and_Other_APIs), [NOAA: Beer–Lambert attenuation in seawater](https://repository.library.noaa.gov/view/noaa/20744/noaa_20744_DS1.pdf), [iquilezles: raymarching distance fields](https://iquilezles.org/articles/raymarchingdf/).

---

## 5. Data Models

```rust
// crates/vox_render/src/splat_backend.rs — additive, #[cfg(feature = "spectra-native")].

/// One flat water surface level (sea or one lake/river body).
#[derive(Debug, Clone, Copy)]
pub struct WaterPlane {
    surface_y: f32,
}
impl WaterPlane {
    pub fn new(surface_y: f32) -> Self { Self { surface_y } }
    pub fn surface_y(&self) -> f32 { self.surface_y }
}

/// The generic fluid-surface layer the engine traces. Game-agnostic: grids +
/// heights + material. The GAME cooks these from CellClass::Water/WaterMask;
/// no game noun crosses this boundary.
pub struct WaterSurfaceInput {
    planes: Vec<WaterPlane>,          // ≤ MAX_WATER_PLANES = 4 distinct levels
    grid_width: u32,                  // cells, x-fastest like Heightmap::data
    grid_height: u32,
    cell_size: f32,                   // metres
    origin: [f32; 2],                 // world metres
    water_cell_height: Vec<f32>,      // len = w*h; f32::NEG_INFINITY = dry
    shore_distance: Vec<f32>,         // len = w*h; signed metres, + inside water
    water_depth: Vec<f32>,            // len = w*h; surface_y - terrain_y, 0 dry
    material: WaterMaterial,
}
impl WaterSurfaceInput {
    pub fn new(planes: Vec<WaterPlane>, grid_width: u32, grid_height: u32,
               cell_size: f32, origin: [f32; 2], water_cell_height: Vec<f32>,
               shore_distance: Vec<f32>, water_depth: Vec<f32>,
               material: WaterMaterial) -> Result<Self, String>; // validates lens == w*h, planes ≤ 4
    pub fn planes(&self) -> &[WaterPlane] { &self.planes }
    pub fn wet_cell_count(&self) -> usize;   // water_cell_height > NEG_INF
    // ... accessor per field; fields private per the template rule.
}

/// Dielectric water material. Defaults: IOR 1.333, σ=(0.45,0.12,0.06) m⁻¹,
/// roughness 0.02 calm / 0.18 choppy, foam 4 m, wet edge 2 m, deep ε 0.05.
#[derive(Debug, Clone, Copy)]
pub struct WaterMaterial {
    ior: f32,
    absorption_per_m: [f32; 3],   // σ (R,G,B), 1/metres — beer_lambert input
    choppiness: f32,              // 0 calm .. 1 choppy: scales roughness + amplitudes
    foam_width_m: f32,
    wet_width_m: f32,
    deep_cut_transmittance: f32,  // ε for d_cut = ln(1/ε)/min(σ)
    deep_color: [f32; 3],         // returned where refraction is skipped
    wave: [WaveOctave; 4],        // Gerstner octaves
}
#[derive(Debug, Clone, Copy)]
pub struct WaveOctave { dir: [f32; 2], wavelength_m: f32, amplitude_m: f32, speed_mps: f32 }

/// Per-frame proof + budget instrumentation (read back from g_water_stats).
pub struct WaterFrameReport {
    water_px: u32,
    continuations: u32,
    reflections: u32,
    refractions: u32,
    skipped_deep: u32,
    rouletted_off: u32,      // budget enforcement events (0 on the floor tier)
    frame_seconds: f32,
    dry_frame_seconds: f32,  // same scene, u_water_enabled=0, measured
}
impl WaterFrameReport {
    pub fn cost_multiplier(&self) -> f32 { self.frame_seconds / self.dry_frame_seconds.max(1e-6) }
    // ... accessor per field.
}
```

```slang
// ~/src/spectra/slang/megakernel.slang — additive bindings beside the g_sdf_* block.
StructuredBuffer<float> g_water_cell_height;  // w*h, x-fastest, -1e30 = dry
StructuredBuffer<float> g_water_shore_dist;   // signed metres
StructuredBuffer<float> g_water_depth;        // metres
RWStructuredBuffer<uint> g_water_stats;       // [px, cont, refl, refr, deep, roulette]
uniform int    u_water_enabled;               // 0 default = byte-identical
uniform uint   u_num_water_planes;            // ≤ 4
uniform float4 u_water_plane_y;               // the ≤4 heights
uniform float  u_water_time;                  // wave phase (deterministic)
uniform float  u_water_ior;                   // 1.333
uniform float3 u_water_absorption;            // σ per metre
uniform float  u_water_roughness;             // from choppiness
uniform float  u_water_foam_width, u_water_wet_width, u_water_deep_cut;
uniform float3 u_water_deep_color;
uniform float  u_water_roulette_p;            // 1.0 floor; <1 under W_max
uniform float4 u_water_grid;                  // origin.xy, cell_size, width (height derived)
// Wave octaves: uniform float4 u_water_wave_dir_len[4]; u_water_wave_amp_speed[4];
```

GPU layout constraint: the three grids upload as plain `StructuredBuffer<float>` (same packing discipline as `g_sdf_distances` — `uploader.rs` precedent), 256²·3·4 B ≈ 0.8 MB.

---

## 6. API

```rust
// crates/vox_render/src/splat_backend.rs — additive; the M0–M2 entries unchanged.

/// One-shot still: path-trace SDF buildings + bottom + the WATER dielectric.
/// Extends pathtrace_sdf_scene_with_atoms_to_rgba (same camera/spp/rig
/// semantics, NRC off, max_bounces forced >= 3) with the fluid surface layer.
/// Returns RGBA8 (w*h*4) + the measured WaterFrameReport (dry frame re-rendered
/// internally with u_water_enabled=0 for the cost multiplier).
/// Errors: empty volumes/instances, grid len mismatches, kernel compile (named).
#[cfg(feature = "spectra-native")]
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_with_water_to_rgba(
    volumes: &[SdfVolumeInput],
    instances: &[SdfSceneInstance],
    atoms_per_instance: &[Vec<SdfSceneAtom>],
    water: &WaterSurfaceInput,
    eye: [f32; 3], target: [f32; 3], fov_y: f32,
    width: u32, height: u32, spp: u32,
    rig: &LightRig,
) -> Result<(Vec<u8>, WaterFrameReport), String>;

/// Cook a heightfield region into a sphere-traceable SdfVolumeInput (the water
/// bottom / terrain). d(p) = (p.y - h(x,z)) * inv_lipschitz, clamped to the
/// narrow band — conservative (Lipschitz ≤ 1), never punches through.
/// inv_lipschitz = 1/sqrt(1 + max|∇h|²), computed from the data.
#[cfg(feature = "spectra-native")]
pub fn heightfield_to_sdf_volume(
    heights: &[f32], width: usize, depth: usize, cell_size: f32,
    y_pad: f32, voxel_size: f32,
) -> Result<SdfVolumeInput, String>;

/// Cook shore_distance + water_depth from water_cell_height + terrain heights
/// (two-pass chamfer distance transform; full 256² grid well under 1 ms CPU).
/// Game-agnostic: takes grids, returns grids.
pub fn cook_fluid_shore_fields(
    water_cell_height: &[f32], terrain_heights: &[f32],
    width: usize, height: usize, cell_size: f32,
) -> Result<(Vec<f32>, Vec<f32>), String>; // (shore_distance, water_depth)

// Existing signatures relied on VERBATIM (verified in code 2026-06-12 — do not re-derive):
//   sample_glass(float3 wo, float3 n, float ior, float roughness,
//                float3 absorption_color, float absorption_depth, float distance,
//                float u1, float u2, float u3) -> BSDFSample          // brdf_glass.slang:227
//   beer_lambert(float3 absorption_color, float absorption_depth, float distance)
//                -> float3                                            // brdf_glass.slang:77
//   sdf_render_intersect(float3 ro, float3 rd, float t_max,
//                        out float3 hit_normal, out int hit_inst) -> float // megakernel.slang:687
//   pathtrace_sdf_scene_with_atoms_to_rgba(volumes, instances, atoms_per_instance,
//        eye, target, fov_y, width, height, spp, rig) -> Result<Vec<u8>, String>
//                                                                     // splat_backend.rs:982
//   SdfVolumeInput { resolution, origin, voxel_size, narrow_band, distances } // :454
//   SdfSceneInstance { volume_index, position, rotation_xyzw, uniform_scale, albedo } // :660
//   LightRig { sun_dir, sun_color, sun_intensity, sky_intensity, camera_fill }        // :192
//   SdfLayer::from_parts_with_atoms(volumes, instances, distances, albedo,
//        atom_positions, atom_colors, atom_channels, instance_atom_range)   // layers.rs:271
// Threading: one-shot stills on the caller's thread (the M0–M2 model); the
// resident-frame integration follows the universal renderer's live_update seam later.
```

Game side (`~/Ochroma/projects/urban_horizon`):

```rust
/// Build the engine's fluid layer from the game's authoritative water field.
/// A cell is wet IFF terrain.is_water(centre) — the unification invariant.
pub fn build_water_surface_input(terrain: &MapTerrain, water: &WaterMask,
                                 material: WaterMaterial) -> Result<WaterSurfaceInput, String>;
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| Water analytic intersect + dielectric branch | the megakernel SDF block (after `sdf_render_intersect`, before the opaque shade; every bounce, the M2 gating pattern) | `~/src/spectra/slang/megakernel.slang` (new `water_surface_intersect` + branch beside :1208–1338) + new `~/src/spectra/slang/water_surface.slang` (waves/fields helpers) | gated `u_water_enabled` (0 default); nearest-t vs SDF/triangle hits |
| `WaterLayer` scene resource | `SceneState` layer set, beside `SdfLayer` | `~/src/spectra/rust/spectra-scene-state/src/layers.rs` | `WaterLayer::from_parts(planes, grids, material)` |
| Water buffer/uniform upload | the uploader, beside the `sdf_*` packing | `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` + `rust/spectra-renderer/src/renderer.rs` bind group | the `pack_sdf_volume_headers` precedent |
| `pathtrace_sdf_scene_with_water_to_rgba` | the engine acceptance tests; the game `--shot-water` harness | `crates/vox_render/src/splat_backend.rs` | mirrors the :982 with-atoms entry; NRC off, bounces ≥ 3 |
| `heightfield_to_sdf_volume` / `cook_fluid_shore_fields` | inside the with-water entry's callers (test scene build; game cook) | `crates/vox_render/src/splat_backend.rs` | generic grids in/out |
| `build_water_surface_input` | the `--shot-water` harness; later the live view bridge | `~/Ochroma/projects/urban_horizon/src/map/water_fields.rs` (new) + `src/bin/play.rs` | reads `MapTerrain::is_water`/`WaterMask` — the unification seam |
| Terraform dirty → field re-cook (v1.5) | `CivitasGame::end_stroke` where `water_flips > 0` (the `childcare_field_dirty` site) | `src/game/mod.rs` → `water_fields.rs` | dirty-AABB re-cook + re-upload; no BLAS, no bake |
| `WaterFrameReport` counters | read back after every with-water render | `splat_backend.rs` (host) ← `g_water_stats` (device) | printed in every harness run — the budget is observable |

---

## 8. Open Questions

- [ ] **Spectral absorption.** The film is spectral; water absorption is the textbook spectral phenomenon. v1 attenuates RGB throughput (the existing glass-branch model, `megakernel.slang:1307–1310`); a 16-band σ(λ) through the spectral film is the honest upgrade and a differentiator. Decide after v1 images exist (cost: per-band exp in the water branch).
- [ ] **Exact in-water path length via the IOR stack** (`g_ray_ior_stack_*`) vs the v1 depth/|wi.y| estimate — measure the visual delta on the shallow-band test before paying the per-ray state.
- [ ] **Reflected-ray TLAS LOD** (tier 2 ≤ r < 3): does the selector emit a second, coarser selection for water-reflected rays, or do reflected rays just take the standard selection with a shorter t_max? Decide on `tomespensin` numbers, owned by the realtime design.
- [ ] **β (the water share of B)** — 0.25 default; the perf agent's verdict may move it. The roulette mechanism makes any β enforceable without code change.
- [ ] **Per-cell river gradients** (the §4.8 terrace limitation) — the `water_cell_height` field already supports per-cell heights; the open question is wave alignment + the analytic intersect becoming a short heightfield march. v1.5+.

---

## 9. Out of Scope

- **Caustics** — deferred to the photon/radiance-cache tier (§4.8); no fake shipped silently.
- **Simulated fluid** (vox_physics `fluid.rs`/`pbf.rs` → SDF surfacing) — v2; the seam is named in §4.8, nothing more.
- **Geometric wave displacement / FFT ocean spectra** — v2 (§4.2 arithmetic).
- **The realtime millisecond gate itself** — owned by [Spectra Realtime](./2026-06-10-spectra-realtime-design.md); this design only parameterizes water's affordances on its verdict (§4.6).
- **Boats, buoyancy, swimming agents, water-network gameplay (pumps/sewage)** — game-side features on top of the same mask; not rendering.
- **Shadow rays through water / wet-surface SSS** — the SDF rig-light shading model has no shadow traversal yet (M2 precedent); when it gains one, water inherits the vertical-column attenuation already specced in §4.3.

---

## 10. Related Plans / Designs

- **Implementation plan:** [SDF Water — Wave 1](../plans/2026-06-12-sdf-water-wave1.md) (v1 static bodies, floor-proven, budget-instrumented).
- **Depends on:** [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (M0–M2 landed: spectra `1480e2c` / ochroma `160469b`), [SDF Pillar](./2026-06-10-sdf-pillar-design.md) (volume/atlas shapes), the perf agent's megakernel verdict (the budget B).
- **Couples to:** [Terraform Tool Design](./2026-06-11-terraform-tool-design.md) + landed wave 1 (the `water_flips` dirty signal → v1.5 flood/drain).
- **Defers to:** [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (HW-RT, ReSTIR, denoiser — the tiers of §4.6's ladder above the floor).

---

## Milestone Ladder (metric-led; ray cost stated per step)

**W0 — Flat dielectric water over a real bottom (780M floor).** Analytic plane + mask clip + `sample_glass`(IOR 1.333) continuation; bottom = `heightfield_to_sdf_volume`. *Metric:* reflection-evidence px > 800 vs the opaque control; shallow bottom-visible > 0.5. *Ray cost:* +1 continuation per water sample (k_w ≈ 1.0 — worst case, no savers yet); detection O(1).
**W1 — Waves + sea state.** 4-octave Gerstner normals + choppiness-driven roughness, `u_water_time`. *Metric:* > 30 % of water px change between t=0 and t=1.5 s; zero geometry re-upload. *Ray cost:* +0 rays; ~80 ALU per water hit.
**W2 — Beer-Lambert depth + shore fields + the budget counters.** σ absorption, deep cutoff, foam/wet bands, `g_water_stats` + `WaterFrameReport`. *Metric:* deep transmittance < 0.05; deep B/R > 2× shallow; foam confined to the band; `skipped_deep > 0`; cost multiplier ≤ 2.5× printed. *Ray cost:* k_w DROPS from ~1.0 to ~0.1–0.5 (the deep skip) — W2 is a net ray *saver*.
**W3 — The game's water, unified.** `--shot-water` on tidewater_flats: `build_water_surface_input` from `MapTerrain`/`WaterMask`; 65536/65536 cell agreement; the headline PNG (the game's own shore buildings mirrored in the game's own water). *Ray cost:* same as W2, real f_w measured and printed.
**v1.5 — Terraform flood/drain coupling.** Dirty-AABB field re-cook on `water_flips > 0`; the flood that severs coverage also fills the picture. *Ray cost:* unchanged; re-cook < 1 ms CPU.
**v2 — (separate designs)** spectral σ(λ), FFT ocean, geometric displacement, fluid.rs/pbf.rs surfacing, caustics via the photon/cache tier — each with its own arithmetic, none promised here.
