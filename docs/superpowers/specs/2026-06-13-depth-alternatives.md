# Design: Surface-Depth Alternatives in a Spectral Path Tracer — Beyond POM & Baked Displacement (2026-06-13)

**Status:** Draft
**Scope:** Survey and rank *alternative* techniques for real/perceived surface DEPTH (mortar recesses, reveals, panel relief, brick) on building facades in Spectra's spectral path tracer — explicitly excluding plain POM and baked/displaced micro-geometry (covered by `2026-06-13-texture-depth-cost.md`). Quantify feasibility (existing machinery vs net-new, with `file:line`), GPU-ms@4K + VRAM cost vs the POM baseline, artifact profile, and novelty. Target: 60 fps @ 4K / RTX 4070 Ti / 12 GB / 100K instances / ~300 types.
**Related:** `2026-06-13-texture-depth-cost.md` (POM + displaced-BLAS baselines — the cost anchors used here); `hybrid-atom-sdf-lod-direction` (MESH renders the visible world, SDF = engine substrate); `facade-photorealism-roadmap` (Wave 3 = facade depth); `spectra-sdf-perf-verdict` (4070 Ti tier).

---

## 1. Problem Statement

- The companion doc establishes the two anchors: **POM** (`~0.3–1.0 ms @ 4K-internal`, `~0.1–0.3 GB`, flat silhouette, swims at grazing) and **displaced KHR BLAS** (true silhouette but `~20 MB/type`, only ~20–40 hero types fit 12 GB). There is a realism gap between them: POM's silhouette failure at grazing angles, and the cost wall that stops displacement from going wide. We want techniques that fill that gap.
- Spectra is a **real spectral path tracer with real GI/reflections** (HWSS, 4 hero wavelengths, `spectral_common.slang:248-345`; thin-film Airy `brdf_thinfilm.slang:37-101`; Cauchy dispersion `materialx_eval.slang:867`). Several depth cues that rasterizers fake — contact shadows in a recess, inter-reflection between a brick face and its mortar valley, wavelength-dependent edge color — fall out *for free* once relief is real geometry/field. The path tracer changes which techniques are worth their cost.
- We have an **SDF substrate** that is currently a *separate primary trace path*, not a hit-point detail layer: `sdf_render_intersect` (`megakernel_sdf.slang:721`) is called after the triangle BVH in the megakernel (`megakernel.slang:847`) and the nearest `t` wins. `sdf_eval.slang`'s analytic CSG evaluator (`eval_sdf_tree`, `sdf_eval.slang:85`) is **not** called from the mesh shading path. The question is whether that machinery can be borrowed as a relief layer at a mesh hit.
- We have **participating-media** machinery (delta/ratio tracking `volume_march.slang:112-195`, multi-scatter `volume_integrate.slang:288`) and a **functional neural-BRDF MLP** (`neural_material.slang:97-147` — *not* a stub, 4-layer 5→64→64→64→3). Both are candidate depth substrates nobody has pointed at facades.

---

## 2. Done When

This is a **design/decision doc** (read-only investigation; no code changed). It is "done" when:

Running `ls ~/src/ochroma/docs/superpowers/specs/2026-06-13-depth-alternatives.md` shows the file, and a human reading §3–§8 can answer, without reading code: (a) each alternative depth technique, whether it is Spectra-feasible-today vs net-new, with exact `file:line`; (b) its GPU-ms@4K + VRAM envelope **relative to the POM baseline** from the companion doc; (c) its artifact profile and novelty; (d) the ranked recommendation and which techniques are *uniquely strong because we are a spectral path tracer*. The follow-on *plan* (separate file) owns any render-gate command.

---

## 3. Capabilities (what a follow-on plan could build, and the real test for each)

These are illustrative real-behavior tests so a plan can't ship a stub. Only the techniques we actually recommend (§8) need a plan.

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| **Relief / cone-step march** (POM++ in megakernel) replaces the linear POM march | Brick wall at 75° grazing, dolly laterally: assert the brick/mortar boundary shifts `> 2 px` between frames AND the relief edge has `< 1 px` of the staircase "swim" that linear POM at the same 16 steps shows (diff two PNGs, measure edge jitter) | `assert!(cone_steps > 0)` — passes with linear march |
| **SDF detail layer at mesh hit** (sphere-trace a tiled facade SDF in tangent space at the triangle hit, return real depth + normal) | At a mortar joint, assert the returned hit offset along -N equals the SDF valley depth within `voxel_size`, AND a shadow ray into the recess returns occluded (real contact shadow) while the face pixel is lit | `assert!(sdf_detail_eval(p) < 1.0)` — passes without marching |
| **Shell-mapping prism layer** (extrude facade tris into a thin shell, ray-march the shell against a tiled height/SDF) | Cornice profile at the roofline silhouettes correctly against sky (true outline), assert the shell hit `t` differs from the flat-wall `t` by `>= relief_depth*0.5` at the silhouette edge | shell built but ray falls through to flat tri — `assert!(shell_built)` |
| **Thin-volume relief** (participating-media slab over the facade, density = relief height field) | Deep-recess panel: assert transmittance through the recess column is `< 0.5` of the face column (delta-tracked), reading as soft depth occlusion | `assert!(density_sampled)` |
| **Spectral micro-iridescence as microstructure cue** | Weathered/anodized panel: assert the per-hero-wavelength reflectance spread (max-min over 4 wavelengths) at a grazing edge is `> 15%`, producing a visible structural-color edge that reads as fine relief | `assert!(thinfilm_weight > 0)` |
| **Neural displacement field** (extend the MLP to output height/normal from UV+derivatives) | Trained MLP reproduces a reference 1K height map with `< 5%` RMSE at `< 1/8` the VRAM of the R8 texture; megakernel reads MLP output as the POM height source | MLP returns constant — `assert!(out.x == out.x)` |

---

## 4. Architecture — the candidate techniques

Each subsection: **what it is → Spectra feasibility (existing vs net-new, `file:line`) → where it hooks → cost/artifact/novelty**. The shading hook for every screen-space technique is the same: the megakernel surface shade just before the albedo/normal sample (`megakernel.slang:~1974`, TBN at `:1981-1990`, `wo` at `:661`, UV derivatives `:1554-1673`, BSDF dispatch at `:2566` / `material_dispatch.slang:176`).

### 4.1 Relief / cone-step / steep-parallax mapping [NET-NEW, near-zero marginal cost over POM, recommended]

POM cousins that fix POM's worst artifact (the staircase "swim" and silhouette wobble at grazing) by changing only the *march strategy*, not the infrastructure:

- **Steep parallax** = POM with many linear steps + no refine (crisper, more aliased).
- **Relief mapping** = linear search to bracket the crossing, then **binary search** to refine — kills the swim with `~6` extra taps.
- **Cone-step mapping (CSM)** = precompute a per-texel cone angle (the empty-space "safe radius") into a second channel of the height texture; the march takes conservative variable-length steps and converges in `~8–12` taps instead of POM's 16–32 — *fewer* taps for *better* quality, at the cost of a one-time cook of the cone map.

**Feasibility — trivial reuse.** This is the same tangent-space march as the planned POM (`2026-06-13-texture-depth-cost.md` §4.1) with a different inner loop. Everything it needs already exists at the hit: TBN (`megakernel.slang:1981-1990`), `wo`, UV derivatives for LOD (`:1554-1673`), and `sample_texture_ewa` for the height/cone tap (`megakernel.slang:1976`). Cone map = a second texture channel cooked offline (no engine change beyond an extra packed texture id; the packer slot is the same one the POM plan un-stubs at `splat_backend.rs:2612`). **No new infrastructure beyond POM.**

- **Cost vs POM:** CSM is *cheaper or equal* (8–12 taps vs 16–32) for crisper relief; relief mapping is `~POM + 6 taps` (negligible). VRAM: cone map adds one channel — pack into the height map's RG → effectively **+0 GB** over POM's `~0.1–0.3 GB`.
- **Artifacts:** still **flat silhouette** (no outline relief — this is the shared POM-family ceiling); CSM removes swim; relief mapping removes staircase. No new artifact classes.
- **Novelty:** low (industry-standard) — but it is the **highest-leverage upgrade** because it strictly dominates plain POM on quality-per-tap.

### 4.2 SDF-based surface micro-detail at the mesh hit [NET-NEW path, moderate cost, recommended for near/hero — high novelty]

We sphere-trace a **tiled facade detail SDF** in tangent space at the triangle hit, returning a real depth offset *and* an analytic normal, then optionally re-issue the shadow/secondary ray from the displaced point. Unlike POM (silhouette-flat) this gives true within-shell silhouette and real contact shadows; unlike a full displaced BLAS it stores **one tiny SDF tile per facade family**, not per-type geometry.

**Feasibility — machinery exists but is wired as a separate primary path, not a hit detail layer.**
- The sphere-trace core exists and is GPU-proven: `sdf_render_intersect_stats` (`megakernel_sdf.slang:581`), interval-culled, surf_eps keyed to voxel extents (`:596`), 384-step cap (`:676`), instance normal via central differences (`sdf_instance_normal`, `:544`). But it is invoked as a **scene-level primary trace** after the triangle BVH (`megakernel.slang:847`); it is **not** callable as a local detail eval at a mesh hit today — that wiring is net-new.
- The analytic CSG evaluator `eval_sdf_tree` (`sdf_eval.slang:85`, primitives `:36-61`, ops `:128-133`) **can** be evaluated at any point cheaply (box/round-box = a few ALU) — ideal for a *procedural* facade tile (mortar grid = repeated round-boxes; reveal = subtract box) with **zero texture VRAM**. `eval_op` (`:65`) plus domain repetition is enough to author brick/mortar/panel relief analytically. This is the cheap, high-novelty variant: relief defined as a tiny CSG directive, evaluated on the fly, no grid upload.
- Procedural break-up noise is on hand: `domain_warped_fbm` / `ridge_fbm` / `swiss_noise_gpu` (`gpu_noise.slang:43,93,66`) already feed the SDF noise field (`sdf_eval.slang:56-61`) — weathering and irregular mortar come free.

**Two sub-variants:**
1. **Analytic tile SDF (recommended near/hero):** evaluate `eval_sdf_tree` over a *repeated* tangent-space domain at the hit; sphere-trace a few steps within a thin shell `[0, relief_depth]`. VRAM = the CSG node buffer (`64 bytes/node`, `sdf_eval.slang:12-20`) — **kilobytes per family**, not MB. Normal = central difference of the analytic field (cheap).
2. **Voxel tile SDF:** a small baked grid (e.g. `128² × 16` narrow-band) sampled via the existing trilinear path (`megakernel_sdf.slang:98`). VRAM `~1 MB/tile` (`128·128·16·4 ≈ 1.05 MB`); shared across all instances of a family.

- **Cost vs POM:** the *march itself* is comparable to POM (both are short tangent-space ray-marches); the SDF tap is `~equal` to a height tap for the voxel variant, and **cheaper** (pure ALU, no texture fetch / cache miss) for the analytic variant. The win is that **the path tracer makes the depth real for free**: shadow rays (`rt_query.slang:83`) and GI bounces fired from the displaced hit produce true mortar contact shadows and brick-to-mortar inter-reflection — a screen-space POM cannot. Budget: **`~POM + 0.2–0.5 ms`** at near/hero LOD (the extra is secondary-ray cost in recesses), **`< 0.05 GB`** VRAM (analytic) or **`~0.3 GB`** for ~300 voxel tiles.
- **Artifacts:** within-shell silhouette is correct; the *triangle* silhouette is still flat unless combined with shell mapping (§4.3). Sphere-trace can over-step thin features → use surf_eps floor (`:596`). No swim.
- **Novelty:** **high.** "SDF micro-relief evaluated at the mesh hit and lit by a spectral path tracer" is a genuine differentiator — it reuses our substrate pillar (`sdf-sota-pillar-directive`) for facades, exactly the kind of cross-pillar leverage the engine is supposed to have, while staying inside the MESH-renders-the-world rule (SDF is a *detail field on the mesh*, not the primary surface).

### 4.3 Shell mapping (prism extrusion) [NET-NEW, moderate-high cost, recommended ONLY for silhouette-critical hero edges]

Extrude each facade triangle along its normal into a **triangular prism** of thickness = relief depth; ray-march the prism interior against a tiled height field or SDF (§4.2). This is the *only* alternative that gives a **true silhouette** (cornices, deep reveals, projecting brick courses outline correctly against the sky) **without** per-type displaced geometry.

**Feasibility — net-new; nothing shell/prism exists.** Confirmed absent: no `shell` / `prism` / `tetrahedral` code anywhere in `slang/`. We would build: prism BLAS (3 extruded tris per facade tri → cheap proxy geometry the existing KHR build can host, `rt_query.slang:83`), a ray→prism parameterization (barycentric + height), and the inner march (reuse §4.2 SDF or POM tap). The displaced-normal helper `displaced_normal` (`hitpoint_displacement.slang:21`) and `apply_displacement` (`:14`) help re-seat the shading normal/point after a shell hit.

- **Cost vs POM:** highest of the recommended set. Prism BLAS adds geometry — but only the *extruded coarse* facade tris (a few hundred per type, not the subdivided millions a full displaced BLAS needs), so VRAM is **`~1–3 MB/type`** vs displaced BLAS's `~20 MB/type` — roughly **5–10× cheaper than displacement** for a true silhouette, and it can therefore go *wider* than the 20–40-type displacement cap (maybe 80–120 hero types in budget). Per-ray cost is the prism intersect + inner march, `~POM + 0.5–1.0 ms` at the LOD it runs.
- **Artifacts:** correct silhouette; classic shell-mapping distortion on highly curved/non-planar surfaces (facades are near-planar → minimal); seams at prism boundaries if the tiling isn't continuous across edges (manageable on a planar wall).
- **Novelty:** moderate (known technique) but **rare in path tracers** and a strong fit: it is the cost-effective bridge between POM (no silhouette) and displaced BLAS (silhouette but VRAM-bound).

### 4.4 Thin participating-media relief [PARTIAL machinery, NOT recommended as primary depth]

Place a thin participating-media slab over the facade whose density = the relief height field; delta-track through it so recesses read as soft depth occlusion / ambient darkening.

**Feasibility — machinery exists but is mismatched.** Delta/ratio tracking (`volume_march.slang:112-195`), multi-scatter (`volume_integrate.slang:288`), VDB + dense grid sampling (`volume_march.slang:137`, `volume_integrate.slang:141`) are all real and GPU-proven. But volumes integrate as a *separate march* (`march_volumes`, `volume_march.slang:429`), not a per-hit modifier, and a participating medium produces **soft/foggy** occlusion, not the **sharp** brick/mortar edges facades need. It is the wrong tool for hard relief.

- **Cost vs POM:** **much higher** — delta tracking is many density taps + RNG per pixel (`volume_march.slang:557-629`); VDB VRAM is non-trivial. Worse cost, worse look for this use.
- **Niche where it wins:** genuinely soft/volumetric facade elements — frosted/etched glass depth, deep grime AO in recesses, dusty light shafts in deep reveals (the spectral path tracer makes these *beautiful*). Keep as an **optional polish layer over §4.2/§4.3**, never the depth primary.
- **Novelty:** high *for the niche*; irrelevant for brick/mortar.

### 4.5 Neural displacement / neural BSDF [FEASIBLE base, net-new for displacement, speculative]

The neural-material MLP is **real and working** — `neural_material_eval` (`neural_material.slang:97-147`), 4-layer 5→64→64→64→3, scalar `dense_layer_forward` (`:64`) with a warp-cooperative variant (`:232`) and an sm_100a CoopVec fast path noted but not required (`:179-197`). Today it outputs a 3-channel RGB BRDF from `(wo,wi,roughness)`.

**Feasibility — repurposing is net-new but small.** Extend the feature vector to `(uv, ddx_uv, ddy_uv)` and the output to height (+ optional analytic normal); feed the predicted height into the §4.1/§4.2 march. The MLP forward pass already exists; the change is input encoding + output dim + a training pipeline (offline, net-new). Could compress ~300 facade height maps into one small weight set.

- **Cost vs POM:** MLP eval per tapped pixel is **expensive on the 4070 Ti without CoopVec** (scalar 64×64 matmuls, `:64-83`); the sm_100a fast path is Blackwell, not Ada (`:179`). So per-frame neural *displacement evaluation* is likely *more* GPU-ms than just sampling an R8 height texture — the texture is the cheaper oracle on our tier. VRAM **win** (weights ≪ texture set) but the GPU-ms loss dominates at 4K.
- **Verdict:** **defer.** Strong on VRAM, weak on Ada GPU-ms; revisit when CoopVec/Blackwell is in tier, or use it as an *offline* height-map generator (train MLP → bake R8 → feed POM/CSM) which gives the VRAM-free authoring without the runtime cost.
- **Novelty:** very high, but premature for the perf tier.

### 4.6 Spectral-unique depth cues [FEASIBLE today, free polish — uniquely ours]

The spectral path tracer enables depth/microstructure cues a rasterizer or RGB tracer cannot, and the machinery is **already wired**:

- **Thin-film / structural-color micro-iridescence on edges.** Full Airy with s/p polarization exists: `thin_film_reflectance` (`brdf_thinfilm.slang:37-101`), integrated into the surface lobe via `iridescence_weight` (`spectra_surface.slang:144-152`) and a per-band spectral variant `lama_iridescence_evaluate_spectral` (`materialx_eval.slang:633`). Weathered/anodized/oxidized facade metal, glazing coatings, and patina read as subtle wavelength-shifting color on relief edges and grazing faces — a strong fine-microstructure cue that **costs nothing extra** because it rides the existing HWSS lobe evaluation (`megakernel.slang:2602-2639`).
- **Dispersion at glazing reveals.** Cauchy `n(λ)=A+B/λ²` (`materialx_eval.slang:867-908`) already splits hero wavelengths through glass — deep window reveals get real chromatic edge fringing, a true depth/edge cue, free on the existing dielectric path.
- **Wavelength-dependent Fresnel on relief edges.** Per-hero-wavelength spectral weight (`dispatch_eval_spectral_weight`, `material_dispatch.slang:916`) makes grazing relief edges shift color subtly with view — the brain reads spectral edge-color variation as fine 3D structure.

These are **not a depth-displacement technique** — they are the *force multiplier* that makes any real relief (§4.2/§4.3) read as photoreal rather than CG. They are **free** (already evaluated) and **uniquely ours**.

### 4.7 Hybrid / LOD combination [the actual recommendation]

No single technique wins all distances. The frontier-beating stack, distance-banded by the per-instance technique selector (the same LOD lever the companion doc defines):

```
NEAR / HERO   : shell-prism (§4.3) OR displaced KHR BLAS  → true silhouette
              + SDF detail at hit (§4.2)                  → real recess depth + contact shadow
              + spectral edge cues (§4.6, free)
MID           : cone-step / relief mapping (§4.1)         → crisp interior relief, no silhouette
              + spectral edge cues (§4.6, free)
FAR           : flat normal map                           → cheapest
```

Spectral cues (§4.6) ride along at **every** band for free.

---

## 5. Data Models

No new GPU buffer layouts for §4.1/§4.6. They reuse:
- the height/cone texture (cone packed into a spare channel of the height map);
- the existing material fields `displacement_tex / displacement_scale / displacement_midlevel` (`material_types.slang:69-71`) the companion-doc POM plan un-stubs;
- the existing SDF CSG node buffer for §4.2 analytic tiles (`SDFOpGPU`, 16 floats / 64 B, `sdf_eval.slang:12-20`) — a tiny per-family directive, plus the volume/instance headers (12/19 floats, `megakernel_sdf.slang:62-63`) for the voxel-tile variant.

Shell mapping (§4.3) adds a per-type **prism proxy BLAS** entry to the existing content-addressed scene-upload BLAS cache (ordinary extruded-triangle geometry; no new layout).

---

## 6. API (net-new surface a plan would add)

```slang
// §4.1 — drop-in replacement for the planned linear POM inner loop, same call site (megakernel.slang:~1974):
float2 cone_step_uv(float2 uv, float3 wo, float3 T, float3 B, float3 N,
                    int height_cone_tex,            // RG = (height, cone_ratio)
                    float scale, float midlevel,
                    float2 ddx_uv, float2 ddy_uv,   // LOD the step count
                    out float out_self_shadow);     // 8–12 taps typical (vs POM 16–32)

// §4.2 — SDF detail evaluated AT the mesh hit, in tangent space, within a thin shell:
// Returns depth offset along -N (>=0) and analytic normal; -1 depth = no relief hit.
float sdf_detail_at_hit(float2 uv, float3 wo, float3 T, float3 B, float3 N,
                        int sdf_tile_id, float shell_depth,
                        out float3 detail_normal);  // analytic via central diff of eval_sdf_tree
// Reuses eval_sdf_tree (sdf_eval.slang:85) over a repeated tangent-space domain.
// After hit: re-seat shading point (hitpoint_displacement.slang:14) so shadow/GI rays
// fire from the recess floor → real contact shadow via rt_query.slang:83.
```

```rust
// §4.1/§4.2 packer: route the cooked height/cone tile id + SDF-tile id, replacing the hard -1
// at splat_backend.rs:2612 (the same slot the POM plan owns).
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `cone_step_uv()` (§4.1) | megakernel surface shade, before albedo/normal sample | `~/src/spectra/slang/megakernel.slang:~1974` | reuse TBN `:1981-1990`, `wo` `:661`, UV derivs `:1554-1673`; cone map cooked offline |
| `sdf_detail_at_hit()` (§4.2) | same hit point, gated to near/hero LOD | `~/src/spectra/slang/megakernel.slang:~1974` | reuse `eval_sdf_tree` `sdf_eval.slang:85`, central-diff normal like `sdf_instance_normal` `megakernel_sdf.slang:544`; secondary rays via `rt_query.slang:83` |
| prism proxy BLAS build (§4.3) | scene cook, hero types only | `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` (KHR build) | extruded coarse tris; cache by type; trace via `rt_query.slang:83` |
| spectral edge cues (§4.6) | already wired in lobe eval — no new call | `spectra_surface.slang:144-152`, `materialx_eval.slang:633,867` | free; ensure facade materials set `iridescence_weight`/dielectric where appropriate |
| LOD technique selector | per-instance setup before TLAS feed | engine instance governor (vox_app/vox_sim) | distance → {shell+SDF / cone-step / flat} per §4.7 |

---

## 8. Cost vs Budget — ranking by (realism gain × feasibility) / cost

Budget anchor (from `2026-06-13-texture-depth-cost.md`): **16.6 ms/frame** @ 60 fps 4K, 4070 Ti, 12 GB; POM baseline **`~0.3–1.0 ms` / `~0.1–0.3 GB`**, flat silhouette.

| Rank | Technique | vs-POM GPU-ms | vs-POM VRAM | Silhouette | Spectral-unique | Feasibility | Score |
|---|---|---|---|---|---|---|---|
| **1** | **Cone-step / relief (§4.1)** | **≤ POM** (8–12 vs 16–32 taps) | **≈ +0** (cone in spare channel) | flat (POM ceiling) | no | reuse POM path, +inner loop | **Highest** — strictly dominates plain POM |
| **2** | **SDF detail at hit (§4.2, analytic)** | +0.2–0.5 ms | **< +0.05 GB** (CSG = KB/family) | within-shell true | partially (real contact shadow/GI) | march exists, hit-wiring net-new | **High** — real depth, tiny VRAM, our substrate |
| **3** | **Spectral edge cues (§4.6)** | **≈ +0** (rides existing lobe) | +0 | n/a (cue, not relief) | **yes — uniquely ours** | already wired | **High** — free photoreal multiplier |
| **4** | **Shell-prism (§4.3)** | +0.5–1.0 ms | ~1–3 MB/type (5–10× < displaced BLAS) | **TRUE** | with §4.2/§4.6 | net-new, moderate | **Medium-High** — only cheap true silhouette |
| **5** | **Neural displacement (§4.5)** | **+ (worse on Ada, no CoopVec)** | big win (weights ≪ textures) | depends | no | MLP exists, training net-new | **Low now** — defer to Blackwell or use offline-only |
| **6** | **Thin-volume relief (§4.4)** | **much higher** | non-trivial VDB | soft only | yes (frosted/grime niche) | volume path exists, mismatched | **Low** — niche polish, not depth primary |

### Which are uniquely strong *because we are a spectral path tracer*

- **§4.2 SDF-detail-at-hit** — relief becomes *real geometry-like depth*, so shadow rays and GI bounces fired from the recess produce **true mortar contact shadows and brick↔mortar inter-reflection** that screen-space POM physically cannot. The path tracer is what makes a cheap SDF tile look expensive.
- **§4.6 spectral edge cues** — thin-film micro-iridescence (`brdf_thinfilm.slang:37`), glazing dispersion (`materialx_eval.slang:867`), and wavelength-dependent grazing Fresnel are **structural-color microstructure cues** no RGB renderer has. Already wired, free, and a genuine product differentiator.

### What beats POM on the realism/cost frontier

- **§4.1 cone-step strictly beats plain POM** — equal-or-fewer taps, crisper relief, ~zero extra VRAM. This should be the *default* mass technique, replacing plain POM outright in the companion-doc plan.
- **§4.2 analytic SDF-at-hit beats the displaced-BLAS for near/mid depth on cost** — real recess depth + contact shadow at kilobytes/family instead of ~20 MB/type, lifting the "real depth" tier far above the 20–40-type displacement cap (it does not give the *triangle* silhouette — that's where §4.3 shell-prism comes in, still 5–10× cheaper VRAM than full displacement).

### Recommendation

1. **Replace plain POM with cone-step/relief mapping (§4.1)** as the mass technique — same plan, better inner loop, no extra cost.
2. **Add SDF-detail-at-hit (§4.2 analytic)** for near/mid — real recess depth + path-traced contact shadows at trivial VRAM; reuses the SDF pillar for facades while honoring MESH-renders-the-world (SDF is a detail field *on* the mesh).
3. **Turn on spectral edge cues (§4.6) everywhere** — free, uniquely ours, the photoreal multiplier; just author facade materials to use the already-wired thin-film/dielectric paths.
4. **Use shell-prism (§4.3) for silhouette-critical hero edges** (cornices, deep reveals) — the only cheap path to a true outline, 5–10× less VRAM than displaced BLAS, so it can go wider.
5. **Defer neural displacement (§4.5)** to a Blackwell tier or as an offline height/SDF-tile generator; **keep thin-volume (§4.4)** only for frosted-glass/grime niches.

---

## 9. Open Questions

- [ ] §4.2: analytic CSG tile vs baked voxel tile per facade family — analytic is VRAM-free but limits the relief grammar to CSG primitives + noise; is brick/mortar/reveal expressible enough analytically (likely yes via repeated round-box + subtract + `ridge_fbm`)?
- [ ] §4.2 shadow cost: re-issuing shadow rays from the displaced recess floor multiplies NEE cost in recessed pixels — cap to near LOD, or accept and measure?
- [ ] §4.3 shell continuity: do we need per-edge tile alignment across adjacent facade triangles to avoid prism seams, and does the cooked mesh already carry consistent tangents?
- [ ] §4.1 cone-map authoring: cook cone ratios from the same source height field, or from the Forge facade grammar directly (single-source per `forge-directive-factory`)?
- [ ] §4.6: which facade material families opt into thin-film/dispersion (metal panels, glazing, patina) without making everything look oily?

---

## 10. Out of Scope

- **Plain POM and baked/displaced micro-geometry** — owned by `2026-06-13-texture-depth-cost.md`; this doc only references their costs as anchors.
- **DMM/CLAS micromaps** — a stub (`spectra-rtxmg/src/subd.rs:182`); a separate plan owns un-stubbing it.
- **Tessellation-shader (raster) displacement** — Spectra is a path tracer; relief must live in the BVH or as a hit-point march.
- **Neural-MLP training pipeline** — content/ML task, not engine; §4.5 is deferred regardless.
- **SDF as a *primary* facade surface** — violates the MESH-renders-the-world rule (`hybrid-atom-sdf-lod-direction`); §4.2 uses SDF strictly as a detail field on the mesh hit.
- **Authoring the height/cone/SDF-tile asset set** — content task.

---

## 11. Related Plans / Designs

- Depends on: the POM wiring from `2026-06-13-texture-depth-cost.md` (megakernel material unpack + packer un-stub at `splat_backend.rs:2612`); KHR HW-RT BLAS/TLAS (verified, `rt_query.slang:83`) for §4.3.
- Required before: `facade-photorealism-roadmap` Wave 3 implementation plan (this doc widens its technique menu).
- Related: `sdf-sota-pillar-directive` (§4.2 reuses the SDF pillar); `spectra-sdf-perf-verdict` (4070 Ti tier numbers); `spectra-texture-bridge` (texture packing for height/cone maps).
