# Design: SDF Housing to 100% Quality at 100 Building Types (2026-06-12)

**Status:** Draft
**Scope:** Take the proven SDF-only Spectra render path (M0 solid building 0.976 coverage, M1 12-building block, M2 k-NN atom material + real path-traced glass — spectra `1480e2c`, ochroma `160469b`) from "honest but soft" to crisp, believable, textured housing — PBR textures on the SDF surface, sub-voxel detail recovered without sub-voxel SDF resolution, channel→BSDF material dispatch, lighting that reads real — and prove it across ~100 distinct Forge-generated building archetypes resident in one SDF atlas. Every quality feature is **budget-parameterized**: its cost is stated per-ray/per-hit/per-MB so the realtime perf track's verdict (per-frame ray/sample budget **B**) selects what is affordable without re-design.
**Related:** [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (SDF-only direction, atoms-as-appearance-field, the scene-feed seam), [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) (`SdfBaker`/`SdfAtlasGpu`, the 64 MB atlas quota, narrow-band math), [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (owns the millisecond budget **B** this design parameterizes on — never restated here), `spectra-texture-bridge` memory (the verified `set_texture_atlas`/EWA path + its four landmines), [AI Asset Factory](./2026-06-10-ai-asset-factory-design.md) (the `.asset.json` directive pipeline that scales to 100 archetypes).

---

## 1. Problem Statement

Concrete, observable symptoms (every claim verified against code 2026-06-12):

- **The M2 render is k-NN-blended flat colour, not texture.** `megakernel.slang::sdf_gather_atom_material` (line 463) inverse-distance-blends the K=6 nearest cooked atoms' RGB and shades Lambert with that single colour. The cook *sampled PolyHaven textures to colour those atoms* (`game_asset_cook.rs::atomize_mesh_triangles:1679` — `texture.sample_rgb(forge_box_uv(...))`), so the render shows the texture's *mean* at ~0.2–0.3 m atom pitch: clapboard courses, brick joints, shingle rows are all averaged away. The M2 acceptance proof is hue-bucket variety (7/12), not surface detail.
- **Sub-voxel geometry is flattened by construction.** The cooked craftsman SDF is 64×56×51 at ~0.22 m/voxel (`SDF_GPU_TARGET` `max_axis_res: 64`, `game_asset_cook.rs:2438`; the probe asset measures ~0.26 m). Window recesses (~5 cm), trim reveals (~3 cm), muntins (~2 cm) are far below one voxel: the facade sphere-traces as a flush plane. Windows survive *only* because glass-channel atoms mark them (the M2 mechanism); muntins, sills, and clapboard relief do not exist in the image at all.
- **The k-NN gather is a per-hit linear scan over every atom of the instance.** `sdf_gather_atom_material` loops `for (uint a = 0; a < cnt; a++)` over the instance's whole atom range — 12,229 iterations × ~4 buffer loads for the craftsman, **per SDF hit, per bounce**. At 320×320 that is ~5 × 10⁹ buffer loads/frame for primaries alone. Any added bounce (GI, glass) multiplies it. This is the single largest per-hit cost in the SDF path and it is O(asset atoms), not O(K).
- **Opaque SDF shading is a headlight hack, not light transport.** The opaque branch (megakernel.slang:1340–1356) is Lambert N·L from the directional rig **plus `albedo * (0.35 + 0.45 * N·V)` view-facing fill**, then **terminates the ray** (`g_next_ray_alive = 0`, line 1407). No shadow rays (a building never shades its neighbour), no GI bounce (no colour bleed, no sky occlusion), no contact darkening. It reads "clay model under a ring light," not "house."
- **The proof exists for ~1 archetype, not 100.** The M2 gate ran one cooked craftsman (12,229 atoms, 186 glass). The cook today produces 67 cooked atom sets (starter + directives + variants) of which the SDF+atoms+glass path has been exercised on exactly one. The atlas capacity claim (≈200 assets in 64 MB) and the per-asset cost (SDF + atoms + grids + textures) have never been measured across a full catalog.
- **Glass transmits into a hollow void.** The M2 glass continuation refracts into the building's empty SDF interior — the pane reads as dark blue glass over nothing. Real windows read as rooms.

---

## 2. Done When

**Headline (M1+M2 landed — wave 1), on the AMD 780M (RADV, Linux), seconds/frame acceptable:**

Running

```bash
cd ~/src/ochroma && SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
  --lib sdf_craftsman_textured_quality -- --nocapture --test-threads=1
```

prints

```
[sdf-quality] gather: grid=22 atoms/hit vs linear=12229 atoms/hit, speedup=78x, max|Δrgb|=0.0000 (identical image)
[sdf-quality] M1 texture: facade detail energy 4.1x flat-blend (gate >= 3.0x) -> PASS
[sdf-quality] M1 consistency: mean |hit_rgb - atom_gather_rgb| = 0.071 (gate < 0.12) -> PASS
[sdf-quality] M2 normals: facade normal-AOV variance 6.3x flat (gate >= 4.0x) -> PASS
[sdf-quality] M2 muntins: window crop crossings v=1 h=1 (DoubleHung expects v=1 h=1) -> PASS
[sdf-quality] M2 interior: 1063 window px, A/B vs hollow mean|ΔRGB|=0.21 (gate >= 200 px, >= 0.15) -> PASS
wrote sdf_craftsman_textured.png + sdf_craftsman_flat_control.png
```

and `sdf_craftsman_textured.png` passes eyeball: clapboard courses and shingle rows visible as texture, muntin bars crossing the panes, window glass showing a lit room instead of a void — where `sdf_craftsman_flat_control.png` (the M2 path, same camera) shows flat colour fields. A human at the keyboard verifies this without reading code.

**Phased Done When per milestone (M3–M5 are later waves; gates fixed now):**

- **M3 — Lighting reads real.** Same test binary, `sdf_block_lighting` test: a 2-building scene prints `contact band luminance -19% vs open facade (gate <= -15%)`, `sun/shade facade ratio 3.4 (gate in [2,6])`, `GI bounce adds +8% in shaded court (gate >= 5%)`, and the A/B PNGs (`L1` NEE-only vs `L2` +1 GI bounce) show a real shadow cast by one building onto the other.
- **M4 — 100 archetypes, one atlas.** `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin game_asset_cook` cooks ≥ 100 building archetypes printing per-asset `[sdf] <id> ...` lines and a final `[sdf-quality] 100 archetypes: atlas 24.1 MB / 64 MB quota, atoms 36.2 MB, grids 14.8 MB, textures 31.0 MB (34 sets)`; then the ochroma contact-sheet test renders all 100 (10×10 sheet, one instance each) printing `coverage >= 0.97: 100/100` and `distinct pairs (mean |ΔRGB| > 0.04): >= 95%` and writes `sdf_contact_sheet_100.png` where a human can see 100 visibly different, solid, textured buildings.
- **M5 (budget-gated stretch) — true window recesses.** Only if the perf track's budget B affords it (see §4.8): grazing-angle depth AOV inside window reveals differs ≥ 0.05 m from the facade plane, printed.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Cell-grid atom gather ≡ linear gather, ~80× cheaper | same camera, both gathers: `assert!(max_abs_rgb_dev < 1.0/255.0)` AND `assert!(atoms_tested_per_hit_grid < 80)` with linear = 12,229, both printed | `assert!(gather_returns_color)` |
| Textured SDF surface shows sub-atom detail | facade-crop mean gradient energy: `assert!(e_textured >= 3.0 * e_flat_blend)`, both energies printed from real renders | `assert!(png_nonempty)` |
| Texture layer stays consistent with the cooked atoms | `assert!(mean_abs_dev(hit_rgb, atom_knn_rgb) < 0.12)` over facade hits (the tint-normalized texture mean IS the atom colour) | `assert!(color != black)` |
| Channel→material dispatch (per-channel PBR) | roof crop roughness AOV ≠ facade crop roughness AOV (`assert!((r_roof - r_facade).abs() > 0.05)`), values printed; door crop albedo within 0.15 of cooked door material | one material id everywhere |
| Normal maps bend SDF gradient normals | facade normal-AOV variance ≥ 4× the unmapped render, printed | `assert!(normal.is_normalized())` |
| Muntins/trim detail at sub-voxel scale | count dark bar crossings in a window crop: `assert_eq!((v,h), expected_for(WindowStyle))`, printed | `assert!(window_pixels > 0)` |
| Glass panes are exact rectangles, not k-NN blobs | per-pixel glass classification from cooked aperture rects: `assert!(glass_px_outside_rects == 0)` with rect count printed | radius-only detection passing |
| Interior mapping behind glass | A/B vs hollow-interior control: `assert!(window_px >= 200 && mean_abs_rgb_dev >= 0.15)`, both printed | checking the pane is "not black" |
| NEE shadows + GI bounce on opaque SDF | contact-band test: `assert!(contact_lum <= 0.85 * open_lum)`; GI A/B: `assert!(court_lum_gi >= 1.05 * court_lum_nee)`, all printed | `assert!(image_changed)` |
| 100-archetype atlas residency + cost ledger | `assert!(atlas_bytes <= 64<<20)` after 100 inserts, per-asset and total bytes printed; `assert_eq!(cov_ge_097, 100)` over the contact sheet | loading 5 assets and extrapolating |

---

## 4. Architecture

### 4.1 The honest baseline and the one cost model everything plugs into

The proven state (all on the 780M, RADV): M0 one craftsman solid at **0.976** facade coverage (splat bridge: 0.58 confetti); M1 12-building block, 12/12 solid, 0.9475; M2 per-surface k-NN atom material with **real path-traced glass** (1,063 window pixels transmitting/refracting, A/B-proven against an opaque control; two 320×320 renders ≈ 10 s). The perf baseline the realtime track is attacking is **~9.3 s/frame at probe resolution on the 780M** — software sphere-trace, no HW RT, 1–2 ms target on the RTX 4070 Ti.

**Cost model.** Let the perf track's verdict be a per-frame budget **B** (total traced rays/frame at target frame time). A frame costs

```
R(W,H,spp,b,s) = W·H·spp · (1 + b + (1+b)·s)      rays/frame
   b = continuation bounces per path (glass + GI),  s = shadow (NEE) rays per path vertex
C_frame ≈ R · C_trace + H_hits · C_hit
   C_trace = one sphere-trace: ~100–384 steps × 8 trilinear loads ≈ 0.8k–3k buffer loads
   C_hit   = per-hit shading: gather + textures + ALU  (THE knob this design adds to)
```

The structural fact this design exploits: **everything in Track A except lighting is a `C_hit` item, not an `R` item.** Textures, normal maps, muntins, interiors add per-hit loads/ALU that are 10–30% of one `C_trace`; they survive *any* budget verdict. Lighting (NEE `s`, GI `b`) multiplies `R` and is therefore tiered (§4.7) and selected by B. Per-feature deltas, stated against `C_trace ≈ 800–3000 loads`:

| Feature | Δ per hit | Δ rays | Notes |
|---|---|---|---|
| k-NN gather TODAY | ~49k loads (12,229 atoms × 4) | 0 | **16–60× one C_trace — the bug** |
| k-NN gather via cell grid (§4.2) | ~110 loads (27 cells + ~22 atoms) | 0 | ~450× cheaper; ≈ 4–14% of C_trace |
| Box-UV + albedo/rough/normal, bilinear | 3 samples × 4 = 12 loads + ~60 ALU | 0 | ≈ 1% of C_trace |
| Same, EWA (primary hits only) | ≤ 3 × 64 = 192 loads | 0 | ≈ 6–24% of C_trace; secondary hits stay bilinear |
| Muntin/trim procedural detail | ~30 ALU, 0 loads | 0 | pure ALU on rect-local coords |
| Interior mapping behind glass | ~40 ALU + 1 sample | **−1 trace per transmitted glass ray** | *cheaper* than today's M2 continuation |
| NEE sun shadow (L1) | — | +1 trace per path vertex | ×~1.8 frame cost at b=0 |
| +1 GI bounce + its NEE (L2) | — | b=1, s=1 | ×~3.4 frame cost |
| +2nd GI bounce (L3) | — | b=2, s=1 | ×~5 frame cost |

**Given budget B**: spend `C_hit` items unconditionally (they are noise next to the trace), then choose the largest lighting tier with `R ≤ B`. On the 780M floor (stills, seconds) wave 1 runs L0/L1; the 4070-Ti tier selection is the realtime design's call, made trivial by this table.

### 4.2 Kill the linear gather first: per-asset atom cell grid

The M2 gather scans all 12,229 atoms per hit. Fix: at upload, sort each instance's atoms by cell of a per-asset uniform grid (cell = 0.5 m ≈ 2× atom pitch; craftsman 14×12×11 m → 28×24×22 = 14.8k cells); upload a `(start,count)` cell table + the cell-sorted atom SoA. The kernel gathers the 3×3×3 cell neighbourhood of the hit (≈ 27 cell headers + ~22 atoms at ~0.8 atoms/cell) — identical K-nearest result for any detect radius ≤ one cell, ~450× fewer loads. Memory: cell table ~118 KB + atoms re-sorted in place (no index buffer) → **+118 KB/asset, ~12 MB at 100 archetypes**. This must land *before* lighting: every added bounce re-pays the gather.

This is also the Lumen-shaped LOD lever (Lumen samples its low-res **surface cache** at SW-RT hits rather than evaluating materials — our atoms ARE that prefiltered appearance): when the pixel footprint at the hit (`t · pixel_angle`) exceeds the texture tile (2.5 m), **skip textures entirely and shade with the atom-gather colour** — the atoms are the correct prefiltered mip of the building's appearance. Texture sampling thus costs nothing exactly where it cannot be seen.

### 4.3 Textures on the SDF surface — replicate the cook's box projection at the hit (decision, with the rejected alternatives)

**Decision: dominant-axis box-projected UVs computed at the SDF hit, sampling the SAME PolyHaven maps through Spectra's existing flat texture atlas.** The cook colours atoms by `forge_box_uv` (game_asset_cook.rs:1711): asset-local `p = [x + w/2, y, d/2 − z]`, dominant normal axis picks the plane, 1 tile per 2.5 m. The megakernel reproduces this exactly: at an SDF hit, transform world→instance-local (the transform already exists — `sdf_instance_xform`), apply per-volume UV params `(w/2, d/2, 1/2.5)` (new 4-float/volume buffer), pick the plane from the gradient normal's dominant axis, sample. Because the cook and the kernel share one convention, **the texture layer lands exactly where the atom colours came from** — consistency is testable (gate: mean |hit RGB − atom-gather RGB| < 0.12; the TextureCache tint-normalization makes the texture mean equal the authored zone colour by construction).

The delivery vehicle already exists and is GPU-verified (spectra-texture-bridge memory, 2026-06-10): `Renderer::set_texture_atlas(descs, data, n)` + `sample_texture_ewa` + `MaterialData.albedo_tex/roughness_tex/normal_tex`, with the four landmines (warm-up frame, EWA UV wrap, backface gating, ≤256²/≤128² downsampling) already found and fixed. The SDF path binds the same atlas; nothing new is invented.

*Why not blended 3-tap triplanar:* the cook's atoms were sampled with hard dominant-axis selection; blending at runtime would diverge from the cooked colours near 45° normals and triple the fetch cost. Buildings are axis-dominant; hard selection matches both the content and the cook. *Why not UV-from-atoms (interpolating cooked per-atom UVs):* atoms do not store UVs (28 B/atom, `ReadyAssetAtom{position,scale,color,opacity,channel}`); adding them costs +8 B × 1.2 M atoms and k-NN-interpolated UVs swim at atom pitch — strictly worse than recomputing the deterministic projection. *Why not baking higher-res atom colours:* halving atom pitch quadruples atom count and still shows blur at 10 cm; textures are the correct frequency separation. *Alignment honesty:* the SDF surface sits ≤ voxel/2 (~0.11 m) off the true mesh surface, so the projection can shift by ≤ 4% of a tile — invisible at 2.5 m tiling; the consistency gate is on the mean, which is invariant to this shift.

### 4.4 Channel → material dispatch: the cooked per-channel PBR table

M2 dispatches on 9 atom channel ids (`ATOM_CH_FACADE..OTHER`) with glass→`sample_glass` and everything else→flat Lambert. This design completes the table: the cooked payload already carries **per-channel PBR materials** (`ReadyAssetPbrMaterial { channel, base_color_factor, metallic_factor, roughness_factor, textures: ReadyAssetTextureSet }` — asset/mod.rs:231) with `polyhaven://` URIs resolved by the game's `TextureCache` (textures.rs, 19 pinned sets). Host packs, per instance, a 9-slot channel→material-id table (`g_sdf_channel_material`, 9 floats/instance) pointing into the standard `MaterialLayer` materials (so `albedo_tex/roughness_tex/normal_tex/base_color/uv_scale` ride the existing `MaterialData` layout — no parallel material system). At a hit: gather channel (k-NN, gridded) → table → `MaterialData` → textured Lambert/Oren-Nayar with per-channel roughness; glass keeps its M2 BSDF route. Engine stays game-agnostic: vox_render sees generic channels + `PbrMaterial`/`TextureImage`, never "building".

Texture memory ledger (f32 flat atlas, the existing format): per PolyHaven set ≈ 256²·3 (diffuse) + 128²·3 (normal) + 256²·1 (rough) floats ≈ **1.05 MB**. ~30–34 sets for 100 archetypes ≈ **31–36 MB**. Tints are applied host-side per (set,tint) today; to stop tint-variants multiplying entries, upload UNTINTED sets once and move the mean-normalized tint factor into `MaterialData.base_color` (multiply in shader — the kernel already multiplies base_color when no texture; for textured materials the albedo texel REPLACES base_color today, so the tint moves to a per-material `base_color` multiply — small kernel change, listed in wiring). Quantizing the atlas to rgba8 (4× saving) is Open Question #3, not wave-1.

### 4.5 Fine detail WITHOUT sub-voxel SDF — the resolution arithmetic that forces normal maps

The recess/muntin/trim problem cannot be solved with SDF resolution, and the design must say so with numbers. To *geometrically* hold a 5 cm window reveal needs ≤ 2.5 cm voxels → 560³ for a 14 m building → 175 M voxels **dense (impossible)**; even narrow-band sparse (8³ bricks, AMD GDC-2023-style, ~636 m² of craftsman surface × ±2-voxel band at 2.5 cm) ≈ 8 M voxels ≈ **8 MB/asset → 800 MB at 100 archetypes (rejected on the 780M UMA)**. Raising the base cook to 96³ (0.148 m voxels, +884 KB/asset, SDF Pillar hero tier) still leaves every target feature sub-voxel. Conclusion: **base SDF stays 64³ massing; all sub-voxel appearance is shading**, in three cheap layers (this is the relief/normal-mapping discipline games have used since GPU Gems 3's relaxed cone stepping, and what MIP-plicits/Nested-Neighborhoods do for neural SDFs — detail as normal field over a coarse base):

1. **PolyHaven normal maps per channel** (clapboard relief, brick joints, shingle courses): tangent frame is trivial for a box projection (the two in-plane axes ARE tangent/bitangent); perturb the SDF gradient normal through the existing `normal_tex` path. Cost: 1 texture sample/hit.
2. **Procedural aperture detail** (the sub-voxel features with *identity*): muntin bars, sash split, frame/sill band, recess bevel — generated analytically in window-rect-local coordinates (§4.6) from `WindowStyle` (forge `description.rs`: DoubleHung/Casement/Bay/Lancet/Sash). A hit inside a bar re-classifies to TRIM (painted opaque) and bends the normal; a hit within 6 cm of the rect edge gets a bevel normal + contact AO darkening (the recess *shading* without recess geometry). Cost: ~30 ALU, zero fetches.
3. **(M5, budget-gated) window-cell detail SDF**: one cooked high-res SDF slab per WindowStyle (~1.2×1.5×0.16 m at 1 cm ≈ 120×150×16 ≈ 288 KB snorm8, **shared by every window of that style in the city** — 5 styles ≈ 1.4 MB total), CSG-subtracted during the last sphere-trace steps inside an aperture rect for true grazing-angle parallax and silhouette. Cost: ~10–20 extra trace steps for rays entering aperture rects only (~2–5% of pixels). Gated on B because it lengthens the trace itself.

### 4.6 Aperture rects: windows become exact rectangles (cook-side derivation, no Forge change)

The M2 window mechanism (any glass atom within `0.6 × voxel_max` of the hit) wobbles at the pane boundary by ~atom pitch and gives muntins nothing to anchor to. The cook can do better with data it already has: the forge contract's `material_zones` name the glass triangles (`window_glass`); cluster glass-zone triangles by connectivity, fit each cluster an oriented rect (dominant normal + in-plane PCA), and emit per-asset **aperture rects** `{center, right, up, half_extents, style_id}` (~32 rects × 56 B ≈ 1.8 KB/asset; serde-default so old payloads load). The kernel classifies glass by point-in-rect (exact pane bounds; k-NN radius stays as fallback for assets without rects), anchors muntin/sill/bevel detail in rect-local (s,t), and gives interior mapping its room frame. Engine naming stays generic (`SdfApertureRect` — an optical aperture, not a game concept).

### 4.7 Lighting that reads real — tiered NEE + GI replacing the headlight (the `R` spenders)

Replace the opaque branch's matcap fill with measured transport, in tiers selected by B (§4.1 table):

- **L1 — NEE direct:** at each opaque hit, 1 shadow ray (sphere-trace toward the sun, capped 96 steps + sky visibility from the existing dome) — buildings finally shadow each other and themselves; contact darkening appears where facades meet ground. The view-fill hack is deleted; the sky dome provides ambient.
- **L2 — +1 cosine GI bounce:** the opaque hit stops terminating (`g_next_ray_alive = 1` with cosine-sampled `wi`, Lambert throughput) for one bounce — colour bleed, sky occlusion in courts, soft ambient gradients. The M2 glass continuation already proves the next-ray plumbing works for SDF hits.
- **L3 — 2 bounces:** stills/cinematic tier.

On the 780M these tiers cost ×1.8 / ×3.4 / ×5 of today's frame (table §4.1) — acceptable for stills, and exactly the numbers the perf agent's budget B prices for realtime (ReSTIR/radiance-cache amortization of L2 is the realtime design's machinery, cited not restated).

**Interior mapping behind glass** (SimCity-2013/Spider-Man lineage, Joost van Dongen): transmitted glass rays stop re-tracing the scene into a hollow void; in rect-local space intersect an analytic box room (depth 3 m), shade procedural walls/floor/ceiling + per-window hashed warm emissive — ~40 ALU + 1 fetch, REPLACING a full `C_trace` continuation per transmitted ray (a net frame *saving* at ~2–5% glass pixels). Fresnel *reflection* continuation (neighbouring buildings/sky in the pane) is kept when B affords it: +1 trace on glass pixels only.

### 4.8 Scale to 100 archetypes — generator, atlas, and the cost ledger

**Variety source:** Forge already exposes 10 `Style`s × 4 `RoofStyle`s × 5 `WindowStyle`s × 4 `Condition`s plus footprint/floors/porch/seed knobs (`BuildingDescription`), driven through the shipped `.asset.json` directive pipeline (41 directives on disk: 32 nyc + 6 residential + 3 commercial; 67 cooked atom sets including variants). Reaching 100 *distinct* archetypes = authoring ~40 more directives across under-represented programs (row-houses, duplexes, corner-commercial, civic) + per-directive variants — cook work, no engine change. New styles need their PolyHaven stems pinned in `textures.rs::STEM_SETS` + `polyhaven_fetch` manifest (each unmapped stem falls back to flat zone colour — visible, not fatal).

**Per-asset GPU cost (measured shape, 100-archetype ledger):**

| Item | Per asset | ×100 | Quota / note |
|---|---|---|---|
| SDF field (64³ cap, snorm8 atlas boxes, 1-voxel border) | ~190–290 KB | 19–29 MB | 64 MB `SdfAtlasGpu` quota; capacity ≈ 200+ typical assets — **verified headroom ≥ 2×** |
| Atoms (SoA pos+rgb+channel, 28 B) | ~340 KB (12 k atoms) | 34 MB | asset-LOCAL dedup required at city scale (universal-renderer seam §4.5); contact sheet (1 instance each) costs the same 34 MB either way |
| Atom cell grid (0.5 m cells, start+count) | ~118 KB | 12 MB | §4.2 |
| Aperture rects | ~2 KB | 0.2 MB | §4.6 |
| Textures (shared per PolyHaven set, f32) | — | 31–36 MB (30–34 sets) | §4.4; rgba8 quantization = 4× saving, Open Question #3 |
| **Total new resident** | ~0.65 MB/asset + shared | **≈ 96–111 MB** | vs the ~370 MB splat-chain budget this path replaces on the 780M UMA |

**Scale honesty:** the M2 host path uploads atoms PER INSTANCE in world space (`pathtrace_sdf_scene_with_atoms_to_rgba` transforms each instance's atoms) — fine for ≤ ~100 instances (contact sheet), ruinous at 10 k placements (120 M atoms). The asset-local atom buffer + per-instance indirection belongs to the universal renderer's scene-feed seam and is **cited as the M-scale obligation, not re-solved here**; M4's contact sheet (100 instances) stays inside the world-space mechanism deliberately.

### 4.9 Threading / ownership / boundaries

Unchanged from the universal renderer: the game cook produces SDF + atoms + per-channel materials + aperture rects (game-side, `game_asset_cook.rs`); vox_render's `splat_backend` packs generic buffers (`SdfLayer` extensions, `PbrMaterial`/`TextureImage`) on the render/test thread; Spectra owns the trace. No game noun crosses into vox_render or spectra (channels and aperture rects are material/optics vocabulary, already shipped). All renders remain one-shot blocking calls on the caller's thread (the proven M0–M2 model). **Coordination rail:** a perf agent is concurrently editing `megakernel.slang`; all kernel work in this track lands as additive, uniform-gated blocks (`u_sdf_textured`, default 0 = byte-identical frames), rebased onto the perf agent's head before merge.

### 4.10 SOTA grounding (verified 2026-06-12)

- **Material-at-low-frequency + detail-at-hit is the shipped industry pattern:** Lumen's software-RT tier shades hits from a low-res **surface cache** atlas rather than evaluating full materials ([Lumen Technical Details](https://dev.epicgames.com/documentation/unreal-engine/lumen-technical-details-in-unreal-engine), [Mesh Distance Fields](https://dev.epicgames.com/documentation/en-us/unreal-engine/mesh-distance-fields-in-unreal-engine)) — our atoms are exactly that cache; this design adds the near-field texture layer Lumen gets from raster.
- **Triplanar/box projection is the standard SDF texturing answer** (no UVs on implicit surfaces): [iquilezles raymarching distance fields](https://iquilezles.org/articles/raymarchingdf/), [shaderbits SDF ray tracing](https://shaderbits.com/blog/distance-field-ray-tracing-part1), [electricsquare raymarching workshop](https://github.com/electricsquare/raymarching-workshop). We use the cook's own dominant-axis variant for cook/runtime consistency.
- **Detail-as-normal-field over a coarse implicit base** is current research practice: neural implicit **normal mapping** transfers fine-SDF detail onto a coarse base without parameterization ([MIP-plicits](https://deepai.org/publication/mip-plicits-level-of-detail-factorization-of-neural-implicits-sphere-tracing), [Nested Neighborhoods, arXiv 2201.09147](https://arxiv.org/abs/2201.09147)); classical relief detail via [Relaxed Cone Stepping, GPU Gems 3 ch. 18](https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-18-relaxed-cone-stepping-relief-mapping).
- **Sparse/narrow-band SDF bricks** (the rejected-with-arithmetic alternative for sub-voxel geometry): [AMD GPUOpen, Real-Time Sparse Distance Fields for Games, GDC 2023](https://gpuopen.com/gdc-presentations/2023/GDC-2023-Sparse-Distance-Fields-For-Games.pdf) (8³ bricks) — the §4.5 math shows why even bricks blow the UMA budget at 2.5 cm city-wide; the M5 *shared per-style detail slab* is the affordable form.
- **Interior mapping** for rooms behind glass without geometry: [Joost van Dongen's interior mapping](http://joostdevblog.blogspot.com/2018/09/interior-mapping-real-rooms-without.html), [80.lv overview](https://80.lv/articles/interior-mapping-rendering-real-rooms-without-geometry) — used by SimCity 2013 and Insomniac's Spider-Man; here it additionally *saves* a trace per transmitted ray.

---

## 5. Data Models

```rust
// ── game cook (urban_horizon/src/asset/mod.rs) — cooked payload addition ──────
/// One glass aperture (window/door-light) fitted from the forge glass-zone
/// triangles at cook. Local space = asset-local (same space as atoms + SDF).
/// serde-default so v1 payloads load unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReadyAssetApertureRect {
    pub center: [f32; 3],
    pub right: [f32; 3],        // unit, in-plane
    pub up: [f32; 3],           // unit, in-plane
    pub half_extents: [f32; 2], // metres along right/up
    pub style_id: u8,           // forge WindowStyle as u8 (DoubleHung=0 ...)
}
// ReadyAssetPayload gains: #[serde(default)] pub apertures: Vec<ReadyAssetApertureRect>,
```

```rust
// ── vox_render::splat_backend (engine, spectra-native) ──────────────────────
/// Generic optical aperture on an SDF instance (engine-side mirror of the
/// cooked rect; world-space after instance transform, packed SoA for the GPU).
#[derive(Debug, Clone, Copy)]
pub struct SdfApertureRect {
    pub center: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub half_extents: [f32; 2],
    pub style_id: u32,
}

/// Per-instance material dispatch for the textured SDF path: channel id
/// (0..=8, the SDF_ATOM_CH_* order) -> index into the `materials` slice.
#[derive(Debug, Clone, Copy)]
pub struct SdfChannelMaterials {
    pub material_for_channel: [i32; 9], // -1 = flat atom colour (M2 behaviour)
}

/// Per-volume UV params replicating the cook's forge_box_uv convention.
#[derive(Debug, Clone, Copy)]
pub struct SdfUvParams {
    pub offset_x: f32,   // forge_width  * 0.5
    pub offset_z: f32,   // forge_depth  * 0.5
    pub tile_recip: f32, // 1.0 / 2.5  (tiles per metre)
}
```

```
// ── megakernel.slang — new buffers (all uniform-gated, default-off) ─────────
StructuredBuffer<float> g_sdf_atom_cell_table;   // (start,count) per cell, per instance grid
StructuredBuffer<float> g_sdf_atom_grid_headers; // per instance: cell_dims[3], cell_size, table_offset, atom_base
StructuredBuffer<float> g_sdf_channel_material;  // 9 floats / instance -> material index
StructuredBuffer<float> g_sdf_volume_uv_params;  // 4 floats / volume (offset_x, offset_z, tile_recip, pad)
StructuredBuffer<float> g_sdf_aperture_rects;    // 15 floats / rect (world-space), per-instance (offset,count) range
StructuredBuffer<float> g_sdf_instance_aperture_range; // 2 floats / instance
uniform int u_sdf_textured;        // 0 = M2 behaviour byte-identical, 1 = textured path
uniform int u_sdf_lighting_tier;   // 0 = M2 headlight, 1 = NEE, 2 = +1 GI bounce, 3 = +2
```

GPU layout constraints: atom SoA buffers keep their shipped M2 layout (3/3/1 floats per atom) — the grid only *re-orders* atoms within an instance range and adds the cell table; `SdfLayer::from_parts_with_atoms` stays valid and gains a sibling constructor. `MaterialData` keeps its verified 156-float Vulkan layout (albedo_tex=a[28], roughness_tex=a[29], normal_tex=a[30] — spectra-texture-bridge memory); the channel table stores indices into it.

---

## 6. API

```rust
// ── vox_render::splat_backend (additive; M0–M2 entry points unchanged) ──────
/// Textured SDF still: SDF massing + cell-gridded atom material + per-channel
/// PBR textures + aperture-rect glass/detail + tiered lighting. `materials`
/// and `textures` ride the SAME packing as pathtrace_mesh_textured_to_rgba
/// (set_texture_atlas AFTER load_scene_state — order matters).
/// `lighting_tier`: 0 headlight (M2 parity), 1 NEE, 2 NEE+1 GI bounce.
/// Errors: String, naming the failing buffer/validation (M0–M2 convention).
#[allow(clippy::too_many_arguments)]
pub fn pathtrace_sdf_scene_textured_to_rgba(
    volumes: &[SdfVolumeInput],
    uv_params: &[SdfUvParams],                 // len == volumes.len()
    instances: &[SdfSceneInstance],
    atoms_per_instance: &[Vec<SdfSceneAtom>],  // local space, as M2
    apertures_per_instance: &[Vec<SdfApertureRect>], // local space
    channel_materials: &[SdfChannelMaterials], // len == instances.len()
    materials: &[PbrMaterial],
    textures: &[TextureImage],
    eye: [f32; 3], target: [f32; 3], fov_y: f32,
    width: u32, height: u32, spp: u32,
    lighting_tier: u32,
    rig: &LightRig,
) -> Result<Vec<u8>, String>;

// Existing signatures relied on verbatim (verified in code — do NOT re-derive):
//   pathtrace_sdf_scene_with_atoms_to_rgba(volumes, instances, atoms_per_instance,
//       eye, target, fov_y, width, height, spp, rig) -> Result<Vec<u8>, String>
//   pathtrace_mesh_textured_to_rgba(..., materials: &[PbrMaterial],
//       textures: &[TextureImage], ...)            // the texture-packing oracle
//   PbrMaterial { base_color:[f32;3], roughness, metallic, emission_strength,
//       albedo_tex:i32, roughness_tex:i32, normal_tex:i32, uv_scale:[f32;2] }
//   TextureImage { width, height, channels, data: Vec<f32> }   // LINEAR, caller decodes sRGB
//   SdfVolumeInput { resolution:[u32;3], origin:[f32;3], voxel_size, narrow_band, distances }
//   SdfSceneInstance { volume_index, position, rotation_xyzw, uniform_scale, albedo }
//   SdfSceneAtom { position:[f32;3], color:[f32;3], channel:u32 }
//   sdf_atom_channel_from_str(&str) -> u32
//   Renderer::set_texture_atlas(&[u32], &[f32], u32) -> RenderResult<()>  // AFTER load_scene_state
//   SdfLayer::from_parts_with_atoms(volumes, instances, distances, albedo,
//       atom_positions, atom_colors, atom_channels, instance_atom_range) -> SdfLayer
//   megakernel: sdf_gather_atom_material(int inst, float3 world_p, float radius)
//       -> SdfAtomGather { color, channel, is_glass, nearest_d }
//   megakernel: sdf_render_intersect(ro, rd, t_max, out normal, out inst) -> t
//   megakernel: sample_glass(wo, n, ior, rough, ..., u1, u2, u3) -> BSDFSample
//   megakernel: sample_texture_ewa(g_tex_descs, g_tex_data, u_num_textures, tex_id, uv, ddx, ddy) -> float4
// Threading: one-shot blocking render on the caller's thread (M0–M2 model).
// Env: SPECTRA_BACKEND=vulkan, VK_ICD_FILENAMES=radeon_icd.json, scripts/build-spectra-native.sh.
```

```rust
// ── game cook (urban_horizon game_asset_cook.rs) ─────────────────────────────
/// Fit aperture rects from the forge contract's glass-zone triangles.
/// Deterministic: cluster glass triangles by shared edges, fit oriented rect
/// (dominant normal, in-plane PCA). Returns asset-local rects.
fn derive_aperture_rects(mesh: &ForgeMesh, zones: &[ForgeMaterialZone],
                         desc: &Value) -> Vec<ReadyAssetApertureRect>;
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| Cell-grid build (sort atoms by cell, emit table) | `pathtrace_sdf_scene_textured_to_rgba` upload | `crates/vox_render/src/splat_backend.rs` | host-side, per instance; parity-tested vs linear gather |
| `sdf_gather_atom_material` grid rewrite | existing call sites in the SDF shade block | `~/src/spectra/slang/megakernel.slang` | gated `u_sdf_textured`; linear path kept as the parity oracle |
| `SdfLayer` grid/material/aperture extensions | `from_parts_textured` (new sibling ctor) | `~/src/spectra/rust/spectra-scene-state/src/layers.rs` | additive fields, M2 ctor untouched |
| Buffer upload + binds | `GpuScene::bind_to_map` | `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` | mirrors the M2 atom-buffer pattern |
| Box-UV + textured shade + channel dispatch | SDF shade block, opaque branch | `~/src/spectra/slang/megakernel.slang` | EWA primary / bilinear secondary; footprint > tile ⇒ atom colour only (§4.2 LOD) |
| Normal-map TBN for box projection | same opaque branch | `~/src/spectra/slang/megakernel.slang` | in-plane axes = tangent frame |
| Aperture-rect glass classification + muntin/bevel detail | glass + opaque branches | `~/src/spectra/slang/megakernel.slang` | point-in-rect first, k-NN radius fallback |
| Interior box-room shade | glass transmission branch | `~/src/spectra/slang/megakernel.slang` | replaces the hollow continuation; reflection continuation kept budget-gated |
| NEE shadow trace + GI continuation (tiers) | opaque branch terminate site (line ~1407 today) | `~/src/spectra/slang/megakernel.slang` | `u_sdf_lighting_tier`; reuses next-ray plumbing the M2 glass proved |
| `derive_aperture_rects` + payload field | cook per-recipe body, beside `ready_sdf` | `~/Ochroma/projects/urban_horizon/src/bin/game_asset_cook.rs` + `src/asset/mod.rs` | prints `[apertures] <id>: <n> rects` |
| Per-channel material packing (cooked PBR → `PbrMaterial` + atlas) | textured test/probe harness | `crates/vox_render/src/splat_backend.rs` tests + civitas probe bin | reuses `TextureCache` (downsampled, tint-normalized) |
| 100-directive cook + contact sheet | `game_asset_cook` + new ochroma test `sdf_contact_sheet_100` | civitas directives dir + `crates/vox_render/src/splat_backend.rs` | M4; prints the §2 ledger |

---

## 8. Open Questions

- [ ] **Tint-in-shader vs tint-baked textures.** §4.4 proposes untinted uploads + `base_color` multiply to stop (set,tint) entry multiplication; the kernel currently REPLACES base_color with the texel for textured materials. Decide at M1 implementation: either keep host-baked tints (more atlas MB, zero kernel change) or add the multiply (one-line kernel change, perf-agent coordination). Wave-1 plan assumes host-baked (zero kernel risk); revisit at M4 if atlas MB bites.
- [ ] **EWA footprint without ray differentials on the SDF path.** Primary hits can approximate an isotropic footprint from `t · pixel_angle`; if EWA shimmer appears in the M1 render, fall back to bilinear + the §4.2 footprint LOD and close the question with the measured image.
- [ ] **rgba8 texture atlas quantization** (4× memory, bandwidth win) — needs a kernel fetch change; decide at M4 with the measured 31–36 MB ledger in hand.
- [ ] **Asset-local atom dedup at city scale** — owned by the universal-renderer scene seam; this track must not ship anything that *blocks* it (the cell grid is per-asset-local already, so it dedups for free when atoms do).
- [ ] **M5 detail-SDF slab trigger:** which budget B makes +10–20 trace steps on 2–5% of pixels affordable, and does the recess read at city camera distances at all? Measure after M2 with a grazing-angle still before building anything.

---

## 9. Out of Scope

- **The realtime millisecond ladder** (HW RT, ReSTIR, DLSS-RR, the 1–2 ms budget) — owned by [Spectra Realtime](./2026-06-10-spectra-realtime-design.md); this design only prices its features against that track's budget verdict B.
- **Gaussian-as-geometry rendering** — rejected by direction; atoms remain an appearance field only.
- **Asset-local atom/instance dedup and the live `SceneUpdateBatch` feed** — the universal renderer's seam (§4.8 honesty note).
- **Interior GEOMETRY behind glass** (real rooms, furniture SDFs) — interior mapping only.
- **Terrain, scatter/grass, roads texturing** — same machinery later; this track is housing.
- **New Forge geometry features** (real recess modeling, higher window detail in the mesh) — the cook consumes Forge as-is; Forge upstream work is its own backlog.
- **snorm16 / 96³ hero SDF upgrades** — SDF Pillar's open question; nothing here depends on it.

---

## 10. Related Plans / Designs

- **Implemented by:** [SDF Housing Quality — Wave 1 (M1+M2)](../plans/2026-06-12-sdf-housing-quality-wave1.md); M3 (lighting), M4 (100-type), M5 (detail slab) get their own waves after wave-1 metrics land.
- **Depends on:** the shipped M0–M2 SDF path (spectra `1480e2c`, ochroma `160469b`), `SdfBaker`/`SdfAtlasGpu` (SDF Pillar M1, shipped in the cook via `SDF_GPU_TARGET` 64³), the verified texture bridge (`set_texture_atlas` + `pathtrace_mesh_textured_to_rgba`), the civitas `TextureCache` (downsampling + tint normalization, ALL-P1-1), the forge directive pipeline (`.asset.json` → `generate_asset`).
- **Coordinates with:** the Spectra perf agent currently editing `megakernel.slang` — all kernel additions are uniform-gated and rebased onto its head; the budget verdict B it produces selects the §4.7 lighting tier and the M5 gate.
- **Related:** `spectra-pathtracer-runs` memory (run env recipe), `spectra-texture-bridge` memory (atlas semantics + landmines), [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (the UMA budget this path inherits), CS2 asset model memory (theme/archetype breadth target).

---

## Milestone Ladder (metric-led; every gate from §2/§3)

**M1 — Textured single building (wave 1).** Cell-grid gather (parity + ≥20× speedup printed) + per-channel PBR + box-UV textures at the hit. *Gate:* facade detail energy ≥ 3× flat-blend; consistency < 0.12; roof/facade roughness differ. *Cost:* +~120 loads/hit (≈ 5% of C_trace), −49k loads/hit from the grid (net **cheaper** than M2), 0 extra rays; +~20 MB textures (starter sets).

**M2 — Detail crispness (wave 1).** Normal maps + aperture rects + procedural muntins/sill/bevel + interior mapping. *Gate:* normal-AOV variance ≥ 4×; muntin crossings == WindowStyle expectation; interior A/B ≥ 0.15 over ≥ 200 px. *Cost:* +1 fetch + ~70 ALU/hit; transmitted glass rays get **cheaper** (interior map replaces a full re-trace); +1.8 KB/asset rects.

**M3 — Lighting reads real.** NEE shadow rays + tiered GI replacing the headlight. *Gate:* contact band ≤ −15%; sun/shade ratio in [2,6]; GI court A/B ≥ +5%. *Cost:* ×1.8 (L1) / ×3.4 (L2) / ×5 (L3) frame rays — the only `R`-budget spender; tier chosen by the perf verdict B.

**M4 — 100 archetypes, one atlas.** ~40 new directives + variants → ≥100 cooked; contact-sheet render. *Gate:* 100/100 coverage ≥ 0.97; ≥95% pairs distinct; ledger printed ≤ quotas (atlas ≤ 64 MB; total new resident ≈ 96–111 MB). *Cost:* cook-time only + the §4.8 ledger.

**M5 — True window recesses (budget-gated stretch).** Shared per-style detail-SDF slabs (~1.4 MB total), CSG inside aperture rects. *Gate:* reveal depth AOV ≥ 0.05 m at grazing angle. *Cost:* +10–20 trace steps on 2–5% of pixels — only if B affords it; killed without ceremony if the M2 bevel already reads at game cameras.
