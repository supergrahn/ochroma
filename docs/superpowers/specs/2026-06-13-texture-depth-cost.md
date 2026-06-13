# Design: Real Surface Depth in Facades — POM vs Displaced Micro-Geometry, Cost vs Budget (2026-06-13)

**Status:** Draft
**Scope:** Decide how Spectra gives building facades REAL/perceived depth (mortar recesses, brick relief, panel reveals, window reveals) beyond flat normal-map shading — quantify GPU-time and VRAM cost against the 60 fps @ 4K / RTX 4070 Ti / 100K instances / ~300 types budget, and recommend a technique split.
**Related:** `hybrid-atom-sdf-lod-direction` (MESH + RTX-MG render direction), `spectra-sdf-perf-verdict` (measured 780M→4070 Ti tier), `facade-photorealism-roadmap` (Wave 3 = facade depth).

---

## 1. Problem Statement

- Spectra's megakernel applies a tangent-space normal map at the triangle hit (`~/src/spectra/slang/megakernel.slang:1974-1992`, builds TBN from triangle edges/UVs, `apply_normal_map` from `normal_map.slang`). This perturbs *lighting only* — the silhouette stays flat, there is **no parallax**: brick relief, mortar recesses, and panel reveals do not shift as the camera moves, and grazing-angle views read as a painted-on flat wall.
- The material model already reserves height/displacement inputs — `MaterialData.displacement_tex / displacement_scale / displacement_midlevel` (`~/src/spectra/slang/material_types.slang:69-71`) — but **nothing reads them**: the megakernel material decode never unpacks those fields (grep of `displacement` in `megakernel.slang` returns only `import`s, no struct read), and the engine packer hard-writes `a[31] = pack_i32(-1)` for `displacement_tex` (`~/src/ochroma/crates/vox_render/src/splat_backend.rs:2612`, midlevel `a[33]=0.5` at :2613). The depth input is wired to OFF end-to-end.
- There is **no parallax / POM / relief-mapping code path anywhere** in Spectra's shading kernels — `grep -i parallax` over `slang/` and `rust/` hits only vendored FidelityFX SSR/denoiser reprojection, never a height-field UV raymarch. POM is net-new.
- The RTX Micro-Geometry CLAS path that would give true displaced silhouettes is a **stub on every tier**: `SubdivisionPipeline::build_cas` returns `None` with "RTXMG hardware detection would go here" (`~/src/spectra/rust/spectra-rtxmg/src/subd.rs:106,157-182`); `trace` returns `f32::INFINITY`. The "displaced terrain" we built is CPU-tessellated geometry, not a hardware micromap.

---

## 2. Done When

This is a **design/decision doc** (read-only investigation). It is "done" when:

Running `ls ~/src/ochroma/docs/superpowers/specs/2026-06-13-texture-depth-cost.md` shows the file, and a human reading §3–§8 can answer, without reading code: (a) which depth techniques are Spectra-feasible vs net-new with exact `file:line` citations; (b) the per-technique GPU-ms and VRAM cost envelope against 60 fps @ 4K on a 4070 Ti; (c) the recommended near/hero-vs-mass split. No code is changed by this doc; the follow-on *plan* (separate file) owns the `Done When` render-gate command.

---

## 3. Capabilities (what a follow-on plan would build, and the real test for each)

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| **POM in the megakernel** (height-field UV raymarch in tangent space, replaces flat normal lookup on tagged materials) | Render a brick wall at a 70° grazing angle, camera dollying laterally: assert the brick-vs-mortar boundary in screen-space shifts by `> 2 px` between frame A and frame B (parallax), and self-shadowed mortar pixels are `> 20%` darker than face pixels. Diff two stored PNGs. | `assert!(pom_enabled)` / "normal map still applied" — passes with zero parallax |
| **Packer writes a real height texture id** | After cook, assert `material.displacement_tex >= 0` AND the megakernel-unpacked `mat.displacement_scale > 0.0` for a brick material (read back the packed atom buffer) | `assert!(displacement_tex != 0)` — passes with the current hard `-1` only if sign flips |
| **Displaced micro-mesh BLAS for hero facades** (per-type displaced mesh → KHR BLAS, instanced) | Build the BLAS for one displaced facade type; assert `blas.triangle_count == coarse_tris * 4^level` (real subdivision happened) and a primary ray grazing the wall returns a hit `t` that differs from the flat-wall `t` by `>= displacement_scale * 0.5` (true silhouette) | `build_cas` returns `Some(handle)` but `trace` returns `INFINITY` — current rtxmg stub |
| **LOD selector** (displace near/hero, POM mid, normal-map far) | At 5 m camera distance assert path = displaced-BLAS; at 40 m assert path = POM; at 150 m assert path = normal-map-only — read the per-instance technique tag the selector writes | `assert!(lod_level <= 2)` — passes without distance logic |

---

## 4. Architecture

### 4.1 Technique 1 — Parallax Occlusion Mapping (POM) in the megakernel [NET-NEW, cheap, recommended for MASS]

POM raymarches a height field in **tangent space** to find where the view ray actually pierces the relief, then offsets the UV used for albedo/normal/roughness lookups and optionally self-shadows. No geometry, no BLAS, no silhouette change — the triangle stays flat but the *interior* gets parallax + self-occlusion, which is exactly what reads as mortar recesses and panel reveals from normal viewing distance.

**Spectra slots it in with no new infrastructure.** Everything POM needs already exists at the hit point in `megakernel.slang`:
- Per-hit TBN basis is already computed from triangle edges + UV deltas (`megakernel.slang:1981-1990`) — reuse verbatim.
- View direction `wo` is in scope (`megakernel.slang:661` signature; used at :1991 `dot(normal, wo)`).
- UV derivatives `ddx_uv`/`ddy_uv` are computed (`megakernel.slang:1554-1673`) for mip/EWA — POM uses them to LOD the step count.
- `apply_uv_transform(hit_uv, …)` already produces `tex_uv` (`megakernel.slang:1914`).
- A height texture is sampleable via the same `sample_texture_ewa(g_tex_descs, g_tex_data, …)` used for albedo/normal (`megakernel.slang:1976`).

The new code is a ~30-line function: transform `wo` into tangent space (`Vtan = (dot(wo,T), dot(wo,B), dot(wo,N))`), march `N` linear steps from height 1→0 stepping the UV by `Vtan.xy/Vtan.z * displacement_scale / N`, refine the crossing with one linear interpolation (the standard Tatarchuk POM), then feed the parallax-corrected UV into the existing albedo/normal/roughness samples. Self-shadow = a second short march toward the light. Gate on `mat.displacement_tex >= 0` (today always false → zero cost when off).

Wiring blockers to clear in the plan: (1) megakernel must **unpack** `displacement_tex/scale/midlevel` into its material struct — it currently doesn't (no read in `megakernel.slang`); (2) packer must stop writing `-1` (`splat_backend.rs:2612`) and route a cooked height-map texture id. The `material_types.slang:69-71` fields and the `spectra-displacement` crate's `DisplacementMap::sample_uv` (`~/src/spectra/rust/spectra-displacement/src/lib.rs:62`) already define the data contract.

### 4.2 Technique 2 — Real displaced micro-geometry (true silhouette) [NET-NEW build path, expensive, recommended for NEAR/HERO only]

Real depth + real silhouette needs real geometry. Two sub-paths:

- **CPU/GPU pre-tessellated displaced mesh → KHR BLAS (works on our box today).** Spectra already has the full displacement machinery: `displacement.slang`'s 5-pass adaptive subdivision (`displace_pass` at :312 offsets each vertex along its normal by the height sample, then `normal_recompute_pass`), `patch_eval.slang:575-602` for subdiv-surface displacement, and the `spectra-displacement` crate (`apply`, `recompute_normals`, `DisplacementEvaluator::compute_level`). The output triangle soup feeds the **real, GPU-verified KHR hardware-RT BLAS+TLAS build** — `SceneUploader{use_hw_rt:true}` builds BLAS+TLAS on the GPU and `trace_bvh_hw`/`trace_shadow_hw` (`~/src/spectra/slang/rt_query.slang:4,83-101`, RayQuery inline) return hits matching the software BVH (proven on RADV in `~/src/spectra/rust/spectra-scene-upload/tests/khr_rt_gpu.rs`). **This contradicts the older "Spectra has ZERO wired hardware RT" note — KHR HW-RT is real and verified.** So a displaced facade BLAS is buildable and traceable on the 780M and the 4070 Ti *today*; the only missing piece is the cook step that tessellates+displaces a facade type and registers its BLAS as an instanced type.
- **NVIDIA DMM / CLAS micromap (4070 Ti only, far cheaper memory).** `VK_NV_displacement_micromap` stores displacement at ~0.5 bits/microtriangle and cuts builder triangle count up to 1024× ([NVIDIA Micro-Mesh](https://developer.nvidia.com/rtx/ray-tracing/micro-mesh)). Spectra's `spectra-rtxmg` has the *intent* (`build_cas`) but it is a **stub returning `None`** (`subd.rs:182`) — DMM is net-new on the NVIDIA tier and should NOT be assumed.

**Instancing is the whole game for VRAM.** 100K instances of ~300 types means ~300 displaced BLASes shared across all instances (the TLAS holds 100K cheap transform references). VRAM scales with TYPES, not instances. This is what makes near/hero displacement affordable at all.

### 4.3 Technique 3 — Bump/height in the BRDF [PARTIAL, the cheap middle ground]

`normal_map.slang:43` `compute_bump_normal(h_center, h_right, h_up, …)` already derives a perturbed normal from height-map finite differences — and `spectra-displacement` mirrors it on CPU. This is *cheaper than POM* (3 texture taps, no march) and gives crisper lit relief than a baked normal map, but still **fakes** depth (no parallax, no silhouette). It is wired into neither the megakernel nor the packer today. Use only as the far-LOD fallback, or skip in favor of the existing normal map.

### 4.4 Hit-point displacement [adjunct, secondary rays only]

`hitpoint_displacement.slang` `apply_displacement(hit_pos, N, amount)` offsets the hit along the normal "WITHOUT geometry tessellation" — good for shadow/secondary-ray silhouette breakup, but it does not fix the primary-ray silhouette and is not a primary-depth solution. Optional polish.

---

## 5. Data Models

POM reuses the existing material fields — no new struct. The plan must make the megakernel **read** them (it currently ignores them):

```slang
// material_types.slang:69-71 — ALREADY EXISTS, currently unread by megakernel
int   displacement_tex;       // height map texture id (-1 = off, the gate)
float displacement_scale;     // relief depth in UV-height units (world ~1–4 cm for brick)
float displacement_midlevel;  // 0.5 = surface plane; controls in/out bias
```

Displaced-BLAS path adds a per-type registry entry (TYPE → displaced BLAS handle + subdiv level), held in the existing scene-upload BLAS cache (content-addressed); no new GPU buffer layout — it is ordinary triangle geometry.

---

## 6. API (net-new surface for the plan)

```slang
// New in a pom.slang module, called from megakernel.slang ~line 1974 (replacing/wrapping the flat normal lookup):
// Returns parallax-corrected UV; out_self_shadow in [0,1].
float2 parallax_occlusion_uv(
    float2 uv, float3 wo, float3 T, float3 B, float3 N,
    int height_tex, float scale, float midlevel,
    float2 ddx_uv, float2 ddy_uv,   // drive step count by footprint (LOD)
    out float out_self_shadow);
// Steps: 8 (far) … 32 (near), clamped by ddx/ddy footprint. Tangent-space linear march + 1 lerp refine.
```

```rust
// splat_backend.rs:2612 must change from the hard -1 to the cooked id:
// a[31] = pack_i32(m.displacement_tex);   // was: pack_i32(-1)
// a[32]/a[33] carry displacement_scale / displacement_midlevel (today a[33]=0.5 only)
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `parallax_occlusion_uv()` | megakernel surface shade, just before albedo/normal sample | `~/src/spectra/slang/megakernel.slang:~1974` | gated on `mat.displacement_tex >= 0`; reuse TBN at :1981-1990, `wo` (:661), `ddx_uv/ddy_uv` (:1554-1673) |
| megakernel material unpack of `displacement_*` | megakernel material decode | `~/src/spectra/slang/megakernel.slang` | currently MISSING — fields never read into the kernel struct |
| height-map id packing | cook → atom/material pack | `~/src/ochroma/crates/vox_render/src/splat_backend.rs:2612` | replace `pack_i32(-1)` |
| displaced facade BLAS build | scene cook for hero types → `SceneUploader{use_hw_rt:true}` | `~/src/spectra/rust/spectra-scene-upload/src/gpu_scene.rs` (KHR build) + `displacement.slang` passes | per-type, instanced; traced via `rt_query.slang:83` |
| LOD technique selector | per-instance setup before TLAS feed | engine instance governor (vox_app/vox_sim) | distance→{displaced BLAS / POM / normal-map} |

---

## 8. Cost vs Budget — the verdict

**Budget anchors:** 60 fps @ 4K = **16.6 ms/frame total**. Measured tier (`spectra-sdf-perf-verdict`): 780M→4070 Ti multiplier **5–7×**; full-frame primary-visibility headroom on the 4070 Ti is real but tight (honest tier 8–16 ms full PT @ 720p internal with reconstruction; 4K relies on DLSS-class upscaling). 4K = ~8.3M pixels native; with DLSS-Quality the path tracer shades ~3.7M internal pixels.

### POM — the cheap mass technique
- **GPU cost** = per-shaded-pixel raymarch: ~8–32 height-texture taps + a few ALU per primary hit, only on POM-tagged surfaces. Industry guidance: 10 samples is the common floor, 100 is "far too expensive" ([Valve POM](https://developer.valvesoftware.com/wiki/Parallax_mapping)). With footprint-driven LOD (8 far / 16–32 near) and the fact that buildings cover a fraction of the frame, the realistic added cost is **~0.3–1.0 ms at 4K-internal** on the 4070 Ti for a city view. Self-shadow march roughly doubles the tapped pixels' cost; make it near-LOD-only.
- **VRAM** = one extra single-channel height texture per facade material. At ~300 types and 1K² R8 height maps that's **~0.3 GB**, or ~0.08 GB at 512² — trivial.
- **No BLAS, no build cost.** Fits the budget comfortably. **VERDICT: fits 60 fps @ 4K.**
- Limit: flat silhouette + edge swim at extreme grazing angles; that's the hero/near gap displacement fills.

### Displaced micro-geometry — the expensive hero technique
- **BLAS VRAM (AMD/our box, KHR):** standard BVH is **~88–114 bytes/triangle** ([zeux.io](https://zeux.io/2025/03/31/measuring-acceleration-structures/)). A displaced facade type subdivided to ~200K triangles ≈ **~20 MB BLAS**. Instancing amortizes: **~300 types × ~20 MB ≈ ~6 GB** if *all* types are displaced at full detail — that **blows the VRAM budget** on a 12 GB 4070 Ti once textures/film/atoms are added. So displacement MUST be hero/near-only: ~20–40 displaced types × ~20 MB = **~0.4–0.8 GB**, affordable. Build cost is a one-time/streamed cook, not per-frame.
- **DMM (NVIDIA 4070 Ti):** ~0.5 bits/microtri + up to 1024× fewer builder triangles ([NVIDIA Micro-Mesh](https://developer.nvidia.com/rtx/ray-tracing/micro-mesh)) → the same hero set drops to **tens of MB** and near-instant builds. But it's a **stub in Spectra** (`subd.rs:182`) — net-new, do not budget on it yet.
- **Per-frame trace cost:** more triangles in the BVH = deeper traversal, but instanced + LOD-capped + KHR-HW-accelerated it is bounded; the dominant risk is VRAM, not ms.
- **VERDICT:** fits 60 fps @ 4K **only** as a near/hero LOD over ~20–40 types via KHR BLAS today; full-scene displacement does not fit VRAM. DMM would lift the cap but is unbuilt.

### Recommendation
1. **POM for the mass** — every facade material gets a height map; POM runs at all but the closest LOD. Cheap (~0.3–1 ms, ~0.1–0.3 GB), slots into the existing megakernel TBN/UV-derivative path with no new infrastructure. This is the high-leverage win and directly serves `facade-photorealism-roadmap` Wave 3.
2. **Displaced KHR BLAS for near/hero** — the camera-closest ~20–40 facade types (and any silhouette-critical reveal: deep window jambs, projecting cornices) get real displaced geometry via the already-verified KHR HW-RT build. Instanced, VRAM-capped (~0.4–0.8 GB), LOD-gated to near distance.
3. **LOD selector is the budget lever** — distance bands: displaced BLAS (near) → POM (mid) → flat normal map (far). This keeps both the ms and the VRAM inside the 60 fps @ 4K / 12 GB envelope.
4. **DMM is a 4070-Ti upside, not a dependency** — design the displaced path so a future DMM backend swaps in under the same per-type registry, but do not budget on it (it's a stub).

---

## 9. Open Questions

- [ ] Height-map source: author per material family, or derive from the same Forge geometry that defines brick/panel grammar? (Deriving keeps the directive-factory single-source principle.)
- [ ] POM self-shadow: near-LOD only, or skip entirely for v1? (Cost vs the recessed-mortar payoff.)
- [ ] Displaced-type budget: hard cap at N hero types, or dynamic by VRAM watermark fed from the instance governor?
- [ ] Does the per-hit TBN (triangle-edge derived) stay stable enough across a façade for POM, or do we need per-vertex tangents in the cooked mesh?

---

## 10. Out of Scope

- DMM/CLAS implementation on the NVIDIA tier (it's a stub; a separate plan owns un-stubbing `spectra-rtxmg`).
- Tessellation-shader (raster) displacement — Spectra is a path tracer; geometry must be in the BVH.
- SDF-based relief — SDF is substrate-only per `hybrid-atom-sdf-lod-direction`; facades render as mesh.
- Authoring/curating the height-map texture set (content task, not engine).

---

## 11. Related Plans / Designs

- Depends on: KHR HW-RT BLAS/TLAS (built + verified, `khr_rt_gpu.rs`); the mesh render direction (`hybrid-atom-sdf-lod-direction`).
- Required before: `facade-photorealism-roadmap` Wave 3 (facade depth) implementation plan.
- Related: `spectra-sdf-perf-verdict` (the 4070 Ti tier numbers used here); `spectra-texture-bridge` (texture packing the height map rides on).
