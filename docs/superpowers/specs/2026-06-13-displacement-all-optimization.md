# Design: Displacement-for-All — Can a Spectral Path Tracer Approach POM's Cost? (2026-06-13)

**Status:** Draft
**Scope:** Decide whether *every* building type (~300) can carry TRUE silhouette displacement (real depth at the surface, true silhouette + self-shadow) without baking ~6 GB of per-type micro-geometry into BLAS — by exploiting path-tracer-specific tricks (trace-time/procedural intersection, ray-cone adaptive tessellation, on-demand subdivision, height-map sharing, DMM compression) to bring the cost near the POM baseline (~0.3–1.0 ms @ 4K + ~0.1–0.3 GB). Read-only investigation; no code changed.
**Related:** `2026-06-13-texture-depth-cost.md` (POM vs baked-displacement cost; this doc is its follow-on), `hybrid-atom-sdf-lod-direction` (MESH renders the visible world; SDF = substrate), `spectra-sdf-perf-verdict` (780M→4070 Ti tier), `facade-photorealism-roadmap` (Wave 3 = facade depth).

---

## 1. Problem Statement

- The depth-cost doc found the two endpoints: **POM** ≈ 0.3–1.0 ms @ 4K-internal + ~0.1–0.3 GB (one R8 height map / type), but **flat silhouette**; **baked displaced micro-geometry** gives a true silhouette but costs **~88–114 B/triangle** ([zeux.io](https://zeux.io/2025/03/31/measuring-acceleration-structures/)) → ~20 MB per ~200K-tri facade type → **~6 GB for all ~300 types**, which blows the 12 GB 4070 Ti budget once textures/film/atoms are added (`2026-06-13-texture-depth-cost.md:132`).
- The open question: can a path tracer do displacement *during the ray hit* — march a height field on the base triangle to produce a real displaced intersection (true silhouette + self-shadow) at POM memory cost (height map only, no baked geometry)? This is "displacement as an intersection shader" / DMM-without-baking.
- **The verified hardware traversal seam is triangle-only.** `RayQueryTracer::trace_closest` issues `RayQuery<RAY_FLAG_FORCE_OPAQUE>` and reads `COMMITTED_TRIANGLE_HIT` / `CommittedTriangleBarycentrics` (`~/src/spectra/slang/rt_tracer.slang:137-150`). There is **no procedural / AABB-primitive candidate path** (`CANDIDATE_PROCEDURAL_PRIMITIVE`) and **no intersection shader** anywhere on the KHR path. A height-field march cannot run inside the hardware traversal as written.
- The **software** BVH, by contrast, already has a custom-intersection seam: a `PRIM_PATCH` leaf dispatches to `intersect_patch()` Newton–Raphson instead of Möller-Trumbore (`~/src/spectra/slang/bvh_traverse.slang:64-86,365-389`). So custom intersection is *architecturally proven in Spectra* — but only on the slow software traversal, not the GPU-verified KHR path.
- All existing displacement machinery **bakes geometry pre-BVH**: `displacement.slang`'s 5-pass tessellate→displace→recompute-normals (`displacement.slang:91-420`), `patch_eval.slang:573-602` `displace_at_limit`, and the `spectra-displacement` crate's `DisplacementMap::apply` (`~/src/spectra/rust/spectra-displacement/src/lib.rs:112-128`). None of these intersect a displaced surface at trace time.
- The DMM/CLAS compression path that would shrink the 6 GB is a **stub**: `SubdivisionPipeline::build_cas` returns `None` ("RTXMG hardware detection would go here", `~/src/spectra/rust/spectra-rtxmg/src/subd.rs:106-108,157-182`) and `trace` returns `INFINITY` (`subd.rs:188-196`).

---

## 2. Done When

This is a **design/decision doc** (read-only investigation). It is "done" when:

Running `ls ~/src/ochroma/docs/superpowers/specs/2026-06-13-displacement-all-optimization.md` shows the file, and a human reading §3–§8 can answer, without reading code: (a) for each candidate optimization, whether it is Spectra-feasible today / net-new / blocked, with exact `file:line` citations; (b) a per-approach GPU-ms@4K + VRAM estimate for ALL ~300 types @ 100K instances against 60 fps @ 4K / 12 GB on a 4070 Ti, set beside the POM baseline; (c) the verdict on whether displacement-for-all reaches ~POM numbers, and if not, the closest achievable and where the wall is. No code is changed by this doc; a follow-on *plan* owns any render-gate command.

---

## 3. Capabilities — the candidate approaches and the real test each would need

| Approach | What it would do | Real behavior test (for the follow-on plan) | Stub test (forbidden) |
|---|---|---|---|
| **A. Trace-time heightfield march on the HW path** | Custom intersection on the building BLAS: from base triangle + height map, march the relief during the ray hit → real displaced `t`, silhouette, self-shadow. VRAM = height maps only. | Grazing primary ray on a brick wall returns a hit `t` differing from the flat-wall `t` by `>= displacement_scale*0.5` (true silhouette), measured on the **KHR HW path** on RADV. | `assert!(hit.is_some())` with the flat triangle `t` — passes with zero displacement |
| **A′. Trace-time march on the SOFTWARE BVH** | Add a `PRIM_HEIGHTFIELD` leaf beside `PRIM_PATCH`; march in `intersect_*`. Proves the silhouette but on the slow traversal. | Same `t`-difference assertion, but via the software `trace_bvh` kernel; AND a perf gate showing the per-ray march cost. | leaf added but `u_prim_type` never set → never reached |
| **B. Ray-cone / footprint-adaptive baked subdiv** | Bake displaced geometry but pick subdiv level from the ray footprint (`ddx_uv/ddy_uv`), so distant types carry far fewer triangles → less BLAS VRAM. | Cook one type at two camera bands; assert near-band `blas.tri_count` >> far-band, and total VRAM for 300 types < a fixed-level baseline by a measured factor. | `tri_count` fixed regardless of distance |
| **C. POM-in-megakernel (the baseline being chased)** | Height-field UV raymarch in tangent space; parallax interior, **flat silhouette**. | Brick at 70° dollying laterally: brick/mortar boundary shifts `>2 px` between two PNGs; mortar pixels `>20%` darker. | `assert!(pom_enabled)` |
| **D. DMM/CLAS micromap** | Compress baked displacement to ~0.5 bits/microtri; full set drops to tens of MB. | `build_cas` returns a real handle AND a ray through it returns a finite displaced `t`. | `build_cas` returns `Some` but `trace` returns `INFINITY` — the current stub (`subd.rs:182,195`) |

---

## 4. Architecture — feasibility of each approach, grounded in the code

### 4.1 Approach A — Trace-time heightfield march on the hardware path [BLOCKED on the verified path]

This is the holy grail in the prompt: keep one base mesh + a height map, and let the path tracer compute the *displaced* intersection during traversal — exactly what an OptiX/DXR **intersection shader** over **procedural AABB primitives** does (each base triangle's prism becomes an AABB; the intersection shader ray-marches the height field inside it, like the inverse-bilinear / heightfield-prism march in NVIDIA's micro-mesh and Thonat et al. "Tessellation-Free Displacement Mapping"). VRAM = height maps only ≈ POM's ~0.1–0.3 GB; the silhouette and self-shadow are **real**.

**Why it is blocked on the path we actually ship:** Spectra's GPU-verified traversal is the KHR inline `RayQuery`, and it is wired triangle-only. `RayQueryTracer::trace_closest` declares `RayQuery<RAY_FLAG_FORCE_OPAQUE>`, calls `q.Proceed()` once with no candidate loop, and reads only `COMMITTED_TRIANGLE_HIT` (`rt_tracer.slang:135-153`). A procedural primitive surfaces as `CANDIDATE_PROCEDURAL_PRIMITIVE` and **requires** the host to (1) build the BLAS with AABB geometry (`VkAccelerationStructureGeometryAabbsDataKHR`) and (2) run a candidate loop that computes the hit and calls `q.CommitProceduralPrimitiveHit(tHit)`. Neither exists:
- The host BLAS builder only consumes triangle soup; the only `aabb_min/aabb_max` in scene-upload are **bounding boxes for node culling**, not procedural geometry (`~/src/spectra/rust/spectra-scene-upload/src/gpu_scene.rs:76-120`, `content_cache.rs:17-18,90-93`, `gpu_stage.rs:174-206`) — grep for AABB-geometry BLAS / intersection-shader plumbing returns nothing.
- The kernel has no procedural candidate handling (`RAY_FLAG_FORCE_OPAQUE` explicitly forces every candidate committed without a shader round-trip — see the comment at `rt_tracer.slang:136`).

So Approach A is **net-new infrastructure on the hardware path**: an AABB-BLAS build mode + a `RayQuery` candidate loop running the march + a `CommitProceduralPrimitiveHit`. It is feasible in principle on RADV (KHR ray-query supports procedural primitives) but it is **not** a small slot-in, and it changes the BLAS build that `khr_rt_gpu.rs` verified for triangles. Inline-query procedural intersection is also typically **slower per ray** than hardware triangle traversal (the march runs in-shader, defeating part of the RT-core advantage), and it disables the watertight HW triangle intersector for those instances.

**Important nuance — self-shadow is multi-ray, not in-march.** Spectra is a wavefront tracer: shadow/secondary rays are *separate dispatches* (`trace_shadow_hw`, `rt_query.slang:153-175`), not a sub-loop inside the primary hit. A procedural primitive gives a correct displaced *primary* `t`, but a shadow ray toward the light must **also** hit the procedural primitive to self-shadow — i.e. the AABB-BLAS + candidate march must be added to the shadow path too (`run_shadow`, `rt_query.slang:135-151`), doubling the net-new surface. POM, by contrast, fakes self-shadow with a cheap second tangent-space march inside the *same* hit shader (no extra ray).

### 4.2 Approach A′ — Trace-time march on the SOFTWARE BVH [FEASIBLE today, but it is the slow path]

The software BVH already proves custom intersection: `bvh_traverse.slang` leaf dispatch branches on `u_prim_type`, and `PRIM_PATCH` leaves call `intersect_patch()` (Newton–Raphson ray-patch) instead of the triangle test (`bvh_traverse.slang:64-86,363-389`). A `PRIM_HEIGHTFIELD` leaf could be added the same way: store `{tri_id, height_tex, scale, midlevel}`, expand the triangle's prism to the leaf AABB, and march the height field in a new `intersect_heightfield()` (the march math already exists on CPU in `spectra-displacement` and as the bilinear sampler in `displacement.slang:59-85`). This would give a **real displaced silhouette at height-map-only VRAM** and is **buildable today** with no host BLAS change (the software BVH is plain buffers).

The catch: the software `trace_bvh` is the *fallback* traversal, materially slower than KHR HW-RT (the whole point of the verified KHR path). Running the city's primary visibility through the software BVH to get displacement-for-all trades the 6 GB VRAM problem for a large per-ray-ms problem. Useful as a **correctness oracle / hero-shot path**, not the 100K-instance city runtime.

### 4.3 Approach B — Ray-cone / footprint-adaptive baked subdivision [FEASIBLE, the realistic VRAM lever]

The 6 GB figure assumes every type is baked at full (~200K-tri) detail. A path tracer knows each ray's footprint: `ddx_uv/ddy_uv` are already computed for mip/EWA (`megakernel.slang:1554-1607`), and `displacement.slang`'s pass-1 already chooses subdiv level from world-space edge length vs a threshold (`tessellate_count_pass`, `displacement.slang:91-123`); `DisplacementEvaluator::compute_level` mirrors it on CPU (`~/src/spectra/rust/spectra-displacement/src/lib.rs:142-148`). So a **streaming, distance-banded bake** is in reach: only the camera-near types carry full triangle counts; mid/far types are baked at level 1–2 (or fall back to POM). This does **not** make displacement-for-all free, but it turns "6 GB for all-at-full" into "VRAM proportional to how many types are near *right now*", which the instance governor can cap to a VRAM watermark. It is the same conclusion as the depth-cost doc's LOD selector, sharpened: the budget lever is *how many types are displaced at full detail at once*, not *how many types exist*.

A true **on-demand / lazy subdivision cache** (build the displaced BLAS for a type only when the camera first needs it, evict by LRU) is the strongest form of B and is buildable on the existing content-addressed BLAS cache (`content_cache.rs`), but it is net-new cache-management code, not a slot-in.

### 4.4 Approach C — POM in the megakernel [the baseline; cheap, slots in]

Unchanged from `2026-06-13-texture-depth-cost.md:39-52`: a ~30-line tangent-space march in the megakernel, reusing the TBN at `megakernel.slang:1978-1992`, `wo` (in scope at the hit), and `ddx_uv/ddy_uv` (`:1554-1607`) to LOD the step count; gated on `mat.displacement_tex >= 0`. Flat silhouette, parallax interior, fake self-shadow via a second short march. ~0.3–1.0 ms @ 4K-internal, ~0.1–0.3 GB for 300 R8 maps. This is the number every other approach is measured against.

### 4.5 Approach D — DMM / CLAS micromap compression [STUB; the only thing that makes baked-all fit VRAM]

`VK_NV_displacement_micromap` stores displacement at ~0.5 bits/microtri and cuts builder triangle count up to 1024× ([NVIDIA Micro-Mesh](https://developer.nvidia.com/rtx/ray-tracing/micro-mesh)). If real, the baked-all set drops from ~6 GB to **tens of MB** and the silhouette is hardware-correct — this is the *one* approach that gets a true-silhouette all-types set into VRAM cheaply. But in Spectra it is a **stub on every tier**: `build_cas` returns `None` (`subd.rs:157-182`), `trace` returns `INFINITY` (`subd.rs:188-196`), `rtxmg_available` is hard-`false` (`subd.rs:106-110`). It is also **NVIDIA-only** (4070 Ti), not the cross-vendor RADV path the engine verifies on. Cannot be budgeted on today.

---

## 5. Data Models

POM and the trace-time march reuse the existing material fields (already defined, currently unread by the megakernel — `material_types.slang:69-71`):

```slang
int   displacement_tex;       // height-map texture id (-1 = off, the gate)
float displacement_scale;     // relief depth in UV-height units (~1–4 cm brick)
float displacement_midlevel;  // 0.5 = surface plane
```

Approach A′ adds a software-BVH leaf type beside `PRIM_PATCH` (`bvh_traverse.slang:64-86`):

```slang
static const int PRIM_HEIGHTFIELD = 3;  // new, next to PRIM_TRIANGLE=0, PRIM_PATCH=2
struct HeightfieldLeaf {
    int base_tri_id;   // the flat triangle whose prism we march
    int height_tex;    // single-channel height map id
    float scale;       // displacement_scale
    float midlevel;    // displacement_midlevel
};
```

Approach A (HW path) adds an AABB-geometry BLAS build mode in scene-upload (net-new; today only triangle BLAS exists, `gpu_scene.rs:76-120`). Approach B/D reuse the existing content-addressed per-type BLAS cache (`content_cache.rs`) — VRAM scales with TYPES displaced-at-full, not instances.

---

## 6. API (net-new surface a follow-on plan would own)

```slang
// Approach A′ (software BVH), new in patch_eval.slang or a heightfield.slang,
// called from the bvh_traverse leaf loop beside intersect_patch():
struct HeightfieldHitResult { bool hit; float t; float u; float v; };
HeightfieldHitResult intersect_heightfield(
    int base_tri_id, int height_tex, float scale, float midlevel,
    float3 origin, float3 dir, float t_min, float t_max);
// Marches the prism over the base triangle; reuses the bilinear sampler from
// displacement.slang:59. Returns the displaced hit (true silhouette).

// Approach C (POM), per 2026-06-13-texture-depth-cost.md:90-99 — unchanged.
```

```rust
// Approach A (HW path) — net-new in spectra-scene-upload:
// build a BLAS from procedural AABBs (VkAccelerationStructureGeometryAabbsDataKHR)
// instead of triangles; bind it so RayQueryTracer can run a candidate loop.
// This is a NEW build mode, not a parameter on the existing triangle build.

// splat_backend.rs:2612 must stop writing the hard -1 for displacement_tex
// (shared with the depth-cost doc) so any of A/A′/C can read a height map.
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| POM `parallax_occlusion_uv()` | megakernel surface shade, before albedo/normal sample | `~/src/spectra/slang/megakernel.slang:~1974` | gated on `mat.displacement_tex>=0`; reuse TBN (:1978-1992), `wo`, `ddx_uv/ddy_uv` (:1554-1607) |
| `intersect_heightfield()` (A′) | software BVH leaf loop, beside `intersect_patch` | `~/src/spectra/slang/bvh_traverse.slang:365` | new `PRIM_HEIGHTFIELD` branch on `u_prim_type`; software path only |
| AABB-BLAS build + procedural candidate loop (A) | scene cook → `SceneUploader{use_hw_rt:true}`; `RayQueryTracer::trace_closest`/`trace_shadow` | `~/src/spectra/rust/spectra-scene-upload/src/gpu_scene.rs` + `~/src/spectra/slang/rt_tracer.slang:135-175` | NET-NEW; changes the verified triangle-only KHR path; must also be added to the shadow dispatch |
| Footprint-adaptive bake / LRU BLAS cache (B) | scene cook per type, fed by instance governor VRAM watermark | `content_cache.rs` + `displacement.slang` passes | reuses `compute_level` (`lib.rs:142`) and `tessellate_count_pass` (`displacement.slang:91`) |
| height-map id packing | cook → atom/material pack | `~/src/ochroma/crates/vox_render/src/splat_backend.rs:2612` | replace `pack_i32(-1)`; shared prerequisite for A/A′/C |

---

## 8. Cost vs Budget — the honest model

**Anchors:** 60 fps @ 4K = 16.6 ms/frame; 4K = ~8.3M px native, ~3.7M internal with DLSS-Quality; 780M→4070 Ti = 5–7× (`spectra-sdf-perf-verdict`). VRAM budget 12 GB. ~300 types, 100K instances; **instancing means VRAM scales with TYPES (and subdiv level), not instances** — the 100K transforms live cheaply in the TLAS (`2026-06-13-texture-depth-cost.md:61,132`). BLAS ≈ 88–114 B/tri ([zeux.io]); full facade ≈ 200K tri ≈ ~20 MB/type.

| Approach | Silhouette | GPU-ms @ 4K (city view) | VRAM, ALL 300 types | Fits 60 fps@4K / 12 GB? |
|---|---|---|---|---|
| **C. POM** (baseline) | Fake (flat) | ~0.3–1.0 ms | ~0.1–0.3 GB (R8 maps) | **Yes** |
| **A. Trace-time march, HW** | **Real** | POM-march cost **+ procedural-traversal overhead** (in-shader march defeats part of RT-core triangle accel; +shadow-ray march). Est. **~2–6 ms** city-wide if all facades are procedural; un-measured. | **~0.1–0.3 GB** (height maps only) — *VRAM solved* | **VRAM yes; ms uncertain & likely tight** — and it is net-new infra on the verified path |
| **A′. Trace-time march, software BVH** | **Real** | Software traversal is the fallback path; city-wide primary visibility through it is **far above budget** (the reason KHR HW-RT exists). | ~0.1–0.3 GB | **No** for 100K-instance city; yes as hero/oracle only |
| **B. Adaptive baked subdiv** | **Real** | HW triangle traversal (fast); LOD-capped. ms ~ flat-mesh + bounded BVH depth. | **Proportional to types-near-at-full**: ~20–40 near types × ~20 MB = ~0.4–0.8 GB; far types at level≤2 or POM. NOT 6 GB. | **Yes, as near/hero set**; all-at-full still ~6 GB → no |
| **D. DMM/CLAS** | **Real** | HW micromap traversal (fast) | **Tens of MB** for all 300 — *the only true-silhouette-all that fits* | Would be **yes** — but it's a **stub** (`subd.rs:182,195`), NVIDIA-only |

### The verdict on "displacement-for-all at ~POM cost"

- **VRAM can be brought to POM levels** by Approach A / A′ (height maps only, ~0.1–0.3 GB) — that part is achievable, and it is the path-tracer-specific win the prompt asks about (intersect the displaced surface; don't bake it).
- **But the GPU-ms does not come for free on the path we ship.** The verified, fast traversal is **triangle-only HW RayQuery** (`rt_tracer.slang:137-150`); a trace-time march needs either (A) net-new procedural-AABB BLAS + an in-shader candidate march on both the closest-hit and shadow dispatches — which adds infra *and* per-ray cost that erodes the RT-core advantage — or (A′) the software BVH, which is the slow fallback and won't carry a 100K-instance city at 4K/60.
- **DMM (D) is the only approach that makes a true-silhouette all-types set both cheap in VRAM and fast in ms — and it is a stub** (`subd.rs:106-108,157-196`) and NVIDIA-only. It cannot be budgeted on today.

**So: displacement-for-all does NOT reach ~POM *total* cost on the engine as it stands.** POM-VRAM is reachable; POM-*ms* with a true silhouette is not, because the fast path can't march and the path that can march is slow.

**Closest achievable today (the wall and where it is):**
1. **POM for the mass (C)** — true POM cost, fake silhouette. The 300-type baseline.
2. **Adaptive baked KHR BLAS for the near/hero set (B)** — true silhouette on the fast HW path, VRAM capped to ~0.4–0.8 GB by displacing only the ~20–40 types the camera is near *now* (footprint-LOD + LRU cache), with POM behind them. This is the realistic "as close to displacement-for-all as the budget allows" — silhouette-correct where the camera can see the silhouette, POM everywhere else.
3. **The wall is the hardware traversal seam.** True silhouette for *all* types at POM ms needs either a procedural-AABB intersection path on the KHR tracer (net-new, and slower per ray) or DMM (stub, NVIDIA-only). Until one of those lands, "displacement-for-all" = "POM-for-all + real displacement on a VRAM-capped near set."

**Recommended sequencing:** ship C (POM) + B (adaptive near-set BLAS, the depth-cost doc's LOD selector) now; prototype A′ on the software BVH as the correctness oracle (it proves the march math and the visual target cheaply); treat A (HW procedural) and D (DMM) as the two competing routes to *true* displacement-for-all, to be measured — A is cross-vendor but un-built and likely ms-heavy; D is cheap-and-fast but a stub and NVIDIA-only.

---

## 9. Open Questions

- [ ] Approach A: measure the per-ray cost of an inline-`RayQuery` procedural march on RADV vs HW triangle traversal — is the ms within ~2–3× of POM, or does in-shader marching erase the RT-core win entirely?
- [ ] Approach B: hard cap on near-displaced types, or dynamic by VRAM watermark from the instance governor? What near-distance band flips a type from baked-BLAS to POM?
- [ ] Is a true LRU BLAS cache (lazy per-type bake on first near-approach) worth the cache-management code, or is a fixed near-set sufficient for the city camera?
- [ ] Self-shadow: for A/A′ the shadow ray must also hit the procedural primitive — is the doubled traversal surface worth it over POM's cheap second-march fake at mid distance?
- [ ] DMM: is un-stubbing `spectra-rtxmg` (NVIDIA-only) a better long-term bet than building the cross-vendor procedural-AABB path (A)? Decide before committing either.

---

## 10. Out of Scope

- Implementing DMM/CLAS on the NVIDIA tier (it's a stub; a separate plan owns un-stubbing `spectra-rtxmg`).
- Authoring/curating the per-type height-map texture set (content task).
- Raster/tessellation-shader displacement — Spectra is a path tracer; geometry/relief must live in the BVH or the hit shader.
- SDF-based relief — SDF is substrate-only per `hybrid-atom-sdf-lod-direction`; facades render as mesh.
- OptiX procedural intersection on CUDA — the `OptixTracer` is itself a stub (`rt_tracer.slang:OPTIX_TRACER_IMPLEMENTED=false`).

---

## 11. Related Plans / Designs

- Depends on / extends: `2026-06-13-texture-depth-cost.md` (POM baseline + baked-displacement cost; this doc answers its implied "can we do better than near/hero-only").
- Depends on: KHR HW-RT BLAS/TLAS (triangle-only, verified `khr_rt_gpu.rs`); the mesh render direction (`hybrid-atom-sdf-lod-direction`).
- Required before: `facade-photorealism-roadmap` Wave 3 (facade depth) implementation plan; any `spectra-rtxmg` un-stub plan (Approach D).
- Related: `spectra-sdf-perf-verdict` (4070 Ti tier numbers); `spectra-texture-bridge` (height-map packing).
