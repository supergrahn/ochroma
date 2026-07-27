# QEM Mesh Simplifier — buildings must survive their own LODs

**Goal:** Replace the vertex-grid-clustering decimator with a quadric-error-metric
edge-collapse simplifier that preserves silhouette, planarity and sharp edges while
keeping UVs and per-triangle material ids exact.

**Done When:** `cargo test -p vox_render qem` prints
`test mesh_simplify::tests::qem_beats_grid_on_a_box ... ok`, and that test asserts
the QEM one-sided Hausdorff error on a unit box decimated to 25% is **at least 4x
lower** than the grid clusterer's on the same input — a number printed by
`cargo test -p vox_render qem -- --nocapture` as
`qem_hausdorff=<a> grid_hausdorff=<b> ratio=<b/a>`. Then a rendered witness at
`OCHROMA_FORCE_MESH_LOD=2` shows buildings with intact flat facades and straight
rooflines instead of the current broken masses.

**Architecture:** Garland–Heckbert quadric error metrics over half-edge-free
adjacency. Each vertex accumulates the sum of its incident face plane quadrics,
plus perpendicular "constraint" quadrics along OPEN BOUNDARY and MATERIAL-SEAM
edges so those lines cannot drift. Collapses are taken cheapest-first from a
deterministic binary heap, each validated against normal flipping before it is
accepted. Triangle material ids and source-triangle provenance ride through
verbatim, so the existing cook contract is unchanged.

**Design Document:** inline (this plan) — the algorithm is standard QEM; the
project-specific parts are the seam constraints and the determinism contract.

**Tech Stack:** Rust 2021, `vox_render` crate, no new dependencies.

**Build:** `cargo test -p vox_render`

---

## IMPORTANT NOTES

- Public API is FROZEN — `simplify_mesh(input: &MeshInput<'_>, target_ratio: f32)
  -> MeshOutput` keeps its exact signature. `MeshInput` / `MeshOutput` field sets
  do not change. The cook (`offline_cook::cook_prototype_lods`) and every caller
  must compile untouched.
- `MeshOutput.source_triangle_indices` MUST stay parallel to `indices` — offline
  cooks recover semantic attributes through it.
- **DETERMINISM IS A PRODUCT MOAT.** No `HashMap` iteration may reach the output.
  Ties in the collapse heap break on `(cost_bits, v0, v1)` so two runs are
  byte-identical. `f32` costs compare via `to_bits()` on a canonicalised
  non-negative value — never a raw float `PartialOrd`.
- Never weld across a MATERIAL SEAM: an edge whose two incident triangles carry
  different `material_id` is constrained, not free.
- `todo!()` / `unimplemented!()` / empty bodies are forbidden.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `crates/vox_render/src/mesh_simplify.rs` | QEM collapse replaces grid weld; API unchanged |
| Test   | `crates/vox_render/src/mesh_simplify.rs` (`mod tests`) | silhouette/planarity/determinism/seam assertions |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Planar faces stay planar | Decimate a subdivided cube to 25%; assert max deviation of every output vertex from its source face plane `< 1e-4` | `assert!(!out.positions.is_empty())` |
| Beats grid clustering | Hausdorff(QEM) * 4.0 <= Hausdorff(grid) on the same box + ratio | `assert!(out.indices.len() < input.len())` |
| Sharp edges survive | Cube corner vertices still present within 1e-4 after 25% decimation | count-only assertion |
| Material seams intact | Two-material mesh: assert no output triangle changed material, and seam vertices are preserved | `assert_eq!(mats.len(), tris.len())` |
| Deterministic | Two runs on the same input produce byte-identical `positions`/`indices` | run once, assert non-empty |
| UVs preserved | UV range of output within the input's UV bounds; parallel to positions | `assert!(!uvs.is_empty())` |

---

## Task 1: QEM core — quadrics, constrained edges, validated collapse

**Files:** Modify `crates/vox_render/src/mesh_simplify.rs`

- [ ] `Quadric` (10 f64 coefficients, symmetric 4x4) with `from_plane`, `add`, `error_at`.
- [ ] Per-vertex quadric = sum of incident face plane quadrics, area-weighted.
- [ ] Constraint quadrics on OPEN BOUNDARY and MATERIAL-SEAM edges: build the plane
      through the edge perpendicular to the incident face, weight it heavily
      (`SEAM_WEIGHT`), add to both endpoints. This is what keeps a roofline straight
      and a facade's border from drifting.
- [ ] Collapse target position: solve the 3x3 quadric system; if singular
      (`|det| < 1e-12`), fall back to the cheaper of the two endpoints and the
      midpoint. Never invent an unconstrained optimum.
- [ ] Validity: reject a collapse that flips any incident triangle normal by more
      than 90 degrees, or that produces a degenerate (zero-area) triangle.
- [ ] Deterministic heap keyed on `(cost.to_bits(), v0, v1)`.
- [ ] Iterate until triangle count reaches `target_ratio`, or no legal collapse
      remains (the exact-preservation floor the current contract already documents).

**Verification:** `cargo test -p vox_render mesh_simplify` — all capability tests above pass.

---

## Task 2: Wire + witness

- [ ] Keep `simplify_mesh` signature; grid path deleted, not left dead.
- [ ] Re-cook prototypes, render `OCHROMA_FORCE_MESH_LOD=2`, confirm facades and
      rooflines are intact where they were previously broken masses.
- [ ] Record before/after Hausdorff and triangle counts in the commit.

---

## Why this also buys FPS

Grid clustering spends its triangle budget badly: it keeps triangles where the grid
happens to land, not where curvature is. QEM spends them where error is highest, so
the SAME visual quality needs materially fewer triangles — or the same triangle
budget looks far better. With 1264 of 1376 groups at LOD1+ at a 180 m camera, that
budget is most of the visible city.
