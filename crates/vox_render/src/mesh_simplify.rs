//! Deterministic, replay-safe mesh decimator that PRESERVES UVs and per-triangle
//! material ids — the engine-side companion to spectra's positions-only
//! `decimate_mesh` (`spectra-optix/src/clas.rs`).
//!
//! # Why this exists
//!
//! Vegetation / asset meshes carried by [`HybridMesh`](crate::hybrid_compose)
//! need LOD reduction *without* losing the two attributes the live render path
//! depends on:
//!
//! - **UVs** — dropped UVs collapse a textured leaf/bark mesh to a single texel
//!   (the instanced-UV gap the engine has been fighting). UVs are a vertex
//!   attribute and are AVERAGED per weld cluster.
//! - **Per-triangle material ids** — a multi-material mesh (bark + leaf) must keep
//!   each triangle on its own material. We NEVER weld across a material seam:
//!   every source vertex/material pair is clustered independently, so a source
//!   vertex shared by two materials is explicitly duplicated in the output.
//!   Degenerate triangles (two or three corners welded to the same output vertex)
//!   are dropped.
//!
//! # Algorithm — vertex-GRID CLUSTERING
//!
//! 1. Compute an AABB for the vertices referenced by each material. A small
//!    authored material island therefore gets its own spatial budget instead of
//!    disappearing inside the full mesh's much larger bounds.
//! 2. Pick a grid resolution per material from `target_ratio` (fewer cells →
//!    fewer output vertices → fewer triangles). The resolution is derived
//!    purely from that material's referenced vertex count, bounds, and the ratio,
//!    so it is a deterministic function of the input. Every nontrivial material
//!    receives at least a four-cell budget so a small textured quad can survive.
//! 3. Snap every referenced source vertex/material pair to its integer grid cell.
//!    The cluster key is `(cell_x, cell_y, cell_z, triangle_material)`. All pairs
//!    sharing a key weld to one OUTPUT vertex whose position and UV are the
//!    arithmetic mean of the distinct source vertices (a stable, fixed-order
//!    reduction — we accumulate in a Vec indexed by first-seen cluster order, so
//!    there is no HashMap iteration in the output ordering).
//! 4. Rebuild the triangle list against the welded vertices; drop any triangle
//!    whose three corners no longer reference three distinct output vertices
//!    (degenerate after the collapse). Each surviving triangle keeps its ORIGINAL
//!    `material_id`.
//! 5. If clustering would erase an authored material or collapse a material's
//!    varied UV stream to one texel, retain that material's exact source triangles
//!    in the cooked output. Detail loss is never accepted as a valid LOD.
//!
//! # Determinism
//!
//! Replay-exact: there is NO RNG, NO float-key HashMap iteration in any path that
//! affects output ordering, and all folds are over id-ordered Vecs. The cluster
//! lookup uses a `HashMap` keyed by integer cell coordinates ONLY to find the
//! cluster index; the *output* vertex order is the first-seen order recorded in a
//! Vec, so two runs on identical input produce byte-identical output. Position/UV
//! averaging is a fixed-order sum (input vertex order) divided by the count.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Input to [`simplify_mesh`]: parallel position / UV vertex streams, a triangle
/// index list, and a per-triangle material id.
#[derive(Debug, Clone)]
pub struct MeshInput<'a> {
    /// World/object-space vertex positions.
    pub positions: &'a [[f32; 3]],
    /// Per-vertex UVs, parallel to `positions`. Pass an empty slice for an
    /// untextured mesh; the output then also has empty UVs.
    pub uvs: &'a [[f32; 2]],
    /// Triangle list (each entry is three indices into `positions`/`uvs`).
    pub indices: &'a [[u32; 3]],
    /// Per-triangle material id (parallel to `indices`). Pass an empty slice to
    /// treat the whole mesh as material 0; the output material_ids is then also
    /// all-zero of the surviving triangle count.
    pub material_ids: &'a [u32],
}

/// Output of [`simplify_mesh`]: the same four parallel streams as
/// [`MeshInput`], on the decimated mesh.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshOutput {
    /// Welded vertex positions (cluster means).
    pub positions: Vec<[f32; 3]>,
    /// Welded vertex UVs (cluster means). Empty iff the input UVs were empty.
    pub uvs: Vec<[f32; 2]>,
    /// Rebuilt triangle list against the welded vertices (degenerates dropped).
    pub indices: Vec<[u32; 3]>,
    /// Per-triangle material id of each surviving triangle (parallel to
    /// `indices`), carried verbatim from the input.
    pub material_ids: Vec<u32>,
    /// Source triangle index for every surviving triangle, parallel to
    /// `indices`. Offline cooks use this stable provenance to recover semantic
    /// attributes that deliberately are not part of the simplifier's weld key.
    pub source_triangle_indices: Vec<u32>,
}

/// Deterministically decimate `input` toward `target_ratio` of its triangles
/// (clamped to `(0, 1]`) via vertex-grid clustering, **preserving UVs (averaged
/// per cluster) and per-triangle material ids (verbatim; never welded across a
/// material seam)**.
///
/// - `target_ratio >= 1.0`, or a mesh too small to cluster (`< 4` verts /`<= 1`
///   tri), returns the input unchanged (re-packed into a [`MeshOutput`]).
/// - Output triangle count is always `<= input`. The grid resolution targets the
///   ratio; exact counts vary with geometry (a flat fan clusters harder than a
///   sphere), as is normal for grid clustering.
/// - UVs: if `input.uvs` is non-empty it MUST be parallel to `positions`; a
///   mismatched length disables UV averaging (output UVs empty) so a bad input can
///   never desync the streams.
/// - `material_ids`: if non-empty it MUST be parallel to `indices`; a mismatched
///   length is treated as all-zero (single material) for the same safety reason.
pub fn simplify_mesh(input: &MeshInput<'_>, target_ratio: f32) -> MeshOutput {
    let n_verts = input.positions.len();
    let n_tris = input.indices.len();

    // UVs are only honored when parallel to positions; otherwise treat untextured.
    let have_uvs = !input.uvs.is_empty() && input.uvs.len() == n_verts;
    // material_ids only honored when parallel to indices; else all-zero.
    let have_mats = !input.material_ids.is_empty() && input.material_ids.len() == n_tris;
    let mat_of = |t: usize| -> u32 { if have_mats { input.material_ids[t] } else { 0 } };

    // Passthrough: nothing to gain (or too small to cluster safely).
    if target_ratio >= 1.0 || n_tris <= 1 || n_verts < 4 {
        return MeshOutput {
            positions: input.positions.to_vec(),
            uvs: if have_uvs {
                input.uvs.to_vec()
            } else {
                Vec::new()
            },
            indices: input.indices.to_vec(),
            material_ids: (0..n_tris).map(mat_of).collect(),
            source_triangle_indices: (0..n_tris as u32).collect(),
        };
    }
    let ratio = target_ratio.clamp(1e-4, 1.0);

    // -- Per-material AABBs and grid resolutions. ---------------------------
    // Material ids may also encode a semantic surface in their upper bits. That
    // is intentional: each authored surface then retains a local detail budget.
    let mut referenced = BTreeMap::<u32, BTreeSet<usize>>::new();
    for (ti, triangle) in input.indices.iter().enumerate() {
        let vertices = referenced.entry(mat_of(ti)).or_default();
        for &index in triangle {
            let index = index as usize;
            if index < n_verts {
                vertices.insert(index);
            }
        }
    }
    let mut material_grids = BTreeMap::<u32, ([f32; 3], [u32; 3], [f32; 3])>::new();
    for (material, vertices) in referenced {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for index in &vertices {
            let position = input.positions[*index];
            for axis in 0..3 {
                if position[axis].is_finite() {
                    lo[axis] = lo[axis].min(position[axis]);
                    hi[axis] = hi[axis].max(position[axis]);
                }
            }
        }
        let mut extent = [0.0_f32; 3];
        for axis in 0..3 {
            extent[axis] = (hi[axis] - lo[axis]).max(0.0);
            if !extent[axis].is_finite() {
                lo[axis] = 0.0;
                extent[axis] = 0.0;
            }
        }
        let minimum_cells = vertices.len().min(4);
        let target_cells = ((vertices.len() as f64 * ratio as f64).round() as usize)
            .max(minimum_cells)
            .max(1);
        let resolution = grid_resolution(target_cells, extent);
        let inverse_cell = std::array::from_fn(|axis| {
            if extent[axis] > 0.0 {
                resolution[axis] as f32 / extent[axis]
            } else {
                0.0
            }
        });
        material_grids.insert(material, (lo, resolution, inverse_cell));
    }

    // -- Cluster: (cell_x, cell_y, cell_z, triangle_material) -> cluster id. --
    // A source vertex may belong to several triangle materials. It therefore
    // receives one output mapping per material; using one "incident material"
    // bucket for the vertex would let another material reuse that output vertex
    // and would cross the seam. Accumulate each distinct source-vertex/material
    // pair exactly once so UV means are independent of triangle valence.
    // The HashMap is used ONLY to resolve a key to an already-seen cluster index;
    // the OUTPUT vertex order is the first-seen order recorded in `out_pos`, so the
    // result is byte-identical across runs (no HashMap iteration in output order).
    let mut cluster_of: HashMap<(i32, i32, i32, u32), u32> = HashMap::new();
    // Per-cluster accumulators (Vec indexed by first-seen cluster id).
    let mut acc_pos: Vec<[f64; 3]> = Vec::new();
    let mut acc_uv: Vec<[f64; 2]> = Vec::new();
    let mut acc_count: Vec<u32> = Vec::new();
    let mut vertex_material_cluster: HashMap<(usize, u32), u32> = HashMap::new();
    let mut corner_clusters = vec![[u32::MAX; 3]; n_tris];

    for (ti, tri) in input.indices.iter().enumerate() {
        let material = mat_of(ti);
        for (corner, &source_index) in tri.iter().enumerate() {
            let vi = source_index as usize;
            if vi >= n_verts {
                continue;
            }
            let pair = (vi, material);
            let cid = if let Some(&cid) = vertex_material_cluster.get(&pair) {
                cid
            } else {
                let p = input.positions[vi];
                let (lo, resolution, inverse_cell) = material_grids
                    .get(&material)
                    .expect("referenced triangle material has a deterministic grid");
                let mut cell = [0_i32; 3];
                for axis in 0..3 {
                    if inverse_cell[axis] > 0.0 && p[axis].is_finite() {
                        // Clamp into [0, resolution-1] so a vertex exactly on the
                        // upper bound lands in the last cell, not one past it.
                        let value = ((p[axis] - lo[axis]) * inverse_cell[axis]).floor();
                        cell[axis] =
                            (value as i32).clamp(0, resolution[axis].saturating_sub(1) as i32);
                    }
                }
                let key = (cell[0], cell[1], cell[2], material);
                let cid = match cluster_of.get(&key) {
                    Some(&cid) => cid,
                    None => {
                        let cid = acc_pos.len() as u32;
                        cluster_of.insert(key, cid);
                        acc_pos.push([0.0; 3]);
                        acc_uv.push([0.0; 2]);
                        acc_count.push(0);
                        cid
                    }
                };
                let position_sum = &mut acc_pos[cid as usize];
                position_sum[0] += p[0] as f64;
                position_sum[1] += p[1] as f64;
                position_sum[2] += p[2] as f64;
                if have_uvs {
                    let uv = input.uvs[vi];
                    let uv_sum = &mut acc_uv[cid as usize];
                    uv_sum[0] += uv[0] as f64;
                    uv_sum[1] += uv[1] as f64;
                }
                acc_count[cid as usize] += 1;
                vertex_material_cluster.insert(pair, cid);
                cid
            };
            corner_clusters[ti][corner] = cid;
        }
    }

    // -- Finalize welded vertices (cluster means). --------------------------
    let n_out = acc_pos.len();
    let mut out_pos: Vec<[f32; 3]> = Vec::with_capacity(n_out);
    let mut out_uv: Vec<[f32; 2]> = if have_uvs {
        Vec::with_capacity(n_out)
    } else {
        Vec::new()
    };
    for c in 0..n_out {
        let n = acc_count[c].max(1) as f64;
        let p = acc_pos[c];
        out_pos.push([(p[0] / n) as f32, (p[1] / n) as f32, (p[2] / n) as f32]);
        if have_uvs {
            let uv = acc_uv[c];
            out_uv.push([(uv[0] / n) as f32, (uv[1] / n) as f32]);
        }
    }

    // -- Rebuild triangles against welded vertices; drop degenerates. -------
    let mut out_idx: Vec<[u32; 3]> = Vec::with_capacity(n_tris);
    let mut out_mat: Vec<u32> = Vec::with_capacity(n_tris);
    let mut out_source: Vec<u32> = Vec::with_capacity(n_tris);
    for (ti, _) in input.indices.iter().enumerate() {
        let [ca, cb, cc] = corner_clusters[ti];
        // Out-of-range source index → an unmapped corner; drop defensively.
        if ca == u32::MAX || cb == u32::MAX || cc == u32::MAX {
            continue;
        }
        // Degenerate after the collapse: two or three corners welded together.
        if ca == cb || cb == cc || ca == cc {
            continue;
        }
        out_idx.push([ca, cb, cc]);
        out_mat.push(mat_of(ti));
        out_source.push(ti as u32);
    }

    // -- Exact material/UV contract preservation. --------------------------
    // A tiny trim, leaf, decal, or reveal can still vanish if every one of its
    // triangles degenerates in a coarse grid. Likewise, the only surviving UVs
    // could all average to one value. Neither is an acceptable textured LOD: for
    // those material groups, remove the partial simplified result and append the
    // exact source triangles. This happens in the offline cook callers; runtime
    // only consumes the resulting immutable mesh.
    let mut source_materials = BTreeSet::new();
    let mut source_uv_contract = BTreeMap::<u32, ([f32; 2], bool)>::new();
    for (ti, triangle) in input.indices.iter().enumerate() {
        let material = mat_of(ti);
        source_materials.insert(material);
        if !have_uvs {
            continue;
        }
        for &index in triangle {
            let Some(&uv) = input.uvs.get(index as usize) else {
                continue;
            };
            source_uv_contract
                .entry(material)
                .and_modify(|(first, varied)| {
                    *varied |=
                        (uv[0] - first[0]).abs() > 1.0e-6 || (uv[1] - first[1]).abs() > 1.0e-6;
                })
                .or_insert((uv, false));
        }
    }
    let mut output_materials = BTreeSet::new();
    let mut output_uv_contract = BTreeMap::<u32, ([f32; 2], bool)>::new();
    for (triangle, &material) in out_idx.iter().zip(&out_mat) {
        output_materials.insert(material);
        if !have_uvs {
            continue;
        }
        for &index in triangle {
            let Some(&uv) = out_uv.get(index as usize) else {
                continue;
            };
            output_uv_contract
                .entry(material)
                .and_modify(|(first, varied)| {
                    *varied |=
                        (uv[0] - first[0]).abs() > 1.0e-6 || (uv[1] - first[1]).abs() > 1.0e-6;
                })
                .or_insert((uv, false));
        }
    }
    let restore_materials: BTreeSet<u32> = source_materials
        .into_iter()
        .filter(|material| {
            if !output_materials.contains(material) {
                return true;
            }
            have_uvs
                && source_uv_contract
                    .get(material)
                    .is_some_and(|(_, varied)| *varied)
                && !output_uv_contract
                    .get(material)
                    .is_some_and(|(_, varied)| *varied)
        })
        .collect();
    if !restore_materials.is_empty() {
        let mut retained_indices = Vec::with_capacity(out_idx.len());
        let mut retained_materials = Vec::with_capacity(out_mat.len());
        let mut retained_sources = Vec::with_capacity(out_source.len());
        for ((triangle, material), source) in out_idx.into_iter().zip(out_mat).zip(out_source) {
            if !restore_materials.contains(&material) {
                retained_indices.push(triangle);
                retained_materials.push(material);
                retained_sources.push(source);
            }
        }
        out_idx = retained_indices;
        out_mat = retained_materials;
        out_source = retained_sources;

        let mut exact_vertex = HashMap::<(usize, u32), u32>::new();
        for (ti, triangle) in input.indices.iter().enumerate() {
            let material = mat_of(ti);
            if !restore_materials.contains(&material) {
                continue;
            }
            let mut restored = [0_u32; 3];
            let mut complete = true;
            for (corner, &source_index) in triangle.iter().enumerate() {
                let source_index = source_index as usize;
                let Some(&position) = input.positions.get(source_index) else {
                    complete = false;
                    break;
                };
                let key = (source_index, material);
                restored[corner] = if let Some(&index) = exact_vertex.get(&key) {
                    index
                } else {
                    let index = out_pos.len() as u32;
                    out_pos.push(position);
                    if have_uvs {
                        out_uv.push(input.uvs[source_index]);
                    }
                    exact_vertex.insert(key, index);
                    index
                };
            }
            if complete {
                out_idx.push(restored);
                out_mat.push(material);
                out_source.push(ti as u32);
            }
        }
    }

    MeshOutput {
        positions: out_pos,
        uvs: out_uv,
        indices: out_idx,
        material_ids: out_mat,
        source_triangle_indices: out_source,
    }
}

/// Choose a per-axis integer grid resolution whose cell-product is ~`target_cells`,
/// distributed in proportion to the AABB extents. Deterministic. Each axis is at
/// least 1; a zero-extent axis stays at 1.
fn grid_resolution(target_cells: usize, extent: [f32; 3]) -> [u32; 3] {
    // Count axes with real extent; collapse zero-extent axes to a single layer.
    let live: Vec<usize> = (0..3).filter(|&k| extent[k] > 0.0).collect();
    if live.is_empty() {
        return [1, 1, 1];
    }
    // Geometric mean cell-count per live axis, then scale by each axis' fraction of
    // the total live extent so longer axes get more subdivisions.
    let total_extent: f64 = live.iter().map(|&k| extent[k] as f64).sum();
    let per_axis = (target_cells as f64).powf(1.0 / live.len() as f64);
    let mut res = [1u32; 3];
    for &k in &live {
        let frac = (extent[k] as f64) / total_extent * live.len() as f64;
        let r = (per_axis * frac).round().max(1.0);
        res[k] = r as u32;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an N×N grid of quads on the z=0 plane (one material), with UVs
    /// spanning [0,1]². Returns parallel positions/uvs/indices.
    fn grid_plane(n: usize) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[u32; 3]>) {
        let mut pos = Vec::new();
        let mut uv = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let fx = x as f32 / n as f32;
                let fy = y as f32 / n as f32;
                pos.push([fx, fy, 0.0]);
                uv.push([fx, fy]);
            }
        }
        let mut idx = Vec::new();
        let stride = (n + 1) as u32;
        for y in 0..n as u32 {
            for x in 0..n as u32 {
                let i0 = y * stride + x;
                let i1 = i0 + 1;
                let i2 = i0 + stride;
                let i3 = i2 + 1;
                idx.push([i0, i1, i2]);
                idx.push([i1, i3, i2]);
            }
        }
        (pos, uv, idx)
    }

    #[test]
    fn reduces_triangle_count_and_preserves_uv_range() {
        let (pos, uv, idx) = grid_plane(16); // 32*16 = 512 tris, 289 verts
        let mats = vec![0u32; idx.len()];
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &mats,
        };
        let out = simplify_mesh(&input, 0.1);

        assert!(
            out.indices.len() < idx.len(),
            "must reduce triangles: {} -> {}",
            idx.len(),
            out.indices.len()
        );
        assert!(!out.indices.is_empty(), "must keep some triangles");
        // UVs preserved (non-empty, parallel to positions, in the [0,1] range).
        assert_eq!(
            out.uvs.len(),
            out.positions.len(),
            "UVs parallel to positions"
        );
        for uv in &out.uvs {
            assert!(
                (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
                "averaged UV must stay in [0,1]: {uv:?}"
            );
        }
        // material_ids parallel to surviving triangles, all material 0.
        assert_eq!(out.material_ids.len(), out.indices.len());
        assert_eq!(out.source_triangle_indices.len(), out.indices.len());
        assert!(out.material_ids.iter().all(|&m| m == 0));
        assert!(
            out.source_triangle_indices
                .iter()
                .all(|&source| (source as usize) < idx.len()),
            "every simplified triangle must retain valid source provenance"
        );
        // Every index references a valid output vertex.
        for t in &out.indices {
            for &i in t {
                assert!((i as usize) < out.positions.len(), "index out of range");
            }
            assert!(
                t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
                "no degenerate survives"
            );
        }
    }

    #[test]
    fn never_welds_across_material_seam() {
        // Two well-extended quads — one material 0, one material 5 — share the
        // SAME source vertex at their seam. The material-aware corner mapping must
        // explicitly duplicate it; assigning one incident-material bucket to the
        // source vertex would bridge the two output material groups.
        let pos = vec![
            // Quad A (material 0).
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 4.0, 0.0], // shared source seam vertex
            // Quad B (material 5).
            [8.0, 4.0, 0.0],
            [4.0, 8.0, 0.0],
            [8.0, 8.0, 0.0],
        ];
        let uv = vec![[0.0, 0.0]; 7];
        let idx = vec![
            [0, 1, 2],
            [1, 3, 2], // material 0 (quad A)
            [3, 4, 5],
            [4, 6, 5], // material 5 (quad B)
        ];
        let mats = vec![0u32, 0, 5, 5];
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &mats,
        };
        // Moderate ratio keeps each quad's triangles while the shared source
        // vertex would bridge the materials without per-material duplication.
        let out = simplify_mesh(&input, 0.99);

        // Both materials survive with their own triangles.
        let m0 = out.material_ids.iter().filter(|&&m| m == 0).count();
        let m5 = out.material_ids.iter().filter(|&&m| m == 5).count();
        assert!(m0 >= 1, "material-0 triangles must survive, got {m0}");
        assert!(m5 >= 1, "material-5 triangles must survive, got {m5}");
        // No surviving triangle may reference a vertex shared between the two
        // material groups: check the two groups use disjoint vertex sets.
        let mut verts_m0 = std::collections::BTreeSet::new();
        let mut verts_m5 = std::collections::BTreeSet::new();
        for (t, &m) in out.indices.iter().zip(out.material_ids.iter()) {
            let set = if m == 0 { &mut verts_m0 } else { &mut verts_m5 };
            for &v in t {
                set.insert(v);
            }
        }
        let shared: Vec<_> = verts_m0.intersection(&verts_m5).collect();
        assert!(
            shared.is_empty(),
            "material 0 and 5 must not share welded vertices (seam crossed): {shared:?}"
        );
    }

    #[test]
    fn deterministic_byte_identical() {
        let (pos, uv, idx) = grid_plane(20);
        let mats: Vec<u32> = (0..idx.len()).map(|i| (i % 3) as u32).collect();
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &mats,
        };
        let a = simplify_mesh(&input, 0.15);
        let b = simplify_mesh(&input, 0.15);
        assert_eq!(
            a, b,
            "two runs on identical input must be byte-identical (replay-safe)"
        );
    }

    #[test]
    fn passthrough_when_ratio_one_or_tiny_mesh() {
        let (pos, uv, idx) = grid_plane(4);
        let mats = vec![2u32; idx.len()];
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &mats,
        };
        let out = simplify_mesh(&input, 1.0);
        assert_eq!(out.positions, pos, "ratio>=1 returns positions unchanged");
        assert_eq!(out.indices, idx, "ratio>=1 returns indices unchanged");
        assert_eq!(out.uvs, uv);
        assert_eq!(out.material_ids, mats);
        assert_eq!(
            out.source_triangle_indices,
            (0..idx.len() as u32).collect::<Vec<_>>()
        );

        // Tiny mesh (1 triangle) also passes through.
        let tpos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let tidx = vec![[0u32, 1, 2]];
        let tmats = vec![9u32];
        let tin = MeshInput {
            positions: &tpos,
            uvs: &[],
            indices: &tidx,
            material_ids: &tmats,
        };
        let tout = simplify_mesh(&tin, 0.01);
        assert_eq!(tout.indices, tidx, "single-triangle mesh passes through");
        assert_eq!(tout.material_ids, tmats);
        assert_eq!(tout.source_triangle_indices, vec![0]);
        assert!(tout.uvs.is_empty(), "untextured input -> empty output UVs");
    }

    #[test]
    fn source_triangle_provenance_survives_clustering_and_material_restore() {
        let (mut positions, mut uvs, mut indices) = grid_plane(12);
        let mut materials = vec![0_u32; indices.len()];
        let restored_source = indices.len() as u32;
        let base = positions.len() as u32;
        positions.extend([[2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0]]);
        uvs.extend([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        indices.push([base, base + 1, base + 2]);
        materials.push(7);

        let input = MeshInput {
            positions: &positions,
            uvs: &uvs,
            indices: &indices,
            material_ids: &materials,
        };
        let out = simplify_mesh(&input, 0.02);

        assert!(out.indices.len() < indices.len());
        assert_eq!(out.source_triangle_indices.len(), out.indices.len());
        assert!(
            out.source_triangle_indices.contains(&restored_source),
            "the tiny textured material must be restored with its original triangle identity"
        );
        for (&material, &source) in out.material_ids.iter().zip(&out.source_triangle_indices) {
            assert_eq!(material, materials[source as usize]);
        }
        assert_eq!(out, simplify_mesh(&input, 0.02));
    }

    #[test]
    fn empty_material_ids_treated_as_zero() {
        let (pos, uv, idx) = grid_plane(8);
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &[], // empty -> all material 0
        };
        let out = simplify_mesh(&input, 0.2);
        assert_eq!(out.material_ids.len(), out.indices.len());
        assert!(
            out.material_ids.iter().all(|&m| m == 0),
            "empty ids -> all-zero"
        );
    }
}
