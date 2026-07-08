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
//!   each triangle on its own material. We NEVER weld a vertex across a material
//!   seam: a vertex is clustered together with another only if both share the same
//!   grid cell *and* the same incident-material set bucket, so a collapse can never
//!   merge two differently-materialled surfaces. Degenerate triangles (two or three
//!   corners welded to the same output vertex) are dropped.
//!
//! # Algorithm — vertex-GRID CLUSTERING
//!
//! 1. Compute the AABB of all positions.
//! 2. Pick a uniform grid resolution from `target_ratio` (fewer cells → fewer
//!    output vertices → fewer triangles). The resolution is derived purely from
//!    the input vertex count and the target ratio, so it is a deterministic
//!    function of the input.
//! 3. Snap every vertex to its integer grid cell. The cluster key is
//!    `(cell_x, cell_y, cell_z, material_bucket)` — see the material-seam note
//!    above. All vertices sharing a key weld to one OUTPUT vertex whose position
//!    and UV are the arithmetic mean of the members (a stable, order-independent
//!    reduction — we accumulate in a Vec indexed by first-seen cluster order, so
//!    there is no HashMap iteration in the output ordering).
//! 4. Rebuild the triangle list against the welded vertices; drop any triangle
//!    whose three corners no longer reference three distinct output vertices
//!    (degenerate after the collapse). Each surviving triangle keeps its ORIGINAL
//!    `material_id`.
//!
//! # Determinism
//!
//! Replay-exact: there is NO RNG, NO float-key HashMap iteration in any path that
//! affects output ordering, and all folds are over id-ordered Vecs. The cluster
//! lookup uses a `HashMap` keyed by integer cell coordinates ONLY to find the
//! cluster index; the *output* vertex order is the first-seen order recorded in a
//! Vec, so two runs on identical input produce byte-identical output. Position/UV
//! averaging is a fixed-order sum (input vertex order) divided by the count.

use std::collections::HashMap;

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
        };
    }
    let ratio = target_ratio.clamp(1e-4, 1.0);

    // -- AABB over all positions (skip non-finite components). --------------
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in input.positions {
        for k in 0..3 {
            if p[k].is_finite() {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    // A zero/degenerate extent on any axis collapses that axis to a single layer.
    let mut extent = [0.0f32; 3];
    for k in 0..3 {
        extent[k] = (hi[k] - lo[k]).max(0.0);
        if !extent[k].is_finite() {
            extent[k] = 0.0;
        }
    }

    // -- Grid resolution from the target ratio. -----------------------------
    // We want roughly `n_verts * ratio` output vertices. Spread that budget over
    // the three axes proportionally to extent (so a thin mesh gets a coarse grid
    // on its thin axis). Deterministic function of (n_verts, ratio, extents).
    let target_cells = ((n_verts as f64 * ratio as f64).round() as usize).max(1);
    let res = grid_resolution(target_cells, extent);
    let inv_cell = [
        if extent[0] > 0.0 {
            res[0] as f32 / extent[0]
        } else {
            0.0
        },
        if extent[1] > 0.0 {
            res[1] as f32 / extent[1]
        } else {
            0.0
        },
        if extent[2] > 0.0 {
            res[2] as f32 / extent[2]
        } else {
            0.0
        },
    ];
    let cell_of = |p: [f32; 3]| -> [i32; 3] {
        let mut c = [0i32; 3];
        for k in 0..3 {
            if inv_cell[k] > 0.0 && p[k].is_finite() {
                // Clamp into [0, res-1] so a vertex exactly on `hi` lands in the
                // last cell, not one past it.
                let f = ((p[k] - lo[k]) * inv_cell[k]).floor();
                c[k] = (f as i32).clamp(0, res[k].saturating_sub(1) as i32);
            }
        }
        c
    };

    // -- Per-vertex incident-material bucket (the material-seam guard). ------
    // A vertex is welded with another ONLY if both share a grid cell AND the same
    // incident-material bucket. We bucket a vertex by the SMALLEST material id of
    // any triangle that touches it; this guarantees two vertices on opposite sides
    // of a bark/leaf seam (different smallest incident material) never weld, so a
    // collapse can never merge two differently-materialled surfaces. (A vertex with
    // no incident triangle buckets to u32::MAX and clusters only with its own kind.)
    let mut vert_mat_bucket = vec![u32::MAX; n_verts];
    for (ti, tri) in input.indices.iter().enumerate() {
        let m = mat_of(ti);
        for &vi in tri {
            let vi = vi as usize;
            if vi < n_verts {
                vert_mat_bucket[vi] = vert_mat_bucket[vi].min(m);
            }
        }
    }

    // -- Cluster: (cell_x, cell_y, cell_z, material_bucket) -> cluster index. -
    // The HashMap is used ONLY to resolve a key to an already-seen cluster index;
    // the OUTPUT vertex order is the first-seen order recorded in `out_pos`, so the
    // result is byte-identical across runs (no HashMap iteration in output order).
    let mut cluster_of: HashMap<(i32, i32, i32, u32), u32> = HashMap::new();
    // Per-cluster accumulators (Vec indexed by first-seen cluster id).
    let mut acc_pos: Vec<[f64; 3]> = Vec::new();
    let mut acc_uv: Vec<[f64; 2]> = Vec::new();
    let mut acc_count: Vec<u32> = Vec::new();
    // Map every input vertex -> its output cluster id.
    let mut vert_to_cluster = vec![u32::MAX; n_verts];

    for vi in 0..n_verts {
        let p = input.positions[vi];
        let cell = cell_of(p);
        let key = (cell[0], cell[1], cell[2], vert_mat_bucket[vi]);
        let cid = match cluster_of.get(&key) {
            Some(&c) => c,
            None => {
                let c = acc_pos.len() as u32;
                cluster_of.insert(key, c);
                acc_pos.push([0.0; 3]);
                acc_uv.push([0.0; 2]);
                acc_count.push(0);
                c
            }
        };
        vert_to_cluster[vi] = cid;
        let a = &mut acc_pos[cid as usize];
        a[0] += p[0] as f64;
        a[1] += p[1] as f64;
        a[2] += p[2] as f64;
        if have_uvs {
            let uv = input.uvs[vi];
            let au = &mut acc_uv[cid as usize];
            au[0] += uv[0] as f64;
            au[1] += uv[1] as f64;
        }
        acc_count[cid as usize] += 1;
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
    for (ti, tri) in input.indices.iter().enumerate() {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        // Out-of-range index → drop the triangle (defensive; never panic).
        if a >= n_verts || b >= n_verts || c >= n_verts {
            continue;
        }
        let ca = vert_to_cluster[a];
        let cb = vert_to_cluster[b];
        let cc = vert_to_cluster[c];
        // Degenerate after the collapse: two or three corners welded together.
        if ca == cb || cb == cc || ca == cc {
            continue;
        }
        out_idx.push([ca, cb, cc]);
        out_mat.push(mat_of(ti));
    }

    MeshOutput {
        positions: out_pos,
        uvs: out_uv,
        indices: out_idx,
        material_ids: out_mat,
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
        assert!(out.material_ids.iter().all(|&m| m == 0));
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
        // Two spatially-separated, well-extended quads — one material 0, one
        // material 5 — that nonetheless SHARE a coincident seam-vertex pair (one
        // vertex per material at the SAME position). A material-BLIND grid clusterer
        // would weld that pair, bridging the two materials onto one welded vertex;
        // the seam guard (incident-material bucket in the cluster key) must keep
        // them apart. The quads are large/separated enough that a moderate ratio
        // keeps each quad's three distinct corners (so triangles survive) while the
        // coincident seam pair still wants to weld.
        let pos = vec![
            // Quad A (material 0) — left, far from the right quad except the seam.
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 4.0, 0.0], // <- A's seam corner
            // Quad B (material 5) — right, but its bottom-left corner is COINCIDENT
            // with A's top-right corner [4,4,0] (the shared seam vertex pair).
            [4.0, 4.0, 0.0], // <- B's seam corner (same position as pos[3])
            [8.0, 4.0, 0.0],
            [4.0, 8.0, 0.0],
            [8.0, 8.0, 0.0],
        ];
        let uv = vec![[0.0, 0.0]; 8];
        let idx = vec![
            [0, 1, 2],
            [1, 3, 2], // material 0 (quad A)
            [4, 5, 6],
            [5, 7, 6], // material 5 (quad B)
        ];
        let mats = vec![0u32, 0, 5, 5];
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &mats,
        };
        // Moderate ratio: a material-blind clusterer WOULD weld the coincident seam
        // pair (pos[3] and pos[4]) into one vertex, bridging the materials. The seam
        // guard must prevent it while still keeping each quad's triangles.
        let out = simplify_mesh(&input, 0.9);

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
        assert!(tout.uvs.is_empty(), "untextured input -> empty output UVs");
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
