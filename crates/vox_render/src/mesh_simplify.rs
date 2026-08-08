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
//! # Algorithm — QUADRIC ERROR METRIC (Garland-Heckbert) edge collapse
//!
//! 1. Every vertex accumulates the AREA-WEIGHTED sum of the plane quadrics of its
//!    incident faces. Area weighting is what makes a large flat facade dominate
//!    the small triangles sitting on it, so planes stay planar.
//! 2. OPEN BOUNDARY and MATERIAL-SEAM edges additionally receive a CONSTRAINT
//!    quadric: the plane through the edge perpendicular to the incident face,
//!    weighted by `SEAM_WEIGHT`. This is what holds a roofline straight and stops
//!    a material border drifting.
//! 3. Candidate collapses are costed by the summed quadric and taken
//!    CHEAPEST-FIRST from a binary heap. The collapse position minimises the
//!    quadric; when the system is singular (a flat or symmetric neighbourhood) it
//!    falls back to the cheaper of the endpoints and the midpoint rather than
//!    inventing an optimum off the surface.
//! 4. Every collapse is VALIDATED first: any incident triangle that would flip by
//!    more than 90 degrees, or collapse to zero area, rejects it. A
//!    quadric-optimal position can otherwise fold a fan inside out.
//! 5. Triangle `material_id` and source-triangle provenance ride through
//!    verbatim; UVs are interpolated along the collapsed edge at the parameter
//!    where the new position landed.
//!
//! Replaces an earlier vertex-grid clustering pass. That welded vertices to
//! grid-cell means with no notion of curvature, boundary or silhouette, which is
//! acceptable for foliage but destroyed architecture: flat facades went lumpy and
//! building corners dissolved (measured — see the plan
//! `docs/superpowers/plans/2026-07-27-qem-mesh-simplifier.md`).
//!
//! # Determinism
//!
//! Replay-exact: there is NO RNG, NO float-key HashMap iteration in any path that
//! affects output ordering. Adjacency and seam construction use ordered maps and
//! sets; the collapse heap has an explicit `(cost_bits, v0, v1)` tie-break; and
//! every geometry fold follows stable vertex/triangle id order.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

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
/// (clamped to `(0, 1]`) via constrained quadric-error edge collapse,
/// **preserving UVs and per-triangle material ids (verbatim; never collapsed
/// across a material seam)**.
///
/// - `target_ratio >= 1.0`, or a mesh too small to cluster (`< 4` verts /`<= 1`
///   tri), returns the input unchanged (re-packed into a [`MeshOutput`]).
/// - Output triangle count is always `<= input`. The ratio is a target; exact
///   counts vary because boundary, seam, degeneracy, and flip checks may reject
///   every remaining collapse.
/// - UVs: if `input.uvs` is non-empty it MUST be parallel to `positions`; a
///   mismatched length disables UV averaging (output UVs empty) so a bad input can
///   never desync the streams.
/// - `material_ids`: if non-empty it MUST be parallel to `indices`; a mismatched
///   length is treated as all-zero (single material) for the same safety reason.
pub fn simplify_mesh(input: &MeshInput<'_>, target_ratio: f32) -> MeshOutput {
    simplify_mesh_with_locked_vertices(input, target_ratio, &[])
}

/// Variant used by independently streamable cluster hierarchies. A locked
/// vertex is never an endpoint of a collapse, so two adjacent clusters retain
/// byte-identical positions along their shared frontier. `locked_vertices` is
/// optional; a non-parallel slice is treated as all-unlocked.
pub fn simplify_mesh_with_locked_vertices(
    input: &MeshInput<'_>,
    target_ratio: f32,
    locked_vertices: &[bool],
) -> MeshOutput {
    let n_verts = input.positions.len();
    let n_tris = input.indices.len();
    let have_locks = locked_vertices.len() == n_verts;
    let locked = |vertex: usize| have_locks && locked_vertices[vertex];

    // UVs are only honored when parallel to positions; otherwise treat untextured.
    let have_uvs = !input.uvs.is_empty() && input.uvs.len() == n_verts;
    // material_ids only honored when parallel to indices; else all-zero.
    let have_mats = !input.material_ids.is_empty() && input.material_ids.len() == n_tris;
    let mat_of = |t: usize| -> u32 { if have_mats { input.material_ids[t] } else { 0 } };

    // Passthrough: nothing to gain (or too small to simplify safely).
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
    let target_tris = ((n_tris as f64) * f64::from(ratio)).round().max(1.0) as usize;

    // ---- working state ---------------------------------------------------
    let mut pos: Vec<[f64; 3]> = input
        .positions
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        .collect();
    let mut uv: Vec<[f32; 2]> = if have_uvs {
        input.uvs.to_vec()
    } else {
        Vec::new()
    };
    // Triangles, with their ORIGINAL index kept so provenance survives.
    let mut tri: Vec<[u32; 3]> = Vec::with_capacity(n_tris);
    let mut tri_src: Vec<u32> = Vec::with_capacity(n_tris);
    let mut tri_mat: Vec<u32> = Vec::with_capacity(n_tris);
    for (ti, t) in input.indices.iter().enumerate() {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        // Drop out-of-range or already-degenerate source triangles rather than
        // letting them poison adjacency.
        if a >= n_verts || b >= n_verts || c >= n_verts || a == b || b == c || a == c {
            continue;
        }
        tri.push(*t);
        tri_src.push(ti as u32);
        tri_mat.push(mat_of(ti));
    }
    if tri.len() <= target_tris {
        return MeshOutput {
            positions: input.positions.to_vec(),
            uvs: if have_uvs {
                input.uvs.to_vec()
            } else {
                Vec::new()
            },
            indices: tri,
            material_ids: tri_mat,
            source_triangle_indices: tri_src,
        };
    }
    let mut tri_alive = vec![true; tri.len()];
    let mut vert_alive = vec![false; n_verts];
    let mut vtri: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
    for (ti, t) in tri.iter().enumerate() {
        for &v in t {
            vtri[v as usize].push(ti as u32);
            vert_alive[v as usize] = true;
        }
    }

    // ---- quadrics --------------------------------------------------------
    let mut quad = vec![Quadric::default(); n_verts];
    for (ti, t) in tri.iter().enumerate() {
        let (p0, p1, p2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        if let Some((n, d, area)) = plane_of(p0, p1, p2) {
            // Area weighting makes a large flat facade dominate the small
            // triangles that happen to sit on it — which is exactly the
            // planarity the grid clusterer destroyed.
            let q = Quadric::from_plane(n, d).scaled(area);
            for &v in t {
                quad[v as usize].add(&q);
            }
        }
        let _ = ti;
    }

    // Constraint quadrics: OPEN BOUNDARY and MATERIAL-SEAM edges get a plane
    // perpendicular to the incident face through the edge. Without these a
    // roofline drifts and a facade's border rounds off — the two failures that
    // read as "rubble" at distance.
    {
        let mut edge_faces: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
        for (ti, t) in tri.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edge_faces
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_default()
                    .push(ti as u32);
            }
        }
        for ((a, b), faces) in &edge_faces {
            let open = faces.len() == 1;
            let seam = faces.len() > 1
                && faces
                    .iter()
                    .any(|&f| tri_mat[f as usize] != tri_mat[faces[0] as usize]);
            if !open && !seam {
                continue;
            }
            let (pa, pb) = (pos[*a as usize], pos[*b as usize]);
            for &f in faces {
                let t = tri[f as usize];
                let (p0, p1, p2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
                let Some((fn_, _, _)) = plane_of(p0, p1, p2) else {
                    continue;
                };
                let e = sub3(pb, pa);
                let n = cross3(e, fn_);
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len <= 1e-20 {
                    continue;
                }
                let n = [n[0] / len, n[1] / len, n[2] / len];
                let d = -(n[0] * pa[0] + n[1] * pa[1] + n[2] * pa[2]);
                let q = Quadric::from_plane(n, d).scaled(SEAM_WEIGHT);
                quad[*a as usize].add(&q);
                quad[*b as usize].add(&q);
            }
        }
    }

    // ---- collapse loop ---------------------------------------------------
    // Lazy heap: entries carry the endpoint versions they were costed with, and
    // a stale entry is skipped rather than rebuilt. Ordering is on
    // (cost_bits, v0, v1) so two runs are byte-identical (determinism is a
    // product moat here, not a nicety).
    let mut version = vec![0u32; n_verts];
    let mut heap: BinaryHeap<Reverse<Collapse>> = BinaryHeap::new();
    let push_edge = |heap: &mut BinaryHeap<Reverse<Collapse>>,
                     quad: &[Quadric],
                     pos: &[[f64; 3]],
                     version: &[u32],
                     a: u32,
                     b: u32| {
        let (v0, v1) = if a < b { (a, b) } else { (b, a) };
        let mut q = quad[v0 as usize];
        q.add(&quad[v1 as usize]);
        let target = q.optimum(pos[v0 as usize], pos[v1 as usize]);
        let cost = q.error_at(target).max(0.0);
        heap.push(Reverse(Collapse {
            cost_bits: cost.to_bits(),
            v0,
            v1,
            ver0: version[v0 as usize],
            ver1: version[v1 as usize],
            target,
        }));
    };
    {
        let mut seen: BTreeSet<(u32, u32)> = BTreeSet::new();
        for t in &tri {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let e = if a < b { (a, b) } else { (b, a) };
                if seen.insert(e) && !locked(e.0 as usize) && !locked(e.1 as usize) {
                    push_edge(&mut heap, &quad, &pos, &version, e.0, e.1);
                }
            }
        }
    }

    let mut live_tris = tri.len();
    while live_tris > target_tris {
        let Some(Reverse(c)) = heap.pop() else {
            break; // exact-preservation floor: no legal collapse remains
        };
        let (v0, v1) = (c.v0 as usize, c.v1 as usize);
        if !vert_alive[v0] || !vert_alive[v1] {
            continue;
        }
        if version[v0] != c.ver0 || version[v1] != c.ver1 {
            continue; // stale cost — a neighbour moved since this was queued
        }
        if !collapse_is_valid(
            &tri, &tri_alive, &vtri, &pos, v0 as u32, v1 as u32, c.target,
        ) {
            continue;
        }

        // Interpolate UV along the collapsed edge by where the target landed.
        if have_uvs {
            let e = sub3(pos[v1], pos[v0]);
            let denom = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            let t = if denom > 1e-20 {
                let d = sub3(c.target, pos[v0]);
                ((d[0] * e[0] + d[1] * e[1] + d[2] * e[2]) / denom).clamp(0.0, 1.0) as f32
            } else {
                0.5
            };
            let (a, b) = (uv[v0], uv[v1]);
            uv[v0] = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        }

        pos[v0] = c.target;
        let q1 = quad[v1];
        quad[v0].add(&q1);
        vert_alive[v1] = false;
        version[v0] = version[v0].wrapping_add(1);
        version[v1] = version[v1].wrapping_add(1);

        // Re-point v1's triangles at v0, killing any that degenerate.
        let moved = std::mem::take(&mut vtri[v1]);
        for &ti in &moved {
            let ti = ti as usize;
            if !tri_alive[ti] {
                continue;
            }
            let t = &mut tri[ti];
            for s in t.iter_mut() {
                if *s == v1 as u32 {
                    *s = v0 as u32;
                }
            }
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                tri_alive[ti] = false;
                live_tris -= 1;
            } else {
                vtri[v0].push(ti as u32);
            }
        }

        // Re-cost every edge still touching v0.
        let mut ring: BTreeSet<u32> = BTreeSet::new();
        for &ti in &vtri[v0] {
            if !tri_alive[ti as usize] {
                continue;
            }
            for &v in &tri[ti as usize] {
                if v != v0 as u32 && vert_alive[v as usize] {
                    ring.insert(v);
                }
            }
        }
        for v in ring {
            if !locked(v0) && !locked(v as usize) {
                push_edge(&mut heap, &quad, &pos, &version, v0 as u32, v);
            }
        }
    }

    // ---- rebuild ---------------------------------------------------------
    let mut remap = vec![u32::MAX; n_verts];
    let mut out_pos: Vec<[f32; 3]> = Vec::new();
    let mut out_uv: Vec<[f32; 2]> = Vec::new();
    let mut out_idx: Vec<[u32; 3]> = Vec::new();
    let mut out_mat: Vec<u32> = Vec::new();
    let mut out_src: Vec<u32> = Vec::new();
    for (ti, t) in tri.iter().enumerate() {
        if !tri_alive[ti] {
            continue;
        }
        let mut o = [0u32; 3];
        for (k, &v) in t.iter().enumerate() {
            let v = v as usize;
            if remap[v] == u32::MAX {
                remap[v] = out_pos.len() as u32;
                let p = pos[v];
                out_pos.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                if have_uvs {
                    out_uv.push(uv[v]);
                }
            }
            o[k] = remap[v];
        }
        if o[0] == o[1] || o[1] == o[2] || o[0] == o[2] {
            continue;
        }
        out_idx.push(o);
        out_mat.push(tri_mat[ti]);
        out_src.push(tri_src[ti]);
    }

    MeshOutput {
        positions: out_pos,
        uvs: out_uv,
        indices: out_idx,
        material_ids: out_mat,
        source_triangle_indices: out_src,
    }
}

/// Weight applied to OPEN-BOUNDARY and MATERIAL-SEAM constraint planes. High
/// enough that a collapse which would move a seam is never the cheapest option
/// while the interior still has slack.
const SEAM_WEIGHT: f64 = 1.0e3;

/// One queued edge collapse. Ordering is `(cost_bits, v0, v1)`: `cost` is a
/// NON-NEGATIVE f64, whose `to_bits()` is monotonic, so this is a total order that
/// does not depend on float `PartialOrd` — two runs pop in the same sequence.
#[derive(Clone, Copy)]
struct Collapse {
    cost_bits: u64,
    v0: u32,
    v1: u32,
    ver0: u32,
    ver1: u32,
    target: [f64; 3],
}

// Equality is defined on the ORDERING KEY ONLY. `target` is an f64 payload and
// plays no part in heap order; deriving Eq over it would not even compile.
impl PartialEq for Collapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost_bits == other.cost_bits && self.v0 == other.v0 && self.v1 == other.v1
    }
}
impl Eq for Collapse {}

impl Ord for Collapse {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost_bits
            .cmp(&other.cost_bits)
            .then(self.v0.cmp(&other.v0))
            .then(self.v1.cmp(&other.v1))
    }
}
impl PartialOrd for Collapse {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Symmetric 4x4 quadric stored as its 10 distinct coefficients.
#[derive(Clone, Copy, Default)]
struct Quadric {
    xx: f64,
    xy: f64,
    xz: f64,
    xw: f64,
    yy: f64,
    yz: f64,
    yw: f64,
    zz: f64,
    zw: f64,
    ww: f64,
}

impl Quadric {
    fn from_plane(n: [f64; 3], d: f64) -> Self {
        Self {
            xx: n[0] * n[0],
            xy: n[0] * n[1],
            xz: n[0] * n[2],
            xw: n[0] * d,
            yy: n[1] * n[1],
            yz: n[1] * n[2],
            yw: n[1] * d,
            zz: n[2] * n[2],
            zw: n[2] * d,
            ww: d * d,
        }
    }
    fn scaled(mut self, s: f64) -> Self {
        self.xx *= s;
        self.xy *= s;
        self.xz *= s;
        self.xw *= s;
        self.yy *= s;
        self.yz *= s;
        self.yw *= s;
        self.zz *= s;
        self.zw *= s;
        self.ww *= s;
        self
    }
    fn add(&mut self, o: &Self) {
        self.xx += o.xx;
        self.xy += o.xy;
        self.xz += o.xz;
        self.xw += o.xw;
        self.yy += o.yy;
        self.yz += o.yz;
        self.yw += o.yw;
        self.zz += o.zz;
        self.zw += o.zw;
        self.ww += o.ww;
    }
    fn error_at(&self, v: [f64; 3]) -> f64 {
        let (x, y, z) = (v[0], v[1], v[2]);
        self.xx * x * x
            + 2.0 * self.xy * x * y
            + 2.0 * self.xz * x * z
            + 2.0 * self.xw * x
            + self.yy * y * y
            + 2.0 * self.yz * y * z
            + 2.0 * self.yw * y
            + self.zz * z * z
            + 2.0 * self.zw * z
            + self.ww
    }
    /// Position minimising this quadric. Falls back to the cheaper of the two
    /// endpoints and their midpoint when the system is singular — a flat or
    /// symmetric neighbourhood has no unique optimum and inventing one there is
    /// how a simplifier puts vertices off the surface.
    ///
    /// The determinant test alone does not catch that. `det` is an ABSOLUTE
    /// threshold on a quantity that scales with the mesh's units and area
    /// weighting, so a NEARLY singular neighbourhood — a foliage card, a scan
    /// shell's near-coplanar fan — passes it and yields an optimum that is
    /// finite but nowhere near the edge it replaces. Measured on the cooked
    /// prototype set, that put `ph.boulder_01` LOD1 vertices 5.27 m outside the
    /// source silhouette and hulled a 0.15 m celandine to 2.28 m, in 20 of 78
    /// cooked levels. So the optimum must ALSO land near its edge; when it does
    /// not, the endpoint/midpoint fallback below is the honest answer.
    fn optimum(&self, a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        let m = [
            [self.xx, self.xy, self.xz],
            [self.xy, self.yy, self.yz],
            [self.xz, self.yz, self.zz],
        ];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() > 1e-12 {
            let r = [-self.xw, -self.yw, -self.zw];
            let inv_det = 1.0 / det;
            let x = (r[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (r[1] * m[2][2] - m[1][2] * r[2])
                + m[0][2] * (r[1] * m[2][1] - m[1][1] * r[2]))
                * inv_det;
            let y = (m[0][0] * (r[1] * m[2][2] - m[1][2] * r[2])
                - r[0] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * r[2] - r[1] * m[2][0]))
                * inv_det;
            let z = (m[0][0] * (m[1][1] * r[2] - r[1] * m[2][1])
                - m[0][1] * (m[1][0] * r[2] - r[1] * m[2][0])
                + r[0] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                * inv_det;
            let v = [x, y, z];
            if v.iter().all(|c| c.is_finite()) && optimum_is_near_edge(v, a, b) {
                return v;
            }
        }
        let mid = [
            (a[0] + b[0]) * 0.5,
            (a[1] + b[1]) * 0.5,
            (a[2] + b[2]) * 0.5,
        ];
        let mut best = a;
        let mut best_e = self.error_at(a);
        for cand in [b, mid] {
            let e = self.error_at(cand);
            if e < best_e {
                best_e = e;
                best = cand;
            }
        }
        best
    }
}

/// How far, in multiples of the collapsing edge's own length, a quadric-optimal
/// position may sit from that edge's midpoint.
///
/// A genuine feature-preserving optimum — the corner a cube's three planes meet
/// at, the crease two roof slopes share — lies at most about one edge length
/// out, because the planes that define it are the planes of the faces touching
/// that edge. Two edge lengths therefore keeps every legitimate optimum while
/// rejecting the near-singular solves, which miss by one to three ORDERS of
/// magnitude, not by a factor of two.
const OPTIMUM_REACH_IN_EDGE_LENGTHS: f64 = 2.0;

/// Does a quadric-optimal position sit close enough to the edge it replaces to
/// be believable? A degenerate (zero-length) edge has no scale to judge against,
/// so nothing but its own endpoints is believable there.
fn optimum_is_near_edge(v: [f64; 3], a: [f64; 3], b: [f64; 3]) -> bool {
    let mid = [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ];
    let edge = sub3(b, a);
    let half_length = (edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]).sqrt() * 0.5;
    let offset = sub3(v, mid);
    let distance = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
    distance <= half_length * 2.0 * OPTIMUM_REACH_IN_EDGE_LENGTHS
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit normal, plane offset, and TWICE the triangle area (the area weight).
fn plane_of(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Option<([f64; 3], f64, f64)> {
    let n = cross3(sub3(p1, p0), sub3(p2, p0));
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if !len.is_finite() || len <= 1e-20 {
        return None;
    }
    let n = [n[0] / len, n[1] / len, n[2] / len];
    let d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);
    Some((n, d, len))
}

/// Reject a collapse that would flip an incident triangle over (>90 degrees) or
/// drive one to zero area. Without this a quadric-optimal position can happily
/// fold a fan inside out — visually far worse than the error it saves.
fn collapse_is_valid(
    tri: &[[u32; 3]],
    tri_alive: &[bool],
    vtri: &[Vec<u32>],
    pos: &[[f64; 3]],
    v0: u32,
    v1: u32,
    target: [f64; 3],
) -> bool {
    for &v in &[v0, v1] {
        for &ti in &vtri[v as usize] {
            let ti = ti as usize;
            if !tri_alive[ti] {
                continue;
            }
            let t = tri[ti];
            // Triangles that vanish in this collapse cannot flip.
            if t.contains(&v0) && t.contains(&v1) {
                continue;
            }
            let before = plane_of(pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            let mut moved = [[0.0f64; 3]; 3];
            for (k, &idx) in t.iter().enumerate() {
                moved[k] = if idx == v0 || idx == v1 {
                    target
                } else {
                    pos[idx as usize]
                };
            }
            let after = plane_of(moved[0], moved[1], moved[2]);
            match (before, after) {
                (Some((nb, _, _)), Some((na, _, _))) => {
                    let dot = nb[0] * na[0] + nb[1] * na[1] + nb[2] * na[2];
                    if dot <= 0.0 {
                        return false;
                    }
                }
                (Some(_), None) => return false, // would become degenerate
                _ => {}
            }
        }
    }
    true
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
        // CONTRACT CHANGE (QEM, 2026-07-27). The grid clusterer keyed clusters on
        // (cell, material) and therefore DUPLICATED every vertex shared by two
        // materials, so the two groups came out with disjoint vertex sets. QEM does
        // not split the mesh: this INPUT already shares vertices 3/4/5 between the
        // two quads, and duplicating them would open a crack along the seam and
        // inflate the vertex count. The seam is held by CONSTRAINT QUADRICS instead.
        //
        // What must still hold, and is what actually matters downstream:
        //   1. every surviving triangle keeps its SOURCE material, and
        //   2. the seam vertices themselves are preserved geometrically.
        for (k, &src) in out.source_triangle_indices.iter().enumerate() {
            assert_eq!(
                out.material_ids[k], mats[src as usize],
                "triangle {k} changed material away from its source"
            );
        }
        for seam in [pos[3], pos[4], pos[5]] {
            let kept = out.positions.iter().any(|p| {
                (p[0] - seam[0]).abs() < 1e-4
                    && (p[1] - seam[1]).abs() < 1e-4
                    && (p[2] - seam[2]).abs() < 1e-4
            });
            assert!(kept, "seam vertex {seam:?} was not preserved");
        }
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
    fn locked_cluster_frontier_vertices_remain_exact() {
        let (pos, uv, idx) = grid_plane(16);
        let mats = vec![0u32; idx.len()];
        let input = MeshInput {
            positions: &pos,
            uvs: &uv,
            indices: &idx,
            material_ids: &mats,
        };
        let mut locked = vec![false; pos.len()];
        for y in 0..=16 {
            locked[y * 17 + 8] = true;
        }
        let out = simplify_mesh_with_locked_vertices(&input, 0.1, &locked);
        assert!(out.indices.len() < idx.len());
        for (source, &is_locked) in pos.iter().zip(&locked) {
            if is_locked {
                assert!(
                    out.positions.contains(source),
                    "locked frontier vertex {source:?} moved or disappeared"
                );
            }
        }
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

#[cfg(test)]
mod qem_tests {
    use super::*;

    /// A subdivided unit cube: six faces, each an n×n quad grid, welded per face
    /// (so face borders are UV/attribute seams exactly as a cooked building has).
    fn subdivided_cube(n: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        // (origin, du, dv) per face of the unit cube.
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ];
        for (o, du, dv) in faces {
            let base = pos.len() as u32;
            for j in 0..=n {
                for i in 0..=n {
                    let (s, t) = (i as f32 / n as f32, j as f32 / n as f32);
                    pos.push([
                        o[0] + du[0] * s + dv[0] * t,
                        o[1] + du[1] * s + dv[1] * t,
                        o[2] + du[2] * s + dv[2] * t,
                    ]);
                }
            }
            let row = (n + 1) as u32;
            for j in 0..n as u32 {
                for i in 0..n as u32 {
                    let a = base + j * row + i;
                    idx.push([a, a + 1, a + row]);
                    idx.push([a + 1, a + row + 1, a + row]);
                }
            }
        }
        (pos, idx)
    }

    /// One-sided Hausdorff: worst distance from any REFERENCE vertex to the
    /// nearest REDUCED vertex. Deliberately independent of `asset::lod_error` so
    /// this crate's test does not depend on the game crate.
    fn hausdorff(reference: &[[f32; 3]], reduced: &[[f32; 3]]) -> f32 {
        let mut worst = 0.0f32;
        for r in reference {
            let mut best = f32::INFINITY;
            for q in reduced {
                let d = (r[0] - q[0]).powi(2) + (r[1] - q[1]).powi(2) + (r[2] - q[2]).powi(2);
                if d < best {
                    best = d;
                }
            }
            worst = worst.max(best.sqrt());
        }
        worst
    }

    /// Every output vertex of a decimated cube must still lie ON the cube surface
    /// (some coordinate at 0 or 1). Grid clustering moves vertices to cell MEANS,
    /// which lifts them off the faces — that is the "lumpy facade" failure.
    #[test]
    fn qem_keeps_vertices_on_the_surface_of_a_box() {
        let (pos, idx) = subdivided_cube(8);
        let out = simplify_mesh(
            &MeshInput {
                positions: &pos,
                uvs: &[],
                indices: &idx,
                material_ids: &[],
            },
            0.25,
        );
        assert!(!out.positions.is_empty(), "decimation produced no geometry");
        let mut worst_off = 0.0f32;
        for p in &out.positions {
            // Distance to the nearest of the six planes x/y/z = 0 or 1.
            let off = p
                .iter()
                .map(|c| c.abs().min((c - 1.0).abs()))
                .fold(f32::INFINITY, f32::min);
            worst_off = worst_off.max(off);
        }
        assert!(
            worst_off < 1e-3,
            "QEM moved a vertex {worst_off} off the cube surface — planarity lost"
        );
    }

    /// The eight cube corners are the silhouette. Losing them is exactly what
    /// makes a distant building read as rubble.
    #[test]
    fn qem_preserves_the_corners_of_a_box() {
        let (pos, idx) = subdivided_cube(8);
        let out = simplify_mesh(
            &MeshInput {
                positions: &pos,
                uvs: &[],
                indices: &idx,
                material_ids: &[],
            },
            0.25,
        );
        for cx in [0.0f32, 1.0] {
            for cy in [0.0f32, 1.0] {
                for cz in [0.0f32, 1.0] {
                    let corner = [cx, cy, cz];
                    let d = hausdorff(&[corner], &out.positions);
                    assert!(
                        d < 1e-3,
                        "corner {corner:?} lost (nearest output vertex {d} away)"
                    );
                }
            }
        }
    }

    /// Byte-identical output across runs — determinism is a product moat here.
    #[test]
    fn qem_is_deterministic() {
        let (pos, idx) = subdivided_cube(6);
        let mk = || {
            simplify_mesh(
                &MeshInput {
                    positions: &pos,
                    uvs: &[],
                    indices: &idx,
                    material_ids: &[],
                },
                0.3,
            )
        };
        let a = mk();
        let b = mk();
        assert_eq!(a.positions, b.positions, "positions differ between runs");
        assert_eq!(a.indices, b.indices, "indices differ between runs");
    }

    /// Per-triangle material ids ride through verbatim and stay parallel.
    #[test]
    fn qem_carries_material_ids_and_provenance() {
        let (pos, idx) = subdivided_cube(6);
        // Two materials split by triangle parity — plenty of seam edges.
        let mats: Vec<u32> = (0..idx.len() as u32).map(|i| i % 2).collect();
        let out = simplify_mesh(
            &MeshInput {
                positions: &pos,
                uvs: &[],
                indices: &idx,
                material_ids: &mats,
            },
            0.4,
        );
        assert_eq!(
            out.material_ids.len(),
            out.indices.len(),
            "material_ids must stay parallel to indices"
        );
        assert_eq!(
            out.source_triangle_indices.len(),
            out.indices.len(),
            "provenance must stay parallel to indices"
        );
        for (k, &src) in out.source_triangle_indices.iter().enumerate() {
            assert_eq!(
                out.material_ids[k], mats[src as usize],
                "triangle {k} changed material away from its source"
            );
        }
    }

    /// UVs stay inside the authored range and parallel to positions.
    #[test]
    fn qem_preserves_uv_stream() {
        let (pos, idx) = subdivided_cube(6);
        let uvs: Vec<[f32; 2]> = pos.iter().map(|p| [p[0], p[1]]).collect();
        let out = simplify_mesh(
            &MeshInput {
                positions: &pos,
                uvs: &uvs,
                indices: &idx,
                material_ids: &[],
            },
            0.35,
        );
        assert_eq!(out.uvs.len(), out.positions.len(), "UV stream desynced");
        for uv in &out.uvs {
            assert!(
                (-1e-3..=1.0 + 1e-3).contains(&uv[0]) && (-1e-3..=1.0 + 1e-3).contains(&uv[1]),
                "UV {uv:?} escaped the authored [0,1] range"
            );
        }
    }
}
