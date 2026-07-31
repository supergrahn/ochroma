//! CONTACT DECALS — where an object meets a surface.
//!
//! # The defect
//!
//! Every rock, trunk, plinth and building in the live frame meets the ground as
//! a bare polygonal seam: a hard intersection line with no dirt, no contact
//! darkening, no transition. It is one of the strongest "this is a game" tells
//! in every frame the renderer produces. Real objects do not meet the ground
//! like that — soil splash, leaf litter, moss and wind-blown grime accumulate in
//! a band along the contact line, and the crevice itself is deeply occluded.
//!
//! # Why not a screen-space decal pass
//!
//! In a deferred rasteriser a decal is a projection pass over a G-buffer. **This
//! renderer has no such pass.** It is a Slang megakernel over a hardware-RT
//! acceleration structure (per-proto BLAS + instanced TLAS on Vulkan, CLAS/OptiX
//! on NVIDIA). There is no G-buffer to project onto and no second geometry pass
//! to spend. A decal here has to be something a *ray hit* can evaluate, at hit
//! time, from world-space data.
//!
//! # The architecture
//!
//! An **analytic, world-space, spatially-binned decal field**, consulted during
//! material evaluation and blended into albedo/roughness before the BSDF.
//!
//! * **Analytic, not a texture.** Each decal is a rounded-box distance field in
//!   the XZ plane (the object's footprint) plus a vertical band. Coverage is a
//!   closed-form falloff, so the contact band stays crisp at any zoom — a baked
//!   mask texture would need sub-decimetre texels over a 40 km map (~10^10
//!   texels) to do the same, which is not representable.
//! * **World-space, not object-space.** A decal is keyed to a position on the
//!   ground, not to the instance that caused it. This is what makes it survive
//!   TLAS refit: refit rewrites instance transforms and nothing else, and the
//!   decal field is not indexed by instance id, so a refit cannot invalidate it.
//!   It also means the decal lands correctly on *whatever* is underneath — the
//!   terrain, a road, a sidewalk, a plinth top — instead of only on the one mesh
//!   a projector was bound to.
//! * **Binned, not a linear scan.** Decals are bucketed into a uniform XZ grid.
//!   A hit reads one cell range and iterates only the handful of decals that can
//!   possibly reach it, so the per-hit cost is O(decals in cell), not O(scene).
//!
//! # Determinism
//!
//! Replay-exactness is the product moat, so every step here is order-fixed:
//! instances are visited in ascending index order, binning is a counting sort
//! (stable by construction), per-cell overflow keeps the lowest decal indices,
//! and there is no hashing, no RNG and no wall-clock anywhere in the build. The
//! same [`SceneState`] geometry always produces a byte-identical field.
//!
//! # What this mechanism will and will not support later
//!
//! **Will**, with only a new decal *source* (the GPU side is already general):
//! grime streaks under window sills and at wall bases, puddles in road
//! depressions, ground-level scorch/wear patches, and per-lot dirt whose
//! strength the simulation drives (the `strength` field per decal is already in
//! the record and unused at 1.0).
//!
//! **Will not**, without further work: anything that needs a rotated or
//! perspective projection (posters and road markings on a curved or banked
//! surface — the footprint here is an axis-aligned box in XZ, deliberately, so
//! the hit-time test stays a handful of ALU), anything needing its own normal
//! map or albedo *texture* (there is no UV frame here, only coverage), and
//! anything on a vertical surface far from the ground (the vertical band is
//! anchored to the contact height).
//!
//! This module is pure CPU maths with no GPU or game dependency — it builds the
//! buffers, and [`crate::resident_renderer`] uploads them.

/// Floats per packed decal record. Must match `CONTACT_DECAL_FLOATS` in
/// `spectra/slang/contact_decal.slang`.
pub const CONTACT_DECAL_FLOATS: usize = 8;

/// Authored tuning for the contact-decal field.
///
/// Everything here is config-first: it arrives from `render.ron` via
/// `RenderConfig` and is never a hardcoded literal in a render path. The values
/// on [`Default`] mirror the shipped `render.ron` block exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactDecalParams {
    /// Master switch. `false` ⇒ [`ContactDecalField::build`] returns an empty
    /// field ⇒ the kernel gate is 0 ⇒ the frame is byte-identical to the
    /// pre-decal render.
    pub enabled: bool,
    /// Multiplies each object's footprint half-extents. `1.0` puts the contact
    /// line exactly on the object's bounding footprint.
    pub footprint_scale: f32,
    /// Objects shorter than this (world metres) emit no decal. Filters the flat
    /// meshes that ARE the ground — terrain chunks, road ribbons, water planes —
    /// so the ground never dirties itself.
    pub min_object_height_m: f32,
    /// Objects whose footprint exceeds this (world metres, longest XZ side) emit
    /// no decal. Second half of the ground filter: a terrain chunk is both flat
    /// and enormous, and a whole-map mesh must never become one giant decal.
    pub max_footprint_m: f32,
    /// Minimum height / contact-footprint ratio for an object to earn a decal.
    ///
    /// The SCALE-FREE half of the ground filter, and the one that actually
    /// works. A contact decal only means something for an object that STANDS on
    /// a surface, and standing is about proportion, not size: a lamp post is
    /// 0.2 m wide and 5 m tall (ratio 25), a house 12 m wide and 8 m tall (0.67),
    /// but a paving slab, kerb strip or crossing pad is 30 m wide and 1.5 m tall
    /// (0.05) — that is ground furniture, part of the surface rather than
    /// something resting on it.
    ///
    /// Measured: without this gate the wide low slabs in the witness scene
    /// produced 30 m-wide "contact" decals that washed a third of the frame.
    /// The absolute `max_footprint_m` cannot catch them, because a 30 m slab is
    /// legitimately smaller than a building that SHOULD get a decal.
    pub min_aspect: f32,
    /// Height of the slice above a prototype's lowest vertex that defines its
    /// CONTACT FOOTPRINT (object-space metres).
    ///
    /// This is the single most important parameter in the module. The naive
    /// choice — use the prototype's full AABB — is WRONG for anything whose
    /// widest part is not at the bottom, and a tree is the worst case: its AABB
    /// is its CANOPY, so an AABB-derived decal paints dirt under the entire
    /// crown. Measured on the witness scene that turned the feature into a
    /// ground-wide wash covering 47% of the frame instead of a contact band.
    ///
    /// Taking only the vertices within this slice of the base gives the real
    /// contact shape: the TRUNK for a tree, the WALLS (not the roof overhang)
    /// for a building, the plinth for a plinth.
    pub contact_slice_m: f32,
    /// Base outward fade distance (world metres) of the dirt band, before the
    /// object-size term. This is the width of the transition on a lamp post.
    ///
    /// SHARED with the kernel: `contact_decal.slang` computes exactly
    /// `clamp(band_base_m + band_per_size * size_m, band_base_m, band_max_m)`,
    /// and [`max_band_for_size`] reproduces it so binning can never clip a decal
    /// the kernel would still shade. Change the formula in one place only by
    /// changing it in both.
    pub band_base_m: f32,
    /// Extra fade distance per metre of object footprint, so a tower gets a
    /// wider skirt of dirt than a lamp post.
    pub band_per_size: f32,
    /// Ceiling on the fade distance (world metres).
    pub band_max_m: f32,
    /// Grid cell size (world metres). Cost knob: larger cells mean fewer cells
    /// to allocate but more decals to test per hit.
    pub cell_size_m: f32,
    /// Hard ceiling on grid cells. If the scene's decal bounds would need more,
    /// the cell size is doubled until it fits, so a sprawling map degrades to a
    /// coarser grid rather than allocating unboundedly.
    pub max_cells: usize,
    /// Hard ceiling on decals stored per cell. Bounds the kernel's inner loop so
    /// a pathological pile-up cannot stall a wave. Overflow keeps the lowest
    /// decal indices (deterministic) and is reported in
    /// [`ContactDecalField::dropped`].
    pub max_per_cell: usize,
}

impl Default for ContactDecalParams {
    fn default() -> Self {
        Self {
            enabled: true,
            footprint_scale: 1.0,
            min_object_height_m: 1.0,
            max_footprint_m: 90.0,
            min_aspect: 0.35,
            contact_slice_m: 0.5,
            band_base_m: 0.35,
            band_per_size: 0.12,
            band_max_m: 4.0,
            cell_size_m: 12.0,
            max_cells: 1 << 20,
            max_per_cell: 24,
        }
    }
}

/// A built contact-decal field, ready to upload.
///
/// Three flat buffers, mirroring the `g_contact_*` globals in the megakernel:
///
/// * [`Self::decals`] — [`CONTACT_DECAL_FLOATS`] floats per decal:
///   `[cx, base_y, cz, half_x, half_z, size_m, strength, reserved]`.
/// * [`Self::cell_start`] — `res_x * res_z + 1` prefix-sum offsets into
///   [`Self::cell_items`]. Cell `c` owns `cell_items[cell_start[c] ..
///   cell_start[c + 1]]`.
/// * [`Self::cell_items`] — decal indices, grouped by cell, ascending within a
///   cell.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContactDecalField {
    /// Packed decal records, [`CONTACT_DECAL_FLOATS`] floats each.
    pub decals: Vec<f32>,
    /// Prefix-sum cell offsets, length `res[0] * res[1] + 1` (empty when there
    /// are no decals).
    pub cell_start: Vec<u32>,
    /// Decal indices grouped by cell.
    pub cell_items: Vec<u32>,
    /// Grid resolution in cells, `[x, z]`.
    pub res: [u32; 2],
    /// World XZ position of the *lower corner* of cell `(0, 0)`.
    pub origin: [f32; 2],
    /// World metres per cell.
    pub cell_size: f32,
    /// Decal/cell insertions refused by [`ContactDecalParams::max_per_cell`].
    /// Non-zero means the grid is too coarse or the scene too dense; the frame
    /// is still correct, just missing contact dirt in the busiest cells.
    pub dropped: usize,
    /// Prototypes whose contact footprint came from the base-slice scan of their
    /// real geometry — the correct path.
    pub protos_from_geometry: usize,
    /// Prototypes that fell back to the full AABB because the vertex soup or the
    /// prototype range was unusable. NON-ZERO IS A WARNING: the AABB of anything
    /// wider at the top than the bottom (every tree) is the wrong contact shape,
    /// and a fallback silently reintroduces the canopy-wide wash this module
    /// exists to avoid.
    pub protos_from_aabb: usize,
}

impl ContactDecalField {
    /// Number of decals in the field.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decals.len() / CONTACT_DECAL_FLOATS
    }

    /// `true` when the field carries no decals (kernel gate off).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decals.is_empty()
    }

    /// Build the field from the instance columns of a scene's geometry layer.
    ///
    /// * `instance_transforms` — row-major 4×4 per instance (16 floats),
    ///   translation in indices 12/13/14, exactly as `GeometryLayer` stores it.
    /// * `instance_proto_index` — per-instance prototype index; empty ⇒ instance
    ///   `i` uses proto `i` (the legacy single-soup convention).
    /// * `proto_aabbs` — per-prototype object-space `(min, max)`. Empty ⇒ no
    ///   decals can be derived and the field is empty.
    /// * `positions` / `proto_ranges` — the shared vertex soup and each
    ///   prototype's `(v_off, v_cnt, t_off, t_cnt)` sub-range. Used to derive the
    ///   real CONTACT footprint from the geometry at each prototype's base.
    ///   Empty ⇒ fall back to the full AABB, which is correct only for objects
    ///   whose widest part IS at the bottom (see
    ///   [`ContactDecalParams::contact_slice_m`] for why that fallback makes
    ///   trees paint dirt under their whole canopy).
    ///
    /// One decal per qualifying instance, in ascending instance order.
    #[must_use]
    pub fn build(
        instance_transforms: &[f32],
        instance_proto_index: &[u32],
        proto_aabbs: &[([f32; 3], [f32; 3])],
        positions: &[f32],
        proto_ranges: &[(u32, u32, u32, u32)],
        params: &ContactDecalParams,
    ) -> Self {
        if !params.enabled || proto_aabbs.is_empty() || instance_transforms.len() < 16 {
            return Self::default();
        }
        let instance_count = instance_transforms.len() / 16;

        // --- pass 0: per-prototype CONTACT footprint, computed ONCE ---------
        // The object-space box the decal is derived from: the prototype's XZ
        // extent within `contact_slice_m` of its lowest vertex, which is what
        // actually touches the ground. Falls back to the full AABB when the soup
        // is unavailable. Cost is one linear scan of each prototype's vertices
        // at scene build, not per instance.
        let mut protos_from_geometry = 0usize;
        let mut protos_from_aabb = 0usize;
        let contact_boxes: Vec<([f32; 3], [f32; 3])> = (0..proto_aabbs.len())
            .map(|p| {
                let (bx, from_geometry) =
                    contact_box_for_proto(p, proto_aabbs, positions, proto_ranges, params);
                if from_geometry {
                    protos_from_geometry += 1;
                } else {
                    protos_from_aabb += 1;
                }
                bx
            })
            .collect();

        // --- pass 1: one decal per qualifying instance, in instance order ----
        // Ascending index order is the determinism contract: no hashing, no
        // parallel reduction, no iteration over a map.
        let mut decals: Vec<f32> = Vec::new();
        // Parallel to `decals`, in decal order: the XZ half-extent the grid must
        // cover for this decal (footprint + the widest band it can fade over).
        let mut reach: Vec<[f32; 2]> = Vec::new();

        for i in 0..instance_count {
            let m = &instance_transforms[i * 16..i * 16 + 16];
            let proto = instance_proto_index.get(i).copied().unwrap_or(i as u32) as usize;
            let Some(&(lo, hi)) = proto_aabbs.get(proto) else {
                continue;
            };
            // FULL bounds decide whether the object qualifies at all (is it tall
            // enough to stand on the ground? is it small enough not to BE the
            // ground?) — those are questions about the whole object.
            let Some((wmin, wmax)) = world_aabb(m, lo, hi) else {
                continue;
            };
            let height = wmax[1] - wmin[1];
            let full_footprint = (wmax[0] - wmin[0]).max(wmax[2] - wmin[2]);

            // The ground never dirties itself: reject anything flat (road
            // ribbons, water planes) or enormous (terrain chunks). Both tests
            // are purely geometric, so this stays game-agnostic — the engine
            // never learns what a "building" or a "road" is.
            if height < params.min_object_height_m || full_footprint > params.max_footprint_m {
                continue;
            }

            // CONTACT bounds decide the decal's shape. This is the whole point
            // of pass 0: a tree's full AABB is its canopy, and using it here
            // paints dirt under the entire crown instead of a band round the
            // trunk.
            let (clo, chi) = contact_boxes[proto];
            let Some((cmin, cmax)) = world_aabb(m, clo, chi) else {
                continue;
            };
            let ext_x = cmax[0] - cmin[0];
            let ext_z = cmax[2] - cmin[2];
            let footprint = ext_x.max(ext_z);

            // A degenerate footprint has no contact line to dirty.
            if !(ext_x.is_finite() && ext_z.is_finite()) || footprint <= 0.0 {
                continue;
            }

            // ASPECT GATE — does this object STAND on the surface, or is it PART
            // of it? Scale-free, so it separates a 0.2 m lamp post and a 12 m
            // house (which stand) from a 30 m paving slab or kerb strip (which
            // do not) without an arbitrary size threshold that would have to
            // exclude the house too. Applied to the CONTACT footprint, because
            // that is what touches the ground.
            if height < params.min_aspect.max(0.0) * footprint {
                continue;
            }

            let scale = params.footprint_scale.max(0.0);
            let half_x = 0.5 * ext_x * scale;
            let half_z = 0.5 * ext_z * scale;
            // Characteristic size drives the band width in the kernel, so a
            // tower gets a wider skirt of dirt than a lamp post without needing
            // a rebuild when the band config is retuned.
            let size_m = footprint;

            decals.extend_from_slice(&[
                0.5 * (cmin[0] + cmax[0]), // cx — centre of the CONTACT footprint
                cmin[1],                   // base_y — the object's lowest point
                0.5 * (cmin[2] + cmax[2]), // cz
                half_x,
                half_z,
                size_m,
                1.0, // strength — reserved for sim-driven grime
                0.0, // reserved
            ]);
            // Widest band the kernel can ask for is bounded by the size-scaled
            // formula in `contact_decal.slang`; cover generously so binning can
            // never clip a decal the kernel would still shade.
            let band = max_band_for_size(size_m, params);
            reach.push([half_x + band, half_z + band]);
        }

        if decals.is_empty() {
            return Self::default();
        }

        // --- grid bounds --------------------------------------------------
        let mut min_x = f32::INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for (d, r) in decals.chunks_exact(CONTACT_DECAL_FLOATS).zip(&reach) {
            min_x = min_x.min(d[0] - r[0]);
            max_x = max_x.max(d[0] + r[0]);
            min_z = min_z.min(d[2] - r[1]);
            max_z = max_z.max(d[2] + r[1]);
        }
        if !(min_x.is_finite() && min_z.is_finite() && max_x.is_finite() && max_z.is_finite()) {
            return Self::default();
        }

        // Grow the cell until the grid fits the ceiling. Doubling keeps this a
        // handful of iterations and the result independent of iteration order.
        let mut cell = params.cell_size_m.max(0.25);
        let max_cells = params.max_cells.max(1);
        let (mut res_x, mut res_z) = grid_res(min_x, max_x, min_z, max_z, cell);
        while res_x as usize * res_z as usize > max_cells {
            cell *= 2.0;
            let (nx, nz) = grid_res(min_x, max_x, min_z, max_z, cell);
            res_x = nx;
            res_z = nz;
        }
        let cells = res_x as usize * res_z as usize;

        // --- counting sort into the grid (stable, order-fixed) ------------
        let mut counts = vec![0u32; cells];
        let cap = params.max_per_cell.max(1) as u32;
        let mut dropped = 0usize;

        let cell_range = |cx: f32, cz: f32, rx: f32, rz: f32| -> (u32, u32, u32, u32) {
            let x0 = (((cx - rx) - min_x) / cell).floor().max(0.0) as u32;
            let x1 = (((cx + rx) - min_x) / cell).floor().max(0.0) as u32;
            let z0 = (((cz - rz) - min_z) / cell).floor().max(0.0) as u32;
            let z1 = (((cz + rz) - min_z) / cell).floor().max(0.0) as u32;
            (
                x0.min(res_x - 1),
                x1.min(res_x - 1),
                z0.min(res_z - 1),
                z1.min(res_z - 1),
            )
        };

        for (d, r) in decals.chunks_exact(CONTACT_DECAL_FLOATS).zip(&reach) {
            let (x0, x1, z0, z1) = cell_range(d[0], d[2], r[0], r[1]);
            for iz in z0..=z1 {
                for ix in x0..=x1 {
                    let c = iz as usize * res_x as usize + ix as usize;
                    if counts[c] < cap {
                        counts[c] += 1;
                    } else {
                        dropped += 1;
                    }
                }
            }
        }

        let mut cell_start = vec![0u32; cells + 1];
        let mut acc = 0u32;
        for c in 0..cells {
            cell_start[c] = acc;
            acc += counts[c];
        }
        cell_start[cells] = acc;

        let mut cursor = cell_start[..cells].to_vec();
        let mut cell_items = vec![0u32; acc as usize];
        for (idx, (d, r)) in decals
            .chunks_exact(CONTACT_DECAL_FLOATS)
            .zip(&reach)
            .enumerate()
        {
            let (x0, x1, z0, z1) = cell_range(d[0], d[2], r[0], r[1]);
            for iz in z0..=z1 {
                for ix in x0..=x1 {
                    let c = iz as usize * res_x as usize + ix as usize;
                    // Ascending decal index fills each cell, so overflow keeps
                    // the lowest ids — a fixed, replayable choice.
                    if cursor[c] < cell_start[c + 1] {
                        cell_items[cursor[c] as usize] = idx as u32;
                        cursor[c] += 1;
                    }
                }
            }
        }

        Self {
            decals,
            cell_start,
            cell_items,
            res: [res_x, res_z],
            origin: [min_x, min_z],
            cell_size: cell,
            dropped,
            protos_from_geometry,
            protos_from_aabb,
        }
    }
}

/// Outward fade (metres) the kernel will produce for an object of this
/// characteristic size, plus a metre of slack.
///
/// MUST reproduce `contact_band_m()` in `spectra/slang/contact_decal.slang`
/// exactly, or binning would clip decals the kernel still wants to shade — the
/// symptom is contact dirt that pops in and out along cell boundaries. The
/// slack absorbs the float difference between the two evaluations.
fn max_band_for_size(size_m: f32, p: &ContactDecalParams) -> f32 {
    let base = p.band_base_m.max(0.0);
    (base + p.band_per_size.max(0.0) * size_m).clamp(base, p.band_max_m.max(base)) + 1.0
}

/// Object-space box describing what prototype `p` actually puts on the ground.
///
/// Scans the prototype's own vertices, finds the lowest, and takes the XZ extent
/// of everything within `contact_slice_m` of it. That is the trunk of a tree,
/// the walls of a building, the plinth of a plinth — as opposed to the full
/// AABB, which for a tree is the CANOPY and for a building includes the roof
/// overhang.
///
/// Falls back to the full AABB when the vertex soup or the prototype's range is
/// unavailable, or when no vertex lands in the slice.
fn contact_box_for_proto(
    p: usize,
    proto_aabbs: &[([f32; 3], [f32; 3])],
    positions: &[f32],
    proto_ranges: &[(u32, u32, u32, u32)],
    params: &ContactDecalParams,
) -> (([f32; 3], [f32; 3]), bool) {
    let full = proto_aabbs[p];
    let Some(&(v_off, v_cnt, _, _)) = proto_ranges.get(p) else {
        return (full, false);
    };
    let (start, count) = (v_off as usize, v_cnt as usize);
    let Some(end) = start.checked_add(count) else {
        return (full, false);
    };
    if count == 0 || end * 3 > positions.len() {
        return (full, false);
    }

    let mut min_y = f32::INFINITY;
    for v in start..end {
        let y = positions[v * 3 + 1];
        if y.is_finite() && y < min_y {
            min_y = y;
        }
    }
    if !min_y.is_finite() {
        return (full, false);
    }

    let cut = min_y + params.contact_slice_m.max(0.0);
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut found = false;
    for v in start..end {
        let y = positions[v * 3 + 1];
        if !(y.is_finite() && y <= cut) {
            continue;
        }
        let x = positions[v * 3];
        let z = positions[v * 3 + 2];
        if !(x.is_finite() && z.is_finite()) {
            continue;
        }
        lo[0] = lo[0].min(x);
        hi[0] = hi[0].max(x);
        lo[2] = lo[2].min(z);
        hi[2] = hi[2].max(z);
        found = true;
    }
    if !found {
        return (full, false);
    }
    // The box keeps the object's real base height and a thin vertical extent —
    // only the XZ footprint and the base Y are read downstream.
    lo[1] = min_y;
    hi[1] = cut;
    ((lo, hi), true)
}

fn grid_res(min_x: f32, max_x: f32, min_z: f32, max_z: f32, cell: f32) -> (u32, u32) {
    let nx = (((max_x - min_x) / cell).ceil() as i64 + 1).clamp(1, i64::from(u32::MAX >> 1)) as u32;
    let nz = (((max_z - min_z) / cell).ceil() as i64 + 1).clamp(1, i64::from(u32::MAX >> 1)) as u32;
    (nx, nz)
}

/// Transform an object-space AABB by a row-major 4×4 (translation at 12/13/14,
/// the `GeometryLayer::instance_transforms` convention) and return the world
/// AABB. `None` when the matrix produces a non-finite bound.
fn world_aabb(m: &[f32], lo: [f32; 3], hi: [f32; 3]) -> Option<([f32; 3], [f32; 3])> {
    let mut wmin = [f32::INFINITY; 3];
    let mut wmax = [f32::NEG_INFINITY; 3];
    for corner in 0..8u32 {
        let p = [
            if corner & 1 == 0 { lo[0] } else { hi[0] },
            if corner & 2 == 0 { lo[1] } else { hi[1] },
            if corner & 4 == 0 { lo[2] } else { hi[2] },
        ];
        let w = [
            m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
            m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
        ];
        for a in 0..3 {
            if !w[a].is_finite() {
                return None;
            }
            wmin[a] = wmin[a].min(w[a]);
            wmax[a] = wmax[a].max(w[a]);
        }
    }
    Some((wmin, wmax))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity transform, row-major with translation at 12/13/14.
    fn xform(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            tx, ty, tz, 1.0,
        ]
    }

    /// A 2 m × 6 m × 2 m upright box — a tree trunk / lamp post shape.
    fn post_aabb() -> ([f32; 3], [f32; 3]) {
        ([-1.0, 0.0, -1.0], [1.0, 6.0, 1.0])
    }

    #[test]
    fn an_upright_object_emits_one_decal_at_its_base() {
        let t = xform(100.0, 30.0, -50.0);
        let field = ContactDecalField::build(
            &t,
            &[0],
            &[post_aabb()],
            &[],
            &[],
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 1, "one qualifying instance ⇒ one decal");
        let d = &field.decals[..CONTACT_DECAL_FLOATS];
        assert!((d[0] - 100.0).abs() < 1e-4, "cx: {}", d[0]);
        assert!(
            (d[1] - 30.0).abs() < 1e-4,
            "base_y must be the object's LOWEST point, got {}",
            d[1]
        );
        assert!((d[2] + 50.0).abs() < 1e-4, "cz: {}", d[2]);
        assert!((d[3] - 1.0).abs() < 1e-4, "half_x: {}", d[3]);
        assert!((d[4] - 1.0).abs() < 1e-4, "half_z: {}", d[4]);
        assert!((d[5] - 2.0).abs() < 1e-4, "size_m: {}", d[5]);
    }

    #[test]
    fn a_flat_ground_mesh_emits_no_decal_so_the_ground_never_dirties_itself() {
        // A road ribbon: 40 m long, 8 m wide, 5 cm thick.
        let road = ([-20.0, 0.0, -4.0], [20.0, 0.05, 4.0]);
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[road],
            &[],
            &[],
            &ContactDecalParams::default(),
        );
        assert_eq!(
            field.len(),
            0,
            "a 5 cm-thick road must be rejected by min_object_height_m"
        );
    }

    #[test]
    fn an_enormous_mesh_emits_no_decal_so_terrain_never_becomes_one_giant_decal() {
        // A terrain chunk: 512 m across and genuinely tall (relief), so it
        // passes the height test and MUST be caught by the footprint test.
        let chunk = ([-256.0, 0.0, -256.0], [256.0, 90.0, 256.0]);
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[chunk],
            &[],
            &[],
            &ContactDecalParams::default(),
        );
        assert_eq!(
            field.len(),
            0,
            "a 512 m terrain chunk must be rejected by max_footprint_m"
        );
    }

    #[test]
    fn disabled_params_produce_an_empty_field_so_the_kernel_gate_stays_off() {
        let params = ContactDecalParams {
            enabled: false,
            ..ContactDecalParams::default()
        };
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[post_aabb()],
            &[],
            &[],
            &params,
        );
        assert!(field.is_empty());
        assert_eq!(field.res, [0, 0]);
        assert!(field.cell_start.is_empty(), "no grid ⇒ nothing to bind");
    }

    #[test]
    fn the_grid_indexes_every_decal_reachable_from_its_cell() {
        // Three posts 30 m apart on X, cell 12 m ⇒ they land in distinct cells.
        let mut t = Vec::new();
        for i in 0..3 {
            t.extend_from_slice(&xform(i as f32 * 30.0, 0.0, 0.0));
        }
        let field = ContactDecalField::build(
            &t,
            &[0, 0, 0],
            &[post_aabb()],
            &[],
            &[],
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 3);
        assert_eq!(
            field.cell_start.len(),
            field.res[0] as usize * field.res[1] as usize + 1
        );

        // Every decal must be findable from the cell containing its own centre —
        // that is exactly the lookup the kernel performs.
        for i in 0..field.len() {
            let d = &field.decals[i * CONTACT_DECAL_FLOATS..];
            let ix = ((d[0] - field.origin[0]) / field.cell_size).floor() as usize;
            let iz = ((d[2] - field.origin[1]) / field.cell_size).floor() as usize;
            let c = iz * field.res[0] as usize + ix;
            let lo = field.cell_start[c] as usize;
            let hi = field.cell_start[c + 1] as usize;
            assert!(
                field.cell_items[lo..hi].contains(&(i as u32)),
                "decal {i} at ({}, {}) is not listed in its own cell {c}",
                d[0],
                d[2]
            );
        }
    }

    #[test]
    fn a_decal_is_listed_in_every_cell_its_band_reaches() {
        // One decal, cell size deliberately smaller than the decal's reach, so
        // it must appear in a block of cells rather than only its centre cell.
        let params = ContactDecalParams {
            cell_size_m: 1.0,
            ..ContactDecalParams::default()
        };
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[post_aabb()],
            &[],
            &[],
            &params,
        );
        assert_eq!(field.len(), 1);
        let listed = field.cell_items.iter().filter(|&&i| i == 0).count();
        assert!(
            listed > 1,
            "a decal whose reach spans many 1 m cells must be binned into all \
             of them, else the kernel misses it; listed in {listed} cell(s)"
        );

        // THE contract the kernel depends on: sweep world points densely and
        // check that any point inside the decal's reach finds it in the cell
        // the kernel would look up. A miss here is a decal that silently
        // vanishes for some pixels.
        let d = &field.decals[..CONTACT_DECAL_FLOATS];
        let band = max_band_for_size(d[5], &params);
        let reach = [d[3] + band, d[4] + band];
        let mut checked = 0;
        let mut steps = -60i32;
        while steps <= 60 {
            let mut stepz = -60i32;
            while stepz <= 60 {
                let px = d[0] + steps as f32 * 0.05;
                let pz = d[2] + stepz as f32 * 0.05;
                stepz += 1;
                if (px - d[0]).abs() > reach[0] || (pz - d[2]).abs() > reach[1] {
                    continue;
                }
                let ix = ((px - field.origin[0]) / field.cell_size).floor();
                let iz = ((pz - field.origin[1]) / field.cell_size).floor();
                assert!(
                    ix >= 0.0 && iz >= 0.0,
                    "point ({px}, {pz}) fell outside the grid origin"
                );
                let (ix, iz) = (ix as usize, iz as usize);
                assert!(
                    ix < field.res[0] as usize && iz < field.res[1] as usize,
                    "point ({px}, {pz}) fell outside the grid extent"
                );
                let c = iz * field.res[0] as usize + ix;
                let lo = field.cell_start[c] as usize;
                let hi = field.cell_start[c + 1] as usize;
                assert!(
                    field.cell_items[lo..hi].contains(&0),
                    "point ({px}, {pz}) is inside the decal's reach but its cell \
                     {c} does not list the decal — the kernel would miss it"
                );
                checked += 1;
            }
            steps += 1;
        }
        assert!(checked > 1000, "sweep covered only {checked} points");
    }

    #[test]
    fn the_build_is_byte_identical_across_runs_determinism_is_law() {
        let mut t = Vec::new();
        for i in 0..64 {
            let f = i as f32;
            t.extend_from_slice(&xform(f * 7.3, f * 0.5, -f * 3.1));
        }
        let protos: Vec<u32> = (0..64).map(|i| i % 3).collect();
        let aabbs = vec![
            post_aabb(),
            ([-3.0, 0.0, -4.0], [3.0, 12.0, 4.0]),
            ([-0.4, 0.0, -0.4], [0.4, 4.0, 0.4]),
        ];
        let p = ContactDecalParams::default();
        let a = ContactDecalField::build(&t, &protos, &aabbs, &[], &[], &p);
        let b = ContactDecalField::build(&t, &protos, &aabbs, &[], &[], &p);
        assert_eq!(a, b, "identical input must give a byte-identical field");
        assert!(a.len() > 0, "the fixture must actually produce decals");
    }

    #[test]
    fn per_cell_overflow_is_capped_and_reported_so_the_kernel_loop_is_bounded() {
        // 40 posts stacked on the same spot, cap 8.
        let mut t = Vec::new();
        for _ in 0..40 {
            t.extend_from_slice(&xform(0.0, 0.0, 0.0));
        }
        let protos = vec![0u32; 40];
        let params = ContactDecalParams {
            max_per_cell: 8,
            ..ContactDecalParams::default()
        };
        let field = ContactDecalField::build(&t, &protos, &[post_aabb()], &[], &[], &params);
        assert_eq!(field.len(), 40, "every decal is still stored");
        for c in 0..field.res[0] as usize * field.res[1] as usize {
            let n = field.cell_start[c + 1] - field.cell_start[c];
            assert!(n <= 8, "cell {c} holds {n} decals, cap is 8");
        }
        assert!(
            field.dropped > 0,
            "overflow must be reported, not silently swallowed"
        );
    }

    /// REGRESSION. The first version of this module used each prototype's full
    /// AABB as the contact footprint. For a tree the AABB is the CANOPY, so the
    /// decal painted dirt under the entire crown — measured on the witness scene
    /// as a ground-wide wash touching 47% of the frame instead of a band round
    /// the trunk. The footprint must come from the geometry AT THE BASE.
    #[test]
    fn a_tree_gets_a_trunk_sized_footprint_not_a_canopy_sized_one() {
        // A 0.6 m-wide trunk from y=0 to y=3, then a 9 m-wide canopy above it.
        // The full AABB is 9 m across; the CONTACT footprint must be 0.6 m.
        let mut positions: Vec<f32> = Vec::new();
        for (x, y, z) in [
            (-0.3f32, 0.0f32, -0.3f32),
            (0.3, 0.0, 0.3),
            (-0.3, 3.0, -0.3),
            (0.3, 3.0, 0.3),
            (-4.5, 4.0, -4.5), // canopy
            (4.5, 4.0, 4.5),
            (-4.5, 7.0, -4.5),
            (4.5, 7.0, 4.5),
        ] {
            positions.extend_from_slice(&[x, y, z]);
        }
        let aabb = ([-4.5, 0.0, -4.5], [4.5, 7.0, 4.5]);
        let ranges = vec![(0u32, 8u32, 0u32, 0u32)];

        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[aabb],
            &positions,
            &ranges,
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 1);
        let half_x = field.decals[3];
        let half_z = field.decals[4];
        assert!(
            (half_x - 0.3).abs() < 1e-4 && (half_z - 0.3).abs() < 1e-4,
            "contact footprint must be the TRUNK (0.3 m half-extent), not the \
             canopy (4.5 m); got half_x={half_x} half_z={half_z}"
        );
        assert!(
            (field.decals[5] - 0.6).abs() < 1e-4,
            "size_m must be the trunk width, got {}",
            field.decals[5]
        );

        // And the tall canopy must still qualify the object: the HEIGHT test
        // reads the full AABB, only the FOOTPRINT comes from the base slice.
        assert!(field.len() == 1, "the tree must still emit a decal");
    }

    #[test]
    fn a_building_footprint_comes_from_its_walls_not_its_roof_overhang() {
        // Walls 8 m square to y=6; a roof overhanging to 11 m square above.
        let mut positions: Vec<f32> = Vec::new();
        for (x, y, z) in [
            (-4.0f32, 0.0f32, -4.0f32),
            (4.0, 0.0, 4.0),
            (-4.0, 6.0, -4.0),
            (4.0, 6.0, 4.0),
            (-5.5, 6.5, -5.5),
            (5.5, 6.5, 5.5),
        ] {
            positions.extend_from_slice(&[x, y, z]);
        }
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[([-5.5, 0.0, -5.5], [5.5, 6.5, 5.5])],
            &positions,
            &[(0, 6, 0, 0)],
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 1);
        assert!(
            (field.decals[3] - 4.0).abs() < 1e-4,
            "contact line must sit on the WALL (4 m), not the roof edge (5.5 m); \
             got {}",
            field.decals[3]
        );
    }

    /// REGRESSION. Wide, low ground furniture (paving slabs, kerb strips,
    /// crossing pads) passed both the height and the absolute-footprint filters
    /// and produced 30 m-wide "contact" decals that washed a third of the
    /// witness frame. Absolute size cannot separate them from a building — only
    /// the height/footprint PROPORTION can.
    #[test]
    fn a_wide_low_slab_is_rejected_but_a_building_of_similar_width_is_not() {
        let params = ContactDecalParams::default();

        // A 30 m x 1.5 m paving slab: tall enough for min_object_height_m,
        // smaller than max_footprint_m, but flat in proportion.
        let slab = ([-15.0, 0.0, -15.0], [15.0, 1.5, 15.0]);
        let f = ContactDecalField::build(&xform(0.0, 0.0, 0.0), &[0], &[slab], &[], &[], &params);
        assert_eq!(
            f.len(),
            0,
            "a 30 m x 1.5 m slab is ground furniture and must not get a decal"
        );

        // A building of comparable width but real height MUST still qualify.
        let building = ([-15.0, 0.0, -15.0], [15.0, 18.0, 15.0]);
        let f =
            ContactDecalField::build(&xform(0.0, 0.0, 0.0), &[0], &[building], &[], &[], &params);
        assert_eq!(
            f.len(),
            1,
            "a 30 m-wide, 18 m-tall building stands on the ground and must keep \
             its decal — the gate is PROPORTION, not size"
        );
    }

    #[test]
    fn a_missing_vertex_soup_falls_back_to_the_full_aabb() {
        // Empty positions/ranges must not panic or drop the decal — the caller
        // may legitimately have no soup (legacy single-proto scenes).
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[post_aabb()],
            &[],
            &[],
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 1);
        assert!((field.decals[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn an_out_of_range_proto_range_falls_back_instead_of_reading_out_of_bounds() {
        // A range that overruns the soup must be refused, not indexed.
        let field = ContactDecalField::build(
            &xform(0.0, 0.0, 0.0),
            &[0],
            &[post_aabb()],
            &[0.0; 9], // 3 vertices
            &[(0, 999, 0, 0)],
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 1, "must fall back, not panic or drop");
        assert!((field.decals[3] - 1.0).abs() < 1e-4, "full-AABB fallback");
    }

    #[test]
    fn a_rotated_instance_gets_the_world_footprint_not_the_object_one() {
        // 45° yaw of a 2x2 footprint ⇒ world footprint widens to 2*sqrt(2).
        let c = std::f32::consts::FRAC_1_SQRT_2;
        let m = [
            c, 0.0, -c, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            c, 0.0, c, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let field = ContactDecalField::build(
            &m,
            &[0],
            &[post_aabb()],
            &[],
            &[],
            &ContactDecalParams::default(),
        );
        assert_eq!(field.len(), 1);
        let half_x = field.decals[3];
        assert!(
            (half_x - std::f32::consts::SQRT_2).abs() < 1e-3,
            "rotated half-extent should be sqrt(2), got {half_x}"
        );
    }

    #[test]
    fn a_coarse_grid_is_forced_when_the_scene_would_blow_the_cell_ceiling() {
        // Two posts 20 km apart with a 1 m cell would need 4e8 cells.
        let mut t = xform(0.0, 0.0, 0.0).to_vec();
        t.extend_from_slice(&xform(20_000.0, 0.0, 20_000.0));
        let params = ContactDecalParams {
            cell_size_m: 1.0,
            max_cells: 4096,
            ..ContactDecalParams::default()
        };
        let field = ContactDecalField::build(&t, &[0, 0], &[post_aabb()], &[], &[], &params);
        assert_eq!(field.len(), 2);
        let cells = field.res[0] as usize * field.res[1] as usize;
        assert!(cells <= 4096, "cell ceiling not honoured: {cells} cells");
        assert!(
            field.cell_size > 1.0,
            "cell size must have grown, got {}",
            field.cell_size
        );
    }
}
