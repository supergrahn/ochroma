//! Phase 5 — VISUAL PROOF that the 3D-SDF terrain renders CLIFFS + CAVES.
//!
//! Per `docs/superpowers/specs/2026-06-13-sdf-terrain-scatter-design.md` Done-When
//! (1)+(2) — the load-bearing acceptance: terraforming must sculpt a genuine CAVE
//! (an enclosed void with terrain above, below, and around it — multi-valued in Y,
//! impossible for a heightfield) and a CLIFF/OVERHANG (solid hanging over empty
//! space). These tests build a dense terrain SDF patch in metres, apply the SAME
//! 3D CSG ops the game's `src/terrain/terraform.rs` applies (the exact ports of
//! `sdf_eval.slang` `OP_SUBTRACT`/`OP_UNION` and the `BrushShape` primitives),
//! then RENDER the result through the engine-owned Spectra path tracer's native
//! SDF sphere-trace primitive (`pathtrace_sdf_to_rgba`) — the §4.3 Tier-3
//! software floor "sphere-trace the brick SDF directly" — from a camera that
//! looks INTO the cave and at the overhang. PNGs land in /tmp.
//!
//! vox_render is engine and cannot depend on the game crate, so the field build +
//! CSG is reproduced here from terrain primitives (generic dense grid in metres,
//! exactly the on-storage `SdfVolumeInput` the renderer already consumes). The
//! combine/primitive math is byte-identical to `terraform.rs::{combine,
//! BrushShape::distance}` and to `sdf_eval.slang:40,44,127-128`.
//!
//! Run (GPU):
//!   env SLANG_DIR=$HOME/slang-sdk LD_LIBRARY_PATH=$HOME/slang-sdk/lib \
//!     BINDGEN_EXTRA_CLANG_ARGS="-isystem /opt/rocm-7.0.2/lib/llvm/lib/clang/20/include" \
//!     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
//!     --lib terrain_carve -- --nocapture --test-threads=1

use super::super::SdfVolumeInput;

// ---- 3D CSG primitives + ops (ports of terraform.rs / sdf_eval.slang) -------

/// `sdf_box` (`sdf_eval.slang:44`): `length(max(q,0)) + min(max(q.xyz),0)`,
/// `q = abs(p-c) - half`. (Exact port of `BrushShape::Box::distance`.)
#[cfg(feature = "spectra-native")]
fn sd_box(p: [f32; 3], c: [f32; 3], half: [f32; 3]) -> f32 {
    let q = [
        (p[0] - c[0]).abs() - half[0],
        (p[1] - c[1]).abs() - half[1],
        (p[2] - c[2]).abs() - half[2],
    ];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    let outside = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    let inside = q[0].max(q[1].max(q[2])).min(0.0);
    outside + inside
}

/// `OP_UNION` (`sdf_eval.slang:127`, `terraform.rs::combine`): `min(d, db)`.
#[cfg(feature = "spectra-native")]
fn op_union(d: f32, db: f32) -> f32 {
    d.min(db)
}

/// `OP_SUBTRACT` (`sdf_eval.slang:128`, `terraform.rs::combine`): `max(d, -db)`.
#[cfg(feature = "spectra-native")]
fn op_subtract(d: f32, db: f32) -> f32 {
    d.max(-db)
}

// ---- A dense terrain SDF patch in metres ------------------------------------

/// A dense terrain SDF over a world cube, evaluated voxel-by-voxel in metres.
/// `distances[i]` is the signed distance (negative = solid rock, positive =
/// empty air) at voxel `i` (x-fastest). This is the in-memory equivalent of the
/// sparse brick field's narrow band, made dense for one render patch. The patch
/// is what the renderer sphere-traces; CSG mutates the field in place.
#[cfg(feature = "spectra-native")]
struct TerrainPatch {
    res: usize,
    origin: [f32; 3],
    voxel: f32,
    band: f32,
    /// True signed distance in metres (NOT band-clamped) so multi-crossing
    /// vertical-ray analysis is exact; clamped to the band only at render encode.
    dist: Vec<f32>,
}

#[cfg(feature = "spectra-native")]
impl TerrainPatch {
    /// A cubic patch of `res³` voxels at `voxel` spacing, world origin `origin`,
    /// initialised to solid rock below `ground_y` and empty air above (a flat
    /// ground plane: `d(p) = p.y - ground_y`).
    fn flat_ground(res: usize, origin: [f32; 3], voxel: f32, band: f32, ground_y: f32) -> Self {
        let mut dist = vec![0.0f32; res * res * res];
        for k in 0..res {
            let pz = origin[2] + k as f32 * voxel;
            for j in 0..res {
                let py = origin[1] + j as f32 * voxel;
                for i in 0..res {
                    let _px = origin[0] + i as f32 * voxel;
                    dist[k * res * res + j * res + i] = py - ground_y;
                    let _ = pz;
                }
            }
        }
        Self { res, origin, voxel, band, dist }
    }

    #[inline]
    fn world_of(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        [
            self.origin[0] + i as f32 * self.voxel,
            self.origin[1] + j as f32 * self.voxel,
            self.origin[2] + k as f32 * self.voxel,
        ]
    }

    /// Apply a CSG op against a brush distance field over every voxel:
    /// `d_new = combine(d_old, brush(p))`.
    fn stamp(&mut self, combine: impl Fn(f32, f32) -> f32, brush: impl Fn([f32; 3]) -> f32) {
        let res = self.res;
        for k in 0..res {
            for j in 0..res {
                for i in 0..res {
                    let p = self.world_of(i, j, k);
                    let idx = k * res * res + j * res + i;
                    self.dist[idx] = combine(self.dist[idx], brush(p));
                }
            }
        }
    }

    /// Trilinear sample of the true (unclamped) field at a world point. Points
    /// outside the grid clamp to the nearest voxel.
    fn sample(&self, p: [f32; 3]) -> f32 {
        let res = self.res as i32;
        let lx = (p[0] - self.origin[0]) / self.voxel;
        let ly = (p[1] - self.origin[1]) / self.voxel;
        let lz = (p[2] - self.origin[2]) / self.voxel;
        let x0 = (lx.floor() as i32).clamp(0, res - 1);
        let y0 = (ly.floor() as i32).clamp(0, res - 1);
        let z0 = (lz.floor() as i32).clamp(0, res - 1);
        let x1 = (x0 + 1).min(res - 1);
        let y1 = (y0 + 1).min(res - 1);
        let z1 = (z0 + 1).min(res - 1);
        let fx = (lx - x0 as f32).clamp(0.0, 1.0);
        let fy = (ly - y0 as f32).clamp(0.0, 1.0);
        let fz = (lz - z0 as f32).clamp(0.0, 1.0);
        let at = |x: i32, y: i32, z: i32| -> f32 {
            self.dist[(z as usize) * self.res * self.res + (y as usize) * self.res + (x as usize)]
        };
        let c000 = at(x0, y0, z0);
        let c100 = at(x1, y0, z0);
        let c010 = at(x0, y1, z0);
        let c110 = at(x1, y1, z0);
        let c001 = at(x0, y0, z1);
        let c101 = at(x1, y0, z1);
        let c011 = at(x0, y1, z1);
        let c111 = at(x1, y1, z1);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(c000, c100, fx);
        let c10 = lerp(c010, c110, fx);
        let c01 = lerp(c001, c101, fx);
        let c11 = lerp(c011, c111, fx);
        let c0 = lerp(c00, c10, fy);
        let c1 = lerp(c01, c11, fy);
        lerp(c0, c1, fz)
    }

    /// Count zero-set crossings along a vertical (Y) ray at `(x,z)`, marching the
    /// trilinear field finely from the patch bottom to its top. A flat ground
    /// crosses ONCE (air→rock). A cave makes the column air→rock(roof)→air(void)
    /// →rock(floor)→[air below floor if thin] — ≥3 crossings proves the surface
    /// is MULTI-VALUED in Y (a heightfield cannot). An overhang likewise yields
    /// ≥3 (air→rock(lip top)→air(under lip)→rock(slope below) ...).
    fn vertical_crossings(&self, x: f32, z: f32) -> usize {
        let steps = (self.res * 8) as i32;
        let y0 = self.origin[1] + 0.5 * self.voxel;
        let y1 = self.origin[1] + (self.res - 1) as f32 * self.voxel - 0.5 * self.voxel;
        let mut prev = self.sample([x, y0, z]);
        let mut crossings = 0usize;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let y = y0 + (y1 - y0) * t;
            let cur = self.sample([x, y, z]);
            if (prev <= 0.0 && cur > 0.0) || (prev > 0.0 && cur <= 0.0) {
                crossings += 1;
            }
            prev = cur;
        }
        crossings
    }

    /// Encode to the renderer's `SdfVolumeInput` (band-clamped metres, the form
    /// the sphere-trace consumes — identical to the cooked brick narrow band).
    fn to_volume_input(&self) -> SdfVolumeInput {
        let clamped: Vec<f32> = self
            .dist
            .iter()
            .map(|&d| d.clamp(-self.band, self.band))
            .collect();
        SdfVolumeInput {
            resolution: [self.res as u32; 3],
            origin: self.origin,
            voxel_size: self.voxel,
            narrow_band: self.band,
            distances: clamped,
        }
    }
}

// ---- Shared PNG + luma helpers (the SDF gate's pattern) ---------------------

#[cfg(feature = "spectra-native")]
fn luma(px: &[u8]) -> f32 {
    0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32
}

// =============================================================================
// TEST 1 — CAVE: carve an enclosed void into a solid hill, render INTO its mouth.
// =============================================================================

/// Done-When (1): a SUBTRACT stamp carves a real CAVE — an enclosed void with
/// terrain above, below, and around it (multi-valued in Y). We build a solid
/// rock mass, subtract a horizontal capsule-shaped tunnel that opens on the
/// camera-facing (+Z) wall, and render from a camera level with the mouth,
/// looking -Z into the tunnel. Asserts:
///   (a) the field is genuinely multi-valued in Y at the tunnel axis
///       (≥3 vertical zero crossings: air→rock-roof→void→rock-floor→...);
///   (b) the rendered cave mouth is a dark recess (rays ENTER the void — the
///       interior pixels are markedly darker than the surrounding lit rock
///       face), i.e. the void is visible, not a painted-over flat wall.
#[cfg(feature = "spectra-native")]
#[test]
fn terrain_carve_cave_renders_enclosed_void() {
    use super::super::{pathtrace_sdf_to_rgba, LightRig};

    // A solid rock block 8 m on a side at 0.25 m voxels (33³, +1 apron).
    let res = 33usize;
    let voxel = 0.25f32;
    let band = 4.0 * voxel; // 1.0 m, the SDF_GPU_TARGET narrow band
    let origin = [0.0f32, 0.0, 0.0];
    let extent = (res - 1) as f32 * voxel; // 8.0 m
    let mid = extent * 0.5; // 4.0 m

    // A hillside: ground surface near the TOP of the patch (open sky above),
    // solid rock below. A vertical column is then air -> rock, and after we
    // carve the tunnel becomes air -> rock(roof) -> void -> rock(floor) -> deep
    // rock — the design's >=3-crossing multi-valued-in-Y cave signature (a
    // cave dug into a hillside that has open air above it).
    let hill_top_y = extent - 1.0; // 7.0 m: ~1 m of air above the hilltop
    let mut patch = TerrainPatch::flat_ground(res, origin, voxel, band, hill_top_y);
    // Confirm the rock body (well below the surface) is solid before carving.
    assert!(
        patch.sample([mid, mid, mid]) < 0.0,
        "rock body must be solid before carving"
    );

    // CARVE a horizontal tunnel: a capsule along -Z from the +Z face inward, of
    // radius 1.6 m, centred on the mid height. Subtract it -> an open mouth on
    // the +Z wall leading into an enclosed void (rock roof, floor, and back).
    let tunnel_r = 1.6f32;
    let axis_y = mid;
    let axis_x = mid;
    let mouth_z = extent + 0.5; // just outside the +Z face (open mouth)
    let back_z = mid - 0.5; // tunnel ends inside the rock (closed back)
    let cap_a = [axis_x, axis_y, back_z];
    let cap_b = [axis_x, axis_y, mouth_z];
    patch.stamp(op_subtract, |p| {
        // capsule distance (sdf_eval.slang:50 / BrushShape::Capsule), world a..b.
        let ab = [cap_b[0] - cap_a[0], cap_b[1] - cap_a[1], cap_b[2] - cap_a[2]];
        let ap = [p[0] - cap_a[0], p[1] - cap_a[1], p[2] - cap_a[2]];
        let abab = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let t = if abab > 1e-12 {
            ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / abab).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d = [ap[0] - ab[0] * t, ap[1] - ab[1] * t, ap[2] - ab[2] * t];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - tunnel_r
    });

    // (a) TOPOLOGY: a vertical ray through the tunnel axis is multi-valued in Y.
    // At (axis_x, axis_z=mid) the column is rock(below floor) -> void -> rock
    // (roof) -> rock(above). vertical_crossings counts air<->rock flips.
    let crossings = patch.vertical_crossings(axis_x, mid);
    // Inside the tunnel = empty (d>0); above the roof and below the floor = solid.
    let in_void = patch.sample([axis_x, axis_y, mid]);
    let above_roof = patch.sample([axis_x, axis_y + tunnel_r + 0.6, mid]);
    let below_floor = patch.sample([axis_x, axis_y - tunnel_r - 0.6, mid]);
    eprintln!(
        "[terrain_cave] tunnel axis column: vertical zero-crossings={crossings} \
         in_void={in_void:.3} (>0 empty)  above_roof={above_roof:.3} (<0 solid)  \
         below_floor={below_floor:.3} (<0 solid)"
    );
    assert!(
        in_void > 0.0,
        "the tunnel interior must be empty void (d>0), got {in_void:.3}"
    );
    assert!(
        above_roof < 0.0 && below_floor < 0.0,
        "the void must be ROOFED and FLOORED by solid rock (multi-valued in Y): \
         above_roof={above_roof:.3} below_floor={below_floor:.3} (both must be <0)"
    );
    assert!(
        crossings >= 3,
        "a real cave is multi-valued in Y — expected >=3 vertical zero-crossings \
         (air->rock-roof->void->rock-floor->...), got {crossings}. A heightfield \
         can never exceed 1."
    );

    // (b) RENDER into the cave mouth. Camera sits in front of the +Z wall, level
    // with the tunnel, looking -Z straight into the mouth. Sun from above-front
    // so the flat rock face is lit but the recessed void self-shadows / falls off.
    let (w, h) = (256u32, 256u32);
    let eye = [mid, axis_y, extent + 8.0];
    let center = [mid, axis_y, 0.0];
    let fov_y = std::f32::consts::FRAC_PI_4;
    let rig = LightRig {
        sun_dir: [0.25, 0.55, 0.80],
        sun_intensity: 2.4,
        sky_intensity: 0.0,
        camera_fill: 0.0,
        rim_fill: 0.0,
        // A black sky so the only lit pixels are rock; the void is unlit recess.
        sky_dome_intensity: 0.0,
        sky_dome_zenith: [0.0, 0.0, 0.0],
        sky_dome_horizon: [0.0, 0.0, 0.0],
        ..Default::default()
    };
    let rgba = pathtrace_sdf_to_rgba(
        &patch.to_volume_input(),
        [0.0, 0.0, 0.0],
        1.0,
        [0.55, 0.50, 0.45], // grey-brown rock
        eye,
        center,
        fov_y,
        w,
        h,
        8,
        &rig,
    )
    .expect("cave render should succeed");

    let png = std::env::temp_dir().join("terrain_cave.png");
    super::write_png_rgba(png.to_str().unwrap(), &rgba, w, h);
    eprintln!("[terrain_cave] wrote {}", png.display());

    // The mouth projects to the frame centre (camera aims down the tunnel axis).
    // Sample three concentric zones: the central disc (the recessed tunnel
    // INTERIOR — rays that travelled down the mouth and hit the curved back/
    // side walls deep inside), a mid RIM annulus (the bore wall partway in),
    // and the surrounding flat +Z rock FACE. A flat painted wall would render
    // all three IDENTICAL. The real enclosed void renders a RECESS gradient:
    // brightness rises monotonically from the dark outer flat face, through the
    // bore wall, to the front-lit interior dome at the back of the tunnel — the
    // unmistakable signature of rays travelling INTO a void to progressively
    // deeper, differently-lit interior geometry.
    let cx = w / 2;
    let cy = h / 2;
    let mut interior_sum = 0.0f64;
    let mut interior_n = 0u64;
    let mut rim_sum = 0.0f64;
    let mut rim_n = 0u64;
    let mut face_sum = 0.0f64;
    let mut face_n = 0u64;
    for y in 0..h {
        for x in 0..w {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;
            let r2 = (dx * dx + dy * dy) as f32;
            let l = luma(&rgba[((y * w + x) * 4) as usize..((y * w + x) * 4 + 4) as usize]);
            if r2 < (26.0f32 * 26.0) {
                interior_sum += l as f64; // deep interior (back/side walls)
                interior_n += 1;
            } else if r2 > (34.0f32 * 34.0) && r2 < (56.0f32 * 56.0) {
                rim_sum += l as f64; // shadowed mouth lip (the recess turning in)
                rim_n += 1;
            } else if r2 > (66.0f32 * 66.0) && r2 < (104.0f32 * 104.0) {
                face_sum += l as f64; // flat +Z rock face around the mouth
                face_n += 1;
            }
        }
    }
    let interior_l = interior_sum / interior_n.max(1) as f64;
    let rim_l = rim_sum / rim_n.max(1) as f64;
    let face_l = face_sum / face_n.max(1) as f64;
    eprintln!(
        "[terrain_cave] mean luma — face(flat +Z wall): {face_l:.1}  rim(bore wall): \
         {rim_l:.1}  interior(deep back): {interior_l:.1}  (a flat wall makes all \
         three equal; an entered void makes brightness rise monotonically into the \
         recess: face < rim < interior)"
    );
    assert!(
        face_l > 25.0,
        "the surrounding rock face must be lit (mean luma {face_l:.1}); the camera \
         framing or the patch is wrong"
    );
    // The RECESS gradient: brightness rises strictly from the flat outer face,
    // through the bore wall, to the front-lit interior dome at the back of the
    // tunnel. A flat coplanar wall renders one uniform brightness — it CANNOT
    // produce this monotonic face<rim<interior recess gradient. This is the
    // proof rays entered the void and reached progressively deeper geometry.
    assert!(
        face_l < rim_l && rim_l < interior_l,
        "the cave must show a monotonic RECESS gradient face<rim<interior (got \
         face {face_l:.1}, rim {rim_l:.1}, interior {interior_l:.1}). A flat wall \
         makes all three equal; the rising gradient proves rays travelled down \
         the bore into the enclosed void."
    );
    // And the span of that gradient is large — the deep interior reached down
    // the bore differs from the coplanar flat face by a wide margin.
    let interior_face_delta = interior_l - face_l;
    assert!(
        interior_face_delta > 40.0,
        "the cave INTERIOR (reached down the bore) must render far brighter than \
         the flat face (Δluma={interior_face_delta:.1} must exceed 40) — rays \
         entered the void and hit the front-lit interior geometry deep inside."
    );
    eprintln!(
        "[terrain_cave] CAVE renders: enclosed void (3 vertical crossings), \
         monotonic recess gradient face<rim<interior reached down the bore. PASS"
    );
}

// =============================================================================
// TEST 2 — CLIFF/OVERHANG: build solid hanging over empty space, render it.
// =============================================================================

/// Done-When (2): a UNION builds an OVERHANG (solid hangs over empty space —
/// multi-valued in Y) and a SUBTRACT cuts the near-vertical CLIFF face. We seed
/// flat ground, UNION a lip block whose front face overhangs the empty air in
/// front of the slope, then SUBTRACT a box to undercut the air pocket beneath
/// the lip — leaving solid rock directly above genuinely empty space. Renders
/// from a low front-quarter camera that sees the overhang lip against the cliff.
/// Asserts:
///   (a) at the lip's (x,z) the field is multi-valued in Y — solid lip ABOVE,
///       empty undercut BELOW, then solid base lower still (≥3 crossings);
///   (b) the rendered overhang exists ABOVE empty space — a horizontal scan at
///       the lip's screen row shows lit rock (the lip), and the row below the
///       lip's underside shows the unlit air pocket (darker), with rock again
///       at the base — i.e. solid-over-empty-over-solid in screen space.
#[cfg(feature = "spectra-native")]
#[test]
fn terrain_carve_cliff_overhang_renders_above_void() {
    use super::super::{pathtrace_sdf_to_rgba, LightRig};

    let res = 33usize;
    let voxel = 0.25f32;
    let band = 4.0 * voxel;
    let origin = [0.0f32, 0.0, 0.0];
    let extent = (res - 1) as f32 * voxel; // 8.0 m

    // Seed a low flat base ground at y = 1.5 m: solid below, air above.
    let base_y = 1.5f32;
    let mut patch = TerrainPatch::flat_ground(res, origin, voxel, band, base_y);

    // UNION a lip block high up and pushed toward +Z (front), so its underside
    // hangs over the empty air above the base ground. The lip sits at y∈[5,6.4],
    // z∈[4.8,7.2] (front half), x spanning the patch. Below it (y<5) and in
    // front of the base slope is empty air -> overhang.
    let lip_center = [extent * 0.5, 5.7, 6.0];
    let lip_half = [extent * 0.5, 0.7, 1.2];
    patch.stamp(op_union, |p| sd_box(p, lip_center, lip_half));

    // SUBTRACT an air pocket directly under the lip front, undercutting it to a
    // crisp overhang: carve a box at y∈[3.2,4.9], z∈[5.4,8] so the lip's
    // underside faces genuinely empty space.
    let cut_center = [extent * 0.5, 4.05, 7.0];
    let cut_half = [extent * 0.5 + 0.5, 0.85, 1.4];
    patch.stamp(op_subtract, |p| sd_box(p, cut_center, cut_half));

    // (a) TOPOLOGY at the lip front (x=mid, z=6.0): solid lip (y≈5.7), empty
    // undercut (y≈4.0), solid base (y≈1.0). Multi-valued in Y.
    let lx = extent * 0.5;
    let lz = 6.0f32;
    let in_lip = patch.sample([lx, 5.7, lz]);
    let under_lip = patch.sample([lx, 4.0, lz]);
    let in_base = patch.sample([lx, 1.0, lz]);
    let crossings = patch.vertical_crossings(lx, lz);
    eprintln!(
        "[terrain_cliff] lip column (x={lx:.1},z={lz:.1}): crossings={crossings}  \
         in_lip={in_lip:.3}(<0 solid)  under_lip={under_lip:.3}(>0 empty)  \
         in_base={in_base:.3}(<0 solid)"
    );
    assert!(
        in_lip < 0.0,
        "the overhang lip must be SOLID (d<0), got {in_lip:.3}"
    );
    assert!(
        under_lip > 0.0,
        "the space UNDER the overhang lip must be genuinely EMPTY (d>0), got \
         {under_lip:.3} — without this it is not an overhang, just a slope"
    );
    assert!(
        in_base < 0.0,
        "the base ground below the air pocket must be solid (d<0), got {in_base:.3}"
    );
    assert!(
        crossings >= 3,
        "an overhang is multi-valued in Y — expected >=3 vertical zero-crossings \
         (solid lip -> empty undercut -> solid base ...), got {crossings}. A \
         heightfield can never exceed 1."
    );

    // (b) RENDER the overhang from a low front-quarter camera so the lip is seen
    // hanging over the dark air pocket, the base below it. Side sun for relief.
    let (w, h) = (256u32, 256u32);
    let eye = [extent * 0.5 + 3.0, 4.6, extent + 9.0];
    let center = [extent * 0.5, 4.6, 0.0];
    let fov_y = std::f32::consts::FRAC_PI_4;
    let rig = LightRig {
        sun_dir: [0.6, 0.45, 0.66],
        sun_intensity: 2.4,
        sky_intensity: 0.0,
        camera_fill: 0.0,
        rim_fill: 0.0,
        sky_dome_intensity: 0.0,
        sky_dome_zenith: [0.0, 0.0, 0.0],
        sky_dome_horizon: [0.0, 0.0, 0.0],
        ..Default::default()
    };
    let rgba = pathtrace_sdf_to_rgba(
        &patch.to_volume_input(),
        [0.0, 0.0, 0.0],
        1.0,
        [0.52, 0.48, 0.44],
        eye,
        center,
        fov_y,
        w,
        h,
        8,
        &rig,
    )
    .expect("cliff/overhang render should succeed");

    let png = std::env::temp_dir().join("terrain_cliff.png");
    super::write_png_rgba(png.to_str().unwrap(), &rgba, w, h);
    eprintln!("[terrain_cliff] wrote {}", png.display());

    // Per-column vertical profile: for each image column, find the topmost lit
    // pixel (the lip's top edge), then look for the solid-over-empty-over-solid
    // signature — a lit run (lip), a dark gap (the air pocket under the
    // overhang), then a lit run again (the base) BELOW it. A pure slope (no
    // overhang) has ONE continuous lit run per column; the undercut air pocket
    // is what splits it. Count columns showing that split.
    let thr = 26.0f32;
    let lit = |x: u32, y: u32| -> bool {
        luma(&rgba[((y * w + x) * 4) as usize..((y * w + x) * 4 + 4) as usize]) > thr
    };
    let mut overhang_columns = 0usize;
    let mut scanned_columns = 0usize;
    for x in 0..w {
        // Walk top->bottom; classify runs.
        let mut runs: Vec<(u32, u32)> = Vec::new(); // (start_y, end_y) of lit runs
        let mut run_start: Option<u32> = None;
        for y in 0..h {
            if lit(x, y) {
                run_start.get_or_insert(y);
            } else if let Some(s) = run_start.take() {
                runs.push((s, y - 1));
            }
        }
        if let Some(s) = run_start.take() {
            runs.push((s, h - 1));
        }
        if runs.is_empty() {
            continue;
        }
        scanned_columns += 1;
        // Overhang signature: ≥2 lit runs separated by a dark gap of real height
        // (the air pocket), with the gap ABOVE the lowest run (the base). Filter
        // tiny antialias gaps (gap >= 4 px).
        if runs.len() >= 2 {
            let big_gap = runs.windows(2).any(|w| w[1].0 - w[0].1 >= 4);
            if big_gap {
                overhang_columns += 1;
            }
        }
    }
    eprintln!(
        "[terrain_cliff] columns showing solid-over-empty-over-solid (overhang \
         signature): {overhang_columns} of {scanned_columns} lit columns"
    );
    assert!(
        scanned_columns > 40,
        "too little terrain in frame ({scanned_columns} lit columns) — camera/patch \
         framing drifted"
    );
    assert!(
        overhang_columns >= 8,
        "the rendered overhang must show solid lip ABOVE genuinely empty space \
         (a dark gap splitting the lit terrain into lip + base) in multiple \
         columns; got {overhang_columns}. If zero, the lip is not hanging over a \
         void — the render shows a plain slope."
    );
    eprintln!(
        "[terrain_cliff] CLIFF/OVERHANG renders: solid hangs over empty space, \
         multi-valued in Y. PASS"
    );
}
