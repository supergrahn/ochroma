//! Canonical signed-distance-field representation, engine-wide (design
//! `2026-06-10-sdf-pillar-design.md` §5).
//!
//! This module owns the *representation* only: grid descriptor, quantized
//! payload, sampling math, and the two validators every cook gate runs —
//! the generalized-winding-number (GWN) sign oracle and the eikonal
//! (`|∇d| ≈ 1`) metric-validity check. No GPU, no game concepts.
//!
//! `vox_physics::sdf` re-exports these types so query-layer paths keep
//! compiling; `vox_render::gpu::sdf_bake` produces `SdfField`s on GPU and
//! is gated against the CPU oracles defined here.

use glam::{DVec3, Vec3};
use std::fmt;

/// How the payload's sign was established (design §4.9).
///
/// `Closed`: sign verified by generalized winding number — inside is negative.
/// `Shell`: unsigned field, placement-only semantics; never used for
/// inside/outside tests. The explicit escape hatch for open meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfSign {
    Closed,
    Shell,
}

/// Construction-time validation failures for [`SdfDesc`] / [`SdfField`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdfError {
    /// Each axis must be ≥ 2 and ≤ 96 (tier-1 contract, design §5).
    ResolutionOutOfRange([u32; 3]),
    InvalidVoxelSize,
    InvalidNarrowBand,
    PayloadCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for SdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResolutionOutOfRange(res) => {
                write!(f, "SDF resolution must be 2..=96 on every axis: {res:?}")
            }
            Self::InvalidVoxelSize => write!(f, "SDF voxel_size must be positive and finite"),
            Self::InvalidNarrowBand => write!(f, "SDF narrow_band must be positive and finite"),
            Self::PayloadCountMismatch { expected, actual } => write!(
                f,
                "SDF payload count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for SdfError {}

/// Grid descriptor shared by every tier-1 field. Mirrors (and is the canonical
/// replacement for the divergent copies of) `SdfVolumeDesc` /
/// `ReadyAssetSdfVolume` / `SdfSpatial`. Private fields — construction
/// validates, accessors expose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfDesc {
    resolution: [u32; 3],
    origin: Vec3,
    voxel_size: f32,
    narrow_band: f32,
    sign: SdfSign,
}

impl SdfDesc {
    pub fn new(
        resolution: [u32; 3],
        origin: Vec3,
        voxel_size: f32,
        narrow_band: f32,
        sign: SdfSign,
    ) -> Result<Self, SdfError> {
        if resolution.iter().any(|&n| !(2..=96).contains(&n)) {
            return Err(SdfError::ResolutionOutOfRange(resolution));
        }
        if !(voxel_size.is_finite() && voxel_size > 0.0) {
            return Err(SdfError::InvalidVoxelSize);
        }
        if !(narrow_band.is_finite() && narrow_band > 0.0) {
            return Err(SdfError::InvalidNarrowBand);
        }
        Ok(Self {
            resolution,
            origin,
            voxel_size,
            narrow_band,
            sign,
        })
    }

    pub fn resolution(&self) -> [u32; 3] {
        self.resolution
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    pub fn narrow_band(&self) -> f32 {
        self.narrow_band
    }

    pub fn sign(&self) -> SdfSign {
        self.sign
    }

    pub fn sample_count(&self) -> usize {
        self.resolution[0] as usize * self.resolution[1] as usize * self.resolution[2] as usize
    }

    /// Asset-local AABB covered by the grid (voxel (0,0,0) centre to the
    /// last voxel centre — the sampleable domain).
    pub fn local_bounds(&self) -> (Vec3, Vec3) {
        let max = self.origin
            + Vec3::new(
                (self.resolution[0] - 1) as f32,
                (self.resolution[1] - 1) as f32,
                (self.resolution[2] - 1) as f32,
            ) * self.voxel_size;
        (self.origin, max)
    }
}

/// Quantized distance payload, X-fastest then Y then Z (the existing engine
/// layout). Decode = `value / I::MAX * narrow_band` metres.
#[derive(Debug, Clone, PartialEq)]
pub enum SdfPayload {
    Snorm8(Vec<i8>),
    Snorm16(Vec<i16>),
}

impl SdfPayload {
    pub fn len(&self) -> usize {
        match self {
            Self::Snorm8(v) => v.len(),
            Self::Snorm16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One asset-local distance field. Construction validates desc/payload;
/// immutable after build — share via `Arc`.
#[derive(Debug, Clone, PartialEq)]
pub struct SdfField {
    desc: SdfDesc,
    payload: SdfPayload,
}

impl SdfField {
    pub fn new(desc: SdfDesc, payload: SdfPayload) -> Result<Self, SdfError> {
        let expected = desc.sample_count();
        if payload.len() != expected {
            return Err(SdfError::PayloadCountMismatch {
                expected,
                actual: payload.len(),
            });
        }
        Ok(Self { desc, payload })
    }

    pub fn desc(&self) -> SdfDesc {
        self.desc
    }

    pub fn payload(&self) -> &SdfPayload {
        &self.payload
    }

    /// Raw payload bytes for atlas upload (snorm8 = 1 B/voxel, snorm16 = 2).
    pub fn payload_bytes(&self) -> &[u8] {
        match &self.payload {
            SdfPayload::Snorm8(v) => bytemuck::cast_slice(v),
            SdfPayload::Snorm16(v) => bytemuck::cast_slice(v),
        }
    }

    /// Decoded distance (metres) at flat voxel index — saturates at
    /// ±narrow_band by construction of the encoding.
    fn voxel_metres(&self, index: usize) -> f32 {
        let normalized = match &self.payload {
            SdfPayload::Snorm8(v) => {
                let raw = v[index];
                if raw == i8::MIN {
                    -1.0
                } else {
                    raw as f32 / i8::MAX as f32
                }
            }
            SdfPayload::Snorm16(v) => {
                let raw = v[index];
                if raw == i16::MIN {
                    -1.0
                } else {
                    raw as f32 / i16::MAX as f32
                }
            }
        };
        normalized.clamp(-1.0, 1.0) * self.desc.narrow_band
    }

    /// Trilinear sample on decoded metres. `None` outside the grid.
    pub fn sample_local(&self, p: Vec3) -> Option<f32> {
        let grid = (p - self.desc.origin) / self.desc.voxel_size;
        let res = self.desc.resolution;
        if grid.x < 0.0
            || grid.y < 0.0
            || grid.z < 0.0
            || grid.x > (res[0] - 1) as f32
            || grid.y > (res[1] - 1) as f32
            || grid.z > (res[2] - 1) as f32
        {
            return None;
        }
        let (x0, x1, tx) = axis_sample(grid.x, res[0]);
        let (y0, y1, ty) = axis_sample(grid.y, res[1]);
        let (z0, z1, tz) = axis_sample(grid.z, res[2]);

        let idx = |x: u32, y: u32, z: u32| -> usize {
            (z as usize * res[1] as usize + y as usize) * res[0] as usize + x as usize
        };
        let c000 = self.voxel_metres(idx(x0, y0, z0));
        let c100 = self.voxel_metres(idx(x1, y0, z0));
        let c010 = self.voxel_metres(idx(x0, y1, z0));
        let c110 = self.voxel_metres(idx(x1, y1, z0));
        let c001 = self.voxel_metres(idx(x0, y0, z1));
        let c101 = self.voxel_metres(idx(x1, y0, z1));
        let c011 = self.voxel_metres(idx(x0, y1, z1));
        let c111 = self.voxel_metres(idx(x1, y1, z1));

        let c00 = c000 + (c100 - c000) * tx;
        let c10 = c010 + (c110 - c010) * tx;
        let c01 = c001 + (c101 - c001) * tx;
        let c11 = c011 + (c111 - c011) * tx;
        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;
        Some(c0 + (c1 - c0) * tz)
    }

    /// Central-difference surface normal on decoded metres. `None` outside
    /// the grid or where the gradient degenerates.
    pub fn normal_local(&self, p: Vec3) -> Option<Vec3> {
        let h = self.desc.voxel_size * 0.5;
        let dx = self.sample_local(p + Vec3::X * h)? - self.sample_local(p - Vec3::X * h)?;
        let dy = self.sample_local(p + Vec3::Y * h)? - self.sample_local(p - Vec3::Y * h)?;
        let dz = self.sample_local(p + Vec3::Z * h)? - self.sample_local(p - Vec3::Z * h)?;
        let g = Vec3::new(dx, dy, dz);
        if g.length_squared() < 1.0e-12 {
            return None;
        }
        Some(g / g.length())
    }
}

/// Validation report printed by the cook gate (design §4.9). Private fields,
/// accessors only; built by the producers (e.g. `vox_render::gpu::sdf_bake`).
#[derive(Debug, Clone, PartialEq)]
pub struct SdfValidation {
    inside_negative: bool,
    eikonal_p95: f32,
    winding_min_abs: f32,
    open_mesh: bool,
}

impl SdfValidation {
    pub fn new(
        inside_negative: bool,
        eikonal_p95: f32,
        winding_min_abs: f32,
        open_mesh: bool,
    ) -> Self {
        Self {
            inside_negative,
            eikonal_p95,
            winding_min_abs,
            open_mesh,
        }
    }

    /// Every GWN-interior probe sampled ≤ one quantization step below zero.
    pub fn inside_negative(&self) -> bool {
        self.inside_negative
    }

    /// p95 of `| |∇d| − 1 |` over in-band samples; gate is < 0.15.
    pub fn eikonal_p95(&self) -> f32 {
        self.eikonal_p95
    }

    /// min |w| over interior-ish probes; fractional (< ~0.85) ⇒ open mesh.
    pub fn winding_min_abs(&self) -> f32 {
        self.winding_min_abs
    }

    pub fn open_mesh(&self) -> bool {
        self.open_mesh
    }
}

fn axis_sample(coord: f32, resolution: u32) -> (u32, u32, f32) {
    let max_index = resolution - 1;
    let clamped = coord.clamp(0.0, max_index as f32);
    let i0 = clamped.floor() as u32;
    let i1 = (i0 + 1).min(max_index);
    (i0, i1, clamped - i0 as f32)
}

/// Signed solid angle of triangle (a, b, c) seen from the origin
/// (Van Oosterom & Strackee), f64 throughout.
fn triangle_solid_angle(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let la = a.length();
    let lb = b.length();
    let lc = c.length();
    let numerator = a.dot(b.cross(c));
    let denominator = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
    2.0 * numerator.atan2(denominator)
}

/// Generalized winding number of the indexed triangle mesh at query point `q`:
/// the f64 sum of signed solid angles over all triangles, divided by 4π.
///
/// |w| ≈ 1 inside a closed mesh, ≈ 0 outside, fractional near/through holes —
/// robust on the exact failure modes ray-parity has (coplanar shared edges,
/// grazing rays, slivers). Re-implemented here because forge's
/// `generalized_winding_number` is a private fn; cook sign and forge winding
/// repair share the same *mathematical authority*, not the same symbol.
/// Orientation-independent consumers should test `|w|` (a consistently
/// inward-wound closed mesh sums to −1).
///
/// Out-of-range indices are skipped (the producer validates separately).
pub fn generalized_winding_number(positions: &[[f32; 3]], indices: &[[u32; 3]], q: Vec3) -> f64 {
    let qd = DVec3::new(q.x as f64, q.y as f64, q.z as f64);
    let vert = |i: u32| -> Option<DVec3> {
        let p = positions.get(i as usize)?;
        Some(DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64) - qd)
    };
    let mut total = 0.0f64;
    for tri in indices {
        let (Some(a), Some(b), Some(c)) = (vert(tri[0]), vert(tri[1]), vert(tri[2])) else {
            continue;
        };
        total += triangle_solid_angle(a, b, c);
    }
    total / (4.0 * std::f64::consts::PI)
}

/// p95 of `| |∇d| − 1 |` over up to `samples` deterministic hash-jittered
/// in-band points (one-voxel interior margin so the half-voxel
/// central-difference stencil stays on-grid).
///
/// "In-band" is stencil-aware: `|d| < min(0.9·band, band − 1.5h)` so the
/// stencil never reaches the saturated skirt of the encoding — the payload
/// clamps at ±band *by design*, and a clamped value is not a distance, so
/// measuring its gradient would charge the field for its own contract.
///
/// A true SDF reads ≪ 0.15; an occupancy grid (air = +k / solid = −k) reads
/// ≈ 1 in flats and ≫ 1 across jumps — the design §4.9 metric-validity gate.
/// Returns `f32::INFINITY` when no in-band sample is found (nothing to
/// validate ⇒ the field cannot pass the gate vacuously).
pub fn eikonal_p95(field: &SdfField, samples: usize, seed: u64) -> f32 {
    let desc = field.desc();
    let res = desc.resolution();
    let voxel = desc.voxel_size();
    let band = desc.narrow_band();
    let origin = desc.origin();
    let h = voxel * 0.5;

    // Sampleable interior: one voxel in from each face so p ± h stays on-grid.
    let lo = origin + Vec3::splat(voxel);
    let extent = Vec3::new(
        (res[0] as f32 - 3.0).max(0.0),
        (res[1] as f32 - 3.0).max(0.0),
        (res[2] as f32 - 3.0).max(0.0),
    ) * voxel;

    let mut errs: Vec<f32> = Vec::with_capacity(samples);
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next_unit = move || -> f32 {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    };

    let in_band_limit = (0.9 * band).min(band - 1.5 * h);
    if in_band_limit <= 0.0 {
        return f32::INFINITY; // band thinner than the stencil — nothing measurable
    }
    let max_attempts = samples.saturating_mul(20).max(1);
    let mut attempts = 0usize;
    while errs.len() < samples && attempts < max_attempts {
        attempts += 1;
        let p = lo + Vec3::new(next_unit(), next_unit(), next_unit()) * extent;
        let Some(d) = field.sample_local(p) else {
            continue;
        };
        if d.abs() >= in_band_limit {
            continue;
        }
        let stencil = [
            field.sample_local(p + Vec3::X * h),
            field.sample_local(p - Vec3::X * h),
            field.sample_local(p + Vec3::Y * h),
            field.sample_local(p - Vec3::Y * h),
            field.sample_local(p + Vec3::Z * h),
            field.sample_local(p - Vec3::Z * h),
        ];
        let [Some(px), Some(mx), Some(py), Some(my), Some(pz), Some(mz)] = stencil else {
            continue;
        };
        let grad = Vec3::new(px - mx, py - my, pz - mz) / (2.0 * h);
        errs.push((grad.length() - 1.0).abs());
    }

    if errs.is_empty() {
        return f32::INFINITY;
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((errs.len() as f32 * 0.95).ceil() as usize).clamp(1, errs.len());
    errs[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// 12 outward-wound triangles of the axis-aligned box [min, max].
    fn cube_mesh(min: [f32; 3], max: [f32; 3]) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let (x0, y0, z0) = (min[0], min[1], min[2]);
        let (x1, y1, z1) = (max[0], max[1], max[2]);
        let positions = vec![
            [x0, y0, z0], // 0
            [x1, y0, z0], // 1
            [x1, y1, z0], // 2
            [x0, y1, z0], // 3
            [x0, y0, z1], // 4
            [x1, y0, z1], // 5
            [x1, y1, z1], // 6
            [x0, y1, z1], // 7
        ];
        // CCW seen from outside (right-hand normals point outward).
        let indices = vec![
            [0u32, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        (positions, indices)
    }

    /// GWN must read |w| ≈ 1 inside a closed cube, ≈ 0 far outside, and a
    /// fractional value at the centroid once a face is removed — the open-mesh
    /// failure the cook gate keys on.
    #[test]
    fn gwn_separates_inside_outside_and_flags_holes() {
        let (positions, indices) = cube_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let w_inside =
            generalized_winding_number(&positions, &indices, Vec3::new(0.5, 0.5, 0.5)).abs();
        let w_outside =
            generalized_winding_number(&positions, &indices, Vec3::new(3.0, 3.0, 3.0)).abs();

        // Remove the +Z face (two triangles) — the mesh is now open.
        let holed: Vec<[u32; 3]> = indices[..2]
            .iter()
            .chain(indices[4..].iter())
            .copied()
            .collect();
        assert_eq!(holed.len(), 10, "exactly one quad face removed");
        let w_holed =
            generalized_winding_number(&positions, &holed, Vec3::new(0.5, 0.5, 0.5)).abs();

        println!("[gwn] cube inside w={w_inside:.4} outside w={w_outside:.4} holed w={w_holed:.4}");

        assert!(
            (w_inside - 1.0).abs() < 1e-4,
            "interior winding must be ≈1, got {w_inside}"
        );
        assert!(
            w_outside < 1e-4,
            "exterior winding must be ≈0, got {w_outside}"
        );
        assert!(
            w_holed > 0.05 && w_holed < 0.95,
            "holed-mesh winding must be fractional, got {w_holed}"
        );
        // The analytic value: one cube face seen from the centre subtends 4π/6.
        assert!(
            (w_holed - (1.0 - 1.0 / 6.0)).abs() < 1e-3,
            "holed centroid winding should be ≈ 5/6, got {w_holed}"
        );
    }

    /// The eikonal validator must pass a true SDF (analytic box, snorm16) and
    /// FAIL an occupancy grid (air = +1.0 / solid = −1.0 — the `TerrainVolume`
    /// failure mode the design §4.9 metric gate exists to catch).
    #[test]
    fn eikonal_p95_passes_true_sdf_fails_occupancy() {
        // -- True SDF: analytic box half-extents (0.5, 0.4, 0.3) on a 64³ grid.
        // (Resolution matters: the eikonal check measures the *grid's* metric
        // validity — at 24³ the box's edge singularities occupy enough of the
        // band that even the analytic SDF honestly reads p95 ≈ 0.2.)
        let res = 64u32;
        let voxel = 0.04f32;
        let band = 0.5f32;
        let origin = Vec3::splat(-(res as f32 - 1.0) * voxel * 0.5);
        let half = Vec3::new(0.5, 0.4, 0.3);
        let mut snorm16 = Vec::with_capacity((res * res * res) as usize);
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;
                    let q = p.abs() - half;
                    let d = q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0);
                    let n = (d / band).clamp(-1.0, 1.0);
                    snorm16.push((n * i16::MAX as f32).round() as i16);
                }
            }
        }
        let desc = SdfDesc::new([res, res, res], origin, voxel, band, SdfSign::Closed)
            .expect("valid desc");
        let field =
            SdfField::new(desc, SdfPayload::Snorm16(snorm16)).expect("payload count matches");
        let box_p95 = eikonal_p95(&field, 10_000, 42);

        // -- Occupancy grid: ±1.0 m everywhere, band 4.0 so nothing saturates.
        let ores = 16u32;
        let ovoxel = 0.2f32;
        let oband = 4.0f32;
        let oorigin = Vec3::ZERO;
        let mut occ = Vec::with_capacity((ores * ores * ores) as usize);
        for z in 0..ores {
            for y in 0..ores {
                for x in 0..ores {
                    let solid =
                        (3..=12).contains(&x) && (3..=12).contains(&y) && (3..=12).contains(&z);
                    let d: f32 = if solid { -1.0 } else { 1.0 };
                    occ.push(((d / oband) * i16::MAX as f32).round() as i16);
                }
            }
        }
        let odesc = SdfDesc::new([ores, ores, ores], oorigin, ovoxel, oband, SdfSign::Closed)
            .expect("valid desc");
        let ofield = SdfField::new(odesc, SdfPayload::Snorm16(occ)).expect("payload count matches");
        let occ_p95 = eikonal_p95(&ofield, 10_000, 42);

        println!("[eikonal] box p95={box_p95:.4} occupancy p95={occ_p95:.4}");

        assert!(
            box_p95 < 0.15,
            "analytic box SDF must satisfy the eikonal gate, p95={box_p95}"
        );
        assert!(
            occ_p95 > 1.0,
            "occupancy grid must be flagged (|∇d| is 0 in flats, huge at jumps), p95={occ_p95}"
        );
    }
}
