//! Spectral splat ray tracing — CPU reference for the 3DGRT-style approach.
//!
//! Instead of *projecting* Gaussians to the screen (rasterization), we shoot a
//! ray per sample and accumulate 16-band spectral radiance through the
//! particles it pierces. This unlocks effects rasterization cannot do —
//! reflection/refraction/shadow rays — because every secondary ray is just
//! another `trace_ray` call.
//!
//! Per Gaussian, the ray's point of maximum density `t_peak` has a closed form:
//! transform the ray into the Gaussian's unit-sphere space (apply the inverse
//! rotation and divide by the half-axes), then `t_peak = -dot(o,d)/dot(d,d)`
//! is the argmin of the squared Mahalanobis distance along the ray. The alpha
//! contribution is `opacity * exp(-0.5 * d^2)` at `t_peak`, skipped past the
//! 3σ cutoff (matching the rasterizer's `power>0 → exp` convention and its
//! `radius = 3√λ_max` extent). Compositing is the standard front-to-back
//! `over`: `radiance += T*alpha*spectral; T *= 1-alpha`, early-out at
//! `T < 0.001` or after a hard `budget` of splats (bounded frame cost, like
//! the atom-budget renderer). Hits are gathered then sorted by `t_peak`, so
//! input order never changes the result.
//!
//! The GPU port reuses this exact `RayHit` math and `composite` ordering inside
//! a hardware-BVH any-hit shader; the `clas::SplatCluster` BVH traversed here
//! is the same acceleration structure the atom-budget splat renderer already
//! uses for LOD selection (the EWA rasterizer itself is BVH-free).

use glam::{Mat3, Quat, Vec3};
use vox_core::types::GaussianSplat;

use crate::clas::{ClusterBVHNode, SplatCluster};

/// σ Mahalanobis cutoff: beyond this the Gaussian contributes < exp(-0.5σ²) and
/// is skipped. Matches the rasterizer's `radius = σ·√λ_max` footprint extent.
/// CONFIG-FIRST (`materials.splat_sigma_cutoff`, default 3.0): this was a `const`
/// but is now read at the call site so the cutoff is retunable WITHOUT a rebuild.
/// The GPU mirror (`gpu/splat_rt_gpu.wgsl`) specializes the SAME value into its
/// `SIGMA_CUTOFF`/`POWER_CUTOFF` consts so the two paths stay bit-equivalent.
#[inline]
fn sigma_cutoff() -> f32 {
    vox_config::config().materials.splat_sigma_cutoff
}
/// `0.5 * SIGMA_CUTOFF^2` — compared against the peak power directly. Refactored
/// from a `const` to a runtime value derived from [`sigma_cutoff`] (so it always
/// tracks the configured cutoff) now that the cutoff is config-driven.
#[inline]
fn power_cutoff() -> f32 {
    let s = sigma_cutoff();
    0.5 * s * s
}
/// Stop compositing once remaining transmittance falls below this — identical
/// to `spectra_gaussian_render::TRANSMITTANCE_THRESHOLD` used by the rasterizer.
const TRANSMITTANCE_THRESHOLD: f32 = 0.001;
/// Minimum alpha worth compositing — identical to the rasterizer's
/// `ALPHA_THRESHOLD = 1/255` (a contribution below one 8-bit level is noise).
const ALPHA_THRESHOLD: f32 = 1.0 / 255.0;
/// Floor for ellipsoid half-axes so the covariance stays positive-definite
/// (2DGS splats carry `scale_w == 0`); mirrors the rasterizer's `max(1e-4)`.
const SCALE_FLOOR: f32 = 1e-4;

/// Number of spectral bands carried through compositing.
pub const BANDS: usize = GaussianSplat::BANDS;

/// Accumulated radiance along one ray: 16 spectral bands plus coverage alpha.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralRadiance {
    /// Per-band radiance (front-to-back `over`-composited).
    pub bands: [f32; BANDS],
    /// Coverage = `1 - transmittance` after the walk.
    pub alpha: f32,
}

impl SpectralRadiance {
    /// A fully transparent (zero-radiance) ray result.
    pub const EMPTY: SpectralRadiance = SpectralRadiance {
        bands: [0.0; BANDS],
        alpha: 0.0,
    };
}

/// Bounded-cost statistics for one `trace_ray` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceStats {
    /// Number of splats whose Gaussian was within the 3σ cutoff (candidate hits).
    pub hits_found: usize,
    /// Number of splats actually composited (`<= budget`).
    pub composited: usize,
    /// True if compositing stopped because transmittance saturated.
    pub saturated: bool,
}

/// One ray/Gaussian hit: where along the ray the density peaks, and the alpha
/// (opacity × Gaussian falloff) the Gaussian contributes there.
#[derive(Debug, Clone, Copy)]
struct RayHit {
    /// Ray parameter at the peak-density point.
    t_peak: f32,
    /// `opacity * exp(-0.5 * d^2)` evaluated at `t_peak`, clamped like the raster path.
    alpha: f32,
    /// Index of the contributing splat (for spectral lookup).
    splat: u32,
}

/// Analytic ray-vs-Gaussian peak for a single splat.
///
/// Returns `None` if the ray's closest approach is behind the origin, the
/// Gaussian is degenerate, or the peak lies beyond the 3σ cutoff. The returned
/// `alpha` follows the rasterizer convention exactly: `opacity * exp(power)`
/// with `power = -0.5 * d^2`, clamped to `0.99`, and dropped below
/// `ALPHA_THRESHOLD`.
fn ray_gaussian_hit(
    origin: Vec3,
    dir: Vec3,
    splat: &GaussianSplat,
    splat_index: u32,
) -> Option<RayHit> {
    let opacity = splat.opacity() as f32 / 255.0;
    let center = Vec3::from(splat.position());

    if splat.is_surface() {
        // ---- 2DGS surface disk: ray-plane intersect, 2D Gaussian in (u,v) ----
        let u = Vec3::from(splat.tangent_u());
        let v = Vec3::from(splat.tangent_v());
        let n = u.cross(v);
        let n_len = n.length();
        if n_len < 1e-8 {
            return None;
        }
        let normal = n / n_len;
        let denom = dir.dot(normal);
        if denom.abs() < 1e-8 {
            return None; // ray parallel to disk
        }
        let t = (center - origin).dot(normal) / denom;
        if t <= 0.0 {
            return None; // behind the ray origin
        }
        let hit = origin + dir * t;
        let rel = hit - center;
        // Disk-local coordinates: project onto the (unit) tangent axes, divide
        // by the semi-axis radii to get a unit-circle Mahalanobis distance.
        let su = u.dot(u).sqrt().max(SCALE_FLOOR);
        let sv = v.dot(v).sqrt().max(SCALE_FLOOR);
        let ru = splat.scale_u().max(SCALE_FLOOR);
        let rv = splat.scale_v().max(SCALE_FLOOR);
        let du = rel.dot(u / su) / ru;
        let dv = rel.dot(v / sv) / rv;
        let d2 = du * du + dv * dv;
        finish_hit(d2, opacity, t, splat_index)
    } else {
        // ---- 3DGS ellipsoid: transform ray into the unit-sphere space ----
        // Σ = R·diag(s^2)·Rᵀ. In the Gaussian's local frame, scaling each axis
        // by 1/s maps the ellipsoid to a unit sphere. We bring the ray into that
        // space and find the closest approach to the origin (the peak).
        let rot = Quat::from_xyzw(
            splat.rotation_raw()[0] as f32 / 32767.0,
            splat.rotation_raw()[1] as f32 / 32767.0,
            splat.rotation_raw()[2] as f32 / 32767.0,
            splat.rotation_raw()[3] as f32 / 32767.0,
        )
        .normalize();
        let r_mat = Mat3::from_quat(rot);
        let inv_s = Vec3::new(
            1.0 / splat.scale_u().max(SCALE_FLOOR),
            1.0 / splat.scale_v().max(SCALE_FLOOR),
            1.0 / splat.scale_w().max(SCALE_FLOOR),
        );
        // World→local is Rᵀ (rows of R). Map origin-relative ray into unit space.
        let r_t = r_mat.transpose();
        let o_local = (r_t * (origin - center)) * inv_s;
        let d_local = (r_t * dir) * inv_s;
        let dd = d_local.dot(d_local);
        if dd < 1e-12 {
            return None;
        }
        // argmin over t of |o_local + t·d_local|^2  ⇒  t = -o·d / d·d.
        let t_peak = -o_local.dot(d_local) / dd;
        if t_peak <= 0.0 {
            return None; // peak is behind the ray origin
        }
        let closest = o_local + d_local * t_peak;
        let d2 = closest.dot(closest); // squared Mahalanobis distance at the peak
        finish_hit(d2, opacity, t_peak, splat_index)
    }
}

/// Shared tail: apply the 3σ cutoff and the rasterizer's alpha conventions.
#[inline]
fn finish_hit(d2: f32, opacity: f32, t: f32, splat_index: u32) -> Option<RayHit> {
    let power = 0.5 * d2;
    if power > power_cutoff() {
        return None; // beyond the σ-cutoff — contributes < exp(-0.5σ²)
    }
    let alpha = (opacity * (-power).exp()).min(0.99);
    if alpha < ALPHA_THRESHOLD {
        return None;
    }
    Some(RayHit {
        t_peak: t,
        alpha,
        splat: splat_index,
    })
}

/// The 3σ world-space radius of a splat's Gaussian footprint — the largest
/// distance from its center at which it can still contribute past the cutoff.
fn splat_radius(s: &GaussianSplat) -> f32 {
    let r = if s.is_surface() {
        s.scale_u().abs().max(s.scale_v().abs())
    } else {
        s.scale_u()
            .abs()
            .max(s.scale_v().abs())
            .max(s.scale_w().abs())
    };
    sigma_cutoff() * r
}

/// Slab ray-AABB test on an AABB padded by `pad` on every side. Returns true if
/// the ray hits the padded box at any `t >= 0`.
#[inline]
fn ray_aabb_padded(origin: Vec3, inv_dir: Vec3, lo: Vec3, hi: Vec3, pad: f32) -> bool {
    let p = Vec3::splat(pad + 1e-4);
    let t1 = (lo - p - origin) * inv_dir;
    let t2 = (hi + p - origin) * inv_dir;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let t_near = tmin.x.max(tmin.y).max(tmin.z);
    let t_far = tmax.x.min(tmax.y).min(tmax.z);
    t_near <= t_far && t_far >= 0.0
}

/// Gather candidate hits from the BVH (or brute force if `bvh` is `None`).
///
/// The cluster/BVH AABBs returned by `clas` bound splat *centers* only, not
/// their Gaussian footprints, so we pad every box by the cluster's 3σ extent
/// before testing. This is conservative: traversal may visit a few extra
/// clusters, but never *skips* a splat whose Gaussian reaches the ray. The set
/// of hits passing the cutoff is therefore identical to brute force, so
/// radiance is unchanged — only cost differs.
fn gather_hits(
    origin: Vec3,
    dir: Vec3,
    splats: &[GaussianSplat],
    clusters: &[SplatCluster],
    bvh: Option<&ClusterBVHNode>,
    out: &mut Vec<RayHit>,
) {
    let inv_dir = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

    match bvh {
        None => {
            // Brute force: every splat is a candidate.
            for (i, s) in splats.iter().enumerate() {
                if let Some(h) = ray_gaussian_hit(origin, dir, s, i as u32) {
                    out.push(h);
                }
            }
        }
        Some(node) => {
            // Per-cluster footprint padding, and a global max for internal nodes
            // (whose AABBs are built from cluster center-AABBs, no extent info).
            let cluster_pad: Vec<f32> = clusters
                .iter()
                .map(|c| {
                    c.splat_indices
                        .iter()
                        .map(|&si| splat_radius(&splats[si as usize]))
                        .fold(0.0f32, f32::max)
                })
                .collect();
            let global_pad = cluster_pad.iter().copied().fold(0.0f32, f32::max);

            // Descend the BVH, testing splats only in clusters the ray's (padded)
            // AABB hits.
            let mut stack: Vec<&ClusterBVHNode> = vec![node];
            while let Some(n) = stack.pop() {
                match n {
                    ClusterBVHNode::Leaf { cluster_id } => {
                        let cluster = &clusters[*cluster_id as usize];
                        let pad = cluster_pad[*cluster_id as usize];
                        if !ray_aabb_padded(
                            origin,
                            inv_dir,
                            cluster.aabb_min,
                            cluster.aabb_max,
                            pad,
                        ) {
                            continue;
                        }
                        for &si in &cluster.splat_indices {
                            if let Some(h) = ray_gaussian_hit(origin, dir, &splats[si as usize], si)
                            {
                                out.push(h);
                            }
                        }
                    }
                    ClusterBVHNode::Internal {
                        aabb_min,
                        aabb_max,
                        left,
                        right,
                    } => {
                        if ray_aabb_padded(origin, inv_dir, *aabb_min, *aabb_max, global_pad) {
                            stack.push(left);
                            stack.push(right);
                        }
                    }
                }
            }
        }
    }
}

/// Trace a single ray, accumulating spectral radiance front-to-back.
///
/// `clusters`/`bvh` are the `clas` acceleration structure; pass `bvh = None` to
/// brute-force every splat (identical radiance, higher cost). `budget` is a
/// hard upper bound on composited splats — never unbounded.
pub fn trace_ray(
    origin: Vec3,
    dir: Vec3,
    splats: &[GaussianSplat],
    clusters: &[SplatCluster],
    bvh: Option<&ClusterBVHNode>,
    budget: usize,
) -> (SpectralRadiance, TraceStats) {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return (
            SpectralRadiance::EMPTY,
            TraceStats {
                hits_found: 0,
                composited: 0,
                saturated: false,
            },
        );
    }

    let mut hits: Vec<RayHit> = Vec::new();
    gather_hits(origin, dir, splats, clusters, bvh, &mut hits);
    let hits_found = hits.len();

    // Sort by depth along the ray — front-to-back. Stable, depth-keyed: input
    // order of the splat array is irrelevant to the composited result.
    hits.sort_by(|a, b| {
        a.t_peak
            .partial_cmp(&b.t_peak)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.splat.cmp(&b.splat))
    });

    let mut bands = [0.0f32; BANDS];
    let mut transmittance = 1.0f32;
    let mut composited = 0usize;
    let mut saturated = false;

    for h in &hits {
        if composited >= budget {
            break; // hard budget bound
        }
        if transmittance < TRANSMITTANCE_THRESHOLD {
            saturated = true;
            break;
        }
        let s = &splats[h.splat as usize];
        let weight = h.alpha * transmittance;
        for (b, band) in bands.iter_mut().enumerate() {
            *band += weight * s.spectral_f32(b);
        }
        transmittance *= 1.0 - h.alpha;
        composited += 1;
    }

    let radiance = SpectralRadiance {
        bands,
        alpha: 1.0 - transmittance,
    };
    (
        radiance,
        TraceStats {
            hits_found,
            composited,
            saturated,
        },
    )
}

/// A scene = splats plus a prebuilt `clas` acceleration structure.
pub struct RtScene {
    pub splats: Vec<GaussianSplat>,
    pub clusters: Vec<SplatCluster>,
    pub bvh: Option<ClusterBVHNode>,
}

impl RtScene {
    /// Build clusters + BVH from a splat list (`target_size` per cluster).
    pub fn build(splats: Vec<GaussianSplat>, target_size: usize) -> Self {
        let clusters = crate::clas::build_clusters(&splats, target_size);
        let bvh = crate::clas::build_cluster_bvh(&clusters);
        Self {
            splats,
            clusters,
            bvh,
        }
    }

    /// Trace one ray through this scene with a hard splat `budget`.
    pub fn trace(&self, origin: Vec3, dir: Vec3, budget: usize) -> (SpectralRadiance, TraceStats) {
        trace_ray(
            origin,
            dir,
            &self.splats,
            &self.clusters,
            self.bvh.as_ref(),
            budget,
        )
    }
}

/// A minimal orthographic camera for the cross-check tile renderer.
#[derive(Debug, Clone, Copy)]
pub struct OrthoCamera {
    /// Eye position (rays start here, offset across the image plane).
    pub eye: Vec3,
    /// Normalized forward direction (all rays share it — orthographic).
    pub forward: Vec3,
    /// Normalized right (image +x) axis.
    pub right: Vec3,
    /// Normalized up (image +y) axis.
    pub up: Vec3,
    /// World-space width of the image plane.
    pub width: f32,
    /// World-space height of the image plane.
    pub height: f32,
}

/// Render the scene with one orthographic ray per pixel.
///
/// Returns `width*height` spectral pixels in row-major order (y from top). This
/// is the ray-traced cross-check harness against the rasterizer.
pub fn render_orthographic(
    scene: &RtScene,
    camera: &OrthoCamera,
    width: u32,
    height: u32,
    budget: usize,
) -> Vec<[f32; BANDS]> {
    let mut out = vec![[0.0f32; BANDS]; (width * height) as usize];
    let fwd = camera.forward.normalize();
    let right = camera.right.normalize();
    let up = camera.up.normalize();
    for py in 0..height {
        // +y in image space is "up"; map row 0 to the top of the plane.
        let v = ((py as f32 + 0.5) / height as f32 - 0.5) * camera.height;
        for px in 0..width {
            let u = ((px as f32 + 0.5) / width as f32 - 0.5) * camera.width;
            let origin = camera.eye + right * u + up * v;
            let (rad, _) = scene.trace(origin, fwd, budget);
            out[(py * width + px) as usize] = rad.bands;
        }
    }
    out
}

/// Transmittance of a shadow / occlusion ray from `from` to `to`.
///
/// Returns the fraction of light that survives the segment: the product of
/// `(1 - alpha)` over every Gaussian the segment pierces. This is
/// **order-independent** — multiplication commutes — so unlike `trace_ray` we
/// do **not** sort the hits. We also bound the segment by its length so
/// Gaussians past the light are ignored. `from == to` (degenerate) returns 1.0.
pub fn transmittance(
    from: Vec3,
    to: Vec3,
    splats: &[GaussianSplat],
    clusters: &[SplatCluster],
    bvh: Option<&ClusterBVHNode>,
    budget: usize,
) -> f32 {
    let seg = to - from;
    let len = seg.length();
    if len < 1e-8 {
        return 1.0; // degenerate: nothing between the endpoints
    }
    let dir = seg / len;

    let mut hits: Vec<RayHit> = Vec::new();
    gather_hits(from, dir, splats, clusters, bvh, &mut hits);

    let mut t = 1.0f32;
    let mut used = 0usize;
    for h in &hits {
        if used >= budget {
            break; // hard budget bound, same as trace_ray
        }
        // Only count Gaussians whose peak lies on the segment (before the light).
        if h.t_peak < 0.0 || h.t_peak > len {
            continue;
        }
        t *= 1.0 - h.alpha;
        used += 1;
        if t < TRANSMITTANCE_THRESHOLD {
            break; // fully occluded
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    const O: u8 = 255; // fully opaque
    /// Opacity-255 quantization tolerance: 255/255 = 1.0 exactly, so opacity
    /// itself is exact; the only error is spectral f16. We compute spectral
    /// tolerance separately where bands are compared.
    fn flat_spectral(value: f32) -> [u16; 16] {
        [f16::from_f32(value).to_bits(); 16]
    }
    fn band_spectral(band: usize, value: f32) -> [u16; 16] {
        let mut s = [f16::from_f32(0.0).to_bits(); 16];
        s[band] = f16::from_f32(value).to_bits();
        s
    }
    /// The f16 round-trip value of `v` — the honest target a band should match.
    fn f16_round(v: f32) -> f32 {
        f16::from_f32(v).to_f32()
    }

    /// Single axis-aligned 3DGS at origin; ray straight through center along -Z.
    /// alpha must equal opacity (opacity 255 → exactly 1.0, but clamped 0.99).
    #[test]
    fn single_splat_center_alpha_equals_opacity() {
        let splat = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            O,
            flat_spectral(1.0),
        );
        let scene = RtScene::build(vec![splat], 64);
        // Ray from +Z toward origin passes exactly through the center → d2 = 0.
        let (rad, stats) = scene.trace(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 64);
        assert_eq!(stats.composited, 1, "exactly one splat composited");
        // opacity 1.0 is clamped to 0.99 by the raster convention.
        assert!(
            (rad.alpha - 0.99).abs() < 1e-6,
            "center alpha should be the clamped opacity 0.99, got {}",
            rad.alpha
        );
    }

    /// Ray offset by exactly 1σ → alpha is exp(-0.5) of the (unclamped) center.
    #[test]
    fn single_splat_one_sigma_falloff() {
        // Use opacity 128 so center alpha = 128/255 ≈ 0.502 stays below the 0.99
        // clamp and the exp(-0.5) ratio is exact, not masked by clamping.
        let opacity = 128u8;
        let base = opacity as f32 / 255.0;
        let splat = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            opacity,
            flat_spectral(1.0),
        );
        let scene = RtScene::build(vec![splat], 64);

        // Center ray.
        let (center, _) = scene.trace(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 64);
        // Offset ray by exactly 1σ (= 1.0 here) in x.
        let (offset, _) = scene.trace(Vec3::new(1.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 64);

        assert!(
            (center.alpha - base).abs() < 1e-6,
            "center alpha {}",
            center.alpha
        );
        let ratio = offset.alpha / center.alpha;
        assert!(
            (ratio - (-0.5f32).exp()).abs() < 1e-3,
            "1σ offset alpha ratio should be exp(-0.5)={}, got {}",
            (-0.5f32).exp(),
            ratio
        );
    }

    /// Ray beyond the 3σ cutoff → exactly zero contribution.
    #[test]
    fn single_splat_beyond_cutoff_zero() {
        let splat = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            O,
            flat_spectral(1.0),
        );
        let scene = RtScene::build(vec![splat], 64);
        // Offset by 3.5σ (> 3σ cutoff) → no hit.
        let (rad, stats) = scene.trace(Vec3::new(3.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 64);
        assert_eq!(stats.hits_found, 0, "beyond 3σ: no candidate hits");
        assert_eq!(rad.alpha, 0.0, "beyond cutoff alpha must be exactly 0");
        assert_eq!(rad.bands, [0.0; BANDS], "no spectral contribution");
    }

    /// Two splats on one ray: near (red band hot) dominates far (blue band hot),
    /// and REVERSING the array order yields a bit-identical result.
    #[test]
    fn ordering_near_dominates_and_is_order_independent() {
        // A is nearer (z=1), red-hot (band 2). B is farther (z=-1), blue-hot (band 13).
        let a = GaussianSplat::volume(
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            200,
            band_spectral(2, 1.0),
        );
        let b = GaussianSplat::volume(
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            200,
            band_spectral(13, 1.0),
        );

        let scene_ab = RtScene::build(vec![a, b], 64);
        let scene_ba = RtScene::build(vec![b, a], 64);
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let (rad_ab, _) = scene_ab.trace(origin, dir, 64);
        let (rad_ba, _) = scene_ba.trace(origin, dir, 64);

        // Near (A, band 2) must dominate far (B, band 13) after attenuation.
        assert!(
            rad_ab.bands[2] > rad_ab.bands[13],
            "near red band {} should dominate far blue band {}",
            rad_ab.bands[2],
            rad_ab.bands[13]
        );
        // Order independence: sort is by t, not index → bit-identical.
        assert_eq!(
            rad_ab.bands, rad_ba.bands,
            "reversing splat order must give bit-identical radiance"
        );
        assert_eq!(
            rad_ab.alpha, rad_ba.alpha,
            "alpha must be order-independent"
        );
    }

    /// Transmittance: single splat → 1-a; two splats → (1-a1)(1-a2); degenerate → 1.
    #[test]
    fn transmittance_products() {
        let opacity = 128u8;
        let a = opacity as f32 / 255.0; // center alpha (d2=0, below 0.99 clamp)
        let s1 = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            opacity,
            flat_spectral(1.0),
        );
        let s2 = GaussianSplat::volume(
            [0.0, 0.0, -2.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            opacity,
            flat_spectral(1.0),
        );

        // One splat: from +Z(5) toward -Z far past it.
        let scene1 = RtScene::build(vec![s1], 64);
        let t1 = transmittance(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -5.0),
            &scene1.splats,
            &scene1.clusters,
            scene1.bvh.as_ref(),
            64,
        );
        assert!(
            (t1 - (1.0 - a)).abs() < 1e-6,
            "single-splat transmittance {} vs {}",
            t1,
            1.0 - a
        );

        // Two splats: product of (1-a) each.
        let scene2 = RtScene::build(vec![s1, s2], 64);
        let t2 = transmittance(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -5.0),
            &scene2.splats,
            &scene2.clusters,
            scene2.bvh.as_ref(),
            64,
        );
        let expected = (1.0 - a) * (1.0 - a);
        assert!(
            (t2 - expected).abs() < 1e-6,
            "two-splat transmittance {} vs {}",
            t2,
            expected
        );

        // Degenerate from==to → 1.0.
        let td = transmittance(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
            &scene2.splats,
            &scene2.clusters,
            scene2.bvh.as_ref(),
            64,
        );
        assert_eq!(td, 1.0, "degenerate segment is fully transmissive");
    }

    /// Budget bound: 10k co-located splats, budget 64 → exactly ≤64 composited.
    #[test]
    fn budget_caps_composited_count() {
        let mut splats = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            splats.push(GaussianSplat::volume(
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                Quat::IDENTITY,
                10, // low opacity so transmittance doesn't saturate before budget
                flat_spectral(0.5),
            ));
        }
        let scene = RtScene::build(splats, 256);
        let (_, stats) = scene.trace(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 64);
        assert!(
            stats.hits_found >= 64,
            "all co-located splats are candidates: {}",
            stats.hits_found
        );
        assert!(
            stats.composited <= 64,
            "budget must cap composited at 64, got {}",
            stats.composited
        );
        assert_eq!(
            stats.composited, 64,
            "with low opacity, exactly the budget is consumed"
        );
    }

    /// BVH equivalence: brute force (bvh=None) vs BVH on a 200-splat random scene
    /// → bit-identical radiance. Traversal changes cost, not results.
    #[test]
    fn bvh_matches_brute_force() {
        // Deterministic pseudo-random scene (LCG, no rand dep).
        let mut seed = 0x1234_5678u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 // [0,1)
        };
        let mut splats = Vec::with_capacity(200);
        for _ in 0..200 {
            let p = [
                (rng() - 0.5) * 8.0,
                (rng() - 0.5) * 8.0,
                (rng() - 0.5) * 8.0,
            ];
            let band = (rng() * 16.0) as usize % 16;
            splats.push(GaussianSplat::volume(
                p,
                [0.6, 0.6, 0.6],
                Quat::IDENTITY,
                150,
                band_spectral(band, 1.0),
            ));
        }
        let scene = RtScene::build(splats.clone(), 32);

        // A handful of rays from different directions.
        let rays = [
            (Vec3::new(0.0, 0.0, 12.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(12.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
            (Vec3::new(0.0, 12.0, 3.0), Vec3::new(0.0, -1.0, -0.2)),
            (Vec3::new(7.0, 7.0, 7.0), Vec3::new(-1.0, -1.0, -1.0)),
        ];
        for (o, d) in rays {
            let (brute, _) = trace_ray(o, d, &scene.splats, &scene.clusters, None, 256);
            let (bvh, _) = trace_ray(
                o,
                d,
                &scene.splats,
                &scene.clusters,
                scene.bvh.as_ref(),
                256,
            );
            assert_eq!(
                brute.bands, bvh.bands,
                "BVH radiance must match brute force bit-for-bit"
            );
            assert_eq!(brute.alpha, bvh.alpha, "BVH alpha must match brute force");
        }
    }

    /// Determinism: same scene + rays twice → bit-equal.
    #[test]
    fn deterministic() {
        let splats = vec![
            GaussianSplat::volume(
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                Quat::IDENTITY,
                200,
                band_spectral(4, 1.0),
            ),
            GaussianSplat::volume(
                [1.0, 0.0, -1.0],
                [0.8, 0.8, 0.8],
                Quat::IDENTITY,
                180,
                band_spectral(9, 1.0),
            ),
            GaussianSplat::volume(
                [-1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                Quat::IDENTITY,
                220,
                band_spectral(12, 1.0),
            ),
        ];
        let scene = RtScene::build(splats, 64);
        let o = Vec3::new(0.5, 0.5, 5.0);
        let d = Vec3::new(0.0, 0.0, -1.0);
        let (r1, s1) = scene.trace(o, d, 64);
        let (r2, s2) = scene.trace(o, d, 64);
        assert_eq!(r1.bands, r2.bands, "radiance must be deterministic");
        assert_eq!(r1.alpha, r2.alpha);
        assert_eq!(s1, s2, "stats must be deterministic");
    }

    /// Spectral f16 quantization is honoured: a band traced through one opaque
    /// splat equals opacity-weighted f16-rounded value within f16 tolerance.
    #[test]
    fn spectral_quantization_tolerance() {
        let value = 0.7f32;
        let opacity = 128u8;
        let splat = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            opacity,
            band_spectral(5, value),
        );
        let scene = RtScene::build(vec![splat], 64);
        let (rad, _) = scene.trace(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 64);
        // alpha = opacity (d2=0), radiance = alpha * f16_round(value).
        let alpha = opacity as f32 / 255.0;
        let expected = alpha * f16_round(value);
        // f16 relative precision ~2^-11; absolute tolerance scaled by magnitude.
        let tol = expected.abs() * 1e-3 + 1e-5;
        assert!(
            (rad.bands[5] - expected).abs() < tol,
            "band 5 {} vs expected {} (f16 tol {})",
            rad.bands[5],
            expected,
            tol
        );
    }
}
