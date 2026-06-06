//! Hybrid mesh + Gaussian-splat compositing in a single depth-correct pass.
//!
//! This is a CPU reference implementation, mirroring the engine's software-first
//! pattern (`gpu::software_rasteriser`). It renders triangle meshes and spectral
//! Gaussian splats into one coherent [`SpectralFramebuffer`] with correct mutual
//! occlusion: splats behind mesh geometry are rejected by the shared depth buffer,
//! splats in front composite over the mesh background normally.
//!
//! # Pipeline
//!
//! 1. **Mesh pass** — each triangle is projected through the same camera
//!    convention used by the splat rasteriser (spectra OpenCV-style: +X right,
//!    +Y down, looking down +Z; `screen = f·cam_xy/cam_z + size/2`,
//!    `depth = cam_z`). Triangles are rasterised with barycentric coverage,
//!    perspective-correct depth (we interpolate INVERSE depth `1/cam_z`, which is
//!    exactly linear in screen space, then invert to recover the true
//!    perspective `cam_z` stored in the depth buffer — so the mesh's stored depth
//!    matches the splats' true `proj.depth` and inter-surface occlusion ordering
//!    is correct, including on steep/grazing triangles spanning a large depth
//!    range), and flat/Lambertian shading: per-mesh spectral reflectance scaled by
//!    `max(0, dot(n, sun_dir))` plus a small ambient term. The shaded spectrum,
//!    `cam_z` depth, and world normal are written to the framebuffer with a
//!    standard z-test (nearest wins).
//!
//! 2. **Splat pass** — splats are projected and sorted front-to-back, then
//!    composited per pixel exactly like [`SoftwareRasteriser::render_gaussian`],
//!    with one addition: at each pixel a splat fragment is only allowed to
//!    contribute if its `cam_z` is in front of (less than) the mesh depth already
//!    stored in the framebuffer. A fragment behind the mesh is rejected. The
//!    accumulated splat spectrum then composites *over* the mesh background using
//!    the residual transmittance `(1 - coverage)`.
//!
//! The result is a single frame where splats and meshes occlude each other
//! correctly through one shared depth buffer.
//!
//! # GPU generalisation
//!
//! This maps cleanly onto a GPU hybrid pass: the mesh pass becomes a normal
//! rasterised depth+gbuffer prepass writing a depth texture; the splat pass
//! becomes the existing tiled 3DGS compute/raster path with an added per-fragment
//! depth-texture sample and `discard`/attenuate against it, then an `over` blend
//! against the mesh gbuffer. The depth convention (`cam_z`) and the
//! reject-if-behind rule are identical; only the storage moves from `Vec` to GPU
//! textures.

use half::f16;
use vox_core::spectral::{Illuminant, SpectralBands};
use vox_core::types::GaussianSplat;

use spectra_gaussian_render::renderer::{
    project_gaussian, Gaussian3D, GaussianCamera, ALPHA_THRESHOLD, TRANSMITTANCE_THRESHOLD,
};

use crate::gpu::software_rasteriser::build_gaussian_camera;
use crate::spectral::RenderCamera;
use crate::spectral_framebuffer::SpectralFramebuffer;

/// A triangle mesh with engine-agnostic geometry and per-mesh spectral
/// reflectance. Positions are world-space; indices are triangle list (groups of
/// three indices into `positions`).
#[derive(Debug, Clone)]
pub struct HybridMesh {
    /// World-space vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Triangle-list indices into `positions` (length should be a multiple of 3).
    pub indices: Vec<u32>,
    /// 16-band spectral reflectance applied to the whole mesh.
    pub reflectance: [f32; 16],
    /// Object/entity id written to the framebuffer for these pixels.
    pub object_id: u32,
}

impl HybridMesh {
    /// Build a mesh whose reflectance comes from an RGB colour (uplifted to a
    /// 16-band reflectance via [`vox_core::spectral::rgb_to_spectral`]).
    pub fn from_rgb(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        rgb: [f32; 3],
        object_id: u32,
    ) -> Self {
        let bits = vox_core::spectral::rgb_to_spectral(rgb[0], rgb[1], rgb[2]);
        let reflectance = std::array::from_fn(|i| f16::from_bits(bits[i]).to_f32());
        Self {
            positions,
            indices,
            reflectance,
            object_id,
        }
    }
}

/// A scene of triangle meshes and Gaussian splats to be composited together.
pub struct HybridScene<'a> {
    pub meshes: Vec<HybridMesh>,
    pub splats: &'a [GaussianSplat],
}

/// Per-render statistics, chiefly for hardening visibility.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HybridStats {
    /// Triangles skipped because an index was out of range.
    pub skipped_oob_index: usize,
    /// Triangles skipped because a vertex was non-finite (NaN/inf).
    pub skipped_nonfinite: usize,
    /// Triangles skipped because they were degenerate (zero area / behind near).
    pub skipped_degenerate: usize,
    /// Triangles actually rasterised (at least one covered pixel attempted).
    pub triangles_drawn: usize,
}

impl HybridStats {
    /// Total number of triangles rejected by the hardening checks.
    pub fn warnings(&self) -> usize {
        self.skipped_oob_index + self.skipped_nonfinite + self.skipped_degenerate
    }
}

/// Direction *toward* the sun (will be normalised). Used for Lambertian shading.
#[derive(Debug, Clone, Copy)]
pub struct SunLight {
    pub direction: [f32; 3],
    /// Ambient fraction added to the diffuse term (keeps back-faces non-black).
    pub ambient: f32,
}

impl Default for SunLight {
    fn default() -> Self {
        Self {
            direction: [0.3, 0.8, 0.5],
            ambient: 0.15,
        }
    }
}

/// Render a hybrid scene into `fb` with default sun lighting.
///
/// Returns hardening statistics. The framebuffer is **not** cleared first; the
/// caller controls accumulation. To render a fresh frame, call `fb.clear()`
/// beforehand.
pub fn render_hybrid(
    scene: &HybridScene,
    camera: &RenderCamera,
    illuminant: &Illuminant,
    fb: &mut SpectralFramebuffer,
) -> HybridStats {
    render_hybrid_lit(scene, camera, illuminant, &SunLight::default(), fb)
}

/// Render a hybrid scene with an explicit sun light.
pub fn render_hybrid_lit(
    scene: &HybridScene,
    camera: &RenderCamera,
    illuminant: &Illuminant,
    sun: &SunLight,
    fb: &mut SpectralFramebuffer,
) -> HybridStats {
    let width = fb.width as usize;
    let height = fb.height as usize;
    let gcam = build_gaussian_camera(camera, width, height);

    // -- Pass 1: meshes into the shared spectral + depth buffer. -------------
    let stats = rasterise_meshes(&scene.meshes, &gcam, sun, fb);

    // -- Pass 2: splats composited over the mesh, depth-tested per pixel. -----
    composite_splats(scene.splats, &gcam, illuminant, fb);

    stats
}

/// Normalise a sun direction, guarding against a zero vector.
fn normalised(d: [f32; 3]) -> [f32; 3] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len > 1e-8 {
        [d[0] / len, d[1] / len, d[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Transform a world vertex into camera space (spectra convention). Returns
/// `(cam_x, cam_y, cam_z)` or `None` if any component is non-finite.
fn to_camera_space(p: [f32; 3], gcam: &GaussianCamera) -> Option<[f32; 3]> {
    let v = &gcam.view_matrix;
    let cam_x = v[0] * p[0] + v[1] * p[1] + v[2] * p[2] + v[3];
    let cam_y = v[4] * p[0] + v[5] * p[1] + v[6] * p[2] + v[7];
    let cam_z = v[8] * p[0] + v[9] * p[1] + v[10] * p[2] + v[11];
    if !(cam_x.is_finite() && cam_y.is_finite() && cam_z.is_finite()) {
        return None;
    }
    Some([cam_x, cam_y, cam_z])
}

/// Project a camera-space vertex to screen. Returns `(screen_x, screen_y, cam_z)`
/// or `None` if behind the near plane or beyond the far plane.
fn project_cam(cam: [f32; 3], gcam: &GaussianCamera) -> Option<(f32, f32, f32)> {
    let cam_z = cam[2];
    if cam_z < gcam.near || cam_z > gcam.far {
        return None;
    }
    let inv_z = 1.0 / cam_z;
    let sx = gcam.fx * cam[0] * inv_z + gcam.width as f32 * 0.5;
    let sy = gcam.fy * cam[1] * inv_z + gcam.height as f32 * 0.5;
    Some((sx, sy, cam_z))
}

/// Sutherland–Hodgman clip of a camera-space triangle against the near plane
/// (`cam_z >= near`). Returns the in-front polygon's vertices (0, 3, or 4 of
/// them), fanned by the caller into 1 or 2 sub-triangles. Vertices on a clipped
/// edge are linearly interpolated to the exact `cam_z == near` crossing, which
/// preserves the planar surface (the interpolation is in camera space, so the
/// recovered screen positions and perspective depth are exact at the boundary).
fn clip_triangle_near(tri: [[f32; 3]; 3], near: f32) -> Vec<[f32; 3]> {
    let inside = |v: &[f32; 3]| v[2] >= near;
    let intersect = |a: &[f32; 3], b: &[f32; 3]| -> [f32; 3] {
        // Parametric crossing of the near plane along a->b in cam_z.
        let t = (near - a[2]) / (b[2] - a[2]);
        [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            near,
        ]
    };
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(4);
    for i in 0..3 {
        let cur = tri[i];
        let prev = tri[(i + 2) % 3];
        let cur_in = inside(&cur);
        let prev_in = inside(&prev);
        if cur_in {
            if !prev_in {
                out.push(intersect(&prev, &cur));
            }
            out.push(cur);
        } else if prev_in {
            out.push(intersect(&prev, &cur));
        }
    }
    out
}

/// Edge function (signed area ×2) for the triangle (a, b) and point p in 2D.
#[inline]
fn edge(a: (f32, f32), b: (f32, f32), px: f32, py: f32) -> f32 {
    (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
}

fn rasterise_meshes(
    meshes: &[HybridMesh],
    gcam: &GaussianCamera,
    sun: &SunLight,
    fb: &mut SpectralFramebuffer,
) -> HybridStats {
    let mut stats = HybridStats::default();
    let sun_dir = normalised(sun.direction);
    let ambient = sun.ambient.clamp(0.0, 1.0);
    let width = fb.width as i32;
    let height = fb.height as i32;

    for mesh in meshes {
        let nverts = mesh.positions.len();
        for tri in mesh.indices.chunks(3) {
            if tri.len() < 3 {
                stats.skipped_degenerate += 1;
                continue;
            }
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= nverts || i1 >= nverts || i2 >= nverts {
                stats.skipped_oob_index += 1;
                continue;
            }
            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];

            // Hostile/broken geometry: a NaN/inf vertex would poison the raster.
            let finite = |v: [f32; 3]| v[0].is_finite() && v[1].is_finite() && v[2].is_finite();
            if !(finite(p0) && finite(p1) && finite(p2)) {
                stats.skipped_nonfinite += 1;
                continue;
            }

            // World-space normal (flat shading) from the geometric face.
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let nrm = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let nlen = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if nlen <= 1e-12 {
                // Degenerate (collinear) triangle: no area to shade.
                stats.skipped_degenerate += 1;
                continue;
            }
            let world_n = [nrm[0] / nlen, nrm[1] / nlen, nrm[2] / nlen];

            // Two-sided lighting: light whichever face we see.
            let ndl = sun_dir[0] * world_n[0] + sun_dir[1] * world_n[1] + sun_dir[2] * world_n[2];
            let diffuse = ndl.abs().clamp(0.0, 1.0);
            let shade = (ambient + (1.0 - ambient) * diffuse).clamp(0.0, 1.0);
            let shaded: [f32; 16] = std::array::from_fn(|k| mesh.reflectance[k] * shade);
            // The shaded face normal as written to the gbuffer (front-facing).
            let stored_n = if ndl >= 0.0 {
                world_n
            } else {
                [-world_n[0], -world_n[1], -world_n[2]]
            };

            // Transform to camera space, then clip against the near plane BEFORE
            // projection. A triangle straddling the near plane is split into the
            // in-front polygon (3 or 4 verts) instead of being dropped whole, so
            // its visible portion still rasterises and still occludes splats.
            let (Some(ca), Some(cb), Some(cc)) = (
                to_camera_space(p0, gcam),
                to_camera_space(p1, gcam),
                to_camera_space(p2, gcam),
            ) else {
                // Non-finite after transform.
                stats.skipped_degenerate += 1;
                continue;
            };
            let clipped = clip_triangle_near([ca, cb, cc], gcam.near);
            if clipped.len() < 3 {
                // Entirely behind the near plane.
                stats.skipped_degenerate += 1;
                continue;
            }

            // Project the clipped polygon's vertices; fan it into triangles.
            let mut proj: Vec<(f32, f32, f32)> = Vec::with_capacity(clipped.len());
            let mut all_projected = true;
            for cv in &clipped {
                match project_cam(*cv, gcam) {
                    Some(s) => proj.push(s),
                    None => {
                        all_projected = false;
                        break;
                    }
                }
            }
            if !all_projected {
                // A clipped vertex fell beyond the far plane (or otherwise
                // unprojectable); drop this triangle conservatively.
                stats.skipped_degenerate += 1;
                continue;
            }

            let mut any_drawn = false;
            // Fan: (0, i, i+1) for i in 1..len-1 — 1 sub-tri for a triangle, 2 for
            // a quad produced by the near clip.
            for i in 1..proj.len() - 1 {
                let a = proj[0];
                let b = proj[i];
                let c = proj[i + 1];
                let (sa, sb, sc) = ((a.0, a.1), (b.0, b.1), (c.0, c.1));
                let area2 = edge(sa, sb, sc.0, sc.1);
                if area2.abs() < 1e-6 {
                    // Zero screen-space area (edge-on / degenerate sub-triangle).
                    continue;
                }
                let inv_area2 = 1.0 / area2;

                // Screen-space bounding box, clamped to the framebuffer.
                let min_x = (sa.0.min(sb.0).min(sc.0).floor() as i32).max(0);
                let max_x = (sa.0.max(sb.0).max(sc.0).ceil() as i32).min(width - 1);
                let min_y = (sa.1.min(sb.1).min(sc.1).floor() as i32).max(0);
                let max_y = (sa.1.max(sb.1).max(sc.1).ceil() as i32).min(height - 1);
                if min_x > max_x || min_y > max_y {
                    continue;
                }
                any_drawn = true;

                for py in min_y..=max_y {
                    let pyf = py as f32 + 0.5;
                    for px in min_x..=max_x {
                        let pxf = px as f32 + 0.5;
                        // Barycentric coordinates via edge functions.
                        let w0 = edge(sb, sc, pxf, pyf) * inv_area2;
                        let w1 = edge(sc, sa, pxf, pyf) * inv_area2;
                        let w2 = edge(sa, sb, pxf, pyf) * inv_area2;
                        // Inside test: all weights same sign as area (>= 0 here
                        // since we normalised by signed area).
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        // Perspective-correct depth: interpolate INVERSE depth
                        // (1/cam_z is exactly linear in screen space), then invert
                        // to recover the true perspective cam_z. This matches the
                        // splats' true `proj.depth`, so occlusion ordering is
                        // correct even on triangles spanning a large depth range.
                        let inv_z = w0 / a.2 + w1 / b.2 + w2 / c.2;
                        if inv_z <= 0.0 {
                            continue;
                        }
                        let depth = 1.0 / inv_z;
                        let idx = (py as u32 * fb.width + px as u32) as usize;
                        // Z-test against the shared depth buffer (nearest wins).
                        if depth >= fb.depth[idx] {
                            continue;
                        }
                        fb.depth[idx] = depth;
                        fb.spectral[idx] = shaded;
                        fb.albedo[idx] = mesh.reflectance;
                        fb.normals[idx] = stored_n;
                        fb.object_id[idx] = mesh.object_id;
                        fb.sample_count[idx] = 1;
                    }
                }
            }

            if any_drawn {
                stats.triangles_drawn += 1;
            } else {
                stats.skipped_degenerate += 1;
            }
        }
    }

    stats
}

fn composite_splats(
    splats: &[GaussianSplat],
    gcam: &GaussianCamera,
    illuminant: &Illuminant,
    fb: &mut SpectralFramebuffer,
) {
    if splats.is_empty() {
        return;
    }
    let width = fb.width as usize;
    let height = fb.height as usize;

    struct SplatRecord {
        screen_pos: [f32; 2],
        conic: [f32; 3],
        radius: f32,
        depth: f32,
        spectral: [f32; 16],
        opacity: f32,
    }

    let mut records: Vec<SplatRecord> = Vec::with_capacity(splats.len());
    for splat in splats {
        let scales = splat.scales();
        let log_scale = [
            scales[0].max(1e-4).ln(),
            scales[1].max(1e-4).ln(),
            scales[2].max(1e-4).ln(),
        ];
        let q = splat.decoded_rotation();
        let g3d = Gaussian3D {
            position: splat.position(),
            log_scale,
            rotation: [q.w, q.x, q.y, q.z],
            color: [0.0, 0.0, 0.0],
            opacity: 1.0,
            sh_coeffs: None,
        };
        let Some(proj) = project_gaussian(&g3d, gcam) else {
            continue;
        };
        let spectral: [f32; 16] =
            std::array::from_fn(|i| f16::from_bits(splat.spectral()[i]).to_f32());
        records.push(SplatRecord {
            screen_pos: proj.screen_pos,
            conic: proj.conic,
            radius: proj.radius,
            depth: proj.depth,
            spectral,
            opacity: splat.opacity() as f32 / 255.0,
        });
    }
    if records.is_empty() {
        return;
    }

    // Front-to-back (nearest first) for correct transmittance accumulation.
    records.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Per-pixel accumulator + transmittance for the splat layer only.
    let mut accum = vec![[0.0f32; 16]; width * height];
    let mut transmittance = vec![1.0f32; width * height];
    // Nearest splat depth that actually contributed to each pixel (for the
    // shared depth buffer: a covering splat closer than the mesh updates depth).
    let mut splat_depth = vec![f32::MAX; width * height];

    for rec in &records {
        let cy = rec.screen_pos[1];
        let cx = rec.screen_pos[0];
        let y_min = ((cy - rec.radius).floor() as i32).max(0);
        let y_max = ((cy + rec.radius).ceil() as i32).min(height as i32 - 1);
        let x_min = ((cx - rec.radius).floor() as i32).max(0);
        let x_max = ((cx + rec.radius).ceil() as i32).min(width as i32 - 1);
        if y_min > y_max || x_min > x_max {
            continue;
        }
        for py in y_min..=y_max {
            let dy = py as f32 + 0.5 - cy;
            let row = py as usize * width;
            for px in x_min..=x_max {
                let pidx = row + px as usize;
                let t = transmittance[pidx];
                if t < TRANSMITTANCE_THRESHOLD {
                    continue;
                }
                // DEPTH-CORRECT OCCLUSION: a splat fragment behind the mesh
                // depth stored in the framebuffer is occluded — reject it.
                if rec.depth >= fb.depth[pidx] {
                    continue;
                }
                let dx = px as f32 + 0.5 - cx;
                let power = -0.5
                    * (rec.conic[0] * dx * dx
                        + 2.0 * rec.conic[1] * dx * dy
                        + rec.conic[2] * dy * dy);
                if power > 0.0 {
                    continue;
                }
                let alpha = (rec.opacity * power.exp()).min(0.99);
                if alpha < ALPHA_THRESHOLD {
                    continue;
                }
                let weight = alpha * t;
                let acc = &mut accum[pidx];
                for (av, &s) in acc.iter_mut().zip(rec.spectral.iter()) {
                    *av += weight * s;
                }
                transmittance[pidx] = t * (1.0 - alpha);
                if rec.depth < splat_depth[pidx] {
                    splat_depth[pidx] = rec.depth;
                }
            }
        }
    }

    // Resolve: composite the accumulated splat spectrum OVER the mesh spectrum
    // already in the framebuffer using residual transmittance.
    let _ = illuminant; // bands are kept linear; illuminant applies at display.
    for pidx in 0..width * height {
        let cov = 1.0 - transmittance[pidx];
        if cov <= 0.0 {
            continue; // no splat contribution here — mesh (or empty) stands.
        }
        // out = splat_accum + transmittance * mesh_background.
        let bg = fb.spectral[pidx];
        let acc = accum[pidx];
        let mut out = [0.0f32; 16];
        for k in 0..16 {
            out[k] = acc[k] + transmittance[pidx] * bg[k];
        }
        fb.spectral[pidx] = out;
        // If a splat covered this pixel and is nearer than the mesh, the visible
        // surface depth is the splat's.
        if splat_depth[pidx] < fb.depth[pidx] {
            fb.depth[pidx] = splat_depth[pidx];
        }
        if fb.sample_count[pidx] == 0 {
            fb.sample_count[pidx] = 1;
        }
    }
}

/// Resolve one framebuffer pixel's spectrum to display sRGB (8-bit).
pub fn resolve_srgb(spectral: &[f32; 16], illuminant: &Illuminant) -> [u8; 3] {
    use vox_core::spectral::{linear_to_srgb_gamma, spectral_to_xyz, xyz_to_srgb};
    let bands = SpectralBands(*spectral);
    let xyz = spectral_to_xyz(&bands, illuminant);
    let lin = xyz_to_srgb(xyz);
    [
        (linear_to_srgb_gamma(lin[0]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (linear_to_srgb_gamma(lin[1]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (linear_to_srgb_gamma(lin[2]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat, Vec3};

    const W: u32 = 64;
    const H: u32 = 64;

    fn head_on_camera() -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 0.0, 20.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                W as f32 / H as f32,
                0.1,
                500.0,
            ),
        }
    }

    fn illum() -> Illuminant {
        Illuminant::d65()
    }

    fn single_band(band: usize, value: f32) -> [u16; 16] {
        let mut s = [f16::from_f32(0.0).to_bits(); 16];
        s[band] = f16::from_f32(value).to_bits();
        s
    }

    fn single_band_f32(band: usize, value: f32) -> [f32; 16] {
        let mut s = [0.0f32; 16];
        s[band] = value;
        s
    }

    /// A 1×1 quad (two triangles) in the XY plane at world z = `z`, facing the
    /// camera, spanning [-half, +half] in x and y.
    fn quad(z: f32, half: f32, refl: [f32; 16], object_id: u32) -> HybridMesh {
        HybridMesh {
            positions: vec![
                [-half, -half, z],
                [half, -half, z],
                [half, half, z],
                [-half, half, z],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            reflectance: refl,
            object_id,
        }
    }

    /// Sum of all band energy in a screen region.
    fn region_band_sums(
        fb: &SpectralFramebuffer,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> [f32; 16] {
        let mut sums = [0.0f32; 16];
        for y in y0..y1 {
            for x in x0..x1 {
                let s = fb.spectral[fb.idx(x, y)];
                for k in 0..16 {
                    sums[k] += s[k];
                }
            }
        }
        sums
    }

    // A splat large enough to robustly cover the image centre.
    fn big_splat(z: f32, band: usize, value: f32, opacity: u8) -> GaussianSplat {
        GaussianSplat::volume(
            [0.0, 0.0, z],
            [3.0, 3.0, 3.0],
            Quat::IDENTITY,
            opacity,
            single_band(band, value),
        )
    }

    /// Occlusion both ways. A wall mesh (band 11 / "red") at world z=0 with a
    /// splat (band 3 / "blue") behind it at z=-8: the splat must NOT be visible
    /// through the wall (mesh band dominates the wall region). Then move the
    /// splat in front (z=+8): the splat band now dominates.
    #[test]
    fn occlusion_both_ways() {
        let cam = head_on_camera();
        let il = illum();

        // Mesh wall at z=0, red-ish reflectance only in band 11.
        let wall = quad(0.0, 4.0, single_band_f32(11, 1.0), 7);

        // --- Splat BEHIND the wall (z = -8, farther from +Z eye). ---
        let behind = big_splat(-8.0, 3, 1.0, 255);
        let mut fb = SpectralFramebuffer::new(W, H);
        let stats = render_hybrid(
            &HybridScene {
                meshes: vec![wall.clone()],
                splats: std::slice::from_ref(&behind),
            },
            &cam,
            &il,
            &mut fb,
        );
        assert_eq!(stats.warnings(), 0, "clean scene must not warn");
        let sums_behind = region_band_sums(&fb, 24, 24, 40, 40);

        // --- Splat IN FRONT of the wall (z = +8, nearer the eye). ---
        let front = big_splat(8.0, 3, 1.0, 255);
        let mut fb2 = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![wall],
                splats: std::slice::from_ref(&front),
            },
            &cam,
            &il,
            &mut fb2,
        );
        let sums_front = region_band_sums(&fb2, 24, 24, 40, 40);

        // BEHIND: mesh band 11 dominates, splat band 3 is suppressed.
        assert!(
            sums_behind[11] > 10.0 * sums_behind[3].max(1e-6),
            "occluded splat must not show through wall: band11={} band3={}",
            sums_behind[11],
            sums_behind[3]
        );
        assert!(
            sums_behind[3] < 1e-3,
            "behind splat band 3 should be ~zero in wall region, got {}",
            sums_behind[3]
        );

        // IN FRONT: splat band 3 now dominates the centre region.
        assert!(
            sums_front[3] > sums_front[11],
            "front splat band 3 must dominate: band3={} band11={}",
            sums_front[3],
            sums_front[11]
        );
        assert!(
            sums_front[3] > 1.0,
            "front splat band 3 should be strongly present, got {}",
            sums_front[3]
        );

        // Surface as the report's headline numbers.
        eprintln!(
            "occlusion_both_ways: behind band11={:.4} band3={:.6} | front band3={:.4} band11={:.6}",
            sums_behind[11], sums_behind[3], sums_front[3], sums_front[11]
        );
    }

    /// Partial overlap. A wall covering only the LEFT half of the view, with a
    /// centred splat behind it. The splat's lit fragments must appear on the
    /// RIGHT (uncovered) half only — assert a strong left/right asymmetry.
    #[test]
    fn partial_overlap_left_right_asymmetry() {
        let cam = head_on_camera();
        let il = illum();

        // Wall occupies x in [-8, 0] (left half of the centred view) at z=0, tall
        // enough to cover the full vertical extent of the splat.
        let wall = HybridMesh {
            positions: vec![
                [-8.0, -8.0, 0.0],
                [0.0, -8.0, 0.0],
                [0.0, 8.0, 0.0],
                [-8.0, 8.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            reflectance: single_band_f32(11, 1.0),
            object_id: 1,
        };
        // Splat behind the wall, centred, band 3, wide enough to span both halves.
        let splat = big_splat(-8.0, 3, 1.0, 255);

        let mut fb = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![wall],
                splats: std::slice::from_ref(&splat),
            },
            &cam,
            &il,
            &mut fb,
        );

        // Count splat-signature (band 3 dominant) pixels on each half.
        let mut left = 0usize;
        let mut right = 0usize;
        for y in 0..H {
            for x in 0..W {
                let s = fb.spectral[fb.idx(x, y)];
                if s[3] > 0.05 && s[3] > s[11] {
                    if x < W / 2 {
                        left += 1;
                    } else {
                        right += 1;
                    }
                }
            }
        }
        eprintln!("partial_overlap: left_splat_px={left} right_splat_px={right}");
        assert!(
            right > 5 * left.max(1),
            "splat should be visible mostly on the uncovered right half: left={left} right={right}"
        );
        assert!(right > 20, "uncovered half should have many splat pixels: {right}");
    }

    /// Depth correctness: at a pixel where both a near mesh and a farther splat
    /// project, the stored depth equals the (smaller) mesh cam_z, not the splat's.
    #[test]
    fn depth_buffer_keeps_nearest() {
        let cam = head_on_camera();
        let il = illum();

        // Mesh near the eye (z=+5 -> cam_z ~ 15), splat far (z=-5 -> cam_z ~ 25).
        let mesh = quad(5.0, 4.0, single_band_f32(11, 1.0), 1);
        let splat = big_splat(-5.0, 3, 1.0, 255);

        let mut fb = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![mesh],
                splats: std::slice::from_ref(&splat),
            },
            &cam,
            &il,
            &mut fb,
        );
        let c = fb.idx(W / 2, H / 2);
        let depth = fb.depth[c];
        // Eye at z=20 looking at origin; mesh at world z=5 -> cam_z = 20-5 = 15.
        assert!(
            (depth - 15.0).abs() < 0.5,
            "centre depth should be the near mesh cam_z ~15, got {depth}"
        );
        // And the splat (cam_z ~25) must not have overwritten it.
        assert!(depth < 20.0, "near mesh depth must beat far splat: {depth}");
        eprintln!("depth_buffer_keeps_nearest: centre depth={depth:.4}");
    }

    /// Spectral integrity: red-band mesh + blue-band splat. With the splat in
    /// front both signatures appear; with the splat occluded behind the mesh the
    /// blue contribution is removed.
    #[test]
    fn spectral_integrity_both_signatures() {
        let cam = head_on_camera();
        let il = illum();

        // Small red mesh wall (band 11), only covering part of the frame so the
        // splat (band 3) also has uncovered area when in front.
        let wall = quad(0.0, 1.5, single_band_f32(11, 1.0), 1);
        let front = big_splat(8.0, 3, 1.0, 200);
        let behind = big_splat(-8.0, 3, 1.0, 200);

        let mut fb_front = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![wall.clone()],
                splats: std::slice::from_ref(&front),
            },
            &cam,
            &il,
            &mut fb_front,
        );
        let mut fb_behind = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![wall],
                splats: std::slice::from_ref(&behind),
            },
            &cam,
            &il,
            &mut fb_behind,
        );

        let f = region_band_sums(&fb_front, 0, 0, W, H);
        let b = region_band_sums(&fb_behind, 0, 0, W, H);

        // Front: both red (band 11, from uncovered mesh edge + splat-over) and
        // blue (band 3) present.
        assert!(f[11] > 0.5, "mesh red band must be present (front): {}", f[11]);
        assert!(f[3] > 1.0, "splat blue band must be present (front): {}", f[3]);
        // Behind: the splat is occluded by the wall over the wall area, so blue
        // is strictly smaller than when the splat is in front.
        assert!(
            b[3] < f[3],
            "occluded config must have less blue energy: behind={} front={}",
            b[3],
            f[3]
        );
        eprintln!(
            "spectral_integrity: front[red11]={:.4} front[blue3]={:.4} | behind[blue3]={:.4}",
            f[11], f[3], b[3]
        );
    }

    /// Hostile mesh input: out-of-range indices + NaN vertices. Render must
    /// complete with warnings > 0 and no panic.
    #[test]
    fn hostile_mesh_input() {
        let cam = head_on_camera();
        let il = illum();

        let hostile = HybridMesh {
            positions: vec![
                [0.0, 0.0, 0.0],
                [f32::NAN, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            // First triangle references index 99 (OOB); second has a NaN vertex.
            indices: vec![0, 1, 99, 0, 1, 2],
            reflectance: single_band_f32(11, 1.0),
            object_id: 1,
        };
        let splat = big_splat(8.0, 3, 1.0, 255);

        let mut fb = SpectralFramebuffer::new(W, H);
        let stats = render_hybrid(
            &HybridScene {
                meshes: vec![hostile],
                splats: std::slice::from_ref(&splat),
            },
            &cam,
            &il,
            &mut fb,
        );
        assert!(
            stats.warnings() > 0,
            "hostile input must produce warnings, got {stats:?}"
        );
        assert_eq!(stats.skipped_oob_index, 1, "one OOB triangle expected");
        assert_eq!(stats.skipped_nonfinite, 1, "one NaN triangle expected");
        // The splat still renders fine — frame is not poisoned.
        let sums = region_band_sums(&fb, 24, 24, 40, 40);
        assert!(sums[3] > 0.5, "splat should still render: band3={}", sums[3]);
    }

    /// End-to-end proof: a procedural cube above a splat ground carpet. Render,
    /// then assert many lit pixels with BOTH signatures present.
    #[test]
    fn cube_over_splat_carpet_end_to_end() {
        let cam = RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 6.0, 18.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                W as f32 / H as f32,
                0.1,
                500.0,
            ),
        };
        let il = illum();

        // Procedural unit cube centred at origin, side 4, red reflectance (band 11).
        let cube = cube_mesh([0.0, 0.0, 0.0], 2.0, single_band_f32(11, 1.0), 1);

        // Splat ground carpet: a grid of blue (band 3) splats on the y=-3 plane,
        // spread across x/z so they fill the lower frame behind/around the cube.
        let mut carpet = Vec::new();
        for gx in -3..=3 {
            for gz in -3..=3 {
                carpet.push(GaussianSplat::volume(
                    [gx as f32 * 2.0, -3.0, gz as f32 * 2.0],
                    [1.2, 0.2, 1.2],
                    Quat::IDENTITY,
                    220,
                    single_band(3, 1.0),
                ));
            }
        }

        let mut fb = SpectralFramebuffer::new(W, H);
        let stats = render_hybrid(
            &HybridScene {
                meshes: vec![cube],
                splats: &carpet,
            },
            &cam,
            &il,
            &mut fb,
        );
        assert_eq!(stats.warnings(), 0, "clean cube scene must not warn");
        assert!(stats.triangles_drawn >= 6, "cube faces should draw: {stats:?}");

        // Count lit pixels and which signature dominates.
        let mut lit = 0usize;
        let mut red_px = 0usize;
        let mut blue_px = 0usize;
        for y in 0..H {
            for x in 0..W {
                let s = fb.spectral[fb.idx(x, y)];
                let energy: f32 = s.iter().sum();
                if energy > 1e-3 {
                    lit += 1;
                    if s[11] > s[3] {
                        red_px += 1;
                    } else if s[3] > s[11] {
                        blue_px += 1;
                    }
                }
            }
        }
        eprintln!("cube_over_carpet: lit={lit} red(cube)={red_px} blue(carpet)={blue_px}");
        assert!(lit > 200, "expected many lit pixels, got {lit}");
        assert!(red_px > 30, "cube (red) signature must be present: {red_px}");
        assert!(blue_px > 30, "carpet (blue) signature must be present: {blue_px}");
    }

    /// Build an axis-aligned cube of half-extent `h` centred at `c` with a single
    /// reflectance. 8 verts, 12 triangles (CCW outward winding not required —
    /// shading is two-sided).
    fn cube_mesh(c: [f32; 3], h: f32, refl: [f32; 16], object_id: u32) -> HybridMesh {
        let [cx, cy, cz] = c;
        let positions = vec![
            [cx - h, cy - h, cz - h], // 0
            [cx + h, cy - h, cz - h], // 1
            [cx + h, cy + h, cz - h], // 2
            [cx - h, cy + h, cz - h], // 3
            [cx - h, cy - h, cz + h], // 4
            [cx + h, cy - h, cz + h], // 5
            [cx + h, cy + h, cz + h], // 6
            [cx - h, cy + h, cz + h], // 7
        ];
        let indices = vec![
            // -Z face
            0, 1, 2, 0, 2, 3, // +Z face
            4, 6, 5, 4, 7, 6, // -X face
            0, 3, 7, 0, 7, 4, // +X face
            1, 5, 6, 1, 6, 2, // -Y face
            0, 4, 5, 0, 5, 1, // +Y face
            3, 2, 6, 3, 6, 7,
        ];
        HybridMesh {
            positions,
            indices,
            reflectance: refl,
            object_id,
        }
    }

    /// A steep triangle slanting from a near apex (cam_z ~1) to a far base
    /// (cam_z ~100). Eye at z=20 looking at origin, so cam_z = 20 - world_z.
    fn steep_tri(refl: [f32; 16], object_id: u32) -> HybridMesh {
        HybridMesh {
            positions: vec![
                [0.0, 0.0, 19.0],     // cam_z = 1   (near apex)
                [-30.0, 20.0, -80.0], // cam_z = 100 (far base)
                [30.0, 20.0, -80.0],  // cam_z = 100 (far base)
            ],
            indices: vec![0, 1, 2],
            reflectance: refl,
            object_id,
        }
    }

    /// Regression for the perspective-correct depth fix (wave-6). On a steep
    /// triangle spanning cam_z {1..100}, the visible surface's TRUE perspective
    /// depth in the apex region is ~1.7 (probe: depth values 1.0–1.9), while the
    /// OLD linear interpolation stored ~40 there. A splat at true cam_z 21.1 is
    /// GENUINELY BEHIND that surface (21.1 > 1.7) and MUST be rejected — but under
    /// the old linear depth (40.6 > 21.1) it wrongly composited over the mesh.
    /// A splat genuinely in FRONT (cam_z 0.8 < 1.7) must still composite.
    #[test]
    fn perspective_depth_rejects_splat_behind_steep_surface() {
        let cam = head_on_camera();
        let il = illum();

        // Mesh: band 11 ("red"). Splat: band 3 ("blue").
        let tri = steep_tri(single_band_f32(11, 1.0), 1);

        // Render the mesh alone first to find exactly which pixels the visible
        // surface covers (so we measure splat leakage ONLY over true mesh pixels,
        // never over empty background the big splat also overlaps).
        let mut fb_m = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene { meshes: vec![tri.clone()], splats: &[] },
            &cam,
            &il,
            &mut fb_m,
        );
        let mesh_px: Vec<(u32, u32)> = (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .filter(|&(x, y)| fb_m.spectral[fb_m.idx(x, y)][11] > 1e-3)
            .collect();
        assert!(!mesh_px.is_empty(), "mesh must cover some pixels");

        let band_sums_over = |fb: &SpectralFramebuffer| -> [f32; 16] {
            let mut s = [0.0f32; 16];
            for &(x, y) in &mesh_px {
                let px = fb.spectral[fb.idx(x, y)];
                for k in 0..16 {
                    s[k] += px[k];
                }
            }
            s
        };

        // --- Splat BEHIND the true surface: true cam_z = 21.1 (world_z = -1.1),
        // big enough to cover the apex region. Old linear depth (~40) would have
        // wrongly let it through; correct perspective depth (~1.7) rejects it.
        let behind = GaussianSplat::volume(
            [0.0, -3.0, -1.1], // cam_z = 20 - (-1.1) = 21.1
            [6.0, 6.0, 6.0],
            Quat::IDENTITY,
            255,
            single_band(3, 1.0),
        );
        let mut fb_b = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![tri.clone()],
                splats: std::slice::from_ref(&behind),
            },
            &cam,
            &il,
            &mut fb_b,
        );
        let b = band_sums_over(&fb_b);

        // --- Splat IN FRONT of the surface: true cam_z = 0.8 (world_z = 19.2),
        // nearer than the apex's ~1.0–1.9; must composite.
        let front = GaussianSplat::volume(
            [0.0, -3.0, 19.2], // cam_z = 20 - 19.2 = 0.8
            [6.0, 6.0, 6.0],
            Quat::IDENTITY,
            255,
            single_band(3, 1.0),
        );
        let mut fb_f = SpectralFramebuffer::new(W, H);
        render_hybrid(
            &HybridScene {
                meshes: vec![tri],
                splats: std::slice::from_ref(&front),
            },
            &cam,
            &il,
            &mut fb_f,
        );
        let f = band_sums_over(&fb_f);

        eprintln!(
            "perspective_depth: BEHIND mesh11={:.4} splat3={:.6} | FRONT mesh11={:.6} splat3={:.4}",
            b[11], b[3], f[11], f[3]
        );

        // BEHIND: the splat is correctly REJECTED — its band 3 is ~zero in the
        // mesh region and the mesh band 11 dominates. (Under the old linear depth
        // this band 3 would be large and composite over the mesh.)
        assert!(
            b[11] > 1.0,
            "mesh surface must be present in apex region: band11={}",
            b[11]
        );
        assert!(
            b[3] < 1e-3,
            "splat at cam_z 21.1 is behind true surface ~1.7 and MUST be rejected, got band3={}",
            b[3]
        );
        assert!(
            b[11] > 100.0 * b[3].max(1e-9),
            "rejected splat must not show through mesh: band11={} band3={}",
            b[11],
            b[3]
        );

        // FRONT: a genuinely nearer splat still composites (band 3 present).
        assert!(
            f[3] > 1.0,
            "splat at cam_z 0.8 is in front of the surface and MUST composite, got band3={}",
            f[3]
        );
        assert!(
            f[3] > f[11],
            "front splat band 3 must dominate the mesh in this region: band3={} band11={}",
            f[3],
            f[11]
        );
    }

    /// Regression for the near-plane clipping fix (wave-6). A large ground quad
    /// extends from well in front of the camera to behind it (one edge straddles
    /// the near plane). The OLD code dropped any triangle with a vertex behind
    /// near entirely, so the quad rendered NOTHING; the clip path rasterises the
    /// in-front portion and still occludes a splat behind it.
    #[test]
    fn near_plane_clip_renders_visible_portion_and_occludes() {
        // Camera above the ground looking forward-and-down so the ground plane
        // recedes from in-front to behind the eye.
        let cam = RenderCamera {
            view: Mat4::look_at_rh(
                Vec3::new(0.0, 4.0, 0.0),
                Vec3::new(0.0, 0.0, -10.0),
                Vec3::Y,
            ),
            proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, W as f32 / H as f32, 0.1, 500.0),
        };
        let il = illum();

        // A big ground quad on the y=0 plane, spanning z from +20 (behind the
        // eye, which sits at z=0 looking toward -z) to -60 (far in front). Two
        // triangles; each straddles the near plane (part behind the camera).
        let ground = HybridMesh {
            positions: vec![
                [-40.0, 0.0, 20.0],  // behind the eye
                [40.0, 0.0, 20.0],   // behind the eye
                [40.0, 0.0, -60.0],  // far in front
                [-40.0, 0.0, -60.0], // far in front
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            reflectance: single_band_f32(11, 1.0),
            object_id: 1,
        };

        // Splat behind a chunk of the ground (below the plane, far out), band 3.
        let splat = GaussianSplat::volume(
            [0.0, -2.0, -30.0],
            [8.0, 2.0, 8.0],
            Quat::IDENTITY,
            255,
            single_band(3, 1.0),
        );

        let mut fb = SpectralFramebuffer::new(W, H);
        let stats = render_hybrid(
            &HybridScene {
                meshes: vec![ground],
                splats: std::slice::from_ref(&splat),
            },
            &cam,
            &il,
            &mut fb,
        );

        // Count lit mesh pixels (band 11 present). The old whole-triangle-drop
        // path produced ZERO here.
        let mut lit_mesh = 0usize;
        for y in 0..H {
            for x in 0..W {
                if fb.spectral[fb.idx(x, y)][11] > 1e-3 {
                    lit_mesh += 1;
                }
            }
        }
        eprintln!(
            "near_plane_clip: lit_mesh_px={lit_mesh} (old path=0) triangles_drawn={}",
            stats.triangles_drawn
        );
        assert!(
            lit_mesh > 50,
            "near-clipped ground must render its visible portion (old path rendered 0), got {lit_mesh}"
        );
        assert!(
            stats.triangles_drawn >= 1,
            "at least one ground sub-triangle must rasterise: {stats:?}"
        );

        // The ground still occludes the splat where it covers it: in the lower
        // frame (ground in front), band 11 (mesh) dominates band 3 (splat).
        let lower = region_band_sums(&fb, 0, H * 3 / 4, W, H);
        assert!(
            lower[11] > lower[3],
            "ground must occlude the splat behind it: mesh11={} splat3={}",
            lower[11],
            lower[3]
        );
    }

}
