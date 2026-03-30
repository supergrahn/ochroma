//! Spectra-compatible tile-based Gaussian splatting renderer.
//!
//! Reimplements the Kerbl et al. 2023 EWA splatting algorithm from
//! `spectra-gaussian-render` directly in Ochroma, bridging our spectral
//! `GaussianSplat` format to the renderer's internal `Gaussian3D` format.
//!
//! This avoids workspace conflicts with AetherSpectra while using an
//! identical algorithm: project -> 2D covariance via EWA -> tile-sort ->
//! front-to-back alpha blend.

use half::f16;
use rayon::prelude::*;
use vox_core::spectral::{spectral_to_xyz, xyz_to_srgb, Illuminant, SpectralBands};
use vox_core::types::GaussianSplat;

use crate::spectral::RenderCamera;

// ---------------------------------------------------------------------------
// Constants (matching spectra-gaussian-render/src/renderer.rs)
// ---------------------------------------------------------------------------

/// Tile size for tile-based rasterization.
const TILE_SIZE: usize = 16;
/// Alpha threshold for Gaussian contribution (1/255).
const ALPHA_THRESHOLD: f32 = 1.0 / 255.0;
/// Transmittance threshold for early termination.
const TRANSMITTANCE_THRESHOLD: f32 = 0.001;

// ---------------------------------------------------------------------------
// Internal Gaussian3D representation (matches Spectra)
// ---------------------------------------------------------------------------

/// A single 3D Gaussian primitive in Spectra-compatible format.
#[derive(Debug, Clone)]
struct Gaussian3D {
    position: [f32; 3],
    log_scale: [f32; 3],
    /// Quaternion rotation [w, x, y, z].
    rotation: [f32; 4],
    /// RGB color [r, g, b] in [0, 1].
    color: [f32; 3],
    /// Opacity in [0, 1].
    opacity: f32,
}

impl Gaussian3D {
    fn scales(&self) -> [f32; 3] {
        [
            self.log_scale[0].exp(),
            self.log_scale[1].exp(),
            self.log_scale[2].exp(),
        ]
    }
}

/// Projected 2D Gaussian for rasterization.
#[derive(Debug, Clone)]
struct ProjectedGaussian {
    screen_pos: [f32; 2],
    depth: f32,
    /// 2D covariance conic coefficients [a, b, c] where Q = ax^2 + 2bxy + cy^2.
    conic: [f32; 3],
    color: [f32; 3],
    opacity: f32,
    radius: f32,
    index: usize,
}

/// Camera parameters for the Spectra-style renderer.
#[derive(Debug, Clone)]
struct SpectraCamera {
    view_matrix: [f32; 16],
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    near: f32,
    far: f32,
}

// ---------------------------------------------------------------------------
// Conversion: Ochroma -> Spectra-compatible
// ---------------------------------------------------------------------------

/// Convert an Ochroma spectral GaussianSplat to internal Gaussian3D + RGB.
fn ochroma_to_gaussian3d(splat: &GaussianSplat, illuminant: &Illuminant) -> Gaussian3D {
    // Decode spectral bands to RGB
    let bands = SpectralBands(std::array::from_fn(|i| {
        f16::from_bits(splat.spectral()[i]).to_f32()
    }));
    let xyz = spectral_to_xyz(&bands, illuminant);
    let rgb = xyz_to_srgb(xyz);

    let (su, sv, sw) = match splat.kind() {
        0 => (splat.scale_u(), splat.scale_v(), 1e-4_f32), // 2DGS disk
        _ => (splat.scale_u(), splat.scale_v(), splat.scale_w()), // 3DGS ellipsoid
    };
    Gaussian3D {
        position: splat.position(),
        log_scale: [
            su.max(0.001).ln(),
            sv.max(0.001).ln(),
            sw.max(0.001).ln(),
        ],
        rotation: [
            // Ochroma stores [x, y, z, w] as i16; Spectra expects [w, x, y, z] as f32
            splat.rotation_raw()[3] as f32 / 32767.0, // w
            splat.rotation_raw()[0] as f32 / 32767.0, // x
            splat.rotation_raw()[1] as f32 / 32767.0, // y
            splat.rotation_raw()[2] as f32 / 32767.0, // z
        ],
        color: [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ],
        opacity: splat.opacity() as f32 / 255.0,
    }
}

/// Maps a scalar intensity [0, 1] to a 5-stop false-color gradient:
/// black → blue → cyan → green → yellow → red.
pub fn band_to_heatmap(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.2 {
        let s = t / 0.2;
        [0.0, 0.0, s]
    } else if t < 0.4 {
        let s = (t - 0.2) / 0.2;
        [0.0, s, 1.0]
    } else if t < 0.6 {
        let s = (t - 0.4) / 0.2;
        [0.0, 1.0, 1.0 - s]
    } else if t < 0.8 {
        let s = (t - 0.6) / 0.2;
        [s, 1.0, 0.0]
    } else {
        let s = (t - 0.8) / 0.2;
        [1.0, 1.0 - s, 0.0]
    }
}

/// Converts a GaussianSplat to Gaussian3D using one spectral band as false-color.
fn splat_to_gaussian3d_band(splat: &GaussianSplat, band: usize) -> Gaussian3D {
    let (su, sv, sw) = match splat.kind() {
        0 => (splat.scale_u(), splat.scale_v(), 1e-4_f32), // 2DGS disk
        _ => (splat.scale_u(), splat.scale_v(), splat.scale_w()), // 3DGS ellipsoid
    };
    let log_scale = [
        su.max(1e-6).ln(),
        sv.max(1e-6).ln(),
        sw.max(1e-6).ln(),
    ];
    let rot = splat.rotation_raw();
    let qx = rot[0] as f32 / 32767.0;
    let qy = rot[1] as f32 / 32767.0;
    let qz = rot[2] as f32 / 32767.0;
    let qw = rot[3] as f32 / 32767.0;
    let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt().max(1e-8);
    let rotation = [qw / len, qx / len, qy / len, qz / len];
    let opacity = splat.opacity() as f32 / 255.0;
    let intensity = splat.spectral_f32(band.min(15));
    let color = band_to_heatmap(intensity);
    Gaussian3D { position: splat.position(), log_scale, rotation, color, opacity }
}

/// Render the scene as a false-color heatmap of a single spectral band.
/// `band` must be 0..8; values out of range clamp to band 7.
/// Returns `width * height` RGBA pixels in the same format as `render_with_spectra_u8`.
pub fn render_spectral_band_u8(
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    width: u32,
    height: u32,
    band: usize,
) -> Vec<[u8; 4]> {
    let cam = ochroma_to_spectra_camera(camera, width, height);
    let gaussians: Vec<Gaussian3D> = splats
        .iter()
        .map(|s| splat_to_gaussian3d_band(s, band))
        .collect();
    let floats = render_cpu_internal(&gaussians, &cam, None);
    floats
        .chunks_exact(4)
        .map(|px| {
            [
                (px[0].clamp(0.0, 1.0) * 255.0) as u8,
                (px[1].clamp(0.0, 1.0) * 255.0) as u8,
                (px[2].clamp(0.0, 1.0) * 255.0) as u8,
                (px[3].clamp(0.0, 1.0) * 255.0) as u8,
            ]
        })
        .collect()
}

/// Convert Ochroma's RenderCamera to the Spectra-compatible SpectraCamera.
fn ochroma_to_spectra_camera(cam: &RenderCamera, width: u32, height: u32) -> SpectraCamera {
    // glam Mat4 is column-major; Spectra renderer uses row-major view_matrix.
    // glam `to_cols_array()` gives column-major, so we transpose.
    //
    // glam's look_at_rh produces a view matrix where the camera looks down -Z
    // (cam_z is negative for objects in front). Spectra's renderer expects
    // positive cam_z for visible objects (looks down +Z in camera space).
    // We negate the third row (Z axis) to convert.
    let cols = cam.view.to_cols_array();
    let view_row_major = [
        cols[0],  cols[4],  cols[8],  cols[12],
        cols[1],  cols[5],  cols[9],  cols[13],
        -cols[2], -cols[6], -cols[10], -cols[14],  // negate Z row
        cols[3],  cols[7],  cols[11], cols[15],
    ];

    let proj_cols = cam.proj.to_cols_array();
    // Extract focal lengths from projection matrix:
    // proj[0][0] = 2*fx/width => fx = proj[0][0] * width / 2
    let fx = proj_cols[0] * width as f32 / 2.0;
    let fy = proj_cols[5] * height as f32 / 2.0;

    SpectraCamera {
        view_matrix: view_row_major,
        width: width as usize,
        height: height as usize,
        fx: fx.abs(),
        fy: fy.abs(),
        near: 0.1,
        far: 1000.0,
    }
}

// ---------------------------------------------------------------------------
// EWA splatting core (ported from spectra-gaussian-render/src/renderer.rs)
// ---------------------------------------------------------------------------

/// Build rotation matrix from quaternion [w, x, y, z].
fn quat_to_rotation(q: &[f32; 4]) -> [[f32; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
    ]
}

/// Compute 3D covariance: Sigma = R * S * S^T * R^T where S = diag(scales).
fn compute_cov3d(rotation: &[[f32; 3]; 3], scales: &[f32; 3]) -> [[f32; 3]; 3] {
    // M = R * S
    let mut m = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = rotation[i][j] * scales[j];
        }
    }
    // Sigma = M * M^T
    let mut cov = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                cov[i][j] += m[i][k] * m[j][k];
            }
        }
    }
    cov
}

/// Project a 3D Gaussian to 2D screen space using EWA splatting.
fn project_gaussian(
    gaussian: &Gaussian3D,
    camera: &SpectraCamera,
) -> Option<ProjectedGaussian> {
    let pos = &gaussian.position;
    let vm = &camera.view_matrix;

    // Transform to camera space (row-major view matrix)
    let cam_x = vm[0] * pos[0] + vm[1] * pos[1] + vm[2] * pos[2] + vm[3];
    let cam_y = vm[4] * pos[0] + vm[5] * pos[1] + vm[6] * pos[2] + vm[7];
    let cam_z = vm[8] * pos[0] + vm[9] * pos[1] + vm[10] * pos[2] + vm[11];

    // Near/far clip
    if cam_z < camera.near || cam_z > camera.far {
        return None;
    }

    // Project to screen
    let inv_z = 1.0 / cam_z;
    let screen_x = camera.fx * cam_x * inv_z + camera.width as f32 * 0.5;
    let screen_y = camera.fy * cam_y * inv_z + camera.height as f32 * 0.5;

    // Build 3D covariance
    let rot = quat_to_rotation(&gaussian.rotation);
    let scales = gaussian.scales();
    let cov3d = compute_cov3d(&rot, &scales);

    // Jacobian of perspective projection (EWA)
    let inv_z2 = inv_z * inv_z;
    let j00 = camera.fx * inv_z;
    let j02 = -camera.fx * cam_x * inv_z2;
    let j11 = camera.fy * inv_z;
    let j12 = -camera.fy * cam_y * inv_z2;

    // Extract 3x3 rotation from view matrix (upper-left block, row-major)
    let w_rot = [
        [vm[0], vm[1], vm[2]],
        [vm[4], vm[5], vm[6]],
        [vm[8], vm[9], vm[10]],
    ];

    // cov3d_cam = W_rot * cov3d * W_rot^T
    let mut temp = [[0.0f32; 3]; 3];
    let mut cov3d_cam = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                temp[i][j] += w_rot[i][k] * cov3d[k][j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                cov3d_cam[i][j] += temp[i][k] * w_rot[j][k];
            }
        }
    }

    // 2D covariance via Jacobian: J * cov3d_cam * J^T
    let a = j00 * j00 * cov3d_cam[0][0]
        + 2.0 * j00 * j02 * cov3d_cam[0][2]
        + j02 * j02 * cov3d_cam[2][2];
    let b = j00 * j11 * cov3d_cam[0][1]
        + j00 * j12 * cov3d_cam[0][2]
        + j02 * j11 * cov3d_cam[2][1]
        + j02 * j12 * cov3d_cam[2][2];
    let c = j11 * j11 * cov3d_cam[1][1]
        + 2.0 * j11 * j12 * cov3d_cam[1][2]
        + j12 * j12 * cov3d_cam[2][2];

    // Numerical stability
    let a = a + 0.3;
    let c = c + 0.3;

    // Conic = inverse of 2D covariance
    let det = a * c - b * b;
    if det <= 0.0 {
        return None;
    }
    let inv_det = 1.0 / det;
    let conic = [c * inv_det, -b * inv_det, a * inv_det];

    // Radius: 3-sigma from max eigenvalue
    let trace = a + c;
    let discriminant = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let lambda_max = 0.5 * (trace + discriminant);
    let radius = (3.0 * lambda_max.sqrt()).ceil();

    // Screen bounds check
    if screen_x + radius < 0.0
        || screen_x - radius >= camera.width as f32
        || screen_y + radius < 0.0
        || screen_y - radius >= camera.height as f32
    {
        return None;
    }

    Some(ProjectedGaussian {
        screen_pos: [screen_x, screen_y],
        depth: cam_z,
        conic,
        color: gaussian.color,
        opacity: gaussian.opacity,
        radius,
        index: 0,
    })
}

/// Tile-Gaussian pair for sorting.
#[derive(Debug, Clone)]
struct TileGaussian {
    tile_id: u32,
    depth: f32,
    gaussian_idx: usize,
}

/// Core tile-based render (identical algorithm to spectra-gaussian-render).
fn render_cpu_internal(
    gaussians: &[Gaussian3D],
    camera: &SpectraCamera,
    shadow_mask: Option<&[f32]>,
) -> Vec<f32> {
    let w = camera.width;
    let h = camera.height;
    let tiles_x = w.div_ceil(TILE_SIZE);
    let tiles_y = h.div_ceil(TILE_SIZE);

    // Step 1: Project all Gaussians
    let mut projected: Vec<ProjectedGaussian> = Vec::with_capacity(gaussians.len());
    for (i, g) in gaussians.iter().enumerate() {
        if let Some(mut pg) = project_gaussian(g, camera) {
            pg.index = i;
            projected.push(pg);
        }
    }

    // Step 2: Assign to tiles
    let mut tile_gaussians: Vec<TileGaussian> = Vec::new();
    for pg in &projected {
        let min_tx = ((pg.screen_pos[0] - pg.radius).max(0.0) as usize / TILE_SIZE)
            .min(tiles_x.saturating_sub(1));
        let max_tx = ((pg.screen_pos[0] + pg.radius).max(0.0) as usize / TILE_SIZE)
            .min(tiles_x.saturating_sub(1));
        let min_ty = ((pg.screen_pos[1] - pg.radius).max(0.0) as usize / TILE_SIZE)
            .min(tiles_y.saturating_sub(1));
        let max_ty = ((pg.screen_pos[1] + pg.radius).max(0.0) as usize / TILE_SIZE)
            .min(tiles_y.saturating_sub(1));

        for ty in min_ty..=max_ty {
            for tx in min_tx..=max_tx {
                tile_gaussians.push(TileGaussian {
                    tile_id: (ty * tiles_x + tx) as u32,
                    depth: pg.depth,
                    gaussian_idx: pg.index,
                });
            }
        }
    }

    // Step 3: Sort by tile then depth
    tile_gaussians.sort_by(|a, b| {
        a.tile_id
            .cmp(&b.tile_id)
            .then(a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Build per-tile ranges
    let num_tiles = tiles_x * tiles_y;
    let mut tile_ranges: Vec<(usize, usize)> = vec![(0, 0); num_tiles];
    if !tile_gaussians.is_empty() {
        let mut start = 0;
        let mut current_tile = tile_gaussians[0].tile_id;
        for (i, tg) in tile_gaussians.iter().enumerate() {
            if tg.tile_id != current_tile {
                tile_ranges[current_tile as usize] = (start, i);
                start = i;
                current_tile = tg.tile_id;
            }
        }
        tile_ranges[current_tile as usize] = (start, tile_gaussians.len());
    }

    // Lookup from gaussian index -> projected index (owned, no references, so Send)
    let mut proj_idx_map: Vec<Option<usize>> = vec![None; gaussians.len()];
    for (i, pg) in projected.iter().enumerate() {
        proj_idx_map[pg.index] = Some(i);
    }

    // Step 4: Per-pixel front-to-back alpha blending — tiles processed in parallel
    let tile_pixel_bufs: Vec<(usize, Vec<f32>)> = (0..num_tiles)
        .into_par_iter()
        .filter_map(|tile_id| {
            let (start, end) = tile_ranges[tile_id];
            if start == end {
                return None;
            }

            let tx = tile_id % tiles_x;
            let ty = tile_id / tiles_x;
            let px_start_x = tx * TILE_SIZE;
            let px_start_y = ty * TILE_SIZE;
            let px_end_x = (px_start_x + TILE_SIZE).min(w);
            let px_end_y = (px_start_y + TILE_SIZE).min(h);
            let tile_w = px_end_x - px_start_x;
            let tile_h = px_end_y - px_start_y;
            let mut tile_pixels = vec![0.0f32; tile_w * tile_h * 4];

            for (local_py, py) in (px_start_y..px_end_y).enumerate() {
                for (local_px, px) in (px_start_x..px_end_x).enumerate() {
                    let local_idx = (local_py * tile_w + local_px) * 4;
                    let mut transmittance = 1.0f32;
                    let pxf = px as f32 + 0.5;
                    let pyf = py as f32 + 0.5;

                    let pixel_shadow = shadow_mask
                        .map(|m| m[py * w + px])
                        .unwrap_or(1.0);

                    for tg_idx in start..end {
                        if transmittance < TRANSMITTANCE_THRESHOLD {
                            break;
                        }

                        let tg = &tile_gaussians[tg_idx];
                        let pg = match proj_idx_map[tg.gaussian_idx] {
                            Some(idx) => &projected[idx],
                            None => continue,
                        };

                        let dx = pxf - pg.screen_pos[0];
                        let dy = pyf - pg.screen_pos[1];
                        let power = -0.5
                            * (pg.conic[0] * dx * dx
                                + 2.0 * pg.conic[1] * dx * dy
                                + pg.conic[2] * dy * dy);

                        if power > 0.0 {
                            continue;
                        }

                        let alpha = (pg.opacity * pixel_shadow * power.exp()).min(0.99);
                        if alpha < ALPHA_THRESHOLD {
                            continue;
                        }

                        let weight = alpha * transmittance;
                        tile_pixels[local_idx] += weight * pg.color[0];
                        tile_pixels[local_idx + 1] += weight * pg.color[1];
                        tile_pixels[local_idx + 2] += weight * pg.color[2];
                        tile_pixels[local_idx + 3] += weight;
                        transmittance *= 1.0 - alpha;
                    }
                }
            }
            Some((tile_id, tile_pixels))
        })
        .collect();

    // Merge tile pixel buffers into the main image (sequential)
    let mut image = vec![0.0f32; w * h * 4];
    for (tile_id, tile_pixels) in tile_pixel_bufs {
        let tx = tile_id % tiles_x;
        let ty = tile_id / tiles_x;
        let px_start_x = tx * TILE_SIZE;
        let px_start_y = ty * TILE_SIZE;
        let px_end_x = (px_start_x + TILE_SIZE).min(w);
        let px_end_y = (px_start_y + TILE_SIZE).min(h);
        let tile_w = px_end_x - px_start_x;
        for (local_py, py) in (px_start_y..px_end_y).enumerate() {
            for (local_px, px) in (px_start_x..px_end_x).enumerate() {
                let local_idx = (local_py * tile_w + local_px) * 4;
                let global_idx = (py * w + px) * 4;
                image[global_idx] = tile_pixels[local_idx];
                image[global_idx + 1] = tile_pixels[local_idx + 1];
                image[global_idx + 2] = tile_pixels[local_idx + 2];
                image[global_idx + 3] = tile_pixels[local_idx + 3];
            }
        }
    }

    image
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render Ochroma GaussianSplats using the Spectra-compatible tile-based
/// EWA splatting algorithm.
///
/// Returns RGBA f32 pixel data `[height * width * 4]`.
pub fn render_with_spectra(
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    width: u32,
    height: u32,
    illuminant: &Illuminant,
) -> Vec<f32> {
    let gaussians: Vec<Gaussian3D> = splats
        .iter()
        .map(|s| ochroma_to_gaussian3d(s, illuminant))
        .collect();

    let cam = ochroma_to_spectra_camera(camera, width, height);
    render_cpu_internal(&gaussians, &cam, None)
}

/// Render and convert to u8 RGBA framebuffer.
pub fn render_with_spectra_u8(
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    width: u32,
    height: u32,
    illuminant: &Illuminant,
) -> Vec<[u8; 4]> {
    let float_pixels = render_with_spectra(splats, camera, width, height, illuminant);

    float_pixels
        .chunks(4)
        .map(|rgba| {
            [
                (rgba[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgba[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgba[2].clamp(0.0, 1.0) * 255.0) as u8,
                (rgba[3].clamp(0.0, 1.0) * 255.0) as u8,
            ]
        })
        .collect()
}

/// Render with optional per-pixel shadow mask.
///
/// `shadow_mask`: `width * height` f32 values in `[0,1]`; `None` = fully lit.
/// Shadow mask is applied by multiplying each Gaussian's effective opacity at
/// the corresponding pixel, darkening fully-shadowed (0.0) areas.
pub fn render_with_spectra_u8_shadowed(
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    width: u32,
    height: u32,
    illuminant: &Illuminant,
    shadow_mask: Option<&[f32]>,
) -> Vec<[u8; 4]> {
    let gaussians: Vec<Gaussian3D> = splats
        .iter()
        .map(|s| ochroma_to_gaussian3d(s, illuminant))
        .collect();
    let cam = ochroma_to_spectra_camera(camera, width, height);
    let float_pixels = render_cpu_internal(&gaussians, &cam, shadow_mask);
    float_pixels
        .chunks(4)
        .map(|rgba| {
            [
                (rgba[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgba[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgba[2].clamp(0.0, 1.0) * 255.0) as u8,
                (rgba[3].clamp(0.0, 1.0) * 255.0) as u8,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn make_test_splat(pos: [f32; 3], spd_val: f32) -> GaussianSplat {
        GaussianSplat::volume(
            pos,
            [0.3, 0.3, 0.3],
            glam::Quat::IDENTITY,
            230,
            std::array::from_fn(|_| f16::from_f32(spd_val).to_bits()),
        )
    }

    fn make_camera(eye: Vec3, target: Vec3, w: u32, h: u32) -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                w as f32 / h as f32,
                0.1,
                500.0,
            ),
        }
    }

    #[test]
    fn render_with_spectra_u8_returns_correct_pixel_count() {
        use vox_core::spectral::Illuminant;
        let splats = vec![];
        let camera = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 64, 64);
        let result = render_with_spectra_u8(&splats, &camera, 64, 64, &Illuminant::d65());
        assert_eq!(result.len(), 64 * 64, "should produce one pixel per texel");
    }

    #[test]
    fn spectra_render_empty_produces_zeros() {
        let cam = make_camera(Vec3::new(0.0, 5.0, 15.0), Vec3::ZERO, 64, 64);
        let pixels = render_with_spectra(&[], &cam, 64, 64, &Illuminant::d65());
        assert_eq!(pixels.len(), 64 * 64 * 4);
        assert!(pixels.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn spectra_render_single_splat_has_colour() {
        let splat = make_test_splat([0.0, 0.0, 0.0], 0.5);
        let cam = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 64, 64);
        let pixels = render_with_spectra(&[splat], &cam, 64, 64, &Illuminant::d65());
        let total_alpha: f32 = pixels.iter().skip(3).step_by(4).sum();
        assert!(total_alpha > 0.0, "Single splat should produce some alpha");
    }

    #[test]
    fn spectra_u8_output_has_correct_length() {
        let splat = make_test_splat([0.0, 0.0, 0.0], 0.5);
        let cam = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 128, 64);
        let pixels = render_with_spectra_u8(&[splat], &cam, 128, 64, &Illuminant::d65());
        assert_eq!(pixels.len(), 128 * 64);
    }

    #[test]
    fn parallel_render_is_deterministic() {
        let splat = make_test_splat([0.0, 0.0, 0.0], 0.5);
        let cam = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 64, 64);
        let r1 = render_with_spectra_u8(&[splat.clone()], &cam, 64, 64, &Illuminant::d65());
        let r2 = render_with_spectra_u8(&[splat], &cam, 64, 64, &Illuminant::d65());
        assert_eq!(r1, r2, "parallel render must be deterministic across two runs");
    }

    #[test]
    fn parallel_render_matches_expected_pixel_count() {
        let splats: Vec<GaussianSplat> = (0..10)
            .map(|i| make_test_splat([i as f32 * 0.2 - 1.0, 0.0, 0.0], 0.5))
            .collect();
        let cam = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 128, 128);
        let result = render_with_spectra_u8(&splats, &cam, 128, 128, &Illuminant::d65());
        assert_eq!(result.len(), 128 * 128, "output must be width×height pixels");
    }

    #[test]
    fn band_to_heatmap_zero_is_black() {
        let c = band_to_heatmap(0.0);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn band_to_heatmap_one_is_red() {
        let c = band_to_heatmap(1.0);
        assert_eq!(c[0], 1.0);
        assert!(c[1] < 0.01, "green channel should be near 0 at t=1");
        assert_eq!(c[2], 0.0);
    }

    #[test]
    fn band_to_heatmap_clamps_out_of_range() {
        assert_eq!(band_to_heatmap(-1.0), band_to_heatmap(0.0));
        assert_eq!(band_to_heatmap(2.0), band_to_heatmap(1.0));
    }

    #[test]
    fn render_spectral_band_returns_correct_pixel_count() {
        let splats: Vec<GaussianSplat> = Vec::new();
        let camera = RenderCamera {
            view: glam::Mat4::IDENTITY,
            proj: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
        };
        let pixels = render_spectral_band_u8(&splats, &camera, 16, 16, 0);
        assert_eq!(pixels.len(), 16 * 16);
    }

    #[test]
    fn render_spectral_band_differs_per_band_when_nonzero_spectral() {
        let mut spectral = [0u16; 16];
        spectral[0] = half::f16::from_f32(1.0).to_bits();
        spectral[7] = half::f16::from_f32(0.0).to_bits();
        let splat = GaussianSplat::volume(
            [0.0, 0.0, -2.0],
            [0.5, 0.5, 0.5],
            glam::Quat::IDENTITY,
            255,
            spectral,
        );
        let camera = RenderCamera {
            view: glam::Mat4::IDENTITY,
            proj: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
        };
        let pixels_b0 = render_spectral_band_u8(&[splat], &camera, 32, 32, 0);
        let pixels_b7 = render_spectral_band_u8(&[splat], &camera, 32, 32, 7);
        let sum_b0: u32 = pixels_b0.iter().map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32).sum();
        let sum_b7: u32 = pixels_b7.iter().map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32).sum();
        assert_ne!(sum_b0, sum_b7, "band 0 (full) and band 7 (empty) should differ");
    }

    #[test]
    fn shadow_mask_darkens_pixel() {
        use vox_core::spectral::Illuminant;
        let splat = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            glam::Quat::IDENTITY,
            220,
            std::array::from_fn(|_| f16::from_f32(0.8).to_bits()),
        );
        let cam = make_camera(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, 32, 32);
        let lit = render_with_spectra_u8_shadowed(
            &[splat.clone()], &cam, 32, 32, &Illuminant::d65(), None,
        );
        let shadow_mask = vec![0.0f32; 32 * 32];
        let shadowed = render_with_spectra_u8_shadowed(
            &[splat], &cam, 32, 32, &Illuminant::d65(), Some(&shadow_mask),
        );
        let centre = 16 * 32 + 16;
        let lit_lum = lit[centre][0] as u32 + lit[centre][1] as u32 + lit[centre][2] as u32;
        let shad_lum = shadowed[centre][0] as u32 + shadowed[centre][1] as u32 + shadowed[centre][2] as u32;
        assert!(shad_lum <= lit_lum, "shadowed pixel must not be brighter than lit");
    }

    #[cfg(all(test, feature = "spectra-native"))]
    #[test]
    fn spectra_backend_system_new_does_not_panic() {
        use crate::splat_backend::SpectraRenderBackend;
        // Construction requires GPU — just verify the type builds.
        let _: fn(u32, u32) -> Result<SpectraRenderBackend, String> =
            SpectraRenderBackend::realtime;
    }
}

// ---------------------------------------------------------------------------
// Spectra native backend system (spectra-native feature only)
// ---------------------------------------------------------------------------

#[cfg(feature = "spectra-native")]
pub mod native {
    use crate::splat_backend::SpectraRenderBackend;
    use crate::splat_convert::convert_splats;
    use spectra_renderer::{SplatScene, CameraParams};
    use vox_core::types::GaussianSplat;

    /// Per-viewport Spectra backend. One instance per active Spectra viewport.
    pub struct SpectraBackendSystem {
        backend: SpectraRenderBackend,
        scene_dirty: bool,
        last_splat_count: usize,
    }

    impl SpectraBackendSystem {
        pub fn realtime(width: u32, height: u32) -> Result<Self, String> {
            Ok(Self {
                backend: SpectraRenderBackend::realtime(width, height)?,
                scene_dirty: true,
                last_splat_count: 0,
            })
        }

        pub fn cinematic(width: u32, height: u32) -> Result<Self, String> {
            Ok(Self {
                backend: SpectraRenderBackend::cinematic(width, height)?,
                scene_dirty: true,
                last_splat_count: 0,
            })
        }

        /// Call once per Bevy render tick.
        ///
        /// `splats`  — all GaussianSplat components from the ECS world.
        /// `camera`  — extracted Bevy camera this frame.
        ///
        /// Returns `Some(rgba8)` when a completed frame is available (one tick of latency),
        /// or `None` if no frame has been completed yet.
        pub fn tick(
            &mut self,
            splats:  &[GaussianSplat],
            camera:  CameraParams,
        ) -> Option<Vec<u8>> {
            // Rebuild scene only when splat count changes (cheap dirty check).
            // Full hash-based change detection is a follow-on improvement.
            let new_scene = if self.scene_dirty || splats.len() != self.last_splat_count {
                let (surfaces, volumes) = convert_splats(splats);
                self.last_splat_count = splats.len();
                self.scene_dirty = false;
                Some(SplatScene::new(surfaces, volumes))
            } else {
                None
            };

            if let Err(e) = self.backend.submit_frame(new_scene, camera) {
                eprintln!("[SpectraBackendSystem] submit_frame error: {e}");
                return None;
            }

            let output = self.backend.read_last_output();
            if output.is_empty() { None } else { Some(output) }
        }

        /// Force scene rebuild on next tick (e.g. after ECS structural change).
        pub fn mark_dirty(&mut self) { self.scene_dirty = true; }

        /// Consecutive failure count — caller should fallback to BuiltIn at 3.
        pub fn fail_count(&self) -> u32 { self.backend.fail_count() }

        pub fn dimensions(&self) -> (u32, u32) {
            (self.backend.width(), self.backend.height())
        }
    }
}
