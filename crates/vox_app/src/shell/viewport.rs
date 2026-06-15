//! The REAL viewport — an actual engine frame rendered into the dock tab.
//!
//! LAW: the editor's viewport is rendered by the SPECTRA PATH TRACER. Under the
//! `spectra` feature the small spectral-splat scene is path-traced on the
//! Spectra Vulkan device via [`vox_render::splat_backend::pathtrace_splats_to_rgba`],
//! producing an RGBA frame that is uploaded as an `egui::TextureHandle` and drawn
//! as an `egui::Image` inside the Viewport tab. The CPU `SoftwareRasteriser`
//! viewport path is DELETED for the editor — it survives ONLY as a no-GPU
//! fallback for the headless `shell_snapshot` / `scale_trial` bins that do not
//! link the Spectra sibling (without the `spectra` feature).

use glam::Quat;
use vox_core::types::GaussianSplat;

// The CPU `SoftwareRasteriser` fallback is the banned software-rasterizer stack:
// available only when `spectra` is off AND the opt-in `legacy-raster` feature is
// on. A plain default build (no spectra, no legacy-raster) compiles neither the
// import nor the fallback; the dispatcher returns a solid dark frame instead.
#[cfg(all(not(feature = "spectra"), feature = "legacy-raster"))]
use glam::{Mat4, Vec3};
#[cfg(all(not(feature = "spectra"), feature = "legacy-raster"))]
use vox_core::spectral::Illuminant;
#[cfg(all(not(feature = "spectra"), feature = "legacy-raster"))]
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
#[cfg(all(not(feature = "spectra"), feature = "legacy-raster"))]
use vox_render::spectral::RenderCamera;

/// Resolution of the off-screen splat frame uploaded to the viewport texture.
pub const VIEW_W: usize = 480;
pub const VIEW_H: usize = 320;

/// Build a small, deterministic spectral-splat scene: a green-ish ground band of
/// volume splats plus a few warm "structure" clusters and a bright sky-lit
/// accent, so the rendered frame has real scene-like color variance (not a flat
/// fill). All splats are `GaussianSplat::volume` with per-band f16 spectra.
pub fn build_scene() -> Vec<GaussianSplat> {
    let mut splats = Vec::new();

    // Spectral helper: a 16-band reflectance with a value in [lo..hi] biased to a
    // band window (so different materials read as different colors).
    let spd = |window: std::ops::RangeInclusive<usize>, hi: f32, lo: f32| -> [u16; 16] {
        std::array::from_fn(|i| {
            let v = if window.contains(&i) { hi } else { lo };
            half::f16::from_f32(v).to_bits()
        })
    };

    // Ground: a grid of low green splats (long-wavelength low, mid high => green).
    let ground = spd(5..=9, 0.85, 0.18);
    for gz in -6..=6 {
        for gx in -6..=6 {
            let x = gx as f32 * 1.4;
            let z = gz as f32 * 1.4 - 6.0;
            splats.push(GaussianSplat::volume(
                [x, -1.0, z],
                [0.7, 0.25, 0.7],
                Quat::IDENTITY,
                220,
                ground,
            ));
        }
    }

    // Warm structures (buildings): a few short-wavelength-high clusters (amber).
    let amber = spd(9..=14, 0.9, 0.25);
    for (bx, bz, h) in [(-3.0f32, -4.0f32, 3), (2.5, -7.0, 4), (0.0, -9.0, 5)] {
        for iy in 0..h {
            splats.push(GaussianSplat::volume(
                [bx, -0.5 + iy as f32 * 0.6, bz],
                [0.5, 0.5, 0.5],
                Quat::IDENTITY,
                235,
                amber,
            ));
        }
    }

    // A cool bright accent (sky-lit) — broad high reflectance.
    let bright = spd(0..=15, 0.95, 0.6);
    splats.push(GaussianSplat::volume(
        [-1.5, 1.6, -5.0],
        [0.9, 0.9, 0.9],
        Quat::IDENTITY,
        255,
        bright,
    ));

    // Violet field markers (short + long high) for extra color spread.
    let violet = spd(0..=3, 0.9, 0.2);
    for vx in [-4.0f32, 3.5] {
        splats.push(GaussianSplat::volume(
            [vx, 0.2, -6.5],
            [0.4, 0.4, 0.4],
            Quat::IDENTITY,
            240,
            violet,
        ));
    }

    splats
}

/// Camera the viewport looks through: eye slightly above the origin, gazing into
/// the scene down -Z. Used by the Spectra path tracer.
#[cfg(feature = "spectra")]
const VIEW_EYE: [f32; 3] = [0.0, 1.2, 6.0];
#[cfg(feature = "spectra")]
const VIEW_TARGET: [f32; 3] = [0.0, 0.0, -6.0];
#[cfg(feature = "spectra")]
const VIEW_FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
/// Direction toward the sun (matches the warm key the path tracer expects).
#[cfg(feature = "spectra")]
const VIEW_SUN_DIR: [f32; 3] = [0.4, 0.85, 0.35];
/// Samples per pixel for the viewport still. Kept modest so the in-editor frame
/// stays interactive; the headless proof renders the same path.
#[cfg(feature = "spectra")]
const VIEW_SPP: u32 = 24;

/// Rasterize the scene to an RGBA8 buffer (row-major, 4 bytes/px) at
/// [`VIEW_W`]x[`VIEW_H`], looking down the -Z axis at the scene.
pub fn render_scene_rgba() -> Vec<u8> {
    render_scene_rgba_with(&[])
}

/// Render the base scene + `overlay` to RGBA8. Under the `spectra` feature this
/// PATH-TRACES on the Spectra Vulkan device; without it (snapshot bins) it falls
/// back to the CPU rasteriser.
pub fn render_scene_rgba_with(overlay: &[GaussianSplat]) -> Vec<u8> {
    #[cfg(feature = "spectra")]
    {
        render_scene_rgba_pathtraced(overlay)
    }
    #[cfg(all(not(feature = "spectra"), feature = "legacy-raster"))]
    {
        render_scene_rgba_cpu(overlay)
    }
    // Default build (no Spectra path tracer linked, no legacy CPU rasterizer):
    // THE LAW forbids a non-Spectra renderer, so there is nothing to draw with.
    // Return a solid dark studio frame rather than pulling in the banned stack.
    #[cfg(all(not(feature = "spectra"), not(feature = "legacy-raster")))]
    {
        let _ = overlay;
        (0..VIEW_W * VIEW_H).flat_map(|_| [16u8, 18, 26, 255]).collect()
    }
}

/// Path-trace the base scene + `overlay` on the Spectra Vulkan device. This is
/// the editor's REAL viewport. If the GPU/Spectra stack is unavailable the call
/// errors; we surface a solid dark frame rather than panicking the UI thread.
#[cfg(feature = "spectra")]
pub fn render_scene_rgba_pathtraced(overlay: &[GaussianSplat]) -> Vec<u8> {
    let mut splats = build_scene();
    splats.extend_from_slice(overlay);
    match vox_render::splat_backend::pathtrace_splats_to_rgba(
        &splats,
        VIEW_EYE,
        VIEW_TARGET,
        VIEW_FOV_Y,
        VIEW_W as u32,
        VIEW_H as u32,
        VIEW_SPP,
        VIEW_SUN_DIR,
    ) {
        Ok(rgba) if rgba.len() == VIEW_W * VIEW_H * 4 => rgba,
        Ok(other) => {
            eprintln!(
                "[viewport] path-traced frame had {} bytes, expected {}",
                other.len(),
                VIEW_W * VIEW_H * 4
            );
            vec![0u8; VIEW_W * VIEW_H * 4]
                .chunks_exact(4)
                .flat_map(|_| [16u8, 18, 26, 255])
                .collect()
        }
        Err(e) => {
            eprintln!("[viewport] path-trace failed: {e}");
            (0..VIEW_W * VIEW_H)
                .flat_map(|_| [16u8, 18, 26, 255])
                .collect()
        }
    }
}

/// CPU-rasteriser fallback for non-Spectra bins. Composites an additive `overlay`
/// of splats (e.g. a grown FloraPrime tree the shell owns) ON TOP of the base
/// [`build_scene`]. The base scene stays fixed; the overlay is what the shell
/// grows/undoes.
#[cfg(all(not(feature = "spectra"), feature = "legacy-raster"))]
pub fn render_scene_rgba_cpu(overlay: &[GaussianSplat]) -> Vec<u8> {
    let mut splats = build_scene();
    splats.extend_from_slice(overlay);
    let eye = Vec3::new(0.0, 1.2, 6.0);
    let target = Vec3::new(0.0, 0.0, -6.0);
    let camera = RenderCamera {
        view: Mat4::look_at_rh(eye, target, Vec3::Y),
        proj: Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            VIEW_W as f32 / VIEW_H as f32,
            0.1,
            200.0,
        ),
    };
    let illuminant = Illuminant::d65();
    let mut ras = SoftwareRasteriser::new(VIEW_W as u32, VIEW_H as u32);
    let fb = ras.render(&splats, &camera, &illuminant, None);

    // Composite over a dark studio background so empty pixels aren't pure black
    // (matches the surface.bg viewport feel) and flatten to opaque RGBA.
    let bg = [16u8, 18, 26];
    let mut out = Vec::with_capacity(VIEW_W * VIEW_H * 4);
    for p in &fb.pixels {
        let a = p[3] as f32 / 255.0;
        let r = (p[0] as f32 * a + bg[0] as f32 * (1.0 - a)).round() as u8;
        let g = (p[1] as f32 * a + bg[1] as f32 * (1.0 - a)).round() as u8;
        let b = (p[2] as f32 * a + bg[2] as f32 * (1.0 - a)).round() as u8;
        out.extend_from_slice(&[r, g, b, 255]);
    }
    out
}

/// Like [`render_scene_rgba_with`] but RE-ILLUMINATES the overlay under a target
/// illuminant before compositing — the AAA Spec-03 forgery view. The base scene
/// stays fixed (rendered under d65); only the shell-owned `overlay` (the planted
/// forgery surfaces) is relit from `reference` to `target` through
/// [`vox_render::relight::relight_scene`], so two metameric surfaces that read
/// identical under the matched/gallery light split into visibly divergent hues
/// under the inspection light — a hue shift no RGB engine can produce, because an
/// RGB capture stored a single triple. The hue change lives in the splat
/// radiance, so the camera observer stays d65 (the redistribution already
/// happened in the spectral relight).
pub fn render_scene_rgba_relit(
    overlay: &[GaussianSplat],
    reference: &vox_render::relight::IlluminantSpec,
    target: &vox_render::relight::IlluminantSpec,
) -> Vec<u8> {
    use vox_render::relight::{relight_scene, RelightSettings};
    let settings = RelightSettings::new(reference.clone(), target.clone())
        .with_sky_ambient(false)
        .with_shadows(false);
    let (relit, _report) = relight_scene(overlay, &settings);
    render_scene_rgba_with(&relit)
}

/// Build (or reuse) the viewport scene texture on `ctx`, compositing `overlay`
/// (the shell-owned grown splats) over the base scene. The handle is cached in
/// `cache`; the first call (or the first after the shell invalidates the cache by
/// setting it to `None`, e.g. after growing/undoing a tree, or switching the
/// inspection light) rasterizes the scene + overlay and uploads it.
///
/// When `target` differs from `reference` (the user switched the inspection
/// light), the overlay is re-illuminated first via [`render_scene_rgba_relit`];
/// when they match (the default), the cheap non-relit path is byte-identical to
/// the historical behavior.
pub fn scene_texture(
    ctx: &egui::Context,
    cache: &mut Option<egui::TextureHandle>,
    overlay: &[GaussianSplat],
    reference: &vox_render::relight::IlluminantSpec,
    target: &vox_render::relight::IlluminantSpec,
) -> egui::TextureHandle {
    if let Some(h) = cache {
        return h.clone();
    }
    let rgba = if reference.name() == target.name() {
        render_scene_rgba_with(overlay)
    } else {
        render_scene_rgba_relit(overlay, reference, target)
    };
    let color =
        egui::ColorImage::from_rgba_unmultiplied([VIEW_W, VIEW_H], &rgba);
    let handle = ctx.load_texture("viewport_scene", color, egui::TextureOptions::LINEAR);
    *cache = Some(handle.clone());
    handle
}

// These tests exercise the CPU rasteriser contract (no GPU). They build only
// without the `spectra` feature AND with the opt-in `legacy-raster` feature,
// where `render_scene_rgba_with` is the CPU software-rasterizer path. In a plain
// default build that path is a flat dark frame, so the visible-pixel assertions
// would be vacuous — hence gated with the rasterizer they exercise.
#[cfg(all(test, not(feature = "spectra"), feature = "legacy-raster"))]
mod tests {
    use super::*;

    #[test]
    fn scene_has_real_color_variance_not_flat() {
        let rgba = render_scene_rgba();
        assert_eq!(rgba.len(), VIEW_W * VIEW_H * 4);
        // Count non-background pixels and measure channel variance.
        let bg = [16i32, 18, 26];
        let mut non_bg = 0usize;
        let (mut sr, mut sg, mut sb) = (0f64, 0f64, 0f64);
        let n = (rgba.len() / 4) as f64;
        for px in rgba.chunks_exact(4) {
            let d = (0..3).map(|i| (px[i] as i32 - bg[i]).abs()).max().unwrap();
            if d > 12 {
                non_bg += 1;
            }
            sr += px[0] as f64;
            sg += px[1] as f64;
            sb += px[2] as f64;
        }
        let (mr, mg, mb) = (sr / n, sg / n, sb / n);
        let mut var = 0f64;
        for px in rgba.chunks_exact(4) {
            var += (px[0] as f64 - mr).powi(2)
                + (px[1] as f64 - mg).powi(2)
                + (px[2] as f64 - mb).powi(2);
        }
        var /= n;
        assert!(
            non_bg > 5000,
            "rendered scene has only {non_bg} non-background pixels (expected >5000)"
        );
        assert!(var > 50.0, "rendered scene is too flat (variance {var:.1})");
    }

    /// An overlay of grown-tree splats adds visible pixels: the composited frame
    /// (base + overlay) lights MORE non-background pixels than the base alone, and
    /// the base is recovered exactly when the overlay is empty.
    #[test]
    fn overlay_adds_visible_pixels_over_base() {
        use crate::shell::plugins::{grow_tree_skeleton, skeleton_to_splats};
        let count_non_bg = |rgba: &[u8]| -> usize {
            let bg = [16i32, 18, 26];
            rgba.chunks_exact(4)
                .filter(|px| (0..3).map(|i| (px[i] as i32 - bg[i]).abs()).max().unwrap() > 12)
                .count()
        };
        // Empty overlay reproduces the base frame byte-for-byte.
        assert_eq!(
            render_scene_rgba_with(&[]),
            render_scene_rgba(),
            "empty overlay must reproduce the base scene exactly"
        );
        let base_non_bg = count_non_bg(&render_scene_rgba());

        // A grown Silver Birch overlay must add visible pixels.
        let skel = grow_tree_skeleton(0, 3.0, 200);
        let tree = skeleton_to_splats(&skel, "broadleaf", 0);
        assert!(!tree.is_empty());
        let with_tree_non_bg = count_non_bg(&render_scene_rgba_with(&tree));
        assert!(
            with_tree_non_bg > base_non_bg + 200,
            "grown tree overlay must add visible pixels (base {base_non_bg}, with tree {with_tree_non_bg})"
        );
    }
}
