//! `prune` subcommand: importance-pruning + quantized splat optimization.
//!
//! Loads any supported splat asset (`.vxm`, `.ply`, `.spz`), scores every splat
//! by visual contribution (see [`vox_render::importance`]), drops the lowest-
//! importance splats toward a target keep-fraction, and writes a `.vxm`. A
//! render guard (default on) ensures aggressive pruning does not visibly hollow
//! the scene: it backs off (keeps more) if the rendered diff exceeds a bound.
//!
//! This is an OFFLINE/asset-time pass. It is complementary to the runtime
//! atom-budget LOD selector (frame-time): prune once to ship a smaller asset,
//! then let the budget trim further per frame.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use glam::{Mat4, Vec3};
use vox_core::types::GaussianSplat;
use vox_render::importance::{prune, prune_with_render_guard, PruneResult, PruneTarget};
use vox_render::spectral::RenderCamera;

/// Load splats from any supported format, dispatching on the file extension.
fn load_any(path: &Path) -> Result<Vec<GaussianSplat>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "vxm" => {
            let f = std::fs::File::open(path)
                .with_context(|| format!("open {}", path.display()))?;
            let vxm = vox_data::vxm::VxmFile::read(f)
                .with_context(|| format!("read vxm {}", path.display()))?;
            Ok(vxm.splats)
        }
        "ply" => vox_data::ply_loader::load_ply(path)
            .map_err(|e| anyhow!("read ply {}: {e}", path.display())),
        "spz" => vox_data::spz::load_spz(path)
            .map_err(|e| anyhow!("read spz {}: {e}", path.display())),
        other => Err(anyhow!(
            "unsupported input extension {:?} (expected vxm, ply, or spz)",
            other
        )),
    }
}

/// Write splats to a `.vxm` file.
fn write_vxm(path: &Path, splats: Vec<GaussianSplat>) -> Result<()> {
    let file = vox_data::vxm::VxmFile {
        header: vox_data::vxm::VxmHeader::new(
            uuid::Uuid::new_v4(),
            splats.len() as u32,
            vox_data::vxm::MaterialType::Generic,
        ),
        splats,
    };
    let mut out = std::fs::File::create(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write(&mut out)
        .with_context(|| format!("write vxm {}", path.display()))?;
    Ok(())
}

/// Bounding sphere of the splat set: (center, radius).
fn bounding_sphere(splats: &[GaussianSplat]) -> (Vec3, f32) {
    if splats.is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for s in splats {
        let p = Vec3::from(s.position());
        min = min.min(p);
        max = max.max(p);
    }
    let center = (min + max) * 0.5;
    let radius = (max - center).length().max(1e-3);
    (center, radius)
}

/// Build a deterministic camera framing the whole scene head-on, used both for
/// the render guard and the reported diff.
fn framing_camera(splats: &[GaussianSplat]) -> RenderCamera {
    let (center, radius) = bounding_sphere(splats);
    let fov = std::f32::consts::FRAC_PI_4;
    // Distance so the bounding sphere fits in the vertical FOV with margin.
    let dist = radius / (fov * 0.5).tan() * 1.5 + radius;
    let eye = center + Vec3::new(0.0, 0.0, dist);
    RenderCamera {
        view: Mat4::look_at_rh(eye, center, Vec3::Y),
        proj: Mat4::perspective_rh(fov, 1.0, 0.01_f32.max(dist * 1e-4), dist + radius * 4.0),
    }
}

/// Run the prune subcommand. Prints a real before/after report and returns the
/// resulting [`PruneResult`] for testability.
pub fn run_prune(
    input: &Path,
    output: &Path,
    keep: f32,
    max_pixel_diff: f32,
    no_guard: bool,
) -> Result<PruneResult> {
    let splats = load_any(input)?;
    let before_count = splats.len();
    let in_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);

    let camera = framing_camera(&splats);

    // Compute the render diff for reporting either way.
    let result = if no_guard {
        prune(&splats, PruneTarget::KeepFraction(keep))
    } else {
        prune_with_render_guard(&splats, keep, &camera, max_pixel_diff)
    };

    // Measure the final render diff for the report (original vs pruned).
    let render_diff = {
        use vox_core::spectral::Illuminant;
        use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
        let illum = Illuminant::d65();
        let mut ras = SoftwareRasteriser::new(96, 96);
        let reference = ras.render_gaussian(&splats, &camera, &illum, None);
        let pruned = ras.render_gaussian(&result.kept, &camera, &illum, None);
        vox_render::importance::mean_pixel_diff(&reference, &pruned)
    };

    let kept_count = result.kept.len();
    let final_fraction = if before_count > 0 {
        kept_count as f32 / before_count as f32
    } else {
        0.0
    };

    write_vxm(output, result.kept.clone())?;
    let out_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    let size_reduction = if in_size > 0 {
        100.0 * (1.0 - out_size as f32 / in_size as f32)
    } else {
        0.0
    };

    println!("prune: {} -> {}", input.display(), output.display());
    println!(
        "  splats:          {} -> {} ({} removed, kept {:.1}%)",
        before_count,
        kept_count,
        result.removed,
        final_fraction * 100.0
    );
    println!(
        "  file size:       {} bytes -> {} bytes ({:.1}% smaller)",
        in_size, out_size, size_reduction
    );
    println!(
        "  energy_retained: {:.4} (kept spectral-energy / original)",
        result.energy_retained
    );
    if no_guard {
        println!(
            "  render diff:     {:.5} mean abs pixel (guard disabled, requested keep={:.2})",
            render_diff, keep
        );
    } else {
        println!(
            "  render diff:     {:.5} mean abs pixel (guard bound {:.5}, requested keep={:.2})",
            render_diff, max_pixel_diff, keep
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use half::f16;

    fn flat_spectral(value: f32) -> [u16; 16] {
        [f16::from_f32(value).to_bits(); 16]
    }

    fn gen_scene() -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        // 300 redundant low-opacity clustered splats.
        for i in 0..300u32 {
            let fx = ((i.wrapping_mul(2654435761)) % 1000) as f32 / 1000.0 - 0.5;
            let fy = ((i.wrapping_mul(40503)) % 1000) as f32 / 1000.0 - 0.5;
            splats.push(GaussianSplat::volume(
                [fx * 0.2, fy * 0.2, 0.0],
                [0.05, 0.05, 0.05],
                Quat::IDENTITY,
                20,
                flat_spectral(0.1),
            ));
        }
        // 50 distinct bright opaque splats.
        for i in 0..50u32 {
            let gx = (i % 10) as f32 * 3.0 - 13.5;
            let gy = (i / 10) as f32 * 3.0 - 6.0;
            splats.push(GaussianSplat::volume(
                [gx, gy, 5.0],
                [0.5, 0.5, 0.5],
                Quat::IDENTITY,
                255,
                flat_spectral(1.0),
            ));
        }
        splats
    }

    #[test]
    fn cli_end_to_end_roundtrip() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let input = dir.join(format!("ochroma_prune_in_{pid}.vxm"));
        let output = dir.join(format!("ochroma_prune_out_{pid}.vxm"));

        // Write a temp .vxm input.
        let scene = gen_scene();
        let scene_len = scene.len();
        write_vxm(&input, scene).expect("write input vxm");

        // Run the prune (guard on, lenient bound so 0.4 keep is accepted).
        let result =
            run_prune(&input, &output, 0.4, 0.5, false).expect("prune runs");

        // The output must load and its count must match the reported kept count.
        let reloaded = load_any(&output).expect("reload output");
        assert_eq!(
            reloaded.len(),
            result.kept.len(),
            "reloaded count {} must equal reported kept {}",
            reloaded.len(),
            result.kept.len()
        );
        assert_eq!(result.kept.len() + result.removed, scene_len);
        assert!(
            result.kept.len() < scene_len,
            "pruning must remove some splats: kept {} of {}",
            result.kept.len(),
            scene_len
        );
        // Energy must be in a valid range.
        assert!(result.energy_retained > 0.0 && result.energy_retained <= 1.0);

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
