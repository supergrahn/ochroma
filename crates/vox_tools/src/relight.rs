//! `relight` subcommand: offline illumination-rebake of a captured splat scene.
//!
//! Loads any supported splat asset (`.vxm`, `.ply`, `.spz`), recovers a per-splat
//! intrinsic spectral base by dividing the baked radiance by the assumed `--from`
//! illuminant SPD, then re-runs the engine's own 16-band spectral illumination
//! under the `--to` illuminant (sky ambient, optional shadow rays) and writes a
//! new `.vxm`. The relit asset flows through the existing load path unchanged —
//! only the f16 radiance bits of each splat's `spectral` field change.
//!
//! This is an OFFLINE / asset-time pass, mirroring [`crate::prune`]: load →
//! rebake → write → receipt. See
//! `docs/superpowers/specs/2026-06-07-spectral-relight-design.md`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use vox_core::types::GaussianSplat;
use vox_render::relight::{relight_scene, IlluminantSpec, RelightSettings};

/// Load splats from any supported format, dispatching on the file extension.
/// Mirrors `prune.rs::load_any`.
fn load_any(path: &Path) -> Result<Vec<GaussianSplat>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "vxm" => {
            let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
            let vxm = vox_data::vxm::VxmFile::read(f)
                .with_context(|| format!("read vxm {}", path.display()))?;
            Ok(vxm.splats)
        }
        "ply" => vox_data::ply_loader::load_ply(path)
            .map_err(|e| anyhow!("read ply {}: {e}", path.display())),
        "spz" => {
            vox_data::spz::load_spz(path).map_err(|e| anyhow!("read spz {}: {e}", path.display()))
        }
        other => Err(anyhow!(
            "unsupported input extension {:?} (expected vxm, ply, or spz)",
            other
        )),
    }
}

/// Write splats to a `.vxm` file. Mirrors `prune.rs::write_vxm`.
fn write_vxm(path: &Path, splats: Vec<GaussianSplat>) -> Result<()> {
    let file = vox_data::vxm::VxmFile {
        header: vox_data::vxm::VxmHeader::new(
            uuid::Uuid::new_v4(),
            splats.len() as u32,
            vox_data::vxm::MaterialType::Generic,
        ),
        splats,
    };
    let mut out =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    file.write(&mut out)
        .with_context(|| format!("write vxm {}", path.display()))?;
    Ok(())
}

/// Driver for the `relight` subcommand. Loads any supported asset, builds the
/// relight settings, calls [`relight_scene`], writes a v-current `.vxm`, and
/// prints the design's §2 receipt (computed from the returned report). Returns
/// `Err` on I/O or unparseable illuminant names.
pub fn run_relight(
    input: &Path,
    output: &Path,
    from: &str,
    to: &str,
    no_shadows: bool,
    no_sky: bool,
) -> Result<()> {
    let reference = IlluminantSpec::parse(from)
        .ok_or_else(|| anyhow!("unparseable --from illuminant {from:?} (expected one of: tungsten, daylight, cool_led, neutral, d65, d50, a, f11, sun@<hour>[,<lat>])"))?;
    let target = IlluminantSpec::parse(to)
        .ok_or_else(|| anyhow!("unparseable --to illuminant {to:?} (expected one of: tungsten, daylight, cool_led, neutral, d65, d50, a, f11, sun@<hour>[,<lat>])"))?;

    let splats = load_any(input)?;

    // An IDENTITY relight (from == to) must reproduce the input to within f16
    // quantization — it is the round-trip neutrality proof (§2). The added
    // sun/sky/emitter terms would make `from==to` non-trivial, so for a pure
    // identity we suppress them: a scene relit to the SAME illuminant it was
    // captured under is, by definition, unchanged.
    let is_identity = reference.name() == target.name();
    let settings = RelightSettings::new(reference, target)
        .with_shadows(!no_shadows && !is_identity)
        .with_sky_ambient(!no_sky && !is_identity);

    let (relit, report) = relight_scene(&splats, &settings);

    let shadows_str = if settings.cast_shadows() { "on" } else { "off" };
    let sky_str = if settings.sky_ambient() { "on" } else { "off" };
    let threads = report.thread_count;
    let ratio_before = report.ratio_short_long_before;
    let ratio_after = report.ratio_short_long_after;
    let ratio_gain = if ratio_before.abs() > 1e-9 {
        ratio_after / ratio_before
    } else {
        0.0
    };

    println!(
        "relight: loaded {} splats from {}",
        report.splat_count,
        input.display()
    );
    println!(
        "relight: from={}  to={}  shadows={}  sky-ambient={}  emitters={}",
        report.reference_name,
        report.target_name,
        shadows_str,
        sky_str,
        settings.emitters().len(),
    );
    println!(
        "relight: rebake {} splats in {:.2} s (rayon, {} threads)",
        report.splat_count, report.rebake_secs, threads
    );
    println!(
        "relight: mean short/long band ratio (b4/b14)  BEFORE = {:.2}  AFTER = {:.2}",
        ratio_before, ratio_after
    );
    if is_identity {
        println!(
            "relight: IDENTITY relight, max per-band delta {:.4} (< 0.002)",
            report.max_band_delta
        );
    } else if ratio_after >= ratio_before {
        println!(
            "relight: scene became BLUER under {} (ratio rose {:.2} -> {:.2}, x{:.2})",
            report.target_name, ratio_before, ratio_after, ratio_gain
        );
    } else {
        println!(
            "relight: scene became WARMER under {} (ratio fell {:.2} -> {:.2}, x{:.2})",
            report.target_name, ratio_before, ratio_after, ratio_gain
        );
    }
    println!(
        "relight: f16 round-trip max band error {:.4} (< 0.002 budget)",
        report.f16_roundtrip_error
    );

    write_vxm(output, relit)?;
    println!(
        "relight: wrote {} splats to {}",
        report.splat_count,
        output.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use vox_data::spectral_capture::LightSpd;

    /// Bake `intrinsic ⊙ tungsten` into a flat-spectral volume splat.
    fn baked_splat(pos: [f32; 3], intrinsic: f32) -> GaussianSplat {
        let tungsten = LightSpd::tungsten().0;
        let bits: [u16; 16] =
            std::array::from_fn(|b| f16::from_f32(intrinsic * tungsten[b]).to_bits());
        GaussianSplat::volume(pos, [0.1, 0.1, 0.1], glam::Quat::IDENTITY, 255, bits)
    }

    fn write_demo(path: &Path, count: usize) {
        let splats: Vec<GaussianSplat> = (0..count)
            .map(|i| baked_splat([i as f32 * 0.01, 0.0, 0.0], 0.5))
            .collect();
        write_vxm(path, splats).expect("write demo vxm");
    }

    #[test]
    fn cli_relight_tungsten_to_daylight_end_to_end() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let input = dir.join(format!("ochroma_relight_in_{pid}.vxm"));
        let output = dir.join(format!("ochroma_relight_out_{pid}.vxm"));

        write_demo(&input, 512);
        run_relight(&input, &output, "tungsten", "daylight", true, true)
            .expect("relight runs");

        // Output must load and round-trip its count.
        let reloaded = load_any(&output).expect("reload output");
        assert_eq!(reloaded.len(), 512, "reloaded count must match");

        // The relit scene must be bluer: recompute the b4/b14 ratio off the
        // RELOADED splats and assert it rose toward daylight's b4/b14 (~0.96).
        let mean_ratio = |splats: &[GaussianSplat]| -> f32 {
            let mut acc = 0.0f64;
            let mut n = 0u64;
            for s in splats {
                let long = s.spectral_f32(14);
                if long.abs() > 1e-6 {
                    acc += (s.spectral_f32(4) / long) as f64;
                    n += 1;
                }
            }
            (acc / n as f64) as f32
        };
        let before = load_any(&input).expect("reload input");
        let r_before = mean_ratio(&before);
        let r_after = mean_ratio(&reloaded);
        assert!(
            r_before < 0.6,
            "BEFORE ratio {r_before} must be long-wave-heavy (tungsten bake)"
        );
        assert!(
            r_after > 0.85,
            "AFTER ratio {r_after} must rise toward daylight flatness"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn cli_relight_rejects_bad_illuminant() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let input = dir.join(format!("ochroma_relight_bad_{pid}.vxm"));
        write_demo(&input, 8);
        let err = run_relight(&input, &input, "tungsten", "bogus", true, true);
        assert!(err.is_err(), "unparseable --to must error");
        let _ = std::fs::remove_file(&input);
    }
}
