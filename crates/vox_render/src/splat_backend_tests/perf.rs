use super::super::*;
use super::*;

/// PERF BREAKDOWN (run explicitly with --ignored): per-kernel + per-cost
/// breakdown of the M1 city-block frame on this box. Prints the hard table
/// that proves where the seconds go — march vs normal+shade vs everything
/// else — plus per-ray march counters (steps, empty-space steps, instance
/// evaluations) harvested from the kernel via u_sdf_debug_mode=1.
///
/// Run:
///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
///   SPECTRA_SLANG_DIR=$HOME/src/spectra-perf/slang \
///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release \
///     --lib sdf_city_block_perf_breakdown -- --ignored --nocapture --test-threads=1
#[cfg(feature = "spectra-native")]
#[test]
#[ignore = "perf harness — run explicitly; seconds-per-frame on the iGPU"]
fn sdf_city_block_perf_breakdown() {
    use super::{SdfScenePerfKnobs, pathtrace_sdf_scene_perf};

    let (volumes, instances, eye, center, fov_y, (w, h), rig) = city_block_scene();
    let _n_inst = instances.len() as f64;
    let n_px = (w * h) as usize;

    #[derive(Clone, Copy)]
    struct V {
        spp: u32,
        debug: i32,
        legacy: bool,
        lean: bool,
    }
    let run = |v: V, label: &str| {
        let knobs = SdfScenePerfKnobs {
            debug_mode: v.debug,
            disable_denoiser: true,
            max_bounces: None,
            collect_kernel_timing: true,
            frames: 3,
            legacy_march: v.legacy,
            lean_pipeline: v.lean,
        };
        let t0 = std::time::Instant::now();
        let (rgba, report) = pathtrace_sdf_scene_perf(
            &volumes, &instances, eye, center, fov_y, w, h, v.spp, &rig, &knobs,
        )
        .expect("perf render should succeed");
        let wall_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "\n[perf] ===== {label} (spp={} debug={} legacy_march={} lean={}) =====\n\
                 [perf] frame times ms (f0=cold compile+alloc): {:?}\n\
                 [perf] steady-state frame = {:.1} ms (wall incl. setup {:.1}s) samples_done={}",
            v.spp,
            v.debug,
            v.legacy,
            v.lean,
            report
                .frame_times_ms
                .iter()
                .map(|t| (t * 10.0).round() / 10.0)
                .collect::<Vec<_>>(),
            report.render_time_ms,
            wall_s,
            report.samples_done
        );
        let rows = aggregate_kernel_ms(&report.per_kernel_ms);
        let total_kernel_ms: f32 = rows.iter().map(|r| r.2).sum();
        eprintln!(
            "[perf] {:<28} {:>5} {:>12} {:>8}",
            "kernel", "n", "total ms", "share"
        );
        for (klabel, n, total) in rows.iter().take(8) {
            eprintln!(
                "[perf] {:<28} {:>5} {:>12.1} {:>7.1}%",
                klabel,
                n,
                total,
                100.0 * total / total_kernel_ms
            );
        }
        eprintln!(
            "[perf] {:<28} {:>5} {:>12.1} {:>7.1}%",
            "ALL KERNELS",
            report.per_kernel_ms.len(),
            total_kernel_ms,
            100.0
        );
        (rgba, report)
    };

    // March counter harvest from a debug_mode=1 run's raw film:
    // (marching_rays, steps mean/p99/max, empty mean, evals mean, totals)
    let harvest = |r: &super::SdfScenePerfReport| {
        let mut steps: Vec<f32> = Vec::with_capacity(n_px);
        let mut empty: Vec<f32> = Vec::with_capacity(n_px);
        let mut evals: Vec<f32> = Vec::with_capacity(n_px);
        for i in 0..n_px {
            steps.push(r.raw_film[i * 3]);
            empty.push(r.raw_film[i * 3 + 1]);
            evals.push(r.raw_film[i * 3 + 2]);
        }
        let sum = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
        let marching = steps.iter().filter(|s| **s > 0.0).count();
        let mut sorted = steps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = sorted[((sorted.len() - 1) as f64 * 0.99) as usize];
        let max = *sorted.last().unwrap();
        (
            marching,
            sum(&steps) / n_px as f64, // steps mean (all px)
            sum(&empty) / n_px as f64, // empty mean
            sum(&evals) / n_px as f64, // evals mean
            p99 as f64,
            max as f64,
            sum(&steps),
            sum(&evals),
        )
    };
    let shade_ms = |r: &super::SdfScenePerfReport| -> f32 {
        r.per_kernel_ms
            .iter()
            .filter(|(l, _)| l == "shade_megakernel")
            .map(|(_, ms)| *ms)
            .sum()
    };
    let kernels_total =
        |r: &super::SdfScenePerfReport| -> f32 { r.per_kernel_ms.iter().map(|(_, ms)| *ms).sum() };

    // ---- The variant matrix (all steady-state, frames=3, denoiser off) --
    let (_, w2) = run(
        V {
            spp: 4,
            debug: 2,
            legacy: true,
            lean: false,
        },
        "W2 no-SDF floor",
    );
    let (_, w1) = run(
        V {
            spp: 4,
            debug: 1,
            legacy: true,
            lean: false,
        },
        "W1 LEGACY march-only",
    );
    let (w0_rgba, w0) = run(
        V {
            spp: 4,
            debug: 0,
            legacy: true,
            lean: false,
        },
        "W0 LEGACY full (M1 baseline)",
    );
    let (_, a1) = run(
        V {
            spp: 4,
            debug: 1,
            legacy: false,
            lean: false,
        },
        "A1 FIXA march-only",
    );
    let (a0_rgba, a0) = run(
        V {
            spp: 4,
            debug: 0,
            legacy: false,
            lean: false,
        },
        "A0 FIXA full",
    );
    let (l0_rgba, l0) = run(
        V {
            spp: 1,
            debug: 0,
            legacy: false,
            lean: true,
        },
        "L0 FIXA+LEAN 1spp (realtime-shaped)",
    );
    let (_, l1) = run(
        V {
            spp: 1,
            debug: 0,
            legacy: true,
            lean: true,
        },
        "L1 LEGACY+LEAN 1spp",
    );
    // Lean march isolation (single shade dispatch, no ReSTIR noise):
    let (_, l2) = run(
        V {
            spp: 1,
            debug: 2,
            legacy: false,
            lean: true,
        },
        "L2 LEAN no-SDF floor",
    );
    let (_, l1d) = run(
        V {
            spp: 1,
            debug: 1,
            legacy: true,
            lean: true,
        },
        "L1d LEGACY+LEAN march-only",
    );
    let (_, l0d) = run(
        V {
            spp: 1,
            debug: 1,
            legacy: false,
            lean: true,
        },
        "L0d FIXA+LEAN march-only",
    );

    // ---- Correctness proof: Fix A must render the same city block. ------
    let out_dir = std::env::temp_dir();
    write_png_rgba(
        out_dir
            .join("ochroma_sdf_perf_legacy.png")
            .to_str()
            .unwrap(),
        &w0_rgba,
        w,
        h,
    );
    write_png_rgba(
        out_dir.join("ochroma_sdf_perf_fixa.png").to_str().unwrap(),
        &a0_rgba,
        w,
        h,
    );
    write_png_rgba(
        out_dir.join("ochroma_sdf_perf_lean.png").to_str().unwrap(),
        &l0_rgba,
        w,
        h,
    );
    let mut diff_sum = 0u64;
    let mut diff_max = 0u8;
    let mut diff_cnt = 0usize;
    for i in 0..(n_px * 4) {
        let d = w0_rgba[i].abs_diff(a0_rgba[i]);
        diff_sum += d as u64;
        diff_max = diff_max.max(d);
        if d > 8 {
            diff_cnt += 1;
        }
    }
    let thr = 30.0f32;
    let lit_of = |rgba: &[u8]| {
        (0..n_px)
            .filter(|&p| luma(&rgba[p * 4..p * 4 + 4]) > thr)
            .count()
    };
    let (lit_w0, lit_a0, lit_l0) = (lit_of(&w0_rgba), lit_of(&a0_rgba), lit_of(&l0_rgba));
    eprintln!(
        "\n[perf] image diff legacy vs fixA: mean={:.3}/255 max={} px>8={}/{} | lit px: legacy={} fixA={} lean1spp={}",
        diff_sum as f64 / (n_px * 4) as f64,
        diff_max,
        diff_cnt,
        n_px * 4,
        lit_w0,
        lit_a0,
        lit_l0
    );

    // ---- Counters before/after. -----------------------------------------
    let (mar_b, st_b, em_b, ev_b, p99_b, max_b, tot_st_b, tot_ev_b) = harvest(&w1);
    let (mar_a, st_a, em_a, ev_a, p99_a, max_a, tot_st_a, tot_ev_a) = harvest(&a1);

    // March-only isolation in the LEAN pipeline (1 shade dispatch, no
    // ReSTIR/NRC dispatch noise): debug1 (march, no normal/shade) minus
    // debug2 (no SDF work at all).
    let march_b = shade_ms(&l1d) - shade_ms(&l2);
    let march_a = shade_ms(&l0d) - shade_ms(&l2);
    let _ = (&w1, &a1); // full-pipeline debug variants (counters only)

    eprintln!(
        "\n[perf] ========== HARD TABLE (320x240, 12 instances; steady-state frame) =========="
    );
    eprintln!(
        "[perf] cold frame0 (compile+alloc) ms  : {:.0} (runtime slangc — every fresh Renderer)",
        w0.frame_times_ms[0]
    );
    eprintln!(
        "[perf] steady frame ms                 : LEGACY={:.1}  FIXA={:.1}  FIXA+LEAN(1spp)={:.1}  LEGACY+LEAN={:.1}",
        w0.render_time_ms, a0.render_time_ms, l0.render_time_ms, l1.render_time_ms
    );
    eprintln!(
        "[perf] GPU kernels total ms            : LEGACY={:.1}  FIXA={:.1}  FIXA+LEAN={:.1}  no-SDF floor={:.1}",
        kernels_total(&w0),
        kernels_total(&a0),
        kernels_total(&l0),
        kernels_total(&w2)
    );
    eprintln!(
        "[perf] shade_megakernel ms             : LEGACY={:.1}  FIXA={:.1}  LEAN={:.1}  no-SDF={:.1}",
        shade_ms(&w0),
        shade_ms(&a0),
        shade_ms(&l0),
        shade_ms(&w2)
    );
    eprintln!(
        "[perf] march-only ms (lean, 1 dispatch) : LEGACY={march_b:.1}  FIXA={march_a:.1}  speedup x{:.1}",
        march_b / march_a.max(1e-3)
    );
    eprintln!(
        "[perf] CPU+readback overhead ms        : LEGACY={:.1}  FIXA+LEAN={:.1}",
        w0.render_time_ms - kernels_total(&w0) as f64,
        l0.render_time_ms - kernels_total(&l0) as f64
    );
    eprintln!("[perf] --- march counters (mean over all px; 55.9% of rays enter the AABB) ---");
    eprintln!(
        "[perf] steps/ray mean|p99|max          : LEGACY {st_b:.1}|{p99_b:.0}|{max_b:.0}   FIXA {st_a:.1}|{p99_a:.0}|{max_a:.0}"
    );
    eprintln!(
        "[perf] empty steps/ray mean            : LEGACY {em_b:.1} ({:.1}%)   FIXA {em_a:.1} ({:.1}%)",
        100.0 * em_b / st_b.max(1e-9),
        100.0 * em_a / st_a.max(1e-9)
    );
    eprintln!(
        "[perf] instance evals/ray mean         : LEGACY {ev_b:.1}   FIXA {ev_a:.1}   reduction x{:.1}",
        ev_b / ev_a.max(1e-9)
    );
    eprintln!(
        "[perf] totals/frame@1spp  steps|evals  : LEGACY {tot_st_b:.2e}|{tot_ev_b:.2e}   FIXA {tot_st_a:.2e}|{tot_ev_a:.2e}"
    );
    eprintln!(
        "[perf] marching rays                   : LEGACY {mar_b} (union-AABB entrants)  FIXA {mar_a} (instance-AABB entrants)"
    );
    eprintln!(
        "[perf] ==============================================================================\n"
    );

    // ---- Real-outcome gates. --------------------------------------------
    assert!(
        w0.render_time_ms > 0.0 && shade_ms(&w0) > 0.0,
        "timing sink empty"
    );
    assert!(
        mar_b > 0 && tot_st_b > 0.0,
        "march counters empty — debug mode not wired"
    );
    assert!(
        max_b > 16.0,
        "legacy max steps implausibly low — counters clamped?"
    );
    // Fix A must visit the same surfaces — the IMAGE is the invariant.
    // (The marching-ray count legitimately drops: legacy marches every
    // union-AABB entrant; interval culling only marches rays that enter
    // at least one instance AABB.)
    assert!(
        mar_a > 0 && mar_a <= mar_b,
        "interval culling should march a subset of legacy rays: {mar_b} -> {mar_a}"
    );
    assert!(
        lit_a0 as f64 >= lit_w0 as f64 * 0.98 && lit_a0 as f64 <= lit_w0 as f64 * 1.02,
        "Fix A changed lit coverage: {lit_w0} -> {lit_a0}"
    );
    assert!(
        diff_sum as f64 / ((n_px * 4) as f64) < 1.0,
        "Fix A image diverged from legacy: mean diff {:.3}/255",
        diff_sum as f64 / (n_px * 4) as f64
    );
    // Fix A must actually reduce the work, not just match the image.
    assert!(
        ev_a < ev_b * 0.5,
        "Fix A failed to cut instance evals: {ev_b:.1} -> {ev_a:.1}"
    );
}

/// 720p PROBE (run explicitly with --ignored): the realtime-shaped frame
/// (interval-culled march + lean pipeline, 1 spp) at the design doc's
/// contract-point internal resolution, measured on THIS box — the number
/// the 4070 Ti projection scales from (no assumed pixel-scaling).
#[cfg(feature = "spectra-native")]
#[test]
#[ignore = "perf harness — run explicitly"]
fn sdf_city_block_perf_720p() {
    use super::{SdfScenePerfKnobs, pathtrace_sdf_scene_perf};

    let (volumes, instances, eye, center, fov_y, _wh, rig) = city_block_scene();
    let (w, h) = (1280u32, 720u32);
    let knobs = SdfScenePerfKnobs {
        debug_mode: 0,
        disable_denoiser: true,
        max_bounces: None,
        collect_kernel_timing: true,
        frames: 4,
        legacy_march: false,
        lean_pipeline: true,
    };
    let (rgba, report) = pathtrace_sdf_scene_perf(
        &volumes, &instances, eye, center, fov_y, w, h, 1, &rig, &knobs,
    )
    .expect("720p perf render should succeed");
    let knobs_legacy = SdfScenePerfKnobs {
        legacy_march: true,
        ..knobs.clone()
    };
    let (_, report_legacy) = pathtrace_sdf_scene_perf(
        &volumes,
        &instances,
        eye,
        center,
        fov_y,
        w,
        h,
        1,
        &rig,
        &knobs_legacy,
    )
    .expect("720p legacy perf render should succeed");

    let rows = aggregate_kernel_ms(&report.per_kernel_ms);
    eprintln!("\n[perf720] ===== 1280x720, 1spp, lean, interval-culled march =====");
    eprintln!(
        "[perf720] frame times ms: {:?}",
        report
            .frame_times_ms
            .iter()
            .map(|t| (t * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
    for (label, n, total) in rows.iter() {
        eprintln!("[perf720] {label:<24} n={n} {total:>8.1} ms");
    }
    let kernels: f32 = report.per_kernel_ms.iter().map(|(_, ms)| ms).sum();
    let kernels_legacy: f32 = report_legacy.per_kernel_ms.iter().map(|(_, ms)| ms).sum();
    let n_px = (w * h) as usize;
    let thr = 30.0f32;
    let lit = (0..n_px)
        .filter(|&p| luma(&rgba[p * 4..p * 4 + 4]) > thr)
        .count();
    eprintln!(
        "[perf720] steady frame: FIXA+LEAN={:.1} ms (kernels {:.1}) | LEGACY+LEAN={:.1} ms (kernels {:.1}) | lit coverage {:.3}",
        report.render_time_ms,
        kernels,
        report_legacy.render_time_ms,
        kernels_legacy,
        lit as f64 / n_px as f64
    );
    let png = std::env::temp_dir().join("ochroma_sdf_perf_720p.png");
    write_png_rgba(png.to_str().unwrap(), &rgba, w, h);
    eprintln!("[perf720] wrote {}", png.display());
    assert!(report.render_time_ms > 0.0 && kernels > 0.0);
    assert!(
        lit as f64 / n_px as f64 > 0.15,
        "720p frame lost the city block (lit {:.3})",
        lit as f64 / n_px as f64
    );
}
