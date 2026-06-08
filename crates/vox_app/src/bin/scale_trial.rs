//! Synthetic scale trial — prove (or honestly disprove) that Ochroma's
//! atom-budget pipeline holds at 2M+ splats.
//!
//! This is a headless calibration harness. It procedurally generates a
//! landscape-shaped scene (a 4km × 4km ground carpet + scattered tree/building
//! blobs + a dense town centre) totalling >= 2,000,000 deterministic splats,
//! builds the `AtomBudgetSelector` over it, sweeps a 100-position camera flight
//! path calling `select()` at several budgets, runs one real software-raster
//! frame on the selected subset, and asserts the documented bounds:
//!
//!   * every `select()` returns <= budget splats,
//!   * median `select_us` < 5000 (5 ms),
//!   * build time < 60 s,
//!   * the rendered frame has > 10% non-black pixels.
//!
//! Exits non-zero (with a printed reason) on any failure — if the pipeline has
//! a scaling cliff, that *is* the finding, and we report it rather than
//! weakening the asserts.
//!
//! Run:  `cargo run --release -p vox_app --bin scale_trial`

use std::process::ExitCode;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use half::f16;

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::atom_budget::{AtomBudgetSelector, Selection};
use vox_render::clas;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::tiled_splat_renderer::TiledSplatRenderer;
use vox_render::gpu::GpuContext;
use vox_render::spectral::RenderCamera;

// --- Scene scale knobs (chosen to clear 2M splats deterministically) -------

/// Side length of the square world, metres. 4 km × 4 km.
const WORLD_M: f32 = 4000.0;

/// Ground carpet grid resolution per axis. 1280² = 1,638,400 ground splats.
const GROUND_N: usize = 1280;

/// Number of scattered vegetation/structure blobs across the landscape.
const BLOB_COUNT: usize = 6000;
/// Splats per scattered blob.
const BLOB_SPLATS: usize = 48;
// => 6000 * 48 = 288,000 scatter splats.

/// Dense town-centre footprint, metres (a 300 m × 300 m downtown).
const TOWN_M: f32 = 300.0;
/// Town building-cell grid resolution per axis.
const TOWN_N: usize = 26;
/// Splats per town building (a vertical-ish blob).
const TOWN_BLDG_SPLATS: usize = 180;
// => 26*26 * 180 = 121,680 town splats.
// Grand total ~ 2,048,080 splats.

/// Bytes per splat for the memory estimate (matches `GaussianSplat`'s size).
const SPLAT_BYTES: usize = 96;

/// Cluster target size handed to the BVH builder.
const TARGET_CLUSTER_SIZE: usize = 256;

/// Camera flight path length.
const FLIGHT_FRAMES: usize = 100;

/// Render resolution for the single proof frame.
const RENDER_W: u32 = 640;
const RENDER_H: u32 = 360;

// --- Deterministic hashing (no RNG crate) ----------------------------------

/// Cheap, well-mixed 64-bit integer hash (splitmix64 finalizer).
#[inline]
fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Hash a pair of seeds into a uniform `f32` in `[0, 1)`.
#[inline]
fn hash01(a: u64, b: u64) -> f32 {
    let h = hash_u64(a ^ hash_u64(b.wrapping_mul(0x100_0000_01B3)));
    // top 24 bits -> [0,1)
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// A rolling-hills heightfield over the world plane. Deterministic, no trig
/// tables needed beyond `sin`. Returns a height in metres.
fn ground_height(x: f32, z: f32) -> f32 {
    let xf = x / WORLD_M;
    let zf = z / WORLD_M;
    // Two octaves of smooth waves + a broad valley toward the town centre.
    let h1 = (xf * 6.3).sin() * (zf * 5.1).sin() * 38.0;
    let h2 = (xf * 19.0 + 1.7).sin() * (zf * 17.0 - 0.9).sin() * 11.0;
    let dist = (x * x + z * z).sqrt() / WORLD_M;
    let valley = -22.0 * (1.0 - dist).clamp(0.0, 1.0);
    h1 + h2 + valley
}

/// Encode a flat spectral profile (peak weighted to give visible luminance).
fn spectral_flat(level: f32, warm: bool) -> [u16; 16] {
    std::array::from_fn(|i| {
        let v = if warm {
            // weight long wavelengths (warm) vs short (cool)
            if i >= 8 { level } else { level * 0.55 }
        } else if i < 8 {
            level
        } else {
            level * 0.55
        };
        f16::from_f32(v).to_bits()
    })
}

/// Generate the full deterministic landscape scene.
fn generate_scene() -> Vec<GaussianSplat> {
    let mut splats: Vec<GaussianSplat> =
        Vec::with_capacity(GROUND_N * GROUND_N + BLOB_COUNT * BLOB_SPLATS + TOWN_N * TOWN_N * TOWN_BLDG_SPLATS);

    // --- 1. Ground carpet: a heightfield-shaped grid of surface-ish splats. --
    let step = WORLD_M / GROUND_N as f32;
    let half = WORLD_M * 0.5;
    let ground_spd = spectral_flat(0.45, false); // cool/green ground
    for gz in 0..GROUND_N {
        for gx in 0..GROUND_N {
            let x = gx as f32 * step - half;
            let z = gz as f32 * step - half;
            let y = ground_height(x, z);
            // A little per-cell jitter on opacity so clustering/LOD sorting has
            // real variation to chew on.
            let op = 120
                + (hash_u64((gx as u64 * 73_856_093) ^ (gz as u64 * 19_349_663)) % 120) as u8;
            splats.push(GaussianSplat::volume(
                [x, y, z],
                [step * 0.6, 0.4, step * 0.6],
                Quat::IDENTITY,
                op,
                ground_spd,
            ));
        }
    }

    // --- 2. Scattered tree/building blobs across the landscape. -------------
    for b in 0..BLOB_COUNT {
        let seed = b as u64;
        let bx = (hash01(seed, 1) - 0.5) * WORLD_M;
        let bz = (hash01(seed, 2) - 0.5) * WORLD_M;
        let base_y = ground_height(bx, bz);
        let is_tree = hash01(seed, 3) < 0.7;
        let spd = if is_tree {
            spectral_flat(0.6, false) // green-ish foliage
        } else {
            spectral_flat(0.7, true) // warm structure
        };
        let height = if is_tree { 6.0 } else { 12.0 };
        let radius = if is_tree { 2.2 } else { 4.0 };
        for s in 0..BLOB_SPLATS {
            let ss = s as u64;
            let ang = hash01(seed ^ ss, 10) * std::f32::consts::TAU;
            let rr = hash01(seed ^ ss, 11).sqrt() * radius;
            let hh = hash01(seed ^ ss, 12) * height;
            let px = bx + rr * ang.cos();
            let pz = bz + rr * ang.sin();
            let py = base_y + hh + 1.0;
            let op = 160 + (hash_u64(seed ^ ss.wrapping_mul(2_654_435_761)) % 90) as u8;
            splats.push(GaussianSplat::volume(
                [px, py, pz],
                [0.5, 0.5, 0.5],
                Quat::IDENTITY,
                op,
                spd,
            ));
        }
    }

    // --- 3. Dense town centre: a tight grid of taller building blobs. -------
    let town_step = TOWN_M / TOWN_N as f32;
    let town_half = TOWN_M * 0.5;
    let town_spd = spectral_flat(0.8, true);
    for tz in 0..TOWN_N {
        for tx in 0..TOWN_N {
            let cx = tx as f32 * town_step - town_half;
            let cz = tz as f32 * town_step - town_half;
            let base_y = ground_height(cx, cz);
            // Building height varies per cell deterministically.
            let bh = 8.0 + hash01(tx as u64, tz as u64) * 40.0;
            for s in 0..TOWN_BLDG_SPLATS {
                let ss = s as u64;
                let key = ((tx as u64) << 40) ^ ((tz as u64) << 20) ^ ss;
                let jx = (hash01(key, 1) - 0.5) * town_step * 0.8;
                let jz = (hash01(key, 2) - 0.5) * town_step * 0.8;
                let jy = hash01(key, 3) * bh;
                let op = 180 + (hash_u64(key) % 70) as u8;
                splats.push(GaussianSplat::volume(
                    [cx + jx, base_y + jy + 1.0, cz + jz],
                    [0.7, 0.7, 0.7],
                    Quat::IDENTITY,
                    op,
                    town_spd,
                ));
            }
        }
    }

    splats
}

/// Build a perspective camera looking from `eye` at `target`.
fn camera_at(eye: Vec3, target: Vec3) -> RenderCamera {
    RenderCamera {
        view: Mat4::look_at_rh(eye, target, Vec3::Y),
        proj: Mat4::perspective_rh(
            60f32.to_radians(),
            RENDER_W as f32 / RENDER_H as f32,
            0.5,
            6000.0,
        ),
    }
}

/// A camera flight path: a high orbit that descends into the town centre.
fn flight_path() -> Vec<RenderCamera> {
    let mut cams = Vec::with_capacity(FLIGHT_FRAMES);
    let town = Vec3::new(0.0, ground_height(0.0, 0.0) + 20.0, 0.0);
    for i in 0..FLIGHT_FRAMES {
        let t = i as f32 / (FLIGHT_FRAMES - 1) as f32; // 0..1
        let ang = t * std::f32::consts::TAU; // one full orbit
        // Radius shrinks 1800 -> 120 m; height descends 1200 -> 60 m.
        let radius = 1800.0 * (1.0 - t) + 120.0 * t;
        let height = 1200.0 * (1.0 - t) + 60.0 * t;
        let eye = Vec3::new(radius * ang.cos(), height, radius * ang.sin());
        cams.push(camera_at(eye, town));
    }
    cams
}

/// Median + p99 of a slice of microsecond timings.
fn med_p99(samples: &mut [u64]) -> (u64, u64) {
    if samples.is_empty() {
        return (0, 0);
    }
    samples.sort_unstable();
    let med = samples[samples.len() / 2];
    let p99_idx = ((samples.len() as f32 * 0.99).ceil() as usize).min(samples.len() - 1);
    (med, samples[p99_idx])
}

/// Run one budget sweep over the flight path. Returns
/// `(min_us, med_us, p99_us, max_us, min_selected, max_selected, budget_hits,
///   total_frames, over_budget_count)`.
struct SweepResult {
    min_us: u64,
    med_us: u64,
    p99_us: u64,
    max_us: u64,
    min_selected: usize,
    max_selected: usize,
    /// Frames where selection exactly hit the budget (saturated).
    budget_hits: usize,
    frames: usize,
    /// Frames where selection EXCEEDED the budget (a bound violation).
    over_budget: usize,
    min_culled: usize,
    max_culled: usize,
}

fn run_sweep(
    sel: &mut AtomBudgetSelector,
    path: &[RenderCamera],
    budget: usize,
    out: &mut Selection,
) -> SweepResult {
    let mut times = Vec::with_capacity(path.len());
    let mut min_selected = usize::MAX;
    let mut max_selected = 0usize;
    let mut budget_hits = 0usize;
    let mut over_budget = 0usize;
    let mut min_culled = usize::MAX;
    let mut max_culled = 0usize;

    for cam in path {
        let stats = sel.select(cam, budget, out);
        times.push(stats.select_us);
        min_selected = min_selected.min(stats.selected);
        max_selected = max_selected.max(stats.selected);
        if stats.selected > budget {
            over_budget += 1;
        }
        if stats.selected == budget {
            budget_hits += 1;
        }
        min_culled = min_culled.min(stats.clusters_culled);
        max_culled = max_culled.max(stats.clusters_culled);
    }

    let min_us = *times.iter().min().unwrap_or(&0);
    let max_us = *times.iter().max().unwrap_or(&0);
    let (med_us, p99_us) = med_p99(&mut times);

    SweepResult {
        min_us,
        med_us,
        p99_us,
        max_us,
        min_selected: if min_selected == usize::MAX { 0 } else { min_selected },
        max_selected,
        budget_hits,
        frames: path.len(),
        over_budget,
        min_culled: if min_culled == usize::MAX { 0 } else { min_culled },
        max_culled,
    }
}

/// Build a headless [`GpuContext`] on the local hardware GPU. Returns `None`
/// (the caller then prints "SKIPPED no adapter" and exits 0) when no hardware
/// adapter is available — protecting the green gate on GPU-less CI.
fn headless_context() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let info = adapter.get_info();
    if vox_render::gpu::adapter::ensure_hardware(&info).is_err() {
        return None;
    }
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("scale_trial_gpu_tiled_device"),
            required_features: features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some(GpuContext::from_parts(&device, &queue, &info))
}

/// Run the on-device tiled renderer over `subset` at the proof resolution and
/// print the Done-When line. Exit 0 when >10% non-black; else exit 1 with a
/// printed reason. On no adapter, prints "SKIPPED no adapter" and exits 0.
fn run_gpu_tiled(subset: &[GaussianSplat], render_cam: &RenderCamera) -> ExitCode {
    let Some(ctx) = headless_context() else {
        println!("[scale_trial] SKIPPED no adapter");
        return ExitCode::SUCCESS;
    };
    let mut renderer = match TiledSplatRenderer::new(ctx.clone(), subset, RENDER_W, RENDER_H) {
        Ok(r) => r,
        Err(e) => {
            println!("[scale_trial] gpu_tiled FAIL: renderer construction: {e}");
            return ExitCode::FAILURE;
        }
    };
    let frame = match renderer.render(render_cam) {
        Ok(f) => f,
        Err(e) => {
            println!("[scale_trial] gpu_tiled FAIL: render: {e}");
            return ExitCode::FAILURE;
        }
    };
    let illuminant = Illuminant::d65();
    let (_pixels, non_black) = frame.resolve_to_srgb(&ctx, &illuminant);
    let total = (RENDER_W * RENDER_H) as usize;
    let pct = non_black as f64 / total as f64 * 100.0;

    let (ms, label) = match frame.raster_gpu_ms {
        Some(g) => (g, "GPU"),
        None => (frame.wall_ms, "wall"),
    };
    println!(
        "[scale_trial] gpu_tiled raster {:.3} ms {} | subset_splats={} | non_black_px={}/{} ({:.1}%)",
        ms, label, subset.len(), non_black, total, pct
    );
    if pct > 10.0 {
        ExitCode::SUCCESS
    } else {
        println!(
            "[scale_trial] gpu_tiled FAIL: only {:.1}% non-black (need > 10%)",
            pct
        );
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    println!("[scale_trial] === Ochroma atom-budget scale trial ===");

    // --- 1. Generate ------------------------------------------------------
    let t_gen = Instant::now();
    let scene = generate_scene();
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    let n = scene.len();
    let mem_mb = (n * SPLAT_BYTES) as f64 / (1024.0 * 1024.0);
    println!(
        "[scale_trial] generated splats={} in {:.1} ms | est. mem {:.1} MB ({} B/splat)",
        n, gen_ms, mem_mb, SPLAT_BYTES
    );

    if n < 2_000_000 {
        eprintln!(
            "[scale_trial] FAIL: scene has {} splats (< 2,000,000 required)",
            n
        );
        return ExitCode::FAILURE;
    }

    // --- 2. Build ---------------------------------------------------------
    // Build the selector (the production path), then separately build the
    // raw clusters+BVH so we can print `clas::compute_stats` (the selector
    // does not expose its internals). Both calls use identical inputs, so the
    // stats describe the selector's actual cluster set.
    let t_build = Instant::now();
    let mut sel = AtomBudgetSelector::build(&scene, TARGET_CLUSTER_SIZE);
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let t_stats = Instant::now();
    let clusters = clas::build_clusters(&scene, TARGET_CLUSTER_SIZE);
    let bvh = clas::build_cluster_bvh(&clusters);
    let stats = clas::compute_stats(&clusters, &bvh);
    let stats_ms = t_stats.elapsed().as_secs_f64() * 1000.0;

    println!(
        "[scale_trial] build {:.1} ms | clusters={} (selector reports {}) | splats/cluster avg={:.1} min={} max={} | bvh_depth={} | stats-recompute {:.1} ms",
        build_ms,
        stats.cluster_count,
        sel.cluster_count(),
        stats.avg_splats_per_cluster,
        stats.min_splats_per_cluster,
        stats.max_splats_per_cluster,
        stats.bvh_depth,
        stats_ms,
    );

    // --- 3/4. Camera sweeps at several budgets ----------------------------
    let path = flight_path();
    let mut out = Selection::new();

    let budgets = [24_000usize, 8_000, 100_000];
    let mut sweeps: Vec<(usize, SweepResult)> = Vec::new();
    for &budget in &budgets {
        let r = run_sweep(&mut sel, &path, budget, &mut out);
        println!(
            "[scale_trial] sweep budget={:>6} frames={} select_us[min={} med={} p99={} max={}] selected[min={} max={}] culled[min={} max={}] budget_hits={} over_budget={}",
            budget,
            r.frames,
            r.min_us,
            r.med_us,
            r.p99_us,
            r.max_us,
            r.min_selected,
            r.max_selected,
            r.min_culled,
            r.max_culled,
            r.budget_hits,
            r.over_budget,
        );
        sweeps.push((budget, r));
    }

    // --- 5. One real software-raster frame on the selected subset ---------
    // Pick the final, closest viewpoint (descended into the town) at the
    // primary budget, render the selected subset, count non-black pixels.
    let primary_budget = 24_000usize;
    let render_cam = path[path.len() - 1].clone();
    let render_stats = sel.select(&render_cam, primary_budget, &mut out);
    let subset: Vec<GaussianSplat> = out
        .indices()
        .iter()
        .map(|&i| scene[i as usize])
        .collect();

    // --- 5b. Optional GPU tiled-raster proof (`--gpu-tiled`) ----------------
    // Reuses the selected `subset` + `render_cam`. Runs the on-device tiled
    // chain (tile_assign → radix_sort → tile_range_build → splat_raster) and
    // prints the Done-When line. Exits here (does not continue the CPU sweep
    // assertions) — it is a self-contained renderer proof.
    if std::env::args().any(|a| a == "--gpu-tiled") {
        return run_gpu_tiled(&subset, &render_cam);
    }

    let t_raster = Instant::now();
    let mut rasteriser = SoftwareRasteriser::new(RENDER_W, RENDER_H);
    let illuminant = Illuminant::d65();
    let fb = rasteriser.render(&subset, &render_cam, &illuminant, None);
    let raster_ms = t_raster.elapsed().as_secs_f64() * 1000.0;

    let total_px = (RENDER_W * RENDER_H) as usize;
    let non_black = fb
        .pixels
        .iter()
        .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
        .count();
    let pct = non_black as f64 / total_px as f64 * 100.0;
    println!(
        "[scale_trial] raster {:.1} ms | subset_splats={} | non_black_px={}/{} ({:.1}%)",
        raster_ms, subset.len(), non_black, total_px, pct
    );

    // --- 6. Assertions ----------------------------------------------------
    let mut failures: Vec<String> = Vec::new();

    for (budget, r) in &sweeps {
        if r.over_budget > 0 || r.max_selected > *budget {
            failures.push(format!(
                "budget {} violated: {} frames over budget (max_selected={})",
                budget, r.over_budget, r.max_selected
            ));
        }
    }
    if render_stats.selected > primary_budget {
        failures.push(format!(
            "render-frame selection {} > budget {}",
            render_stats.selected, primary_budget
        ));
    }

    // Primary timing gate: median select at the 24k budget must be < 5 ms.
    let primary = sweeps
        .iter()
        .find(|(b, _)| *b == primary_budget)
        .map(|(_, r)| r)
        .expect("primary budget sweep present");
    if primary.med_us >= 5000 {
        failures.push(format!(
            "median select_us {} >= 5000 (5 ms) at budget {}",
            primary.med_us, primary_budget
        ));
    }

    if build_ms >= 60_000.0 {
        failures.push(format!("build {:.1} ms >= 60000 ms (60 s)", build_ms));
    }

    if pct <= 10.0 {
        failures.push(format!(
            "rendered frame only {:.1}% non-black (need > 10%)",
            pct
        ));
    }

    if !failures.is_empty() {
        eprintln!("[scale_trial] FAIL ({} issue(s)):", failures.len());
        for f in &failures {
            eprintln!("[scale_trial]   - {}", f);
        }
        // Still print the summary line so the run is machine-readable.
        println!(
            "[scale_trial] FAIL splats={} clusters={} build_ms={:.0} select_us_med={}/p99={} budget_hit={}/{} frames={}",
            n,
            stats.cluster_count,
            build_ms,
            primary.med_us,
            primary.p99_us,
            primary.budget_hits,
            primary.frames,
            // total select() calls across all sweeps + the render-frame select
            budgets.len() * path.len() + 1,
        );
        return ExitCode::FAILURE;
    }

    // --- 7. Final machine-readable summary line ---------------------------
    let total_frames = budgets.len() * path.len() + 1;
    println!(
        "[scale_trial] PASS splats={} clusters={} build_ms={:.0} select_us_med={}/p99={} budget_hit={}/{} frames={}",
        n,
        stats.cluster_count,
        build_ms,
        primary.med_us,
        primary.p99_us,
        primary.budget_hits,
        primary.frames,
        total_frames,
    );
    ExitCode::SUCCESS
}
