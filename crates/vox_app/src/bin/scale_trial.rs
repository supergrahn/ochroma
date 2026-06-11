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
//!
//! `--instanced [--buildings N] [--gate-ms G]` runs the virtualized-rendering
//! gate harness (M3): N synthetic buildings instanced from a shared
//! `AssetAtomLibrary`, GPU-resident select (`select_resident` — the budget
//! walk runs on device, draws never visit the host) → indirect GPU expand
//! (`encode_indirect`) → tiled render over a ≥ 120-frame governed orbit at
//! 1280×720, printing real p50/p99 frame milliseconds, writing
//! `scale_trial_instanced.png`, and asserting in-binary: p50 ≤ gate
//! (default 16.6 = the 60 fps target defined on the tomespensin RTX 4070 Ti;
//! the 780M dev box runs `--gate-ms 33`, its 30 fps tracking floor — the
//! printed gate line always names tier + adapter), `entry_overflow max=0`,
//! `gpu_fallbacks=0`, coverage ≥ 25% at ≥ 10,000 buildings, plus the M1/M2
//! invariants (budget band, truncation on the fallback path, black frame).
//! Exit 1 with a printed reason on any violation.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use half::f16;

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::atom_budget::{AtomBudgetSelector, Selection};
use vox_app::shell::cpu_render;
use vox_render::atom_instances::{
    AssetAtomLibrary, AtomInstance, FrameBudgetGovernor, InstancedSelection,
};
use vox_render::clas;
use vox_render::gpu::expand_draws::ExpandDrawsPass;
use vox_render::gpu::instanced_select_gpu::{InstancedSelectGpu, SelectPath};
use vox_render::gpu::resident_gi_raster::ResidentGiRaster;
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
    let entries = renderer.last_entry_count();
    let entries_per_splat = entries as f64 / subset.len().max(1) as f64;
    println!(
        "[scale_trial] gpu_tiled raster {:.3} ms {} | subset_splats={} | tile_entries={} ({:.2}/splat) | non_black_px={}/{} ({:.1}%)",
        ms, label, subset.len(), entries, entries_per_splat, non_black, total, pct
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

/// Read layers 0 and 1 (the 8 spectral bands) of a resident-render output
/// texture back to the host and count pixels with any non-zero band. This is the
/// proof readback for the resident path (whose `render_frame` returns a raw
/// `wgpu::Texture`, not a `TiledFrame`). Returns `(non_black_count, total)`.
fn count_nonblack_spectral(
    ctx: &GpuContext,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
) -> (usize, usize) {
    let device = ctx.device();
    let queue = ctx.queue();
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = (width * 16).div_ceil(align) * align;
    let layer_bytes = (padded_bpr * height) as u64;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resident_proof_readback"),
        size: layer_bytes * 2,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("resident_proof_enc"),
    });
    for layer in 0u32..2 {
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: layer as u64 * layer_bytes,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let total = (width * height) as usize;
    if !matches!(rx.recv(), Ok(Ok(()))) {
        return (0, total);
    }
    let mut non_black = 0usize;
    {
        let data = slice.get_mapped_range();
        // Read f32 directly from the mapped bytes (no bytemuck dep in vox_app).
        let f32_at = |byte_idx: usize| -> f32 {
            f32::from_le_bytes([
                data[byte_idx],
                data[byte_idx + 1],
                data[byte_idx + 2],
                data[byte_idx + 3],
            ])
        };
        let padded_bpr_b = padded_bpr as usize;
        let layer1_off_b = layer_bytes as usize;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let b0 = y * padded_bpr_b + x * 16; // 16 bytes/texel (rgba32f)
                let b1 = layer1_off_b + y * padded_bpr_b + x * 16;
                let any = (0..4).any(|k| f32_at(b0 + k * 4) != 0.0)
                    || (0..4).any(|k| f32_at(b1 + k * 4) != 0.0);
                if any {
                    non_black += 1;
                }
            }
        }
    }
    readback.unmap();
    (non_black, total)
}

/// Run the RESIDENT GI -> raster path (AAA Spec 11) over `subset`: GI compute
/// output bound directly into the rasterizer with ZERO CPU readback between GI
/// and the raster (no per-frame GI poll). Prints the Done-When line. Exits 0 when
/// >10% non-black AND `map_async/frame == 0`; else 1. On no adapter, prints
/// "SKIPPED no adapter" and exits 0.
fn run_gpu_resident(subset: &[GaussianSplat], render_cam: &RenderCamera) -> ExitCode {
    let Some(ctx) = headless_context() else {
        println!("[scale_trial] SKIPPED no adapter");
        return ExitCode::SUCCESS;
    };
    let mut resident = match ResidentGiRaster::new(ctx.clone(), subset, RENDER_W, RENDER_H) {
        Ok(r) => r,
        Err(e) => {
            println!("[scale_trial] gpu_resident FAIL: construction: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Noon hour drives a non-trivial sky-ambient GI term.
    let tex = match resident.render_frame(render_cam, 12.0) {
        Ok(t) => t,
        Err(e) => {
            println!("[scale_trial] gpu_resident FAIL: render: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (non_black, total) = count_nonblack_spectral(&ctx, &tex, RENDER_W, RENDER_H);
    let pct = non_black as f64 / total as f64 * 100.0;
    let map_async = resident.map_async_count();

    println!(
        "[scale_trial] gpu_resident frames=1 | non_black_px={}/{} ({:.1}%) | map_async/frame={}",
        non_black, total, pct, map_async
    );
    if pct > 10.0 && map_async == 0 {
        ExitCode::SUCCESS
    } else {
        if pct <= 10.0 {
            println!("[scale_trial] gpu_resident FAIL: only {:.1}% non-black (need > 10%)", pct);
        }
        if map_async != 0 {
            println!(
                "[scale_trial] gpu_resident FAIL: map_async/frame={} (need 0 — GI poll not eliminated)",
                map_async
            );
        }
        ExitCode::FAILURE
    }
}

// --- Instanced gate harness (`--instanced`, M3) -----------------------------
//
// The 60 fps dev-floor gate: GPU select + governed budget + PNG artifact,
// with p50 ≤ 16.6 ms / overflow / fallback / coverage asserted in-binary.

/// Render resolution for the instanced trial (pinned by the plan).
const INST_W: u32 = 1280;
const INST_H: u32 = 720;
/// Renderer/expand CAPACITY — the fixed upper bound the GPU chain is built
/// for. M3's `FrameBudgetGovernor` varies the per-frame budget BELOW this;
/// the capacity itself is unchanged at 1,000,000.
const INST_BUDGET_CAP: usize = 1_000_000;
/// Unmeasured settle frames before the measured orbit: the governor converges
/// on its budget over the same code path before any percentile is recorded.
const INST_WARMUP: usize = 30;
/// Distinct synthetic building assets shared (deduplicated) by all instances.
const INST_ASSETS: usize = 5;
/// Atoms per building asset (~5,000 per the plan → 10k × 5k = 50M virtual).
const INST_ASSET_ATOMS: usize = 5000;
/// Grid spacing between building instances, metres (the proven city-scale
/// selector-test geometry: 100×100 × 4 m ⇒ ±198 m).
const INST_SPACING: f32 = 4.0;
/// Orbit length (the plan requires ≥ 120 measured frames).
const INST_FRAMES: usize = 120;

/// One deterministic hash-jittered building blob: a box-ish volume of
/// `INST_ASSET_ATOMS` atoms rooted at the local origin (asset-local space —
/// instances place it). No RNG crate.
fn building_asset(asset: usize) -> Vec<GaussianSplat> {
    let seed = 0xB17D_0000u64 + asset as u64;
    let half_w = 1.2 + hash01(seed, 100) * 0.6; // footprint half-extents 1.2–1.8 m
    let half_d = 1.2 + hash01(seed, 101) * 0.6;
    let height = 8.0 + hash01(seed, 102) * 12.0; // 8–20 m tall
    let warm = asset % 2 == 0;
    (0..INST_ASSET_ATOMS)
        .map(|i| {
            let key = seed ^ (i as u64).wrapping_mul(0x9E37_79B9);
            let px = (hash01(key, 1) - 0.5) * 2.0 * half_w;
            let pz = (hash01(key, 2) - 0.5) * 2.0 * half_d;
            let py = hash01(key, 3) * height;
            let op = 160 + (hash_u64(key) % 90) as u8;
            let level = 0.45 + hash01(key, 4) * 0.4;
            GaussianSplat::volume(
                [px, py, pz],
                [0.3, 0.3, 0.3],
                Quat::IDENTITY,
                op,
                spectral_flat(level, warm),
            )
        })
        .collect()
}

/// Place `n` building instances on a city-like square grid with deterministic
/// quarter-turn yaws, assets round-robin (so `virtual = n × INST_ASSET_ATOMS`
/// exactly when every asset has `INST_ASSET_ATOMS` atoms). Returns the
/// instances and the grid half-extent in metres (drives the orbit).
fn building_instances(n: usize) -> (Vec<AtomInstance>, f32) {
    let side = (n as f32).sqrt().ceil().max(1.0) as usize;
    let half = (side - 1) as f32 * 0.5;
    let instances = (0..n)
        .map(|i| {
            let gx = (i % side) as f32 - half;
            let gz = (i / side) as f32 - half;
            let q = Quat::from_rotation_y((i % 4) as f32 * std::f32::consts::FRAC_PI_2);
            AtomInstance::new(
                (i % INST_ASSETS) as u32,
                i as u32,
                [gx * INST_SPACING, 0.0, gz * INST_SPACING],
                [q.x, q.y, q.z, q.w],
            )
        })
        .collect();
    (instances, half * INST_SPACING)
}

/// Orbit camera for the measured loop, scaled to the grid: a high ring looking
/// steeply down just inside itself with a wide FOV — the proven city-scale
/// selector-test geometry (high + wide ⇒ the < 150 m near disk is almost fully
/// in frustum ⇒ thousands of cluster-granular instances ⇒ the budget walk can
/// spend its budget; the oracle walk never promotes ABOVE a cluster's distance
/// LOD, so saturation must come from instance count, not promotion).
fn instanced_orbit_camera(frame: usize, grid_half: f32) -> RenderCamera {
    let altitude = (grid_half * 0.5).clamp(24.0, 100.0);
    let ring = (grid_half * 0.28).clamp(10.0, 55.0);
    let ang = frame as f32 / INST_FRAMES as f32 * std::f32::consts::TAU;
    let eye = Vec3::new(ring * ang.cos(), altitude, ring * ang.sin());
    // Aim halfway between the orbit ring's ground point and the grid centre:
    // a steep look-down that keeps the city under the camera on screen.
    let target = Vec3::new(eye.x * 0.5, 0.0, eye.z * 0.5);
    RenderCamera {
        view: Mat4::look_at_rh(eye, target, Vec3::Y),
        proj: Mat4::perspective_rh(2.2, INST_W as f32 / INST_H as f32, 0.5, 2000.0),
    }
}

/// The `--instanced` gate harness: library → instances → ONE
/// `new_with_capacity` renderer + ONE `ExpandDrawsPass` + ONE
/// `InstancedSelectGpu` on the shared context → per frame
/// `select → encode+submit → set_active_splat_count → render`, wall-clocking
/// the whole span (`raster_gpu_ms` is `None` on this chain — wall IS the
/// honest measure). The per-frame budget is closed-loop: a
/// `FrameBudgetGovernor` (14.5 ms setpoint) reads `budget()` before every
/// select and is fed the full-loop wall ms after every frame, over 30
/// unmeasured warmup frames and then the ≥ 120 measured ones. The final frame
/// resolves to sRGB and is written to `scale_trial_instanced.png` (the
/// human-visible artifact). Exits 1 with a printed reason on ANY gate
/// violation: p50 > `gate_ms` (default 16.6 — the 60 fps target defined on
/// the `tomespensin` RTX 4070 Ti; the 780M dev box asserts the 33 ms
/// tracking floor explicitly via `--gate-ms 33`; the gate line always names
/// its tier + adapter so no number travels without its hardware context),
/// `entry_overflow max != 0`, GPU near-pair fallbacks != 0, coverage < 25%
/// at ≥ 10,000 buildings, or the M1/M2 invariants (selection over the
/// frame's governed budget, per-frame budget band not met, expand
/// truncation, a black frame).
fn run_instanced(buildings: usize, gate_ms: f64, min_budget: usize) -> ExitCode {
    let Some(ctx) = headless_context() else {
        println!("[scale_trial] SKIPPED no adapter");
        return ExitCode::SUCCESS;
    };

    // --- Cook the shared library + place the instances. --------------------
    let t_build = Instant::now();
    let sources: Vec<Vec<GaussianSplat>> = (0..INST_ASSETS).map(building_asset).collect();
    let library = Arc::new(AssetAtomLibrary::build(&sources, 128));
    let (instances, grid_half) = building_instances(buildings);
    let virtual_atoms: usize = instances
        .iter()
        .map(|inst| sources[inst.asset() as usize].len())
        .sum();
    let library_atoms = library.total_atoms();
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    // --- Renderer + expand pass: constructed ONCE at CAPACITY, never
    // rebuilt — the governed budget moves below this fixed ceiling.
    let mut renderer = match TiledSplatRenderer::new_with_capacity(
        ctx.clone(),
        INST_BUDGET_CAP as u32,
        INST_W,
        INST_H,
    ) {
        Ok(r) => r,
        Err(e) => {
            println!("[scale_trial] instanced FAIL: renderer construction: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut pass = match ExpandDrawsPass::new_with_context(
        &ctx,
        &library,
        instances.len() as u32,
        INST_BUDGET_CAP as u32,
    ) {
        Ok(p) => p,
        Err(e) => {
            println!("[scale_trial] instanced FAIL: expand pass construction: {e}");
            return ExitCode::FAILURE;
        }
    };
    // GPU selector on the SAME shared context (one device, zero extra
    // adapters); 262,144 near units is the pinned pair-band capacity — the
    // internal CPU fallback stays available but its use FAILS the gate.
    let mut gpu_sel = match InstancedSelectGpu::new_with_context(
        &ctx,
        library.clone(),
        instances.len() as u32,
        262_144,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("[scale_trial] instanced FAIL: gpu selector construction: {e}");
            return ExitCode::FAILURE;
        }
    };
    gpu_sel.set_instances(&instances);

    println!(
        "[scale_trial] instanced setup: assets={} library_atoms={} resident={:.1} MiB | instances={} | capacity={} | build {:.1} ms | adapter={}",
        library.asset_count(),
        library_atoms,
        library.resident_bytes() as f64 / (1024.0 * 1024.0),
        instances.len(),
        INST_BUDGET_CAP,
        build_ms,
        ctx.adapter_name(),
    );

    // --- The closed-loop orbit: 30 governor-settle warmup frames (nothing
    // recorded), then the ≥ 120 measured ones. `governor.budget()` feeds
    // every select; `governor.update(frame_ms)` runs after EVERY frame.
    let mut governor = FrameBudgetGovernor::new(14.5, 300_000, min_budget, INST_BUDGET_CAP);
    let mut frame_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut select_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    // M3.1 Task 1 — per-stage spans (u64 µs vecs, p50 via `med_p99`): the
    // select breakdown (host spans + optional kernel timestamps), the expand
    // encode+submit span, and the renderer's assign/clear/chain spans.
    let mut gpu_span_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut k1_gpu_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut k2_gpu_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut walk_gpu_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut pair_build_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut assemble_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut walk_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut expand_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut assign_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut clear_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut chain_us: Vec<u64> = Vec::with_capacity(INST_FRAMES);
    let mut last_syncs = 0u32;
    // (budget_used, selected) per MEASURED frame — the 0.9-band invariant is
    // per-frame now that the budget moves under the governor.
    let mut frame_budgets: Vec<(usize, usize)> = Vec::with_capacity(INST_FRAMES);
    let mut last_selected = 0usize;
    let mut last_budget = governor.budget();
    let mut last_entries = 0u32;
    let mut entries_max = 0u32;
    let mut overflow_max = 0u32;
    let mut sel = InstancedSelection::new();
    let mut last_frame = None;

    for f in 0..(INST_WARMUP + INST_FRAMES) {
        let measured = f >= INST_WARMUP;
        let orbit = (if measured { f - INST_WARMUP } else { f }) % INST_FRAMES;
        let cam = instanced_orbit_camera(orbit, grid_half);
        let budget_used = governor.budget();
        let t0 = Instant::now();
        // M3.2 Task 4 — the resident path: the budget walk runs on device,
        // draws are emitted GPU-side and consumed by the indirect expand; the
        // host sees only the 64 B stats block.
        let (stats, path) = match gpu_sel.select_resident(&cam, budget_used, &mut sel) {
            Ok(s) => s,
            Err(e) => {
                println!("[scale_trial] instanced FAIL: gpu select frame {f}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let breakdown = gpu_sel.last_breakdown();
        let t_expand = Instant::now();
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scale_trial_instanced_expand"),
            });
        match path {
            SelectPath::GpuResident => {
                pass.encode_indirect(
                    &mut enc,
                    &gpu_sel,
                    &instances,
                    renderer.splat_buf(),
                    renderer.transform_buf(),
                );
                ctx.queue().submit(Some(enc.finish()));
                renderer.set_active_splat_count(stats.selected as u32);
            }
            SelectPath::CpuFallback => {
                // The host bridge (near-pair overflow): `sel` holds the CPU
                // twin's selection; route it through the host encode. ANY use
                // of this branch fails the run via the unchanged
                // `gpu_fallbacks=0` gate. The M1/M2 truncation check applies
                // only here — the resident path has no host selection to
                // compare; its invariants are the per-frame stats-vs-band
                // checks in the failures vec below.
                let n = pass.encode(
                    &mut enc,
                    &sel,
                    &instances,
                    renderer.splat_buf(),
                    renderer.transform_buf(),
                );
                ctx.queue().submit(Some(enc.finish()));
                renderer.set_active_splat_count(n);
                if n as usize != sel.atom_count() {
                    println!(
                        "[scale_trial] instanced FAIL: expand truncated frame {f}: encoded {} of {} selected atoms",
                        n,
                        sel.atom_count()
                    );
                    return ExitCode::FAILURE;
                }
            }
        }
        let expand_elapsed = t_expand.elapsed();
        let frame = match renderer.render(&cam) {
            Ok(fr) => fr,
            Err(e) => {
                println!("[scale_trial] instanced FAIL: render frame {f}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let elapsed = t0.elapsed();
        // The honest full select+expand+render loop wall ms closes the loop —
        // `raster_gpu_ms` is hard-coded `None` on this chain (frozen passes).
        governor.update(elapsed.as_secs_f64() as f32 * 1000.0);

        overflow_max = overflow_max.max(renderer.tile_entry_overflow());
        if measured {
            let entries = renderer.last_entry_count();
            entries_max = entries_max.max(entries);
            frame_us.push(elapsed.as_micros() as u64);
            select_us.push(stats.select_us);
            frame_budgets.push((budget_used, stats.selected));
            last_selected = stats.selected;
            last_budget = budget_used;
            last_entries = entries;
            // Stage spans for this measured frame (f32 ms → u64 µs).
            let ms_to_us = |ms: f32| (ms.max(0.0) as f64 * 1000.0) as u64;
            gpu_span_us.push(ms_to_us(breakdown.gpu_span_ms));
            if let Some(k1) = breakdown.k1_gpu_ms {
                k1_gpu_us.push(ms_to_us(k1));
            }
            if let Some(k2) = breakdown.k2_gpu_ms {
                k2_gpu_us.push(ms_to_us(k2));
            }
            if let Some(wg) = breakdown.walk_gpu_ms {
                walk_gpu_us.push(ms_to_us(wg));
            }
            pair_build_us.push(ms_to_us(breakdown.pair_build_ms));
            assemble_us.push(ms_to_us(breakdown.assemble_ms));
            walk_us.push(ms_to_us(breakdown.walk_ms));
            expand_us.push(expand_elapsed.as_micros() as u64);
            assign_us.push(ms_to_us(frame.assign_ms));
            clear_us.push(ms_to_us(frame.clear_ms));
            chain_us.push(ms_to_us(frame.wall_ms));
            last_syncs = breakdown.syncs;
            last_frame = Some(frame);
        }
    }

    // --- Non-black proof + human-visible artifact: final frame only
    // (readback is proof, not hot path). The PNG goes through the zero-dep
    // writer (vox_app has no bytemuck/image — `as_flattened` does the cast).
    let (pixels, non_black) = last_frame
        .expect("INST_FRAMES > 0")
        .resolve_to_srgb(&ctx, &Illuminant::d65());
    let total_px = (INST_W * INST_H) as usize;
    let pct = non_black as f64 / total_px as f64 * 100.0;
    let png_path = "scale_trial_instanced.png";
    if let Err(e) = cpu_render::write_png(png_path, pixels.as_flattened(), INST_W, INST_H) {
        println!("[scale_trial] instanced FAIL: write {png_path}: {e}");
        return ExitCode::FAILURE;
    }

    let (p50_us, p99_us) = med_p99(&mut frame_us);
    let (sel_p50_us, _) = med_p99(&mut select_us);

    // M3.1 Task 1 — per-stage p50s (ms). `chain` = `TiledFrame::wall_ms`;
    // `stage_sum` = gpu_span+pair_build+assemble+walk+expand+assign+clear+chain;
    // the residual `frame − stage_sum` is host encode/validation time, printed
    // implicitly by the two numbers. k1/k2 are GPU-internal (inside gpu_span),
    // shown for attribution, NOT summed; "n/a" when timestamps are unavailable.
    let p50_ms_of = |v: &mut Vec<u64>| med_p99(v).0 as f64 / 1000.0;
    let gpu_span_p50 = p50_ms_of(&mut gpu_span_us);
    let pair_build_p50 = p50_ms_of(&mut pair_build_us);
    let assemble_p50 = p50_ms_of(&mut assemble_us);
    let walk_p50 = p50_ms_of(&mut walk_us);
    let expand_p50 = p50_ms_of(&mut expand_us);
    let assign_p50 = p50_ms_of(&mut assign_us);
    let clear_p50 = p50_ms_of(&mut clear_us);
    let chain_p50 = p50_ms_of(&mut chain_us);
    // 3 decimals on the kernel timestamps: both kernels are µs-scale at small
    // scenes and a 2-decimal 0.00 would be indistinguishable from a dead query.
    let k1_p50 = if k1_gpu_us.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.3}", p50_ms_of(&mut k1_gpu_us))
    };
    let k2_p50 = if k2_gpu_us.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.3}", p50_ms_of(&mut k2_gpu_us))
    };
    // The GPU walk span (timestamp slot 2, build_far_units → emit_finalize):
    // inside gpu_span, attribution-only like k1/k2 — never summed.
    let walk_gpu_p50 = if walk_gpu_us.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.3}", p50_ms_of(&mut walk_gpu_us))
    };
    let stage_sum_ms = gpu_span_p50
        + pair_build_p50
        + assemble_p50
        + walk_p50
        + expand_p50
        + assign_p50
        + clear_p50
        + chain_p50;

    println!(
        "[scale_trial] instanced: {} buildings | library_atoms={} virtual_atoms={} | budget={} selected={} | select+expand+render p50={:.2} ms p99={:.2} ms @ {}x{}",
        buildings,
        library_atoms,
        virtual_atoms,
        last_budget,
        last_selected,
        p50_us as f64 / 1000.0,
        p99_us as f64 / 1000.0,
        INST_W,
        INST_H,
    );
    println!(
        "[scale_trial] instanced detail: select_ms p50={:.2} | gpu_fallbacks={} | entry_overflow max={} | entries_last={} entries/selected={:.2} entries_max={} | budget_settled={} ema_ms={:.2} | non_black={}/{} ({:.1}%) | png={}",
        sel_p50_us as f64 / 1000.0,
        gpu_sel.fallback_count(),
        overflow_max,
        last_entries,
        last_entries as f64 / last_selected.max(1) as f64,
        entries_max,
        governor.budget(),
        governor.ema_ms(),
        non_black,
        total_px,
        pct,
        png_path,
    );
    let p50_ms_headline = p50_us as f64 / 1000.0;
    println!(
        "[scale_trial] instanced stages p50 ms: select[gpu_span={:.2} k1_gpu={} k2_gpu={} walk_gpu={} pair_build={:.2} assemble={:.2} walk={:.2} syncs={}] expand={:.2} assign={:.2} clear={:.2} chain={:.2} | stage_sum={:.2} vs frame={:.2}",
        gpu_span_p50,
        k1_p50,
        k2_p50,
        walk_gpu_p50,
        pair_build_p50,
        assemble_p50,
        walk_p50,
        last_syncs,
        expand_p50,
        assign_p50,
        clear_p50,
        chain_p50,
        stage_sum_ms,
        p50_ms_headline,
    );
    // M3.2 Task 3 — the honest gate line, printed ALWAYS (pass and fail):
    // tier + adapter travel with every number so a relaxed run can never
    // masquerade as the target. 16.6 = 60fps-target (defined on the
    // tomespensin RTX 4070 Ti, verified there manually); 33.0 = 30fps-floor
    // (the 780M dev-box tracking floor); anything else prints "custom".
    let tier = if (gate_ms - 16.6).abs() < 1e-6 {
        "60fps-target"
    } else if (gate_ms - 33.0).abs() < 1e-6 {
        "30fps-floor"
    } else {
        "custom"
    };
    println!(
        "[scale_trial] gate: p50={:.2} ms vs gate={:.1} ms (tier={}) adapter={} -> {}",
        p50_ms_headline,
        gate_ms,
        tier,
        ctx.adapter_name(),
        if p50_ms_headline <= gate_ms {
            "PASS"
        } else {
            "FAIL"
        },
    );

    // --- THE GATE (M3) + the M1/M2 invariants. Milliseconds first: p50 must
    // clear `gate_ms` ALWAYS (default 16.6 — the 60 fps target gate; the
    // tier print above names every relaxation). Overflow must be 0 (Task 1's
    // grow-and-retry self-heals capacity);
    // GPU near-pair fallbacks must be 0 (the band must fit 262,144 units at
    // these scenes); coverage ≥ 25% is asserted at ≥ 10,000 buildings only
    // (design §2 defines the headline criterion at city scale — smaller
    // scenes print their real pct ungated). The M1/M2 budget checks are PER
    // MEASURED FRAME against that frame's governed budget: over-budget
    // always; the 0.9·B floor only in the ≥ 10× virtual regime, where the
    // selector must be able to spend it. Smaller scenes are scene-limited by
    // the distance-LOD fractions (the proven walk never promotes a cluster
    // ABOVE its distance LOD) and print their real (smaller) selected counts
    // without failing; black-frame stays enforced everywhere. First
    // offending frame reported, not 120 lines of spam.
    let mut failures: Vec<String> = Vec::new();
    let p50_ms = p50_us as f64 / 1000.0;
    if p50_ms > gate_ms {
        failures.push(format!(
            "p50 {:.2} ms > {gate_ms:.1} ms (tier={tier}; select_ms p50={:.2}, rest={:.2}; stages p50: gpu_span={:.2} walk_gpu={walk_gpu_p50} pair_build={:.2} assemble={:.2} walk={:.2} expand={:.2} assign={:.2} clear={:.2} chain={:.2})",
            p50_ms,
            sel_p50_us as f64 / 1000.0,
            p50_ms - sel_p50_us as f64 / 1000.0,
            gpu_span_p50,
            pair_build_p50,
            assemble_p50,
            walk_p50,
            expand_p50,
            assign_p50,
            clear_p50,
            chain_p50,
        ));
    }
    // M3.1 Task 1 sanity gate: the stage table must ACCOUNT for the frame —
    // a stage_sum drifting past 15% of the frame p50 means the instrument is
    // lying and every later optimization would be judged on bad data.
    if (stage_sum_ms - p50_ms).abs() > 0.15 * p50_ms {
        failures.push(format!(
            "stage_sum {stage_sum_ms:.2} ms not within 15% of frame p50 {p50_ms:.2} ms (per-stage accounting broken)"
        ));
    }
    if overflow_max != 0 {
        failures.push(format!(
            "entry_overflow max={overflow_max} (need 0 — grow-and-retry should self-heal)"
        ));
    }
    if gpu_sel.fallback_count() != 0 {
        failures.push(format!(
            "gpu fallbacks={} (need 0 — near-pair demand exceeded the 262144-unit band)",
            gpu_sel.fallback_count(),
        ));
    }
    if buildings >= 10_000 && pct < 25.0 {
        failures.push(format!(
            "coverage {pct:.1}% < 25.0% at {buildings} buildings (design §2 headline criterion; see {png_path})"
        ));
    }
    if let Some((i, &(b, s))) = frame_budgets
        .iter()
        .enumerate()
        .find(|&(_, &(b, s))| s > b)
    {
        failures.push(format!(
            "selection over budget: measured frame {i} selected {s} > budget_used {b}"
        ));
    }
    if let Some((i, &(b, s))) = frame_budgets
        .iter()
        .enumerate()
        .find(|&(_, &(b, s))| virtual_atoms >= 10 * b && s < (0.9 * b as f64) as usize)
    {
        failures.push(format!(
            "budget not spent: measured frame {i} selected {s} < 0.9*budget_used = {} (budget_used {b})",
            (0.9 * b as f64) as usize
        ));
    }
    if non_black == 0 {
        failures.push("final frame is black (non_black=0)".to_string());
    }
    if !failures.is_empty() {
        eprintln!("[scale_trial] instanced FAIL ({} issue(s)):", failures.len());
        for f in &failures {
            eprintln!("[scale_trial]   - {f}");
        }
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    // --- 0. `--instanced` measurement harness (M1/M2) -----------------------
    // Dispatches BEFORE the legacy 2M-splat scene generation and its
    // `< 2_000_000` failure gate. `--buildings <N>` takes a value (default
    // 10,000); the legacy flagless/--gpu-tiled/--gpu-resident paths are
    // untouched.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--instanced") {
        let buildings = args
            .iter()
            .position(|a| a == "--buildings")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10_000);
        // `--gate-ms <f32>` (M3.2 Task 3): the p50 gate threshold. Default
        // 16.6 = the 60 fps target (defined on the tomespensin RTX 4070 Ti);
        // the 780M acceptance runs pass `--gate-ms 33` (the 30 fps tracking
        // floor) EXPLICITLY — the default never weakens, and the printed
        // gate line names the tier + adapter on every run.
        let gate_ms = args
            .iter()
            .position(|a| a == "--gate-ms")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(16.6);
        // `--min-budget <usize>` (M3.2 Task 5): the governor's budget floor.
        // Default 100_000 — UNCHANGED; the flag exists only for the recorded
        // 50k-vs-100k quality/cost experiment. Any future default change is
        // the user's decision, made against the numbers recorded in the plan.
        let min_budget = args
            .iter()
            .position(|a| a == "--min-budget")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100_000);
        return run_instanced(buildings, gate_ms, min_budget);
    }

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

    // --- 5c. Optional RESIDENT GI->raster proof (`--gpu-resident`) -----------
    // AAA Spec 11: runs GI compute with its output bound DIRECTLY into the
    // rasterizer — ZERO CPU readback between GI and the raster (no per-frame GI
    // poll). Reuses the selected `subset` + `render_cam`; prints the Done-When
    // line and exits (self-contained renderer proof).
    if std::env::args().any(|a| a == "--gpu-resident") {
        return run_gpu_resident(&subset, &render_cam);
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
