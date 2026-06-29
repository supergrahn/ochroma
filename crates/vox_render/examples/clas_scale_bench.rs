//! CLAS-scale perf spike — RTX Mega Geometry go/no-go at ~1M dynamic instances.
//!
//! Answers ONE question with REAL timings on an NVIDIA 4070 Ti:
//!
//!   "Does ~1M dynamic instances through the RTX Mega Geometry path
//!    (per-prototype CLAS → GAS-over-CLAS → IAS/TLAS) hold a real-time per-frame
//!    REFIT budget?"
//!
//! It builds ONE tiny prototype (a unit cube, 12 triangles) and N instances on a
//! deterministic grid (NO RNG, NO HashMap iteration — id-ordered), uploads it
//! through the LIVE game drive API ([`ResidentCityRenderer`] — the wrapper the
//! game uses to drive the Spectra path tracer), then for each frame perturbs
//! EVERY instance transform (the worst-case 1M-mover scenario) and times the GPU
//! refit and the render SEPARATELY.
//!
//! This is a PERF spike: correctness of the rendered image matters less than
//! getting honest build / refit / render numbers at scale. The cube is placed
//! mostly off-camera on a huge grid — pixels are not asserted.
//!
//! ## Build (NVIDIA box, Windows, CUDA in PATH)
//!
//! ```powershell
//! cargo build --release --example clas_scale_bench --features spectra-native
//! ```
//!
//! ## Run
//!
//! ```powershell
//! $env:SPECTRA_USE_OPTIX_RT = "1"      # arm the OptiX HW-RT traversal
//! $env:SPECTRA_CLAS_THRESHOLD = "1"    # force the CLAS path even for small N
//! $env:SPIKE_INSTANCES = "100000,250000,500000,1000000"   # default
//! $env:SPIKE_FRAMES = "60"             # frames per N (default 60)
//! cargo run --release --example clas_scale_bench --features spectra-native
//! ```
//!
//! `SPECTRA_CLAS_THRESHOLD` is also forced to `1` IN-PROCESS below (so the CLAS
//! path engages even if the env is unset), but setting it on the command line is
//! harmless and documents intent. `SPECTRA_USE_OPTIX_RT=1` is the reliable
//! activation switch on the `ResidentCityRenderer` path (see
//! `spectra-renderer/src/renderer/mod.rs:165`).
//!
//! ## Output (one line per N)
//!
//! ```text
//! N=1000000 build_ms=... refit_ms_avg=... render_ms_avg=... fps=... instances=... clusters=... vram_mb=...
//! ```
//!
//! `fps` is `1000 / (refit_ms_avg + render_ms_avg)` — the per-frame budget the
//! mover loop must fit. `instances`/`clusters` come from the device CLAS stats
//! (the witness the Mega-Geometry path actually engaged); they fall back to the
//! TLAS instance count + `0` clusters when CLAS stats are unavailable.

// Only compiles on the spectra-native (CUDA/OptiX) stack. Off-box this is a
// no-op `main` so `cargo build` does not choke on the missing CUDA toolchain.
#![cfg_attr(not(feature = "spectra-native"), allow(unused))]

#[cfg(not(feature = "spectra-native"))]
fn main() {
    eprintln!(
        "clas_scale_bench requires --features spectra-native (CUDA/OptiX). \
         Build on the NVIDIA box: cargo run --release --example clas_scale_bench \
         --features spectra-native"
    );
}

#[cfg(feature = "spectra-native")]
fn main() {
    // Internal render resolution — small so the bench is REFIT/BUILD-bound (the
    // thing under test), not pixel-shading-bound. Override with SPIKE_RES.
    let res: u32 = env_u32("SPIKE_RES", 1280);
    let width = res;
    let height = (res * 9 / 16).max(1);

    // Force the RTX Mega Geometry CLAS path even for the smallest N. Set
    // in-process so a forgotten env still exercises the path under test.
    // SAFETY: single-threaded, set before the renderer reads it (in render()).
    unsafe {
        std::env::set_var("SPECTRA_CLAS_THRESHOLD", "1");
    }

    let instance_counts =
        env_usize_list("SPIKE_INSTANCES", &[100_000, 250_000, 500_000, 1_000_000]);
    let frames: u32 = env_u32("SPIKE_FRAMES", 60);

    eprintln!(
        "[clas_scale_bench] res={width}x{height} frames={frames} sweep={instance_counts:?} \
         (SPECTRA_CLAS_THRESHOLD=1 forced; set SPECTRA_USE_OPTIX_RT=1 to arm OptiX)"
    );

    for &n in &instance_counts {
        match run_one(n, frames, width, height) {
            Ok(line) => println!("{line}"),
            // Keep sweeping the remaining N — a single OOM/build failure at the
            // top of the sweep should not hide the smaller-N numbers.
            Err(e) => println!("N={n} ERROR: {e}"),
        }
    }
}

/// Run the bench for a single instance count N. Returns the formatted line.
#[cfg(feature = "spectra-native")]
fn run_one(n: usize, frames: u32, width: u32, height: u32) -> Result<String, String> {
    use std::time::Instant;
    use vox_render::resident_renderer::ResidentCityRenderer;
    use vox_render::splat_backend::{LightRig, PbrMaterial};
    use vox_render::splat_convert::meshes_to_instanced_scene;

    // ONE prototype cube, ONE material. Every instance references proto 0.
    let cube = unit_cube_blas();
    let material = PbrMaterial {
        base_color: [0.6, 0.6, 0.62],
        roughness: 0.7,
        ..Default::default()
    };

    // --- deterministic instance grid (id-ordered, no RNG, no HashMap) ---
    let instances = grid_instances(n);

    // Build the SceneState (one proto, N instances). spd empty.
    let scene = meshes_to_instanced_scene(
        std::slice::from_ref(&cube),
        &instances,
        std::slice::from_ref(&material),
        &[],
        width,
        height,
    );
    debug_assert_eq!(scene.geometry.instance_count, n);

    // --- BUILD: construct the resident renderer (device init + kernel compile +
    // initial CLAS/GAS/IAS build). This is the one-time scale build we time.
    // spp=1, max_bounces=1 keep the render cheap so the loop is refit-bound. ---
    let build_t = Instant::now();
    let mut r = ResidentCityRenderer::new(width, height, LightRig::default(), 1, 1, scene)
        .map_err(|e| format!("renderer build: {e}"))?;
    let build_ms = build_t.elapsed().as_secs_f64() * 1000.0;

    let (view, proj) = camera(n, width, height);

    // Warm-up frame: forces lazy OptiX device + CLAS build + first dispatch
    // (kernel JIT, graph capture, allocations) so the timed loop is steady.
    r.render_only(view, proj)
        .map_err(|e| format!("warmup render: {e}"))?;

    // --- TIMED LOOP: every instance refit every frame, then render. ---
    let mut refit_total = 0.0f64;
    let mut render_total = 0.0f64;
    // Reused dirty buffer (all N instances), repopulated per frame.
    let mut dirty: Vec<(usize, [[f32; 4]; 3])> = Vec::with_capacity(n);
    for frame in 0..frames {
        dirty.clear();
        perturb_all(&mut dirty, n, frame);

        let refit_t = Instant::now();
        r.refit_instances(&dirty)
            .map_err(|e| format!("refit (frame {frame}): {e}"))?;
        refit_total += refit_t.elapsed().as_secs_f64() * 1000.0;

        let render_t = Instant::now();
        r.render_only(view, proj)
            .map_err(|e| format!("render (frame {frame}): {e}"))?;
        render_total += render_t.elapsed().as_secs_f64() * 1000.0;
    }

    let f = frames.max(1) as f64;
    let refit_ms_avg = refit_total / f;
    let render_ms_avg = render_total / f;
    let frame_ms = refit_ms_avg + render_ms_avg;
    let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

    // CLAS stats (the Mega-Geometry witness) or TLAS fallback.
    let (instances_built, clusters) = r.clas_stats().unwrap_or((r.tlas_instance_count(), 0));

    let vram = vram_used_mb()
        .map(|mb| format!(" vram_mb={mb}"))
        .unwrap_or_default();

    Ok(format!(
        "N={n} build_ms={build_ms:.1} refit_ms_avg={refit_ms_avg:.3} \
         render_ms_avg={render_ms_avg:.3} fps={fps:.1} \
         instances={instances_built} clusters={clusters}{vram}"
    ))
}

// ---------------------------------------------------------------------------
// Geometry / scene construction (deterministic, no RNG, no HashMap iteration).
// ---------------------------------------------------------------------------

/// One axis-aligned unit cube (12 triangles, 24 verts) centered at the object-
/// space origin — the single shared prototype. Verts/indices are fully
/// deterministic. Mirrors the proven cube in `tests/instanced_scene_test.rs`.
#[cfg(feature = "spectra-native")]
fn unit_cube_blas() -> vox_render::splat_backend::BlasDesc {
    use vox_render::splat_backend::BlasDesc;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();

    let h = 0.5f32; // unit cube, half-extent 0.5
    let corners = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 1, 2, 3], [0.0, 0.0, -1.0]),
        ([5, 4, 7, 6], [0.0, 0.0, 1.0]),
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0]),
        ([1, 5, 6, 2], [1.0, 0.0, 0.0]),
        ([3, 2, 6, 7], [0.0, 1.0, 0.0]),
        ([4, 5, 1, 0], [0.0, -1.0, 0.0]),
    ];
    for (quad, n) in faces {
        let base = positions.len() as u32;
        for &ci in quad.iter() {
            positions.push(corners[ci]);
            normals.push(n);
            uvs.push([0.0, 0.0]);
        }
        indices.push([base, base + 1, base + 2]);
        indices.push([base, base + 2, base + 3]);
        material_ids.push(0);
        material_ids.push(0);
    }

    BlasDesc {
        proto_id: 1,
        positions,
        normals,
        uvs,
        indices,
        material_ids,
        aabb_min: [-h, -h, -h],
        aabb_max: [h, h, h],
    }
}

/// N instances on a deterministic square-ish grid centered at the origin, all
/// referencing proto 0 with material base 0. Id-ordered (no RNG, no HashMap).
/// Row-major 4×4 with translation in the last ROW (indices 12/13/14 — the
/// `SceneState::instance_transforms` layout the uploader reads).
#[cfg(feature = "spectra-native")]
fn grid_instances(n: usize) -> Vec<vox_render::splat_backend::InstanceRecordGpu> {
    use vox_render::splat_backend::InstanceRecordGpu;
    let cols = (n as f64).sqrt().ceil() as usize;
    let cols = cols.max(1);
    let spacing = 2.0f32; // 2 units apart so unit cubes do not overlap
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let gx = (i % cols) as f32;
        let gz = (i / cols) as f32;
        // Center the grid about the origin.
        let x = (gx - cols as f32 * 0.5) * spacing;
        let z = (gz - (n as f32 / cols as f32) * 0.5) * spacing;
        out.push(InstanceRecordGpu {
            proto_index: 0,
            transform: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                x, 0.0, z, 1.0,
            ],
            material_base: 0,
        });
    }
    out
}

/// Perturb EVERY instance's transform deterministically by frame index — the
/// worst-case "all N instances are movers this frame" refit load. Translation
/// only (cheapest valid transform mutation; tests the TLAS refit, not the
/// transform math). `dirty` is filled id-ordered: `(instance_index, 3×4)`.
#[cfg(feature = "spectra-native")]
fn perturb_all(dirty: &mut Vec<(usize, [[f32; 4]; 3])>, n: usize, frame: u32) {
    let cols = (n as f64).sqrt().ceil() as usize;
    let cols = cols.max(1);
    let spacing = 2.0f32;
    // Small deterministic bob: amplitude scaled by frame so each frame is a
    // genuinely new transform (forces a real MODE_UPDATE, no caching shortcut).
    let bob = ((frame % 16) as f32) * 0.05 - 0.4;
    for i in 0..n {
        let gx = (i % cols) as f32;
        let gz = (i / cols) as f32;
        let x = (gx - cols as f32 * 0.5) * spacing;
        let z = (gz - (n as f32 / cols as f32) * 0.5) * spacing;
        // Row-major 3×4 (upper rows of the 4×4): rows are [r0; r1; r2], each a
        // [f32;4] whose 4th element is the translation component.
        dirty.push((
            i,
            [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, bob],
                [0.0, 0.0, 1.0, z],
            ],
        ));
    }
}

/// A camera that frames the whole grid from above-and-back so the build/refit
/// touches the full instance set. Returns column-major (view, proj) arrays —
/// the `render_camera`/`render_only` contract.
#[cfg(feature = "spectra-native")]
fn camera(n: usize, width: u32, height: u32) -> ([f32; 16], [f32; 16]) {
    let cols = (n as f64).sqrt().ceil() as f32;
    let extent = cols.max(1.0) * 2.0; // grid spacing == 2.0
    let dist = extent * 1.2 + 5.0;
    let eye = glam::Vec3::new(0.0, dist * 0.6, dist);
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y).to_cols_array();
    let aspect = width as f32 / height.max(1) as f32;
    let proj = glam::Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, dist * 4.0 + 100.0)
        .to_cols_array();
    (view, proj)
}

// ---------------------------------------------------------------------------
// Env / VRAM helpers
// ---------------------------------------------------------------------------

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "spectra-native")]
fn env_usize_list(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(s) => {
            let parsed: Vec<usize> = s
                .split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .collect();
            if parsed.is_empty() {
                default.to_vec()
            } else {
                parsed
            }
        }
        Err(_) => default.to_vec(),
    }
}

/// Best-effort device VRAM-used in MB via `nvidia-smi`. Returns `None` when the
/// tool is absent (the line then omits `vram_mb=`). Not load-bearing for the
/// go/no-go; the timing numbers are the verdict.
#[cfg(feature = "spectra-native")]
fn vram_used_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse::<u64>().ok()
}
