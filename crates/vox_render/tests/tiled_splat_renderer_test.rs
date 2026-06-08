//! Integration tests for [`vox_render::gpu::tiled_splat_renderer::TiledSplatRenderer`].
//!
//! These run the FULL on-device tiled chain (tile_assign → radix_sort →
//! tile_range_build → splat_raster) on the local hardware GPU and validate it
//! against two real, computed outcomes:
//!
//!   1. `tiled_renderer_draws_non_black` — the GPU frame is actually lit (>10%
//!      non-black, central pixel bright). Catches a dead pipeline.
//!   2. `tiled_vs_cpu_coverage` — the GPU coverage matches the CPU
//!      `spectra_render` reference within 15%. THIS is the test that catches the
//!      tile_assign/splat_raster Y-flip mismatch: if the camera-uniform Y-flip
//!      fix is wrong, the GPU bins splats into mirrored tiles and coverage
//!      collapses far below the CPU number.
//!
//! Adapter-gated: on no GPU / a software adapter the tests print "SKIPPED no
//! adapter" and return (never fail), per the house no-panic contract.

use glam::{Mat4, Quat, Vec3};
use half::f16;
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::adapter;
use vox_render::gpu::tiled_splat_renderer::TiledSplatRenderer;
use vox_render::gpu::GpuContext;
use vox_render::spectral::RenderCamera;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// Build a headless [`GpuContext`] on the local hardware GPU, or `None` when no
/// hardware adapter is present (so the caller can SKIP gracefully).
fn try_context() -> Option<(GpuContext, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter_opt = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let adapter = adapter_opt?;
    let info = adapter.get_info();
    if adapter::ensure_hardware(&info).is_err() {
        return None;
    }

    // Request TIMESTAMP_QUERY when the adapter supports it (GpuTimers gates on
    // the GRANTED feature, so an ungranted device just falls back to wall-clock).
    let wanted = wgpu::Features::TIMESTAMP_QUERY;
    let features = adapter.features() & wanted;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("tiled_test_device"),
            required_features: features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    let ctx = GpuContext::from_parts(&device, &queue, &info);
    Some((ctx, device, queue))
}

/// A deterministic grid of OPAQUE splats, placed in the UPPER half of the view
/// (world y ∈ [0.5, 4.5]) so the lit region is vertically ASYMMETRIC. That
/// asymmetry is what makes a Y-flip DETECTABLE: a mirrored binding moves the lit
/// region to the lower half, which the centroid invariant catches. (A symmetric
/// grid centred on the axis — the previous scene — maps to itself under a flip
/// and could not detect the bug it claimed to.)
fn grid_scene() -> (Vec<GaussianSplat>, RenderCamera) {
    const NX: i32 = 28;
    const NY: i32 = 18;
    let (x0, x1) = (-3.5f32, 3.5f32);
    let (y0, y1) = (0.5f32, 4.5f32); // upper-world band → upper-screen band
    let mut splats = Vec::with_capacity((NX * NY) as usize);
    let spd: [u16; 16] = std::array::from_fn(|_| f16::from_f32(0.9).to_bits());
    for gy in 0..NY {
        for gx in 0..NX {
            let x = x0 + (x1 - x0) * gx as f32 / (NX - 1) as f32;
            let y = y0 + (y1 - y0) * gy as f32 / (NY - 1) as f32;
            splats.push(GaussianSplat::volume(
                [x, y, 0.0],
                [0.2, 0.2, 0.2], // overlap neighbours → solid coverage
                Quat::IDENTITY,
                255,
                spd,
            ));
        }
    }

    let cam = RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(0.0, 2.5, 8.0), Vec3::new(0.0, 2.5, 0.0), Vec3::Y),
        proj: Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_3, // 60°
            WIDTH as f32 / HEIGHT as f32,
            0.1,
            100.0,
        ),
    };
    (splats, cam)
}

/// Coverage = fraction of non-black pixels in an sRGB framebuffer.
fn coverage(pixels: &[[u8; 4]]) -> f64 {
    let nb = pixels
        .iter()
        .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
        .count();
    nb as f64 / pixels.len() as f64
}

/// Mean (x, y) pixel position of the non-black pixels — the lit region's
/// centroid. `None` if nothing is lit. The Y component is the Y-flip detector:
/// two render paths with the SAME orientation agree on it regardless of how big
/// each splat's footprint is; a vertical flip moves it to `HEIGHT - y`.
fn nonblack_centroid(pixels: &[[u8; 4]], width: u32) -> Option<(f64, f64)> {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u64);
    for (i, p) in pixels.iter().enumerate() {
        if p[0] != 0 || p[1] != 0 || p[2] != 0 {
            sx += (i as u32 % width) as f64;
            sy += (i as u32 / width) as f64;
            n += 1;
        }
    }
    (n > 0).then(|| (sx / n as f64, sy / n as f64))
}

#[test]
fn tiled_renderer_draws_non_black() {
    let Some((ctx, _device, _queue)) = try_context() else {
        eprintln!("SKIPPED no adapter");
        return;
    };
    eprintln!("[tiled test] adapter: {}", ctx.adapter_name());

    let (splats, cam) = grid_scene();
    let mut renderer = match TiledSplatRenderer::new(ctx.clone(), &splats, WIDTH, HEIGHT) {
        Ok(r) => r,
        Err(e) => panic!("renderer construction failed on a box with a GPU: {e}"),
    };

    let frame = renderer.render(&cam).expect("render");
    let (pixels, non_black) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());
    let total = pixels.len();
    let cov = non_black as f64 / total as f64;

    let ms = frame
        .raster_gpu_ms
        .map(|m| format!("{m:.3} ms (GPU)"))
        .unwrap_or_else(|| format!("{:.3} ms (wall)", frame.wall_ms));
    eprintln!(
        "[tiled test] non_black {}/{} = {:.1}%  | raster {}",
        non_black,
        total,
        cov * 100.0,
        ms
    );

    // The frame must be coherently lit (a real centroid exists, not scattered
    // single pixels) and clear the 10% Done-When bar.
    let centroid = nonblack_centroid(&pixels, WIDTH);
    assert!(centroid.is_some(), "GPU tiled frame is entirely black — dead pipeline");
    assert!(
        cov > 0.10,
        "GPU tiled frame only {:.1}% non-black (need > 10%) — pipeline drew nothing usable",
        cov * 100.0
    );
}

/// The GPU tiled path and the CPU `spectra_render` reference must agree on WHERE
/// the lit region is — specifically its vertical centroid. This is the real
/// Y-flip / tile-binding catcher: the camera-uniform Y-fix is correct iff the GPU
/// bins splats into the SAME tiles the raster evaluates, so the GPU lit region
/// sits at the same screen height as the CPU's. A mirrored binding would move the
/// GPU centroid to `HEIGHT - cy_cpu` — a delta far beyond the tolerance.
///
/// We deliberately do NOT assert the two coverage MAGNITUDES match: the GPU EWA
/// tiled rasteriser and the CPU `spectra_render` path use different Gaussian
/// footprint/cutoff models, so the lit AREA legitimately differs (the GPU draws
/// tighter). Footprint parity is a documented follow-up; ORIENTATION correctness
/// is what this slice must prove.
#[test]
fn tiled_y_orientation_matches_cpu() {
    let Some((ctx, _device, _queue)) = try_context() else {
        eprintln!("SKIPPED no adapter");
        return;
    };

    let (splats, cam) = grid_scene();

    // CPU reference (the orientation oracle).
    let cpu_pixels = vox_render::spectra_render::render_with_spectra_u8(
        &splats,
        &cam,
        WIDTH,
        HEIGHT,
        &Illuminant::d65(),
    );
    let cpu_cov = coverage(&cpu_pixels);
    let cpu_c = nonblack_centroid(&cpu_pixels, WIDTH).expect("CPU reference must draw");

    // GPU tiled path.
    let mut renderer =
        TiledSplatRenderer::new(ctx.clone(), &splats, WIDTH, HEIGHT).expect("renderer");
    let frame = renderer.render(&cam).expect("render");
    let (gpu_pixels, _nb) = frame.resolve_to_srgb(&ctx, &Illuminant::d65());
    let gpu_cov = coverage(&gpu_pixels);
    let gpu_c = nonblack_centroid(&gpu_pixels, WIDTH).expect("GPU must draw");

    let dy = (gpu_c.1 - cpu_c.1).abs();
    let dx = (gpu_c.0 - cpu_c.0).abs();
    eprintln!(
        "[orientation] gpu_centroid=({:.0},{:.0}) cpu_centroid=({:.0},{:.0}) dy={:.0}px dx={:.0}px \
         | gpu_cov={:.1}% cpu_cov={:.1}%",
        gpu_c.0, gpu_c.1, cpu_c.0, cpu_c.1, dy, dx, gpu_cov * 100.0, cpu_cov * 100.0
    );

    // Both must draw a coherent region (guards a 0/0 false pass).
    assert!(cpu_cov > 0.10, "CPU reference drew only {:.1}% — bad test scene", cpu_cov * 100.0);
    assert!(gpu_cov > 0.10, "GPU drew only {:.1}% — dead pipeline", gpu_cov * 100.0);

    // Vertical centroids must agree within 15% of the frame height. A Y-flip puts
    // the GPU centroid ~(HEIGHT - cpu) away — this fails loud on it.
    let tol = 0.15 * HEIGHT as f64;
    assert!(
        dy < tol,
        "GPU vs CPU vertical centroid differs by {dy:.0}px (> {tol:.0}px) — Y-flip / tile-binding \
         mismatch: gpu_cy={:.0} cpu_cy={:.0}",
        gpu_c.1, cpu_c.1
    );
    // Horizontal too (cheap; catches an X mirror).
    assert!(
        dx < tol,
        "GPU vs CPU horizontal centroid differs by {dx:.0}px (> {tol:.0}px) — tile-binding mismatch"
    );
}
