use std::time::Instant;
use vox_core::types::GaussianSplat;
use vox_core::spectral::Illuminant;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::spectral::RenderCamera;
use glam::{Mat4, Vec3};
use half::f16;

fn make_splats(count: usize) -> Vec<GaussianSplat> {
    (0..count).map(|i| {
        let _t = i as f32 / count as f32;
        GaussianSplat {
            position: [(i % 100) as f32 * 0.5, (i / 100 % 100) as f32 * 0.5, (i / 10000) as f32 * 0.5],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [f16::from_f32(0.5).to_bits(); 8],
        }
    }).collect()
}

fn bench_render(splat_count: usize, width: u32, height: u32) -> (f32, f32) {
    let splats = make_splats(splat_count);
    let mut rast = SoftwareRasteriser::new(width, height);
    let cam = RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(25.0, 25.0, 50.0), Vec3::new(25.0, 25.0, 0.0), Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, width as f32 / height as f32, 0.1, 200.0),
    };

    let start = Instant::now();
    let fb = rast.render(&splats, &cam, &Illuminant::d65());
    let render_ms = start.elapsed().as_secs_f32() * 1000.0;

    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    (render_ms, coverage)
}

#[test]
fn benchmark_1k_splats() {
    let (ms, coverage) = bench_render(1_000, 256, 256);
    println!("[bench] 1k splats @ 256x256: {:.1}ms, {:.1}% coverage", ms, coverage);
    assert!(ms < 1000.0, "1k splats should render in <1s: {:.1}ms", ms);
}

#[test]
fn benchmark_10k_splats() {
    let (ms, coverage) = bench_render(10_000, 256, 256);
    println!("[bench] 10k splats @ 256x256: {:.1}ms, {:.1}% coverage", ms, coverage);
    assert!(ms < 5000.0, "10k splats should render in <5s: {:.1}ms", ms);
}

#[test]
fn benchmark_50k_splats() {
    let (ms, coverage) = bench_render(50_000, 256, 256);
    println!("[bench] 50k splats @ 256x256: {:.1}ms, {:.1}% coverage", ms, coverage);
    assert!(ms < 30000.0, "50k splats should render in <30s: {:.1}ms", ms);
}

#[test]
fn benchmark_low_resolution() {
    let (ms, _) = bench_render(10_000, 64, 64);
    println!("[bench] 10k splats @ 64x64: {:.1}ms", ms);
    // Lower res should be faster
    let (ms_high, _) = bench_render(10_000, 256, 256);
    // Allow some variance — low res is usually faster but not always due to OS scheduling
    assert!(ms < ms_high * 2.0, "Lower res should be roughly faster: {}ms vs {}ms", ms, ms_high);
}

#[test]
fn benchmark_frustum_culling() {
    use vox_render::frustum::Frustum;

    let splats = make_splats(100_000);
    let view = Mat4::look_at_rh(Vec3::new(25.0, 25.0, 50.0), Vec3::new(25.0, 25.0, 0.0), Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 200.0);
    let frustum = Frustum::from_view_proj(proj * view);

    let start = Instant::now();
    let visible: usize = splats.iter().filter(|s| {
        frustum.contains_sphere(Vec3::from(s.position), 0.5)
    }).count();
    let ms = start.elapsed().as_secs_f32() * 1000.0;

    println!("[bench] Frustum cull 100k splats: {:.1}ms, {} visible", ms, visible);
    assert!(ms < 100.0, "100k frustum tests should be <100ms: {:.1}ms", ms);
}

#[test]
fn benchmark_summary() {
    println!("=== OCHROMA PERFORMANCE BENCHMARKS ===");
    let tests = [(1_000, "1k"), (5_000, "5k"), (10_000, "10k"), (25_000, "25k")];
    for (count, label) in &tests {
        let (ms, coverage) = bench_render(*count, 256, 256);
        let fps = if ms > 0.0 { 1000.0 / ms } else { 0.0 };
        println!("  {} splats: {:.1}ms ({:.1} fps) — {:.1}% coverage", label, ms, fps, coverage);
    }
}
