use vox_core::types::GaussianSplat;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::spectral::RenderCamera;
use vox_core::spectral::Illuminant;
use glam::{Vec3, Mat4};
use std::time::Instant;

fn make_splat(x: f32, y: f32, z: f32) -> GaussianSplat {
    GaussianSplat {
        position: [x, y, z],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 32767],
        opacity: 200,
        _pad: [0; 3],
        spectral: [15360; 8],
    }
}

#[test]
fn render_1000_splats_under_100ms() {
    let splats: Vec<GaussianSplat> = (0..1000)
        .map(|i| make_splat((i % 50) as f32, 0.0, (i / 50) as f32))
        .collect();

    let mut rasteriser = SoftwareRasteriser::new(256, 256);
    let camera = RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(25.0, 20.0, 25.0), Vec3::new(25.0, 0.0, 10.0), Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 200.0),
    };

    let start = Instant::now();
    let _fb = rasteriser.render(&splats, &camera, &Illuminant::d65());
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 100, "1000 splats should render in <100ms, took {}ms", elapsed.as_millis());
}

#[test]
fn frustum_cull_10000_instances_under_10ms() {
    use vox_render::frustum::Frustum;

    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 1000.0);
    let frustum = Frustum::from_view_proj(proj * view);

    let positions: Vec<Vec3> = (0..10000)
        .map(|i| Vec3::new((i % 100) as f32 * 10.0 - 500.0, 0.0, -((i / 100) as f32 * 10.0)))
        .collect();

    let start = Instant::now();
    let visible_count: usize = positions.iter()
        .filter(|p| frustum.contains_sphere(**p, 5.0))
        .count();
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 10, "10k frustum tests should complete in <10ms, took {}ms", elapsed.as_millis());
    assert!(visible_count > 0 && visible_count < 10000, "Some should be visible, some culled");
}

#[test]
fn citizen_simulation_100k_under_100ms() {
    use vox_sim::citizen::CitizenManager;

    let mut mgr = CitizenManager::new();
    for i in 0..1000 {
        mgr.spawn(20.0 + (i % 50) as f32, None);
    }

    let start = Instant::now();
    for _ in 0..100 {
        mgr.tick(0.001); // 100 ticks
    }
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 100, "1000 citizens x 100 ticks should complete in <100ms, took {}ms", elapsed.as_millis());
}
