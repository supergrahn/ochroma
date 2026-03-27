use glam::{Mat4, Vec3};
use vox_core::spectral::Illuminant;
use vox_render::gpu::software_rasteriser::{Framebuffer, SoftwareRasteriser};
use vox_render::spectral::RenderCamera;
use vox_terrain::volume::*;

fn save_ppm(fb: &Framebuffer, path: &std::path::Path) {
    let mut data = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    for pixel in &fb.pixels { data.push(pixel[0]); data.push(pixel[1]); data.push(pixel[2]); }
    std::fs::write(path, &data).unwrap();
}

#[test]
fn render_volumetric_terrain_with_cave_and_cliff() {
    let vol = generate_demo_volume(42);
    let materials = default_volume_materials();
    let splats = volume_to_splats(&vol, &materials, 42);

    println!("Volumetric terrain: {} surface splats from {} solid voxels",
        splats.len(), vol.solid_count());

    let mut rast = SoftwareRasteriser::new(512, 384);
    let cam = RenderCamera {
        view: Mat4::look_at_rh(
            Vec3::new(25.0, 15.0, 30.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::Y,
        ),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 512.0 / 384.0, 0.1, 200.0),
    };

    let fb = rast.render(&splats, &cam, &Illuminant::d65());

    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    let dir = std::env::temp_dir().join("ochroma_visual");
    std::fs::create_dir_all(&dir).unwrap();
    save_ppm(&fb, &dir.join("volumetric_terrain.ppm"));

    println!("Volumetric terrain render: {:.1}% coverage, saved to {}",
        coverage, dir.join("volumetric_terrain.ppm").display());

    assert!(splats.len() > 500, "Should have surface splats");
    assert!(coverage > 5.0, "Terrain should be visible: {:.1}%", coverage);
}
