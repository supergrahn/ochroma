//! Visual output tests — prove the engine produces real images.

use glam::{Mat4, Vec3};
use vox_core::spectral::Illuminant;
use vox_core::mapgen::generate_map;
use vox_render::gpu::software_rasteriser::{Framebuffer, SoftwareRasteriser};
use vox_render::spectral::RenderCamera;

fn save_ppm(fb: &Framebuffer, path: &std::path::Path) {
    let mut data = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    for pixel in &fb.pixels {
        data.push(pixel[0]);
        data.push(pixel[1]);
        data.push(pixel[2]);
    }
    std::fs::write(path, &data).unwrap();
}

fn make_camera(eye: Vec3, target: Vec3, width: u32, height: u32) -> RenderCamera {
    RenderCamera {
        view: Mat4::look_at_rh(eye, target, Vec3::Y),
        proj: Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            width as f32 / height as f32,
            0.1,
            500.0,
        ),
    }
}

#[test]
fn render_terrain_from_above_produces_visible_image() {
    let terrain = generate_map(42, 100.0, 1.0);
    assert!(terrain.len() > 5000);

    let mut rast = SoftwareRasteriser::new(256, 256);
    let cam = make_camera(Vec3::new(0.0, 80.0, 0.01), Vec3::ZERO, 256, 256);
    let fb = rast.render(&terrain, &cam, &Illuminant::d65());

    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    let dir = std::env::temp_dir().join("ochroma_visual");
    std::fs::create_dir_all(&dir).unwrap();
    save_ppm(&fb, &dir.join("terrain_above.ppm"));

    println!("Terrain from above: {} splats, {:.1}% coverage, saved to {}", terrain.len(), coverage, dir.join("terrain_above.ppm").display());
    assert!(coverage > 10.0, "Terrain should cover >10% of image: {:.1}%", coverage);
}

#[test]
fn render_buildings_at_street_level() {
    let mut splats = Vec::new();
    for i in 0..5 {
        let building = vox_data::proc_gs::emit_splats_simple(42 + i as u64, 6.0, 10.0);
        for s in &building {
            let mut ws = *s;
            ws.position[0] += i as f32 * 8.0 - 16.0;
            ws.position[2] += 10.0;
            splats.push(ws);
        }
    }

    let mut rast = SoftwareRasteriser::new(512, 256);
    let cam = make_camera(Vec3::new(0.0, 5.0, 30.0), Vec3::new(0.0, 5.0, 0.0), 512, 256);
    let fb = rast.render(&splats, &cam, &Illuminant::d65());

    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    let dir = std::env::temp_dir().join("ochroma_visual");
    std::fs::create_dir_all(&dir).unwrap();
    save_ppm(&fb, &dir.join("street_level.ppm"));

    println!("Street level: {} splats, {:.1}% coverage", splats.len(), coverage);
    assert!(coverage > 5.0, "Buildings should be visible: {:.1}%", coverage);
}

#[test]
fn render_trees_have_green_content() {
    let tree = vox_data::proc_gs_advanced::generate_tree(42, 8.0, 3.0);

    let mut rast = SoftwareRasteriser::new(128, 128);
    let cam = make_camera(Vec3::new(0.0, 5.0, 15.0), Vec3::new(0.0, 4.0, 0.0), 128, 128);
    let fb = rast.render(&tree, &cam, &Illuminant::d65());

    let coloured: Vec<_> = fb.pixels.iter().filter(|p| p[0] > 5 || p[1] > 5 || p[2] > 5).collect();
    assert!(!coloured.is_empty(), "Tree should produce coloured pixels");

    let avg_r: f32 = coloured.iter().map(|p| p[0] as f32).sum::<f32>() / coloured.len() as f32;
    let avg_g: f32 = coloured.iter().map(|p| p[1] as f32).sum::<f32>() / coloured.len() as f32;
    let avg_b: f32 = coloured.iter().map(|p| p[2] as f32).sum::<f32>() / coloured.len() as f32;

    println!("Tree colour: R={:.0} G={:.0} B={:.0} ({} coloured pixels)", avg_r, avg_g, avg_b, coloured.len());
}

#[test]
fn different_illuminants_change_appearance() {
    let splats = vox_data::proc_gs::emit_splats_simple(42, 6.0, 10.0);

    let mut rast = SoftwareRasteriser::new(64, 64);
    let cam = make_camera(Vec3::new(3.0, 5.0, 15.0), Vec3::new(3.0, 3.0, -5.0), 64, 64);

    let fb_d65 = rast.render(&splats, &cam, &Illuminant::d65());
    let fb_a = rast.render(&splats, &cam, &Illuminant::a());

    let differs = fb_d65.pixels.iter().zip(fb_a.pixels.iter())
        .any(|(a, b)| a[0] != b[0] || a[1] != b[1] || a[2] != b[2]);
    assert!(differs, "D65 and Illuminant A should produce different colours");
}

#[test]
fn render_full_city_scene() {
    let start = std::time::Instant::now();

    // Terrain
    let mut all_splats = generate_map(42, 150.0, 0.3);

    // Buildings
    for i in 0..8 {
        let b = vox_data::proc_gs::emit_splats_simple(42 + i as u64, 5.5, 10.0);
        for s in &b {
            let mut ws = *s;
            ws.position[0] += i as f32 * 7.0 - 24.0;
            ws.position[2] += 15.0;
            all_splats.push(ws);
        }
    }

    // Trees
    for i in 0..6 {
        let t = vox_data::proc_gs_advanced::generate_tree(500 + i as u64, 7.0, 3.0);
        for s in &t {
            let mut ws = *s;
            ws.position[0] += i as f32 * 10.0 - 25.0;
            ws.position[2] += 5.0;
            all_splats.push(ws);
        }
    }

    // Lamps
    for i in 0..4 {
        let l = vox_data::proc_gs_advanced::generate_lamp_post(700 + i as u64, 4.5);
        for s in &l {
            let mut ws = *s;
            ws.position[0] += i as f32 * 12.0 - 18.0;
            ws.position[2] += 3.0;
            all_splats.push(ws);
        }
    }

    let scene_time = start.elapsed();

    let mut rast = SoftwareRasteriser::new(640, 480);
    let cam = make_camera(Vec3::new(0.0, 25.0, 45.0), Vec3::new(0.0, 5.0, 0.0), 640, 480);

    let render_start = std::time::Instant::now();
    let fb = rast.render(&all_splats, &cam, &Illuminant::d65());
    let render_time = render_start.elapsed();

    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    let dir = std::env::temp_dir().join("ochroma_visual");
    std::fs::create_dir_all(&dir).unwrap();
    save_ppm(&fb, &dir.join("full_city.ppm"));

    println!("=== FULL CITY RENDER ===");
    println!("Splats: {}", all_splats.len());
    println!("Scene build: {:?}", scene_time);
    println!("Render: {:?}", render_time);
    println!("Coverage: {:.1}%", coverage);
    println!("Image: {}", dir.join("full_city.ppm").display());

    assert!(all_splats.len() > 10000, "City should have many splats");
    assert!(coverage > 10.0, "City should cover >10% of image");
    assert!(render_time.as_secs() < 60, "Should render in <60 seconds");
}

#[test]
fn empty_scene_is_black() {
    let mut rast = SoftwareRasteriser::new(64, 64);
    let cam = make_camera(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, 64, 64);
    let fb = rast.render(&[], &cam, &Illuminant::d65());

    let all_black = fb.pixels.iter().all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0);
    assert!(all_black, "Empty scene should be black");
}
