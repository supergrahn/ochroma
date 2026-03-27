//! PLY Integration Test — verifies the full pipeline from PLY file generation
//! through loading to software rasterisation, producing a visible image.
//!
//! This test creates a 1000-splat PLY file with realistic 3DGS training output
//! values (SH coefficients, log-space scales, logit-space opacity, quaternion
//! rotations), loads it through the PLY loader, renders it with the software
//! rasteriser, and verifies the resulting image has visible content (>5% coverage).
//! The rendered image is saved to disk for visual inspection.

use std::io::{Cursor, Write};

use glam::{Mat4, Vec3};
use vox_core::spectral::Illuminant;
use vox_data::ply_loader::{load_ply, load_ply_from_reader};
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::spectral::RenderCamera;

/// Generate a 1000-splat PLY file that mimics real 3DGS training output.
///
/// The splats form a colourful torus shape with realistic parameter ranges:
/// - Positions: on a torus with R=4, r=1.5
/// - Scales: log-space in [-6, -3] (world-space 0.002 to 0.05)
/// - Rotations: normalised quaternions with varied orientations
/// - Opacity: logit-space in [1, 5] (sigmoid -> 0.73 to 0.99)
/// - SH DC: varied coefficients producing distinct R/G/B colours
fn generate_1000_splat_ply() -> Vec<u8> {
    let num_vertices: usize = 1000;
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
         property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
         property float opacity\n\
         property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
         end_header\n",
        num_vertices
    );

    let mut data = header.into_bytes();

    let big_r = 4.0f32; // torus major radius
    let small_r = 1.5f32; // torus minor radius

    for i in 0..num_vertices {
        let t = i as f32 / num_vertices as f32;

        // Torus parametric: theta goes around the ring, phi around the tube
        let theta = t * std::f32::consts::TAU * 3.0; // 3 loops
        let phi = t * std::f32::consts::TAU * 37.0; // many cross-section loops (prime for coverage)

        let x = (big_r + small_r * phi.cos()) * theta.cos();
        let y = small_r * phi.sin();
        let z = (big_r + small_r * phi.cos()) * theta.sin();

        // Scales in log-space: vary between -5.5 and -3.5
        let log_scale_base = -4.5 + (i as f32 * 0.037).sin() * 1.0;
        let log_sx = log_scale_base + (i as f32 * 0.13).cos() * 0.3;
        let log_sy = log_scale_base + (i as f32 * 0.17).sin() * 0.3;
        let log_sz = log_scale_base + (i as f32 * 0.23).cos() * 0.3;

        // Quaternion rotation: axis-angle derived from position on torus
        let angle = theta * 0.5 + phi * 0.3;
        let axis_x = theta.sin();
        let axis_y = phi.cos();
        let axis_z = (theta + phi).sin();
        let axis_len = (axis_x * axis_x + axis_y * axis_y + axis_z * axis_z).sqrt().max(1e-6);
        let half_angle = angle * 0.5;
        let qw = half_angle.cos();
        let qx = half_angle.sin() * axis_x / axis_len;
        let qy = half_angle.sin() * axis_y / axis_len;
        let qz = half_angle.sin() * axis_z / axis_len;

        // Opacity in logit-space: range [1.0, 5.0] -> sigmoid -> [0.73, 0.99]
        let logit_opacity = 3.0 + (i as f32 * 0.019).sin() * 2.0;

        // SH DC coefficients: create distinct colours based on position on the torus
        // Red channel peaks at theta=0, green at theta=2pi/3, blue at theta=4pi/3
        let r_dc = (theta).cos() * 1.5;
        let g_dc = (theta - std::f32::consts::TAU / 3.0).cos() * 1.5;
        let b_dc = (theta - 2.0 * std::f32::consts::TAU / 3.0).cos() * 1.5;

        let values: [f32; 14] = [
            x, y, z, log_sx, log_sy, log_sz, qw, qx, qy, qz, logit_opacity, r_dc, g_dc, b_dc,
        ];
        for val in &values {
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    data
}

#[test]
fn ply_integration_load_1000_splats() {
    let ply_data = generate_1000_splat_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply_data)).unwrap();
    assert_eq!(splats.len(), 1000, "Should load exactly 1000 splats");
}

#[test]
fn ply_integration_splats_have_realistic_values() {
    let ply_data = generate_1000_splat_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply_data)).unwrap();

    for (i, s) in splats.iter().enumerate() {
        // Positions should be on/near the torus (distance from origin between 2.5 and 5.5)
        let dist_xz = (s.position[0] * s.position[0] + s.position[2] * s.position[2]).sqrt();
        assert!(
            dist_xz < 7.0,
            "Splat {} has unreasonable XZ distance: {}",
            i,
            dist_xz
        );

        // Scales should be in reasonable range (exp(-5.5) ~ 0.004, exp(-3.5) ~ 0.03)
        for (axis, &scale) in s.scale.iter().enumerate() {
            assert!(
                scale > 0.001 && scale < 0.2,
                "Splat {} axis {} scale out of range: {}",
                i,
                axis,
                scale
            );
        }

        // Opacity should be > 0 (sigmoid of logit > 1.0 gives > 0.73 * 255 ~ 186)
        assert!(
            s.opacity > 100,
            "Splat {} opacity too low: {} (expected >100 from logit-space >1.0)",
            i,
            s.opacity
        );
    }
}

#[test]
fn ply_integration_render_has_visible_content() {
    let ply_data = generate_1000_splat_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply_data)).unwrap();

    // Camera looking at the torus from a good vantage point
    let eye = Vec3::new(10.0, 5.0, 10.0);
    let target = Vec3::ZERO;
    let cam = RenderCamera {
        view: Mat4::look_at_rh(eye, target, Vec3::Y),
        proj: Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            1.0, // square aspect
            0.1,
            100.0,
        ),
    };

    let mut rasteriser = SoftwareRasteriser::new(256, 256);
    let fb = rasteriser.render(&splats, &cam, &Illuminant::d65());

    // Count non-black pixels
    let non_black = fb
        .pixels
        .iter()
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count();
    let total = fb.pixels.len();
    let coverage = non_black as f64 / total as f64 * 100.0;

    println!(
        "[ply_integration] Rendered 1000-splat torus: {}/{} pixels lit ({:.1}% coverage)",
        non_black, total, coverage
    );

    assert!(
        coverage > 5.0,
        "Rendered image should have >5% coverage but got {:.1}%",
        coverage
    );

    // Save the image as PPM for visual inspection
    let dir = std::env::temp_dir().join("ochroma_test");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("ply_integration_1000_torus.ppm");

    let (w, h) = (256u32, 256u32);
    let mut ppm = format!("P6\n{} {}\n255\n", w, h).into_bytes();
    for p in &fb.pixels {
        ppm.push(p[0]);
        ppm.push(p[1]);
        ppm.push(p[2]);
    }
    std::fs::write(&path, &ppm).unwrap();
    println!("[ply_integration] Saved rendered image to: {}", path.display());
}

#[test]
fn ply_integration_splats_have_varied_colours() {
    let ply_data = generate_1000_splat_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply_data)).unwrap();

    // The SH DC coefficients vary with theta, so spectral bands should differ across splats
    let first_spectral = splats[0].spectral;
    let unique_count = splats
        .iter()
        .filter(|s| s.spectral != first_spectral)
        .count();

    assert!(
        unique_count > 900,
        "Expected >900 unique spectral signatures from 1000 splats, got {}",
        unique_count
    );
}

#[test]
fn ply_integration_render_from_multiple_angles() {
    let ply_data = generate_1000_splat_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply_data)).unwrap();

    let angles = [
        Vec3::new(10.0, 5.0, 0.0),   // side
        Vec3::new(0.0, 12.0, 0.1),    // top-down
        Vec3::new(-8.0, 2.0, -8.0),   // behind
    ];

    for (idx, &eye) in angles.iter().enumerate() {
        let cam = RenderCamera {
            view: Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0),
        };

        let mut rasteriser = SoftwareRasteriser::new(128, 128);
        let fb = rasteriser.render(&splats, &cam, &Illuminant::d65());

        let non_black = fb
            .pixels
            .iter()
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        let coverage = non_black as f64 / fb.pixels.len() as f64 * 100.0;

        println!(
            "[ply_integration] Angle {} ({:.0},{:.0},{:.0}): {:.1}% coverage",
            idx, eye.x, eye.y, eye.z, coverage
        );
        assert!(
            coverage > 2.0,
            "Angle {} should have >2% coverage but got {:.1}%",
            idx,
            coverage
        );
    }
}

/// Load a realistic 5000-splat PLY from a file path (not a reader).
/// Mimics a real 3DGS bicycle scene capture with hemisphere distribution,
/// varied SH DC colours, log-space scales, and logit-space opacity.
#[test]
fn load_ply_from_file_path() {
    // Create a realistic PLY file on disk
    let dir = std::env::temp_dir().join("ochroma_ply_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_scene.ply");

    // Generate 5000 splats mimicking real 3DGS output
    let num_splats: usize = 5000;
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
         property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
         property float opacity\n\
         property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
         end_header\n",
        num_splats
    );

    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(header.as_bytes()).unwrap();

    // Write realistic splat data — hemisphere distribution like a captured object
    for i in 0..num_splats {
        let t = i as f32 / num_splats as f32;

        // Hemisphere distribution (like a captured object)
        let phi = t * std::f32::consts::PI * 0.8; // hemisphere
        let theta = t * std::f32::consts::TAU * 7.0; // spiral
        let r = 2.0 + (i as f32 * 0.0031).sin() * 0.5; // varying radius

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.cos() + 1.0; // offset up
        let z = r * phi.sin() * theta.sin();

        // Realistic scales (log-space: -7 to -3, most around -5)
        let base_scale = -5.0 + (i as f32 * 0.017).sin() * 1.5;

        // Quaternion with some rotation variety
        let angle = i as f32 * 0.05;
        let qw = (angle * 0.5).cos();
        let qx = (angle * 0.5).sin() * 0.3;
        let qy = (angle * 0.3).cos() * 0.5;
        let qz = (angle * 0.7).sin() * 0.2;

        // Opacity (logit-space: -2 to 5, most around 2-3)
        let logit_opacity = 2.0 + (i as f32 * 0.011).sin() * 2.0;

        // SH DC with colour variation (not uniform)
        let hue = (i as f32 * 0.013) % 1.0;
        let r_dc = if hue < 0.33 { 1.0 } else { -0.5 } + (i as f32 * 0.007).sin() * 0.3;
        let g_dc = if hue >= 0.33 && hue < 0.66 { 1.0 } else { -0.5 } + (i as f32 * 0.011).cos() * 0.3;
        let b_dc = if hue >= 0.66 { 1.0 } else { -0.5 } + (i as f32 * 0.003).sin() * 0.3;

        for val in &[x, y, z, base_scale, base_scale * 0.8, base_scale * 1.2,
                     qw, qx, qy, qz, logit_opacity, r_dc, g_dc, b_dc] {
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    drop(file);

    // Load via file path
    let splats = load_ply(&path).unwrap();
    assert_eq!(splats.len(), 5000, "Should load all 5000 splats");

    // Verify realistic properties
    let avg_opacity: f32 = splats.iter().map(|s| s.opacity as f32).sum::<f32>() / splats.len() as f32;
    assert!(avg_opacity > 50.0 && avg_opacity < 240.0,
        "Average opacity should be realistic: {}", avg_opacity);

    let avg_scale: f32 = splats.iter().map(|s| s.scale[0]).sum::<f32>() / splats.len() as f32;
    assert!(avg_scale > 0.001 && avg_scale < 0.1,
        "Average scale should be small: {}", avg_scale);

    // Verify colour variation
    let first_spectral = splats[0].spectral;
    let has_variation = splats.iter().any(|s| s.spectral != first_spectral);
    assert!(has_variation, "Splats should have varied colours");

    // Render and check visibility
    let mut rast = SoftwareRasteriser::new(256, 256);
    let cam = RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(0.0, 2.0, 6.0), Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 50.0),
    };
    let fb = rast.render(&splats, &cam, &Illuminant::d65());
    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    println!("Realistic PLY: {} splats, avg opacity {:.0}, avg scale {:.4}, coverage {:.1}%",
        splats.len(), avg_opacity, avg_scale, coverage);

    assert!(coverage > 5.0, "Loaded PLY should render visibly: {:.1}%", coverage);

    // Save rendered image
    let dir2 = std::env::temp_dir().join("ochroma_visual");
    std::fs::create_dir_all(&dir2).unwrap();
    let img_path = dir2.join("ply_loaded_scene.ppm");
    let mut data = format!("P6\n256 256\n255\n").into_bytes();
    for p in &fb.pixels { data.push(p[0]); data.push(p[1]); data.push(p[2]); }
    std::fs::write(&img_path, &data).unwrap();
    println!("Saved: {}", img_path.display());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_real_aetherspectra_ply() {
    let path = std::path::Path::new("/home/tomespen/git/aetherspectra/output/character_hero.ply");
    if !path.exists() {
        println!("Skipping: real PLY file not found at {}", path.display());
        return;
    }

    let start = std::time::Instant::now();
    let splats = vox_data::ply_loader::load_ply(path).unwrap();
    let load_time = start.elapsed();

    println!("=== REAL PLY FILE LOADED ===");
    println!("File: {}", path.display());
    println!("Splats: {}", splats.len());
    println!("Load time: {:?}", load_time);
    println!("First splat: pos=({:.3},{:.3},{:.3}) opacity={} scale=({:.4},{:.4},{:.4})",
        splats[0].position[0], splats[0].position[1], splats[0].position[2],
        splats[0].opacity, splats[0].scale[0], splats[0].scale[1], splats[0].scale[2]);

    assert_eq!(splats.len(), 308078, "Should load all 308k splats");
    assert!(load_time.as_secs() < 30, "Should load in under 30 seconds");

    // Render it
    use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
    use vox_render::spectral::RenderCamera;
    use vox_core::spectral::Illuminant;
    use glam::{Mat4, Vec3};

    let mut rast = SoftwareRasteriser::new(512, 384);
    let cam = RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(0.0, 0.5, 2.0), Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 512.0/384.0, 0.01, 50.0),
    };

    let render_start = std::time::Instant::now();
    let fb = rast.render(&splats, &cam, &Illuminant::d65());
    let render_time = render_start.elapsed();

    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    println!("Render time: {:?}", render_time);
    println!("Coverage: {:.1}%", coverage);

    // Save image
    let dir = std::env::temp_dir().join("ochroma_visual");
    std::fs::create_dir_all(&dir).unwrap();
    let img_path = dir.join("real_character_hero.ppm");
    let mut data = format!("P6\n512 384\n255\n").into_bytes();
    for p in &fb.pixels { data.push(p[0]); data.push(p[1]); data.push(p[2]); }
    std::fs::write(&img_path, &data).unwrap();
    println!("Image saved: {}", img_path.display());

    assert!(coverage > 1.0, "Real PLY should render visibly: {:.1}%", coverage);
}
