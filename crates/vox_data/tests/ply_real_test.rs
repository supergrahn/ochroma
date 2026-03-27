use vox_data::ply_loader::*;
use std::io::Cursor;

/// Create a realistic PLY file mimicking actual 3DGS training output.
/// Includes proper SH DC coefficients (not just zeros) and realistic scales/opacities.
fn create_realistic_ply() -> Vec<u8> {
    let num_vertices = 500;
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nproperty float opacity\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nend_header\n",
        num_vertices
    );

    let mut data = header.into_bytes();

    // Generate a sphere of splats with realistic parameters
    for i in 0..num_vertices {
        let t = i as f32 / num_vertices as f32;
        let phi = t * std::f32::consts::PI;
        let theta = t * std::f32::consts::TAU * 5.0; // spiral

        // Position on a sphere of radius 3
        let x = 3.0 * phi.sin() * theta.cos();
        let y = 3.0 * phi.cos();
        let z = 3.0 * phi.sin() * theta.sin();

        // Scales in log-space (typical: -4 to -6)
        let log_scale = -5.0 + (i as f32 * 0.003).sin();

        // Quaternion: slight random rotation
        let angle = i as f32 * 0.1;
        let qw = angle.cos();
        let qx = angle.sin() * 0.5;
        let qy = angle.cos() * 0.3;
        let qz = angle.sin() * 0.1;

        // Opacity in logit-space (typical: -2 to 5)
        let logit_opacity = 2.0 + (i as f32 * 0.01).sin() * 2.0;

        // SH DC coefficients (typical: -1 to 1, maps to color via 0.5 + SH_C0 * val)
        let r_dc = (i as f32 * 0.02).sin();       // varying red
        let g_dc = (i as f32 * 0.03 + 1.0).sin(); // varying green
        let b_dc = (i as f32 * 0.01 + 2.0).cos(); // varying blue

        for val in &[x, y, z, log_scale, log_scale, log_scale, qw, qx, qy, qz, logit_opacity, r_dc, g_dc, b_dc] {
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    data
}

#[test]
fn load_realistic_ply_500_splats() {
    let ply = create_realistic_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply)).unwrap();
    assert_eq!(splats.len(), 500);
}

#[test]
fn realistic_ply_has_valid_positions() {
    let ply = create_realistic_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply)).unwrap();
    for s in &splats {
        // All positions should be on a sphere of radius ~3
        let dist = (s.position[0].powi(2) + s.position[1].powi(2) + s.position[2].powi(2)).sqrt();
        assert!(dist > 1.0 && dist < 5.0, "Position should be on sphere: dist={}", dist);
    }
}

#[test]
fn realistic_ply_has_valid_scales() {
    let ply = create_realistic_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply)).unwrap();
    for s in &splats {
        // Scales should be small (exp(-5) ~ 0.007)
        assert!(s.scale[0] > 0.001 && s.scale[0] < 0.1, "Scale should be small: {}", s.scale[0]);
    }
}

#[test]
fn realistic_ply_has_varied_colors() {
    let ply = create_realistic_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply)).unwrap();
    // Spectral bands should vary across splats (not all the same)
    let first = splats[0].spectral;
    let differs = splats.iter().any(|s| s.spectral != first);
    assert!(differs, "Realistic PLY should have varied colors");
}

#[test]
fn realistic_ply_renders_visible_sphere() {
    let ply = create_realistic_ply();
    let splats = load_ply_from_reader(&mut Cursor::new(&ply)).unwrap();

    use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
    use vox_render::spectral::RenderCamera;
    use vox_core::spectral::Illuminant;
    use glam::{Mat4, Vec3};

    let mut rast = SoftwareRasteriser::new(128, 128);
    let cam = RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 50.0),
    };

    let fb = rast.render(&splats, &cam, &Illuminant::d65());
    let non_black = fb.pixels.iter().filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    let coverage = non_black as f32 / fb.pixels.len() as f32 * 100.0;

    println!("Realistic PLY sphere: {} splats, {:.1}% coverage", splats.len(), coverage);
    assert!(coverage > 5.0, "Sphere should be visible: {:.1}%", coverage);
}
