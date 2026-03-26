use vox_core::types::GaussianSplat;
use vox_core::spectral::Illuminant;
use vox_render::spectral::RenderCamera;
use vox_render::gpu::software_rasteriser::{SoftwareRasteriser, Framebuffer};
use glam::{Vec3, Mat4};

fn make_test_splat(pos: [f32; 3]) -> GaussianSplat {
    GaussianSplat {
        position: pos,
        scale: [0.5, 0.5, 0.5],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        spectral: [15360; 8], // f16 1.0 on all bands = white
        _pad: [0; 3],
    }
}

#[test]
fn framebuffer_starts_black() {
    let fb = Framebuffer::new(64, 64);
    assert_eq!(fb.width, 64);
    assert_eq!(fb.height, 64);
    assert!(fb.pixels.iter().all(|p| *p == [0u8; 4]));
}

#[test]
fn single_splat_renders_nonblack_pixel() {
    let mut rasteriser = SoftwareRasteriser::new(64, 64);
    let splats = vec![make_test_splat([0.0, 0.0, -5.0])];
    let camera = RenderCamera {
        view: Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
    };
    let fb = rasteriser.render(&splats, &camera, &Illuminant::d65());
    let has_colour = fb.pixels.iter().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    assert!(has_colour, "Expected at least one non-black pixel");
}

#[test]
fn two_splats_at_different_positions_both_render() {
    let mut rasteriser = SoftwareRasteriser::new(128, 128);
    let splats = vec![
        make_test_splat([-2.0, 0.0, -5.0]),
        make_test_splat([2.0, 0.0, -5.0]),
    ];
    let camera = RenderCamera {
        view: Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
    };
    let fb = rasteriser.render(&splats, &camera, &Illuminant::d65());
    let left_has_colour = fb.pixels.iter().enumerate()
        .any(|(i, p)| (i % 128) < 64 && (p[0] > 0 || p[1] > 0 || p[2] > 0));
    let right_has_colour = fb.pixels.iter().enumerate()
        .any(|(i, p)| (i % 128) >= 64 && (p[0] > 0 || p[1] > 0 || p[2] > 0));
    assert!(left_has_colour, "Left splat should produce pixels");
    assert!(right_has_colour, "Right splat should produce pixels");
}
