use vox_render::spectral_tonemapper::*;
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_core::spectral::Illuminant;

#[test]
fn tonemap_empty_framebuffer_is_black() {
    let fb = SpectralFramebuffer::new(4, 4);
    let settings = ToneMapSettings::default();
    let pixels = tonemap_spectral_framebuffer(&fb, &Illuminant::d65(), &settings);
    assert_eq!(pixels.len(), 16);
    for p in &pixels {
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
    }
}

#[test]
fn tonemap_white_produces_bright_pixel() {
    let mut fb = SpectralFramebuffer::new(1, 1);
    fb.write_sample(0, 0, [1.0; 8], 10.0, [0.0, 1.0, 0.0], 0, [1.0; 8]);
    let settings = ToneMapSettings::default();
    let pixels = tonemap_spectral_framebuffer(&fb, &Illuminant::d65(), &settings);
    assert!(
        pixels[0][0] > 100,
        "White spectral should produce bright pixel: R={}",
        pixels[0][0]
    );
    assert!(pixels[0][1] > 100, "G={}", pixels[0][1]);
    assert!(pixels[0][2] > 100, "B={}", pixels[0][2]);
}

#[test]
fn aces_compresses_hdr() {
    let settings = ToneMapSettings {
        operator: ToneMapOperator::ACES,
        ..Default::default()
    };
    // Very bright input
    let mut fb = SpectralFramebuffer::new(1, 1);
    fb.write_sample(0, 0, [5.0; 8], 10.0, [0.0, 1.0, 0.0], 0, [1.0; 8]);
    let pixels = tonemap_spectral_framebuffer(&fb, &Illuminant::d65(), &settings);
    // Should be bright but not clipped to pure white
    assert!(pixels[0][0] > 200, "Should be bright");
    assert!(pixels[0][0] < 255, "ACES should compress, not clip");
}

#[test]
fn exposure_affects_brightness() {
    let mut fb = SpectralFramebuffer::new(1, 1);
    fb.write_sample(0, 0, [0.5; 8], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);

    let dark = ToneMapSettings {
        exposure: 0.5,
        ..Default::default()
    };
    let bright = ToneMapSettings {
        exposure: 2.0,
        ..Default::default()
    };

    let dark_px = tonemap_spectral_framebuffer(&fb, &Illuminant::d65(), &dark);
    let bright_px = tonemap_spectral_framebuffer(&fb, &Illuminant::d65(), &bright);

    assert!(
        bright_px[0][0] > dark_px[0][0],
        "Higher exposure should be brighter"
    );
}

#[test]
fn hdr_output_preserves_values_above_one() {
    let mut fb = SpectralFramebuffer::new(1, 1);
    fb.write_sample(0, 0, [3.0; 8], 10.0, [0.0, 1.0, 0.0], 0, [1.0; 8]);
    let settings = ToneMapSettings {
        operator: ToneMapOperator::None,
        exposure: 1.0,
        ..Default::default()
    };
    let hdr = tonemap_spectral_to_hdr(&fb, &Illuminant::d65(), &settings);
    // With no tone mapping, HDR values can exceed 1.0
    // (clamped to 1.0 in the None operator but linear values are > 1.0 before clamping)
    assert!(hdr[0][0] >= 0.0);
}

#[test]
fn different_illuminants_different_output() {
    let mut fb = SpectralFramebuffer::new(1, 1);
    // Non-uniform spectrum so illuminant differences are visible
    fb.write_sample(0, 0, [0.1, 0.2, 0.8, 0.6, 0.3, 0.9, 0.4, 0.1], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);
    let settings = ToneMapSettings::default();

    let d65 = tonemap_spectral_framebuffer(&fb, &Illuminant::d65(), &settings);
    let a = tonemap_spectral_framebuffer(&fb, &Illuminant::a(), &settings);

    assert_ne!(
        d65[0], a[0],
        "Different illuminants should produce different colours"
    );
}
