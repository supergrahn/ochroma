use vox_render::denoiser::SpectralDenoiser;

#[test]
fn zero_strength_no_change() {
    let mut pixels = vec![[100u8, 150, 200, 255]; 16];
    let original = pixels.clone();
    let denoiser = SpectralDenoiser::new(0.0);
    denoiser.denoise(&mut pixels, 4, 4);
    assert_eq!(pixels, original);
}

#[test]
fn solid_color_unchanged() {
    let mut pixels = vec![[128u8, 128, 128, 255]; 64];
    let denoiser = SpectralDenoiser::new(1.0);
    denoiser.denoise(&mut pixels, 8, 8);
    // Solid color should remain unchanged (bilateral filter preserves uniform regions)
    for p in &pixels {
        assert!((p[0] as i32 - 128).abs() <= 1);
    }
}

#[test]
fn noisy_image_gets_smoother() {
    // Create a checkerboard-like noisy pattern with small color differences
    // (bilateral filter preserves strong edges, so we use subtle noise)
    let mut pixels: Vec<[u8; 4]> = (0..64).map(|i| {
        if (i / 8 + i % 8) % 2 == 0 { [135, 135, 135, 255] } else { [120, 120, 120, 255] }
    }).collect();

    let before_variance = compute_variance(&pixels);
    let mut denoiser = SpectralDenoiser::new(0.8);
    denoiser.spectral_sigma = 0.5; // wider spectral tolerance to smooth similar colors
    denoiser.denoise(&mut pixels, 8, 8);
    let after_variance = compute_variance(&pixels);

    assert!(after_variance < before_variance, "Denoising should reduce variance");
}

fn compute_variance(pixels: &[[u8; 4]]) -> f32 {
    let mean: f32 = pixels.iter().map(|p| p[0] as f32).sum::<f32>() / pixels.len() as f32;
    pixels.iter().map(|p| (p[0] as f32 - mean).powi(2)).sum::<f32>() / pixels.len() as f32
}
