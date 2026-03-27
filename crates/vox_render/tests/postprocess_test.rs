use vox_render::postprocess::*;

#[test]
fn aces_compresses_hdr() {
    let mut pixels = vec![[1.0f32, 1.0, 1.0, 1.0]];
    apply_tone_mapping(&mut pixels, ToneMapping::ACES);
    // ACES(1.0) = (2.51 + 0.03) / (2.43 + 0.59 + 0.14) = 2.54 / 3.16 ≈ 0.8038
    let val = pixels[0][0];
    assert!(
        (val - 0.8038).abs() < 0.01,
        "ACES(1.0) = {} (expected ~0.8038)",
        val
    );
    // Alpha should be untouched.
    assert!((pixels[0][3] - 1.0).abs() < 1e-5);
}

#[test]
fn aces_zero_stays_near_zero() {
    let mut pixels = vec![[0.0f32, 0.0, 0.0, 1.0]];
    apply_tone_mapping(&mut pixels, ToneMapping::ACES);
    // ACES(0) = 0/0.14 = 0
    assert!(pixels[0][0].abs() < 1e-5);
}

#[test]
fn reinhard_compresses() {
    let mut pixels = vec![[1.0f32, 1.0, 1.0, 1.0]];
    apply_tone_mapping(&mut pixels, ToneMapping::Reinhard);
    // Reinhard(1.0) = 1/(1+1) = 0.5
    assert!((pixels[0][0] - 0.5).abs() < 1e-5);
}

#[test]
fn no_tone_mapping_is_identity() {
    let mut pixels = vec![[2.0f32, 3.0, 4.0, 1.0]];
    apply_tone_mapping(&mut pixels, ToneMapping::None);
    assert!((pixels[0][0] - 2.0).abs() < 1e-5);
    assert!((pixels[0][1] - 3.0).abs() < 1e-5);
}

#[test]
fn bloom_increases_brightness_near_bright_source() {
    let w = 5;
    let h = 5;
    let mut pixels = vec![[0.1f32, 0.1, 0.1, 1.0]; w * h];
    // Place a very bright pixel in the center.
    pixels[2 * w + 2] = [10.0, 10.0, 10.0, 1.0];

    let original_neighbor = pixels[2 * w + 3];
    apply_bloom(&mut pixels, w, h, 0.5, 1.0);

    // Neighbor should be brighter than before.
    assert!(
        pixels[2 * w + 3][0] > original_neighbor[0],
        "neighbor R: {} should be > {}",
        pixels[2 * w + 3][0],
        original_neighbor[0]
    );
}

#[test]
fn vignette_darkens_corners_more_than_center() {
    let w = 10;
    let h = 10;
    let mut pixels = vec![[1.0f32, 1.0, 1.0, 1.0]; w * h];

    apply_vignette(&mut pixels, w, h, 1.0);

    let center = pixels[5 * w + 5][0];
    let corner = pixels[0][0];
    assert!(
        corner < center,
        "corner {} should be darker than center {}",
        corner,
        center
    );
}

#[test]
fn pipeline_applies_all_effects() {
    let w = 4;
    let h = 4;
    let mut pixels = vec![[1.0f32, 1.0, 1.0, 1.0]; w * h];

    let pipeline = PostProcessPipeline {
        tone_mapping: ToneMapping::ACES,
        bloom_enabled: false,
        bloom_threshold: 1.0,
        bloom_intensity: 0.3,
        vignette_enabled: true,
        vignette_strength: 0.5,
    };

    pipeline.apply(&mut pixels, w, h);

    // After ACES, 1.0 -> ~0.80; then vignette darkens further.
    let corner = pixels[0][0];
    assert!(corner < 0.8, "corner after ACES+vignette: {}", corner);
}
