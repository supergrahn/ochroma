use vox_render::spectral_framebuffer::SpectralFramebuffer;

#[test]
fn new_framebuffer_is_clear() {
    let fb = SpectralFramebuffer::new(64, 64);
    assert_eq!(fb.pixel_count(), 4096);
    assert_eq!(fb.spectral[0], [0.0; 8]);
    assert_eq!(fb.depth[0], f32::MAX);
    assert_eq!(fb.sample_count[0], 0);
}

#[test]
fn write_single_sample() {
    let mut fb = SpectralFramebuffer::new(4, 4);
    fb.write_sample(1, 1, [0.5; 8], 10.0, [0.0, 1.0, 0.0], 42, [0.3; 8]);
    let i = fb.idx(1, 1);
    assert_eq!(fb.spectral[i], [0.5; 8]);
    assert_eq!(fb.depth[i], 10.0);
    assert_eq!(fb.object_id[i], 42);
    assert_eq!(fb.sample_count[i], 1);
}

#[test]
fn accumulate_multiple_samples() {
    let mut fb = SpectralFramebuffer::new(4, 4);
    fb.write_sample(0, 0, [1.0; 8], 10.0, [0.0, 1.0, 0.0], 1, [0.5; 8]);
    fb.write_sample(0, 0, [0.0; 8], 5.0, [0.0, 1.0, 0.0], 1, [0.5; 8]);
    let i = fb.idx(0, 0);
    // Average of 1.0 and 0.0 = 0.5
    assert!((fb.spectral[i][0] - 0.5).abs() < 0.01);
    // Depth should keep nearest (5.0)
    assert!((fb.depth[i] - 5.0).abs() < 0.01);
    assert_eq!(fb.sample_count[i], 2);
}

#[test]
fn clear_resets_all() {
    let mut fb = SpectralFramebuffer::new(4, 4);
    fb.write_sample(0, 0, [1.0; 8], 10.0, [0.0, 1.0, 0.0], 1, [0.5; 8]);
    fb.clear();
    assert_eq!(fb.spectral[0], [0.0; 8]);
    assert_eq!(fb.sample_count[0], 0);
}

#[test]
fn memory_calculation() {
    let fb = SpectralFramebuffer::new(1920, 1080);
    let mb = fb.memory_bytes() as f32 / (1024.0 * 1024.0);
    println!("1080p spectral framebuffer: {:.1} MB", mb);
    assert!(mb > 50.0 && mb < 200.0, "Should be 50-200MB at 1080p");
}

#[test]
fn motion_vectors() {
    let mut fb = SpectralFramebuffer::new(4, 4);
    fb.write_motion(2, 2, [1.5, -0.5]);
    let i = fb.idx(2, 2);
    assert_eq!(fb.motion[i], [1.5, -0.5]);
}
