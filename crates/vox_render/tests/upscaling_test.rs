use vox_render::upscaling::*;

#[test]
fn native_no_scaling() {
    assert_eq!(UpscaleQuality::Native.scale_factor(), 1.0);
    assert_eq!(UpscaleQuality::Native.internal_resolution(1920, 1080), (1920, 1080));
}

#[test]
fn performance_halves_resolution() {
    let (w, h) = UpscaleQuality::Performance.internal_resolution(1920, 1080);
    assert_eq!(w, 960);
    assert_eq!(h, 540);
}

#[test]
fn performance_multiplier_increases() {
    assert!(UpscaleQuality::Performance.performance_multiplier() > UpscaleQuality::Quality.performance_multiplier());
    assert!(UpscaleQuality::UltraPerformance.performance_multiplier() > UpscaleQuality::Performance.performance_multiplier());
}

#[test]
fn bilinear_upscale_preserves_solid_color() {
    let src = vec![[128u8, 64, 32, 255]; 4]; // 2x2 solid color
    let dst = bilinear_upscale(&src, 2, 2, 4, 4);
    assert_eq!(dst.len(), 16);
    for pixel in &dst {
        assert!((pixel[0] as i32 - 128).abs() <= 1);
        assert!((pixel[1] as i32 - 64).abs() <= 1);
    }
}

#[test]
fn upscale_manager_native_passthrough() {
    let mgr = UpscaleManager::new(1920, 1080, UpscaleQuality::Native);
    let pixels = vec![[255u8, 0, 0, 255]; 1920 * 1080];
    let result = mgr.upscale(&pixels, 1920, 1080);
    assert_eq!(result.len(), pixels.len());
}

#[test]
fn upscale_manager_performance_mode() {
    let mgr = UpscaleManager::new(1920, 1080, UpscaleQuality::Performance);
    let (rw, rh) = mgr.render_resolution();
    assert_eq!((rw, rh), (960, 540));

    let pixels = vec![[100u8, 200, 50, 255]; (rw * rh) as usize];
    let result = mgr.upscale(&pixels, rw, rh);
    assert_eq!(result.len(), 1920 * 1080);
}
