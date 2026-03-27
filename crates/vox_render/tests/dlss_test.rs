use vox_render::dlss::*;

#[test]
fn off_mode_no_upscale() {
    let mut pipeline = DlssPipeline::new(1920, 1080, DlssQuality::Off);
    assert_eq!(pipeline.render_resolution(), (1920, 1080));
    let pixels = vec![[128u8; 4]; 1920 * 1080];
    let result = pipeline.upscale(&pixels, 1920, 1080, &[], &[]);
    assert_eq!(result.len(), pixels.len());
}

#[test]
fn performance_mode_quarter_res() {
    let pipeline = DlssPipeline::new(3840, 2160, DlssQuality::Performance);
    let (w, h) = pipeline.render_resolution();
    assert!(w < 1920, "Performance should render below half: {}", w);
    assert!(h < 1080, "Performance should render below half: {}", h);
}

#[test]
fn upscale_increases_resolution() {
    let mut pipeline = DlssPipeline::new(1920, 1080, DlssQuality::Performance);
    let (rw, rh) = pipeline.render_resolution();
    let pixels = vec![[100u8; 4]; (rw * rh) as usize];
    let result = pipeline.upscale(&pixels, rw, rh, &[], &[]);
    assert_eq!(result.len(), 1920 * 1080);
}

#[test]
fn frame_generation_produces_intermediate() {
    let mut pipeline = DlssPipeline::new(64, 64, DlssQuality::Off);
    pipeline.frame_gen = FrameGeneration::On;

    let frame1 = vec![[0u8; 4]; 64 * 64];
    let frame2 = vec![[200u8, 200, 200, 255]; 64 * 64];
    let motion = vec![[0.0f32; 2]; 64 * 64];

    // First frame: no previous, no generation
    let gen1 = pipeline.generate_frame(&frame1, &motion);
    assert!(gen1.is_none());

    // Second frame: should generate intermediate
    let gen2 = pipeline.generate_frame(&frame2, &motion);
    assert!(gen2.is_some());
    let intermediate = gen2.unwrap();
    // Should be approximately halfway between frame1 (0) and frame2 (200)
    assert!(intermediate[0][0] > 50 && intermediate[0][0] < 150,
        "Intermediate should blend: got {}", intermediate[0][0]);
}

#[test]
fn fps_multiplier_scales_correctly() {
    let p_off = DlssPipeline::new(1920, 1080, DlssQuality::Off);
    let p_perf = DlssPipeline::new(1920, 1080, DlssQuality::Performance);
    assert!(p_perf.fps_multiplier() > p_off.fps_multiplier());

    let mut p_gen = DlssPipeline::new(1920, 1080, DlssQuality::Performance);
    p_gen.frame_gen = FrameGeneration::On;
    assert!(p_gen.fps_multiplier() > p_perf.fps_multiplier());
}
