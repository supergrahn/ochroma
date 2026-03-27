use vox_render::temporal::TemporalAccumulator;
use vox_render::spectral_framebuffer::SpectralFramebuffer;

#[test]
fn first_frame_equals_current() {
    let mut accum = TemporalAccumulator::new(4, 4);
    let mut fb = SpectralFramebuffer::new(4, 4);
    fb.write_sample(0, 0, [0.8; 8], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);
    accum.accumulate(&fb);
    let result = accum.get(0, 0);
    assert!(
        (result[0] - 0.8).abs() < 0.01,
        "First frame should equal current"
    );
}

#[test]
fn accumulation_converges() {
    let mut accum = TemporalAccumulator::new(4, 4);

    // Accumulate the same frame 10 times
    for _ in 0..10 {
        let mut fb = SpectralFramebuffer::new(4, 4);
        fb.write_sample(0, 0, [0.5; 8], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);
        accum.accumulate(&fb);
    }

    let result = accum.get(0, 0);
    // Should converge toward 0.5
    assert!(
        (result[0] - 0.5).abs() < 0.1,
        "Should converge: got {}",
        result[0]
    );
}

#[test]
fn noisy_samples_get_smoothed() {
    let mut accum = TemporalAccumulator::new(4, 4);

    // Alternate between bright and dark (simulating noise)
    for i in 0..20 {
        let mut fb = SpectralFramebuffer::new(4, 4);
        let val = if i % 2 == 0 { 0.8 } else { 0.2 };
        fb.write_sample(0, 0, [val; 8], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);
        accum.accumulate(&fb);
    }

    let result = accum.get(0, 0);
    // Should be near the average (0.5)
    assert!(
        (result[0] - 0.5).abs() < 0.2,
        "Noisy samples should converge to average: got {}",
        result[0]
    );
}

#[test]
fn reset_clears_history() {
    let mut accum = TemporalAccumulator::new(4, 4);
    let mut fb = SpectralFramebuffer::new(4, 4);
    fb.write_sample(0, 0, [1.0; 8], 10.0, [0.0, 1.0, 0.0], 0, [1.0; 8]);
    accum.accumulate(&fb);
    assert!(accum.get(0, 0)[0] > 0.5);

    accum.reset();
    assert_eq!(accum.get(0, 0), [0.0; 8]);
    assert_eq!(accum.avg_accumulated_frames(), 0.0);
}

#[test]
fn avg_frames_tracks_accumulation() {
    let mut accum = TemporalAccumulator::new(2, 2);
    assert_eq!(accum.avg_accumulated_frames(), 0.0);

    let mut fb = SpectralFramebuffer::new(2, 2);
    fb.write_sample(0, 0, [0.5; 8], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);
    fb.write_sample(1, 0, [0.5; 8], 10.0, [0.0, 1.0, 0.0], 0, [0.5; 8]);
    accum.accumulate(&fb);

    assert!(accum.avg_accumulated_frames() > 0.0);
}
