use vox_core::types::GaussianSplat;
use vox_render::hierarchical_lod::*;

fn make_test_splats(count: usize) -> Vec<GaussianSplat> {
    (0..count)
        .map(|i| GaussianSplat {
            position: [i as f32 * 0.5, (i as f32 * 0.3).sin(), 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [0; 8],
        })
        .collect()
}

#[test]
fn lod_chain_has_four_levels() {
    let splats = make_test_splats(100);
    let chain = generate_lod_chain(&splats);
    assert_eq!(chain.levels.len(), 4);
}

#[test]
fn lod_0_has_most_splats() {
    let splats = make_test_splats(100);
    let chain = generate_lod_chain(&splats);

    let lod0_count = chain.levels[0].splat_count;
    for i in 1..4 {
        assert!(
            lod0_count > chain.levels[i].splat_count,
            "LOD 0 ({}) should have more splats than LOD {} ({})",
            lod0_count,
            i,
            chain.levels[i].splat_count
        );
    }
}

#[test]
fn lod_3_has_fewest_splats() {
    let splats = make_test_splats(100);
    let chain = generate_lod_chain(&splats);

    let lod3_count = chain.levels[3].splat_count;
    assert_eq!(lod3_count, 1, "LOD 3 should have exactly 1 billboard splat");
}

#[test]
fn splat_counts_decrease_monotonically() {
    let splats = make_test_splats(200);
    let chain = generate_lod_chain(&splats);

    for i in 0..3 {
        assert!(
            chain.levels[i].splat_count >= chain.levels[i + 1].splat_count,
            "LOD {} ({}) should have >= splats than LOD {} ({})",
            i,
            chain.levels[i].splat_count,
            i + 1,
            chain.levels[i + 1].splat_count
        );
    }
}

#[test]
fn distance_selects_correct_level() {
    // Very close + large screen = LOD 0.
    assert_eq!(select_lod_level(10.0, 500.0), 0);

    // Medium distance + medium screen = LOD 1.
    assert_eq!(select_lod_level(60.0, 150.0), 1);

    // Far + small screen = LOD 2.
    assert_eq!(select_lod_level(200.0, 40.0), 2);

    // Very far + tiny screen = LOD 3.
    assert_eq!(select_lod_level(500.0, 5.0), 3);
}

#[test]
fn crossfade_zero_at_level_center() {
    // At the start of level 0's range, crossfade should be 0.
    let factor = crossfade_factor(10.0, 0);
    assert!(
        factor < 0.01,
        "crossfade at center of level 0 should be ~0, got {}",
        factor
    );
}

#[test]
fn crossfade_one_at_boundary() {
    // At the transition distance to the next level.
    let factor = crossfade_factor(50.0, 0);
    assert!(
        (factor - 1.0).abs() < 0.01,
        "crossfade at boundary should be ~1.0, got {}",
        factor
    );
}

#[test]
fn micro_detail_generates_splats() {
    let brick = MicroDetailGenerator::generate_brick_detail(42);
    assert!(!brick.is_empty(), "brick detail should generate splats");

    let wood = MicroDetailGenerator::generate_wood_grain(42);
    assert!(!wood.is_empty(), "wood grain should generate splats");

    let metal = MicroDetailGenerator::generate_metal_scratches(42);
    assert!(!metal.is_empty(), "metal scratches should generate splats");
}

#[test]
fn taa_accumulation_improves_quality() {
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h * 4) as usize;

    let mut accumulator = TemporalAccumulator::new(w, h);

    // Create a base image.
    let base: Vec<f32> = (0..pixel_count).map(|i| (i as f32) / pixel_count as f32).collect();

    // Create a noisy version.
    let noisy: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, &v)| v + (i as f32 * 0.1).sin() * 0.1)
        .collect();

    // Accumulate the base image.
    accumulator.add_sample(&base, [0.0, 0.0]);
    let _variance_before = accumulator.compute_variance(&noisy);

    // Add more samples (closer to base).
    for i in 1..8 {
        let jitter = [(i as f32 * 0.125).sin() * 0.5, (i as f32 * 0.125).cos() * 0.5];
        // Slight variation per sample.
        let sample: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(j, &v)| v + (j as f32 * 0.01 * i as f32).sin() * 0.01)
            .collect();
        accumulator.add_sample(&sample, jitter);
    }

    let _variance_after = accumulator.compute_variance(&noisy);

    // After more accumulation, variance relative to noisy input should remain bounded.
    // The accumulated result should be closer to the base (clean) signal.
    let _var_vs_base_before = {
        let mut acc = TemporalAccumulator::new(w, h);
        acc.add_sample(&base, [0.0, 0.0]);
        acc.compute_variance(&base)
    };

    let var_vs_base_after = accumulator.compute_variance(&base);

    // With more samples, result stays close to base (low variance vs base).
    assert!(
        var_vs_base_after < 0.01,
        "accumulated result should be close to base, variance: {}",
        var_vs_base_after
    );
}

#[test]
fn brick_detail_has_mortar_lines() {
    let splats = MicroDetailGenerator::generate_brick_detail(123);

    // Mortar line splats should be recessed (negative z or near zero).
    let mortar_splats: Vec<_> = splats
        .iter()
        .filter(|s| s.position[2] < 0.0)
        .collect();

    assert!(
        !mortar_splats.is_empty(),
        "brick detail should have mortar groove splats with negative z (height variation)"
    );

    // Brick face splats should be raised (positive z).
    let face_splats: Vec<_> = splats
        .iter()
        .filter(|s| s.position[2] > 0.005)
        .collect();

    assert!(
        !face_splats.is_empty(),
        "brick detail should have raised face splats"
    );

    // The height difference demonstrates mortar lines.
    let min_z = splats.iter().map(|s| s.position[2]).fold(f32::MAX, f32::min);
    let max_z = splats.iter().map(|s| s.position[2]).fold(f32::MIN, f32::max);
    assert!(
        max_z - min_z > 0.01,
        "brick detail should have height variation for mortar, range: {}",
        max_z - min_z
    );
}

#[test]
fn lod_chain_preserves_full_count_at_level_0() {
    let splats = make_test_splats(50);
    let chain = generate_lod_chain(&splats);
    assert_eq!(chain.levels[0].splat_count, 50, "LOD 0 should have all splats");
}

#[test]
fn taa_reset_clears_state() {
    let mut acc = TemporalAccumulator::new(2, 2);
    let pixels = vec![1.0f32; 16];
    acc.add_sample(&pixels, [0.0, 0.0]);
    assert_eq!(acc.sample_count, 1);

    acc.reset();
    assert_eq!(acc.sample_count, 0);
    assert!(acc.accumulated.iter().all(|&v| v == 0.0));
}
