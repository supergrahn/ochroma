use vox_render::benchmark::*;

#[test]
fn generate_1000_splats() {
    let splats = generate_benchmark_splats(1000);
    assert_eq!(splats.len(), 1000);

    // Verify deterministic grid layout — first splat at origin.
    assert_eq!(splats[0].position, [0.0, 0.0, 0.0]);
}

#[test]
fn benchmark_result_creation() {
    let frame_times = vec![16.0, 17.0, 15.0, 18.0, 16.5];
    let result = BenchmarkResult::from_samples(
        BenchmarkScene::SplatGrid1M,
        (1920, 1080),
        1_000_000,
        &frame_times,
        2.0,
        10.0,
    );

    assert_eq!(result.scene, BenchmarkScene::SplatGrid1M);
    assert_eq!(result.splat_count, 1_000_000);
    assert!(result.avg_frame_ms > 15.0 && result.avg_frame_ms < 18.0);
    assert!((result.min_frame_ms - 15.0).abs() < 0.01);
    assert!((result.max_frame_ms - 18.0).abs() < 0.01);
    assert!(result.fps > 50.0 && result.fps < 70.0);
}

#[test]
fn export_json_format() {
    let frame_times = vec![16.0, 16.5];
    let result = BenchmarkResult::from_samples(
        BenchmarkScene::CityBlock,
        (1280, 720),
        500_000,
        &frame_times,
        1.5,
        8.0,
    );

    let json = export_results_json(&[result]);
    assert!(json.contains("CityBlock"));
    assert!(json.contains("splat_count"));
    assert!(json.contains("avg_frame_ms"));

    // Verify it's valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_array());
}

#[test]
fn regression_detection_slower_flagged() {
    let baseline = vec![BenchmarkResult::from_samples(
        BenchmarkScene::SplatGrid1M,
        (1920, 1080),
        1_000_000,
        &[10.0; 10],
        2.0,
        6.0,
    )];

    // 20% slower.
    let current = vec![BenchmarkResult::from_samples(
        BenchmarkScene::SplatGrid1M,
        (1920, 1080),
        1_000_000,
        &[12.0; 10],
        2.4,
        7.2,
    )];

    let regressions = compare_results(&baseline, &current);
    assert!(!regressions.is_empty());
    assert!(regressions.iter().any(|r| r.metric == "avg_frame_ms"));
    assert!(regressions[0].percent_change > 5.0);
}

#[test]
fn regression_detection_faster_ok() {
    let baseline = vec![BenchmarkResult::from_samples(
        BenchmarkScene::SplatGrid1M,
        (1920, 1080),
        1_000_000,
        &[10.0; 10],
        2.0,
        6.0,
    )];

    // 20% faster — no regression.
    let current = vec![BenchmarkResult::from_samples(
        BenchmarkScene::SplatGrid1M,
        (1920, 1080),
        1_000_000,
        &[8.0; 10],
        1.6,
        4.8,
    )];

    let regressions = compare_results(&baseline, &current);
    assert!(regressions.is_empty());
}

#[test]
fn benchmark_suite_add_and_query() {
    let mut suite = BenchmarkSuite::new();
    let result = BenchmarkResult::from_samples(
        BenchmarkScene::StressTest,
        (1920, 1080),
        10_000_000,
        &[33.0; 5],
        5.0,
        20.0,
    );
    suite.add_result(result);

    assert_eq!(suite.results.len(), 1);
    assert!(suite.result_for(BenchmarkScene::StressTest).is_some());
    assert!(suite.result_for(BenchmarkScene::CityBlock).is_none());
}
