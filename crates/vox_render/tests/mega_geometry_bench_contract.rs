//! Local contract tests for the City MegaGeometry three-mode benchmark.
//!
//! These exercise the benchmark's REAL result-processing + verdict logic (the
//! module shared verbatim with `examples/mega_geometry_bench.rs`) with zero GPU.
//! Every test checks a real computed outcome of that logic — a fixed pass string
//! is exactly what the "Evidence gate" capability forbids, and these prove the
//! suppression rules actually fire.
//!
//! The builders below simulate a FULLY-WIRED future (every gate measured and
//! passing) so the winning path is testable; the harness itself feeds most gates
//! as `None`/`Unavailable` and therefore always fails closed.

// Include the same pure logic the hardware harness uses.
#[path = "../examples/support/mega_geometry_bench_logic.rs"]
mod logic;

use logic::*;

// ---------------------------------------------------------------------------
// Builders — clean, honest, FULLY-MEASURED records that pass every gate.
// ---------------------------------------------------------------------------

fn exact_correctness() -> Correctness {
    Correctness {
        source_primitives: 12,
        mapped_primitives: 12,
        material_mismatches: Some(0),
        uv_mismatches: Some(0),
        hit_mismatches: Some(0),
    }
}

fn clean_backend() -> BackendResolution {
    BackendResolution {
        requested_rt_backend: "optix".into(),
        actual_rt_backend: "optix".into(),
        requested_compute: ComputeVariant::PortableSubgroup,
        actual_compute: ComputeVariant::PortableSubgroup,
        available: true,
        silently_downgraded: false,
        cpu_fallbacks: 0,
        cluster_requested_without_hardware_clas: false,
    }
}

fn clean_prewarm() -> PrewarmState {
    PrewarmState {
        shaders_compiled_before_timed: true,
        pipeline_created_in_timed_frame: false,
        cache_repaired_in_timed_frame: false,
    }
}

fn clean_stats() -> StatsCollection {
    StatsCollection {
        collected_after_timed_frames: true,
        inside_present_timing: false,
    }
}

fn full_vram(peak_mb: f64) -> VramBreakdown {
    VramBreakdown {
        resident_mb: peak_mb,
        as_mb: Some(0.0),
        page_mb: Some(0.0),
        native_args_mb: Some(0.0),
        scratch_mb: Some(0.0),
    }
}

fn mode_result(mode: Mode, stage_p95: f64, peak_mb: f64) -> ModeResult {
    ModeResult {
        mode,
        compute: ComputeVariant::PortableSubgroup,
        correctness: exact_correctness(),
        stage: GeometryStage {
            geometry_stage_ms_p95: stage_p95,
            frame_ms_p95: stage_p95,
            required_page_misses: Some(0),
            native_lowering_ms_p95: Some(0.0),
            ..Default::default()
        },
        vram: full_vram(peak_mb),
        cpu: CpuCounters::default(),
        prewarm: Some(clean_prewarm()),
        stats: Some(clean_stats()),
        backend: clean_backend(),
        unexpected_fallback: false,
    }
}

/// A full seven-fixture suite whose measurements yield the ideal outcomes. To
/// exercise the WIN path, `mixed_city` (externally authorized in reality) is
/// overridden to internally-authorized here, simulating the future where its
/// Urban Horizon present witness is fed in.
fn winning_suite() -> Vec<FixtureResult> {
    required_fixtures()
        .into_iter()
        .map(|mut spec| {
            spec.authorizing_witness_external = false; // simulate wired future
            let (hybrid, reference) = match spec.role {
                FixtureRole::TargetWin => (
                    mode_result(Mode::CityHybrid, 5.0, 100.0),
                    mode_result(Mode::Reference, 10.0, 100.0),
                ),
                FixtureRole::Parity => (
                    mode_result(Mode::CityHybrid, 10.05, 100.0),
                    mode_result(Mode::Reference, 10.0, 100.0),
                ),
                FixtureRole::CorrectnessOnly => (
                    mode_result(Mode::CityHybrid, 10.0, 100.0),
                    mode_result(Mode::Reference, 10.0, 100.0),
                ),
            };
            // Fully-wired future: distinct execution proven.
            let outcome = evaluate_fixture(spec.role, false, true, &hybrid, &reference);
            FixtureResult {
                spec,
                outcome,
                hybrid,
                reference,
            }
        })
        .collect()
}

/// All global signals measured and passing (the wired future).
fn measured_global() -> GlobalSignals {
    GlobalSignals {
        packet_abi_match: Some(true),
        control_hash_match: Some(true),
        adapter_overhead_pct: Some(0.0),
        cpu_fallbacks: Some(0),
        fabric_cpu_scheduling: Some(0),
        cache_valid: Some(true),
        mode_distinct: Some(true),
        scale_small: ScaleSample {
            total_units: 1,
            cpu: CpuCounters::default(),
        },
        scale_large: ScaleSample {
            total_units: 1_000_000,
            cpu: CpuCounters::default(),
        },
    }
}

fn rescore(fr: &mut FixtureResult) {
    fr.outcome = evaluate_fixture(
        fr.spec.role,
        fr.spec.authorizing_witness_external,
        true,
        &fr.hybrid,
        &fr.reference,
    );
}

fn program_pass() -> ProgramBreakthrough {
    ProgramBreakthrough {
        repeated_storage_ratio: 0.20,
        mixed_storage_ratio: 0.45,
        fixed_budget_detail_ratio: 2.5,
        lod_temporal_invalidations_reduction: 0.60,
        full_frame_regression_pct: 1.0,
        losses: 0,
        axis_storage_bytes_win: true,
        axis_peak_vram_win: true,
        axis_geometry_stage_win: true,
        axis_visible_detail_win: false,
        axis_lod_temporal_win: false,
    }
}

fn visibility_pass() -> VisibilityBreakthrough {
    VisibilityBreakthrough {
        ray_native_coverage: 0.70,
        materialized_geometry_ratio: 0.20,
        parameter_update_fine_as_builds: 0,
        geometry_stage_speedup: 1.8,
        fixed_budget_detail_ratio: 4.2,
        ray_native_temporal_invalidations: 0,
        correctness_mismatches: 0,
        real_asset_families: 2,
        losses: 0,
        procedural_intersection_win: true,
        removed_triangles_counted_as_axis: true,
        as_bytes_counted_as_axis: false,
    }
}

fn fabric_pass() -> FabricBreakthrough {
    FabricBreakthrough {
        structured_ray_coverage: 0.65,
        certified_without_per_ray_traversal: 0.40,
        active_packet_lanes: 0.80,
        intersection_speedup_vs_scalar: 2.3,
        geometry_stage_speedup: 2.1,
        routing_queue_overhead_pct: 8.0,
        correctness_mismatches: 0,
        cpu_scheduling: 0,
        real_asset_families: 2,
        losses: 0,
        queue_overflows: 0,
        only_ray_sorting: false,
        includes_failed_certificates: true,
        includes_fixed_empty_launches: true,
    }
}

// ---------------------------------------------------------------------------
// The 21 required contract tests
// ---------------------------------------------------------------------------

#[test]
fn requested_backend_fallback_exits_nonzero() {
    assert_eq!(clean_backend().exit_code(), 0);

    let mut fell_back = clean_backend();
    fell_back.actual_rt_backend = "vulkan_khr".into(); // requested optix
    assert_eq!(fell_back.exit_code(), 1);

    let mut unavailable = clean_backend();
    unavailable.available = false;
    assert_eq!(unavailable.exit_code(), 1);

    // A cluster path requested but producing zero HARDWARE CLAS builds is a
    // silent downgrade (a derived cluster count is not CLAS evidence).
    let mut derived_cluster = clean_backend();
    derived_cluster.cluster_requested_without_hardware_clas = true;
    assert_eq!(derived_cluster.exit_code(), 1);
}

#[test]
fn correctness_failure_suppresses_performance_verdict() {
    let base = compute_overall(&winning_suite(), &measured_global());
    assert_eq!(base.winner, Some(Mode::CityHybrid));

    let mut suite = winning_suite();
    let target = suite
        .iter_mut()
        .find(|f| f.spec.role == FixtureRole::TargetWin)
        .unwrap();
    target.hybrid.correctness.hit_mismatches = Some(7);
    rescore(target);
    assert!(matches!(
        target.outcome,
        FixtureOutcome::CorrectnessFailure { .. }
    ));

    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.hard_abort);
    assert!(v.suppressed_reason.unwrap().contains("correctness"));
}

#[test]
fn result_json_contains_environment_and_fixture_hashes() {
    let suite = winning_suite();
    let fixtures: Vec<FixtureReport> = suite
        .iter()
        .map(|fr| FixtureReport {
            name: fr.spec.name.clone(),
            role: fr.spec.role,
            asset_id: fr.spec.asset_id.clone(),
            family: fr.spec.family.clone(),
            instances: fr.spec.instances,
            seed: fr.spec.seed,
            identity_hash: fr.spec.identity_hash_hex(),
            asset_content_hash: None,
            outcome: fr.outcome.clone(),
            hybrid: fr.hybrid.clone(),
            reference: fr.reference.clone(),
            triangle: None,
        })
        .collect();
    let report = BenchReport {
        env: EnvLine {
            gpu: "NVIDIA_RTX_4070_Ti".into(),
            api: "cuda".into(),
            driver: "560.0".into(),
            rt_backend: "optix".into(),
            capability_hash: "deadbeefcap".into(),
            shader_abi: "abihash01".into(),
            cook_schema: 3,
        },
        suite: "city".into(),
        modes: MODES.iter().map(|m| m.to_string()).collect(),
        warmup_frames: 60,
        timed_frames: 240,
        seed: 42,
        command_line: "mega_geometry_bench --suite city".into(),
        git_revisions: GitRevisions {
            ochroma: "efe7ec3".into(),
            spectra: "478015f".into(),
            forge: "9cc1621".into(),
            urban: "2d23bdf".into(),
        },
        render_config_hash: "rcfg01".into(),
        corpus_hash: "corpus01".into(),
        fixtures,
        global: measured_global(),
        overall: compute_overall(&suite, &measured_global()),
        program_breakthrough: None,
        visibility_breakthrough: None,
        fabric_breakthrough: None,
    };

    let json = report.to_json().unwrap();
    assert!(json.contains("NVIDIA_RTX_4070_Ti"));
    assert!(json.contains("deadbeefcap"));
    assert!(json.contains("478015f"));
    let round = BenchReport::from_json(&json).unwrap();
    assert_eq!(round.fixtures.len(), 7);
    for f in &round.fixtures {
        assert_eq!(f.identity_hash.len(), 16);
        assert!(json.contains(&f.identity_hash));
        // The real-asset content is not loaded here; it must NOT be certified.
        assert!(f.asset_content_hash.is_none());
    }
    assert_eq!(round.env.gpu, "NVIDIA_RTX_4070_Ti");
    assert_eq!(round.git_revisions.spectra, "478015f");
}

#[test]
fn overall_win_requires_no_losses_and_all_target_wins() {
    let v = compute_overall(&winning_suite(), &measured_global());
    assert_eq!(v.winner, Some(Mode::CityHybrid));
    assert_eq!(v.losses, 0);
    assert_eq!(v.target_wins, REQUIRED_TARGET_WINS);
    assert_eq!(v.claim, "better_city_geometry_system");

    let mut suite = winning_suite();
    let t = suite
        .iter_mut()
        .find(|f| f.spec.role == FixtureRole::TargetWin)
        .unwrap();
    t.hybrid.stage.geometry_stage_ms_p95 = 20.0;
    rescore(t);
    assert!(matches!(t.outcome, FixtureOutcome::Loss { .. }));
    let v2 = compute_overall(&suite, &measured_global());
    assert_eq!(v2.winner, None);
    assert_eq!(v2.losses, 1);
    assert!(v2.target_wins < REQUIRED_TARGET_WINS);
    assert_ne!(v2.claim, "better_city_geometry_system");
}

#[test]
fn parity_and_correctness_only_fixtures_cannot_be_mislabeled_wins() {
    let hybrid = mode_result(Mode::CityHybrid, 1.0, 100.0);
    let reference = mode_result(Mode::Reference, 10.0, 100.0);
    let parity = evaluate_fixture(FixtureRole::Parity, false, true, &hybrid, &reference);
    assert!(!parity.is_win());
    assert!(matches!(parity, FixtureOutcome::Parity { .. }));

    let correctness_only =
        evaluate_fixture(FixtureRole::CorrectnessOnly, false, true, &hybrid, &reference);
    assert!(!correctness_only.is_win());
    assert!(matches!(
        correctness_only,
        FixtureOutcome::CorrectnessOnly { .. }
    ));
}

#[test]
fn forbidden_cpu_counter_suppresses_all_winners() {
    let mut suite = winning_suite();
    suite[0].hybrid.cpu.lod_decisions = 1;
    assert!(suite[0].hybrid.cpu.any_forbidden_nonzero());
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.hard_abort);
    assert!(v.suppressed_reason.unwrap().contains("CPU"));

    // CPU cost that scales with problem size aborts.
    let mut g = measured_global();
    g.scale_large.cpu.page_decisions = 5;
    assert!(cpu_cost_scales_with_size(&g.scale_small, &g.scale_large));
    let v2 = compute_overall(&winning_suite(), &g);
    assert_eq!(v2.winner, None);
    assert!(v2.suppressed_reason.unwrap().contains("scales"));
}

#[test]
fn delayed_stats_are_outside_present_timing() {
    assert!(clean_stats().is_valid());
    let inside = StatsCollection {
        collected_after_timed_frames: true,
        inside_present_timing: true,
    };
    assert!(!inside.is_valid());
    let early = StatsCollection {
        collected_after_timed_frames: false,
        inside_present_timing: false,
    };
    assert!(!early.is_valid());

    let mut suite = winning_suite();
    suite[2].reference.stats = Some(inside);
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.suppressed_reason.unwrap().contains("present timing"));
}

#[test]
fn capability_or_reflection_mismatch_invalidates_cache() {
    let a = CacheKey {
        capability_hash: [1u8; 32],
        shader_abi: [2u8; 32],
        driver_id: 7,
        cook_schema: 3,
    };
    assert!(cache_is_valid(&a, &a));
    let mut cap_changed = a;
    cap_changed.capability_hash = [9u8; 32];
    assert!(!cache_is_valid(&a, &cap_changed));
    let mut abi_changed = a;
    abi_changed.shader_abi = [9u8; 32];
    assert!(!cache_is_valid(&a, &abi_changed));

    let mut g = measured_global();
    g.cache_valid = Some(false);
    let v = compute_overall(&winning_suite(), &g);
    assert_eq!(v.winner, None);
    assert!(v.suppressed_reason.unwrap().contains("cache"));
}

#[test]
fn frame_time_pipeline_creation_suppresses_winner() {
    let mut suite = winning_suite();
    let mut bad = clean_prewarm();
    bad.pipeline_created_in_timed_frame = true;
    assert!(!bad.is_valid());
    suite[1].hybrid.prewarm = Some(bad);
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.suppressed_reason.unwrap().contains("pipeline creation"));
}

#[test]
fn peak_vram_includes_build_scratch_and_native_args() {
    let v = VramBreakdown {
        resident_mb: 10.0,
        as_mb: Some(5.0),
        page_mb: Some(1.0),
        native_args_mb: Some(4.0),
        scratch_mb: Some(20.0),
    };
    assert_eq!(v.peak_geometry_mb(), Some(40.0));
    let without_transient = v.resident_mb + v.as_mb.unwrap() + v.page_mb.unwrap();
    assert!(v.peak_geometry_mb().unwrap() > without_transient);

    // Lower resident but a bigger scratch spike -> higher peak -> VRAM loss.
    let mut hybrid = mode_result(Mode::CityHybrid, 3.0, 0.0);
    hybrid.vram = VramBreakdown {
        resident_mb: 10.0,
        as_mb: Some(0.0),
        page_mb: Some(0.0),
        native_args_mb: Some(0.0),
        scratch_mb: Some(200.0),
    };
    let mut reference = mode_result(Mode::Reference, 10.0, 0.0);
    reference.vram = VramBreakdown {
        resident_mb: 100.0,
        as_mb: Some(0.0),
        page_mb: Some(0.0),
        native_args_mb: Some(0.0),
        scratch_mb: Some(0.0),
    };
    assert!(hybrid.vram.resident_mb < reference.vram.resident_mb);
    assert!(hybrid.vram.peak_geometry_mb().unwrap() > reference.vram.peak_geometry_mb().unwrap());
    let outcome = evaluate_fixture(FixtureRole::TargetWin, false, true, &hybrid, &reference);
    assert!(matches!(outcome, FixtureOutcome::Loss { .. }));

    // An unmeasured VRAM component makes the peak Unavailable -> not a hidden win.
    let mut unmeasured = mode_result(Mode::CityHybrid, 3.0, 0.0);
    unmeasured.vram.scratch_mb = None;
    assert_eq!(unmeasured.vram.peak_geometry_mb(), None);
    let outcome2 = evaluate_fixture(FixtureRole::TargetWin, false, true, &unmeasured, &reference);
    assert!(matches!(outcome2, FixtureOutcome::Unavailable { .. }));
}

#[test]
fn adapter_or_compute_downgrade_is_never_silent() {
    let mut downgraded = clean_backend();
    downgraded.silently_downgraded = true;
    assert_eq!(downgraded.exit_code(), 1);
    assert!(!downgraded.is_honest());

    let mut cpu_fell_back = clean_backend();
    cpu_fell_back.cpu_fallbacks = 1;
    assert_eq!(cpu_fell_back.exit_code(), 1);

    let mut suite = winning_suite();
    suite[3].hybrid.backend.silently_downgraded = true;
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.suppressed_reason.unwrap().contains("downgrade"));
}

#[test]
fn breakthrough_claim_requires_independent_axis_wins() {
    let mut b = program_pass();
    assert_eq!(b.independent_axis_wins(), 3);
    assert_eq!(b.claim(), Some("city_geometry_program_breakthrough"));
    b.axis_geometry_stage_win = false;
    assert_eq!(b.independent_axis_wins(), 2);
    assert_eq!(b.claim(), None);
}

#[test]
fn visible_detail_counts_only_oracle_valid_geometry() {
    let d = VisibleDetail {
        total_detail_units: 1000,
        oracle_valid_units: 400,
    };
    assert_eq!(d.counted(), 400);
    assert_ne!(d.counted(), d.total_detail_units);
    assert_eq!(d.detail_ratio_vs(200), 2.0);
    let overclaim = VisibleDetail {
        total_detail_units: 100,
        oracle_valid_units: 999,
    };
    assert_eq!(overclaim.counted(), 100);
}

#[test]
fn storage_win_cannot_hide_decode_build_or_temporal_loss() {
    let mut b = program_pass();
    b.full_frame_regression_pct = 6.0;
    assert!(b.axis_storage_bytes_win);
    assert_eq!(b.claim(), None);

    let mut b2 = program_pass();
    b2.lod_temporal_invalidations_reduction = 0.10;
    assert_eq!(b2.claim(), None);
}

#[test]
fn ray_native_claim_requires_two_real_asset_families() {
    assert_eq!(visibility_pass().claim(), Some("ray_native_city_geometry"));
    let mut one_family = visibility_pass();
    one_family.real_asset_families = 1;
    assert_eq!(one_family.claim(), None);
}

#[test]
fn removed_triangles_and_as_bytes_are_not_double_counted_as_independent_wins() {
    assert_eq!(visibility_pass().claim(), Some("ray_native_city_geometry"));
    let mut double = visibility_pass();
    double.as_bytes_counted_as_axis = true;
    assert!(double.removed_triangles_counted_as_axis && double.as_bytes_counted_as_axis);
    assert_eq!(double.claim(), None);
}

#[test]
fn procedural_intersection_loss_suppresses_visibility_claim_only() {
    let mut vis = visibility_pass();
    vis.procedural_intersection_win = false;
    assert_eq!(vis.claim(), None);
    let overall = compute_overall(&winning_suite(), &measured_global());
    assert_eq!(overall.winner, Some(Mode::CityHybrid));
    assert_eq!(fabric_pass().claim(), Some("certified_visibility_fabric"));
}

#[test]
fn fabric_claim_requires_certified_work_elimination_not_only_ray_sorting() {
    assert_eq!(fabric_pass().claim(), Some("certified_visibility_fabric"));
    let mut sort_only = fabric_pass();
    sort_only.only_ray_sorting = true;
    assert_eq!(sort_only.claim(), None);
    let mut low_cert = fabric_pass();
    low_cert.certified_without_per_ray_traversal = 0.20;
    assert_eq!(low_cert.claim(), None);
}

#[test]
fn fabric_claim_includes_failed_certificates_and_fixed_empty_launches() {
    let mut no_failed = fabric_pass();
    no_failed.includes_failed_certificates = false;
    assert_eq!(no_failed.claim(), None);
    let mut no_empty = fabric_pass();
    no_empty.includes_fixed_empty_launches = false;
    assert_eq!(no_empty.claim(), None);
}

#[test]
fn fabric_loss_suppresses_only_fabric_claim() {
    let mut lost = fabric_pass();
    lost.losses = 1;
    assert_eq!(lost.claim(), None);
    assert_eq!(
        compute_overall(&winning_suite(), &measured_global()).winner,
        Some(Mode::CityHybrid)
    );
    assert_eq!(visibility_pass().claim(), Some("ray_native_city_geometry"));
}

#[test]
fn cpu_queue_scheduling_suppresses_fabric_and_overall_claims() {
    let mut sched = fabric_pass();
    sched.cpu_scheduling = 1;
    assert_eq!(sched.claim(), None);

    let mut g = measured_global();
    g.fabric_cpu_scheduling = Some(1);
    let v = compute_overall(&winning_suite(), &g);
    assert_eq!(v.winner, None);
    assert!(v.suppressed_reason.unwrap().contains("CPU scheduling"));
}

// ---------------------------------------------------------------------------
// Additional rule-locking tests (fail-closed law + fixture-set pinning)
// ---------------------------------------------------------------------------

#[test]
fn verdict_requires_complete_fixture_set() {
    assert_eq!(
        compute_overall(&winning_suite(), &measured_global()).winner,
        Some(Mode::CityHybrid)
    );

    // Only the four target fixtures: a subset is not a suite verdict.
    let subset: Vec<FixtureResult> = winning_suite()
        .into_iter()
        .filter(|f| f.spec.role == FixtureRole::TargetWin)
        .collect();
    assert_eq!(subset.len(), 4);
    let names: Vec<String> = subset.iter().map(|f| f.spec.name.clone()).collect();
    assert!(!all_required_fixtures_present(&names));
    let v = compute_overall(&subset, &measured_global());
    assert_eq!(v.winner, None, "a 4-fixture subset must not win");
    assert!(v.hard_abort);
    assert!(v.suppressed_reason.unwrap().contains("fixture set"));
}

#[test]
fn all_seven_required_fixtures_have_distinct_stable_hashes() {
    let f = required_fixtures();
    assert_eq!(f.len(), 7);
    let mut hashes: Vec<u64> = f.iter().map(|s| s.identity_hash()).collect();
    hashes.sort_unstable();
    hashes.dedup();
    assert_eq!(hashes.len(), 7, "fixture identity hashes must be distinct");
    let mixed = f.iter().find(|s| s.name == "mixed_city").unwrap();
    assert!(mixed.authorizing_witness_external);
}

#[test]
fn unmeasured_global_signal_suppresses_verdict() {
    // The harness's real default: every global signal unavailable -> fail closed.
    let g = GlobalSignals::unavailable(
        ScaleSample {
            total_units: 1,
            cpu: CpuCounters::default(),
        },
        ScaleSample {
            total_units: 1_000_000,
            cpu: CpuCounters::default(),
        },
    );
    let v = compute_overall(&winning_suite(), &g);
    assert_eq!(v.winner, None, "unavailable global signals must fail closed");
    assert!(v.hard_abort);

    // Each individual signal, when unavailable, suppresses on its own.
    let mutators: [fn(&mut GlobalSignals); 7] = [
        |g| g.packet_abi_match = None,
        |g| g.control_hash_match = None,
        |g| g.adapter_overhead_pct = None,
        |g| g.cpu_fallbacks = None,
        |g| g.fabric_cpu_scheduling = None,
        |g| g.cache_valid = None,
        |g| g.mode_distinct = None,
    ];
    for mutate in mutators {
        let mut g = measured_global();
        mutate(&mut g);
        let v = compute_overall(&winning_suite(), &g);
        assert_eq!(v.winner, None);
        assert!(v.hard_abort);
    }
}

#[test]
fn unmeasured_mode_evidence_suppresses_verdict() {
    // VRAM breakdown unavailable -> target win Unavailable.
    let mut suite = winning_suite();
    suite[3].hybrid.vram.scratch_mb = None;
    rescore(&mut suite[3]);
    assert!(matches!(suite[3].outcome, FixtureOutcome::Unavailable { .. }));
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.hard_abort);

    // Correctness material/uv unavailable -> Unavailable (not zero mismatches).
    let mut c = exact_correctness();
    c.material_mismatches = None;
    assert_eq!(c.is_exact(), None);

    // Page-miss counter unavailable -> suppress.
    let mut suite2 = winning_suite();
    suite2[0].hybrid.stage.required_page_misses = None;
    let v2 = compute_overall(&suite2, &measured_global());
    assert_eq!(v2.winner, None);
    assert!(v2.hard_abort);

    // Prewarm evidence unavailable -> suppress.
    let mut suite3 = winning_suite();
    suite3[1].hybrid.prewarm = None;
    let v3 = compute_overall(&suite3, &measured_global());
    assert_eq!(v3.winner, None);
    assert!(v3.hard_abort);

    // Submit-time budget unavailable -> suppress.
    let mut suite4 = winning_suite();
    suite4[2].hybrid.cpu.host_submit_ms_p95 = None;
    let v4 = compute_overall(&suite4, &measured_global());
    assert_eq!(v4.winner, None);
    assert!(v4.hard_abort);
}

#[test]
fn externally_authorized_fixture_is_never_a_local_win() {
    let spec = required_fixtures()
        .into_iter()
        .find(|s| s.name == "mixed_city")
        .unwrap();
    assert!(spec.authorizing_witness_external);
    let hybrid = mode_result(Mode::CityHybrid, 1.0, 10.0);
    let reference = mode_result(Mode::Reference, 100.0, 100.0);
    let outcome = evaluate_fixture(spec.role, true, true, &hybrid, &reference);
    assert!(matches!(outcome, FixtureOutcome::Unavailable { .. }));
}

#[test]
fn non_distinct_execution_cannot_win_a_target_fixture() {
    // Even with a huge speedup, if city_hybrid is not a distinct execution path
    // it cannot be a win — two identical configs timed on noise is not a system.
    let hybrid = mode_result(Mode::CityHybrid, 1.0, 10.0);
    let reference = mode_result(Mode::Reference, 100.0, 100.0);
    let outcome = evaluate_fixture(FixtureRole::TargetWin, false, false, &hybrid, &reference);
    assert!(matches!(outcome, FixtureOutcome::Unavailable { .. }));
    let win = evaluate_fixture(FixtureRole::TargetWin, false, true, &hybrid, &reference);
    assert!(win.is_win());
}

// ---------------------------------------------------------------------------
// Re-judge round: no zero-measurement may map to a passing value.
// ---------------------------------------------------------------------------

#[test]
fn zero_host_submit_calls_suppresses_submit_budget() {
    // A default (wired-future) counter has one recorded submission and passes.
    assert!(CpuCounters::default().submit_budget_reason().is_none());

    // Zero recorded host-submit calls is "nothing measured" and must suppress,
    // even with a stored 0.0 p95 (a measured-pass from zero measurements).
    let mut c = CpuCounters::default();
    c.host_submit_calls = 0;
    c.host_submit_ms_p95 = Some(0.0);
    assert!(
        c.submit_budget_reason().is_some(),
        "zero host-submit calls must not certify the submit budget"
    );

    // And it suppresses the whole verdict.
    let mut suite = winning_suite();
    suite[2].hybrid.cpu.host_submit_calls = 0;
    suite[2].hybrid.cpu.host_submit_ms_p95 = Some(0.0);
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None);
    assert!(v.hard_abort);
}

#[test]
fn unprobed_cpu_counters_suppress() {
    // The default probe is wired; the suite wins.
    assert_eq!(
        compute_overall(&winning_suite(), &measured_global()).winner,
        Some(Mode::CityHybrid)
    );

    // Unprobed counters (all zeros, but never actually read) are not evidence.
    let mut suite = winning_suite();
    suite[0].hybrid.cpu.counters_probed = false;
    assert!(!suite[0].hybrid.cpu.any_forbidden_nonzero()); // zeros look "clean"...
    let v = compute_overall(&suite, &measured_global());
    assert_eq!(v.winner, None, "unprobed zero counters must not pass");
    assert!(v.hard_abort);
    assert!(v.suppressed_reason.unwrap().contains("not probed"));

    // The harness ships with the probe unwired, so it fails closed by default.
    assert!(!HARNESS_CPU_OWNERSHIP_COUNTERS_PROBED);
}

#[test]
fn harness_proves_distinct_execution() {
    // Pin the harness constant. This was flipped to `true` when the renderer
    // began honoring MEGAGEOMETRY_MODE=city_hybrid by building a hierarchical
    // root→child IAS (distinct from reference's flat CLAS IAS), witnessed on the
    // 4070 Ti with the provenance oracle at 0/0/0. A stray flip BACK to false
    // (regressing the wiring) is caught here.
    assert!(
        HARNESS_CITY_HYBRID_DISTINCT_EXECUTION,
        "city_hybrid IS a distinct execution path (hierarchical IAS); do not flip \
         this to false unless the per-mode execution wiring is removed"
    );
}

#[test]
fn hit_oracle_never_zero_from_zero_comparisons() {
    // Triangle mode is its own oracle: self-compare is zero.
    assert_eq!(hit_oracle_mismatches(true, true, &[0.0, 1.0], Some(&[0.0, 1.0])), Some(0));

    // No triangle oracle present -> Unavailable, never Some(0).
    assert_eq!(hit_oracle_mismatches(false, false, &[0.0], Some(&[0.0])), None);
    assert_eq!(hit_oracle_mismatches(false, true, &[0.0], None), None);

    // Empty candidate/oracle -> Unavailable (a 0-from-0 compare is not evidence).
    assert_eq!(hit_oracle_mismatches(false, true, &[], Some(&[])), None);
    assert_eq!(hit_oracle_mismatches(false, true, &[1.0], Some(&[])), None);
    assert_eq!(hit_oracle_mismatches(false, true, &[], Some(&[1.0])), None);

    // A real difference beyond the channel threshold is counted.
    let m = hit_oracle_mismatches(false, true, &[0.0, 0.0], Some(&[1.0, 0.0])).unwrap();
    assert_eq!(m, 1);
    // A length mismatch is a mismatch, not a pass.
    assert!(hit_oracle_mismatches(false, true, &[0.0], Some(&[0.0, 0.0])).unwrap() > 0);
}
