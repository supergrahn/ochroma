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

#[test]
fn hardware_clas_is_an_optix_capability_requirement_not_a_product_requirement() {
    assert!(requires_hardware_clas("optix", Mode::Reference));
    assert!(requires_hardware_clas("optix", Mode::CityHybrid));
    assert!(!requires_hardware_clas("optix", Mode::Triangle));

    for mode in [Mode::Triangle, Mode::Reference, Mode::CityHybrid] {
        assert!(
            !requires_hardware_clas("vulkan_khr", mode),
            "Vulkan mode {mode:?} must remain a valid native triangle/procedural path"
        );
        assert!(
            !requires_hardware_clas("metal", mode),
            "Metal mode {mode:?} must remain a valid native triangle/procedural path"
        );
    }
}

#[test]
fn only_city_hybrid_consumes_ochroma_runtime_megageometry() {
    assert!(!mode_uses_runtime_megageometry(Mode::Triangle));
    assert!(!mode_uses_runtime_megageometry(Mode::Reference));
    assert!(mode_uses_runtime_megageometry(Mode::CityHybrid));
}

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
        hardware_clas_builds: Some(1),
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
        packet_abi_hash: Some("packet".into()),
        control_hash: Some("control".into()),
        capability_fingerprint: Some("capability".into()),
        visibility_fabric: Some(VisibilityQueueCounters {
            domains_seeded: 1,
            rays_certified_miss: 0,
            rays_certified_hit: 64,
            domains_split: 0,
            explicit_rays: 0,
            packet_rays: 64,
            packet_active_lanes: 64,
            packet_total_lanes: 64,
            scalar_refinement_rays: 0,
            native_residual_rays: 0,
            failed_certificates: 0,
            queue_overflow_mask: 0,
            cpu_scheduling: 0,
            correctness_mismatches: 0,
            source_ray_capacity: 64,
            live_source_rays: 64,
        }),
        visibility_profile: Some(VisibilityKernelProfile {
            measured_frames: 8,
            certification_ms_p95: 0.25,
            subdivision_ms_p95: 0.10,
            routing_queue_ms_p95: 0.10,
            packet_intersection_ms_p95: 0.20,
            exact_refinement_ms_p95: 0.10,
            ordinary_native_ms_p95: 0.20,
            native_residual_ms_p95: 0.10,
            hit_merge_ms_p95: 0.05,
            fixed_empty_launch_ms_p95: 0.05,
            complete_visibility_ms_p95: 1.0,
            includes_fixed_empty_launches: true,
        }),
        temporal_reuse: Some(TemporalReuseEvidence {
            relevant_surface_pixels: 64,
            invalidated_surface_pixels: 0,
            no_map_invalidated_surface_pixels: 32,
            lod_cut_changes: 8,
        }),
        parameter_update_fine_as_builds: Some(0),
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

fn conformance_report(api: &str, gpu: &str) -> BenchReport {
    let suite = winning_suite();
    let fixtures = suite
        .iter()
        .map(|fr| FixtureReport {
            name: fr.spec.name.clone(),
            role: fr.spec.role,
            asset_id: fr.spec.asset_id.clone(),
            family: fr.spec.family.clone(),
            instances: fr.spec.instances,
            seed: fr.spec.seed,
            identity_hash: fr.spec.identity_hash_hex(),
            asset_content_hash: Some(format!("asset-{}", fr.spec.name)),
            outcome: fr.outcome.clone(),
            hybrid: fr.hybrid.clone(),
            reference: fr.reference.clone(),
            triangle: None,
            scalar_visibility_profile: fr.reference.visibility_profile,
            runtime_program: None,
        })
        .collect();
    BenchReport {
        env: EnvLine {
            gpu: gpu.into(),
            api: api.into(),
            driver: "driver".into(),
            rt_backend: if api == "cuda" {
                "optix".into()
            } else {
                "vulkan_khr".into()
            },
            capability_hash: format!("{api}-capability"),
            shader_abi: "shared-shader-abi".into(),
            runtime_geometry_schema: vox_data::mega_geometry::READY_MEGA_GEOMETRY_SCHEMA,
        },
        suite: "city".into(),
        modes: MODES.iter().map(|mode| mode.to_string()).collect(),
        warmup_frames: 60,
        timed_frames: 240,
        seed: 42,
        command_line: format!("mega_geometry_bench --api {api}"),
        git_revisions: GitRevisions {
            ochroma: "ochroma-rev".into(),
            spectra: "spectra-rev".into(),
            urban: "urban-rev".into(),
        },
        render_config_hash: "render-config".into(),
        corpus_hash: "corpus".into(),
        fixtures,
        global: GlobalSignals::unavailable(
            ScaleSample {
                total_units: 1,
                cpu: CpuCounters::default(),
            },
            ScaleSample {
                total_units: 2,
                cpu: CpuCounters::default(),
            },
        ),
        overall: compute_overall(&suite, &measured_global()),
        program_breakthrough: None,
        visibility_breakthrough: None,
        fabric_breakthrough: None,
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

#[test]
fn independent_backend_composer_is_the_only_cross_backend_match_authority() {
    let cuda = conformance_report("cuda", "NVIDIA");
    let vulkan = conformance_report("vulkan", "AMD");
    let composed = compose_backend_conformance(&cuda, &vulkan);
    assert!(composed.passes(), "{:?}", composed.reasons);
    assert_eq!(composed.packet_abi_match, Some(true));
    assert_eq!(composed.control_hash_match, Some(true));
    assert_eq!(composed.compared_mode_records, 12);
}

#[test]
fn backend_composer_fails_closed_on_missing_or_mismatched_raw_hash() {
    let cuda = conformance_report("cuda", "NVIDIA");
    let mut vulkan = conformance_report("vulkan", "AMD");
    vulkan.fixtures[0].hybrid.control_hash = None;
    let missing = compose_backend_conformance(&cuda, &vulkan);
    assert!(!missing.passes());
    assert_eq!(missing.control_hash_match, None);
    assert!(missing.unavailable_records > 0);

    vulkan.fixtures[0].hybrid.control_hash = Some("different".into());
    let mismatch = compose_backend_conformance(&cuda, &vulkan);
    assert!(!mismatch.passes());
    assert_eq!(mismatch.control_hash_match, Some(false));
    assert!(
        mismatch
            .reasons
            .iter()
            .any(|reason| reason.contains("control hash mismatch"))
    );
}

#[test]
fn backend_composer_rejects_cross_backend_correctness_failure() {
    let cuda = conformance_report("cuda", "nvidia");
    let mut vulkan = conformance_report("vulkan", "amd");
    vulkan.fixtures[0].hybrid.correctness.hit_mismatches = Some(1);
    let composed = compose_backend_conformance(&cuda, &vulkan);
    assert!(!composed.passes());
    assert!(
        composed
            .reasons
            .iter()
            .any(|reason| reason.contains("correctness"))
    );
}

#[test]
fn final_join_can_authorize_only_the_external_product_fixture() {
    let mut suite = winning_suite();
    let external = suite
        .iter_mut()
        .find(|fixture| fixture.spec.name == "mixed_city")
        .expect("mixed city");
    external.spec.authorizing_witness_external = true;
    external.outcome = FixtureOutcome::Unavailable {
        reason: "awaiting product artifact".into(),
    };
    external.hybrid = ModeResult::unavailable(Mode::CityHybrid, "optix");
    external.reference = ModeResult::unavailable(Mode::Reference, "optix");

    let raw = compute_overall(&suite, &measured_global());
    assert_eq!(raw.winner, None);
    assert!(raw.hard_abort);

    let joined = compute_overall_with_external(&suite, &measured_global(), true);
    assert_eq!(joined.winner, Some(Mode::CityHybrid));
    assert_eq!(joined.losses, 0);
    assert_eq!(joined.target_wins, REQUIRED_TARGET_WINS);
}

#[test]
fn adapter_overhead_is_derived_from_measured_submit_receipts() {
    let report = conformance_report("cuda", "gpu");
    assert_eq!(measured_adapter_overhead_pct(&report), Some(0.0));
    let mut missing = report;
    missing.fixtures[0].hybrid.cpu.host_submit_ms_p95 = None;
    assert_eq!(measured_adapter_overhead_pct(&missing), None);
}

#[test]
fn raw_backend_artifact_can_pass_without_claiming_external_or_cross_backend_evidence() {
    let mut report = conformance_report("cuda", "gpu");
    report.global = measured_global();
    for fixture in &mut report.fixtures {
        if fixture.name != "mixed_city" {
            fixture.triangle = Some(mode_result(Mode::Triangle, 10.0, 100.0));
        }
    }
    assert!(
        raw_backend_validation_reasons(&report).is_empty(),
        "{:?}",
        raw_backend_validation_reasons(&report)
    );

    report.fixtures[0].hybrid.control_hash = None;
    assert!(
        raw_backend_validation_reasons(&report)
            .iter()
            .any(|reason| reason.contains("control hash"))
    );
}

#[test]
fn suite_page_miss_total_ignores_only_the_external_product_fixture() {
    let mut report = conformance_report("cuda", "gpu");
    for fixture in &mut report.fixtures {
        if fixture.name == "mixed_city" {
            fixture.hybrid.stage.required_page_misses = None;
            fixture.reference.stage.required_page_misses = None;
            fixture.triangle = None;
        } else {
            fixture.hybrid.stage.required_page_misses = Some(0);
            fixture.reference.stage.required_page_misses = Some(0);
            if let Some(triangle) = fixture.triangle.as_mut() {
                triangle.stage.required_page_misses = Some(0);
            }
        }
    }
    assert_eq!(aggregate_required_page_misses(&report.fixtures), Some(0));
    let internal = report
        .fixtures
        .iter_mut()
        .find(|fixture| fixture.name != "mixed_city")
        .unwrap();
    internal.hybrid.stage.required_page_misses = None;
    assert_eq!(aggregate_required_page_misses(&report.fixtures), None);
}

#[test]
fn suite_correctness_ignores_only_the_external_product_fixture() {
    let mut report = conformance_report("cuda", "gpu");
    for fixture in &mut report.fixtures {
        if fixture.name == "mixed_city" {
            fixture.hybrid.correctness.hit_mismatches = None;
            fixture.reference.correctness.hit_mismatches = None;
            fixture.triangle = None;
        } else {
            fixture.triangle = Some(mode_result(Mode::Triangle, 10.0, 100.0));
        }
    }
    assert!(renderer_local_correctness_passes(&report.fixtures));
    let internal = report
        .fixtures
        .iter_mut()
        .find(|fixture| fixture.name != "mixed_city")
        .unwrap();
    internal.hybrid.correctness.hit_mismatches = None;
    assert!(!renderer_local_correctness_passes(&report.fixtures));
}

#[test]
fn suite_fallback_total_includes_triangle_and_excludes_external_placeholder() {
    let mut report = conformance_report("cuda", "gpu");
    for fixture in &mut report.fixtures {
        if fixture.name != "mixed_city" {
            fixture.triangle = Some(mode_result(Mode::Triangle, 10.0, 100.0));
        }
    }
    assert_eq!(renderer_local_unexpected_fallbacks(&report.fixtures), 0);
    let internal = report
        .fixtures
        .iter_mut()
        .find(|fixture| fixture.name != "mixed_city")
        .unwrap();
    internal.triangle.as_mut().unwrap().unexpected_fallback = true;
    assert_eq!(renderer_local_unexpected_fallbacks(&report.fixtures), 1);
    let external = report
        .fixtures
        .iter_mut()
        .find(|fixture| fixture.name == "mixed_city")
        .unwrap();
    external.hybrid.unexpected_fallback = true;
    assert_eq!(renderer_local_unexpected_fallbacks(&report.fixtures), 1);
}

#[test]
fn backend_composer_rejects_different_inputs() {
    let cuda = conformance_report("cuda", "NVIDIA");
    let mut vulkan = conformance_report("vulkan", "AMD");
    vulkan.render_config_hash = "different-config".into();
    let composed = compose_backend_conformance(&cuda, &vulkan);
    assert!(!composed.passes());
    assert!(!composed.same_inputs);
}

fn program_pass() -> ProgramBreakthrough {
    ProgramBreakthrough {
        repeated_storage_ratio: Some(0.20),
        mixed_storage_ratio: Some(0.45),
        fixed_budget_detail_ratio: Some(2.5),
        lod_temporal_invalidations_reduction: Some(0.60),
        full_frame_regression_pct: Some(1.0),
        losses: Some(0),
        axis_storage_bytes_win: Some(true),
        axis_peak_vram_win: Some(true),
        axis_geometry_stage_win: Some(true),
        axis_visible_detail_win: Some(false),
        axis_lod_temporal_win: Some(false),
    }
}

fn visibility_pass() -> VisibilityBreakthrough {
    VisibilityBreakthrough {
        ray_native_coverage: Some(0.70),
        materialized_geometry_ratio: Some(0.20),
        parameter_update_fine_as_builds: Some(0),
        geometry_stage_speedup: Some(1.8),
        fixed_budget_detail_ratio: Some(4.2),
        ray_native_temporal_invalidations: Some(0),
        correctness_mismatches: Some(0),
        real_asset_families: Some(2),
        losses: Some(0),
        procedural_intersection_win: Some(true),
        removed_triangles_counted_as_axis: Some(true),
        as_bytes_counted_as_axis: Some(false),
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

fn measured_fabric_fixtures() -> Vec<FixtureReport> {
    let mut selected = Vec::new();
    for mut fixture in conformance_report("cuda", "gpu").fixtures {
        if !REAL_ASSET_FAMILIES.contains(&fixture.family.as_str())
            || selected
                .iter()
                .any(|existing: &FixtureReport| existing.family == fixture.family)
        {
            continue;
        }
        fixture.hybrid.stage.geometry_stage_ms_p95 = 5.0;
        fixture.reference.stage.geometry_stage_ms_p95 = 15.0;
        fixture.hybrid.visibility_fabric = Some(VisibilityQueueCounters {
            domains_seeded: 2,
            rays_certified_miss: 40,
            rays_certified_hit: 0,
            domains_split: 1,
            explicit_rays: 0,
            packet_rays: 30,
            packet_active_lanes: 80,
            packet_total_lanes: 100,
            scalar_refinement_rays: 0,
            native_residual_rays: 30,
            failed_certificates: 1,
            queue_overflow_mask: 0,
            cpu_scheduling: 0,
            correctness_mismatches: 0,
            source_ray_capacity: 100,
            live_source_rays: 100,
        });
        let candidate = fixture.hybrid.visibility_profile.as_mut().unwrap();
        candidate.routing_queue_ms_p95 = 0.05;
        candidate.complete_visibility_ms_p95 = 1.0;
        candidate.includes_fixed_empty_launches = true;
        let reference = fixture.scalar_visibility_profile.as_mut().unwrap();
        reference.complete_visibility_ms_p95 = 3.0;
        reference.includes_fixed_empty_launches = true;
        selected.push(fixture);
    }
    selected
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
            scalar_visibility_profile: fr.reference.visibility_profile,
            runtime_program: None,
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
            runtime_geometry_schema: vox_data::mega_geometry::READY_MEGA_GEOMETRY_SCHEMA,
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
    assert!(
        !json.to_ascii_lowercase().contains("forge"),
        "runtime benchmark artifacts must not carry an authoring-tool revision or contract"
    );
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

    let correctness_only = evaluate_fixture(
        FixtureRole::CorrectnessOnly,
        false,
        true,
        &hybrid,
        &reference,
    );
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
        runtime_geometry_schema: vox_data::mega_geometry::READY_MEGA_GEOMETRY_SCHEMA,
    };
    assert!(cache_is_valid(&a, &a));
    let mut cap_changed = a;
    cap_changed.capability_hash = [9u8; 32];
    assert!(!cache_is_valid(&a, &cap_changed));
    let mut abi_changed = a;
    abi_changed.shader_abi = [9u8; 32];
    assert!(!cache_is_valid(&a, &abi_changed));
    let mut runtime_schema_changed = a;
    runtime_schema_changed.runtime_geometry_schema = runtime_schema_changed
        .runtime_geometry_schema
        .saturating_add(1);
    assert!(!cache_is_valid(&a, &runtime_schema_changed));

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
    b.axis_geometry_stage_win = Some(false);
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
    b.full_frame_regression_pct = Some(6.0);
    assert_eq!(b.axis_storage_bytes_win, Some(true));
    assert_eq!(b.claim(), None);

    let mut b2 = program_pass();
    b2.lod_temporal_invalidations_reduction = Some(0.10);
    assert_eq!(b2.claim(), None);
}

#[test]
fn ray_native_claim_requires_two_real_asset_families() {
    assert_eq!(visibility_pass().claim(), Some("ray_native_city_geometry"));
    let mut one_family = visibility_pass();
    one_family.real_asset_families = Some(1);
    assert_eq!(one_family.claim(), None);
}

#[test]
fn removed_triangles_and_as_bytes_are_not_double_counted_as_independent_wins() {
    assert_eq!(visibility_pass().claim(), Some("ray_native_city_geometry"));
    let mut double = visibility_pass();
    double.as_bytes_counted_as_axis = Some(true);
    assert_eq!(double.removed_triangles_counted_as_axis, Some(true));
    assert_eq!(double.as_bytes_counted_as_axis, Some(true));
    assert_eq!(double.claim(), None);
}

#[test]
fn procedural_intersection_loss_suppresses_visibility_claim_only() {
    let mut vis = visibility_pass();
    vis.procedural_intersection_win = Some(false);
    assert_eq!(vis.claim(), None);
    let overall = compute_overall(&winning_suite(), &measured_global());
    assert_eq!(overall.winner, Some(Mode::CityHybrid));
    assert_eq!(fabric_pass().claim(), Some("certified_visibility_fabric"));
}

#[test]
fn runtime_program_compositor_preserves_measured_partial_evidence() {
    let mut report = conformance_report("vulkan", "AMD");
    for fixture in &mut report.fixtures {
        if REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()) {
            fixture.hybrid.parameter_update_fine_as_builds = None;
            fixture.hybrid.temporal_reuse = None;
            fixture.reference.temporal_reuse = None;
            fixture.scalar_visibility_profile = None;
            fixture.runtime_program = Some(RuntimeProgramEvidence {
                source_triangles: 100,
                covered_triangles: 80,
                program_count: 4,
                template_count: 2,
                source_geometry_bytes: 1_000,
                covered_source_geometry_bytes: 500,
                residual_geometry_bytes: 100,
                runtime_program_bytes: 100,
            });
        }
    }

    let program = compute_program_breakthrough(&report.fixtures);
    assert_eq!(program.repeated_storage_ratio, Some(0.2));
    assert_eq!(program.mixed_storage_ratio, Some(0.2));
    assert_eq!(program.lod_temporal_invalidations_reduction, None);
    assert_eq!(program.axis_lod_temporal_win, None);
    assert_eq!(program.claim(), None);

    let visibility = compute_visibility_breakthrough(&report.fixtures);
    assert_eq!(visibility.ray_native_coverage, Some(0.8));
    assert_eq!(visibility.real_asset_families, Some(2));
    assert_eq!(visibility.parameter_update_fine_as_builds, None);
    assert_eq!(visibility.procedural_intersection_win, None);
    assert_eq!(visibility.claim(), None);
}

#[test]
fn settled_frame_cannot_authorize_temporal_reuse_claims() {
    let mut report = conformance_report("vulkan", "AMD");
    for fixture in &mut report.fixtures {
        if REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()) {
            fixture.runtime_program = Some(RuntimeProgramEvidence {
                source_triangles: 100,
                covered_triangles: 80,
                program_count: 4,
                template_count: 2,
                source_geometry_bytes: 1_000,
                covered_source_geometry_bytes: 500,
                residual_geometry_bytes: 100,
                runtime_program_bytes: 100,
            });
            let temporal = fixture
                .hybrid
                .temporal_reuse
                .as_mut()
                .expect("conformance fixture has a receipt");
            temporal.lod_cut_changes = 0;
        }
    }

    let program = compute_program_breakthrough(&report.fixtures);
    assert_eq!(program.lod_temporal_invalidations_reduction, None);
    assert_eq!(program.axis_lod_temporal_win, None);

    let visibility = compute_visibility_breakthrough(&report.fixtures);
    assert_eq!(visibility.ray_native_temporal_invalidations, None);
    assert_eq!(visibility.claim(), None);
}

#[test]
fn temporal_reduction_uses_same_transition_raw_key_oracle() {
    let mut report = conformance_report("vulkan", "AMD");
    for fixture in &mut report.fixtures {
        if REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()) {
            fixture.runtime_program = Some(RuntimeProgramEvidence {
                source_triangles: 100,
                covered_triangles: 80,
                program_count: 4,
                template_count: 2,
                source_geometry_bytes: 1_000,
                covered_source_geometry_bytes: 500,
                residual_geometry_bytes: 100,
                runtime_program_bytes: 100,
            });
        }
    }
    let program = compute_program_breakthrough(&report.fixtures);
    assert_eq!(program.lod_temporal_invalidations_reduction, Some(1.0));
    assert_eq!(program.axis_lod_temporal_win, Some(true));
}

#[test]
fn procedural_intersection_compares_matching_device_event_profiles() {
    let mut report = conformance_report("vulkan", "AMD");
    for fixture in &mut report.fixtures {
        if REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()) {
            fixture.runtime_program = Some(RuntimeProgramEvidence {
                source_triangles: 100,
                covered_triangles: 80,
                program_count: 4,
                template_count: 2,
                source_geometry_bytes: 1_000,
                covered_source_geometry_bytes: 500,
                residual_geometry_bytes: 100,
                runtime_program_bytes: 100,
            });
            fixture
                .scalar_visibility_profile
                .as_mut()
                .expect("scalar profile")
                .complete_visibility_ms_p95 = 1.0;
            fixture
                .reference
                .visibility_profile
                .as_mut()
                .expect("native profile")
                .complete_visibility_ms_p95 = 2.0;
        }
    }
    assert_eq!(
        compute_visibility_breakthrough(&report.fixtures).procedural_intersection_win,
        Some(true)
    );

    for fixture in &mut report.fixtures {
        if REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()) {
            fixture
                .scalar_visibility_profile
                .as_mut()
                .expect("scalar profile")
                .complete_visibility_ms_p95 = 3.0;
        }
    }
    assert_eq!(
        compute_visibility_breakthrough(&report.fixtures).procedural_intersection_win,
        Some(false)
    );
}

#[test]
fn unavailable_breakthrough_fields_serialize_as_unavailable_not_zero() {
    let program = ProgramBreakthrough::unavailable();
    let visibility = VisibilityBreakthrough::unavailable();
    assert!(breakthrough_line(&program).contains("repeated_storage_ratio=unavailable"));
    assert!(
        visibility_breakthrough_line(&visibility)
            .contains("parameter_update_fine_as_builds=unavailable")
    );
    assert_eq!(program.claim(), None);
    assert_eq!(visibility.claim(), None);
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
    assert_eq!(
        v.winner, None,
        "unavailable global signals must fail closed"
    );
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
    assert!(matches!(
        suite[3].outcome,
        FixtureOutcome::Unavailable { .. }
    ));
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
    assert!(HARNESS_CPU_OWNERSHIP_COUNTERS_PROBED);
}

#[test]
fn harness_proves_distinct_execution() {
    // Pin the harness constant. Public reference consumes the complete source
    // mesh through native cluster/CLAS acceleration; only city_hybrid consumes
    // the Ochroma runtime representation. Scalar ray-native calibration is a
    // separate non-reference receipt.
    assert!(
        HARNESS_CITY_HYBRID_DISTINCT_EXECUTION,
        "city_hybrid IS a distinct certified-Fabric execution path; do not flip \
         this to false unless the per-mode execution wiring is removed"
    );
}

#[test]
fn fabric_breakthrough_is_composed_from_two_measured_asset_families() {
    let result = compute_fabric_breakthrough(&measured_fabric_fixtures());
    assert_eq!(result.real_asset_families, 2);
    assert_eq!(result.losses, 0);
    assert_eq!(result.structured_ray_coverage, 0.70);
    assert_eq!(result.certified_without_per_ray_traversal, 0.40);
    assert_eq!(result.active_packet_lanes, 0.80);
    assert_eq!(result.intersection_speedup_vs_scalar, 3.0);
    assert_eq!(result.geometry_stage_speedup, 3.0);
    assert_eq!(result.routing_queue_overhead_pct, 5.0);
    assert_eq!(result.claim(), Some("certified_visibility_fabric"));
}

#[test]
fn fabric_breakthrough_fails_closed_on_missing_profile_or_timing_loss() {
    let mut missing = measured_fabric_fixtures();
    missing[0].scalar_visibility_profile = None;
    let missing_result = compute_fabric_breakthrough(&missing);
    assert!(missing_result.losses > 0);
    assert_eq!(missing_result.claim(), None);

    let mut slower = measured_fabric_fixtures();
    slower[0]
        .hybrid
        .visibility_profile
        .as_mut()
        .unwrap()
        .complete_visibility_ms_p95 = 4.0;
    let slower_result = compute_fabric_breakthrough(&slower);
    assert!(slower_result.losses > 0);
    assert_eq!(slower_result.claim(), None);
}

#[test]
fn fabric_breakthrough_rejects_inconsistent_gpu_telemetry() {
    let mut fixtures = measured_fabric_fixtures();
    let receipt = fixtures[0].hybrid.visibility_fabric.as_mut().unwrap();
    receipt.rays_certified_miss = 80;
    receipt.packet_rays = 30;
    assert_eq!(compute_fabric_breakthrough(&fixtures).claim(), None);
}

#[test]
fn hit_oracle_never_zero_from_zero_comparisons() {
    // Triangle mode is its own oracle: self-compare is zero.
    assert_eq!(
        hit_oracle_mismatches(true, true, &[0.0, 1.0], Some(&[0.0, 1.0])),
        Some(0)
    );

    // No triangle oracle present -> Unavailable, never Some(0).
    assert_eq!(
        hit_oracle_mismatches(false, false, &[0.0], Some(&[0.0])),
        None
    );
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
