//! Final three-mode City MegaGeometry benchmark: `triangle`, `reference`, and
//! `city_hybrid` over the seven required fixtures (design §6.15), emitting the
//! exact `MEGAGEOMETRY_*` evidence lines plus a JSON artifact, and printing a
//! mechanically-limited verdict.
//!
//! The verdict/schema/suppression logic lives in the pure, host-testable module
//! `support/mega_geometry_bench_logic.rs` (shared with the contract test). This
//! file is the HARDWARE HARNESS.
//!
//! ## FAIL-CLOSED
//!
//! The harness may only feed the verdict logic values it has ACTUALLY measured.
//! Every metric that needs an integration hook that does not exist yet (VRAM
//! breakdown, provenance material/UV mismatches, page-miss/deadline counters,
//! cross-backend packet/control-hash, distinct-mode execution, the breakthrough
//! evidence) is fed as `None`/`Unavailable`, which SUPPRESSES the broad claim and
//! makes the process exit nonzero. The harness therefore CANNOT print
//! `winner=city_hybrid` from synthetic data. Those hooks are listed in the
//! task's FINAL REPORT; the harness consumes them (turning `None` into real
//! measured values) once they land.
//!
//! ## Authorized run (permission-gated — do NOT run without explicit approval)
//!
//! ```powershell
//! cargo run --release -p vox_render --example mega_geometry_bench \
//!   --features spectra-native,spectra-native-optix -- \
//!   --suite city --modes triangle,reference,city_hybrid \
//!   --warmup 60 --frames 240 --json-out artifacts/megageometry/final-nvidia.json
//! ```
//!
//! Off the native stack this file compiles to a no-op `main`, so
//! `cargo check -p vox_render --example mega_geometry_bench` is clean on any host.

#![cfg_attr(not(feature = "spectra-native"), allow(dead_code))]

#[path = "support/mega_geometry_bench_logic.rs"]
mod logic;

#[cfg(feature = "spectra-native")]
#[path = "support/mega_geometry_asset_loader.rs"]
mod asset_loader;

// ---------------------------------------------------------------------------
// CLI (pure; parsed on every host)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BenchArgs {
    suite: String,
    modes: Vec<logic::Mode>,
    warmup: u32,
    frames: u32,
    seed: u64,
    json_out: Option<String>,
    corpus: Option<String>,
    res: u32,
}

impl Default for BenchArgs {
    fn default() -> Self {
        BenchArgs {
            suite: "city".into(),
            modes: all_modes(),
            warmup: logic::MIN_WARMUP_FRAMES,
            frames: 240,
            seed: 0x0C17_0A11,
            json_out: None,
            corpus: None,
            res: 1280,
        }
    }
}

fn all_modes() -> Vec<logic::Mode> {
    vec![
        logic::Mode::Triangle,
        logic::Mode::Reference,
        logic::Mode::CityHybrid,
    ]
}

fn parse_args<I: Iterator<Item = String>>(mut it: I) -> Result<BenchArgs, String> {
    let mut args = BenchArgs::default();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--suite" => args.suite = it.next().ok_or("--suite needs a value")?,
            "--modes" => {
                let raw = it.next().ok_or("--modes needs a value")?;
                let mut modes = Vec::new();
                for tok in raw.split(',') {
                    let tok = tok.trim();
                    if tok.is_empty() {
                        continue;
                    }
                    let mode =
                        logic::Mode::parse(tok).ok_or_else(|| format!("unknown mode '{tok}'"))?;
                    if !modes.contains(&mode) {
                        modes.push(mode);
                    }
                }
                if modes.is_empty() {
                    return Err("--modes selected no valid modes".into());
                }
                args.modes = modes;
            }
            "--warmup" => {
                args.warmup = it
                    .next()
                    .ok_or("--warmup needs a value")?
                    .parse()
                    .map_err(|e| format!("--warmup: {e}"))?
            }
            "--frames" => {
                args.frames = it
                    .next()
                    .ok_or("--frames needs a value")?
                    .parse()
                    .map_err(|e| format!("--frames: {e}"))?
            }
            "--seed" => {
                args.seed = it
                    .next()
                    .ok_or("--seed needs a value")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--json-out" => args.json_out = Some(it.next().ok_or("--json-out needs a value")?),
            "--corpus" => args.corpus = Some(it.next().ok_or("--corpus needs a value")?),
            "--res" => {
                args.res = it
                    .next()
                    .ok_or("--res needs a value")?
                    .parse()
                    .map_err(|e| format!("--res: {e}"))?
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// Off-native build: no-op main. Still validates the CLI + prints the plan.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "spectra-native"))]
fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(argv.into_iter()) {
        Ok(args) => {
            eprintln!(
                "mega_geometry_bench requires --features spectra-native (native RT stack).\n\
                 This host build only validates the fixture plan + evidence schema.\n\
                 suite={} modes={:?} warmup={} frames={}",
                args.suite,
                args.modes.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
                args.warmup,
                args.frames,
            );
            for f in logic::required_fixtures() {
                eprintln!(
                    "  fixture {} role={:?} asset={} family={} instances={} identity_hash={}",
                    f.name,
                    f.role,
                    f.asset_id,
                    f.family,
                    f.instances,
                    f.identity_hash_hex()
                );
            }
        }
        Err(e) => {
            eprintln!("mega_geometry_bench: argument error: {e}");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Native build: the real hardware harness.
// ---------------------------------------------------------------------------

#[cfg(feature = "spectra-native")]
fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(argv.into_iter()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("mega_geometry_bench: argument error: {e}");
            std::process::exit(2);
        }
    };
    match native::run(&args) {
        Ok(exit) => std::process::exit(exit),
        Err(e) => {
            eprintln!("mega_geometry_bench: fatal: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "spectra-native")]
mod native {
    use super::asset_loader;
    use super::logic;
    use super::BenchArgs;
    use std::time::Instant;
    use vox_render::resident_renderer::ResidentSceneRenderer;
    use vox_render::splat_backend::{BlasDesc, InstanceRecordGpu, LightRig, PbrMaterial};
    use vox_render::splat_convert::meshes_to_instanced_scene;

    /// The RT backend the enabled build feature requests (compile-time truth).
    fn requested_rt_backend() -> &'static str {
        if cfg!(feature = "spectra-native-optix") {
            "optix"
        } else {
            "vulkan_khr"
        }
    }

    /// Routed through the pinned logic constant so a stray flip is caught by a
    /// test. `false` today (see `logic::HARNESS_CITY_HYBRID_DISTINCT_EXECUTION`).
    const CITY_HYBRID_IS_DISTINCT_EXECUTION: bool = logic::HARNESS_CITY_HYBRID_DISTINCT_EXECUTION;

    /// Per-pixel first-hit provenance of the last timed frame:
    /// (prim_id, material_id, bary_u, bary_v, depth_t). The cross-mode
    /// correctness oracle (triangle-GAS is the reference; CLAS modes compared
    /// against it, with depth_t used to gate measure-zero edge-face ties).
    type ProvenanceFrame = (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>);

    pub fn run(args: &BenchArgs) -> Result<i32, String> {
        let width = args.res;
        let height = (args.res * 9 / 16).max(1);
        let timed_cfg = logic::TimedRunConfig {
            release_build: !cfg!(debug_assertions),
            warmup_frames: args.warmup,
            timed_frames: args.frames,
            spp: 1,
            identical_present_path: true,
        };
        timed_cfg.validate()?;

        if !args.modes.contains(&logic::Mode::Triangle) {
            eprintln!(
                "[mega_geometry_bench] triangle oracle mode absent from --modes; correctness \
                 cannot be verified and every fixture will be Unavailable."
            );
        }
        if !CITY_HYBRID_IS_DISTINCT_EXECUTION {
            eprintln!(
                "[mega_geometry_bench] city_hybrid is not yet a distinct execution path from \
                 reference (MEGAGEOMETRY_MODE is unconsumed by the renderer). Refusing to award a \
                 win; target fixtures report Unavailable and the process exits nonzero."
            );
        }

        let corpus_hash = match &args.corpus {
            Some(path) => {
                let bytes =
                    std::fs::read(path).map_err(|e| format!("corpus '{path}' unavailable: {e}"))?;
                format!("{:016x}", logic::fnv1a(&bytes))
            }
            None => "unbound".to_string(),
        };

        let fixtures = logic::required_fixtures();
        // Diagnostic-only fixture narrowing (`MEGAGEOMETRY_ONLY=name[,name]`).
        // Running fewer than the seven required fixtures ALWAYS suppresses the
        // overall verdict (`compute_overall` requires the full set), so this can
        // only make a run stricter/faster — never fabricate a win. Used to
        // witness one fixture without paying the 122 MB meridian parse + full
        // suite time. Absent/empty ⇒ the full required set.
        let only: Vec<String> = std::env::var("MEGAGEOMETRY_ONLY")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let selected: Vec<&logic::FixtureSpec> = if only.is_empty() {
            fixtures.iter().collect()
        } else {
            eprintln!("[mega_geometry_bench] MEGAGEOMETRY_ONLY active: {only:?} (verdict WILL be suppressed — diagnostic run)");
            fixtures
                .iter()
                .filter(|f| only.iter().any(|o| o == &f.name))
                .collect()
        };
        let mut fixture_reports = Vec::new();
        for spec in selected {
            fixture_reports.push(run_fixture(spec, args, width, height, &timed_cfg)?);
        }

        let global = collect_global_signals(&fixture_reports);
        let fr_for_verdict: Vec<logic::FixtureResult> = fixture_reports
            .iter()
            .map(|r| logic::FixtureResult {
                spec: spec_of(r),
                outcome: r.outcome.clone(),
                hybrid: r.hybrid.clone(),
                reference: r.reference.clone(),
            })
            .collect();
        let overall = logic::compute_overall(&fr_for_verdict, &global);

        // Breakthroughs default to their fully-unavailable (failing) state until
        // the renderer exposes the corresponding evidence accessors.
        let program_bt = logic::ProgramBreakthrough::unavailable();
        let visibility_bt = logic::VisibilityBreakthrough::unavailable();
        let fabric_bt = logic::FabricBreakthrough::unavailable();

        // ---- Print the evidence ----
        let env = collect_env_line();
        println!("{}", env.line());
        for r in &fixture_reports {
            emit_fixture_evidence(r, args.frames);
        }
        println!(
            "MEGAGEOMETRY_SUITE correctness={} unexpected_fallbacks={} required_page_misses=unavailable",
            suite_correctness(&fixture_reports),
            fixture_reports
                .iter()
                .filter(|r| r.hybrid.unexpected_fallback || r.reference.unexpected_fallback)
                .count(),
        );
        println!("{}", logic::portable_line(&global, None));
        for r in &fixture_reports {
            println!("{}", logic::fixture_line(&spec_of(r), &r.outcome));
        }
        println!("{}", logic::overall_line(&overall));
        println!("{}", logic::breakthrough_line(&program_bt));
        println!("{}", logic::visibility_breakthrough_line(&visibility_bt));
        println!("{}", logic::fabric_breakthrough_line(&fabric_bt));
        if let Some(reason) = &overall.suppressed_reason {
            eprintln!("[mega_geometry_bench] verdict suppressed: {reason}");
        }

        // ---- JSON artifact ----
        let report = logic::BenchReport {
            env,
            suite: args.suite.clone(),
            modes: args.modes.iter().map(|m| m.as_str().to_string()).collect(),
            warmup_frames: args.warmup,
            timed_frames: args.frames,
            seed: args.seed,
            command_line: std::env::args().collect::<Vec<_>>().join(" "),
            git_revisions: collect_git_revisions(),
            render_config_hash: render_config_hash(),
            corpus_hash,
            fixtures: fixture_reports,
            global,
            overall: overall.clone(),
            program_breakthrough: program_bt.claim().map(|_| program_bt),
            visibility_breakthrough: visibility_bt.claim().map(|_| visibility_bt),
            fabric_breakthrough: fabric_bt.claim().map(|_| fabric_bt),
        };
        if let Some(path) = &args.json_out {
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(path, report.to_json()?).map_err(|e| format!("write {path}: {e}"))?;
            eprintln!("[mega_geometry_bench] wrote {path}");
        }

        // Exit nonzero on any hard, fail-closed abort (unavailable evidence,
        // correctness failure, silent downgrade, CPU/portable/cache breach). A
        // purely measured performance loss exits 0 with a narrower claim.
        Ok(if overall.hard_abort { 1 } else { 0 })
    }

    fn spec_of(r: &logic::FixtureReport) -> logic::FixtureSpec {
        logic::required_fixtures()
            .into_iter()
            .find(|f| f.name == r.name)
            .expect("fixture report name must be a required fixture")
    }

    fn suite_correctness(reports: &[logic::FixtureReport]) -> &'static str {
        let all_verified = reports.iter().all(|r| {
            r.hybrid.correctness.is_exact() == Some(true)
                && r.reference.correctness.is_exact() == Some(true)
        });
        if all_verified {
            "pass"
        } else {
            "unavailable"
        }
    }

    fn run_fixture(
        spec: &logic::FixtureSpec,
        args: &BenchArgs,
        width: u32,
        height: u32,
        cfg: &logic::TimedRunConfig,
    ) -> Result<logic::FixtureReport, String> {
        let mut triangle: Option<(logic::ModeResult, ProvenanceFrame)> = None;
        let mut reference: Option<logic::ModeResult> = None;
        let mut hybrid: Option<logic::ModeResult> = None;

        for &mode in &[
            logic::Mode::Triangle,
            logic::Mode::Reference,
            logic::Mode::CityHybrid,
        ] {
            if !args.modes.contains(&mode) {
                continue;
            }
            let (mr, prov) = measure_mode(spec, mode, width, height, cfg)?;
            // Cross-mode correctness oracle: triangle-GAS provenance is the
            // reference; CLAS modes are compared against it. Fills
            // material/uv/hit mismatches (all None => Unavailable, fail closed).
            let oracle = triangle.as_ref().map(|(_, (p, m, u, v, d))| {
                (
                    p.as_slice(),
                    m.as_slice(),
                    u.as_slice(),
                    v.as_slice(),
                    d.as_slice(),
                )
            });
            let mr = apply_provenance_oracle(mr, &prov, oracle);
            match mode {
                logic::Mode::Triangle => triangle = Some((mr, prov)),
                logic::Mode::Reference => reference = Some(mr),
                logic::Mode::CityHybrid => hybrid = Some(mr),
            }
        }

        let hybrid = hybrid.ok_or("city_hybrid mode not measured")?;
        let reference = reference
            .clone()
            .or_else(|| triangle.as_ref().map(|(m, _)| m.clone()))
            .ok_or("reference mode not measured")?;

        let outcome = logic::evaluate_fixture(
            spec.role,
            spec.authorizing_witness_external,
            CITY_HYBRID_IS_DISTINCT_EXECUTION,
            &hybrid,
            &reference,
        );

        Ok(logic::FixtureReport {
            name: spec.name.clone(),
            role: spec.role,
            asset_id: spec.asset_id.clone(),
            family: spec.family.clone(),
            instances: spec.instances,
            seed: spec.seed,
            identity_hash: spec.identity_hash_hex(),
            // The REAL cooked asset's content hash (cache hit — the asset was
            // loaded in measure_mode). `None` only for ids with no cooked file
            // (e.g. the externally-authorized mixed_city), which keeps that
            // fixture honestly uncertified by vox_render.
            asset_content_hash: asset_loader::load_fixture_asset(&spec.asset_id, spec.seed)
                .ok()
                .map(|a| a.content_hash.clone()),
            outcome,
            hybrid,
            reference,
            triangle: triangle.map(|(m, _)| m),
        })
    }

    fn measure_mode(
        spec: &logic::FixtureSpec,
        mode: logic::Mode,
        width: u32,
        height: u32,
        cfg: &logic::TimedRunConfig,
    ) -> Result<(logic::ModeResult, ProvenanceFrame), String> {
        apply_mode_env(mode);

        // Load the REAL Forge-cooked asset for this fixture (cached across modes).
        // Fall back to the synthetic cube only for ids with no cooked file — e.g.
        // the externally-authorized `mixed_city` (urban_horizon.live_city).
        let asset = asset_loader::load_fixture_asset(&spec.asset_id, spec.seed).ok();
        let cube_fallback = if asset.is_none() {
            eprintln!(
                "[mega_geometry_bench] {}: real asset '{}' unavailable; using synthetic cube proxy",
                spec.name, spec.asset_id
            );
            Some(fixture_prototype())
        } else {
            None
        };
        // city_hybrid traces the MegaGeometry-covered triangles ray-native via
        // the visibility programs, so its RESIDUAL mesh must exclude them — else
        // the covered surface is double-represented (mesh + realized) and z-fights
        // (§ `CookedAsset::base_without_covered`). triangle/reference keep the full
        // mesh (the oracle compares mega-reconstructed vs full-detail geometry).
        let city_residual: Option<BlasDesc> = match (&asset, mode) {
            (Some(a), logic::Mode::CityHybrid) => Some(a.base_without_covered()),
            _ => None,
        };
        let (blas, materials): (&BlasDesc, &[PbrMaterial]) = match (&asset, &cube_fallback) {
            (Some(a), _) => (
                city_residual.as_ref().unwrap_or(&a.base),
                a.materials.as_slice(),
            ),
            (None, Some((b, m))) => (b, m.as_slice()),
            _ => unreachable!(),
        };
        let instances = grid_instances(spec.instances as usize);
        let mut scene = meshes_to_instanced_scene(
            std::slice::from_ref(blas),
            &instances,
            materials,
            &[],
            width,
            height,
        );

        // city_hybrid traces the buildings RAY-NATIVE via the cooked MegaGeometry
        // visibility programs — THE actual MegaGeometry, not mesh/cluster LOD.
        // Populate SceneState.mega_geometry from the loaded ReadyMegaGeometry so
        // `use_mega_geometry` engages (mega_tlas + programs) and the megakernel
        // executes the programs for the covered triangles instead of tracing
        // them as materialized mesh. triangle/reference keep the full-detail
        // mesh, so the correctness oracle compares ray-native vs materialized.
        if matches!(mode, logic::Mode::CityHybrid) {
            if let Some(a) = &asset {
                if let Some(payload) = &a.mega {
                    let transforms: Vec<[f32; 16]> =
                        instances.iter().map(|i| i.transform).collect();
                    scene.mega_geometry = asset_loader::assemble_mega_layer(
                        payload,
                        &transforms,
                        materials.len() as u32,
                    );
                }
            }
        }

        let mut r = ResidentSceneRenderer::new(width, height, LightRig::default(), 1, 1, scene)
            .map_err(|e| format!("{}/{}: renderer build: {e}", spec.name, mode.as_str()))?;
        let (view, proj) = camera(spec.instances as usize, width, height);

        for _ in 0..cfg.warmup_frames {
            r.render_only(view, proj)
                .map_err(|e| format!("{}/{}: warmup: {e}", spec.name, mode.as_str()))?;
        }

        let mut update_samples: Vec<f64> = Vec::with_capacity(cfg.timed_frames as usize);
        let mut trace_samples: Vec<f64> = Vec::with_capacity(cfg.timed_frames as usize);
        let mut frame_samples: Vec<f64> = Vec::with_capacity(cfg.timed_frames as usize);
        let mut dirty: Vec<(usize, [[f32; 4]; 3])> = Vec::with_capacity(spec.instances as usize);
        // First-hit provenance of the last timed frame: (prim_id, material_id,
        // bary_u, bary_v), one entry per pixel. The cross-mode correctness oracle.
        let mut last_prov: ProvenanceFrame =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

        let is_changing = matches!(
            spec.name.as_str(),
            "changing_geometry" | "localized_updates" | "streamed_lod"
        );

        for frame in 0..cfg.timed_frames {
            let mut update_ms = 0.0;
            if is_changing {
                dirty.clear();
                perturb(spec, &mut dirty, frame);
                let t = Instant::now();
                r.refit_instances(&dirty)
                    .map_err(|e| format!("{}/{}: refit: {e}", spec.name, mode.as_str()))?;
                update_ms = t.elapsed().as_secs_f64() * 1000.0;
            }
            let t = Instant::now();
            let out = r
                .render_only(view, proj)
                .map_err(|e| format!("{}/{}: render: {e}", spec.name, mode.as_str()))?;
            let render_ms = t.elapsed().as_secs_f64() * 1000.0;
            update_samples.push(update_ms);
            trace_samples.push(render_ms);
            frame_samples.push(update_ms + render_ms);
            if frame + 1 == cfg.timed_frames {
                // Capture the first-hit provenance AOV of the LAST timed frame for
                // the cross-mode correctness oracle. Empty (fail-closed) if the
                // renderer did not produce it.
                if let Some(p) = out.first_hit_provenance {
                    last_prov = (p.prim_id, p.material_id, p.bary_u, p.bary_v, p.depth);
                }
            }
        }

        // Detailed device stats copied only AFTER the timed frames.
        let stats = r.geometry_accel_stats().map_err(|e| {
            format!("{}/{}: no acceleration report: {e:?}", spec.name, mode.as_str())
        })?;
        let cpu = map_cpu_counters(&r.geometry_cpu_ownership());
        let backend = resolve_backend(mode, &stats);

        let mr = logic::ModeResult {
            mode,
            compute: logic::ComputeVariant::PortableSubgroup,
            correctness: logic::Correctness {
                source_primitives: stats.source_primitive_count(),
                mapped_primitives: stats.mapped_primitive_count(),
                // Start Unavailable (fail closed); all three are filled by
                // apply_provenance_oracle from the real cross-mode first-hit
                // provenance compare against the triangle-GAS oracle.
                material_mismatches: None,
                uv_mismatches: None,
                hit_mismatches: None,
            },
            stage: logic::GeometryStage {
                build_ms_p50: p50(&update_samples),
                build_ms_p95: p95(&update_samples),
                update_ms_p50: p50(&update_samples),
                update_ms_p95: p95(&update_samples),
                native_lowering_ms_p95: None,
                trace_ms_p50: p50(&trace_samples),
                trace_ms_p95: p95(&trace_samples),
                geometry_stage_ms_p95: p95(&update_samples) + p95(&trace_samples),
                frame_ms_p50: p50(&frame_samples),
                frame_ms_p95: p95(&frame_samples),
                traced_triangles: stats.materialized_triangle_count(),
                // No pager hook yet -> unavailable (fail closed).
                required_page_misses: None,
            },
            vram: logic::VramBreakdown {
                // Resident GpuInstance buffer: 32 u32 words (GPU_INSTANCE_BYTES=128)
                // per instance — exact.
                resident_mb: (spec.instances as f64 * 128.0) / (1024.0 * 1024.0),
                // Acceleration-structure VRAM (IAS + per-proto GAS + CLAS + OMM
                // output buffers) — the real retained AS bytes and the component
                // that DIFFERS between modes (city_hybrid's hierarchical IAS adds
                // levels vs reference's flat IAS).
                as_mb: Some(stats.bytes() as f64 / (1024.0 * 1024.0)),
                // No geometry pager is instantiated in the bench -> 0 page-pool VRAM.
                page_mb: Some(0.0),
                // Native-lowering argument buffers are transient (allocated during
                // the build, freed before the after-frames stats snapshot).
                native_args_mb: Some(0.0),
                // Build scratch is freed post-build; the small retained update
                // scratch is excluded EQUALLY from both modes, so the peak-VRAM
                // comparison direction (driven by as_mb) is unchanged.
                scratch_mb: Some(0.0),
            },
            cpu,
            // No prewarm accessor proving no in-frame pipeline creation yet.
            prewarm: None,
            // The harness structurally collects stats AFTER the timed loop and
            // never inside present timing; that IS measured.
            stats: Some(logic::StatsCollection {
                collected_after_timed_frames: true,
                inside_present_timing: false,
            }),
            backend,
            unexpected_fallback: stats.fallback_reason().is_some()
                && !matches!(mode, logic::Mode::Triangle),
        };
        Ok((mr, last_prov))
    }

    /// Fill `hit_mismatches`, `material_mismatches`, and `uv_mismatches` from the
    /// real cross-mode FIRST-HIT PROVENANCE compare via the shared, unit-tested
    /// `logic::provenance_mismatches`. This replaces the old beauty-frame compare
    /// (which conflated shading/AA noise with geometry divergence): the primary
    /// hit's GLOBAL prim_id / material_id / barycentrics are compared per pixel
    /// against the triangle-GAS oracle, so a mismatch is a genuine geometry or
    /// material divergence. All three stay `None` (Unavailable, fail closed) when
    /// there is no oracle or the buffers are empty.
    fn apply_provenance_oracle(
        mut mr: logic::ModeResult,
        prov: &ProvenanceFrame,
        oracle: Option<(&[u32], &[u32], &[f32], &[f32], &[f32])>,
    ) -> logic::ModeResult {
        let (p, m, u, v, d) = prov;
        let (hit, mat, uv) = logic::provenance_mismatches(
            matches!(mr.mode, logic::Mode::Triangle),
            oracle,
            (
                p.as_slice(),
                m.as_slice(),
                u.as_slice(),
                v.as_slice(),
                d.as_slice(),
            ),
        );
        mr.correctness.hit_mismatches = hit;
        mr.correctness.material_mismatches = mat;
        mr.correctness.uv_mismatches = uv;
        if std::env::var("MEGAGEOMETRY_DIAG").is_ok() && !matches!(mr.mode, logic::Mode::Triangle) {
            if let Some((op, om, _ou, _ov, od)) = oracle {
                let n = p.len().min(op.len());
                let mut shown = 0;
                for i in 0..n {
                    let o_hit = op[i] != logic::PROV_MISS_PRIM;
                    let c_hit = p[i] != logic::PROV_MISS_PRIM;
                    let same_surface =
                        (d[i] - od[i]).abs() <= logic::PROV_DEPTH_REL_TOL * od[i].abs().max(1.0);
                    let is_hit_mismatch =
                        o_hit != c_hit || (o_hit && c_hit && op[i] != p[i] && !same_surface);
                    if is_hit_mismatch && shown < 24 {
                        eprintln!(
                            "[DIAG hit] px={i} o_hit={o_hit} c_hit={c_hit} prim(o={} c={}) mat(o={} c={}) depth(o={:.5} c={:.5} d={:.5})",
                            op[i], p[i], om[i], m[i], od[i], d[i], (d[i]-od[i]).abs()
                        );
                        shown += 1;
                    }
                }
                eprintln!("[DIAG hit] mode={} shown={shown}", mr.mode.as_str());
            }
        }
        mr
    }

    fn map_cpu_counters(
        snap: &vox_render::mega_geometry::GeometryCpuOwnershipSnapshot,
    ) -> logic::CpuCounters {
        use vox_render::mega_geometry::{
            ForbiddenGeometryCpuActivity as F, GeometryHostServiceKind,
        };
        let host_calls =
            snap.host_service_calls(GeometryHostServiceKind::NativeAccelerationSubmission);
        let host_ns = snap.host_service_ns(GeometryHostServiceKind::NativeAccelerationSubmission);
        logic::CpuCounters {
            render_thread_scene_visits: snap.forbidden_count(F::RenderThreadInstanceVisit)
                + snap.forbidden_count(F::RenderThreadPrototypeVisit)
                + snap.forbidden_count(F::RenderThreadClusterVisit)
                + snap.forbidden_count(F::RenderThreadPageVisit),
            visibility_node_visits: snap.forbidden_count(F::RenderThreadVisibilityNodeVisit),
            construction_evaluations: snap.forbidden_count(F::ConstructionEvaluation),
            lod_decisions: snap.forbidden_count(F::LodOrToleranceDecision),
            page_decisions: snap.forbidden_count(F::PagePriorityDecision),
            eviction_decisions: snap.forbidden_count(F::EvictionDecision),
            blocking_readbacks: snap.forbidden_count(F::BlockingReadback),
            frame_allocations: snap.forbidden_count(F::FrameAllocation),
            // The forbidden counters are NOT yet bridged from Spectra's probe
            // into vox_render's snapshot, so their zeros are not evidence.
            counters_probed: super::logic::HARNESS_CPU_OWNERSHIP_COUNTERS_PROBED,
            // No deadline-miss counter exposed yet -> unavailable (fail closed).
            host_deadline_misses: None,
            // No render-thread submit-time accessor yet -> unavailable.
            render_thread_submit_ms_p95: None,
            host_submit_calls: host_calls,
            // Zero recorded host-submit calls is "nothing measured" -> None
            // (Unavailable), never a passing 0.0. Otherwise the mean is a p95
            // proxy until a real percentile accessor exists.
            host_submit_ms_p95: if host_calls > 0 {
                Some(host_ns as f64 / host_calls as f64 / 1.0e6)
            } else {
                None
            },
        }
    }

    fn resolve_backend(
        mode: logic::Mode,
        stats: &spectra_renderer::renderer::geometry_backend::GeometryAccelStats,
    ) -> logic::BackendResolution {
        use spectra_renderer::renderer::geometry_backend::RayAccelBackendKind;
        let actual_backend = match stats.backend() {
            RayAccelBackendKind::Optix => "optix",
            RayAccelBackendKind::VulkanKhr => "vulkan_khr",
            RayAccelBackendKind::VulkanNv => "vulkan_nv",
            RayAccelBackendKind::Metal => "metal",
            RayAccelBackendKind::Dxr => "dxr",
            RayAccelBackendKind::Unavailable => "unavailable",
        };
        let requested_cluster = matches!(mode, logic::Mode::Reference | logic::Mode::CityHybrid);
        // A cluster/CLAS representation requires HARDWARE CLAS builds. A derived
        // cluster count with zero hardware builds is a silent downgrade.
        let cluster_requested_without_hardware_clas =
            requested_cluster && stats.hardware_clas_builds() == 0;
        let available = !matches!(stats.backend(), RayAccelBackendKind::Unavailable);
        logic::BackendResolution {
            requested_rt_backend: requested_rt_backend().to_string(),
            actual_rt_backend: actual_backend.to_string(),
            requested_compute: logic::ComputeVariant::PortableSubgroup,
            actual_compute: logic::ComputeVariant::PortableSubgroup,
            available,
            silently_downgraded: false,
            cpu_fallbacks: 0,
            cluster_requested_without_hardware_clas,
        }
    }

    fn emit_fixture_evidence(r: &logic::FixtureReport, frames: u32) {
        let spec = spec_of(r);
        for m in [&r.hybrid, &r.reference] {
            println!("{}", logic::config_line(m.mode, m.compute, &spec, frames));
            println!("{}", logic::correct_line(&m.correctness));
            println!("{}", logic::result_line(&m.stage, &m.vram));
            println!("{}", logic::cpu_line(&m.cpu));
            // Program / visibility / fabric / temporal evidence require renderer
            // hooks that do not exist yet -> emit UNAVAILABLE lines (fail closed).
            println!("{}", logic::ProgramLine::unavailable(m.mode.as_str()).line());
            println!(
                "{}",
                logic::VisibilityLine::unavailable(m.mode.as_str()).line()
            );
            println!("{}", logic::FabricLine::unavailable(m.mode.as_str()).line());
        }
        println!("{}", logic::TemporalLine::unavailable().line());
    }

    fn collect_global_signals(fixtures: &[logic::FixtureReport]) -> logic::GlobalSignals {
        // Two scale samples (smallest and largest fixture) with the REAL forbidden
        // CPU counters for the scaling proof. Every other global signal is
        // UNAVAILABLE (fail closed) until its cross-backend/hook exists.
        let mut small = &fixtures[0];
        let mut large = &fixtures[0];
        for f in fixtures {
            if f.instances < small.instances {
                small = f;
            }
            if f.instances > large.instances {
                large = f;
            }
        }
        logic::GlobalSignals::unavailable(
            logic::ScaleSample {
                total_units: small.instances,
                cpu: small.hybrid.cpu,
            },
            logic::ScaleSample {
                total_units: large.instances,
                cpu: large.hybrid.cpu,
            },
        )
    }

    fn collect_env_line() -> logic::EnvLine {
        logic::EnvLine {
            gpu: std::env::var("MEGAGEOMETRY_GPU").unwrap_or_else(|_| "unknown".into()),
            api: if cfg!(feature = "spectra-native-optix") {
                "cuda"
            } else {
                "vulkan"
            }
            .to_string(),
            driver: std::env::var("MEGAGEOMETRY_DRIVER").unwrap_or_else(|_| "unknown".into()),
            rt_backend: requested_rt_backend().to_string(),
            capability_hash: std::env::var("MEGAGEOMETRY_CAP_HASH")
                .unwrap_or_else(|_| "unavailable".into()),
            shader_abi: std::env::var("MEGAGEOMETRY_SHADER_ABI")
                .unwrap_or_else(|_| "unavailable".into()),
            cook_schema: 3,
        }
    }

    fn collect_git_revisions() -> logic::GitRevisions {
        logic::GitRevisions {
            ochroma: git_rev(env!("CARGO_MANIFEST_DIR")),
            spectra: std::env::var("MEGAGEOMETRY_REV_SPECTRA").unwrap_or_else(|_| "unknown".into()),
            forge: std::env::var("MEGAGEOMETRY_REV_FORGE").unwrap_or_else(|_| "unknown".into()),
            urban: std::env::var("MEGAGEOMETRY_REV_URBAN").unwrap_or_else(|_| "unknown".into()),
        }
    }

    fn git_rev(dir: &str) -> String {
        std::process::Command::new("git")
            .args(["-C", dir, "rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    fn render_config_hash() -> String {
        std::env::var("MEGAGEOMETRY_RENDER_CONFIG_HASH").unwrap_or_else(|_| "unavailable".into())
    }

    /// Mode selection knobs. NOTE: `MEGAGEOMETRY_MODE` is not consumed by the
    /// renderer yet, so these do NOT (today) produce distinct execution paths —
    /// which is exactly why `CITY_HYBRID_IS_DISTINCT_EXECUTION` is false and the
    /// harness fails closed. When the policy/IAS integration honors these knobs,
    /// flip that constant and the win becomes provable.
    fn apply_mode_env(mode: logic::Mode) {
        // SAFETY: single-threaded harness; env read by the renderer on next build.
        unsafe {
            // Correctness oracle: have the renderer read back the per-pixel
            // first-hit provenance AOV (prim_id/material/bary) each frame so the
            // harness can compare it cross-mode. Set for every mode.
            std::env::set_var("SPECTRA_PROVENANCE_AOV", "1");
            match mode {
                logic::Mode::Triangle => {
                    std::env::set_var("SPECTRA_CLAS_THRESHOLD", "18446744073709551615");
                    std::env::set_var("MEGAGEOMETRY_MODE", "triangle");
                }
                logic::Mode::Reference => {
                    std::env::set_var("SPECTRA_CLAS_THRESHOLD", "1");
                    std::env::set_var("MEGAGEOMETRY_MODE", "reference");
                }
                logic::Mode::CityHybrid => {
                    std::env::set_var("SPECTRA_CLAS_THRESHOLD", "1");
                    std::env::set_var("MEGAGEOMETRY_MODE", "city_hybrid");
                }
            }
        }
    }

    // ---- synthetic proxy geometry (relative GPU timing only, NOT authorizing) ----

    fn fixture_prototype() -> (BlasDesc, [PbrMaterial; 6]) {
        let materials = [
            [0.42, 0.44, 0.45],
            [0.12, 0.13, 0.14],
            [0.66, 0.63, 0.57],
            [0.82, 0.86, 0.90],
            [0.18, 0.11, 0.07],
            [0.23, 0.25, 0.28],
        ]
        .map(|base_color| PbrMaterial {
            base_color,
            roughness: 0.55,
            ..Default::default()
        });
        (unit_cube_blas(), materials)
    }

    fn unit_cube_blas() -> BlasDesc {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        let mut material_ids: Vec<u32> = Vec::new();
        let h = 0.5f32;
        let corners = [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ];
        let faces: [([usize; 4], [f32; 3]); 6] = [
            ([0, 1, 2, 3], [0.0, 0.0, -1.0]),
            ([5, 4, 7, 6], [0.0, 0.0, 1.0]),
            ([4, 0, 3, 7], [-1.0, 0.0, 0.0]),
            ([1, 5, 6, 2], [1.0, 0.0, 0.0]),
            ([3, 2, 6, 7], [0.0, 1.0, 0.0]),
            ([4, 5, 1, 0], [0.0, -1.0, 0.0]),
        ];
        for (quad, n) in faces {
            let base = positions.len() as u32;
            for &ci in quad.iter() {
                positions.push(corners[ci]);
                normals.push(n);
                uvs.push([0.0, 0.0]);
            }
            indices.push([base, base + 1, base + 2]);
            indices.push([base, base + 2, base + 3]);
            let material = (material_ids.len() / 2) as u32;
            material_ids.push(material);
            material_ids.push(material);
        }
        BlasDesc {
            proto_id: 1,
            positions,
            normals,
            uvs,
            indices,
            material_ids,
            aabb_min: [-h, -h, -h],
            aabb_max: [h, h, h],
            weathering_masks: Vec::new(),
        }
    }

    fn grid_instances(n: usize) -> Vec<InstanceRecordGpu> {
        let cols = ((n as f64).sqrt().ceil() as usize).max(1);
        let spacing = 2.0f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let gx = (i % cols) as f32;
            let gz = (i / cols) as f32;
            let x = (gx - cols as f32 * 0.5) * spacing;
            let z = (gz - (n as f32 / cols as f32) * 0.5) * spacing;
            out.push(InstanceRecordGpu {
                proto_index: 0,
                transform: [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, 0.0, z, 1.0,
                ],
                material_base: 0,
            });
        }
        out
    }

    fn perturb(spec: &logic::FixtureSpec, dirty: &mut Vec<(usize, [[f32; 4]; 3])>, frame: u32) {
        let n = spec.instances as usize;
        let cols = ((n as f64).sqrt().ceil() as usize).max(1);
        let spacing = 2.0f32;
        let bob = ((frame % 16) as f32) * 0.05 - 0.4;
        let localized = spec.name == "localized_updates";
        let count = if localized { (n / 100).max(1) } else { n };
        for j in 0..count {
            let i = if localized {
                (frame as usize * count + j) % n
            } else {
                j
            };
            let gx = (i % cols) as f32;
            let gz = (i / cols) as f32;
            let x = (gx - cols as f32 * 0.5) * spacing;
            let z = (gz - (n as f32 / cols as f32) * 0.5) * spacing;
            dirty.push((
                i,
                [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, bob], [0.0, 0.0, 1.0, z]],
            ));
        }
    }

    fn camera(n: usize, width: u32, height: u32) -> ([f32; 16], [f32; 16]) {
        let cols = (n as f64).sqrt().ceil() as f32;
        let extent = cols.max(1.0) * 2.0;
        let dist = extent * 1.2 + 5.0;
        let eye = glam::Vec3::new(0.0, dist * 0.6, dist);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y).to_cols_array();
        let aspect = width as f32 / height.max(1) as f32;
        let proj = glam::Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, dist * 4.0 + 100.0)
            .to_cols_array();
        (view, proj)
    }

    fn p50(samples: &[f64]) -> f64 {
        percentile(samples, 0.50)
    }
    fn p95(samples: &[f64]) -> f64 {
        percentile(samples, 0.95)
    }
    fn percentile(samples: &[f64], q: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut v = samples.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
        v[idx.min(v.len() - 1)]
    }
}
