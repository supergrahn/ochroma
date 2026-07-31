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
    use super::BenchArgs;
    use super::asset_loader;
    use super::logic;
    use sha2::{Digest as _, Sha256};
    use std::collections::BTreeSet;
    use std::time::Instant;
    use vox_render::resident_renderer::ResidentSceneRenderer;
    use vox_render::splat_backend::{BlasDesc, InstanceRecordGpu, LightRig, PbrMaterial};
    use vox_render::splat_convert::meshes_to_instanced_scene;

    fn hex32(bytes: [u8; 32]) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(64);
        for byte in bytes {
            let _ = write!(&mut out, "{byte:02x}");
        }
        out
    }

    fn finished_object_corpus_hash(fixtures: &[logic::FixtureReport]) -> String {
        let mut records = fixtures
            .iter()
            .map(|fixture| {
                format!(
                    "{}\0{}\0{}\0{}",
                    fixture.name,
                    fixture.asset_id,
                    fixture.identity_hash,
                    fixture.asset_content_hash.as_deref().unwrap_or("external")
                )
            })
            .collect::<Vec<_>>();
        records.sort();
        hex32(Sha256::digest(records.join("\n").as_bytes()).into())
    }

    /// The RT backend the enabled build feature requests (compile-time truth).
    fn requested_rt_backend() -> &'static str {
        if cfg!(feature = "spectra-native-optix") {
            "optix"
        } else {
            "vulkan_khr"
        }
    }

    /// Routed through the pinned logic constant so a stray flip is caught by a
    /// test. The harness must also construct genuinely different scene state
    /// below; this constant is not accepted as evidence by itself.
    const CITY_HYBRID_IS_DISTINCT_EXECUTION: bool = logic::HARNESS_CITY_HYBRID_DISTINCT_EXECUTION;

    /// Per-pixel first-hit provenance of the last timed frame:
    /// (prim_id, material_id, bary_u, bary_v, depth_t). The cross-mode
    /// correctness oracle (triangle-GAS is the reference; CLAS modes compared
    /// against it, with depth_t used to gate measure-zero edge-face ties).
    type ProvenanceFrame = (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>);
    const VISIBILITY_CALIBRATION_FRAMES: u32 = 8;

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
                "[mega_geometry_bench] city_hybrid is not a distinct certified-Fabric execution \
                 path from the native-scalar reference. Refusing to award a win; target fixtures \
                 report Unavailable and the process exits nonzero."
            );
        }

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
            eprintln!(
                "[mega_geometry_bench] MEGAGEOMETRY_ONLY active: {only:?} (verdict WILL be suppressed — diagnostic run)"
            );
            fixtures
                .iter()
                .filter(|f| only.iter().any(|o| o == &f.name))
                .collect()
        };
        let mut fixture_reports = Vec::new();
        for spec in selected {
            fixture_reports.push(run_fixture(spec, args, width, height, &timed_cfg)?);
        }
        let corpus_hash = finished_object_corpus_hash(&fixture_reports);

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

        // Program and broad ray-native claims are composed with their dedicated
        // probes. The live full-frame Fabric claim is composed here from the
        // real finished-object fixture receipts and calibration profiles.
        let program_bt = logic::compute_program_breakthrough(&fixture_reports);
        let visibility_bt = logic::compute_visibility_breakthrough(&fixture_reports);
        let fabric_bt = logic::compute_fabric_breakthrough(&fixture_reports);

        // ---- Print the evidence ----
        let env = collect_env_line(&fixture_reports);
        println!("{}", env.line());
        for r in &fixture_reports {
            emit_fixture_evidence(r, args.frames);
        }
        println!(
            "MEGAGEOMETRY_SUITE correctness={} unexpected_fallbacks={} required_page_misses={}",
            suite_correctness(&fixture_reports),
            logic::renderer_local_unexpected_fallbacks(&fixture_reports),
            logic::aggregate_required_page_misses(&fixture_reports)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unavailable".into()),
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
            // Preserve partial/measured losing evidence. `None` is reserved for
            // old artifacts; the claim methods remain fail-closed on every
            // unmeasured field.
            program_breakthrough: Some(program_bt),
            visibility_breakthrough: Some(visibility_bt),
            // Preserve measured losing evidence in the raw backend artifact;
            // the final composer, not serialization, decides whether it earns
            // the claim.
            fabric_breakthrough: Some(fabric_bt),
        };
        if let Some(path) = &args.json_out {
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(path, report.to_json()?).map_err(|e| format!("write {path}: {e}"))?;
            eprintln!("[mega_geometry_bench] wrote {path}");
        }

        // Raw producers cannot authorize cross-backend equality or the
        // external Urban Horizon product fixture. They exit successfully when
        // every backend-local receipt is complete and eligible for the final
        // compositor, while the printed overall claim remains suppressed until
        // that join. Missing local evidence still exits nonzero.
        let raw_reasons = logic::raw_backend_validation_reasons(&report);
        for reason in &raw_reasons {
            eprintln!("[mega_geometry_bench] raw artifact rejected: {reason}");
        }
        if raw_reasons.is_empty() {
            println!(
                "MEGAGEOMETRY_RAW_ARTIFACT api={} gpu={} validation=pass",
                report.env.api, report.env.gpu
            );
            Ok(0)
        } else {
            println!(
                "MEGAGEOMETRY_RAW_ARTIFACT api={} gpu={} validation=fail reasons={}",
                report.env.api,
                report.env.gpu,
                raw_reasons.len()
            );
            Ok(1)
        }
    }

    fn spec_of(r: &logic::FixtureReport) -> logic::FixtureSpec {
        logic::required_fixtures()
            .into_iter()
            .find(|f| f.name == r.name)
            .expect("fixture report name must be a required fixture")
    }

    fn suite_correctness(reports: &[logic::FixtureReport]) -> &'static str {
        if logic::renderer_local_correctness_passes(reports) {
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
        if spec.authorizing_witness_external {
            let requested_backend = requested_rt_backend();
            return Ok(logic::FixtureReport {
                name: spec.name.clone(),
                role: spec.role,
                asset_id: spec.asset_id.clone(),
                family: spec.family.clone(),
                instances: spec.instances,
                seed: spec.seed,
                identity_hash: spec.identity_hash_hex(),
                asset_content_hash: None,
                outcome: logic::FixtureOutcome::Unavailable {
                    reason: concat!(
                        "requires independently captured Urban Horizon live-city ",
                        "product evidence; proxy geometry is forbidden"
                    )
                    .into(),
                },
                hybrid: logic::ModeResult::unavailable(logic::Mode::CityHybrid, requested_backend),
                reference: logic::ModeResult::unavailable(
                    logic::Mode::Reference,
                    requested_backend,
                ),
                triangle: None,
                scalar_visibility_profile: None,
                runtime_program: None,
            });
        }

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
        let scalar_visibility_profile =
            measure_scalar_visibility_profile(spec, width, height, cfg.warmup_frames)?;

        let loaded_for_evidence = asset_loader::load_fixture_asset(&spec.asset_id, spec.seed).ok();
        let runtime_program = loaded_for_evidence.as_ref().map(|asset| {
            let payload = asset.mega.as_deref();
            logic::RuntimeProgramEvidence {
                source_triangles: asset.base_triangles() as u64,
                covered_triangles: payload
                    .map(|payload| u64::from(payload.source_triangle_count()))
                    .unwrap_or(0),
                program_count: payload
                    .map(|payload| u64::from(payload.program_count()))
                    .unwrap_or(0),
                template_count: payload
                    .map(|payload| u64::from(payload.template_count()))
                    .unwrap_or(0),
                source_geometry_bytes: asset.source_geometry_bytes,
                covered_source_geometry_bytes: asset.covered_source_geometry_bytes,
                residual_geometry_bytes: asset.residual_geometry_bytes,
                runtime_program_bytes: asset.runtime_program_bytes,
            }
        });
        Ok(logic::FixtureReport {
            name: spec.name.clone(),
            role: spec.role,
            asset_id: spec.asset_id.clone(),
            family: spec.family.clone(),
            instances: spec.instances,
            seed: spec.seed,
            identity_hash: spec.identity_hash_hex(),
            // The real finished asset's content hash (cache hit — the asset was
            // loaded in measure_mode). `None` only for ids with no installed file
            // (e.g. the externally-authorized mixed_city), which keeps that
            // fixture honestly uncertified by vox_render.
            asset_content_hash: loaded_for_evidence
                .as_ref()
                .map(|asset| asset.content_hash.clone()),
            outcome,
            hybrid,
            reference,
            triangle: triangle.map(|(m, _)| m),
            scalar_visibility_profile,
            runtime_program,
        })
    }

    /// Measure the scalar ray-native calibration baseline over the same
    /// residual mesh and Ochroma runtime programs as city_hybrid. This is not
    /// the public `reference` result: reference remains the complete source
    /// mesh lowered through native cluster/CLAS acceleration.
    fn measure_scalar_visibility_profile(
        spec: &logic::FixtureSpec,
        width: u32,
        height: u32,
        warmup_frames: u32,
    ) -> Result<Option<logic::VisibilityKernelProfile>, String> {
        apply_mode_env(logic::Mode::Reference);
        let Some(asset) = asset_loader::load_fixture_asset(&spec.asset_id, spec.seed).ok() else {
            return Ok(None);
        };
        let Some(payload) = asset.mega.as_ref() else {
            return Ok(None);
        };
        let residual = asset.base_without_covered();
        let instances = grid_instances(spec.instances as usize);
        let mut scene = meshes_to_instanced_scene(
            std::slice::from_ref(&residual),
            &instances,
            asset.materials.as_slice(),
            &[],
            width,
            height,
        );
        let transforms: Vec<[f32; 16]> = instances
            .iter()
            .map(|instance| instance.transform)
            .collect();
        scene.mega_geometry =
            asset_loader::assemble_mega_layer(payload, &transforms, asset.materials.len() as u32);
        let mut renderer =
            ResidentSceneRenderer::new(width, height, LightRig::default(), 1, 1, scene).map_err(
                |error| format!("{}/scalar_visibility: renderer build: {error}", spec.name),
            )?;
        let (view, projection) = camera(spec.instances as usize, width, height);
        for frame in 0..warmup_frames {
            renderer.render_only(view, projection).map_err(|error| {
                format!(
                    "{}/scalar_visibility: warmup frame {frame}: {error}",
                    spec.name
                )
            })?;
        }
        collect_visibility_profile(
            &mut renderer,
            view,
            projection,
            logic::Mode::Reference,
            spec,
            true,
        )
        .map(Some)
    }

    fn measure_mode(
        spec: &logic::FixtureSpec,
        mode: logic::Mode,
        width: u32,
        height: u32,
        cfg: &logic::TimedRunConfig,
    ) -> Result<(logic::ModeResult, ProvenanceFrame), String> {
        apply_mode_env(mode);

        // Load the real finished game object for this fixture (cached across modes).
        // Externally witnessed product fixtures returned before this function,
        // so no live-city claim can ever fall back to proxy geometry.
        let asset = asset_loader::load_fixture_asset(&spec.asset_id, spec.seed).ok();
        let cube_fallback = if asset.is_none() {
            eprintln!(
                "[mega_geometry_bench] {}: real asset '{}' unavailable; using a non-authorizing synthetic diagnostic proxy",
                spec.name, spec.asset_id
            );
            Some(fixture_prototype())
        } else {
            None
        };
        // Only city_hybrid may remove source triangles covered by an admitted
        // Ochroma runtime representation. `reference` remains the independent
        // native cluster/CLAS baseline over the complete authoritative mesh;
        // feeding MegaGeometry to both modes would compare one execution path
        // against itself and manufacture "distinct-mode" evidence.
        let city_residual: Option<BlasDesc> = match (&asset, mode) {
            (Some(a), mode) if logic::mode_uses_runtime_megageometry(mode) => {
                Some(a.base_without_covered())
            }
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

        // Populate MegaGeometry only for city_hybrid. Triangle remains the
        // source-GAS oracle and reference remains the complete-mesh native
        // cluster/CLAS competitor.
        if logic::mode_uses_runtime_megageometry(mode) {
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
        let pipeline_before_timed = r.geometry_pipeline_evidence();

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
        let pipeline_after_timed = r.geometry_pipeline_evidence();

        // Detailed device stats copied only AFTER the timed frames.
        let stats = r.geometry_accel_stats().map_err(|e| {
            format!(
                "{}/{}: no acceleration report: {e:?}",
                spec.name,
                mode.as_str()
            )
        })?;
        let streaming_memory = r.geometry_streaming_memory_stats();
        let required_page_misses = r.geometry_required_page_misses();
        let cpu = map_cpu_counters(&r.geometry_cpu_ownership());
        let backend = resolve_backend(mode, &stats);
        let packet_abi_hash = Some(hex32(r.geometry_packet_abi_hash()));
        let capability_fingerprint = Some(hex32(r.geometry_gpu_capability_fingerprint()));
        let visibility_fabric = if logic::mode_uses_runtime_megageometry(mode) {
            r.visibility_queue_stats_after_timed_frames()
                .ok()
                .map(|stats| logic::VisibilityQueueCounters {
                    domains_seeded: u64::from(stats.domains_seeded),
                    rays_certified_miss: u64::from(stats.rays_certified_miss),
                    rays_certified_hit: u64::from(stats.rays_certified_hit),
                    domains_split: u64::from(stats.domains_split),
                    explicit_rays: u64::from(stats.explicit_rays),
                    packet_rays: u64::from(stats.packet_rays),
                    packet_active_lanes: u64::from(stats.packet_active_lanes),
                    packet_total_lanes: u64::from(stats.packet_total_lanes),
                    scalar_refinement_rays: u64::from(stats.scalar_refinement_rays),
                    native_residual_rays: u64::from(stats.native_residual_rays),
                    failed_certificates: u64::from(stats.failed_certificates),
                    queue_overflow_mask: u64::from(stats.queue_overflow_mask),
                    cpu_scheduling: u64::from(stats.cpu_scheduling),
                    correctness_mismatches: u64::from(stats.correctness_mismatches),
                    source_ray_capacity: u64::from(stats.source_ray_capacity),
                    live_source_rays: u64::from(stats.live_source_rays),
                })
        } else {
            None
        };
        // Only city_hybrid owns the persistent GPU geometry-control graph.
        // Triangle/reference have no such state and report no hash; the
        // independent compositor requires equality for the candidate only.
        let control_hash = if logic::mode_uses_runtime_megageometry(mode) {
            Some(hex32(
                r.geometry_control_hash_after_timed_frames()
                    .map_err(|error| {
                        format!(
                            "{}/{}: GPU control hash unavailable: {error}",
                            spec.name,
                            mode.as_str()
                        )
                    })?,
            ))
        } else {
            None
        };
        // Exercise actual GPU detail-cut transitions separately from the timed
        // product interval. A static settled frame is not temporal evidence.
        // Each receipt is read only after its calibration frame completes and
        // is never fed back into rendering or residency decisions.
        let temporal_reuse = if logic::mode_uses_runtime_megageometry(mode) {
            match collect_temporal_reuse_evidence(&mut r, spec, width, height, mode) {
                Ok(evidence) => Some(evidence),
                Err(error) => {
                    eprintln!(
                        "[mega_geometry_bench] {}/{}: temporal calibration unavailable: {error}",
                        spec.name,
                        mode.as_str()
                    );
                    None
                }
            }
        } else {
            None
        };
        // Serialized device timing is intentionally collected after both the
        // ordinary timed interval and its receipts. These calibration frames
        // never contribute to product frame percentiles.
        let has_runtime_program = asset
            .as_ref()
            .and_then(|loaded| loaded.mega.as_ref())
            .is_some();
        let visibility_profile = if (logic::mode_uses_runtime_megageometry(mode)
            && has_runtime_program)
            || mode == logic::Mode::Reference
        {
            Some(collect_visibility_profile(
                &mut r,
                view,
                proj,
                mode,
                spec,
                logic::mode_uses_runtime_megageometry(mode),
            )?)
        } else {
            None
        };
        let parameter_update_fine_as_builds =
            if logic::mode_uses_runtime_megageometry(mode) && has_runtime_program {
                match collect_parameter_update_evidence(&mut r, view, proj, mode, spec) {
                    Ok(builds) => Some(builds),
                    Err(error) => {
                        eprintln!(
                            "[mega_geometry_bench] {}/{}: parameter update unavailable: {error}",
                            spec.name,
                            mode.as_str()
                        );
                        None
                    }
                }
            } else {
                None
            };
        let mib = 1024.0 * 1024.0;
        let streaming_as = streaming_memory
            .map(|memory| memory.fixed_as_storage_bytes)
            .unwrap_or(0);
        let streaming_native = streaming_memory
            .map(|memory| {
                memory
                    .native_argument_bytes
                    .saturating_add(memory.control_bytes)
                    .saturating_add(memory.fixed_as_build_input_bytes)
            })
            .unwrap_or(0);
        let streaming_scratch = streaming_memory
            .map(|memory| memory.fixed_as_scratch_bytes)
            .unwrap_or(0);
        let streaming_pages = streaming_memory
            .map(|memory| memory.page_pool_bytes)
            .unwrap_or(0);
        let resident_geometry = stats
            .materialized_geometry_bytes()
            .ok_or_else(|| {
                format!(
                    "{}/{}: materialized geometry memory unavailable",
                    spec.name,
                    mode.as_str()
                )
            })?
            .saturating_add(spec.instances as u64 * 128);
        let as_storage = stats
            .as_storage_bytes()
            .ok_or_else(|| {
                format!(
                    "{}/{}: acceleration storage memory unavailable",
                    spec.name,
                    mode.as_str()
                )
            })?
            .saturating_add(streaming_as);
        let native_arguments = stats
            .retained_build_input_bytes()
            .map(|bytes| bytes.saturating_add(streaming_native));
        let build_scratch = stats
            .retained_scratch_bytes()
            .map(|bytes| bytes.saturating_add(streaming_scratch));

        let mr = logic::ModeResult {
            mode,
            compute: logic::ComputeVariant::PortableSubgroup,
            packet_abi_hash,
            control_hash,
            capability_fingerprint,
            visibility_fabric,
            visibility_profile,
            temporal_reuse,
            parameter_update_fine_as_builds,
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
                required_page_misses: if logic::mode_uses_runtime_megageometry(mode) {
                    required_page_misses
                } else {
                    // The two materialized controls intentionally have no
                    // streaming graph, hence no page that can miss.
                    Some(0)
                },
            },
            vram: logic::VramBreakdown {
                resident_mb: resident_geometry as f64 / mib,
                as_mb: Some(as_storage as f64 / mib),
                page_mb: Some(streaming_pages as f64 / mib),
                native_args_mb: native_arguments.map(|bytes| bytes as f64 / mib),
                scratch_mb: build_scratch.map(|bytes| bytes as f64 / mib),
            },
            cpu,
            prewarm: Some(logic::PrewarmState {
                shaders_compiled_before_timed: pipeline_before_timed.0 > 0,
                pipeline_created_in_timed_frame: pipeline_after_timed.0 != pipeline_before_timed.0,
                cache_repaired_in_timed_frame: pipeline_after_timed.1 != pipeline_before_timed.1,
            }),
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

    fn collect_parameter_update_evidence(
        renderer: &mut ResidentSceneRenderer,
        view: [f32; 16],
        proj: [f32; 16],
        mode: logic::Mode,
        spec: &logic::FixtureSpec,
    ) -> Result<u64, String> {
        // 1.015 is non-identity and lies strictly inside the runtime schema's
        // [0.975, 1.025] conservative facade-taper envelope.
        renderer
            .update_geometry_visibility_parameters(&[(0, 1.015)])
            .map_err(|error| {
                format!(
                    "{}/{}: enqueue bounded parameter update: {error}",
                    spec.name,
                    mode.as_str()
                )
            })?;
        renderer.render_only(view, proj).map_err(|error| {
            format!(
                "{}/{}: render bounded parameter update: {error}",
                spec.name,
                mode.as_str()
            )
        })?;
        let receipt = renderer
            .geometry_parameter_update_stats()
            .map_err(|error| {
                format!(
                    "{}/{}: parameter update receipt: {error}",
                    spec.name,
                    mode.as_str()
                )
            })?;
        if receipt.submitted != 1 || receipt.changed != 1 || receipt.rejected != 0 {
            return Err(format!(
                "GPU receipt is not a nontrivial accepted update: submitted={} changed={} rejected={}",
                receipt.submitted, receipt.changed, receipt.rejected
            ));
        }
        Ok(receipt.fine_as_builds)
    }

    fn collect_temporal_reuse_evidence(
        renderer: &mut ResidentSceneRenderer,
        spec: &logic::FixtureSpec,
        width: u32,
        height: u32,
        mode: logic::Mode,
    ) -> Result<logic::TemporalReuseEvidence, String> {
        let mut evidence = logic::TemporalReuseEvidence {
            relevant_surface_pixels: 0,
            invalidated_surface_pixels: 0,
            no_map_invalidated_surface_pixels: 0,
            lod_cut_changes: 0,
        };
        // Alternate near/far views so the GPU selector must evaluate materially
        // different projected errors. Returning to 1.0 also proves the state is
        // persistent rather than a one-time admission artifact.
        for distance_scale in [0.35_f32, 1.80, 0.55, 1.0] {
            let (view, proj) =
                camera_at_distance(spec.instances as usize, width, height, distance_scale);
            renderer.render_only(view, proj).map_err(|error| {
                format!(
                    "{}/{} distance_scale={distance_scale}: render: {error}",
                    spec.name,
                    mode.as_str()
                )
            })?;
            let frame = renderer.geometry_temporal_reuse_stats().map_err(|error| {
                format!(
                    "{}/{} distance_scale={distance_scale}: receipt: {error}",
                    spec.name,
                    mode.as_str()
                )
            })?;
            evidence.relevant_surface_pixels = evidence
                .relevant_surface_pixels
                .saturating_add(frame.relevant_surface_pixels);
            evidence.invalidated_surface_pixels = evidence
                .invalidated_surface_pixels
                .saturating_add(frame.invalidated_surface_pixels);
            evidence.no_map_invalidated_surface_pixels = evidence
                .no_map_invalidated_surface_pixels
                .saturating_add(frame.no_map_invalidated_surface_pixels);
            evidence.lod_cut_changes = evidence
                .lod_cut_changes
                .saturating_add(frame.lod_cut_changes);
        }
        Ok(evidence)
    }

    #[derive(Default)]
    struct VisibilityFrameStages {
        certification: f64,
        subdivision: f64,
        routing_queue: f64,
        packet_intersection: f64,
        exact_refinement: f64,
        ordinary_native: f64,
        native_residual: f64,
        hit_merge: f64,
        fixed_launch: f64,
        saw_fixed_launch: bool,
    }

    fn collect_visibility_profile(
        renderer: &mut ResidentSceneRenderer,
        view: [f32; 16],
        proj: [f32; 16],
        mode: logic::Mode,
        spec: &logic::FixtureSpec,
        require_fixed_native_launch: bool,
    ) -> Result<logic::VisibilityKernelProfile, String> {
        let mut frames = Vec::with_capacity(VISIBILITY_CALIBRATION_FRAMES as usize);
        for frame in 0..VISIBILITY_CALIBRATION_FRAMES {
            renderer
                .begin_visibility_calibration_frame()
                .map_err(|error| {
                    format!(
                        "{}/{}: arm visibility calibration frame {frame}: {error}",
                        spec.name,
                        mode.as_str()
                    )
                })?;
            renderer.render_only(view, proj).map_err(|error| {
                format!(
                    "{}/{}: visibility calibration frame {frame}: {error}",
                    spec.name,
                    mode.as_str()
                )
            })?;
            let dispatches = renderer
                .finish_visibility_calibration_frame()
                .map_err(|error| {
                    format!(
                        "{}/{}: finish visibility calibration frame {frame}: {error}",
                        spec.name,
                        mode.as_str()
                    )
                })?;
            frames.push(classify_visibility_dispatches(mode, &dispatches));
        }
        if frames.iter().any(|frame| {
            (require_fixed_native_launch && !frame.saw_fixed_launch)
                || frame.ordinary_native + frame.native_residual <= 0.0
        }) {
            return Err(format!(
                "{}/{}: calibration omitted required native visibility work",
                spec.name,
                mode.as_str()
            ));
        }

        let values = |select: fn(&VisibilityFrameStages) -> f64| -> Vec<f64> {
            frames.iter().map(select).collect()
        };
        let complete: Vec<f64> = frames
            .iter()
            .map(|frame| {
                frame.certification
                    + frame.subdivision
                    + frame.routing_queue
                    + frame.packet_intersection
                    + frame.exact_refinement
                    + frame.ordinary_native
                    + frame.native_residual
                    + frame.hit_merge
            })
            .collect();
        let profile = logic::VisibilityKernelProfile {
            measured_frames: VISIBILITY_CALIBRATION_FRAMES,
            certification_ms_p95: p95(&values(|frame| frame.certification)),
            subdivision_ms_p95: p95(&values(|frame| frame.subdivision)),
            routing_queue_ms_p95: p95(&values(|frame| frame.routing_queue)),
            packet_intersection_ms_p95: p95(&values(|frame| frame.packet_intersection)),
            exact_refinement_ms_p95: p95(&values(|frame| frame.exact_refinement)),
            ordinary_native_ms_p95: p95(&values(|frame| frame.ordinary_native)),
            native_residual_ms_p95: p95(&values(|frame| frame.native_residual)),
            hit_merge_ms_p95: p95(&values(|frame| frame.hit_merge)),
            // This is a subset of ordinary/native-residual timing and is not
            // added to `complete_visibility_ms_p95` a second time.
            fixed_empty_launch_ms_p95: p95(&values(|frame| frame.fixed_launch)),
            complete_visibility_ms_p95: p95(&complete),
            includes_fixed_empty_launches: frames.iter().all(|frame| frame.saw_fixed_launch),
        };
        profile.is_measured().then_some(profile).ok_or_else(|| {
            format!(
                "{}/{}: visibility calibration produced an incomplete profile",
                spec.name,
                mode.as_str()
            )
        })
    }

    fn classify_visibility_dispatches(
        mode: logic::Mode,
        dispatches: &[(String, f32)],
    ) -> VisibilityFrameStages {
        let mut stages = VisibilityFrameStages::default();
        for (label, elapsed_ms) in dispatches {
            let ms = f64::from(*elapsed_ms);
            match label.as_str() {
                "visibility_fabric_build_primary" | "visibility_domain_classify_exact" => {
                    stages.certification += ms;
                }
                "visibility_domain_subdivide" => stages.subdivision += ms,
                "visibility_fabric_reset"
                | "visibility_route"
                | "visibility_route_subdivision"
                | "visibility_fabric_packetize"
                | "visibility_fabric_fail_closed" => stages.routing_queue += ms,
                "visibility_packet_intersect" => stages.packet_intersection += ms,
                "visibility_hit_reduce" => stages.hit_merge += ms,
                "visibility_native_residual_optix" => {
                    stages.native_residual += ms;
                    stages.fixed_launch += ms;
                    stages.saw_fixed_launch = true;
                }
                "visibility_native_scalar_optix"
                | "triangle_and_visibility_native_scalar_optix" => {
                    stages.ordinary_native += ms;
                    stages.fixed_launch += ms;
                    stages.saw_fixed_launch = true;
                }
                "triangle_trace_optix" | "optix_traverse" => stages.ordinary_native += ms,
                _ if label.starts_with("megageometry_merge_") => {
                    if mode == logic::Mode::CityHybrid {
                        stages.native_residual += ms;
                    } else {
                        stages.ordinary_native += ms;
                    }
                    stages.fixed_launch += ms;
                    stages.saw_fixed_launch = true;
                }
                _ if label.starts_with("megageometry_fused_trace_") => {
                    stages.ordinary_native += ms;
                    stages.fixed_launch += ms;
                    stages.saw_fixed_launch = true;
                }
                _ if label.starts_with("bvh_traverse")
                    || label.starts_with("shadow_trace")
                    || label.starts_with("megageometry_shadow_")
                    || label.starts_with("megageometry_triangle_detail_") =>
                {
                    stages.ordinary_native += ms;
                }
                _ => {}
            }
        }
        stages
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
                            op[i],
                            p[i],
                            om[i],
                            m[i],
                            od[i],
                            d[i],
                            (d[i] - od[i]).abs()
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
        let host_calls = snap
            .host_service_calls(GeometryHostServiceKind::NativeAccelerationSubmission)
            + snap.host_service_calls(GeometryHostServiceKind::OpaquePageTransport);
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
            counters_probed: super::logic::HARNESS_CPU_OWNERSHIP_COUNTERS_PROBED,
            host_deadline_misses: Some(snap.host_deadline_misses()),
            render_thread_submit_ms_p95: Some(snap.render_thread_submit_p95_ns() as f64 / 1.0e6),
            host_submit_calls: host_calls,
            host_submit_ms_p95: if host_calls > 0 {
                Some(snap.host_submit_p95_ns() as f64 / 1.0e6)
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
        let requested_cluster = logic::requires_hardware_clas(requested_rt_backend(), mode);
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
            hardware_clas_builds: Some(u64::from(stats.hardware_clas_builds())),
            cluster_requested_without_hardware_clas,
        }
    }

    fn emit_fixture_evidence(r: &logic::FixtureReport, frames: u32) {
        let spec = spec_of(r);
        for m in [&r.hybrid, &r.reference] {
            println!("{}", logic::config_line(m.mode, m.compute, &spec, frames));
            println!("{}", logic::identity_line(m));
            println!("{}", logic::correct_line(&m.correctness));
            println!("{}", logic::result_line(&m.stage, &m.vram));
            println!("{}", logic::cpu_line(&m.cpu));
            // Program / visibility timing evidence remains unavailable until
            // the corresponding GPU timer receipts are exposed.
            println!(
                "{}",
                logic::ProgramLine::unavailable(m.mode.as_str()).line()
            );
            println!(
                "{}",
                logic::VisibilityLine::unavailable(m.mode.as_str()).line()
            );
            println!(
                "{}",
                m.visibility_fabric
                    .map(|stats| stats.fabric_line(m.mode))
                    .unwrap_or_else(|| logic::FabricLine::unavailable(m.mode.as_str()))
                    .line()
            );
        }
        println!("{}", logic::TemporalLine::unavailable().line());
    }

    fn collect_global_signals(fixtures: &[logic::FixtureReport]) -> logic::GlobalSignals {
        // Two scale samples (smallest and largest fixture) with the REAL forbidden
        // CPU counters for the scaling proof. The externally authorized live
        // product fixture has no renderer-local receipts and is joined only by
        // `mega_geometry_acceptance`; it must not poison or fabricate raw
        // backend-global measurements.
        let internal: Vec<_> = fixtures
            .iter()
            .filter(|fixture| !spec_of(fixture).authorizing_witness_external)
            .collect();
        let mut small = internal[0];
        let mut large = internal[0];
        for &f in &internal {
            if f.instances < small.instances {
                small = f;
            }
            if f.instances > large.instances {
                large = f;
            }
        }
        let cpu_fallbacks = internal
            .iter()
            .flat_map(|fixture| [&fixture.hybrid, &fixture.reference])
            .map(|mode| mode.backend.cpu_fallbacks)
            .sum();
        let prewarm_states = internal
            .iter()
            .flat_map(|fixture| [&fixture.hybrid, &fixture.reference])
            .map(|mode| mode.prewarm);
        let cache_valid = if prewarm_states.clone().all(|state| state.is_some()) {
            Some(prewarm_states.flatten().all(|state| state.is_valid()))
        } else {
            None
        };
        let fabric_receipts: Vec<_> = internal
            .iter()
            .filter(|fixture| logic::REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()))
            .map(|fixture| fixture.hybrid.visibility_fabric)
            .collect();
        let fabric_cpu_scheduling =
            if !fabric_receipts.is_empty() && fabric_receipts.iter().all(Option::is_some) {
                Some(
                    fabric_receipts
                        .iter()
                        .flatten()
                        .map(|stats| stats.cpu_scheduling)
                        .sum(),
                )
            } else {
                None
            };

        logic::GlobalSignals {
            // A single-backend producer emits raw identities. Only the artifact
            // composer may compare independent CUDA and Vulkan artifacts.
            packet_abi_match: None,
            control_hash_match: None,
            adapter_overhead_pct: None,
            cpu_fallbacks: Some(cpu_fallbacks),
            fabric_cpu_scheduling,
            cache_valid,
            mode_distinct: Some(CITY_HYBRID_IS_DISTINCT_EXECUTION),
            scale_small: logic::ScaleSample {
                total_units: small.instances,
                cpu: small.hybrid.cpu,
            },
            scale_large: logic::ScaleSample {
                total_units: large.instances,
                cpu: large.hybrid.cpu,
            },
        }
    }

    fn collect_env_line(fixtures: &[logic::FixtureReport]) -> logic::EnvLine {
        let mode_records = fixtures
            .iter()
            .filter(|fixture| {
                !logic::required_fixtures()
                    .iter()
                    .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
            })
            .flat_map(|fixture| {
                let mut modes = vec![&fixture.reference, &fixture.hybrid];
                modes.extend(fixture.triangle.iter());
                modes
            })
            .collect::<Vec<_>>();
        let capability_hashes = mode_records
            .iter()
            .filter_map(|mode| mode.capability_fingerprint.clone())
            .collect::<BTreeSet<_>>();
        let packet_abi_hashes = mode_records
            .iter()
            .filter_map(|mode| mode.packet_abi_hash.clone())
            .collect::<BTreeSet<_>>();
        let unique = |values: BTreeSet<String>| {
            (values.len() == 1)
                .then(|| values.into_iter().next())
                .flatten()
                .unwrap_or_else(|| "unavailable".into())
        };
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
            capability_hash: unique(capability_hashes),
            shader_abi: unique(packet_abi_hashes),
            runtime_geometry_schema: vox_data::mega_geometry::READY_MEGA_GEOMETRY_SCHEMA,
        }
    }

    fn collect_git_revisions() -> logic::GitRevisions {
        logic::GitRevisions {
            ochroma: git_rev(env!("CARGO_MANIFEST_DIR")),
            spectra: std::env::var("MEGAGEOMETRY_REV_SPECTRA").unwrap_or_else(|_| "unknown".into()),
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

    /// Benchmark-only mode selection. The product does not read these values:
    /// its visibility policy remains config-first and requires persisted
    /// calibration evidence.
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
                    std::env::remove_var("MEGAGEOMETRY_VISIBILITY_CALIBRATION_CANDIDATE");
                }
                logic::Mode::Reference => {
                    std::env::set_var("SPECTRA_CLAS_THRESHOLD", "1");
                    std::env::set_var("MEGAGEOMETRY_MODE", "reference");
                    std::env::remove_var("MEGAGEOMETRY_VISIBILITY_CALIBRATION_CANDIDATE");
                }
                logic::Mode::CityHybrid => {
                    std::env::set_var("SPECTRA_CLAS_THRESHOLD", "1");
                    std::env::set_var("MEGAGEOMETRY_MODE", "city_hybrid");
                    std::env::set_var("MEGAGEOMETRY_VISIBILITY_CALIBRATION_CANDIDATE", "1");
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
                dynamic: false,
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
        camera_at_distance(n, width, height, 1.0)
    }

    fn camera_at_distance(
        n: usize,
        width: u32,
        height: u32,
        distance_scale: f32,
    ) -> ([f32; 16], [f32; 16]) {
        let cols = (n as f64).sqrt().ceil() as f32;
        let extent = cols.max(1.0) * 2.0;
        let dist = (extent * 1.2 + 5.0) * distance_scale.max(0.1);
        let eye = glam::Vec3::new(0.0, dist * 0.6, dist);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y).to_cols_array();
        let aspect = width as f32 / height.max(1) as f32;
        let scene_extent = extent * 1.2 + 5.0;
        let proj =
            glam::Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, scene_extent * 8.0 + 100.0)
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
