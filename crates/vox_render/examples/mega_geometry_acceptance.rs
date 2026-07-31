//! Fail-closed final join for independently produced MegaGeometry evidence.
//!
//! This process does no rendering and knows nothing about asset authoring. It
//! accepts only finished benchmark artifacts from the Urban Horizon -> Ochroma
//! -> Spectra runtime chain and publishes a complete claim only when every
//! independently measured component agrees.

#[path = "support/mega_geometry_bench_logic.rs"]
mod logic;

use logic::{
    BenchReport, FixtureResult, Mode, compose_backend_conformance, compute_fabric_breakthrough,
    compute_overall_with_external, compute_program_breakthrough, compute_visibility_breakthrough,
    measured_adapter_overhead_pct, raw_backend_validation_reasons, required_fixtures,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct ProductComposition {
    schema: u32,
    outcome: String,
    winner: Option<String>,
    scene_hash: String,
    camera_hash: String,
    render_config_hash: String,
    reference_witness_sha256: String,
    city_hybrid_witness_sha256: String,
    reference_capture_sha256: String,
    city_hybrid_capture_sha256: String,
    visual_inspection_sha256: String,
    city_hybrid_total_ms_p95: f64,
    p50_improvement_pct: f64,
    p95_improvement_pct: f64,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AcceptanceReport {
    schema: u32,
    status: &'static str,
    claim: Option<&'static str>,
    cuda_gpu: String,
    vulkan_gpu: String,
    program_devices: [String; 2],
    visibility_devices: [String; 2],
    corpus_hash: String,
    backend_conformance: bool,
    native_clas: bool,
    geometry_program_probe: bool,
    ray_native_probe: bool,
    product_ab: bool,
    program_breakthrough: bool,
    visibility_breakthrough: bool,
    fabric_breakthrough: bool,
    reasons: Vec<String>,
}

struct Args {
    cuda: PathBuf,
    vulkan: PathBuf,
    product: PathBuf,
    json_out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, PathBuf>::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a path"))?;
        if !matches!(
            flag.as_str(),
            "--cuda" | "--vulkan" | "--product" | "--json-out"
        ) {
            return Err(format!("unknown argument {flag}"));
        }
        if values.insert(flag.clone(), value.into()).is_some() {
            return Err(format!("{flag} was supplied more than once"));
        }
    }
    let mut take = |flag: &str| {
        values
            .remove(flag)
            .ok_or_else(|| format!("{flag} is required"))
    };
    Ok(Args {
        cuda: take("--cuda")?,
        vulkan: take("--vulkan")?,
        product: take("--product")?,
        json_out: take("--json-out")?,
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.is_empty() {
        return Err(format!("{} is empty", path.display()));
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_git_revision(value: &str) -> bool {
    (7..=40).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_report_identity(label: &str, report: &BenchReport, reasons: &mut Vec<String>) {
    for (field, value) in [
        ("GPU", report.env.gpu.as_str()),
        ("driver", report.env.driver.as_str()),
        ("RT backend", report.env.rt_backend.as_str()),
    ] {
        if value.trim().is_empty() || matches!(value, "unknown" | "unavailable") {
            reasons.push(format!("{label} {field} identity is unavailable"));
        }
    }
    for (field, value) in [
        ("capability", report.env.capability_hash.as_str()),
        ("shader ABI", report.env.shader_abi.as_str()),
        ("render config", report.render_config_hash.as_str()),
        ("finished-object corpus", report.corpus_hash.as_str()),
    ] {
        if !is_sha256(value) {
            reasons.push(format!("{label} {field} identity is not a SHA-256 digest"));
        }
    }
    for (field, value) in [
        ("Ochroma revision", report.git_revisions.ochroma.as_str()),
        ("Spectra revision", report.git_revisions.spectra.as_str()),
        (
            "Urban Horizon revision",
            report.git_revisions.urban.as_str(),
        ),
    ] {
        if !is_git_revision(value) {
            reasons.push(format!("{label} {field} is not a Git revision"));
        }
    }
    for fixture in report.fixtures.iter().filter(|fixture| {
        !required_fixtures()
            .iter()
            .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
    }) {
        let mut modes = vec![&fixture.reference, &fixture.hybrid];
        modes.extend(fixture.triangle.iter());
        for mode in modes {
            if mode.capability_fingerprint.as_deref() != Some(report.env.capability_hash.as_str()) {
                reasons.push(format!(
                    "{label} {}/{} capability identity disagrees with its measured receipt",
                    fixture.name,
                    mode.mode.as_str()
                ));
            }
            if mode.packet_abi_hash.as_deref() != Some(report.env.shader_abi.as_str()) {
                reasons.push(format!(
                    "{label} {}/{} packet ABI disagrees with its measured receipt",
                    fixture.name,
                    mode.mode.as_str()
                ));
            }
        }
    }
}

fn validate_product(product: &ProductComposition, reasons: &mut Vec<String>) -> bool {
    let before = reasons.len();
    if product.schema != 1
        || product.outcome != "win"
        || product.winner.as_deref() != Some("city_hybrid")
        || !product.reasons.is_empty()
    {
        reasons.push("Urban Horizon product A/B did not produce a clean city_hybrid win".into());
    }
    for (field, value) in [
        ("scene", product.scene_hash.as_str()),
        ("camera", product.camera_hash.as_str()),
        ("render config", product.render_config_hash.as_str()),
        (
            "reference witness",
            product.reference_witness_sha256.as_str(),
        ),
        (
            "city_hybrid witness",
            product.city_hybrid_witness_sha256.as_str(),
        ),
        (
            "reference capture",
            product.reference_capture_sha256.as_str(),
        ),
        (
            "city_hybrid capture",
            product.city_hybrid_capture_sha256.as_str(),
        ),
        (
            "visual inspection",
            product.visual_inspection_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            reasons.push(format!("Urban Horizon product {field} identity is invalid"));
        }
    }
    if !finite_positive(product.city_hybrid_total_ms_p95)
        || 1000.0 / product.city_hybrid_total_ms_p95 < 30.0
    {
        reasons.push("Urban Horizon city_hybrid misses the AMD 30 fps p95 gate".into());
    }
    if !product.p50_improvement_pct.is_finite()
        || !product.p95_improvement_pct.is_finite()
        || product.p50_improvement_pct < -2.0
        || product.p95_improvement_pct < 2.0
    {
        reasons.push("Urban Horizon product A/B performance thresholds failed".into());
    }
    reasons.len() == before
}

fn native_clas_passes(cuda: &BenchReport, reasons: &mut Vec<String>) -> bool {
    let before = reasons.len();
    let mut measured = 0_u64;
    for fixture in cuda.fixtures.iter().filter(|fixture| {
        !logic::required_fixtures()
            .iter()
            .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
    }) {
        for mode in [&fixture.reference, &fixture.hybrid] {
            if mode.backend.requested_rt_backend == "optix" {
                match mode.backend.hardware_clas_builds {
                    Some(count) if count > 0 => measured += count,
                    Some(_) => reasons.push(format!(
                        "{}/{} measured zero hardware CLAS builds",
                        fixture.name,
                        mode.mode.as_str()
                    )),
                    None => reasons.push(format!(
                        "{}/{} has no hardware CLAS counter",
                        fixture.name,
                        mode.mode.as_str()
                    )),
                }
            }
        }
    }
    if measured == 0 {
        reasons.push("no native OptiX CLAS build was measured".into());
    }
    reasons.len() == before
}

fn claim_agrees(
    label: &str,
    claim_name: &str,
    serialized: Option<&str>,
    recomputed: Option<&str>,
    expected: &str,
    reasons: &mut Vec<String>,
) -> bool {
    if serialized != recomputed {
        reasons.push(format!(
            "{label} {claim_name} producer/composer disagreement: serialized={} recomputed={}",
            serialized.unwrap_or("none"),
            recomputed.unwrap_or("none")
        ));
        return false;
    }
    if recomputed != Some(expected) {
        reasons.push(format!(
            "{label} {claim_name} raw fixture evidence did not authorize {expected}"
        ));
        return false;
    }
    true
}

fn validate_recomputed_breakthroughs(
    label: &str,
    report: &BenchReport,
    reasons: &mut Vec<String>,
) -> (bool, bool, bool) {
    let recomputed_program = compute_program_breakthrough(&report.fixtures);
    let recomputed_visibility = compute_visibility_breakthrough(&report.fixtures);
    let recomputed_fabric = compute_fabric_breakthrough(&report.fixtures);
    let program = claim_agrees(
        label,
        "geometry-program breakthrough",
        report
            .program_breakthrough
            .as_ref()
            .and_then(|claim| claim.claim()),
        recomputed_program.claim(),
        "city_geometry_program_breakthrough",
        reasons,
    );
    let visibility = claim_agrees(
        label,
        "ray-native breakthrough",
        report
            .visibility_breakthrough
            .as_ref()
            .and_then(|claim| claim.claim()),
        recomputed_visibility.claim(),
        "ray_native_city_geometry",
        reasons,
    );
    let fabric = claim_agrees(
        label,
        "visibility-fabric breakthrough",
        report
            .fabric_breakthrough
            .as_ref()
            .and_then(|claim| claim.claim()),
        recomputed_fabric.claim(),
        "certified_visibility_fabric",
        reasons,
    );
    (program, visibility, fabric)
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let cuda: BenchReport = read_json(&args.cuda)?;
    let vulkan: BenchReport = read_json(&args.vulkan)?;
    let product: ProductComposition = read_json(&args.product)?;

    let mut reasons = Vec::new();
    validate_report_identity("CUDA", &cuda, &mut reasons);
    validate_report_identity("Vulkan", &vulkan, &mut reasons);
    for reason in raw_backend_validation_reasons(&cuda) {
        reasons.push(format!("CUDA raw report: {reason}"));
    }
    for reason in raw_backend_validation_reasons(&vulkan) {
        reasons.push(format!("Vulkan raw report: {reason}"));
    }
    let conformance = compose_backend_conformance(&cuda, &vulkan);
    let backend_conformance = conformance.passes();
    if !backend_conformance {
        reasons.extend(
            conformance
                .reasons
                .iter()
                .map(|reason| format!("backend conformance: {reason}")),
        );
        if conformance.reasons.is_empty() {
            reasons.push("backend conformance failed".into());
        }
    }

    let native_clas = native_clas_passes(&cuda, &mut reasons);
    let product_ab = validate_product(&product, &mut reasons);
    if product.render_config_hash != cuda.render_config_hash
        || product.render_config_hash != vulkan.render_config_hash
    {
        reasons.push(
            "Urban Horizon and backend reports do not share one render-config identity".into(),
        );
    }

    let mut joined_global = cuda.global.clone();
    joined_global.packet_abi_match = conformance.packet_abi_match;
    joined_global.control_hash_match = conformance.control_hash_match;
    joined_global.adapter_overhead_pct = match (
        measured_adapter_overhead_pct(&cuda),
        measured_adapter_overhead_pct(&vulkan),
    ) {
        (Some(cuda), Some(vulkan)) => Some(cuda.max(vulkan)),
        _ => None,
    };
    joined_global.cpu_fallbacks = match (cuda.global.cpu_fallbacks, vulkan.global.cpu_fallbacks) {
        (Some(cuda), Some(vulkan)) => Some(cuda.saturating_add(vulkan)),
        _ => None,
    };
    joined_global.fabric_cpu_scheduling = match (
        cuda.global.fabric_cpu_scheduling,
        vulkan.global.fabric_cpu_scheduling,
    ) {
        (Some(cuda), Some(vulkan)) => Some(cuda.saturating_add(vulkan)),
        _ => None,
    };
    joined_global.cache_valid = match (cuda.global.cache_valid, vulkan.global.cache_valid) {
        (Some(cuda), Some(vulkan)) => Some(cuda && vulkan),
        _ => None,
    };
    joined_global.mode_distinct = match (cuda.global.mode_distinct, vulkan.global.mode_distinct) {
        (Some(cuda), Some(vulkan)) => Some(cuda && vulkan),
        _ => None,
    };
    let cuda_fixtures = cuda
        .fixtures
        .iter()
        .map(|report| {
            let spec = required_fixtures()
                .into_iter()
                .find(|spec| spec.name == report.name)
                .ok_or_else(|| format!("CUDA report contains unknown fixture {}", report.name))?;
            Ok(FixtureResult {
                spec,
                outcome: report.outcome.clone(),
                hybrid: report.hybrid.clone(),
                reference: report.reference.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let joined_overall = compute_overall_with_external(&cuda_fixtures, &joined_global, product_ab);
    if joined_overall.winner != Some(Mode::CityHybrid)
        || joined_overall.losses != 0
        || joined_overall.claim != "better_city_geometry_system"
    {
        reasons.push(format!(
            "joined city benchmark did not authorize the complete system claim: {}",
            joined_overall
                .suppressed_reason
                .as_deref()
                .unwrap_or(&joined_overall.claim)
        ));
    }

    // Never trust a producer-authored aggregate label. Recompose every
    // breakthrough from the raw per-fixture measurements in each independent
    // backend artifact, then require the producer's serialized aggregate to
    // agree. This catches both tampering and producer/composer drift.
    let (cuda_program_breakthrough, cuda_visibility_breakthrough, cuda_fabric_breakthrough) =
        validate_recomputed_breakthroughs("CUDA", &cuda, &mut reasons);
    let (vulkan_program_breakthrough, vulkan_visibility_breakthrough, vulkan_fabric_breakthrough) =
        validate_recomputed_breakthroughs("Vulkan", &vulkan, &mut reasons);
    let program_breakthrough = cuda_program_breakthrough && vulkan_program_breakthrough;
    let geometry_program_probe = program_breakthrough;
    let visibility_breakthrough = cuda_visibility_breakthrough && vulkan_visibility_breakthrough;
    let ray_native_probe = visibility_breakthrough;
    let fabric_breakthrough = cuda_fabric_breakthrough && vulkan_fabric_breakthrough;
    if cuda.env.api != "cuda" || vulkan.env.api != "vulkan" {
        reasons.push("final reports are not CUDA and Vulkan respectively".into());
    }
    let passes = reasons.is_empty();
    let cuda_gpu = cuda.env.gpu.clone();
    let vulkan_gpu = vulkan.env.gpu.clone();
    let report = AcceptanceReport {
        schema: 2,
        status: if passes { "pass" } else { "fail" },
        claim: passes.then_some("complete_ochroma_megageometry"),
        cuda_gpu: cuda_gpu.clone(),
        vulkan_gpu: vulkan_gpu.clone(),
        program_devices: [cuda_gpu.clone(), vulkan_gpu.clone()],
        visibility_devices: [cuda_gpu, vulkan_gpu],
        corpus_hash: cuda.corpus_hash,
        backend_conformance,
        native_clas,
        geometry_program_probe,
        ray_native_probe,
        product_ab,
        program_breakthrough,
        visibility_breakthrough,
        fabric_breakthrough,
        reasons,
    };

    if let Some(parent) = args.json_out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize acceptance report: {error}"))?;
    let temporary = args.json_out.with_extension("json.tmp");
    std::fs::write(&temporary, bytes)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, &args.json_out).map_err(|error| {
        format!(
            "publish {} -> {}: {error}",
            temporary.display(),
            args.json_out.display()
        )
    })?;
    println!("{}", logic::overall_line(&joined_overall));
    println!(
        "MEGAGEOMETRY_ACCEPTANCE backend_conformance={} native_clas={} geometry_program={} ray_native={} product_ab={} program_breakthrough={} visibility_breakthrough={} fabric_breakthrough={} status={} claim={}",
        report.backend_conformance,
        report.native_clas,
        report.geometry_program_probe,
        report.ray_native_probe,
        report.product_ab,
        report.program_breakthrough,
        report.visibility_breakthrough,
        report.fabric_breakthrough,
        report.status,
        report.claim.unwrap_or("none"),
    );
    for reason in &report.reasons {
        eprintln!("[mega_geometry_acceptance] {reason}");
    }
    Ok(passes)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("[mega_geometry_acceptance] error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(frame_ms_p95: f64) -> ProductComposition {
        ProductComposition {
            schema: 1,
            outcome: "win".into(),
            winner: Some("city_hybrid".into()),
            scene_hash: "1".repeat(64),
            camera_hash: "2".repeat(64),
            render_config_hash: "3".repeat(64),
            reference_witness_sha256: "4".repeat(64),
            city_hybrid_witness_sha256: "5".repeat(64),
            reference_capture_sha256: "6".repeat(64),
            city_hybrid_capture_sha256: "7".repeat(64),
            visual_inspection_sha256: "8".repeat(64),
            city_hybrid_total_ms_p95: frame_ms_p95,
            p50_improvement_pct: 0.0,
            p95_improvement_pct: 3.0,
            reasons: Vec::new(),
        }
    }

    #[test]
    fn product_gate_requires_amd_30_fps_at_p95() {
        let mut reasons = Vec::new();
        assert!(!validate_product(&product(40.0), &mut reasons));
        assert!(reasons.iter().any(|reason| reason.contains("30 fps")));
    }

    #[test]
    fn product_gate_accepts_clean_runtime_win() {
        let mut reasons = Vec::new();
        assert!(
            validate_product(&product(30.0), &mut reasons),
            "{reasons:?}"
        );
    }

    #[test]
    fn serialized_breakthrough_cannot_override_raw_recomposition() {
        let mut reasons = Vec::new();
        assert!(!claim_agrees(
            "CUDA",
            "geometry-program breakthrough",
            Some("city_geometry_program_breakthrough"),
            None,
            "city_geometry_program_breakthrough",
            &mut reasons,
        ));
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("producer/composer disagreement"))
        );
    }

    #[test]
    fn recomputed_breakthrough_must_reach_the_expected_claim() {
        let mut reasons = Vec::new();
        assert!(!claim_agrees(
            "Vulkan",
            "visibility-fabric breakthrough",
            None,
            None,
            "certified_visibility_fabric",
            &mut reasons,
        ));
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("raw fixture evidence"))
        );
    }

    #[test]
    fn evidence_identity_requires_real_digests_and_revisions() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256("unavailable"));
        assert!(is_git_revision("0123456"));
        assert!(!is_git_revision("unknown"));
    }
}
