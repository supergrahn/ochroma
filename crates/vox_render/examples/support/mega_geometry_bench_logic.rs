//! Pure result-processing, schema, and verdict logic for the three-mode City
//! MegaGeometry benchmark (`triangle`, `reference`, `city_hybrid`).
//!
//! This module has **zero** GPU / Spectra / OptiX dependency: it takes measured
//! records as plain data and computes the evidence lines, the fixture outcomes,
//! the mechanically-limited overall verdict, and each scoped breakthrough claim.
//! Because it is pure Rust it compiles and is unit-tested on any host, so the
//! verdict logic is proven locally while the *hardware* run that produces the
//! records stays permission-gated.
//!
//! It is shared verbatim (via `#[path]`) by:
//!   - `examples/mega_geometry_bench.rs`     — the hardware harness, and
//!   - `tests/mega_geometry_bench_contract.rs` — the local contract tests.
//!
//! ## FAIL-CLOSED LAW
//!
//! Every gate defaults in the FAILING direction. Unmeasured evidence is an
//! explicit `None` / `Unavailable` that SUPPRESSES the broad claim — it is never
//! a passing constant. A win requires every gate to be present AND passing. This
//! is why the harness (which lacks most integration hooks today) can never print
//! `winner=city_hybrid` from synthetic data, while the tests (which simulate a
//! fully-wired future) can exercise both the winning and every suppressing path.
//!
//! The design contract it encodes lives in
//! `docs/superpowers/specs/2026-07-20-city-megageometry-design.md` §4/§6.15 and
//! the plan's "Common Evidence Rules".

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Deterministic identity hashing (FNV-1a 64). No external crate so the example,
// which only has the crate's normal deps, can hash identically to the test.
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf29ce4_84222325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_mix(FNV_OFFSET, bytes)
}

fn fnv1a_mix(hash: u64, bytes: &[u8]) -> u64 {
    let mut h = hash;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

pub const MODES: [&str; 3] = ["triangle", "reference", "city_hybrid"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Triangle,
    Reference,
    CityHybrid,
}

/// Only the candidate consumes Ochroma's derived runtime representation.
///
/// Triangle is the authoritative source-GAS oracle. Reference is the complete
/// source mesh lowered through the selected native cluster/CLAS baseline.
/// Keeping this decision in the shared contract prevents the hardware harness
/// from accidentally comparing MegaGeometry with itself.
pub const fn mode_uses_runtime_megageometry(mode: Mode) -> bool {
    matches!(mode, Mode::CityHybrid)
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Triangle => "triangle",
            Mode::Reference => "reference",
            Mode::CityHybrid => "city_hybrid",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "triangle" => Some(Mode::Triangle),
            "reference" => Some(Mode::Reference),
            "city_hybrid" => Some(Mode::CityHybrid),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeVariant {
    PortableSubgroup,
    CooperativeMatrix,
}

impl ComputeVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            ComputeVariant::PortableSubgroup => "portable_subgroup",
            ComputeVariant::CooperativeMatrix => "cooperative_matrix",
        }
    }
}

// ---------------------------------------------------------------------------
// Fixtures (§6.15). Roles decide how a fixture can ever be scored.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureRole {
    /// `Auto` may deliberately retain the validated triangle path; `city_hybrid`
    /// must not regress by more than 2% vs `reference`. Never a "win".
    Parity,
    /// Absolute correctness gate; produces no winner.
    CorrectnessOnly,
    /// A performance fixture `city_hybrid` is required to win outright.
    TargetWin,
}

/// A fixture is a real checked-in asset family plus a workload shape.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureSpec {
    pub name: String,
    pub role: FixtureRole,
    /// Real, checked-in asset id consumed by this fixture (never synthetic-only).
    pub asset_id: String,
    pub family: String,
    pub instances: u64,
    pub seed: u64,
    /// When true, the mixed live-city witness authorizes this fixture through the
    /// Urban Horizon present path (a hook OUTSIDE vox_render). The vox_render
    /// harness can never certify a win for such a fixture on its own.
    pub authorizing_witness_external: bool,
}

impl FixtureSpec {
    /// Identity hash over the declared fixture identity. This certifies the
    /// fixture *definition*, NOT its rendered asset bytes (see
    /// `FixtureReport::asset_content_hash`, which is `None` until the real asset
    /// content is loaded and hashed).
    pub fn identity_hash(&self) -> u64 {
        let mut h = FNV_OFFSET;
        h = fnv1a_mix(h, self.name.as_bytes());
        h = fnv1a_mix(h, &[self.role as u8]);
        h = fnv1a_mix(h, self.asset_id.as_bytes());
        h = fnv1a_mix(h, self.family.as_bytes());
        h = fnv1a_mix(h, &self.instances.to_le_bytes());
        h = fnv1a_mix(h, &self.seed.to_le_bytes());
        h
    }

    pub fn identity_hash_hex(&self) -> String {
        format!("{:016x}", self.identity_hash())
    }
}

/// The seven required fixtures (§6.15), each bound to a real checked-in asset id.
pub fn required_fixtures() -> Vec<FixtureSpec> {
    vec![
        FixtureSpec {
            name: "static_meridian".into(),
            role: FixtureRole::Parity,
            asset_id: "city.res_high.l5.8x8.meridian_house_01".into(),
            family: "repeated_highrise".into(),
            instances: 4_096,
            seed: 0x5EED_0001,
            authorizing_witness_external: false,
        },
        FixtureSpec {
            name: "multi_material".into(),
            role: FixtureRole::CorrectnessOnly,
            asset_id: "city.res_high.l5.8x8.meridian_house_01".into(),
            family: "repeated_highrise".into(),
            instances: 1,
            seed: 0x5EED_0002,
            authorizing_witness_external: false,
        },
        FixtureSpec {
            name: "omm_foliage".into(),
            role: FixtureRole::Parity,
            asset_id: "ph.fir_tree_01".into(),
            family: "opacity_micromap_foliage".into(),
            instances: 65_536,
            seed: 0x5EED_0003,
            authorizing_witness_external: false,
        },
        FixtureSpec {
            name: "changing_geometry".into(),
            role: FixtureRole::TargetWin,
            asset_id: "city.res_high.l5.8x8.meridian_house_01".into(),
            family: "repeated_highrise".into(),
            instances: 2_048,
            seed: 0x5EED_0004,
            authorizing_witness_external: false,
        },
        FixtureSpec {
            name: "streamed_lod".into(),
            role: FixtureRole::TargetWin,
            asset_id: "city.res_low.l1.2x2.cottage_01".into(),
            family: "detached_residential".into(),
            instances: 32_768,
            seed: 0x5EED_0005,
            authorizing_witness_external: false,
        },
        FixtureSpec {
            name: "localized_updates".into(),
            role: FixtureRole::TargetWin,
            asset_id: "city.res_low.l1.2x2.cottage_01".into(),
            family: "detached_residential".into(),
            instances: 1_000_000,
            seed: 0x5EED_0006,
            authorizing_witness_external: false,
        },
        FixtureSpec {
            name: "mixed_city".into(),
            role: FixtureRole::TargetWin,
            asset_id: "urban_horizon.live_city".into(),
            family: "mixed_live_city".into(),
            instances: 600_000,
            seed: 0x5EED_0007,
            // The mixed live city is authorized ONLY by the Urban Horizon present
            // witness; the vox_render harness never certifies it.
            authorizing_witness_external: true,
        },
    ]
}

/// The two real finished-object structural families the ray-native / Fabric
/// claims require.
pub const REAL_ASSET_FAMILIES: [&str; 2] = ["repeated_highrise", "detached_residential"];

/// Whether the harness can prove `city_hybrid` resolves to a DISTINCT execution
/// representation from `reference`. Only city_hybrid consumes the
/// finished-object-derived MegaGeometry payload and residual triangles;
/// `reference` retains the complete source mesh and selects native
/// cluster/CLAS acceleration. Scalar ray-native visibility is measured in a
/// separate calibration receipt.
pub const HARNESS_CITY_HYBRID_DISTINCT_EXECUTION: bool = true;

/// Whether the forbidden CPU-ownership counters are actually bridged/probed in
/// the harness's live path. This remains a pinned contract bit so removal of
/// the live snapshot bridge fails tests instead of turning zero-initialized
/// counters into evidence.
pub const HARNESS_CPU_OWNERSHIP_COUNTERS_PROBED: bool = true;

/// Hardware CLAS is mandatory only when the independently requested native RT
/// backend is OptiX. Vulkan KHR and Metal remain valid native triangle/
/// procedural acceleration backends; treating their lack of NVIDIA CLAS as a
/// silent fallback would make cross-vendor acceptance impossible by design.
pub fn requires_hardware_clas(requested_rt_backend: &str, mode: Mode) -> bool {
    requested_rt_backend == "optix" && matches!(mode, Mode::Reference | Mode::CityHybrid)
}

/// Max per-channel error for the cross-mode hit oracle (candidate vs the
/// `triangle` oracle frame).
pub const HIT_ORACLE_MAX_CHANNEL_ERROR: f32 = 2.0e-3;

/// Compute hit mismatches for one mode's frame against the `triangle` oracle
/// frame. Returns `None` (Unavailable) — never `Some(0)` — when there is no
/// oracle, or when either buffer is empty (a 0-from-0 comparison is not
/// evidence). The triangle mode is its own oracle (self-compare = 0).
pub fn hit_oracle_mismatches(
    is_triangle_mode: bool,
    has_triangle_oracle: bool,
    candidate: &[f32],
    oracle: Option<&[f32]>,
) -> Option<u64> {
    if is_triangle_mode {
        return Some(0);
    }
    let oracle = oracle?;
    if !has_triangle_oracle {
        return None;
    }
    if candidate.is_empty() || oracle.is_empty() {
        // A comparison over empty buffers proves nothing.
        return None;
    }
    if candidate.len() != oracle.len() {
        return Some(candidate.len().max(oracle.len()) as u64);
    }
    Some(
        oracle
            .iter()
            .zip(candidate.iter())
            .filter(|(o, c)| (**o - **c).abs() > HIT_ORACLE_MAX_CHANNEL_ERROR)
            .count() as u64,
    )
}

/// The `prim_id` sentinel written for a primary-ray MISS (`asfloat(0xFFFFFFFF)`
/// on the device → this `u32` on readback). Distinct from any valid triangle
/// index (including 0), so hit-vs-miss divergence is detectable.
pub const PROV_MISS_PRIM: u32 = 0xFFFF_FFFF;

/// Max barycentric delta for the provenance UV compare (float interpolation).
pub const PROV_UV_TOL: f32 = 1.0e-3;

/// Relative hit-distance (`t`) tolerance for DEPTH-GATING a prim/material edge
/// tie. Two acceleration structures grazing a shared triangle edge resolve a
/// different FACE at the SAME surface point → identical `t`. A genuinely
/// different surface has a `t` that differs by far more than 0.1% of depth, so
/// this tolerates the measure-zero edge tie while still catching real geometry
/// divergence. Scaled by `max(|t|, 1.0)` so it is meaningful at all depths.
pub const PROV_DEPTH_REL_TOL: f32 = 1.0e-3;

/// Compare candidate first-hit provenance against the `triangle` oracle and
/// return `(hit_mismatches, material_mismatches, uv_mismatches)`.
///
/// `triangle` mode is its OWN oracle → `(Some(0), Some(0), Some(0))`. Returns
/// all-`None` (Unavailable — never `Some(0)` conjured from nothing) when there
/// is no oracle or the buffers are empty / mismatched length. Semantics (per
/// pixel, primary ray, `g_hit_prim_id` is a GLOBAL `g_triangles` index in every
/// HW-RT mode so it is directly comparable across triangle-GAS and CLAS):
/// - **hit**: HIT/MISS status differs, OR both hit a DIFFERENT prim_id AT A
///   DIFFERENT SURFACE (`t` differs beyond [`PROV_DEPTH_REL_TOL`]). A different
///   prim at the SAME `t` is a measure-zero silhouette-edge face tie between two
///   acceleration structures — NOT a geometry error — and is not counted.
/// - **material**: both hit, different global material_id, AT A DIFFERENT
///   SURFACE (same depth-gate — an edge face tie legitimately changes the face's
///   material without being an error).
/// - **uv**: both hit the SAME prim but barycentrics differ beyond tolerance.
#[allow(clippy::type_complexity)]
pub fn provenance_mismatches(
    is_triangle_mode: bool,
    oracle: Option<(&[u32], &[u32], &[f32], &[f32], &[f32])>,
    cand: (&[u32], &[u32], &[f32], &[f32], &[f32]),
) -> (Option<u64>, Option<u64>, Option<u64>) {
    let (cp, cm, cu, cv, cd) = cand;
    if is_triangle_mode {
        // Self-compare = exact. A zero-length buffer is "nothing measured".
        if cp.is_empty() {
            return (None, None, None);
        }
        return (Some(0), Some(0), Some(0));
    }
    let Some((op, om, ou, ov, od)) = oracle else {
        return (None, None, None);
    };
    let n = cp.len();
    if n == 0 || op.len() != n {
        return (None, None, None);
    }
    // Every channel (oracle + candidate) must agree in length or it is not a
    // valid per-pixel comparison.
    if cm.len() != n
        || cu.len() != n
        || cv.len() != n
        || cd.len() != n
        || om.len() != n
        || ou.len() != n
        || ov.len() != n
        || od.len() != n
    {
        return (None, None, None);
    }
    let (mut hit, mut mat, mut uv) = (0u64, 0u64, 0u64);
    for i in 0..n {
        let o_hit = op[i] != PROV_MISS_PRIM;
        let c_hit = cp[i] != PROV_MISS_PRIM;
        if o_hit != c_hit {
            hit += 1;
            continue; // hit-vs-miss divergence; material/uv not meaningful
        }
        if !o_hit {
            continue; // both miss — sky; nothing to compare
        }
        // Both hit. Depth-gate a differing prim/material: same surface point
        // (identical t) ⇒ a measure-zero edge FACE tie, not a divergence.
        let same_surface = (cd[i] - od[i]).abs() <= PROV_DEPTH_REL_TOL * od[i].abs().max(1.0);
        if op[i] != cp[i] && !same_surface {
            hit += 1;
        }
        if om[i] != cm[i] && !same_surface {
            mat += 1;
        }
        if op[i] == cp[i]
            && ((cu[i] - ou[i]).abs() > PROV_UV_TOL || (cv[i] - ov[i]).abs() > PROV_UV_TOL)
        {
            uv += 1;
        }
    }
    (Some(hit), Some(mat), Some(uv))
}

/// True only if `names` contains exactly the seven required fixtures. A verdict
/// computed over a subset is not a suite verdict.
pub fn all_required_fixtures_present(names: &[String]) -> bool {
    let required = required_fixtures();
    if names.len() != required.len() {
        return false;
    }
    required.iter().all(|r| names.iter().any(|n| n == &r.name))
}

// ---------------------------------------------------------------------------
// Timed-run configuration + validation (release, 60 warmup, >=120 timed, 1 spp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TimedRunConfig {
    pub release_build: bool,
    pub warmup_frames: u32,
    pub timed_frames: u32,
    pub spp: u32,
    pub identical_present_path: bool,
}

pub const MIN_WARMUP_FRAMES: u32 = 60;
pub const MIN_TIMED_FRAMES: u32 = 120;

impl TimedRunConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.release_build {
            return Err("timed runs require a release build".into());
        }
        if self.warmup_frames < MIN_WARMUP_FRAMES {
            return Err(format!(
                "warmup {} < required {}",
                self.warmup_frames, MIN_WARMUP_FRAMES
            ));
        }
        if self.timed_frames < MIN_TIMED_FRAMES {
            return Err(format!(
                "timed {} < required {}",
                self.timed_frames, MIN_TIMED_FRAMES
            ));
        }
        if self.spp != 1 {
            return Err(format!("present path must be 1 spp, got {}", self.spp));
        }
        if !self.identical_present_path {
            return Err("compared modes must share camera/resolution/shaders".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU ownership counters.
//
// FAIL-CLOSED CAVEAT: these counters are only trustworthy if they were actually
// probed this run. Today vox_render's `geometry_cpu_ownership` snapshot is
// constructed `default()` and NO live path bridges Spectra's probed
// `geometry_cpu_contract` counters into it, so the harness sets
// `counters_probed = false` (Unavailable → suppress). When the coordinator
// wires that bridge, the flag flips to true and the zeros become real evidence.
// The submit-time and deadline-miss series are additionally `Option` because
// they need their own accessors. Nothing here maps zero measurements to a pass.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CpuCounters {
    pub render_thread_scene_visits: u64,
    pub visibility_node_visits: u64,
    pub construction_evaluations: u64,
    pub lod_decisions: u64,
    pub page_decisions: u64,
    pub eviction_decisions: u64,
    pub blocking_readbacks: u64,
    pub frame_allocations: u64,
    /// True only when the forbidden-counter probe was actually wired and read
    /// this run. When false the eight forbidden counters are Unavailable (their
    /// zeros are not evidence) and the CPU-ownership gate suppresses.
    pub counters_probed: bool,
    /// `None` until the IAS-submission worker exposes it — fails closed.
    pub host_deadline_misses: Option<u64>,
    /// `None` until a render-thread submit-time accessor exists — fails closed.
    pub render_thread_submit_ms_p95: Option<f64>,
    pub host_submit_calls: u64,
    /// `None` until the host-submission worker exposes p95 — fails closed.
    pub host_submit_ms_p95: Option<f64>,
}

impl Default for CpuCounters {
    fn default() -> Self {
        // Default is the ALL-MEASURED-CLEAN state used by tests to simulate a
        // fully-wired future: the probe is wired (`counters_probed = true`) and a
        // real host submission was recorded (`host_submit_calls = 1`). The harness
        // overrides these with the real (or `None`/false) values it actually has.
        CpuCounters {
            render_thread_scene_visits: 0,
            visibility_node_visits: 0,
            construction_evaluations: 0,
            lod_decisions: 0,
            page_decisions: 0,
            eviction_decisions: 0,
            blocking_readbacks: 0,
            frame_allocations: 0,
            counters_probed: true,
            host_deadline_misses: Some(0),
            render_thread_submit_ms_p95: Some(0.0),
            host_submit_calls: 1,
            host_submit_ms_p95: Some(0.0),
        }
    }
}

pub const RENDER_THREAD_SUBMIT_BUDGET_MS: f64 = 0.25;
pub const HOST_SUBMIT_BUDGET_MS: f64 = 0.50;

impl CpuCounters {
    pub fn any_forbidden_nonzero(&self) -> bool {
        self.render_thread_scene_visits != 0
            || self.visibility_node_visits != 0
            || self.construction_evaluations != 0
            || self.lod_decisions != 0
            || self.page_decisions != 0
            || self.eviction_decisions != 0
            || self.blocking_readbacks != 0
            || self.frame_allocations != 0
    }

    fn forbidden_vector(&self) -> [u64; 8] {
        [
            self.render_thread_scene_visits,
            self.visibility_node_visits,
            self.construction_evaluations,
            self.lod_decisions,
            self.page_decisions,
            self.eviction_decisions,
            self.blocking_readbacks,
            self.frame_allocations,
        ]
    }

    /// Gate the submit-time budgets. `None` (unmeasured) fails closed, and zero
    /// recorded host-submit calls is "nothing measured" — never a pass.
    pub fn submit_budget_reason(&self) -> Option<String> {
        match self.render_thread_submit_ms_p95 {
            None => return Some("render-thread submit time unavailable".into()),
            Some(v) if v > RENDER_THREAD_SUBMIT_BUDGET_MS => {
                return Some(format!(
                    "render-thread submit {v:.3}ms > {RENDER_THREAD_SUBMIT_BUDGET_MS}ms"
                ));
            }
            _ => {}
        }
        // Zero host-submit calls means nothing was recorded; a stored 0.0 is a
        // measured-pass from zero measurements and must not certify the budget.
        if self.host_submit_calls == 0 {
            return Some("host submit time not recorded (zero submit calls)".into());
        }
        match self.host_submit_ms_p95 {
            None => return Some("host submit time unavailable".into()),
            Some(v) if v > HOST_SUBMIT_BUDGET_MS => {
                return Some(format!("host submit {v:.3}ms > {HOST_SUBMIT_BUDGET_MS}ms"));
            }
            _ => {}
        }
        match self.host_deadline_misses {
            None => return Some("host deadline-miss counter unavailable".into()),
            Some(v) if v != 0 => return Some(format!("{v} host deadline misses")),
            _ => {}
        }
        None
    }
}

/// A CPU sample at a given problem size for the scaling proof.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ScaleSample {
    pub total_units: u64,
    pub cpu: CpuCounters,
}

/// True if any forbidden CPU counter grows with problem size — a verdict abort.
pub fn cpu_cost_scales_with_size(small: &ScaleSample, large: &ScaleSample) -> bool {
    if large.total_units <= small.total_units {
        return false;
    }
    let s = small.cpu.forbidden_vector();
    let l = large.cpu.forbidden_vector();
    s.iter().zip(l.iter()).any(|(&a, &b)| b > a)
}

/// True if render-thread or host submission TIME grows with problem size. `None`
/// series are treated as "not scaling here" (their availability is gated
/// separately by `submit_budget_reason`).
pub fn submit_time_scales_with_size(small: &ScaleSample, large: &ScaleSample) -> bool {
    if large.total_units <= small.total_units {
        return false;
    }
    let grew = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => b > a * 1.10, // >10% growth with size
        _ => false,
    };
    grew(
        small.cpu.render_thread_submit_ms_p95,
        large.cpu.render_thread_submit_ms_p95,
    ) || grew(small.cpu.host_submit_ms_p95, large.cpu.host_submit_ms_p95)
}

// ---------------------------------------------------------------------------
// Backend resolution + honesty
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendResolution {
    /// The backend the enabled build feature REQUESTED (compile-time truth).
    pub requested_rt_backend: String,
    /// The backend the driver actually built.
    pub actual_rt_backend: String,
    pub requested_compute: ComputeVariant,
    pub actual_compute: ComputeVariant,
    pub available: bool,
    pub silently_downgraded: bool,
    pub cpu_fallbacks: u64,
    /// Native CLAS builds completed for this measured mode. `None` means the
    /// backend did not expose a trustworthy counter; zero is a measured zero.
    #[serde(default)]
    pub hardware_clas_builds: Option<u64>,
    /// A cluster/CLAS representation was requested but zero HARDWARE CLAS builds
    /// occurred — a derived cluster count is not CLAS evidence.
    pub cluster_requested_without_hardware_clas: bool,
}

impl BackendResolution {
    pub fn exit_code(&self) -> i32 {
        if !self.available
            || self.requested_rt_backend != self.actual_rt_backend
            || self.silently_downgraded
            || self.cpu_fallbacks != 0
            || self.cluster_requested_without_hardware_clas
        {
            1
        } else {
            0
        }
    }

    pub fn is_honest(&self) -> bool {
        self.exit_code() == 0
    }
}

// ---------------------------------------------------------------------------
// Prewarm / stats — carried as Option on ModeResult (None = unavailable).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PrewarmState {
    pub shaders_compiled_before_timed: bool,
    pub pipeline_created_in_timed_frame: bool,
    pub cache_repaired_in_timed_frame: bool,
}

impl PrewarmState {
    pub fn is_valid(&self) -> bool {
        self.shaders_compiled_before_timed
            && !self.pipeline_created_in_timed_frame
            && !self.cache_repaired_in_timed_frame
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct StatsCollection {
    pub collected_after_timed_frames: bool,
    pub inside_present_timing: bool,
}

impl StatsCollection {
    pub fn is_valid(&self) -> bool {
        self.collected_after_timed_frames && !self.inside_present_timing
    }
}

// ---------------------------------------------------------------------------
// Pipeline / capability cache key
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheKey {
    pub capability_hash: [u8; 32],
    pub shader_abi: [u8; 32],
    pub driver_id: u64,
    #[serde(alias = "cook_schema")]
    pub runtime_geometry_schema: u32,
}

pub fn cache_is_valid(loaded: &CacheKey, active: &CacheKey) -> bool {
    loaded == active
}

// ---------------------------------------------------------------------------
// VRAM breakdown — peak MUST include native args + build scratch. Components are
// `Option` because they need per-backend accessors; `None` => peak unavailable
// => a win cannot be certified.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VramBreakdown {
    pub resident_mb: f64,
    pub as_mb: Option<f64>,
    pub page_mb: Option<f64>,
    pub native_args_mb: Option<f64>,
    pub scratch_mb: Option<f64>,
}

impl VramBreakdown {
    /// `None` if any component is unmeasured — a partial peak cannot certify a
    /// VRAM win.
    pub fn peak_geometry_mb(&self) -> Option<f64> {
        Some(
            self.resident_mb
                + self.as_mb?
                + self.page_mb?
                + self.native_args_mb?
                + self.scratch_mb?,
        )
    }
}

// ---------------------------------------------------------------------------
// Correctness — every component is Option. `None` => not verified => suppress.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Correctness {
    pub source_primitives: u64,
    pub mapped_primitives: u64,
    pub material_mismatches: Option<u64>,
    pub uv_mismatches: Option<u64>,
    /// From the cross-mode hit oracle. `None` when the `triangle` oracle mode was
    /// absent (no comparison performed) — never reported as zero mismatches.
    pub hit_mismatches: Option<u64>,
}

impl Correctness {
    /// `None` if any component is unmeasured; otherwise `Some(true)` only when
    /// mapping is exact and every mismatch count is zero.
    pub fn is_exact(&self) -> Option<bool> {
        match (
            self.material_mismatches,
            self.uv_mismatches,
            self.hit_mismatches,
        ) {
            (Some(m), Some(u), Some(h)) => {
                Some(self.mapped_primitives == self.source_primitives && m == 0 && u == 0 && h == 0)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GeometryStage {
    pub build_ms_p50: f64,
    pub build_ms_p95: f64,
    pub update_ms_p50: f64,
    pub update_ms_p95: f64,
    /// `None` until the native-lowering timer exists.
    pub native_lowering_ms_p95: Option<f64>,
    pub trace_ms_p50: f64,
    pub trace_ms_p95: f64,
    pub geometry_stage_ms_p95: f64,
    pub frame_ms_p50: f64,
    pub frame_ms_p95: f64,
    pub traced_triangles: u64,
    /// `None` until the pager exposes required-page misses — fails closed.
    pub required_page_misses: Option<u64>,
}

/// The complete measured record for one (mode, fixture) run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModeResult {
    pub mode: Mode,
    pub compute: ComputeVariant,
    /// Raw identities emitted by this backend. Equality is decided only by an
    /// artifact composer that has independent backend reports.
    #[serde(default)]
    pub packet_abi_hash: Option<String>,
    #[serde(default)]
    pub control_hash: Option<String>,
    #[serde(default)]
    pub capability_fingerprint: Option<String>,
    /// GPU-authored visibility-fabric receipt captured only after the timed
    /// interval. `None` means the mode had no active fabric or no diagnostic
    /// readback was available.
    #[serde(default)]
    pub visibility_fabric: Option<VisibilityQueueCounters>,
    /// Serialized GPU-event/timestamp calibration frames collected strictly
    /// after the ordinary timed interval. `None` is unavailable, never zero.
    #[serde(default)]
    pub visibility_profile: Option<VisibilityKernelProfile>,
    /// Completed-frame GPU-authored temporal correspondence receipt.
    #[serde(default)]
    pub temporal_reuse: Option<TemporalReuseEvidence>,
    /// Fine geometry AS builds observed across a bounded program-parameter
    /// update. Instance-only IAS/TLAS refits do not populate this field.
    #[serde(default)]
    pub parameter_update_fine_as_builds: Option<u64>,
    pub correctness: Correctness,
    pub stage: GeometryStage,
    pub vram: VramBreakdown,
    pub cpu: CpuCounters,
    /// `None` until a prewarm accessor proves no in-frame pipeline creation.
    pub prewarm: Option<PrewarmState>,
    pub stats: Option<StatsCollection>,
    pub backend: BackendResolution,
    pub unexpected_fallback: bool,
}

impl ModeResult {
    /// Construct a deliberately non-authorizing record for a mode that this
    /// process did not execute. Numeric fields are placeholders only; every
    /// evidence-bearing optional field and probe is unavailable, and backend
    /// resolution fails closed. This is used for externally witnessed product
    /// fixtures so a microbenchmark can never replace them with proxy geometry.
    pub fn unavailable(mode: Mode, requested_rt_backend: impl Into<String>) -> Self {
        ModeResult {
            mode,
            compute: ComputeVariant::PortableSubgroup,
            packet_abi_hash: None,
            control_hash: None,
            capability_fingerprint: None,
            visibility_fabric: None,
            visibility_profile: None,
            temporal_reuse: None,
            parameter_update_fine_as_builds: None,
            correctness: Correctness {
                source_primitives: 0,
                mapped_primitives: 0,
                material_mismatches: None,
                uv_mismatches: None,
                hit_mismatches: None,
            },
            stage: GeometryStage {
                build_ms_p50: 0.0,
                build_ms_p95: 0.0,
                update_ms_p50: 0.0,
                update_ms_p95: 0.0,
                native_lowering_ms_p95: None,
                trace_ms_p50: 0.0,
                trace_ms_p95: 0.0,
                geometry_stage_ms_p95: 0.0,
                frame_ms_p50: 0.0,
                frame_ms_p95: 0.0,
                traced_triangles: 0,
                required_page_misses: None,
            },
            vram: VramBreakdown {
                resident_mb: 0.0,
                as_mb: None,
                page_mb: None,
                native_args_mb: None,
                scratch_mb: None,
            },
            cpu: CpuCounters {
                render_thread_scene_visits: 0,
                visibility_node_visits: 0,
                construction_evaluations: 0,
                lod_decisions: 0,
                page_decisions: 0,
                eviction_decisions: 0,
                blocking_readbacks: 0,
                frame_allocations: 0,
                counters_probed: false,
                host_deadline_misses: None,
                render_thread_submit_ms_p95: None,
                host_submit_calls: 0,
                host_submit_ms_p95: None,
            },
            prewarm: None,
            stats: None,
            backend: BackendResolution {
                requested_rt_backend: requested_rt_backend.into(),
                actual_rt_backend: "unavailable".into(),
                requested_compute: ComputeVariant::PortableSubgroup,
                actual_compute: ComputeVariant::PortableSubgroup,
                available: false,
                silently_downgraded: false,
                cpu_fallbacks: 0,
                hardware_clas_builds: None,
                cluster_requested_without_hardware_clas: mode != Mode::Triangle,
            },
            unexpected_fallback: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VisibilityQueueCounters {
    pub domains_seeded: u64,
    pub rays_certified_miss: u64,
    pub rays_certified_hit: u64,
    pub domains_split: u64,
    pub explicit_rays: u64,
    pub packet_rays: u64,
    pub packet_active_lanes: u64,
    pub packet_total_lanes: u64,
    pub scalar_refinement_rays: u64,
    pub native_residual_rays: u64,
    pub failed_certificates: u64,
    pub queue_overflow_mask: u64,
    pub cpu_scheduling: u64,
    pub correctness_mismatches: u64,
    pub source_ray_capacity: u64,
    pub live_source_rays: u64,
}

/// Complete GPU visibility-stage cost assembled from labeled dispatch events.
/// Each field is a p95 of per-frame stage sums, not a sum of unrelated
/// per-kernel percentiles.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VisibilityKernelProfile {
    pub measured_frames: u32,
    pub certification_ms_p95: f64,
    pub subdivision_ms_p95: f64,
    pub routing_queue_ms_p95: f64,
    pub packet_intersection_ms_p95: f64,
    pub exact_refinement_ms_p95: f64,
    pub ordinary_native_ms_p95: f64,
    pub native_residual_ms_p95: f64,
    pub hit_merge_ms_p95: f64,
    pub fixed_empty_launch_ms_p95: f64,
    pub complete_visibility_ms_p95: f64,
    pub includes_fixed_empty_launches: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TemporalReuseEvidence {
    pub relevant_surface_pixels: u64,
    pub invalidated_surface_pixels: u64,
    #[serde(default)]
    pub no_map_invalidated_surface_pixels: u64,
    /// Sum of GPU-authored detail-cut changes across the untimed calibration.
    #[serde(default)]
    pub lod_cut_changes: u64,
}

impl VisibilityKernelProfile {
    pub fn is_measured(self) -> bool {
        self.measured_frames > 0
            && [
                self.certification_ms_p95,
                self.subdivision_ms_p95,
                self.routing_queue_ms_p95,
                self.packet_intersection_ms_p95,
                self.exact_refinement_ms_p95,
                self.ordinary_native_ms_p95,
                self.native_residual_ms_p95,
                self.hit_merge_ms_p95,
                self.fixed_empty_launch_ms_p95,
                self.complete_visibility_ms_p95,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
            && self.complete_visibility_ms_p95 > 0.0
    }
}

impl VisibilityQueueCounters {
    pub fn fabric_line(self, mode: Mode) -> FabricLine {
        // A common-hit certificate narrows a domain to one program, but that
        // program is still intersected per ray and is therefore already part
        // of `packet_rays`. Only guaranteed misses avoid per-ray traversal.
        let certified_without_per_ray_traversal = self.rays_certified_miss;
        let structured_rays = certified_without_per_ray_traversal.saturating_add(self.packet_rays);
        FabricLine {
            mode: mode.as_str().into(),
            input_rays: (self.live_source_rays > 0).then_some(self.live_source_rays),
            structured_rays: Some(structured_rays.min(self.live_source_rays)),
            certified_rays: Some(certified_without_per_ray_traversal.min(self.live_source_rays)),
            subdivided_domains: Some(self.domains_split),
            explicit_rays: Some(self.explicit_rays),
            active_packet_lanes: if self.packet_total_lanes > 0 {
                Some(self.packet_active_lanes as f64 / self.packet_total_lanes as f64)
            } else {
                None
            },
            routing_ms_p95: None,
            certification_ms_p95: None,
            packet_ms_p95: None,
            refinement_ms_p95: None,
            residual_rt_ms_p95: None,
            merge_ms_p95: None,
            queue_overflows: Some(self.queue_overflow_mask),
            cpu_scheduling: Some(self.cpu_scheduling),
            hit_hash: "unavailable".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture outcome
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum FixtureOutcome {
    Parity {
        regression_pct: f64,
    },
    CorrectnessOnly {
        mismatches: u64,
    },
    Win {
        winner: Mode,
    },
    Loss {
        reason: String,
    },
    CorrectnessFailure {
        reason: String,
    },
    /// Evidence required to score this fixture is not available — suppresses.
    Unavailable {
        reason: String,
    },
}

impl FixtureOutcome {
    pub fn is_win(&self) -> bool {
        matches!(self, FixtureOutcome::Win { .. })
    }
    pub fn suppresses(&self) -> bool {
        matches!(
            self,
            FixtureOutcome::Loss { .. }
                | FixtureOutcome::CorrectnessFailure { .. }
                | FixtureOutcome::Unavailable { .. }
        )
    }
    pub fn is_hard_abort(&self) -> bool {
        matches!(
            self,
            FixtureOutcome::CorrectnessFailure { .. } | FixtureOutcome::Unavailable { .. }
        )
    }
}

pub const PARITY_REGRESSION_BUDGET_PCT: f64 = 2.0;

/// Evaluate one fixture. `distinct_execution` asserts that `city_hybrid`
/// resolved to a DIFFERENT execution representation than `reference`; when it is
/// false a target fixture cannot be a win (two identical configs timed on noise
/// is not a system win).
pub fn evaluate_fixture(
    role: FixtureRole,
    spec_authorizing_external: bool,
    distinct_execution: bool,
    hybrid: &ModeResult,
    reference: &ModeResult,
) -> FixtureOutcome {
    // Correctness is absolute and must be VERIFIED (not merely unmeasured).
    match hybrid.correctness.is_exact() {
        None => {
            return FixtureOutcome::Unavailable {
                reason: "city_hybrid correctness oracle unavailable".into(),
            };
        }
        Some(false) => {
            return FixtureOutcome::CorrectnessFailure {
                reason: "city_hybrid hit/material/uv/mapping mismatch".into(),
            };
        }
        Some(true) => {}
    }
    match reference.correctness.is_exact() {
        None => {
            return FixtureOutcome::Unavailable {
                reason: "reference correctness oracle unavailable".into(),
            };
        }
        Some(false) => {
            return FixtureOutcome::CorrectnessFailure {
                reason: "reference hit/material/uv/mapping mismatch".into(),
            };
        }
        Some(true) => {}
    }

    match role {
        FixtureRole::CorrectnessOnly => FixtureOutcome::CorrectnessOnly { mismatches: 0 },
        FixtureRole::Parity => {
            if hybrid.unexpected_fallback {
                return FixtureOutcome::Loss {
                    reason: "unexpected fallback on parity fixture".into(),
                };
            }
            let base = reference.stage.geometry_stage_ms_p95;
            let regression_pct = if base > 0.0 {
                (hybrid.stage.geometry_stage_ms_p95 / base - 1.0) * 100.0
            } else {
                0.0
            };
            if regression_pct > PARITY_REGRESSION_BUDGET_PCT {
                FixtureOutcome::Loss {
                    reason: format!(
                        "parity regression {regression_pct:.2}% > {PARITY_REGRESSION_BUDGET_PCT}%"
                    ),
                }
            } else {
                FixtureOutcome::Parity { regression_pct }
            }
        }
        FixtureRole::TargetWin => {
            if spec_authorizing_external {
                return FixtureOutcome::Unavailable {
                    reason: "authorizing witness is the Urban Horizon present path, not vox_render"
                        .into(),
                };
            }
            if !distinct_execution {
                return FixtureOutcome::Unavailable {
                    reason: "city_hybrid is not yet a distinct execution path from reference"
                        .into(),
                };
            }
            if hybrid.unexpected_fallback {
                return FixtureOutcome::Loss {
                    reason: "unexpected fallback on target fixture".into(),
                };
            }
            let (hp, rp) = match (
                hybrid.vram.peak_geometry_mb(),
                reference.vram.peak_geometry_mb(),
            ) {
                (Some(h), Some(r)) => (h, r),
                _ => {
                    return FixtureOutcome::Unavailable {
                        reason: "peak geometry VRAM breakdown unavailable".into(),
                    };
                }
            };
            let faster = hybrid.stage.geometry_stage_ms_p95 < reference.stage.geometry_stage_ms_p95;
            let no_worse_vram = hp <= rp;
            if faster && no_worse_vram {
                FixtureOutcome::Win {
                    winner: Mode::CityHybrid,
                }
            } else if !faster {
                FixtureOutcome::Loss {
                    reason: "city_hybrid geometry-stage p95 not below reference".into(),
                }
            } else {
                FixtureOutcome::Loss {
                    reason: "city_hybrid peak geometry VRAM above reference".into(),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Overall verdict
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureResult {
    pub spec: FixtureSpec,
    pub outcome: FixtureOutcome,
    pub hybrid: ModeResult,
    pub reference: ModeResult,
}

/// Global signals. Every field is `Option`; `None` is UNAVAILABLE and suppresses
/// the broad claim. No signal defaults in the passing direction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalSignals {
    pub packet_abi_match: Option<bool>,
    pub control_hash_match: Option<bool>,
    pub adapter_overhead_pct: Option<f64>,
    pub cpu_fallbacks: Option<u64>,
    pub fabric_cpu_scheduling: Option<u64>,
    pub cache_valid: Option<bool>,
    /// `city_hybrid` resolved to a distinct execution representation from
    /// `reference`. `None`/`Some(false)` suppresses (identical configs are not a
    /// system).
    pub mode_distinct: Option<bool>,
    pub scale_small: ScaleSample,
    pub scale_large: ScaleSample,
}

impl GlobalSignals {
    /// The fully-unavailable global state the harness uses today (fail closed).
    pub fn unavailable(scale_small: ScaleSample, scale_large: ScaleSample) -> Self {
        GlobalSignals {
            packet_abi_match: None,
            control_hash_match: None,
            adapter_overhead_pct: None,
            cpu_fallbacks: None,
            fabric_cpu_scheduling: None,
            cache_valid: None,
            mode_distinct: None,
            scale_small,
            scale_large,
        }
    }
}

pub const ADAPTER_OVERHEAD_BUDGET_PCT: f64 = 2.0;

/// Conservative measured upper bound for the CPU adapter/submit island as a
/// percentage of complete frame time. Every non-product mode must expose both
/// render-thread and host-submit p95 receipts; missing data remains
/// unavailable rather than becoming zero.
pub fn measured_adapter_overhead_pct(report: &BenchReport) -> Option<f64> {
    let mut worst = 0.0_f64;
    let mut measured = 0_u32;
    for fixture in report.fixtures.iter().filter(|fixture| {
        !required_fixtures()
            .iter()
            .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
    }) {
        for (_, mode) in mode_records(fixture) {
            let frame = mode.stage.frame_ms_p95;
            let render_submit = mode.cpu.render_thread_submit_ms_p95?;
            let host_submit = mode.cpu.host_submit_ms_p95?;
            if !frame.is_finite()
                || frame <= 0.0
                || !render_submit.is_finite()
                || render_submit < 0.0
                || !host_submit.is_finite()
                || host_submit < 0.0
            {
                return None;
            }
            worst = worst.max((render_submit + host_submit) * 100.0 / frame);
            measured = measured.saturating_add(1);
        }
    }
    (measured > 0).then_some(worst)
}

/// Validate one independently captured backend artifact without pretending it
/// can prove cross-backend equality or authorize the external Urban Horizon
/// product fixture. A clean raw artifact exits successfully and is eligible
/// for the final compositor; it still publishes no overall system claim.
pub fn raw_backend_validation_reasons(report: &BenchReport) -> Vec<String> {
    let mut reasons = Vec::new();
    let internal_specs: Vec<_> = required_fixtures()
        .into_iter()
        .filter(|spec| !spec.authorizing_witness_external)
        .collect();
    for spec in &internal_specs {
        let Some(fixture) = report
            .fixtures
            .iter()
            .find(|fixture| fixture.name == spec.name)
        else {
            reasons.push(format!("missing internal fixture {}", spec.name));
            continue;
        };
        if fixture.asset_content_hash.is_none() {
            reasons.push(format!("{} did not load a real finished asset", spec.name));
        }
        if matches!(
            fixture.outcome,
            FixtureOutcome::Unavailable { .. } | FixtureOutcome::CorrectnessFailure { .. }
        ) {
            reasons.push(format!(
                "{} has unavailable or incorrect evidence",
                spec.name
            ));
        }
        if fixture.triangle.is_none() {
            reasons.push(format!("{} has no triangle oracle mode", spec.name));
        }
        for (mode_name, mode) in mode_records(fixture) {
            if mode.correctness.is_exact() != Some(true) {
                reasons.push(format!(
                    "{}/{} correctness is not exact",
                    spec.name, mode_name
                ));
            }
            if mode.packet_abi_hash.is_none() || mode.capability_fingerprint.is_none() {
                reasons.push(format!(
                    "{}/{} identity is unavailable",
                    spec.name, mode_name
                ));
            }
            if mode_name == "city_hybrid" && mode.control_hash.is_none() {
                reasons.push(format!(
                    "{}/city_hybrid control hash is unavailable",
                    spec.name
                ));
            }
            if !mode.backend.is_honest() {
                reasons.push(format!(
                    "{}/{} backend is unavailable or dishonest",
                    spec.name, mode_name
                ));
            }
            if !mode.cpu.counters_probed
                || mode.cpu.any_forbidden_nonzero()
                || mode.cpu.submit_budget_reason().is_some()
            {
                reasons.push(format!("{}/{} CPU ownership failed", spec.name, mode_name));
            }
            if mode.prewarm.is_none_or(|prewarm| !prewarm.is_valid())
                || mode.stats.is_none_or(|stats| !stats.is_valid())
            {
                reasons.push(format!(
                    "{}/{} timing boundary failed",
                    spec.name, mode_name
                ));
            }
            if mode.stage.required_page_misses != Some(0) {
                reasons.push(format!(
                    "{}/{} has unavailable/required page misses",
                    spec.name, mode_name
                ));
            }
        }
        if REAL_ASSET_FAMILIES.contains(&fixture.family.as_str())
            && (fixture.hybrid.visibility_fabric.is_none()
                || fixture.hybrid.visibility_profile.is_none()
                || fixture.scalar_visibility_profile.is_none())
        {
            reasons.push(format!(
                "{} lacks scalar/fabric visibility calibration evidence",
                spec.name
            ));
        }
    }
    if report.global.cpu_fallbacks != Some(0) {
        reasons.push("raw backend CPU-fallback counter is unavailable/nonzero".into());
    }
    if report.global.fabric_cpu_scheduling != Some(0) {
        reasons.push("raw backend fabric CPU scheduling is unavailable/nonzero".into());
    }
    if report.global.cache_valid != Some(true) {
        reasons.push("raw backend cache/prewarm evidence failed".into());
    }
    if report.global.mode_distinct != Some(true) {
        reasons.push("raw backend mode-distinct evidence failed".into());
    }
    match measured_adapter_overhead_pct(report) {
        Some(value) if value <= ADAPTER_OVERHEAD_BUDGET_PCT => {}
        Some(value) => reasons.push(format!(
            "raw backend adapter overhead {value:.2}% exceeds {ADAPTER_OVERHEAD_BUDGET_PCT:.2}%"
        )),
        None => reasons.push("raw backend adapter overhead is unavailable".into()),
    }
    reasons
}

/// Sum required-page misses only across renderer-local fixtures. The
/// `mixed_city` fixture is authorized later by the Urban Horizon product
/// witness and has deliberately unavailable renderer-local modes; it must not
/// turn a complete raw backend receipt into a misleading `unavailable` line.
pub fn aggregate_required_page_misses(fixtures: &[FixtureReport]) -> Option<u64> {
    let mut total = 0_u64;
    for fixture in fixtures.iter().filter(|fixture| {
        !required_fixtures()
            .iter()
            .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
    }) {
        let mut modes = vec![&fixture.reference, &fixture.hybrid];
        modes.extend(fixture.triangle.iter());
        for mode in modes {
            total = total.checked_add(mode.stage.required_page_misses?)?;
        }
    }
    Some(total)
}

pub fn renderer_local_correctness_passes(fixtures: &[FixtureReport]) -> bool {
    fixtures
        .iter()
        .filter(|fixture| {
            !required_fixtures()
                .iter()
                .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
        })
        .all(|fixture| {
            fixture.hybrid.correctness.is_exact() == Some(true)
                && fixture.reference.correctness.is_exact() == Some(true)
                && fixture
                    .triangle
                    .as_ref()
                    .is_some_and(|mode| mode.correctness.is_exact() == Some(true))
        })
}

pub fn renderer_local_unexpected_fallbacks(fixtures: &[FixtureReport]) -> u64 {
    fixtures
        .iter()
        .filter(|fixture| {
            !required_fixtures()
                .iter()
                .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
        })
        .flat_map(|fixture| {
            let mut modes = vec![&fixture.reference, &fixture.hybrid];
            modes.extend(fixture.triangle.iter());
            modes
        })
        .filter(|mode| mode.unexpected_fallback)
        .count() as u64
}

pub const REQUIRED_TARGET_WINS: u32 = 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OverallVerdict {
    pub winner: Option<Mode>,
    pub losses: u32,
    pub target_wins: u32,
    pub claim: String,
    pub suppressed_reason: Option<String>,
    /// True when suppression is due to a hard evidence/correctness/CPU/portable/
    /// availability failure (as opposed to a purely measured performance loss).
    /// The harness exits nonzero on a hard abort.
    pub hard_abort: bool,
}

/// Compute the mechanically-limited overall verdict. Requires the complete
/// seven-fixture set, no losing fixture, verified correctness, zero forbidden CPU
/// (non-scaling), a satisfied portable/adapter/cache/mode-distinct contract, and
/// valid prewarm/stats. Anything unavailable suppresses and is a hard abort.
pub fn compute_overall(fixtures: &[FixtureResult], global: &GlobalSignals) -> OverallVerdict {
    compute_overall_with_external(fixtures, global, false)
}

/// Final compositor variant. The raw backend producer always passes `false`
/// through [`compute_overall`]; only the independent acceptance join may pass
/// `true` after validating the Urban Horizon live-product artifact.
pub fn compute_overall_with_external(
    fixtures: &[FixtureResult],
    global: &GlobalSignals,
    external_product_authorized: bool,
) -> OverallVerdict {
    let mut losses = 0u32;
    let mut target_wins = 0u32;
    let mut suppressed: Option<String> = None;
    let mut hard = false;

    let mut suppress = |reason: String, hard_flag: bool| {
        if suppressed.is_none() {
            suppressed = Some(reason);
        }
        if hard_flag {
            hard = true;
        }
    };

    // The verdict must cover the exact seven-fixture suite.
    let names: Vec<String> = fixtures.iter().map(|f| f.spec.name.clone()).collect();
    if !all_required_fixtures_present(&names) {
        suppress(
            "incomplete fixture set: the verdict requires all seven required fixtures".into(),
            true,
        );
    }

    for fr in fixtures {
        if fr.spec.authorizing_witness_external && external_product_authorized {
            if fr.spec.role == FixtureRole::TargetWin {
                target_wins = target_wins.saturating_add(1);
            }
            continue;
        }
        match &fr.outcome {
            FixtureOutcome::CorrectnessFailure { reason } => {
                suppress(
                    format!("correctness failure on {}: {reason}", fr.spec.name),
                    true,
                );
                losses += 1;
            }
            FixtureOutcome::Unavailable { reason } => {
                suppress(
                    format!("evidence unavailable on {}: {reason}", fr.spec.name),
                    true,
                );
            }
            FixtureOutcome::Loss { reason } => {
                suppress(format!("loss on {}: {reason}", fr.spec.name), false);
                losses += 1;
            }
            FixtureOutcome::Win { .. } => target_wins += 1,
            FixtureOutcome::Parity { .. } | FixtureOutcome::CorrectnessOnly { .. } => {}
        }

        for m in [&fr.hybrid, &fr.reference] {
            match &m.prewarm {
                None => suppress(
                    format!("prewarm evidence unavailable on {}", fr.spec.name),
                    true,
                ),
                Some(p) if !p.is_valid() => suppress(
                    format!("frame-time pipeline creation on {}", fr.spec.name),
                    true,
                ),
                _ => {}
            }
            match &m.stats {
                None => suppress(
                    format!("stats-timing evidence unavailable on {}", fr.spec.name),
                    true,
                ),
                Some(s) if !s.is_valid() => suppress(
                    format!("stats collected inside present timing on {}", fr.spec.name),
                    true,
                ),
                _ => {}
            }
            if !m.backend.is_honest() {
                suppress(
                    format!("silent backend/compute downgrade on {}", fr.spec.name),
                    true,
                );
            }
            if !m.cpu.counters_probed {
                suppress(
                    format!(
                        "CPU-ownership counters not probed on {} (zeros are not evidence)",
                        fr.spec.name
                    ),
                    true,
                );
            }
            if m.cpu.any_forbidden_nonzero() {
                suppress(
                    format!("forbidden CPU counter nonzero on {}", fr.spec.name),
                    true,
                );
            }
            if let Some(reason) = m.cpu.submit_budget_reason() {
                suppress(format!("{} on {}", reason, fr.spec.name), true);
            }
            match m.stage.required_page_misses {
                None => suppress(
                    format!("required-page-miss counter unavailable on {}", fr.spec.name),
                    true,
                ),
                Some(v) if v != 0 => {
                    suppress(format!("required page miss on {}", fr.spec.name), true)
                }
                _ => {}
            }
        }
    }

    // Global contracts — every `None` is a hard, fail-closed abort.
    match global.packet_abi_match {
        None => suppress("packet ABI comparison unavailable".into(), true),
        Some(false) => suppress("packet ABI mismatch across backends".into(), true),
        _ => {}
    }
    match global.control_hash_match {
        None => suppress("control-hash comparison unavailable".into(), true),
        Some(false) => suppress("integer control hash mismatch across backends".into(), true),
        _ => {}
    }
    match global.cpu_fallbacks {
        None => suppress("cpu-fallback counter unavailable".into(), true),
        Some(v) if v != 0 => suppress("CPU fallback path taken".into(), true),
        _ => {}
    }
    match global.adapter_overhead_pct {
        None => suppress("adapter overhead unavailable".into(), true),
        Some(v) if v > ADAPTER_OVERHEAD_BUDGET_PCT => suppress(
            format!("adapter overhead {v:.2}% > {ADAPTER_OVERHEAD_BUDGET_PCT}%"),
            true,
        ),
        _ => {}
    }
    match global.fabric_cpu_scheduling {
        None => suppress("fabric CPU-scheduling counter unavailable".into(), true),
        Some(v) if v != 0 => suppress("CPU scheduling of fabric queues".into(), true),
        _ => {}
    }
    match global.cache_valid {
        None => suppress("cache validity unavailable".into(), true),
        Some(false) => suppress("pipeline/calibration cache key mismatch".into(), true),
        _ => {}
    }
    match global.mode_distinct {
        None => suppress(
            "cannot prove city_hybrid is a distinct execution path from reference".into(),
            true,
        ),
        Some(false) => suppress(
            "city_hybrid is not a distinct execution path from reference".into(),
            true,
        ),
        _ => {}
    }
    if cpu_cost_scales_with_size(&global.scale_small, &global.scale_large) {
        suppress(
            "CPU cost scales with total instances/clusters/pages".into(),
            true,
        );
    }
    if submit_time_scales_with_size(&global.scale_small, &global.scale_large) {
        suppress(
            "render-thread/host submit time scales with total scene size".into(),
            true,
        );
    }

    let winner = if suppressed.is_none() && losses == 0 && target_wins == REQUIRED_TARGET_WINS {
        Some(Mode::CityHybrid)
    } else {
        None
    };

    let claim = if winner.is_some() {
        "better_city_geometry_system".to_string()
    } else {
        format!("winning_subset:target_wins={target_wins}:losses={losses}")
    };

    OverallVerdict {
        winner,
        losses,
        target_wins,
        claim,
        suppressed_reason: suppressed,
        hard_abort: hard,
    }
}

// ---------------------------------------------------------------------------
// Visible detail — only oracle-valid geometry counts
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VisibleDetail {
    pub total_detail_units: u64,
    pub oracle_valid_units: u64,
}

impl VisibleDetail {
    pub fn counted(&self) -> u64 {
        self.oracle_valid_units.min(self.total_detail_units)
    }
    pub fn detail_ratio_vs(&self, baseline_valid_units: u64) -> f64 {
        if baseline_valid_units == 0 {
            0.0
        } else {
            self.counted() as f64 / baseline_valid_units as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry-program breakthrough claim
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ProgramBreakthrough {
    pub repeated_storage_ratio: Option<f64>,
    pub mixed_storage_ratio: Option<f64>,
    pub fixed_budget_detail_ratio: Option<f64>,
    pub lod_temporal_invalidations_reduction: Option<f64>,
    pub full_frame_regression_pct: Option<f64>,
    pub losses: Option<u32>,
    pub axis_storage_bytes_win: Option<bool>,
    pub axis_peak_vram_win: Option<bool>,
    pub axis_geometry_stage_win: Option<bool>,
    pub axis_visible_detail_win: Option<bool>,
    pub axis_lod_temporal_win: Option<bool>,
}

impl ProgramBreakthrough {
    /// The fully-unavailable default: every claim gate fails.
    pub fn unavailable() -> Self {
        ProgramBreakthrough {
            repeated_storage_ratio: None,
            mixed_storage_ratio: None,
            fixed_budget_detail_ratio: None,
            lod_temporal_invalidations_reduction: None,
            full_frame_regression_pct: None,
            losses: None,
            axis_storage_bytes_win: None,
            axis_peak_vram_win: None,
            axis_geometry_stage_win: None,
            axis_visible_detail_win: None,
            axis_lod_temporal_win: None,
        }
    }

    pub fn independent_axis_wins(&self) -> u32 {
        [
            self.axis_storage_bytes_win == Some(true),
            self.axis_peak_vram_win == Some(true),
            self.axis_geometry_stage_win == Some(true),
            self.axis_visible_detail_win == Some(true),
            self.axis_lod_temporal_win == Some(true),
        ]
        .iter()
        .filter(|&&w| w)
        .count() as u32
    }

    pub fn claim(&self) -> Option<&'static str> {
        let thresholds = self.losses == Some(0)
            && self
                .repeated_storage_ratio
                .is_some_and(|value| value <= 0.25)
            && self.mixed_storage_ratio.is_some_and(|value| value <= 0.50)
            && self
                .fixed_budget_detail_ratio
                .is_some_and(|value| value >= 2.0)
            && self
                .lod_temporal_invalidations_reduction
                .is_some_and(|value| value >= 0.50)
            && self
                .full_frame_regression_pct
                .is_some_and(|value| value <= 2.0);
        if thresholds && self.independent_axis_wins() >= 3 {
            Some("city_geometry_program_breakthrough")
        } else {
            None
        }
    }

    pub fn winning_axes(&self) -> u32 {
        self.independent_axis_wins()
    }
}

// ---------------------------------------------------------------------------
// Ray-native (visibility) breakthrough claim
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VisibilityBreakthrough {
    pub ray_native_coverage: Option<f64>,
    pub materialized_geometry_ratio: Option<f64>,
    pub parameter_update_fine_as_builds: Option<u64>,
    pub geometry_stage_speedup: Option<f64>,
    pub fixed_budget_detail_ratio: Option<f64>,
    pub ray_native_temporal_invalidations: Option<u64>,
    pub correctness_mismatches: Option<u64>,
    pub real_asset_families: Option<u32>,
    pub losses: Option<u32>,
    pub procedural_intersection_win: Option<bool>,
    pub removed_triangles_counted_as_axis: Option<bool>,
    pub as_bytes_counted_as_axis: Option<bool>,
}

impl VisibilityBreakthrough {
    pub fn unavailable() -> Self {
        VisibilityBreakthrough {
            ray_native_coverage: None,
            materialized_geometry_ratio: None,
            parameter_update_fine_as_builds: None,
            geometry_stage_speedup: None,
            fixed_budget_detail_ratio: None,
            ray_native_temporal_invalidations: None,
            correctness_mismatches: None,
            real_asset_families: None,
            losses: None,
            procedural_intersection_win: None,
            removed_triangles_counted_as_axis: None,
            as_bytes_counted_as_axis: None,
        }
    }

    fn double_counts(&self) -> bool {
        self.removed_triangles_counted_as_axis == Some(true)
            && self.as_bytes_counted_as_axis == Some(true)
    }

    pub fn claim(&self) -> Option<&'static str> {
        let ok = self.ray_native_coverage.is_some_and(|value| value >= 0.60)
            && self
                .materialized_geometry_ratio
                .is_some_and(|value| value <= 0.25)
            && self.parameter_update_fine_as_builds == Some(0)
            && self
                .geometry_stage_speedup
                .is_some_and(|value| value >= 1.50)
            && self
                .fixed_budget_detail_ratio
                .is_some_and(|value| value >= 4.0)
            && self.ray_native_temporal_invalidations == Some(0)
            && self.correctness_mismatches == Some(0)
            && self.real_asset_families.is_some_and(|value| value >= 2)
            && self.losses == Some(0)
            && self.procedural_intersection_win == Some(true)
            && !self.double_counts();
        if ok {
            Some("ray_native_city_geometry")
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Certified visibility fabric breakthrough claim
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FabricBreakthrough {
    pub structured_ray_coverage: f64,
    pub certified_without_per_ray_traversal: f64,
    pub active_packet_lanes: f64,
    pub intersection_speedup_vs_scalar: f64,
    pub geometry_stage_speedup: f64,
    pub routing_queue_overhead_pct: f64,
    pub correctness_mismatches: u64,
    pub cpu_scheduling: u64,
    pub real_asset_families: u32,
    pub losses: u32,
    pub queue_overflows: u64,
    pub only_ray_sorting: bool,
    pub includes_failed_certificates: bool,
    pub includes_fixed_empty_launches: bool,
}

impl FabricBreakthrough {
    pub fn unavailable() -> Self {
        FabricBreakthrough {
            structured_ray_coverage: 0.0,
            certified_without_per_ray_traversal: 0.0,
            active_packet_lanes: 0.0,
            intersection_speedup_vs_scalar: 0.0,
            geometry_stage_speedup: 0.0,
            routing_queue_overhead_pct: 100.0,
            correctness_mismatches: 0,
            cpu_scheduling: 0,
            real_asset_families: 0,
            losses: 0,
            queue_overflows: 0,
            only_ray_sorting: false,
            includes_failed_certificates: false,
            includes_fixed_empty_launches: false,
        }
    }

    pub fn claim(&self) -> Option<&'static str> {
        let ok = self.structured_ray_coverage >= 0.60
            && self.certified_without_per_ray_traversal >= 0.35
            && self.active_packet_lanes >= 0.75
            && self.intersection_speedup_vs_scalar >= 2.0
            && self.geometry_stage_speedup >= 2.0
            && self.routing_queue_overhead_pct <= 10.0
            && self.correctness_mismatches == 0
            && self.cpu_scheduling == 0
            && self.real_asset_families >= 2
            && self.losses == 0
            && self.queue_overflows == 0
            && !self.only_ray_sorting
            && self.includes_failed_certificates
            && self.includes_fixed_empty_launches;
        if ok {
            Some("certified_visibility_fabric")
        } else {
            None
        }
    }
}

/// Compose the visibility-Fabric breakthrough solely from real finished-object
/// fixture receipts. Missing profiles, missing GPU receipts, absent correctness
/// evidence, or incomplete family coverage fail closed and can never be
/// interpreted as a measured zero.
pub fn compute_fabric_breakthrough(fixtures: &[FixtureReport]) -> FabricBreakthrough {
    let mut result = FabricBreakthrough::unavailable();
    result.includes_failed_certificates = true;
    result.includes_fixed_empty_launches = true;
    let mut input_rays = 0u64;
    let mut structured_rays = 0u64;
    let mut certified_miss_rays = 0u64;
    let mut packet_active_lanes = 0u64;
    let mut packet_total_lanes = 0u64;
    let mut reference_visibility_ms = 0.0;
    let mut candidate_visibility_ms = 0.0;
    let mut reference_geometry_ms = 0.0;
    let mut candidate_geometry_ms = 0.0;
    let mut candidate_routing_ms = 0.0;
    let mut credited_families: Vec<&str> = Vec::new();

    for fixture in fixtures.iter().filter(|fixture| {
        REAL_ASSET_FAMILIES.contains(&fixture.family.as_str())
            && fixture.asset_content_hash.is_some()
    }) {
        let Some(receipt) = fixture.hybrid.visibility_fabric else {
            result.losses = result.losses.saturating_add(1);
            continue;
        };
        let (Some(reference), Some(candidate)) = (
            fixture.scalar_visibility_profile,
            fixture.hybrid.visibility_profile,
        ) else {
            result.losses = result.losses.saturating_add(1);
            continue;
        };
        if !reference.is_measured()
            || !candidate.is_measured()
            || receipt.live_source_rays == 0
            || receipt.packet_active_lanes > receipt.packet_total_lanes
            || receipt
                .rays_certified_miss
                .saturating_add(receipt.packet_rays)
                > receipt.live_source_rays
        {
            result.losses = result.losses.saturating_add(1);
            continue;
        }

        input_rays = input_rays.saturating_add(receipt.live_source_rays);
        certified_miss_rays = certified_miss_rays.saturating_add(receipt.rays_certified_miss);
        structured_rays = structured_rays.saturating_add(
            receipt
                .rays_certified_miss
                .saturating_add(receipt.packet_rays),
        );
        packet_active_lanes = packet_active_lanes.saturating_add(receipt.packet_active_lanes);
        packet_total_lanes = packet_total_lanes.saturating_add(receipt.packet_total_lanes);
        reference_visibility_ms += reference.complete_visibility_ms_p95;
        candidate_visibility_ms += candidate.complete_visibility_ms_p95;
        reference_geometry_ms += fixture.reference.stage.geometry_stage_ms_p95;
        candidate_geometry_ms += fixture.hybrid.stage.geometry_stage_ms_p95;
        candidate_routing_ms += candidate.routing_queue_ms_p95;
        result.cpu_scheduling = result.cpu_scheduling.saturating_add(receipt.cpu_scheduling);
        result.queue_overflows = result
            .queue_overflows
            .saturating_add(receipt.queue_overflow_mask);
        result.correctness_mismatches = result
            .correctness_mismatches
            .saturating_add(receipt.correctness_mismatches);
        result.includes_fixed_empty_launches &=
            candidate.includes_fixed_empty_launches && reference.includes_fixed_empty_launches;
        if receipt.rays_certified_miss > 0 {
            result.only_ray_sorting = false;
        }

        for correctness in [&fixture.reference.correctness, &fixture.hybrid.correctness] {
            match (
                correctness.material_mismatches,
                correctness.uv_mismatches,
                correctness.hit_mismatches,
            ) {
                (Some(material), Some(uv), Some(hit)) => {
                    result.correctness_mismatches = result
                        .correctness_mismatches
                        .saturating_add(material)
                        .saturating_add(uv)
                        .saturating_add(hit)
                        .saturating_add(
                            correctness
                                .source_primitives
                                .abs_diff(correctness.mapped_primitives),
                        );
                }
                _ => result.losses = result.losses.saturating_add(1),
            }
        }
        if candidate.complete_visibility_ms_p95 >= reference.complete_visibility_ms_p95 {
            result.losses = result.losses.saturating_add(1);
        }
        if !credited_families.contains(&fixture.family.as_str()) {
            credited_families.push(&fixture.family);
        }
    }

    result.real_asset_families = credited_families.len() as u32;
    if input_rays > 0 {
        result.structured_ray_coverage = structured_rays as f64 / input_rays as f64;
        result.certified_without_per_ray_traversal = certified_miss_rays as f64 / input_rays as f64;
    }
    if packet_total_lanes > 0 {
        result.active_packet_lanes = packet_active_lanes as f64 / packet_total_lanes as f64;
    }
    if candidate_visibility_ms > 0.0 {
        result.intersection_speedup_vs_scalar = reference_visibility_ms / candidate_visibility_ms;
        result.routing_queue_overhead_pct = candidate_routing_ms / candidate_visibility_ms * 100.0;
    }
    if candidate_geometry_ms > 0.0 {
        result.geometry_stage_speedup = reference_geometry_ms / candidate_geometry_ms;
    }
    result.only_ray_sorting = certified_miss_rays == 0;
    if result.real_asset_families < REAL_ASSET_FAMILIES.len() as u32 {
        result.losses = result.losses.saturating_add(1);
    }
    result
}

// ---------------------------------------------------------------------------
// Evidence-line formatting (exact MEGAGEOMETRY_* schema). Unmeasured `Option`
// values print `unavailable`, never a synthetic zero.
// ---------------------------------------------------------------------------

fn opt_f(o: Option<f64>) -> String {
    o.map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "unavailable".into())
}
fn opt_u(o: Option<u64>) -> String {
    o.map(|v| v.to_string())
        .unwrap_or_else(|| "unavailable".into())
}
fn opt_u32(o: Option<u32>) -> String {
    o.map(|v| v.to_string())
        .unwrap_or_else(|| "unavailable".into())
}
fn opt_match(o: Option<bool>) -> &'static str {
    match o {
        Some(true) => "match",
        Some(false) => "mismatch",
        None => "unavailable",
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvLine {
    pub gpu: String,
    pub api: String,
    pub driver: String,
    pub rt_backend: String,
    pub capability_hash: String,
    pub shader_abi: String,
    #[serde(alias = "cook_schema")]
    pub runtime_geometry_schema: u32,
}

impl EnvLine {
    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_ENV gpu={} api={} driver={} rt_backend={} capability_hash={} shader_abi={} runtime_geometry_schema={}",
            self.gpu,
            self.api,
            self.driver,
            self.rt_backend,
            self.capability_hash,
            self.shader_abi,
            self.runtime_geometry_schema
        )
    }
}

pub fn config_line(mode: Mode, compute: ComputeVariant, spec: &FixtureSpec, frames: u32) -> String {
    format!(
        "MEGAGEOMETRY_CONFIG mode={} compute={} fixture={} instances={} frames={} seed={}",
        mode.as_str(),
        compute.as_str(),
        spec.name,
        spec.instances,
        frames,
        spec.seed
    )
}

pub fn correct_line(c: &Correctness) -> String {
    format!(
        "MEGAGEOMETRY_CORRECT source_primitives={} mapped_primitives={} material_mismatches={} uv_mismatches={} hit_mismatches={}",
        c.source_primitives,
        c.mapped_primitives,
        opt_u(c.material_mismatches),
        opt_u(c.uv_mismatches),
        opt_u(c.hit_mismatches)
    )
}

pub fn result_line(s: &GeometryStage, v: &VramBreakdown) -> String {
    format!(
        "MEGAGEOMETRY_RESULT build_ms_p50={:.4} build_ms_p95={:.4} update_ms_p50={:.4} update_ms_p95={:.4} native_lowering_ms_p95={} trace_ms_p50={:.4} trace_ms_p95={:.4} frame_ms_p50={:.4} frame_ms_p95={:.4} resident_vram_mb={:.3} as_vram_mb={} page_vram_mb={} scratch_vram_mb={} peak_geometry_vram_mb={} traced_triangles={} required_page_misses={}",
        s.build_ms_p50,
        s.build_ms_p95,
        s.update_ms_p50,
        s.update_ms_p95,
        opt_f(s.native_lowering_ms_p95),
        s.trace_ms_p50,
        s.trace_ms_p95,
        s.frame_ms_p50,
        s.frame_ms_p95,
        v.resident_mb,
        opt_f(v.as_mb),
        opt_f(v.page_mb),
        opt_f(v.scratch_mb),
        opt_f(v.peak_geometry_mb()),
        s.traced_triangles,
        opt_u(s.required_page_misses)
    )
}

pub fn cpu_line(c: &CpuCounters) -> String {
    format!(
        "MEGAGEOMETRY_CPU render_thread_scene_visits={} visibility_node_visits={} construction_evaluations={} lod_decisions={} page_decisions={} eviction_decisions={} blocking_readbacks={} frame_allocations={} host_deadline_misses={} render_thread_submit_ms_p95={} host_submit_calls={} host_submit_ms_p95={}",
        c.render_thread_scene_visits,
        c.visibility_node_visits,
        c.construction_evaluations,
        c.lod_decisions,
        c.page_decisions,
        c.eviction_decisions,
        c.blocking_readbacks,
        c.frame_allocations,
        opt_u(c.host_deadline_misses),
        opt_f(c.render_thread_submit_ms_p95),
        c.host_submit_calls,
        opt_f(c.host_submit_ms_p95)
    )
}

pub fn portable_line(g: &GlobalSignals, cooperative_error: Option<f64>) -> String {
    format!(
        "MEGAGEOMETRY_PORTABLE packet_abi={} control_hash={} cooperative_error={} adapter_overhead_pct={} cpu_fallbacks={}",
        opt_match(g.packet_abi_match),
        opt_match(g.control_hash_match),
        opt_f(cooperative_error),
        opt_f(g.adapter_overhead_pct),
        opt_u(g.cpu_fallbacks)
    )
}

pub fn identity_line(result: &ModeResult) -> String {
    format!(
        "MEGAGEOMETRY_IDENTITY mode={} packet_abi_hash={} control_hash={} capability_fingerprint={} hardware_clas_builds={}",
        result.mode.as_str(),
        result.packet_abi_hash.as_deref().unwrap_or("unavailable"),
        result.control_hash.as_deref().unwrap_or("unavailable"),
        result
            .capability_fingerprint
            .as_deref()
            .unwrap_or("unavailable"),
        opt_u(result.backend.hardware_clas_builds),
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgramLine {
    pub mode: String,
    pub encoded_bytes: Option<u64>,
    pub decoded_bytes: Option<u64>,
    pub topology_templates: Option<u32>,
    pub reused_blocks: Option<u32>,
    pub decode_ms_p95: Option<f64>,
    pub decoded_hash: String,
}

impl ProgramLine {
    pub fn unavailable(mode: &str) -> Self {
        ProgramLine {
            mode: mode.into(),
            encoded_bytes: None,
            decoded_bytes: None,
            topology_templates: None,
            reused_blocks: None,
            decode_ms_p95: None,
            decoded_hash: "unavailable".into(),
        }
    }
    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_PROGRAM mode={} encoded_bytes={} decoded_bytes={} topology_templates={} reused_blocks={} decode_ms_p95={} decoded_hash={}",
            self.mode,
            opt_u(self.encoded_bytes),
            opt_u(self.decoded_bytes),
            opt_u32(self.topology_templates),
            opt_u32(self.reused_blocks),
            opt_f(self.decode_ms_p95),
            self.decoded_hash
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisibilityLine {
    pub mode: String,
    pub source_surface_coverage: Option<f64>,
    pub residual_coverage: Option<f64>,
    pub coarse_aabb_bytes: Option<u64>,
    pub materialized_geometry_bytes: Option<u64>,
    pub fine_as_builds: Option<u64>,
    pub intersection_ms_p95: Option<f64>,
    pub geometry_stage_ms_p95: Option<f64>,
    pub stable_surface_hash: String,
}

impl VisibilityLine {
    pub fn unavailable(mode: &str) -> Self {
        VisibilityLine {
            mode: mode.into(),
            source_surface_coverage: None,
            residual_coverage: None,
            coarse_aabb_bytes: None,
            materialized_geometry_bytes: None,
            fine_as_builds: None,
            intersection_ms_p95: None,
            geometry_stage_ms_p95: None,
            stable_surface_hash: "unavailable".into(),
        }
    }
    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_VISIBILITY mode={} source_surface_coverage={} residual_coverage={} coarse_aabb_bytes={} materialized_geometry_bytes={} fine_as_builds={} intersection_ms_p95={} geometry_stage_ms_p95={} stable_surface_hash={}",
            self.mode,
            opt_f(self.source_surface_coverage),
            opt_f(self.residual_coverage),
            opt_u(self.coarse_aabb_bytes),
            opt_u(self.materialized_geometry_bytes),
            opt_u(self.fine_as_builds),
            opt_f(self.intersection_ms_p95),
            opt_f(self.geometry_stage_ms_p95),
            self.stable_surface_hash
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FabricLine {
    pub mode: String,
    pub input_rays: Option<u64>,
    pub structured_rays: Option<u64>,
    pub certified_rays: Option<u64>,
    pub subdivided_domains: Option<u64>,
    pub explicit_rays: Option<u64>,
    pub active_packet_lanes: Option<f64>,
    pub routing_ms_p95: Option<f64>,
    pub certification_ms_p95: Option<f64>,
    pub packet_ms_p95: Option<f64>,
    pub refinement_ms_p95: Option<f64>,
    pub residual_rt_ms_p95: Option<f64>,
    pub merge_ms_p95: Option<f64>,
    pub queue_overflows: Option<u64>,
    pub cpu_scheduling: Option<u64>,
    pub hit_hash: String,
}

impl FabricLine {
    pub fn unavailable(mode: &str) -> Self {
        FabricLine {
            mode: mode.into(),
            input_rays: None,
            structured_rays: None,
            certified_rays: None,
            subdivided_domains: None,
            explicit_rays: None,
            active_packet_lanes: None,
            routing_ms_p95: None,
            certification_ms_p95: None,
            packet_ms_p95: None,
            refinement_ms_p95: None,
            residual_rt_ms_p95: None,
            merge_ms_p95: None,
            queue_overflows: None,
            cpu_scheduling: None,
            hit_hash: "unavailable".into(),
        }
    }
    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_FABRIC mode={} input_rays={} structured_rays={} certified_rays={} subdivided_domains={} explicit_rays={} active_packet_lanes={} routing_ms_p95={} certification_ms_p95={} packet_ms_p95={} refinement_ms_p95={} residual_rt_ms_p95={} merge_ms_p95={} queue_overflows={} cpu_scheduling={} hit_hash={}",
            self.mode,
            opt_u(self.input_rays),
            opt_u(self.structured_rays),
            opt_u(self.certified_rays),
            opt_u(self.subdivided_domains),
            opt_u(self.explicit_rays),
            opt_f(self.active_packet_lanes),
            opt_f(self.routing_ms_p95),
            opt_f(self.certification_ms_p95),
            opt_f(self.packet_ms_p95),
            opt_f(self.refinement_ms_p95),
            opt_f(self.residual_rt_ms_p95),
            opt_f(self.merge_ms_p95),
            opt_u(self.queue_overflows),
            opt_u(self.cpu_scheduling),
            self.hit_hash
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalLine {
    pub surface_map_coverage: Option<f64>,
    pub lod_switch_invalidations: Option<u64>,
    pub radiance_mismatches: Option<u64>,
    pub required_sequence_match: Option<bool>,
}

impl TemporalLine {
    pub fn unavailable() -> Self {
        TemporalLine {
            surface_map_coverage: None,
            lod_switch_invalidations: None,
            radiance_mismatches: None,
            required_sequence_match: None,
        }
    }
    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_TEMPORAL surface_map_coverage={} lod_switch_invalidations={} radiance_mismatches={} feedback_owner=gpu required_sequence_match={}",
            opt_f(self.surface_map_coverage),
            opt_u(self.lod_switch_invalidations),
            opt_u(self.radiance_mismatches),
            self.required_sequence_match
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unavailable".into())
        )
    }
}

pub fn breakthrough_line(b: &ProgramBreakthrough) -> String {
    format!(
        "MEGAGEOMETRY_BREAKTHROUGH repeated_storage_ratio={} mixed_storage_ratio={} fixed_budget_detail_ratio={} lod_temporal_invalidations_reduction={} full_frame_regression_pct={} winning_axes={} losses={} claim={}",
        opt_f(b.repeated_storage_ratio),
        opt_f(b.mixed_storage_ratio),
        opt_f(b.fixed_budget_detail_ratio),
        opt_f(b.lod_temporal_invalidations_reduction),
        opt_f(b.full_frame_regression_pct),
        b.winning_axes(),
        opt_u32(b.losses),
        b.claim().unwrap_or("experimental_subset")
    )
}

pub fn visibility_breakthrough_line(b: &VisibilityBreakthrough) -> String {
    format!(
        "MEGAGEOMETRY_VISIBILITY_BREAKTHROUGH ray_native_coverage={} materialized_geometry_ratio={} parameter_update_fine_as_builds={} geometry_stage_speedup={} fixed_budget_detail_ratio={} ray_native_temporal_invalidations={} correctness_mismatches={} real_asset_families={} losses={} procedural_intersection_win={} claim={}",
        opt_f(b.ray_native_coverage),
        opt_f(b.materialized_geometry_ratio),
        opt_u(b.parameter_update_fine_as_builds),
        opt_f(b.geometry_stage_speedup),
        opt_f(b.fixed_budget_detail_ratio),
        opt_u(b.ray_native_temporal_invalidations),
        opt_u(b.correctness_mismatches),
        opt_u32(b.real_asset_families),
        opt_u32(b.losses),
        opt_match(b.procedural_intersection_win),
        b.claim().unwrap_or("experimental_subset")
    )
}

pub fn fabric_breakthrough_line(b: &FabricBreakthrough) -> String {
    format!(
        "MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage={:.4} certified_without_per_ray_traversal={:.4} active_packet_lanes={:.4} intersection_speedup_vs_scalar={:.4} geometry_stage_speedup={:.4} routing_queue_overhead_pct={:.4} correctness_mismatches={} cpu_scheduling={} real_asset_families={} losses={} claim={}",
        b.structured_ray_coverage,
        b.certified_without_per_ray_traversal,
        b.active_packet_lanes,
        b.intersection_speedup_vs_scalar,
        b.geometry_stage_speedup,
        b.routing_queue_overhead_pct,
        b.correctness_mismatches,
        b.cpu_scheduling,
        b.real_asset_families,
        b.losses,
        b.claim().unwrap_or("experimental_subset")
    )
}

pub fn fixture_line(spec: &FixtureSpec, outcome: &FixtureOutcome) -> String {
    let tail = match outcome {
        FixtureOutcome::Parity { regression_pct } => {
            format!("outcome=parity regression_pct={regression_pct:.2}")
        }
        FixtureOutcome::CorrectnessOnly { mismatches } => {
            format!("outcome=correctness_only mismatches={mismatches}")
        }
        FixtureOutcome::Win { winner } => format!("outcome=win winner={}", winner.as_str()),
        FixtureOutcome::Loss { reason } => format!("outcome=loss reason=\"{reason}\""),
        FixtureOutcome::CorrectnessFailure { reason } => {
            format!("outcome=correctness_failure reason=\"{reason}\"")
        }
        FixtureOutcome::Unavailable { reason } => {
            format!("outcome=unavailable reason=\"{reason}\"")
        }
    };
    format!("MEGAGEOMETRY_FIXTURE name={} {tail}", spec.name)
}

pub fn overall_line(v: &OverallVerdict) -> String {
    match v.winner {
        Some(w) => format!(
            "MEGAGEOMETRY_OVERALL winner={} losses={} target_wins={} claim={}",
            w.as_str(),
            v.losses,
            v.target_wins,
            v.claim
        ),
        None => format!(
            "MEGAGEOMETRY_OVERALL winner=none losses={} target_wins={} claim={}",
            v.losses, v.target_wins, v.claim
        ),
    }
}

// ---------------------------------------------------------------------------
// Serializable aggregate report (JSON output).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureReport {
    pub name: String,
    pub role: FixtureRole,
    pub asset_id: String,
    pub family: String,
    pub instances: u64,
    pub seed: u64,
    /// Hash of the fixture DEFINITION (identity), not its rendered bytes.
    pub identity_hash: String,
    /// Hash of the real loaded asset content. `None` until a real asset loader is
    /// wired — the fixture is then NOT authorized as the real corpus.
    pub asset_content_hash: Option<String>,
    pub outcome: FixtureOutcome,
    pub hybrid: ModeResult,
    pub reference: ModeResult,
    pub triangle: Option<ModeResult>,
    /// Scalar ray-native calibration over the same Ochroma runtime
    /// representation as `hybrid`. This is separate from `reference`, which is
    /// the complete-mesh native cluster/CLAS competitor.
    #[serde(default)]
    pub scalar_visibility_profile: Option<VisibilityKernelProfile>,
    /// Exact byte/count evidence derived by Ochroma from the authoritative
    /// finished mesh loaded for this fixture. This is runtime evidence, never
    /// an asset-authored or asset-cooked MegaGeometry record.
    #[serde(default)]
    pub runtime_program: Option<RuntimeProgramEvidence>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RuntimeProgramEvidence {
    pub source_triangles: u64,
    pub covered_triangles: u64,
    pub program_count: u64,
    pub template_count: u64,
    /// Complete authoritative source-mesh GPU input bytes.
    pub source_geometry_bytes: u64,
    /// Exact source-mesh bytes needed by the covered subset, including its
    /// unique vertices and triangle streams.
    pub covered_source_geometry_bytes: u64,
    /// Exact residual source-mesh bytes retained beside ray-native programs.
    pub residual_geometry_bytes: u64,
    /// Every Ochroma-derived GPU program/hierarchy/correspondence byte.
    pub runtime_program_bytes: u64,
}

fn scoped_losses(fixtures: &[FixtureReport]) -> u32 {
    fixtures
        .iter()
        .filter(|fixture| REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()))
        .filter(|fixture| fixture.role != FixtureRole::CorrectnessOnly)
        .filter(|fixture| {
            matches!(
                fixture.outcome,
                FixtureOutcome::Loss { .. }
                    | FixtureOutcome::CorrectnessFailure { .. }
                    | FixtureOutcome::Unavailable { .. }
            )
        })
        .count() as u32
}

fn representative_runtime_programs(
    fixtures: &[FixtureReport],
) -> BTreeMap<&str, RuntimeProgramEvidence> {
    let mut by_family = BTreeMap::new();
    for family in REAL_ASSET_FAMILIES {
        if let Some(evidence) = fixtures
            .iter()
            .filter(|fixture| fixture.family == family)
            .find_map(|fixture| fixture.runtime_program)
        {
            by_family.insert(family, evidence);
        }
    }
    by_family
}

fn structural_performance_fixtures(fixtures: &[FixtureReport]) -> Vec<&FixtureReport> {
    fixtures
        .iter()
        .filter(|fixture| REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()))
        .filter(|fixture| fixture.role == FixtureRole::TargetWin)
        .collect()
}

/// Compose geometry-program evidence exclusively from the authoritative
/// finished meshes, Ochroma-derived runtime payloads, and complete measured
/// mode results. Missing temporal/update experiments remain `None`.
pub fn compute_program_breakthrough(fixtures: &[FixtureReport]) -> ProgramBreakthrough {
    let mut result = ProgramBreakthrough::unavailable();
    let programs = representative_runtime_programs(fixtures);

    result.repeated_storage_ratio = programs
        .get(REAL_ASSET_FAMILIES[0])
        .filter(|evidence| evidence.covered_source_geometry_bytes > 0)
        .map(|evidence| {
            evidence.runtime_program_bytes as f64 / evidence.covered_source_geometry_bytes as f64
        });

    if programs.len() == REAL_ASSET_FAMILIES.len() {
        let source = programs
            .values()
            .map(|evidence| evidence.source_geometry_bytes)
            .sum::<u64>();
        let hybrid = programs
            .values()
            .map(|evidence| {
                evidence
                    .residual_geometry_bytes
                    .saturating_add(evidence.runtime_program_bytes)
            })
            .sum::<u64>();
        result.mixed_storage_ratio = (source > 0).then_some(hybrid as f64 / source as f64);
    }

    let performance = structural_performance_fixtures(fixtures);
    if !performance.is_empty() {
        result.full_frame_regression_pct = performance
            .iter()
            .map(|fixture| {
                let reference = fixture.reference.stage.frame_ms_p95;
                (reference > 0.0)
                    .then_some((fixture.hybrid.stage.frame_ms_p95 / reference - 1.0) * 100.0)
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.into_iter().reduce(f64::max));

        result.fixed_budget_detail_ratio = performance
            .iter()
            .map(|fixture| {
                Some(
                    fixture.reference.vram.peak_geometry_mb()?
                        / fixture.hybrid.vram.peak_geometry_mb()?,
                )
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.into_iter().reduce(f64::min));

        result.axis_peak_vram_win = Some(performance.iter().all(|fixture| {
            matches!(
                (
                    fixture.hybrid.vram.peak_geometry_mb(),
                    fixture.reference.vram.peak_geometry_mb()
                ),
                (Some(hybrid), Some(reference)) if hybrid < reference
            )
        }));
        result.axis_geometry_stage_win = Some(performance.iter().all(|fixture| {
            fixture.hybrid.stage.geometry_stage_ms_p95
                < fixture.reference.stage.geometry_stage_ms_p95
        }));
        let temporal = performance
            .iter()
            .map(|fixture| {
                let runtime = fixture.hybrid.temporal_reuse?;
                (runtime.relevant_surface_pixels > 0 && runtime.lod_cut_changes > 0).then_some((
                    runtime.no_map_invalidated_surface_pixels,
                    runtime.invalidated_surface_pixels,
                ))
            })
            .collect::<Option<Vec<_>>>();
        if let Some(samples) = temporal {
            let reference = samples.iter().map(|sample| sample.0).sum::<u64>();
            let runtime = samples.iter().map(|sample| sample.1).sum::<u64>();
            result.lod_temporal_invalidations_reduction = if reference == 0 {
                (runtime == 0).then_some(0.0)
            } else {
                Some(1.0 - runtime as f64 / reference as f64)
            };
        }
    }

    result.losses = Some(scoped_losses(fixtures));
    result.axis_storage_bytes_win =
        match (result.repeated_storage_ratio, result.mixed_storage_ratio) {
            (Some(repeated), Some(mixed)) => Some(repeated <= 0.25 && mixed <= 0.50),
            _ => None,
        };
    result.axis_visible_detail_win = result.fixed_budget_detail_ratio.map(|ratio| ratio >= 2.0);
    result.axis_lod_temporal_win = result
        .lod_temporal_invalidations_reduction
        .map(|reduction| reduction >= 0.50);
    result
}

/// Compose the scalar ray-native scope without borrowing the certified-Fabric
/// result. Procedural intersection is compared only when both the scalar
/// procedural path and complete native reference have matching device-event
/// visibility profiles.
pub fn compute_visibility_breakthrough(fixtures: &[FixtureReport]) -> VisibilityBreakthrough {
    let mut result = VisibilityBreakthrough::unavailable();
    let programs = representative_runtime_programs(fixtures);
    if programs.len() == REAL_ASSET_FAMILIES.len() {
        let source = programs
            .values()
            .map(|evidence| evidence.source_triangles)
            .sum::<u64>();
        let covered = programs
            .values()
            .map(|evidence| evidence.covered_triangles)
            .sum::<u64>();
        result.ray_native_coverage = (source > 0).then_some(covered as f64 / source as f64);
        result.real_asset_families = Some(programs.len() as u32);
        result.parameter_update_fine_as_builds = structural_performance_fixtures(fixtures)
            .iter()
            .map(|fixture| fixture.hybrid.parameter_update_fine_as_builds)
            .collect::<Option<Vec<_>>>()
            .map(|values| values.into_iter().sum());
        result.ray_native_temporal_invalidations = structural_performance_fixtures(fixtures)
            .iter()
            .map(|fixture| {
                fixture.hybrid.temporal_reuse.and_then(|evidence| {
                    (evidence.relevant_surface_pixels > 0 && evidence.lod_cut_changes > 0)
                        .then_some(evidence.invalidated_surface_pixels)
                })
            })
            .collect::<Option<Vec<_>>>()
            .map(|values| values.into_iter().sum());
    }

    let performance = structural_performance_fixtures(fixtures);
    if !performance.is_empty() {
        result.procedural_intersection_win = performance
            .iter()
            .map(|fixture| {
                let procedural = fixture.scalar_visibility_profile?;
                let native = fixture.reference.visibility_profile?;
                (procedural.is_measured() && native.is_measured()).then_some(
                    procedural.complete_visibility_ms_p95 < native.complete_visibility_ms_p95,
                )
            })
            .collect::<Option<Vec<_>>>()
            .map(|values| values.into_iter().all(|won| won));
        result.materialized_geometry_ratio = {
            let reference = performance
                .iter()
                .map(|fixture| fixture.reference.stage.traced_triangles)
                .sum::<u64>();
            let hybrid = performance
                .iter()
                .map(|fixture| fixture.hybrid.stage.traced_triangles)
                .sum::<u64>();
            (reference > 0).then_some(hybrid as f64 / reference as f64)
        };
        result.geometry_stage_speedup = performance
            .iter()
            .map(|fixture| {
                let hybrid = fixture.hybrid.stage.geometry_stage_ms_p95;
                (hybrid > 0.0).then_some(fixture.reference.stage.geometry_stage_ms_p95 / hybrid)
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.into_iter().reduce(f64::min));
        result.fixed_budget_detail_ratio = performance
            .iter()
            .map(|fixture| {
                Some(
                    fixture.reference.vram.peak_geometry_mb()?
                        / fixture.hybrid.vram.peak_geometry_mb()?,
                )
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.into_iter().reduce(f64::min));
    }

    result.correctness_mismatches = fixtures
        .iter()
        .filter(|fixture| REAL_ASSET_FAMILIES.contains(&fixture.family.as_str()))
        .map(|fixture| {
            let correctness = fixture.hybrid.correctness;
            Some(
                correctness.material_mismatches?
                    + correctness.uv_mismatches?
                    + correctness.hit_mismatches?
                    + correctness
                        .source_primitives
                        .abs_diff(correctness.mapped_primitives),
            )
        })
        .collect::<Option<Vec<_>>>()
        .map(|values| values.into_iter().sum());
    result.losses = Some(scoped_losses(fixtures));
    // These two counters state that correlated byte/triangle reductions are not
    // being counted as separate breakthrough axes.
    result.removed_triangles_counted_as_axis = Some(false);
    result.as_bytes_counted_as_axis = Some(false);
    result
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchReport {
    pub env: EnvLine,
    pub suite: String,
    pub modes: Vec<String>,
    pub warmup_frames: u32,
    pub timed_frames: u32,
    pub seed: u64,
    pub command_line: String,
    pub git_revisions: GitRevisions,
    pub render_config_hash: String,
    pub corpus_hash: String,
    pub fixtures: Vec<FixtureReport>,
    pub global: GlobalSignals,
    pub overall: OverallVerdict,
    pub program_breakthrough: Option<ProgramBreakthrough>,
    pub visibility_breakthrough: Option<VisibilityBreakthrough>,
    pub fabric_breakthrough: Option<FabricBreakthrough>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GitRevisions {
    pub ochroma: String,
    pub spectra: String,
    pub urban: String,
}

impl BenchReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialize bench report: {e}"))
    }
    pub fn from_json(s: &str) -> Result<BenchReport, String> {
        serde_json::from_str(s).map_err(|e| format!("parse bench report: {e}"))
    }
}

/// Cross-backend equality is never asserted by a raw producer. This composed
/// artifact is the only authority allowed to compare independently captured
/// CUDA and Vulkan reports.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConformance {
    pub schema: u32,
    pub cuda_gpu: String,
    pub vulkan_gpu: String,
    pub same_inputs: bool,
    pub packet_abi_match: Option<bool>,
    pub control_hash_match: Option<bool>,
    pub compared_mode_records: u32,
    pub unavailable_records: u32,
    pub reasons: Vec<String>,
}

impl BackendConformance {
    pub fn passes(&self) -> bool {
        self.same_inputs
            && self.packet_abi_match == Some(true)
            && self.control_hash_match == Some(true)
            && self.compared_mode_records > 0
            && self.unavailable_records == 0
            && self.reasons.is_empty()
    }

    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_CONFORMANCE packet_abi={} control_hash={} same_inputs={} compared_mode_records={} unavailable_records={} result={}",
            opt_match(self.packet_abi_match),
            opt_match(self.control_hash_match),
            self.same_inputs,
            self.compared_mode_records,
            self.unavailable_records,
            if self.passes() { "pass" } else { "fail" },
        )
    }
}

fn merge_comparison(current: Option<bool>, next: Option<bool>) -> Option<bool> {
    match (current, next) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn mode_records(fixture: &FixtureReport) -> Vec<(&'static str, &ModeResult)> {
    let mut records = vec![
        ("reference", &fixture.reference),
        ("city_hybrid", &fixture.hybrid),
    ];
    if let Some(triangle) = fixture.triangle.as_ref() {
        records.push(("triangle", triangle));
    }
    records
}

/// Compose two raw reports captured by independent backends. Capability
/// fingerprints are required to exist but are deliberately not compared for
/// equality: they describe different hardware. Packet ABI and the
/// backend-neutral post-timing GPU control digest must match for every
/// corresponding non-product fixture/mode.
pub fn compose_backend_conformance(cuda: &BenchReport, vulkan: &BenchReport) -> BackendConformance {
    let mut reasons = Vec::new();
    if cuda.env.api != "cuda" {
        reasons.push(format!("CUDA input reports api={}", cuda.env.api));
    }
    if vulkan.env.api != "vulkan" {
        reasons.push(format!("Vulkan input reports api={}", vulkan.env.api));
    }

    let mut same_inputs = cuda.suite == vulkan.suite
        && cuda.modes == vulkan.modes
        && cuda.warmup_frames == vulkan.warmup_frames
        && cuda.timed_frames == vulkan.timed_frames
        && cuda.seed == vulkan.seed
        && cuda.render_config_hash == vulkan.render_config_hash
        && cuda.corpus_hash == vulkan.corpus_hash
        && cuda.env.shader_abi == vulkan.env.shader_abi
        && cuda.env.runtime_geometry_schema == vulkan.env.runtime_geometry_schema
        && cuda.git_revisions.ochroma == vulkan.git_revisions.ochroma
        && cuda.git_revisions.spectra == vulkan.git_revisions.spectra;
    if !same_inputs {
        reasons.push(
            "input identity differs (suite/modes/timing/seed/config/corpus/ABI/schema/revision)"
                .into(),
        );
    }

    let mut packet_match = Some(true);
    let mut control_match = Some(true);
    let mut compared = 0u32;
    let mut unavailable = 0u32;

    for cuda_fixture in cuda.fixtures.iter().filter(|fixture| {
        !required_fixtures()
            .iter()
            .any(|spec| spec.name == fixture.name && spec.authorizing_witness_external)
    }) {
        let Some(vulkan_fixture) = vulkan
            .fixtures
            .iter()
            .find(|fixture| fixture.name == cuda_fixture.name)
        else {
            same_inputs = false;
            reasons.push(format!(
                "Vulkan report missing fixture {}",
                cuda_fixture.name
            ));
            continue;
        };
        if cuda_fixture.identity_hash != vulkan_fixture.identity_hash
            || cuda_fixture.asset_content_hash != vulkan_fixture.asset_content_hash
        {
            same_inputs = false;
            reasons.push(format!(
                "fixture identity/content differs for {}",
                cuda_fixture.name
            ));
        }

        let vulkan_modes = mode_records(vulkan_fixture);
        for (mode_name, cuda_mode) in mode_records(cuda_fixture) {
            let Some((_, vulkan_mode)) = vulkan_modes
                .iter()
                .find(|(candidate, _)| *candidate == mode_name)
            else {
                unavailable += 1;
                packet_match = merge_comparison(packet_match, None);
                control_match = merge_comparison(control_match, None);
                reasons.push(format!(
                    "Vulkan report missing {}/{}",
                    cuda_fixture.name, mode_name
                ));
                continue;
            };
            if cuda_mode.capability_fingerprint.is_none()
                || vulkan_mode.capability_fingerprint.is_none()
            {
                unavailable += 1;
                reasons.push(format!(
                    "capability fingerprint unavailable for {}/{}",
                    cuda_fixture.name, mode_name
                ));
            }
            if cuda_mode.correctness.is_exact() != Some(true)
                || vulkan_mode.correctness.is_exact() != Some(true)
            {
                unavailable += 1;
                reasons.push(format!(
                    "cross-backend correctness unavailable or failed for {}/{}",
                    cuda_fixture.name, mode_name
                ));
            }
            if !cuda_mode.backend.is_honest()
                || !vulkan_mode.backend.is_honest()
                || !cuda_mode.cpu.counters_probed
                || !vulkan_mode.cpu.counters_probed
                || cuda_mode.cpu.any_forbidden_nonzero()
                || vulkan_mode.cpu.any_forbidden_nonzero()
            {
                reasons.push(format!(
                    "backend honesty or CPU-ownership contract failed for {}/{}",
                    cuda_fixture.name, mode_name
                ));
            }
            let packet = match (
                cuda_mode.packet_abi_hash.as_deref(),
                vulkan_mode.packet_abi_hash.as_deref(),
            ) {
                (Some(a), Some(b)) => Some(a == b),
                _ => {
                    unavailable += 1;
                    None
                }
            };
            let control = if mode_name == "city_hybrid" {
                match (
                    cuda_mode.control_hash.as_deref(),
                    vulkan_mode.control_hash.as_deref(),
                ) {
                    (Some(a), Some(b)) => Some(a == b),
                    _ => {
                        unavailable += 1;
                        None
                    }
                }
            } else {
                // Source triangle/reference modes own no persistent
                // MegaGeometry control graph. Their packet ABI is still
                // compared, but absence of an inapplicable control hash is not
                // missing evidence.
                Some(true)
            };
            if packet == Some(false) {
                reasons.push(format!(
                    "packet ABI mismatch for {}/{}",
                    cuda_fixture.name, mode_name
                ));
            }
            if control == Some(false) {
                reasons.push(format!(
                    "control hash mismatch for {}/{}",
                    cuda_fixture.name, mode_name
                ));
            }
            packet_match = merge_comparison(packet_match, packet);
            control_match = merge_comparison(control_match, control);
            if packet.is_some() && control.is_some() {
                compared += 1;
            }
        }
    }

    BackendConformance {
        schema: 1,
        cuda_gpu: cuda.env.gpu.clone(),
        vulkan_gpu: vulkan.env.gpu.clone(),
        same_inputs,
        packet_abi_match: packet_match,
        control_hash_match: control_match,
        compared_mode_records: compared,
        unavailable_records: unavailable,
        reasons,
    }
}
