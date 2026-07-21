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

/// The two real Forge-authored structural families the ray-native / fabric
/// claims require (the corpus families in `mega_geometry_corpus.ron`).
pub const REAL_ASSET_FAMILIES: [&str; 2] = ["repeated_highrise", "detached_residential"];

/// Whether the harness can currently prove `city_hybrid` resolves to a DISTINCT
/// execution representation from `reference`. It cannot today (`MEGAGEOMETRY_MODE`
/// is not consumed by the renderer), so this is `false` and every target fixture
/// is `Unavailable`. Routing the harness's distinct-execution flag through this
/// pinned constant means a stray flip-to-true is caught by a test
/// (`harness_does_not_yet_prove_distinct_execution`) in addition to the redundant
/// `mode_distinct` global gate.
pub const HARNESS_CITY_HYBRID_DISTINCT_EXECUTION: bool = false;

/// Whether the forbidden CPU-ownership counters are actually bridged/probed in
/// the harness's live path. They are not yet (Spectra's probed
/// `geometry_cpu_contract` counters are not wired into vox_render's
/// `geometry_cpu_ownership`), so this is `false` and the CPU-ownership gate
/// suppresses (the zero counters are not evidence).
pub const HARNESS_CPU_OWNERSHIP_COUNTERS_PROBED: bool = false;

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
        let same_surface =
            (cd[i] - od[i]).abs() <= PROV_DEPTH_REL_TOL * od[i].abs().max(1.0);
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
    required
        .iter()
        .all(|r| names.iter().any(|n| n == &r.name))
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
                ))
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
                return Some(format!("host submit {v:.3}ms > {HOST_SUBMIT_BUDGET_MS}ms"))
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
    pub cook_schema: u32,
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
            self.resident_mb + self.as_mb? + self.page_mb? + self.native_args_mb? + self.scratch_mb?,
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
        match (self.material_mismatches, self.uv_mismatches, self.hit_mismatches) {
            (Some(m), Some(u), Some(h)) => Some(
                self.mapped_primitives == self.source_primitives && m == 0 && u == 0 && h == 0,
            ),
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

// ---------------------------------------------------------------------------
// Fixture outcome
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum FixtureOutcome {
    Parity { regression_pct: f64 },
    CorrectnessOnly { mismatches: u64 },
    Win { winner: Mode },
    Loss { reason: String },
    CorrectnessFailure { reason: String },
    /// Evidence required to score this fixture is not available — suppresses.
    Unavailable { reason: String },
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
            }
        }
        Some(false) => {
            return FixtureOutcome::CorrectnessFailure {
                reason: "city_hybrid hit/material/uv/mapping mismatch".into(),
            }
        }
        Some(true) => {}
    }
    match reference.correctness.is_exact() {
        None => {
            return FixtureOutcome::Unavailable {
                reason: "reference correctness oracle unavailable".into(),
            }
        }
        Some(false) => {
            return FixtureOutcome::CorrectnessFailure {
                reason: "reference hit/material/uv/mapping mismatch".into(),
            }
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
                    reason: "city_hybrid is not yet a distinct execution path from reference".into(),
                };
            }
            if hybrid.unexpected_fallback {
                return FixtureOutcome::Loss {
                    reason: "unexpected fallback on target fixture".into(),
                };
            }
            let (hp, rp) =
                match (hybrid.vram.peak_geometry_mb(), reference.vram.peak_geometry_mb()) {
                    (Some(h), Some(r)) => (h, r),
                    _ => {
                        return FixtureOutcome::Unavailable {
                            reason: "peak geometry VRAM breakdown unavailable".into(),
                        }
                    }
                };
            let faster =
                hybrid.stage.geometry_stage_ms_p95 < reference.stage.geometry_stage_ms_p95;
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
    pub repeated_storage_ratio: f64,
    pub mixed_storage_ratio: f64,
    pub fixed_budget_detail_ratio: f64,
    pub lod_temporal_invalidations_reduction: f64,
    pub full_frame_regression_pct: f64,
    pub losses: u32,
    pub axis_storage_bytes_win: bool,
    pub axis_peak_vram_win: bool,
    pub axis_geometry_stage_win: bool,
    pub axis_visible_detail_win: bool,
    pub axis_lod_temporal_win: bool,
}

impl ProgramBreakthrough {
    /// The fully-unavailable default: every claim gate fails.
    pub fn unavailable() -> Self {
        ProgramBreakthrough {
            repeated_storage_ratio: 1.0,
            mixed_storage_ratio: 1.0,
            fixed_budget_detail_ratio: 0.0,
            lod_temporal_invalidations_reduction: 0.0,
            full_frame_regression_pct: 100.0,
            losses: 0,
            axis_storage_bytes_win: false,
            axis_peak_vram_win: false,
            axis_geometry_stage_win: false,
            axis_visible_detail_win: false,
            axis_lod_temporal_win: false,
        }
    }

    pub fn independent_axis_wins(&self) -> u32 {
        [
            self.axis_storage_bytes_win,
            self.axis_peak_vram_win,
            self.axis_geometry_stage_win,
            self.axis_visible_detail_win,
            self.axis_lod_temporal_win,
        ]
        .iter()
        .filter(|&&w| w)
        .count() as u32
    }

    pub fn claim(&self) -> Option<&'static str> {
        let thresholds = self.losses == 0
            && self.repeated_storage_ratio <= 0.25
            && self.mixed_storage_ratio <= 0.50
            && self.fixed_budget_detail_ratio >= 2.0
            && self.lod_temporal_invalidations_reduction >= 0.50
            && self.full_frame_regression_pct <= 2.0;
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
    pub ray_native_coverage: f64,
    pub materialized_geometry_ratio: f64,
    pub parameter_update_fine_as_builds: u64,
    pub geometry_stage_speedup: f64,
    pub fixed_budget_detail_ratio: f64,
    pub ray_native_temporal_invalidations: u64,
    pub correctness_mismatches: u64,
    pub real_asset_families: u32,
    pub losses: u32,
    pub procedural_intersection_win: bool,
    pub removed_triangles_counted_as_axis: bool,
    pub as_bytes_counted_as_axis: bool,
}

impl VisibilityBreakthrough {
    pub fn unavailable() -> Self {
        VisibilityBreakthrough {
            ray_native_coverage: 0.0,
            materialized_geometry_ratio: 1.0,
            parameter_update_fine_as_builds: 0,
            geometry_stage_speedup: 0.0,
            fixed_budget_detail_ratio: 0.0,
            ray_native_temporal_invalidations: 0,
            correctness_mismatches: 0,
            real_asset_families: 0,
            losses: 0,
            procedural_intersection_win: false,
            removed_triangles_counted_as_axis: false,
            as_bytes_counted_as_axis: false,
        }
    }

    fn double_counts(&self) -> bool {
        self.removed_triangles_counted_as_axis && self.as_bytes_counted_as_axis
    }

    pub fn claim(&self) -> Option<&'static str> {
        let ok = self.ray_native_coverage >= 0.60
            && self.materialized_geometry_ratio <= 0.25
            && self.parameter_update_fine_as_builds == 0
            && self.geometry_stage_speedup >= 1.50
            && self.fixed_budget_detail_ratio >= 4.0
            && self.ray_native_temporal_invalidations == 0
            && self.correctness_mismatches == 0
            && self.real_asset_families >= 2
            && self.losses == 0
            && self.procedural_intersection_win
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

// ---------------------------------------------------------------------------
// Evidence-line formatting (exact MEGAGEOMETRY_* schema). Unmeasured `Option`
// values print `unavailable`, never a synthetic zero.
// ---------------------------------------------------------------------------

fn opt_f(o: Option<f64>) -> String {
    o.map(|v| format!("{v:.4}")).unwrap_or_else(|| "unavailable".into())
}
fn opt_u(o: Option<u64>) -> String {
    o.map(|v| v.to_string()).unwrap_or_else(|| "unavailable".into())
}
fn opt_u32(o: Option<u32>) -> String {
    o.map(|v| v.to_string()).unwrap_or_else(|| "unavailable".into())
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
    pub cook_schema: u32,
}

impl EnvLine {
    pub fn line(&self) -> String {
        format!(
            "MEGAGEOMETRY_ENV gpu={} api={} driver={} rt_backend={} capability_hash={} shader_abi={} cook_schema={}",
            self.gpu, self.api, self.driver, self.rt_backend, self.capability_hash, self.shader_abi, self.cook_schema
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
        c.source_primitives, c.mapped_primitives, opt_u(c.material_mismatches), opt_u(c.uv_mismatches), opt_u(c.hit_mismatches)
    )
}

pub fn result_line(s: &GeometryStage, v: &VramBreakdown) -> String {
    format!(
        "MEGAGEOMETRY_RESULT build_ms_p50={:.4} build_ms_p95={:.4} update_ms_p50={:.4} update_ms_p95={:.4} native_lowering_ms_p95={} trace_ms_p50={:.4} trace_ms_p95={:.4} frame_ms_p50={:.4} frame_ms_p95={:.4} resident_vram_mb={:.3} as_vram_mb={} page_vram_mb={} scratch_vram_mb={} peak_geometry_vram_mb={} traced_triangles={} required_page_misses={}",
        s.build_ms_p50, s.build_ms_p95, s.update_ms_p50, s.update_ms_p95, opt_f(s.native_lowering_ms_p95),
        s.trace_ms_p50, s.trace_ms_p95, s.frame_ms_p50, s.frame_ms_p95,
        v.resident_mb, opt_f(v.as_mb), opt_f(v.page_mb), opt_f(v.scratch_mb), opt_f(v.peak_geometry_mb()),
        s.traced_triangles, opt_u(s.required_page_misses)
    )
}

pub fn cpu_line(c: &CpuCounters) -> String {
    format!(
        "MEGAGEOMETRY_CPU render_thread_scene_visits={} visibility_node_visits={} construction_evaluations={} lod_decisions={} page_decisions={} eviction_decisions={} blocking_readbacks={} frame_allocations={} host_deadline_misses={} render_thread_submit_ms_p95={} host_submit_calls={} host_submit_ms_p95={}",
        c.render_thread_scene_visits, c.visibility_node_visits, c.construction_evaluations,
        c.lod_decisions, c.page_decisions, c.eviction_decisions, c.blocking_readbacks,
        c.frame_allocations, opt_u(c.host_deadline_misses), opt_f(c.render_thread_submit_ms_p95),
        c.host_submit_calls, opt_f(c.host_submit_ms_p95)
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
            self.mode, opt_f(self.source_surface_coverage), opt_f(self.residual_coverage),
            opt_u(self.coarse_aabb_bytes), opt_u(self.materialized_geometry_bytes),
            opt_u(self.fine_as_builds), opt_f(self.intersection_ms_p95),
            opt_f(self.geometry_stage_ms_p95), self.stable_surface_hash
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
            self.mode, opt_u(self.input_rays), opt_u(self.structured_rays), opt_u(self.certified_rays),
            opt_u(self.subdivided_domains), opt_u(self.explicit_rays), opt_f(self.active_packet_lanes),
            opt_f(self.routing_ms_p95), opt_f(self.certification_ms_p95), opt_f(self.packet_ms_p95),
            opt_f(self.refinement_ms_p95), opt_f(self.residual_rt_ms_p95), opt_f(self.merge_ms_p95),
            opt_u(self.queue_overflows), opt_u(self.cpu_scheduling), self.hit_hash
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
        "MEGAGEOMETRY_BREAKTHROUGH repeated_storage_ratio={:.4} mixed_storage_ratio={:.4} fixed_budget_detail_ratio={:.4} lod_temporal_invalidations_reduction={:.4} winning_axes={} claim={}",
        b.repeated_storage_ratio,
        b.mixed_storage_ratio,
        b.fixed_budget_detail_ratio,
        b.lod_temporal_invalidations_reduction,
        b.winning_axes(),
        b.claim().unwrap_or("experimental_subset")
    )
}

pub fn visibility_breakthrough_line(b: &VisibilityBreakthrough) -> String {
    format!(
        "MEGAGEOMETRY_VISIBILITY_BREAKTHROUGH ray_native_coverage={:.4} materialized_geometry_ratio={:.4} parameter_update_fine_as_builds={} geometry_stage_speedup={:.4} fixed_budget_detail_ratio={:.4} ray_native_temporal_invalidations={} correctness_mismatches={} real_asset_families={} losses={} claim={}",
        b.ray_native_coverage, b.materialized_geometry_ratio, b.parameter_update_fine_as_builds,
        b.geometry_stage_speedup, b.fixed_budget_detail_ratio, b.ray_native_temporal_invalidations,
        b.correctness_mismatches, b.real_asset_families, b.losses,
        b.claim().unwrap_or("experimental_subset")
    )
}

pub fn fabric_breakthrough_line(b: &FabricBreakthrough) -> String {
    format!(
        "MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage={:.4} certified_without_per_ray_traversal={:.4} active_packet_lanes={:.4} intersection_speedup_vs_scalar={:.4} geometry_stage_speedup={:.4} routing_queue_overhead_pct={:.4} correctness_mismatches={} cpu_scheduling={} real_asset_families={} losses={} claim={}",
        b.structured_ray_coverage, b.certified_without_per_ray_traversal, b.active_packet_lanes,
        b.intersection_speedup_vs_scalar, b.geometry_stage_speedup, b.routing_queue_overhead_pct,
        b.correctness_mismatches, b.cpu_scheduling, b.real_asset_families, b.losses,
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
    pub forge: String,
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
