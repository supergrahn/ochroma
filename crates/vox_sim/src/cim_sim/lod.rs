//! Behavior-LOD: Cohort ↔ Flow ↔ Individual crystallize/dissolve, with the
//! bit-identity guarantee.
//!
//! ~99.9% of the city lives as cohort mass. A cim is *promoted* to Individual
//! tier only when something needs its individual state (a follow-cim panel, a
//! contingent care-gate branch). The contract (integration brief §4.4) is that a
//! cim crystallized from its cohort reconstructs EXACTLY the state an
//! always-Individual run would have produced — so demote-then-promote is
//! bit-identical to never-demoted.
//!
//! This holds because the reconstruction is path-independent:
//!   1. The cim's home cohort is a pure function of its id ([`CohortTable::cohort_index_for_cim`]).
//!   2. Its start-of-day need values are seeded from the cohort's `need_mean`
//!      jittered by `hash(cim_id, day)` — a *value*, never an order or a branch
//!      (brief §9 rule 2), so the seed is identical no matter when the cim is
//!      hydrated within the day.
//!   3. Closed-form needs (Task 1) advance the seed to `up_to_tick` with one FMA
//!      that depends ONLY on elapsed time — no intermediate accumulation, so the
//!      result is the same whether the cim was integrated step-by-step or jumped
//!      straight to `up_to_tick`.
//!
//! There is therefore no per-tick state to lose on demote and nothing to
//! reconstruct incorrectly on promote: both paths run the exact same closed-form
//! from the exact same hash-seeded start.

use crate::cim_sim::cohort::CohortTable;
use crate::cim_sim::hash::hash;
use crate::cim_sim::store::{NEED_FOOD, NEED_HEALTH};
use crate::cim_sim::TICKS_PER_DAY;

/// The crystallized per-cim state the LOD layer reconstructs. Field-by-field
/// comparable (`PartialEq`) so the bit-identity test can assert
/// `state_fine == state_promoted` directly. Holds the two needs the test reads;
/// every channel is reconstructed from the same closed-form so adding channels
/// here cannot break path-independence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CimState {
    /// Cim id this state belongs to (part of the identity comparison).
    pub id: u32,
    /// Closed-form food need at the reconstructed tick.
    pub food: f32,
    /// Closed-form health need at the reconstructed tick.
    pub health: f32,
    /// The tick this state was reconstructed at (part of the identity comparison).
    pub tick: u64,
}

/// Per-need closed-form parameters seeded for a hydrated individual: the
/// start-of-day value and the per-day drift. Reconstructed purely from the
/// cohort distribution + `hash(cim_id, day)` (path-independent).
#[derive(Clone, Copy, Debug, PartialEq)]
struct SeededNeed {
    value_at_day_start: f32,
    rate_per_day: f32,
}

/// Seed one need channel for cim `cim_id` on `day` from its cohort distribution.
/// The value is `need_mean[need] + jitter`, where `jitter` is a deterministic
/// band derived from `hash(cim_id ^ need_salt, day)` scaled by the cohort's
/// per-need standard deviation. The rate is a small deterministic drift in the
/// same hash family. Pure function of `(cohort, cim_id, day, need)` — never a
/// branch on wall-clock, never an iteration order (brief §9 rule 2).
fn seed_need(cohorts: &CohortTable, cim_id: u32, day: u64, need: usize) -> SeededNeed {
    let idx = cohorts
        .cohort_index_for_cim(cim_id)
        .expect("LOD reconstruction requires seeded cohorts");
    let row = &cohorts.rows()[idx];
    let mean = row.need_mean[need];
    let std = row.need_var[need].max(0.0).sqrt();

    // Per-need salt keeps the food/health jitter independent yet deterministic.
    let salt = (need as u32).wrapping_mul(0x9E37);
    let h = hash(cim_id ^ salt, day);
    // Map the high bits to a centered fraction in [-0.5, 0.5], scaled by std.
    let frac = ((h >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5;
    let value_at_day_start = (mean + frac * std).clamp(0.0, 1.0);

    // Deterministic per-day drift in [-0.02, 0.00] per day (needs decay across
    // the day); independent low bits of the same hash.
    let drift_frac = (h & 0xFFFF) as f32 / 65535.0; // 0..1
    let rate_per_day = -0.02 * drift_frac;

    SeededNeed { value_at_day_start, rate_per_day }
}

/// Reconstruct a need's closed-form value at `up_to_tick` for a cim hydrated on
/// `day`. Identical FMA to [`crate::cim_sim::store::CimStore::need_now`]: the
/// elapsed-time advance depends ONLY on `up_to_tick - day_start`, so it is the
/// same whether the cim was stepped tick-by-tick or jumped straight here.
fn need_at(cohorts: &CohortTable, cim_id: u32, day: u64, need: usize, up_to_tick: u64) -> f32 {
    let s = seed_need(cohorts, cim_id, day, need);
    let day_start = day * TICKS_PER_DAY;
    let elapsed_ticks = up_to_tick.saturating_sub(day_start);
    let elapsed_days = elapsed_ticks as f32 / TICKS_PER_DAY as f32;
    (s.value_at_day_start + s.rate_per_day * elapsed_days).clamp(0.0, 1.0)
}

/// Crystallize cim `cim_id` to Individual tier and reconstruct its state at
/// `up_to_tick`. The defining LOD operation: pop one unit of cohort mass (the
/// caller does the conserved `hydrate` bookkeeping) and rebuild the individual's
/// closed-form needs from the cohort distribution + `hash(cim_id, day)`. Because
/// the reconstruction is path-independent, the result is bit-identical whether
/// the cim was hydrated at the start of the day or promoted mid-day from cohort
/// mass (brief §4.4).
pub fn promote(cohorts: &CohortTable, cim_id: u32, day: u64, up_to_tick: u64) -> CimState {
    CimState {
        id: cim_id,
        food: need_at(cohorts, cim_id, day, NEED_FOOD, up_to_tick),
        health: need_at(cohorts, cim_id, day, NEED_HEALTH, up_to_tick),
        tick: up_to_tick,
    }
}

/// Demote cim `cim_id` back into cohort mass. The conserved `dissolve`
/// bookkeeping is the caller's; LOD-wise there is nothing to persist because the
/// individual carries no path-dependent state beyond its hash-seeded closed-form
/// (which `promote` reconstructs identically). Returning the cohort index the cim
/// folds back into lets `set_relevance` assert the presence contribution is
/// unchanged across the transition.
pub fn demote(cohorts: &CohortTable, cim_id: u32) -> usize {
    cohorts
        .cohort_index_for_cim(cim_id)
        .expect("LOD demotion requires seeded cohorts")
}
