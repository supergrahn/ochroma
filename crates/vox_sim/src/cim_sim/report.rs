//! The headless run report — the single auditable integer the care-gate produces.
//!
//! The thesis of cim-life is that a worker who cannot drop off a dependent does
//! not arrive at work, and the city loses exactly the tax on that lost labor —
//! one integer, derivable two ways that MUST agree:
//!
//! - `lost_tax_per_mo`   — the city ledger figure (the month-settle accumulator).
//! - `followcim_lost_tax_sum` — the same number re-folded per-worker, in id order,
//!   the way the "follow this cim" panel would attribute it.
//!
//! Both read the SAME source absences (the per-worker `absence_hours` column), so
//! `lost_tax_per_mo == followcim_lost_tax_sum` is an invariant, not a coincidence.
//! The fold is id-ordered and integer-only (brief §9 rules 1/4); no HashMap, no
//! RNG, no wall-clock.

use crate::cim_sim::hash::splitmix64;

/// One headless run's headline numbers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Workers who hit a care gap (an uncovered dependent) during the run.
    pub care_gap_workers: u32,
    /// City-ledger lost tax for the month — the month-settle accumulator.
    pub lost_tax_per_mo: u64,
    /// The same lost tax re-folded per-worker (the follow-cim panel sum). Equal to
    /// `lost_tax_per_mo` by construction (same source integers).
    pub followcim_lost_tax_sum: u64,
}

/// The running replay-hash accumulator. It folds the run's *integer* ledger
/// transfers — presence deltas, cohort head-count transfers, and care-gap
/// absences — in id order, via `splitmix64` mixing (brief §9 rule 9). Only
/// integers are folded (no f64 bit patterns, no wall-clock, no ephemeral
/// per-tick timing — rules 6/7), so two replays of the same action log produce a
/// bit-identical hash. The accumulator is order-sensitive *by construction*: the
/// caller MUST fold transfers in a deterministic (id-ordered) sequence, which the
/// cohort/queue drains already guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayHasher {
    state: u64,
}

impl ReplayHasher {
    /// A fresh accumulator (zero state — the empty-run hash).
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Mix one integer transfer into the running hash. The mix is
    /// `state = splitmix64(state ^ splitmix64(value))`, which is associative-free
    /// (order-sensitive) and avalanches every bit — so a single flipped transfer
    /// changes the final hash. `value` carries the transfer's full identity (e.g.
    /// `tick<<32 | id` packed, or a cohort head-count delta) so distinct transfers
    /// never alias.
    pub fn mix(&mut self, value: u64) {
        self.state = splitmix64(self.state ^ splitmix64(value));
    }

    /// The current hash value.
    pub fn finish(&self) -> u64 {
        self.state
    }
}
