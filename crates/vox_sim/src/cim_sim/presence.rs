//! Presence-delta ledger — the cim-side wiring of [`crate::sim_types::PresenceLedger`]
//! into the day-chain `ArriveWork`/`LeaveWork` handlers.
//!
//! The engine used to track building occupancy by mutating `b.occupants` per
//! citizen. The cim path retires that: when an Individual-tier cim's `ArriveWork`
//! event fires it does a single integer `+1` on its workplace building; its
//! `LeaveWork` does a saturating `-1`. The economy reads `present_fraction(b)`
//! — `(present_count + cohort_fraction) / staff_target` — and NEVER touches
//! `b.occupants`. `firm_productivity` is owned by the economy plan (brief §6);
//! cim code only WRITES `present_fraction`, it never defines or calls the
//! productivity join.
//!
//! Determinism (integration brief §9): an integer `±1` is commutative, so the
//! ledger is order-independent regardless of the SENSE/COMMIT drain order
//! (rule 4, brief §4.6 "integer atomicAdd is bit-exact"). No HashMap, no RNG, no
//! wall-clock. The ledger is replay-rebuilt runtime state, never serialized (no
//! SAVE_VERSION bump).

use crate::sim_types::PresenceLedger;

/// Thin cim-owned wrapper around [`PresenceLedger`]. Exists so the cim module
/// can grow presence-specific behavior (cohort fractional attendance in Task 6)
/// without leaking `sim_types` internals; for now it forwards arrive/leave/read
/// and the staff-target seeding the building loop performs.
pub struct CimPresence {
    ledger: PresenceLedger,
}

impl CimPresence {
    /// A presence ledger sized to `n_buildings` (all columns zero-init). The cim
    /// store grows lazily, so this is sized to the building count, not the cim
    /// count.
    pub fn new(n_buildings: usize) -> Self {
        Self { ledger: PresenceLedger::new(n_buildings) }
    }

    /// Record an Individual-tier cim arriving at workplace building `b` (integer
    /// `+1`). Called by the `ArriveWork` COMMIT handler. Grows the ledger to
    /// cover `b` first so a freshly-seen workplace is addressable.
    pub fn arrive(&mut self, b: u32) {
        self.ledger.ensure_building(b);
        self.ledger.arrive(b);
    }

    /// Record an Individual-tier cim leaving workplace building `b` (saturating
    /// `-1`). Called by the `LeaveWork` COMMIT handler. A `leave` for a building
    /// never seen is a no-op (saturating, and ensured to length so the index is
    /// valid).
    pub fn leave(&mut self, b: u32) {
        self.ledger.ensure_building(b);
        self.ledger.leave(b);
    }

    /// Set building `b`'s staff target (the denominator of `present_fraction`).
    pub fn set_staff_target(&mut self, b: u32, n: u32) {
        self.ledger.ensure_building(b);
        self.ledger.set_staff_target(b, n);
    }

    /// Set building `b`'s fractional cohort attendance (Task 6 cohort path).
    pub fn set_cohort_fraction(&mut self, b: u32, f: f64) {
        self.ledger.ensure_building(b);
        self.ledger.set_cohort_fraction(b, f);
    }

    /// The economy's only occupancy read: `(present_count + cohort_fraction) /
    /// staff_target`, `0.0` when `staff_target == 0`. Never reads `b.occupants`.
    /// A building the ledger has never seen reads `0.0` (no staff target).
    pub fn present_fraction(&self, b: u32) -> f64 {
        if (b as usize) >= self.ledger.len() {
            return 0.0;
        }
        self.ledger.present_fraction(b)
    }

    /// Immutable access to the underlying ledger (for the economy month-settle
    /// read path).
    pub fn ledger(&self) -> &PresenceLedger {
        &self.ledger
    }
}
