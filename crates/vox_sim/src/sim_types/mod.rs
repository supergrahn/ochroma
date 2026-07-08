// crates/vox_sim/src/sim_types/mod.rs
//
// The canonical SHARED simulation vocabulary — the engine seam that the
// beyond-CS2 cim-sim / economy / wellbeing domains all import (design
// `2026-06-24-cim-sim-beyond-cs2-design.md` §5, build-order A1).
//
// These are the GAME-AGNOSTIC primitives (per `CLAUDE.md`: engine crates carry
// NO buildings/zoning/traffic concepts — only generic agent/cohort/event/money/
// presence vocabulary). The engine owns the *mechanism* (these types + the
// conservation invariant); the game (`urban_horizon`) owns the *meaning* (which
// need is mental-health, which `SkillTier` band an education level maps to, the
// care-gate policy) and re-exports these symbols so its existing `cim/` callers
// keep compiling during the A2 spine migration.
//
// Determinism contract (the moat): `DistrictId`/`CohortId` index id-sorted
// `Vec`s (no HashMap/RNG iteration order); `PresenceLedger` deltas are integer
// ±1 (commutative ⇒ order-independent ⇒ replay-exact); `assert_population_
// conserved` is the cross-domain conservation gate run every COMMIT. Pure
// runtime vocabulary — never serialized, so no save-format change.

/// A district index (id-sorted `Vec` key — never a HashMap key).
pub type DistrictId = u32;

/// A cohort index into the conserved, ascending-`id` `CohortTable` (design §4.4).
pub type CohortId = u32;

/// Money in a single fixed unit. `f64` so the fixed-ε determinism rule applies
/// (compared with a fixed epsilon / hashed by `to_bits` in the witness).
pub type Money = f64;

/// A cim's skill band. `as usize` is `0..=3` and MATCHES the game-side labor
/// tier index — the engine defines the band vocabulary; the game owns the
/// education→band mapping (`core-architecture-engine-game-map`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum SkillTier {
    Unskilled,
    Skilled,
    Educated,
    Expert,
}

/// The LOD tier a piece of state is observed at (design §4.4 lossless LOD):
/// statistical cohort mass, an integer presence flow, or a crystallized
/// Individual cim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relevance {
    Cohort,
    Flow,
    Individual,
}

/// The engine's event vocabulary for the wake queue (design §5). `#[repr(u8)]`
/// so an `EventRef` can carry the kind as one byte and the `(wake_tick,id,kind)`
/// drain order is total + deterministic. Variant *order* is load-bearing: it is
/// the tie-break within a single `wake_tick` for a single `id`, so two replays
/// drain identically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EventKind {
    DayRoll,
    Wake,
    DropOff,
    ArriveWork,
    LeaveWork,
    Sleep,
    Birthday,
    Retire,
    CohortDepart,
    CohortReturn,
    MonthSettle,
    BondService,
    EconEvent,
    LeaveLink,
    RouteInvalidate,
    HydrationDrain,
}

/// Integer presence accounting for a set of workplaces/venues (design §4.6).
///
/// Three building-indexed columns. `present_count` is bumped by integer ±1 from
/// Individual-tier cims (`arrive`/`leave`); `cohort_fraction` carries fractional
/// cohort-mass attendance; `staff_target` is the denominator. `present_fraction`
/// = `(present_count + cohort_fraction) / staff_target`. The ±1 deltas are
/// commutative, so the ledger is order-independent and therefore replay-exact —
/// business productivity reads `present_fraction`, not a per-cim sweep.
pub struct PresenceLedger {
    present_count: Vec<u32>, // building-indexed; integer ±1 from Individual-tier cims
    cohort_fraction: Vec<f64>, // building-indexed; fractional cohort attendance
    staff_target: Vec<u32>,  // building-indexed
}

impl PresenceLedger {
    /// A ledger sized for `n_buildings` workplaces, all zero-init.
    pub fn new(n_buildings: usize) -> Self {
        Self {
            present_count: vec![0; n_buildings],
            cohort_fraction: vec![0.0; n_buildings],
            staff_target: vec![0; n_buildings],
        }
    }

    /// Grow all three building-indexed columns so building index `b` is valid
    /// (zero-init the new slots). A no-op when `b` is already addressable. Lets
    /// the cim path size the ledger lazily as workplaces are first seen without
    /// the building count being known at `CimSim::new` time.
    pub fn ensure_building(&mut self, b: u32) {
        let want = b as usize + 1;
        if self.present_count.len() < want {
            self.present_count.resize(want, 0);
            self.cohort_fraction.resize(want, 0.0);
            self.staff_target.resize(want, 0);
        }
    }

    /// One Individual-tier cim arrives at workplace `b` (integer +1).
    pub fn arrive(&mut self, b: u32) {
        self.present_count[b as usize] += 1;
    }

    /// One Individual-tier cim leaves workplace `b` (saturating integer -1).
    pub fn leave(&mut self, b: u32) {
        let s = &mut self.present_count[b as usize];
        *s = s.saturating_sub(1);
    }

    /// Set the fractional cohort-mass attendance at workplace `b`.
    pub fn set_cohort_fraction(&mut self, b: u32, f: f64) {
        self.cohort_fraction[b as usize] = f;
    }

    /// Set the staffing target (the `present_fraction` denominator) at `b`.
    pub fn set_staff_target(&mut self, b: u32, n: u32) {
        self.staff_target[b as usize] = n;
    }

    /// Number of building slots the ledger currently addresses.
    pub fn len(&self) -> usize {
        self.present_count.len()
    }

    /// True when the ledger addresses no buildings.
    pub fn is_empty(&self) -> bool {
        self.present_count.is_empty()
    }

    /// Staffed fraction at workplace `b`: `(present_count + cohort_fraction) /
    /// staff_target`, or `0.0` when no staff target is set.
    pub fn present_fraction(&self, b: u32) -> f64 {
        let t = self.staff_target[b as usize];
        if t == 0 {
            return 0.0;
        }
        (self.present_count[b as usize] as f64 + self.cohort_fraction[b as usize]) / t as f64
    }
}

/// The cross-domain population-conservation invariant (design §4.4): the sum of
/// crystallized Individual-tier cims (`hydrated`) and the integer cohort-mass
/// head count must always equal the total population `pop`. Run every COMMIT —
/// a violation is a determinism/conservation bug, not a recoverable state, so
/// it panics.
pub fn assert_population_conserved(hydrated: u64, cohort_head_count_int: u64, pop: u64) {
    assert_eq!(
        hydrated + cohort_head_count_int,
        pop,
        "population leak: hydrated {hydrated} + cohort {cohort_head_count_int} != pop {pop}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_tier_order_is_ascending_band_index() {
        // The `as usize` value IS the labor tier index (0..=3); ordering is
        // load-bearing for id-sorted folds over tiers.
        assert_eq!(SkillTier::Unskilled as usize, 0);
        assert_eq!(SkillTier::Skilled as usize, 1);
        assert_eq!(SkillTier::Educated as usize, 2);
        assert_eq!(SkillTier::Expert as usize, 3);
        assert!(SkillTier::Unskilled < SkillTier::Expert);
    }

    #[test]
    fn event_kind_repr_u8_is_stable_drain_tiebreak() {
        // Variant order is the within-(wake_tick,id) drain tie-break; pin the
        // byte values so a reorder is caught (it would change the replay hash).
        assert_eq!(EventKind::DayRoll as u8, 0);
        assert_eq!(EventKind::Wake as u8, 1);
        assert_eq!(EventKind::Sleep as u8, 5);
        assert_eq!(EventKind::CohortReturn as u8, 9);
        assert_eq!(EventKind::HydrationDrain as u8, 15);
    }

    #[test]
    fn presence_fraction_is_integer_delta_and_order_independent() {
        let mut a = PresenceLedger::new(1);
        a.set_staff_target(0, 10);
        // Two arrivals then one leave (count stays >= 0 throughout).
        a.arrive(0);
        a.arrive(0);
        a.leave(0);

        let mut b = PresenceLedger::new(1);
        b.set_staff_target(0, 10);
        // Same multiset of deltas in a different valid order — never underflows,
        // so the commutative integer deltas land on the same count.
        b.arrive(0);
        b.leave(0);
        b.arrive(0);

        assert_eq!(a.present_fraction(0), b.present_fraction(0));
        // 1 present of 10 target, no cohort fraction.
        assert!((a.present_fraction(0) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn present_fraction_zero_when_no_staff_target() {
        let mut p = PresenceLedger::new(1);
        p.arrive(0);
        assert_eq!(p.present_fraction(0), 0.0);
    }

    #[test]
    fn present_fraction_includes_cohort_mass() {
        let mut p = PresenceLedger::new(1);
        p.set_staff_target(0, 4);
        p.arrive(0); // 1 individual
        p.set_cohort_fraction(0, 1.0); // + 1.0 cohort mass
        // (1 + 1.0) / 4 = 0.5
        assert!((p.present_fraction(0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ensure_building_grows_lazily_and_is_noop_when_addressable() {
        let mut p = PresenceLedger::new(0);
        assert!(p.is_empty());
        p.ensure_building(2); // grow to address index 2 -> len 3
        assert_eq!(p.len(), 3);
        p.set_staff_target(2, 5);
        p.arrive(2);
        p.ensure_building(0); // no-op (already addressable), must not clobber
        assert_eq!(p.len(), 3);
        assert!((p.present_fraction(2) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn population_conserved_holds_when_hydrated_plus_cohort_equals_pop() {
        assert_population_conserved(3, 999_997, 1_000_000);
    }

    #[test]
    #[should_panic(expected = "population leak")]
    fn population_conserved_panics_on_leak() {
        assert_population_conserved(3, 999_996, 1_000_000); // sums to 999_999, not 1M
    }
}
