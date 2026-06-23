//! Determinism SHIP GATE for the SIM hot loop.
//!
//! Byte-exact replay is the engine's moat. These tests run a fixed-seed
//! `CitySim` for a fixed number of ticks and assert a bit-exact (`to_bits`)
//! FNV-1a hash of the full deterministic sim state (citizens + buildings +
//! agent positions). The hash MUST stay byte-identical across:
//!   - two independent runs in the same process (replay determinism), and
//!   - the recorded golden constant (regression: optimizations must not change
//!     the simulated result, only how fast it is computed).
//!
//! If an optimization changes any of these values, it has broken the moat and
//! the change is rejected — exactly the gate the perf work must pass.

use vox_sim::city_sim::CitySim;
use vox_sim::sim_hash::sim_state_hash;

/// Run a deterministic city for `ticks` and return the full-state hash.
fn run(ticks: u32) -> u64 {
    let mut sim = CitySim::new_small();
    sim.tick(ticks);
    sim_state_hash(&sim)
}

#[test]
fn replay_is_byte_identical_across_runs() {
    // Two independent constructions of the same seed/scene must produce the
    // exact same simulated state. This is the core replay invariant.
    let a = run(200);
    let b = run(200);
    assert_eq!(a, b, "same seed/scene must yield byte-identical sim state");
}

#[test]
fn replay_hash_matches_golden() {
    // GOLDEN regression value, recorded BEFORE the optimization work.
    // Any change to this hash means the simulated outcome moved — forbidden
    // for a pure performance change.
    const GOLDEN_200: u64 = 0x6c6e_f8e5_39fe_b8bf;
    let h = run(200);
    assert_eq!(
        h, GOLDEN_200,
        "sim state hash changed (got {h:#018x}); an optimization altered the \
         simulated result, not just its speed — determinism moat broken"
    );
}

#[test]
fn replay_hash_stable_at_multiple_horizons() {
    // The state must be reproducible at several tick horizons, not just one.
    for &t in &[10u32, 50, 100, 300] {
        let a = run(t);
        let b = run(t);
        assert_eq!(a, b, "non-reproducible state at horizon {t}");
    }
}
