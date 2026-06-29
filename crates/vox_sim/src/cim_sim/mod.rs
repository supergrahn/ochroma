//! Game-agnostic cim-sim wake-queue spine (the engine substrate).
//!
//! This is the proven wake-queue spine ported DOWN from the game
//! (`urban_horizon/src/cim/`) into the engine crate per the beyond-CS2 cim-sim
//! design (2026-06-24 §5 build-order A2): the engine owns the MECHANISM, the
//! game owns the MEANING (`core-architecture-engine-game-map`). It carries NO
//! game concepts (no buildings/zoning/care): only the generic agent/cohort/
//! event/presence primitives.
//!
//! Contents (all ported byte-for-byte — A2 is a NO-BEHAVIOR-CHANGE move, the
//! golden replay-hash stays bit-identical; feature deltas land in later steps):
//!
//! - [`queue`] — bucket-ring + spill-heap [`queue::WakeQueue`], `(wake_tick, id,
//!   kind)`-ordered. No hashed-container iteration, no RNG, no wall-clock.
//! - [`store`] — SoA [`store::CimStore`] with closed-form (on-wake) need
//!   catch-up (one FMA per read).
//! - [`cohort`] — conserved, id-sorted [`cohort::CohortTable`] (cohort-mass LOD).
//! - [`lod`] — Cohort↔Individual crystallize/demote with the bit-identity
//!   guarantee.
//! - [`schedule`] — the per-cim self-rescheduling day chain.
//! - [`report`] — the [`report::ReplayHasher`] integer-ledger fold (the moat).
//! - [`presence`] — the [`presence::CimPresence`] integer ±1 occupancy ledger.
//! - [`hash`] — the deterministic `splitmix64`/`hash(id, day)` jitter source.
//!
//! The orchestrator (`CimSim`) and the genuinely game-specific pieces
//! (care-gate, crystallize-on-observe naming, visible-cim render packing) stay
//! game-side and consume this spine.

pub mod cohort;
pub mod hash;
pub mod lod;
pub mod presence;
pub mod queue;
pub mod report;
pub mod schedule;
pub mod store;

/// Civic ticks per day (6 ticks/hour × 24h). The engine-side constant the cim
/// spine integrates against; matches the game's `time::TICKS_PER_DAY = 144`
/// exactly, so the ported spine's closed-form need catch-up and day-chain
/// offsets are byte-identical to the game-side prototype this was lifted from
/// (the A2 no-behavior-change contract). A value, never wall-clock.
pub const TICKS_PER_DAY: u64 = 144;
