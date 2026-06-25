//! vox_sim — the game-agnostic simulation ENGINE spine.
//!
//! This crate holds ONLY generic, reusable simulation machinery — nothing that
//! knows what a building, zone, citizen, road, or city *is*. The city-builder
//! game logic that used to live here was split DOWN into the game during the
//! 2026-06-25 engine/game boundary cleanup and now lives in
//! `urban_horizon::sim`. Keep it that way: the engine must stay reusable for
//! other games (a space game, an RPG), so NO building/zone/citizen/city concept
//! may return to this crate.

// --- Generic agent + crowd steering (game-agnostic movers) ---
pub mod agent;
pub mod crowd;

// --- Generic time model (pure-integer calendar over a u64 master tick) ---
pub mod calendar;

// --- Determinism + sharding + spatial machinery (the replay moat) ---
pub mod deterministic;
pub mod sharding;
pub mod spatial_hash;
// Game-agnostic state-fingerprint primitive (`FnvHasher`); the concrete
// city-state fold lives game-side in `urban_horizon::sim::sim_hash`.
pub mod sim_hash;

// The canonical shared sim vocabulary (beyond-CS2 cim-sim/economy/wellbeing
// seam, design 2026-06-24 §5 build-order A1): DistrictId/CohortId/Money/
// SkillTier/Relevance/EventKind/PresenceLedger/assert_population_conserved.
// Game-agnostic engine primitives the game re-exports for its cim/ callers.
pub mod sim_types;
// The game-agnostic cim-sim wake-queue spine (design 2026-06-24 §5 build-order
// A2): queue/store/cohort/lod/schedule/report/presence/hash, ported DOWN from
// urban_horizon/src/cim/ unchanged. The game re-exports it and keeps its
// orchestrator (CimSim) + care-gate game-side.
pub mod cim_sim;
