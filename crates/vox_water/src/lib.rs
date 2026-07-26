//! `vox_water` — the ENGINE's water **surface dynamics** and **sea state**.
//!
//! # Why this crate exists (the architectural move, 2026-07-26)
//!
//! Water simulation used to live in `forge/crates/water/src/sim/` (`forge-water`).
//! It had **zero runtime callers** and could never have had any: project LAW makes
//! Forge an *independent authoring oracle* and forbids the game from depending on
//! it (`cargo tree -p urban_horizon | grep -c forge` must be `0`). A **runtime**
//! simulation living in Forge is therefore unreachable by construction — that is
//! *why* it was dead, not a coincidence.
//!
//! The split this crate implements:
//!
//! | Layer | Owns | Where |
//! |---|---|---|
//! | **Authoring** (cook-time, runs once) | carved river beds, shoreline geometry, baked drainage flow maps | `forge-water`, `forge-terrain` |
//! | **Simulation** (runs every tick/frame, game-agnostic) | sea state, shore response, ripples, wakes, buoyancy | **`vox_water`** (here) |
//! | **Bed hydrology** (per-tick depth/flow over a heightfield) | `WaterField` — depth delta, Saint-Venant flow, contaminant advection, replay fold | `vox_terrain::water` |
//! | **Gameplay coupling** | rain → flood → wellbeing, storm → safety, wind → look | `urban_horizon` |
//!
//! ## Why bed hydrology stayed in `vox_terrain`
//!
//! [`vox_terrain::water::WaterField`] is a mature, replay-hashed, **already-wired**
//! per-tick solver over a heightfield bed. It is not dead and did not need moving;
//! relocating 1 200 lines of live, load-bearing sim would have been pure risk for
//! zero behavioural gain. `vox_water` is deliberately the layer *above* the bed: it
//! knows about a **water surface** and the **wind over it**, never about terrain.
//! Merging the two under this crate is a clean follow-up, not a prerequisite.
//!
//! ## What was deliberately NOT ported from `forge-water`
//!
//! * **`ShallowWaterSolver`** — REDUNDANT and strictly inferior to
//!   [`vox_terrain::water::WaterField`]. It is `f32`, uses a **periodic (toroidal)
//!   viscosity stencil** that wraps momentum across opposite map edges, wraps water
//!   sources with a modulo instead of clamping, has no dry-bank handling, no
//!   equilibrium snap, no settled gate and no replay-hash fold. `WaterField` is
//!   `f64`, no-flux edged, dry-bank clamped, double-buffered for order-independence
//!   and folded into the `ReplayHasher` moat. Porting it would have added a second,
//!   worse hydrology.
//! * **`RiverFlowGenerator` / `FlowMap`** — this is **cook-time authoring** (terrain
//!   drainage gradient → baked flow map + waterfall sites). It runs once, not per
//!   tick, so by the table above it belongs in Forge and stays there. Forge already
//!   has `terrain/src/flow_maps.rs` besides it.
//!
//! # Determinism contract (LAW)
//!
//! Every item here is a **pure function of its inputs** — no interior mutability,
//! no `HashMap` iteration, no RNG, no wall-clock read. Grid passes iterate a fixed
//! `0..n` row-major order and never read a value they wrote in the same pass. That
//! makes each piece *safe to use on either side* of the sim/render line; what
//! decides the side is the **caller's time source**:
//!
//! * [`sea_state`] — input (`wind_speed`) is replay-hashed sim truth; the output is
//!   consumed **render-only** (cosmetic surface amplitude/foam). It never writes
//!   back into sim state, so it cannot perturb a replay. See [`SeaState`].
//! * [`shore`], [`buoyancy`] — deterministic and **gameplay-safe**: fold their
//!   outputs into the replay hash if you drive gameplay with them.
//! * [`ripples`], [`wake`] — evaluated at a `time` the caller supplies. Driven from
//!   **render wall-clock** they are cosmetic; driven from the **sim tick** they are
//!   replay-exact. Both are legitimate; be explicit which one you picked.

pub mod buoyancy;
pub mod ripples;
pub mod sea_state;
pub mod shore;
pub mod wake;

pub use buoyancy::Buoyancy;
pub use ripples::{Ripple, RippleSimulator};
pub use sea_state::{SeaState, WindResponse};
pub use shore::{ShoreConfig, ShoreInteraction, ShorelineResult};
pub use wake::WakeGenerator;

/// Standard gravity (m/s²) — the single constant every water model here shares.
pub const GRAVITY: f32 = 9.81;
