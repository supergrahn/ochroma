//! Deterministic jitter for the cim layer.
//!
//! Mirrors the canonical `splitmix64` used elsewhere in the game (e.g.
//! `ownership::splitmix64`). This is the ONLY source of "randomness" in
//! `src/cim/`: it is always a *value* derived from `(id, day)`, never an
//! iteration order and never a branch on wall-clock — so two replays of the
//! same action log produce bit-identical schedules (integration brief §9 rule 2).

/// SplitMix64 finalizer — the deterministic hash behind cim jitter.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Per-cim, per-day deterministic value. `hash(id, day)` mixes the cim id with
/// the day index so a cim's daily schedule jitter is stable on replay yet varies
/// cim-to-cim and day-to-day.
pub fn hash(id: u32, day: u64) -> u64 {
    splitmix64(id as u64 ^ splitmix64(day))
}
