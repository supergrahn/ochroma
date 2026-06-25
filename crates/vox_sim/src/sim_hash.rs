//! Deterministic, bit-exact state hashing for the SIM replay moat.
//!
//! The engine's competitive moat is byte-exact replay. To defend it under
//! performance work we need a cheap, fully deterministic fingerprint of the
//! simulated state that:
//!   * folds values in a FIXED, id-sorted order (never HashMap/RNG order),
//!   * compares floats by their raw bits (`to_bits`), so a hash collision can
//!     never hide a one-ULP drift introduced by reordering a float sum,
//!   * touches only the reproducible state (citizen + building fields and the
//!     id-ordered agent positions), never the RNG-seeded `Uuid` agent keys.
//!
//! This is the witness the perf changes are measured against: record the
//! golden hash BEFORE, and assert it is unchanged AFTER each optimization.
//!
//! This module holds ONLY the game-agnostic [`FnvHasher`] primitive. The
//! game-specific fold over a concrete city state (citizens/buildings/agents)
//! lives in the game layer (`urban_horizon::sim::sim_hash::sim_state_hash`),
//! because the engine must not know what a citizen or a building is.

/// FNV-1a 64-bit, fed byte-by-byte. Order-sensitive by construction, which is
/// exactly what we want: any reordering of the folded state shows up as a
/// different hash, catching nondeterministic iteration order.
#[derive(Debug, Clone, Copy)]
pub struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub fn new() -> Self {
        Self { state: Self::OFFSET }
    }

    #[inline]
    pub fn write_u8(&mut self, b: u8) {
        self.state ^= b as u64;
        self.state = self.state.wrapping_mul(Self::PRIME);
    }

    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.write_u8(b);
        }
    }

    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.write_u8(b);
        }
    }

    /// Hash a float by its raw bits — NaN-stable and ULP-exact.
    #[inline]
    pub fn write_f32(&mut self, v: f32) {
        self.write_u32(v.to_bits());
    }

    #[inline]
    pub fn write_f64(&mut self, v: f64) {
        self.write_u64(v.to_bits());
    }

    /// Fold an `Option<u32>` deterministically (tag byte + payload).
    #[inline]
    pub fn write_opt_u32(&mut self, v: Option<u32>) {
        match v {
            Some(x) => {
                self.write_u8(1);
                self.write_u32(x);
            }
            None => self.write_u8(0),
        }
    }

    pub fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self::new()
    }
}
