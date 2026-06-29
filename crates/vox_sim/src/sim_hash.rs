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

use crate::city_sim::CitySim;

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

/// Compute the bit-exact, id-ordered hash of the full reproducible sim state.
///
/// The fold order is fixed: citizens in their stored order (which is itself
/// id-monotonic), then buildings in stored (id-monotonic) order, then agent
/// positions sorted by citizen id. Every float is folded by `to_bits`.
pub fn sim_state_hash(sim: &CitySim) -> u64 {
    let mut h = FnvHasher::new();

    // --- Citizens (id-ordered: the Vec is appended in id order and is never
    //     reordered, only id-stable retain removals happen). ---
    let citizens = sim.citizens.all();
    h.write_u64(citizens.len() as u64);
    for c in citizens {
        h.write_u32(c.id);
        h.write_u32(c.agent_id);
        h.write_f32(c.age);
        h.write_u8(c.lifecycle as u8);
        h.write_u8(c.education as u8);
        h.write_opt_u32(c.employment);
        h.write_opt_u32(c.residence);
        h.write_f32(c.satisfaction);
        h.write_f32(c.needs.housing);
        h.write_f32(c.needs.food);
        h.write_f32(c.needs.health);
        h.write_f32(c.needs.safety);
        h.write_f32(c.needs.education);
        h.write_f32(c.needs.employment);
        h.write_f32(c.needs.leisure);
        h.write_u8(c.daily_state as u8);
        h.write_opt_u32(c.workplace);
    }

    // --- Buildings (stored in id-monotonic order). ---
    let buildings = &sim.buildings.buildings;
    h.write_u64(buildings.len() as u64);
    for b in buildings {
        h.write_u32(b.id);
        h.write_u8(b.building_type as u8);
        h.write_f32(b.position[0]);
        h.write_f32(b.position[1]);
        h.write_u32(b.capacity);
        h.write_u32(b.occupants);
        h.write_u8(b.operational as u8);
    }

    // --- Agent positions (citizen-id ordered; never the Uuid map order). ---
    let agents = sim.agent_positions_id_ordered();
    h.write_u64(agents.len() as u64);
    for (cid, pos) in agents {
        h.write_u32(cid);
        let (x, y, z) = pos.to_absolute();
        h.write_f64(x);
        h.write_f64(y);
        h.write_f64(z);
    }

    h.finish()
}
