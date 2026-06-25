//! Dense, deterministic, collision-free render identity allocators.
//!
//! Design: `urban_horizon/docs/superpowers/specs/2026-06-18-solid-id-and-material-system.md`
//! §4.3 + §5. Replaces the two ad-hoc hashes that used to mint render ids:
//!
//! * `object_id_from_parts` (FNV-1a, 32-bit) and `object_base` (djb2, 32-bit) —
//!   both lived in the same 24-bit space (the top byte was STOLEN for the Forge
//!   material channel), with a birthday-bound first collision near ~4830 distinct
//!   groups (reachable on a full city). They are replaced by [`ObjectIdAllocator`]
//!   — a monotonic 1-based counter minted in the caller's id-sorted build order,
//!   so the same scene yields the same ids every run AND no two groups ever
//!   collide.
//! * the `proto_id = idx`/`0`/`0xA6E_07_0000 ^ idx` magic-salt prototype ids —
//!   replaced by [`ProtoIdAllocator`] (monotonic 1-based `u64`).
//!
//! The Forge material channel no longer rides `object_id`'s top byte; it lives in
//! its own `HybridMesh::material_channel` field (engine side), freeing all 32 bits
//! of [`ObjectId`] for pure identity.
//!
//! Determinism: monotonic ids over an id-sorted build are reproducible by
//! construction — no hash, no birthday collision, no HashMap-iteration tie-break.
//! Both allocators are single-threaded, owned by one scene build.

/// Dense, deterministic, collision-free object identity. `0` = NONE; real ids
/// start at 1. Full 32 bits (the channel no longer steals the top byte — see
/// `HybridMesh::material_channel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(u32);

impl ObjectId {
    /// The reserved "no object" id.
    pub const NONE: ObjectId = ObjectId(0);

    /// The raw 32-bit id (written into `HybridMesh::object_id`).
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Monotonic [`ObjectId`] allocator. Ids are minted 1-based in the caller's
/// id-sorted build order, so the same scene yields the same ids every run.
pub struct ObjectIdAllocator {
    next: u32,
}

impl ObjectIdAllocator {
    pub fn new() -> Self {
        Self { next: 1 }
    }

    /// Mint the next dense object id (never 0).
    pub fn next(&mut self) -> ObjectId {
        let id = ObjectId(self.next);
        self.next = self.next.checked_add(1).expect("ObjectId space exhausted");
        id
    }
}

impl Default for ObjectIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable, unique prototype identity. Replaces `proto_id = idx`/`0`/magic-salt.
/// `0` = NONE; real ids start at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProtoId(u64);

impl ProtoId {
    pub const NONE: ProtoId = ProtoId(0);

    /// The raw 64-bit id (written into `BlasDesc::proto_id`).
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Monotonic [`ProtoId`] allocator (1-based `u64`).
pub struct ProtoIdAllocator {
    next: u64,
}

impl ProtoIdAllocator {
    pub fn new() -> Self {
        Self { next: 1 }
    }

    /// Mint the next stable prototype id (never 0).
    pub fn next(&mut self) -> ProtoId {
        let id = ProtoId(self.next);
        self.next = self.next.checked_add(1).expect("ProtoId space exhausted");
        id
    }
}

impl Default for ProtoIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Done-When capability: 50k object ids over distinct labels are unique AND
    /// identical across two runs (collision-free + deterministic).
    #[test]
    fn object_ids_unique_over_50k() {
        let mint = || {
            let mut a = ObjectIdAllocator::new();
            (0..50_000).map(|_| a.next().raw()).collect::<Vec<u32>>()
        };
        let run1 = mint();
        let run2 = mint();
        assert_eq!(run1, run2, "monotonic ids must be identical across runs");
        let set: HashSet<u32> = run1.iter().copied().collect();
        assert_eq!(set.len(), 50_000, "no two object ids may collide");
        // 1-based, dense: first is 1, last is 50_000.
        assert_eq!(run1[0], 1);
        assert_eq!(run1[49_999], 50_000);
        // NONE is reserved and never minted.
        assert!(!set.contains(&ObjectId::NONE.raw()));
    }

    #[test]
    fn proto_ids_are_monotonic_and_nonzero() {
        let mut a = ProtoIdAllocator::new();
        let ids: Vec<u64> = (0..1000).map(|_| a.next().raw()).collect();
        assert_eq!(ids[0], 1);
        assert_eq!(ids[999], 1000);
        let set: HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(set.len(), 1000);
        assert!(!set.contains(&ProtoId::NONE.raw()));
    }
}
