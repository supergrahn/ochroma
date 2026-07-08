//! Deterministic wake queue: a near-term bucket ring + a far-future spill heap.
//!
//! Events are popped in strict `(wake_tick, id, kind)` order within each tick —
//! the determinism contract's application order (brief §9 rule 4). The ring holds
//! a fixed near-term horizon of `RING` ticks indexed by `wake_tick % RING`; events
//! beyond that horizon live in a `BinaryHeap` (min-heap via `Reverse`) and migrate
//! into the ring as `now` advances. No hashed-container iteration anywhere
//! (rule 1) — only `Vec` and a `BinaryHeap`; NO RNG, NO wall-clock — `now` is a
//! `u64` tick.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A scheduled wake event: which cim (`id`), at which `wake_tick`, of which `kind`
/// (an `EventKind as u8` from `sim_types` — no private tag bytes, brief §4.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EventRef {
    pub wake_tick: u64,
    pub id: u32,
    pub kind: u8,
}

/// Near-term horizon in ticks. Events with `wake_tick < now + RING` land in the
/// ring; everything farther out spills to the heap and migrates in later.
const RING: usize = 4096;

/// Bucket-ring + spill-heap wake queue. Pops a `(wake_tick, id, kind)`-sorted
/// batch per tick deterministically.
pub struct WakeQueue {
    /// Indexed by `wake_tick % RING`; each bucket holds the events due at ticks
    /// congruent to that slot within the current horizon.
    ring: Vec<Vec<EventRef>>,
    /// Far-future events (`wake_tick >= horizon_end`), ordered by `(wake_tick,id,kind)`.
    spill: BinaryHeap<Reverse<(u64, u32, u8)>>,
    /// Exclusive upper bound of the ticks currently representable in the ring.
    /// Everything with `wake_tick < horizon_end` is in the ring; `>=` is in spill.
    horizon_end: u64,
}

impl WakeQueue {
    /// A fresh, empty queue with an empty ring horizon `[0, RING)`.
    pub fn new() -> Self {
        Self {
            ring: (0..RING).map(|_| Vec::new()).collect(),
            spill: BinaryHeap::new(),
            horizon_end: RING as u64,
        }
    }

    /// Schedule an event. Lands in the ring if within the current horizon, else
    /// the spill heap. Full body — no stub.
    pub fn push(&mut self, e: EventRef) {
        if e.wake_tick < self.horizon_end {
            self.ring[(e.wake_tick % RING as u64) as usize].push(e);
        } else {
            self.spill.push(Reverse((e.wake_tick, e.id, e.kind)));
        }
    }

    /// Drain all events due exactly at `now`, returned sorted by `(wake_tick,id,kind)`.
    ///
    /// 1. Advance the horizon to cover `now`, migrating any spill entries that have
    ///    entered the ring window.
    /// 2. Take the bucket for `now`, keeping only entries whose `wake_tick == now`
    ///    (defensive — a bucket slot is shared across `now + k*RING` ticks; stragglers
    ///    are re-bucketed).
    /// 3. Sort by `(wake_tick, id, kind)` and return — no hashed container anywhere.
    pub fn pop_due(&mut self, now: u64) -> Vec<EventRef> {
        // 1. Extend the horizon so `now` is representable, pulling in any spill
        //    events that now fall inside the ring window.
        if now >= self.horizon_end {
            let new_end = now + 1;
            while let Some(&Reverse((wt, id, kind))) = self.spill.peek() {
                if wt < new_end {
                    self.spill.pop();
                    self.ring[(wt % RING as u64) as usize].push(EventRef {
                        wake_tick: wt,
                        id,
                        kind,
                    });
                } else {
                    break;
                }
            }
            self.horizon_end = new_end;
        }

        let slot = (now % RING as u64) as usize;
        let bucket = std::mem::take(&mut self.ring[slot]);
        let mut due = Vec::new();
        for e in bucket {
            if e.wake_tick == now {
                due.push(e);
            } else {
                // Straggler sharing this slot at a different tick; re-bucket it.
                self.ring[slot].push(e);
            }
        }
        due.sort_by_key(|e| (e.wake_tick, e.id, e.kind));
        due
    }
}

impl Default for WakeQueue {
    fn default() -> Self {
        Self::new()
    }
}
