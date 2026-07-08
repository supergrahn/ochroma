//! Structure-of-arrays cim store with closed-form (on-wake) need catch-up.
//!
//! The engine's `CitizenManager::tick(dt_years)` advances *every* citizen's
//! needs every tick (a uniform O(N) sweep). The game path retires that: a cim's
//! need is stored as `(value_at_event, last_event_tick, rate_per_day)` and the
//! current value is reconstructed with ONE fused-multiply-add at read time —
//! `value_at_event + rate_per_day * elapsed_days`. A cim that sleeps for 30 days
//! costs nothing until something wakes it, and the catch-up is exact (no
//! per-tick accumulation drift).
//!
//! Determinism (integration brief §9): columns are id-sorted (`id[row]` is the
//! sorted cim id, `row` is the index — rule 1); the only compare is a fixed-ε
//! f64/f32 read; `now` is a `u64` tick, never wall-clock (rule 7).

/// Need column indices. Mirrors `vox_sim::citizen::Needs` field order so the
/// game-side closed-form store maps 1:1 onto the engine substrate without the
/// engine learning game concepts.
pub const NEED_HOUSING: usize = 0;
pub const NEED_FOOD: usize = 1;
pub const NEED_HEALTH: usize = 2;
pub const NEED_SAFETY: usize = 3;
pub const NEED_EDUCATION: usize = 4;
pub const NEED_EMPLOYMENT: usize = 5;
pub const NEED_LEISURE: usize = 6;
/// Total number of need channels (one `Vec<NeedCol>` column each).
pub const NEED_COUNT: usize = 7;

/// One need channel for one cim: the value at the last event that touched it,
/// the tick of that event, and the linear drift per day. `need_now` reconstructs
/// the current value with one FMA — no intervening integration is ever stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeedCol {
    pub value_at_event: f32,
    pub last_event_tick: u64,
    pub rate_per_day: f32,
}

impl Default for NeedCol {
    fn default() -> Self {
        Self {
            value_at_event: 0.0,
            last_event_tick: 0,
            rate_per_day: 0.0,
        }
    }
}

/// Double-buffered SoA cim columns.
///
/// `id[row]` is the sorted cim id; `row` is the dense index every other column
/// shares. The `*_prev` mirrors are the frozen snapshot the parallel SENSE phase
/// reads while the COMMIT phase writes the front columns — readers and writers
/// never touch the same buffer, so SENSE is order-independent (brief §9 rule 4).
pub struct CimStore {
    pub id: Vec<u32>,
    pub home_id: Vec<u32>,
    pub work_id: Vec<u32>,
    pub need: [Vec<NeedCol>; NEED_COUNT],
    pub wake_tick: Vec<u64>,

    // Frozen previous-frame snapshot (SENSE reads these; COMMIT writes the front).
    need_prev: [Vec<NeedCol>; NEED_COUNT],
    wake_tick_prev: Vec<u64>,
}

impl CimStore {
    /// An empty store with room reserved for `cap` cims (no rows yet).
    pub fn with_capacity(cap: usize) -> Self {
        let need: [Vec<NeedCol>; NEED_COUNT] = Default::default();
        let need_prev: [Vec<NeedCol>; NEED_COUNT] = Default::default();
        let mut s = Self {
            id: Vec::with_capacity(cap),
            home_id: Vec::with_capacity(cap),
            work_id: Vec::with_capacity(cap),
            need,
            wake_tick: Vec::with_capacity(cap),
            need_prev,
            wake_tick_prev: Vec::with_capacity(cap),
        };
        for n in 0..NEED_COUNT {
            s.need[n].reserve(cap);
            s.need_prev[n].reserve(cap);
        }
        s
    }

    /// Number of cim rows currently stored.
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// True when no cims are stored.
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Append a cim. Callers push in ascending `id` order so `id[row]` stays
    /// sorted (the determinism contract); returns the new row index. All need
    /// channels start at `NeedCol::default()` (flat, zero-drift) until
    /// `set_need` is called.
    pub fn push_cim(&mut self, id: u32, home: u32, work: u32) -> usize {
        let row = self.id.len();
        self.id.push(id);
        self.home_id.push(home);
        self.work_id.push(work);
        self.wake_tick.push(0);
        self.wake_tick_prev.push(0);
        for n in 0..NEED_COUNT {
            self.need[n].push(NeedCol::default());
            self.need_prev[n].push(NeedCol::default());
        }
        row
    }

    /// Stamp a need channel: its value, the tick it was set, and the per-day
    /// linear drift used to reconstruct it later. This is the ONLY way a need
    /// changes — every event that touches a need re-stamps `(value, now, rate)`.
    pub fn set_need(
        &mut self,
        row: usize,
        need: usize,
        value: f32,
        last_tick: u64,
        rate_per_day: f32,
    ) {
        self.need[need][row] = NeedCol {
            value_at_event: value,
            last_event_tick: last_tick,
            rate_per_day,
        };
    }

    /// Closed-form current value: `value_at_event + rate_per_day * elapsed_days`,
    /// clamped to `[0,1]`. One FMA, O(1), exact regardless of how many days the
    /// cim slept since the last event. THE only needs accessor on the game path
    /// (the engine's per-tick uniform advance is not called for cim needs).
    pub fn need_now(&self, row: usize, need: usize, now: u64) -> f32 {
        let c = self.need[need][row];
        // `now >= last_event_tick` always on the forward sim timeline; guard the
        // unsigned subtraction defensively so a stale read can't underflow.
        let elapsed_ticks = now.saturating_sub(c.last_event_tick);
        let elapsed_days = elapsed_ticks as f32 / crate::cim_sim::TICKS_PER_DAY as f32;
        (c.value_at_event + c.rate_per_day * elapsed_days).clamp(0.0, 1.0)
    }

    /// Per-cim next-wake tick (the schedule chain stamps this; the queue is the
    /// authority, this column is the cim-local mirror).
    pub fn set_wake_tick(&mut self, row: usize, tick: u64) {
        self.wake_tick[row] = tick;
    }

    /// Stamp the wake-tick mirror by cim `id` (the schedule chain knows ids, not
    /// rows). `id` is sorted, so this is an O(log n) `binary_search`; an id with
    /// no store row (e.g. a cohort-only event injected via `schedule`) is a
    /// silent no-op so externally-injected events drain cleanly.
    pub fn set_wake_tick_for_id(&mut self, id: u32, tick: u64) {
        if let Ok(row) = self.id.binary_search(&id) {
            self.wake_tick[row] = tick;
        }
    }

    /// The workplace building id of cim `id`, or `None` if it has no store row.
    /// `id` is sorted, so this is an O(log n) `binary_search` — used by the
    /// presence-delta handlers (`ArriveWork`/`LeaveWork`) to target the right
    /// building.
    pub fn work_id_for(&self, id: u32) -> Option<u32> {
        self.id.binary_search(&id).ok().map(|row| self.work_id[row])
    }

    /// Read the frozen (previous-frame) need value — what the parallel SENSE
    /// phase scores against while COMMIT mutates the front columns.
    pub fn need_now_prev(&self, row: usize, need: usize, now: u64) -> f32 {
        let c = self.need_prev[need][row];
        let elapsed_ticks = now.saturating_sub(c.last_event_tick);
        let elapsed_days = elapsed_ticks as f32 / crate::cim_sim::TICKS_PER_DAY as f32;
        (c.value_at_event + c.rate_per_day * elapsed_days).clamp(0.0, 1.0)
    }

    /// Promote the front columns into the frozen snapshot. Called once per
    /// COMMIT so the next frame's SENSE reads a consistent prior state.
    pub fn swap_buffers(&mut self) {
        for n in 0..NEED_COUNT {
            self.need_prev[n].clone_from(&self.need[n]);
        }
        self.wake_tick_prev.clone_from(&self.wake_tick);
    }
}
