//! Per-cim self-rescheduling day chain.
//!
//! Each Individual-tier cim lives one day as a fixed chain of wake events with a
//! deterministic per-day jitter drawn from `hash(id, day)` — the only "randomness"
//! is a *value*, never an order or a branch (integration brief §9 rule 2). The
//! chain is:
//!
//! ```text
//! Wake(7h+jitter) -> DropOff(8h) -> ArriveWork(9h) -> LeaveWork(17h) -> Sleep(22h)
//!                                                               -> DayRoll(next day)
//! ```
//!
//! `seed_day` enqueues the whole chain for one day plus the `DayRoll` that
//! reseeds the next day. The `DayRoll` handler (in `CimSim::sense_commit`) calls
//! `seed_day` again, so the chain perpetuates without any per-tick settle — every
//! event is scheduled exactly once and fires off the wake queue. `DayRoll` is the
//! only event that crosses a day boundary, and it is gated on the cadence `.day`
//! bit at the `sense_commit` callsite (the brief §4.2 no-per-tick-loop rule).
//!
//! All offsets are ticks-from-midnight with `TICKS_PER_DAY = 144` (6 ticks/hour).
//! The day-chain `ArriveWork` here is **Individual-tier only**: it fires at the
//! jittered fixed chain tick and does NOT compute cohort travel time. The cohort
//! `ArriveWork@T` is enqueued externally via `CimSim::schedule` (pathfinding plan
//! Task 6); this module never schedules a cohort `ArriveWork`.

use crate::cim_sim::hash;
use crate::cim_sim::queue::{EventRef, WakeQueue};
use crate::sim_types::EventKind;
use crate::cim_sim::TICKS_PER_DAY;

/// Wake at 7h (42 ticks) plus the jittered minutes.
pub const OFF_WAKE: u64 = 42;
/// DropOff at 8h (48 ticks).
pub const OFF_DROP_OFF: u64 = 48;
/// ArriveWork at 9h (54 ticks).
pub const OFF_ARRIVE_WORK: u64 = 54;
/// LeaveWork at 17h (102 ticks).
pub const OFF_LEAVE_WORK: u64 = 102;
/// Sleep at 22h (132 ticks).
pub const OFF_SLEEP: u64 = 132;

/// The canonical sample day a freshly spawned cim begins living. Day 5 is the
/// first fully-settled work day used across the time/cadence layer (see
/// `time.rs`'s day-5 calendar tests), so traces and `--inspect` runs land on a
/// representative weekday rather than the world's tick-0 cold start.
pub const START_DAY: u64 = 5;

/// Per-day jitter span in ticks (`hash(id, day) % JITTER`), applied to the whole
/// chain so a cim's morning slides but the within-day spacing is preserved.
const JITTER: u64 = 18;

/// The deterministic morning jitter for `(id, day)` in ticks (`0..JITTER`).
pub fn day_jitter(id: u32, day: u64) -> u64 {
    hash::hash(id, day) % JITTER
}

/// Enqueue the full event chain for `id` on `day`, plus the `DayRoll` that
/// reseeds the next day. Every event uses an `EventKind as u8` byte from
/// `sim_types` — no private tags (brief §4.3).
///
/// The jitter shifts the entire chain uniformly, so the events stay in strict
/// `Wake < DropOff < ArriveWork < LeaveWork < Sleep` order regardless of `j`, and
/// `wake_tick` is strictly non-decreasing within the day.
pub fn seed_day(q: &mut WakeQueue, id: u32, day: u64) {
    let j = day_jitter(id, day);
    let base = day * TICKS_PER_DAY;
    let push = |q: &mut WakeQueue, off: u64, kind: EventKind| {
        q.push(EventRef { wake_tick: base + off + j, id, kind: kind as u8 });
    };
    push(q, OFF_WAKE, EventKind::Wake);
    push(q, OFF_DROP_OFF, EventKind::DropOff);
    push(q, OFF_ARRIVE_WORK, EventKind::ArriveWork);
    push(q, OFF_LEAVE_WORK, EventKind::LeaveWork);
    push(q, OFF_SLEEP, EventKind::Sleep);
    // DayRoll fires at the START of the next day (no jitter — it is the cadence
    // day-boundary reseed hook, not part of the lived chain).
    q.push(EventRef {
        wake_tick: (day + 1) * TICKS_PER_DAY,
        id,
        kind: EventKind::DayRoll as u8,
    });
}

/// Enqueue ONLY the morning prefix of the chain (`Wake`, then `DropOff`) for the
/// care-gate path. The remainder of the chain
/// (`ArriveWork -> LeaveWork -> Sleep -> DayRoll`) is pushed by the `DropOff`
/// COMMIT handler — and ONLY when the worker passes the care gate. A suppressed
/// `DropOff` therefore never enqueues `ArriveWork`: the absence is the *missing*
/// event, not a flag (brief §4.6 / the thesis). The jitter is identical to
/// [`seed_day`] so this path is bit-compatible with the full chain's timing.
pub fn seed_day_gated(q: &mut WakeQueue, id: u32, day: u64) {
    let j = day_jitter(id, day);
    let base = day * TICKS_PER_DAY;
    q.push(EventRef { wake_tick: base + OFF_WAKE + j, id, kind: EventKind::Wake as u8 });
    q.push(EventRef { wake_tick: base + OFF_DROP_OFF + j, id, kind: EventKind::DropOff as u8 });
}

/// Push the post-drop-off remainder of the chain
/// (`ArriveWork -> LeaveWork -> Sleep -> DayRoll`) for `id` on the civic `day`
/// that contains the drop-off. Called by the `DropOff` handler ONLY when the
/// worker is free to work; a suppressed drop-off skips this entirely, so
/// `ArriveWork` is never enqueued.
pub fn push_after_drop_off(q: &mut WakeQueue, id: u32, day: u64) {
    let j = day_jitter(id, day);
    let base = day * TICKS_PER_DAY;
    q.push(EventRef { wake_tick: base + OFF_ARRIVE_WORK + j, id, kind: EventKind::ArriveWork as u8 });
    q.push(EventRef { wake_tick: base + OFF_LEAVE_WORK + j, id, kind: EventKind::LeaveWork as u8 });
    q.push(EventRef { wake_tick: base + OFF_SLEEP + j, id, kind: EventKind::Sleep as u8 });
    q.push(EventRef { wake_tick: (day + 1) * TICKS_PER_DAY, id, kind: EventKind::DayRoll as u8 });
}

/// Reschedule a cim's chain for the day that contains `now`. Called by the
/// `DayRoll` handler (and at spawn) to keep the chain flowing without any
/// per-tick loop.
pub fn reschedule_after(q: &mut WakeQueue, id: u32, now: u64) {
    let day = now / TICKS_PER_DAY;
    seed_day(q, id, day);
}
