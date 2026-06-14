//! The single, generic time model: a pure-integer calendar over one `u64`
//! master tick.
//!
//! This is the keystone that replaces the three contradictory clocks the sim
//! used to carry (a per-tick *seconds* constant, a per-tick *years* constant,
//! and a "one tick = one month" UI assumption). There is now exactly ONE atom
//! — the `u64` tick — and every human-facing unit (day, hour, day-of-week,
//! month, year) is *derived* from it by integer arithmetic.
//!
//! # Why it lives in the engine crate
//!
//! A calendar — "given a tick count and how many ticks make a day, what day /
//! hour / weekday / month is it?" — is game-agnostic arithmetic. It carries NO
//! game concept (no buildings, no care, no zoning). The *tuning* values
//! (`ticks_per_day`, `days_per_month`, `days_per_year`) are NOT baked in here;
//! they are injected by the caller from game-side config. The engine therefore
//! holds the mechanism, the game holds the policy.
//!
//! # Determinism contract
//!
//! Every value is pure integer math on the tick counter. No wall-clock, no
//! floating point, no RNG. A given `(tick, ticks_per_day, days_per_month,
//! days_per_year)` always yields the same `CalendarDate` — replay-exact, on any
//! machine. There is deliberately no leap-year math: a fixed civic calendar of
//! `days_per_month` days per month and `days_per_year` days per year is chosen
//! for determinism over realism.

/// A calendar parameterised by its cadence. Construct it once from config and
/// query any tick with [`Calendar::at`].
///
/// All fields are the injected tuning values — the engine invents none of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Calendar {
    /// Master ticks in one in-game day. The keystone cadence (game-tuned).
    ticks_per_day: u64,
    /// Days in one civic month (fixed; no variable-length months).
    days_per_month: u64,
    /// Days in one civic year (fixed; no leap math).
    days_per_year: u64,
}

/// A tick decomposed into human-facing calendar fields. Every field is computed
/// from the master tick by integer math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalendarDate {
    /// The master tick this date was derived from (the atom).
    pub tick: u64,
    /// Whole days since tick 0. `tick / ticks_per_day`.
    pub day: u64,
    /// Hour of the day, `0..24`. The tick-of-day scaled into 24 hours.
    pub hour: u64,
    /// Minute of the hour, `0..60`. The remainder of the hour scaling.
    pub minute: u64,
    /// Tick offset within the current day, `0..ticks_per_day`.
    pub tick_of_day: u64,
    /// Day of week, `0..7`. `day % 7` (0 = first day of the civic week).
    pub dow: u64,
    /// Month index since tick 0 (absolute, monotonic). `day / days_per_month`.
    pub month: u64,
    /// Month-of-year index, `0..months_per_year`.
    pub month_of_year: u64,
    /// Year since tick 0. `day / days_per_year`.
    pub year: u64,
}

impl Calendar {
    /// Build a calendar from its (game-supplied) cadence.
    ///
    /// # Panics
    /// In debug builds, if any cadence is zero (a zero would make the derived
    /// units meaningless and divide by zero). Callers pass config defaults that
    /// are always non-zero.
    pub const fn new(ticks_per_day: u64, days_per_month: u64, days_per_year: u64) -> Self {
        debug_assert!(ticks_per_day > 0, "ticks_per_day must be > 0");
        debug_assert!(days_per_month > 0, "days_per_month must be > 0");
        debug_assert!(days_per_year > 0, "days_per_year must be > 0");
        Self {
            ticks_per_day,
            days_per_month,
            days_per_year,
        }
    }

    /// Ticks in one in-game day (the keystone cadence).
    pub const fn ticks_per_day(&self) -> u64 {
        self.ticks_per_day
    }

    /// Days in one civic month.
    pub const fn days_per_month(&self) -> u64 {
        self.days_per_month
    }

    /// Days in one civic year.
    pub const fn days_per_year(&self) -> u64 {
        self.days_per_year
    }

    /// Months in one civic year. `days_per_year / days_per_month` (integer).
    pub const fn months_per_year(&self) -> u64 {
        self.days_per_year / self.days_per_month
    }

    /// How many in-game years elapse over `ticks` master ticks, as an exact
    /// rational evaluated in `f32`. This is the *one* sanctioned replacement for
    /// the deleted per-tick years constant: aging derives years from the tick
    /// count and the civic-year length, not from a magic per-tick float.
    ///
    /// `years = ticks / (ticks_per_day * days_per_year)`.
    pub fn years_in(&self, ticks: u64) -> f32 {
        let ticks_per_year = self.ticks_per_day * self.days_per_year;
        ticks as f32 / ticks_per_year as f32
    }

    /// Decompose a master `tick` into its full calendar date. Pure integer math.
    pub fn at(&self, tick: u64) -> CalendarDate {
        let day = tick / self.ticks_per_day;
        let tick_of_day = tick % self.ticks_per_day;
        // Scale the tick-of-day across 24 hours: hour = tick_of_day * 24 / TPD.
        // Minute is the leftover scaled across 60. All integer, so no drift.
        let minutes_of_day = tick_of_day * 24 * 60 / self.ticks_per_day;
        let hour = minutes_of_day / 60;
        let minute = minutes_of_day % 60;
        let dow = day % 7;
        let month = day / self.days_per_month;
        let month_of_year = month % self.months_per_year();
        let year = day / self.days_per_year;
        CalendarDate {
            tick,
            day,
            hour,
            minute,
            tick_of_day,
            dow,
            month,
            month_of_year,
            year,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The game's keystone cadence, re-stated locally so the engine test needs no
    // game crate. (The game owns the canonical config value.)
    const TPD: u64 = 144;
    const DPM: u64 = 30;
    const DPY: u64 = 360;

    #[test]
    fn derives_day_hour_dow_month_from_tick() {
        let cal = Calendar::new(TPD, DPM, DPY);
        // tick 728: day = 728/144 = 5; tick_of_day = 728 - 720 = 8;
        // hour = 8*24/144 = 1 (08:00 would be tick 48). So tick 728 is day 5,
        // very early morning. Assert the exact computed integers.
        let d = cal.at(728);
        assert_eq!(d.day, 5, "728/144 = 5");
        assert_eq!(d.tick_of_day, 8, "728 % 144 = 8");
        assert_eq!(d.hour, 1, "8*24/144 = 1");
        assert_eq!(d.dow, 5, "5 % 7 = 5");
        assert_eq!(d.month, 0, "5 / 30 = 0");

        // 08:00 is tick 48 within a day: hour must read 8 exactly.
        let morning = cal.at(48);
        assert_eq!(morning.hour, 8, "tick 48 of a 144-tick day is 08:00");
        assert_eq!(morning.minute, 0);

        // A tick deep into a later day/month/year, all derived.
        let later = cal.at(3 * TPD + 8 * (TPD / 24));
        assert_eq!(later.day, 3);
        assert_eq!(later.hour, 8, "08:00 on day 3");

        println!(
            "tick {} -> day {} hour {} dow {} month {}",
            d.tick, d.day, d.hour, d.dow, d.month
        );
    }

    #[test]
    fn years_in_is_exact_rational() {
        let cal = Calendar::new(TPD, DPM, DPY);
        // One full year = TPD * DPY ticks.
        let one_year_ticks = TPD * DPY;
        assert!((cal.years_in(one_year_ticks) - 1.0).abs() < 1e-6);
        assert!((cal.years_in(one_year_ticks / 2) - 0.5).abs() < 1e-6);
    }
}
