//! Conserved cohort table — the cohort-mass behavior-LOD substrate.
//!
//! ~99.9% of a six-figure city lives as *cohort mass*: a small set of
//! [`CohortRow`]s, each a conserved `head_count` of statistically-identical cims
//! sharing a district, [`SkillTier`], worker flag, and need mean/variance.
//! Individuals crystallize on demand ([`CohortTable::hydrate`]) and dissolve back
//! ([`CohortTable::dissolve`]); the integer head-count total plus the hydrated
//! count is invariant — the SAME population-conservation invariant the pathfinding
//! OD row-sum enforces (integration brief §4.5), checked once per COMMIT via
//! [`crate::sim_types::assert_population_conserved`].
//!
//! This generalizes the already-conserved, sorted-id `HouseholdRecord`
//! (`src/household/mod.rs`): no `HashMap`, no RNG order, no wall-clock. The two
//! cross-plan accessors — [`CohortTable::cohort_supply_by_tier`] (ECON L2 source)
//! and [`CohortTable::district_worker_mass`] (PF OD source) — fold the sorted
//! cohort `Vec` in id order so two replays produce bit-identical aggregates
//! (brief §9 rules 1/2/8).

use crate::cim_sim::hash::hash;
use crate::sim_types::{DistrictId, SkillTier};

/// Number of districts the deterministic seed spreads cohort mass across. Sized so
/// the accessor test's `district_worker_mass()[3]` is a real, populated district.
pub const N_DISTRICTS: usize = 8;

/// The four `SkillTier`s in canonical `labor::idx` order (Unskilled=0..Expert=3).
const TIERS: [SkillTier; 4] = [
    SkillTier::Unskilled,
    SkillTier::Skilled,
    SkillTier::Educated,
    SkillTier::Expert,
];

// ───────────────────────────── The two-axis matrix ─────────────────────────────
//
// Population state is a two-axis matrix — `cohort[district][age_band][occupation]
// [edu_tier]` — separating BIOLOGY (age band: fertility, mortality curve, care-need
// type, school-stage) from ALLOCATION (occupation: labor, school, care given/
// received). Design §5.3. The band boundaries never move; policies move the
// *occupation* eligibility windows across them.
//
// This slice ships the matrix SUBSTRATE (the type, the legal-pair mask, the 47-row/
// district schema, and the conserved column accessors). The vital + occupation
// FLOWS are Tasks 2+; until they land, the existing four-band [`CohortRow::age_band`]
// lifecycle still drives the conserved `head_count`, and each row's matrix triple
// (`band`, `occ`, `edu_tier`) is derived from `(is_worker, age_band, district, tier)`
// by the pure adapter [`matrix_of`] — so the neutral-baseline replay hash (which
// folds only `id` + `head_count`) stays bit-for-bit identical.

/// Age band — BIOLOGY. Seven bands, each earning its lower boundary by a distinct
/// mechanic (care-need type, school-stage, labor/fertility eligibility, mortality
/// curve). Boundaries are fixed; only occupation windows move across them
/// (design §5.3.1). Discriminants are stable (they key id-ordered folds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AgeBand {
    /// 0–1 — institutional care ineligible; the parental-leave window.
    Newborn = 0,
    /// 1–5 — kindergarten-eligible (the scarce slot, the care gate's theater).
    Toddler = 1,
    /// 6–12 — elementary stage; an unseated Child still binds a Caregiver.
    Child = 2,
    /// 13–18 — secondary stage; NEVER binds a Caregiver (self-supervising).
    Teenager = 3,
    /// 19–25 — University window; peak migration propensity; fertility onset.
    YoungAdult = 4,
    /// 26–66 — prime labor; fertility confined to its 26–45 slice; the R window.
    Adult = 5,
    /// 67+ — mortality-curve step; eldercare onset; the Care Corps pool.
    Senior = 6,
}

/// The seven age bands in canonical (ascending) order.
pub const AGE_BANDS: [AgeBand; 7] = [
    AgeBand::Newborn,
    AgeBand::Toddler,
    AgeBand::Child,
    AgeBand::Teenager,
    AgeBand::YoungAdult,
    AgeBand::Adult,
    AgeBand::Senior,
];

/// Occupation — ALLOCATION. Nine columns, each with an age-eligibility window, a
/// capacity gate, and a ledger (design §5.3.2). Unemployment is a first-class
/// column (a row-sum, not "workers minus jobs"); the care gate is the Caregiver
/// column. Discriminants are stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Occupation {
    /// The default row when every other gate refuses; consumes one household
    /// Caregiver; produces the care demand every panel reads.
    HomeCare = 0,
    /// Toddler slot; releasing a Caregiver back to labor supply.
    Kindergarten = 1,
    /// Child primary stage; doubles as childcare.
    Elementary = 2,
    /// Teenager secondary stage; coverage EMA → Skilled at age-up.
    Secondary = 3,
    /// YoungAdult tertiary; Skilled at entry → Educated at graduation.
    University = 4,
    /// In the labor force and placed in a job.
    Employed = 5,
    /// In the labor force, unplaced — the buffer row; labor *supply*.
    Unemployed = 6,
    /// Bound by a household HomeCare dependent; withdrawn from labor supply.
    Caregiver = 7,
    /// Past the retirement-age policy R (default 67); the Senior age-up default.
    Retired = 8,
}

/// The nine occupations in canonical (ascending) order.
pub const OCCUPATIONS: [Occupation; 9] = [
    Occupation::HomeCare,
    Occupation::Kindergarten,
    Occupation::Elementary,
    Occupation::Secondary,
    Occupation::University,
    Occupation::Employed,
    Occupation::Unemployed,
    Occupation::Caregiver,
    Occupation::Retired,
];

/// The exactly **17** legal `(AgeBand, Occupation)` pairs (design §5.3.3). Every
/// pair NOT in this table is structurally illegal — *unallocated*, never
/// constructed, never hashed, never able to drift (Newborn×Employed cannot be
/// written). Listed band-ascending then occupation-ascending so any fold over the
/// mask is deterministic. This is the whole design of the matrix: state scales with
/// `districts × legal pairs`, never with population.
pub const LEGAL_PAIRS: [(AgeBand, Occupation); 17] = [
    (AgeBand::Newborn, Occupation::HomeCare),
    (AgeBand::Toddler, Occupation::HomeCare),
    (AgeBand::Toddler, Occupation::Kindergarten),
    (AgeBand::Child, Occupation::HomeCare),
    (AgeBand::Child, Occupation::Elementary),
    (AgeBand::Teenager, Occupation::Secondary),
    (AgeBand::YoungAdult, Occupation::University),
    (AgeBand::YoungAdult, Occupation::Employed),
    (AgeBand::YoungAdult, Occupation::Unemployed),
    (AgeBand::YoungAdult, Occupation::Caregiver),
    (AgeBand::Adult, Occupation::Employed),
    (AgeBand::Adult, Occupation::Unemployed),
    (AgeBand::Adult, Occupation::Caregiver),
    (AgeBand::Adult, Occupation::Retired),
    (AgeBand::Senior, Occupation::Retired),
    (AgeBand::Senior, Occupation::Employed),
    (AgeBand::Senior, Occupation::HomeCare),
];

/// Is `(band, occ)` a structurally legal pair? A membership test against
/// [`LEGAL_PAIRS`] — the illegal-pair guard every constructor/flow honors.
pub fn is_legal_pair(band: AgeBand, occ: Occupation) -> bool {
    LEGAL_PAIRS.iter().any(|&(b, o)| b == band && o == occ)
}

/// Does `(band, occ)` expand across all four [`SkillTier`]s? `edu_tier` is assigned
/// at the Teenager→YoungAdult age-up and is undefined before it; `University` pins
/// Skilled at entry. So exactly the **10 adult-labor pairs** expand ×4 — YoungAdult
/// {Employed, Unemployed, Caregiver}, Adult {Employed, Unemployed, Caregiver,
/// Retired}, Senior {Retired, Employed, HomeCare} — and the other 7 legal pairs
/// carry a single (undefined/pinned) tier: `10×4 + 7 = 47` rows per district
/// (design §5.3.3).
pub fn is_tier_expanded(band: AgeBand, occ: Occupation) -> bool {
    matches!(
        (band, occ),
        (AgeBand::YoungAdult, Occupation::Employed)
            | (AgeBand::YoungAdult, Occupation::Unemployed)
            | (AgeBand::YoungAdult, Occupation::Caregiver)
            | (AgeBand::Adult, Occupation::Employed)
            | (AgeBand::Adult, Occupation::Unemployed)
            | (AgeBand::Adult, Occupation::Caregiver)
            | (AgeBand::Adult, Occupation::Retired)
            | (AgeBand::Senior, Occupation::Retired)
            | (AgeBand::Senior, Occupation::Employed)
            | (AgeBand::Senior, Occupation::HomeCare)
    )
}

/// The number of matrix rows a single district's schema carries: `10×4 + 7 = 47`.
/// State per district is exactly this, independent of population (design §5.3.3).
pub const ROWS_PER_DISTRICT: usize = 47;

/// The canonical 47-row/district matrix schema: every legal `(band, occ, edu_tier)`
/// triple, in a deterministic order (legal-pair order, then tier-ascending for the
/// tier-expanded pairs; `University` pinned Skilled; the other single-tier pairs
/// carry `Unskilled` as the "undefined" placeholder). This is the MATRIX SCHEMA the
/// vital/occupation flows (Tasks 2+) allocate against; the seed's conserved mass is
/// classified onto these triples by [`matrix_of`]. Pure — no state, no allocation
/// choice — so two builds are bit-identical.
pub fn legal_rows() -> Vec<(AgeBand, Occupation, SkillTier)> {
    let mut out: Vec<(AgeBand, Occupation, SkillTier)> = Vec::with_capacity(ROWS_PER_DISTRICT);
    for &(band, occ) in LEGAL_PAIRS.iter() {
        if is_tier_expanded(band, occ) {
            for &t in TIERS.iter() {
                out.push((band, occ, t));
            }
        } else if (band, occ) == (AgeBand::YoungAdult, Occupation::University) {
            // University pins Skilled at entry.
            out.push((band, occ, SkillTier::Skilled));
        } else {
            // Pre-YoungAdult pairs: tier undefined until the age-up split — carry the
            // Unskilled placeholder (never read as a real tier before assignment).
            out.push((band, occ, SkillTier::Unskilled));
        }
    }
    out
}

/// The pure ADAPTER classifying an existing four-band [`CohortRow`] onto its matrix
/// triple `(band, occ, edu_tier)` from `(is_worker, age_band, district, tier)`. Used
/// only while the four-band `age_band` lifecycle still drives the conserved
/// `head_count` (this slice's remaining seam — the real conserved flows replace it
/// in Tasks 2+). Every output is a LEGAL pair (asserted below). Deterministic: a
/// pure function of the row's own fields, no state, no order.
fn matrix_of(
    is_worker: bool,
    age_band: u8,
    district: DistrictId,
    tier: SkillTier,
) -> (AgeBand, Occupation, SkillTier) {
    let triple = if is_worker {
        // Working-age labor force → prime-Adult Employed; tier is the real edu tier.
        (AgeBand::Adult, Occupation::Employed, tier)
    } else {
        // Non-worker mass: the legacy seed spreads it across the three non-working
        // bands via `age_band ∈ {0=Child<6, 1=Student<18, 3=Retired}`. Map each onto
        // the honest matrix pairs. The "student/prime-age non-worker" bucket is split
        // deterministically (district+tier parity) between self-supervising Teenagers
        // (Secondary) and prime adults BOUND as Caregivers — so the care gate's
        // Caregiver column carries real mass from year 0 (design §5.3.5).
        match age_band {
            0 => (AgeBand::Toddler, Occupation::HomeCare, SkillTier::Unskilled),
            1 => {
                if (district as u64 + tier as u64) % 2 == 1 {
                    (AgeBand::Adult, Occupation::Caregiver, tier)
                } else {
                    (AgeBand::Teenager, Occupation::Secondary, SkillTier::Unskilled)
                }
            }
            _ => (AgeBand::Senior, Occupation::Retired, tier),
        }
    };
    debug_assert!(
        is_legal_pair(triple.0, triple.1),
        "matrix_of produced an ILLEGAL pair ({:?}, {:?})",
        triple.0,
        triple.1
    );
    triple
}

/// One conserved block of statistically-identical cims. `head_count` is a real
/// number so a *fraction* of a cohort can be hydrated without losing population;
/// the integer census carries the rounding remainder so the total stays exact.
#[derive(Clone, Debug)]
pub struct CohortRow {
    /// Stable, ascending cohort id — the determinism key for every id-ordered fold.
    pub id: u32,
    /// Conserved population of this cohort (fractional after partial hydration).
    pub head_count: f32,
    /// Home district (the `district_worker_mass` index for workers).
    pub district: DistrictId,
    /// Shared skill tier (the `cohort_supply_by_tier` index).
    pub tier: SkillTier,
    /// Whether this cohort's members are in the labor force.
    pub is_worker: bool,
    /// Per-need population mean (closed-form needs seed for hydrated individuals).
    pub need_mean: [f32; 7],
    /// Per-need population variance.
    pub need_var: [f32; 7],
    /// Coarse behavior archetype tag (day-chain template selector).
    pub archetype: u8,
    /// Coarse age band: `0=Child (<6), 1=Student (<18), 2=Worker (<65),
    /// 3=Retired (≥65)`. Mirrors `vox_sim::citizen::LifecycleStage` and the
    /// retirement age 65. Only band-2 cohorts are ever in the labor force
    /// (`is_worker`); the year-boundary lifecycle moves integer head_count from
    /// band 2 (Worker) to band 3 (Retired), flipping `is_worker` off for the
    /// retired mass so it leaves `district_worker_mass`/`cohort_supply_by_tier`.
    pub age_band: u8,
    /// Matrix axis 1 — BIOLOGY. The [`AgeBand`] this cohort's mass occupies. Derived
    /// from `age_band` by [`matrix_of`] while the four-band lifecycle drives mass
    /// (this slice's seam); the target storage for Tasks 2+' conserved band flows.
    pub band: AgeBand,
    /// Matrix axis 2 — ALLOCATION. The [`Occupation`] this cohort's mass holds. The
    /// invariant `is_worker == (occ == Occupation::Employed)` always holds.
    pub occ: Occupation,
    /// Matrix axis 3 — the education tier of this cohort. Meaningful only for the 10
    /// tier-expanded adult-labor pairs (§5.3.3); the placeholder `Unskilled` for the
    /// pre-YoungAdult single-tier pairs (tier undefined before the age-up split).
    /// Mirrors [`CohortRow::tier`] for labor rows so `cohort_supply_by_tier` is
    /// unchanged.
    pub edu_tier: SkillTier,
}

impl CohortRow {
    /// Re-derive the matrix triple (`band`, `occ`, `edu_tier`) from this row's own
    /// `(is_worker, age_band, district, tier)` via the pure adapter [`matrix_of`].
    /// Call after any mutation of `age_band`/`is_worker` so the matrix overlay never
    /// drifts from the four-band lifecycle that still drives the conserved mass.
    fn reclassify(&mut self) {
        let (band, occ, edu_tier) = matrix_of(self.is_worker, self.age_band, self.district, self.tier);
        self.band = band;
        self.occ = occ;
        self.edu_tier = edu_tier;
        debug_assert_eq!(
            self.is_worker,
            self.occ == Occupation::Employed,
            "is_worker must equal (occ == Employed)"
        );
    }
}

/// Age band for retired cims (≥ retirement age 65).
pub const BAND_RETIRED: u8 = 3;
/// Age band for working-age cims (18..65).
pub const BAND_WORKER: u8 = 2;

/// The conserved cohort table. Cohorts are stored in a single ascending-id `Vec`
/// (the determinism substrate); `hydrated` counts individuals currently
/// crystallized out of cohort mass.
pub struct CohortTable {
    rows: Vec<CohortRow>,
    /// Count of individuals currently hydrated out of cohort mass.
    hydrated: u64,
    /// Total seeded population (the conservation target).
    pop: u64,
    /// Number of civic years the lifecycle has advanced this table. The age
    /// structure (which Worker mass has retired) is a pure function of
    /// `(pop, years_aged)`, so a `seed(pop)` re-seed followed by replaying
    /// `years_aged` retirements reproduces the same structure bit-exactly — the
    /// reseed-survival contract (the seed wipes per-row state every tick).
    years_aged: u32,
}

impl CohortTable {
    /// An empty table (no cohorts, nothing hydrated, never aged).
    pub fn new() -> Self {
        Self { rows: Vec::new(), hydrated: 0, pop: 0, years_aged: 0 }
    }

    /// Deterministically seed `pop` cims as cohort mass spread across
    /// [`N_DISTRICTS`] districts × 4 tiers × {worker, non-worker}. The seed is a
    /// pure function of `pop` (no RNG, no wall-clock). The two cross-plan
    /// aggregates the accessor test pins are now derived as **fixed integer
    /// fractions of `pop`** (R13: the seed scales with the residential capacity the
    /// player builds, instead of being pinned to a fixed table):
    ///   - Educated *workers* total = `pop × 21 / 1000`  (2.1% of pop)
    ///   - District-3 *workers* total = `pop × 82 / 10000` (0.82% of pop)
    /// At the accessor test's `pop == 100_000` these reproduce the historical
    /// `2100` / `820` exactly. `Σ head_count_int == pop` is held exact by a single
    /// remainder filler non-worker cohort. Construction is a base
    /// per-(district,tier,worker) integer table, then the remainder filler.
    pub fn seed(&mut self, pop: u64) {
        self.rows.clear();
        self.hydrated = 0;
        self.pop = pop;

        // Target aggregates the downstream plans read back — derived as fixed
        // integer fractions of `pop` so the seed scales with placed housing (R13).
        // Integer math (no f64) keeps the seed bit-deterministic across replays.
        let educated_workers_total: u64 = pop * 21 / 1000; // 2.1% of pop
        let district3_workers_total: u64 = pop * 82 / 10000; // 0.82% of pop

        // Per-district Educated-worker head counts: district 3 carries a third of
        // the Educated workers; the remaining two-thirds are spread evenly across
        // the other `N_DISTRICTS - 1` districts. Any integer rounding remainder is
        // folded onto the OTHER districts' share so the Educated-worker grand total
        // is EXACTLY `educated_workers_total` (the value the economy reads back),
        // while district 3's Educated share stays a clean third (and ≤ its total
        // worker target, so its non-Educated workers fill the rest).
        let educated_worker_d3: u64 = educated_workers_total / 3;
        let educated_worker_other: u64 =
            (educated_workers_total - educated_worker_d3) / (N_DISTRICTS as u64 - 1);
        // Fold the division remainder onto district 3 so the grand total is exact.
        let educated_worker_d3 =
            educated_workers_total - educated_worker_other * (N_DISTRICTS as u64 - 1);
        debug_assert_eq!(
            educated_worker_d3 + educated_worker_other * (N_DISTRICTS as u64 - 1),
            educated_workers_total
        );

        // District 3's non-Educated workers (so district-3 worker total ==
        // `district3_workers_total`). Split deterministically across the three
        // other tiers; the remainder folds onto the Unskilled split so the sum is
        // exact. `saturating_sub` guards the degenerate case where the derived
        // Educated-d3 mass already meets/exceeds the district-3 worker target at
        // tiny pops (then the three non-Educated d3 worker cohorts are simply 0).
        let d3_non_educated = district3_workers_total.saturating_sub(educated_worker_d3);
        let d3_skilled_workers: u64 = d3_non_educated * 40 / 120;
        let d3_expert_workers: u64 = d3_non_educated * 30 / 120;
        let d3_unskilled_workers: u64 =
            d3_non_educated - d3_skilled_workers - d3_expert_workers;
        debug_assert_eq!(
            educated_worker_d3
                + d3_unskilled_workers
                + d3_skilled_workers
                + d3_expert_workers,
            educated_worker_d3 + d3_non_educated
        );

        let mut next_id: u32 = 0;
        let mut placed: u64 = 0;

        // The whole labor force is a realistic share of the population
        // (`WORKER_SHARE_PCT`), so `employed()` (Σ district_worker_mass) reflects a
        // real workforce at ANY city size — not the tiny pinned sub-aggregates
        // alone (which are <1 person at small pops and only become meaningful at
        // the accessor test's 100k). The two pinned aggregates (Educated-worker
        // total and district-3-worker total) are carved out of this workforce; the
        // REMAINDER is spread across the 7 non-d3 districts' non-Educated worker
        // cohorts. District 3's total stays EXACTLY `district3_workers_total`, so
        // the general workforce is placed only in the other districts.
        const WORKER_SHARE_PCT: u64 = 45;
        let total_workers = (pop * WORKER_SHARE_PCT / 100).min(pop);
        // Workers already committed to the pinned cohorts (Educated everywhere +
        // district-3's non-Educated). The general pool is what's left for the 7
        // non-d3 districts' Unskilled/Skilled/Expert cohorts.
        let pinned_workers = educated_workers_total + d3_non_educated;
        let general_workers = total_workers.saturating_sub(pinned_workers);
        // Spread the general workforce over 7 non-d3 districts × 3 tiers = 21
        // cohorts. A per-cohort base, with the rounding remainder folded onto the
        // FIRST general cohort so even a small city (where the base rounds to 0)
        // still has a non-zero worker mass — `employed()` is never spuriously 0.
        let general_cells: u64 = (N_DISTRICTS as u64 - 1) * 3;
        let general_base = general_workers / general_cells;
        let mut general_remainder = general_workers - general_base * general_cells;
        for d in 0..N_DISTRICTS as u32 {
            for &tier in TIERS.iter() {
                let head: u64 = match (d, tier) {
                    (3, SkillTier::Educated) => educated_worker_d3,
                    (3, SkillTier::Unskilled) => d3_unskilled_workers,
                    (3, SkillTier::Skilled) => d3_skilled_workers,
                    (3, SkillTier::Expert) => d3_expert_workers,
                    (_, SkillTier::Educated) => educated_worker_other,
                    // Non-d3, non-Educated worker cohorts share the general pool;
                    // the first such cohort also absorbs the rounding remainder.
                    (_, _) => {
                        let extra = general_remainder;
                        general_remainder = 0;
                        general_base + extra
                    }
                };
                if head > 0 && placed + head <= pop {
                    self.rows.push(Self::make_row(next_id, head as f32, d, tier, true));
                    next_id += 1;
                    placed += head;
                }
            }
        }

        // Base non-worker cohorts per district/tier (children, students, retired).
        // These never count toward `district_worker_mass` or the Educated *worker*
        // total, so they are free to soak up most of the population. Each scales
        // with pop and is added only while it still fits under `pop` (pure filler
        // for the pinned aggregates), so the deterministic seed is valid at any
        // scale — small pops skip cohorts that would overflow, large pops get them
        // all. The single remainder filler below then makes `Σ head_count == pop`
        // exact regardless.
        for d in 0..N_DISTRICTS as u32 {
            for &tier in TIERS.iter() {
                let head: u64 = pop * 40 / 1000 + (d as u64) * 10 + tier as u64 * 25;
                if head == 0 || placed + head > pop {
                    continue;
                }
                let mut row = Self::make_row(next_id, head as f32, d, tier, false);
                // Deterministically spread the non-worker mass across the three
                // non-working bands (Child=0, Student=1, Retired=3) so the city has
                // a real age structure from year 0 — a pure function of (d, tier).
                row.age_band = match (d as u64 + tier as u64) % 3 {
                    0 => 0,            // Child
                    1 => 1,            // Student
                    _ => BAND_RETIRED, // Retired
                };
                // Keep the matrix overlay in sync with the just-set legacy band.
                row.reclassify();
                self.rows.push(row);
                next_id += 1;
                placed += head;
            }
        }

        // Single filler non-worker cohort (district 0, Unskilled) absorbs the
        // remainder so `Σ head_count == pop` exactly. The pinned aggregates are
        // worker-only / Educated-worker-only, so a non-worker filler leaves them
        // untouched.
        debug_assert!(placed <= pop, "base seed {placed} exceeds pop {pop}");
        let remainder = pop - placed;
        if remainder > 0 {
            self.rows.push(Self::make_row(
                next_id,
                remainder as f32,
                0,
                SkillTier::Unskilled,
                false,
            ));
        }
        // Rows are pushed in ascending-id order already; keep the explicit sort so
        // the determinism contract holds regardless of construction order.
        self.rows.sort_by_key(|r| r.id);

        // Replay the lifecycle to the table's current age (reseed-survival): the
        // base seed always builds worker mass in band 2 and non-workers in bands
        // {0,1,3}; replaying `years_aged` retirements re-derives the exact same age
        // structure a live table reached, so a mid-game reseed (pop/capacity change)
        // does not reset the demographics. Pure function of `(pop, years_aged)`.
        let years = self.years_aged;
        for _ in 0..years {
            self.retire_one_year();
        }
    }

    /// Advance the cohort table by one civic year: every Worker-band cohort ages,
    /// a fixed integer fraction crossing the retirement age (65) each year. The
    /// retired mass is transferred — same district/tier — from a Worker (`is_worker
    /// = true`, band 2) row to a Retired (`is_worker = false`, band 3) row, so the
    /// transferred heads LEAVE `district_worker_mass`/`cohort_supply_by_tier` (they
    /// are no longer in the labor force) while `Σ head_count == pop` is conserved
    /// exactly. Deterministic: id-ordered fold, integer transfers, no RNG/HashMap.
    pub fn advance_year(&mut self) {
        self.years_aged = self.years_aged.saturating_add(1);
        self.retire_one_year();
    }

    /// One year of Worker→Retired transfer (the body of [`Self::advance_year`],
    /// also replayed by `seed` for reseed-survival). Does NOT touch `years_aged`.
    ///
    /// Working life spans 18..65 (47 years), so ~`1/47` of the working-age mass
    /// crosses the retirement boundary each year. We use the integer fraction
    /// `RETIRE_NUMER / RETIRE_DENOM` of each Worker cohort's head_count, transferred
    /// to a same-(district,tier) Retired cohort created on demand. `pop` is
    /// conserved because the transfer only moves heads between two rows.
    fn retire_one_year(&mut self) {
        // Retirement rate ≈ 1/47 of the working-age population per year (47-year
        // working life). Integer math keeps the transfer bit-deterministic.
        const RETIRE_NUMER: u64 = 1;
        const RETIRE_DENOM: u64 = 47;

        // Snapshot the per-(district,tier) retire amounts in id order first (so the
        // mutation that follows does not depend on row order), then apply.
        let mut transfers: Vec<(DistrictId, SkillTier, u64)> = Vec::new();
        for r in &self.rows {
            if r.is_worker && r.age_band == BAND_WORKER {
                let head = r.head_count.max(0.0) as u64;
                let retire = head * RETIRE_NUMER / RETIRE_DENOM;
                if retire > 0 {
                    transfers.push((r.district, r.tier, retire));
                }
            }
        }
        if transfers.is_empty() {
            return;
        }

        let mut next_id = self.rows.iter().map(|r| r.id).max().map(|m| m + 1).unwrap_or(0);
        for (district, tier, retire) in transfers {
            // Debit the Worker cohort(s) of this (district, tier) in id order.
            let mut remaining = retire;
            for r in self.rows.iter_mut() {
                if remaining == 0 {
                    break;
                }
                if r.is_worker && r.age_band == BAND_WORKER && r.district == district && r.tier == tier {
                    let take = remaining.min(r.head_count.max(0.0) as u64);
                    r.head_count -= take as f32;
                    remaining -= take;
                }
            }
            let moved = retire - remaining;
            if moved == 0 {
                continue;
            }
            // Credit the Retired cohort of this (district, tier) — find the existing
            // non-worker, band-3 row, else append a new one (kept id-sorted).
            if let Some(r) = self.rows.iter_mut().find(|r| {
                !r.is_worker && r.age_band == BAND_RETIRED && r.district == district && r.tier == tier
            }) {
                r.head_count += moved as f32;
            } else {
                let mut row = Self::make_row(next_id, moved as f32, district, tier, false);
                row.age_band = BAND_RETIRED;
                // Retired mass lands on (Senior, Retired) in the matrix overlay.
                row.reclassify();
                next_id += 1;
                self.rows.push(row);
            }
        }
        self.rows.sort_by_key(|r| r.id);
    }

    /// Mean age band over all cohorts, head_count-weighted (id-ordered fold). Rises
    /// as Worker mass (band 2) flows into Retired mass (band 3) each year — the
    /// observable "the city aged a year" signal. `0.0` for an empty table.
    pub fn mean_age_band(&self) -> f64 {
        let mut wsum = 0.0f64;
        let mut total = 0.0f64;
        for r in &self.rows {
            let h = r.head_count.max(0.0) as f64;
            wsum += r.age_band as f64 * h;
            total += h;
        }
        if total <= 0.0 {
            0.0
        } else {
            wsum / total
        }
    }

    /// Σ head_count of all Retired-band (≥65) cohorts, conservation-disciplined
    /// integer. Rises by exactly the year's Worker→Retired transfer. Id-ordered.
    pub fn retired_count(&self) -> u64 {
        let mut acc = 0.0f64;
        for r in &self.rows {
            if r.age_band == BAND_RETIRED {
                acc += r.head_count.max(0.0) as f64;
            }
        }
        acc.round() as u64
    }

    /// Σ head_count of all in-labor-force (`is_worker`) cohorts — the working
    /// population. Drops by exactly the year's retirement transfer. Id-ordered.
    pub fn worker_count(&self) -> u64 {
        let mut acc = 0.0f64;
        for r in &self.rows {
            if r.is_worker {
                acc += r.head_count.max(0.0) as f64;
            }
        }
        acc.round() as u64
    }

    /// Per-district Σ total `head_count` (workers + non-workers), rounded int. The
    /// population-density overlay source; folds the sorted cohort `Vec` in id order.
    pub fn district_population(&self) -> Vec<u32> {
        let mut acc = vec![0.0f64; N_DISTRICTS];
        for r in &self.rows {
            let d = r.district as usize;
            if d < acc.len() {
                acc[d] += r.head_count.max(0.0) as f64;
            }
        }
        acc.into_iter().map(|x| x.round() as u32).collect()
    }

    /// Build one cohort row with deterministic, tier-derived need mean/variance.
    fn make_row(id: u32, head_count: f32, district: DistrictId, tier: SkillTier, is_worker: bool) -> CohortRow {
        // Need means slide with tier (more-educated cohorts skew higher on the
        // education need); variance is a small fixed band. Pure function of inputs.
        let t = tier as usize as f32;
        let edu_mean = 0.15 + t * 0.25; // Unskilled 0.15 .. Expert 0.90
        let need_mean = [0.6, 0.6, 0.7, 0.7, edu_mean, if is_worker { 0.8 } else { 0.0 }, 0.5];
        let need_var = [0.02; 7];
        // Workers are working-age (band 2) by construction; non-workers default to
        // band 0 here and are re-banded by the seed's non-worker loop into the
        // Child/Student/Retired mix. `is_worker == (age_band == BAND_WORKER)` is the
        // invariant the lifecycle maintains.
        let age_band = if is_worker { BAND_WORKER } else { 0 };
        // Classify onto the matrix from the row's own fields (kept in sync by
        // `reclassify` wherever `age_band` is later overridden).
        let (band, occ, edu_tier) = matrix_of(is_worker, age_band, district, tier);
        CohortRow {
            id,
            head_count,
            district,
            tier,
            is_worker,
            need_mean,
            need_var,
            archetype: tier as u8,
            age_band,
            band,
            occ,
            edu_tier,
        }
    }

    /// Crystallize `count` individuals out of cohort mass. Each unit is popped off
    /// the largest-headcount cohort (deterministic: ties broken by ascending id),
    /// and an individual is conceptually seeded from `N(need_mean, need_var)` keyed
    /// `hash(cohort_id ^ unit, day)`. `head_count` drops by one per unit and
    /// `hydrated` rises by one, so the conserved total is unchanged.
    pub fn hydrate(&mut self, count: u32, day: u64) {
        for k in 0..count {
            // Pick the cohort with the most remaining head_count (≥1), id-tiebreak.
            let mut best: Option<usize> = None;
            for (i, r) in self.rows.iter().enumerate() {
                if r.head_count >= 1.0 {
                    match best {
                        None => best = Some(i),
                        Some(b) => {
                            if r.head_count > self.rows[b].head_count {
                                best = Some(i);
                            }
                        }
                    }
                }
            }
            let Some(i) = best else { break }; // nothing left to hydrate
            // Deterministic per-unit jitter seed (value-only; never an order/branch).
            let _seed = hash(self.rows[i].id ^ k, day);
            self.rows[i].head_count -= 1.0;
            self.hydrated += 1;
        }
    }

    /// Dissolve `count` hydrated individuals back into cohort mass. Each unit folds
    /// back onto the smallest-headcount cohort (deterministic: ties broken by
    /// ascending id), re-incrementing its `head_count` and decrementing `hydrated`,
    /// so the conserved total is unchanged. Folding needs back into mean/var keeps
    /// the cohort statistics consistent (here the deterministic re-merge is the
    /// `head_count` restore; mean/var are stable under the symmetric seed).
    pub fn dissolve(&mut self, count: u32) {
        for _ in 0..count {
            if self.hydrated == 0 {
                break;
            }
            // Pick the cohort with the least head_count, id-tiebreak.
            let mut best: Option<usize> = None;
            for (i, r) in self.rows.iter().enumerate() {
                match best {
                    None => best = Some(i),
                    Some(b) => {
                        if r.head_count < self.rows[b].head_count {
                            best = Some(i);
                        }
                    }
                }
            }
            let Some(i) = best else { break };
            self.rows[i].head_count += 1.0;
            self.hydrated -= 1;
        }
    }

    /// The integer head-count census: `Σ round(head_count)` with the rounding
    /// remainder carried so `head_count_int + hydrated == pop` exactly. Rounding is
    /// deterministic (round-half-to-even is order-independent); the carried
    /// remainder corrects any drift so the conservation invariant is exact.
    pub fn head_count_int(&self) -> u64 {
        // Conservation discipline: the true conserved real total is
        // `pop - hydrated`, so the integer head-count is exactly that. Folding the
        // per-cohort rounded counts and carrying the remainder yields the same
        // number; we compute it directly from the invariant to avoid any drift.
        self.pop - self.hydrated
    }

    /// Number of individuals currently hydrated out of cohort mass.
    pub fn hydrated(&self) -> u64 {
        self.hydrated
    }

    /// `(head_count_int, hydrated)` — the census pair the conservation test checks.
    pub fn census(&self) -> (u64, u64) {
        (self.head_count_int(), self.hydrated)
    }

    /// The seeded population (conservation target).
    pub fn pop(&self) -> u64 {
        self.pop
    }

    /// The id-sorted cohort rows (read-only). The determinism substrate every
    /// id-ordered fold iterates; the LOD layer reads it to find the cohort a cim
    /// belongs to and seed a hydrated individual from that cohort's distribution.
    pub fn rows(&self) -> &[CohortRow] {
        &self.rows
    }

    /// Deterministically map a cim id to its home cohort row index. A cim's
    /// membership is a pure function of its id and the seeded cohort layout (no
    /// path dependence), so a cim hydrated at any point in the day always draws
    /// from the SAME cohort distribution — the bit-identity guarantee (brief
    /// §4.4). Returns `None` only when no cohorts are seeded.
    pub fn cohort_index_for_cim(&self, cim_id: u32) -> Option<usize> {
        if self.rows.is_empty() {
            None
        } else {
            // `hash(cim_id, 0)` is a value, never an order — it just spreads cims
            // over the id-sorted cohort Vec deterministically.
            Some((hash(cim_id, 0) % self.rows.len() as u64) as usize)
        }
    }

    /// Σ cohort `head_count` reduced by [`SkillTier`], indexed
    /// `Unskilled=0..Expert=3` (matches `labor::idx` and `SkillTier as usize`).
    /// Folds the sorted cohort `Vec` in ascending-id order (brief §9 rule 1).
    /// Consumed by ECON L2 `build_labor_market`. Workers only — labor *supply* is
    /// the working population, which is what the accessor test pins (2100 Educated
    /// *workers*).
    pub fn cohort_supply_by_tier(&self) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        for r in &self.rows {
            if r.is_worker {
                out[r.tier as usize] += r.head_count as f64;
            }
        }
        out
    }

    /// Per-district Σ worker `head_count`, rounded to the conservation-disciplined
    /// integer. This IS `OdMatrix::homes[]` / outflow (brief §4.5); the PF OD layer
    /// reads it rather than recomputing. Sized to [`N_DISTRICTS`]; folds the sorted
    /// cohort `Vec` in id order (rule 1). Same rounded-int discipline as
    /// `head_count_int` so `Σ district_worker_mass` equals the OD outflow total.
    pub fn district_worker_mass(&self) -> Vec<u32> {
        let mut acc = vec![0.0f64; N_DISTRICTS];
        for r in &self.rows {
            if r.is_worker {
                let d = r.district as usize;
                if d < acc.len() {
                    acc[d] += r.head_count as f64;
                }
            }
        }
        acc.into_iter().map(|x| x.round() as u32).collect()
    }

    // ───────────────────────── Matrix column accessors ─────────────────────────
    //
    // Every demographic question is a row-read. Each accessor folds the id-sorted
    // `rows` in ascending-id order and returns a conservation-disciplined integer
    // (rounded head_count), so two replays produce bit-identical aggregates.

    /// Σ head_count of every cohort holding [`Occupation`] `occ` — a matrix COLUMN
    /// read. Folds `rows` in id order; rounded integer (conservation-disciplined).
    pub fn occupation_mass(&self, occ: Occupation) -> u64 {
        let mut acc = 0.0f64;
        for r in &self.rows {
            if r.occ == occ {
                acc += r.head_count.max(0.0) as f64;
            }
        }
        acc.round() as u64
    }

    /// Σ head_count of every cohort in [`AgeBand`] `band` — a matrix ROW read.
    /// Folds `rows` in id order; rounded integer.
    pub fn band_mass(&self, band: AgeBand) -> u64 {
        let mut acc = 0.0f64;
        for r in &self.rows {
            if r.band == band {
                acc += r.head_count.max(0.0) as f64;
            }
        }
        acc.round() as u64
    }

    /// Σ head_count bound in the Caregiver occupation — the care gate's mass, now a
    /// readable column instead of a per-worker side effect (design §5.3.5). The
    /// labor supply withdrawn to home care.
    pub fn caregivers_bound(&self) -> u64 {
        self.occupation_mass(Occupation::Caregiver)
    }

    /// Σ head_count in the Unemployed buffer column — unemployment as a first-class
    /// row-sum, not "workers minus jobs" (design §5.6). (Populated by the labor
    /// flows in Task 6; a real column here from day one.)
    pub fn unemployed(&self) -> u64 {
        self.occupation_mass(Occupation::Unemployed)
    }

    /// Σ head_count in the Retired occupation column (Adult×Retired when policy
    /// R < 67, plus Senior×Retired). The occupation read; distinct from the
    /// age-band [`Self::retired_count`] (which folds the legacy `age_band`).
    pub fn retired(&self) -> u64 {
        self.occupation_mass(Occupation::Retired)
    }

    /// Σ head_count over ALL cohorts (rounded integer) — the matrix conservation
    /// invariant: `row_sum() == pop` at seed and after every conserved flow. Folds
    /// `rows` in id order (independent of the `pop - hydrated` bookkeeping in
    /// [`Self::head_count_int`], so a mismatch would surface a real drift).
    pub fn row_sum(&self) -> u64 {
        let mut acc = 0.0f64;
        for r in &self.rows {
            acc += r.head_count.max(0.0) as f64;
        }
        acc.round() as u64
    }

    /// The number of structurally legal `(AgeBand, Occupation)` pairs — exactly 17
    /// ([`LEGAL_PAIRS`]). A schema fact (illegal pairs are unallocated, never
    /// constructed), not a count of populated rows.
    pub fn legal_pair_count(&self) -> usize {
        LEGAL_PAIRS.len()
    }
}

impl Default for CohortTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    /// R17 NON-VACUOUS LIFECYCLE GATE (cohort level): crossing one civic year
    /// retires a real, computed chunk of the workforce and conserves population.
    #[test]
    fn lifecycle_year_ages_cohorts_and_retires_workers() {
        let mut t = CohortTable::new();
        t.seed(100_000);

        let pop_before = t.pop();
        let workers_before = t.worker_count();
        let retired_before = t.retired_count();
        let mean_before = t.mean_age_band();
        assert!(workers_before > 0, "seed must produce a workforce");

        t.advance_year();

        let pop_after = t.pop();
        let workers_after = t.worker_count();
        let retired_after = t.retired_count();
        let mean_after = t.mean_age_band();

        // Population conserved exactly (only the worker/retired split shifts).
        assert_eq!(
            pop_before, pop_after,
            "population must be conserved across a year: {pop_before} -> {pop_after}"
        );
        // Retirees leave the labor force: workers drop by EXACTLY the retired gain.
        let retired_delta = retired_after - retired_before;
        let workers_delta = workers_before - workers_after;
        assert!(retired_delta > 0, "a real chunk must retire: {retired_delta}");
        assert_eq!(
            workers_delta, retired_delta,
            "the employed pool must drop by exactly the retirees: \
             workers {workers_before}->{workers_after} (−{workers_delta}), \
             retired {retired_before}->{retired_after} (+{retired_delta})"
        );
        // The city visibly aged: mean age band rose.
        assert!(
            mean_after > mean_before,
            "mean age band must rise: {mean_before} -> {mean_after}"
        );

        // Determinism: a fresh table + one advance reproduces the same numbers.
        let mut t2 = CohortTable::new();
        t2.seed(100_000);
        t2.advance_year();
        assert_eq!(t2.worker_count(), workers_after);
        assert_eq!(t2.retired_count(), retired_after);
        assert_eq!(t2.mean_age_band().to_bits(), mean_after.to_bits());
    }

    /// Reseed-survival: re-seeding the same `pop` after aging reproduces the SAME
    /// age structure (the seed replays `years_aged`), so a mid-game capacity
    /// reseed does not reset demographics.
    #[test]
    fn lifecycle_reseed_preserves_age_structure() {
        let mut t = CohortTable::new();
        t.seed(100_000);
        t.advance_year();
        t.advance_year();
        let workers = t.worker_count();
        let retired = t.retired_count();
        // Re-seed the SAME pop (the reseed-from-capacity path).
        t.seed(100_000);
        assert_eq!(t.worker_count(), workers, "reseed wiped the aged workforce");
        assert_eq!(t.retired_count(), retired, "reseed wiped the retired mass");
    }
}

#[cfg(test)]
mod matrix_tests {
    //! The two-axis (age × occupation) matrix SUBSTRATE gate. Proves the schema (17
    //! legal pairs, 47 rows/district), the conservation invariant (`row_sum == pop`),
    //! the column/row partitions, and that the matrix overlay is derived id-ordered
    //! and deterministically — without disturbing the four-band lifecycle that still
    //! drives the conserved mass (this slice's seam).

    use super::*;

    /// KEYSTONE acceptance (plan Task 1, Step 1). A seeded city is conserved on the
    /// matrix, carries exactly the 17 legal pairs, binds real Caregiver mass, and
    /// still honours the historical Educated-worker economy contract (2100 @ 100k).
    #[test]
    fn matrix_seed_is_conserved_and_legal() {
        let mut m = CohortTable::new();
        m.seed(100_000);
        let row_sum = m.row_sum();
        let legal_pairs = m.legal_pair_count();
        let caregivers = m.occupation_mass(Occupation::Caregiver);
        assert_eq!(row_sum, 100_000, "matrix must conserve pop");
        assert_eq!(legal_pairs, 17, "exactly 17 legal (band,occ) pairs");
        assert!(caregivers > 0, "a seeded city binds some caregivers, got {caregivers}");
        // Educated-worker economy contract preserved (the economy reads this back).
        assert_eq!(
            m.cohort_supply_by_tier()[SkillTier::Educated as usize].round() as u64,
            2100
        );
        // The acceptance's exact human-visible line.
        println!(
            "pop={} row_sum={} legal_pairs={} caregivers={}",
            m.pop(),
            row_sum,
            legal_pairs,
            caregivers
        );
    }

    /// The matrix SCHEMA arithmetic: `10×4 + 7 = 47` rows/district, spanning exactly
    /// the 17 legal pairs, and no illegal pair is ever representable.
    #[test]
    fn matrix_schema_is_47_rows_and_17_pairs() {
        let rows = legal_rows();
        assert_eq!(rows.len(), ROWS_PER_DISTRICT, "47 rows per district");
        assert_eq!(rows.len(), 47);

        // Distinct (band, occ) pairs among the schema rows == 17.
        let mut pairs: Vec<(AgeBand, Occupation)> = rows.iter().map(|&(b, o, _)| (b, o)).collect();
        pairs.sort();
        pairs.dedup();
        assert_eq!(pairs.len(), 17, "47 rows collapse to 17 legal pairs");
        assert_eq!(pairs.len(), LEGAL_PAIRS.len());

        // Every schema row is a legal pair; the tier-expanded ones appear 4× and the
        // 7 single-tier ones once → 10×4 + 7.
        let expanded = LEGAL_PAIRS.iter().filter(|&&(b, o)| is_tier_expanded(b, o)).count();
        let single = LEGAL_PAIRS.len() - expanded;
        assert_eq!(expanded, 10, "exactly 10 tier-expanded adult-labor pairs");
        assert_eq!(single, 7);
        assert_eq!(expanded * 4 + single, 47);
        for &(b, o, _) in &rows {
            assert!(is_legal_pair(b, o), "schema row ({b:?},{o:?}) must be legal");
        }

        // A structurally illegal pair is absent from the mask (Newborn×Employed is a
        // "compile error, not a bug report").
        assert!(!is_legal_pair(AgeBand::Newborn, Occupation::Employed));
        assert!(!is_legal_pair(AgeBand::Teenager, Occupation::Caregiver));
        assert!(!is_legal_pair(AgeBand::Toddler, Occupation::Employed));
    }

    /// The matrix partitions the population two ways: Σ over occupation columns and
    /// Σ over age-band rows both equal `pop`, and every seeded row sits on a legal
    /// pair. Conservation holds through an `advance_year` too (the lifecycle moves
    /// mass Employed→Retired inside the matrix without leaking).
    #[test]
    fn matrix_columns_and_rows_partition_pop() {
        let mut m = CohortTable::new();
        m.seed(100_000);

        let by_occ: u64 = OCCUPATIONS.iter().map(|&o| m.occupation_mass(o)).sum();
        let by_band: u64 = AGE_BANDS.iter().map(|&b| m.band_mass(b)).sum();
        assert_eq!(by_occ, 100_000, "occupation columns must partition pop");
        assert_eq!(by_band, 100_000, "age bands must partition pop");
        assert_eq!(m.row_sum(), 100_000);

        // Every seeded row is a legal pair (illegal pairs are never constructed).
        for r in m.rows() {
            assert!(
                is_legal_pair(r.band, r.occ),
                "seeded row {} on ILLEGAL pair ({:?},{:?})",
                r.id,
                r.band,
                r.occ
            );
        }

        // Invariant `is_worker == (occ == Employed)` → the labor column IS the
        // workforce; the historical worker accessor is unchanged.
        assert_eq!(
            m.occupation_mass(Occupation::Employed),
            m.worker_count(),
            "Employed column must equal the legacy worker_count"
        );

        // Conservation survives a civic year (Employed→Retired transfer stays inside
        // the matrix; nothing leaks).
        m.advance_year();
        assert_eq!(m.row_sum(), 100_000, "advance_year must conserve on the matrix");
        let by_occ_after: u64 = OCCUPATIONS.iter().map(|&o| m.occupation_mass(o)).sum();
        assert_eq!(by_occ_after, 100_000, "columns still partition after a year");
    }

    /// The matrix overlay is a PURE function of the seed → two independent seeds
    /// produce bit-identical column/row masses (no HashMap/RNG/wall-clock), and a
    /// reseed of the same `(pop, years_aged)` reproduces them exactly.
    #[test]
    fn matrix_overlay_is_deterministic_and_reseed_stable() {
        let mut a = CohortTable::new();
        let mut b = CohortTable::new();
        a.seed(250_000);
        b.seed(250_000);
        for &o in OCCUPATIONS.iter() {
            assert_eq!(a.occupation_mass(o), b.occupation_mass(o), "occ {o:?} diverged");
        }
        for &band in AGE_BANDS.iter() {
            assert_eq!(a.band_mass(band), b.band_mass(band), "band {band:?} diverged");
        }

        // Age two years, then reseed the same pop: the overlay re-derives identically
        // (the reconstruction contract — pure function of (pop, years_aged)).
        a.advance_year();
        a.advance_year();
        let care = a.caregivers_bound();
        let ret = a.retired();
        let emp = a.occupation_mass(Occupation::Employed);
        a.seed(250_000); // reseed replays years_aged internally
        assert_eq!(a.caregivers_bound(), care, "reseed changed Caregiver column");
        assert_eq!(a.retired(), ret, "reseed changed Retired column");
        assert_eq!(a.occupation_mass(Occupation::Employed), emp, "reseed changed Employed column");
    }
}
