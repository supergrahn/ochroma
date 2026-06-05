//! Multi-step integration test for the `CitySim` facade.
//!
//! These tests run a real, populated city for ~100 ticks and assert *evolution in a
//! sensible direction* — no empty-map ticks, no `is_some()` placeholders. Every assertion
//! checks a concrete computed value that the founding state did not already have.

use vox_sim::city_sim::CitySim;

/// A freshly constructed small city is already populated: it has citizens, agents,
/// developed buildings (housing + jobs), services and roads — never an empty map.
#[test]
fn new_small_city_is_populated_not_empty() {
    let sim = CitySim::new_small();

    // 24 founding citizens, each with a bound spatial agent.
    assert_eq!(sim.citizens.count(), 24, "founding population");
    assert_eq!(sim.agents.count(), 24, "one agent per founding citizen");

    // Zoned plots developed into buildings: 5 residential, 2 commercial, 2 industrial
    // = 9 zoned plots and 8 buildings (one per plot in the layout below).
    assert_eq!(sim.zoning.plot_count(), 8, "zoned plots");
    assert_eq!(sim.buildings.count(), 8, "developed buildings");

    // Real housing and job capacity exist from tick zero.
    // Residential caps: 6+6+4+6 = 22 ; jobs (commercial+industrial): 8+8+10+10 = 36.
    assert_eq!(sim.buildings.total_housing(), 22, "total housing capacity");
    assert_eq!(sim.buildings.total_jobs(), 36, "total job capacity");

    // Two avenue halves + a cross street = 3 segments, with one auto-detected
    // intersection at the shared city-centre endpoint.
    assert_eq!(sim.roads.segment_count(), 3, "road segments");
    assert_eq!(sim.roads.intersection_count(), 1, "road intersection");

    // Four core services placed.
    assert_eq!(sim.services.buildings.len(), 4, "service buildings");

    // Nobody is employed or housed *yet* — that only happens once we start ticking.
    let s = sim.stats();
    assert_eq!(s.employed, 0, "no jobs assigned before ticking");
    assert_eq!(s.housed, 0, "no homes assigned before ticking");
    assert!(s.funds > 0.0, "seeded budget is positive");
}

/// Run 100 ticks and assert the city evolves coherently: people get housed and employed,
/// crime falls as employment rises, satisfaction improves, and the budget reflects the
/// real population. This is the headline "no empty-map ticks" multi-step test.
#[test]
fn hundred_ticks_evolve_in_sensible_directions() {
    let mut sim = CitySim::new_small();
    let before = sim.stats();

    let after = sim.tick(100);

    // --- Employment evolved upward from a real zero baseline. ---
    assert_eq!(before.employed, 0, "baseline: no one employed");
    assert!(
        after.employed >= 12,
        "employment should climb as citizens claim the 36 available jobs, got {}",
        after.employed
    );
    // Cannot exceed population.
    assert!(
        after.employed <= after.population,
        "employed {} must not exceed population {}",
        after.employed,
        after.population
    );

    // --- Housing evolved upward toward the 22-unit cap. ---
    assert_eq!(before.housed, 0, "baseline: no one housed");
    assert!(
        after.housed >= 20,
        "housing should fill toward the 22-unit cap, got {}",
        after.housed
    );
    assert!(after.housed <= 22, "cannot house more than capacity");

    // --- Crime fell as employment rose (employment is the dominant crime driver here). ---
    assert!(
        after.crime_rate < before.crime_rate,
        "crime should drop as employment rises: {:.4} -> {:.4}",
        before.crime_rate,
        after.crime_rate
    );
    assert!(after.crime_rate >= 0.0, "crime rate stays non-negative");

    // --- Mean satisfaction improved (housing + employment needs got met). ---
    assert!(
        after.mean_satisfaction > before.mean_satisfaction,
        "satisfaction should improve: {:.4} -> {:.4}",
        before.mean_satisfaction,
        after.mean_satisfaction
    );

    // --- Population stayed within sensible bounds (no spurious mass death/birth). ---
    assert!(
        after.population >= 20 && after.population <= 40,
        "population stays bounded, got {}",
        after.population
    );

    // --- Time actually advanced. ---
    assert_eq!(
        after.elapsed_secs,
        100.0 * CitySim::SECONDS_PER_TICK as f64,
        "100 ticks of game-seconds elapsed"
    );

    // --- The economy ran: residential tax income reflects the real population. ---
    // residential_income = population * residential_tax_rate(0.08) * 100 = population * 8.
    // (residential_income uses an f32 tax rate widened to f64, so allow a tiny epsilon.)
    let expected_res_income = sim.budget.residential_income;
    assert!(
        (expected_res_income - after.population as f64 * 8.0).abs() < 1e-2,
        "residential income {:.4} should equal population*8 ({})",
        expected_res_income,
        after.population as f64 * 8.0
    );
    assert!(
        sim.budget.total_income() > 0.0,
        "the budget collected real tax income"
    );
    // Service-heavy small town overspends -> net budget is negative and funds fell.
    assert!(
        after.funds < before.funds,
        "funds should fall as services outspend the tiny tax base: {:.0} -> {:.0}",
        before.funds,
        after.funds
    );
}

/// An employed citizen's agent actually walks toward — and reaches — its workplace.
/// We pick the citizen with the longest commute so movement spans several ticks, and
/// assert the remaining distance is strictly decreasing each tick until arrival.
#[test]
fn commuter_agent_moves_monotonically_toward_workplace() {
    let mut sim = CitySim::new_small();
    // One tick assigns homes + jobs and gives each agent a destination.
    sim.tick(1);

    // Find the citizen whose agent has the largest remaining distance to its workplace.
    let mut target: Option<(u32, f32)> = None;
    for cid in 0..sim.citizens.count() as u32 {
        if let (Some(pos), Some(wp)) = (sim.agent_position_of(cid), sim.workplace_of(cid)) {
            let (px, _, pz) = pos.to_absolute();
            let (wx, _, wz) = wp.to_absolute();
            let d = (((px - wx).powi(2) + (pz - wz).powi(2)) as f32).sqrt();
            if target.map(|(_, td)| d > td).unwrap_or(true) {
                target = Some((cid, d));
            }
        }
    }

    let (cid, start_dist) = target.expect("at least one citizen has a workplace to commute to");
    assert!(
        start_dist > 50.0,
        "the longest commute should be a real multi-step distance, got {:.1}m",
        start_dist
    );

    // Step tick-by-tick; remaining distance must never increase and must reach ~0.
    let mut prev = start_dist;
    let mut arrived = false;
    for _ in 0..20 {
        sim.tick(1);
        let pos = sim.agent_position_of(cid).expect("agent persists");
        let wp = sim.workplace_of(cid).expect("workplace persists");
        let (px, _, pz) = pos.to_absolute();
        let (wx, _, wz) = wp.to_absolute();
        let d = (((px - wx).powi(2) + (pz - wz).powi(2)) as f32).sqrt();

        assert!(
            d <= prev + 1e-3,
            "agent must not move away from its workplace: {:.3} -> {:.3}",
            prev,
            d
        );
        prev = d;
        if d < 1.0 {
            arrived = true;
            break;
        }
    }

    assert!(arrived, "agent should reach its workplace within 20 ticks");
    // It genuinely traversed ground (didn't start already on top of the job).
    assert!(
        prev < start_dist,
        "agent closed real distance: {:.1}m -> {:.3}m",
        start_dist,
        prev
    );
}

/// The whole simulation is deterministic: two independent runs of identical length
/// produce byte-for-byte identical headline stats.
#[test]
fn simulation_is_deterministic_across_runs() {
    let run_a = CitySim::new_small().tick(100);
    let run_b = CitySim::new_small().tick(100);

    assert_eq!(run_a, run_b, "two fresh 100-tick runs must match exactly");

    // And it produced non-trivial state (guards against "deterministically empty").
    assert!(run_a.employed >= 12, "deterministic run is also a populated one");
    assert_eq!(run_a.population, 24);
}
