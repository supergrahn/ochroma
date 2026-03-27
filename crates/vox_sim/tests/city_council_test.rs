use vox_sim::city_council::{CityCouncil, CityState, CouncilMember, Ideology, PolicyArea};

fn make_council() -> CityCouncil {
    let mut council = CityCouncil::new();
    council.add_member(CouncilMember::new(
        0,
        "Alice".to_string(),
        Ideology::Progressive,
        vec![PolicyArea::Education, PolicyArea::Healthcare],
    ));
    council.add_member(CouncilMember::new(
        1,
        "Bob".to_string(),
        Ideology::Conservative,
        vec![PolicyArea::PublicSafety, PolicyArea::Taxation],
    ));
    council.add_member(CouncilMember::new(
        2,
        "Carol".to_string(),
        Ideology::Moderate,
        vec![PolicyArea::Infrastructure, PolicyArea::Housing],
    ));
    council
}

#[test]
fn test_submit_and_count_proposals() {
    let mut council = make_council();
    assert_eq!(council.pending_count(), 0);

    council.submit_proposal(
        "Build new school".to_string(),
        PolicyArea::Education,
        50_000.0,
        80_000.0,
    );
    council.submit_proposal(
        "Hire more police".to_string(),
        PolicyArea::PublicSafety,
        20_000.0,
        40_000.0,
    );

    assert_eq!(council.pending_count(), 2);
}

#[test]
fn test_vote_on_proposal() {
    let mut council = make_council();
    let id = council.submit_proposal(
        "Expand healthcare".to_string(),
        PolicyArea::Healthcare,
        60_000.0,
        100_000.0,
    );

    let result = council.vote(id).expect("Vote should succeed");
    // With 3 members, we should get a definitive result
    assert_eq!(result.votes_for + result.votes_against, 3);
    // Proposal should be removed from pending
    assert_eq!(council.pending_count(), 0);
}

#[test]
fn test_progressive_favours_high_spend_education() {
    let mut council = CityCouncil::new();
    // All progressive members who care about education
    for i in 0..3 {
        council.add_member(CouncilMember::new(
            i,
            format!("Member {}", i),
            Ideology::Progressive,
            vec![PolicyArea::Education],
        ));
    }

    let id = council.submit_proposal(
        "Major education reform".to_string(),
        PolicyArea::Education,
        80_000.0,
        200_000.0,
    );

    let result = council.vote(id).unwrap();
    assert!(
        result.passed,
        "Progressive council should pass education spending"
    );
}

#[test]
fn test_conservative_rejects_high_spend() {
    let mut council = CityCouncil::new();
    // All conservative members who don't care about education
    for i in 0..3 {
        council.add_member(CouncilMember::new(
            i,
            format!("Member {}", i),
            Ideology::Conservative,
            vec![PolicyArea::Taxation],
        ));
    }

    let id = council.submit_proposal(
        "Massive spending program".to_string(),
        PolicyArea::Education,
        200_000.0,
        100_000.0, // bad cost/benefit ratio
    );

    let result = council.vote(id).unwrap();
    assert!(
        !result.passed,
        "Conservative council should reject high-spend low-benefit proposal"
    );
}

#[test]
fn test_council_tick_generates_proposals_for_high_crime() {
    let mut council = make_council();
    let state = CityState {
        crime_rate: 0.8,
        ..Default::default()
    };

    council.council_tick(&state);
    assert!(
        council.pending_count() > 0,
        "High crime should generate proposals"
    );

    // Should contain a public safety proposal
    let has_safety = council
        .proposals
        .iter()
        .any(|p| p.area == PolicyArea::PublicSafety);
    assert!(has_safety, "Should generate public safety proposal");
}

#[test]
fn test_council_tick_generates_proposals_for_unemployment() {
    let mut council = make_council();
    let state = CityState {
        unemployment_rate: 0.5,
        ..Default::default()
    };

    council.council_tick(&state);
    let has_infra = council
        .proposals
        .iter()
        .any(|p| p.area == PolicyArea::Infrastructure);
    assert!(
        has_infra,
        "High unemployment should generate infrastructure proposal"
    );
}

#[test]
fn test_council_tick_no_proposals_for_healthy_city() {
    let mut council = make_council();
    let state = CityState {
        population: 10_000,
        average_satisfaction: 0.8,
        crime_rate: 0.1,
        unemployment_rate: 0.05,
        pollution_level: 0.1,
        budget: 50_000.0,
        education_coverage: 0.9,
        health_coverage: 0.9,
        transport_coverage: 0.8,
    };

    council.council_tick(&state);
    assert_eq!(
        council.pending_count(),
        0,
        "Healthy city should not generate proposals"
    );
}

#[test]
fn test_vote_all() {
    let mut council = make_council();
    council.submit_proposal(
        "Proposal A".to_string(),
        PolicyArea::Education,
        30_000.0,
        60_000.0,
    );
    council.submit_proposal(
        "Proposal B".to_string(),
        PolicyArea::PublicSafety,
        15_000.0,
        30_000.0,
    );

    let results = council.vote_all();
    assert_eq!(results.len(), 2);
    assert_eq!(council.pending_count(), 0);
    assert_eq!(council.vote_history.len(), 2);
}

#[test]
fn test_enacted_policies_tracked() {
    let mut council = CityCouncil::new();
    // Progressive council that will pass education proposal
    for i in 0..3 {
        council.add_member(CouncilMember::new(
            i,
            format!("Prog {}", i),
            Ideology::Progressive,
            vec![PolicyArea::Education],
        ));
    }

    let id = council.submit_proposal(
        "Education boost".to_string(),
        PolicyArea::Education,
        60_000.0,
        150_000.0,
    );
    let result = council.vote(id).unwrap();

    if result.passed {
        assert_eq!(council.enacted_count(), 1);
        assert_eq!(council.enacted_policies[0].description, "Education boost");
    }
}

#[test]
fn test_remove_member() {
    let mut council = make_council();
    assert_eq!(council.members.len(), 3);
    council.remove_member(1);
    assert_eq!(council.members.len(), 2);
    assert!(council.members.iter().all(|m| m.id != 1));
}
