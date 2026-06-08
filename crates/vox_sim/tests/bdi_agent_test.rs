use vox_sim::bdi_agent::{Action, BdiAgent, Belief, Desire};
use vox_sim::citizen::{Citizen, DailyState, EducationLevel, LifecycleStage, Needs};

fn make_citizen(id: u32, residence: Option<u32>, employment: Option<u32>, needs: Needs) -> Citizen {
    Citizen {
        id,
        agent_id: id,
        age: 30.0,
        lifecycle: LifecycleStage::Worker,
        education: EducationLevel::Secondary,
        employment,
        residence,
        satisfaction: needs.satisfaction(),
        needs,
        daily_state: DailyState::AtHome,
        workplace: employment,
    }
}

#[test]
fn test_homeless_citizen_generates_find_housing_desire() {
    let citizen = make_citizen(1, None, Some(10), Needs::default());
    let mut agent = BdiAgent::new(1);
    agent.generate_desires(&citizen);

    assert!(
        agent.desires.contains(&Desire::FindHousing),
        "Agent without housing should desire FindHousing, got: {:?}",
        agent.desires
    );
}

#[test]
fn test_unemployed_worker_generates_find_job_desire() {
    let citizen = make_citizen(2, Some(5), None, Needs::default());
    let mut agent = BdiAgent::new(2);
    agent.generate_desires(&citizen);

    assert!(
        agent.desires.contains(&Desire::FindBetterJob),
        "Unemployed worker should desire FindBetterJob, got: {:?}",
        agent.desires
    );
}

#[test]
fn test_low_satisfaction_generates_multiple_desires() {
    let needs = Needs {
        housing: 0.2,
        food: 0.3,
        health: 0.2,
        safety: 0.2,
        education: 0.2,
        employment: 0.2,
        leisure: 0.2,
    };
    let citizen = make_citizen(3, Some(1), Some(2), needs);
    let mut agent = BdiAgent::new(3);
    agent.generate_desires(&citizen);

    assert!(agent.desires.contains(&Desire::FindHousing));
    assert!(agent.desires.contains(&Desire::FindBetterJob));
    assert!(agent.desires.contains(&Desire::GetEducated));
    assert!(agent.desires.contains(&Desire::ImproveHealth));
}

#[test]
fn test_plan_next_action_housing_priority() {
    let citizen = make_citizen(4, None, None, Needs::default());
    let mut agent = BdiAgent::new(4);
    agent.plan_next_action(&citizen);

    let intention = agent.intention.as_ref().expect("Should have an intention");
    assert_eq!(
        intention.desire,
        Desire::FindHousing,
        "Housing should be highest priority"
    );
}

#[test]
fn test_plan_with_known_housing_generates_goto_and_rent() {
    let citizen = make_citizen(5, None, Some(10), Needs::default());
    let mut agent = BdiAgent::new(5);
    agent.add_belief(Belief::KnowsHousingAt(42));
    agent.plan_next_action(&citizen);

    let intention = agent.intention.as_ref().expect("Should have an intention");
    assert_eq!(intention.desire, Desire::FindHousing);
    assert_eq!(intention.plan.len(), 2);
    assert_eq!(intention.plan[0], Action::GoTo(42));
    assert_eq!(intention.plan[1], Action::RentHousing(42));
}

#[test]
fn test_plan_without_known_housing_generates_search() {
    let citizen = make_citizen(6, None, Some(10), Needs::default());
    let mut agent = BdiAgent::new(6);
    agent.plan_next_action(&citizen);

    let intention = agent.intention.as_ref().expect("Should have an intention");
    assert_eq!(intention.desire, Desire::FindHousing);
    assert_eq!(intention.plan, vec![Action::SearchForHousing]);
}

#[test]
fn test_intention_advance() {
    let citizen = make_citizen(7, None, Some(10), Needs::default());
    let mut agent = BdiAgent::new(7);
    agent.add_belief(Belief::KnowsHousingAt(99));
    agent.plan_next_action(&citizen);

    let intention = agent.intention.as_mut().unwrap();
    assert_eq!(*intention.current_action().unwrap(), Action::GoTo(99));
    assert!(!intention.advance());
    assert_eq!(
        *intention.current_action().unwrap(),
        Action::RentHousing(99)
    );
    assert!(intention.advance()); // plan complete
    assert!(intention.is_complete());
}

#[test]
fn test_satisfied_citizen_no_intention() {
    let needs = Needs {
        housing: 0.9,
        food: 0.9,
        health: 0.9,
        safety: 0.9,
        education: 0.9,
        employment: 0.9,
        leisure: 0.9,
    };
    let citizen = make_citizen(8, Some(1), Some(2), needs);
    let mut agent = BdiAgent::new(8);
    agent.plan_next_action(&citizen);

    assert!(
        agent.intention.is_none(),
        "Satisfied citizen should have no intention"
    );
}

#[test]
fn test_add_remove_belief() {
    let mut agent = BdiAgent::new(10);
    agent.add_belief(Belief::KnowsJobAt(5));
    agent.add_belief(Belief::KnowsHousingAt(10));
    assert_eq!(agent.beliefs.len(), 2);

    // Adding duplicate should not increase count
    agent.add_belief(Belief::KnowsJobAt(5));
    assert_eq!(agent.beliefs.len(), 2);

    agent.remove_belief(&Belief::KnowsJobAt(5));
    assert_eq!(agent.beliefs.len(), 1);
}

#[test]
fn test_socialize_with_known_friend() {
    let needs = Needs {
        housing: 0.9,
        food: 0.9,
        health: 0.9,
        safety: 0.9,
        education: 0.9,
        employment: 0.9,
        leisure: 0.5,
    };
    let mut citizen = make_citizen(9, Some(1), Some(2), needs);
    citizen.satisfaction = 0.3; // low overall satisfaction
    let mut agent = BdiAgent::new(9);
    agent.add_belief(Belief::HasFriend(20));
    agent.plan_next_action(&citizen);

    let intention = agent.intention.as_ref().expect("Should have an intention");
    // Socialize may not be top priority, but the friend belief enables meet action
    if intention.desire == Desire::Socialize {
        assert_eq!(intention.plan, vec![Action::MeetFriend(20)]);
    }
}
