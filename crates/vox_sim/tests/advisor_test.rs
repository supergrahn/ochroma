use vox_sim::advisor::*;

#[test]
fn budget_crisis_detected() {
    let mut advisor = AdvisorSystem::new();
    advisor.evaluate(1000, -500.0, 0.1, false, false, false, false, false, false, false, 0.3);
    assert!(advisor.messages.iter().any(|m| m.title.contains("Crisis")));
}

#[test]
fn housing_shortage_detected() {
    let mut advisor = AdvisorSystem::new();
    advisor.evaluate(1000, 50000.0, 0.1, true, false, false, false, false, false, false, 0.3);
    assert!(advisor.messages.iter().any(|m| m.title.contains("Housing")));
}

#[test]
fn no_issues_no_messages() {
    let mut advisor = AdvisorSystem::new();
    advisor.evaluate(1000, 50000.0, 0.05, false, false, false, false, false, false, false, 0.3);
    // Should only have low-priority or no messages
    assert!(advisor.messages.iter().all(|m| m.priority <= 2));
}

#[test]
fn messages_sorted_by_priority() {
    let mut advisor = AdvisorSystem::new();
    advisor.evaluate(1000, -100.0, 0.5, true, true, true, true, true, true, true, 0.9);
    // Multiple issues -- verify sorted by priority
    if advisor.messages.len() >= 2 {
        assert!(advisor.messages[0].priority >= advisor.messages[1].priority);
    }
}

#[test]
fn cooldown_prevents_spam() {
    let mut advisor = AdvisorSystem::new();
    advisor.evaluate(1000, -100.0, 0.5, true, false, false, false, false, false, false, 0.3);
    assert!(!advisor.messages.is_empty());
    advisor.evaluate(1000, -100.0, 0.5, true, false, false, false, false, false, false, 0.3);
    assert!(advisor.messages.is_empty(), "Cooldown should prevent immediate re-evaluation");
}
