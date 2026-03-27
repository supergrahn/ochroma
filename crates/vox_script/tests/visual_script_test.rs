use std::collections::HashMap;
use vox_script::visual_script::*;

#[test]
fn test_building_placed_celebration_script() {
    // on BuildingPlaced -> if population > 100 -> spawn celebration
    let mut script = VisualScript::new("celebration_script");
    let event = script.add_node(ScriptNode::Event("BuildingPlaced".into()));
    let condition = script.add_node(ScriptNode::Condition(
        "population".into(),
        ComparisonOp::GreaterThan,
        100.0,
    ));
    let action = script.add_node(ScriptNode::Action(
        "spawn_celebration".into(),
        Some("fireworks".into()),
    ));
    script.connect(event, condition);
    script.connect(condition, action);

    // With population > 100, celebration should fire
    let mut vars = HashMap::new();
    vars.insert("population".into(), 250.0);
    let result = execute_script(&script, "BuildingPlaced", &vars);
    assert_eq!(result.triggered_actions.len(), 1);
    assert_eq!(result.triggered_actions[0].0, "spawn_celebration");
    assert_eq!(
        result.triggered_actions[0].1,
        Some("fireworks".to_string())
    );

    // With population <= 100, no celebration
    vars.insert("population".into(), 50.0);
    let result = execute_script(&script, "BuildingPlaced", &vars);
    assert!(result.triggered_actions.is_empty());
}

#[test]
fn test_unmatched_trigger_does_nothing() {
    let mut script = VisualScript::new("noop");
    let event = script.add_node(ScriptNode::Event("BuildingPlaced".into()));
    let action = script.add_node(ScriptNode::Action("boom".into(), None));
    script.connect(event, action);

    let result = execute_script(&script, "ZoneChanged", &HashMap::new());
    assert!(result.triggered_actions.is_empty());
}

#[test]
fn test_loop_node_repeats_action() {
    let mut script = VisualScript::new("loop_test");
    let event = script.add_node(ScriptNode::Event("Start".into()));
    let loop_node = script.add_node(ScriptNode::Loop(3));
    let action = script.add_node(ScriptNode::Action("ping".into(), None));
    script.connect(event, loop_node);
    script.connect(loop_node, action);

    let result = execute_script(&script, "Start", &HashMap::new());
    assert_eq!(result.triggered_actions.len(), 3);
    assert!(result.triggered_actions.iter().all(|(n, _)| n == "ping"));
}

#[test]
fn test_variable_set_and_math_op() {
    let mut script = VisualScript::new("math_test");
    let event = script.add_node(ScriptNode::Event("Start".into()));
    let set_var = script.add_node(ScriptNode::Variable(VarAccess::Set("score".into(), 10.0)));
    let math = script.add_node(ScriptNode::MathOp(
        "score".into(),
        MathOperation::Mul,
        5.0,
    ));
    script.connect(event, set_var);
    script.connect(set_var, math);

    let result = execute_script(&script, "Start", &HashMap::new());
    assert_eq!(result.variables.get("score"), Some(&50.0));
}

#[test]
fn test_branch_node() {
    let mut script = VisualScript::new("branch_test");
    let event = script.add_node(ScriptNode::Event("Check".into()));
    let cond = script.add_node(ScriptNode::Condition(
        "health".into(),
        ComparisonOp::LessThan,
        20.0,
    ));
    let action_heal = script.add_node(ScriptNode::Action("heal".into(), None));
    let action_attack = script.add_node(ScriptNode::Action("attack".into(), None));
    let branch = script.add_node(ScriptNode::Branch(action_heal, action_attack));

    // Event -> Condition -> Branch (condition feeds into branch as predecessor)
    // Branch evaluates its incoming condition to decide true/false path.
    script.connect(event, branch);
    script.connect(cond, branch); // condition as input to branch

    // health < 20 => heal
    let mut vars = HashMap::new();
    vars.insert("health".into(), 10.0);
    let result = execute_script(&script, "Check", &vars);
    assert_eq!(result.triggered_actions.len(), 1);
    assert_eq!(result.triggered_actions[0].0, "heal");

    // health >= 20 => attack
    vars.insert("health".into(), 50.0);
    let result = execute_script(&script, "Check", &vars);
    assert_eq!(result.triggered_actions.len(), 1);
    assert_eq!(result.triggered_actions[0].0, "attack");
}

#[test]
fn test_compile_to_json_and_restore() {
    let mut script = VisualScript::new("serialization_test");
    let event = script.add_node(ScriptNode::Event("BuildingPlaced".into()));
    let cond = script.add_node(ScriptNode::Condition(
        "population".into(),
        ComparisonOp::GreaterThan,
        100.0,
    ));
    let action = script.add_node(ScriptNode::Action(
        "spawn_celebration".into(),
        Some("fireworks".into()),
    ));
    script.connect(event, cond);
    script.connect(cond, action);

    let json = script.compile_to_json().unwrap();
    assert!(json.contains("serialization_test"));
    assert!(json.contains("BuildingPlaced"));

    let restored = VisualScript::from_json(&json).unwrap();
    assert_eq!(restored.name, "serialization_test");
    assert_eq!(restored.nodes.len(), 3);
    assert_eq!(restored.connections.len(), 2);

    // Restored script should produce the same execution result
    let mut vars = HashMap::new();
    vars.insert("population".into(), 200.0);
    let result = execute_script(&restored, "BuildingPlaced", &vars);
    assert_eq!(result.triggered_actions.len(), 1);
    assert_eq!(result.triggered_actions[0].0, "spawn_celebration");
}

#[test]
fn test_multiple_actions_chained() {
    let mut script = VisualScript::new("chain");
    let event = script.add_node(ScriptNode::Event("Go".into()));
    let a1 = script.add_node(ScriptNode::Action("step1".into(), None));
    let a2 = script.add_node(ScriptNode::Action("step2".into(), None));
    let a3 = script.add_node(ScriptNode::Action("step3".into(), None));
    script.connect(event, a1);
    script.connect(a1, a2);
    script.connect(a2, a3);

    let result = execute_script(&script, "Go", &HashMap::new());
    assert_eq!(result.triggered_actions.len(), 3);
    assert_eq!(result.triggered_actions[0].0, "step1");
    assert_eq!(result.triggered_actions[1].0, "step2");
    assert_eq!(result.triggered_actions[2].0, "step3");
}
