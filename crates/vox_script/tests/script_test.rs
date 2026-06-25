use vox_script::{ScriptEvent, ScriptRuntime};

#[test]
fn subscribe_and_dispatch() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("test_mod", &[]).unwrap();
    rt.subscribe("test_mod", "BuildingPlaced");

    let handlers = rt.dispatch_event(&ScriptEvent::new("BuildingPlaced", vec![]));
    assert_eq!(handlers, vec!["test_mod"]);
}

#[test]
fn wildcard_subscription() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("logger", &[]).unwrap();
    rt.subscribe("logger", "*");

    let handlers = rt.dispatch_event(&ScriptEvent::named("CitizenBorn"));
    assert!(handlers.contains(&"logger".to_string()));
}

#[test]
fn unmatched_event_no_handlers() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("test_mod", &[]).unwrap();
    rt.subscribe("test_mod", "BuildingPlaced");

    let handlers = rt.dispatch_event(&ScriptEvent::named("BudgetTick"));
    assert!(handlers.is_empty());
}
