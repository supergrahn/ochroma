use vox_script::{GameEvent, ScriptRuntime};

#[test]
fn subscribe_and_dispatch() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("test_mod", &[]).unwrap();
    rt.subscribe("test_mod", "BuildingPlaced");

    let handlers = rt.dispatch_event(&GameEvent::BuildingPlaced {
        position: [0.0, 0.0, 0.0],
        asset_id: "house".into(),
    });
    assert_eq!(handlers, vec!["test_mod"]);
}

#[test]
fn wildcard_subscription() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("logger", &[]).unwrap();
    rt.subscribe("logger", "*");

    let handlers = rt.dispatch_event(&GameEvent::CitizenBorn { citizen_id: 1 });
    assert!(handlers.contains(&"logger".to_string()));
}

#[test]
fn unmatched_event_no_handlers() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("test_mod", &[]).unwrap();
    rt.subscribe("test_mod", "BuildingPlaced");

    let handlers = rt.dispatch_event(&GameEvent::BudgetTick { funds: 1000.0 });
    assert!(handlers.is_empty());
}
