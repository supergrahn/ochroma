use vox_core::script_interface::*;

struct TestScript {
    update_count: u32,
}

impl GameScript for TestScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        ctx.log("TestScript started");
    }

    fn on_update(&mut self, ctx: &mut ScriptContext, _dt: f32) {
        self.update_count += 1;
        if self.update_count == 10 {
            ctx.spawn("cube.ply", [5.0, 0.0, 0.0]);
        }
    }

    fn on_destroy(&mut self, ctx: &mut ScriptContext) {
        ctx.log("TestScript destroyed");
    }

    fn name(&self) -> &str { "TestScript" }
}

#[test]
fn script_lifecycle() {
    let mut script: Box<dyn GameScript> = Box::new(TestScript { update_count: 0 });
    let mut ctx = ScriptContext::new(0);

    script.on_start(&mut ctx);
    let cmds = ctx.take_commands();
    assert_eq!(cmds.len(), 1);
    match &cmds[0] { ScriptCommand::Log { message } => assert!(message.contains("started")), _ => panic!() }

    // Run 10 updates
    for _ in 0..10 {
        script.on_update(&mut ctx, 0.016);
    }
    let cmds = ctx.take_commands();
    // Should have a Spawn command from update 10
    assert!(cmds.iter().any(|c| matches!(c, ScriptCommand::Spawn { .. })));
}

#[test]
fn script_registry() {
    let mut registry = ScriptRegistry::new();
    registry.register("TestScript", || Box::new(TestScript { update_count: 0 }));

    let scripts = registry.registered_scripts();
    assert!(scripts.contains(&"TestScript"));

    let script = registry.create("TestScript");
    assert!(script.is_some());
    assert_eq!(script.unwrap().name(), "TestScript");
}

#[test]
fn unknown_script_returns_none() {
    let registry = ScriptRegistry::new();
    assert!(registry.create("NonExistent").is_none());
}

#[test]
fn script_context_commands() {
    let mut ctx = ScriptContext::new(42);
    assert_eq!(ctx.entity_id, 42);

    ctx.spawn("building.ply", [10.0, 0.0, 5.0]);
    ctx.set_position([1.0, 2.0, 3.0]);
    ctx.play_sound("explosion.wav", 0.8);

    let cmds = ctx.take_commands();
    assert_eq!(cmds.len(), 3);
    assert!(ctx.take_commands().is_empty()); // consumed
}
