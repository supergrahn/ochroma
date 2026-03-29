# Ochroma Demo Scene

Run with:

    cargo run --bin ochroma -- --demo

Demonstrates:

- Scripted entity via Rhai (`main.rhai`)
- Hot-reload: edit `main.rhai` while running; changes apply within 1 second
- Log output from script via the `log()` API
- Foundation for Sprint 5: add WGSL shader files here for GPU hot-reload

## Hot-reload workflow

1. Start the engine: `cargo run --bin ochroma -- --demo`
2. Open `examples/demo_scene/main.rhai` in your editor
3. Change the `on_update` body (e.g., change the log message)
4. Save the file — the engine reloads automatically within 1 second
5. Observe the new log output in the console
