# Simple Game -- Ochroma Engine Example

This example shows how to build a game on the Ochroma engine.

## How It Works

1. **Assets**: Gaussian splat files (.ply) trained from photos or generated procedurally
2. **Map**: A .ochroma_map file defining where objects are placed
3. **Scripts**: Rust structs implementing `GameScript` for game logic
4. **Run**: `cargo run --bin demo` to play

## Game Scripts

```rust
struct MyScript;

impl GameScript for MyScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        ctx.log("Hello from my script!");
    }

    fn on_update(&mut self, ctx: &mut ScriptContext, dt: f32) {
        // Game logic runs here every frame
    }

    fn name(&self) -> &str { "MyScript" }
}
```

## Creating Assets

Option 1: Train from photos using any 3DGS tool, then export .ply
Option 2: Use Ochroma's procedural generators
Option 3: Download from the asset marketplace

## Running

```bash
cargo run --example simple_game -p vox_app  # Run this example
cargo run --bin demo                         # Default scene
cargo run --bin demo -- my_scene.ply         # Load a .ply file
```
