//! Example game built on the Ochroma Engine.
//!
//! This demonstrates how a game developer uses the engine:
//! 1. Define game scripts (logic that runs each frame)
//! 2. Create a scene (place objects in the world)
//! 3. Handle input
//! 4. Run!
//!
//! To run: cargo run --example simple_game -p vox_app

// This is what a game developer writes — not engine code.

use vox_core::script_interface::{GameScript, ScriptContext, ScriptCommand};

/// A simple player controller script.
/// Attach this to the player entity — it handles movement each frame.
struct PlayerController {
    speed: f32,
    _jump_force: f32,
    _grounded: bool,
}

impl PlayerController {
    fn new() -> Self {
        Self { speed: 10.0, _jump_force: 8.0, _grounded: true }
    }
}

impl GameScript for PlayerController {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        ctx.log("Player spawned!");
    }

    fn on_update(&mut self, ctx: &mut ScriptContext, dt: f32) {
        // Game logic: move forward continuously (in a real game, check input)
        let new_z = -self.speed * dt;
        ctx.set_position([0.0, 1.0, new_z]);
    }

    fn on_collision(&mut self, ctx: &mut ScriptContext, other: u32) {
        ctx.log(&format!("Player hit entity {}", other));
        ctx.play_sound("hit.wav", 0.8);
    }

    fn name(&self) -> &str { "PlayerController" }
}

/// A collectible item that spins and plays a sound when collected.
struct Collectible {
    rotation: f32,
    collected: bool,
}

impl Collectible {
    fn new() -> Self { Self { rotation: 0.0, collected: false } }
}

impl GameScript for Collectible {
    fn on_update(&mut self, ctx: &mut ScriptContext, dt: f32) {
        if self.collected { return; }
        // Spin slowly
        self.rotation += dt * 2.0;
        ctx.commands.push(ScriptCommand::SetRotation {
            rotation: [0.0, self.rotation.sin() * 0.5, 0.0, self.rotation.cos()],
        });
    }

    fn on_collision(&mut self, ctx: &mut ScriptContext, _other: u32) {
        if !self.collected {
            self.collected = true;
            ctx.play_sound("collect.wav", 1.0);
            ctx.log("Item collected!");
            ctx.destroy(ctx.entity_id);
        }
    }

    fn name(&self) -> &str { "Collectible" }
}

/// A trigger zone that spawns enemies when the player enters.
struct SpawnTrigger {
    triggered: bool,
    spawn_count: u32,
}

impl SpawnTrigger {
    fn new(count: u32) -> Self { Self { triggered: false, spawn_count: count } }
}

impl GameScript for SpawnTrigger {
    fn on_collision(&mut self, ctx: &mut ScriptContext, _other: u32) {
        if !self.triggered {
            self.triggered = true;
            ctx.log(&format!("Trigger activated! Spawning {} enemies", self.spawn_count));
            for i in 0..self.spawn_count {
                ctx.spawn("enemies/zombie.ply", [i as f32 * 3.0, 0.0, -20.0]);
            }
            ctx.play_sound("alarm.wav", 1.0);
        }
    }

    fn name(&self) -> &str { "SpawnTrigger" }
}

// ====== How to set up the game ======

fn main() {
    println!("=== Ochroma Engine -- Simple Game Example ===");
    println!();
    println!("This example shows how to build a game on the Ochroma engine.");
    println!();
    println!("In a real game, you would:");
    println!("  1. Create .ply assets (train Gaussian splats from photos or generate procedurally)");
    println!("  2. Create a .ochroma_map file placing assets in the world");
    println!("  3. Write GameScript implementations for game logic");
    println!("  4. Register scripts with the ScriptRegistry");
    println!("  5. Run the engine with your map");
    println!();

    // Register game scripts
    let mut registry = vox_core::script_interface::ScriptRegistry::new();
    registry.register("PlayerController", || Box::new(PlayerController::new()));
    registry.register("Collectible", || Box::new(Collectible::new()));
    registry.register("SpawnTrigger", || Box::new(SpawnTrigger::new(3)));

    println!("Registered scripts: {:?}", registry.registered_scripts());

    // Create a scene programmatically (normally loaded from .ochroma_map)
    let mut map = vox_data::map_file::MapFile::new("Example Level");
    map.description = "A simple game level demonstrating Ochroma engine features".into();

    // Place objects
    map.place_object("Player", "characters/player.ply", [0.0, 1.0, 0.0]);
    map.place_object("House", "buildings/house.ply", [10.0, 0.0, -5.0]);
    map.place_object("Tree", "trees/oak.ply", [5.0, 0.0, -10.0]);
    map.place_object("Coin", "items/coin.ply", [3.0, 1.0, -5.0]);

    // Add lights
    map.add_light("directional", [0.0, 100.0, 0.0], [1.0, 0.95, 0.9], 1.0);
    map.add_light("point", [10.0, 3.0, -5.0], [1.0, 0.8, 0.5], 30.0);

    // Attach scripts to entities
    map.placed_objects[0].scripts.push("PlayerController".into());
    map.placed_objects[3].scripts.push("Collectible".into());

    println!();
    println!("Scene: {} objects, {} lights", map.object_count(), map.light_count());

    // Save the map
    let map_path = std::env::temp_dir().join("ochroma_example.ochroma_map");
    map.save(&map_path).unwrap();
    println!("Map saved to: {}", map_path.display());

    // In a real game, you'd now run:
    //   cargo run --bin demo -- path/to/your/map.ochroma_map
    // Or use the engine API directly to load and run the scene.

    // Demonstrate script execution
    println!();
    println!("--- Script Execution Demo ---");

    let mut player = registry.create("PlayerController").unwrap();
    let mut ctx = vox_core::script_interface::ScriptContext::new(0);

    player.on_start(&mut ctx);
    for cmd in ctx.take_commands() {
        println!("  Command: {:?}", cmd);
    }

    for frame in 0..5 {
        player.on_update(&mut ctx, 0.016);
        let cmds = ctx.take_commands();
        if !cmds.is_empty() {
            println!("  Frame {}: {} commands", frame, cmds.len());
        }
    }

    println!();
    println!("=== Game setup complete! ===");
    println!("To play: cargo run --bin demo");
}
