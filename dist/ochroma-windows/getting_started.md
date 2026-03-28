# Getting Started with Ochroma Engine

## Prerequisites

- Rust 1.85+ (`rustup update`)
- A GPU with Vulkan or Metal support (wgpu handles the backend)
- Windows 10+ or Linux (WSL2 works)
- On Linux: `libasound2-dev` for audio (`sudo apt-get install libasound2-dev`)

## Building

```bash
git clone https://github.com/supergrahn/ochroma.git
cd ochroma
cargo build --release
```

## Running the Binaries

```bash
# Full engine with all systems: spectral rendering, CLAS, lighting, physics, scripts
cargo run --bin ochroma

# Load a .ply Gaussian splat file
cargo run --bin ochroma -- path/to/scene.ply

# Load a saved map file
cargo run --bin ochroma -- level.ochroma_map

# Walking simulator: first game built on the engine (collect 10 orbs to win)
cargo run --bin walking_sim

# Interactive demo with editor overlay
cargo run --bin demo

# Exercise all 76 engine modules (headless, saves render_showcase_output.ppm)
cargo run --bin render_showcase
```

## Controls

### `ochroma` and `demo`

| Key | Action |
|-----|--------|
| WASD | Move camera |
| Space / Left Shift | Move up / down |
| Right-click + drag | Look around |
| Left-click | Place a tree at cursor |
| T | Advance time of day (+1 hour, wraps at 24) |
| +/- | Adjust exposure |
| M | Cycle tone mapping (Linear/ACES/Reinhard/Filmic) |
| Q | Cycle DLSS quality (Off/Quality/Balanced/Performance/Ultra Perf) |
| G | Toggle frame generation flag |
| P | Toggle fast/spectral render mode (ochroma only) |
| Tab | Toggle scene editor overlay |
| Ctrl+S | Save scene to `.ochroma_map` (JSON format) |
| F12 | Save screenshot as PPM to temp directory |
| Escape | Quit |

### `walking_sim`

| Key | Action |
|-----|--------|
| WASD | Move player |
| Right-click + drag | Look around |
| `` ` `` | Run a test Rhai expression in the console |
| Escape | Quit |

## Writing Game Scripts

Implement the `GameScript` trait from `vox_core::script_interface`:

```rust
use vox_core::script_interface::{GameScript, ScriptContext};

struct MyPlayerScript {
    speed: f32,
}

impl GameScript for MyPlayerScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        ctx.log("Player ready!");
    }

    fn on_update(&mut self, ctx: &mut ScriptContext, dt: f32) {
        // Move forward each frame
        ctx.set_position([0.0, 1.0, -self.speed * dt]);
    }

    fn name(&self) -> &str { "MyPlayerScript" }
}
```

The full trait also provides `on_destroy` and `on_collision` callbacks.

## Creating a Map File

```rust
use vox_data::map_file::MapFile;
use std::path::Path;

let mut map = MapFile::new("My Level");
map.place_object("Player", "player.ply", [0.0, 1.0, 0.0]);
map.place_object("House", "house.ply", [10.0, 0.0, -5.0]);
map.add_light("point", [5.0, 3.0, 0.0], [1.0, 0.9, 0.8], 50.0);
map.save(&Path::new("my_level.ochroma_map")).unwrap();
```

Map files are saved as JSON (`.ochroma_map`). Load them with:

```bash
cargo run --bin ochroma -- my_level.ochroma_map
```

## Registering Scripts

```rust
use vox_core::script_interface::ScriptRegistry;

let mut registry = ScriptRegistry::new();
registry.register("MyPlayerScript", || Box::new(MyPlayerScript { speed: 5.0 }));

// Create a script instance by name
let script = registry.create("MyPlayerScript");
```

## Architecture Overview

Ochroma is a Spectral Gaussian Splatting engine. Key concepts:

- **Gaussian Splats**: The rendering primitive. Volumetric Gaussian distributions rendered from .ply files (standard 3DGS format).
- **Spectral Rendering**: Materials store 8-band spectral reflectance (380-660nm). Illuminant varies with time of day.
- **Volumetric Terrain**: SDF-based terrain with overhangs and caves.
- **Bevy ECS**: Entity-Component-System architecture via `bevy_ecs`. Entities have components, systems process them.
- **GameScript**: Your game logic. Implement the `GameScript` trait and attach to entities.
- **Rhai Scripting**: Hot-reloadable scripts using the Rhai embedded scripting language.

## Project Structure

```
your_game/
├── assets/
│   ├── characters/player.ply
│   ├── buildings/house.ply
│   └── sounds/jump.wav
├── maps/
│   └── level_1.ochroma_map
├── src/
│   ├── main.rs          # Entry point
│   ├── player.rs        # Player script (implements GameScript)
│   └── enemies.rs       # Enemy scripts
└── Cargo.toml
```

## Next Steps

- [Example: Simple Game](../crates/vox_app/examples/simple_game.rs) — demonstrates GameScript, Collectible, and TriggerZone
- [Example: City Builder](../crates/vox_app/examples/city_builder_demo.rs)
- [Engine Core Specification](spec/engine-core-spec.md)
