# Getting Started with Ochroma Engine

## Prerequisites

- Rust 1.85+ (`rustup update`)
- Vulkan-capable GPU (NVIDIA recommended, AMD/Intel work via wgpu)
- Windows 10+ or Linux (WSL2 works!)

## Building

```bash
git clone https://github.com/supergrahn/ochroma.git
cd ochroma
cargo build --release
```

## Running the Demo

```bash
# Interactive demo with default scene (terrain + buildings + trees)
cargo run --bin demo --release

# Load a .ply Gaussian splat file
cargo run --bin demo --release -- path/to/scene.ply
```

## Controls

| Key | Action |
|-----|--------|
| WASD | Move camera |
| Space / Shift | Up / Down |
| Right-click + drag | Look around |
| Left-click | Place object |
| T | Advance time of day |
| +/- | Adjust exposure |
| M | Cycle tone mapping |
| Q | Cycle DLSS quality |
| G | Toggle frame generation |
| F12 | Screenshot |
| Escape | Quit |

## Creating Your First Game

### Step 1: Define a Game Script

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

### Step 2: Create a Map

```rust
use vox_data::map_file::MapFile;

let mut map = MapFile::new("My Level");
map.place_object("Player", "player.ply", [0.0, 1.0, 0.0]);
map.place_object("House", "house.ply", [10.0, 0.0, -5.0]);
map.add_light("point", [5.0, 3.0, 0.0], [1.0, 0.9, 0.8], 50.0);
map.save(&Path::new("my_level.ochroma_map")).unwrap();
```

### Step 3: Register Scripts and Run

```rust
use vox_core::script_interface::ScriptRegistry;

let mut registry = ScriptRegistry::new();
registry.register("MyPlayerScript", || Box::new(MyPlayerScript { speed: 5.0 }));

// Load map and run
// cargo run --bin demo -- my_level.ochroma_map
```

## Architecture Overview

Ochroma is a Spectral Gaussian Splatting engine. Key concepts:

- **Gaussian Splats**: The rendering primitive. Instead of triangles, we render volumetric Gaussian distributions. Load from .ply files (standard 3DGS format).
- **Spectral Rendering**: Materials store 8-band spectral reflectance (380-660nm). Lighting is physically correct under any illuminant.
- **Volumetric Terrain**: SDF-based terrain with overhangs, caves, and arches -- geometry that heightmap engines can't represent.
- **ECS**: Entity-Component-System architecture (Bevy ECS). Entities have components, systems process them.
- **GameScript**: Your game logic. Implement the `GameScript` trait and attach to entities.

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
│   ├── player.rs        # Player script
│   └── enemies.rs       # Enemy scripts
└── Cargo.toml
```

## Next Steps

- [Example: Simple Game](../examples/simple_game/)
- [Engine Core Specification](spec/engine-core-spec.md)
- [Rendering Pipeline](spec/rendering-pipeline-spec.md)
