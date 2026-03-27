# Ochroma Engine

**Spectral Gaussian Splatting Game Engine**

Ochroma is a game engine built on 3D Gaussian Splatting with spectral rendering. It renders scenes using volumetric Gaussian splats with physically-modelled spectral reflectance.

## What Works Today

- **Spectral Rendering**: 8-band spectral reflectance (380-660nm). Materials respond to any illuminant — time of day cycles from warm sunrise through noon to cool moonlight.
- **Gaussian Splat Native**: Load .ply files from any 3DGS training tool and render them directly.
- **Procedural Scene Generation**: Terrain, buildings, and trees generated from scratch — no external assets required.
- **CLAS Clustering**: Splat clustering and BVH acceleration structure (software implementation).
- **Volumetric Terrain**: SDF-based terrain with overhangs and caves.
- **Rhai Scripting**: Embedded scripting via the Rhai engine, used for game config and debug console.
- **Physics**: AABB collision (built-in) and Rapier rigid body physics (optional feature).
- **Spatial Audio**: Distance-attenuation audio synthesis; WAV output on Linux when `audio-backend` feature is enabled.
- **Walking Simulator**: A complete first game built on the engine — collect 10 orbs to win.

## Quick Start

```bash
# Run the full engine (default scene)
cargo run --bin ochroma

# Load a .ply Gaussian splat file
cargo run --bin ochroma -- path/to/scene.ply

# Run the walking simulator game
cargo run --bin walking_sim

# Run the interactive demo
cargo run --bin demo

# Exercise all 76 engine modules (outputs render_showcase_output.ppm)
cargo run --bin render_showcase

# Run the simple_game example (demonstrates the GameScript API)
cargo run --example simple_game -p vox_app
```

## Controls

### `ochroma` and `demo` binaries

| Key | Action |
|-----|--------|
| WASD | Move camera |
| Space / Left Shift | Move up / down |
| Right-click + drag | Look around |
| Left-click | Place a tree at cursor |
| T | Advance time of day (+1 hour) |
| +/- | Adjust exposure |
| M | Cycle tone mapping (Linear/ACES/Reinhard/Filmic) |
| Q | Cycle DLSS quality mode (Off/Quality/Balanced/Performance/Ultra) |
| G | Toggle frame generation flag |
| P | Toggle fast/spectral render mode (`ochroma` only) |
| Tab | Toggle scene editor overlay |
| Ctrl+S | Save scene to `.ochroma_map` file |
| Delete | Delete selected entity in editor |
| Arrow keys | Move selected entity in editor |
| F12 | Save screenshot as PPM to temp directory |
| Escape | Quit |

### `walking_sim` binary

| Key | Action |
|-----|--------|
| WASD | Move player |
| Right-click + drag | Look around |
| `` ` `` | Evaluate a test Rhai expression in console |
| Escape | Quit |

## Architecture

```
ochroma_engine (SDK crate)
├── vox_core     -- Types, ECS (Bevy ECS), math, spectral, input, scripting API
├── vox_render   -- Spectral pipeline, DLSS (software), CLAS, particles, lighting
├── vox_data     -- Asset formats (.vxm, .ply loader), procedural generation, maps
├── vox_terrain  -- Volumetric SDF terrain, heightmaps, foliage
├── vox_sim      -- Game simulation systems
├── vox_audio    -- Spatial audio with distance attenuation, WAV synthesis
├── vox_physics  -- AABB physics (built-in) + Rapier rigid body (optional feature)
├── vox_net      -- Multiplayer networking with CRDT replication (in development)
├── vox_script   -- Rhai scripting runtime, visual scripting, plugin system
├── vox_nn       -- LLM integration, procedural city generation (in development)
├── vox_ui       -- UI framework
└── vox_tools    -- CLI asset pipeline tool (turnaround capture, GLTF import)
```

## Rendering Pipeline

```
Scene splats -> Software Rasteriser (CPU)
             -> Spectral Framebuffer (8 bands + depth + normals)
             -> Temporal Accumulation (history blending)
             -> Spectral Tone Mapper (Linear / ACES / Reinhard / Filmic)
             -> DLSS quality mode (software upscaling pipeline)
             -> Display via wgpu
```

Note: GPU hardware rasterisation (`GpuRasteriser`) and hardware DLSS require
an NVIDIA GPU with appropriate driver support. The engine falls back to the
software rasteriser automatically.

## Building a Game

See `examples/simple_game/` in the `vox_app` crate for a complete example.

1. Implement the `GameScript` trait from `vox_core::script_interface`
2. Create a map file using `vox_data::map_file::MapFile`
3. Register scripts with `ScriptRegistry`
4. Load the map and run with `cargo run --bin ochroma -- your_map.ochroma_map`

## License

MIT
