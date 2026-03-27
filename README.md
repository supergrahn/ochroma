# Ochroma Engine

**Spectral Gaussian Splatting Game Engine**

Ochroma is a next-generation game engine built on 3D Gaussian Splatting with physically correct spectral rendering. It renders cinematic-quality scenes at real-time framerates.

## What Makes Ochroma Different

- **Spectral Rendering**: 8-band spectral reflectance (380-660nm), not RGB approximation. Materials respond correctly to any illuminant -- time of day, weather, and artificial lighting are physically accurate.
- **Gaussian Splat Native**: No triangles, no UV maps, no texture artists. Load .ply files from any 3DGS training tool and render them directly.
- **Volumetric Terrain**: SDF-based terrain with overhangs, caves, arches -- geometry that heightmap engines cannot represent.
- **NVIDIA Next-Gen Ready**: CLAS clustering, MegaGeometry dispatch, DLSS pipeline built for Blackwell architecture.
- **AI-Native Asset Pipeline**: Procedural generation from rules, LLM-driven layout, text-to-city generation.

## Quick Start

```bash
# Run the interactive demo
cargo run --bin demo

# Load a .ply Gaussian splat file
cargo run --bin demo -- path/to/scene.ply

# Run the example game
cargo run --example simple_game -p vox_app
```

## Controls (Demo)

| Key | Action |
|-----|--------|
| WASD | Move camera |
| Space / Shift | Up / Down |
| Right-click + drag | Look around |
| Left-click | Place a tree |
| T | Advance time of day |
| +/- | Adjust exposure |
| M | Cycle tone mapping (ACES/Reinhard/Filmic) |
| Q | Cycle DLSS quality (Off/Quality/Balanced/Performance/Ultra) |
| G | Toggle frame generation |
| F12 | Screenshot |
| Escape | Quit |

## Architecture

```
ochroma_engine (SDK crate)
├── vox_core     -- Types, ECS, math, spectral, input, scripting API
├── vox_render   -- GPU rendering, spectral pipeline, DLSS, CLAS, particles
├── vox_data     -- Asset formats (.vxm, .ply), procedural generation, maps
├── vox_terrain  -- Volumetric SDF terrain, heightmaps, foliage
├── vox_sim      -- Game simulation (optional, for city builders etc.)
├── vox_audio    -- Spatial audio with acoustic ray tracing
├── vox_physics  -- Rigid body physics with AABB collision
├── vox_net      -- Multiplayer networking with CRDT replication
├── vox_script   -- Wasm mod runtime with visual scripting
├── vox_nn       -- AI/ML systems, LLM integration, procedural generation
├── vox_ui       -- UI framework
└── vox_tools    -- CLI tools (asset pipeline, build system)
```

## Rendering Pipeline

```
Scene -> Spectral Framebuffer (8 bands + depth + normals + motion)
      -> Temporal Accumulation (denoise via history blending)
      -> Spectral Tone Mapper (ACES / Reinhard / Filmic)
      -> DLSS Super Resolution (render at quarter res, AI upscale to 4K)
      -> Display
```

## Building a Game

See `examples/simple_game/` for a complete example.

1. Create assets (.ply files from 3DGS training or procedural generators)
2. Create a map file (.ochroma_map) placing assets in the world
3. Write game scripts implementing `GameScript` trait
4. Run with `cargo run --bin demo -- your_map.ochroma_map`

## License

MIT
