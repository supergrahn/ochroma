# City Builder Demo

This is a **game** built on the Ochroma engine, not part of the engine itself.

## Architecture

- **Ochroma Engine** (`vox_core`, `vox_render`, `vox_data`, `vox_audio`, etc.) -- game-agnostic infrastructure
- **vox_sim** -- a game-specific simulation library for city-building mechanics
- **City Builder Demo** -- a complete game that uses both the engine and vox_sim

The engine knows nothing about buildings, citizens, zoning, or traffic.
All city-building concepts live in `vox_sim` and this demo.

## Running

```bash
cargo run --example city_builder_demo -p vox_app
```

This initializes all 31 vox_sim subsystems, builds a small city, runs 100
simulation ticks, and prints stats from every system.
