# Getting Started with Ochroma

Ochroma is a spectral Gaussian splatting game engine written in Rust. This guide gets you from zero to running your first scene in under 15 minutes.

## Prerequisites

- **Rust** (stable, 1.85+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Git**
- **Linux:** `sudo apt install libasound2-dev libvulkan-dev`
- **Windows:** Visual Studio Build Tools (C++ workload)
- **Mac:** Xcode Command Line Tools (`xcode-select --install`)

## Clone and Build

```bash
git clone https://github.com/ochroma-engine/ochroma
cd ochroma
cargo build --release
```

First build takes 2–5 minutes. Subsequent builds are incremental.

## Run the Hello Splat Demo

```bash
cargo run --bin hello_splat
```

This opens a window with an orbit camera looking at the default spectral splat scene.

## Run the Walking Sim Example

```bash
cargo run --bin walking_sim
```

Walk around with WASD, collect orbs. Spectral damage effects shift the music.

## Run in the Browser

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
bash web/build.sh
python3 -m http.server 8080 --directory web/dist
# Open http://localhost:8080 in Chrome or Firefox
```

## Project Structure

```
crates/
  ochroma_engine/   — public API façade (start here)
  vox_core/         — shared types, math, ECS components
  vox_render/       — GPU rendering, spectral pipeline
  vox_app/          — binaries and game examples
  vox_physics/      — Rapier physics integration
  vox_audio/        — HRTF, reverb, adaptive music
  vox_script/       — Lua scripting runtime
  vox_data/         — asset formats, GLTF import
  vox_net/          — QUIC networking, rollback netcode
  vox_terrain/      — heightmap + volumetric terrain
  vox_ui/           — game UI (Vello/Taffy)
  vox_nn/           — AI/LLM integration
docs/
  getting-started.md — this file
```

## Writing Your First Script

Create `my_game.lua`:

```lua
function on_ready()
  scene.spawn("prefabs/player.vxm")
  audio.play("sounds/ambient.wav", vec3(0, 0, 0))
end

function on_update(dt)
  -- dt is seconds since last frame
end
```

Load it:

```bash
cargo run --bin engine_runner -- --script my_game.lua
```

## Next Steps

- Read [CONTRIBUTING.md](../CONTRIBUTING.md) for architecture overview and PR checklist
- See [docs/specs/](specs/) for engine design specifications
- Browse [crates/vox_app/src/bin/](../crates/vox_app/src/bin/) for example programs
