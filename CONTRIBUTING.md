# Contributing to Ochroma

## Dev Setup

```bash
git clone https://github.com/ochroma-engine/ochroma
cd ochroma
cargo build
cargo test --workspace
```

Linux extra deps: `sudo apt install libasound2-dev libvulkan-dev`

## Architecture

The engine is split into domain crates. The rule: **engine crates must never contain game-specific concepts** (buildings, NPCs, quests). Game logic lives in `vox_app` or `vox_sim`.

| Crate | Purpose |
|-------|---------|
| `vox_core` | Shared types, math, ECS, character controller |
| `vox_render` | GPU rendering, spectral pipeline, post-processing |
| `vox_physics` | Rapier physics, fluids, destruction |
| `vox_audio` | CPAL backend, HRTF, adaptive music |
| `vox_script` | Lua 5.4 scripting runtime |
| `vox_data` | Asset formats, GLTF import, scene serialisation |
| `vox_net` | QUIC networking, rollback netcode |
| `vox_terrain` | Heightmap and SDF terrain |
| `vox_ui` | Game UI via Vello/Taffy |
| `vox_sim` | Simulation systems (city, ecosystems) |
| `vox_nn` | AI/LLM integration via candle |
| `ochroma_engine` | Public façade crate |

## The Spectral Invariant

Every system that touches a `GaussianSplat` **must** preserve or intentionally modify its spectral energy field (16 bands, 380–755 nm). Never zero it out as a shortcut. The spectral data drives both rendering (physically-based spectral tonemapping) and audio (spectral-to-frequency mapping).

## Spectral Band Reference

Ochroma uses 16 spectral bands mapping wavelength → visual/audio frequency:

| Band | Wavelength | Approx Color | Audio |
|------|-----------|--------------|-------|
| 0 | 380–405nm | Violet | 8 kHz |
| 1 | 405–430nm | Violet-blue | 6 kHz |
| 2 | 430–455nm | Blue-violet | 4 kHz |
| 3 | 455–480nm | Blue | 3 kHz |
| 4 | 480–505nm | Blue-cyan | 2 kHz |
| 5 | 505–530nm | Cyan-green | 1.5 kHz |
| 6 | 530–555nm | Green | 1 kHz |
| 7 | 555–580nm | Yellow-green | 800 Hz |
| 8 | 580–605nm | Yellow | 600 Hz |
| 9 | 605–630nm | Orange | 400 Hz |
| 10 | 630–655nm | Red-orange | 300 Hz |
| 11 | 655–680nm | Red | 200 Hz |
| 12 | 680–705nm | Deep red | 150 Hz |
| 13 | 705–730nm | Near-IR | 100 Hz |
| 14 | 730–755nm | Near-IR | 80 Hz |
| 15 | 755nm+ | IR | 60 Hz |

## GaussianSplat Construction Rules

- Always use `GaussianSplat::surface(...)` or `GaussianSplat::volume(...)` — never struct literals
- Access fields via accessor methods only — never direct field access
- Spectral data is `[u16; 16]` normalized to 0–65535

## Before Submitting a PR

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` is clean
- [ ] New public APIs have `///` doc comments with a usage example
- [ ] New systems respect the spectral invariant
- [ ] Commit messages follow: `type(scope): description` (e.g. `feat(render): add DOF pass`)

## PR Size

Keep PRs focused. One domain, one feature, one bug fix per PR.

## Getting Started

See [docs/getting-started.md](docs/getting-started.md) for setup and first-run instructions.
