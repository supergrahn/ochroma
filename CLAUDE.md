# Ochroma Engine

Spectral Gaussian Splatting game engine.

## Build

```bash
cargo build
cargo test
```

## Architecture

- `vox_core` — shared types, math, spectral definitions (ENGINE — game-agnostic)
- `vox_data` — .vxm file format, asset I/O (ENGINE — game-agnostic)
- `vox_render` — GPU rendering, spectral pipeline (ENGINE — game-agnostic)
- `vox_app` — application binary, UI (GAME layer)

**Rule: Engine crates must NEVER contain game-specific concepts (buildings, zoning, traffic). Game logic belongs in vox_app or vox_sim.**

## Specs

See `docs/spec/` for phase specifications.
