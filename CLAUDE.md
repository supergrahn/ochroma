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

## Plans and Design Docs

**Every new plan must use `docs/templates/plan.md` as its base.**
**Every new design doc must use `docs/templates/design.md` as its base.**

Key rules enforced by the templates:
- `Done When` must name an exact command and exact human-visible output — "tests pass" is never acceptable
- Every task implements AND wires in the same step — no "wire later" tasks
- `todo!()` / `unimplemented!()` / empty function bodies = task failure
- Every test checks a real computed outcome — `assert!(result.is_some())` is forbidden
- `IMPORTANT NOTES` must contain real API signatures so agents don't invent their own

Plans go in `docs/superpowers/plans/`. Design docs go in `docs/superpowers/specs/`.
