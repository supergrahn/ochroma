# Ochroma Engine

Spectral Gaussian Splatting game engine.

## Why this exists (product north-star)

The engine exists to ship a flagship game — **Urban Horizon**, a care-first city
builder (`../../Ochroma/projects/urban_horizon`, see its `CLAUDE.md` + `README.md` for
the full **"why people will buy this"**). That value proposition is validated by both a
SOTA audit and player-demand research and is the engine's mandate. The engine's job is to
make these real:

- **A REAL simulation at scale** — ~1M agents, deterministic, so failures propagate and
  decisions have consequences (players are tired of faked/ghost sims).
- **Beauty WITHOUT breaking** — the Spectra path tracer is the ONLY renderer, but it ships
  behind a **hard real-time floor + fidelity ladder** (Performance/Balanced/Beauty; PT never
  the only mode; the Performance tier holds ≥30fps on the AMD 780M). Performance is a SHIP GATE.
- **Determinism as a moat** — replay-exact sim enables shareable replays + what-if rewind,
  which no competitor has. Protect it: id-sorted folds, no HashMap/RNG iteration-order, fixed-ε f64.
- **Unique, authorable content** — node-DAG + LLM-authorable geometry (Forge).

**No self-deception:** every claim is held to a measured, in-the-live-path witness — never a
passing unit test, a population-independent bench, or a memory note. The roadmap that drives
the work is the game's `docs/superpowers/plans/2026-06-15-sota-roadmap.md`.

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
