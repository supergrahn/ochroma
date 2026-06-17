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

## Render/material capabilities (engine side) — built, most one wire from the world

The Spectra path tracer + `vox_render` already implement far more than the live game shows. Full Forge geometry/material inventory: `../forge/CLAUDE.md`. Engine-side, verified 2026-06-17:

- **Glass/transmission** — `PbrMaterial.transmission/ior` (`vox_render/src/splat_backend.rs:180`); the Glass material channel → `transmission 0.9, ior 1.5, roughness 0.05` (`spectra_frame.rs:267`), packed to `MAT_GLASS` → `brdf_glass.slang` (auto bounce-bump to 8). Real semi-transparent glass.
- **Normal + roughness + POM/cone-step relief** — fully in the megakernel (`spectra/slang/megakernel.slang:2172-2295`) + packer (`splat_backend.rs:2974-2989`); proven by the still-path binary. **Structural choke point: `HybridMesh` (`vox_render/src/hybrid_compose.rs:62`) carries ONLY `albedo_tex_path`** — no normal/roughness/displacement, so 46 on-disk PolyHaven normal/rough maps are unreachable live. Fix = add 3 `Option<String>` fields + collect them.
- **7-channel weathering** (`apply_weathering_full`, `megakernel.slang:2332`) — cooked per-vertex into `ReadyAssetMesh.weathering_masks`; **no setter on `ResidentCityRenderer`** so the live path drops them.
- **Hero-wavelength spectral** (16-band) — plumbed but `resident_renderer.rs:208` forces `SpectralMode::Single` (dodges a black-buildings CUDA bug).
- **Emission / lit windows** (`MAT_GLASS_LIT` id 7), advanced BSDF lobes (clearcoat/sheen/leaf-translucency, MaterialX), À-Trous denoise, Bruneton atmosphere + celestial key light, DaylitCity tonemap — all real; mostly unrequested by the game.

**Rule:** a SOTA render needs ZERO new render tech — only wiring (drive Forge's directive path → real geometry+zones; widen the `HybridMesh` texture seam; route cooked `material_zones` to the renderer). Don't validate render/content on box-stub scenes — the witness is a hero-camera frame.
