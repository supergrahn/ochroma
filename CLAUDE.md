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

The Spectra path tracer + `vox_render` already implement far more than the live game shows. Full Forge geometry/material inventory: `../forge/CLAUDE.md`. Engine-side, re-verified 2026-07-10 (visual audit): **the three historical choke points in earlier versions of this section are ALL FIXED — do not re-fix them**:

- **Glass/transmission** — `PbrMaterial.transmission/ior` (`vox_render/src/splat_backend.rs:180`); the Glass material channel → `transmission 0.9, ior 1.5, roughness 0.05` (`spectra_frame.rs:267`), packed to `MAT_GLASS` → `brdf_glass.slang` (auto bounce-bump to 8). Real semi-transparent glass.
- **Normal + roughness + POM/cone-step relief** — fully in the megakernel + packer, and the `HybridMesh` seam is WIDE (`vox_render/src/hybrid_compose.rs:109-120` carries normal/roughness/displacement paths). The old "albedo-only choke point" is gone.
- **7-channel weathering** — cooked masks now have live setters on `ResidentCityRenderer` (`resident_renderer.rs:782-786, 1096-1107`).
- **Hero-wavelength spectral** — the force-to-`SpectralMode::Single` is GONE; Hero4 is the default (`resident_renderer.rs:531-554`).
- **Clouds are LIVE** in the sky-miss path (`megakernel.slang:2661`) — a bare-looking sky is an art-direction gap, not missing tech. Emission / lit windows (`MAT_GLASS_LIT`), advanced BSDF lobes, denoise stack, Bruneton atmosphere + celestial, DaylitCity tonemap — all real.
- Still genuinely unwired (2026-07-10): SVT setter has zero callers; vox_aether weather beyond sky_model; sim-driven weathering intensity; vehicles; vox_audio.

**Rule:** a SOTA render needs ZERO new render tech — only wiring (drive Forge's directive path → real geometry+zones; widen the `HybridMesh` texture seam; route cooked `material_zones` to the renderer). Don't validate render/content on box-stub scenes — the witness is a hero-camera frame.

## Witness protocol + real-time pipeline + terrain (hard-won 2026-06-23)

A multi-day terrain saga was caused almost entirely by bad witnesses, a wrong target, and trusting stale "root cause" notes — NOT missing render tech. Bake these in:

- **Witness at 1 spp + DENOISE, never 128 spp beauty.** Real-time = 1 spp + denoise (you cannot accumulate at 60 fps), so that IS the shipping image — a 128-spp still flatters a look the player never sees. Witness with **SCATTER ON** (the vegetation IS the terrain's richness — bare ground always looks dead) and a **real camera** (the `--shot-map` `SHOT_HERO` fallback points *into a ridge wall* on cityless maps → use `SHOT_TERRAIN` or `OCHROMA_SHOT_DIST`/`OCHROMA_SHOT_PITCH`). A **noisy/low-res** witness = fix the denoiser/settings, **never add spp**.
- **Always `Read` the actual rendered frame before claiming a fix, and ISOLATE before declaring a root cause** (flat single material / toggle ONE variable / an AOV). This session a stale memory note and un-isolated guesses were wrong repeatedly (OptiX-denoise "running" — it wasn't; DLSS "out of date" — it was a DLL-load env; scatter "dropped" — it was tri-budget decimating canopies; striation "texturing" — it was alpine geometry). Disprove with evidence, don't relay claims.
- **Real-time reconstruction pipeline** (1 spp → photoreal 4K@60): 1 spp @ ~1080p **internal** → ReSTIR → **temporal accumulation** → denoise (**DLSS-RR** on NVIDIA / À-Trous+**SVGF** on AMD) → upscale to 4K → Frame Gen. **Never trace 4K.** The pieces mostly EXIST unwired (e.g. `temporal_reproject.slang` was authored but never registered) — "SOTA = wiring not inventing" holds for the render path too. Plan: `urban_horizon/docs/superpowers/plans/2026-06-23-realtime-1spp-4k60-reconstruction.md`.
- **Terrain = BUILDABLE city-builder topography (CS2-style: ~70% gentle buildable core + water + scenic relief at the borders), NOT dramatic alpine.** Landscape beauty shots (TrueTerrain) set the QUALITY bar (materials/lighting), not the topography — spiky alpine amplifies every artifact and is unplayable.
- **Box render mechanics** (`tomespen@tomespensin` — the only NVIDIA / RTX 4070-Ti GPU; local is AMD, no CUDA): spectra `=/mnt/c/Users/tom_e/src/spectra`, game `=/mnt/c/Users/tom_e/ochroma/projects/urban_horizon`. CUDA is **intermittent → retry renders** until `WROTE`. **Serialize build+render** (overlap = exit-101 file lock; `taskkill /F /IM play.exe` first). **Edit dev files under `/home/tom-espen/...` then rsync to the box** (editing box copies directly is the silent-no-build trap); `.slang` changes need `touch build.rs`. DLSS-RR needs `SPECTRA_NGX_DLSSD_DIR` set (see `play-rr-windows.cmd`). Full recall lives in the auto-memory; this is the floor.

## Real-time render + config LAWS (hard-won 2026-06-26 — these SUPERSEDE the `--shot-map` witness above)

A brutal session: slow, ugly frames, repeated misdiagnosis, and tuning by editing hardcoded values + rebuilding (minutes per change). The user was furious and right. Bake these in, hard:

- **THE BAR IS A REAL-TIME FRAME: ~16 ms (60 fps) / 33 ms floor (30 fps on the 780M). Seconds or minutes per frame is NOT a game.** Never present a slow frame as acceptable; never lowball the target ("seconds" is not a game either).
- **The offline `--shot-map` render is BANNED. EVER.** It traces toward a clean image (seconds-to-minutes per frame) and is NEVER representative of the game. ALL rendering + ALL witnesses go through the **REAL-TIME present path**: the game loop at 1 spp + temporal reconstruction + denoise + DLSS. Measure with `OCHROMA_FORCE_CITY=1 OCHROMA_PRESENT_BENCH_FRAMES=N` → prints per-frame ms (`[present-bench] … render_ms=… total_median=… (fps)`). Capture witness PNGs from the present frame, never the offline trace.
- **CONFIG-FIRST IS LAW. Tune via `urban_horizon/assets/config/render.ron`, NEVER by editing a hardcoded value + rebuilding.** render.ron is runtime-loaded and drives spp, the fidelity tiers (performance/balanced/beauty), internal res (`resolution_in`), bounces, lighting/celestial. Change it → sync the file → re-run → **ZERO rebuild**. Editing a hardcoded value (spectra presets like `near_realtime`, the `spectra-renderer/render_config.rs` gates) forces a ~5-min box rebuild AND is often a dead override (render.ron wins) — it wastes the user's time and is forbidden. Compile ONLY for a real code change. Remaining hardcoded render values are config-first **DEBT** to migrate into render.ron (single source of truth — what the user mandated).
- **NEVER rebuild the scene every frame.** Build the CLAS/TLAS acceleration structure ONCE and **REFIT** it (`apply_scene_delta_and_refit_ias`, `update_instance_transform`). A full `build_instanced_scene` rebuild of the 600K+ instances per frame is the ~2500 ms / 0.4 fps choke. STRUCTURAL change (instance add/remove, proto swap) → rebuild; TRANSFORM/material change (movement, growth animation) → REFIT. Bug class: bumping `scene_structure_rev` for a per-tick transform (e.g. rising-construction `y_scale`, `urban_horizon/src/instance/mod.rs`) forces a rebuild every frame — killing one such bump took the loop **~24 s → ~23 ms/frame**.
- **The reconstruction pipeline is BUILT + WIRED, not unwired** (the "never registered" notes were stale): OptiX/CLAS/TLAS RTX Mega-Geometry runs on the RT cores (software BVH on 600K instances = catastrophic 30-min frames; needs `nvoptix.dll` loadable — the driver puts it only in `C:\Windows\System32\DriverStore\FileRepository\*\nvoptix.dll`, so copy it next to `play.exe`; durable fix in `spectra-optix/src/detect.rs`), `temporal_reproject` is registered + dispatched, À-Trous/SVGF/DLSS-RR denoise + DLSS upscale exist. 1 spp + reconstruct ACROSS frames, never trace-to-clean. Full state + the 5-step ms-measured perf plan: auto-memory `realtime-pipeline-state.md`.
