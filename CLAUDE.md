# Ochroma Engine

Spectral Gaussian Splatting game engine.

## Working principle (LAW)

**Every step takes us forward. No shortcuts — we do things properly.** No old or
outdated technologies or techniques; we are forward-leaning and aim for **AAA SOTA
in everything we do.** A step that isn't SOTA-quality isn't done.

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

## Assets are NEVER cooked with the game (LAW)

**No asset is cooked with the game. Not buildings, not maps, no asset at all.**
The game and the assets are **completely separated** until an asset is created as a
**game object**.

- A **game object** = the **mesh** + the attributes it needs to work in the game and
  in the **simulation**.
- The mesh is **still not cooked with the game**. It is only **COPIED** into
  `game/assets/official/`, and **loaded at runtime when the game starts**.
- **Game, engine and renderer are PRE-BUILT.** Authoring or cooking an asset must
  trigger **ZERO** game/engine/renderer recompilation. If touching an asset rebuilds
  the game, the separation is broken — fix the dependency, do not work around it.
- **Cooking is incremental and standalone**: cook **only the newly authored asset**
  (never the corpus), then **render it to inspect**. Author → cook-one → render-one
  → look. That loop is seconds-to-a-minute, not tens of minutes.
- The asset pipeline (Forge + the cook tools) is its **own** workspace. It may depend
  on Forge freely. **The game/engine depends on NEITHER Forge NOR the cook** — only on
  the asset bundle format it reads at runtime.

### Asset container format — `.vxp` (vox pack)

**Ours, not borrowed.** The container `.vxp` is a sibling of the engine's existing
`.vxm` (VXM v3, `vox_data`). We studied Cities: Skylines II's `.cok` as a *reference
implementation* — `.cok` is Colossal Order's proprietary extension, **not a standard** —
but the underlying pattern (a stored ZIP of content-addressed entries) is standard
practice: Unreal `.pak`, Quake `.pk3`, and Unity asset bundles are all zip-family
containers.

- **Container:** a ZIP with compression method **STORE** (uncompressed), extension
  **`.vxp`**, written by the pipeline into `assets/official/`. Uncompressed so entries
  map directly with no inflate cost at load. (Textures/meshes are already compressed
  internally — zip-deflating them again would cost CPU for ~nothing.)
- **Entry naming:** `<asset_id>_<32-hex-content-hash>.<Kind>`, each with a **32-byte
  `.cid`** sidecar. Content-addressed: identical content dedupes, integrity is checkable.
- **Kinds — together these ARE the "game object" (mesh + the attributes it needs in game and sim):**
  - `.Geometry` — the finished mesh, **BINARY** (never pretty-printed JSON).
  - `.Geometry` per LOD as separate entries (`_LOD1`, `_LOD2`, …).
  - `.Surface` — material / PBR data.
  - `.Metadata` — the game/sim attributes (footprint, zone, levels, capacities, sockets).
    Keep it **small** so it can be read without the mesh.
  - `.Texture` — a **thumbnail, REQUIRED for every asset**, for in-game display
    (build menus, asset pickers).
- **Grouping:** many assets per pack, grouped by category/theme — not one file per asset.
  (CS2 ships 512 entries in one 96 MB blob; that density is the right order of magnitude.)
- **SUB-ASSETS are first-class entries.** Verified in CS2's `Blob_Bikes.cok`: `Bicycle01`,
  then `Bicycle01Battery01` and `Bicycle02Basket01` as their OWN entries, each with its own
  `_LOD1`/`_LOD2` chain. Naming grammar is compositional: `<Parent><Part><NN>`.
  So a rooftop HVAC unit, a vent, an aerial or a shopfront sign is authored ONCE as its own
  asset (`OfficeTower01HVAC01` + LODs) and composed onto parents — reusable across buildings,
  not duplicated inside each one.
- **Two further kinds seen in CS2 and worth supporting:** `.Animation` (moving parts ride the
  same container — e.g. `Bicycle01_Cycling`) and `.Atlas` (a shared texture atlas, which is
  how CS2 avoids per-asset textures).

**Where we deliberately DIFFER from CS2** (verified against the install — do not describe
these as "adopting CS2"):
- CS2 has **no `.Metadata` kind**; it keeps gameplay data outside the container. We put the
  game/sim attributes **in** the pack, because the "game object" is defined here as mesh +
  attributes travelling together.
- CS2's `.Texture` is **not** a per-asset thumbnail — only ~165 of 27,910 entries are
  textures, and they are **atlased**, with mips in separate `MidMips*.cok`. Our required
  per-asset `.Texture` thumbnail is **our own addition**, for in-game display.
- **Index:** a top-level index lets the game enumerate content and read `.Metadata` +
  `.Texture` at startup **without loading geometry**; geometry loads on demand. This is the
  single most valuable idea from the CS2 study: they put a 644-byte metadata entry beside a
  74 MB payload, which is how listing and previews stay instant.
- **Settings/config ship as PLAIN readable files** (`assets/official/config/*.ron`) — not
  packed, not compiled into the binary.
- **Only finished meshes** live in the game folder. No authored source, no directives.

**Mechanical test (must hold):** `cargo tree -p urban_horizon | grep -c forge` == `0`,
and the game builds and renders with **no** asset/cook feature enabled.

**Why this is a LAW:** welding the cook into the game crate made every render witness
rebuild the entire authoring toolchain (plus Slang), pushed one project's `target/` to
137 GB, put 36 GB of cooked output inside the game repo, and made a Forge edit force a
game rebuild — turning a should-be-seconds authoring loop into 20+ minutes. It also
inverts the dependency-direction LAW below: the engine's job is to **load and render**
game objects, never to produce them.

## Dependency direction — the engine NEVER asks Forge for anything (LAW)

Forge is an **independent authoring oracle**. The flow is strictly one-way:

1. **We** (design) decide what assets, terrains, and maps we need.
2. **We** ask Forge for exactly those, **independently** of the engine.
3. Forge authors the game objects.
4. The **engine / game / renderer** just **make assets from the Forge game objects, load them, and render them** — faithfully, as well as possible.

The engine's job is to render whatever a game object contains — not to make demands
of Forge or the cook. **"Feature X can't win until Forge/the cook produces something
richer (more coverage, more detail, a different representation)" is a category error
and a forbidden conclusion.** The cook is a faithful converter (Forge game object →
renderable asset), not a negotiation partner the engine lobbies. If a game object has
16% of its geometry as MegaGeometry programs, the engine renders 16% programs + 84%
mesh, correctly — full stop. If we want that content shaped differently, **we ask
Forge, independently** — that is a design/content decision, never an engine gap.

Why this is a LAW: inverting this arrow — letting the engine's success depend on the
content pipeline delivering more — manufactures a false dependency that always
dead-ends at the cook (the one component downstream of everything and never "done"),
turning it into a permanent scapegoat. Scope every engine/render investigation to
what the engine owns: correctly and efficiently converting, loading, and rendering the
game objects **as given**. Coverage/detail/representation of the content is out of
scope for the engine by construction.

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

## Render/material capabilities (engine side) — ONE SOURCE OF TRUTH, and it is not this file

**Canonical inventory: `../../Ochroma/projects/urban_horizon/docs/reference/capability-matrix.md`.**
710 lines, four columns per capability — **BUILT / REACHABLE / ENABLED / WITNESSED** — with `file:line`
evidence, a DEAD list (zero non-test callers), a DORMANT list (gate + shipped default), the
CUDA-vs-Vulkan-vs-Metal split, and a §10 block of re-verification commands with expected output.
Full Forge geometry/material inventory: `../forge/CLAUDE.md`.

**Why this section no longer lists capabilities.** It used to, and on 2026-07-26 an audit found **ten
of its claims contradicted by the code** — including a type name that does not exist
(`ResidentCityRenderer`), a material id that is not a material (`MAT_GLASS_LIT`), "Bruneton
atmosphere" and "DaylitCity tonemap" (neither ships), and "clouds are LIVE… not missing tech" (they
were a scalar dimming factor). Every line number it quoted was stale. A full working day was lost
reasoning from those claims. **A hand-maintained capability list drifts silently; a matrix of
re-runnable commands does not.** Do not re-add one here.

**The four columns exist because they disagree.** Repeatedly, in one session: the `.vxp` reader was
complete and tested with 2.9 GB of packs the game could not open; vegetation wind was finished on both
sides and imported by nothing; `refit_proto_vertices` ran every frame, failed every frame, and logged
why into a void because the game installed **no tracing subscriber**; `prev_transform` was written for
motion vectors and never read. **A flag, struct or function existing is NOT evidence the path runs —
check the CALLER and the ARTIFACTS.**

**Standing rules (these are the durable part):**
- Most render gaps are **wiring, not missing tech** — but verify against the matrix, never against
  memory or a comment. Three load-bearing comments were found false on 2026-07-26 alone
  (`mesh_convert.rs:209` "CLAS assumes de-indexed" — false, and de-indexed is ~3× *worse* for cluster
  packing; three comments claiming the tick drives celestial time; `render.ron`'s "NVIDIA/OptiX only"
  on the vertex-refit path).
- **Don't validate render/content on box-stub scenes** — the witness is a hero-camera frame on the
  real-time present path.
- **Never conclude a render gap needs richer content from Forge** — see the dependency-direction LAW.

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
