# Ochroma Engine Agent Rules

This file is for Codex and other coding agents. It summarizes the actionable
parts of `CLAUDE.md`; when in doubt, read `CLAUDE.md` and the game
`../../Ochroma/projects/urban_horizon/AGENTS.md` too.

## Working principle (LAW)

- Every step takes us forward — no shortcuts, done properly.
- No old/outdated tech or techniques; forward-leaning, **AAA SOTA in everything**.
- A step that isn't SOTA-quality isn't done.

## Purpose

- Ochroma exists to ship Urban Horizon, a real-time deterministic city builder.
- Engine crates must stay game-agnostic. Do not introduce game concepts such as
  zoning, childcare, policy, or building gameplay into `vox_*`,
  `ochroma_engine`, Spectra, or Forge engine code.
- Engine work should expose primitives the game can wire, not game-specific
  shortcuts.

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

## Non-Negotiables

- Spectra is the only 3-D renderer. No raster renderer or CPU renderer for world
  frames. Software BVH is a diagnostic path only.
- CPU may orchestrate resources, scene deltas, residency, and material parameter
  selection; live rendering must not CPU-shade, repaint, sample textures,
  generate render mips, or de-index source assets.
- Real render validation uses the live real-time present path, not offline
  trace-to-clean stills.
- Performance is a ship gate: Performance tier must hold >=30 fps on AMD 780M.
  Avoid per-frame scene/renderer rebuilds. Prefer resident state, deltas, refit,
  and config-driven settings.
- Protect determinism: id-sorted folds, no HashMap/HashSet iteration-order
  dependence, no RNG-order dependence, fixed-epsilon f64, replay-exact command
  logs, and schema version bumps when formats change.
- No self-deception: no `todo!()`, `unimplemented!()`, empty bodies,
  `assert!(x.is_some())` tests, unwired features, or population-independent
  benches presented as product evidence.
- Dependency direction (LAW): the engine NEVER asks Forge for anything. Forge is
  an independent authoring oracle. Design decides what assets/terrains/maps are
  needed and asks Forge for them independently; the engine/game/renderer only
  make assets from Forge game objects, load, and render them faithfully. "Feature
  X can't win until Forge/the cook covers more / produces something richer" is a
  category error and a forbidden conclusion. Scope every engine/render task to
  what the engine owns — correctly and efficiently converting, loading, and
  rendering the game objects AS GIVEN. Content coverage/detail/representation is a
  design decision made by asking Forge, never an engine gap. Inverting this arrow
  is what makes investigations false-dead-end at the cook.

## Render And Config Laws

- `urban_horizon/assets/config/render.ron` is the game render source of truth.
  Do not hardcode look, lighting, resolution, or fidelity knobs in engine code
  when they belong in config.
- If adding a render feature, expose it through the config/runtime path and
  verify it is live with an env sweep, config change, or visible probe.
- Witness real-time rendering at 1 spp plus reconstruction/denoise/upscale.
  Never claim a render/content fix without inspecting the rendered frame.
- Local Linux may compile pure Rust and some Slang front-end pieces, but
  `spectra-native` can require `SLANG_DIR`, `LD_LIBRARY_PATH`, and GPU/OptiX
  resources that are not present locally.

## Workflow

- Preserve dirty worktrees. Do not revert user changes. If a file was dirty
  before your edit, inspect the existing change and make minimal compatible
  edits.
- Do not commit unless explicitly asked.
- Do not launch games, start remote box builds, run remote renders, or touch a
  live user session without explicit permission.
- Prefer local `cargo check`/`cargo test` for compile-level verification. For
  GPU/render claims, report the exact limitation if the local environment lacks
  Slang, CUDA, OptiX, or required DLLs.
- Use `rg`/`rg --files` first for search. Use structured parsers/APIs where
  reasonable. Keep edits scoped to the requested behavior.

## Notes To Future Agents

- Lean on the newest implementation/config/runtime evidence; stale plans and
  old comments lose to current code and witnessed output.
- Leaf alpha currently crosses repo boundaries: `HybridMesh`/atlas/packer in
  `vox_render`, scatter tagging in `urban_horizon`, and the Slang cutout branch
  in `spectra/slang/megakernel.slang`.
- `opacity_tex == albedo_tex` is the foliage base-color-alpha signal; keep
  separate opacity maps and non-foliage textured materials out of `MAT_VEGETATION`.
- Wave-2 OMM currently bakes 2-state OptiX masks for LOD0 vegetation protos only;
  decimated LOD OMM needs UV/material-preserving decimation first.
- Local Linux lacks CUDA toolkit headers; OptiX/OMM FFI checks are box-gated even
  when non-OptiX `spectra-native` checks pass.
- Shipped games use one self-contained Ochroma runtime bundle: `runtime/engine`
  for engine runtime data/config and `runtime/renderer` for the complete Spectra
  closure (kernels, reconstruction, frame generation, and every legal native
  dependency for that target). Spectra is not separately installed and games
  never resolve source-tree, SDK, cache, or system-installed renderer payloads.
- Local `spectra-native` tests can fail before compiling code if `libslang.so`
  is not on the loader path; report that as an environment gap, not a code result.
- Avoid broad `rustfmt` over large dirty files; it can create noisy unrelated
  diffs. Prefer targeted edits and restore any clean files accidentally touched.

## Plans And Docs

- New plans use `docs/templates/plan.md`.
- New designs use `docs/templates/design.md`.
- Plans live in `docs/superpowers/plans/`; designs live in
  `docs/superpowers/specs/`.
- `Done When` must name an exact command and exact human-visible output.
- Every task should implement and wire the feature in the same step.
