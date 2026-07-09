# Ochroma Engine Agent Rules

This file is for Codex and other coding agents. It summarizes the actionable
parts of `CLAUDE.md`; when in doubt, read `CLAUDE.md` and the game
`../../Ochroma/projects/urban_horizon/AGENTS.md` too.

## Purpose

- Ochroma exists to ship Urban Horizon, a real-time deterministic city builder.
- Engine crates must stay game-agnostic. Do not introduce game concepts such as
  zoning, childcare, policy, or building gameplay into `vox_*`,
  `ochroma_engine`, Spectra, or Forge engine code.
- Engine work should expose primitives the game can wire, not game-specific
  shortcuts.

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
- Shipped games use an Ochroma runtime bundle: `runtime/engine` for engine
  runtime data/config and `runtime/renderer` for Spectra/Slang/CUDA/DLSS assets.
  Do not require game launchers to point at source-tree Spectra or Slang paths.
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
