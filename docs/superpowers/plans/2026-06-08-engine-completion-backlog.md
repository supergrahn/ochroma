# Ochroma Engine — Honest Completion Backlog (2026-06-08)

What "engine complete" honestly means, and the finite work to get there. Produced
by a grounded two-agent planning pass (`wf_26b64dfc`).

## Definition of "complete" (and what it can't mean)
**Complete = proven kernels + wired spine + the bounded Phase-4 plumbing that makes
a 20–60h experience runnable, crash-free, at city scale.** The kernels are proven
(spectral GI, atom-budget LOD, relight, USD, the 6 bit-exact twins) and the spine
is wired (one `GpuContext` → tiled rasterizer → resident GI→raster zero-readback →
ms budget → save/load → multi-step AI → Play-in-Editor; AAA specs 02–12, on master).

It does **not** mean: externally verified (Spec 01 / CI is user-gated), shippable
(no Windows build, soak harness, or crash-save), scalable past **~1.68M splats**
(the hard `max_storage_buffer_binding_size` wall the `city_ceiling` bench found),
content-supplied (no real 3DGS training), or populated (`CitySim` is a fixed small
city). An engine is never literally "done"; this is a defensible line.

## Sequenced backlog

**Phase 0 — credibility (the floor):**
- **Spec 01** — push + reproduce CI green off-machine (user-gated; the openusd-rs CI
  checkout is already wired, needs the `SIBLING_REPOS_PAT` to cover openusd-rs).
  Re-run `cargo test` to re-confirm the ~2677 claim the whole backlog assumes.

**Phase 1 — cheap kernel/ship hardening (days each, before #20 reshards passes):**
- **Deterministic radix tie-break** (~1d) — fold the splat index into the sort key
  low bits so equal-tile-key blend order is reproducible on both CPU+GPU paths.
- **Crash recovery** (#25/#28, ~2–3d) — `panic::set_hook` → save-on-crash; atomic
  tmp→fsync→rename; wire into `main.rs` (builds on the existing `autosave.rs` timer).
- **Save versioning + migration** (#22, ~2d) — version tag + `load_migrated()`.
- **Runtime state persistence** (#14, ~2–3d) — a `runtime_state` blob (rhai/anim/
  physics) round-tripped through save→load.
- **Extensible scheduler** (#15, ~2–3d) — `add_systems`/`Schedule` on EngineRuntime.
- **Facade + one-call boot** (#32, ~2d) — `App::run` in the `ochroma_engine` prelude.
- **Project scaffolder** (#23, ~3–4d) — `vox_tools new-game --template <t> --name <n>`
  emits a game crate + `ochroma_project.toml` + starter scene. (The "New Project"
  command — depends on #15 + #32.)

**Phase 2 — the scale leap:**
- **GPU residency/streaming manager** (#20, XL ~2–4wk, but DE-RISKED) — breaks the
  1.68M wall. The pieces already EXIST and are unit-tested but wired to nothing:
  `SplatBufferPool` (slot pool), `WorldPartition`. The work is wiring
  `TiledSplatRenderer` to a pooled/sharded resident set instead of one monolithic
  `splat_buf`. This is the #1 highest-leverage engine task.
- **CitySim scale constructor** (XL ~1–2wk) — populate thousands of plots (the
  100k-citizen milestone needs this; today `new_small` caps ~58 housing).

**Phase 3 — shippability breadth (XL):**
- Device-lost recovery + min-spec (#21), GameUiKit (#35), 20h-world streaming
  hardening (#30), reflectance/emission data-model split (VXM v3, #34).

## Out of scope / not "completion"
Anti-cheat (rejected), Windows/Steam packaging (non-gating), **real dense spectral
3DGS training (#26/#29 — research-grade, multi-month)**, ORCA crowds (#36, quality
upgrade not a gap), DCC re-import (#27) + full glTF fidelity (#31). Also noted: ~28
standalone `vox_sim` subsystems (weather/disasters/pollution/utilities/…) exist but
aren't wired into `CitySim`'s loop — that's GAME integration, not engine completion.
