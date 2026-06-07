# AAA Capability Program — Spec Index

**Status:** Draft program index · 12 specs · 2026-06-07
**Source roadmap:** [`2026-06-07-aaa-capability-roadmap.md`](../2026-06-07-aaa-capability-roadmap.md) — read it for the full 36-gap audit and the wedge thesis.

---

## The wedge thesis (1 paragraph)

Ochroma is not chasing Unreal/Unity parity — it is "SOTA our way" on a defensible wedge that an RGB engine **structurally cannot copy**: a 16-band spectral Gaussian-splatting engine where runtime spectral relight / metamerism is a *frame-rate game mechanic* (a captured world played under a light it was never captured in; a forgery that reads identical under the gallery lamp and divergent under the inspection lamp — a non-zero number no RGB pipeline can produce). The product thesis is "capture-relight-populate" as an in-editor, AI-assisted loop for domain-knowledgeable non-game-devs, built on a provability culture (headless pixel/state-asserted gates, 5 CPU-oracle→WGSL twins validated bit-exact on RADV) and an "atom-budget" path to infinite detail. The strategic gap is that the wedge mechanics are today **asset-time, not runtime** — they live in CLI exports and `cargo test`, not in a GPU-resident frame loop a player can touch. The binding constraint underneath everything is "GPU is Alfa omega": the five validated twins each prove a kernel in isolation, but no single `wgpu::Device` yet hosts a GI buffer bound directly as a rasterizer input — every cross-module handoff is a CPU round-trip. These 12 specs make the floor verifiable (CI green on a machine that is not the author's laptop), unify the device, build the resident frame, turn the relight kernel into a mechanic scripts and the editor can reach, and give the editor the productive workflows (save/load, duplicate, validation, play-in-editor, multi-step AI) that turn a viewer into a tool a small team can ship with.

---

## The 12 specs

| Rank | Spec | Dimension | Effort | One-line what |
|---|---|---|---|---|
| 1 | [01 — Push the blitz + CI green](01-push-blitz-ci-green-the-absurd-gap.md) | Stability / Shipping | S | Make the 158-commit blitz branch pass its own gates on a machine that isn't the laptop — fix the missing openusd-rs CI checkout, create the PAT, observe one externally-reproduced green run. |
| 2 | [02 — Runtime spectral relight kernel (GPU)](02-runtime-spectral-relight-kernel-gpu.md) | Wedge | M | Port the verified CPU relight oracle to a WGSL compute pass that re-illuminates the on-GPU splat radiance buffer per dispatch — the 6th bit-exact twin, the wedge mechanic made GPU-resident. |
| 3 | [03 — Metamer/relight game-mechanic seam](03-metamer-relight-game-mechanic-seam.md) | Wedge | M | Expose the relight physics to rhai (`metamer_divergence`, `set_illuminant`) and a one-key editor "forgery" demo with a live computed ΔsRGB HUD receipt — the wedge made script- and player-reachable. |
| 4 | [04 — Unify on one wgpu device (`GpuContext`)](04-unify-shared-wgpu-device-gpu-context.md) | GPU Frame Loop | L | Replace the 7-independent-device topology with one shared `GpuContext { Arc<Device>, Arc<Queue> }` threaded into every GPU module — the hard floor under all GPU-loop work. |
| 5 | [05 — Wire the GPU tiled rasterizer as the viewport path](05-wire-gpu-tiled-rasterizer-viewport-path.md) | GPU Frame Loop | XL | Chain sort → tile-range-build (the missing pass) → raster entirely on-GPU; persistent buffers written once; first honestly-measured GPU rasterization at 2M-splat scale. |
| 6 | [06 — Real project + scene save/load](06-real-project-scene-save-load-shell.md) | Editor Workflows | L | Make `file.save`/`file.open` real — round-trip the shell's entities AND their 16-band spectral splat geometry through a lossless `SavedSplatGeom` codec; the floor under every editor workflow. |
| 7 | [07 — Ask-Ochroma as a multi-step collaborator](07-ask-ochroma-multi-step-collaborator.md) | Editor / AI | L | Lift Ask-Ochroma from one-sentence→one-action to a sequenced `Plan` of validated actions executed as one grouped-undo transaction — "add 5 birch trees" → 5 distinct entities, one Ctrl+Z. |
| 8 | [08 — GPU timestamp + frame-budget HUD](08-gpu-timestamp-frame-budget-hud.md) | GPU Frame Loop | M | Measure real per-pass GPU-ms with `TIMESTAMP_QUERY`, surface it in the status bar, and assert a bounded non-zero delta headlessly — the first ms-asserted gate later GPU gaps regress against. |
| 9 | [09 — Prefab / duplicate / multi-select](09-prefab-duplicate-multiselect-shell-workflow.md) | Editor Workflows | L | A real selection model + `edit.duplicate` cloning selected entities AND their exact overlay splat ranges at an offset, one `PlacedAsset` undo per copy; introduces the entity→range provenance index. |
| 10 | [10 — Asset validation / import gate](10-asset-validation-import-gate.md) | Content Pipeline | S | Gate `import_asset` on integrity (NaN position / non-positive scale), budget, and the wedge-specific spectral-validity lint (non-finite / all-zero radiance — invisible in RGB engines). |
| 11 | [11 — Frame-loop GPU residency for the wedge passes](11-frame-loop-gpu-residency-wedge-passes.md) | Wedge / GPU Frame Loop | XL | Bind the GI output buffer directly as a rasterizer input on one device, zero CPU readback, proven bit-identical to the readback oracle — the GI→raster slice of the resident frame. |
| 12 | [12 — Play-in-Editor](12-play-in-editor-run-game-inside-editor-window.md) | Engine API | L | Press Play in `ochroma_editor`: snapshot authored state (cheap `Vec` clone), run a fresh `EngineLoop` composing existing sub-steps in the viewport, Stop restores exactly. |

---

## 4-phase sequencing (mapped from the roadmap to these specs)

The roadmap's critical-path spine is verbatim: **#4 (verifiable) → #2 (one device) → #7/#12/#3/#8 (measured resident frame) → #5/#34 (relight playable) → #20/#30 (at scale) → #19/#18 (shippable).** Editor (#1→#9/#17/#25) and AI (#6/#13) run as parallel tracks that rejoin at Phase 3. Mapped onto these 12 specs:

### Phase 1 — Floors you can start tomorrow on this exact codebase
Independently startable; nothing depends on architecture not yet present.
- **Spec 01** (roadmap #4) — Push + CI green: turns the wedge from self-attested to externally verifiable. The cheapest, highest-leverage-per-hour item; the floor under every later "headless-proven" claim.
- **Spec 04** (roadmap #2) — Unify on one `GpuContext`: the hard floor under all GPU-loop work (specs 02, 05, 08, 11 all consume it).
- **Spec 06** (roadmap #1) — Editor save/load: the hard floor under all editor workflow work (specs 09, 12 reuse its provenance/snapshot seam).
- **Spec 09** (roadmap #17) — Duplicate/multi-select: introduces the entity→range provenance index that 06 must serialize; Editor-track, parallel to 06.
- **Spec 10** (roadmap #33) — Asset validation gate: a Phase-4 leaf pulled forward — zero architectural prerequisites, hardens every importer immediately.

*Unlocks:* a tool that holds a project, a single device buffers can flow through, an externally-verified green gate, and a hardened import surface.

### Phase 2 — The integrated GPU frame loop + measurement
Now that one device exists (spec 04):
- **Spec 08** (roadmap #7) — GPU timestamps: so every later GPU gap has a *measured* Done-When. First in this phase. (Can actually start before 04 — see verification summary.)
- **Spec 05** (roadmap #3) — Wire the tiled GPU rasterizer as the viewport path.
- **Spec 11** (roadmap #8 + #2/#3 merge) — Resident GI→raster buffers: kill the per-frame `poll(Wait)` stall, proven bit-identical to the readback oracle.

*Unlocks:* the first honestly-measured sustained GPU frame at scale — the prerequisite for any wedge mechanic at frame rate.

### Phase 3 — The wedge made playable, on the now-resident loop
- **Spec 02** (roadmap #5) — GPU relight kernel (bit-exact twin, on baked radiance) — depends on 04/11.
- **Spec 03** (roadmap #5 reachability + #11/#13) — Metamer/relight script + editor seam: the CPU-oracle-first half that wires the wedge into a product surface, waiting for 02's GPU twin.
- **Spec 07** (roadmap #6) — Multi-step Ask-Ochroma: the AI collaborator's qualitative jump from demo to product; establishes the `Plan` contract the real LLM backend (#13) inherits.
- **Spec 12** (roadmap #9) — Play-in-Editor: turns the editor into a place you author *and test* real gameplay; the stage on which relight becomes a clickable moment.

*Unlocks:* "capture-relight-populate" as an in-editor, frame-rate, AI-assisted loop — the actual product thesis.

### Phase 4 — Shippability + content supply at scale
None of the 12 specs sit squarely in Phase 4, but two pay it forward early:
- **Spec 10** (roadmap #33) is nominally Phase 4 (validation gates / the daily art-team edit loop) but pulled into Phase 1.
- The provenance/codec work in **specs 06 and 09** is the foundation #14 (runtime state persistence), #22 (save versioning), and #25 (crash recovery + autosave) extend in Phase 4.

---

## Verification summary

Every spec carries an adversarial-verifier verdict at its head, and (where the skeptic flagged anything) a `## Verification corrections` section that surfaces the finding rather than silently fixing it.

| Verdict | Count | Specs |
|---|---|---|
| **Sound** (no design defect found) | **3** | 01, 08, 09 |
| **Flagged** (`sound=false` recorded) | **9** | 02, 03, 04, 05, 06, 07, 10, 11, 12 |

The `sound=false` marks are **not uniform** — they fall into four honest classes, and only two are real design defects:

- **Real defects to fix during implementation (2):**
  - **Spec 11** — *blocker:* the headline Done-When command (`scale_trial --gpu-resident`) is not launchable because `scale_trial.rs` has no argv parsing; the flag-parsing addition is a hidden prerequisite relocated to the front of Step 3.
  - **Spec 12** — *fatal data-flow bug:* the enter-Play spawn tuple omits the asset-registration component `gather_splats_system` requires, so it produces zero rendered splats; the fix is to copy the known-gathering spawn tuple from `engine_runner.rs` verbatim.
- **Re-grounding needed, one half affected (1):**
  - **Spec 03** — *load-bearing false premise:* `EditorShell` has no live rhai `Engine` to register host functions on; the rhai-binding half (§2b, Step 3) must be re-grounded against where rhai actually lives. The metamer-kernel half (Steps 1–2) is unaffected and verified.
- **Honesty/label issues, design sound (2):**
  - **Spec 06** — effort label says M; the roadmap and actual scope say **L** (new type + new codec module + new provenance field + 5 steps). Design verified accurate.
  - **Spec 05** — `sound=false` is about scope gaps the spec already discloses (8-of-16-band truncation, two divergent camera uniforms, the genuinely-new `TileRangeBuildPass`), not a false claim.
- **Process/audit-limit, no defect found (4):**
  - **Specs 02, 04, 07** — `sound=false` reflects truncation of the markdown the skeptic received (it could not certify the first launchable step it could not see); all checkable grounding refs verified clean.
  - **Spec 10** — `sound=false` is one mischaracterized grounding line (a loose line-number citation); every accessor the validator actually calls is independently verified.

Net: of 9 flagged specs, **2 carry a true implementation blocker** (11, 12), **1 needs a half re-grounded** (03), and **6 are sound designs flagged for honesty/process reasons** (02, 04, 05, 06, 07, 10). All grounding references that were checkable resolved accurately — the provability culture held up under adversarial review.

See [`00-SURPRISES.md`](00-SURPRISES.md) for the cross-spec surprises register (wedge synergies, cheap-because-of-existing-capability, first-mover angles, reuse).
