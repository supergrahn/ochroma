# Design: AAA-Capability Roadmap for Ochroma (2026-06-07)

**Status:** Draft
**Scope:** Synthesizes seven dimension audits into one ranked, dependency-ordered roadmap from the current "validated islands" state toward shipped-quality 20–60h experiences, framed around Ochroma's unforgeable wedge rather than generic engine parity.
**Related:** `2026-06-06-editor-sota-shell-design.md`, `2026-06-07-spectral-relight-design.md`, `2026-06-06-atom-budget-splat-renderer-design.md`, `2026-06-06-engine-competitive-research.json`

> Honesty preface (binding): AAA here means a 20–60h experience at shipped quality produced by a small AI-assisted team — not a demo, not a tech showcase. Ochroma today ships two demo loops ("collect 10 orbs", "jump to the gold platform"). The distance is large and is mostly **architectural integration and shipping plumbing**, not missing algorithms. The hard kernels (spectral GI, atom-budget LOD, motion matching, relight, USD import) exist and are individually proven. This roadmap orders the integration. Timelines are stated as ORDER, never dates.

---

## 1. What AAA Means for Ochroma (the wedge IS the strategy)

The Wedge dimension is explicit: do **not** chase Unreal's feature checklist. Ochroma's AAA is defined by four differentiators that compound into one experience no RGB engine can ship, plus the table-stakes that gate them from being playable.

1. **Spectral relighting / metamerism as a frame-rate game mechanic.** The crown jewel. `relight_scene()` (vox_render/relight.rs, 1251 lines) is proven, shadow-aware, and metamer-validated — but it is a CPU offline batch op wired only into `vox_tools relight`; it has zero hits across the editor/engine frame loop and no WGSL twin (Wedge dim). AAA-for-Ochroma = relight running per-frame on the GPU so a world captured under one illuminant can be played under any other. This is the unforgeable moat; everything else is in service of it.

2. **Splat-native capture + scale.** Worlds reconstructed from photographs, streamed at 20h scale, then relit. Today the COLMAP path emits a sparse point cloud (1 splat/point, fixed scale 0.01) and `turnaround.rs` is a self-documented synthetic placeholder (Content + Wedge dims) — categorically below Postshot/Polycam/Luma/Scaniverse. The capture half must look real before the relight half becomes the killer combo.

3. **Provability culture as a product property.** 2,598 test functions; both binaries have headless `--smoke` gates with behavioral pixel/state assertions (Stability dim). "Headless pixel-asserted everything" is the credibility wedge. AAA-for-Ochroma extends this from correctness to **performance** (ms-asserted), **durability** (crash/soak-asserted), and **external verification** (CI green on a machine that is not the laptop).

4. **AI-native creation.** Ask Ochroma generates compile-verified rhai behind a schema-validated, fully-undoable IntentBackend seam (Editor + Wedge dims). AAA-for-Ochroma = the AI is a multi-step collaborator ("build me a village" → a validated `Vec<IntentAction>` executed as one undo group) and a real LLM backend is connected behind the existing validation gate.

**The synthesis:** Ochroma's AAA is "capture a real place, relight it under light it never saw, populated by AI-authored content, with every claim externally provable." The generic checklist (consoles, anti-cheat, 300-person tooling) is explicitly NOT the target. The wedge mechanics are currently asset-time, not runtime — making them runtime is the strategic spine of this roadmap.

---

## 2. The Honest Distance (current state per dimension)

**Runtime Performance & GPU Frame Loop.** Validated islands, no integrated frame. 17 `Instance::new()` sites; 6 compute modules each own a separate device, plus the present surface owns a 7th — every cross-module handoff is a CPU round-trip. Every twin reads back and stalls (`poll(Maintain::Wait)`). The render graph is CPU-only scheduling with zero wgpu types; the GPU-driven rasterizer pieces (RadixSort, TileAssign, tiled-EWA SplatRaster) are built, tested, and wired into nothing. No GPU timestamps anywhere. The "2.05M splats" headline measures CPU LOD select + one CPU-rasterized frame, not a sustained GPU frame rate.

**Gameplay Systems Depth.** Broad but uniformly demo-grade. Genuine strengths: motion matching (production-shaped, drives walking_sim), Rapier physics (~5k LoC, live, spectral fracture differentiator), spectral AI perception (NPC flees fire-band radiance — unique and wired). Hollow: behavior trees read a pre-filled HashMap and execute nothing (wired into zero binaries); anim "state machine" is a 2-state crossfade; crowds are separation-only with no RVO; no quest/inventory/dialogue/faction frameworks exist anywhere.

**Editor Workflows.** One-day-old windowed editor opens a hardcoded demo and discards everything on close. `file.save`/`file.open` are literal `|| {}` no-ops; shell state (entities/overlay/graph) is never serialized. No dirty tracking, no crash recovery, no multi-select/copy-paste. Ask Ochroma is structurally ahead but single-step. Undo is the strongest piece (range-tracked, hash-guarded, real revert assertions).

**Content Pipeline.** Strong, honest import breadth (VXM v1–3, PLY, SPZ, glTF KHR round-trip, native USD — a genuine first-mover) and a differentiated asset-time trio (prune with perceptual guard, relight, USD). But the capture path stops at a sparse point cloud (no 3DGS training anywhere), glTF import reads only `base_color_factor` (textures skipped entirely), skinning is single-bone nearest-joint, and there is no DCC iteration loop (no re-import, no batch, no validation gates).

**Stability / Platforms / Shipping.** Strong provability, near-zero shipping infra that has ever executed. CI has NEVER run green: HEAD is 155 commits ahead of an origin >2 months stale; ci.yml's own comment admits every run since March died at manifest-load. `panic="abort"` + ~840 unwrap/expect = instant death with no crash report, no save-on-crash. Main game loop swallows GPU errors (`Err(_) => return`). Non-atomic saves; save version field written but never checked. No Windows build has ever compiled; no soak test longer than 2.6s; no min-spec.

**Engine Architecture & API.** 154.5k lines, deep render stack, thin/immature game-facing API. ECS is a bare bevy `World` with a HARDCODED 6-system tick and no way to register game systems. Editor and runtime are separate binaries — the editor never constructs or ticks an EngineLoop (no Play button). rhai host bindings are global-state stubs; the wasm runtime is a 51-line shell with an empty Linker (mods cannot call into the engine). rustdoc near-absent; the `ochroma_engine` facade is an unused re-export shim.

**The Wedge as Strategy.** Differentiators are real but asset-time, not runtime-mechanic-grade. The decisive fact: `GaussianSplat` stores only `spectral:[u16;16]` (baked radiance) — no separate reflectance/illuminant. Runtime "relighting" today = full-scene GI re-bake returning a fresh clone. Making relight a frame mechanic is a DATA-MODEL change, not just a shader port.

---

## 3. The Ranked Roadmap

Ranking is the judges' totals (/90) across the merged dimension set. Effort: S/M/L/XL. "solo" = achievable by one engineer.

| # | Gap | Dimension | Effort | Score/90 | First slice (crate · file) |
|---|---|---|---|---|---|
| 1 | Project + open/save/save-as in the shell (wire WorldSave to editor) | Editor | L | 74 | `EditorShell::save_world/load_world`, wire `file.save`/`file.open` — vox_app/src/shell/mod.rs |
| 2 | Unify on one wgpu device/queue (`GpuContext`) | GPU Loop | L | 72 | `GpuContext{device,queue}`, `GpuGi::new_with_context` — vox_render/src/gpu/* + editor `resumed()` |
| 3 | Wire GPU-driven tiled rasterizer as viewport path | GPU Loop | XL | 71 | `TiledSplatRenderer::new(&GpuContext)` chaining RadixSort→TileAssign→tiled-EWA — vox_render/src/gpu/* |
| 4 | Push repo + CI green on real machines | Stability | M | 71 | `SIBLING_REPOS_PAT` secret, push blitz branch, watch `test` job — .github/workflows/ci.yml |
| 5 | GPU runtime spectral-relight kernel | Wedge | L | 68 | WGSL relight pass + `OCHROMA_RELIGHT=gpu`, bit-exact vs `relight_scene` — vox_render/src/relight* |
| 6 | Ask Ochroma multi-step collaborator (compound intents) | Editor | L | 67 | `IntentAction::Plan(Vec<..>)`, "row of N" expander, one undo group — vox_app/src/shell/mod.rs |
| 7 | GPU timestamp instrumentation + frame-budget HUD | GPU Loop | M | 66 | Enable `TIMESTAMP_QUERY` in `GpuContext`, wrap raster pass — vox_render + editor status bar |
| 8 | Resident buffers for in-frame GPU→GPU handoff | GPU Loop | L | 65 | `GpuGi::step_resident(&Buffer)->Buffer` (no map/poll) — vox_render/src/gpu/spectral_gi.rs |
| 9 | Play-in-Editor: bridge editor to EngineLoop | Engine API | L | 65 | `PlayState` + embedded `EngineLoop` blitting to docked viewport — vox_app/src/bin/ochroma_editor.rs |
| 10 | Behavior tree that ticks real game logic | Gameplay | L | 63 | Action/condition registry + Running-persistent `BTContext`, port NPC flee — vox_core/src/behavior_tree.rs |
| 11 | Real script/mod host API (bind rhai/wasm to entity+world) | Engine API | L | 63 | `trait ScriptHost` over World, rebind rhai `set_position` to bound entity — vox_script/src/rhai_runtime.rs |
| 12 | Render graph as GPU executor (record into one encoder) | GPU Loop | XL | 62 | `trait GpuPass{record(enc,res)}`, wrap bloom_pass, 2-node graph — vox_render/src/render_graph.rs |
| 13 | Runtime AI content generation (connect a real LLM backend) | Wedge | M | 62 | Wire `LlmBackend::Remote` behind existing schema gate — vox_ai/src/llm.rs |
| 14 | Runtime game-state persistence (script/AI/anim/physics) | Gameplay | L | 60 | `runtime_state` section in WorldSave (rhai globals + anim state) — vox_data/src/world_save.rs |
| 15 | Extensible system scheduler (register game systems/stages) | Engine API | M | 60 | bevy `Schedule` per phase + `EngineLoop::add_fixed_system` — vox_core/src/engine_runtime.rs |
| 16 | In-editor profiling / performance HUD | Editor | S | 59 | Status-bar segment: frame ms + overlay/base splat counts — vox_app/src/shell (EditorShell::ui) |
| 17 | Duplicate / copy-paste / prefab workflow | Editor | M | 58 | `EditorShell::duplicate_selected()` re-planting range +offset — vox_app/src/shell/mod.rs |
| 18 | Soak / endurance harness (multi-hour, leak-asserted) | Stability | L | 58 | `walking_sim --soak SECS` sampling RSS, slope assert — vox_app/src/bin/walking_sim |
| 19 | Real Windows build + Steam-first packaging | Stability | L | 58 | Promote windows-latest CI job to gating — .github/workflows/ci.yml + release.yml |
| 20 | GPU residency manager (streamer → persistent buffer pool) | GPU Loop | XL | 57 | `GpuResidencyManager::new(&GpuContext, budget)` + free-list — vox_render + vox_data/tile_streamer.rs |
| 21 | Robust device-lost recovery in game loop + min-spec | Stability | M | 57 | Replace `Err(_)=>return` with Lost/Outdated→reconfigure — vox_app/src/main.rs + MIN_SPEC.md |
| 22 | Save format versioning + forward migration | Stability | S | 57 | `load_migrated()` version dispatch + golden v1 fixture — vox_app scene_serialize.rs |
| 23 | Project scaffolding (`vox_tools new-game`) | Engine API | M | 57 | Scaffolder emitting facade-only game crate — vox_tools/src/main.rs |
| 24 | Condition-driven animation state-machine graph | Gameplay | L | 55 | `AnimGraph{states,transitions,params}` — vox_render/src/animation/ |
| 25 | Session crash recovery + autosave of editor document | Editor | M | 55 | `dirty:bool` + autosave tick → `<project>.recovery` — vox_app/src/bin/ochroma_editor.rs |
| 26 | Real 3DGS training backend for capture | Content | XL | 55 | COLMAP camera-pose export → `CaptureSession` JSON — vox_data/src/colmap_pipeline.rs |
| 27 | DCC source→engine iteration loop (re-import/watch/batch) | Content | M | 55 | `<asset>.import.json` sidecar + `vox_tools reimport` — vox_tools + import_pipeline.rs |
| 28 | Crash reporting + save-on-crash (panic policy) | Stability | M | 55 | `set_hook` → crash file + atomic save (tmp+fsync+rename) — vox_app + scene_serialize.rs |
| 29 | Production-grade capture (dense 3DGS, not sparse dots) | Wedge | XL | 55 | Wire COLMAP DENSE (`patch_match_stereo`+`stereo_fusion`) — vox_data/src/colmap_pipeline.rs |
| 30 | Scale + streaming hardening for a 20h relit world | Wedge | XL | 53 | Headless harness: stream >5M splats + GPU GI, measure re-bake stalls — vox_app scale_trial |
| 31 | Full glTF/DCC fidelity: textures + weighted skinning + retarget | Content | L | 52 | Sample base-color TEXTURE → Smits 16-band per surfel — vox_data/gltf_import.rs |
| 32 | API docs + stability contract for ochroma_engine facade | Engine API | S | 51 | `cargo test --doc -p ochroma_engine` doctest on `EngineLoop::new` — ochroma_engine/src/lib.rs |
| 33 | Asset validation gates + LOD bake-on-import | Content | M | 50 | `vox_tools validate <asset> --max-splats N`, nonzero exit — vox_tools/src/main.rs |
| 34 | Reflectance/emission data-model split on GaussianSplat | Wedge | XL | 50 | Optional parallel `reflectance:[u16;16]` in VXM v3 — vox_core/src/types.rs |
| 35 | Reusable game-UI framework (menus/HUD/inventory/dialogue) | Gameplay | XL | 47 | `GameUiKit` Panel/Button/List over Vello + focus model — vox_core/src/game_ui.rs |
| 36 | Local-avoidance crowds + game-wired nav (RVO/ORCA) | Gameplay | L | 46 | 2D ORCA half-plane solver fed by navmesh A* — vox_sim/src/crowd.rs |

### Top 10 — expanded

**1. Project + open/save/save-as in the shell (Editor, L, 74).**
*Why blocking:* this is the single hard floor below every other workflow gap — an editor that discards all work on close cannot hold a project, so there is no real authoring, review, or collaboration. *Current:* `file.save`/`file.open` are `|| {}` no-ops (shell/mod.rs:1749-1750); `entities`/`overlay`/`bridge` never serialized; the only app save path writes a city header with empty `data`. *Needed:* an `EditorDocument` that serializes the shell's full state and reloads it bit-for-bit, with menu commands actually invoking it and an Output Log receipt naming path + entity count. The primitives exist unwired: `vox_data::world_save::WorldSave` (bit-exact round-trip test, `#[serde(default)]` tolerance) and `vox_data::prefab::Prefab`. *First slice (agent task):* in `crates/vox_app/src/shell/mod.rs`, add `EditorShell::save_world(&Path)` building a `WorldSave` from `entities` + planted-overlay provenance + `bridge` graph params, and `EditorShell::load_world(&Path)` that rebuilds them; wire the two `file.save`/`file.open` closures via a `ShellRequest` (mirror the existing Undo path) to a fixed `project.vxworld`. Done When: a headless test plants a tree, saves, news a shell, loads, and asserts entity name + overlay splat count match exactly.

**2. Unify on one wgpu device/queue (GPU Loop, L, 72).**
*Why blocking:* no per-kernel optimization can hit 16.6ms while data crosses PCIe and stalls the queue between every stage; with 7 independent devices every cross-module handoff (GI→raster, LOD→raster, many-light→shade) is forced through a CPU `Vec`. Judges verified 6–9 distinct `request_device` sites. *Current:* 17 `Instance::new` sites; GpuGi/atom_budget_gpu/many_light_gpu/splat_rt_gpu/hybrid_compose_gpu/material_gpu_eval each own a device; WgpuBackend owns the present device separately. *Needed:* one `Arc<Device>`+`Arc<Queue>` (`GpuContext`) created once at surface bring-up and threaded into every pass constructor, so buffers from one pass bind directly into the next — while each twin keeps a `new_with_device` constructor so the standalone-device validation tests still run (provability preserved). *First slice (agent task):* introduce `GpuContext{device:Arc<Device>,queue:Arc<Queue>}`; add `GpuGi::new_with_context(&GpuContext, capacity)` alongside the existing `new()`; in `ochroma_editor` `resumed()`, after WgpuBackend is built, construct GpuContext from `backend.device()/queue()` and pass it to a GpuGi. Done When: `OCHROMA_GI=gpu cargo run --bin ochroma_editor -- --frames 2 --shot /tmp/gi.png` prints "GI on shared present device", GpuGi reports the SAME adapter name as the backend, and the log shows no second `Instance::new`.

**3. Wire the GPU-driven tiled rasterizer as the viewport path (GPU Loop, XL, 71).**
*Why blocking:* the shipped `GpuRasteriser` CPU-sorts every splat (`sort_unstable_by`, gpu_rasteriser.rs:554) and reallocs the entire splat storage buffer every frame (`create_buffer_init`:560) — that alone caps frame rate well below 60fps at scale, O(N log N) on CPU. This is the splat-native wedge made real-time. *Current:* the real GPU tiled path (splat_raster.rs / tile_assign.rs / radix_sort_pass.rs, all individually verified with CPU references) has zero callers outside their own tests. *Needed:* a `TiledSplatRenderer` that uploads splats ONCE into a persistent buffer and runs RadixSort→TileAssign→tiled-EWA composite entirely on-device with indirect dispatch, drawing to the graph's target. *First slice (agent task):* in `crates/vox_render/src/gpu/`, build `TiledSplatRenderer::new(&GpuContext)` owning a persistent splat buffer (written once) and `render(camera)->target` chaining the three existing passes on the shared device for a fixed 100k-splat scene. Done When: `cargo run --bin scale_trial -- --gpu-tiled` renders the 2.05M scene's selected subset through the GPU tiled path, reads back, asserts >10% non-black pixels (matching the current CPU assertion within tolerance), and prints a GPU-timestamp-measured raster cost (depends on #7).

**4. Push the repo and make CI run green on real machines (Stability, M, 71).**
*Why blocking:* the entire provability culture — "11 consecutive green / adversarial waves" — is self-attested LOCALLY. 155 commits sit unpushed; origin/master is >2 months stale; ci.yml's own comment admits every run since March died at manifest-load. A green streak nobody but the author can reproduce is a claim, not provability — and provability IS the wedge. *Current:* well-designed workflows (3-repo side-by-side checkout for spectra/crucible path deps, `-D warnings`, clippy, both smoke gates) that have NEVER produced a green run. *Needed:* blitz branch pushed, `SIBLING_REPOS_PAT` secret created (fine-grained read-only Contents on supergrahn/spectra + supergrahn/crucible, plus the pending openusd-rs), one observed green run, a visible badge. *First slice (agent task):* create the secret, push the blitz branch and sibling repos, open a PR, watch the `test` job. Done When: GitHub Actions shows a green check on a SHA matching local HEAD with "Smoke walking_sim" and "Smoke ochroma" passing on ubuntu-latest — i.e. a machine other than the author's has run the gate.

**5. GPU runtime spectral-relight kernel (Wedge, L, 68).**
*Why blocking:* this is the wedge made playable — the one capability no RGB engine can clone. Today `relight_scene()` is proven and metamer-validated but runs only in `vox_tools relight`; `grep relight` across the editor/engine frame loop returns zero hits, and no WGSL twin exists. *Current:* CPU relight + a GPU GI pass that already gathers per-band radiance on-device. *Needed:* a WGSL relight kernel that re-illuminates the on-GPU splat buffer in place per frame given an illuminant SPD, hitting the 5-twin bit-exact-vs-CPU-oracle bar. *First slice (agent task):* in `crates/vox_render/src/`, port the per-splat inner loop of `relight_scene` (intrinsic ÷ reference-SPD → ⊗ target-SPD, sky-ambient add) into a WGSL compute pass overwriting the on-GPU radiance buffer; add `OCHROMA_RELIGHT=gpu`. Done When: a headless test asserts bit-exactness vs `relight_scene` on a 1k-splat scene under a tungsten→daylight swap (same oracle pattern as the GI twin). No data-model change yet — operate on baked radiance, just on-GPU and per-frame. (Depends on #2 for the shared device; pairs with #34 to become physically honest.)

**6. Ask Ochroma as a multi-step collaborator (Editor, L, 67).**
*Why blocking:* this is the differentiation wedge for the target audience — domain creators who can describe a village but can't hand-place 200 buildings. Single-step intent is a demo; the collaborator is the product. *Current:* `run_intent` (shell/mod.rs:639) parses ONE sentence to exactly ONE `IntentAction`; the plant cores (`plant_grown_tree`, `plant_forge_terrain`, `plant_forge_building`, `plant_crucible_scene`) all exist and are individually drivable. *Needed:* a `Plan = Vec<IntentAction>` the parser/LLM can emit, an executor running the steps as one coalesced undo group with per-step receipts, and a few grammar patterns ("a village", "a row of N houses") expanding into a plan. The hard safety part is done — schema validation + undoability already gate every action. *First slice (agent task):* in `crates/vox_app/src/shell/mod.rs`, introduce `IntentAction::Plan(Vec<IntentAction>)` (or `parse_plan -> Vec<IntentAction>`), add a deterministic "a row of N <asset>" pattern expanding to N place actions at stepped positions, execute as one undo group. Done When: `run_intent('place 3 trees in a row')` yields 3 entities with distinct positions and a SINGLE undo reverting all 3; the LLM seam reuses the same Plan validation.

**7. GPU timestamp instrumentation + frame-budget HUD (GPU Loop, M, 66).**
*Why blocking:* you cannot hit or defend a 16.6ms budget you cannot measure per-pass; today telemetry is wall-clock `frame_time_ms` only, conflating CPU sort, upload, GPU work, and present. This extends provability from correctness to performance — "headless pixel-asserted" becomes "headless ms-asserted", a defensible perf-regression gate every later GPU gap can prove against. *Current:* zero `TimestampQuery`/`QuerySet` usage anywhere; all devices request `Features::empty()`. *Needed:* enable `wgpu::Features::TIMESTAMP_QUERY` on the shared device, wrap each graph pass's record with `timestamp_writes`, resolve into a query buffer, surface per-pass GPU ms in the editor HUD. *First slice (agent task):* in `GpuContext` creation enable `TIMESTAMP_QUERY` with graceful fallback (mirror the GI fallback pattern); instrument the splat-raster pass with a begin/end pair, resolve, print "raster: X.X ms (GPU)". Done When: `cargo run --bin ochroma_editor -- --frames 30` prints a per-pass GPU-ms line with a plausible non-zero value, and a headless test asserts the resolved delta is >0 and < a generous ceiling for the fixed scene. (Depends on #2; unblocks measured Done-Whens on #3, #8, #20.)

**8. Resident buffers for in-frame GPU→GPU handoff (GPU Loop, L, 65).**
*Why blocking:* `device.poll(Maintain::Wait)` is a hard CPU stall (judges verified 23 `Maintain::Wait` sites); doing it per GI step and per LOD select inside a frame fully serializes CPU and GPU — the opposite of the async overlap AAA depends on. At 60fps there is no budget for even one full sync. *Current:* `GpuGi::step` and `atom_budget_gpu::select` both `map_async` + poll and hand back CPU `Vec`s every call. *Needed:* in the integrated path, GI writes its lit-splat result into a GPU storage buffer the rasterizer binds directly — no map, no poll, no Vec; the readback path stays ONLY for headless validation. *First slice (agent task):* in `crates/vox_render/src/gpu/spectral_gi.rs`, add `GpuGi::step_resident(&self, splats_buf:&wgpu::Buffer) -> wgpu::Buffer` running the same compute pass on the shared device but leaving the result in a storage buffer; bind that output as the rasterizer's splat input in the 2-node graph from #12. Done When: a headless test asserts the resident path's output buffer, read back ONCE at the end, is bit-identical to the existing `step()` Vec result — proving zero-readback handoff preserves verified semantics. (Depends on #2.)

**9. Play-in-Editor: bridge the editor to EngineLoop (Engine API, L, 65).**
*Why blocking:* AAA iteration IS press-play-in-editor; today the editor shows a static viewport and the game runs in a totally separate 3606-line binary (engine_runner.rs). A team cannot test gameplay without leaving the tool and launching another process — capping iteration at minutes-per-loop, which thousands of iterations cannot absorb. *Current:* two disjoint winit binaries; `ochroma_editor.rs` never instantiates `EngineLoop`. *Needed:* the editor embeds an EngineLoop in a Play/Pause/Stop state machine ticking into the SAME docked viewport texture, with an authoritative snapshot to restore on Stop (reuse the existing UndoStack + world save). *First slice (agent task):* in `crates/vox_app/src/bin/ochroma_editor.rs`, add an `EngineLoop` field + `PlayState{Editing,Playing,Paused}`; Play snapshots the world (reuse save), constructs `EngineLoop::new(EngineConfig{enable_editor:true,..}, SystemMask::game_minimal())`, and each Playing frame calls `step_scripts/step_physics` and blits into the docked Viewport; Stop restores. Done When: `ochroma_editor --frames 60 --play --shot out.png` shows a script-driven entity at a different position than frame 0, pixel-asserted headless. (Depends on #1 for the snapshot/restore seam.)

**10. Behavior tree that ticks real game logic (Gameplay, L, 63).**
*Why blocking:* AAA game AI (guards, companions, enemies) runs behavior trees that call perception/navigation/combat each tick; Ochroma's BT evaluates correctly but its leaves only look up a pre-filled `HashMap<String,BTStatus>` — nothing executes — and it is wired into zero binaries. Repeater ignores its count; no Cooldown/Wait/Parallel/Retry decorators. As-is it cannot drive a single autonomous NPC. *Current:* `BTNode::Action(String)` returns `ctx.action_results.get(name)` (behavior_tree.rs:105); blackboard is `HashMap<String,String>`. *Needed:* leaves invoke a registered `fn(&mut Blackboard,&World)->BTStatus`, a typed blackboard, Running-state persistence across ticks, and the missing decorators — then wire one NPC in walking_sim to patrol+investigate+flee through it, reusing the LIVE spectral-perception flee signal as a Condition leaf (fuses with the unique wedge). *First slice (agent task):* in `crates/vox_core/src/behavior_tree.rs`, add an action/condition registry and Running-persistent `BTContext`; port the walking_sim NPC flee logic to a 3-node tree `Selector[ flee-if-fire-near, patrol ]`. Done When: `cargo test -p vox_core bt_npc` asserts that with fire radiance above threshold the tree returns the flee Action and the NPC velocity points away from the player (exact computed direction), and below threshold it returns the patrol waypoint.

---

## 4. What We Don't Do (explicitly rejected)

- **Console ports (PS5/Xbox/Switch).** Zero console references in the tree; requires NDA SDKs, devkits, and cert teams. Steam/PC is the only realistic solo platform (Stability dim). One target, cashed in on Windows first.
- **Anti-cheat / competitive netcode hardening.** Rollback netcode exists as a generic mechanism, but kernel-level anti-cheat and matchmaking are solo-infeasible and off-wedge. Determinism stays a single-player/co-op asset, not an esports claim.
- **300-person-team tooling** (asset DB at studio scale, review pipelines, perforce-style locking, MMO-grade live ops). The audience is small AI-assisted teams; the AI collaborator (#6, #13) is our substitute for headcount.
- **Generic Unreal/Unity feature parity** (Lumen/Nanite clones, full PBR-material editor, mesh-first workflows). Binding directive: "SOTA our way." We win on spectral + splat-native + provable + AI-native, not by re-implementing RGB-engine features that are already best-in-class elsewhere.
- **Mobile / WASM splat runtime as a near-term target.** vox_web is a clear-color demo (splat pipeline "pending" per its own docstring). Mobile thermal/driver matrix is a distraction from the desktop GPU frame loop that everything else depends on.
- **Full FBX/Maya/Max native importers.** glTF + USD cover the modern interchange path; chasing legacy DCC formats is low wedge-leverage versus deepening glTF fidelity (#31).
- **Fixed-point determinism rewrite.** f32/f64 throughout; determinism is the game's responsibility via the rollback seam. A full deterministic-math rewrite is a multi-month tax with no wedge payoff at this stage.

---

## 5. Sequencing (dependency-ordered, 4 phases)

**Phase 1 — Floors you can start tomorrow on this exact codebase.** Everything here is independently startable and unblocks the rest; nothing depends on architecture not yet present.
- #4 Push + CI green (turns the wedge from self-attested to verifiable — costs a secret + a push).
- #1 Editor open/save/save-as (the hard floor under all editor work; #9, #17, #25 depend on it).
- #2 Unify on one `GpuContext` (the hard floor under all GPU-loop work; #3, #5, #7, #8, #12, #20 depend on it).
- #16 In-editor profiling HUD and #22 save versioning + #28 atomic-save/crash-hook (cheap honesty/durability wins, no deps).
*Unlocks:* a tool that holds a project, a single device buffers can flow through, and an externally-verified green gate to defend everything that follows.

**Phase 2 — The integrated GPU frame loop + measurement.** Now that one device exists (#2):
- #7 GPU timestamps (so every later GPU gap has a measured Done-When).
- #12 Render graph becomes a GPU executor (`GpuPass`, one encoder, one submit).
- #3 Wire the tiled GPU rasterizer as the viewport path.
- #8 Resident buffers (kill the per-frame `poll(Wait)` stalls).
*Unlocks:* the first honestly-measured sustained GPU frame at scale — the thing "2.05M splats" does not yet prove. This is the prerequisite for any wedge mechanic at frame rate.

**Phase 3 — The wedge made playable, on the now-resident loop.**
- #5 GPU relight kernel (bit-exact twin, on baked radiance) — depends on #2/#8.
- #34 Reflectance/emission data-model split (makes relight a pure multiply, physically honest) — de-risks and completes #5.
- #6 Multi-step Ask Ochroma + #13 real LLM backend (the AI collaborator).
- #9 Play-in-Editor + #10 behavior tree + #11 script host API (turn the editor into a place you author and test real gameplay) — #9 depends on #1.
- #20 GPU residency manager + #30 streaming hardening (scale the relit world past one small scene).
*Unlocks:* "capture-relight-populate" as an in-editor, frame-rate, AI-assisted loop — the actual product thesis.

**Phase 4 — Shippability + content supply at scale.**
- #19 Windows build + Steam packaging, #21 device-lost recovery + min-spec, #18 soak harness (the "crash-free 20–60h" contract).
- #26/#29 real 3DGS training backend + dense capture (the content supply for captured hero assets — spectral 3DGS training would be a genuine first).
- #27/#31/#33 DCC iteration loop, full glTF fidelity, validation gates (the daily art-team edit loop).
- #14 runtime game-state persistence, #15 extensible scheduler, #23 scaffolder, #32 facade docs, #35 game-UI framework, #36 ORCA crowds (the breadth to fill a 20–60h experience).
*Unlocks:* a game a small team can actually finish and ship on Steam, with externally-provable quality.

> Critical-path spine: **#4 (verifiable) → #2 (one device) → #7/#12/#3/#8 (measured resident frame) → #5/#34 (relight playable) → #20/#30 (at scale) → #19/#18 (shippable).** Editor (#1→#9/#17/#25) and AI (#6/#13) run as parallel tracks that rejoin at Phase 3.

---

## Sources

Engines and prior art the dimension agents cited for AAA targets:
- **Unreal Engine 5.5 / Unity 6** — GPU-driven rendering (GPU does cull+LOD+draw-arg generation, CPU kicks indirect), bindless residency, async-compute overlap, stable 16.6ms on mid hardware (GPU Loop dim).
- **UE Render Dependency Graph (RDG) / Frostbite FrameGraph** — frame graphs as the command-recording layer: transient texture allocation, barrier tracking, single-submit-per-frame (GPU Loop dim, #12).
- **Detour-crowd / RVO2 / ORCA** — reciprocal velocity-obstacle local avoidance + dynamic navmesh tiles as the 2026 AAA crowd target (Gameplay dim, #36).
- **Postshot / Polycam / Luma / Scaniverse (2025–26)** — full SfM + 3DGS training with densification, SH, and PSNR/SSIM quality gates as the capture-quality bar Ochroma's sparse path falls below (Content + Wedge dims, #26/#29).
- **3D Gaussian Splatting (Kerbl et al.)** — the densify/prune/opacity-reset/gradient-descent training loop and GPU tile-based depth-sorted rasterization the splat-native path must match (GPU Loop + Content dims, #3/#26).
- **Steamworks SDK / steamworks-rs / steamcmd depot upload** — the (non-Rust-native) PC shipping path; plan steamworks-rs binding or DRM-free first (Stability dim, #19).
- **wgpu** portability (Vulkan→GL fallback, `downlevel_defaults`, `TIMESTAMP_QUERY`, `TEXTURE_BINDING_ARRAY`/`PARTIALLY_BOUND` bindless) — the platform/feature substrate (GPU Loop + Stability dims).
- **bevy_ecs 0.16** — `Schedule`/`add_systems` already paid for but hidden at the EngineLoop level (Engine API dim, #15).
- Internal grounding: `FEATURES.md` (live/library/experimental inventory), the repo's gap-analysis doc ("25fps" Spectra path, "not tested at real scale"), and direct code spot-checks cited inline per gap.
