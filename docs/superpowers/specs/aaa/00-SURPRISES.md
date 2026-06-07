# AAA Program — Surprises Register

The user wants surprises as a first-class output. This register collates every spec's surfaced surprises and advantages into a deduplicated, grouped list. Each is tagged with the spec(s) that surfaced it. Groups: **the big free win** · **wedge synergies** · **cheap-because-of-existing-capability** · **first-mover angles** · **reuse** · **honest costs surfaced** · **grounding corrections that de-risked the seed**.

---

## ★ The biggest surprise: a general first-mover capability gained for free

**Teaching openusd-rs to parse text arrays (commit `9fd19fa`) — done to unblock Crucible's cooked-scene import — ALSO upgraded the whole engine's USD import to read text `.usda` geometry.** *(surfaced by Spec 01)*

The USDA *text* parser in the openusd-rs sibling previously skipped `[ … ]` array and `( … )` tuple attribute values entirely (`parse_array`/`parse_tuple` returned `vt::Value::empty()`), so cooked-scene geometry — `point3f[] points`, `int[] faceVertexIndices`, `float3[] extent`, `normal3f[] normals`, `texCoord2f[]` — silently imported as **nothing**, surfacing an `UnsupportedTextArray`-class failure downstream. Commit `9fd19fa` ("feat(usda): typed array & tuple parsing in the text parser", verified `git -C ~/src/openusd-rs show 9fd19fa`) now decodes typed arrays and tuples into the **same `vt` value the binary `.usdc` path yields**, so any consumer `try_get`s identical types regardless of source format. Why this is the headline surprise:

- It was motivated by one narrow need (Crucible reading its own cooked `.usda` scenes), but because `vox_usd` is built on this exact sibling (`vox_usd/Cargo.toml:12`, "USD scene import = live" per `FEATURES.md:70`), **the entire engine's USD import gained the ability to read text `.usda` geometry natively** — not just binary `.usdc`.
- It is a genuine first-mover capability: FEATURES.md already claims "no major engine reads USD scenes natively into splats"; this extends that to the human-readable, hand-editable, diff-able, version-controllable text USD format — the one artists and pipelines actually author and review by hand.
- It was hardened in the same commit (no unwraps on parsed input, no length-header pre-allocation from a dishonest count, bounded recursion, malformed values propagate as `Err` instead of silently misparsing a stage) — so the free capability arrived provability-clean, not as tech debt.
- **It is also the reason Spec 01 caught the CI gap.** Grounding the "spectra+crucible PAT" seed against the real manifests surfaced that openusd-rs is a THIRD non-optional sibling no CI job checks out (`grep -c openusd .github/workflows/*.yml` → 0/0). The free capability and the CI-correctness catch are the same discovery.

---

## Wedge synergies (the spectral data model paying off across specs)

- **Metamerism is a literally-passing assertion no RGB engine can reproduce.** The HUD receipt `daylight: ΔsRGB 0.009 (metamer) · cool_led: ΔsRGB 0.041 (forgery)` is a screenshot whose *second number is structurally 0.000 for any RGB pipeline* — an RGB engine stored one triple at capture, so it yields exactly zero divergence under any relight. The demo *is* the competitive proof. *(Spec 03; echoed by Spec 02's metamer-survival test, Spec 11's "16-band GI buffer bound resident" claim)*
- **The spectral-validity import lint is a wedge-aligned capability disguised as hygiene.** "Your asset has a non-finite / all-zero radiance band" is a check no RGB engine can offer (no per-band radiance to inspect), and it is the *exact* precondition the relight kernel needs (a zero-radiance splat NaN-poisons `relight_scene`'s intrinsic recovery). Building the mundane import gate de-risks the crown-jewel feature for free. *(Spec 10, fusing with Specs 02/03)*
- **Duplicate composes with relight automatically because the data model is radiance-uniform.** Duplicated splats carry their baked `spectral[16]` bit-exact; the moment the GPU relight kernel lands, a stamped-out forest of duplicates ALL relight correctly with no per-copy bake. *(Spec 09 × Spec 02)*
- **Spectral splats survive the save round-trip; RGB-engine transforms-only saves carry no radiance.** A persisted `project.ochroma_world` is human-readable JSON containing the literal 16-band radiance per splat — a capture/relight project you can diff, version, and inspect in a text editor. No mainstream engine's scene file carries per-primitive spectral data. *(Spec 06)*
- **Play-in-Editor for a spectral world is a stage no RGB engine has.** Because the editor and game share one `EngineLoop`, the relight mechanic and spectral AI perception run in the editor viewport the instant they're wired — "press-play into a metameric scene" is structurally impossible for an RGB engine. *(Spec 12 × Spec 02)*
- **The resident GI→raster seam is the host for the relight kernel.** Once the GI output buffer is bound resident as a rasterizer input, the relight compute pass (Spec 02) drops in between GI-dispatch and the pack pass on the *same encoder* — no redesign. *(Spec 11 × Spec 02)*

---

## Cheap-because-of-existing-capability (specs cheaper than their effort estimate)

- **The relight oracle is already factored into exactly the two pure functions the port needs.** `derive_intrinsic` (`relight.rs:349`) and `reilluminate_one` (`:385`) are allocation-free, `self`-free, band-indexed free functions in the precise op order a WGSL `for b in 0..16` reproduces — the port is a near-mechanical transcription, arguing M not L. *(Spec 02)*
- **The metamer proof already exists as runnable asserted code.** `relight_breaks_metamers` (`relight.rs:869`) and `metamer_pair()` (`:828`) already compute the 0.012/0.03 ΔsRGB bounds via `forward_rgb`. Step 1 is promoting a test body to a public function — the hard kernel-math risk is already committed and green. *(Spec 03)*
- **`metamer_pair()` is a ready-made demo-asset generator** — it already searches for a sharp metamer pair invisible under neutral and maximally divergent under cool_led, so the "two forgery canvases" are a function call away, no authoring. *(Spec 03)*
- **wgpu 24's `Device`/`Queue` already derive `Clone` (Arc-backed handles).** `GpuContext::from_parts` builds from the backend's existing `device()`/`queue()` accessors with a cheap clone — no restructuring of `WgpuBackend`; only one field (`adapter_info`) added. The riskiest-looking part (touching the present path every frame depends on) collapses to an accessor-only change. *(Specs 04, 11)*
- **The CPU oracle for the genuinely-new `TileRangeBuildPass` exists verbatim** at `spectra_render.rs:460-471` — the XL gap's riskiest unknown is its cheapest first step, with an oracle available on day one. *(Spec 05)*
- **`WorldSave` is already production-grade and bit-exact-proven** — `world_save_test.rs::test_full_world_round_trip_spectral_and_prefab` asserts whole-world structural equality (`assert_eq!(loaded, original)`). The save gap is the shell↔WorldSave bridge, not the serializer. *(Spec 06)*
- **The LLM safety surface is already built and free to inherit.** `resolve_intent`/`parse_llm_intent`/`SchemaContext` validate every model action against the live schema with `deny_unknown_fields` + clamp-at-edge; a `Plan` of validated leaves adds ZERO new safety surface. *(Spec 07)*
- **The editor's present pass is already a single isolated `begin_render_pass` with `timestamp_writes: None` at the exact injection point** (`ochroma_editor.rs:248`) — adding GPU measurement is a one-field swap, not new pass plumbing. *(Spec 08)*
- **The `poll(Wait)` + `map_async` + readback dance is already written** in `GpuGi::step` (`spectral_gi.rs:621-645`) — the timestamp readback is the same pattern on an 8-byte buffer. *(Spec 08)*
- **The whole Play-in-Editor gap is cheaper than scored (L → realistically M):** every API needed (composable `EngineLoop` sub-steps, `gather_splats_system`, `transform_splat`/`apply_transform`, `render_scene_rgba_with` compositing an arbitrary splat slice) already exists and is tested. The work is mostly calling `engine_runner`'s functions from a second site. *(Spec 12)*
- **Snapshot/restore is a cheap `Vec` clone, not a bevy-World deep clone.** The editor has no bevy World — its authored state is two `Clone` `Vec`s; restore is two `=` assignments. *(Spec 12)*
- **The validation gate's "exactly one error" test needs no file I/O** — `GaussianSplat::volume` takes `position:[f32;3]` directly and stores it raw, so a NaN survives construction; the unit test is one constructor call. *(Spec 10)*
- **A committed, deterministic relight fixture already exists** (`assets/relight_demo.vxm`, 4096 splats) plus the `relight_100k_cost_budget` scene generator — the GPU relight test needs zero new asset authoring. *(Spec 02)*

---

## First-mover angles (provability-as-product the competition can't match)

- **No RGB engine has a runtime spectral relight kernel** because no RGB engine stores 16 bands — the metamer-survival test makes the moat a passing assertion. *(Spec 02)*
- **No competitor markets "headless, ms-asserted per-pass GPU budget gates in CI."** Combined with the 5 bit-exact twins, an ms-asserted gate turns "we're fast" into "here is the test that fails if a commit regresses the raster past X ms" — provability extended from correctness to performance. *(Spec 08)*
- **No RGB engine has a "16-band spectral GI buffer bound resident into a tile rasterizer, proven pixel-identical to its readback oracle"** — shipping the residency *with* its bit-identity proof is a defensible claim competitors structurally cannot make. *(Spec 11)*
- **The provability culture moves from the laptop to GitHub's runners** — a reviewer can open the run in a browser and see a green check on the blitz HEAD; provability stops being self-attested. *(Spec 01)*
- **The on-device "Nanite for splats" seam:** AtomBudgetSelector (cull/LOD, done) + GPU tiled raster (Spec 05) fuse the two halves of the splat wedge in one frame — no RGB engine has a *spectral* tile-binned splat rasterizer. *(Spec 05)*
- **First-mover tidiness:** Play reuses the exact `EditorPlayMode::{Editing,Playing,Paused}` vocabulary already on the old `EditorState`, so the eventual editor unification is a mechanical merge, not a concept reconciliation. *(Spec 12)*

---

## Reuse (the house pattern instantiated, not reinvented)

- **The 5 shipped GPU twins are copy-pasteable harnesses.** `gpu_gi_matches_cpu_step_for_large_strided_scene` gives the scene-build + per-band-max-delta + `try_gpu` graceful-skip + `eprintln` pattern; `gpu_selection_exactly_equals_cpu_oracle` gives the multi-config sweep + assert-above-measured-bound idiom. The relight twin instantiates the house style on a 6th kernel. *(Spec 02; the `OCHROMA_GI=gpu` graceful-fallback idiom reused by Specs 08, 11, 12)*
- **`plant_asset` is the universal asset on-ramp.** Trees/terrain/buildings/Crucible scenes all funnel through one core handling overlay insertion + monotonic naming + range-tracked undo + viewport invalidation + receipts. Duplicate, save-load replay, and multi-step plan execution all inherit "one PlacedAsset undo per asset" by calling it. *(Specs 06, 07, 09)*
- **The range-shift undo machinery already solves the hard half of grouped/duplicate undo** — `PlacedAsset` undo drains an exact `[start,end)` range and shifts later entries down (`mod.rs:969-975`, tested at `:3618`); duplicating N items and undoing one-at-a-time is the exact interleaved-range scenario it was built for. *(Specs 07, 09)*
- **The `--frames N --shot` headless proof harness** (`ochroma_editor.rs` + `non_background_fraction` + `write_png`) is reused verbatim by the relight demo, the resident-raster proof, and Play-in-Editor — the Done-Whens need no new proof infrastructure. *(Specs 03, 11, 12)*
- **All three tiled passes already take `&mut CommandEncoder` + a raw `splat_buf: &Buffer`** — they were designed for encoder-chaining, so the one-encoder/one-submit resident chain needs zero signature changes. *(Specs 05, 11)*
- **`ImportResult.warnings: Vec<String>` already exists and is already displayed** (propagated through `import_and_cache` → `ImportedAsset.warnings`) — the zero-spectral lint needs zero new reporting plumbing. *(Spec 10)*
- **The `replay-through-plant_asset` trick gives undo-after-load for free** — a freshly-loaded world is immediately fully undoable with correct range-tracking, a feature that falls out of honoring the existing chokepoint. *(Spec 06)*
- **The menu surfaces a new command for free** — `menu_bar` iterates the registry by category, so one `Command::new` in category "Edit" lights up the menu, the Ctrl+D shortcut, AND the Ctrl+K palette with zero menu-code changes. *(Spec 09)*
- **`GpuGiPass::new(&device, …)` cleanly separates pass-construction from device-creation** — so `new_with_context` is ~6 lines reusing the pass builder; the same seam exists in all 5 other twins, making the rollout mechanical. *(Spec 04)*
- **An existing 120-frame integration test is a near-exact template** for the Play-in-Editor proof (`engine_loop_integration.rs:36` already drives `step_scripts`+`step_physics` with a stateful script over 120 frames) — and it's also the site to copy the gathering spawn tuple from. *(Spec 12)*
- **`prompt_schema_pins_every_variant` is a compiler-enforced safety net** — adding a `Plan` variant won't compile until the prompt text, serde DTO, and validator all agree, so the AI's action set can never silently drift from what the model is told. *(Spec 07)*

---

## Honest costs surfaced (advantages disguised as costs, and real costs flagged)

- **The directional `Sun` relight path (n_dot_l + BVH shadow rays) is genuinely harder and explicitly deferred** — a GPU shadow twin must reuse `splat_rt_gpu` rather than naively port `transmittance` (the 100k+shadow CPU pass measures ~7s). Slice 1 sidesteps it by mirroring the ambient-only config the headline metamer claim already uses. *(Spec 02)*
- **`GpuSplatFull` carries only 8 of 16 spectral bands** — fine for the `>10% non-black` proof (luminance survives), but the spec must NOT claim full 16-band GPU rasterization; widening to 16 bands is a follow-on paired with relight (#5) and the reflectance split (#34). *(Spec 05)*
- **The two `CameraUniform` layouts diverged** between `tile_assign.wgsl` and `splat_raster.wgsl` — not a blocker (the renderer builds both from one `RenderCamera`), but merging them would mutate two validated shaders, so the spec chooses the non-invasive path. *(Spec 05)*
- **The band-count seam (16→8) is a real gotcha the seed omits** — copying `splats_to_gpu`'s first-8 rule keeps bit-identity; implementing the seed verbatim without `GiToRasterPack` would bind a mismatched buffer or diverge from the proven raster. *(Spec 11)*
- **The `engine_runner.rs:build_world_save` (line 1858) hardcodes `splats: Vec::new()`** — the game binary also throws splats away today; the same `SavedSplatGeom` codec can later fix it at near-zero marginal cost, but that is out of scope and must not be silently bundled. *(Spec 06)*
- **Two real implementation blockers caught by adversarial verification** (not author surprises, but program-critical): Spec 11's headline command can't honor `--gpu-resident`/`--frames` until argv parsing is added to `scale_trial.rs`; Spec 12's enter-Play spawn tuple produces zero rendered splats until it carries the asset-registration component `gather_splats_system` queries. Both are surfaced in those specs' `## Verification corrections`. *(Specs 11, 12 via the verifier)*

---

## Grounding corrections that de-risked the seed (the roadmap seeds were SEEDS, not gospel)

Each of these is a place where grounding the code caught a wrong seed *before* it became a wrong implementation:

- **The CI seed said "spectra+crucible PAT" — but a THIRD non-optional sibling (openusd-rs) is checked out by zero CI jobs.** Following the seed literally yields a red run dying at `vox_usd`'s manifest load. The fix is now a diagnosed edit, not a blind watch-and-guess. *(Spec 01)*
- **The relight seed assumed a reflectance channel to read; `GaussianSplat` stores baked radiance only** (`types.rs:40`) — so the GPU pass must derive the per-splat intrinsic on-device, exactly as the CPU oracle does. *(Spec 02)*
- **The metamer seed's `metamer_divergence(a,b,l1,l2)` is ill-posed without a *reference* illuminant** — `derive_intrinsic` divides by the assumed capture SPD; the honest engine function takes `(reference, illum_a, illum_b)`, with the simple 4-arg face preserved for the designer. *(Spec 03)*
- **The Ask-Ochroma seed said reuse `IntentAction::AddNode` — but that adds a node-graph node, not a world entity.** "Add 5 trees" must plant via `plant_grown_tree`, a path the parser never reached; a new `PlantTree` leaf is required. The seed's "row of N" also hid a stacking bug (`plant_grown_tree` always plants at one fixed origin) — fixed by a per-step offset, asserted as exact x-coords `[-4,0,4,8,12]`. *(Spec 07)*
- **The GPU-timestamp seed said "instrument the splat-raster pass" — but that pass has zero frame callers**, so it would produce a *dead* editor line. The fix instruments the egui present pass (the pass that actually runs) and ships the harness the tiled raster wraps later. *(Spec 08)*
- **The duplicate seed assumed a `ShellEntity` knows its overlay range — it does not** (`ShellEntity` is `{name,kind,pos}`); the entity→range index had to become a first-class field, which is precisely the seam save/load (#1) and crash recovery (#25) both need. *(Spec 09)*
- **The Play-in-Editor seed named `EngineLoop::tick`, `EditorPlayMode` on the windowed editor, and a bevy World to snapshot — all three are wrong.** There is no `tick` (composable sub-steps instead), `EditorPlayMode` lives on the *old* immediate-mode editor, and the windowed editor has no bevy World. The corrected design is *more* tractable than the seed implied. *(Spec 12)*
- **The save/load seed assumed `SavedSplat`↔`GaussianSplat` conversion exists — it does not** (zero hits outside `world_save.rs`), and a naive map would silently drop disk geometry and double-round f16. The fix is a versioned full-fidelity `SavedSplatGeom` carrying raw f16 bits. *(Spec 06)*
