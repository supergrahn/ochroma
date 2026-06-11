# AI Asset Factory v1 — Brief → Directive → Sandbox Cook → Pixel Gates → Pool Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A closed-loop asset factory in the GAME layer: one brief drives `llama-cli` (schema-constrained) to a Forge directive, the directive is semantically validated, cooked deterministically into a sandbox, pixel-gated through the Spectra path tracer, retried ≤ 4 times with verbatim reject feedback, and on accept registered into the live zonable pools with a pool-proof line and a provenance record. Vision judge is v2 — v1 prints `judge: skipped (v1)`.

**Done When:** Both of these hold, verified by a human at the keyboard without reading code:

**Accept path.** Running

```bash
cd ~/Ochroma/projects/civitas_care && \
LD_LIBRARY_PATH=$HOME/slang-sdk/lib SPECTRA_SLANG_DIR=$HOME/src/spectra/slang \
SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
cargo run --release --features spectra --bin asset_factory -- \
  "weathered brick rowhouse for a medium-density residential block, level 3, 3x4 bucket"
```

prints, in order (real values, exact line shapes):

```
attempt 1/4: directive city.res_med.l3.3x4.<name> (schema-valid, sampling seed <s>)
semantics: PASS (0 kill criteria)
cook: cooked city.res_med.l3.3x4.<name>  <V> mesh verts -> <M> atoms, <W> walls
gate city.res_med.l3.3x4.<name>/iso: mean luma <x> (band 60..120), channel std <a>/<b>/<c> (floor 22) -> PASS
gate city.res_med.l3.3x4.<name>/front: mean luma <x> (band 110..150), channel std <a>/<b>/<c> (floor 35) -> PASS
gate city.res_med.l3.3x4.<name>/side: mean luma <x> (band 60..120), channel std <a>/<b>/<c> (floor 15) -> PASS
gates: PASS
judge: skipped (v1)
ACCEPTED city.res_med.l3.3x4.<name> after 1 attempt(s)
registered: pack.json ResidentialMed L3 pool 3 -> 4 assets
pool-proof: zonable_for(ResidentialMed, 3, <w> x <d>) selects city.res_med.l3.3x4.<name> at seed <k>
provenance: assets/source/buildings/generated/city.res_med.l3.3x4.<name>.provenance.json
```

with `<V>`, `<M>`, `<W>` non-zero, attempt count ≤ 4 (more attempts acceptable as long as the final line set is the accept set), and a human opens `assets/buildings/forge_starter/renders/factory/city.res_med.l3.3x4.<name>/iso.png` and sees a brick rowhouse — not a black, washed, or flat frame.

**Reject path (selftest, no LLM run, no GPU).** Running

```bash
cd ~/Ochroma/projects/civitas_care && \
ASSET_FACTORY_SELFTEST=collide cargo run --bin asset_factory -- "any brief"
```

prints exactly

```
attempt 1/1: directive forge.house.craftsman (schema-valid, replay)
semantics: FAIL — K2 asset_id collides with existing catalog asset forge.house.craftsman
REJECTED: kill criterion K2, 0 retries permitted in selftest
```

and exits 1 — the factory's analog of `FORGE_GATE_SELFTEST=1`. (The same env on the `--features spectra` build prints the same three lines.)

**Architecture:** The LLM emits *directives only* (JSON, one schema — never geometry); the directive is the determinism boundary. Stage chain per attempt: `LlmRunner` (llama-cli `--json-schema-file`, per-attempt `--seed`) → `factory::validate` (serde shape + catalog-aware kill criteria K1–K4) → `factory::cook` (existing `game_asset_cook` subprocess with `--source/--output` + new `--no-starters`, K5/K10) → `qa::pathtrace_gates::gate_views` (the forge_pathtrace gate machinery factored into a lib module, K6) → register (quarantine → `generated/`, full recook, pool-proof, provenance). Single-threaded; every stage a sequential subprocess (LLM and path tracer share the 780M iGPU). Retryable rejects feed their reason **verbatim** into the next prompt; K8 (budget) and K9 (model collapse) are terminal.
**Design Document:** `docs/superpowers/specs/2026-06-10-ai-asset-factory-design.md`
**Tech Stack:** Rust edition 2024 (civitas_care), serde/serde_json, image 0.25 (PNG decode in gate tests), sha2 0.10 (new dev-dep, determinism hashes), llama.cpp b9372 (`~/git/llama.cpp/build/bin/llama-cli`) + `~/models/gemma4-e4b.gguf`, Spectra path tracer via `vox_render/spectra-native` (game feature `spectra`).
**Build:** `cd ~/Ochroma/projects/civitas_care && cargo build` / `cargo test --lib <filter>`. Spectra builds/runs need the env block in **Done When**. Cook subprocess needs Forge + AssemblyPrime discoverable (existing `GAME_FORGE_BIN`/`OCHROMA_FORGE_BIN`, `GAME_ASSEMBLYPRIME_PROJECT`/`OCHROMA_ASSEMBLYPRIME_PROJECT` discovery — already working on this box).

---

## IMPORTANT NOTES

- **Repos:** game `~/Ochroma/projects/civitas_care` (git master, uncommitted), engine `~/src/ochroma` (branch `blitz/day1-foundation`, uncommitted), `~/src/forge` (**NO git**), `~/src/spectra` (git, uncommitted). **Do NOT run `git commit` anywhere** — the user commits. Template commit steps are replaced by **"leave uncommitted"**. This plan touches ONLY `~/Ochroma/projects/civitas_care` (GAME layer); engine crates untouched.
- **Coordination gates (check before starting the gated tasks):**
  - **Task 8** refactors `src/bin/forge_pathtrace.rs` — the porch-closure agent is releasing that file about now. Do not open it until that track is complete (its plan checked off / no other agent holds the file).
  - **Any `--features spectra` build compiles `vox_render` via the path dep** (`vox_render = { path = "../../../src/ochroma/crates/vox_render" }`). The SDF M1 track (Tasks 1–2 + Group B) is executing in `vox_render` NOW — do not run spectra-feature builds (Task 8 acceptance, Task 11) until it completes. Tasks 1–7, 9, and Task 10's featureless selftest need no spectra build and are not gated.
- **Test discipline:** always use targeted filters (`cargo test --lib factory_llm -- --nocapture`), never bare `cargo test` over everything. Never run the full `cargo test -p vox_render --lib` suite — it has ONE pre-existing flake (spectral_gi resident seam, radix_sort raster nondeterminism); this plan never needs it.
- **Verified existing signatures (code-checked 2026-06-10; code wins over the design doc):**
  - `game_asset_cook.rs` (5,989 lines, post-remediation): `config_from_args() -> Result<CookConfig, CookError>` at **line 236** (design says :135 — stale); today it accepts only `--output PATH`, `--source PATH`, `--help`. `CookConfig { output_dir: PathBuf, source_dir: PathBuf }`.
  - `AssetDirective` (bin-private today, lifted to the lib by Task 1): `{ version: u32, asset_id: String, display_name: String, arch_style: ArchStyle, usage: DirectiveUsage, level: u8, footprint_m: [f32; 2], footprint_bucket: Option<String> (serde default), floors: [u32; 2], gameplay: DirectiveGameplay, forge: DirectiveForge, palette: DirectivePalette, kind: CookedKind, variants: Vec<Value> (serde default) }`. Unknown keys are tolerated (no `deny_unknown_fields`) — the live directives carry a `quality_notes` array the struct never sees.
  - `DirectiveUsage`: `#[serde(tag = "kind", rename_all = "snake_case")] enum { Zonable { zone: ZoneType }, Ploppable { category: PloppableCategory } }` — JSON shape is `{"kind": "zonable", "zone": "ResidentialMed"}`.
  - `DirectiveForge`: `{ seed: u64, style: String, width: f32, depth: f32, floors: u8, floor_height: f32 (default 3.0), footprint_shape: String (default "Rectangular"), roof_style: String (default "Flat"), hero_level: String (default "supporting"), channels: Vec<String> (default STRUCTURE/ORNAMENT/DETAIL), window_density: f32 (default 0.62), generate_interior: bool (default false), condition: String (serde default "New") }` — **`condition` lives on `forge`** (`default_condition()`, validated against `FORGE_CONDITIONS: ["New", "Aged", "Weathered", "Derelict"]` in `recipe_from_directive`). This RESOLVES the design's open question; the live `rowhouse_01.asset.json` variant 2 patches `{"forge": {"condition": "Aged"}}`.
  - `expand_directive_variants(raw: Value) -> Result<Vec<DirectiveExpansion>, CookError>` (line 517): RFC-7396 merge patches applied to the **raw** Value before deserialization; variants do not recurse (`variants` key removed from each clone); **variant `asset_id` is force-overwritten to `{base}.v{n}` AFTER the patch** — a patch touching `asset_id` is silently ignored by the cook, so K3's "variant patch touches asset_id" MUST inspect the raw patch object, not the expansion result. `merge_patch` (line 555): object patches merge recursively, `null` deletes, everything else replaces wholesale.
  - `recipe_from_directive(directive: AssetDirective) -> Result<AssetRecipe, CookError>` (line 618) already rejects: version ≠ 1, product-named ids (`contains("civitas") || starts_with("civ_")`), non-positive `footprint_m`, `forge.width <= 1.0 || forge.depth <= 1.0 || forge.floors == 0`, unknown condition.
  - `validate_quality(recipe: &AssetRecipe, payload: &ReadyAssetPayload, source: &AssemblyPrimeExport) -> Result<(), CookError>` (line 3141) runs **inside** the cook — floating-rooftop-atom gate, flat-roof seated-equipment-kit gate, vegetation/ground-coverage gates. The factory inherits all of it for free via the subprocess (K5).
  - Cook stdout contract per recipe (line 126): `cooked {:24} {:4} mesh verts -> {:3} atoms, {} walls, source {}` plus two extra lines (`materials: N from zones, M fallback roles`, `plot: ground ... tris ...`); flora summary `flora: ...`; final `wrote <output_dir>`. `CookError::Directive(msg)` displays as `asset directive error: {msg}`.
  - `run()` cooks `starter_recipes()` (5) + `starter_variants` (House-kind seed-shifted clones) + `load_directive_recipes(source_dir)` (recursive `*.asset.json` scan), then **unconditionally** `cook_flora` (4 Forge vegetation runs). `clean_generated_pack_dirs` wipes `atoms/usd/requests` but **never `textures/`** — the pinned PolyHaven store under `<output>/textures/polyhaven` is what `TextureCache` samples; **a fresh sandbox output dir has no store, so the factory must symlink the main store into the sandbox before cooking** or sandbox atom tints diverge from the final recook's (see Task 6).
  - `AssetCatalog` (`src/asset/mod.rs:546`): `packs: Vec<AssetPack>` is **pub**; `zonable_for(&self, zone: ZoneType, level: u8, plot_w: f32, plot_d: f32, seed: u64) -> Option<&BuildingAsset>` — pool = assets where `is_zonable(zone) && level == level && fits(w, d)`, indexed `seed % pool.len()`; **falls back to ignoring `fits` when nothing fits**. `load_project_assets() -> Self` (vanilla packs + recursive `assets/` pack scan); `get(&self, id) -> Option<&BuildingAsset>` (searches packs **rev**). `BuildingAsset.fits` = `footprint[0] <= plot_w + 0.5 && footprint[1] <= plot_d + 0.5`. `max_level(zone) -> u8`: 3 for ResidentialLow/CommercialLocal/IndustrialLight/Agricultural/Park, 4 for ResidentialMed/MixedUse/IndustrialHeavy, 5 for ResidentialHigh/CommercialRegional/Office.
  - Gate machinery in `forge_pathtrace.rs` (1,420 lines, spectra-feature bin; Task 8 factors it): `pixel_stats(pixels: &[u8]) -> (f32, [f32; 3], f32)` (mean luma, per-channel std, lit ratio; luma = 0.2126R+0.7152G+0.0722B, lit > 12.0) — **pure CPU, no feature gate needed**; `GateBand { mean_lo, mean_hi, std_floor }`; `gate_band(label: &str, view: &str) -> GateBand` — iso (60..120, floor 22), front (110..150, floor 35), side (60..120, floor 15), **panics on unknown view**; `GATE_HARD_MIN_SPP: u32 = 16` (mean band ADVISORY below, std floors hard at every spp; factory renders at spp 32 so everything is hard); gate print: `gate {label}: mean luma {:.1} (band {:.0}..{:.0}), channel std {:.1}/{:.1}/{:.1} (floor {:.0}) -> {PASS|FAIL|ADVISORY...}`; summary `gates: PASS` / `gates: FAIL <labels>` (exit 1) — this printed contract must survive the refactor **verbatim**. Render path: `load_ready_asset_mesh(path) -> Result<(ReadyAssetPayload, Mesh), String>`, `door_anchor`, `with_lot_dressing`, `ready_textured_materials(payload, material_ids, &mut TextureCache)`, `render_mesh_view(... rig: &LightRig, black_level, out, spp) -> Result<Vec<u8>, String>` (errors if mean < 5.0 or lit < 1.5%), `review_rig()`, `camera_for`, `inspection_views` (iso/front/side/porch/debug_*; W=H=768, fov 42°). Candidate gating uses **iso/front/side only** (+ `debug_mat` as a non-gated artifact); porch is starter-specific.
  - `vox_render::splat_backend::pathtrace_mesh_lit_to_rgba(positions, normals, uvs, indices, material_ids, materials: &[PbrMaterial], textures: &[TextureImage], eye: [f32;3], target: [f32;3], fov_radians: f32, width, height, spp, rig: &LightRig) -> Result<Vec<u8>, String>` — do not alter; engine stays untouched.
  - Seed bin `asset_directive_from_llm.rs` (210 lines, retired in Task 4): `extract_json_object(text: &str) -> Option<&str>` (balanced-brace, string/escape-aware — move it verbatim WITH its unit test); defaults `DEFAULT_LLAMA_CLI = "/home/tom-espen/git/llama.cpp/build/bin/llama-cli"`, `DEFAULT_MODEL = "/home/tom-espen/models/gemma4-e4b.gguf"`, env overrides `GAME_ASSET_LLAMA_CLI`, `GAME_ASSET_LLM_MODEL` — carry all four over unchanged.
  - `llama-cli` b9372 flags (verified in `--help` today): `-jf, --json-schema-file FILE` (no external `$ref`s allowed), `-s, --seed SEED`, `-n`, `--temp`, `--no-display-prompt`. v1 invocation: `llama-cli -m <model> -p <prompt> --json-schema-file <schema> --seed <s> -n 2600 --temp 0.22 --no-display-prompt`. Keep the schema inside llama.cpp's json-schema→grammar subset: `type/enum/const/required/properties/items/pattern/min*/max*` only, no `$ref`.
- **New API contracts (implement EXACTLY; design §5/§6 reconciled with code):**
  - `civitas_care::asset::directive` (Task 1, lifted DTOs): fields stay **`pub`** — these are serde value objects crossing the lib→bin boundary; this is a documented exception to the private-fields house rule. Everything else below uses **private fields + accessors**.
  - `LlmRunner` enum: `LlamaCli { cli: PathBuf, model: PathBuf, n_predict: u32, temp: f32 }`, `Replay { transcripts: Vec<PathBuf>, cursor: AtomicUsize, seen_prompts: Mutex<Vec<String>> }`. **No `Remote` variant in v1** (deferred with the judge to v2). `LlmRunner::from_env() -> Self`; `generate_directive(&self, prompt: &str, sampling_seed: u64) -> Result<serde_json::Value, LlmError>`; `LlmError { Spawn(String), Status(String), NoJson, Parse(String) }` — every variant maps to K1.
  - `pub fn validate_semantics(directive: &serde_json::Value, catalog: &AssetCatalog) -> Result<(), Vec<Rejection>>` — pure, never panics on hostile input; `Ok(())` means "safe to cook".
  - `Rejection { criterion: KillCriterion, reason: String }` (accessors `criterion()`, `reason()`); `enum KillCriterion { K1, ..., K10 }` with `Display` = `"K1"`..`"K10"`.
  - `pub fn cook_candidate(directive_path: &Path, sandbox: &Path) -> Result<CookedCandidate, String>` — `Err` text goes verbatim into feedback (K5/K10). `CookedCandidate` accessors: `asset_id()`, `atoms_path()`, `atom_count()`, `mesh_verts()`, `walls()`, `directive_path()`, `variant_ids()`.
  - `pub fn gate_views(payload: &Path, out_dir: &Path, spp: u32) -> Result<Vec<GateStat>, String>` in `civitas_care::qa::pathtrace_gates`, `#[cfg(feature = "spectra")]`; `GateStat` (unconditional type): accessors `view()`, `mean()`, `stds()`, `pass()`, plus `print_line(&self, label_prefix: &str)` reproducing the binary's gate-line format.
  - `pub fn register_accepted(candidate: &CookedCandidate, brief: &str, attempts: &[AttemptRecord]) -> Result<RegistrationProof, String>`; `RegistrationProof` accessors `pool_before()`, `pool_after()`, `proof_seed()`, `provenance_path()`.
  - `AttemptRecord { index: u32 /* 1-based */, sampling_seed: u64, directive: Option<Value>, outcome: AttemptOutcome }`; `AttemptOutcome { SchemaReject(Vec<String>), SemanticReject(Vec<Rejection>), CookReject(String), GateFail(Vec<GateStat>), Accepted { asset_id: String } }` (**no `JudgeReject` in v1**); `FactoryOutcome { Accepted { asset_id, attempts, pool_before, pool_after }, Rejected { terminal: KillCriterion, attempts } }`; `ProvenanceRecord { brief, model, llama_build, attempts, accepted_id, gate_stats }` (serde Serialize+Deserialize).
  - `pub fn run_factory(brief: &str, cfg: &FactoryConfig) -> Result<FactoryOutcome, FactoryError>` — `#[cfg(feature = "spectra")]`, blocking, main thread only (iGPU shared by LLM and path tracer; never concurrent). Internally delegates to a featureless `run_loop(brief, cfg, runner, catalog, stages: FactoryStages) -> Result<(FactoryOutcome, Vec<AttemptRecord>), FactoryError>` where `FactoryStages` carries `cook`, `gate`, `register` closures — production passes the real `cook_candidate` / `gate_views` / `register_accepted`; hermetic tests pass canned closures (the assertions stay real: verbatim prompt content, attempt-record outcomes, provenance on disk).
- **Pinned numbers and formulas:**
  - Per-attempt sampling seed: `splitmix64(fnv1a64(brief) ^ attempt_index as u64)`. fnv1a64: basis `0xcbf2_9ce4_8422_2325`, prime `0x0000_0100_0000_01b3`, over the brief's UTF-8 bytes. splitmix64: copy the exact constants from `game_asset_cook.rs:579` (`0x9E37_79B9_7F4A_7C15`, `0xBF58_476D_1CE4_E5B9`, `0x94D0_49BB_1331_11EB`) as a private fn in `factory/mod.rs` (house precedent: private re-declaration, do not export the bin's).
  - **K10 atom cap = 37,596** (4 × 9,399, the craftsman base — the largest starter payload in today's post-remediation recook; rowhouse 5,451, school 6,491, police 4,235, victorian 3,670). `const FACTORY_ATOM_CAP: usize = 37_596;` in `factory/cook.rs`, with this derivation in a comment.
  - Footprint-bucket rule (K3): CS-style cell = 8 m; for bucket `"CxD"`, each `footprint_m` axis must be ≤ cells × 8.0 and > (cells − 1) × 8.0 (width↔C, depth↔D).
  - K4 coherence table (pinned to the cook's four kinds — the cook cannot cook anything else in v1): Zonable ResidentialLow → class `LowResidential`, kind ∈ {house}; Zonable ResidentialMed → `MediumResidential`, kind ∈ {house, rowhouse}; Zonable ResidentialHigh → `HighResidential`, kind ∈ {rowhouse}; **any other zone → K4** (reason names the four v1 kinds); Ploppable Education → `Civic` + kind school; Ploppable Safety → `Civic` + kind police; **any other category → K4**. Zonable `level` must be 1..=`max_level(zone)`; ploppable `level` must be 0.
  - Enum universes for the schema file (verified): ZoneType = ResidentialLow, ResidentialMed, ResidentialHigh, CommercialLocal, CommercialRegional, IndustrialLight, IndustrialHeavy, Office, MixedUse, Agricultural, Park (11); BuildingClass = LowResidential, MediumResidential, HighResidential, Commercial, Office, Industrial, Civic, Park (8); PloppableCategory = Childcare, Eldercare, Health, Safety, Education, Park, Unique (7); ArchStyle = Victorian, Georgian, Tudor, Gothic, Mediterranean, Colonial, Craftsman, Modern, Brutalist, Industrial (10); kind (CookedKind, snake_case) = house, rowhouse, school, police (4); forge.condition = New, Aged, Weathered, Derelict (4); forge.hero_level ∈ {background, supporting} (Hero is Forge's own remote path — out of scope).
  - Attempt budget default 4, env `ASSET_FACTORY_ATTEMPTS`. Quarantine dir `assets/factory/candidates/` (deliberately OUTSIDE `assets/source/buildings`, which the cook scans recursively). Accepted directives + provenance land in `assets/source/buildings/generated/`. Factory renders at spp 32 to `assets/buildings/forge_starter/renders/factory/<asset_id>/`.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.

---

## File Map

| Action | Path (under `~/Ochroma/projects/civitas_care` unless noted) | Responsibility |
|--------|------|----------------|
| Create | `src/asset/directive.rs` | directive DTOs + `FORGE_CONDITIONS` + `expand_directive_variants`/`merge_patch` lifted from the cook bin; lib tests incl. schema roundtrip |
| Modify | `src/asset/mod.rs` | register `pub mod directive;` |
| Modify | `src/bin/game_asset_cook.rs` | import lifted types; `--no-starters` flag (skips starters, starter variants, AND flora) |
| Create | `assets/source/prompts/asset_directive.schema.json` | single-source-of-truth JSON Schema (post-remediation shape: `variants`, `forge.condition`) with embedded example; feeds `--json-schema-file` |
| Modify | `assets/source/prompts/asset_directive_prompt.md` | document `variants` merge-patches, `forge.condition`, the retry feedback block |
| Create | `src/factory/mod.rs` | `pub mod` decls; `AttemptRecord`/`AttemptOutcome`/`FactoryOutcome`/`FactoryConfig`/`FactoryError`/`ProvenanceRecord`; seed derivation (fnv1a64 + splitmix64); `run_loop` + `run_factory`; loop tests |
| Create | `src/factory/llm.rs` | `LlmRunner` (LlamaCli + Replay), `LlmError`, `extract_json_object`, `build_prompt` with verbatim feedback block; tests |
| Create | `src/factory/validate.rs` | `KillCriterion`, `Rejection`, `validate_semantics` (K1 deser + K2–K4); kill-criteria table tests |
| Create | `src/factory/cook.rs` | `CookedCandidate`, `cook_candidate` (sandbox + store symlink + subprocess + parse), `FACTORY_ATOM_CAP` (K10); parse + `#[ignore]` determinism tests |
| Create | `src/factory/register.rs` | quarantine→generated move, full recook, `pack_pool_size`, `pool_proof` sweep, provenance write, `register_accepted`; hermetic pool/provenance tests |
| Create | `src/qa/mod.rs` | register `pub mod pathtrace_gates;` |
| Create | `src/qa/pathtrace_gates.rs` | gate machinery factored from forge_pathtrace: unconditional `pixel_stats`/`GateBand`/`gate_band`/`GateStat`; spectra-gated render path + `gate_views`; gate tests |
| Modify | `src/bin/forge_pathtrace.rs` | thin caller of `qa::pathtrace_gates`; `COOKED_SPECS`, porch/contact-sheet/selftest stay; printed contract unchanged (**gated on porch-agent release**) |
| Create | `src/bin/asset_factory.rs` | orchestrator binary: brief + env knobs, Done-When line set, `ASSET_FACTORY_SELFTEST=collide` path; featureless build supports selftest only |
| Delete | `src/bin/asset_directive_from_llm.rs` | retired — helpers absorbed into `factory::llm` (same task) |
| Create | `tests/fixtures/factory/replay_good.txt` | canned chatty llama-cli stdout containing one valid post-remediation directive |
| Create | `tests/fixtures/factory/replay_bad_geometry.txt` | canned stdout whose directive has `forge.width: 200.0` (K3 fodder) |
| Create | `tests/fixtures/factory/collide_directive.json` | canned directive with `asset_id: forge.house.craftsman` (selftest + K2 test) |
| Modify | `Cargo.toml` | `[dev-dependencies] sha2 = "0.10"` |
| Modify | `src/lib.rs` | register `pub mod factory;` and `pub mod qa;` |

---

## Capabilities

All factory unit tests are **hermetic** (Replay backend reads canned fixtures; no test shells a model or a GPU). Two explicitly non-hermetic tests are `#[ignore]`d and run by name in acceptance: the cook-determinism double-cook (needs Forge + AssemblyPrime) and the live llama smoke.

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Directive shape lives in the lib | `cargo test --lib directive_ -- --nocapture`: the live `rowhouse_01.asset.json` deserializes through `asset::directive::AssetDirective`; expansion yields ids `[base, base.v1, base.v2]`, v2 has floors 4 + condition Aged (printed) | `assert!(parse(json).is_ok())` |
| Schema ↔ serde lockstep | `directive_schema_roundtrip`: the schema file's embedded example deserializes as `AssetDirective` AND every schema enum value deserializes into its Rust enum (counts printed: 11 zones, 10 styles, 4 conditions, 4 kinds) | checking the schema file parses as JSON |
| `--no-starters` sandbox cook | real cook run over a one-directive source dir prints exactly 3 `cooked city.…` lines, zero `cooked forge.…`/`cooked civic.…`, **no** `flora:` line, ends `wrote <out>` | asserting the flag parses |
| Directive extraction + K1 | `factory_llm_directive_from_replay`: chatty fixture (preamble + JSON + trailer) → extracted Value deserializes as `AssetDirective`, `asset_id` starts `city.`, both `footprint_m` > 0; id printed | `assert!(extract_json_object(s).is_some())` — passes on `{}` |
| Semantic kill criteria | `factory_validate_kill_criteria`: table of 6 poisoned directives (id collision, product id, forge dims > footprint, class/zone mismatch, forge.floors outside range, bucket/metric mismatch) each rejected with its named `K*` code, all six printed; clean renamed directive passes with `0 kill criteria` printed | `assert!(validate(d).is_err())` — one generic error passes |
| Retry feedback loop | `factory_loop_retry_feedback`: Replay yields `[bad_geometry, good]`; attempt-2 prompt (captured by the Replay runner) contains attempt-1's K3 reason **verbatim** (printed); outcome `Accepted` on attempt 2; exactly 2 `AttemptRecord`s with `[SemanticReject, Accepted]` | `assert_eq!(outcome, Accepted)` without feedback plumbing |
| Attempt budget + collapse | `factory_loop_exhausts_after_4`: 4 bad attempts → `Rejected { terminal: K8, attempts: 4 }`, provenance JSON on disk holds 4 records with per-attempt reasons; `factory_loop_detects_collapse`: identical directive twice → terminal K9 at attempt 2 (both printed) | `assert!(outcome.is_err())` |
| Cook determinism | `factory_cook_determinism` (`#[ignore]`): cook the fixture directive twice into two sandboxes; SHA-256 of both `atoms/<id>.atoms.json` byte-identical; **both hashes printed** | comparing `atoms.len()` |
| Gate reuse on real buffers | `qa_gate_black_frame_fails` / `qa_gate_flat_grey_fails`: synthetic 768² black buffer fails every view's mean floor, flat-grey-128 fails every std floor (per-view stats printed); `qa_gate_real_render_passes`: the committed `renders/spectra/craftsman/iso.png` decodes and lands inside the iso band (printed; skip-line if file absent) | `assert!(gate_result.is_ok())` |
| Registration + pool growth | `factory_register_pool_proof`: synthetic catalog, pool 3 → 4 after adding the asset; sweep `seed in 0..pool_len` selects the new id **exactly once** per cycle (seed printed) | `assert!(pack.assets.iter().any(...))` without the selection proof |
| Binary contract preserved | post-refactor `forge_pathtrace` spp-32 run prints `gates: PASS`; `FORGE_GATE_SELFTEST=1` run prints `gates: FAIL craftsman/front` and exits 1 — byte-identical line shapes to today | "it compiles" |
| Reject-path selftest | `ASSET_FACTORY_SELFTEST=collide cargo run --bin asset_factory -- "x"` prints the exact 3-line reject set and exits 1, with no LLM/GPU touched | exit code checked without the lines |

---

## Task 1: Lift the directive schema into the lib (`asset::directive`)

The factory must deserialize and expand directives without owning a copy of the schema — and the schema types are private to the 5,989-line cook bin today. Lift them; the cook re-imports.

**Files:**
- Create: `src/asset/directive.rs`
- Modify: `src/asset/mod.rs` (add `pub mod directive;`)
- Modify: `src/bin/game_asset_cook.rs` (delete moved items, import from the lib)

**Acceptance:** `cd ~/Ochroma/projects/civitas_care && cargo test --lib directive_ -- --nocapture` → prints `rowhouse_01: 2 variants -> ids [city.res_med.l3.3x4.rowhouse_01, …v1, …v2], v2 floors 4 condition Aged`; then `cargo test --bin game_asset_cook condition -- --nocapture` and `cargo test --bin game_asset_cook variant -- --nocapture` stay green with their existing printed values.

**Wiring requirement:** `directive_recipes` in `src/bin/game_asset_cook.rs` calls `civitas_care::asset::directive::expand_directive_variants` (the lib copy — the bin copy is **deleted**, not duplicated). `recipe_from_directive` consumes the lib `AssetDirective` and the lib `FORGE_CONDITIONS`. `todo!()` / empty bodies = **task failure**.

- [x] **Step 1: Write the failing test** — in `src/asset/directive.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn directive_lib_parses_live_rowhouse_and_expands_variants() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/source/buildings/residential/rowhouse_01.asset.json");
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("live directive exists")).unwrap();
    let expansions = expand_directive_variants(raw).expect("expansion succeeds");
    let ids: Vec<String> = expansions.iter()
        .map(|e| e.directive["asset_id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids, [
        "city.res_med.l3.3x4.rowhouse_01",
        "city.res_med.l3.3x4.rowhouse_01.v1",
        "city.res_med.l3.3x4.rowhouse_01.v2",
    ]);
    let v2: AssetDirective = serde_json::from_value(expansions[2].directive.clone()).unwrap();
    assert_eq!(v2.forge.floors, 4, "v2 patches forge.floors to 4");
    assert_eq!(v2.forge.condition, "Aged", "v2 patches forge.condition");
    let base: AssetDirective = serde_json::from_value(expansions[0].directive.clone()).unwrap();
    assert_eq!(base.forge.condition, "New", "base keeps the serde default");
    println!("rowhouse_01: {} variants -> ids {:?}, v2 floors {} condition {}",
        base.variants.len(), ids, v2.forge.floors, v2.forge.condition);
}
```

- [x] **Step 2: Run to verify it fails** — `cargo test --lib directive_ 2>&1 | tail -5` → FAIL: `E0433` (module `asset::directive` does not exist).
- [x] **Step 3: Implement** — move VERBATIM from `game_asset_cook.rs` into `src/asset/directive.rs`: `AssetDirective`, `DirectiveUsage` (+ `to_asset_use`), `DirectiveGameplay` (+ `to_attributes`), `DirectiveForge`, `DirectivePalette`, `CookedKind`, all serde default fns (`default_condition`, `default_forge_floor_height`, `default_rectangular`, `default_flat_roof`, `default_supporting`, `channels`, `default_window_density_directive`), `FORGE_CONDITIONS`, `DirectiveExpansion`, `expand_directive_variants`, `merge_patch`. Make every struct field and the listed items `pub`. Change `expand_directive_variants` to return `Result<Vec<DirectiveExpansion>, String>` (the same message strings, unwrapped from `CookError`).
- [x] **Step 4: Wire at exact callsite** — in `game_asset_cook.rs`: delete the moved items; add `use civitas_care::asset::directive::{AssetDirective, CookedKind, DirectiveExpansion, expand_directive_variants, FORGE_CONDITIONS};`; in `directive_recipes`, wrap the lib error: `expand_directive_variants(raw).map_err(CookError::Directive)?`. Existing bin tests (`condition_*`, `*variant*`) keep passing untouched — they prove the lift changed nothing.
- [x] **Step 5: Run — verify non-trivial output** — the two acceptance commands above; the lib test prints the three ids and `floors 4 condition Aged` (not defaults).
- [x] **Step 6: Leave uncommitted** (the user commits).

---

## Task 2: Directive JSON Schema file + prompt update + roundtrip lockstep test

**Files:**
- Create: `assets/source/prompts/asset_directive.schema.json`
- Modify: `assets/source/prompts/asset_directive_prompt.md`
- Modify: `src/asset/directive.rs` (roundtrip test in the same tests module)

**Acceptance:** `cargo test --lib directive_schema_roundtrip -- --nocapture` → prints `schema example: city.res_med.l3.3x4.example_rowhouse (condition Weathered, 1 variant) deserializes; enums in lockstep: 11 zones, 10 arch styles, 8 classes, 7 categories, 4 kinds, 4 conditions`.

**Wiring requirement:** the schema file path is the one `LlmRunner::generate_directive` passes to `--json-schema-file` (Task 4 reads this exact path); the roundtrip test reads the file from `assets/source/prompts/asset_directive.schema.json` — no copies. Stub bodies = **task failure**.

- [x] **Step 1: Write the failing test** — `directive_schema_roundtrip`: load the schema file; deserialize `schema["examples"][0]` through `AssetDirective` AND run `expand_directive_variants` over it (it must carry exactly 1 variant patching `forge.condition`); then for every value in the schema's enums (`usage.zone`, `arch_style`, `gameplay.class`, ploppable `category`, `kind`, `forge.condition`) assert `serde_json::from_value::<RustEnum>(json!(value)).is_ok()` so a schema value the code can't parse fails CI; print the counts line.
- [x] **Step 2: Run to verify failure** — `cargo test --lib directive_schema_roundtrip 2>&1 | tail -5` → FAIL: file not found.
- [x] **Step 3: Implement the schema** — `type: object`, `required`: `[version, asset_id, display_name, arch_style, usage, level, footprint_m, footprint_bucket, floors, gameplay, forge, palette, kind]`; `version` const 1; `asset_id` pattern `^city\.[a-z0-9_.]+$`; `usage` with `kind` enum `[zonable, ploppable]` + `zone`/`category` enums (full universes from IMPORTANT NOTES); `level` integer 0–5; `footprint_m` 2-item number array, each 4–64; `floors` 2-item integer array, each 1–8; `gameplay` object (class enum, ranges non-negative); `forge` object (`seed` integer ≥ 0, `width`/`depth` 2–60, `floors` 1–8, `floor_height` 2.4–4.5, `window_density` 0–1, `condition` enum, `hero_level` enum `[background, supporting]`, `footprint_shape`/`roof_style`/`style` free strings, `channels` string array, `generate_interior` boolean); `palette` (three 3-item 0–1 arrays); `kind` enum; `variants` array of objects; `quality_notes` optional string array. Stay inside the llama.cpp grammar subset (no `$ref`, no `oneOf` gymnastics — `usage` is a plain object whose `zone`/`category` are both optional properties; the semantic validator enforces the pairing). Embed one full example under `"examples"` (id `city.res_med.l3.3x4.example_rowhouse`, condition `Weathered`, one variant patching `forge.condition` to `Aged`).
- [x] **Step 4: Wire the prompt** — extend `asset_directive_prompt.md`: document `forge.condition` (enum + meaning), `variants` (RFC-7396 merge patches; ids become `{asset_id}.v{n}` automatically; **never patch `asset_id`**), and the retry contract ("when a `Previous attempt failed:` block is present, fix exactly the quoted reasons and change nothing else gratuitously"). Update the embedded example to include `forge.condition` and one variant.
- [x] **Step 5: Run — verify non-trivial output** — acceptance command prints the full counts line.
- [x] **Step 6: Leave uncommitted.**

---

## Task 3: `--no-starters` flag on `game_asset_cook` (skips starters, starter variants, AND flora)

**Files:**
- Modify: `src/bin/game_asset_cook.rs` (`CookConfig`, `config_from_args`, `run`)

**Acceptance:**
```bash
cd ~/Ochroma/projects/civitas_care && rm -rf /tmp/factory_t3_{src,out} && mkdir -p /tmp/factory_t3_src && \
cp assets/source/buildings/residential/rowhouse_01.asset.json /tmp/factory_t3_src/ && \
mkdir -p /tmp/factory_t3_out/textures && ln -s ~/Ochroma/projects/civitas_care/assets/buildings/forge_starter/textures/polyhaven /tmp/factory_t3_out/textures/polyhaven && \
cargo run --bin game_asset_cook -- --source /tmp/factory_t3_src --output /tmp/factory_t3_out --no-starters
```
→ stdout contains exactly **3** `cooked city.res_med.l3.3x4.rowhouse_01` lines (base, `.v1`, `.v2`) with non-zero verts/atoms/walls, **zero** `cooked forge.` / `cooked civic.` lines, **no** `flora:` line, and ends `wrote /tmp/factory_t3_out`.

**Wiring requirement:** `CookConfig` gains `no_starters: bool`; in `run()` the `starter_recipes()` + `starter_variants(...)` block AND the `cook_flora(...)` call are skipped when set (the pack's `flora` vec is empty then). Default behavior (no flag) stays byte-identical. Empty bodies = **task failure**.

- [x] **Step 1: Write the failing test** — bin test `no_starters_flag_parses_and_default_is_off`: drive `config_from_args`-equivalent parsing (factor the arg loop into `config_from_iter(args: impl Iterator<Item = String>)` so it's testable without `std::env`); assert `--no-starters` sets the flag, absence leaves it false, and `--no-starters extra` still errors on `extra`; print both configs.
- [x] **Step 2: Run to verify failure** — `cargo test --bin game_asset_cook no_starters 2>&1 | tail -5` → FAIL: `E0425` (`config_from_iter` not found).
- [x] **Step 3: Implement** — `config_from_iter` + `"--no-starters" => config.no_starters = true`; update the `--help` usage string; in `run()`: `let mut recipes = if config.no_starters { Vec::new() } else { let mut r = starter_recipes(); r.extend(starter_variants(&r)); r };` then `recipes.extend(load_directive_recipes(...))`; guard `cook_flora` with `if config.no_starters { Vec::new() } else { cook_flora(&forge, &atoms_dir)? }`.
- [x] **Step 4: Wire at exact callsite** — `main`/`run` path uses `config_from_args()` which delegates to `config_from_iter(std::env::args().skip(1))`.
- [x] **Step 5: Run — verify non-trivial output** — the acceptance command (real Forge + AssemblyPrime run; the symlink line pre-seeds the texture store exactly as `factory::cook` will in Task 6). Count the lines: `... | grep -c '^cooked city'` = 3.
- [x] **Step 6: Leave uncommitted.**

---

## Task 4: `factory::llm` — LlmRunner (LlamaCli + Replay), extraction, feedback prompt; retire the seed bin

**Files:**
- Create: `src/factory/mod.rs` (module shell: `pub mod llm;` for now), `src/factory/llm.rs`
- Create: `tests/fixtures/factory/replay_good.txt`, `tests/fixtures/factory/replay_bad_geometry.txt`
- Modify: `src/lib.rs` (add `pub mod factory;`)
- Delete: `src/bin/asset_directive_from_llm.rs`

**Acceptance:** `cargo test --lib factory_llm -- --nocapture` → prints `replay directive: city.res_med.l3.3x4.gabled_brick_01 footprint 14x20 condition Weathered` and `feedback block carried verbatim: "K3 forge.width 200.0 exceeds footprint_m[0] 14.0"`; and `ls src/bin/asset_directive_from_llm.rs` → No such file.

**Wiring requirement:** `LlmRunner::generate_directive` is the ONLY place `llama-cli` is shelled; it passes `--json-schema-file assets/source/prompts/asset_directive.schema.json` (Task 2's file) and `--seed <sampling_seed>`. `extract_json_object` moves verbatim from the seed bin **with its existing unit test**. The seed bin is deleted in this same task. Stubs = **task failure**.

- [x] **Step 1: Write the failing tests** —
  - `factory_llm_directive_from_replay`: build `LlmRunner::Replay` over `tests/fixtures/factory/replay_good.txt` (write the fixture: 3 lines of chatty preamble, a complete valid directive JSON for `city.res_med.l3.3x4.gabled_brick_01` with `forge.condition: "Weathered"` and one variant, 2 lines of trailer); call `generate_directive("ignored", 7)`; deserialize through `asset::directive::AssetDirective`; assert id starts with `city.`, both `footprint_m` > 0, condition == "Weathered"; print the line.
  - `factory_llm_prompt_carries_feedback_verbatim`: `build_prompt(base, brief, &["K3 forge.width 200.0 exceeds footprint_m[0] 14.0".into()])` contains the exact string and the header `Previous attempt failed:`; `build_prompt(base, brief, &[])` contains neither; print the carried line.
  - move `extracts_first_json_object_from_chatty_output` from the seed bin unchanged.
- [x] **Step 2: Run to verify failure** — `cargo test --lib factory_llm 2>&1 | tail -5` → FAIL: `E0433` (`factory` module does not exist).
- [x] **Step 3: Implement** — per the IMPORTANT NOTES contract: `LlmRunner::{LlamaCli, Replay}`; `from_env()` (seed-bin defaults + `GAME_ASSET_LLAMA_CLI`/`GAME_ASSET_LLM_MODEL`, n_predict 2600, temp 0.22); `generate_directive` shells `llama-cli -m <model> -p <prompt> --json-schema-file <schema> --seed <s> -n 2600 --temp 0.22 --no-display-prompt`, captures stdout, `extract_json_object`, `serde_json::from_str` → `Value`; Replay reads `transcripts[cursor.fetch_add(1)]` and pushes the prompt into `seen_prompts` (accessor `seen_prompts() -> Vec<String>`); `build_prompt(base_prompt, brief, feedback: &[String])` = base + (if non-empty) `\n\nPrevious attempt failed:\n` + one `- <reason>` line per entry, verbatim + `\n\nAsset request:\n<brief>\n\nOutput exactly one JSON object now.\n`. Add `#[ignore] factory_llm_live_smoke` shelling the real `llama-cli` with the schema and a one-line brief, asserting the output parses as `AssetDirective` (run manually: `cargo test --lib factory_llm_live_smoke -- --ignored --nocapture`).
- [x] **Step 4: Wire** — `pub mod factory;` in `src/lib.rs`; delete `src/bin/asset_directive_from_llm.rs` (its CLI is subsumed by `asset_factory`); `cargo build` proves nothing else referenced it.
- [x] **Step 5: Run — verify non-trivial output** — acceptance prints real ids/strings, not placeholders.
- [x] **Step 6: Leave uncommitted.**

---

## Task 5: `factory::validate` — kill criteria K1–K4 over Value + catalog

**Files:**
- Create: `src/factory/validate.rs` (+ `pub mod validate;` in `src/factory/mod.rs`)
- Create: `tests/fixtures/factory/collide_directive.json` (valid directive, `asset_id: "forge.house.craftsman"`)

**Acceptance:** `cargo test --lib factory_validate -- --nocapture` → prints six lines, one per poisoned directive, each starting with its code — `K2 asset_id collides with existing catalog asset forge.house.craftsman`, `K2 …product-named…`, `K3 forge.width 200.0 exceeds footprint_m[0] 14.0`, `K3 forge.floors 9 outside floors range [3, 4]`, `K3 footprint_bucket 2x2 inconsistent with footprint_m [14.0, 20.0] (cell = 8 m)`, `K4 class MediumResidential incoherent with zone IndustrialHeavy (v1 kinds: house, rowhouse, school, police)` — plus `clean directive: PASS (0 kill criteria)`.

**Wiring requirement:** `validate_semantics(directive: &Value, catalog: &AssetCatalog) -> Result<(), Vec<Rejection>>` per the pinned contract; internally it MUST route through `asset::directive::expand_directive_variants` + `serde_json::from_value::<AssetDirective>` (deser failure → a `K1` Rejection), then apply K2–K4 to the base AND every variant expansion (variant id collisions count). Raw `variants` patches are checked as raw objects (non-object patch → K3; patch containing `asset_id` → K3 — the cook silently overwrites it, so the validator is the only honest place to catch it). Never panics on hostile input. Stubs = **task failure**.

- [x] **Step 1: Write the failing test** — `factory_validate_kill_criteria`: load the live rowhouse fixture as the template Value and a real `AssetCatalog::load_project_assets()`; build the 6-row poison table by mutating clones (collide id from `collide_directive.json`'s id; `civ_tower_01`; `forge.width = 200.0`; `usage.zone = IndustrialHeavy` keeping class MediumResidential; `forge.floors = 9`; `footprint_bucket = "2x2"` with footprint [14, 20]); assert each returns `Err` whose first `Rejection.criterion()` is the expected K-code and print every `"{code} {reason}"`; then a clean clone renamed to `city.res_med.l3.3x4.validator_clean_01` (id NOT in the catalog) returns `Ok(())` — print the PASS line. Add `factory_validate_variant_patch_asset_id_is_k3` (patch `{"asset_id": "city.sneaky"}` → K3, reason printed).
- [x] **Step 2: Run to verify failure** — `cargo test --lib factory_validate 2>&1 | tail -5` → FAIL: `validate_semantics` not found.
- [x] **Step 3: Implement** — `KillCriterion` (Display `"K1"`..`"K10"`), `Rejection`, and the checks in this exact order, accumulating ALL rejections (not first-only): K1 deser per expansion; K2 `catalog.get(id).is_some()` for base + every `{base}.v{n}`, product names (`contains("civitas") || starts_with("civ_")`); K3 geometry (`forge.width > footprint_m[0]`, `forge.depth > footprint_m[1]`, `forge.floors` outside `floors` `[min, max]`, the 8 m bucket rule from IMPORTANT NOTES when `footprint_bucket` is present, raw-patch rules); K4 the pinned coherence table + level bounds (`1..=max_level(zone)` zonable, `0` ploppable). One-line reasons, exactly the shapes shown in Acceptance — the same string goes to stdout, provenance, and the retry prompt.
- [x] **Step 4: Wire** — `pub mod validate;`; tests consume only the public API.
- [x] **Step 5: Run — verify non-trivial output** — all six codes + PASS line printed with real values.
- [x] **Step 6: Leave uncommitted.**

---

## Task 6: `factory::cook` — sandboxed deterministic cook, K5 + K10

**Files:**
- Create: `src/factory/cook.rs` (+ `pub mod cook;` in `src/factory/mod.rs`)
- Modify: `Cargo.toml` (`[dev-dependencies] sha2 = "0.10"`)

**Acceptance:** hermetic: `cargo test --lib factory_cook_parse -- --nocapture` → prints `parsed: city.res_med.l3.3x4.rowhouse_01 verts 1968 atoms 5451 walls 4` (the exact values from the pasted real transcript fixture string in the test) and `K10: 40000 atoms > cap 37596 -> reject`. Real (`#[ignore]`): `cargo test --lib factory_cook_determinism -- --ignored --nocapture` → prints two **identical** 64-hex SHA-256 lines for the double-cooked `atoms.json`.

**Wiring requirement:** `cook_candidate(directive_path, sandbox)`: (1) creates `<sandbox>/source/` containing ONLY the candidate file (copied), (2) creates `<sandbox>/out/textures/` and **symlinks** `assets/buildings/forge_starter/textures/polyhaven` to `<sandbox>/out/textures/polyhaven` (without this, sandbox atom tints bake from an empty store and diverge from the final recook — see IMPORTANT NOTES), (3) shells `cargo run --bin game_asset_cook -- --source <sandbox>/source --output <sandbox>/out --no-starters` from the project root, (4) parses every `cooked ` stdout line via `parse_cooked_line(line: &str) -> Option<(String, u32, u32, u32)>`, (5) non-zero exit → `Err(<last 4 KB of stderr + stdout, verbatim>)` (K5), (6) base `atom_count > FACTORY_ATOM_CAP (37_596)` → `Err("K10 cooked {n} atoms > cap 37596 (4x largest starter); reduce detail: fewer floors, lower window_density, smaller footprint")`. Returns `CookedCandidate` with the base row + `variant_ids`. Stubs = **task failure**.

- [x] **Step 1: Write the failing tests** — `factory_cook_parse_cooked_line` over a multi-line string literal pasted from a REAL cook transcript (run Task 3's acceptance once and paste: the 3 `cooked …` lines plus `materials:`/`plot:` noise lines that must be ignored); assert 3 parses with the exact numbers, noise lines parse to `None`; `factory_cook_k10_cap` calls the cap check helper with 40 000 → `Err` containing `"K10"` and `"37596"`.
- [x] **Step 2: Run to verify failure** — `cargo test --lib factory_cook 2>&1 | tail -5` → FAIL: `E0425`.
- [x] **Step 3: Implement** — per the wiring contract; sandbox path = `std::env::temp_dir().join(format!("asset_factory_{:016x}", splitmix64(nanos)))` (no rand crate; reuse the factory's private splitmix64). `parse_cooked_line` splits on whitespace after the `cooked ` prefix and reads `<id> <verts> mesh verts -> <atoms> atoms, <walls> walls, source …`.
- [x] **Step 4: Wire** — `pub mod cook;`; the `#[ignore]` determinism test calls the REAL `cook_candidate` twice on a copy of `rowhouse_01.asset.json` renamed to `city.res_med.l3.3x4.determinism_probe` (avoid an id that confuses a human reading sandbox dirs), hashes both `out/atoms/<id>.atoms.json` with sha2, asserts byte-identical, prints both hex digests.
- [x] **Step 5: Run — verify non-trivial output** — hermetic filter green with printed values; then the `--ignored` run (Forge + AssemblyPrime, ~minutes) printing two equal hashes.
- [x] **Step 6: Leave uncommitted.**

---

## Task 7: `factory::register` — pool proof, provenance, registration

**Files:**
- Create: `src/factory/register.rs` (+ `pub mod register;` in `src/factory/mod.rs`; `ProvenanceRecord` lives in `src/factory/mod.rs` with the other records)

**Acceptance:** `cargo test --lib factory_register -- --nocapture` → prints `pool ResidentialMed L3: 3 -> 4 assets` and `pool-proof: zonable_for(ResidentialMed, 3, 14 x 20) selects city.res_med.l3.3x4.probe at seed 3 (each pool member selected exactly once over seeds 0..4)` and `provenance roundtrip: 2 attempts, reasons survive`.

**Wiring requirement:** three public pieces, all consumed by `register_accepted`: `pack_pool_size(pack: &AssetPack, zone: ZoneType, level: u8) -> usize` (assets where `is_zonable(zone) && a.level == level`); `pool_proof(catalog: &AssetCatalog, zone, level, plot_w, plot_d, new_id: &str) -> Option<u64>` (sweep `seed in 0..pool_len as u64`, return the first seed where `zonable_for` returns `new_id` — and assert in tests every pool member is selected exactly once across the sweep); `write_provenance(path: &Path, record: &ProvenanceRecord) -> Result<(), String>`. `register_accepted(candidate, brief, attempts)` then: move the directive `assets/factory/candidates/<id>.asset.json` → `assets/source/buildings/generated/<id>.asset.json` (`fs::create_dir_all` + `fs::rename`), shell the standard full recook (`cargo run --bin game_asset_cook` — default args, registers the asset in `pack.json`), reload `assets/buildings/forge_starter/pack.json` for before/after pool counts (before-count captured pre-recook), build `AssetCatalog::load_project_assets()`, run `pool_proof` with the directive's `footprint_m`, write `<id>.provenance.json` next to the directive (model path + `gemma4 7.5B E4B` label, `llama_build: "b9372 (8ad8aef44)"`, all `AttemptRecord`s, final gate stats). Stubs = **task failure**.

- [x] **Step 1: Write the failing tests** — `factory_register_pool_proof`: build a synthetic `AssetCatalog::new(vec![pack])` whose pack holds 3 ResidentialMed L3 rowhouse-like `BuildingAsset`s (footprint [14, 20]) plus the new `city.res_med.l3.3x4.probe`; assert `pack_pool_size` 4 (and 3 before adding), sweep seeds 0..4 asserting each id selected exactly once and `pool_proof` returns the probe's seed; print both lines. `factory_register_provenance_roundtrip`: write a `ProvenanceRecord` with 2 attempts (one `SemanticReject` carrying a K3 reason, one `Accepted`), read it back, assert the reason string survives byte-identical; print.
- [x] **Step 2: Run to verify failure** — `cargo test --lib factory_register 2>&1 | tail -5` → FAIL: `E0425`.
- [x] **Step 3: Implement** — per the wiring contract. The full-recook subprocess path of `register_accepted` is NOT exercised by unit tests (it is exercised by Task 11's live run); everything else is.
- [x] **Step 4: Wire** — `pub mod register;`; `register_accepted` composes exactly the three public pieces (no duplicate pool logic).
- [x] **Step 5: Run — verify non-trivial output** — acceptance lines with real seeds/counts.
- [x] **Step 6: Leave uncommitted.**

---

## Task 8: `qa::pathtrace_gates` — factor the gate machinery out of `forge_pathtrace.rs` (GATED)

**GATE 1: do not start until the porch-closure agent has released `src/bin/forge_pathtrace.rs`.** **GATE 2: the acceptance GPU runs build `--features spectra`, which compiles `vox_render` — wait for the SDF M1 Group B track to finish in `~/src/ochroma/crates/vox_render`.** The hermetic tests in this task need neither gate's GPU run but DO touch the same file — respect Gate 1 for the whole task.

**Files:**
- Create: `src/qa/mod.rs`, `src/qa/pathtrace_gates.rs`
- Modify: `src/lib.rs` (add `pub mod qa;`), `src/bin/forge_pathtrace.rs`

**Acceptance:** (a) `cargo test --lib qa_gate -- --nocapture` → prints per-view stats for the synthetic frames (`black frame: iso mean 0.0 < floor 60 -> FAIL`, `flat grey: front std 0.0/0.0/0.0 < floor 35 -> FAIL` shapes) and `real render craftsman/iso: mean <60..120> std <≥22> -> PASS` (or the printed skip line if the PNG is absent). (b) Binary contract: the Done-When env + `FORGE_PREVIEW_SPP=32 cargo run --release --features spectra --bin forge_pathtrace` prints `gates: PASS`, and the same with `FORGE_GATE_SELFTEST=1` prints `selftest: corrupted craftsman/front buffer…` then `gates: FAIL craftsman/front` and exits 1 — line shapes byte-identical to the pre-refactor binary.

**Wiring requirement:** unconditional in `qa::pathtrace_gates`: `pixel_stats`, `GateBand`, `gate_band(label, view)` (iso/front/side/porch/debug bands moved verbatim, including the craftsman-porch special case and the unknown-view panic), `GATE_HARD_MIN_SPP`, `GateStat` (+ `print_line`), `gates_summary(stats: &[GateStat], spp: u32) -> bool`. Spectra-gated (`#[cfg(feature = "spectra")]`): `review_rig`, `apply_rig_env`, `load_ready_asset_mesh`, `door_anchor`, `with_lot_dressing`, `TextureAtlas`, `ready_textured_materials`, `render_mesh_view`, `grade_preview`, `save_png_rgba`, `camera_for`, `structure_aabb`, `standard_views(mesh, door) -> Vec<InspectionView>` (iso/front/side ONLY — the candidate set), and `gate_views(payload: &Path, out_dir: &Path, spp: u32) -> Result<Vec<GateStat>, String>` (loads payload → dressing → textured materials → renders iso/front/side + a non-gated `debug_mat.png` → stats per view → returns all three GateStats; does NOT print the summary — callers do). `forge_pathtrace.rs` keeps `COOKED_SPECS`, porch views/rig, `print_door_contrast`, contact sheet, the selftest corrupt-one-buffer trick, and the final summary print — but imports every moved item from `civitas_care::qa::pathtrace_gates`. **Its printed contract must not change by one byte.** Stubs = **task failure**.

- [ ] **Step 1: Write the failing tests** — `qa_gate_black_frame_fails` (vec![0u8; 768*768*4] with alpha 255 → for iso/front/side: `mean < band.mean_lo`, stat `pass() == false`; print each), `qa_gate_flat_grey_fails` (all channels 128 → std floors fail every view even though front's mean 128 is in band — the exact wash-detector property; print), `qa_gate_real_render_passes` (decode `assets/buildings/forge_starter/renders/spectra/craftsman/iso.png` with the `image` crate → `pixel_stats` → inside the iso band; `println!("skip: no committed render at <path>")` + return if absent).
- [ ] **Step 2: Run to verify failure** — `cargo test --lib qa_gate 2>&1 | tail -5` → FAIL: `E0433` (`qa` does not exist).
- [ ] **Step 3: Implement** — move the items listed in Wiring (cut from `forge_pathtrace.rs`, paste, make `pub`, keep doc comments); `GateStat` wraps the old `GateResult` fields plus the band; `print_line` reproduces the binary's `gate {label}: …` format string exactly (copy it, don't retype).
- [ ] **Step 4: Wire at exact callsite** — `forge_pathtrace.rs::run()` builds its views (full set incl. porch) but computes stats via `qa::pathtrace_gates::pixel_stats` + `gate_band` + `GateStat`, and prints via `print_line` + the existing summary block. `pub mod qa;` in `src/lib.rs`.
- [ ] **Step 5: Run — verify non-trivial output** — acceptance (a) without features (hermetic), then acceptance (b) on the GPU (both gates cleared). Compare the run's gate lines against a pre-refactor log if available.
- [ ] **Step 6: Leave uncommitted.**

---

## Task 9: `factory::run_loop` / `run_factory` — the attempt loop with verbatim feedback, K8/K9

**Files:**
- Modify: `src/factory/mod.rs` (records, config, errors, seed derivation, `run_loop`, `run_factory`)

**Acceptance:** `cargo test --lib factory_loop -- --nocapture` → prints `attempt 2 prompt carries: "K3 forge.width 200.0 exceeds footprint_m[0] 14.0"`, `outcome: Accepted city.res_med.l3.3x4.gabled_brick_01 after 2 attempts, records [SemanticReject, Accepted]`, `exhausted: K8 after 4 attempts, provenance has 4 records`, `collapse: K9 at attempt 2 (identical directive despite new seed)` — all real values from the Replay fixtures.

**Wiring requirement:** `run_loop(brief, cfg, runner: &LlmRunner, catalog: &AssetCatalog, stages: FactoryStages)` per the pinned contract: for `attempt in 1..=cfg.attempts()`: seed = `splitmix64(fnv1a64(brief) ^ attempt)`; prompt = `build_prompt(base, brief, &feedback)`; `generate_directive` (Err → K1 `SchemaReject`, reasons pushed to feedback); **K9 check**: canonical `serde_json::to_string` of the directive equals the previous attempt's → terminal `Rejected { terminal: K9 }`; `validate_semantics` (Err → `SemanticReject`, every reason pushed verbatim); write candidate to `cfg.quarantine_dir()/<id>.asset.json`; `(stages.cook)(…)` (Err → `CookReject`, text pushed verbatim — K5 and K10 both arrive here); `(stages.gate)(…)` (any `!pass()` → `GateFail`, per-view stat lines pushed); print `judge: skipped (v1)`; `(stages.register)(…)` → `Accepted`. Budget exhausted → `Rejected { terminal: K8 }` AND a provenance record written to `cfg.quarantine_dir()/<brief-slug>.provenance.json` with every `AttemptRecord`. `run_factory` (`#[cfg(feature = "spectra")]`) wires `FactoryStages { cook: cook_candidate, gate: gate_views @ spp 32 into renders/factory/<id>/, register: register_accepted }` — the real functions, nothing else. Stubs = **task failure**.

- [x] **Step 1: Write the failing tests** — `factory_loop_retry_feedback` (Replay `[replay_bad_geometry.txt, replay_good.txt]`, canned cook/gate/register closures returning success for the good directive; assert `runner.seen_prompts()[1]` contains the K3 reason **verbatim** and `seen_prompts()[0]` does not; outcome + records as in Acceptance); `factory_loop_exhausts_after_4` (Replay with the bad fixture 4×... use 4 distinct bad fixtures or 4 copies with different `forge.seed` values so K9 doesn't fire first — vary a byte per transcript; assert K8, 4 records, provenance file exists with 4 attempts); `factory_loop_detects_collapse` (the SAME bad transcript twice → K9 at attempt 2). All hermetic — no subprocess, no GPU.
- [x] **Step 2: Run to verify failure** — `cargo test --lib factory_loop 2>&1 | tail -5` → FAIL: `run_loop` not found.
- [x] **Step 3: Implement** — records + config + loop per the contract; `FactoryConfig::from_env()` reads `ASSET_FACTORY_ATTEMPTS` (default 4), schema/prompt/quarantine paths.
- [x] **Step 4: Wire** — `run_factory` passes the three real stage functions (exact names: `crate::factory::cook::cook_candidate`, `crate::qa::pathtrace_gates::gate_views`, `crate::factory::register::register_accepted`).
- [x] **Step 5: Run — verify non-trivial output** — acceptance lines with the real verbatim string.
- [x] **Step 6: Leave uncommitted.**

---

## Task 10: `asset_factory` binary — Done-When line set + `ASSET_FACTORY_SELFTEST=collide`

**Files:**
- Create: `src/bin/asset_factory.rs`

**Acceptance (hermetic, ungated):** `cd ~/Ochroma/projects/civitas_care && ASSET_FACTORY_SELFTEST=collide cargo run --bin asset_factory -- "any brief"` prints exactly the three reject-path lines from **Done When** and exits 1 (`echo $?` → 1). No LLM started, no GPU touched, no spectra build needed.

**Wiring requirement:** the bin compiles in BOTH feature modes. Featureless: `main` handles ONLY the selftest path (otherwise prints `asset_factory: build with --features spectra for the full loop` and exits 2). With `spectra`: full path calls `factory::run_factory` and prints the accept/reject line sets — the attempt line (`attempt {i}/{budget}: directive {id} (schema-valid, sampling seed {s})`), `semantics: PASS (0 kill criteria)` / `semantics: FAIL — {code} {reason}`, the `cook: cooked {id}  {V} mesh verts -> {M} atoms, {W} walls` echo (values from `CookedCandidate`), gate lines via `GateStat::print_line` with label `{id}/{view}`, `gates: PASS|FAIL`, `judge: skipped (v1)`, `ACCEPTED {id} after {n} attempt(s)`, `registered: pack.json {Zone} L{level} pool {before} -> {after} assets`, `pool-proof: zonable_for({Zone}, {level}, {w} x {d}) selects {id} at seed {k}`, `provenance: {path}`. Selftest path: load `tests/fixtures/factory/collide_directive.json` via `include_str!`, run `validate_semantics` against the REAL `AssetCatalog::load_project_assets()`, print the three lines (the K2 reason comes from the validator, not a hardcoded string — the line is real output of real validation), exit 1. Stubs = **task failure**.

- [x] **Step 1: Write the failing check** — the acceptance command itself (bins are checked by running them; the validator/loop logic is already lib-tested). Before implementing: `cargo run --bin asset_factory -- x` → error: no bin target.
- [x] **Step 2: Run to verify failure** — as above.
- [x] **Step 3: Implement** — arg parsing in the seed bin's style (brief = joined free args; `--help`); env knobs (`ASSET_FACTORY_ATTEMPTS`, `ASSET_FACTORY_SELFTEST`); both `main`s via `#[cfg]` like `forge_pathtrace.rs`, sharing a `selftest_collide() -> i32` fn compiled unconditionally.
- [x] **Step 4: Wire** — spectra `main` calls `run_factory(brief, &FactoryConfig::from_env())` and maps `FactoryOutcome` to the line set + exit code (Accepted → 0, Rejected → 1).
- [x] **Step 5: Run — verify non-trivial output** — the acceptance command: three exact lines, exit 1. The K2 reason names `forge.house.craftsman` because the catalog really contains it.
- [x] **Step 6: Leave uncommitted.**

---

## Task 11: Live end-to-end accept run (the Done When) — GATED

**GATE: requires Task 8 complete (and therefore the porch release) AND the SDF M1 Group B track finished in `vox_render` (this is a `--features spectra` build).** Budget: an E4B generation is minutes per attempt on the 780M; LLM and path tracer run strictly sequentially.

**Files:** none (verification run; fix-forward anything it exposes within the files already in the File Map).

**Acceptance:** the **Done When** accept-path command, verbatim, prints the full accept line set with real values (≤ 4 attempts; a multi-attempt run is a PASS if the final lines are the accept set and the rejected attempts' feedback lines appear verbatim in between), AND a human opens `assets/buildings/forge_starter/renders/factory/<id>/iso.png` and sees a brick rowhouse, AND `assets/source/buildings/generated/<id>.provenance.json` exists with one `AttemptRecord` per attempt, AND `python3 -c "import json; p=json.load(open('assets/buildings/forge_starter/pack.json')); print(len([a for a in p['assets'] if a.get('usage',{}).get('Zonable')=='ResidentialMed' and a['level']==3]))"` prints `4` (was 3).

**Wiring requirement:** no new wiring — this proves the wiring of Tasks 1–10. If a gate fails legitimately (e.g. the model writes a washed palette), that is the loop working: let it retry; only fix code if the loop itself misbehaves.

- [ ] **Step 1:** run the reject-path selftest under the spectra build first (cheap smoke of the full binary): same three lines, exit 1.
- [ ] **Step 2:** run the `#[ignore]`d live pieces once each: `cargo test --lib factory_llm_live_smoke -- --ignored --nocapture` (model emits a schema-valid directive) and `cargo test --lib factory_cook_determinism -- --ignored --nocapture` (two identical hashes printed).
- [ ] **Step 3:** run the Done-When accept command; capture the full log.
- [ ] **Step 4:** human checks: iso.png is a rowhouse; pool count line; provenance file; `pool-proof:` seed reproduces (`re-running zonable_for` with the printed seed via the printed line is the proof).
- [ ] **Step 5:** record the final printed line set at the bottom of this plan file (paste, dated).
- [ ] **Step 6: Leave uncommitted** (the user commits the new directive + provenance + plan updates together).

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — Task 1 rewires the cook in the lift step; Task 4 deletes the seed bin in the task that absorbs it; Task 9 wires the real stage fns into `run_factory`; no "wire later" tasks exist
- [x] Every `Acceptance` names a real non-trivial expected output (exact ids, exact K-code strings, exact hashes/counts — no "tests pass")
- [x] Every `Wiring requirement` names an exact function and exact file
- [x] `IMPORTANT NOTES` contains real, code-verified signatures (checked against `game_asset_cook.rs`, `forge_pathtrace.rs`, `asset/mod.rs`, the seed bin, and `llama-cli --help` on 2026-06-10; line numbers cited where the design doc was stale)
- [x] `File Map` lists every file that appears in any task (including the fixture files and the deletion)
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies
- [x] `Done When` names specific commands and specific human-observable results (line shapes + a PNG a human looks at + exit codes)
- [x] Types/method names consistent across tasks (`cook_candidate`, `gate_views`, `register_accepted`, `validate_semantics`, `run_loop`/`run_factory` used identically in Tasks 6–11)
- [x] Commit steps replaced by "leave uncommitted" throughout (forge has no git; user commits everything)
- [x] Coordination gates stated where tasks touch contested files (`forge_pathtrace.rs` porch release; `vox_render` SDF Group B for any spectra build)
