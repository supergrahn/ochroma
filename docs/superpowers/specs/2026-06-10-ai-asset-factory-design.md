# Design: AI Asset Factory — LLM Directives, Deterministic Cook, Spectra Gates (2026-06-10)

**Status:** Draft
**Scope:** A closed-loop asset factory for Civitas Care: a local LLM writes Forge *directives* (JSON only — it never touches geometry), `game_asset_cook` cooks them deterministically, the Spectra path tracer renders and pixel-gates the result, and accepted assets register into the live zonable pools. All new code lives in the GAME layer (`~/Ochroma/projects/civitas_care`); engine crates untouched.
**Related:** `[SOTA City Block Phase 1 Plan](../plans/2026-06-10-sota-city-block-phase1.md)` (SOTA item 7, "sequenced after instances"), `[Asset-Audit Remediation Plan](../plans/2026-06-10-asset-audit-remediation.md)` (this design targets the **post-remediation** directive schema: `variants` + `forge.condition`), `[Living Building Instances Design](./2026-06-10-living-building-instances-design.md)` (consumes the same catalog pools)

---

## 1. Problem Statement

Concrete symptoms in today's code (`~/Ochroma/projects/civitas_care`):

- **The LLM seed is open-loop.** `src/bin/asset_directive_from_llm.rs` shells `llama-cli` (temp 0.22, `-n 1800`), regex-extracts the first balanced JSON object from chatty stdout, checks only that 13 top-level keys *exist*, and writes the result straight into `assets/source/buildings/generated/` — the live directive dir that `load_directive_recipes` scans recursively. A directive with `forge.width: 200.0`, a colliding `asset_id`, or a `gameplay.class` that contradicts its zone lands in the next cook unguarded.
- **No constrained decoding.** The seed binary asks nicely for JSON in prose (`asset_directive_prompt.md`) and hopes; `llama-cli` on this box supports `--json-schema-file` GBNF-constrained generation (verified in `--help`, build b9372) and nothing uses it. Truncated or malformed output is a hard error with no retry.
- **No visual validation of generated assets.** `src/bin/forge_pathtrace.rs` has calibrated pixel gates (`gates: PASS` / `gates: FAIL <view>`, mean-luma band + per-channel std floor, `FORGE_GATE_SELFTEST=1` proves they fire) — but its `COOKED_SPECS` is a hardcoded `const` of the 5 starter payloads. A generated asset is never rendered, never gated, never seen by a human or a model before it can be zoned.
- **No acceptance state.** There is no record of what brief produced an asset, which model wrote the directive, how many attempts it took, or what the gate stats were. "Generated" and "accepted" are the same state.
- **One-shot, no feedback.** When the cook rejects a directive (`CookError::Directive`) or `validate_quality` fails, the error dies in stderr. The model never hears why, so retrying is a coin flip instead of a correction.

---

## 2. Done When

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
gate city.res_med.l3.3x4.<name>/iso:   mean luma <x> (band 60..120),  channel std <a>/<b>/<c> (floor 22) -> PASS
gate city.res_med.l3.3x4.<name>/front: mean luma <x> (band 110..150), channel std <a>/<b>/<c> (floor 35) -> PASS
gate city.res_med.l3.3x4.<name>/side:  mean luma <x> (band 60..120),  channel std <a>/<b>/<c> (floor 15) -> PASS
gates: PASS
judge: skipped (v1)
ACCEPTED city.res_med.l3.3x4.<name> after 1 attempt(s)
registered: pack.json ResidentialMed L3 pool 3 -> 4 assets
pool-proof: zonable_for(ResidentialMed, 3, 14.0 x 20.0) selects city.res_med.l3.3x4.<name> at seed <k>
provenance: assets/source/buildings/generated/city.res_med.l3.3x4.<name>.provenance.json
```

and a human opens `assets/buildings/forge_starter/renders/factory/city.res_med.l3.3x4.<name>/iso.png` and sees a brick rowhouse — not a black, washed, or flat frame. (`<M>`, `<V>`, `<W>` non-zero; attempt count ≤ 4; more attempts acceptable as long as the final line set is the accept set.)

**Reject path (selftest, no LLM run burned).** Running the same command with `ASSET_FACTORY_SELFTEST=collide` (a canned directive whose `asset_id` is `forge.house.craftsman`) prints

```
attempt 1/1: directive forge.house.craftsman (schema-valid, replay)
semantics: FAIL — K2 asset_id collides with existing catalog asset forge.house.craftsman
REJECTED: kill criterion K2, 0 retries permitted in selftest
```

and exits 1 — the factory's analog of `FORGE_GATE_SELFTEST=1`.

A human at the keyboard verifies both without reading code.

---

## 3. Capabilities

All factory unit tests are hermetic: the LLM runner has a `Replay` backend that reads canned `llama-cli` stdout fixtures from `tests/fixtures/factory/`, so no test shells a model. One `#[ignore]`d live smoke test exercises the real `llama-cli`.

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Directive extraction + schema validation | `cargo test --bin asset_factory directive_from_replay -- --nocapture`: feed a real captured llama-cli transcript (chatty preamble + JSON + trailer); assert the extracted value deserializes as `AssetDirective` (post-remediation shape incl. `variants`/`forge.condition`), `asset_id` matches `^city\.[a-z0-9_.]+$`, `footprint_m` both > 0; print the id | `assert!(extract_json_object(s).is_some())` — passes on `{}` |
| Semantic kill criteria | `cargo test --bin asset_factory kill_criteria -- --nocapture`: table of 6 poisoned directives (id collision, product-named id, forge dims > footprint, class/zone mismatch, floors outside range, bucket/metric mismatch); each rejected with its named `K*` code, all six codes printed | `assert!(validate(d).is_err())` — one generic error passes |
| Retry feedback loop | replay backend yields `[bad_directive, good_directive]`; assert the attempt-2 prompt contains the attempt-1 rejection text **verbatim** (printed), outcome is `Accepted` on attempt 2, and `AttemptRecord` history has exactly 2 entries with the right outcomes | `assert_eq!(outcome, Accepted)` — passes without any feedback plumbing |
| Cook determinism | cook the accepted fixture directive twice into two sandbox dirs (`--source <tmp> --output <sb{1,2}> --no-starters`); assert SHA-256 of the two `atoms/<id>.atoms.json` are byte-identical; print both hashes | comparing `atoms.len()` only |
| Gate reuse on real buffers | `cargo test --bin asset_factory gate_machinery -- --nocapture`: a synthetic black frame fails every view's mean floor and a flat grey frame fails every std floor (per-view stats printed); a captured real spp-32 render buffer fixture passes its band | `assert!(gate_result.is_ok())` |
| Catalog registration + pool growth | after `register_accepted`, load the pack via `AssetCatalog`; sweep `zonable_for(zone, level, w, d, seed)` for `seed in 0..pool_len`; assert the new id is selected exactly once per cycle; print pool size before/after | `assert!(pack.assets.iter().any(...))` without the pool-selection proof |
| Attempt budget + terminal kill | replay backend yields 4 bad directives; assert outcome is `Exhausted` (K8) after exactly 4 attempts and the provenance JSON on disk records 4 `AttemptRecord`s with per-attempt rejection reasons; print the summary line | `assert!(outcome.is_err())` |
| v2: judge verdict parsing | replay an `llama-mtmd-cli` verdict fixture; assert all 4 criterion scores parse as numbers, composite is their documented weighted mean (printed to 2 dp), and composite 4.9 < threshold 6.0 ⇒ `JudgeReject` | `assert!(verdict.is_some())` |

---

## 4. Architecture

The loop, end to end:

```
brief (one sentence, human or batch file)
  │
  ▼
[A] LLM runner ── llama-cli, --json-schema-file, --seed per attempt ──► directive JSON
  │                 (the directive IS the contract; the LLM never emits geometry,
  │                  meshes, atoms, or textures — only this one schema)
  ▼
[B] schema + semantic validator ──reject──► feedback appended to next prompt ─┐
  │ pass                                                                      │
  ▼                                                                           │
[C] sandbox cook ── game_asset_cook --source <tmp> --output <sandbox>         │
  │                 --no-starters (deterministic: same directive ⇒            │
  │ pass            byte-identical atoms.json) ──reject──────────────────────►│
  ▼                                                                           │
[D] render + pixel gates ── qa::pathtrace_gates (the forge_pathtrace          │
  │                 machinery, factored to a lib module): iso/front/side      │
  │ PASS            views, mean-luma band + std floor ──FAIL─────────────────►│
  ▼                                                                           │
[E] v2: vision judge ── llama-mtmd-cli + mmproj scores the views against     │
  │                 the brief, schema-constrained verdict ──reject───────────►│
  │ accept                                                                    │
  ▼                                                                    retry (≤ budget)
[F] register ── directive moves to assets/source/buildings/generated/,        │
                full recook registers it in pack.json, pool-proof printed,    ▼
                provenance written                                     attempts exhausted
                                                                       ⇒ REJECTED (K8)
```

### 4.1 Local model runtime — what is actually on this box (verified 2026-06-10)

- **No Ollama.** `which ollama` → not found; no `~/.ollama`. Do not design for it.
- **llama.cpp** at `/home/tom-espen/git/llama.cpp`, build **b9372** (commit 8ad8aef44, GCC 13.3.0), with CPU **and Vulkan** ggml backends (`libggml-vulkan.so` present) — the Vulkan backend runs on the same AMD Radeon 780M / RADV stack the Spectra path tracer is already proven on. Relevant binaries: `llama-cli` (supports `--json-schema-file` / `--grammar-file` constrained decoding and `--seed` — verified in `--help`), `llama-server` (OpenAI-compatible HTTP, not needed v1), `llama-mtmd-cli` (multimodal image+text).
- **Models** in `/home/tom-espen/models/`:
  - `gemma4-e4b.gguf` — 5.3 GB, GGUF arch `gemma4`, size label 7.5B (E4B effective), 131,072-token context, tagged `any-to-any`. This is the model the seed binary already drives at temp 0.22.
  - `gemma4-e4b-mmproj.gguf` — 560 MB vision projector for the same model ⇒ **local vision judging is feasible** via `llama-mtmd-cli` with zero new downloads.
  - `bge-m3-Q8_0.gguf` — embedding model; not used in v1 (candidate for v2+ near-duplicate detection of briefs/assets).

**Decision:** v1 backend is the `llama-cli` subprocess path the seed binary proved, upgraded with `--json-schema-file <schema>` and `--seed <per-attempt>`, `-n 2600` (room for `variants`), temp 0.22. Defaults and env overrides (`GAME_ASSET_LLAMA_CLI`, `GAME_ASSET_LLM_MODEL`) carry over unchanged. The factory is an **offline batch tool**: on a 780M iGPU an E4B-class generation of a ~2-kilotoken directive takes minutes per attempt, and the judge adds image-encode time per view — acceptable; the Done When sets no latency target. LLM and path tracer both want the iGPU, so stages run strictly sequentially (they do anyway).

**Optional remote fallback (clearly optional, OFF by default, never required by Done When):** `ASSET_FACTORY_LLM=remote` plus `ASSET_FACTORY_LLM_ENDPOINT` / `ASSET_FACTORY_LLM_KEY_ENV` selects an OpenAI-compatible `/v1/chat/completions` client (the same wire shape `llama-server` speaks, so local-server and remote share one code path). Remote responses cannot be grammar-constrained, so the schema validator is the only contract enforcement on that path — which is fine, because it already is the contract enforcement on every path.

### 4.2 LLM runner and prompt

`factory::llm::LlmRunner` is an enum: `LlamaCli { cli, model, n_predict, temp }` (v1 default), `Replay { transcripts }` (tests/selftest), `Remote { … }` (optional). `generate_directive(prompt, sampling_seed)` shells `llama-cli -m <model> -p <prompt> --json-schema-file <schema> --seed <s> -n 2600 --temp 0.22 --no-display-prompt`, then runs the seed binary's balanced-brace `extract_json_object` as defense-in-depth (constrained decode can still truncate at the `-n` cap ⇒ invalid JSON ⇒ K1 retry). The prompt is `asset_directive_prompt.md` extended with: the post-remediation schema (`variants` merge-patch array, `forge.condition` ∈ New/Aged/Weathered/Derelict), the brief, and — on retries — a `Previous attempt failed:` block quoting every rejection verbatim. `asset_directive_from_llm.rs` is retired; its extraction/validation helpers move into `factory::llm` (the bin's CLI is subsumed by `asset_factory`).

### 4.3 The directive schema — single source of truth

One JSON Schema file, `assets/source/prompts/asset_directive.schema.json`, describes the **post-remediation** `AssetDirective`: everything `recipe_from_directive` consumes today, plus `variants: [merge-patch]` (remediation Task 4: RFC-7396 patches applied to the raw directive `Value` before deserialization, ids `{asset_id}.v{n}`) and `forge.condition` (remediation Task 3: serde-default `"New"`, enum-validated). It constrains enums (`arch_style`, zones, ploppable categories, `kind`, `footprint_shape`, `roof_style`, `hero_level`, `condition`), numeric ranges (floors 1–8, window_density 0–1, positive dims), and the `asset_id` pattern. The same file feeds `--json-schema-file` and the validator — the LLM physically cannot emit a key the cook doesn't understand. A cook unit test (`directive_schema_roundtrip`) deserializes the schema's embedded example through the real `AssetDirective` serde shape so schema drift fails CI, including the remediation plan's Task-3-vs-Task-4 ambiguity about where `condition` lives (this design follows Task 3's wiring: `forge.condition`; the roundtrip test pins whatever the remediation track actually lands).

### 4.4 Semantic validator (kill criteria K1–K4, §"Kill criteria" below)

Pure function over the parsed `Value` + the loaded `AssetCatalog`: id uniqueness and product-neutrality, forge-dims-fit-footprint, bucket/metric consistency (CS-style bucket cell = 8 m: each `footprint_m` axis ≤ cells × 8 and > (cells − 1) × 8), gameplay class ↔ zone ↔ kind coherence, `forge.floors` within `floors` range, variant patches must parse as objects and may not patch `asset_id`. Every failure carries a `K*` code and a one-line human/LLM-readable reason — the same string goes to stdout, the provenance record, and the retry prompt.

### 4.5 Sandbox cook (determinism boundary)

The factory writes the candidate directive to a temp source dir and shells the **existing** cook: `cargo run --bin game_asset_cook -- --source <tmp> --output <sandbox> --no-starters`. `--source`/`--output` already exist (`config_from_args`, game_asset_cook.rs:135); `--no-starters` is one additive flag that skips `starter_recipes()` so a candidate cook doesn't re-run Forge + AssemblyPrime for the 5 starters. Determinism is the cook's existing guarantee, hardened by remediation Task 2 (zone-derived materials, `HashMap::values().next()` fallback killed): Forge geometry is `seed`-driven (Pcg64), merge-patch expansion is pure, AssemblyPrime exports are reused per directive — **same directive file ⇒ byte-identical `atoms.json`**, asserted by test. `validate_quality` runs inside the cook as today (atom minimums, floating-atom gate, ground coverage post-Task-5).

### 4.6 Render + pixel gates (reuse, not reinvention)

The gate machinery in `forge_pathtrace.rs` — `review_rig()`, `pixel_stats`, `GateBand`/`gate_band`, the PASS/FAIL print contract — is factored into `civitas_care::qa::pathtrace_gates` (lib module, `#[cfg(feature = "spectra")]`); `forge_pathtrace.rs` becomes a thin caller with its `COOKED_SPECS` const and **unchanged output contract** (its calibrated bands, selftest mode, and `gates:` line survive verbatim — this respects the remediation track's read-only contract and happens only after that track releases the file). The factory renders the candidate payload from the sandbox through the same `load_ready_asset_mesh` + `ready_textured_materials` path at spp 32, but only the **style-agnostic views: iso, front, side** (+ `debug_mat` as a non-gated artifact). The porch view is starter-specific and is not rendered for candidates. Bands are reused as-is: they are sanity gates (black / washed / flat frame detectors), not aesthetics — calibrated with margin against converged spp-32 renders, and path-trace noise sits well inside them. Outputs land in `renders/factory/<asset_id>/`.

### 4.7 Vision judge (v2 — explicitly second iteration)

`llama-mtmd-cli -m gemma4-e4b.gguf --mmproj gemma4-e4b-mmproj.gguf` scores the three gated views against the brief with a schema-constrained verdict: `{ "scores": { "reads_as_brief": 0-10, "facade_coherence": 0-10, "roof_and_ground": 0-10, "no_artifacts": 0-10 }, "verdict": "accept"|"reject", "defects": ["..."] }`. Composite = mean of the four scores; accept requires composite ≥ threshold **and** every score ≥ floor (threshold/floor are constants calibrated in the v2 plan by running the judge over the 5 known-good starter payloads and one known-bad corrupted render — the same calibration discipline the pixel bands used). `defects` strings feed the retry prompt exactly like cook errors. The judge is a **gate in v2 and absent in v1** — v1 prints `judge: skipped (v1)` so the output contract is stable across versions. The judge's nondeterminism is acceptable because it gates *acceptance of a new asset*, never *reproduction of an accepted one* (§Determinism).

### 4.8 Acceptance, registration, provenance

On accept: (1) the directive file moves from the factory's quarantine (`assets/factory/candidates/` — deliberately **outside** `assets/source/buildings`, which `collect_directive_files` scans recursively) to `assets/source/buildings/generated/<asset_id>.asset.json`; (2) the factory runs the standard full recook (`game_asset_cook` with defaults), which registers the `BuildingAsset` in `pack.json` — from that moment `zonable_for` / `zonable_for_theme` pick it up automatically: pools are just `assets() filtered by is_zonable(zone) && level && fits(w,d)`, indexed `seed % pool.len()`, so registration **is** pool membership; (3) the factory proves it, sweeping seeds until the new id is selected and printing the `pool-proof:` line; (4) a provenance record (`<asset_id>.provenance.json`, sibling of the directive) captures brief, model path + GGUF arch/size, llama.cpp build, every `AttemptRecord` (sampling seed, outcome, rejection reasons), final gate stats, and (v2) the judge verdict. Directives may carry `variants` — the base is what gets gated; variants are merge-patches of an already-accepted directive and are re-validated by the same in-cook `validate_quality` during the recook, growing the pool by `1 + variants.len()` (the remediation plan's variant-pool mechanism, unchanged).

### 4.9 Orchestrator

`src/bin/asset_factory.rs` owns the attempt loop: for `attempt in 1..=budget` (default 4, `ASSET_FACTORY_ATTEMPTS` override): generate → validate → cook → gate → (judge) → accept/collect-feedback. Single-threaded; every external stage is a sequential subprocess (iGPU contention makes parallelism pointless). Per-attempt sampling seed = `splitmix64(fnv1a(brief) ^ attempt)` so a factory run is itself replayable. Terminal kills (K8 exhausted, K9 model collapse) stop the loop early.

---

## 5. Data Models

All in `civitas_care::factory` (GAME layer). Private fields, accessor methods, per house rules.

```rust
/// Which model backend the factory drives. v1 default: LlamaCli.
pub enum LlmBackend {
    LlamaCli { cli: PathBuf, model: PathBuf, n_predict: u32, temp: f32 },
    Replay { transcripts: Vec<PathBuf> },          // tests + ASSET_FACTORY_SELFTEST
    Remote { endpoint: String, key_env: String },  // OPTIONAL, env-gated, off by default
}

/// One attempt of the loop, recorded into provenance.
pub struct AttemptRecord {
    index: u32,            // 1-based
    sampling_seed: u64,    // splitmix64(fnv1a(brief) ^ index)
    directive: Option<serde_json::Value>, // None if generation itself failed
    outcome: AttemptOutcome,
}
impl AttemptRecord { pub fn index(&self) -> u32 { self.index } /* … accessors … */ }

pub enum AttemptOutcome {
    SchemaReject(Vec<String>),       // K1
    SemanticReject(Vec<Rejection>),  // K2..K4
    CookReject(String),              // K5 — CookError text verbatim
    GateFail(Vec<GateStat>),         // K6 — per-view mean/std + band
    JudgeReject(JudgeVerdict),       // K7 (v2)
    Accepted { asset_id: String },
}

/// A named kill with the human/LLM-readable reason (one line).
pub struct Rejection { criterion: KillCriterion, reason: String }

pub enum KillCriterion { K1, K2, K3, K4, K5, K6, K7, K8, K9, K10 }

/// Per-view gate stats, mirroring qa::pathtrace_gates output.
pub struct GateStat { view: String, mean: f32, stds: [f32; 3], pass: bool }

/// v2 judge verdict (schema-constrained model output, parsed).
pub struct JudgeVerdict {
    scores: [(String, f32); 4],
    composite: f32,
    verdict_accept: bool,
    defects: Vec<String>,
}

/// Written next to the accepted directive as <asset_id>.provenance.json.
pub struct ProvenanceRecord {
    brief: String,
    model: String,            // path + GGUF arch/size label
    llama_build: String,      // e.g. "b9372 (8ad8aef44)"
    attempts: Vec<AttemptRecord>,
    accepted_id: Option<String>,
    gate_stats: Vec<GateStat>,
    judge: Option<JudgeVerdict>,
}

pub enum FactoryOutcome {
    Accepted { asset_id: String, attempts: u32, pool_before: usize, pool_after: usize },
    Rejected { terminal: KillCriterion, attempts: u32 },
}
```

---

## 6. API

```rust
// civitas_care::factory — orchestrator entry. Blocking; runs every stage as a
// sequential subprocess. Threading: main thread only (iGPU is shared by the
// LLM and the path tracer; never run them concurrently).
pub fn run_factory(brief: &str, cfg: &FactoryConfig) -> Result<FactoryOutcome, FactoryError>;

// civitas_care::factory::llm — one attempt's generation.
// Errors: spawn/exit failures, no-JSON-in-output, parse failure (all -> K1).
impl LlmRunner {
    pub fn generate_directive(&self, prompt: &str, sampling_seed: u64)
        -> Result<serde_json::Value, LlmError>;
}

// civitas_care::factory::validate — schema is enforced by serde deserialization
// of the post-remediation AssetDirective; semantics by the catalog-aware checks.
// Ok(()) means "safe to cook". Never panics on hostile input.
pub fn validate_semantics(directive: &serde_json::Value, catalog: &AssetCatalog)
    -> Result<(), Vec<Rejection>>;

// civitas_care::factory::cook — shells `game_asset_cook --source <tmp>
// --output <sandbox> --no-starters`; parses the printed `cooked …` line.
pub fn cook_candidate(directive_path: &Path, sandbox: &Path)
    -> Result<CookedCandidate, String>;   // Err -> K5, text verbatim

// civitas_care::qa::pathtrace_gates — factored from forge_pathtrace.rs,
// #[cfg(feature = "spectra")], output contract identical to today's binary.
// Renders iso/front/side at `spp`, writes PNGs under out_dir, gates each view.
pub fn gate_views(payload: &Path, out_dir: &Path, spp: u32)
    -> Result<Vec<GateStat>, String>;     // any !pass -> K6

// civitas_care::factory::judge — v2 only. Shells llama-mtmd-cli with the
// mmproj; verdict is schema-constrained JSON parsed into JudgeVerdict.
pub fn judge_views(renders: &[PathBuf], brief: &str, runner: &LlmRunner)
    -> Result<JudgeVerdict, LlmError>;

// civitas_care::factory::register — move directive out of quarantine, full
// recook, pool-proof sweep, provenance write. Returns before/after pool sizes
// and the seed at which zonable_for selected the new asset.
pub fn register_accepted(candidate: &CookedCandidate, brief: &str,
    attempts: &[AttemptRecord]) -> Result<RegistrationProof, String>;
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `asset_factory` main → `run_factory` | new binary | `~/Ochroma/projects/civitas_care/src/bin/asset_factory.rs` | parses brief + env knobs; prints the Done-When line set |
| `factory::*` modules | `run_factory` | `~/Ochroma/projects/civitas_care/src/factory/{mod,llm,validate,cook,judge,register}.rs` | new lib modules (GAME layer) |
| `qa::pathtrace_gates::gate_views` | `run_factory` **and** `forge_pathtrace::run` | `~/Ochroma/projects/civitas_care/src/qa/pathtrace_gates.rs` | factored from forge_pathtrace.rs after the porch-closure track releases it; the binary's printed contract is unchanged |
| `--no-starters` flag | `config_from_args` + `run` recipe assembly | `~/Ochroma/projects/civitas_care/src/bin/game_asset_cook.rs` (lines 135, 49) | additive; default behavior identical |
| `asset_directive.schema.json` | `LlmRunner::generate_directive` (`--json-schema-file`) + cook test `directive_schema_roundtrip` | `~/Ochroma/projects/civitas_care/assets/source/prompts/asset_directive.schema.json` | single source of truth; pins post-remediation shape |
| prompt update (`variants`, `forge.condition`, feedback block) | prompt assembly in `factory::llm` | `~/Ochroma/projects/civitas_care/assets/source/prompts/asset_directive_prompt.md` | extends the existing file |
| quarantine dir | `run_factory` writes candidates here | `~/Ochroma/projects/civitas_care/assets/factory/candidates/` | outside the cook's scanned `assets/source/buildings` tree |
| accepted directives + provenance | `register_accepted` | `~/Ochroma/projects/civitas_care/assets/source/buildings/generated/` | the dir the seed binary already targeted; now reachable only via acceptance |
| `asset_directive_from_llm.rs` retirement | helpers absorbed into `factory::llm`; bin deleted | `~/Ochroma/projects/civitas_care/src/bin/asset_directive_from_llm.rs` | same task that lands `factory::llm` |

---

## 8. Kill criteria — when a generated asset is auto-rejected

Retryable kills feed their reason verbatim into the next attempt's prompt; terminal kills end the run.

| Code | Criterion | Stage | Retry? |
|---|---|---|---|
| K1 | Output is not valid JSON / fails schema deserialization (constrained decode truncated at `-n`, or remote path emitted junk) | LLM | yes |
| K2 | `asset_id` collides with any catalog/pack asset, or is product-named (`civitas`, `civ_`, final title) | validate | yes |
| K3 | Geometry incoherence: `forge.width/depth` exceed `footprint_m`; `forge.floors` outside `floors` range; `footprint_bucket` inconsistent with `footprint_m` (8 m/cell rule, §4.4); any variant patch touches `asset_id` | validate | yes |
| K4 | Gameplay incoherence: `gameplay.class` ↔ `usage` zone ↔ `kind` mismatch (e.g. `MediumResidential` directive zoned `IndustrialHeavy`; `kind: school` as a zonable) | validate | yes |
| K5 | Cook failure: `CookError` of any sort, including the cook's own `validate_quality` gates (atom minimums, floating rooftop atoms, missing plot ground coverage) | cook | yes |
| K6 | Any pixel gate FAIL on iso/front/side (black, washed, or flat frame) | gates | yes |
| K7 | v2: judge composite < threshold, any criterion score < floor, or `verdict: reject` | judge | yes |
| K8 | Attempt budget exhausted (default 4, `ASSET_FACTORY_ATTEMPTS`) | loop | **terminal** |
| K9 | Model collapse: attempt *k*'s directive is byte-identical to attempt *k−1*'s despite a different sampling seed | loop | **terminal** |
| K10 | Runtime budget: cooked atom count > 4× the largest starter payload's count (protects frame budget; pinned to a number at implementation from today's recook output) | cook | yes ("reduce detail" feedback) |

---

## 9. Determinism guarantees (honest boundaries)

- **Stage A (LLM) — best-effort reproducible, not guaranteed.** Fixed temp + per-attempt `--seed` makes `llama-cli` reproducible for the same GGUF, build, backend, and thread count on this box — but that is an implementation detail, not a contract. **The emitted directive JSON is the determinism boundary**: it is stored, provenance-stamped, and everything downstream depends only on it.
- **Stage B (cook) — bit-deterministic, the core guarantee.** Same directive file ⇒ byte-identical `atoms.json` and pack entry. Rests on: seeded Forge generation (Pcg64 from `params.seed`), pure RFC-7396 expansion, remediation Task 2's removal of HashMap-order texture fallback, splitmix64-style in-cook jitter, AssemblyPrime export reuse. Asserted by the SHA-256 double-cook test (§3). A regenerated asset on any machine with the same forge/AssemblyPrime checkouts is the same asset.
- **Stage C (gates) — deterministic decision, not deterministic pixels.** GPU path tracing varies per run/driver; the bands were calibrated with ~20 % margin against converged spp-32 renders precisely so noise never flips a verdict. The gate *decision* is treated as stable; raw pixel buffers are never compared.
- **Stage D (judge, v2) — nondeterministic, quarantined by design.** The judge gates only *first acceptance*. Once a directive is accepted, reproduction of the asset never re-consults the LLM or the judge — re-cooking the committed directive is Stage B.

---

## 10. Open Questions

- [ ] Judge threshold/floor values (v2): to be calibrated by scoring the 5 starter payloads (must accept) and one `FORGE_GATE_SELFTEST`-corrupted render (must reject) — numbers go in the v2 plan, not guessed here.
- [ ] Where `condition` lands in the directive after remediation: Task 3's wiring says `forge.condition`, Task 4's example patch shows top-level `condition`. The schema-roundtrip test pins whichever ships; this design assumes `forge.condition`.
- [ ] K10's concrete atom cap: read the largest starter count from the post-remediation recook output and pin 4× that in the plan.

---

## 11. Out of Scope

- **Geometry, mesh, texture, or atom generation by the LLM** — the directive is the entire interface, permanently. No "LLM fixes the mesh" path will ever be added to this loop.
- Model fine-tuning, LoRA training, or prompt-optimization loops over the local model.
- `HeroLevel::Hero`'s 3–5 Gemini iterations inside Forge — that is Forge's own (remote, optional) path; the factory drives `background`/`supporting` hero levels only.
- Interactive in-editor generation UI; the factory is a CLI batch tool. (A UI can shell the same binary later.)
- Asset kinds beyond the cook's four (`house`, `rowhouse`, `school`, `police`); new kinds require cook work first and arrive with their own plans.
- Near-duplicate detection via `bge-m3` embeddings (noted as available on-box; v2+ candidate, not designed here).
- Making the remote-API fallback a requirement of anything — it exists as an env-gated option only.
- Cross-machine bit-determinism of the **LLM stage** (Stage A); only the directive artifact and everything below it carry guarantees.

---

## 12. Related Plans / Designs

- Depends on: `[Asset-Audit Remediation Plan](../plans/2026-06-10-asset-audit-remediation.md)` (Tasks 2–5 — directive `variants`, `forge.condition`, deterministic materials, plot bake) landing first; the porch-closure track releasing `forge_pathtrace.rs` (for the §4.6 refactor).
- Required before: the AI-asset-factory implementation plan (to be written from this design using `docs/templates/plan.md`).
- Related: `[SOTA City Block Phase 1 Plan](../plans/2026-06-10-sota-city-block-phase1.md)`, `[Living Building Instances Design](./2026-06-10-living-building-instances-design.md)`, `[Virtualized Splat Rendering Design](./2026-06-10-virtualized-splat-rendering-design.md)`.
