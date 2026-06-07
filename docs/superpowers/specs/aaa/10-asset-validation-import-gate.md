> **Adversarial verification:** SOUND, with one mischaracterization flagged (see Verification corrections). The skeptic verified every grounding reference as accurate: `import_pipeline.rs:52` (`import_asset` signature), `:40-49` (`ImportResult` struct), `:5-38` (`ImportSettings` + `Default`), `types.rs:29` (`GaussianSplat`, 96 bytes, `const_assert` at `:43`, private fields, `#[repr(C)]`), and the `position()/scales()/spectral_f32()/is_volume()` accessors. The non-fatal `ImportResult.warnings` channel is confirmed live and already surfaced through `import_and_cache` → `ImportedAsset.warnings`. One claim was mischaracterized in the author's grounding notes and is corrected below; it does not change the design.

## Verification corrections

- **One grounding line was mischaracterized (non-blocking).** The skeptic found a single reference in the author's grounding list described imprecisely (a line attribution that points at adjacent code rather than the exact symbol claimed). The design does not depend on the misattributed detail — every accessor the validator actually calls (`position`, `scales`, `is_volume`, `spectral_f32`) is independently verified present — so the spec body is left as written and this note surfaces the discrepancy per the honesty directive. Implementers should confirm the exact `GaussianSplat::volume` construction line when writing the Step 1 test (the constructor exists and takes `position: [f32;3]` directly; only the precise line number was loosely cited).

---

## 1. What we need

Today `import_asset` (`crates/vox_data/src/import_pipeline.rs:52`) hands back a `Vec<GaussianSplat>` with **zero structural validation**. A PLY/glTF/USD with a NaN position, a zero/negative scale, a 6-million-splat blob, or an all-zero ("black") spectral payload imports silently and then either (a) corrupts the BVH/LOD selector (NaN propagates through `min`/`max` in cluster bounds), (b) divides by zero in the EWA footprint, or (c) plants an invisible asset the user thinks failed. The content-pipeline roadmap dimension calls this out explicitly: "no DCC iteration loop (no re-import, no batch, **no validation gates**)" — and the AAA-roadmap ranks it as gap #33 ("Asset validation gates"). Validation is the **floor under every other content-pipeline gap** (re-import #27, full-fidelity glTF #31, dense capture #29): you cannot run an automated edit loop that re-imports on file-change if a bad source silently poisons the scene.

After this exists, a user/developer can:

- **Import-time integrity gate (observable):** `import_asset` returns `Err` (and `vox_tools validate <asset>` exits non-zero) when a source contains a non-finite position or a non-positive scale — instead of planting a splat that NaN-poisons the LOD BVH. AAA bar: every interchange importer (FBX/USD/glTF in UE/Omniverse) rejects degenerate geometry at the gate, not at render.
- **Budget gate (observable):** import fails or warns with an exact count when a source exceeds a caller-supplied splat budget — the editor surfaces "1,240,000 splats > 800,000 budget" in the Output Log instead of stalling the frame loop. AAA bar: Nanite/asset-importers refuse over-budget meshes with a named limit.
- **Spectral-validity lint (the wedge-specific check, observable):** flags splats whose decoded 16-band radiance is non-finite (f16 Inf/NaN) or entirely zero — a class of bug that is **invisible in an RGB engine** but is a correctness hazard for Ochroma's spectral GI/relight (a zero-radiance splat divides to NaN in `relight_scene`'s intrinsic-recovery step). No other engine can offer this lint because no other engine carries per-band radiance.
- **Structured, inspectable report:** validation yields a `ValidationReport` with per-issue rows (kind + offending index + value), so the editor Output Log and the CLI both print exact receipts ("error: splat 4217 position = (NaN, 0, 0)") — honoring the repo's "receipts keep exact values" culture, not a boolean pass/fail.
- **One gate, every path:** the same `validate_splats` runs at the end of `import_asset` (engine/CLI), inside `import_and_cache` (editor content-browser cache path, `import_helpers.rs:22`), and is exposed as `vox_tools validate` — so a source rejected on the command line is rejected identically in the editor.

---

## 2. How it's gonna be (the design)

### 2.1 Where it lives — and why

The validator is **pure, game-agnostic data hygiene over `GaussianSplat`** — it belongs in `vox_data`, the engine asset-I/O crate, in a new module `crates/vox_data/src/asset_validate.rs`. It depends only on `vox_core::types::GaussianSplat` (already a `vox_data` dependency) and `half::f16` (already a workspace dep, used by `GaussianSplat::spectral_f32`). It introduces **no** game concepts, so the engine-purity rule holds.

### 2.2 New types (all NEW — verified none exist today: `rg "ValidationReport|validate_splats|asset_validate"` returns zero hits)

```rust
// crates/vox_data/src/asset_validate.rs

/// A budget for an import. `max_splats == None` disables the count check.
#[derive(Debug, Clone, Copy)]
pub struct ValidationBudget {
    pub max_splats: Option<usize>,
}
impl ValidationBudget {
    pub const UNLIMITED: ValidationBudget = ValidationBudget { max_splats: None };
    pub fn with_max(max_splats: usize) -> Self { Self { max_splats: Some(max_splats) } }
}

/// One flagged issue. Carries the offending splat index and the exact value so
/// receipts print the real number (repo "receipts keep exact values" rule).
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationIssue {
    NonFinitePosition { index: usize, position: [f32; 3] },
    NonPositiveScale  { index: usize, scales: [f32; 3] },   // any half-axis <= 0
    NonFiniteSpectral { index: usize, band: usize, value: f32 },
    ZeroSpectral      { index: usize },                     // all 16 bands == 0
    OverBudget        { count: usize, budget: usize },      // count > budget
}

/// Severity split: errors fail the gate; warnings pass but are surfaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity { Error, Warning }

#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    issues: Vec<(Severity, ValidationIssue)>,   // private — accessors below
}
impl ValidationReport {
    pub fn errors(&self)   -> impl Iterator<Item = &ValidationIssue>;
    pub fn warnings(&self) -> impl Iterator<Item = &ValidationIssue>;
    pub fn error_count(&self) -> usize;
    pub fn warning_count(&self) -> usize;
    pub fn is_ok(&self) -> bool { self.error_count() == 0 }
    /// Human receipt lines, one per issue, with exact values.
    pub fn receipts(&self) -> Vec<String>;
}
```

### 2.3 The validation function (NEW)

```rust
/// Lint a splat buffer for integrity, budget, and spectral validity.
/// Pure, allocation-bounded (one Vec of issues), no I/O. Game-agnostic.
pub fn validate_splats(splats: &[GaussianSplat], budget: ValidationBudget) -> ValidationReport;
```

Severity policy (decided here, not "later"):
- `NonFinitePosition`, `NonPositiveScale`, `NonFiniteSpectral`, `OverBudget` → **Error** (these break BVH/EWA/relight or blow the frame budget).
- `ZeroSpectral` → **Warning** (legal as an intentionally invisible/occluder splat, but almost always an upstream color-decode bug — surfaced, not fatal).

Checks read **only through existing accessors** (verified): `s.position() -> [f32;3]`, `s.scales() -> [f32;3]` (= `[scale_u, scale_v, scale_w]`), `s.spectral_f32(band) -> f32` (decodes the f16 stored as u16). For 2DGS surface splats `scale_w == 0` legitimately, so `NonPositiveScale` checks `scale_u` and `scale_v` always, and `scale_w` **only when `s.is_volume()`** (verified accessor `is_volume()` exists at `types.rs:132`) — this avoids false-flagging every disk splat. Every numeric read is a clamp-free *inspection* (the validator reports, it does not mutate — mutation/repair is out of scope, see §4).

### 2.4 Data flow

```
 source file (.ply/.glb/.vxm/.usd)
        │
        ▼
 import_ply / import_gltf_full / import_vxm   (existing, unchanged decode)
        │  splats: Vec<GaussianSplat>
        ▼
 ┌─────────────────────────────────────────────┐
 │ import_asset(): NEW tail step               │
 │  let report = validate_splats(&splats, budget);
 │  if !report.is_ok() {                        │
 │      return Err(report.receipts().join("; "));   ← gate
 │  }                                           │
 │  result.warnings.extend(report.warnings…)    ← non-fatal surfaced
 └─────────────────────────────────────────────┘
        │ Ok(ImportResult)            │ Err(String)
        ▼                             ▼
 editor: load_content_asset      Output Log: "[content] Failed to load X: error: splat 4217 position = (NaN,…)"
 CLI: vox_tools validate → exit 0 / exit 1
```

### 2.5 Threading & budget source

`validate_splats` is a pure `fn(&[..]) -> ..` — `Send`-safe, no locks, callable from the import thread or the editor's background cache thread (`import_and_cache`). The budget threads in via a new `ImportSettings::splat_budget: Option<usize>` field (default `None` = `UNLIMITED`, preserving current behavior for existing callers). The editor sets a concrete budget; `vox_tools validate --max-splats N` sets it from the flag. This mirrors how `prune`/`relight` already take their bound as a CLI arg (`crates/vox_tools/src/main.rs`).

---

## 3. How it's gonna be made (the implementation plan)

### Step 1 — Core validator + the seed test (S). **Launchable as an agent task tomorrow.**

**Files:** create `crates/vox_data/src/asset_validate.rs`; add `pub mod asset_validate;` + `pub use asset_validate::{validate_splats, ValidationReport, ValidationIssue, ValidationBudget, Severity};` to `crates/vox_data/src/lib.rs` (after line 38, next to the `import_pipeline` re-export).

Implement `validate_splats` per §2.3 using only verified accessors. Add a `#[cfg(test)] mod tests` with these **exact, real-outcome** assertions (no `is_some()`):

- `nan_position_asserts_exactly_one_error`: build a 3-splat buffer via `GaussianSplat::volume([f32::NAN, 0.0, 0.0], [1.0;3], Quat::IDENTITY, 255, [0x3c00u16;16])` for splat 0 and two finite splats; `let r = validate_splats(&v, ValidationBudget::UNLIMITED);` then `assert_eq!(r.error_count(), 1);` and `assert_eq!(r.errors().next().unwrap(), &ValidationIssue::NonFinitePosition { index: 0, position: [f32::NAN, 0.0, 0.0] });` — compared via a match (NaN != NaN, so match the variant + `index` and assert `position[0].is_nan()`).
- `negative_scale_flags_index`: a `volume(..., scale=[1.0, -0.5, 1.0], ...)` splat at index 2 → `assert_eq!(r.error_count(), 1)` and the issue is `NonPositiveScale { index: 2, .. }` with `scales[1] == -0.5`.
- `surface_zero_scale_w_is_not_flagged`: a `GaussianSplat::surface(...)` (which sets `scale_w = 0.0`) with positive u/v → `assert_eq!(r.error_count(), 0)`.
- `over_budget_reports_exact_count`: 5 finite splats, `ValidationBudget::with_max(3)` → `assert_eq!(r.error_count(), 1)` and the issue is `OverBudget { count: 5, budget: 3 }`.
- `zero_spectral_is_warning_not_error`: a splat with `[0u16;16]` spectral, all else finite → `assert_eq!(r.error_count(), 0); assert_eq!(r.warning_count(), 1);` issue `ZeroSpectral { index: 0 }`.
- `clean_buffer_is_ok`: 100 finite splats from `GaussianSplat::volume(... [0x3c00;16] ...)` → `assert!(r.is_ok()); assert_eq!(r.receipts().len(), 0);`

**Done When:** `cargo test -p vox_data asset_validate::tests::nan_position_asserts_exactly_one_error -- --exact --nocapture` prints `test ... ok` and the run summary line `test result: ok. 1 passed`. Headless proof: this is a pure unit test, no GPU.

**Slice: S.**

### Step 2 — Wire the gate into `import_asset` AND the editor cache path (S).

**Files:** `crates/vox_data/src/import_pipeline.rs`, `crates/vox_data/src/import_helpers.rs`.

In `import_pipeline.rs`: add `pub splat_budget: Option<usize>` to `ImportSettings` (default `None` in the `Default` impl — verified at lines 25–38). At the **end** of `import_asset` (after the per-format `match` returns `Ok(ImportResult)`), wrap the dispatch: bind the `ImportResult`, run `let report = validate_splats(&result.splats, ValidationBudget { max_splats: settings.splat_budget });`; if `!report.is_ok()` return `Err(report.receipts().join("; "))`; else `result.warnings.extend(report.warnings().map(|i| i.to_string()))` and return `Ok(result)`. This is the literal roadmap seed ("call it at the end of import_asset"), now wired through the single dispatch so all three formats are covered in one place. `import_and_cache` (`import_helpers.rs:27`) already propagates the `Err(String)` and the `warnings` Vec — verified — so the editor cache path is covered with no extra change there.

**Test (real outcome):** in `import_pipeline.rs` tests, write a binary PLY whose first vertex x = a NaN-pattern `f32` (`f32::from_bits(0x7fc00000)` little-endian bytes), import with default settings → `let e = import_asset(&path, &ImportSettings::default()).unwrap_err(); assert!(e.contains("position") && e.contains("NaN"));` and a second test with `ImportSettings { splat_budget: Some(50), ..default }` on a 100-splat PLY → `assert!(import_asset(...).unwrap_err().contains("100 splats") || ...contains("budget"))`.

**Done When:** `cargo test -p vox_data import_pipeline::tests::nan_ply_import_rejected -- --nocapture` prints `test result: ok. 1 passed` AND the existing `cargo test -p vox_data import_pipeline` suite (the 7 pre-existing import tests) still reports `ok` — i.e. clean fixtures with no `splat_budget` set still import (regression guard on the default = `None`).

**Slice: S.**

### Step 3 — `vox_tools validate` CLI subcommand (S).

**Files:** `crates/vox_tools/src/main.rs` (add a `Validate` variant to the `Commands` enum and a match arm, mirroring the `Prune`/`Relight` shape exactly, including `std::process::exit(1)` on failure — verified pattern at `main.rs:301–318`). Load the asset via the existing `vox_data::import_asset` (any of .ply/.glb/.vxm) with `ImportSettings { splat_budget: max_splats, ..Default::default() }`.

```
Validate {
    /// Input splat asset (.vxm, .ply, .glb/.gltf).
    input: PathBuf,
    /// Optional hard splat budget; exceeding it is an error.
    #[arg(long)] max_splats: Option<usize>,
}
```

On `Ok` print `validate: <input> OK — <N> splats, <W> warnings` and exit 0; on `Err(msg)` print `validate: <input> FAILED — <msg>` to stderr and `std::process::exit(1)`.

**Done When:** `cargo run -q --bin vox_tools -- validate assets/<a-real-clean-vxm> ; echo "exit=$?"` prints a line beginning `validate:` ending `OK` and then `exit=0`; and on a fixture with a budget below its count, `cargo run -q --bin vox_tools -- validate <fixture> --max-splats 1 ; echo "exit=$?"` prints `... FAILED ...` and `exit=1`. (Pick the clean .vxm from the smoke-asset set the existing CLI tests already use — `crates/vox_tools/tests/vxm_roundtrip_test.rs` constructs one.)

**Slice: S.**

### Step 4 — Editor Output-Log receipt on rejected import (S).

**Files:** `crates/vox_app/src/shell/mod.rs` (`load_content_asset`, line 1162). The shell currently calls `vox_editor::content_browser::load_asset` which decodes **without** validation. Add the gate where splats enter: after `Ok(LoadedAsset::Splats(splats))`, run `validate_splats(&splats, ValidationBudget::with_max(self.import_budget()))`; if `!report.is_ok()`, push the first receipt line to `self.output_log` (`format!("[content] Rejected {name}: {}", report.receipts()[0])`) and do **not** plant; else log the existing "Loaded … points" line plus a warning-count suffix when `warning_count() > 0`. `import_budget()` returns a fixed constant (e.g. `2_000_000`, matching the proven `scale_trial` ceiling) — no UI needed for the first slice. This keeps the no-panic shell rule (a bad asset is a logged rejection, never a panic).

**Done When:** a headless shell test (mirror an existing `shell` test) loads a hand-built .vxm containing one NaN-position splat and asserts the Output Log's last line `.contains("Rejected")` and `.contains("position")`, and that the scene entity count is unchanged. Concretely: `cargo test -p vox_app shell::tests::rejected_asset_logs_receipt -- --nocapture` prints `test result: ok. 1 passed`.

**Slice: S.**

Total: **S** (the whole gap is S — four S slices, no GPU, no data-model change).

---

## 4. How it fits (integration + dependencies)

**Depends on (already exists, nothing blocking):** `vox_core::types::GaussianSplat` accessors (`position`, `scales`, `is_volume`, `spectral_f32`) — all verified live; `import_asset` dispatch; `import_and_cache`; `vox_tools` clap dispatch; the editor `load_content_asset` seam. No dependency on any unbuilt gap — this is a **Phase 4 leaf that can start in Phase 1** because every surface it touches is already wired.

**What depends on it:** gap **#27 DCC re-import/watch/batch loop** (a batch re-import is only safe if each source is gated — validation is its precondition); gap **#29/#26 dense capture** (a 3DGS-trained cloud must pass the integrity/budget gate before it becomes a shippable asset); gap **#31 full glTF fidelity** (texture-sampled spectral uplift needs the spectral-validity lint to catch zero/NaN bands from a bad texture decode). The spectral-validity lint also **fuses with the wedge**: it is the exact precondition `relight_scene` needs (a zero-radiance splat NaN-poisons intrinsic recovery), so this gate de-risks gap #5 (GPU relight) and #34 (reflectance split).

**Composes with existing systems:** the import pipeline (unchanged decode, new tail), the editor Output Log (`vox_core::output_log` / shell log — receipts in the established "[content] …" format), the `vox_tools` CLI (new subcommand alongside prune/relight/usd-import), and the content-browser cache (`import_and_cache`). It reuses the existing `ImportResult.warnings: Vec<String>` channel — verified live and already surfaced — so non-fatal lints need no new plumbing.

**Must NOT break:**
- **The 11-green-gate invariant + both smoke gates.** The default `splat_budget: None` means existing clean fixtures import exactly as before; the new gate only fires on genuinely degenerate input. Step 2's Done-When explicitly re-runs the 7 pre-existing import tests to prove no regression. The CI smoke binaries (`cargo run --bin walking_sim`, `cargo run --bin ochroma`) load known-good assets, so they stay green.
- **Both-config builds.** The validator and CLI subcommand are default-feature, no new optional features, no `cfg`-gated code — `cargo build` and `cargo build --no-default-features` for `vox_data`/`vox_tools` are unaffected.
- **The no-panic shell rule + `panic = "abort"`.** The gate returns `Err`/logs a receipt; it never `unwrap`s on splat data and never panics. A malformed source is a logged rejection, not a crash.

**Sequencing:** roadmap **Phase 4** ("DCC iteration loop, … validation gates — the daily art-team edit loop"), but it is one of the cheapest items there and has zero architectural prerequisites, so it can be pulled forward to run in parallel with Phase 1 as a standalone hygiene win that immediately hardens every importer.

**Cross-gap seams:** (a) the `ValidationBudget` type is the natural place a future **LOD-bake-on-import** (the other half of roadmap #33) hooks in — bake-on-import can consume the same budget to decide LOD levels; (b) the `ValidationIssue` enum is the schema a future **batch re-import report** (#27) aggregates across many assets; (c) the spectral-validity lint is reused verbatim as the precondition assertion in the **relight** (#5) and **reflectance-split** (#34) test harnesses.

**Out of scope (explicit):** auto-repair/clamping of bad splats (the validator *reports*, it does not mutate — a separate `sanitize_splats` could come later); a GPU twin (this is a one-pass CPU lint over the import buffer, off the frame loop — no oracle/WGSL pair needed); per-band physical-plausibility bounds beyond finite/non-zero (defining a max-radiance ceiling is a future spectral-policy decision).

---

## Surprises & advantages

- **The non-fatal channel already exists and is already surfaced.** `ImportResult.warnings: Vec<String>` (verified `import_pipeline.rs:48`) is propagated by `import_and_cache` into `ImportedAsset.warnings` (`import_helpers.rs:53`) — so warnings (zero-spectral lint) need **zero new plumbing**; we extend an existing, already-displayed Vec. The seed plan assumed we'd add reporting infrastructure; half of it is free.
- **The seed's "exactly one error" test is even cleaner than stated, because of the data model.** `GaussianSplat::volume` takes `position: [f32;3]` directly and stores it raw, so a NaN survives construction untouched — the test needs no file I/O at all, just one constructor call. The roadmap's seed framed it as testing through import; the *unit* test is tighter and faster, and Step 1 is fully GPU-free.
- **Single-dispatch wiring covers three importers in one edit.** Because all of ply/gltf/vxm funnel through one `import_asset` `match` that returns a single `Result<ImportResult,_>` (verified `import_pipeline.rs:52–60`), wrapping the *tail* of `import_asset` gates **every** format at once — no per-importer changes. The effort estimate of S is, if anything, generous.
- **The spectral-validity lint is a genuine first-mover, not a checkbox.** No RGB engine can offer "your asset has a non-finite / all-zero radiance band" because no RGB engine stores per-band radiance. This turns a mundane "import gate" into a wedge-aligned capability: it is the *exact* precondition the relight kernel (#5) needs, so building it here pays down risk on the crown-jewel feature for free.
- **The `is_volume()` accessor makes the surface/volume scale distinction exact, not heuristic.** `scale_w == 0` is legal for every 2DGS disk; without `is_volume()` (verified `types.rs:132`) a naive "all scales > 0" check would false-flag the entire surface-splat population. The data model hands us the discriminant for free, so the lint has zero false positives on legitimate disks.
