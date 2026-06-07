> **Adversarial verification:** `sound=false` on one concrete inconsistency: the spec header declares **Effort: M**, but the source roadmap (`2026-06-07-aaa-capability-roadmap.md` lines 51 and 90) rates gap #1 as **Effort: L**. The skeptic judged the actual scope (new `SavedSplatGeom` type + new `splat_codec` module in `vox_data` + new `EntityRange` provenance field + undo-shift coherence + 5 implementation steps) to be consistent with **L**, not M. The grounding refs all verified accurate (`WorldSave` round-trip test, `SavedSplat` fields, `GaussianSplat` public accessors, the `plant_asset`/undo chokepoints). See **Verification corrections** — the design is sound; only the effort label is wrong.

_Status: Draft · Dimension: Editor Workflows for Production Teams · Effort: M · Roadmap rank #1 (74/90)_
_Related: `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #1), `2026-06-06-editor-sota-shell-design.md`_

> Verbatim directive shape: what we need · how it's gonna be · how it's gonna be made · how it fits. Every Done-When names an exact command and an exact human-visible output (repo plan-template rule). No "tests pass."

> **Effort note (synthesis):** header says M; the roadmap and the skeptic say **L**. Treat L as binding for planning — see Verification corrections.

---

## 1. What we need

The windowed editor (`ochroma_editor`) opens a hardcoded demo and discards every edit on close. `file.save` and `file.open` are literal `|| {}` no-ops (`crates/vox_app/src/shell/mod.rs:1749-1750`); the shell's `entities`, `overlay` splats, and `bridge` graph params are never serialized. After this gap closes, a user/developer can:

- **Save the live shell to a real file and re-open it.** Grow a tree (or raise terrain, cook a Crucible scene), `Ctrl+S` to `project.ochroma_world`, quit, relaunch, `Ctrl+O` — the same number of World entities reappear AND the grown tree's overlay splats are present in the viewport. (Today: zero of this survives a relaunch.)
- **Round-trip the spectral splats themselves, not just transforms.** The reopened world's overlay contains the same splat count with the same 16-band radiance — the splat-native wedge persisted, not just an entity name list. (Today even the one existing save path, `engine_runner.rs:1858`, writes `splats: Vec::new()`.)
- **Get an honest Output Log receipt on every save/load** naming the absolute path and the exact entity + overlay-splat count written/read — the editor's provability culture extended to persistence.
- **Trust the file is human-readable and version-tagged** (`WorldSave.version`, pretty JSON) so a save made today still loads after the format grows (`#[serde(default)]` tolerance already proven).
- **Recover the project across a fresh `EditorShell`**, headless — the same code path the windowed editor uses is exercisable with no egui input, so the capability is pixel/state-provable, not click-only.

**Why blocking (Editor Workflows dimension).** The roadmap ranks this #1 (74/90) and calls it "the single hard floor below every other workflow gap — an editor that discards all work on close cannot hold a project, so there is no real authoring, review, or collaboration." Three later gaps depend on the seam built here: #9 Play-in-Editor reuses the world snapshot for Play/Stop restore, #17 duplicate/prefab reuses the entity↔overlay provenance, #25 crash recovery/autosave reuses the document write. It is Phase 1, startable tomorrow on this exact codebase.

---

## 2. How it's gonna be (the design)

### The core problem the data discovers

`EditorShell.overlay: Vec<GaussianSplat>` (mod.rs:212) and `entities: Vec<ShellEntity>` (mod.rs:185) are **two parallel vectors with no link between them.** A `ShellEntity` is only `{name, kind, pos}` (mod.rs:88-92). The only thing that records *which overlay range belongs to which entity* is the undo stack's `UndoEntry::PlacedAsset { name, start, len }` (mod.rs:120). To save and faithfully rebuild, we must persist each planted asset's **splat range as entity-owned provenance** — otherwise a reload cannot re-associate splats with entities or replay them through the real `plant_asset` core.

The second discovery: `SavedSplat` (`vox_data/src/world_save.rs:48`) stores only `{ position:[f32;3], spectral:[f32;16], opacity:f32 }`, but `GaussianSplat` (`vox_core/src/types.rs:29-41`) is a 96-byte struct with **private** fields carrying `kind`, anisotropic `tangent_u/v`, `scale_u/v/w`, `rotation:[i16;4]`, and `spectral:[u16;16]` (f16 bits). **No `SavedSplat`↔`GaussianSplat` conversion exists anywhere in the tree** (verified: zero hits outside `world_save.rs`/its test). A naive map would silently drop the disk geometry and downgrade f16→f32→f16. So the bridge must either (a) extend the saved splat to carry full geometry, or (b) accept the documented lossy reduction. **Decision: extend.** We add a versioned full-fidelity splat record so the splat-native wedge survives a round-trip bit-for-bit, matching the engine's "provability extends to durability" bar.

### Architecture (where each piece lives, and why)

```
            ochroma_editor (winit/egui) ──Ctrl+S/Ctrl+O──┐
                                                          ▼
crates/vox_app/src/shell/mod.rs  (GAME layer)
  EditorShell
    ├─ file.save closure ── pushes ShellRequest::SaveWorld(PathBuf)
    ├─ file.open closure ── pushes ShellRequest::OpenWorld(PathBuf)
    ├─ drain_requests() ──▶ self.save_world(&path) / self.load_world(&path)
    │
    ├─ save_world(&Path)  builds WorldSave from (entities ⊗ overlay-ranges ⊗ bridge params)
    │     │  uses entity_ranges  : Vec<(name, start, len)>  ← NEW provenance
    │     │  uses splat_bridge   : GaussianSplat → SavedSplat (full geometry)  ← NEW
    │     └─ WorldSave::save_to_file (EXISTS, atomic-extendable)
    │
    └─ load_world(&Path)  clears state, replays each SavedEntity's splats
          through the SAME plant_asset core ──▶ overlay + entities + undo rebuilt
                                  │
                                  └─ SavedSplat → GaussianSplat (full geometry)  ← NEW
                                              │
crates/vox_data/src/world_save.rs  (ENGINE layer, game-agnostic)
  WorldSave / SavedEntity / SavedSplat (EXISTS) + SavedSplatGeom (NEW, engine-agnostic)
  splat_codec::{to_saved, from_saved}   ← NEW, lives in vox_data, knows only GaussianSplat
```

**Crate placement rule honored.** The `GaussianSplat`↔`SavedSplat` codec is **pure spectral/geometry math over an engine type** — it goes in `vox_data` (already depends on `vox_core`, already owns `SavedSplat`), keeping it game-agnostic. The *shell orchestration* (which entity owns which range, the `ShellRequest` plumbing, the receipts) is game-workflow logic and stays in `vox_app/src/shell`. No game concept (tree, terrain, building) leaks into `vox_data`: the codec sees only `GaussianSplat`, and the `kind`/`label` provenance rides as the already-generic `SavedEntity.tags` + a `custom_data["overlay_len"]` JSON number.

### New / changed types (verified-or-NEW, full signatures)

**NEW — `vox_data::world_save` (engine, game-agnostic):**
```rust
// Full-fidelity geometry so a 96-byte GaussianSplat round-trips bit-for-bit.
// f16 bands are stored as their raw u16 bits (NOT downcast through f32) so
// load(save(x)) == x for the on-disk splat, matching the world round-trip bar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedSplatGeom {
    pub position:   [f32; 3],
    pub kind:       u32,            // 0=surface(2DGS) 1=volume(3DGS)
    pub tangent_u:  [f32; 3],
    pub scale_u:    f32,
    pub tangent_v:  [f32; 3],
    pub scale_v:    f32,
    pub rotation:   [i16; 4],       // raw quantized quat XYZW
    pub scale_w:    f32,
    pub opacity:    u8,
    pub spectral:   [u16; 16],      // raw f16 bits, byte-exact
}

// On SavedEntity, ADD (with #[serde(default)] so old saves still load):
//   #[serde(default)] pub geom_splats: Vec<SavedSplatGeom>,
// The legacy `splats: Vec<SavedSplat>` field STAYS (back-compat); geom_splats is
// the lossless path the shell writes/reads.
```

**NEW — `vox_data::splat_codec` (engine, game-agnostic):**
```rust
/// Lossless GaussianSplat -> SavedSplatGeom. Reads via the existing public
/// accessors (position(), kind(), tangent_u(), scale_u(), ..., rotation_raw(),
/// opacity(), spectral()) — all already pub on GaussianSplat.
pub fn to_saved_geom(s: &vox_core::types::GaussianSplat) -> SavedSplatGeom;

/// SavedSplatGeom -> GaussianSplat, reconstructing 2DGS via GaussianSplat::surface
/// and 3DGS via GaussianSplat::volume (both pub constructors). NUMERIC CLAMP:
/// opacity and scales are taken as-is from disk but kind is validated (any value
/// other than 0/1 -> 1, never panics); a NaN/inf scale is clamped to 0.0.
pub fn from_saved_geom(g: &SavedSplatGeom) -> vox_core::types::GaussianSplat;
```

**NEW — `EditorShell` provenance + methods (`vox_app/src/shell/mod.rs`, game layer):**
```rust
// FIELD: parallel to `entities`, records each planted asset's overlay range so a
// save knows which splats belong to which entity, and a load can replay them
// through plant_asset. Maintained in plant_asset() and undo() exactly where
// PlacedAsset is pushed/reverted (single chokepoint, mirrors the undo range math).
//   entity_ranges: Vec<EntityRange>,   where EntityRange { name:String, kind:String, start:usize, len:usize }

/// Serialize the full shell document to `path`. Builds one SavedEntity per
/// ShellEntity; planted entities (those with an overlay range) carry their
/// geom_splats sliced from `overlay[start..start+len]`. Bridge params ride in
/// resources via custom_data. Returns the (entity_count, splat_count) it wrote.
pub fn save_world(&self, path: &std::path::Path) -> Result<(usize, usize), String>;

/// Clear entities/overlay/undo/asset_counts, then rebuild from `path`: replay
/// each SavedEntity carrying geom_splats through the SAME plant_asset core
/// (so overlay, numbered entity, undo entry, viewport invalidation all happen
/// exactly as live planting does); restore non-splat entities as ShellEntity.
/// Returns (entity_count, splat_count) loaded.
pub fn load_world(&mut self, path: &std::path::Path) -> Result<(usize, usize), String>;
```

**CHANGED — `ShellRequest` (mod.rs:148):** add `SaveWorld(PathBuf)` and `OpenWorld(PathBuf)` variants; `drain_requests` (mod.rs:1022) gains two arms calling `save_world`/`load_world` and pushing an Output-Log receipt. The `file.save`/`file.open` registry closures (mod.rs:1749-1750) stop being `|| {}` and push these requests to a fixed `project.ochroma_world` in the CWD — exactly mirroring the existing `edit.undo` closure that pushes `ShellRequest::Undo` (mod.rs:1754-1756). No file-dialog dependency is added (rfd is absent today); a real picker is a follow-up slice.

### Key design decisions + rationale

- **Replay through `plant_asset`, do not poke `overlay` directly.** `load_world` reconstructs by calling the same private planting core live planting uses (mod.rs:1117), so entity numbering, undo ranges, viewport-cache invalidation, and receipts are produced by ONE code path. This is the engine's established "headless mirrors the button" discipline (`grow_tree_headless` → `plant_grown_tree` → `plant_asset`). It also means a loaded world is immediately undoable, for free.
- **Full-fidelity `SavedSplatGeom`, raw f16 bits.** Storing `spectral:[u16;16]` as raw bits (not `f32`) makes the splat round-trip byte-exact and avoids an f16→f32→f16 double-rounding that would quietly mutate the wedge's spectral data. JSON serializes `u16`/`i16` losslessly.
- **Entity↔range provenance as an explicit field, not inferred from the undo stack.** The undo stack is capped at `HISTORY_CAP=200` (mod.rs:288) and entries fall off permanently (the wave-14 finding); range provenance must outlive history, so it lives in its own `entity_ranges` vector kept in lockstep at the same chokepoints.
- **Fixed path for the first slice.** No `rfd` dep exists; the seed (and rank-#22 "save versioning") both assume a path is handed in. A fixed `project.ochroma_world` makes the headless Done-When deterministic and unblocks the dependents; the dialog is additive later.
- **Engine crate stays game-agnostic.** The codec in `vox_data` names only `GaussianSplat`; "tree/terrain/building" never appear there. The `kind` string is generic provenance carried on the already-generic `SavedEntity`.

---

## 3. How it's gonna be made (the implementation plan)

### Step 1 — Lossless splat codec + the headless save/load round-trip (the launchable agent task). Slice: M.

**Files:**
- `crates/vox_data/src/world_save.rs` — add `SavedSplatGeom` struct and `#[serde(default)] pub geom_splats: Vec<SavedSplatGeom>` on `SavedEntity` (update `SavedEntity::new` to init it empty).
- `crates/vox_data/src/splat_codec.rs` — NEW: `to_saved_geom` / `from_saved_geom`; declare `pub mod splat_codec;` in `crates/vox_data/src/lib.rs`.
- `crates/vox_app/src/shell/mod.rs` — add `entity_ranges` field (init empty in `new`), maintain it in `plant_asset` (push `{name,kind,start,len}` right where `PlacedAsset` is pushed, mod.rs:1141) and in `undo()`'s `PlacedAsset` arm (remove the matching range and shift later `start`s, mirroring the existing overlay range-shift at mod.rs:958-966); add `save_world` and `load_world`.

**Wiring (same step):** add `SaveWorld`/`OpenWorld` to `ShellRequest`, the two `drain_requests` arms, and replace the `file.save`/`file.open` `|| {}` closures with request-pushers to a fixed path. (Wiring is in THIS step — no "wire later.")

**Done-When (exact command + exact output):**
`cargo test -p vox_app shell::tests::save_then_fresh_open_round_trips_tree -- --nocapture` prints `ok` and the test asserts the real computed outcomes:
```rust
let dir = tempfile::tempdir().unwrap();
let path = dir.path().join("project.ochroma_world");
let mut a = EditorShell::default();
let entities_before = a.entities.len();
a.grow_tree_headless("Silver Birch", "broadleaf", 0);   // real plant path
let grown = a.overlay.len();
assert_eq!(a.entities.len(), entities_before + 1);
assert!(grown >= 200, "tree planted ≥200 splats");
let band7_first = a.overlay[0].spectral_f32(6);          // a concrete band value
let (we, ws) = a.save_world(&path).unwrap();
assert_eq!(we, a.entities.len());
assert_eq!(ws, grown);

let mut b = EditorShell::default();                       // FRESH shell
let (le, lsp) = b.load_world(&path).unwrap();
assert_eq!(le, entities_before + 1, "entity count matches exactly");
assert_eq!(b.entities.len(), entities_before + 1);
assert_eq!(b.entities.last().unwrap().name, "Silver Birch 01");
assert_eq!(lsp, grown);
assert_eq!(b.overlay.len(), grown, "grown tree's splats present");
assert_eq!(b.overlay[0].spectral_f32(6), band7_first, "16-band radiance byte-exact");
assert_eq!(b.overlay[0].rotation_raw(), a.overlay[0].rotation_raw(), "geometry preserved");
```
This is the gap's defining proof: a headless `grow_tree_headless`+save, then a fresh shell open, shows the same entity count and the grown tree's splats present — with a concrete spectral band and the raw rotation asserted equal (never `is_some()`).

### Step 2 — Codec lossless-fidelity unit test in the engine crate. Slice: S.

**File:** `crates/vox_data/tests/splat_codec_test.rs` (NEW).
**Done-When:** `cargo test -p vox_data splat_codec_roundtrip_is_byte_exact` prints `ok`; the test builds a `GaussianSplat::volume` with a non-identity rotation and a distinct per-band spectral signature, runs `from_saved_geom(to_saved_geom(&s))`, and asserts `s2.rotation_raw() == s.rotation_raw()`, `s2.scales() == s.scales()`, `s2.opacity() == s.opacity()`, and `*s2.spectral() == *s.spectral()` (all 16 raw u16 bands equal) — plus a 2DGS `surface` splat asserting `kind()==0` and `tangent_u()` preserved.

### Step 3 — Wire `Ctrl+S`/`Ctrl+O` end-to-end through `drain_requests` and assert the receipt. Slice: S.

**File:** `crates/vox_app/src/shell/mod.rs` (test in the existing `#[cfg(test)] mod tests`).
**Done-When:** `cargo test -p vox_app shell::tests::save_command_writes_file_and_logs_receipt` prints `ok`; the test grows a tree, calls `shell.registry.run("file.save")`, `shell.drain_requests()`, then asserts the fixed path exists on disk AND `shell.output_log.last().unwrap()` equals `format!("[save] Wrote {} — {} things, {} splats", path.display(), n_ent, n_splat)` with the exact counts, and that `file.open` on a fresh shell repopulates `entities`/`overlay` to those same counts. (Exercises the real registry→request→drain path, mirroring `edit.undo`.)

### Step 4 — Bridge graph params + camera/time-of-day into `SavedResources`, prove they restore. Slice: M.

**File:** `crates/vox_app/src/shell/mod.rs` + use `GraphBridge::param_value_of_kind` / `apply_param_by_kind` (graph_bridge.rs:450,485, verified).
**Done-When:** `cargo test -p vox_app shell::tests::graph_param_survives_save_load` prints `ok`; the test does `shell.run_intent("set terrain resolution to 128")`, reads `shell.bridge.param_value_of_kind("Terrain","resolution")` (asserts it equals the clamped applied value, a real number), saves, loads into a fresh shell, and asserts `b.bridge.param_value_of_kind("Terrain","resolution")` returns that same value — graph state, not just splats, persisted.

### Step 5 — Windowed proof via the snapshot binary (visible-output gate). Slice: S.

**File:** `crates/vox_app/src/bin/shell_snapshot.rs` (add a `--save-load` flag that grows a tree, saves, news a shell, loads, then shots).
**Done-When:** `cargo run -p vox_app --bin shell_snapshot -- --grow-tree --save-load --shot /tmp/reopened.png` prints to stderr `[shell_snapshot] reopened: N things in the world, M overlay splats` with `M >= 200`, and writes a non-empty PNG whose pixel readback has `>1%` non-background pixels (the tree's splats rendered after reload) — the same non-black assertion idiom the existing snapshot path uses.

---

## 4. How it fits (integration + dependencies)

**Depends on (must exist first):** Nothing new — this is Phase 1, startable tomorrow. It builds entirely on shipped primitives: `vox_data::world_save::WorldSave` (bit-exact round-trip proven), `GaussianSplat` public accessors/constructors (`vox_core::types`), `EditorShell::plant_asset` + `grow_tree_headless` + `drain_requests` (all live), and the `ShellRequest`/registry plumbing. It does **not** depend on #2 (GpuContext) or any GPU work.

**Depended on by (this is the floor):**
- **#9 Play-in-Editor** — its Play snapshots the world and Stop restores it; the roadmap explicitly says #9 "depends on #1 for the snapshot/restore seam." `save_world`/`load_world` (or an in-memory `WorldSave` variant of them) IS that seam.
- **#17 Duplicate / copy-paste / prefab** — needs the entity↔overlay range provenance (`entity_ranges`) this gap introduces; duplicate is "re-plant a saved range at +offset."
- **#25 Crash recovery + autosave** — reuses `save_world` to write `<project>.recovery` on a dirty tick.
- **#22 Save versioning / forward migration** — extends the `WorldSave.version` dispatch this gap starts writing; #14 runtime-state persistence adds a `runtime_state` section to the same `WorldSave`.

**Composes with existing systems:** the planting core (`plant_asset` and its four callers — tree/terrain/building/scene), the undo stack (loaded worlds are immediately undoable because load replays through the same chokepoint), the content browser (a saved `project.ochroma_world` appears as a scannable asset), and the Output Log (receipts).

**Must NOT break:**
- **The 11-green-gate invariant / both smoke gates.** New code is additive; the existing `world_save_test.rs` round-trip and all `shell::tests::*` plant/undo tests must stay green. The `#[serde(default)]` on `geom_splats` keeps every existing `.ochroma_save` loadable, and the legacy `splats: Vec<SavedSplat>` field is untouched.
- **Both-config builds.** No new crate features; `vox_app` already deps `vox_data` + `serde_json`. Works identically with and without `forge-native`/`crucible-native` (the codec never names those).
- **The no-panic shell rule.** `save_world`/`load_world` return `Result<_, String>` and surface errors as honest Output-Log lines (mirroring `load_content_asset`); `from_saved_geom` clamps NaN/inf scales and out-of-range `kind` instead of panicking — a corrupt or truncated file logs "couldn't load" and leaves the current world intact, never crashes.

**4-phase placement:** Phase 1 ("Floors you can start tomorrow"), alongside #4 (CI green) and #2 (GpuContext). **Cross-gap seam:** the `WorldSave` document is the shared currency — #9 reads it for Play/Stop, #14 extends it with runtime state, #22 versions it, #25 autosaves it. Build the document right here and every dependent inherits a sound foundation.

---

## Surprises & advantages

Discovered while grounding — each is a concrete, grounded reason this gap is **cheaper than its rank-#1 "L" billing suggests** (the author argued an honest M; the skeptic and roadmap hold L — see Verification corrections):

- **`WorldSave` is already production-grade and bit-exact-proven.** `crates/vox_data/tests/world_save_test.rs::test_full_world_round_trip_spectral_and_prefab` already asserts whole-world structural equality across entities, transforms, per-splat 16-band spectral data, AND prefab refs (`assert_eq!(loaded, original)`). The persistence *format* is done and adversarially tested; this gap is almost entirely the **shell↔WorldSave bridge**, not the serializer. The seed under-sells how much is pre-built.
- **`SavedSplat` already exists with a 16-band field and a documented lossless-f32 rationale** — the format authors anticipated splat persistence; the only missing piece is the geometry (tangents/scales/rotation/kind), which the **public `GaussianSplat` accessors already expose** (`rotation_raw()`, `scales()`, `tangent_u/v()`, `spectral()`, `opacity()`, `kind()` — all verified `pub`). No need to touch the private struct layout or unsafe code.
- **The "replay through `plant_asset`" trick gives undo-after-load for free.** Because load reuses the same planting chokepoint, a freshly-loaded world is immediately fully undoable with correct range-tracking — a feature I did not have to design, it falls out of honoring the existing pattern. The `grow_tree_headless → plant_grown_tree → plant_asset` chain is the exact template.
- **First-mover wedge synergy: spectral splats survive the round-trip, transforms-only RGB-engine saves don't carry radiance.** A persisted `project.ochroma_world` is human-readable JSON that contains the literal 16-band radiance per splat — a capture/relight project you can diff, version, and inspect in a text editor. No mainstream engine's scene file carries per-primitive spectral data; this is a small, free differentiator that lands the instant the codec exists.
- **The dependents were architected to consume exactly this.** #9 Play-in-Editor's roadmap text already names "reuse the existing UndoStack + world save" and "#9 depends on #1 for the snapshot/restore seam" — the snapshot API #9 wants is `save_world`/`load_world` (or their in-memory twin). Building #1 well pre-pays #9, #17, #25, and #22 with a single shared `WorldSave` document. That is unusually high leverage for a Phase-1 floor.
- **Honest caveat (not an advantage):** the existing `engine_runner.rs:build_world_save` (line 1858) hardcodes `splats: Vec::new()`, so the *game* binary also throws splats away today. Closing the shell gap with a real `SavedSplatGeom` codec means the same fix can later upgrade `build_world_save` for the game binary at near-zero marginal cost — but that is out of scope here and should not be silently bundled in.

---

## Verification corrections

The skeptic flagged `sound=false`. The single issue, surfaced honestly:

- **Effort label contradiction (the only defect found):** the spec header declares **Effort: M**, but the source roadmap rates gap #1 as **Effort: L** (`2026-06-07-aaa-capability-roadmap.md` lines 51 and 90). The skeptic assessed the real scope — a new `SavedSplatGeom` type, a new `splat_codec` module in `vox_data`, a new `EntityRange` provenance field maintained at two chokepoints with an undo-shift coherence invariant, plus 5 implementation steps and a bridge-param round-trip — and judged it consistent with **L**, not M. For planning, **treat L as binding.** The author's "honest M" argument rests on how much is pre-built (the `WorldSave` serializer is done and tested), which is real leverage, but it does not shrink the new surface area to M.
- **Everything else verified accurate:** the `WorldSave` bit-exact round-trip test, the `SavedSplat` field set, the full set of `pub` `GaussianSplat` accessors the codec needs (`rotation_raw`, `scales`, `tangent_u/v`, `spectral`, `opacity`, `kind`), the `plant_asset`/undo chokepoint line refs, and the `file.save`/`file.open` `|| {}` no-op claim all check out. No false test assertion, no broken Done-When command, no nonexistent API was found. The design is sound; only the effort estimate is wrong.
