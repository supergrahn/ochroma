# Design: Urban Horizon Terraforming Tool (2026-06-11)

**Status:** Approved
**Scope:** A CS2-class terraforming brush (raise/lower/level/smooth) for Urban Horizon that edits the game's heightfield terrain at runtime, bundles each drag into ONE replayable `AuthoredAction`, charges kr per moved m³, blocks edits under the built city, and cascades invalidation into the water mask → geodesic childcare coverage field and the terrain render layer.
**Related:** [Virtualized Splat Rendering design](./2026-06-10-virtualized-splat-rendering-design.md) (M4 is the execution gate), [Game UI Chrome design](./2026-06-10-game-ui-chrome-design.md) (the command spine, tool panel, receipts), `~/Ochroma/projects/urban_horizon/docs/specs/map-system.md` (MapTerrain, derive, editor), [engine terrain-editor plan](../plans/2026-03-28-terrain-editor.md) (prior art — NOT the path taken, see §4.1).

---

## 1. Problem Statement

- Urban Horizon plays on real heightfield maps (`MapTerrain`: heights + the authoritative `CellClass` buildable/water mask) but the player cannot touch the terrain in-game: `MapEditor::sculpt` exists only as the headless map-authoring path (`src/map/editor.rs`), unreachable from the play binary's tool system.
- Terrain shapes the flagship mechanic — `coverage_field.rs` builds the childcare geodesic field from `terrain.is_water` and water severs care reach — yet the only way to get a river where you want one is to author a whole new map offline.
- The live game view renders the ground as a flat ray-cast backdrop at y=0 (`city_scene_inner`: "we do NOT splat a ground grid here"); on a real map the heights the sim consults (`sample_y`, slope culling, water barriers) are invisible to the player.
- Undo is truncate-and-replay over the `AuthoredAction` log, but `rebuilt_with_log` seeds the rebuilt game with the **live** terrain (`fresh.terrain = self.terrain.clone()`, `src/game/mod.rs:1380`) — if terraform strokes mutate terrain and live in the log, every undo would double-apply them. The terrain/log split must be redesigned before any terrain-mutating action can exist.

---

## 2. Done When

Running `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play -- --shot-terraform terraform_shots` exits 0 and prints, with real non-zero numbers:

```
[terraform] raise stroke: <S> stamps · <V> m³ · <K> kr · funds <F0> -> <F1>   (S in 5..=200, V > 500, F1 = F0 - K)
[terraform] flood stroke: water cells flipped <W> (> 0) · childcare field dirty: true
[terraform] undo: heights bit-exact to pre-stroke: true · funds restored: true
wrote terraform_shots/terraform_before.png / terraform_after.png / terraform_undo.png
[terraform] after differs from before on <N> px (N > 20000)
```

and a human opening `terraform_after.png` sees a hill that is absent in `terraform_before.png` and absent again in `terraform_undo.png`. In the windowed game (`cargo run --release --bin play`, New Game → any bundled map → Start Game): arming **Terrain → Raise** shows the brush ring at the cursor, dragging raises visible ground, releasing prints the receipt `Terraformed (Raise) — <V> m³ · <K> kr` in the status bar, and Ctrl+Z restores the ground with `Undid: Terraformed (Raise) — <V> m³ · <K> kr` (byte-identical numbers).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Heightfield stamp on `MapTerrain` | `apply_stamp` on founders_vale raises `sample_y(c)` by a computed Δ; printed `stamp: <cells> cells · <m³> m³` with cells > 50, m³ > 10 | `assert!(stats.is_some())` — passes with a no-op stamp |
| Regional buildability re-derive | lowering a shore cell below `sea_level` flips `terrain.is_water(c)` false→true and the test prints the flipped count > 0 | asserting the function ran without checking any cell class |
| One action per stroke | a 12-stamp drag yields `game.log.len() == before + 1` and the logged `stamps.len() == 12` | counting log growth without asserting the stamp bundle |
| Replay-exact terraform | save → `from_save_on` → `heights.data` bit-compare equal (`to_bits()` per cell), funds equal | comparing lengths or a single sampled height |
| Undo restores terrain | Ctrl+Z after a stroke: every height cell bit-equal to the pre-stroke snapshot, funds restored | `assert!(game.log.is_empty())` only |
| Cost charged through funds | funds delta == `volume_m3 * TERRAFORM_KR_PER_M3` exactly; receipt carries the same kr via `thousands()` | asserting funds changed at all |
| Block under the city | a stamp whose disc overlaps a placed lot returns `Blocked("Terraforming blocked — under <what>")`, heights unchanged, nothing logged; the preview caption equals the stamp verdict (one source) | asserting `!valid` without comparing the two verdict strings |
| Coverage cascade | the flood stroke sets `childcare_field_dirty`; next tick's rebuilt field severs a probe the pre-stroke field reached | asserting the dirty flag without the reach flip |
| Terrain render layer | `terrain_splats(&terrain, 1)` on founders_vale returns > 50_000 splats; the stamped cell's splat y moves by the stamp Δ | `assert!(!splats.is_empty())` |

---

## 4. Architecture

### 4.1 Prior-art inventory: engine voxel terrain vs the game heightfield (the honest decision)

**What the ENGINE has (`~/src/ochroma/crates/vox_terrain`):** a full voxel-SDF terrain stack — `volume.rs` (`TerrainVolume` SDF grid, `sculpt::{add_sphere, remove_sphere, add_ground_plane, add_cliff, add_cave, add_arch}`, `volume_to_splats`), `brushes.rs`/`deform.rs` (brushes over the voxel volume), `scene.rs` (`TerrainScene` facade: volume + material palette + splat-map painting + foliage scatter), `navmesh_bridge.rs` (`extract_from_volume` / `extract_region` NavMesh from the SDF). The 2026-03-28 terrain-editor plan wires that voxel path into the engine editor (`vox_app`). It supports caves, overhangs and arches — none of which a city builder's terraforming needs — and its splat extraction (`volume_to_splats`) is per-surface-voxel with RNG jitter, a different render path from the game's.

**What the GAME has (`~/Ochroma/projects/urban_horizon`):** the runtime terrain is heightfield, not voxel. `MapTerrain` (`src/map/terrain.rs`) wraps a `vox_terrain::heightmap::Heightmap` plus the **authoritative** `CellGrid<CellClass>` buildable mask (Buildable / TooSteep / Water / Reserved). Water is represented as `CellClass::Water` in that mask, derived in exactly one place (`src/map/derive.rs::derive_buildable`): a cell is Water if its height `y <= water.sea_level` **or** its centre lies inside a `WaterBody` outline polygon. The game even already owns heightfield **brush math**: `MapEditor::sculpt` (`src/map/editor.rs`) with `HeightBrush { radius, strength }`, smoothstep falloff, and `BrushOp::{Raise(f32), Lower(f32), Flatten(f32), Smooth}` — used today only for offline map authoring.

**Decision: the game owns the simple heightfield path.** Terraforming reuses the game's existing `MapEditor` sculpt kernel (factored into a shared `src/map/sculpt.rs` so the map editor and the runtime tool cannot drift), edits `MapTerrain` in place, and regionally re-derives the `CellClass` mask. The engine's voxel `TerrainVolume`/`TerrainScene`/navmesh stack is NOT adopted: it solves a different problem (volumetric sculpting for the engine editor), would force a heightfield↔SDF conversion both ways, and its navmesh consumer doesn't exist in the game. Engine crates stay untouched (game-agnostic rule holds — zero engine changes in this design).

### 4.2 The stroke state machine (continuous-apply, one action per drag)

**Decision: continuous-apply, like CS2, with undo as the safety net** — NOT a speculative ghost-then-commit preview. Justification: a height-delta ghost would require a second shadow heightfield plus a speculative render path, all to defer feedback the player will accept anyway; the game already has replay-exact one-keypress undo, the verdict chip + red ring give the *blocked* feedback before any mutation, and CS2 — the UX reference — applies continuously. The accepted trade: a stroke you regret costs one Ctrl+Z (full-replay undo is the accepted M1 cost per the existing undo doctrine).

`CivitasGame` owns an `ActiveStroke` (never serialized). The play binary drives it from the mouse gesture, exactly as zone-corner clicks drive `view.draft` today (input gestures are binary-owned; the **log entry** is the auditable unit):

- `begin_stroke(mode, radius, amount)` on mouse-down with a Terrain tool armed. Refused on the flat world (`terrain.is_flat()`) with the receipt `Terraforming needs a real map`.
- `stroke_stamp(wx, wz)` per mouse-move frame. **Spacing-gated, never time-gated**: a stamp is emitted only when the cursor has moved ≥ `radius * 0.25` from the last applied stamp (determinism: the stamp list must be a pure function of the gesture path, not of frame rate). Each stamp runs the guard verdict (§4.5); blocked stamps are refused and never recorded. Applied stamps mutate `game.terrain` immediately (the continuous feel) and accumulate `volume_m3` (Σ|Δh| · cell_area) and water-flip counts.
- `end_stroke()` on mouse-up: logs **ONE** `AuthoredAction::Terraform` carrying the applied stamp list + params + measured volume/cost, charges `sim.budget.funds -= cost_kr` (the plop pattern — funds may go negative, house behavior), sets `childcare_field_dirty` if any cell flipped to/from Water, clears `undone`, and returns the receipt `Terraformed (<mode>) — <V> m³ · <K> kr`. A stroke whose every stamp was blocked or that never stamped logs **nothing** and receipts `Nothing terraformed`.
- Lifecycle guarantee: `GameView` calls `end_stroke()` before dispatching any non-stamp command, before save, and on exit — a begun stroke is always ended-and-logged, so live state and log can never diverge. Esc does **not** cancel a mid-drag stroke in wave 1 (release commits; undo is the cancel).

### 4.3 Stroke encoding + replay determinism + SAVE_VERSION (the honest call)

**Encoding decision: brush path + params, NOT height-delta grids.** The house precedent is `Zone`: the log stores the drawn outline and replay re-runs `plan_lots` (a pure function of terrain + seed) — inputs, never outputs. `Terraform` stores `{ mode, radius, amount, target, stamps: Vec<[f32;2]>, volume_m3, cost_kr }`. A long drag is ~40–200 stamps × 2 f32 ≈ 0.3–1.6 KB of pretty JSON; a delta grid for the same stroke (an 80 m brush over 4 m cells dragged 500 m ≈ 25k changed cells) would be ~100–500 KB per stroke in the pretty-JSON `.civsave` — two orders of magnitude worse, for no determinism gain we actually lack. Replay re-applies the stamps through the same kernel in the same order: pure f32 arithmetic, fixed row-major cell iteration, no RNG, no dt — bit-exact, the same contract `plan_lots` already lives under (the falloff kernel and per-stamp amounts thereby join the replay contract; that is why `amount` and the Level `target` are stored **in the action**, resolved at stroke time — future brush retuning can never silently re-shape old saves). The stored `volume_m3`/`cost_kr` are the live-measured values: they make the commit and undo receipts byte-match without re-simulation, and replay `debug_assert`s the recomputed volume against them — a free kernel-drift alarm.

**SAVE_VERSION stays 5 — no v6, no migration.** The exact `AddRoad` precedent applies (`src/game/save.rs`: "Purely additive: a v5 save without road actions replays byte-identically"): a new serde enum variant is invisible to saves that don't contain it, so every existing v5/v4/v3/v2/v1 save loads and replays byte-identically. The honest cost, stated plainly: a **save written with terraform strokes cannot be opened by an older binary** — serde fails on the unknown variant and `from_json` errors *loudly* (never a silently-wrong city). That is the same forward-compat behavior `AddRoad` shipped with and is the house rule for when a bump is needed: bump only when an old reader would **misread silently**. It would not. The `SAVE_VERSION` doc comment gains a line recording the variant's addition under v5.

### 4.4 The terrain/log split: `base_terrain` (fixing the undo landmine)

`rebuilt_with_log` today seeds the rebuild with the **live** terrain — correct while no action mutates terrain, fatally wrong the moment one does (undo would replay strokes onto already-stroked ground). `CivitasGame` gains `base_terrain: MapTerrain` — the pristine as-bound terrain, set by `new_small` (flat), `new_on_map`/`bind_map` (the map's terrain). `terrain` remains the live, possibly-terraformed surface every existing query keeps reading (zero call-site churn). `rebuilt_with_log` seeds `fresh.terrain = self.base_terrain.clone()` (and carries `base_terrain` itself); replaying the log's `Terraform` actions rebuilds the live surface exactly. `from_save_on` already binds the map before replay, so load replays strokes onto pristine map terrain — correct by construction. `MapTerrain` is already `Clone` (added for universal undo); the extra resident copy is one heightmap + mask (~4 MB on a 1024² Small map, 0.5 MB on the 256² bundled maps) — accepted.

### 4.5 Guard rails: BLOCK under the built city (M1 cut), one verdict source

Per-stamp verdict `terraform_blocked(game, [wx, wz], radius) -> Option<String>` in `src/ui/preview.rs`, the same one-source pattern as `plop_blocked` (the ghost tint and the stamp refusal share THE function, so preview/commit divergence is structurally impossible). Blocked when the brush disc (centre, radius) intersects: any placed lot footprint (developed **or** vacant — a vacant parcel is authored zoning whose plan must not be invalidated under it), any care building (`game.care.buildings`), any city service building (`game.sim.services.buildings`) within footprint margin, or any road segment within `radius + half-width`. Verdict string: `Terraforming blocked — under <what>` (the clearance-chip idiom). No re-seat, no auto-demolish in M1 — blocked means blocked. Water cells are NOT blocked (terraforming a lakebed/shore is the point); sea-level-derived water responds to height edits; **outline `WaterBody` cells stay Water regardless of height** (derive reads polygon membership) — an outline lake cannot be drained by raising its bed in M1, stated as a known cut. The flat world refuses at `begin_stroke` (§4.2).

### 4.6 Downstream invalidation cascade

```
stamp applied
 └─ MapTerrain.heights edited (kernel) ──────────────► MapTerrain.version += 1
 └─ regional re-derive: CellClass over the stamp AABB + 1-cell ring
     (slope reads neighbours; sea_level + body outlines via the stored WaterMask)
         └─ water flips counted into the stroke
end_stroke
 └─ water flips > 0 ──► childcare_field_dirty = true
         └─ next tick: rebuild_childcare_field_if_dirty → 1024² mask + GPU JFA (~10 ms,
            the measured placement-cadence cost) → geodesic childcare reach updated
 └─ ONE AuthoredAction::Terraform logged · funds charged
render (per frame)
 └─ GameView caches last-seen terrain.version; on change, re-cook the terrain
    splat layer and re-upload residuals (§4.7)
```

- **Water mask → coverage field:** the field mask is water-only (the M1 coverage cut), so only Water-class flips dirty it; slope flips don't. The rebuild stays lazy at the existing placement cadence (`rebuild_childcare_field_if_dirty`), never per stamp.
- **Land value: nothing to invalidate — honestly.** `src/land_value/mod.rs` computes `BASE + parks + schools + health + safety + care + frontage − industry − congestion`; **no terrain input exists** in the formula today. Terrain-fed land value (waterfront, elevation views) is a later wave; this design adds no fake hook.
- **Navmesh: deferred.** The game runs no navmesh; the engine's `navmesh_bridge` consumes the voxel volume the game doesn't use. Out of scope (§9).
- **`MapTerrain` gains the `WaterMask`:** regional re-derivation needs `sea_level` + body outlines, which `MapTerrain` doesn't carry today; `from_map` clones `map.layers.water`, `flat_default` uses `WaterMask::none()`.

### 4.7 Terrain render layer + rebuild scope (rides virtualized M4's residual path)

Today the live view draws NO terrain geometry — flat ray-cast backdrop at y=0. Wave 1 adds a game-side `terrain_splats(terrain: &MapTerrain, spc: u32) -> Vec<GaussianSplat>` — one (spc=1) volume splat per heightmap cell at the cell's height, tinted by `CellClass` (Water dark blue, TooSteep rock grey, Buildable grass green; the brightened-albedo recipe `render_map_3d` proved). On the 256² bundled maps that is ~65k splats — small enough to skip camera-windowing in wave 1 (windowing via the existing `render_region_view` pattern is the Region-map wave). The layer is included only when `!terrain.is_flat()` (the flat/legacy world keeps the clean backdrop — every existing screenshot harness is pixel-unaffected). **Rebuild scope:** the splats join virtualized M4's host-uploaded *residual* set (roads, pads, fallback massing, uploaded via `upload_splats_at`); `GameView` re-cooks them only when `terrain.version` changes — per stroke-in-progress this means at most once per frame while stamping, and the persistent M4 renderer (`[scene] renderer constructed 1x`) absorbs it as a residual re-upload, never a renderer reconstruction. **Known seam, stated honestly:** buildings/roads render at y≈0 regardless of terrain height (placement *probes* use `sample_y` but the splat cook does not seat on it), so terrain drawn at true height can disagree with the city's y where the map isn't flat near the start region. The §4.5 guard blocks strokes under the city, which contains the seam; seating the city on terrain height is its own later wave and is NOT promised here.

### 4.8 UI surface: a new center "Terrain" tab, option rows, the brush ring

Per the layout spec (`docs/references/ui-layout-spec.md`): centre = the build menu, right = inspect/overlays/stats. Terraforming is a build activity (CS2 files it under the build menu's landscaping category), so it becomes the **fifth centre tab "Terrain"** in `build_categories()`, inserted after Utilities and before the last-appended Inspect. Filing it under Utilities (power/water plops) would be semantically dishonest and would bury four brush tools behind a two-tile tab. Honest costs, owned in the plan: `panel_anchor_x`/`toolbar_layout` hardcode 4 build tabs (`hud.rs:861/1215/1585`) → become 5; `play.rs` slices `self.categories[..4]` → `[..5]`; Inspect's category index shifts 4→5; the registry test pinning `27` tools becomes `31` (11+5+8+2+4+1).

- **Tiles** = the four modes: Raise, Lower, Level, Smooth (`BuildTool::Terraform(TerraformMode)`, price `None` — terraforming is priced per m³, not per tile; the cost shows in the ghost caption and receipt).
- **Option rows** (the chrome Show-row idiom, `ToolPanelOptionRow`): `Size` → `20 m / 40 m / 80 m`, `Strength` → `Soft / Medium / Hard` (amount × 0.5/1.0/2.0). Each value button dispatches a registry command (`terrain.size.small…`, `terrain.strength.soft…`) mutating new additive `ViewState` fields — buttons and any future hotkeys share one command id, the chrome rule.
- **Brush ghost** = the care-reach-ring pattern: `hud::draw_world_circle` at the cursor with `view.brush_radius`, ACCENT when `tool_preview(...).valid`, red when blocked; the status-bar centre carries the preview caption (`Raise · 40 m — drag to sculpt` / the blocked verdict / `Terraforming needs a real map`), produced by the same `tool_preview` every armed tool already flows through.
- **Commands:** arming (`build.terrain.raise` …) and brush params are registry commands; the drag itself is an input gesture (like zone-corner clicks), with the logged action as the auditable unit and the receipt pushed through the same `view.receipts`/`status_receipt` surface.

### 4.9 Execution gate (sequenced honestly)

Wave 1 **must not start before virtualized M4 (`docs/superpowers/plans/2026-06-11-virtualized-m4-game-wiring.md`) has executed and demonstrated its Done-When prints** (`[m4] frame: … path=gpu-resident`, `[scene] renderer constructed 1x`). Three hard reasons: (1) the terrain layer is designed onto M4's residual-splat upload path (`upload_splats_at` + persistent renderer) — pre-M4, every per-stroke scene refresh reconstructs the whole `SceneRenderer`, making continuous-apply unusable; (2) M4 rewrites the same `render_gpu/mod.rs` + `bin/play.rs` ownership this wave touches (shared render_gpu/play ownership — sequencing avoids a rebase bloodbath); (3) M4's `--shot-m4` establishes the harness conventions `--shot-terraform` extends. If M4 slips, the only de-gated subset is Tasks 1–2 (pure sim: kernel, action, replay/undo — no render contact); Tasks 3–6 stay gated.

---

## 5. Data Models

All in the game repo (`~/Ochroma/projects/urban_horizon`); engine crates untouched.

```rust
// src/game/save.rs — additive serde variant (SAVE_VERSION stays 5, §4.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerraformMode { Raise, Lower, Level, Smooth }

pub enum AuthoredAction {
    // ... existing variants unchanged ...
    /// One bundled terraform stroke (mouse-down → mouse-up). Stamps are the
    /// APPLIED stamp centres in order (blocked stamps are never recorded);
    /// replay re-applies them through the shared sculpt kernel. `amount` and
    /// `target` are resolved at stroke time so brush retuning can never
    /// re-shape old saves. `volume_m3`/`cost_kr` are the live-measured values:
    /// receipt source + replay drift checksum.
    Terraform {
        mode: TerraformMode,
        radius: f32,
        amount: f32,
        target: Option<f32>,      // Level only: height sampled at the first applied stamp
        stamps: Vec<[f32; 2]>,
        volume_m3: f32,
        cost_kr: f64,
    },
}
```

```rust
// src/map/sculpt.rs — the shared kernel (moved verbatim from MapEditor::sculpt)
pub use crate::map::editor::{BrushOp, HeightBrush}; // or relocated here; editor re-exports

/// What one stamp did to the heightfield.
pub struct StampDelta {
    cells_changed: usize,        // private — .cells_changed()
    volume_m3: f32,              // Σ|Δh| · cell_area over changed cells
    aabb_cells: [u32; 4],        // [x0, z0, x1, z1] inclusive, for regional re-derive
}
```

```rust
// src/map/terrain.rs — MapTerrain grows (fields stay private)
pub struct MapTerrain {
    cell_size: f32,
    origin: [f32; 2],
    buildable: CellGrid<CellClass>,
    heights: Heightmap,
    flat: bool,
    water: WaterMask,            // NEW: sea_level + body outlines for regional re-derive
    version: u64,                // NEW: bumped per applied stamp; render layer watches it
}

/// One applied stamp's downstream facts.
pub struct StampStats {
    cells_changed: usize,
    volume_m3: f32,
    water_flips: u32,            // cells whose CellClass crossed to/from Water
}
```

```rust
// src/game/mod.rs — CivitasGame grows (never serialized)
pub struct CivitasGame {
    // ... existing fields unchanged ...
    /// Pristine as-bound terrain; undo/replay rebuilds start here (§4.4).
    base_terrain: MapTerrain,
    /// The in-flight terraform stroke (mouse-down → mouse-up), if any.
    stroke: Option<ActiveStroke>,
}

struct ActiveStroke {
    mode: TerraformMode,
    radius: f32,
    amount: f32,
    target: Option<f32>,
    stamps: Vec<[f32; 2]>,
    last_stamp: Option<[f32; 2]>, // spacing gate anchor
    volume_m3: f32,
    water_flips: u32,
}

/// Outcome of one stroke_stamp call.
pub enum StampOutcome {
    Applied { cells: usize, m3: f32 },
    Blocked(String),             // the one-source verdict (§4.5)
    TooClose,                    // spacing-gated, no-op
    NoStroke,                    // no begin_stroke active
}

/// kr per moved m³ (tunable const; the logged action stores measured cost).
pub const TERRAFORM_KR_PER_M3: f64 = 10.0;
/// Stamp spacing as a fraction of brush radius (determinism: spacing-, never time-gated).
pub const STAMP_SPACING_FRAC: f32 = 0.25;
/// Base per-stamp height step (m) at Medium strength for Raise/Lower.
pub const STAMP_AMOUNT_M: f32 = 1.5;
```

```rust
// src/ui/build_tools.rs — BuildTool grows; Terrain category inserted at index 4
pub enum BuildTool {
    Zone(ZoneType),
    Care(CareKind),
    Service(CityServiceKind),
    Terraform(TerraformMode),    // NEW
    Inspect,                     // shifts to category index 5
}
```

```rust
// src/command/mod.rs — ViewState additive fields (never reorder existing)
pub struct ViewState {
    // ... existing fields unchanged ...
    /// Terraform brush radius in metres (Size row: 20/40/80; default 40).
    pub brush_radius: f32,
    /// Terraform strength multiplier (Soft 0.5 / Medium 1.0 / Hard 2.0).
    pub brush_strength: f32,
}
```

---

## 6. API

```rust
// src/map/sculpt.rs — the ONE kernel both MapEditor and MapTerrain call.
// Pure f32, row-major cell iteration, smoothstep falloff (s=1-t; s*s*(3-2s)),
// origin assumed [0,0] (the R-Origin invariant from_data enforces).
// Part of the replay contract: changing this kernel re-shapes logged strokes.
pub fn sculpt_heightmap(hm: &mut Heightmap, brush: &HeightBrush, wx: f32, wz: f32, op: BrushOp) -> StampDelta;
// Threading: called on the game thread only. Panics: never (out-of-grid cells skipped).

// src/map/derive.rs — regional re-derivation (same classification as derive_buildable,
// restricted to an inclusive cell AABB inflated by 1 cell for slope neighbourhoods).
// Returns the number of cells whose class crossed to/from CellClass::Water.
pub fn rederive_region(
    hm: &Heightmap, water: &WaterMask, grid: &mut CellGrid<CellClass>,
    max_buildable_slope_deg: f32, aabb_cells: [u32; 4],
) -> u32;

// src/map/terrain.rs
impl MapTerrain {
    /// Apply one brush stamp: kernel edit + regional re-derive + version bump.
    /// No-op Err on the flat world. Game thread only.
    pub fn apply_stamp(&mut self, brush: &HeightBrush, wx: f32, wz: f32, op: BrushOp) -> Result<StampStats, String>;
    pub fn version(&self) -> u64;
    pub fn heights(&self) -> &Heightmap;       // read-only: render layer + tests
}

// src/game/mod.rs — the stroke state machine (§4.2). Game thread only.
impl CivitasGame {
    pub fn begin_stroke(&mut self, mode: TerraformMode, radius: f32, strength: f32) -> Result<(), String>;
    pub fn stroke_stamp(&mut self, wx: f32, wz: f32) -> StampOutcome;
    /// Logs ONE AuthoredAction::Terraform, charges funds, dirties coverage on
    /// water flips, clears redo. Returns the receipt, or None if nothing applied.
    pub fn end_stroke(&mut self) -> Option<String>;
    pub fn stroke_active(&self) -> bool;
}

// src/ui/preview.rs — the one verdict source (preview tint AND stamp refusal).
pub fn terraform_blocked(game: &CivitasGame, pos: [f32; 2], radius: f32) -> Option<String>;

// src/render_gpu/mod.rs — the terrain layer (§4.7).
pub fn terrain_splats(terrain: &MapTerrain, spc: u32) -> Vec<GaussianSplat>;
```

Registry command ids (all dispatched through `GameRegistry::dispatch`, receipts as shown):

| id | action | receipt |
|---|---|---|
| `build.terrain.raise` / `.lower` / `.level` / `.smooth` | `Ui(ArmTool)` (positions from `build_categories()`) | `Armed: Terrain: Raise` |
| `terrain.size.small` / `.medium` / `.large` | `Ui(SetBrushRadius(20.0/40.0/80.0))` | `Brush: 20 m` |
| `terrain.strength.soft` / `.medium` / `.hard` | `Ui(SetBrushStrength(0.5/1.0/2.0))` | `Brush: Soft` |

`describe_action` gains the Terraform arm: `Terraformed (<mode>) — <thousands(volume_m3)> m³ · <thousands(cost_kr)> kr` — read from the stored fields so the commit receipt and `Undid: <same>` byte-match (the house receipt rule).

---

## 7. Wiring

| Component | Called from | File (game repo) | Notes |
|---|---|---|---|
| `sculpt_heightmap` | `MapEditor::sculpt` (delegation, behavior-identical) + `MapTerrain::apply_stamp` | `src/map/editor.rs`, `src/map/terrain.rs` | existing editor tests must stay green untouched |
| `rederive_region` | `MapTerrain::apply_stamp` | `src/map/terrain.rs` | AABB + 1-cell ring |
| `MapTerrain::apply_stamp` | `CivitasGame::stroke_stamp` | `src/game/mod.rs` | after the `terraform_blocked` gate |
| `begin/stroke_stamp/end_stroke` | mouse down/move/up with a Terrain tool armed; `end_stroke` also before any non-stamp dispatch, save and exit | `src/bin/play.rs` (`GameView`) | receipt pushed to `view.receipts` + `status_receipt` |
| `AuthoredAction::Terraform` replay | `CivitasGame::apply_action` | `src/game/mod.rs` | re-applies stamps; `debug_assert` volume checksum |
| `base_terrain` seeding | `new_small` / `new_on_map` / `bind_map` / `rebuilt_with_log` | `src/game/mod.rs` | replaces the live-terrain carry-over at `mod.rs:1380` |
| `childcare_field_dirty` on water flips | `CivitasGame::end_stroke` | `src/game/mod.rs` | rebuild stays lazy in `rebuild_childcare_field_if_dirty` |
| `terraform_blocked` | `tool_preview` (ghost) + `stroke_stamp` (refusal) | `src/ui/preview.rs`, `src/game/mod.rs` | ONE source, the `plop_blocked` pattern |
| Terrain category + `Terraform` tools | `build_categories()` index 4; registry `standard()` | `src/ui/build_tools.rs`, `src/command/mod.rs` | Inspect shifts to 5; tool-count test 27→31 |
| Size/Strength option rows | tool-panel build when a Terrain tool is armed | `src/bin/play.rs` (~line 700, the Show-row site) | dispatch `terrain.size.*` / `terrain.strength.*` |
| Brush ring | per-frame HUD pass when a Terrain tool is armed | `src/bin/play.rs` (the `draw_world_circle` call site, ~line 630) | ACCENT/red by `tool_preview(...).valid` |
| `terrain_splats` + version watch | M4 residual-splat cook in `GameView`; re-cook when `terrain.version()` changes | `src/render_gpu/mod.rs`, `src/bin/play.rs` | only when `!terrain.is_flat()` |
| `--shot-terraform` | arg parse beside `--shot-landvalue` | `src/bin/play.rs` | founders_vale; 3 PNGs + the §2 prints |
| Walkthrough W3 | `script(3)` + `run(3)` + `--walkthrough 3` | `src/walkthrough/mod.rs` | terraform arm → drag → receipt → undo steps |

---

## 8. Open Questions

All resolved in-document: stroke encoding (§4.3: path+params, not deltas), SAVE_VERSION (§4.3: stays 5, AddRoad precedent, loud forward-incompat), continuous vs preview (§4.2: continuous, undo as safety net), under-building policy (§4.5: BLOCK, one verdict source), toolbar placement (§4.8: new fifth centre tab, not Utilities), execution gate (§4.9: after virtualized M4; Tasks 1–2 are the only de-gateable subset).

---

## 9. Out of Scope

- **Slope (start→end ramp) brush and a numeric flatten-to-target option row** — wave 2; wave 1's Level mode samples its target at the first applied stamp (CS2's "level" gesture), which IS flatten-to-target with a sampled target.
- **Draining outline `WaterBody` lakes by raising their bed** — derive reads polygon membership for bodies; only sea-level-derived water responds to height in M1.
- **Re-seating buildings/roads on terraformed ground (allow-with-re-seat)** — M1 blocks instead; the building-y-on-terrain render seam (§4.7) is its own wave.
- **Terrain-fed land value** — the formula has no terrain term today; no hook is added.
- **Navmesh invalidation** — the game runs no navmesh; the engine bridge is voxel-based and unused.
- **Camera-windowed terrain streaming for Region-scale maps** — bundled 256² maps render whole; `render_region_view` is the existing pattern for the later wave.
- **Any engine (`vox_*`) change** — the whole feature is game-side.
- **Tax/funds refusal on insufficient funds** — strokes charge like plops (funds may go negative), matching every existing build action.

---

## 10. Related Plans / Designs

- Depends on: [Virtualized Splat Rendering M4 plan](../plans/2026-06-11-virtualized-m4-game-wiring.md) — **execution gate** for Tasks 3–6 (§4.9).
- Implemented by: [Terraform Tool Wave-1 plan](../plans/2026-06-11-terraform-tool-wave1.md).
- Related: [Game UI Chrome design](./2026-06-10-game-ui-chrome-design.md) (command spine, option rows, receipts, walkthrough), [SDF Pillar design](./2026-06-10-sdf-pillar-design.md) (the coverage field this cascades into), engine prior art [2026-03-28 terrain-editor plan](../plans/2026-03-28-terrain-editor.md) (voxel path, not taken — §4.1).
