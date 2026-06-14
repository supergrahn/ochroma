# Design: Ochroma Authoring / Editor — Unified Edit Loops (2026-06-14)

**Status:** Draft
**Scope:** Answer concretely "how do I use Ochroma to edit an ASSET, edit the CODE, and edit HOW THE GAME WORKS" by joining the three already-real-but-disjoint edit surfaces (asset node-DAG, script hot-reload, sim balance) into one windowed editor with a live edit→see-result loop, GPU-rendered.
**Related:** `[Forge directive factory](../../../...)` (memory), `[GPU-first directive](../...)`, `[Render direction: MESH + RTX Mega Geometry](../...)`, `2026-06-12-sdf-scene-content-roadmap.md`

---

## 1. Problem Statement

The pieces of an editor exist but do not form a loop. Concretely observable today:

- `cargo run -p vox_app --bin ochroma_editor` opens a real 1600x900 egui_dock window, but the Viewport is the **CPU** `SoftwareRasteriser` (`crates/vox_app/src/shell/viewport.rs:16,120`) — authoring decisions are made against a different renderer than ships (`engine_runner` GPU spectral pipeline).
- Editing a node param in the editor **does** live-cook (`apply_param`, `crates/vox_app/src/shell/mod.rs:1008/2125`) but the cooked `SplatizeNode` splats are **never planted into the viewport overlay** — the overlay is fed only by plugin plant paths (`grow_tree`/`raise_terrain`/`generate_building`, drained at `mod.rs:768-780`). So the asset loop's output is invisible.
- Script hot-reload (`RhaiRuntime`/`ScriptWatcher`/`poll_reload`) runs **only** in `crates/vox_app/src/bin/engine_runner.rs:270,599,2928` — grep across `shell/` and `ochroma_editor.rs` for `RhaiRuntime`/`poll_reload`/`ScriptWatcher` returns nothing. You cannot edit behavior and see it in the editor; you must launch the game binary.
- City sim balance is **hardcoded in Rust** (`CareKind::params()` match `service.rs:68-106`; dev consts `game/mod.rs:40-53`; thresholds `progression/mod.rs:24-28`). Changing any rule number requires `cargo build`, violating the config-first law. `urban_horizon` reads no `balance.toml` (no `toml` dep).
- The full spectra-native build chain is slow; there is no fast in-editor preview path for engine/kernel code changes.

---

## 2. Done When

Running:

```
cargo run -p vox_app --bin ochroma_editor -- --frames 120 --shot gpu.png
```

produces `gpu.png` whose Viewport-region pixels come from `TiledSplatRenderer` (a spectral-resolve color the `SoftwareRasteriser` cannot produce — assert a sampled pixel's channel ratio), **and** in the live window: scrubbing the Terrain node's `amplitude` field visibly changes the GPU-rendered terrain within one ~100 ms throttle window; pressing **Play** makes an authored-script entity move in that same GPU viewport and **Stop** restores the exact authored pose; saving a watched `.rhai` changes the running Play view within one frame. A human at the keyboard verifies all four without reading code.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| GPU viewport renders authored splats | `cargo test -p vox_app gpu_viewport_resolve -- --nocapture` asserts a sampled Viewport pixel has `b/r > 1.3` (spectral resolve), unreachable by `SoftwareRasteriser` | `assert!(frame.is_ok())` — passes with empty texture |
| Cook-graph sink → viewport overlay | `cargo test -p vox_app sink_plants_overlay` mutates Terrain amplitude, asserts `shell.overlay_len()` changes by exactly the new sink splat count | `assert!(bridge.sink_splat_count() > 0)` — passes without planting |
| In-editor Play moves entity | `cargo test -p vox_app play_in_editor_moves` runs `PlayController::tick` N frames, asserts entity position delta `> 0.5`, then `stop()` asserts position equals snapshot | `assert!(session.is_some())` |
| In-editor script hot-reload | `cargo test -p vox_app play_reload_changes_behavior` writes a new `.rhai`, polls reload, asserts the per-frame callback's emitted command differs | `assert!(runtime.poll_reload())` returns bool only |
| Data-driven balance, no rebuild | `cargo test -p urban_horizon balance_capacity_changes_gated` sets `balance.care[NursingHome].capacity = 160`, asserts `CivitasGame::tick` yields fewer `gated_workers` than default | `assert!(balance.is_some())` |
| LLM text→forge graph (gated) | `cargo test -p vox_app intent_emits_valid_forge_graph --features forge-native,local-llm` asserts emitted graph passes `forge graph::type_check` and rejects an unbounded one | `assert!(action == AddNode)` |

---

## 4. Architecture

The spine: **ONE windowed editor process** that hosts the GPU spectral renderer plus the live sim/script loop, so all three edit surfaces feed the SAME viewport. All wiring lives in `vox_app/shell` + `vox_editor` + the game crate `urban_horizon` — engine crates (`vox_core`/`vox_data`/`vox_render`) stay game-agnostic and untouched.

### 4.1 GPU Viewport (closes the fidelity gap)

A `GpuViewport` owns a `TiledSplatRenderer` on the editor's existing `WgpuBackend` device (`crates/vox_app/src/bin/ochroma_editor.rs` already creates the wgpu-24 surface + `egui_wgpu::Renderer`). Each frame: `upload_splats_at` the overlay (patch only the changed `asset_range`, not a full re-upload), `render(&mut self, camera) -> Result<TiledFrame, TiledRenderError>` (**note the Result** — consume the Err, the renderer never panics on no-GPU), register `TiledFrame.spectral_texture` via `egui_wgpu::Renderer::register_native_texture`, draw as the Viewport `egui::Image` where `viewport.rs:120` draws the CPU upload today. **Call `free_texture` each frame** to avoid leaking the per-frame `TextureId`. `SoftwareRasteriser` stays as the headless snapshot oracle for `shell_snapshot` tests (deterministic; guards against the known `tile_assign`/`radix_sort` nondeterminism landmines surfacing as flicker). Decision: the GPU viewport renders **only the authored overlay**, dropping the hardcoded `build_scene()` demo base (`viewport.rs:27`).

### 4.2 Live Asset Loop (one genuinely missing wire)

The node-DAG (`crates/vox_editor/src/node_graph.rs`) is the canonical asset substrate: typed ports, `cook()`, deterministic Kahn topo-sort, trailing-edge live re-cook (`request_recook`/`live_cook`, 100 ms budget, `node_graph.rs:226-277,240`). `SplatizeNode` converts `EditorMesh → GaussianSplat[]` with spectral-from-material. `graph_bridge.rs` maps the real graph onto `vox_ui::node_canvas`, instantiated from `vox_editor::templates::instantiate_by_name` (can't drift). **The missing seam (not a no-op confirmation): plant `bridge` sink splats into `shell.overlay`.** Today `apply_param` live-cooks but neither inspector (`mod.rs:2118-2147`) nor intent (`mod.rs:998-1024`) path pushes `get_output(sink,"splats")` (`graph_bridge.rs:602-624`) into the overlay. Add that plant + downstream undo-range bookkeeping. After 4.1, the cooked splats then appear in the real GPU render at the 100 ms throttle.

Also generalize `param_schema` (`graph_bridge.rs:120-146` hand-written match for Terrain/Biome/Vegetation only) into a **descriptor-driven scalar param surface** — a NEW `param_descriptors()` method on the `OchromaNode` trait (the existing `NodeDescriptor` carries ports, not scalar ranges). This unlocks scrubbing ANY node kind, including the building node, with no per-node inspector code.

### 4.3 In-Editor Play (UI wiring of a proven controller)

`PlayController` (`crates/vox_app/src/shell/play.rs` `enter_play:97`/`tick:164`/`stop:194`/`AuthoredSnapshot:57`) is headless-proven only — no `PlayController` reference exists in `shell/mod.rs` or `ochroma_editor.rs`. Surface Play/Pause/Stop on the toolbar: Play snapshots `{entities, overlay}` and starts a fresh `EngineLoop`; while Playing feed `session.render_splats` into the 4.1 `GpuViewport`; Stop restores the pristine snapshot. Thread the editor camera/illuminant into the session so Edit and Play look identical.

### 4.4 In-Editor Behavior Hot-Reload (new wiring, honestly)

Route the editor's `PlayController`/`EngineLoop` through the same `rhai.poll_reload()` + per-script `on_update` path that today only `engine_runner.rs:2926-2939` runs (the real shared substrate is `ochroma_engine::EngineLoop` + `EngineLoop::step_scripts`, also driven by `play.rs`; do **not** "lift engine_runner's loop" — route the existing one). The Rhai bindings stay game-agnostic (entities + 16-band spectral). Combined with `script_gen::generate` (compile-gated `.rhai` to disk, `shell/script_gen.rs:224`) and `intent.rs parse_intent`, the loop is: intent → compile-gated `.rhai` → watcher reload → running Play re-runs → GPU viewport updates. Errors surface in Output Log via last-good-AST + error text. Run the script tick **only inside the PlaySession** so a hung user script can't freeze the editor; Stop drops it.

### 4.5 City Rules as DATA (config-first, game crate)

In `urban_horizon/src/balance/mod.rs` introduce one `BalanceConfig` owning every hardcoded number: `HashMap<CareKind, CareParams>` (replacing the `service.rs:68` match), policy coefficients (`policy.rs:39-67`), `CareEconomy` (`game_mechanics/economy/mod.rs:16-26`), dev consts (`game/mod.rs:40-53`), progression thresholds (`progression/mod.rs:24-28`), wellbeing weights. `default()` returns today's exact literals (behavior unchanged until edited). `CareKind::params()` / `CarePolicies::*` become methods taking `&BalanceConfig` — budget for threading it through all **22** `.params()` call sites (`staffing.rs`, `land_value.rs`, `care/mod.rs`, `ui/mod.rs`, `preview.rs`). `BalanceConfig::load(assets/balance.toml)` mirrors `AssetCatalog::load_project_assets` with load-over-default fallback. Generalize the proven live-lever (`UiAction::TogglePolicy`, `command/mod.rs:822` — mutates `game.policies`, next `tick` re-reads) into `UiAction::SetBalance`. **UI note (corrected fantasy):** civitas has **no egui** — the play UI is a hand-rolled tiny-skia + ab_glyph overlay (`src/render/mod.rs:3-36`, `play.rs:1546`). The Balance panel is a hand-built overlay panel with winit pointer hit-testing, modeled on existing tool panels (`play.rs:1944-1975`), **not** an extension of the test-locked `ui/panels.rs` row gatekeeper. Embed a `BalanceConfig` snapshot in `GameSave` (bump `SAVE_VERSION` past 5; `save.rs` already carries `economy: CareEconomy`) so old saves replay under their authored balance; swap balance only at tick boundaries. Hot-reload `balance.toml` via the `vox_script` `poll_reload` pattern (add `toml`+`notify` deps — absent today). Wellbeing/mental-health is genuinely NEW sim state (per-citizen field + feedback), gated behind a what-if re-sim before it ships into the live loop.

### 4.6 Code / Kernel Edits (honest boundary)

Rust + slang-kernel edits remain a `cargo build` — do not pretend otherwise. Mitigations: (1) keep the script tier (4.4) wide so most gameplay iteration never rebuilds; (2) a **"Reload Game"** toolbar action that relaunches the engine path after a build; (3) **beat the slow spectra-native build** for live editing by keeping `forge-native`/`crucible-native`/`local-llm` features OFF by default (the stock build skips the slow chain and uses the live node-DAG + preview fallback), and using the incremental-cook cache + shift-left seal gate (don't rebuild the binary mid-cache). Surface the single `RenderConfig` as a Settings inspector (config-first law).

---

## 5. Data Models

```rust
/// Owns the GPU renderer + its current frame for the editor Viewport.
/// Lives in vox_app (tooling); engine crates untouched.
pub struct GpuViewport {
    renderer: TiledSplatRenderer,       // private — built on the editor WgpuBackend device
    current_tex_id: Option<egui::TextureId>, // freed each frame via egui_wgpu free_texture
}

/// Game-side, NOT in engine crates. default() == today's hardcoded literals.
pub struct BalanceConfig {
    care: HashMap<CareKind, CareParams>,   // CareParams must gain Serialize/Deserialize (only derives Debug/Clone/Copy/PartialEq today)
    policy: PolicyCoeffs,
    economy: CareEconomy,
    dev: DevConsts,                        // lifted from bare `const` at game/mod.rs:40-53
    progression: ProgressionThresholds,    // lifted from bare `const` at progression/mod.rs:24-28
    wellbeing: WellbeingWeights,
}

impl BalanceConfig {
    pub fn default() -> Self { /* exact current literals */ }
    pub fn load(path: &Path) -> Self;      // load-over-default; missing entries fall back to default
}
```

---

## 6. API

```rust
// --- GPU viewport (vox_render, already exists — reuse exactly) ---
impl TiledSplatRenderer {
    pub fn new_with_capacity(device: &wgpu::Device, queue: &wgpu::Queue, cap: usize) -> Self; // :518
    pub fn upload_splats_at(&mut self, offset: usize, splats: &[GaussianSplat]);              // :831
    pub fn render(&mut self, camera: &Camera) -> Result<TiledFrame, TiledRenderError>;        // :974 — Result!
}
// TiledFrame.spectral_texture: wgpu::Texture (:133); TiledFrame::resolve_to_srgb (:173)
// egui_wgpu::Renderer::register_native_texture(..) -> TextureId; .free_texture(&id)

// --- Cook graph (vox_editor — reuse) ---
impl OchromaNodeGraph {
    pub fn request_recook(&mut self, node: NodeId);   // node_graph.rs:226
    pub fn live_cook(&mut self, budget_ms: u64);      // node_graph.rs:240, default 100ms
}
// NEW trait method to add:
pub trait OchromaNode {
    fn param_descriptors(&self) -> Vec<ParamDesc>;    // scalar name/range/value — NEW, drives inspector
}

// --- Play (vox_app/shell — surface) ---
impl PlayController {
    pub fn enter_play(&mut self, snap: AuthoredSnapshot);  // play.rs:97
    pub fn tick(&mut self, dt: f32);                        // :164
    pub fn stop(&mut self) -> AuthoredSnapshot;            // :194 — restores pristine

}

// --- Script host (route existing EngineLoop, do NOT lift engine_runner) ---
impl EngineLoop { pub fn step_scripts(&mut self); }        // engine_loop.rs:372; rhai poll_reload+on_update

// --- Balance live lever (urban_horizon — generalize TogglePolicy) ---
pub enum UiAction { TogglePolicy(PolicyId), SetBalance(BalanceField, f32), /* ... */ }
// GameApi for scripts (if added): abstract, object-safe, NO vox_sim/CitySim types in signatures
// — lives in vox_script; concrete impl over CitySim lives in vox_app/vox_sim (no vox_script->vox_sim cycle).
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `GpuViewport::render` | Viewport panel draw | `crates/vox_app/src/shell/viewport.rs:120` | replaces `SoftwareRasteriser` upload; consume `Result`, `free_texture` each frame |
| sink-splats → overlay plant | `apply_param` / intent path | `crates/vox_app/src/shell/mod.rs:1008,998-1024` | the one missing asset-loop wire; update undo ranges |
| `OchromaNode::param_descriptors` | inspector build | `crates/vox_app/src/shell/graph_bridge.rs:120-146` | replaces hand-written match |
| `PlayController` Play/Pause/Stop | toolbar action | `crates/vox_app/src/shell/mod.rs` | feed `session.render_splats` into `GpuViewport` |
| `EngineLoop::step_scripts` (poll_reload) | PlaySession frame | `crates/vox_app/src/shell/play.rs` | watcher against `assets/scripts/`; tick only while Playing |
| `BalanceConfig::load` | game init | `urban_horizon/src/game/mod.rs` (init) | mirrors `AssetCatalog::load_project_assets` |
| `UiAction::SetBalance` | hand-rolled Balance overlay panel | `urban_horizon/src/play.rs:1944` style | NOT egui; NOT `ui/panels.rs` |
| forge `blueprint::eval::evaluate` | `BuildingNode::cook` | `vox_editor/src/nodes/building_node.rs` | behind `forge-native`; NEW integration (forge blueprint DAG is not wired today) |

---

## 8. Open Questions

- [x] Building authority: forge blueprint DAG vs `vox_editor::BuildingNode` (a self-described WFC stub) vs `forge_native.rs` flat `BuildingParams` — **pick forge blueprint DAG behind `forge-native`; retire the stub.** (Re-label as NEW integration, not reuse — `blueprint::eval::evaluate`/`forge_catalog`/`type_check` have 0 hits in ochroma today; the only wired forge path is the flat `forge_building::generate`.)
- [x] GameApi for scripts touching `CitySim`: **abstract object-safe trait in `vox_script` (no `vox_sim` types in signatures); concrete impl in `vox_app`/`vox_sim`** — else a `vox_script→vox_sim` cycle + separation violation.
- [ ] Should the descriptor param surface (`ParamDesc`) also carry enum sets (for forge style enums) in v1, or scalars only?
- [ ] Does `CitySim` need to tick inside the rhai-driven `EngineLoop` host before any GameApi can mutate citizens? (Today `engine_runner` runs rhai but NOT `CitySim`; `simulation.rs`/`main.rs` run `CitySim` but not rhai. This bridge is a precondition for "edit a rule live and watch citizens respond.")

---

## 9. Out of Scope

- In-editor Rust/kernel hot-reload — engine code stays a `cargo build` + "Reload Game".
- Full WASM mod ABI execution — `wasm.rs` is a 51-line empty-`Linker` wrapper; `mod_manager.rs` never instantiates. Modding lands later (see §Modding below) once the GameApi contract is stable.
- "See citizen stress shift in the GPU viewport" as a near-term outcome — sim-state→entity→spectral-splat is a separate multi-system bridge, later work.
- Multi-GPU; the dead surfaces `editor.rs` (1252 lines), `main.rs` (728), `gizmo_interaction.rs` (not wired) — left untouched.

---

## Modding / UGC tie-in (WT-1 atomize/instance path)

External and user assets enter the **same** asset loop, not a special case. Import is already broad in `vox_data`: `gltf_import`, `ply_loader`, `spz`, `osm_import`, `colmap_pipeline`, native `.vxm`/`.vxm_v2`; the content browser (`shell/content_panel.rs`) surfaces `.vxm/.spz/.ply/.gltf` drops. An imported mesh becomes splats through the **same `SplatizeNode`/`mesh_to_splats` path** authored assets use, then **atomized→instanced** via the `PointInstancer` direction so UGC props/foliage are data + (later) WASM, never engine edits. The WASM host-import set, when built, must **mirror the Tier-B GameApi** so first-party scripts and third-party mods share one contract; `plugin_system.rs` `is_compatible`/`version_in_range` already exist + are tested but are **unwired** (no running caller) — wire them from day one. WASM is the sandboxed distribution format; Rhai/Lua stay the first-party live-authoring format.

---

## 10. Phased Roadmap (cheapest path to a real edit→see-result loop)

**Phase 1 — GPU viewport (cheapest, biggest win).** Add `GpuViewport` owning `TiledSplatRenderer` on the editor's `WgpuBackend`; render the overlay to a `TiledFrame`, hand `spectral_texture` to egui via `register_native_texture`; consume `render()`'s `Result`; `free_texture` each frame. Keep CPU path for `shell_snapshot`.
*Done when:* `cargo run -p vox_app --bin ochroma_editor -- --frames 120 --shot gpu.png` produces a PNG whose Viewport region has a sampled pixel `b/r > 1.3` (spectral resolve unreachable by `SoftwareRasteriser`).

**Phase 2 — trustworthy live asset loop.** Add the missing `bridge` sink-splats → `shell.overlay` plant (with undo-range bookkeeping); patch only the changed `asset_range` via `upload_splats_at`; replace `param_schema` match with `param_descriptors()`.
*Done when:* in the live window, scrubbing Terrain `amplitude` visibly changes the GPU-rendered terrain within one 100 ms throttle window, AND `cargo test -p vox_app sink_plants_overlay` passes (overlay len changes by the new sink splat count).

**Phase 3 — Play/Pause/Stop in the shell.** Wire `PlayController` onto the toolbar; feed `session.render_splats` into the `GpuViewport`; thread editor camera/illuminant.
*Done when:* pressing Play makes the OrbitMover/authored-script entity visibly move in the GPU viewport, and Stop returns the scene to its exact authored pose (`cargo test -p vox_app play_in_editor_moves` asserts delta then snapshot-equality).

**Phase 4 — in-editor behavior hot-reload.** Route the PlaySession's `EngineLoop` through `poll_reload` + `on_update` against `assets/scripts/`; surface errors in Output Log.
*Done when:* with a Play session running, editing a watched `.rhai` changes the live viewport behavior within a frame, and a deliberate syntax error shows the runtime error in Output Log while last-good behavior keeps running.

**Phase 5 — data-driven balance (config-first).** `BalanceConfig` (default == today's literals); thread `&BalanceConfig` through the 22 `.params()` sites; add `Serialize`/`Deserialize` to `CareParams`; `load(assets/balance.toml)` (+ `toml` dep); `UiAction::SetBalance` + hand-rolled overlay panel; snapshot balance into `GameSave` (bump version).
*Done when:* `cargo test -p urban_horizon balance_capacity_changes_gated` passes (capacity 100→160 yields fewer `gated_workers` from `CivitasGame::tick`), AND editing `balance.toml` and starting a new game yields the new capacity with no Rust change.

**Phase 6 — forge BuildingNode + LLM text→graph (gated).** Behind `forge-native`: NEW integration calling `blueprint::eval::evaluate` → `ForgeAsset`/`Mesh` → splats (verify `ForgeAsset` vs `forge_mesh::Mesh` conversion — different types); async/coarse-LOD cook so it stays under interactive budget. Extend `intent.rs` Llm backend to emit a forge `BlueprintGraph` validated by `graph::type_check` before any cook.
*Done when:* `cargo run --bin ochroma_editor --features forge-native`, add Building node, scrub storeys 2→5 raises sink splat count and the GPU viewport shows the taller building; default build still plants the preview box; `intent_emits_valid_forge_graph` rejects an unbounded LLM graph via `type_check`.

---

## 11. Risks + Corrections (carried from reality-check)

- **GPU-in-egui texture lifetime/threading** (hardest slice): `register_native_texture` binds a `wgpu::Texture` into egui's pass; the `TiledFrame` texture must outlive the paint and be re-registered each frame or `free_texture`'d — else leak / use-after-free. Keep CPU path as fallback if it slips.
- **Known GPU landmines** (`tile_assign` latent bug, `radix_sort` raster nondeterminism): a live-recook viewport surfaces nondeterminism as flicker; keep `SoftwareRasteriser` as the deterministic test oracle.
- **Asset-loop wire is real new code, not "90% there":** the sink→overlay plant does not exist even on CPU today (corrected from the per-axis design's "no new code path").
- **Forge blueprint DAG is NOT wired** (0 hits in ochroma): Phase 6 is NEW integration, not the `forge_native.rs` reuse pattern; pick ONE building authority (forge DAG, retire the WFC stub).
- **Descriptor param surface is NEW API** on `OchromaNode` (existing `NodeDescriptor` carries ports, not scalar ranges) — not reflection over something existing.
- **GameApi separation:** abstract trait in `vox_script` with no `vox_sim` types; concrete over `CitySim` in `vox_app`/`vox_sim` — the literal "GameApi trait in vox_script wrapping &mut CitySim" is a separation violation.
- **CitySim is not ticked by `engine_runner`** (0 hits) — "watch citizens respond live" needs a precondition task co-locating `CitySim` ticking with the rhai-driven host.
- **egui in civitas is fantasy:** the Balance panel is hand-rolled tiny-skia overlay, not egui, and not the test-locked `ui/panels.rs`.
- **`CareParams` is not serde today** (only `Debug/Clone/Copy/PartialEq`); dev consts/thresholds are bare `const` — add derives / lift to fields first. Phase 5 is "foundational/medium," not "cheapest."
- **Recook latency:** a full forge building generate + surfel sample exceeds the 100 ms trailing-edge budget — async cook or coarse-preview LOD is a Phase-6 requirement, not a footnote.
- **Replay determinism:** snapshot `BalanceConfig` (or hash) into `GameSave` (the Terraform-receipt precedent shows the codebase anticipates this); swap balance only at tick boundaries.
- **LLM-emitted graphs** must pass `forge graph::type_check` + clamp-before-cook (`BuildingSpec::clamped`, `apply_param` finite/clamp guards) before cooking.
- **Render-file edit conflicts:** Phase 1 GPU viewport work must land after other agents' in-flight render-file changes settle.

---

## 12. Related Plans / Designs

- Depends on: render-file changes settling (in-flight); `2026-06-12-mesh-megageometry-render-design.md`
- Required before: any "usable editor" milestone / building-content authoring plans
- Related: `2026-06-12-sdf-scene-content-roadmap.md`, Forge directive-factory + LLM-blueprint memory notes, config-first / GPU-first directives
