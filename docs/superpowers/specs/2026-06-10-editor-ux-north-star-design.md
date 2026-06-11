# Design: Editor/UX North Star — The Most User-Friendly Editor Ever Built (2026-06-10)

**Status:** Draft
**Scope:** The absolute UX bar for every editing surface in the ecosystem — the Ochroma `EditorShell`, the Civitas Care in-game tools, and the asset/graph/animation/terrain editors to come — expressed as testable invariants, the architecture that makes them cheap, a ranked surface roadmap, and phased milestones whose Done Whens are scripted 60-second walkthroughs a human performs at the keyboard.
**Related:** [Editor SOTA Shell design](./2026-06-06-editor-sota-shell-design.md) (the shell this extends), [Living Building Instances design](./2026-06-10-living-building-instances-design.md) (in-flight — designed WITH, not against), [Asset-Audit Remediation plan](../plans/2026-06-10-asset-audit-remediation.md), the beat-Unreal directive (memory: ochroma-beat-unreal-directive)

> **The user's directive, verbatim:** "I want the engine to be the most efficient and the editor/ui to be the most user friendly ever."
>
> The north star is absolute, not relative. We do not measure against Unreal's editor — Unreal violates half of this document's invariants (multi-minute cold starts, shader-compile stalls that freeze the viewport, modal dialogs in the save path, expert-hostile jargon). We measure against the invariants below, the way Linear measures latency and Figma measures live collaboration: a numeric or binary acceptance per principle, enforced by tests, never argued by adjectives. Where Blender 4.x, Figma, or Linear already proved a pattern (single-window docking, palette-as-spine, infinite undo, scrub-everything), we compile it wholesale — "they did the iteration for us."

---

## 1. Problem Statement

Concrete, observable symptoms across the three UI realities that exist today:

- **The engine editor edits a demo, not a world.** `EditorShell` (`crates/vox_app/src/shell/mod.rs`, 4,889 lines) has the SOTA chrome — egui_dock, tokens, Phosphor icons, a fuzzy Ctrl+K palette, range-tracked undo, a working plugin host with Crucible/Forge/FloraPrime tabs — but its viewport is a 480×320 CPU `SoftwareRasteriser` still (`shell/viewport.rs:20-21`) of a hardcoded demo scene (`viewport::build_scene`), re-uploaded only when the overlay changes. It cannot open the actual game (Civitas Care) world, and the GPU directive ("GPU is alfa omega") is unmet on the editor path.
- **The game has zero undo.** `grep -rn "undo" ~/Ochroma/projects/civitas_care/src/bin/play.rs src/game/mod.rs` → zero hits. Drawing a district in the wrong place is permanent until you abandon the session — in a game whose save system is a *replayable action log* (`GameSave { actions, action_ticks, .. }`, src/game/save.rs:62) that makes exact undo nearly free.
- **The game's tools are unreachable by keyboard, palette, or agent.** Every interaction in `play.rs` (1,130 lines) is bespoke control flow: bottom-bar clicks hit-tested by paired `*_at` functions (`hud.rs:440,551,771`), fixed hotkeys, no command registry. The engine shell's one-command-surface (`CommandRegistry`, shell/command_palette.rs:50) stops at the engine repo's edge.
- **Four UI faces coexist.** egui shell (engine editor), UiTree+Vello overlay (engine HUD), a 968-line hand-rolled ab_glyph raster HUD (`civitas_care/src/render_gpu/hud.rs`), and a tiny-skia menu renderer (`civitas_care/src/menu/render.rs`). Only the first two share `Tokens`.
- **Asset quality judgment is a terminal ritual.** The six-view inspection harness (`civitas_care/src/bin/forge_pathtrace.rs` — six views per flagship asset, calibrated pixel gates printing `gates: PASS`/`gates: FAIL <view>`) requires Slang/Vulkan env vars, a release build, and an external PNG viewer. The audit it serves found 55 findings; the judge loop is minutes when it should be seconds.
- **Orphaned editor surfaces.** `vox_render/src/anim_editor_ui.rs` is a floating `egui::Window` (line 36) not docked in the shell, on the deprecated `NodeGraphWidget` rather than `NodeCanvas`; `gizmos.rs` draws into a CPU pixel buffer wired only to the retired `engine_runner` face; `frame_debugger.rs` is a data model with no panel.
- **Nobody measures cold start.** No binary prints time-to-interactive. The invariant "cold start to editing < 2 s" cannot even be checked today.

### 1.1 Honest editor-debt inventory (verified 2026-06-10)

| Surface | What's real today | The debt |
|---|---|---|
| `EditorShell` (vox_app/src/shell/) | Dock, tokens, palette, intent (Ask Ochroma), range-tracked grouped undo (`UndoEntry`, mod.rs:218), plugin host, save/load world, headless `shell_snapshot` proofs | Viewport is a CPU still of a demo scene; no camera orbit; no game-world host; closure commands (`Rc<dyn Fn()>`) can't serialize for replay/scripting |
| Civitas in-game tools (play.rs + hud.rs) | Zone-draft polygon overlay (`draw_zone_draft`, hud.rs:176), CS2-style bottom bar, coverage-circle green/red site preview (`site_has_spare_care`, ui/mod.rs:187), info windows, Inspect tool (in-flight, living-instances design) | No undo, no commands, no palette, no preview of parcel count/cost before commit, every widget needs a hand-written hit-tester, no tokens |
| Asset QA (forge_pathtrace.rs) | Six-view set + calibrated per-view pixel gates + selftest corruption mode | CLI-only; env-var recipe; PNGs judged in a file manager |
| Anim editor (vox_render) | `AnimGraphDefinition` + blend space eval, tested; egui window renders the graph | Floating window, old widget, not connected to the runtime `animation_driver`, not in the shell dock |
| Gizmos (vox_render/gizmos.rs) | Full translate/rotate/scale with hit-test + drag, tested | CPU `[[u8;4]]` overlay; wired only to the retired software face; absent from `EditorShell`'s viewport |
| Frame debugger (vox_render) | Phase-timing model with history | No panel anywhere |
| Cold start | — | Unmeasured everywhere |

---

## 2. Done When

Running `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play -- --walkthrough` prints the active milestone's numbered 60-second checklist to stdout (exact scripts in §8), the **first** log line of the subsequent normal launch reads `[startup] first_interactive_frame_ms=<N>` with `N < 2000`, and a human at the keyboard performs the final (W5) checklist with every line passing — draw a district, watch the live parcel-count ghost, commit it, Ctrl+Z it away, Ctrl+Shift+Z it back identically, drive the same flow entirely through the Ctrl+K palette, and open a risen building's inspector — all without one modal dialog, one viewport stall, or one mouse-only affordance. No code reading required.

Each milestone in §8 names its own exact command and checklist; this Done When is the last one.

---

## 3. Capabilities — the UX invariants (testable, not platitudes)

Every invariant carries a measurable acceptance. An invariant without a number or a binary check is not in this table.

| Invariant | Measurable acceptance | Real behavior test | Stub test (forbidden) |
|---|---|---|---|
| **I1 — Every action previews live before commit** | Any world-mutating tool shows a ghost (tinted geometry + caption) tracking the cursor BEFORE the committing click; the preview's numbers equal the committed result | `cargo test --lib preview::tests::zone_draft_count_matches_commit -- --nocapture`: build a draft polygon, assert `zone_preview(...).parcels == game.zone_lots(...)` committed count; prints `preview 8 == committed 8` | `assert!(preview.is_some())` — passes with an empty ghost |
| **I2 — No operation blocks the viewport** | During any cook/generate/IO job the frame loop keeps presenting: ≥ 25 frames advance during a synthetic 500 ms worker job | `cargo test -p vox_app shell::tests::worker_job_never_stalls_frames`: spawn a 500 ms job through the shell's worker seam, pump the headless frame loop, assert `frames_presented >= 25` and the job result lands as a `ShellRequest` | asserting the job completed (says nothing about frames) |
| **I3 — 100% keyboard-reachable; the palette is the spine** | Every menu/toolbar/bottom-bar affordance dispatches a registered command; the palette fuzzy-finds each | `cargo test --lib command::tests::registry_covers_every_build_tool`: assert a 1:1 mapping from every `build_categories()` tool (ui/build_tools.rs:45) to a command id, and `registry.search("kind")[0].id == "build.care.kindergarten"` | counting commands `> 0` |
| **I4 — Undo is universal and effectively infinite** | Every world mutation is one Ctrl+Z away; undo→redo is byte-exact | `cargo test --lib undo::tests::undo_redo_roundtrip_byte_exact -- --nocapture`: author 5 actions over 30 ticks, undo all 5, redo all 5, assert `game.to_save().to_json()` byte-equals the pre-undo JSON; prints `roundtrip: 5 actions byte-equal` | `assert!(game.can_undo())` |
| **I5 — Cold start to editing < 2 s** | `[startup] first_interactive_frame_ms=<N>` printed by every interactive binary; `N < 2000` release on the 780M box | The walkthrough harness greps its own log for the metric and fails the checklist line if missing or ≥ 2000 | a test that the timer struct exists |
| **I6 — Every property is scrubbable with immediate feedback** | Label-drag on any numeric field mutates the bound value the same frame; world-affecting params show the I1 ghost within 500 ms | engine: existing `inspector::tests::scrub_and_foldout` (drag mutates bound f32 by drag·speed); game: `budget::tests::tax_scrub_applies_same_tick` asserting the scrubbed rate is in `finances()` the next tick | a static label row with a spacer |
| **I7 — Zero modal dialogs in the core loop** | With any window/panel open in build/zone/inspect/param flows, camera orbit and tool hotkeys still work; confirmations are undo-based, never "Are you sure?" | `cargo test --lib ui::tests::info_window_never_captures_camera`: open the city-info window, feed an orbit drag, assert camera yaw changed | asserting the window opened |
| **I8 — Plain language, domain units** | Player/creator-facing labels use domain words and physical units (metres, households, lux); engine jargon lives in tooltips | `cargo test --lib ui::tests::labels_speak_domain`: lint every `tile_label`/panel title against a forbidden-jargon list (`BSDF`, `draw call`, `recook`, `LOD`) — zero hits | spelling-only snapshot |
| **I9 — One look everywhere (tokens)** | Both UI stacks resolve identical bytes from `ochroma.theme.json`; game surfaces adopt the same `Tokens` | existing `cargo test -p vox_ui tokens::tests::egui_and_uitree_resolve_same_accent` (prints `EQUAL`); new: civitas HUD colors asserted equal to `Tokens::color("accent.base")` bytes | hardcoded `Color32`/RGB literals in any new surface |
| **I10 — Agent-operable ⇔ human-operable** | Every walkthrough has a headless twin executing the same command ids and asserting the same checks | `cargo test --lib walkthrough::tests::m1_runs_headless -- --nocapture`: execute the M1 script's command list against a headless game, assert each step's check; prints the checklist with `[x]` per line | running the commands without asserting the checks |
| **I11 — Every action returns a receipt** | Each dispatched command appends one human-readable line (what happened, with numbers) to the status/log surface, including undo ("Undid: …") | `undo::tests::receipts_name_the_action`: zone, undo, assert the log's last two lines are `Zoned Res Low — 8 parcels` then `Undid: Zoned Res Low — 8 parcels` | asserting log length grew |

These invariants are regression locks, not launch goals: once a milestone makes one true on a surface, its test stays in CI forever.

---

## 4. Architecture — what makes the invariants cheap

### 4.1 The UI-stack decision: immediate-mode editor face, owned raster game frame, one token spine

Grounded in what actually exists and works: the editor face **is and stays egui** (immediate mode) — `EditorShell` on egui 0.31/egui_dock/wgpu 24 already delivers docking, palette, scrub fields, plugin styling enforcement, and headless pixel proofs (`cpu_render.rs` rasterizes real egui frames CPU-side for CI). Immediate mode is what makes I6 (scrub-everything) and I10 (headless twins) nearly free: there is no retained widget tree to invalidate or diff — state IS the UI. We explicitly do **not** build or adopt a retained-mode framework for the editor; the UiTree+Vello retained path remains exactly what the shell design scoped it to: game-rate viewport overlays (HUD, perf graph, gizmo labels). Civitas's in-game HUD **stays an owned raster overlay** (it composites over the softbuffer game frame at game rate and is fully headless-testable), but it adopts the two spines: `vox_ui::Tokens` for every color/spacing it draws, and the command registry (§4.2) for every action it triggers. The convergence line (Vello 0.5+/wgpu 24 unification) is unchanged from the shell design. **Decision: no fifth UI stack, ever; the existing four reduce toward three (menu renderer folds into the HUD raster kit at M2).**

### 4.2 The command spine: one registry, two dialects — closures in the editor, data in the game

The engine shell already proves the one-command-surface (menus, toolbar, palette, keys, and Ask-Ochroma all dispatch `CommandRegistry::run(id)`; mutations queue `ShellRequest`s drained next frame). The game gets the same spine with one deliberate difference: **game commands are data, not closures.** A `GameCommand` resolves to either a `Ui` action (arm a tool, open a panel — not logged) or an `Author` action — a `save::AuthoredAction` value (save.rs:35). That single decision buys three invariants at once:

- **I4 undo for free:** authored actions already ARE the save format; undo = truncate the log and replay (§4.3).
- **I10 scripting for free:** a walkthrough script is a list of command ids + check strings, serializable, executable headless.
- **I3 palette for free:** the palette searches the same registry the bottom bar dispatches; the existing `draw_palette`/`palette_row_at` raster widgets (hud.rs:797,851) render it in-game, the existing fuzzy ranker (`CommandRegistry::search`'s exact > prefix > word-prefix > substring > subsequence ladder) ranks it.

Every input path — bottom-bar click, hotkey, palette Enter, AI intent, walkthrough runner — converges on `GameRegistry::dispatch(id)`. No tool logic is ever called around the registry (the engine shell's documented rule, extended to the game).

### 4.3 Undo: replay is the transaction log (designed WITH living-instances)

The living-instances design (§4.7 there) hardens exactly the property game-undo needs: **every per-tick rule is a deterministic pure function of replayed state** — no RNG, no wall clock, no hash-order dependence. So undo is `from_save_on(&truncated_save, &map)`: drop the last `(action, tick)` pair, replay. Redo re-appends the popped pair and replays forward. Byte-exactness is guaranteed by the same mechanism that makes saves exact, and the I4 roundtrip test enforces it. Cost control: replay from tick 0 is acceptable for young cities; a periodic in-memory checkpoint (a cloned `CivitasGame` every `CHECKPOINT_EVERY = 256` ticks, replay only the suffix) bounds undo latency to a budget of **< 100 ms** at 10k ticks — the checkpoint is an optimization invisible to semantics, so it can land after M1. The engine shell keeps its existing `UndoEntry` stack (range-tracked, grouped transactions) — the two undo systems never mix because the two worlds never mix mid-session.

### 4.4 Live-preview transactionality: ghost → commit | cancel

Formalize what both codebases half-do already (the shell's `overlay` + range-tracked planting; civitas's zone draft + green/red coverage preview) into one rule: **a tool is a transaction.** While armed, it produces a `ToolPreview` every frame — ghost geometry (tinted, never written to world state) plus a caption with the I1 numbers ("≈ 8 parcels · −12,400 kr"). Commit dispatches the `Author` command (entering the log, undoable); Esc/right-click cancels by simply dropping the preview. Previews are computed from the same pure functions the commit uses (`plan_lots` for zoning, `site_has_spare_care` for plops), so preview/commit divergence is structurally impossible — that is the I1 test. Nothing about a preview touches the save log, which is what keeps I7 honest: there is never a "discard changes?" modal because uncommitted state is, by construction, discardable.

### 4.5 The viewport is never blocked, never stale

Two rules. (1) **Long work happens on workers:** cooks, Forge/Crucible generation, asset IO run on `std::thread` workers that post results as `ShellRequest`s (engine) / queued events (game) — the existing sink pattern (`flora_sink`/`forge_sink`/`scene_sink`) generalized; the UI thread never joins a worker. (2) **The editor viewport becomes a live GPU frame:** the splat frame renders on the engine's wgpu-24 device and enters egui via `Renderer::register_native_texture` (already specified in the shell design's test 7; unimplemented debt). The 480×320 CPU still and its `viewport_tex` invalidation dance are retired on the same milestone. Gizmos (`vox_render::gizmos`) re-target the egui overlay layer (screen-space lines via `egui::Painter` over the viewport image) so their tested hit-test/drag math survives while the CPU pixel-buffer path retires with `engine_runner`'s face.

### 4.6 Cold start is a measured budget, not a hope

Every interactive binary records `std::time::Instant::now()` first in `main` and prints `[startup] first_interactive_frame_ms=<N>` on its first presented frame. Budget allocation for `play` (release, 780M): process+window < 300 ms, catalog/payload lazy-load < 700 ms (load the menu instantly; load `ready_assets` behind it), first city frame < 1000 ms. The shell gets the same metric in `ochroma_editor`. Anything over budget is a named regression with a number attached — the Linear discipline.

### 4.7 The ranked editor-surfaces roadmap

Ranked by leverage for the product thesis (people with good game ideas; world/asset creation through ecosystem plugins; AI as a first-class creator):

1. **In-game city tools polish (build / zone / inspect flow)** — the product IS the game; this surface is touched every session. Scope: command spine + undo + I1 ghosts + palette + the living-instances inspector. The in-flight inspector window (living-instances §4.5) ships as designed — this roadmap then makes it non-modal-verified (I7), palette-armable (`tool.inspect`), and pinnable; its §2 Done When is untouched.
2. **Asset inspection — the six-view harness as an editor panel ("Asset Lab")** — asset quality is the current bottleneck (the audit's 55 findings) and the judge loop is the slowest in the project. An Asset Lab tab renders a chosen `ReadyAssetPayload` in the six-view grid with per-view gate chips (PASS/FAIL + mean-luma numbers inline), and scrubs `condition` / `seed` / variant with I6 feedback. The CLI harness remains the CI oracle; the panel is the human loop.
3. **Node graphs (materials / Forge directives)** — `NodeCanvas` + `GraphBridge` + live-cook exist in the shell; the step is making the Forge directive JSON a first-class graph whose param scrubs drive a viewport ghost (I1/I6) and whose Commit plants with undo. This is the ecosystem-plugin product surface, directly on the beat-Unreal priorities (graphs).
4. **Animation / sequencer** — converge `anim_editor_ui` into a shell dock tab on `NodeCanvas` via a `GraphModel` adapter over `AnimGraphDefinition`; a sequencer timeline panel follows only once a game consumer exists (the seq_ecs plugin currently has none).
5. **Terrain editor** — deliberately last: for our persona, graph-driven Forge terrain beats brush sculpting as the primary flow; manual brushes (vox_terrain, planned 2026-03-28) are a refinement layer to wire after graphs prove the loop.

### 4.8 The walkthrough harness — acceptance as an instrument

Each milestone's Done When is a script: an ordered list of `WalkthroughStep { say, command, check }`. `--walkthrough` prints it as a numbered human checklist (the exact text in §8). The headless twin (`walkthrough::run(milestone)`) executes each step's command id against a real game/shell instance and asserts each check — the same affordance agents use (I10). The walkthrough is data in the repo, so the printed checklist and the CI assertion can never drift apart.

---

## 5. Data Models

All game-side types live in `~/Ochroma/projects/civitas_care` (GAME layer); engine crates gain no game concepts. Fields follow the codebase's existing `pub`-field convention.

```rust
// civitas_care: src/command/mod.rs (NEW)

/// One game command — the unit of the one-command-surface, as DATA so it
/// serializes into walkthrough scripts and resolves into the authored-action
/// log (unlike the engine shell's Rc<dyn Fn()> closures, which can't).
#[derive(Debug, Clone, PartialEq)]
pub struct GameCommand {
    pub id: &'static str,        // "build.zone.res_low", "edit.undo", "view.info"
    pub title: String,           // plain language (I8): "Zone: Low-density homes"
    pub category: &'static str,  // "Build" | "Edit" | "View" | "Sim"
    pub shortcut: &'static str,  // "Ctrl+Z", "1".."9", "" if none
    pub action: GameAction,
}

/// What a command does. The Author/Ui split is the undo boundary (§4.2).
#[derive(Debug, Clone, PartialEq)]
pub enum GameAction {
    /// Arms a tool or changes view state. Not logged, not undoable.
    Ui(UiAction),
    /// Mutates the world: becomes a `save::AuthoredAction` appended to the
    /// replay log at the current tick. Always undoable (I4).
    Author(AuthorKind),
    Undo,
    Redo,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UiAction {
    ArmTool { category: usize, tool: usize }, // indices into build_categories()
    OpenInfo,
    OpenPalette,
    TickOnce,                                  // Space
    CloseTop,                                  // Esc: preview > window > palette
}

/// Authoring commands carry the tool identity; the cursor/outline payload is
/// supplied at dispatch time by the armed tool's committed ToolPreview.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthorKind {
    CommitZone,     // -> AuthoredAction::Zone { .. }
    CommitPlop,     // -> PlaceCare { .. } | PlaceService { .. }
}

/// The single dispatch surface (the game twin of vox_app's CommandRegistry).
pub struct GameRegistry {
    commands: Vec<GameCommand>,
}

/// One frame's tool transaction state (§4.4) — pure presentation + numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolPreview {
    pub caption: String,      // "≈ 8 parcels · −12,400 kr" (I1 numbers)
    pub valid: bool,          // green/red ghost tint
    pub parcels: usize,       // zoning: plan_lots count; plops: 1
    pub cost: f64,
}

// civitas_care: src/walkthrough/mod.rs (NEW)

/// One line of a milestone walkthrough — printed for humans, executed headless.
#[derive(Debug, Clone)]
pub struct WalkthroughStep {
    pub say: String,                  // the printed checklist line
    pub command: Option<&'static str>,// GameCommand id the headless twin runs
    pub check: Check,                 // the assertion both twins share
}

#[derive(Debug, Clone)]
pub enum Check {
    LogEndsWith(String),              // receipt assertions (I11)
    ParcelDelta(i64),                 // world-state assertions
    SaveBytesEqual,                   // I4 roundtrip
    StartupUnderMs(u64),              // I5
    Manual(String),                   // human-eye lines ("the ghost follows the cursor")
}
```

Constants: `CHECKPOINT_EVERY: u64 = 256` (undo replay checkpoint cadence, §4.3), `PREVIEW_BUDGET_MS: u64 = 500` (I6 world-ghost latency), `STARTUP_BUDGET_MS: u64 = 2000` (I5).

---

## 6. API

Existing signatures this design binds to (verified in code today — implementations must match, not reinvent):

```rust
// ENGINE SHELL (vox_app/src/shell/command_palette.rs) — the proven spine being mirrored
pub struct Command { pub id: String, pub title: String, pub category: String,
                     pub shortcut: String, pub run: Rc<dyn Fn()> }       // :19
impl CommandRegistry {
    pub fn add(&mut self, cmd: Command) -> &mut Self;  // same-id REPLACES (:66)
    pub fn run(&self, id: &str) -> bool;               // :77
    pub fn search(&self, query: &str) -> Vec<&Command>;// fuzzy ladder (:90)
}
// vox_app/src/shell/mod.rs
impl EditorShell {
    pub fn ui(&mut self, ctx: &egui::Context);                    // :720
    pub fn run_intent(&mut self, text: &str) -> String;           // :855
    pub fn undo(&mut self) -> String;                             // :1192
    pub fn install_plugin(&mut self, p: Box<dyn host::EditorPlugin>); // :561
}

// GAME (civitas_care) — the surfaces being commanded
pub fn build_categories() -> Vec<(&'static str, Vec<BuildTool>)>; // ui/build_tools.rs:45
pub enum AuthoredAction { Zone{..}, PlaceCare{..}, PlaceService{..}, AddJobs{..}, SeedHousehold{..} } // save.rs:35
impl CivitasGame {
    pub fn to_save(&self) -> GameSave;                            // game/mod.rs:1066
    pub fn from_save_on(save: &GameSave, map: &Map) -> Self;      // game/mod.rs:1118
    pub fn instance_at(&self, p: [f32; 2]) -> Option<InstanceId>; // living-instances design §6
}
// hud raster kit reused for the in-game palette (render_gpu/hud.rs)
pub fn draw_palette(px: &mut [[u8;4]], w: u32, h: u32, items: &[(String, bool)]); // :797
pub fn palette_row_at(w: u32, x: f32, y: f32, n: usize) -> Option<usize>;          // :851
```

New, this design's contract (single-threaded game-side unless noted):

```rust
// civitas_care: src/command/mod.rs
impl GameRegistry {
    /// Build the full registry FROM build_categories() + edit/view/sim verbs,
    /// so registry coverage of the bottom bar is by construction (I3 test
    /// asserts the 1:1 anyway, locking against future drift).
    pub fn standard(game: &CivitasGame) -> GameRegistry;
    /// Dispatch by id. Ui actions mutate view state; Author actions append to
    /// the log + apply; Undo/Redo run §4.3. Returns the receipt line (I11)
    /// or Err(closest-matches) for an unknown id.
    pub fn dispatch(&self, view: &mut GameView, game: &mut CivitasGame, id: &str)
        -> Result<String, Vec<String>>;
    pub fn search(&self, query: &str) -> Vec<&GameCommand>; // reuse the shell's fuzzy ladder
}

impl CivitasGame {
    /// I4: truncate the authored log by one and replay (checkpoint-accelerated).
    /// Returns the undone action for the receipt; None if the log is empty.
    pub fn undo_last_action(&mut self, map: &Map) -> Option<AuthoredAction>;
    /// Re-append the most recently undone (action, tick) and replay forward.
    pub fn redo_last_action(&mut self, map: &Map) -> Option<AuthoredAction>;
}

// civitas_care: src/ui/preview.rs
/// Pure preview of the armed tool at the cursor — the SAME functions commit
/// uses (plan_lots / site_has_spare_care), so I1 holds by construction.
pub fn tool_preview(game: &CivitasGame, tool: BuildTool, draft: &[[f32; 2]],
                    cursor: [f32; 2]) -> ToolPreview;

// civitas_care: src/walkthrough/mod.rs
pub fn script(milestone: u8) -> Vec<WalkthroughStep>;
/// Print the human checklist (the --walkthrough flag's body).
pub fn print(milestone: u8) -> String;
/// Headless twin: run every step's command on a real game, assert every check.
/// Panics with the failing step's `say` line. (I10)
pub fn run(milestone: u8) -> Result<(), String>;

// ENGINE: vox_app/src/shell/viewport.rs (replaces the CPU-still path, §4.5)
/// Register the live wgpu-24 splat frame as the Viewport tab's texture.
/// Re-registers on resize. No CPU readback on the display path.
pub fn register_live_frame(renderer: &mut egui_wgpu::Renderer, device: &wgpu::Device,
                           view: &wgpu::TextureView) -> egui::TextureId;
```

Threading: game registry/undo/preview are single-threaded game state like every other `CivitasGame` field. Worker jobs (cooks, generation) communicate only via the existing sink/queue pattern; no locks on the UI thread. Panics: `dispatch` never panics on unknown ids (returns nearest matches — the shell's `nearest_commands` behavior, intent.rs:1180).

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `GameRegistry::standard()` | `GameView::new` | `civitas_care/src/bin/play.rs` (~:112) | built once per game session |
| `GameRegistry::dispatch()` | every input arm: bottom-bar hit, hotkey match, palette Enter | `play.rs` `window_event` (~:804) + `key` (~:543) | NOTHING calls tool logic around it |
| `tool_preview()` + ghost draw | `GameView::render`, while a tool is armed | `play.rs` (~:452) + `hud::draw_zone_draft` extension | caption drawn beside the cursor |
| `undo_last_action` / `redo_last_action` | `dispatch` Undo/Redo arms (Ctrl+Z / Ctrl+Shift+Z) | `src/game/mod.rs` | replay via existing `from_save_on` |
| in-game palette (Ctrl+K) | `GameView::render` overlay + click/key routing | `play.rs` + `hud.rs` `draw_palette`/`palette_row_at` | searches `GameRegistry` |
| `[startup] first_interactive_frame_ms` | first presented frame | `play.rs::main` + `redraw`; `vox_app` `ochroma_editor.rs` | I5 instrument, both repos |
| `walkthrough::print` (`--walkthrough`) | arg parse in `main` | `play.rs::main` (~:1110) | prints §8 checklists |
| `walkthrough::run` | `#[test]` per milestone | `src/walkthrough/mod.rs` tests | the I10 headless twins |
| Asset Lab panel | new tab in the game's edit surface (M3) | `civitas_care/src/bin/asset_lab.rs` reusing `forge_pathtrace`'s view/gate fns (read-only — that binary is owned by the porch-closure agent until released; extract shared fns to a module, do not edit it) | gates inline as PASS/FAIL chips |
| `register_live_frame` | Viewport tab arm of `EditorShell::ui` | `vox_app/src/shell/viewport.rs` | retires the 480×320 CPU still |
| gizmo overlay re-target | viewport overlay painter | `vox_app/src/shell/mod.rs` viewport arm | reuses `gizmos.rs` math, drops the pixel-buffer path |
| `Tokens` into the game HUD | color constants in `hud.rs` become `Tokens` reads | `civitas_care/src/render_gpu/hud.rs` | I9 across repos |

---

## 8. Phased Milestones — every Done When is a scripted 60-second walkthrough

Phasing orders the build; the invariant bar never drops. Each milestone's checklist is printed verbatim by `--walkthrough` (walkthrough data and checklist cannot drift, §4.8), and each has a headless twin test.

### M1 — "Undo the city" (command spine + universal undo + I1 ghosts)
Command: `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play -- --walkthrough` then `cargo run --release --bin play`
```
W1 — 60 seconds:
 1. [ ] New Game → any scenario → Start Game. The log's first line shows
        [startup] first_interactive_frame_ms=<N> with N < 2000.
 2. [ ] Zones → Res Low. Move the mouse: the draft ghost shows a tinted fill
        and a caption "≈ N parcels · −<cost> kr" tracking the cursor.
 3. [ ] Right-click to commit. Status bar prints "Zoned Res Low — N parcels"
        and N matches the ghost's number.
 4. [ ] Ctrl+Z. The district vanishes; status prints "Undid: Zoned Res Low — N parcels".
 5. [ ] Ctrl+Shift+Z. It returns with the SAME parcel count.
 6. [ ] Care → Kindergarten. The site ghost is green inside a served gap,
        red elsewhere, BEFORE you click. Place it; Ctrl+Z removes it.
 7. [ ] Press i (city info), then drag middle-mouse: the camera still orbits
        with the window open. Esc closes the window only.
```
Engine-side ride-along: the I5 metric lands in `ochroma_editor` the same milestone.

### M2 — "The palette is the spine" (keyboard-complete + in-game Ctrl+K)
Command: `cargo run --release --bin play` (after `-- --walkthrough` prints this list)
```
W2 — 60 seconds:
 1. [ ] Ctrl+K opens the palette over the running city; the sim frame behind
        it keeps animating (clouds/water move).
 2. [ ] Type "kind" → top hit "Build: Kindergarten" → Enter arms the tool;
        the bottom bar highlights it.
 3. [ ] Type "undo" in the palette → Enter → the last action reverts with its receipt.
 4. [ ] Zone a district touching the mouse ZERO times: Ctrl+K "res low",
        arrow keys to steer the draft cursor, Enter to drop points, Enter on
        "Commit zone". The receipt prints.
 5. [ ] Hover any bottom-bar tile: the tooltip shows its command id and shortcut.
 6. [ ] cargo test --lib command::tests::registry_covers_every_build_tool passes
        (the 1:1 lock — printed by the walkthrough runner).
```

### M3 — "Asset Lab" (the six-view harness becomes a panel)
Command: `cd ~/Ochroma/projects/civitas_care && LD_LIBRARY_PATH=$HOME/slang-sdk/lib SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SPECTRA_BACKEND=vulkan cargo run --release --features spectra --bin asset_lab`
```
W3 — 60 seconds:
 1. [ ] The Asset Lab opens listing every cooked payload in the catalog
        (craftsman, rowhouse, victorian, school, police, …) with thumbnails.
 2. [ ] Click "Forge Craftsman House": the six-view grid renders (iso, front,
        side, rear, top, detail) with a PASS/FAIL gate chip and the mean-luma
        number on each view — the same gates forge_pathtrace prints.
 3. [ ] Scrub Condition New → Weathered: the views re-render and the receipt
        names the changed material ids. Scrub Seed: the variant changes.
 4. [ ] A failing gate view shows its band ("mean 61.2, band 77.6–84.5") inline —
        no terminal, no PNG viewer, no env-var spelunking for the judge loop.
```

### M4 — "Graphs drive the world" (Forge directive graph, live ghost, undoable plant)
Command: `cargo run -p vox_app --release --bin ochroma_editor`
```
W4 — 60 seconds:
 1. [ ] The Node Graph tab shows a Forge building directive as a graph
        (storeys/footprint/style params on nodes, typed colored wires).
 2. [ ] Scrub "Storeys" 2 → 4 on the node: a ghost building in the viewport
        updates within 500 ms, viewport never freezes (the spinner is on the
        node header, not the frame).
 3. [ ] Press Enter on "Add to world": the building plants, the World panel
        gains "Forge Building NN", the Output Log prints the receipt.
 4. [ ] Ctrl+Z removes exactly that building (the demo scene's other assets
        are untouched). Ctrl+K "storeys" finds the param as a command.
```

### M5 — "One world, one editor" (the marquee; the §2 Done When)
Command: `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play -- --walkthrough` then `cargo run --release --bin play`
```
W5 — 60 seconds:
 1. [ ] [startup] first_interactive_frame_ms < 2000 on the 780M box.
 2. [ ] Perform W1 steps 2–6 (ghost → commit → undo → redo → plop preview).
 3. [ ] Perform W2 step 4 (the zero-mouse district).
 4. [ ] Space ×10, Tab to Inspect, click a risen house: the Building window
        (living-instances §2 shape) opens; the camera still orbits behind it;
        Ctrl+K "inspect" would have armed the same tool.
 5. [ ] Quit and relaunch: Load Game reproduces the city exactly — and one
        more Ctrl+Z still works (the log survived the roundtrip).
 6. [ ] Not once did a modal dialog appear, a frame visibly hitch, or a
        mouse-only affordance block a step.
```

### M6 — "Animation + terrain converge" (lowest rank, after the loop above is daily-driver)
`anim_editor_ui` becomes a shell dock tab on `NodeCanvas` (GraphModel over `AnimGraphDefinition`); terrain brushes wire as the refinement layer behind the Forge graph flow. Checklist authored when M4 ships and the graph substrate is proven.

---

## 9. Open Questions

- [ ] **Where does `EditorShell` live so a game project can host it?** Today it's `vox_app` (engine repo, game layer). Options: civitas depends on `vox_app` as a lib, or the shell extracts to a `vox_shell` crate (still non-engine; `vox_core/data/render` stay pure). Decide before the M4 plan; lean: extract, since two hosts (engine demo bin + civitas) is the proof the plugin contract wants anyway.
- [ ] **Does the civitas raster HUD ever migrate to egui, or is Tokens+commands its end state?** Decide at M2 planning. Current lean: it stays owned raster (it composites at game rate, is headless-testable, and the menu renderer folds into it) — egui enters the game only if M5's editor-hosting makes it free.
- [ ] **Undo checkpoint mechanics:** clone-`CivitasGame` checkpoints vs serialized prefix saves; measure replay cost at 10k ticks first (budget: < 100 ms per undo). Pure optimization — semantics are fixed by §4.3 regardless.
- [ ] **Asset Lab placement:** standalone `asset_lab` bin first (named in W3) vs an `EditorPlugin` tab from day one. Standalone is faster to ship and shares the gate fns; fold into the shell when the M5 hosting question resolves.

---

## 10. Out of Scope — deliberately NOT built (and why)

- **A file-browser asset manager** (folder trees, drag-from-Explorer, thumbnails-of-everything). Our asset truth is the cook pipeline — `AssetCatalog` + cooked `ReadyAssetPayload` + packs — not loose files. File-manager UIs invite editing un-cooked sources and breed Unreal's redirector/fixup swamp, the single largest UX-debt machine in that editor. Discovery is the palette, typed catalog search, and the Asset Lab grid. The shell's existing minimal Content tab is frozen, not grown.
- **Detached-OS-window docking / multi-window layout wars.** egui_dock can float windows; we don't. Blender settled this: one window, splittable, with maximize-area — every hour on dock-tree edge cases (focus, DPI, restore-across-monitors) is stolen from the core loop. Layout persists to one `editor_layout.ron`, full stop.
- **A bespoke retained-mode UI framework** (a GPUI clone). The immediate-mode face + Vello overlay + token spine already meet every invariant; a framework rewrite is years of iteration we explicitly refuse to repeat (§4.1).
- **Modal wizards and a settings labyrinth.** Settings are a flat, searchable, scrubbable list reached through the palette ("settings: " prefix). Project setup is a command with defaults, not a five-page wizard.
- **An in-editor code IDE.** AI-generated scripts open in the user's editor; Ochroma writes files with receipts + content-checked undo (the existing `GeneratedScript` undo entry) and never hosts a text-editing surface beyond rename/search fields.
- **Editor theming marketplace, per-user cloud layout sync, gamepad editor navigation.** Token JSON swap covers theming (proven by test); the rest is surface area without a user.
- **This design does not change** the living-instances design's Done When, the engine/game purity boundary, or the save schema (`SAVE_VERSION` 5; undo is log truncation + replay, not a new format).

---

## 11. Related Plans / Designs

- Extends: [Editor SOTA Shell + Host-Plugin design](./2026-06-06-editor-sota-shell-design.md) — its tokens/dock/palette/plugin contract are this document's substrate; its UX Principles 1–2 (plain language, AI-native one-command-surface) are subsumed into invariants I3/I8/I10/I11.
- Designed with: [Living Building Instances](./2026-06-10-living-building-instances-design.md) — its replay-determinism guarantee (§4.7 there) is the mechanism behind I4 game undo; its inspector is roadmap rank 1's seed surface.
- Consumes: [Asset-Audit Remediation plan](../plans/2026-06-10-asset-audit-remediation.md) (the six-view gates Asset Lab surfaces; `forge_pathtrace.rs` stays read-only per its ownership rule).
- Directives honored: beat-Unreal (UI/flow/graphs/rendering priorities; SOTA-not-parity; the domain-knowledgeable non-game-dev persona), GPU-is-alfa-omega (live GPU viewport, §4.5), native-but-optional (worker-sink pattern for plugin backends).
- Required next: per-milestone implementation plans (M1 first) in `docs/superpowers/plans/`, each turning one walkthrough into TDD tasks per the plan template.
