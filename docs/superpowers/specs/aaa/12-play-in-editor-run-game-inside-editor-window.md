> **Adversarial verification:** ISSUES FOUND (one fatal data-flow bug, surfaced not silently fixed). The skeptic agreed with the spec's three grounding corrections to the roadmap seed — they are all real and accurate: (1) `EngineLoop::tick` does NOT exist (the loop exposes à-la-carte sub-steps `step_scripts`/`step_physics`/`step_audio`/`step_gi`/`step_shadows` at the named lines); (2) `EditorPlayMode` lives on the OLD immediate-mode `EditorState` (`editor.rs:61,89`), not the windowed `ochroma_editor`/`EditorShell`; (3) the windowed editor has no `bevy_ecs` world, only `entities: Vec<ShellEntity>` + `overlay: Vec<GaussianSplat>` — so the snapshot is a cheap `Vec` clone, not a World deep-clone. BUT the spec's enter-Play spawn recipe has a fatal omission that makes the headline "moved splats" proof unpassable. See Verification corrections.

## Verification corrections

- **FATAL DATA-FLOW BUG (headline proof cannot pass as written):** the spec's enter-Play flow spawns each authored entity as `world.spawn((NameComponent, TransformComponent{pos}, SplatAssetComponent{splats}, Visible, ScriptComponent{...}))` — but the verifier flagged that this component tuple is **missing the asset-handle/registration component the `gather_splats_system` requires to emit splats into `RenderBuffer`**. As written, the spawn produces **zero rendered splats**, so the Step 1 Done-When (`frame120_x` reads a "first lit splat's position") has no lit splat to read and the test cannot pass. **Correction (must be folded into Step 1):** the enter-Play spawn must include whatever asset-registration/handle component `gather_splats_system` (`engine_runtime.rs:508`) actually queries — confirm the exact component set by reading the existing `engine_runner.rs` spawn site that already produces gathered splats, and replicate it verbatim, rather than hand-assembling the tuple. Until the spawn matches a known-gathering spawn, the moved-splat proof is unreachable. This does not invalidate the snapshot/restore design (which the skeptic confirmed sound) — it pins down the one spawn-recipe detail the author under-specified.
- The remaining design — fresh `EngineLoop` per session, drop-on-Stop auto-restore, `Vec`-clone snapshot, `OrbitMover` script routing `SetPosition` through `process_script_commands_system`, and the `--frames/--shot` headless proof harness — is sound and well-grounded.

---

## Grounding correction (read first)

The roadmap's seed for this gap (#9) says: *"Wire `EditorPlayMode::Playing` to call `EngineLoop::tick` once per egui frame over the editor's `bevy_ecs` world, with a world-snapshot clone on enter-Play."* **Three parts of that seed are wrong against the real code and this spec corrects them — verified, not assumed:**

1. **`EngineLoop::tick` does not exist.** `ochroma_engine::engine_loop::EngineLoop` exposes *à-la-carte* sub-steps — `step_scripts(dt) -> Vec<GaussianSplat>`, `step_physics(dt)`, `step_audio(dt, pos, fwd)`, `step_gi(&[splat], hour) -> Vec<GaussianSplat>`, `step_shadows(...)` (verified `engine_loop.rs:372,385,414,457,510`). The per-frame composition is the caller's job — `engine_runner.rs:2847,2902,2916,1376` calls them in order. There is no single `tick`.
2. **`EditorPlayMode` lives on the WRONG editor.** The enum `EditorPlayMode::{Editing,Playing,Paused}` is a field of `EditorState` in `crates/vox_app/src/editor.rs:61,89` — the *old immediate-mode* editor used by `main.rs`/`engine_runner.rs`. Its Play button (`editor.rs:640`) only flips a `play_requested: bool` that a test asserts (`editor.rs:1158`); **it ticks no engine.** The *windowed* editor — the one with the `--frames N --shot` proof harness the Done-When needs — is a different binary, `ochroma_editor` (`crates/vox_app/src/bin/ochroma_editor.rs`), which drives `EditorShell` (`shell/mod.rs:182`), an editor that has **no play mode at all**.
3. **The editor has no `bevy_ecs` world to tick.** `EditorShell`'s "world" is `entities: Vec<ShellEntity{name,kind,pos:[f32;3]}>` (`shell/mod.rs:185`, struct at `:88`) plus `overlay: Vec<GaussianSplat>` (`:212`). The viewport renders `viewport::build_scene() + overlay` through a `SoftwareRasteriser` with a **hardcoded camera** (`viewport.rs:96–135`); the `ShellEntity` rows are decorative inspector content, *not* linked to any rendered splat. There is no `World`, no `EngineLoop`, nowhere a snapshot-clone of a bevy world would apply.

So this spec wires Play into **`ochroma_editor` + `EditorShell`** (the binary that can be proven headlessly), constructs a **fresh** `EngineLoop` on enter-Play, drives the simulation by **composing the existing sub-steps** (not a fictional `tick`), and renders the engine's `RenderBuffer.splats` into the docked viewport. The snapshot/restore is of the *shell's* authored state (`entities` + `overlay`), which is cheap and already `Clone`. This is **more** tractable than the seed implied, not less (see Surprises).

---

## 1. What we need

After this lands, a developer using `ochroma_editor` can:

- **Press Play and watch authored content come alive in the same window.** A spectral-splat entity that has a movement script (or velocity/physics) is **at one position on frame 1 and a visibly different position on frame 120**, rendered into the docked Viewport tab — no second process, no `engine_runner` launch. Observable: `ochroma_editor --frames 120 --play --shot` produces a PNG whose lit-splat centroid has moved ≥ a threshold versus frame 1.
- **Press Stop and get the authored scene back, bit-for-bit.** Stop discards the play-session world and restores the exact `entities`/`overlay` captured on enter-Play. Observable: a headless test asserts the moved entity's authored position is byte-identical before-Play and after-Stop.
- **Pause and Resume.** Paused freezes the simulation (no sub-step calls) but keeps presenting the last play frame; Resume continues from the frozen state, not from the snapshot. Observable: entity position is constant across paused frames, then advances again after Resume.
- **Trust that Play cannot corrupt authoring.** Play runs on a *separate* `EngineLoop` world; the shell's undo stack, planted-asset ranges, and asset-count provenance are untouched by a play session (Play never calls `plant_asset`/`push_undo`). Observable: undo-stack length is identical before-Play and after-Stop.
- **See an honest mode indicator and receipts.** The status bar shows `EDITING | PLAYING | PAUSED`; entering Play and pressing Stop each append an Output-Log receipt (frame count, entity count). Observable: Output Log contains `[play] entered — N entities snapshotted` and `[play] stopped — restored N entities`.

**Why it is blocking (Engine Architecture & API Surface dimension, roadmap §2):** *"Editor and runtime are separate binaries — the editor never constructs or ticks an EngineLoop (no Play button)."* AAA iteration *is* press-play-in-editor; today gameplay can only be tested by leaving the tool and launching the 3606-line `engine_runner` binary, capping iteration at minutes-per-loop. This is the Engine-API floor under "the editor is a place you author *and test* real gameplay" (roadmap §5, Phase 3).

The AAA bar (Unreal/Unity PIE): enter-Play snapshots the authored world, runs the real game loop in the editor viewport at frame rate, Stop restores authoring state exactly, Pause/step is available. We hit the **snapshot-restore-correctness** and **same-window simulation** core of that bar; we explicitly defer multi-viewport PIE, input-routing-to-game, and hot-reload-during-play (see §4 non-goals).

---

## 2. How it's gonna be (the design)

### Where it lives
Two crates, mirroring the engine/game split (CLAUDE.md rule — engine crates stay game-agnostic):

- **`crates/vox_app/src/shell/play.rs` (NEW, game layer).** Owns `PlayState`, the `EngineLoop`, the snapshot, and the per-frame sub-step composition. This is game-layer because it decides *which* systems run and *how* `ShellEntity`/`overlay` map onto ECS spawns — game policy, not engine mechanism.
- **`crates/vox_app/src/bin/ochroma_editor.rs` (EDIT).** Adds a `--play` CLI flag and calls `shell.play_tick()` once per redraw before `run_shell_frame`.
- **`crates/ochroma_engine` / `vox_core` — UNTOUCHED.** Every engine API used already exists (verified below). No engine change; provability of the engine crates is preserved by construction.

### The new types (full proposed signatures)

```rust
// crates/vox_app/src/shell/play.rs  (NEW)
use ochroma_engine::engine_loop::{EngineLoop, SystemMask};
use vox_core::engine_runtime::{EngineConfig, RenderBuffer};
use vox_core::types::GaussianSplat;
use crate::shell::ShellEntity;

/// Editor run mode. Mirrors the names already on the old `EditorState`
/// (editor.rs:89) so future unification is a rename, not a redesign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayMode { Editing, Playing, Paused }

/// The authored shell state captured on enter-Play and restored on Stop.
/// Both fields are already `Clone` (ShellEntity:#[derive(Clone)] shell/mod.rs:88;
/// GaussianSplat: #[derive(Clone, Copy)] types.rs:27) — the snapshot is a cheap
/// Vec clone, NOT a bevy-World deep clone (there is no editor bevy World).
#[derive(Clone)]
pub struct AuthoredSnapshot {
    pub entities: Vec<ShellEntity>,
    pub overlay:  Vec<GaussianSplat>,
}

/// Owns the live play session. Absent (None) while Editing.
pub struct PlaySession {
    pub loop_: EngineLoop,
    /// Frames stepped since enter-Play (for receipts + the proof assertion).
    pub frame: u64,
    /// The splats produced by the most recent stepped frame, ready to
    /// composite into the viewport in place of the authored overlay.
    pub render_splats: Vec<GaussianSplat>,
}

pub struct PlayController {
    pub mode: PlayMode,
    pub session: Option<PlaySession>,
    pub snapshot: Option<AuthoredSnapshot>,
}

impl PlayController {
    pub fn new() -> Self { /* Editing, no session, no snapshot */ }

    /// Enter Play: snapshot authored state, build a fresh EngineLoop, spawn
    /// the authored scene into its ECS world, start scripts. Returns a receipt.
    pub fn enter_play(&mut self, snap: AuthoredSnapshot) -> String;

    /// Advance ONE editor frame of simulation when Playing (no-op when
    /// Editing/Paused). Composes the engine sub-steps and refreshes
    /// `session.render_splats`. `dt` is the fixed editor frame step.
    pub fn tick(&mut self, dt: f32);

    pub fn pause(&mut self);   // Playing -> Paused
    pub fn resume(&mut self);  // Paused  -> Playing

    /// Stop: drop the session, return the snapshot for the shell to restore.
    pub fn stop(&mut self) -> Option<AuthoredSnapshot>;
}
```

`PlayController` becomes one field on `EditorShell`: `pub play: PlayController` (`shell/mod.rs:182` struct, constructed in `new()` at `:313`).

### The enter-Play → spawn → step → render data flow

```
 PLAY pressed
   │
   ▼
 EditorShell::enter_play()
   │  snapshot = AuthoredSnapshot{ entities.clone(), overlay.clone() }   (cheap Vec clone)
   ▼
 PlayController::enter_play(snapshot)
   │  loop_ = EngineLoop::new(EngineConfig{enable_audio:false,..}, SystemMask::game_minimal())
   │  register "OrbitMover" script (game-layer, deterministic)
   │  for each authored splat-group: world.spawn((
   │        NameComponent, TransformComponent{pos}, SplatAssetComponent{splats},
   │        <asset-registration/handle component gather_splats_system queries>,  ◄── see Verification corrections
   │        Visible, ScriptComponent{["OrbitMover"]}, Collider/Velocity as authored))
   │  loop_.runtime.start();  loop_.runtime.init_scripts()
   ▼
 ── each editor redraw, when Playing ──
 PlayController::tick(dt)
   │  loop_.step_scripts(dt)         // runs script_update + process_script_commands
   │                                 //   → SetPosition mutates TransformComponent
   │  loop_.step_physics(dt)         // rapier + ECS transform sync (game_minimal: on)
   │  // gather_splats_system already ran inside runtime.tick() (step_scripts),
   │  // writing transformed splats into RenderBuffer.splats. Drain them:
   │  render_splats = mem::take(world.resource_mut::<RenderBuffer>().splats)
   │  frame += 1
   ▼
 EditorShell::ui() / viewport
   │  if Playing|Paused: viewport composites session.render_splats
   │                     (NOT the authored overlay) over build_scene()
   ▼
 STOP pressed
   │  snap = play.stop();  self.entities = snap.entities; self.overlay = snap.overlay;
   │  self.viewport_tex = None;   // force re-rasterize authored scene
```

### Key design decisions + rationale

- **A fresh `EngineLoop` per play session, dropped on Stop — not a long-lived loop reset.** `EngineLoop::new` (`engine_loop.rs:200`) builds a Rapier ground plane, audio backend, GI cache, ShadowMapper, atmosphere, sun. Building it on enter-Play and dropping it on Stop means Stop restoration is *automatic* (the simulated world simply ceases to exist) and there is zero risk of a play mutation leaking into the next session. The authored snapshot is the only thing that must round-trip, and it is shell-side `Vec`s.
- **Snapshot the SHELL state, not a bevy World.** The seed assumed an editor bevy world to clone. There is none. The authored state that must survive Play is `entities` + `overlay`, both already `Clone`. This is *cheaper and simpler* than the seed — the restore is two `=` assignments.
- **`SystemMask::game_minimal()` for v1** (`engine_loop.rs:72`: physics+audio+animation+shadows, no GI/scripts) — **with one correction**: the moving entity needs scripts. v1 uses a mask with `scripts:true, physics:true, gi:false, audio:false` (audio off avoids opening a device in the editor process; GI off keeps the frame cheap). Constructed inline as `SystemMask{ scripts:true, physics:true, audio:false, animation:true, gi:false, shadows:false }` — all fields are `pub` (`engine_loop.rs:49`).
- **The "moving entity" is a deterministic game-layer script, registered at enter-Play.** `OrbitMover` is a `GameScript` impl (trait at `script_interface.rs:3`) whose `on_update(ctx, dt)` calls `ctx.set_position([...])` (`script_interface.rs:80`) advancing X by `dt`. This routes `ScriptCommand::SetPosition` (`:38`) through the *existing* `process_script_commands_system` (`engine_runtime.rs:283`) which mutates `TransformComponent` (`:289`). Then `gather_splats_system` (`engine_runtime.rs:508`) applies `transform_splat` (`:501` → `GaussianSplat::apply_transform`, `types.rs:198`) so the **rendered splats move because the transform moved** — exactly the engine's real path, not a viewport hack. **(See Verification corrections: the spawn must carry the asset-registration component `gather_splats_system` queries, or no splat is gathered.)**
- **Rendering: drain `RenderBuffer.splats`, composite in the viewport in place of `overlay`.** `render_scene_rgba_with(&[GaussianSplat])` (`viewport.rs:105`) already composites an arbitrary splat slice over `build_scene()`. When Playing/Paused the viewport passes `session.render_splats`; when Editing it passes `overlay` (today's behavior). One branch, reusing the proven CPU rasteriser path the headless snapshot already asserts on.
- **No engine-crate change, no GPU change.** v1 stays on the `SoftwareRasteriser` viewport path (the accepted v1 path per `ochroma_editor.rs:14–19`). When the GPU viewport (roadmap #3) lands, Play composites into the same texture with no Play-side change. This honors "GPU work mirrors a CPU oracle" by *not* introducing un-mirrored GPU work here.
- **Every numeric input clamps.** The `--play` flag composes with existing `--frames` clamping (`ochroma_editor.rs:67`). `dt` is a fixed `1.0/60.0` (never user wall-clock in proof mode) so frame-1-vs-120 displacement is deterministic. `OrbitMover` clamps its position to a bounded radius so a long run can't NaN/escape.

### Patterns honored
- **No-panic shell rule:** `enter_play` returns a `String` receipt; if `EngineLoop::new` or a spawn fails it logs to the Output Log and stays in `Editing` (mirrors the `OCHROMA_GI=gpu` graceful-fallback pattern at `engine_loop.rs:229`). No `unwrap` on the play path.
- **Provability culture:** the proof is headless pixel/state-asserted via the existing `--frames/--shot` harness + `non_background_fraction`/`write_png` (`cpu_render.rs:373,284`) and a `cargo test -p vox_app` state assertion on the moved centroid.
- **Receipts:** enter/stop append honest Output-Log lines (`push_output_log`, `shell/mod.rs:1148`).

---

## 3. How it's gonna be made (the implementation plan)

### Step 1 — `OrbitMover` script + `PlayController` core, headless state-proven (M) ← launchable tomorrow

**Files:**
- NEW `crates/vox_app/src/shell/play.rs` — `PlayMode`, `AuthoredSnapshot`, `PlaySession`, `PlayController` (signatures in §2), plus a `pub struct OrbitMover { t: f32, start_x: f32 }` impl of `vox_core::script_interface::GameScript` whose `on_update` does `self.t += dt; ctx.set_position([self.start_x + self.t * 2.0, 0.0, 0.0])`.
- EDIT `crates/vox_app/src/shell/mod.rs:25` — add `pub mod play;`.
- NEW `crates/vox_app/src/shell/play.rs` `#[cfg(test)]` module.

**What it wires:** `PlayController::enter_play` builds the `EngineLoop`, spawns ONE authored splat-group entity carrying the **full gathering spawn tuple** (replicated verbatim from `engine_runner.rs`'s known-gathering spawn site, per Verification corrections — `SplatAssetComponent{splats}` PLUS the asset-registration/handle component `gather_splats_system` queries) + `TransformComponent` + `Visible` + `ScriptComponent{["OrbitMover"]}`, registers `OrbitMover`, calls `runtime.start()` + `runtime.init_scripts()`. `tick(dt)` composes `step_scripts(dt)` then `step_physics(dt)` then drains `RenderBuffer.splats` into `session.render_splats`.

**Done When:** `cargo test -p vox_app play::tick_moves_a_scripted_entity_and_stop_restores -- --nocapture` prints
`frame1_x=5.000 frame120_x=8.967 restored_x=5.000` and **passes**, where the test asserts:
- `let p1 = first lit splat's position()[0]` after `enter_play` + 1 `tick(1.0/60.0)`; capture `p1`.
- after 119 more `tick(1.0/60.0)` calls, `let p120 = session.render_splats[0].position()[0]`.
- `assert!((p120 - p1).abs() > 1.0, "entity must move ≥1m over 120 frames: {p1} -> {p120}")` (computed real outcome, not `is_some`).
- `let snap = play.stop().unwrap();` then `assert_eq!(snap.overlay, authored_overlay_clone)` (byte-identical restore of the authored `Vec<GaussianSplat>`, using `GaussianSplat`'s `Pod` byte-equality via `bytemuck::cast_slice`).
- `assert_eq!(play.mode, PlayMode::Editing)` after stop.

This step needs no window and no GPU; it is a pure `cargo test` an agent can run tomorrow. **(Precondition: the spawn tuple must gather splats — see Verification corrections; assert `session.render_splats` is non-empty after the first tick before reading position[0].)**

### Step 2 — Wire `PlayController` into `EditorShell` + Stop restore + status/receipts (S)

**Files:** EDIT `shell/mod.rs` — add `pub play: PlayController` field (`:182` struct, `:313` ctor); add `pub fn enter_play(&mut self)` (clones `entities`+`overlay` into `AuthoredSnapshot`, calls `play.enter_play`, logs receipt), `pub fn stop_play(&mut self)` (calls `play.stop()`, restores both Vecs, sets `viewport_tex=None`), `pub fn play_tick(&mut self, dt: f32)` (delegates to `play.tick`). Add `ShellRequest::EnterPlay`/`StopPlay`/`PauseResume` variants (`:148`) and registry commands so the palette/menu can trigger them through the existing `drain_requests` path (`:1022`).

**Done When:** `cargo test -p vox_app play::enter_play_does_not_touch_undo_or_provenance` passes, asserting: capture `shell.undo_stack.len()` and `shell.entities.len()` → `enter_play()` → 60× `play_tick(1.0/60.0)` → `stop_play()` → assert `undo_stack.len()` unchanged AND `shell.entities == entities_before` (the authored 4 demo entities at `shell/mod.rs:340`, restored exactly).

### Step 3 — `--play` flag + viewport composites play splats, headless pixel proof (M)

**Files:** EDIT `crates/vox_app/src/bin/ochroma_editor.rs` — parse `--play` in `parse_cli` (`:52`, with the same fail-fast discipline as `--frames`); in `EditorHost::new`/`redraw`, on the FIRST redraw when `--play` is set call `self.shell.enter_play()`, and each redraw call `self.shell.play_tick(1.0/60.0)` before `run_shell_frame`. EDIT `shell/viewport.rs` viewport branch (called from `shell/mod.rs:581` `scene_texture`) so that when `play.mode != Editing` the composited overlay is `play.session.render_splats` instead of `overlay`; invalidate `viewport_tex` every play frame so the moved splats re-rasterize.

**Done When:** `cargo run -p vox_app --bin ochroma_editor -- --frames 120 --play --shot /tmp/play120.png` exits 0 and prints a line like `[ochroma_editor] wrote /tmp/play120.png (… bytes), NN.N% non-background pixels, 1600x900, format=…`; AND a headless test `play::shot_at_frame120_differs_from_frame1` renders the viewport-only RGBA via the existing `render_scene_rgba_with(&play.session.render_splats)` at frame 1 and frame 120 and asserts the **lit-splat centroid X moved > 20 px** between the two frames (computed centroid of non-background pixels, not a fraction-is-nonzero check). Stop is proven in Step 2; this proves the moved pixels are visible in the actual render path.

### Step 4 — Pause/Resume + mode indicator in the status bar (S)

**Files:** EDIT `shell/mod.rs` status bar render + `play.rs` `pause`/`resume`. **Done When:** `cargo test -p vox_app play::paused_frames_do_not_advance_then_resume_does` passes: `enter_play` → `tick` → record `x_a` → `pause()` → 10× `tick` → assert `session.render_splats[0].position()[0] == x_a` (frozen) → `resume()` → 10× `tick` → assert position X strictly greater than `x_a`.

---

## 4. How it fits (integration + dependencies)

### Depends on
- **#1 Project + open/save in the shell (Editor, L)** — *for the seam, not as a hard blocker.* The roadmap says #9 depends on #1 "for the snapshot/restore seam." In reality this spec's snapshot is the simpler shell-state `Vec` clone (`AuthoredSnapshot`), so **Step 1–4 can ship before #1**. When #1 lands, `enter_play` should snapshot via the same `WorldSave` serialization path (`vox_data::world_save::WorldSave`, `world_save.rs:6`) so a play session round-trips through the *identical* serializer the save button uses — a one-function swap of `AuthoredSnapshot` internals, no API change. Cross-gap seam: **share the serializer, not just the concept.**
- **Existing, verified, no new work:** `ochroma_engine::engine_loop::{EngineLoop,SystemMask}` (already a `vox_app` dep, `Cargo.toml:43`), `vox_core::engine_runtime::{EngineConfig,RenderBuffer}`, `gather_splats_system`/`process_script_commands_system`/`transform_splat`, `GaussianSplat::apply_transform`, `SoftwareRasteriser` viewport path, the `--frames/--shot` proof harness.

### What depends on it
- **#10 Behavior tree that ticks real game logic** and **#11 script/mod host API** — both become *demonstrable in-editor* the moment Play exists; an authored NPC's BT runs in the play viewport. Play is the showcase surface for both.
- **The wedge demo (#5 GPU relight as a mechanic):** "play a captured world under a light it never saw" needs a Play button to *be* a playable moment. Play-in-Editor is the frame where relight stops being a CLI receipt and becomes a mechanic the user toggles live.

### Composes with existing systems
- **`EngineLoop` sub-steps** (the same ones `engine_runner` composes) — Play reuses them verbatim; if the editor's composition and `engine_runner`'s diverge, that is a *signal*, which is the point of sharing one loop.
- **`ShellRequest`/`drain_requests`/registry** — Play/Pause/Stop dispatch through the existing one-command-surface (menus + Ctrl+K palette), exactly like Undo and theme-swap do.
- **Viewport `scene_texture`/`render_scene_rgba_with`** — Play feeds it a different splat slice; zero new rendering code.

### What it must NOT break
- **The 11-green-gate invariant.** No engine-crate edit; `cargo test` across the workspace stays green. Play code is additive in `vox_app`; the existing `ochroma_editor --frames N` (no `--play`) path is byte-unchanged (the `--play` branch is gated).
- **Both-config builds.** Nothing here touches feature flags (`spectra-native`, `forge-native`); `EngineLoop`/`bevy_ecs` are already unconditional `vox_app` deps. Default and feature builds both compile.
- **The no-panic shell rule.** `enter_play` is `Result`-shaped internally and falls back to `Editing` with an Output-Log line on any failure (audio/GPU/spawn) — never panics, mirroring `engine_loop.rs:229`.
- **Authoring integrity.** Play runs on a throwaway `EngineLoop` world; it never calls `plant_asset`/`push_undo`/`asset_counts`. Step 2's test pins this.

### 4-phase sequencing
**Phase 3** ("the editor becomes a place you author *and test* real gameplay"), alongside #10/#11. It is unusually startable *now* because (per Surprises) every dependency already exists — it does not actually need to wait for #1.

---

## Surprises & advantages

- **The whole gap is far cheaper than the roadmap scored it (L → realistically M).** Every API the design needs already exists and is tested: `EngineLoop` with composable sub-steps, `gather_splats_system` turning moved transforms into moved splats, `transform_splat`/`apply_transform`, and a viewport (`render_scene_rgba_with`) that *already* composites an arbitrary splat slice over the base scene. The "embed a game loop in the editor" work is mostly *calling functions that are already wired into `engine_runner`* from a second site.
- **The snapshot/restore is trivial, not a deep clone.** The seed feared a bevy-World snapshot. The editor has no bevy World — its authored state is two `Clone` `Vec`s (`entities`, `overlay`). Restore is two `=` assignments. The roadmap listed #9 as depending on #1 "for the snapshot seam"; **that dependency is soft** — Play can ship first and adopt `WorldSave` later as a refinement.
- **A ready-made headless proof harness.** `ochroma_editor` *already* has `--frames N --shot` with GPU readback + `non_background_fraction` + `write_png` (`ochroma_editor.rs`, `cpu_render.rs:373,284`). The Done-When ("press play, frame 120 shows a moved entity, pixel-asserted headless") needs **no new proof infrastructure** — just a `--play` flag and a centroid assertion.
- **An existing integration test is a near-exact template for Step 1.** `crates/ochroma_engine/tests/engine_loop_integration.rs:36` already drives 120 frames of `step_scripts`+`step_physics`+`step_gi` with a stateful script and asserts cross-system end state. Step 1's test is that test, narrowed to "a `SetPosition` script moves a `SplatAssetComponent` entity and the gathered splat moved." The hard part (does the composed loop actually advance state coherently over 120 frames?) is *already proven* — and it is also the existing site to copy the gathering spawn tuple from (the fix named in Verification corrections).
- **The name `EditorPlayMode::{Editing,Playing,Paused}` already exists** on the old `EditorState` (`editor.rs:89`). Reusing the identical names on `PlayController::PlayMode` means the eventual editor unification (old `EditorState` → `EditorShell`) is a mechanical merge, not a concept reconciliation. First-mover tidiness: we don't invent a competing vocabulary.
- **First-mover wedge angle:** because the editor and game share *one* `EngineLoop`, Play-in-Editor for a *spectral* world means the relight mechanic (#5) and spectral AI perception (already live) run in the editor viewport the instant they're wired — no RGB engine can press-play into a metameric scene, so this Play button is the stage on which the unforgeable wedge becomes a clickable moment.
