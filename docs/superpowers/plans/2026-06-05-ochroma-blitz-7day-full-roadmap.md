# Ochroma Blitz — 7-Day Full-Roadmap Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take Ochroma from "doesn't build" to a cross-platform, building, end-to-end-playable spectral game engine with a Vello+Taffy UI, advancing all 12 roadmap domains to live-and-wired (Days 1–5 domains to "done", Days 6 frontier domains to "present + demoable").
**Done When:** On a clean checkout with `libasound2-dev` installed, `cargo build --workspace` and `cargo test --workspace` both exit 0, and `cargo run --bin walking_sim` opens a window where you can walk with WASD, watch a dropped object fall (Rapier) and emit a synthesized impact sound (no WAV file), see a Vello-rendered HUD showing collected-orb count, and win by collecting 10 orbs — with the GI cache feeding live (non-constant) spectral radiance into render, audio, and UI.
**Architecture:** The engine is ~97K–165K LOC across 18 crates, already 80–95% complete per `docs/superpowers/specs/2026-03-30-ochroma-comprehensive-roadmap-design.md`. This is a **wiring + completion** plan, not a greenfield build. Execution follows the roadmap's dependency order: **Spectral GI wiring first** (nine domains consume live radiance), then the per-domain completion order, with two long-running tracks (Vello+Taffy UI, Build/Platform) spanning the full week. The 8-band `[u16; 8]` spectral representation is the cross-system invariant and must not be stubbed downstream of Day 1.
**Design Document:** `docs/superpowers/specs/2026-03-30-ochroma-comprehensive-roadmap-design.md` (existing comprehensive roadmap)
**Tech Stack:** Rust edition 2024, wgpu, glam 0.29, bevy_ecs 0.16, rapier3d, rhai, cpal + fundsp, egui 0.31 → vello 0.4 + parley 0.3 + taffy 0.7, zstd, tokio.
**Build:** `cargo build --workspace` / `cargo test --workspace`. System prerequisite: `sudo apt-get install -y libasound2-dev pkg-config` (ALSA dev headers for `alsa-sys`/`cpal`). Verified present: ALSA 1.2.11.

---

## IMPORTANT NOTES

<!-- Real API signatures and constraints. Verify each against the live source before a task starts;
     the engine is large and these were captured 2026-06-05 — confirm with `grep`/`cargo doc` if a task fails to compile. -->

- **The build was broken by a stale path dep** — `crates/vox_render/Cargo.toml` pointed `spectra-renderer`/`spectra-gpu` at a deleted git worktree (`aetherspectra/.worktrees/ochroma-spectra-integration/…`). **Spectra is now its own repo at `~/src/spectra`**; deps repointed to `../../../spectra/rust/spectra-{renderer,gpu}`. **Do not reintroduce the aetherspectra/worktree paths.** (Crucible, used by `vox_nodes`' default `crucible` feature, remains at `aetherspectra/crucible/rust/crates/` and still resolves — leave it unless told otherwise.)
- **`spectra-native` is a non-default feature and does NOT currently compile** — `cargo check -p vox_render --features spectra-native` fails building `shader-slang-sys` (needs the native Slang shader-compiler toolchain, a system prerequisite like ALSA was). The default build uses the `crucible` feature and is fully green. **Resolving the Slang toolchain is a prerequisite for the real Spectra-native GI path (Task 1.3 / Domain 12a) — until then, GI wiring uses the crucible/CPU path.**
- **GameScript already exists** — `crates/vox_core/src/script_interface.rs` and `crates/vox_core/src/engine_runtime.rs`. Audit and extend; do **not** re-author a new trait.
- **The dogfood game already exists** — `crates/vox_app/src/bin/walking_sim.rs` (988 LOC). It already wires `CharacterController`, `RapierPhysicsWorld`, `RhaiRuntime`, `SpatialAudioManager`, `ShadowMapper`, `SpectralFramebuffer`, `GameUI`. Extend it; do not rewrite from scratch.
- **The game UI today is egui** (`vox_core::game_ui::{GameState, GameUI, UIElement, UIPosition, UISize}`, `crates/vox_ui` on egui 0.31). Vello/Taffy are present as **optional cargo features** in `vox_ui` (`vello`, `parley`, `taffy`) — the migration turns these on and ports screens, it does not start from zero.
- **Spectral GI primitives exist but are not wired** — `SpectralRadianceCache`, `GpuGiPass`, `SpectralAtmosphere` in `vox_render`. Confirm exact module paths with `grep -rn "SpectralRadianceCache" crates/vox_render/src` before wiring.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task. (Baseline is currently clean: `grep -rn "todo!\|unimplemented!"` returns 0.)
- **Crate boundary rule (from CLAUDE.md):** engine crates (`vox_core`, `vox_data`, `vox_render`, and the other `vox_*` engine crates) must NEVER contain game-specific concepts. Game logic lives in `vox_app`.
- Baseline truth as of 2026-06-05: 13 non-audio crates build; **1762 tests pass, 0 fail**. Any drop below 1762 passing without an intentional, documented reason is a regression.

---

## File Map

<!-- Domain-level map. Each Day's tasks touch the listed crates; per-task file lists are refined by the
     subagent from the live tree (the engine is too large to enumerate every file up front). -->

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `crates/vox_render/Cargo.toml` | Stale spectra worktree path → canonical path (DONE) |
| Modify | `crates/vox_render/src/` (GI modules) | Wire `SpectralRadianceCache`/`GpuGiPass`/`SpectralAtmosphere` into the live frame |
| Modify | `crates/vox_app/src/bin/walking_sim.rs` | Consume live GI; Vello HUD; physics-impact audio; orb win loop |
| Modify | `crates/vox_audio/` | rodio→CPAL backend; spectral impact synth; spectral reverb from GI |
| Modify | `crates/vox_ui/` (+ Cargo features) | Enable `vello`/`parley`/`taffy`; port HUD + editor screens |
| Modify | `crates/vox_tools/src/` | GLTF→Splat converter (currently 439 LOC, the thinnest crate) |
| Modify | `crates/vox_net/` | CRDT/rollback hardening; 2-player walking_sim |
| Modify | `crates/vox_editor/`, `crates/vox_app/src/editor.rs` | Vello gizmos + property inspector |
| Modify | `crates/vox_ai/`, `crates/vox_nn/` | AI/LLM domain wiring (Day 6, depth-limited) |
| Create | `.github/workflows/ci.yml` | test + clippy + doc + cargo-dist dry-run matrix |
| Modify | `crates/vox_web/` | WebGPU `hello_splat` (currently 15 LOC framework-only) |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Full workspace builds incl. audio | `cargo build --workspace` exits 0; `walking_sim` binary exists | `cargo build -p vox_core` only |
| Live spectral GI | radiance sample in a lit scene differs across bands & across two positions by > 1e-3 | `assert!(cache.sample(p).is_some())` |
| CPAL cross-platform audio | `cargo test -p vox_audio cpal_device_opens` enumerates ≥1 output device | `assert!(backend.is_ok())` |
| Spectral impact synth | synthesized glass-strike signal has spectral centroid > stone-strike centroid | returns a non-empty `Vec<f32>` |
| GLTF→Splat | `ochroma-tools gltf2splat cube.glb` → `.vxm` with `splat_count == expected` (>0) | output file exists |
| Vello HUD | `walking_sim` HUD orb counter increments 0→10 on collect, drawn via vello | `GameUI::new().is_ok()` |
| 2-player net | client B's transform for A changes after A moves; delta < 1 frame | both clients connect |
| Editor gizmo | dragging X-gizmo changes selected entity `transform.translation.x` by drag delta | gizmo widget renders |

---

## Day 1 — Foundation: green build, CI, live Spectral GI (sequential gate)

> **This is the real foundation, not the GameScript trait (which already exists).** Nothing downstream is trustworthy until the full workspace builds with audio and GI feeds live data. Days 2–7 each assume this gate passed.

### Task 1.1: Full workspace builds and tests green (incl. audio/app/binary)

**Files:**
- Modify: `crates/vox_render/Cargo.toml` (DONE — stale path fixed)
- Modify: any crate surfaced by the full build as broken

**Acceptance:** `cargo build --workspace 2>&1 | tail -1` → `Finished` (exit 0), AND `cargo test --workspace 2>&1 | grep "test result" | awk '{p+=$4;f+=$6} END{print p,f}'` → `≥1762 0` (≥1762 passed, 0 failed).

**Wiring requirement:** No code stubs introduced to make it pass. `todo!()`/`unimplemented!()` = task failure.

- [ ] **Step 1: Confirm ALSA prerequisite** — `pkg-config --modversion alsa` prints a version (currently 1.2.11)
- [ ] **Step 2: Full build** — `cargo build --workspace 2>&1 | tail -20`; record any non-audio breakage
- [ ] **Step 3: Fix** real compile errors (no stubs); keep engine/game crate boundary
- [ ] **Step 4: Full test** — `cargo test --workspace`; fix real failures
- [ ] **Step 5: Launch** — `cargo run --bin walking_sim` opens a window (screenshot it)
- [ ] **Step 6: Commit** — `git commit -m "fix(build): repoint spectra dep, restore full-workspace green build"`

### Task 1.2: Fix CI matrix — it EXISTS but has never been green

> **BLOCKER (2026-06-05): the engine is not self-contained.** The DEFAULT build depends on `crucible-core`/`crucible-types` via out-of-repo local path deps (`crates/vox_nodes/Cargo.toml` → `../../../aetherspectra/crucible/rust/crates/`), pulled by `vox_render`'s `default = ["crucible"]`. No git submodule, nothing vendored. A fresh `git clone` (CI runner or any other dev) CANNOT load the manifest, let alone build — this is why the old CI used `--no-default-features` (it dodges crucible AND ALSA). CI cannot test the real config until crucible is reachable on a clean checkout. **Options:** (a) vendor crucible-core/types into the repo like Track S did for shader-slang; (b) git-submodule `aetherspectra`; (c) publish crucible to crates.io; (d) make `crucible` non-default if the game doesn't need it. **Decision gated on the audit** (is crucible load-bearing for walking_sim?). The same applies to `spectra-*` (optional) for the spectra-native path. This is a Domain-1 (Build/Platform) foundational issue, not a yaml fix.
>
> **Finding (2026-06-05):** `.github/workflows/ci.yml` already exists and is broken in 3 ways (below). Remote is `git@github.com:supergrahn/ochroma.git`. This is a FIX task, not a create task.
> 1. **No ALSA install step** on the ubuntu runner. It currently masks this by running `cargo test --workspace --no-default-features`, which disables `vox_audio`'s `default = ["audio-backend"]` (cpal/fundsp/rodio/lewton) AND `vox_render`'s default `crucible` feature → **CI tests a degenerate config no user runs and never exercises audio or GI**. The local green baseline (2059 tests) was with DEFAULT features.
> 2. **`RUSTFLAGS: "-D warnings"` + `clippy -D warnings`** fail on the existing `unused_mut` warning at `crates/vox_physics/src/wetness.rs:33` (`let mut lcg`). Must fix the warning (drop `mut`) for CI to pass.
> 3. **`build-web` job** runs `web/build.sh` and asserts `web/dist/hello_splat_bg.wasm` exists; `vox_web` is 15 LOC (framework-only), so this fails until Day 3's web task. Mark `continue-on-error` until then.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `crates/vox_physics/src/wetness.rs` (drop spurious `mut` so `-D warnings` passes)

**Acceptance:** A pushed branch shows a green CI run that runs `cargo test --workspace` + `cargo clippy --workspace -- -D warnings` **with default features** (ALSA installed on ubuntu via `apt-get install -y libasound2-dev pkg-config`), ≥2059 tests pass. Windows/macos in matrix (CoreAudio/WASAPI need no apt step); `build-web` is `continue-on-error` until Day 3.

**Wiring requirement:** Runs on push + PR to `master`. Must test the REAL default-feature config, not `--no-default-features`. Empty/always-pass job = task failure.

- [ ] Step 1: Add `libasound2-dev pkg-config` apt step (ubuntu only); switch test/clippy to default features
- [ ] Step 2: Fix the `vox_physics` warning so `-D warnings` passes
- [ ] Step 3: Mark `build-web` `continue-on-error` (flip in Day-3 web task)
- [ ] Step 4: Push branch, observe run, capture the green check URL
- [ ] Step 5: Commit

### Task 1.3: Wire Spectral GI (Domain 12a) — live radiance at runtime — via crucible/CPU path

> **Backend decision (2026-06-05):** GI wires through the **default `crucible`/CPU path**, which compiles and is green. The `spectra-native` GPU path is NOT on the blitz critical path because Spectra's Linux Vulkan+Slang support is itself not yet working (see **Track S** below). The GI interface MUST stay backend-agnostic so spectra-native drops in later with no caller changes.

**Files:**
- Modify: `crates/vox_render/src/` (GI modules — confirm exact paths via grep)
- Modify: `crates/vox_app/src/bin/engine_runner.rs` / `walking_sim.rs` (consume cache in the frame loop)
- Test: `crates/vox_render/tests/` GI integration test

**Acceptance:** `cargo test -p vox_render gi_cache_is_live -- --nocapture` → sampled radiance at two distinct lit positions differs per-band by > 1e-3 (prove it's not a constant); `walking_sim` visibly changes scene tint when the illuminant changes. Runs with **default features only** (no spectra-native).

**Wiring requirement:** `SpectralRadianceCache` must be updated each frame inside the engine's render/update tick (name the exact function once located) and read by at least the render pass. Constants/stubs downstream = task failure. Backend abstraction must not leak `spectra-*` types into the caller.

- [ ] Step 1: `grep -rn "SpectralRadianceCache\|GpuGiPass\|SpectralAtmosphere" crates/vox_render/src` — map real signatures
- [ ] Step 2: Failing test asserting non-constant, per-band, per-position radiance
- [ ] Step 3: Wire cache update into the frame tick (crucible/CPU backend); feed render pass
- [ ] Step 4: Run test → PASS with real differing values
- [ ] Step 5: Commit — `feat(render): wire SpectralRadianceCache into live frame via crucible path (Domain 12a)`

---

## Track S — Spectra-native (Vulkan + Slang) Linux bring-up — PARALLEL, NON-BLOCKING

> Runs in its own track; **must not gate any blitz day**. When it lands, `vox_render`'s `spectra-native` feature replaces the crucible/CPU GI backend with no caller changes (Task 1.3's abstraction guarantees this). Owner: a dedicated background agent.

**Known facts (2026-06-05):**
- `shader-slang-sys 0.1.0` build.rs needs env `SLANG_DIR` (→ `$SLANG_DIR/include/slang.h` + `$SLANG_DIR/lib/libslang.so`), or `SLANG_INCLUDE_DIR`+`SLANG_LIB_DIR`, or `VULKAN_SDK`. Links `dylib=slang`. Uses bindgen → libclang (libclang-18 present ✓).
- Rust crates pinned at `shader-slang` / `shader-slang-sys` **0.1.0** (early-2024) — the installed Slang SDK's `slang.h` must still satisfy the 0.1.0 bindgen allowlists (`spReflection.*`, `slang_.*`, `SLANG_.*`). Version match is the main risk.
- `~/src/spectra/slang/` holds `.slang` SOURCE shaders, NOT the SDK. SDK must be obtained separately (prebuilt release from shader-slang/slang).
- User reports Spectra's Linux Vulkan+Slang path "not working" — scope unknown; likely a compile gap in `spectra-gpu`, a Vulkan loader/ICD gap, or a Slang version mismatch.

**Completion criterion:** `cargo check -p vox_render --features spectra-native` exits 0, AND a spectra-native GI smoke test produces non-constant radiance matching the crucible path within tolerance, on Linux.

**Steps (agent-owned, time-boxed probe first):**
- [x] Probe: classified — was a stack of layers, not one issue (Slang version + ABI + loader + API drift).
- [x] Install matching Slang SDK (v2024.14.5) to `~/.local/slang`; `SLANG_DIR` wired in `.cargo/config.toml`.
- [x] `shader-slang-sys` compiles — required vendoring `shader-slang 0.1.0` with `spReflectionVariable_GetDefaultValueInt` stubbed (`[patch.crates-io]`), and `BINDGEN_EXTRA_CLANG_ARGS` for missing clang resource headers.
- [x] Entire spectra dep graph (`spectra-renderer`, `spectra-usd`, `spectra-scene-upload`, `spectra-checkpoint`, `crucible-core`) compiles, given `LD_LIBRARY_PATH=~/.local/slang/lib` exported in the SHELL (cargo `[env]` does not reach build-script loader resolution).
- [ ] **NEXT: fix vox_render's own spectra-native glue — 8 API-drift errors** (E0432 unresolved imports, E0599 e.g. `read_splat_output_into` not on `Renderer<CudarcSlangBackend>`). vox_render's integration code targets an older spectra-renderer API; update to the current `~/src/spectra` API. Repro: `LD_LIBRARY_PATH=/home/tom-espen/.local/slang/lib cargo check -p vox_render --features spectra-native`.
- [ ] Resolve Vulkan runtime (loader/ICD/validation) so a spectra-native smoke test actually RUNS on a GPU — may need `mesa-vulkan-drivers`/`vulkan-tools` (sudo).
- [ ] Persist `LD_LIBRARY_PATH` (wrapper script or `sudo ldconfig`); report; flip Task 1.3 backend when green.

---

## Day 2 — Audio + Scripting (build on live GI)

### Task 2.1: CPAL cross-platform backend + spectral impact synthesis
**Acceptance:** `cargo test -p vox_audio cpal_device_opens` enumerates ≥1 device; `cargo test -p vox_audio spectral_strike_centroid` → glass-profile strike spectral centroid > stone-profile centroid (real ordered values printed). In `walking_sim`, dropping an object plays a synthesized impact with **no WAV file loaded**.
**Wiring requirement:** Called from the walking_sim collision callback; rodio path becomes optional, CPAL default.

### Task 2.2: Spectral reverb from GI cache + Rhai per-frame hooks
**Acceptance:** `cargo test -p vox_audio reverb_from_geometry` → a high-uniform-reflectance room yields a longer reverb tail (samples) than a mid-absorption room (printed tail lengths, real ordering). A Rhai script attached in `walking_sim` mutates entity state each frame (orb pulse visible).

---

## Day 3 — Assets + Rendering

### Task 3.1: GLTF→Splat converter in vox_tools (the real greenfield gap — 439 LOC today)
**Acceptance:** `cargo run --bin ochroma-tools -- gltf2splat assets/cube.glb /tmp/cube.vxm` writes a `.vxm`; `cargo test -p vox_tools gltf2splat_cube` asserts `splat_count > 0` and round-trips load in `vox_data`. The converted asset loads and renders in `walking_sim`.
**Wiring requirement:** Output consumable by `vox_data`'s loader; no placeholder splats.

### Task 3.2: Rendering domain polish on live GI (shadows/tonemap consistency)
**Acceptance:** named visible improvement with a before/after screenshot + a numeric test (e.g. tonemapped luminance within target range).

---

## Day 4 — Networking + Character/Animation

### Task 4.1: vox_net CRDT/rollback hardening + 2-player walking_sim
**Acceptance:** Two `walking_sim` clients connect; `cargo test -p vox_net rollback_converges` shows divergent-then-reconciled state; manually, client B sees A's avatar move within 1 frame.

### Task 4.2: Character animation blend trees
**Acceptance:** `cargo test -p vox_render blend_tree_interpolates` → blended pose between walk/idle at t=0.5 differs from both endpoints (real joint values); avatar in walking_sim transitions smoothly idle↔walk.

---

## Day 5 — Editor + Physics (Vello track converges here)

### Task 5.1: Vello-based transform gizmo + property inspector
**Acceptance:** In the editor, selecting an entity shows its real component values; dragging the X gizmo changes `transform.translation.x` by the drag delta (`cargo test -p vox_editor gizmo_drag_updates_translation` asserts the delta).
**Wiring requirement:** Rendered via the Vello path (Track A), not egui fallback.

### Task 5.2: Spectral-aware physics pass
**Acceptance:** named physics behavior driven by spectral material (e.g. restitution/wetness) with a real numeric test.

---

## Day 6 — AI/LLM + Spectral Frontier (12b–12e) — DEPTH-LIMITED

> **Honest scope marker:** these are the least-complete domains. Target is **wired + demoable**, not production-hardened. Every stub-or-shortcut must be logged in the task's commit body so it does not read as "finished."

### Task 6.1: AI/LLM domain wiring (vox_ai / vox_nn)
**Acceptance:** an NPC in walking_sim makes a spectral-perception-driven decision with a real test asserting the decision changes when the spectral input changes. **Commit body lists what is stubbed.**

### Task 6.2: Spectral frontier demo (one frontier capability live)
**Acceptance:** one frontier feature (capture pipeline OR infinite-detail OR spectral-audio frontier) runs end-to-end on a toy input with a real numeric assertion; remaining frontier domains explicitly logged as not-started.

---

## Day 7 — Integration dogfood + ship

### Task 7.1: Full-system dogfood playthrough
**Acceptance:** A single `walking_sim` run exercises render+GI, physics, synth audio, Rhai, Vello UI, and (optional) net simultaneously: walk, drop object (falls + sounds), collect 10 orbs (HUD counts), win screen — no panic. Recorded screen capture attached.

### Task 7.2: Cross-platform artifacts + clippy gate
**Acceptance:** `cargo clippy --workspace --deny warnings` exits 0; `cargo dist build` produces installable artifacts for at least linux (windows/macos attempted, results logged); release tag pushed.

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks exist
- [x] Every `Acceptance` criterion names a real non-trivial expected output (not "tests pass", not zeroes)
- [x] Every `Wiring requirement` names an exact function/file (or instructs the agent to locate it via grep before editing, given engine size)
- [x] `IMPORTANT NOTES` contains real API signatures / module locations and the build-fix gotcha
- [x] `File Map` lists every crate that appears in any task
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies
- [x] `Done When` names a specific command and specific human-observable result (walk + drop + synth sound + Vello HUD + win)
- [x] Day 6 frontier scope is explicitly marked depth-limited rather than implied "finished"
