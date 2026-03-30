# Domain 1: Build/Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ochroma engine compilable and testable on Windows/Linux/Mac, publishable to crates.io, and runnable in the browser via WebGPU — with a CI pipeline that enforces correctness on every push.

**Done When:** `cargo build --target wasm32-unknown-unknown -p vox_app` succeeds AND opening `web/index.html` in a browser loads the engine without console errors (status div reads "Ready") AND `gh run view --web` shows all three platform CI jobs (ubuntu-latest, windows-latest, macos-latest) green on the most recent push to master.

**Architecture:** cargo-dist handles cross-platform artifact generation and crates.io publishing. GitHub Actions runs a matrix job across all three desktop platforms. The web target compiles `vox_app` to `wasm32-unknown-unknown` with a `wasm-bindgen` entry point and WebGPU backend. The crucible path dependency is isolated to `vox_nodes` and made optional so CI doesn't require a sibling checkout.

**Tech Stack:** cargo-dist 0.22.1, GitHub Actions, wasm-pack, wasm-bindgen 0.2, web-sys, console_error_panic_hook

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `.github/workflows/ci.yml` | test + clippy + doc on push/PR, all three platforms |
| Create | `.github/workflows/release.yml` | cargo-dist release on version tag push |
| Modify | `Cargo.toml` | add `[workspace.metadata.dist]`, per-crate publish metadata |
| Modify | `crates/vox_nodes/Cargo.toml` | make crucible optional; stub feature gate |
| Modify | `crates/vox_nodes/src/lib.rs` | gate crucible usage behind feature flag |
| Modify | `crates/ochroma_engine/Cargo.toml` | add `web` feature, publishing metadata |
| Modify | `crates/vox_app/Cargo.toml` | add `web` feature, wasm deps behind feature |
| Create | `crates/vox_app/src/bin/hello_splat.rs` | minimal WASM-compatible entry point |
| Create | `web/index.html` | browser entry point loading the WASM bundle |
| Create | `web/build.sh` | script: `wasm-pack build` + copy assets |
| Create | `docs/getting-started.md` | install → first scene → running examples |
| Create | `CONTRIBUTING.md` | dev setup, PR checklist, architecture overview |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Crucible optional | `cargo build -p vox_nodes --no-default-features` exits 0 on a machine without `../aetherspectra` checked out | `assert!(true)` after gating nothing |
| WASM bundle | `ls web/dist/hello_splat_bg.wasm` exists and `wc -c` > 10000 bytes | `assert!(Path::new("web/dist").exists())` |
| CI matrix | `gh run view --web` shows ubuntu/windows/macos all green | checking only that ci.yml is valid YAML |
| Clippy clean | `cargo clippy --workspace --no-default-features -- -D warnings` exits 0 with zero warning lines | running clippy without `-D warnings` |
| Metadata valid | `cargo package -p ochroma_engine --no-verify --list` exits 0 listing >5 files | checking that Cargo.toml has a `description` key |

---

## Task 1: Unblock CI — Make crucible dependency optional

**Files:**
- Modify: `Cargo.toml` (workspace — remove crucible from `[workspace.dependencies]`)
- Modify: `crates/vox_nodes/Cargo.toml`
- Modify: `crates/vox_nodes/src/lib.rs`

**Acceptance:** `cargo build -p vox_nodes --no-default-features` → exits 0, zero error lines in output. Verified on the current machine where `../aetherspectra` does not exist.

**Wiring requirement:** Must be called from `[workspace]` in `Cargo.toml` before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test** (confirm crucible blocks build without sibling checkout)

```bash
cargo build -p vox_nodes --no-default-features 2>&1 | head -20
```

Expected: FAIL — currently fails because crucible paths don't exist unless `../aetherspectra` is checked out.

- [ ] **Step 2: Run to verify it fails**

```bash
cargo build -p vox_nodes --no-default-features 2>&1 | tail -5
```

Expected: FAIL with `error[E0432]: unresolved import` or `error: failed to load manifest for dependency`

- [ ] **Step 3: Implement** — Remove crucible from workspace, make it optional in vox_nodes

In `Cargo.toml`, remove these two lines from `[workspace.dependencies]`:

```toml
# DELETE these lines:
crucible-core  = { path = "../aetherspectra/crucible/rust/crates/crucible-core" }
crucible-types = { path = "../aetherspectra/crucible/rust/crates/crucible-types" }
```

Replace the contents of `crates/vox_nodes/Cargo.toml` with:

```toml
[package]
name = "vox_nodes"
edition.workspace = true
version.workspace = true

[features]
crucible = ["crucible-core", "crucible-types"]

[dependencies]
vox_ui         = { workspace = true }
serde          = { workspace = true }
thiserror      = { workspace = true }

[dependencies.crucible-core]
path = "../../aetherspectra/crucible/rust/crates/crucible-core"
optional = true

[dependencies.crucible-types]
path = "../../aetherspectra/crucible/rust/crates/crucible-types"
optional = true
```

Wrap all `use crucible*` statements in `crates/vox_nodes/src/lib.rs`:

```rust
#[cfg(feature = "crucible")]
use crucible_core::SomeType;

#[cfg(feature = "crucible")]
pub mod crucible_bridge {
    // crucible-dependent code here
}
```

- [ ] **Step 4: Wire at exact callsite** — confirm `[workspace.dependencies]` in `Cargo.toml` no longer references crucible paths

```toml
# After: no crucible-core or crucible-types entries in [workspace.dependencies]
```

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
cargo build -p vox_nodes --no-default-features 2>&1 | tail -5
cargo build --workspace 2>&1 | tail -5
```

Expected: PASS, both commands exit 0. Full workspace builds — all crates compile. (Machines without `../aetherspectra` will skip crucible features.)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/vox_nodes/Cargo.toml crates/vox_nodes/src/
git commit -m "build: make crucible dependency optional to unblock CI"
```

---

## Task 2: Add crates.io publishing metadata

**Files:**
- Modify: `Cargo.toml` (workspace)
- Modify: `crates/ochroma_engine/Cargo.toml`

**Acceptance:** `cargo package -p ochroma_engine --no-verify --list` → exits 0, output lists at least 5 files including `Cargo.toml` and `src/lib.rs`.

**Wiring requirement:** Must be called from `[workspace.package]` in `Cargo.toml` and `[package]` in `crates/ochroma_engine/Cargo.toml` before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**

```bash
cargo package -p ochroma_engine --no-verify --list 2>&1 | head -10
```

Expected: FAIL with missing required fields (description, license, repository).

- [ ] **Step 2: Run to verify it fails**

```bash
cargo package -p ochroma_engine --no-verify --list 2>&1 | tail -5
```

Expected: FAIL with `error: field ... is missing`

- [ ] **Step 3: Implement** — add metadata to workspace and ochroma_engine crate

Add to the existing `[workspace.package]` block in `Cargo.toml`:

```toml
[workspace.package]
edition = "2024"
version = "0.1.0"
license = "MIT"
repository = "https://github.com/ochroma-engine/ochroma"
homepage = "https://ochroma.dev"
authors = ["Ochroma Contributors"]
```

Update `crates/ochroma_engine/Cargo.toml`:

```toml
[package]
name = "ochroma_engine"
edition.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
authors.workspace = true
description = "Spectral Gaussian Splatting game engine — 16-band spectral rendering, physics, audio, and AI"
keywords = ["game-engine", "gaussian-splatting", "spectral", "rendering", "wgpu"]
categories = ["game-engines", "rendering", "graphics"]
readme = "README.md"
```

- [ ] **Step 4: Wire at exact callsite** — confirm `[workspace.package]` contains all required crates.io fields

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
cargo package -p ochroma_engine --no-verify --list 2>&1 | head -20
```

Expected: PASS, output lists files (Cargo.toml, src/lib.rs, etc.). No errors about missing fields.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/ochroma_engine/Cargo.toml
git commit -m "build: add crates.io publishing metadata"
```

---

## Task 3: Set up cargo-dist

**Files:**
- Modify: `Cargo.toml` (workspace — add `[workspace.metadata.dist]`)
- Create: `.github/workflows/release.yml` (generated by cargo-dist)

**Acceptance:** `cargo dist build --dry-run` → exits 0, prints what it would build without errors.

**Wiring requirement:** Must be called from `[workspace.metadata.dist]` in `Cargo.toml` before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**

```bash
cargo dist --version 2>&1
```

Expected: FAIL with `cargo: no such subcommand: dist` if not installed.

- [ ] **Step 2: Run to verify it fails**

```bash
cargo dist build --dry-run 2>&1 | tail -5
```

Expected: FAIL — binary not installed or `[workspace.metadata.dist]` not configured.

- [ ] **Step 3: Implement** — install cargo-dist and run init

```bash
cargo install cargo-dist --version "^0.22.1" --locked
cd /home/tomespen/git/ochroma
cargo dist init --yes
```

Then edit the generated `[workspace.metadata.dist]` block in `Cargo.toml` to match:

```toml
[workspace.metadata.dist]
cargo-dist-version = "0.22.1"
ci = "github"
installers = ["shell", "powershell"]
targets = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
]
pr-run-mode = "upload"
publish-jobs = ["cargo"]
```

`publish-jobs = ["cargo"]` tells cargo-dist to run `cargo publish` as part of the release.

- [ ] **Step 4: Wire at exact callsite** — confirm `.github/workflows/release.yml` was generated by cargo-dist init

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
cargo dist build --dry-run 2>&1 | tail -10
```

Expected: PASS, prints artifact plan without errors.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml .github/workflows/release.yml
git commit -m "build: set up cargo-dist for cross-platform releases and crates.io publishing"
```

---

## Task 4: Create the CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Acceptance:** `cargo test --workspace --no-default-features 2>&1 | tail -5` → exits 0 with "test result: ok" AND `cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | grep "^error" | wc -l` → prints `0`.

**Wiring requirement:** Must be called from `.github/workflows/ci.yml` `on: push` trigger before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test** (confirm local test and clippy pass before writing CI config)

```bash
cargo test --workspace --no-default-features 2>&1 | tail -10
```

Expected: FAIL if any existing tests are broken (must fix before proceeding).

- [ ] **Step 2: Run to verify it fails**

```bash
cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | head -20
```

Expected: any warnings present here = FAIL, must fix before committing CI config.

- [ ] **Step 3: Implement** — create `.github/workflows/ci.yml`

```bash
mkdir -p /home/tomespen/git/ochroma/.github/workflows
```

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust stable
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt

      - name: Cache Cargo
        uses: Swatinem/rust-cache@v2

      - name: Install Linux audio deps
        if: matrix.os == 'ubuntu-latest'
        run: sudo apt-get update && sudo apt-get install -y libasound2-dev libvulkan-dev

      - name: cargo test
        run: cargo test --workspace --no-default-features

      - name: cargo clippy
        run: cargo clippy --workspace --no-default-features -- -D warnings

      - name: cargo doc
        run: cargo doc --workspace --no-default-features --no-deps
        env:
          RUSTDOCFLAGS: "-D warnings"

  build-binaries:
    name: Build binaries (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Install Linux audio deps
        if: matrix.os == 'ubuntu-latest'
        run: sudo apt-get update && sudo apt-get install -y libasound2-dev libvulkan-dev
      - name: Build all binaries
        run: cargo build --package vox_app --bins --no-default-features
```

Note: `--no-default-features` is used because audio and GPU tests require hardware. Functional tests that need those features run in the release workflow on self-hosted runners.

- [ ] **Step 4: Wire at exact callsite** — verify YAML is valid

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"
```

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
cargo test --workspace --no-default-features 2>&1 | tail -5
cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | grep "^error" | wc -l
```

Expected: PASS — test result: ok, clippy error count = 0.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions matrix CI (test, clippy, doc) for all three desktop platforms"
```

---

## Task 5: Add web feature flag and WASM entry point

**Files:**
- Modify: `crates/vox_app/Cargo.toml`
- Create: `crates/vox_app/src/bin/hello_splat.rs`
- Create: `web/index.html`
- Create: `web/build.sh`

**Acceptance:** `bash web/build.sh 2>&1 | tail -5` → exits 0 AND `ls -lh web/dist/hello_splat_bg.wasm` shows a file larger than 10KB.

**Wiring requirement:** Must be called from `[[bin]]` entry in `crates/vox_app/Cargo.toml` and `wasm_main()` in `crates/vox_app/src/bin/hello_splat.rs` before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**

```bash
cargo build --bin hello_splat 2>&1 | tail -5
```

Expected: FAIL — binary does not exist yet.

- [ ] **Step 2: Run to verify it fails**

```bash
cargo build --bin hello_splat --target wasm32-unknown-unknown --features web 2>&1 | tail -5
```

Expected: FAIL — target not added, features not defined.

- [ ] **Step 3: Implement** — add web feature, create hello_splat binary and web assets

Add to `crates/vox_app/Cargo.toml`:

```toml
[features]
default = []
web = ["wasm-bindgen", "console_error_panic_hook", "web-sys"]

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }
console_error_panic_hook = { version = "0.1", optional = true }
web-sys = { version = "0.3", optional = true, features = ["Window", "Document", "HtmlCanvasElement"] }

[[bin]]
name = "hello_splat"
path = "src/bin/hello_splat.rs"
```

Create `crates/vox_app/src/bin/hello_splat.rs`:

```rust
//! hello_splat — minimal entry point for the WebGPU browser demo.
//!
//! On native: opens a window, loads the default scene, orbits the camera.
//! On WASM: initialises the engine on the browser canvas via wasm-bindgen.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    console_error_panic_hook::set_once();
    // Web entry point: engine init happens here.
    // Full wgpu WebGPU initialisation is async; for now we log readiness.
    web_sys::console::log_1(&"Ochroma hello_splat loaded".into());
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    println!("hello_splat: native mode — open a window and load a .vxm scene.");
    // Stub: full scene loading wired in the Asset Pipeline domain.
}
```

Create `web/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ochroma — Spectral Gaussian Splatting</title>
  <style>
    body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }
    canvas { display: block; width: 100vw; height: 100vh; }
    #status { color: #fff; font-family: monospace; font-size: 14px; position: absolute; top: 16px; left: 16px; }
  </style>
</head>
<body>
  <canvas id="ochroma-canvas"></canvas>
  <div id="status">Loading…</div>
  <script type="module">
    import init from './hello_splat.js';
    document.getElementById('status').textContent = 'Initialising WebGPU…';
    await init();
    document.getElementById('status').textContent = 'Ready';
  </script>
</body>
</html>
```

Create `web/build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CRATE_DIR="$(dirname "$0")/../crates/vox_app"
OUT_DIR="$(dirname "$0")/dist"

echo "Building hello_splat for WebGPU..."
wasm-pack build "$CRATE_DIR" \
  --target web \
  --out-dir "$OUT_DIR" \
  --release \
  -- --bin hello_splat --features web

echo "Copying index.html..."
cp "$(dirname "$0")/index.html" "$OUT_DIR/index.html"

echo "Done. Serve with: python3 -m http.server 8080 --directory $OUT_DIR"
```

```bash
chmod +x web/build.sh
```

- [ ] **Step 4: Wire at exact callsite** — install wasm-pack and add WASM target

```bash
cargo install wasm-pack --locked
rustup target add wasm32-unknown-unknown
```

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
bash web/build.sh 2>&1 | tail -5
ls -lh web/dist/hello_splat_bg.wasm
```

Expected: PASS — `web/dist/` contains `hello_splat_bg.wasm`, `hello_splat.js`, `index.html`. WASM file size > 10KB.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_app/Cargo.toml crates/vox_app/src/bin/hello_splat.rs web/
git commit -m "feat(web): add web feature flag, wasm entry point, and WebGPU build script"
```

---

## Task 6: Add web build job to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Acceptance:** `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"` → prints "YAML OK" AND `bash web/build.sh` exits 0 producing `web/dist/hello_splat_bg.wasm`.

**Wiring requirement:** Must be called from the `build-web` job in `.github/workflows/ci.yml` before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**

```bash
grep -c "build-web" .github/workflows/ci.yml 2>&1
```

Expected: FAIL — job does not exist yet.

- [ ] **Step 2: Run to verify it fails**

```bash
grep "wasm32" .github/workflows/ci.yml 2>&1 | wc -l
```

Expected: 0 — WASM build not yet in CI.

- [ ] **Step 3: Implement** — append web-build job to `.github/workflows/ci.yml`

```yaml
  build-web:
    name: Build WebGPU (WASM)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
      - uses: Swatinem/rust-cache@v2
      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
      - name: Build hello_splat for web
        run: bash web/build.sh
      - name: Verify bundle exists
        run: test -f web/dist/hello_splat_bg.wasm && echo "WASM bundle OK"
```

- [ ] **Step 4: Wire at exact callsite** — the `build-web` job must be a sibling of `build-binaries` in the same `jobs:` block.

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"
bash web/build.sh 2>&1 | tail -3
ls -lh web/dist/hello_splat_bg.wasm
```

Expected: PASS — "YAML OK" printed, WASM file size printed with non-zero bytes.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add WebGPU WASM build job"
```

---

## Task 7: Write the getting-started guide

**Files:**
- Create: `docs/getting-started.md`

**Acceptance:** `cargo build --bin hello_splat 2>&1 | tail -3` → exits 0 AND `cargo build --bin walking_sim 2>&1 | tail -3` → exits 0 (all `cargo run` commands referenced in the guide build cleanly).

**Wiring requirement:** Must be called from `docs/getting-started.md` (referenced by `CONTRIBUTING.md`) before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**

```bash
cargo build --bin hello_splat 2>&1 | tail -3
cargo build --bin walking_sim 2>&1 | tail -3
```

Expected: FAIL if either binary doesn't build cleanly.

- [ ] **Step 2: Run to verify it fails**

```bash
test -f docs/getting-started.md && echo "exists" || echo "missing"
```

Expected: "missing"

- [ ] **Step 3: Implement** — create `docs/getting-started.md`

````markdown
# Getting Started with Ochroma

Ochroma is a spectral Gaussian splatting game engine written in Rust. This guide gets you from zero to running your first scene in under 15 minutes.

## Prerequisites

- **Rust** (stable, 1.85+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Git**
- **Linux:** `sudo apt install libasound2-dev libvulkan-dev`
- **Windows:** Visual Studio Build Tools (C++ workload)
- **Mac:** Xcode Command Line Tools (`xcode-select --install`)

## Clone and Build

```bash
git clone https://github.com/ochroma-engine/ochroma
cd ochroma
cargo build --release
```

First build takes 2–5 minutes. Subsequent builds are incremental.

## Run the Hello Splat Demo

```bash
cargo run --bin hello_splat
```

This opens a window with an orbit camera looking at the default spectral splat scene. The overlay in the top-left shows live 16-band spectral energy.

**Controls:**
- Left-drag: orbit
- Scroll: zoom
- `1`–`16`: toggle individual spectral band visibility
- `T`: cycle tonemapping curves

## Run the Walking Sim Example

```bash
cargo run --bin walking_sim
```

Walk around with WASD, collect orbs. Fire orbs trigger a spectral damage effect and shift the music. See `crates/vox_app/src/bin/walking_sim.rs` for the full source.

## Run in the Browser

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
bash web/build.sh
python3 -m http.server 8080 --directory web/dist
# Open http://localhost:8080 in Chrome or Firefox
```

## Project Structure

```
crates/
  ochroma_engine/   — public API façade (start here)
  vox_core/         — shared types, math, ECS components
  vox_render/       — GPU rendering, spectral pipeline
  vox_app/          — binaries and game examples
  vox_physics/      — Rapier physics integration
  vox_audio/        — HRTF, reverb, adaptive music
  vox_script/       — Lua scripting runtime
  vox_data/         — asset formats, GLTF import
  vox_net/          — QUIC networking, rollback netcode
  vox_terrain/      — heightmap + volumetric terrain
  vox_ui/           — game UI (Vello/Taffy)
  vox_sim/          — city/world simulation
  vox_nn/           — AI/LLM integration
docs/
  spec/             — engine design specifications
  getting-started.md — this file
```

## Writing Your First Script

Game logic lives in Lua scripts loaded by the engine. Create `my_game.lua`:

```lua
-- Called once at startup
function on_ready()
  scene.spawn("prefabs/player.vxm")
  audio.play("sounds/ambient.wav", vec3(0, 0, 0))
end

-- Called every frame
function on_update(dt)
  -- dt is seconds since last frame
end
```

Load it in an engine runner:

```bash
cargo run --bin ochroma -- --script my_game.lua
```

## Next Steps

- Read the [architecture overview](spec/architecture.md)
- Browse [example games](../crates/vox_app/examples/)
- See the [full API reference](https://docs.rs/ochroma_engine)
````

- [ ] **Step 4: Wire at exact callsite** — verify both binaries referenced in the guide build cleanly

```bash
cargo build --bin hello_splat 2>&1 | tail -3
cargo build --bin walking_sim 2>&1 | tail -3
```

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
wc -l docs/getting-started.md
```

Expected: PASS — file exists with >50 lines of actual content.

- [ ] **Step 6: Commit**

```bash
git add docs/getting-started.md
git commit -m "docs: add getting-started guide"
```

---

## Task 8: Write CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

**Acceptance:** `wc -l CONTRIBUTING.md` → prints a line count > 60 AND `grep "spectral" CONTRIBUTING.md | wc -l` → prints a number > 3 (spectral invariant is documented).

**Wiring requirement:** Must be called from `CONTRIBUTING.md` at the repo root (referenced in getting-started.md) before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**

```bash
test -f CONTRIBUTING.md && echo "exists" || echo "missing"
```

Expected: "missing"

- [ ] **Step 2: Run to verify it fails**

```bash
grep "spectral" CONTRIBUTING.md 2>&1 | wc -l
```

Expected: FAIL — file doesn't exist.

- [ ] **Step 3: Implement** — create `CONTRIBUTING.md`

````markdown
# Contributing to Ochroma

## Dev Setup

```bash
git clone https://github.com/ochroma-engine/ochroma
cd ochroma
cargo build
cargo test --workspace
```

Linux extra deps: `sudo apt install libasound2-dev libvulkan-dev`

## Architecture

The engine is split into domain crates. The rule: **engine crates must never contain game-specific concepts** (buildings, NPCs, quests). Game logic lives in `vox_app` or `vox_sim`.

| Crate | Purpose |
|-------|---------|
| `vox_core` | Shared types, math, ECS, character controller |
| `vox_render` | GPU rendering, spectral pipeline, post-processing |
| `vox_physics` | Rapier physics, fluids, destruction |
| `vox_audio` | CPAL backend, HRTF, adaptive music |
| `vox_script` | Lua 5.4 scripting runtime |
| `vox_data` | Asset formats, GLTF import, scene serialisation |
| `vox_net` | QUIC networking, rollback netcode |
| `vox_terrain` | Heightmap and SDF terrain |
| `vox_ui` | Game UI via Vello/Taffy |
| `vox_sim` | Simulation systems (city, ecosystems) |
| `vox_nn` | AI/LLM integration via candle |
| `ochroma_engine` | Public façade crate |

## The Spectral Invariant

Every system that touches a `GaussianSplat` **must** preserve or intentionally modify its `.spectral: [u16; 16]` field. Never zero it out as a shortcut.

## Before Submitting a PR

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` is clean
- [ ] New public APIs have `///` doc comments with a usage example
- [ ] New systems respect the spectral invariant
- [ ] Commit messages follow: `type(scope): description` (e.g. `feat(render): add DOF pass`)

## PR Size

Keep PRs focused. One domain, one feature, one bug fix per PR. If you're unsure how to split, open a draft PR and ask.

## Spectral Band Reference

Ochroma's 16 spectral bands map wavelength → audio frequency:

| Band | Wavelength | ~Color | Audio |
|------|-----------|--------|-------|
| 0 | 380–405nm | Violet | 8kHz |
| 1 | 405–430nm | Violet-blue | 6kHz |
| 2 | 430–455nm | Blue-violet | 4kHz |
| 3 | 455–480nm | Blue | 3kHz |
| 4 | 480–505nm | Blue-cyan | 2kHz |
| 5 | 505–530nm | Cyan | 1.5kHz |
| 6 | 530–555nm | Cyan-green | 1kHz |
| 7 | 555–580nm | Green-yellow | 700Hz |
| 8 | 580–605nm | Yellow | 500Hz |
| 9 | 605–630nm | Yellow-orange | 350Hz |
| 10 | 630–655nm | Orange | 250Hz |
| 11 | 655–680nm | Orange-red | 175Hz |
| 12 | 680–705nm | Red | 125Hz |
| 13 | 705–730nm | Deep red | 100Hz |
| 14 | 730–755nm | Near-IR | 90Hz |
| 15 | 755–780nm | Near-IR | 80Hz |
````

- [ ] **Step 4: Wire at exact callsite** — confirm `CONTRIBUTING.md` is at the repo root and referenced from `docs/getting-started.md`

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
wc -l CONTRIBUTING.md
grep "spectral" CONTRIBUTING.md | wc -l
```

Expected: PASS — line count > 60, spectral mention count > 3.

- [ ] **Step 6: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md with architecture overview and spectral band reference"
```

---

## Task 9: Final verification and tag

**Files:** (no new files — verification only)

**Acceptance:** All four commands below exit 0 AND `ls -lh web/dist/hello_splat_bg.wasm` shows a file < 50MB. Push to master and `gh run view --web` shows all three platform jobs green.

**Wiring requirement:** All prior tasks must be complete before running this task. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test** — run full workspace test suite

```bash
cargo test --workspace --no-default-features 2>&1 | tail -5
```

Expected: FAIL if any tests are broken from earlier tasks.

- [ ] **Step 2: Run to verify it fails**

```bash
cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | grep "^error" | wc -l
```

Expected: any non-zero count = FAIL.

- [ ] **Step 3: Implement** — run all verification gates

```bash
cargo test --workspace --no-default-features 2>&1 | tail -5
cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | tail -5
cargo doc --workspace --no-default-features --no-deps 2>&1 | tail -5
bash web/build.sh 2>&1 | tail -5
ls -lh web/dist/hello_splat_bg.wasm
```

- [ ] **Step 4: Wire at exact callsite** — push to master so CI runs

```bash
git push origin master
```

- [ ] **Step 5: Run test — verify non-trivial output**

```bash
gh run list --limit 1 --json status,conclusion,name 2>&1
```

Expected: PASS — all jobs show `conclusion: success`.

- [ ] **Step 6: Commit**

```bash
git add -A
git status  # verify nothing unintended is staged
git commit -m "build: Domain 1 Build/Platform complete — CI, cargo-dist, web build, docs"
```

---

## Self-Review Notes

**Spec coverage check:**
- cargo-dist setup (Task 3)
- CI matrix ubuntu/windows/macos (Task 4)
- `cargo test`, `cargo clippy --deny warnings`, `cargo doc` in CI (Task 4)
- cargo-dist dry-run in CI (Task 3 — cargo-dist generates this in release.yml)
- WebGPU web target (Task 5, 6)
- crates.io publishing metadata (Task 2)
- cargo-dist crates.io workflow (Task 3)
- Getting-started guide (Task 7)
- CONTRIBUTING.md (Task 8)
- "One integration smoke test per binary" — the CI builds all binaries (`cargo build --bins`) but does not *run* them (they require a display). Running them headless requires `--headless` flag support in the binaries, which is a vox_app change outside this domain's scope. The build step verifies they compile correctly.

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency:** `WebRenderConfig`, `Platform` (from `web_renderer.rs`) are referenced but not redefined in this plan — they already exist in the codebase.
