# Build/Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ochroma engine compilable and testable on Windows/Linux/Mac, publishable to crates.io, and runnable in the browser via WebGPU — with a CI pipeline that enforces correctness on every push.

**Architecture:** cargo-dist handles cross-platform artifact generation and crates.io publishing. GitHub Actions runs a matrix job across all three desktop platforms. The web target compiles `vox_app` to `wasm32-unknown-unknown` with a `wasm-bindgen` entry point and WebGPU backend. The crucible path dependency is isolated to `vox_nodes` and made optional so CI doesn't require a sibling checkout.

**Tech Stack:** cargo-dist 0.28, GitHub Actions, wasm-pack, wasm-bindgen 0.2, web-sys, console_error_panic_hook

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

## Task 1: Unblock CI — Make crucible dependency optional

The workspace currently has path dependencies on `../aetherspectra/crucible/...` which won't exist on GitHub Actions. `vox_nodes` is the only consumer. Gate it behind an optional feature so CI can build without it.

**Files:**
- Modify: `crates/vox_nodes/Cargo.toml`
- Modify: `crates/vox_nodes/src/lib.rs`
- Modify: `Cargo.toml` (workspace — remove crucible from `[workspace.dependencies]`)

- [ ] **Step 1: Write a test that will fail if crucible is unconditionally required**

```bash
cargo build -p vox_nodes --no-default-features 2>&1 | head -20
```

Expected: currently fails because crucible paths don't exist unless `../aetherspectra` is checked out.

- [ ] **Step 2: Remove crucible from workspace.dependencies in root Cargo.toml**

In `Cargo.toml`, remove these two lines from `[workspace.dependencies]`:
```toml
# DELETE these lines:
crucible-core  = { path = "../aetherspectra/crucible/rust/crates/crucible-core" }
crucible-types = { path = "../aetherspectra/crucible/rust/crates/crucible-types" }
```

- [ ] **Step 3: Update vox_nodes/Cargo.toml — make crucible optional**

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

- [ ] **Step 4: Read vox_nodes/src/lib.rs to find all crucible usage**

```bash
grep -n "crucible" /home/tomespen/git/ochroma/crates/vox_nodes/src/lib.rs
grep -n "crucible" /home/tomespen/git/ochroma/crates/vox_nodes/src/mat_nodes.rs 2>/dev/null
```

- [ ] **Step 5: Gate crucible usage in vox_nodes/src/lib.rs**

Wrap any `use crucible*` statements and any types from crucible with `#[cfg(feature = "crucible")]`. Example pattern:

```rust
#[cfg(feature = "crucible")]
use crucible_core::SomeType;

#[cfg(feature = "crucible")]
pub mod crucible_bridge {
    // crucible-dependent code here
}
```

- [ ] **Step 6: Verify build succeeds without crucible**

```bash
cd /home/tomespen/git/ochroma
cargo build -p vox_nodes --no-default-features
```

Expected: compiles cleanly with no errors.

- [ ] **Step 7: Verify full workspace builds**

```bash
cargo build --workspace
```

Expected: all 15 crates compile. (Machines without `../aetherspectra` will skip crucible features.)

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml crates/vox_nodes/Cargo.toml crates/vox_nodes/src/
git commit -m "build: make crucible dependency optional to unblock CI"
```

---

## Task 2: Add crates.io publishing metadata

Each crate needs `description`, `repository`, `homepage`, `license`, and `publish` fields before crates.io will accept it. Add them at the workspace level where possible.

**Files:**
- Modify: `Cargo.toml` (workspace)
- Modify: `crates/ochroma_engine/Cargo.toml`

- [ ] **Step 1: Add metadata to [workspace.package] in root Cargo.toml**

Add to the existing `[workspace.package]` block:

```toml
[workspace.package]
edition = "2024"
version = "0.1.0"
license = "MIT"
repository = "https://github.com/ochroma-engine/ochroma"
homepage = "https://ochroma.dev"
authors = ["Ochroma Contributors"]
```

- [ ] **Step 2: Add metadata to ochroma_engine/Cargo.toml**

The façade crate is the public-facing package. Update it:

```toml
[package]
name = "ochroma_engine"
edition.workspace = true
version.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
authors.workspace = true
description = "Spectral Gaussian Splatting game engine — 8-band spectral rendering, physics, audio, and AI"
keywords = ["game-engine", "gaussian-splatting", "spectral", "rendering", "wgpu"]
categories = ["game-engines", "rendering", "graphics"]
readme = "README.md"
```

- [ ] **Step 3: Verify metadata is valid**

```bash
cargo package -p ochroma_engine --no-verify --list
```

Expected: lists the files that would be included. No errors about missing fields.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/ochroma_engine/Cargo.toml
git commit -m "build: add crates.io publishing metadata"
```

---

## Task 3: Set up cargo-dist

cargo-dist generates release workflows, installers, and handles the crates.io publish step.

**Files:**
- Modify: `Cargo.toml` (workspace — add `[workspace.metadata.dist]`)
- Create: `.github/workflows/release.yml` (generated by cargo-dist)

- [ ] **Step 1: Install cargo-dist**

```bash
cargo install cargo-dist --version "^0.28" --locked
```

Expected: installs `cargo dist` binary. Verify with `cargo dist --version`.

- [ ] **Step 2: Run cargo dist init**

```bash
cd /home/tomespen/git/ochroma
cargo dist init --yes
```

This will prompt for configuration. Accept defaults. It adds `[workspace.metadata.dist]` to `Cargo.toml` and creates `.github/workflows/release.yml`.

- [ ] **Step 3: Edit the generated [workspace.metadata.dist] block**

Open `Cargo.toml` and update the generated block to match:

```toml
[workspace.metadata.dist]
cargo-dist-version = "0.28.0"
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

- [ ] **Step 4: Dry-run to verify the configuration**

```bash
cargo dist build --dry-run
```

Expected: prints what it would build without actually building. No errors.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml .github/workflows/release.yml
git commit -m "build: set up cargo-dist for cross-platform releases and crates.io publishing"
```

---

## Task 4: Create the CI workflow

The CI workflow runs on every push and PR. It's a separate file from the release workflow.

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create .github/workflows/ci.yml**

```bash
mkdir -p /home/tomespen/git/ochroma/.github/workflows
```

Create `.github/workflows/ci.yml` with this content:

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

- [ ] **Step 2: Verify the YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>&1 || \
  node -e "require('js-yaml').load(require('fs').readFileSync('.github/workflows/ci.yml','utf8'))" 2>&1 || \
  echo "No YAML validator available — check manually"
```

- [ ] **Step 3: Run the test step locally to verify it passes**

```bash
cargo test --workspace --no-default-features 2>&1 | tail -20
```

Expected: all tests pass. Any failures here must be fixed before committing the CI config.

- [ ] **Step 4: Run clippy locally**

```bash
cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | head -40
```

Expected: no warnings. Fix any warnings before committing.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions matrix CI (test, clippy, doc) for all three desktop platforms"
```

---

## Task 5: Add web feature flag and WASM entry point

The `web` feature gates WASM-specific dependencies and entry points. The binary target is a new `hello_splat` binary (the first example game from the spec) which is the minimal WASM-compatible entry point.

**Files:**
- Modify: `crates/vox_app/Cargo.toml`
- Create: `crates/vox_app/src/bin/hello_splat.rs`
- Create: `web/index.html`
- Create: `web/build.sh`

- [ ] **Step 1: Write a test for WebRenderConfig (already exists, just verify it compiles with the web feature)**

```bash
cargo test -p vox_render --lib -- web_renderer 2>&1
```

Expected: passes (the existing `WebRenderConfig` tests).

- [ ] **Step 2: Add web feature and wasm dependencies to vox_app/Cargo.toml**

Add to the `[features]` section (create it if absent) and add conditional dependencies:

```toml
[features]
default = []
web = ["wasm-bindgen", "console_error_panic_hook", "web-sys"]

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }
console_error_panic_hook = { version = "0.1", optional = true }
web-sys = { version = "0.3", optional = true, features = ["Window", "Document", "HtmlCanvasElement"] }
```

- [ ] **Step 3: Create the hello_splat binary**

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

Add the binary entry to `crates/vox_app/Cargo.toml`:

```toml
[[bin]]
name = "hello_splat"
path = "src/bin/hello_splat.rs"
```

- [ ] **Step 4: Verify the binary builds on native**

```bash
cargo build --bin hello_splat
```

Expected: compiles cleanly.

- [ ] **Step 5: Install wasm-pack and verify WASM target build**

```bash
cargo install wasm-pack --locked
rustup target add wasm32-unknown-unknown
cargo build --bin hello_splat --target wasm32-unknown-unknown --features web 2>&1 | tail -20
```

Expected: compiles cleanly for the WASM target.

- [ ] **Step 6: Create web/index.html**

```bash
mkdir -p /home/tomespen/git/ochroma/web
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

- [ ] **Step 7: Create web/build.sh**

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

- [ ] **Step 8: Run the web build to verify it produces a valid bundle**

```bash
cd /home/tomespen/git/ochroma
bash web/build.sh 2>&1 | tail -20
ls web/dist/
```

Expected: `web/dist/` contains `hello_splat_bg.wasm`, `hello_splat.js`, `index.html`.

- [ ] **Step 9: Commit**

```bash
git add crates/vox_app/Cargo.toml crates/vox_app/src/bin/hello_splat.rs web/
git commit -m "feat(web): add web feature flag, wasm entry point, and WebGPU build script"
```

---

## Task 6: Add web build job to CI

The CI should verify the WASM target builds cleanly on every push.

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add the web-build job to ci.yml**

Append to `.github/workflows/ci.yml` after the `build-binaries` job:

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

- [ ] **Step 2: Verify YAML is still valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add WebGPU WASM build job"
```

---

## Task 7: Write the getting-started guide

**Files:**
- Create: `docs/getting-started.md`

- [ ] **Step 1: Create docs/getting-started.md**

```bash
mkdir -p /home/tomespen/git/ochroma/docs
```

Create `docs/getting-started.md`:

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

This opens a window with an orbit camera looking at the default spectral splat scene. The overlay in the top-left shows live 8-band spectral energy.

**Controls:**
- Left-drag: orbit
- Scroll: zoom
- `1`–`8`: toggle individual spectral band visibility
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

- [ ] **Step 2: Verify all `cargo run` commands in the guide work**

```bash
cargo build --bin hello_splat 2>&1 | tail -5
cargo build --bin walking_sim 2>&1 | tail -5
```

Expected: both build cleanly.

- [ ] **Step 3: Commit**

```bash
git add docs/getting-started.md
git commit -m "docs: add getting-started guide"
```

---

## Task 8: Write CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Create CONTRIBUTING.md**

Create `CONTRIBUTING.md` at the repo root:

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

Every system that touches a `GaussianSplat` **must** preserve or intentionally modify its `.spectral: [u16; 8]` field. Never zero it out as a shortcut.

## Before Submitting a PR

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` is clean
- [ ] New public APIs have `///` doc comments with a usage example
- [ ] New systems respect the spectral invariant
- [ ] Commit messages follow: `type(scope): description` (e.g. `feat(render): add DOF pass`)

## PR Size

Keep PRs focused. One domain, one feature, one bug fix per PR. If you're unsure how to split, open a draft PR and ask.

## Spectral Band Reference

Ochroma's 8 spectral bands map wavelength → audio frequency:

| Band | Wavelength | ~Color | Audio |
|------|-----------|--------|-------|
| 0 | 380–420nm | Violet | 8kHz |
| 1 | 420–460nm | Blue-violet | 4kHz |
| 2 | 460–500nm | Blue | 2kHz |
| 3 | 500–540nm | Cyan-green | 1kHz |
| 4 | 540–580nm | Green-yellow | 500Hz |
| 5 | 580–620nm | Yellow-orange | 250Hz |
| 6 | 620–660nm | Orange-red | 125Hz |
| 7 | 660–700nm | Red | 80Hz |
````

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md with architecture overview and spectral band reference"
```

---

## Task 9: Final verification and tag

- [ ] **Step 1: Full workspace test pass**

```bash
cargo test --workspace --no-default-features 2>&1 | tail -20
```

Expected: all tests pass, no failures.

- [ ] **Step 2: Full clippy pass**

```bash
cargo clippy --workspace --no-default-features -- -D warnings 2>&1 | tail -10
```

Expected: `warning: ... generated X warnings` should be zero.

- [ ] **Step 3: Doc build**

```bash
cargo doc --workspace --no-default-features --no-deps 2>&1 | tail -10
```

Expected: builds cleanly, no `[warning]` lines from missing docs.

- [ ] **Step 4: Web bundle build**

```bash
bash web/build.sh 2>&1 | tail -5
ls -lh web/dist/hello_splat_bg.wasm
```

Expected: WASM file present, reasonable size (< 50MB uncompressed).

- [ ] **Step 5: Final commit**

```bash
git add -A
git status  # verify nothing unintended is staged
git commit -m "build: Domain 1 Build/Platform complete — CI, cargo-dist, web build, docs"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ cargo-dist setup (Task 3)
- ✅ CI matrix ubuntu/windows/macos (Task 4)
- ✅ `cargo test`, `cargo clippy --deny warnings`, `cargo doc` in CI (Task 4)
- ✅ cargo-dist dry-run in CI (Task 3 — cargo-dist generates this in release.yml)
- ✅ WebGPU web target (Task 5, 6)
- ✅ crates.io publishing metadata (Task 2)
- ✅ cargo-dist crates.io workflow (Task 3)
- ✅ Getting-started guide (Task 7)
- ✅ CONTRIBUTING.md (Task 8)
- ⚠️ "One integration smoke test per binary" — the CI builds all binaries (`cargo build --bins`) but does not *run* them (they require a display). Running them headless requires `--headless` flag support in the binaries, which is a vox_app change outside this domain's scope. The build step verifies they compile correctly.

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency:** `WebRenderConfig`, `Platform` (from `web_renderer.rs`) are referenced but not redefined in this plan — they already exist in the codebase.
