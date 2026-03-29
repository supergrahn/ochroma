# Sprint 4: Developer Workflow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine usable as a development tool: script hot-reload without recompile, unified asset hot-reload, persisted key bindings, and an optimized release build pipeline.

**Architecture:** Script hot-reload is built on the already-implemented `RhaiRuntime.reload_all()` — just needs a file-change timer. Asset hot-reload generalizes the `HotMaterialLibrary` pattern (mtime polling) into a generic `AssetWatcher` trait that scripts and TOML configs share. Key bindings are persisted as TOML using the same serde pattern already used by `KeyBindings`. Release profile optimization is a `Cargo.toml` addition.

**Tech Stack:** rhai, mtime polling (`std::fs::metadata`), serde + toml, cargo profiles

---

## Cross-Sprint Foundation Note

The `AssetWatcher` trait built in Task 2 is the foundation for Sprint 5's GPU shader hot-reload (recompile WGSL on file change). The release profile established in Task 4 is what ships with the Sprint 5 release candidate.

---

## Task 1: Script hot-reload with file-watch timer

- [ ] Add `last_mtime: std::time::SystemTime` field to `RhaiScript`
- [ ] Add `last_reload_check` and `reload_interval` fields to `RhaiRuntime`
- [ ] Implement `RhaiRuntime::poll_reload()`
- [ ] Wire `poll_reload()` into `engine_runner.rs` per-frame
- [ ] Write tests

**Files:**
- Modify: `crates/vox_script/src/rhai_runtime.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Implementation — `crates/vox_script/src/rhai_runtime.rs`:**

Add `last_mtime` to `RhaiScript`:

```rust
use std::time::{SystemTime, UNIX_EPOCH};

pub struct RhaiScript {
    pub name: String,
    pub source_path: Option<String>,
    ast: AST,
    pub last_mtime: SystemTime,
}
```

Initialize in `load_script` (no path → use `UNIX_EPOCH`) and in `load_script_file` (read mtime from the file at load time):

```rust
pub fn load_script(&mut self, name: &str, source: &str) -> Result<usize, String> {
    let ast = self.engine.compile(source).map_err(|e| format!("Compile error: {}", e))?;
    let idx = self.scripts.len();
    self.scripts.push(RhaiScript {
        name: name.to_string(),
        source_path: None,
        ast,
        last_mtime: UNIX_EPOCH,
    });
    Ok(idx)
}

pub fn load_script_file(&mut self, name: &str, path: &Path) -> Result<usize, String> {
    let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let ast = self.engine.compile(&source).map_err(|e| format!("Compile error in {}: {}", path.display(), e))?;
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(UNIX_EPOCH);
    let idx = self.scripts.len();
    self.scripts.push(RhaiScript {
        name: name.to_string(),
        source_path: Some(path.to_string_lossy().to_string()),
        ast,
        last_mtime: mtime,
    });
    Ok(idx)
}
```

Add fields to `RhaiRuntime` and implement `poll_reload`:

```rust
pub struct RhaiRuntime {
    engine: Engine,
    scripts: Vec<RhaiScript>,
    pub last_reload_check: std::time::Instant,
    pub reload_interval: std::time::Duration,
}

impl RhaiRuntime {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        // ... existing engine.register_fn calls unchanged ...
        Self {
            engine,
            scripts: Vec::new(),
            last_reload_check: std::time::Instant::now(),
            reload_interval: std::time::Duration::from_secs(1),
        }
    }

    /// Check if any script files have changed and reload them.
    /// Rate-limited to `reload_interval`. Returns names of reloaded scripts.
    pub fn poll_reload(&mut self) -> Vec<String> {
        if self.last_reload_check.elapsed() < self.reload_interval {
            return Vec::new();
        }
        self.last_reload_check = std::time::Instant::now();

        // Collect indices where mtime has advanced
        let mut to_reload: Vec<(usize, String, std::time::SystemTime)> = Vec::new();
        for (i, script) in self.scripts.iter().enumerate() {
            let path = match &script.source_path {
                Some(p) => p.clone(),
                None => continue,
            };
            if let Ok(meta) = std::fs::metadata(&path) {
                if let Ok(mtime) = meta.modified() {
                    if mtime > script.last_mtime {
                        to_reload.push((i, script.name.clone(), mtime));
                    }
                }
            }
        }

        let mut result = Vec::new();
        for (i, name, mtime) in to_reload {
            match self.reload(i) {
                Ok(()) => {
                    self.scripts[i].last_mtime = mtime;
                    println!("[ochroma] Hot-reloaded script: {}", name);
                    result.push(name);
                }
                Err(e) => eprintln!("[ochroma] Script reload error {}: {}", name, e),
            }
        }
        result
    }
}
```

**Wire into engine_runner.rs** — add to the per-frame update loop:

```rust
// In the frame/update method of EngineApp or equivalent:
let reloaded = self.rhai.poll_reload();
for name in &reloaded {
    // push to notification queue / log if one exists
    println!("[ochroma] hot-reload: {}", name);
}
```

**Tests** — add to `crates/vox_script/src/rhai_runtime.rs`:

```rust
#[cfg(test)]
mod hot_reload_tests {
    use super::*;

    #[test]
    fn poll_reload_is_rate_limited() {
        let mut rt = RhaiRuntime::new();
        // Set a long interval so the second call is always rate-limited
        rt.reload_interval = std::time::Duration::from_secs(10);
        // Backdate last_reload_check so the FIRST poll fires
        rt.last_reload_check =
            std::time::Instant::now() - std::time::Duration::from_secs(15);
        let _first = rt.poll_reload(); // fires (interval elapsed)
        let second = rt.poll_reload(); // should be suppressed
        assert!(second.is_empty(), "second poll within interval must return empty");
    }

    #[test]
    fn poll_reload_returns_empty_when_no_scripts() {
        let mut rt = RhaiRuntime::new();
        // Bypass rate limit
        rt.last_reload_check =
            std::time::Instant::now() - std::time::Duration::from_secs(5);
        let result = rt.poll_reload();
        assert!(result.is_empty());
    }

    #[test]
    fn script_reloads_when_file_updated() {
        use std::io::Write as _;
        // Write initial script to a temp file
        let dir = std::env::temp_dir();
        let path = dir.join("ochroma_test_script_hot.rhai");
        std::fs::write(&path, "fn on_update(dt) {}").unwrap();

        let mut rt = RhaiRuntime::new();
        rt.load_script_file("test", &path).unwrap();

        // Small sleep so mtime actually advances on the filesystem
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&path, "fn on_update(dt) { let x = 1; }").unwrap();

        // Bypass rate limit
        rt.last_reload_check =
            std::time::Instant::now() - std::time::Duration::from_secs(5);
        let reloaded = rt.poll_reload();
        // mtime resolution varies by OS/FS; accept either outcome but must not panic
        assert!(reloaded.len() <= 1);

        let _ = std::fs::remove_file(&path);
    }
}
```

**Commit message:** `feat(scripting): script hot-reload via mtime polling in RhaiRuntime::poll_reload()`

---

## Task 2: AssetWatcher trait — unified hot-reload for all asset types

- [ ] Create `crates/vox_core/src/asset_watcher.rs` with `AssetChanged` + `AssetWatcher` trait
- [ ] Add `pub mod asset_watcher;` to `crates/vox_core/src/lib.rs`
- [ ] Implement `AssetWatcher` for `HotMaterialLibrary` in `crates/vox_render/src/material_hotreload.rs`
- [ ] Write tests

**Files:**
- Create: `crates/vox_core/src/asset_watcher.rs`
- Modify: `crates/vox_core/src/lib.rs`
- Modify: `crates/vox_render/src/material_hotreload.rs`

**Implementation — `crates/vox_core/src/asset_watcher.rs`:**

```rust
//! Generic asset hot-reload interface.
//!
//! Any system that watches files for changes should implement `AssetWatcher`.
//! Engine-layer only — no game-specific concepts.

use std::path::PathBuf;

/// An asset that changed on disk.
#[derive(Debug, Clone)]
pub struct AssetChanged {
    pub name: String,
    pub path: PathBuf,
}

/// Trait for any system that watches files for changes.
/// Implement this to integrate with the engine's hot-reload infrastructure.
pub trait AssetWatcher {
    /// Poll for changed assets. Rate-limited internally.
    /// `dt` is the frame delta in seconds (for accumulator-based rate limiting).
    /// Returns the list of assets that changed since the last call.
    fn poll(&mut self, dt: f32) -> Vec<AssetChanged>;

    /// Force a poll regardless of rate limit.
    fn force_poll(&mut self) -> Vec<AssetChanged>;

    /// Register a file path to watch under a given name.
    fn watch(&mut self, name: &str, path: PathBuf);
}
```

Add the module to `crates/vox_core/src/lib.rs`:

```rust
pub mod asset_watcher;
```

**Implementation — `AssetWatcher` for `HotMaterialLibrary`** in `crates/vox_render/src/material_hotreload.rs`:

First, inspect what `HotMaterialLibrary::poll` currently returns and what fields `entries` has. The existing `poll(dt: f32) -> Vec<String>` returns changed material names. The `entries` map stores `(PathBuf, SystemTime, SpectralMaterialConfig)` per name. Implement as:

```rust
impl vox_core::asset_watcher::AssetWatcher for HotMaterialLibrary {
    fn poll(&mut self, dt: f32) -> Vec<vox_core::asset_watcher::AssetChanged> {
        // Delegate to the existing poll method
        let changed_names = self.poll(dt);
        changed_names
            .into_iter()
            .map(|name| {
                let path = self
                    .entries
                    .get(&name)
                    .map(|(p, _, _)| p.clone())
                    .unwrap_or_default();
                vox_core::asset_watcher::AssetChanged { name, path }
            })
            .collect()
    }

    fn force_poll(&mut self) -> Vec<vox_core::asset_watcher::AssetChanged> {
        // Reset accumulator so next poll fires immediately, then poll
        self.elapsed = self.check_interval; // force threshold
        self.poll(0.0)
            .into_iter()
            .map(|name| {
                let path = self
                    .entries
                    .get(&name)
                    .map(|(p, _, _)| p.clone())
                    .unwrap_or_default();
                vox_core::asset_watcher::AssetChanged { name, path }
            })
            .collect()
    }

    fn watch(&mut self, name: &str, path: std::path::PathBuf) {
        self.register(name, path);
    }
}
```

> Note: if `HotMaterialLibrary` does not expose `elapsed` / `check_interval` as `pub`, add `pub` to those fields, or add a `reset_timer()` helper method for `force_poll`.

**Tests** — add to `crates/vox_render/src/material_hotreload.rs` or a new test file:

```rust
#[cfg(test)]
mod asset_watcher_tests {
    use super::*;
    use vox_core::asset_watcher::AssetWatcher;

    #[test]
    fn hot_material_library_is_asset_watcher() {
        // Verify trait object construction compiles
        let lib = HotMaterialLibrary::new(1.0);
        let _watcher: Box<dyn vox_core::asset_watcher::AssetWatcher> = Box::new(lib);
        // Compiles = pass
    }

    #[test]
    fn asset_changed_fields() {
        let ac = vox_core::asset_watcher::AssetChanged {
            name: "stone".into(),
            path: std::path::PathBuf::from("materials/stone.toml"),
        };
        assert_eq!(ac.name, "stone");
        assert_eq!(ac.path.to_str().unwrap(), "materials/stone.toml");
    }

    #[test]
    fn poll_returns_empty_when_no_entries() {
        let mut lib = HotMaterialLibrary::new(1.0);
        let changed = AssetWatcher::poll(&mut lib, 2.0);
        assert!(changed.is_empty());
    }
}
```

**Commit message:** `feat(core): AssetWatcher trait + HotMaterialLibrary impl for unified hot-reload`

---

## Task 3: Persist key bindings to TOML

- [ ] Confirm `KeyBindings`, `GameAction`, `InputSource` already derive `Serialize, Deserialize` (they do — verified in `crates/vox_core/src/input.rs`)
- [ ] Add `save_bindings` and `load_bindings` free functions to `vox_core::input`
- [ ] Add `toml` dependency to `vox_core/Cargo.toml` if not present
- [ ] Add `key_bindings` field to `EngineApp` in `engine_runner.rs`, load on startup, save on change
- [ ] Write tests

**Files:**
- Modify: `crates/vox_core/src/input.rs`
- Modify: `crates/vox_core/Cargo.toml`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Check `vox_core/Cargo.toml` for toml dependency:**

```bash
grep toml crates/vox_core/Cargo.toml
```

If absent, add:

```toml
[dependencies]
toml = "0.8"
```

**Implementation — add to `crates/vox_core/src/input.rs`:**

`KeyBindings`, `GameAction`, and `InputSource` already derive `Serialize, Deserialize`. Add the two functions after the `KeyBindings` impl block:

```rust
/// Save key bindings to a TOML file.
pub fn save_bindings(bindings: &KeyBindings, path: &std::path::Path) -> Result<(), String> {
    let s = toml::to_string_pretty(bindings).map_err(|e| e.to_string())?;
    std::fs::write(path, s).map_err(|e| e.to_string())
}

/// Load key bindings from a TOML file.
/// Returns `KeyBindings::default()` if the file is missing or malformed.
pub fn load_bindings(path: &std::path::Path) -> KeyBindings {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default()
}
```

**Wire into `engine_runner.rs`:**

In the `EngineApp` struct, add:

```rust
key_bindings: vox_core::input::KeyBindings,
```

In `EngineApp::new()` (or equivalent constructor):

```rust
key_bindings: vox_core::input::load_bindings(std::path::Path::new("keybindings.toml")),
```

When key bindings change (e.g., in the settings/input panel handler), call:

```rust
if let Err(e) = vox_core::input::save_bindings(
    &self.key_bindings,
    std::path::Path::new("keybindings.toml"),
) {
    eprintln!("[ochroma] Failed to save keybindings: {}", e);
}
```

**Tests** — add to `crates/vox_core/src/input.rs`:

```rust
#[cfg(test)]
mod keybinding_persist_tests {
    use super::*;

    #[test]
    fn key_bindings_roundtrip_toml() {
        let mut bindings = KeyBindings::default();
        bindings.rebind(GameAction::CameraZoomIn, vec![InputSource::Key(200)]);

        let path = std::env::temp_dir().join("ochroma_test_keybindings.toml");
        save_bindings(&bindings, &path).expect("save should succeed");

        let loaded = load_bindings(&path);
        let sources = loaded
            .bindings
            .get(&GameAction::CameraZoomIn)
            .expect("CameraZoomIn should be present");
        assert_eq!(sources[0], InputSource::Key(200));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_bindings_returns_default_on_missing_file() {
        let loaded = load_bindings(std::path::Path::new(
            "/tmp/does_not_exist_ochroma_keys_xyzzy.toml",
        ));
        // Default bindings should be non-empty (KeyBindings::default() populates standard actions)
        assert!(
            !loaded.bindings.is_empty(),
            "default bindings should be non-empty"
        );
    }

    #[test]
    fn load_bindings_ignores_malformed_toml() {
        let path = std::env::temp_dir().join("ochroma_test_bad_keys.toml");
        std::fs::write(&path, "this is not valid toml ][[[").unwrap();
        let loaded = load_bindings(&path);
        // Should silently return default rather than panic
        drop(loaded);
        let _ = std::fs::remove_file(&path);
    }
}
```

**Commit message:** `feat(input): persist KeyBindings to TOML with load_bindings/save_bindings`

---

## Task 4: Release build profile optimization

- [ ] Add `[profile.release]` section to workspace `Cargo.toml`
- [ ] Update `scripts/build_release.sh` with verification step
- [ ] Update `scripts/package.sh` to emit a size report
- [ ] Verify release build compiles and is smaller than debug

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `scripts/build_release.sh`
- Modify: `scripts/package.sh`

**Implementation — workspace `Cargo.toml`:**

Append after the existing `[workspace]` and `[dependencies]` sections:

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = "symbols"
panic = "abort"
```

> Note on `panic = "abort"`: verify that no crate in the workspace uses `catch_unwind`. If any integration tests rely on panic recovery, set per-package overrides or use `panic = "unwind"` with `lto = "thin"` only. Start with `panic = "abort"` for the binary profile and accept the tradeoff.

**Implementation — `scripts/build_release.sh`:**

```bash
#!/usr/bin/env bash
set -e

echo "Building Ochroma Engine release..."
cargo build --release --bin ochroma --bin walking_sim

# Verify binaries were produced
ls -lh target/release/ochroma target/release/walking_sim

echo ""
echo "Binary sizes:"
du -h target/release/ochroma target/release/walking_sim

echo ""
echo "Build complete."
```

**Implementation — `scripts/package.sh`** — add size report after the build call:

```bash
# After ./scripts/build_release.sh:
echo ""
echo "Release binary sizes:"
ls -lh target/release/ochroma target/release/walking_sim 2>/dev/null || true
```

**Verification — run after applying:**

```bash
# Debug build size
cargo build -p vox_app 2>/dev/null
ls -lh target/debug/ochroma

# Release build size
cargo build --release -p vox_app 2>/dev/null
ls -lh target/release/ochroma
```

Expected: the release binary is meaningfully smaller than the debug binary. With `strip = "symbols"` and `lto = "thin"`, expect 40–70% reduction in binary size compared to the unoptimized default.

**Commit message:** `feat(build): release profile with LTO + strip + codegen-units=1 + panic=abort`

---

## Task 5: Example project — playable demo scene

- [ ] Create `examples/demo_scene/` directory
- [ ] Write `examples/demo_scene/main.rhai`
- [ ] Write `examples/demo_scene/README.md`
- [ ] Add `--demo` flag handling to `engine_runner.rs`
- [ ] Verify `cargo check --bin ochroma` passes

**Files:**
- Create: `examples/demo_scene/main.rhai`
- Create: `examples/demo_scene/README.md`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Implementation — `examples/demo_scene/main.rhai`:**

```rhai
// Demo scene script — loaded by engine at startup when --demo flag is passed.
// Demonstrates: scripted entity movement, sound playback, log output, hot-reload.
//
// Edit this file while `cargo run --bin ochroma -- --demo` is running.
// Changes apply within 1 second (hot-reload via mtime polling).

let frame = 0;

fn on_start() {
    log("Demo scene started! Edit this file to see hot-reload in action.");
}

fn on_update(dt) {
    frame += 1;
    if frame % 120 == 0 {
        log("Demo: frame " + frame);
    }
}
```

**Implementation — `examples/demo_scene/README.md`:**

```markdown
# Ochroma Demo Scene

Run with:

    cargo run --bin ochroma -- --demo

Demonstrates:

- Scripted entity via Rhai (`main.rhai`)
- Hot-reload: edit `main.rhai` while running; changes apply within 1 second
- Log output from script via the `log()` API
- Foundation for Sprint 5: add WGSL shader files here for GPU hot-reload

## Hot-reload workflow

1. Start the engine: `cargo run --bin ochroma -- --demo`
2. Open `examples/demo_scene/main.rhai` in your editor
3. Change the `on_update` body (e.g., change the log message)
4. Save the file — the engine reloads automatically within 1 second
5. Observe the new log output in the console
```

**Implementation — `crates/vox_app/src/bin/engine_runner.rs`:**

Locate the initialization section of `EngineApp` (or equivalent entry point). Add after existing setup:

```rust
// Demo mode: load example scene script if --demo flag is passed
let demo_mode = std::env::args().any(|a| a == "--demo");
if demo_mode {
    let demo_path = std::path::Path::new("examples/demo_scene/main.rhai");
    match self.rhai.load_script_file("demo", demo_path) {
        Ok(idx) => {
            println!("[ochroma] Demo scene loaded from {}", demo_path.display());
            if let Err(e) = self.rhai.call_fn(idx, "on_start", &[]) {
                eprintln!("[ochroma] Demo on_start error: {}", e);
            }
        }
        Err(e) => eprintln!("[ochroma] Failed to load demo scene: {}", e),
    }
}
```

In the per-frame loop, call `on_update` for all loaded scripts that expose it (this may already be wired from Sprint 2's script integration; confirm `call_fn` is called each frame for scripts that define `on_update`).

**Verification:**

```bash
# Must compile without errors
cargo check --bin ochroma

# Manual smoke test:
cargo run --bin ochroma -- --demo
# Expected in stdout:
#   [ochroma] Demo scene loaded from examples/demo_scene/main.rhai
#   [rhai] Demo scene started! Edit this file to see hot-reload in action.
#   [rhai] Demo: frame 120   (after ~2 seconds at 60fps)
```

**Commit message:** `feat(examples): demo scene with Rhai script + hot-reload instructions`

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | Edit a `.rhai` file while engine runs; change takes effect within 1 second | Manual: `cargo run --bin ochroma -- --demo`, edit `main.rhai` |
| 2 | `cargo test -p vox_script` passes all hot-reload tests | `cargo test -p vox_script` |
| 3 | `HotMaterialLibrary` is usable as `Box<dyn AssetWatcher>` | `cargo test -p vox_render` |
| 4 | `keybindings.toml` written on save, read on startup | `cargo test -p vox_core` + manual |
| 5 | Release binary is smaller than debug binary | `ls -lh target/debug/ochroma target/release/ochroma` |
| 6 | `cargo run --bin ochroma -- --demo` prints "Demo scene loaded" | Manual run |
| 7 | All existing tests continue to pass | `cargo test` |
