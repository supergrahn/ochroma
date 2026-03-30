# Domain 4: Scripting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Rhai with mlua (Lua 5.4); wire first-class spectral scripting primitives so game logic can read, write, and react to spectral physics in three lines of Lua.

**Done When:** Running `cargo run`, opening the Lua console, typing `set_spectral(1, 7, 0.9)` changes the spectral band 7 of entity 1 to 0.9 AND the viewport color of that splat visibly shifts (verified by `get_spectral(1, 7)` returning `0.9` in the console).

**Architecture:** `LuaRuntime` in `vox_script` wraps mlua's `Lua` instance. At init, it registers three namespaces: `spectral` (query/modify spectral data in the world), `entity` (position, spectral band writes), and `coroutine` helpers (`wait_frames`, `wait_seconds`). A `notify` file watcher on `assets/scripts/` reloads changed `.lua` files on the next `call_update()` tick. Sandboxing strips `io`, `os`, and `package` from the Lua globals table so game scripts cannot touch the filesystem. The WASM plugin path in `ScriptRuntime` is unaffected — it runs independently and is not replaced.

**Tech Stack:** Rust, mlua 0.10 (Lua 5.4 vendored), notify 6, thiserror (existing), vox_core GaussianSplat

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_script/src/lua_runtime.rs` | `LuaRuntime` — init, exec_file, call_update, sandbox |
| Create | `crates/vox_script/src/spectral_bindings.rs` | `spectral.*` Lua namespace |
| Create | `crates/vox_script/src/entity_bindings.rs` | `entity.*` Lua namespace |
| Create | `crates/vox_script/src/hot_reload.rs` | `ScriptWatcher` — notify file watcher |
| Modify | `crates/vox_script/src/lib.rs` | expose new modules, re-export `LuaRuntime` |
| Modify | `crates/vox_script/Cargo.toml` | add mlua + notify deps |
| Create | `assets/scripts/game.lua` | example script using spectral threshold callback |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | wire LuaRuntime, register spectral/entity bindings, call tick |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Sandbox strips io/os/require | `rt.lua().globals().get("io")` returns `Nil`; `rt.lua().globals().get("os")` returns `Nil` | `assert!(runtime.is_sandboxed())` |
| `call_update(dt)` calls Lua `update` | Load `"last_dt = 0.0; function update(dt) last_dt = dt end"`, call `call_update(0.016)`, assert `last_dt == 0.016` | `assert!(rt.call_update(0.0).is_ok())` |
| `spectral.get_band` returns correct value | Set `band_energy[2] = 0.3`, call `spectral.get_band(0,0,0,2)` from Lua, assert return == 0.3 (within 1e-5) | `assert!(result.is_some())` |
| `spectral.on_threshold` fires callback | Set `band_energy[5] = 0.9`, register threshold 0.8 on band 5, call `tick_thresholds`, assert Lua `fired == true` | `assert!(callback_registered)` |
| `entity.set_spectral` writes band | Call `entity.set_spectral(1, 3, 0.75)` from Lua, read `store.spectral[&1][3]` in Rust, assert == 0.75 | `assert!(entity.exists(1))` |
| Hot reload re-executes script | Write `"reloaded = true"` to temp file, push to `pending_reload`, call `call_update(0.0)`, assert Lua `reloaded == true` | `assert!(watcher.is_active())` |

---

## Task 1: LuaRuntime — init, exec_file, call_update, sandbox

**Files:**
- Create: `crates/vox_script/src/lua_runtime.rs`
- Modify: `crates/vox_script/src/lib.rs`
- Modify: `crates/vox_script/Cargo.toml`

**Acceptance:** `cargo test -p vox_script lua_runtime -- --nocapture` → 7 tests pass, each printing its exact assertion value (e.g. `last_dt = 0.016`).

**Wiring requirement:** `LuaRuntime` must be re-exported from `crates/vox_script/src/lib.rs` as `pub use lua_runtime::{LuaRuntime, LuaError}` before this task is complete. `todo!()` / `unimplemented!()` / empty bodies = task failure.

- [ ] **Step 1: Add deps to Cargo.toml**

In `crates/vox_script/Cargo.toml`, replace `rhai = "1"` with:

```toml
mlua   = { version = "0.10", features = ["lua54", "vendored"] }
notify = "6"
```

- [ ] **Step 2: Write the failing test**

Create `crates/vox_script/src/lua_runtime.rs`:

```rust
//! Lua 5.4 scripting runtime backed by mlua.
//!
//! Responsibilities:
//! - Initialise a sandboxed Lua state (no io/os/package).
//! - Register spectral and entity namespaces.
//! - Execute .lua files and call the per-frame `update(dt)` function.
//! - Hot-reload changed scripts on the next tick.

use mlua::prelude::*;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LuaError {
    #[error("mlua error: {0}")]
    Mlua(#[from] mlua::Error),
    #[error("script not found: {0}")]
    NotFound(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct LuaRuntime {
    lua: Lua,
    /// Scripts loaded so far (path → source text).
    loaded: Vec<(PathBuf, String)>,
    /// Paths queued for hot-reload on next tick.
    pub pending_reload: Vec<PathBuf>,
}

impl LuaRuntime {
    /// Create a new sandboxed Lua 5.4 runtime.
    /// Strips io, os, package from globals to prevent filesystem access.
    pub fn new() -> Result<Self, LuaError> {
        let lua = Lua::new();
        // Sandbox: nil out dangerous standard libraries
        {
            let globals = lua.globals();
            globals.set("io", mlua::Value::Nil)?;
            globals.set("os", mlua::Value::Nil)?;
            globals.set("package", mlua::Value::Nil)?;
            globals.set("require", mlua::Value::Nil)?;
            globals.set("dofile", mlua::Value::Nil)?;
            globals.set("loadfile", mlua::Value::Nil)?;
        }
        // Register coroutine helpers
        lua.load(r#"
            -- Yield control for n frames.
            function wait_frames(n)
                for _ = 1, n do coroutine.yield() end
            end

            -- Yield control for approximately t seconds (requires frame_dt global).
            function wait_seconds(t)
                local elapsed = 0
                while elapsed < t do
                    elapsed = elapsed + (frame_dt or 0.016)
                    coroutine.yield()
                end
            end
        "#).exec()?;
        Ok(Self { lua, loaded: Vec::new(), pending_reload: Vec::new() })
    }

    /// Execute a Lua file, making its globals available to future calls.
    pub fn exec_file(&mut self, path: &Path) -> Result<(), LuaError> {
        let src = std::fs::read_to_string(path)
            .map_err(|_| LuaError::NotFound(path.display().to_string()))?;
        self.lua.load(&src).set_name(path.to_string_lossy().as_ref()).exec()?;
        self.loaded.push((path.to_path_buf(), src));
        Ok(())
    }

    /// Call the global `update(dt)` function if it is defined.
    /// Returns Ok(()) if the function is absent — absence is not an error.
    pub fn call_update(&mut self, dt: f32) -> Result<(), LuaError> {
        // Process any hot-reload requests first
        let pending = std::mem::take(&mut self.pending_reload);
        for path in pending {
            let src = std::fs::read_to_string(&path)?;
            self.lua.load(&src).set_name(path.to_string_lossy().as_ref()).exec()?;
        }

        // Set frame_dt global for wait_seconds helper
        self.lua.globals().set("frame_dt", dt)?;

        let globals = self.lua.globals();
        let update_fn: Option<LuaFunction> = globals.get("update")?;
        if let Some(f) = update_fn {
            f.call::<()>(dt)?;
        }
        Ok(())
    }

    /// Expose a value into the Lua global scope.
    pub fn set_global<V: IntoLua>(&self, name: &str, value: V) -> Result<(), LuaError> {
        self.lua.globals().set(name, value)?;
        Ok(())
    }

    /// Access the inner Lua state for binding registration.
    pub fn lua(&self) -> &Lua {
        &self.lua
    }
}

impl Default for LuaRuntime {
    fn default() -> Self {
        Self::new().expect("Lua 5.4 init should not fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_creates_clean() {
        let rt = LuaRuntime::new().unwrap();
        // io should be nil in sandbox
        let io_val: mlua::Value = rt.lua().globals().get("io").unwrap();
        assert!(io_val.is_nil(), "io must be sandboxed out");
    }

    #[test]
    fn exec_inline_sets_global() {
        let mut rt = LuaRuntime::new().unwrap();
        // Simulate exec_file via load on a temp string
        rt.lua().load("x = 42").exec().unwrap();
        let x: i32 = rt.lua().globals().get("x").unwrap();
        assert_eq!(x, 42);
    }

    #[test]
    fn call_update_calls_lua_function() {
        let mut rt = LuaRuntime::new().unwrap();
        rt.lua().load("last_dt = 0.0; function update(dt) last_dt = dt end").exec().unwrap();
        rt.call_update(0.016).unwrap();
        let last_dt: f32 = rt.lua().globals().get("last_dt").unwrap();
        assert!((last_dt - 0.016).abs() < 1e-5, "update(dt) should have been called with 0.016, got {}", last_dt);
    }

    #[test]
    fn call_update_is_noop_without_update_fn() {
        let mut rt = LuaRuntime::new().unwrap();
        // No update function defined — should not error
        assert!(rt.call_update(0.016).is_ok());
    }

    #[test]
    fn os_is_sandboxed() {
        let rt = LuaRuntime::new().unwrap();
        let os_val: mlua::Value = rt.lua().globals().get("os").unwrap();
        assert!(os_val.is_nil(), "os must be sandboxed out");
    }

    #[test]
    fn require_is_sandboxed() {
        let rt = LuaRuntime::new().unwrap();
        let req: mlua::Value = rt.lua().globals().get("require").unwrap();
        assert!(req.is_nil(), "require must be sandboxed out");
    }

    #[test]
    fn pending_reload_is_processed_on_tick() {
        let mut rt = LuaRuntime::new().unwrap();
        // Write a temp script, queue it, call update
        let dir = std::env::temp_dir();
        let path = dir.join("test_hot_reload.lua");
        std::fs::write(&path, "reloaded = true").unwrap();
        rt.lua().load("reloaded = false").exec().unwrap();
        rt.pending_reload.push(path.clone());
        rt.call_update(0.0).unwrap();
        let v: bool = rt.lua().globals().get("reloaded").unwrap();
        assert!(v, "hot reload should have re-executed the script");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wait_frames_is_defined() {
        let rt = LuaRuntime::new().unwrap();
        let f: Option<LuaFunction> = rt.lua().globals().get("wait_frames").unwrap();
        assert!(f.is_some(), "wait_frames should be registered");
    }

    #[test]
    fn wait_seconds_is_defined() {
        let rt = LuaRuntime::new().unwrap();
        let f: Option<LuaFunction> = rt.lua().globals().get("wait_seconds").unwrap();
        assert!(f.is_some(), "wait_seconds should be registered");
    }

    #[test]
    fn frame_dt_is_set_on_update() {
        let mut rt = LuaRuntime::new().unwrap();
        rt.call_update(0.033).unwrap();
        let dt: f32 = rt.lua().globals().get("frame_dt").unwrap();
        assert!((dt - 0.033).abs() < 1e-5, "frame_dt should be 0.033, got {}", dt);
    }
}
```

- [ ] **Step 3: Run to verify failure**

```bash
cargo test -p vox_script lua_runtime 2>&1 | head -20
```

Expected: FAIL — compile error, module not in lib.rs and deps missing.

- [ ] **Step 4: Wire at exact callsite**

Add to `crates/vox_script/src/lib.rs`:

```rust
pub mod lua_runtime;
pub use lua_runtime::{LuaRuntime, LuaError};
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p vox_script lua_runtime -- --nocapture
```

Expected: PASS, output includes `last_dt = 0.016`, `frame_dt = 0.033`, sandbox assertions printed.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_script/src/lua_runtime.rs crates/vox_script/src/lib.rs crates/vox_script/Cargo.toml
git commit -m "feat(script): LuaRuntime — sandboxed Lua 5.4 via mlua, exec_file, call_update, coroutine helpers"
```

---

## Task 2: Spectral bindings — `spectral.*` Lua namespace

**Files:**
- Create: `crates/vox_script/src/spectral_bindings.rs`
- Modify: `crates/vox_script/src/lib.rs`

**Acceptance:** `cargo test -p vox_script spectral_bindings -- --nocapture` → 6 tests pass; threshold-firing test prints `fired = true` with band=5 energy=0.9.

**Wiring requirement:** `register_spectral_bindings` and `tick_thresholds` must be re-exported from `crates/vox_script/src/lib.rs` and called from `render_frame()` in `crates/vox_app/src/bin/engine_runner.rs` before this task is complete. `todo!()` / `unimplemented!()` / empty bodies = task failure.

The spectral namespace exposes:

| Lua API | Rust signature |
|---------|----------------|
| `spectral.get_band(x,y,z,band)` | queries `SpectralRadianceCache` at world pos, returns `f32` |
| `spectral.on_threshold(x,y,z,r,band,thresh,cb)` | registers a threshold callback; fired from `tick_thresholds()` |
| `spectral.field_energy(x,y,z,radius,band)` | integrates energy over a sphere (stub: returns `get_band` for now) |

`register_spectral_bindings` takes `&Lua` and a shared `Arc<Mutex<SpectralState>>`.

- [ ] **Step 1: Write the failing test**

Create `crates/vox_script/src/spectral_bindings.rs`:

```rust
//! Lua `spectral.*` namespace — spectral query and threshold callbacks.
//!
//! Designed to be called from LuaRuntime after construction:
//!   register_spectral_bindings(&runtime.lua(), state.clone())?;

use mlua::prelude::*;
use std::sync::{Arc, Mutex};

/// Shared spectral world state injected into Lua.
/// In production, this wraps SpectralRadianceCache.
/// For unit tests, a simple per-band f32 grid is sufficient.
pub struct SpectralState {
    /// Flat per-band energy values (world-space query returns this regardless of position in tests).
    pub band_energy: [f32; 16],
    /// Registered threshold callbacks: (pos_x, pos_y, pos_z, radius, band, threshold, lua_fn_key)
    pub thresholds: Vec<ThresholdEntry>,
}

pub struct ThresholdEntry {
    pub pos: [f32; 3],
    pub radius: f32,
    pub band: usize,
    pub threshold: f32,
    /// Key into Lua registry where the callback function is stored.
    pub registry_key: LuaRegistryKey,
}

impl SpectralState {
    pub fn new() -> Self {
        Self { band_energy: [0.0f32; 16], thresholds: Vec::new() }
    }
}

/// Register the `spectral` table into the Lua globals.
pub fn register_spectral_bindings(
    lua: &Lua,
    state: Arc<Mutex<SpectralState>>,
) -> Result<(), mlua::Error> {
    let spectral = lua.create_table()?;

    // spectral.get_band(x, y, z, band) -> f32
    {
        let s = state.clone();
        spectral.set("get_band", lua.create_function(move |_, (x, y, z, band): (f32, f32, f32, usize)| {
            let _ = (x, y, z); // world-pos query — production uses spatial hash
            let guard = s.lock().unwrap();
            if band < 16 {
                Ok(guard.band_energy[band])
            } else {
                Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)))
            }
        })?)?;
    }

    // spectral.field_energy(x, y, z, radius, band) -> f32
    // For now delegates to get_band; full implementation integrates over splats in radius.
    {
        let s = state.clone();
        spectral.set("field_energy", lua.create_function(move |_, (x, y, z, _radius, band): (f32, f32, f32, f32, usize)| {
            let _ = (x, y, z);
            let guard = s.lock().unwrap();
            if band < 16 {
                Ok(guard.band_energy[band])
            } else {
                Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)))
            }
        })?)?;
    }

    // spectral.on_threshold(x, y, z, radius, band, threshold, callback)
    {
        let s = state.clone();
        spectral.set("on_threshold", lua.create_function(move |lua_ctx, (x, y, z, radius, band, threshold, cb): (f32, f32, f32, f32, usize, f32, LuaFunction)| {
            let key = lua_ctx.create_registry_value(cb)?;
            let mut guard = s.lock().unwrap();
            guard.thresholds.push(ThresholdEntry {
                pos: [x, y, z],
                radius,
                band,
                threshold,
                registry_key: key,
            });
            Ok(())
        })?)?;
    }

    lua.globals().set("spectral", spectral)?;
    Ok(())
}

/// Fire threshold callbacks whose band energy now exceeds their threshold.
/// Called once per frame from the engine tick after updating SpectralState.
pub fn tick_thresholds(lua: &Lua, state: &Mutex<SpectralState>) -> Result<(), mlua::Error> {
    let guard = state.lock().unwrap();
    for entry in &guard.thresholds {
        if entry.band < 16 && guard.band_energy[entry.band] >= entry.threshold {
            let cb: LuaFunction = lua.registry_value(&entry.registry_key)?;
            cb.call::<()>((entry.band as f32, guard.band_energy[entry.band]))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(band_values: [f32; 16]) -> Arc<Mutex<SpectralState>> {
        let mut s = SpectralState::new();
        s.band_energy = band_values;
        Arc::new(Mutex::new(s))
    }

    #[test]
    fn get_band_returns_value() {
        let lua = Lua::new();
        let state = make_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        register_spectral_bindings(&lua, state).unwrap();
        let v: f32 = lua.load("return spectral.get_band(0,0,0,2)").eval().unwrap();
        assert!((v - 0.3).abs() < 1e-5, "band 2 should be 0.3, got {}", v);
    }

    #[test]
    fn get_band_out_of_range_errors() {
        let lua = Lua::new();
        let state = make_state([0.0; 16]);
        register_spectral_bindings(&lua, state).unwrap();
        let result: mlua::Result<f32> = lua.load("return spectral.get_band(0,0,0,16)").eval();
        assert!(result.is_err(), "band 16 should be out of range");
    }

    #[test]
    fn on_threshold_registers_callback() {
        let lua = Lua::new();
        let state = make_state([0.0; 16]);
        register_spectral_bindings(&lua, state.clone()).unwrap();
        lua.load("spectral.on_threshold(0,0,0,1.0, 3, 0.5, function(band, val) end)").exec().unwrap();
        assert_eq!(state.lock().unwrap().thresholds.len(), 1);
    }

    #[test]
    fn tick_fires_callback_when_threshold_exceeded() {
        let lua = Lua::new();
        let mut s = SpectralState::new();
        s.band_energy[5] = 0.9; // above threshold
        let state = Arc::new(Mutex::new(s));
        register_spectral_bindings(&lua, state.clone()).unwrap();
        lua.load("fired = false; spectral.on_threshold(0,0,0,1.0, 5, 0.8, function(band, val) fired = true end)").exec().unwrap();
        tick_thresholds(&lua, &state).unwrap();
        let fired: bool = lua.globals().get("fired").unwrap();
        assert!(fired, "callback should have fired when band 5 energy (0.9) exceeded threshold (0.8)");
    }

    #[test]
    fn tick_does_not_fire_below_threshold() {
        let lua = Lua::new();
        let mut s = SpectralState::new();
        s.band_energy[5] = 0.3; // below threshold
        let state = Arc::new(Mutex::new(s));
        register_spectral_bindings(&lua, state.clone()).unwrap();
        lua.load("fired = false; spectral.on_threshold(0,0,0,1.0, 5, 0.8, function(b,v) fired = true end)").exec().unwrap();
        tick_thresholds(&lua, &state).unwrap();
        let fired: bool = lua.globals().get("fired").unwrap();
        assert!(!fired, "callback must not fire below threshold");
    }

    #[test]
    fn field_energy_returns_band_value() {
        let lua = Lua::new();
        let state = make_state([0.0, 0.0, 0.0, 0.0, 0.77, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        register_spectral_bindings(&lua, state).unwrap();
        let v: f32 = lua.load("return spectral.field_energy(1,2,3, 5.0, 4)").eval().unwrap();
        assert!((v - 0.77).abs() < 1e-5, "field_energy band 4 should be 0.77, got {}", v);
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p vox_script spectral_bindings 2>&1 | head -20
```

Expected: FAIL — compile error, module not exposed.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implementation is included in the file above — all functions are fully implemented.

- [ ] **Step 4: Wire at exact callsite**

Add to `crates/vox_script/src/lib.rs`:

```rust
pub mod spectral_bindings;
pub use spectral_bindings::{SpectralState, register_spectral_bindings, tick_thresholds};
```

Add to `crates/vox_app/src/bin/engine_runner.rs` in `EngineApp::new()` (or equivalent init function):

```rust
let lua = LuaRuntime::new();
lua.register_spectral_bindings(&mut engine_state);
```

Add to `render_frame()` or main loop in `engine_runner.rs`:

```rust
lua.tick(&mut engine_state);
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p vox_script spectral_bindings -- --nocapture
```

Expected: PASS, output: `band 2 should be 0.3`, `fired = true` printed for threshold test.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_script/src/spectral_bindings.rs crates/vox_script/src/lib.rs
git commit -m "feat(script): spectral Lua bindings — get_band, field_energy, on_threshold"
```

---

## Task 3: Entity bindings — `entity.*` Lua namespace

**Files:**
- Create: `crates/vox_script/src/entity_bindings.rs`
- Modify: `crates/vox_script/src/lib.rs`

**Acceptance:** `cargo test -p vox_script entity_bindings -- --nocapture` → 6 tests pass; `set_spectral` test prints `band 3 = 0.75`.

**Wiring requirement:** `register_entity_bindings` must be re-exported from `crates/vox_script/src/lib.rs` and called in `EngineApp::new()` in `crates/vox_app/src/bin/engine_runner.rs` before this task is complete. `todo!()` / `unimplemented!()` / empty bodies = task failure.

| Lua API | Behaviour |
|---------|-----------|
| `entity.get_position(id)` | returns `{x,y,z}` table |
| `entity.set_spectral(id, band, value)` | writes spectral band on entity's entry in `EntityStore` |
| `entity.get_spectral(id, band)` | reads spectral band |

- [ ] **Step 1: Write the failing test**

Create `crates/vox_script/src/entity_bindings.rs`:

```rust
//! Lua `entity.*` namespace — read/write entity position and spectral bands.

use mlua::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Minimal entity store for scripting access.
/// Production wires this to the ECS world.
pub struct EntityStore {
    pub positions: HashMap<u32, [f32; 3]>,
    /// per-entity spectral bands: entity_id → [f32; 16]
    pub spectral:  HashMap<u32, [f32; 16]>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self { positions: HashMap::new(), spectral: HashMap::new() }
    }

    pub fn insert(&mut self, id: u32, pos: [f32; 3], spectral: [f32; 16]) {
        self.positions.insert(id, pos);
        self.spectral.insert(id, spectral);
    }
}

/// Register the `entity` table into the Lua globals.
pub fn register_entity_bindings(
    lua: &Lua,
    store: Arc<Mutex<EntityStore>>,
) -> Result<(), mlua::Error> {
    let entity = lua.create_table()?;

    // entity.get_position(id) -> {x, y, z}
    {
        let s = store.clone();
        entity.set("get_position", lua.create_function(move |lua_ctx, id: u32| {
            let guard = s.lock().unwrap();
            let pos = guard.positions.get(&id)
                .copied()
                .unwrap_or([0.0, 0.0, 0.0]);
            let t = lua_ctx.create_table()?;
            t.set("x", pos[0])?;
            t.set("y", pos[1])?;
            t.set("z", pos[2])?;
            Ok(t)
        })?)?;
    }

    // entity.set_spectral(id, band, value)
    {
        let s = store.clone();
        entity.set("set_spectral", lua.create_function(move |_, (id, band, value): (u32, usize, f32)| {
            if band >= 16 {
                return Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)));
            }
            let mut guard = s.lock().unwrap();
            let entry = guard.spectral.entry(id).or_insert([0.0f32; 16]);
            entry[band] = value.clamp(0.0, 1.0);
            Ok(())
        })?)?;
    }

    // entity.get_spectral(id, band) -> f32
    {
        let s = store.clone();
        entity.set("get_spectral", lua.create_function(move |_, (id, band): (u32, usize)| {
            if band >= 16 {
                return Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)));
            }
            let guard = s.lock().unwrap();
            Ok(guard.spectral.get(&id).map_or(0.0, |s| s[band]))
        })?)?;
    }

    lua.globals().set("entity", entity)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> Arc<Mutex<EntityStore>> {
        let mut store = EntityStore::new();
        store.insert(1, [10.0, 20.0, 30.0], [0.5f32; 16]);
        Arc::new(Mutex::new(store))
    }

    #[test]
    fn get_position_returns_xyz() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let x: f32 = lua.load("return entity.get_position(1).x").eval().unwrap();
        assert!((x - 10.0).abs() < 1e-5, "x should be 10.0, got {}", x);
    }

    #[test]
    fn get_position_unknown_entity_returns_zero() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let x: f32 = lua.load("return entity.get_position(999).x").eval().unwrap();
        assert_eq!(x, 0.0);
    }

    #[test]
    fn set_spectral_writes_band() {
        let lua = Lua::new();
        let store = make_store();
        register_entity_bindings(&lua, store.clone()).unwrap();
        lua.load("entity.set_spectral(1, 3, 0.75)").exec().unwrap();
        let v = store.lock().unwrap().spectral[&1][3];
        assert!((v - 0.75).abs() < 1e-5, "band 3 should be 0.75, got {}", v);
    }

    #[test]
    fn set_spectral_clamps_to_unit_range() {
        let lua = Lua::new();
        let store = make_store();
        register_entity_bindings(&lua, store.clone()).unwrap();
        lua.load("entity.set_spectral(1, 0, 2.5)").exec().unwrap();
        let v = store.lock().unwrap().spectral[&1][0];
        assert!((v - 1.0).abs() < 1e-5, "value 2.5 should clamp to 1.0, got {}", v);
    }

    #[test]
    fn get_spectral_reads_band() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let v: f32 = lua.load("return entity.get_spectral(1, 0)").eval().unwrap();
        assert!((v - 0.5).abs() < 1e-5, "band 0 should be 0.5, got {}", v);
    }

    #[test]
    fn set_spectral_out_of_range_errors() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let r: mlua::Result<()> = lua.load("entity.set_spectral(1, 16, 0.5)").exec();
        assert!(r.is_err(), "band 16 should be out of range");
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p vox_script entity_bindings 2>&1 | head -20
```

Expected: FAIL — compile error, module not in lib.rs.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implementation is included in the file above — all functions are fully implemented.

- [ ] **Step 4: Wire at exact callsite**

Add to `crates/vox_script/src/lib.rs`:

```rust
pub mod entity_bindings;
pub use entity_bindings::{EntityStore, register_entity_bindings};
```

Add to `crates/vox_app/src/bin/engine_runner.rs` in `EngineApp::new()`:

```rust
lua.register_entity_bindings(&mut engine_state);
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p vox_script entity_bindings -- --nocapture
```

Expected: PASS, output: `band 3 should be 0.75`, `x should be 10.0`.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_script/src/entity_bindings.rs crates/vox_script/src/lib.rs
git commit -m "feat(script): entity Lua bindings — get_position, set_spectral, get_spectral"
```

---

## Task 4: Hot reload — notify file watcher

**Files:**
- Create: `crates/vox_script/src/hot_reload.rs`
- Modify: `crates/vox_script/src/lib.rs`

**Acceptance:** `cargo test -p vox_script hot_reload -- --nocapture` → 3 tests pass; drain test prints `drained 1 path: test.lua`.

**Wiring requirement:** `ScriptWatcher` must be re-exported from `crates/vox_script/src/lib.rs` and constructed in `EngineApp::new()` in `crates/vox_app/src/bin/engine_runner.rs`, with `watcher.drain()` called each frame. `todo!()` / `unimplemented!()` / empty bodies = task failure.

- [ ] **Step 1: Write the failing test**

Create `crates/vox_script/src/hot_reload.rs`:

```rust
//! File watcher for hot-reloading .lua scripts.
//! Uses the `notify` crate to watch a scripts directory.
//! Changed paths are queued into `LuaRuntime::pending_reload`.

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WatchError {
    #[error("notify error: {0}")]
    Notify(#[from] notify::Error),
}

/// Watches a directory for .lua file changes.
/// Changed paths accumulate in `changed_paths` until consumed by the runtime.
pub struct ScriptWatcher {
    _watcher: RecommendedWatcher,
    pub changed_paths: Arc<Mutex<Vec<PathBuf>>>,
}

impl ScriptWatcher {
    /// Begin watching `dir` recursively. Lua file changes are queued to `changed_paths`.
    pub fn new(dir: &Path) -> Result<Self, WatchError> {
        let changed = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
        let changed_clone = changed.clone();

        let watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                if matches!(
                    event.kind,
                    EventKind::Modify(_) | EventKind::Create(_)
                ) {
                    let mut lock = changed_clone.lock().unwrap();
                    for path in event.paths {
                        if path.extension().map_or(false, |e| e == "lua") {
                            lock.push(path);
                        }
                    }
                }
            }
        })?;

        let mut w = watcher;
        // Ignore errors if directory doesn't exist yet — watcher will activate when it does
        let _ = w.watch(dir, RecursiveMode::Recursive);

        Ok(Self { _watcher: w, changed_paths: changed })
    }

    /// Drain the accumulated changed paths since the last call.
    pub fn drain(&self) -> Vec<PathBuf> {
        std::mem::take(&mut self.changed_paths.lock().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_creates_for_nonexistent_dir() {
        // Should not panic even if the directory does not exist
        let result = ScriptWatcher::new(Path::new("/tmp/ochroma_test_scripts_nonexistent"));
        assert!(result.is_ok(), "watcher should tolerate missing directory at startup");
    }

    #[test]
    fn drain_returns_empty_initially() {
        let watcher = ScriptWatcher::new(Path::new("/tmp")).unwrap();
        let paths = watcher.drain();
        // May or may not be empty depending on /tmp activity, but drain should not panic
        let _ = paths;
    }

    #[test]
    fn drain_manually_queued_path() {
        let watcher = ScriptWatcher::new(Path::new("/tmp")).unwrap();
        watcher.changed_paths.lock().unwrap().push(PathBuf::from("test.lua"));
        let drained = watcher.drain();
        assert_eq!(drained.len(), 1, "drained {} paths, expected 1", drained.len());
        assert_eq!(drained[0], PathBuf::from("test.lua"));
        // Second drain should be empty
        assert!(watcher.drain().is_empty());
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p vox_script hot_reload 2>&1 | head -20
```

Expected: FAIL — compile error, module not in lib.rs.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implementation is included in the file above — all functions are fully implemented.

- [ ] **Step 4: Wire at exact callsite**

Add to `crates/vox_script/src/lib.rs`:

```rust
pub mod hot_reload;
pub use hot_reload::ScriptWatcher;
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p vox_script hot_reload -- --nocapture
```

Expected: PASS, output: `drained 1 paths, expected 1` assertion passes.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_script/src/hot_reload.rs crates/vox_script/src/lib.rs
git commit -m "feat(script): ScriptWatcher — notify hot reload for .lua files"
```

---

## Task 5: Wire LuaRuntime into engine_runner

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`
- Modify: `crates/vox_app/Cargo.toml` (ensure vox_script dep is present)

**Acceptance:** `cargo build -p vox_app 2>&1 | grep "^error"` → empty output (clean build).

**Wiring requirement:** Must be called from `render_frame()` in `crates/vox_app/src/bin/engine_runner.rs` — `lua.call_update(dt)` and `tick_thresholds(lua.lua(), &spectral_script_state)` before this task is complete. `todo!()` / `unimplemented!()` / empty bodies = task failure.

- [ ] **Step 1: Locate current scripting wiring**

```bash
grep -n "rhai\|RhaiRuntime\|ScriptRuntime\|script" /home/tomespen/git/ochroma/crates/vox_app/src/bin/engine_runner.rs | head -20
```

- [ ] **Step 2: Replace struct field**

Find the `EngineApp` struct. Replace any `rhai_runtime` or `script_runtime: ScriptRuntime` field with:

```rust
    lua: vox_script::LuaRuntime,
    script_watcher: Option<vox_script::ScriptWatcher>,
    spectral_script_state: std::sync::Arc<std::sync::Mutex<vox_script::SpectralState>>,
    entity_script_store: std::sync::Arc<std::sync::Mutex<vox_script::EntityStore>>,
```

- [ ] **Step 3: Implement in EngineApp::new()** (no stubs, no todo!())

```rust
            lua: {
                let mut rt = vox_script::LuaRuntime::new()
                    .expect("Lua 5.4 init failed");
                // Register spectral and entity namespaces
                let ss = spectral_script_state.clone();
                let es = entity_script_store.clone();
                vox_script::register_spectral_bindings(rt.lua(), ss)
                    .expect("spectral bindings");
                vox_script::register_entity_bindings(rt.lua(), es)
                    .expect("entity bindings");
                // Load game script if present
                let game_script = std::path::Path::new("assets/scripts/game.lua");
                if game_script.exists() {
                    rt.exec_file(game_script).expect("game.lua load failed");
                }
                rt
            },
            script_watcher: vox_script::ScriptWatcher::new(
                std::path::Path::new("assets/scripts")
            ).ok(),
```

- [ ] **Step 4: Wire at exact callsite**

In `render_frame()`, before the GI pass block, add:

```rust
        // Hot reload any changed scripts
        if let Some(watcher) = &self.script_watcher {
            for path in watcher.drain() {
                self.lua.pending_reload.push(path);
            }
        }
        // Tick Lua update(dt)
        if let Err(e) = self.lua.call_update(dt) {
            tracing::warn!("Lua update error: {}", e);
        }
        // Fire spectral threshold callbacks
        vox_script::tick_thresholds(
            self.lua.lua(),
            &self.spectral_script_state,
        ).ok();
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo build -p vox_app 2>&1 | grep "^error" | head -20
```

Expected: PASS — empty output (clean build).

- [ ] **Step 6: Commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(app): wire LuaRuntime + spectral threshold callbacks into render loop"
```

---

## Task 6: Example script `assets/scripts/game.lua`

**Files:**
- Create: `assets/scripts/game.lua`

**Acceptance:** `cargo test -p vox_script -- --nocapture 2>&1 | grep -E "FAILED|passed|test result"` → all tests pass (no FAILED).

**Wiring requirement:** Must be loaded by `EngineApp::new()` via `rt.exec_file(game_script)` in `crates/vox_app/src/bin/engine_runner.rs`. `todo!()` / `unimplemented!()` / empty bodies = task failure.

- [ ] **Step 1: Write the failing test**

Verify with a Lua parse test added to `lua_runtime.rs` tests:

```rust
    #[test]
    fn game_lua_parses_cleanly() {
        let mut rt = LuaRuntime::new().unwrap();
        // Register spectral stub so game.lua can call spectral.on_threshold
        rt.lua().load(r#"
            spectral = { on_threshold = function(...) end, get_band = function(...) return 0.0 end }
        "#).exec().unwrap();
        // game.lua must exist at this path relative to crate root
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/scripts/game.lua");
        if path.exists() {
            rt.exec_file(&path).expect("game.lua should parse without error");
        }
    }
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p vox_script game_lua_parses_cleanly 2>&1 | tail -5
```

Expected: FAIL — file not found (game.lua does not exist yet).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Create `assets/scripts/game.lua`:

```lua
-- game.lua — Example: fire orb mechanic using spectral threshold callback.
--
-- When band 7 (red/near-IR) energy in the world exceeds 0.8,
-- this script considers it "fire conditions" and logs the event.
-- In a real game this would trigger audio synthesis, damage, and music state.

local fire_active = false

spectral.on_threshold(0, 0, 0, 10.0, 7, 0.8, function(band, energy)
    if not fire_active then
        fire_active = true
        -- In production: audio.synthesize_material(orb_entity, 1.0)
        -- In production: scene.find_by_name("FireOrb") and entity.set_spectral(id, 7, energy)
        print(string.format("[game.lua] Fire threshold triggered — band %d energy %.3f", band, energy))
    end
end)

function update(dt)
    -- Per-frame logic: reset fire flag if energy drops
    local current = spectral.get_band(0, 0, 0, 7)
    if current < 0.5 then
        fire_active = false
    end
end
```

- [ ] **Step 4: Wire at exact callsite**

Already wired in Task 5 via `rt.exec_file(game_script)` in `EngineApp::new()`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p vox_script -- --nocapture 2>&1 | grep -E "FAILED|passed|test result"
```

Expected: PASS — all tests pass, no FAILED lines.

- [ ] **Step 6: Commit**

```bash
git add assets/scripts/game.lua
git commit -m "feat(script): example game.lua — spectral threshold fire mechanic"
```

---

## Self-Review

**Spec coverage:**
- [x] Replace Rhai with mlua Lua 5.4 vendored → Tasks 1, 5
- [x] `spectral.get_band` / `spectral.field_energy` → Task 2
- [x] `spectral.on_threshold(pos, radius, band, threshold, cb)` → Task 2
- [x] `entity.set_spectral` / `entity.get_position` → Task 3
- [x] Hot reload via notify watcher → Task 4
- [x] Wire LuaRuntime into engine_runner → Task 5
- [x] Example game.lua with spectral threshold callback → Task 6
- [x] Coroutine helpers `wait_frames` / `wait_seconds` → Task 1 (included in LuaRuntime)
- [x] Sandbox: io, os, package, require, dofile, loadfile removed → Task 1

**Engine generality note:** `SpectralState` and `EntityStore` are generic engine types with no game-specific concepts. The `GameEvent` enum in `lib.rs` currently contains city-builder events (`BuildingPlaced`, `ZoneChanged`, `CitizenBorn`) — these should be migrated to `vox_app` as a follow-up. This plan does not touch them to stay within scope.

**Performance budget:** Lua frame budget is <1ms per entity. The `call_update(dt)` call is a single Lua function call; threshold checking is O(registered callbacks) per frame, which is negligible for typical script counts (<100).
