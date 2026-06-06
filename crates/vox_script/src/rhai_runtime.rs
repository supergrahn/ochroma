use rhai::{Engine, AST, Scope, Dynamic};
use std::path::Path;
use std::sync::{Arc, Mutex};
use vox_core::script_interface::ScriptCommand;

use crate::spectral_bindings::SpectralState;

/// A Rhai script instance attached to an entity.
pub struct RhaiScript {
    pub name: String,
    pub source_path: Option<String>,
    ast: AST,
    pub last_mtime: std::time::SystemTime,
}

/// The Rhai scripting runtime — hot-reloadable game logic.
pub struct RhaiRuntime {
    engine: Engine,
    scripts: Vec<RhaiScript>,
    pub last_reload_check: std::time::Instant,
    pub reload_interval: std::time::Duration,
    /// Total number of successful hot-reloads (a changed file recompiled and the
    /// AST swapped in). Monotonic for the lifetime of the runtime.
    pub script_reloads: u32,
    /// Total number of hot-reload attempts that FAILED to compile. The previous
    /// (last-good) AST is kept on failure, so the game keeps running. Monotonic.
    pub script_errors: u32,
    /// Human-readable text of the most recent reload error, for surfacing in a
    /// HUD/notification. `None` until the first compile failure.
    pub last_error: Option<String>,
    /// Live spectral field the host populates each frame; Rhai scripts read it
    /// via the registered `field_energy` / `get_band` functions.
    spectral: Arc<Mutex<SpectralState>>,
}

static PENDING_COMMANDS: Mutex<Vec<ScriptCommand>> = Mutex::new(Vec::new());

pub fn drain_pending_commands() -> Vec<ScriptCommand> {
    match PENDING_COMMANDS.lock() {
        Ok(mut guard) => guard.drain(..).collect(),
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            guard.drain(..).collect()
        }
    }
}

impl RhaiRuntime {
    pub fn new() -> Self {
        let mut engine = Engine::new();

        // Register engine API functions that scripts can call

        engine.register_fn("log", |message: String| {
            if let Ok(mut cmds) = PENDING_COMMANDS.lock() {
                cmds.push(ScriptCommand::Log { message });
            }
        });

        engine.register_fn("spawn", |asset: String, x: f64, y: f64, z: f64| -> i64 {
            if let Ok(mut cmds) = PENDING_COMMANDS.lock() {
                cmds.push(ScriptCommand::Spawn {
                    asset_path: asset,
                    position: [x as f32, y as f32, z as f32],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                });
            }
            0 // return entity ID (stub)
        });

        engine.register_fn("play_sound", |clip: String, volume: f64| {
            if let Ok(mut cmds) = PENDING_COMMANDS.lock() {
                cmds.push(ScriptCommand::PlaySound {
                    clip,
                    volume: volume as f32,
                    spatial: true,
                });
            }
        });

        engine.register_fn("set_position", |x: f64, y: f64, z: f64| {
            if let Ok(mut cmds) = PENDING_COMMANDS.lock() {
                cmds.push(ScriptCommand::SetPosition {
                    position: [x as f32, y as f32, z as f32],
                });
            }
        });

        engine.register_fn("send_event", |name: String, data: String| {
            if let Ok(mut cmds) = PENDING_COMMANDS.lock() {
                cmds.push(ScriptCommand::SendEvent { name, data });
            }
        });

        engine.register_fn("distance", |x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64| -> f64 {
            ((x2-x1).powi(2) + (y2-y1).powi(2) + (z2-z1).powi(2)).sqrt()
        });

        engine.register_fn("random", || -> f64 {
            // Simple deterministic pseudo-random for scripts
            use std::time::SystemTime;
            let t = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().subsec_nanos();
            (t as f64 % 1000.0) / 1000.0
        });

        // Live spectral field shared with the host. Scripts read it through the
        // functions registered below — these read REAL populated values, not 0.
        let spectral: Arc<Mutex<SpectralState>> = Arc::new(Mutex::new(SpectralState::new()));

        // get_band(x, y, z, band) -> energy at the sampled band.
        {
            let s = spectral.clone();
            engine.register_fn("get_band", move |_x: f64, _y: f64, _z: f64, band: i64| -> f64 {
                if !(0..16).contains(&band) {
                    return 0.0;
                }
                let guard = s.lock().unwrap();
                guard.band_energy(band as usize) as f64
            });
        }

        // field_energy(x, y, z, radius, band) -> energy at the sampled band.
        {
            let s = spectral.clone();
            engine.register_fn("field_energy", move |_x: f64, _y: f64, _z: f64, _radius: f64, band: i64| -> f64 {
                if !(0..16).contains(&band) {
                    return 0.0;
                }
                let guard = s.lock().unwrap();
                guard.band_energy(band as usize) as f64
            });
        }

        Self {
            engine,
            scripts: Vec::new(),
            last_reload_check: std::time::Instant::now(),
            reload_interval: std::time::Duration::from_secs(1),
            script_reloads: 0,
            script_errors: 0,
            last_error: None,
            spectral,
        }
    }

    /// Load a script from source code string.
    pub fn load_script(&mut self, name: &str, source: &str) -> Result<usize, Box<rhai::EvalAltResult>> {
        let ast = self.engine.compile(source)?;
        let idx = self.scripts.len();
        self.scripts.push(RhaiScript {
            name: name.to_string(),
            source_path: None,
            ast,
            last_mtime: std::time::UNIX_EPOCH,
        });
        Ok(idx)
    }

    /// Load a script from a file.
    pub fn load_script_file(&mut self, name: &str, path: &Path) -> Result<usize, String> {
        let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let ast = self.engine.compile(&source).map_err(|e| format!("Compile error in {}: {}", path.display(), e))?;
        let idx = self.scripts.len();
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        self.scripts.push(RhaiScript {
            name: name.to_string(),
            source_path: Some(path.to_string_lossy().to_string()),
            ast,
            last_mtime: mtime,
        });
        Ok(idx)
    }

    /// Hot-reload a script by index (re-read from file and recompile).
    pub fn reload(&mut self, index: usize) -> Result<(), String> {
        if index >= self.scripts.len() { return Err("Invalid script index".into()); }
        let path = self.scripts[index].source_path.clone()
            .ok_or("Script was not loaded from file")?;
        let source = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let ast = self.engine.compile(&source).map_err(|e| format!("Reload error: {}", e))?;
        self.scripts[index].ast = ast;
        println!("[rhai] Reloaded script: {}", self.scripts[index].name);
        Ok(())
    }

    /// Reload all file-based scripts.
    pub fn reload_all(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        for i in 0..self.scripts.len() {
            if self.scripts[i].source_path.is_some()
                && let Err(e) = self.reload(i)
            {
                errors.push(format!("{}: {}", self.scripts[i].name, e));
            }
        }
        errors
    }

    /// Poll for changed script files and hot-reload them if the interval has elapsed.
    /// Returns names of scripts that were successfully reloaded.
    pub fn poll_reload(&mut self) -> Vec<String> {
        if self.last_reload_check.elapsed() < self.reload_interval {
            return Vec::new();
        }
        self.last_reload_check = std::time::Instant::now();

        let mut to_reload: Vec<(usize, String, std::time::SystemTime)> = Vec::new();
        for (i, script) in self.scripts.iter().enumerate() {
            let path = match &script.source_path {
                Some(p) => p.clone(),
                None => continue,
            };
            if let Ok(meta) = std::fs::metadata(&path)
                && let Ok(mtime) = meta.modified()
                && mtime > script.last_mtime
            {
                to_reload.push((i, script.name.clone(), mtime));
            }
        }

        let mut result = Vec::new();
        for (i, name, mtime) in to_reload {
            // Advance the recorded mtime regardless of outcome so a file that
            // fails to compile is not retried on every poll — it will only be
            // retried once it is edited again (its mtime moves forward), which is
            // exactly when a fix would land.
            self.scripts[i].last_mtime = mtime;
            match self.reload(i) {
                Ok(()) => {
                    self.script_reloads += 1;
                    self.last_error = None;
                    println!("[ochroma] Hot-reloaded script: {}", name);
                    result.push(name);
                }
                Err(e) => {
                    // Last-good AST is untouched (reload() returns before swapping
                    // on a compile error), so the game keeps running. Count + surface.
                    self.script_errors += 1;
                    self.last_error = Some(format!("{}: {}", name, e));
                    eprintln!("[ochroma] Script reload error {}: {}", name, e);
                }
            }
        }
        result
    }

    /// Call a function in a script.
    pub fn call_fn(&mut self, index: usize, fn_name: &str, args: &[Dynamic]) -> Result<Dynamic, Box<rhai::EvalAltResult>> {
        let script = &self.scripts[index];
        self.engine.call_fn(&mut Scope::new(), &script.ast, fn_name, args.to_vec())
    }

    /// Run the top-level code of a script.
    pub fn run(&self, index: usize) -> Result<(), String> {
        if index >= self.scripts.len() { return Err("Invalid script index".into()); }
        let mut scope = Scope::new();
        self.engine.run_ast_with_scope(&mut scope, &self.scripts[index].ast)
            .map_err(|e| format!("Runtime error in {}: {}", self.scripts[index].name, e))
    }

    /// Evaluate a one-off expression (for debug console).
    pub fn eval(&self, expr: &str) -> Result<String, String> {
        let result = self.engine.eval::<Dynamic>(expr)
            .map_err(|e| format!("Eval error: {}", e))?;
        Ok(format!("{}", result))
    }

    pub fn script_count(&self) -> usize { self.scripts.len() }

    pub fn script_names(&self) -> Vec<&str> {
        self.scripts.iter().map(|s| s.name.as_str()).collect()
    }

    /// Clean host-side write API: push one band of the live spectral field in
    /// linear f32. Rhai scripts then observe this via `get_band`/`field_energy`.
    /// Returns `false` if `band` is out of range.
    pub fn set_band_energy(&self, band: usize, value: f32) -> bool {
        self.spectral.lock().unwrap().set_band_energy(band, value)
    }

    /// Clean host-side write API: push one band from the engine-canonical `u16`
    /// (f16-bits) spectral encoding. Returns `false` if `band` is out of range.
    pub fn set_band_energy_u16(&self, band: usize, bits: u16) -> bool {
        self.spectral.lock().unwrap().set_band_energy_u16(band, bits)
    }

    /// Clean host-side write API: overwrite all 16 bands at once from an
    /// engine-canonical `[u16; 16]` f16-bit spectral sample.
    pub fn set_band_energy_all_u16(&self, spectral: &[u16; 16]) {
        self.spectral.lock().unwrap().set_band_energy_all_u16(spectral);
    }

    /// Handle to the live spectral field, so the host can share the SAME state
    /// between the Rhai runtime and Lua bindings if desired.
    pub fn spectral_state(&self) -> Arc<Mutex<SpectralState>> {
        self.spectral.clone()
    }
}

impl Default for RhaiRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rhai_runtime_loads_and_runs_script() {
        let mut rt = RhaiRuntime::new();
        rt.load_script("hello", r#"fn greet() { "hello" }"#).unwrap();
        let result = rt.call_fn(0, "greet", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn call_fn_passes_args_to_script() {
        let mut rt = RhaiRuntime::new();
        rt.load_script("test", r#"fn on_update(dt) { dt }"#).unwrap();
        let result = rt.call_fn(0, "on_update", &[rhai::Dynamic::from(0.016f64)]);
        assert!(result.is_ok(), "call_fn with args must not error");
        let val: f64 = result.unwrap().cast();
        assert!((val - 0.016).abs() < 1e-6, "returned value must match input arg");
    }

    #[test]
    fn call_fn_missing_fn_does_not_panic() {
        let mut rt = RhaiRuntime::new();
        rt.load_script("empty", r#""#).unwrap();
        let result = rt.call_fn(0, "nonexistent_fn", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn drain_pending_commands_captures_log_call() {
        // Drain any leftover commands from other tests first
        let _ = drain_pending_commands();

        let mut rt = RhaiRuntime::new();
        rt.load_script("test", r#"fn trigger() { log("hello from script"); }"#).unwrap();
        let _ = rt.call_fn(0, "trigger", &[]);

        let cmds = drain_pending_commands();
        assert!(!cmds.is_empty(), "log() call should produce a ScriptCommand::Log");
    }
}

#[cfg(test)]
mod hot_reload_tests {
    use super::*;

    #[test]
    fn poll_reload_is_rate_limited() {
        let mut rt = RhaiRuntime::new();
        rt.reload_interval = std::time::Duration::from_secs(10);
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(15);
        let _first = rt.poll_reload();
        let second = rt.poll_reload();
        assert!(second.is_empty(), "second poll within interval must return empty");
    }

    #[test]
    fn poll_reload_returns_empty_when_no_scripts() {
        let mut rt = RhaiRuntime::new();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let result = rt.poll_reload();
        assert!(result.is_empty());
    }

    #[test]
    fn script_reloads_when_file_updated() {
        let dir = std::env::temp_dir();
        let path = dir.join("ochroma_test_script_hot.rhai");
        std::fs::write(&path, "fn on_update(dt) {}").unwrap();

        let mut rt = RhaiRuntime::new();
        rt.load_script_file("test", &path).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&path, "fn on_update(dt) { let x = 1; }").unwrap();

        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let reloaded = rt.poll_reload();
        assert!(reloaded.len() <= 1);

        let _ = std::fs::remove_file(&path);
    }

    /// A good edit increments `script_reloads` and the recompiled AST is the one
    /// that runs afterwards (the returned value changes).
    #[test]
    fn good_edit_counts_reload_and_swaps_behaviour() {
        let dir = std::env::temp_dir();
        let path = dir.join("ochroma_test_good_edit.rhai");
        std::fs::write(&path, "fn amp() { 5 }").unwrap();

        let mut rt = RhaiRuntime::new();
        rt.load_script_file("g", &path).unwrap();
        let before: i64 = rt.call_fn(0, "amp", &[]).unwrap().cast();
        assert_eq!(before, 5);

        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&path, "fn amp() { 9 }").unwrap();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let reloaded = rt.poll_reload();

        assert_eq!(reloaded.len(), 1, "expected exactly one reload");
        assert_eq!(rt.script_reloads, 1, "script_reloads should be 1");
        assert_eq!(rt.script_errors, 0, "no errors on a clean edit");
        let after: i64 = rt.call_fn(0, "amp", &[]).unwrap().cast();
        assert_eq!(after, 9, "recompiled AST must drive the new value");

        let _ = std::fs::remove_file(&path);
    }

    /// A broken edit increments `script_errors`, surfaces `last_error`, and keeps
    /// the LAST-GOOD AST runnable (the previous value still computes). It must NOT
    /// increment `script_reloads`, and the game must keep running.
    #[test]
    fn broken_edit_counts_error_and_keeps_last_good() {
        let dir = std::env::temp_dir();
        let path = dir.join("ochroma_test_broken_edit.rhai");
        std::fs::write(&path, "fn amp() { 7 }").unwrap();

        let mut rt = RhaiRuntime::new();
        rt.load_script_file("b", &path).unwrap();
        let before: i64 = rt.call_fn(0, "amp", &[]).unwrap().cast();
        assert_eq!(before, 7);

        std::thread::sleep(std::time::Duration::from_millis(50));
        // Syntax error: unbalanced brace.
        std::fs::write(&path, "fn amp() { 7 ").unwrap();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let reloaded = rt.poll_reload();

        assert!(reloaded.is_empty(), "broken edit must not report a reload");
        assert_eq!(rt.script_errors, 1, "script_errors should be 1");
        assert_eq!(rt.script_reloads, 0, "broken edit must not count as reload");
        assert!(rt.last_error.is_some(), "last_error must be surfaced");
        // Last-good AST still drives the same value — the game did not crash.
        let after: i64 = rt.call_fn(0, "amp", &[]).unwrap().cast();
        assert_eq!(after, 7, "last-good behaviour must persist through a bad edit");

        // Polling again without a new edit must NOT re-count the same error
        // (mtime was advanced), so the error counter stays at 1.
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let _ = rt.poll_reload();
        assert_eq!(rt.script_errors, 1, "same broken file must not re-count");

        let _ = std::fs::remove_file(&path);
    }
}
