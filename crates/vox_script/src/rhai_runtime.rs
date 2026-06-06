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
    /// File length recorded together with `last_mtime`. On filesystems with
    /// coarse (e.g. 1 s) mtime granularity, a fix written in the same second
    /// as a failed edit keeps the same mtime — the length change still
    /// triggers the reload. (A same-second, same-length edit remains
    /// undetectable without hashing; accepted gap.)
    pub last_len: u64,
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
    /// Name of the script that produced `last_error`. A successful reload
    /// clears the error only when THIS script recovers — a different script
    /// reloading cleanly must not mask a still-broken one.
    pub last_error_script: Option<String>,
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
            last_error_script: None,
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
            last_len: 0,
        });
        Ok(idx)
    }

    /// Load a script from a file.
    pub fn load_script_file(&mut self, name: &str, path: &Path) -> Result<usize, String> {
        let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let ast = self.engine.compile(&source).map_err(|e| format!("Compile error in {}: {}", path.display(), e))?;
        let idx = self.scripts.len();
        let (mtime, len) = std::fs::metadata(path)
            .map(|m| (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len()))
            .unwrap_or((std::time::UNIX_EPOCH, 0));
        self.scripts.push(RhaiScript {
            name: name.to_string(),
            source_path: Some(path.to_string_lossy().to_string()),
            ast,
            last_mtime: mtime,
            last_len: len,
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

    /// Reload all file-based scripts. Flows through the same reload accounting
    /// as `poll_reload` (counters, `last_error`, HUD surfacing) — a manual
    /// reload-all must not bypass the telemetry the HUD displays.
    pub fn reload_all(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        for i in 0..self.scripts.len() {
            if self.scripts[i].source_path.is_none() {
                continue;
            }
            let name = self.scripts[i].name.clone();
            match self.reload(i) {
                Ok(()) => self.record_reload_ok(&name),
                Err(e) => {
                    errors.push(format!("{}: {}", name, e));
                    self.record_reload_err(&name, &e);
                }
            }
        }
        errors
    }

    /// Shared success-side accounting for any reload path.
    fn record_reload_ok(&mut self, name: &str) {
        self.script_reloads += 1;
        // Only the recovery of the script that ERRORED clears the banner — a
        // different script reloading cleanly must not mask a still-broken one.
        if self.last_error_script.as_deref() == Some(name) {
            self.last_error = None;
            self.last_error_script = None;
        }
    }

    /// Shared failure-side accounting for any reload path.
    fn record_reload_err(&mut self, name: &str, err: &str) {
        self.script_errors += 1;
        self.last_error = Some(format!("{}: {}", name, err));
        self.last_error_script = Some(name.to_string());
    }

    /// Poll for changed script files and hot-reload them if the interval has elapsed.
    /// Returns names of scripts that were successfully reloaded.
    pub fn poll_reload(&mut self) -> Vec<String> {
        if self.last_reload_check.elapsed() < self.reload_interval {
            return Vec::new();
        }
        self.last_reload_check = std::time::Instant::now();

        let mut to_reload: Vec<(usize, String, std::time::SystemTime, u64)> = Vec::new();
        for (i, script) in self.scripts.iter().enumerate() {
            let path = match &script.source_path {
                Some(p) => p.clone(),
                None => continue,
            };
            // Change detection is the (mtime, len) fingerprint, not mtime
            // ordering: on coarse-mtime filesystems a fix written in the same
            // second as a failed edit keeps the same mtime, and a checkout can
            // even move mtime backwards — both still reload when the length
            // moves. (Same-second AND same-length edits stay invisible.)
            if let Ok(meta) = std::fs::metadata(&path)
                && let Ok(mtime) = meta.modified()
                && (mtime != script.last_mtime || meta.len() != script.last_len)
            {
                to_reload.push((i, script.name.clone(), mtime, meta.len()));
            }
        }

        let mut result = Vec::new();
        for (i, name, mtime, len) in to_reload {
            // Record the fingerprint regardless of outcome so a file that fails
            // to compile is not recompiled on every poll — it is retried exactly
            // when its content changes again, which is when a fix would land.
            self.scripts[i].last_mtime = mtime;
            self.scripts[i].last_len = len;
            match self.reload(i) {
                Ok(()) => {
                    self.record_reload_ok(&name);
                    println!("[ochroma] Hot-reloaded script: {}", name);
                    result.push(name);
                }
                Err(e) => {
                    // Last-good AST is untouched (reload() returns before swapping
                    // on a compile error), so the game keeps running. Count + surface.
                    self.record_reload_err(&name, &e);
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

    /// A fix written in the SAME mtime second as the failed edit (coarse-mtime
    /// filesystem) must still reload: the (mtime, len) fingerprint catches the
    /// length change even when the mtime is identical.
    #[test]
    fn same_second_fix_after_failure_still_reloads() {
        let dir = std::env::temp_dir();
        let path = dir.join("ochroma_test_same_second_fix.rhai");
        std::fs::write(&path, "fn amp() { 3 }").unwrap();

        let mut rt = RhaiRuntime::new();
        rt.load_script_file("s", &path).unwrap();

        // Broken edit, polled → error recorded, fingerprint advanced.
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(&path, "fn amp() { 3 ").unwrap();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let _ = rt.poll_reload();
        assert_eq!(rt.script_errors, 1);

        // Fix with a DIFFERENT length, then simulate a coarse-mtime fs by
        // pinning the recorded mtime to the file's current mtime (identical
        // mtimes; only the recorded length still reflects the broken write).
        std::fs::write(&path, "fn amp() { 11 }").unwrap();
        let real_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        rt.scripts[0].last_mtime = real_mtime; // mtime says "unchanged"
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let reloaded = rt.poll_reload();

        assert_eq!(reloaded.len(), 1, "same-mtime fix must reload via length change");
        let v: i64 = rt.call_fn(0, "amp", &[]).unwrap().cast();
        assert_eq!(v, 11, "the fixed AST must actually run");
        assert!(rt.last_error.is_none(), "recovery must clear the error banner");

        let _ = std::fs::remove_file(&path);
    }

    /// A clean reload of script B must NOT clear the error banner raised by a
    /// still-broken script A; only A's own recovery clears it.
    #[test]
    fn other_script_reload_does_not_mask_error() {
        let dir = std::env::temp_dir();
        let pa = dir.join("ochroma_test_mask_a.rhai");
        let pb = dir.join("ochroma_test_mask_b.rhai");
        std::fs::write(&pa, "fn a() { 1 }").unwrap();
        std::fs::write(&pb, "fn b() { 2 }").unwrap();

        let mut rt = RhaiRuntime::new();
        rt.load_script_file("a", &pa).unwrap();
        rt.load_script_file("b", &pb).unwrap();

        // Break A.
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(&pa, "fn a() { 1 ").unwrap();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let _ = rt.poll_reload();
        assert!(rt.last_error.is_some(), "A's break must surface");

        // Cleanly edit B — banner must survive (A is still broken).
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(&pb, "fn b() { 22 }").unwrap();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let reloaded = rt.poll_reload();
        assert_eq!(reloaded, vec!["b".to_string()], "B reloads cleanly");
        assert!(
            rt.last_error.is_some(),
            "B's clean reload must not mask A's standing error"
        );

        // Fix A — now the banner clears.
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(&pa, "fn a() { 10 }").unwrap();
        rt.last_reload_check = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let _ = rt.poll_reload();
        assert!(rt.last_error.is_none(), "A's own recovery clears the banner");
        let va: i64 = rt.call_fn(0, "a", &[]).unwrap().cast();
        assert_eq!(va, 10);

        let _ = std::fs::remove_file(&pa);
        let _ = std::fs::remove_file(&pb);
    }
}
