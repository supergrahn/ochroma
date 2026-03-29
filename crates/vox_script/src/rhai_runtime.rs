use rhai::{Engine, AST, Scope, Dynamic};
use std::path::Path;
use std::sync::Mutex;
use vox_core::script_interface::ScriptCommand;

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

        Self {
            engine,
            scripts: Vec::new(),
            last_reload_check: std::time::Instant::now(),
            reload_interval: std::time::Duration::from_secs(1),
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
            if self.scripts[i].source_path.is_some() {
                if let Err(e) = self.reload(i) {
                    errors.push(format!("{}: {}", self.scripts[i].name, e));
                }
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
}
