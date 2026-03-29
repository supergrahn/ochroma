use rhai::{Engine, AST, Scope, Dynamic};
use std::path::Path;
use std::sync::Mutex;
use vox_core::script_interface::ScriptCommand;

/// A Rhai script instance attached to an entity.
pub struct RhaiScript {
    pub name: String,
    pub source_path: Option<String>,
    ast: AST,
}

/// The Rhai scripting runtime — hot-reloadable game logic.
pub struct RhaiRuntime {
    engine: Engine,
    scripts: Vec<RhaiScript>,
}

static PENDING_COMMANDS: Mutex<Vec<ScriptCommand>> = Mutex::new(Vec::new());

pub fn drain_pending_commands() -> Vec<ScriptCommand> {
    PENDING_COMMANDS.lock().unwrap().drain(..).collect()
}

impl RhaiRuntime {
    pub fn new() -> Self {
        let mut engine = Engine::new();

        // Register engine API functions that scripts can call

        engine.register_fn("log", |message: String| {
            PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::Log { message });
        });

        engine.register_fn("spawn", |asset: &str, x: f64, y: f64, z: f64| -> i64 {
            println!("[rhai] spawn {} at ({}, {}, {})", asset, x, y, z);
            0 // return entity ID (stub)
        });

        engine.register_fn("play_sound", |clip: String, volume: f64| {
            PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::PlaySound {
                clip,
                volume: volume as f32,
                spatial: true,
            });
        });

        engine.register_fn("set_position", |x: f64, y: f64, z: f64| {
            PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::SetPosition {
                position: [x as f32, y as f32, z as f32],
            });
        });

        engine.register_fn("send_event", |name: String, data: String| {
            PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::SendEvent { name, data });
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

        Self { engine, scripts: Vec::new() }
    }

    /// Load a script from source code string.
    pub fn load_script(&mut self, name: &str, source: &str) -> Result<usize, Box<rhai::EvalAltResult>> {
        let ast = self.engine.compile(source)?;
        let idx = self.scripts.len();
        self.scripts.push(RhaiScript {
            name: name.to_string(),
            source_path: None,
            ast,
        });
        Ok(idx)
    }

    /// Load a script from a file.
    pub fn load_script_file(&mut self, name: &str, path: &Path) -> Result<usize, String> {
        let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let ast = self.engine.compile(&source).map_err(|e| format!("Compile error in {}: {}", path.display(), e))?;
        let idx = self.scripts.len();
        self.scripts.push(RhaiScript {
            name: name.to_string(),
            source_path: Some(path.to_string_lossy().to_string()),
            ast,
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
}
