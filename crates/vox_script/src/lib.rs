use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("module not found: {0}")]
    ModuleNotFound(String),
    #[error("wasm compilation failed: {0}")]
    CompilationFailed(String),
    #[error("memory budget exceeded")]
    MemoryBudgetExceeded,
    #[error("cpu budget exceeded")]
    CpuBudgetExceeded,
}

#[derive(Debug, Clone)]
pub struct ScriptModule {
    pub name: String,
    pub memory_budget_bytes: usize,
    pub cpu_budget_ms: f32,
}

pub struct ScriptRuntime {
    modules: HashMap<String, ScriptModule>,
}

impl ScriptRuntime {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    pub fn load_module(&mut self, name: &str, _wasm_bytes: &[u8]) -> Result<(), ScriptError> {
        let module = ScriptModule {
            name: name.to_string(),
            memory_budget_bytes: 64 * 1024 * 1024, // 64 MiB default
            cpu_budget_ms: 16.0,
        };
        self.modules.insert(name.to_string(), module);
        Ok(())
    }

    pub fn tick(&mut self, _dt: f32) {
        // Tick all loaded modules
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

impl Default for ScriptRuntime {
    fn default() -> Self {
        Self::new()
    }
}
