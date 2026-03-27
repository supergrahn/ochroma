use std::collections::HashMap;
use thiserror::Error;

pub mod mod_manager;

#[cfg(feature = "wasm-runtime")]
mod wasm;
#[cfg(feature = "wasm-runtime")]
pub use wasm::{LoadedModule, WasmEngine};

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("module not found: {0}")]
    ModuleNotFound(String),
    #[error("wasm compilation failed: {0}")]
    CompilationFailed(String),
    #[error("load failed: {0}")]
    LoadFailed(String),
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

/// Events that mods can subscribe to.
#[derive(Debug, Clone)]
pub enum GameEvent {
    BuildingPlaced { position: [f32; 3], asset_id: String },
    ZoneChanged { position: [f32; 2], zone_type: String },
    CitizenBorn { citizen_id: u32 },
    CitizenDied { citizen_id: u32 },
    BudgetTick { funds: f64 },
    Custom { name: String, data: Vec<u8> },
}

/// A mod's event handler registration.
pub struct EventSubscription {
    pub module_name: String,
    /// Event name to match, or "*" for all events.
    pub event_pattern: String,
}

pub struct ScriptRuntime {
    modules: HashMap<String, ScriptModule>,
    subscriptions: Vec<EventSubscription>,
    #[cfg(feature = "wasm-runtime")]
    wasm_engine: Option<WasmEngine>,
    #[cfg(feature = "wasm-runtime")]
    wasm_modules: Vec<LoadedModule>,
}

impl ScriptRuntime {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            subscriptions: Vec::new(),
            #[cfg(feature = "wasm-runtime")]
            wasm_engine: None,
            #[cfg(feature = "wasm-runtime")]
            wasm_modules: Vec::new(),
        }
    }

    /// Initialize the Wasm engine. Only available with the `wasm-runtime` feature.
    #[cfg(feature = "wasm-runtime")]
    pub fn init_wasm(&mut self) -> Result<(), ScriptError> {
        self.wasm_engine = Some(WasmEngine::new()?);
        Ok(())
    }

    /// Load a Wasm module from raw bytes. Requires `init_wasm()` to have been called first.
    #[cfg(feature = "wasm-runtime")]
    pub fn load_wasm_module(&mut self, name: &str, wasm_bytes: &[u8]) -> Result<(), ScriptError> {
        if let Some(engine) = &self.wasm_engine {
            let module = engine.compile(name, wasm_bytes)?;
            self.wasm_modules.push(module);
            Ok(())
        } else {
            Err(ScriptError::LoadFailed(
                "Wasm engine not initialized".into(),
            ))
        }
    }

    /// Returns the number of loaded Wasm modules.
    #[cfg(feature = "wasm-runtime")]
    pub fn wasm_module_count(&self) -> usize {
        self.wasm_modules.len()
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

    pub fn subscribe(&mut self, module_name: &str, event_pattern: &str) {
        self.subscriptions.push(EventSubscription {
            module_name: module_name.to_string(),
            event_pattern: event_pattern.to_string(),
        });
    }

    /// Dispatch an event and return the names of all matching module handlers.
    pub fn dispatch_event(&self, event: &GameEvent) -> Vec<String> {
        let event_name = match event {
            GameEvent::BuildingPlaced { .. } => "BuildingPlaced",
            GameEvent::ZoneChanged { .. } => "ZoneChanged",
            GameEvent::CitizenBorn { .. } => "CitizenBorn",
            GameEvent::CitizenDied { .. } => "CitizenDied",
            GameEvent::BudgetTick { .. } => "BudgetTick",
            GameEvent::Custom { name, .. } => name.as_str(),
        };

        self.subscriptions
            .iter()
            .filter(|s| s.event_pattern == "*" || s.event_pattern == event_name)
            .map(|s| s.module_name.clone())
            .collect()
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
