use std::collections::HashMap;
use thiserror::Error;

pub mod lua_runtime;
pub use lua_runtime::{LuaError, LuaRuntime};

pub mod spectral_bindings;
pub use spectral_bindings::{SpectralState, register_spectral_bindings, tick_thresholds};

pub mod entity_bindings;
pub use entity_bindings::{EntityStore, register_entity_bindings};

pub mod hot_reload;
pub use hot_reload::{ScriptWatcher, WatchError};

pub mod mod_manager;
pub mod plugin_system;
pub mod rhai_runtime;
pub mod visual_graph;
pub mod visual_script;

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

/// A generic, string-keyed event that mods can subscribe to.
///
/// The engine knows nothing about building/zone/citizen/city specifics — the game
/// registers its own event names and packs any payload into `data`.
#[derive(Debug, Clone)]
pub struct ScriptEvent {
    /// Logical event name, e.g. `"BuildingPlaced"` or `"custom:explosion"`.
    pub name: String,
    /// Arbitrary serialised payload (game-defined; may be empty).
    pub data: Vec<u8>,
}

impl ScriptEvent {
    pub fn new(name: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }

    /// Convenience: create an event with no payload.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: Vec::new(),
        }
    }
}

/// Back-compat alias — callers that still import `GameEvent` compile without change.
/// Deprecated: prefer `ScriptEvent` directly.
#[deprecated(since = "0.0.0", note = "use ScriptEvent")]
pub type GameEvent = ScriptEvent;

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
    pub fn dispatch_event(&self, event: &ScriptEvent) -> Vec<String> {
        self.subscriptions
            .iter()
            .filter(|s| s.event_pattern == "*" || s.event_pattern == event.name)
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
