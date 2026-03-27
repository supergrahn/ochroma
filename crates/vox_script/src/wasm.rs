use wasmtime::{Engine, Instance, Linker, Module, Store};

use crate::ScriptError;

/// The core Wasm execution engine, wrapping wasmtime.
pub struct WasmEngine {
    engine: Engine,
}

/// A compiled Wasm module ready for instantiation.
pub struct LoadedModule {
    pub name: String,
    module: Module,
}

impl WasmEngine {
    /// Create a new Wasm engine with default configuration.
    pub fn new() -> Result<Self, ScriptError> {
        let engine =
            Engine::default();
        Ok(Self { engine })
    }

    /// Compile a Wasm module from raw bytes (binary `.wasm` format).
    pub fn compile(&self, name: &str, wasm_bytes: &[u8]) -> Result<LoadedModule, ScriptError> {
        let module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| ScriptError::LoadFailed(format!("Compilation failed: {}", e)))?;
        Ok(LoadedModule {
            name: name.to_string(),
            module,
        })
    }

    /// Instantiate a compiled module in a fresh store, returning the store and instance.
    pub fn instantiate(
        &self,
        loaded: &LoadedModule,
    ) -> Result<(Store<()>, Instance), ScriptError> {
        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);
        let instance = linker
            .instantiate(&mut store, &loaded.module)
            .map_err(|e| ScriptError::LoadFailed(format!("Instantiation failed: {}", e)))?;
        Ok((store, instance))
    }

    /// Returns a reference to the underlying wasmtime engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}
