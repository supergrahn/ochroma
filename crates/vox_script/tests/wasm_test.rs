use vox_script::{GameEvent, ScriptRuntime};

/// Minimal valid Wasm binary (empty module).
#[cfg(feature = "wasm-runtime")]
const EMPTY_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, // magic: \0asm
    0x01, 0x00, 0x00, 0x00, // version: 1
];

#[cfg(feature = "wasm-runtime")]
#[test]
fn compile_empty_module() {
    let engine = vox_script::WasmEngine::new().unwrap();
    let module = engine.compile("empty", EMPTY_WASM);
    assert!(module.is_ok());
    assert_eq!(module.unwrap().name, "empty");
}

#[cfg(feature = "wasm-runtime")]
#[test]
fn instantiate_empty_module() {
    let engine = vox_script::WasmEngine::new().unwrap();
    let loaded = engine.compile("empty", EMPTY_WASM).unwrap();
    let result = engine.instantiate(&loaded);
    assert!(result.is_ok());
}

#[cfg(feature = "wasm-runtime")]
#[test]
fn compile_invalid_wasm_fails() {
    let engine = vox_script::WasmEngine::new().unwrap();
    let result = engine.compile("bad", &[0x00, 0x01, 0x02]);
    assert!(result.is_err());
}

#[cfg(feature = "wasm-runtime")]
#[test]
fn runtime_wasm_integration() {
    let mut rt = ScriptRuntime::new();
    rt.init_wasm().unwrap();
    rt.load_wasm_module("empty", EMPTY_WASM).unwrap();
    assert_eq!(rt.wasm_module_count(), 1);
}

#[cfg(feature = "wasm-runtime")]
#[test]
fn runtime_wasm_not_initialized_error() {
    let mut rt = ScriptRuntime::new();
    // Don't call init_wasm
    let result = rt.load_wasm_module("empty", EMPTY_WASM);
    assert!(result.is_err());
}

// Tests that work without the wasm-runtime feature
#[test]
fn event_dispatch_works() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("test", &[]).unwrap();
    rt.subscribe("test", "BuildingPlaced");
    let handlers = rt.dispatch_event(&GameEvent::BuildingPlaced {
        position: [0.0, 0.0, 0.0],
        asset_id: "house".into(),
    });
    assert_eq!(handlers.len(), 1);
    assert_eq!(handlers[0], "test");
}

#[test]
fn wildcard_subscription() {
    let mut rt = ScriptRuntime::new();
    rt.load_module("logger", &[]).unwrap();
    rt.subscribe("logger", "*");
    let handlers = rt.dispatch_event(&GameEvent::CitizenBorn { citizen_id: 1 });
    assert_eq!(handlers.len(), 1);
}
