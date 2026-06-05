//! End-to-end proof that script I/O is NOT hollow:
//! the host populates a non-zero spectral state via the clean write API, and
//! BOTH a Lua and a Rhai script read the SAME band and react to the REAL
//! non-zero value (not the old dead 0.0 placeholder).

use std::sync::{Arc, Mutex};

use half::f16;
use mlua::prelude::*;
use vox_script::entity_bindings::{register_entity_bindings, EntityStore};
use vox_script::rhai_runtime::RhaiRuntime;
use vox_script::spectral_bindings::{register_spectral_bindings, SpectralState};

/// Lua script reads band 7 of the populated field, scales it, and a threshold
/// callback flips a flag because the real energy exceeds the threshold.
#[test]
fn lua_script_reads_real_nonzero_band_and_reacts() {
    // --- Host populates a REAL, non-zero spectral field via the write API ---
    let mut state = SpectralState::new();
    // 0.875 is exactly representable in f16, so decode is lossless.
    assert!(state.set_band_energy(7, 0.875), "band 7 must be in range");
    // Sanity: writing out of range is rejected, not silently corrupting.
    assert!(!state.set_band_energy(16, 1.0), "band 16 must be rejected");
    let state = Arc::new(Mutex::new(state));

    let lua = Lua::new();
    register_spectral_bindings(&lua, state.clone()).unwrap();

    // Script reads band 7 and derives a value from the REAL energy.
    lua.load(
        r#"
        observed = spectral.field_energy(1.0, 2.0, 3.0, 5.0, 7)
        reacted  = observed * 2.0
        "#,
    )
    .exec()
    .unwrap();

    let observed: f32 = lua.globals().get("observed").unwrap();
    let reacted: f32 = lua.globals().get("reacted").unwrap();

    // The bug being fixed: this used to be 0.0 because the store was inert.
    assert!(observed > 0.0, "Lua must observe a NON-ZERO band, got {observed}");
    assert!(
        (observed - 0.875).abs() < 1e-4,
        "Lua must observe the REAL written value 0.875, got {observed}"
    );
    assert!(
        (reacted - 1.75).abs() < 1e-4,
        "Lua reaction (observed * 2.0) must be 1.75, got {reacted}"
    );
}

/// Entity store: host writes an engine-canonical [u16;16] spectral signature;
/// the Lua `entity.get_spectral` binding reads back the decoded REAL value.
#[test]
fn lua_entity_spectral_reads_real_decoded_u16() {
    let mut store = EntityStore::new();
    let mut sig = [0u16; 16];
    sig[4] = f16::from_f32(0.25).to_bits(); // exactly representable
    sig[9] = f16::from_f32(0.5).to_bits();
    store.set_entity_position(42, [1.0, 2.0, 3.0]);
    store.set_entity_spectral(42, sig);

    // Accessor reflects the decoded f32.
    assert!(
        (store.entity_band(42, 4) - 0.25).abs() < 1e-4,
        "entity_band(42,4) should decode to 0.25, got {}",
        store.entity_band(42, 4)
    );

    let store = Arc::new(Mutex::new(store));
    let lua = Lua::new();
    register_entity_bindings(&lua, store).unwrap();

    let band4: f32 = lua.load("return entity.get_spectral(42, 4)").eval().unwrap();
    let band9: f32 = lua.load("return entity.get_spectral(42, 9)").eval().unwrap();

    assert!(band4 > 0.0, "Lua entity band 4 must be NON-ZERO, got {band4}");
    assert!(
        (band4 - 0.25).abs() < 1e-4,
        "Lua entity band 4 must read REAL 0.25, got {band4}"
    );
    assert!(
        (band9 - 0.5).abs() < 1e-4,
        "Lua entity band 9 must read REAL 0.5, got {band9}"
    );
}

/// Rhai script reads band 7 of the host-populated field and reacts. Previously
/// the Rhai runtime had no spectral access at all, so any read was impossible /
/// dead. Now `field_energy` returns the REAL value.
#[test]
fn rhai_script_reads_real_nonzero_band_and_reacts() {
    let mut rt = RhaiRuntime::new();

    // --- Host populates band 7 via the engine-canonical u16 write API ---
    let bits = f16::from_f32(0.875).to_bits();
    assert!(rt.set_band_energy_u16(7, bits), "band 7 must be in range");

    // Script reads the field and reacts: returns true only if energy is high.
    let idx = rt
        .load_script(
            "reactor",
            r#"
            fn react() {
                let e = field_energy(1.0, 2.0, 3.0, 5.0, 7);
                // React: report doubled energy iff above a threshold.
                if e > 0.8 { e * 2.0 } else { -1.0 }
            }
            "#,
        )
        .unwrap();

    let out = rt
        .call_fn(idx, "react", &[])
        .expect("react() must run");
    let reacted: f64 = out.cast();

    // The bug being fixed: a dead store returns 0.0, so react() returns -1.0.
    assert!(
        reacted > 0.0,
        "Rhai must react to a NON-ZERO band (dead store would give -1.0), got {reacted}"
    );
    assert!(
        (reacted - 1.75).abs() < 1e-3,
        "Rhai reaction (energy 0.875 * 2.0) must be 1.75, got {reacted}"
    );
}

/// Both runtimes, driven off the SAME written band value, agree on the real
/// number — proving the write API feeds both bindings consistently.
#[test]
fn lua_and_rhai_agree_on_same_written_band() {
    let band = 11usize;
    let value = 0.75f32; // exactly representable in f16

    // Rhai side.
    let mut rt = RhaiRuntime::new();
    assert!(rt.set_band_energy(band, value));
    let idx = rt
        .load_script("read", r#"fn read() { field_energy(0.0,0.0,0.0,1.0,11) }"#)
        .unwrap();
    let rhai_val: f64 = rt.call_fn(idx, "read", &[]).unwrap().cast();

    // Lua side, same band, same value.
    let mut state = SpectralState::new();
    state.set_band_energy(band, value);
    let lua = Lua::new();
    register_spectral_bindings(&lua, Arc::new(Mutex::new(state))).unwrap();
    let lua_val: f32 = lua
        .load("return spectral.field_energy(0.0,0.0,0.0,1.0,11)")
        .eval()
        .unwrap();

    assert!(rhai_val > 0.0 && lua_val > 0.0, "both must be non-zero");
    assert!(
        (rhai_val as f32 - lua_val).abs() < 1e-4,
        "Rhai ({rhai_val}) and Lua ({lua_val}) must observe the same real band value"
    );
    assert!(
        (lua_val - 0.75).abs() < 1e-4,
        "value must be the real written 0.75, got {lua_val}"
    );
}
