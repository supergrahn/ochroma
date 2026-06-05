//! Lua `spectral.*` namespace — spectral query and threshold callbacks.

use mlua::prelude::*;
use std::sync::{Arc, Mutex};

pub struct SpectralState {
    pub band_energy: [f32; 16],
    pub thresholds: Vec<ThresholdEntry>,
}

pub struct ThresholdEntry {
    pub pos: [f32; 3],
    pub radius: f32,
    pub band: usize,
    pub threshold: f32,
    pub registry_key: LuaRegistryKey,
}

impl SpectralState {
    pub fn new() -> Self {
        Self { band_energy: [0.0f32; 16], thresholds: Vec::new() }
    }

    /// Write a single band's energy in linear f32. This is the clean host-side
    /// write API the engine uses every frame to push real field samples in.
    /// Returns `false` (and writes nothing) if `band` is out of range.
    pub fn set_band_energy(&mut self, band: usize, value: f32) -> bool {
        if band >= 16 {
            return false;
        }
        self.band_energy[band] = value;
        true
    }

    /// Write a single band's energy from the engine-canonical `u16` (f16-bits)
    /// spectral encoding used by `vox_core` (`[u16; 16]`, 380–755 nm).
    /// Decodes via `vox_core::spectral_damage::decode_spectral_u16` semantics.
    pub fn set_band_energy_u16(&mut self, band: usize, bits: u16) -> bool {
        if band >= 16 {
            return false;
        }
        self.band_energy[band] = half::f16::from_bits(bits).to_f32();
        true
    }

    /// Overwrite all 16 bands at once from an engine-canonical `[u16; 16]`
    /// f16-bit spectral sample (the format `Splat`/terrain/mapgen produce).
    pub fn set_band_energy_all_u16(&mut self, spectral: &[u16; 16]) {
        self.band_energy = vox_core::spectral_damage::decode_spectral_u16(spectral);
    }

    /// Read a single band's energy in linear f32. Returns 0.0 for out-of-range.
    pub fn band_energy(&self, band: usize) -> f32 {
        if band < 16 { self.band_energy[band] } else { 0.0 }
    }
}

impl Default for SpectralState {
    fn default() -> Self { Self::new() }
}

pub fn register_spectral_bindings(
    lua: &Lua,
    state: Arc<Mutex<SpectralState>>,
) -> Result<(), mlua::Error> {
    let spectral = lua.create_table()?;

    {
        let s = state.clone();
        spectral.set("get_band", lua.create_function(move |_, (x, y, z, band): (f32, f32, f32, usize)| {
            let _ = (x, y, z);
            let guard = s.lock().unwrap();
            if band < 16 {
                Ok(guard.band_energy[band])
            } else {
                Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)))
            }
        })?)?;
    }

    {
        let s = state.clone();
        spectral.set("field_energy", lua.create_function(move |_, (x, y, z, _radius, band): (f32, f32, f32, f32, usize)| {
            let _ = (x, y, z);
            let guard = s.lock().unwrap();
            if band < 16 {
                Ok(guard.band_energy[band])
            } else {
                Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)))
            }
        })?)?;
    }

    {
        let s = state.clone();
        spectral.set("on_threshold", lua.create_function(move |lua_ctx, (x, y, z, radius, band, threshold, cb): (f32, f32, f32, f32, usize, f32, LuaFunction)| {
            let key = lua_ctx.create_registry_value(cb)?;
            let mut guard = s.lock().unwrap();
            guard.thresholds.push(ThresholdEntry {
                pos: [x, y, z],
                radius,
                band,
                threshold,
                registry_key: key,
            });
            Ok(())
        })?)?;
    }

    lua.globals().set("spectral", spectral)?;
    Ok(())
}

pub fn tick_thresholds(lua: &Lua, state: &Mutex<SpectralState>) -> Result<(), mlua::Error> {
    // Snapshot energies and collect indices of entries that should fire — lock released after this block
    let (energies, fired_indices) = {
        let guard = state.lock().unwrap();
        let energies = guard.band_energy;
        let fired: Vec<usize> = guard.thresholds.iter().enumerate()
            .filter(|(_, e)| e.band < 16 && energies[e.band] >= e.threshold)
            .map(|(i, _)| i)
            .collect();
        (energies, fired)
    };

    // For each fired entry: briefly re-lock to fetch the callback, then call outside the lock
    for idx in fired_indices {
        let (band_f, energy_f, cb) = {
            let guard = state.lock().unwrap();
            let entry = match guard.thresholds.get(idx) {
                Some(e) => e,
                None => continue,
            };
            let cb: LuaFunction = lua.registry_value(&entry.registry_key)?;
            (entry.band as f32, energies[entry.band], cb)
        }; // lock released here
        cb.call::<()>((band_f, energy_f))?; // called outside the lock
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(band_values: [f32; 16]) -> Arc<Mutex<SpectralState>> {
        let mut s = SpectralState::new();
        s.band_energy = band_values;
        Arc::new(Mutex::new(s))
    }

    #[test]
    fn get_band_returns_value() {
        let lua = Lua::new();
        let state = make_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        register_spectral_bindings(&lua, state).unwrap();
        let v: f32 = lua.load("return spectral.get_band(0,0,0,2)").eval().unwrap();
        println!("band 2 = {}", v);
        assert!((v - 0.3).abs() < 1e-5, "band 2 should be 0.3, got {}", v);
    }

    #[test]
    fn get_band_out_of_range_errors() {
        let lua = Lua::new();
        let state = make_state([0.0; 16]);
        register_spectral_bindings(&lua, state).unwrap();
        let result: mlua::Result<f32> = lua.load("return spectral.get_band(0,0,0,16)").eval();
        assert!(result.is_err(), "band 16 should be out of range");
    }

    #[test]
    fn on_threshold_registers_callback() {
        let lua = Lua::new();
        let state = make_state([0.0; 16]);
        register_spectral_bindings(&lua, state.clone()).unwrap();
        lua.load("spectral.on_threshold(0,0,0,1.0, 3, 0.5, function(band, val) end)").exec().unwrap();
        assert_eq!(state.lock().unwrap().thresholds.len(), 1);
    }

    #[test]
    fn tick_fires_callback_when_threshold_exceeded() {
        let lua = Lua::new();
        let mut s = SpectralState::new();
        s.band_energy[5] = 0.9;
        let state = Arc::new(Mutex::new(s));
        register_spectral_bindings(&lua, state.clone()).unwrap();
        lua.load("fired = false; spectral.on_threshold(0,0,0,1.0, 5, 0.8, function(band, val) fired = true; print('fired = true band=' .. band .. ' energy=' .. val) end)").exec().unwrap();
        tick_thresholds(&lua, &state).unwrap();
        let fired: bool = lua.globals().get("fired").unwrap();
        assert!(fired, "callback should have fired when band 5 energy (0.9) exceeded threshold (0.8)");
    }

    #[test]
    fn tick_does_not_fire_below_threshold() {
        let lua = Lua::new();
        let mut s = SpectralState::new();
        s.band_energy[5] = 0.3;
        let state = Arc::new(Mutex::new(s));
        register_spectral_bindings(&lua, state.clone()).unwrap();
        lua.load("fired = false; spectral.on_threshold(0,0,0,1.0, 5, 0.8, function(b,v) fired = true end)").exec().unwrap();
        tick_thresholds(&lua, &state).unwrap();
        let fired: bool = lua.globals().get("fired").unwrap();
        assert!(!fired, "callback must not fire below threshold");
    }

    #[test]
    fn field_energy_returns_band_value() {
        let lua = Lua::new();
        let state = make_state([0.0, 0.0, 0.0, 0.0, 0.77, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        register_spectral_bindings(&lua, state).unwrap();
        let v: f32 = lua.load("return spectral.field_energy(1,2,3, 5.0, 4)").eval().unwrap();
        println!("field_energy band 4 = {}", v);
        assert!((v - 0.77).abs() < 1e-5, "field_energy band 4 should be 0.77, got {}", v);
    }
}
