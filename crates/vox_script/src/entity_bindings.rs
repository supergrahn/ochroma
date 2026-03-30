//! Lua `entity.*` namespace — read/write entity position and spectral bands.

use mlua::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct EntityStore {
    pub positions: HashMap<u32, [f32; 3]>,
    pub spectral:  HashMap<u32, [f32; 16]>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self { positions: HashMap::new(), spectral: HashMap::new() }
    }

    pub fn insert(&mut self, id: u32, pos: [f32; 3], spectral: [f32; 16]) {
        self.positions.insert(id, pos);
        self.spectral.insert(id, spectral);
    }
}

impl Default for EntityStore {
    fn default() -> Self { Self::new() }
}

pub fn register_entity_bindings(
    lua: &Lua,
    store: Arc<Mutex<EntityStore>>,
) -> Result<(), mlua::Error> {
    let entity = lua.create_table()?;

    {
        let s = store.clone();
        entity.set("get_position", lua.create_function(move |lua_ctx, id: u32| {
            let guard = s.lock().unwrap();
            let pos = guard.positions.get(&id).copied().unwrap_or([0.0, 0.0, 0.0]);
            let t = lua_ctx.create_table()?;
            t.set("x", pos[0])?;
            t.set("y", pos[1])?;
            t.set("z", pos[2])?;
            Ok(t)
        })?)?;
    }

    {
        let s = store.clone();
        entity.set("set_spectral", lua.create_function(move |_, (id, band, value): (u32, usize, f32)| {
            if band >= 16 {
                return Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)));
            }
            let mut guard = s.lock().unwrap();
            let entry = guard.spectral.entry(id).or_insert([0.0f32; 16]);
            entry[band] = value.clamp(0.0, 1.0);
            Ok(())
        })?)?;
    }

    {
        let s = store.clone();
        entity.set("get_spectral", lua.create_function(move |_, (id, band): (u32, usize)| {
            if band >= 16 {
                return Err(mlua::Error::RuntimeError(format!("band {} out of range [0,15]", band)));
            }
            let guard = s.lock().unwrap();
            Ok(guard.spectral.get(&id).map_or(0.0, |s| s[band]))
        })?)?;
    }

    lua.globals().set("entity", entity)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> Arc<Mutex<EntityStore>> {
        let mut store = EntityStore::new();
        store.insert(1, [10.0, 20.0, 30.0], [0.5f32; 16]);
        Arc::new(Mutex::new(store))
    }

    #[test]
    fn get_position_returns_xyz() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let x: f32 = lua.load("return entity.get_position(1).x").eval().unwrap();
        println!("x = {}", x);
        assert!((x - 10.0).abs() < 1e-5, "x should be 10.0, got {}", x);
    }

    #[test]
    fn get_position_unknown_entity_returns_zero() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let x: f32 = lua.load("return entity.get_position(999).x").eval().unwrap();
        assert_eq!(x, 0.0);
    }

    #[test]
    fn set_spectral_writes_band() {
        let lua = Lua::new();
        let store = make_store();
        register_entity_bindings(&lua, store.clone()).unwrap();
        lua.load("entity.set_spectral(1, 3, 0.75)").exec().unwrap();
        let v = store.lock().unwrap().spectral[&1][3];
        println!("band 3 = {}", v);
        assert!((v - 0.75).abs() < 1e-5, "band 3 should be 0.75, got {}", v);
    }

    #[test]
    fn set_spectral_clamps_to_unit_range() {
        let lua = Lua::new();
        let store = make_store();
        register_entity_bindings(&lua, store.clone()).unwrap();
        lua.load("entity.set_spectral(1, 0, 2.5)").exec().unwrap();
        let v = store.lock().unwrap().spectral[&1][0];
        assert!((v - 1.0).abs() < 1e-5, "value 2.5 should clamp to 1.0, got {}", v);
    }

    #[test]
    fn get_spectral_reads_band() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let v: f32 = lua.load("return entity.get_spectral(1, 0)").eval().unwrap();
        assert!((v - 0.5).abs() < 1e-5, "band 0 should be 0.5, got {}", v);
    }

    #[test]
    fn set_spectral_out_of_range_errors() {
        let lua = Lua::new();
        register_entity_bindings(&lua, make_store()).unwrap();
        let r: mlua::Result<()> = lua.load("entity.set_spectral(1, 16, 0.5)").exec();
        assert!(r.is_err(), "band 16 should be out of range");
    }
}
