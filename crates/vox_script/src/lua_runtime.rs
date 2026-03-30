//! Lua 5.4 scripting runtime backed by mlua.

use mlua::prelude::*;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LuaError {
    #[error("mlua error: {0}")]
    Mlua(#[from] mlua::Error),
    #[error("script not found: {0}")]
    NotFound(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct LuaRuntime {
    lua: Lua,
    pub pending_reload: Vec<PathBuf>,
}

impl LuaRuntime {
    pub fn new() -> Result<Self, LuaError> {
        let lua = Lua::new();
        {
            let globals = lua.globals();
            globals.set("io", mlua::Value::Nil)?;
            globals.set("os", mlua::Value::Nil)?;
            globals.set("package", mlua::Value::Nil)?;
            globals.set("require", mlua::Value::Nil)?;
            globals.set("dofile", mlua::Value::Nil)?;
            globals.set("loadfile", mlua::Value::Nil)?;
        }
        lua.load(r#"
            function wait_frames(n)
                for _ = 1, n do coroutine.yield() end
            end
            function wait_seconds(t)
                local elapsed = 0
                while elapsed < t do
                    elapsed = elapsed + (frame_dt or 0.016)
                    coroutine.yield()
                end
            end
        "#).exec()?;
        Ok(Self { lua, pending_reload: Vec::new() })
    }

    pub fn exec_file(&mut self, path: &Path) -> Result<(), LuaError> {
        if !path.exists() {
            return Err(LuaError::NotFound(path.display().to_string()));
        }
        let src = std::fs::read_to_string(path)?;
        self.lua.load(&src).set_name(path.to_string_lossy().as_ref()).exec()?;
        Ok(())
    }

    pub fn call_update(&mut self, dt: f32) -> Result<(), LuaError> {
        let pending = std::mem::take(&mut self.pending_reload);
        for path in pending {
            let src = std::fs::read_to_string(&path)?;
            self.lua.load(&src).set_name(path.to_string_lossy().as_ref()).exec()?;
        }
        self.lua.globals().set("frame_dt", dt)?;
        let globals = self.lua.globals();
        let update_fn: Option<LuaFunction> = globals.get("update")?;
        if let Some(f) = update_fn {
            f.call::<()>(dt)?;
        }
        Ok(())
    }

    pub fn set_global<V: IntoLua>(&self, name: &str, value: V) -> Result<(), LuaError> {
        self.lua.globals().set(name, value)?;
        Ok(())
    }

    pub fn lua(&self) -> &Lua {
        &self.lua
    }
}

impl Default for LuaRuntime {
    fn default() -> Self {
        Self::new().expect("Lua 5.4 init should not fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_creates_clean() {
        let rt = LuaRuntime::new().unwrap();
        let io_val: mlua::Value = rt.lua().globals().get("io").unwrap();
        assert!(io_val.is_nil(), "io must be sandboxed out");
    }

    #[test]
    fn exec_inline_sets_global() {
        let rt = LuaRuntime::new().unwrap();
        rt.lua().load("x = 42").exec().unwrap();
        let x: i32 = rt.lua().globals().get("x").unwrap();
        assert_eq!(x, 42);
    }

    #[test]
    fn call_update_calls_lua_function() {
        let mut rt = LuaRuntime::new().unwrap();
        rt.lua().load("last_dt = 0.0; function update(dt) last_dt = dt end").exec().unwrap();
        rt.call_update(0.016).unwrap();
        let last_dt: f32 = rt.lua().globals().get("last_dt").unwrap();
        println!("last_dt = {}", last_dt);
        assert!((last_dt - 0.016).abs() < 1e-5, "update(dt) should have been called with 0.016, got {}", last_dt);
    }

    #[test]
    fn call_update_is_noop_without_update_fn() {
        let mut rt = LuaRuntime::new().unwrap();
        assert!(rt.call_update(0.016).is_ok());
    }

    #[test]
    fn os_is_sandboxed() {
        let rt = LuaRuntime::new().unwrap();
        let os_val: mlua::Value = rt.lua().globals().get("os").unwrap();
        assert!(os_val.is_nil(), "os must be sandboxed out");
    }

    #[test]
    fn require_is_sandboxed() {
        let rt = LuaRuntime::new().unwrap();
        let req: mlua::Value = rt.lua().globals().get("require").unwrap();
        assert!(req.is_nil(), "require must be sandboxed out");
    }

    #[test]
    fn pending_reload_is_processed_on_tick() {
        let mut rt = LuaRuntime::new().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join("test_hot_reload.lua");
        std::fs::write(&path, "reloaded = true").unwrap();
        rt.lua().load("reloaded = false").exec().unwrap();
        rt.pending_reload.push(path.clone());
        rt.call_update(0.0).unwrap();
        let v: bool = rt.lua().globals().get("reloaded").unwrap();
        assert!(v, "hot reload should have re-executed the script");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wait_frames_is_defined() {
        let rt = LuaRuntime::new().unwrap();
        let f: Option<LuaFunction> = rt.lua().globals().get("wait_frames").unwrap();
        assert!(f.is_some(), "wait_frames should be registered");
    }

    #[test]
    fn wait_seconds_is_defined() {
        let rt = LuaRuntime::new().unwrap();
        let f: Option<LuaFunction> = rt.lua().globals().get("wait_seconds").unwrap();
        assert!(f.is_some(), "wait_seconds should be registered");
    }

    #[test]
    fn frame_dt_is_set_on_update() {
        let mut rt = LuaRuntime::new().unwrap();
        rt.call_update(0.033).unwrap();
        let dt: f32 = rt.lua().globals().get("frame_dt").unwrap();
        println!("frame_dt = {}", dt);
        assert!((dt - 0.033).abs() < 1e-5, "frame_dt should be 0.033, got {}", dt);
    }

    #[test]
    fn game_lua_parses_cleanly() {
        let mut rt = LuaRuntime::new().unwrap();
        // Register spectral stub so game.lua can call spectral.on_threshold/get_band
        rt.lua().load(r#"
            spectral = {
                on_threshold = function(...) end,
                get_band = function(...) return 0.0 end,
            }
        "#).exec().unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/scripts/game.lua");
        if path.exists() {
            rt.exec_file(&path).expect("game.lua should parse without error");
            println!("game.lua parsed successfully from {:?}", path);
        } else {
            println!("game.lua not found at {:?} — skipping", path);
        }
    }
}
