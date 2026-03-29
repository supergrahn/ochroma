# Sprint 2: Make a Game Someone Can Play

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire input, character controller, scripting, spatial audio, and world save/load so the engine can run a game someone can actually play.

**Architecture:** Tasks build on each other: input system provides InputState used by character controller; character controller's camera position feeds the spatial audio listener; scripting dispatches all ScriptCommands including SetPosition (which moves Rapier bodies); save/load serializes the resulting entity positions. All five systems share the same entity model: entity_id → (TransformComponent, PhysicsBodyComponent, optional script_index).

**Tech Stack:** bevy_ecs 0.16, rapier3d (via vox_physics), rhai (via vox_script), rodio (via vox_audio), serde_json (via vox_data::world_save), winit KeyCode

---

## Cross-Sprint Foundation Note

The entity model established in this sprint (entity_id → position → physics body → script binding) is the shared foundation for Sprint 3's animated characters, Sprint 4's hot-reloading scripts, and Sprint 5's spectral material assignment. Keep types in vox_core (engine layer) and implementations in vox_app (game layer).

---

## Task 1: Wire InputState into engine_runner.rs

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Add `input_state: vox_core::input::InputState` field to `EngineApp` struct.
- [ ] Initialize `input_state: vox_core::input::InputState::default()` in the `EngineApp` constructor.
- [ ] In the keyboard event handler (where `self.keys.insert(key)` already exists), add after each insert:
  ```rust
  self.input_state.press(vox_core::input::InputSource::Key(key as u32));
  ```
- [ ] In the keyboard release handler (where `self.keys.remove(key)` already exists), add after each remove:
  ```rust
  self.input_state.release(vox_core::input::InputSource::Key(key as u32));
  ```
- [ ] At the **end** of each frame (after all update logic, before present), call:
  ```rust
  self.input_state.end_frame();
  ```
- [ ] Write a unit test in `crates/vox_core/src/input.rs` (or a `#[cfg(test)]` block at the bottom):
  ```rust
  #[test]
  fn input_state_just_pressed_clears_after_end_frame() {
      let mut state = InputState::default();
      state.press(InputSource::Key(42));
      assert!(state.was_just_pressed(&InputSource::Key(42)));
      state.end_frame();
      assert!(!state.was_just_pressed(&InputSource::Key(42)));
      assert!(state.is_pressed(&InputSource::Key(42)));
  }
  ```
- [ ] Run `cargo test -p vox_core` and confirm the new test passes.

---

## Task 2: Character Controller

**Files:**
- Create: `crates/vox_app/src/character_controller.rs`
- Modify: `crates/vox_app/src/lib.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Create `crates/vox_app/src/character_controller.rs` with the following content:

```rust
use glam::Vec3;
use vox_physics::RapierPhysicsWorld;
use vox_core::input::{InputState, InputSource};

pub struct CharacterController {
    pub body_handle: vox_physics::RigidBodyHandle,
    pub collider_handle: vox_physics::ColliderHandle,
    pub position: Vec3,
    pub yaw: f32,
    pub speed: f32,
    pub jump_velocity: f32,
    pub on_ground: bool,
    pub vertical_velocity: f32,
    pub enabled: bool,
}

impl CharacterController {
    pub fn new(physics: &mut RapierPhysicsWorld, spawn_pos: Vec3) -> Self {
        let (body_handle, collider_handle) = physics.add_character_controller(
            [spawn_pos.x, spawn_pos.y, spawn_pos.z],
            0.4,
            1.8,
        );
        Self {
            body_handle,
            collider_handle,
            position: spawn_pos,
            yaw: 0.0,
            speed: 5.0,
            jump_velocity: 8.0,
            on_ground: true,
            vertical_velocity: 0.0,
            enabled: false,
        }
    }

    /// Update position based on input. Call before physics.step().
    pub fn update(
        &mut self,
        input: &InputState,
        dt: f32,
        physics: &mut RapierPhysicsWorld,
    ) {
        if !self.enabled {
            return;
        }

        // WASD using Linux/X11 scancodes: W=17, A=30, S=31, D=32
        let mut move_dir = Vec3::ZERO;
        let forward = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());

        if input.is_pressed(&InputSource::Key(17)) { move_dir += forward; }  // W
        if input.is_pressed(&InputSource::Key(31)) { move_dir -= forward; }  // S
        if input.is_pressed(&InputSource::Key(30)) { move_dir -= right; }    // A
        if input.is_pressed(&InputSource::Key(32)) { move_dir += right; }    // D

        if move_dir.length_squared() > 0.001 {
            move_dir = move_dir.normalize() * self.speed;
        }

        // Simple gravity
        if !self.on_ground {
            self.vertical_velocity -= 9.81 * dt;
        }

        // Jump: Space = key 57
        if self.on_ground && input.was_just_pressed(&InputSource::Key(57)) {
            self.vertical_velocity = self.jump_velocity;
            self.on_ground = false;
        }

        let next_pos = self.position + (move_dir + Vec3::Y * self.vertical_velocity) * dt;

        // Ground check: y <= 0.9 (half capsule height)
        if next_pos.y <= 0.9 {
            self.position = Vec3::new(next_pos.x, 0.9, next_pos.z);
            self.vertical_velocity = 0.0;
            self.on_ground = true;
        } else {
            self.position = next_pos;
        }

        physics.set_kinematic_position(
            self.body_handle,
            [self.position.x, self.position.y, self.position.z],
        );
    }

    /// Camera position: eye level above capsule center
    pub fn camera_position(&self) -> Vec3 {
        self.position + Vec3::Y * 0.8
    }

    /// Camera forward direction from yaw
    pub fn camera_forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos()).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_controller_camera_position_is_above_body() {
        let ctrl = CharacterController {
            body_handle: Default::default(),
            collider_handle: Default::default(),
            position: Vec3::new(0.0, 0.9, 0.0),
            yaw: 0.0,
            speed: 5.0,
            jump_velocity: 8.0,
            on_ground: true,
            vertical_velocity: 0.0,
            enabled: true,
        };
        let cam = ctrl.camera_position();
        assert!(cam.y > ctrl.position.y);
    }

    #[test]
    fn character_stays_on_ground_when_below_threshold() {
        // Gravity would pull below 0.9 — verify ground clamp
        let ctrl = CharacterController {
            body_handle: Default::default(),
            collider_handle: Default::default(),
            position: Vec3::new(0.0, 0.9, 0.0),
            yaw: 0.0,
            speed: 5.0,
            jump_velocity: 8.0,
            on_ground: false,
            vertical_velocity: -50.0, // strong downward
            enabled: true,
        };
        // next_pos.y = 0.9 + (-50.0) * 0.016 = 0.9 - 0.8 = 0.1 < 0.9 → clamped
        let next_y = ctrl.position.y + ctrl.vertical_velocity * 0.016;
        assert!(next_y < 0.9, "expected to go below ground threshold before clamp");
    }
}
```

- [ ] Add `pub mod character_controller;` to `crates/vox_app/src/lib.rs`.
- [ ] In `engine_runner.rs`, add `character: crate::character_controller::CharacterController` field to `EngineApp`.
- [ ] Initialize in the `EngineApp` constructor after `physics` is created:
  ```rust
  character: crate::character_controller::CharacterController::new(
      &mut physics,
      glam::Vec3::new(0.0, 0.9, 0.0),
  ),
  ```
- [ ] In the per-frame update, before `self.physics.step(dt)`, add:
  ```rust
  self.character.update(&self.input_state, dt, &mut self.physics);
  ```
- [ ] After character update, if `self.character.enabled`, update the camera position:
  ```rust
  if self.character.enabled {
      let cam_pos = self.character.camera_position();
      // update your existing camera position fields here
      // e.g. self.camera_pos = cam_pos; (adapt to actual field name)
  }
  ```
- [ ] Toggle character mode with `KeyP` (key 25). In the keyboard press event handler, add:
  ```rust
  if key as u32 == 25 {
      self.character.enabled = !self.character.enabled;
      println!("[ochroma] Character controller: {}", if self.character.enabled { "ON" } else { "OFF" });
  }
  ```
- [ ] Wire mouse delta to yaw when character is enabled and editor is not visible. In the mouse move handler:
  ```rust
  if self.character.enabled && !self.editor_visible {
      self.character.yaw += mouse_dx * 0.002;
  }
  ```
- [ ] Run `cargo test -p vox_app` and confirm character controller tests pass.

---

## Task 3: Scripting in Game Loop

**Files:**
- Modify: `crates/vox_script/src/rhai_runtime.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Fix the `_args` bug** in `crates/vox_script/src/rhai_runtime.rs`. Find the `call_fn` method and change:
  ```rust
  // BEFORE (broken — args ignored):
  pub fn call_fn(&mut self, index: usize, fn_name: &str, _args: &[rhai::Dynamic]) -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
      // ... args never used ...
  }

  // AFTER (fixed — args passed):
  pub fn call_fn(&mut self, index: usize, fn_name: &str, args: &[rhai::Dynamic]) -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
      let script = &self.scripts[index];
      self.engine.call_fn(&mut rhai::Scope::new(), &script.ast, fn_name, args.to_vec())
  }
  ```
- [ ] Add a `load_script` method that accepts a name and inline source string (not just file path), used by tests:
  ```rust
  pub fn load_script(&mut self, name: &str, source: &str) -> Result<usize, Box<rhai::EvalAltResult>> {
      let ast = self.engine.compile(source)?;
      let index = self.scripts.len();
      self.scripts.push(LoadedScript { name: name.to_string(), ast });
      Ok(index)
  }
  ```
- [ ] Add a global staging buffer for script commands in `crates/vox_script/src/rhai_runtime.rs`:
  ```rust
  use std::sync::Mutex;
  use vox_core::script_interface::ScriptCommand;

  static PENDING_COMMANDS: Mutex<Vec<ScriptCommand>> = Mutex::new(Vec::new());

  pub fn drain_pending_commands() -> Vec<ScriptCommand> {
      PENDING_COMMANDS.lock().unwrap().drain(..).collect()
  }
  ```
- [ ] Register Rhai API functions in `RhaiRuntime::new()` that push to `PENDING_COMMANDS`:
  ```rust
  engine.register_fn("set_position", |entity_id: i64, x: f64, y: f64, z: f64| {
      PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::SetPosition {
          entity_id: entity_id as u32,
          x: x as f32, y: y as f32, z: z as f32,
      });
  });
  engine.register_fn("play_sound", |clip_path: String, volume: f64| {
      PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::PlaySound {
          clip_path, volume: volume as f32,
      });
  });
  engine.register_fn("log", |message: String| {
      PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::Log { message });
  });
  engine.register_fn("send_event", |event_name: String, payload: String| {
      PENDING_COMMANDS.lock().unwrap().push(ScriptCommand::SendEvent {
          event_name, payload,
      });
  });
  ```
- [ ] Add `rhai: vox_script::rhai_runtime::RhaiRuntime` field to `EngineApp`.
- [ ] Initialize `rhai: vox_script::rhai_runtime::RhaiRuntime::new()` in the constructor.
- [ ] After scene setup in `EngineApp::new()` (or at first-frame init), attempt to load scripts from `scripts/` directory:
  ```rust
  if let Ok(entries) = std::fs::read_dir("scripts") {
      for entry in entries.flatten() {
          let path = entry.path();
          if path.extension().and_then(|e| e.to_str()) == Some("rhai") {
              match self.rhai.load_script_file(
                  path.file_stem().unwrap_or_default().to_str().unwrap_or(""),
                  &path,
              ) {
                  Ok(idx) => println!("[ochroma] Loaded script #{}: {}", idx, path.display()),
                  Err(e) => println!("[ochroma] Script load error {}: {}", path.display(), e),
              }
          }
      }
  }
  ```
- [ ] Per-frame in the update loop, call `on_update` for each loaded script:
  ```rust
  for i in 0..self.rhai.script_count() {
      let _ = self.rhai.call_fn(i, "on_update", &[rhai::Dynamic::from(dt as f64)]);
  }
  ```
  Note: add `pub fn script_count(&self) -> usize { self.scripts.len() }` to `RhaiRuntime` if it doesn't exist.
- [ ] After the per-frame script calls, drain and dispatch all `ScriptCommand`s:
  ```rust
  for cmd in vox_script::rhai_runtime::drain_pending_commands() {
      match cmd {
          ScriptCommand::SetPosition { entity_id, x, y, z } => {
              if let Some(&body_handle) = self.entity_rapier_bodies.get(&entity_id) {
                  self.physics.set_kinematic_position(body_handle, [x, y, z]);
              }
          }
          ScriptCommand::PlaySound { clip_path, volume } => {
              if let Some(ref h) = self.audio_handle {
                  let _ = h.play(std::path::Path::new(&clip_path), volume);
              }
          }
          ScriptCommand::Log { message } => {
              println!("[script] {}", message);
          }
          other => {
              println!("[script] unhandled command: {:?}", other);
          }
      }
  }
  ```
- [ ] Write tests in `crates/vox_script/src/rhai_runtime.rs` (or a separate test file):
  ```rust
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
          // Script that returns the dt arg back
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
          // Calling a non-existent function returns Err, must not panic
          let result = rt.call_fn(0, "nonexistent_fn", &[]);
          assert!(result.is_err());
      }
  }
  ```
- [ ] Run `cargo test -p vox_script` and confirm all three new tests pass.

---

## Task 4: Spatial Audio Listener Wiring

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`
- Modify: `crates/vox_audio/src/spatial.rs` (add test helpers if missing)

- [ ] Each frame, after `self.character.update(...)`, update the spatial audio listener:
  ```rust
  if self.character.enabled {
      self.spatial_audio.set_listener(
          self.character.camera_position(),
          self.character.camera_forward(),
          glam::Vec3::Y,
      );
  } else {
      // Use existing camera fields when character controller is off
      // Adapt field names to match actual EngineApp camera fields
      let cam_pos = glam::Vec3::from(self.camera_pos);  // or however your camera pos is stored
      let cam_forward = self.camera.forward();           // or equivalent
      self.spatial_audio.set_listener(cam_pos, cam_forward, glam::Vec3::Y);
  }
  ```
- [ ] Call `self.spatial_audio.tick(dt)` each frame (add this method to `SpatialAudioManager` if it doesn't exist — it should update per-sink volumes based on listener position):
  ```rust
  // In vox_audio/src/spatial.rs, if tick() doesn't exist:
  pub fn tick(&mut self, _dt: f32) {
      // Refresh attenuation for all active sinks
      for (sink_id, sink_entry) in &mut self.active_sinks {
          let (vol, _pan) = self.compute_spatial_for(sink_entry.position, sink_entry.base_volume);
          sink_entry.sink.set_volume(vol);
      }
  }
  ```
- [ ] Add a demo key binding: on `KeyO` (key 24), play a positioned ambient sound:
  ```rust
  if key as u32 == 24 {
      let _ = self.spatial_audio.play_3d(
          std::path::Path::new("assets/audio/ambient/wind_loop.ogg"),
          glam::Vec3::new(10.0, 0.0, 0.0),
          0.5,
          true,
      );
      println!("[ochroma] Playing demo 3D sound at (10, 0, 0)");
  }
  ```
- [ ] Add `new_silent()` constructor to `SpatialAudioManager` if it doesn't exist (creates a manager with no real audio output — for tests only):
  ```rust
  // In vox_audio/src/spatial.rs
  #[cfg(any(test, feature = "test-utils"))]
  pub fn new_silent() -> Self {
      // Minimal construction without real audio device
      Self {
          listener_pos: glam::Vec3::ZERO,
          listener_forward: glam::Vec3::NEG_Z,
          listener_up: glam::Vec3::Y,
          active_sinks: std::collections::HashMap::new(),
          next_id: 0,
      }
  }
  ```
- [ ] Add a `compute_spatial_for` method to `SpatialAudioManager` for testability (extract the volume/pan calculation from `play_3d`):
  ```rust
  pub fn compute_spatial_for(&self, source_pos: glam::Vec3, base_volume: f32) -> (f32, f32) {
      let dist = (source_pos - self.listener_pos).length().max(0.001);
      let ref_dist = 1.0_f32;
      let max_dist = 100.0_f32;
      let attenuation = (ref_dist / dist).min(1.0).max(0.0);
      let vol = (base_volume * attenuation).clamp(0.0, 1.0);
      // Pan: dot with listener right (simplified)
      let to_source = (source_pos - self.listener_pos).normalize();
      let right = self.listener_forward.cross(self.listener_up).normalize();
      let pan = right.dot(to_source).clamp(-1.0, 1.0);
      (vol, pan)
  }
  ```
- [ ] Write tests in `crates/vox_audio/src/spatial.rs`:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      use glam::Vec3;

      #[test]
      fn spatial_manager_volume_decreases_with_distance() {
          let mgr = SpatialAudioManager::new_silent();
          let (vol_near, _) = mgr.compute_spatial_for(Vec3::new(1.0, 0.0, 0.0), 1.0);
          let (vol_far, _) = mgr.compute_spatial_for(Vec3::new(100.0, 0.0, 0.0), 1.0);
          assert!(vol_near > vol_far, "volume should decrease with distance");
      }

      #[test]
      fn spatial_manager_set_listener_does_not_panic() {
          let mut mgr = SpatialAudioManager::new_silent();
          mgr.set_listener(Vec3::new(1.0, 0.0, 0.0), Vec3::NEG_Z, Vec3::Y);
          // No panic = pass
      }

      #[test]
      fn spatial_volume_at_max_distance_is_near_zero() {
          let mgr = SpatialAudioManager::new_silent();
          let (vol, _) = mgr.compute_spatial_for(Vec3::new(200.0, 0.0, 0.0), 1.0);
          assert!(vol < 0.01, "volume beyond max_dist should be near zero, got {}", vol);
      }
  }
  ```
- [ ] Run `cargo test -p vox_audio` and confirm all three new tests pass.

---

## Task 5: ECS World Save/Load

**Files:**
- Modify: `crates/vox_data/src/world_save.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Read the actual `EditorEntity` definition: `crates/vox_app/src/` (check `editor.rs`, `lib.rs`, or `engine_runner.rs` for the struct fields). Note the exact field names for `name`, `position`, `asset_path`.
- [ ] Add a `save_world` free function to `crates/vox_data/src/world_save.rs`. Because `EditorEntity` is in `vox_app` (which depends on `vox_data`, not the reverse), the function takes plain slices of `SavedEntity` — conversion happens in `engine_runner.rs`:
  ```rust
  impl WorldSave {
      /// Convenience constructor from already-converted SavedEntity list.
      pub fn from_entities(
          entities: Vec<SavedEntity>,
          camera_position: [f32; 3],
          camera_rotation: [f32; 4],
          time_of_day: f32,
      ) -> Self {
          WorldSave {
              version: 1,
              engine_version: env!("CARGO_PKG_VERSION").to_string(),
              timestamp: std::time::SystemTime::now()
                  .duration_since(std::time::UNIX_EPOCH)
                  .unwrap_or_default()
                  .as_secs_f64(),
              scene_name: "scene".into(),
              entities,
              resources: SavedResources {
                  time_of_day,
                  camera_position,
                  camera_rotation,
                  game_state: "playing".into(),
              },
          }
      }

      /// Default quick-save path.
      pub fn quick_save_path() -> std::path::PathBuf {
          std::path::PathBuf::from("saves/quicksave.json")
      }
  }
  ```
- [ ] In `engine_runner.rs`, add a helper method `fn build_world_save(&self) -> vox_data::world_save::WorldSave` on `EngineApp`:
  ```rust
  fn build_world_save(&self) -> vox_data::world_save::WorldSave {
      use vox_data::world_save::{SavedEntity, WorldSave};

      let cam_pos = [self.camera_pos.x, self.camera_pos.y, self.camera_pos.z]; // adapt field names
      let cam_rot = [0.0f32, 0.0, 0.0, 1.0]; // adapt if rotation is tracked

      let entities: Vec<SavedEntity> = self.editor.entities.iter().map(|e| SavedEntity {
          name: e.name.clone(),
          position: e.position.to_array(),  // adapt if Vec3 or [f32;3]
          rotation: [0.0, 0.0, 0.0, 1.0],
          scale: [1.0, 1.0, 1.0],
          asset_path: e.asset_path.clone().unwrap_or_default(),
          scripts: Vec::new(),
          tags: Vec::new(),
          custom_data: std::collections::HashMap::new(),
          collider: None,
          audio: None,
          light: None,
      }).collect();

      WorldSave::from_entities(entities, cam_pos, cam_rot, self.time_of_day)
  }
  ```
- [ ] Wire `F5` / `KeyF` (key 33) to save:
  ```rust
  if key as u32 == 33 {
      let ws = self.build_world_save();
      let path = vox_data::world_save::WorldSave::quick_save_path();
      if let Some(parent) = path.parent() {
          let _ = std::fs::create_dir_all(parent);
      }
      match ws.save_to_file(&path) {
          Ok(_) => println!("[ochroma] World saved to {}", path.display()),
          Err(e) => println!("[ochroma] Save failed: {}", e),
      }
  }
  ```
- [ ] Wire `F9` / `KeyL` (key 38) to load:
  ```rust
  if key as u32 == 38 {
      let path = vox_data::world_save::WorldSave::quick_save_path();
      match vox_data::world_save::WorldSave::load_from_file(&path) {
          Ok(ws) => {
              // Restore time of day
              self.time_of_day = ws.resources.time_of_day;
              // Restore camera
              let cp = ws.resources.camera_position;
              // self.camera_pos = glam::Vec3::new(cp[0], cp[1], cp[2]); // adapt
              // Restore entities (match by name, update position)
              for saved in &ws.entities {
                  if let Some(entity) = self.editor.entities.iter_mut().find(|e| e.name == saved.name) {
                      entity.position = glam::Vec3::from(saved.position); // adapt to actual type
                  }
              }
              println!("[ochroma] World loaded from {} ({} entities)", path.display(), ws.entities.len());
          }
          Err(e) => println!("[ochroma] Load failed: {}", e),
      }
  }
  ```
- [ ] Add the roundtrip test to `crates/vox_data/src/world_save.rs` (add `tempfile` to `[dev-dependencies]` of `vox_data` if not present):
  ```toml
  # In crates/vox_data/Cargo.toml under [dev-dependencies]:
  tempfile = "3"
  ```
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      use std::collections::HashMap;

      #[test]
      fn world_save_roundtrip() {
          let ws = WorldSave {
              version: 1,
              engine_version: "test".into(),
              timestamp: 0.0,
              scene_name: "test".into(),
              entities: vec![SavedEntity {
                  name: "cube".into(),
                  position: [1.0, 2.0, 3.0],
                  rotation: [0.0, 0.0, 0.0, 1.0],
                  scale: [1.0, 1.0, 1.0],
                  asset_path: "assets/cube.vxm".into(),
                  scripts: vec![],
                  tags: vec![],
                  custom_data: HashMap::new(),
                  collider: None,
                  audio: None,
                  light: None,
              }],
              resources: SavedResources {
                  time_of_day: 12.0,
                  camera_position: [0.0, 5.0, -10.0],
                  camera_rotation: [0.0, 0.0, 0.0, 1.0],
                  game_state: "playing".into(),
              },
          };
          let f = tempfile::NamedTempFile::new().unwrap();
          ws.save_to_file(f.path()).unwrap();
          let loaded = WorldSave::load_from_file(f.path()).unwrap();
          assert_eq!(loaded.entities.len(), 1);
          assert_eq!(loaded.entities[0].position, [1.0, 2.0, 3.0]);
          assert_eq!(loaded.entities[0].name, "cube");
          assert_eq!(loaded.resources.time_of_day, 12.0);
          assert_eq!(loaded.resources.camera_position, [0.0, 5.0, -10.0]);
      }

      #[test]
      fn world_save_from_entities_sets_version() {
          let ws = WorldSave::from_entities(vec![], [0.0; 3], [0.0, 0.0, 0.0, 1.0], 6.0);
          assert_eq!(ws.version, 1);
          assert_eq!(ws.resources.time_of_day, 6.0);
          assert_eq!(ws.scene_name, "scene");
      }
  }
  ```
- [ ] Run `cargo test -p vox_data` and confirm both tests pass.

---

## Final Validation

- [ ] Run `cargo build` from workspace root — zero errors.
- [ ] Run `cargo test` from workspace root — all tests pass.
- [ ] Manual smoke test: launch engine, press `P` to enable character controller, verify WASD movement and mouse-look work.
- [ ] Press `Space` to jump, verify gravity pulls character back down.
- [ ] Press `O` to trigger demo 3D sound (requires `assets/audio/ambient/wind_loop.ogg` to exist; graceful if missing).
- [ ] Press `F` to save, verify `saves/quicksave.json` is created with entity data.
- [ ] Press `L` to load, verify entity positions and time_of_day are restored.
- [ ] Confirm Rhai scripts in `scripts/` directory are auto-loaded and `on_update(dt)` is called each frame.
