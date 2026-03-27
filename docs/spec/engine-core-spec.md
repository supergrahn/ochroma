# Ochroma Engine — Core Specification

**This is the engine.** Everything else is a game built on top of it.

## What the Engine Does

### 1. Asset System
- **Load** .ply files (standard 3DGS format from any Gaussian splat tool)
- **Load** .vxm files (Ochroma's native format with spectral data)
- **Asset registry** — UUID-indexed, tag-searchable, hot-reloadable
- **Asset bundles** — pack multiple assets into a single file for distribution

### 2. Scene System
- **Scene file** (.ochroma_scene) — JSON describing entity hierarchy
- **Entity** — position, rotation, scale, attached components
- **Components** — data attached to entities (mesh, script, collider, audio source, light, etc.)
- **Prefabs** — reusable entity templates

### 3. Game Object Model
- Entity-Component architecture (Bevy ECS)
- **Built-in components:**
  - `Transform` — position, rotation, scale
  - `GaussianRenderer` — renders a .ply/.vxm asset
  - `Collider` — AABB/sphere collision shape
  - `RigidBody` — physics simulation
  - `AudioSource` — plays sound
  - `Light` — point/directional/spot light
  - `Camera` — view into the scene
  - `Script` — attached game logic

### 4. Scripting
- Game logic written in **Rust** (native, fastest)
- Or in **Wasm** (sandboxed mods)
- Script lifecycle: `on_start()`, `on_update(dt)`, `on_destroy()`
- Scripts can: query/modify entities, spawn/destroy entities, read input, play audio, cast rays

### 5. Input System
- Unified input map: physical keys → game actions
- Rebindable at runtime
- Mouse, keyboard, gamepad, touch support
- Input consumed by UI first, then game

### 6. Rendering
- Gaussian splat rendering via Spectra EWA algorithm (or wgpu shader)
- Spectral pipeline: 8-band SPD → CIE XYZ → sRGB
- Multiple illuminants (time of day)
- Camera system: perspective, orthographic
- Post-processing: tone mapping, bloom, vignette

### 7. Physics
- AABB and sphere collision detection
- Rigid body dynamics (gravity, forces)
- Ray casting (for picking, line of sight)
- Collision callbacks to scripts

### 8. Audio
- Load .wav/.ogg files
- 3D spatial audio with distance attenuation
- Play/stop/loop controls
- Volume, pitch adjustment

### 9. Editor
- Place entities in the scene
- Transform gizmo (move, rotate, scale)
- Property inspector for components
- Asset browser
- Play/Stop to test the game
- Save/load scenes

### 10. Build & Run
- `cargo run` starts the engine with a scene
- Load scene → create entities → run scripts → render → handle input → repeat
- Package game as standalone executable

## What the Engine Does NOT Do
- Generate assets (that's a tool, not the engine)
- Simulate cities (that's a game, not the engine)
- Manage citizens/economy/traffic (that's game logic)
- Run LLMs (that's an external tool)

The engine is the PLATFORM. Games are built ON it.
