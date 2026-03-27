# Ochroma — Unreal Parity Specification

Based on expert analysis: stop polishing the renderer. The renderer is already better than the engine around it. Focus on what makes developers choose an engine: workflow, iteration speed, and shipping games.

## Priority Order (by developer impact)

### 1. DOGFOOD: Build a 5-Minute Game
**Why:** An engine built without a game has blind spots. We cannot know what's missing until we try to ship.
**MVP:** A walking simulator with: menu screen, walk around a scene, collect items, win condition, sound effects, build to standalone .exe.
**What this forces us to build:** Everything else on this list.

### 2. Native Windows Build
**Why:** 95% of game devs are on Windows. WSL2 is not a deployment target.
**MVP:** `cargo build --target x86_64-pc-windows-msvc` produces a working .exe. Native Vulkan window, no Linux subsystems.

### 3. Real Asset Pipeline
**Why:** If a dev can't get art into the engine in 60 seconds, they quit.
**MVP:**
- Bulletproof PLY importer (tested with real 3DGS training output)
- CLI tool to convert GLTF mesh → splat cloud
- Hot-reload: change a .ply file → engine updates without restart

### 4. Visual Scene Editor
**Why:** Nobody builds levels by typing coordinates.
**MVP:** egui overlay on the viewport with:
- Scene hierarchy panel (entity tree)
- Property inspector (transform, scale, asset path)
- Translation/rotation gizmo in viewport (click + drag to move objects)
- Save/load scene to .json

### 5. Real Physics (Rapier)
**Why:** Games need collision, raycasting, character controllers.
**MVP:** Integrate rapier3d. Attach Box/Sphere/Capsule colliders to entities. Kinematic character controller that slides along walls. Raycasting for click-to-select.

### 6. Hot-Reload Scripting
**Why:** 30-second Rust compile kills iteration speed.
**MVP:** Either:
- Dynamic library hot-reload for GameScript trait
- Or embed Rhai/Lua for high-level gameplay logic
- Change a number, see it instantly

### 7. In-Game UI
**Why:** Every game needs menus, HUD, pause screen.
**MVP:** egui as game UI overlay. Scripts can draw buttons, text, health bars. Main menu → Play → Pause → Quit flow.

### 8. Real Audio
**Why:** Silent games feel broken.
**MVP:** Integrate Kira or rodio. Load .wav/.ogg, trigger from scripts, 3D spatial panning (left/right based on entity position relative to camera).

### 9. Basic Animation
**Why:** Static worlds are lifeless.
**MVP:** Rigid hierarchical animation — parent/child transforms. A car entity's wheels rotate. Basic keyframe playback.

### 10. Getting Started Documentation
**Why:** Devs evaluate engines by "time to first entity on screen."
**MVP:**
- README with build instructions (done)
- getting_started.md: spawn entity, attach asset, move with script
- One complete example game with heavy comments

## What Does NOT Matter Right Now
- More rendering features (we have enough)
- Multiplayer networking (ship single-player first)
- Visual scripting / Blueprints (code is fine for first 1000 users)
- AI/LLM integration (cool but not needed to ship games)
- Distributed simulation (premature scaling)
- Console ports (PC first)

## The Plan

Build these in order 1-10. Each one is blocked by the previous. The game (item 1) drives everything — as we build it, we discover what's actually broken and fix it.
