# Ochroma Full Gap Closure Specification

## Current Reality

The codebase has type definitions and isolated algorithms but **nothing works end-to-end**:
- The app crashes on GPU init (wgpu device creation fails on WSL2)
- ECS systems exist but are never scheduled
- Simulation tick methods exist but are never called
- UI panels render but have no event handlers
- 7,219 lines across 103 files, ~90% orphaned

## Architecture Change: Spectra Integration

**Decision:** Replace our homegrown wgpu rasteriser with the Spectra renderer at `~/git/aetherspectra/spectra`.

**Why:** Spectra is a production-grade spectral path tracer with:
- 3D Gaussian Splatting tile rasteriser (`engine/gaussian_renderer.py`)
- Spectral rendering (380-780nm, hero wavelength strategy)
- Real-time mode with OptiX denoiser + DLSS
- USD scene format support
- 80+ Rust crates we can call directly

**Integration approach:**
- Spectra's Rust crates (`spectra-gaussian-render`, `spectra-accel`, etc.) become workspace dependencies
- Ochroma's game loop calls Spectra's rendering API directly
- Scene data flows: Ochroma ECS → Spectra SceneState → GPU
- Our WGSL shader becomes a fallback for non-NVIDIA hardware

## What Must Be Built (Ordered by Priority)

### Priority 1: Make It Run

Before anything else, the app must start and render pixels.

**1.1 Fix GPU initialisation**
- The wgpu device creation crashes on WSL2. Two paths:
  - Path A: Request software adapter (`wgpu::Backends::GL` with `llvmpipe`)
  - Path B: Add graceful fallback — try Vulkan, fall back to GL, fall back to CPU software rasteriser
- The app MUST start without crashing on any machine

**1.2 Verify rendering output**
- After GPU init works, the two demo buildings must appear on screen
- The spectral-to-RGB conversion must produce correct colours (brick = reddish, slate = grey)
- Camera orbit must work smoothly

**1.3 FPS measurement**
- puffin must be wired into the render loop
- Frame time must be printed/displayed
- Target: 200k splats at 60fps for Phase 0 exit

### Priority 2: Make It a Game Loop

**2.1 Bevy ECS integration**
- Create a `bevy_ecs::World` in main
- Spawn `SplatAssetComponent` and `SplatInstanceComponent` entities
- Schedule `frustum_cull_system`, `lod_select_system`, `gather_splats_system` to run each frame
- The render loop reads `VisibleSplats` resource and submits to GPU

**2.2 Input handling**
- Camera control: mouse orbit (middle-drag), pan (right-drag), zoom (scroll)
- Click detection: ray cast from mouse position through frustum to world
- Mode switching: keyboard shortcuts for place/select mode

**2.3 Game simulation tick**
- A `SimulationState` resource holding:
  - `CitizenManager`
  - `CityBudget`
  - `ZoningManager`
  - `ServiceManager`
  - `TrafficNetwork`
  - `AgentManager`
- A `simulation_tick_system` that calls all `.tick()` methods each frame
- Game speed controls: pause, 1x, 2x, 4x

### Priority 3: Make It Interactive

**3.1 egui integration into render loop**
- `egui_winit::State` processes window events
- `egui_wgpu::Renderer` renders UI on top of the 3D viewport
- The plop UI's `show()` method runs each frame

**3.2 Asset placement**
- Click terrain in Place mode → spawn new `SplatInstanceComponent`
- The spawned entity's splats appear in the next frame
- The entity is added to the spatial hash

**3.3 Asset selection**
- Click an asset in Select mode → entity ID buffer lookup
- Selected entity highlighted (outline or tint)
- Spectral shift sliders modify the selected instance's wear/colour

**3.4 Terrain rendering**
- Generate terrain splats from `TerrainPlane`
- Render as a ground plane beneath buildings
- Support multiple surface materials (grass, cobblestone, asphalt)

### Priority 4: Make It Simulate

**4.1 Zoning → building growth**
- Player paints zones on terrain (click-drag)
- `ZoningManager` creates `ZonePlot` entries
- Growth system checks demand, selects Proc-GS rules, spawns buildings over game-time
- Construction animation: buildings fade in from transparent to opaque

**4.2 Citizens**
- Citizens spawn when residential buildings are complete
- `CitizenManager::tick()` runs each game tick
- Citizens age, change lifecycle stage, seek employment, die
- Migration: citizens arrive if satisfaction > threshold, leave if below

**4.3 Economy**
- `CityBudget::tick()` runs each game tick
- Income from taxes (residential × rate × citizen count)
- Expenses from services (each service building has operational cost)
- Budget display in UI

**4.4 Roads → traffic**
- Player draws roads via Bezier tool
- Roads update the `TrafficNetwork`
- `TrafficNetwork::tick()` runs each game tick
- Vehicle density rendered on road segments

**4.5 Services → coverage**
- Player places service buildings
- Coverage radius visualised as overlay
- Citizens within coverage get needs fulfilled

### Priority 5: Make It Complete

**5.1 Save/Load**
- Serialise the entire `SimulationState` + all ECS entities
- Write to `.ochroma_save` with zstd compression
- Load restores full game state

**5.2 Undo/Redo**
- Every placement/zone/road action pushes to `UndoStack`
- Ctrl+Z undoes, Ctrl+Y redoes
- Undo stack saved with game state

**5.3 Utilities**
- Water, power, sewage networks as graph overlays
- Buildings require all three to function
- Deficit warnings in UI

**5.4 Transport**
- Player creates bus/tram routes
- Vehicles visible on routes
- Revenue from fares in budget

**5.5 Time-of-day + Weather**
- Illuminant shifts: D65 → D50 → A as day progresses
- Weather affects SPDs in real-time
- Seasons cycle with gameplay effects

## What We're Deferring

These are explicitly NOT in this plan — they require external model integration or hardware we don't have:

- **LLM integration** (Neural Layout Interpreter needs a real model)
- **Diffusion model infill** (needs trained model)
- **Lyra video capture** (needs NVIDIA's pipeline)
- **AR/VR** (needs OpenXR + headset)
- **DLSS/neural denoiser** (needs OptiX)
- **CRDT multiplayer** (needs QUIC transport + server infrastructure)
- **Wasm scripting** (needs wasmtime integration — significant effort)
- **Steam integration** (needs Steamworks SDK)
- **Skeletal animation** (needs animation format + blending system)

These are real features that require real external dependencies. We'll add them in focused follow-up specs rather than pretending they're done.

## Exit Criteria (What "Done" Actually Means)

The game is **done** when a player can:

1. Start the app and see a terrain ground plane
2. Select an asset from the browser and click to place it
3. Draw roads with curves
4. Paint zones along roads
5. Watch buildings grow on zoned plots
6. See citizens spawned in residential buildings
7. Open the budget screen and adjust tax rates
8. Place service buildings and see coverage overlays
9. See traffic density on busy roads
10. Watch the day/night cycle change lighting
11. Save the game, close, reopen, and load the save
12. Undo a placement with Ctrl+Z
13. Run for 30 minutes without crashing

Every one of these must actually work — not just compile.
