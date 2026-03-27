# Phase 5 — Engine Completion Specification

**Goal:** Transform Ochroma from a prototype into a production game engine capable of shipping a city builder in the Manor Lords / Cities: Skylines 2 class, and serving as a general-purpose engine that surpasses Unreal for Gaussian splatting workflows.

## Current State (After Phases 0-4 Integration)

**What's REAL:**
- 13 crates, 8,328 lines, 156 tests, all passing
- GPU rasteriser pipeline (wgpu + WGSL spectral shader)
- Software rasteriser fallback (CPU)
- Bevy ECS with frustum cull, LOD, gather systems running each frame
- Interactive camera (orbit/pan/zoom/WASD)
- Click-to-place asset spawning with ground ray cast
- Terrain ground plane (grass splats)
- Simulation tick (citizens age/die, budget collects taxes, traffic flows, agents move)
- egui UI with asset browser, budget panel, zone/service/road tools
- Time-of-day illuminant blending
- Save/load game files
- Undo/redo command stack
- Proc-GS rule emission (deterministic building generation)
- Zoning → building growth system
- Road surface splat generation
- LWC coordinate system
- Tile streaming state management

**What's MISSING to ship a real city builder:**

### Rendering Gaps
1. **Spectra integration** — Our WGSL shader is basic. Spectra has production-grade Gaussian splatting with OptiX denoising, DLSS, advanced lighting
2. **Shadow rendering** — Shadow catchers exist but aren't in the render pipeline
3. **Multi-scale LOD transitions** — LOD levels defined but no opacity crossfade between levels
4. **Particle/VFX system** — Types exist but no emitters running
5. **Post-processing** — No bloom, no tone mapping, no ambient occlusion

### Simulation Gaps
6. **Citizen daily routines** — Citizens age/die but don't commute, work, shop, or have schedules
7. **Building functionality** — Buildings spawn but don't employ citizens, produce goods, or provide services
8. **Resource flow** — SupplyChain exists but isn't connected to buildings
9. **Dynamic land value** — No land value tensor affecting growth patterns
10. **District policies** — Types exist but no policy application
11. **Migration** — No citizens arriving/departing based on satisfaction

### Gameplay Gaps
12. **Road drawing tool** — Bezier math exists but no interactive drawing UX (click-drag to draw)
13. **Zoning painting** — Zone action exists but no drag-to-paint area selection
14. **Service building placement** — Action exists but doesn't create ServiceManager entries
15. **Utility network laying** — UtilityNetwork exists but no tool to lay pipes/wires
16. **Transport route creation** — TransportManager exists but no route drawing tool
17. **Overlay system** — No visual overlays (traffic heat, land value, coverage, pollution)
18. **Map generation** — No procedural terrain (hills, rivers, coastlines)

### Engine Gaps
19. **Real audio playback** — AudioEngine manages sources but no actual sound output (needs cpal/rodio)
20. **Real physics integration** — PhysicsWorld ticks but isn't connected to ECS entities
21. **Wasm mod runtime** — ScriptRuntime has events but no actual wasmtime integration
22. **Multiplayer networking** — NetMessage protocol exists but no TCP/QUIC transport
23. **Animation system** — No skeletal animation, no Gaussian bone binding
24. **AR/VR** — No OpenXR integration

### Production Gaps
25. **Error recovery** — No graceful handling of corrupted saves, missing assets, etc.
26. **Performance profiling** — puffin integrated but no GPU timing, no VRAM tracking
27. **Asset hot-reloading** — No file watching, no live reload
28. **Modding documentation** — No mod API docs
29. **Localisation** — No i18n support
30. **Accessibility** — No colour-blind modes, no screen reader support

---

## Prioritised Implementation Order

### Wave 1: Make It Playable (Critical Path)

These make the difference between "tech demo" and "playable game":

**W1.1: Interactive road drawing** — Click start point, drag control point, click end point. Road appears with asphalt splats. Connected to traffic network.

**W1.2: Zoning paint tool** — Click-drag to paint zone areas. Visual feedback showing zone colour. Connected to growth system.

**W1.3: Service placement wiring** — Clicking in Service mode creates a real ServiceBuilding in SimulationState. Coverage calculated. Citizens' needs updated.

**W1.4: Citizen daily routines** — Citizens wake up, go to work (nearest commercial/industrial building), come home. Generates agent movement. Commute time affects satisfaction.

**W1.5: Building functionality** — Residential buildings provide housing capacity. Commercial buildings provide jobs + goods. Industrial buildings provide jobs + raw materials.

**W1.6: Resource flow** — Timber from trees → sawmill → planks. Wheat from farms → bakery → bread. Market distributes to citizens.

**W1.7: Overlay system** — Toggle views: traffic density (red=congested), land value (green=high), service coverage (blue circles), zone colours.

**W1.8: Map generation** — Procedural terrain with hills, flat areas, a river. Player picks a map before starting.

### Wave 2: Make It Feel Good (Polish)

**W2.1: Audio integration** — cpal backend. Ambient sounds based on camera position. UI click sounds. Construction sounds.

**W2.2: Shadow rendering** — Shadow catcher meshes submitted to a shadow pass. Buildings cast shadows on terrain.

**W2.3: LOD crossfade** — Opacity transition over 2 frames when switching LOD levels.

**W2.4: Particle effects** — Smoke from chimneys, construction dust, rain/snow.

**W2.5: Post-processing** — ACES tone mapping, bloom on bright surfaces, ambient occlusion.

**W2.6: Better UI** — Notification popups, milestone celebrations, advisor messages, mini-map.

### Wave 3: Make It Deep (Simulation Depth)

**W3.1: Land value tensor** — 2D grid updated when buildings/services placed. Affects growth patterns.

**W3.2: District policies** — Player draws districts, sets per-district tax rates and policies.

**W3.3: Migration** — Citizens arrive/leave based on city satisfaction vs regional average.

**W3.4: Transport integration** — Player draws bus/tram routes. Citizens use transit. Revenue from fares.

**W3.5: Utility simulation** — Water/power/sewage must be connected. Brownouts if power insufficient.

**W3.6: Seasons** — Visual changes (snow, autumn leaves). Gameplay effects (heating costs, crop cycles).

### Wave 4: Engine Generalisation

**W4.1: Spectra renderer integration** — Use Spectra's Gaussian splatting pipeline for production rendering.

**W4.2: Wasmtime mod runtime** — Load and execute Wasm modules with sandboxed API.

**W4.3: QUIC networking** — Real multiplayer with authoritative server.

**W4.4: Skeletal animation** — Bone-driven Gaussian clusters for citizen/vehicle animation.

**W4.5: OpenXR VR output** — Stereo rendering, hand tracking.

---

## Exit Criteria

The engine is **complete** when ALL of the following work end-to-end:

### Playability
- [ ] Player starts the app, sees procedural terrain with hills and a river
- [ ] Player draws curved roads with mouse
- [ ] Player paints residential/commercial/industrial zones along roads
- [ ] Buildings grow on zoned plots over game-time
- [ ] Citizens spawn in residential buildings and commute to workplaces
- [ ] Budget screen shows live income/expenses, player adjusts tax rates
- [ ] Service buildings (schools, hospitals, fire, police) placed and show coverage
- [ ] Traffic visible on roads, congestion propagates
- [ ] Resource flow: trees → sawmill → market → citizens
- [ ] Toggle overlays: traffic, land value, coverage, zones

### Visuals
- [ ] Day/night cycle changes lighting (dawn golden, night dark with artificial lights)
- [ ] Weather affects surfaces (rain makes things shiny, snow covers horizontal surfaces)
- [ ] Shadows from buildings on terrain
- [ ] Particle effects: chimney smoke, construction dust
- [ ] Smooth LOD transitions (no popping)
- [ ] Audio: ambient city sounds, construction, nature

### Persistence
- [ ] Save game to disk, load it back with full state preserved
- [ ] Undo/redo placement actions with Ctrl+Z/Y
- [ ] Auto-save every 5 minutes

### Performance
- [ ] 200k splats at 1080p ≥ 60fps (Phase 0 target)
- [ ] 5M splats at 1080p ≥ 60fps (Phase 1 target)
- [ ] 100,000 citizens simulated without frame regression
- [ ] Tile streaming loads/unloads without visible pop-in

### Robustness
- [ ] App runs for 30 minutes without crashing
- [ ] Corrupted save file shows error, doesn't crash
- [ ] Missing asset shows placeholder, doesn't crash
- [ ] Window resize works without artefacts
