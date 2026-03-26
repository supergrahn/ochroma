# Phase 4 — Ship the City Builder

**Goal:** A shippable city builder that competes with Manor Lords and Cities: Skylines 2.
Phases 0–3 deliver the rendering engine and AI generation pipeline. Phase 4 delivers the
**game** — every system a player expects when they buy a city builder on Steam.

Builds on Phase 3's 100km world, agent simulation, streaming, and procedural zoning.

All engine-layer work stays game-agnostic in `vox_core`, `vox_render`, `vox_data`. All
city-builder-specific logic lives in `vox_sim` and `vox_app`.

---

## Scope Boundary

### In scope (Phase 4)

**Engine layer (game-agnostic):**
- CRDT networking crate (`vox_net`)
- Wasm scripting runtime (`vox_script`)
- Neural denoiser for spectral noise at high Gaussian density
- Skeletal animation system (bone-driven Gaussian cluster transforms)
- Spatial audio engine (`vox_audio`)
- AR/VR rendering paths (OpenXR — Vision Pro, Quest)
- Engine-level state snapshots and rollback (Neural Undo)
- General-purpose physics (`vox_physics`) — rigid body, collision
- Particle / VFX system — Gaussian-native emitters (fire, smoke, dust)
- Terrain engine — heightmap with elevation, rivers, coastlines
- Water simulation — surface flow, flooding, rainfall drainage
- General-purpose navmesh (replaces road-graph-only pathfinding)
- Camera system — orbit, pan, zoom, cinematic, constraints
- Input abstraction — mouse, keyboard, gamepad, touch, VR
- Save/load — serialise full world state to disk
- Undo/redo — command pattern with full state rollback
- Proper UI framework (replace egui dev tools with game-quality UI)

**Game layer (city-builder-specific, `vox_sim` + `vox_app`):**
- Interactive road tools (draw curves, intersections, bridges, tunnels, roundabouts)
- Zoning system (residential, commercial, industrial, office, mixed-use)
- Utilities infrastructure (water, power, sewage pipe networks)
- Public transport (bus routes, rail, metro, tram, stations, stops)
- City services (education, healthcare, fire, police — coverage radius model)
- Budget / taxation / finance (income tax, property tax, bonds, loans, expenses)
- Citizen lifecycle (birth, education, employment, aging, death, migration)
- Resource gathering and supply chains (raw materials → processing → goods → market)
- Advanced economy (dynamic pricing, supply/demand, imports/exports, trade routes)
- Agent social graphs (relationships, satisfaction, needs hierarchy, protests)
- District policies (rent control, tax incentives, noise ordinances, speed limits)
- City progression (tech eras, milestones, unlockable buildings)
- Seasons with gameplay impact (heating costs, crop cycles, snow removal, flooding)
- Map editor / terrain sculpting
- Game UI (not dev tools — menus, overlays, info panels, graphs, advisors)
- Tutorials and onboarding
- Steam integration (achievements, workshop for mods, cloud saves)
- Music and ambient soundscapes

### Explicitly out of scope for Phase 4
- Engine SDK / public API packaging (Phase 5)
- General-purpose editor for non-city games (Phase 5)
- Material graph / visual node editor (Phase 5)
- Second game proving engine generality (Phase 5)
- Console ports (Phase 5)

---

## New Crates

```
ochroma/
├── crates/
│   ├── vox_core/       (exists)
│   ├── vox_render/     (exists) + denoiser, VFX, AR/VR paths
│   ├── vox_data/       (exists) + save/load serialisation
│   ├── vox_app/        (exists) + game UI, tutorials, Steam integration
│   ├── vox_sim/        NEW — city simulation: economy, citizens, services, zoning
│   ├── vox_nn/         (exists from Phase 2) + denoiser model
│   ├── vox_net/        NEW — CRDT networking, entity replication, lobby
│   ├── vox_script/     NEW — Wasm scripting runtime, mod API
│   ├── vox_audio/      NEW — spatial audio, music, ambient
│   ├── vox_physics/    NEW — rigid body, collision, fluid
│   ├── vox_terrain/    NEW — heightmap, erosion, water table, sculpting
│   ├── vox_ui/         NEW — game-quality UI framework (replaces raw egui)
│   └── vox_tools/      (exists) + map editor, mod packaging tools
```

---

## 1. Terrain Engine (`vox_terrain`)

Replaces the Phase 1 flat ground plane. The terrain is the foundation everything else
sits on.

```
Heightmap: 4096×4096 per 1km tile (0.25m resolution)
Data: f32 height + u8 surface_type per sample
Surface types: grass, dirt, rock, sand, snow, water_bed
```

### Elevation and Sculpting

```
Sculpt tools (available in map editor and during gameplay):
  - Raise / lower (brush radius + strength)
  - Flatten to height
  - Smooth
  - Plateau (flat top, sloped sides)
  - Cliff stamp (vertical face)

All sculpt operations update:
  1. Heightmap
  2. Surface Gaussian density (re-scatter terrain splats in affected area)
  3. Water flow (recalculate drainage after terrain change)
  4. Navmesh (rebuild affected navmesh tiles)
  5. Road grade (flag roads on too-steep slopes for warning)
```

### Rivers and Water Bodies

```
Water defined by:
  - Water table height per tile (global parameter)
  - Any terrain below water table = submerged
  - Rivers: spline-defined channels carved into heightmap
  - Lakes: depressions below water table
  - Coastlines: world-edge water table

Water surface: Gaussian sheet at water table height
  - SPD: water_still or water_flowing based on flow velocity
  - Specular: environment reflection via Gaussian reflection probe
  - Transparency: depth-dependent opacity fade
```

### Seasonal Terrain

```
Spring:  mud_wet SPD on paths, grass growth density increases
Summer:  dry soil patches, full vegetation
Autumn:  leaf_fall particle emitter on deciduous vegetation
Winter:  snow_accumulation Gaussian layer on horizontal surfaces
         ice on water surfaces (water SPD → ice SPD transition)
```

---

## 2. Interactive Road Tools

Phase 2 generates road graphs from prompts. Phase 4 adds player-drawn roads.

```
Road tool modes:
  - Straight segment
  - Curved segment (Bezier with adjustable control points)
  - Roundabout (place centre, set radius)
  - Intersection (auto-generated when roads cross)
  - Bridge (elevated segment, terrain clearance check)
  - Tunnel (underground segment, bore through terrain)

Road types:
  - Dirt path         (1 lane, no markings, low capacity)
  - Local street      (2 lanes, low speed)
  - Avenue            (4 lanes, median, trees)
  - Highway           (6 lanes, grade-separated interchanges)
  - Rail track        (dedicated, no road traffic)

Each road segment:
  - Generates surface Gaussians via Proc-GS tileable panels
  - Updates the navmesh
  - Updates the road graph for traffic simulation
  - Auto-places kerbs, markings, drainage based on road type
  - Snaps to terrain elevation, auto-grades slopes
```

---

## 3. Zoning System

```
Zone types:
  - Residential (low / medium / high density)
  - Commercial (local / regional)
  - Industrial (light / heavy)
  - Office
  - Mixed-use (commercial ground floor, residential above)
  - Agricultural (crops, livestock — for Manor Lords-style play)
  - Parks / recreation

Zone placement:
  Player paints zones on plots adjacent to roads
    → Growth model selects Proc-GS rules based on:
       - Zone type + density
       - Land value tensor
       - Adjacent zone types
       - Era / tech level
       - District policy modifiers
    → Buildings grow over game-time (construction animation)
    → Demand meter: R/C/I demand drives growth rate
```

---

## 4. Utilities Infrastructure

```
Three network types, all simulated as directed graphs on the road/pipe layout:

Water:
  - Water source (river intake, well, desalination plant)
  - Water treatment plant
  - Pipe network (follows road layout by default, can be custom-routed)
  - Water tower (pressure buffer)
  - Coverage: buildings within N metres of a pressurised pipe = served

Power:
  - Power plant (coal, gas, nuclear, solar, wind, hydro)
  - Transmission lines (high voltage, long distance)
  - Distribution network (follows roads)
  - Coverage: buildings within N metres of distribution line = served
  - Generation must meet or exceed consumption (brownouts if deficit)

Sewage:
  - Sewage pipe network (gravity-fed, follows road layout)
  - Sewage treatment plant
  - Outfall (treated water discharged to river — affects water quality downstream)
  - Coverage: buildings within N metres of sewer pipe = served

All three networks use the same graph simulation:
  - Capacity per edge (pipe diameter, wire gauge)
  - Flow per edge (consumption-driven)
  - Bottleneck detection (highlight under-capacity segments)
  - Failure propagation (break a node → downstream loses service)
```

---

## 5. City Services

```
Service types and coverage model:

Education:
  - Primary school     (coverage: 1km radius, capacity: 500 students)
  - Secondary school   (coverage: 2km radius, capacity: 1000 students)
  - University         (coverage: city-wide, capacity: 5000 students)
  - Effect: educated citizens → higher skill jobs → higher tax revenue

Healthcare:
  - Clinic             (coverage: 1km, capacity: 200 patients/day)
  - Hospital           (coverage: 3km, capacity: 1000 patients/day)
  - Effect: health coverage → longer lifespan → population stability

Fire:
  - Fire station       (coverage: 2km, response time based on road distance)
  - Effect: fire risk reduction, disaster response

Police:
  - Police station     (coverage: 2km)
  - Effect: crime reduction → land value increase

All services:
  - Require staffing (citizens with appropriate education level)
  - Consume budget (operational cost per building per month)
  - Visualised as coverage overlay on the map
```

---

## 6. Budget and Economy

```
Income:
  - Residential tax    (per citizen, scaled by property value)
  - Commercial tax     (per business, scaled by revenue)
  - Industrial tax     (per factory, scaled by output)
  - Import/export tariffs
  - Public transport fares

Expenses:
  - Service buildings  (operational cost)
  - Infrastructure maintenance (roads, pipes, power lines degrade over time)
  - Loan repayments
  - Emergency spending (disasters, shortfalls)

Financial tools:
  - Tax rate sliders per zone type and district
  - Bonds (borrow against future revenue, interest accrues)
  - Budget graph (income vs expense over time)
  - Advisor warnings (deficit, debt ceiling, service underfunding)
```

### Supply Chains (Manor Lords influence)

```
Raw materials:
  - Timber (forest zones), Stone (quarry), Iron (mine), Clay (pit)
  - Food: wheat, vegetables, livestock (agricultural zones)

Processing:
  - Sawmill: timber → planks
  - Smithy: iron → tools, weapons
  - Bakery: wheat → bread
  - Market: distributes goods to citizens

Flow:
  Resource node → processing building → market/warehouse → citizen consumption
  Each step requires workers and transport (carts on road network)
  Bottlenecks visible: stockpile full, no workers, road congested
```

---

## 7. Citizen Simulation

Replaces Phase 3's basic pedestrian agents with full lifecycle citizens.

```rust
// Engine layer: generic agent (vox_core)
struct Agent {
    id: u32,
    position: WorldCoord,
    velocity: Vec3,
    navmesh_path: NavPath,
    animation_state: AnimationState,
}

// Game layer: citizen (vox_sim)
struct Citizen {
    agent: AgentId,              // references engine-layer agent
    age: f32,                    // years
    lifecycle: LifecycleStage,   // Child, Student, Worker, Retired
    education: EducationLevel,   // None, Primary, Secondary, University
    employment: Option<BuildingId>,
    residence: Option<BuildingId>,
    satisfaction: f32,           // 0.0–1.0
    needs: Needs,                // Maslow-like hierarchy
    social_links: Vec<CitizenId>,
}

enum LifecycleStage { Child, Student, Worker, Retired }

struct Needs {
    housing: f32,      // has a home, quality of home
    food: f32,         // access to food supply
    health: f32,       // access to healthcare
    safety: f32,       // police coverage, crime rate
    education: f32,    // access to schools for children
    employment: f32,   // has a job, commute distance
    leisure: f32,      // parks, entertainment nearby
}
```

### Lifecycle Flow

```
Birth (residential building, probability based on satisfaction + housing)
  → Child (0–6): stays near home, needs: housing, food, safety
  → Student (6–18): attends nearest school with capacity
  → Worker (18–65): seeks employment matching education level
      - Commutes daily (generates traffic on road network)
      - Pays taxes
      - Satisfaction drives migration (unhappy → leaves city)
  → Retired (65+): no employment, consumes healthcare
  → Death: natural (age) or event (disaster, no healthcare)

Migration:
  - Citizens migrate IN if city satisfaction > threshold and housing available
  - Citizens migrate OUT if satisfaction < threshold for N months
  - Migration rate drives city growth/decline
```

---

## 8. Public Transport

```
Transport types:
  - Bus: follows road network, player-defined routes + stops
  - Tram: dedicated track on road median, player-defined routes
  - Metro: underground network, stations connect to surface
  - Rail: inter-district, high capacity, dedicated tracks

Route definition:
  Player draws route on map → places stops/stations → assigns vehicles
  → Citizens evaluate: walk to stop + wait + ride + walk to destination
     vs. drive/walk entire distance
  → If transit is faster or citizen has no car: use transit

Simulation:
  - Vehicles are instanced .vxm assets on the route path
  - Capacity per vehicle (bus: 40, tram: 100, metro car: 200)
  - Frequency adjustable by player
  - Revenue from fares, costs from operation
  - Overcrowding visible (citizens waiting at stops)
```

---

## 9. Spatial Audio (`vox_audio`)

Engine-layer. No city-specific concepts.

```
Architecture:
  - Audio graph: source nodes → effect nodes → spatial nodes → listener
  - Sources: .ogg/.wav clips, procedural generators (wind, rain, fire)
  - Spatial: 3D position in SVO, distance attenuation, occlusion via SVO rayCast
  - Reverb: computed from SVO density around listener (open field vs. narrow street)
  - Output: stereo, surround, binaural (for VR)

Integration with spectral engine:
  - Material properties include acoustic absorption coefficient
  - Sound bouncing off glass_clear vs brick_red produces different reverb character
  - This is unique to Ochroma — no other engine ties material physics to audio

Performance:
  - Max 64 simultaneous sources
  - Occlusion queries batched per frame, cached for 4 frames
  - Music and ambient on separate non-spatial bus
```

### City builder audio (game layer, `vox_app`):

```
Ambient layers (blended by camera position + time + weather):
  - Urban hum (traffic density → volume)
  - Nature (parks, forests, water)
  - Weather (rain, wind, thunder)
  - Time of day (birds at dawn, crickets at night)

Building sounds:
  - Construction (hammering, machinery — during build animation)
  - Industrial (factory hum, smoke stack)
  - Commercial (market chatter)
  - Emergency (sirens from fire/police when responding)

Music:
  - Adaptive soundtrack: intensity scales with city growth milestones
  - Era-appropriate instrumentation
```

---

## 10. Physics (`vox_physics`)

Engine-layer. Gaussian-native where possible.

```
Rigid body:
  - Collision shapes derived from Gaussian cloud convex hull (same as Shadow Catchers)
  - Dynamic objects: vehicles, debris, falling trees
  - Static objects: buildings, terrain, roads
  - Solver: position-based dynamics (XPBD) — GPU-friendly, stable

Fluid (simplified):
  - Water flow on terrain: shallow water equations on heightmap grid
  - Flooding: rain accumulation → water table rise → terrain submerged
  - River current: velocity field on water surface Gaussians

Destruction (extends Phase 3 negative Gaussians):
  - Impact point → negative Gaussian sphere
  - Debris: positive Gaussian clusters ejected with rigid body dynamics
  - Settles: debris Gaussians come to rest, become static
  - Performance: max 10,000 debris Gaussians active simultaneously
```

---

## 11. Skeletal Animation

Engine-layer. Drives Gaussian clusters via bone transforms.

```
Skeleton: tree of bones, each with local transform
Animation clip: keyframed bone transforms, sampled at 30fps
Blending: linear blend between clips (walk → run, idle → wave)

Gaussian binding:
  - Each Gaussian in a .vxm is bound to a bone via bone_id (u8)
  - At runtime: Gaussian world position = bone_world_transform × local_position
  - Skinning: single bone per Gaussian (no dual-quaternion needed at splat scale)

Animation types for city builder:
  - Citizen: idle, walk, run, sit, work (per job type)
  - Vehicle: wheel rotation, door open/close
  - Construction: crane swing, scaffold build-up
  - Nature: tree sway (wind-driven procedural), leaf fall
```

---

## 12. Wasm Scripting (`vox_script`)

Engine-layer. The mod system.

```
Runtime: wasmtime (Rust-native, JIT, sandboxed)

API exposed to Wasm modules:
  - Entity queries (read ECS components)
  - Entity mutation (spawn, despawn, modify components)
  - Event subscription (on_build, on_zone, on_citizen_born, etc.)
  - UI extension points (add panels, overlays, buttons)
  - Asset registration (register new .vxm assets, new Proc-GS rules)
  - Audio triggers

Sandboxing:
  - No filesystem access (assets loaded via engine API only)
  - No network access
  - CPU budget: 2ms per frame per mod (kill if exceeded)
  - Memory budget: 64MB per mod

Mod packaging:
  - .ochroma_mod archive: manifest.toml + .wasm + assets
  - Steam Workshop integration for distribution
  - Load order and conflict resolution via manifest dependencies
```

---

## 13. CRDT Networking (`vox_net`)

Engine-layer. Generic entity replication.

```
Architecture:
  - Authoritative server (can be player-hosted or dedicated)
  - CRDT state: each entity component is a CRDT register
  - Conflict resolution: last-writer-wins for most components,
    merge for inventory/resource counters
  - Transport: QUIC (UDP-based, encrypted, multiplexed)

Replication:
  - Server owns simulation tick
  - Clients send input (place road, zone, adjust budget)
  - Server validates, applies, broadcasts delta
  - Clients apply delta to local ECS
  - Interest management: clients only receive entities in their viewport + buffer

Scale target: 1,000 concurrent players per server
Tick rate: 10Hz for city sim, 30Hz for agent movement

City builder specifics (game layer):
  - Per-player permissions (mayor, councillor, spectator)
  - Voting system for major decisions (budget, zoning policy)
  - Chat
```

---

## 14. AR/VR Rendering

Engine-layer via OpenXR.

```
Targets:
  - Meta Quest 3 (standalone, reduced LOD budget)
  - Apple Vision Pro (high quality, passthrough AR)
  - SteamVR (PC tethered, full quality)

Rendering changes:
  - Stereo rendering: two viewpoints per frame
  - Foveated rendering: full density at gaze point, reduced peripheral
  - Reprojection: engine targets 72fps, runtime reprojects to 90/120fps
  - AR passthrough: city rendered as overlay on real-world camera feed

Interaction:
  - Hand tracking: grab/place assets, draw roads, sculpt terrain
  - Controller: standard VR gamepad mapping
  - Gaze: select buildings by looking at them
```

---

## 15. Neural Denoiser

Engine-layer. Reduces spectral noise at high Gaussian density.

```
Problem: at 50M+ splats, alpha-blending noise becomes visible as shimmer
         at sub-pixel Gaussian scale

Solution: lightweight CNN denoiser as post-process
  - Input: noisy spectral framebuffer (8-channel)
  - Output: denoised spectral framebuffer
  - Architecture: U-Net variant, 4 downsampling stages
  - Inference: TensorRT, ~1ms at 1440p
  - Training data: rendered pairs (noisy high-density, clean reference)
```

---

## 16. Game UI (`vox_ui`)

Engine-layer UI framework, city builder skin on top.

```
Engine (vox_ui):
  - Retained-mode UI graph (not immediate-mode like egui)
  - GPU-rendered text (SDF font atlas)
  - Theming system (colours, fonts, spacing defined in TOML)
  - Layout: flexbox model
  - Widgets: button, slider, text, panel, scroll, dropdown, graph, map overlay
  - Animation: property tweening (fade, slide, scale)
  - Mod extension points: mods can register new panels

City builder UI (vox_app):
  - Top bar: population, money, date/time, speed controls
  - Bottom bar: tool palette (roads, zones, buildings, services, transport)
  - Side panels: selected building info, budget, overlays
  - Overlay system: toggle views (traffic heat, land value, pollution, coverage)
  - Advisor popups: contextual tips, warnings, milestones
  - Graphs: population, budget, satisfaction over time
  - Mini-map with zone colouring
```

---

## 17. Save/Load and Undo

Engine-layer.

```
Save format: .ochroma_save
  - World state serialised via serde (all ECS components)
  - Tile data (.vxt files) referenced, not duplicated
  - Compressed with zstd
  - Versioned: migration functions for format changes

Auto-save: every 5 minutes (configurable)
Quick-save: single slot, instant
Named saves: unlimited

Undo/redo:
  - Command pattern: every player action is a reversible Command
  - Undo stack: 100 commands deep
  - Group commands: "place road" = multiple segment commands grouped as one undo unit
  - Serialisable: undo stack saved with game state
```

---

## 18. Camera and Input System

Engine-layer.

```
Camera modes:
  - City overview: orbit around focal point, zoom 50m–10km altitude
  - Street level: free camera, WASD + mouse, collision with terrain
  - Cinematic: spline-based camera path, smooth interpolation
  - First-person citizen follow: attach to agent, see their world

Controls:
  - Mouse: orbit (middle drag), pan (right drag), zoom (scroll)
  - Keyboard: WASD movement, Q/E rotate, R/F altitude
  - Edge scroll: move camera when cursor hits screen edge
  - Gamepad: dual stick (left = pan, right = orbit), triggers = zoom
  - VR: head tracking drives camera, controllers for interaction

Input abstraction:
  - Unified input map: actions (place, select, cancel, rotate) bound to devices
  - Rebindable keybindings (saved in user config)
  - Device detection: mouse+keyboard, gamepad, touch, VR controllers
  - Input consumed by UI first, then game layer

Constraints:
  - Cannot go below terrain
  - Smooth altitude-based speed (faster when zoomed out)
  - LOD system (Phase 3) responds to camera altitude
```

---

## 19. Particle / VFX System

Engine-layer. Gaussian-native effects.

```
Emitter types:
  - Point: emit from a single position (campfire, chimney smoke)
  - Line: emit along a line (dust trail behind vehicle)
  - Surface: emit from a Gaussian surface (rain on rooftop)
  - Volume: emit within a bounding box (fog, dust cloud)

Particle properties (per particle = one Gaussian):
  - Lifetime (seconds)
  - Velocity + acceleration (gravity, wind)
  - SPD curve (fire: blackbody shift from orange to grey as it cools)
  - Opacity curve (fade in, sustain, fade out)
  - Scale curve (grow, shrink)

Performance:
  - Max 100,000 active particle Gaussians
  - Particle Gaussians mixed into the main depth sort (no separate pass)
  - Emitters culled by frustum like any other instance

City builder VFX (game layer):
  - Construction dust
  - Chimney smoke (scales with industrial output)
  - Fire (on buildings with fire event)
  - Rain/snow (weather system drives global emitter)
  - Exhaust from vehicles
  - Demolition debris cloud
```

---

## 20. District Policies and City Progression

Game-layer (`vox_sim`).

```
District policies:
  - Player draws district boundaries on the map
  - Per-district policy toggles:
    - Tax modifier (+/- 20% from base rate)
    - Rent control (caps land value growth)
    - Noise ordinance (blocks industrial adjacent to residential)
    - Speed limits (affects traffic flow in district)
    - Historical preservation (blocks demolition, restricts building height)
  - Policies affect citizen satisfaction, land value, and growth rate

City progression:
  - Milestones trigger at population thresholds (100, 500, 2k, 10k, 50k, 100k)
  - Each milestone unlocks: new building types, services, road types, policies
  - Era progression: village → town → city → metropolis
  - Era affects default Proc-GS style (rural → urban density and architecture)
```

---

## Performance Budget (Phase 4)

| Metric | Target |
|---|---|
| GPU | RTX 4080 (16GB VRAM) |
| Resolution | 2560×1440 |
| World size | 100km × 100km |
| Splat count (visible) | ≤ 50,000,000 |
| Frame time | ≤ 6.9ms (144fps) at street level |
| Citizen count | ≤ 100,000 (full lifecycle) |
| Active agents (rendered) | ≤ 10,000 |
| Audio sources | ≤ 64 simultaneous |
| Sim tick | ≤ 10ms at 10Hz |
| Save file size | ≤ 500MB for 100km city |
| Mod budget | ≤ 2ms per mod per frame |

---

## Phase 4 Exit Criteria

- [ ] Player can draw curved roads with intersections and roundabouts
- [ ] Zoning produces buildings that grow over game-time with construction animation
- [ ] Utilities (water, power, sewage) can be laid and buildings show served/unserved status
- [ ] Budget screen shows income/expenses, player can adjust tax rates
- [ ] 100,000 citizens with full lifecycle (birth → death) without frame regression
- [ ] Supply chain: timber flows from forest → sawmill → market → citizen consumption
- [ ] Bus route with stops transports citizens, visible at stops and on vehicles
- [ ] Terrain has hills, rivers, coastlines — not flat
- [ ] Spatial audio: approaching a market sounds different from approaching a park
- [ ] Physics: building destruction produces debris that settles realistically
- [ ] Skeletal animation: citizens visibly walk, sit, work with blended transitions
- [ ] Wasm mod can add a new building type with custom behaviour
- [ ] 4 players can build in the same city simultaneously via CRDT networking
- [ ] Game saves and loads with full state including undo history
- [ ] Full game UI with overlays, advisors, graphs (not egui dev tools)
- [ ] AR/VR: city viewable in VR headset with hand-tracked interaction
- [ ] Neural denoiser eliminates shimmer at 50M splats
- [ ] Camera smoothly transitions from 10km overview to street level
- [ ] Seasons cycle with visible and gameplay effects (snow, flooding, crops)
- [ ] Steam integration: achievements, workshop mod upload, cloud saves

---

## Phase 5 Preview

Once Phase 4 exits — the city builder is shipped:
- Engine SDK: `ochroma_engine` as a public crate with documented API
- General-purpose scene editor (city builder editor generalised)
- Material graph (visual spectral material editor)
- General-purpose terrain (not just city heightmaps)
- Navmesh generalisation (3D, not just 2D surface)
- Build/deploy pipeline (package games for platforms)
- Second game (different genre) to prove engine generality
- Console ports (PlayStation, Xbox, Switch)
- Marketplace for assets and mods

---

## Mapping to Master Requirements List

| Phase 4 item | Master list ref |
|---|---|
| Interactive road tools | #37 (extended) |
| Zoning system | #36 (extended) |
| Utilities | NEW |
| Public transport | NEW |
| City services | NEW |
| Budget / economy | NEW |
| Citizen lifecycle | #31, #112 (extended) |
| Supply chains | NEW |
| Agent social graphs | NEW |
| Terrain engine | NEW |
| Water simulation | NEW |
| Spatial audio | NEW |
| Physics | NEW |
| Skeletal animation | NEW |
| Wasm scripting | NEW |
| CRDT networking | NEW |
| AR/VR | NEW |
| Neural denoiser | NEW |
| Game UI framework | #41 (extended) |
| Save/load | NEW |
| Undo/redo | NEW |
| Camera system | NEW |
| Seasons gameplay | #35 (extended) |
| Steam integration | NEW |
