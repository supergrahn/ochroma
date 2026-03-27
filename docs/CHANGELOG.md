# Changelog

## v0.1.0 (2026-03-27)

Initial engine release.

### Core Engine (`vox_core`)

- Bevy ECS integration (`bevy_ecs 0.16`) — entities, components, systems
- Fixed-timestep game loop (`GameClock`) with variable render timestep
- Character controller component with gravity, jump force, sprint multiplier
- Spectral definitions: 8-band (380-660nm) `Illuminant` (D65, Illuminant A, custom)
- `GaussianSplat` type: position, scale, rotation (i16 quaternion), opacity, 8-band spectral reflectance
- `GameScript` trait: `on_start`, `on_update`, `on_destroy`, `on_collision` callbacks
- `ScriptContext` with commands: Spawn, Destroy, SetPosition, SetRotation, PlaySound, ApplyForce, SendEvent, Log
- `ScriptRegistry`: register and instantiate scripts by name
- Input state management
- Undo/redo stack
- Hot-reload manager: polls watched files and emits `ScriptChanged`/`AssetChanged` events
- Large world coordinates (LWC) tile system
- Sparse Voxel Octree (SVO) spatial hash
- Navigation mesh
- Procedural terrain generation (heightmap-based)
- Procedural map generation (splat-based)
- Game UI framework (bitmap font, panels, labels, buttons)
- i18n locale scaffolding

### Rendering (`vox_render`)

- Software rasteriser: CPU Gaussian splat rasterisation, depth sorting, alpha blending
- wgpu backend: presents CPU framebuffer via wgpu surface (Vulkan/Metal/DX12)
- GPU rasteriser: wgpu compute pipeline for splat rasterisation (requires GPU)
- Spectral framebuffer: 8-band radiance buffer with depth, normals, motion vectors, albedo
- Temporal accumulator: history blending for denoising
- Spectral tone mappers: Linear, ACES, Reinhard, Filmic
- Spectral shift: time-of-day illuminant blending, weather effects, wear effects
- DLSS pipeline (software): quality mode API (Off/Quality/Balanced/Performance/Ultra Performance), frame generation flag
- CLAS clustering: splat cluster BVH for acceleration (software, CPU)
- MegaGeometry dispatch scaffolding
- Frustum culling
- LOD system: distance-based splat count reduction, LOD crossfade
- Particle system: emitters with velocity, lifetime, spectral colour
- Lighting system: `PointLight`, `LightManager`, sun model, sky colour
- Animation: keyframe clips, skeleton, bone hierarchy, animation player
- Rigid animation: state machine-based animation controller
- Post-processing pipeline: tone mapping, atmospheric scattering, god rays, fog
- Subsurface scattering profile
- Denoiser (`SpectralDenoiser`)
- Shadow catcher geometry
- Material graph: `SpectralMaterialGraph` with material nodes
- Gizmos: 3D translate/rotate/scale handles
- Spatial UI
- Water surface
- Destruction masks and debris generation
- Atmosphere: sky colour, fog, god ray intensity
- Performance inspector: frame breakdown, VRAM breakdown, entity breakdown

### Assets (`vox_data`)

- PLY loader: binary little-endian 3DGS PLY files with standard properties (position, scale, rotation, opacity, f_dc SH coefficients, f_rest SH)
- VXM format: custom compressed Gaussian splat asset format
- Map file: JSON `.ochroma_map` with terrain config, placed objects, lights, fog, gravity, time-of-day, spawn points
- Procedural Gaussian splat generators: terrain, buildings (victorian style), trees
- Advanced procedural generators: detailed buildings with floors and style variants
- GLTF/GLB import (reference quality, via vox_tools)
- Asset hot-reload watcher
- Asset catalog
- Content browser classification by file extension
- OSM (OpenStreetMap) import scaffolding
- Marketplace asset search scaffolding
- Neural compression estimation
- Creator tools: sculpt brush, SDF operations

### Application Layer (`vox_app`)

- `ochroma` binary: full engine — spectral pipeline, CLAS, lighting, particles, physics, Rhai scripts, editor overlay, Ctrl+S save
- `demo` binary: interactive demo with time-of-day, tone mapping, DLSS cycling, scene editor
- `walking_sim` binary: complete game — walk, collect 10 orbs, win; building AABB collision, animated orbs, bitmap-font HUD, Rhai scripting
- `render_showcase` binary: headless, exercises all 76 modules, saves `render_showcase_output.ppm`
- Scene editor: entity hierarchy (parent/child), property inspector, entity add/delete/move
- Content browser: file type classification
- Autosave system
- Persistence / save-load game state
- Debug console
- Undo integration
- Tutorial system scaffolding
- Settings persistence (JSON)
- Soundscape system
- Minimap
- Notifications overlay
- Road builder scaffolding

### Physics (`vox_physics`)

- Built-in AABB rigid body physics with gravity and collision response
- Rapier 3D integration (optional `rapier` feature, enabled by default): full rigid body simulation

### Audio (`vox_audio`)

- Spatial audio: distance-attenuation volume calculation
- Acoustic ray tracer: RT60 estimation, impulse response
- Audio synthesis: tone generation, collect sound, click sound, place sound
- WAV file output
- Rodio backend (optional `audio-backend` feature, enabled by default): real audio playback on Linux/Windows/macOS

### Networking (`vox_net`)

- CRDT replication: LWW set operations, operation log
- Client/server transport (TCP scaffolding)
- Replication server and client
- Lobby system
- World hosting

### Scripting (`vox_script`)

- Rhai scripting runtime: load and run `.rhai` scripts, eval expressions
- Visual scripting: node graph with event, condition, action nodes
- Plugin system: dynamic plugin loading scaffolding
- Wasm module runtime (optional `wasm-runtime` feature)

### AI/Procedural (`vox_nn`)

- LLM client (mock, for offline development)
- Text-to-city generation: generate district from text prompt
- History generation: procedural era and civilization history
- Scene query engine
- Natural language command parser
- Street layout generation
- Multi-tile city generation

### Tools (`vox_tools`)

- `vox_tools` CLI binary
- Turnaround capture pipeline: photo set to `.vxm` asset
- GLTF/GLB import and conversion to `.vxm`
- Build system manifests

### Known Limitations

- DLSS is a software-only quality mode API; hardware NVIDIA DLSS requires an NVIDIA GPU and SDK not included in this release
- GPU rasteriser falls back gracefully — CPU software rasteriser is used by default
- Audio backend requires `libasound2-dev` on Linux
- PLY test scenes in `assets/test_scenes/` are empty placeholders; bring your own trained scenes
- Networking, AI/procedural, and VR systems are scaffolded but not wired into the main binaries
