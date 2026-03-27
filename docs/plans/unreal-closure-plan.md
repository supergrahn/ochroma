# Ochroma — Plan to Close Unreal 5 Gaps

## Strategy

Don't match Unreal feature-for-feature. Be the best at what Unreal CAN'T do, and good enough at what it can.

## Skip List (don't build these)
- **Marketplace** — let users import from Sketchfab, Polycam, Luma AI
- **Networking** — ship single-player first, add later
- **9 platforms** — Windows + Linux only, high-end PC, scale down later

## Auto-solved by Spectra
- **GPU rendering speed** — path tracer on GPU = millions of splats at 60fps
- **Material quality** — path tracer handles complex light transport natively
- **VFX lighting** — particles automatically cast shadows and emit light

## The 4 Phases

### Phase 1: Performance (GPU rendering)
**The engine is useless at 18fps.** This is the #1 blocker.

| Task | What | Agent work |
|---|---|---|
| 1A | Integrate Spectra's Rust crate OR build wgpu compute sort + rasterise pipeline | Copy spectra-gaussian-render source into our workspace (fix workspace dep issue) |
| 1B | Hit 60fps at 1M splats | Profile, optimize sort, tile-based rasterise |
| 1C | Verify with real 308k splat scene | Load character_hero.ply, render at 60fps |

### Phase 2: Editor & Physics (developer experience)
**Developers need to place things and have them collide.**

| Task | What | Agent work |
|---|---|---|
| 2A | Wire Rapier3D into engine runtime as default physics | Replace AABB with Rapier rigid bodies, colliders, raycasting |
| 2B | Editor 3D viewport with egui rendering | egui panels render in the GPU surface, not burned into framebuffer |
| 2C | Property inspector with live editing | Change entity position/rotation/scale in inspector → see it move in viewport |
| 2D | Content browser wired to viewport | Drag .ply from browser → appears in scene |

### Phase 3: Animation & AI (bring scenes to life)
**Static worlds are boring.**

| Task | What | Agent work |
|---|---|---|
| 3A | GLTF animation import + playback | Load .glb with animations, play them back on splat groups |
| 3B | Animation state machine in engine | Idle→Walk→Run blend based on velocity |
| 3C | NavMesh generation from SDF terrain | Recast-rs integration, auto-generate walkable areas |
| 3D | AI FSM trait + pathfinding | Enemies that patrol, chase, flee using navmesh paths |

### Phase 4: VFX & Polish (ship it)
**Make it look professional.**

| Task | What | Agent work |
|---|---|---|
| 4A | GPU particle system | Compute shader particles rendered as splats |
| 4B | Material hot-reload | Edit TOML material → scene updates instantly |
| 4C | Windows native build + test | Compile and run on real Windows |
| 4D | Second example game (different genre) | Proves engine is general-purpose |
| 4E | v0.2.0 release | Package, changelog, GitHub release |

## Timeline at AI Agent Speed

| Phase | Parallel agents | Estimated time |
|---|---|---|
| 1 (GPU) | 3 agents | 4-6 hours |
| 2 (Editor/Physics) | 4 agents | 3-4 hours |
| 3 (Animation/AI) | 4 agents | 3-4 hours |
| 4 (VFX/Polish) | 5 agents | 3-4 hours |
| Integration + testing | 1 sequential | 2 hours per phase |
| **Total** | | **~20-25 hours** |

## What This Gets Us

After all 4 phases:
- **GPU rendering at 60fps with millions of splats** (via Spectra or wgpu compute)
- **Full editor** with egui panels rendering on GPU surface
- **Rapier physics** with rigid bodies, colliders, raycasting
- **Animation playback** from GLTF files
- **AI with navmesh** pathfinding
- **GPU particles**
- **Two shipped games** proving it's general-purpose
- **Windows + Linux** support

Combined with our existing unique advantages (spectral rendering, volumetric terrain, Gaussian splat native), this makes Ochroma a VIABLE ALTERNATIVE to Unreal for developers who want:
- Photorealistic spectral lighting
- Volumetric terrain with caves/overhangs
- AI-native asset pipeline (scan → splat → render)
- Rust performance and safety
- Alignment with NVIDIA's hardware roadmap
