# Unreal 5 Gap Analysis — Honest Assessment

Based on Gemini's analysis: MVP game engine = 90-140 engineering-months.
At our pace, that's the roadmap.

## Feature-by-Feature Status

| # | System | UE5 MVP Cost | Ochroma Status | Gap | Priority |
|---|--------|-------------|----------------|-----|----------|
| 1 | **Rendering** | 12-18 EM | Spectral pipeline + GPU rasteriser + Spectra (25fps). Unique advantage. | Compositing splats with rasterized geometry | HIGH — our differentiator |
| 2 | **Lighting** | 4-6 EM | SunModel + point lights + spectral illuminants. Working. | No shadow maps, no irradiance probes | MEDIUM |
| 3 | **Materials** | 3-5 EM | MaterialNode graph + 11 spectral materials. Working. | No visual editor, no texture maps | LOW (spectral is better) |
| 4 | **Post-Processing** | 2-3 EM | ACES/Reinhard/Filmic tone mapping + bloom + denoiser. **Done.** | No LUTs, no auto-exposure | LOW |
| 5 | **Physics** | 3-5 EM | Rapier3D integrated + AABB collision in runtime. | Character controller needs work | MEDIUM |
| 6 | **Audio** | 2-3 EM | Rodio backend plays sounds. Synth generates tones. | No 3D spatialization wired in, no file loading (.wav/.ogg) | HIGH |
| 7 | **Animation** | 6-8 EM | Bone/clip/blend types exist. Not used in any game. | No GPU skinning, no state machines, no blend trees running | HIGH |
| 8 | **Networking** | 6-10 EM | TCP transport + CRDT + lobbies. Types only. | Never tested between two machines | LOW (skip for v1) |
| 9 | **UI** | 4-6 EM | egui overlay in editor. Walking sim has bitmap font HUD. | No retained-mode game UI, no layout engine | MEDIUM |
| 10 | **Editor** | 8-12 EM | Scene hierarchy + inspector + 3D picking + save/load. | No visual gizmos in viewport, no content browser | HIGH |
| 11 | **Asset Pipeline** | 6-8 EM | PLY loader works (308k splats loaded). VXM format. | No GLTF/FBX import, no texture compression, no hot-reload working | HIGH |
| 12 | **Scripting** | 3-5 EM | Rhai runtime + GameScript trait. Working. | Not wired into game loop in engine_runner | MEDIUM |
| 13 | **AI/Navigation** | 4-6 EM | NavMesh with A* pathfinding. Working. | Not used by any game entity | LOW |
| 14 | **Platform** | 2-4 EM | Windows + Linux via winit/wgpu. | Never tested on native Windows | MEDIUM |
| 15 | **Build System** | 1-2 EM | Cargo workspace. **Done.** | Need shipping/release profiles | LOW |
| 16 | **Profiling** | 2-3 EM | Puffin integrated. Frame stats in HUD. | No GPU timing, no memory tracking | LOW |
| 17 | **VFX/Particles** | 3-4 EM | ParticleSystem with emitters. Splat conversion. Working. | Not visible in games (only in engine_runner) | MEDIUM |
| 18 | **Cinematics** | 2-4 EM | CinematicCamera with keyframes. Working. | Not wired into any binary | LOW |
| 19 | **Terrain** | 4-6 EM | Volumetric SDF + heightmap + foliage. **Unique advantage.** | Working and integrated | DONE |
| 20 | **Foliage** | 2-3 EM | Scatter system with rules. Working. | Not GPU instanced | LOW |
| 21 | **Streaming/LOD** | 4-6 EM | Tile manager + LOD levels + crossfade. | Not tested at real scale | MEDIUM |
| 22 | **Input** | 1-2 EM | InputState + KeyBindings + rebinding. Working. | Not wired into engine_runner | LOW |
| 23 | **Localization** | 1 EM | I18nManager with locale bundles. Working. | Not wired into any UI | LOW |
| 24 | **Save System** | 1-2 EM | MapFile + scene serialization. Working. | Need ECS world serialization | MEDIUM |
| 25 | **Replay** | 3-5 EM | SimulationRecorder exists. | Never tested | LOW (skip for v1) |
| 26 | **Source Control** | 0 EM | Git. **Done.** | — | DONE |

## What's Actually Done (honestly)
- Rendering pipeline (spectral, unique)
- Volumetric terrain (unique, no other engine has this)
- Post-processing (tone mapping, denoiser)
- Build system (Cargo)
- Source control (Git)
- PLY asset loading (real 308k splats loaded)
- Basic editor (hierarchy, inspector, 3D picking)

## What Needs Work (HIGH priority)
1. **Editor** — needs viewport gizmos, content browser
2. **Asset Pipeline** — needs GLTF import, hot-reload
3. **Audio** — needs .wav/.ogg loading + 3D spatialization
4. **Animation** — needs to actually animate something visible
5. **Rendering** — needs shadow maps to look professional

## What to Skip for v1
- Networking (ship single-player first)
- Replay
- Cinematics editor
- Localization
- Console platforms

## Estimated Remaining Work
Using Gemini's estimates, filtering to HIGH + MEDIUM priority only:

| System | Estimated EM |
|--------|-------------|
| Editor improvements | 4 |
| Asset pipeline (GLTF) | 3 |
| Audio (proper 3D) | 2 |
| Animation (working) | 4 |
| Shadow maps | 2 |
| Character controller | 2 |
| Game UI | 3 |
| Platform testing | 1 |
| Streaming at scale | 2 |
| Save system | 1 |
| **Total** | **~24 EM** |

24 engineering-months to reach "shippable engine" from where we are.
That's 2 years solo, or 6 months with 4 engineers, or 3 months with AI-assisted development at our pace.

## The Plan

### Sprint 1 (immediate): Make a game someone can play
- Fix animation (animate an NPC walking)
- Fix audio (load and play .wav file)
- Fix editor gizmos (visual translate/rotate handles)
- Ship walking sim as a downloadable demo

### Sprint 2: Professional rendering
- Shadow maps (cascaded for directional light)
- Wire Spectra as renderer backend
- GLTF mesh import (converted to splats)

### Sprint 3: Game developer workflow
- Content browser in editor
- Hot-reload for scripts and assets
- Documentation that matches reality
- Example project with tutorial

### Sprint 4: Polish and ship
- Character controller (Rapier-based)
- Game UI framework
- Windows native testing
- Release build pipeline
