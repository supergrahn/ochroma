# Ochroma — Parallel Execution Plan to Match Unreal 5

## Strategy
- 5-10 agents working in parallel
- Each agent gets a self-contained task with clear inputs/outputs
- No agent depends on another agent's unfinished work
- Integration happens after each batch

## Workstream Layout

### Batch 1: Core Engine Systems (5 agents, parallel)

**Agent 1: Shadow Maps**
- Implement cascaded shadow maps for directional light
- Shadow map render pass (depth-only) for each cascade
- Shadow sampling in the splat shader
- Soft shadows via PCF filtering
- Test: shadow visible on terrain from a building

**Agent 2: Audio System Complete**
- Load .wav and .ogg files from disk (use rodio)
- 3D spatialization (distance attenuation + stereo panning based on listener/source positions)
- Sound manager: play, stop, loop, volume, pitch
- Integrate into engine_runner: collision → play sound, ambient loop, UI clicks
- Test: walk near a source, hear it louder on one side

**Agent 3: Animation System Working**
- GPU-friendly skeletal animation: bone hierarchy with world transforms
- Animation state machine: idle → walk → run with blend transitions
- Apply bone transforms to splat positions each frame
- Create a simple animated entity in the walking sim (a spinning windmill or walking NPC)
- Test: visible animation playing in the walking sim

**Agent 4: Editor Viewport Gizmos**
- Visual translate gizmo: 3 coloured arrows (X=red, Y=green, Z=blue) rendered as splat clusters at selected entity position
- Click-drag an arrow to move entity along that axis
- Visual rotate gizmo: 3 coloured rings
- Scale gizmo: 3 coloured cubes on axes
- W/E/R keys switch between translate/rotate/scale
- Test: select entity, drag gizmo, entity moves

**Agent 5: GLTF Import**
- Parse .gltf/.glb files (use gltf crate)
- Extract mesh vertices and convert to Gaussian splat cloud
- Each triangle becomes a cluster of splats on its surface
- Import textures as spectral material approximation
- CLI tool: `vox_tools import model.glb --output model.ply`
- Test: import a simple .glb cube, render as splats

### Batch 2: Game Systems (5 agents, parallel)

**Agent 6: Character Controller**
- Rapier-based kinematic character controller
- Ground detection (raycast down)
- Gravity + jump
- Slope handling (slide on steep slopes)
- Step climbing (auto-step small obstacles)
- Integrate into walking sim
- Test: player walks up a slope, jumps, can't walk through walls

**Agent 7: Game UI Framework**
- Retained-mode UI on top of egui
- Panel system: health bar, inventory, dialogue box, minimap
- UI responds to game state (health changes → bar updates)
- Main menu → Play → Pause → Quit flow
- Style system: fonts, colours, spacing from config
- Test: walking sim has a proper HUD, pause menu works

**Agent 8: Content Browser**
- File browser panel in the editor showing assets directory
- Thumbnails for .ply files (render a small preview)
- Drag asset from browser to viewport to place it
- Filter by file type (.ply, .vxm, .wav, .rhai)
- Search bar
- Test: browse assets, drag a .ply into the scene

**Agent 9: Hot-Reload**
- File watcher on scripts directory (Rhai .rhai files)
- When file changes, recompile and swap the script
- File watcher on assets directory (.ply files)
- When asset changes, reload and re-render
- Notification popup: "Reloaded player.rhai"
- Test: change a Rhai script, see effect without restart

**Agent 10: Save/Load ECS World**
- Serialize entire ECS world state to binary (serde + bincode)
- Include: all entities, transforms, scripts, colliders, lights
- Load restores exact state
- Auto-save every 5 minutes
- Quick-save (F5) / Quick-load (F9)
- Test: place entities, save, restart, load, entities restored

### Batch 3: Polish (5 agents, parallel)

**Agent 11: Shadow + Lighting Polish**
- Ambient occlusion (SSAO or splat-based approximation)
- HDR bloom pass
- Auto-exposure (histogram-based)
- Fog with distance and height
- Sky rendering (Preetham/Hosek model integrated into pipeline)

**Agent 12: Windows Native Build**
- Test compilation on x86_64-pc-windows-msvc
- Fix any platform-specific issues
- Create release build script
- Package as .zip with README
- GitHub Release with downloadable binary

**Agent 13: Documentation That Matches Reality**
- Audit every doc against actual code
- Rewrite Getting Started to match what actually works
- API reference for engine_runtime, GameScript, SceneEditor
- Tutorial: "Build Your First Game in 30 Minutes"
- Video script for demo recording

**Agent 14: Performance Optimization**
- Profile with puffin
- GPU timing queries
- Identify top 3 bottlenecks
- Optimize: batch splat uploads, reduce allocations per frame
- Target: 60fps at 100k splats on GPU path

**Agent 15: Example Games**
- Walking sim polished (animation, audio, proper HUD)
- Simple platformer (character controller, jump on platforms)
- Showcase demo (terrain, buildings, trees, weather, time of day)
- Each game demonstrates different engine features

## Integration Points

After Batch 1: All 5 systems working independently. Integration test: engine_runner uses all of them.
After Batch 2: Game developer workflow complete. Integration test: build a new game from scratch using the editor.
After Batch 3: Release-ready. Integration test: someone else downloads and runs it.

## Timeline at AI Speed

| Batch | Agents | Estimated time |
|-------|--------|----------------|
| 1 | 5 parallel | 2-4 hours |
| 2 | 5 parallel | 2-4 hours |
| 3 | 5 parallel | 2-4 hours |
| Integration + fixes | 1 sequential | 2 hours |
| **Total** | | **8-14 hours** |
