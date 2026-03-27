# Batch 1 Specs v2 — Revised After Architecture Review

## Pre-Batch 0: Engine Refactor (MUST happen first)

Before ANY feature work, we must fix the engine core. This is not a feature — it's the foundation everything else builds on.

### Refactor: engine_runtime.rs → Bevy ECS-based

**What:** Rewrite EngineRuntime to use Bevy ECS World instead of custom Vec<Entity>.

**Why:** Every feature agent needs to add Bevy systems. If the engine doesn't use Bevy, agents will fight over data structures.

**Scope:**
1. Delete custom Entity struct
2. Create Bevy components: Transform, AssetRef, Name, Tags, Collider, ScriptAttachment, CustomData
3. EngineRuntime owns a `World` and two `Schedule`s (fixed + frame)
4. Rewrite tick() to use fixed timestep accumulator
5. Move ALL systems (scripts, physics, animation, audio, culling) into Bevy system functions
6. Engine binary becomes thin: window events → tick() → read RenderBuffer → present

**Acceptance test:**
- Walking sim works using new engine
- `engine.spawn()` creates a Bevy entity with components
- Scripts can query/modify other entities via ScriptContext
- Fixed timestep: physics runs at 60Hz regardless of frame rate

### Refactor: Verify GPU Rendering

**What:** Render a known scene with the GPU rasteriser, save to file, compare with software rasteriser.

**Why:** We've never verified the WGSL shader produces correct output. Building shadows on a broken shader is pointless.

**Scope:**
1. Create a test that renders the same scene with both software and GPU rasteriser
2. Save both images
3. Compare: same general shape and colours (not pixel-exact, different algorithms)
4. If GPU output is wrong, fix the shader

**Acceptance test:** Side-by-side images showing the same scene from both renderers.

---

## Feature 1: Shadow Maps (GPU only)

### What
Depth-only render pass from the sun's perspective. Main render samples this to determine shadow/lit.

### Pre-requisite
GPU rendering verified (Pre-Batch 0).

### Architecture fit
- New Bevy system: `shadow_update_system` runs in frame schedule before `gather_splats_system`
- Writes shadow data to a `ShadowMap` resource
- The GPU rasteriser reads the ShadowMap resource during rendering

### Implementation
Add a second render pass to the WGSL shader:
```wgsl
// Pass 1: Render splats from sun's view → depth texture
// Pass 2: Render splats from camera view → sample shadow depth texture
```

The shadow map is a wgpu Texture created once, updated each frame.

### NOT on software rasteriser
CPU shadow maps are too slow. If using software fallback, no shadows.

### Acceptance test
- GPU render of a building on terrain shows shadow on terrain
- Shadow moves when T (time of day) is pressed
- Toggle shadow on/off with a key
- Save to PPM and compare with/without shadows

---

## Feature 2: Complete Audio System

### What
Load .wav from disk, play through speakers with 3D spatial positioning.

### Architecture fit
- New Bevy component: `AudioEmitter { clip: AudioClipHandle, volume: f32, looping: bool, playing: bool }`
- New Bevy resource: `AudioManager` wrapping the rodio backend
- New Bevy system: `audio_tick_system` in frame schedule
  - Reads camera position → sets listener
  - For each entity with AudioEmitter + Transform: update spatial volume/panning
  - Start/stop sources based on `playing` flag

### Implementation
```rust
fn audio_tick_system(
    audio: ResMut<AudioManager>,
    camera: Res<CameraState>,
    query: Query<(&Transform, &AudioEmitter)>,
) {
    audio.set_listener(camera.position, camera.forward);
    for (transform, emitter) in query.iter() {
        if emitter.playing {
            let distance = transform.position.distance(camera.position);
            let volume = emitter.volume / (1.0 + distance * 0.1);
            audio.update_source(emitter.handle, volume, transform.position);
        }
    }
}
```

### Fallback
If rodio init fails, AudioManager is a no-op. Engine continues silently.

### Acceptance test
- Place an entity with AudioEmitter in the scene
- Walk toward it → hear sound getting louder
- Walk away → sound fades
- Click → hear click sound
- No crash if audio unavailable

---

## Feature 3: Rigid Animation

### What
Groups of splats move as rigid bodies driven by a bone hierarchy. Windmill blades spin, doors open, wheels rotate.

### Architecture fit
- New Bevy component: `RigidAnimator { bone_group: u32, current_animation: String, time: f32, speed: f32 }`
- Animation data stored in AssetManager (loaded from a simple JSON/TOML format)
- New Bevy system: `animation_tick_system` in frame schedule
  - For each entity with RigidAnimator: advance time, compute bone transform
  - Transform the entity's bound splats by the bone transform

### Implementation
The key insight: for rigid animation, we don't transform individual splats. We transform the ENTITY'S transform component. The splats are offset from the entity origin at load time. When the entity moves/rotates, all its splats move with it.

For a windmill:
- Entity "Windmill Base" at position (10, 0, 5), no animation → static
- Entity "Windmill Blades" at position (10, 8, 5), parent = Base
  - RigidAnimator: rotate around Z at 1 rad/sec
  - Each frame: rotation.z += dt * speed
  - Splats are centered at origin → entity transform places them correctly

This is just animating Transform components. No splat-level modification needed.

### Acceptance test
- Walking sim has a windmill with visibly spinning blades
- Animation speed can be changed from a Rhai script
- Door entity that opens (rotates 90°) when player approaches

---

## Feature 4: Editor Viewport Gizmos (2D Overlay)

### What
Coloured arrows/rings overlaid on the selected entity for translate/rotate/scale.

### Architecture fit
- Part of the editor mode, not a Bevy system
- Runs AFTER scene render, draws directly into the framebuffer
- Uses screen-space projection (2D lines, not 3D splats)

### Implementation
After the main render produces a framebuffer:
1. Project selected entity position to screen space
2. Draw 3 coloured arrows from that screen position:
   - X axis → red line going right
   - Y axis → green line going up
   - Z axis → blue line going into screen (foreshortened)
3. Arrow length: 80 pixels (constant screen size)
4. Click detection: check if mouse is within 5 pixels of an arrow line
5. Drag: track mouse delta, convert to world-space movement along the clicked axis

For software rasteriser: draw lines directly into the pixel buffer (Bresenham line algorithm).
For GPU: render as a second pass with no depth test.

### Acceptance test
- Select entity → coloured arrows appear at its position
- Click and drag red arrow → entity moves along X axis in world space
- Press E → arrows change to rotation rings
- Gizmos always visible (not occluded by geometry)
- Undo (Ctrl+Z) restores entity to pre-drag position

---

## Feature 5: GLTF Import (Reference Quality)

### What
Convert .gltf/.glb meshes to rough Gaussian splat clouds for placement and reference.

### Architecture fit
- Tool in vox_tools, not a runtime system
- Produces a .ply file that the engine loads normally
- NOT production quality — documented as "reference geometry"

### Implementation
Use the `gltf` crate:
```rust
fn import_gltf(path: &Path) -> Result<Vec<GaussianSplat>, ImportError> {
    let (document, buffers, _) = gltf::import(path)?;
    let mut splats = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
            let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();

            // For each triangle: place splats on the surface
            for tri in indices.chunks(3) {
                let v0 = Vec3::from(positions[tri[0] as usize]);
                let v1 = Vec3::from(positions[tri[1] as usize]);
                let v2 = Vec3::from(positions[tri[2] as usize]);

                let area = (v1 - v0).cross(v2 - v0).length() * 0.5;
                let normal = (v1 - v0).cross(v2 - v0).normalize();
                let splats_for_tri = (area * 100.0).ceil() as usize; // ~100 splats per m²

                for i in 0..splats_for_tri {
                    // Random barycentric coordinate (deterministic from index)
                    let t = i as f32 / splats_for_tri as f32;
                    let u = (t * 7.3).fract();
                    let v = (t * 13.7).fract();
                    let (u, v) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };
                    let pos = v0 * (1.0 - u - v) + v1 * u + v2 * v;

                    splats.push(make_splat(pos, normal, area));
                }
            }
        }
    }
    Ok(splats)
}
```

### Documentation
Clearly state in README and import tool output:
```
NOTE: GLTF import produces REFERENCE QUALITY splat clouds.
For production assets, train proper 3DGS from multi-view captures.
```

### Acceptance test
- Import a .glb cube → splat cloud resembles a cube
- Import a .glb sphere → splat cloud resembles a sphere
- Import a .glb with colour → splats have approximate colour
- Exported .ply loads in the engine and renders

---

## Execution Order

1. **Pre-Batch 0** (sequential, me): Engine refactor + GPU verification
2. **Batch 1** (5 parallel agents): Shadows, Audio, Animation, Gizmos, GLTF
3. **Integration** (sequential, me): Wire all 5 into the engine, test end-to-end

Pre-Batch 0 MUST complete before dispatching agents. It defines the Bevy component structure that all agents need.
