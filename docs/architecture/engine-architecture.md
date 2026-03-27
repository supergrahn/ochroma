# Ochroma Engine Architecture

## The Core Loop

Every frame, the engine executes these phases in order:

```
1. INPUT        → Read window events, update InputState
2. SCRIPTS      → Run on_update() for all scripted entities
3. PHYSICS      → Step physics simulation, resolve collisions
4. ANIMATION    → Update bone transforms, apply to splat positions
5. AUDIO        → Update listener position, tick spatial sources
6. SCENE        → Process spawn/destroy commands from scripts
7. CULLING      → Frustum cull entities, select LOD levels
8. RENDER       → Submit visible splats to GPU, present frame
9. EDITOR       → If editor mode: render gizmos, panels, handle editor input
```

Every system has ONE entry point called by the engine loop. No system calls another directly. They communicate through the scene graph (shared entity data).

## System Interfaces

### Scene Graph (the shared state)

All systems read/write entities in the scene. An entity is:

```rust
struct Entity {
    id: u32,
    active: bool,
    transform: Transform,        // position, rotation, scale
    asset: Option<AssetHandle>,  // what to render
    collider: Option<Collider>,  // physics shape
    scripts: Vec<ScriptHandle>,  // attached game logic
    audio_sources: Vec<AudioSourceHandle>,
    animator: Option<AnimatorHandle>,
    tags: Vec<String>,
    children: Vec<u32>,
    parent: Option<u32>,
}
```

Systems don't own entities. They operate on entity data each frame.

### Asset Manager

All assets loaded through one system:

```
AssetManager
  ├── load_ply(path) → AssetHandle     (Gaussian splats)
  ├── load_gltf(path) → AssetHandle    (mesh → converted to splats)
  ├── load_wav(path) → AudioClipHandle (audio file)
  ├── load_rhai(path) → ScriptHandle   (script file)
  ├── get_splats(handle) → &[GaussianSplat]
  ├── hot_reload_check() → Vec<AssetHandle>  (changed files)
  └── unload(handle)
```

### Render Pipeline

Abstracted behind a trait so we can swap backends:

```rust
trait RenderBackend {
    fn begin_frame(&mut self);
    fn submit_splats(&mut self, splats: &[GaussianSplat], camera: &Camera);
    fn submit_shadow_casters(&mut self, casters: &[ShadowCaster]);
    fn submit_lights(&mut self, lights: &[Light]);
    fn submit_particles(&mut self, particles: &[GaussianSplat]);
    fn end_frame(&mut self) -> Framebuffer;
}
```

Implementations:
- `SoftwareBackend` — CPU reference (works everywhere)
- `WgpuBackend` — GPU via wgpu (Vulkan/Metal/DX12)
- `SpectraBackend` — Spectra path tracer (highest quality)

### Physics Interface

```rust
trait PhysicsBackend {
    fn add_body(&mut self, entity_id: u32, transform: &Transform, collider: &Collider);
    fn remove_body(&mut self, entity_id: u32);
    fn step(&mut self, dt: f32);
    fn get_transform(&self, entity_id: u32) -> Option<Transform>;
    fn raycast(&self, origin: Vec3, direction: Vec3, max_dist: f32) -> Option<RayHit>;
    fn overlaps(&self, entity_id: u32) -> Vec<u32>;
}
```

Implementations:
- `SimplePhysics` — AABB overlap (built-in)
- `RapierPhysics` — full Rapier3D (when feature enabled)

### Audio Interface

```rust
trait AudioBackend {
    fn set_listener(&mut self, position: Vec3, forward: Vec3, up: Vec3);
    fn play(&mut self, clip: AudioClipHandle, position: Vec3, volume: f32, looping: bool) -> AudioInstanceHandle;
    fn stop(&mut self, handle: AudioInstanceHandle);
    fn set_position(&mut self, handle: AudioInstanceHandle, position: Vec3);
    fn tick(&mut self, dt: f32);  // update spatialization, remove finished
}
```

Implementations:
- `RodioAudioBackend` — real audio via rodio (when libasound available)
- `SilentAudioBackend` — no output (fallback)
- `SynthAudioBackend` — procedural tones via our synth module

### Animation Interface

```rust
struct Animator {
    skeleton: Skeleton,
    current_clip: AnimationClip,
    blend_target: Option<(AnimationClip, f32)>,
    time: f32,
    speed: f32,
}

// Each frame:
fn tick_animation(animator: &mut Animator, dt: f32) -> Vec<BoneTransform>;
fn apply_animation(splats: &mut [GaussianSplat], bones: &[BoneTransform], bindings: &[u8]);
```

### Editor Interface

The editor is a MODE of the engine, not a separate program:

```
Engine Mode:
  PLAY  → game runs normally
  EDIT  → camera is free, gizmos visible, entities selectable
  PAUSE → game paused, can inspect state

Toggle: Tab (or F1)
```

Editor systems:
- Gizmo renderer (visual arrows/rings at selected entity)
- Property inspector (egui panel)
- Scene hierarchy (egui panel)
- Content browser (egui panel showing asset directory)
- Toolbar (play/pause/stop)

### Script Interface

Scripts are the game developer's code. They see a limited API:

```rust
trait GameScript {
    fn on_start(&mut self, ctx: &mut ScriptContext);
    fn on_update(&mut self, ctx: &mut ScriptContext, dt: f32);
    fn on_collision(&mut self, ctx: &mut ScriptContext, other: u32);
    fn on_destroy(&mut self, ctx: &mut ScriptContext);
}

// ScriptContext provides:
ctx.get_position() → Vec3
ctx.set_position(pos)
ctx.get_rotation() → Quat
ctx.set_rotation(rot)
ctx.spawn(asset, position) → u32
ctx.destroy(entity_id)
ctx.play_sound(clip, volume)
ctx.raycast(origin, direction) → Option<RayHit>
ctx.find_by_tag(tag) → Vec<u32>
ctx.get_input() → &InputState
ctx.log(message)
```

## Data Flow

```
Frame N:
  Window Events → InputState
  InputState → Scripts (ctx.get_input())
  Scripts → Commands (spawn, destroy, move, play_sound)
  Commands → Scene Graph (entity positions updated)
  Scene Graph → Physics (sync transforms)
  Physics → Scene Graph (resolved transforms)
  Scene Graph → Animation (bone transforms applied to splats)
  Scene Graph → Audio (listener + source positions)
  Scene Graph → Culler (frustum test, LOD)
  Culler → Render Backend (visible splats only)
  Render Backend → Framebuffer
  Framebuffer → Window
```

No circular dependencies. Each system reads scene graph, does its work, writes back.

## File Layout

```
crates/
├── vox_core/          # Types, math, ECS, engine runtime
│   └── engine_runtime.rs  ← THE engine loop
├── vox_render/        # All rendering (backends, pipeline)
│   ├── backends/      # Software, wgpu, Spectra
│   ├── shadows.rs     # Shadow mapping
│   ├── gizmos.rs      # Editor gizmos
│   └── ...
├── vox_physics/       # Physics (simple + Rapier)
├── vox_audio/         # Audio (rodio + synth)
├── vox_anim/          # Animation (NEW crate)
├── vox_data/          # Asset loading, formats
├── vox_script/        # Rhai runtime
├── vox_editor/        # Editor mode (NEW crate)
├── vox_terrain/       # Terrain systems
└── vox_app/           # Binaries (ochroma, tools)
```
