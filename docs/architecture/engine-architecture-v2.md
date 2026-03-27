# Ochroma Engine Architecture v2

Revised after critical review. Fixes all 14 identified problems.

## Core Principle

The engine is a Bevy ECS application. Everything is a system. Every entity is a Bevy entity with components. The engine binary creates the World, adds systems, and runs the schedule.

## Entity Model (Bevy ECS)

NO custom Entity struct. Use Bevy components:

```rust
// Core components (every entity can have these)
#[derive(Component)] struct Transform { position: Vec3, rotation: Quat, scale: Vec3 }
#[derive(Component)] struct AssetRef { handle: AssetHandle }
#[derive(Component)] struct Name(String);
#[derive(Component)] struct Tags(Vec<String>);
#[derive(Component)] struct Active(bool);
#[derive(Component)] struct Parent(Entity);
#[derive(Component)] struct Children(Vec<Entity>);

// Optional components (attached as needed)
#[derive(Component)] struct Collider { shape: ColliderShape }
#[derive(Component)] struct ScriptAttachment { scripts: Vec<String> }
#[derive(Component)] struct AudioEmitter { clip: AudioClipHandle, volume: f32, looping: bool }
#[derive(Component)] struct RigidAnimator { bone_group: u32, animation: String }
#[derive(Component)] struct PointLightComponent { color: Vec3, intensity: f32, radius: f32 }
#[derive(Component)] struct CustomData(HashMap<String, serde_json::Value>)
```

## The Frame Loop

```rust
// In engine_runtime.rs — this IS the engine

pub struct EngineRuntime {
    world: bevy_ecs::World,
    fixed_schedule: Schedule,     // runs at fixed timestep (physics, scripts)
    frame_schedule: Schedule,     // runs once per frame (render prep, audio, editor)
    asset_manager: AssetManager,
    accumulator: f32,
    fixed_dt: f32,                // 1/60 = 16.67ms
}

impl EngineRuntime {
    pub fn tick(&mut self, frame_dt: f32) {
        // Fixed timestep: physics + scripts (deterministic)
        self.accumulator += frame_dt;
        while self.accumulator >= self.fixed_dt {
            self.fixed_schedule.run(&mut self.world);
            self.accumulator -= self.fixed_dt;
        }

        // Variable timestep: rendering prep, audio, editor (once per frame)
        self.world.insert_resource(FrameDt(frame_dt));
        self.world.insert_resource(Interpolation(self.accumulator / self.fixed_dt));
        self.frame_schedule.run(&mut self.world);
    }
}
```

### Fixed Schedule Systems (60Hz, deterministic)
```
1. script_update_system      — run on_update() for scripted entities
2. process_commands_system    — apply spawn/destroy/move from scripts
3. physics_step_system        — step Rapier (or simple AABB)
4. collision_notify_system    — call on_collision() for overlapping entities
```

### Frame Schedule Systems (once per frame, variable rate)
```
5. animation_tick_system      — advance animations, compute bone transforms
6. audio_tick_system          — update listener, spatialize sources
7. hot_reload_system          — check for changed files
8. frustum_cull_system        — mark visible entities
9. lod_select_system          — choose LOD per entity
10. gather_splats_system      — collect visible splats into render buffer
11. particle_tick_system      — advance particles, add to render buffer
```

The binary (engine_runner.rs) handles:
- Window events → InputState resource
- Camera update from input
- Call engine.tick(frame_dt)
- Read render buffer from world → submit to GPU
- Editor overlay (if active)

## Render Pipeline

Three-tier fallback:

```
Tier 1: Spectra (offline, highest quality, user's separate tool)
Tier 2: wgpu GPU Rasteriser (real-time, primary for games)
Tier 3: Software Rasteriser (CPU fallback, always works)
```

The engine doesn't know which backend is active. It writes to a `RenderBuffer` resource. The binary reads it and submits to whichever backend is available.

```rust
#[derive(Resource)]
struct RenderBuffer {
    visible_splats: Vec<GaussianSplat>,
    lights: Vec<LightData>,
    camera: CameraData,
    particle_splats: Vec<GaussianSplat>,
}
```

### Shadows
GPU only. Deferred to Tier 2 (wgpu) implementation:
- Add a shadow pass to the WGSL shader
- Render depth map from sun perspective
- Sample in main pass

NOT implemented on CPU. If using software rasteriser, no shadows.

### Gizmos
Rendered as 2D overlay AFTER the scene, not mixed into splats:
- Project gizmo geometry to screen space
- Draw coloured lines/triangles directly into the framebuffer
- Always on top (no depth test)

## Asset Manager

```rust
struct AssetManager {
    splat_cache: HashMap<PathBuf, Arc<Vec<GaussianSplat>>>,
    clip_cache: HashMap<PathBuf, Arc<AudioClip>>,
    script_cache: HashMap<PathBuf, Arc<CompiledScript>>,
    file_watcher: AssetWatcher,
}

impl AssetManager {
    fn load_splats(&mut self, path: &Path) -> AssetHandle;  // cached, ref-counted
    fn load_audio(&mut self, path: &Path) -> AudioClipHandle;
    fn load_script(&mut self, path: &Path) -> ScriptHandle;
    fn check_hot_reload(&mut self) -> Vec<ReloadEvent>;
    fn get_splats(&self, handle: AssetHandle) -> Option<&[GaussianSplat]>;
}
```

Caching: same path = same data. Arc reference counting. Unload when no entity references it.

## Animation (v1: Rigid Only)

No deformable splat skinning. Only rigid groups:

```
Entity "Windmill"
  ├── Bone 0 "Base" (static)
  │   └── splats: [base tower splats]
  └── Bone 1 "Blades" (rotates around Z)
      └── splats: [blade splats]

Each frame:
  - Compute bone world transforms from hierarchy
  - For each splat bound to bone N: world_pos = bone_transform * local_pos
```

This is transform-only — the splat's covariance/shape is NOT transformed. Acceptable for rigid animation (doors opening, wheels spinning, windmills). NOT acceptable for character animation (deferred to v2).

## Script Context (Enriched)

```rust
impl ScriptContext {
    // Self
    fn get_position(&self) -> Vec3;
    fn set_position(&mut self, pos: Vec3);
    fn get_rotation(&self) -> Quat;
    fn set_rotation(&mut self, rot: Quat);

    // Other entities
    fn get_entity_position(&self, id: u32) -> Option<Vec3>;
    fn find_by_tag(&self, tag: &str) -> Vec<u32>;
    fn find_nearest(&self, tag: &str) -> Option<u32>;

    // Physics
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit>;

    // Game data
    fn get_data(&self, key: &str) -> Option<serde_json::Value>;
    fn set_data(&mut self, key: &str, value: serde_json::Value);

    // Engine
    fn get_time(&self) -> f32;
    fn get_dt(&self) -> f32;
    fn get_input(&self) -> &InputState;

    // Actions
    fn spawn(&mut self, asset: &str, pos: Vec3) -> u32;
    fn destroy(&mut self, id: u32);
    fn play_sound(&mut self, clip: &str, volume: f32);
    fn log(&mut self, msg: &str);
}
```

## Error Recovery

Every system phase is wrapped:
```rust
fn run_phase<F: FnOnce(&mut World) -> Result<(), String>>(
    world: &mut World,
    phase_name: &str,
    f: F,
) {
    match f(world) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("[engine] {} error: {}", phase_name, e);
            // Continue running — don't crash the engine
        }
    }
}
```

## Games Are Thin

A game built on Ochroma:
```rust
fn main() {
    let mut engine = EngineRuntime::new(EngineConfig {
        window_title: "My Game".into(),
        window_width: 1920,
        window_height: 1080,
        ..Default::default()
    });

    // Register game scripts
    engine.register_script("Player", || Box::new(PlayerController::new()));
    engine.register_script("Enemy", || Box::new(EnemyAI::new()));

    // Load a scene (map file or build programmatically)
    engine.load_map("maps/level1.ochroma_map");
    // OR:
    engine.spawn("Player", "characters/player.ply", Vec3::new(0.0, 2.0, 0.0))
          .with_script("Player")
          .with_collider(ColliderShape::Capsule { radius: 0.3, height: 1.8 });

    // Run — blocks until quit
    engine.run();
}
```

The game developer NEVER writes a render loop, window event handler, or ECS schedule. The engine handles everything.
