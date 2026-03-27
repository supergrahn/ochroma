# Batch 1 — Detailed Specifications

Each spec defines: what to build, how it integrates, exact API, acceptance test.

---

## Feature 1: Shadow Maps

### What
Cascaded Shadow Maps (CSM) for the directional sun light. Buildings and trees cast shadows on terrain.

### How it integrates
- Rendering phase: before main splat render, render a depth-only pass from the sun's perspective
- The shadow map texture is passed to the main render as a uniform
- During splat shading: sample shadow map to determine if pixel is in shadow → darken by 50%

### API
```rust
// In vox_render/src/shadows.rs
pub struct ShadowMapper {
    cascade_count: u32,           // 3 cascades: near (0-20m), mid (20-100m), far (100-500m)
    shadow_map_size: u32,         // 1024x1024 per cascade
    light_direction: Vec3,
    cascade_splits: [f32; 4],     // frustum split distances
    light_view_projs: [Mat4; 3],  // view-projection per cascade
}

impl ShadowMapper {
    fn new(size: u32) -> Self;
    fn update(&mut self, camera: &Camera, light_dir: Vec3);
    fn get_light_vp(&self, cascade: usize) -> Mat4;
    fn is_in_shadow(&self, world_pos: Vec3, cascade_depths: &[f32]) -> bool;
}
```

### Integration in engine loop
```
Phase 7 (RENDER):
  1. shadow_mapper.update(camera, sun_direction)
  2. For each cascade: render splats from light's perspective → depth buffer
  3. Main render: for each pixel, check shadow map → darken if shadowed
```

### For software rasteriser (where we can implement this now)
Since the software rasteriser runs on CPU, implement shadow testing per-pixel:
- Pre-compute shadow map as a depth buffer from the sun's perspective
- During main render, for each visible splat, project its position into shadow space and compare depth

### Acceptance test
- Render a scene with a building and terrain
- Building's shadow visible as a dark region on the terrain
- Shadow moves when time-of-day changes (sun position)
- Save to PPM and verify visually

---

## Feature 2: Complete Audio System

### What
Load .wav/.ogg from disk, play with 3D spatial positioning, manage multiple simultaneous sources.

### How it integrates
- Phase 5 (AUDIO): update listener from camera, tick all playing sources
- Scripts trigger sounds via `ctx.play_sound("explosion.wav", 1.0)`
- Engine_runner manages the audio backend lifecycle

### API
```rust
// In vox_audio/src/spatial.rs (NEW)
pub struct SpatialAudioManager {
    backend: Box<dyn AudioBackend>,
    listener_pos: Vec3,
    listener_forward: Vec3,
    sources: Vec<ActiveSource>,
}

struct ActiveSource {
    handle: u32,
    position: Vec3,
    clip: AudioClipHandle,
    volume: f32,
    looping: bool,
    distance_model: DistanceModel,
}

pub enum DistanceModel {
    Linear { max_distance: f32 },
    InverseDistance { ref_distance: f32 },
}

impl SpatialAudioManager {
    fn new() -> Self;  // tries rodio, falls back to silent
    fn load_clip(&mut self, path: &Path) -> Result<AudioClipHandle, String>;
    fn play_3d(&mut self, clip: AudioClipHandle, position: Vec3, volume: f32, looping: bool) -> u32;
    fn play_2d(&mut self, clip: AudioClipHandle, volume: f32) -> u32;  // UI sounds, music
    fn stop(&mut self, handle: u32);
    fn set_listener(&mut self, position: Vec3, forward: Vec3);
    fn tick(&mut self, dt: f32);  // update volumes based on distance, remove finished
}
```

### For rodio backend
- Use `rodio::Sink` per source
- Volume adjusted each tick based on distance: `vol = base_vol / (1 + dist * 0.1)`
- Left/right panning from dot product of (source - listener) with listener's right vector
- If rodio init fails, fall back to SilentAudioBackend (no crash)

### Acceptance test
- Walking sim: approach a point light source → hear a tone getting louder
- Walking sim: collect an orb → hear a "collect" sound
- Engine_runner: click → hear a click sound
- No crash if audio system unavailable

---

## Feature 3: Working Animation

### What
Animate entity transforms over time. Support: rigid hierarchical animation (parent-child), and basic Gaussian splat deformation via bone weights.

### How it integrates
- Phase 4 (ANIMATION): for each entity with an Animator, advance time, compute bone transforms, apply to splats
- Entity's splats are modified in-place each frame based on bone positions

### API
```rust
// In vox_render/src/animation.rs (MODIFY existing)

pub struct AnimationSystem {
    animators: HashMap<u32, EntityAnimator>,  // entity_id → animator
}

pub struct EntityAnimator {
    skeleton: Skeleton,
    state_machine: AnimationStateMachine,
    current_bone_transforms: Vec<Mat4>,
}

pub struct AnimationStateMachine {
    states: HashMap<String, AnimationState>,
    current_state: String,
    transition_time: f32,
}

pub struct AnimationState {
    clip: AnimationClip,
    loop_mode: LoopMode,
    speed: f32,
    transitions: Vec<Transition>,  // conditions to move to another state
}

pub struct Transition {
    target_state: String,
    condition: TransitionCondition,
    blend_duration: f32,
}

pub enum TransitionCondition {
    AfterTime(f32),           // transition after N seconds
    OnEvent(String),          // transition when event triggered
    BoolParam(String, bool),  // transition when parameter is true/false
}

impl AnimationSystem {
    fn add_animator(&mut self, entity_id: u32, skeleton: Skeleton, initial_state: &str);
    fn set_parameter(&mut self, entity_id: u32, name: &str, value: bool);
    fn trigger_event(&mut self, entity_id: u32, event: &str);
    fn tick(&mut self, dt: f32) -> Vec<(u32, Vec<Mat4>)>;  // returns (entity_id, bone_world_transforms)
}
```

### For the walking sim (immediate proof)
Create a simple animated object — a windmill with rotating blades:
- 2 bones: base (static) + blades (rotating around Y axis)
- Blades splats bound to bone 1
- Each frame: blade bone rotates by `dt * speed`
- Visible rotation in the walking sim

### Acceptance test
- Walking sim has a windmill with visibly rotating blades
- Rotation speed can be changed from a script
- Blending works: transition between two states smoothly

---

## Feature 4: Editor Viewport Gizmos

### What
Visual handles in the 3D viewport that let you move, rotate, and scale selected entities by clicking and dragging.

### How it integrates
- Phase 9 (EDITOR): if editor mode, render gizmo splats at selected entity position
- Mouse input: detect if click is on a gizmo axis → enter drag mode → move entity along that axis

### API
```rust
// In vox_app/src/editor_gizmos.rs (NEW)

pub struct GizmoRenderer {
    mode: GizmoMode,  // Translate, Rotate, Scale
    active_axis: Option<Axis>,  // which axis is being dragged
    drag_start: Option<Vec3>,
}

pub enum Axis { X, Y, Z }

impl GizmoRenderer {
    fn generate_gizmo_splats(&self, entity_position: Vec3, camera_distance: f32) -> Vec<GaussianSplat>;
    fn hit_test(&self, entity_position: Vec3, ray_origin: Vec3, ray_dir: Vec3) -> Option<Axis>;
    fn begin_drag(&mut self, axis: Axis, mouse_pos: Vec3);
    fn update_drag(&mut self, mouse_pos: Vec3) -> Vec3;  // returns delta to apply to entity
    fn end_drag(&mut self);
}
```

### Visual design
- Translate: 3 arrow shafts (cylinders of splats) + 3 arrow heads (cones of splats)
  - X = red splats pointing right
  - Y = green splats pointing up
  - Z = blue splats pointing forward
  - Each arrow ~2m long, scaled by camera distance so they're always same screen size
- Rotate: 3 torus rings of splats (same colours)
- Scale: 3 lines with cube endpoints

### Hit testing
- Each axis is a cylinder in world space
- Ray-cylinder intersection test
- The closest intersected axis wins

### Acceptance test
- Select entity, see coloured arrows appear
- Click red arrow, drag right → entity moves along X
- Press E → see rotation rings instead
- Undo (Ctrl+Z) restores position

---

## Feature 5: GLTF Import

### What
Convert standard 3D models (.gltf/.glb) into Gaussian splat clouds that the engine can render.

### How it integrates
- Asset pipeline tool: `cargo run --bin vox_tools -- import model.glb -o model.vxm`
- Also callable from editor: drag .glb into content browser → auto-converts
- Uses the `gltf` crate for parsing

### API
```rust
// In vox_data/src/gltf_import.rs (NEW)

pub struct GltfImporter;

impl GltfImporter {
    /// Import a GLTF file and convert to Gaussian splats.
    pub fn import(path: &Path) -> Result<ImportResult, ImportError>;
}

pub struct ImportResult {
    pub splats: Vec<GaussianSplat>,
    pub node_count: usize,
    pub mesh_count: usize,
    pub material_count: usize,
}
```

### Conversion algorithm
For each mesh in the GLTF:
1. Extract triangles (vertices + indices)
2. For each triangle:
   - Compute area
   - Place N splats on the triangle surface (N proportional to area)
   - Each splat: position = random barycentric point on triangle
   - Scale = proportional to triangle edge lengths (small triangles → small splats)
   - Rotation = aligned to triangle normal
   - Spectral colour = from the mesh's material (base colour → approximate spectral)
3. Merge all triangle splats into one cloud

### Acceptance test
- Import a .glb cube → renders as a cube-shaped splat cloud
- Import a .glb sphere → renders as a sphere
- Import a .glb with materials → colours are approximately correct
- Vertex count → proportional splat count (1000 vertices → ~5000 splats)

---

## Integration Contract

After all 5 agents finish, the engine_runner.rs must use:
1. ShadowMapper in the render phase (shadows on terrain)
2. SpatialAudioManager in the audio phase (3D sounds)
3. AnimationSystem in the animation phase (windmill spins)
4. GizmoRenderer in the editor phase (coloured arrows on selected entity)
5. Assets loaded via AssetManager which can load .ply AND .gltf

The integration agent (me) will wire these together after all 5 are built.
