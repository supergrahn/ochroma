# Domain 2 — Editor & Tooling

**Status:** Draft — 2026-03-29
**Scope:** Core editor architecture, material editor, terrain editor, visual scripting, asset pipeline, profiler & frame debugger
**Engine:** Ochroma spectral Gaussian Splatting — Rust workspace, wgpu 24, WGSL shaders, egui, naga validation

---

## Goals

The Ochroma editor is a standalone application in `vox_app` that hosts a full scene editing experience with spectral-first workflows not found in any existing engine. It must support non-destructive editing with unlimited undo/redo, real-time spectral material preview, SDF terrain sculpting, Rhai-backed visual scripting, a streaming-ready asset pipeline, and an integrated GPU profiler. All editor state is isolated from the runtime simulation world; the editor drives the engine crates (`vox_core`, `vox_render`, `vox_terrain`, `vox_script`) without coupling engine types to editor concepts. The editor must launch in under 3 seconds on a mid-range workstation and never block the main thread during asset cooking.

---

## 2.1 Core Editor Architecture

### EditorWorld

The editor maintains a parallel scene representation that is distinct from any runtime `World`. It owns the authoritative scene tree, component storage, selection state, and command history. Runtime worlds are derived from `EditorWorld` by serialization + deserialization; no direct pointer sharing occurs between the two.

```rust
// crates/vox_app/src/editor/world.rs

pub struct EditorWorld {
    pub nodes:          HashMap<NodeId, SceneNode>,
    pub root_ids:       Vec<NodeId>,
    pub components:     HashMap<NodeId, ComponentStore>,
    pub command_history: CommandHistory,
    pub selection:      SelectionState,
    pub next_id:        NodeId,
}

pub struct SceneNode {
    pub id:        NodeId,
    pub name:      String,
    pub parent:    Option<NodeId>,
    pub children:  Vec<NodeId>,
    pub transform: LocalTransform,
    pub enabled:   bool,
}

#[derive(Clone, Debug)]
pub struct LocalTransform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale:    glam::Vec3,
}

impl LocalTransform {
    pub fn world_transform(&self, parent_world: Option<glam::Mat4>) -> glam::Mat4 {
        let local = glam::Mat4::from_scale_rotation_translation(
            self.scale, self.rotation, self.position
        );
        match parent_world {
            Some(p) => p * local,
            None    => local,
        }
    }
}
```

World transforms are computed lazily by traversing the `SceneNode` parent chain. `EditorWorld::world_transform(id)` caches results in a `HashMap<NodeId, glam::Mat4>` dirty-marked by `MoveEntityCommand` and `ReparentCommand`. The cache is invalidated for a node and all its descendants when a transform changes.

### Command Pattern

```rust
// crates/vox_app/src/editor/command.rs

pub trait EditorCommand: Send + Sync + std::fmt::Debug {
    fn execute(&mut self, world: &mut EditorWorld);
    fn undo(&mut self, world: &mut EditorWorld);
    fn description(&self) -> &str;
    /// If `true`, this command merges with the previous command of the same type
    /// (e.g., continuous drag produces one undo entry, not thousands).
    fn merge_with_previous(&self, prev: &dyn EditorCommand) -> bool { false }
}

pub struct CommandHistory {
    commands: Vec<Box<dyn EditorCommand>>,
    cursor:   usize,  // points to the next redo slot; commands[0..cursor] are the done stack
}

impl CommandHistory {
    pub fn execute(&mut self, mut cmd: Box<dyn EditorCommand>, world: &mut EditorWorld) {
        cmd.execute(world);
        // Truncate redo stack: anything past cursor is discarded
        self.commands.truncate(self.cursor);
        // Attempt merge with previous command
        if let Some(prev) = self.commands.last_mut() {
            if cmd.merge_with_previous(prev.as_ref()) {
                *prev = cmd;  // replace previous with merged (prev already executed its execute())
                return;
            }
        }
        self.commands.push(cmd);
        self.cursor = self.commands.len();
    }

    pub fn undo(&mut self, world: &mut EditorWorld) {
        if self.cursor == 0 { return; }
        self.cursor -= 1;
        self.commands[self.cursor].undo(world);
    }

    pub fn redo(&mut self, world: &mut EditorWorld) {
        if self.cursor == self.commands.len() { return; }
        self.commands[self.cursor].execute(world);
        self.cursor += 1;
    }
}
```

**Concrete command types:**

- `SpawnEntityCommand { id: NodeId, name: String, parent: Option<NodeId>, template: Option<AssetPath> }` — undo removes the node and all its descendants.
- `DeleteEntityCommand { snapshot: SceneNodeSnapshot }` — stores the full subtree snapshot; undo re-inserts it.
- `MoveEntityCommand { id: NodeId, old_transform: LocalTransform, new_transform: LocalTransform }` — merges with previous `MoveEntityCommand` for the same `id` during drag (checked in `merge_with_previous`).
- `SetComponentCommand<T: Component> { id: NodeId, old_value: T, new_value: T }` — generic over component type.
- `ReparentCommand { id: NodeId, old_parent: Option<NodeId>, new_parent: Option<NodeId>, old_index: usize }` — preserves sibling order in the old parent and inserts at `new_index` in the new parent.

### Multi-Select and Selection State

```rust
pub struct SelectionState {
    pub selected:    HashSet<NodeId>,
    pub pivot_mode:  PivotMode,
    pub last_active: Option<NodeId>,
}

pub enum PivotMode {
    BoundingBoxCenter,
    IndividualOrigins,
    WorldOrigin,
}
```

Selection changes are themselves commands (`SetSelectionCommand`) so they participate in undo/redo. Box selection in the viewport dispatches a single `SetSelectionCommand` with the full new `HashSet<NodeId>`.

### Gizmo System

Gizmos are rendered as a separate overlay draw after the scene. Hit testing is CPU-side against gizmo geometry projected to screen space:

- `TranslateGizmo`: 3 axis arrows + 3 plane squares. Hit test: ray vs. cone (arrow tip) + ray vs. square.
- `RotateGizmo`: 3 torus rings (X/Y/Z planes) + outer screen-space ring. Hit test: ray vs. torus (solve quartic or approximate with a circle at the viewing angle).
- `ScaleGizmo`: 3 axis cubes + center cube for uniform scale.

Each gizmo handle has a `GizmoHandle { axis: Axis, gizmo_type: GizmoType }` identifier. On mouse-down, the hit handle is stored; on mouse-drag, a `MoveEntityCommand` / `RotateEntityCommand` / `ScaleEntityCommand` is dispatched for each selected node (with `merge_with_previous` returning `true`). Snapping is applied before dispatch:

- **Grid snap** (translate): `snap_to_grid(v, grid_size) = round(v / grid_size) * grid_size`.
- **Degree snap** (rotate): quantize Euler angle to nearest `snap_degrees` (default 15°), convert back to Quat.
- **Uniform scale lock**: when the center cube is dragged, all three scale components change by the same factor.

### Viewport — Multiple Views

```rust
pub struct EditorViewport {
    pub id:          ViewportId,
    pub view_type:   ViewType,  // Perspective, Top, Front, Side, UV
    pub camera:      RenderCamera,
    pub resolution:  [u32; 2],
    pub show_gizmos: bool,
    pub overlay:     PerSplatDebugView,  // from profiler subsystem
}

pub enum ViewType {
    Perspective,
    Orthographic { axis: OrthoAxis },
}
```

The main window defaults to a 4-pane layout (perspective + top + front + side), each pane backed by its own `wgpu::Texture` render target. Panes can be resized by dragging dividers; the `split_ratio` is stored in `EditorLayout` and persisted to `editor_prefs.toml`.

### egui Integration

All panels are egui widgets calling into `EditorWorld` via `Arc<Mutex<EditorWorld>>`:

- `HierarchyPanel::ui(&mut self, ui: &mut egui::Ui, world: &mut EditorWorld)` — renders a tree view of `SceneNode`s with drag-and-drop reparenting (dispatches `ReparentCommand`).
- `InspectorPanel::ui(&mut self, ui: &mut egui::Ui, world: &mut EditorWorld)` — reflects the selected entity's components using a `ComponentUi` registry mapping `TypeId → Box<dyn ComponentUi>`.
- `ViewportToolbar::ui(&mut self, ui: &mut egui::Ui, state: &mut ViewportState)` — gizmo mode toggle, snap settings, camera speed.

---

## 2.2 Material Editor

### Graph Data Model

```rust
// crates/vox_app/src/editor/material_graph.rs

pub struct MaterialGraph {
    pub nodes: HashMap<NodeId, MaterialNode>,
    pub edges: Vec<Edge>,
}

pub struct Edge {
    pub from_node: NodeId,
    pub from_pin:  PinId,
    pub to_node:   NodeId,
    pub to_pin:    PinId,
}

pub enum MaterialNode {
    SpectralCurve(SpectralCurveNode),
    TextureSample(TextureSampleNode),
    Multiply(BinaryOpNode),
    Add(BinaryOpNode),
    Lerp(LerpNode),
    Fresnel(FresnelNode),
    SpectralEmission(SpectralEmissionNode),
    Output(OutputNode),
}
```

Each node variant carries its parameter values and a `NodeLayout { position: egui::Pos2 }` for the canvas position. `NodeId` is a `u32` newtype, `PinId` is a `u8` (pin index within the node).

### SpectralCurveNode

The most distinctive node type in the system. Stores 8 control points, one per spectral band, as `[f32; 8]` where values are in `[0, 1]`. The curve is displayed in egui as an 8-bar equalizer with draggable bars. Between bands, values are interpolated by a natural cubic spline (computed on the CPU at edit time, not at render time) to produce a smooth spectral reflectance profile. The interpolated 8-value array is the node's output pin type `SpectralProfile`.

```rust
pub struct SpectralCurveNode {
    pub control_points: [f32; 8],
    pub label:          String,
}

impl SpectralCurveNode {
    pub fn evaluate(&self) -> [f32; 8] {
        self.control_points  // Direct output; interpolation is for the UI preview, not for shader output
    }
}
```

The egui widget `spectral_curve_widget(ui, &mut node.control_points)` renders a 200×80 pixel widget. Each of the 8 bars is a draggable rect. A thin connecting line (using `egui::Painter::line_segment`) drawn between bar tops provides a visual spline hint.

### Graph Evaluation

`MaterialGraph::evaluate()` performs a topological sort of nodes (Kahn's algorithm on the `edges` adjacency list), then evaluates nodes in topological order, passing output pin values to the input pins of downstream nodes via a `HashMap<(NodeId, PinId), PinValue>` scratch map.

```rust
pub enum PinValue {
    Float(f32),
    SpectralProfile([f32; 8]),
    TextureHandle(AssetPath),
}
```

The `Output` node writes its input values to a `SpectralMaterialInstance`:

```rust
pub struct SpectralMaterialInstance {
    pub bands:          [f32; 8],
    pub roughness:      f32,
    pub metallic:       f32,
    pub emissive_scale: f32,
    pub translucency:   [f32; 8],
}
```

This instance is serialized to GPU memory via a per-material uniform buffer, updated within one frame of any graph change.

### Hot-Reload Pipeline

`MaterialHotReload` (existing in `material_hotreload.rs`) watches material graph files for changes via `notify` crate. When a change is detected, `MaterialGraph::evaluate()` runs on the background asset cook thread, producing a new `SpectralMaterialInstance`. The GPU buffer is updated via `queue.write_buffer()` on the next frame. The scene updates within one frame of the file change with no pipeline rebuild required, because the material instance data is a uniform (not a pipeline specialization constant).

### Material Library

20 preset spectral materials stored in `assets/materials/presets/` as `.spectral_mat` files (TOML format with embedded `SpectralCurveNode` control point arrays). Measurements sourced from spectroradiometer data:

- Skin (Fitzpatrick II): high band-2 and band-3 absorption (hemoglobin), subsurface emphasis.
- Glass (borosilicate): translucency `[f32; 8]` = ~`[0.92; 8]` flat; roughness 0.0.
- Foliage (green leaf): strong band-3 and band-4 reflectance (chlorophyll peak ~550nm), low band-0 and band-7.
- Metal (aluminum): near-flat high reflectance across all 8 bands; metallic 1.0.
- Concrete: low uniform reflectance, moderate roughness, zero translucency.
- Water (clear): very low reflectance in bands 0-3, moderate band-4 peak, specular emphasis.
- (14 more: rust, asphalt, sand, snow, bark, ceramic, cloth-cotton, cloth-silk, gold, copper, LED-warm, LED-cool, neon-red, neon-blue)

### Export Format

`.spectral_mat` is TOML with:
```toml
[meta]
name    = "Borosilicate Glass"
version = 2

[curve]
bands = [0.92, 0.93, 0.94, 0.94, 0.93, 0.92, 0.91, 0.90]

[properties]
roughness      = 0.0
metallic       = 0.0
emissive_scale = 0.0

[translucency]
bands = [0.92, 0.93, 0.94, 0.94, 0.93, 0.92, 0.91, 0.90]

[[nodes]]
# ... serialized MaterialGraph nodes for graph-based materials
```

---

## 2.3 Terrain Editor

### Brush Architecture

```rust
// crates/vox_app/src/editor/terrain_brush.rs

pub trait TerrainBrush: Send + Sync {
    fn apply(
        &self,
        vol:      &mut TerrainVolume,
        center:   glam::Vec3,
        radius:   f32,
        strength: f32,
        dt:       f32,
    );
    fn name(&self) -> &'static str;
    fn preview_radius(&self) -> f32;  // may differ from brush radius (e.g., noise adds stochastic extent)
}
```

**Concrete brush types:**

- `RaiseBrush`: at each SDF sample within `radius`, `sdf[p] -= strength * dt * falloff(distance(p, center) / radius)`, where `falloff` is a smooth quintic `f(r) = 1 - 6r^5 + 15r^4 - 10r^3`.
- `LowerBrush`: same but adds instead of subtracts.
- `SmoothBrush`: computes the mean SDF value in the radius and blends each voxel toward the mean by `strength * dt`.
- `FlattenBrush`: records the SDF value at `center` on mouse-down; each subsequent sample in the stroke blends toward `target_height * strength * dt`.
- `NoiseBrush`: applies a `simplex_noise_3d(p * frequency)` offset scaled by `strength * dt`. Frequency and octaves are brush parameters.
- `CarveSphereBrush`: `sdf[p] = max(sdf[p], distance(p, center) - radius)` — creates a spherical void.
- `FillSphereBrush`: `sdf[p] = min(sdf[p], distance(p, center) - radius)` — fills a sphere with material.

All brushes work in world space. `TerrainVolume::sample_mut(world_pos)` converts to voxel index via `floor((world_pos - origin) / voxel_size)` and returns a mutable `f32` reference. Bounds checking clamps to the volume extent without panicking.

### Cursor Raycast

Mouse pixel → camera ray → SDF bisection:

```rust
pub fn raycast_sdf(ray: Ray, vol: &TerrainVolume, max_steps: u32) -> Option<glam::Vec3> {
    let mut t = 0.0f32;
    for _ in 0..max_steps {
        let p = ray.origin + ray.direction * t;
        let d = vol.sample(p);  // trilinear interpolated SDF
        if d.abs() < 0.01 { return Some(p); }
        if d < 0.0 {
            // Overshot: bisect back
            t -= d.abs() * 0.5;
        } else {
            t += d * 0.9;  // sphere-march forward, 0.9 safety factor
        }
        if t > vol.diagonal_length() { return None; }
    }
    None
}
```

The brush preview is a translucent disc/sphere rendered as a splat assembly (a ring of low-opacity splats) positioned at the raycast hit, updated every mouse-move event.

### Undo for Terrain Strokes

`TerrainStrokeCommand` stores a sparse `HashMap<VoxelIndex, f32>` of before/after SDF values for all voxels modified during a single brush stroke (mouse-down to mouse-up). Before-values are recorded on the first modification of each voxel per stroke. The command uses `BitVec` to compactly track which voxels were dirtied; at `undo()`, only the recorded voxels are restored. Memory cost: a typical 10m-radius stroke modifying ~50K voxels at `f32` = 200 KB per undo entry. The history is capped at 64 terrain stroke entries; older entries are dropped.

### Material Paint

`TerrainMaterialBrush` writes to a separate `material_weight_texture: wgpu::Texture` (Rgba8Unorm, 4 material weights per texel, tiling to cover terrain chunks). The brush blends the target material index's weight channel upward and normalizes all 4 channels to sum to 1.0. The terrain shader samples this texture and blends between the 4 `SpectralMaterialInstance` uniforms accordingly.

### Procedural Layer System

```rust
pub struct LayerRule {
    pub condition:     LayerCondition,
    pub material_index: u8,
    pub blend:         f32,
}

pub enum LayerCondition {
    HeightRange  { min: f32, max: f32, falloff: f32 },
    SlopeRange   { min_deg: f32, max_deg: f32, falloff: f32 },
    CurvatureRange { min: f32, max: f32 },
}
```

`ProceduralLayerBake::run(&[LayerRule], vol: &TerrainVolume) -> MaterialWeightTexture` iterates every surface voxel (where `|SDF| < voxel_size`), computes height (from world position), slope (from `‖∇SDF‖`), and curvature (from `∇²SDF` via central differences), evaluates each rule's condition with falloff blending, and accumulates normalized weights into the output texture. This runs on a rayon threadpool and is triggered on demand from the editor "Bake Layers" button.

### Foliage Scatter

`FoliageBrush` samples surface points within `radius` using Poisson-disk sampling (Bridson 2007) at the requested `density` (splats per m²). At each sample:

1. SDF gradient `g = ∇SDF(p)` (normalized) gives surface normal.
2. Check `dot(g, up) >= cos(max_slope_deg)` — reject if too steep.
3. Load the `splat_assembly` from `AssetRegistry` (a pre-authored `Vec<GaussianSplat>` in object space).
4. Transform each splat by: translate to `p`, rotate to align splat up-axis with `g`, apply `jitter` (random rotation around `g`).
5. Append to the scene's splat buffer.

The resulting foliage splats are stored as a `FoliageLayer` component on the terrain entity, not baked into the base terrain splat cloud, so they can be cleared and re-scattered without re-baking the terrain.

### Erosion Bake

`HydraulicErosionBaker::run(&mut TerrainVolume, iterations: u32)` runs a water particle simulation on the rayon threadpool:

- Spawn 1 water particle per iteration at a random surface point.
- Each particle: flow downhill along `−∇SDF` (gradient descent), carry sediment capacity proportional to velocity.
- At each step: if `capacity > sediment`, erode (decrease SDF, add to `sediment`); else deposit (increase SDF, decrease `sediment`).
- Evaporate water linearly; stop when water_volume < 0.001.

1,000 iterations complete in approximately 4 seconds on a 256³ volume using rayon parallelism (particles are independent). The SDF is protected by a `RwLock` per chunk to avoid write contention. Output is a modified `TerrainVolume` with realistic drainage channels, alluvial fans, and ridge sharpening.

### Heightmap Import

`HeightmapImporter::import(path: &Path, scale: HeightmapScale) -> TerrainVolume` reads a 32-bit EXR file via the `exr` crate, treating pixel values as height in meters. SDF conversion uses the "exact SDF from heightfield" algorithm: for each voxel, the SDF value is set to `height_at_xy - voxel_z` clamped by the nearest-neighbor distance. This is not a true Euclidean SDF (which would require a 3D EDT), but for a smooth heightfield it is an excellent approximation within a few voxels of the surface. True EDT can be baked offline as a post-process if needed.

### Terrain Streaming

Terrain is divided into `TerrainChunk` cells, each owning a `TerrainVolume` of configurable resolution (default 64³ voxels, 1 m/voxel = 64m per side). A `TerrainStreamingManager` maintains a `HashMap<ChunkCoord, ChunkState>` where `ChunkState` is `Unloaded | Loading | Loaded(TerrainVolume)`. As the camera moves, chunks within `LOAD_RADIUS = 5` cells are scheduled for background loading; chunks beyond `UNLOAD_RADIUS = 7` are serialized to disk and freed. Chunk files are stored as `terrain/{cx}_{cz}.vxm` in the project directory.

---

## 2.4 Visual Scripting (Spectral Blueprint)

### Graph Model

```rust
// crates/vox_app/src/editor/visual_script.rs

pub struct VisualScript {
    pub name:      String,
    pub nodes:     HashMap<NodeId, ScriptNode>,
    pub edges:     Vec<ScriptEdge>,
    pub variables: HashMap<String, ScriptVariable>,
}

pub struct ScriptVariable {
    pub name:     String,
    pub var_type: PinType,
    pub default:  ScriptValue,
}

pub struct ScriptEdge {
    pub from_node: NodeId,
    pub from_pin:  PinId,
    pub to_node:   NodeId,
    pub to_pin:    PinId,
}
```

### Node Categories

**Event nodes** (one execution pin out):
- `OnTick { delta_seconds: PinOut<Float> }` — fires every frame.
- `OnBeginPlay` — fires once when the scene starts.
- `OnOverlap { other_entity: PinOut<EntityRef> }` — fires on physics overlap trigger.
- `OnSpectralPulse { assembly_name: String, band: u8, threshold: f32; entity: PinOut<EntityRef> }` — fires when the average value of spectral band `band` across a named splat assembly exceeds `threshold`. This node has no Unreal equivalent; it enables spectral-reactive gameplay (e.g., a door that opens when a UV emitter shines on it).
- `OnImpact { impulse: PinOut<Float>, normal: PinOut<Vec3> }` — fires on physics impact.

**Action nodes** (execution in + out):
- `MoveTo { target: PinIn<Vec3>, speed: PinIn<Float> }` — moves the entity to target via `AnimationDriver`.
- `SpawnEmitter { emitter_path: PinIn<AssetPath>, at: PinIn<Vec3> }` — spawns a `SplatEmitter`.
- `PlayAudio { clip_path: PinIn<AssetPath>, volume: PinIn<Float> }`.
- `SetSpectralBand { entity: PinIn<EntityRef>, band: PinIn<u8>, value: PinIn<Float> }` — sets a single spectral band on a splat assembly.
- `FireRayCast { origin: PinIn<Vec3>, direction: PinIn<Vec3>; hit: PinOut<Bool>, hit_pos: PinOut<Vec3>, hit_entity: PinOut<EntityRef> }`.

**Value nodes** (data only, no execution pins):
- `GetPosition { entity: PinIn<EntityRef>; position: PinOut<Vec3> }`.
- `GetSpectralBand { entity: PinIn<EntityRef>, band: PinIn<u8>; value: PinOut<Float> }`.
- `MathOp { op: MathOpKind, a: PinIn<Float>, b: PinIn<Float>; result: PinOut<Float> }`.

**Flow nodes** (execution branching):
- `Branch { condition: PinIn<Bool>; true_exec: PinOut<Exec>, false_exec: PinOut<Exec> }`.
- `Sequence { out_1: PinOut<Exec>, out_2: PinOut<Exec>, out_3: PinOut<Exec> }` — fires each exec pin in order.
- `ForEach { collection: PinIn<EntityList>; body: PinOut<Exec>, element: PinOut<EntityRef> }`.
- `Delay { seconds: PinIn<Float>; then: PinOut<Exec> }` — uses Rhai's coroutine/async.
- `Gate { enabled: PinIn<Bool>; in: PinIn<Exec>, out: PinOut<Exec> }` — passes exec through only when enabled.

### Pin Types

```rust
pub enum PinType {
    Execution,
    Float,
    Vec3,
    EntityRef,
    SpectralProfile,  // [f32; 8]
    Bool,
    String,
}
```

Execution pins use a white wire; `Float` uses yellow; `Vec3` uses blue; `EntityRef` uses orange; `SpectralProfile` uses a rainbow gradient wire drawn via egui `Painter::line_segment` calls per pixel segment.

### Compiler: VisualScript → Rhai

`VisualScriptCompiler::compile(script: &VisualScript) -> String` produces a Rhai script string:

1. Topological sort of all non-event nodes reachable from event node exec chains.
2. For each sorted node, `emit_node(node) -> String` produces a Rhai snippet. Examples:
   - `MathOp { Add }`: `let var_42 = var_38 + var_40;`
   - `SetSpectralBand`: `set_spectral_band(entity_var, 3, val_var);`
   - `Branch`: `if cond_var { /* true branch */ } else { /* false branch */ }`
3. Event nodes emit Rhai function definitions wrapping their subtree: `fn on_tick(delta) { ... }`.
4. `OnSpectralPulse` emits a registration call: `register_spectral_pulse("assembly_name", 2, 0.7, |entity| { ... })`.
5. Variables panel contents emit Rhai `let` declarations at the top of the script.

The compiled Rhai string is written to `<script_name>.rhai` in the project's `scripts/compiled/` directory. The Rhai engine runtime (`vox_script`) loads this file. Hot-reload: the file watcher detects the `.rhai` write and triggers `RhaiEngine::reload_script(path)`, completing within one frame.

### Live Debugging

When playing in editor mode, `VisualScriptDebugger::tick()` receives execution trace events from the Rhai runtime via a `crossbeam_channel`. Each event is `DebugEvent::NodeExecuted { node_id, pin_values: HashMap<PinId, ScriptValue> }`. The debugger maintains `last_executed: HashMap<NodeId, (Instant, HashMap<PinId, ScriptValue>)>`. The egui canvas dims nodes that have not executed in the last 500 ms and highlights recently-executed nodes with a pulsing border animation. Hovering a pin shows a tooltip with its last value.

---

## 2.5 Asset Pipeline

### AssetRegistry

```rust
// crates/vox_app/src/editor/asset_registry.rs

pub struct AssetRegistry {
    pub assets: HashMap<AssetPath, AssetMeta>,
    pub cook_queue: crossbeam_channel::Sender<CookRequest>,
    pub dirty:  HashSet<AssetPath>,
}

pub struct AssetMeta {
    pub source_path: PathBuf,
    pub cooked_path: PathBuf,
    pub asset_type:  AssetType,
    pub hash:        u64,  // xxhash3 of source file contents
    pub deps:        Vec<AssetPath>,
    pub state:       CookState,
}

pub enum CookState {
    Uncoooked,
    Cooking { started: Instant },
    Cooked { timestamp: Instant },
    Failed { error: String },
}
```

A background `CookThread` receives `CookRequest { path: AssetPath }` from the channel, runs the appropriate importer, and sends a `CookResult` back via a response channel. The main thread polls the response channel on each frame update and updates `AssetMeta.state`.

### Importers

All importers implement:

```rust
pub trait AssetImporter: Send + Sync {
    fn extensions(&self) -> &[&'static str];
    fn import(&self, path: &Path, registry: &AssetRegistry) -> Result<CookedAsset, ImportError>;
}
```

- **`PlyImporter`**: parses PLY files via `ply-rs` crate. Reads `x, y, z, scale_0..2, rot_0..3, opacity, f_dc_0..2` properties. Converts DC spherical harmonics coefficients to RGB via `color = SH0 * 0.2821 + 0.5`, then uplifts to spectral via `SpectralUplifter`. Outputs `Vec<GaussianSplat>`.
- **`GltfImporter`**: parses via `gltf` crate. Extracts mesh geometry, materials (PBR metallic-roughness), and skeleton. Passes mesh to `MeshToSplatConverter`. Outputs `SplatCloud + SplatSkeleton + Vec<AnimationClip>`.
- **`FbxImporter`**: uses `fbxcel-dom` crate for DOM-based FBX parsing. Same output as `GltfImporter`. FBX support targets the FBX 2020 binary format only.
- **`ExrImporter`**: reads `f32` EXR layers via the `exr` crate. Used for HDR skyboxes and heightmaps.
- **`PngImporter`**: reads via `image` crate. Used for material textures; outputs `Rgba8Unorm` GPU texture.
- **`HdrImporter`**: reads Radiance HDR (.hdr) via `image::io::Reader`. Used for environment maps.

### Mesh to Splat Conversion

`MeshToSplatConverter::convert(mesh: &GltfMesh, config: &ConversionConfig) -> Vec<GaussianSplat>`:

1. **Poisson disk sampling** on mesh surface: generate candidate samples by randomly sampling triangles proportional to area, then apply dart-throwing rejection until minimum separation `r = 1 / (2 * sqrt(config.density))` is satisfied between all accepted samples. Implementation uses a spatial grid (cell size `r/sqrt(2)`) for O(N) rejection testing.
2. Per sample: `position = surface_point`.
3. `scale = [r/2, r/2, r/2]` (initial isotropic, the optimizer can later refine).
4. `rotation` = quaternion aligning Z-axis with surface normal (Gram-Schmidt against world up, then to normal).
5. `opacity = 200` (u8, ≈ 0.78 initial opacity; allows optimizer headroom).
6. `spectral`: call `SpectralUplifter::lift(albedo_rgb, roughness, metallic)`.

`SpectralUplifter::lift(rgb: [f32;3], roughness: f32, metallic: f32) -> [u16; 8]`:
- Uses Smits (1999) basis function decomposition extended to 8 bands. The 8-band Smits basis matrix `B: [[f32;8]; 3]` maps RGB to an initial 8-band estimate.
- Metallic adjustment: lerp each band toward the metal reflectance function (near-flat high reflectance) by `metallic`.
- Roughness adjustment: no per-band adjustment (roughness affects the Gaussian shape, not the spectral value).
- Pack output as `u16` f16 bits via `half::f16::from_f32(v).to_bits()`.

### Splat Compression

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatCompressed {
    // Position: 16-bit fixed point, 1mm precision over [-16m, +16m] per axis
    pub pos_xy:     u32,  // packed as (pos_x_u16 << 16) | pos_y_u16
    pub pos_z_rot:  u32,  // pos_z in high u16, rotation.w in low u16 (largest component encoding)
    pub scale_opacity: u32,  // scale_x u8 << 24 | scale_y u8 << 16 | scale_z u8 << 8 | opacity
    pub spectral:   [u16; 8],  // f16 bits, unchanged from GaussianSplat
    // Total: 4 + 4 + 4 + 16 = 28 bytes + alignment → 32 bytes padded, vs. 52 bytes uncompressed
}
```

The rotation is stored using the "smallest three" encoding: the largest quaternion component is identified and its sign is stored in a bit; the remaining three are packed as i8 (scaled to [-1, 1]). This fits in 32 bits. Position quantization: `u16_val = round((world_pos - chunk_origin) / 0.001)` where `chunk_origin` and scale are stored per `SplatTile`.

### Streaming Format

```rust
pub struct SplatTile {
    pub bounds:             vox_core::Aabb,
    pub chunk_origin:       [f32; 3],
    pub compressed_splats:  Vec<SplatCompressed>,
}
```

Tiles are serialized to `.vxm` binary files (the existing Ochroma asset format from `vox_data`). On load, `SplatCompressed` is decompressed to `GaussianSplat` on a rayon threadpool before upload to the GPU splat buffer.

### Dependency Graph

`AssetRegistry::mark_dirty(path)` walks the dependency graph (stored as an adjacency list in `AssetMeta.deps`) and marks all transitive dependents dirty. The cook thread processes dirty assets in topological order (leaf assets first). Cycles are detected during import and reported as errors (they cannot occur in a valid asset tree but malformed hand-edited files may cause them).

---

## 2.6 Profiler & Frame Debugger

### Tracy Integration

Every major subsystem wraps its per-frame work in `tracing::span!()` calls, collected by `tracing-tracy`. The Tracy client connects automatically when the engine is launched with `OCHROMA_TRACY=1`. Spans are named consistently: `"render/tile_assign"`, `"render/radix_sort"`, `"render/splat_raster"`, `"gi/probe_update"`, `"shadow/atlas_pack"`, etc.

### GPU Timestamps

```rust
// crates/vox_render/src/profiling.rs (extend existing)

pub struct GpuTimestampRing {
    pub query_sets:     Vec<wgpu::QuerySet>,  // double-buffered, 2 per pass × 64 passes
    pub resolve_buffer: wgpu::Buffer,
    pub readback_buffer: wgpu::Buffer,
    pub history:        VecDeque<FrameSnapshot>,  // last 120 frames
    current_frame:      usize,
}

pub struct FrameSnapshot {
    pub frame_index:   u64,
    pub cpu_ms:        f32,
    pub gpu_passes:    Vec<GpuPassTiming>,
    pub splat_count:   u32,
    pub memory:        GpuMemorySnapshot,
}

pub struct GpuPassTiming {
    pub name:   &'static str,
    pub gpu_ms: f32,
}
```

Two `wgpu::QuerySet` instances (double-buffered) hold `TIMESTAMP_QUERY_COUNT = 128` timestamp queries. At the start/end of each render pass, `encoder.write_timestamp(&query_set, idx)` records the GPU clock. After the frame, `encoder.resolve_query_set()` copies timestamps to `resolve_buffer`, then `queue.submit()`, then `resolve_buffer.slice(..).map_async(Read)` schedules a readback. The previous frame's resolved buffer is read on CPU the following frame (one frame latency), populating `GpuPassTiming.gpu_ms` via `(end_ts - start_ts) * timestamp_period_ns / 1_000_000.0`.

### ProfilerUi

```rust
// crates/vox_app/src/editor/profiler_ui.rs

pub struct ProfilerUi {
    pub visible:        bool,
    pub selected_frame: Option<usize>,  // index into GpuTimestampRing.history
    pub cpu_flame:      FlameGraph,
    pub gpu_timeline:   GpuTimeline,
}
```

Rendered as an egui window. The CPU flame graph uses `egui::Painter` to draw horizontal bars for Tracy span data (hierarchical, color-coded by subsystem). The GPU timeline is a horizontal waterfall chart: each `GpuPassTiming` is a colored bar with width proportional to `gpu_ms`. Hovering any bar shows a tooltip with name and exact timing. Clicking a frame index in a "last 120 frames" sparkline sets `selected_frame`, freezing the display on that frame's data.

### PerSplatDebugView

```rust
pub enum PerSplatDebugView {
    None,
    SplatCount,
    Opacity,
    SpectralBand(usize),  // 0..7
    LodLevel,
    TileId,
}
```

When not `None`, the `splat_raster.wgsl` is recompiled with a `#define DEBUG_MODE N` preprocessor define (wgpu shader macro via `naga` preprocessor or string substitution before compilation). In debug mode, the rasterizer outputs a heat-map color instead of the spectral accumulation result:
- `SpectralBand(b)`: maps `splat.spectral[b]` to a blue-red gradient.
- `Opacity`: maps `splat.opacity` to blue-red.
- `TileId`: maps tile index modulo 16 to a categorical color palette.

### Memory Tracker

```rust
pub struct GpuMemoryTracker {
    pub splat_buffer_mb:    f32,
    pub texture_mb:         f32,
    pub shadow_atlas_mb:    f32,
    pub gi_probe_mb:        f32,
    pub froxel_volume_mb:   f32,
    pub oit_textures_mb:    f32,
    pub total_mb:           f32,
}
```

Each subsystem reports its allocation size in `GpuMemoryTracker::update()`. The profiler UI shows a stacked bar chart (egui bar chart widget) of these categories, updated every second. An alert threshold at `VRAM_WARN_MB = 7168` (7 GB) triggers a yellow warning in the UI.

### RenderDoc Integration

`FrameDebugger::capture_frame()` is called when F12 is pressed in the editor. It calls into the RenderDoc API via `renderdoc-sys` crate:

```rust
#[cfg(feature = "renderdoc")]
pub fn capture_frame(&self) {
    use renderdoc::{RenderDoc, V141};
    if let Ok(mut rd) = RenderDoc::<V141>::new() {
        rd.trigger_capture();
    }
}
```

The `renderdoc` feature is enabled only in debug builds. The capture opens automatically in the RenderDoc UI if RenderDoc is installed. A status bar message in the editor confirms "RenderDoc capture triggered" or "RenderDoc not available" if the library is not found.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_app/src/editor/world.rs` | CREATE | `EditorWorld`, `SceneNode`, `LocalTransform`, world transform cache |
| `crates/vox_app/src/editor/command.rs` | CREATE | `EditorCommand` trait, `CommandHistory`, `SpawnEntityCommand`, `DeleteEntityCommand`, `MoveEntityCommand`, `SetComponentCommand`, `ReparentCommand` |
| `crates/vox_app/src/editor/selection.rs` | CREATE | `SelectionState`, `PivotMode`, `SetSelectionCommand` |
| `crates/vox_app/src/editor/gizmos.rs` | CREATE | `TranslateGizmo`, `RotateGizmo`, `ScaleGizmo`, hit test, snapping |
| `crates/vox_app/src/editor/viewport.rs` | CREATE | `EditorViewport`, `ViewType`, 4-pane layout, `EditorLayout` |
| `crates/vox_app/src/editor/panels/hierarchy.rs` | CREATE | `HierarchyPanel` egui widget with drag-and-drop reparent |
| `crates/vox_app/src/editor/panels/inspector.rs` | CREATE | `InspectorPanel`, `ComponentUi` registry |
| `crates/vox_app/src/editor/panels/toolbar.rs` | CREATE | `ViewportToolbar`, gizmo mode buttons, snap controls |
| `crates/vox_app/src/editor/material_graph.rs` | CREATE | `MaterialGraph`, `MaterialNode`, `Edge`, `SpectralCurveNode`, graph evaluation, topological sort |
| `crates/vox_app/src/editor/material_library.rs` | CREATE | 20 preset loader, `.spectral_mat` TOML serialize/deserialize |
| `crates/vox_app/src/editor/widgets/spectral_curve.rs` | CREATE | egui 8-bar spectral curve widget |
| `crates/vox_app/src/editor/terrain_brush.rs` | CREATE | `TerrainBrush` trait, all 7 brush impls, raycast, `TerrainStrokeCommand` |
| `crates/vox_app/src/editor/terrain_procedural.rs` | CREATE | `LayerRule`, `LayerCondition`, `ProceduralLayerBake`, erosion baker |
| `crates/vox_app/src/editor/foliage_brush.rs` | CREATE | `FoliageBrush`, Poisson-disk scatter, `FoliageLayer` component |
| `crates/vox_app/src/editor/terrain_streaming.rs` | CREATE | `TerrainStreamingManager`, `TerrainChunk`, background load/unload |
| `crates/vox_app/src/editor/visual_script.rs` | CREATE | `VisualScript`, `ScriptNode` enum, `ScriptEdge`, `ScriptVariable`, all node types |
| `crates/vox_app/src/editor/visual_script_compiler.rs` | CREATE | `VisualScriptCompiler::compile()`, Rhai code generation per node type |
| `crates/vox_app/src/editor/visual_script_debugger.rs` | CREATE | `VisualScriptDebugger`, trace event channel, egui overlay |
| `crates/vox_app/src/editor/asset_registry.rs` | CREATE | `AssetRegistry`, `AssetMeta`, `CookState`, background cook thread, dependency graph |
| `crates/vox_app/src/editor/importers/ply.rs` | CREATE | `PlyImporter` — PLY → `Vec<GaussianSplat>` |
| `crates/vox_app/src/editor/importers/gltf.rs` | CREATE | `GltfImporter` — glTF → `SplatCloud + SplatSkeleton + AnimationClips` |
| `crates/vox_app/src/editor/importers/fbx.rs` | CREATE | `FbxImporter` — FBX 2020 binary → same as glTF output |
| `crates/vox_app/src/editor/importers/exr.rs` | CREATE | `ExrImporter` — EXR → heightmap or HDR environment |
| `crates/vox_app/src/editor/importers/image.rs` | CREATE | `PngImporter`, `HdrImporter` |
| `crates/vox_app/src/editor/mesh_to_splat.rs` | CREATE | `MeshToSplatConverter`, Poisson-disk sampler, `SpectralUplifter` (Smits basis) |
| `crates/vox_app/src/editor/splat_compression.rs` | CREATE | `SplatCompressed`, pack/unpack, "smallest three" rotation encoding |
| `crates/vox_app/src/editor/profiler_ui.rs` | CREATE | `ProfilerUi`, `FlameGraph`, `GpuTimeline`, memory bar chart |
| `crates/vox_render/src/profiling.rs` | MODIFY | Add `GpuTimestampRing`, `FrameSnapshot`, `GpuPassTiming`, `GpuMemoryTracker` |
| `crates/vox_render/src/frame_debugger.rs` | MODIFY | Add RenderDoc capture trigger via `renderdoc-sys` |
| `crates/vox_app/src/editor/mod.rs` | CREATE | Module root, re-exports |
| `assets/materials/presets/` | CREATE | 20 `.spectral_mat` TOML files |

---

## Milestones

### M1 — Core Editor Shell (Weeks 1–8)
`EditorWorld`, `CommandHistory`, all 5 command types, egui window layout with `HierarchyPanel` and `InspectorPanel`. Basic viewport rendering the scene via the existing `GpuRasteriser`. Multi-select. Undo/redo working for entity spawn/delete/move.

### M2 — Gizmos + Material Editor (Weeks 9–20)
`TranslateGizmo`, `RotateGizmo`, `ScaleGizmo` with snapping. `MaterialGraph` evaluation, `SpectralCurveNode` widget, `SpectralMaterialInstance` hot-reload. Material library with 20 presets loadable from the editor.

### M3 — Terrain Editor (Weeks 21–34)
All 7 brush types, SDF raycast cursor, brush preview, undo/redo, material paint. Procedural layer bake. Foliage scatter. Erosion bake. Heightmap import. Streaming.

### M4 — Visual Scripting (Weeks 35–46)
All node categories implemented. Compiler producing valid Rhai. `OnSpectralPulse` event working end-to-end. Live debugger showing node execution highlight in editor play mode.

### M5 — Asset Pipeline (Weeks 47–58)
`AssetRegistry`, background cook thread, `PlyImporter`, `GltfImporter`, `FbxImporter`, `ExrImporter`. `MeshToSplatConverter` with Poisson-disk sampling and `SpectralUplifter`. `SplatCompressed` format. Dependency graph dirty-marking.

### M6 — Profiler & Frame Debugger (Weeks 59–66)
`GpuTimestampRing`, `ProfilerUi` with CPU + GPU timeline, `PerSplatDebugView` overlays, `GpuMemoryTracker`, RenderDoc integration.

---

## Acceptance Criteria

- Undo/redo for entity spawn, delete, move, reparent, and component set operations all round-trip correctly with zero data loss.
- `CommandHistory` handles 1,000 consecutive undo/redo cycles on a 10,000-node scene in under 1 second total.
- Gizmo hit-testing correctly identifies the intended handle axis with no false positives at any camera angle (tested via automated raycast comparisons against known handle positions).
- `MaterialGraph::evaluate()` for a 20-node graph completes in under 1 ms.
- Material hot-reload: file change to GPU buffer update completes within one rendered frame (< 16 ms at 60 FPS).
- All 20 preset materials render visually correct spectral responses, verified against reference spectroradiometer data (per-band error < 5%).
- `TerrainBrush` undo restores all modified voxels to exact pre-stroke SDF values (bit-identical comparison).
- Foliage scatter respects `max_slope` constraint: no foliage placed on surfaces steeper than the configured limit (automated check: sample placed splat normals vs. up vector).
- `VisualScriptCompiler::compile()` produces syntactically valid Rhai for all node type combinations (fuzz test: generate random graphs, compile, run through Rhai parser, assert no parse errors).
- `OnSpectralPulse` node fires within one frame of the triggering condition being met.
- `PlyImporter` round-trips a reference PLY file (known splat positions, scales, rotations, colors) with positional error < 0.1 mm and per-band spectral error < 1%.
- `SplatCompressed` packs and unpacks with positional error < 1 mm and spectral error < 0.5% (f16 precision limit).
- GPU timestamps report per-pass timings within 5% of RenderDoc measured timings on the same frame.
- `PerSplatDebugView::SpectralBand(b)` overlay correctly colors splats proportional to their `spectral[b]` value (verified by comparing pixel colors at known splat screen positions against expected gradient values).
- Editor launches in under 3 seconds on the target workstation (Ryzen 5800X + RTX 3070).
- No game-specific concepts appear in any engine crate (`vox_core`, `vox_data`, `vox_render`, `vox_terrain`, `vox_audio`, `vox_physics`, `vox_script`).

---

## Effort

| Subsystem | Estimated Effort |
|-----------|-----------------|
| 2.1 Core Editor Architecture | 3 months |
| 2.2 Material Editor | 2 months |
| 2.3 Terrain Editor | 3.5 months |
| 2.4 Visual Scripting | 3 months |
| 2.5 Asset Pipeline | 3.5 months |
| 2.6 Profiler & Frame Debugger | 1.5 months |
| **Total** | **~16.5 months** |

Work can be parallelized across two engineers: one owning 2.1 + 2.2 + 2.6 (editor shell + visualization), one owning 2.3 + 2.4 + 2.5 (content authoring + pipeline). The `EditorWorld` and `CommandHistory` (M1) are on the critical path for all other editor subsystems and must ship first. The asset pipeline (M5) is partially independent but is needed to make the terrain editor and material editor work on non-trivial scenes, so M3/M4 can prototype with hard-coded test assets until M5 catches up.
