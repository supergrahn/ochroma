# Crucible & Forge — Complete Capability Reference

> Generated 2026-03-30 by exhaustive source reading.
> Use this document before making any architectural decisions about what to adopt.

---

# CRUCIBLE

## What It Is

Crucible is a **reactive DAG (Directed Acyclic Graph) scene evaluation engine**. It converts a `CrucibleDirective` JSON input into a typed node graph, evaluates it topologically (Kahn's algorithm, ascending NodeId for determinism), caches outputs per node, propagates dirty flags downstream on parameter change, and writes USD sublayers as output.

**Core pipeline:** `CrucibleDirective JSON → CrucibleGraph (nodes + ports) → Topological Cook → USD Sublayers → Spectra Renderer`

---

## Workspace Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| axum | 0.8 | HTTP async framework (cook-daemon) |
| uuid | 1 | Session tracking |
| hashbrown | 0.15 | Fast HashMap for graph edges |
| thiserror | 2.0 | Error handling |
| serde | 1.0 | Serialization |
| serde_json | 1.0 | JSON I/O |
| noise | 0.9 | Perlin fBm terrain |
| async-trait | 0.1 | Async trait objects |
| reqwest | 0.12 | HTTP client (Gemini API) |
| schemars | 0.8 | JSON schema generation |
| clap | 4 | CLI argument parsing |
| tokio | 1 | Async runtime |
| kamadak-exif | 0.5 | EXIF metadata |
| base64 | 0.22 | Base64 encoding |
| anyhow | 1.0 | Error context |
| sha2 | 0.10 | SHA hashing |
| rand | 0.8 | RNG |
| image | 0.25 | Image I/O |
| zip | 2 | ZIP archive |
| meshopt | 0.3 | LOD simplification |
| bytemuck | 1.0 | Safe byte casting |
| nalgebra | 0.33 | Linear algebra (FloraPrime) |

---

## crucible-types

**Standalone type definitions. No internal dependencies.**

### geometry.rs

```rust
pub struct FaceSet {
    pub material_id:  u32,
    pub face_indices: Vec<u32>,
}

pub struct Mesh {
    pub positions:           Vec<[f32; 3]>,
    pub normals:             Vec<[f32; 3]>,
    pub uvs:                 Vec<[f32; 2]>,
    pub tangents:            Option<Vec<[f32; 3]>>,
    pub indices:             Vec<u32>,           // flat triangle indices
    pub material_id:         u32,
    pub face_sets:           Vec<FaceSet>,
    pub subd:                bool,               // Catmull-Clark flag
    pub spectral_embedding:  Option<Vec<[f32; 6]>>, // per-vertex PCA spectral (PROSPECT-PRO at 6 wavelengths)
    pub crease_indices:      Vec<u32>,
    pub crease_lengths:      Vec<u32>,
    pub crease_sharpnesses:  Vec<f32>,           // 10.0 = infinitely sharp
}
// impl: triangle_count(), is_valid()

pub struct CurveData {
    pub points:                Vec<[f32; 3]>,
    pub curve_vertex_counts:   Vec<u32>,
    pub width:                 f32,
    pub material_id:           u32,
}
```

### terrain.rs

```rust
pub struct TerrainGrid {
    pub resolution:      u32,
    pub world_size:      f32,
    pub heights:         Vec<f32>,           // row-major, Z fastest
    pub material_ids:    Vec<u32>,           // per-quad
    pub flow_map:        Vec<f32>,
    pub deposition_map:  Vec<f32>,
    pub curvature_map:   Vec<f32>,
    pub disturbed_earth: Vec<f32>,
    pub aeolian_map:     Vec<f32>,
    pub biome_ids:       Vec<u8>,
    pub splat_weights:   Vec<[f32; 4]>,
}
// impl: new(), height_at(row, col) -> Option<f32> (bilinear)
```

### scatter.rs

```rust
pub struct ScatterInstances {
    pub positions:     Vec<[f32; 3]>,
    pub orientations:  Vec<[f32; 4]>,    // XYZW quaternions
    pub scales:        Vec<f32>,
    pub asset_ids:     Vec<u32>,
    pub mesh_labels:   Vec<String>,
    pub asset_paths:   Vec<String>,      // external USD file references
}
// impl: count(), is_valid()
```

### light.rs

```rust
pub enum LightType {
    Directional,
    Point   { radius: f32 },
    Area    { width: f32, height: f32 },
    Sky     { hdri_path: Option<String> },
}

pub struct LightDescriptor {
    pub direction:  [f32; 3],
    pub color:      [f32; 3],            // linear RGB
    pub intensity:  f32,                 // Lux (directional) / Lumens (point/area)
    pub light_type: LightType,
    pub role:       Option<String>,      // "key", "fill", "rim", "practical"
}
```

### camera.rs

```rust
pub struct CameraState {
    pub position:        [f32; 3],
    pub target:          [f32; 3],
    pub up:              [f32; 3],
    pub fov_deg:         f32,
    pub near:            f32,
    pub far:             f32,
    pub aperture_mm:     f32,            // 0 = pinhole
    pub focus_distance:  f32,
    pub focal_length_mm: Option<f32>,
    pub shot_type:       Option<String>, // "wide", "medium", "close"
}
// Default: 1.7m above z=-5, looking at origin, fov=60, aperture=0
```

### atmosphere.rs

```rust
pub struct AtmosphereState {
    pub turbidity:     f32,              // Hosek-Wilkie [1, 10], default 3.0
    pub sun_elevation: f32,              // degrees
    pub sun_azimuth:   f32,              // degrees
    pub ground_albedo: [f32; 3],         // linear RGB
}

pub struct FogState {
    pub density:        f32,
    pub color:          [f32; 3],
    pub height_falloff: f32,
}
```

### material.rs

```rust
pub struct MaterialSpec {
    pub name:                 String,
    pub base_color:           [f32; 3],
    pub roughness:            f32,
    pub metallic:             f32,
    pub emission:             [f32; 3],
    pub albedo_texture:       Option<String>,
    pub normal_texture:       Option<String>,
    pub roughness_texture:    Option<String>,
    pub emission_texture:     Option<String>,
    pub subsurface:           f32,
    pub subsurface_color:     [f32; 3],
    pub subsurface_radius:    [f32; 3],
    pub coat:                 f32,
    pub coat_roughness:       f32,
    pub anisotropy:           f32,
    pub sheen:                f32,
    pub displacement_texture: Option<String>,
    pub opacity:              f32,
    pub spectral_reflectance: Vec<(u32, [f32; 16])>, // (material_id, 16 wavelength @ 25nm)
}
```

### directive.rs

```rust
pub struct CrucibleDirective {
    pub volumes:    Vec<Volume>,
    pub scatter:    Vec<ScatterInstances>,
    pub lights:     Vec<LightDescriptor>,
    pub camera:     CameraState,
    pub atmosphere: Option<AtmosphereState>,
    pub fog:        Option<FogState>,
    pub materials:  Vec<MaterialSpec>,
    pub output_dir: PathBuf,
}
```

---

## crucible-core

### error.rs

```rust
pub enum CookError {
    MissingInput(String),
    TypeMismatch { port: String, expected: String, got: String },
    NodeNotFound(NodeId),
    CycleDetected { from: u32, to: u32 },
    UnknownPort { node: NodeId, port: String },
    CookFailed { node: String, reason: String },
    UnknownParam { key: String, node: String },
    InvalidParam { key: String, reason: String },
    Serialization(serde_json::Error),
}
```

### port.rs

```rust
pub enum PortDataType {
    Terrain, Geometry, LodGeometry, Instances,
    Light, Camera, Atmosphere, Fog, Material, Scalar, Null,
}

pub enum PortData {
    Terrain(TerrainGrid),
    Geometry(Mesh),
    LodGeometry(Vec<Mesh>),
    Instances(ScatterInstances),
    Light(LightDescriptor),
    Camera(CameraState),
    Atmosphere(AtmosphereState),
    Fog(FogState),
    Material(MaterialSpec),
    Scalar(f64),
    Null,
}

pub type PortMap = HashMap<String, PortData>;

pub enum ParamValue {
    Float(f64), Int(i64), Str(String), Bool(bool),
    Vec2([f64; 2]), Vec3([f64; 3]),
}
// impl: as_float(), as_int(), as_str(), as_bool(), as_vec2(), as_vec3(), as_float_coerce()
```

### node.rs

```rust
pub struct PortSpec {
    pub name:      &'static str,
    pub data_type: PortDataType,
    pub optional:  bool,
}

pub struct NodeDescriptor {
    pub type_name: &'static str,
    pub inputs:    Vec<PortSpec>,
    pub outputs:   Vec<PortSpec>,
}
// impl: input_type(name), output_type(name)

pub trait CrucibleNode: Send + Sync {
    fn descriptor(&self) -> NodeDescriptor;
    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), CookError>;
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError>;
}
```

### graph.rs

```rust
pub struct NodeId(pub u32);

pub struct CrucibleGraph { /* opaque */ }

impl CrucibleGraph {
    pub fn new() -> Self
    pub fn add_node(name: &str, node: Box<dyn CrucibleNode>) -> NodeId
    pub fn node_count() -> usize
    pub fn remove_node(id: NodeId) -> Result<(), CookError>
    pub fn is_dirty(id: NodeId) -> bool
    // Edge management:
    pub fn connect(from: NodeId, from_port: &str, to: NodeId, to_port: &str) -> Result<(), CookError>
    //   - type-checks declared ports
    //   - DFS cycle detection before insertion
    //   - idempotent: duplicate edges are no-ops
    pub fn downstream_of(id: NodeId) -> Result<Vec<NodeId>, CookError>
    pub fn upstream_of(id: NodeId) -> Result<Vec<NodeId>, CookError>
    // Topological sort (Kahn's, min-heap for determinism):
    pub fn topo_sort() -> Result<Vec<NodeId>, CookError>
    // Dirty management:
    pub fn mark_dirty(id: NodeId)           // cascades downstream
    pub fn mark_clean_all()
    pub fn set_param(id: NodeId, key: &str, value: ParamValue) -> Result<(), CookError>
    //   - marks node + all descendants dirty
    // Cooking:
    pub fn cook() -> Result<(), CookError>  // evaluates only dirty nodes in topo order, caches output
    pub fn get_output(id: NodeId, port: &str) -> Option<&PortData>
    // Serialization:
    pub fn snapshot() -> GraphSnapshot
}
```

### serial.rs

```rust
pub struct GraphSnapshot {
    pub nodes: Vec<NodeSnapshot>,
    pub edges: Vec<EdgeSnapshot>,
}
pub struct NodeSnapshot {
    pub id:        u32,
    pub name:      String,
    pub type_name: String,
    pub params:    serde_json::Value,
}
pub struct EdgeSnapshot {
    pub from: u32, pub from_port: String,
    pub to:   u32, pub to_port:   String,
}
// impl: to_json(), from_json()
```

### quality_report.rs

```rust
pub struct SceneQualityReport {
    pub lighting:   LightingRigReport,
    pub camera:     CameraReport,
    pub atmosphere: AtmosphereReport,
    pub scatter:    ScatterReport,
}
pub struct LightingRigReport {
    pub has_key: bool, pub has_fill: bool, pub has_rim: bool, pub has_practical: bool,
    pub light_count: usize,
}
pub struct CameraReport {
    pub focal_length_mm: f32, pub dof_enabled: bool, pub shot_type: String,
}
// from_directive(d: &CrucibleDirective) -> Self
// Shot type from FOV: <35° = "close", <60° = "medium", ≥60° = "wide"
// DOF enabled when aperture_mm > 0
```

---

## crucible-nodes

### TerrainNode — type_name: "terrain"

**Output**: `out: Terrain`

**Parameters**:
- `resolution` int [2, 8192] default 128
- `world_size` float default 500.0
- `seed` int default 0 (wraps to u32)
- `octaves` int default 6
- `amplitude` float default 80.0
- `displacement_map_path` str optional (PNG/JPEG)
- `displacement_scale` float default 0.3

**Algorithm**:
- `Fbm<Perlin>` with configurable octaves
- Sample point: `nx = 4.0 * col / (res-1) - 2.0`, same for z
- Height: `fbm.get([nx, nz]) * amplitude`
- Optional bilinear displacement map overlay

### ScatterNode — type_name: "scatter"

**Input**: `surface: Terrain` (required)
**Output**: `out: Instances`, `lod_levels: Scalar`

**Parameters**:
- `count` int default 100
- `seed` int default 0
- `margin` float [0, 0.49] default 0.05
- `asset_id` int default 0
- `asset_catalog_json` str optional: `[{id: u32, weight: f32}, ...]`
- `lod_levels_json` str optional

**Algorithm**:
- LCG: `x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407`, `seed ^= 0xdeadbeef`
- Random (x, z) in world bounds minus margin
- Bilinear Y from terrain grid
- Random yaw → quaternion `(0, sin(yaw/2), 0, cos(yaw/2))`
- Asset via CDF from catalog weights
- Scale: `0.8 + random * 0.4`

### SunLightNode — type_name: "sun_light"

**Parameters**: elevation (deg), azimuth (deg), intensity (Lux), color_r/g/b
**Algorithm**: `dy = -sin(el)`, `dx = cos(el)*sin(az)`, `dz = cos(el)*cos(az)`

### CameraNode — type_name: "camera"
### AtmosphereNode — type_name: "atmosphere"
### FogNode — type_name: "fog"

(Passthrough parameter nodes)

### lod.rs: `simplify_mesh(positions, indices, target_ratio) -> Vec<u32>`

Uses meshopt quadric simplification. Returns simplified index buffer.

### flora_prime_node.rs: FloraPrimeNode — type_name: "flora_prime"

**Output**: `out: Geometry`

**Species matching** (case-insensitive): spruce/fir/pine → conifer, cypress/poplar → columnar, oak → spreading, birch/beech, ash, willow → weeping

**Per-species parameters**: branch_angle, trunk_taper, branch_length_ratio

**PROSPECT-PRO leaf params**: n (refractive index), cab (chlorophyll a+b µg/cm²), car (carotenoid µg/cm²), cbrown (brown pigment), cw (water cm), cm (dry matter g/cm²), anth (anthocyanin µmol/cm²)

**Output**: Mesh with `spectral_embedding: Vec<[f32; 6]>` — PCA projection of PROSPECT-PRO at 6 wavelengths per vertex

### input_nodes.rs: Passthrough nodes

```rust
pub struct TerrainInputNode   { pub data: TerrainGrid }
pub struct MeshInputNode      { pub data: Mesh }
pub struct ScatterInputNode   { pub data: ScatterInstances }
pub struct LightInputNode     { pub data: LightDescriptor }
pub struct MeshLodInputNode   { pub data: Vec<Mesh> }
pub struct MaterialInputNode  { pub data: MaterialSpec }
```

### usd_export.rs: UsdExportNode — type_name: "usd_export"

**Inputs**: terrain, camera, atmosphere, fog (optional), dynamic light_*, scatter_*, material_*, geometry ports
**Output**: writes USD to disk

---

## crucible-usd

```rust
pub fn write_scene(
    output_dir:  &Path,
    terrain:     Option<&TerrainGrid>,
    meshes:      &[(String, &Mesh)],
    scatter:     &[&ScatterInstances],
    lod_meshes:  &[&Vec<Mesh>],
    lights:      &[&LightDescriptor],
    camera:      &CameraState,
    atmosphere:  &AtmosphereState,
    fog:         Option<&FogState>,
    materials:   &[MaterialSpec],
    curves:      &[(String, &CurveData)],
) -> Result<(), CookError>
```

**Output files**: geometry.usda, lighting.usda, camera.usda, atmosphere.usda, materials.usda, scene.usda (root layer composing all sublayers)

**USD Prim Hierarchy**:
```
/geometry (Scope, defaultPrim)
  /terrain (Mesh)
  /<label> (Mesh, per building/volume)
  /scatter_N (PointInstancer)
/lights (Scope)
  /light_N
/camera
/atmosphere
  /sky (DomeLight)
  /fog (Volume)
/materials (optional)
scene.usda: subLayers = [@geometry.usda@, @lighting.usda@, ...]
```

---

## cook binary (aetherspectra-cook)

- Reads `CrucibleDirective` JSON
- Builds `CrucibleGraph` programmatically
- Calls `graph.cook()`
- `UsdExportNode` writes USD output

---

## cook-daemon (aetherspectra-cook-daemon)

- Axum HTTP server, default port 7420
- Session-aware (UUID per request)
- Per-session result caching
- Async cooking via Tokio

---

## director (aetherspectra-director)

**AI scene direction pipeline.**

**Features**: `native-splat` (enables 3DGS via splat-train + CUDA)

**Stages**:
1. User natural language prompt
2. Gemini API → `CrucibleDirective` JSON
3. Invoke `aetherspectra-cook`
4. Optional: 3DGS training (`native-splat` feature)

---

## splat-train

**3D Gaussian Splatting trainer.**

**Dependencies**: spectra-gpu (CUDA), spectra-gaussian-render

**Public API**:
```rust
pub struct TrainerConfig {
    pub slang_dir:       PathBuf,
    pub iterations:      usize,
    // hyperparameters: lr_pos, lr_color, lr_sh, lr_opacity, lr_scale, lr_rot, etc.
}

pub struct SplatTrainer {
    pub fn new(config: TrainerConfig) -> Self
    pub fn train(&self, images: &[PathBuf], output_ply: &Path, gpu: &impl GpuBackend) -> Result<()>
}

pub fn train_on_device(
    device_idx: usize,
    config: TrainerConfig,
    images: &[PathBuf],
    output_ply: &Path,
) -> Result<()>  // requires `cuda` feature
```

**Algorithm**:
1. Load images, init Gaussians from SfM (or random)
2. Forward rasterization via `gaussian_rasterize.slang`
3. Per-Gaussian backward via `gaussian_backward.slang`
4. Adam optimizer updates (GPU)
5. Densification/pruning per N iterations
6. Export final splat model as PLY

**Modules**: adam.rs, buffers.rs, camera.rs, init.rs, sort.rs, trainer.rs, carve.rs, depth.rs

---

## Algorithms Summary

### Topological Sort (Kahn's)
1. Compute in-degrees
2. Queue nodes with in-degree 0 (min-heap for determinism by NodeId)
3. Dequeue, decrement children's in-degrees, enqueue those reaching 0
4. If result count < node count → cycle detected

### Dirty Propagation
- `mark_dirty(node)` → DFS cascade to all reachable descendants
- `set_param()` → auto-dirty node + descendants
- Cook skips clean nodes

### Port Type Checking
- Declared ports: type must match
- Undeclared input ports: dynamic, any type accepted
- Cycle detection via DFS reachability before edge insertion

### LCG Scatter RNG
```
a = 6364136223846793005
c = 1442695040888963407
seed ^= 0xdeadbeef
x_{n+1} = a * x_n + c
```

### Bilinear Height Sample
```
u = (x + half_world) / world_size * (res-1)
v = (z + half_world) / world_size * (res-1)
h = h00*(1-fu)*(1-fv) + h10*fu*(1-fv) + h01*(1-fu)*fv + h11*fu*fv
```

---

# FORGE

## What It Is

Forge is a **procedural world generation toolkit**: 15 domain-specific crates, each producing a `ForgeVolume` — a unified container for geometry or field data, spectral reflectance, physical properties, and LOD groups. The CLI (`aetherspectra-forge <command> '<json>'`) is the primary interface. Forge creates geometry and simulation data, not rendering or engine infrastructure.

---

## Workspace Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| noise | 0.9 | Perlin/Voronoi noise |
| rand | 0.8, rand_pcg 0.3 | RNG (Pcg64 for determinism) |
| glam | 0.27 | Vec3, Mat4, Quat |
| thiserror | 1 | Error handling |
| rayon | 1 | Parallelism (optional) |
| serde | 1, serde_json 1 | Serialization |
| json-patch | 2 | JSON patching |
| criterion | 0.5 | Benchmarks |

---

## forge-spectral

**Dependencies**: serde only

### Constants
```rust
pub const WAVELENGTH_COUNT:    usize = 16
pub const WAVELENGTH_START_NM: f32   = 380.0
pub const WAVELENGTH_STEP_NM:  f32   = 25.0
pub const WAVELENGTHS: [f32; 16] = [380, 405, 430, 455, 480, 505, 530, 555, 580, 605, 630, 655, 680, 705, 730, 755]
```

### SpectralRefl

```rust
pub struct SpectralRefl {
    pub samples: [f32; 16],   // reflectance in [0, 1] per wavelength
}
// impl:
// new(samples: [f32; 16]) -> Self
// flat(value: f32) -> Self
// at_wavelength(wavelength_nm: f32) -> f32  (linear interpolation, clamped at boundaries)
// scaled(self, factor: f32) -> Self         (clamp to [0,1])
```

### MaterialKind

```rust
pub enum MaterialKind {
    Soil, Rock, Bark, Water, Glass, Concrete, Foliage,
    Snow, Asphalt, Gravel, Brick, Metal, Sand
}
```

### lookup(kind: MaterialKind) -> SpectralRefl

Exact curves (16 values, 380–755 nm, USGS spectral library):

```
Foliage:  [0.04,0.04,0.05,0.07,0.08,0.10,0.12,0.12, 0.08,0.05,0.04,0.04,0.05,0.20,0.45,0.55]
Soil:     [0.07,0.09,0.11,0.13,0.14,0.16,0.18,0.20, 0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.30]
Rock:     [0.15,0.17,0.19,0.21,0.22,0.23,0.24,0.25, 0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33]
Water:    [0.03,0.04,0.05,0.05,0.05,0.04,0.03,0.03, 0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01]
Glass:    [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04, 0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]
Concrete: [0.18,0.20,0.22,0.25,0.28,0.30,0.32,0.33, 0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41]
Snow:     [0.93,0.94,0.95,0.95,0.95,0.94,0.93,0.92, 0.91,0.90,0.89,0.88,0.87,0.86,0.85,0.85]
Asphalt:  [0.04,0.04,0.05,0.05,0.06,0.06,0.07,0.07, 0.07,0.08,0.08,0.08,0.09,0.09,0.09,0.10]
Gravel:   [0.12,0.13,0.15,0.17,0.18,0.20,0.21,0.22, 0.23,0.24,0.25,0.25,0.26,0.27,0.27,0.28]
Brick:    [0.05,0.05,0.06,0.07,0.07,0.08,0.09,0.11, 0.20,0.35,0.42,0.43,0.43,0.43,0.43,0.43]
Metal:    [0.60,0.62,0.63,0.64,0.65,0.65,0.66,0.66, 0.67,0.67,0.68,0.68,0.69,0.69,0.70,0.70]
Sand:     [0.25,0.28,0.31,0.34,0.36,0.38,0.39,0.40, 0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48]
Bark:     [0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12, 0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20]
```

### rgb_to_spectral(r: f32, g: f32, b: f32) -> SpectralRefl

**Smits 1999 RGB→spectral upsampling.**

Basis spectra (16 bands each):
```
SMITS_WHITE:   [0.97; 16]
SMITS_CYAN:    [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.10,0.02,0.01,0.01,0.01,0.01]
SMITS_MAGENTA: [0.97,0.97,0.97,0.97,0.97,0.10,0.02,0.01,0.02,0.10,0.97,0.97,0.97,0.97,0.97,0.97]
SMITS_YELLOW:  [0.01,0.01,0.01,0.01,0.03,0.30,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]
SMITS_RED:     [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.05,0.20,0.80,0.97,0.97,0.97,0.97,0.97]
SMITS_GREEN:   [0.01,0.01,0.01,0.02,0.10,0.70,0.97,0.97,0.70,0.10,0.02,0.01,0.01,0.01,0.01,0.01]
SMITS_BLUE:    [0.97,0.97,0.97,0.97,0.97,0.20,0.03,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
```

Algorithm: decompose sRGB into basis weights, output `Σ weight[i] * basis[i][λ]`.

---

## forge-mesh

**Dependencies**: glam, thiserror, optional serde, optional rayon

**Coordinate system**: Y-up, right-handed. X=east, Y=up, Z=south. Clockwise winding from outside.

### Mesh

```rust
pub struct Mesh {
    pub positions:          Vec<[f32; 3]>,
    pub normals:            Vec<[f32; 3]>,
    pub tangents:           Vec<[f32; 3]>,
    pub uvs:                Vec<[f32; 2]>,
    pub indices:            Vec<[u32; 3]>,       // CW triangles
    pub material_ids:       Vec<u8>,             // per-triangle
    pub crease_indices:     Vec<u32>,
    pub crease_lengths:     Vec<i32>,
    pub crease_sharpnesses: Vec<f32>,
}
// impl: new(), triangle_count(), vertex_count(), aabb() -> ([f32;3],[f32;3])
```

### SubdivisionCage

```rust
pub struct SubdivisionCage {
    pub positions: Vec<[f32; 3]>,
    pub quads:     Vec<[u32; 4]>,    // CCW winding from outside
    pub uvs:       Vec<[f32; 2]>,
}
```

### ForgeError

```rust
pub enum ForgeError {
    InvalidParam(String),
    MeshError(String),
}
```

### Functions

**compute_normals(mesh: &mut Mesh)**
Accumulate face normals → normalize per vertex. Fallback `[0,1,0]` if length < 1e-8.

**compute_tangents(mesh: &mut Mesh)**
Mikktspace-style tangent from UV derivatives: `dp1*dv2 - dp2*dv1`, orthogonalized vs normal.

**merge(meshes: &[&Mesh]) -> Mesh**
Concatenate with index re-offset. Checks u32::MAX overflow.

**transform(mesh: &mut Mesh, matrix: &[[f32;4];4])**
Positions via `mat.transform_point3()`, normals via `inverse().transpose()`.

**flip_normals(mesh: &mut Mesh)**
Negate normals + swap triangle index 0↔2.

**apply_box_projection(mesh: &mut Mesh, texel_density: f32)**
Triplanar UV: Y-dominant → XZ, X-dominant → ZY, Z-dominant → XY.

**weld_vertices(mesh: &mut Mesh, threshold: f32)**
O(n²) pairwise distance check, union-find merge, remove degenerate triangles.

**subdivide(mesh: &Mesh, level: u8) -> Mesh**
Loop subdivision: edge midpoints + Loop weighting (`beta = 3/(8k)` for k>3, else `3/16`). 4 sub-triangles per original.

**decimate(mesh: &Mesh, ratio: f32) -> Mesh**
Quadric error metrics (Lindstrom-Turk): plane quadrics per triangle accumulated to vertices, edges sorted by collapse cost, union-find collapse loop. Fallback single triangle on full collapse.

**boolean_union(a: &Mesh, b: &Mesh) -> Result<Mesh, ForgeError>**
Approximate CSG: filter a-faces outside b + b-faces outside a, merge.

**boolean_subtract(a: &Mesh, b: &Mesh) -> Result<Mesh, ForgeError>**
Approximate CSG: filter a-faces outside b + b-faces inside a (flipped), merge.

*Helper*: `point_in_mesh(point: Vec3, mesh: &Mesh) -> bool` via Möller–Trumbore ray-triangle intersection (epsilon 1e-7).

---

## forge-volume

**Dependencies**: forge-mesh (with serde), forge-spectral, serde

### ForgeVolume

```rust
pub struct ForgeVolume {
    pub label:        String,
    pub spatial:      SpatialRepr,
    pub spectral:     Option<SpectralAttribs>,
    pub physical:     PhysicalAttribs,
    pub lod_clusters: Option<LodClusters>,
}
// Builder: new(label, spatial), with_spectral(), with_physical(), with_lod_clusters()
```

### SpatialRepr

```rust
pub enum SpatialRepr {
    Heightfield(HeightfieldSpatial),
    Mesh(Mesh),
    Sdf(SdfSpatial),
    DensityField(DensityFieldSpatial),
    ParticipatingMedia(ParticipatingMediaSpatial),
    Instances(InstancesSpatial),
}
```

### HeightfieldSpatial

```rust
pub struct HeightfieldSpatial {
    pub resolution:       u32,
    pub world_size:       f32,
    pub heights:          Vec<f32>,          // row-major, Z fastest: cell[x,z] = heights[z*res+x]
    pub material_ids:     Vec<u32>,
    pub mesh:             Option<Mesh>,
    pub sdf:              Option<SdfSpatial>,
    // Primvars:
    pub flow_map:         Vec<f32>,          // MFD flow accumulation, sqrt-normalized [0,1]
    pub deposition_map:   Vec<f32>,          // sediment deposition [0,1]
    pub curvature_map:    Vec<f32>,          // discrete Laplacian curvature
    pub disturbed_earth:  Vec<f32>,          // erosion intensity [0,1]
    pub aeolian_map:      Vec<f32>,          // wind erosion [0,1]
    pub biome_ids:        Vec<u8>,           // per-cell biome palette index
    pub splat_weights:    Vec<[f32; 4]>,     // 4-layer material blend per cell
}
// new(resolution, world_size) -> Result<Self, ForgeError>
```

### SdfSpatial

```rust
pub struct SdfSpatial {
    pub resolution:  [u32; 3],
    pub bounds_min:  [f32; 3],
    pub bounds_max:  [f32; 3],
    pub values:      Vec<f32>,   // signed distances (m). index = x + res[0]*(y + res[1]*z)
}
// is_valid(): values.len() == product(resolution)
```

### DensityFieldSpatial

```rust
pub struct DensityFieldSpatial {
    pub resolution:  [u32; 3],
    pub bounds_min:  [f32; 3],
    pub bounds_max:  [f32; 3],
    pub density:     Vec<f32>,   // [0,1]. index = x + res[0]*(y + res[1]*z)
}
// is_valid(): density.len() == product(resolution)
```

### ParticipatingMediaSpatial

```rust
pub struct ParticipatingMediaSpatial {
    pub surface_mesh:  Option<Mesh>,
    pub absorption:    SpectralRefl,    // extinction coefficients m⁻¹ (NOT reflectance — do not call scaled())
    pub scattering:    SpectralRefl,    // extinction coefficients m⁻¹
    pub anisotropy:    f32,             // Henyey-Greenstein g ∈ (-1, 1)
    pub density_scale: f32,
}
// is_valid(): anisotropy in (-1,1), density_scale finite >= 0
```

### InstancesSpatial

```rust
pub struct InstancesSpatial {
    pub positions:    Vec<[f32; 3]>,
    pub orientations: Vec<[f32; 4]>,    // XYZW quaternions (Hamilton, same as glam::Quat)
    pub scales:       Vec<f32>,
    pub asset_ids:    Vec<String>,       // resolved at assemble time
}
// count(), is_valid()
```

### SpectralAttribs

```rust
pub struct SpectralAttribs {
    pub reflectance: Vec<(u32, SpectralRefl)>,   // (material_id, curve)
}
// uniform(refl) -> Self
// lookup(material_id) -> SpectralRefl  (fallback: exact → ID 0 → first → grey 0.5)
```

### PhysicalAttribs

```rust
pub struct PhysicalAttribs {
    pub roughness: f32,          // [0,1], default 0.5
    pub ior:       Option<f32>,  // None = opaque
    pub density:   Option<f32>,  // for participating media
}
// Default: roughness=0.5, ior=None, density=None
```

### LodClusters / LodCluster

```rust
pub struct LodClusters {
    pub clusters: Vec<LodCluster>,   // sorted ascending by lod_level
}
pub struct LodCluster {
    pub lod_level:        u8,
    pub triangle_start:   u32,
    pub triangle_count:   u32,
    pub screen_threshold: f32,
}
// LodClusters::is_valid(): sorted + screen_thresholds in [0,1]
```

---

## forge-terrain

**Dependencies**: noise, rand, rand_pcg, glam, thiserror, optional rayon, serde, image

### TerrainParams

```rust
pub struct TerrainParams {
    pub resolution: u32,   // default 512, valid [16, 4096]
    pub world_size: f32,   // default 1000.0
    pub amplitude:  f32,   // default 40.0
    pub octaves:    u8,    // default 6, valid [1, 12]
    pub frequency:  f32,   // default 1.0
    pub seed:       u64,
    pub erosion:    Option<ErosionParams>,
    pub biomes:     bool,
}
```

### ErosionParams

```rust
pub struct ErosionParams {
    pub iterations:      u32,   // default 50,000
    pub erosion_rate:    f32,   // default 0.3
    pub deposition_rate: f32,   // default 0.3
    pub evaporation:     f32,   // default 0.02
    pub min_slope:       f32,   // default 0.01
    pub seed:            u64,
}
```

### TerrainGrid (output)

```rust
pub struct TerrainGrid {
    pub heights:        Vec<f32>,
    pub normals:        Vec<[f32; 3]>,
    pub biomes:         Option<Vec<Biome>>,
    pub noise_values:   Option<Vec<f32>>,
    pub resolution:     u32,
    pub world_size:     f32,
    pub cell_size:      f32,
    // Primvars (all Vec<f32> unless noted):
    pub flow_map:       Vec<f32>,       // MFD flow accumulation, sqrt-normalized
    pub deposition_map: Vec<f32>,       // sediment deposition
    pub curvature_map:  Vec<f32>,       // discrete Laplacian
    pub disturbed_earth: Vec<f32>,      // erosion disturbance
    pub traffic_wear:   Vec<f32>,       // cumulative wear
    pub aeolian_map:    Vec<f32>,       // wind erosion
    pub biome_ids:      Vec<u8>,
    pub splat_weights:  Vec<[f32; 4]>,
}
// flat_grid(resolution, world_size) -> Self
// height_at(x: f32, z: f32) -> f32           (bilinear)
// normal_at(x: f32, z: f32) -> [f32; 3]      (nearest-neighbour)
// compute_normals(&mut self)
```

### Biome enum

Variants: `Alpine, Tundra, Forest, Grassland, Desert, Wetland, Coastal, SubalpineShrub, Savanna, Taiga, TropicalRainforest, ...`

### generate(params: TerrainParams) -> Result<TerrainGrid, ForgeError>

**Algorithm**:
1. Validate params
2. `Fbm<Perlin>` with seed, octaves, persistence=0.5, lacunarity=2.0
3. High-freq noise at 8× base frequency (seed + 0xDEAD_BEEF)
4. Height: `(fbm.get([nx, nz]) + 1.0) * 0.5 * amplitude`
5. `compute_normals()`
6. If erosion: `erode(grid, params)`
7. If biomes: `classify_biomes()`, `compute_splat_weights()`

---

## forge-building

**Dependencies**: forge-mesh, glam, rand, rand_pcg, thiserror, serde, serde_json, json-patch

### BuildingParams

```rust
pub struct BuildingParams {
    pub floors:             u8,
    pub floor_height:       f32,
    pub width:              f32,
    pub depth:              f32,
    pub style:              Style,
    pub footprint:          FootprintShape,
    pub roof:               Option<RoofStyle>,
    pub window_density:     f32,
    pub seed:               u64,
    pub generate_interior:  bool,
}
```

### Style enum

Variants: `Victorian, Modern, Colonial, Industrial, Gothic, Brutalist, Medieval`

### FootprintShape enum

Variants: `Rectangular, LShape, TShape, HShape, CircularBase`

### RoofStyle enum

Variants: `Flat, Gabled, Hipped, Mansard, Gambrel, Domed`

### Material IDs

- 0: Outer wall
- 1: Window glass
- 2: Window frame
- 3: Window reveal (recessed)
- 4+: Roof, doors, ornaments

### generate(params: BuildingParams) -> Result<Mesh, ForgeError>

**Algorithm**:
1. Validate: floors≥1, width≥2, depth≥2, floor_height≥1.5
2. Style defaults (bay width, windows/bay, roof, cornice)
3. Wall geometry via `wall::build_wall()`
4. Cornice: outward normal = 90° CW rotation of edge dir in XZ
5. Roof via `roof::build_roof()`
6. Optional interior proxy
7. Merge all → single Mesh

### WFC (Wave Function Collapse) for façade

- `TileType` enum
- `WFCParams`, `WFCSolution` structs
- `solve_wfc(params: &WFCParams) -> Result<WFCSolution, ForgeError>`

### Other building functions

- `build_catalog() -> PartCatalog`
- `wall_edges(shape: FootprintShape, width, depth, height) -> Vec<Edge>`

---

## forge-water

**Dependencies**: forge-mesh, glam, thiserror, serde

### OceanParams

```rust
pub struct OceanParams {
    pub resolution: u32,
    pub world_size: f32,
    pub wave_height: f32,
    pub wind_speed:  f32,
    pub wind_dir:    f32,
    pub chop:        f32,
    pub time:        f32,
    pub seed:        u64,
}
```

### RiverParams

```rust
pub struct RiverParams {
    pub waypoints: Vec<[f32; 3]>,
    // width, animation params, etc.
}
```

### generate_ocean(params: OceanParams) -> Result<Mesh, ForgeError>

**Algorithm**: Tessendorf FFT-based wave simulation using JONSWAP spectrum.

### generate_river(params: RiverParams) -> Result<Mesh, ForgeError>

**Algorithm**: Catmull-Rom spline + ribbon cross-section.

---

## forge-vegetation

**Dependencies**: forge-mesh, glam, rand, rand_pcg, thiserror, serde

### Species enum

Variants: `OakTree, PineTree, SpruceTree, BirchTree, Shrub, TallGrass, Fern, Cactus`

### VegetationParams

```rust
pub struct VegetationParams {
    pub species:    Species,
    pub height:     f32,
    pub age:        f32,      // [0, 1]
    pub wind_lean:  f32,
    pub seed:       u64,
}
```

### generate(params: VegetationParams) -> Result<Mesh, ForgeError>

L-system with branching + leaf generation.

### generate_lod_set(params: VegetationParams) -> Result<[Mesh; 4], ForgeError>

- LOD0: Full branches + leaves
- LOD1: Coarser branches
- LOD2: Simplified geometry
- LOD3: 2-triangle billboard

---

## forge-scatter

**Dependencies**: forge-mesh, forge-terrain, glam, rand, rand_pcg, thiserror, serde

### ScatterParams

```rust
pub struct ScatterParams {
    pub density:         f32,
    pub seed:            u64,
    pub exclude_angles:  Option<(f32, f32)>,
    pub asset_ids:       Vec<u32>,
    pub asset_weights:   Vec<f32>,
    pub scale_min:       f32,
    pub scale_max:       f32,
    pub random_rotation: bool,
}
```

### scatter(grid: &TerrainGrid, params: ScatterParams) -> Result<ScatterInstances, ForgeError>

Poisson disk sampling on heightfield with slope rejection.

### scatter_on_mesh(mesh: &Mesh, params: ScatterParams) -> Result<ScatterInstances, ForgeError>

Poisson disk sampling on triangle mesh with normal orientation rejection.

---

## forge-road

**Dependencies**: forge-mesh, glam, thiserror, serde

### RoadParams

```rust
pub struct RoadParams {
    pub control_points: Vec<[f32; 3]>,
    pub width:          f32,
    pub camber:         f32,
    pub segments:       u32,
}
```

### NetworkParams — road network from intersection graph

### generate(params: RoadParams) -> Result<Mesh, ForgeError>

Catmull-Rom spline → uniform arc-length sampling → ribbon mesh with camber.

---

## forge-flow-maps

**Dependencies**: glam, rand, rand_pcg, serde, thiserror

### run_drip_simulation(params: &DripParams) -> DripResult

Rain particle simulation for drainage flow maps and weathering.

---

## forge-inhabitation

**Dependencies**: glam, serde, rand, rand_pcg, thiserror

### solve_catenary(params: &CatenaryParams) -> CatenaryResult

Physics-based hanging cable / catenary curve.

### run_prop_placement(params: &PlacementParams) -> Result<Vec<PropPlacement>, ForgeError>

Prop placement with collision/spacing constraints.

---

## forge-urban-sim

**Dependencies**: glam, serde, rand, rand_pcg, thiserror

### run_urban_sim(params: &UrbanSimParams) -> UrbanVoxelGrid

Reaction-diffusion simulation: traffic, population density, wear, moisture, wind exposure.

---

## forge-plot

**Dependencies**: forge-mesh, forge-spectral, forge-volume, glam, thiserror, rand, rand_pcg, serde, serde_json

### PlotParams

```rust
pub struct PlotParams {
    pub archetype:          PlotArchetype,
    pub footprint:          [f32; 2],
    pub building_footprint: [f32; 2],
    pub seed:               u64,
}
```

### PlotArchetype enum

Variants: `ResidentialSuburban, ResidentialUrban, Commercial, ...`

### PlotAsset (output)

```rust
pub struct PlotAsset {
    pub ground_mesh:     Mesh,
    pub fence_instances: Vec<...>,
    pub driveway_mesh:   Mesh,
    pub prop_placements: Vec<...>,
    pub garden_zones:    Vec<...>,
}
```

### generate(params: &PlotParams) -> Result<PlotAsset, ForgeError>

---

## forge-neural

**Dependencies**: (minimal — reserved for future backends)

**Status**: Stubbed. Pluggable inference backends defined but not implemented:
- `LocalOnnx`: ONNX model, CPU/GPU
- `LocalCandle`: candle inference
- `ModalApi`: Modal-hosted remote generation (Trellis, TripoSR, Shap-E)
- `Mock`: Procedural fallback

---

## forge-cli (aetherspectra-forge binary)

```bash
aetherspectra-forge <command> '<json-params>'
aetherspectra-forge <command> '-'   # stdin
```

| Command | Output | Domain |
|---|---|---|
| terrain | ForgeVolume (Heightfield) | forge-terrain |
| building | ForgeVolume (Mesh) | forge-building |
| scatter | ForgeVolume (Instances) | forge-scatter |
| water | ForgeVolume (Mesh) | forge-water |
| vegetation | ForgeVolume (Mesh) | forge-vegetation |
| road | ForgeVolume (Mesh) | forge-road |
| flow_maps | JSON | forge-flow-maps |
| inhabitation | JSON | forge-inhabitation |
| urban_sim | JSON | forge-urban-sim |
| plot | JSON | forge-plot |

---

## Full Data Flow

```
JSON params
    │
    ▼
forge-<domain>::generate()
    │
    ▼
ForgeVolume { spatial, spectral, physical, lod_clusters }
    │ (JSON serialization)
    ▼
Python/conductor orchestration
    │
    ▼
CrucibleDirective JSON
    │
    ▼
aetherspectra-cook → CrucibleGraph.cook() → USD sublayers
    │
    ▼
Spectra renderer
```

---

## Important Notes

1. **ParticipatingMediaSpatial**: `absorption` and `scattering` use `SpectralRefl` struct for storage but values are **extinction coefficients (m⁻¹)**, NOT reflectances. Do NOT call `.scaled()` on these.

2. **Storage order inconsistency**: HeightfieldSpatial is row-major Z-fastest (`heights[z*res+x]`). SdfSpatial and DensityFieldSpatial are X-major X-fastest (`values[x + res[0]*(y + res[1]*z)]`). Conversion requires re-striding.

3. **Quaternion convention**: XYZW throughout (Hamilton convention, same as `glam::Quat`).

4. **Y-up right-handed**: X=east, Y=up, Z=south. CW winding from outside for forge-mesh. CCW for SubdivisionCage quads.

5. **forge-neural**: Completely stubbed. All backends return `ForgeError::InvalidParam("neural backend not implemented")`.

6. **splat-train**: Requires CUDA (`spectra-gpu`). Not usable from wgpu/CPU path without a new backend.

7. **MaterialSpec.spectral_reflectance**: Declared in crucible-types but marked "Phase 2" — the USD materials.usda writer does not yet output spectral data. It is a forward-declared type.

8. **forge-scatter ScatterInstances vs crucible-types ScatterInstances are different types**:
   - forge-scatter: `{ transforms: Vec<[[f32; 4]; 4]>, asset_id: String, count: usize }` — full 4×4 transform matrices
   - crucible-types: `{ positions: Vec<[f32;3]>, orientations: Vec<[f32;4]>, scales: Vec<f32>, asset_ids: Vec<u32>, mesh_labels, asset_paths }` — decomposed TRS

9. **TerrainGrid struct differs between forge-terrain and crucible-types**:
   - forge-terrain adds `traffic_wear: Vec<f32>` (not in crucible-types)
   - forge-terrain adds `noise_values: Option<Vec<f32>>`, `biomes: Option<Vec<Biome>>`, `cell_size: f32`, `normal_at()`, `compute_normals()`
   - crucible-types does not expose `traffic_wear`

---

## Additions — Second Pass (items missing from first pass)

### forge-building: description.rs (complete)

These types are NOT produced by `generate()` but are the descriptive/semantic layer for LLM-driven or authored building specs:

```rust
#[repr(u8)] pub enum Program { Residential=0, Agricultural=1, Civic=2, Religious=3, Commercial=4, Industrial=5, Utility=6 }
#[repr(u8)] pub enum Setting { Urban=0, Suburban=1, Rural=2, Industrial=3, Waterfront=4, HistoricalOldTown=5 }
#[repr(u8)] pub enum AssemblyChannel { Structure=0, Ornament=1, Detail=2, Organic=3 }
#[repr(u8)] pub enum HeroLevel { Background=0, Supporting=1, Hero=2 }
#[repr(u8)] pub enum SidingType { Clapboard=0, Brick=1, Stone=2, Stucco=3, Vinyl=4, TimberFrame=5 }
#[repr(u8)] pub enum WindowStyle { DoubleHung=0, Casement=1, Bay=2, Lancet=3, Sash=4 }
// FootprintShape: Rectangular, LShaped, UShaped, TShaped (adds UShaped vs base BuildingParams)
// RoofStyle: Flat, Gabled, Hip, Mansard (subset of base BuildingParams' 6 variants)

pub struct PorchSpec { pub depth: f32, pub columns: bool, pub roof_height: Option<f32> }

pub struct FootprintDesc { pub shape: FootprintShape, pub width: f32, pub depth: f32 }

pub struct RoofDesc {
    pub style: RoofStyle,
    pub overhang: f32,
    pub dormers: u8,
    pub ridge_height_factor: f32,
}

pub struct ResidentialSpec {
    pub porch: Option<PorchSpec>,
    pub chimney: bool,
    pub siding: SidingType,
    pub window_style: WindowStyle,
    pub bay_windows: bool,
    pub foundation_visible: bool,
    pub dormers: u8,
    pub attached_garage: bool,
}

pub enum BuildingSpec { Residential(ResidentialSpec) }

pub struct BuildingDescription {
    pub id: Option<String>,
    pub variation_of: Option<String>,
    pub program: Program,
    pub setting: Setting,
    pub style: StyleKey,        // String — free-form style name
    pub era: String,            // e.g. "1920s", "contemporary"
    pub condition: Condition,   // New/Aged/Weathered/Derelict
    pub seed: u64,
    pub footprint: FootprintDesc,
    pub floors: u8,
    pub floor_height: f32,
    pub roof: RoofDesc,
    pub channels: Vec<AssemblyChannel>,
    pub hero_level: HeroLevel,
    pub uv_texel_density: f32,
    pub detail_atoms: Option<Vec<String>>,   // e.g. ["brass_doorknob", "ornate_cornice"]
    pub organic_atoms: Option<Vec<String>>,  // e.g. ["ivy_creep", "weathered_brick"]
    pub spec: BuildingSpec,
}
```

**Note**: `detail_atoms` and `organic_atoms` are LLM-generated string tags that drive the Ornament and Organic assembly channels. They bridge natural-language building description to procedural geometry.

```rust
pub struct StyleDefaults {
    pub floor_height_mult: f32,
    pub bay_width: f32,
    pub windows_per_bay: u8,
    pub roof: RoofStyle,
    pub cornice_projection: f32,
    pub cornice_height: f32,
}
pub fn style_defaults(style: Style) -> StyleDefaults
```

---

### forge-terrain: biomes.rs (complete)

```rust
pub enum Biome { Alpine, Tundra, Forest, Grassland, Desert, Wetland, Coastal
                 /* + SubalpineShrub, Savanna, Taiga, TropicalRainforest from enum */ }

pub struct BiomeMap { pub cells: Vec<Biome> }

pub fn classify_biomes(grid: &TerrainGrid) -> BiomeMap
// Algorithm: classify by height + moisture:
//   height > 0.8*amplitude → Alpine; height > 0.6*amplitude → Tundra
//   flow_map > 0.7 → Wetland; near sea-level + high flow → Coastal
//   etc.

pub fn biome_to_splat_weights(biome: &Biome, height: f32, world_height: f32) -> [f32; 4]
// Returns 4-channel blend weights for terrain material splat texture.
// Channel mapping: [0]=base_ground, [1]=rock, [2]=vegetation, [3]=special
// Examples: Forest → high [2]; Desert → high [0] (sand); Alpine → high [1]+[3](snow)

pub fn compute_splat_weights(grid: &mut TerrainGrid, biome_map: &BiomeMap)
// Writes biome_to_splat_weights() into grid.splat_weights for all cells
```

---

### forge-terrain: grade.rs (complete)

```rust
pub enum GradingStrategy { LevelPad, Stepped, Pier, CutIntoSlope }
impl GradingStrategy {
    pub fn from_str(s: &str) -> Self  // "level_pad" | "stepped" | "pier" | "cut_into_slope"
}

pub fn grade_building_footprint(
    grid: &mut TerrainGrid,
    center: [f32; 2],
    size: [f32; 2],
    strategy: GradingStrategy,
    blend_radius: f32,
)
// Modifies grid.heights in-place to flatten terrain under a building footprint.
// LevelPad: sample median height in footprint, level to that, gaussian blend at edges
// Stepped: staircase at floor_height intervals
// Pier: raise land to average, no blend (pier/stilts implied)
// CutIntoSlope: excavate into uphill side, flat pad
```

---

### forge-terrain: spectral.rs

```rust
pub struct SpectralTerrainMaterials {
    pub slots: [SpectralRefl; 7],
    // Slot order: Water, Sand, Grass, Dirt, Rock, Snow, Forest
    // Default: forge-spectral USGS curves for each
}
impl Default for SpectralTerrainMaterials { fn default() -> Self }
// Maps to splat_weights[4] via: slots[0..3] = splat weight channels + extras
```

---

### forge-scatter: ScatterParams (complete fields)

Two param structures exist — the one in forge-scatter vs. the simpler one in other contexts:

```rust
// forge-scatter/src/params.rs (full):
pub struct ScatterParams {
    pub density:          f32,             // instances per 100m²
    pub min_distance:     f32,             // metres between instances
    pub slope_min:        f32,             // degrees
    pub slope_max:        f32,             // degrees
    pub altitude_min:     f32,
    pub altitude_max:     f32,
    pub align_to_normal:  bool,
    pub scale_variance:   f32,
    pub asset_id:         String,
    pub seed:             u64,
    pub exclude_angles:   Option<(f32, f32)>,  // from terrain.rs version
    pub asset_ids:        Vec<u32>,
    pub asset_weights:    Vec<f32>,
    pub scale_min:        f32,
    pub scale_max:        f32,
    pub random_rotation:  bool,
}
```

**Note**: The two agent reports describe slightly different field sets — the exact fields depend on which source file was read. The key point: ScatterParams has altitude/slope filtering for terrain placement AND asset-weight CDF for multi-asset scatter.

---

### forge-plot: PlotParams (complete fields)

```rust
pub struct PlotParams {
    pub archetype:          PlotArchetype,
    pub footprint:          [f32; 2],           // total plot [width, depth]
    pub building_footprint: [f32; 2],           // inner building pad
    pub condition:          Condition,           // New/Aged/Weathered/Derelict
    pub seed:               u64,
    pub fence_style:        String,             // "picket"|"chain_link"|"stone_wall"|"hedge"|"none"
    pub driveway_surface:   String,             // "gravel"|"concrete"|"asphalt"|"none"
}

pub struct FenceInstance { pub position: [f32; 3], pub rotation_y: f32, pub asset_query: String }
pub struct PropPlacement  { pub position: [f32; 3], pub asset_query: String, pub rotation_y: f32 }
pub struct GardenZone     { pub center: [f32; 2], pub radius: f32, pub density: f32, pub asset_query: String }
```

---

### forge-inhabitation: complete structs

```rust
// catenary.rs:
pub struct CatenaryParams { pub start: [f32; 3], pub end: [f32; 3], pub slack: f32, pub segments: u32 }
pub struct CatenaryResult { pub points: Vec<[f32; 3]> }
pub fn solve_catenary(params: &CatenaryParams) -> CatenaryResult
// Algorithm: Newton's method to find catenary parameter 'a' given span and arc_length = end_distance * (1 + slack)
// fn find_catenary_a(h: f32, arc_length: f32) -> f32

// placement.rs:
pub struct PlacementParams { pub area: [f32; 2], pub count: u32, pub min_clearance: f32, pub seed: u64 }
pub struct PropPlacement   { pub position: [f32; 3], pub asset_query: String, pub rotation_y: f32 }
pub fn run_prop_placement(params: &PlacementParams) -> Vec<PropPlacement>
// Algorithm: Poisson-disk rejection sampling within area with min_clearance spacing
```

---

### forge-urban-sim: complete structs

```rust
pub struct UrbanVoxel {
    pub traffic_weight: f32,    // cumulative traffic load [0,1]
    pub refuse_level:   f32,    // waste accumulation [0,1]
    pub civic_upkeep:   f32,    // maintenance quality [0,1]
    pub wind_exposure:  f32,    // exposed to wind [0,1]
    pub moisture:       f32,    // ambient moisture [0,1]
}
impl Default for UrbanVoxel  // all 0.5

pub struct UrbanVoxelGrid { pub cells: Vec<UrbanVoxel>, pub width: u32, pub height: u32 }

pub struct UrbanSimParams {
    pub grid_w:     u32,    // default 64
    pub grid_h:     u32,    // default 64
    pub iterations: u32,    // default 100
    pub seed:       u64,
}

pub fn run_urban_sim(params: &UrbanSimParams) -> UrbanVoxelGrid
// Algorithm: reaction-diffusion over grid cells; traffic diffuses from seed points,
// moisture from flow, wind from gradient, civic_upkeep inversely proportional to traffic
```

**UrbanVoxel fields map directly to spectral material modulation**:
- `traffic_weight` → blend toward Asphalt/Concrete spectral curves
- `moisture` → blend toward Water spectral curve (wet materials darker, less reflective)
- `refuse_level` → blend toward Soil/Dirt
- `wind_exposure` → drives aeolian weathering (sand-blasted materials)
- `civic_upkeep` → Condition: high = New, low = Derelict

---

### forge-flow-maps: complete structs

```rust
pub struct DripParams {
    pub particle_count: u32,   // default 10,000
    pub max_steps:      u32,   // default 500
    pub seed:           u64,
}
pub struct DripResult {
    pub drip_intensity: Vec<f32>,   // per-cell rain erosion intensity [0,1]
    pub resolution:     u32,
}
pub fn run_drip_simulation(
    heights: &[f32],
    normals: &[[f32; 3]],
    resolution: u32,
    params: &DripParams,
) -> DripResult
// Algorithm: particle drip — spawn rain particles at random, follow steepest descent,
// accumulate hit-count per cell, normalize to [0,1]
```

**DripResult.drip_intensity** is a flow/wetness map: high values = drainage channels, low = ridges. Can drive:
- Puddle placement (high intensity + flat curvature)
- Wet material blending (Water spectral curve)
- Moss/algae growth zones (high moisture + low slope)

---

### crucible-nodes: FloraPrimeNode (complete)

```rust
// SpeciesProfile (internal):
struct LeafParams {
    n: f32,      // refractive index
    cab: f32,    // chlorophyll a+b µg/cm²
    car: f32,    // carotenoid µg/cm²
    cbrown: f32, // brown pigment
    cw: f32,     // water cm
    cm: f32,     // dry matter g/cm²
    anth: f32,   // anthocyanin µmol/cm²
}

struct SpeciesProfile {
    branch_angle:        f32,
    trunk_taper:         f32,
    branch_length_ratio: f32,
    leaf_params:         LeafParams,
    species_class:       SpeciesClass,  // Conifer, Columnar, Spreading, Weeping
}

fn compute_spectral_embedding(leaf: &LeafParams, class: SpeciesClass) -> [f32; 6]
// Evaluates PROSPECT-PRO radiative transfer model at 6 wavelengths
// Output: per-vertex [f32; 6] packed into Mesh.spectral_embedding
// This is a PCA projection, NOT full 16-band — 6 principal components explain ~97% of variance

// Species name → SpeciesClass:
// spruce/fir/pine → Conifer
// cypress/poplar → Columnar
// oak → Spreading
// birch/beech, ash, willow → Weeping

impl FloraPrimeNode (params: species, species_id, crown_radius, n_nodes, seed)
// cook() generates TreeQSM (Quantitative Structure Model):
//   - Recursive branching with species angle + taper
//   - qsm_to_cylinder_mesh() converts QSM to cylinder mesh
//   - Spectral embedding computed per vertex from LeafParams
```

---

### crucible-usd: MaterialX output

```rust
// materialx.rs:
pub fn write_standard_surface(out: &mut String, spec: &MaterialSpec, mat_index: usize)
// Writes MaterialX StandardSurface node for each MaterialSpec:
//   base_color, roughness, metallic, emission, subsurface params, coat, anisotropy, sheen
// Note: spectral_reflectance is NOT written to MaterialX — it is forward-declared Phase 2
```

---

### forge-water: RiverParams (complete)

```rust
pub struct RiverParams {
    pub waypoints:    Vec<[f32; 3]>,
    pub width:        f32,
    pub depth_visual: f32,
    pub flow_speed:   f32,
    pub time:         f32,
}
```

---

### forge-road: RoadParams + NetworkParams (complete)

```rust
pub struct RoadParams {
    pub control_points: Vec<[f32; 3]>,
    pub width:          f32,
    pub segments:       u32,
    pub camber:         f32,
    pub curbs:          bool,   // generate kerb geometry (future use)
}
pub struct NetworkParams {
    pub roads:         Vec<RoadParams>,
    pub intersections: bool,    // generate intersection geometry
}
```

---

### director binary (aetherspectra-director): full pipeline

```
User NL prompt
    │
    ▼ Gemini API call
CrucibleDirective JSON
    │
    ▼ invoke aetherspectra-cook
CrucibleGraph.cook() → USD sublayers
    │ (if native-splat feature enabled)
    ▼ splat-train (CUDA)
3DGS .ply output
    │
    ▼ Spectra renderer
```

**Features**:
- `native-splat`: enables 3DGS training path (requires CUDA / spectra-gpu)
- Without `native-splat`: produces USD scene only, rendering via Spectra USD loader

**Stage machine pattern**: director uses a stage-machine (not state machine — "stage" = production stage) to sequence: brief → direct → cook → train → render.

---

### crucible-types: Volume (complete, not in original reference)

```rust
pub enum SpatialRepr {
    Terrain(TerrainGrid),
    Mesh(Mesh),
    Meshes(Vec<Mesh>),
    // NOTE: crucible-types SpatialRepr is simpler than forge-volume SpatialRepr
    // forge-volume adds: Sdf, DensityField, ParticipatingMedia, Instances, Heightfield
}

pub struct Volume {
    pub label:        String,
    pub spatial:      SpatialRepr,
    pub lod_clusters: Option<LodClusters>,
}
impl Volume {
    pub fn terrain(label: impl Into<String>, grid: TerrainGrid) -> Self
    pub fn mesh(label: impl Into<String>, mesh: Mesh) -> Self
    pub fn meshes(label: impl Into<String>, meshes: Vec<Mesh>) -> Self
}

// NOTE: crucible-types Volume does NOT have spectral or physical attribs
// Those live in forge-volume ForgeVolume only
// The forge→crucible pipeline strips spectral attribs to crucible MaterialSpec
```

---

## Corrections to First Pass

| Item | First Pass | Correction |
|---|---|---|
| forge-water algorithm | "Tessendorf FFT (JONSWAP)" | ✓ Correct |
| forge-neural | "stubbed" | ✓ Correct |
| splat-train | "requires CUDA" | ✓ Correct |
| forge-building Style enum | "Victorian, Modern, Colonial, Industrial, Gothic, Brutalist, Medieval" | Also: Tudor=7, Mediterranean=8, Craftsman=9 |
| forge-building FootprintShape | "Rectangular, LShape, TShape, HShape, CircularBase" | description.rs has: Rectangular, LShaped, UShaped, TShaped (only 4 variants in description.rs vs more in BuildingParams) |
| forge-building RoofStyle | "Flat, Gabled, Hipped, Mansard, Gambrel, Domed" | description.rs subset: Flat, Gabled, Hip, Mansard only |
| forge-scatter ScatterInstances | crucible-types version | Two distinct types — see note #8 above |
| TerrainGrid | same in forge and crucible | Different — forge adds traffic_wear, noise_values, biomes field, methods |
| forge-terrain Biome variants | partial list | Full: Alpine, Tundra, Forest, Grassland, Desert, Wetland, Coastal + SubalpineShrub, Savanna, Taiga, TropicalRainforest |
