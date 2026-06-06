# Design: USD Scene Import — `vox_usd` (2026-06-06)

**Status:** Draft
**Scope:** A new game-agnostic `vox_usd` engine crate that reads composed `.usdc`/`.usd` scenes via the extracted `openusd-rs` sibling repo — meshes→2DGS splats, PointInstancer→3DGS splats, lights, camera, entities — wired into the content browser and a `vox_tools usd-import` CLI. Read-only slice; export/round-trip deferred.
**Related:** [Competitive research](./2026-06-06-engine-competitive-research.json) (USD/MaterialX interchange), [Atom-budget renderer](./2026-06-06-atom-budget-splat-renderer-design.md)

> Produced by a 3-design judge-panel workflow (import-first / stage-centric / tool-bridge mandates); import-first won with grafts. Every openusd-rs capability claim below was verified against its source (file:line cited in IMPORTANT NOTES) — the stage-centric session-layer approach was explicitly rejected for this slice because openusd-rs has no mutable authoring API (see Out of Scope).

---

## Problem Statement

- Ochroma can WRITE USD (crucible-usd) and LOAD native splat formats (vxm/spz/ply/gltf), but cannot READ a USD scene into the engine — there is no path from a Blender/DCC `.usd*` to entities + splats + lights + camera.
- `openusd-rs` (sibling repo, never modified here) already provides a real, composed read API (`pcp` composition + typed binary-crate arrays) and is entirely unused by Ochroma today.
- Strategic window: OpenUSD is moving Gaussian splats toward a first-class Particle-Fields / PointInstancer schema direction, and no major engine reads splats from USD natively — a working importer positions Ochroma first, with the `PointInstancer`/`PointBased` path as the natural landing spot for the forthcoming splat schema.
- Constraint reality: the `openusd-rs` USDA *text* parser is array/tuple-blind, so import must lean on binary `.usdc` (Blender's default export) and be explicit about the `.usda` limitation rather than silently importing empty geometry.
- VXM stays the runtime splat payload; USD is the scene/interchange layer only. Round-trip *export* of edited scenes back to USD is explicitly out of scope for this slice (the writers cannot author bulk float/vec arrays — see IMPORTANT NOTES).

## Done When

A real Blender-exported scene is committed at `crates/vox_usd/tests/data/cube_lit.usdc` — one textured 2 m cube `Mesh` (~24 faces triangulated, ~24 m² surface), one `SphereLight` (intensity 1000, white), one `Camera` (focalLength 50, horizontalAperture 36, world pos (0,1,6)) — authored via Blender 4.x "Export USD" → `.usdc`.

Command:

```
cargo run -p vox_tools -- usd-import crates/vox_usd/tests/data/cube_lit.usdc
```

Exact human-visible stdout (values match the committed fixture; cube sampled at 200 splats/m²):

```
[usd] opened stage: cube_lit.usdc (upAxis=Y, metersPerUnit=1)
[usd] /Cube    Mesh         -> 4812 splats
[usd] /Light   SphereLight  color=(1.00,1.00,1.00) intensity=1000
[usd] /Camera  Camera       fovY=39.6deg pos=(0.0,1.0,6.0)
[usd] import OK: 1 mesh, 1 light, 1 camera, 4812 splats, 0 warnings
```

AND:

```
cargo test -p vox_usd
```

prints `test result: ok.` with every capability test below passing.

AND the engine-isolation gate holds — `cargo tree -p vox_core -p vox_render -p vox_app | grep openusd` returns empty (no engine GPU/app crate takes the openusd-rs dep; only `vox_usd` and the `vox_tools` CLI do).

## Capabilities

| Capability | Real-behavior test (asserts a computed/composed outcome) | Forbidden stub (must NOT be the test) |
|---|---|---|
| Open + compose `.usdc` | `import_usd("cube_lit.usdc")` → `stats.prims >= 3` AND `stats.meshes == 1` | `assert!(result.is_ok())` |
| Read `point3f[]` from binary crate | mesh splats' bbox min/max equal authored cube corners within 1e-4: `assert_eq!(round(min), [-1.,-1.,-1.])` | `assert!(!splats.is_empty())` |
| Apply composed Xform | prim authored `xformOp:translate=(5,0,0)` → splats' mean x ≈ 5.0 ± 0.05 | `assert!(matrix.is_some())` |
| Mesh → 2DGS sampling | splat count == sampler formula `Σ clamp(ceil(area·spm),1,50)` for fixture (exact integer, e.g. 4812); every splat `.is_surface()` | `assert!(count > 0)` |
| PointInstancer → 3DGS | fixture with 3 instances at known positions → exactly 3 splats, each `.is_volume()`, positions == authored `positions` within 1e-4 | `assert!(splats.len() >= 1)` |
| color3f → spectrum | red material `diffuseColor=(1,0,0)` → splat `.spectral_f32(b)` == `f16::from_f32(SpectralUpsampler::from_rgb(1.,0.,0.)[b]).to_f32()` for all 16 b | `assert!(spectral[0] != 0)` |
| Light read | `SphereLight intensity=1000 color=(1,1,1)` → `lights[0].intensity == 1000.0 && lights[0].color == [1.,1.,1.]` | `assert!(!lights.is_empty())` |
| Camera read (schemaless) | `focalLength=50 horizontalAperture=36` → `camera.fov_y_deg` within 0.1 of `2·atan(18/50)` deg; world pos == (0,1,6) ± 1e-4 | `assert!(camera.is_some())` |
| USDA array limitation surfaced | `import_usd("points_text.usda")` whose only geometry is `point3f[] points=[…]` → `Err(UsdError::UnsupportedTextArray)` AND no silent empty-Ok | silently returning `Ok` with empty splats and no warning |
| Editor wiring | `content_browser::load_asset(p, AssetKind::Usd)` → `LoadedAsset::Splats(v)` with `v.len()` == direct `import_usd(p).splats.len()` | `matches!(res, Ok(LoadedAsset::Splats(_)))` or "the enum arm exists" |
| Engine crates stay USD-free | `cargo tree -p vox_core -p vox_render -p vox_app \| grep openusd` is empty | n/a (CI grep gate) |

## Architecture

### `vox_usd` crate (ENGINE layer, game-agnostic)

New crate at `crates/vox_usd`, sibling to `vox_data`. Contains zero game concepts (no buildings/zoning/traffic). Depends on `openusd-rs` (sibling path dep, exact form crucible-usd already uses: `openusd-rs = { path = "../../../openusd-rs" }`), `vox_core` (for `GaussianSplat`), `vox_data` (for `SpectralUpsampler::from_rgb`, game-agnostic engine crate — no game-layer violation; verified there is no `vox_data → vox_usd` back-edge), and `glam`. The crate is the *only* engine crate that takes the openusd-rs dep; `vox_core`/`vox_render`/`vox_app` never do (enforced by the `cargo tree` gate).

No openusd-rs source is modified. Every capability maps onto existing `pub fn`s (verified — see IMPORTANT NOTES).

### Stage open + composition (delegated to openusd-rs)

`import_usd` opens `usd::Stage::open(path)`, which auto-detects `.usda` (text) vs `.usdc`/`.usd` (binary crate) by extension and runs full PCP composition (sublayers, references, payloads, inherits, variants, relocates, listOps, dictionary merge, active-based pruning). No new composition logic is written.

Stage metadata `upAxis` and `metersPerUnit` are read via `pseudo_root().metadata::<tf::Token>(...)` / `::<f64>(...)` to build a `root_correction: DMat4` (USD default Y-up; Blender often exports Z-up — correction is on by default, configurable). A known post-correction camera position is asserted in tests to catch a wrong correction matrix.

### Traversal + transform accumulation

Recursive DFS over `prim.children()`, accumulating world transform as `parent_world * local`, where `local` comes from `XformOp::get_local_matrix(&prim)` (fully implemented: translate/scale/rotate*/orient/transform + inverse ops + all rotation orders). `gf::Matrix4d` is row-major; transpose into `glam::DMat4`. Invalid prims (`!prim.is_valid()`, i.e. not `Def`/`Class`) are skipped but their subtree is not.

Per-prim dispatch on `prim.type_name().as_str()`:

- **"Mesh"** → read `points` (`vt::Array<gf::Vec3f>`), `faceVertexIndices`/`faceVertexCounts` (`vt::Array<i32>`); triangulate via `usd_geom::triangulate(&mesh)` (naive fan — fine for Blender quads/tris); transform to world; barycentric-sample to 2DGS surface splats at `settings.mesh_splats_per_sqm` (default 200), `clamp(ceil(area·spm), 1, 50)` per triangle (mirrors `gltf_import`'s sampler). Color from bound material `inputs:diffuseColor`, else default grey.
- **"PointInstancer"** → read `positions`/`scales`/`orientations`/`protoIndices`; emit one 3DGS volume splat per instance. This is the splat-native path and the Particle-Fields landing spot.
- **"Points" / PointBased with `widths`** → 3DGS volume splats directly from `points` + `widths`. (No `UsdGeomPoints` schema exists — read via `type_name()` + raw attributes; `PointBased::points_attr` is available.)
- **"SphereLight"/"RectLight"/"DiskLight"/"DistantLight"/"DomeLight"** → `UsdLight` = `color × intensity × 2^exposure`. `DistantLight` has no schema; read by type-name + raw `inputs:intensity`/`inputs:color`/`inputs:exposure` attribute lookups (schema-agnostic, total via `try_get`).
- **"Camera"** → `UsdCamera`. No `UsdGeomCamera` schema exists, so read generically via raw `Attribute` lookups on `focalLength`/`horizontalAperture`/`focusDistance`; `fov_y = 2·atan(verticalAperture / (2·focalLength))`; world pos/rotation from accumulated transform.
- **"Xform"/"Scope"/default** → recurse only; emit a `UsdEntity` (named node for the outliner).

### Color → spectrum

Diffuse RGB from `MaterialBindingAPI::bound_material(&stage)` → `surface_shader` → `get_input::<gf::Vec3f>("diffuseColor")` (connection-following exists), default grey `(0.5,0.5,0.5)` when unbound. Each of 16 bands: `spectral[b] = f16::from_f32(SpectralUpsampler::from_rgb(r,g,b)[b]).to_bits()` (Smits 1999 — same path spz/ply/gltf already use).

### Safety: never call `Attribute::get`

`Attribute::get<T>()` (attribute.rs:17) `.unwrap()`s and panics on type mismatch/absence. The crate calls **only** `try_get::<T>()` / `get_value()` everywhere — both are total. This is a hard rule, enforced by a clippy-style review note in the crate.

### Editor + CLI wiring

The editor's `content_browser::load_asset` gains an `AssetKind::Usd` arm that calls `import_usd(path)` and returns `LoadedAsset::Splats(import.splats)` (mirroring vxm/spz/ply, which already return splats directly; gltf returns a path because its import lives in the shell — USD follows the splat-returning path so the editor gets geometry immediately). Lights/camera/entities flow to the runtime via `EngineRuntime::spawn(name).with_position(p).with_light(c,i,r)` and `world.resource_mut::<CameraState>()` in the shell layer that already consumes `LoadedAsset`. A `vox_tools usd-import <file>` subcommand prints the Done-When stats line and is the CLI smoke surface.

## Data Models

```rust
// crates/vox_usd/src/lib.rs

use glam::{Quat, Vec3};
use vox_core::types::GaussianSplat;

/// Result of importing a USD scene — mirrors vox_data::gltf_import::ImportResult
/// closely enough that the editor consumes it like the other splat importers.
pub struct UsdImport {
    pub splats:   Vec<GaussianSplat>,
    pub lights:   Vec<UsdLight>,
    pub camera:   Option<UsdCamera>,
    pub entities: Vec<UsdEntity>,   // named transform nodes for the outliner
    pub warnings: Vec<String>,
    pub stats:    UsdImportStats,
}

pub struct UsdLight {
    pub name: String,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub kind: UsdLightKind,
}
pub enum UsdLightKind { Sphere, Rect, Disk, Distant, Dome }

pub struct UsdCamera {
    pub name: String,
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_y_deg: f32,
}

pub struct UsdEntity {
    pub name: String,
    pub path: String,
    pub world: [[f32; 4]; 4],
    pub type_name: String,
}

#[derive(Default)]
pub struct UsdImportStats {
    pub prims: usize,
    pub meshes: usize,
    pub points: usize,
    pub instancers: usize,
    pub splats: usize,
    pub lights: usize,
}

#[derive(Debug)]
pub enum UsdError {
    Open(String),         // file missing / unreadable
    Empty,                // stage composed but yielded no importable prims
    UnsupportedTextArray, // .usda whose only geometry is array/tuple-valued (parser limitation)
}
impl std::fmt::Display for UsdError { /* … */ }
impl std::error::Error for UsdError {}

pub struct UsdImportSettings {
    pub up_axis_correction: bool,   // default true
    pub default_opacity: u8,        // default 240
    pub mesh_splats_per_sqm: f32,   // default 200.0
    pub max_splats: usize,          // hard ceiling; overflow → warning, default 5_000_000
}
impl Default for UsdImportSettings { /* spm=200.0, opacity=240, correction=true, max=5e6 */ }

// internal traversal state (private)
struct Walk<'s> {
    stage: &'s openusd_rs::usd::Stage,
    settings: &'s UsdImportSettings,
    out: UsdImport,
    root_correction: glam::DMat4,
}
impl<'s> Walk<'s> {
    fn visit(&mut self, prim: &openusd_rs::usd::Prim, parent_world: glam::DMat4);
    fn read_local(&self, prim: &openusd_rs::usd::Prim) -> glam::DMat4;  // get_local_matrix → DMat4 (transpose)
    fn mesh_to_splats(&mut self, prim: &openusd_rs::usd::Prim, world: glam::DMat4, color_rgb: [f32; 3]);
    fn instancer_to_splats(&mut self, prim: &openusd_rs::usd::Prim, world: glam::DMat4);
    fn points_to_splats(&mut self, prim: &openusd_rs::usd::Prim, world: glam::DMat4);
    fn read_light(&mut self, prim: &openusd_rs::usd::Prim, world: glam::DMat4);
    fn read_camera(&mut self, prim: &openusd_rs::usd::Prim, world: glam::DMat4);
    fn bound_diffuse_rgb(&self, prim: &openusd_rs::usd::Prim) -> [f32; 3]; // default grey (0.5,0.5,0.5)
}
```

Splat construction reuses existing ctors: `GaussianSplat::surface(pos, tan_u, tan_v, scale_u, scale_v, opacity, spectral)` for meshes; `GaussianSplat::volume(pos, [sx,sy,sz], quat, opacity, spectral)` for instancer/points.

## API

```rust
// crates/vox_usd/src/lib.rs
use std::path::Path;

/// Top-level entry. Opens + composes the stage and traverses it.
pub fn import_usd(path: &Path) -> Result<UsdImport, UsdError>;

/// Same, with explicit settings (splat density, up-axis correction, ceiling).
pub fn import_usd_with(path: &Path, settings: &UsdImportSettings) -> Result<UsdImport, UsdError>;
```

```rust
// ─── exact openusd-rs signatures this crate calls (verified against source) ───
// usd::Stage::open(path: impl AsRef<Path>) -> Stage                         (usd/stage.rs:19)
// Stage::pseudo_root(&self) -> Prim<'_>                                     (usd/stage.rs:27)
// Prim::children<'b>(&'b self) -> ChildrenIter<'b>                          (usd/prim.rs:18)
// Prim::type_name(&self) -> tf::Token                                       (usd/prim.rs:26)
// Prim::specifier(&self) -> Option<sdf::Specifier>                          (usd/prim.rs:14)
// Prim::is_valid(&self) -> bool   // true only for Def or Class             (usd/prim.rs:35)
// Prim::attribute<'b>(&'b self, name: &tf::Token) -> Attribute<'b>          (usd/prim.rs:64)
// Prim::relationship<'b>(&'b self, name: &tf::Token) -> Relationship<'b>    (usd/prim.rs:78)
// Object::name(&self) -> tf::Token ; Object::path(&self) -> &sdf::Path      (usd/object.rs:40/45)
// Object::metadata<T: ValueType>(&self, key: &tf::Token) -> Option<T>       (usd/object.rs:55)
// Attribute::try_get<T: ValueType>(&self) -> Option<T>                      (usd/attribute.rs:25)  // USE THIS
// Attribute::get_value(&self) -> Option<vt::Value>                          (usd/attribute.rs:31)  // OR THIS
// Attribute::get<T>(&self) -> T   // PANICS on mismatch — NEVER CALL        (usd/attribute.rs:17)
// Relationship::targets(&self) -> Vec<sdf::Path>                            (usd/relationship.rs)
// usd_geom::XformOp::get_local_matrix(prim: &Prim) -> Option<gf::Matrix4d>  (usd_geom/xform_op.rs:172)
// usd_geom::triangulate(mesh: &Mesh) -> Vec<i32>                            (usd_geom/generated.rs:261)
// usd_geom::{Mesh, PointBased, PointInstancer}::define + read attrs         (usd_geom/generated.rs:119/169)
// usd_lux::{SphereLight,RectLight,DiskLight,DomeLight}::define + LightAPI    (usd_lux/generated.rs:247-387)
// usd_shade::MaterialBindingAPI::bound_material(&self, &Stage) -> Option<Material>  (usd_shade/shader.rs:118)
// usd_shade::Shader::get_input::<T>(&self, name: &str) -> Option<T>         (usd_shade/shader.rs:40)
// tf::Token::new(s) ; gf::Vec3f{x,y,z} ; gf::Matrix4d (row-major [[f64;4];4])
// vox_data::SpectralUpsampler::from_rgb(r: f32, g: f32, b: f32) -> [f32; 16]   // Smits 1999
// vox_core::types::GaussianSplat::{surface, volume, is_surface, is_volume, spectral_f32}
```

## Wiring

Every component is wired in the *same* task that implements it — no "wire later". Each helper (mesh sampler, instancer reader, light reader, camera reader, editor `load_asset` arm) ships with its own real-outcome test.

| Component | Called from | File |
|---|---|---|
| `vox_usd` crate registered | workspace `members += "crates/vox_usd"` | `Cargo.toml` |
| sibling path dep on openusd-rs | crate manifest | `crates/vox_usd/Cargo.toml` (`openusd-rs = { path = "../../../openusd-rs" }`) |
| `AssetKind::Usd` + ext detection | `AssetKind::from_path` adds `usd`/`usdc`/`usda`; `label()` + `all()` updated | `crates/vox_editor/src/content_browser.rs:38/56` |
| `import_usd(path)` → splats | new arm `AssetKind::Usd => Ok(LoadedAsset::Splats(vox_usd::import_usd(path)?.splats))`; add `BrowserError::Usd` | `crates/vox_editor/src/content_browser.rs:543` |
| lights/camera/entities → runtime | shell consuming `LoadedAsset`: `EngineRuntime::spawn(name).with_position(p).with_light(c,i,r)` per `UsdLight`/`UsdEntity`; camera → `world.resource_mut::<CameraState>()` | `crates/vox_core/src/engine_runtime.rs:644/841/877` |
| RGB → spectral | `SpectralUpsampler::from_rgb` (vox_data, engine crate) | `crates/vox_data/src/spectral_upsampler.rs:36` |
| CLI smoke path | `vox_tools` subcommand `usd-import <file>` printing the Done-When stats lines | `crates/vox_tools/src/main.rs` (new clap arm) |
| engine-isolation gate | CI step `cargo tree -p vox_core -p vox_render -p vox_app \| grep openusd` must be empty | CI workflow |

## Open Questions (resolved)

- **Does `vox_usd` depend on `vox_data` or only `vox_core`?** → Depends on `vox_data` for `SpectralUpsampler::from_rgb`. `vox_data` is itself a game-agnostic engine crate, so this is not a game-layer violation, and there is no `vox_data → vox_usd` back-edge (verified: `vox_data` has no need to import `vox_usd`). Keeps color→spectrum self-contained in the importer.
- **USDA vs USDC happy path?** → `.usdc`/`.usd` (binary) is the documented happy path and Blender's default export. `.usda` with array/tuple geometry is honestly unsupported (parser limitation) and returns `UsdError::UnsupportedTextArray` rather than a silent empty import.
- **Camera/DistantLight with no schema?** → Read generically by `type_name()` string + raw `Attribute::try_get` on `focalLength`/`horizontalAperture`/`inputs:intensity` etc. Verified zero Camera/DistantLight schema in openusd-rs; this is required, not optional.
- **Up-axis correction default?** → On by default (USD default Y-up; Blender often Z-up). A known post-correction camera position is asserted in tests to catch a wrong matrix.
- **Editor: return splats or a path?** → Return `LoadedAsset::Splats` (like vxm/spz/ply) so geometry lands immediately, rather than `LoadedAsset::Scene(PathBuf)` (gltf's deferred path), because the import is done synchronously in `vox_usd`.

## Out of Scope

- **USD export / round-trip of edited scenes.** The USDC writer authors only `Float` + `I32Array`; the USDA writer has no `quatf[]`/`float3[]` impls and no `over` specifier; the USDA parser cannot read arrays/tuples back. Authoring splat blobs or a non-destructive session-layer round-trip is a separate, larger effort and is deferred (the stage-centric session-layer approach was considered and rejected for this slice on exactly these gaps).
- **Editing splats per-element non-destructively.** Requires real array authoring (absent). VXM remains the editable runtime payload.
- **Arbitrary DCC `.usda` text import with array geometry.** Blocked by the parser until openusd-rs gains array parsing; re-export as `.usdc` is the workaround.
- **In-process re-import of an edited file.** The openusd-rs layer cache is process-global with no invalidation; tests use unique temp paths, and the editor must re-import via a fresh path or process.
- **Concurrent `vox_render`/`vox_ui`/`vox_app` WIP.** Not touched here; the importer feeds them through the existing `LoadedAsset` → splat path.

## Related

- `crates/vox_data/src/gltf_import.rs` — `ImportResult` contract + barycentric splat sampler this importer mirrors.
- `crates/vox_data/src/spectral_upsampler.rs` — `SpectralUpsampler::from_rgb` (Smits 1999), shared RGB→16-band path.
- `crates/vox_core/src/types.rs` — `GaussianSplat` (96 B) `surface`/`volume` ctors.
- `crates/vox_editor/src/content_browser.rs` — `AssetKind` / `LoadedAsset` / `load_asset` dispatch.
- `~/src/crucible/rust/crates/crucible-usd` — existing USD *writers* (`RawAttr`+`LayerWriter` technique) and the proven sibling-path-dep precedent.
- `~/src/openusd-rs` — the read/compose engine (NOT modified).

---

## IMPORTANT NOTES — verified openusd-rs gap list (read against source, not guessed)

These are the load-bearing constraints. Each is handled WITHOUT modifying openusd-rs.

**REAL GAPS (capability missing):**

1. **USDA text parser cannot read arrays `[...]` or tuples `(...)` — both return `vt::Value::empty()`** (`src/usda/parser.rs:385-386`; `parse_array`/`parse_tuple` at :332/:341 skip contents; a unit test at :821/:834 confirms even single tuples like `(1, 2.5, "world")` are skipped). `number` is f64-only and lacks scientific notation/signs (parser.rs:263 `// TODO: Handle floats, scientific notation, signs`). **Consequence:** from `.usda`, `point3f[] points`, `color3f` tuples, and `float3 xformOp:translate` tuples are all EMPTY. **Workaround:** require `.usdc`/`.usd` (binary) for geometry — the binary parser reads every typed array (`Half`, `Vec3f`/`Vec3d`, `Quatf`, `Matrix4d`, `FloatArray`, `TokenArray` — `src/usdc/parser.rs:728-784`). This is the single load-bearing constraint of the design.

2. **No `UsdGeomCamera` schema** anywhere (grep over `src/` returns nothing). **Workaround:** read Camera-type prims via `type_name()` + raw `Attribute::try_get` on `focalLength`/`horizontalAperture`/`focusDistance`.

3. **No `UsdLuxDistantLight` schema** (`src/usd_lux/generated.rs` has only Sphere/Rect/Disk/Dome + LightAPI). **Workaround:** handle `"DistantLight"` by type-name string + raw `inputs:intensity`/`inputs:color`/`inputs:exposure` reads.

4. **No `UsdGeomPoints` (splat-native) schema** — only `PointBased` (points/normals/widths) and `PointInstancer` (`src/usd_geom/generated.rs:119/169`). Splats import via `PointInstancer` positions/scales/orientations or raw `PointBased` points + widths. This is exactly where OpenUSD's forthcoming Particle-Fields splat schema will slot.

5. **Layer cache never invalidates** — process-global `static LAYER_CACHE_LOCK: OnceLock<Mutex<HashMap<PathBuf, Arc<Layer>>>>` (`src/sdf/layer.rs:9`), keyed by absolute path, with no public flush/reload. Re-reading an edited file in-process returns stale specs. **Mitigation only** (unique temp paths / fresh process); not fixable without modifying openusd-rs.

6. **`Attribute::get<T>()` panics** on type mismatch/absence (`src/usda/attribute.rs:17`, `.unwrap()`). Not a gap to fix — **never call it**; `try_get` (:25) and `get_value` (:31) are total.

7. **USDC writer authors only `Float` + `I32Array`** (`src/usdc/writer.rs:91-94`; `IntoAttrValue` impls at :919/:925). The USDA writer's `ToVtValue` impls cover scalars, `Vec<f32>`(`float[]`), `Vec<i32>`(`int[]`), `Vec<[f32;3]>`(`point3f[]`), `Vec<[f32;4]>`(`color4f[]`), tokens/strings — but **no `quatf[]`/`float3[]`**, and `PrimNode::serialize` always emits `def`, never `over`. Together these block USD *export*/round-trip of splat arrays — hence export is Out of Scope (the `RawAttr` newtype trick from `crucible-usd/src/{atmosphere,lighting}.rs` — a private per-file struct with field `prefix`, copied not imported — would be the route if export is later wanted).

**NON-GAPS (confirmed present and sufficient for the read side):**

- Full PCP composition works: sublayer strength ordering, references, payloads, inherits (incl. implied global), specializes, variants + selections, relocates, listOp combine (Int/Token/Path/Reference/Payload), dictionary merge, active-based pruning. `Stage::open` auto-dispatches `.usda` vs `.usdc`/`.usd` by extension.
- Xform composition fully implemented incl. inverse ops and all rotation orders (`usd_geom/xform_op.rs:172` `get_local_matrix`).
- Material binding + UsdPreviewSurface input/connection following exists (`usd_shade/shader.rs:40/118` — `get_input`, `bound_material`).
- `vt` value types cover everything needed: `gf::Vec3f/Vec3d/Vec3h` arrays, `Quatf`, `Matrix4d`, Token/AssetPath/String arrays, f32/f64/i32.

Verification commands run: `grep` over `~/src/openusd-rs/src/{usda,usdc,usd,usd_geom,usd_lux,usd_shade,sdf}` for the signatures above, and over `~/src/ochroma/crates/{vox_core,vox_data,vox_editor}` confirming `GaussianSplat::{surface,volume,is_surface,is_volume,spectral_f32}`, `SpectralUpsampler::from_rgb`, `LoadedAsset::{Splats,Scene}`, `AssetKind::from_path`, and `EngineRuntime::{spawn,with_position,with_light}` / `CameraState`.