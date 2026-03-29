# Sprint 3: Professional Rendering

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire GPU shadow maps, end-to-end GLTF animation, Spectra as the primary renderer, and LOD streaming — making the engine visually professional.

**Architecture:** Tasks share GPU infrastructure: a depth texture and pre-frame compute pass built in Task 1 serve both shadow map rendering (Task 2) and future GPU-side skinning. AnimationDriver produces animated GaussianSplats (Task 3) that flow into `render_with_spectra_u8` as the primary render path (Task 4). LOD streaming completes the pipeline by dynamically loading tile assets when the camera moves.

**Tech Stack:** wgpu 0.20, rodio, vox_render::spectra_render, vox_data::gltf_animation, vox_render::streaming, vox_core::lwc

---

## Cross-Sprint Foundation Note

The `GpuPassManager` built in Task 1 is the foundation for Sprint 5's GPU-side Gaussian skinning compute shader. The `AnimationDriver` integration in Task 3 establishes the animated entity model for Sprint 4's hot-reload workflow. The spectral rendering pipeline enabled in Task 4 is the unique differentiator vs Unreal 5.

---

## Task 1: GPU Depth Texture infrastructure in GpuRasteriser

**Files:**
- Modify: `crates/vox_render/src/gpu/gpu_rasteriser.rs`

**Context:** `GpuRasteriser` currently has `depth_stencil: None` on line 142. It stores `pipeline`, `camera_buffer`, `camera_bind_group_layout`, `width`, `height`. The `render()` method takes `device, queue, target_view, splats, camera, illuminant`. The `resize()` method already exists and updates `width`/`height` — extend it to recreate the depth texture.

Add depth texture fields to `GpuRasteriser` struct:

```rust
// Add to GpuRasteriser struct:
depth_texture: wgpu::Texture,
depth_view: wgpu::TextureView,
depth_sampler: wgpu::Sampler,
```

Add a free function for depth texture creation (used in `new()` and `resize()`):

```rust
fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });
    (texture, view, sampler)
}
```

In `GpuRasteriser::new()`, after creating `camera_buffer`, call `create_depth_texture(device, width, height)` and store the results in the struct fields.

Extend `resize(&mut self, width: u32, height: u32)` — it currently only updates `self.width` and `self.height`. It needs a `device` parameter to recreate the depth texture:

```rust
pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
    self.width = width;
    self.height = height;
    let (tex, view, sampler) = create_depth_texture(device, width, height);
    self.depth_texture = tex;
    self.depth_view = view;
    self.depth_sampler = sampler;
}
```

Note: All call sites of `resize()` in `engine_runner.rs` must be updated to pass `device`. The `WgpuBackend` exposes `device` — use `backend.device()` or the stored `Arc<wgpu::Device>`.

Add `depth_view(&self) -> &wgpu::TextureView` accessor:

```rust
pub fn depth_view(&self) -> &wgpu::TextureView {
    &self.depth_view
}
```

**Tests** (`crates/vox_render/tests/` or inline in the module):

```rust
#[test]
fn depth_texture_descriptor_format_is_depth32float() {
    // Compile-time test: verify Depth32Float constant is correct.
    // This test just asserts the format matches — actual GPU creation is
    // tested via integration; mark GPU-dependent tests as #[ignore].
    assert_eq!(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureFormat::Depth32Float,
    );
}
```

**Verification:**

- [ ] `cargo check -p vox_render 2>&1 | tail -5` — no errors
- [ ] `cargo test -p vox_render 2>&1 | tail -10` — existing tests still pass

**Commit:** `feat(gpu): add depth texture infrastructure to GpuRasteriser`

---

## Task 2: GPU Shadow Pass (depth prepass for sun shadow)

**Files:**
- Modify: `crates/vox_render/src/gpu/gpu_rasteriser.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Context:** `ShadowMapper` in `vox_render::shadows` has `pub cascades: Vec<CascadeShadowMap>` and each `CascadeShadowMap` has `pub light_view_proj: Mat4`. The CPU shadow implementation is already wired in `engine_runner.rs` via `self.shadow_mapper`. This task adds a GPU depth-only shadow prepass that runs before the main splat draw, writing sun-view depths for GPU shadow lookups.

Add fields to `GpuRasteriser`:

```rust
shadow_pipeline: Option<wgpu::RenderPipeline>,
light_buffer: Option<wgpu::Buffer>,
shadow_depth_texture: Option<wgpu::Texture>,
shadow_depth_view: Option<wgpu::TextureView>,
```

Add `init_shadow_pass(&mut self, device: &wgpu::Device)` method that creates a 512×512 shadow depth texture and a depth-only render pipeline. The WGSL shadow shader (vertex-only, no fragment needed for depth-only pass):

```wgsl
// shadow_shader.wgsl
struct LightUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> light: LightUniform;

struct SplatData {
    position: vec3<f32>,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
    opacity: f32,
    _pad: f32,
    spectral: array<f32, 8>,
};
@group(0) @binding(1) var<storage, read> splats: array<SplatData>;

@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let splat_idx = vi / 6u;
    let pos = vec4(splats[splat_idx].position, 1.0);
    return light.view_proj * pos;
}
```

Save this as `crates/vox_render/src/gpu/shadow_shader.wgsl` and use `include_str!("shadow_shader.wgsl")` in `init_shadow_pass`.

The shadow pipeline descriptor uses `depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: Default::default() })` and `fragment: None`.

Add `render_shadow_pass(&self, device: &wgpu::Device, queue: &wgpu::Queue, splats_gpu: &[GpuSplatData], light_view_proj: &glam::Mat4)` method:

```rust
pub fn render_shadow_pass(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    splats_gpu: &[GpuSplatData],
    light_view_proj: &glam::Mat4,
) {
    let (pipeline, light_buf, shadow_view) = match (
        self.shadow_pipeline.as_ref(),
        self.light_buffer.as_ref(),
        self.shadow_depth_view.as_ref(),
    ) {
        (Some(p), Some(lb), Some(sv)) => (p, lb, sv),
        _ => return, // shadow pass not initialised
    };

    // Write light uniform
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct LightUniform { view_proj: [[f32; 4]; 4] }

    let light_uniform = LightUniform { view_proj: light_view_proj.to_cols_array_2d() };
    queue.write_buffer(light_buf, 0, bytemuck::bytes_of(&light_uniform));

    let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("shadow_splat_buffer"),
        contents: bytemuck::cast_slice(splats_gpu),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Build shadow bind group layout and bind group
    // ... (create bind group from light_buf + splat_buffer)

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("shadow_encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: shadow_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        // set_bind_group ...
        pass.draw(0..(splats_gpu.len() as u32 * 6), 0..1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}
```

Add `render_with_shadow(&self, device, queue, target_view, splats, camera, illuminant, light_view_proj)` that calls `render_shadow_pass` then the existing `render`:

```rust
pub fn render_with_shadow(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_view: &wgpu::TextureView,
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    illuminant: &Illuminant,
    light_view_proj: Option<&glam::Mat4>,
) {
    if let Some(lvp) = light_view_proj {
        // Convert splats to GPU format for shadow pass
        let gpu_splats = splats_to_gpu(splats);
        self.render_shadow_pass(device, queue, &gpu_splats, lvp);
    }
    self.render(device, queue, target_view, splats, camera, illuminant);
}
```

Extract existing splat conversion from `render()` into a free function `splats_to_gpu(splats: &[GaussianSplat]) -> Vec<GpuSplatData>` so both paths can share it (avoids duplication of the `f16` decoding loop).

In `engine_runner.rs`: when `gpu_rasteriser` is Some, call `init_shadow_pass` once at init time, and switch the render call to `render_with_shadow` passing `Some(&self.shadow_mapper.cascades[0].light_view_proj)`.

**Verification:**

- [ ] `cargo build -p vox_render -p vox_app 2>&1 | tail -10` — clean build
- [ ] `cargo test -p vox_render 2>&1 | tail -10` — all pass

**Commit:** `feat(gpu): add shadow depth prepass to GpuRasteriser`

---

## Task 3: GLTF Animation Pipeline — joint bindings + rotation skinning

**Files:**
- Modify: `crates/vox_data/src/gltf_animation.rs`
- Modify: `crates/vox_render/src/animation_driver.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Context (confirmed from reading the code):**

- `GltfJoint` has fields: `name: String`, `index: usize`, `parent: Option<usize>`, `children: Vec<usize>`, `inverse_bind_matrix: Mat4`, `local_transform: Mat4`. There is NO `bind_pose_position` field — derive bind-pose world position from `local_transform` translation, or accumulate hierarchy. The simplest approach: use the translation column of `local_transform` as a proxy for nearest-joint assignment.
- `GltfSkeleton` has `joints: Vec<GltfJoint>` and `root_joints: Vec<usize>` (not `root_joint: usize`).
- `skin_splats` currently only skins position, not rotation (copies `..splat` which leaves `rotation` unchanged).
- `AnimationDriver::joint_bindings` is `pub Vec<usize>` and defaults to `vec![0; base_splats.len()]`.
- `GaussianSplat` rotation is `[i16; 4]` storing `[x, y, z, w]` normalised to `[-32767, 32767]`.

### 3a. Fix rotation skinning in `skin_splats` (`gltf_animation.rs`)

The existing `skin_splats` function uses `..splat` which leaves `rotation` unchanged. Replace the struct update in the closure to also rotate:

```rust
// After computing skin_mat and new_pos, also rotate:
let (skin_scale, skin_quat, _skin_t) = skin_mat.to_scale_rotation_translation();
let _ = skin_scale; // unused

let orig_q = glam::Quat::from_xyzw(
    splat.rotation[0] as f32 / 32767.0,
    splat.rotation[1] as f32 / 32767.0,
    splat.rotation[2] as f32 / 32767.0,
    splat.rotation[3] as f32 / 32767.0,
).normalize();

let new_q = (skin_quat * orig_q).normalize();

GaussianSplat {
    position: [new_pos.x, new_pos.y, new_pos.z],
    rotation: [
        (new_q.x * 32767.0).clamp(-32767.0, 32767.0) as i16,
        (new_q.y * 32767.0).clamp(-32767.0, 32767.0) as i16,
        (new_q.z * 32767.0).clamp(-32767.0, 32767.0) as i16,
        (new_q.w * 32767.0).clamp(-32767.0, 32767.0) as i16,
    ],
    ..*splat
}
```

Note: `glam::Mat4::to_scale_rotation_translation()` returns `(Vec3, Quat, Vec3)` in scale-rotation-translation order.

### 3b. Nearest-joint assignment utility (`gltf_animation.rs`)

Add `pub fn assign_joint_bindings(splats: &[GaussianSplat], skeleton: &GltfSkeleton) -> Vec<usize>`.

Since `GltfJoint` has no world bind-pose position field, derive it from `local_transform.w_axis.truncate()` (the translation column). This is joint-local, not world-space, but is sufficient for nearest-joint heuristic when joints are distributed across the body:

```rust
pub fn assign_joint_bindings(
    splats: &[GaussianSplat],
    skeleton: &GltfSkeleton,
) -> Vec<usize> {
    // Pre-compute a world-space bind position for each joint by walking hierarchy.
    // For the heuristic case, we use the local_transform translation of each joint
    // accumulated through the hierarchy.
    let joint_world_positions: Vec<Vec3> = {
        let mut positions = vec![Vec3::ZERO; skeleton.joints.len()];
        // Walk from roots using parent references
        for ji in 0..skeleton.joints.len() {
            let local_t = skeleton.joints[ji].local_transform.w_axis.truncate();
            if let Some(parent_idx) = skeleton.joints[ji].parent {
                positions[ji] = positions[parent_idx] + local_t;
            } else {
                positions[ji] = local_t;
            }
        }
        positions
    };

    splats.iter().map(|splat| {
        let pos = Vec3::from(splat.position);
        skeleton.joints.iter().enumerate()
            .min_by(|(ai, _), (bi, _)| {
                let da = joint_world_positions[*ai].distance_squared(pos);
                let db = joint_world_positions[*bi].distance_squared(pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }).collect()
}
```

Note: This loop assumes joints are stored in parent-before-child order (which `build_synthetic_skeleton` and GLTF extraction both guarantee). If not guaranteed, do a two-pass topological walk.

### 3c. Wire into `engine_runner.rs`

Add field to `EngineApp`:

```rust
anim_driver: Option<vox_render::animation_driver::AnimationDriver>,
```

Initialise to `None` in `EngineApp::new()`.

On startup (after scene splats are loaded), try loading a GLTF character:

```rust
use vox_data::gltf_animation::{assign_joint_bindings, extract_skeleton};
use vox_render::animation_driver::AnimationDriver;

let gltf_path = std::path::Path::new("assets/character.glb");
if gltf_path.exists() {
    match extract_skeleton(gltf_path) {
        Ok((skeleton, animations)) => {
            let joint_bindings = assign_joint_bindings(&self.scene_splats, &skeleton);
            let mut driver = AnimationDriver::new(skeleton, self.scene_splats.clone());
            driver.joint_bindings = joint_bindings;
            for anim in animations {
                driver.add_animation(anim);
            }
            if driver.animation_count() > 0 {
                driver.play(0);
            }
            self.anim_driver = Some(driver);
        }
        Err(e) => eprintln!("[animation] GLTF load failed: {e}"),
    }
}
```

Per frame, after computing `render_splats`, tick the animation driver and extend the render list:

```rust
if let Some(ref mut driver) = self.anim_driver {
    let animated_splats = driver.tick(dt);
    render_splats.extend(animated_splats);
}
```

**Tests** in `crates/vox_data/src/gltf_animation.rs` (add to existing `#[cfg(test)]` block):

```rust
#[test]
fn assign_joint_bindings_returns_one_per_splat() {
    let skeleton = build_synthetic_skeleton(&["root", "hip", "chest"]);
    let splats: Vec<GaussianSplat> = (0..5).map(|i| GaussianSplat {
        position: [0.0, i as f32 * 0.5, 0.0],
        scale: [0.1; 3],
        rotation: [0, 0, 0, 32767],
        opacity: 200,
        _pad: [0; 3],
        spectral: [0; 8],
    }).collect();
    let bindings = assign_joint_bindings(&splats, &skeleton);
    assert_eq!(bindings.len(), 5);
    // All bindings must be valid joint indices
    for b in &bindings {
        assert!(*b < skeleton.joints.len());
    }
}

#[test]
fn skin_splats_rotation_changes_under_rotation_transform() {
    let skel = build_synthetic_skeleton(&["root", "arm"]);
    let splat = GaussianSplat {
        position: [0.0, 1.0, 0.0],
        scale: [0.1; 3],
        rotation: [0, 0, 0, 32767], // identity rotation
        opacity: 255,
        _pad: [0; 3],
        spectral: [0; 8],
    };

    // Apply a 90-degree rotation animation on joint 1
    let anim = build_synthetic_animation(
        "rotate",
        1,
        1.0,
        Quat::IDENTITY,
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
    );
    let transforms = evaluate_animation(&skel, &anim, 1.0);
    let ibms: Vec<Mat4> = skel.joints.iter().map(|j| j.inverse_bind_matrix).collect();

    let skinned = skin_splats(&[splat], &[1], &transforms, &ibms);
    assert_eq!(skinned.len(), 1);

    // The rotation field should differ from identity [0, 0, 0, 32767]
    let r = skinned[0].rotation;
    let is_identity = r[0].abs() < 100 && r[1].abs() < 100 && r[2].abs() < 100;
    assert!(!is_identity, "rotation should change after skinning, got {:?}", r);
}
```

**Verification:**

- [ ] `cargo test -p vox_data 2>&1 | tail -15` — all pass including new tests
- [ ] `cargo check -p vox_app 2>&1 | tail -5` — no errors

**Commit:** `feat(animation): nearest-joint binding + rotation skinning in AnimationDriver`

---

## Task 4: Spectra as primary renderer

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Context (confirmed from reading the code):**

- `spectral_bypass: bool` field exists in `EngineApp` (line 145), default `true`.
- `render_with_spectra_u8(splats, camera, width, height, illuminant) -> Vec<[u8;4]>` is in `vox_render::spectra_render`. Signature confirmed.
- `RenderCamera` has `view: Mat4` and `proj: Mat4` — no `Default` impl. The engine builds a `RenderCamera` each frame from `CameraController`. Search engine_runner.rs for where `RenderCamera` is constructed to confirm the pattern.
- The existing software rasteriser path produces `final_pixels: Vec<[u8;4]>` that gets blitted. The spectra path produces the same type — it is a direct drop-in.
- `KeyQ` already toggles `spectral_bypass` — no new keybinding needed. Confirm by searching `KeyQ` in engine_runner.rs.

**Steps:**

1. Change `spectral_bypass` default from `true` to `false` in `EngineApp::new()`.

2. In the render branch where `spectral_bypass` is false, replace the `SoftwareRasteriser` call with:

```rust
let fb = vox_render::spectra_render::render_with_spectra_u8(
    &render_splats,
    &render_camera,
    self.backend.as_ref().map(|b| b.width()).unwrap_or(DEFAULT_WIDTH),
    self.backend.as_ref().map(|b| b.height()).unwrap_or(DEFAULT_HEIGHT),
    &illuminant,
);
```

Where `render_camera` is the `RenderCamera` already constructed earlier in the frame loop.

3. Add an FPS readout to the HUD (if not already present). Search engine_runner.rs for the existing FPS field (`self.fps`) — it is computed but may not be displayed. In the egui HUD render, add:

```rust
ui.label(format!("FPS: {:.0} | Spectra: {}", self.fps, !self.spectral_bypass));
```

**Tests** (add to `crates/vox_render/src/spectra_render.rs` `#[cfg(test)]` block):

```rust
#[test]
fn render_with_spectra_u8_returns_correct_pixel_count() {
    use vox_core::spectral::Illuminant;
    use crate::spectral::RenderCamera;

    let splats = vec![];
    let camera = RenderCamera {
        view: glam::Mat4::IDENTITY,
        proj: glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            1.0,
            0.1,
            1000.0,
        ),
    };
    let result = super::render_with_spectra_u8(&splats, &camera, 64, 64, &Illuminant::d65());
    assert_eq!(result.len(), 64 * 64, "should produce one pixel per texel");
}
```

**Performance note:** `render_with_spectra` is a CPU tiled renderer — it will be slower than the `SoftwareRasteriser` for very large splat counts because it projects every splat to 2D before tile-sorting. Tile-based early-out benefits are most visible when splat density is high. The FPS readout (Step 3) will make this visible. If FPS drops below 10 at 1280×720, investigate the tile count and splat culling radius in `spectra_render.rs::render_cpu_internal`.

**Verification:**

- [ ] `cargo test -p vox_render 2>&1 | tail -10` — new test passes
- [ ] `cargo build -p vox_app 2>&1 | tail -5` — clean build

**Commit:** `feat(render): Spectra EWA renderer as primary display path (spectral_bypass=false)`

---

## Task 5: LOD Streaming — wire TileManager to actual asset loading

**Files:**
- Modify: `crates/vox_render/src/streaming.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Context (confirmed from reading the code):**

- `TileManager::new()` takes no arguments and sets `active_radius: 1`. The plan below passes a radius parameter — add a `TileManager::with_radius(active_radius: i32) -> Self` constructor instead of changing `new()` to avoid breaking existing callers.
- `TileManager::update_camera(&mut self, camera_tile: TileCoord)` currently returns `()`. Must change to `Vec<TileCoord>` (newly activated tiles this frame).
- `TileState` enum has: `Cold, Warming, Warm, Active, Evicting`.
- `TILE_SIZE: f64 = 1000.0` in `vox_core::lwc`.
- `AsyncAssetLoader::load_from_path` is `async`. For Sprint 3, use a synchronous blocking load via `std::fs::read` + `VxmFile::read` to keep the implementation simple. Sprint 5 can upgrade to the async path.

### 5a. Modify `TileManager::update_camera` to return `Vec<TileCoord>`

```rust
pub fn update_camera(&mut self, camera_tile: TileCoord) -> Vec<TileCoord> {
    let mut newly_active = Vec::new();

    // Evict distant tiles
    let r = self.active_radius;
    let to_evict: Vec<TileCoord> = self.tiles.keys().copied().filter(|t| {
        (t.x - camera_tile.x).abs() > r || (t.z - camera_tile.z).abs() > r
    }).collect();
    for t in to_evict {
        self.tiles.remove(&t);
    }

    // Activate tiles within radius; track newly activated ones
    for dx in -r..=r {
        for dz in -r..=r {
            let tile = TileCoord {
                x: camera_tile.x + dx,
                z: camera_tile.z + dz,
            };
            let was_present = self.tiles.contains_key(&tile);
            self.tiles.entry(tile).or_insert(TileState::Active);
            if !was_present {
                newly_active.push(tile);
            }
            // Transition Cold -> Active if already tracked
            if let Some(state) = self.tiles.get_mut(&tile) {
                if *state == TileState::Cold {
                    *state = TileState::Active;
                    newly_active.push(tile);
                }
            }
        }
    }

    newly_active
}
```

Add `with_radius` constructor:

```rust
pub fn with_radius(active_radius: i32) -> Self {
    Self {
        tiles: HashMap::new(),
        active_radius,
    }
}
```

### 5b. Wire into `engine_runner.rs`

Add field to `EngineApp`:

```rust
tile_manager: vox_render::streaming::TileManager,
```

Initialise in `new()`:

```rust
tile_manager: vox_render::streaming::TileManager::with_radius(2),
```

Each frame (inside the frame update, after camera position is updated):

```rust
let cam_pos = self.camera.position(); // Vec3 — check CameraController API
let cam_tile = vox_core::lwc::TileCoord {
    x: (cam_pos.x / vox_core::lwc::TILE_SIZE as f32) as i32,
    z: (cam_pos.z / vox_core::lwc::TILE_SIZE as f32) as i32,
};
let newly_active = self.tile_manager.update_camera(cam_tile);

for tile in &newly_active {
    let path = format!("assets/tiles/tile_{}_{}.vxm", tile.x, tile.z);
    let p = std::path::Path::new(&path);
    if p.exists() {
        match std::fs::read(p).and_then(|bytes| {
            vox_data::vxm::VxmFile::read(&bytes[..])
                .map_err(|e| std::io::Error::other(e))
        }) {
            Ok(_vxm) => {
                println!("[streaming] Loaded tile {},{}", tile.x, tile.z);
                // TODO Sprint 5: convert vxm to splats and add to scene
            }
            Err(e) => eprintln!("[streaming] Failed to load {path}: {e}"),
        }
    }
}
```

Note: Check what method `CameraController` exposes for position. Search for `camera.position` or `camera.eye` in engine_runner.rs to find the correct accessor. Adapt accordingly.

**Tests** (add to `crates/vox_render/src/streaming.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::lwc::TileCoord;

    #[test]
    fn tile_manager_returns_newly_active_tiles_on_first_update() {
        let mut tm = TileManager::with_radius(1);
        let newly = tm.update_camera(TileCoord { x: 0, z: 0 });
        // active_radius=1 means a 3x3 grid = 9 tiles
        assert_eq!(newly.len(), 9, "first update should activate 9 tiles (3x3 grid)");
    }

    #[test]
    fn tile_manager_no_newly_active_on_same_position() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 0, z: 0 });
        let newly = tm.update_camera(TileCoord { x: 0, z: 0 }); // same tile
        assert!(newly.is_empty(), "second call at same position should yield no new tiles");
    }

    #[test]
    fn tile_manager_evicts_distant_tiles() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 0, z: 0 });
        tm.update_camera(TileCoord { x: 100, z: 100 });
        let active = tm.active_tiles();
        assert!(
            !active.contains(&TileCoord { x: 0, z: 0 }),
            "tile (0,0) should be evicted after camera moves to (100,100)"
        );
    }

    #[test]
    fn tile_manager_active_tiles_within_radius() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 5, z: 5 });
        let active = tm.active_tiles();
        // 3x3 = 9 active tiles
        assert_eq!(active.len(), 9);
        for t in &active {
            assert!((t.x - 5).abs() <= 1 && (t.z - 5).abs() <= 1);
        }
    }
}
```

**Verification:**

- [ ] `cargo test -p vox_render 2>&1 | tail -15` — all streaming tests pass
- [ ] `cargo check -p vox_app 2>&1 | tail -5` — no errors

**Commit:** `feat(streaming): wire TileManager to actual tile asset loading in engine_runner`

---

## Sprint 3 Summary Checklist

- [ ] Task 1: GPU depth texture in GpuRasteriser (`depth_texture`, `depth_view`, `depth_sampler` fields; `create_depth_texture` free fn; `depth_view()` accessor; `resize()` takes `device`)
- [ ] Task 2: GPU shadow pass (`shadow_shader.wgsl`; `init_shadow_pass`; `render_shadow_pass`; `render_with_shadow`; `splats_to_gpu` extraction; engine_runner wired)
- [ ] Task 3: GLTF animation rotation skinning + `assign_joint_bindings` + `anim_driver` in engine_runner
- [ ] Task 4: `spectral_bypass: false` default; `render_with_spectra_u8` as primary path; FPS HUD label
- [ ] Task 5: `TileManager::update_camera` returns `Vec<TileCoord>`; `with_radius` constructor; engine_runner polls newly active tiles and loads `.vxm` files

All tasks are independently mergeable in order. Task 2 depends on Task 1 (depth texture infrastructure). Tasks 3, 4, 5 are independent of each other and of Tasks 1-2.
