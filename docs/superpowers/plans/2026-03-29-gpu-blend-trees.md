# GPU Blend Tree Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `AnimationDriver` with a state machine that evaluates blend trees (weighted sums of multiple animation poses) entirely on GPU — zero CPU skinning in the hot path.

**Architecture:** A `BlendTree` is a simple weighted sum of N animation clips evaluated at the current time. The CPU evaluates the state machine (which clip plays, what weights, transitions) — a cheap O(states) operation. It then uploads N joint-transform arrays and N blend weights to the GPU. An extended `skinning.wgsl` compute shader reads all N pose buffers, blends joint matrices on GPU, then skins splats in the same dispatch. Net result: one compute dispatch per character, zero CPU skinning allocations, N-way blending in parallel.

**Why better than Unreal:** Unreal's animation graph runs on CPU (even with "GPU cloth simulation" the joint evaluation is CPU). Ochroma moves joint evaluation + blending + skinning entirely to one GPU compute pass. For 10 characters each with 50 joints and 308k splats, Unreal's CPU cost scales linearly; Ochroma's GPU cost scales with cores.

**Tech Stack:** wgpu 24 compute, WGSL, existing `SkinningCompute` + `skinning.wgsl`, `vox_data::gltf_animation::{GltfAnimation, GltfSkeleton, evaluate_animation}`.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_render/src/gpu/blend_skinning.wgsl` | Create | Extended skinning shader with N-pose blending |
| `crates/vox_render/src/gpu/blend_skinning_compute.rs` | Create | `BlendSkinningCompute` — blends N poses on GPU |
| `crates/vox_render/src/gpu/mod.rs` | Modify | `pub mod blend_skinning_compute;` |
| `crates/vox_render/src/animation_driver.rs` | Modify | Add `BlendTree`, `AnimState`, `tick_blend_gpu()` |
| `crates/vox_render/Cargo.toml` | Modify | naga already in dev-deps (no change needed) |

---

## Task 1: Blend skinning WGSL shader

**Files:**
- Create: `crates/vox_render/src/gpu/blend_skinning.wgsl`

The shader takes `MAX_POSES=4` pose buffers. Each pose is an array of `JointTransform`. A `blend_weights` array specifies how much each pose contributes.

- [ ] Create `crates/vox_render/src/gpu/blend_skinning.wgsl`:

```wgsl
// Blend Tree Skinning Compute Shader
//
// Blends up to 4 animation poses on GPU before skinning.
// One thread per splat. Workgroup size 64.

const MAX_POSES: u32 = 4u;

struct GpuSkinSplat {
    position: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,
    opacity: f32,
    rotation: vec4<f32>,
    spectral: array<f32, 8>,
};

struct JointTransform {
    skin_matrix: mat4x4<f32>,
};

struct BlendUniform {
    weights: vec4<f32>,   // blend weight for each pose (must sum to 1.0)
    joint_count: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read>       base_splats:    array<GpuSkinSplat>;
@group(0) @binding(1) var<storage, read>       joint_bindings: array<u32>;
@group(0) @binding(2) var<storage, read>       pose0:          array<JointTransform>;
@group(0) @binding(3) var<storage, read>       pose1:          array<JointTransform>;
@group(0) @binding(4) var<storage, read>       pose2:          array<JointTransform>;
@group(0) @binding(5) var<storage, read>       pose3:          array<JointTransform>;
@group(0) @binding(6) var<uniform>             blend:          BlendUniform;
@group(0) @binding(7) var<storage, read_write> skinned_splats: array<GpuSkinSplat>;

// Matrix lerp: weighted sum of 4 matrices
fn blend_matrices(
    m0: mat4x4<f32>, w0: f32,
    m1: mat4x4<f32>, w1: f32,
    m2: mat4x4<f32>, w2: f32,
    m3: mat4x4<f32>, w3: f32,
) -> mat4x4<f32> {
    return m0 * w0 + m1 * w1 + m2 * w2 + m3 * w3;
}

fn mat4_to_quat(m: mat4x4<f32>) -> vec4<f32> {
    let sx = length(m[0].xyz);
    let sy = length(m[1].xyz);
    let sz = length(m[2].xyz);
    let r = mat3x3<f32>(m[0].xyz / sx, m[1].xyz / sy, m[2].xyz / sz);
    let trace = r[0][0] + r[1][1] + r[2][2];
    var q: vec4<f32>;
    if trace > 0.0 {
        let s = 0.5 / sqrt(trace + 1.0);
        q = vec4<f32>((r[2][1]-r[1][2])*s, (r[0][2]-r[2][0])*s, (r[1][0]-r[0][1])*s, 0.25/s);
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]);
        q = vec4<f32>(0.25*s, (r[0][1]+r[1][0])/s, (r[0][2]+r[2][0])/s, (r[2][1]-r[1][2])/s);
    } else if r[1][1] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]);
        q = vec4<f32>((r[0][1]+r[1][0])/s, 0.25*s, (r[1][2]+r[2][1])/s, (r[0][2]-r[2][0])/s);
    } else {
        let s = 2.0 * sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]);
        q = vec4<f32>((r[0][2]+r[2][0])/s, (r[1][2]+r[2][1])/s, 0.25*s, (r[1][0]-r[0][1])/s);
    }
    return normalize(q);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}

@compute @workgroup_size(64)
fn cs_blend_skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&base_splats) { return; }

    let splat = base_splats[idx];
    let joint_idx = joint_bindings[idx];

    // Blend 4 pose matrices for this joint
    let blended = blend_matrices(
        pose0[joint_idx].skin_matrix, blend.weights.x,
        pose1[joint_idx].skin_matrix, blend.weights.y,
        pose2[joint_idx].skin_matrix, blend.weights.z,
        pose3[joint_idx].skin_matrix, blend.weights.w,
    );

    var out = splat;
    out.position = (blended * vec4<f32>(splat.position, 1.0)).xyz;
    let joint_q = mat4_to_quat(blended);
    out.rotation = normalize(quat_mul(joint_q, splat.rotation));

    skinned_splats[idx] = out;
}
```

- [ ] Commit:
```bash
git commit -m "feat(gpu): blend_skinning.wgsl — 4-pose GPU blend tree skinning shader"
```

---

## Task 2: BlendSkinningCompute Rust struct

**Files:**
- Create: `crates/vox_render/src/gpu/blend_skinning_compute.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`

- [ ] Add `pub mod blend_skinning_compute;` to `crates/vox_render/src/gpu/mod.rs`.

- [ ] Create `crates/vox_render/src/gpu/blend_skinning_compute.rs`:

```rust
//! GPU blend tree skinning — blends up to 4 animation poses before skinning.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::gpu::skinning_compute::{GpuSkinSplat, GpuJointTransform};

const MAX_POSES: usize = 4;

/// Blend weights uniform — sent to GPU each frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlendUniform {
    pub weights: [f32; 4],   // Must sum to 1.0
    pub joint_count: u32,
    pub _pad: [u32; 3],
}

/// Owns GPU resources for 4-pose blend tree skinning.
pub struct BlendSkinningCompute {
    pipeline: wgpu::ComputePipeline,
    _base_splat_buffer: wgpu::Buffer,
    _joint_binding_buffer: wgpu::Buffer,
    pose_buffers: [wgpu::Buffer; MAX_POSES],
    blend_uniform_buffer: wgpu::Buffer,
    pub skinned_splat_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub splat_count: u32,
    pub joint_count: u32,
}

impl BlendSkinningCompute {
    pub fn new(
        device: &wgpu::Device,
        base_splats: &[GpuSkinSplat],
        joint_bindings: &[u32],
        joint_count: usize,
    ) -> Self {
        assert_eq!(base_splats.len(), joint_bindings.len());
        let splat_count = base_splats.len() as u32;
        let jc = joint_count.max(1);
        let joint_buf_size = (jc * std::mem::size_of::<GpuJointTransform>()) as u64;
        let identity = GpuJointTransform { skin_matrix: glam::Mat4::IDENTITY.to_cols_array_2d() };
        let identity_data: Vec<GpuJointTransform> = vec![identity; jc];

        let base_splat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_base_splats"),
            contents: bytemuck::cast_slice(base_splats),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let joint_binding_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_joint_bindings"),
            contents: bytemuck::cast_slice(joint_bindings),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pose_buffers: [wgpu::Buffer; MAX_POSES] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("blend_pose_{}", i)),
                contents: bytemuck::cast_slice(&identity_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });

        let blend_uniform = BlendUniform {
            weights: [1.0, 0.0, 0.0, 0.0],
            joint_count: jc as u32,
            _pad: [0; 3],
        };
        let blend_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_uniform"),
            contents: bytemuck::bytes_of(&blend_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let skinned_splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blend_skinned_output"),
            size: ((base_splats.len() * std::mem::size_of::<GpuSkinSplat>()) as u64).max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend_bgl"),
            entries: &[
                Self::storage_entry(0, true),   // base_splats
                Self::storage_entry(1, true),   // joint_bindings
                Self::storage_entry(2, true),   // pose0
                Self::storage_entry(3, true),   // pose1
                Self::storage_entry(4, true),   // pose2
                Self::storage_entry(5, true),   // pose3
                wgpu::BindGroupLayoutEntry {    // blend uniform
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                Self::storage_entry(7, false),  // skinned_splats output
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: base_splat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: joint_binding_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pose_buffers[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pose_buffers[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pose_buffers[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pose_buffers[3].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: blend_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: skinned_splat_buffer.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blend_skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blend_skinning.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blend_skinning_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blend_skinning_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_blend_skin"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            _base_splat_buffer: base_splat_buf,
            _joint_binding_buffer: joint_binding_buf,
            pose_buffers,
            blend_uniform_buffer,
            skinned_splat_buffer,
            bind_group,
            splat_count,
            joint_count: jc as u32,
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Upload joint matrices for pose slot `pose_idx` (0–3).
    pub fn update_pose(&self, queue: &wgpu::Queue, pose_idx: usize, joints: &[GpuJointTransform]) {
        assert!(pose_idx < MAX_POSES);
        queue.write_buffer(&self.pose_buffers[pose_idx], 0, bytemuck::cast_slice(joints));
    }

    /// Upload blend weights. `weights` must sum to approximately 1.0.
    pub fn update_weights(&self, queue: &wgpu::Queue, weights: [f32; 4]) {
        let u = BlendUniform { weights, joint_count: self.joint_count, _pad: [0; 3] };
        queue.write_buffer(&self.blend_uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Encode compute dispatch.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wgs = self.splat_count.div_ceil(64);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blend_skinning_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wgs, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn blend_skinning_wgsl_parses_and_validates() {
        let src = include_str!("blend_skinning.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        v.validate(&module).expect("WGSL validation error");
    }

    #[test]
    fn blend_uniform_is_pod() {
        use bytemuck::Zeroable;
        let _ = super::BlendUniform::zeroed();
    }
}
```

- [ ] Run:
```bash
cargo test -p vox_render blend_skinning
```

- [ ] Commit:
```bash
git commit -m "feat(gpu): BlendSkinningCompute — 4-pose GPU blend tree skinning"
```

---

## Task 3: AnimationDriver blend tree integration

**Files:**
- Modify: `crates/vox_render/src/animation_driver.rs`

- [ ] Add `blend_gpu: Option<BlendSkinningCompute>` field to `AnimationDriver` (all construction sites, init `None`).

- [ ] Add `AnimState` and `BlendTree` types and `tick_blend_gpu()` method:

```rust
use crate::gpu::blend_skinning_compute::{BlendSkinningCompute, BlendUniform};
use vox_data::gltf_animation::evaluate_animation;

/// Describes a transition between two animation states.
#[derive(Clone, Debug)]
pub struct AnimTransition {
    pub from: usize,
    pub to: usize,
    pub duration: f32,   // crossfade time in seconds
}

/// Simple state machine: current/next animation + crossfade progress.
pub struct AnimStateMachine {
    pub current: usize,
    pub next: Option<usize>,
    pub blend: f32,       // 0.0 = fully current, 1.0 = fully next
    pub transition_duration: f32,
}

impl AnimStateMachine {
    pub fn new(start: usize) -> Self {
        Self { current: start, next: None, blend: 0.0, transition_duration: 0.2 }
    }

    /// Request a transition to `target` animation.
    pub fn transition_to(&mut self, target: usize) {
        if self.current == target { return; }
        self.next = Some(target);
        self.blend = 0.0;
    }

    /// Advance the state machine. Returns `(current_idx, next_idx_or_same, blend_weight)`.
    pub fn tick(&mut self, dt: f32) -> (usize, usize, f32) {
        if let Some(next) = self.next {
            self.blend += dt / self.transition_duration;
            if self.blend >= 1.0 {
                self.current = next;
                self.next = None;
                self.blend = 0.0;
            }
            (self.current, next, self.blend.clamp(0.0, 1.0))
        } else {
            (self.current, self.current, 0.0)
        }
    }
}

impl AnimationDriver {
    /// GPU-accelerated tick with blend tree support (up to 2 active poses).
    ///
    /// - If `blend_gpu` is None, falls back to `tick()`.
    /// - `state_machine`: drives which animations to blend and at what weight.
    pub fn tick_blend_gpu(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        state_machine: &mut AnimStateMachine,
    ) -> bool {
        let blend_gpu = match &self.blend_gpu {
            Some(g) => g,
            None => return false,
        };
        if self.animations.is_empty() { return false; }

        self.time += dt * self.speed;

        let (cur_idx, next_idx, next_weight) = state_machine.tick(dt);
        let cur_weight = 1.0 - next_weight;

        let cur_anim = &self.animations[cur_idx.min(self.animations.len() - 1)];
        let next_anim = &self.animations[next_idx.min(self.animations.len() - 1)];

        let cur_time = if self.looping && cur_anim.duration > 0.0 {
            self.time % cur_anim.duration
        } else { self.time };

        let inv_binds: Vec<glam::Mat4> = self.skeleton.joints.iter()
            .map(|j| j.inverse_bind_matrix)
            .collect();

        // Pose 0: current animation
        let cur_transforms = evaluate_animation(&self.skeleton, cur_anim, cur_time);
        let pose0: Vec<crate::gpu::skinning_compute::GpuJointTransform> = cur_transforms.iter()
            .zip(inv_binds.iter())
            .map(|(t, inv)| crate::gpu::skinning_compute::GpuJointTransform {
                skin_matrix: (*t * *inv).to_cols_array_2d(),
            })
            .collect();

        // Pose 1: next animation (or same if no transition)
        let next_transforms = evaluate_animation(&self.skeleton, next_anim, cur_time);
        let pose1: Vec<crate::gpu::skinning_compute::GpuJointTransform> = next_transforms.iter()
            .zip(inv_binds.iter())
            .map(|(t, inv)| crate::gpu::skinning_compute::GpuJointTransform {
                skin_matrix: (*t * *inv).to_cols_array_2d(),
            })
            .collect();

        blend_gpu.update_pose(queue, 0, &pose0);
        blend_gpu.update_pose(queue, 1, &pose1);
        blend_gpu.update_weights(queue, [cur_weight, next_weight, 0.0, 0.0]);
        blend_gpu.dispatch(encoder);
        true
    }
}
```

- [ ] Run:
```bash
cargo check -p vox_render
```

- [ ] Commit:
```bash
git commit -m "feat(render): AnimStateMachine + tick_blend_gpu for GPU blend tree transitions"
```

---

## Acceptance Criteria

| # | Test | Command |
|---|------|---------|
| 1 | WGSL shader validates | `cargo test -p vox_render blend_skinning_wgsl_parses_and_validates` |
| 2 | BlendUniform is Pod | `cargo test -p vox_render blend_uniform_is_pod` |
| 3 | vox_render compiles | `cargo check -p vox_render` |
| 4 | Full workspace green | `cargo test` |
