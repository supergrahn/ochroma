# Sprint 5: Surpass Unreal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the features that make Ochroma genuinely better than Unreal 5: GPU Gaussian skinning compute shader, spectral audio procedural generation, real-time SDF terrain deformation, and rayon-parallelized EWA tile rendering.

**Architecture:** Each task adds a capability Unreal 5 cannot replicate. GPU skinning (Task 1) moves `AnimationDriver::tick()`'s per-frame `Vec<GaussianSplat>` allocation to a WGSL compute shader running entirely on GPU. Spectral audio (Task 2) adds frequency-domain synthesis to `vox_audio`, generating material-resonant impact sounds that Unreal's time-domain audio engine cannot produce. SDF terrain deformation (Task 3) wraps the existing `volume::sculpt` module behind a real-time API that physics bodies respond to. EWA tile parallelism (Task 4) uses rayon to make `render_cpu_internal`'s sequential tile loop parallel.

**Tech Stack:** wgpu 24 (compute pipelines, storage buffers), WGSL, naga 24 (WGSL validation in tests), rayon 1.10, `vox_audio::synth`, `vox_terrain::volume::sculpt`, rapier3d.

**Confirmed codebase state before starting:**
- `GpuSplatData` in `crates/vox_render/src/gpu/gpu_rasteriser.rs` — existing GPU struct with `position: [f32; 3]`, `scale_x/y/z: f32`, `opacity: f32`, `_pad: f32`, `spectral: [f32; 8]`. No rotation quaternion field yet — Task 1 adds a rotation field to the GPU compute struct (separate from `GpuSplatData`).
- `AnimationDriver::tick()` in `crates/vox_render/src/animation_driver.rs` — calls `skin_splats()` which allocates a new `Vec<GaussianSplat>` every frame.
- `TerrainVolume` in `crates/vox_terrain/src/volume.rs` — SDF with `get(x,y,z)`, `set(x,y,z,v)`, `world_to_voxel()`, `voxel_to_world()`. Module `volume::sculpt` already has `add_sphere` and `remove_sphere` — Task 3 wraps these with a `deform` module behind a cleaner real-time API.
- `render_cpu_internal` in `crates/vox_render/src/spectra_render.rs` — tile loop at lines 387-443: sequential `for ty in 0..tiles_y { for tx in 0..tiles_x { ... } }`. The inner pixel loop per tile is already self-contained. rayon is not yet a `vox_render` dependency.
- wgpu version: `24`. naga version: `24.0.0`. rayon in workspace lockfile at `1.11.0` but not yet in `vox_render/Cargo.toml`.

---

## Task 1: GPU Gaussian Skinning via wgpu Compute Shader

**Files:**
- Create: `crates/vox_render/src/gpu/skinning_compute.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`
- Create: `crates/vox_render/src/gpu/skinning.wgsl`
- Modify: `crates/vox_render/src/animation_driver.rs`
- Modify: `crates/vox_render/Cargo.toml` (add naga to dev-dependencies)

**Why:** `AnimationDriver::tick()` calls `skin_splats()` which allocates a new `Vec<GaussianSplat>` every frame. At 308k splats × 64 bytes ≈ 19MB allocation per frame at 60fps. Moving to GPU compute eliminates CPU allocation and makes skinning scale with GPU parallelism instead of CPU core count.

**Steps:**

- [ ] Create `crates/vox_render/src/gpu/skinning.wgsl` with the compute shader below.
- [ ] Create `crates/vox_render/src/gpu/skinning_compute.rs` with `SkinningCompute` struct.
- [ ] Add `pub mod skinning_compute;` to `crates/vox_render/src/gpu/mod.rs`.
- [ ] Add `gpu_skinning: Option<SkinningCompute>` field to `AnimationDriver` and implement `tick_gpu`.
- [ ] Add `naga = { version = "24", features = ["wgsl-in"] }` to `[dev-dependencies]` in `crates/vox_render/Cargo.toml`.
- [ ] Add the WGSL validation test.
- [ ] Run `cargo test -p vox_render skinning` — all tests pass.
- [ ] Commit: `feat(gpu): GPU Gaussian skinning via wgpu compute shader`

**`skinning.wgsl`** — create at `crates/vox_render/src/gpu/skinning.wgsl`:

```wgsl
// GPU Gaussian Skinning Compute Shader
// Reads base splat positions/rotations + per-frame joint transforms,
// writes skinned splats to an output buffer for the render pass.
//
// One thread per splat. Workgroup size 64 = standard GPU occupancy sweet-spot.

struct GpuSkinSplat {
    position: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,
    opacity: f32,
    rotation: vec4<f32>,       // normalized quaternion [x, y, z, w]
    spectral: array<f32, 8>,
};

struct JointTransform {
    skin_matrix: mat4x4<f32>,  // world_transform * inverse_bind_matrix
};

@group(0) @binding(0) var<storage, read>       base_splats:      array<GpuSkinSplat>;
@group(0) @binding(1) var<storage, read>       joint_bindings:   array<u32>;
@group(0) @binding(2) var<storage, read>       joint_transforms: array<JointTransform>;
@group(0) @binding(3) var<storage, read_write> skinned_splats:   array<GpuSkinSplat>;

// Extract rotation quaternion from a 4x4 matrix (Shepperd's method).
// Assumes the upper-left 3x3 is a pure rotation matrix (no shear/scale).
fn mat4_to_quat(m: mat4x4<f32>) -> vec4<f32> {
    let sx = length(m[0].xyz);
    let sy = length(m[1].xyz);
    let sz = length(m[2].xyz);
    let r = mat3x3<f32>(
        m[0].xyz / sx,
        m[1].xyz / sy,
        m[2].xyz / sz,
    );
    let trace = r[0][0] + r[1][1] + r[2][2];
    var q: vec4<f32>;
    if trace > 0.0 {
        let s = 0.5 / sqrt(trace + 1.0);
        q = vec4<f32>(
            (r[2][1] - r[1][2]) * s,
            (r[0][2] - r[2][0]) * s,
            (r[1][0] - r[0][1]) * s,
            0.25 / s,
        );
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]);
        q = vec4<f32>(0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s);
    } else if r[1][1] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]);
        q = vec4<f32>((r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s);
    } else {
        let s = 2.0 * sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]);
        q = vec4<f32>((r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s, (r[1][0] - r[0][1]) / s);
    }
    return normalize(q);
}

// Multiply two quaternions: result = a * b (Hamilton product).
// Both in [x, y, z, w] convention.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

@compute @workgroup_size(64)
fn cs_skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&base_splats) { return; }

    let splat = base_splats[idx];
    let joint_idx = joint_bindings[idx];
    let skin = joint_transforms[joint_idx].skin_matrix;

    var out = splat;

    // Transform position by skin matrix
    out.position = (skin * vec4<f32>(splat.position, 1.0)).xyz;

    // Compose joint rotation with splat orientation quaternion
    let joint_q = mat4_to_quat(skin);
    out.rotation = normalize(quat_mul(joint_q, splat.rotation));

    skinned_splats[idx] = out;
}
```

**`skinning_compute.rs`** — create at `crates/vox_render/src/gpu/skinning_compute.rs`:

```rust
//! GPU compute pass for Gaussian splat skinning.
//!
//! Replaces `AnimationDriver::tick()`'s CPU `skin_splats()` call with a
//! wgpu compute dispatch.  One GPU thread per splat; eliminates the per-frame
//! Vec<GaussianSplat> allocation.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU-side splat struct for the skinning compute shader.
/// Fields must match the WGSL `GpuSkinSplat` struct layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSkinSplat {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub scale: [f32; 3],
    pub opacity: f32,
    pub rotation: [f32; 4],       // [x, y, z, w] normalized quaternion
    pub spectral: [f32; 8],
}

/// GPU-side joint transform — one per skeleton joint, updated each frame.
/// Must match WGSL `JointTransform` layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuJointTransform {
    pub skin_matrix: [[f32; 4]; 4], // world_transform * inverse_bind, column-major
}

/// Owns the wgpu resources for GPU skinning: input buffers, output buffer,
/// compute pipeline, and bind group.
pub struct SkinningCompute {
    pipeline: wgpu::ComputePipeline,
    base_splat_buffer: wgpu::Buffer,
    joint_binding_buffer: wgpu::Buffer,
    joint_transform_buffer: wgpu::Buffer,
    /// Output buffer: read by the render pass after dispatch.
    pub skinned_splat_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub splat_count: u32,
}

impl SkinningCompute {
    /// Create a new `SkinningCompute`.
    ///
    /// - `base_splats`: bind-pose splat data — uploaded once, never mutated.
    /// - `joint_bindings`: splat_idx → joint_idx mapping — uploaded once.
    /// - `joint_count`: number of joints (pre-allocates the joint transform buffer).
    pub fn new(
        device: &wgpu::Device,
        base_splats: &[GpuSkinSplat],
        joint_bindings: &[u32],
        joint_count: usize,
    ) -> Self {
        assert_eq!(base_splats.len(), joint_bindings.len(),
            "each splat must have exactly one joint binding");

        let splat_count = base_splats.len() as u32;

        // --- Buffers ---
        let base_splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinning_base_splats"),
            contents: bytemuck::cast_slice(base_splats),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let joint_binding_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinning_joint_bindings"),
            contents: bytemuck::cast_slice(joint_bindings),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let joint_transform_size =
            (joint_count * std::mem::size_of::<GpuJointTransform>()) as u64;
        let joint_transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_joint_transforms"),
            size: joint_transform_size.max(64), // minimum 64 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let skinned_splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_output_splats"),
            size: (base_splats.len() * std::mem::size_of::<GpuSkinSplat>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // --- Shader ---
        let shader_src = include_str!("skinning.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // --- Bind group layout ---
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinning_bgl"),
            entries: &[
                // binding 0: base_splats (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: joint_bindings (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: joint_transforms (read-only storage, written by CPU each frame)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: skinned_splats (read-write storage, output)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: base_splat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: joint_binding_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: joint_transform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: skinned_splat_buffer.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinning_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skinning_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_skin"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            base_splat_buffer,
            joint_binding_buffer,
            joint_transform_buffer,
            skinned_splat_buffer,
            bind_group,
            splat_count,
        }
    }

    /// Upload new joint transforms for the current frame.
    /// Call this before `dispatch()` in the same frame.
    pub fn update_joints(&self, queue: &wgpu::Queue, joint_transforms: &[GpuJointTransform]) {
        queue.write_buffer(
            &self.joint_transform_buffer,
            0,
            bytemuck::cast_slice(joint_transforms),
        );
    }

    /// Encode a compute dispatch into `encoder`.
    /// After submission, `skinned_splat_buffer` contains the skinned result.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = self.splat_count.div_ceil(64);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skinning_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
```

**Changes to `animation_driver.rs`:**

Add optional GPU skinning path. The CPU path (`tick`) continues to work unchanged for test/headless contexts; `tick_gpu` is the new GPU-accelerated path.

```rust
// Add to imports at top of animation_driver.rs:
use crate::gpu::skinning_compute::{GpuJointTransform, SkinningCompute};

// Add field to AnimationDriver:
pub struct AnimationDriver {
    // ... existing fields unchanged ...
    /// Optional GPU skinning compute pass. None = CPU path (tick).
    pub gpu_skinning: Option<SkinningCompute>,
}

// In AnimationDriver::new(), initialize to None:
//   gpu_skinning: None,

impl AnimationDriver {
    // ... existing methods unchanged ...

    /// GPU-accelerated tick: upload joint transforms and dispatch compute.
    ///
    /// Returns `true` if GPU skinning was dispatched.
    /// If `gpu_skinning` is None, returns `false` (caller should fall back to `tick()`).
    pub fn tick_gpu(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
    ) -> bool {
        let gpu = match &self.gpu_skinning {
            Some(g) => g,
            None => return false,
        };

        if self.animations.is_empty() {
            return false;
        }

        self.time += dt * self.speed;
        let anim = &self.animations[self.current_animation];
        if self.looping && anim.duration > 0.0 {
            self.time %= anim.duration;
        }

        let transforms = evaluate_animation(&self.skeleton, anim, self.time);
        let inverse_binds: Vec<glam::Mat4> = self.skeleton.joints.iter()
            .map(|j| j.inverse_bind_matrix)
            .collect();

        // Build skin matrices: world_transform * inverse_bind
        let joint_transforms: Vec<GpuJointTransform> = transforms.iter()
            .zip(inverse_binds.iter())
            .map(|(world_t, inv_bind)| {
                let skin = *world_t * *inv_bind;
                GpuJointTransform { skin_matrix: skin.to_cols_array_2d() }
            })
            .collect();

        gpu.update_joints(queue, &joint_transforms);
        gpu.dispatch(encoder);
        true
    }
}
```

**Changes to `crates/vox_render/src/gpu/mod.rs`:**

Add one line:
```rust
pub mod skinning_compute;
```

**Changes to `crates/vox_render/Cargo.toml`:**

Add under `[dev-dependencies]`:
```toml
naga = { version = "24", features = ["wgsl-in"] }
```

**Tests** — add to `skinning_compute.rs` inside `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn skinning_wgsl_shader_compiles() {
        let source = include_str!("skinning.wgsl");
        let result = naga::front::wgsl::parse_str(source);
        assert!(result.is_ok(), "WGSL parse error: {:?}", result.err());
    }
}
```

---

## Task 2: Spectral Audio Procedural Synthesis

**Files:**
- Create: `crates/vox_audio/src/spectral_synth.rs`
- Modify: `crates/vox_audio/src/lib.rs`

**Why:** Unreal 5's audio engine operates entirely in the time domain. Ochroma's spectral material system assigns 8 wavelength bands per surface — these can directly drive audio resonance frequencies, generating impact sounds that physically match the material. A glass surface has different spectral energy distribution than rock; this becomes audibly different in Ochroma.

**Design:** `SpectralSynth` maps Ochroma's 8 spectral bands (380–700nm, sampled as f16 in `GaussianSplat::spectral`) to 8 audio frequencies via a psychoacoustic mapping (short wavelengths = high-frequency audio). Impact sounds use exponential decay envelopes.

**Steps:**

- [ ] Create `crates/vox_audio/src/spectral_synth.rs`.
- [ ] Add `pub mod spectral_synth;` and re-exports to `crates/vox_audio/src/lib.rs`.
- [ ] Run `cargo test -p vox_audio spectral` — all tests pass.
- [ ] Commit: `feat(audio): spectral audio synthesis mapping material spectral bands to resonance frequencies`

**`spectral_synth.rs`** — create at `crates/vox_audio/src/spectral_synth.rs`:

```rust
//! Spectral audio synthesis.
//!
//! Maps Ochroma's 8 spectral bands (380nm–700nm) to audio resonance frequencies
//! via a physically-motivated psychoacoustic mapping:
//!
//!   Band 0 (380nm, blue-violet) → 8 kHz  (bright, glassy)
//!   Band 1 (428nm, violet)      → 4 kHz
//!   Band 2 (476nm, blue)        → 2 kHz
//!   Band 3 (524nm, cyan-green)  → 1 kHz  (mid)
//!   Band 4 (572nm, yellow)      → 500 Hz
//!   Band 5 (620nm, orange)      → 250 Hz
//!   Band 6 (652nm, red-orange)  → 125 Hz
//!   Band 7 (700nm, red)         →  80 Hz  (deep, rocky)
//!
//! This means a material with high blue spectral energy (e.g. glass) sounds
//! bright and glassy on impact; a material with high red energy (e.g. clay)
//! sounds low and dull — physically matching the material appearance.

/// Audio frequency assigned to each of the 8 spectral bands.
/// Index 0 = shortest wavelength (blue-violet); index 7 = longest (red).
const FREQ_MAP: [f32; 8] = [8000.0, 4000.0, 2000.0, 1000.0, 500.0, 250.0, 125.0, 80.0];

/// Generate a material impact sound from 8 spectral band weights.
///
/// `spectral_weights`: one weight per band, each in `[0.0, 1.0]`.
/// `duration_secs`: audio length in seconds (0.05–0.5 recommended).
/// `sample_rate`: Hz (typically 44100).
///
/// Returns a `Vec<f32>` of PCM samples normalized to `[-1.0, 1.0]`.
///
/// # Example
/// ```
/// let weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // pure red = low thud
/// let samples = vox_audio::spectral_synth::synthesize_impact(&weights, 0.2, 44100);
/// assert_eq!(samples.len(), 8820);
/// ```
pub fn synthesize_impact(spectral_weights: &[f32; 8], duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut output = vec![0.0f32; n_samples];

    for (band, &freq) in FREQ_MAP.iter().enumerate() {
        let weight = spectral_weights[band];
        if weight < 0.01 {
            continue;
        }
        // Exponential decay: e^(decay * t) where decay controls the falloff speed.
        // Faster decay for high-frequency bands (shorter attack, snappier).
        let decay_rate = -8.0 - (band as f32 * 2.0); // -8 to -22
        for (i, sample) in output.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            let envelope = (decay_rate * t).exp();
            *sample += weight * envelope * (2.0 * std::f32::consts::PI * freq * t).sin();
        }
    }

    // Normalize to [-1.0, 1.0]
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.001 {
        for s in &mut output {
            *s /= peak;
        }
    }

    output
}

/// Generate a spectral impact sound and write it to a temporary WAV file.
///
/// Returns the path of the created file.
/// The file persists until the caller removes it (or the OS cleans temp).
pub fn create_impact_wav(spectral_weights: &[f32; 8], duration_secs: f32) -> std::path::PathBuf {
    let samples = synthesize_impact(spectral_weights, duration_secs, 44100);
    let path = std::env::temp_dir().join("ochroma_impact.wav");
    crate::synth::save_wav(&samples, &path, 44100);
    path
}

/// Convenience: build `spectral_weights` from raw `[u16; 8]` f16 bits
/// (the format used in `GaussianSplat::spectral`) and synthesize.
pub fn synthesize_impact_from_splat_spectral(
    splat_spectral: &[u16; 8],
    duration_secs: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let weights: [f32; 8] = std::array::from_fn(|i| {
        half::f16::from_bits(splat_spectral[i]).to_f32().clamp(0.0, 1.0)
    });
    synthesize_impact(&weights, duration_secs, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_impact_returns_correct_length() {
        let weights = [0.5f32; 8];
        let samples = synthesize_impact(&weights, 0.1, 44100);
        assert_eq!(samples.len(), 4410, "0.1s × 44100Hz = 4410 samples");
    }

    #[test]
    fn synthesize_impact_is_normalized() {
        let weights = [1.0f32; 8];
        let samples = synthesize_impact(&weights, 0.1, 44100);
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 1.0 + 1e-5, "peak should be ≤ 1.0, got {}", peak);
    }

    #[test]
    fn synthesize_impact_all_zero_returns_silence() {
        let weights = [0.0f32; 8];
        let samples = synthesize_impact(&weights, 0.1, 44100);
        assert!(
            samples.iter().all(|&s| s == 0.0),
            "all-zero weights should produce silence"
        );
    }

    #[test]
    fn high_blue_weight_sounds_different_from_high_red_weight() {
        let mut blue_weights = [0.0f32; 8];
        blue_weights[0] = 1.0; // 8 kHz (blue-violet band)
        let mut red_weights = [0.0f32; 8];
        red_weights[7] = 1.0; // 80 Hz (red band)

        let blue = synthesize_impact(&blue_weights, 0.1, 44100);
        let red = synthesize_impact(&red_weights, 0.1, 44100);

        let diff: f32 = blue.iter().zip(red.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "different spectral weights should produce different audio, diff={}", diff);
    }

    #[test]
    fn create_impact_wav_creates_file() {
        let weights = [0.3f32; 8];
        let path = create_impact_wav(&weights, 0.05);
        assert!(path.exists(), "WAV file should exist at {:?}", path);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn synthesize_impact_from_splat_spectral_round_trips() {
        // f16 value 0.5 ≈ 0x3800
        let f16_half = half::f16::from_f32(0.5).to_bits();
        let splat_spectral = [f16_half; 8];
        let samples = synthesize_impact_from_splat_spectral(&splat_spectral, 0.1, 44100);
        assert_eq!(samples.len(), 4410);
    }
}
```

**Changes to `crates/vox_audio/src/lib.rs`:**

Add after the existing `pub mod synth;` line:
```rust
pub mod spectral_synth;
pub use spectral_synth::{synthesize_impact, create_impact_wav, synthesize_impact_from_splat_spectral};
```

Also add `half` to `vox_audio/Cargo.toml` if not already present (check first — it may already be there as a workspace dep):
```toml
half = { workspace = true }
```

**Wiring to physics collisions** (in `vox_app/src/bin/engine_runner.rs` or equivalent):

When Rapier reports a collision event, extract the spectral weights of one of the colliding entities' splat materials and call `create_impact_wav` then play via `AudioHandle`:

```rust
// In the collision event loop (where ContactForceEvent or CollisionEvent is processed):
for collision in collision_events.read() {
    if collision.is_started() {
        // Get spectral material from one of the colliding entities' splats
        // (use a default rock-like material if entity has no splat)
        let spectral_weights = entity_spectral_weights.get(collision.entity1())
            .cloned()
            .unwrap_or([0.15f32; 8]); // rock fallback
        let wav_path = vox_audio::create_impact_wav(&spectral_weights, 0.15);
        if let Some(handle) = &audio_handle {
            handle.play(wav_path.to_str().unwrap_or(""), 0.8, false);
        }
    }
}
```

Note: Wire this only if `engine_runner.rs` already has a Rapier collision event loop from Sprint 2. If Sprint 2 is not yet implemented, leave this as a stub comment and mark it `// TODO: wire after Sprint 2 physics`.

---

## Task 3: Real-time SDF Terrain Deformation

**Files:**
- Create: `crates/vox_terrain/src/deform.rs`
- Modify: `crates/vox_terrain/src/lib.rs`

**Why:** Unreal 5's Landscape is a static heightmap — runtime deformation requires Chaos Destruction and is limited to mesh fracture, not continuous SDF editing. Ochroma's SDF terrain can be carved or filled at runtime by modifying `TerrainVolume::data` and regenerating splats. This enables craters from explosions, player digging, tunnelling, etc.

**Confirmed `TerrainVolume` API (from `crates/vox_terrain/src/volume.rs`):**
- `get(x: usize, y: usize, z: usize) -> f32` — returns SDF value (negative = solid, positive = air, 1.0 outside bounds).
- `set(x: usize, y: usize, z: usize, value: f32)` — sets SDF value (bounds-checked, no-op if out-of-bounds).
- `world_to_voxel(wx, wy, wz) -> (usize, usize, usize)` — clamped to volume bounds.
- `voxel_to_world(x, y, z) -> [f32; 3]` — world position of voxel centre.
- `volume::sculpt::remove_sphere` — already exists and works (used in `add_cave`).
- `volume::sculpt::add_sphere` — already exists and works.
- No `mark_dirty()` method exists — after deformation, the caller regenerates splats by calling `volume_to_splats()` directly.

**Steps:**

- [ ] Read `crates/vox_terrain/src/lib.rs` to see current public exports.
- [ ] Create `crates/vox_terrain/src/deform.rs`.
- [ ] Add `pub mod deform;` and re-exports to `crates/vox_terrain/src/lib.rs`.
- [ ] Run `cargo test -p vox_terrain deform` — all tests pass.
- [ ] Wire `deform::carve_sphere` to `KeyG` + right-click in `engine_runner.rs`.
- [ ] Commit: `feat(terrain): real-time SDF deformation via carve_sphere + fill_sphere`

**`deform.rs`** — create at `crates/vox_terrain/src/deform.rs`:

```rust
//! Real-time SDF terrain deformation.
//!
//! Thin wrappers around `volume::sculpt` that present a world-space API
//! for runtime terrain carving and filling.
//!
//! After any deformation call, regenerate terrain splats by calling
//! `vox_terrain::volume_to_splats(&volume, &materials, seed)`.
//!
//! # Unreal 5 comparison
//! Unreal Landscape uses a static heightmap; Chaos Destruction provides
//! mesh fracture only.  Ochroma's SDF volume supports continuous carving,
//! filling, and tunnelling at runtime — any shape, any depth.

use crate::volume::{sculpt, TerrainVolume};

/// Carve a sphere-shaped hole into the SDF terrain.
///
/// Sets voxels inside `radius` from `center` to air (positive SDF).
/// Uses `sculpt::remove_sphere` internally: SDF at voxel = max(current, -(dist - radius)).
///
/// `center` is a world-space position `[x, y, z]`.
/// `radius` is in metres.
///
/// After calling, invoke `volume_to_splats()` to regenerate GPU-visible splats.
pub fn carve_sphere(volume: &mut TerrainVolume, center: [f32; 3], radius: f32) {
    sculpt::remove_sphere(volume, center, radius);
}

/// Fill a sphere-shaped region with solid terrain.
///
/// Sets voxels inside `radius` from `center` to solid (negative SDF).
/// Uses `sculpt::add_sphere` with `material` for the new solid region.
///
/// After calling, invoke `volume_to_splats()` to regenerate GPU-visible splats.
pub fn fill_sphere(volume: &mut TerrainVolume, center: [f32; 3], radius: f32, material: u8) {
    sculpt::add_sphere(volume, center, radius, material);
}

/// Carve a tunnel (capsule shape) between two world-space points.
///
/// Equivalent to `sculpt::add_cave`.
pub fn carve_tunnel(volume: &mut TerrainVolume, start: [f32; 3], end: [f32; 3], radius: f32) {
    sculpt::add_cave(volume, start, end, radius);
}

/// Apply a spherical explosion deformation at `center` with `radius`.
///
/// Equivalent to `carve_sphere` but named for game-logic clarity.
pub fn apply_explosion(volume: &mut TerrainVolume, center: [f32; 3], radius: f32) {
    carve_sphere(volume, center, radius);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::TerrainVolume;

    fn make_solid_volume() -> TerrainVolume {
        // 16×16×16 volume, 1m voxels, entirely solid (all SDF = -1.0)
        let mut vol = TerrainVolume::new(16, 16, 16, 1.0);
        for z in 0..16usize {
            for y in 0..16usize {
                for x in 0..16usize {
                    vol.set(x, y, z, -1.0);
                }
            }
        }
        vol
    }

    #[test]
    fn carve_sphere_makes_center_air() {
        let mut vol = make_solid_volume();
        // Volume origin is at (-8, -4, -8) for a 16^3 volume with voxel_size=1.
        // Place sphere at world (0, 0, 0) which is near the middle of the volume.
        carve_sphere(&mut vol, [0.0, 0.0, 0.0], 2.0);

        // After carving, the voxel at world (0,0,0) should be air (SDF > 0)
        // or at least greater than its pre-carve value of -1.0.
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let val = vol.get(cx, cy, cz);
        assert!(
            val > -1.0,
            "SDF at sphere center should increase after carving, got {}",
            val
        );
    }

    #[test]
    fn carve_sphere_does_not_affect_far_voxels() {
        let mut vol = make_solid_volume();
        carve_sphere(&mut vol, [0.0, 0.0, 0.0], 1.0);

        // Voxels far from the sphere should remain solid (SDF still -1.0 or similar).
        // World position (6, 0, 0) is ~6m away from sphere center (radius 1).
        let (fx, fy, fz) = vol.world_to_voxel(6.0, 0.0, 0.0);
        let val = vol.get(fx, fy, fz);
        assert!(
            val < 0.0,
            "Far voxels should remain solid after a small carve, got {}",
            val
        );
    }

    #[test]
    fn fill_sphere_makes_region_solid() {
        let mut vol = TerrainVolume::new(16, 16, 16, 1.0); // starts as all-air (SDF = 1.0)
        fill_sphere(&mut vol, [0.0, 0.0, 0.0], 2.0, 1);

        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let val = vol.get(cx, cy, cz);
        assert!(
            val < 0.0,
            "SDF at sphere center should be solid (< 0) after fill, got {}",
            val
        );
    }

    #[test]
    fn carve_then_fill_restores_solid() {
        let mut vol = make_solid_volume();
        let center = [0.0f32, 0.0, 0.0];
        carve_sphere(&mut vol, center, 2.0);

        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let after_carve = vol.get(cx, cy, cz);
        assert!(after_carve > -1.0, "carve should make center less solid");

        fill_sphere(&mut vol, center, 2.0, 0);
        let after_fill = vol.get(cx, cy, cz);
        assert!(
            after_fill < 0.0,
            "fill after carve should restore solid (< 0), got {}",
            after_fill
        );
    }

    #[test]
    fn apply_explosion_same_as_carve_sphere() {
        let mut vol_a = make_solid_volume();
        let mut vol_b = make_solid_volume();

        carve_sphere(&mut vol_a, [0.0, 0.0, 0.0], 3.0);
        apply_explosion(&mut vol_b, [0.0, 0.0, 0.0], 3.0);

        // Data should be identical
        assert_eq!(
            vol_a.data, vol_b.data,
            "apply_explosion should produce identical result to carve_sphere"
        );
    }

    #[test]
    fn carve_tunnel_connects_two_points() {
        let mut vol = make_solid_volume();
        // Carve from world (-3,0,0) to (3,0,0)
        carve_tunnel(&mut vol, [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0], 1.0);

        // Center of tunnel should be air
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let val = vol.get(cx, cy, cz);
        assert!(val > -1.0, "tunnel center should be air after carve_tunnel, got {}", val);
    }
}
```

**Changes to `crates/vox_terrain/src/lib.rs`:**

Add `pub mod deform;` and re-exports. Read the file first to find the right insertion point. Typical addition:
```rust
pub mod deform;
pub use deform::{carve_sphere, fill_sphere, carve_tunnel, apply_explosion};
```

**Wiring in `engine_runner.rs`:**

```rust
// KeyG held + left mouse click = carve sphere at raycast hit
KeyCode::KeyG if mouse_left_pressed => {
    if let Some(hit_pos) = physics_raycast(cam_pos, cam_forward, 100.0) {
        vox_terrain::carve_sphere(
            &mut terrain_volume,
            [hit_pos.x, hit_pos.y, hit_pos.z],
            2.0, // 2m radius
        );
        // Regenerate splats from the modified SDF
        let materials = vox_terrain::volume::default_volume_materials();
        terrain_splats = vox_terrain::volume::volume_to_splats(&terrain_volume, &materials, 42);
        println!("[ochroma] Carved terrain at {:?}", hit_pos);
    }
}
```

---

## Task 4: Rayon-Parallelized EWA Tile Rendering

**Files:**
- Modify: `crates/vox_render/src/spectra_render.rs`
- Modify: `crates/vox_render/Cargo.toml`

**Why:** `render_cpu_internal` in `spectra_render.rs` iterates tiles sequentially (lines 387–443: `for ty in 0..tiles_y { for tx in 0..tiles_x { ... } }`). At 1920×1080 with 16×16 tiles = 8160 tiles. Each tile is fully independent — no shared mutable state between tiles, only reads from `tile_gaussians` and `proj_map`. Rayon `into_par_iter()` gives ~8–16× speedup on an 8-core machine with zero algorithmic changes.

**Confirmed tile loop structure** (from `spectra_render.rs` lines 387–443):
- Outer loop: `for ty in 0..tiles_y { for tx in 0..tiles_x {`
- Per-tile: reads `tile_ranges[tile_id]`, iterates pixels `py`/`px`, reads `tile_gaussians[start..end]` and `proj_map[tg.gaussian_idx]`, writes to `image[pixel_idx..]`.
- The write target `image` is the only shared mutable state — parallelizing requires either disjoint slice writes or collecting per-tile buffers and merging.

**Parallelization strategy:** collect per-tile pixel buffers in parallel, then merge into `image` sequentially. This avoids any unsafe shared mutation.

**Steps:**

- [ ] Check `crates/vox_render/Cargo.toml` — confirm rayon is absent (it is: only in lockfile via other crates).
- [ ] Add `rayon = "1.10"` to `[dependencies]` in `crates/vox_render/Cargo.toml`.
- [ ] Refactor `render_cpu_internal` in `spectra_render.rs` as shown below.
- [ ] Run `cargo test -p vox_render` — existing tests pass unchanged.
- [ ] Commit: `perf(render): rayon-parallelized EWA tile rendering in spectra_render`

**Refactored `render_cpu_internal`:**

Replace the current sequential tile loop (lines 387–443) with:

```rust
use rayon::prelude::*;

// ... keep Steps 1–3 (project, assign to tiles, sort, build tile_ranges, proj_map) unchanged ...

// Step 4: Per-tile parallel rendering
// Each tile computes its pixel buffer independently, then we merge into image.
let tile_pixel_bufs: Vec<(usize, Vec<f32>)> = (0..num_tiles)
    .into_par_iter()
    .filter_map(|tile_id| {
        let (start, end) = tile_ranges[tile_id];
        if start == end {
            return None; // empty tile — skip
        }

        let tx = tile_id % tiles_x;
        let ty = tile_id / tiles_x;
        let px_start_x = tx * TILE_SIZE;
        let px_start_y = ty * TILE_SIZE;
        let px_end_x = (px_start_x + TILE_SIZE).min(w);
        let px_end_y = (px_start_y + TILE_SIZE).min(h);

        let tile_w = px_end_x - px_start_x;
        let tile_h = px_end_y - px_start_y;
        let mut tile_pixels = vec![0.0f32; tile_w * tile_h * 4];

        for (local_py, py) in (px_start_y..px_end_y).enumerate() {
            for (local_px, px) in (px_start_x..px_end_x).enumerate() {
                let local_idx = (local_py * tile_w + local_px) * 4;
                let mut transmittance = 1.0f32;
                let pxf = px as f32 + 0.5;
                let pyf = py as f32 + 0.5;

                for tg_idx in start..end {
                    if transmittance < TRANSMITTANCE_THRESHOLD {
                        break;
                    }

                    let tg = &tile_gaussians[tg_idx];
                    let pg = match proj_map[tg.gaussian_idx] {
                        Some(pg) => pg,
                        None => continue,
                    };

                    let dx = pxf - pg.screen_pos[0];
                    let dy = pyf - pg.screen_pos[1];
                    let power = -0.5
                        * (pg.conic[0] * dx * dx
                            + 2.0 * pg.conic[1] * dx * dy
                            + pg.conic[2] * dy * dy);

                    if power > 0.0 {
                        continue;
                    }

                    let alpha = (pg.opacity * power.exp()).min(0.99);
                    if alpha < ALPHA_THRESHOLD {
                        continue;
                    }

                    let weight = alpha * transmittance;
                    tile_pixels[local_idx] += weight * pg.color[0];
                    tile_pixels[local_idx + 1] += weight * pg.color[1];
                    tile_pixels[local_idx + 2] += weight * pg.color[2];
                    tile_pixels[local_idx + 3] += weight;

                    transmittance *= 1.0 - alpha;
                }
            }
        }

        Some((tile_id, tile_pixels))
    })
    .collect();

// Merge tile pixel buffers into the main image (sequential, O(pixels) total)
let mut image = vec![0.0f32; w * h * 4];
for (tile_id, tile_pixels) in tile_pixel_bufs {
    let tx = tile_id % tiles_x;
    let ty = tile_id / tiles_x;
    let px_start_x = tx * TILE_SIZE;
    let px_start_y = ty * TILE_SIZE;
    let px_end_x = (px_start_x + TILE_SIZE).min(w);
    let px_end_y = (px_start_y + TILE_SIZE).min(h);
    let tile_w = px_end_x - px_start_x;

    for (local_py, py) in (px_start_y..px_end_y).enumerate() {
        for (local_px, px) in (px_start_x..px_end_x).enumerate() {
            let local_idx = (local_py * tile_w + local_px) * 4;
            let global_idx = (py * w + px) * 4;
            image[global_idx]     = tile_pixels[local_idx];
            image[global_idx + 1] = tile_pixels[local_idx + 1];
            image[global_idx + 2] = tile_pixels[local_idx + 2];
            image[global_idx + 3] = tile_pixels[local_idx + 3];
        }
    }
}

image
```

**Important implementation note:** `proj_map` is `Vec<Option<&ProjectedGaussian>>` which holds references into `projected`. This is fine for `into_par_iter()` because:
- `projected` and `tile_gaussians` are read-only after Steps 1–3.
- Rayon requires `Send` for shared references — `&ProjectedGaussian` is `Send` because `ProjectedGaussian` is `Send`.
- If the compiler rejects the closure due to lifetime issues with `proj_map`, switch to indexing `projected` directly via `tg.gaussian_idx` and store `pg.index` in `TileGaussian` (it already does via `gaussian_idx`).

**Changes to `crates/vox_render/Cargo.toml`:**

Add under `[dependencies]`:
```toml
rayon = "1.10"
```

**Additional tests** (add to the `#[cfg(test)]` block in `spectra_render.rs`):

```rust
#[test]
fn parallel_render_is_deterministic() {
    use half::f16;
    let splat = GaussianSplat {
        position: [0.0, 0.0, 0.0],
        scale: [0.3, 0.3, 0.3],
        rotation: [0, 0, 0, 32767],
        opacity: 200,
        _pad: [0; 3],
        spectral: std::array::from_fn(|_| f16::from_f32(0.5).to_bits()),
    };
    let cam = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 64, 64);
    let r1 = render_with_spectra_u8(&[splat.clone()], &cam, 64, 64, &Illuminant::d65());
    let r2 = render_with_spectra_u8(&[splat], &cam, 64, 64, &Illuminant::d65());
    assert_eq!(r1, r2, "parallel render must be deterministic across two runs");
}

#[test]
fn parallel_render_matches_expected_pixel_count() {
    use half::f16;
    let splats: Vec<GaussianSplat> = (0..10)
        .map(|i| GaussianSplat {
            position: [i as f32 * 0.2 - 1.0, 0.0, 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: std::array::from_fn(|_| f16::from_f32(0.5).to_bits()),
        })
        .collect();
    let cam = make_camera(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, 128, 128);
    let result = render_with_spectra_u8(&splats, &cam, 128, 128, &Illuminant::d65());
    assert_eq!(result.len(), 128 * 128, "output must be width×height pixels");
}
```

**Performance validation** (run manually after implementing):
```bash
# Compare timing before and after rayon (run from workspace root)
time cargo run -p vox_render --example spectra_bench --release 2>&1 | tail -5
# Or use cargo bench if a bench target exists:
cargo bench -p vox_render -- spectra 2>&1 | tail -10
```

Expected: 4–16× speedup on multi-core hardware. On a dual-core CI box, expect 1.5–2×.

---

## Completion checklist

- [ ] Task 1 complete: `cargo test -p vox_render skinning` passes (WGSL validation + compute struct tests)
- [ ] Task 2 complete: `cargo test -p vox_audio spectral` passes (5 tests)
- [ ] Task 3 complete: `cargo test -p vox_terrain deform` passes (5 tests)
- [ ] Task 4 complete: `cargo test -p vox_render` passes (all existing tests + 2 new determinism tests)
- [ ] Full workspace: `cargo test` passes with no regressions
- [ ] Commits: 4 commits as specified per task

## Why this beats Unreal 5

| Feature | Unreal 5 | Ochroma (after Sprint 5) |
|---|---|---|
| GPU character skinning | Niagara + CPU morph targets for Gaussian equivalents | Native wgpu compute shader, one thread per splat, zero CPU allocation |
| Audio synthesis | Time-domain WAV samples; no material coupling | Frequency-domain spectral synthesis — material appearance drives audio |
| Terrain deformation | Static heightmap; Chaos Destruction = mesh fracture only | Real-time SDF carving/filling, arbitrary shapes, continuous at 60fps |
| Gaussian splatting render | Not supported | EWA tiled rendering, rayon-parallel, N×speedup with core count |
