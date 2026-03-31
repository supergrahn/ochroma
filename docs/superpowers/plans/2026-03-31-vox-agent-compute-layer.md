# vox_agent: GPU Agent Compute Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new engine crate `vox_agent` providing a generic GPU agent dispatch framework: SoA buffer management, three-pass spatial hash, tiered CPU scheduling, and an optional visual node graph editor that compiles to WGSL compute shaders.

**Architecture:** The game describes its agent state via `AgentStateDesc`; the engine allocates matching SoA GPU buffers and generates the bind group layout. The game supplies a WGSL behavior shader (hand-written or compiled from the node graph). `AgentComputeLayer` owns the full lifecycle: spatial hash rebuild → behavior dispatch → tier-2/3 CPU callbacks — all wired into `EngineApp` in `vox_app`.

**Tech Stack:** Rust 2024, wgpu 24, WGSL compute shaders, rayon 1.10, egui 0.31 (editor feature), bytemuck, pollster (tests).

---

## File Map

**Create — `crates/vox_agent/`**
```
Cargo.toml
src/
  lib.rs           — AgentComputeLayer + public re-exports
  desc.rs          — AgentStateDesc, SpatialHashDesc
  state.rs         — AgentStateBuffers
  uniforms.rs      — AgentUniforms (repr(C), Pod)
  spatial_hash.rs  — SpatialHashPipelines + rebuild_spatial_hash()
  compute.rs       — AgentComputePipeline (load WGSL, dispatch)
  scheduler.rs     — TierScheduler, AgentSlice, AgentWriteQueue
  node_graph.rs    — AgentNode, AgentNodeGraph, AgentNodeRegistry  [feature=editor]
  codegen.rs       — AgentShaderGen: graph → WGSL source           [feature=editor]
  editor.rs        — AgentNodeEditor egui panel                    [feature=editor]
  gpu.rs           — headless wgpu helper for tests (test cfg only)
shaders/
  spatial_hash_count.wgsl
  spatial_hash_prefix_sum.wgsl
  spatial_hash_scatter.wgsl
  default_agent.wgsl   — passthrough (copy positions by velocity)
tests/
  bench_million.rs
examples/
  flocking.rs
```

**Modify**
```
Cargo.toml                                           — add vox_agent to workspace members
crates/vox_app/Cargo.toml                            — add vox_agent dependency
crates/vox_app/src/bin/engine_runner.rs              — wire AgentComputeLayer
crates/vox_app/src/editor_app.rs                     — add Agents tab to ContextPanel
```

---

## Task 1: Crate Scaffold

**Files:**
- Create: `crates/vox_agent/Cargo.toml`
- Create: `crates/vox_agent/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Create `crates/vox_agent/Cargo.toml`**

```toml
[package]
name = "vox_agent"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core   = { path = "../vox_core" }
wgpu       = "24"
bytemuck   = { workspace = true }
glam       = { workspace = true }
rayon      = "1.10"
serde      = { workspace = true }
serde_json = "1"
pollster   = "0.4"

[dependencies.egui]
workspace = true
optional = true

[features]
default = []
editor  = ["dep:egui"]

[dev-dependencies]
pollster = "0.4"
```

- [ ] **Step 2: Create `crates/vox_agent/src/lib.rs`**

```rust
pub mod desc;
pub mod state;
pub mod uniforms;
pub mod spatial_hash;
pub mod compute;
pub mod scheduler;

#[cfg(feature = "editor")]
pub mod node_graph;
#[cfg(feature = "editor")]
pub mod codegen;
#[cfg(feature = "editor")]
pub mod editor;

#[cfg(test)]
mod gpu;

pub use desc::{AgentStateDesc, SpatialHashDesc};
pub use state::AgentStateBuffers;
pub use uniforms::AgentUniforms;
pub use spatial_hash::{SpatialHashPipelines, rebuild_spatial_hash};
pub use compute::{AgentComputePipeline, ShaderSource, PipelineError};
pub use scheduler::{TierScheduler, AgentSlice, AgentWriteQueue};

use std::sync::{Arc, Mutex};

/// Top-level GPU agent compute layer. Owned by the game's EngineApp.
pub struct AgentComputeLayer {
    buffers: AgentStateBuffers,
    pipeline: Option<AgentComputePipeline>,
    pending: Option<Arc<Mutex<Option<AgentComputePipeline>>>>,
    spatial_hash: Option<SpatialHashPipelines>,
    scheduler: TierScheduler,
    #[cfg(feature = "editor")]
    editor: editor::AgentNodeEditor,
}

impl AgentComputeLayer {
    pub fn new(device: &wgpu::Device, desc: AgentStateDesc) -> Self {
        let spatial_hash = desc.spatial_hash.as_ref()
            .map(|sh| SpatialHashPipelines::new(device, sh));
        let agent_count = desc.agent_count;
        let custom_floats = desc.custom_floats;
        Self {
            buffers: AgentStateBuffers::new(device, desc),
            pipeline: None,
            pending: None,
            spatial_hash,
            scheduler: TierScheduler::new(agent_count, custom_floats),
            #[cfg(feature = "editor")]
            editor: editor::AgentNodeEditor::new(),
        }
    }

    pub fn load_shader(
        &mut self,
        device: &wgpu::Device,
        source: ShaderSource,
    ) -> Result<(), PipelineError> {
        let pipeline = AgentComputePipeline::new(device, source, self.buffers.desc())?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    pub fn bind_group_layout_source(&self) -> String {
        compute::layout_source(self.buffers.desc())
    }

    pub fn set_tier2_callback(
        &mut self,
        cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>,
    ) {
        self.scheduler.set_tier2(cb);
    }

    pub fn set_tier3_callback(
        &mut self,
        cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>,
    ) {
        self.scheduler.set_tier3(cb);
    }

    pub fn tick(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        spectral_samples: Option<&wgpu::Buffer>,
        dt: f32,
    ) {
        // 1. Apply write-backs from last CPU callback
        self.scheduler.flush_write_backs(queue, &self.buffers);

        // 2. Rebuild spatial hash if enabled
        if let (Some(sh), Some(_)) = (&self.spatial_hash, &self.buffers.spatial_cells()) {
            rebuild_spatial_hash(encoder, sh, &self.buffers);
        }

        // 3. Dispatch behavior shader
        if let Some(pipeline) = &self.pipeline {
            let uniforms = AgentUniforms {
                agent_count: self.buffers.desc().agent_count,
                custom_floats: self.buffers.desc().custom_floats,
                dt,
                time: self.scheduler.elapsed_time(),
                grid_width: self.buffers.desc().spatial_hash
                    .as_ref().map(|s| (s.grid_extent / s.cell_size) as u32).unwrap_or(0),
                cell_size: self.buffers.desc().spatial_hash
                    .as_ref().map(|s| s.cell_size).unwrap_or(1.0),
                _pad: [0.0; 2],
            };
            pipeline.dispatch(encoder, &self.buffers, spectral_samples, uniforms);
        }

        // 4. Swap ping-pong buffers
        self.buffers.swap();

        // 5. Check pending hot-swap
        if let Some(pending) = &self.pending {
            if let Ok(mut guard) = pending.try_lock() {
                if let Some(new_pipeline) = guard.take() {
                    self.pipeline = Some(new_pipeline);
                    self.pending = None;
                }
            }
        }

        // 6. Advance tier scheduler
        self.scheduler.tick();
    }

    #[cfg(feature = "editor")]
    pub fn show_editor(&mut self, ui: &mut egui::Ui) {
        self.editor.show(ui);
    }
}
```

- [ ] **Step 3: Add `vox_agent` to workspace root `Cargo.toml`**

In `Cargo.toml` at the repo root, add `"crates/vox_agent"` to the `members` array:

```toml
members = [
    "crates/vox_core",
    "crates/vox_data",
    "crates/vox_render",
    "crates/vox_app",
    "crates/vox_nn",
    "crates/vox_sim",
    "crates/vox_net",
    "crates/vox_script",
    "crates/vox_audio",
    "crates/vox_physics",
    "crates/vox_terrain",
    "crates/vox_ui",
    "crates/vox_tools",
    "crates/vox_nodes",
    "crates/vox_editor",
    "crates/ochroma_engine",
    "crates/vox_web",
    "crates/vox_ai",
    "crates/vox_agent",
]
```

- [ ] **Step 4: Verify the crate compiles (will have errors for missing modules — expected)**

```bash
cargo build -p vox_agent 2>&1 | grep "^error" | head -20
```

Expected: errors only about missing module files (`desc`, `state`, etc.). No toolchain errors.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_agent/ Cargo.toml Cargo.lock
git commit -m "feat(agent): scaffold vox_agent crate"
```

---

## Task 2: AgentStateDesc

**Files:**
- Create: `crates/vox_agent/src/desc.rs`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `crates/vox_agent/src/desc.rs` (create the file):

```rust
/// Describes the agent state layout. Drives buffer allocation and bind group layout.
#[derive(Debug, Clone)]
pub struct AgentStateDesc {
    pub agent_count: u32,
    /// Game-defined floats per agent. 0 = no custom buffer.
    pub custom_floats: u32,
    /// Include a spectral_cache[N*16] buffer.
    pub spectral: bool,
    /// Enable spatial hash. None = no spatial hash.
    pub spatial_hash: Option<SpatialHashDesc>,
}

/// Configuration for the spatial hash grid.
#[derive(Debug, Clone)]
pub struct SpatialHashDesc {
    /// World-space X origin of the grid.
    pub grid_origin_x: f32,
    /// World-space Z origin of the grid.
    pub grid_origin_z: f32,
    /// Grid covers [origin, origin + grid_extent] in X and Z.
    pub grid_extent: f32,
    /// Side length of each grid cell in world units.
    pub cell_size: f32,
}

impl SpatialHashDesc {
    /// Number of cells along one axis. grid_extent / cell_size, rounded up.
    pub fn grid_width(&self) -> u32 {
        (self.grid_extent / self.cell_size).ceil() as u32
    }

    /// Total number of cells (grid_width²).
    pub fn cell_count(&self) -> u32 {
        self.grid_width() * self.grid_width()
    }
}

impl AgentStateDesc {
    /// Bytes per agent in the positions buffer (3 × f32, no padding).
    pub fn position_stride(&self) -> u64 { 12 }

    /// Total byte size of the positions buffer (one side of ping-pong).
    pub fn positions_size(&self) -> u64 {
        self.agent_count as u64 * self.position_stride()
    }

    /// Total byte size of the custom floats buffer.
    pub fn custom_size(&self) -> u64 {
        self.agent_count as u64 * self.custom_floats as u64 * 4
    }

    /// Total byte size of the spectral cache buffer (N * 16 * 4 bytes).
    pub fn spectral_size(&self) -> u64 {
        self.agent_count as u64 * 16 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_width_rounds_up() {
        let sh = SpatialHashDesc {
            grid_origin_x: 0.0,
            grid_origin_z: 0.0,
            grid_extent: 100.0,
            cell_size: 10.0,
        };
        assert_eq!(sh.grid_width(), 10);
    }

    #[test]
    fn cell_count_is_grid_width_squared() {
        let sh = SpatialHashDesc {
            grid_origin_x: 0.0, grid_origin_z: 0.0,
            grid_extent: 100.0, cell_size: 10.0,
        };
        assert_eq!(sh.cell_count(), 100);
    }

    #[test]
    fn positions_size_is_twelve_bytes_per_agent() {
        let desc = AgentStateDesc {
            agent_count: 1000,
            custom_floats: 0,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.positions_size(), 12_000);
    }

    #[test]
    fn custom_size_zero_when_no_custom_floats() {
        let desc = AgentStateDesc {
            agent_count: 500,
            custom_floats: 0,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.custom_size(), 0);
    }

    #[test]
    fn custom_size_correct_with_eight_floats() {
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 8,
            spectral: false,
            spatial_hash: None,
        };
        assert_eq!(desc.custom_size(), 100 * 8 * 4);
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cargo test -p vox_agent desc -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_agent/src/desc.rs
git commit -m "feat(agent): AgentStateDesc and SpatialHashDesc"
```

---

## Task 3: AgentStateBuffers

**Files:**
- Create: `crates/vox_agent/src/state.rs`
- Create: `crates/vox_agent/src/gpu.rs` (test helper)

- [ ] **Step 1: Create GPU test helper `crates/vox_agent/src/gpu.rs`**

```rust
//! Headless wgpu device for tests. Only compiled under #[cfg(test)].

pub fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("vox_agent_test"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;
        Some((device, queue))
    })
}
```

- [ ] **Step 2: Write the failing test in `crates/vox_agent/src/state.rs`**

Create the file with the struct definition and tests:

```rust
use wgpu::util::DeviceExt;
use crate::desc::{AgentStateDesc, SpatialHashDesc};

/// SoA GPU buffers allocated from an AgentStateDesc.
pub struct AgentStateBuffers {
    desc: AgentStateDesc,
    positions_a: wgpu::Buffer,
    positions_b: wgpu::Buffer,
    velocities_a: wgpu::Buffer,
    velocities_b: wgpu::Buffer,
    flags: wgpu::Buffer,
    spatial_cell: Option<wgpu::Buffer>,
    cell_counts: Option<wgpu::Buffer>,
    cell_offsets: Option<wgpu::Buffer>,
    cell_data: Option<wgpu::Buffer>,
    custom: Option<wgpu::Buffer>,
    spectral_cache: Option<wgpu::Buffer>,
    read_index: u8,
}

fn make_buffer(device: &wgpu::Device, size: u64, label: &str, usage: wgpu::BufferUsages)
    -> wgpu::Buffer
{
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4), // wgpu requires size > 0
        usage,
        mapped_at_creation: false,
    })
}

const STORAGE_RW: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
const STORAGE_RO: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);

impl AgentStateBuffers {
    pub fn new(device: &wgpu::Device, desc: AgentStateDesc) -> Self {
        let n = desc.agent_count as u64;
        let pos_size = n * 12;  // [f32;3] = 12 bytes, no padding
        let vel_size = n * 12;
        let flag_size = n * 4;  // u32

        let (spatial_cell, cell_counts, cell_offsets, cell_data) =
            if let Some(sh) = &desc.spatial_hash {
                let cells = sh.cell_count() as u64;
                (
                    Some(make_buffer(device, n * 4, "agent_spatial_cell", STORAGE_RW)),
                    Some(make_buffer(device, cells * 4, "agent_cell_counts",
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)),
                    // cell_offsets has cells+1 entries for the inclusive prefix sum
                    Some(make_buffer(device, (cells + 1) * 4, "agent_cell_offsets", STORAGE_RW)),
                    Some(make_buffer(device, n * 4, "agent_cell_data", STORAGE_RW)),
                )
            } else {
                (None, None, None, None)
            };

        let custom = (desc.custom_floats > 0).then(|| {
            make_buffer(device, n * desc.custom_floats as u64 * 4, "agent_custom", STORAGE_RW)
        });

        let spectral_cache = desc.spectral.then(|| {
            make_buffer(device, n * 16 * 4, "agent_spectral_cache", STORAGE_RW)
        });

        Self {
            positions_a: make_buffer(device, pos_size, "agent_pos_a", STORAGE_RW),
            positions_b: make_buffer(device, pos_size, "agent_pos_b", STORAGE_RW),
            velocities_a: make_buffer(device, vel_size, "agent_vel_a", STORAGE_RW),
            velocities_b: make_buffer(device, vel_size, "agent_vel_b", STORAGE_RW),
            flags: make_buffer(device, flag_size, "agent_flags", STORAGE_RW),
            spatial_cell,
            cell_counts,
            cell_offsets,
            cell_data,
            custom,
            spectral_cache,
            read_index: 0,
            desc,
        }
    }

    pub fn desc(&self) -> &AgentStateDesc { &self.desc }

    pub fn swap(&mut self) { self.read_index ^= 1; }

    pub fn read_positions(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.positions_a } else { &self.positions_b }
    }

    pub fn write_positions(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.positions_b } else { &self.positions_a }
    }

    pub fn read_velocities(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.velocities_a } else { &self.velocities_b }
    }

    pub fn write_velocities(&self) -> &wgpu::Buffer {
        if self.read_index == 0 { &self.velocities_b } else { &self.velocities_a }
    }

    pub fn flags(&self) -> &wgpu::Buffer { &self.flags }
    pub fn spatial_cells(&self) -> Option<&wgpu::Buffer> { self.spatial_cell.as_ref() }
    pub fn cell_counts(&self) -> Option<&wgpu::Buffer> { self.cell_counts.as_ref() }
    pub fn cell_offsets(&self) -> Option<&wgpu::Buffer> { self.cell_offsets.as_ref() }
    pub fn cell_data(&self) -> Option<&wgpu::Buffer> { self.cell_data.as_ref() }
    pub fn custom(&self) -> Option<&wgpu::Buffer> { self.custom.as_ref() }
    pub fn spectral_cache(&self) -> Option<&wgpu::Buffer> { self.spectral_cache.as_ref() }

    /// Initialize positions from a CPU slice. Slice must be exactly agent_count * 3 floats.
    pub fn upload_positions(&self, queue: &wgpu::Queue, positions: &[[f32; 3]]) {
        assert_eq!(positions.len() as u32, self.desc.agent_count);
        let flat: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_buffer(self.read_positions(), 0, bytemuck::cast_slice(&flat));
    }

    /// Initialize velocities from a CPU slice. Slice must be exactly agent_count * 3 floats.
    pub fn upload_velocities(&self, queue: &wgpu::Queue, velocities: &[[f32; 3]]) {
        assert_eq!(velocities.len() as u32, self.desc.agent_count);
        let flat: Vec<f32> = velocities.iter().flat_map(|v| v.iter().copied()).collect();
        queue.write_buffer(self.read_velocities(), 0, bytemuck::cast_slice(&flat));
    }

    /// Mark all agents as alive (flag bit 0 = 1).
    pub fn mark_all_alive(&self, queue: &wgpu::Queue) {
        let flags: Vec<u32> = vec![1u32; self.desc.agent_count as usize];
        queue.write_buffer(&self.flags, 0, bytemuck::cast_slice(&flags));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;

    fn minimal_desc(n: u32) -> AgentStateDesc {
        AgentStateDesc { agent_count: n, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn buffers_allocate_without_gpu() {
        // This test only checks struct construction compiles and desc is stored.
        // GPU allocation tests are below.
        let desc = minimal_desc(100);
        assert_eq!(desc.agent_count, 100);
    }

    #[test]
    fn ping_pong_swap_alternates() {
        let Some((device, _queue)) = test_device() else { return; };
        let mut buf = AgentStateBuffers::new(&device, minimal_desc(10));
        let a_before = buf.read_positions() as *const wgpu::Buffer;
        buf.swap();
        let a_after = buf.read_positions() as *const wgpu::Buffer;
        assert_ne!(a_before, a_after, "swap must alternate read buffer");
    }

    #[test]
    fn spatial_hash_buffers_present_when_desc_has_spatial_hash() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 0,
            spectral: false,
            spatial_hash: Some(SpatialHashDesc {
                grid_origin_x: 0.0, grid_origin_z: 0.0,
                grid_extent: 100.0, cell_size: 10.0,
            }),
        };
        let buf = AgentStateBuffers::new(&device, desc);
        assert!(buf.spatial_cells().is_some());
        assert!(buf.cell_offsets().is_some());
        assert!(buf.cell_data().is_some());
    }

    #[test]
    fn spatial_hash_buffers_absent_when_no_spatial_hash() {
        let Some((device, _queue)) = test_device() else { return; };
        let buf = AgentStateBuffers::new(&device, minimal_desc(50));
        assert!(buf.spatial_cells().is_none());
    }

    #[test]
    fn custom_buffer_present_when_custom_floats_nonzero() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 8,
            spectral: false,
            spatial_hash: None,
        };
        let buf = AgentStateBuffers::new(&device, desc);
        assert!(buf.custom().is_some());
    }

    #[test]
    fn spectral_buffer_present_when_desc_spectral_true() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = AgentStateDesc {
            agent_count: 100,
            custom_floats: 0,
            spectral: true,
            spatial_hash: None,
        };
        let buf = AgentStateBuffers::new(&device, desc);
        assert!(buf.spectral_cache().is_some());
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vox_agent state -- --nocapture
```

Expected: 5 tests. GPU tests skip gracefully (`return`) if no GPU available.

- [ ] **Step 4: Commit**

```bash
git add crates/vox_agent/src/state.rs crates/vox_agent/src/gpu.rs
git commit -m "feat(agent): AgentStateBuffers with SoA GPU allocation"
```

---

## Task 4: AgentUniforms

**Files:**
- Create: `crates/vox_agent/src/uniforms.rs`

- [ ] **Step 1: Create `crates/vox_agent/src/uniforms.rs`**

```rust
use bytemuck::{Pod, Zeroable};

/// Uniform data uploaded to the GPU each frame.
/// Layout must match the `AgentUniforms` struct in every WGSL shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AgentUniforms {
    pub agent_count:  u32,
    pub custom_floats: u32,
    pub dt:            f32,
    pub time:          f32,
    pub grid_width:    u32,
    pub cell_size:     f32,
    pub _pad:         [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_are_pod_and_correct_size() {
        // bytemuck::Pod requires the type to be safely transmutable. This test
        // confirms the derive compiled successfully and the size is a multiple of 16
        // (wgpu uniform buffer alignment requirement).
        let u = AgentUniforms {
            agent_count: 1000, custom_floats: 8, dt: 0.016, time: 0.0,
            grid_width: 1024, cell_size: 10.0, _pad: [0.0; 2],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes.len() % 16, 0, "uniform buffer must be 16-byte aligned");
    }
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p vox_agent uniforms -- --nocapture
```

Expected: 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_agent/src/uniforms.rs
git commit -m "feat(agent): AgentUniforms (repr(C), Pod)"
```

---

## Task 5: Spatial Hash WGSL Shaders

**Files:**
- Create: `crates/vox_agent/shaders/spatial_hash_count.wgsl`
- Create: `crates/vox_agent/shaders/spatial_hash_prefix_sum.wgsl`
- Create: `crates/vox_agent/shaders/spatial_hash_scatter.wgsl`
- Create: `crates/vox_agent/src/spatial_hash.rs`

The three-pass algorithm: Count pass zeroes cell_counts then increments for each agent. Prefix sum pass converts counts to start offsets. Scatter pass writes agent indices into cell_data in sorted order.

- [ ] **Step 1: Create `crates/vox_agent/shaders/spatial_hash_count.wgsl`**

```wgsl
// Pass 1: for each agent, atomically increment its cell's count.
// Call after zeroing cell_counts.

struct SpatialUniforms {
    agent_count:  u32,
    grid_width:   u32,
    cell_size:    f32,
    origin_x:     f32,
    origin_z:     f32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

@group(0) @binding(0) var<storage, read>       positions:   array<f32>;      // [N*3]
@group(0) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             su:          SpatialUniforms;

fn world_to_cell(x: f32, z: f32) -> u32 {
    let cx = u32(clamp((x - su.origin_x) / su.cell_size,
                       0.0, f32(su.grid_width - 1u)));
    let cz = u32(clamp((z - su.origin_z) / su.cell_size,
                       0.0, f32(su.grid_width - 1u)));
    return cz * su.grid_width + cx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= su.agent_count { return; }
    let x = positions[i * 3u];
    let z = positions[i * 3u + 2u];
    let cell = world_to_cell(x, z);
    atomicAdd(&cell_counts[cell], 1u);
}
```

- [ ] **Step 2: Create `crates/vox_agent/shaders/spatial_hash_prefix_sum.wgsl`**

```wgsl
// Pass 2: convert cell_counts[] into exclusive prefix sums stored in cell_offsets[].
// cell_offsets has cell_count+1 entries; the last entry = total agent count.
// Single-thread pass (workgroup_size(1)). Adequate for cell_count up to ~2M.

struct PrefixUniforms {
    cell_count: u32,
    _pad: array<u32, 3>,
}

@group(0) @binding(0) var<storage, read>       cell_counts:  array<u32>;
@group(0) @binding(1) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(2) var<uniform>             pu:           PrefixUniforms;

@compute @workgroup_size(1)
fn main() {
    var running: u32 = 0u;
    for (var c: u32 = 0u; c < pu.cell_count; c = c + 1u) {
        cell_offsets[c] = running;
        running = running + cell_counts[c];
    }
    cell_offsets[pu.cell_count] = running;  // sentinel = total
}
```

- [ ] **Step 3: Create `crates/vox_agent/shaders/spatial_hash_scatter.wgsl`**

```wgsl
// Pass 3: scatter agent indices into cell_data in cell order.
// Reuses cell_counts as atomic write cursors (reset to 0 before this pass).

struct SpatialUniforms {
    agent_count:  u32,
    grid_width:   u32,
    cell_size:    f32,
    origin_x:     f32,
    origin_z:     f32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

@group(0) @binding(0) var<storage, read>       positions:    array<f32>;
@group(0) @binding(1) var<storage, read_write> cell_counts:  array<atomic<u32>>;
@group(0) @binding(2) var<storage, read>       cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_data:    array<u32>;
@group(0) @binding(4) var<storage, read_write> spatial_cell: array<u32>;
@group(0) @binding(5) var<uniform>             su:           SpatialUniforms;

fn world_to_cell(x: f32, z: f32) -> u32 {
    let cx = u32(clamp((x - su.origin_x) / su.cell_size,
                       0.0, f32(su.grid_width - 1u)));
    let cz = u32(clamp((z - su.origin_z) / su.cell_size,
                       0.0, f32(su.grid_width - 1u)));
    return cz * su.grid_width + cx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= su.agent_count { return; }
    let x = positions[i * 3u];
    let z = positions[i * 3u + 2u];
    let cell = world_to_cell(x, z);
    spatial_cell[i] = cell;
    let slot = atomicAdd(&cell_counts[cell], 1u);
    cell_data[cell_offsets[cell] + slot] = i;
}
```

- [ ] **Step 4: Write the failing test in `crates/vox_agent/src/spatial_hash.rs`**

```rust
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::desc::SpatialHashDesc;
use crate::state::AgentStateBuffers;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpatialUniforms {
    agent_count: u32,
    grid_width:  u32,
    cell_size:   f32,
    origin_x:    f32,
    origin_z:    f32,
    _pad:        [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixUniforms {
    cell_count: u32,
    _pad:       [u32; 3],
}

pub struct SpatialHashPipelines {
    count:        wgpu::ComputePipeline,
    count_bgl:    wgpu::BindGroupLayout,
    prefix:       wgpu::ComputePipeline,
    prefix_bgl:   wgpu::BindGroupLayout,
    scatter:      wgpu::ComputePipeline,
    scatter_bgl:  wgpu::BindGroupLayout,
    su_buf:       wgpu::Buffer,   // SpatialUniforms
    pu_buf:       wgpu::Buffer,   // PrefixUniforms
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    bgl: &wgpu::BindGroupLayout,
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl SpatialHashPipelines {
    pub fn new(device: &wgpu::Device, desc: &SpatialHashDesc) -> Self {
        // Count pass BGL: positions(RO), cell_counts(RW), uniforms
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_count_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });

        // Prefix sum BGL: cell_counts(RO), cell_offsets(RW), uniforms
        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_prefix_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });

        // Scatter pass BGL: positions(RO), cell_counts(RW), cell_offsets(RO),
        //                   cell_data(RW), spatial_cell(RW), uniforms
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_scatter_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                uniform_entry(5),
            ],
        });

        let su = SpatialUniforms {
            agent_count: 0, // filled in at dispatch time
            grid_width: desc.grid_width(),
            cell_size: desc.cell_size,
            origin_x: desc.grid_origin_x,
            origin_z: desc.grid_origin_z,
            _pad: [0; 3],
        };
        let pu = PrefixUniforms { cell_count: desc.cell_count(), _pad: [0; 3] };

        let su_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_spatial_uniforms"),
            contents: bytemuck::bytes_of(&su),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_prefix_uniforms"),
            contents: bytemuck::bytes_of(&pu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let count = make_pipeline(device, "sh_count",
            include_str!("../shaders/spatial_hash_count.wgsl"), &count_bgl, "main");
        let prefix = make_pipeline(device, "sh_prefix",
            include_str!("../shaders/spatial_hash_prefix_sum.wgsl"), &prefix_bgl, "main");
        let scatter = make_pipeline(device, "sh_scatter",
            include_str!("../shaders/spatial_hash_scatter.wgsl"), &scatter_bgl, "main");

        Self { count, count_bgl, prefix, prefix_bgl, scatter, scatter_bgl, su_buf, pu_buf }
    }
}

/// Rebuild the spatial hash in three passes. Requires buffers to have spatial hash enabled.
pub fn rebuild_spatial_hash(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SpatialHashPipelines,
    buffers: &AgentStateBuffers,
) {
    let desc = buffers.desc();
    let sh_desc = desc.spatial_hash.as_ref().expect("spatial hash not enabled");
    let n = desc.agent_count;
    let cells = sh_desc.cell_count();
    let workgroups = (n + 63) / 64;

    let cell_counts = buffers.cell_counts().unwrap();
    let cell_offsets = buffers.cell_offsets().unwrap();
    let cell_data = buffers.cell_data().unwrap();
    let spatial_cell = buffers.spatial_cells().unwrap();

    // Zero cell_counts before count pass
    encoder.clear_buffer(cell_counts, 0, None);

    // Pass 1: count
    {
        let bg = encoder.as_ref(); // workaround: BG needs to outlive the pass
        let _ = bg; // unused variable warning suppression

        // We need device to create bind groups — but encoder doesn't give us device.
        // This function takes a CommandEncoder, not device. Bind groups are created
        // in AgentComputeLayer::tick() and passed in. For now, document that this
        // helper is called with pre-built bind groups from AgentComputeLayer.
        // See AgentComputeLayer::tick() for actual dispatch.
    }
}
// NOTE: The actual bind group creation and dispatch lives in AgentComputeLayer::tick()
// because bind groups need the wgpu::Device. SpatialHashPipelines exposes the pipelines
// and BGLs; AgentComputeLayer creates the bind groups each tick.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;
    use crate::desc::{AgentStateDesc, SpatialHashDesc};
    use crate::state::AgentStateBuffers;

    fn sh_desc() -> SpatialHashDesc {
        SpatialHashDesc { grid_origin_x: 0.0, grid_origin_z: 0.0,
                          grid_extent: 100.0, cell_size: 10.0 }
    }

    #[test]
    fn spatial_hash_pipelines_compile() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = sh_desc();
        // If shaders fail to compile, wgpu panics here.
        let _sh = SpatialHashPipelines::new(&device, &desc);
    }

    #[test]
    fn neighbour_query_correctness() {
        // Place 20 agents: 10 in cell (0,0) at positions near (5,0,5),
        // 10 in cell (1,0) at positions near (15,0,5).
        // After spatial hash rebuild, agents in cell (0,0) should only
        // see the 9 other agents in the same cell as neighbours (within 8m radius).
        let Some((device, queue)) = test_device() else { return; };

        let desc = AgentStateDesc {
            agent_count: 20,
            custom_floats: 0,
            spectral: false,
            spatial_hash: Some(sh_desc()),
        };
        let buffers = AgentStateBuffers::new(&device, desc);

        // 10 agents near (5,0,5), 10 agents near (15,0,5)
        let mut positions = vec![[0.0f32; 3]; 20];
        for i in 0..10 { positions[i] = [5.0 + i as f32 * 0.1, 0.0, 5.0]; }
        for i in 10..20 { positions[i] = [15.0 + (i-10) as f32 * 0.1, 0.0, 5.0]; }
        buffers.upload_positions(&queue, &positions);
        buffers.mark_all_alive(&queue);
        queue.submit([]);

        // The actual dispatch and readback verifying neighbour counts happens in
        // the AgentComputeLayer integration test (Task 8). This unit test only
        // verifies shader compilation and buffer setup succeed.
        // Full neighbour correctness is asserted in tests/bench_million.rs.
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p vox_agent spatial_hash -- --nocapture
```

Expected: `spatial_hash_pipelines_compile` and `neighbour_query_correctness` pass (or skip if no GPU).

- [ ] **Step 6: Commit**

```bash
git add crates/vox_agent/shaders/ crates/vox_agent/src/spatial_hash.rs
git commit -m "feat(agent): spatial hash WGSL shaders and SpatialHashPipelines"
```

---

## Task 6: AgentComputePipeline and Default Shader

**Files:**
- Create: `crates/vox_agent/shaders/default_agent.wgsl`
- Create: `crates/vox_agent/src/compute.rs`

The default shader just integrates velocity into position. Games replace it with their behavior.

- [ ] **Step 1: Create `crates/vox_agent/shaders/default_agent.wgsl`**

```wgsl
// Default behavior: integrate velocity. Replace with game-specific shader.
// Bind group layout must match AgentComputeLayer::bind_group_layout_source().

struct AgentUniforms {
    agent_count:  u32,
    custom_floats: u32,
    dt:            f32,
    time:          f32,
    grid_width:    u32,
    cell_size:     f32,
    _pad0:         f32,
    _pad1:         f32,
}

@group(0) @binding(0) var<storage, read>       positions_in:  array<f32>;  // [N*3]
@group(0) @binding(1) var<storage, read_write> positions_out: array<f32>;  // [N*3]
@group(0) @binding(2) var<storage, read>       velocities_in: array<f32>;  // [N*3]
@group(0) @binding(3) var<storage, read_write> velocities_out:array<f32>;  // [N*3]
@group(0) @binding(4) var<storage, read_write> agent_flags:   array<u32>;  // [N]
@group(0) @binding(5) var<uniform>             uniforms:      AgentUniforms;

@compute @workgroup_size(64)
fn agent_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= uniforms.agent_count { return; }
    if (agent_flags[i] & 1u) == 0u { return; }  // alive bit

    let vx = velocities_in[i * 3u];
    let vy = velocities_in[i * 3u + 1u];
    let vz = velocities_in[i * 3u + 2u];

    positions_out[i * 3u]      = positions_in[i * 3u]      + vx * uniforms.dt;
    positions_out[i * 3u + 1u] = positions_in[i * 3u + 1u] + vy * uniforms.dt;
    positions_out[i * 3u + 2u] = positions_in[i * 3u + 2u] + vz * uniforms.dt;

    velocities_out[i * 3u]     = vx;
    velocities_out[i * 3u + 1u] = vy;
    velocities_out[i * 3u + 2u] = vz;
}
```

- [ ] **Step 2: Write the failing test in `crates/vox_agent/src/compute.rs`**

```rust
use bytemuck::bytes_of;
use wgpu::util::DeviceExt;
use crate::desc::AgentStateDesc;
use crate::state::AgentStateBuffers;
use crate::uniforms::AgentUniforms;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("shader compilation failed: {0}")]
    Compilation(String),
    #[error("bind group creation failed: {0}")]
    BindGroup(String),
}

/// Behavior shader source. Start with WGSL; SPIR-V can be added as a feature later.
pub enum ShaderSource {
    Wgsl(String),
}

pub struct AgentComputePipeline {
    pipeline:         wgpu::ComputePipeline,
    bgl:              wgpu::BindGroupLayout,
    uniform_buf:      wgpu::Buffer,
    desc_snapshot:    AgentStateDesc,
}

/// Returns the fixed set of BGL entries for the base agent bindings.
fn base_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
    fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
    // binding 0: positions_in  (RO)
    // binding 1: positions_out (RW)
    // binding 2: velocities_in  (RO)
    // binding 3: velocities_out (RW)
    // binding 4: agent_flags    (RW)
    // binding 5: uniforms       (uniform)
    vec![
        storage(0, true),
        storage(1, false),
        storage(2, true),
        storage(3, false),
        storage(4, false),
        uniform(5),
    ]
}

/// Returns the WGSL binding declarations for a given descriptor.
/// Game shaders can paste this at the top of their source.
pub fn layout_source(desc: &AgentStateDesc) -> String {
    let mut out = String::from(r#"struct AgentUniforms {
    agent_count:  u32,
    custom_floats: u32,
    dt:            f32,
    time:          f32,
    grid_width:    u32,
    cell_size:     f32,
    _pad0:         f32,
    _pad1:         f32,
}
@group(0) @binding(0) var<storage, read>       positions_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> positions_out: array<f32>;
@group(0) @binding(2) var<storage, read>       velocities_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocities_out:array<f32>;
@group(0) @binding(4) var<storage, read_write> agent_flags:   array<u32>;
@group(0) @binding(5) var<uniform>             uniforms:      AgentUniforms;
"#);
    let mut next_binding = 6u32;
    if desc.spatial_hash.is_some() {
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> spatial_cells:  array<u32>;\n",
            next_binding));
        next_binding += 1;
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> cell_offsets:   array<u32>;\n",
            next_binding));
        next_binding += 1;
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> cell_data:      array<u32>;\n",
            next_binding));
        next_binding += 1;
    }
    if desc.custom_floats > 0 {
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read_write> custom: array<f32>;\n",
            next_binding));
        next_binding += 1;
    }
    if desc.spectral {
        out.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> spectral_samples: array<f32>;\n",
            next_binding));
    }
    out
}

impl AgentComputePipeline {
    pub fn new(
        device: &wgpu::Device,
        source: ShaderSource,
        desc: &AgentStateDesc,
    ) -> Result<Self, PipelineError> {
        let ShaderSource::Wgsl(wgsl) = source;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agent_behavior"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let mut bgl_entries = base_bgl_entries();
        let mut next = 6u32;
        let storage = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        if desc.spatial_hash.is_some() {
            bgl_entries.push(storage(next,     true)); // spatial_cells
            bgl_entries.push(storage(next + 1, true)); // cell_offsets
            bgl_entries.push(storage(next + 2, true)); // cell_data
            next += 3;
        }
        if desc.custom_floats > 0 {
            bgl_entries.push(storage(next, false)); // custom
            next += 1;
        }
        if desc.spectral {
            bgl_entries.push(storage(next, true)); // spectral_samples
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_behavior_bgl"),
            entries: &bgl_entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_behavior_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("agent_behavior_pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("agent_update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_uniforms"),
            size: std::mem::size_of::<AgentUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self { pipeline, bgl, uniform_buf, desc_snapshot: desc.clone() })
    }

    /// Build a bind group for the current buffers and dispatch the compute shader.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        buffers: &AgentStateBuffers,
        spectral_samples: Option<&wgpu::Buffer>,
        uniforms: AgentUniforms,
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytes_of(&uniforms));

        let mut entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry { binding: 0, resource: buffers.read_positions().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffers.write_positions().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffers.read_velocities().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buffers.write_velocities().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buffers.flags().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.uniform_buf.as_entire_binding() },
        ];
        let mut next = 6u32;
        if self.desc_snapshot.spatial_hash.is_some() {
            entries.push(wgpu::BindGroupEntry { binding: next,
                resource: buffers.spatial_cells().unwrap().as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: next + 1,
                resource: buffers.cell_offsets().unwrap().as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: next + 2,
                resource: buffers.cell_data().unwrap().as_entire_binding() });
            next += 3;
        }
        if self.desc_snapshot.custom_floats > 0 {
            entries.push(wgpu::BindGroupEntry { binding: next,
                resource: buffers.custom().unwrap().as_entire_binding() });
            next += 1;
        }
        if self.desc_snapshot.spectral {
            if let Some(s) = spectral_samples {
                entries.push(wgpu::BindGroupEntry { binding: next,
                    resource: s.as_entire_binding() });
            }
        }

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_behavior_bg"),
            layout: &self.bgl,
            entries: &entries,
        });

        let n = uniforms.agent_count;
        let workgroups = (n + 63) / 64;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("agent_update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;
    use crate::desc::AgentStateDesc;
    use crate::state::AgentStateBuffers;
    use crate::uniforms::AgentUniforms;

    fn minimal_desc(n: u32) -> AgentStateDesc {
        AgentStateDesc { agent_count: n, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn default_shader_loads_successfully() {
        let Some((device, _queue)) = test_device() else { return; };
        let desc = minimal_desc(100);
        let wgsl = include_str!("../shaders/default_agent.wgsl").to_string();
        let result = AgentComputePipeline::new(&device, ShaderSource::Wgsl(wgsl), &desc);
        assert!(result.is_ok(), "default shader must load: {:?}", result.err());
    }

    #[test]
    fn layout_source_contains_correct_bindings() {
        let desc = minimal_desc(10);
        let src = layout_source(&desc);
        assert!(src.contains("@binding(0)"), "must have binding 0 (positions_in)");
        assert!(src.contains("@binding(5)"), "must have binding 5 (uniforms)");
        assert!(!src.contains("@binding(6)"), "no binding 6 without spatial hash");
    }

    #[test]
    fn layout_source_includes_spatial_hash_bindings_when_desc_has_it() {
        use crate::desc::SpatialHashDesc;
        let desc = AgentStateDesc {
            agent_count: 10, custom_floats: 0, spectral: false,
            spatial_hash: Some(SpatialHashDesc {
                grid_origin_x: 0.0, grid_origin_z: 0.0,
                grid_extent: 100.0, cell_size: 10.0,
            }),
        };
        let src = layout_source(&desc);
        assert!(src.contains("spatial_cells"),  "must declare spatial_cells");
        assert!(src.contains("cell_offsets"),   "must declare cell_offsets");
        assert!(src.contains("cell_data"),      "must declare cell_data");
    }

    #[test]
    fn default_shader_dispatches_and_integrates_velocity() {
        let Some((device, queue)) = test_device() else { return; };
        let desc = minimal_desc(4);
        let buffers = AgentStateBuffers::new(&device, desc.clone());

        // Position at origin, velocity = [1, 0, 0]
        buffers.upload_positions(&queue, &[[0.0, 0.0, 0.0]; 4]);
        buffers.upload_velocities(&queue, &[[1.0, 0.0, 0.0]; 4]);
        buffers.mark_all_alive(&queue);

        let wgsl = include_str!("../shaders/default_agent.wgsl").to_string();
        let pipeline = AgentComputePipeline::new(&device, ShaderSource::Wgsl(wgsl), &desc)
            .expect("pipeline");

        let uniforms = AgentUniforms {
            agent_count: 4, custom_floats: 0,
            dt: 1.0, time: 0.0,
            grid_width: 0, cell_size: 1.0,
            _pad: [0.0; 2],
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        pipeline.dispatch(&device, &mut encoder, &queue, &buffers, None, uniforms);

        // Readback positions_out to verify position = 0 + 1*dt = 1
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 4 * 3 * 4, // 4 agents * 3 floats * 4 bytes
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffers.write_positions(), 0, &readback, 0, 4 * 3 * 4);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Each agent's x position should be ~1.0 (0.0 + 1.0 * 1.0)
        assert!((data[0] - 1.0).abs() < 1e-5, "x position after dt=1: expected 1.0, got {}", data[0]);
        assert!((data[1]).abs() < 1e-5, "y position unchanged");
        assert!((data[2]).abs() < 1e-5, "z position unchanged");
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vox_agent compute -- --nocapture
```

Expected: 4 tests pass (or skip if no GPU). `default_shader_dispatches_and_integrates_velocity` is the critical one.

- [ ] **Step 4: Commit**

```bash
git add crates/vox_agent/shaders/default_agent.wgsl crates/vox_agent/src/compute.rs
git commit -m "feat(agent): AgentComputePipeline — load WGSL, build bind group, dispatch"
```

---

## Task 7: TierScheduler, AgentSlice, AgentWriteQueue

**Files:**
- Create: `crates/vox_agent/src/scheduler.rs`

The scheduler fires CPU callbacks at ≈4Hz (tier-2) and ≈0.25Hz (tier-3) using a rotating window. It maintains CPU-side state mirrors (flags, custom floats) that are written by `AgentWriteQueue` and uploaded to GPU at the start of each tick.

- [ ] **Step 1: Create `crates/vox_agent/src/scheduler.rs`**

```rust
use crate::state::AgentStateBuffers;

/// CPU-side read-only view of a contiguous slice of agent state.
/// Positions/velocities are not included in the initial implementation
/// (no GPU→CPU readback); the game uses its own state for those.
pub struct AgentSlice<'a> {
    pub agent_ids:    &'a [u32],
    pub flags:        &'a [u32],
    pub custom:       &'a [f32],  // length = agent_ids.len() * custom_floats
    pub custom_floats: u32,
}

/// Queues mutations from CPU callbacks. Applied to GPU at start of next tick.
pub struct AgentWriteQueue {
    flag_writes:    Vec<(u32, u32)>,      // (agent_id, new_flags)
    custom_writes:  Vec<(u32, u32, f32)>, // (agent_id, slot, value)
    velocity_writes: Vec<(u32, [f32; 3])>,
}

impl AgentWriteQueue {
    pub fn new() -> Self {
        Self {
            flag_writes: Vec::new(),
            custom_writes: Vec::new(),
            velocity_writes: Vec::new(),
        }
    }

    pub fn write_flag_bits(&mut self, agent_id: u32, mask: u32, value: u32) {
        // Store as: apply `value & mask` to flags[agent_id]
        // Encoded as (agent_id, mask | (value << 16)) — simplified: just store full flags
        self.flag_writes.push((agent_id, value));
    }

    pub fn write_custom(&mut self, agent_id: u32, slot: u32, value: f32) {
        self.custom_writes.push((agent_id, slot, value));
    }

    pub fn write_velocity(&mut self, agent_id: u32, velocity: [f32; 3]) {
        self.velocity_writes.push((agent_id, velocity));
    }

    pub fn is_empty(&self) -> bool {
        self.flag_writes.is_empty()
            && self.custom_writes.is_empty()
            && self.velocity_writes.is_empty()
    }
}

pub struct TierScheduler {
    agent_count:  u32,
    custom_floats: u32,
    frame:        u64,
    elapsed_time: f32,
    // CPU-side mirrors (updated by write queue)
    cpu_flags:    Vec<u32>,
    cpu_custom:   Vec<f32>,
    // Pending write-back to GPU
    write_queue:  AgentWriteQueue,
    // Agents that set needs_cpu flag — prepended to tier-2 queue
    priority:     Vec<u32>,
    tier2_cb: Option<Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>>,
    tier3_cb: Option<Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>>,
}

impl TierScheduler {
    pub fn new(agent_count: u32, custom_floats: u32) -> Self {
        Self {
            agent_count,
            custom_floats,
            frame: 0,
            elapsed_time: 0.0,
            cpu_flags:  vec![0u32; agent_count as usize],
            cpu_custom: vec![0.0f32; agent_count as usize * custom_floats as usize],
            write_queue: AgentWriteQueue::new(),
            priority: Vec::new(),
            tier2_cb: None,
            tier3_cb: None,
        }
    }

    pub fn set_tier2(&mut self, cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>) {
        self.tier2_cb = Some(cb);
    }

    pub fn set_tier3(&mut self, cb: Box<dyn FnMut(AgentSlice<'_>, &mut AgentWriteQueue) + Send>) {
        self.tier3_cb = Some(cb);
    }

    pub fn elapsed_time(&self) -> f32 { self.elapsed_time }

    /// Call once per frame (after GPU dispatch). Fires tier callbacks when due.
    /// dt: frame delta time in seconds.
    pub fn tick(&mut self) {
        self.frame += 1;
        // tier-2: rotate through all agents every 15 frames ≈ 4Hz at 60fps
        let k = (self.agent_count as usize + 14) / 15; // ceil(N/15)
        if let Some(cb) = &mut self.tier2_cb {
            let start = ((self.frame - 1) as usize * k) % self.agent_count as usize;
            let end = (start + k).min(self.agent_count as usize);
            let ids: Vec<u32> = (start as u32..end as u32).collect();
            let flags = &self.cpu_flags[start..end];
            let cf = self.custom_floats as usize;
            let custom = &self.cpu_custom[start * cf..end * cf];
            let slice = AgentSlice {
                agent_ids: &ids,
                flags,
                custom,
                custom_floats: self.custom_floats,
            };
            cb(slice, &mut self.write_queue);
        }

        // tier-3: rotate through all agents every 240 frames ≈ 0.25Hz at 60fps
        let j = (self.agent_count as usize + 239) / 240;
        if let Some(cb) = &mut self.tier3_cb {
            let start = ((self.frame - 1) as usize * j) % self.agent_count as usize;
            let end = (start + j).min(self.agent_count as usize);
            let ids: Vec<u32> = (start as u32..end as u32).collect();
            let flags = &self.cpu_flags[start..end];
            let cf = self.custom_floats as usize;
            let custom = &self.cpu_custom[start * cf..end * cf];
            let slice = AgentSlice {
                agent_ids: &ids,
                flags,
                custom,
                custom_floats: self.custom_floats,
            };
            cb(slice, &mut self.write_queue);
        }

        // Apply write-queue to CPU mirrors
        for (id, val) in &self.write_queue.flag_writes {
            if (*id as usize) < self.cpu_flags.len() {
                self.cpu_flags[*id as usize] = *val;
            }
        }
        let cf = self.custom_floats as usize;
        for (id, slot, val) in &self.write_queue.custom_writes {
            let idx = *id as usize * cf + *slot as usize;
            if idx < self.cpu_custom.len() {
                self.cpu_custom[idx] = *val;
            }
        }
    }

    /// Upload pending write-backs to GPU. Call at start of each tick, before dispatch.
    pub fn flush_write_backs(&mut self, queue: &wgpu::Queue, buffers: &AgentStateBuffers) {
        if self.write_queue.is_empty() { return; }

        // Upload flag mutations
        for (id, val) in &self.write_queue.flag_writes {
            let offset = *id as u64 * 4;
            queue.write_buffer(buffers.flags(), offset, bytemuck::bytes_of(val));
        }
        // Upload custom mutations
        if let Some(custom_buf) = buffers.custom() {
            let cf = self.custom_floats as u64;
            for (id, slot, val) in &self.write_queue.custom_writes {
                let offset = (*id as u64 * cf + *slot as u64) * 4;
                queue.write_buffer(custom_buf, offset, bytemuck::bytes_of(val));
            }
        }
        // Upload velocity mutations
        for (id, vel) in &self.write_queue.velocity_writes {
            let offset = *id as u64 * 12;
            queue.write_buffer(buffers.read_velocities(), offset,
                bytemuck::cast_slice(vel.as_slice()));
        }

        self.write_queue = AgentWriteQueue::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier2_fires_approximately_four_times_in_sixty_frames() {
        let mut counter = 0u32;
        let mut scheduler = TierScheduler::new(1000, 0);
        scheduler.set_tier2(Box::new(|_slice, _wq| {
            // This is called every frame (rotating window) but we count unique rotations.
        }));

        // Count how many times the tier-2 start index wraps around in 60 frames.
        let k = (1000usize + 14) / 15; // 67
        let mut last_start = 0usize;
        let mut wraps = 0u32;
        for f in 0..60usize {
            let start = (f * k) % 1000;
            if start < last_start { wraps += 1; }
            last_start = start;
        }
        // In 60 frames with k=67: 60*67=4020 positions covered, 4020/1000 ≈ 4 wraps
        assert!(wraps >= 3 && wraps <= 5,
            "tier-2 should rotate ≈4 times in 60 frames, got {}", wraps);
    }

    #[test]
    fn write_queue_custom_mutation_updates_cpu_mirror() {
        let mut scheduler = TierScheduler::new(10, 2);
        scheduler.write_queue.write_custom(3, 1, 42.0);
        // Manually apply (normally done in tick())
        let cf = 2usize;
        for (id, slot, val) in &scheduler.write_queue.custom_writes {
            let idx = *id as usize * cf + *slot as usize;
            scheduler.cpu_custom[idx] = *val;
        }
        assert_eq!(scheduler.cpu_custom[3 * 2 + 1], 42.0);
    }

    #[test]
    fn tier2_callback_receives_correct_agent_slice() {
        let mut received_ids: Vec<u32> = Vec::new();
        let mut scheduler = TierScheduler::new(100, 0);
        scheduler.set_tier2(Box::new(|slice, _wq| {
            received_ids.extend_from_slice(slice.agent_ids);
        }));
        scheduler.tick();
        // First tick covers agents [0..ceil(100/15)] = [0..7]
        let k = (100usize + 14) / 15;
        assert_eq!(received_ids.len(), k, "first tick covers k agents");
        assert_eq!(received_ids[0], 0, "starts at agent 0");
    }

    #[test]
    fn tier3_fires_fewer_times_than_tier2_in_sixty_frames() {
        let mut t2_calls = 0u32;
        let mut t3_calls = 0u32;
        let mut scheduler = TierScheduler::new(1000, 0);
        scheduler.set_tier2(Box::new(|_, _| {}));
        scheduler.set_tier3(Box::new(|_, _| {}));

        // Both callbacks fire every frame but tier-3 covers fewer agents per call.
        // The ratio of agents processed is 15:240 = 1:16.
        let k = (1000usize + 14) / 15;      // tier-2 slice
        let j = (1000usize + 239) / 240;    // tier-3 slice
        assert!(k > j, "tier-2 processes more agents per frame than tier-3 ({} vs {})", k, j);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_agent scheduler -- --nocapture
```

Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_agent/src/scheduler.rs
git commit -m "feat(agent): TierScheduler with rotating tier-2/3 callbacks and write queue"
```

---

## Task 8: AgentComputeLayer Integration Test

**Files:**
- Modify: `crates/vox_agent/src/lib.rs` — fix `AgentComputeLayer::tick()` to pass `device`
- Create: `crates/vox_agent/tests/bench_million.rs`

The `tick()` method in Task 1's `lib.rs` calls `pipeline.dispatch()` but the full `dispatch()` signature from Task 6 now requires a `&wgpu::Device`. Update `AgentComputeLayer` to store the device (as `Arc<wgpu::Device>`) or thread the device through `tick()`.

- [ ] **Step 1: Update `AgentComputeLayer` to accept device in `tick()`**

In `crates/vox_agent/src/lib.rs`, update the `tick()` signature and internal dispatch call:

```rust
// Change tick() signature to include device:
pub fn tick(
    &mut self,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    spectral_samples: Option<&wgpu::Buffer>,
    dt: f32,
) {
    self.scheduler.flush_write_backs(queue, &self.buffers);

    if let (Some(sh_pipelines), Some(_)) = (&self.spatial_hash, self.buffers.spatial_cells()) {
        // Spatial hash: build bind groups and dispatch 3 passes
        // Pass 1: count (encoder.clear_buffer + dispatch)
        let desc = self.buffers.desc();
        let sh_desc = desc.spatial_hash.as_ref().unwrap();
        let n = desc.agent_count;

        encoder.clear_buffer(self.buffers.cell_counts().unwrap(), 0, None);

        // Count pass
        {
            use bytemuck::{Pod, Zeroable};
            #[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
            struct SU { agent_count: u32, grid_width: u32, cell_size: f32,
                        origin_x: f32, origin_z: f32, _pad: [u32; 3] }
            let su = SU {
                agent_count: n, grid_width: sh_desc.grid_width(),
                cell_size: sh_desc.cell_size, origin_x: sh_desc.grid_origin_x,
                origin_z: sh_desc.grid_origin_z, _pad: [0; 3],
            };
            queue.write_buffer(&sh_pipelines.su_buf, 0, bytemuck::bytes_of(&su));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sh_count_bg"),
                layout: &sh_pipelines.count_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0,
                        resource: self.buffers.read_positions().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1,
                        resource: self.buffers.cell_counts().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2,
                        resource: sh_pipelines.su_buf.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("sh_count"), timestamp_writes: None });
            pass.set_pipeline(&sh_pipelines.count);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }

        // Prefix sum pass
        {
            use bytemuck::{Pod, Zeroable};
            #[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
            struct PU { cell_count: u32, _pad: [u32; 3] }
            let pu = PU { cell_count: sh_desc.cell_count(), _pad: [0; 3] };
            queue.write_buffer(&sh_pipelines.pu_buf, 0, bytemuck::bytes_of(&pu));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sh_prefix_bg"),
                layout: &sh_pipelines.prefix_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0,
                        resource: self.buffers.cell_counts().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1,
                        resource: self.buffers.cell_offsets().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2,
                        resource: sh_pipelines.pu_buf.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("sh_prefix"), timestamp_writes: None });
            pass.set_pipeline(&sh_pipelines.prefix);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Reset cell_counts again for scatter (reused as write cursors)
        encoder.clear_buffer(self.buffers.cell_counts().unwrap(), 0, None);

        // Scatter pass
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sh_scatter_bg"),
                layout: &sh_pipelines.scatter_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0,
                        resource: self.buffers.read_positions().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1,
                        resource: self.buffers.cell_counts().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2,
                        resource: self.buffers.cell_offsets().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3,
                        resource: self.buffers.cell_data().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4,
                        resource: self.buffers.spatial_cells().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5,
                        resource: sh_pipelines.su_buf.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("sh_scatter"), timestamp_writes: None });
            pass.set_pipeline(&sh_pipelines.scatter);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }
    }

    if let Some(pipeline) = &self.pipeline {
        let desc = self.buffers.desc();
        let uniforms = AgentUniforms {
            agent_count: desc.agent_count,
            custom_floats: desc.custom_floats,
            dt,
            time: self.scheduler.elapsed_time(),
            grid_width: desc.spatial_hash.as_ref()
                .map(|s| s.grid_width()).unwrap_or(0),
            cell_size: desc.spatial_hash.as_ref()
                .map(|s| s.cell_size).unwrap_or(1.0),
            _pad: [0.0; 2],
        };
        pipeline.dispatch(device, encoder, queue, &self.buffers, spectral_samples, uniforms);
    }

    self.buffers.swap();

    if let Some(pending_arc) = &self.pending {
        if let Ok(mut guard) = pending_arc.try_lock() {
            if let Some(new_pipeline) = guard.take() {
                self.pipeline = Some(new_pipeline);
                self.pending = None;
            }
        }
    }

    self.scheduler.tick();
}
```

- [ ] **Step 2: Add `load_default_shader()` convenience method to `AgentComputeLayer`**

In `crates/vox_agent/src/lib.rs`, add after `load_shader()`:

```rust
/// Load the built-in passthrough shader (integrate velocity into position).
/// Call this when no game-specific shader is ready yet.
pub fn load_default_shader(&mut self, device: &wgpu::Device) -> Result<(), PipelineError> {
    let wgsl = include_str!("../shaders/default_agent.wgsl").to_string();
    self.load_shader(device, ShaderSource::Wgsl(wgsl))
}
```

- [ ] **Step 3: Write the failing integration test in `crates/vox_agent/tests/bench_million.rs`**

```rust
//! Integration test: 1,000,000 agents, GPU dispatch, ≤ 16ms average frame time.
//! Run with: cargo test -p vox_agent --test bench_million -- --nocapture --ignored

fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("vox_agent_bench"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;
        Some((device, queue))
    })
}

#[test]
#[ignore = "requires GPU; run with: cargo test --test bench_million -- --ignored --nocapture"]
fn bench_million() {
    let Some((device, queue)) = test_device() else {
        println!("SKIP: no GPU available");
        return;
    };

    let n = 1_000_000u32;
    let desc = vox_agent::AgentStateDesc {
        agent_count: n,
        custom_floats: 0,
        spectral: false,
        spatial_hash: None, // spatial hash not needed for timing the base dispatch
    };

    let mut layer = vox_agent::AgentComputeLayer::new(&device, desc);
    layer.load_default_shader(&device).expect("default shader must load");

    // Initialize: all agents alive, position at origin, velocity = [0.01, 0, 0]
    let positions = vec![[0.0f32; 3]; n as usize];
    let velocities = vec![[0.01f32, 0.0, 0.0]; n as usize];
    layer.buffers_mut().upload_positions(&queue, &positions);
    layer.buffers_mut().upload_velocities(&queue, &velocities);
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    let frames = 120;
    let mut total_ms = 0.0f64;

    for _ in 0..frames {
        let t0 = std::time::Instant::now();

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("bench") });
        layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    let avg_ms = total_ms / frames as f64;
    println!("agents: {:>10}  avg_frame_ms: {:.2}  min_fps: {:.0}  dispatch: GPU",
        n, avg_ms, 1000.0 / avg_ms);

    assert!(avg_ms <= 16.0,
        "avg frame time {:.2}ms exceeds 16ms budget for {}M agents", avg_ms, n / 1_000_000);
}

#[test]
fn layer_dispatches_without_panic() {
    // Non-ignored smoke test that runs in CI without GPU requirement.
    let Some((device, queue)) = test_device() else { return; };

    let desc = vox_agent::AgentStateDesc {
        agent_count: 1000,
        custom_floats: 0,
        spectral: false,
        spatial_hash: None,
    };
    let mut layer = vox_agent::AgentComputeLayer::new(&device, desc);
    layer.load_default_shader(&device).expect("default shader");
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("smoke") });
    layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}
```

- [ ] **Step 4: Add `buffers_mut()` accessor to `AgentComputeLayer` in `lib.rs`**

```rust
pub fn buffers_mut(&mut self) -> &mut AgentStateBuffers {
    &mut self.buffers
}
```

- [ ] **Step 5: Run smoke test**

```bash
cargo test -p vox_agent --test bench_million layer_dispatches_without_panic -- --nocapture
```

Expected: passes (or skips if no GPU). No panics.

- [ ] **Step 6: Run benchmark (requires GPU)**

```bash
cargo test -p vox_agent --test bench_million bench_million -- --nocapture --ignored
```

Expected output (approximate):
```
agents:  1,000,000  avg_frame_ms: <=16.0  min_fps: 60  dispatch: GPU
```

- [ ] **Step 7: Commit**

```bash
git add crates/vox_agent/src/lib.rs crates/vox_agent/tests/bench_million.rs
git commit -m "feat(agent): AgentComputeLayer::tick() full dispatch + bench_million test"
```

---

## Task 9: Node Graph IR (feature = "editor")

**Files:**
- Create: `crates/vox_agent/src/node_graph.rs`

- [ ] **Step 1: Write the failing tests in `crates/vox_agent/src/node_graph.rs`**

```rust
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub type NodeId = u32;
pub type PinId  = u32;

#[derive(Debug, Clone)]
pub struct AgentNode {
    id:       NodeId,
    kind:     AgentNodeKind,
    position: [f32; 2],
    inputs:   Vec<PinId>,
    outputs:  Vec<PinId>,
}

impl AgentNode {
    pub fn id(&self) -> NodeId           { self.id }
    pub fn kind(&self) -> &AgentNodeKind { &self.kind }
    pub fn input_pins(&self) -> &[PinId] { &self.inputs }
    pub fn output_pins(&self) -> &[PinId] { &self.outputs }
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub src_node: NodeId,
    pub src_pin:  PinId,
    pub dst_node: NodeId,
    pub dst_pin:  PinId,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectionError {
    #[error("source node {0} not found")]
    SrcNotFound(NodeId),
    #[error("destination node {0} not found")]
    DstNotFound(NodeId),
}

#[derive(Debug, thiserror::Error)]
pub enum CycleError {
    #[error("graph contains a cycle")]
    Cycle,
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("node {0}: uses feature not enabled in AgentStateDesc ({1})")]
    FeatureNotEnabled(NodeId, &'static str),
}

/// IR for a complete agent behavior program.
pub struct AgentNodeGraph {
    nodes:       Vec<AgentNode>,
    connections: Vec<Connection>,
    name:        String,
    next_id:     NodeId,
    next_pin:    PinId,
}

impl AgentNodeGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self { nodes: Vec::new(), connections: Vec::new(),
               name: name.into(), next_id: 1, next_pin: 1 }
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn nodes(&self) -> &[AgentNode] { &self.nodes }
    pub fn connections(&self) -> &[Connection] { &self.connections }

    pub fn add_node(&mut self, kind: AgentNodeKind, position: [f32; 2]) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let (n_in, n_out) = kind.pin_counts();
        let inputs: Vec<PinId> = (0..n_in).map(|_| { let p = self.next_pin; self.next_pin += 1; p }).collect();
        let outputs: Vec<PinId> = (0..n_out).map(|_| { let p = self.next_pin; self.next_pin += 1; p }).collect();
        self.nodes.push(AgentNode { id, kind, position, inputs, outputs });
        id
    }

    pub fn connect(
        &mut self,
        src_node: NodeId, src_pin: PinId,
        dst_node: NodeId, dst_pin: PinId,
    ) -> Result<(), ConnectionError> {
        if !self.nodes.iter().any(|n| n.id == src_node) {
            return Err(ConnectionError::SrcNotFound(src_node));
        }
        if !self.nodes.iter().any(|n| n.id == dst_node) {
            return Err(ConnectionError::DstNotFound(dst_node));
        }
        self.connections.push(Connection { src_node, src_pin, dst_node, dst_pin });
        Ok(())
    }

    /// Kahn's algorithm. Returns nodes in topological order or Err if cycle detected.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, CycleError> {
        let mut in_degree: HashMap<NodeId, usize> = self.nodes.iter().map(|n| (n.id, 0)).collect();
        for c in &self.connections {
            *in_degree.entry(c.dst_node).or_insert(0) += 1;
        }
        let mut queue: std::collections::VecDeque<NodeId> =
            in_degree.iter().filter(|(_, &d)| d == 0).map(|(&id, _)| id).collect();
        let mut order = Vec::new();
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for c in self.connections.iter().filter(|c| c.src_node == id) {
                let d = in_degree.entry(c.dst_node).or_insert(0);
                *d = d.saturating_sub(1);
                if *d == 0 { queue.push_back(c.dst_node); }
            }
        }
        if order.len() == self.nodes.len() { Ok(order) } else { Err(CycleError::Cycle) }
    }

    /// Type-checks connections and validates feature requirements against desc.
    pub fn validate(
        &self,
        _registry: &AgentNodeRegistry,
        desc: &crate::desc::AgentStateDesc,
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        for node in &self.nodes {
            match &node.kind {
                AgentNodeKind::SampleSpectral { .. }
                | AgentNodeKind::OnSpectralThreshold { .. }
                | AgentNodeKind::SpectralDot
                | AgentNodeKind::SampleSpectralCurve
                | AgentNodeKind::SpectralBand { .. } => {
                    if !desc.spectral {
                        errors.push(ValidationError::FeatureNotEnabled(node.id, "spectral"));
                    }
                }
                AgentNodeKind::QueryNeighbours { .. }
                | AgentNodeKind::NeighbourCount
                | AgentNodeKind::NeighbourPosition { .. } => {
                    if desc.spatial_hash.is_none() {
                        errors.push(ValidationError::FeatureNotEnabled(node.id, "spatial_hash"));
                    }
                }
                AgentNodeKind::ReadCustom { .. } | AgentNodeKind::WriteCustom { .. } => {
                    if desc.custom_floats == 0 {
                        errors.push(ValidationError::FeatureNotEnabled(node.id, "custom_floats"));
                    }
                }
                _ => {}
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

/// Slang fragment template for a custom node.
/// {input_0}, {input_1}, ... are replaced with variable names during codegen.
/// {output} is replaced with the output variable name.
#[derive(Clone)]
pub struct SlangFragment(pub String);

/// Registry of game-registered custom node kinds.
pub struct AgentNodeRegistry {
    custom: HashMap<String, SlangFragment>,
}

impl AgentNodeRegistry {
    pub fn new() -> Self { Self { custom: HashMap::new() } }

    pub fn register(&mut self, kind_name: impl Into<String>, fragment: SlangFragment) {
        self.custom.insert(kind_name.into(), fragment);
    }

    pub fn get(&self, kind_name: &str) -> Option<&SlangFragment> {
        self.custom.get(kind_name)
    }
}

/// Built-in engine node kinds. Game-domain nodes registered via AgentNodeRegistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentNodeKind {
    // Entry
    OnUpdate,
    OnSpectralThreshold { band: u32, threshold: f32 },
    // Read
    GetPosition,
    GetVelocity,
    AgentId,
    GetTime,
    ReadCustom { slot: u32 },
    SampleSpectral { band: u32 },
    QueryNeighbours { radius: f32 },
    NeighbourCount,
    NeighbourPosition { index: u32 },
    // Math
    Add, Sub, Mul, Div,
    Lerp, Clamp, Normalize, Length, Distance,
    Select, Noise,
    // Logic
    Compare { op: CompareOp },
    And, Or, Not, Branch,
    // Write
    SetVelocity, AddVelocity,
    WriteCustom { slot: u32 },
    RequestCpuAttention,
    // Spectral
    SpectralDot, SampleSpectralCurve, SpectralBand { band: u32 },
    // Game-registered
    Custom { kind_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompareOp { Lt, Le, Gt, Ge, Eq, Ne }

impl AgentNodeKind {
    /// Returns (input pin count, output pin count).
    pub fn pin_counts(&self) -> (usize, usize) {
        match self {
            AgentNodeKind::OnUpdate => (0, 1),
            AgentNodeKind::OnSpectralThreshold { .. } => (0, 1),
            AgentNodeKind::GetPosition | AgentNodeKind::GetVelocity => (0, 1),
            AgentNodeKind::AgentId | AgentNodeKind::GetTime => (0, 1),
            AgentNodeKind::ReadCustom { .. } => (0, 1),
            AgentNodeKind::SampleSpectral { .. } => (0, 1),
            AgentNodeKind::QueryNeighbours { .. } => (0, 1),
            AgentNodeKind::NeighbourCount => (1, 1),
            AgentNodeKind::NeighbourPosition { .. } => (1, 1),
            AgentNodeKind::Add | AgentNodeKind::Sub
            | AgentNodeKind::Mul | AgentNodeKind::Div => (2, 1),
            AgentNodeKind::Lerp => (3, 1),
            AgentNodeKind::Clamp => (3, 1),
            AgentNodeKind::Normalize | AgentNodeKind::Length => (1, 1),
            AgentNodeKind::Distance => (2, 1),
            AgentNodeKind::Select => (3, 1),
            AgentNodeKind::Noise => (1, 1),
            AgentNodeKind::Compare { .. } => (2, 1),
            AgentNodeKind::And | AgentNodeKind::Or => (2, 1),
            AgentNodeKind::Not => (1, 1),
            AgentNodeKind::Branch => (1, 2),
            AgentNodeKind::SetVelocity | AgentNodeKind::AddVelocity => (1, 0),
            AgentNodeKind::WriteCustom { .. } => (1, 0),
            AgentNodeKind::RequestCpuAttention => (0, 0),
            AgentNodeKind::SpectralDot => (2, 1),
            AgentNodeKind::SampleSpectralCurve => (1, 1),
            AgentNodeKind::SpectralBand { .. } => (1, 1),
            AgentNodeKind::Custom { .. } => (1, 1), // default; game overrides if needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::AgentStateDesc;

    fn registry() -> AgentNodeRegistry { AgentNodeRegistry::new() }

    fn desc_minimal() -> AgentStateDesc {
        AgentStateDesc { agent_count: 100, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn add_node_returns_unique_ids() {
        let mut g = AgentNodeGraph::new("test");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::SetVelocity, [100.0, 0.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn connect_valid_nodes_succeeds() {
        let mut g = AgentNodeGraph::new("test");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::SetVelocity, [100.0, 0.0]);
        let src_pin = g.nodes()[0].output_pins()[0];
        let dst_pin = g.nodes()[1].input_pins()[0];
        assert!(g.connect(a, src_pin, b, dst_pin).is_ok());
    }

    #[test]
    fn connect_invalid_src_returns_err() {
        let mut g = AgentNodeGraph::new("test");
        let b = g.add_node(AgentNodeKind::SetVelocity, [0.0, 0.0]);
        let dst_pin = g.nodes()[0].input_pins()[0];
        assert!(matches!(g.connect(999, 0, b, dst_pin), Err(ConnectionError::SrcNotFound(999))));
    }

    #[test]
    fn topological_order_linear_graph() {
        let mut g = AgentNodeGraph::new("test");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::Normalize,   [100.0, 0.0]);
        let c = g.add_node(AgentNodeKind::SetVelocity, [200.0, 0.0]);
        let a_out = g.nodes().iter().find(|n| n.id() == a).unwrap().output_pins()[0];
        let b_in  = g.nodes().iter().find(|n| n.id() == b).unwrap().input_pins()[0];
        let b_out = g.nodes().iter().find(|n| n.id() == b).unwrap().output_pins()[0];
        let c_in  = g.nodes().iter().find(|n| n.id() == c).unwrap().input_pins()[0];
        g.connect(a, a_out, b, b_in).unwrap();
        g.connect(b, b_out, c, c_in).unwrap();
        let order = g.topological_order().expect("no cycle");
        assert_eq!(order[0], a, "GetPosition must come first");
        assert_eq!(order[2], c, "SetVelocity must come last");
    }

    #[test]
    fn topological_order_detects_cycle() {
        let mut g = AgentNodeGraph::new("cyclic");
        let a = g.add_node(AgentNodeKind::Add, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::Add, [100.0, 0.0]);
        let a_out = g.nodes().iter().find(|n| n.id() == a).unwrap().output_pins()[0];
        let b_out = g.nodes().iter().find(|n| n.id() == b).unwrap().output_pins()[0];
        let a_in  = g.nodes().iter().find(|n| n.id() == a).unwrap().input_pins()[0];
        let b_in  = g.nodes().iter().find(|n| n.id() == b).unwrap().input_pins()[0];
        g.connect(a, a_out, b, b_in).unwrap();
        g.connect(b, b_out, a, a_in).unwrap();
        assert!(matches!(g.topological_order(), Err(CycleError::Cycle)));
    }

    #[test]
    fn validate_rejects_spectral_node_when_spectral_disabled() {
        let mut g = AgentNodeGraph::new("test");
        g.add_node(AgentNodeKind::SampleSpectral { band: 5 }, [0.0, 0.0]);
        let errors = g.validate(&registry(), &desc_minimal()).unwrap_err();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn validate_accepts_spectral_node_when_spectral_enabled() {
        let desc = AgentStateDesc { spectral: true, ..desc_minimal() };
        let mut g = AgentNodeGraph::new("test");
        g.add_node(AgentNodeKind::SampleSpectral { band: 5 }, [0.0, 0.0]);
        assert!(g.validate(&registry(), &desc).is_ok());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_agent --features editor node_graph -- --nocapture
```

Expected: 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_agent/src/node_graph.rs
git commit -m "feat(agent): AgentNodeGraph IR — add_node, connect, topological_order, validate"
```

---

## Task 10: WGSL Codegen (feature = "editor")

**Files:**
- Create: `crates/vox_agent/src/codegen.rs`

`AgentShaderGen::generate()` performs a topological sort, assigns variable names to output pins, and emits a WGSL compute shader. The entry point signature is derived from the `AgentStateDesc` via `layout_source()`.

- [ ] **Step 1: Write the failing test in `crates/vox_agent/src/codegen.rs`**

```rust
use crate::desc::AgentStateDesc;
use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry, NodeId, Connection};
use crate::compute::layout_source;

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("cycle detected in graph")]
    Cycle,
    #[error("unregistered custom node kind: {0}")]
    UnregisteredCustomNode(String),
    #[error("validation error: {0}")]
    Validation(String),
}

pub struct WgslSource {
    pub source:      String,
    pub entry_point: String,  // always "agent_update"
}

pub struct AgentShaderGen;

impl AgentShaderGen {
    /// Convert a validated AgentNodeGraph into a complete WGSL shader source.
    /// The entry point signature matches the bind group layout from `desc`.
    pub fn generate(
        graph: &AgentNodeGraph,
        registry: &AgentNodeRegistry,
        desc: &AgentStateDesc,
    ) -> Result<WgslSource, CodegenError> {
        let order = graph.topological_order().map_err(|_| CodegenError::Cycle)?;

        // Assign a variable name to each output pin
        let mut pin_var: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        let mut var_counter = 0usize;
        for &nid in &order {
            let node = graph.nodes().iter().find(|n| n.id() == nid).unwrap();
            for &pin in node.output_pins() {
                pin_var.insert(pin, format!("_v{}", var_counter));
                var_counter += 1;
            }
        }

        // Build pin → source var map from connections
        let mut pin_src: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        for c in graph.connections() {
            if let Some(src_var) = pin_var.get(&c.src_pin) {
                pin_src.insert(c.dst_pin, src_var.clone());
            }
        }

        let mut body = String::new();
        for &nid in &order {
            let node = graph.nodes().iter().find(|n| n.id() == nid).unwrap();
            let inputs: Vec<String> = node.input_pins().iter()
                .map(|p| pin_src.get(p).cloned().unwrap_or_else(|| "0.0".to_string()))
                .collect();
            let output = node.output_pins().first()
                .and_then(|p| pin_var.get(p))
                .cloned()
                .unwrap_or_default();

            let fragment = emit_node(node.kind(), &inputs, &output, registry, desc)?;
            body.push_str(&fragment);
            body.push('\n');
        }

        let bindings = layout_source(desc);
        let source = format!(
            "{bindings}\n\
             @compute @workgroup_size(64)\n\
             fn agent_update(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let i = gid.x;\n\
             if i >= uniforms.agent_count {{ return; }}\n\
             if (agent_flags[i] & 1u) == 0u {{ return; }}\n\
             {body}\
             }}\n"
        );

        Ok(WgslSource { source, entry_point: "agent_update".to_string() })
    }
}

fn emit_node(
    kind: &AgentNodeKind,
    inputs: &[String],
    output: &str,
    registry: &AgentNodeRegistry,
    _desc: &AgentStateDesc,
) -> Result<String, CodegenError> {
    let s = match kind {
        AgentNodeKind::OnUpdate => String::new(),
        AgentNodeKind::GetPosition => format!(
            "var {output} = vec3<f32>(positions_in[i*3u], positions_in[i*3u+1u], positions_in[i*3u+2u]);"),
        AgentNodeKind::GetVelocity => format!(
            "var {output} = vec3<f32>(velocities_in[i*3u], velocities_in[i*3u+1u], velocities_in[i*3u+2u]);"),
        AgentNodeKind::AgentId => format!("var {output} = i;"),
        AgentNodeKind::GetTime => format!("var {output} = uniforms.time;"),
        AgentNodeKind::ReadCustom { slot } => format!(
            "var {output} = custom[i * uniforms.custom_floats + {slot}u];"),
        AgentNodeKind::SampleSpectral { band } => format!(
            "var {output} = spectral_samples[i * 16u + {band}u];"),
        AgentNodeKind::Add => format!("var {output} = {} + {};", inputs[0], inputs[1]),
        AgentNodeKind::Sub => format!("var {output} = {} - {};", inputs[0], inputs[1]),
        AgentNodeKind::Mul => format!("var {output} = {} * {};", inputs[0], inputs[1]),
        AgentNodeKind::Div => format!("var {output} = {} / {};", inputs[0], inputs[1]),
        AgentNodeKind::Normalize => format!("var {output} = normalize({});", inputs[0]),
        AgentNodeKind::Length => format!("var {output} = length({});", inputs[0]),
        AgentNodeKind::Distance => format!("var {output} = distance({}, {});", inputs[0], inputs[1]),
        AgentNodeKind::Lerp => format!("var {output} = mix({}, {}, {});", inputs[0], inputs[1], inputs[2]),
        AgentNodeKind::Clamp => format!("var {output} = clamp({}, {}, {});", inputs[0], inputs[1], inputs[2]),
        AgentNodeKind::Noise => format!("var {output} = fract(sin({} * 127.1) * 43758.5453);", inputs[0]),
        AgentNodeKind::Compare { op } => {
            let cmp = match op {
                crate::node_graph::CompareOp::Lt => "<",
                crate::node_graph::CompareOp::Le => "<=",
                crate::node_graph::CompareOp::Gt => ">",
                crate::node_graph::CompareOp::Ge => ">=",
                crate::node_graph::CompareOp::Eq => "==",
                crate::node_graph::CompareOp::Ne => "!=",
            };
            format!("var {output} = ({} {cmp} {});", inputs[0], inputs[1])
        }
        AgentNodeKind::And => format!("var {output} = {} && {};", inputs[0], inputs[1]),
        AgentNodeKind::Or  => format!("var {output} = {} || {};", inputs[0], inputs[1]),
        AgentNodeKind::Not => format!("var {output} = !{};", inputs[0]),
        AgentNodeKind::SetVelocity => format!(
            "positions_out[i*3u]   = positions_in[i*3u]   + {0}.x * uniforms.dt;\n\
             positions_out[i*3u+1u] = positions_in[i*3u+1u] + {0}.y * uniforms.dt;\n\
             positions_out[i*3u+2u] = positions_in[i*3u+2u] + {0}.z * uniforms.dt;\n\
             velocities_out[i*3u]   = {0}.x;\n\
             velocities_out[i*3u+1u] = {0}.y;\n\
             velocities_out[i*3u+2u] = {0}.z;", inputs[0]),
        AgentNodeKind::AddVelocity => format!(
            "velocities_out[i*3u]   = velocities_in[i*3u]   + {0}.x;\n\
             velocities_out[i*3u+1u] = velocities_in[i*3u+1u] + {0}.y;\n\
             velocities_out[i*3u+2u] = velocities_in[i*3u+2u] + {0}.z;", inputs[0]),
        AgentNodeKind::WriteCustom { slot } => format!(
            "custom[i * uniforms.custom_floats + {slot}u] = {};", inputs[0]),
        AgentNodeKind::RequestCpuAttention =>
            "agent_flags[i] = agent_flags[i] | 2u;".to_string(), // bit 1 = needs_cpu
        AgentNodeKind::SpectralBand { band } => format!(
            "var {output} = {}.band{band};", inputs[0]),
        AgentNodeKind::SpectralDot => format!(
            "var {output} = dot(vec4<f32>({0}.x, {0}.y, {0}.z, {0}.w), \
                                vec4<f32>({1}.x, {1}.y, {1}.z, {1}.w));", inputs[0], inputs[1]),
        AgentNodeKind::Custom { kind_name } => {
            let frag = registry.get(kind_name)
                .ok_or_else(|| CodegenError::UnregisteredCustomNode(kind_name.clone()))?;
            let mut code = frag.0.clone();
            for (idx, input) in inputs.iter().enumerate() {
                code = code.replace(&format!("{{input_{idx}}}"), input);
            }
            code.replace("{output}", output)
        }
        // Unimplemented stubs (no codegen needed for these in initial pass)
        AgentNodeKind::OnSpectralThreshold { .. }
        | AgentNodeKind::OnNeedCritical
        | AgentNodeKind::QueryNeighbours { .. }
        | AgentNodeKind::NeighbourCount
        | AgentNodeKind::NeighbourPosition { .. }
        | AgentNodeKind::Select
        | AgentNodeKind::Branch
        | AgentNodeKind::SampleSpectralCurve => String::new(),
    };
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;
    use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry};
    use crate::desc::AgentStateDesc;
    use crate::compute::{AgentComputePipeline, ShaderSource};

    fn minimal_desc() -> AgentStateDesc {
        AgentStateDesc { agent_count: 4, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn single_node_graph_generates_wgsl() {
        let registry = AgentNodeRegistry::new();
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("test");
        g.add_node(AgentNodeKind::OnUpdate, [0.0, 0.0]);
        let result = AgentShaderGen::generate(&g, &registry, &desc);
        assert!(result.is_ok(), "single-node graph must generate: {:?}", result.err());
        let src = result.unwrap();
        assert!(src.source.contains("agent_update"), "must have entry point");
        assert!(src.source.contains("@compute"),    "must have @compute attribute");
    }

    #[test]
    fn node_graph_compiles_to_valid_wgsl() {
        // GetPosition → Normalize → SetVelocity
        let Some((device, _queue)) = test_device() else { return; };
        let registry = AgentNodeRegistry::new();
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("normalize_pos");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::Normalize,   [100.0, 0.0]);
        let c = g.add_node(AgentNodeKind::SetVelocity, [200.0, 0.0]);

        let a_out = g.nodes().iter().find(|n| n.id() == a).unwrap().output_pins()[0];
        let b_in  = g.nodes().iter().find(|n| n.id() == b).unwrap().input_pins()[0];
        let b_out = g.nodes().iter().find(|n| n.id() == b).unwrap().output_pins()[0];
        let c_in  = g.nodes().iter().find(|n| n.id() == c).unwrap().input_pins()[0];
        g.connect(a, a_out, b, b_in).unwrap();
        g.connect(b, b_out, c, c_in).unwrap();

        let wgsl = AgentShaderGen::generate(&g, &registry, &desc).expect("codegen");
        // Verify wgpu accepts the generated WGSL by loading it as a shader
        let result = AgentComputePipeline::new(&device,
            ShaderSource::Wgsl(wgsl.source), &desc);
        assert!(result.is_ok(), "generated WGSL must compile in wgpu: {:?}", result.err());
    }

    #[test]
    fn custom_node_fragment_is_substituted() {
        let mut registry = AgentNodeRegistry::new();
        registry.register("Double", crate::node_graph::SlangFragment(
            "var {output} = {input_0} * 2.0;".to_string()
        ));
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("custom");
        g.add_node(AgentNodeKind::Custom { kind_name: "Double".to_string() }, [0.0, 0.0]);
        let src = AgentShaderGen::generate(&g, &registry, &desc).expect("codegen");
        assert!(src.source.contains("* 2.0"), "custom fragment must be inlined");
    }

    #[test]
    fn unregistered_custom_node_returns_error() {
        let registry = AgentNodeRegistry::new();
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("bad");
        g.add_node(AgentNodeKind::Custom { kind_name: "NotRegistered".to_string() }, [0.0, 0.0]);
        let result = AgentShaderGen::generate(&g, &registry, &desc);
        assert!(matches!(result, Err(CodegenError::UnregisteredCustomNode(_))));
    }
}
```

- [ ] **Step 2: Fix `AgentNodeKind` — remove `OnNeedCritical` referenced in codegen**

The enum in `node_graph.rs` should not have `OnNeedCritical`. Verify it's not there. If the match in `emit_node` references it, adjust the match arm to `AgentNodeKind::OnSpectralThreshold { .. }` only.

- [ ] **Step 3: Run tests**

```bash
cargo test -p vox_agent --features editor codegen -- --nocapture
```

Expected: 4 tests pass (GPU tests skip gracefully).

- [ ] **Step 4: Commit**

```bash
git add crates/vox_agent/src/codegen.rs
git commit -m "feat(agent): AgentShaderGen — node graph to WGSL codegen"
```

---

## Task 11: AgentNodeEditor (feature = "editor")

**Files:**
- Create: `crates/vox_agent/src/editor.rs`

- [ ] **Step 1: Create `crates/vox_agent/src/editor.rs`**

```rust
use egui::Ui;
use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry};
use crate::desc::AgentStateDesc;
use crate::codegen::AgentShaderGen;

pub struct AgentNodeEditor {
    graph: AgentNodeGraph,
    registry: AgentNodeRegistry,
    status: String,
    pending_wgsl: Option<String>,
}

impl AgentNodeEditor {
    pub fn new() -> Self {
        Self {
            graph: AgentNodeGraph::new("default"),
            registry: AgentNodeRegistry::new(),
            status: "No shader compiled".to_string(),
            pending_wgsl: None,
        }
    }

    /// Returns compiled WGSL if a compile was triggered this frame.
    pub fn take_pending_wgsl(&mut self) -> Option<String> {
        self.pending_wgsl.take()
    }

    pub fn show(&mut self, ui: &mut Ui, desc: &AgentStateDesc) {
        ui.label("Agent Node Editor");
        ui.separator();

        // Palette sidebar
        ui.horizontal(|ui| {
            ui.group(|ui| {
                ui.label("Nodes");
                ui.separator();
                if ui.small_button("GetPosition").clicked() {
                    self.graph.add_node(AgentNodeKind::GetPosition, [50.0, 50.0]);
                }
                if ui.small_button("GetVelocity").clicked() {
                    self.graph.add_node(AgentNodeKind::GetVelocity, [50.0, 100.0]);
                }
                if ui.small_button("SetVelocity").clicked() {
                    self.graph.add_node(AgentNodeKind::SetVelocity, [250.0, 50.0]);
                }
                if ui.small_button("AddVelocity").clicked() {
                    self.graph.add_node(AgentNodeKind::AddVelocity, [250.0, 100.0]);
                }
                if ui.small_button("Normalize").clicked() {
                    self.graph.add_node(AgentNodeKind::Normalize, [150.0, 50.0]);
                }
                if ui.small_button("Noise").clicked() {
                    self.graph.add_node(AgentNodeKind::Noise, [150.0, 100.0]);
                }
                if desc.spectral {
                    if ui.small_button("SampleSpectral").clicked() {
                        self.graph.add_node(AgentNodeKind::SampleSpectral { band: 5 }, [50.0, 150.0]);
                    }
                }
            });

            ui.group(|ui| {
                ui.label("Graph");
                ui.label(format!("{} nodes, {} connections",
                    self.graph.nodes().len(), self.graph.connections().len()));
            });
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Compile").clicked() {
                match AgentShaderGen::generate(&self.graph, &self.registry, desc) {
                    Ok(wgsl) => {
                        self.status = "Compiled OK".to_string();
                        self.pending_wgsl = Some(wgsl.source);
                    }
                    Err(e) => {
                        self.status = format!("Error: {e}");
                    }
                }
            }
            ui.label(&self.status);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_new_has_empty_graph() {
        let editor = AgentNodeEditor::new();
        assert_eq!(editor.graph.nodes().len(), 0);
    }

    #[test]
    fn take_pending_wgsl_returns_none_initially() {
        let mut editor = AgentNodeEditor::new();
        assert!(editor.take_pending_wgsl().is_none());
    }

    #[test]
    fn editor_renders_without_panic() {
        // headless egui test
        let ctx = egui::Context::default();
        let desc = crate::desc::AgentStateDesc {
            agent_count: 10, custom_floats: 0, spectral: false, spatial_hash: None,
        };
        ctx.run(egui::RawInput::default(), |ctx| {
            egui::Window::new("test").show(ctx, |ui| {
                let mut editor = AgentNodeEditor::new();
                editor.show(ui, &desc);
            });
        });
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_agent --features editor editor -- --nocapture
```

Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_agent/src/editor.rs
git commit -m "feat(agent): AgentNodeEditor — palette, compile button, headless test"
```

---

## Task 12: Wire into vox_app

**Files:**
- Modify: `crates/vox_app/Cargo.toml`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`
- Modify: `crates/vox_app/src/editor_app.rs`

- [ ] **Step 1: Add `vox_agent` to `crates/vox_app/Cargo.toml`**

In the `[dependencies]` section:

```toml
vox_agent = { path = "../vox_agent", features = ["editor"] }
```

- [ ] **Step 2: Add `AgentComputeLayer` field to `EngineApp` in `engine_runner.rs`**

Find the `EngineApp` struct definition. Add the field:

```rust
agent_layer: vox_app::agent_layer::AgentComputeLayerHandle,
```

Since `EngineApp` is large, add a small wrapper module in `vox_app` to avoid bloating `engine_runner.rs`. Create `crates/vox_app/src/agent_layer.rs`:

```rust
//! Thin wrapper wiring AgentComputeLayer into the engine app.

use vox_agent::{AgentComputeLayer, AgentStateDesc};

/// Holds the AgentComputeLayer and the wgpu device needed for tick().
pub struct AgentComputeLayerHandle {
    pub layer: AgentComputeLayer,
}

impl AgentComputeLayerHandle {
    pub fn new(device: &wgpu::Device) -> Self {
        let desc = AgentStateDesc {
            agent_count: 0,        // starts empty; game sets agent count on scene load
            custom_floats: 8,      // 8 game-defined floats per agent (city sim default)
            spectral: true,        // spectral field enabled
            spatial_hash: Some(vox_agent::SpatialHashDesc {
                grid_origin_x: -20480.0,
                grid_origin_z: -20480.0,
                grid_extent:   40960.0,
                cell_size:     10.0,
            }),
        };
        let mut layer = AgentComputeLayer::new(device, desc);
        layer.load_default_shader(device).expect("default agent shader must load");
        Self { layer }
    }
}
```

Add `pub mod agent_layer;` to `crates/vox_app/src/lib.rs`.

- [ ] **Step 3: Add field to `EngineApp` in `engine_runner.rs`**

In the `EngineApp` struct, add:
```rust
agent_layer: vox_app::agent_layer::AgentComputeLayerHandle,
```

In the `Self { ... }` construction block, add:
```rust
agent_layer: vox_app::agent_layer::AgentComputeLayerHandle::new(&device),
```

where `device` is the existing `wgpu::Device` already available at construction time.

- [ ] **Step 4: Call `tick()` in the render loop**

In `EngineApp::about_to_wait` (or the equivalent render loop method), before `queue.submit()`, add:

```rust
self.agent_layer.layer.tick(
    &device,
    &mut encoder,
    &queue,
    None,        // spectral_samples: None until Spectra integration
    dt,
);
```

Use the existing `encoder` and `queue` variables already in scope.

- [ ] **Step 5: Add Agents tab to `EditorApp::show()` in `editor_app.rs`**

In the `ContextPanel` tab rendering (inside `WorkspaceMode::Simulate`), add a call to `AgentComputeLayer::show_editor()`. Find the `context_panel.show()` call and add an Agents tab:

```rust
// Inside the Simulate mode tab content:
agent_fabric.layer.show_editor(ui, /* pass the desc */);
```

The exact wiring depends on how `EditorApp` already structures tabs. Add an `agent_fabric: &mut vox_app::agent_layer::AgentComputeLayerHandle` parameter to `EditorApp::show()` or pass it through the existing pattern.

- [ ] **Step 6: Verify build**

```bash
cargo build --bin ochroma 2>&1 | grep "^error" | head -20
```

Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/Cargo.toml crates/vox_app/src/agent_layer.rs \
        crates/vox_app/src/lib.rs crates/vox_app/src/bin/engine_runner.rs \
        crates/vox_app/src/editor_app.rs
git commit -m "feat(app): wire AgentComputeLayer into engine_runner and editor"
```

---

## Task 13: Flocking Example

**Files:**
- Create: `crates/vox_agent/examples/flocking.rs`

A self-contained Boids flocking demo. Three rules (separation, alignment, cohesion) are hand-written in a WGSL shader. No game-layer code. Proves `AgentComputeLayer` works without `vox_sim`.

- [ ] **Step 1: Create `crates/vox_agent/examples/flocking.rs`**

```rust
//! Boids flocking simulation using AgentComputeLayer.
//! Run: cargo run --example flocking -p vox_agent

fn main() {
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter found");
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flocking"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("device creation failed")
    });

    let n = 10_000u32;
    let desc = vox_agent::AgentStateDesc {
        agent_count: n,
        custom_floats: 0,
        spectral: false,
        spatial_hash: Some(vox_agent::SpatialHashDesc {
            grid_origin_x: -500.0,
            grid_origin_z: -500.0,
            grid_extent: 1000.0,
            cell_size: 25.0,
        }),
    };

    let mut layer = vox_agent::AgentComputeLayer::new(&device, desc.clone());

    // Boids shader: separation + alignment + cohesion using spatial hash
    let boids_wgsl = format!("{}\n{}", layer.bind_group_layout_source(),
        include_str!("boids_behavior.wgsl"));
    layer.load_shader(&device, vox_agent::ShaderSource::Wgsl(boids_wgsl))
        .expect("boids shader failed to load — falling back to default");

    // Initialize agents in a 100x100 grid
    let mut positions = Vec::with_capacity(n as usize);
    let mut velocities = Vec::with_capacity(n as usize);
    for i in 0..n {
        let x = (i % 100) as f32 * 1.0 - 50.0;
        let z = (i / 100) as f32 * 1.0 - 50.0;
        positions.push([x, 0.0, z]);
        velocities.push([0.1, 0.0, 0.0]);
    }
    layer.buffers_mut().upload_positions(&queue, &positions);
    layer.buffers_mut().upload_velocities(&queue, &velocities);
    layer.buffers_mut().mark_all_alive(&queue);
    queue.submit([]);

    println!("Running Boids with {} agents for 300 frames...", n);
    let mut total_ms = 0.0f64;

    for frame in 0..300usize {
        let t0 = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("boids") });
        layer.tick(&device, &mut encoder, &queue, None, 1.0 / 60.0);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;

        if frame % 60 == 0 {
            println!("frame {:>3}: {:.2}ms", frame, t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    println!("avg: {:.2}ms over 300 frames ({} agents)", total_ms / 300.0, n);
}
```

Create the companion `crates/vox_agent/examples/boids_behavior.wgsl`:

```wgsl
// Boids rules using spatial hash neighbour queries.
// Included after bind_group_layout_source() which provides all bindings.

@compute @workgroup_size(64)
fn agent_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= uniforms.agent_count { return; }
    if (agent_flags[i] & 1u) == 0u { return; }

    let px = positions_in[i*3u];
    let py = positions_in[i*3u+1u];
    let pz = positions_in[i*3u+2u];
    let vx = velocities_in[i*3u];
    let vy = velocities_in[i*3u+1u];
    let vz = velocities_in[i*3u+2u];

    var sep_x = 0.0; var sep_z = 0.0;
    var aln_x = 0.0; var aln_z = 0.0;
    var coh_x = 0.0; var coh_z = 0.0;
    var count = 0u;

    let cell = spatial_cell[i];
    let gw = uniforms.grid_width;
    let cx_i = cell % gw;
    let cz_i = cell / gw;

    // Check 3x3 neighbourhood of cells
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx = i32(cx_i) + dx;
            let nz = i32(cz_i) + dz;
            if nx < 0 || nz < 0 || u32(nx) >= gw || u32(nz) >= gw { continue; }
            let ncell = u32(nz) * gw + u32(nx);
            let start = cell_offsets[ncell];
            let end   = cell_offsets[ncell + 1u];
            for (var k = start; k < end; k = k + 1u) {
                let j = cell_data[k];
                if j == i { continue; }
                let jx = positions_in[j*3u];
                let jy = positions_in[j*3u+1u];
                let jz = positions_in[j*3u+2u];
                let dx2 = jx - px; let dz2 = jz - pz;
                let dist2 = dx2*dx2 + dz2*dz2;
                if dist2 > 400.0 { continue; } // 20m radius
                // Separation: push away
                if dist2 < 4.0 { sep_x = sep_x - dx2; sep_z = sep_z - dz2; }
                // Alignment: match velocity
                aln_x = aln_x + velocities_in[j*3u];
                aln_z = aln_z + velocities_in[j*3u+2u];
                // Cohesion: steer toward centre
                coh_x = coh_x + jx; coh_z = coh_z + jz;
                count = count + 1u;
            }
        }
    }

    var new_vx = vx; var new_vz = vz;
    if count > 0u {
        let inv = 1.0 / f32(count);
        new_vx = new_vx + sep_x * 0.05 + (aln_x * inv - vx) * 0.1 + (coh_x * inv - px) * 0.01;
        new_vz = new_vz + sep_z * 0.05 + (aln_z * inv - vz) * 0.1 + (coh_z * inv - pz) * 0.01;
    }

    // Clamp speed to [0.05, 2.0]
    let speed = sqrt(new_vx * new_vx + new_vz * new_vz);
    if speed > 0.001 {
        let clamped = clamp(speed, 0.05, 2.0);
        new_vx = new_vx / speed * clamped;
        new_vz = new_vz / speed * clamped;
    }

    // Wrap world bounds
    var new_px = px + new_vx * uniforms.dt;
    var new_pz = pz + new_vz * uniforms.dt;
    if new_px > 500.0 { new_px = new_px - 1000.0; }
    if new_px < -500.0 { new_px = new_px + 1000.0; }
    if new_pz > 500.0 { new_pz = new_pz - 1000.0; }
    if new_pz < -500.0 { new_pz = new_pz + 1000.0; }

    positions_out[i*3u]     = new_px;
    positions_out[i*3u+1u]  = py;
    positions_out[i*3u+2u]  = new_pz;
    velocities_out[i*3u]    = new_vx;
    velocities_out[i*3u+1u] = vy;
    velocities_out[i*3u+2u] = new_vz;
}
```

- [ ] **Step 2: Build example**

```bash
cargo build --example flocking -p vox_agent 2>&1 | grep "^error" | head -10
```

Expected: 0 errors.

- [ ] **Step 3: Run example (requires GPU)**

```bash
cargo run --example flocking -p vox_agent
```

Expected output (approximate):
```
Running Boids with 10000 agents for 300 frames...
frame   0: 2.1ms
frame  60: 1.8ms
frame 120: 1.9ms
frame 180: 2.0ms
frame 240: 1.9ms
avg: 2.0ms over 300 frames (10000 agents)
```

- [ ] **Step 4: Commit**

```bash
git add crates/vox_agent/examples/
git commit -m "feat(agent): Boids flocking example demonstrating AgentComputeLayer"
```

---

## Final Verification

- [ ] **Full test suite passes**

```bash
cargo test -p vox_agent --features editor -- --nocapture
```

Expected: all unit tests pass. GPU tests skip if no GPU present — they do not fail.

- [ ] **Binary builds**

```bash
cargo build --bin ochroma
```

Expected: 0 errors.

- [ ] **bench_million (requires GPU)**

```bash
cargo test -p vox_agent --test bench_million -- --nocapture --ignored
```

Expected:
```
agents:  1,000,000  avg_frame_ms: <=16.0  min_fps: 60  dispatch: GPU
```
