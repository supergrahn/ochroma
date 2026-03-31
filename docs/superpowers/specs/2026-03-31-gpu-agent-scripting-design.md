# Design: GPU Agent Compute Layer (2026-03-31)

**Status:** Draft
**Scope:** A generic GPU agent dispatch framework for `vox_agent`. The engine provides the rails — SoA buffer allocation, spatial hash, tiered CPU scheduling, optional visual scripting infrastructure. The game supplies the agent state descriptor and behavior shader. No game concepts exist in this crate.
**Related:** Domain 11 AI/LLM Plan, Domain 12 Spectral Frontier Plan

---

## 1. Problem Statement

- Large-scale agent simulation saturates CPU before reaching interesting densities. ~50,000 agents exhaust a single thread at 16.67ms on a desktop CPU.
- No path exists from the visual scripting infrastructure (`visual_graph.rs`, `node_graph_widget.rs`) to execution — the graph is a data model with no evaluator, no compiler, no runtime.
- Agent behavior is hardcoded in Rust. Changing a decision rule requires a recompile. There is no designer-facing tool for expressing agent logic.
- The spectral field (16-band GPU buffer produced by the render pipeline) is inaccessible to agent behavior logic.
- Deliberative CPU decisions that only need to happen at 4Hz are running at 60Hz.

The engine should provide the dispatch infrastructure so that any game built on Ochroma can author GPU-speed agent behavior without re-implementing buffer management, spatial queries, or tiered scheduling.

---

## 2. Done When

Running `cargo test -p vox_agent --test bench_million -- --nocapture` prints:

```
agents: 1_000_000  avg_frame_ms: <=16.0  min_fps: 60  dispatch: GPU
```

This test dispatches 1,000,000 agents for 120 consecutive frames on the GPU and asserts average frame time ≤ 16.0ms. The test runs headless (no window), supplies a hand-written SPIR-V behavior shader (no node graph), and uses a zero-filled spectral buffer. Verified on RTX 3060 or equivalent (13 TFLOPS FP32, 360 GB/s bandwidth).

Additionally, `cargo run --example flocking` demonstrates a self-contained flocking simulation (Boids rules) using a node-graph authored behavior, with no game-layer code. Agents visibly steer using only engine-provided node types.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| 1M agents GPU dispatch ≤ 16ms | `bench_million` asserts `avg_frame_ms <= 16.0` over 120 frames with 1M agents and a hand-written SPIR-V shader | `assert!(dispatch_returned_ok)` — passes with empty kernel |
| State descriptor drives buffer layout | `cargo test state_desc_matches_bind_group` creates two `AgentStateDesc` configurations (with and without spectral, with and without spatial hash), asserts the generated bind group layout matches the descriptor in both cases | `assert!(layout.is_some())` |
| Game shader binds successfully | `cargo test game_shader_binds` loads a hand-written SPIR-V shader whose entry point matches the layout from a known `AgentStateDesc`, asserts `AgentComputeLayer::load_shader()` returns `Ok` | `assert!(result.is_ok())` with stub that accepts anything |
| Spatial hash returns correct neighbours | `cargo test neighbour_query_correctness` places 100 agents in known positions, asserts each agent's neighbour query returns exactly the agents within the query radius | `assert!(neighbours.len() > 0)` |
| Spectral field readable in shader | `cargo test spectral_reads_live_field` dispatches 1000 agents with spectral enabled and a CPU-filled synthetic spectral buffer (band 5 hotspot), asserts agents near the hotspot have average velocity pointing toward it | `assert!(velocity != Vec3::ZERO)` — passes with random velocity |
| Tier scheduling fires at correct rates | `cargo test tier_scheduling` runs 1000 agents for 60 frames, asserts tier-1 GPU dispatches happened 60 times and tier-2 CPU callbacks fired 4 times (±1) | `assert!(tier2_ran)` — passes if it ran once |
| Node graph compiles to valid SPIR-V | `cargo test node_graph_compiles_to_spirv` passes a 10-node Boids graph through codegen + Slang compiler, asserts output is non-empty valid SPIR-V bytes | `assert!(spirv_bytes.len() > 0)` with stub returning `[0u8; 4]` |
| Hot-swap compiled shader | `cargo test shader_hot_swap` loads shader A, dispatches it, loads shader B, swaps, dispatches again, asserts agents respond to new behavior within 1 rendered frame *after the swap completes* | `assert!(swap_succeeded)` |
| Custom node round-trips through codegen | `cargo test custom_node_round_trips` registers a `CustomMultiply` node via `register_node()`, builds a graph containing it, runs codegen, asserts the generated Slang contains the expected fragment | `assert!(slang.contains("custom"))` |

---

## 4. Architecture

### 4.1 Overview

```
  Game layer                         Engine layer (vox_agent)
  ──────────────────────────────     ────────────────────────────────────────

  AgentStateDesc  ─────────────────► AgentComputeLayer::new(desc)
  (state layout)                       allocates AgentStateBuffers
                                        builds SpatialHash (if desc.spatial_hash)
                                        builds TierScheduler

  Behavior shader (SPIR-V)          ┐
    — hand-written, OR              │
    — output of node graph editor   ├─► AgentComputeLayer::load_shader(spirv)
    — output of SlangCodegen        ┘     hot-swappable

  tier2_callback: FnMut(AgentSlice) ──► TierScheduler  (4Hz, rayon)
  tier3_callback: FnMut(AgentSlice) ──► TierScheduler  (0.25Hz, rayon)

  AgentComputeLayer::tick()
    1. apply write-backs from last CPU callback
    2. rebuild SpatialHash  (3 compute passes, if enabled)
    3. dispatch behavior shader  (1 compute pass)
    4. buffers.swap()
    5. fire tier-2/3 callbacks on rayon for this frame's slice
```

### 4.2 Crate: `vox_agent`

Engine crate. No dependency on `vox_sim`, `vox_render`, or `vox_app`.

Dependencies: `vox_core`, `wgpu`, `shader-slang` (optional feature), `egui` (optional feature).

The spectral samples buffer is passed into `tick()` as `Option<&wgpu::Buffer>`. `vox_agent` binds it read-only when present. Who fills it (Spectra, a CPU upload, a test fixture) is not `vox_agent`'s concern.

`vox_app` depends on `vox_agent` and wires it into `EngineApp`. `vox_sim` knows nothing about `vox_agent`.

### 4.3 Agent State Descriptor

The game describes its agent state layout. The engine allocates buffers and generates the matching bind group layout.

**Fixed engine-managed fields (always present):**

```
positions[N]    : [f32; 3]  — world-space position (ping-pong)
velocities[N]   : [f32; 3]  — current velocity (ping-pong)
flags[N]        : u32       — alive[0], needs_cpu[1], agent_type[2..9], reserved[10..31]
spatial_cell[N] : u32       — current cell index (written by spatial hash pass)
```

**Game-controlled optional fields:**

```
custom[N * custom_floats] : f32   — game assigns meaning to each slot
spectral_cache[N * 16]    : f32   — per-agent spectral sample (only if desc.spectral = true)
```

Per-agent byte cost: 12 + 12 + 4 + 4 + (4 × custom_floats) + (64 if spectral) = **32 + 4K bytes** (K = custom_floats). At K=8, spectral enabled: 128 bytes/agent. At 1M agents: 128 MB VRAM. Ping-pong for positions + velocities = 152 MB total. Within budget of RTX 3060 (12 GB VRAM).

Spatial hash overhead: (grid_width²) × 8 bytes (count + offset) + N × 4 bytes (indices). At 4096×4096 grid: 134 MB fixed. Callers should be aware this is a significant fixed cost.

### 4.4 Spatial Hash (GPU, optional)

Enabled via `AgentStateDesc::spatial_hash: Some(SpatialHashDesc { ... })`. When enabled, rebuilt each frame in three passes before the behavior shader runs:

1. **Count pass** — each agent atomically increments its cell's count
2. **Prefix sum pass** — exclusive scan over cell counts → cell offsets
3. **Scatter pass** — each agent writes its index into `cell_data[cell_offset[cell] + local_idx]`

Pattern from `vox_physics/src/pbf.rs`. Will be duplicated initially; shared `vox_gpu_util` extraction is deferred until `vox_physics` also needs the shared version.

Grid dimensions are set at construction. Agents outside bounds are clamped to the nearest edge cell.

### 4.5 Tiered CPU Scheduling

```
Frame N (16.67ms budget):
  GPU submit:  [spatial hash 3 passes] + [behavior shader 1 pass] — async compute queue
  CPU thread:  tier-2 callback for agents [N*K .. (N+1)*K], K = count/15  (≈ 4Hz at 60fps)
  CPU thread:  tier-3 callback for agents [N*J .. (N+1)*J], J = count/240  (≈ 0.25Hz at 60fps)
```

The engine calls game-provided callbacks with an `AgentSlice` (CPU-side view of a contiguous agent range) and an `AgentWriteQueue`. The callback reads agent state and enqueues mutations. Write-backs are applied at the start of the next tick.

Tier callbacks run on rayon and must not block the render thread. Agents with `needs_cpu` flag set are prepended to the tier-2 queue ahead of the rotation schedule.

The engine provides no CPU state. `AgentSlice` contains: agent IDs, positions, velocities, flags, and custom floats for the slice. Any game state lookup (navmesh, economy, AI planner) is the callback's own responsibility.

### 4.6 Behavior Shader Interface

The game's SPIR-V shader must export an entry point `agent_update` with a signature that matches the bind group layout derived from `AgentStateDesc`. The engine provides `AgentComputeLayer::bind_group_layout_source() -> String` — a Slang/WGSL snippet declaring all bindings — so game shaders can `#include` it without manually tracking layout.

**Bind group layout (derived from descriptor):**

```slang
// Always present
StructuredBuffer<float3>    positions_in;
RWStructuredBuffer<float3>  positions_out;
StructuredBuffer<float3>    velocities_in;
RWStructuredBuffer<float3>  velocities_out;
RWStructuredBuffer<uint>    agent_flags;
uniform AgentUniforms       uniforms;

// Present when desc.spatial_hash.is_some()
StructuredBuffer<uint>      spatial_cells;
StructuredBuffer<uint>      cell_offsets;

// Present when desc.custom_floats > 0
RWStructuredBuffer<float>   custom;   // [N * custom_floats]

// Present when desc.spectral = true
StructuredBuffer<float>     spectral_samples;  // [N * 16]
```

`AgentUniforms` always contains: `agent_count`, `custom_floats`, `dt`, `time`, `grid_width`, `cell_size`.

### 4.7 Optional: Node Graph + Slang Codegen

The node graph editor and `SlangCodegen` are optional infrastructure — useful for games that want visual authoring but not required. A game can bypass both and supply raw SPIR-V directly.

**Engine-provided node types** are structural — they map to the engine-managed buffers:

- **Entry:** `OnUpdate { dt: Float }`, `OnSpectralThreshold { band: Int, threshold: Float }` *(spectral only)*
- **Read:** `GetPosition → Vec3`, `GetVelocity → Vec3`, `AgentId → Int`, `GetTime → Float`, `ReadCustom(slot: Int) → Float` *(custom only)*, `SampleSpectral(band: Int) → Float` *(spectral only)*, `QueryNeighbours(radius: Float) → NeighbourList` *(spatial hash only)*, `NeighbourCount(list) → Int`, `NeighbourPosition(list, i: Int) → Vec3`
- **Math:** `Add`, `Sub`, `Mul`, `Div`, `Lerp`, `Clamp`, `Normalize`, `Length`, `Distance`, `Select(Bool, a, b)`, `Noise(seed: Float) → Float`
- **Logic:** `Compare(op: CompareOp) → Bool`, `And`, `Or`, `Not`, `Branch(Bool) → Flow×2`
- **Write:** `SetVelocity(Vec3)`, `AddVelocity(Vec3)`, `WriteCustom(slot: Int, value: Float)` *(custom only)*, `RequestCpuAttention`
- **Spectral:** `SpectralDot(a, b) → Float`, `SampleSpectralCurve(pos: Vec3) → SpectralValue`, `SpectralBand(curve: SpectralValue, band: Int) → Float` *(spectral only)*

**Game-registered node types** extend this set via `AgentNodeRegistry::register(kind_name, SlangFragment)`. A Slang fragment is a code template with `{input_N}` and `{output}` placeholders filled in by codegen. The game can add pathfinding requests, domain-specific reads, event emission, or any other behavior — without modifying `vox_agent`.

`SlangCodegen::generate(graph, registry, desc)` produces a Slang source file whose entry point signature exactly matches the bind group layout derived from `desc`. Generated source is human-readable.

**Compilation pipeline:**

- *Build-time:* `slang-hal-build` in `vox_agent/build.rs` compiles `.slang` files in `assets/agents/` to SPIR-V. Zero runtime compiler dependency for shipped behaviors.
- *Runtime:* When the user saves a node graph, `SlangCodegen` emits Slang source, `shader-slang` compiles to SPIR-V on a background rayon thread. Old pipeline stays active until compilation completes, then swapped atomically.

### 4.8 Node Graph Editor

An egui panel (optional feature flag `feature = "editor"`) wrapping the existing node graph widget in `vox_app::node_graph_panel`. Adds:

- A node palette sidebar (built-in nodes filtered by active `AgentStateDesc`; registered custom nodes grouped separately) with drag-to-add
- Type-colored pin rendering
- A compile button triggering async Slang compilation
- A status bar: "Compiled OK — 1,247 agents" or inline compile errors

---

## 5. Data Models

```rust
// crates/vox_agent/src/lib.rs

/// Describes the agent state layout. Drives buffer allocation and bind group layout.
pub struct AgentStateDesc {
    pub agent_count: u32,
    pub custom_floats: u32,          // game-defined floats per agent; 0 = no custom buffer
    pub spectral: bool,              // include spectral_cache[N*16] buffer
    pub spatial_hash: Option<SpatialHashDesc>,
}

pub struct SpatialHashDesc {
    pub grid_origin: [f32; 2],
    pub grid_extent: f32,            // world units; grid is extent × extent
    pub cell_size: f32,              // world units per cell
}

// crates/vox_agent/src/state.rs

/// SoA GPU buffers allocated from an AgentStateDesc.
pub struct AgentStateBuffers {
    desc: AgentStateDesc,
    positions_a: wgpu::Buffer,
    positions_b: wgpu::Buffer,
    velocities_a: wgpu::Buffer,
    velocities_b: wgpu::Buffer,
    flags: wgpu::Buffer,
    spatial_cell: wgpu::Buffer,
    custom: Option<wgpu::Buffer>,
    spectral_cache: Option<wgpu::Buffer>,
    read_index: u8,
}

impl AgentStateBuffers {
    pub fn new(device: &wgpu::Device, desc: AgentStateDesc) -> Self;
    pub fn desc(&self) -> &AgentStateDesc;
    pub fn swap(&mut self);
    pub fn read_positions(&self) -> &wgpu::Buffer;
    pub fn write_positions(&self) -> &wgpu::Buffer;
    pub fn read_velocities(&self) -> &wgpu::Buffer;
    pub fn write_velocities(&self) -> &wgpu::Buffer;
}

// crates/vox_agent/src/compute.rs

/// A loaded compute pipeline. Matches the bind group layout of the AgentStateDesc
/// it was loaded against.
pub struct AgentComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    source_hash: u64,
}

impl AgentComputePipeline {
    pub fn from_spirv(
        device: &wgpu::Device,
        spirv: &[u32],
        desc: &AgentStateDesc,
    ) -> Result<Self, PipelineError>;

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &AgentStateBuffers,
        spectral_samples: Option<&wgpu::Buffer>,
        uniforms: AgentUniforms,
    );
}

// crates/vox_agent/src/spatial_hash.rs

/// Three-pipeline bundle for the spatial hash rebuild.
pub struct SpatialHashPipelines {
    count: wgpu::ComputePipeline,
    prefix_sum: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

impl SpatialHashPipelines {
    pub fn new(device: &wgpu::Device, desc: &SpatialHashDesc) -> Self;
}

pub fn rebuild_spatial_hash(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SpatialHashPipelines,
    buffers: &AgentStateBuffers,
);

// crates/vox_agent/src/scheduler.rs

/// CPU-side read-only view of a contiguous agent slice.
pub struct AgentSlice<'a> {
    pub agent_ids: &'a [u32],
    pub positions: &'a [[f32; 3]],
    pub velocities: &'a [[f32; 3]],
    pub flags: &'a [u32],
    pub custom: &'a [f32],       // length = slice_len * desc.custom_floats
    pub custom_floats: u32,
}

/// Queues write-back mutations from CPU callbacks to GPU buffers.
/// Applied at the start of the next tick, before the compute dispatch.
pub struct AgentWriteQueue { /* opaque */ }

impl AgentWriteQueue {
    pub fn write_custom(&mut self, agent_id: u32, slot: u32, value: f32);
    pub fn write_flag_bits(&mut self, agent_id: u32, mask: u32, value: u32);
    pub fn write_velocity(&mut self, agent_id: u32, velocity: [f32; 3]);
}

// crates/vox_agent/src/uniforms.rs

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AgentUniforms {
    pub agent_count: u32,
    pub custom_floats: u32,
    pub dt: f32,
    pub time: f32,
    pub grid_width: u32,
    pub cell_size: f32,
    pub _pad: [f32; 2],
}

// crates/vox_agent/src/node_graph.rs  (feature = "editor")

pub struct AgentNode {
    id: NodeId,
    kind: AgentNodeKind,
    position: [f32; 2],
    inputs: Vec<PinId>,
    outputs: Vec<PinId>,
}

impl AgentNode {
    pub fn id(&self) -> NodeId;
    pub fn kind(&self) -> &AgentNodeKind;
    pub fn input_pins(&self) -> &[PinId];
    pub fn output_pins(&self) -> &[PinId];
}

pub struct AgentNodeGraph {
    nodes: Vec<AgentNode>,
    connections: Vec<Connection>,
    name: String,
}

impl AgentNodeGraph {
    pub fn new(name: impl Into<String>) -> Self;
    pub fn add_node(&mut self, kind: AgentNodeKind, position: [f32; 2]) -> NodeId;
    /// Structural connection. Type checking is in validate().
    pub fn connect(&mut self, src: NodeId, src_pin: PinId, dst: NodeId, dst_pin: PinId)
        -> Result<(), ConnectionError>;
    pub fn validate(&self, registry: &AgentNodeRegistry, desc: &AgentStateDesc)
        -> Result<(), Vec<ValidationError>>;
    pub fn topological_order(&self) -> Result<Vec<NodeId>, CycleError>;
}

/// Registry of game-registered custom node kinds.
pub struct AgentNodeRegistry {
    custom: HashMap<String, SlangFragment>,
}

impl AgentNodeRegistry {
    pub fn new() -> Self;
    /// `fragment` is a Slang template with {input_N} and {output} placeholders.
    pub fn register(&mut self, kind_name: impl Into<String>, fragment: SlangFragment);
}

/// Built-in engine node types. All game-domain nodes are registered via AgentNodeRegistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentNodeKind {
    // Entry
    OnUpdate,
    OnSpectralThreshold { band: u32, threshold: f32 },  // requires desc.spectral
    // Read
    GetPosition,
    GetVelocity,
    AgentId,
    GetTime,
    ReadCustom { slot: u32 },                            // requires desc.custom_floats > 0
    SampleSpectral { band: u32 },                        // requires desc.spectral
    QueryNeighbours { radius: f32 },                     // requires desc.spatial_hash
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
    WriteCustom { slot: u32 },                           // requires desc.custom_floats > 0
    RequestCpuAttention,
    // Spectral
    SpectralDot, SampleSpectralCurve, SpectralBand { band: u32 },  // requires desc.spectral
    // Game-registered
    Custom { kind_name: String },
}
```

---

## 6. API

```rust
// crates/vox_agent/src/lib.rs

/// Top-level engine primitive. Owned by the game's EngineApp, ticked each frame.
pub struct AgentComputeLayer {
    buffers: AgentStateBuffers,
    pipeline: Option<AgentComputePipeline>,
    pending_pipeline: Option<Arc<Mutex<Option<AgentComputePipeline>>>>,
    spatial_hash: Option<SpatialHashPipelines>,
    tier_scheduler: TierScheduler,
    registry: AgentNodeRegistry,
    editor: Option<AgentNodeEditor>,    // None when feature = "editor" is off
}

impl AgentComputeLayer {
    /// Allocate GPU buffers from descriptor. Call once at scene load.
    pub fn new(device: &wgpu::Device, desc: AgentStateDesc) -> Self;

    /// Load a behavior shader. The SPIR-V entry point must match the bind group
    /// layout derived from the descriptor passed to new(). Returns Err if the
    /// layout is incompatible.
    pub fn load_shader(&mut self, device: &wgpu::Device, spirv: &[u32])
        -> Result<(), PipelineError>;

    /// Submit a node graph for async compilation. Old pipeline stays active until done.
    /// Requires feature = "editor". Graph must have been validated against this
    /// layer's AgentStateDesc before submission.
    #[cfg(feature = "editor")]
    pub fn submit_graph(&mut self, graph: AgentNodeGraph);

    /// Register a custom node kind for the editor and codegen.
    #[cfg(feature = "editor")]
    pub fn register_node(&mut self, kind_name: impl Into<String>, fragment: SlangFragment);

    /// Returns a Slang/WGSL snippet declaring the bind group bindings for this
    /// descriptor. Include this in hand-written shaders to stay in sync with the layout.
    pub fn bind_group_layout_source(&self) -> String;

    /// Call once per frame from the render loop.
    /// `spectral_samples`: N×16 float buffer, caller-owned — pass None to skip spectral.
    pub fn tick(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        spectral_samples: Option<&wgpu::Buffer>,
        dt: f32,
    );

    /// Set the tier-2 callback (≈4Hz). Fires on a rayon thread. Must be Send.
    pub fn set_tier2_callback(
        &mut self,
        cb: Box<dyn FnMut(AgentSlice, &mut AgentWriteQueue) + Send>,
    );

    /// Set the tier-3 callback (≈0.25Hz). Fires on a rayon thread. Must be Send.
    pub fn set_tier3_callback(
        &mut self,
        cb: Box<dyn FnMut(AgentSlice, &mut AgentWriteQueue) + Send>,
    );

    /// Render the node graph editor panel. Call inside egui frame.
    #[cfg(feature = "editor")]
    pub fn show_editor(&mut self, ui: &mut egui::Ui);

    /// Read agent positions back to CPU. Expensive — for debug/test only.
    /// Requires polling: call device.poll(wgpu::Maintain::Wait) after this
    /// returns before awaiting the future.
    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue)
        -> impl Future<Output = Vec<[f32; 3]>>;
}

// crates/vox_agent/src/codegen.rs  (feature = "editor")

pub struct SlangCodegen;

impl SlangCodegen {
    /// Convert a validated AgentNodeGraph to Slang source whose entry point
    /// matches the bind group layout derived from `desc`.
    /// Threading: pure function, safe to call from any thread.
    pub fn generate(
        graph: &AgentNodeGraph,
        registry: &AgentNodeRegistry,
        desc: &AgentStateDesc,
    ) -> Result<SlangSource, CodegenError>;
}
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `AgentComputeLayer::new()` | `EngineApp` construction | `crates/vox_app/src/bin/engine_runner.rs` | after wgpu device init; desc comes from game config |
| `AgentComputeLayer::load_shader()` | `EngineApp` construction | `crates/vox_app/src/bin/engine_runner.rs` | loads default SPIR-V from `assets/agents/` |
| `AgentComputeLayer::set_tier2_callback()` | `EngineApp` construction | `crates/vox_app/src/bin/engine_runner.rs` | game installs its closure; closure may capture `vox_sim` state |
| `AgentComputeLayer::register_node()` | `EngineApp` construction | `crates/vox_app/src/bin/engine_runner.rs` | game registers domain nodes before any graph is compiled |
| `AgentComputeLayer::tick()` | `EngineApp::about_to_wait` render loop | `crates/vox_app/src/bin/engine_runner.rs` | before `queue.submit()`; same encoder as render pass |
| `AgentComputeLayer::show_editor()` | `EditorApp::show()` — "Agents" tab in `ContextPanel` | `crates/vox_app/src/editor_app.rs` | only when `WorkspaceMode::Simulate` is active |
| `AgentComputeLayer::submit_graph()` | compile button in `AgentNodeEditor` | `crates/vox_agent/src/editor.rs` | fires on user action |

`vox_agent` does not call into `vox_sim` or `vox_render`. All coupling is in `vox_app`.

---

## 8. Open Questions

- [ ] **Slang SPIR-V + wgpu feature flag:** wgpu requires the `spirv` feature flag and a Vulkan backend for SPIR-V shader loading. Confirm this is enabled in `vox_render`'s wgpu instance config, or add it. Metal and DX12 backends require SPIR-V → MSL/DXIL cross-compilation via wgpu's built-in Naga path.
- [ ] **CPU mirror strategy for AgentSlice:** Tier callbacks need a CPU-side copy of agent state. Preferred approach: maintain a permanent CPU mirror buffer updated via async GPU→CPU copy at tier rate (one readback per tier tick, never blocking the render thread). Alternative: blocking readback per callback invocation. Confirm before implementation — choice affects TierScheduler design.
- [ ] **Bind group layout validation:** When `load_shader()` is called, the engine needs to verify the SPIR-V's declared bindings match the layout derived from `AgentStateDesc`. wgpu surfaces this as a pipeline creation error; decide whether to add explicit pre-validation (via SPIR-V reflection) or rely on wgpu's error. Reflection is more actionable but requires a dependency (`rspirv` or `naga`).

---

## 9. Out of Scope

- **Any game-specific node type** (pathfinding, need mutation, event emission). These are registered by the game layer via `register_node()`.
- **CPU fallback for GPUs without compute support.** Ochroma requires wgpu compute. No software fallback.
- **Multi-GPU distribution.** Single GPU only.
- **Agent rendering.** This design covers simulation state only. Visual representation is `vox_render`'s responsibility.
- **Networking / replication of agent state.** GPU-resident state; network replication is out of scope.
- **Text-language scripting.** The scripting surface is the optional visual node graph. A text language is a future layer that targets the same `AgentNodeGraph` IR.
- **GPU-side pathfinding.** Path requests are handled CPU-side by the game's tier-2 callback.
- **Multiple simultaneous active graphs.** Phase 1 supports one compiled behavior per `AgentComputeLayer` instance.

---

## 10. Spectra Integration Note

The `spectral_samples` buffer passed to `tick()` will eventually come from Spectra (the CUDA path tracer). Three work items land on Spectra's side — **deferred to a separate session:**

1. **SpectralProbeGather pass** — after main render, a CUDA gather kernel samples the rendered spectral texture at each agent's screen-space position and writes a compact `N×16` float buffer. Screen-space first; sparse 3D probe grid is the upgrade path.
2. **CUDA → wgpu buffer transfer** — CPU readback/upload shim as first-pass (acceptable: agents update spectrally at 4Hz, so latency = 1 render frame ≈ 16ms). Zero-copy upgrade: `cudaExternalMemory` + `VK_KHR_external_memory`, following the existing DLSS interop pattern.
3. **32 → 16 band resampling** — Spectra's HWSS-4 accumulates 32 bands (380–780nm at 12.5nm). Agent buffer uses 16 bands (380–755nm at 25nm), matching `GaussianSplat::spectral`. Gather kernel resamples by linear interpolation at the 16 band centers.

**Until Spectra produces this buffer**, pass `None` to `tick()`. The spectral capability test uses a CPU-filled synthetic buffer — no Spectra dependency for initial implementation.

---

## 11. Related Plans / Designs

- Depends on: Domain 06 Rendering Plan (spectral samples buffer must be a `wgpu::Buffer` accessible to callers outside `vox_render`)
- Depends on: Domain 10 Physics Plan (spatial hash counting sort pattern from PBF solver)
- Required before: any design that adds per-agent visual scripting from the editor UI
- Related: Domain 11 AI/LLM Plan (LLM-generated node graphs target `AgentNodeGraph` IR)
- Related: Domain 12 Spectral Frontier Plan (spectral field queries in agents)
- Blocks: Spectra SpectralProbeGather session (see §10)
