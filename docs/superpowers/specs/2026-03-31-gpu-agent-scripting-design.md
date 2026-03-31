# Design: GPU Agent Scripting Layer (2026-03-31)

**Status:** Draft
**Scope:** A visual node graph whose output compiles to SPIR-V compute shaders, enabling 1M+ agents to update at 60fps on a consumer GPU. Engine-level primitive — knows nothing about game-specific concepts (needs, pathfinding, event systems). Game layers extend via registered custom nodes and CPU callbacks.
**Related:** Domain 11 AI/LLM Plan, Domain 12 Spectral Frontier Plan

---

## 1. Problem Statement

- Large-scale agent simulation saturates CPU before reaching interesting densities. ~50,000 agents exhaust a single thread at 16.67ms on a desktop CPU.
- No path exists from the visual scripting infrastructure (`visual_graph.rs`, `node_graph_widget.rs`) to execution — the graph is a data model with no evaluator, no compiler, no runtime.
- Agent behavior is hardcoded in Rust. Changing a decision rule requires a recompile. There is no designer-facing tool for expressing agent logic.
- The spectral field (16-band GPU buffer produced by the render pipeline) is inaccessible to agent behavior logic.
- The existing BDI system (`vox_core`) handles deliberative decisions well but runs too frequently for its workload — decisions that only need to happen at 4Hz are running at 60Hz.

---

## 2. Done When

Running `cargo test -p vox_agent --test bench_million -- --nocapture` prints:

```
agents: 1_000_000  avg_frame_ms: <=16.0  min_fps: 60  dispatch: GPU
```

This test dispatches 1,000,000 agents for 120 consecutive frames on the GPU and asserts average frame time ≤ 16.0ms. The test runs headless (no window). Verified on RTX 3060 or equivalent (13 TFLOPS FP32, 360 GB/s bandwidth).

Additionally, `cargo run --bin ochroma` with 100,000 agents shows a working node graph editor where connecting a `SampleSpectralField` node to a `SetVelocity` node causes agents to visibly steer toward spectral hotspots in the viewport. The spectral hotspots are injected via a CPU-uploaded synthetic buffer — no Spectra dependency required.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| 1M agents GPU dispatch ≤ 16ms | `bench_million` asserts `avg_frame_ms <= 16.0` over 120 frames with 1M agents | `assert!(dispatch_returned_ok)` — passes with empty kernel |
| Node graph compiles to valid SPIR-V | `cargo test node_graph_compiles_to_spirv` passes a 10-node wander graph through codegen + Slang compiler, asserts output is non-empty valid SPIR-V bytes | `assert!(spirv_bytes.len() > 0)` with stub returning `[0u8; 4]` |
| Spectral field readable in shader | `cargo test spectral_node_reads_live_field` dispatches 1000 agents with a `SampleSpectralField(band=5) → SetVelocity` graph and a CPU-filled synthetic spectral buffer, asserts agents near a spectral hotspot (band 5 > 0.5) have average velocity pointing toward it | `assert!(velocity != Vec3::ZERO)` — passes with random velocity |
| Neighbour query returns nearby agents | `cargo test neighbour_query_correctness` places 100 agents in known positions, asserts each agent's `QueryNeighbours(radius=10.0)` returns exactly the agents within 10m | `assert!(neighbours.len() > 0)` |
| Tier scheduling: GPU 60Hz, CPU 4Hz | `cargo test tier_scheduling` runs 1000 agents for 60 frames, asserts tier-1 GPU updates happened 60 times and tier-2 CPU callbacks fired 4 times (±1) | `assert!(tier2_ran)` — passes if it ran once |
| Node graph editor renders | `cargo test editor_renders_without_panic` runs egui in headless mode, creates a 5-node graph, calls `show()`, asserts no panic | test file compiles |
| Hot-swap compiled shader | `cargo test shader_hot_swap` compiles graph A, dispatches it, recompiles graph B, swaps, dispatches again, asserts agents respond to the new behavior within 1 rendered frame *after the swap completes* | `assert!(swap_succeeded)` |
| Custom node extension | `cargo test custom_node_round_trips` registers a `CustomMultiply` node via `register_node_kind()`, builds a graph containing it, runs codegen, asserts the generated Slang contains the expected fragment | `assert!(slang.contains("custom"))` |
| Aux buffer read/write | `cargo test aux_buffer_rw` dispatches agents with a `ReadAux(slot=0) → AddVelocity` graph and non-zero aux data at slot 0, asserts agent velocities change proportionally | `assert!(velocities_changed)` — passes with any change |

---

## 4. Architecture

### 4.1 Overview

```
┌─────────────────────────────────────────────────┐
│  AgentNodeEditor  (egui, crates/vox_agent/src/editor.rs)
│  — visual node graph UI, produces AgentNodeGraph  │
└──────────────────┬──────────────────────────────┘
                   │ AgentNodeGraph (IR)
┌──────────────────▼──────────────────────────────┐
│  SlangCodegen  (crates/vox_agent/src/codegen.rs) │
│  — topological sort → emit Slang source text     │
└──────────────────┬──────────────────────────────┘
                   │ Slang source (.slang)
┌──────────────────▼──────────────────────────────┐
│  shader-slang Rust crate  (build-time or runtime)│
│  — compiles Slang → SPIR-V                       │
└──────────────────┬──────────────────────────────┘
                   │ SPIR-V bytes
┌──────────────────▼──────────────────────────────┐
│  AgentComputePipeline  (crates/vox_agent/src/compute.rs)
│  — wgpu ComputePipeline, bind groups, dispatch   │
└──────────────────┬──────────────────────────────┘
                   │ GPU execution
┌──────────────────▼──────────────────────────────┐
│  AgentStateBuffers  (crates/vox_agent/src/state.rs)
│  — SoA GPU buffers (ping-pong), spatial hash     │
└─────────────────────────────────────────────────┘
```

### 4.2 New Crate: `vox_agent`

A new engine crate, separate from `vox_sim`. `vox_sim` remains the authoritative CPU simulation layer for game-specific state (economy, pathfinding, BDI). `vox_agent` is the GPU fast path — it knows nothing about citizens, needs, or buildings.

`vox_agent` depends on: `vox_core`, `wgpu`, `shader-slang`, `egui`.

`vox_agent` does **not** depend on `vox_render` or `vox_sim`. The spectral field buffer is passed in as a `&wgpu::Buffer`; `vox_agent` binds it read-only without knowing how it was produced.

`vox_app` depends on `vox_agent` and wires it into `EngineApp`.

### 4.3 Agent State Model (SoA, GPU-resident)

Two memory domains: **Tier 1** lives on GPU (VRAM), updated every frame by compute shader. **Tier 2/3** live on CPU (RAM), serviced by game-provided callbacks at 4Hz and 0.25Hz.

**Tier 1 GPU buffers** (Structure of Arrays — mandatory for coalesced reads):

```
positions[N]          : [f32; 3]   — world-space position
velocities[N]         : [f32; 3]   — current velocity
aux[N * K]            : f32        — K caller-defined floats per agent (game layer assigns meaning)
spectral_cache[N * 16]: f32        — cached spectral field sample at agent position
flags[N]              : u32        — bits: agent_type[0..7], state[8..15], alive[16], needs_cpu[17]
spatial_cell[N]       : u32        — current spatial hash cell index
```

`K` (aux floats per agent) is set at construction time via `AgentFabric::new(device, agent_count, aux_per_agent)`. The engine imposes no meaning on aux slots — that is the game layer's responsibility.

Per-agent byte cost: 12 + 12 + (4×K) + 64 + 4 + 4 = **96 + 4K bytes**. At K=8: 128 bytes/agent. At 1M agents with K=8: 128 MB VRAM. Ping-pong = 256 MB. Within budget of RTX 3060 (12 GB VRAM).

**Tier 2/3 CPU state** is entirely game-owned. The engine exposes `AgentSlice` — a read-only CPU view of a contiguous range of agent positions, velocities, and flags — for the callbacks to consume.

### 4.4 Spatial Hash (GPU)

The spatial hash is rebuilt each frame in a three-pass compute pipeline:

1. **Count pass** — each agent atomically increments its cell's count
2. **Prefix sum pass** — exclusive scan over cell counts → cell offsets
3. **Scatter pass** — each agent writes its index into `cell_data[cell_offset[cell] + local_idx]`

This is identical to the pattern in `vox_physics/src/pbf.rs` (the PBF fluid solver). The implementation will be duplicated initially; extraction into a shared `vox_gpu_util` crate is the right long-term move but is deferred until `vox_physics` also needs the shared version.

Cell size: 10.0m × 10.0m. Grid: 4096 × 4096 cells covering a 40.96km × 40.96km world. Agents outside this bounds are clamped to the nearest edge cell. Grid dimensions are configurable at construction time.

Memory cost of the spatial hash structure: (4096×4096) × 8 bytes (count + offset) ≈ 134 MB, plus N × 4 bytes for agent index storage. This is fixed overhead regardless of agent count and should be documented to callers.

### 4.5 AgentNodeGraph IR

A directed acyclic graph where nodes have typed input and output pins. Serializes to/from JSON.

Nodes have a `NodeKind` that determines what Slang fragment they emit during codegen. Connections are directed edges from output pin to input pin. The graph is validated (type-checked, cycle-detected) before codegen.

**Pin types:** `Float`, `Vec3`, `Bool`, `Int`, `SpectralValue` (f32×16 curve), `NeighbourList`, `Flow` (execution order).

**Built-in node categories (engine-level only):**

- **Entry:** `OnUpdate { dt: Float }`, `OnSpectralThreshold { band: Int, threshold: Float }`
- **Read:** `GetPosition → Vec3`, `GetVelocity → Vec3`, `ReadAux(slot: Int) → Float`, `SampleSpectralField(offset: Vec3, band: Int) → Float`, `QueryNeighbours(radius: Float) → NeighbourList`, `NeighbourCount(list) → Int`, `NeighbourPosition(list, i: Int) → Vec3`, `GetTime → Float`, `AgentId → Int`
- **Math:** `Add`, `Sub`, `Mul`, `Div`, `Lerp`, `Clamp`, `Normalize`, `Length`, `Distance`, `Select(Bool, a, b)`, `Noise(seed: Float) → Float`
- **Logic:** `Compare(op: CompareOp) → Bool`, `And`, `Or`, `Not`, `Branch(Bool) → Flow×2`
- **Write:** `SetVelocity(Vec3)`, `AddVelocity(Vec3)`, `WriteAux(slot: Int, value: Float)`, `RequestCpuAttention`
- **Spectral:** `SpectralDot(a: SpectralValue, b: SpectralValue) → Float`, `SampleSpectralCurve(pos: Vec3) → SpectralValue`, `SpectralBand(curve: SpectralValue, band: Int) → Float`

**`RequestCpuAttention`** sets the `needs_cpu` flag bit. The engine's tier scheduler picks this up and includes the agent in the next CPU callback slice. What that callback does is entirely the game layer's business.

**Custom nodes** are registered via `AgentNodeRegistry::register(kind_name, SlangFragment)`. Registered nodes appear in the editor palette and participate in codegen. Game layers use this to add domain-specific nodes (pathfinding requests, need mutations, event emission) without touching `vox_agent`.

### 4.6 Slang Codegen

`SlangCodegen::generate(graph: &AgentNodeGraph, registry: &AgentNodeRegistry) -> Result<String, CodegenError>` performs:

1. Topological sort of nodes (Kahn's algorithm)
2. Allocate a temporary variable name per output pin (`let _v0`, `_v1`, …)
3. For each node in topological order, emit a Slang statement or expression (built-in or registered fragment)
4. Wrap in the standard compute entry point boilerplate with all required buffer bindings
5. Emit helper functions (`sample_spectral_field`, `query_neighbours`, `noise`) as a preamble

The generated Slang is human-readable. An engineer debugging a graph can read the output.

**Entry point signature (fixed, every graph):**

```slang
[numthreads(64, 1, 1)]
void agent_update(
    uint3 tid                                    : SV_DispatchThreadID,
    uniform AgentUniforms                          uniforms,
    StructuredBuffer<float3>                       positions_in,
    RWStructuredBuffer<float3>                     positions_out,
    StructuredBuffer<float3>                       velocities_in,
    RWStructuredBuffer<float3>                     velocities_out,
    RWStructuredBuffer<float>                      aux,              // [N * aux_per_agent]
    StructuredBuffer<float>                        spectral_samples, // [N * 16]
    StructuredBuffer<uint>                         spatial_cells,
    StructuredBuffer<uint>                         cell_offsets,
    RWStructuredBuffer<uint>                       agent_flags
)
```

### 4.7 Compilation Pipeline

**Build-time (default behaviors):** `slang-hal-build` in `vox_agent/build.rs` compiles `.slang` files in `assets/agents/` to SPIR-V at build time. Zero runtime compiler dependency for shipped behaviors.

**Runtime (user-authored node graphs):** When the user saves a node graph in the editor, `SlangCodegen::generate()` produces Slang source, which is compiled to SPIR-V via the `shader-slang` crate on a background rayon thread. The old pipeline remains active until the new one is ready, then swapped atomically. `AgentNodeGraph` and `AgentNodeRegistry` must be `Send`.

### 4.8 Tiered Simulation Scheduling

```
Frame N (16.67ms budget):
  GPU submit:   Spatial hash rebuild (3 passes) + Agent update (1 pass) — async compute queue
  CPU thread:   Tier-2 callback fires for agents [N*K .. (N+1)*K], K = agent_count/15
                (rotates through all agents in 15 frames ≈ 4Hz at 60fps)
  CPU thread:   Tier-3 callback fires for agents [N*J .. (N+1)*J], J = agent_count/240
                (rotates in 240 frames ≈ 0.25Hz at 60fps)
```

The engine calls the game-provided callbacks with an `AgentSlice` — a CPU-side read-only view of positions, velocities, flags, and aux data for that slice. The callback can read state and enqueue write-back commands via `AgentWriteQueue`. Write-back (CPU → GPU aux/flags upload) happens at the start of the next tick, before the compute dispatch.

Agents that set `needs_cpu` via `RequestCpuAttention` are prepended to the tier-2 queue ahead of the rotation schedule.

Tier-2 and tier-3 callbacks run on a Rayon thread pool and must not block the render thread. They receive no game engine context other than `AgentSlice` — any game state lookup (navmesh, economy) is the callback's own responsibility.

### 4.9 AgentNodeEditor

An egui panel that wraps the existing node graph widget in `vox_app::node_graph_panel`. It adds:

- A node palette sidebar (grouped by category: built-in and registered custom nodes) with drag-to-add
- Type-colored pin rendering (matching `VisualPinType` colors already defined)
- A compile button that triggers `SlangCodegen::generate()` + `shader-slang` compilation on a background thread
- A status bar showing "Compiled OK — 1,247 agents responding" or compile errors inline

The editor does not reinvent the node graph widget — it only adds the palette, compile action, and status display.

---

## 5. Data Models

```rust
// crates/vox_agent/src/node_graph.rs

/// A single node in the agent behavior graph.
pub struct AgentNode {
    id: NodeId,
    kind: AgentNodeKind,
    position: [f32; 2],   // editor canvas position
    inputs: Vec<PinId>,
    outputs: Vec<PinId>,
}

impl AgentNode {
    pub fn id(&self) -> NodeId { self.id }
    pub fn kind(&self) -> &AgentNodeKind { &self.kind }
    pub fn input_pins(&self) -> &[PinId] { &self.inputs }
    pub fn output_pins(&self) -> &[PinId] { &self.outputs }
}

/// The IR for a complete agent behavior program.
pub struct AgentNodeGraph {
    nodes: Vec<AgentNode>,
    connections: Vec<Connection>,  // (src_node, src_pin, dst_node, dst_pin)
    name: String,
}

impl AgentNodeGraph {
    pub fn new(name: impl Into<String>) -> Self;
    pub fn add_node(&mut self, kind: AgentNodeKind, position: [f32; 2]) -> NodeId;
    /// Structural connection only. Type checking happens in validate().
    pub fn connect(&mut self, src: NodeId, src_pin: PinId, dst: NodeId, dst_pin: PinId)
        -> Result<(), ConnectionError>;
    /// Type-checks all connections and detects cycles.
    pub fn validate(&self, registry: &AgentNodeRegistry) -> Result<(), Vec<ValidationError>>;
    pub fn topological_order(&self) -> Result<Vec<NodeId>, CycleError>;
}

/// Registry of built-in and game-registered custom node kinds.
pub struct AgentNodeRegistry {
    custom: HashMap<String, SlangFragment>,
}

impl AgentNodeRegistry {
    pub fn new() -> Self;
    /// Register a custom node kind. `fragment` is a Slang code template with
    /// `{input_N}` and `{output}` placeholders substituted during codegen.
    pub fn register(&mut self, kind_name: impl Into<String>, fragment: SlangFragment);
}

/// SoA GPU state for tier-1 agents.
pub struct AgentStateBuffers {
    agent_count: u32,
    aux_per_agent: u32,
    positions_a: wgpu::Buffer,       // ping
    positions_b: wgpu::Buffer,       // pong
    velocities_a: wgpu::Buffer,      // ping
    velocities_b: wgpu::Buffer,      // pong
    aux: wgpu::Buffer,               // RW, no ping-pong (aux persists across frames)
    spectral_cache: wgpu::Buffer,    // RW
    flags: wgpu::Buffer,             // RW
    spatial_cells: wgpu::Buffer,     // rebuilt each frame
    cell_offsets: wgpu::Buffer,      // rebuilt each frame
    read_index: u8,                  // 0 = read A/write B, 1 = read B/write A
}

impl AgentStateBuffers {
    pub fn new(device: &wgpu::Device, agent_count: u32, aux_per_agent: u32) -> Self;
    pub fn agent_count(&self) -> u32;
    pub fn aux_per_agent(&self) -> u32;
    pub fn swap(&mut self) { self.read_index ^= 1; }
    pub fn read_positions(&self) -> &wgpu::Buffer;
    pub fn write_positions(&self) -> &wgpu::Buffer;
    pub fn read_velocities(&self) -> &wgpu::Buffer;
    pub fn write_velocities(&self) -> &wgpu::Buffer;
}

/// Compiled and loaded compute pipeline ready for dispatch.
pub struct AgentComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    source_hash: u64,    // hash of Slang source — used for hot-swap detection
}

impl AgentComputePipeline {
    pub fn from_spirv(device: &wgpu::Device, spirv: &[u32]) -> Result<Self, PipelineError>;
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &AgentStateBuffers,
        spectral_samples: &wgpu::Buffer,
        uniforms: AgentUniforms,
    );
}

/// Three-pipeline bundle for the spatial hash rebuild.
pub struct SpatialHashPipelines {
    count: wgpu::ComputePipeline,
    prefix_sum: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

/// Uniform data passed to the compute shader each frame.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AgentUniforms {
    pub agent_count: u32,
    pub aux_per_agent: u32,
    pub dt: f32,
    pub time: f32,
    pub grid_width: u32,
    pub cell_size: f32,
    pub _pad: [f32; 2],
}

/// What AgentNodeKind emits during codegen (built-in nodes only).
/// Custom nodes are registered via AgentNodeRegistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentNodeKind {
    // Entry
    OnUpdate,
    OnSpectralThreshold { band: u32, threshold: f32 },
    // Read
    GetPosition,
    GetVelocity,
    ReadAux { slot: u32 },
    SampleSpectralField { band: u32 },
    QueryNeighbours { radius: f32 },
    NeighbourCount,
    NeighbourPosition { index: u32 },
    GetTime,
    AgentId,
    // Math
    Add, Sub, Mul, Div,
    Lerp, Clamp, Normalize, Length, Distance,
    Select, Noise,
    // Logic
    Compare { op: CompareOp },
    And, Or, Not, Branch,
    // Write
    SetVelocity, AddVelocity,
    WriteAux { slot: u32 },
    RequestCpuAttention,
    // Spectral
    SpectralDot, SampleSpectralCurve, SpectralBand { band: u32 },
    // Custom (game-registered)
    Custom { kind_name: String },
}
```

---

## 6. API

```rust
// Top-level entry point — owned by EngineApp, updated each frame.

/// crates/vox_agent/src/lib.rs
pub struct AgentFabric {
    buffers: AgentStateBuffers,
    pipeline: Option<AgentComputePipeline>,
    pending_pipeline: Option<Arc<Mutex<Option<AgentComputePipeline>>>>,
    spatial_hash: SpatialHashPipelines,   // built-in fixed WGSL shaders
    tier_scheduler: TierScheduler,
    editor: AgentNodeEditor,
    registry: AgentNodeRegistry,
}

impl AgentFabric {
    /// Allocate GPU buffers for `agent_count` agents with `aux_per_agent`
    /// caller-defined float slots. Call once at scene load.
    pub fn new(device: &wgpu::Device, agent_count: u32, aux_per_agent: u32) -> Self;

    /// Register a custom node kind available in the editor and codegen.
    pub fn register_node(&mut self, kind_name: impl Into<String>, fragment: SlangFragment);

    /// Submit a new node graph. Compiles async on a rayon thread;
    /// old pipeline stays active until compilation completes.
    pub fn submit_graph(&mut self, graph: AgentNodeGraph);

    /// Call once per frame from the render loop.
    /// Rebuilds spatial hash, dispatches agent update, fires tier callbacks.
    /// `spectral_samples`: N×16 float buffer, caller-owned, bound read-only.
    pub fn tick(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        spectral_samples: &wgpu::Buffer,
        dt: f32,
    );

    /// Register callbacks for tier-2 (4Hz) and tier-3 (0.25Hz) CPU updates.
    /// Each callback receives a slice of agent state and a write queue.
    pub fn set_tier2_callback(&mut self, cb: Box<dyn FnMut(AgentSlice, &mut AgentWriteQueue) + Send>);
    pub fn set_tier3_callback(&mut self, cb: Box<dyn FnMut(AgentSlice, &mut AgentWriteQueue) + Send>);

    /// Render the node graph editor panel. Call inside egui frame.
    pub fn show_editor(&mut self, ui: &mut egui::Ui);

    /// Read agent positions back to CPU. Expensive — for debug/test only.
    /// Requires the caller to poll `device` until the future resolves:
    ///   let fut = fabric.read_positions(device, queue);
    ///   device.poll(wgpu::Maintain::Wait);
    ///   let positions = fut.await;
    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue)
        -> impl Future<Output = Vec<[f32; 3]>>;
}

/// CPU-side read-only view of a contiguous slice of agent state,
/// provided to tier-2/3 callbacks. Backed by a CPU mirror updated at callback rate.
pub struct AgentSlice<'a> {
    pub agent_ids: &'a [u32],
    pub positions: &'a [[f32; 3]],
    pub velocities: &'a [[f32; 3]],
    pub flags: &'a [u32],
    pub aux: &'a [f32],          // length = slice_len * aux_per_agent
    pub aux_per_agent: u32,
}

/// Queues write-back commands from CPU callbacks to GPU buffers.
/// Applied at the start of the next tick, before compute dispatch.
pub struct AgentWriteQueue {
    // opaque; game layer calls methods, not fields
}

impl AgentWriteQueue {
    pub fn write_aux(&mut self, agent_id: u32, slot: u32, value: f32);
    pub fn write_flag_bits(&mut self, agent_id: u32, mask: u32, value: u32);
    pub fn write_velocity(&mut self, agent_id: u32, velocity: [f32; 3]);
}

/// Codegen — standalone, no GPU dependency.
/// crates/vox_agent/src/codegen.rs
pub struct SlangCodegen;

impl SlangCodegen {
    /// Convert a validated AgentNodeGraph to Slang source.
    /// `registry` provides Slang fragments for custom nodes.
    /// Threading: pure function, safe to call from any thread.
    pub fn generate(
        graph: &AgentNodeGraph,
        registry: &AgentNodeRegistry,
    ) -> Result<SlangSource, CodegenError>;
}

/// Spatial hash rebuild — internal, but tested directly.
/// crates/vox_agent/src/spatial_hash.rs
pub fn rebuild_spatial_hash(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SpatialHashPipelines,
    buffers: &AgentStateBuffers,
    grid_width: u32,
    cell_size: f32,
);
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `AgentFabric::new()` | `EngineApp` construction | `crates/vox_app/src/bin/engine_runner.rs` | after wgpu device init, before first frame |
| `AgentFabric::set_tier2_callback()` | game setup in `EngineApp` | `crates/vox_app/src/bin/engine_runner.rs` | game layer installs its callback here |
| `AgentFabric::register_node()` | game setup in `EngineApp` | `crates/vox_app/src/bin/engine_runner.rs` | game registers domain-specific nodes |
| `AgentFabric::tick()` | `EngineApp::about_to_wait` render loop | `crates/vox_app/src/bin/engine_runner.rs` | before `queue.submit()`; uses same encoder as render pass |
| `AgentFabric::show_editor()` | `EditorApp::show()` — "Agents" tab in `ContextPanel` | `crates/vox_app/src/editor_app.rs` | only when `WorkspaceMode::Simulate` is active |
| `AgentFabric::submit_graph()` | `AgentNodeEditor` compile button callback | `crates/vox_agent/src/editor.rs` | fires on user action |
| `rebuild_spatial_hash()` | `AgentFabric::tick()` | `crates/vox_agent/src/lib.rs` | first pass in tick, before agent update dispatch |
| `TierScheduler::tick()` | `AgentFabric::tick()` | `crates/vox_agent/src/lib.rs` | after GPU submit; dispatches CPU callbacks on rayon |

The spectral samples buffer (`N×16` floats) is passed into `AgentFabric::tick()` as a `&wgpu::Buffer`. `vox_agent` binds it read-only. Who fills it (Spectra, a CPU upload, a test fixture) is not `vox_agent`'s concern.

---

## 8. Open Questions

- [ ] **Slang SPIR-V + wgpu feature flag:** wgpu requires the `spirv` feature flag and a Vulkan backend for SPIR-V shader loading. Confirm this is enabled in `vox_render`'s wgpu instance config, or add it. Metal and DX12 backends require SPIR-V → MSL/DXIL cross-compilation via wgpu's built-in Naga path.
- [ ] **Spatial hash grid world bounds:** The spatial hash needs fixed world bounds at construction time. Agents outside bounds are clamped. The caller must pass `grid_origin: [f32; 2]` and `grid_extent: f32` to `AgentFabric::new()` — or use a sensible default (origin 0,0, extent 40.96km). Needs a decision before implementation.
- [ ] **Tier-2 CPU mirror strategy:** `AgentSlice` requires a CPU-side copy of agent state. Options: (a) maintain a permanent CPU mirror updated via GPU→CPU copy at tier rate, (b) do a blocking readback per callback invocation. Option (a) is preferred — one async readback per tier tick, never blocking the render thread. Needs confirmation before implementation.

---

## 9. Out of Scope

- **Game-specific node types** (pathfinding requests, need mutations, event emission) — registered by the game layer via `register_node()`. Not in `vox_agent`.
- **CPU fallback for GPUs that don't support compute shaders.** Ochroma requires wgpu with compute support. No software fallback.
- **Multi-GPU distribution.** Single GPU only.
- **Agent rendering.** This design covers simulation state only. Visual representation is handled by `vox_render`.
- **Networking / replication of agent state.** GPU-resident state; network replication is not addressed here.
- **Full C# or text-language scripting.** The scripting surface is the visual node graph. Text language support is a future layer targeting the same `AgentNodeGraph` IR.
- **GPU-side pathfinding.** Path requests from `RequestCpuAttention` callbacks are handled CPU-side by the game layer.
- **More than one active node graph simultaneously.** Phase 1 supports one compiled behavior per agent population.

---

## 10. Spectra Integration Note

The `spectral_samples` buffer supplied to the agent compute kernel will eventually come from Spectra (the CUDA path tracer). Three concrete items land on Spectra's side — **deferred to a separate session:**

1. **SpectralProbeGather pass** — after main render, a CUDA gather kernel samples the rendered spectral texture at each agent's screen-space position and writes a compact `N×16` float buffer. Screen-space first; a sparse 3D probe grid is the upgrade path.
2. **CUDA → wgpu buffer transfer** — CPU readback/upload shim as first-pass (acceptable given agents update spectrally at 4Hz behavioral tier, so latency = 1 render frame ≈ 16ms). CUDA external memory → Vulkan (`cudaExternalMemory` + `VK_KHR_external_memory`) is the zero-copy upgrade, following the existing DLSS interop pattern.
3. **32 → 16 band resampling** — Spectra's HWSS-4 accumulates 32 bands (380–780nm at 12.5nm). The agent buffer uses 16 bands (380–755nm at 25nm), matching `GaussianSplat::spectral`. The gather kernel resamples by linear interpolation at the 16 band centers before writing.

**Until Spectra produces this buffer**, `vox_agent` accepts any caller-supplied `wgpu::Buffer`. The integration test (`spectral_node_reads_live_field`) uses a CPU-filled synthetic buffer — no Spectra dependency for the initial implementation.

---

## 11. Related Plans / Designs

- Depends on: Domain 06 Rendering Plan (spectral samples buffer must be a `wgpu::Buffer` accessible outside `vox_render`)
- Depends on: Domain 10 Physics Plan (spatial hash counting sort pattern from PBF solver)
- Required before: any design that adds per-agent visual scripting from the editor UI
- Related: Domain 11 AI/LLM Plan (LLM-generated node graphs target `AgentNodeGraph` IR)
- Related: Domain 12 Spectral Frontier Plan (spectral field queries in agents)
- Blocks: Spectra SpectralProbeGather session (see §10)
