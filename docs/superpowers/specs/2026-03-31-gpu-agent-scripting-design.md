# Design: GPU Agent Scripting Layer (2026-03-31)

**Status:** Draft
**Scope:** A visual node graph whose output compiles to WGSL/SPIR-V compute shaders, enabling 1M+ agents to update at 60fps on a consumer GPU, with spectral field and simulation state accessible as first-class node types.
**Related:** Domain 11 AI/LLM Plan, Domain 12 Spectral Frontier Plan

---

## 1. Problem Statement

- `vox_sim` citizen agents run on CPU, single-threaded ECS tick. Profiling shows the system saturates at ~50,000 agents before frame time exceeds 16.67ms on a desktop CPU.
- No path exists from the visual scripting infrastructure (`visual_graph.rs`, `node_graph_widget.rs`) to execution — the graph is a data model with no evaluator, no compiler, no runtime.
- Agent behavior is hardcoded in Rust. Changing a decision rule requires a recompile. There is no designer-facing tool for expressing agent logic.
- The spectral field (16-band GPU buffer, updated every frame) is inaccessible to agent behavior logic. Agents are blind to the light around them.
- The existing BDI system (`vox_core`) handles deliberative decisions well but runs at 60Hz — wasted CPU on decisions that only need to happen at 4Hz.

---

## 2. Done When

Running `cargo test -p vox_agent --test bench_million -- --nocapture` prints:

```
agents: 1_000_000  avg_frame_ms: <=16.0  min_fps: 60  dispatch: GPU
```

This test dispatches 1,000,000 agents for 120 consecutive frames on the GPU and asserts average frame time ≤ 16.0ms. The test runs headless (no window). Verified on RTX 3060 or equivalent (13 TFLOPS FP32, 360 GB/s bandwidth).

Additionally, `cargo run --bin ochroma` with 100,000 agents shows a working node graph editor where connecting a `SampleSpectralField` node to a `SetVelocity` node causes agents to visibly steer toward spectral hotspots in the viewport.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| 1M agents GPU dispatch ≤ 16ms | `bench_million` test asserts `avg_frame_ms <= 16.0` over 120 frames with 1M agents | `assert!(dispatch_returned_ok)` — passes with empty kernel |
| Node graph compiles to valid SPIR-V | `cargo test node_graph_compiles_to_spirv` passes a 10-node hunger-wander graph through codegen + Slang compiler, asserts output is non-empty valid SPIR-V bytes | `assert!(spirv_bytes.len() > 0)` with stub that returns `[0u8; 4]` |
| Spectral field readable in shader | `cargo test spectral_node_reads_live_field` dispatches 1000 agents with a `SampleSpectralField(band=5) → SetVelocity` graph, asserts agents near a spectral hotspot (band 5 > 0.5) have average velocity pointing toward it | `assert!(velocity != Vec3::ZERO)` — passes with random velocity |
| Neighbour query returns nearby agents | `cargo test neighbour_query_correctness` places 100 agents in known positions, asserts each agent's `QueryNeighbours(radius=10.0)` returns exactly the agents within 10m | `assert!(neighbours.len() > 0)` |
| Tier scheduling: GPU 60Hz, CPU 4Hz | `cargo test tier_scheduling` runs 1000 agents for 60 frames, asserts tier-1 GPU updates happened 60 times and tier-2 CPU updates happened 4 times (±1) | `assert!(tier2_ran)` — passes if it ran once |
| Node graph editor renders | `cargo test editor_renders_without_panic` runs egui in headless mode, creates a 5-node graph, calls `show()`, asserts no panic | test file compiles |
| Hot-swap compiled shader | `cargo test shader_hot_swap` compiles graph A, dispatches it, recompiles graph B, swaps, dispatches again, asserts agents respond to the new behavior within 1 frame | `assert!(swap_succeeded)` |

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

A new crate, not an extension of `vox_sim`. `vox_sim` remains the authoritative CPU simulation layer (tier 2 and 3 state, economy, pathfinding). `vox_agent` is the GPU fast path (tier 1 state, per-frame behavior).

`vox_agent` depends on: `vox_core`, `vox_render` (for spectral field buffer handles), `wgpu`, `shader-slang`, `egui`.
`vox_app` depends on `vox_agent` and wires it into `EngineApp`.

### 4.3 Agent State Model (SoA, GPU-resident)

Two memory domains: **Tier 1** lives on GPU (VRAM), updated every frame by compute shader. **Tier 2/3** lives on CPU (RAM), updated at 4Hz and 0.25Hz by the existing `vox_sim` systems.

**Tier 1 GPU buffers** (Structure of Arrays — mandatory for coalesced reads):

```
positions[N]         : [f32; 3]    — world-space position
velocities[N]        : [f32; 3]    — current velocity
needs[N * 8]         : f32         — indexed by NeedIndex enum
spectral_cache[N*16] : f32         — cached spectral field sample at agent position
flags[N]             : u32         — bits: agent_type[0..7], state[8..15], alive[16], needs_cpu[17]
spatial_cell[N]      : u32         — current spatial hash cell index
```

Total per agent: 12 + 12 + 32 + 64 + 4 + 4 = **128 bytes**

At 1M agents: 128 MB VRAM. Ping-pong = 256 MB. Within budget of RTX 3060 (12 GB VRAM).

**Tier 2 CPU state** is the existing `vox_sim::Citizen` struct. The agent index (`u32`) is the shared key between GPU tier-1 state and CPU tier-2 state.

### 4.4 Spatial Hash (GPU)

The spatial hash is rebuilt each frame in a three-pass compute pipeline:

1. **Count pass** — each agent atomically increments its cell's count
2. **Prefix sum pass** — exclusive scan over cell counts → cell offsets
3. **Scatter pass** — each agent writes its index into `cell_data[cell_offset[cell] + local_idx]`

This is identical to the pattern in `vox_physics/src/pbf.rs` (the PBF fluid solver). The implementation will be extracted into a shared `vox_gpu_util` crate or duplicated with adaptation.

Cell size: 10.0m × 10.0m. Grid: 4096 × 4096 cells covering a 40.96km × 40.96km world (matching LWC tile size).

### 4.5 AgentNodeGraph IR

A directed acyclic graph where nodes have typed input and output pins. Serializes to/from JSON.

Nodes have a `NodeKind` enum that determines what Slang fragment they emit during codegen. Connections are directed edges from output pin to input pin. The graph is validated (type-checked, cycle-detected) before codegen.

**Pin types:** `Float`, `Vec3`, `Bool`, `Int`, `Entity`, `SpectralValue` (f32×16 curve), `NeighbourList`, `Flow` (execution order).

**Node categories:**

- **Entry:** `OnUpdate { dt: Float }`, `OnSpectralThreshold { band: Int, threshold: Float }`, `OnNeedCritical { need: Int }`
- **Read:** `GetPosition → Vec3`, `GetVelocity → Vec3`, `GetNeed(index: Int) → Float`, `SampleSpectralField(offset: Vec3, band: Int) → Float`, `QueryNeighbours(radius: Float) → NeighbourList`, `NeighbourCount(list) → Int`, `NeighbourPosition(list, i: Int) → Vec3`, `NeighbourNeed(list, i: Int, need: Int) → Float`, `GetTime → Float`, `AgentId → Int`
- **Math:** `Add`, `Sub`, `Mul`, `Div`, `Lerp`, `Clamp`, `Normalize`, `Length`, `Distance`, `Select(Bool, a, b)`, `Noise(seed: Float) → Float`
- **Logic:** `Compare(op: CompareOp) → Bool`, `And`, `Or`, `Not`, `Branch(Bool) → Flow×2`
- **Write:** `SetVelocity(Vec3)`, `AddVelocity(Vec3)`, `SetNeed(index: Int, value: Float)`, `ModifyNeed(index: Int, delta: Float)`, `RequestPathTo(Vec3)`, `EmitEvent(type: Int, data: Float)`, `RequestCpuDecision`
- **Spectral:** `SpectralDot(a: SpectralValue, b: SpectralValue) → Float`, `SampleCurve(pos: Vec3) → SpectralValue`, `SpectralBand(curve: SpectralValue, band: Int) → Float`

### 4.6 Slang Codegen

`SlangCodegen::generate(graph: &AgentNodeGraph) -> Result<String, CodegenError>` performs:

1. Topological sort of nodes (Kahn's algorithm — already in `vox_editor::node_graph::OchromaNodeGraph`)
2. Allocate a temporary variable name per output pin (`let _v0`, `_v1`, …)
3. For each node in topological order, emit a Slang statement or expression
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
    RWStructuredBuffer<float>                      needs,        // [N * 8]
    StructuredBuffer<float>                        spectral_field, // [GRID * 16]
    StructuredBuffer<uint>                         spatial_cells,
    StructuredBuffer<uint>                         cell_offsets,
    RWStructuredBuffer<uint>                       agent_flags
)
```

### 4.7 Compilation Pipeline

**Build-time (default behaviors):** `slang-hal-build` in `vox_agent/build.rs` compiles `.slang` files in `assets/agents/` to SPIR-V at build time. Zero runtime compiler dependency for shipped behaviors.

**Runtime (user-authored node graphs):** When the user saves a node graph in the editor, `SlangCodegen::generate()` produces Slang source, which is compiled to SPIR-V via the `shader-slang` crate at runtime. Compilation is async (background thread). The old pipeline remains active until the new one is ready, then swapped atomically.

### 4.8 Tiered Simulation Scheduling

```
Frame N (16.67ms budget):
  GPU submit:   Spatial hash rebuild (3 passes) + Agent update (1 pass) — async compute queue
  CPU thread:   Tier-2 update for agents [N*K .. (N+1)*K] where K = agent_count/15
                (rotates through all agents in 15 frames ≈ 4Hz at 60fps)
  CPU thread:   Tier-3 update for agents [N*J .. (N+1)*J] where J = agent_count/240
                (rotates in 240 frames ≈ 0.25Hz at 60fps)
```

Tier-2 and tier-3 run on a Rayon thread pool, never blocking the render thread. They read from CPU-side `vox_sim` state. When a GPU agent sets the `needs_cpu` flag bit, the agent is added to the priority queue for tier-2 update next frame (before rotation would normally reach it).

### 4.9 AgentNodeEditor

An egui panel that wraps the existing `vox_ui::node_graph_widget::NodeGraphWidget`. It adds:

- A node palette sidebar (grouped by category) with drag-to-add
- Type-colored pin rendering (matching `VisualPinType` colors already defined)
- A compile button that triggers `SlangCodegen` + `shader-slang` compilation
- A status bar showing "Compiled OK — 1,247 agents responding" or compile errors

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
    pub fn connect(&mut self, src: NodeId, src_pin: PinId, dst: NodeId, dst_pin: PinId)
        -> Result<(), ConnectionError>;
    pub fn validate(&self) -> Result<(), Vec<ValidationError>>;
    pub fn topological_order(&self) -> Result<Vec<NodeId>, CycleError>;
}

/// SoA GPU state for tier-1 agents.
/// All buffers have length `agent_count` except `needs` (agent_count × 8)
/// and `spectral_cache` (agent_count × 16).
pub struct AgentStateBuffers {
    agent_count: u32,
    positions_a: wgpu::Buffer,    // ping
    positions_b: wgpu::Buffer,    // pong
    velocities_a: wgpu::Buffer,
    velocities_b: wgpu::Buffer,
    needs: wgpu::Buffer,          // RW, no ping-pong (needs persist)
    spectral_cache: wgpu::Buffer, // RW
    flags: wgpu::Buffer,          // RW
    spatial_cells: wgpu::Buffer,  // rebuilt each frame
    cell_offsets: wgpu::Buffer,   // rebuilt each frame
    read_index: u8,               // 0 = read A/write B, 1 = read B/write A
}

impl AgentStateBuffers {
    pub fn new(device: &wgpu::Device, agent_count: u32) -> Self;
    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn swap(&mut self) { self.read_index ^= 1; }
    pub fn read_positions(&self) -> &wgpu::Buffer;
    pub fn write_positions(&self) -> &wgpu::Buffer;
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
        spectral_field: &wgpu::Buffer,
        uniforms: AgentUniforms,
    );
}

/// Uniform data passed to the compute shader each frame.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AgentUniforms {
    pub agent_count: u32,
    pub dt: f32,
    pub time: f32,
    pub grid_width: u32,
    pub cell_size: f32,
    pub _pad: [f32; 3],
}

/// Codegen result.
pub struct SlangSource {
    pub source: String,
    pub entry_point: String,  // always "agent_update"
}

/// What AgentNodeKind emits during codegen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentNodeKind {
    // Entry
    OnUpdate,
    OnSpectralThreshold { band: u32, threshold: f32 },
    OnNeedCritical { need: u32 },
    // Read
    GetPosition,
    GetVelocity,
    GetNeed { index: u32 },
    SampleSpectralField { band: u32 },
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
    SetNeed { index: u32 },
    ModifyNeed { index: u32 },
    RequestPathTo,
    EmitEvent { event_type: u32 },
    RequestCpuDecision,
    // Spectral
    SpectralDot, SampleSpectralCurve, SpectralBand { band: u32 },
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
    spatial_hash_pipeline: AgentComputePipeline,  // built-in, not user-authored
    tier_scheduler: TierScheduler,
    editor: AgentNodeEditor,
}

impl AgentFabric {
    /// Allocate GPU buffers for `agent_count` agents. Call once at scene load.
    pub fn new(device: &wgpu::Device, agent_count: u32) -> Self;

    /// Submit a new node graph. Compiles async; old pipeline stays active until done.
    /// Threading: spawns a rayon task. Returns immediately.
    pub fn submit_graph(&mut self, graph: AgentNodeGraph);

    /// Call once per frame from the render loop.
    /// Rebuilds spatial hash, dispatches agent update, advances tier scheduler.
    /// Threading: GPU work submitted to `queue`; CPU tier work on rayon.
    pub fn tick(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        spectral_field: &wgpu::Buffer,
        dt: f32,
    );

    /// Render the node graph editor panel. Call inside egui frame.
    pub fn show_editor(&mut self, ui: &mut egui::Ui);

    /// Read agent positions back to CPU (async, for debug/test only — expensive).
    /// Returns a future that resolves when the GPU readback completes.
    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue)
        -> impl Future<Output = Vec<[f32; 3]>>;
}

/// Codegen — standalone, no GPU dependency.
/// crates/vox_agent/src/codegen.rs
pub struct SlangCodegen;

impl SlangCodegen {
    /// Convert a validated AgentNodeGraph to Slang source.
    /// Returns Err if the graph contains unsupported node combinations.
    /// Threading: pure function, no shared state, safe to call from any thread.
    pub fn generate(graph: &AgentNodeGraph) -> Result<SlangSource, CodegenError>;
}

/// Spatial hash rebuild — internal, but tested directly.
/// crates/vox_agent/src/spatial_hash.rs
pub fn rebuild_spatial_hash(
    encoder: &mut wgpu::CommandEncoder,
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
| `AgentFabric::tick()` | `EngineApp::about_to_wait` render loop | `crates/vox_app/src/bin/engine_runner.rs` | before `queue.submit()`; uses same encoder as render pass |
| `AgentFabric::show_editor()` | `EditorApp::show()` — new "Agents" tab in `ContextPanel` | `crates/vox_app/src/editor_app.rs` | only when `WorkspaceMode::Simulate` is active |
| `AgentFabric::submit_graph()` | `AgentNodeEditor` compile button callback | `crates/vox_agent/src/editor.rs` | fires on user action |
| `AgentStateBuffers::new()` | `AgentFabric::new()` | `crates/vox_agent/src/lib.rs` | |
| `SlangCodegen::generate()` | background rayon task in `submit_graph()` | `crates/vox_agent/src/lib.rs` | never on render thread |
| `rebuild_spatial_hash()` | `AgentFabric::tick()` | `crates/vox_agent/src/lib.rs` | first pass in tick, before agent update dispatch |
| `TierScheduler::tick()` | `AgentFabric::tick()` | `crates/vox_agent/src/lib.rs` | after GPU submit, on CPU; reads/writes `vox_sim` state |

The spectral field buffer (`vox_render`'s 16-band GPU buffer) is passed into `AgentFabric::tick()` as a `&wgpu::Buffer` handle. `vox_render` owns the buffer; `vox_agent` binds it read-only in the compute bind group.

---

## 8. Open Questions

- [ ] **Slang SPIR-V + wgpu feature flag:** wgpu requires the `spirv` feature flag and a Vulkan backend for SPIR-V shader loading. Confirm this is enabled in `vox_render`'s wgpu instance config, or add it. Metal and DX12 backends require SPIR-V → MSL/DXIL cross-compilation via wgpu's built-in Naga path.
- [ ] **Spatial hash grid world bounds:** The city sim world is currently unbounded. The spatial hash needs fixed world bounds. Agents outside bounds need a fallback (wrap, clamp, or skip). Decision needed before implementation.
- [ ] **`vox_gpu_util` extraction:** The spatial hash counting sort is duplicated from `vox_physics/src/pbf.rs`. Should this become a shared crate or stay duplicated? Shared crate is cleaner but adds a dependency edge.
- [ ] **Tier-2 state sync:** When the GPU changes an agent's position, tier-2 CPU logic needs to see the new position. Currently no GPU→CPU readback path exists for per-frame positions. Options: (a) readback a compressed summary, (b) let CPU tier-2 use its own position estimate, (c) full readback at 4Hz. Decision affects tier scheduler design.

---

## 9. Out of Scope

- **CPU fallback for GPUs that don't support compute shaders.** Ochroma requires wgpu with compute support. No software fallback.
- **Multi-GPU distribution.** Single GPU only. Distributed agent simulation is a separate future design.
- **Agent rendering.** This design covers simulation state only. How agents are visually represented (instanced splats, billboards, skeletal meshes) is handled by `vox_render` and is not changed here.
- **Networking / replication of agent state.** Agent positions are GPU-resident. Network replication of GPU state is not addressed here.
- **Full C# or text-language scripting.** The scripting surface is the visual node graph. Text language support is a future layer that can target the same `AgentNodeGraph` IR.
- **Agent pathfinding on GPU.** `RequestPathTo` queues a CPU pathfinding request. GPU-side A*/navmesh is out of scope for this design.
- **More than one active node graph simultaneously.** Phase 1 supports one compiled behavior per agent population. Per-agent-type graphs are a future extension.

---

## 10. Spectra Integration Note

The `spectral_field` buffer supplied to the agent compute kernel will eventually come from Spectra (the CUDA path tracer). Three concrete items land on Spectra's side — **deferred to a separate session:**

1. **SpectralProbeGather pass** — after main render, a CUDA gather kernel samples the rendered spectral texture at each agent's screen-space position and writes a compact `N_agents × 16` float buffer. Screen-space first; a sparse 3D probe grid is the upgrade path.
2. **CUDA → wgpu buffer transfer** — CPU readback/upload shim as first-pass (acceptable given agents update at 4Hz behavioral tier, so latency = 1 render frame = ~16ms). CUDA external memory → Vulkan (`cudaExternalMemory` + `VK_KHR_external_memory`) is the zero-copy upgrade, following the existing DLSS interop pattern.
3. **32 → 16 band resampling** — Spectra's HWSS-4 accumulates 32 bands (380–780nm at 12.5nm). The agent buffer uses 16 bands (380–755nm at 25nm), matching `GaussianSplat::spectral`. The gather kernel resamples by linear interpolation at the 16 band centers before writing.

**Until Spectra produces this buffer**, `vox_agent` accepts a wgpu buffer passed in from the caller. The integration test (`spectral_node_reads_live_field`) uses a CPU-filled test buffer — no Spectra dependency for the initial implementation.

---

## 11. Related Plans / Designs

- Depends on: Domain 06 Rendering Plan (spectral field buffer must be a `wgpu::Buffer` accessible outside `vox_render`)
- Depends on: Domain 10 Physics Plan (spatial hash counting sort pattern from PBF solver)
- Required before: any design that adds per-agent visual scripting from the editor UI
- Related: Domain 11 AI/LLM Plan (LLM-generated node graphs target `AgentNodeGraph` IR)
- Related: Domain 12 Spectral Frontier Plan (spectral field queries in agents)
- Blocks: Spectra SpectralProbeGather session (see §10)
