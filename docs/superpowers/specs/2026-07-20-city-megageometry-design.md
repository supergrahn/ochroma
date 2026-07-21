# Design: City MegaGeometry (2026-07-20)

**Status:** Draft; GPU-ownership, portable-backend, ray-native, and certified-visibility-fabric breakthrough review completed 2026-07-20
**Scope:** Forge geometry cook, Urban Horizon payload/wiring, Ochroma resident bridge, and Spectra scene state/renderer/OptiX.
**Related:** [Implementation plan](../plans/2026-07-20-city-megageometry.md)
**Owners:** Ochroma, Spectra, Forge
**Supersedes:** The CLAS and MegaGeometry portions of `2026-06-16-optix-hwrt-traversal.md`, `2026-06-24-beat-nvidia-integration-layer.md`, and `docs/spec/nvidia-next-gen-integration.md`

## 1. Summary

City MegaGeometry is a deterministic, resident geometry system for Urban Horizon. It uses NVIDIA hardware primitives where they help, but it does not define success as reproducing an NVIDIA sample.

The product advantage is the complete city workload:

- static repeated geometry stays in compacted triangle GAS when that traces faster;
- deforming, streamed, or continuously changing geometry may use true OptiX CLAS;
- foliage keeps hardware opacity micromaps;
- instance changes remain resident, id-sorted GPU deltas;
- large worlds use stable top-level partitions;
- continuous LOD and geometry pages are cooked by Forge, selected on GPU, and consumed without runtime QEM;
- supported Forge construction DAGs compile to a ray-native visibility IR whose bounded primitives are intersected directly behind native AABBs, without first materializing triangles or CLAS;
- the preferred high-volume execution path is a GPU-owned factorized visibility fabric: rays are routed and queued by spatial domain plus primitive algebra, then processed as coherent packets; native single-ray custom intersection is the low-fill baseline, not the assumed endpoint;
- Forge preserves construction provenance as portable geometry programs: shared topology dictionaries, quantized parameter/delta streams, bounded schedule candidates, and parent/child surface correspondence rather than flattening every asset permanently into unrelated triangles;
- GPU ray feedback drives advisory detail residency, while deterministic integer control and always-resident parents retain correctness;
- GPU kernels generate CLAS build arguments, LOD cuts, page requests, eviction choices, and dirty-partition work packets;
- one backend-neutral GPU work graph and stable buffer ABI feed separate compute and ray-acceleration adapters;
- ordinary subgroup/SIMD compute is the mandatory portable baseline, while Slang cooperative matrices are an optional measured acceleration path for naturally matrix-shaped work;
- the CPU is excluded from geometry decisions and scene walks; it only submits APIs and transports opaque page/build packets where OptiX or the operating system requires a host call;
- every backend reports the acceleration structure it actually built.

The initial NVIDIA truth path remains OptiX. OptiX does not expose `VK_NV_partitioned_acceleration_structure`, so this design uses a root IAS containing child partition IASes. Vulkan KHR/NV, Metal, and later DXR adapters consume the same common work records but retain their native acceleration-structure semantics; they are never disguised as OptiX features.

## 2. Problem Statement

The current code has useful pieces but no truthful end-to-end MegaGeometry system:

1. `OptixDevice::build_clas_scene` currently builds ordinary per-prototype triangle GAS and reports them as CLAS.
2. A real `optixClusterAccelBuild` implementation exists but is not called by the live scene path.
3. The existing million-instance benchmark proves shared prototypes, resident instances, and IAS refit scaling. Its reported `clusters=1` does not prove CLAS.
4. Runtime LOD is discrete per-prototype selection and is disabled in interactive rendering while cooked assets are awaited.
5. Current “streaming” is handle bookkeeping, not a budgeted RAM-to-VRAM geometry pager.
6. Current top-level traversal is a single-level IAS-to-GAS graph. It has no partitioned update structure.
7. Forge emits full-detail source clusters as untyped JSON, without the typed cluster-group, LOD, page, and provenance contract required by the renderer.
8. The current design has CUDA, Vulkan, and Metal implementations of `GpuBackend`, but no immutable capability fingerprint, cooperative-matrix shape/type inventory, common geometry packet ABI, or cross-backend conformance gate. Without those, portable code can silently become CUDA-shaped.
9. Existing cooperative/neural shader experiments are not a reusable contract and do not prove that matrix hardware executed, matched the baseline, or improved the full frame.
10. Forge already retains typed `BlueprintGraph` construction, assembly instances, subdivision cages, descriptions, sweeps, extrusions, and repeated façade modules, but the renderer receives the evaluated mesh rather than an executable visibility representation. The most valuable structure is lost before ray tracing begins.

## 3. Goals and Non-goals

### Goals

- Build and trace real OptiX CLAS through the live Spectra present path.
- Preserve triangle material, UV, weathering, and semantic identity after spatial cluster reordering.
- Select ray-native procedural AABBs, triangle GAS, or CLAS per cooked domain from measured workload characteristics.
- Update local instance partitions without rebuilding or refitting the entire million-instance top level.
- Cook deterministic, crack-free continuous cluster LOD and immutable geometry pages in Forge.
- Enforce a fixed VRAM budget without CPU geometry decisions, CPU scene walks, runtime mesh repair, or synchronous readback.
- Feed `optixClusterAccelBuild` with device-resident argument arrays and a device-resident argument count generated by GPU compute.
- Keep unavoidable OptiX IAS submission and storage I/O in bounded asynchronous host services outside the render thread.
- Produce a reproducible city-workload comparison against the ordinary triangle path and a reference spatial cluster schedule.
- Keep the AMD path functional and at or above the Performance-tier 30 fps gate.
- Extend the existing `spectra_gpu::GpuBackend` boundary with truthful immutable capability discovery for CUDA, Vulkan, and Metal instead of adding a second generic compute abstraction.
- Define backend-neutral geometry work records and buffer handles, then lower them on GPU into OptiX, Vulkan, or Metal acceleration-structure commands.
- Use Slang `linalg::CoopMat` as the portable cooperative-matrix source layer; select it only for measured matrix-shaped kernels and retain an ordinary subgroup/SIMD implementation for every required geometry stage.
- Prove cross-backend control-plane determinism, shader-layout compatibility, lack of CPU fallback, and bounded adapter overhead.
- Dual-lower supported Forge construction nodes into both the existing mesh oracle and a bounded ray-native visibility IR, then prove hit/material/UV/normal/surface identity between them.
- Keep supported parametric surfaces encoded through traversal so their triangles, fine-detail AS builds, topology-changing LOD, and parameter-only rebuilds are eliminated rather than merely accelerated.
- Convert programmable-intersection divergence into explicit GPU scheduling: first certify coherent ray domains against factorized surfaces, then bin only unresolved rays by cell/family, evaluate ray-by-surface coefficient tiles with subgroup/cooperative kernels, and send irregular residual queues to native RT traversal.
- Beat raw independent cluster storage with a portable geometry-program representation that exploits repeated topology within and across authored assets, while retaining a dense-block fallback for unique geometry.
- Preserve surface identity across topology-changing LOD so ReSTIR/temporal reconstruction can reuse valid samples instead of discarding history at every cut change.
- Cook a bounded Pareto portfolio of schedules and select among them at structural admission using measured build, trace, decode, residency, and update costs; never optimize only triangle count.

### Non-goals

- Reimplement NVIDIA hardware, drivers, or OptiX.
- Claim CLAS is faster for static geometry without measurement.
- Add a raster or CPU world renderer.
- Claim “zero CPU” while ordinary OptiX IAS or operating-system storage APIs still require host calls.
- Let a host worker recalculate LOD, visibility, page priority, eviction, partition membership, or cluster topology.
- Put camera-driven asset substitution, runtime QEM, or geometry repair in Urban Horizon gameplay code.
- Call hierarchical OptiX IAS “PTLAS.”
- Silently add or substitute Vulkan PTLAS inside the OptiX scope; a PTLAS backend requires a separate architectural decision and live present proof.
- Change the visual design of any shipped building, prop, terrain, or vegetation asset.
- Route scans, sorting, allocation, BVH traversal, or arbitrary geometry work through matrix hardware merely to claim tensor-core use.
- Depend on cuBLAS, cuDNN, MPSGraph, or another vendor algorithm library for the common geometry control plane. Platform compilers, drivers, and native RT/build APIs remain required.
- Advertise a CUDA, Vulkan, Metal, or future DXR adapter as supported before its hardware conformance and live present witnesses pass.
- Require every asset to use procedural/topology reuse. Unique meshes must retain a competitive dense-block path.
- Interpret arbitrary BlueprintGraph bytecode or compile per-asset shaders at runtime. The visibility IR is a finite validated data language consumed by AOT primitive-family kernels.
- Claim ray-native geometry wins merely because it materializes fewer triangles. It must beat the complete native triangle/CLAS path after AABB build, intersection-shader occupancy/divergence, shading, and full-frame cost.
- Let ray feedback become the sole source of required visibility or residency. It is delayed advisory evidence, not a correctness oracle.

## 4. Done When

City MegaGeometry is complete only when all of these are true:

1. A hardware witness reports `accel=optix_clas`, `hardware_clas_builds>0`, `clusters>prototypes`, and `mapped_primitives=source_primitives`.
2. A multi-material fixture reports zero material, UV, or source-triangle mapping mismatches between triangle GAS and CLAS hit buffers.
3. Static prototypes choose compacted triangle GAS unless CLAS wins the stored calibration; the static trace path regresses by no more than 2%.
4. A deforming or streamed fixture has lower p95 geometry update time with CLAS/templates than rebuilding its triangle GAS, with no correctness difference.
5. A one-percent-dirty million-instance run updates only affected child IAS partitions and is at least four times faster than full flat-IAS refit, with p95 at or below 2 ms on the reference NVIDIA box.
6. Continuous cluster LOD reports no open boundary seams, no missing required page, and at least 50% fewer traced triangles on its long-range fixture while remaining within the agreed one-pixel screen-space error.
7. A 600-frame paging run stays within its configured geometry budget and reports zero synchronous readbacks and zero required-page misses after warmup.
8. A GPU-ownership witness reports zero render-thread scene/instance/cluster walks, zero CPU LOD/page/eviction decisions, zero blocking readbacks, zero per-frame geometry allocations, and no CPU geometry cost that scales with total scene size.
9. Fixed render-thread geometry API submission remains below 0.25 ms p95 and does not scale with total instances/clusters/pages. The bounded IAS host-submission island is reported separately and remains below 0.50 ms p95 on the localized one-percent-dirty NVIDIA fixture. Storage transport is asynchronous and absent from render-thread timing.
10. The real-time 1 spp present path renders the product witness. AMD 780M Performance tier remains at or above 30 fps.
11. The final comparison reports zero losing fixtures, parity on static/OMM/correctness-only gates, and `city_hybrid` wins on changing geometry, streamed continuous LOD, localized million-instance updates, and the mixed live city. If it does not, the system ships the best measured subset and makes no broad “better city geometry system” claim.
12. CUDA and Vulkan print immutable capability fingerprints, execute the same backend-neutral geometry packet ABI, and produce identical integer control hashes. Cooperative-matrix outputs stay within the declared per-kernel tolerance, never drive persistent LOD/page/topology decisions, and are selected only when their complete stage is no more than 2% slower than the portable subgroup baseline.
13. The adapter layer adds no readback or per-dispatch allocation, and its GPU-stage time is within 2% of an equivalent backend-native kernel after warmup. Metal is reported as `experimental` until the same hardware conformance and live present witness run on a named Apple Silicon device.
14. The repeated-structure fixture stores geometry-program pages in at most 25% of the raw independent-cluster bytes, the mixed city uses at most 50%, decoding/building does not regress full-frame p95 by more than 2%, and the same VRAM/frame budget exposes at least twice the source-equivalent geometric detail within the one-pixel error gate.
15. Parent/child surface correspondence covers every accepted LOD primitive without ambiguity, reduces LOD-switch temporal/reservoir invalidations by at least 50% against the no-map path, and introduces no hit, material, UV, motion-vector, or radiance-oracle mismatch.
16. The broad breakthrough claim is emitted only if the geometry-program path has zero losing fixtures and wins at least three independent axes among storage bytes, peak VRAM, geometry-stage p95, visible detail at fixed budget, and LOD-switch temporal quality. Otherwise it remains an experimental subset.
17. On at least two real Forge-authored structural asset families, ray-native visibility covers at least 60% of source-equivalent visible detail, reduces materialized fine-geometry bytes to at most 25%, performs zero fine-geometry AS builds for parameter-only updates inside admitted bounds, produces zero hit/material/UV/normal/surface-key mismatches, and improves complete geometry-stage p95 by at least 1.5x over the winning triangle/CLAS representation.
18. The ray-native breakthrough claim additionally requires at least four times the oracle-valid visible detail at the same peak-VRAM/frame-time budget and zero topology/LOD temporal invalidations for ray-native surfaces. If those gates fail, topology programs and dense blocks remain independently shippable and no ray-native claim is emitted.
19. The stronger visibility-fabric claim requires at least 60% of traced structured rays to use the fabric, at least 35% of all measured rays to be resolved by exact domain certificates without per-ray traversal, at least 75% active lanes in the remaining explicit-ray packets, at least 2x intersection-stage speedup over native scalar custom intersection, at least 2x complete geometry-stage speedup over the winning native triangle/CLAS/custom path, no more than 10% routing/queue overhead, zero oracle mismatches, zero CPU scheduling, and zero losses across two real asset families.

The final authorized NVIDIA command is:

```powershell
cargo run --release -p vox_render --example mega_geometry_bench --features spectra-native,spectra-native-optix -- --suite city --modes triangle,reference,city_hybrid --warmup 60 --frames 240 --json-out artifacts/megageometry/final-nvidia.json
```

It must print `MEGAGEOMETRY_OVERALL winner=city_hybrid losses=0 target_wins=4 claim=better_city_geometry_system`. The authorized AMD product witness must print `performance_fps_p50>=30.0 runtime_qem_calls=0 missing_geometry=0`. A human must also inspect both live real-time present captures.

When the stronger breakthrough claim is requested, the same artifact set must additionally print `MEGAGEOMETRY_BREAKTHROUGH geometry_program=pass repeated_storage_ratio<=0.25 mixed_storage_ratio<=0.50 fixed_budget_detail_ratio>=2.0 lod_temporal_invalidations_reduction>=0.50 full_frame_regression_pct<=2.0 winning_axes>=3 losses=0 claim=city_geometry_program_breakthrough`. Absence of that line narrows the claim; it does not permit weakening any individual threshold.

The larger ray-native claim is separate and must print `MEGAGEOMETRY_VISIBILITY_BREAKTHROUGH ray_native_coverage>=0.60 materialized_geometry_ratio<=0.25 parameter_update_fine_as_builds=0 geometry_stage_speedup>=1.50 fixed_budget_detail_ratio>=4.0 ray_native_temporal_invalidations=0 correctness_mismatches=0 real_asset_families>=2 losses=0 claim=ray_native_city_geometry`. Its absence does not invalidate a narrower geometry-program or core MegaGeometry result.

The strongest bulk-execution claim must print `MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage>=0.60 certified_without_per_ray_traversal>=0.35 active_packet_lanes>=0.75 intersection_speedup_vs_scalar>=2.0 geometry_stage_speedup>=2.0 routing_queue_overhead_pct<=10.0 correctness_mismatches=0 cpu_scheduling=0 real_asset_families>=2 losses=0 claim=certified_visibility_fabric`. It is independent of the scalar ray-native claim: failure retains the measured native custom-intersection or residual path.

## 5. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Truthful AS identity | Forced CLAS exits nonzero unless `hardware_clas_builds>0` | Returning prototype or instance count as cluster count |
| Primitive provenance | Reordered multi-material clusters produce zero packed-triangle/material/UV mismatches | Checking only that a mapping buffer exists |
| Deterministic Forge cook | Two independent cooks are byte-identical and cover every source triangle exactly once | `assert!(clusters.is_some())` |
| Hybrid AS selection | Static fixture selects measured triangle GAS; changing fixture selects CLAS only after an update win | Always returning CLAS from `Auto` |
| GPU-owned frame preparation | Device kernels emit CLAS args/count, LOD cuts, dirty partitions, and page work while CPU decision counters remain zero | A Rust loop that fills “GPU” buffers |
| Local top-level updates | Localized one-percent-dirty million-instance witness updates only dirty child IASes with stable root handle and bounded host submission | CPU traversal over all instances or partitions |
| Continuous cluster LOD | GPU-selected real present witness has zero open seams, at most one-pixel error, and at least 50% fewer traced triangles | CPU selection or whole-mesh distance substitution |
| Geometry paging | GPU-managed fixed page pool runs 600 frames with no required miss, CPU eviction decision, or synchronous readback | A CPU pager or handle map called streaming |
| Portable GPU contract | CUDA/Vulkan adapters consume the same packet bytes and emit the same integer control hash | Separate vendor structs populated by CPU loops |
| Cooperative compute | Capability-checked Slang cooperative kernel matches its subgroup oracle and is selected only on an end-to-end stage win | Checking an extension bit or calling a kernel `tensor` |
| Adapter honesty | Result names API, RT kind, compute variant, capability hash, and actual fallback | Inferring hardware use from the GPU vendor name |
| Geometry programs | Repeated topology plus quantized parameters reconstruct exact accepted blocks with 4x storage reduction on the repetition fixture | Compressing a duplicated triangle buffer with a general byte codec |
| Ray-native visibility | Real BlueprintGraph nodes intersect through native AABB/custom-primitive traversal with exact mesh-oracle hits and no fine triangle materialization | Rendering a coarse proxy, SDF impostor, or CPU-generated procedural mesh |
| Certified visibility fabric | Exact ray-domain certificates eliminate per-ray traversal where possible; GPU queues group unresolved rays by cell and primitive algebra; packet kernels beat scalar custom intersection including routing/compaction cost | Timing matrix instructions while excluding certification, subdivision, queue, reduction, residual, or empty-packet overhead |
| Stable parametric surfaces | Construction-path surface keys and coordinates survive tolerance changes and parameter updates with zero LOD invalidation | Reconstructing identity from world position or resetting history |
| Schedule co-design | Bounded cooked candidates are measured on build, trace, decode, paging, and memory; admission selects the Pareto winner | Picking the hierarchy with the fewest triangles |
| Surface-stable LOD | Parent/child mapping preserves hits and at least halves temporal invalidations across topology changes | Resetting ReSTIR/TAA history whenever LOD changes |
| Ray-feedback residency | Quantized delayed hit/footprint feedback improves optional prefetch while required parents remain independently safe | Treating last frame's visibility as current-frame truth |
| Evidence-limited claim | Overall winner is suppressed on any correctness failure or losing fixture | Printing a fixed winner string |

## 6. Architecture

### 6.0 CPU exclusion law

The live geometry frame has one owner for decisions: the GPU. The CPU must not, per frame:

- walk instances, prototypes, visibility nodes, clusters, groups, or pages;
- interpret BlueprintGraph, evaluate construction, compute projected error, or choose an LOD/tolerance;
- sort or deduplicate dirty partitions;
- prioritize page requests or choose eviction victims;
- generate CLAS argument arrays or read back their count;
- allocate or free geometry pages;
- synchronously read geometry counters before present.

Two bounded host services remain because the shipping APIs require them:

1. ordinary OptiX IAS builds are host-submitted;
2. file I/O and CUDA/Vulkan transfer submission are host APIs.

These services consume opaque, GPU-generated work packets. They may validate packet bounds/generations, submit the named API operation, and publish completion tokens. They may not reinterpret camera, geometry, visibility, priority, or partition policy. They run outside the render thread and are measured separately.

As with every renderer, the host also submits the fixed compute/ray pass graph. That is API invocation, not scheduling ownership: the pass sequence and admitted launch bounds do not vary with GPU queue contents, and the host never reads a count or hit to choose the next pass. Backends with device-indirect dispatch consume device counts; backends without it pay a measured fixed-capacity launch with device-side early exit.

Static prototype admission, Forge payload validation, device calibration, buffer allocation, and pipeline creation are load/structural work and may use CPU. They never run as a hidden per-frame fallback.

### 6.1 Three orthogonal execution planes

MegaGeometry does not have one “tensor backend.” It has three deliberately separate planes:

```text
backend-neutral geometry work graph and resident buffers
  -> deterministic control compute: subgroup/SIMD scans, sorts, selection, allocation
  -> optional cooperative compute: dense transforms, codecs, inference, reconstruction
  -> native ray acceleration: triangle/cluster AS build, update, and traversal
```

The control plane is required everywhere and has a bit-exact integer/fixed-point contract. The cooperative plane is optional and cannot become a prerequisite for correctness. The ray plane owns native acceleration-structure objects and commands; cooperative matrices are not treated as an RT-core replacement.

The existing `spectra_gpu::GpuBackend` remains the common compute/resource boundary. It gains immutable `GpuCapabilities` rather than being wrapped by a competing generic interface. A separate `GeometryAccelBackend` consumes common GPU buffers and lowers backend-neutral build records into OptiX, Vulkan, Metal, or later DXR commands. Common code never carries an `Optix*`, `Vk*`, `MTL*`, raw device pointer, or native queue type.

`GeometryAccelKind` identifies the structure (`Triangle`, `Cluster`, or `Procedural`); `RayAccelBackendKind` identifies the implementation (`Optix`, `VulkanKhr`, `VulkanNv`, `Metal`, or `Dxr`). Keeping these axes separate prevents “cluster means OptiX” or “procedural means software traversal” from leaking into scene state, policies, payloads, or benchmark claims.

### 6.2 Capability-driven cooperative compute

Slang `linalg::CoopMat` is the source-level cooperative-matrix abstraction. Backends compile it to their supported native target: CUDA WMMA/MMA-family instructions, Vulkan `VK_KHR_cooperative_matrix`/SPIR-V cooperative matrices, or Metal SIMD-group matrix operations. Spectra does not invent a portable fragment layout and does not pass opaque matrix fragments across modules or backend boundaries.

This is specialization, not source-level uniformity. Current Slang Metal cooperative matrices have a much narrower 8x8 half/float operation set than CUDA/Vulkan and omit per-element access, transpose, reductions, and several helpers. Kernels use only the intersection required by their selected specialization; unsupported operations compile out to a separate subgroup variant rather than being emulated.

Every device reports an immutable, hashed capability set at initialization:

- API/backend and shader target versions;
- subgroup/SIMD width and supported scopes;
- cooperative M/N/K shapes, input/accumulator types, layouts, alignment, saturation, and transpose support;
- indirect dispatch, device address, sparse residency, hardware RT, procedural-AABB/custom-intersection support, shader/invocation reordering, indirect/address-driven AS build, and queue/timeline features;
- driver/compiler identity used by the pipeline-cache key.

The renderer selects a precompiled kernel variant at pipeline admission. Required variants are `PortableSubgroup` and zero or more exact cooperative shape/type specializations. There is no virtual call per tile, frame-time JIT, CPU emulation, or silent matrix emulation. Unsupported devices select the ordinary GPU path.

Only naturally matrix-shaped work is eligible: batched affine/bounds transforms, learned or block-linear geometry decompression, compact inference, factorized ray-by-plane/transform/polynomial coefficient tiles, ray-coherence scoring, and reconstruction. Prefix scans, radix sorting, dirty-bit compaction, free-list/page allocation, pointer chasing, domain subdivision, and BVH traversal remain ordinary compute unless a full-stage benchmark proves otherwise.

### 6.3 Determinism, ABI, compilation, and synchronization

Persistent topology, group cuts, partition ids, page requests, eviction order, and native build counts are produced by integer/fixed-point kernels with stable id tie-breaks. Floating cooperative results may estimate or rank advisory work, but the final persistent decision is quantized and resolved by the deterministic control path. Cross-backend tests require identical control-buffer bytes; render-only or codec outputs use explicit error bounds.

Common records use fixed-width fields, explicit alignment, version, byte length, capacity, generation, and content/ABI hashes. Rust layout tests and Slang reflection validate offsets for every target. Native adapters resolve `GpuBufferHandle` to their own addresses only at the final lowering boundary. No raw `u64` device addresses enter scene state or asset payloads.

All shipping variants are AOT-compiled or loaded during structural admission, prewarmed, and keyed by shader source, reflection ABI, capability fingerprint, driver/compiler, and cook schema. Each kernel family has a small explicit specialization manifest; capability enumeration does not generate a Cartesian product of every shape/type/scope. Cache entries are rejected rather than reused across an incompatible fingerprint. Compilation and pipeline creation cannot occur after the first timed/present frame.

The work graph uses backend-native queues and timeline/fence primitives behind opaque queue/epoch handles. Buffer ownership, transitions, AS-build barriers, scratch aliasing, and multi-frame eviction safety are explicit. Cross-API CUDA/OptiX sharing must use one CUDA context and stream-ordered events; Vulkan and Metal stay within their native resource domains. The host never polls a count to decide the next dispatch.

### 6.4 Truthful acceleration kinds

The renderer exposes the structure it built, not the name the caller requested:

```rust
pub enum GeometryAccelKind {
    Triangle,
    Cluster,
    Procedural,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RayAccelBackendKind {
    Optix,
    VulkanKhr,
    VulkanNv,
    Metal,
    Dxr,
    Unavailable,
}

pub struct GeometryAccelStats {
    backend: RayAccelBackendKind,
    kind: GeometryAccelKind,
    prototype_count: u32,
    triangle_gas_builds: u32,
    hardware_clas_builds: u32,
    procedural_aabb_builds: u32,
    visibility_program_count: u32,
    materialized_triangle_count: u64,
    cluster_count: u32,
    source_primitive_count: u64,
    mapped_primitive_count: u64,
    bytes: u64,
    fallback_reason: Option<String>,
}
```

Fields remain private outside their owning crate and are exposed through accessors. Unavailable backends return `Unavailable`; they never synthesize cluster counts from instance or prototype counts.

The old `clas_stats() -> Option<(usize, usize)>` interface is removed after all call sites migrate. Benchmarks fail if the requested acceleration kind was not built.

### 6.5 Forge ray-native visibility IR

The central breakthrough hypothesis is to virtualize authored surfaces rather than triangles. Forge already owns a typed construction DAG and deterministic evaluation. For supported construction nodes, the cook performs two lowerings from the same validated source:

```text
BlueprintGraph / CompleteBuildingDirective
  -> existing evaluated mesh                         # correctness oracle and fallback
  -> bounded ForgeVisibilityProgram                  # ray-native product candidate
```

The first visibility version contains a deliberately small, game-agnostic set of data-only primitive families: oriented box/prism, planar polygon with holes, linear extrusion, bounded profile sweep, repetition lattice, interval union/difference, and residual triangle leaf. A node has fixed parameter widths, child limits, conservative bounds, material domains, source provenance, and worst-case intersection work. Unsupported, numerically ill-conditioned, or over-budget subgraphs lower locally to residual triangle/dense-block leaves; one unsupported dormer must not force the entire building back to triangles.

The runtime never interprets BlueprintGraph and never compiles asset-specific code. Shared Slang intersection math plus thin AOT native entry wrappers implement the finite primitive families; the OptiX wrapper may remain CUDA/PTX where its program ABI requires it, but it consumes the same reflected records and must pass cross-backend ray-oracle hashes. OptiX custom primitives, Vulkan AABB procedural hit groups, Metal bounding-box intersection functions, and later DXR procedural primitives are native adapters over identical visibility records. Fixed-function hardware traverses the coarse AABB hierarchy; programmable GPU cores perform the custom leaf intersection, so the benchmark must include their occupancy and divergence rather than calling the whole path “RT-core execution.” Each adapter builds only the coarse hierarchy required to route a ray to a program domain. Inside a repetition lattice, the intersection kernel computes the relevant cell and surface analytically or with a bounded DDA; it does not emit one hardware leaf per repeated bay.

Every reported hit contains a stable surface key derived from the source construction path plus a quantized parametric coordinate. Material, UV, normal, motion, and temporal identity are evaluated from that domain. Changing evaluation tolerance changes precision, not topology or identity. Parameter updates that remain inside their admitted conservative envelope update only GPU parameter buffers and perform zero fine-geometry AS builds. An envelope escape emits GPU rebuild/refit work through the normal native-adapter path.

Ray-native, CLAS, and triangle/dense representations coexist inside one asset and one frame. Structural admission uses complete measured costs: coarse-AABB bytes/build, intersection-shader divergence and occupancy, trace, shading, parameter update, residual traffic, and peak VRAM. Hardware triangle intersection remains the expected winner for arbitrary dense residuals; ray-native is admitted only where eliminating materialization/build/LOD outweighs programmable intersection cost.

This design is stronger than topology compression because successful regions never decode back to triangles. It is also narrower than a general procedural renderer: the mesh oracle is always retained for validation, unsupported regions fail locally, and every shipping primitive family must pass cross-backend hit equivalence and full-frame gates.

### 6.6 Certified factorized visibility fabric

Native custom-intersection APIs still expose a single-ray programming model. Ray sorting improves utilization but preserves one traversal per ray. They are the portability baseline and low-volume fallback, not the preferred execution path when enough structured visibility exists. Spectra instead treats both authored surfaces and coherent ray sets as factorable domains:

```text
GPU ray-domain queue (screen tiles, coherent shadow bundles, continuation packets)
  -> conservative front-to-back cell/domain classification
  -> certify common nearest surface or miss; emit analytic per-ray hits
  -> subdivide only ambiguous domains
  -> materialize unresolved rays into GPU queues
  -> front-to-back hierarchical cell routing
  -> bin by (cell, visibility family, ray class)
  -> packet intersection over factorized surface coefficients
  -> deterministic nearest-hit reduction
  -> irregular/residual ray queue to native RT
  -> merge hit candidates; terminate or advance to next cell
```

Forge cooks a generic sparse visibility-cell DAG from conservative program-domain bounds. It is a render acceleration structure, not a city/gameplay hierarchy. A ray domain stores bounded origin/direction intervals plus a deterministic subdivision key. For planar, prismatic, extruded, and repeated domains, the kernel computes outward-rounded hit-parameter and containment intervals over the entire ray domain. It certifies a common nearest surface only when that candidate's worst valid hit is strictly before every competing structured surface and residual/native-AABB lower bound and the entire domain is inside the candidate's parametric validity region. It certifies a miss only when every structured and residual candidate is conservatively excluded. Otherwise the domain splits in fixed child order until it reaches the explicit-ray threshold.

This is the central work-elimination mechanism. A certified tile still evaluates material, UV, normal, motion, and radiance per sample, but it does not issue one traversal or one intersection control flow per ray. Plane and bounded-sweep hits are reconstructed from the certified surface id and per-ray analytic parameters. Silhouette, disocclusion, numerical-boundary, high-curvature, and incoherent secondary-ray regions naturally fail certification and continue through the exact packet/native paths; no approximation or image-space impostor is accepted.

An explicit ray carries its current cell interval and best hit. Cells advance monotonically along the ray; a ray terminates only when its best hit precedes the next cell boundary. Large/global objects and dynamic residuals use explicit side domains. This provides a correctness argument for mixing certified analytic hits, bulk packet hits, and native residual hits without relying on approximate visibility.

Within a `(cell, family)` queue, program records expose factorized coefficient blocks rather than independent control flow. Examples include ray-origin/direction matrices against plane/half-space coefficient matrices for prisms and extrusions, batched transforms into repeated-cell coordinates, and bounded polynomial coefficients for supported sweeps. Cooperative matrices accelerate only these dense dot-product tiles after domain certification; cooperative or reduced-precision output never establishes a whole-domain proof. Subgroup kernels implement the same algebra and handle divisions, interval min/max, masks, boundary refinement, and deterministic nearest-hit reduction. Rays near numerical boundaries use the declared robust scalar refinement or residual path; a learned classifier may rank work but can never confirm or reject a hit.

Certificate acceptance is performance state, not persistent scene/simulation state. Backends may conservatively decline different borderline domains because of floating-point behavior, but every accepted interval must remain outward-conservative and every final per-ray hit must match the declared oracle. Cross-backend conformance compares final hits and persistent control buffers, while reporting certificate-rate drift separately.

All domain classification, subdivision, routing, queue allocation, binning, packet formation, compaction, reductions, and indirect dispatch counts remain GPU-resident with fixed admitted capacities. Low-fill queues select native scalar custom intersection or subgroup packets rather than padding tensor work. CUDA, Vulkan, and Metal consume the same queue/record ABI; NVIDIA/Vulkan shader invocation reordering is an optional adapter experiment, not the portable scheduler. Queue spills retain a correct native/residual path and suppress the fabric claim—never a CPU scheduler.

Native submission semantics remain honest. Vulkan may consume GPU-written trace dimensions where supported. OptiX still requires a host `optixLaunch`, so its adapter submits a pre-admitted fixed maximum and the device program exits lanes beyond the GPU-written count; empty-lane cost is measured. If a backend requires several fixed passes, the host submits the same bounded pre-recorded sequence regardless of queue contents. It may not read counts, choose a branch, or loop until a queue is empty.

The measured boundary includes domain construction, interval certification, failed certificates, subdivision, ray routing, global-memory traffic, queue atomics, compaction, empty lanes, cooperative setup, exact refinement, residual native tracing, and hit merge. Packet utilization, certificate rate, or tensor instruction occupancy alone is not evidence. The fabric is promoted only when the complete intersection and geometry stages beat the best scalar custom/triangle/CLAS path.

### 6.7 Forge-owned geometry schedule

Forge receives the full-detail authored mesh and owns the deterministic runtime schedule. The cook produces:

```rust
// Owned by Ochroma `vox_data`; games embed this engine type in their own
// ready-asset envelope instead of defining a game-specific geometry ABI.
pub struct ReadyMegaGeometry {
    schema: u32,
    visibility_programs: Vec<ReadyVisibilityProgram>,
    visibility_nodes: Vec<ReadyVisibilityNode>,
    visibility_cells: Vec<ReadyVisibilityCell>,
    visibility_coefficients: Vec<ReadyVisibilityCoefficientBlock>,
    visibility_parameters: Vec<u8>,
    levels: Vec<ReadyMegaGeometryLevel>,
    clusters: Vec<ReadyMegaGeometryCluster>,
    groups: Vec<ReadyMegaGeometryGroup>,
    pages: Vec<ReadyMegaGeometryPage>,
    topology_templates: Vec<ReadyTopologyTemplate>,
    program_blocks: Vec<ReadyGeometryProgramBlock>,
    surface_maps: Vec<ReadySurfaceCorrespondence>,
    schedule_variants: Vec<ReadyGeometryScheduleVariant>,
    content_hash: [u8; 32],
}
```

Each cluster records:

- stable asset-local cluster id;
- LOD level and parent group;
- a contiguous range in the cooked level mesh;
- 4 to 256 triangles;
- material-boundary-safe source triangle provenance;
- bounds, geometric error, and immutable page id.

Each group records its parent and children, boundary-lock information, aggregate bounds, and screen-space error coefficients. Each page records immutable byte ranges and dependencies.

The cook is deterministic for identical source bytes and settings. It spatially orders triangle centroids with stable tie-breaking by source triangle id, preserves material and hard-edge seams, locks shared boundaries during simplification, and emits id-sorted tables. Hash iteration order and random seed order cannot affect the result.

#### 6.7.1 Portable geometry programs

Forge must not throw away useful construction provenance by flattening every repeated façade panel, window frame, roof bay, railing segment, road module, or imported repeated submesh into unrelated vertex/index ranges. It computes a game-agnostic topology signature and factors equivalent clusters into:

```text
shared topology template
  + quantized canonical vertex/control block
  + per-use transform/material/semantic parameters
  + optional bounded residual block
  = exact accepted runtime cluster
```

The mechanism is structural, not city-specific: engine types know templates, parameters, residuals, and provenance—not windows or buildings. Equality and reuse are validated from decoded topology/attributes, never trusted from authoring labels alone.

Unique geometry uses a dense independently decodable block representation. AMD DGF/DGFS is the required external baseline because it demonstrates RT-friendly fixed-size geometry blocks and portable cluster-granular supercompression. Spectra's portable source representation may adopt or adapt those ideas, but it must decode to identical common cluster records and must not require future DGF hardware. Capable native adapters may later consume a native block form directly.

Program blocks are independently addressable, content-hashed, bounded to one page dependency set, and decode on GPU into a fixed transient arena. The ordinary subgroup decoder is mandatory. Cooperative-matrix decode is allowed only for block-linear residuals that win transfer-plus-decode-plus-build cost. Topology templates map directly to CLAS template reuse where supported; conventional backends decode the same program into triangle AS input.

`TopologyProgramV1` is a finite data codec, not an arbitrary shader or interpreter. Its version fixes the allowed block kinds, maximum decoded vertices/triangles, parameter widths, residual bounds, memory accesses, and worst-case work. Admission validates those limits and the decoded hash before the block can enter resident state. This keeps execution bounded, cacheable, fuzzable, and portable across shader compilers.

#### 6.7.2 Bounded schedule portfolio

A single QEM/error hierarchy bakes in one vendor-neutral guess. Forge instead emits a bounded Pareto portfolio—normally two to four schedules—covering spatial locality, topology reuse, dense-block packing, page locality, update frequency, and boundary quality. It does not emit a Cartesian search space.

The cook optimizes device-independent metrics deterministically and records the complete objective vector. A versioned admission calibration measures decode, native build, trace, update, page traffic, and peak memory on the active capability fingerprint, then selects one existing schedule. It never changes topology or generates a new hierarchy at runtime. The selected schedule is diagnostic render state, not simulation/save state.

#### 6.7.3 Cross-LOD surface correspondence

Every child primitive stores a deterministic mapping to its parent surface domain: parent group/triangle, barycentric or parametric coordinates, orientation, material domain, and bounded mapping error. Boundary and many-to-one cases use stable ids and explicit validity masks. This correspondence serves three consumers:

- closest-hit provenance and motion-vector reconstruction;
- temporal/ReSTIR sample remapping across topology-changing LOD cuts;
- ray-feedback aggregation from fine hits into stable parent groups.

The no-map path remains the oracle. Invalid correspondence rejects the cooked schedule; runtime never guesses a surface mapping.

Ochroma `vox_data` owns `ReadyMegaGeometry` and its serialized validation contract. Urban Horizon's `ReadyAssetPayload` advances to schema version 3 and embeds that engine type; another Ochroma game may embed the same type in its own asset envelope without depending on Urban Horizon. Forge converts its game-agnostic cooked result into the `vox_data` wire contract at the offline engine-adapter boundary, and Spectra consumes only generic scene descriptors. Version-2 Urban payloads either migrate deterministically or are rejected with an exact recook instruction. Untyped `forge_clusters: Option<Value>` is retained only during the migration window and is not consumed by the renderer. Gate A must select `ray_native`, `geometry_programs`, or `dense_only` before this serialized schema is frozen.

### 6.8 Source-primitive provenance

Spatial clustering can reorder triangles. Therefore an OptiX pair `(clusterId, localPrimitiveIndex)` is not assumed to be a global triangle id.

The scene conversion builds a device-resident map:

```text
cluster primitive -> prototype-level cooked triangle -> packed scene triangle
```

Closest-hit resolves the packed scene triangle before reading material ids, UVs, normals, weathering, or semantic metadata. This mapping is used by both the hardware hit program and the deterministic hit-buffer oracle.

For a cluster with `n` primitives, exactly `n` provenance entries must exist. The build rejects duplicates, gaps, out-of-range source ids, and a total mapped count different from the source primitive count.

### 6.9 True indirect OptiX CLAS

The live CLAS path calls `optixClusterAccelBuild`. OptiX 9.1 defines this as an indirect multi-build: the argument array and optional argument count are device pointers. Spectra allocates fixed-capacity device work buffers at scene admission. GPU preparation kernels consume cooked cluster descriptors and write `OptixClusterAccelBuildInput*Args` plus `argsCount` directly. The host submits one fixed build sequence and never reads the selected cluster count before the build.

The sequence can build clusters from triangles/templates and GASes from the resulting cluster handles without a CPU round trip. Output handles, sizes, provenance ids, and counts stay device-resident through trace. Stats are copied only asynchronously or after an explicit benchmark synchronization outside timed frames.

Initial support is deliberately narrow:

- indexed triangle clusters;
- no more than the queried OptiX vertex and triangle limits;
- deterministic cluster order;
- material/UV provenance mapping;
- ordinary OMM foliage path remains separate until CLAS-plus-OMM is explicitly validated.

The pipeline sets `allowClusteredGeometry=1`. The build fails closed if the driver, device, or OptiX version does not support the requested form. It may then rebuild as `GeometryAccelKind::Triangle`, but stats and benchmark output must say so.

### 6.10 Hybrid per-domain policy

No representation is the universal default. Each cooked visibility/residual domain is assigned one of these policies; an asset may therefore mix procedural, cluster, and triangle traversables:

```rust
pub enum GeometryAccelPolicy {
    Auto,
    Triangle,
    Cluster,
    RayNative,
}

pub struct GeometryDomainTraits {
    domain_id: u32,
    ray_native_kind: Option<VisibilityPrimitiveKind>,
    max_intersection_steps: u16,
    residual_surface_count: u32,
    triangle_count: u32,
    cluster_count: u32,
    flags: u32,
    material_feature_mask: u64,
}
```

`Auto` is resolved when a domain enters resident state or when its versioned calibration changes, never in the frame loop. It uses cooked traits and a versioned device calibration:

- static, compact, frequently traced geometry prefers compacted triangle GAS;
- supported construction domains with high materialization/build cost and cheap bounded intersection are ray-native candidates;
- deforming geometry, continuous-LOD group cuts, and out-of-core pages are CLAS candidates;
- unsupported combinations use triangle GAS;
- a candidate switches only if its measured geometry-stage cost improves without violating correctness or VRAM gates.

The calibration key contains GPU architecture, driver, OptiX version, shader ABI, cluster-cook schema, and material feature mask. Calibration output is diagnostic state, not save-game state, and cannot affect simulation determinism.

### 6.11 Stable top-level partitions

The OptiX NVIDIA graph becomes:

```text
root IAS
  -> partition IAS 0
       -> triangle GAS, CLAS, or procedural-AABB traversables
  -> partition IAS 1
       -> triangle GAS, CLAS, or procedural-AABB traversables
  -> ...
```

The pipeline enables a graph flag that permits the multi-level traversable graph and uses stack depth 3. Partitions are stable spatial cells computed once at structural admission from prototype bounds and instance transforms with deterministic boundary rules. Static city instances live in these partitions. Continuously moving instances use a dedicated dynamic IAS with one unconditional update submission per frame; they do not migrate across static partitions. Large static objects use an explicit conservative/global path.

Each resident instance record stores its stable partition id on GPU. `apply_scene_delta` scatters transforms into the next-frame instance buffer and atomically marks a device dirty-partition bitset. A deterministic GPU compaction kernel emits an id-sorted dirty-partition packet. No CPU code walks commands, instances, or partitions to derive it.

Ordinary OptiX IAS updates are not device-indirect. A dedicated host-submission worker therefore consumes the compact packet asynchronously one frame ahead and submits only those child IAS updates. It records a build-stream event and publishes a monotonically increasing completed epoch. The render thread consumes only already-published epochs and never blocks on the worker; a missed fixed one-frame deadline may retain the previous memory-safe but visibly stale static epoch, flags the witness, and suppresses the claim. Dynamic transforms use their dedicated update path and are never described as correct when stale. Root IAS updates only when a child handle is created, destroyed, or changed.

The benchmark separates localized one-percent dirtiness from uniform one-percent dirtiness. Hierarchical IAS targets localized edits; `Auto` may choose the flat/dynamic IAS for uniformly distributed changes. The implementation is called `hierarchical_ias`, not PTLAS.

### 6.12 Continuous cluster LOD

Runtime selection operates on Forge-cooked cluster groups, never on whole-mesh gameplay substitutions. A GPU compute kernel evaluates projected geometric error, deterministic hysteresis, neighbor constraints, and locked boundaries. It writes the selected group cut to resident GPU state and emits only changed cluster work into the indirect CLAS argument buffers.

The CPU provides camera constants already required for rendering but never reads projected errors or selected groups. The selected cut may instantiate cached CLAS templates or select resident immutable cluster sets. Interactive rendering no longer invokes runtime QEM.

The one-pixel gate is evaluated from a deterministic camera path at the configured internal resolution. A no-LOD capture remains the visual oracle.

The tracer also accumulates quantized per-group hit count, ray-footprint/ray-cone class, miss-neighbor pressure, and LOD-transition confidence into a double-buffered GPU feedback table. Selection uses this one-frame-delayed signal only to rank optional detail and prefetch among cuts that already satisfy the conservative camera/error contract. Zero or stale feedback cannot remove required geometry.

When a cut changes, the surface-correspondence table remaps primary-hit identity, motion vectors, and eligible ReSTIR/temporal samples into the new topology. Invalid or high-error mappings reject reuse locally rather than resetting the whole object's history. A deterministic no-remap mode and highest-detail mode remain quality oracles.

### 6.13 Budgeted RAM-to-VRAM paging

Cooked pages are immutable and content-addressed. At scene admission, Spectra allocates a fixed-size geometry page pool from the configured budget. The GPU owns its page table, free-slot stack, generations, resident/requested/uploading/evictable states, and deterministic request/eviction selection. No per-page runtime allocation occurs.

GPU requests are sorted and compacted by:

```text
(required_frame ascending,
 geometric_error descending,
 asset_id ascending,
 group_id ascending,
 page_id ascending)
```

Eviction is deterministic within equivalent visibility information and never evicts a page required by an in-flight frame. The GPU emits opaque transfer records containing source-pack id, content hash, source byte range, destination slot, byte count, and slot generation. A host transport worker performs only file I/O and asynchronous copy into the exact GPU-selected slot, then publishes a completion token. It cannot change priority, destination, victim, LOD, or visibility.

Upload completion changes visible page state only at a GPU frame boundary after generation/fence validation. Missing optional detail falls back to an already resident parent group; missing required base geometry is a hard witness failure. There is no synchronous readback, CPU page decision, per-page allocation, or render-thread file I/O.

Pages carry compressed and decoded byte counts, codec id/version, alignment, and checksums. Baseline decoding is an ordinary GPU kernel. A block-linear or learned cooperative-matrix decoder is permitted only when it reduces end-to-end transfer-plus-decode latency without increasing required-page misses or peak VRAM. Stale transfer completions are discarded by generation without mutating a reused slot. The witness includes camera teleport, oscillating-budget pressure, and delayed-I/O cases; an always-resident parent guarantees bounded degradation rather than holes.

### 6.14 GPU frame graph

The steady-state geometry order is fixed:

```text
resident deltas -> GPU transform scatter + dirty bits
                -> GPU visibility-parameter update + envelope test
                -> GPU LOD/group-cut selection
                -> GPU page-table update + request/eviction compaction
                -> common geometry build records + device recordCount
                -> GPU native-argument lowering
                -> native indirect triangle/cluster/procedural-AABB AS builds when required
                -> bounded host-submitted updates only where the native API requires them
                -> trace/present
```

Frames are double-buffered so the bounded IAS submission packet for frame N+1 is produced while frame N traces. The worker publishes only completed build epochs; present consumes an already-complete epoch or retains an explicitly stale-but-memory-safe static epoch and records a deadline miss. Host services are not allowed to make the render thread block on a work-packet transfer or API call.

The common selection/page kernels compile for CUDA, Vulkan, and Metal. OptiX-, Vulkan-, or Metal-specific argument packing is a final adapter kernel. Vulkan KHR uses a fixed admitted build schedule plus GPU-written indirect primitive counts and zeroed inactive slots because its indirect command does not provide a device count for the number of build infos. Metal common compute remains supported, but GPU-record-driven AS submission reports unavailable until the device exposes a compatible address-driven path; it never falls back to CPU translation. AMD and Apple may not replace GPU selection with a CPU fallback.

Fixed-capacity scratch arenas are admitted with the scene and may alias only across proven non-overlapping epochs. Results report resident geometry, AS storage, page pool, native argument buffers, and peak build scratch separately; a win cannot hide a transient memory spike. Queue overlap is selected from full-frame measurements because asynchronous geometry compute that steals occupancy or bandwidth from tracing is a regression even when its isolated kernel is faster.

### 6.15 Reference comparison

The benchmark has three modes using identical cooked meshes, instances, camera path, shaders, and render configuration:

- `triangle`: compacted triangle GAS and flat IAS baseline;
- `reference`: a stable spatial meshlet/cluster schedule matching NVIDIA/meshoptimizer reference practice, true CLAS where supported;
- `city_hybrid`: Forge visibility/geometry programs, measured procedural/GAS/CLAS selection, hierarchical IAS, continuous LOD, and paging.

Within `city_hybrid`, the breakthrough bake-off holds the selected schedule and decoded geometry constant while sweeping `raw_clusters`, `dense_blocks`, and `topology_programs`. A second fixed-budget sweep allows each representation to expose as much oracle-valid detail as it can within identical frame-time and peak-VRAM limits. This separates codec/storage wins from schedule/quality wins.

The ray-native bake-off is stricter: it dual-lowers the same checked-in BlueprintGraph/directive into the mesh oracle and `visibility_programs`, then compares `triangle`, `clas`, `program_to_triangles`, and `ray_native` over identical rays and present frames. It reports coarse AABB build, candidate count, intersection instructions/time, divergence, materialized geometry, residual coverage, parameter-update builds, complete geometry-stage p95, peak VRAM, and stable-surface temporal invalidations. A ray-native result cannot borrow the topology-program storage win or count a removed triangle twice as independent evidence.

Required fixtures are:

1. static repeated `meridian_house_01` instances;
2. a multi-material building correctness fixture;
3. dense OMM foliage;
4. a deforming or page-changing geometry fixture;
5. a streamed continuous-cluster-LOD camera path;
6. a localized million-instance update fixture;
7. a mixed live Urban Horizon city witness.

The benchmark stores raw JSON and prints a human-readable verdict. Correctness gates are absolute. Performance is reported as build p50/p95, update p50/p95, native-lowering p50/p95, geometry trace p50/p95, traced triangles, resident/AS/page/scratch/peak VRAM, page misses, full present frame time, render-thread geometry CPU time, host-submission time/calls, CPU scene visits, CPU LOD/page/eviction decisions, blocking readbacks, per-frame geometry allocations, compute variant, capability fingerprint, and actual native RT backend/kind.

`city_hybrid` wins a performance fixture only when it has equal correctness, lower p95 geometry-stage frame cost, and no higher peak geometry VRAM than `reference`. Static and OMM fixtures are parity gates because `Auto` may deliberately retain the validated triangle path; multi-material is a correctness-only gate. The public claim requires zero losses and wins on changing geometry, streamed LOD, localized updates, and mixed city; otherwise the report names the winning subset.

## 7. Data Models

The following are the cross-crate contracts. Their fields are private; constructors validate invariants and accessors expose read-only data.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryAccelKind {
    Triangle,
    Cluster,
    Procedural,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RayAccelBackendKind {
    Optix,
    VulkanKhr,
    VulkanNv,
    Metal,
    Dxr,
    Unavailable,
}

#[derive(Clone, Debug)]
pub struct GeometryAccelStats {
    backend: RayAccelBackendKind,
    kind: GeometryAccelKind,
    prototype_count: u32,
    triangle_gas_builds: u32,
    hardware_clas_builds: u32,
    procedural_aabb_builds: u32,
    visibility_program_count: u32,
    materialized_triangle_count: u64,
    cluster_count: u32,
    source_primitive_count: u64,
    mapped_primitive_count: u64,
    bytes: u64,
    fallback_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryAccelPolicy {
    Auto,
    Triangle,
    Cluster,
    RayNative,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryKernelVariant {
    PortableSubgroup,
    CooperativeMatrix { shape_index: u16 },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuApi {
    Cuda,
    Vulkan,
    Metal,
    D3d12,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CooperativeElementType {
    F16,
    Bf16,
    F32,
    I8,
    U8,
    I32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CooperativeScope {
    Subgroup,
    Workgroup,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryKernelFamily {
    BatchedBounds,
    PageDecode,
    VisibilityDomainClassify,
    VisibilityPacketIntersect,
}

pub struct CooperativeMatrixShape {
    m: u16,
    n: u16,
    k: u16,
    input: CooperativeElementType,
    accumulator: CooperativeElementType,
    scope: CooperativeScope,
    alignment: u16,
}

pub struct GpuCapabilities {
    api: GpuApi,
    subgroup_width: u16,
    cooperative_shapes: Vec<CooperativeMatrixShape>,
    indirect_dispatch: bool,
    hardware_rt: bool,
    custom_intersection: bool,
    ray_invocation_reorder: bool,
    indirect_as_build: bool,
    address_driven_as_build: bool,
    sparse_residency: bool,
    fingerprint: [u8; 32],
}

pub struct GeometryAccelCapabilities {
    backend: RayAccelBackendKind,
    triangle: bool,
    cluster: bool,
    procedural_aabb: bool,
    indirect_primitive_counts: bool,
    device_build_count: bool,
    address_driven_build: bool,
    max_build_records: u32,
    fingerprint: [u8; 32],
}

pub struct GeometryKernelCalibration {
    capability_fingerprint: [u8; 32],
    shader_abi: [u8; 32],
    family: GeometryKernelFamily,
    selected: GeometryKernelVariant,
    subgroup_ms_p95: f32,
    selected_ms_p95: f32,
    max_abs_error: f32,
}

#[repr(C, align(16))]
pub struct GeometryBuildRecord {
    schema: u32,
    byte_len: u32,
    operation: u32,
    flags: u32,
    prototype_id: u32,
    group_id: u32,
    page_id: u32,
    generation: u32,
    primitive_offset: u64,
    primitive_count: u32,
    output_slot: u32,
}

pub struct MegaGeometryCookInput<'a> {
    positions: &'a [[f32; 3]],
    indices: &'a [u32],
    triangle_materials: &'a [u32],
    triangle_seams: &'a [u32],
}

pub struct MegaGeometryCookSettings {
    schema: u32,
    min_cluster_triangles: u16,
    max_cluster_triangles: u16,
    page_bytes: u32,
    max_levels: u8,
}

pub struct CookedMegaGeometry {
    visibility_programs: Vec<CookedVisibilityProgram>,
    visibility_nodes: Vec<CookedVisibilityNode>,
    visibility_cells: Vec<CookedVisibilityCell>,
    visibility_coefficients: Vec<CookedVisibilityCoefficientBlock>,
    visibility_parameters: Vec<u8>,
    levels: Vec<CookedMegaGeometryLevel>,
    clusters: Vec<CookedMegaGeometryCluster>,
    groups: Vec<CookedMegaGeometryGroup>,
    pages: Vec<CookedMegaGeometryPage>,
    topology_templates: Vec<CookedTopologyTemplate>,
    program_blocks: Vec<CookedGeometryProgramBlock>,
    surface_maps: Vec<CookedSurfaceCorrespondence>,
    schedule_variants: Vec<CookedGeometryScheduleVariant>,
    content_hash: [u8; 32],
}

pub struct CookedVisibilityProgramSet {
    programs: Vec<CookedVisibilityProgram>,
    nodes: Vec<CookedVisibilityNode>,
    cells: Vec<CookedVisibilityCell>,
    coefficients: Vec<CookedVisibilityCoefficientBlock>,
    parameters: Vec<u8>,
    residual_source_ranges: Vec<Range<u32>>,
    content_hash: [u8; 32],
}

// `vox_data::ReadyMegaGeometry`; embedded by a game-owned asset envelope.
pub struct ReadyMegaGeometry {
    schema: u32,
    visibility_programs: Vec<ReadyVisibilityProgram>,
    visibility_nodes: Vec<ReadyVisibilityNode>,
    visibility_cells: Vec<ReadyVisibilityCell>,
    visibility_coefficients: Vec<ReadyVisibilityCoefficientBlock>,
    visibility_parameters: Vec<u8>,
    levels: Vec<ReadyMegaGeometryLevel>,
    clusters: Vec<ReadyMegaGeometryCluster>,
    groups: Vec<ReadyMegaGeometryGroup>,
    pages: Vec<ReadyMegaGeometryPage>,
    topology_templates: Vec<ReadyTopologyTemplate>,
    program_blocks: Vec<ReadyGeometryProgramBlock>,
    surface_maps: Vec<ReadySurfaceCorrespondence>,
    schedule_variants: Vec<ReadyGeometryScheduleVariant>,
    content_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisibilityPrimitiveKind {
    OrientedPrism,
    PlanarPolygon,
    LinearExtrusion,
    ProfileSweep,
    RepetitionLattice,
    IntervalUnion,
    IntervalDifference,
    ResidualTriangles,
}

pub struct ReadyVisibilityProgram {
    program_id: u32,
    root_node: u32,
    node_range: Range<u32>,
    cell_range: Range<u32>,
    coefficient_range: Range<u32>,
    parameter_range: Range<u32>,
    residual_block_range: Range<u32>,
    source_graph_hash: [u8; 32],
    decoded_surface_hash: [u8; 32],
}

pub struct ReadyVisibilityNode {
    kind: VisibilityPrimitiveKind,
    child_range: Range<u32>,
    parameter_range: Range<u32>,
    conservative_bounds_q: [i32; 6],
    surface_key_base: [u64; 2],
    material_domain: u32,
    max_intersection_steps: u16,
    flags: u16,
}

pub struct ReadyVisibilityCell {
    cell_id: u32,
    child_range: Range<u32>,
    program_range: Range<u32>,
    coefficient_range: Range<u32>,
    conservative_bounds_q: [i32; 6],
    max_domain_subdivisions: u16,
    flags: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisibilityCoefficientKind {
    PlaneHalfspace,
    LocalTransform,
    BoundedPolynomial,
}

pub struct ReadyVisibilityCoefficientBlock {
    kind: VisibilityCoefficientKind,
    format: VisibilityCoefficientFormat,
    row_count: u16,
    coefficient_count: u16,
    row_stride_bytes: u16,
    error_bound_q: u16,
    parameter_offset: u64,
    parameter_bytes: u32,
    exact_refinement_node: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisibilityCoefficientFormat {
    F32,
    F16WithBound,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisibilityExecutionMode {
    NativeScalar,
    CertifiedDomains,
    PacketSubgroup,
    PacketCooperative { shape_index: u16 },
    NativeResidual,
}

pub struct ReadyGeometryProgramBlock {
    block_id: u32,
    topology_template_id: u32,
    codec: GeometryProgramCodec,
    parameter_offset: u64,
    parameter_bytes: u32,
    residual_offset: u64,
    residual_bytes: u32,
    decoded_vertex_count: u16,
    decoded_triangle_count: u16,
    decoded_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryProgramCodec {
    DenseBlocksV1,
    TopologyProgramV1,
}

pub struct GeometryScheduleObjective {
    geometric_error_q: u32,
    ray_native_surface_count: u32,
    residual_surface_count: u32,
    encoded_bytes: u64,
    decoded_bytes: u64,
    topology_reuse_count: u32,
    page_dependency_count: u32,
    update_class: u16,
}

pub struct ReadySurfaceCorrespondence {
    child_cluster_id: u32,
    child_primitive: u16,
    parent_group_id: u32,
    parent_primitive: u16,
    parent_barycentrics_q: [u16; 2],
    error_q: u16,
    flags: u16,
}

pub struct ReadyGeometryScheduleVariant {
    variant_id: u16,
    objective: GeometryScheduleObjective,
    cluster_range: Range<u32>,
    group_range: Range<u32>,
    page_range: Range<u32>,
    encoded_bytes: u64,
    decoded_bytes: u64,
}

pub struct GeometryScheduleCalibration {
    capability_fingerprint: [u8; 32],
    shader_abi: [u8; 32],
    measurements: Vec<GeometryScheduleMeasurement>,
}

pub struct GeometryScheduleMeasurement {
    variant_id: u16,
    coarse_aabb_build_ms_p95: f32,
    procedural_intersection_ms_p95: f32,
    decode_ms_p95: f32,
    build_ms_p95: f32,
    update_ms_p95: f32,
    trace_ms_p95: f32,
    residual_trace_ms_p95: f32,
    materialized_geometry_bytes: u64,
    temporal_invalidations: u64,
    page_bytes_p95: u64,
    peak_vram_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryScheduleReason {
    ConservativeCacheMiss,
    MeasuredParetoWinner,
    ForcedDiagnostic,
}

pub struct GeometryScheduleVariantDesc {
    variant_id: u16,
    objective: GeometryScheduleObjective,
    program_block_range: Range<u32>,
    surface_map_range: Range<u32>,
}

pub struct GeometryScheduleDecision {
    variant_id: u16,
    reason: GeometryScheduleReason,
}

pub struct MegaGeometryLayer {
    prototype_ranges: Vec<PrototypeMegaGeometryRange>,
    visibility_programs: Vec<GeometryVisibilityProgramDesc>,
    visibility_nodes: Vec<GeometryVisibilityNodeDesc>,
    visibility_cells: Vec<GeometryVisibilityCellDesc>,
    visibility_coefficients: Vec<GeometryVisibilityCoefficientDesc>,
    visibility_parameters: GpuBufferHandle,
    clusters: Vec<GeometryClusterDesc>,
    groups: Vec<GeometryGroupDesc>,
    pages: Vec<GeometryPageDesc>,
    primitive_ids: Vec<u32>,
    topology_templates: Vec<GeometryTopologyTemplateDesc>,
    program_blocks: Vec<GeometryProgramBlockDesc>,
    surface_maps: Vec<GeometrySurfaceMapDesc>,
    schedule_variants: Vec<GeometryScheduleVariantDesc>,
}

pub struct GeometryAdmissionInput<'a> {
    prototypes: &'a [GeometryPrototypeDesc],
    instances: &'a [GeometryInstanceDesc],
    proto_tri_base: &'a [u32],
    mega_geometry: Option<&'a MegaGeometryLayer>,
    policy: GeometryAccelPolicy,
}

pub struct GeometryGpuWorkBuffers {
    build_records: GpuBufferHandle,
    build_record_count: GpuBufferHandle,
    native_build_args: GpuBufferHandle,
    native_build_arg_count: GpuBufferHandle,
    dirty_partition_ids: GpuBufferHandle,
    dirty_partition_count: GpuBufferHandle,
    page_transfers: GpuBufferHandle,
    page_transfer_count: GpuBufferHandle,
    selected_group_cut: GpuBufferHandle,
    visibility_domain_queue: GpuBufferHandle,
    visibility_domain_count: GpuBufferHandle,
    visibility_ray_queue: GpuBufferHandle,
    visibility_ray_count: GpuBufferHandle,
    visibility_bin_ranges: GpuBufferHandle,
    visibility_packet_hits: GpuBufferHandle,
    visibility_residual_rays: GpuBufferHandle,
    visibility_residual_count: GpuBufferHandle,
    visibility_queue_stats: GpuBufferHandle,
    device_stats: GpuBufferHandle,
}

pub struct VisibilityDispatchInput<'a> {
    resident: &'a GeometryResidentHandle,
    ray_domains: GpuBufferHandle,
    ray_domain_count: GpuBufferHandle,
    explicit_rays: GpuBufferHandle,
    explicit_ray_count: GpuBufferHandle,
    output_hits: GpuBufferHandle,
    execution_policy: VisibilityExecutionPolicy,
    max_domain_subdivisions: u16,
    explicit_ray_threshold: u16,
    min_packet_fill_q: u16,
    expected_abi_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisibilityExecutionPolicy {
    Auto,
    NativeScalar,
    CertifiedFabric,
}

#[repr(C, align(16))]
pub struct VisibilityRayDomainRecord {
    schema: u32,
    ray_class: u32,
    first_ray: u32,
    ray_count: u32,
    origin_min: [f32; 3],
    origin_max: [f32; 3],
    direction_min: [f32; 3],
    direction_max: [f32; 3],
    current_cell: u32,
    subdivision_key: u32,
    subdivision_depth: u16,
    flags: u16,
}

#[repr(C, align(16))]
pub struct VisibilityBinRangeRecord {
    cell_id: u32,
    primitive_family: u16,
    ray_class: u16,
    ray_offset: u32,
    ray_count: u32,
    execution_mode: u16,
    cooperative_shape_index: u16,
}

#[repr(C, align(16))]
pub struct VisibilityCandidateHit {
    ray_id: u32,
    source_kind: u16,
    flags: u16,
    t: f32,
    primitive_or_node: u32,
    surface_key: [u64; 2],
    parametric_uv: [f32; 2],
}

pub struct GeometryIndirectBuildInput<'a> {
    work: &'a GeometryGpuWorkBuffers,
    max_build_records: u32,
    expected_schema: u32,
    expected_abi_hash: [u8; 32],
}

pub struct GeometryTransferRecord {
    source_pack_id: u32,
    content_hash: [u8; 32],
    source_offset: u64,
    byte_count: u32,
    destination_slot: u32,
    slot_generation: u32,
}

pub struct GpuQueueToken {
    backend: GpuApi,
    queue_id: u32,
}

pub struct GpuEpoch {
    backend: GpuApi,
    value: u64,
}
```

`GeometryAccelStats` provides `kind()`, count, byte, and fallback accessors. Every cook/ready/layer type provides a validating constructor plus slice accessors. No caller directly mutates an accepted schedule.

## 8. API

These signatures are the implementation contract. Changing one requires updating this design and plan first.

```rust
// forge-building: dual lowering from validated construction source
pub fn lower_visibility_graph(
    graph: &BlueprintGraph,
) -> Result<forge_mesh::VisibilityConstructionGraph, ForgeError>;

// forge-mesh
pub fn cook_visibility_program(
    input: &VisibilityConstructionGraph,
    settings: &VisibilityCookSettings,
) -> Result<CookedVisibilityProgramSet, VisibilityCookError>;

pub fn cook_mega_geometry(
    input: &MegaGeometryCookInput<'_>,
    visibility: Option<&CookedVisibilityProgramSet>,
    settings: &MegaGeometryCookSettings,
) -> Result<CookedMegaGeometry, MegaGeometryCookError>;

// Urban Horizon payload adapter
impl ReadyMegaGeometry {
    pub fn try_from_cooked(
        cooked: &forge_mesh::CookedMegaGeometry,
    ) -> Result<Self, ReadyAssetError>;
}

// spectra-scene-state
impl GeometryLayer {
    pub fn set_mega_geometry(
        &mut self,
        mega_geometry: MegaGeometryLayer,
    ) -> Result<(), SceneStateError>;

    pub fn mega_geometry(&self) -> Option<&MegaGeometryLayer>;
}

// Exact addition to the existing spectra_gpu::GpuBackend trait;
// immutable after backend initialization.
fn capabilities(&self) -> &GpuCapabilities;

pub fn select_geometry_kernel_variant(
    caps: &GpuCapabilities,
    family: GeometryKernelFamily,
    calibration: Option<&GeometryKernelCalibration>,
) -> GeometryKernelVariant;

pub fn resolve_geometry_schedule(
    variants: &[GeometryScheduleVariantDesc],
    calibration: Option<&GeometryScheduleCalibration>,
) -> GeometryScheduleDecision;

// spectra-renderer: native RT adapters consume only common device records.
pub trait GeometryAccelBackend: Send {
    fn kind(&self) -> RayAccelBackendKind;
    fn capabilities(&self) -> &GeometryAccelCapabilities;

    fn admit_geometry_scene(
        &mut self,
        input: &GeometryAdmissionInput<'_>,
        queue: &GpuQueueToken,
    ) -> Result<GeometryResidentHandle, GeometryBackendError>;

    fn encode_geometry_builds(
        &mut self,
        input: &GeometryIndirectBuildInput<'_>,
        queue: &GpuQueueToken,
    ) -> Result<GpuEpoch, GeometryBackendError>;

    fn collect_geometry_stats(
        &mut self,
        completed: &GpuEpoch,
    ) -> Result<GeometryAccelStats, GeometryBackendError>;
}

// spectra-optix: validating constructor for the initial adapter input
impl<'a> GeometryAdmissionInput<'a> {
    pub fn new(
        prototypes: &'a [GeometryPrototypeDesc],
        instances: &'a [GeometryInstanceDesc],
        proto_tri_base: &'a [u32],
        mega_geometry: Option<&'a MegaGeometryLayer>,
        policy: GeometryAccelPolicy,
    ) -> Result<Self, OptixError>;
}

// spectra-renderer: prototype admission/calibration only.
// A leading `device: &GeometryDeviceKey` is required so a calibration cooked for
// a foreign device (fingerprint/driver/OptiX/ABI/cook-schema mismatch) is
// rejected and the domain fails closed to triangle GAS. Implemented in
// `geometry_policy.rs` as `resolve_geometry_accel(device, traits, calibration)`,
// delegating to `resolve_geometry_accel_with_policy(device, policy, traits, calibration)`.
pub fn resolve_geometry_accel(
    device: &GeometryDeviceKey,
    traits: &GeometryDomainTraits,
    calibration: Option<&GeometryCalibration>,
) -> GeometryPolicyDecision;

// spectra-renderer: fixed GPU frame preparation
impl GeometryGpuPipeline {
    pub fn dispatch_prepare(
        &mut self,
        input: &GeometryPrepareInput<'_>,
        queue: &GpuQueueToken,
    ) -> Result<&GeometryGpuWorkBuffers, GeometryGpuError>;
}

// spectra-renderer: GPU-owned certified-domain, packet, and residual visibility.
impl VisibilityGpuPipeline {
    pub fn dispatch_visibility(
        &mut self,
        input: &VisibilityDispatchInput<'_>,
        geometry: &GeometryGpuWorkBuffers,
        queue: &GpuQueueToken,
    ) -> Result<GpuEpoch, VisibilityGpuError>;
}

// Dedicated host transport worker; executes opaque GPU decisions only.
impl GeometryPageTransport {
    pub fn service_transfer_batch(
        &mut self,
        records: &[GeometryTransferRecord],
        completed_fence: u64,
    ) -> Result<GeometryTransferCompletions, GeometryTransportError>;
}
```

`cook_visibility_program` and `cook_mega_geometry` are pure and thread-safe. The latter receives the optional already-validated visibility result so it can exclude ray-native source surfaces from residual clusters while proving exact total coverage. Capabilities are queried once and remain immutable. Admission/calibration are structural operations. `dispatch_prepare`, `dispatch_visibility`, and `encode_geometry_builds` enqueue fixed GPU work without inspecting results or reading queue counts. `VisibilityDispatchInput` contains only handles to GPU-produced ray-domain/ray records, fixed capacities, immutable execution policy, and ABI hashes. `service_transfer_batch` runs only on the transport worker and cannot change GPU-selected records. IAS packet submission runs only on its dedicated worker. GPU resources are retired only after their recorded epoch. No method panics for malformed asset or device input; it returns the typed error.

## 9. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `lower_visibility_graph` | validated Forge building evaluation | `forge-building/src/blueprint/visibility.rs` | Dual-lowers supported construction nodes; unsupported subgraphs become explicit residuals |
| `cook_visibility_program` | focused asset cook before mesh flattening | `forge-mesh/src/visibility_program.rs` | Emits bounded portable nodes, conservative AABBs, stable surface keys, and residual references |
| `cook_mega_geometry` | `cook_mesh_lods` replacement | `urban_horizon/src/bin/game_asset_cook/cli.rs` | Full-detail Forge input to typed hierarchy; never runs per frame |
| `ReadyMegaGeometry::try_from_cooked` | focused asset cook | `urban_horizon/src/bin/game_asset_cook/forge_runner.rs` | Writes payload schema 3 |
| geometry-program factor/decode | `cook_mega_geometry` then GPU page admission | `forge-mesh/src/geometry_program.rs`, `spectra/slang/geometry_program_decode.slang` | Reuses topology and decodes bounded blocks without CPU expansion |
| `resolve_geometry_schedule` | prototype admission/calibration | `spectra-renderer/src/renderer/geometry_policy.rs` | Selects one pre-cooked Pareto candidate, never cooks runtime topology |
| `GeometryLayer::set_mega_geometry` | packed mesh conversion beside `mesh_to_blas_material` | `urban_horizon/src/spectra_frame/mesh_convert.rs` | Converts asset-local to packed-scene ids |
| `GpuBackend::capabilities` | renderer/device initialization | `spectra-gpu/src/{cudarc_backend,vulkan_backend,metal_backend}.rs` | Immutable feature inventory and cache fingerprint |
| `select_geometry_kernel_variant` | pipeline admission | `spectra-renderer/src/renderer/geometry_compute.rs` | Chooses precompiled subgroup or exact cooperative specialization |
| `GeometryAccelBackend::admit_geometry_scene` | scene admission | backend adapter | Allocates fixed-capacity triangle/CLAS/procedural-AABB state; structural only |
| `GeometryGpuPipeline::dispatch_prepare` | live sample/update path | `spectra-renderer/src/renderer/sample.rs` | GPU owns LOD, pages, dirty bits, and CLAS args |
| `VisibilityGpuPipeline::dispatch_visibility` | wavefront sample path after ray-domain generation | `spectra-renderer/src/renderer/visibility_fabric.rs` | GPU classifies/splits domains, routes explicit rays, dispatches packet kernels, sends residuals to native RT, and merges exact nearest hits |
| visibility-domain certification | visibility fabric indirect work graph | `spectra/slang/visibility_domain_classify.slang` | Certifies common nearest surface/miss only from conservative bounds; ambiguous domains split deterministically |
| packet intersection and reduction | visibility fabric indirect work graph | `spectra/slang/visibility_packet_intersect.slang`, `spectra/slang/visibility_hit_reduce.slang` | Subgroup baseline, optional cooperative dot tiles, scalar boundary refinement |
| `GeometryAccelBackend::encode_geometry_builds` | fixed post-prepare submission | OptiX/Vulkan/Metal adapter | GPU-lowers common records and consumes device counts without readback |
| provenance lookup | OptiX closest-hit programs | `spectra-optix/ptx/programs/device_programs.cu` | Resolves packed triangle before semantic reads |
| `resolve_geometry_accel` | cooked-domain admission/calibration | `spectra-renderer/src/renderer/scene.rs` | Allows one asset to mix procedural/cluster/triangle domains; never runs per frame |
| dirty partition compaction | `apply_scene_delta` then geometry prepare | `spectra/slang/apply_scene_delta.slang` | GPU emits stable opaque IAS packet |
| IAS submission worker | asynchronous packet consumer | `spectra-optix/src/traversal.rs` | Only irreducible OptiX host submission; no decisions |
| cooked cluster LOD selection | `GeometryGpuPipeline::dispatch_prepare` | `spectra/slang/mega_geometry_prepare.slang` | Replaces CPU/legacy runtime simplification |
| ray-feedback aggregation | closest-hit then geometry prepare | `spectra/slang/mega_geometry_feedback.slang` | Delayed advisory hit/footprint signal; cannot remove required parents |
| surface correspondence | LOD transition in temporal/ReSTIR path | `spectra/slang/restir_pt.slang` | Remaps valid samples across topology changes and rejects invalid mappings locally |
| GPU page allocator/selector | `GeometryGpuPipeline::dispatch_prepare` | `spectra/slang/mega_geometry_prepare.slang` | Fixed pool, deterministic requests and victims |
| `GeometryPageTransport::service_transfer_batch` | dedicated transport worker | `spectra-renderer/src/renderer/geometry_page_transport.rs` | Performs opaque I/O/copies only |
| product config | render config load | `urban_horizon/assets/config/render.ron` | Source of truth for budget/error/hysteresis |
| product witness | `ResidentSceneRenderer::apply_scene_delta` and present stats | `ochroma/crates/vox_render/src/resident_renderer.rs` | Reports actual backend and live metrics |

## 10. Configuration

Product-facing geometry budget, error, hysteresis, and diagnostic knobs live in `urban_horizon/assets/config/render.ron` and pass through the normal config/runtime path. Engine defaults are safety limits only.

Required settings:

- acceleration policy (`auto`, `triangle`, `clas`, `ray_native` for diagnostics);
- visibility-execution policy (`auto`, `native_scalar`, `certified_fabric` for diagnostics), maximum domain subdivision depth, explicit-ray threshold, minimum packet fill, and fixed domain/ray/residual queue capacities;
- geometry VRAM budget;
- maximum screen-space error;
- LOD hysteresis;
- partition cell size or target instances per partition;
- GPU work-buffer capacities and page-pool size derived from the budget;
- maximum allowed host-submission time for diagnostics;
- witness/stat logging toggle.
- cooperative-compute policy (`auto`, `off`, `force` for diagnostics only); `force` fails closed when unsupported;
- geometry-program policy (`auto`, `dense_blocks`, `topology_programs`, `ray_native` for diagnostics), bounded schedule-candidate count, ray-native intersection-work cap, residual threshold, conservative-envelope margin, and ray-feedback/prefetch weights;
- adapter support tier and capability fingerprint are emitted diagnostics, not user-tunable product look settings.

Changing each setting must alter a visible probe or emitted witness value. Benchmark-only overrides are command-line arguments and are printed in the result header.

## 11. Failure and Fallback Rules

- Unsupported CLAS: use triangle GAS and report `accel=triangle_gas fallback=unsupported_clas`.
- Unsupported custom intersection: keep the same subgraph's residual triangle/dense representation and report `accel=<triangle_gas|clas> fallback=unsupported_ray_native`; never evaluate construction geometry on CPU.
- Invalid visibility node, unbounded intersection work, conservative-bound failure, or mesh-oracle mismatch: reject that node family and use its cooked residual representation. One local failure does not silently disable validation for the rest of the asset.
- A parameter update outside its admitted envelope emits GPU native-AS refit/rebuild work or retains the last correct representation; it cannot continue tracing stale bounds.
- A ray domain that cannot prove one common nearest surface or miss within its subdivision budget materializes exact explicit rays; certification failure is normal fallback, never permission to accept an approximate hit.
- Visibility domain/ray/residual queue overflow keeps the admitted native triangle/CLAS/custom path, sets a device flag, and suppresses the fabric claim. The CPU may not drain, resize, or reschedule a queue in response.
- Cooperative packet intersection outside tolerance or below its measured fill threshold selects the subgroup packet path. Sparse bins select scalar custom/native residual traversal; no path pads fake work to improve utilization metrics.
- Invalid cooked cluster schedule: reject the asset with asset id, schema, first invalid cluster, and recook command.
- Calibration cache miss: use conservative triangle GAS while calibration runs outside the timed witness.
- Page budget pressure: select a resident parent group before dropping required geometry.
- Partition overflow: split deterministically at a frame boundary; do not silently rebuild the whole root each frame.
- GPU work-buffer overflow: keep the resident parent/static fallback, set a device overflow flag, and suppress the performance verdict; never fall back to a CPU builder.
- Host packet unavailable by its next-frame deadline: retain the last correct resident geometry and fail the witness; never block the render thread or make a CPU geometry decision.
- Page transport failure: publish a failed completion token and keep the resident parent; the transport worker cannot choose a substitute page.
- Any hit-buffer mismatch: abort the performance verdict.
- Any use of an unavailable/stub backend: benchmark exits non-zero.
- Missing cooperative capability: use `PortableSubgroup` and report it; never emulate on CPU or claim matrix hardware.
- Cooperative output outside tolerance or complete-stage regression above 2%: quarantine that exact capability fingerprint and select `PortableSubgroup`.
- Capability, reflection ABI, compiler, or driver mismatch: invalidate the pipeline/calibration cache before admission; never compile in the live frame.
- Native adapter cannot consume the common record schema or device count: retain the last safe resident parent/static structure and fail the witness; never translate records on CPU.
- A late static-partition packet may display the explicitly labeled previous safe epoch and fails the witness. Dynamic transforms may not be called “correct” when stale; their dedicated update must meet its deadline or suppress present/claim according to product policy.
- Device loss or queue reset invalidates every native handle, queue token, epoch, capability fingerprint, and cache admission. Recovery is a structural re-admission from validated resident/cooked data; it is not a live CPU geometry fallback.
- A topology template whose decoded hash, material domain, or source provenance differs from its use site is rejected; never fall back to trusting Forge labels.
- Invalid parent/child surface correspondence rejects that schedule at cook/admission. Runtime may reject individual temporal reuse, but may not guess a mapping.
- Missing/stale ray feedback is equivalent to zero advisory priority. It cannot suppress required parents, camera/error-selected geometry, or correctness fallbacks.

## 12. Rollout

The rollout is staged so each layer produces evidence before the next depends on it:

1. run a real-asset dual-lowering and custom-intersection preflight and either authorize ray-native visibility, fall back to topology programs, or retain dense triangles;
2. prove the GPU/host boundary and add CPU-loop counters;
3. lock the common packet ABI, immutable capability fingerprint, subgroup baseline, cooperative specialization, and CUDA/Vulkan/Metal conformance harness;
4. truthful stats and fail-closed witnesses;
5. live indirect true CLAS plus provenance correctness;
6. dual-lower Forge construction to mesh oracle and bounded visibility IR, then run the GPU ray-native bake-off before freezing the payload schema;
7. prove certified ray-domain visibility, then explicit-ray packet scheduling, against scalar custom intersection and native RT before promoting the fabric;
8. geometry-program GPU bake-off against raw clusters and DGF/DGFS-style dense blocks for residual/unsupported regions;
9. typed Forge visibility programs/cells/coefficients, cluster hierarchy, topology dictionary, program blocks, surface maps, bounded schedule portfolio, and payload schema;
10. measured admission-time hybrid certified-fabric/procedural/GAS/CLAS and schedule selection;
11. GPU frame preparation and bounded hierarchical-IAS submission;
12. GPU-selected continuous cooked cluster LOD plus ray feedback and temporal surface remapping for residual geometry;
13. GPU-managed fixed-pool geometry paging with opaque host transport and GPU decode;
14. device-indirect CLAS templates for changing/reused topology;
15. final city-workload comparison and live present validation.

The preflight first measures what can remain ray-native: exact source-surface coverage by finite primitive families, residual coverage, conservative-bound quality, stable surface keys, parameter-update envelopes, mesh-oracle agreement, and complete GPU AABB/intersection cost. It then measures topology-equivalence frequency, parameter/residual entropy, dense-block lower bounds, independently decodable page overhead, and surface-map coverage for residual geometry. A synthetic repeated façade cannot authorize either schema. Failure of ray-native visibility narrows the plan to topology programs; failure of topology programs narrows it to dense blocks and native triangle/CLAS paths.

Each stage retains a triangle-GAS fallback. A failed optimization is removed or disabled; it is not hidden behind a misleading benchmark label.

## 13. Opportunity Register

These opportunities are intentionally gated rather than assumed:

| Opportunity | Why it may matter | Entry gate |
|---|---|---|
| Vulkan KHR indirect BLAS/TLAS on AMD/Intel | Removes CPU-derived primitive counts and broadens hardware competition | Same packet ABI, live present correctness, lower full geometry-stage p95 |
| Vulkan NV CLAS/PTLAS on NVIDIA | Could remove the OptiX partition-submission island and test NVIDIA's own Vulkan path against OptiX | Explicit project-law decision plus zero-loss head-to-head witness |
| Metal 4 address-driven AS builds | Could preserve GPU ownership more completely on Apple silicon | Named Apple9+ device, native capability proof, present witness |
| Native DGF/DGFS consumption | Avoids decode expansion on capable future AMD/API paths while retaining one portable source representation | Same decoded/provenance oracle and lower end-to-end geometry-stage p95/VRAM |
| Cage plus bounded residual blocks | Factor smooth or repeated surfaces into a coarse cage and compact displacement/residual data without depending on deprecated displacement-micromap APIs | Better mixed-asset ratio than dense blocks, bounded geometric error, independently decodable pages, and no seam/radiance mismatch |
| GPU storage decompression | DirectStorage GDeflate/Zstd or analogous platform paths can remove CPU decompression and reduce transport bandwidth | Opaque transport contract, lower I/O-plus-decode p95, no CPU decision or required miss |
| GPU predictive prefetch | Can hide storage latency on fast camera movement | Advisory only; resident-parent correctness, deterministic final request order, teleport fixture |
| Native shader/invocation reordering adapter | May improve scalar/native residual utilization on hardware that exposes it without owning the portable scheduler | Full-frame win after reorder cost, no bias/latency regression, identical common queue ABI |
| Ray-class LOD | Primary/specular rays may need detail that diffuse/shadow rays do not | Equal radiance oracle, lower total trace cost including extra AS/storage, valid surface remapping |
| Certified temporal hit reuse | Stable parametric surface keys plus conservative motion/envelope certificates may allow selected ray hits to bypass traversal across frames | Exact disocclusion/motion validation, zero stale hits, and a full-frame win after certificate checks; never a correctness assumption |
| Device execution graphs | Can fuse classify/decode/lower stages and reduce global-memory/barrier traffic | Stable production API only; provisional Vulkan shader enqueue remains research, with indirect-dispatch baseline |
| AS compaction/serialization cache | Can reduce load time and steady VRAM | Driver/API compatibility key, measured load/VRAM win, no stale cache acceptance |
| Cooperative material/denoise/reconstruction reuse | Amortizes the portable compute layer beyond geometry | Separate plan; must not expand this geometry milestone's critical path |

Tensorized BVH traversal is research, not a dependency. It enters only after a packet-traversal prototype beats native hardware traversal on a named workload including build, sorting, memory, and full-frame costs.

## 14. Out of Scope and External Technical Basis

- Local OptiX 9.1 indirect-build contract: `spectra/rust/spectra-optix/optix-sdk/include/optix_host.h` (`optixClusterAccelBuild` arguments and count are device memory)
- NVIDIA RTX Mega Geometry Vulkan samples overview: https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/
- NVIDIA OptiX clustered geometry sample: https://github.com/NVIDIA/optix-sdk-samples
- Vulkan partitioned acceleration structure extension: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_partitioned_acceleration_structure.html
- Vulkan cooperative matrix extension and property enumeration: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html
- Slang cooperative matrix type: https://docs.shader-slang.org/en/stable/external/core-module-reference/types/coopmat-04/index.html
- Vulkan KHR acceleration structures and indirect builds: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_acceleration_structure.html
- Apple Metal feature tables and ray-tracing acceleration structures: https://developer.apple.com/metal/capabilities/ and https://developer.apple.com/documentation/metal/ray-tracing-with-acceleration-structures
- Vulkan procedural AABB/intersection-shader contract: https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-intersection and https://docs.vulkan.org/spec/latest/chapters/raytraversal.html
- OptiX custom-primitive contract: https://raytracing-docs.nvidia.com/optix8/guide/index.html#basic_concepts#custom-primitives
- AMD Dense Geometry Format and DGF SuperCompression: https://gpuopen.com/download/publications/DGF.pdf and https://gpuopen.com/learn/introducing-amd-dgf-supercompression/
- NVIDIA Shader Execution Reordering: https://developer.nvidia.com/blog/improve-shader-performance-and-in-game-frame-rates-with-shader-execution-reordering/
- Vulkan ray-tracing invocation reorder: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_ray_tracing_invocation_reorder.html
- Ray-reordering cost study: https://arxiv.org/abs/2506.11273
- Neural Intersection Function (dense-matrix precedent, not an exactness dependency): https://arxiv.org/abs/2306.07191
- Beam-tracing geometric precedent: https://doi.org/10.1145/323233.323241
- GPU ray sorting and breadth-first packet traversal: https://doi.org/10.1111/j.1467-8659.2009.01598.x
- Hybrid coherent-packet/incoherent-single-ray tracing: https://doi.org/10.1109/TVCG.2011.277
- GPU traversal efficiency and hierarchical-tracing limits: https://research.nvidia.com/sites/default/files/publications/aila2009hpg_paper.pdf
