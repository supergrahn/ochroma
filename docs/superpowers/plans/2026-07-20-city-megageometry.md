# City MegaGeometry Implementation Plan

> **For agentic workers:** Execute one task at a time, preserve every dirty worktree, and stop at each hardware gate until remote/game execution is explicitly authorized. Use a task-execution skill if one is available in that session.
>
> **Status:** In progress; the AMD Vulkan path now completes Forge directive -> typed Ochroma payload -> Urban placement -> GPU-only realization -> native triangle BLAS/TLAS -> live Spectra present. The compressed-source/native-execution breakthrough is witnessed; CUDA/OptiX, Metal, multi-family product performance, and the stronger visibility-fabric/NVIDIA claims remain hardware gates.
> **Design:** [City MegaGeometry Design](../specs/2026-07-20-city-megageometry-design.md)
> **Scope:** Forge authoring/cook, Urban Horizon payload and wiring, Ochroma resident bridge, Spectra scene state/renderer/OptiX
> **Hardware gates:** NVIDIA RTX 4070 Ti OptiX box and AMD 780M Vulkan product machine; named Apple Silicon Metal 4 hardware is required before the Metal adapter graduates from experimental, but does not block the NVIDIA/AMD city release

**Goal:** Ship a truthful deterministic, GPU-owned city visibility system that keeps supported Forge construction compact and backend-neutral through storage and streaming, then chooses the measured device-native execution representation at residency. That may be certified ray domains, coherent GPU packets, native custom primitives, CLAS, or GPU-realized fixed-function triangles; source representation is never confused with execution representation. The system eliminates CPU geometry materialization, topology-changing runtime LOD, and individual ray work where each measured path wins; preserves stable parametric surface identity; and delivers measured wins without moving scheduling to CPU or violating visual correctness.

**Done When:** The authorized NVIDIA suite command prints `MEGAGEOMETRY_OVERALL winner=city_hybrid losses=0 target_wins=4 claim=better_city_geometry_system`, the geometry-program and scalar ray-native results are honestly scoped, and the largest result prints `MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage>=0.60 certified_without_per_ray_traversal>=0.35 active_packet_lanes>=0.75 intersection_speedup_vs_scalar>=2.0 geometry_stage_speedup>=2.0 routing_queue_overhead_pct<=10.0 correctness_mismatches=0 cpu_scheduling=0 real_asset_families>=2 losses=0 claim=certified_visibility_fabric`; CPU ownership remains zero, CUDA/Vulkan packet and control contracts match, the authorized AMD witness holds at least 30 fps, and a human has inspected both real-time 1 spp present captures. Metal remains `support=experimental` until its independent hardware gate passes. Failure of a breakthrough line narrows the claim but does not invalidate a winning subset.

**Architecture:** Forge dual-lowers its validated construction DAG into the existing mesh oracle and a finite data-only visibility IR with sparse visibility cells and factorized surface coefficients. Spectra first classifies coherent ray domains using exact conservative bounds: certified domains emit analytic hits without per-ray traversal, ambiguous domains split, unresolved rays bin by cell/family/ray class for subgroup or cooperative packet intersection, and irregular residue enters native RT. Supported surfaces can also use scalar native custom intersection; unsupported subgraphs become local topology-program or dense-triangle residuals. Every queue, count, subdivision, dispatch, reduction, and fallback selection remains GPU-owned with fixed admitted capacities.

**Design Document:** `docs/superpowers/specs/2026-07-20-city-megageometry-design.md`

**Tech Stack:** Rust, Slang, Forge BlueprintGraph dual lowering, conservative interval ray-domain classification, GPU wavefront queues, subgroup packet algebra, optional cooperative matrices for dense coefficient tiles, CUDA/PTX/OptiX 9.1 custom primitives and CLAS, Vulkan/SPIR-V procedural AABB and KHR RT, Metal/MSL bounding-box intersection, DGF/DGFS reference formats, optional DirectStorage GPU decompression, Forge asset cook, RON product configuration

**Build:** Local pure-Rust crates use `cargo test`/`cargo check`; CUDA/OptiX and live NVIDIA gates run only on the authorized NVIDIA box, Vulkan gates on the AMD product machine, and Metal support requires a named Apple Silicon runner. Hardware adapter commands remain permission-gated.

### Implementation checkpoint — 2026-07-20

- **The missing execution-layer opportunity is now implemented:** compressed Forge programs are the storage/streaming representation, not a mandate to execute programmable intersections. Spectra realizes certified exact-cell programs once on the GPU into its canonical vertex/shading stream plus a tight native index stream, synchronizes on-device, and builds ordinary opaque triangle BLASes. The CPU performs no geometry generation, de-indexing, repacking, readback, or per-frame scheduling.
- The Meridian House product payload carries 3,199 exact programs, 49 shared templates, 3,199 selectors, and 9,597 patches in 409,472 program bytes. Those programs replace 19,194 source facade triangles; the remaining source geometry stays an exact residual. The live log prints `MEGAGEOMETRY_GPU_REALIZED programs=3199 patches=9597 triangles=19194 cpu_geometry_work=0 status=device_resident`.
- A same-binary, same-map, same-camera, 60-frame live-present A/B on AMD 780M measured `render_median=39.03ms` for GPU-realized MegaGeometry and `render_median=38.31ms` for the source-triangle control: a 1.88% delta, placing the compressed path in native-triangle execution parity rather than the much slower programmable-AABB path. Xwayland throttled presentation to 1 Hz during this long background run, so present/sustained FPS is not product-performance evidence; the render medians are the valid relative GPU result.
- The focused full-render equivalence test removes the source wall triangles, realizes them on-device, shades through the normal wavefront renderer, and compares against an identical triangle control. It reports `max_image_error=0.000000149`, `triangle_primary_ms=0.245358`, `mega_primary_ms=0.171390`, `cpu_geometry_work=0`, and exits cleanly. A teardown ownership defect found by this test was fixed so caller-owned GPU index buffers are never unmapped or freed by the Vulkan acceleration adapter.
- The inspected live 1 spp + temporal reconstruction witness is `urban_horizon/artifacts/mega_geometry_live/meridian_gpu_realized_current_frontage.png`; a settled close crop separately exposes authored glass, mullion, spandrel, reveal, trim, and interior-card material zones.

- Forge now factors 10,334 exact visibility patches / 20,668 local triangles from two real authored asset families into 14 broad surface programs, 53 shared cell templates, and 3,448 selectors. The real corpus reports `coarse_primitive_reduction=738.143x`, exact material/normal/UV identity `20668/20668`, and nearest-hit mesh-oracle agreement `2090/2090`.
- The checked-in census is deterministic: two independent cooks produced SHA-256 `838aabef0d86d8134f6b0c84e45a972c8b715f1e9c0c3f607749f9a023fd5c37`.
- Spectra has a fixed backend-neutral 128-byte surface-program record, 112-byte four-rectangle template, device-owned ray count/index packet, and a Slang broad-surface kernel. The encoded broad records occupy 21,520 bytes versus 1,322,752 bytes for exact patch records (`61.466x` reduction).
- Twisted surfaces use a fixed 32-segment multi-root sweep plus 20 bisection steps and exact local 3x3 cell refinement. Work is bounded independently of source patch/triangle count; no CPU per-frame scheduler or readback participates.
- The compute-only Vulkan hardware gate passes on `AMD Radeon 780M Graphics (RADV PHOENIX)`: 1,549 broad-surface rays compared against the exact patch GPU kernel produce zero misses and zero hit/material/key/distance/normal/UV mismatches.
- Vulkan native routing is now real: 14 procedural AABBs build as instanced BLAS/TLAS state and the 1,549-ray native query agrees with the exact-patch GPU oracle at zero mismatches. The latest microprobe reports native build `3.780618 ms` and native dispatch `2.637698 ms`; the exact-patch known-index dispatch is only a correctness lower bound, not a fair RT baseline.
- `ReadyMegaGeometry` is a validated, backend-neutral Ochroma asset ABI. The real cottage product cook emits four programs replacing 172 source triangles (`2256` encoded bytes), and validates through `ReadyAssetPayload` schema 3.
- Urban LOD0 placement removes exactly the covered source triangle ids only when prototype sharing and native MegaGeometry are admitted. It preserves all residual triangles, instance transforms, and authored-to-resident material mappings. Spectra deduplicates immutable program storage by content hash and builds true transformed AABB prototype instances.
- Native MegaGeometry now participates in the real Spectra wavefront nearest-hit contract, material/BSDF/lighting/AOV path, and shadow occlusion without CPU ray scheduling or readback. Vulkan closest-hit triangle and MegaGeometry work is fused into one wavefront dispatch; the shadow variant remains split because the fused variant measured substantially worse register pressure on RDNA3.
- The AMD 780M offscreen full-render witness deliberately removes the visible wall's two source triangles, retains one ordinary residual triangle, and still shades the procedural wall red through the complete renderer. It prints `mean_luminance=0.701619`, `primary_merge_ms=1.839133`, `shadow_merge_ms=2.002087`, `status=pass` at 128x96, 1 spp, two bounces.
- Product admission is config-first through `assets/config/render.ron` (`mega_geometry.native_enabled`), with the environment retained only as an A/B override. Primary-ray generation and both triangle traversal paths explicitly clear the procedural hit discriminator, preventing stale layer tags across scene transitions.
- Procedural hits now preserve exact object-space `dpdu`/`dpdv` through native traversal and solve screen-space UV derivatives in the common megakernel. EWA mip selection, parallax, and tangent-space normal mapping therefore consume the same authored differential geometry as triangle hits instead of zero gradients or a synthetic tangent frame.
- Urban now converts the established row-major proto-instance linear transform explicitly before composing the column-major MegaGeometry transform. A rotation/mirror/non-uniform-scale contract test passes, and live AMD beauty plus material-slot A/B captures align the procedural walls with Forge-authored rustication, plaster/trim, windows, and residual facade dressing.
- Forge oracle refinement is fail-closed at runtime primitive granularity: a mismatch expands to the complete authored analytic surface, its original triangles remain residual, and its broad surface program is omitted. The Forge oracle suite reports `8 passed; 0 failed`; the freshly recooked cottage remains fully certified at four programs and 172 removed source triangles.
- The dominant untwisted/untapered architectural family now takes an exact plane solve before cell lookup, bypassing Newton and inverse deformation. In the equal-image wall witness this reduced primary procedural intersection from roughly `1.87 ms` to `0.39 ms` in the first observed run; timings remain too noisy and undersized for a product claim. The native-triangle control was `0.26 ms`, while the rendered images agreed to `1.401e-6` maximum channel error.
- A fair real-corpus AS admission comparison now builds both 20,668 exact source triangles and the 14 broad AABBs on the same AMD Vulkan device. It reports `triangle_bytes=744048`, `broad_aabb_bytes=336`, `storage_reduction=2214.429x`, `triangle_build_ms=2.340191`, and `broad_build_ms=2.243490`. This is a major representation/transfer result but only a `1.043x` build win at this tiny two-asset scale, so it does not satisfy the complete-stage speedup gate by itself.
- The **compressed-source/native-fixed-function-execution** breakthrough is authorized for the witnessed AMD Vulkan path. It is not evidence for `better_city_geometry_system`, `ray_native_city_geometry`, or `certified_visibility_fabric`: those broader claims still require the plan's two-family, 120-frame product, NVIDIA CUDA/OptiX, and independent Metal hardware gates.

---

## IMPORTANT NOTES

- `GeometryAccelStats` and every new cross-crate cook/layer type have private fields and validating constructors. Use accessors; do not mutate accepted schedules.
- `cook_mega_geometry(input: &MegaGeometryCookInput<'_>, visibility: Option<&CookedVisibilityProgramSet>, settings: &MegaGeometryCookSettings) -> Result<CookedMegaGeometry, MegaGeometryCookError>` is the exact Forge entry point; it proves ray-native plus residual source coverage exactly once.
- `forge_building::blueprint::lower_visibility_graph(graph: &BlueprintGraph) -> Result<forge_mesh::VisibilityConstructionGraph, ForgeError>` dual-lowers the validated source without changing the mesh oracle.
- `forge_mesh::cook_visibility_program(input: &VisibilityConstructionGraph, settings: &VisibilityCookSettings) -> Result<CookedVisibilityProgramSet, VisibilityCookError>` emits only finite validated primitive families plus explicit residuals.
- `ReadyMegaGeometry::try_from_cooked(cooked: &forge_mesh::CookedMegaGeometry) -> Result<Self, ReadyAssetError>` is the exact Urban payload adapter.
- `GeometryLayer::set_mega_geometry(&mut self, mega_geometry: MegaGeometryLayer) -> Result<(), SceneStateError>` is the exact scene-state setter.
- `GpuBackend::capabilities(&self) -> &GpuCapabilities` is immutable after device initialization and its fingerprint participates in every shader/calibration cache key.
- `select_geometry_kernel_variant(caps: &GpuCapabilities, family: GeometryKernelFamily, calibration: Option<&GeometryKernelCalibration>) -> GeometryKernelVariant` selects only precompiled variants at structural admission.
- `resolve_geometry_schedule(variants: &[GeometryScheduleVariantDesc], calibration: Option<&GeometryScheduleCalibration>) -> GeometryScheduleDecision` selects one pre-cooked Pareto candidate at admission and never generates topology.
- `GeometryAccelBackend::admit_geometry_scene(&mut self, input: &GeometryAdmissionInput<'_>, queue: &GpuQueueToken) -> Result<GeometryResidentHandle, GeometryBackendError>` is structural/load-time only.
- `GeometryAccelBackend::encode_geometry_builds(&mut self, input: &GeometryIndirectBuildInput<'_>, queue: &GpuQueueToken) -> Result<GpuEpoch, GeometryBackendError>` consumes GPU-written records and GPU-written counts without readback.
- `GeometryAccelBackend::collect_geometry_stats(&mut self, completed: &GpuEpoch) -> Result<GeometryAccelStats, GeometryBackendError>` is benchmark/diagnostic only and never runs before present.
- `resolve_geometry_accel(traits: &GeometryDomainTraits, calibration: Option<&GeometryCalibration>) -> GeometryPolicyDecision` runs only at cooked-domain admission/calibration and cannot enter simulation or save state.
- `GeometryGpuPipeline::dispatch_prepare(&mut self, input: &GeometryPrepareInput<'_>, queue: &GpuQueueToken) -> Result<&GeometryGpuWorkBuffers, GeometryGpuError>` is the sole steady-state LOD/page/build-work decision path.
- `VisibilityGpuPipeline::dispatch_visibility(&mut self, input: &VisibilityDispatchInput<'_>, geometry: &GeometryGpuWorkBuffers, queue: &GpuQueueToken) -> Result<GpuEpoch, VisibilityGpuError>` is the sole certified-domain/packet/residual visibility scheduler. It consumes GPU count handles and cannot read them on CPU.
- `GeometryPageTransport::service_transfer_batch(&mut self, records: &[GeometryTransferRecord], completed_fence: u64) -> Result<GeometryTransferCompletions, GeometryTransportError>` executes opaque GPU decisions only.
- OptiX uses hierarchical IAS and graph depth 3. Do not name it PTLAS; Vulkan PTLAS is out of the initial scope.
- Forge owns cluster topology, continuous LOD hierarchy, boundaries, and pages. Urban gameplay and Spectra runtime do not run QEM or repair authored geometry.
- Per frame, CPU code may not walk scene geometry, choose LOD, prioritize/evict pages, compact dirty partitions, allocate page slots, generate CLAS args, or synchronously read counts/stats.
- Common scene state and packets may contain only fixed-width backend-neutral records and `GpuBufferHandle`s; native pointers, queues, and API structs exist only inside the final adapter.
- Slang `linalg::CoopMat` is the cooperative source abstraction. Every eligible kernel retains an ordinary subgroup/SIMD GPU implementation; matrix support is optional, measured, and never CPU-emulated.
- Forge topology reuse is proven by decoded topology/attribute/provenance hashes, never authoring labels. Unique geometry uses a dense-block fallback benchmarked against DGF/DGFS-style encoding.
- Ray-native programs are finite data, never BlueprintGraph interpretation, arbitrary bytecode, runtime code generation, or per-asset shaders. Every node has conservative bounds and a fixed worst-case intersection budget.
- Ray-domain certification is exact work elimination, not an image-space approximation: a common hit requires conservative containment plus strict nearest-order proof over the whole domain. Ambiguous domains split or become explicit rays.
- Domain subdivision, ray binning, packet fill decisions, residual routing, nearest-hit reduction, and all indirect counts remain GPU-owned. Queue overflow retains a correct admitted native path and suppresses the fabric claim; it never invokes a CPU scheduler.
- Supported construction surfaces keep a stable `(construction_path, parametric_coordinate)` identity across tolerance/parameter changes. Residual triangle regions retain the explicit cross-LOD mapping contract.
- Parent/child surface correspondence is required for accepted topology-changing LOD. Invalid maps reject the schedule; the renderer never guesses mappings.
- Ray feedback is delayed advisory input only. It may prioritize optional detail but cannot suppress required parents or override the conservative camera/error cut.
- Floating cooperative results may not directly decide topology, LOD cuts, partition ids, residency, eviction, or build counts. Those outputs remain integer/fixed-point and bit-identical across supported backends.
- No cuBLAS, cuDNN, MPSGraph, or vendor algorithm library owns the common geometry control plane. Platform compilers, drivers, and native RT/build APIs remain required.
- Ordinary OptiX IAS submission and storage/upload APIs are explicit bounded host islands. They run off the render thread, consume opaque GPU packets, and are measured separately.
- `opacity_tex == albedo_tex` foliage keeps its currently validated OMM triangle path until a dedicated CLAS-plus-OMM witness exists.
- `todo!()`, `unimplemented!()`, empty bodies, synthetic benchmark counters, and `assert!(value.is_some())` acceptance tests are task failures.
- Do not commit, launch the game, start a remote build, or run a remote render unless the user explicitly asks.

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `$FORGE/crates/mesh/src/mega_geometry.rs` | Deterministic spatial cluster, group, and page cook |
| Create | `$FORGE/crates/mesh/src/geometry_program.rs` | Topology factoring, dense blocks, quantized parameters/residuals, surface maps, and schedule candidates |
| Create | `$FORGE/crates/mesh/src/visibility_program.rs` | Generic bounded visibility IR, sparse visibility-cell DAG, factorized coefficient blocks, primitive families, validation, stable surface keys, residual references, and cook |
| Create | `$FORGE/crates/mesh/tests/mega_geometry.rs` | Cook determinism, seam, hierarchy, and coverage tests |
| Create | `$FORGE/crates/mesh/tests/geometry_program.rs` | Decode identity, topology reuse, surface-map, and compression gates |
| Create | `$FORGE/crates/mesh/tests/visibility_program.rs` | Bounds, finite-work, surface-key, local-residual, and deterministic-cook gates |
| Modify | `$FORGE/crates/mesh/src/lib.rs` | Export the validated cook API |
| Modify | `$FORGE/crates/mesh/Cargo.toml` | Cook implementation dependencies |
| Modify | `$FORGE/crates/building/src/asset.rs` | Carry typed Forge MegaGeometry data |
| Modify | `$FORGE/crates/building/src/directive/completion.rs` | Validate full hierarchy instead of source chunks only |
| Create | `$FORGE/crates/building/src/blueprint/visibility.rs` | Dual-lower supported BlueprintGraph construction into generic visibility nodes |
| Modify | `$FORGE/crates/building/src/blueprint/mod.rs` | Export the validated visibility lowering |
| Create | `$FORGE/crates/building/tests/visibility_lowering.rs` | Mesh-oracle/source-coverage and unsupported-subgraph residual tests |
| Create | `$OCHROMA/crates/vox_data/src/mega_geometry.rs` | Engine-owned serialized ReadyMegaGeometry contract reusable by every Ochroma game |
| Create | `$OCHROMA/crates/vox_data/tests/mega_geometry_asset.rs` | Wire-schema validation, deterministic round-trip, and no-game-concepts contract |
| Modify | `$OCHROMA/crates/vox_data/src/lib.rs` | Export the engine-owned MegaGeometry asset contract |
| Modify | `$URBAN/src/asset/payload.rs` | ReadyAssetPayload v3 embeds `vox_data::ReadyMegaGeometry`; it does not define a game ABI |
| Modify | `$URBAN/src/asset/validation.rs` | Payload schema and cluster hierarchy validation |
| Modify | `$URBAN/src/bin/game_asset_cook/cli.rs` | Call Forge cook from the focused asset cook |
| Modify | `$URBAN/src/bin/game_asset_cook/forge_runner.rs` | Convert Forge cook output to ready payload |
| Modify | `$URBAN/src/bin/game_asset_cook/tests/mesh.rs` | Payload/cook integration evidence |
| Create | `$URBAN/assets/config/mega_geometry_corpus.ron` | Ordered production-asset corpus and expected content hashes for the representation preflight |
| Modify | `$URBAN/docs/superpowers/specs/2026-07-19-unified-forge-content-directive.md` | Clarify Forge-owned continuous cluster schedule |
| Modify | `$URBAN/src/spectra_frame/mesh_convert.rs` | Convert asset-local cluster provenance to packed-scene ids |
| Modify | `$URBAN/src/spectra_frame/scene_build.rs` | Attach typed MegaGeometry to the packed GeometryLayer |
| Create | `$URBAN/src/spectra_frame/mega_geometry_conversion_tests.rs` | Conversion and permutation-stability tests |
| Modify | `$URBAN/src/spectra_frame/mod.rs` | Wire payload, resident state, LOD, and pager into frames |
| Modify | `$URBAN/src/spectra_frame/gpu_frame.rs` | Carry geometry deltas and witness state |
| Modify | `$URBAN/src/spectra_frame/witnesses.rs` | Product-facing correctness/performance witnesses |
| Modify | `$URBAN/assets/config/render.ron` | Product geometry budget/error/hysteresis source of truth |
| Create | `$URBAN/docs/reference/city-megageometry-results.md` | Hardware commands, revisions, raw metrics, captures, verdict |
| Modify | `$SPECTRA/rust/spectra-scene-state/src/layers.rs` | Generic MegaGeometry layer and descriptors |
| Modify | `$SPECTRA/rust/spectra-scene-state/src/lib.rs` | Export validated scene-state API |
| Modify | `$SPECTRA/rust/spectra-scene-state/src/resident.rs` | Device-resident cluster/page/partition metadata and fixed capacities |
| Modify | `$SPECTRA/rust/spectra-gpu/src/backend.rs` | Immutable compute/resource capability contract on the existing backend trait |
| Create | `$SPECTRA/rust/spectra-gpu/src/capabilities.rs` | Stable capability types, cooperative shapes, and fingerprinting |
| Modify | `$SPECTRA/rust/spectra-gpu/src/lib.rs` | Export capability contracts |
| Modify | `$SPECTRA/rust/spectra-gpu/src/cudarc_backend/cuda_real/backend_impl.rs` | CUDA capability discovery and queue/epoch implementation |
| Modify | `$SPECTRA/rust/spectra-gpu/src/vulkan_backend.rs` | Vulkan cooperative-matrix/indirect-AS capability discovery |
| Modify | `$SPECTRA/rust/spectra-gpu/src/metal_backend.rs` | Metal SIMD-matrix/address-driven-AS capability discovery |
| Create | `$SPECTRA/rust/spectra-gpu/tests/capability_contract.rs` | Stable fingerprint, no-fake-capability, and cache-key tests |
| Modify | `$SPECTRA/rust/spectra-optix/src/optix_host.rs` | Device-indirect CLAS/templates, hierarchical IAS, actual stats |
| Modify | `$SPECTRA/rust/spectra-optix/src/optix_host_unavailable.rs` | API-compatible truthful unavailable backend |
| Modify | `$SPECTRA/rust/spectra-optix/src/traversal.rs` | Live geometry-build bridge and removal of fake streaming |
| Modify | `$SPECTRA/rust/spectra-optix/src/lib.rs` | Export typed OptiX geometry contracts |
| Modify | `$SPECTRA/rust/spectra-optix/ptx/programs/device_programs.cu` | Cluster provenance plus thin OptiX custom-primitive entry wrappers over common visibility records |
| Create | `$SPECTRA/rust/spectra-optix/tests/accel_stats_contract.rs` | Fail-closed acceleration identity tests |
| Create | `$SPECTRA/rust/spectra-optix/tests/indirect_clas_contract.rs` | Device args/count and no-readback contract |
| Create | `$SPECTRA/rust/spectra-optix/tests/cluster_provenance.rs` | Reordered multi-material mapping tests |
| Create | `$SPECTRA/rust/spectra-optix/tests/hierarchical_ias.rs` | Stable partitions and root-handle tests |
| Create | `$SPECTRA/rust/spectra-optix/tests/clas_templates.rs` | Template key, mapping, compatibility, and lifetime tests |
| Modify | `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs` | Register policy, LOD, pager modules |
| Modify | `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs` | Structural admission and fixed GPU/host-service wiring |
| Modify | `$SPECTRA/rust/spectra-renderer/src/renderer/sample.rs` | Dispatch fixed GPU geometry preparation in interactive rendering |
| Modify | `$SPECTRA/rust/spectra-renderer/src/spp_loop.rs` | Insert certified-domain/packet/native-residual visibility before the existing shade pass |
| Modify | `$SPECTRA/rust/spectra-renderer/src/render_state.rs` | Calibration, GPU work buffers, and asynchronous stats state |
| Modify | `$SPECTRA/rust/spectra-renderer/build.rs` | AOT subgroup/cooperative and native-lowering variants plus reflection hashes |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_backend.rs` | Backend-neutral records plus OptiX/Vulkan/Metal RT adapter trait |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_compute.rs` | Subgroup/cooperative variant selection and calibration |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_program.rs` | GPU program-block admission, decode arenas, and schedule selection |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/visibility_program.rs` | Ray-native admission, fixed parameter state, envelope updates, and measured selection |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/visibility_fabric.rs` | Fixed-capacity GPU ray-domain/packet/residual work graph and exact hit merge |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_policy.rs` | Measured per-prototype AS selection |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_gpu.rs` | Dispatch GPU LOD, page, dirty-work, and CLAS-args kernels |
| Create | `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_page_transport.rs` | Opaque asynchronous file/upload transport worker |
| Create | `$SPECTRA/rust/spectra-renderer/tests/geometry_policy.rs` | Hybrid policy gates |
| Create | `$SPECTRA/rust/spectra-renderer/tests/geometry_gpu.rs` | GPU ownership, LOD, compaction, and no-CPU-decision tests |
| Create | `$SPECTRA/rust/spectra-renderer/tests/geometry_page_transport.rs` | Opaque batch, fence, and no-policy transport tests |
| Create | `$SPECTRA/rust/spectra-renderer/tests/cpu_loop_contract.rs` | Render-thread visits, readback, allocation, and decision counters |
| Create | `$SPECTRA/rust/spectra-renderer/tests/geometry_backend_conformance.rs` | Packet ABI, control hash, shader reflection, and adapter honesty |
| Create | `$SPECTRA/rust/spectra-renderer/tests/geometry_program.rs` | GPU decode, fixed arenas, surface maps, and schedule-selection contracts |
| Create | `$SPECTRA/rust/spectra-renderer/tests/visibility_program.rs` | Custom-intersection capability, oracle, envelope, and local-fallback contracts |
| Create | `$SPECTRA/rust/spectra-renderer/tests/visibility_fabric.rs` | Domain-certificate exactness, deterministic subdivision, GPU queue ownership, overflow fallback, and packet-oracle contracts |
| Create | `$SPECTRA/rust/spectra-renderer/examples/geometry_backend_conformance.rs` | CUDA/Vulkan/Metal hardware adapter and compute witness |
| Create | `$SPECTRA/rust/spectra-renderer/examples/geometry_program_bench.rs` | Raw cluster, DGF/DGFS-style, and topology-program breakthrough bake-off |
| Create | `$SPECTRA/rust/spectra-renderer/examples/visibility_program_bench.rs` | Mesh/CLAS/program-to-triangle/ray-native end-to-end bake-off |
| Modify | `$SPECTRA/slang/apply_scene_delta.slang` | GPU transform scatter and dirty-partition bit marking |
| Create | `$SPECTRA/slang/geometry_compute.slang` | Shared deterministic subgroup baseline and cooperative helpers |
| Create | `$SPECTRA/slang/geometry_program_decode.slang` | Dense/program block decode into fixed transient cluster arenas |
| Create | `$SPECTRA/slang/visibility_intersection.slang` | AOT bounded primitive-family intersection, stable surface keys, materials, UVs, and normals |
| Create | `$SPECTRA/slang/visibility_bounds.slang` | GPU parameter update, conservative-envelope test, and procedural-AABB build records |
| Create | `$SPECTRA/slang/visibility_domain_classify.slang` | Conservative whole-domain hit/miss certificates and deterministic ambiguity subdivision |
| Create | `$SPECTRA/slang/visibility_route.slang` | Front-to-back cell routing and GPU bin construction by cell/family/ray class |
| Create | `$SPECTRA/slang/visibility_packet_intersect.slang` | Subgroup baseline and cooperative dense ray-by-surface coefficient tiles |
| Create | `$SPECTRA/slang/visibility_hit_reduce.slang` | Exact refinement, deterministic nearest-hit merge, and residual continuation |
| Create | `$SPECTRA/slang/mega_geometry_feedback.slang` | Quantized ray-hit/footprint aggregation and advisory prefetch state |
| Create | `$SPECTRA/slang/geometry_backend_lowering.slang` | Common-record to native-argument lowering entry points |
| Create | `$SPECTRA/slang/mega_geometry_prepare.slang` | GPU LOD, page-table, work compaction, and indirect args generation |
| Modify | `$SPECTRA/slang/restir_pt.slang` | Surface-correspondence remap across topology-changing LOD |
| Modify | `$OCHROMA/crates/vox_render/src/resident_renderer.rs` | Bridge typed stats/deltas into the live renderer |
| Modify then replace | `$OCHROMA/crates/vox_render/examples/clas_scale_bench.rs` | Early fail-closed witnesses; becomes final benchmark |
| Create | `$OCHROMA/crates/vox_render/examples/mega_geometry_bench.rs` | Final three-mode city suite |
| Modify | `$OCHROMA/crates/vox_render/Cargo.toml` | Register benchmark example/dependencies |
| Create | `$OCHROMA/crates/vox_render/tests/mega_geometry_bench_contract.rs` | Result schema and verdict-suppression tests |

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Actual AS reporting | Forced CLAS exits code 2 until `optixClusterAccelBuild` succeeds | Deriving clusters from prototype count |
| True CLAS | Live witness reports nonzero hardware builds and exact primitive mapping | Calling triangle GAS from a method named CLAS |
| GPU-owned preparation | Device kernels produce LOD cuts, page work, dirty partitions, and CLAS args/count while CPU decision counters stay zero | Rust loops that fill buffers before upload |
| Portable packet ABI | CUDA and Vulkan consume byte-identical common records and emit identical integer control hashes | Per-vendor CPU structs with similar field names |
| Capability honesty | Actual API features, matrix shapes/types, RT kind, and fingerprint are printed and cache-keyed | Guessing support from vendor or architecture name |
| Cooperative compute | Slang cooperative variant matches its subgroup oracle and is retained only on a complete-stage win | Checking an extension bit or renaming scalar math “tensor” |
| Native RT adapters | OptiX/Vulkan/Metal lower common GPU records without CPU translation or count readback | A generic trait whose implementations rebuild CPU vectors |
| Ray-native visibility | Real BlueprintGraph nodes traverse as native procedural AABBs, match the mesh oracle, and avoid fine triangle materialization/builds | Coarse proxies, SDF impostors, or meshes generated on CPU before tracing |
| Certified visibility fabric | Whole ray domains resolve only under strict conservative nearest/miss proofs; unresolved rays packetize coherently; complete cost beats scalar/native paths with zero CPU scheduling | Ray sorting alone, approximate tile hits, or tensor utilization that excludes failed certificates/queues/residuals |
| Parametric surface identity | Construction-path keys and coordinates survive tolerance and parameter changes with zero temporal invalidation | Guessing identity from world position or clearing history |
| Geometry programs | Decoded program blocks match provenance/material/UV oracles while repeated topology uses at most 25% of raw cluster bytes | Applying a general byte compressor to duplicated triangle pages |
| Pareto schedule cook | Two-to-four deterministic candidates expose complete objective vectors and admission selects a measured winner | One triangle-count-minimized hierarchy for every device/workload |
| Surface-stable LOD | Exact parent/child maps at least halve LOD-switch temporal invalidations without radiance bias | Clearing all temporal/ReSTIR history on every cut change |
| Ray-feedback detail | GPU hit/footprint feedback improves optional residency with identical required-parent sequence under zero/stale feedback | Using last-frame visibility as a correctness cull |
| Forge hierarchy | Repeated Meridian cooks are byte-identical with exact source coverage | Returning one source-order chunk |
| Hybrid policy | Measured static/changing fixtures select their actual winner | Hardcoding CLAS for all NVIDIA geometry |
| Hierarchical IAS | Localized one-percent dirty million-instance run keeps root stable with bounded opaque host submission | CPU traversal over instances/partitions |
| Continuous LOD | GPU-selected one-pixel, zero-seam live witness with 50% triangle reduction | CPU or whole-mesh distance selection |
| Paging | GPU-managed fixed pool runs 600 frames with no CPU page/eviction decisions or required miss | CPU pager or handle map called streaming |
| Evidence gate | Any loss suppresses the broad claim; correctness-only/parity fixtures cannot be mislabeled wins | Fixed success output |

## Outcome

Ship a truthful, deterministic city visibility system whose common GPU work graph and packet ABI are not owned by one vendor—and whose source representation is not forced through triangles or one traversal per ray. Supported Forge construction remains parametric; coherent ray domains can be certified in bulk; unresolved structured rays become factorized packet algebra; irregular residue returns to fixed-function RT; residual repetition becomes shared topology programs; unique geometry remains dense-block encoded; surface identity survives precision and LOD changes; and GPU ray feedback improves the optional working set.

The plan does not promise that CLAS wins every workload. The final claim requires zero losses, honest parity where `Auto` retains triangle GAS/OMM, and wins on the changing, streamed-LOD, localized-update, and mixed-city workloads under the exact correctness, performance, CPU-ownership, and VRAM rules in the design.

## Ground Truth Before Work Starts

- `spectra-optix/src/optix_host.rs::build_clas_scene` currently calls the ordinary triangle-GAS builder.
- The unused `build_proto_gas_over_clas` path contains the live `optixClusterAccelBuild` call.
- `clas_scale_bench` currently permits a synthetic fallback cluster count and therefore is not CLAS evidence.
- The recorded million-instance result reaches 1,000,000 instances but reports one cluster. It is IAS/shared-prototype evidence only.
- Spectra's pipeline currently permits a single GAS or one IAS over GAS and uses traversal depth 2.
- Spectra already has wavefront ray/hit buffers and an OptiX pass that traces those buffers before the Slang shade stage, providing the integration seam for a visibility fabric without moving shading to CPU.
- Current OptiX tracing is host-launched. The fabric therefore needs fixed admitted launch bounds and device-side count checks rather than pretending OptiX can consume a device-indirect launch count.
- Existing `ser_reorder.slang` reorganizes shader work but is not a portable whole-visibility scheduler or evidence that individual ray traversal was eliminated.
- `ReadyAssetPayload` schema 2 has discrete `mesh_lods` plus untyped `forge_clusters` JSON.
- Forge's current cluster manifold is a contiguous, full-detail, source-order schedule. It is not a continuous LOD hierarchy.
- Forge already owns typed BlueprintGraph DAGs, assembly/module instances, sweeps, extrusions, subdivision cages, and a reconstructable description, but `ForgeAsset` hands the renderer a mesh and cluster manifold. There is no retained visibility lowering or custom-primitive runtime path.
- The resident scene delta ring and indexed instance scatter are real and remain the update authority.
- Local OptiX 9.1 headers define `optixClusterAccelBuild` as an indirect multi-build whose argument array and optional count live in device memory; the plan must use that path rather than generating per-frame args on CPU.
- Ordinary OptiX IAS builds remain host-submitted. This is an explicit bounded submission island, not permission for CPU scene/geometry decisions.
- OMM foliage is real and stays in its validated triangle-GAS path until CLAS-plus-OMM has its own witness.
- `spectra-gpu` already has a broad `GpuBackend` trait plus CUDA, Vulkan, and macOS Metal implementations. Extend it; do not create a parallel generic compute framework.
- Slang already exposes `linalg::CoopMat`, so Spectra should own capability discovery, specialization, ABI validation, and measurement rather than a second portable matrix language.
- The current cross-backend trait exposes several raw-`u64` CUDA/RT seams and host-slice AS methods. MegaGeometry common records must stop at `GpuBufferHandle`; native pointers and host-shaped adapter calls cannot leak upward into the frame graph.
- The current cook/runtime pipeline flattens Forge output into independent clusters and optimizes a single hierarchy. It does not exploit repeated procedural topology, DGF/DGFS-style dense blocks, multiple schedule candidates, or parent/child surface mapping.
- Current LOD correctness covers seams and pixel error but not temporal/ReSTIR validity across topology changes. That omission can erase much of the path-tracing quality/performance benefit.

If any of these facts has changed when implementation begins, update the design and this plan before coding against the stale assumption.

## Repository Roots

Use these names throughout the plan:

```text
OCHROMA=/home/tom-espen/src/ochroma
SPECTRA=/home/tom-espen/src/spectra
FORGE=/home/tom-espen/src/forge
URBAN=/home/tom-espen/Ochroma/projects/urban_horizon
```

Resolve the live sibling paths with `git -C <root> rev-parse --show-toplevel` before editing. Preserve every dirty worktree and do not commit unless requested.

## Common Evidence Rules

Every benchmark run must print and store:

```text
MEGAGEOMETRY_ENV gpu=<name> api=<cuda|vulkan|metal> driver=<version> rt_backend=<optix|vulkan_khr|vulkan_nv|metal> capability_hash=<hash> shader_abi=<hash> cook_schema=<n>
MEGAGEOMETRY_CONFIG mode=<triangle|reference|city_hybrid> compute=<portable_subgroup|cooperative_matrix> fixture=<id> instances=<n> frames=<n> seed=<n>
MEGAGEOMETRY_CORRECT source_primitives=<n> mapped_primitives=<n> material_mismatches=0 uv_mismatches=0 hit_mismatches=0
MEGAGEOMETRY_RESULT build_ms_p50=<n> build_ms_p95=<n> update_ms_p50=<n> update_ms_p95=<n> native_lowering_ms_p95=<n> trace_ms_p50=<n> trace_ms_p95=<n> frame_ms_p50=<n> frame_ms_p95=<n> resident_vram_mb=<n> as_vram_mb=<n> page_vram_mb=<n> scratch_vram_mb=<n> peak_geometry_vram_mb=<n> traced_triangles=<n> required_page_misses=<n>
MEGAGEOMETRY_CPU render_thread_scene_visits=<n> visibility_node_visits=<n> construction_evaluations=<n> lod_decisions=<n> page_decisions=<n> eviction_decisions=<n> blocking_readbacks=<n> frame_allocations=<n> host_deadline_misses=<n> render_thread_submit_ms_p95=<n> host_submit_calls=<n> host_submit_ms_p95=<n>
MEGAGEOMETRY_PORTABLE packet_abi=<match|mismatch> control_hash=<match|mismatch> cooperative_error=<n> adapter_overhead_pct=<n> cpu_fallbacks=<n>
MEGAGEOMETRY_PROGRAM mode=<raw_clusters|dense_blocks|topology_programs> encoded_bytes=<n> decoded_bytes=<n> topology_templates=<n> reused_blocks=<n> decode_ms_p95=<n> decoded_hash=<hash>
MEGAGEOMETRY_VISIBILITY mode=<triangle|clas|program_to_triangles|ray_native> source_surface_coverage=<n> residual_coverage=<n> coarse_aabb_bytes=<n> materialized_geometry_bytes=<n> fine_as_builds=<n> intersection_ms_p95=<n> geometry_stage_ms_p95=<n> stable_surface_hash=<hash>
MEGAGEOMETRY_FABRIC mode=<native_scalar|certified_fabric|invocation_reorder> input_rays=<n> structured_rays=<n> certified_rays=<n> subdivided_domains=<n> explicit_rays=<n> active_packet_lanes=<n> routing_ms_p95=<n> certification_ms_p95=<n> packet_ms_p95=<n> refinement_ms_p95=<n> residual_rt_ms_p95=<n> merge_ms_p95=<n> queue_overflows=<n> cpu_scheduling=<n> hit_hash=<hash>
MEGAGEOMETRY_TEMPORAL surface_map_coverage=<n> lod_switch_invalidations=<n> radiance_mismatches=<n> feedback_owner=gpu required_sequence_match=<true|false>
MEGAGEOMETRY_BREAKTHROUGH repeated_storage_ratio=<n> mixed_storage_ratio=<n> fixed_budget_detail_ratio=<n> lod_temporal_invalidations_reduction=<n> winning_axes=<n> claim=<scoped>
MEGAGEOMETRY_VISIBILITY_BREAKTHROUGH ray_native_coverage=<n> materialized_geometry_ratio=<n> parameter_update_fine_as_builds=<n> geometry_stage_speedup=<n> fixed_budget_detail_ratio=<n> ray_native_temporal_invalidations=<n> correctness_mismatches=<n> real_asset_families=<n> losses=<n> claim=<scoped>
MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage=<n> certified_without_per_ray_traversal=<n> active_packet_lanes=<n> intersection_speedup_vs_scalar=<n> geometry_stage_speedup=<n> routing_queue_overhead_pct=<n> correctness_mismatches=<n> cpu_scheduling=<n> real_asset_families=<n> losses=<n> claim=<scoped>
```

The JSON result contains the same fields, raw samples, fixture content hashes, command line, git revisions, render-config hash, warmup count, and timed-frame count.

Timed runs must:

- use a release build;
- warm up 60 frames and measure at least 120 frames;
- use the same 1 spp present path, camera path, internal resolution, and shaders for compared modes;
- exclude shader compilation, payload cook, disk cache construction, and device calibration;
- abort the verdict on a correctness failure;
- abort the verdict if any forbidden CPU counter is nonzero or CPU cost scales with total instances/clusters/pages;
- abort the verdict if common packet bytes/control hashes differ across supported backends, an adapter translates records on CPU, or capability/reflection hashes do not match the loaded pipelines;
- compare `PortableSubgroup` with every eligible cooperative specialization at complete-stage and full-frame boundaries; `Auto` may select cooperative compute only when it is no more than 2% slower and does not increase peak geometry VRAM;
- compare raw clusters, DGF/DGFS-style dense blocks, and topology programs using identical decoded geometry, schedules, camera/ray traces, and full-frame boundaries;
- compare mesh oracle, CLAS, program-to-triangle decode, and ray-native custom intersection from the same validated construction source and count the full AABB-build/intersection/shading/residual cost;
- compare scalar custom intersection, certified domains plus subgroup packets, certified domains plus cooperative packets, optional invocation reordering, and native triangle/CLAS using identical rays; include failed certificates, subdivision, routing, queue traffic, refinement, residual RT, merge, and peak working memory;
- report visible source-equivalent detail only for geometry that passes the one-pixel, seam, hit, material, UV, motion-vector, and radiance oracles;
- collect detailed device stats only after timed frames or through delayed asynchronous telemetry;
- exit non-zero if a requested backend was unavailable or silently fell back.

Do not run the game, a remote build, or a remote render until the user explicitly authorizes that execution. Local compile and unit tests are allowed during implementation.

---

## Gate A: Prove Ray-Native Visibility Before Freezing the Stack

**Why first:** Portable adapters, CLAS wiring, compression, and GPU work graphs are still downstream of triangle materialization. The larger hypothesis is that much of a city can remain a compact construction program through ray intersection, eliminating tessellation, fine-detail AS builds, and topology-changing LOD. This gate tests that on real authored assets before either visibility or geometry-program ABIs become expensive to change.

**Files**

- Add `$FORGE/crates/mesh/src/visibility_program.rs` as an explicitly disposable finite visibility IR
- Add `$FORGE/crates/mesh/tests/visibility_program.rs`
- Add `$FORGE/crates/building/src/blueprint/visibility.rs` and `$FORGE/crates/building/tests/visibility_lowering.rs`
- Add `$FORGE/crates/mesh/src/geometry_program.rs` as an explicitly disposable experimental encoder until the gate passes
- Add `$FORGE/crates/mesh/tests/geometry_program.rs`
- Modify `$URBAN/src/bin/game_asset_cook/cli.rs` and `$URBAN/src/bin/game_asset_cook/forge_runner.rs` to expose a census mode through the real importer
- Add `$URBAN/assets/config/mega_geometry_corpus.ron`
- Add `$SPECTRA/rust/spectra-renderer/examples/geometry_program_bench.rs` only for the minimal GPU decode/build probe; do not build the final integration yet
- Add `$SPECTRA/rust/spectra-renderer/examples/visibility_program_bench.rs` for the minimal native procedural-AABB probe
- Add `$SPECTRA/slang/geometry_program_decode.slang` only for the bounded probe codec
- Add `$SPECTRA/slang/visibility_intersection.slang` and `$SPECTRA/slang/visibility_bounds.slang` only for finite AOT primitive families

**Wiring requirement:** The real Forge importer evaluates each checked-in BlueprintGraph/directive twice: the unchanged mesh path supplies the oracle, while `lower_visibility_graph` supplies a generic finite construction graph. `cook_visibility_program` feeds those exact records unchanged to native custom-primitive traversal; unsupported subgraphs feed the residual geometry-program probe. Synthetic repetition is a sensitivity fixture, never the authorizing corpus.

**Implementation steps**

- [ ] Select a checked-in corpus spanning repeated Forge-authored structures, imported unique buildings, roads/props, multi-material meshes, and awkward seam-heavy geometry; store its ordered asset ids and content hashes.
- [ ] Dual-lower supported real construction nodes into oriented prisms, planar polygons, extrusions, profile sweeps, repetition lattices, bounded interval operations, and explicit residual leaves. Report source-surface coverage by primitive family and asset, not only an aggregate.
- [ ] Assign stable construction-path surface keys and parametric coordinates. Fire deterministic oracle rays, including grazing, inside-origin, seams, openings, negative transforms, and multi-hit cases, against both the evaluated mesh and visibility program.
- [ ] Validate conservative AABBs, fixed child/step/memory limits, parameter-update envelopes, and local residualization. Arbitrary bytecode, runtime graph interpretation, and per-asset shader compilation are forbidden.
- [ ] Compute decoded-identity topology equivalence within and across assets. Report frequency distribution and source-byte coverage, not only the best repeated asset.
- [ ] Measure quantized parameter/residual entropy, dictionary/index/hash overhead, page padding, dependency amplification, dense-block lower bounds, and the fraction of blocks that remain unique.
- [ ] Attempt deterministic parent/child surface correspondence and report exact valid, ambiguous, seam-rejected, and out-of-domain counts.
- [ ] Decode emitted blocks on GPU into a fixed arena and include transfer, decode, build, trace, and peak-memory cost. A CPU decoder may validate the oracle but cannot provide performance evidence.
- [ ] Keep `TopologyProgramV1` finite and non-programmable: fixed block kinds, bounds, widths, memory regions, and worst-case work. Reject arbitrary bytecode or per-asset shader compilation.
- [ ] Probe native custom-primitive intersection over coarse program-domain AABBs. A repetition lattice computes the relevant cell directly or with a bounded DDA; it may not expand to one leaf per repeated component.
- [ ] Compare mesh triangles, true CLAS, program-to-triangle decode, and ray-native traversal including AABB build, intersection divergence/occupancy, shading, residuals, updates, peak VRAM, and full-frame time.
- [ ] Do not freeze the payload schema unless the result selects one of three explicit outcomes: `ray_native`, `geometry_programs`, or `dense_only`.

**Local preflight**

```bash
cd "$URBAN"
cargo run --release --bin game_asset_cook -- --mega-geometry-census assets/config/mega_geometry_corpus.ron --json-out artifacts/geometry-program-census.json
```

Required lines:

```text
MEGAGEOMETRY_CENSUS corpus_hash=<hash> assets=<n> repeated_source_coverage=<n> unique_source_coverage=<n> topology_entropy_bits_per_triangle=<n> residual_bits_per_vertex=<n> repeated_ratio_lower_bound<=0.20 mixed_ratio_lower_bound<=0.40 page_overhead_included=true
MEGAGEOMETRY_VISIBILITY_CENSUS ray_native_source_coverage>=0.60 residual_coverage<=0.40 primitive_families=<n> stable_surface_coverage=1.0 unbounded_nodes=0 runtime_codegen=0
MEGAGEOMETRY_SURFACE_PREFLIGHT valid_coverage>=0.99 ambiguous=<n> seam_rejected=<n> out_of_domain=0
```

The minimal GPU probe runs on available supported hardware after the normal permission gate:

```bash
cd "$SPECTRA/rust"
cargo run --release -p spectra-renderer --example geometry_program_bench -- --probe --corpus "$URBAN/artifacts/geometry-program-census.json" --modes raw_clusters,dense_blocks,topology_programs --json-out artifacts/geometry-program-probe.json
cargo run --release -p spectra-renderer --example visibility_program_bench -- --probe --corpus "$URBAN/artifacts/geometry-program-census.json" --modes triangle,clas,program_to_triangles,ray_native --json-out artifacts/visibility-program-probe.json
```

Required line:

```text
MEGAGEOMETRY_PROGRAM_PROBE decoded_hash=match cpu_decode=0 fixed_arena=true transfer_decode_build_trace_ms_p95=<n> full_frame_regression_pct<=2.0 peak_geometry_vram_bytes=<n>
MEGAGEOMETRY_VISIBILITY_PROBE hit_hash=match material_hash=match uv_normal_hash=match stable_surface_hash=match ray_native_coverage>=0.60 materialized_geometry_ratio<=0.25 parameter_update_fine_as_builds=0 geometry_stage_speedup>=1.50 cpu_geometry=0
```

Requested-but-unavailable execution exits nonzero; compile-only evidence cannot pass Gate A.

**Done When**

- [ ] The checked-in corpus represents real production asset classes and has stable content hashes.
- [ ] Lower bounds include every byte required for independent paging and safe decode, not just idealized vertex payloads.
- [ ] At least two real asset families pass exact mesh-oracle intersection and shading identity through native custom primitives.
- [ ] Ray-native regions have fixed worst-case intersection work, conservative bounds, stable surface identity, and zero fine-AS builds for in-envelope parameter updates.
- [ ] Surface-map failures are localized and classified rather than discarded from the denominator.
- [ ] GPU transfer-plus-decode-plus-build stays within the eventual 2% full-frame-regression budget on the probe fixture.
- [ ] The decision is written as `MEGAGEOMETRY_HYPOTHESIS primary=<ray_native|geometry_programs|dense_only> ray_native_reason=<measured_reason> residual_reason=<measured_reason>`; only measured paths enter the frozen schema and Tasks 4–11.

---

## Task 0: Lock the GPU/Host Boundary Before Building Features

**Why first:** “GPU-driven” is otherwise easy to fake with CPU loops that merely upload their answers. OptiX CLAS supports device-indirect arguments/counts, while ordinary IAS and storage remain host API boundaries.

**Files**

- Modify `$SPECTRA/rust/spectra-optix/src/optix_host.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/traversal.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/render_state.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$OCHROMA/crates/vox_render/src/resident_renderer.rs`
- Add `$SPECTRA/rust/spectra-optix/tests/indirect_clas_contract.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/cpu_loop_contract.rs`

**Wiring requirement:** `Renderer` initializes the CPU-ownership counters with resident scene state; `OptiXTraversal` records host API submissions, and `ResidentSceneRenderer` exposes the counters without forcing a per-frame synchronization.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run them and confirm they fail because the indirect/CPU-ownership contract does not yet exist.
- [ ] Encode allowed load/structural CPU work and the two allowed steady-state host services: IAS API submission and opaque page transport.
- [ ] Add counters for render-thread instance/prototype/visibility-node/cluster/page visits, CPU construction evaluation, CPU LOD/tolerance decisions, CPU page-priority decisions, CPU eviction decisions, blocking readbacks, per-frame geometry allocations, fixed render-thread submission time, host submission calls, and host submission time.
- [ ] Add typed wrappers proving that CLAS args, CLAS argsCount, output handles, and output sizes are device pointers with fixed admitted capacities.
- [ ] Forbid a null `argsCount` in the dynamic/LOD CLAS path; a null count is allowed only for a static admission build whose fixed count is not camera-dependent.
- [ ] Ensure detailed geometry stats are copied only after an explicit diagnostic request or asynchronously at a configured sparse interval.
- [ ] Add a fatal diagnostic in witness mode if any forbidden CPU counter becomes nonzero.
- [ ] Add test-only negative probes that deliberately perform one forbidden CPU decision, one blocking stats read, and one frame allocation; verify each counter increments and suppresses the verdict. Default-zero counters alone are not evidence.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-optix --test indirect_clas_contract
cargo test -p spectra-renderer --test cpu_loop_contract
```

Required lines:

```text
test dynamic_clas_requires_device_args_and_device_count ... ok
test stats_collection_is_not_in_the_present_dependency_chain ... ok
test cpu_geometry_decisions_are_forbidden ... ok
test host_services_accept_only_opaque_bounded_packets ... ok
test negative_probe_increments_each_forbidden_counter ... ok
test nonzero_cpu_counter_suppresses_winner ... ok
```

**NVIDIA boundary witness, after permission**

Required line:

```text
MEGAGEOMETRY_CPU_BOUNDARY clas_args=device clas_count=device clas_outputs=device regular_ias_submission=host page_transport=host render_thread_geometry_decisions=0 blocking_readbacks=0
```

**Done When**

- [ ] Every later task can name whether its state is load-time, GPU-resident, or an opaque host-service packet.
- [ ] Device-indirect CLAS is mechanically enforced before true CLAS wiring begins.
- [ ] No benchmark can print a winner without the CPU-ownership counters.

---

## Task 1: Establish the Portable GPU Work Graph and Cooperative-Compute Contract

**Why now:** The common contract must exist before OptiX types spread through cook, scene state, policy, and paging. Spectra already has CUDA, Vulkan, and Metal `GpuBackend` implementations and Slang has cooperative matrices; this task extends those seams instead of creating another framework.

**Files**

- Modify `$SPECTRA/rust/spectra-gpu/src/backend.rs`
- Create `$SPECTRA/rust/spectra-gpu/src/capabilities.rs`
- Modify `$SPECTRA/rust/spectra-gpu/src/lib.rs`
- Modify `$SPECTRA/rust/spectra-gpu/src/cudarc_backend/cuda_real/backend_impl.rs`
- Modify `$SPECTRA/rust/spectra-gpu/src/vulkan_backend.rs`
- Modify `$SPECTRA/rust/spectra-gpu/src/metal_backend.rs`
- Create `$SPECTRA/rust/spectra-gpu/tests/capability_contract.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs`
- Create `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_backend.rs`
- Create `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_compute.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/sample.rs`
- Create `$SPECTRA/rust/spectra-renderer/tests/geometry_backend_conformance.rs`
- Create `$SPECTRA/rust/spectra-renderer/examples/geometry_backend_conformance.rs`
- Create `$SPECTRA/slang/geometry_compute.slang`
- Create `$SPECTRA/slang/geometry_backend_lowering.slang`
- Modify `$SPECTRA/rust/spectra-renderer/build.rs`

**Wiring requirement:** Device initialization freezes `GpuCapabilities`; `Renderer::new` validates the capability/reflection/cache fingerprint and selects precompiled geometry variants; the live sample path emits `GeometryBuildRecord`s and calls the selected `GeometryAccelBackend` with `GpuBufferHandle`s and an opaque queue token. No common code may create an OptiX/Vulkan/Metal struct or inspect a GPU-written count.

**Implementation steps**

- [ ] Write the capability, layout, packet, cross-backend control-hash, negative-fallback, and variant-selection tests first; confirm they fail because the contract does not exist.
- [ ] Add `GpuCapabilities` to the existing `GpuBackend` trait. Query real API/version, subgroup width, cooperative shapes/types/scopes/alignment, indirect dispatch, hardware RT, native procedural-AABB/custom-intersection support, shader/invocation reordering, indirect/address-driven AS build, sparse residency, queue/timeline support, driver, and compiler. Do not infer features from vendor names.
- [ ] Hash the canonical capability serialization. Include the hash in AOT artifact, reflection, pipeline-cache, geometry-calibration, and benchmark keys. Prove that changing one shape/type/driver/ABI field invalidates the key.
- [ ] Define versioned, `#[repr(C, align(16))]`, fixed-width `GeometryBuildRecord` and header fields. Validate Rust sizes/offsets against Slang reflection for PTX, SPIR-V, and Metal targets. Put native argument structs only in `geometry_backend_lowering.slang` and the final native adapter.
- [ ] Replace common raw device pointers with `GpuBufferHandle`; add opaque `GpuQueueToken`/`GpuEpoch` ownership checks. The adapter may resolve a native address only after validating backend identity, schema, ABI hash, capacity, generation, and lifetime.
- [ ] Add `GeometryAccelBackend` implementations for OptiX and Vulkan KHR RT with truthful triangle, cluster, and procedural-AABB/custom-intersection capability bits. Vulkan uses a fixed admitted build schedule plus GPU-written indirect primitive counts/zeroed inactive slots; it never reads a variable build count. Add a truthful Metal adapter boundary that reports unavailable for GPU-record-driven AS builds until address-driven support is present; Metal compute/conformance remains real and may not synthesize RT, cluster, or procedural support.
- [ ] Implement `PortableSubgroup` as the required path for all geometry control kernels. Use integer/fixed-point keys and stable ids for LOD/page/partition/build decisions; require byte-identical output records on CUDA and Vulkan.
- [ ] Implement exact Slang `linalg::CoopMat` specializations only for a representative matrix-shaped geometry stage such as batched affine/bounds transform or block-linear page decode. Do not route scans, radix sort, allocation, dirty-bit compaction, or BVH traversal through this layer.
- [ ] Precompile and prewarm subgroup/cooperative variants at structural admission. No frame-time Slang invocation, shader JIT, pipeline creation, virtual call per tile, per-dispatch allocation, or CPU emulation is allowed.
- [ ] Give each kernel family a bounded checked-in specialization manifest. Do not AOT-compile the Cartesian product of every enumerated shape/type/scope; unsupported capability combinations remain data, not shader variants.
- [ ] Measure the complete eligible stage, including packing, synchronization, and dispatch. `Auto` selects a cooperative variant only when correctness passes, peak VRAM does not increase, and stage p95 is no more than 2% slower than the subgroup path. A losing specialization remains diagnostic and is not selected.
- [ ] Add artifact inspection or native performance-counter evidence that `compute=cooperative_matrix` actually lowered to supported cooperative instructions. An extension bit, source spelling, or faster time alone is insufficient.
- [ ] Audit existing cooperative/neural shaders and route reusable capability checks through this contract; do not silently convert unrelated neural/material work as part of this task.
- [ ] Make requested-but-unavailable adapters fail closed. Metal prints `support=experimental` until a named Apple Silicon hardware run passes; a compile-only or green skip cannot promote support.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-gpu --test capability_contract -- --nocapture
cargo test -p spectra-renderer --test geometry_backend_conformance -- --nocapture
```

Required lines:

```text
test capability_fingerprint_is_canonical_and_cache_complete ... ok
test unsupported_features_are_not_inferred_from_vendor ... ok
test rust_and_slang_geometry_record_layouts_match ... ok
test native_types_do_not_cross_the_common_packet_boundary ... ok
test persistent_control_path_rejects_floating_cooperative_output ... ok
test specialization_manifest_is_bounded_and_explicit ... ok
test stale_queue_epoch_is_rejected_after_device_reset ... ok
test requested_unavailable_adapter_fails_closed ... ok
test custom_intersection_capability_is_queried_not_vendor_inferred ... ok
test invocation_reorder_capability_is_queried_not_vendor_inferred ... ok
```

**CUDA/Vulkan hardware conformance, after permission**

```bash
cargo run --release -p spectra-renderer --example geometry_backend_conformance -- --backends cuda,vulkan --compute auto --warmup 60 --frames 240
```

Required lines:

```text
MEGAGEOMETRY_PORTABLE backends=cuda,vulkan packet_abi=match control_hash=match cpu_fallbacks=0 frame_allocations=0 adapter_overhead_pct<=2.0
MEGAGEOMETRY_COMPUTE backend=cuda requested=auto selected=<portable_subgroup|cooperative_matrix> oracle=match cooperative_error<=0.001 stage_regression_pct<=2.0 instruction_proof=pass
MEGAGEOMETRY_COMPUTE backend=vulkan requested=auto selected=<portable_subgroup|cooperative_matrix> oracle=match cooperative_error=<n<=0.001|n/a> stage_regression_pct<=2.0 instruction_proof=<pass|unsupported_or_not_selected>
```

**Metal graduation gate, on named Apple Silicon hardware and after permission**

```bash
cargo run --release -p spectra-renderer --example geometry_backend_conformance -- --backends metal --compute auto --warmup 60 --frames 240
```

Required line:

```text
MEGAGEOMETRY_PORTABLE backend=metal support=supported packet_abi=match control_hash=match cpu_fallbacks=0 frame_allocations=0 adapter_overhead_pct<=2.0 present_witness=pass
```

**Done When**

- [ ] Common geometry state, packets, policies, and benchmarks contain no vendor-native structs, raw pointers, or OptiX-specific acceleration enum.
- [ ] CUDA and Vulkan share one packet ABI and bit-identical persistent control output without CPU translation.
- [ ] Cooperative compute is truthful, optional, and selected only by complete-stage measurement.
- [ ] Unsupported paths remain on ordinary GPU compute or truthful native triangle RT; there is no live CPU fallback.
- [ ] Metal cannot be advertised as supported until its independent hardware gate passes.

---

## Task 2: Replace Synthetic CLAS Claims With Truthful Acceleration Stats

**Depends on:** Tasks 0 and 1

**Why first:** No later benchmark is meaningful while triangle GAS can be labeled CLAS.

**Files**

- Modify `$SPECTRA/rust/spectra-optix/src/optix_host.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/optix_host_unavailable.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/traversal.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/lib.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs`
- Modify `$OCHROMA/crates/vox_render/src/resident_renderer.rs`
- Modify `$OCHROMA/crates/vox_render/examples/clas_scale_bench.rs`
- Add `$SPECTRA/rust/spectra-optix/tests/accel_stats_contract.rs`

**Wiring requirement:** `ResidentSceneRenderer::clas_stats` in `$OCHROMA/crates/vox_render/src/resident_renderer.rs` becomes an explicit diagnostic `collect_geometry_stats` path after `Renderer`/`OptiXTraversal` propagation. It is never a present dependency, and no legacy tuple call remains.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Extend the Task-1 `GeometryAccelKind`/`RayAccelBackendKind` contract with asynchronously collected `GeometryAccelStats` state and public read-only accessors as specified by the design.
- [ ] Record a triangle-GAS build only where `optixAccelBuild` creates a triangle GAS.
- [ ] Record a hardware CLAS build only after `optixClusterAccelBuild` succeeds and returns a live traversable.
- [ ] Remove the tuple `clas_stats()` API and migrate all callers to the typed stats object.
- [ ] Make the unavailable backend return `GeometryAccelKind::Unavailable`, zero hardware builds, and a stable reason.
- [ ] Remove `clas_scale_bench` fallback values based on instance/prototype counts. Preserve the filename for command compatibility, but change its header to `MEGAGEOMETRY_*`.
- [ ] Add `--require-accel triangle_gas|optix_clas`; mismatch exits with code 2 before timing.
- [ ] Make the benchmark synchronize and collect detailed stats only after the timed frame interval.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-optix --test accel_stats_contract
```

Exact output contract:

```text
test triangle_gas_is_not_reported_as_clas ... ok
test unavailable_backend_never_synthesizes_hardware_counts ... ok
```

```bash
cd "$OCHROMA"
cargo check -p vox_render --example clas_scale_bench
```

Exact final line:

```text
Finished `dev` profile
```

**NVIDIA witness, after permission**

Run the existing example with `--require-accel optix_clas`. Until Task 3 lands, it must fail exactly with:

```text
MEGAGEOMETRY_ERROR requested=optix_clas actual=triangle_gas reason=hardware_clas_not_built
```

**Done When**

- [ ] The two contract tests pass.
- [ ] No production or benchmark call site exposes the old tuple stats.
- [ ] The pre-Task-3 CLAS witness fails closed with the exact error above.

---

## Task 3: Wire True OptiX CLAS Into the Live Scene Path

**Depends on:** Tasks 1 and 2

**Files**

- Modify `$SPECTRA/rust/spectra-optix/src/optix_host.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/optix_host_unavailable.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/traversal.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/lib.rs`
- Modify `$SPECTRA/rust/spectra-optix/ptx/programs/device_programs.cu`
- Modify `$SPECTRA/rust/spectra-scene-state/src/layers.rs`
- Modify `$SPECTRA/rust/spectra-scene-state/src/lib.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/sample.rs`
- Add `$SPECTRA/rust/spectra-optix/tests/cluster_provenance.rs`

**Wiring requirement:** `OptiXTraversal::build_clas_scene_device` in `$SPECTRA/rust/spectra-optix/src/traversal.rs` becomes the OptiX implementation of `GeometryAccelBackend`; `sample.rs` submits common device records/count through `encode_geometry_builds`, whose GPU lowering kernel writes the native OptiX args/count without host inspection.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Replace the triangle-GAS body of `build_clas_scene` with admission and calls to the existing true CLAS implementation.
- [ ] Split any prototype exceeding queried OptiX CLAS limits into deterministic source-order clusters at admission for this first hardware witness. Task 4 replaces that temporary schedule with cooked spatial clusters; no frame-time CPU partitioner is allowed.
- [ ] Allocate fixed-capacity common build records/count plus adapter-private triangle/template args, GAS-over-CLAS args, native counts, output handles, output sizes, and overflow flags.
- [ ] Lower common records to OptiX argument structs on GPU, then call `optixClusterAccelBuild` with nonzero device pointers for `argsArray` and `argsCount`; chain GAS-over-CLAS from device output handles without host readback.
- [ ] Add scene-state buffers for:
   - cluster descriptors;
   - cluster-local primitive to prototype-level triangle mapping;
   - prototype-level triangle base in the packed scene.
- [ ] Validate exactly one mapping entry per clustered primitive, no duplicates or gaps, and bounds against the prototype triangle range.
- [ ] Update closest-hit so `clusterId` and local primitive id resolve through the mapping before reading triangle material, UV, normal, and weathering data.
- [ ] Keep triangle-GAS closest-hit unchanged and make the oracle compare the final packed scene triangle id.
- [ ] Retain buffers and cluster build inputs for the full traversable lifetime.
- [ ] Keep common records, native counts, and handles device-resident through trace and collect them only after the witness timing interval.
- [ ] Propagate all OptiX errors; do not convert a failed CLAS build into a successful empty scene.
- [ ] Keep OMM prototypes on the triangle-GAS path and print `fallback=omm_not_yet_validated_with_clas`.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-optix --test cluster_provenance
cargo test -p spectra-scene-state
cargo check -p spectra-renderer
```

Exact test lines:

```text
test reordered_clusters_preserve_packed_triangle_ids ... ok
test duplicate_source_triangle_is_rejected ... ok
test missing_source_triangle_is_rejected ... ok
test multi_material_cluster_mapping_is_exact ... ok
```

If local `spectra-native` loading fails because `libslang.so`, CUDA, OptiX, or driver resources are absent, record the exact loader error and do not call it a code failure or a passing hardware witness.

**NVIDIA witness, after permission**

```powershell
cargo run --release -p vox_render --example clas_scale_bench --features spectra-native,spectra-native-optix -- --require-accel optix_clas --fixture meridian_house_01 --instances 1 --frames 120 --json-out artifacts/megageometry/task3-clas.json
```

Required lines:

```text
MEGAGEOMETRY_ACCEL accel=optix_clas prototypes=1 triangle_gas_builds=0 hardware_clas_builds=1 clusters_gt_prototypes=true
MEGAGEOMETRY_CORRECT source_primitives=<n> mapped_primitives=<same-n> material_mismatches=0 uv_mismatches=0 hit_mismatches=0
MEGAGEOMETRY_CPU clas_args=device clas_count=device blocking_readbacks=0 frame_allocations=0
```

**Done When**

- [ ] The hardware witness calls `optixClusterAccelBuild` and reports a nonzero hardware build count.
- [ ] The build consumes device args/count and performs no count/handle readback before trace.
- [ ] The real multi-material fixture has exact hit-buffer parity.
- [ ] No current source path calls `build_proto_triangle_gas` while reporting `backend=optix accel=cluster`.

---

## Task 4: Compile Forge Visibility Programs, Geometry Residuals, Surface Maps, and Schedule Candidates

**Depends on:** Task 1's custom-intersection capability/ABI contract and Task 3's provenance contract. Pure dual-lowering, cook, and oracle work can proceed locally; native intersection evidence remains permission- and hardware-gated.

**Files**

- Add `$FORGE/crates/mesh/src/mega_geometry.rs`
- Add `$FORGE/crates/mesh/src/geometry_program.rs`
- Add `$FORGE/crates/mesh/src/visibility_program.rs`
- Modify `$FORGE/crates/mesh/src/lib.rs`
- Modify `$FORGE/crates/mesh/Cargo.toml`
- Modify `$FORGE/crates/building/src/asset.rs`
- Modify `$FORGE/crates/building/src/directive/completion.rs`
- Add `$FORGE/crates/building/src/blueprint/visibility.rs`
- Modify `$FORGE/crates/building/src/blueprint/mod.rs`
- Add `$FORGE/crates/building/tests/visibility_lowering.rs`
- Add `$FORGE/crates/mesh/tests/mega_geometry.rs`
- Add `$FORGE/crates/mesh/tests/geometry_program.rs`
- Add `$FORGE/crates/mesh/tests/visibility_program.rs`
- Add `$OCHROMA/crates/vox_data/src/mega_geometry.rs`
- Add `$OCHROMA/crates/vox_data/tests/mega_geometry_asset.rs`
- Modify `$OCHROMA/crates/vox_data/src/lib.rs`
- Modify `$URBAN/src/asset/payload.rs`
- Modify `$URBAN/src/asset/validation.rs`
- Modify `$URBAN/src/bin/game_asset_cook/cli.rs`
- Modify `$URBAN/src/bin/game_asset_cook/forge_runner.rs`
- Modify `$URBAN/src/bin/game_asset_cook/tests/mesh.rs`
- Modify `$URBAN/docs/superpowers/specs/2026-07-19-unified-forge-content-directive.md`
- Add `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_program.rs`
- Add `$SPECTRA/rust/spectra-renderer/src/renderer/visibility_program.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/geometry_program.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/visibility_program.rs`
- Add `$SPECTRA/rust/spectra-renderer/examples/geometry_program_bench.rs`
- Add `$SPECTRA/rust/spectra-renderer/examples/visibility_program_bench.rs`
- Add `$SPECTRA/slang/geometry_program_decode.slang`
- Add `$SPECTRA/slang/visibility_intersection.slang`
- Add `$SPECTRA/slang/visibility_bounds.slang`
- Modify `$SPECTRA/rust/spectra-optix/ptx/programs/device_programs.cu`
- Modify `$SPECTRA/rust/spectra-renderer/build.rs`

**Wiring requirement:** Before mesh structure is discarded, the validated BlueprintGraph path calls `lower_visibility_graph`; the existing mesh evaluation remains the oracle. `forge_runner.rs` passes the generic construction graph to `cook_visibility_program`, then passes that validated result with the evaluated mesh to `cook_mega_geometry` so ray-native plus residual coverage is exact, and writes the combined typed result into ReadyAssetPayload v3. `visibility_program_bench` traces those exact visibility records through native procedural AABBs, while `geometry_program_bench` decodes only residual blocks through `GpuBackend`. No CPU-expanded or proxy benchmark surrogate is allowed.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Add a pure, game-agnostic Forge mesh cook that accepts indexed positions, per-triangle material/seam keys, and versioned settings.
- [ ] Add a second deterministic lowering from BlueprintGraph construction to the generic finite `VisibilityConstructionGraph` without changing the existing evaluated mesh.
- [ ] Cook supported oriented prisms, polygons, extrusions, sweeps, repetition lattices, and bounded interval operations into data-only visibility nodes with fixed child/step/memory limits, conservative AABBs, source provenance, material domains, and stable surface keys.
- [ ] Cook a deterministic sparse visibility-cell DAG and factorized coefficient blocks for admitted planes/half-spaces, local transforms, and bounded polynomials. Record conservative cell bounds, front-to-back traversal relations, side domains, exact-refinement nodes, maximum domain subdivision, and decoded hashes.
- [ ] Residualize unsupported or ill-conditioned subgraphs locally into the ordinary mesh path. Validate that every source surface is covered exactly once by either ray-native or residual representation.
- [ ] Implement shared AOT Slang intersection math for every admitted primitive family plus thin native entry wrappers where the RT API requires them, including the OptiX CUDA/PTX wrapper. Every target consumes the same reflected records and emits exact material/UV/normal/surface-key output. The runtime may dispatch by finite node kind but may not interpret BlueprintGraph or compile asset-specific shaders.
- [ ] Admit parameter envelopes. In-envelope changes update only GPU parameter buffers; an envelope escape emits GPU procedural-AABB refit/rebuild records and cannot trace stale bounds.
- [ ] Create 4-to-256-triangle clusters with deterministic spatial centroid ordering and source triangle id tie-breaking.
- [ ] Never cross material, opacity class, hard-normal, or locked boundary seams within a simplification operation.
- [ ] Build parent cluster groups bottom-up with locked shared boundaries and monotonic geometric error.
- [ ] Preserve generic Forge construction provenance where available, then independently hash decoded topology/attributes to discover reusable templates within and across assets. Authoring labels are hints only and cannot establish equality.
- [ ] Factor accepted reuse into shared topology templates, quantized canonical parameters, per-use transform/material/semantic parameters, and bounded residual blocks. Preserve exact source/material/UV provenance after decode.
- [ ] Implement a dense independently decodable unique-geometry path and compare it with public DGF/DGFS format/codec behavior. Do not require native DGF hardware and do not call a general compressed triangle blob a geometry program.
- [ ] Emit deterministic parent/child surface correspondence with parent primitive, quantized barycentric/parametric coordinates, orientation/material domain, validity, and bounded error. Reject ambiguous or out-of-domain mappings.
- [ ] Emit two-to-four bounded schedule candidates spanning spatial locality, topology reuse, dense packing, page locality, update cost, and boundary quality. Record the complete objective vector; never choose solely by triangle count.
- [ ] Pack immutable pages by group dependency and stable ids. Emit page content hashes.
- [ ] Emit complete source triangle provenance for the full-detail level and explicit cooked-level triangle ids for simplified levels.
- [ ] Replace Forge's source-only completion check with validation of:
   - full source coverage exactly once;
   - cluster sizes 4 through 256 except a documented final tiny remainder;
   - valid acyclic parent groups;
   - monotonic error;
   - boundary-lock consistency;
   - complete page dependencies.
- [ ] After Gate A selects `ray_native`, `geometry_programs`, or `dense_only`, define typed `ReadyMegaGeometry` in Ochroma `vox_data`; Urban Horizon embeds that engine-owned type in `ReadyAssetPayload`, bumps `READY_PAYLOAD_VERSION` from 2 to 3, and keeps the old untyped JSON only as migration input. Other games must be able to serialize the same type without depending on Urban Horizon.
- [ ] Add the ordinary subgroup GPU decoder and fixed transient decode arena. Cooperative residual decode is optional and must use the Task-1 selection contract.
- [ ] Benchmark raw independent clusters, DGF/DGFS-style dense blocks, and topology programs on one repetition-heavy structural fixture and one unique/mixed fixture using identical decoded geometry and build/trace boundaries.
- [ ] Benchmark `triangle`, `clas`, `program_to_triangles`, and `ray_native` from identical construction sources/rays, including coarse AABB build, programmable intersection, shading, residual, update, peak-VRAM, and full-frame boundaries.
- [ ] Move runtime-authoritative LOD schedule creation out of Urban gameplay code. `game_asset_cook` adapts the Forge output and never invents camera-specific meshes.
- [ ] Add a deterministic version-2 migration only if old payload inputs exist in the current asset pipeline; otherwise reject version 2 with the focused recook command.

**Local verification**

```bash
cd "$FORGE"
cargo test -p forge-mesh --test mega_geometry
cargo test -p forge-mesh --test geometry_program
cargo test -p forge-mesh --test visibility_program
cargo test -p forge-building --test visibility_lowering
```

Required tests:

```text
test cook_is_byte_identical_across_repeated_runs ... ok
test spatial_clusters_cover_every_source_triangle_once ... ok
test clusters_respect_material_and_boundary_seams ... ok
test group_errors_are_monotonic ... ok
test page_dependencies_are_acyclic ... ok
test topology_reuse_requires_decoded_identity ... ok
test unique_geometry_uses_dense_block_fallback ... ok
test surface_correspondence_is_complete_and_bounded ... ok
test schedule_portfolio_is_bounded_deterministic_and_non_dominated ... ok
test visibility_lowering_covers_every_source_surface_once ... ok
test unsupported_subgraph_becomes_local_residual ... ok
test visibility_nodes_have_conservative_bounds_and_finite_work ... ok
test visibility_cells_cover_program_domains_without_gaps ... ok
test factorized_coefficients_match_scalar_node_evaluation ... ok
test stable_surface_keys_survive_tolerance_and_parameter_changes ... ok
```

```bash
cd "$URBAN"
cargo test --bin game_asset_cook mesh
cargo test --lib asset::validation
```

Required tests:

```text
test ready_payload_v3_round_trips_typed_mega_geometry ... ok
test v2_payload_has_explicit_migration_or_recook_error ... ok
test cooked_meridian_cluster_mapping_preserves_materials ... ok
```

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test geometry_program -- --nocapture
cargo test -p spectra-renderer --test visibility_program -- --nocapture
cargo run --release -p spectra-renderer --example geometry_program_bench -- --modes raw_clusters,dense_blocks,topology_programs --warmup 60 --frames 240
cargo run --release -p spectra-renderer --example visibility_program_bench -- --modes triangle,clas,program_to_triangles,ray_native --warmup 60 --frames 240
```

Required lines:

```text
test subgroup_decode_matches_position_index_material_uv_and_provenance_oracles ... ok
test decode_arena_has_fixed_capacity_and_zero_frame_allocations ... ok
test surface_map_round_trip_preserves_parent_domain ... ok
test ray_native_hits_match_mesh_material_uv_normal_and_surface_oracles ... ok
test in_envelope_parameter_update_builds_no_fine_geometry ... ok
test custom_intersection_unavailable_uses_cooked_residual_not_cpu ... ok
MEGAGEOMETRY_PROGRAM fixture=repeated_structure mode=topology_programs decoded_hash=match encoded_ratio_vs_raw<=0.25 cpu_decode=0
MEGAGEOMETRY_PROGRAM fixture=mixed_unique mode=<dense_blocks|topology_programs> decoded_hash=match encoded_ratio_vs_raw<=0.50 full_frame_regression_pct<=2.0
MEGAGEOMETRY_VISIBILITY fixture=<real-family> mode=ray_native hit_hash=match source_surface_coverage>=0.60 materialized_geometry_ratio<=0.25 parameter_update_fine_as_builds=0 geometry_stage_speedup>=1.50 cpu_geometry=0
```

Run the focused cook without launching the game:

```bash
cd "$URBAN"
cargo run --release --bin game_asset_cook -- --source assets/source/buildings --only city.res_high.l5.8x8.meridian_house_01 --output artifacts/cook/meridian_house_01
```

Required final line:

```text
MEGAGEOMETRY_COOK asset=city.res_high.l5.8x8.meridian_house_01 schema=3 deterministic=true max_cluster_triangles=256 source_coverage=exact material_mismatches=0 ray_native_surface_coverage=<n> residual_coverage=<n> visibility_cells=<n> coefficient_blocks=<n> stable_surface_coverage=1.0 schedule_candidates=<2..4> topology_templates=<n> surface_map_coverage=1.0
```

**Done When**

- [ ] Repeated cooks are byte-identical.
- [ ] `ReadyAssetPayload` no longer relies on untyped JSON at render time.
- [ ] At least two real asset families dual-lower with exact source coverage, stable surface keys, local residuals, and mesh-oracle hit/material/UV/normal parity.
- [ ] Ray-native candidates meet the Gate-A materialization, update-build, and complete geometry-stage targets; otherwise their exact unsupported regions remain on the measured geometry-program/dense path.
- [ ] Repetition-heavy geometry uses at most 25% and mixed/unique geometry at most 50% of raw independent-cluster bytes without exceeding 2% full-frame regression.
- [ ] Every accepted topology-changing LOD schedule has complete bounded surface correspondence.
- [ ] The selected portable representation decodes on GPU into the Task-3 build contract without CPU expansion.
- [ ] The existing Meridian asset crosses Forge, cook, payload validation, and renderer conversion without visual-authoring changes.

---

## Gate B: Prove Certified Visibility Domains and GPU Packet Execution

**Depends on:** Task 4's exact visibility IR/cell/coefficients and Task 1's capability, reflection, queue, and cooperative-kernel contract.

**Why before scene integration:** Scalar custom intersection still inherits divergence, register pressure, and occupancy loss from the one-ray programming model. Sorting rays alone only rearranges that work. This gate tests the larger hypothesis: coherent ray domains can eliminate individual traversals under exact certificates, unresolved rays can be scheduled as dense GPU work, and irregular residue can return to fixed-function RT cores without a CPU scheduler.

**Files**

- Add `$SPECTRA/rust/spectra-renderer/src/renderer/visibility_fabric.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/sample.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/spp_loop.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/render_state.rs`
- Modify `$SPECTRA/rust/spectra-renderer/build.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/traversal.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/visibility_fabric.rs`
- Modify `$SPECTRA/rust/spectra-renderer/examples/visibility_program_bench.rs`
- Add `$SPECTRA/slang/visibility_domain_classify.slang`
- Add `$SPECTRA/slang/visibility_route.slang`
- Add `$SPECTRA/slang/visibility_packet_intersect.slang`
- Add `$SPECTRA/slang/visibility_hit_reduce.slang`

**Wiring requirement:** The existing wavefront camera, shadow, and bounce-ray buffers feed `VisibilityGpuPipeline::dispatch_visibility`. The fixed AOT GPU graph classifies and subdivides ray domains, routes explicit rays front-to-back, bins them, intersects packet coefficients, refines boundaries, emits a residual queue, invokes the backend's fixed native RT pass, and merges nearest hits before the existing shade megakernel. Device counts remain device handles. Vulkan may use indirect trace dimensions; an OptiX adapter launches a pre-admitted fixed maximum and device-side bounds-checks the count. The host submits the same bounded pass sequence every time and may not read a count, select a mode, loop to queue exhaustion, or inspect a hit before shading.

**Implementation steps**

- [ ] Define fixed-width `VisibilityDispatchInput`, ray-domain, explicit-ray, bin-range, candidate-hit, and queue-stat records with Rust/Slang reflection tests. Allocate all queues once from admitted scene/frame bounds.
- [ ] Seed primary screen tiles as perspective ray domains. Add coherent shadow/continuation domains only where their origin/direction bounds are conservative; otherwise seed explicit rays directly.
- [ ] Implement outward-rounded interval containment and hit-parameter bounds for plane/half-space, prism, extrusion, repetition, and admitted bounded-sweep coefficient families. A common hit certificate requires its worst valid hit to precede every competing structured surface and residual/native-AABB lower bound over the whole domain; a miss certificate requires conservative exclusion of every structured and residual candidate.
- [ ] Split ambiguous domains in deterministic child order to a fixed depth/explicit-ray threshold. Silhouette, disocclusion, numerical-boundary, and high-curvature regions must become explicit rays, never approximate hits.
- [ ] Route explicit rays monotonically through the sparse visibility-cell DAG and bin by `(cell, primitive_family, ray_class)` using GPU scans/atomics plus indirect dispatch. Record failed-certificate and routing cost separately.
- [ ] Implement a subgroup packet baseline for every factorized coefficient family. Use cooperative matrices only for dense ray-by-plane/transform/polynomial dot tiles after domain certification; cooperative/reduced-precision output can never establish a whole-domain proof. Subgroup code owns division, interval logic, masks, refinement, and deterministic nearest-hit reduction.
- [ ] Route boundary cases to robust scalar refinement and unsupported/irregular geometry to the native triangle/CLAS/custom residual queue. Merge certified, packet, scalar, and native candidates by the same deterministic `t`/surface-key tie rule as the oracle.
- [ ] When a bin is underfilled, choose subgroup/scalar/native execution on GPU from admitted thresholds. Do not pad packets or report inactive lanes as useful work.
- [ ] Make optional NVIDIA/Vulkan invocation reordering a separately measured adapter mode. It cannot define the portable ABI or replace the subgroup scheduler on backends without the feature.
- [ ] Execute a fixed bounded pass graph. Native APIs that lack device-indirect launch use admitted maximum dimensions and device-side count checks; the cost of empty work remains in the measurement.
- [ ] On any queue overflow, set a device flag and retain the already-admitted native path for correctness. The benchmark suppresses the fabric claim; no CPU resize, drain, replay, or scheduling path is allowed.
- [ ] Compare `native_scalar`, `certified_subgroup`, `certified_cooperative`, optional `invocation_reorder`, triangle GAS, and CLAS with identical primary, shadow, and bounce-ray captures. Attribute certification, subdivision, routing, packet, refinement, residual RT, merge, fixed-empty-launch, and full geometry-stage time.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test visibility_fabric -- --nocapture
cargo check -p spectra-renderer --example visibility_program_bench
```

Required tests:

```text
test certified_hit_requires_strict_whole_domain_nearest_proof ... ok
test ambiguous_domain_splits_or_materializes_exact_rays ... ok
test certified_miss_conservatively_excludes_every_candidate ... ok
test subdivision_and_hit_reduction_are_deterministic ... ok
test all_queue_counts_remain_device_handles ... ok
test optix_fixed_launch_does_not_read_device_ray_count ... ok
test queue_overflow_retains_native_correctness_and_suppresses_claim ... ok
test subgroup_packet_hits_match_scalar_and_mesh_oracles ... ok
test cooperative_tiles_cannot_bypass_exact_refinement ... ok
test sparse_bins_do_not_pad_or_force_cooperative_work ... ok
```

**Hardware gate, after permission**

```bash
cd "$SPECTRA/rust"
cargo run --release -p spectra-renderer --example visibility_program_bench -- --fabric --modes native_scalar,certified_subgroup,certified_cooperative,invocation_reorder,triangle,clas --warmup 60 --frames 240 --json-out artifacts/visibility-fabric-probe.json
```

Requested unsupported modes exit nonzero. The comparison must report all modes rather than silently omitting a loss. Required claim:

```text
MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage>=0.60 certified_without_per_ray_traversal>=0.35 active_packet_lanes>=0.75 intersection_speedup_vs_scalar>=2.0 geometry_stage_speedup>=2.0 routing_queue_overhead_pct<=10.0 correctness_mismatches=0 cpu_scheduling=0 real_asset_families>=2 losses=0 claim=certified_visibility_fabric
```

**Done When**

- [ ] Domain certificates are exact against mesh/scalar/native hit, material, UV, normal, motion, and surface-key oracles, including silhouettes, openings, grazing rays, and competing surfaces.
- [ ] At least 35% of measured rays avoid per-ray traversal, at least 60% enter the structured fabric overall, and remaining packets sustain at least 75% active lanes on two real Forge asset families.
- [ ] The complete fabric is at least 2x faster than scalar custom intersection and the winning triangle/CLAS/custom geometry stage after all overhead and fixed empty native launches.
- [ ] Every queue/count/mode decision is GPU-owned; CPU scheduling and readbacks are zero.
- [ ] Failure leaves Task 4's scalar ray-native or native residual path intact and removes `CertifiedFabric` from the shipping policy. It does not weaken the gate.

---

## Task 5: Carry Cooked Visibility and Geometry Through Ochroma Into Spectra

**Depends on:** Task 4

**Files**

- Modify `$URBAN/src/spectra_frame/mesh_convert.rs`
- Modify `$URBAN/src/spectra_frame/scene_build.rs`
- Modify `$URBAN/src/spectra_frame/mod.rs`
- Modify `$OCHROMA/crates/vox_render/src/resident_renderer.rs`
- Modify `$SPECTRA/rust/spectra-scene-state/src/layers.rs`
- Modify `$SPECTRA/rust/spectra-scene-state/src/resident.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/optix_host.rs`
- Add `$URBAN/src/spectra_frame/mega_geometry_conversion_tests.rs`

**Wiring requirement:** The structural/load invocation of `build_instanced_scene_cached_with_focus_and_clock` in `$URBAN/src/spectra_frame/scene_build.rs` attaches visibility programs/nodes/cells/coefficients/parameters, residual clusters, topology templates, program blocks, surface maps, and schedule candidates to `MegaGeometryLayer`; admission selects representations, allocates fixed domain/ray/bin/residual/intersection/decode state, and passes common records to `VisibilityGpuPipeline` and `GeometryAccelBackend`. The steady-state present path may not revisit payloads, interpret BlueprintGraph, generate topology, or resize queues.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Add generic scene-state descriptors for cooked prototype cluster ranges, group ranges, pages, errors, bounds, and provenance offsets.
- [ ] Carry topology-template ranges, encoded program blocks, decoded hashes, parent/child surface maps, and bounded schedule candidates without flattening or duplicating them in Urban/Ochroma.
- [ ] Carry visibility programs, finite node kinds, sparse visibility cells, factorized coefficient blocks, conservative bounds, parameter buffers, stable surface keys, and residual ranges without converting ray-native regions back into meshes.
- [ ] Convert asset-local cooked triangle ids to packed-scene triangle ids exactly once in `mesh_convert`.
- [ ] Keep ids sorted by `(prototype, level, group, cluster)` and reject duplicate ids.
- [ ] Upload immutable metadata with the resident prototype state, not every frame.
- [ ] Call `resolve_geometry_schedule` at admission with the versioned capability/calibration key; store only the selected variant id in resident render state. A cache miss chooses the conservative dense/raw candidate while calibration remains outside timed frames.
- [ ] Key admission by scene/prototype content hash and prove that unchanged frames reuse the resident layer/work buffers without rebuilding `GeometryLayer` or revisiting payloads.
- [ ] Change the Task-3 OptiX CLAS adapter to consume cooked cluster boundaries when present. Source-order runtime partitioning remains diagnostic fallback only and is labeled as such.
- [ ] Ensure ordinary triangle-GAS prototypes ignore cluster metadata without changing their output.

**Local verification**

```bash
cd "$URBAN"
cargo test mega_geometry_conversion
```

Required lines:

```text
test packed_scene_mapping_matches_cooked_provenance ... ok
test conversion_is_stable_under_asset_input_permutation ... ok
test triangle_gas_ignores_cluster_metadata_without_output_drift ... ok
test unchanged_frames_perform_zero_geometry_admissions ... ok
test topology_templates_remain_shared_after_scene_packing ... ok
test surface_maps_survive_asset_and_scene_id_remapping ... ok
test schedule_selection_never_changes_cooked_topology ... ok
test visibility_programs_survive_scene_packing_without_mesh_expansion ... ok
test visibility_cells_and_coefficients_survive_packing_byte_exact ... ok
test stable_surface_keys_survive_asset_and_scene_id_remapping ... ok
```

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-scene-state
cargo check -p spectra-renderer
```

**NVIDIA witness, after permission**

Repeat Task 3 with `--cluster-source cooked`. Required line:

```text
MEGAGEOMETRY_ACCEL accel=optix_clas cluster_source=forge_cooked hardware_clas_builds=1 source_coverage=exact
MEGAGEOMETRY_CPU geometry_admissions_during_frames=0 payload_visits_during_frames=0
```

**Done When**

- [ ] The live CLAS path consumes the typed Forge schedule.
- [ ] The live procedural/fabric paths consume the typed Forge visibility program, cells, coefficients, and residuals without frame-time graph interpretation or triangle generation.
- [ ] No frame-time path parses Forge JSON or rebuilds cluster topology.
- [ ] Triangle GAS and CLAS still resolve the same packed scene triangles.

---

## Task 6: Add Measured Ray-Native, GAS, CLAS, and Geometry-Schedule Selection

**Depends on:** Task 5

**Files**

- Add `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_policy.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/render_state.rs`
- Modify `$OCHROMA/crates/vox_render/src/resident_renderer.rs`
- Modify `$OCHROMA/crates/vox_render/examples/clas_scale_bench.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/geometry_policy.rs`

**Wiring requirement:** Prototype/subgraph admission in `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs` calls `resolve_geometry_accel`, `resolve_geometry_schedule`, and the structural visibility-execution calibration once per calibration epoch. It stores certified-fabric/scalar-custom/native-residual and triangle/cluster selections on separate resident policy axes and records their complete cost/reason vectors in `RenderState`. The frame loop only consumes those bits; per-bin fill/fallback decisions remain GPU-owned.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Add backend-neutral `Auto`, `Triangle`, `Cluster`, and `RayNative` policy values. Diagnostic CLI may request composite backend modes, but scene state stores structure and backend on separate axes.
- [ ] Add separate `VisibilityExecutionPolicy::{Auto, NativeScalar, CertifiedFabric}` and measured packet variants. Do not encode fabric selection as a fake acceleration-structure kind.
- [ ] Define immutable cooked-domain traits: static, deforming, page-changing, OMM, continuous-LOD, ray-native family/work bound, residual coverage, triangle count, cluster count, and material feature mask.
- [ ] Add a versioned per-device calibration record keyed exactly as described in the design.
- [ ] Run calibration outside timed frames and persist only diagnostic device data.
- [ ] Measure each bounded cooked schedule on decode, native lowering/build, trace, update, page traffic, and peak memory. Discard dominated candidates and select from the Pareto set by the configured workload objective; triangle count alone is never the objective.
- [ ] For every ray-native candidate, include coarse-AABB build/refit, intersection-shader instructions/divergence/occupancy, shading, residual trace, materialized bytes, parameter updates, temporal invalidations, and peak memory. Fewer triangles alone cannot select it.
- [ ] For every fabric candidate, include certification attempts/failures, subdivision, routing, queue atomics/traffic, empty launches, packet fill, cooperative setup, exact refinement, native residual trace, hit merge, peak working memory, and full geometry-stage/frame cost. Certificate rate or active lanes alone cannot select it.
- [ ] Resolve policy only at cooked-domain admission/calibration invalidation; prohibit per-frame domain/prototype walks or policy calls.
- [ ] Select compacted triangle GAS conservatively on a cache miss.
- [ ] Keep OMM on triangle GAS until a dedicated combined witness exists.
- [ ] Print requested policy, selected kind, selection reason, and calibration key for every benchmark fixture.
- [ ] Print schedule candidate objective vectors, selected variant id, selection reason, and calibration key.
- [ ] Add forced modes only for diagnostics; forced unsupported CLAS exits non-zero.
- [ ] Forced unsupported `ray_native` exits non-zero; `Auto` may use only already-cooked residuals and cannot ask CPU code to evaluate construction.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test geometry_policy
```

Required tests:

```text
test static_uncalibrated_defaults_to_triangle_gas ... ok
test omm_defaults_to_validated_triangle_path ... ok
test deforming_candidate_uses_clas_only_after_measured_win ... ok
test calibration_key_changes_with_shader_or_cook_schema ... ok
test policy_does_not_enter_simulation_or_save_state ... ok
test policy_resolution_never_runs_in_frame_prepare ... ok
test schedule_selection_uses_complete_cost_vector ... ok
test dominated_schedule_is_never_selected ... ok
test schedule_calibration_cannot_change_topology_or_simulation ... ok
test ray_native_requires_complete_stage_win_not_triangle_count ... ok
test ray_native_and_residual_modes_can_coexist_within_one_asset ... ok
test custom_intersection_cache_miss_defaults_to_cooked_triangle_residual ... ok
test certified_fabric_selection_requires_complete_stage_win ... ok
test visibility_execution_policy_is_separate_from_accel_kind ... ok
test per_bin_fill_decisions_never_run_on_cpu ... ok
```

**NVIDIA gate, after permission**

Run `triangle`, forced `clas`, forced `ray_native`, and `auto` on static, structural, and changing fixtures. Required verdicts:

```text
MEGAGEOMETRY_POLICY fixture=static_meridian requested=auto selected=triangle_gas reason=measured_trace_cost
MEGAGEOMETRY_POLICY fixture=changing_geometry requested=auto selected=optix_clas reason=measured_update_win
MEGAGEOMETRY_POLICY fixture=ray_native_structure requested=auto selected=procedural reason=measured_complete_stage_win
MEGAGEOMETRY_VISIBILITY_POLICY fixture=ray_native_structure requested=auto selected=<certified_fabric|native_scalar> reason=<measured_complete_stage_win|conservative_cache_miss>
MEGAGEOMETRY_SCHEDULE fixture=mixed_city candidates=<2..4> selected=<id> dominated_selected=false cost_vector=complete
MEGAGEOMETRY_VERDICT static_trace_regression_pct<=2.0 changing_update_winner=optix_clas correctness=pass
MEGAGEOMETRY_CPU prototype_policy_calls_during_frames=0 render_thread_prototype_visits=0
```

If CLAS does not win the changing fixture, retain triangle GAS and close this task with `selected=triangle_gas reason=no_measured_clas_win`; do not advance a CLAS performance claim.

**Done When**

- [ ] Auto selection is measured, versioned, and fail-safe.
- [ ] Static geometry does not regress by more than 2% in trace p95.
- [ ] The benchmark reports the actual selected structure and reason.

---

## Task 7: Replace the Flat Million-Instance Top Level With Hierarchical IAS Partitions

**Depends on:** Tasks 0, 1, 5, and 6

**Files**

- Modify `$SPECTRA/rust/spectra-optix/src/optix_host.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/optix_host_unavailable.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/traversal.rs`
- Modify `$SPECTRA/rust/spectra-scene-state/src/resident.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/mod.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/sample.rs`
- Add `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_gpu.rs`
- Modify `$SPECTRA/slang/apply_scene_delta.slang`
- Add `$SPECTRA/slang/mega_geometry_prepare.slang`
- Modify `$OCHROMA/crates/vox_render/examples/clas_scale_bench.rs`
- Add `$SPECTRA/rust/spectra-optix/tests/hierarchical_ias.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/geometry_gpu.rs`

**Wiring requirement:** `apply_scene_delta.slang` scatters transforms and marks the device dirty bitset; `GeometryGpuPipeline::dispatch_prepare` compacts it into the next-frame opaque partition packet. A dedicated `OptiXTraversal` worker submits that packet; `Renderer::apply_scene_delta_and_refit_ias` performs no CPU command/instance/partition walk.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Change pipeline traversable graph flags from the current single-level restriction to the OptiX flag that permits root IAS to child IAS to triangle-GAS/CLAS/procedural-AABB traversables.
- [ ] Set and verify traversable graph depth 3 and recompute stack sizes through the OptiX utility path.
- [ ] Partition static instances deterministically at structural admission by stable spatial cell and id. Put continuously moving instances in a dedicated dynamic IAS with one unconditional update submission; do not migrate them through static partitions per frame.
- [ ] Keep one child IAS and backing instance buffer per partition.
- [ ] Store stable partition ids in resident GPU instance records. In `apply_scene_delta.slang`, scatter each transform into the next-frame buffer and atomically mark its partition bit.
- [ ] Compact the device bitset into a deterministic id-sorted, fixed-capacity partition packet on GPU. Detect overflow on device; never rebuild a CPU list.
- [ ] Double-buffer the packet one frame ahead. Transfer only the bounded opaque packet asynchronously to the dedicated submission worker; the render thread never waits on that transfer.
- [ ] Submit only packet-named child IASes. Root IAS updates only when a child traversable handle changes.
- [ ] Publish completed build epochs from the worker. Present consumes only an already-complete epoch; on a missed deadline it may retain the prior memory-safe but visibly stale static epoch, increments `host_deadline_misses`, and suppresses the verdict without blocking. Dynamic transforms are never labeled correct when stale.
- [ ] Pre-record/retain all child IAS metadata so the worker performs O(packet_count) API submission only and never visits scene geometry.
- [ ] Preserve the flat IAS path as benchmark baseline and unsupported-driver fallback.
- [ ] Add counters for dirty partitions, child updates, root updates, and full rebuilds.
- [ ] Never label this implementation PTLAS.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-optix --test hierarchical_ias
cargo test -p spectra-scene-state resident
cargo check -p spectra-renderer
```

Required tests:

```text
test partition_assignment_is_deterministic ... ok
test one_instance_edit_marks_one_expected_partition ... ok
test child_transform_update_keeps_root_handle_stable ... ok
test structural_partition_change_updates_root_once ... ok
test multi_level_stack_depth_is_three ... ok
test moving_instances_use_dynamic_ias_without_static_migration ... ok
test render_thread_never_compacts_dirty_partitions ... ok
test missed_worker_deadline_retains_previous_epoch_without_blocking ... ok
```

**NVIDIA scale gate, after permission**

```powershell
cargo run --release -p vox_render --example clas_scale_bench --features spectra-native,spectra-native-optix -- --fixture mixed_city --instances 1000000 --dirty-percent 1 --dirty-distribution localized,uniform --top-level flat,hierarchical --frames 240 --json-out artifacts/megageometry/task6-ias.json
```

Required verdict:

```text
MEGAGEOMETRY_PARTITIONS instances=1000000 dirty_percent=1 distribution=localized root_updates=0 full_rebuilds=0 dirty_partitions=<n> child_updates=<same-n>
MEGAGEOMETRY_VERDICT hierarchical_update_ms_p95<=2.0 speedup_vs_flat>=4.0 correctness=pass
MEGAGEOMETRY_CPU render_thread_scene_visits=0 render_thread_partition_visits=0 partition_decisions=0 blocking_readbacks=0 host_deadline_misses=0 render_thread_submit_ms_p95<=0.25 host_submit_ms_p95<=0.50
```

Also run uniform one-percent and 100% dirty. Hierarchical may be no more than 10% slower than the flat baseline; otherwise admission policy chooses flat/dynamic IAS for those distributions. It does not switch by walking dirtiness on CPU per frame.

**Done When**

- [ ] A normal local edit keeps the root IAS handle stable.
- [ ] The one-percent-dirty gate passes.
- [ ] The render thread performs zero geometry/partition visits and the bounded host island stays below 0.50 ms p95.
- [ ] All-dirty behavior has a measured fallback rather than an unconditional partition claim.

---

## Task 8: Wire Forge-Cooked Continuous Cluster LOD

**Depends on:** Tasks 3, 4, and 6

**Files**

- Modify `$SPECTRA/rust/spectra-scene-state/src/resident.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/sample.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_gpu.rs`
- Modify `$SPECTRA/slang/mega_geometry_prepare.slang`
- Modify `$SPECTRA/slang/geometry_compute.slang`
- Modify `$SPECTRA/slang/geometry_program_decode.slang`
- Modify `$SPECTRA/slang/visibility_intersection.slang`
- Modify `$SPECTRA/slang/visibility_bounds.slang`
- Add `$SPECTRA/slang/mega_geometry_feedback.slang`
- Modify `$SPECTRA/slang/restir_pt.slang`
- Modify `$OCHROMA/crates/vox_render/src/resident_renderer.rs`
- Modify `$URBAN/src/spectra_frame/mod.rs`
- Modify `$URBAN/src/spectra_frame/gpu_frame.rs`
- Modify `$URBAN/src/spectra_frame/witnesses.rs`
- Modify `$URBAN/assets/config/render.ron`
- Modify `$SPECTRA/rust/spectra-renderer/tests/geometry_gpu.rs`

**Wiring requirement:** The interactive path dispatches GPU geometry preparation; ray-native regions receive an evaluation-tolerance/ray-class update without topology replacement, residual regions receive a cooked cluster cut, closest-hit/trace writes quantized group hit/footprint feedback, and `restir_pt.slang` consumes stable parametric surface keys or validated residual surface maps. No Rust LOD selector, feedback interpreter, or global history reset exists.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Add config-backed maximum screen-space error and hysteresis settings.
- [ ] Upload camera constants already required for rendering; do not expose selected groups or projected errors to CPU.
- [ ] In `mega_geometry_prepare.slang`, compute projected group error and select a deterministic compatible cut with hysteresis, neighbor, and locked-boundary constraints.
- [ ] For ray-native surfaces, derive a bounded evaluation tolerance/ray class from the same conservative error contract while retaining identical construction-path surface keys. Do not tessellate or switch topology merely to change visible precision.
- [ ] Store selected groups in resident GPU state and compact only changed groups into device-indirect CLAS/template args.
- [ ] Reuse resident group traversables or update CLAS from GPU-written cached-template inputs; do not invoke runtime QEM or read back `argsCount`.
- [ ] Remove the interactive-render blanket disable for this cooked path. Preserve the disable only for legacy runtime simplification.
- [ ] Add a no-LOD oracle mode and a deterministic seam/error witness.
- [ ] Accumulate quantized per-group hit density, ray-footprint/cone class, neighboring-miss pressure, and LOD-transition confidence into double-buffered GPU feedback. Use stable integer reductions and clear/rotate entirely on GPU.
- [ ] Apply one-frame-delayed feedback only after the conservative camera/error cut to rank optional detail/prefetch. Zero, stale, overflowed, or disabled feedback must produce the same required-parent sequence.
- [ ] Use cooked parent/child correspondence to remap primary-hit identity, motion vectors, and eligible ReSTIR/temporal reservoirs across cut changes. Reject invalid/high-error mappings locally; do not clear an entire object's history.
- [ ] Use direct parametric coordinates for ray-native motion/temporal reuse. Its expected invalidation count from precision-only changes is zero; cross-LOD maps remain the residual geometry path.
- [ ] Add no-feedback, no-remap, and highest-detail oracles plus fast camera, disocclusion, and repeated threshold-crossing fixtures.
- [ ] Ensure gameplay never chooses a distance mesh or substitutes geometry by camera range.
- [ ] Add CPU-ownership counters proving zero frame-time instance/group visits and zero CPU LOD decisions.
- [ ] Compile common LOD/page selection for both NVIDIA and AMD GPU targets. Keep only the final OptiX argument-packing kernel NVIDIA-specific; AMD cannot use a CPU selection fallback.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test geometry_gpu
```

Required tests:

```text
test group_cut_is_id_stable ... ok
test hysteresis_prevents_threshold_flapping ... ok
test neighboring_groups_keep_locked_boundaries ... ok
test selected_cut_respects_one_pixel_error ... ok
test interactive_path_never_calls_runtime_qem ... ok
test selected_cut_and_clas_args_remain_device_resident ... ok
test frame_prepare_has_zero_cpu_lod_decisions ... ok
test amd_and_nvidia_use_the_same_gpu_group_cut_contract ... ok
test zero_or_stale_feedback_cannot_remove_required_geometry ... ok
test feedback_reduction_is_integer_stable_across_backends ... ok
test valid_surface_map_preserves_hit_material_uv_and_motion_identity ... ok
test invalid_surface_map_rejects_reuse_locally ... ok
test ray_native_tolerance_change_preserves_surface_key_and_temporal_reuse ... ok
test ray_native_precision_change_materializes_no_triangles ... ok
```

```bash
cd "$URBAN"
cargo test spectra_frame::witnesses::mega_geometry_lod
```

Required final line:

```text
MEGAGEOMETRY_LOD source=forge_cooked selector=gpu feedback_owner=gpu surface_map_coverage=1.0 max_error_px=1.0 open_boundary_seams=0 runtime_qem_calls=0 cpu_lod_decisions=0
```

**Live present gate, after permission**

Run the fixed long-range camera path in the actual 1 spp present path. Required verdict:

```text
MEGAGEOMETRY_VERDICT open_boundary_seams=0 max_error_px<=1.0 traced_triangle_reduction_pct>=50.0 hit_mismatches=0 radiance_mismatches=0 lod_temporal_invalidations_reduction>=0.50 required_sequence_match=true cpu_lod_decisions=0 blocking_readbacks=0
```

Inspect the rendered frames. Numeric gates alone are not visual evidence.

**Done When**

- [ ] Interactive rendering consumes cooked cluster LOD.
- [ ] The real present witness has no visible cracks or material changes.
- [ ] The triangle reduction and one-pixel gates pass.
- [ ] Valid surface remapping at least halves LOD-switch temporal invalidations without radiance bias.
- [ ] Ray-native precision changes have zero topology-driven temporal invalidations and materialize no fine triangles.
- [ ] Ray feedback improves optional residency/quality on its fixture and cannot affect required correctness.

---

## Task 9: Add a GPU-Managed Fixed Geometry Page Pool

**Depends on:** Task 8

**Files**

- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_gpu.rs`
- Add `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_page_transport.rs`
- Modify `$SPECTRA/slang/mega_geometry_prepare.slang`
- Modify `$SPECTRA/slang/geometry_program_decode.slang`
- Modify `$SPECTRA/slang/visibility_intersection.slang`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/render_state.rs`
- Modify `$SPECTRA/rust/spectra-scene-state/src/resident.rs`
- Modify `$OCHROMA/crates/vox_render/src/resident_renderer.rs`
- Modify `$URBAN/src/spectra_frame/mod.rs`
- Modify `$URBAN/assets/config/render.ron`
- Modify `$SPECTRA/rust/spectra-renderer/tests/geometry_gpu.rs`
- Add `$SPECTRA/rust/spectra-renderer/tests/geometry_page_transport.rs`

**Wiring requirement:** Scene admission preallocates encoded-page and transient-decode arenas. `GeometryGpuPipeline::dispatch_prepare` owns request priority, slot allocation, eviction, residual decode work, and page-table changes. `GeometryPageTransport` can only move selected encoded ranges into selected slots; ray-native node/parameter pages remain directly intersectable, while GPU decode produces common cluster input only for residual blocks and validates decoded hashes before native build.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Preallocate one fixed geometry page pool and all page-table/free-stack/request/completion buffers at scene admission from the config-backed budget. No per-frame page allocation/free is allowed.
- [ ] Add GPU-resident, requested, uploading, and evictable page states keyed by content hash, stable page id, slot, and generation.
- [ ] Carry codec id/version, compressed/decoded byte counts, alignment, and checksums in every page. Decode on GPU with an ordinary subgroup kernel first; a cooperative block-linear/learned decoder is eligible only after an end-to-end transfer-plus-decode win.
- [ ] Keep geometry-program pages encoded in VRAM when profitable and decode only selected blocks into a fixed transient arena. Reuse decoded blocks by content hash/generation; never materialize the full asset merely because one page is requested.
- [ ] Keep visibility-node and parameter pages directly addressable without triangle expansion. Their page dependencies are independently bounded so a ray-native domain cannot force the full asset resident.
- [ ] Implement the exact deterministic request ordering, slot allocation, and eviction selection in `mega_geometry_prepare.slang`.
- [ ] Keep the lowest-error parent group resident as the fallback for every visible asset.
- [ ] Emit fixed-size opaque transfer records containing only source-pack id, content hash, source range, destination slot, byte count, and slot generation.
- [ ] Make the transport worker perform file I/O/asynchronous copy only. It may validate bounds/generation but may not reprioritize, choose a slot/victim, or substitute a page.
- [ ] Add a transport-codec adapter boundary. Windows DirectStorage GDeflate/Zstd GPU decompression or future native equivalents may satisfy the byte-decode stage; the portable path remains asynchronous read plus Spectra GPU decode. CPU decompression is forbidden in live geometry.
- [ ] Apply transfer completions and page-table visibility on GPU only at frame boundaries after generation/fence validation.
- [ ] Discard delayed completions whose slot generation no longer matches; prove they cannot corrupt a reused slot or resurrect an evicted page.
- [ ] Track required-page misses, optional-detail misses, upload bytes, evictions, peak bytes, synchronous readbacks, CPU wait time, CPU page decisions, CPU eviction decisions, and frame allocations.
- [ ] Remove or rename the old `OptiXTraversal` handle-map “streaming” API so it cannot be mistaken for paging.
- [ ] Add a deterministic replay test over the same camera/request trace and GPU page-table sequence.
- [ ] Add camera-teleport, oscillating-budget, delayed-I/O, and multi-frame-in-flight eviction tests. Required parents remain visible; optional detail may degrade without holes.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test geometry_gpu
cargo test -p spectra-renderer --test geometry_page_transport
```

Required tests:

```text
test request_order_is_deterministic ... ok
test pager_never_exceeds_budget ... ok
test required_parent_is_not_evicted ... ok
test upload_visibility_changes_only_at_frame_boundary ... ok
test replay_produces_identical_residency_sequence ... ok
test transport_cannot_choose_priority_slot_or_victim ... ok
test page_pool_has_zero_frame_allocations ... ok
test stale_completion_cannot_mutate_reused_slot ... ok
test camera_teleport_retains_required_parent_without_holes ... ok
test in_flight_page_is_never_evicted ... ok
test encoded_page_decodes_only_selected_blocks ... ok
test decoded_hash_mismatch_never_reaches_native_build ... ok
test live_geometry_transport_never_decompresses_on_cpu ... ok
```

**NVIDIA paging gate, after permission**

Run the fixed 600-frame traversal with a budget low enough to force eviction. Required verdict:

```text
MEGAGEOMETRY_PAGER owner=gpu frames=600 representation=<dense_blocks|topology_programs> decode_owner=gpu required_page_misses=0 synchronous_readbacks=0 peak_bytes<=budget_bytes deterministic_replay=true cpu_page_decisions=0 cpu_eviction_decisions=0 cpu_decompression=0 frame_allocations=0 render_thread_io=0
```

Any optional-detail miss must be accompanied by a resident parent group and must not create missing geometry.

**Done When**

- [ ] The 600-frame gate passes.
- [ ] No frame blocks on page-selection readback or transport I/O.
- [ ] GPU owns page priority, slot allocation, and eviction; the CPU transport worker executes opaque records only.
- [ ] The configured budget is observable and live through `render.ron`.

---

## Task 10: Use CLAS Templates for Changing and Spatially Reused Topology

**Depends on:** Tasks 5 and 8

**Files**

- Modify `$SPECTRA/rust/spectra-optix/src/optix_host.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/optix_host_unavailable.rs`
- Modify `$SPECTRA/rust/spectra-optix/src/traversal.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/scene.rs`
- Modify `$SPECTRA/rust/spectra-renderer/src/renderer/geometry_gpu.rs`
- Modify `$SPECTRA/slang/mega_geometry_prepare.slang`
- Modify `$OCHROMA/crates/vox_render/examples/clas_scale_bench.rs`
- Add `$SPECTRA/rust/spectra-optix/tests/clas_templates.rs`

**Wiring requirement:** Admission stores topology-template handles in resident GPU metadata. `mega_geometry_prepare.slang` emits common template-instantiation records; the OptiX adapter's native lowering kernel writes `OptixClusterAccelBuildInputTemplatesArgs` and device `argsCount`, then `GeometryAccelBackend::encode_geometry_builds` submits the fixed sequence without a CPU template lookup.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Query and record template limits/capabilities from the live OptiX device.
- [ ] Cache templates by cooked topology hash and compatible build settings at admission; upload handles/compatibility ids into resident GPU metadata.
- [ ] Reuse one template across every decoded geometry-program block with the same proven topology signature, including spatial reuse within one frame; positions, transforms, materials, and provenance remain per use.
- [ ] Generate changing vertex data/group cuts and template-instantiation args on GPU without changing semantic triangle mapping.
- [ ] Keep template lifetime fenced and versioned with the shader/build ABI.
- [ ] Add a real changing-geometry fixture from an existing engine primitive; do not create a game-specific renderer shortcut.
- [ ] Compare ordinary triangle-GAS rebuild, non-template CLAS build, and template CLAS update.
- [ ] Compare topology-program spatial reuse with independent CLAS templates on the repetition-heavy fixture, including template storage, decoded vertices, build time, trace time, and peak VRAM.
- [ ] Let admission-time `Auto` choose templates only when the measured update win survives trace and VRAM gates; the frame loop consumes the resident policy bit.

**Local verification**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-optix --test clas_templates
```

Required tests:

```text
test template_key_covers_topology_and_build_abi ... ok
test vertex_update_preserves_source_triangle_mapping ... ok
test incompatible_topology_rejects_template_reuse ... ok
test template_lifetime_outlives_in_flight_build ... ok
test template_args_and_count_are_device_generated ... ok
test frame_loop_performs_zero_cpu_template_lookups ... ok
test spatial_program_uses_share_one_proven_topology_template ... ok
test topology_hash_collision_or_attribute_mismatch_rejects_reuse ... ok
```

**NVIDIA gate, after permission**

Required verdict:

```text
MEGAGEOMETRY_VERDICT fixture=changing_geometry correctness=pass winner=template_clas update_ms_p95_lt_triangle_rebuild=true geometry_vram_not_worse=true template_args=device args_count=device cpu_template_lookups=0 blocking_readbacks=0
MEGAGEOMETRY_VERDICT fixture=repeated_structure topology_program_reuse=true independent_templates_avoided=<n> encoded_ratio_vs_raw<=0.25 correctness=pass
```

If template CLAS does not win, keep it disabled in `Auto` and record the measured result. A non-win is valid engineering evidence, not a reason to weaken the gate.

**Done When**

- [ ] Template use is real, capability-checked, and measured.
- [ ] The changing fixture either selects the winning CLAS template path or truthfully remains on triangle GAS.

---

## Task 11: Build the Head-to-Head City Benchmark and Ship Gate

**Depends on:** Tasks 0 through 10 and Gate B

**Files**

- Rename or replace `$OCHROMA/crates/vox_render/examples/clas_scale_bench.rs` as `$OCHROMA/crates/vox_render/examples/mega_geometry_bench.rs`
- Modify `$OCHROMA/crates/vox_render/Cargo.toml`
- Add `$OCHROMA/crates/vox_render/tests/mega_geometry_bench_contract.rs`
- Modify `$URBAN/src/spectra_frame/witnesses.rs`
- Add `$URBAN/docs/reference/city-megageometry-results.md`
- Modify `$URBAN/assets/config/render.ron`

**Wiring requirement:** `$OCHROMA/crates/vox_render/examples/mega_geometry_bench.rs` constructs all three modes through `ResidentSceneRenderer`, reads actual typed renderer stats, and calls the same Urban Horizon witness/present path for the mixed-city fixture.

**Implementation steps**

- [ ] Write the named real-behavior tests first.
- [ ] Run the task test command and confirm it fails for the missing or incorrect behavior, not for an unrelated environment error.
- [ ] Implement the exact common evidence schema at the top of this plan.
- [ ] Print compute backend/variant, native RT backend/structure, capability fingerprint, packet/reflection ABI, and calibration key. Requested variants or adapters that silently downgrade exit nonzero.
- [ ] Fail closed if render-thread geometry visits/decisions, blocking readbacks, per-frame geometry allocations, or hidden CPU fallbacks are nonzero.
- [ ] Add `triangle`, `reference`, and `city_hybrid` modes.
- [ ] Make `reference` use a stable spatial cluster schedule matching the selected NVIDIA/meshoptimizer reference algorithm while retaining the same renderer, shaders, and inputs.
- [ ] Add seven hashed fixtures:
   - static repeated Meridian;
   - multi-material correctness building;
   - dense OMM foliage;
   - changing/page-changing geometry;
   - streamed continuous-cluster-LOD camera path;
   - localized million-instance updates;
   - mixed live city witness.
- [ ] Add three breakthrough axes over those scenes: repetition-heavy topology factoring, unique/dense-block fallback, and repeated topology-changing LOD with ReSTIR/temporal remapping. These may share scene assets but have independent hashes and verdicts.
- [ ] Add a separate ray-native suite across at least two real Forge-authored structural families. Dual-lower the same source into mesh oracle and visibility IR, then compare `triangle`, `clas`, `program_to_triangles`, and `ray_native` with identical rays, shaders, camera paths, and full-frame boundaries.
- [ ] Add the certified-fabric suite over identical captured primary, shadow, and bounce-ray streams. Compare native scalar custom intersection, certified subgroup/cooperative variants, optional invocation reordering, triangle GAS, and CLAS; report certification, failed subdivision, packet, residual, merge, fixed-empty-launch, and full-frame costs separately.
- [ ] Add fixed camera paths, warmup/timed counts, deterministic seed, and render config hash.
- [ ] Store raw JSON per mode and a joined comparison JSON.
- [ ] Measure render-thread geometry CPU time separately from the bounded IAS submission worker and page transport worker. Prove no CPU metric scales with total instances/clusters/pages.
- [ ] Measure native-record lowering, resident/AS/page/native-args/scratch VRAM, and full peak geometry VRAM separately. A lower steady allocation cannot hide a build-scratch spike.
- [ ] Prove all shaders/pipelines were loaded and prewarmed before timed frames; any frame-time compilation, pipeline creation, or cache repair suppresses the verdict.
- [ ] Compute a fixture winner only under the design's correctness, p95 geometry cost, and peak geometry VRAM rules.
- [ ] Print a scoped overall claim. Never print `winner=city_hybrid` if any required fixture loses or falls back unexpectedly.
- [ ] Add the AMD conventional-AS run. NVIDIA-only features may be unavailable there, but the same resident scene and cooked LOD must keep the 30 fps Performance-tier gate.
- [ ] Include the paging teleport/thrash fixture and cooperative `on/off/auto` sweep. Cooperative compute is a win only on full-stage/full-frame cost, never instruction utilization alone.
- [ ] At identical frame/VRAM/error settings, measure encoded storage, peak decoded working set, source-equivalent visible detail, temporal invalidations after LOD switches, and full-frame cost for raw clusters, dense blocks, and topology programs.
- [ ] Emit `city_geometry_program_breakthrough` only when topology programs meet both storage ratios, at least double valid visible detail at fixed budget, halve LOD temporal invalidations, have zero losing fixtures, and win at least three independent axes. Never manufacture three wins from correlated counters.
- [ ] Emit `ray_native_city_geometry` only with at least 60% ray-native source-surface coverage, at most 25% materialized geometry, zero fine-AS builds for in-envelope parameter updates, at least 1.5x complete geometry-stage speedup, at least 4x oracle-valid visible detail at fixed budget, zero topology-driven temporal invalidations on ray-native regions, two real asset families, zero correctness mismatches, and zero losses.
- [ ] Emit `certified_visibility_fabric` only when at least 60% of rays use structured visibility, at least 35% are resolved without per-ray traversal, remaining packets sustain at least 75% active lanes, complete intersection and geometry-stage speedups each reach 2x, routing/queue overhead is at most 10%, every CPU scheduling/overflow/correctness counter is zero, and two real families have zero losses.
- [ ] Record witnessed commands, exact revisions, hardware, raw artifact paths, metrics, screenshots, and visual inspection notes in the results document.

**Local contract verification**

```bash
cd "$OCHROMA"
cargo test -p vox_render --test mega_geometry_bench_contract
cargo check -p vox_render --example mega_geometry_bench
```

Required tests:

```text
test requested_backend_fallback_exits_nonzero ... ok
test correctness_failure_suppresses_performance_verdict ... ok
test result_json_contains_environment_and_fixture_hashes ... ok
test overall_win_requires_no_losses_and_all_target_wins ... ok
test parity_and_correctness_only_fixtures_cannot_be_mislabeled_wins ... ok
test forbidden_cpu_counter_suppresses_all_winners ... ok
test delayed_stats_are_outside_present_timing ... ok
test capability_or_reflection_mismatch_invalidates_cache ... ok
test frame_time_pipeline_creation_suppresses_winner ... ok
test peak_vram_includes_build_scratch_and_native_args ... ok
test adapter_or_compute_downgrade_is_never_silent ... ok
test breakthrough_claim_requires_independent_axis_wins ... ok
test visible_detail_counts_only_oracle_valid_geometry ... ok
test storage_win_cannot_hide_decode_build_or_temporal_loss ... ok
test ray_native_claim_requires_two_real_asset_families ... ok
test removed_triangles_and_as_bytes_are_not_double_counted_as_independent_wins ... ok
test procedural_intersection_loss_suppresses_visibility_claim_only ... ok
test fabric_claim_requires_certified_work_elimination_not_only_ray_sorting ... ok
test fabric_claim_includes_failed_certificates_and_fixed_empty_launches ... ok
test fabric_loss_suppresses_only_fabric_claim ... ok
test cpu_queue_scheduling_suppresses_fabric_and_overall_claims ... ok
```

**NVIDIA final run, after permission**

```powershell
cargo run --release -p vox_render --example mega_geometry_bench --features spectra-native,spectra-native-optix -- --suite city --modes triangle,reference,city_hybrid --warmup 60 --frames 240 --json-out artifacts/megageometry/final-nvidia.json
```

The claim gate requires:

```text
MEGAGEOMETRY_SUITE correctness=pass unexpected_fallbacks=0 required_page_misses=0
MEGAGEOMETRY_PORTABLE packet_abi=match control_hash=match cpu_fallbacks=0 adapter_overhead_pct<=2.0
MEGAGEOMETRY_CPU render_thread_scene_visits=0 visibility_node_visits=0 construction_evaluations=0 lod_decisions=0 page_decisions=0 eviction_decisions=0 blocking_readbacks=0 frame_allocations=0 host_deadline_misses=0 render_thread_submit_ms_p95<=0.25 host_submit_ms_p95<=0.50
MEGAGEOMETRY_FIXTURE name=static_meridian outcome=parity regression_pct<=2.0
MEGAGEOMETRY_FIXTURE name=multi_material outcome=correctness_only mismatches=0
MEGAGEOMETRY_FIXTURE name=omm_foliage outcome=parity unexpected_fallbacks=0
MEGAGEOMETRY_FIXTURE name=changing_geometry outcome=win winner=city_hybrid
MEGAGEOMETRY_FIXTURE name=streamed_lod outcome=win winner=city_hybrid
MEGAGEOMETRY_FIXTURE name=localized_updates outcome=win winner=city_hybrid
MEGAGEOMETRY_FIXTURE name=mixed_city outcome=win winner=city_hybrid
MEGAGEOMETRY_OVERALL winner=city_hybrid losses=0 target_wins=4 claim=better_city_geometry_system
MEGAGEOMETRY_BREAKTHROUGH geometry_program=pass repeated_storage_ratio<=0.25 mixed_storage_ratio<=0.50 fixed_budget_detail_ratio>=2.0 lod_temporal_invalidations_reduction>=0.50 full_frame_regression_pct<=2.0 winning_axes>=3 losses=0 claim=city_geometry_program_breakthrough
MEGAGEOMETRY_VISIBILITY_BREAKTHROUGH ray_native_coverage>=0.60 materialized_geometry_ratio<=0.25 parameter_update_fine_as_builds=0 geometry_stage_speedup>=1.50 fixed_budget_detail_ratio>=4.0 ray_native_temporal_invalidations=0 correctness_mismatches=0 real_asset_families>=2 losses=0 claim=ray_native_city_geometry
MEGAGEOMETRY_FABRIC_BREAKTHROUGH structured_ray_coverage>=0.60 certified_without_per_ray_traversal>=0.35 active_packet_lanes>=0.75 intersection_speedup_vs_scalar>=2.0 geometry_stage_speedup>=2.0 routing_queue_overhead_pct<=10.0 correctness_mismatches=0 cpu_scheduling=0 real_asset_families>=2 losses=0 claim=certified_visibility_fabric
```

**AMD final run, after permission**

Run the same mixed-city witness through the product Performance configuration and real present path. Required line:

```text
MEGAGEOMETRY_AMD gpu=AMD_780M api=vulkan rt_backend=vulkan_khr present_path=realtime_1spp packet_abi=match control_hash=match performance_fps_p50>=30.0 runtime_qem_calls=0 missing_geometry=0 cpu_construction_evaluations=0 cpu_lod_decisions=0 cpu_page_decisions=0 blocking_readbacks=0
```

Visually inspect both live present captures. Record the exact screenshots and any differences. Do not claim a render/content fix from the benchmark buffers alone.

**Done When**

- [ ] All local benchmark contract tests pass.
- [ ] Both authorized hardware runs have complete raw artifacts.
- [ ] AMD retains the 30 fps product gate.
- [ ] GPU owns steady-state geometry decisions on both NVIDIA and AMD; only the documented opaque host services remain.
- [ ] CUDA and Vulkan consume the same packet ABI and produce the same persistent control hash; actual cooperative selection is measured and reported.
- [ ] The geometry-program, ray-native, and certified-fabric breakthrough lines are each either fully proven or absent; core MegaGeometry may still ship with a narrower truthful claim.
- [ ] The results document prints either the exact supported win above or a narrower truthful verdict naming the losing fixtures and selected fallbacks.

---

## Backend Expansion and Opportunity Gates

The common contract is part of this plan; promoting additional native features remains evidence-gated:

| Candidate | Opportunity | Promotion gate |
|---|---|---|
| Vulkan KHR indirect BLAS/TLAS | More complete GPU ownership on AMD/Intel | Same records/control hash, live present parity, lower complete geometry-stage p95 |
| Vulkan NV CLAS/PTLAS | Compete directly with OptiX and remove the partition host island | Explicit change to the OptiX-truth project law plus zero-loss NVIDIA head-to-head witness |
| Metal 4 address-driven AS | Device-addressed builds on Apple silicon | Named Apple9+ capability proof, no CPU translation/readback, live present and 30-fps-class product witness |
| DXR adapter | Windows vendor breadth | Same ABI/conformance suite and independent hardware witness; no lowest-common-denominator rewrite |
| Cooperative geometry decode | Lower streaming bandwidth/latency | Better transfer-plus-decode p95 and peak VRAM with bounded decoded error and zero extra required misses |
| Native DGF/DGFS consumption | Avoid decode expansion on capable AMD/API paths | Same decoded/provenance oracle and lower complete geometry-stage p95/VRAM |
| Cage plus bounded residual blocks | Extend factoring beyond exact repeated topology without relying on deprecated displacement-micromap APIs | Better mixed ratio than dense blocks, bounded error, independent pages, zero seam/radiance mismatch |
| GPU storage decompression | DirectStorage GDeflate/Zstd or analogous native path can remove CPU decompression | Opaque transport contract, lower I/O-plus-decode p95, zero CPU decisions/decompression |
| Predictive GPU prefetch | Hide camera-motion I/O | Advisory only, deterministic final request order, parent-safe teleport/thrash witness |
| Native shader/invocation reordering adapter | Let NVIDIA/Vulkan reorder the residual scalar/native queues without owning the portable scheduler | Full-frame win after reorder overhead, identical hit/radiance oracle, and no CPU or ABI fork |
| Ray-class LOD | Use different detail needs for primary/specular versus diffuse/shadow rays | Equal radiance oracle and lower total cost including extra AS/storage |
| Certified temporal hit reuse | Stable parametric surface keys plus conservative motion/envelope certificates may let selected ray hits bypass traversal | Exact disocclusion/motion validation, zero stale hits, and full-frame win after certificate cost; never a correctness assumption |
| Device execution graphs | Fuse classify/decode/lower and reduce barriers/global traffic | Stable production API; provisional Vulkan shader enqueue remains research with indirect baseline |
| Tensorized packet traversal | Research path independent of native RT | Must beat native hardware traversal including build, sort, memory, and full-frame costs before entering the shipping plan |

`VK_NV_partitioned_acceleration_structure` remains the mandatory escalation if the bounded OptiX IAS host island fails its gate. Do not call hierarchical OptiX IAS “PTLAS,” and do not switch the shipping NVIDIA backend without the explicit architectural decision and witnessed win required above.

## Dependency Order

```text
real-asset dual-lowering + native custom-intersection preflight
  -> GPU/host boundary + CPU counters
    -> portable capabilities + packet ABI + subgroup/cooperative variants + native adapter boundary
      -> truthful stats
        -> true device-indirect CLAS + provenance
          -> Forge ray-native visibility + geometry residuals + surface maps + schedule portfolio
            -> certified ray-domain + packet + native-residual fabric gate
              -> typed scene/payload wiring
                -> admission-time hybrid visibility execution + AS + schedule policy
                  -> GPU dirty compaction + bounded hierarchical IAS submission
                    -> GPU continuous cluster LOD + ray feedback + temporal remap
                      -> GPU fixed-pool encoded paging + decode + opaque transport
                        -> device-indirect CLAS template spatial/temporal reuse
                          -> final head-to-head gate
```

Task 4's pure dual-lowering/cook work can proceed alongside Task 3 after Task 1 fixes the common record/capability contract, but native wiring cannot land until Task 3's provenance contract is stable.

## Stop Conditions

Stop and revise the design rather than pushing forward if any of these occurs:

- The common record ABI, Rust/Slang reflection layout, or persistent integer control hash differs between CUDA and Vulkan.
- The real-asset census cannot leave overhead-adjusted headroom below the final repeated/mixed storage targets. Keep the dense path and pivot before freezing `TopologyProgramV1`.
- Ray-native lowering cannot cover at least 60% of source-equivalent surfaces on two real asset families, cannot bound intersection work/AABBs, or loses its complete-stage target. Keep those regions on cooked residual geometry and do not freeze/promote the visibility schema.
- Any ray-native hit, material, UV, normal, motion, or stable-surface key differs from the mesh oracle beyond the exact declared tolerance.
- A whole-domain certificate cannot prove conservative containment and strict nearest ordering, accepts any approximation, or differs from explicit-ray oracles. Ambiguous domains must split or materialize rays.
- Certified-domain, ray-bin, packet-fill, residual-route, or queue-overflow decisions require CPU inspection/readback; native host calls may submit only the same pre-admitted bounded pass graph.
- The fabric cannot count failed certificates, subdivision, queue traffic, exact refinement, fixed empty native launches, residual RT, merge, and peak working memory inside its complete-stage gate.
- The fabric misses its work-elimination, lane-fill, 2x complete-stage, overhead, correctness, or real-family thresholds. Retain scalar/native paths and remove the fabric claim rather than weakening it.
- A visibility path interprets BlueprintGraph, executes arbitrary bytecode, performs runtime/per-asset shader compilation, expands repetition into one leaf per component, or materializes triangles before the measured intersection path.
- An in-envelope parameter update rebuilds fine geometry, or an out-of-envelope update can trace against stale conservative bounds.
- A backend requires common code to expose native structs/pointers, translate GPU records on CPU, read a device count, or compile a shader/pipeline in the live frame.
- Cooperative compute cannot prove actual native instruction lowering, exceeds its numerical tolerance, changes a persistent decision, or loses its complete-stage/full-frame gate.
- Topology reuse depends on authoring labels rather than decoded topology/attribute/provenance identity, or a hash collision can silently share incompatible geometry.
- Geometry programs miss the 25% repeated or 50% mixed storage gates, require full-asset decode, exceed 2% full-frame regression, or lack a competitive dense unique-geometry fallback. In that case do not freeze/promote the program schema.
- Parent/child surface correspondence is incomplete/ambiguous, biases the radiance oracle, or cannot at least halve LOD-switch temporal invalidations on its fixture.
- Zero/stale ray feedback changes required-parent residency, correctness, or the deterministic required request sequence.
- Capability, shader ABI, compiler, driver, or cook-schema changes can reuse a stale pipeline/calibration cache entry.
- Capability enumeration causes unbounded shader-variant growth or a frame-time compilation path.
- The current OptiX headers or driver do not support the required true CLAS form.
- CLAS hit programs cannot recover exact material/UV/source identity.
- Multi-level IAS is unsupported or invalid in the shipping OptiX pipeline.
- The OptiX host-submission worker makes the render thread wait, exceeds 0.50 ms p95 on localized dirtiness, or scales with total scene size rather than compact packet size.
- Any frame-time CPU path walks geometry/visibility nodes, evaluates construction, selects LOD/tolerance, prioritizes/evicts pages, builds native AS args, or allocates geometry pages.
- The Forge hierarchy requires game-specific camera or asset behavior.
- The one-pixel LOD gate requires runtime mesh repair.
- Paging needs synchronous GPU readback to avoid missing geometry.
- GPU work queues cannot detect overflow and retain a correct resident fallback without invoking a CPU builder.
- A stale page completion can mutate a reused slot, an in-flight page can be evicted, or camera teleport/budget thrash produces missing required geometry.
- Peak geometry VRAM including native argument buffers and build scratch exceeds the compared reference even if steady resident bytes do not.
- Device loss/recovery can reuse stale native handles, queue tokens, epochs, or cache admissions instead of performing a structural re-admission.
- AMD's 30 fps product floor is violated.
- The reference comparison cannot use identical assets, camera, shaders, and timing boundaries.

The correct response to a stop condition is a measured fallback and an updated scope, not a weaker witness.

## Final Deliverables

- Engine-owned `vox_data::ReadyMegaGeometry`, embedded by Urban Horizon's typed version-3 ReadyAssetPayload and reusable by other Ochroma games, with deterministic Forge visibility programs, sparse cells, factorized coefficient blocks, residual ranges, and cluster hierarchy.
- Dual lowering from the same validated BlueprintGraph/construction source into the unchanged mesh oracle and a finite game-agnostic visibility IR.
- AOT Slang primitive-family intersections over native OptiX/Vulkan/Metal procedural AABBs, with stable construction-path surface keys and parameter-only zero-fine-build updates.
- A certified visibility fabric that resolves coherent ray domains without per-ray traversal, packetizes unresolved structured rays, and sends irregular residue to fixed-function RT with deterministic exact hit merge.
- Portable Forge geometry programs with proven topology dictionaries, quantized parameter/residual blocks, DGF/DGFS-style dense fallback, complete surface correspondence, and a bounded Pareto schedule portfolio.
- Fixed-arena GPU decode, topology-template spatial reuse, quantized ray feedback, and LOD-aware ReSTIR/temporal remapping.
- One versioned backend-neutral geometry record ABI, immutable capability fingerprint, opaque queue/epoch model, and cross-backend conformance harness.
- CUDA, Vulkan, and experimental Metal adapters over the existing `GpuBackend`, with support status limited by their independent hardware evidence.
- Mandatory portable subgroup kernels plus optional measured Slang cooperative-matrix specializations; no live CPU fallback or vendor algorithm-library dependency.
- True OptiX CLAS build and exact primitive provenance.
- Typed truthful acceleration stats and fail-closed benchmarks.
- Measured per-prototype triangle-GAS/CLAS selection.
- Measured per-subgraph ray-native/triangle-GAS/CLAS selection using complete AABB/intersection/shading/residual cost.
- GPU dirty-partition compaction plus bounded hierarchical OptiX IAS submission.
- GPU-selected cooked continuous cluster LOD with crack/error evidence.
- GPU-managed fixed geometry page pool plus opaque asynchronous transport.
- Measured device-indirect CLAS template path for changing geometry.
- CPU-ownership evidence with zero forbidden decisions/visits/readbacks/allocations.
- Raw NVIDIA and AMD benchmark artifacts.
- CUDA/Vulkan packet, control-hash, capability, reflection, native-lowering, adapter-overhead, and compute-variant artifacts; Metal hardware evidence when promoted.
- Live present captures and inspection notes.
- A results document whose claim is mechanically limited by the evidence.
- A separate `city_geometry_program_breakthrough` verdict requiring the storage, fixed-budget detail, temporal, full-frame, and independent-axis gates.
- A stronger separate `ray_native_city_geometry` verdict requiring coverage, materialization, update-build, speedup, fourfold-detail, stable-temporal, real-family, and zero-loss gates.
- A strongest separate `certified_visibility_fabric` verdict requiring exact traversal elimination, packet utilization, complete-stage 2x wins, bounded overhead, GPU-only scheduling, real-family coverage, and zero losses.

## Self-Review Checklist

- [x] Every task implements and wires one complete capability in the same task.
- [x] Every acceptance section names non-trivial values or exact output strings.
- [x] Every wiring requirement names a live function/path or the exact replacement entry point.
- [x] IMPORTANT NOTES mirrors the public API signatures in the design.
- [x] The File Map contains every file named by a task.
- [x] No implementation step permits `todo!()`, `unimplemented!()`, empty bodies, or synthetic success counters.
- [x] Done When names the exact final commands and human-visible verdicts.
- [x] API/type names are consistent between the design and plan.
- [x] Existing dirty worktrees are preserved and commit/remote/game operations remain permission-gated.
- [x] Forge owns asset topology/LOD/page authoring; renderer and gameplay shortcuts are explicitly forbidden.
- [x] GPU owns every steady-state geometry decision; CPU work is limited to named, bounded, opaque host API services.
- [x] GPU owns ray-domain subdivision, queue/bin construction, packet selection, residual routing, and counts; native host APIs only submit the fixed admitted pass graph without reading device state.
- [x] Device-indirect OptiX CLAS args/count are required, not an optional optimization.
- [x] The remaining OptiX IAS host boundary is disclosed, measured, and has a PTLAS escalation condition.
- [x] The existing `GpuBackend` is extended rather than duplicated, and native RT is a separate adapter plane from cooperative compute.
- [x] Common records contain no vendor-native structs or raw addresses; lowering and pointer resolution occur only inside the native adapter.
- [x] Every required kernel has a GPU subgroup/SIMD path; cooperative matrices are optional, capability-checked, instruction-proven, numerically gated, and full-stage measured.
- [x] Certified ray domains have a strict whole-domain correctness rule; ambiguity becomes deterministic subdivision or exact explicit rays, never an approximation.
- [x] Persistent topology/LOD/page/partition/build decisions are bit-exact integer/fixed-point outputs across CUDA and Vulkan.
- [x] AOT/reflection/cache compatibility, queue/barrier ownership, build scratch, multi-frame lifetime, stale completions, camera teleports, and budget thrash have explicit tasks or stop conditions.
- [x] Metal and future adapters cannot be advertised from compile-only evidence or green skips.
- [x] The opportunity register separates immediate architecture from measured follow-ons and keeps tensorized BVH traversal out of the critical path.
- [x] The plan exploits Forge construction structure rather than merely producing better independent clusters, while keeping engine types game-agnostic.
- [x] Unique geometry has a dense-block baseline; procedural reuse is never mandatory or label-trusted.
- [x] Cook optimization covers decode/build/trace/update/page/memory Pareto costs rather than triangle count alone.
- [x] Cross-LOD surface identity, ReSTIR/temporal reuse, ray-feedback safety, and fixed-budget visible detail have independent correctness and performance gates.
- [x] Each breakthrough claim requires its independent product gates and can fail without corrupting narrower measured results.
- [x] A real-asset entropy/cost preflight can kill or redirect the representation hypothesis before schema freeze and broad backend integration.
- [x] Certified ray-domain work elimination is the strongest measured hypothesis; packet intersection, scalar custom intersection, native RT, topology programs, cage/residual blocks, and dense triangles form explicit progressively narrower fallback layers.
- [x] The primary breakthrough path bypasses triangles entirely for supported construction nodes; topology programs are now the residual strategy, not the north star.
- [x] Dual lowering keeps the existing mesh as a permanent oracle, unsupported nodes fail locally, and no BlueprintGraph interpreter or asset-specific runtime shader is introduced.
- [x] Conservative bounds, finite intersection work, stable parametric surface identity, parameter envelopes, and complete programmable-intersection cost are explicit gates.
