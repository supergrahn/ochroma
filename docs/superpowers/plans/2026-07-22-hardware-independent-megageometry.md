# Hardware-Independent MegaGeometry Plan

> **Status:** Revised around the MegaGeometry Execution Fabric: a portable,
> software-defined visibility and transport machine that prepares, eliminates,
> and coheres work before invoking each device's native RT hardware. This is the
> replacement direction for the vendor-shaped parts of
> [City MegaGeometry](2026-07-20-city-megageometry.md). Its already-witnessed
> GPU-realized native-triangle path remains valid evidence; CLAS, OMM, and SER
> are explicitly reclassified as optional execution accelerators.
>
> **Design:** Amend the MegaGeometry design before implementation so the shared
> representation, portability contract, and acceptance gates below outrank any
> backend-specific implementation note.

> **Implementation checkpoint — 2026-07-22:** Task 4 has started in Spectra.
> The shared deterministic domain oracle, fixed-capacity portable GPU ABI,
> GPU-only residual queue reset/classification, bounded residual-ray packetizer,
> and shader registrations now exist. A certificate currently proves only a
> conservative miss; every other domain is expanded on-GPU into the existing
> native active-ray interface, with a fail-closed overflow bit. The frame-loop
> admission, richer certificates, writer/nearest-hit merge, and live witness
> remain unfinished, so no performance or image-quality claim is implied by
> this checkpoint.
>
> The product configuration seam is also now explicit: Urban Horizon's
> `render.ron` owns portable hierarchical-acceleration admission and its
> partition size, validates both values, and requests a resident-scene rebuild
> when either changes. The renderer's legacy environment reads are therefore
> fed by the authored game policy rather than ambient machine state. This does
> not yet make the Fabric's GPU packet producer, surface correspondence writer,
> template reuse, or city witness complete.
>
> **Live Fabric checkpoint — 2026-07-22:** the non-OptiX primary path now
> allocates resident Fabric buffers at scene admission, builds conservative
> ray-domain envelopes on GPU from the camera queue, classifies them, writes
> certified sky misses into the normal wavefront hit contract, and submits the
> unresolved queue to the existing exact native traversal. A GPU-only
> fail-closed pass restores the complete identity queue on packet overflow.
> OptiX remains on its full native launch path until its launch ABI supports the
> same active-ray indirection. The local machine could compile the Rust wiring
> but cannot compile/execute these Slang kernels because NVRTC is unavailable;
> the authorized Windows witness is still required to validate live execution.
>
> **Surface-history checkpoint — 2026-07-22:** `ReadyMegaGeometry` schema 6
> now carries a Forge-cooked, content-hashed correspondence record for every
> exact program. Scene assembly offsets parent/child program groups during
> instancing; the engine-neutral scene layer validates and uploads both records
> and slots. Camera generation clears each primary slot to invalid and an exact
> procedural hit writes only its cooked slot. These are identity parents for
> unchanged full-detail programs. Camera generation clears each primary slot,
> exact MegaGeometry traversal writes its cooked slot, and the renderer
> dispatches the GPU remap-validity mask before ReSTIR temporal reuse; no CPU
> readback is involved. Non-identity LOD-cut records and stable reservoir
> target re-keying are still required before claiming completed cross-LOD
> temporal transport. The ReSTIR path now also ping-pongs a GPU-only stable
> `(parent_group,parent_primitive)` surface key derived from that cooked table:
> equal valid keys re-key history across a representation change; corrupt,
> ambiguous, or ordinary-to-MegaGeometry transitions invalidate only their
> local pixel. Authoring non-identity representation records and live present
> validation remain required.
>
> **Full-detail native-adapter checkpoint — 2026-07-22:** the legacy OptiX
> runtime CPU-QEM decimation chain and cache have been removed from the renderer.
> Native prototype lowering now receives only exact cooked source geometry. This
> preserves authored content on every path; hierarchy/page scheduling remains
> the only allowed MegaGeometry detail mechanism.
>
> **Deterministic Fabric checkpoint — 2026-07-22:** the residual classifier
> records source-domain positions, not atomic append order. A single GPU lane
> scans those classifications into a stable `(domain, local_ray)` native queue,
> eliminating schedule-dependent queue order across CUDA, Vulkan, and Metal.
> Packet admission now requires native-ray capacity to cover every source ray;
> the device-side overflow fallback restores the complete identity queue and
> cannot truncate it. These portable-contract tests pass locally. Live Slang
> compilation and presentation remain Windows/NVRTC hardware gates.

## Goal

Ship one MegaGeometry system for every supported GPU that changes the cost
structure of city path tracing. Forge compiles each complete, deterministic,
full-detail building into a portable visibility and transport program; Urban
places it unchanged; Spectra's GPU-resident Execution Fabric certifies coherent
visibility, packetizes unresolved work, preserves surface-space transport, and
invokes native RT only for the work that hardware traversal handles best.
Backends may accelerate the same contract, but they may not require a separate
asset, change authored detail, or alter the visible result.

NVIDIA OptiX CLAS, opacity micromaps, and shader execution reordering are not
the product architecture. They are optional RTX adapters that can improve a
measured workload after the portable path is correct. Vulkan RT, Metal, and
future backends consume the same cooked pages and emit the same stable hit
record.

Every capable device must nevertheless use its hardware ray-tracing path. The
portable layer supplies deterministic, bounded cluster pages and GPU-produced
build records; the selected adapter lowers those records to the device's native
acceleration structures. "Hardware independent" means common assets and
contracts, not deliberately bypassing RT hardware.

**Done when:** the product gate and leadership gate both pass. The product gate
requires an authorized 1440p, 1 spp reconstructed real-time present witness,
the same source-asset hash and hit/radiance parity for every available backend,
AMD 780M at least 30 FPS without lower-detail content, and zero steady-state CPU
geometry, LOD, page-priority, or ray-scheduling work. The leadership gate
requires at least 60% structured-ray coverage, at least 35% of rays resolved by
conservative certification without individual native-RT traversal, at least 2x
geometry-stage speedup or 2x visible authored detail per geometry-VRAM byte,
at least 50% fewer valid-history invalidations, no quality regression, and a
same-GPU win over the strongest native-RT-only baseline.

## Non-negotiable decisions

- **MegaGeometry owns LOD.** It uses the complete cooked detail hierarchy and
  stable surface correspondence. It is not a conventional alternate-asset LOD
  scheme and never creates missing geometry at runtime.
- **Forge stays hardware-neutral.** Cooked content contains no `CLAS`, `OMM`,
  `SER`, CUDA, OptiX, or vendor capability flag. A page represents full authored
  content plus deterministic detail/correspondence data.
- **One optical truth.** Every backend returns the same surface identity, UV,
  differential frame, material domain, interior/optical-layer metadata, and
  motion/correspondence information required by the spectral renderer.
- **No vendor downgrade.** A non-RTX backend can select a different native
  execution representation, but it cannot select lower authored quality,
  substitute placeholders, or silently disable required materials/details.
- **GPU ownership.** CPU may upload cooked pages and submit bounded opaque
  work. It may not inspect per-frame camera visibility, choose detail, page
  priority, evictions, or ray queues.
- **Software prepares hardware work.** A GPU-resident reader/classifier/compute/
  writer pipeline removes certified work and increases coherence before native
  RT. It is deterministic dataflow, not CPU scheduling or runtime asset creation.
- **RT hardware is a first-class coprocessor.** NVIDIA, AMD, Intel, and Apple RT
  hardware remains mandatory on capable product devices for unresolved,
  irregular, dynamic, and locally routed geometry. It is not required to trace
  rays whose exact result the Execution Fabric has already certified.
- **Config is the authority.** Runtime choices live in
  `urban_horizon/assets/config/render.ron`; environment variables are temporary
  diagnostics only, never product controls.

## Architecture

```text
Forge full-detail directive / BlueprintGraph
  -> Forge visibility + transport compiler
       exact mesh oracle
       parametric surface programs + repetition lattices
       pages + cluster hierarchy + residuals + stable surface map
       optical/material domains + deterministic schedule candidates
  -> Urban ReadyAssetPayload / resident scene delta
  -> MegaGeometry Execution Fabric (GPU resident)
       reader: stage hot geometry / bounds / optical data
       classifier: certify ray domains and subdivide ambiguity
       packet compute: factorized surface + spectral/material work
       native RT: unresolved / irregular / dynamic residuals
       writer: merge hits, continue rays, update surface-space transport
  -> compact stable hit identity
  -> GPU material / optical / correspondence resolve
  -> spectral path tracing + reconstruction + present
```

The portable package is the product boundary. The adapter selection happens
only after device capability discovery and may choose a measured native build
format. It cannot mutate the package, serialise a backend-specific result, or
change the shared hit ABI.

## Revolutionary thesis

MegaGeometry does not win by recreating a vendor acceleration structure in
portable compute. It wins by compiling knowledge of a city before traversal,
eliminating work that does not need traversal, increasing coherence for work
that remains, and treating native acceleration structures as an elastic cache
of the unresolved working set.

### Breakthrough A — Compiled city visibility

Forge dual-lowers every supported construction into the exact mesh oracle and
a finite, bounded visibility program. Repeated floors, bays, windows, balconies,
structural grids, façade panels, interiors, and solar-control systems remain
factorized rather than becoming anonymous independent triangles. Each repeated
element retains its own stable identity, material, blind/screen state, interior
class, and optical layer.

### Breakthrough B — Certified ray-domain elimination

The GPU classifier operates on coherent primary tiles, sun-shadow bundles, and
eligible reflection/transmission bundles. Conservative proofs may certify a
whole domain as miss, common nearest surface family, bounded repetition-cell
interval, or complete occlusion. Ambiguous domains subdivide deterministically;
unresolved rays enter packet compute or native RT. Certification is exact work
elimination, never an image approximation.

### Breakthrough C — Cross-vendor packet machine

Unresolved structured rays are binned jointly by visibility cell, surface
family, ray class, material domain, optical stack, wavelength packet, and
intersection complexity. Portable subgroup kernels are mandatory; cooperative
matrix paths are optional and selected only by complete-stage measurement. RTX
may additionally use SER, but the common system must already produce coherent
work on AMD, Intel, and Apple.

### Breakthrough D — Elastic native-AS cache

The portable visibility program is authoritative. Native acceleration
structures are materialized for the currently required unresolved pages:
OptiX triangle GAS/IAS or CLAS, Vulkan KHR BLAS/TLAS, and Metal
primitive/instance AS. Certified domains do not require redundant native-AS
work. Required parent coverage and irregular residuals remain exact and
fail-closed under pressure.

### Breakthrough E — Surface-space transport memory

Temporal radiance, reservoirs, and reconstruction validity attach to stable
`(asset, construction_path, parametric_coordinate, optical_layer)` identity,
not transient primitive ids or screen pixels alone. Valid history survives page
movement, native-AS rebuild, continuous detail changes, and analytic/triangle
representation changes. Invalid or ambiguous correspondence discards history
locally rather than clearing it globally or reusing it incorrectly.

## Native RT strategy — same source, best device execution

| Hardware family | Native execution path | MegaGeometry advantage we retain | Not allowed |
|---|---|---|---|
| NVIDIA RTX | OptiX triangle GAS/IAS for the generic route; CLAS/templates for measured eligible pages; OMM/SER only where proven | Shared page hierarchy, surface identity, material queues, and full Forge asset package | Making CLAS, OMM, or SER required for correctness or content |
| AMD | Vulkan KHR triangle BLAS/TLAS, GPU-built from portable page records; profile fast-build versus fast-trace per page class | Page partitioning limits empty-space/overlap and preserves full detail while GPU chooses residency | Emulating NVIDIA CLAS in compute or shipping an RTX-shaped asset |
| Intel | Vulkan KHR triangle BLAS/TLAS from the same page records; capability/profile-driven build and compaction policy | The same compact page hierarchy and shader-material bins, validated on real Intel hardware | Assuming Intel behavior from AMD or NVIDIA results |
| Apple silicon | Metal primitive/instance acceleration structures, indirect instance builds, and per-primitive data where it improves measured material access | Stable page/surface keys map naturally to instance and per-primitive identifiers | Treating Metal as a CPU or reduced-detail fallback |

The portable equivalent of a CLAS page is **not** a fake CLAS implementation.
It is a deterministic cluster page with conservative bounds, reusable topology,
and stable provenance. Each backend lowers that page into the native RT
structure it actually supports. This lets RTX use its special cluster hardware
where it wins, while AMD, Intel, and Apple use their own hardware traversal
without asset forks or a CPU traversal substitute.

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `$FORGE/crates/mesh/src/mega_geometry.rs` | Portable pages, hierarchy, stable surface mapping, deterministic validation |
| Modify | `$FORGE/crates/mesh/tests/mega_geometry.rs` | Cook determinism, full-coverage, detail/correspondence tests |
| Modify | `$URBAN/src/.../game_asset_cook/*` | Carry typed payloads; never invent content at load time |
| Modify | `$OCHROMA/crates/vox_render/src/resident_renderer.rs` | Upload typed resident payload and scene deltas only |
| Modify | `$SPECTRA/rust/spectra-renderer/src/geometry_gpu_lod.rs` | GPU-only cut, feedback and page-work production path |
| Modify | `$SPECTRA/rust/spectra-renderer/src/geometry_gpu_page.rs` | GPU-owned page request/eviction packets and fail-closed capacity handling |
| Modify | `$SPECTRA/rust/spectra-renderer/src/geometry_accel.rs` | Shared `GeometryBackend` / shared hit ABI |
| Modify | `$SPECTRA/rust/spectra-renderer/src/render_state.rs` | Resident handles and temporal/correspondence state, no CPU mirror authority |
| Add | `$SPECTRA/rust/spectra-renderer/src/visibility_fabric.rs` | Fixed-capacity GPU reader/classifier/packet/native-RT/writer execution graph |
| Add | `$SPECTRA/rust/spectra-renderer/src/surface_transport.rs` | Stable surface-space history, validity, and local invalidation contract |
| Add | `$SPECTRA/slang/visibility_domain_classify.slang` | Conservative domain certificates and deterministic ambiguity subdivision |
| Add | `$SPECTRA/slang/visibility_packet_intersect.slang` | Portable subgroup and optional cooperative structured intersection |
| Add | `$SPECTRA/slang/visibility_hit_reduce.slang` | Exact nearest-hit merge across certified, packet, and native-RT paths |
| Add | `$SPECTRA/slang/surface_transport_remap.slang` | Surface-space temporal/reservoir remap and validity output |
| Modify | `$SPECTRA/slang/megakernel.slang` | Consume the shared hit ABI and portable material/spectral queues |
| Modify | `$SPECTRA/rust/spectra-optix/*` | Optional OptiX adapter only; no common policy dependency |
| Modify | `$URBAN/assets/config/render.ron` | Product MegaGeometry controls and tier budgets |
| Add | `$SPECTRA/rust/spectra-renderer/tests/megageometry_portability.rs` | Package, hit, and frame-contract conformance |
| Add | `$URBAN/src/bin/mega_geometry_witness.rs` | Authorized live-present evidence runner and machine-readable report |

`$FORGE`, `$URBAN`, `$OCHROMA`, and `$SPECTRA` resolve to their sibling
repositories at execution time. Confirm the live roots before editing; all
worktrees may be dirty.

## Shared contracts to freeze first

The exact Rust placement may follow existing module ownership, but the common
contract must be equivalent to:

```rust
pub struct PortableMegaGeometryPage {
    pub page_id: u32,
    pub content_hash: [u8; 32],
    pub hierarchy: GpuBufferHandle,
    pub full_detail: GpuBufferHandle,
    pub residual: GpuBufferHandle,
    pub surface_map: GpuBufferHandle,
}

pub struct StableSpectralHit {
    pub surface_key: SurfaceKey,
    pub parametric_uv: [f32; 2],
    pub geometric_normal: [f32; 3],
    pub shading_frame: ShadingFrame,
    pub material_domain: MaterialDomain,
    pub optical_layer: OpticalLayerKey,
    pub correspondence: SurfaceCorrespondence,
}

pub struct VisibilityExecutionPacket {
    pub ray_domains: GpuBufferHandle,
    pub domain_count: GpuBufferHandle,
    pub packet_rays: GpuBufferHandle,
    pub packet_count: GpuBufferHandle,
    pub native_rt_rays: GpuBufferHandle,
    pub native_rt_count: GpuBufferHandle,
    pub merged_hits: GpuBufferHandle,
    pub admitted_capacity: u32,
}

pub trait GeometryBackend {
    fn admit_package(&mut self, package: &PortableMegaGeometryPackage,
        queue: &GpuQueueToken) -> Result<GeometryResidentHandle, GeometryBackendError>;
    fn encode_gpu_work(&mut self, work: &GeometryGpuWorkBuffers,
        queue: &GpuQueueToken) -> Result<GpuEpoch, GeometryBackendError>;
    fn hit_contract(&self) -> GeometryHitContract;
}
```

No trait method accepts a CUDA pointer, `Optix*` type, host slice of selected
geometry, or a CPU-produced LOD/page decision. Native handles remain inside
the final adapter. `StableSpectralHit` must be packed and versioned identically
for every shader backend.

The traversal payload is deliberately compact: it contains only the page,
primitive/surface key, barycentrics or parametric coordinate, distance, and
minimal flags. The full shading frame, material domain, optical stack, and
temporal correspondence resolve in a subsequent GPU pass from those stable
keys. This avoids inflating RT live state on every device while preserving the
complete spectral/material result.

`VisibilityExecutionPacket` is fixed-capacity and entirely GPU-owned. The CPU
cannot read its counts to choose dispatches or create ray lists. Queue overflow
retains an exact admitted native-RT route, increments a failure counter, and
suppresses the leadership verdict.

## Task 1 — Make the Forge package fully portable and complete

**Why:** Every later hardware choice is safe only if the source package is
complete, deterministic, and independent of the device that happens to cook or
first load it.

- [ ] Define typed pages, cluster hierarchy, residual coverage, material slots,
  stable surface keys, and parent/child correspondence in Forge. Retain all
  authored doors, windows, props, textures, optical metadata, and construction
  detail. A package with uncovered source primitives fails the cook.
- [ ] Extend `ReadyAssetPayload` with the typed portable package and a schema
  version. Remove any untyped JSON or backend label from this boundary.
- [ ] Validate no holes, duplicate ownership, invalid UV/material references,
  broken surface maps, or missing complete-detail pages. Test rotation, mirrors,
  non-uniform transforms, glazing, cutouts, and multi-material façades.
- [ ] Prove two independent cooks have identical package bytes and hashes.

**Verification:**

```bash
cd "$URBAN"
cargo test -p game-asset-cook portable_megageometry
cargo run --release --bin game_asset_cook -- --verify-megageometry-assets assets/product_ready_manifest.ron
```

Required output:

```text
MEGAGEOMETRY_PACKAGE schema=<n> backend_tags=0 uncovered_primitives=0 missing_materials=0 missing_textures=0 stable_surface_coverage=1.0 deterministic_hash=match
```

## Task 2 — Wire GPU-owned detail, residency, and correspondence

**Why:** Existing LOD/page components are useful only when the real present
path uses their GPU-produced work records rather than an environment-gated or
CPU-owned side path.

- [ ] Promote `geometry_gpu_lod` and `geometry_gpu_page` to the only
  steady-state selection path. GPU buffers carry required parent pages,
  detail cuts, page requests, page releases, and conservative fallback state.
- [ ] Make capacity overflow fail closed: retain a coarser *already-authored*
  parent that preserves coverage and flag telemetry. It must not create proxy
  meshes or omit a visible surface.
- [ ] Use stable surface correspondence for motion vectors, temporal reuse, and
  reservoir invalidation. A missing/ambiguous map invalidates history locally.
- [ ] Add product config fields for enablement, page budget, hierarchy budget,
  correspondence validation, and diagnostic sampling; bind them through the
  runtime configuration path and prove a config sweep changes live telemetry.

**Verification:**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test geometry_gpu
cargo test -p spectra-renderer --test megageometry_portability gpu_owned_detail_and_pages
```

Required output:

```text
MEGAGEOMETRY_GPU_WORK lod_owner=gpu page_owner=gpu required_page_misses=0 cpu_lod_decisions=0 cpu_page_decisions=0 ambiguous_history_reuse=0
```

## Task 3 — Establish the portable acceleration adapter

**Why:** The engine needs a single geometry policy that can execute complete
assets on all hardware. A vendor path must not leak into gameplay, Forge, or
the shading contract.

- [ ] Make native hardware RT the exact residual and comparison baseline for
  every RT-capable backend. The adapter builds its own triangle BLAS/TLAS,
  primitive/instance AS, or proven vendor extension from the same portable GPU
  work records. Software traversal is diagnostic-only and can never satisfy a
  product witness; exact visibility certificates may bypass traversal because
  they have already proven the result.
- [ ] Keep the GPU-realized triangle path as a valid generic implementation,
  with residual geometry exact and material provenance intact. Do not claim it
  is a shortcut or a second asset path.
- [ ] Add a `NativeAccelerationPlan` selected on GPU capability and measured
  page class: static/reused pages favor trace-optimized native builds; dynamic
  pages favor measured refit/build choices; pages with excessive bounds overlap
  split at Forge-cooked boundaries. The plan contains page ids and build modes,
  never vendor geometry or CPU camera decisions.
- [ ] Add backend-specific build/trace telemetry: BLAS/TLAS or Metal
  primitive/instance build time, compaction/refit result, page overlap, scratch
  use, traversal time, and hardware capability fingerprint. It is compared
  against the same source package, camera, and shading configuration.
- [ ] Add explicit unavailable-backend behavior: fail the requested quality
  tier with a precise diagnostic; never silently fall back to CPU rendering or
  a low-detail asset set.
- [ ] Add cross-backend ABI/reflection tests and ray-oracle fixtures spanning
  glass, opaque façades, alpha vegetation, industrial detail, and mixed
  materials. Compare nearest hit, UV, normal, material, optical layer, and
  surface correspondence.

**Verification:**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test geometry_backend_conformance
cargo test -p spectra-renderer --test megageometry_portability hit_contract
```

Required output:

```text
MEGAGEOMETRY_PORTABLE source_hash=match hit_abi=match native_rt=1 hit_mismatches=0 material_mismatches=0 optical_layer_mismatches=0 cpu_fallbacks=0
MEGAGEOMETRY_NATIVE_RT api=<optix|vulkan_khr|metal> build_mode=<build|refit|compact> page_overlap_pct=<n> as_build_ms_p95=<n> trace_ms_p95=<n> capability_hash=<hash>
```

## Task 4 — Build the MegaGeometry Execution Fabric

**Why:** Portable native-AS lowering alone changes storage and build policy but
does not eliminate enough traversal or shading work. This task implements the
software-defined GPU dataflow that can create a system-level advantage over a
native-RT-only renderer.

- [ ] Implement a fixed-capacity GPU reader/classifier/packet/native-RT/writer
  graph. Reader stages required program/pages and optical headers; classifier
  writes exact certificates or deterministic subdivisions; packet compute
  resolves structured ambiguity; native RT consumes only unresolved rays;
  writer performs one deterministic nearest-hit merge.
- [ ] Define conservative miss, common-nearest-family, repetition-cell interval,
  and complete-occlusion certificates. Every certificate carries the bounds and
  ordering proof needed for an independent scalar oracle to reject it.
- [ ] Partition ambiguous domains using fixed integer rules. The same inputs and
  capacities produce the same child-domain sequence on every backend.
- [ ] Route irregular geometry, failed certificates, dynamic actors, and queue
  overflow to an exact native-RT residual path. Approximate proxy hits are
  forbidden.
- [ ] Keep all live counts, queues, fill decisions, subdivision and merging on
  the GPU. The host submits only fixed admitted capacities and opaque epochs.
- [ ] Fuse stages where measurement shows dispatch or memory traffic dominates;
  retain independently testable kernels and counters so fusion cannot hide
  incorrect certification or fabricated work elimination.

**Verification:**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer --test visibility_fabric
cargo run --release -p spectra-renderer --example visibility_fabric_bench -- --corpus "$URBAN/assets/config/mega_geometry_corpus.ron" --modes native_rt,certified_packet --warmup 60 --frames 240
```

Required output:

```text
MEGAGEOMETRY_FABRIC structured_ray_coverage>=0.60 certified_without_rt_traversal>=0.35 active_packet_lanes>=0.75 routing_overhead_pct<=10.0 queue_overflows=0 cpu_scheduling=0
MEGAGEOMETRY_FABRIC_CORRECT hit_mismatches=0 material_mismatches=0 uv_normal_mismatches=0 occlusion_mismatches=0 radiance_mismatches=0
MEGAGEOMETRY_FABRIC_VERDICT geometry_stage_speedup>=2.0 real_asset_families>=5 claim=certified_visibility_fabric
```

## Task 5 — Build the spectral packet machine and surface-space transport

**Why:** MegaGeometry surpasses a mere AS feature only when its stable identity
and full material data reduce wasted spectral/path work without changing light
transport or making glass/architecture look generic.

- [ ] Extend the shared hit data with material domain and optical-layer keys for
  office, residential, commercial, industrial, vegetation, and infrastructure
  surfaces. These are authored material decisions, not random runtime content.
- [ ] Add deterministic GPU bins for secondary rays by visibility cell, surface
  family, ray class, material domain, optical stack, wavelength packet, and
  work complexity. Each bin has fixed admitted capacity and an exact overflow
  route; ordering cannot depend on hash-map iteration or thread timing.
- [ ] Implement a portable subgroup packet kernel and retain scalar/native RT as
  the oracle. Add cooperative-matrix variants only when instruction reflection
  proves the intended hardware path and complete-stage time improves.
- [ ] Store temporal radiance/reservoir identity in surface space. Validate and
  remap history across page movement, AS rebuild, continuous detail changes,
  and analytic/triangle representation changes.
- [ ] Feed correspondence into ReSTIR only after the hit/material/normal/UV
  validity test succeeds. Do not enable ReSTIR in product tiers until its shade
  endpoint and temporal correctness path are complete and witnessed.
- [ ] Measure spectral queue divergence, any-hit samples, temporal
  invalidations, noise, and radiance parity before claiming a performance win.

**Verification:**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-renderer spectral_hit_and_temporal_contract
```

Required output:

```text
MEGAGEOMETRY_SPECTRAL packet_bins=deterministic active_packet_lanes>=0.75 packet_speedup_vs_scalar>=2.0 radiance_mismatches=0 invalid_reuse=0 overflow_route=exact
MEGAGEOMETRY_SURFACE_MEMORY valid_history_survival_gain>=0.50 local_invalidations=<n> global_clears=0 correspondence_mismatches=0
```

## Task 6 — Add vendor accelerators as contained adapters

**Why:** RTX features can improve an RTX build, but their absence must leave a
complete, fast, visually equivalent MegaGeometry product on other hardware.

| Optional adapter | Permitted role | Explicit prohibition |
|---|---|---|
| OptiX CLAS/templates | Native build/traversal acceleration for a measured static opaque package | No CLAS fields in Forge/Urban payloads; no global NVIDIA policy |
| OptiX OMM | Alpha traversal acceleration where material coverage is proven | No replacement of authored alpha or vegetation detail; generic path remains exact |
| NVIDIA SER | Reorder eligible shader work after shared hit construction | No changes to ray visibility, surface identity, or queue correctness |
| Vulkan KHR BLAS/TLAS | AMD, Intel, and capable NVIDIA native RT from portable pages | No vendor inference; page build policy is profiled per device |
| Metal primitive/instance AS | Apple-native RT with indirect instances and measured per-primitive data | No CPU traversal or reduced-detail Apple path |

- [ ] Gate every adapter behind capability discovery and a measured policy
  decision that has an equivalent generic outcome.
- [ ] Treat all four native RT paths as first-class hardware gates. An adapter
  is only eligible after its own real device witness proves complete source
  coverage, exact hit contract, and no CPU fallback.
- [ ] Move hierarchy and page controls from environment-only gates into
  `render.ron`; telemetry records the selected adapter and reason.
- [ ] Mark every vendor-only benchmark clearly. It can establish an adapter win
  but cannot establish a universal MegaGeometry win.
- [ ] Keep alpha-cutout semantics identical across adapters. OMM is additive;
  the portable alpha-tested path remains the oracle.

**Verification:**

```bash
cd "$SPECTRA/rust"
cargo test -p spectra-optix --test indirect_clas_contract
cargo test -p spectra-renderer --test geometry_policy
```

Required output:

```text
MEGAGEOMETRY_ADAPTER selected=<optix_*|vulkan_khr|metal_native> reason=<measured> portable_equivalence=pass native_rt=1 asset_fork=0
```

## Task 7 — Make the evidence a real city product and leadership gate

**Why:** Small synthetic million-instance CLAS tests and offscreen images do
not prove an AAA city builder. The final gate must exercise complete Forge
assets, real streaming, 1 spp reconstruction, and the normal present path.

- [ ] Create a deterministic city witness route with dense mixed-use streets,
  glazed office towers, homes, commercial fronts, industry, roads, vehicles,
  vegetation, and terrain. It uses cooked complete assets only.
- [ ] Capture the same camera trace on every available hardware adapter at
  1440p output, 1 spp, product reconstruction, and identical `render.ron`
  tier. Inspect real present frames; no offline clean trace substitutes.
- [ ] Report AS/build time, trace time, shading time, reconstruction time,
  frame percentiles, geometry/AS/page VRAM, page misses, alpha samples,
  material divergence, image parity, CPU ownership counters, structured-ray
  coverage, certified-ray coverage, packet occupancy, surface-history survival,
  native-RT residual coverage, and routing overhead.
- [ ] Require AMD 780M >=30 FPS with no `content_quality` reduction, then run
  equivalent native-RT witnesses on an Intel GPU and Apple silicon before
  graduating those adapters. RTX comparisons identify optional adapter gains;
  they never become the default implementation criterion.
- [ ] On each device compare `native_rt_only` and `execution_fabric` using the
  same source hash, rays, shaders, materials, reconstruction, camera trace, and
  VRAM limit. On RTX the baseline enables every proven native accelerator,
  including CLAS, OMM, and SER where eligible.
- [ ] Run four fixed 240-frame workloads: dense downtown, glass-heavy business
  district, industrial/alpha-heavy district, and rapid transit/flyover. Include
  cold-to-warm streaming and structural scene-delta intervals.

**Authorized witness command:**

```bash
cd "$URBAN"
cargo run --release --bin mega_geometry_witness -- --map real_monterey --routes dense_downtown,glass_business,industrial_alpha,rapid_transit --modes native_rt_only,execution_fabric --output 2560x1440 --spp 1 --frames 240 --warmup 120 --config assets/config/render.ron --report artifacts/mega_geometry/city_witness.json
```

Required output:

```text
MEGAGEOMETRY_WITNESS source_assets=complete source_hash=<hash> backend=<optix_*|vulkan_khr|metal_native> mode=<native_rt_only|execution_fabric> output=2560x1440 spp=1 frames=240
MEGAGEOMETRY_FRAME trace_ms_p95=<n> shade_ms_p95=<n> reconstruct_ms_p95=<n> frame_ms_p95=<n> fps_p5=<n> geometry_vram_mb=<n> required_page_misses=0
MEGAGEOMETRY_QUALITY hit_mismatches=0 material_mismatches=0 radiance_mismatches=0 missing_detail=0 missing_textures=0 temporal_invalid_reuse=0
MEGAGEOMETRY_CPU lod_decisions=0 page_decisions=0 ray_scheduling=0 blocking_readbacks=0
MEGAGEOMETRY_PRODUCT portable_product=pass amd_780m_1440p_fps_p5>=30 adapter_gain_scoped=true
MEGAGEOMETRY_LEADERSHIP structured_ray_coverage>=0.60 certified_without_rt_traversal>=0.35 geometry_stage_speedup>=2.0 whole_frame_speedup>=1.20 visible_detail_per_vram_gain=<n> valid_history_invalidations_reduction>=0.50 quality_regressions=0 real_asset_families>=5
```

## Rollout order and stop conditions

1. Implement and validate Tasks 1–3 as the portable foundation.
2. Implement Task 4 as the primary breakthrough track. It must beat the
   native-RT-only complete-stage baseline before broad integration proceeds.
3. Land Task 5 after exact hit/certificate contracts are green; keep ReSTIR
   disabled in product config until the complete temporal path is proven.
4. Implement Task 6 adapters continuously against the same packet and hit ABI;
   no vendor adapter may redefine the execution fabric.
5. Run Task 7 only with explicit authorization for remote build/game launch.

Stop and fix the common path if any backend diverges in hit/radiance result,
requires a content fork, produces missing detail/textures, moves decisions to
CPU, or passes only through an offline render. An RTX win that does not meet
portable parity is an adapter experiment, not product progress.

## Self-review checklist

- [ ] Forge contains no vendor-specific geometry directive or cooked flag.
- [ ] Full authored content, materials, textures, and optical layers are
  covered by each package and retained across all detail cuts.
- [ ] Generic backend is implemented and witnessed before optional accelerators.
- [ ] The Execution Fabric eliminates proven work before native RT and its
  routing cost is included in every result.
- [ ] Native RT remains the exact unresolved/irregular/dynamic route on every
  capable product device; software traversal remains diagnostic-only.
- [ ] The packet machine has a portable subgroup implementation and optional
  hardware specializations selected only by measurement.
- [ ] Surface-space transport survives valid representation/detail changes and
  rejects ambiguous history locally.
- [ ] `render.ron`, not environment variables, owns product policy.
- [ ] All generic and optional paths return the same versioned hit ABI.
- [ ] No CPU-driven visibility, LOD, page scheduling, source geometry creation,
  or low-detail content fallback can pass the witness.
- [ ] The final claim distinguishes portable product evidence from
  adapter-specific performance evidence.
- [ ] The final leadership claim beats the strongest same-device native-RT-only
  baseline on complete city workloads, not synthetic geometry counts.
