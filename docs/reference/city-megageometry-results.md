# City MegaGeometry Results

## 2026-07-31 runtime-only benchmark evidence correction

The final benchmark no longer consumes or expects a Forge/cooker geometry
corpus. It loads the one authoritative `.Geometry` entry from each finished
`.vxp`, asks Ochroma to derive its disposable runtime programs, and measures
the resulting Spectra execution.

The raw JSON now retains partial and losing evidence instead of serializing it
only after a claim passes. Missing values are encoded and printed as
`unavailable`, never as a plausible zero. Exact runtime evidence includes:

- source, covered, and residual mesh bytes from the actual renderer streams;
- every Ochroma-derived program, template, selector, hierarchy,
  correspondence, visibility-cell, coefficient, and parameter byte;
- source/covered triangle counts and runtime program/template counts;
- measured full-frame, geometry-stage, peak-VRAM, correctness, coverage, and
  materialization comparisons from the same finished-object fixtures.

All three previously missing experiments are now wired without changing asset
authority:

- Ochroma derives one conservative taper envelope per recognized program.
  Spectra applies a non-identity update through a bounded GPU kernel, rejects
  out-of-envelope values without replacing the last correct value, and reports
  accepted changes, rejects, and fine-AS builds in a completed GPU receipt.
- The remap pass retains representation-local hit keys solely for an untimed
  no-correspondence oracle. The same real GPU cut transition now reports both
  raw-key invalidations and correspondence-preserved invalidations.
- Scalar procedural and complete native-reference visibility are timed with
  the same serialized GPU-event stage profile. The result is a measured win or
  loss, never a comparison between a micro-dispatch and full-frame wall time.

The persisted visibility calibration key now uses
`runtime_geometry_schema`, not `cook_schema`, and hashes every live
`visibility*.slang` source plus the shared ray-queue module. A manifest test
fails if a new visibility shader is added without joining the ABI hash, so
source or runtime-schema changes cannot reuse stale calibration evidence.
Legacy `cook_schema` JSON/RON is accepted only as a read-only migration alias.

The Urban Horizon product A/B compositor now fails closed on malformed
SHA-256 identities, empty timing intervals, non-finite or non-positive
metrics, inverted p50/p95 frame-time ordering, contradictory percentile FPS
ordering, a non-live present path, the wrong Vulkan/RT/fidelity identity,
missing streaming memory evidence, and asynchronous submit p95 over 0.25 ms
on the render thread or 0.50 ms on the host service. The live AMD witness
itself applies the same ship gates rather than trusting the composition step:
at least 30 FPS at p95 frame time, not merely at median, plus both submit
budgets.
Its `runtime_qem_calls` counter is now the warmup-boundary-to-final-frame
delta; legitimate asynchronous structural admission before measurement is
reported separately and no longer creates a false product failure.

The AMD acceptance artifacts now have unambiguous roles:
`final-amd-vulkan-bench.json` is Ochroma's `BenchReport` for cross-backend
composition, while `product-reference.json` and
`product-city-hybrid.json` are Urban Horizon `ProductWitness` inputs for the
inspected live product A/B. Reusing one filename for these incompatible schemas
would make final acceptance impossible and is forbidden.

Final acceptance no longer trusts the breakthrough labels serialized by either
raw backend producer. It recomputes geometry-program, ray-native, and certified
visibility-fabric claims from every backend's raw fixture receipts and requires
the stored aggregate to agree with that independent recomposition. A tampered
label or producer/composer drift therefore fails closed even when the stored
claim string says `pass`.

It also reruns the complete raw-backend validator on both artifacts. Packet and
control-hash agreement can no longer conceal missing prewarm evidence, a page
miss, an invalid timing boundary, dishonest execution selection, a forbidden
CPU operation, or a render-thread/host submission-budget failure on either
device.

The corpus identity is now SHA-256 rather than a 64-bit FNV diagnostic hash.
Capability and packet-ABI identities are derived from the renderer's measured
mode receipts rather than copied from operator-provided environment variables.
Final acceptance requires real GPU, driver, render-config, corpus, and Git
revision identities and rejects any environment/receipt disagreement.

The composed Urban Horizon product artifact now retains its scene, camera,
render-config, source-witness, displayed-capture, and visual-inspection
SHA-256 identities. Final acceptance requires the live product and both backend
reports to share exactly one render-config identity, closing the previous gap
between a valid product A/B and otherwise unrelated backend evidence.
The product composer independently enforces the candidate's 30 FPS-at-p95
floor and requires identical GPU-device and capability fingerprints across the
reference and city-hybrid captures; a manually asserted `gate_pass` cannot
override either condition.

The raw suite line now reports the measured sum of required page misses across
renderer-local fixtures. The separately authorized `mixed_city` product fixture
no longer forces that summary to print `unavailable`; any missing internal
counter still does.
The same renderer-local boundary now governs the suite correctness summary, and
the triangle oracle is mandatory for every internal fixture.
Unexpected-fallback totals now include the triangle oracle as well as reference
and city-hybrid modes while excluding only the external product placeholder.

The final `MEGAGEOMETRY_OVERALL` line is emitted by final acceptance, not by a
raw CUDA or Vulkan producer that lacks the external product witness and the
other backend. Acceptance report schema 2 attributes program and visibility
evidence to both measured devices rather than incorrectly naming CUDA alone.
The Task-11 hardware contract now reflects this mechanically: raw CUDA and
Vulkan reports must leave `mixed_city` unavailable and the overall winner
unset, while carrying their own raw packet/control/capability identities and
backend-local breakthrough measurements.
Each producer ends with an explicit
`MEGAGEOMETRY_RAW_ARTIFACT api=<cuda|vulkan> gpu=<device> validation=pass`;
process exit zero is no longer the only visible indication that its complete
backend-local artifact passed.

No breakthrough is claimed from local compilation. The raw CUDA and Vulkan
hardware suites must execute these experiments and the final compositor
requires both backends to pass each geometry-program, ray-native, and Fabric
claim; CUDA evidence can no longer authorize both backends.

Local verification:

```text
cargo test -p vox_render --test mega_geometry_bench_contract
51 passed; 0 failed

cargo test -p vox_render mesh_program --lib
4 passed; 0 failed

cargo test -p vox_render --example mega_geometry_bench --features spectra-native
2 passed; 0 failed; 1 ignored (installed product packs)

cargo test -p vox_render --example mega_geometry_acceptance
5 passed; 0 failed

cargo check -p vox_render --example mega_geometry_bench --features spectra-native
passed

cargo test -p vox_render --lib
436 passed; 0 failed; 5 ignored

cargo test -p spectra-renderer --no-default-features --lib
114 passed; 0 failed; 5 ignored

cargo test -p spectra-optix --test indirect_clas_contract --test cluster_provenance --test clas_templates
19 passed; 0 failed

cargo test -p spectra-scene-state
68 passed; 0 failed

cargo test -p spectra-gpu --test capability_contract
18 passed; 0 failed

cargo test --test vxp_runtime_load
6 passed; 0 failed (installed Urban Horizon packs)

cargo test --no-default-features --bin mega_geometry_product_compose
5 passed; 0 failed

slangc visibility_parameter_update.slang -entry updateVisibilityParameters -stage compute -target spirv
passed

slangc mega_geometry_remap.slang -entry writeLodRemapValidity -stage compute -target spirv
passed
```

## 2026-07-20 AMD Vulkan witness

Verdict: **PASS — compressed finished-object source, GPU-only realization,
native fixed-function triangle execution.** Forge was not linked or called by
the game, engine, or renderer. This result does not claim the CUDA/OptiX,
Metal, two-family, or certified-visibility-fabric gates.

The witnessed Meridian House payload contains 3,199 exact-cell programs, 49 templates, 3,199 selectors, and 9,597 patches. Its 409,472 program bytes replace 19,194 source facade triangles. Spectra materializes the canonical vertices, shading triangles, and tight RT indices with a one-shot Slang compute pass at residency; Vulkan consumes the tight device buffer directly when building triangle BLASes. There is no CPU geometry generation, de-indexing, repack, readback, or per-frame geometry scheduling.

Live realization proof:

```text
MEGAGEOMETRY_LIVE programs=3199 instances=1 source_triangles_removed=19194 program_bytes=409472 cpu_frame_scheduling=0 status=resident_scene
MEGAGEOMETRY_GPU_REALIZED programs=3199 patches=9597 triangles=19194 cpu_geometry_work=0 status=device_resident
```

Same release binary, map, camera, 1920x1080 display, 960x540 internal, 30 warmup + 60 measured frames:

```text
GPU-realized: render_median=39.03ms retained_rebuilds=0
triangle control: render_median=38.31ms retained_rebuilds=0
relative delta: +1.88%
```

Xwayland throttled presentation to 1 Hz after the foreground window lost compositor priority. Therefore the present/sustained FPS fields from these long runs are invalid as a product ship metric; only the same-session render medians are used for the representation A/B.

The focused 128x96, 1 spp, two-bounce renderer equivalence test reports:

```text
execution=native-triangle source_triangles_removed=2 gpu_realized_triangles=2
cpu_geometry_work=0 triangle_primary_ms=0.245358 mega_primary_ms=0.171390
max_image_error=0.000000149 status=pass
```

Human-inspected live-present captures:

- `/home/tom-espen/Ochroma/projects/urban_horizon/artifacts/mega_geometry_live/meridian_gpu_realized_current_frontage.png`
- `/home/tom-espen/Ochroma/projects/urban_horizon/artifacts/mega_geometry_live/meridian_gpu_realized_v4_detail_settled.png`

Verification commands and results:

```text
cargo test -p spectra-renderer --test mega_geometry_render_gpu -- --nocapture
1 passed; 0 failed; max_image_error=0.000000149

cargo test -p spectra-renderer --test cpu_loop_contract --test geometry_backend_conformance
9 passed; 0 failed

cargo test -p spectra-gpu --test capability_contract
10 passed; 0 failed

cargo test -p vox_data --test mega_geometry_asset
2 passed; 0 failed

cargo test --release --features asset-cook --bin game_asset_cook meridian_directive_requires_and_cooks_exact_procedural_payload -- --nocapture
MERIDIAN_MEGAGEOMETRY programs=3199 patches=9597 triangles=19194
1 passed; 0 failed
```

## 2026-07-21 NVIDIA OptiX true-CLAS witness

Verdict: **PASS — the live OptiX path calls `optixClusterAccelBuild` and the
typed diagnostic reports a real hardware CLAS, not a renamed triangle GAS.**
This is a focused 1,000-instance cube witness at 256x144 internal resolution;
it is not the production-corpus, million-instance, or 4K ship gate.

The shipping scene builder now uses true CLAS for static prototypes. Dynamic
vertex prototypes and OMM prototypes remain explicit, counted triangle-GAS
compatibility paths. Deterministic source-order cluster partitioning preserves
`global_triangle_base + local_primitive`, rejects out-of-range indices, and is
covered by a hardware-independent provenance test.

Final RTX 4070 Ti run:

```text
[clas-build] OK: IAS over 1000 instances / 1 protos is the traversable
[optix-timing] clas_host_build=0.5ms clas_device_build=6.4ms protos=1 instances=1000 omm_cache=true
N=1000 build_ms=91.8 refit_ms_avg=0.012 render_ms_avg=2.911 fps=342.1 instances=1000 clusters=1 hardware_clas=1 triangle_gas_fallbacks=0 vram_mb=8702
RUN_DONE_EXIT=0
```

The 2.911 ms number is a two-frame micro-witness and is not a stable performance
claim. Its purpose is acceleration-kind proof. The benchmark now fails closed
when OptiX is absent or the hardware CLAS count is zero; a Vulkan-only build was
deliberately exercised and rejected before the OptiX-featured run.

Build and contract proof:

```text
Windows default GPU wrapper:
cargo build -p vox_render --features spectra-native,spectra-native-optix
BUILD_DONE_EXIT=0

cargo test -p spectra-optix --test accel_stats_contract
2 passed; 0 failed

cargo test -p spectra-optix --test cluster_provenance
3 passed; 0 failed

cargo test -p spectra-optix --test indirect_clas_contract
2 passed; 0 failed

cargo test -p spectra-renderer --test cpu_loop_contract
5 passed; 0 failed

cargo test -p spectra-gpu --test capability_contract
10 passed; 0 failed

cargo test -p spectra-renderer --test geometry_backend_conformance
4 passed; 0 failed
```

### 2026-07-21 device-native OptiX build chain

The CLAS bridge no longer materializes native OptiX argument structs on the
host or reads a CLAS/GAS handle back before building the IAS. Structural
admission uploads compact source records only. CUDA kernels in the shipped PTX
write the SDK-native triangle/GAS argument structs and their `argsCount` values
on the render stream; OptiX writes each GAS handle to retained device memory;
another CUDA kernel dereferences those outputs to build native `OptixInstance`
records for the IAS. Transform-only IAS updates use the same device record path.

The resulting target-machine build and live OptiX run completed successfully:

```text
[build-gpu] BUILD_EXIT=0
BUILD_DONE_EXIT=0
N=100000 build_ms=129.0 refit_ms_avg=1.374 render_ms_avg=8.262 fps=103.8 instances=100000 clusters=1 hardware_clas=1 triangle_gas_fallbacks=0
N=1000000 build_ms=223.1 refit_ms_avg=15.337 render_ms_avg=8.120 fps=42.6 instances=1000000 clusters=1 hardware_clas=1 triangle_gas_fallbacks=0
RUN_DONE_EXIT=0
```

This validates the live native chain, including CUDA module loading and the
device-built IAS records. It is still the synthetic single-material cube sweep;
the numbers are not a production performance claim.

Focused contracts now pass:

```text
cargo test -p spectra-optix --test indirect_clas_contract
3 passed; 0 failed
```

Still open before the full Task-3 claim: multi-material production-fixture
image/hit parity, followed by the hierarchical IAS, continuous LOD, paging,
template, policy, and full fixture/ship gates.

## Rejected 2026-07-21 authoring-boundary experiment

This historical experiment incorrectly made Forge emit a MegaGeometry product.
Its numbers are retained only as old diagnostic evidence and do not describe
the current architecture. Forge now emits one ordinary finished mesh; Ochroma
derives runtime geometry after load.

```text
[blueprint] city.res_high.l5.8x8.meridian_house_01: 446 sub-meshes, tris=114064 -> boundary=386 non-manifold=0, 512-sweep interior=261 exterior=251 fractional=0
MERIDIAN_MEGAGEOMETRY programs=3199 patches=9597 triangles=19194
test mega_geometry_ready::tests::meridian_directive_requires_and_cooks_exact_procedural_payload ... ok
```

The RTX CLAS scale fixture now uses six distinct per-triangle material zones
(facade, roof, trim, glass, door, and metal) rather than one material. The
target hardware sweep completed with true CLAS at every scale, including one
million instances:

```text
N=100000 build_ms=112.4 refit_ms_avg=1.175 render_ms_avg=7.723 fps=112.4 instances=100000 clusters=1 hardware_clas=1 triangle_gas_fallbacks=0
N=1000000 build_ms=224.5 refit_ms_avg=15.323 render_ms_avg=8.843 fps=41.4 instances=1000000 clusters=1 hardware_clas=1 triangle_gas_fallbacks=0
```

This is material-proven true-CLAS plumbing, not yet the required live Meridian
image/hit-parity witness. That remains an explicit gate.
