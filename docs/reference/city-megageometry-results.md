# City MegaGeometry Results

## 2026-07-20 AMD Vulkan witness

Verdict: **PASS — compressed Forge source, GPU-only realization, native fixed-function triangle execution.** This result does not claim the CUDA/OptiX, Metal, two-family, or certified-visibility-fabric gates.

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

## 2026-07-21 Forge source and multi-material CLAS checks

The real Forge Meridian source cook passed its exact-procedural MegaGeometry
contract:

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
