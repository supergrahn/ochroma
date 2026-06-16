# Design: OptiX HWRT Traversal (2026-06-16)

**Status:** Draft
**Scope:** Wire OptiX RT-core traversal as the ray-intersection back-end for the Spectra path tracer on Windows/NVIDIA, replacing the software BVH and KHR ray-query kernels with a GPU-resident wavefront split. Affects `spectra-optix`, `spectra-renderer`, and the Slang megakernel.
**Related:** [Render pipeline audit 2026-06-13](../../memory/render-pipeline-audit-2026-06-13.md), [DLSS NGX-D3D12 architecture](../../memory/dlss-ngx-d3d12-architecture.md)

---

## 1. Problem Statement

- `detect::should_use_optix_rt()` had no Windows branch: on the target GPU box (`tomespensin`, RTX 4070 Ti) the function always returned `false` even with `nvoptix.dll` present, so `OptiXTraversal::available()` was permanently `false` and the OptiX path was never taken.
- `trace_ias` and `trace_clas_ias` in `traversal.rs` / `clas.rs` are pure-CPU software fallbacks — no `optixAccelBuild` / `optixLaunch` call exists in the Rust layer; OptiX traversal is claimed but not implemented.
- The renderer's wavefront split (`sample.rs` lines 710-758) has a dead `use_optix` branch (`// OptiX hardware path: traversal handled...` with no actual dispatch), so CUDA megakernel and OptiX traversal are never bridged via hit buffers.
- The denoiser path (`denoiser_ffi.rs`) is gated on `denoiser_ffi_compiled` cfg and not yet exercised on Windows.

---

## 2. Done When

Running `cargo run --example optix_smoke` from `/home/tom-espen/src/spectra/rust/spectra-optix/` on the Windows build box (`tomespensin`) with the NVIDIA driver installed prints:

```
OPTIX_SMOKE: init=ok context=ok device="NVIDIA GeForce RTX 4070 Ti"
```

(Phase 0 gate — proves the detection fix works.)

Running the renderer with `--shot-map reference.exr` on a 256x256 Cornell box scene produces a pixel diff of < 1% between the OptiX traversal path and the software BVH path:

```
spectra-render --scene cornell_box.json --shot-map sw_ref.exr --no-optix
spectra-render --scene cornell_box.json --shot-map hw_ref.exr
python tools/pixel_diff.py sw_ref.exr hw_ref.exr   # prints "max_diff=0.XXX < 0.01: PASS"
```

(Phase 1 gate.)

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Windows DLL detection | `nvoptix_dll_present()` returns `Some(true)` on `tomespensin`; `optix_smoke` exits 0 | `should_use_optix_rt()` returns `true` unconditionally |
| optixInitWithFlags probe | `optix_init_probe()` returns `true`; logged as `OPTIX_SUCCESS` in SPECTRA_LOG=debug | function exists but always returns true |
| GAS build calls optixAccelBuild | `build_gas()` allocates a real device buffer and the returned handle's `gas_built()` is `true` with a non-zero `OptixTraversableHandle` | `gas_built` flag set to `true` with no device call |
| Wavefront hit buffer write | After `OptiXTraversal::trace()`, `g_hit_prim_id[0]` equals the known triangle index for a ray through the centroid of the Cornell box floor | `prim_ids[0] != MISS_PRIM_ID` assertion |
| CLAS at 100K instances | `trace_clas_ias` on a 100K-instance scene completes in < 5ms GPU time | result count matches input ray count |
| AI denoiser (optional) | `CudaDenoiserHandle::denoise()` reduces noise PSNR by >= 5dB on a 1-SPP render | `is_available()` returns true |

---

## 4. Architecture

### 4.1 Windows Detection Fix (`detect.rs`)

The `should_use_optix_rt()` function is the single entry point used by `OptiXTraversal::available()`. It previously fell through to `optix_rt_loadable()` which only checked Linux `.so` paths.

The fix adds a `#[cfg(windows)]` early-return that calls `should_use_optix_rt_windows()`, which runs three checks in order:
1. `SPECTRA_USE_OPTIX=0` env var -> return `false` immediately with an `info!` log.
2. `nvoptix_dll_present()` — uses `libloading::Library::new("nvoptix.dll")` (System32 path) then falls back to enumerating `C:\Windows\System32\DriverStore\FileRepository\nv_dispsi.inf_*\nvoptix.dll` with a 1MB size gate to reject stubs.
3. `optix_init_probe()` — loads `nvoptix.dll` again and calls `optixInitWithFlags(0)` directly (the exported symbol underlying the static-inline `optixInit` from `optix_stubs.h`). Returns `OPTIX_SUCCESS == 0` on success.

All Linux functions (`is_wsl2`, `optix_rt_loadable`, `optix_rt_ext_available`, `optix_lib_search_paths`) are gated to `#[cfg(unix)]`. Platform-specific tests are gated to match.

### 4.2 GAS/IAS Build — Real OptiX AccelBuild (Phase 1)

`OptiXTraversal::build_gas()` currently stores geometry CPU-side for the software fallback. Phase 1 adds a `#[cfg(all(feature = "optix-sdk", optix_ptx_compiled))]` branch that:

1. Uploads `vertices` / `indices` to device memory as `CUdeviceptr` buffers.
2. Fills `OptixBuildInput` (triangle array type).
3. Calls `optixAccelBuild` with `OPTIX_BUILD_FLAG_ALLOW_COMPACTION`.
4. Stores the returned `OptixTraversableHandle` in `OptiXTraversal.gas_handle: Option<u64>`.

`build_ias` similarly calls `optixAccelBuild` with `OptixInstance` array inputs.

`build_clas_ias` will call the OptiX 9 `optixClusterAccelBuild` API (gated on `rt-cores` feature) when available; the existing software fallback (`clas.rs::build_clas_ias`) remains as the fallback path.

### 4.3 Wavefront Split — Trace to Hit Buffers (Phase 1)

The current dead branch in `sample.rs` lines 717-723 must be wired:

```rust
if use_optix {
    // Phase 1: call OptiXTraversal::trace() or trace_ias(),
    // which launches the PTX ray-gen program via optixLaunch.
    // Results write directly into the GPU SoA hit buffers
    // (g_hit_prim_id, g_hit_instance, g_hit_t, g_hit_u, g_hit_v)
    // that are already bound by ray_queue_bufs.bind_shade_inputs().
    // The CUDA megakernel (SHADE_MEGAKERNEL) reads these buffers
    // exactly as it does after the software BVH traversal kernel.
    if let Some(trav) = self.state.as_ref().and_then(|s| s.optix_traversal.as_ref()) {
        trav.launch_trace_into_hit_buffers(&self.gpu, &bindings)?;
    }
}
```

The PTX program (`optix_programs.ptx`, compiled by `build.rs`) implements the ray-gen / closest-hit shaders that write to the bound `CUdeviceptr` hit buffers. The CUDA megakernel reads those buffers unchanged.

### 4.4 CUDA/OptiX Buffer Interop

All buffers are `CUdeviceptr` allocated by `spectra-gpu`. The OptiX context is created with `cuCtxCurrent()` — the same context used by the CUDA megakernel — so no buffer copies or interop barriers are required. The flow per bounce:

```
camera_ray_gen (CUDA) -> writes g_ray_o, g_ray_d
optixLaunch (RT cores) -> reads g_ray_o/d, writes g_hit_prim_id/instance/t/u/v
shade_megakernel (CUDA) -> reads all buffers, writes next-bounce rays + radiance
```

### 4.5 CLAS / RTX Mega Geometry (Phase 3)

`trace_clas_ias` in `traversal.rs` calls `clas::trace_clas_ias` which is currently software-only. Phase 3 adds a hardware path that calls `optixClusterAccelBuild` (OptiX 9 API) and `optixLaunch` with a CLAS-aware PTX. The streaming LOD machinery (`load_clas_cluster` / `evict_clas_cluster`) maps directly to OptiX 9's cluster residency model.

### 4.6 OptiX AI Denoiser (Phase 4, optional)

`CudaDenoiserHandle` in `denoiser_ffi.rs` is already structured. Phase 4 gates it on the `ai-denoiser` feature, wires `spectra_denoiser_ffi_compiled` cfg in `build.rs`, and integrates it into the per-frame resolve in `render.rs` after the sample accumulation loop. The temporal variant (`denoise_temporal`) uses the existing motion vector buffers.

---

## 5. Data Models

```rust
/// Extended to hold the OptiX traversable handle (hardware GAS).
pub struct OptiXTraversal {
    // ... existing fields ...

    /// Non-zero when a real OptiX GAS has been built (hardware path only).
    /// Zero means the software fallback is active.
    gas_handle: u64,  // OptixTraversableHandle
    ias_handle: u64,  // OptixTraversableHandle for IAS
}

/// Result written by the OptiX PTX ray-gen program into GPU SoA hit buffers.
/// Layout must match the Slang `HitRecord` struct in the megakernel.
#[repr(C)]
pub struct GpuHitRecord {
    pub prim_id: i32,       // -1 = miss
    pub instance_id: i32,   // -1 = miss
    pub t_hit: f32,         // 1e9 = miss
    pub hit_u: f32,
    pub hit_v: f32,
}
```

---

## 6. API

Real function signatures from the codebase (read from `traversal.rs`, `clas.rs`, `ffi.rs`, `denoiser_ffi.rs`):

```rust
// traversal.rs
impl OptiXTraversal {
    pub fn new() -> Self;
    pub fn available(&self) -> bool;  // checks #[cfg(all(feature="optix-sdk", optix_ptx_compiled))]
    pub fn build_gas(&mut self, vertices: &[[f32; 3]], indices: &[[u32; 3]], allow_update: bool) -> bool;
    pub fn refit_gas(&mut self, vertices: &[[f32; 3]]) -> bool;
    pub fn build_prototype_gas(&mut self, proto_id: i32, vertices: &[[f32; 3]], indices: &[[u32; 3]]) -> bool;
    pub fn build_ias(&mut self, transforms: &[[f32; 12]], proto_ids: &[i32]) -> bool;
    pub fn trace(&self, origins: &[[f32; 3]], directions: &[[f32; 3]]) -> BatchTraceResult;
    pub fn trace_ias(&self, origins: &[[f32; 3]], directions: &[[f32; 3]]) -> InstancedTraceResult;
    pub fn build_cluster_gas(&self, vertices: &[[f32; 3]], indices: &[[u32; 3]], cluster_size: usize) -> Result<CLASHandle, OptixError>;
    pub fn build_clas_ias(&mut self, instances: Vec<CLASInstance>) -> Result<CLASIASHandle, OptixError>;
    pub fn trace_clas_ias(&self, origins: &[[f32; 3]], directions: &[[f32; 3]]) -> Result<Vec<TraceResult>, OptixError>;
    pub fn load_clas_cluster(&mut self, instance_id: i32, clas: CLASHandle);
    pub fn evict_clas_cluster(&mut self, instance_id: i32) -> bool;
    pub fn build_vegetation_clas(&mut self, prototype_meshes: &[PrototypeMesh], instance_transforms: &[[[f32; 4]; 4]]) -> Result<bool, OptixError>;
    pub fn destroy(&mut self);
}

// clas.rs
pub fn build_cluster_gas(vertices: &[[f32; 3]], indices: &[[u32; 3]], cluster_size: usize) -> Result<CLASHandle, OptixError>;
pub fn build_clas_ias(instances: Vec<CLASInstance>) -> Result<CLASIASHandle, OptixError>;
pub fn trace_clas_ias(ias: &CLASIASHandle, origins: &[[f32; 3]], directions: &[[f32; 3]]) -> Vec<TraceResult>;
pub fn build_vegetation_clas(prototype_meshes: &[PrototypeMesh], instance_transforms: &[[[f32; 4]; 4]]) -> Result<CLASIASHandle, OptixError>;

// detect.rs (new Windows API added in this commit)
#[cfg(windows)]
pub fn nvoptix_dll_present() -> Option<bool>;
#[cfg(windows)]
pub fn optix_init_probe() -> bool;
#[cfg(windows)]
pub fn should_use_optix_rt_windows() -> bool;
pub fn should_use_optix_rt() -> bool;  // dispatches by platform

// ffi.rs (note: optixInit is static inline in optix_stubs.h, NOT callable via FFI)
// raw::* generated by bindgen when optix-sdk feature is active
// OPTIX_PTX: &[u8] — embedded PTX, always exists (empty placeholder when uncompiled)

// denoiser_ffi.rs
impl CudaDenoiserHandle {
    pub fn is_available() -> bool;
    pub fn try_new() -> Option<Self>;
    pub fn denoise(&self, width: u32, height: u32, color: &[f32], albedo: Option<&[f32]>, normal: Option<&[f32]>) -> Result<Vec<f32>, String>;
    pub fn denoise_temporal(&self, width: u32, height: u32, color: &[f32], albedo: Option<&[f32]>, normal: Option<&[f32]>, flow: Option<&[f32]>) -> Result<Vec<f32>, String>;
    pub fn reset_temporal(&self);
}
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `should_use_optix_rt_windows()` | `should_use_optix_rt()` | `spectra-optix/src/detect.rs` | `#[cfg(windows)]` early return |
| `OptiXTraversal::available()` | `RendererState` check per bounce | `spectra-renderer/src/renderer/sample.rs:710` | Already called; gate is the detection fix |
| `optixAccelBuild` (Phase 1) | `OptiXTraversal::build_gas()` | `spectra-optix/src/traversal.rs` | Behind `#[cfg(all(feature="optix-sdk", optix_ptx_compiled))]` |
| `optixLaunch` (Phase 1) | dead branch lines 717-723 | `spectra-renderer/src/renderer/sample.rs` | Replaces the empty comment block; hits `g_hit_prim_id` etc. |
| `trace_clas_ias` hw path (Phase 3) | `OptiXTraversal::trace_clas_ias()` | `spectra-optix/src/traversal.rs:468` | Adds `optixClusterAccelBuild` call |
| `CudaDenoiserHandle` (Phase 4) | frame resolve | `spectra-renderer/src/renderer/render.rs` | After sample accumulation loop |

---

## 8. Open Questions

- [ ] **Does `trace_ias` / `trace_clas_ias` call `optixAccelBuild`?** Confirmed NO — both are pure-CPU software implementations (`traversal.rs` and `clas.rs` read in full). `optixAccelBuild` does not appear anywhere in the Rust layer. This is the primary Phase 1 gap.
- [ ] **PTX compilation on Windows**: `build.rs` currently writes an empty placeholder PTX when `nvcc` / Slang is absent. Phase 1 requires a real PTX ray-gen + closest-hit program. The build script must detect `nvcc` on Windows (CUDA toolkit in PATH) and compile `optix_programs.cu` to `optix_programs.ptx`.
- [ ] **OptiX device context ownership**: The CUDA context created by `cuCtxCreate` in the smoke test is separate from any context `spectra-gpu` creates. Phase 1 must ensure `optixDeviceContextCreate` receives the same `CUcontext` that the GPU buffer allocator uses, or interop copies will be needed.
- [ ] **`required_kernel_names` guard in `render.rs:905`**: the `use_optix` branch suppresses `BVH_TRAVERSE` / `BVH_TRAVERSE_HW` from the required kernel list. After Phase 1, this logic is correct; but if `optixLaunch` fails at runtime, the renderer must fall back gracefully to the software BVH kernel rather than hanging.

---

## 9. Phased Plan

### P0 — Detection Fix (this commit)
**Done When:** `cargo run --example optix_smoke` on `tomespensin` (Windows, RTX 4070 Ti, NVIDIA driver installed) prints exactly:
```
OPTIX_SMOKE: init=ok context=ok device="NVIDIA GeForce RTX 4070 Ti"
```
Files changed: `spectra-optix/src/detect.rs` (Windows branch added), `spectra-optix/examples/optix_smoke.rs` (new).

### P1 — Real GAS/IAS + Wavefront Wiring
**Done When:**
```
spectra-render --scene cornell_box.json --shot-map hw.exr
python tools/pixel_diff.py sw.exr hw.exr
```
prints `max_diff=X.XX < 0.01: PASS` and the render log shows `[optix] GAS built: N vertices, M triangles (device)`.

Files to change:
- `spectra-optix/src/traversal.rs` — add `optixAccelBuild` call in `build_gas`/`build_ias` behind feature gate.
- `spectra-renderer/src/renderer/sample.rs` lines 717-723 — wire `trav.launch_trace_into_hit_buffers(...)`.
- `spectra-optix/build.rs` — compile `optix_programs.cu` to PTX on Windows when `nvcc` is in PATH.

### P2 — Trace Time < 1s for Reference Scene
**Done When:** `spectra-render --scene reference_256.json --spp 1 --timing` prints `trace_ms=XX` where XX < 1000 on the RTX 4070 Ti.

### P3 — CLAS at 100K Instances
**Done When:** `spectra-render --scene city_100k.json --spp 1` completes in < 5ms GPU trace time, logged as `[clas] trace_ms=X.X`.

Files to change:
- `spectra-optix/src/clas.rs` — add `optixClusterAccelBuild` path behind `rt-cores` feature.
- `spectra-optix/src/traversal.rs:468` — `trace_clas_ias` hardware dispatch.

### P4 — < 33ms at 4K with DLSS-RR
**Done When:** `spectra-render --res 3840x2160 --spp 1 --dlss rr` shows `frame_ms < 33` in the window title on the RTX 4070 Ti.

Files to change: `spectra-renderer/src/renderer/render.rs` (denoiser wiring after sample loop).

---

## 10. Out of Scope

- Multi-GPU OptiX: this design targets a single device context on the primary GPU.
- OptiX motion blur (`OptixMotionOptions`): not required for Phase 1-3.
- CPU-side path-tracing fallback when OptiX is unavailable: the existing software BVH is the fallback; this design does not remove it.
- Linux/AMD OptiX: OptiX is NVIDIA-only; Linux uses the existing KHR ray-query / software BVH path unchanged.

---

## 11. Related Plans / Designs

- Depends on: [DLSS NGX-D3D12 architecture](../../memory/dlss-ngx-d3d12-architecture.md) (shared CUDA context + D3D12 present)
- Required before: instancing/LOD WT-1 design (CLAS at 100K instances is the Phase 3 gate)
- Related: [Render pipeline audit 2026-06-13](../../memory/render-pipeline-audit-2026-06-13.md) — identified the dead OptiX branch and the instancing gap
