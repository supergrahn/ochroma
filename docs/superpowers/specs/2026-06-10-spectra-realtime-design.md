# Design: Spectra Realtime — fixed 1–2 ms/frame path tracing by construction (2026-06-10)

**Status:** Draft
**Scope:** Turn Spectra from a seconds-per-still spectral path tracer into a fixed-budget realtime renderer — ≤ 2.0 ms GPU per frame on the target machine (`tomespensin`: Windows, RTX 4070 Ti), with frame cost bounded by pixel count, never by scene size — via hardware RT traversal, ReSTIR reservoir reuse, world-space radiance caching for deep bounces, and DLSS-RR-class neural reconstruction. Quality accumulates across frames; milliseconds do not.
**Related:** [Virtualized Splat Rendering design](./2026-06-10-virtualized-splat-rendering-design.md) (the interactive wgpu path this must converge with — §4.8), [SOTA City Block Phase 1 plan](../plans/2026-06-10-sota-city-block-phase1.md) (Tasks 3–8 harden the stills path this builds on).

**The directive, verbatim:** *"I want Spectra to be realtime, and I mean realtime. 1-2ms/frame, tops. No matter what scene and no matter what complexity."*

This document commits to that directive and states precisely what it physically means: the **milliseconds are fixed by construction**; the **image quality at any instant is the only free variable**, and it converges over fractions of a second of temporal reuse. Any reading of the directive under which a single 2 ms frame contains a converged path-traced image is physically impossible on any hardware that exists (§4.7); no design is offered for that reading.

---

## 1. Problem Statement

Concrete, observable symptoms in the code as of 2026-06-10:

- **Spectra renders stills in seconds, not milliseconds.** The proven runs on the dev floor (AMD 780M / RADV, `spectra-pathtracer-runs` memory) are seconds per frame. The one "realtime" artifact, `RenderConfig::near_realtime` (`rust/spectra-renderer/src/render_config.rs:230`), targets "~25 fps on RTX 4070 Ti at 1080p with DLSS" — *in a comment, never measured on that machine* — i.e. ~40 ms, 20–40× over the directive.
- **All ray traversal in the actual frame loop is software compute.** `spectra-gpu/src/vulkan_backend.rs` (1,026 lines) contains zero `VK_KHR_ray_tracing` usage; traversal is `bvh_traverse.slang` (stack-based BVH, Möller–Trumbore) dispatched as a plain compute kernel. The RT cores on the 4070 Ti are never touched by a Spectra frame.
- **The hardware-RT code that exists is unwired.** `render_state.rs:235` hardcodes `optix_traversal: None`; the `use_optix` branch in `renderer.rs:1548` is an empty block with a comment admitting "the software stub always returns false, so this block never executes." `rust/spectra-vulkan` (a real ash `VK_KHR_ray_tracing_pipeline` crate) and `optix/rt` (a real OptiX module with GAS/IAS builds and SER programs) both expose **host-slice batch APIs** (`trace(&self, origins: &[[f32;3]], …)`) built for the old Python engine — incompatible with a GPU-resident wavefront, and `VulkanRTTraversal::trace_tlas` currently falls through to a *CPU* software trace.
- **RTX Mega Geometry support is an explicit stub.** `optix/clas/clas_builder.cu` is 8 lines: `bool is_available() { return false; }  // Stub until OptiX 9 CLAS API wired`.
- **The engine-side seam is a CPU readback.** `vox_render::splat_backend::SpectraRenderBackend` ships frames as `Arc<Vec<u8>>` host pixels (14.7 MB/frame at 1440p RGBA8). `vox_render/src/dlss.rs` and `denoiser.rs` are CPU stand-ins (bilinear upscale fallback; bilateral filter over `[[u8;4]]`) — no real reconstruction exists engine-side.
- **The spectral reservoir layout cannot fit a realtime bandwidth budget.** `restir_pt.slang`'s spectral `PathReservoir` is 48 floats = 192 B/pixel; at 0.92 M internal pixels × multiple reservoir passes that is >1 GB/frame of traffic — ~2 ms of the 4070 Ti's 504 GB/s on reservoirs alone.
- **DLSS exists but is Linux-shaped.** `spectra-dlss` runtime-loads `libnvidia-ngx.so` with Linux/WSL2 search paths only (`detect.rs`); the target machine is Windows, where the surface is `nvngx_dlss*.dll` / Streamline, and DLSS Ray Reconstruction is not integrated at all (only DLSS-SR).

---

## 2. Done When

**Headline (all milestones landed), on `tomespensin` (Windows, RTX 4070 Ti, 1440p):**

Running `cargo run --release --bin spectra_realtime_probe -- --width 2560 --height 1440 --frames 600 --scene city_block` prints

```
[spectra-rt] gpu=NVIDIA GeForce RTX 4070 Ti backend=optix|vk-rt internal=1280x720 output=2560x1440
[spectra-rt] pass ms p50/p99: as_update A/A' | primary B/B' | restir_di C/C' | restir_gi D/D' | cache E/E' | reconstruct F/F' | total T/T'
[spectra-rt] total p50 = T ms (budget 2.00) p99 = T' ms (budget 2.40)  -> PASS
[spectra-rt] convergence: FLIP vs offline reference @frame 60 = X (gate 0.15) -> PASS
```

with **T ≤ 2.00** and **T' ≤ 2.40** (GPU timestamps, not wall clock), and a human at the keyboard sees a window orbiting the scene with no visible hitching and lighting that visibly settles within ~1 second after each camera cut. Doubling the scene's instance count (`--scene city_block_2x`) changes `total p50` by **< 10%** — printed side by side by `--scene-scaling-check`.

The same binary on the dev floor (AMD 780M / RADV, Linux) prints the same table with `backend=vk-rt` and **no ms gate** — the floor proves correctness and prints honest numbers; it is never the bar. Until CI exists on `tomespensin`, the 4070 Ti lines are verified manually at that keyboard and pasted into the milestone PR description (protocol in §4.5/M0).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Fixed budget vs scene size | probe renders scene N and scene 2N: `assert!((p50_2n - p50_n).abs() / p50_n < 0.10)` with both p50 values printed | `assert!(p50 > 0.0)` |
| HW traversal actually faster | M1 prints compute-BVH vs HW-RT traversal ms on the same scene/rays; `assert!(sw_ms / hw_ms >= 5.0)` on the 4070 Ti, ratio printed | `assert!(hw_backend.is_some())` |
| HW/SW image parity | per-pixel hit comparison HW vs compute oracle on 100k deterministic rays: `assert!(mismatched_prim_ids < 0.1%)`, count printed | `assert!(image.len() > 0)` |
| Reservoir reuse buys variance | M2 prints relMSE vs offline reference with temporal reuse on/off at equal ray budget; `assert!(mse_on < 0.25 * mse_off)`, both printed | asserting the reservoir buffer is non-zero |
| Cache replaces deep bounces | M3: GI image with traced 4-bounce vs bounce-1 + cache after 120-frame warmup; `assert!(flip_score < 0.10)` printed, and deep-bounce pass ms row drops to the cache-query cost (both ms printed) | `assert!(cache.entry_count() > 0)` |
| Reconstruction ratio | M4 prints `internal=1280x720 output=2560x1440 reconstruct_ms=F`; `assert!(f >= 0.0)` is forbidden — gate is `assert!(reconstruct_ms <= 1.0)` on the 4070 Ti, printed | upscaled buffer dimensions match |
| Cut recovery envelope | scripted teleport at frame 300: probe prints per-frame FLIP for frames 300–360; `assert!(flip[330] < 0.5 * flip[300])` (printed curve) and `assert!(total_ms[300..310].iter().all(|m| *m <= 2.4))` — ms must NOT spike during recovery | asserting no crash on teleport |
| Zero-copy present | M2: frame counter in window title advances ≥ 120 fps while probe prints `present=shared-image readback=0 B/frame` | checking the swapchain exists |

---

## 4. Architecture

### 4.1 The budget contract, precisely

**The contract:** every Spectra-realtime frame submits a GPU pass stack whose measured GPU time is ≤ 2.0 ms p50 / 2.4 ms p99 at 1440p output on the RTX 4070 Ti. The stack's work is a function of **internal pixel count and fixed per-pixel ray/sample counts only**. Scene complexity may move the *constants* (BVH depth, cache hit rate) — the governor (§4.4) absorbs constant drift by adjusting internal resolution, never by exceeding the budget. Quality at rest converges over frames; quality during motion is whatever the reservoirs + cache + reconstructor can carry.

**Hardware constants (RTX 4070 Ti, AD104):** 60 SMs / 7,680 CUDA cores, 60 third-gen RT cores, ~40 TFLOPS FP32, 12 GB GDDR6X at **504 GB/s**, 48 MB L2. Two budget axes matter and both are ledgered: ray-segment throughput and DRAM bandwidth. At 2.0 ms the frame may touch at most ~1.0 GB of DRAM traffic (504 GB/s × 2 ms), in practice far less because reservoirs/G-buffer should live in L2 at 720p internal.

**Resolution ladder (DLSS-RR ratios, per axis):** 1440p output → internal 1707×960 (Quality, 67%), 1486×835 (Balanced, 58%), **1280×720 (Performance, 50%) — the contract point**, 854×480 (Ultra Perf, 33%, governor floor). 4K output on this card is **not** contracted at 2 ms: 4K Performance = 1920×1080 internal = 2.07 M pixels ≈ 2.25× the trace work — the honest 4K statement is Ultra Performance (1280×720 internal) *or* a 3 ms tier; this is stated now, not discovered later.

**Ray-segment ledger at the contract point (0.92 M internal pixels):**

| Rays | Count/frame | What |
|---|---|---|
| Primary visibility | 0.92 M (coherent) | RT primary OR consumed from the splat-raster G-buffer (§4.8); coherent rays are the cheap case |
| ReSTIR DI shadow | 0.92 M | 1 visibility ray for the winning reservoir candidate |
| ReSTIR GI bounce | 0.92 M (incoherent) | 1 guided bounce segment per pixel (path-guide / NRC-informed) |
| ReSTIR GI shadow (NEE at bounce vertex) | 0.92 M | 1 |
| Cache update (sparse) | ~0.25 M | 1/16 of pixels trace short multi-bounce update paths to feed the cache |
| **Total** | **≈ 3.9 M segments** | everything deeper than bounce 1 is a **cache query, not a ray** |

Planning assumption (to be **replaced by the M1 measured number**, which is the load-bearing one): the 4070 Ti sustains 2–5 G incoherent ray segments/s through hardware traversal with SER and hit-shading in a wavefront. 3.9 M segments → **0.8–2.0 ms of trace**. Production reference for plausibility: Cyberpunk 2077 full path tracing ships at "2 rays / 2 bounces" per pixel with ReSTIR DI/GI on this hardware class at comparable internal resolutions ([NVIDIA/CDPR](https://www.nexusmods.com/cyberpunk2077/articles/980)). Our per-pixel traced work is the same order.

**Target per-pass ms ledger at the contract point (4070 Ti, planning targets — every number is replaced by a printed measurement at its milestone):**

| Pass | Target ms | Bound by |
|---|---|---|
| AS update/refit (amortized, capped) | 0.10 | budgeted queue, §4.5/M5 |
| Primary visibility (RT, coherent) | 0.20 | RT cores |
| ReSTIR DI (gen + temporal + spatial + 1 shadow ray) | 0.35 | bandwidth + RT |
| ReSTIR GI (1 guided bounce + NEE + temporal + spatial) | 0.55 | RT (incoherent) |
| Radiance cache query + sparse update + resolve | 0.20 | bandwidth (hash grid) |
| Neural reconstruction (DLSS-RR, 720p→1440p) | 0.50–0.90 | tensor cores; **measured at M4** |
| Tonemap + present (shared image, zero readback) | 0.05 | trivial |
| **Total** | **1.95–2.35** | governor trims internal res to hold 2.0 |

**Reservoir bandwidth math (why RGB-only reservoirs are mandatory):** the RGB `PathReservoir` is 16 floats = 64 B/px (`restir_pt.slang`). 0.92 M px × 64 B × ~6 reservoir reads+writes across temporal/spatial passes ≈ 350 MB ≈ 0.7 ms of DRAM at 504 GB/s — acceptable, and mostly L2-resident at this footprint (59 MB working set vs 48 MB L2). The **spectral** reservoir (192 B/px) triples this and falls out of budget: **realtime mode stores hero-wavelength results resolved to RGB in reservoirs and film** (`RESTIR_RGB_ONLY` already exists as a define); full 32-band spectral film remains the offline mode. This is a real, permanent scope statement of realtime Spectra, not a temporary cut.

### 4.2 Why cost can be pixel-bound: the amortization identity

A converged path-traced frame needs O(100–1000) samples/pixel. The directive's budget buys ~4–6 ray segments/pixel. The architecture closes that gap by making every expensive quantity **a persistent estimator updated incrementally**, so each frame pays O(pixels) update cost into state whose quality is O(frames-since-discontinuity):

1. **ReSTIR temporal reservoirs** carry the light-sample / path posterior across frames — each frame contributes 1 candidate and inherits M-weighted history (effective sample count grows toward the M-cap, ~8–32× variance reduction for ~0 extra rays).
2. **World-space radiance cache** (SHaRC-class spatial hash; `nrc_cache.slang`'s hash grid and the NRC CUDA impl already exist) holds multi-bounce incident radiance keyed by position/normal — deep bounces become one hash lookup; the cache is fed by the sparse update rays and converges across frames. SHaRC is shader-only and vendor-agnostic (works on any DX/Vulkan RT GPU — [NVIDIA-RTX/SHARC](https://github.com/NVIDIA-RTX/SHARC)), so it runs on the dev floor too.
3. **Neural reconstruction** (DLSS Ray Reconstruction) replaces the hand-tuned denoiser+upscaler pair with a single temporal network that consumes the noisy radiance + guide buffers and accumulates detail across frames at 4× pixel ratio.
4. **Hardware traversal + SER** make the per-segment constant small enough that the O(pixels) update fits 2 ms.

Each layer is a *cache with an invalidation story* (§4.7). The frame cost is the sum of fixed-size update passes; nothing in the per-frame loop iterates over scene primitives except AS refit, which is capped and amortized.

### 4.3 Backend matrix — what exists vs what is needed

| | **Vulkan compute** (dev floor, today) | **VK_KHR_ray_tracing / ray query** (portable HW RT) | **OptiX 9 / CUDA** (NVIDIA-max) |
|---|---|---|---|
| Traversal | `bvh_traverse.slang` compute kernel — **exists, is the only wired path** | `rust/spectra-vulkan` exists (ash, real AS build code) but host-batch API + CPU fallback; **needed:** ray-query traversal *inside* the wavefront kernels (Slang `RayQuery` on SPIR-V target), device-resident, replacing the `trace_bvh` dispatch behind the same SoA buffers | `optix/rt` exists (GAS/IAS, SER programs `ser_programs.cu`) but pybind11-facing; `render_state.rs` never constructs it. **Needed:** `optixLaunch` sharing the cudarc device context + SoA buffers; SER on Ada |
| AS build | n/a (CPU BVH upload) | exists in `spectra-vulkan` (build/refit/TLAS); needs wiring to `GpuScene` + per-frame refit budget | exists (`build_gas/refit_gas/build_ias`); needs frame-loop wiring |
| Mega Geometry / CLAS | n/a | `VK_NV_cluster_acceleration_structure` — shipped in production drivers (Windows 572.16+ / Linux 550.144+), all RTX GPUs Turing+ ([NVIDIA](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/)); **nothing exists in-repo** | `optix/clas/clas_builder.cu` is an 8-line stub (`is_available() → false`); needs OptiX 9 cluster API |
| ReSTIR DI/GI/PT kernels | **exist and wired** (`restir_di/gi/pt.slang`, 4-pass each, `kernel_set.rs`) | same kernels recompile to SPIR-V — traversal calls swap to ray query | same kernels via cudarc (already the CUDA path) |
| Radiance cache | NRC query/train kernels + hash grid **exist, wired** (`use_nrc`, `nrc_query_bounce`) | works as-is (compute) — add SHaRC-style resolve pass | NRC CUDA impl exists (`optix/nrc/nrc_impl.cu`, tiny-MLP) |
| Reconstruction | `realtime_denoise.slang` + FSR (`spectra-upscale::fsr`) — **exists**; the floor's reconstructor | FSR (exists) on non-NVIDIA; DLSS-RR via NGX Vulkan on NVIDIA | DLSS-SR **exists** (NGX-CUDA, interop buffers, Halton jitter, pack kernels wired in `renderer.rs`); **DLSS-RR needed** — Streamline `sl.dlss_d`, needs guide buffers (depth, MVs, normal+roughness packed, demodulated albedo, specular hitT), HDR-mandatory ([Streamline guide](https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md)) |
| Frame replay | none (per-dispatch CPU cost) | needed: prebuilt command buffers / one VkCommandBuffer re-submit per frame | CUDA graphs **exist and wired** (`use_cuda_graphs`, capture at sample 1, replay after) |
| Dev-floor status (780M / RADV) | proven (seconds/frame) | RADV exposes `VK_KHR_ray_tracing_pipeline`/ray query on RDNA3, on by default, still gaining (Wave32 RT in Mesa 26.0 — [Phoronix](https://www.phoronix.com/news/RADV-RT-RDNA3-RDNA4-Wave32)); 780M has RDNA3 ray accelerators → **dev parity for correctness, never for ms** | no NVIDIA hardware — OptiX untestable on the floor; all OptiX work is verified on `tomespensin` only |

**Backend decision:** the realtime path is built **VK-ray-query-first** (one Slang source tree, runs on both machines, dev floor keeps parity), with the **OptiX backend as the NVIDIA-max overlay** (SER, OptiX denoiser already built, NRC CUDA, CLAS when OptiX 9 lands) behind the existing `GpuBackend` trait. The empty `use_optix` branch in `renderer.rs:1548` becomes the real seam. Both HW paths must produce hit-parity with the compute oracle (the capabilities gate) — the compute path is permanently retained as the reference and the no-RT-hardware fallback.

### 4.4 The realtime frame graph and the governor

Per frame (after M2): `as_refit(capped) → primary(RT or splat G-buffer) → restir_di(gen/temporal/spatial/shade) → restir_gi(1 guided bounce, temporal/spatial) → cache_query(bounce-1 termination) + sparse cache_update → resolve → reconstruct(RR) → present(shared image)`. All passes pre-recorded (CUDA graph / reused command buffer); per-frame CPU work is uniform patching + one submit. The **first-frame compile gate** (`kernel_set.rs::ensure_compiled`, already shipped) moves to load time: realtime mode precompiles every kernel of the frame graph before the first frame and caches SPIR-V/PTX (`kernel_cache.rs` exists) — a frame is never allowed to wait on `slangc`.

**Governor:** a proportional controller on the GPU-timestamp total (the virtualized-splat design §4.7 pattern, same code shape): `internal_scale ← clamp(internal_scale · (1.9 / ema_ms)^0.5, 0.33, 0.67)`, plus discrete sheds in priority order (spatial-reuse pass count → GI rays to ½ rate checkerboard → DI candidates). The governor changes *quality knobs only*; it never lengthens the frame. Its state and the per-pass ms are printed by the probe and shown in the HUD line `spectra-rt T ms @ WxH`.

### 4.5 Milestones — the amortization stack, each independently measurable

Until CI exists on `tomespensin`, every "4070 Ti" Done When is verified **manually at that keyboard** (RemoteTrigger/RDP session), and the printed lines are pasted into the milestone PR. The dev-floor variant runs in local CI. Both variants print the same table format.

**M0 — Truth harness (`spectra_realtime_probe`).** A binary that drives `spectra-renderer` frame-by-frame (no spp loop) over named scenes (`cornell`, `city_block`, `city_block_2x`), records per-pass GPU timestamps, prints the §2 table, and dumps `probe_<scene>_<gpu>.json`. No optimization work.
*Done When:* on the 780M, `cargo run --release --bin spectra_realtime_probe -- --scene cornell --frames 120` prints the full pass table with nonzero ms and `backend=vk-compute`; on `tomespensin` the same command prints the table with `gpu=NVIDIA GeForce RTX 4070 Ti`. No gate yet — the printed baseline *is* the deliverable.
*Budget table:* none — M0 establishes it.

**M1 — Hardware traversal in the wavefront.** Slang ray-query traversal entry points (`trace_bvh_rq`, `trace_shadow_rq`) over a device TLAS, dispatched behind the same SoA ray buffers; AS built from `GpuScene` at load; `spectra-vulkan`'s build code wired device-side (no host-slice trace API anywhere in the frame). OptiX variant on `tomespensin` via `optixLaunch` over shared cudarc buffers.
*Done When:* probe on `tomespensin` prints `traversal: sw=<A> ms hw=<B> ms ratio=<A/B>` with **ratio ≥ 5.0** on `city_block`, and `hit parity: 100000 rays, mismatches=<m>` with **m < 100**; on the 780M the same lines print (any ratio — RADV RT on 12 CUs may be modest) with the same parity gate.
*Budget targets after M1 (4070 Ti):* primary 0.20 ms, DI shadow 0.15 ms, GI segment 0.40 ms — printed, not asserted yet.

**M2 — The fixed-budget frame.** Per-frame API (`render_realtime_frame`, §6): 1 candidate ReSTIR DI + 1-bounce ReSTIR GI, RGB reservoirs persisted across frames (temporal reuse on), no convergence loop, prerecorded dispatch (CUDA graph / reused command buffer), and **zero-copy present** — render into a shared Vulkan image consumed by the window/engine; the `Arc<Vec<u8>>` seam in `vox_render::splat_backend` becomes debug-only.
*Done When:* `tomespensin` probe prints `restir_di+restir_gi+primary total p50 ≤ 1.20 ms` at 1280×720 internal on `city_block`, `present=shared-image readback=0 B/frame`, and the variance gate from §3 (`mse_on < 0.25 × mse_off`). Dev floor: same lines, no ms gate, reuse-variance gate enforced.
*Budget table (1280×720, 4070 Ti):* primary 0.20 │ DI 0.35 │ GI 0.55 │ resolve 0.10 │ **subtotal 1.20**.

**M3 — Cache-terminated deep bounces.** Bounce-1 hits terminate into the radiance cache (SHaRC-style spatial-hash resolve pass added to the existing NRC hash grid; both query paths kept, flag-selected); 1/16-pixel sparse update rays feed it; MNEE/photon kernels (exist) run *inside the sparse update budget* to inject caustic energy into the cache over frames, never per-pixel per-frame.
*Done When:* probe prints `gi mode=cache flip@120f=<X>` with **X < 0.10** vs the traced-4-bounce reference on `cornell` and `city_block`, and the cache pass row prints **≤ 0.20 ms** on `tomespensin` (query+update+resolve). Dev floor: same FLIP gate, ms printed.
*Budget table adds:* cache 0.20 │ **subtotal 1.40**.

**M4 — Neural reconstruction (DLSS-RR) + floor fallback.** Streamline `sl.dlss_d` on Windows: tag depth, motion vectors (pack kernels exist), packed normal+roughness, demodulated albedo, specular hit-distance; HDR pipeline (mandatory per the RR guide). Internal 1280×720 → 1440p output. Dev floor fallback: `realtime_denoise.slang` + FSR (both exist) behind the same `Upscaler` trait (`spectra-upscale` already abstracts DLSS/FSR/Null).
*Done When:* `tomespensin` probe prints `reconstruct=dlss-rr ms=<F>` with **F ≤ 1.00** and full-stack `total p50 ≤ 2.00 / p99 ≤ 2.40` at 2560×1440 output — **this is the directive gate**, a human sees the orbit at 1440p with lighting settling ≤ 1 s after cuts. Floor: `reconstruct=fsr+rt-denoise`, ms printed, no gate.
*Budget table adds:* reconstruct 0.50–0.90 │ tonemap/present 0.05 │ **total 1.95–2.35, governor holds 2.00**.

**M5 — Scene dynamism under budget (Mega Geometry).** Capped AS-update queue (refit preferred, rebuild queued across frames); CLAS via OptiX 9 cluster API (replacing the 8-line stub) or `VK_NV_cluster_acceleration_structure` for per-cluster build of animated/destructible geometry and splat-proxy clusters.
*Done When:* probe `--scene city_block --animate 2000` on `tomespensin` prints `as_update p50 ≤ 0.10 ms p99 ≤ 0.30 ms` with 2,000 instances animating, total still ≤ 2.00/2.40, and geometry lag (frames between transform change and visible AS update) printed and ≤ 2.
*Budget table adds:* as_update 0.10 │ unchanged total.

**M6 — Governor + scene-independence certification.** The controller of §4.4; the `--scene-scaling-check` mode; the cut-recovery probe (§3); 30-minute soak on `tomespensin` printing p50/p99/p99.9.
*Done When:* the full §2 headline holds, including the < 10% scene-doubling delta and the cut-recovery curve gates, on `tomespensin`, manually verified and pasted into the PR.

### 4.6 Windows build reality check (`tomespensin`)

What the stack needs on Windows, component by component — with what breaks today:

- **wgpu / ochroma engine:** builds on Windows (DX12/Vulkan). No issue.
- **slangc + libslang:** Slang ships official `windows-x64` releases. **Breaks today:** `spectra-gpu`'s build scripts and the Vulkan backend dlopen `libslang.so` and shell out to `slangc` with Unix assumptions; `scripts/build-spectra-native.sh` is bash + `LD_LIBRARY_PATH`. Work: load `slang.dll`, search `%SLANG_DIR%\bin`, and a `build-spectra-native.ps1` twin. (The runtime-compile dependency is also why the M2 load-time precompile + SPIR-V cache matters — `slangc.exe` must never run mid-frame.)
- **cudarc:** supports Windows via runtime loading of `nvcuda.dll` etc. CUDA Toolkit install on `tomespensin` required for nvcc-built parts only.
- **OptiX:** SDK installs on Windows; runtime ships in the driver (`nvoptix.dll`). **Breaks today:** `optix/` builds via `build.sh` + CMake with pybind11 targets, and `optix/lib/` vendors Linux driver libs (`libnvidia-rtcore.so`, `nvoptix.bin`) that must not ship and have no Windows analogue. Work: a Rust-FFI-only CMake preset (MSVC + nvcc), drop vendored driver libs, kill the pybind11 dependency from the realtime path.
- **DLSS-SR (existing NGX-CUDA path):** **breaks today** — `spectra-dlss/detect.rs` searches only Linux/WSL2 `.so` paths. On Windows the NGX snippets are `nvngx_dlss.dll`/`nvngx_dlssd.dll` next to the executable + driver-side core. Work: Windows search paths + the NGX Win32 init variant.
- **DLSS-RR:** integrate via **Streamline** (`sl.dlss_d`), which is Windows-only (x64, D3D11/12/Vulkan) — fine, because RR is only ever exercised on the Windows target; the floor uses FSR. Streamline interposer ships as DLLs; needs HDR color, depth-only aspect mask on the Vulkan depth view (documented Vulkan gotcha), and the guide buffers listed in M4.
- **CUDA↔Vulkan interop:** existing interop (`spectra-vk-interop`, DLSS interop buffers in `renderer.rs`) uses POSIX fd external memory (`OPAQUE_FD`). **Breaks on Windows:** must add `OPAQUE_WIN32` handle types on both the ash and CUDA sides.
- **Driver floor:** Mega Geometry Vulkan extension needs driver ≥ 572.16 on Windows — pin and print the driver version in the probe header.
- **CI:** none on `tomespensin`. Until a runner exists, milestone verification there is manual (§4.5 preamble); a scheduled self-hosted runner is tracked as an open question, not assumed.

### 4.7 Honest physics — what 1–2 ms can never contain, and the discontinuity contract

**Never in a single frame, on any hardware:**

- **Convergence.** A 2 ms frame contains ~4–6 traced segments/pixel; a clean path-traced still wants hundreds. The realtime image is *always* an estimator state, not a result. The honest formulation shipped in this design: **image quality is a function of (scene, time since last discontinuity), and milliseconds are constant.** At rest, DI is effectively converged within ~0.1 s (reservoir M-cap), GI within ~0.3–0.5 s (cache + reservoir + RR accumulation), deep/caustic energy within ~0.5–2 s (cache warmup). These envelopes are *measured by the probe*, printed, and gated at M6 — not promised.
- **Caustic completeness without warmup.** MNEE/photon work runs inside the sparse cache-update budget; caustics fade in as the cache warms. SDS paths remain approximate forever in realtime mode; the offline integrator remains ground truth.
- **Per-frame full-spectral transport.** 32-band reservoirs/film exceed the bandwidth budget by ~3× (§4.1). Realtime is hero-wavelength → RGB; spectral output stays offline.
- **O(1) traversal.** HW traversal is ~O(log n) with cache effects; "no matter what scene" is honored by the governor holding ms constant while constants drift — *quality floats, milliseconds don't*. The scene-doubling gate (< 10% p50 delta) is the enforceable form of the directive's "no matter what scene".

**Teleports, cuts, disocclusion — the invalidation cost curve.** Every amortization layer is a cache; a discontinuity is a (partial) flush. The contract under a cut: **the frame budget does not move; quality absorbs the hit on a bounded curve.**

| Layer | What invalidates on a camera cut | Frame-0 behavior | Recovery |
|---|---|---|---|
| ReSTIR temporal reservoirs | screen-space history (M→0) | DI = candidate-only (noisy but unbiased); GI = 1-sample | M rebuilds geometrically; usable DI ≈ frame 0, settled ≈ 5–10 frames |
| Spatial reuse | nothing (same-frame) | full effect immediately | — |
| Radiance cache | **nothing for same-scene cuts** — keyed in world space, survives any camera motion; only *scene/lighting* changes invalidate touched cells | hit-rate dips only for newly seen regions | newly seen cells fill at the sparse-update rate: ~0.25 M paths/frame → city-block working set warm in ~0.5–2 s; stale cells decay by exponential blend |
| DLSS-RR history | screen-space history | single-frame reconstruction: soft, no ghosting (reset flag tagged on cut) | sharpens over ~10–30 frames |
| Acceleration structure | nothing (camera-independent) | — | — |

Disocclusion is the same table applied per-pixel. Scene *mutation* (destruction) additionally invalidates AS (budgeted queue, ≤ 2-frame lag, M5) and the touched cache cells. Known-destination teleports may pre-warm the cache by pointing the sparse-update rays at the destination for N frames before the cut — an optimization, not a contract requirement. The cut-recovery probe (§3) makes this entire table a printed, regression-gated curve.

### 4.8 Relationship to the virtualized splat path — replace, coexist, or merge

The honest answer is **staged convergence with a shared spine, ending in replacement of the *lit image* on RTX-class hardware — and permanent coexistence as the floor.** Precisely:

- **Today (and all of Phase 1/2):** coexist, unequal. The wgpu virtualized splat rasterizer ([its design](./2026-06-10-virtualized-splat-rendering-design.md)) is *the game's frame* — budget-bounded, runs on the 780M; Spectra is stills. Nothing in this design touches that. The two already share doctrine: fixed budget by construction, governor on measured ms, quality sheds before milliseconds.
- **M2–M4:** Spectra-realtime runs **on the same scene the splat path renders** (the instance/atom library is the single source; Spectra's triangle scene is derived from it — today via the lossy quad bridge in `splat_convert.rs`, later via mesh/cluster proxies and native splat primitives: `splat_intersect.slang` ellipsoid intersection and `gaussian_volumetric.slang` delta-tracking already exist for the RT path). On RTX hardware Spectra-realtime becomes the **hero-quality mode** toggle of the same window, consuming the splat raster's G-buffer for primary visibility where that wins the ledger.
- **End state on RTX-class (post-M6): replace the lighting, keep the rasterizer as the visibility engine.** The splat rasterizer remains the primary-visibility pass (it is already pixel-budget-bounded and splat-native); Spectra-realtime supplies DI/GI/reflections/shadows per pixel inside its 2 ms; the wgpu shading path (`ResidentGiRaster` et al.) is bypassed on this tier. Full RT primary visibility of native splats (no rasterizer at all) is a *measured later decision* — it requires custom intersection (ellipsoids) which forfeits some ray-query simplicity, and it must beat the raster G-buffer's ~0 ms marginal cost to justify itself.
- **Permanently:** on hardware without usable RT (and on the 780M as dev floor), the virtualized splat path **is** the product. Spectra-realtime never becomes a minimum requirement.
- **Merge obligations starting now:** one governor codebase (the splat design's §4.7 controller and this design's §4.4 controller must be the same type), one scene/instance source, one HUD ms line format, and the shared-image present (M2) is the same surface the splat path composites into.

### 4.9 Kill criteria

Each is a measured tripwire with a named decision, evaluated at its milestone — not at the end:

1. **M1:** HW/SW traversal ratio < 5× on the 4070 Ti on `city_block` → the premise that RT cores buy the budget is false for this workload; stop, re-scope the directive to a 16–33 ms hero mode and say so.
2. **M2:** trace+resample subtotal > 1.2 ms at 1280×720 after SER + graph replay tuning → 2.0 ms total is unreachable on this card; renegotiate to a 3–4 ms tier *before* building M3+ (the stack above M2 cannot claw back trace time).
3. **M3:** cache-terminated GI fails FLIP < 0.10 after 120 frames on the named scenes → the deep-bounce amortization is dishonest; fall back to traced 2-bounce at reduced internal res and re-run the M2 gate; if that also fails, criterion 2 applies.
4. **M4:** DLSS-RR > 1.2 ms at 1440p on the 4070 Ti, or RR guide-buffer cost pushes total > 2.4 p99 → drop output to 1080p or ship FSR+`realtime_denoise` on Windows too; if neither holds 2.0/2.4, criterion 2 applies.
5. **M6:** scene-doubling delta ≥ 10% p50 that the governor cannot remove without dropping below the 0.33 internal-scale floor → the "no matter what scene" clause is not met; the feature ships as "fixed budget at bounded scene class" with the bound printed, or not at all.
6. **Any point:** the realtime image at 60-frames-stationary differs from the offline reference by FLIP > 0.15 on any named scene → quality amortization is failing its honesty bar; the mode does not ship while the gate fails. No marketing exception exists for this criterion.

---

## 5. Data Models

```rust
/// Fixed per-frame GPU budget contract. Immutable per session; the governor
/// moves `internal_scale` within [min_scale, max_scale], never the ms numbers.
pub struct RealtimeBudget {
    total_ms_p50: f32,        // 2.0 on tomespensin @1440p
    total_ms_p99: f32,        // 2.4
    output_extent: (u32, u32),
    min_scale: f32,           // 0.33 (UltraPerformance)
    max_scale: f32,           // 0.67 (Quality)
}
impl RealtimeBudget {
    pub fn contract_1440p_4070ti() -> Self;
    pub fn total_ms_p50(&self) -> f32;
    pub fn internal_extent(&self, scale: f32) -> (u32, u32);
}

/// Per-pass GPU-timestamp report for one frame. Everything the probe prints.
pub struct FrameReport {
    pass_ms: [f32; PASS_COUNT],   // indexed by RealtimePass enum
    total_ms: f32,
    internal_extent: (u32, u32),
    traversal: TraversalBackend,
    readback_bytes: u64,          // MUST be 0 after M2
    governor_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimePass { AsUpdate, Primary, RestirDi, RestirGi, Cache, Reconstruct, Present }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalBackend {
    ComputeBvh,        // bvh_traverse.slang — reference + no-RT fallback, permanent
    VulkanRayQuery,    // portable HW RT (RADV 780M + NVIDIA)
    OptixSer,          // NVIDIA-max: optixLaunch + SER (+ CLAS after M5)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructBackend { DlssRr, FsrPlusRtDenoise, None }
```

GPU layout constraints: RGB `PathReservoir` stays 16 floats / 64 B (`restir_pt.slang` layout is frozen for realtime; the spectral 48-float variant is compile-gated out via `RESTIR_RGB_ONLY`). Radiance-cache entry ≤ 32 B (hash key + RGB radiance + sample count + age), capacity fixed at session start (default 2^22 entries = 128 MB ceiling).

---

## 6. API

```rust
// rust/spectra-renderer — additive; render()/spp-loop API unchanged for offline.
impl<G: GpuBackend> Renderer<G> {
    /// Enter realtime mode: precompiles ALL frame-graph kernels (hard gate —
    /// refuses to return Ok until every kernel in the realtime set compiled;
    /// reuses KernelSet::ensure_compiled), builds the AS, allocates persistent
    /// reservoir/cache/history buffers, records the dispatch graph.
    /// Errors: kernel compile failure (named kernel + GpuError), AS build failure,
    /// missing RT support when `traversal != ComputeBvh`.
    pub fn begin_realtime(&mut self, budget: RealtimeBudget,
                          traversal: TraversalBackend,
                          reconstruct: ReconstructBackend) -> Result<(), RealtimeError>;

    /// Render exactly one frame within the budget. Never loops to convergence.
    /// Persistent state (reservoirs, cache, RR history) carries across calls.
    /// `camera_cut: true` tags history-reset to reservoirs + RR (cache untouched).
    /// Returns the GPU-timestamped report for the PREVIOUS frame (1-frame
    /// latency on timestamps; values are real, never estimated).
    pub fn render_realtime_frame(&mut self, view: &CameraLayer, camera_cut: bool)
        -> Result<FrameReport, RealtimeError>;

    /// Shared-image output handle for zero-copy present/composite (M2+).
    /// Vulkan: VkImage + layout/queue-family contract; CUDA backend exports
    /// via external memory (OPAQUE_FD on Linux, OPAQUE_WIN32 on Windows).
    pub fn realtime_output_image(&self) -> Result<SharedImageHandle, RealtimeError>;
}
// Threading: all three are &mut self on the render thread that owns the GpuBackend
// (same single-owner model as today's Renderer). FrameReport is Send.

// Existing signatures relied on (verbatim — implementations must match, do not re-derive):
// Renderer::new(gpu: G, config: RenderConfig) -> Self
// KernelSet::ensure_compiled(&mut self, names: &[&str], gpu: &mut G)
//     -> Result<(), (String, GpuError)>          // the shipped first-frame gate
// fn dlss_jitter(frame_idx: u32) -> (f32, f32)   // Halton(2,3), [-0.5,0.5] (renderer.rs:2657)
// vox_render::splat_backend::SpectraRenderBackend::{realtime(w,h), submit_frame(..),
//     read_last_output() -> Arc<Vec<u8>>}        // becomes debug-only after M2
// spectra_upscale::{UpscaleQuality, UpscalerPreference, Upscaler trait}
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `spectra_realtime_probe` (M0) | new bin target | `rust/spectra-bin/src/bin/spectra_realtime_probe.rs` | the truth harness; CI on floor, manual on tomespensin |
| Ray-query traversal kernels (M1) | wavefront dispatch replacing `trace_bvh`/`trace_shadow` | `slang/bvh_traverse_rq.slang`; seam at `renderer.rs:1548` (today's empty `use_optix` block) | compute oracle retained behind `TraversalBackend::ComputeBvh` |
| AS build/refit | `begin_realtime` + capped per-frame queue | `rust/spectra-vulkan/src/traversal.rs` (device-side rework), `rust/spectra-optix/src/traversal.rs` | host-slice trace APIs deleted from the frame path |
| `begin_realtime` / `render_realtime_frame` | engine: `SpectraRenderBackend` render thread (replacing the per-still `render()` call) | `rust/spectra-renderer/src/renderer.rs`; `crates/vox_render/src/splat_backend.rs` | one frame of latency stays; pixel readback does not |
| Shared-image present (M2) | engine composite / probe window | `rust/spectra-vk-interop`; `crates/vox_render/src/splat_backend.rs` | `readback_bytes=0` printed every frame |
| SHaRC-style cache resolve (M3) | frame graph between GI and resolve | `slang/nrc_cache.slang` (extend) + new `slang/sharc_resolve.slang` | vendor-agnostic; runs on floor |
| Streamline DLSS-RR (M4) | `ReconstructBackend::DlssRr` in the frame graph | new `rust/spectra-streamline` crate; guide-buffer packing extends `pack_rgba.slang`/`pack_motion_vectors.slang` | Windows-only; floor uses `spectra-upscale::fsr` + `realtime_denoise.slang` |
| CLAS / Mega Geometry (M5) | AS-update queue | `optix/clas/clas_builder.cu` (replace stub), or `VK_NV_cluster_acceleration_structure` in `spectra-vulkan` | driver ≥ 572.16 printed in probe header |
| Governor (M6) | wraps `render_realtime_frame` budget knobs | shared type with the splat path's controller — `crates/vox_render/src/frame_governor.rs` (new, used by both) | §4.8 merge obligation |
| HUD line | game/probe overlay | probe window + `urban_horizon` HUD | same format as splat path: `spectra-rt T ms @ WxH` |

---

## 8. Open Questions

- [ ] **Slang `RayQuery` → SPIR-V on RADV:** verified-by-build that the Slang version pinned in-repo emits valid `SPV_KHR_ray_query` consumable by RADV on gfx1103 — M1's first task; if Slang's SPIR-V ray-query path is broken on the pinned version, the floor's M1 variant compiles the same kernels to GLSL-equivalent via a thin wrapper, decided then.
- [ ] **Primary visibility source on the merged path (§4.8):** RT primary vs splat-raster G-buffer — decided by the M2 ledger (whichever wins ms at equal output), not by preference.
- [ ] **DLSS-RR via raw NGX Vulkan vs Streamline:** Streamline is the documented surface but adds an interposer; raw NGX (`nvngx_dlssd`) is lighter but worse-documented for RR. Decide at M4 start with a 1-day spike on `tomespensin`.
- [ ] **Reservoir M-caps and spatial-pass counts** at the contract point (variance vs bandwidth) — swept by the probe at M2, values frozen into `RealtimeBudget` defaults.
- [ ] **CI runner on `tomespensin`** (self-hosted, scheduled): owner and timeline — until then the manual protocol of §4.5 stands.
- [ ] **Native splat RT primitives** (ellipsoid custom intersection vs mesh proxies) — measured decision after M5, per §4.8.

---

## 9. Out of Scope

- Any change to the offline/cinematic spectral integrator, the spp loop, 32-band spectral film, LPE/cryptomatte/deep output — the stills path is untouched and remains ground truth.
- Frame generation (DLSS-FG / `sl.dlss_g`) — output-side frame multiplication is orthogonal to the render budget and adds latency; revisit only after M6 holds.
- 4K@2 ms on the 4070 Ti — excluded by arithmetic (§4.1); 4K is a 3 ms tier or Ultra Performance ratio, stated now.
- Multi-GPU, non-NVIDIA HW-RT tuning beyond correctness (RADV is a parity floor, not a perf target), macOS/MetalFX.
- The virtualized splat path's own milestones (M1–M4 of [that design](./2026-06-10-virtualized-splat-rendering-design.md)) — only the merge obligations in §4.8 bind the two.
- Gameplay/sim consumers of spectral probes (`spectral_probe_gather` stays on the offline/interactive path until a realtime consumer is designed).

---

## 10. Related Plans / Designs

- Depends on: [Design: Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (shared scene + governor spine), [SOTA City Block Phase 1 plan](../plans/2026-06-10-sota-city-block-phase1.md) Tasks 3–8 (stills-path hardening this realtime mode forks from)
- Required before: the per-milestone implementation plans (one plan per M0–M6, each using `docs/templates/plan.md`)
- Related: `spectra-pathtracer-runs` memory (dev-floor run recipe, kernel-dir fix), `spectra-crucible-deps` memory (Slang SDK gotchas), AAA Spec 05/11 (resident-frame doctrine the governor reuses)
- External references (facts verified 2026-06-10): [RTX Mega Geometry Vulkan samples + driver floor](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/), [VK_NV_cluster_acceleration_structure proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cluster_acceleration_structure.html), [Streamline DLSS-RR programming guide](https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md), [NVIDIA-RTX/SHARC integration guide](https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md), [RTXGI 2.0 (NRC + SHaRC)](https://github.com/NVIDIA-RTX/RTXGI), [CP2077 path-tracing ray/bounce defaults](https://www.nexusmods.com/cyberpunk2077/articles/980), [RADV RDNA3 RT, Mesa 26.0 Wave32](https://www.phoronix.com/news/RADV-RT-RDNA3-RDNA4-Wave32)
