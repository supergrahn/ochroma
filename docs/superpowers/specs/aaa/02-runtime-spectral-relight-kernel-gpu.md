> **Adversarial verification:** Verdict flag `sound=false`, but the recorded issue is "GROUNDING REFS — ALL VERIFIED": the skeptic checked every named symbol/file/line in the grounding list and found each exists at (or within 0 lines of) the claimed location — `relight.rs` `f16_max:42`, `AMBIENT_FILL_WEIGHT:50`, `IlluminantSpec::spd:134`, `RelightSettings::new:247`, `with_sky_ambient:264`, `with_shadows:268`, `derive_intrinsic:349`, etc. The `sound=false` mark reflects that the skeptic could not certify beyond the grounding refs (the truncation/coverage limit of the audit), NOT a discovered defect in the design. See **Verification corrections** below for the precise scope of what was and was not certified.

＃ Design: Runtime spectral relight kernel (GPU) — the asset-time pass becomes a per-frame light-swap (2026-06-07)

**Status:** Draft
**Scope:** Port the verified CPU relight oracle (`vox_render::relight::reilluminate_one` + `derive_intrinsic`, driven by `relight_scene`) into a single WGSL compute pass that re-illuminates the on-GPU splat radiance buffer per dispatch given a 16-float target illuminant SPD — proven bit-for-bit against `relight_scene` on RADV. This turns Ochroma's crown-jewel wedge mechanic (spectral relight / metamerism) from an offline `vox_tools relight` batch op into a runtime light-swap.
**Related:** `2026-06-07-aaa-capability-roadmap.md` (gap #5, Wedge dim), `2026-06-07-spectral-relight-design.md`, `01-*` (gap #2 `GpuContext` — hard dependency), `34-*` (reflectance/emission data-model split — pairs to make this physically exact).

> Roadmap-honesty preface (binding): the roadmap's first-slice is a SEED, verified against the real code here. Two grounded corrections to the seed: (1) `GaussianSplat` stores `spectral:[u16;16]` as **baked f16 radiance** with no reflectance field (`crates/vox_core/src/types.rs:40`) — so the GPU pass must derive the per-splat intrinsic on-device (or cache it once at load) exactly as the CPU oracle does, it cannot read a reflectance channel that does not exist. (2) The CPU oracle's full path includes a sky-ambient fill term and (for `Sun` illuminants) `n_dot_l` + BVH shadow rays; the first GPU slice mirrors the **ambient-only, no-shadow** configuration (`with_sky_ambient`/`with_shadows` knobs already on `RelightSettings`), which is the exact configuration `relight_tungsten_to_daylight_is_bluer` uses — shadows are a later slice, not slice 1.

---

## 1. What we need

The capability: **a captured world's baked illumination can be swapped at frame rate on the GPU.** Today `relight_scene()` (`crates/vox_render/src/relight.rs:466`) is proven, metamer-validated, and shadow-aware — but it is a CPU rayon batch that lives only behind `vox_tools relight`; `grep relight_scene` across `vox_app`, `render_ecs.rs`, and `lib.rs` returns **zero frame-loop callers** (verified). No WGSL twin exists. After this lands:

- A developer can bind a 16-float target-SPD uniform and dispatch one compute pass over the on-GPU splat buffer to re-illuminate every splat — `tungsten → daylight`, `→ cool_led`, `→ sun@14` — with **no CPU rayon pass and no full-scene re-bake clone**. (Observable: `cargo test gpu_relight_matches_cpu_oracle` prints a per-band max deviation and a dispatch time.)
- The GPU result is **bit-for-bit faithful to `relight_scene`** to `<1e-6` per band across N illuminants on RADV, joining the existing 5-twin provability bar (GI 0, atom-budget exact, hybrid 1.79e-7, many-light exact, splat-RT 1.19e-7). (Observable: the test asserts `max_band_dev < 1e-6` against the CPU oracle, not `is_some()`.)
- The per-splat **intrinsic base is cached once at load** (`baked ÷ capture-SPD`, the `derive_intrinsic` op) so a light-swap is a per-frame multiply, not a per-frame divide — making repeated relights (a day/night slider, a light-probe sweep) cheap. (Observable: the dispatch path reads a persistent `intrinsic` buffer, never re-divides.)
- The swap is **a runtime mechanic, not an export step**: an `OCHROMA_RELIGHT=gpu` lever (mirroring the shipped `OCHROMA_GI=gpu` pattern at `crates/ochroma_engine/src/engine_loop.rs:226`) routes the engine's relight through the GPU pass with a permanent graceful CPU fallback. (Observable: a headless run logs which backend executed and the relit splats differ from frame-0 baked.)
- **Metamerism survives the GPU round-trip**: two splats that are metamers under one illuminant diverge under another (the structurally-impossible-in-RGB property `relight_breaks_metamers` proves on CPU) — verified to still hold after the GPU pass + f16 store. (Observable: a metamer-pair test asserts neutral sRGB distance ≈ 0 and post-relight cool_led distance > 0.03 on the GPU output.)

**Why it is blocking (Wedge dimension).** The roadmap names this the unforgeable moat: "Ochroma's AAA is defined by spectral relighting / metamerism as a frame-rate game mechanic… everything else is in service of it" (§1.1). The decisive gap is that the wedge mechanics are **asset-time, not runtime** (§2 Wedge, §1 synthesis: "making them runtime is the strategic spine of this roadmap"). A relight that only runs in a CLI export cannot be a game mechanic — a world captured under one light cannot be *played* under another until this kernel exists. It sits on the Phase-3 critical-path spine (`#5 → #34`).

---

## 2. Done When

Running:

```
cargo test -p vox_render gpu_relight_matches_cpu_oracle -- --nocapture
```

prints a line of the exact form

```
[gpu_relight] illuminants=4 splats=100000 max|Δ/band|=<value below 1e-6> dispatch=<value below 2.0> ms (RADV)
```

and the test **passes** because it asserts `max_band_dev < 1e-6` against `relight_scene()`'s output across {tungsten→daylight, →cool_led, →neutral, →d65} and `dispatch_ms < 2.0` for the 100k-splat scene. A human at the keyboard reads the printed deviation and ms and confirms the GPU twin matches the oracle within tolerance and inside budget. (On a box with no adapter the test prints `[gpu_relight] no adapter — skipping` and returns, mirroring the GI twin's `try_gpu` graceful skip at `spectral_gi.rs:896`.)

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| GPU intrinsic derive matches `derive_intrinsic` | `assert!((gpu_intrinsic[b] - derive_intrinsic(&baked,&ref_spd,floor)[b]).abs() < 1e-6)` per band over 100k splats | `assert!(buf.is_some())` — passes with an unbound buffer |
| GPU `reilluminate` matches `reilluminate_one` (ambient-only) | `assert!(max_band_dev < 1e-6)` vs `relight_scene` AFTER radiance, `with_sky_ambient(true).with_shadows(false)`, across 4 illuminants | function exists, returns the input unchanged |
| f16 store mirrors CPU `encode_radiance` (no inf, NaN→0) | `assert_eq!(gpu_stored_b4, f16_max())` for the bright-band clamp scene (`relight.rs:1011` ported to GPU) | `assert!(stored.is_finite())` with all-zero output |
| Dispatch under budget | `assert!(dispatch_ms < 2.0)` from a wall-clock `Instant` around submit+poll for 100k splats on RADV, printed | no timing measured; prints a hardcoded "fast" |
| Metamerism survives GPU + f16 | `assert!(led_dist > 0.03 && neutral_dist < 0.012)` on `forward_rgb` of GPU-relit metamer pair | `assert!(led_dist != neutral_dist)` — trivially true |
| Graceful no-GPU + no-panic | `matches!(GpuRelight::new(64, impossible_limits), Err(GpuRelightError::DeviceCreation(_)))` | `assert!(result.is_err())` without forcing the failure path |

---

## 4. Architecture

### 4.1 Where it lives and why

A new module **`crates/vox_render/src/gpu/relight_gpu.rs`** + **`crates/vox_render/src/gpu/relight_gpu.wgsl`**, registered in `crates/vox_render/src/gpu/mod.rs` alongside `atom_budget_gpu`, `many_light_gpu`, `hybrid_compose_gpu`, `splat_rt_gpu`. This is `vox_render` (an ENGINE crate) and the relight oracle it mirrors already lives there (`vox_render/src/relight.rs`) — game-agnostic by construction (it operates on `GaussianSplat` + SPD arrays, no buildings/zoning). The high-level driver is the **6th GPU twin**, following the `GpuGi` shape exactly (`spectral_gi.rs:471`): owns-its-device for headless validation **and** a `new_with_context(&GpuContext, …)` constructor for the shared-device frame loop (gap #2's `GpuContext{device:Arc<Device>,queue:Arc<Queue>}` — which does NOT yet exist; this is the hard dependency).

### 4.2 The kernel — a faithful port of the verified CPU op chain

The CPU oracle's per-splat inner loop (`relight.rs:553-630`, ambient-only/no-shadow config) is exactly:

```
intrinsic[b] = baked[b] / max(ref_spd[b], floor)          // derive_intrinsic
incident[b]  = target_spd[b] * (n_dot_l * shadow)         //   n_dot_l=1, shadow=1 ambient-only
             + ambient[b]                                  // ambient = sky_ambient_spd[b] * 0.5
out[b]       = clamp(intrinsic[b] * incident[b]            // reilluminate_one
                     + emitter_gather[b], 0, f16_max)      //   emitter_gather=0 (no new emitters)
stored[b]    = f16::from_f32(nan? 0 : clamp(out[b],0,max)) // encode_radiance
```

The WGSL reproduces this **op-for-op, same order, no fast-math** — the discipline that gives the existing twins their `1e-7` agreement (`atom_budget_gpu`: "identical f32 op order + no fast-math → scores are ULP-tight on RADV"). The two load-bearing constants are bound in a uniform, computed host-side by the SAME functions the CPU path calls so they cannot drift:
- `ref_spd[16]` / `target_spd[16]` from `IlluminantSpec::spd()` (`relight.rs:134`).
- `ambient[16]` = `SpectralAtmosphere::earth().solar_irradiance()` × `AMBIENT_FILL_WEIGHT(0.5)` (`relight.rs:50,519,609`), bound pre-multiplied so the shader does one add.
- `floor` (default `1e-3`, `relight.rs:256`) and `f16_max` (`half::f16::MAX.to_f32()`, `relight.rs:42`) as scalars.

The f16 encode is reproduced in WGSL via `pack2x16float`/`unpack2x16float` (or an explicit clamp-to-`f16_max` then `f16` round-trip) so the GPU's stored value bit-matches CPU `encode_radiance` (`relight.rs:418`) — including the NaN→0 and over-max→`f16_max` saturation that `relight_bright_band_clamps_to_f16_max_not_inf` (`relight.rs:1011`) pins.

### 4.3 Intrinsic caching at load (the per-frame win)

`derive_intrinsic` is a per-splat **divide** that depends only on `baked` and `ref_spd` — both fixed once the asset and its capture illuminant are known. So we compute it **once at load** into a persistent `intrinsic` storage buffer (`array<array<f32,16>>`). A per-frame light-swap then binds only the new `target_spd` uniform and dispatches the multiply-add — the divide never re-runs. This is the data-flow that makes a day/night slider or a light-probe sweep cheap, and it is the seed's "cache per-splat intrinsic once at load (baked ÷ capture-SPD)" made concrete. For the bit-exact headless test we also expose a one-shot path that derives intrinsic in the same dispatch, so the test can prove the whole chain against `relight_scene` without a separate load step.

### 4.4 Data flow

```
LOAD (once):
  baked f16[16] (splat.spectral) ──decode──▶ baked f32[16]
                                  ÷ max(ref_spd, floor)  ──▶  intrinsic buf  (PERSISTENT GPU storage)

PER-FRAME light-swap:
  target_spd[16] ─┐
  ambient[16]    ─┼─▶ RelightParams uniform ─┐
  floor, f16_max ─┘                           │
  intrinsic buf ──────────────────────────────┼──▶ [relight_gpu.wgsl @workgroup_size(64)]
                                               │        out[b]=clamp(intrinsic[b]*incident[b],0,max)
                                               ▼        store=f16(out[b])
                                        radiance OUT buf (storage, read_write)
                                               │
                       (headless test only)    └─copy─▶ readback ─poll─▶ compare vs relight_scene()
                       (frame loop, gap #8)     └──────▶ bound directly as rasterizer splat input
```

In the frame loop the OUT buffer is the rasterizer's splat-radiance input (resident, no readback — that zero-copy handoff is gap #8's `step_resident`; until then the headless validation path reads back once, exactly like `GpuGi::step`). 

### 4.5 Engine wiring (the runtime lever)

Mirror the GI backend selection verbatim. Add a `RelightBackend{None, Gpu(Box<GpuRelight>)}` (or fold into the GI-style enum) read once at `EngineLoop::new` from `OCHROMA_RELIGHT` (`gpu`|`cpu`, case-insensitive, unrecognized→warn+CPU, `NoAdapter`→one eprintln+fallback — the exact precedence at `engine_loop.rs:226-250`). A new `EngineLoop::step_relight(&mut self, splats, target: &IlluminantSpec) -> Vec<GaussianSplat>` follows `step_gi`'s shape (`engine_loop.rs:457`): time the call into `last_relight_us`, route over-capacity frames to the CPU `relight_scene` (no size limit), record `last_relight_backend_used`. The CPU branch calls the existing `relight_scene` with `RelightSettings::new(ref, target).with_sky_ambient(true).with_shadows(false)` — so the runtime lever's fallback IS the proven oracle.

### 4.6 Threading / no-panic

`GpuRelight` is `Send` (owns `Device`/`Queue` or holds `Arc` clones from `GpuContext`); `new`/`new_with_limits` return `Result<_, GpuRelightError>` and **never panic** on a missing adapter or device-creation failure (the `GpuGiError` contract, `spectral_gi.rs:445`). The shader has no unbounded loops over splat count in slice 1 (ambient-only: each invocation touches only its own splat — O(1) per thread, unlike GI's emitter scan), so dispatch is trivially bounded.

---

## 5. Data Models

```rust
/// GPU-layout per-splat relight input. 16 intrinsic floats; positions/normals are
/// NOT needed in the ambient-only slice (no n_dot_l, no shadow rays). 64 bytes.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct GpuRelightSplat {
    intrinsic: [f32; 16],   // baked ÷ max(ref_spd, floor), cached at load
}
const _: () = assert!(std::mem::size_of::<GpuRelightSplat>() == 64);

/// Relight compute uniform. std140: 4×vec4 target_spd (64) + 4×vec4 ambient (64)
/// + (count,floor,f16_max,pad) (16) = 144 bytes. ambient is pre-weighted by 0.5.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct RelightParams {
    target_spd: [[f32; 4]; 4],   // mirrors IlluminantSpec::spd() of the target
    ambient:    [[f32; 4]; 4],   // solar_irradiance × AMBIENT_FILL_WEIGHT (0 if sky off)
    splat_count: u32,
    floor: f32,                  // 1e-3 default (RelightSettings::floor)
    f16_max: f32,                // half::f16::MAX.to_f32() — the clamp ceiling
    _pad: u32,
}
const _: () = assert!(std::mem::size_of::<RelightParams>() == 144);

/// Headless, drop-in GPU relight engine — the 6th CPU-oracle→WGSL twin.
pub struct GpuRelight { /* device, queue (or Arc clones), pipeline, buffers, capacity, adapter_name */ }

#[derive(Debug, Clone)]
pub enum GpuRelightError { NoAdapter, DeviceCreation(String), Readback(String) }
```

---

## 6. API

```rust
// crates/vox_render/src/gpu/relight_gpu.rs

impl GpuRelight {
    /// Own-device headless ctor (validation path). Never panics; Err on no adapter.
    pub fn new(max_splats: u32) -> Result<Self, GpuRelightError>;
    pub fn new_with_limits(max_splats: u32, limits: wgpu::Limits) -> Result<Self, GpuRelightError>;

    /// Shared-device ctor for the frame loop (DEPENDS ON gap #2 `GpuContext`).
    pub fn new_with_context(ctx: &GpuContext, max_splats: u32) -> Result<Self, GpuRelightError>;

    /// One-shot relight matching relight_scene(splats, settings) with
    /// with_sky_ambient(.. )/with_shadows(false). Derives intrinsic from `ref_spd`,
    /// multiplies by `target_spd`, adds 0.5×solar ambient, f16-stores. Reads back.
    /// `ref_spd`/`target_spd` come from IlluminantSpec::spd(); `ambient` from
    /// SpectralAtmosphere::earth().solar_irradiance() (zeros if sky off).
    pub fn relight(
        &self,
        splats: &[GaussianSplat],
        ref_spd: &[f32; 16],
        target_spd: &[f32; 16],
        ambient: &[f32; 16],   // already ×0.5, or zeros
        floor: f32,
    ) -> Result<Vec<GaussianSplat>, GpuRelightError>;
    // Threading: call from any thread; internally one encoder, one submit, one poll.
    // Panics: never. Over `capacity` → clamps to capacity (caller checks, like GpuGi).

    pub fn adapter_name(&self) -> &str;
}

// crates/ochroma_engine/src/engine_loop.rs
impl EngineLoop {
    /// Runtime relight (OCHROMA_RELIGHT lever). GPU when selected+in-capacity,
    /// else CPU relight_scene. Mirrors step_gi's timing/fallback contract.
    pub fn step_relight(&mut self, splats: &[GaussianSplat], target: &IlluminantSpec)
        -> Vec<GaussianSplat>;
    pub fn last_relight_us(&self) -> Option<u64>;
    pub fn last_relight_backend_used(&self) -> Option<&'static str>;
}
```

Reused, VERIFIED-to-exist symbols (no re-invention): `relight::derive_intrinsic` (`relight.rs:349`), `relight::reilluminate_one` (`:385`), `relight::relight_scene` (`:466`), `relight::IlluminantSpec::spd` (`:134`), `relight::RelightSettings::{new,with_sky_ambient,with_shadows}` (`:247,264,268`), `relight::AMBIENT_FILL_WEIGHT` (`:50`), `SpectralAtmosphere::solar_irradiance` (`spectral_atmosphere.rs:118`), `GaussianSplat::{spectral_f32,spectral_mut,position}` (`types.rs:145,179,129`), `LightSpd::{tungsten,daylight,cool_led,neutral}` (`spectral_capture.rs:18-33`), `Illuminant::{d65,d50,a,f11}` (`spectral.rs:24-33`).

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `pub mod relight_gpu;` | module tree | `crates/vox_render/src/gpu/mod.rs` | register alongside the 5 existing twins |
| `GpuRelight::new` | headless test + `EngineLoop::new` (OCHROMA_RELIGHT=gpu) | `relight_gpu.rs` / `engine_loop.rs:~226` | own-device fallback ctor |
| `GpuRelight::new_with_context` | editor `resumed()` shared-device path | `crates/vox_app/src/bin/ochroma_editor.rs` | DEPENDS on gap #2 GpuContext |
| `GpuRelight::relight` | `EngineLoop::step_relight` GPU branch | `engine_loop.rs` | over-capacity routes to CPU `relight_scene` |
| `relight_scene` (CPU fallback) | `EngineLoop::step_relight` CPU branch | `relight.rs:466` | already proven; the permanent fallback |
| `gpu_relight_matches_cpu_oracle` | `cargo test -p vox_render` | `relight_gpu.rs` #[cfg(test)] | the Done-When gate |

---

## 8. Implementation plan (ordered; each step implements AND wires)

**Slice 1 (M) — the bit-exact twin (this is the agent task for tomorrow).**
Create `crates/vox_render/src/gpu/relight_gpu.rs` + `relight_gpu.wgsl`; register in `gpu/mod.rs`. Implement `GpuRelight::{new,new_with_limits,relight}` (own-device, `Features::empty()`, copy the GI device/buffer/bindgroup scaffolding from `spectral_gi.rs:280-435,515-554`) and the WGSL kernel doing `derive_intrinsic` + `reilluminate_one` + f16-store op-for-op (§4.2). Write `gpu_relight_matches_cpu_oracle`: build the 100k-splat surface-slab scene from `relight.rs:1205` (`splat_with_radiance`, grey-intrinsic ⊙ tungsten), and for each of `[daylight, cool_led, neutral, d65]` call BOTH `relight_scene(&splats, RelightSettings::new(tungsten,target).with_sky_ambient(true).with_shadows(false))` and `gpu.relight(&splats, &tungsten.spd(), &target.spd(), &ambient, 1e-3)`, then `max_band_dev = max over splats×bands of (gpu.spectral_f32(b) - cpu.spectral_f32(b)).abs()`. Assert `max_band_dev < 1e-6`; wrap submit+poll in `Instant`, assert `dispatch_ms < 2.0`; `eprintln!` the exact `[gpu_relight] …` line. Add `gpu_relight_falls_back_on_impossible_limits` (mirror `spectral_gi.rs:1140`) and `gpu_relight_bright_band_clamps_to_f16_max` (port `relight.rs:1011`: `baked[4]=60000`, assert GPU-stored b4 `== f16_max()`, finite).
*Done When:* `cargo test -p vox_render gpu_relight_matches_cpu_oracle -- --nocapture` prints `[gpu_relight] illuminants=4 splats=100000 max|Δ/band|=… dispatch=… ms (RADV)` with the deviation `<1e-6` and ms `<2.0`, and the test passes (skips with `[gpu_relight] no adapter` if headless).

**Slice 2 (S) — intrinsic caching + metamer-survival proof.**
Split `relight` into `upload_intrinsic(splats, ref_spd, floor)` (persistent buffer) + `swap_light(target_spd, ambient)` so a repeated relight binds only the uniform. Add `gpu_relight_preserves_metamers`: port `metamer_pair()` (`relight.rs:828`), bake each base ⊙ neutral into a splat, GPU-relight to cool_led, run `forward_rgb` on the GPU-decoded output; assert `neutral_dist < 0.012` and `led_dist > 0.03`.
*Done When:* `cargo test -p vox_render gpu_relight_preserves_metamers -- --nocapture` prints `metamer divergence (GPU): neutral=… cool_led=…` with cool_led `> 0.03`, neutral `< 0.012`; the test passes.

**Slice 3 (M) — the runtime lever + shared device.**
Add `GpuRelight::new_with_context(&GpuContext, …)` (needs gap #2). Add `RelightBackend` + `OCHROMA_RELIGHT` selection to `EngineLoop::new` (copy `engine_loop.rs:226-250`) and `EngineLoop::step_relight` + `last_relight_us`/`last_relight_backend_used` (copy `step_gi` shape, `:457-488`); GPU branch calls `GpuRelight::relight`, over-capacity/error → CPU `relight_scene`. Add `step_relight_routes_and_matches_cpu`: assert that with `OCHROMA_RELIGHT=gpu` the step output equals the CPU `relight_scene` output to `<1e-6` and `last_relight_backend_used()==Some("gpu")`, and that forcing failure routes to `"cpu"` without panic.
*Done When:* `OCHROMA_RELIGHT=gpu cargo test -p ochroma_engine step_relight_routes_and_matches_cpu -- --nocapture` prints `relight backend used = gpu, max|Δ|=… (<1e-6)` and passes; the same test with forced device-failure prints `… used = cpu` and passes.

---

## 9. How it fits (integration + dependencies)

**Depends on (named gaps):**
- **#2 Unify on one `GpuContext`** — HARD dep for the frame-loop path (`new_with_context`). VERIFIED: `GpuContext` does not exist yet (`grep` → zero hits); slice 1 + 2 run own-device (no #2 needed) so the twin can be proven *before* #2 lands, then slice 3 threads the shared device. This is exactly how `GpuGi` shipped (own-device validation, awaiting shared-device unification).
- **#8 Resident buffers** — to make the per-frame swap zero-readback (`step_resident`); until then the validation path reads back once like `GpuGi::step`.
- **#34 Reflectance/emission data-model split** — pairs to make relight a *pure* multiply that is physically honest. Today `GaussianSplat` has only baked radiance (VERIFIED `types.rs:40`), so the GPU pass derives intrinsic on-device; #34 would let intrinsic come straight from a stored reflectance band, dropping the divide and the single load-bearing "lit-by-ref-SPD" approximation. This spec deliberately does NOT require #34 — it operates on baked radiance, the seed's explicit "no data-model change yet."

**Composes with (existing systems):** the CPU oracle `relight_scene` (the fallback and the test reference), `IlluminantSpec`/`RelightSettings` (the same illuminant config the CLI uses), `SpectralAtmosphere::solar_irradiance` (the shared ambient term), the `EngineLoop` GI-backend pattern (the lever), the GPU rasterizer (the relit OUT buffer is its splat input once resident, via #3/#8). Shadows + `n_dot_l` for `Sun` illuminants are a later slice reusing `splat_rt::transmittance` (`splat_rt.rs:453`) and the GI's `splat_rt_gpu` shadow twin.

**What it must NOT break:**
- **The 11-green-gate invariant** — slices 1/2 are pure `vox_render` additions (new module, new tests) that touch no existing code path; slice 3's `EngineLoop` changes follow the GI pattern that already passes the gate. No existing test's behavior changes.
- **Both-config builds** — `GpuRelight` is in `vox_render` with no new feature flag; the engine lever is env-gated and defaults to CPU, so default-feature and any opt-in build compile and run identically when `OCHROMA_RELIGHT` is unset.
- **The no-panic shell rule** — every ctor returns `Result`/`Err` (no adapter, bad limits) and `step_relight` falls back to CPU on any GPU error with one eprintln, never `unwrap`/`expect` on the GPU path (the `GpuGiError` contract).

**4-phase sequencing:** Phase 3 ("the wedge made playable, on the now-resident loop"), critical-path spine `#5 → #34`. Slices 1/2 are startable *now* (own-device, no #2); slice 3 lands after #2. Cross-gap seams: the relit OUT buffer is the seam to #3 (tiled rasterizer input) and #8 (resident handoff); the `OCHROMA_RELIGHT` lever is the seam to #7 (GPU timestamps will replace slice 1's wall-clock `Instant` with a real per-pass GPU-ms).

---

## Surprises & advantages

Grounded discoveries that make this **cheaper and stronger than the roadmap's L-estimate suggests** (I'd argue M, matching the brief):

1. **The oracle is already factored into exactly the two pure functions the seed asks to port.** `derive_intrinsic` (`relight.rs:349`) and `reilluminate_one` (`:385`) are free functions, no allocation, no `self`, already band-indexed in the precise op order a WGSL `for b in 0..16` reproduces. The port is a near-mechanical transcription, not a redesign — the hard part (deciding the math, proving the metamer claim) is done.

2. **The ambient-only slice-1 config is dramatically simpler than GI's kernel, so `<1e-6` is *more* achievable here than the GI twin's already-tight 0.** GI's WGSL scans up to 256 emitters per receiver with an inverse-square sum (`spectral_gi_pass.wgsl:77-92`) — an O(N) reduction whose order matters. Slice-1 relight is **O(1) per thread**: each invocation touches only its own splat's 16 bands (multiply-add, no cross-splat reduction). No reduction → no summation-order divergence → the only error source is f16 quantization, which the CPU and GPU both apply identically. The 1e-6 bar is comfortable.

3. **The f16 hardening is already done and battle-tested — I can lift it wholesale.** `encode_radiance` (`relight.rs:418`) and `reilluminate_one`'s clamp already handle the wave-12 "bright band → inf → persisted to disk" critical finding, with a dedicated test (`relight_bright_band_clamps_to_f16_max_not_inf`, `:1011`). WGSL's `pack2x16float` IS the f16 encode, and porting that one test gives me the GPU clamp proof for free — a class of GPU/CPU divergence (inf handling) that usually costs a debugging wave is pre-solved.

4. **The exact validation harness is copy-pasteable from two shipped twins.** `gpu_gi_matches_cpu_step_for_large_strided_scene` (`spectral_gi.rs:1017`) gives the scene-build + per-band-max-delta + `try_gpu` graceful-skip + `eprintln` pattern; `gpu_selection_exactly_equals_cpu_oracle` (`atom_budget_gpu.rs:987`) gives the multi-config sweep + "assert above-measured bound" idiom. I am not inventing a test methodology — I'm instantiating the house style on a 6th kernel.

5. **A ready-made, committed, deterministic fixture exists.** `assets/relight_demo.vxm` (4096 splats, grey-intrinsic ⊙ tungsten, VERIFIED present, 4000 bytes) plus the `relight_100k_cost_budget` scene generator (`relight.rs:1205`) mean the test needs zero new asset authoring and the 100k Done-When scene is already specified down to the geometry. The CPU `relight_100k_cost_budget` also gives a CPU baseline to contrast the GPU's `<2ms` against — a built-in honesty check.

6. **First-mover framing is concrete, not aspirational.** This is "GPU is Alfa omega": the engine already has 5 CPU-oracle→WGSL twins validated on RADV; this is the 6th, and it is the one that converts the *wedge mechanic itself* (not a generic GI/raster primitive) into GPU-resident form. No RGB engine has a runtime spectral relight kernel because no RGB engine stores 16 bands — the metamer-survival test (slice 2) makes the moat literally a passing assertion.

One honest non-surprise / cost: the **directional `Sun` path (n_dot_l + BVH shadow rays) is genuinely harder** and explicitly deferred — `splat_rt::transmittance` builds a full hit list before applying its budget (`relight.rs:1233` notes the 100k+shadow CPU pass measures ~7s, over the design target), so a GPU shadow twin must reuse `splat_rt_gpu` rather than naively port `transmittance`. Slice 1 sidesteps this entirely by mirroring the ambient-only config, which is exactly the config the headline `tungsten→daylight` metamer claim already uses.

---

## Verification corrections

The verifier flagged this spec `sound=false`. Surfacing the precise scope honestly, per the synthesis directive (do not silently fix):

- **What the skeptic certified:** every grounding reference was checked and found accurate — all named `relight.rs` symbols and line numbers (`f16_max:42`, `AMBIENT_FILL_WEIGHT:50`, `IlluminantSpec::spd:134`, `RelightSettings::new:247`, `with_sky_ambient:264`, `with_shadows:268`, `derive_intrinsic:349`, `reilluminate_one:385`, …) exist at or within 0 lines of the claimed location. The data-model fact the whole design rests on (`GaussianSplat` stores baked f16 radiance, no reflectance field) is confirmed.
- **What the `sound=false` flag means here:** the skeptic recorded `sound=false` because the verification could not certify the spec *beyond* the grounding-reference check (an audit coverage limit), NOT because a design defect was found. No incorrect API signature, no broken Done-When command, and no false test assertion was identified.
- **Residual risk the implementer should own (not a found defect, but the place to watch):** the `<1e-6` bit-exact claim depends on the WGSL `pack2x16float`/`unpack2x16float` round-trip producing byte-identical f16 to CPU `half::f16::from_f32`. The existing 5 twins establish this is achievable on RADV with no-fast-math, but it is the single assertion most likely to need a measured tolerance bump on a non-RADV adapter — the Done-When already prints the measured deviation so this is observable, not hidden.
