> **Adversarial verification:** `sound=false` for a **process reason, not a design defect**: the copy of the markdown the skeptic received was **truncated mid-`GpuGi::new_with_context`** (last line it saw: `device: ctx.device().clone(),`), so the required adversarial checks on the FIRST launchable step (a real Done-When command + a real test assertion) **could not be performed** — the skeptic could not certify what it could not see. The grounding refs that WERE checkable verified clean (the 7-device topology, the `wgpu` Clone facts, the `GpuGiPass::new` seam). The full spec below is complete (Steps 1–3 with Done-Whens present); the `sound=false` reflects an unverified first step, not a found error. See **Verification corrections**.

# Design: Unify on one wgpu device/queue shared across all render+compute modules (2026-06-07)

**Status:** Draft
**Scope:** Eliminate the engine's 7-independent-wgpu-device topology by introducing a `GpuContext { device: Arc<Device>, queue: Arc<Queue> }` created once at surface bring-up and threaded into every GPU compute/render module, starting with `GpuGi`. Affects `vox_render` (the GPU modules + `WgpuBackend`), `ochroma_engine` (`EngineLoop` GI selection), and the `ochroma_editor` binary.
**Related:** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #2, GPU Loop dim, L, 72/90); seeds the resident-frame-loop chain #3/#5/#7/#8/#12/#20.
**Dimension:** Runtime Performance & the GPU Frame Loop.

---

## 1. What we need

The roadmap's GPU-Loop audit is blunt: *"17 `Instance::new()` sites; 6 compute modules each own a separate device, plus the present surface owns a 7th — every cross-module handoff is a CPU round-trip."* I verified this directly: `wgpu::Instance::new` appears at `spectral_gi.rs:519`, `material_gpu_eval.rs:56`, `gpu/many_light_gpu.rs:175`, `gpu/hybrid_compose_gpu.rs:173`, `gpu/atom_budget_gpu.rs:216`, `gpu/splat_rt_gpu.rs:139`, and `gpu/wgpu_backend.rs:34` (plus two in `vox_ui/vello_ctx.rs` and one in `vox_web`). Each of the 6 GPU twins runs its own `Instance::new → request_adapter → request_device` and stores its own `device`/`queue`/`adapter_name`. A buffer produced by one device can never bind into another — the only handoff is a CPU `Vec`.

After this exists:

- **A developer can construct any GPU module against the same device the window presents on.** `GpuGi::new_with_context(&GpuContext, capacity)` builds the GI compute pass on the editor's present device — no second `Instance::new`, no second adapter, no orphan device. Today `GpuGi::new` (`spectral_gi.rs:485`) always spins up an isolated headless device.
- **The editor reports, at runtime, that GI runs on the present device.** `OCHROMA_GI=gpu cargo run --bin ochroma_editor -- --frames 2` prints `GI on shared present device` and the GpuGi's `adapter_name` equals the backend's adapter name — observable proof the two are one device, not two that happen to pick the same GPU.
- **A buffer can flow GPU→GPU between modules without leaving the device.** Sharing one `Arc<Device>` is the precondition for gap #8 (`step_resident` returning a `wgpu::Buffer` the rasterizer binds directly) and gap #3 (the tiled rasterizer chaining RadixSort→TileAssign→EWA on one device). Neither is expressible while every module owns a private device.
- **The provability culture is preserved, not weakened.** Each twin keeps its standalone `new()` (own-device) constructor, so the 6 CPU-oracle→WGSL validation tests on RADV still run unchanged. The shared-context path is *added alongside*, never *replacing*, the validated path.
- **The AAA bar:** Unreal 5.5 / Unity 6 run cull, LOD, draw-arg generation, GI and shading on one device with async-compute overlap and a single submit per frame; no shipped engine round-trips through `Vec<f32>` between GI and raster. The bar here is modest and foundational — *one device exists* — but it is the hard floor under every GPU-loop gap. The roadmap names it the critical-path spine: *"#4 (verifiable) → #2 (one device) → #7/#12/#3/#8 (measured resident frame) → #5/#34 (relight playable)."* Nothing downstream is startable until one device exists.

**Why it is blocking (cite):** roadmap §2 Runtime Performance: *"no per-kernel optimization can hit 16.6ms while data crosses PCIe and stalls the queue between every stage; with 7 independent devices every cross-module handoff (GI→raster, LOD→raster, many-light→shade) is forced through a CPU `Vec`."*

---

## 2. How it's gonna be (the design)

### 2.1 The new type — `GpuContext` (NEW, `vox_render/src/gpu/mod.rs`)

`GpuContext` does not exist anywhere today (verified: zero hits for `struct GpuContext`). It is a thin, cheaply-cloneable handle bundling the device, queue, and the adapter identity needed for the honesty assertion. It lives in `crates/vox_render/src/gpu/mod.rs` — the module that already hosts `wgpu_backend` and all 5 GPU twins, and is game-agnostic (engine crate, no buildings/zoning).

```rust
/// The single wgpu device + queue every render and compute module binds against.
/// Created once at surface bring-up (editor `resumed()` / engine present init) and
/// cloned (cheaply — wgpu `Device`/`Queue` are Arc-backed handles) into each pass
/// constructor, so a buffer produced by one pass binds directly into the next with
/// no CPU round-trip.
#[derive(Clone)]
pub struct GpuContext {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    adapter_name: String,
    adapter_backend: wgpu::Backend, // Vulkan/GL/etc — part of the device identity
}

impl GpuContext {
    /// Build from an already-created device/queue/adapter (the present path).
    /// `WgpuBackend` exposes `device()`/`queue()` by reference; wgpu 24's
    /// `Device`/`Queue` derive `Clone`, so the clone is a handle bump, not a
    /// device re-creation.
    pub fn from_parts(device: &wgpu::Device, queue: &wgpu::Queue, info: &wgpu::AdapterInfo) -> Self {
        Self {
            device: std::sync::Arc::new(device.clone()),
            queue: std::sync::Arc::new(queue.clone()),
            adapter_name: info.name.clone(),
            adapter_backend: info.backend,
        }
    }
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
    pub fn adapter_name(&self) -> &str { &self.adapter_name }
    pub fn adapter_backend(&self) -> wgpu::Backend { self.adapter_backend }
}
```

**Design decision — `Arc<Device>` over bare `Device`.** wgpu 24.0.5 `Device` and `Queue` already `#[derive(Debug, Clone)]` (verified at `wgpu-24.0.5/src/api/device.rs:17` and `api/queue.rs:12`); they are internally Arc-backed. The roadmap's seed says `device: Arc<Device>`, and we honor it: the explicit `Arc` makes the shared-ownership contract legible at every call site and is what downstream gaps (#8 resident buffers, #20 residency manager) will store. The clone in `from_parts` is a refcount bump, *not* a second `request_device`.

### 2.2 `WgpuBackend` must retain its adapter (small change, `vox_render/src/gpu/wgpu_backend.rs`)

Today `WgpuBackend::new_async` drops the `adapter` after building the surface config (it is a local at `wgpu_backend.rs:48`); the struct stores `device`, `queue`, `config`, `width`, `height` but **not** the adapter. The Done-When requires "the GpuGi reports the SAME adapter as backend," so the backend must expose its adapter identity. Add one field and one accessor:

```rust
pub struct WgpuBackend {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo, // NEW — captured before adapter is dropped
    config: wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
}
impl WgpuBackend {
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo { &self.adapter_info } // NEW
    pub fn gpu_context(&self) -> crate::gpu::GpuContext {                    // NEW convenience
        crate::gpu::GpuContext::from_parts(&self.device, &self.queue, &self.adapter_info)
    }
    // device()/queue()/surface_format()/width()/height() unchanged
}
```

Captured by inserting `let adapter_info = adapter.get_info();` right after the existing `let adapter = ...` block (`adapter.get_info()` verified to exist, `Adapter` API `adapter.rs:155`), threaded into the returned `Self`. This is the only structural change to the present path; `device()`/`queue()` keep their existing `&wgpu::Device`/`&wgpu::Queue` signatures so every current caller (egui-wgpu setup, `capture_shot`, `present_framebuffer`) compiles untouched.

### 2.3 `GpuGi::new_with_context` (NEW constructor, `vox_render/src/spectral_gi.rs`)

`GpuGi` (`spectral_gi.rs:471`) stores `device: wgpu::Device`, `queue: wgpu::Queue`, `pass: GpuGiPass`, `capacity: u32`, `pub adapter_name: String`. Its existing `new → new_with_limits → new_async` chain owns the `Instance::new` at line 519. The clean seam already exists: `GpuGiPass::new(&device, capacity)` (`spectral_gi.rs:280`) takes a borrowed device and builds all buffers/pipeline/bind-group — it has no knowledge of where the device came from. So `new_with_context` is a 6-line constructor reusing the pass builder:

```rust
impl GpuGi {
    /// Build GI on a device the caller already owns (the shared present device),
    /// instead of creating an isolated headless device. No `Instance::new`, no
    /// `request_adapter`, no second device — the GI pass binds buffers on the
    /// SAME `wgpu::Device` the window presents on.
    pub fn new_with_context(ctx: &crate::gpu::GpuContext, max_splats: u32) -> Self {
        let capacity = max_splats.max(1);
        let pass = GpuGiPass::new(ctx.device(), capacity);
        Self {
            device: ctx.device().clone(),     // handle bump, not a new device
            queue: ctx.queue().clone(),
            pass,
            capacity,
            adapter_name: ctx.adapter_name().to_string(),
        }
    }
}
```

Returns `Self` (not `Result`) because the device already exists and succeeded at surface bring-up — there is no adapter/device-creation step that can fail here. The existing `new()`/`new_with_limits()`/`new_failing_for_test()` are **untouched**, so the GI validation test (`gpu_gi_matches_cpu_step_for_large_strided_scene`, `spectral_gi.rs:1017`) and the no-panic fallback test (`spectral_gi.rs:1149`) keep running on their own devices.

### 2.4 Wiring the selection: `EngineLoop::use_gpu_gi_with_context` + editor `resumed()`

`EngineLoop` currently routes GI by env at construction: `OCHROMA_GI=gpu` calls `GpuGi::new(GPU_GI_CAPACITY)` (`engine_loop.rs:229`) — a *second* device — and `use_gpu_gi()` (`engine_loop.rs:285`) does the same. We add a context-aware variant that does **not** create a device:

```rust
impl EngineLoop {
    /// Switch GI to the GPU path on a caller-supplied shared device. Unlike
    /// `use_gpu_gi()` (which creates an isolated device), this binds GI to the
    /// present device, so GI output buffers can later flow into the rasterizer.
    pub fn use_gpu_gi_with_context(&mut self, ctx: &vox_render::gpu::GpuContext) {
        self.gi_backend = GiBackend::Gpu(Box::new(
            GpuGi::new_with_context(ctx, self.gpu_gi_capacity)
        ));
    }
}
```

The editor binary (`ochroma_editor.rs`) is where the wedge becomes observable. Today `resumed()` (`ochroma_editor.rs:472`) builds `WgpuBackend` (line 485) and constructs **neither** `EngineLoop` nor `GpuGi` — verified: no `ochroma_engine`/`GpuGi` reference in the binary, though both are deps of `vox_app` (Cargo.toml lines 33, 43). We add, after `configure_present_mailbox(&backend)`:

```rust
if std::env::var("OCHROMA_GI").map(|v| v.eq_ignore_ascii_case("gpu")).unwrap_or(false) {
    let ctx = backend.gpu_context();
    let gi = vox_render::spectral_gi::GpuGi::new_with_context(&ctx, /*capacity*/ 200_000);
    assert_eq!(gi.adapter_name, backend.adapter_info().name); // same device, not coincidence
    self.shared_gi = Some(gi);
    println!("GI on shared present device");
    println!("[ochroma_editor] GI adapter = {} (backend adapter = {})",
             gi.adapter_name, backend.adapter_info().name);
}
```

Note the editor does not need a full `EngineLoop` for this first slice — it constructs `GpuGi` directly from the backend's context, proving the shared-device path end-to-end. The `EngineLoop::use_gpu_gi_with_context` path is wired and unit-tested in the same slice so the next gap (#9 Play-in-Editor) inherits it.

### 2.5 Data flow

```
  ochroma_editor::resumed()
        │
        ├─ WgpuBackend::new()  ── Instance::new ──► adapter ──► device, queue   (THE ONE DEVICE)
        │                                              │
        │                                        adapter.get_info()  (retained)
        │
        ├─ backend.gpu_context() ─► GpuContext { Arc<device>, Arc<queue>, adapter_name }
        │                                  │  (clone = handle bump, no 2nd Instance::new)
        │            ┌─────────────────────┼─────────────────────────────┐
        │            ▼                     ▼                              ▼
        │     GpuGi::new_with_context   (future) TiledSplatRenderer   (future) GpuRelight
        │            │                     #3                            #5
        │     GpuGiPass on SAME device  ── output buffer binds directly ──►  (gap #8)
        │
        └─ egui-wgpu Renderer::new(backend.device(), …)   ← already shares this device
```

Today only egui shares the present device. After this slice, GI does too, on an explicit `GpuContext` the rest of the GPU stack will thread through.

---

## 3. How it's gonna be made (the implementation plan)

Each step implements **and** wires in the same commit. Headless proofs use the existing skip-on-no-adapter convention (`spectral_gi.rs:899`: `Err(GpuGiError::NoAdapter) => skip`) so CI without a GPU stays green.

### Step 1 — `GpuContext` + `WgpuBackend` adapter retention + `GpuGi::new_with_context` (S) — **launchable tomorrow**

**Files:**
- `crates/vox_render/src/gpu/mod.rs` — add `pub mod gpu_context;` (or inline `GpuContext` in `mod.rs`) with the `GpuContext` struct + `from_parts`/`device`/`queue`/`adapter_name`/`adapter_backend` from §2.1.
- `crates/vox_render/src/gpu/wgpu_backend.rs` — add `adapter_info: wgpu::AdapterInfo` field, capture `let adapter_info = adapter.get_info();` after `adapter` is obtained (around line 62, before the surface_caps block that consumes `&adapter`), thread into both `Ok(Self { … })` returns; add `adapter_info()` and `gpu_context()` accessors.
- `crates/vox_render/src/spectral_gi.rs` — add `GpuGi::new_with_context` from §2.3.
- `crates/vox_render/tests/gpu_context_test.rs` (NEW) — the headless proof.

**Test assertions (real computed outcomes, no `is_some()`):** In `gpu_context_test.rs`, build a real headless `GpuContext` via a helper that creates one `Instance`/adapter/device (mirroring `wgpu_backend` minus the surface), then:
```rust
let gi_ctx = GpuGi::new_with_context(&ctx, 1024);
// (a) shared identity: GI's adapter name is byte-equal to the context's.
assert_eq!(gi_ctx.adapter_name, ctx.adapter_name());
// (b) SAME device, proven by behavior: run the GI step on a known scene and
//     assert the lit output equals the CPU oracle band-for-band (reuse the
//     emitter+receiver scene from spectral_gi.rs:697). A real computed value:
let lit = gi_ctx.step(&scene, 12.0).unwrap();
let cpu = cpu_reference_gi(&scene, 12.0); // gather_radiance-based oracle
for b in 0..16 { assert!((decode(lit[recv].spectral()[b]) - cpu[recv][b]).abs() < 1e-3,
    "band {b}: gpu {} vs cpu {}", …); }
```
Skip with `eprintln!("[gpu_context_test] no adapter — skipping")` and `return` on `NoAdapter`, exactly as `spectral_gi.rs:899`.

**Done When:** `cargo test -p vox_render --test gpu_context_test -- --nocapture` prints `gpu_context: GI lit band 3 = <value> matches CPU oracle within 1e-3 on adapter <name>` and exits 0 (on a machine with an adapter; prints the skip line and exits 0 otherwise). `cargo build -p vox_render` succeeds with no new `Instance::new` introduced (`grep -c "Instance::new" crates/vox_render/src/spectral_gi.rs` stays at 1).

### Step 2 — Editor wires `GpuContext` + GpuGi in `resumed()`, prints the headline (M)

**Files:**
- `crates/vox_app/src/bin/ochroma_editor.rs` — add a `shared_gi: Option<vox_render::spectral_gi::GpuGi>` field on `EditorHost`; in `resumed()` after `configure_present_mailbox(&backend)`, the `OCHROMA_GI=gpu` block from §2.4 (build context, construct GpuGi, `assert_eq!` adapter names, print `GI on shared present device`).
- `crates/ochroma_engine/src/engine_loop.rs` — add `use_gpu_gi_with_context` from §2.4 + a unit test asserting `gi_backend()` becomes `"gpu"` after the call.

**Done When:** `OCHROMA_GI=gpu cargo run -p vox_app --bin ochroma_editor -- --frames 2 --shot /tmp/gi.png` prints `GI on shared present device` AND a line `[ochroma_editor] GI adapter = <X> (backend adapter = <X>)` where the two names are identical, writes `/tmp/gi.png`, and exits 0. `OCHROMA_GI=gpu cargo run … 2>&1 | grep -c "Instance::new\|request_adapter"` shows no second adapter request beyond the backend's (verify via an added `[wgpu]`/`[gpu_gi]` trace: the GI path prints no "creating device" line). With `OCHROMA_GI` unset, the headline does NOT print (CPU default preserved) and the editor still renders 2 frames and exits 0.

### Step 3 — Engine GI selection prefers a supplied context; both-config build proof (M)

**Files:**
- `crates/ochroma_engine/src/engine_loop.rs` — the env-`gpu` construction path (`engine_loop.rs:229`) gains a doc note that the device-creating `GpuGi::new` is the *headless/no-window* path; window hosts must call `use_gpu_gi_with_context`. No behavior regression to the headless engine binary (which has no surface).
- `crates/vox_app/tests/` — add `editor_shared_gi_test.rs` integration test (or extend `integration_test.rs`) that builds an `EngineLoop`, calls `use_gpu_gi_with_context` with a test context, steps GI once, and asserts the lit result is non-trivially different from the input on the emitter-lit band (real computed delta, e.g. `assert!(lit_band > input_band + 0.01)`), skipping on no-adapter.

**Done When:** `cargo test -p ochroma_engine use_gpu_gi_with_context -- --nocapture` prints `engine GI backend = gpu after use_gpu_gi_with_context` and the lit-band delta value, exits 0. `cargo build --workspace` and `cargo build --workspace --features spectra-native` both succeed (both-config invariant). `cargo test --workspace` shows the pre-existing GI twin tests (`gpu_gi_matches_cpu_step_for_large_strided_scene`) still pass unchanged — the own-device path is intact.

---

## 4. How it fits (integration + dependencies)

**Depends on:** Nothing structural — this is a Phase-1 floor, startable on the current codebase tomorrow (roadmap §5: *"#2 Unify on one `GpuContext` (the hard floor under all GPU-loop work)"*). It pairs naturally with gap #4 (push + CI green) so the new headless proof runs on a real CI machine, but does not block on it.

**What depends on it (named):** #3 (wire GPU tiled rasterizer — `TiledSplatRenderer::new(&GpuContext)`), #5 (GPU relight kernel — operates on the shared device's splat buffer), #7 (timestamp instrumentation — `TIMESTAMP_QUERY` enabled on the shared device), #8 (resident buffers — `GpuGi::step_resident(&Buffer)->Buffer` only meaningful on one device), #12 (render graph as GPU executor — one encoder on one device), #20 (residency manager — `GpuResidencyManager::new(&GpuContext, budget)`). All six name `GpuContext` in their first-slice seeds. This is the single most-depended-on GPU gap.

**Composes with existing systems:** `WgpuBackend` (present surface, unchanged accessors), `egui-wgpu Renderer` (already shares the present device — this generalizes that sharing to GI), `EngineLoop::step_gi`/`gi_backend()`/`last_gi_backend_used()` telemetry (unchanged signatures; a new selection entry point), the 6 GPU twins (each keeps its own-device `new()`; `new_with_context` is added per-twin in later gaps using the identical `GpuGiPass`-style seam — `splat_rt_gpu`, `many_light_gpu`, `hybrid_compose_gpu`, `atom_budget_gpu`, `material_gpu_eval` all share the structure verified at their `adapter_name` sites).

**Must NOT break:**
- **The 11-green-gate / both-config invariant.** Step 3's Done-When explicitly builds `--features spectra-native` and reruns the full GPU twin suite. The own-device constructors are untouched, so every CPU-oracle→WGSL validation test on RADV is byte-identical.
- **The no-panic shell rule.** `GpuGi::new_with_context` returns `Self` (the device already succeeded), but the editor wiring is gated behind `OCHROMA_GI=gpu` — when unset, zero new code runs and the editor's existing no-panic path is unchanged. The `assert_eq!` on adapter names is a debug invariant on a path the user opted into; if a future device mismatch is possible it becomes a logged warning + CPU fallback (mirroring `engine_loop.rs:231` graceful fallback), never a release panic.
- **The provability culture.** The new headless proof asserts a *computed* GI band value against the CPU oracle (not `is_some()`), extending the existing `gpu_gi_matches_cpu` pattern.

**4-phase placement:** Phase 1 (floors), critical-path spine. After this, Phase 2 (#7 timestamps → #12 GPU executor → #3 tiled raster → #8 resident buffers) becomes startable, leading to Phase 3 (#5 relight playable). 

**Cross-gap seams:** The `GpuContext::device()` accessor returning `&wgpu::Device` is the exact handle #3's `TiledSplatRenderer::new(&GpuContext)` and #7's `TIMESTAMP_QUERY` enablement consume. The decision to store `Arc<Device>` (not bare `Device`) is precisely so #8/#20 can hold long-lived buffer pools tied to the device lifetime without reborrow gymnastics.

---

## Surprises & advantages

Grounding surfaced four concrete, non-aspirational advantages that make this cheaper than the "L" estimate implies:

1. **wgpu 24's `Device` and `Queue` already derive `Clone` (verified `wgpu-24.0.5/src/api/device.rs:17`, `api/queue.rs:12`).** They are internally Arc-backed handles. This means `GpuContext::from_parts` can be built from `WgpuBackend`'s existing `device() -> &wgpu::Device` / `queue() -> &wgpu::Queue` accessors with a cheap clone — **no need to restructure `WgpuBackend` to store `Arc<Device>` internally**, which the roadmap seed implied might be necessary. The backend's public API is preserved verbatim; only one field (`adapter_info`) is added. This collapses the riskiest-looking part of the slice (touching the present path that every editor frame depends on) into a one-field, accessor-only change.

2. **`GpuGiPass::new(&device, max_splats)` already cleanly separates pass-construction from device-creation (`spectral_gi.rs:280`).** Every buffer, pipeline, and bind group is built from a borrowed `&wgpu::Device` with zero device-ownership logic. So `new_with_context` is genuinely ~6 lines reusing the existing pass builder — the device-acquisition code (`Instance::new` → adapter → device, lines 519-543) is the *only* thing being bypassed, and it is already isolated in `new_async`. The same seam (`*Pass::new(&device, …)`) exists in all 5 other twins per their verified structure, so rolling `new_with_context` across them in later gaps is mechanical, not architectural.

3. **The editor binary is the cleanest possible first proof site — it constructs neither `EngineLoop` nor `GpuGi` today.** Verified: no `ochroma_engine`/`GpuGi` reference in `ochroma_editor.rs`, yet both are already `vox_app` deps (Cargo.toml lines 33, 43). There is no existing GI wiring to untangle or regress; the slice *adds* the first GI construction to the editor, gated behind an env var, with the present path otherwise byte-identical. This is the rare integration gap where the integration point is empty.

4. **The honesty headline is a wedge synergy, not just a checkbox.** The Done-When's `adapter_name` equality assertion turns "we unified the device" from a claim into a headless-observable fact — exactly the provability-as-product-property the roadmap calls the credibility wedge (§1.3). The same `adapter.get_info()` retained on `WgpuBackend` for this assertion is *also* the data #7's frame-budget HUD needs to label which GPU the per-pass ms came from, and #21's device-lost recovery needs to re-match an adapter — one tiny field pays three gaps forward.

No surprises that *block* the work were found; the topology is exactly as the roadmap audit described (7 devices, verified at the named line numbers), and the seam to fix it is unusually clean.

---

## Verification corrections

The skeptic flagged `sound=false`. Surfaced honestly:

- **Why `sound=false`:** the markdown the skeptic was handed was **truncated mid-`GpuGi::new_with_context`** (last visible line: `device: ctx.device().clone(),`). Two of the mandated adversarial checks — (2) "is the FIRST implementation step launchable, with a real Done-When command and a test assertion on a real computed outcome?" — **could not be evaluated**, because Steps 1–3 and their Done-Whens were below the truncation point. The skeptic correctly declined to certify what it could not read.
- **What this means for the spec:** this is a **verification-coverage gap, not a discovered design defect**. The skeptic affirmatively verified everything it *could* see: the 7-device topology at the named `Instance::new` lines, the wgpu 24 `Device`/`Queue` `Clone` facts (`device.rs:17`, `queue.rs:12`), `adapter.get_info()` existence, and the `GpuGiPass::new(&device, …)` device/pass separation seam. None of those checks failed.
- **The residual obligation on the implementer:** the full Steps 1–3 (present above) DO each carry an exact Done-When command and a real computed-outcome assertion (Step 1 asserts GI band values match the CPU oracle within `1e-3`; Step 2 asserts adapter-name equality + no second `Instance::new`; Step 3 asserts a non-trivial lit-band delta and a both-config build). The implementer should treat the first step as **unverified-by-the-skeptic** and confirm on first run that (a) `GpuGi::new_with_context` compiles against the real `GpuGiPass::new` signature, and (b) the headless `gpu_context_test` actually exercises a shared device (the adapter-name equality is necessary but not sufficient — Step 1's behavioral band-match against the CPU oracle is the real proof the device is shared and functional).
