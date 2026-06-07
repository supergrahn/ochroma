> **Adversarial verification:** ISSUES FOUND (one blocker, surfaced not silently fixed). The skeptic confirmed the architectural core is sound — every named kernel exists and is RADV-validated (`GpuGi`, `SplatRasterPass`/`TileAssignPass`/`RadixSortPass`), `WgpuBackend::device()/queue()` are real public accessors (`wgpu_backend.rs:289,294`), `GpuGiPass::radiance_buffer` is already `STORAGE | COPY_SRC` (`spectral_gi.rs:293`), and the band-count seam (GI's `array<f32,16>` vs raster's `spectral:array<f32,8>`) is a real, correctly-identified gotcha. BUT the headline Done-When command **is not launchable as written**: `scale_trial.rs`'s `main()` has NO argv parsing, so `cargo run … -- --gpu-resident --frames 30` would ignore both flags and run the existing CPU sweep. See Verification corrections.

## Verification corrections

- **BLOCKER (headline Done-When not launchable as written):** the spec's primary observable is `cargo run -p vox_app --bin scale_trial -- --gpu-resident --frames 30` printing `[scale_trial] splat readback: ELIMINATED …`. But `scale_trial.rs` has no command-line argument parsing at all — its `main()` runs a fixed sweep and ignores all argv. **Correction (must be folded into Step 3 before that Done-When can pass):** Step 3 must FIRST add `--gpu-resident` and `--frames N` flag parsing to `scale_trial.rs` (the spec already names this as a sub-task, but the *flag-parsing addition itself* is the load-bearing prerequisite, not an afterthought — the binary cannot honor the flags until it parses them). Mirror the `Cli` parse already present in `ochroma_editor.rs:53-110`. Until that parsing lands, the `--gpu-resident` Done-When is unreachable. This does not invalidate the design; it relocates a hidden prerequisite to the front of Step 3.
- The remaining design (Steps 1, 2, 4, 5) is sound; the bit-identity-against-readback-oracle proof methodology and the `GiToRasterPack` 16→8 band-reduction design are both well-grounded.

---

# Design: Frame-loop GPU residency for the wedge passes (2026-06-07)

**Status:** Draft
**Scope:** Make the wedge compute passes (GI first) hand their output to the GPU rasteriser **inside one frame, on one device, with zero CPU readback** — turning today's "validated islands joined by PCIe round-trips" into a GPU-resident frame loop. Rank 11. Effort XL. Dimension: *The Wedge as AAA strategy* (the roadmap's critical-path spine: "#2 one device → #7/#12/#3/#8 measured resident frame → #5/#34 relight playable").
**Related:** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gaps #2, #3, #7, #8, #12), `FEATURES.md` (Spectral GI (GPU), GPU rasteriser rows).

> Honesty preface (binding): this gap does **not** invent a new kernel. Every kernel it needs already exists and is RADV-validated: `GpuGi` (spectral_gi.rs), `SplatRasterPass`/`TileAssignPass`/`RadixSortPass` (gpu/*.rs). What does **not** exist is a single `wgpu::Device` on which a GI output buffer is *bound directly* as a rasteriser input. Today `GpuGi::step()` ends in `submit → map_async → device.poll(Maintain::Wait) → Vec<GaussianSplat>`; the rasteriser then re-uploads that Vec via `create_buffer_init`. That is two PCIe crossings and one hard CPU stall per frame, on **two different devices** (`GpuGi` owns its own; `WgpuBackend` owns the present one). This spec eliminates the readback for the GI→raster seam and proves the on-GPU path is bit-identical to the readback path. The seed first-slice in the roadmap is correct in spirit but underspecifies the **band-count seam** (GI radiance is `array<f32,16>`; the raster splat buffer is `spectral:array<f32,8>`) and the **device-unification dependency** (#2) — both are designed here.

---

## 1. What we need

After this exists, a developer can do what is impossible today:

- **Render a GI-lit frame with zero splat readback.** Running `cargo run -p vox_app --bin scale_trial -- --gpu-resident --frames 30` prints a profiling line `[scale_trial] splat readback: ELIMINATED (gi->raster resident, 0 map_async/frame)` where today the only GPU-GI path (`EngineLoop::step_gi` → `GpuGi::step`) prints nothing because it always reads back to a `Vec<GaussianSplat>` (engine_loop.rs:470). **AAA bar:** UE5/Unity GPU-driven rendering keeps lit geometry resident; CPU only kicks indirect. We match that for the GI→raster seam. (NOTE: per Verification corrections, Step 3 must add argv parsing to `scale_trial.rs` before this command honors the flags.)
- **Prove the resident path is bit-identical to the readback path.** A headless run renders frame N two ways — (a) `GpuGi::step` readback then CPU-upload to raster, (b) resident GI buffer bound straight into raster — and asserts the two output textures agree to `< 1e-6` per channel. This extends the existing "headless pixel-asserted" credibility wedge (FEATURES: "headless smoke gates") from *correctness* to *residency*: the optimization provably changes nothing observable.
- **Run GI on the same device as the present surface.** `GpuContext::adapter_name()` reported by the resident path equals `WgpuBackend`'s adapter name, and a debug counter shows exactly **one** `wgpu::Instance::new` for the GI+raster pair, versus today's separate `GpuGi` device (spectral_gi.rs:519) plus the `WgpuBackend` device (wgpu_backend.rs:34). This is the hard floor (#2) under every later wedge mechanic at frame rate.
- **Eliminate the per-frame `poll(Maintain::Wait)` stall for the GI→raster handoff.** Today `GpuGi::step` calls `self.device.poll(wgpu::Maintain::Wait)` (spectral_gi.rs:627) every frame — a full CPU↔GPU sync. The resident path issues the GI compute and the raster compute into one encoder with one submit and no intervening map. **AAA bar:** async-compute overlap; no full sync mid-frame.
- **Keep the readback path alive as the oracle.** The `Vec`-returning `GpuGi::step` stays exactly as-is and becomes the *validation reference* the resident path is asserted against — preserving the 5-twin "GPU mirrors a CPU/readback oracle, bit-exact on RADV" culture (FEATURES: GI/many-light/hybrid/splat-RT/atom-budget all carry a measured-deviation twin claim).
- **Compose the wedge on the resident loop.** Because the resident GI output buffer is now the rasteriser's *input binding*, the GPU relight kernel (#5) and the resident-buffer handoff (#8) drop into the same seam without a redesign — this gap is the load-bearing junction the roadmap's Phase 2→3 transition depends on.

---

## 2. How it's gonna be (the design)

### 2.1 The seam, today vs. resident

```
TODAY (two devices, readback every frame):

  GpuGi(device A) ──compute──▶ radiance_buffer(A) ──copy──▶ readback_buffer(A)
                                                              │ map_async + poll(Wait)   ← STALL
                                                              ▼
                                                       Vec<GaussianSplat> (CPU)
                                                              │ splats_to_gpu + create_buffer_init
                                                              ▼
  GpuRasteriser(device B, the present device) ──draw──▶ surface

RESIDENT (one device, one encoder, one submit):

  GpuContext{Arc<Device>, Arc<Queue>}  (the WgpuBackend present device)
        │
        ├─ GpuGiPass.dispatch(enc) ─────────▶ radiance_buffer (STORAGE, stays on GPU)
        │                                         │  (no copy, no map, no poll)
        └─ GiToRasterPack.dispatch(enc) ────▶ raster_splat_buffer (GpuSplatFull layout)
                                                  │
           SplatRasterPass.dispatch(enc) ◀───────┘  (binds raster_splat_buffer directly)
                  │ one queue.submit(enc.finish())
                  ▼
             output_texture(Rgba32Float)
```

### 2.2 `GpuContext` — the shared device (NEW; gap #2, dependency)

`GpuGi` and `WgpuBackend` each own a private `wgpu::Device`/`Queue`. The resident path requires GI and raster to share one device so a buffer from one binds into the other. We introduce a thin handle that is **cloned**, not owned, so the existing standalone constructors keep working (provability preserved).

```rust
// NEW — crates/vox_render/src/gpu/context.rs
/// One wgpu device/queue shared across every GPU pass in a frame. Cloning is
/// cheap (Arc bump). The standalone `GpuGi::new` device-owning path is retained
/// for the headless oracle tests; the resident path uses `new_with_context`.
#[derive(Clone)]
pub struct GpuContext {
    device: std::sync::Arc<wgpu::Device>,
    queue:  std::sync::Arc<wgpu::Queue>,
    adapter_name: String,
}

impl GpuContext {
    /// Build from a live backend's device/queue (the present surface device).
    /// `device`/`queue` are cloned into Arcs; the backend keeps its own refs.
    pub fn from_backend(device: &wgpu::Device, queue: &wgpu::Queue, adapter_name: String) -> Self { /* ... */ }
    /// Build a headless context owning a fresh device (no surface) — mirrors the
    /// existing `GpuGi::new_async` adapter/device request, never panics.
    pub fn new_headless() -> Result<Self, GpuGiError> { /* ... */ }
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self)  -> &wgpu::Queue  { &self.queue }
    pub fn adapter_name(&self) -> &str { &self.adapter_name }
}
```

`WgpuBackend` already exposes `device(&self) -> &wgpu::Device` and `queue(&self) -> &wgpu::Queue` (wgpu_backend.rs:289,294) — **verified**. `from_backend` is built from those refs. *Why a clone-handle, not a borrow:* passes outlive any single call site and need `'static` ownership of the device; `Arc` is the idiomatic wgpu pattern and keeps the existing `GpuGi::new(max_splats)` device-owning constructor untouched for the oracle tests.

### 2.3 `GpuGiPass::new_with_context` and the resident output (extends existing)

`GpuGiPass` (spectral_gi.rs:268) already owns `radiance_buffer` with usage `STORAGE | COPY_SRC` (spectral_gi.rs:293) — **verified**. A `STORAGE` buffer is already a valid read-only storage *input* to a downstream pass; **no usage change is needed** for the GI→pack handoff. We add:

```rust
// extends crates/vox_render/src/spectral_gi.rs — GpuGiPass
impl GpuGiPass {
    /// Encode ONLY the GI compute pass into `encoder` (NO readback copy). The
    /// radiance result stays resident in `self.radiance_buffer`. Mirrors
    /// `dispatch` minus the `copy_buffer_to_buffer` into `readback_buffer`.
    pub fn dispatch_resident(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue:   &wgpu::Queue,
        splats_gpu: &[GpuSplatEntry],
        splat_count: u32,
        max_emitters: u32,
        sky_ambient: [f32; 16],
    );
    /// The resident GI radiance buffer (`array<array<f32,16>>`, one per splat).
    pub fn radiance_buffer(&self) -> &wgpu::Buffer { &self.radiance_buffer }
}
```

### 2.4 `GiToRasterPack` — the band-count seam (NEW; the part the seed missed)

**This is the load-bearing design decision the roadmap's one-liner glosses over.** GI radiance is 16 floats/splat (`GpuGiPass::radiance_buffer`, layout `array<array<f32,16>>`, spectral_gi_pass.wgsl:44). The tiled rasteriser's storage layout is `GpuSplatFull` with `spectral: array<f32,8>` (splat_raster.wgsl:24) plus `position_depth`, `conic`, `opacity_color`. The CPU path bridges this lossily in `splats_to_gpu` by taking the first 8 bands (gpu_rasteriser.rs:79-81) — **verified**. To stay bit-identical to that proven path, the resident packer must apply the **same 16→8 reduction the readback path applies**, on-GPU.

```rust
// NEW — crates/vox_render/src/gpu/gi_to_raster_pack.rs
/// Compute pass: read the resident GI radiance buffer (16 bands/splat) + a
/// resident geometry buffer (position/scale/opacity), and write a packed
/// `GpuSplatFull` buffer the tiled rasteriser binds directly. The 16->8 band
/// reduction is byte-for-byte the same rule as the CPU `splats_to_gpu`
/// (first 8 bands), so the resident path matches the readback path exactly.
pub struct GiToRasterPack { /* pipeline, bgl */ }

impl GiToRasterPack {
    pub fn new(ctx: &GpuContext) -> Self;
    /// Encode the pack into `encoder`. Inputs are resident GPU buffers; output
    /// is a `GpuSplatFull` storage buffer (allocated once, reused per frame).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        encoder:          &mut wgpu::CommandEncoder,
        gi_radiance_buf:  &wgpu::Buffer,   // from GpuGiPass::radiance_buffer()
        geometry_buf:     &wgpu::Buffer,   // resident GpuSplatEntry positions/opacity
        camera_buf:       &wgpu::Buffer,   // for view-space depth in position_depth.w
        out_splat_full:   &wgpu::Buffer,   // GpuSplatFull array, raster input
        splat_count:      u32,
    );
}
```

The packer also computes `position_depth.w` (view-space depth) and the EWA `conic` — currently done CPU-side in `render_indexed` (gpu_rasteriser.rs:539-543) and in the tiled path's CPU references. Keeping that math in WGSL is what makes the *whole* GI→raster chain resident.

### 2.5 `ResidentGiRaster` — the orchestrator (NEW)

```rust
// NEW — crates/vox_render/src/gpu/resident_gi_raster.rs
/// One-device, zero-readback GI -> pack -> sort -> tile -> raster chain.
/// Allocates all persistent buffers once (splat geometry, GI radiance, packed
/// GpuSplatFull, sort scratch, tile ranges, output texture) and reuses them.
pub struct ResidentGiRaster {
    ctx: GpuContext,
    gi:   GpuGiPass,
    pack: GiToRasterPack,
    sort: RadixSortPass,        // gpu/radix_sort_pass.rs:62  (verified)
    tiles: TileAssignPass,      // gpu/tile_assign.rs:25      (verified)
    raster: SplatRasterPass,    // gpu/splat_raster.rs:32     (verified)
    // persistent buffers + output_texture (SplatRasterPass::create_output_texture, verified)
}

impl ResidentGiRaster {
    pub fn new(ctx: &GpuContext, max_splats: u32, width: u32, height: u32) -> Result<Self, GpuGiError>;
    /// Upload geometry once (or when the scene changes), then per frame: encode
    /// GI -> pack -> sort -> tile-assign -> raster into ONE encoder, ONE submit.
    /// NO map_async, NO poll(Wait). Returns nothing; the lit frame lives in the
    /// output texture (read via `output_texture()` or copied to the surface).
    pub fn render_frame(&mut self, camera: &RenderCamera, hour: f32);
    pub fn output_texture(&self) -> &wgpu::Texture;
    /// Readback ONLY for headless proof (the eliminated per-frame path made
    /// explicit + one-shot at end of run).
    pub fn read_output_rgba(&self) -> Vec<[f32; 4]>;
    /// Diagnostics: 0 in resident mode, asserts the per-frame readback is gone.
    pub fn map_async_count(&self) -> u64;
}
```

`RenderCamera` (spectral.rs:4, with `view: Mat4` and `view_proj()`) — **verified**. `RadixSortPass::sort` (radix_sort_pass.rs:264), `TileAssignPass::dispatch` (tile_assign.rs:136), `SplatRasterPass::dispatch` (splat_raster.rs:169) all take `&wgpu::Device` + an `&mut wgpu::CommandEncoder` + `splat_buf: &wgpu::Buffer` — **verified**, so all four chain into one encoder on one device with no API churn.

### 2.6 Where it lives, and engine-agnostic discipline

Everything lives in `crates/vox_render/src/gpu/` (engine crate, game-agnostic — no buildings/zoning/quests touched). `GpuContext` and the resident orchestrator are pure GPU plumbing over `GaussianSplat`/`RenderCamera`, which already live in `vox_core`/`vox_render`. The editor wiring (Section 3, step 4) goes through the existing `WgpuBackend::device()/queue()` and the `OCHROMA_GI=gpu` lever — **no new game concept enters an engine crate**. Numeric inputs (`max_splats`, `width`, `height`, `hour`) are clamped exactly as the existing code does (`max_splats.max(1)` spectral_gi.rs:545; `count.min(self.max_splats)` spectral_gi.rs:401).

### 2.7 Key design decisions & rationale

- **Retain the readback `GpuGi::step` verbatim as the oracle.** It is the proven, bit-exact, RADV-validated reference (`gpu_gi_matches_cpu_step_for_large_strided_scene`, spectral_gi.rs:1017). The resident path is asserted against *its f16-quantized output*, so "resident" can never silently change a pixel. This is the 5-twin culture applied to an integration, not a kernel.
- **The 16→8 band reduction is copied, not "fixed."** Tempting to carry all 16 bands into raster — but that diverges from the proven CPU raster path and is out of scope here (it belongs with #34 reflectance split). The packer reproduces `splats_to_gpu`'s first-8 rule so the comparison is exact.
- **One encoder, one submit, no mid-frame map.** The whole point. `map_async`/`poll(Wait)` survive **only** in the end-of-run proof readback, behind `read_output_rgba()`.
- **`Arc`-cloned device, standalone constructors retained.** Zero regression risk to the 15 green waves: the device-owning `GpuGi::new` and `GpuGiPass::new(&device, ..)` paths are untouched; the resident path is purely additive.

---

## 3. How it's gonna be made (the implementation plan)

> Repo plan-template rule honored: every Done-When is an exact command + exact observable output; every test asserts a real computed value (never `assert!(x.is_some())`); each step implements **and** wires.

### Step 1 — `GiToRasterPack` + a headless bit-identical proof (the seed, made real). **[M, launchable tomorrow]**

The roadmap's seed first-slice — "feed the GpuGi compute output buffer DIRECTLY into the GPU rasteriser's input binding … proven by a headless pixel-identical comparison against the readback path" — lands here, with the band-seam designed in.

**Files (exact):**
- NEW `crates/vox_render/src/gpu/context.rs` — `GpuContext` (Section 2.2), with `new_headless()` mirroring `GpuGi::new_async` (spectral_gi.rs:515).
- NEW `crates/vox_render/src/gpu/gi_to_raster_pack.rs` + `crates/vox_render/src/gpu/gi_to_raster_pack.wgsl` — `GiToRasterPack` (Section 2.4).
- EDIT `crates/vox_render/src/spectral_gi.rs` — add `GpuGiPass::dispatch_resident` and `radiance_buffer()` accessor (Section 2.3).
- EDIT `crates/vox_render/src/gpu/mod.rs` — `pub mod context; pub mod gi_to_raster_pack;`.
- NEW `crates/vox_render/tests/resident_gi_seam_test.rs` — the proof.

**The test (real computed outcomes, mirrors the `splat_rt_gpu` house pattern, splat_rt_gpu.rs:566):**
On a fixed 1,200-splat scene (reuse the exact builder shape from `gpu_gi_matches_cpu_step_for_large_strided_scene`, spectral_gi.rs:1018-1070: a line of dark receivers + emitters at indices 0,7,1001), at `hour = 12.0`:
1. **Reference (readback):** `GpuGi::step(&scene, 12.0)` → `Vec<GaussianSplat>` → `splats_to_gpu` → bind into `SplatRasterPass::dispatch` → read output texture R/G/B → `ref_pixels: Vec<[f32;4]>`.
2. **Resident:** on one `GpuContext::new_headless()`, encode `GpuGiPass::dispatch_resident` → `GiToRasterPack::dispatch` → `SplatRasterPass::dispatch` into one encoder, one submit → read output → `res_pixels`.
3. Assert `max_abs = max over all pixels,channels of |ref - res|` satisfies `max_abs < 1e-6`, printed: `eprintln!("[resident_gi_seam] max_abs={max_abs:e} (asserting < 1e-6)")`.
4. **Anti-vacuous guard:** assert `ref_pixels` is non-trivial — `let lit = ref_pixels.iter().filter(|p| p[0]+p[1]+p[2] > 1e-4).count(); assert!(lit > 200, "reference frame must light >200 px, got {lit}")` — so a both-black pass cannot pass.
5. Graceful skip if no adapter (mirror `try_gpu`, spectral_gi.rs:896): `Err(GpuGiError::NoAdapter) => return`.

**Done-When:** `cargo test -p vox_render --test resident_gi_seam_test -- --nocapture` prints `[resident_gi_seam] max_abs=<value>e<exp> (asserting < 1e-6)` with a value below `1e-6`, **and** the line `reference frame lights <N> px` with N > 200, and the process exits 0.

### Step 2 — `ResidentGiRaster` orchestrator + one-encoder/one-submit chain. **[L]**

**Files:** NEW `crates/vox_render/src/gpu/resident_gi_raster.rs` (Section 2.5); EDIT `gpu/mod.rs`. Chains `dispatch_resident → GiToRasterPack → RadixSortPass::sort → TileAssignPass::dispatch → SplatRasterPass::dispatch` into the single encoder built in `render_frame`. Persistent buffers allocated once in `new`. `map_async_count()` returns the count of `map_async` calls issued during `render_frame` (a `Cell<u64>` bumped only in `read_output_rgba`).

**Test:** `crates/vox_render/tests/resident_gi_raster_test.rs` — render the same 1,200-splat scene for 5 consecutive `render_frame` calls; assert `rig.map_async_count() == 0` after the 5 frames (zero per-frame readback), then call `read_output_rgba()` once and assert `>200` lit pixels. Assert frame-to-frame determinism: `read_output_rgba()` after frame 2 equals after frame 5 within `1e-6` (deterministic, like `gpu_gi_is_deterministic`, spectral_gi.rs:972).

**Done-When:** `cargo test -p vox_render --test resident_gi_raster_test -- --nocapture` prints `[resident_gi_raster] map_async/frame=0 lit_px=<N>` with N>200 and exits 0.

### Step 3 — `scale_trial --gpu-resident` headless proof at scale, with the elimination line. **[M]**

> **PREREQUISITE (per Verification corrections):** `scale_trial.rs` has no argv parsing today. This step must FIRST add `--gpu-resident` and `--frames N` flag parsing (mirror the `Cli` parse in `ochroma_editor.rs:53-110`) before the binary can honor the flags. The Done-When below is unreachable until that parsing lands.

**Files:** EDIT `crates/vox_app/src/bin/scale_trial.rs` — add `--gpu-resident` and `--frames N` flag parsing. After the existing CPU sweep, if `--gpu-resident`: build a `GpuContext::new_headless()`, construct `ResidentGiRaster` over the selected subset (reuse `sel.select`, atom_budget.rs:246, into the existing `Selection`, atom_budget.rs:168), run N frames, then `read_output_rgba` once, count non-black, and print the profiling line.

**Done-When:** `cargo run -p vox_app --bin scale_trial -- --gpu-resident --frames 30` prints, in order: `[scale_trial] gpu-resident: <N> frames on <adapter_name>`, then `[scale_trial] splat readback: ELIMINATED (gi->raster resident, 0 map_async/frame)`, then `[scale_trial] resident raster non_black_px=<X>/<T> (<P>%)` with P>10.0, and exits 0. (The `>10%` non-black bar matches the existing CPU assertion in scale_trial.rs:416.)

### Step 4 — Wire the resident path into the editor viewport behind `OCHROMA_GI=gpu`. **[L]**

**Files:** EDIT `crates/vox_app/src/bin/ochroma_editor.rs` — in the backend-bring-up (after `WgpuBackend` is built), construct `GpuContext::from_backend(backend.device(), backend.queue(), adapter_name)` and, when `OCHROMA_GI=gpu`, a `ResidentGiRaster`; replace the Viewport tab's CPU `SoftwareRasteriser` upload (viewport.rs:120-121) with a blit of `ResidentGiRaster::output_texture()` into the egui viewport texture. CPU `SoftwareRasteriser` stays the permanent fallback (mirror the `GpuGi` graceful-fallback rule, engine_loop.rs:285-292). This goes through the existing `--frames N --shot` proof harness (ochroma_editor.rs:265 `capture_shot`).

**Done-When:** `OCHROMA_GI=gpu cargo run -p vox_app --bin ochroma_editor -- --frames 2 --shot /tmp/resident.png` prints `[ochroma_editor] viewport: resident GI->raster on <adapter_name> (present device)` with `<adapter_name>` equal to the backend's adapter, writes `/tmp/resident.png` with a non-background fraction > 0 (reusing the existing `non_background_fraction` receipt, ochroma_editor.rs:379), and exits 0. With `OCHROMA_GI` unset the run still succeeds on the CPU `SoftwareRasteriser` path (no panic).

### Step 5 — Wire `EngineLoop` to prefer the resident output when a `GpuContext` is present. **[M]**

**Files:** EDIT `crates/ochroma_engine/src/engine_loop.rs` — add `EngineLoop::use_resident_gi_raster(&mut self, ctx: &GpuContext)` that swaps `GiBackend::Gpu` for a resident variant whose `step_gi`-equivalent returns the output texture handle rather than a `Vec`; keep `step_gi`'s `Vec` signature for over-capacity/CPU frames (engine_loop.rs:457). Surface `gi_backend() == "gpu-resident"`.

**Done-When:** `cargo test -p ochroma_engine --test engine_loop_integration -- --nocapture` includes a new `resident_gi_backend_reports_zero_readback` asserting that after `use_resident_gi_raster`, `gi_backend()` returns `"gpu-resident"` and a 30-frame drive issues zero per-frame `map_async` (assert a counter == 0), exiting 0.

---

## 4. How it fits (integration + dependencies)

**Depends on (named gaps):**
- **#2 Unify on one `GpuContext`** — this spec *defines and ships* `GpuContext` (Section 2.2) as Step 1, so #2 is absorbed into the critical path here rather than being a separate prerequisite. `WgpuBackend::device()/queue()` (verified) make it cheap.
- **#8 Resident buffers for GPU→GPU handoff** — Step 2's persistent-buffer chain *is* #8 for the GI→raster seam; #8's general `step_resident` is the same pattern generalized.
- **#3 Wire the GPU-driven tiled rasteriser** — this spec is the first real caller of `SplatRasterPass`/`TileAssignPass`/`RadixSortPass` outside their own unit tests (confirmed: only in-crate test references exist), so it *completes* #3's wiring for the GI path.

**What depends on this:**
- **#5 GPU runtime spectral-relight kernel** and **#34 reflectance/emission split** — relight becomes a compute pass inserted between `dispatch_resident` and `GiToRasterPack` on the *same encoder*; the resident seam is its host.
- **#7 GPU timestamps** — the single encoder built in `render_frame` is exactly where `timestamp_writes` (currently `None` everywhere, e.g. spectral_gi.rs:418, splat_raster begin_compute_pass) get attached to give per-pass ms.
- **#12 render graph as GPU executor** — `ResidentGiRaster::render_frame` is a hand-rolled 5-node graph; #12 generalizes it to a `trait GpuPass{record(enc,res)}`.
- **#20/#30 residency manager + scale** — the persistent-buffer allocation in `ResidentGiRaster::new` is the seed of the buffer pool.

**Composes with existing systems:** `WgpuBackend` (present device), `EngineLoop::step_gi` + the `OCHROMA_GI` lever (the GI backend selector, unchanged for CPU/over-capacity frames), `AtomBudgetSelector::select`/`Selection` (the LOD subset feeding the resident scene), the editor `--frames N --shot` proof harness, and the CPU `SoftwareRasteriser` permanent fallback.

**Must NOT break:**
- **The 15-green-wave invariant** (history shows 15 `wave-N` closeouts; the latest is `035dd43 wave-14 closeout`). Everything is *additive*: `GpuGi::step`, `GpuGiPass::new/dispatch`, `GpuRasteriser`, and all standalone constructors are untouched; their existing tests (`gpu_gi_matches_cpu_step_for_large_strided_scene`, `gpu_gi_is_deterministic`, `gpu_gi_falls_back_on_impossible_limits`) must still pass verbatim.
- **Both-config builds** — the resident path uses no `forge-native`/`spectra-native` feature; it must compile and pass with default features (the CI gate) *and* with `--all-features`.
- **The no-panic shell rule** — `GpuContext::new_headless` and `ResidentGiRaster::new` return `Result<_, GpuGiError>` and the editor falls back to `SoftwareRasteriser` on `Err`, never `unwrap`. Mirror the existing `OCHROMA_GI=gpu` graceful-fallback (engine_loop.rs:229-238).

**4-phase sequencing:** this is the heart of **Phase 2** ("the integrated GPU frame loop") — specifically the merge point of #2/#3/#8 — and the **prerequisite gate** into Phase 3 (relight playable). Critical-path spine, verbatim: "#2 (one device) → #7/#12/#3/#8 (measured resident frame) → #5/#34 (relight playable)". This spec delivers the GI→raster slice of that resident frame.

**Cross-gap seams:** (a) the `GpuContext` clone-handle is the universal device seam every other GPU gap binds to; (b) the single `render_frame` encoder is the timestamp seam (#7) and the render-graph seam (#12); (c) the `GiToRasterPack` insertion point is the relight seam (#5/#34).

---

## Surprises & advantages

Grounded discoveries that make this **cheaper than the roadmap implies**:

1. **`WgpuBackend` already exposes `device()` and `queue()`** (wgpu_backend.rs:289,294). Gap #2's hardest-sounding part — getting at the present device — is already a public accessor. `GpuContext::from_backend` is a 3-line constructor, not a refactor of 17 `Instance::new` sites. The device unification can be done *incrementally per seam* rather than as a big-bang rewrite.

2. **`GpuGiPass::radiance_buffer` is already `STORAGE | COPY_SRC`** (spectral_gi.rs:293). A `STORAGE` buffer is *already* a legal read-only storage input to a downstream pass — so the GI output needs **no usage-flag change** to be bound into the packer. The readback path's `COPY_SRC` + separate `readback_buffer` (spectral_gi.rs:302) is purely additive; deleting the per-frame copy is subtractive and safe. The resident output buffer the seed asks for *already exists*; it's just always copied out.

3. **The readback path is a ready-made oracle.** Because `GpuGi::step` already returns a bit-exact, RADV-validated `Vec<GaussianSplat>` (proven by `gpu_gi_matches_cpu_step_for_large_strided_scene`), the resident path gets its `< 1e-6` comparison reference for free — no new CPU oracle to write. This is the 5-twin culture turned into a *zero-cost* residency proof, which is itself a wedge artifact ("we can prove our optimizations are observably identical").

4. **All three tiled passes already take `&mut wgpu::CommandEncoder` and a raw `splat_buf: &wgpu::Buffer`** (radix_sort_pass.rs:264, tile_assign.rs:136, splat_raster.rs:169). They were *designed* for encoder-chaining and direct buffer binding — the one-encoder/one-submit resident chain needs **zero signature changes** to those passes. The roadmap calls #3 "wired into nothing"; in fact the wiring surface is already encoder-shaped, so wiring is composition, not surgery.

5. **First-mover angle:** no RGB engine has a "16-band spectral GI buffer bound resident into a tile rasteriser, proven pixel-identical to its readback oracle." Shipping the residency *with its bit-identity proof* is a defensible provability claim competitors structurally cannot make (their GI carries no spectral band buffer to bind).

6. **One honest caveat (not an advantage):** the band-count seam (16→8) is a real gotcha the seed first-slice omits. It is *not* a blocker — copying `splats_to_gpu`'s first-8 rule keeps bit-identity — but anyone implementing the seed verbatim without `GiToRasterPack` would either bind a mismatched buffer (validation error) or carry all 16 bands and diverge from the proven raster. Flagging it here is what keeps Step 1 launchable tomorrow without a mid-task redesign.
