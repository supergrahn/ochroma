> **Adversarial verification:** SOUND. The skeptic confirmed every grounding reference resolves to real code at (or within 0 lines of) the claimed location: `wgpu_backend.rs:64-71` (`request_device` with `Features::empty()`), `:289-306` (the `device()/queue()/surface_format()/surface()` accessors), `splat_raster.rs:26` (`struct SplatRasterPass`), `:136` (`create_output_texture`), `:169-224` (`dispatch(&self, device, …)`), the `ochroma_editor.rs:248` present-pass `timestamp_writes: None`, and the `OCHROMA_GI=gpu` fallback idiom at `engine_loop.rs:226-250`. The wgpu-24.0.5 API surface (`Features::TIMESTAMP_QUERY`, `get_timestamp_period`, `resolve_query_set`, `RenderPassTimestampWrites`/`ComputePassTimestampWrites`, `QUERY_RESOLVE`, the 256-byte resolve alignment) is all present in the vendored crate. No issues flagged — the spec ships as written.

## 0. Header

**Status:** Draft
**Scope:** Extend Ochroma's provability culture from correctness to *performance* by measuring real GPU pass cost with `wgpu::Features::TIMESTAMP_QUERY`, surfacing per-pass GPU-ms in the editor status bar, and asserting a non-zero, bounded delta in a headless test — the first ms-asserted gate the later GPU-loop gaps (#3, #8, #20) can regress against.
**Related:** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #7, GPU Loop dim); depends on #2 (`GpuContext` device unification); unblocks measured Done-Whens on #3 (tiled rasterizer), #8 (resident buffers), #20 (residency manager).
**Roadmap rank:** #7 of 36 · Dimension: Runtime Performance & the GPU Frame Loop · Effort: M · Score 66/90.

> Grounding honesty (binding): the roadmap's seed says "instrument the splat-raster pass." Verified against the code, the editor's viewport renders through a **CPU `SoftwareRasteriser`** (ochroma_editor.rs:15 docstring), and the GPU `SplatRasterPass` (vox_render/src/gpu/splat_raster.rs) has **zero callers outside its own tests** — it is not wired into any frame. The *only* GPU render pass the editor records per frame is the **egui present pass** in `EditorHost::paint` (ochroma_editor.rs:237–253, which already carries `timestamp_writes: None` at line 248). So this spec instruments the pass that actually runs today (the egui present pass → labelled `present`), and ships a **reusable `GpuTimers` harness** that the standalone `SplatRasterPass` is wrapped with in a headless test (labelled `raster`). When #3 wires the tiled rasterizer into the viewport, that same harness produces the editor `raster:` line with zero new infrastructure. This keeps every Done-When true *today* and forward-compatible.

---

## 1. What we need

The capability: a developer can **measure and defend a per-pass GPU-millisecond budget**, headlessly and on-screen, instead of guessing from a single wall-clock number that conflates CPU sort + upload + GPU + present.

Concrete observables after this lands:

- Running `cargo run -p vox_app --bin ochroma_editor -- --frames 30` prints, on stdout, a per-pass GPU-ms line each measured frame, e.g. `[ochroma_editor] gpu: present 0.42 ms` — a real resolved GPU timestamp delta, not a CPU timer.
- The editor status bar (vox_app/src/shell/mod.rs:1329 `status_bar`) shows a live `GPU: present X.X ms` segment, updated every frame from the resolved query.
- A headless test (`cargo test -p vox_render gpu_timers_raster_pass_nonzero`) wraps the real `SplatRasterPass` dispatch in `GpuTimers`, resolves the timestamp pair, and asserts the delta is **> 0.0 and < 50.0 ms** for a fixed 4k-splat / 256×256 scene — an ms-asserted gate, the performance analogue of the existing headless pixel gates.
- On any adapter that lacks `TIMESTAMP_QUERY` (or where device creation with the feature fails), the editor prints `gpu: timestamps unavailable (CPU frame timer only)` exactly once and continues — **no panic, no behavioral change** (mirrors the `OCHROMA_GI=gpu` graceful-fallback pattern in ochroma_engine/src/engine_loop.rs:226–250).
- A second headless test asserts the **fallback path** is non-panicking and returns `None` when the feature is absent (forced via a `GpuTimers::disabled()` constructor), so the no-GPU CI lane stays green.

**Why it is blocking (GPU Loop dimension, roadmap §2 + gap #7):** "you cannot hit or defend a 16.6 ms budget you cannot measure per-pass; today telemetry is wall-clock `frame_time_ms` only." The roadmap's critical-path spine routes through "**measured resident frame**" (§5 Phase 2) — #3, #8, #20 all have Done-Whens that say "prints a GPU-timestamp-measured raster cost (depends on #7)." Without this gap those Done-Whens are unverifiable. This is the capability that turns "headless pixel-asserted" into "headless ms-asserted" — provability is the wedge (roadmap §1.3), and this extends it to performance.

---

## 2. How it's gonna be (the design)

### 2.1 Where it lives, and why

A single new module **`crates/vox_render/src/gpu/gpu_timing.rs`** (registered in `crates/vox_render/src/gpu/mod.rs` alongside the existing `pub mod splat_raster;` etc.). It is **game-agnostic** (a wrapper over `wgpu::QuerySet` + a resolve buffer) so it belongs in the engine crate `vox_render`, never in `vox_app`. The editor (game layer) *consumes* it; the engine *provides* it. This honors the CLAUDE.md rule: "Engine crates must NEVER contain game-specific concepts."

The HUD wiring (status-bar segment + stdout line) lives in **`vox_app`** (the game/editor layer): a `pub last_gpu_pass_ms: Option<(&'static str, f32)>` field on `EditorShell` and a new branch in `EditorHost::paint`.

### 2.2 The `GpuTimers` type (NEW — full proposed signature)

`TIMESTAMP_QUERY` requires three things that must move together: (a) the **feature enabled at device creation**, (b) a `QuerySet` of `QueryType::Timestamp`, (c) a `QUERY_RESOLVE | COPY_SRC` buffer to `resolve_query_set` into, plus a `MAP_READ` readback buffer. `GpuTimers` owns the back half (b/c) and is `None`-safe; the device-feature half (a) is decided once by `GpuContext` / `WgpuBackend` and queried via `adapter.features().contains(...)`.

```rust
// crates/vox_render/src/gpu/gpu_timing.rs  — NEW, game-agnostic

/// A small ring of GPU timestamp query pairs for measuring per-pass GPU cost.
/// `None`-safe: if the device lacks TIMESTAMP_QUERY, every method is a no-op and
/// `resolve_ms` returns `None`. NEVER panics on a missing feature — mirrors the
/// `GpuGiError` graceful-fallback contract (spectral_gi.rs:441).
pub struct GpuTimers {
    enabled: bool,
    period_ns: f32,                  // queue.get_timestamp_period() — ns per tick
    query_set: Option<wgpu::QuerySet>,
    resolve_buf: Option<wgpu::Buffer>,   // QUERY_RESOLVE | COPY_SRC, 8 bytes * 2*pairs
    readback_buf: Option<wgpu::Buffer>,  // MAP_READ | COPY_DST
    pairs: u32,                      // how many begin/end pairs this ring holds
}

impl GpuTimers {
    /// Build a timer ring sized for `pairs` begin/end timestamp pairs.
    /// Returns a disabled (no-op) instance if `features` lacks TIMESTAMP_QUERY,
    /// so callers never branch on availability themselves.
    pub fn new(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        features: wgpu::Features,
        pairs: u32,
    ) -> Self;

    /// An explicitly disabled timer — used by the no-GPU fallback test and by
    /// the CPU-only CI lane. resolve_ms() always returns None.
    pub fn disabled() -> Self;

    pub fn is_enabled(&self) -> bool;

    /// Build the `RenderPassTimestampWrites` for pair `slot` to hand into a
    /// `RenderPassDescriptor.timestamp_writes`. Returns `None` when disabled,
    /// so the caller writes `timestamp_writes: timers.render_writes(0)`.
    pub fn render_writes(&self, slot: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>>;

    /// Same for a compute pass (used to wrap SplatRasterPass::dispatch).
    pub fn compute_writes(&self, slot: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>>;

    /// Record the resolve into `encoder` AFTER the measured pass(es) close,
    /// then copy the resolve buffer into the mappable readback buffer.
    /// No-op when disabled.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder);

    /// Map the readback buffer (caller has already submitted + the device has
    /// progressed past the submit), compute (end-begin)*period for `slot`, and
    /// return milliseconds. Returns `None` when disabled OR if the slot's two
    /// timestamps are equal/zero (unwritten). Unmaps before returning.
    pub fn resolve_ms(&self, device: &wgpu::Device, slot: u32) -> Option<f32>;
}
```

**Verified wgpu-24.0.5 API surface this rests on** (all confirmed present in `~/.cargo/.../wgpu-24.0.5`):
- `wgpu::Features::TIMESTAMP_QUERY` (wgpu-types-24.0.0/src/lib.rs:400).
- `wgpu::Queue::get_timestamp_period() -> f32` (wgpu-24.0.5/src/dispatch.rs:237, backend wgpu_core.rs:1817).
- `wgpu::Device::create_query_set(&QuerySetDescriptor)`; `QueryType::Timestamp` (wgpu-types lib.rs:7504/7532).
- `wgpu::CommandEncoder::resolve_query_set(...)` (dispatch.rs:334).
- `wgpu::RenderPassTimestampWrites { query_set, beginning_of_pass_write_index: Option<u32>, end_of_pass_write_index: Option<u32> }` (render_pass.rs:508–514) and `ComputePassTimestampWrites` with identical fields (compute_pass.rs:151–157). Each pass's `timestamp_writes` field is `Option<…TimestampWrites>` (render_pass.rs:574, compute_pass.rs:175) — so a `None` from `render_writes`/`compute_writes` drops in exactly where today's code has `timestamp_writes: None`.
- `wgpu::BufferUsages::QUERY_RESOLVE` (wgpu-types lib.rs:5517); `QUERY_RESOLVE_BUFFER_ALIGNMENT = 256` (lib.rs:57) — the resolve buffer must be 256-aligned.

### 2.3 Enabling the feature (the device-creation half)

The feature must be requested at `request_device` time. Today every site uses `required_features: wgpu::Features::empty()` (verified: wgpu_backend.rs:68, spectral_gi.rs:536, +9 more). The change is **additive and probed, never forced**:

```rust
// At device bring-up (WgpuBackend::new_async today; GpuContext::new once #2 lands):
let wanted = wgpu::Features::TIMESTAMP_QUERY;
let supported = adapter.features().contains(wanted);
let required_features = if supported { wanted } else { wgpu::Features::empty() };
// ...DeviceDescriptor { required_features, .. }
```

Because #2 (`GpuContext`) is not yet present, **Step 1 lands this against the existing `WgpuBackend`** (which already exposes `device()`/`queue()`/`surface_format()` at wgpu_backend.rs:289–301 and an adapter at creation time). When #2 lands, the identical probe moves into `GpuContext::new`, and `GpuTimers::new(ctx.device(), ctx.queue(), ctx.features(), …)` is the call site — a one-line retarget. The seam is `wgpu::Features` passed in, so `GpuTimers` never needs to know who created the device.

### 2.4 Data flow (editor frame)

```
EditorHost::paint(view, tris, …)                         [ochroma_editor.rs:196]
  ├─ encoder = device.create_command_encoder(...)
  ├─ begin_render_pass {
  │     timestamp_writes: timers.render_writes(0)   ◄── was: None  (line 248)
  │  } … egui_renderer.render(...)                  ← the real GPU work
  ├─ timers.resolve(&mut encoder)                   ◄── NEW: resolve+copy to readback
  ├─ queue.submit([encoder.finish()])
  ├─ surface_tex.present()
  └─ if let Some(ms) = timers.resolve_ms(device, 0) {   ◄── NEW
         self.shell.set_gpu_pass_ms("present", ms);      → status bar (mod.rs:1329)
         println!("[ochroma_editor] gpu: present {ms:.2} ms");  → --frames stdout
     }
```

The resolve buffer is read **one submit late** is *not* required here: the editor present path already serializes per frame (it acquires, paints, presents, then `request_redraw`). For the resolve-and-map we call `device.poll(wgpu::Maintain::Wait)` inside `resolve_ms` exactly as `GpuGi::step` does (spectral_gi.rs:627) — acceptable because the editor is not yet a pipelined GPU loop (gap #8 removes that stall later; until then a single poll per frame in proof mode is the same cost the GI path already pays). The headless test path uses the same poll. **Design decision recorded:** we accept one `poll(Wait)` per measured frame now; #8's resident path will switch to a deferred N-frame-latent readback ring (the `pairs > 1` capacity already in the struct anticipates this — slot rotation, no API change).

### 2.5 Mirroring an oracle / RADV validation

This gap has no CPU "oracle" in the bit-exact sense (a timestamp is inherently a measurement, not a deterministic compute). The provability bar it meets instead is **bounded-and-non-zero on real hardware (RADV)**: the headless test asserts `0.0 < ms < 50.0` for a fixed scene — a real computed outcome, not `assert!(x.is_some())`. The CPU-fallback test asserts the no-feature path returns `None` without panic. Together these are the two-sided contract (works on RADV; degrades on no-GPU CI) that every other GPU twin in the repo follows.

---

## 3. How it's gonna be made (the implementation plan)

### Step 1 — `GpuTimers` + headless raster-pass ms gate (S) — LAUNCHABLE TOMORROW

**Files:**
- NEW `crates/vox_render/src/gpu/gpu_timing.rs` — the `GpuTimers` struct from §2.2.
- EDIT `crates/vox_render/src/gpu/mod.rs` — add `pub mod gpu_timing;`.
- EDIT `crates/vox_render/src/gpu/gpu_timing.rs` (test module) — the two headless tests below.

**Implement AND wire:** `GpuTimers::new/disabled/render_writes/compute_writes/resolve/resolve_ms`, and immediately wire it into a test that drives the **real existing** `SplatRasterPass::dispatch` (splat_raster.rs:169) — so the harness is exercised against a real pass, not a toy, in the same step.

The test builds a standalone wgpu device **with `TIMESTAMP_QUERY` requested** (reusing the adapter-probe in §2.3; if the RADV adapter lacks it the test asserts the fallback `None` branch instead and prints a skip line — never a false green), creates a 256×256 output via `SplatRasterPass::create_output_texture`, fills the minimum buffers (camera uniform, 4096 `GpuSplatFull` splats, sorted_vals, tile_ranges — same shapes the existing splat_raster tests already construct), wraps the compute pass with `timers.compute_writes(0)` by extending `dispatch` to take an `Option<ComputePassTimestampWrites>` (a 1-line signature add, default `None` keeps all current callers compiling), resolves, polls, and reads the ms.

**Done When (exact):**
`cargo test -p vox_render gpu_timers_raster_pass_nonzero -- --nocapture` prints `raster pass measured: <X> ms (period <P> ns)` with `X` in `(0.0, 50.0)` and asserts:
```rust
let ms = timers.resolve_ms(&device, 0).expect("timestamp resolved on RADV");
assert!(ms > 0.0,  "GPU raster pass must take measurable time, got {ms}");
assert!(ms < 50.0, "4k-splat 256x256 raster must be < 50ms, got {ms}");
```
and `cargo test -p vox_render gpu_timers_disabled_returns_none` asserts:
```rust
let t = GpuTimers::disabled();
assert!(!t.is_enabled());
assert!(t.resolve_ms(&device, 0).is_none());   // no panic, real None
assert!(t.render_writes(0).is_none());
```
(Real computed outcomes; no `is_some()` stub. `cargo build -p vox_render` confirms the additive `dispatch` signature didn't break the existing splat_raster callers.)

### Step 2 — Enable the feature at editor device bring-up + stdout line (S)

**Files:** EDIT `crates/vox_render/src/gpu/wgpu_backend.rs` (probe + store `features` + accessor `pub fn features(&self) -> wgpu::Features`), EDIT `crates/vox_app/src/bin/ochroma_editor.rs` (`EditorHost` gains `timers: GpuTimers`, built in `resumed()` from `backend.device()/queue()/features()`; `paint` swaps `timestamp_writes: None` (line 248) → `self.timers.render_writes(0)`, adds `resolve` + `resolve_ms` + the `println!`).

**Implement AND wire:** the feature is probed (not forced); on absence the editor prints `gpu: timestamps unavailable (CPU frame timer only)` once and `timers` is `disabled()`.

**Done When (exact):** `cargo run -p vox_app --bin ochroma_editor -- --frames 30` exits 0 and stdout contains at least 25 lines matching `^\[ochroma_editor\] gpu: present [0-9]+\.[0-9]+ ms$` (verified with `... --frames 30 2>/dev/null | grep -c 'gpu: present'` returning ≥ 25 on the RADV box); on a forced-no-feature run (env `OCHROMA_NO_TIMESTAMP=1` honored by the probe) stdout contains exactly one `gpu: timestamps unavailable` line and the binary still exits 0.

### Step 3 — Status-bar HUD segment (S)

**Files:** EDIT `crates/vox_app/src/shell/mod.rs` — add `pub last_gpu_pass_ms: Option<(&'static str, f32)>` to `EditorShell` (init `None` at mod.rs:313), a `pub fn set_gpu_pass_ms(&mut self, pass: &'static str, ms: f32)` setter, and render it in `status_bar` (mod.rs:1329) as a right-aligned segment beside the existing "N things in the world".

**Implement AND wire:** `EditorHost::paint` calls `self.shell.set_gpu_pass_ms("present", ms)` after `resolve_ms`; the status bar reads the field.

**Done When (exact):** `cargo run -p vox_app --bin ochroma_editor -- --frames 2 --shot /tmp/hud.png` writes a PNG; a headless assertion via the existing `shell_snapshot` proof path (or a new `cargo test -p vox_app status_bar_shows_gpu_ms`) constructs an `EditorShell`, calls `set_gpu_pass_ms("present", 0.42)`, runs one `ui()` frame headlessly, and asserts the rendered status-bar text contains the substring `GPU: present 0.42 ms` (real string assertion on actual egui output, the same technique `shell_snapshot` uses). Human-visible: opening the editor shows a live `GPU: present X.X ms` in the bottom bar that changes frame to frame.

### Step 4 — Retarget onto `GpuContext` when #2 lands (S, sequencing-gated)

**Files:** EDIT `crates/vox_render/src/gpu/` (GpuContext gains `pub fn features(&self) -> wgpu::Features`), retarget the editor's `GpuTimers::new` call from `backend.*` to `ctx.*`.

**Done When (exact):** after #2 is merged, `cargo run -p vox_app --bin ochroma_editor -- --frames 5` still prints the `gpu: present X.X ms` lines AND (per #2's own Done-When) the GI and present share one adapter — verified by grepping the run log for a single adapter name. This step is a no-behavior-change retarget; it exists so #7 does not block on #2 (Steps 1–3 ship against `WgpuBackend` first).

### Step 5 (forward seam, documented not built here) — `raster:` line once #3 wires the tiled path

When gap #3 wires `TiledSplatRenderer` into the viewport, its compute passes take `timers.compute_writes(slot)` (slots 1..N) and the editor prints `gpu: raster X.X ms` from the same harness. **No new code in this gap** — the `pairs > 1` capacity and `compute_writes` already exist from Step 1. This is the cross-gap seam #3's Done-When depends on.

---

## 4. How it fits (integration + dependencies)

**Depends on:**
- **#2 (Unify on one `GpuContext`)** — *soft* dependency. The clean home for the feature-probe and the device/queue handed to `GpuTimers` is `GpuContext`. But because `WgpuBackend` already exposes `device()/queue()` and is created with an adapter in scope, **Steps 1–3 land before #2** and Step 4 retargets. This deliberately decouples #7 from #2's schedule (roadmap §5 lists both in Phase 1/2; this ordering lets #7 start tomorrow).

**Depended on by:**
- **#3 (tiled rasterizer as viewport path)** — its Done-When: "prints a GPU-timestamp-measured raster cost (depends on #7)." Satisfied by Step 5's seam.
- **#8 (resident buffers)** — its Done-When needs to prove the resident path doesn't *regress* GPU cost; `GpuTimers` is the measuring stick. The `pairs > 1` ring is the latency-tolerant readback #8 will use to avoid the per-frame `poll(Wait)`.
- **#20 (GPU residency manager)** — same: budget defense requires per-pass ms.

**Composes with existing systems:**
- **`SplatRasterPass`** (splat_raster.rs) — wrapped, via the additive `Option<ComputePassTimestampWrites>` arg, with no change to its compute logic.
- **The egui present pass** (ochroma_editor.rs paint) — the live editor pass actually measured today.
- **`EditorShell` status bar** (mod.rs:1329) — the HUD surface, alongside the entity-count segment.
- **The `OCHROMA_GI=gpu` fallback idiom** (engine_loop.rs:226–250) — copied verbatim in spirit for the no-feature path (probe, one eprintln/println, degrade, never panic).

**What it must NOT break:**
- **The 11-consecutive-green-gate invariant** — the new tests must pass on RADV *and* the no-GPU lane; the `disabled()` test guarantees the CPU-only CI path stays green. The feature is *probed*, never required, so `WgpuBackend::new` cannot start failing on adapters that lack it.
- **Both-config builds** — the `dispatch` signature change is additive with a `None` default behavior; the `game-ui` / default feature matrix is untouched (no new cargo features introduced).
- **The no-panic shell rule** — `GpuTimers` returns `Option`/no-ops on every failure; `resolve_ms` unmaps before returning even on the error path; no `unwrap` on a query result reaches a shipped binary (the test's `.expect` is test-only).
- **`frame_time_ms` wall-clock telemetry** — left intact; the GPU-ms line is *additive*, so the existing CPU timer and any smoke gate reading it are unaffected.

**4-phase sequencing (roadmap §5):** this is **Phase 2** ("The integrated GPU frame loop + measurement"), explicitly first in that phase ("#7 GPU timestamps (so every later GPU gap has a measured Done-When)"). It is the measurement floor under #12/#3/#8.

**Cross-gap seams:** (a) the `wgpu::Features` argument to `GpuTimers::new` is the seam to #2; (b) `compute_writes(slot)` with `pairs > 1` is the seam to #3; (c) the readback ring is the seam to #8's deferred-readback resident loop.

---

## Surprises & advantages

Grounded discoveries that make this cheaper / stronger than the roadmap's effort-M estimate implies:

1. **The editor's present pass is already a single, isolated `begin_render_pass` with `timestamp_writes: None` literally sitting at the injection point** (ochroma_editor.rs:248). Adding measurement is a one-field swap on an existing pass, not new pass plumbing — Step 2 is genuinely a few lines.

2. **A proven graceful-fallback idiom already exists and is battle-tested** (`OCHROMA_GI=gpu` → CPU, engine_loop.rs:226–250, plus `GpuGiError`/`new_failing_for_test`, spectral_gi.rs:441/502). `GpuTimers` copies a known-good shape rather than inventing a fallback contract — and `GpuGi::new_failing_for_test` is a ready template for the `disabled()` no-panic test.

3. **The `poll(Maintain::Wait)` + `map_async` + readback dance is already written and verified** in `GpuGi::step` (spectral_gi.rs:621–645). The timestamp readback is the *same* pattern on a tiny 8-byte-per-query buffer — copy-adapt, don't author from scratch.

4. **`WgpuBackend` already exposes `device()`, `queue()`, `surface_format()`** (wgpu_backend.rs:289–306) and holds an adapter at creation — so #7 does **not** actually need #2 to start, despite the roadmap listing the dependency. This is the biggest schedule advantage: the seed says "depends on #2" but the verified code lets Steps 1–3 land independently and retarget later. The gap can start tomorrow with no blocker.

5. **First-mover provability angle:** no competitor markets "headless, ms-asserted per-pass GPU budget gates in CI." Combined with the existing 5 bit-exact GPU twins, an ms-asserted gate turns "we're fast" into "here is the test that fails if a commit regresses the raster pass past X ms" — provability extended from correctness to performance, exactly the roadmap §1.3 wedge, for ~M effort.

6. **The `SplatRasterPass::dispatch` already takes a `&mut wgpu::CommandEncoder`** (splat_raster.rs:172) and builds its own `ComputePassDescriptor` with `timestamp_writes: None` (line 219) — so wrapping it is a single additive parameter, and the *real* GPU pass (not a stand-in) is what the headless ms gate measures. The harness is validated against production code in the very step that creates it.

7. **Honest scope correction surfaced by grounding:** the seed's "instrument the splat-raster pass" would have produced a *dead* editor line (the splat-raster pass isn't in the editor frame). Catching this means the shipped Done-Whens are real today (`present`) and the `raster` line arrives automatically when #3 wires the path — no rework, and no false "it works" claim.
