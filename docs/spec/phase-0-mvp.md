# Phase 0 — MVP Specification

**Goal:** A single, demonstrable prototype. One 1km tile, one spectral Gaussian asset loaded
from disk, rendered at 60fps, with the ability to plop a second instance of the same asset.
Nothing else. This validates the Rust↔CUDA↔Spectral pipeline before any simulation or neural
work begins.

---

## Scope Boundary

### In scope (Phase 0)
- Crate layout and workspace
- `.vxm` file format v0.1 (fixed, minimal)
- CUDA kernel bridge from Rust (`cudarc`)
- Spectral → RGB tonemapping (simplified, 4-band minimum)
- Basic rasterizer: depth sort → alpha blend Gaussians to screen
- Vulkan/wgpu swapchain and command buffer recording
- Plop a second instance of the same asset (no UI, just a hardcoded offset)
- Asset loading from disk with zstd decompression
- Frame timing / profiling via `puffin`

### Explicitly out of scope for Phase 0
- ECS (defer to Phase 1)
- SVO / spatial hashing (defer to Phase 1)
- Neural anything (defer to Phase 2+)
- Simulation, agents, weather, physics
- Editor UI
- LOD / mip-mapped splats
- LWC (Large World Coordinates) — single tile stays within f32 range
- Multiplayer, headless mode, modding

---

## Crate Structure

```
ochroma/
├── Cargo.toml               # workspace root
├── crates/
│   ├── vox_core/            # shared math, types, error types, UUID
│   ├── vox_render/          # CUDA kernels, spectral pipeline, Vulkan/wgpu swapchain
│   ├── vox_data/            # .vxm format: read, write, validate
│   └── vox_app/             # binary entry point, window, input, plop demo
```

`vox_sim` and `vox_nn` are reserved names but not created in Phase 0.

---

## `.vxm` Format v0.1

Binary layout (little-endian):

```
[Header: 64 bytes]
  magic:         [u8; 4]   = b"VXMF"
  version:       u16       = 1
  flags:         u16       = 0 (reserved)
  asset_uuid:    [u8; 16]  (UUID v4)
  splat_count:   u32
  material_type: u8        (0=Generic, 1=Concrete, 2=Glass, 3=Vegetation, 4=Metal, 5=Water)
  _pad:          [u8; 23]

[Splat array: splat_count × 52 bytes each, zstd-compressed]
  position:      [f32; 3]  (x, y, z) relative to asset origin
  scale:         [f32; 3]  (half-axes of Gaussian ellipsoid)
  rotation:      [i16; 4]  (unit quaternion, each component mapped to [-1,1] via /32767.0)
  opacity:       u8        (0–255, linear)
  _pad:          u8
  spectral:      [f16; 8]  (8 spectral band coefficients, 380–720nm at 40nm intervals)
```

**Total per-splat:** 52 bytes uncompressed. A 100k-splat building ≈ 5MB uncompressed,
~1.5–2MB zstd-compressed.

**Notes:**
- `rotation` stored as 4× i16 (quantized quaternion). Decode: `q[i] = raw[i] as f32 / 32767.0`.
- `spectral` stored as f16 — sufficient for Phase 0. Phase 1 may expand to PCA-compressed SPDs.
- No LOD data in v0.1. Single-resolution only.

---

## Spectral Pipeline (Simplified)

Phase 0 uses a **4-band approximation** (R, G, B, NIR at 460nm, 530nm, 610nm, 720nm) even
though the format stores 8 bands. The extra 4 bands are loaded but not used in the rasterizer
until Phase 1 expands the CUDA kernel.

**Tonemapping:** Simple CIE observer integration to XYZ → sRGB. No HDR tone operator yet.

**Illuminant:** Hardcoded D65 (daylight) for Phase 0. No time-of-day.

---

## CUDA Kernel Requirements (Phase 0)

Two kernels, compiled via `cudarc` / PTX:

### 1. `depth_radix_sort`
- Input: array of (depth: f32, splat_index: u32) pairs
- Output: sorted indices (back-to-front)
- Algorithm: 2-pass 16-bit radix sort on the f32 depth value
- Tile size: 16×16 pixels

### 2. `spectral_rasterize`
- Input: sorted splat indices, splat buffer (pos/scale/rot/opacity/spectral), camera matrices
- Output: RGBA f16 framebuffer tile
- Algorithm: per-tile alpha-compositing of projected 2D Gaussians
- Phase 0: 4-band spectral → RGB via precomputed CIE observer LUT

Both kernels target **sm_75** minimum (RTX 20xx). No FP16 Tensor Core use yet.

---

## Rendering Backend

Use **`wgpu`** with the Vulkan backend on Linux/Windows. Metal backend deferred.

Responsibilities:
- Create swapchain, depth buffer, per-frame command buffers
- CUDA/Vulkan interop via `VK_KHR_external_memory` to share the framebuffer tile output
  from the CUDA rasterizer without a CPU-side copy
- Blit CUDA output tile → swapchain image
- `puffin` integration for per-frame CPU/GPU timing

---

## Plop System (Phase 0 — Hardcoded)

No UI. The demo binary hard-codes two asset instances:

```rust
let instances = vec![
    SplatInstance { asset_uuid: DEMO_UUID, position: Vec3::ZERO,          rotation: Quat::IDENTITY },
    SplatInstance { asset_uuid: DEMO_UUID, position: Vec3::new(20.0, 0.0, 0.0), rotation: Quat::IDENTITY },
];
```

The render loop transforms each instance's splat positions by its world matrix before feeding
the depth sort kernel. This proves the instancing path without any editor work.

---

## Performance Budget (Phase 0)

| Metric | Target |
|---|---|
| GPU | RTX 3070 (8GB VRAM) |
| Resolution | 1920×1080 |
| Splat count | ≤ 200,000 (2 instances × 100k) |
| Frame time | ≤ 16.7ms (60fps) |
| VRAM usage | ≤ 512MB |
| Load time | ≤ 2s for a 2MB .vxm from NVMe |

If 200k splats cannot hit 60fps, the Phase 0 exit criteria is unmet and the rasterizer kernel
must be profiled before Phase 1 begins.

---

## Dependencies (Cargo)

```toml
# vox_core
glam        = "0.28"       # math (Vec3, Quat, Mat4)
uuid        = { version = "1", features = ["v4"] }
half        = "2"          # f16 support
bytemuck    = "1"          # safe transmutes

# vox_data
zstd        = "0.13"
thiserror   = "1"

# vox_render
cudarc      = "0.11"       # CUDA device/kernel management
wgpu        = "22"
puffin      = "0.19"
raw-window-handle = "0.6"

# vox_app
winit       = "0.29"
puffin_egui = "0.26"       # profiler overlay only
```

No ECS, no `tokio`, no `rayon` in Phase 0 — these are added in Phase 1.

---

## Phase 0 Exit Criteria

All of the following must be true before Phase 1 begins:

- [ ] `.vxm` file round-trips (write → read → identical splat data)
- [ ] CUDA depth sort kernel produces correct back-to-front order (validated against CPU reference)
- [ ] Spectral rasterizer renders a single asset without visual corruption
- [ ] Two instances render correctly at a 20m offset
- [ ] 200k splats at 1080p hits ≥ 60fps on target GPU
- [ ] `puffin` overlay shows frame breakdown (sort time, rasterize time, blit time)
- [ ] No VRAM leaks over 60 seconds of runtime (checked via `nvidia-smi`)

---

## Phase 1 Preview (not spec'd here)

Once Phase 0 exits:
- Add Bevy ECS for entity/component management of instances
- Add SVO + DashMap spatial hash for 1km tile occupancy
- Expand spectral pipeline to full 8 bands with CIE integration
- Async asset loading via `tokio` + io_uring
- Basic terrain mesh as a ground plane for snapping

---

## Foundational Design Decision: Asset Origin

Assets in this engine are **never captured from the real world and never converted from
existing 3D models.** All `.vxm` assets originate from one of two pipelines:

1. **Procedural Generation (Proc-GS)** — Rust rules define geometry and spectral parameters
   directly. A building is a *function*, not a fixed point cloud. Variation (window count,
   material wear, facade style) is a parameter shift, not a separate asset.

2. **Generative / Hallucinated** — A generative model produces Gaussians from a seed or
   prompt, trained on Proc-GS outputs as ground truth. It adds micro-detail and style
   variation on top of the procedural base.

**Why this matters:**
- Spectral coefficients are **physically defined at generation time**, not inferred from a
  photo. Concrete has the concrete SPD. Glass has the glass transmission curve. This is what
  makes the spectral pipeline valuable — re-lighting, weather, and time-of-day all work
  correctly because no lighting was ever baked in.
- One procedural rule set generates infinite variation. The city does not scale with artist
  hours.
- Destruction, aging, and weather are parameter shifts on the same asset, not separate files.
- The generative model (Phase 2+) is tractable because it works *on top of* procedural
  outputs — not from nothing.

**Implication for Phase 0:** The synthetic test asset (a procedurally generated grid of
Gaussians with physically defined spectral values) is not a throwaway. It is the correct
representation of how all real assets will be created.

See `asset-generation-pipeline.md` for the full pipeline specification.

---

## Mapping to Master Requirements List

| Phase 0 item | Master list ref |
|---|---|
| Crate layout | #1 |
| `.vxm` format | #21, #22, #27, #30 |
| CUDA depth sort | #13 |
| Spectral rasterizer | #11, #12 |
| wgpu swapchain | #47 |
| Profiling | #48 |
| Plop instancing (hardcoded) | #3 (partial) |
| Spectral → RGB tonemapping | #11 (partial) |
