---
name: render-spectra-vulkan
description: Render through the Spectra path tracer on the LOCAL Linux dev box via Vulkan (AMD Radeon 780M / RADV, or Mesa/lavapipe). Use when path-tracing locally with no NVIDIA, smoke-testing a render change before the CUDA box, or running the vox_render Vulkan tests. Spectra is first-party and portable — its VulkanSlangBackend (Slang→SPIR-V via ash) runs on any Vulkan device; it is NOT CUDA/RTX-gated.
---

# Render with Spectra on Vulkan (local Linux / AMD)

Spectra's `VulkanSlangBackend` compiles `.slang` → SPIR-V at runtime and runs on any Vulkan device (AMD RADV, Mesa/lavapipe CPU). This is the local dev/smoke-test render path; the CUDA box is the perf target.

## Prerequisites (the Slang gotcha)
The crates.io `shader-slang-sys` bindgen needs the Slang SDK. If a build dies at `shader-slang-sys` / `spectra-optix` with E52002 / "Unable to find libclang" / missing `stddef.h`, **Slang isn't wired** — that's the cause, not your code.
- Use **`scripts/build-spectra-native.sh`** — it auto-picks `SLANG_DIR=~/slang-sdk` (has `libslang-compiler.so`), sets `LD_LIBRARY_PATH`, and the BINDGEN args.
- The engine `.cargo/config.toml` carries `SLANG_DIR` + `BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include"`.
- `spectra-native`/`spectra` is a **non-default feature**; default builds don't touch Slang.

## Backend + device selection
- `SPECTRA_BACKEND=vulkan` — forces the Vulkan backend (PTX/nvrtc `cargo:warning`s are NON-fatal here; Vulkan ignores CUDA).
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` — pins the **real AMD GPU**. (`lvp_icd.json` = lavapipe CPU fallback; correct-but-slow.)
- `SPECTRA_SLANG_DIR` — kernel dir override; default `~/src/spectra/slang` (via `resolve_slang_kernel_dir`). **If frames come back blank, the kernel dir wasn't found** (every dispatch becomes a no-op).

## Canonical smoke test (proves the GPU renders)
```
SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
  --lib vulkan_backend_renders_quad -- --nocapture
```
Pass = `mean=0.41 variance=0.13 samples_done=4` (real GPU output, not blank).

## Run a city/asset path-trace
```
LD_LIBRARY_PATH=$HOME/slang-sdk/lib SPECTRA_BACKEND=vulkan \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo run --release --features spectra --bin <render-bin>
```
**Verify the current render binary/flags against the repo** before asserting — the game's still path is `--shot-map <map> <out.png> <max_lots>` (the 3rd arg is `max_lots`, NOT spp; spp is `OCHROMA_FIDELITY_TIER`-driven). The principle (Vulkan backend + Slang + ICD pin + kernel dir) is stable even as bins evolve.

## Perf reality
Path tracing on the 780M is **seconds per still** — Vulkan is the dev/correctness path, not the real-time target. Use it to confirm a render change is correct (non-black, geometry/materials right) before syncing to the CUDA box for the perf/quality run. See `render-spectra-cuda`.

## THE LAW (same as CUDA)
Render **only** through Spectra (`produce_live_frame` / the path tracer). The wgpu `HybridComposeGpu` rasteriser is NOT a render path and must not be constructed. A panic in a wgpu buffer (`hybrid_tris`) means a banned rasteriser is being built — fix that path, don't raise wgpu limits.

## Honesty
The witness is the rendered frame + its luma/lit stats. `vulkan_backend_renders_quad`'s `mean/variance` is the smoke gate; a city still must be eyeballed (and its `mean_luma`/`lit%` checked) — never claim it works from a passing build alone.
