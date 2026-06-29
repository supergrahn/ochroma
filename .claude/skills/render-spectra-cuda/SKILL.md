---
name: render-spectra-cuda
description: Render a frame/still/witness through the Spectra path tracer on the NVIDIA CUDA Windows box (tomespensin, RTX 4070 Ti). Use when building urban_horizon with spectra-cuda, producing a --shot-map / SHOT_HERO frame, or verifying a render change on the GPU target. The dev Linux box CANNOT build the CUDA/OptiX stack (Slang/CUDA live only on the box).
---

# Render with Spectra on CUDA (the Windows box)

The performance target. The Linux dev box edits code but **cannot compile the CUDA/OptiX stack** (`spectra-optix` dies at `shader-slang-sys` without Slang+CUDA). All CUDA builds + renders run on `tomespensin`.

## Topology
- **Dev box (source of truth):** Linux `~/src/ochroma`, `~/src/spectra`, `~/Ochroma/projects/urban_horizon`.
- **Box `tomespen@tomespensin`:** engine `/mnt/c/Users/tom_e/src/ochroma`, spectra `/mnt/c/Users/tom_e/src/spectra`, game `/mnt/c/Users/tom_e/ochroma/projects/urban_horizon`.
- Game `Cargo.toml` path deps are relative, so `../../../src/ochroma` on the box must match — keep the layout.

## Workflow: edit → sync → build → render → pull
1. **Sync only the changed files** (build reads files, not git — branch on the box is irrelevant):
   ```
   rsync -az ~/src/ochroma/crates/.../file.rs        tomespen@tomespensin:/mnt/c/Users/tom_e/src/ochroma/crates/.../file.rs
   rsync -az ~/Ochroma/projects/urban_horizon/src/... tomespen@tomespensin:/mnt/c/Users/tom_e/ochroma/projects/urban_horizon/src/...
   ```
2. **Build (background — 595s SSH timeout truncates foreground):**
   ```
   ssh tomespen@tomespensin '/mnt/c/Windows/System32/cmd.exe /c "C:\Users\tom_e\build_uh.cmd"'
   ```
   `build_uh.cmd` sets `SLANG_DIR`, `SPECTRA_BACKEND=cuda`, `CUDA_PATH=...\CUDA\v13.3`, `PATH=%SLANG_DIR%\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;...` and builds `vox_render --features spectra-native` + the `play` bin. **Verify `BUILD_DONE_EXIT=0`** in `C:\Users\tom_e\build_uh.log`.
3. **Render (background):**
   ```
   ssh tomespen@tomespensin '/mnt/c/Windows/System32/cmd.exe /c "C:\Users\tom_e\render_hero.cmd"'
   ```
   `runshot.cmd` = whole-city overview; `render_hero.cmd` = `SHOT_HERO=1` close-up on the first developed lot (use this to see surface detail / relief / glass). Both invoke `play.exe --shot-map <map_id> <out.png> <max_lots>`.
4. **Pull the PNG to view:** `rsync -az tomespen@tomespensin:/mnt/c/Users/tom_e/dlss-shots/<out>.png /tmp/ ` then Read it.

## The two arguments people get wrong
- **`--shot-map <map> <out.png> <N>` → `N` is `max_lots`, NOT spp.** It caps how many real-city lots develop (each lot = one Forge building; the debug build blows host memory past a few hundred). 128 is a safe, frame-filling default.
- **spp / bounces / denoiser / GI are FIDELITY-TIER driven, not an arg:** `OCHROMA_FIDELITY_TIER=performance|balanced|beauty`. **Performance ≈ 2–4 spp + denoise** (the real-time path — this is how the game renders, NOT brute-force spp). Never crank spp to clean up noise; lower the tier and let the À-Trous denoiser do it. Display res via `SPECTRA_SHOT_WIDTH/HEIGHT` (default 4K); internal render res is derived from the tier; DLSS upscales internal→output at present time.

## THE LAW (no-rasterizer)
The game renders **only** through Spectra: `urban_horizon::spectra_frame::produce_live_frame` → `ResidentCityRenderer` (CUDA). The wgpu `HybridComposeGpu` is a **rasteriser and is NOT a render path** — `render_gpu::hybrid_renderer_for` returns `None`; never construct it.
- **If a render panics in a wgpu buffer** (`Device::create_buffer label='hybrid_tris'`, "size > max buffer size"), a banned wgpu rasteriser is being built (usually `SceneRenderer::new_city`). **Fix that path to not allocate it — do NOT raise wgpu limits** (that keeps the banned path alive).

## Gotchas
- **Windows logs are UTF-16LE+CRLF.** Read with `cat <full log> | iconv -f UTF-16LE -t UTF-8 | tr -d '\r'`. A byte-`tail -c` MIS-ALIGNS UTF-16 → mojibake; decode the whole file (or `tail -n`) then grep.
- `.cmd` wrappers are plain ASCII — do NOT `iconv` them (that garbles to CJK).
- The 3 toolchain gotchas (nvrtc in `bin\x64` not `bin` on CUDA 13+; `libclang` via VS2022 LLVM; `spectra-dlss` cfg stub) are auto-handled by `scripts/build-windows-gpu.ps1` (engine) — `build_uh.cmd` is the game wrapper built on the same env.
- `target\debug` (debug build) is the norm here; renders are seconds at the Performance tier.

## Honesty
The witness is a **rendered frame a human looks at**, plus the `[shot-map] ... mean_luma / lit%` stdout (catches black frames). Never claim relief/glass/textures work from a passing build alone — confirm in the pixels.
