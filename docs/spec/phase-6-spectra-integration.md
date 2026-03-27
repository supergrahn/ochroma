# Phase 6 — Spectra Integration & Production Polish

**Goal:** Replace Ochroma's basic wgpu rasteriser with the Spectra renderer (`~/git/aetherspectra/spectra`) for production-quality spectral Gaussian splatting, then polish every system to shipping quality.

## Why This Phase Matters

Ochroma currently has 14,047 lines across 20 simulation modules, 27 render modules, and 13 core modules. All compile and pass 282 tests. But the rendering is basic — a single WGSL shader doing flat Gaussian billboards. To surpass Unreal, we need Spectra's:

- **Path-traced global illumination** via wavefront PT integrator
- **OptiX AI denoiser** for real-time quality at interactive framerates
- **DLSS/FSR upscaling** for 4K output from lower internal resolution
- **3DGS tile rasteriser** with proper depth sorting and alpha compositing
- **Spectral hero wavelength** strategy (380-780nm, not just 8 bands)
- **Neural radiance caching** for indirect lighting
- **MaterialX material graphs** compiled to GPU shaders

## Architecture

```
Ochroma Game Engine (Rust)
    │
    ├─ vox_app (game loop, UI, input)
    ├─ vox_sim (city simulation)
    ├─ vox_core (types, math, ECS)
    ├─ vox_data (assets, formats)
    │
    └─ Spectra Bridge (NEW)
         │
         ├─ spectra-gaussian-render (Rust crate, direct dependency)
         ├─ spectra-accel (BVH acceleration)
         ├─ spectra-lod-mgr (LOD management)
         └─ SpectraRenderer (Python, via subprocess or FFI)
```

### Integration Strategy

**Option A: Rust crate dependencies (preferred for real-time)**
- Add Spectra's Rust crates (`spectra-gaussian-render`, `spectra-accel`, `spectra-lod-mgr`) as path dependencies
- Call them directly from `vox_render` for Gaussian splatting
- No Python overhead, no IPC latency

**Option B: Python subprocess (for offline/high-quality)**
- Launch Spectra's Python renderer as a subprocess
- Send scene state via shared memory or files
- Receive rendered frames back
- Used for baked lighting, cinematics, marketing screenshots

We implement both: Option A for the real-time game loop, Option B for offline quality.

## Phase 6 Features

### 6.1 Spectra Rust Crate Integration

- Add `spectra-gaussian-render` as a workspace dependency (path = `~/git/aetherspectra/spectra/rust/spectra-gaussian-render`)
- Create `crates/vox_render/src/spectra_bridge.rs` that wraps Spectra's rendering API
- Route all splat rendering through Spectra when available, fall back to our WGSL shader otherwise
- Use Spectra's BVH for acceleration structure instead of our spatial hash

### 6.2 Scene State Export to USD

- Export Ochroma's ECS world state to USD format for Spectra consumption
- Each frame: serialize visible instances + camera + lights to a USD stage
- Spectra loads the USD and renders

### 6.3 Production Lighting

- Time-of-day sky model (Preetham or Hosek-Wilkie)
- Sun + moon directional lights with spectral emission
- Point lights from buildings/lamps with distance attenuation
- Ambient light from sky dome

### 6.4 Shadow Integration

- Use shadow catcher meshes (already computed) as shadow receivers
- Spectra renders shadow pass using BVH ray tracing
- Composite shadows onto terrain

### 6.5 Real-Time Denoising

- Feed noisy spectral framebuffer to OptiX AI denoiser (via Spectra)
- Output clean frame at interactive rates
- Fall back to our bilateral denoiser on non-NVIDIA hardware

### 6.6 Production UI Polish

- Replace egui dev panels with polished game UI
- Notification system (milestone popups, disaster alerts, advisor messages)
- Mini-map with zone colouring
- Tutorial/onboarding flow for new players
- Settings menu (graphics quality, audio volume, keybindings)

### 6.7 Audio Integration

- Activate rodio backend for real sound output
- Ambient soundscape layers (urban hum, nature, weather)
- UI sounds (click, place, zone, error)
- Construction sounds when buildings grow
- Adaptive music that scales with city era

### 6.8 Performance Optimisation

- Profile with puffin and identify bottlenecks
- GPU compute sort (already implemented) replaces CPU sort for >100k splats
- Instance batching (already implemented) reduces draw calls
- Tile streaming (already implemented) manages VRAM
- Target: 5M visible splats at 1080p 60fps on RTX 4080

### 6.9 Final Integration Testing

- Run the game for 30 minutes without crashes
- Place roads, zone, grow 100+ buildings
- Reach 10,000 citizens with stable FPS
- Save game, close, reopen, load — full state preserved
- Undo/redo 50 actions without corruption

## Exit Criteria

- [ ] Spectra renders the city with path-traced lighting or denoised real-time output
- [ ] Shadows visible on terrain from buildings
- [ ] Day/night cycle produces visually correct lighting at all hours
- [ ] 5M splats at 1080p ≥ 60fps
- [ ] Audio plays: ambient city sounds, UI clicks, construction
- [ ] Polished UI with notifications, mini-map, settings
- [ ] 30 minutes without crash at 10,000 citizens
- [ ] Save/load preserves complete game state
- [ ] Performance profile shows no single-frame spikes > 33ms

## What We're NOT Doing in Phase 6

- Multiplayer (Phase 7)
- Steam integration (Phase 7)
- Console ports (Phase 8)
- Full skeletal animation system (Phase 7)
- Modding documentation (Phase 7)
