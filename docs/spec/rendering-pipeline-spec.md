# Ochroma Rendering Pipeline — Cinematic Quality at Real-Time Framerates

**The Pipeline:**
```
Scene (Gaussian Splats + Spectral Materials)
    ↓
Spectra Path Tracer (CUDA/OptiX)
    - 1-4 samples per pixel
    - Real spectral light transport (8 bands, 380-660nm)
    - CLAS hardware acceleration for splat clusters
    - Output: noisy spectral framebuffer (8 channels) + depth + normals + motion vectors
    ↓
Spectral Denoiser (OptiX AI Denoiser)
    - Input: 8 spectral bands + auxiliary buffers
    - Temporal accumulation with motion-vector reprojection
    - Output: clean spectral framebuffer
    ↓
Spectral → Display Conversion
    - CIE 1931 observer integration
    - Tone mapping (ACES spectral-aware)
    - Output: HDR RGB framebuffer
    ↓
DLSS Super Resolution
    - Input: low-res HDR RGB + motion vectors + depth
    - Neural upscale: 540p → 2160p (4x)
    - Output: 4K HDR framebuffer
    ↓
DLSS Frame Generation
    - Input: current frame + previous frame + motion vectors
    - Generate intermediate frame
    - Output: 2 frames per rendered frame
    ↓
Display
    - HDR10 / Dolby Vision output
    - Result: 4K 120fps from 540p 30fps path-traced input
```

## Why This Works

1. **Path tracer produces physically correct light** — real caustics, real GI, real spectral colour
2. **Low sample count is OK** because the denoiser has spectral data (8 channels not 3)
3. **Spectral denoising is better than RGB** — the network has more signal to separate noise from detail
4. **DLSS 4x upscale** means we only path-trace 1/16th of the pixels
5. **Frame generation doubles FPS** with no additional rendering cost
6. **Net result: path-trace quality at game framerates**

## Auxiliary Buffers

Each frame produces these buffers alongside the spectral framebuffer:

| Buffer | Format | Purpose |
|---|---|---|
| Spectral radiance | 8 × f16 per pixel | The main image in spectral space |
| Depth | f32 per pixel | For DLSS temporal reprojection |
| Normals | 3 × f16 per pixel | For denoiser edge detection |
| Motion vectors | 2 × f16 per pixel | For temporal accumulation + frame gen |
| Object ID | u32 per pixel | For selection/interaction |
| Albedo | 8 × f16 per pixel | Spectral albedo for denoiser |
| Emission | 8 × f16 per pixel | Direct vs indirect separation |

## Fallback Chain

```
NVIDIA RTX (CUDA + OptiX + DLSS):
    Full pipeline — cinematic quality at 120fps

NVIDIA GTX (CUDA only):
    Path tracer + bilateral denoiser (no DLSS)
    Lower quality denoising, no frame gen
    Target: 1080p 60fps

AMD / Intel / Apple (wgpu):
    Tile-based EWA rasteriser (Spectra algorithm ported to wgpu compute)
    Post-process bloom + tone mapping
    No path tracing, no AI denoiser
    Target: 1080p 60fps

CPU (software rasteriser):
    Headless rendering or very low-end hardware
    Target: 720p 15fps
```

## Implementation Order

1. **Spectral framebuffer** — render to 8-channel buffer instead of RGB
2. **Auxiliary buffer generation** — depth, normals, motion vectors per frame
3. **Spectral tone mapper** — 8-band → HDR RGB with ACES
4. **Temporal accumulation** — reproject previous frame, blend with current
5. **Bilateral denoiser** — edge-preserving spectral denoiser (CPU, works everywhere)
6. **CUDA path tracer** — via Spectra's wavefront PT integrator
7. **OptiX denoiser** — hardware AI denoiser (NVIDIA only)
8. **DLSS integration** — super resolution + frame generation (NVIDIA only)
