# Domain 1 — Rendering Pipeline

**Status:** Draft — 2026-03-29
**Scope:** GPU splat rasterizer, real-time spectral GI, volumetric lighting, shadow pipeline, post-processing stack, order-independent transparency
**Engine:** Ochroma spectral Gaussian Splatting — Rust workspace, wgpu 24, WGSL shaders, naga validation

---

## Goals

The rendering pipeline must deliver a fully GPU-accelerated spectral Gaussian Splatting path capable of 4K/60 FPS on a mid-range discrete GPU (RTX 3070 class). The existing CPU rayon path in `spectra_render.rs` remains as a fallback for integrated graphics but is not the primary delivery target. Spectral fidelity (8 bands, ~380–720 nm) must be preserved end-to-end: from GaussianSplat storage through shading, lighting, shadows, atmospherics, and post-processing, with no band collapse to RGB until the final tone-map. Every subsystem must integrate cleanly with the existing `GpuRasteriser`, `SdfShadowPass`, `GiBaker`, and `SplatEmitter` types already in `crates/vox_render/src/`.

---

## 1.1 GPU Splat Rasterizer

### Overview

The CPU EWA tile renderer in `spectra_render.rs` (rayon, 70 FPS / 4K) becomes a fallback. A new GPU compute path handles tile assignment, depth radix sort, and alpha-blend accumulation entirely on the GPU, with indirect dispatch so the CPU never reads back splat counts per frame.

### Data Layout

The canonical GPU-side splat struct (std430-compatible, 80 bytes):

```wgsl
struct GpuSplat {
    position_depth : vec4<f32>,   // xyz = world position, w = view-space depth (set by tile_assign.wgsl)
    conic          : vec3<f32>,   // 2D EWA conic coefficients (a, b, c) in screen space
    _pad0          : f32,
    opacity_color  : vec4<f32>,   // w = opacity; xyz = band 0-2 in current pass
    spectral       : array<f32, 8>, // 8 spectral bands, f32 each (f16 unpacked at upload)
}
```

On the Rust side, this is `GpuSplatFull` in `gpu/splat_buffer.rs`:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSplatFull {
    pub position_depth: [f32; 4],
    pub conic:          [f32; 3],
    pub _pad0:          f32,
    pub opacity_color:  [f32; 4],
    pub spectral:       [f32; 8],
}
```

The upload path in `splat_buffer.rs::upload()` unpacks `GaussianSplat.spectral: [u16; 8]` from f16 bits to f32 via `half::f16::from_bits(x).to_f32()` before writing to the GPU buffer, exactly as the existing `splats_to_gpu()` function does in `gpu_rasteriser.rs`. The spectral field is not re-packed on GPU; f32 throughout shader code avoids f16 emulation overhead on non-native hardware.

### Tile Assignment Shader — `tile_assign.wgsl`

Dispatched once per frame as `dispatch_indirect` from a buffer written by a prior projection pass. Each invocation handles one splat:

1. Transform `position_depth.xyz` by the view-projection matrix (camera uniform at binding 0).
2. Project the 3D covariance matrix (reconstructed from `scale` and `rotation` stored in a separate compact buffer) to 2D screen-space covariance via the Jacobian of the projective transform (Zwicker et al. 2002, eq. 5). Write `conic` (upper triangle of the 2x2 conic matrix inverse).
3. Compute the 2D screen-space bounding rectangle of the Gaussian at 3-sigma. Clamp to viewport. If entirely outside, set `position_depth.w = -1.0` as a sentinel.
4. For each tile the splat touches, emit one entry to the `tile_keys` buffer: a packed `u64` where the high 32 bits hold the tile index and the low 32 bits hold the depth (as `f32::to_bits()`). Simultaneously write the splat index to `tile_vals`. Tile count per splat is bounded at compile time to `MAX_TILES_PER_SPLAT = 16`.

The number of emitted entries is atomically accumulated into a counter buffer. The indirect dispatch arguments for the sort and rasterize passes are written from this counter by a tiny follow-up compute shader (`write_indirect_args.wgsl`, 1 invocation).

### Sort Shader — `radix_sort.wgsl`

Four-pass 8-bit radix sort over the `tile_keys` u64 array, using a workgroup-local histogram with 256 bins. Each pass processes 8 bits. Workgroup size: 256 threads. The sort is stable (preserving insertion order for equal-depth splats from the same tile). This produces a sorted index array `sorted_tile_vals` giving front-to-back order within each tile, which is required for the EWA alpha-blend accumulation.

Radix sort is chosen over bitonic sort because it scales linearly with element count rather than O(N log^2 N); at 6 million splats × 4 tiles average = 24 M entries, radix sort completes in approximately 2–3 ms on an RTX 3070 whereas bitonic would take ~8 ms.

### Rasterize Shader — `splat_raster.wgsl`

One workgroup per tile (16×16 pixels). Workgroup size: 256 threads (16×16).

```wgsl
@compute @workgroup_size(16, 16, 1)
fn rasterize_tile(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_id = wg.x + wg.y * num_tiles_x;
    let pixel   = vec2<u32>(wg.x * 16u + lid.x, wg.y * 16u + lid.y);
    // ...
}
```

Each thread maintains a local `transmittance: f32 = 1.0` and `accum: array<f32, 8>` for spectral accumulation. The workgroup loads the tile's splat range from a `tile_ranges` buffer (start, end index into `sorted_tile_vals`). Splats are iterated front-to-back:

1. Fetch `GpuSplatFull` from the bindless splat buffer (binding 2, storage buffer, read-only).
2. Compute pixel offset from splat screen center. Evaluate EWA Gaussian: `alpha = opacity * exp(-0.5 * (a*dx*dx + 2*b*dx*dy + c*dy*dy))`.
3. If `alpha < 1/255`, skip.
4. For each band `i` in `0..8`: `accum[i] += alpha * transmittance * splat.spectral[i]`.
5. `transmittance *= 1.0 - alpha`.
6. If `transmittance < 0.001`, break (early exit).

The 8-band inner loop is unrolled at WGSL level. No per-band-group splitting pass is used; a single wide pass handles all 8 bands because WGSL allows `array<f32, 8>` in registers and the ALU cost is lower than the memory bandwidth cost of a second dispatch. After iteration, each thread writes its 8-channel result to a `StorageTexture2dArray` with 8 layers (one per band), format `Rgba32Float` with two bands packed per RGBA channel using the `rg` channels of each layer for the first pair and `ba` for the second. This avoids a 128-byte-per-pixel output and keeps the footprint at 32 bytes per pixel.

### Bindless Splat Buffer

The splat SSBO is bound at binding group 1, binding 2 with no per-draw rebinding. When assets stream in, new splats are appended to a persistent GPU buffer allocated at `MAX_SPLATS = 8_000_000` entries at startup (640 MB). A `free_list: Vec<u32>` on the CPU side tracks holes. No per-frame CPU upload occurs unless new splats arrive.

### Depth Pre-Pass

Before tile assignment, `depth_prepass.wgsl` runs as a render pass writing `position_depth.w` (view-space Z) for each splat center to `Depth32Float`. This depth buffer feeds `SdfShadowPass` and the DoF post-process. It also writes a per-pixel velocity vector (from previous frame's projected position) to a `Rg16Float` velocity texture for motion blur.

### CPU Fallback

`spectra_render.rs::render_with_spectra_u8()` remains unchanged. `GpuRasteriser::new()` accepts a `fallback: bool` flag. If the wgpu device does not support `Features::INDIRECT_FIRST_INSTANCE` and `Features::MULTI_DRAW_INDIRECT` (required for indirect dispatch on some WebGPU targets), the fallback is activated automatically.

---

## 1.2 Real-Time Dynamic GI (Spectral Radiance Cache)

### Overview

GI is dual-path: offline `GiBaker` (existing, rayon) produces `BakedGi` for static scenes; a new `GiProbePass` maintains a world-space probe grid updated incrementally at runtime, replacing `GiBaker` for dynamic scenes. The two paths write to the same `GiCache` read by the rasterizer.

### Data Structures

```rust
// crates/vox_render/src/gpu/gi_probe.rs

pub struct ProbeGrid {
    pub origin:  [f32; 3],
    pub spacing: f32,
    pub dims:    [u32; 3],
    pub probes:  Vec<SpectralProbe>,
}

pub struct SpectralProbe {
    /// Radiance arriving from 6 axis-aligned hemisphere faces × 8 spectral bands.
    /// Layout: radiance[face][band], face order: +X -X +Y -Y +Z -Z.
    pub radiance: [[f32; 8]; 6],
    /// Frame index when this probe was last updated.
    pub last_updated: u32,
    /// World-space position (redundant with grid coords, cached for GPU upload).
    pub world_pos: [f32; 3],
}
```

The 6-face layout is chosen over octahedral mapping for simplicity at this stage: each face covers a 90-degree cone, which maps cleanly to a cube-map fetch in the sample shader. The tradeoff is that 6-face probes oversample near the face boundaries (low solid angle per texel) vs. octahedral which distributes samples uniformly; a future upgrade (documented in `docs/spec/gi-octahedral.md`) can swap the layout without changing the ProbeGrid interface. At 8 bands per face × 6 faces × 4 bytes = 192 bytes per probe, a 16×8×16 grid of 2048 probes costs 384 KB GPU.

### GPU Layout

The probe data is uploaded to a `wgpu::Buffer` (SSBO, `STORAGE | COPY_DST`) as a flat array of `GpuProbe` (std430):

```wgsl
struct GpuProbe {
    radiance : array<f32, 48>,  // 6 faces × 8 bands, row-major
    world_pos: vec3<f32>,
    _pad     : f32,
}
```

### Rolling Update — `probe_update.wgsl`

Each frame, `GiProbePass::update()` selects 1–4 probes to refresh using a priority queue sorted by `max(age_weight, camera_proximity_weight)`:

- `age_weight = (current_frame - probe.last_updated) / 120.0` — probes unseen for 120 frames get weight 1.0.
- `camera_proximity_weight = 1.0 - saturate(distance(probe.world_pos, camera_pos) / MAX_PROBE_DIST)`.

Selected probes are written to a small uniform (`ProbeUpdateList { probe_indices: [u32; 4], count: u32 }`). The `probe_update.wgsl` compute shader then dispatches 6 workgroups per probe (one per face). Each workgroup:

1. Fetches the SDF from the SDF SSBO (shared with `SdfShadowPass`).
2. Marches N=16 cones in the face's hemisphere direction using sphere tracing: step along ray by `|SDF(p)|` until `|SDF(p)| < 0.01` or max distance. Occlusion = binary (hit or not).
3. Samples incident radiance from neighboring probes via trilinear interpolation of the `GpuProbe` SSBO (the `probe_sample.wgsl` function, inlined). This is the indirect (secondary bounce) component.
4. Injects the current EWA spectral framebuffer as the first-bounce source: for probes near the camera, the framebuffer's 8-channel output is sampled at the probe's projected screen position and composited with the cone-marched result weighted by `framebuffer_confidence = saturate(1.0 - depth / MAX_PROBE_DIST)`.
5. Writes updated `radiance[face][band]` values back to the probe SSBO with an exponential moving average: `new = old * 0.9 + computed * 0.1`. This temporal smoothing prevents flickering during rolling updates.

### Probe Sampling — `probe_sample.wgsl`

A shader function (not a separate dispatch) callable from the rasterizer and from `probe_update.wgsl`:

```wgsl
fn sample_probe_grid(world_pos: vec3<f32>, normal: vec3<f32>) -> array<f32, 8> {
    // 1. Find 8 surrounding probe grid cells (trilinear interpolation weights)
    // 2. For each: select the face whose center direction is closest to normal
    // 3. Fetch GpuProbe.radiance[face] for that probe
    // 4. Accumulate weighted sum across 8 probes
    // 5. Return 8-band irradiance
}
```

This function is called once per splat during the rasterize pass after EWA accumulation, adding the GI contribution to the final spectral value before writing to the output texture array.

### Integration with GiCache

`GiCache` (existing, in `gi_cache.rs`) currently applies `BakedGi` offsets to splats. After this spec is implemented, `GiCache::update_from_probes(&ProbeGrid)` uploads probe data to the GPU SSBO and the rasterizer reads probes directly, making the CPU-side `GiCache.irradiance: Vec<[f32;8]>` per-splat store optional for dynamic scenes.

---

## 1.3 Volumetric Lighting

### Overview

A froxel (frustum voxel) volume captures in-scattering and transmittance in view space. Sun shafts are computed by ray marching toward the sun through the froxel volume, attenuated by the SDF. A pre-computed Bruneton-style atmospheric LUT provides sky contribution per spectral band.

### Data Structures

```rust
// crates/vox_render/src/gpu/volumetric_pass.rs

pub struct FroxelVolume {
    pub width:  u32,  // typically 160 (screen_width / 12)
    pub height: u32,  // typically 90
    pub depth:  u32,  // 64 slices along Z, exponential distribution
    pub voxels: Vec<FroxelVoxel>,  // width × height × depth
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FroxelVoxel {
    pub scatter:       [f32; 8],  // per-band in-scatter coefficient
    pub transmittance: f32,       // combined extinction along ray
    pub _pad:          [f32; 3],
}

pub struct VolumetricPass {
    pub froxel_buffer:    wgpu::Buffer,
    pub lut_texture:      wgpu::Texture,   // 2D array, 8 layers (one per band)
    pub scatter_pipeline: wgpu::ComputePipeline,
    pub resolve_pipeline: wgpu::RenderPipeline,
}
```

### Froxel Depth Slicing

Slice `k` maps to view-space Z via `Z(k) = Z_near * (Z_far / Z_near)^(k / depth)` (exponential distribution). This concentrates froxel resolution near the camera where volumetric detail matters most. The camera near/far planes are stored in `CameraUniform` and already available at binding 0.

### Scatter Compute Shader — `scatter_compute.wgsl`

Dispatched as `(width/8, height/8, depth/4)` workgroups, each handling a 8×8×4 block:

1. Reconstruct world-space position of froxel center from froxel indices + camera uniform.
2. Fetch density from the SDF volume: density is `max(0, -SDF(p))` normalized to `[0,1]` (inside objects have density 0; participating medium has positive density in the atmosphere model).
3. Apply Rayleigh scattering: spectral scatter coefficient for band `b` is `RAYLEIGH_BASE[b] / density_norm`, where `RAYLEIGH_BASE` is a constant array:
   ```wgsl
   const RAYLEIGH_BASE: array<f32, 8> = array(
       1.0000, 0.7211, 0.5258, 0.3868, 0.2878, 0.2160, 0.1634, 0.1250
   );
   // band 0 (~380nm, violet) scatters ~8× more than band 7 (~720nm, red).
   // The ratio 8× comes from (720/380)^4 ≈ 12.9; clamped to 8× for artistic control.
   ```
4. Apply Mie scattering (aerosols, wavelength-independent): adds a flat `MIE_COEFF` to all bands.
5. Sun shaft contribution: march 16 steps from the froxel center toward `sun_direction`. At each step, sample the SDF. If inside an occluder (`SDF < 0`), accumulate extinction. Write `scatter` and `transmittance` to the froxel buffer.

### Bruneton Sky LUT

Pre-computed offline in `atmosphere.rs` (existing file) as a 2D texture per band: axes are `cos(zenith_angle)` (256 values, −1..+1) and `altitude_normalized` (64 values, 0..1). The LUT stores spectral sky radiance `L_sky[band]` at each (zenith, altitude) pair. At runtime, `scatter_compute.wgsl` samples the LUT at the current froxel's zenith angle and altitude via `textureSampleLevel`. Since the LUT varies per band, it is stored as a 2D texture array with 8 layers, format `R32Float`.

### Resolve Pass

The rasterize pass outputs 8-band splat color to a texture array. The volumetric resolve runs after rasterization as a full-screen fragment shader. For each pixel:

1. Reconstruct view-space Z from depth buffer.
2. Find the froxel index (convert clip position + depth to froxel coordinates).
3. March from the camera to the splat depth through the froxel volume (8 steps, trilinear sample at each step).
4. Accumulate: `final[b] = splat_color[b] * transmittance + in_scatter[b]`.
5. Write the result to the HDR spectral framebuffer.

---

## 1.4 Shadow Pipeline Completeness

### Spectral Translucency Shadow Maps

Point lights cast spectral shadows through translucent splats. The shadow map for a point light stores, per pixel, both a depth value and an 8-band transmittance. Storage format: one `Depth32Float` texture for geometry depth, plus one `Rgba32Float` texture array (2 layers, bands 0-3 in layer 0 as RGBA, bands 4-7 in layer 1) for transmittance. This doubles shadow map memory but enables glass, colored water, and leaf-canopy shadows with correct color temperature shifts.

During the shadow render pass, transparent splats (opacity < 0.95) write their `spectral` bands, attenuated by their opacity, into the transmittance texture via additive blending: `transmittance[b] = product(1 - alpha_i * (1 - spectral_i[b]))` across all contributing splats. This is an approximation; for N transparent splats the exact product requires sorted rendering, which is too expensive for shadow passes. The approximation is acceptable for thin foliage and glass; thick participating media use the volumetric path instead.

### Omnidirectional Shadow Maps — Cube-Map Depth

Point lights render 6 depth passes (one per cube face) into a `wgpu::TextureDimension::D2` with `depth_or_array_layers = 6`. A geometry shader emitting face index is not available in WGSL/wgpu; instead, 6 separate render passes are issued, each with a different view matrix constructed from the light position and face direction. This adds 6 draw calls per point light; budgeted at 4 point lights maximum per frame without tiled deferred.

### Shadow Atlas

All shadow maps (directional, point cube-faces, spot) are packed into a `ShadowAtlas`:

```rust
pub struct ShadowAtlas {
    pub texture: wgpu::Texture,       // D2Array, Depth32Float
    pub entries: Vec<ShadowAtlasEntry>,
}

pub struct ShadowAtlasEntry {
    pub light_id: u32,
    pub layer:    u32,
    pub region:   [u32; 4],  // [x, y, width, height] in texels
}
```

The atlas uses a simple shelf-packing algorithm: lights are sorted by shadow map resolution (largest first), shelves are filled left-to-right. A 4096×4096 atlas with 8 layers holds: 1 directional (2048×2048), 4 point lights (6 × 512×512), and 8 spot lights (256×256) with room to spare. The atlas is rebuilt only when the light set changes, not per frame.

### PCSS Contact Hardening

In the shadow sample shader (`pcss_sample.wgsl`), the penumbra kernel radius is computed from the average blocker depth:

1. Sample the shadow map in a 5×5 Poisson disk at the current pixel's shadow-space UV.
2. Average depth of all samples that are closer than the surface depth = `avg_blocker_depth`.
3. Penumbra width: `w = LIGHT_SIZE * (receiver_depth - avg_blocker_depth) / avg_blocker_depth`.
4. Filter the shadow map in a Poisson disk of radius proportional to `w` (clamped 1–32 texels).

This produces hard contact shadows beneath objects resting on surfaces and soft shadows for occluders far from the receiver, without any screen-space artifacts.

---

## 1.5 Post-Processing Stack

### Architecture

```rust
// crates/vox_render/src/postprocess.rs (extend existing file)

pub trait PostProcessPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn execute(&self, ctx: &mut PostProcessContext);
}

pub struct PostProcessPipeline {
    pub passes: Vec<Box<dyn PostProcessPass>>,
    pub ping:   wgpu::Texture,  // HDR spectral ping-pong buffers
    pub pong:   wgpu::Texture,  // same format as froxel resolve output
}

pub struct PostProcessContext<'a> {
    pub device:       &'a wgpu::Device,
    pub queue:        &'a wgpu::Queue,
    pub encoder:      &'a mut wgpu::CommandEncoder,
    pub source:       &'a wgpu::TextureView,  // current input (ping or pong)
    pub target:       &'a wgpu::TextureView,  // current output
    pub depth:        &'a wgpu::TextureView,
    pub velocity:     &'a wgpu::TextureView,
    pub camera:       &CameraUniform,
    pub frame_index:  u64,
}
```

Default pass order: `[AutoExposure, Bloom, DepthOfField, MotionBlur, ChromaticAberration, ScreenSpaceReflections, ColorGrading, FilmGrain, LensFlare, SpectralTonemap]`.

### Depth of Field — Spectral Bokeh

The `DofPass` reads the splat depth buffer and the 8-band HDR texture. Circle of Confusion radius is computed per pixel from `depth`, `focus_distance`, `aperture`, and `focal_length`. Abbe dispersion: short-wavelength bands defocus more than long-wavelength bands. Band `b`'s CoC radius is `CoC_base * (1.0 + ABBE_COEFF * (7 - b) / 7.0)`, where `ABBE_COEFF = 0.08` is calibrated to match real lens data. This means band 0 (violet, ~380 nm) has a CoC 8% wider than band 7 (red, ~720 nm), producing chromatic bokeh fringes without any post-hoc RGB shift. The bokeh kernel is a hexagonal Poisson disk (18 samples) scaled by the per-band CoC. Each band is blurred independently in a single compute dispatch that loops `b` in `0..8`.

### Motion Blur

The `MotionBlurPass` reads the `Rg16Float` velocity texture written in the depth pre-pass. Per-pixel velocity (in screen-space pixels/frame) is used to reconstruct a motion vector. 8 samples are taken along the velocity direction; each sample fetches from the HDR spectral texture. A neighbor-max velocity dilation (3×3 max filter on the velocity buffer) prevents dark halos behind fast-moving objects. The blur applies uniformly across all 8 spectral bands (motion blur is achromatic).

### Bloom — Spectral Dual-Kawase

The `BloomPass` extracts bright regions per band with per-band thresholds. High-frequency spectral emitters (band 0 and 1) have a lower extraction threshold, producing wider bloom halos for UV-range emitters. Six levels of dual-Kawase downsampling/upsampling occur per band group (bands 0-3 and bands 4-7 processed in two dispatches, 4 bands each via RGBA channels). The two groups are recombined after upsampling. The final bloom is added to the HDR buffer before tonemapping.

### Chromatic Aberration

A fragment shader `chromatic_aberration.wgsl` displaces each band's UV sample position radially outward from the screen center by `aberration_strength * (b / 7.0) * r^2`, where `r` is normalized screen radius. Band 7 (red) is displaced most; band 0 (violet) least. This matches real lens chromatic aberration (lateral, not axial) and is visually distinct from the axial aberration modeled in DoF.

### Auto-Exposure

The `AutoExposurePass` runs a two-pass histogram compute. Pass 1 (`luminance_histogram.wgsl`): each thread bins one pixel's luminance (computed from the 8-band spectral value via CIE Y tristimulus) into a 256-bin histogram in workgroup shared memory, then atomically accumulates into a global histogram buffer. Pass 2 (`ev_bias.wgsl`): one thread reads the histogram, finds the 20th–90th percentile range, computes average log-luminance, and writes an EV bias to a single-float uniform buffer. The EV bias is passed to the tonemap pass. A temporal low-pass filter (alpha = 0.02 per frame) smooths the exposure adaptation.

### Color Grading — Spectral 3D LUT

The `ColorGradingPass` applies a 3D LUT defined in spectral space rather than RGB. The LUT is a `Rgba3d` texture (32×32×32, RGBA32Float) authored in the material editor's spectral curve tool. At runtime, the 8-band spectral value is first reduced to a 3D coordinate (band 0 mapped to R axis, band 3 to G, band 7 to B) via a precomputed PCA basis, then sampled from the LUT, and the resulting adjustment is applied back to all 8 bands. This is an approximation but gives art-directable spectral grading without requiring an 8D LUT.

### Screen-Space Reflections

The `SsrPass` ray-marches in the depth buffer from the splat surface along the reflection direction (computed from view direction and splat normal, where normals are reconstructed from the depth buffer via finite differences). 32 steps with binary refinement. Hits fetch from the HDR spectral texture (current frame). Misses fall back to the nearest probe in the `GiProbePass` SSBO. The SSR result is blended with the rasterizer output by `roughness`-based weight: `blend = saturate(1.0 - roughness * 4.0)`.

### Spectral Tonemapper

`spectral_tonemapper.rs` (existing) maps 8 bands to XYZ via the CIE 1931 color matching functions, then to sRGB. The existing implementation is kept; the post-process stack calls it last, after all spectral passes complete.

---

## 1.6 Order-Independent Transparency

### Moment-Based OIT

Transparent splats (opacity < 0.95) are sorted into a separate `transparent_splat_buffer` during the tile assignment phase. After the opaque EWA pass completes, `oit_accumulate.wgsl` runs as a compute pass over transparent splats only:

For each transparent splat touching a pixel, the shader accumulates the first four power moments of the depth distribution: `b_0 = sum(alpha_i)`, `b_1 = sum(alpha_i * z_i)`, `b_2 = sum(alpha_i * z_i^2)`, `b_3 = sum(alpha_i * z_i^3)`. These four values plus total transmittance are stored in a `Rgba32Float` texture (moments 0-3) and a separate `R32Float` texture (total transmittance). The resolve pass `oit_resolve.wgsl` reconstructs the per-pixel transmittance function from the moments using the Hamburger 2018 moment-based OIT technique (solving a degree-3 polynomial), then composites transparent splats over the opaque result in correct order without any per-pixel sorting.

### Stochastic Transparency Fallback

On hardware without `Features::SHADER_F64` (needed for stable polynomial solve), a stochastic fallback is used: during rasterization, each transparent splat accepts or rejects its contribution via a per-pixel Bayer-matrix threshold compared against the splat's opacity value. The threshold cycles through a 4×4 Bayer matrix indexed by `(pixel.x % 4, pixel.y % 4, frame_index % 16)`. Temporal accumulation via `TemporalAccumulationPass` (using the motion vector buffer) resolves the stochastic noise over 8–16 frames. The result is visually equivalent to moment OIT at lower GPU memory cost.

### Integration with EWA Tile Loop

The `tile_assign.wgsl` shader tags each tile entry as opaque (`opacity >= 0.95`) or transparent. Opaque entries go into `tile_keys_opaque`; transparent into `tile_keys_transparent`. The radix sort and rasterize passes run twice: once for opaque (writing to the main HDR texture array), once for transparent (writing to the OIT moment textures). The OIT resolve runs after both, producing the final composite. This adds one sort pass and one rasterize pass per frame; the transparent splat count is typically << 10% of total, so the overhead is bounded.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_render/src/gpu/splat_buffer.rs` | CREATE | `GpuSplatFull` struct, persistent SSBO upload, free-list allocator |
| `crates/vox_render/src/gpu/tile_assign.rs` | CREATE | `TileAssignPass`: owns projection + tile assignment compute pipeline |
| `crates/vox_render/src/gpu/radix_sort.rs` | CREATE | `RadixSortPass`: 4-pass 8-bit GPU radix sort over `tile_keys` |
| `crates/vox_render/src/gpu/splat_raster.rs` | CREATE | `SplatRasterPass`: tile-based EWA accumulate compute pass |
| `crates/vox_render/src/gpu/depth_prepass.rs` | CREATE | Renders splat center depths + velocity to texture |
| `crates/vox_render/src/gpu/gi_probe.rs` | CREATE | `GiProbePass`, `ProbeGrid`, `SpectralProbe` — rolling probe update |
| `crates/vox_render/src/gpu/volumetric_pass.rs` | CREATE | `VolumetricPass`, `FroxelVolume`, `FroxelVoxel`, froxel scatter + resolve |
| `crates/vox_render/src/gpu/shadow_atlas.rs` | CREATE | `ShadowAtlas`, `ShadowAtlasEntry`, shelf-packing allocator |
| `crates/vox_render/src/gpu/oit_pass.rs` | CREATE | `OitAccumulatePass`, `OitResolvePass`, stochastic fallback |
| `crates/vox_render/src/postprocess.rs` | MODIFY | Add `PostProcessPipeline`, `PostProcessPass` trait, all pass structs |
| `crates/vox_render/src/gpu/mod.rs` | MODIFY | Export all new gpu/ modules |
| `crates/vox_render/src/lib.rs` | MODIFY | Wire new passes into the main render loop |
| `crates/vox_render/src/shaders/tile_assign.wgsl` | CREATE | Splat projection + tile range emit shader |
| `crates/vox_render/src/shaders/radix_sort.wgsl` | CREATE | 4-pass workgroup radix sort |
| `crates/vox_render/src/shaders/splat_raster.wgsl` | CREATE | Tile-parallel EWA accumulate |
| `crates/vox_render/src/shaders/depth_prepass.wgsl` | CREATE | Splat depth + velocity write |
| `crates/vox_render/src/shaders/probe_update.wgsl` | CREATE | Per-probe radiance update via cone march |
| `crates/vox_render/src/shaders/probe_sample.wgsl` | CREATE | Trilinear probe interpolation function (included by other shaders) |
| `crates/vox_render/src/shaders/scatter_compute.wgsl` | CREATE | Froxel scatter + transmittance fill |
| `crates/vox_render/src/shaders/volumetric_resolve.wgsl` | CREATE | Apply froxels to rasterized spectral image |
| `crates/vox_render/src/shaders/pcss_sample.wgsl` | CREATE | PCSS shadow filter with average blocker depth |
| `crates/vox_render/src/shaders/oit_accumulate.wgsl` | CREATE | Moment OIT accumulation for transparent splats |
| `crates/vox_render/src/shaders/oit_resolve.wgsl` | CREATE | Hamburger polynomial transmittance reconstruct |
| `crates/vox_render/src/shaders/bloom.wgsl` | CREATE | Dual-Kawase per-band bloom downsampling + upsampling |
| `crates/vox_render/src/shaders/dof.wgsl` | CREATE | Per-band CoC compute + hexagonal Poisson bokeh |
| `crates/vox_render/src/shaders/motion_blur.wgsl` | CREATE | Velocity-buffer motion blur, 8 samples |
| `crates/vox_render/src/shaders/chromatic_aberration.wgsl` | CREATE | Radial lateral chromatic aberration per band |
| `crates/vox_render/src/shaders/luminance_histogram.wgsl` | CREATE | Two-pass auto-exposure histogram |
| `crates/vox_render/src/shaders/ssr.wgsl` | CREATE | Screen-space reflection ray march + probe fallback |
| `crates/vox_render/src/shaders/color_grading.wgsl` | CREATE | 3D LUT sample in PCA-reduced spectral space |
| `crates/vox_render/src/shaders/write_indirect_args.wgsl` | CREATE | Writes dispatch args from atomic counter |

---

## Milestones

### M1 — GPU Radix Sort (Weeks 1–6)
Implement `radix_sort.wgsl` and `RadixSortPass`. Validate correctness by sorting a CPU-generated array of u64 pairs and comparing GPU output with a reference sort. Integration test: sort 8M random u64s in under 5 ms on RTX 3070.

### M2 — GPU EWA Rasterizer (Weeks 7–18)
Implement tile_assign, depth_prepass, splat_raster shaders and their Rust wrappers. Wire into the render loop behind a `RenderPath::Gpu` flag. Acceptance: GPU path renders the same test scene as the CPU path with PSNR > 45 dB across all 8 bands.

### M3 — Real-Time GI Probes (Weeks 19–30)
Implement `GiProbePass`, `probe_update.wgsl`, `probe_sample.wgsl`. Rolling update must complete in under 1 ms per frame (4 probes × ~0.25 ms each). Integration: GI probes replace `GiBaker` in the dynamic scene test.

### M4 — Volumetrics + Shadow Completeness (Weeks 31–44)
Implement `VolumetricPass`, `scatter_compute.wgsl`, `volumetric_resolve.wgsl`, spectral shadow maps, `ShadowAtlas`, PCSS. Deliverable: outdoor scene with sun shafts and colored glass shadows.

### M5 — Post-Processing Stack (Weeks 45–58)
Implement all `PostProcessPass` impls: Bloom, DoF, MotionBlur, ChromaticAberration, AutoExposure, ColorGrading, SSR, FilmGrain, LensFlare. Each pass independently toggleable via `PostProcessPipeline::enable(name)`.

### M6 — OIT (Weeks 59–66)
Implement moment OIT accumulate + resolve. Stochastic fallback. Integration: glass material test scene renders without sorting artifacts.

---

## Acceptance Criteria

- GPU rasterizer renders a 6M-splat scene at 4K/60 FPS on RTX 3070 (measured via `GpuTimestampRing`).
- GPU and CPU paths produce PSNR > 45 dB on identical test input (all 8 bands independently).
- Radix sort correctly sorts 100% of inputs verified against `Vec::sort_unstable` on 10,000 random inputs of varying sizes.
- `GiProbePass::update()` costs under 1.5 ms per frame with 4 probes on RTX 3070.
- Froxel volume fill costs under 0.8 ms per frame at 160×90×64.
- Spectral translucency shadow renders colored shadow through a flat glass splat plane with correct per-band transmittance (verified by sampling the shadow texture and comparing to ground-truth transmittance).
- PCSS shadow visually verifies: a small cube resting on a plane has hard contact shadow at ground contact, soft shadow at 2m height.
- All post-process passes toggle on/off without pipeline rebuild (only draw call skip).
- OIT test: 16 overlapping transparent splat layers render with no Z-order artifacts (visual inspection + automated pixel comparison vs. CPU ground truth).
- `cargo test` passes on all platforms, including naga shader validation for every `.wgsl` file.
- No game-specific concepts (buildings, zoning, traffic) appear in any engine crate source file.

---

## Effort

| Subsystem | Estimated Effort |
|-----------|-----------------|
| 1.1 GPU Splat Rasterizer | 4 months |
| 1.2 Real-Time Dynamic GI | 3 months |
| 1.3 Volumetric Lighting | 2.5 months |
| 1.4 Shadow Pipeline | 2 months |
| 1.5 Post-Processing Stack | 3.5 months |
| 1.6 OIT | 1.5 months |
| **Total** | **~16.5 months** |

Work can be parallelized across two engineers: one owning 1.1 + 1.6 (GPU compute infrastructure) and one owning 1.2 + 1.3 + 1.4 + 1.5 (lighting). The GPU radix sort (M1) is on the critical path for M2 and must complete first. GI probes (M3) depend on the froxel volume (M4) only loosely; both can develop in parallel after M2 ships.
