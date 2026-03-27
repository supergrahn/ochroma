# NVIDIA Next-Gen Integration — CLAS, MegaGeometry, DLSS 5.0, Atoms

**Goal:** Make Ochroma the first game engine natively designed for NVIDIA's Blackwell architecture. While Unreal retrofits triangles onto new hardware, Ochroma's Gaussian splats are the native primitive these technologies were designed for.

## Why We Win

NVIDIA's next-gen rendering stack is moving away from triangles toward micro-primitives:
- **CLAS** (Cluster Acceleration Structure) — hardware BVH for non-triangle geometry
- **MegaGeometry** — render billions of micro-primitives per frame
- **RTX Atoms** — hardware-accelerated micro-primitive rendering
- **DLSS 5.0** — neural frame generation from sparse samples

Gaussian splats ARE micro-primitives. Every technology NVIDIA is building plays to our strength. Unreal's triangle pipeline is the legacy architecture. We're the future.

## Architecture

```
Ochroma Game Engine
    │
    ├── Scene Graph (ECS)
    │     └── Entities with GaussianRenderer components
    │
    ├── Splat Data (VxM / PLY assets)
    │     └── Per-asset: positions, scales, rotations, spectral bands
    │
    ├── NVIDIA Rendering Backend
    │     ├── CLAS Builder
    │     │     └── Cluster splats into CLAS nodes (64-256 splats per cluster)
    │     │     └── Build BVH over clusters (OptiX IAS)
    │     │     └── Hardware ray-tracing against splat clusters
    │     │
    │     ├── MegaGeometry Dispatch
    │     │     └── Stream splat data to GPU ring buffer
    │     │     └── Dispatch per-tile rasterisation over billions of splats
    │     │     └── Hardware depth sort within tiles
    │     │
    │     ├── DLSS 5.0
    │     │     └── Render at 1/4 resolution with motion vectors
    │     │     └── Neural super-resolution + frame generation
    │     │     └── Spectral-aware: denoise per wavelength band
    │     │
    │     └── Multi-GPU (Atom Bus)
    │           └── Shard splat data across GPUs
    │           └── Each GPU renders its tile range
    │           └── Composite final frame via NVLink/Atom Bus
    │
    └── Fallback Path
          └── wgpu compute (AMD, Intel, Apple)
          └── Software rasteriser (CPU)
```

## 1. CLAS — Cluster Acceleration Structure

### What It Is
CLAS is NVIDIA's hardware acceleration structure for non-triangle geometry. Instead of building a BVH over triangles (like traditional RT), CLAS builds a BVH over **clusters** of arbitrary primitives. Each cluster defines its own intersection test.

### How We Use It
- Group nearby Gaussian splats into clusters of 64-256 splats
- Each cluster has a bounding box and a custom intersection shader
- The intersection shader evaluates Gaussian contribution for a ray
- OptiX hardware traversal finds which clusters a ray hits
- Result: hardware-accelerated ray tracing against Gaussian splat scenes

### Data Structure
```
SplatCluster {
    bounding_box: AABB,
    splat_indices: [u32; 64..256],  // indices into the global splat buffer
    lod_level: u8,                  // for hierarchical LOD
    center_of_mass: Vec3,           // for sorting
}

ClusterBVH {
    clusters: Vec<SplatCluster>,
    bvh_nodes: Vec<BVHNode>,        // built by OptiX
}
```

### Intersection Shader (Pseudo-CUDA)
```
// For each ray-cluster intersection:
for each splat in cluster:
    project splat to ray-local 2D
    compute 2D Gaussian contribution
    if contribution > threshold:
        report_intersection(distance, contribution, color)
```

## 2. MegaGeometry — Billion-Splat Rendering

### What It Is
MegaGeometry is NVIDIA's system for rendering billions of micro-primitives. It streams geometry data from system RAM to GPU, processes it in tiles, and handles memory management automatically.

### How We Use It
- Stream splat data from NVMe → system RAM → GPU via async DMA
- GPU processes splats in screen-space tiles (16×16 pixels)
- Per-tile: frustum cull → depth sort → alpha blend
- Tile results composited into final framebuffer
- Memory management: only splats visible in the current frustum are in VRAM

### Performance Target
- 1 billion splats in the scene
- 50 million visible per frame
- 144 fps at 1440p on RTX 5090

## 3. DLSS 5.0 — Neural Frame Generation

### What It Is
DLSS 5.0 combines:
- Super resolution (render at lower res, AI upscale)
- Frame generation (generate intermediate frames)
- Ray reconstruction (denoise sparse ray-traced samples)

### How We Use It
- Render spectral framebuffer at 960×540 (quarter of 4K)
- Pass motion vectors + depth + spectral bands to DLSS
- DLSS generates full 3840×2160 output
- Frame generation produces 2 frames per rendered frame
- Result: 4K 288fps from 72fps rendered frames

### Spectral Advantage
Standard DLSS works on RGB. Ochroma provides 8 spectral bands — DLSS can use the extra information for better temporal stability and more accurate colour reconstruction. This is a unique advantage no other engine has.

## 4. Multi-GPU via Atom Bus

### What It Is
Atom Bus is NVIDIA's next-gen GPU interconnect (successor to NVLink). It provides cache-coherent shared memory across multiple GPUs.

### How We Use It
- Splat data stored in shared GPU memory pool
- Each GPU handles a screen tile range
- Sort is distributed: each GPU sorts its visible splats
- Composite happens via shared memory (no CPU round-trip)
- Linear scaling: 2 GPUs = 2× performance

## Implementation Phases

### Phase A: CLAS Abstraction Layer (Now)
- Build cluster data structures in Rust
- Implement cluster generation from splat clouds
- Build BVH over clusters (CPU, portable)
- Define OptiX intersection shader interface
- Test with Spectra's OptiX crate

### Phase B: MegaGeometry Dispatch (Now)
- Implement tile-based splat processing pipeline
- Build async splat streaming system
- Implement GPU ring buffer for splat upload
- Test with wgpu compute shaders (portable)

### Phase C: DLSS Integration (Now, via Spectra)
- Connect to Spectra's DLSS manager
- Pass spectral framebuffer + motion vectors
- Implement fallback (bilinear upscale on non-NVIDIA)

### Phase D: Hardware Activation (When RTX 50 ships)
- Replace CPU BVH with OptiX CLAS
- Replace compute sort with hardware sort
- Enable DLSS 5.0 neural frame generation
- Test multi-GPU with Atom Bus

## Exit Criteria

- [ ] Splat clustering produces valid CLAS-ready data structures
- [ ] Billion-splat scene loads and renders (at reduced quality on current hardware)
- [ ] DLSS upscaling produces 4K output from quarter-res spectral render
- [ ] Multi-GPU dispatch distributes work across 2+ GPUs
- [ ] On RTX 5090: 50M visible splats at 4K 144fps
