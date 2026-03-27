# Phase 26 — Performance Supremacy

**Goal:** Ochroma renders more geometry at higher quality and lower latency than Unreal Engine 5, proven by benchmarks.

## 26.1 GPU Compute Radix Sort
- Replace bitonic sort with LSD radix sort on GPU
- O(n) sorting for millions of splats
- Double-buffered: sort while previous frame renders

## 26.2 Tile-Based Deferred Splatting
- Divide screen into 16x16 pixel tiles
- Per-tile splat list (avoid processing distant splats for near tiles)
- Shared memory optimisation within each tile workgroup
- Reduces overdraw by 3-5x

## 26.3 Async Compute Pipeline
- Simulation on CPU while GPU renders previous frame
- Double-buffered splat uploads: prepare frame N+1 while N renders
- Latency hiding: 1 frame of input lag but 2x throughput

## 26.4 Memory Pool Allocator
- Pre-allocated GPU buffer pools for splat data
- Ring buffer for per-frame uploads (no allocation per frame)
- Defragmentation during idle frames

## 26.5 Benchmark Suite
- Standard scenes: 1M, 5M, 10M, 50M splats
- Frame time at 1080p, 1440p, 4K
- Comparison framework: log results in JSON for CI
- Regression detection: fail if frame time increases >5%

## Exit Criteria
- [ ] 50M splats at 1440p ≥ 60fps on RTX 4080
- [ ] Sort time < 2ms for 10M splats
- [ ] Benchmark suite runs in CI with regression detection
- [ ] Memory allocations per frame: zero (pool-based)
