# Phase 24 — Infinite Detail

**Goal:** Gaussian splats at every scale — from satellite view of a 100km city down to individual bricks, window panes, and wood grain. No LOD pop-in, no detail loss at any zoom level.

## 24.1 Hierarchical Gaussian LOD
- LOD 0: individual splats (bricks, tiles, leaves) — street level
- LOD 1: clustered splats (wall sections, canopy volumes) — block level
- LOD 2: building silhouettes (single Gaussian per building) — district level
- LOD 3: billboard splats (per-block colour average) — satellite level
- Smooth opacity crossfade between all levels

## 24.2 Detail Streaming
- Only LOD 0 data loaded for visible tiles
- LOD 1-3 pre-computed and stored per tile
- Predictive loading: pre-fetch LOD 0 for tiles the camera is approaching
- Memory-mapped I/O for instant LOD transitions

## 24.3 Procedural Micro-Detail
- At extreme zoom: generate sub-splat detail on the fly
- Brick texture: each brick gets mortar lines from Proc-GS
- Wood grain: procedural fiber patterns
- Metal: surface scratches and wear marks
- Generated and cached per frame

## 24.4 Temporal Stability
- Anti-aliasing via temporal accumulation
- Sub-pixel jitter for each frame
- History buffer rejects ghosting from camera movement
- Result: rock-solid image at any zoom level

## Exit Criteria
- [ ] Zoom from 10km altitude to 1cm detail without pop-in
- [ ] Individual bricks visible at street level
- [ ] Billboard splats at satellite view are colour-accurate
- [ ] No temporal artefacts (ghosting, shimmer) during camera movement
