# Design: Spectra Primary Renderer Bridge (2026-06-11)

## Scope

Make Spectra the primary live renderer behind Ochroma without moving game policy into the engine renderer or engine policy into Spectra.

The layer contract is:

```text
Civitas game systems running inside Ochroma -> Spectra renderer -> GPU
```

Civitas is a game running on top of the Ochroma game engine. It should not have a renderer and it should not drive the engine loop. Ochroma can change when the engine lacks the right abstraction. Spectra can change when the renderer lacks the right capability. The decision rule is ownership, not repo convenience.

## Ownership

| Layer | Owns | Must not own |
| --- | --- | --- |
| Civitas game | Buildings, cims, zoning, terraforming behavior, gameplay UX, semantic view state | Renderer modules, GPU presentation, renderer handles, TLAS/BLAS, GPU upload policy, engine-loop ownership |
| Ochroma engine | Game lifecycle, ECS, world partitioning, camera/frustum/distance policy, residency decisions, frame budgets, input/window/UI composition, game view APIs, Spectra bridge | Spectral path tracing internals, renderer-specific acceleration details |
| Spectra renderer | Generic geometry/material/SDF/instance handles, dirty range uploads, acceleration structures, spectral lighting, temporal reconstruction, denoise/upscale, GPU-resident output | Buildings, cims, zoning, city streaming semantics, terraforming rules |

## Decision

City-scale instancing is not a Spectra concept. Massive generic instancing is a Spectra capability.

Spectra should expose stable resource handles and fast update paths:

- `GeometryHandle`
- `MaterialHandle`
- `InstanceBatchHandle`
- `SdfVolumeHandle`
- `SdfInstanceBatchHandle`
- `RenderTargetHandle`
- dirty ranges for transforms, materials, geometry, SDF data, and lights
- residency changes for generic resources
- GPU-resident output targets

Ochroma should decide which resources are resident each frame and send generic updates to Spectra. Civitas contributes game systems/state that Ochroma invokes. Civitas must never call Spectra directly, drive presentation, or hide renderer work behind a game-owned facade.

## First Implementation

This spec is paired with two code anchors:

- `ochroma_engine::game_view` defines engine-invoked, renderer-agnostic view intent through `ViewIntent` and `GameViewSource`.
- `spectra-scene-state::live_update` defines the renderer-side generic live update contract.
- `vox_render::spectra_bridge` defines the engine-side ownership boundary and re-exports Spectra live update types under `spectra-native`.

This is intentionally a contract slice, not the whole renderer migration. It prevents the next implementation steps from hard-coding city semantics into Spectra or keeping the live frame on CPU pixels.

## Milestones

### M0 - Boundary Contract

Done when:

- Spectra has generic live-scene update types.
- Ochroma names the bridge ownership rules in code.
- Tests prove the boundary routes `Ultra`/realtime through Spectra and requires GPU-resident output.

### M1 - GPU-Resident Output

Add a Spectra render path that can render into a GPU image/surface selected by Ochroma.

Done when:

- The primary live frame no longer requires `FrameOutput` CPU download.
- CPU readback remains only for stills, diagnostics, and tests.

### M2 - Persistent Scene Handles

Extend Spectra upload/runtime state from batch-style full scene submission to persistent handles and dirty ranges.

Done when:

- Instance transform ranges can be updated without rebuilding the full scene.
- Material/SDF/geometry dirty ranges have explicit upload paths.
- Residency changes add/remove generic resources without game terms.

### M3 - Ochroma Residency Bridge

Map Ochroma world partitioning, atom/SDF residency, asset instances, and camera policy into Spectra update batches.

Done when:

- Ochroma owns the visibility/residency decisions.
- Spectra receives only generic resource handles and ranges.
- A city test can move the camera and produce bounded Spectra update batches.

### M4 - Game View Migration

Move the game live view onto the Ochroma->Spectra bridge and remove renderer ownership from the game project.

Done when:

- Civitas does not call Spectra directly.
- Civitas does not own `render`, `render_gpu`, presenter, shader, or GPU HUD modules.
- Ochroma invokes game systems/view providers; game code does not request or drive presentation.
- Generic camera projection/picking helpers live in `ochroma_engine::game_view`, not in game renderer modules.
- The WGPU tiled path is demoted to fallback/debug/transition path unless still needed for a specific measured tier.
- Presentation/HUD composition stays GPU-resident.

## Non-Goals

- No `CityRenderer` in Spectra.
- No `Building`, `Cim`, `Zone`, `Road`, or `TerraformBrush` concepts in Spectra.
- No game-owned renderer package in Civitas. Existing `src/render*` code is migration debt, not architecture.
- No live renderer path that resolves GPU geometry to CPU pixels for normal presentation.
- No quality compromise hidden as architecture. If performance misses the target, the fix is measured renderer/engine optimization, not moving ownership boundaries until the code becomes convenient.

## Supersedes

The strategic target in this document supersedes the older interactive-WGPU assumption in `2026-06-10-virtualized-splat-rendering-design.md`. That design can still be used as transition/fallback/reference material, but it is no longer the product renderer direction.
