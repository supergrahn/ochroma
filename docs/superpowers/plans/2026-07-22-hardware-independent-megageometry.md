# Hardware-Independent MegaGeometry Plan

> **Status:** Superseded by
> [City MegaGeometry](./2026-07-20-city-megageometry.md) and its corrected
> [design](../specs/2026-07-20-city-megageometry-design.md).
>
> The former body of this document incorrectly assigned visibility programs,
> clusters, pages, correspondence, and runtime schedules to Forge and the asset
> cooker. That architecture is rejected. It must not be used as implementation
> guidance.

## Authoritative boundary

```text
Forge or another offline authoring tool
  -> ordinary finished mesh + surfaces + game metadata
  -> .vxp
  -> Urban Horizon loads the game object
  -> Ochroma derives deterministic runtime geometry/cache data
  -> Spectra selects detail, residency, and native acceleration
```

There is no MegaGeometry exporter, special authoring bridge, construction
receipt, visibility sidecar, cooked cluster hierarchy, authored LOD chain, or
Forge runtime dependency.

Hardware independence remains mandatory:

- one backend-neutral Ochroma runtime representation and control contract;
- deterministic integer/fixed-point persistent control across CUDA, Vulkan,
  and Metal;
- GPU-side visibility, detail selection, residency, eviction, and work
  compaction;
- native triangle, cluster, or procedural acceleration selected truthfully by
  Spectra for the active device;
- bounded asynchronous host services only where storage or native APIs require
  host submission;
- exact correspondence back to the authoritative finished source mesh;
- Urban Horizon city-scale instance/update behavior;
- no runtime QEM and no asset-authored distance meshes.

All tasks, tests, commands, and acceptance gates now live in the corrected
City MegaGeometry plan. This file remains only to prevent old links from
silently reviving the rejected Forge-export architecture.
