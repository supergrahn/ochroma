# Phase 27 — Developer Experience

**Goal:** Make Ochroma the easiest game engine to learn and the most productive to work with. Better docs, better tools, better error messages than Unreal.

## 27.1 Interactive Tutorial System
- In-engine tutorial that teaches by doing
- Step-by-step: place first road → zone → watch city grow
- Contextual hints appear when player seems stuck
- Skippable for experienced players

## 27.2 Error Messages That Help
- Every error includes: what happened, why, how to fix it
- Asset validation errors show which splat/band is out of range
- Shader compilation errors map back to material graph nodes
- Save corruption errors explain what was lost and offer recovery

## 27.3 API Documentation
- Every public type documented with examples
- Crate-level architecture docs
- Tutorial for each major subsystem
- Migration guides between engine versions

## 27.4 Performance Inspector
- In-game overlay showing:
  - Frame time breakdown (sort, cull, render, UI, sim)
  - VRAM usage per tile/asset
  - Entity count by type
  - Splat count (visible, culled, total)
  - Simulation tick time breakdown
- Exportable as JSON for profiling tools

## 27.5 Hot-Reload Everything
- Change a .splat_rule → buildings update in real-time
- Edit a spectral material → all instances re-render
- Modify a .wgsl shader → recompile without restart
- Save game state → edit code → reload state into new build

## Exit Criteria
- [ ] New user completes tutorial and has a running city in 10 minutes
- [ ] Every public API type has doc comments with examples
- [ ] Performance inspector shows all metrics listed above
- [ ] Hot-reload works for rules, materials, and shaders
