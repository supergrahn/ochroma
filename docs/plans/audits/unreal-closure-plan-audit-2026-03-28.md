---
blockers: 5
warnings: 6
audited_at: 2026-03-28T14:40:52.466Z
plan_file: unreal-closure-plan.md
---

# Audit: unreal-closure-plan.md

## 🔴 Blockers

- **[undefined dependency]** Task 1A references "spectra-gaussian-render source" to copy into the workspace but provides no file path, repository URL, or location. Agents cannot execute "Copy spectra-gaussian-render source" without knowing where it lives. *Specify the absolute path or repo source of the Spectra crate before assigning this task.*

- **[ambiguous branch]** Task 1A presents an OR: "Integrate Spectra's Rust crate OR build wgpu compute sort + rasterise pipeline." These are fundamentally different scopes (days apart in effort). Agents will make diverging decisions and produce incompatible integration points. *Make a hard decision in the plan — pick one path.*

- **[undefined mechanism]** Task 3A says "play animations back on splat groups." GLTF animations drive bone transforms on meshes; splat groups have no rig. The mapping from skeleton transforms to splat positions is an open research problem, not a known integration step. *Either scope this to mesh-based rendering with a fallback renderer, or replace with a concrete approach (e.g., rigid splat group transforms driven by bone roots only).*

- **[missing file]** Task 1C references `character_hero.ply` as a verification asset with no path. If it doesn't exist in the repo, the verification step cannot run. *Specify the path or note that the file must be added as part of this task.*

- **[undefined scope]** Task 4D "Second example game (different genre)" has zero specification — no genre, no scope, no exit criteria. Agents cannot begin work on it. *Add at minimum: genre, core loop, 3 exit criteria.*

## 🟡 Warnings

- **[timeline conflict]** The plan estimates 20–25 total hours. Project memory records a 7-hour target for Phases 0–4 starting 2026-03-27. The plan's own timeline is 3–4x over the stated constraint. *Either re-scope phases to fit 7 hours, or update the timeline expectation explicitly so agents prioritize correctly.*

- **[missing rollback]** Task 2A "Replace AABB with Rapier rigid bodies" is a wholesale replacement of collision primitives with no rollback strategy and no mention of what AABB surface area exists. If Rapier integration breaks rendering, the path back is unclear. *Add: enumerate files/systems to touch, define a feature flag or parallel-run period.*

- **[vague exit criteria]** Phase 1 exit criterion is "60fps at 1M splats" but no target hardware is specified. A GTX 1070 and an RTX 4090 give wildly different results for the same code. Agents will mark 1B complete on different hardware with different conclusions. *Specify minimum target GPU.*

- **[layer boundary risk]** Tasks 3C (NavMesh) and 3D (AI FSM) belong ambiguously between engine and game layers. If agents place `NavMesh` generation or `AiFsm` trait in `vox_core` or `vox_render`, it violates the engine-generality rule. *Explicitly note which crate these land in (suggest `vox_app` or a new `vox_sim` game-layer crate).*

- **[unverified crate]** Task 3C specifies "Recast-rs integration." `recast-rs` is a niche binding with infrequent maintenance. No fallback is listed if crate is unmaintained, yanked, or incompatible with current Rust edition. *Add a fallback (e.g., raw `recastnavigation` FFI, or `navmesh` crate) and check crate health before assigning.*

- **[no inter-phase contracts]** Each phase produces outputs consumed by the next, but no interface contracts are defined (e.g., what API does Phase 1 expose for Phase 2's egui viewport?). Parallel agents across phases will make incompatible assumptions. *Add a "Phase N Output API" section to each phase.*

## 🟢 Ready

The skip list (Marketplace, Networking, 9 platforms) is well-reasoned and directly reduces scope to a survivable set — agents will not waste time on excluded areas. The "Auto-solved by Spectra" section cleanly documents what not to re-implement, preventing duplicate work. Phase 2's tasks (2B–2D) are concrete enough to delegate independently with egui as the clear UI layer.

## Memory Conflicts

- **Timeline:** Project memory says 7 hours for Phases 0–4 starting 2026-03-27. This plan estimates 20–25 hours. The plan is in direct conflict with the recorded pace expectation and should be reconciled before execution begins.
- **Engine generality:** Tasks 3C and 3D (NavMesh, AI FSM) are not explicitly placed in the game layer. Given the recorded rule that engine crates must stay game-agnostic, placing pathfinding or AI traits in `vox_core`/`vox_render` would violate the project constraint. The plan does not call this out.
