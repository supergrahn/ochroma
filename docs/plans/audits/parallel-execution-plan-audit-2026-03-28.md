---
blockers: 5
warnings: 6
audited_at: 2026-03-28T14:41:40.423Z
plan_file: parallel-execution-plan.md
---

# Audit: parallel-execution-plan.md

## 🔴 Blockers

- **[stale scope]** Agent 8 (Content Browser) re-implements work already landed in commit `67642ba feat(editor): drag asset from content browser to place in scene`. An agent executing this task will conflict with or duplicate existing code. *Audit what the content browser commit actually shipped and rewrite Agent 8 as a gap-fill task (thumbnails, search, filter) rather than a full build.*

- **[stale scope]** Agent 3 (Animation System) includes "animation state machine" as a deliverable, but `ea2f62a feat(editor): animation state machine editor UI window` is already merged. Executing this task without reading current code risks overwriting or duplicating the state machine. *Read existing animation code first; rewrite the agent task to extend, not rebuild.*

- **[undefined interface]** Agents 1 (Shadow Maps) and 3 (Animation) both modify the splat shader, but there is no shared shader interface or extension protocol defined. Two agents modifying the same GLSL/WGSL entry point in parallel will produce merge conflicts. *Define the shader extension point (e.g., a per-splat lighting hook) before dispatching both agents, or serialize these two tasks.*

- **[undefined symbol]** `engine_runner` is referenced by Agents 2, 3, and 6 as the integration target but does not appear in the CLAUDE.md architecture (vox_core, vox_data, vox_render, vox_app). If this crate or binary entry point doesn't exist, integration steps will have no target. *Identify the actual binary crate name and replace all references.*

- **[missing crate dependency]** Agent 6 (Character Controller) requires Rapier. Cargo.toml inclusion is not confirmed. If Rapier isn't already a dependency, the agent must add it, which modifies Cargo.lock and can break other agents building in parallel. *Verify Rapier is in Cargo.toml before dispatching Batch 2, or make it a pre-batch setup step.*

## 🟡 Warnings

- **[hidden sequential dependency]** Agent 2 (Audio) requires "collision → play sound" integration. This depends on a working collision/physics system that is not guaranteed to exist before Batch 1 completes. The agent may silently stub this or fail at integration. *Either remove the collision-audio hook from Batch 1 scope and add it as a Batch 2 wiring task, or confirm collision events are already emitted.*

- **[architectural complexity underestimated]** Agent 7 specifies "retained-mode UI on top of egui," but egui is immediate-mode by design. Building a retained-mode layer on top requires significant abstraction work that is not scoped. This is a multi-day task, not a parallel batch item. *Descope to: egui-native panel system with game-state binding. Drop the "retained-mode" framing.*

- **[Batch 3 false parallelism]** Agent 15 (Example Games) is in Batch 3 but depends on Batch 1 (animation, audio) and Batch 2 (character controller, UI) being fully integrated and working. If any Batch 1/2 integration fails, Agent 15 blocks. *Move Agent 15 after integration verification, not concurrent with Agents 11-14.*

- **[no rollback strategy]** No agent task specifies what to do if their output is incomplete or breaks the build at integration time. With 5 parallel agents touching core systems, a broken integration is likely on the first attempt. *Add a per-batch integration step with an explicit rollback or revert protocol.*

- **[vague test criteria]** Most agent test criteria are observational ("shadow visible," "hear it louder on one side") rather than automated. These can't be verified by a non-human agent and won't catch regressions. *Each agent should produce at least one `cargo test` that fails before the feature and passes after.*

- **[scope inflation]** Agent 13 (Documentation) includes "Tutorial: Build Your First Game in 30 Minutes" and a "Video script." These are marketing deliverables, not engineering tasks, and don't belong in a parallel engineering batch. *Remove video/tutorial scope; limit to API reference and Getting Started accuracy.*

## 🟢 Ready

The batching strategy is sound: Batch 1 targets independent subsystems (shadows, audio, animation, gizmos, GLTF) with no shared state between agents, and the integration-after-each-batch structure is the right way to catch cross-agent conflicts early. Agent 5 (GLTF Import) and Agent 14 (Performance Optimization) are well-scoped with clear inputs, outputs, and measurable exit criteria. The decision to use established crates (rodio, gltf, rapier, puffin) rather than building from scratch is correct for the target pace.

## Memory Conflicts

- The plan's timeline estimate is **8-14 hours total**; project memory records the user's target as **7 hours for Phases 0-4** starting 2026-03-27, a deadline already passed. The plan does not acknowledge this date or what phases have already been completed. The plan should be scoped against remaining phases only, not treated as a fresh start.
- No engine/game-layer boundary violations found in the plan itself — agent tasks that touch city-specific concepts (buildings in Agent 15) are correctly placed in the game-layer walking sim and showcase demo, consistent with the engine-generality feedback.
