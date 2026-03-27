# Phase 7 — Multiplayer, Modding, Animation & Steam

**Goal:** Add the features that make Ochroma a shippable product: multiplayer co-op building, mod support via Wasm, skeletal animation for citizens/vehicles, and Steam platform integration.

## 7.1 Multiplayer Co-Op

- QUIC transport (already have TCP in vox_net)
- Authoritative server validates all actions
- Client prediction for placement (snap-back on reject)
- Interest management: clients only receive entities in viewport
- Player permissions: mayor, councillor, spectator roles
- Lobby system: create/join/leave games
- Chat: text chat between players
- Target: 4 players building simultaneously without desync

## 7.2 Modding via Wasm

- wasmtime integration (already behind feature flag)
- Mod API: entity queries, mutations, event subscriptions
- Mod packaging: .ochroma_mod archive (manifest + .wasm + assets)
- Mod load order and dependency resolution
- CPU/memory budgets enforced per mod
- Mod manager UI: enable/disable, load order, conflicts

## 7.3 Skeletal Animation

- Bone hierarchy with local/world transforms (already in animation.rs)
- Animation clips with keyframe interpolation (already implemented)
- Citizen animation states: idle, walk, run, sit, work
- Vehicle animations: wheel rotation, door open/close
- Construction animations: crane swing, scaffold build-up
- Blend trees for smooth transitions

## 7.4 Steam Integration

- Steamworks SDK binding
- Achievements (milestone-based)
- Steam Workshop for mod distribution
- Cloud saves
- Rich presence (show city stats in friend list)
- Leaderboards (population, satisfaction, playtime)

## Exit Criteria

- [ ] 4 players can build in the same city via QUIC networking
- [ ] A Wasm mod can add a custom building type
- [ ] Citizens visibly walk between buildings with blended animations
- [ ] Steam achievements trigger on milestones
- [ ] Workshop mods can be uploaded and downloaded
