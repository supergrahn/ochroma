# Phase 22 — Distributed Simulation

**Goal:** Scale simulation beyond a single machine. Ochroma simulates millions of citizens across server clusters — something Unreal's single-process architecture cannot do.

## 22.1 Simulation Sharding
- Divide the world into simulation shards (one per CPU core or server)
- Each shard owns a set of tiles and their entities
- Cross-shard communication via message passing
- Load balancing: hot shards split, cold shards merge

## 22.2 Entity Migration
- Citizens move between shards as they cross tile boundaries
- Seamless handoff: no stutter during migration
- State serialisation optimised for frequent small transfers

## 22.3 Deterministic Lockstep
- All shards advance at the same simulation tick
- Deterministic RNG per shard for reproducible behaviour
- Replay: record inputs, replay simulation identically

## Exit Criteria
- [ ] 1 million citizens simulated across 4 shards
- [ ] Citizens migrate between shards without state loss
- [ ] Simulation replay produces identical results
