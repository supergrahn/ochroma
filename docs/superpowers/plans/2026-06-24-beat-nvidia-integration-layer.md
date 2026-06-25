# Beat-NVIDIA Integration Layer Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between Spectra's path tracer and the stock NVIDIA path on the layer we actually own — integration, per-scene specialization, and determinism — by *finishing* mostly-built features, not inventing new render tech.
**Done When:**
- **Local (GPU-free) slice:** `cargo test -p vox_sim --test social_network_test --test sharding_test` prints `test social_adjacency_iteration_is_id_sorted ... ok` and `test sharding_rebalance_migration_is_deterministic ... ok`, AND `cargo test -p vox_sim --test sim_replay_hash_test` stays green.
- **Box slice (per item):** a hero-camera frame rendered on `tomespensin` through `spectra-cuda` shows the named effect (e.g. `clas_stats()` returns `(N>0, clusters>0)` in a live `urban_horizon` frame with `OCHROMA_UNMERGE_PROTOS` retired) — witnessed by `Read`-ing the actual PNG, 1 spp + denoise.
**Architecture:** The "better than NVIDIA" thesis lives only on the software/integration layer (NVIDIA owns fixed-function RT/Tensor throughput). Two halves: (A) the **determinism moat** — replace HashMap/HashSet iteration + tie-break-less sorts with id-ordered structures so the sim is replay-exact; pure Rust, no GPU. (B) the **render integration layer** — flip on / finish features that other sessions already landed as code (SER, LEAN, CLAS, CUDA Graphs, DLSS-RR) but that the live resident path doesn't yet drive; CUDA/Slang, box-only.
**Design Document:** n/a (grounded in the 2026-06-24 10-agent code audit; see `IMPORTANT NOTES`).
**Tech Stack:** Rust 1.87, `vox_sim` (pure), `spectra` Slang→{SPIR-V, PTX}, OptiX 9.1, CUDA 13.x.
**Build:** Local: `cargo build -p vox_sim && cargo test -p vox_sim`. Box: `scripts/build-windows-gpu.ps1` + `render-spectra-cuda` skill on `tomespensin`.

---

## IMPORTANT NOTES

<!-- Real API signatures and constraints. Grounded in the 2026-06-24 audit. -->

- **Process-stable nondeterminism gotcha:** `HashMap`/`HashSet` iteration order is *stable within one process run* but varies *across* runs (random `RandomState` seed). So a naive `assert_eq!(run(), run())` in one test process **falsely passes**. Witnesses MUST prove *sorted-order iteration* (e.g. `assert_eq!(got, { let mut s = got.clone(); s.sort(); s })`), which `BTreeMap`/`BTreeSet` guarantee and `HashMap`/`HashSet` do not. `assert!(result.is_some())` and run-twice equality are **forbidden** as determinism witnesses.
- `SocialNetwork.adjacency` is **private**; iteration order is observable only via `pub fn all_citizens(&self) -> Vec<u32>` (does `self.adjacency.keys().copied().collect()`, `social_network.rs:109`).
- `pub fn influence_propagation(&self, satisfaction: &HashMap<u32, f32>) -> HashMap<u32, f32>` — signature unchanged; only the *backing store* of `adjacency` changes.
- `TileCoord` (`sharding.rs:10`) derives `Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize` — **no `Ord`**. `BTreeSet<TileCoord>` requires adding `PartialOrd, Ord` to that derive.
- `pub fn rebalance(&mut self) -> Vec<MigrationRecord>` (`sharding.rs:165`) — signature unchanged; `entities`/`tiles` become `BTreeSet`, and `large_shards` (collected from `self.shards.values()`) must be `.sort()`-ed before processing so split order is id-deterministic.
- `spectra-light-mgr/src/light_tree.rs:136` — `indices.select_nth_unstable_by(mid, …)` is **not stable**; equal `position[axis]` partitions vary across std/LLVM. Fix = append `.then_with(|| a.cmp(&b))` (the `a`/`b` indices) to the comparator.
- `city_sim.rs:489` `order.sort_by(|a,b| a.1.partial_cmp(&b.1)…)` — `Vec::sort_by` is **stable** and the input is id-ordered, so this is **defensive hardening** (make the id tie-break explicit), not a live bug. Label it as such.
- `agent.rs:28` `agents: HashMap<Uuid, Agent>` iterated via `.values()`/`.values_mut()` (independent per-agent mutation) — **audit-only this round**; do NOT change without proving an ordered-iteration determinism risk. `Uuid` is `Ord`, so `BTreeMap<Uuid, Agent>` is a drop-in *if* a risk is found.
- `todo!()` / `unimplemented!()` / empty bodies are **forbidden**.
- **Do NOT commit.** Other sessions are active and the tree is shared; leave changes staged-in-tree for review.
- **Box-only crates** (Part B): anything under `spectra` touching CUDA/OptiX/Slang `.slang` cannot build or be witnessed on the local AMD box — edit under `/home/tom-espen/...`, rsync to `tomespensin`, `touch build.rs` for `.slang`, witness by `Read`-ing the rendered PNG.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `crates/vox_sim/src/social_network.rs` | `adjacency` HashMap → BTreeMap (id-sorted iteration) |
| Test   | `crates/vox_sim/tests/social_network_test.rs` | assert `all_citizens()` is id-sorted after scrambled insertion |
| Modify | `crates/vox_sim/src/sharding.rs` | `entities`/`tiles` HashSet → BTreeSet; `TileCoord` +`Ord`; sort `large_shards` |
| Test   | `crates/vox_sim/tests/sharding_test.rs` | assert rebalance migration set is the deterministic id-sorted upper half |
| Modify | `crates/vox_sim/src/city_sim.rs` | departure sort: explicit `(satisfaction, id)` tie-break (defensive) |
| Modify | `../spectra/rust/spectra-light-mgr/src/light_tree.rs` | `select_nth_unstable_by` id tie-break (box-buildable; CPU crate) |
| Audit  | `crates/vox_sim/src/agent.rs` | report ordered-iteration risk; change only if real |
| Stage  | `../spectra/rust/spectra-optix/src/optix_host.rs` (~1271) | **Part B** CLAS per-tri material fix |
| Stage  | `../spectra/rust/spectra-optix/src/optix_host.rs:447` | **Part B** `allowOpacityMicromaps = 1` + OMM build |
| Stage  | `crates/vox_render/src/resident_renderer.rs` | **Part B** set `ser_enabled` + re-enable CUDA Graphs + `DenoiserMode::DlssRR` |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Id-sorted social iteration | `assert_eq!(net.all_citizens(), sorted_copy)` after inserting `[5,2,8,1,9]` | `assert_eq!(run(),run())` (process-stable → false pass) |
| Deterministic shard rebalance | migrated entity set == upper half of `sorted(entity_ids)` | `assert!(!records.is_empty())` |
| Light-tree tie-break | `cargo build -p spectra-light-mgr` green + equal-position lights → fixed split | comparator compiles |
| CLAS live (Part B) | `clas_stats()` → `(N>0, clusters>0)` in a Read-verified hero PNG | `is_available()` returns a bool |

---

## PART A — Determinism moat (LOCAL, GPU-free, EXECUTE NOW)

## Task A1: Id-sorted social-network iteration

**Files:**
- Modify: `crates/vox_sim/src/social_network.rs`
- Test: `crates/vox_sim/tests/social_network_test.rs`

**Acceptance:** `cargo test -p vox_sim --test social_network_test social_adjacency_iteration_is_id_sorted -- --nocapture` → `test ... ok`; the test inserts citizen ids in scrambled order and asserts `all_citizens()` equals its own sorted copy.

**Wiring requirement:** `adjacency` field type changes to `BTreeMap`; `all_citizens()` (`:109`) and `influence_propagation()` (`:120`) automatically iterate id-sorted. No new function — the existing public API is the wiring.

- [ ] **Step 1: Write the failing test** — append to `social_network_test.rs`:

```rust
#[test]
fn social_adjacency_iteration_is_id_sorted() {
    // Insert in deliberately scrambled id order; iteration must be id-sorted,
    // which BTreeMap guarantees and HashMap (random seed) does not.
    let mut net = SocialNetwork::new();
    for &(a, b) in &[(5u32, 2u32), (8, 1), (9, 3), (2, 5), (1, 8)] {
        net.add_relationship(a, b, Relationship::default()); // use real ctor/api
    }
    let got = net.all_citizens();
    let mut want = got.clone();
    want.sort_unstable();
    assert_eq!(got, want, "adjacency iteration must be id-sorted (BTreeMap), got {got:?}");
}
```

- [ ] **Step 2: Run to verify it fails** (with HashMap, scrambled ids come out hash-ordered):

```bash
cargo test -p vox_sim --test social_network_test social_adjacency_iteration_is_id_sorted 2>&1 | tail -5
```

Expected: FAIL — `assertion failed: got != want` (or a compile error if `add_relationship` differs — read the real ctor first).

- [ ] **Step 3: Implement** — in `social_network.rs`: line 2 `use std::collections::{BTreeMap, HashMap};`; line 59 `adjacency: BTreeMap<u32, Vec<(u32, Relationship)>>`; line 65 `adjacency: BTreeMap::new()`. Keep `HashMap` for `influence_propagation`'s `satisfaction` param + return and `find_communities`' `visited`.

- [ ] **Step 4: Wire** — none beyond the type change; `all_citizens()` / `influence_propagation()` now fold id-sorted by construction.

- [ ] **Step 5: Run — verify pass:**

```bash
cargo test -p vox_sim --test social_network_test -- --nocapture
```

Expected: PASS, new test `... ok`, all existing social tests still green.

- [ ] **Step 6:** Do NOT commit (shared tree). Report the diff.

---

## Task A2: Deterministic shard rebalance

**Files:**
- Modify: `crates/vox_sim/src/sharding.rs`
- Test: `crates/vox_sim/tests/sharding_test.rs`

**Acceptance:** `cargo test -p vox_sim --test sharding_test sharding_rebalance_migration_is_deterministic -- --nocapture` → `test ... ok`; inserting >10k entities in scrambled order and rebalancing migrates exactly the **upper half of the id-sorted entity ids** to the new shard.

**Wiring requirement:** `entities`/`tiles` → `BTreeSet`; `TileCoord` gains `PartialOrd, Ord`; `large_shards` (`:169`) `.sort()`-ed before the split loop. `rebalance()` (`:165`) is the live entry point.

- [ ] **Step 1: Write the failing test** — append to `sharding_test.rs`:

```rust
#[test]
fn sharding_rebalance_migration_is_deterministic() {
    let mut mgr = ShardManager::new();
    let sid = mgr.create_shard(/* tiles set per real api */ Default::default());
    // 10_001 entity ids inserted in scrambled order to force a split.
    let ids: Vec<u64> = (0..10_001u64).map(|i| (i.wrapping_mul(2_654_435_761)) % 1_000_003).collect();
    for &e in &ids { mgr.add_entity(sid, e); } // real add api
    let records = mgr.rebalance();
    let mut migrated: Vec<u64> = records.iter().map(|r| r.entity_id).collect();
    migrated.sort_unstable();
    let mut sorted_ids = ids.clone(); sorted_ids.sort_unstable(); sorted_ids.dedup();
    let want: Vec<u64> = sorted_ids[sorted_ids.len()/2..].to_vec();
    assert_eq!(migrated, want, "rebalance must migrate the id-sorted upper half deterministically");
}
```

- [ ] **Step 2: Run to verify failure** (HashSet → migrates a hash-ordered half):

```bash
cargo test -p vox_sim --test sharding_test sharding_rebalance_migration_is_deterministic 2>&1 | tail -8
```

Expected: FAIL — migrated set differs from the id-sorted upper half.

- [ ] **Step 3: Implement** — `sharding.rs`: `use std::collections::{BTreeSet, HashMap};` (drop `HashSet`); `TileCoord` derive gains `PartialOrd, Ord`; `tiles: BTreeSet<TileCoord>`, `entities: BTreeSet<u64>`; update `new()`/`create_shard()`/`tiles()` signatures + `new_tiles` local to `BTreeSet`; after collecting `large_shards`, add `large_shards.sort_unstable();` (and the same for any small-shard merge list). `entities.iter().copied().collect()` now yields a sorted `Vec` → deterministic `[half..]` split.

- [ ] **Step 4: Wire** — `rebalance()` already calls the changed collections; no new callsite.

- [ ] **Step 5: Run:**

```bash
cargo test -p vox_sim --test sharding_test -- --nocapture
```

Expected: PASS, new test ok, existing sharding tests green.

- [ ] **Step 6:** Do NOT commit.

---

## Task A3: Light-tree id tie-break (spectra, CPU crate)

**Files:**
- Modify: `../spectra/rust/spectra-light-mgr/src/light_tree.rs`

**Acceptance:** `cd ../spectra && cargo build -p spectra-light-mgr` green; comparator at `:136` reads `…partial_cmp(…)…then_with(|| a.cmp(&b))` so equal-position lights partition deterministically across std/LLVM versions.

**Wiring requirement:** the tie-break is inside the live `select_nth_unstable_by` partition that builds the light BVH — no separate callsite.

- [ ] **Step 1:** Read `light_tree.rs:130-145` for the exact closure (incl. the `unwrap_or` arm).
- [ ] **Step 2: Implement** — append `.then_with(|| a.cmp(&b))` to the comparator result (`a`,`b` are the partition indices).
- [ ] **Step 3: Verify** — `cd ../spectra && cargo build -p spectra-light-mgr 2>&1 | tail -3` → `Finished`. If the crate can't build standalone locally, the 1-line change is type-checked by `cargo check -p spectra-light-mgr`; note if box verification is needed.
- [ ] **Step 4:** Do NOT commit.

---

## Task A4: Departure tie-break (defensive hardening)

**Files:**
- Modify: `crates/vox_sim/src/city_sim.rs`

**Acceptance:** `city_sim.rs:489` reads `…partial_cmp(&b.1).unwrap_or(Equal).then(a.0.cmp(&b.0))`; `cargo test -p vox_sim --test city_sim_test` stays green. (Honest label: `sort_by` is already stable + input id-ordered, so this makes the existing determinism *explicit*, independent of `all()`'s ordering.)

- [ ] **Step 1: Implement** — append `.then(a.0.cmp(&b.0))` to the comparator at `:489`.
- [ ] **Step 2: Verify** — `cargo test -p vox_sim --test city_sim_test 2>&1 | tail -5` → green.
- [ ] **Step 3:** Do NOT commit.

---

## Task A5: Agent-manager audit (no edit unless risk proven)

**Files:** `crates/vox_sim/src/agent.rs` (audit)

**Acceptance:** a written finding: do `.values()`/`.values_mut()` (`:64`,`:71`) feed any ordered/collected/accumulated result, or only independent per-agent mutation? If independent → **no change** (record why). If an ordered fold exists → `HashMap<Uuid,Agent>` → `BTreeMap<Uuid,Agent>` (Uuid is `Ord`) + a sorted-iteration witness.

---

## PART B — Render integration layer (BOX-GATED, STAGE ONLY)

> These build/witness only on `tomespensin` (CUDA/OptiX/Slang). Other sessions already landed the underlying code (spectra `0a8a337` SER, `62dcd94` LEAN+kernel-cache, `af8909e` CLAS-LOD, DLSS-RR/SVGF). Each task = finish/flip-on + a Read-verified hero frame. Ordered by leverage.

- [ ] **B1 — CLAS material fix → live (keystone).** Fix per-tri material mis-recovery (`spectra-optix/src/optix_host.rs` ~`build_proto_clas_gas:1271` + `optix/rt/programs/device_programs.cu` closest-hit: clusterId→global-tri base off-by-one). Then retire the `OCHROMA_UNMERGE_PROTOS` merged-GAS workaround so multi-archetype scenes travel per-proto GAS→CLAS→IAS. **Done When:** `clas_stats()` → `(N>0, clusters>0)` in a Read-verified live `urban_horizon` hero frame; `clas_scale_bench` records `build_ms/refit_ms/fps` for 100K–1M. Unblocks 100K–1M instances + the per-frame-rebuild choke.
- [ ] **B2 — SER on the resident path.** Set `ser_enabled` in `resident_renderer.rs` (default true on CUDA) + `SPECTRA_SER=1` in `play-rr-windows.cmd`; the 4-pass software reorder (`ser_reorder.slang`) is already wired but never fired live. **Done When:** A/B hero frame is pixel-identical (perturbation-stable) with measured frame-time delta logged.
- [ ] **B3 — OMM for foliage.** `allowOpacityMicromaps = 1` (`optix_host.rs:447`) + `build_omm_array` pass for `MAT_VEGETATION` BLASes + carry via `InstanceData.gacl_metadata`; remove the software any-hit re-traverse (`megakernel.slang:4064-4080`) on NVIDIA. **Done When:** canopy-heavy hero frame shows leaves with fewer shade invocations (bench AOV) and no alpha-cutout culling.
- [ ] **B4 — Re-enable CUDA Graphs in the city path.** Add a scene-change graph-invalidation guard (drop captured graph on `build_instanced_scene`) and lift the `resident_renderer.rs:388` blanket disable. **Done When:** city hero frame renders with `SPECTRA_CUDA_GRAPHS=1` default-on, no WDDM watchdog reset over a 600-frame run.
- [ ] **B5 — Scene-content kernel specialization.** Extend the 2-way LEAN/FULL toggle (`compile.rs` `lean_variant_defines`) to resolve a minimal define-set from the map's actual material types at scene-commit, compile+cache that variant (`compute_key_with_defines` already keys it). **Done When:** a water-free map's loaded kernel omits the Gaussian/terrain defines (logged spec-vector) with lower VGPR.
- [ ] **B6 — DLSS-RR programmatic toggle** (`DenoiserMode::DlssRR` + fidelity-ladder wire) and **L2 persistence window** (pin TLAS-top + `g_materials`) — both **after** the per-frame BVH rebuild fix (B1), else the cache thrashes.
- [ ] **B7 — AMD floor:** decouple FSR from CUDA interop (`fsr.rs`→`ffx_vk.rs`) so the 780M gets upscaling; ReSTIR R1 (secondary-vertex buffer `restir_pt.slang:1164` + Talbot MIS) gated on the `--compare-map` relMSE gate.
- [ ] **B8 — scatter atomicAdd determinism** (`scatter_placement.slang:205`, `scatter_lod_resolve.slang:94`): post-sort scatter output by `(asset_idx,x,z)` or seed deterministic tile-ID thread assignment, so scatter arrays are replay-stable. Box-gated (Slang).

---

## Self-Review Checklist

- [x] Every Part-A task implements AND wires in the same task (type change *is* the wiring; no "wire later")
- [x] Every Acceptance names a real non-trivial output (sorted-vec equality, migrated-set equality), not "tests pass"
- [x] Witnesses defeat the process-stable HashMap false-pass (sorted-iteration assertion, not run-twice)
- [x] `IMPORTANT NOTES` carries the real signatures (`influence_propagation`, `rebalance`, `TileCoord` derive, `light_tree` comparator)
- [x] `File Map` lists every touched file
- [x] No `todo!()`/`unimplemented!()` in implementation steps
- [x] `Done When` names specific commands + specific observable results (local test names; box `clas_stats()`/Read-verified PNG)
- [x] Box-vs-local split is explicit; no Part-B task is claimed executable locally
