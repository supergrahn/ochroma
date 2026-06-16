# Vice Mayor — Local-Trainable Style-Playing AI Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small, CPU-trainable policy that plays Urban Horizon through the R24 GameCommand surface in distinct delegable styles, swappable into the R34 Assistant Mayor seam, trained on the deterministic sim as a gym + on replays as an offline dataset.
**Done When:** `cargo run -p urban_horizon --bin vice_mayor -- play --style frugal --episodes 1 --print-trajectory` prints a line `EPISODE style=frugal ticks=8640 actions=<N>0 final_pop=<N> final_treasury=<F> return=<R> hash=<0x...>` with `actions > 0`, every issued command id in the printed replay log, and a re-run with the same seed reproduces the identical `hash=`. AND `cargo run -p urban_horizon --bin vice_mayor -- compare --styles frugal,carefirst --episodes 8` prints a table where mean treasury and mean satisfaction DIFFER between the two styles (asserted `abs(delta) > 0`).
**Architecture:** A pure gym wrapper over `GameState` (no new sim), a fixed-order observation featurizer and an id-sorted discrete action table with a legal-action mask that reuses R24 validation + R34's budget gate, a potential-based shaped reward parameterised per style, a hand-written-matmul CPU inference policy (zero ML runtime dep), behaviour-cloning + offline-RL training behind `cfg(feature = "vice_train")`, and a `LearnedMayor` that swaps into the R34 `execute_plan` seam.
**Design Document:** `docs/superpowers/specs/2026-06-15-vice-mayor-rl-gym-design.md`
**Tech Stack:** Rust (the `urban_horizon` game crate), no new runtime ML dependency; training tools dev-only behind `feature = "vice_train"`.
**Build:** `cargo build -p urban_horizon` / `cargo test -p urban_horizon` / training: `cargo run -p urban_horizon --features vice_train --bin vice_mayor -- bc ...`
**Repo:** `~/Ochroma/projects/urban_horizon_r39` (the GAME repo — NOT the engine `~/src/ochroma`). All paths below are relative to the game repo.

---

## IMPORTANT NOTES

- This is the GAME repo `urban_horizon`. The enablers are real and SHIPPED here: `driver::execute_plan(view, game, sdf, registry, &plan) -> PlanReport`, `driver::undo_plan(game, &report)`, `driver::replay_commands(...)`, `validate_command(cmd, registry) -> Result<(), String>` (all in `src/command/driver.rs`); `GameState::integration_hash() -> u64`, `wellbeing_history() -> &[f32]`, `policy_context() -> CityContext`, `coverage.coverage_ratio(DependentKind) -> f32`, `last_stats: CareStats`, `cim.population() -> u64`, `sim.budget.funds: f64`, `treasury.cash: f64` (in `src/game/mod.rs`); `whatif::rewind_to(rep, t)`, `whatif::branch(...)`, `whatif::diff(...)` (in `src/game/whatif.rs`); `AssistantMayor`/`Autonomy`/`Recommendation` (in `src/mayor/mod.rs`).
- `PlannedCommand::verb(id)` builds a no-arg command; `Plan { goal, commands }`; `PlanReport::accepted_count()`, `.replay_log`, `.undo_mark`. Do NOT invent new driver APIs — route every agent action through `execute_plan`.
- The budget gate to copy for the mask is in `mayor::recommend`: `tier.cost.charge(ctx.population, ctx.total_jobs) <= funds + BUDGET_EPS` where `BUDGET_EPS = 1e-6`. Reuse it; do not reinvent affordability.
- Observation features MUST be pulled in a FIXED, id-sorted order with FIXED normalisation constants (no per-run statistics) or saved policies silently break and runs stop reproducing.
- The action table MUST be built id-sorted from `GameRegistry::standard()` with `no-op` at index 0, so action indices are stable across builds.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.
- The inference `VicePolicy` forward pass is hand-written `f32` matmul — NO `burn`/`tch`/`candle` in the default build. Training-only crates go under `[features] vice_train = [...]` and `#[cfg(feature = "vice_train")]`.
- Determinism witness: any "this run scored R" claim is validated by re-running the policy's `replay_log` through `replay_commands` and asserting the same `integration_hash` — this is the anti-reward-hacking audit.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/vice/mod.rs` | module root; re-exports gym/spaces/reward/policy/offline |
| Create | `src/vice/spaces.rs` | `Observation`, `ActionMask`, `ActionTable`, `observe`, `legal_mask` |
| Create | `src/vice/reward.rs` | `RewardWeights`, `Style`, `potential`, `shaped_reward` |
| Create | `src/vice/gym.rs` | `CityEnv` reset/step over `execute_plan` + sim ticks |
| Create | `src/vice/policy.rs` | `VicePolicy` (load/save/act/logits), hand-written matmul |
| Create | `src/vice/offline.rs` | `replay_to_transitions`, `Transition` (offline dataset from `.civreplay`) |
| Create | `src/vice/train.rs` | `#[cfg(feature = "vice_train")]` BC + filtered-BC trainers |
| Create | `src/bin/vice_mayor.rs` | runnable `play` / `compare` / `bc` / `offline` / `measure` CLI |
| Modify | `src/lib.rs` | `pub mod vice;` (and `mod` wiring) |
| Modify | `src/mayor/mod.rs` | add `LearnedMayor` behind the existing `Autonomy` + `execute_plan` seam |
| Modify | `Cargo.toml` | `[features] vice_train`, dev-only training deps, the `vice_mayor` bin |
| Test | `src/vice/spaces.rs` (mod tests) | featurizer/mask real-value tests |
| Test | `src/vice/gym.rs` (mod tests) | step determinism + reward monotonicity |
| Test | `src/vice/policy.rs` (mod tests) | masked-argmax never illegal, save/load round-trip |
| Test | `src/vice/offline.rs` (mod tests) | replay→transition real-action match |
| Test | `src/mayor/mod.rs` (mod tests) | `LearnedMayor` enacts/reverses through the seam |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Featurizer | infant-coverage feature == `0.10 ± 1e-6` on the seeded crisis city | `assert_eq!(obs.len(), 48)` |
| Legal mask | unaffordable `tier1` bit == false on a low-funds city; affordable == true on a rich one | `assert!(mask.legal_count() > 0)` |
| Env determinism | same seed + same action seq ⇒ identical `integration_hash` | `assert!(env.step(0,s).done == false)` |
| Reward monotonicity | a coverage-raising step's reward > the no-op step's reward on the same prev state | `assert!(r.is_finite())` |
| Masked policy | `act()` returns an index whose mask bit is true, for 1000 random weight nets | `assert!(a < table.len())` |
| Policy round-trip | `save` then `load` yields byte-identical logits on a fixed obs | `assert!(load().is_ok())` |
| Replay→dataset | a logged `TaxAdjust` becomes a transition whose decoded action id is `view.tax.*.inc/dec` | `assert!(!ds.is_empty())` |
| Style distinctness | `frugal` vs `carefirst` pick different actions on ≥ 30% of an episode's decisions | both return identical action vectors |
| Seam swap | `LearnedMayor` Auto enacts via `execute_plan`, action in `replay_log`, reversible by one `undo_plan` | `recommend()` non-empty |

---

## Task 1: Observation featurizer (fixed-order, fixed-normalisation)

**Files:**
- Create: `src/vice/mod.rs`, `src/vice/spaces.rs`
- Modify: `src/lib.rs`

**Acceptance:** `cargo test -p urban_horizon vice_observe_infant_crisis -- --nocapture` → prints `infant_cov_feat=0.100000` and passes (the seeded 10/100 coverage shows up at the fixed feature index).

**Wiring requirement:** `observe` must be re-exported from `src/vice/mod.rs` and `pub mod vice;` added to `src/lib.rs`. `todo!()`/empty bodies = task failure.

- [ ] **Step 1: Write the failing test** (mirror the mayor test's `city_with_low_childcare` seeding: 100 infants demanded, 10 served)

```rust
#[test]
fn vice_observe_infant_crisis() {
    use crate::game_mechanics::care::service::DependentKind;
    let mut game = crate::game::GameState::new_small();
    game.coverage.demand.insert(DependentKind::Infant, 100);
    game.coverage.served.insert(DependentKind::Infant, 10);
    let obs = observe(&game);
    let idx = INFANT_COVERAGE_IDX; // documented const index
    println!("infant_cov_feat={:.6}", obs.as_slice()[idx]);
    assert!((obs.as_slice()[idx] - 0.10).abs() < 1e-6,
        "expected 0.10, got {}", obs.as_slice()[idx]);
    assert_eq!(obs.as_slice().len(), Observation::DIM);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test -p urban_horizon vice_observe_infant_crisis 2>&1 | tail -5
```

Expected: FAIL — `cannot find function observe` / `cannot find module vice`.

- [ ] **Step 3: Implement** `Observation`, `Observation::DIM`, the documented feature index consts, and `observe(&GameState)` pulling care ratios, `last_stats` (normalised by pop), fiscal (`log1p`-scaled funds/cash/income/CoL), last-K `wellbeing_history`, population + weekly delta, and id-sorted ladder tiers. FIXED constants only.

- [ ] **Step 4: Wire** — `src/vice/mod.rs`: `pub mod spaces; pub use spaces::{observe, Observation, ...};`. `src/lib.rs`: add `pub mod vice;`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon vice_observe_infant_crisis -- --nocapture
```

Expected: PASS, output shows `infant_cov_feat=0.100000` (not 0, not NaN).

- [ ] **Step 6: Commit**

```bash
git add src/vice/mod.rs src/vice/spaces.rs src/lib.rs
git commit -m "feat(vice): fixed-order city-state observation featurizer"
```

---

## Task 2: Action table + legal-action mask (reuses R24 validate + R34 budget gate)

**Files:**
- Modify: `src/vice/spaces.rs`
- Test: `src/vice/spaces.rs` (mod tests)

**Acceptance:** `cargo test -p urban_horizon vice_mask_affordability -- --nocapture` → prints `rich_tier1_legal=true poor_tier1_legal=false` and passes (a `view.ladder.care.childcare_subsidy.tier1` action is legal with 100k funds, illegal with 5.0 funds — the same affordability the mayor enforces).

**Wiring requirement:** `legal_mask` and `ActionTable::build` must be called by the test against `GameRegistry::standard()`; the mask MUST set `-inf`-equivalent (bit=false) for unaffordable tiers. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vice_mask_affordability() {
    use crate::game_mechanics::care::service::DependentKind;
    let reg = crate::command::GameRegistry::standard();
    let table = ActionTable::build(&reg);
    let a = table.index_of("view.ladder.care.childcare_subsidy.tier1")
        .expect("tier1 action present in table");

    let mut rich = crate::game::GameState::new_small();
    rich.coverage.demand.insert(DependentKind::Infant, 100);
    rich.coverage.served.insert(DependentKind::Infant, 10);
    rich.sim.budget.funds = 100_000.0;
    let mut poor = rich.clone();
    poor.sim.budget.funds = 5.0;

    let rich_legal = legal_mask(&rich, &table, &reg).is_legal(a);
    let poor_legal = legal_mask(&poor, &table, &reg).is_legal(a);
    println!("rich_tier1_legal={rich_legal} poor_tier1_legal={poor_legal}");
    assert!(rich_legal, "tier1 must be legal when affordable");
    assert!(!poor_legal, "tier1 must be masked when unaffordable");
    assert!(legal_mask(&poor, &table, &reg).is_legal(0), "no-op (idx 0) always legal");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_mask_affordability 2>&1 | tail -5
```

Expected: FAIL — `cannot find function legal_mask` / no `ActionTable`.

- [ ] **Step 3: Implement** `ActionTable::build` (id-sorted delegable verbs, no-op at 0, `index_of`, `command`), and `legal_mask` reusing: `validate_command` guard, the `tier.cost.charge(ctx.population, ctx.total_jobs) <= funds + 1e-6` gate (copy `BUDGET_EPS` semantics from `mayor`), and the already-on-this-tier no-op predicate.

- [ ] **Step 4: Wire** — re-export `ActionTable`, `ActionMask`, `legal_mask` from `src/vice/mod.rs`.

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon vice_mask_affordability -- --nocapture
```

Expected: PASS, `rich_tier1_legal=true poor_tier1_legal=false`.

- [ ] **Step 6: Commit**

```bash
git add src/vice/spaces.rs src/vice/mod.rs
git commit -m "feat(vice): id-sorted action table + legal mask (reuses R24/R34 gates)"
```

---

## Task 3: Style reward (potential-based, anti-hacking) + monotonicity

**Files:**
- Create: `src/vice/reward.rs`
- Modify: `src/vice/mod.rs`

**Acceptance:** `cargo test -p urban_horizon vice_reward_monotone -- --nocapture` → prints `r_improve=<positive> r_noop=<smaller>` and asserts a step that raises infant coverage scores strictly higher than the no-op step under `carefirst` weights.

**Wiring requirement:** `shaped_reward` and `Style` re-exported from `src/vice/mod.rs`; `Style::from_str` parses `frugal|carefirst|growth|balanced`. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vice_reward_monotone() {
    use crate::game_mechanics::care::service::DependentKind;
    let mut prev = crate::game::GameState::new_small();
    prev.coverage.demand.insert(DependentKind::Infant, 100);
    prev.coverage.served.insert(DependentKind::Infant, 10); // 0.10
    let mut better = prev.clone();
    better.coverage.served.insert(DependentKind::Infant, 80); // 0.80
    let w = Style::from_str("carefirst").unwrap().weights();
    let r_improve = shaped_reward(&prev, &better, 0.0, &w);
    let r_noop = shaped_reward(&prev, &prev, 0.0, &w);
    println!("r_improve={r_improve} r_noop={r_noop}");
    assert!(r_improve > r_noop, "raising coverage must beat no-op: {r_improve} !> {r_noop}");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_reward_monotone 2>&1 | tail -5
```

Expected: FAIL — no `shaped_reward`/`Style`.

- [ ] **Step 3: Implement** `RewardWeights`, `Style` (4 variants → distinct weight vectors + one-hot), `potential(game, w)` over the two-pillars terms + a treasury-FLOOR penalty (penalty only, never reward), and `shaped_reward = gamma*Phi(next) - Phi(prev) - lambda*action_cost`.

- [ ] **Step 4: Wire** — re-export from `src/vice/mod.rs`.

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon vice_reward_monotone -- --nocapture
```

Expected: PASS, `r_improve` positive and `> r_noop`.

- [ ] **Step 6: Commit**

```bash
git add src/vice/reward.rs src/vice/mod.rs
git commit -m "feat(vice): potential-based style reward (anti-reward-hacking shape)"
```

---

## Task 4: CityEnv reset/step (over execute_plan) + determinism

**Files:**
- Create: `src/vice/gym.rs`
- Modify: `src/vice/mod.rs`

**Acceptance:** `cargo test -p urban_horizon vice_env_deterministic -- --nocapture` → prints two equal hashes `hash_a=0x... hash_b=0x...` after applying the SAME fixed action sequence to two fresh envs of the same seed, and asserts they are equal.

**Wiring requirement:** `CityEnv::step` MUST route the decoded action through `driver::execute_plan` (not a private dispatch), and advance the sim by `ticks_per_decision`. `CityEnv::integration_hash` delegates to `GameState::integration_hash`. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vice_env_deterministic() {
    let style = Style::from_str("balanced").unwrap();
    let seq = [0usize, 1, 0, 2, 1]; // fixed action indices (legal-or-no-op handled by step)
    let run = |seed: u64| {
        let mut env = CityEnv::new(seed, 5_000, 144); // ~weekly cadence
        env.reset();
        for &a in &seq { let _ = env.step(a, style); }
        env.integration_hash()
    };
    let hash_a = run(7);
    let hash_b = run(7);
    println!("hash_a={hash_a:#018x} hash_b={hash_b:#018x}");
    assert_eq!(hash_a, hash_b, "same seed + same actions must reproduce");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_env_deterministic 2>&1 | tail -5
```

Expected: FAIL — no `CityEnv`.

- [ ] **Step 3: Implement** `CityEnv::new/reset/step/integration_hash/action_table`. `step` decodes via `ActionTable`, builds a one-command `Plan`, calls `execute_plan`, advances ticks, computes `shaped_reward` from a cloned pre-state, recomputes the next mask, returns `StepOut`. An illegal action ⇒ treated as no-op + small penalty (no panic).

- [ ] **Step 4: Wire** — re-export `CityEnv`, `StepOut` from `src/vice/mod.rs`.

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon vice_env_deterministic -- --nocapture
```

Expected: PASS, the two hashes are equal and non-zero.

- [ ] **Step 6: Commit**

```bash
git add src/vice/gym.rs src/vice/mod.rs
git commit -m "feat(vice): CityEnv gym over R24 execute_plan, determinism-verified"
```

---

## Task 5: CPU inference policy (hand-written matmul) — masked-argmax + round-trip

**Files:**
- Create: `src/vice/policy.rs`
- Modify: `src/vice/mod.rs`

**Acceptance:** `cargo test -p urban_horizon vice_policy_legal_and_roundtrip -- --nocapture` → prints `picked=<idx> legal=true logits_match=true` and asserts (a) over 1000 randomised-but-fixed-seed weight nets the picked action is ALWAYS legal under a mask that forbids it, and (b) save→load yields byte-identical logits.

**Wiring requirement:** `VicePolicy::act` must apply the mask BEFORE argmax so an illegal index is unreachable. No external ML crate. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vice_policy_legal_and_roundtrip() {
    let reg = crate::command::GameRegistry::standard();
    let table = ActionTable::build(&reg);
    let style = Style::from_str("frugal").unwrap();
    // a deterministic pseudo-net seeded from a u64 (no RNG crate): fill weights from a hash
    let pol = VicePolicy::deterministic_fill(0xABCD, Observation::DIM + STYLE_N, table.len());
    let game = crate::game::GameState::new_small();
    let obs = observe(&game);
    // force-mask everything except a single legal index to prove masking bites
    let mut mask = ActionMask::all_illegal(table.len());
    mask.set_legal(3, true); mask.set_legal(0, true);
    let picked = pol.act(&obs, style, &mask);
    println!("picked={picked} legal={}", mask.is_legal(picked));
    assert!(mask.is_legal(picked), "policy picked an illegal action {picked}");

    let dir = std::env::temp_dir().join("vice_rt.vicepolicy");
    pol.save(&dir).unwrap();
    let pol2 = VicePolicy::load(&dir).unwrap();
    let l1 = pol.logits(&obs, style); let l2 = pol2.logits(&obs, style);
    let logits_match = l1 == l2;
    println!("logits_match={logits_match}");
    assert!(logits_match, "save/load changed logits");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_policy_legal_and_roundtrip 2>&1 | tail -5
```

Expected: FAIL — no `VicePolicy`.

- [ ] **Step 3: Implement** `VicePolicy` (2-hidden-layer MLP, `tanh`), hand-written `f32` matmul forward, `logits`, masked-argmax `act` (`-inf` on illegal then argmax), `save`/`load` (versioned little-endian flat dump), `deterministic_fill` test helper, and `ActionMask::all_illegal`/`set_legal`.

- [ ] **Step 4: Wire** — re-export `VicePolicy` from `src/vice/mod.rs`.

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon vice_policy_legal_and_roundtrip -- --nocapture
```

Expected: PASS, `legal=true logits_match=true`.

- [ ] **Step 6: Commit**

```bash
git add src/vice/policy.rs src/vice/mod.rs
git commit -m "feat(vice): CPU matmul inference policy with masked argmax + save/load"
```

---

## Task 6: Offline dataset from replays (`.civreplay` → labelled transitions)

**Files:**
- Create: `src/vice/offline.rs`
- Modify: `src/vice/mod.rs`

**Acceptance:** `cargo test -p urban_horizon vice_replay_to_transitions -- --nocapture` → builds a session replay (the same recipe as the R28 whatif gate: zone + `adjust_tax(Industry,1)` + bond), converts it, prints `transitions=<N>0 has_tax_action=true`, and asserts at least one transition's decoded action id is a `view.tax.industry.*` verb (the logged `TaxAdjust` round-trips to an action index).

**Wiring requirement:** `replay_to_transitions` must re-simulate via the existing `whatif::rewind_to` semantics (pure re-simulation, no snapshot) to recover the observation at each authored tick. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vice_replay_to_transitions() {
    use crate::lot::occupancy::Sector;
    use crate::scenario::Scenario;
    use vox_sim::zoning::ZoneType;
    let mut g = Scenario::Demo.build();
    let initial = g.to_save();
    for _ in 0..40 { g.tick(); }
    g.zone_area(ZoneType::CommercialLocal, &[[60.0,60.0],[90.0,60.0],[90.0,90.0],[60.0,90.0]]);
    g.adjust_tax(Sector::Industry, 1);
    g.issue_bond(7, 25_000.0, 0.035, 96);
    for _ in 0..200 { g.tick(); }
    let rep = g.to_replay("vice ds", initial);

    let ds = replay_to_transitions(&rep, Style::from_str("balanced").unwrap());
    let has_tax = ds.iter().any(|t| {
        let reg = crate::command::GameRegistry::standard();
        let table = ActionTable::build(&reg);
        table.command(t.action).id.starts_with("view.tax.industry")
    });
    println!("transitions={} has_tax_action={has_tax}", ds.len());
    assert!(!ds.is_empty(), "replay produced no transitions");
    assert!(has_tax, "logged industry TaxAdjust must map to a tax action index");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_replay_to_transitions 2>&1 | tail -5
```

Expected: FAIL — no `replay_to_transitions`.

- [ ] **Step 3: Implement** `Transition` and `replay_to_transitions`: walk the replay's authored commands, `rewind_to` the tick before each, `observe`, map the `AuthoredAction` to the matching `ActionTable` index, label `reward` via `shaped_reward(prev, after)` and compute episode `ret_to_go`. Unmappable authored actions (spatial v1-out) are skipped, not errored.

- [ ] **Step 4: Wire** — re-export `replay_to_transitions`, `Transition` from `src/vice/mod.rs`.

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon vice_replay_to_transitions -- --nocapture
```

Expected: PASS, `transitions` > 0 and `has_tax_action=true`.

- [ ] **Step 6: Commit**

```bash
git add src/vice/offline.rs src/vice/mod.rs
git commit -m "feat(vice): offline dataset builder from CivReplay via what-if rewind"
```

---

## Task 7: Behaviour-cloning trainer (dev-only) producing a working policy

**Files:**
- Create: `src/vice/train.rs` (`#[cfg(feature = "vice_train")]`)
- Modify: `src/vice/mod.rs`, `Cargo.toml` (add `[features] vice_train`)

**Acceptance:** `cargo test -p urban_horizon --features vice_train vice_bc_imitates_expert -- --nocapture` → trains a BC policy on transitions generated by the existing rule `AssistantMayor(Auto)` acting on the low-childcare city, prints `top1_match=<>0.7>`, and asserts the trained policy's top-1 action on a held-out crisis state matches the rule mayor's chosen action with > 70% agreement across held-out states.

**Wiring requirement:** Training emits a real `.vicepolicy` file loadable by `VicePolicy::load`; the trainer uses ONLY logged transitions (no live env needed). `todo!()` = failure.

- [ ] **Step 1: Write the failing test** — generate expert (obs, action) pairs by querying the rule `AssistantMayor::recommend` top action (mapped to an action index) across a spread of seeded crisis states; split train/held-out; train BC; measure top-1 agreement on held-out.

```rust
#[cfg(feature = "vice_train")]
#[test]
fn vice_bc_imitates_expert() {
    // build expert dataset from AssistantMayor(Auto) decisions across seeded states
    let (train, held) = build_expert_dataset_split();
    let pol = train_bc(&train, /*epochs*/ 200, /*lr*/ 0.05);
    let mut hits = 0usize;
    for (obs, mask, expert_a) in &held {
        let a = pol.act(obs, Style::from_str("carefirst").unwrap(), mask);
        if a == *expert_a { hits += 1; }
    }
    let top1 = hits as f32 / held.len() as f32;
    println!("top1_match={top1:.3}");
    assert!(top1 > 0.7, "BC must imitate the expert, got top1={top1}");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon --features vice_train vice_bc_imitates_expert 2>&1 | tail -5
```

Expected: FAIL — no `train_bc`/feature not wired.

- [ ] **Step 3: Implement** `train_bc` (cross-entropy on masked logits, plain SGD over the hand-written net's gradients — a small manual backward pass for a 2-layer MLP is < 100 lines and needs no ML crate; OR gate `burn` behind `vice_train` if preferred), `build_expert_dataset_split` (uses `AssistantMayor::recommend` + `ActionTable::index_of`), and `Cargo.toml` `[features] vice_train = []` (plus the bin below).

- [ ] **Step 4: Wire** — `#[cfg(feature="vice_train")] pub mod train;` in `src/vice/mod.rs`.

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon --features vice_train vice_bc_imitates_expert -- --nocapture
```

Expected: PASS, `top1_match` > 0.7.

- [ ] **Step 6: Commit**

```bash
git add src/vice/train.rs src/vice/mod.rs Cargo.toml
git commit -m "feat(vice): behaviour-cloning trainer imitates the rule mayor (>70% top-1)"
```

---

## Task 8: `vice_mayor` binary — play + compare (the Done-When proof)

**Files:**
- Create: `src/bin/vice_mayor.rs`
- Modify: `Cargo.toml` (declare the bin)

**Acceptance:** `cargo run -p urban_horizon --bin vice_mayor -- play --style frugal --episodes 1 --seed 7 --print-trajectory` prints `EPISODE style=frugal ticks=<T> actions=<N>0 final_pop=<N> final_treasury=<F> return=<R> hash=<0x...>` with `actions > 0`; re-running with `--seed 7` prints the identical `hash=`. AND `cargo run -p urban_horizon --bin vice_mayor -- compare --styles frugal,carefirst --episodes 8` prints a two-row table whose treasury and satisfaction columns DIFFER (the binary asserts `abs(delta) > 0` and exits non-zero if not).

**Wiring requirement:** `play` runs `CityEnv` with a loaded `.vicepolicy` (or, if none supplied, the BC-trained-on-the-fly policy / rule-fallback), drives via `step`, and prints the replay log; `compare` runs both styles and computes the means. `todo!()` = failure.

- [ ] **Step 1: Write the failing test** — an integration test invoking the binary's `play` and `compare` entrypoints as library functions (`run_play(args)`, `run_compare(args)`), asserting the printed `actions > 0`, the re-run hash equality, and the cross-style delta.

```rust
#[test]
fn vice_play_and_compare_distinct() {
    let r1 = run_play(PlayArgs{ style:"frugal".into(), seed:7, episodes:1, ..Default::default() });
    let r2 = run_play(PlayArgs{ style:"frugal".into(), seed:7, episodes:1, ..Default::default() });
    println!("actions={} hash1={:#018x} hash2={:#018x}", r1.actions, r1.hash, r2.hash);
    assert!(r1.actions > 0, "policy issued no actions");
    assert_eq!(r1.hash, r2.hash, "same seed must reproduce");
    let cmp = run_compare(&["frugal","carefirst"], 8);
    println!("d_treasury={} d_sat={}", cmp.d_treasury, cmp.d_sat);
    assert!(cmp.d_treasury.abs() > 0.0 || cmp.d_sat.abs() > 0.0, "styles must differ");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_play_and_compare_distinct 2>&1 | tail -5
```

Expected: FAIL — no `run_play`/`run_compare`.

- [ ] **Step 3: Implement** the bin: arg parsing (`play|compare|bc|offline|measure`), `run_play` (build env, load/obtain policy, loop `act`→`step` to `done`, collect replay log + final aggregates + `integration_hash`, print the `EPISODE ...` line), `run_compare` (mean treasury/satisfaction per style over N episodes, print table, assert delta).

- [ ] **Step 4: Wire** — `Cargo.toml`: `[[bin]] name = "vice_mayor" path = "src/bin/vice_mayor.rs"`.

- [ ] **Step 5: Run — the human-visible proof**

```bash
cargo run -p urban_horizon --bin vice_mayor -- play --style frugal --episodes 1 --seed 7 --print-trajectory
cargo run -p urban_horizon --bin vice_mayor -- compare --styles frugal,carefirst --episodes 8
```

Expected: the `EPISODE ...` line with `actions > 0`; identical hash on re-run; a compare table with differing treasury/satisfaction.

- [ ] **Step 6: Commit**

```bash
git add src/bin/vice_mayor.rs Cargo.toml
git commit -m "feat(vice): vice_mayor bin — play + compare prove styles play distinctly"
```

---

## Task 9: `LearnedMayor` swap into the R34 seam (production integration)

**Files:**
- Modify: `src/mayor/mod.rs`
- Test: `src/mayor/mod.rs` (mod tests)

**Acceptance:** `cargo test -p urban_horizon learned_mayor_enacts_and_reverses -- --nocapture` → a `LearnedMayor` at `Autonomy::Auto` on the low-childcare city enacts a higher tier through `driver::execute_plan`, prints `accepted=1 in_log=true`, the enacted verb is in `report.replay_log`, and one `driver::undo_plan` restores the byte-identical pre-Mayor save (the exact contract the rule mayor already satisfies).

**Wiring requirement:** `LearnedMayor::act` MUST go through `driver::execute_plan` (same seam as `AssistantMayor::act`) — validated, replay-logged, reversible. `Autonomy::Off` is a real no-op (returns `None`, mutates nothing). `todo!()` = failure.

- [ ] **Step 1: Write the failing test** (parallel to `auto_action_is_reversible` in the existing mayor tests, but driving the learned policy)

```rust
#[test]
fn learned_mayor_enacts_and_reverses() {
    let (mut view, mut game, sdf, registry) = city_with_low_childcare(100_000.0);
    let pol = VicePolicy::deterministic_fill(0x1, /*in*/0, /*out*/0); // or load a trained one
    let mayor = LearnedMayor::new(pol, Style::from_str("carefirst").unwrap(), Autonomy::Auto);
    let save_before = game.to_save().to_json();
    let report = mayor.act(&mut view, &mut game, &sdf, &registry)
        .expect("Auto enacts and returns a PlanReport");
    println!("accepted={} in_log={}", report.accepted_count(),
        !report.replay_log.is_empty());
    assert_eq!(report.accepted_count(), 1);
    assert!(!report.replay_log.is_empty(), "enacted verb must be replay-logged");
    crate::command::driver::undo_plan(&mut game, &report);
    assert_eq!(game.to_save().to_json(), save_before, "one undo restores pre-Mayor state");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon learned_mayor_enacts_and_reverses 2>&1 | tail -5
```

Expected: FAIL — no `LearnedMayor`.

- [ ] **Step 3: Implement** `LearnedMayor { policy, style, autonomy }` with `recommend` (observe → masked logits → top-K legal actions as `Recommendation`s) and `act` (Off/Advise = `None`; Auto = build a one-command `Plan` from the top legal action and call `driver::execute_plan`). Reuse the exact `Off`/`Advise`/`Auto` contract of `AssistantMayor`.

- [ ] **Step 4: Wire** — `LearnedMayor` lives in `src/mayor/mod.rs` beside `AssistantMayor`; the app shell's mayor dispatch can construct either (a follow-up UI task selects which; this task delivers the type + the seam, tested).

- [ ] **Step 5: Run**

```bash
cargo test -p urban_horizon learned_mayor_enacts_and_reverses -- --nocapture
```

Expected: PASS, `accepted=1 in_log=true`, and the save round-trips.

- [ ] **Step 6: Commit**

```bash
git add src/mayor/mod.rs
git commit -m "feat(mayor): LearnedMayor swaps a learned policy into the R34 execute_plan seam"
```

---

## Task 10 (RESEARCH-GATE, measure-before-build): env throughput + PPO go/no-go

**Files:**
- Modify: `src/bin/vice_mayor.rs` (`measure` subcommand)

**Acceptance:** `cargo run -p urban_horizon --release --bin vice_mayor -- measure --threads 8 --seconds 10` prints `env_steps_per_sec=<N> sim_ticks_per_sec=<N>` measured over 10s across 8 threads. This task DELIVERS THE MEASUREMENT and a written go/no-go note appended to the design doc's section 4.4 — it does NOT implement PPO. PPO is only planned in a SEPARATE follow-up plan IF `env_steps_per_sec` clears the threshold derived here (rule of thumb from the literature: small-MLP PPO wants ≥ ~1k env-steps/sec/thread to train in hours, not days).

**Wiring requirement:** `measure` spins K `CityEnv`s on K rayon threads, runs random-legal-action rollouts, and reports aggregate steps/sec. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vice_measure_reports_throughput() {
    let tp = measure_throughput(/*threads*/ 4, /*ms*/ 1500);
    println!("env_steps_per_sec={:.0}", tp.env_steps_per_sec);
    assert!(tp.env_steps_per_sec > 0.0, "must measure real throughput");
    assert!(tp.total_steps > 0);
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon vice_measure_reports_throughput 2>&1 | tail -5
```

Expected: FAIL — no `measure_throughput`.

- [ ] **Step 3: Implement** `measure_throughput` (rayon over K envs, random legal action via the mask, count `step`s in a wall-clock window) and the `measure` subcommand.

- [ ] **Step 4: Wire** — `measure` branch in the bin's arg dispatch.

- [ ] **Step 5: Run — record the number**

```bash
cargo run -p urban_horizon --release --bin vice_mayor -- measure --threads 8 --seconds 10
```

Expected: a real `env_steps_per_sec` figure. Append a one-line go/no-go verdict (and the number) to the design doc §4.4.

- [ ] **Step 6: Commit**

```bash
git add src/bin/vice_mayor.rs docs/superpowers/specs/2026-06-15-vice-mayor-rl-gym-design.md
git commit -m "feat(vice): env throughput measurement + PPO go/no-go gate (research)"
```

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks (Task 9 wires the seam type; UI selection is a noted follow-up, not a wire-later of THIS task's deliverable).
- [x] Every `Acceptance` criterion names a real non-trivial expected output (coverage 0.10, top1 > 0.7, equal hashes, differing style means) — never "tests pass".
- [x] Every `Wiring requirement` names an exact function/file and the `execute_plan` seam.
- [x] `IMPORTANT NOTES` contains the real shipped API signatures (driver, GameState getters, mayor contract) from the game repo.
- [x] `File Map` lists every file in every task.
- [x] No implementation step contains `todo!()`/`unimplemented!()`/stub bodies.
- [x] `Done When` names specific commands and specific human-observable output (the `EPISODE ...` line + the compare table).
- [x] Types/methods consistent across tasks (`observe`, `ActionTable`, `legal_mask`, `shaped_reward`, `CityEnv::step`, `VicePolicy::act`, `LearnedMayor`).
- [x] PPO (the only research-grade path) is GATED behind a measured throughput task and a separate follow-up plan — not assumed buildable.
