# Vice Mayor — Learned Brain Phased Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the rule-based R34 Assistant Mayor into a small, local-trainable "Vice Mayor" that plays Urban Horizon in different styles, by building a deterministic RL gym over the existing sim, an MCTS planner (no training), a distilled tiny BC policy, style conditioning, and a consultable persona — all enacting through the unchanged R24 command driver.
**Done When:** `cargo run -p urban_horizon --bin vice_mayor_demo -- --style frugal --seed 7` prints a 12-month transcript with ≥3 distinct **accepted** `GameCommand` ids and a final line `style_frugal_wellbeing=<x> vs off_baseline=<y>` where `x > y` on the same seed — a human reads two real numbers and sees the Vice Mayor outperformed the do-nothing baseline.
**Architecture:** A game-layer `src/ai/` module (gym + obs + actions + reward + brains + persona) wraps `GameState`; every action is enacted through `driver::execute_plan` (validated, replay-logged, undoable); the brain plugs in behind the unchanged `AssistantMayor::recommend`/`act` seam. Engine crates stay game-agnostic.
**Design Document:** `docs/superpowers/specs/2026-06-15-vice-mayor-learned-brain-design.md`
**Tech Stack:** Rust (game repo `urban_horizon`); PyTorch (CPU) for offline BC training only; llama.cpp/Vulkan (optional persona voice). No new heavy Rust deps for the MLP (hand-rolled f32 matmul).
**Build:** `cargo build` / `cargo test` in `~/Ochroma/projects/urban_horizon_r39`. **Work in the game repo, not `~/src/ochroma`.** Phases 1-3 require **no Python and no GPU**. Phase 4 (training) needs Python+PyTorch on the CPU. Phase 6 (LLM voice) needs the already-built `llama.cpp` + `gemma4-e4b.gguf`.

---

## IMPORTANT NOTES

- **Repo:** all `src/...` paths are in `~/Ochroma/projects/urban_horizon_r39`. Confirm `git -C ~/Ochroma/projects/urban_horizon_r39 log --oneline -1` is `52eb97f` before starting; if a newer canonical copy exists, work there.
- **The driver is the only mutation path.** Never mutate `GameState` from the brain/gym directly for an *action* — always go through `driver::execute_plan(view, game, sdf, registry, &plan)`. Direct `game.adjust_tax(...)` calls bypass validation + replay-log = **task failure**.
- Exact driver signatures (from `src/command/driver.rs`, do not alter):
  - `pub fn execute_plan(view: &mut ViewState, game: &mut GameState, sdf: &SpatialFieldRegistry, registry: &GameRegistry, plan: &Plan) -> PlanReport`
  - `pub fn validate_command(cmd: &PlannedCommand, registry: &GameRegistry) -> Result<(), String>`
  - `pub fn undo_plan(game: &mut GameState, report: &PlanReport)`
  - `PlannedCommand::verb(id: impl Into<String>) -> Self`; `Plan { goal: String, commands: Vec<PlannedCommand> }`
  - `PlanReport::accepted_count()/rejected_count()/rejection_reason(id)`; `report.replay_log: Vec<PlannedCommand>`.
- Exact mayor seam (from `src/mayor/mod.rs`): `Autonomy::{Off,Advise,Auto}`; `Recommendation { ladder_id, target_tier, tier_name, command_id, reason, est_monthly_cost, score }`; `AssistantMayor::recommend(&self, game: &GameState) -> Vec<Recommendation>`; `act(&self, view, game, sdf, registry) -> Option<PlanReport>`. **Keep `Autonomy::Off` a true no-op and keep `new(autonomy)` defaulting to the rule brain** (back-compat: the existing `mayor/mod.rs` tests must still pass unchanged).
- Determinism rules (match the existing codebase): no `HashMap` iteration in any decision path, no `rand` without a pinned seed, rank/tie-break with `f64::total_cmp` then id-ascending, fixed-ε float compares (`const ..._EPS: f64 = 1e-6`). The observation vector must be byte-identical for equal `integration_hash()`.
- Reward/obs read **only public accessors** that already exist: `game.coverage.coverage_ratio(DependentKind::_)`, `game.last_stats.gated_workers`, `game.cim.population()`, `game.sim.budget.funds`, `game.treasury.cash`, `game.wellbeing_history()`, `game.tax_policy.sector_rate(Sector::_)`, `game.policy_state.get(id)`, `game.ladder_catalog`. Do **not** add `pub` fields to cross-module types.
- Replay safety: a Vice Mayor session stays a valid `CivReplay` because `execute_plan` logs the `PlannedCommand`; **replay re-dispatches the logged command, never the brain.** Do not call the model during `replay_commands`.
- `todo!()` / `unimplemented!()` / empty function bodies = **task failure**.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/ai/mod.rs` | module root; `pub mod gym; obs; actions; reward; brain; mcts; policy_mlp; persona;` and wire into `src/lib.rs` |
| Create | `src/ai/obs.rs` | `OBS_DIM`, `OBS_VERSION`, `encode(&GameState) -> [f32; OBS_DIM]` |
| Create | `src/ai/actions.rs` | frozen `ActionDef` table, `actions()`, `to_planned(idx, view)` |
| Create | `src/ai/reward.rs` | `RewardWeights`, `MayorStyle`, `reward(prev,next,invalid,style)` |
| Create | `src/ai/gym.rs` | `UrbanGym::reset/step/integration_hash` (enacts via driver) |
| Create | `src/ai/mcts.rs` | `MctsConfig`, `plan(&UrbanGym-state) -> action idx` (no learning) |
| Create | `src/ai/policy_mlp.rs` | `PolicyMlp::load/act/value` (flat f32 weights, fixed reduction order) |
| Create | `src/ai/brain.rs` | `MayorBrain::{Rule,Mcts,Bc}`, `recommend(game, style) -> Vec<Recommendation>` |
| Create | `src/ai/persona.rs` | `rationale(cmd, game) -> String` (deterministic template; optional LLM arm) |
| Modify | `src/lib.rs` | `pub mod ai;` |
| Modify | `src/mayor/mod.rs` | add `style` + `brain` to `AssistantMayor`; `with_brain(..)`; delegate `recommend` to brain (keep `new` = rule/Balanced) |
| Create | `src/bin/vice_mayor_demo/main.rs` | the human-verifiable demo binary |
| Create | `tools/train_bc.py` | offline CPU trainer: reads rollout dataset, trains MLP, exports flat f32 weights |
| Test | `src/ai/gym.rs` (`#[cfg(test)]`) | determinism + reward sign + action validity gates |
| Test | `src/ai/mcts.rs` (`#[cfg(test)]`) | MCTS beats rule-based over seeds |
| Test | `src/ai/policy_mlp.rs` (`#[cfg(test)]`) | load+forward exact value; BC top-1 accuracy |
| Test | `src/mayor/mod.rs` (`#[cfg(test)]`) | brain swap preserves Off no-op + replay-bit-identical |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Deterministic gym | same `reset(seed)` + same actions → `assert_eq!(h1, h2)` on real `integration_hash()` u64s | `assert!(gym.step(0).obs.len() > 0)` |
| Observation discriminates states | ≥6 of `OBS_DIM` entries differ between a crisis city and a healthy city; all finite | returns `[0.0; OBS_DIM]` |
| Reward sign is meaningful | `reward(healthy_prev, healthier_next, false, Balanced) > 0` AND `reward(solvent, bankrupt, false, Balanced) < 0` | `assert!(r == r)` |
| Actions are all real registry ids | every `actions()` id resolves via `GameRegistry::get`; `len() >= 20` | `vec!["noop"]` |
| MCTS > rule-based | mean end-wellbeing(MCTS) − mean(rule) `> 0` over 20 seeds, margin printed | `assert!(score >= 0.0)` |
| BC MLP reproduces labels | top-1 match `> 0.60` on held-out states, printed accuracy | `assert!(out.len() == n)` |
| Styles differ | `frugal` 12-month treasury spend `<` `growth` spend, two real totals printed | identical logs |
| Brain swap is replay-safe | Auto with `Mcts`/`Bc` brain → `replay_commands` reproduces a bit-identical save | `assert!(report.is_some())` |
| Persona names the reason | rationale string contains the chosen ladder/command's modeled reason substring | returns `"ok"` |

---

## Task 1: Frozen observation encoder (`obs::encode`)

**Files:**
- Create: `src/ai/obs.rs`
- Modify: `src/ai/mod.rs`, `src/lib.rs`

**Acceptance:** `cargo test -p urban_horizon ai::obs::encode_discriminates -- --nocapture` → prints `differing_entries=<n>` with `n >= 6` and `all_finite=true`.

**Wiring requirement:** `encode` must be the function `UrbanGym::reset`/`step` (Task 4) call to produce observations; no other obs producer exists. For this task wire `pub mod ai;` into `src/lib.rs` and `pub mod obs;` into `src/ai/mod.rs`. `todo!()`/empty body = task failure.

- [ ] **Step 1: Write the failing test** — real crisis vs healthy cities, compare encodings

```rust
#[test]
fn encode_discriminates() {
    use crate::game::GameState;
    use crate::game_mechanics::care::service::DependentKind;
    let mut crisis = GameState::new_small();
    crisis.coverage.demand.insert(DependentKind::Infant, 100);
    crisis.coverage.served.insert(DependentKind::Infant, 5);
    crisis.sim.budget.funds = 100.0;
    let mut healthy = GameState::new_small();
    healthy.coverage.demand.insert(DependentKind::Infant, 100);
    healthy.coverage.served.insert(DependentKind::Infant, 98);
    healthy.sim.budget.funds = 500_000.0;
    let a = super::encode(&crisis);
    let b = super::encode(&healthy);
    let differing = a.iter().zip(b.iter()).filter(|(x, y)| (*x - *y).abs() > 1e-6).count();
    let all_finite = a.iter().chain(b.iter()).all(|v| v.is_finite());
    println!("differing_entries={differing} all_finite={all_finite}");
    assert!(all_finite, "all features must be finite");
    assert!(differing >= 6, "crisis and healthy must differ in >=6 features, got {differing}");
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd ~/Ochroma/projects/urban_horizon_r39 && cargo test -p urban_horizon ai::obs::encode_discriminates 2>&1 | tail -5
```

Expected: FAIL — `error[E0433]`/`cannot find` `ai`/`encode` (module does not exist yet).

- [ ] **Step 3: Implement** — fixed field order, normalized, finite

```rust
pub const OBS_DIM: usize = 32;
pub const OBS_VERSION: u32 = 1;
pub fn encode(game: &GameState) -> [f32; OBS_DIM] {
    let mut v = [0.0f32; OBS_DIM];
    let pop = game.cim.population().max(1) as f32;
    // [0..] care coverage, [.. ] labor, fiscal, wellbeing trend, taxes, policy tiers.
    // Read ONLY public accessors (see IMPORTANT NOTES). Normalize each to ~[-1,1].
    // Field order is the FROZEN contract — document each index inline.
    // ... fill every index; no zeros-only stub ...
    v
}
```

- [ ] **Step 4: Wire at exact callsite** — register the module

In `src/lib.rs` add `pub mod ai;`. In `src/ai/mod.rs` add `pub mod obs;`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::obs::encode_discriminates -- --nocapture
```

Expected: PASS, output `differing_entries=N all_finite=true` with `N >= 6`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/mod.rs src/ai/obs.rs src/lib.rs
git commit -m "feat(ai): frozen observation encoder for the Vice Mayor gym"
```

---

## Task 2: Frozen discrete action set (`actions::actions`/`to_planned`)

**Files:**
- Create: `src/ai/actions.rs`
- Modify: `src/ai/mod.rs`

**Acceptance:** `cargo test -p urban_horizon ai::actions::all_ids_resolve -- --nocapture` → prints `n_actions=<k>` with `k >= 20` and `unresolved=0`.

**Wiring requirement:** `to_planned` must be the function `UrbanGym::step` (Task 4) uses to turn an action index into a `PlannedCommand`. Add `pub mod actions;` to `src/ai/mod.rs`.

- [ ] **Step 1: Write the failing test** — every action id is a real registry command

```rust
#[test]
fn all_ids_resolve() {
    use crate::command::GameRegistry;
    let reg = GameRegistry::standard();
    let acts = super::actions();
    let unresolved = acts.iter().filter(|a| reg.get(a.id()).is_none()).count();
    println!("n_actions={} unresolved={}", acts.len(), unresolved);
    assert_eq!(unresolved, 0, "every action must be a real registry id");
    assert!(acts.len() >= 20, "need >=20 Tier-A actions, got {}", acts.len());
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::actions::all_ids_resolve 2>&1 | tail -5
```

Expected: FAIL — `cannot find function actions` / module missing.

- [ ] **Step 3: Implement** — Tier-A ids (tax inc/dec per sector, all `view.policy.*`, all `view.ladder.*.tierN`, a noop `sim.tick`), each `ActionArgs::None`; `to_planned` maps index → `PlannedCommand::verb(id)`.

- [ ] **Step 4: Wire at exact callsite** — `pub mod actions;` in `src/ai/mod.rs`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::actions::all_ids_resolve -- --nocapture
```

Expected: PASS, `n_actions=K unresolved=0`, `K >= 20`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/actions.rs src/ai/mod.rs
git commit -m "feat(ai): frozen Tier-A discrete action set bound to the command registry"
```

---

## Task 3: Reward function + styles (`reward::reward`)

**Files:**
- Create: `src/ai/reward.rs`
- Modify: `src/ai/mod.rs`

**Acceptance:** `cargo test -p urban_horizon ai::reward::reward_sign -- --nocapture` → prints `r_better=<pos> r_bankrupt=<neg>` with the first `> 0` and the second `< 0`.

**Wiring requirement:** `reward` must be called by `UrbanGym::step` (Task 4) once per transition. Add `pub mod reward;` to `src/ai/mod.rs`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn reward_sign() {
    use crate::game::GameState;
    let prev = GameState::new_small();
    // healthier next: push a wellbeing sample up and keep cash positive
    let mut better = prev.clone();
    better.wellbeing_history.push(0.9);
    let mut bankrupt = prev.clone();
    bankrupt.treasury.cash = -50_000.0;
    let r_better = super::reward(&prev, &better, false, super::MayorStyle::Balanced);
    let r_bankrupt = super::reward(&prev, &bankrupt, false, super::MayorStyle::Balanced);
    println!("r_better={r_better} r_bankrupt={r_bankrupt}");
    assert!(r_better > 0.0, "improving wellbeing must reward, got {r_better}");
    assert!(r_bankrupt < 0.0, "going bankrupt must penalize, got {r_bankrupt}");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::reward::reward_sign 2>&1 | tail -5
```

Expected: FAIL — module/function missing.

- [ ] **Step 3: Implement** — the shaped delta from the design (wellbeing Δ, gated-worker Δ, pop Δ, overdraft penalty, invalid penalty); `MayorStyle::weights()` returns the per-style `RewardWeights`. Fixed-ε compares, no RNG.

- [ ] **Step 4: Wire at exact callsite** — `pub mod reward;` in `src/ai/mod.rs`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::reward::reward_sign -- --nocapture
```

Expected: PASS, `r_better=<+> r_bankrupt=<−>`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/reward.rs src/ai/mod.rs
git commit -m "feat(ai): styled reward function (wellbeing/pillars/pop/solvency)"
```

---

## Task 4: The gym (`UrbanGym::reset/step`) — enacts through the R24 driver

**Files:**
- Create: `src/ai/gym.rs`
- Modify: `src/ai/mod.rs`

**Acceptance:** `cargo test -p urban_horizon ai::gym::deterministic_replay -- --nocapture` → prints `h1=<hex> h2=<hex> equal=true` (two runs of the same `reset(seed)`+action sequence give equal `integration_hash`).

**Wiring requirement:** `step` MUST mutate the game only via `driver::execute_plan` (never direct game methods); it calls `actions::to_planned`, `obs::encode`, and `reward::reward`. Add `pub mod gym;` to `src/ai/mod.rs`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn deterministic_replay() {
    use crate::scenario::Scenario;
    use crate::ai::reward::MayorStyle;
    let seq = [3usize, 0, 7, 2, 0];
    let run = |seed: u64| {
        let (mut g, _o) = super::UrbanGym::reset(seed, Scenario::Demo, MayorStyle::Balanced);
        for &a in &seq { let _ = g.step(a); }
        g.integration_hash()
    };
    let h1 = run(7);
    let h2 = run(7);
    println!("h1={h1:#018x} h2={h2:#018x} equal={}", h1 == h2);
    assert_eq!(h1, h2, "same seed + actions must reproduce bit-identically");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::gym::deterministic_replay 2>&1 | tail -5
```

Expected: FAIL — `UrbanGym` missing.

- [ ] **Step 3: Implement** — `reset` builds `GameState` from the scenario, holds `ViewState`/`GameRegistry::standard()`/`SpatialFieldRegistry::new()`, returns `encode`. `step(a)`: `to_planned(a, &view)` → `Plan` → `execute_plan` → tick forward `TICKS_PER_DAY*DAYS_PER_MONTH` → `reward(prev, &game, !accepted, style)` → `StepOut`. `accepted = report.accepted_count() > 0`.

- [ ] **Step 4: Wire at exact callsite** — `pub mod gym;` in `src/ai/mod.rs`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::gym::deterministic_replay -- --nocapture
```

Expected: PASS, `equal=true`, two identical non-zero hex hashes.

- [ ] **Step 6: Commit**

```bash
git add src/ai/gym.rs src/ai/mod.rs
git commit -m "feat(ai): deterministic UrbanGym reset/step over the R24 driver"
```

---

## Task 5: MCTS planner (no learning) (`mcts::plan`)

**Files:**
- Create: `src/ai/mcts.rs`
- Modify: `src/ai/mod.rs`

**Acceptance:** `cargo test -p urban_horizon ai::mcts::beats_rule_based -- --nocapture` → prints `mcts_mean=<a> rule_mean=<b> margin=<a-b>` with `a > b` over 20 seeds.

**Wiring requirement:** `plan` forks the gym/state via `GameState::from_save` (or `whatif::branch`) for lookahead and uses the existing rule-based mayor as the rollout default policy; selection tie-break is `f64::total_cmp` then action-index-ascending (deterministic). Add `pub mod mcts;` to `src/ai/mod.rs`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn beats_rule_based() {
    use crate::scenario::Scenario;
    use crate::ai::reward::MayorStyle;
    let horizon = 6; // months
    let mut mcts_sum = 0.0f64; let mut rule_sum = 0.0f64;
    for seed in 0..20u64 {
        // MCTS-driven gym
        let (mut g, mut obs) = super::super::gym::UrbanGym::reset(seed, Scenario::Demo, MayorStyle::Balanced);
        for _ in 0..horizon { let a = super::plan(&g, MayorStyle::Balanced, &Default::default()); obs = g.step(a).obs; let _ = obs; }
        mcts_sum += g.wellbeing() as f64; // gym exposes last wellbeing sample
        // rule-based-driven gym (action 0 stands in for "rule mayor pick" via brain::Rule in Task 7;
        // here use the rule mayor's top recommendation mapped to an action index)
        let (mut gr, _o) = super::super::gym::UrbanGym::reset(seed, Scenario::Demo, MayorStyle::Balanced);
        for _ in 0..horizon { let a = gr.rule_action(); let _ = gr.step(a); }
        rule_sum += gr.wellbeing() as f64;
    }
    let (a, b) = (mcts_sum / 20.0, rule_sum / 20.0);
    println!("mcts_mean={a} rule_mean={b} margin={}", a - b);
    assert!(a > b, "MCTS must beat rule-based: {a} vs {b}");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::mcts::beats_rule_based 2>&1 | tail -5
```

Expected: FAIL — `plan`/`rule_action`/`wellbeing` missing.

- [ ] **Step 3: Implement** — `MctsConfig { sims, depth, branching, c_uct }` (`Default` = small, e.g. 32 sims, depth 4). `plan`: for each candidate action, fork state, apply via driver, roll out `depth` months under the rule-based default policy, accumulate discounted `reward`; pick argmax with the deterministic tie-break. Add tiny gym helpers `wellbeing()` and `rule_action()` (the latter maps the rule mayor's top `Recommendation.command_id` to its action index, else noop).

- [ ] **Step 4: Wire at exact callsite** — `pub mod mcts;` in `src/ai/mod.rs`; the gym helpers in `gym.rs`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::mcts::beats_rule_based -- --nocapture
```

Expected: PASS, `margin > 0`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/mcts.rs src/ai/gym.rs src/ai/mod.rs
git commit -m "feat(ai): MCTS test-time planner (no training) beats rule-based mayor"
```

---

## Task 6: Tiny BC policy load + inference (`policy_mlp`) and the CPU trainer

**Files:**
- Create: `src/ai/policy_mlp.rs`, `tools/train_bc.py`
- Modify: `src/ai/mod.rs`

**Acceptance:** `cargo test -p urban_horizon ai::policy_mlp::forward_is_exact_and_accurate -- --nocapture` → prints `golden_value=<v> top1_acc=<p>` where `v` matches a hand-computed value within 1e-5 and `p > 0.60` on a held-out label set generated by MCTS/rule-based.

**Wiring requirement:** `PolicyMlp::act` is the function the `MayorBrain::Bc` arm (Task 7) calls. The trainer must emit the exact flat little-endian f32 layout `policy_mlp::load` reads. Add `pub mod policy_mlp;` to `src/ai/mod.rs`. The trainer runs on the **780M box CPU** (no GPU): `python3 tools/train_bc.py --data rollouts.npz --out policy.bin`.

- [ ] **Step 1: Write the failing test** — two parts: a tiny hand-set weight file gives an exact forward value; a trained file clears 60% top-1.

```rust
#[test]
fn forward_is_exact_and_accurate() {
    // (a) golden: write a 2->2->2 net with known weights, assert forward == hand value.
    let path = std::env::temp_dir().join("vm_golden.bin");
    super::write_golden_for_test(&path); // helper writes known w/b
    let net = super::PolicyMlp::load(&path, 2, 2, 2).unwrap();
    let v = net.value(&pad32([1.0, -1.0])); // helper pads to OBS_DIM for the call shape
    println!("golden_value={v}");
    assert!((v - EXPECTED_GOLDEN).abs() < 1e-5, "exact forward mismatch: {v}");
    // (b) accuracy: load the trained policy.bin + held-out (obs,label) set, measure top-1.
    let (net2, eval) = super::load_trained_and_eval_set();
    let hits = eval.iter().filter(|(o, lbl)| net2.act(o) == *lbl).count();
    let p = hits as f64 / eval.len() as f64;
    println!("top1_acc={p}");
    assert!(p > 0.60, "BC top-1 must exceed 0.60, got {p}");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::policy_mlp::forward_is_exact_and_accurate 2>&1 | tail -5
```

Expected: FAIL — `PolicyMlp` missing (and `policy.bin` not yet generated).

- [ ] **Step 3: Implement** — Rust: `load` reads `[w1,b1,w2,b2,w_pi,b_pi,w_v,b_v]` as flat f32; `act`/`value` do sequential (fixed-order) matmuls + ReLU + argmax (bit-stable). Python `train_bc.py`: build dataset by running `UrbanGym` rollouts labeled with the MCTS/rule action (export a small `rollouts.npz` from a Rust `dump_rollouts` helper or a `--dump` mode of the demo bin), train the MLP with cross-entropy on CPU, export `policy.bin` in the exact layout. Commit a small checked-in `policy.bin` (or generate it in the test's setup) so the test is reproducible.

- [ ] **Step 4: Wire at exact callsite** — `pub mod policy_mlp;` in `src/ai/mod.rs`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
python3 tools/train_bc.py --data rollouts.npz --out policy.bin && \
cargo test -p urban_horizon ai::policy_mlp::forward_is_exact_and_accurate -- --nocapture
```

Expected: PASS, `golden_value` matches and `top1_acc > 0.60`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/policy_mlp.rs src/ai/mod.rs tools/train_bc.py policy.bin
git commit -m "feat(ai): tiny BC policy MLP (CPU-trained) with exact bit-stable inference"
```

---

## Task 7: Pluggable brain + R34 seam swap (`MayorBrain`, `AssistantMayor::with_brain`)

**Files:**
- Create: `src/ai/brain.rs`
- Modify: `src/ai/mod.rs`, `src/mayor/mod.rs`

**Acceptance:** `cargo test -p urban_horizon mayor::brain_swap_replay_identical -- --nocapture` → prints `accepted=1 replay_equal=true`; AND the **existing** `mayor::assistant_mayor` tests still pass unchanged (`cargo test -p urban_horizon mayor:: 2>&1 | tail -3` shows `0 failed`).

**Wiring requirement:** `AssistantMayor::recommend` delegates to `MayorBrain::recommend`; `act` still routes the top pick through `driver::execute_plan` (one undo group, replay-logged). `new(autonomy)` MUST still build `Balanced` + `MayorBrain::Rule` (back-compat). Add `pub mod brain;` to `src/ai/mod.rs`.

- [ ] **Step 1: Write the failing test** — Auto with a learned brain stays replay-bit-identical

```rust
#[test]
fn brain_swap_replay_identical() {
    use crate::ai::brain::MayorBrain;
    use crate::ai::reward::MayorStyle;
    use crate::command::{GameRegistry, ViewState};
    use crate::spatial_field::SpatialFieldRegistry;
    use crate::command::driver;
    // city with a modeled problem (mirror mayor/mod.rs's city_with_low_childcare)
    let mut game = crate::game::GameState::new_small();
    game.coverage.demand.insert(crate::game_mechanics::care::service::DependentKind::Infant, 100);
    game.coverage.served.insert(crate::game_mechanics::care::service::DependentKind::Infant, 10);
    game.sim.budget.funds = 100_000.0;
    let (mut view, sdf, reg) = (ViewState::default(), SpatialFieldRegistry::new(), GameRegistry::standard());
    let mayor = super::super::mayor::AssistantMayor::with_brain(
        super::super::mayor::Autonomy::Auto, MayorStyle::CareFirst,
        MayorBrain::Bc(crate::ai::policy_mlp::PolicyMlp::load_default().unwrap()));
    let report = mayor.act(&mut view, &mut game, &sdf, &reg).expect("Auto enacts");
    let save1 = game.to_save().to_json();
    // replay the LOGGED commands on a fresh equivalent city (brain never re-run)
    let mut game2 = crate::game::GameState::new_small();
    game2.coverage.demand.insert(crate::game_mechanics::care::service::DependentKind::Infant, 100);
    game2.coverage.served.insert(crate::game_mechanics::care::service::DependentKind::Infant, 10);
    game2.sim.budget.funds = 100_000.0;
    let mut v2 = ViewState::default();
    driver::replay_commands(&mut v2, &mut game2, &sdf, &reg, &report.replay_log);
    let equal = game.to_save().to_json() == game2.to_save().to_json();
    println!("accepted={} replay_equal={}", report.accepted_count(), equal);
    assert_eq!(report.accepted_count(), 1);
    assert!(equal, "logged-command replay must be bit-identical (brain not re-run)");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon mayor::brain_swap_replay_identical 2>&1 | tail -5
```

Expected: FAIL — `with_brain`/`MayorBrain`/`PolicyMlp::load_default` missing.

- [ ] **Step 3: Implement** — `MayorBrain::{Rule, Mcts(MctsConfig), Bc(PolicyMlp)}` with `recommend(game, style) -> Vec<Recommendation>`: `Rule` = today's `problem_signals` path (moved/shared); `Mcts` = `mcts::plan` then map chosen action → a one-element `Recommendation` (`command_id` = action's registry id); `Bc` = `PolicyMlp::act` then same mapping. Add `style` + `brain` fields to `AssistantMayor`; `new` keeps `Balanced`+`Rule`; `with_brain` sets both; `recommend` delegates; `act` unchanged except it builds the plan from the brain's top recommendation.

- [ ] **Step 4: Wire at exact callsite** — `pub mod brain;` in `src/ai/mod.rs`; `AssistantMayor::recommend` body calls `self.brain.recommend(game, self.style)`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon mayor:: -- --nocapture 2>&1 | tail -8
```

Expected: PASS for `brain_swap_replay_identical` (`replay_equal=true`) AND all pre-existing `assistant_mayor` tests (`off_enacts_nothing`, `auto_replay_bit_identical`, …) still pass.

- [ ] **Step 6: Commit**

```bash
git add src/ai/brain.rs src/ai/mod.rs src/mayor/mod.rs
git commit -m "feat(ai): pluggable MayorBrain behind the R34 seam (Rule/MCTS/BC), replay-safe"
```

---

## Task 8: Styles differ + the human-verifiable demo binary (`vice_mayor_demo`)

**Files:**
- Create: `src/bin/vice_mayor_demo/main.rs`
- Modify: `src/ai/mod.rs` (export `persona`), Create: `src/ai/persona.rs`

**Acceptance:** `cargo run -p urban_horizon --bin vice_mayor_demo -- --style frugal --seed 7` prints a 12-month transcript with ≥3 distinct accepted command ids, a per-turn first-person rationale line, and a final line `style_frugal_wellbeing=<x> vs off_baseline=<y>` with `x > y`; AND `cargo test -p urban_horizon ai::reward::styles_differ -- --nocapture` prints `frugal_spend=<f> growth_spend=<g>` with `f < g`.

**Wiring requirement:** the demo runs two gyms on the same seed — one driven by `MayorBrain::Mcts`/`Bc` at the chosen style, one `Autonomy::Off` (no-op) — and compares final wellbeing; the per-turn line comes from `persona::rationale(chosen_cmd, &game)`. `persona::rationale` uses the chosen command's modeled reason + `whatif::diff` on a forked branch for the predicted effect (deterministic; LLM arm optional, off by default). Add `pub mod persona;`.

- [ ] **Step 1: Write the failing test** — styles produce different spend

```rust
#[test]
fn styles_differ() {
    use crate::scenario::Scenario;
    use crate::ai::reward::MayorStyle;
    let spend = |style| {
        let (mut g, _o) = super::super::gym::UrbanGym::reset(7, Scenario::Demo, style);
        let start = g.treasury_cash();
        for _ in 0..12 { let a = super::super::mcts::plan(&g, style, &Default::default()); let _ = g.step(a); }
        start - g.treasury_cash() // total spent
    };
    let f = spend(MayorStyle::Frugal);
    let gr = spend(MayorStyle::Growth);
    println!("frugal_spend={f} growth_spend={gr}");
    assert!(f < gr, "frugal must spend less than growth: {f} vs {gr}");
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::reward::styles_differ 2>&1 | tail -5
```

Expected: FAIL — `treasury_cash` helper / `persona` / demo bin missing.

- [ ] **Step 3: Implement** — gym helper `treasury_cash()`; `persona::rationale(cmd, game) -> String` (template naming the modeled reason + predicted `WhatIfDiff` deltas); demo `main.rs` parses `--style`/`--seed`, runs the styled brain gym + the Off baseline gym for 12 months, prints each turn `month N: <persona line> -> accepted <cmd id>` and the final comparison line.

- [ ] **Step 4: Wire at exact callsite** — `pub mod persona;` in `src/ai/mod.rs`; demo bin entry in `src/bin/vice_mayor_demo/main.rs`.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::reward::styles_differ -- --nocapture && \
cargo run -p urban_horizon --bin vice_mayor_demo -- --style frugal --seed 7
```

Expected: test PASS (`frugal_spend < growth_spend`); demo prints the 12-month transcript and `style_frugal_wellbeing=<x> vs off_baseline=<y>` with `x > y`.

- [ ] **Step 6: Commit**

```bash
git add src/ai/persona.rs src/ai/mod.rs src/bin/vice_mayor_demo/main.rs
git commit -m "feat(ai): Vice Mayor persona rationale + styled demo binary (the Done When)"
```

---

## Task 9 (OPTIONAL, RESEARCH-GRADE): local LLM persona voice

**Files:**
- Modify: `src/ai/persona.rs`

**Acceptance:** `OCHROMA_VICE_MAYOR_VOICE=llm cargo run -p urban_horizon --bin vice_mayor_demo -- --style carefirst --seed 7` prints rationale lines whose text differs from the deterministic template (LLM phrasing) while the **chosen commands are byte-identical** to the non-LLM run on the same seed — proving the LLM only phrases, never decides.

**Wiring requirement:** the LLM arm routes the deterministic rationale's facts through `factory::llm::LlmRunner` (pinned `gemma4-e4b.gguf`) or `vox_nn::llm_client` (llama-server `/v1`); it MUST NOT choose actions. Default stays deterministic-template. Off the decision path (called only when rendering the rationale). `todo!()` = task failure.

- [ ] **Step 1: Write the failing test** — same decisions, different prose

```rust
#[test]
fn llm_voice_phrases_only() {
    // run the styled brain once; capture the chosen command ids.
    // render rationale twice: template vs llm (use the Replay LlmRunner backend
    // with a canned transcript so the test is deterministic and offline).
    // assert: command ids identical across both; rationale strings differ.
    // (mirrors driver.rs's llm_planner_wired_via_replay offline pattern)
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p urban_horizon ai::persona::llm_voice_phrases_only 2>&1 | tail -5
```

Expected: FAIL — LLM arm not implemented.

- [ ] **Step 3: Implement** — add the LLM arm gated behind `OCHROMA_VICE_MAYOR_VOICE=llm`; use the `Replay` runner in the test (canned transcript), `from_env` in production. Strip the model output to a single rationale line; on any error fall back to the template (never block the turn).

- [ ] **Step 4: Wire at exact callsite** — branch inside `persona::rationale` on the env/config flag.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon ai::persona::llm_voice_phrases_only -- --nocapture
```

Expected: PASS — identical command ids, differing prose.

- [ ] **Step 6: Commit**

```bash
git add src/ai/persona.rs
git commit -m "feat(ai): optional local-LLM Vice Mayor voice (phrasing only, replay-safe)"
```

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks exist (each task ends by registering its module and being called from the gym/seam/demo).
- [x] Every `Acceptance` criterion names a real non-trivial expected output (hashes, counts, accuracies, signed rewards, two-number comparisons) — never "tests pass" alone.
- [x] Every `Wiring requirement` names an exact function and file (`UrbanGym::step` in `src/ai/gym.rs`, `AssistantMayor::recommend` in `src/mayor/mod.rs`, etc.).
- [x] `IMPORTANT NOTES` contains the real driver + mayor signatures copied from `src/command/driver.rs` and `src/mayor/mod.rs`.
- [x] `File Map` lists every file that appears in any task.
- [x] No implementation step contains `todo!()`/`unimplemented!()`/stub bodies (each names the full logic to write).
- [x] `Done When` names a specific command (`cargo run … vice_mayor_demo --style frugal`) and a specific human-observable result (`style_frugal_wellbeing=x vs off_baseline=y`, `x>y`).
- [x] Types/methods consistent across tasks (`UrbanGym::step`, `MayorBrain::recommend`, `PolicyMlp::act`, `MayorStyle`, `RewardWeights` used identically everywhere).

> **Phasing / feasibility honesty.** Phases 1-5 (Tasks 1-8 minus training) are **buildable now** on the 780M box with **no GPU and no Python** except Task 6's CPU trainer. Task 5 (MCTS) is the recommended first shippable win — strong, no learning. Task 6 (BC) is a tiny CPU train (≈10k-param MLP, minutes). Task 9 (LLM voice) reuses the already-built local gemma path. The Decision Transformer (design §4.5c-ii) is **research-grade and intentionally not a task** — ship per-style BC instead.
