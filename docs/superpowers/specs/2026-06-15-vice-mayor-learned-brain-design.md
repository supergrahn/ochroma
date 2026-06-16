# Design: Vice Mayor — a small, local-trainable learned brain for Urban Horizon (2026-06-15)

**Status:** Draft
**Scope:** A small AI that plays Urban Horizon in different styles and acts as an in-game "Vice Mayor" persona the player can consult or delegate to. It plugs into the existing R34 Assistant Mayor seam (`Off`/`Advise`/`Auto`) and emits validated `GameCommand`s through the R24 driver. Research + design + phased plan only — **no training, no code in this workflow.**
**Related:** R34 Assistant Mayor (`urban_horizon/src/mayor/mod.rs`), R24 command driver (`urban_horizon/src/command/driver.rs`), R22 replay (`urban_horizon/src/game/replay.rs`), R28 what-if (`urban_horizon/src/game/whatif.rs`).

> **Repo note.** The engine crates live in `~/src/ochroma` (this repo). The *game* (Urban Horizon) lives in `~/Ochroma/projects/urban_horizon{,_r38,_r39}` — `r39` is the canonical working copy at design time (HEAD `52eb97f`, same as `r38`). All `src/...` paths below are relative to the game repo unless prefixed `crates/`. **Engine crates must stay game-agnostic (CLAUDE.md): the gym/observation/reward/persona code is GAME-LAYER and belongs in the game repo, not in `crates/vox_*`.** The one exception is generic tensor/inference plumbing, which may live in `crates/vox_nn`.

---

## 1. Problem Statement

- The Vice Mayor today is a **deterministic rule-based advisor** (`AssistantMayor::recommend` in `src/mayor/mod.rs`): it maps three fixed problem signals (infant coverage, elder coverage + gated workers, satisfaction) to "one rung up the policy ladder, within budget." It has **no styles**, **no learning**, **no spatial action** (only policy-ladder tiers), and **no persona/voice**. It cannot zone, build, tax-tune, or trade off competing objectives.
- We have an unusually strong RL substrate that is currently unused for learning: a **deterministic sim** (`integration_hash` folds cim + economy + route metrics; id-sorted folds, no RNG/HashMap-order), a **replay format** (`CivReplay`) that reconstructs any session bit-identically, **rewind/branch/diff** (`whatif.rs`), and a **validated action space** (the R24 driver rejects hallucinated command ids and replay-logs accepted ones). Nothing trains on it.
- There is no **gym** wrapper: no `reset()`/`step(action) -> (obs, reward, done)` surface, no fixed-width observation vector, no scalar reward, no enumerable action set. Each of those is needed before any learning approach (BC, DT, PPO, MCTS) can run.
- A persona LLM path already exists but is **decorative**: `crates/vox_nn/src/llm_client.rs` (Ollama / llama-server `/v1` client) and `src/factory/llm.rs` (a `llama-completion` runner pinned to `/home/tom-espen/models/gemma4-e4b.gguf`). `LlmPlanner` in `driver.rs` is **wired but never exercised** in the deterministic gate. There is no Vice Mayor *character* that explains decisions or holds a style.

---

## 2. Done When

This is a **design + plan** deliverable. It is done when:

Running `ls docs/superpowers/specs/2026-06-15-vice-mayor-learned-brain-design.md docs/superpowers/plans/2026-06-15-vice-mayor-phased-plan.md` lists both files, AND a human reading the plan can point at: (a) the exact gym API (`UrbanGym::reset/step`), (b) the exact observation vector width and field order, (c) the exact reward function, (d) the named smallest-viable approach (MCTS baseline → behavior-cloned MLP → style-conditioned policy), with **measured-or-cited** training/inference cost on the 780M box, and (e) the exact `mayor/mod.rs` seam each phase swaps into. No claim of "trains in N minutes" appears without either a citation or a `Done When` command that measures it.

The downstream *implementation* plan's own `Done When` (the buildable artifact) is:

Running `cargo run -p urban_horizon --bin vice_mayor_demo -- --style frugal` prints a turn-by-turn transcript where the learned policy issues ≥3 distinct accepted `GameCommand` ids over a 12-month sim and ends with a higher `wellbeing_history().last()` than the `Autonomy::Off` baseline on the **same seed** — printed as two real numbers `style_frugal_wellbeing=<x> vs off_baseline=<y>` with `x > y`.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Gym `reset/step` over the deterministic sim | `step` applied twice with the same action from the same `reset` seed yields byte-identical `obs` and equal `integration_hash()` — `assert_eq!(h1, h2)` on real u64 hashes | `assert!(obs.is_some())` |
| Fixed-width observation vector | `assert_eq!(obs.len(), OBS_DIM)` AND `assert!(obs.iter().all(\|x\| x.is_finite()))` AND at least 6 entries differ between a healthy and a crisis city | returns `vec![0.0; OBS_DIM]` |
| Scalar reward from real signals | reward on a city with rising wellbeing + solvent budget is `> 0`; on a bankrupt collapsing city is `< 0` — both printed real values | `assert!(reward == reward)` |
| Enumerable discrete action set from the registry | `actions.len() >= 20` and every id resolves via `registry.get(id).is_some()` | hard-coded `vec!["noop"]` |
| MCTS planner beats the rule-based mayor | over 20 seeds, mean end-wellbeing(MCTS) `>` mean end-wellbeing(rule-based) by a printed margin | `assert!(mcts_score >= 0.0)` |
| Behavior-cloned MLP reproduces MCTS choices | on a held-out set of states, BC policy's top-1 action matches the MCTS/heuristic label `>` 60% — printed accuracy | `assert!(model.forward(x).len() == n)` |
| Style conditioning changes behavior | `frugal` style spends strictly less treasury over 12 months than `growth` style on the same seed — two printed treasury totals | both styles produce identical logs |
| Vice Mayor persona explains a decision | given a chosen `GameCommand`, the persona emits a non-empty rationale string that names the command's modeled reason (deterministic template path, LLM optional) | returns `"ok"` |

---

## 4. Architecture

The design is a **stack of independent layers**, each shippable alone, each strictly stronger than the rule-based mayor it augments. The golden rule: **everything the brain proposes goes through `driver::execute_plan` / `validate`**, so a hallucinated or unaffordable action can never mutate the city and every accepted action is replay-logged. The brain proposes; the R24 driver is the only thing that mutates.

### 4.1 The gym (`src/ai/gym.rs`) — GAME LAYER

A thin, pure wrapper over `GameState` that exposes the standard RL contract without changing the sim. It owns no policy.

- `reset(seed, scenario)` builds a fresh `GameState` (e.g. `Scenario::Demo.build()` or `GameState::new_small()`), runs it to a stable start, and returns the initial observation.
- `step(action)` translates a discrete action index → a `PlannedCommand`, runs it through `driver::execute_plan` (validate → dispatch → replay-log → undo-mark), then advances the sim a fixed number of ticks to the next *decision point* (one in-game month = `TICKS_PER_DAY * DAYS_PER_MONTH` = 4320 ticks, so fiscal/policy effects settle — matching the cadence the R22/R28 gates already use), and returns `(obs, reward, done)`.
- Determinism is inherited, not added: the gym does no RNG of its own; `reset(seed)` + an action sequence reconstructs bit-identically (proven by `integration_hash()`), which is *exactly* what makes this a reproducible training environment. This is the unfair enabler — most RL gyms cannot guarantee bit-identical replay.

**Threading:** single-threaded, owns its `GameState`. Parallel rollouts = N independent `UrbanGym` instances on N threads (the sim is `Send`; each gym is independent — no shared mutable state). This is how we get cheap data generation on a multicore CPU without a GPU.

### 4.2 Observation encoder (`src/ai/obs.rs`) — GAME LAYER

Maps `&GameState` → `[f32; OBS_DIM]` (fixed width, all finite, normalized to roughly `[-1, 1]`). Reads only stable public accessors that already exist:

- care coverage ratios: `game.coverage.coverage_ratio(DependentKind::Infant|Elder)` (and any other `DependentKind`);
- labor/economy: `game.last_stats.gated_workers`, `game.cim.population()`, `game.sim.budget.funds`, `game.treasury.cash`, live bond service from `game.treasury`;
- wellbeing: `game.wellbeing_history().last()` and a short trailing window (slope = trend, which DT/MCTS need);
- demand pressure and tax rates per `Sector` (`game.tax_policy.sector_rate(Sector::...)`);
- current policy-ladder tiers (`game.policy_state.get(ladder_id)` over the id-sorted `ladder_catalog`).

Field order is **frozen and documented** (Section 5) because a BC/DT model's weights are tied to it. `OBS_DIM` starts small (~32) deliberately — see feasibility.

### 4.3 Action space (`src/ai/actions.rs`) — GAME LAYER

A frozen, ordered `Vec<ActionDef>` built from the registry. Two tiers:

- **Tier A (ship first): policy + fiscal verbs** — arg-free registry ids (`view.tax.*.inc/dec`, `view.policy.*`, `view.ladder.*.tierN`, `sim.tick`/noop). ~20-40 discrete actions, all `PlanArgs::None`. This is the whole space the rule-based mayor uses, plus tax and the full ladder, and it is enough to be *visibly* better.
- **Tier B (later): parameterized spatial verbs** — `build.commit_zone` / `build.place_here` need `PlanArgs::Zone{outline}` / `Plop{pos}`. These have a continuous/large sub-space; handle with a small fixed candidate-site generator (deterministic) so the action stays discrete. Defer until Tier A ships.

Each `ActionDef` carries the registry `id`, an arg builder, and a static label. `step` maps index → `PlannedCommand` → `execute_plan`. An action that the driver **rejects** (unaffordable / not-applicable / unknown) is a legal no-op with a small reward penalty — the agent learns affordability the same way the rule-based gate enforces it.

### 4.4 Reward (`src/ai/reward.rs`) — GAME LAYER

A single scalar per `step`, from signals the prompt named and the code already exposes. Per-step **shaped delta** plus terminal terms:

```
r = w_well * Δ(mean_satisfaction)            // wellbeing_history delta
  + w_pill * Δ(productivity_from_care)        // two-pillars: care → fewer gated_workers
  + w_pop  * Δ(population) / pop_scale         // migration/growth
  - w_debt * max(0, -treasury.cash) / cash_scale  // solvency penalty (overdraft)
  - w_inv  * invalid_action_flag               // driver rejected the action
terminal: + w_solvent if budget solvent at horizon; large negative if bankrupt
```

Weights `w_*` are a small struct; **style conditioning is just a different weight vector** (Section 4.6). Determinism: all inputs are id-sorted folds already in `integration_hash`, so reward is reproducible.

### 4.5 The brains (three, in increasing ambition)

**(a) MCTS / one-step + shallow lookahead planner (`src/ai/mcts.rs`) — NO LEARNING.** Because the sim is deterministic and cheaply forkable via `whatif::branch` / `GameState::from_save`, we can do real planning: from the current state, for each Tier-A action, fork the gym, apply it, roll out a short fixed horizon under a cheap default policy (the existing rule-based mayor as the rollout policy), score by the reward, and pick the best. This is a UCT/MCTS or even just **depth-1 expectimax with rollouts** — and the literature says this kind of test-time search is a *surprisingly strong baseline that needs no training* and is robust to rule changes ([AlphaZero-Inspired: MCTS only at test time](https://arxiv.org/pdf/2204.13307); [AlphaZe** strong baselines](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10213697/)). **This is the smallest thing that is actually good and the recommended Phase 1.** Cost is pure CPU sim-stepping; tunable by branching factor × rollout depth × sims. Determinism caveat in 4.7.

**(b) Behavior-cloned MLP (`src/ai/policy_mlp.rs`) — SUPERVISED, TINY.** Distill the MCTS planner (and/or the rule-based mayor, and/or human replays) into a tiny MLP: `OBS_DIM → 64 → 64 → n_actions` (softmax) plus a value head. That is ~6-10k parameters — kilobytes of `f32`. Train by cross-entropy on (obs, chosen-action) pairs harvested from MCTS/heuristic/human rollouts (the gym generates them deterministically and in parallel on CPU). Inference is a couple of matmuls = microseconds, trivially deterministic. BC is the right first learned model: it needs no online interaction, and in low-data/sparse-reward settings sequence/imitation methods are competitive ([When should we prefer DT for offline RL](https://arxiv.org/abs/2305.14550)). Where to run training: PyTorch on the 780M box CPU (a model this small trains in seconds-to-minutes per epoch; MLP forward passes are <10ms on a standard CPU — [CleanRL/PPO MLP on CPU](https://docs.cleanrl.dev/rl-algorithms/ppo/)), export to a flat `f32` weight file, load in Rust (`crates/vox_nn` or a 50-line hand-rolled matmul — no heavy dep needed for a 2-layer MLP).

**(c) Style-conditioned policy / Decision Transformer (`src/ai/dt.rs`) — RESEARCH-GRADE.** To play in *different styles*, condition the policy on a style token. Two ways: (i) cheap — train **one BC-MLP per style** by relabeling rollouts with that style's reward weights (4.6); the styles are just different datasets. (ii) ambitious — a small **Decision Transformer** over `CivReplay` trajectories conditioned on a target return / style token: `(return-to-go, obs, action)` sequence modeling ([Decision Transformer](https://sites.google.com/berkeley.edu/decision-transformer)). DT needs more data than CQL/BC but is more robust in sparse-reward settings ([arxiv 2305.14550](https://arxiv.org/abs/2305.14550)); a city-scale DT is small (a few M params, GPT-nano scale) but is the heaviest thing here. **Flag (c-ii) as research-grade; ship (c-i) as the real styles feature.**

### 4.6 Styles = reward weight vectors

A `MayorStyle` is a named `RewardWeights`. Minimum set: `Frugal` (high `w_debt`, low `w_pop`), `Growth` (high `w_pop`, tolerant `w_debt`), `CareFirst` (high `w_well`+`w_pill`), `Balanced` (the rule-based mayor's implicit weights). MCTS reads the style's weights directly (no training). BC trains one tiny head per style on style-relabeled rollouts. This makes "different styles" a *data/weights* problem, not a new architecture — the cheapest possible path to the prompt's "DIFFERENT STYLES" requirement.

### 4.7 Determinism of inference (the one real subtlety)

- **MLP/DT inference is deterministic** if we (1) use integer/`f32` matmuls with a fixed reduction order (no parallel float reduction in the forward pass — trivial at this size), and (2) pick the argmax action (no sampling), or sample with a seed pinned in `obs`/state. Fixed-point or sorted-sum reductions keep it bit-stable across machines if we ever need cross-machine replay parity; for single-box play, plain `f32` argmax is already stable.
- **MCTS rollouts use the rule-based mayor as the default policy**, which is already deterministic. MCTS node selection must use a **deterministic tie-break** (UCT with `f64::total_cmp` + id-ascending tie-break, exactly the pattern `mayor/mod.rs` already uses) and a **seeded** rollout RNG if any stochastic rollout is introduced. With a deterministic default policy and seeded selection, an MCTS decision is reproducible — so a Vice Mayor game stays a valid `CivReplay`. **This is a hard requirement: the brain's chosen action must be logged as the `PlannedCommand` it produced (already true via `execute_plan`), so replay re-dispatches the command, not the brain.** The model never needs to be re-run during replay — replay re-applies the *logged commands*. That decouples replay validity from model determinism entirely, which is the key insight that makes any brain (even a non-deterministic LLM) replay-safe.

### 4.8 The Vice Mayor persona (`src/ai/persona.rs`)

Two registers, both optional and both replay-safe (they never mutate the city outside the driver):

- **Deterministic rationale (ship first):** every `Recommendation`/chosen action already carries a `reason` string (`mayor/mod.rs` builds e.g. "low infant childcare coverage: raise … to Encourage (predicted cost …)"). The persona is a template that turns the chosen `PlannedCommand` + its modeled reason + the predicted `WhatIfDiff` (from `whatif::diff` on a forked branch) into a first-person line: *"I'd raise childcare to Encourage — infant coverage is at 0.10 and it frees ~N gated parents back into the workforce for ~M/settle."* No model, fully deterministic, ships day one.
- **LLM voice (optional, local):** route the rationale through the **already-wired** local LLM (`factory::llm::LlmRunner` → `llama-completion` on the pinned `gemma4-e4b.gguf`, or `vox_nn::llm_client` → llama-server `/v1`). The LLM **only phrases**; it never picks the action (the brain + driver do). For consult mode, the LLM may also act as `LlmPlanner` (already in `driver.rs`) to turn a player's free-text goal into a command plan — and a hallucinated id is rejected by `validate_command`. The 780M runs the pinned 4B model at usable speed (12-20 tok/s on 7B-Q4, 30-50 tok/s on 3B-Q4 via Vulkan/RADV — [DEV: Gemma on Ryzen mini PC ~21 tok/s](https://dev.to/hrodrig/21-toks-gemma-4-on-a-ryzen-mini-pc-llamacpp-vulkan-and-the-messy-truth-about-local-chat-m82); [hardware-corner RADV update](https://www.hardware-corner.net/llama-cpp-amd-radv-vulkan-driver-update/)), so a one-paragraph rationale is sub-second to a few seconds. ROCm does **not** support the 780M iGPU — Vulkan only ([localaimaster ROCm guide](https://localaimaster.com/blog/amd-rocm-local-llm-setup)).

### 4.9 The integration seam (`src/mayor/mod.rs`) — unchanged shape

`AssistantMayor` stays a thin stateless delegate. We add a brain behind `recommend`/`act`:

- Today `recommend` calls `problem_signals` (rule-based). Phase 1+ adds a `brain: MayorBrain` field; `recommend` calls `brain.recommend(game, style)` which returns the same `Vec<Recommendation>` shape, and `act` in `Auto` still routes the top pick through `driver::execute_plan` — **the seam, undo-group, replay-log, and the three autonomy levels are untouched.** This is the R40 swap the R34 module's own docs anticipate ("R40 swaps in a LEARNED brain at the same seam").

---

## 5. Data Models

```rust
// src/ai/obs.rs — frozen observation layout (GAME LAYER)
/// Fixed observation width. FROZEN: BC/DT weights are tied to this and to the
/// field order in `encode`. Bumping it invalidates trained models (version it).
pub const OBS_DIM: usize = 32;
pub const OBS_VERSION: u32 = 1;

/// Encode a city into a fixed, finite, ~[-1,1]-normalized feature vector.
/// Reads ONLY stable public accessors (no private fields). Field order is the
/// contract — see the doc table in the source. Deterministic: same GameState
/// (same integration_hash) -> byte-identical vector.
pub fn encode(game: &GameState) -> [f32; OBS_DIM];

// src/ai/actions.rs — frozen discrete action set
/// One discrete action: a registry id plus how to build its PlanArgs.
pub struct ActionDef {
    id: &'static str,                // a real GameRegistry command id
    args: ActionArgs,                // None | ZoneCandidate(idx) | PlopCandidate(idx)
    label: &'static str,
}
impl ActionDef { pub fn id(&self) -> &str; pub fn label(&self) -> &str; }

/// The frozen, ordered Tier-A action table. `actions().len()` is the policy
/// output width. Every id MUST resolve via `GameRegistry::get`.
pub fn actions() -> &'static [ActionDef];
/// Turn an action index + the current view into a PlannedCommand for the driver.
pub fn to_planned(index: usize, view: &ViewState) -> Option<PlannedCommand>;

// src/ai/reward.rs — styles are weight vectors
/// Per-objective reward weights. A `MayorStyle` is just a named instance.
#[derive(Clone, Copy)]
pub struct RewardWeights {
    w_well: f32, w_pillars: f32, w_pop: f32, w_debt: f32, w_invalid: f32, w_solvent: f32,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MayorStyle { Balanced, Frugal, Growth, CareFirst }
impl MayorStyle { pub fn weights(self) -> RewardWeights; }

/// Scalar reward for a transition (prev -> next) under a style. Deterministic.
pub fn reward(prev: &GameState, next: &GameState, invalid: bool, style: MayorStyle) -> f32;

// src/ai/gym.rs — the RL environment
pub struct UrbanGym { /* owns GameState, ViewState, registry, sdf, step ticks */ }
pub struct StepOut { pub obs: [f32; OBS_DIM], pub reward: f32, pub done: bool, pub accepted: bool }
impl UrbanGym {
    pub fn reset(seed: u64, scenario: Scenario, style: MayorStyle) -> (Self, [f32; OBS_DIM]);
    pub fn step(&mut self, action: usize) -> StepOut;   // routes via driver::execute_plan
    pub fn integration_hash(&self) -> u64;              // for the determinism gate
}

// src/ai/policy_mlp.rs — tiny BC policy (load-and-infer in Rust)
/// 2-hidden-layer MLP: OBS_DIM -> H -> H -> n_actions (+ scalar value head).
/// Weights loaded from a flat little-endian f32 file exported by the trainer.
/// Forward uses a FIXED reduction order (sequential) so inference is bit-stable.
pub struct PolicyMlp { /* w1,b1,w2,b2,w_pi,b_pi,w_v,b_v as Vec<f32> */ }
impl PolicyMlp {
    pub fn load(path: &Path, obs_dim: usize, hidden: usize, n_actions: usize) -> io::Result<Self>;
    pub fn act(&self, obs: &[f32; OBS_DIM]) -> usize;   // argmax — deterministic
    pub fn value(&self, obs: &[f32; OBS_DIM]) -> f32;
}

// src/ai/brain.rs — the pluggable brain behind the R34 seam
pub enum MayorBrain {
    Rule,                       // today's deterministic advisor (unchanged)
    Mcts(MctsConfig),           // Phase 1: planning, no learning
    Bc(PolicyMlp),              // Phase 2: distilled tiny policy
}
impl MayorBrain {
    /// Same return shape as today's AssistantMayor::recommend.
    pub fn recommend(&self, game: &GameState, style: MayorStyle) -> Vec<Recommendation>;
}
```

---

## 6. API

```rust
// The seam stays identical; AssistantMayor gains a brain + style.
// src/mayor/mod.rs
pub struct AssistantMayor { pub autonomy: Autonomy, pub style: MayorStyle, brain: MayorBrain }
impl AssistantMayor {
    pub fn new(autonomy: Autonomy) -> Self;                 // defaults: Balanced + Rule (back-compat)
    pub fn with_brain(autonomy: Autonomy, style: MayorStyle, brain: MayorBrain) -> Self;
    pub fn recommend(&self, game: &GameState) -> Vec<Recommendation>;  // delegates to brain
    pub fn act(&self, view: &mut ViewState, game: &mut GameState,
               sdf: &SpatialFieldRegistry, registry: &GameRegistry) -> Option<PlanReport>;
}

// All enactment STILL goes through the unchanged R24 driver:
//   driver::execute_plan(view, game, sdf, registry, &plan) -> PlanReport
//   driver::validate_command(cmd, registry) -> Result<(), String>   // rejects hallucinated ids
//   driver::undo_plan(game, &report)                                // one undo group
//   driver::replay_commands(view, game, sdf, registry, &report.replay_log)  // determinism check
// Threading: gym + brain are single-threaded per instance; run N gyms on N
//   threads for data generation. The LLM persona runs on the main thread only
//   (matches factory::mod.rs's "blocking, main thread only" runner).
// Determinism: replay re-dispatches LOGGED COMMANDS, never the brain — so any
//   brain (even an LLM) is replay-safe; the model is not re-run on playback.
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `obs::encode` | `UrbanGym::reset`/`step` | `src/ai/gym.rs` | the only obs producer |
| `actions::to_planned` | `UrbanGym::step` | `src/ai/gym.rs` | index → `PlannedCommand` |
| `driver::execute_plan` | `UrbanGym::step` | `src/ai/gym.rs` | the ONLY mutation path |
| `reward::reward` | `UrbanGym::step` | `src/ai/gym.rs` | per-transition scalar |
| `MayorBrain::recommend` | `AssistantMayor::recommend` | `src/mayor/mod.rs` | the R34→R40 swap seam |
| `PolicyMlp::act` | `MayorBrain::Bc` arm | `src/ai/brain.rs` | argmax over actions |
| `Mcts::plan` | `MayorBrain::Mcts` arm | `src/ai/brain.rs` | forks gym via `from_save`/`whatif::branch` |
| `persona::rationale` | the demo bin + UI Vice Mayor panel | `src/ai/persona.rs` | uses `whatif::diff` for predicted effect |
| `LlmRunner` (optional voice) | `persona::rationale` LLM arm | `src/factory/llm.rs` / `crates/vox_nn/src/llm_client.rs` | phrasing only, never picks actions |
| `vice_mayor_demo` bin | new binary | `src/bin/vice_mayor_demo/main.rs` | the human-verifiable `Done When` |

---

## 8. Open Questions (answered for the plan)

- [x] **Smallest thing that's actually good?** → MCTS/depth-1+rollout planner with the rule-based mayor as the rollout policy. No training, strong baseline (cited), ships in Phase 1. Everything learned (BC/DT) is a *distillation* of it for speed/persona, not a prerequisite.
- [x] **Train where, on the 780M box?** → Tiny MLP BC trains on **CPU** in PyTorch (seconds-to-minutes; the model is ~10k params). The 780M iGPU is **not** a training device (no ROCm); reserve it (via Vulkan/llama.cpp) for the optional LLM *persona voice* only. DT (research-grade) would also train on CPU but is slow — accept hours, or skip.
- [x] **Discrete vs continuous actions?** → Discrete Tier-A (policy/fiscal) first; spatial verbs via a fixed deterministic candidate-site generator later. Never expose raw continuous coordinates to the policy.
- [x] **Replay safety of a learned/LLM brain?** → Solved structurally: replay re-dispatches the **logged `PlannedCommand`s**, not the brain. The model is never re-run on playback, so model determinism is irrelevant to `CivReplay` validity. Only the *live* decision needs the deterministic tie-break (4.7).
- [x] **Inference budget in-game?** → MLP: microseconds (a few matmuls). MCTS: bounded by `sims × depth × branching` sim-steps — cap it so a turn is <50ms; one in-game month is the decision cadence so there is generous headroom. LLM voice: 0.5-3s, off the decision path (async, only when the player opens the Vice Mayor).

---

## 9. Out of Scope

- Online RL (PPO/Q-learning with environment interaction during play). The gym *supports* it, but the data-cheap, deterministic-friendly path is offline planning + BC; online RL is a later option, not this design.
- Training any model in *this* workflow — this is design + plan only.
- Multi-agent / opponent modeling — there is one Vice Mayor, no adversary.
- Spatial Tier-B actions (zoning/placement by the policy) beyond the deferred candidate-site sketch.
- A bespoke deep-learning runtime in the engine crates — engine stays game-agnostic; learning code is game-layer; only generic tensor helpers may sit in `crates/vox_nn`.
- Cross-machine bit-identical *model* inference (only the command log needs to be portable, and it already is).

---

## 10. Related Plans / Designs

- Implements against: R34 Assistant Mayor (`urban_horizon/src/mayor/mod.rs`), R24 driver (`urban_horizon/src/command/driver.rs`), R22 replay (`.../game/replay.rs`), R28 what-if (`.../game/whatif.rs`).
- Plan: `docs/superpowers/plans/2026-06-15-vice-mayor-phased-plan.md`.
- Related engine asset: `crates/vox_nn` (LLM client + would host a generic MLP infer helper if we choose not to hand-roll it).
