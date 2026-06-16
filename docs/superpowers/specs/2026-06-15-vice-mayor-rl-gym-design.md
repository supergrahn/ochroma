# Design: Vice Mayor — a local-trainable AI that plays Urban Horizon in styles (2026-06-15)

**Status:** Draft
**Scope:** A small, locally-trainable policy that drives Urban Horizon through the existing R24 GameCommand surface, swappable into the R34 Assistant Mayor seam as a learned "brain", playable in distinct delegable STYLES (a Vice Mayor persona) and trainable on the deterministic sim as a reproducible RL gym + offline dataset. Affects only the GAME layer (`urban_horizon` repo); engine crates are untouched.
**Related:** R24 driver (`src/command/driver.rs`), R28 what-if (`src/game/whatif.rs`), R22 replay (`src/game/replay.rs`), R34 Assistant Mayor (`src/mayor/mod.rs`).

> NOTE ON REPO: the unfair enablers named in the brief live in the **game** repo `~/Ochroma/projects/urban_horizon_r39`, NOT in `~/src/ochroma` (the engine). This design therefore lands in the engine's docs tree (per CLAUDE.md plan/design location rule) but every code path it references is in the game repo. All file paths below are relative to `~/Ochroma/projects/urban_horizon_r39`.

---

## 1. Problem Statement

- The R34 Assistant Mayor (`src/mayor/mod.rs`) is a hand-written rule brain: it scans three fixed problem signals (`infant`, `elder`, `satisfaction`) and proposes "one rung up" on a ladder. It cannot trade off across axes, cannot plan multi-step, and has exactly ONE behaviour — there is no "style".
- There is no environment surface a learner can `reset()`/`step()` against. The sim ticks (`GameState::tick`) and commands dispatch (`GameRegistry::dispatch`), but nothing exposes (observation, legal-action-mask, reward) as a loop.
- The brief asks for an agent that plays in DIFFERENT STYLES and is a consultable PERSONA. Today nothing maps a "play like X" intent onto behaviour.
- We have three latent enablers that are unused for learning: a deterministic sim with an `integration_hash` witness, a validated action registry, and a replay/what-if system that is a ready-made offline dataset + model-based rollout engine. None of this is wired to a policy.
- Feasibility is unproven on the target box (AMD 780M iGPU, no CUDA, no GPU farm). We must decide what is buildable-now vs research-grade BEFORE committing engineering.

---

## 2. Done When

Running `cargo run -p urban_horizon --bin vice_mayor -- play --style frugal --episodes 1 --print-trajectory` prints, to stdout, a full episode trace ending in a line of the exact form:

```
EPISODE style=frugal ticks=8640 actions=37 final_pop=<N> final_treasury=<F> return=<R> hash=<0x...>
```

where `actions` > 0 (the learned policy actually issued commands through the R24 driver), every issued command id appears in the printed replay log, and re-running the same command with `--seed <same>` reproduces a byte-identical `hash=`. A human at the keyboard sees the Vice Mayor take a sequence of real, legal city actions and sees the run reproduce.

Separately, `cargo run -p urban_horizon --bin vice_mayor -- compare --styles frugal,carefirst --episodes 8` prints a table where the two styles produce **measurably different** mean treasury and mean satisfaction (the `abs(delta) > 0` is asserted), proving the styles are distinct behaviours and not one policy with a relabelled name.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Gym env reset/step | `step()` on a fixed seed twice yields byte-identical `integration_hash` after the same action sequence | `assert!(env.step(a).is_ok())` |
| Observation featurizer | `observe(&game)` on `city_with_low_childcare` returns a vector whose infant-coverage feature equals `0.10 ± 1e-6` (the seeded crisis) | `assert_eq!(obs.len(), N)` — checks shape only |
| Legal-action mask | mask on a 0-funds city has the `view.ladder.*.tier1` bits = 0 for unaffordable tiers (reuses R24's budget gate + `validate_command`) | `assert!(mask.iter().any(\|b\| *b))` |
| Reward function | a step that raises infant coverage from 0.10→0.80 yields strictly higher reward than the no-op step on the same state | `assert!(reward.is_finite())` |
| Style conditioning | `frugal` and `carefirst` policies on the SAME observation pick different actions ≥ 30% of decision points over an episode | two styles return the same action vector |
| Offline dataset from replays | a `.civreplay` converts to ≥1 (obs, action, reward) transition whose action id matches a logged `AuthoredAction` | `assert!(!dataset.is_empty())` |
| Behaviour-cloning policy | BC policy trained on a scripted expert reproduces the expert's first action on a held-out state with > 70% top-1 match | loss decreases (no behaviour check) |
| R34 seam swap | a `LearnedMayor` wired behind `Autonomy::Auto` enacts through `driver::execute_plan` and its action is in the replay log + reversible by one `undo_plan` | `recommend()` returns non-empty |

---

## 4. Architecture

The system is four layers, each independently testable, built game-side. **No engine crate changes.** **No new heavyweight ML dependency in the shipped game build** — training tools live behind a `cfg(feature = "vice_train")` and an optional `ndarray`/`burn` dev-dependency; the *inference* policy at runtime is a plain matrix-multiply over `f32` weights loaded from a file (zero ML runtime dependency, deterministic, CPU-cheap).

### 4.1 The Gym (`src/vice/gym.rs`)

A thin, pure wrapper over the existing sim — it owns NO new simulation. `CityEnv` holds a `GameState`, `ViewState`, `SpatialFieldRegistry`, `GameRegistry`, an episode budget (`max_ticks`), a `ticks_per_decision` cadence (the agent does not act every tick — it acts on a coarse cadence, e.g. once per civic week, then the sim fast-forwards), and an RNG-free seed used only to pick the start scenario.

`reset(seed) -> Observation` builds a scenario (`Scenario::Demo.build()` family, seed selects which) and returns the first observation. `step(action) -> StepOut` decodes the action index into a `PlannedCommand`, runs it through `driver::execute_plan` (so it is validated, replay-logged, one undo group — we get the moat for free), advances the sim `ticks_per_decision` ticks, computes reward as the delta of a scalar "city score" plus shaping, and returns `(obs, reward, done, info)`. Determinism is inherited verbatim from the sim: same seed + same action sequence ⇒ same `integration_hash` (this is the property R28 already proves in `whatif.rs`). The gym adds NO randomness.

A `no-op` action (index 0) is always legal and advances time without issuing a command — this is essential so the agent can "wait", and so an episode of all-no-ops is a valid baseline.

### 4.2 Observation + Action spaces (`src/vice/spaces.rs`)

**Observation** is a fixed-length `f32` vector (target ~48 features), each a normalised aggregate already computed by the sim — NO new sim state:
- care coverage ratios per `DependentKind` (`game.coverage.coverage_ratio(..)`),
- `last_stats` fields normalised by population (`gated_workers/pop`, `unmet_*`/pop, `effective_employed/pop`, `care_period_cost/funds`),
- fiscal: `sim.budget.funds`, `treasury.cash`, `household_income`, `cost_of_living` (each scaled by a fixed `log1p` to tame magnitudes),
- wellbeing: last-K samples of `wellbeing_history()` (the satisfaction ring),
- population + a coarse growth delta (this-week pop − last-week pop),
- the current policy-ladder tier of each catalogued ladder (`policy_state.get(id).magnitude()`), id-sorted.

All features are pulled in **id-sorted / fixed order** so the vector is deterministic. Magnitudes are clamped/normalised with FIXED constants (no per-run statistics) so an observation is reproducible and a saved policy stays valid.

**Action space** is the discrete set of registry verbs the Mayor is allowed to use, enumerated in id-sorted order from `GameRegistry::standard()`, filtered to the *delegable* subset (policy-ladder tier moves, tax inc/dec per sector, bond issue at fixed parameters, subsidy at fixed parameters, and `no-op`). Parameterised verbs (zone outlines, plops) are OUT for v1 — they need a parameter head and a spatial observation; v1 is a POLICY/FISCAL agent, which is exactly the R34 surface. Each action index ↔ a `PlannedCommand` via a static table built once at env construction.

**Legal-action mask** (the single most important correctness lever, per the literature — masking made the difference between stable and unstable training in Ubisoft's and the µRTS work). The mask is computed PER STEP and reuses what R24/R34 already enforce:
1. `validate_command` rejects ids not in the registry (none of ours are fabricated, so this is a guard, not a filter),
2. the R34 budget gate — a `view.ladder.X.tierN` whose `tier.cost.charge(pop, jobs) > funds + ε` is masked off (lifted directly from `mayor::recommend`),
3. an "authored-nothing" predicate: a tier move to the tier you are already on is masked (no-op-disguised-as-action), mirroring the driver's `is_author` log-grew check.
The masked logits are set to `-inf` before the softmax/argmax, so the policy can NEVER select an illegal action and we never waste a sample on a rejected command.

### 4.3 Reward + styles (`src/vice/reward.rs`)

Reward is a **potential-based shaped** delta to resist reward hacking (Ng's potential shaping keeps the optimal policy invariant; combined with hard masks it narrows the exploit surface): `r_t = γ·Φ(s_{t+1}) − Φ(s_t) − λ·cost_t`, where `Φ` is a weighted city-score over the TWO PILLARS (care coverage + wellbeing) AND solvency (a treasury floor as a hard penalty, not a maximisation target — so the agent cannot "win" by hoarding cash and doing nothing), and `cost_t` lightly penalises issuing expensive actions so the agent does not churn policies. A **STYLE** is just a different fixed weight vector `w_style` over the same `Φ` terms plus a small per-style action-bias: `frugal` weights treasury floor + cost penalty heavily; `carefirst` weights coverage; `growth` weights population delta; `balanced` is uniform. Crucially styles share ONE reward *shape* and ONE network *architecture* — they differ only by the conditioning vector appended to the observation and the reward weights at train time. This is "goal-conditioned RL": one policy net, style id as input. (This is what makes "different styles" cheap: we do not train N agents, we train one style-conditioned agent — though v1 may train them separately for simplicity and merge later.)

Anti-reward-hacking measures, named explicitly because the literature shows proxies decouple under optimisation pressure (Goodhart): (a) hard masks remove the cheapest exploits (illegal/unaffordable/no-op actions); (b) potential shaping rather than raw-metric maximisation; (c) the solvency term is a FLOOR penalty, not a reward, so bankruptcy is punished but cash-hoarding is not rewarded; (d) an episode-level audit that re-runs the policy's replay log and asserts the claimed return reproduces (the determinism witness catches a policy that "scored" via nondeterminism); (e) a held-out evaluation on scenarios not seen in training to detect goal-misgeneralisation.

### 4.4 The policy + training (`src/vice/policy.rs`, `src/vice/train.rs` [cfg vice_train])

**Inference policy** (always compiled, zero ML dep): a small MLP — input `obs ⊕ style_onehot` (~52 floats) → 2 hidden layers (e.g. 64, 64, tanh) → logits over the action set. Forward pass is hand-written `f32` matmul; weights load from a `.vicepolicy` file (a versioned, little-endian flat dump). Argmax over masked logits = the action. This is < 10k parameters, microseconds per decision on CPU — trivially within frame budget, and deterministic.

**Training** (dev-only, `cfg(feature = "vice_train")`): three tractable paths, in feasibility order —
1. **Behaviour cloning (BUILDABLE NOW, cheapest):** treat the existing deterministic `StubPlanner`/rule `AssistantMayor` AND any human `.civreplay` as an expert. Convert episodes to (obs, action) pairs, fit the MLP by supervised cross-entropy. No environment interaction loop needed; trains in seconds on CPU. This alone delivers a "learned brain" that imitates the rule mayor and any human style we record — and is the safe v1 to ship.
2. **Offline RL / filtered BC (BUILDABLE, more value):** label replay transitions with reward, keep top-return episodes (return-conditioned / "upside-down RL" / a tiny Decision-Transformer-style return token), fit. Uses ONLY logged data — no live rollouts — so it is as cheap as BC and learns to *exceed* the demonstrator on the logged distribution.
3. **Online PPO on the gym (BUILDABLE-with-care, the literature's CPU-friendly path):** the searched evidence is consistent — small-MLP PPO is meant for CPU; the bottleneck is environment throughput, not the net. Our env step is one `execute_plan` + N sim ticks; the lever is `ticks_per_decision` (act weekly, not per-tick) and vectorising K envs across CPU threads (the sim is `Send`). Feasibility hinges on a measured `steps/sec` — Task in the plan MEASURES it before any PPO is written. This is the only path flagged RESEARCH-GRADE-until-measured.

A model-based bonus: R28 `branch()` lets us roll out "what if the agent did X at tick T" WITHOUT live play — a free model-based planner / data-augmentation source built from the sim itself.

---

## 5. Data Models

```rust
/// One env step's outcome. Pure data; no interior mutability.
pub struct StepOut {
    pub obs: Observation,          // next observation
    pub reward: f32,               // shaped potential delta (this style's weights)
    pub done: bool,                // episode hit max_ticks
    pub legal: ActionMask,         // mask for the NEXT decision
    pub accepted: bool,            // did the chosen command actually author?
}

/// Fixed-length normalised observation. Private buffer; accessor returns a slice.
pub struct Observation { feats: Vec<f32> }      // len == OBS_DIM (const)
impl Observation { pub fn as_slice(&self) -> &[f32]; pub const DIM: usize; }

/// One bit per action index; true == legal this step.
pub struct ActionMask { bits: Vec<bool> }
impl ActionMask { pub fn is_legal(&self, a: usize) -> bool; pub fn legal_count(&self) -> usize; }

/// Maps action index <-> the registry PlannedCommand it issues.
pub struct ActionTable { rows: Vec<PlannedCommand> }   // built from GameRegistry::standard()
impl ActionTable {
    pub fn build(reg: &GameRegistry) -> Self;          // id-sorted, no-op at index 0
    pub fn command(&self, a: usize) -> &PlannedCommand;
    pub fn len(&self) -> usize;
}

/// A play style: a name, reward weights, a one-hot id for conditioning.
#[derive(Clone, Copy)]
pub struct Style { id: u8 }   // Frugal=0, CareFirst=1, Growth=2, Balanced=3
impl Style { pub fn weights(&self) -> RewardWeights; pub fn onehot(&self) -> [f32; STYLE_N]; pub fn from_str(s: &str) -> Option<Style>; }

/// The CPU inference policy: flat f32 weights, hand-written forward pass.
pub struct VicePolicy { /* layer weights/biases, private */ }
impl VicePolicy {
    pub fn load(path: &Path) -> std::io::Result<Self>;
    pub fn save(&self, path: &Path) -> std::io::Result<()>;
    /// Masked argmax. Deterministic. Illegal actions can never be returned.
    pub fn act(&self, obs: &Observation, style: Style, mask: &ActionMask) -> usize;
    /// Logits (for training / inspection), pre-mask.
    pub fn logits(&self, obs: &Observation, style: Style) -> Vec<f32>;
}
```

---

## 6. API

```rust
// --- gym.rs ---
pub struct CityEnv { /* GameState, ViewState, SpatialFieldRegistry, GameRegistry, cfg */ }
impl CityEnv {
    /// Build env. `ticks_per_decision` is the coarse act cadence (e.g. one week).
    pub fn new(seed: u64, max_ticks: u64, ticks_per_decision: u64) -> Self;
    /// Reset to a fresh scenario; returns the first observation + first mask.
    pub fn reset(&mut self) -> (Observation, ActionMask);
    /// Apply action `a` via driver::execute_plan, advance the sim, return StepOut.
    /// Panics: never. An illegal `a` is rejected and treated as no-op + small penalty.
    pub fn step(&mut self, a: usize, style: Style) -> StepOut;
    /// The deterministic witness for replay/audit.
    pub fn integration_hash(&self) -> u64;     // delegates to GameState::integration_hash
    pub fn action_table(&self) -> &ActionTable;
}

// --- spaces.rs ---
pub fn observe(game: &GameState) -> Observation;                  // fixed order, fixed normalisation
pub fn legal_mask(game: &GameState, table: &ActionTable, reg: &GameRegistry) -> ActionMask;

// --- reward.rs ---
pub struct RewardWeights { /* per-Phi-term f32 weights, private */ }
/// Potential of a state under a style's weights. Pure.
pub fn potential(game: &GameState, w: &RewardWeights) -> f32;
/// r = gamma*Phi(next) - Phi(prev) - lambda*action_cost. Pure.
pub fn shaped_reward(prev: &GameState, next: &GameState, action_cost: f64, w: &RewardWeights) -> f32;

// --- offline.rs ---
/// Convert a CivReplay into labelled transitions by re-simulating tick-by-tick
/// (reuses whatif::rewind_to semantics — pure re-simulation, no snapshot).
pub fn replay_to_transitions(rep: &CivReplay, style: Style) -> Vec<Transition>;
pub struct Transition { pub obs: Observation, pub action: usize, pub reward: f32, pub ret_to_go: f32 }

// --- mayor seam (modify src/mayor/mod.rs) ---
/// A learned brain behind the SAME Autonomy seam. Auto enacts top masked action
/// through driver::execute_plan exactly like the rule mayor (validated, logged, reversible).
pub struct LearnedMayor { policy: VicePolicy, style: Style, autonomy: Autonomy }
impl LearnedMayor {
    pub fn recommend(&self, game: &GameState, reg: &GameRegistry) -> Vec<Recommendation>;
    pub fn act(&self, view: &mut ViewState, game: &mut GameState,
               sdf: &SpatialFieldRegistry, reg: &GameRegistry) -> Option<PlanReport>;
}
```

Threading: all of the above are single-threaded per env; `CityEnv` is `Send` (the sim already is, per the what-if re-simulation), so K envs run on K rayon threads for vectorised PPO. No `Arc<Mutex<>>`.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `CityEnv::step` | `vice_mayor` bin `play`/`compare` loop | `src/bin/vice_mayor.rs` | the runnable proof of Done-When |
| `observe` / `legal_mask` | `CityEnv::reset` + `CityEnv::step` | `src/vice/gym.rs` | every decision |
| `VicePolicy::act` | the play loop AND `LearnedMayor::act` | `src/vice/policy.rs` | one forward pass per decision |
| `LearnedMayor::act` | the R34 autonomy dispatch where `AssistantMayor::act` is called today | `src/mayor/mod.rs` + its caller in the app shell | the production swap — same `execute_plan` seam |
| `replay_to_transitions` | `vice_mayor` bin `bc`/`offline` subcommand | `src/vice/offline.rs` | training data build, dev-only |
| `shaped_reward` | `CityEnv::step` | `src/vice/reward.rs` | per step |

---

## 8. Open Questions (resolved for the plan)

- [x] Per-tick or coarse decisions? → **Coarse** (`ticks_per_decision` = one civic week). Per-tick is intractable (8640 ticks/episode) and policy churn is meaningless intra-week.
- [x] One style-conditioned net or N nets? → v1 trains/ships **separate small nets per style** (simplest, proves "distinct styles"); the architecture already takes a style one-hot so merging to one conditioned net is a later, non-breaking change.
- [x] Which training path first? → **Behaviour cloning** from the rule mayor + scripted experts (buildable now, seconds on CPU). PPO is gated behind a MEASURED throughput task and is explicitly research-grade-until-measured.
- [x] New ML dependency in shipped game? → **No.** Inference is hand-written matmul. Training tools are `cfg(feature = "vice_train")` dev-only.
- [x] Parameterised spatial actions (zoning/plopping)? → **Out of v1.** v1 is the policy/fiscal Vice Mayor (exactly R34's surface). Spatial needs a parameter head + spatial obs — a later design.

---

## 9. Out of Scope

- Spatial/geometric actions (zone outlines, ploppable placement, road drawing) — needs a parameter-action head and a spatial observation tensor; deferred.
- Any GPU training. Target is CPU on the 780M box; no CUDA path is assumed.
- Natural-language persona dialogue (the "consult the Vice Mayor in chat" UX) — the persona here is a *behavioural* style + a structured recommendation list; conversational framing is a separate UI design, can reuse the existing `LlmPlanner` seam.
- Engine-crate changes. Everything is game-side under `src/vice/` + the existing `src/mayor/` seam.
- Multi-agent / competing mayors.

---

## 10. Related Plans / Designs

- Depends on: R24 driver, R28 what-if, R22 replay, R34 mayor (all SHIPPED in `urban_horizon_r39`).
- Required before: any "consultable Vice Mayor chat UI" design.
- Related: `LlmPlanner` (`src/command/driver.rs`) — the natural-language goal→command path, complementary (LLM picks the goal, learned policy executes the style).
