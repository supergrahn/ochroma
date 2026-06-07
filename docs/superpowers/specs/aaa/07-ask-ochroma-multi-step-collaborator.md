> **Adversarial verification:** `sound=false` for a **truncation reason, not a design defect**: the copy of the markdown the skeptic received was cut off mid-§2.3 (last line it saw: `...(b) sets ShellEntity.pos = pos (mod.rs:91), and (c`). In that truncated view there was **no Done When, no Capabilities/test table, no first implementation step, and no Wiring section** — so the skeptic correctly judged it "not buildable as written" against the CLAUDE.md + design-template requirements. The **full spec below is complete** (§3 has three ordered steps each with an exact Done-When and a real computed-outcome assertion, and §4 has full Wiring/dependencies). The skeptic's checkable grounding refs verified clean (the LLM safety surface, the prompt-pin test, the planting/undo machinery). See **Verification corrections**.


# Design: Ask-Ochroma as a Multi-Step Collaborator (Sequenced Intents) (2026-06-07)

**Status:** Draft
**Scope:** Lift Ask-Ochroma from one-sentence→one-action to one-sentence→a *sequence* of validated actions executed as a single grouped-undo transaction, so a domain creator can say "add 5 birch trees" and get 5 real, distinctly-placed entities removed by one Ctrl+Z. Affects `vox_app` editor shell only (game layer); engine crates untouched.
**Related:** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #6, Editor dimension), `docs/superpowers/specs/2026-06-06-editor-sota-shell-design.md`
**Effort:** L
**Rank:** 07

> Roadmap gap #6, Editor Workflows dimension. The seed first-slice in the roadmap — "`IntentAction::Plan(Vec<IntentAction>)` + a 'row of N' expander executed as one undo group" — is a SEED, and grounding in the real code revealed three facts that reshape it: (a) the existing `IntentAction::AddNode` adds a *node-graph node*, NOT a planted world entity, so "add 5 trees" cannot reuse it; planting goes through `plant_grown_tree`/`plant_asset`, a path the parser does not currently reach at all; (b) there is NO grouped-undo entry today — `plant_asset` pushes exactly one `UndoEntry::PlacedAsset` and `undo()` pops exactly one, so grouped undo is genuinely new machinery; (c) `plant_grown_tree` always plants at the single fixed `TREE_PLANT_ORIGIN`, so N trees would stack at one point — "distinct positions" requires a per-step offset that does not exist yet. The design below addresses all three.

---

## 1. What we need

The capability: Ask-Ochroma resolves ONE natural-language sentence into a *sequenced plan* of N actions, executes them as a single coalesced undo transaction, and returns one receipt naming all N results. Today `run_intent` (`crates/vox_app/src/shell/mod.rs:639`) maps one sentence to exactly one `IntentAction` and one effect; "build me a forest" is impossible.

After this exists, a developer/user can:

- **Type "add 5 birch trees" and get 5 real entities.** `shell.run_intent("add 5 birch trees")` returns a receipt naming 5 trees ("Planted 5 Silver Birch — Silver Birch 01…05"), and `shell.entities.len()` grows by exactly 5, each at a distinct world position (a row), each carrying its own splat range in the overlay. (Concrete: assert `entities.len() == before + 5` and that the 5 `ShellEntity.pos[0]` x-coordinates are strictly increasing.)
- **Undo the whole plan with ONE Ctrl+Z.** A single `edit.undo` removes all 5 trees and all 5 splat ranges, restoring `entities.len()` and `overlay.len()` to their exact pre-plan values. (Concrete: after one undo, `entities.len() == before` and `overlay.len() == overlay_before`.) AAA bar: a multi-step generative action is *one* atomic transaction in the history, exactly as a "Paste 200 instances" op is one undo in Unreal/Houdini — not N separate undos the user must spam.
- **Read one honest receipt for the whole plan.** The Output Log shows one grouped receipt with a per-step breakdown count, not 5 disconnected lines, and the receipt names the exact count and species (the provability/receipt culture: every value is real and computed, never "done").
- **Get a clamped, bounded plan.** A hostile "add 9999999 trees" is clamped to a documented per-plan cap (e.g. 64) with the receipt stating the clamp ("requested 9999999, planted 64 — capped"); the numeric count flows through a clamp exactly like every other Ask-Ochroma numeric input.
- **Re-use the same Plan contract from the LLM seam.** The LLM backend (`IntentBackend::Llm`) can emit a `Plan` of validated sub-actions through the *same* schema gate that guards single actions today — so connecting a real model (gap #13) inherits multi-step planning for free, with no new safety surface.

Why it is blocking (Editor Workflows dimension, roadmap §3 #6): the target audience is "domain-knowledgeable non-game-devs" who can *describe* a village but cannot hand-place 200 buildings. Single-step intent is a demo; the sequenced collaborator is the product wedge — it is Ochroma's substitute for headcount (roadmap §4: "the AI collaborator is our substitute for headcount"). It also unblocks gap #13 (real LLM backend) by establishing the Plan contract the model emits.

---

## 2. How it's gonna be (the design)

Everything lives in `vox_app` (the GAME layer) — planting trees/buildings is game-specific content, forbidden in engine crates per CLAUDE.md. Two files change: `crates/vox_app/src/shell/intent.rs` (the parser/contract) and `crates/vox_app/src/shell/mod.rs` (the executor + grouped undo).

### 2.1 The Plan contract (NEW — `intent.rs`)

Add one variant to the existing enum (`intent.rs:32`):

```rust
pub enum IntentAction {
    SetParam { .. },            // existing
    AdjustParam { .. },         // existing
    AddNode { .. },             // existing — adds a NODE-GRAPH node, not a world entity
    RunCommand { .. },          // existing
    GenerateScript { .. },      // existing
    Unknown { .. },             // existing
    /// NEW: a sequenced plan of leaf actions, executed in order as ONE
    /// coalesced undo transaction. Flat (no nested Plan) by construction —
    /// the parser/validator never emits a Plan inside a Plan, so execution
    /// is a single non-recursive loop and the undo group is a flat range.
    Plan { label: String, steps: Vec<IntentAction> },
}
```

A NEW leaf variant is also required, because "add 5 trees" must *plant* (not add a graph node). `AddNode` is the wrong target — verified at `intent.rs:926` (`try_add_node` maps "tree"→`"VegetationNode"`, a node-graph kind) and `mod.rs:667` (the executor calls `bridge.add_node_by_kind`, which adds a graph node, never a world entity). The planting path is `plant_grown_tree`→`plant_asset` (`mod.rs:1050`/`1117`), which the parser does not currently reach. So:

```rust
/// NEW leaf: plant a grown FloraPrime tree of a named species at a world
/// position. The executor routes this to `plant_grown_tree_at` (a positioned
/// variant of the existing `plant_grown_tree`).
PlantTree { species_label: String, species_id: i32, class: &'static str, pos: [f32; 3] },
```

`Plan` carries the *positioned* leaves directly, so position assignment (the "row" stepping) happens at PARSE time and the executor stays dumb. `species_label`/`species_id`/`class` mirror a `FLORAPRIME_SPECIES` row verbatim (`plugins.rs:502`: `("Silver Birch", 0, "broadleaf")`), so the leaf can never name a phantom species.

### 2.2 The parse rule (NEW — `intent.rs`)

A new rule `try_plant_plan(lower) -> Option<IntentAction>`, inserted in `parse_intent` *before* `try_add_node` (so "add N trees" wins the noun match over the node-graph "add a tree"):

- Match `add|plant|grow [N] <species-or-"tree"> [trees]`, extracting the leading integer N (default 1 if absent).
- Resolve the species word against a small synonym map over `FLORAPRIME_SPECIES`: `"birch"→("Silver Birch",0,"broadleaf")`, `"oak"→("English Oak",1,...)`, `"pine"→("Scots Pine",2,...)`, `"spruce"→(...,3,...)`, bare `"tree"`→default Silver Birch. (Note: the table is verified to contain these exact rows; the parser today has NO "birch" vocabulary — confirmed by grep.)
- **Clamp N** to `1..=PLAN_MAX_STEPS` (a documented const, e.g. 64). This is the per-step "every numeric input clamps" rule applied to a *count*.
- Lay out the N positions as a row stepped along +X from a base origin: `pos_i = [BASE_X + i*ROW_STEP_M, base_y, base_z]`, where `BASE` = `TREE_PLANT_ORIGIN` (`plugins.rs:643`, `[-4,-1,-6]`) and `ROW_STEP_M` is a documented spacing const (e.g. 4.0 m). This is the fix for the stacking bug — `plant_grown_tree` today always plants at the single fixed origin.
- Emit `IntentAction::Plan { label: "Planted N <species>", steps: vec![PlantTree{..pos_0}, .. PlantTree{..pos_{N-1}}] }`.

### 2.3 The executor + grouped undo (NEW — `mod.rs`)

`run_intent`'s match (`mod.rs:654`) gains a `Plan` arm. Grouped undo is genuinely new — today `plant_asset` (`mod.rs:1117`) pushes one `UndoEntry::PlacedAsset` and `undo()` (`mod.rs:945`) pops one. Two new pieces:

A NEW `UndoEntry::Group`:

```rust
pub enum UndoEntry {
    ParamSet { .. },            // existing
    PlacedAsset { .. },         // existing
    GeneratedScript { .. },     // existing
    /// NEW: a coalesced transaction — undo replays each member in REVERSE,
    /// reusing the existing per-entry undo logic. `members` are the SAME
    /// entry variants a single action would push, captured instead of pushed
    /// individually during plan execution.
    Group { label: String, members: Vec<UndoEntry> },
}
```

A NEW positioned planting core `plant_grown_tree_at(&mut self, species_label, class, species_id, pos) -> UndoEntry` that does what `plant_grown_tree` does but (a) offsets every splat to `pos` via `GaussianSplat::set_position`/`apply_transform` (`types.rs:181`/`198`, both verified) — translating the skeleton from `TREE_PLANT_ORIGIN` to `pos`, (b) sets `ShellEntity.pos = pos` (`mod.rs:91`), and (c) **returns** the `UndoEntry::PlacedAsset` it built instead of calling `push_undo`. The existing `plant_grown_tree` is refactored to call `plant_grown_tree_at(.., TREE_PLANT_ORIGIN)` then `push_undo(entry)` — preserving its exact current behavior (the existing tree tests at `mod.rs:3246`/`3284` stay green unchanged).

The `Plan` executor:

```
run_intent("add 5 birch trees")
        │ parse_intent → IntentAction::Plan{ label, steps:[PlantTree×5 @ stepped pos] }
        ▼
for step in steps:                          ┐
    entry = plant_grown_tree_at(step.pos)   │  splats land in overlay,
    members.push(entry)   // capture, NOT   │  ShellEntity added, viewport
                          //   push_undo     │  invalidated — but undo entry
                                             ┘  is COLLECTED, not pushed
push_undo(UndoEntry::Group{ label, members })   // ONE entry on the stack
return "Planted 5 Silver Birch — Silver Birch 01…05 — undo with Ctrl+Z"
```

And the `undo()` arm for `Group` (`mod.rs:945`): pop the group, replay each member in reverse order through the *existing* per-variant undo logic (factored into a `fn undo_one(&mut self, entry: UndoEntry) -> String` so `PlacedAsset`/`ParamSet`/`GeneratedScript` revert logic is shared, not copy-pasted). Reverse order matters: the last-planted tree owns the highest overlay range, and the existing `PlacedAsset` undo shifts later ranges down (`mod.rs:967-975`) — reverting newest-first keeps every member's `start` valid as the overlay shrinks.

### 2.4 LLM-seam wiring (the prompt-pin constraint — MANDATORY)

There is a load-bearing test, `prompt_schema_pins_every_variant` (`intent.rs:1157`), that does an EXHAUSTIVE match over `IntentAction`. Adding `Plan` and `PlantTree` will FAIL TO COMPILE until four things are updated in lockstep: (1) the exhaustive match arm, (2) `describe_intent_variants()` (`intent.rs:744`, the prompt text), (3) the `LlmIntent` serde DTO (`intent.rs:528`) with a `Plan(Vec<LlmIntent>)` + `PlantTree{species,count}` shape, (4) `parse_llm_intent` (`intent.rs:593`) validating each Plan member against the schema and rejecting nested Plans (flat-only invariant). This is a feature, not a tax: the compiler guarantees the model is never told about a variant the validator doesn't gate. The LLM `Plan` reuses `try_plant_plan`'s positioning so the model emits `{species, count}` and the *validator* lays out positions — the model never invents coordinates.

### 2.5 Key decisions & rationale

- **Positions assigned at parse time, executor stays dumb.** Keeps the executor a flat loop and makes the plan fully inspectable/testable before any side effect (a parse-only unit test asserts the 5 stepped x-coords with zero world mutation).
- **Flat plans only (no nested Plan).** Avoids recursive undo and unbounded fan-out; the validator rejects a Plan-in-Plan. The roadmap's "village" → many leaves is still expressible as one flat Plan of mixed leaves later.
- **`Group` reuses `undo_one`, never re-implements revert.** No copy-pasted planting/undo machinery — the wave-13/14 lesson encoded in the existing range-tracked undo.
- **Count is clamped, not rejected.** A domain user who says "a hundred" gets a capped result with an honest receipt, not an error — matching the existing clamp-don't-reject philosophy (`SchemaContext` ranges, `script_gen` clamps).

---

## 3. How it's gonna be made (the implementation plan)

### Step 1 — Plan contract + parse rule + positioned planting + grouped undo, end-to-end (S→M)

**This is the launchable agent task for tomorrow.** It delivers the roadmap Done-When verbatim. All in `crates/vox_app/src/shell/intent.rs` and `crates/vox_app/src/shell/mod.rs`.

Files & changes:
- `intent.rs`: add `IntentAction::Plan` + `IntentAction::PlantTree`; add `try_plant_plan`; insert it in `parse_intent` before `try_add_node`; add `PLAN_MAX_STEPS=64`, `ROW_STEP_M=4.0` consts; update the four pin points (§2.4) so the crate compiles.
- `mod.rs`: add `UndoEntry::Group`; add `plant_grown_tree_at(..) -> UndoEntry`; refactor `plant_grown_tree` to delegate; factor `undo_one`; add `Plan` arm to `run_intent`; add `Group` arm to `undo()`.

**Done When (exact command + exact observable output):**

`cargo test -p vox_app add_five_birch_trees_is_one_undo_group -- --nocapture` prints `OK: 5 trees, distinct x, one undo restores 0` and passes, where the test asserts (real computed outcomes — no `is_some()`):

```rust
let mut shell = EditorShell::default();
shell.install_floraprime();
let (e0, o0) = (shell.entities.len(), shell.overlay.len());

let receipt = shell.run_intent("add 5 birch trees");
// receipt names the count + species
assert!(receipt.contains("5") && receipt.contains("Silver Birch"), "receipt: {receipt}");
// exactly 5 new entities, all Silver Birch NN
assert_eq!(shell.entities.len(), e0 + 5);
let xs: Vec<f32> = shell.entities[e0..].iter().map(|e| e.pos[0]).collect();
// distinct, strictly-increasing x => a real row, not a stack
for w in xs.windows(2) { assert!(w[1] > w[0], "trees must step in x: {xs:?}"); }
// overlay grew by 5 trees' worth of splats
assert!(shell.overlay.len() > o0, "5 trees added splats");
// exactly ONE undo entry on the stack (the group), not 5
assert_eq!(shell.undo_stack.len(), 1, "the plan is ONE undo entry");

// ONE undo removes all 5
shell.run_intent("__noop__"); // (or directly) -> use the undo path:
assert!(shell.registry.run("edit.undo"));
shell.drain_requests();
assert_eq!(shell.entities.len(), e0, "one undo removes all 5 entities");
assert_eq!(shell.overlay.len(), o0, "one undo restores the overlay exactly");
println!("OK: 5 trees, distinct x, one undo restores 0");
```

Plus a parse-only test `plant_plan_parses_five_stepped_positions` asserting `try_plant_plan` (via `parse_intent`) on `"add 5 birch trees"` yields `IntentAction::Plan{ steps }` with `steps.len()==5`, each a `PlantTree{species_label:"Silver Birch", species_id:0,..}`, and the 5 `pos[0]` values equal `[-4.0, 0.0, 4.0, 8.0, 12.0]` exactly (BASE_X=-4, ROW_STEP=4) — a pure-function assertion, zero world state.

Plus a clamp test `plant_plan_clamps_count`: `parse_intent("add 9999999 trees", ..)` yields a Plan with exactly `PLAN_MAX_STEPS` steps.

**Headless proof:** the existing tree tests (`grow_tree_plants_splats_and_world_entity_through_drain` at `mod.rs:3246`, `undo_removes_grown_tree_splats_and_world_entity` at `mod.rs:3284`) MUST stay green unchanged — proving the `plant_grown_tree` refactor preserved single-tree behavior.

### Step 2 — Mixed-leaf plans (trees + a building) + per-step receipt breakdown (M)

Generalize `Plan` to mix leaf kinds: parse `"add 3 trees and a building"` into a Plan whose steps are `PlantTree×3` + a new `PlantBuilding` leaf routed to `plant_forge_building` (`mod.rs:1079`), each step capturing its `PlacedAsset` entry into the same group. Receipt gains a per-kind breakdown.

**Done When:** `cargo test -p vox_app mixed_plan_groups_trees_and_building -- --nocapture` prints `OK: 4 entities, 1 undo` — asserting `entities.len() == before + 4`, that the receipt contains both `"3 Silver Birch"` and `"1"` building, `undo_stack.len() == 1`, and that one `edit.undo` restores `entities.len()` and `overlay.len()` to before.

### Step 3 — Visual proof in the snapshot binary + LLM-canned plan test (S)

Add a `--plant-plan "add 5 birch trees"` flag to `crates/vox_app/src/bin/shell_snapshot.rs` (mirroring the existing `--grow-tree` flag at `shell_snapshot.rs:117`) that calls `shell.run_intent` and prints the entity/overlay counts. Add an LLM-seam test using the existing `IntentBackend::canned` harness (`intent.rs:1046`): a canned `{"Plan":{...}}` JSON resolves through `resolve_via_llm`→`parse_llm_intent` to a validated 5-step Plan with `Provenance::Llm`, and a nested-Plan JSON is REJECTED (falls back to parser).

**Done When:** `cargo run -p vox_app --bin shell_snapshot -- --plant-plan "add 5 birch trees" --shot /tmp/plan.png` prints a line containing `5 things in the world` and writes a PNG with `>0%` non-background pixels; and `cargo test -p vox_app llm_plan_validates_and_rejects_nested -- --nocapture` prints `OK: plan validated, nested rejected`.

---

## 4. How it fits (integration + dependencies)

**Depends on (already-built, verified):**
- The Ask-Ochroma intent system: `parse_intent`/`resolve_intent`/`run_intent` and the `IntentBackend`/`Provenance` seam (`intent.rs`) — present and tested.
- The planting core `plant_asset`/`plant_grown_tree` and range-tracked `UndoEntry::PlacedAsset` (`mod.rs:1117`/`1050`/`120`) — present and tested.
- `FLORAPRIME_SPECIES` table + `grow_tree_skeleton`/`skeleton_to_splats` (`plugins.rs:502`/`565`/`656`) — present.
- `GaussianSplat::set_position`/`apply_transform` (`types.rs:181`/`198`) for per-step offsetting — present.

**No upstream gap is blocking.** This is a Phase-3 parallel-track item (roadmap §5: "Editor and AI run as parallel tracks that rejoin at Phase 3"); it can start tomorrow on the current code.

**What depends on it:**
- Gap #13 (real LLM backend): inherits the `Plan` contract — connecting a model gives "build me a village" → flat Plan for free, behind the *same* schema gate. This spec deliberately wires the LLM `Plan` DTO so #13 adds zero new safety surface.
- The "AI collaborator" half of the wedge (roadmap §1.4): single-step → multi-step is the qualitative jump from demo to product.

**Composes with existing systems:** the one-command-surface (every effect still routes through `plant_*`/`push_undo`/`registry`); the range-tracked undo (Group reuses `undo_one` over the existing `PlacedAsset` revert + range-shift logic); the receipt/Output-Log culture (one grouped receipt with real counts); the LLM validation gate (`SchemaContext` + `parse_llm_intent`).

**What it must NOT break:**
- **The 11-green-gate invariant:** the existing tree tests (`mod.rs:3246`,`3284`,`3322`) and ALL intent tests (`intent.rs:990`-`1402`) must stay green. The `plant_grown_tree` refactor is behavior-preserving (delegates to the new positioned core at the old fixed origin).
- **Both-config builds:** changes are pure `vox_app` shell logic with no new feature flags; `forge-native`/`crucible-native` configs are untouched (Step 2's `PlantBuilding` routes through the existing `plant_forge_building`, which already handles both backends).
- **The no-panic shell rule:** the count clamp (`1..=PLAN_MAX_STEPS`) and species fallback (bare "tree"→Silver Birch) mean every parse path yields a valid Plan or `Unknown` — never a panic, never an unbounded allocation.
- **The prompt-pin invariant:** `prompt_schema_pins_every_variant` (`intent.rs:1157`) stays green by updating all four pin points together — the design treats this as a required compile gate, not optional.
- **Engine-crate game-agnosticism:** zero engine-crate edits; planting trees is game content and stays in `vox_app`.

**Cross-gap seams:** the `Plan` contract is the seam to #13 (LLM); the positioned `plant_grown_tree_at` is reusable by #17 (duplicate/copy-paste — "re-planting range +offset" is the same primitive); the `Group` undo entry is the general grouped-undo mechanism #17 and Play-in-Editor (#9) snapshot/restore will also want.

**Sequencing:** Phase 3, AI track (roadmap §5). Startable immediately; rejoins the spine when #13 connects a model.

---

## Surprises & advantages

Grounding turned up four concrete, non-aspirational advantages — and two grounding corrections to the seed that make the work *safer*, not just cheaper:

1. **The LLM safety surface is already built and free to inherit.** `resolve_intent`/`parse_llm_intent`/`SchemaContext` (`intent.rs:340`/`593`/`433`) already validate every model-emitted action against the live schema, with `deny_unknown_fields` and clamp-at-the-edge. A `Plan` of validated leaves reuses this verbatim — multi-step planning gains NO new safety surface, and the "hostile value clamps" guarantee (proven at `intent.rs:1135`) extends to the count for free. This is the expensive part, and it is done.

2. **The prompt-pin test is a compiler-enforced safety net, not a chore.** `prompt_schema_pins_every_variant` (`intent.rs:1157`) makes it *impossible* to ship a `Plan` variant the validator doesn't gate or the prompt doesn't describe — the crate won't compile until all four pin points agree. A first-mover angle: the AI collaborator's action set can never silently drift from what the model is told, which is exactly the provability property the wedge sells.

3. **Range-tracked undo already solves the hardest grouped-undo subproblem.** The existing `PlacedAsset` undo (`mod.rs:956`) already removes an *exact* `[start,start+len)` range and shifts later entries down — the one genuinely tricky part of grouped undo (interleaved ranges) is already correct and tested. `Group` just needs to replay members reverse-order over it. The roadmap framed grouped undo as new machinery; in fact ~80% of it (the range bookkeeping) already exists.

4. **`plant_asset` is already kind-agnostic, so mixed plans (Step 2) are nearly free.** All four planters (`plant_grown_tree`/`plant_forge_terrain`/`plant_forge_building`/`plant_crucible_scene`, `mod.rs:1050`-`1106`) are thin wrappers over one `plant_asset` core that already returns a uniform `PlacedAsset`. Capturing the entry instead of pushing it works identically for trees, terrain, buildings, and scenes — "build me a village" of mixed assets is the same loop.

**Two grounding corrections that DE-RISK the seed** (the roadmap first-slice is a seed, not gospel — these are why the design diverges):
- The seed says reuse `IntentAction::AddNode`. **It does not work:** `AddNode` adds a *node-graph node* (`mod.rs:667`→`bridge.add_node_by_kind`), not a world entity. "Add 5 trees" must plant via `plant_grown_tree`, a path the parser never reaches today. Caught before it became a wrong implementation — the new `PlantTree` leaf is required.
- The seed's "row of N at stepped positions" hides a real bug: `plant_grown_tree` always plants at the single fixed `TREE_PLANT_ORIGIN` (`mod.rs:1055`), so N trees would *stack*. The design's `plant_grown_tree_at` + per-step offset (via the already-existing `GaussianSplat::set_position`) is what actually makes the Done-When's "distinct positions" true — and the parse-only test asserts the exact x-coordinates `[-4,0,4,8,12]`, catching a stack regression instantly.

---

## Verification corrections

The skeptic flagged `sound=false`. Surfaced honestly:

- **Why `sound=false`:** the copy of the markdown the skeptic was given ended mid-§2.3 (last visible line: `...(b) sets ShellEntity.pos = pos (mod.rs:91), and (c`). In that truncated view there was no Done When, no Capabilities/test table, no first implementation step, and no Wiring section — so the skeptic correctly judged it "not buildable as written" by the CLAUDE.md + design-template rules (which require an exact Done-When command and a real test assertion in the first step).
- **What this means for the spec:** this is a **truncation artifact, not a discovered design defect.** The full spec above DOES contain §3 with three ordered steps, each with an exact `cargo test`/`cargo run` Done-When and a real computed-outcome assertion (Step 1 asserts entity count, strictly-increasing x-coords, `undo_stack.len() == 1`, and exact restore-to-zero; the parse-only test asserts the literal `[-4,0,4,8,12]` x-coords; the clamp test asserts `PLAN_MAX_STEPS`), plus §4 with full Wiring/dependencies.
- **What the skeptic affirmatively verified (checkable in the truncated view):** the LLM safety surface (`resolve_intent`/`parse_llm_intent`/`SchemaContext` at `intent.rs:340`/`593`/`433`) is real and validates model actions with `deny_unknown_fields` + clamp-at-edge; `prompt_schema_pins_every_variant` (`intent.rs:1157`) is a real exhaustive-match compile gate; the planting/undo machinery refs check out. None of these failed.
- **Residual obligation on the implementer:** confirm the four prompt-pin points (§2.4) are updated in lockstep — the crate will not compile otherwise, which is the intended safety property, but it means Step 1 is not "add one enum variant" in isolation.
