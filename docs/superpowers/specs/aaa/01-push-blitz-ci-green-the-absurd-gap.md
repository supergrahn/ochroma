> **Adversarial verification:** SOUND. The skeptic confirmed the spec is correct: every named manifest path, line reference, and the 158-commit count check out (`git rev-list --count origin/master..HEAD = 158`, `HEAD = f380cf1`). The only flagged issue is cosmetic and NOT in this spec — the upstream roadmap (line 99) says "157 commits unpushed" while the true count is 158; this spec already uses the correct number. No correction needed to the spec body.

## Push the blitz and make CI go green — the absurd gap

**Status:** Draft
**Dimension:** Stability, Platforms & Shipping (roadmap gap #4, score 71/90, effort S)
**Scope:** Take the 158-commit `blitz/day1-foundation` branch from "green only on the author's laptop" to "green on a machine that is not the laptop," by fixing the actual sibling-repo checkout gap the CI was written blind to, creating the access secrets, and observing one externally-reproduced green run.
**Related:** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #4), `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `FEATURES.md` (USD import row, line 70)

> Honesty preface: the roadmap's seed first-slice for this gap says "create a fine-grained read-only PAT for supergrahn/spectra+crucible, add it as SIBLING_REPOS_PAT, push, watch, fix the first real failure." Grounding the code revealed that **the seed is incomplete in exactly the way the gap predicts**: the workspace has a THIRD non-optional sibling path dependency — `openusd-rs` — that no CI job checks out. A spectra+crucible-only PAT produces a run that dies at `vox_usd`'s manifest load, not a green run. This spec corrects the seed and makes the real first failure the thing we fix.

---

## 1. What we need

After this exists, a developer who is not Tom — a CI runner, a reviewer, a future contributor on a fresh clone-plus-three-siblings — can do what is impossible today: **see the blitz pass its own gates without trusting the author's word.** Concretely observable outcomes:

- `gh run list --branch blitz/day1-foundation` shows **at least one CI run with `conclusion: success`** whose `headSha` equals local `git rev-parse HEAD` (today: `f380cf1`). This is the single Done-When for the whole gap.
- The `test` job's `Smoke walking_sim (headless)` and `Smoke engine_runner (headless)` steps both pass **on `ubuntu-latest`** — proving a machine other than the laptop ran the real sim + software-rendered-frame + pixel/state assertions (the provability wedge, externally reproduced).
- `cargo build --workspace` succeeds **on the runner** — which it cannot today, because `vox_usd` (a workspace member, `live` per FEATURES.md:70) hard-depends on `openusd-rs` at `../../../openusd-rs`, a sibling no CI job clones (`grep -c openusd .github/workflows/*.yml` → `0` in both files).
- The three private sibling repos (`spectra`, `crucible`, and the newly-pushed `openusd-rs`) are reachable from Actions via a **fine-grained read-only `SIBLING_REPOS_PAT`** with Contents:read scoped to exactly those repos — no broader token, no classic PAT.
- `blitz/day1-foundation` exists on `origin` (today it does not — `git ls-remote origin blitz/day1-foundation` is empty; origin/master is 158 commits and >2 months stale).

**Why blocking (Stability dimension):** the roadmap is explicit — "the entire provability culture ('11 consecutive green / adversarial waves') is self-attested LOCALLY... A green streak nobody but the author can reproduce is a claim, not provability — and provability IS the wedge." Every downstream gap (#2 GpuContext, #5 GPU relight, #19 Windows packaging) inherits its credibility from a green gate that has, per ci.yml's own comment, **never produced a green run** ("every CI run since March died at manifest-load"). This is the floor under the floor: until CI is green once, no later "headless-asserted" claim is externally true.

---

## 2. How it's gonna be (the design)

This is an infrastructure/shipping gap, not a code-kernel gap. There are no new Rust types and no GPU oracle to twin. The "architecture" is the **dependency-resolution topology** the CI must reproduce, and the **fix to make it reproducible.** Everything below is verified against the live manifests.

### 2.1 The real dependency topology (verified)

`cargo build --workspace` builds every member of the workspace in `Cargo.toml` (19 members including `vox_usd`, `vox_render`, `vox_nodes`). Cargo must **load the manifest of every member and every non-optional path dep before compiling anything** — this is why a missing sibling is a hard, early, total failure, not a feature-gated one. The external, non-engine path deps that the default `cargo build --workspace` must resolve:

```
ochroma/  (checked out at $GITHUB_WORKSPACE/ochroma)
  crates/vox_render/Cargo.toml:30   spectra-gaussian-render = path "../../../spectra/rust/spectra-gaussian-render"   [NON-OPTIONAL]
  crates/vox_render/Cargo.toml      default = ["crucible"]  ->  vox_nodes/crucible
  crates/vox_nodes/Cargo.toml:15,19 crucible-core / crucible-types = path "../../../crucible/rust/crates/..."          [opt, but ON via default]
  crates/vox_usd/Cargo.toml:12      openusd-rs = path "../../../openusd-rs"                                            [NON-OPTIONAL]  <-- THE GAP
  crates/vox_app/Cargo.toml:71-86   forge-* / cook / crucible-* = path "../../../aetherspectra | ../../../crucible"   [optional, OFF]
```

The path arithmetic: a crate at `$GITHUB_WORKSPACE/ochroma/crates/vox_render/` resolving `../../../spectra/...` lands at `$GITHUB_WORKSPACE/spectra/...`. So the CI's three-repo side-by-side checkout (ochroma at `path: ochroma`, spectra at `path: spectra`, crucible at `path: crucible`) is **correct arithmetic for spectra and crucible.** It is simply **missing the openusd checkout entirely.**

```
$GITHUB_WORKSPACE/
  ochroma/    <- actions/checkout path: ochroma
  spectra/    <- actions/checkout repository: supergrahn/spectra  (default branch = main; verified origin/HEAD -> origin/main)
  crucible/   <- actions/checkout repository: supergrahn/crucible
  openusd-rs/ <- MISSING TODAY. must add: repository: supergrahn/openusd-rs
```

### 2.2 Key design decisions and rationale

- **Push `openusd-rs` as a new private repo `supergrahn/openusd-rs`, not vendor it into ochroma.** Verified: its tracked tree is tiny (86 files, 48 KiB packed — the 707 MB on disk is 706 MB of gitignored `target/`), it is dual-licensed Apache/MIT (publishable), and it is self-contained (`grep path Cargo.toml` → only its own `genschema.rs` bin target; no external sibling path deps, no transitive openusd in crucible-core/types). Vendoring would fork a 64-source-file pure-Rust USD stack into the engine repo and break the documented "one sibling per concept" isolation the cargo-tree gate enforces (FEATURES.md:70, vox_usd/Cargo.toml:8-9). A read-only checkout mirrors how spectra/crucible already work.
- **Check out openusd-rs's *default branch*, not a pinned ref** — same as spectra/crucible today. Risk noted in §3 step 5: if the runner's openusd HEAD diverges from the local one the lockfile pins, the build can still fail; mitigation is to push the exact local openusd HEAD (`9fd19fa`) as the repo's default branch tip.
- **Check out spectra's `main` branch** (the default). Verified: `origin/main` contains `rust/spectra-gaussian-render`, `rust/spectra-gpu`, `rust/spectra-renderer`, `rust/spectra-scene-state`, and the local `vulkan-fallback-backend` working branch is 0-behind / 1-ahead of `origin/main`. The default build only compiles `spectra-gaussian-render` (non-optional); the vulkan-branch-only crates are behind `spectra-native`, which is OFF in CI. So `main` is sufficient for the `test` job. **Do not** flip CI to the vulkan branch — that would pull `spectra-native`-only code paths CI does not exercise.
- **Fine-grained PAT, read-only Contents, three repos.** The roadmap names a "fine-grained read-only PAT for spectra+crucible"; the verified gap forces it to be **three** repos (add openusd-rs). Scope = Contents:read on `supergrahn/{spectra,crucible,openusd-rs}` only. Fork PRs intentionally do not receive the secret (ci.yml:11-13 documents this; accepted because forks can't build private deps regardless).
- **Fix both `ci.yml` and `release.yml`** — the openusd-rs omission exists identically in both (verified `grep -c openusd` → 0/0). Leaving release.yml broken would mean the first `v0.1.0` tag dies at the same manifest load.

### 2.3 What does NOT change

No engine code. No `ShellRequest`/`plant_asset`/`push_undo` path is touched. No GPU oracle/twin. No numeric clamps. No data-model change. The engine-crate game-agnostic rule is untouched. The only ochroma-repo edits are the two YAML workflow files (adding one checkout step each + the openusd token line). The rest of the work is GitHub-side (secret, repo creation, branch push) and is verified by observation, not by editing the engine.

---

## 3. How it's gonna be made (the implementation plan)

Ordered. Each step implements AND lands its piece; the run is only watched after the workflow is actually correct, so the "first real failure" we hit is a genuine one, not the openusd gap we already diagnosed.

### Step 1 — Fix the CI workflows to check out openusd-rs (the verified first failure), launchable tomorrow. (S)

This is the FIRST agent task. It edits exactly two files in the ochroma repo and proves the fix with a local reproduction of the runner's checkout layout — no GitHub access required to validate the manifest now resolves.

- **Files:** `.github/workflows/ci.yml` (add an openusd checkout to all THREE jobs: `test` after line 53, `test-portability` after line 120, `build-web` after line 152) and `.github/workflows/release.yml` (add one after line 65). Each new step:
  ```yaml
  - uses: actions/checkout@v4
    with:
      repository: supergrahn/openusd-rs
      token: ${{ secrets.SIBLING_REPOS_PAT }}
      path: openusd-rs
  ```
  Also update the header comment block in ci.yml (lines 3-13) and release.yml (lines 4-23) to name **three** sibling repos, not two, and the REQUIRED SECRET line to list openusd-rs.
- **Headless proof (local, reproduces the runner's flat layout):** create a temp dir with symlinks mirroring `$GITHUB_WORKSPACE` — `ln -s` the real `~/src/{ochroma,spectra,crucible,openusd-rs}` into `/tmp/ci-repro/` — then run `cargo build --workspace --manifest-path /tmp/ci-repro/ochroma/Cargo.toml 2>&1 | tee /tmp/ci-repro/build.log`. (The symlink tree proves the `../../../` arithmetic resolves with openusd present and would fail without it.)
- **Done When:** `grep -c "repository: supergrahn/openusd-rs" .github/workflows/ci.yml` prints `3` and the same grep on `release.yml` prints `1`; AND running `grep -L openusd-rs <(git show HEAD:.github/workflows/ci.yml)` against the edited file confirms the string is now present (regression guard); AND a YAML-lint sanity check `python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('yaml-ok')" .github/workflows/ci.yml .github/workflows/release.yml` prints exactly `yaml-ok`. (We assert on the workflow content, not "tests pass" — the runner-side proof is Step 6.)

### Step 2 — Push `openusd-rs` to a new private `supergrahn/openusd-rs`, default branch at HEAD `9fd19fa`. (S)

- **Commands:** `gh repo create supergrahn/openusd-rs --private --source ~/src/openusd-rs --remote origin --push` (creates and pushes current branch). Then `gh repo edit supergrahn/openusd-rs --default-branch master`.
- **Done When:** `gh api repos/supergrahn/openusd-rs --jq '.default_branch + " " + .visibility'` prints `master private`, and `git ls-remote https://github.com/supergrahn/openusd-rs master | cut -f1` prints `9fd19fa...` matching `git -C ~/src/openusd-rs rev-parse HEAD`.

### Step 3 — Create the fine-grained `SIBLING_REPOS_PAT` and add it as a repo secret on ochroma. (S)

- **Action:** create a fine-grained PAT (GitHub settings UI or `gh`), resource owner = `supergrahn`, repository access = **only** `spectra`, `crucible`, `openusd-rs`, permission = Contents: Read-only. Add to ochroma: `gh secret set SIBLING_REPOS_PAT --repo supergrahn/ochroma --body "<pat>"`.
- **Done When:** `gh secret list --repo supergrahn/ochroma` lists `SIBLING_REPOS_PAT` with an `Updated` timestamp of today; AND a scope smoke check `GH_TOKEN=<pat> gh api repos/supergrahn/openusd-rs --jq .name` prints `openusd-rs` (token can read the private repo) while `GH_TOKEN=<pat> gh api repos/supergrahn/ochroma 2>&1 | grep -c "Not Found\|message"` is non-zero (token is correctly scoped OUT of ochroma — least privilege confirmed).

### Step 4 — Push the blitz branch and the two already-existing siblings' default branches. (S)

- **Commands:** `git push origin blitz/day1-foundation`. Confirm `supergrahn/spectra` default branch (`main`) and `supergrahn/crucible` (`master`) tips are reachable: `git ls-remote https://github.com/supergrahn/spectra main` and `.../crucible master` return SHAs. (If the local working copies are ahead of those tips for crates CI needs, push them; verified spectra `main` already has `spectra-gaussian-render`.)
- **Done When:** `gh api repos/supergrahn/ochroma/branches/blitz/day1-foundation --jq .commit.sha` prints a SHA equal to `git rev-parse HEAD` (`f380cf1...`).

### Step 5 — Watch the triggered run; fix the (now genuinely-first) real failure. (S–M)

The push to `blitz/**` triggers CI (ci.yml:17 matches `blitz/**`). With openusd checked out and the PAT scoped, the manifest-load wall is gone; the first real failure is whatever the compiler/clippy/test/smoke surfaces on `ubuntu-latest` — e.g. a platform-only warning under `RUSTFLAGS: -D warnings` (ci.yml:37), an ALSA/CPAL link issue, or a smoke-frame pixel assertion that is laptop-GPU-dependent. Triage one failure, land a fix commit on `blitz/day1-foundation`, re-push.

- **Commands:** `gh run watch $(gh run list --branch blitz/day1-foundation --limit 1 --json databaseId --jq '.[0].databaseId')`; on failure, `gh run view --log-failed` to read the exact failing step.
- **Done When:** `gh run view <id> --json jobs --jq '.jobs[] | select(.name | startswith("Test")) | .steps[] | select(.name | startswith("Smoke")) | .name + ": " + .conclusion'` prints both `Smoke walking_sim (headless): success` and `Smoke engine_runner (headless): success`. (Asserts the two real behavioral gates ran and passed on the runner, not merely "the job finished.")

### Step 6 — Confirm the externally-reproduced green run (the gap's Done-When). (S)

- **Command:** `gh run list --branch blitz/day1-foundation --json headSha,conclusion,workflowName --jq '.[] | select(.workflowName=="CI" and .conclusion=="success") | .headSha'`
- **Done When:** the command prints at least one SHA, and that SHA equals `git rev-parse HEAD`. A reviewer can open the run in the browser and see the green check on the blitz HEAD — provability moved from the laptop to GitHub's runners.

---

## 4. How it fits (integration + dependencies)

**Depends on:** nothing in the codebase — this is Phase 1 and the roadmap names it as independently startable ("turns the wedge from self-attested to verifiable — costs a secret + a push"). It depends only on GitHub access (org `supergrahn`, ability to create a private repo and a fine-grained PAT). The one *discovered* dependency is **openusd-rs being pushable**, which §2.2 verified it is.

**Depended on by:** every later gap's credibility. #2 (GpuContext), #3 (tiled rasterizer), #5 (GPU relight twin), #7 (GPU timestamps), #18 (soak), #19 (Windows build), #28 (crash-on-save) all assert "headless-proven" Done-Whens whose external truth requires a green gate. #19 in particular *promotes* the currently-`continue-on-error` `test-portability` (win/mac) job to gating — that promotion is meaningless until the `test` job is green first. This spec unblocks the entire "externally provable" half of the wedge.

**Composes with existing systems:** the three-repo side-by-side checkout pattern (ci.yml + release.yml, already designed and commented), the two headless `--smoke` gates (`walking_sim.rs:3130`, `engine_runner.rs:3589` — both verified to parse `--smoke`), the `-D warnings` + clippy ratchet, and the WASM `web/build.sh` bundle check. It adds one checkout step per job and one secret; it changes no gate logic.

**What it must NOT break:**
- *The 11-green-gate / smoke invariant:* this spec does not touch any smoke assertion; it makes them run somewhere new. If Step 5 surfaces a laptop-GPU-dependent pixel assertion, the fix must keep the assertion meaningful (e.g. tolerance widened with justification), never deleted — deleting a behavioral assert to go green would forfeit the exact provability this gap exists to establish.
- *Both-config builds:* the default-features `test` job and the `--no-default-features` `test-portability` job must both still resolve manifests. The openusd checkout helps both (vox_usd is a member in both configs). Do not gate the openusd checkout behind a feature.
- *The no-panic shell rule:* untouched — no shell/editor code changes.
- *Least privilege:* the PAT must be fine-grained and scoped to three repos read-only (Step 3's negative assertion enforces it). A classic or broadly-scoped token would be a security regression even if it makes CI green.

**4-phase sequencing:** Phase 1, first item on the critical-path spine (`#4 (verifiable) → #2 (one device) → ...`). It is the cheapest gap (effort S) and the highest-leverage-per-hour: a secret, a repo push, two YAML edits, one triage. Cross-gap seam: once green, #19 (Windows/Steam) extends the same workflow file, and #18 (soak) adds a job to the same `test` matrix — both inherit the now-working sibling checkout.

---

## Surprises & advantages

Grounding surfaced four concrete, non-aspirational advantages:

1. **The roadmap seed was wrong in the gap's own favor — and we caught it before burning a run.** The seed says "spectra+crucible PAT." The verified manifests show a THIRD non-optional sibling, `openusd-rs` (`vox_usd/Cargo.toml:12`, `live` per FEATURES.md:70), that no CI job clones (`grep -c openusd .github/workflows/*.yml` → 0/0). Following the seed literally would have produced a red run dying at `vox_usd`'s manifest load — the exact "died at manifest-load" failure ci.yml's comment says has happened since March. The fix is now a *diagnosed* edit (Step 1), not a blind watch-and-guess. This is the gap auditing itself.

2. **openusd-rs is trivially pushable — the 707 MB is a paper tiger.** On-disk it looks like a 707 MB monster, but 706 MB is gitignored `target/`; the tracked tree is 86 files / 48 KiB packed, dual-licensed Apache+MIT, with zero external sibling path deps (only its own `genschema.rs` bin). It even ships its own self-contained `.github/workflows/ci.yml`. So the "missing third repo" — which sounds like it could be a blocker — is a five-minute `gh repo create --push`.

3. **The default `test` job needs almost none of the heavy sibling surface.** With `spectra-native` OFF and the forge/cook path deps `optional` and OFF, the default build only compiles **one** spectra crate (`spectra-gaussian-render`, present on `origin/main`) plus `crucible-core`/`crucible-types` plus `openusd-rs`. The CUDA/Vulkan/Slang-heavy `spectra-renderer`/`spectra-gpu`/`spectra-scene-state` and the whole `aetherspectra/forge` tree are **never touched by CI** — so the runner needs no GPU, no CUDA toolkit, no Slang. The green gate is far cheaper to stand up than the engine's full sibling graph suggests.

4. **The branch tip is already 0-behind origin/main on spectra.** The local spectra working branch (`vulkan-fallback-backend`) is 1-ahead / 0-behind `origin/main`, and `main` already contains `spectra-gaussian-render`. So checking out spectra's default branch in CI "just works" for the default build — no branch coordination, no force-push, no pinning dance. One fewer moving part than the gap's framing implied.
