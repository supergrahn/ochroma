# <Title> Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** <One sentence — what this plan achieves>
**Done When:** <Hard, observable criterion — exact command + exact human-visible output that proves the feature works end-to-end. "Tests pass" is never acceptable. Name the screen, log line, or file a human can point at.>
**Architecture:** <2-3 sentences — technical approach and key design decisions>
**Design Document:** `docs/specs/YYYY-MM-DD-<topic>-design.md`
**Tech Stack:** <e.g. Rust 1.87, wgpu 0.20, glam 0.29>
**Build:** `cargo build` / `cargo test` / `<any extra steps required>`

---

## IMPORTANT NOTES

<!-- Real API signatures and constraints that agents MUST follow exactly.
     Use this section to prevent agents from inventing their own APIs or using pseudo-code.
     Copy exact signatures from the design doc's API section.
     Agents that deviate from these signatures produce non-compilable code. -->

- `<TypeName>` fields are **private** — use `.<accessor>()` methods, never direct field access
- `<function_name>(input: &InputType) -> Result<OutputType, Error>` — exact signature, do not alter
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task
- <Any other hard constraints: threading model, lifetime rules, crate boundaries>

---

## File Map

<!-- Complete list of files touched by this plan. Agents use this to know exactly what to create/modify.
     No file should appear in a task that isn't listed here first. -->

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/foo/src/bar.rs` | <one-line purpose> |
| Modify | `crates/foo/src/lib.rs` | <what changes and why> |
| Test   | `crates/foo/tests/bar_test.rs` | <what it exercises> |

---

## Capabilities

<!-- One row per capability. "Real behavior test" is the acceptance criterion — write real assertions.
     "Stub test (forbidden)" shows what a passing-but-meaningless test looks like — name it explicitly. -->

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| <name> | `assert!(result.value > 0.5)` with real input | `assert!(result.is_some())` — passes with stub |
| <name> | `cargo run` shows `<specific string>` in stdout | function exists and returns `()` |

---

## Task 1: <Name — one capability, fully implemented and wired in this task>

<!-- Each task implements AND wires one capability. Never split "implement X" and "wire X" into separate tasks. -->

**Files:**
- Create: `crates/foo/src/bar.rs`
- Modify: `crates/foo/src/lib.rs`
- Test: `crates/foo/tests/bar_test.rs`

**Acceptance:** `cargo test -p foo bar_does_thing -- --nocapture` → output includes `result: <non-trivial real value>` (not zeroes, not empty, not `Some(())`).

**Wiring requirement:** Must be called from `<exact_function_name>` in `crates/foo/src/lib.rs` before this task is complete. `todo!()` / `unimplemented!()` / empty function bodies = **task failure**.

- [ ] **Step 1: Write the failing test** — test real behavior, not interface shape

```rust
#[test]
fn bar_does_thing() {
    let input = <real test data — not zeroes, not default>;
    let result = bar(input);
    // Assert a real computed outcome, not just that it ran
    assert!(result.value > 0.5, "expected > 0.5, got {}", result.value);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test -p foo bar_does_thing 2>&1 | tail -5
```

Expected: FAIL — `error[E0425]: cannot find function 'bar'` (or equivalent — the feature must not exist yet)

- [ ] **Step 3: Implement** — no `todo!()`, no stubs, full working logic

```rust
pub fn bar(input: InputType) -> OutputType {
    // Full implementation — every branch, every computation.
    // If you write todo!() here the task is not done.
}
```

- [ ] **Step 4: Wire at exact callsite**

In `crates/foo/src/lib.rs`, inside `<exact_function_name>`:

```rust
// Before:
// <nothing or old code>

// After:
let result = bar(input);
// The feature is now active in the real code path.
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p foo bar_does_thing -- --nocapture
```

Expected: PASS. Output must show a real computed value — if output is `0.0` or empty, the implementation is a stub.

- [ ] **Step 6: Commit**

```bash
git add crates/foo/src/bar.rs crates/foo/src/lib.rs crates/foo/tests/bar_test.rs
git commit -m "feat(foo): implement bar and wire into <exact_function_name>"
```

---

## Task 2: <Name>

**Files:**
- Modify: `crates/foo/src/baz.rs`

**Acceptance:** <exact `cargo test` or `cargo run` command> → <exact non-trivial output>

**Wiring requirement:** Must be called from `<exact_function>` in `<exact/file.rs>`. `todo!()` / `unimplemented!()` / empty bodies = **task failure**.

- [ ] **Step 1: Write the failing test**

```rust
// full test code — real input, real assertion on real output
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p foo <test_name> 2>&1 | tail -5
```

Expected: FAIL — <specific compiler or runtime error>

- [ ] **Step 3: Implement** — no stubs

```rust
// full implementation
```

- [ ] **Step 4: Wire at exact callsite**

```rust
// exact before/after in <exact_function> in <exact/file.rs>
```

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p foo <test_name> -- --nocapture
```

Expected: PASS, output: <specific real value — not zero, not empty>

- [ ] **Step 6: Commit**

```bash
git add <files>
git commit -m "feat(foo): <description>"
```

---

## Self-Review Checklist

<!-- Run this checklist yourself after writing the plan — before handing it to an agent.
     Fix any failures inline. Do not skip this. -->

- [ ] Every task implements AND wires in the same task — no "wire later" tasks exist
- [ ] Every `Acceptance` criterion names a real non-trivial expected output (not "tests pass", not zeroes)
- [ ] Every `Wiring requirement` names an exact function and exact file
- [ ] `IMPORTANT NOTES` contains the real API signatures from the design doc
- [ ] `File Map` lists every file that appears in any task
- [ ] No step contains `todo!()`, `unimplemented!()`, or stub bodies in the implementation code
- [ ] `Done When` names a specific command and specific human-observable result
- [ ] All types, method names, and signatures are consistent across all tasks (Task 3 calls `foo()`, not `do_foo()`)
