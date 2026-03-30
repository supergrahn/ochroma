# Design: <Title> (YYYY-MM-DD)

**Status:** Draft | Approved
**Scope:** <One sentence — what problem this solves and what components are affected>
**Related:** <!-- links to related designs or plans, e.g. `[Domain 06 Rendering Plan](../plans/...)` -->

---

## 1. Problem Statement

<!-- 2-5 bullet points. Each must be a concrete, observable symptom — not a vague description.
     Bad: "rendering is slow"
     Good: "`cargo run` renders grass at 12 FPS on RTX 3080 due to per-splat CPU spectral conversion" -->

- <Specific symptom>
- <Specific symptom>

---

## 2. Done When

<!-- Hard, observable, specific. Must name exact command + exact human-visible output.
     "Tests pass" is never acceptable here.
     Bad: "spectral rendering works"
     Good: "`cargo run` renders grass as green (G channel > R and B) at ≥60 FPS shown in window title" -->

Running `<exact command>` produces `<exact observable output>`. A human at the keyboard can verify this without reading code.

---

## 3. Capabilities

<!-- One row per user-visible or system-visible capability this design adds or changes.
     "Real behavior test" must be a concrete assertion — write it now, not later.
     "Stub test (forbidden)" shows what a passing-but-wrong test looks like — name it so engineers avoid it. -->

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| <name> | `assert!(<real computed value> > <threshold>)` with real input data | `assert!(result.is_some())` — passes with empty stub |
| <name> | `cargo test <name> -- --nocapture` prints `<specific string>` | function exists, returns unit |

---

## 4. Architecture

<!-- Per subsystem: 1 paragraph describing what gets built, key decisions, data flow.
     Diagrams optional (ASCII or link). Cover threading model if relevant (e.g. "runs on render thread, no locks").
     Be specific enough that an engineer can implement it without asking questions. -->

### 4.1 <Component or Subsystem>

<!-- ~1-3 sentences for simple components; up to 1 paragraph for complex ones.
     Threading model: which thread owns this? Is it Send + Sync? Are there Arc<Mutex<>> boundaries? -->

### 4.2 <Component or Subsystem>

---

## 5. Data Models

<!-- Every struct, enum, or type introduced or significantly changed.
     Rules: private fields with accessor methods — no `pub` fields on types used across crates.
     Include size constraints or layout requirements if relevant (e.g. GPU buffers). -->

```rust
/// <one-line doc comment>
pub struct Foo {
    bar: u32,   // private — use .bar() accessor
    baz: [f32; 16],
}

impl Foo {
    pub fn bar(&self) -> u32 { self.bar }
    // ...
}
```

---

## 6. API

<!-- The public interface contract. Implementations must match exactly — no deviation.
     Include: method signatures, parameter types, return types, error types, threading constraints.
     If async: note executor requirements. If unsafe: document invariants.
     This section is the source of truth for the plan's "IMPORTANT NOTES" section. -->

```rust
// Example:
pub fn process(input: &InputType) -> Result<OutputType, Error>;
// Threading: call from any thread; internally uses rayon for parallelism.
// Panics: if input.len() == 0.
```

---

## 7. Wiring

<!-- For each new component: where exactly is it called from?
     "Will be wired later" is not acceptable — decide now.
     This table becomes the "Wiring requirement" field in every plan task. -->

| Component | Called from | File | Notes |
|---|---|---|---|
| `Foo::new()` | `EngineRuntime::init` | `crates/vox_core/src/engine_runtime.rs` | called once at startup |
| `Foo::process()` | `RenderLoop::tick` | `crates/vox_render/src/lib.rs` | called every frame |

---

## 8. Open Questions

<!-- Decisions not yet made. Each question should be answered before the plan is written.
     If a question remains open at plan-writing time, the plan must document the chosen answer.
     Delete this section when all questions are resolved. -->

- [ ] <Question that needs an answer before implementation begins>
- [ ] <Question>

---

## 9. Out of Scope

<!-- What is explicitly NOT addressed. Prevents scope creep.
     Be specific: "This design does not address multi-GPU" not "performance is out of scope". -->

- <Explicit non-goal>

---

## 10. Related Plans / Designs

<!-- Cross-references. Helps engineers understand dependencies and ordering. -->

- Depends on: <!-- `[Design: Foo](./YYYY-MM-DD-foo-design.md)` -->
- Required before: <!-- `[Domain N Plan](../plans/...)` -->
- Related: <!-- links -->
