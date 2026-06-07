> **Adversarial verification:** `sound=false` — the skeptic found a **load-bearing false premise** in the integration story. The spec asserts "the shell already runs rhai via script_gen compile-verification" and proposes registering `metamer_divergence`/`set_illuminant` "on the shell's own rhai engine instance." Verification of `EditorShell` (`vox_app/src/shell/mod.rs:182-276`) found **no rhai `Engine` field and no Rhai runtime owned by the shell** — the shell's `script_gen` path generates and compile-checks rhai but does not hold a live `Engine` to register host functions on. The rhai binding plan (§2b, Step 3) must be re-grounded against where rhai actually lives. See **Verification corrections**. The metamer-kernel half of the spec (Steps 1–2, the `vox_render` math + the editor illuminant-swap composite) is unaffected and verified.

## Status

Draft — production spec. Effort: **M**. Roadmap rank: **#5 (Wedge, GPU runtime spectral-relight kernel) made script- and editor-reachable**, with the metamer query as its first observable.

**Related (verified to exist):** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #5, #11, #13, #34), `docs/superpowers/specs/2026-06-07-spectral-relight-design.md` (the relight kernel + Appendix B metamer proof), `FEATURES.md`.

**The wedge fact this spec turns into a mechanic (verified):** `GaussianSplat` (`vox_core/src/types.rs:29`) stores `spectral: [u16; 16]` = baked f16 **radiance**, not reflectance. `relight_scene` (`vox_render/src/relight.rs:466`) recovers an intrinsic base (`baked ÷ reference SPD`) and re-multiplies by a target SPD per band. The metamer divergence — two surfaces RGB-identical under one light, divergent under another — is *already proven numerically* by `relight_breaks_metamers` (`relight.rs:869`): neutral sRGB distance `< 0.012`, `cool_led` sRGB distance `> 0.03`. **Today that physics is reachable only from `cargo test` and the offline `vox_tools relight` CLI. It is reachable from zero scripts and zero editor surfaces.** This spec exposes it to rhai and to a one-key editor demo.

---

## 1. What we need

The roadmap's Wedge dimension is binding: "Spectral relighting / metamerism as a frame-rate game mechanic… the unforgeable moat; everything else is in service of it" (roadmap §1.1), and "The wedge mechanics are currently asset-time, not runtime — making them runtime is the strategic spine" (roadmap §1, synthesis). The decisive blocking fact: a game *designer* (the target audience — domain-knowledgeable non-game-devs) cannot today write a single line that asks "are these two surfaces a forgery — identical under the gallery light, divergent under the inspection lamp?", and cannot flip a scene's illuminant from the editor and *see the seam move*. Concretely, after this exists:

- **A rhai script can call `metamer_divergence(entity_a, entity_b, "tungsten", "daylight") -> f32`** and branch on the result (e.g. "if divergence under the UV lamp > 0.03, the painting is a fake → trigger the reveal"). This is the *mechanic*: the physics drives game logic, not just a render. Observable: a rhai `if metamer_divergence(...) > 0.03 { log("forgery!") }` emits the log line headlessly.
- **A rhai script can call `scene.set_illuminant("cool_led")`** and the next composited frame re-illuminates the overlay through the same kernel `relight_scene` uses — the AAA bar is *runtime* relight, not an offline re-bake to disk. Observable: the editor's planted overlay visibly shifts hue when the illuminant changes, pixel-asserted.
- **A one-key editor "forgery" demo exists**: two surfaces placed via the existing `plant_asset` path, identical under `daylight`, divergent under `cool_led`, switched live by a key, with a HUD receipt line reading the live sRGB distance. The AAA bar: a vertical slice a non-engineer can *watch work* and a smoke gate can *prove*.
- **The HUD shows a live, computed receipt** — `daylight: ΔsRGB 0.009 (metamer) · cool_led: ΔsRGB 0.041 (forgery)` — never a hardcoded string. This extends the engine's provability culture ("headless pixel/state-asserted", roadmap §1.3) from `cargo test` to a visible, screenshot-able product surface.
- **Engine crates stay game-agnostic.** "forgery"/"painting"/"gallery" are *game* concepts and live only in `vox_app`; `vox_render` and `vox_core` gain only the generic `metamer_divergence` / illuminant-swap primitives (CLAUDE.md architecture rule).

Why blocking: without a script/editor seam, the wedge is a `cargo test` artifact — provable but unplayable. The roadmap puts #5 on the critical-path spine ("#5/#34 relight playable"); this spec is the *reachability* half of #5 — the part that turns a verified kernel into something a designer's script and a player's keypress can touch. It is the smallest vertical that makes "metamerism as a mechanic" literally true.

---

## 2. How it's gonna be (the design)

Three thin seams over the **already-verified** `vox_render::relight` kernel. No new physics — the kernel, the SPD presets, and the metamer proof all exist. We expose them.

### 2a. The metamer primitive (NEW, `vox_render`, game-agnostic)

A pure function in `crates/vox_render/src/relight.rs`, sitting directly on the existing `forward_rgb` + `IlluminantSpec` machinery:

```rust
// NEW in vox_render/src/relight.rs — generic, no game concepts.
/// sRGB L2 distance between two splat *groups* re-illuminated under one illuminant,
/// computed through the SHIPPED render path (`derive_intrinsic` → `forward_rgb`).
/// Each group is averaged to one intrinsic base first (a "surface"), so the result
/// is the observable color difference a camera under `illum` would record.
/// Pure, never panics; empty group → its base is zeros.
pub fn srgb_distance_under(
    group_a: &[GaussianSplat],
    group_b: &[GaussianSplat],
    reference: &IlluminantSpec,   // what the splats were baked under
    illum: &IlluminantSpec,       // the light to observe them under
) -> f32 { /* mean-intrinsic A,B via derive_intrinsic; forward_rgb(base, illum.spd()); L2 of sRGB */ }

/// The mechanic query: how much do two surfaces diverge between two lights?
/// Returns the sRGB distance UNDER `illum_b` (the "reveal" light). A metamer pair
/// reads ~0 under `illum_a` and large under `illum_b`.
pub fn metamer_divergence(
    group_a: &[GaussianSplat],
    group_b: &[GaussianSplat],
    reference: &IlluminantSpec,
    illum_a: &IlluminantSpec,
    illum_b: &IlluminantSpec,
) -> f32 { srgb_distance_under(group_a, group_b, reference, illum_b) }
```

**Rationale — why this exact shape:** `relight_breaks_metamers` (`relight.rs:869`) already computes precisely this with `forward_rgb(&base, &cool)` minus `forward_rgb(&base, &neutral)` and asserts the 0.012/0.03 bounds. We are *promoting the test's body to a public function* and feeding it splat groups instead of hand-built bases. `forward_rgb` (`spectral_capture.rs:207`) is the shipped, render-consistent observer (CIE 1931 + sRGB primaries, white-balanced against the illuminant). Using it means the script's answer matches what the viewport draws. `IlluminantSpec::parse` (`relight.rs:105`) already maps `"tungsten"|"daylight"|"cool_led"|"neutral"|"d65"|…|"sun@<hour>"` → SPD, so the string args the roadmap's seed names resolve through existing code.

> **Correction to the roadmap seed (per the directive to verify, not trust):** the seed signature was `metamer_divergence(entity_a, entity_b, tungsten, daylight)`. Grounding shows two needed refinements. (1) A *reference* illuminant is required — `derive_intrinsic` divides baked radiance by the assumed capture SPD (`relight.rs:349`); without it the divergence is ill-posed. (2) The divergence is meaningful *between two observation lights*, so the honest primitive takes `(reference, illum_a, illum_b)`. The rhai binding still presents the simple `metamer_divergence(a, b, light1, light2)` face the designer wants (reference defaults to the scene's baked illuminant); the engine function is the honest 5-arg form.

### 2b. The rhai bindings (NEW, `vox_app` — game layer holds the World)

The rhai runtime (`vox_script/src/rhai_runtime.rs`) is **command-emitting, not World-bound**: scripts call `register_fn`-registered closures that push `ScriptCommand`s onto a global `PENDING_COMMANDS` mutex; there is *no* entity/World handle inside a script today (verified — `set_position` just queues a command). Binding `metamer_divergence` therefore happens **where the splat groups live: the editor shell in `vox_app`**, not in `vox_script` (which would drag game/world state into a near-engine crate). Two new registrations on the shell's own rhai engine instance (the shell already runs rhai via `script_gen` compile-verification):

```rust
// NEW: vox_app shell owns a rhai Engine for the demo; entity name → overlay range.
// metamer_divergence("painting_a", "painting_b", "daylight", "cool_led") -> f64
engine.register_fn("metamer_divergence", move |a: String, b: String, l1: String, l2: String| -> f64 {
    // resolve names → overlay splat ranges via the shell's entity table; clamp unknown → 0.0
    // call vox_render::relight::metamer_divergence(...); reference = scene illuminant
});
// scene.set_illuminant("cool_led") -> bool  (false on unparseable name; never panics)
engine.register_fn("set_illuminant", move |name: String| -> bool { /* sets shell.active_illuminant via IlluminantSpec::parse */ });
```

> **NOTE (added in synthesis):** the verifier found the premise behind this subsection — that the shell "already runs rhai via script_gen compile-verification" and owns an `Engine` to register on — is **false**. See **Verification corrections** for the correct binding home. The *capability shape* (a designer-facing `metamer_divergence`/`set_illuminant` resolving names → overlay ranges → the verified `vox_render` math) is sound; only the claimed host for the registration is wrong and must be re-grounded before Step 3 is launched.

**Data flow:**
```
rhai script  "metamer_divergence(a,b,'daylight','cool_led')"
   │  (register_fn closure, vox_app)
   ▼
resolve entity names → overlay GaussianSplat ranges  (shell.entities + UndoEntry::PlacedAsset start/len)
   │
   ▼
vox_render::relight::metamer_divergence(group_a, group_b, ref, l1, l2)   ← VERIFIED kernel path
   │  derive_intrinsic → forward_rgb → sRGB L2
   ▼
f64 back to script   ·   parallel: scene.set_illuminant("cool_led")
                                       │ sets shell.active_illuminant
                                       ▼
                          viewport recomposite: relight_scene(overlay, ref→active) → cpu_render texture
                                       │
                                       ▼
                          HUD receipt: live ΔsRGB(daylight) vs ΔsRGB(active)
```

### 2c. The editor "forgery" demo + HUD (NEW, `vox_app` — game layer)

A new `ShellRequest::SetIlluminant(IlluminantSpec)` variant (mirrors the existing `ShellRequest::GrowTree`/`ForgeTerrain` pattern exactly — `shell/mod.rs:148`) and a `shell.active_illuminant: IlluminantSpec` field (default `Daylight`). When it changes, the shell invalidates `viewport_tex` (the established cache-invalidation, `shell/mod.rs:1140`) and the next composite runs the overlay through `relight_scene(overlay, RelightSettings::new(reference, active))` before handing splats to `cpu_render`. The demo seeds two surface groups through the **existing `plant_asset` core** (`shell/mod.rs:1117`) so they are range-tracked and undoable like every other planted asset — no special path. The HUD line is appended to the existing `status_bar` (`shell/mod.rs:1329`), computed each frame by calling `srgb_distance_under` on the two demo groups under both lights — **never a literal string**.

**Where each piece lives, and why:**
- `srgb_distance_under` / `metamer_divergence` → `vox_render::relight` (generic spectral math; no game words; mirrors the file already housing `relight_scene` and the metamer test).
- rhai `metamer_divergence` / `set_illuminant` bindings, `ShellRequest::SetIlluminant`, `active_illuminant`, the demo seeder, the HUD line → `vox_app` (game layer owns the World, the entity table, and the "forgery" framing; CLAUDE.md rule).
- No change to `vox_script` (it stays command-emitting; the World-aware binding is correctly a game-layer concern).

**Engine-pattern compliance:** every numeric/string input clamps (unknown entity name → 0.0; unparseable illuminant → `set_illuminant` returns false, no state change); editor mutation goes through `ShellRequest` + `plant_asset` + `push_undo`; the relight stays on the verified CPU kernel for this slice (GPU twin is gap #5's separate scope — see §4). No `todo!()`, no empty bodies; every function returns a real computed value.

---

## 3. How it's gonna be made (the implementation plan)

Ordered. Each step implements **and** wires. Done-Whens name an exact command and an exact observable.

### Step 1 — `srgb_distance_under` + `metamer_divergence` in `vox_render` (S) — **launchable tomorrow**

**Files:** `crates/vox_render/src/relight.rs` (add the two `pub fn`s above the `#[cfg(test)]` module, ~line 683; add a test module section).

**Implement:** `srgb_distance_under(group_a, group_b, reference, illum)`:
1. Mean-intrinsic of each group: for each splat, `read_radiance` (`relight.rs:410`) → `derive_intrinsic(&radiance, &reference.spd(), 1e-3)` (`relight.rs:349`); average across the group into one `[f32; 16]` base. Empty group → `[0.0; 16]`.
2. `let rgb_a = forward_rgb(&base_a, &LightSpd(illum.spd()))` and same for B (`spectral_capture.rs:207`). (Wrap `illum.spd()` in `LightSpd` — verified constructor `LightSpd(pub [f32;16])`, `spectral_capture.rs:14`.)
3. Return `((0..3).map(|c| (rgb_a[c]-rgb_b[c]).powi(2)).sum::<f32>()).sqrt()`.
`metamer_divergence` = `srgb_distance_under(a, b, reference, illum_b)`.

**Wire:** re-express the existing `relight_breaks_metamers` test's final assertions through the new functions (so the function is exercised by the shipped proof, not a parallel one).

**Done-When (exact command + exact observable):**
`cargo test -p vox_render relight::tests::metamer_divergence_matches_forward_rgb -- --nocapture`
prints two computed numbers and the test body asserts (real computed outcomes, not `is_some`):
- Build the two splat groups from `metamer_pair()` (`relight.rs:828`) baked under `neutral`. Assert `srgb_distance_under(&a, &b, &neutral_spec, &neutral_spec) < 0.012` AND `metamer_divergence(&a, &b, &neutral_spec, &neutral_spec, &cool_led_spec) > 0.03`. Print line must read e.g. `metamer_divergence: daylight/neutral ΔsRGB 0.0071, cool_led ΔsRGB 0.0414`. (The numbers come from `metamer_pair`'s search — already known to land in [0.012, 0.03] bracket per `relight.rs:882`/`899`.)

### Step 2 — `ShellRequest::SetIlluminant` + `active_illuminant` + relit composite (S/M)

**Files:** `crates/vox_app/src/shell/mod.rs` (new `ShellRequest` variant ~line 178; new `active_illuminant: IlluminantSpec` field + default `Daylight`; drain arm ~line 1039; composite hook where `viewport_tex` rebuilds from `overlay`). `crates/vox_app/src/shell/cpu_render.rs` if the overlay→texture path needs the relit splats threaded in.

**Implement + wire:** drain arm sets `self.active_illuminant`, sets `self.viewport_tex = None` (existing invalidation, `mod.rs:1140`). The texture rebuild runs `relight_scene(&self.overlay, &RelightSettings::new(self.reference.clone(), self.active_illuminant.clone()).with_sky_ambient(false).with_shadows(false))` and composites the `.0` result. Reference illuminant default = `Daylight` (scene baked under daylight for the demo).

**Done-When:**
`cargo test -p vox_app shell::tests::set_illuminant_shifts_overlay_radiance -- --nocapture`
plants a known grey-under-daylight group via `plant_asset`, captures the mean band-4/band-14 ratio of the composited overlay, issues `ShellRequest::SetIlluminant(cool_led)`, drains, recomposites, and asserts the band-4/band-14 ratio **rose by > 0.2** (cool_led is blue-heavy: `cool_led[4]/cool_led[14]=1.00/0.30≈3.3` vs `daylight[4]/[14]=0.91/0.95≈0.96`, per `spectral_capture.rs:24,34`). Print both ratios. (Computed outcome; would fail if relight were a no-op.)

### Step 3 — rhai `metamer_divergence` + `set_illuminant` bindings (M)

> **Re-ground before launch (synthesis note):** the verifier found the shell does NOT own a rhai `Engine`. This step's host must be relocated to where rhai actually lives — see **Verification corrections** for the corrected plan. The Done-When (a script branching on `metamer_divergence` to return `"forgery"`) is unchanged; only the registration site moves.

**Files (as originally written, pending re-grounding):** `crates/vox_app/src/shell/mod.rs` (a `register_metamer_api(&mut Engine, …)` that the shell calls when building its rhai engine; entity-name → overlay-range resolver using `self.entities` + the `UndoEntry::PlacedAsset { start, len }` provenance).

**Implement + wire:** `metamer_divergence(a, b, l1, l2)` resolves names → ranges → `&overlay[start..start+len]` slices → `vox_render::relight::metamer_divergence`. Unknown name → `0.0`. `set_illuminant(name)` → `IlluminantSpec::parse(&name)` → push `ShellRequest::SetIlluminant`; unparseable → return `false`. Both registered on the shell's engine; the existing `RhaiRuntime::eval` path (`rhai_runtime.rs:307`) or the shell's own engine evaluates the script.

**Done-When:**
`cargo test -p vox_app shell::tests::rhai_metamer_query_drives_logic -- --nocapture`
plants the daylight-metamer pair as `"a"`/`"b"`, evaluates the script
`let d = metamer_divergence("a","b","daylight","cool_led"); if d > 0.03 { "forgery" } else { "genuine" }`
and asserts the returned string is exactly `"forgery"` AND the raw `d` (read via a second eval) is `> 0.03` and `< 0.5`. Also asserts `metamer_divergence("a","b","daylight","daylight") < 0.012`. (Real branch on a real number.)

### Step 4 — The "forgery" demo seeder + HUD receipt (M)

**Files:** `crates/vox_app/src/shell/mod.rs` (a `seed_forgery_demo()` that plants two metamer groups via `plant_asset` with labels `"Canvas A"`/`"Canvas B"`; HUD line in `status_bar`, `mod.rs:1329`). `crates/vox_app/src/bin/ochroma_editor.rs` (a `--demo forgery` flag and a key handler that cycles `active_illuminant` daylight↔cool_led).

**Implement + wire:** HUD computes, every frame, `srgb_distance_under(canvas_a, canvas_b, daylight, daylight)` and `…(…, active)` and renders `daylight: ΔsRGB {:.3} (metamer) · {active}: ΔsRGB {:.3} ({verdict})` where verdict = `forgery` if `> 0.03` else `metamer`. The keypress pushes `ShellRequest::SetIlluminant`.

**Done-When:**
`cargo run -p vox_app --bin ochroma_editor -- --demo forgery --illuminant daylight --frames 2 --shot /tmp/forgery_day.png` then `--illuminant cool_led --shot /tmp/forgery_led.png`, and a headless test
`cargo test -p vox_app shell::tests::forgery_demo_hud_receipt -- --nocapture`
that seeds the demo, reads the HUD-receipt string the shell computes (exposed via a `fn hud_receipt(&self) -> String`), and asserts: under daylight the string contains `(metamer)` and a parsed daylight ΔsRGB `< 0.012`; after `SetIlluminant(cool_led)` it contains `(forgery)` and a parsed cool_led ΔsRGB `> 0.03`. The two PNGs differ in mean hue (assert via `cpu_render::non_background_fraction`/mean-channel comparison that the cool_led shot is bluer). (The exact 0.012/0.03 receipt is the roadmap's literal Done-When.)

---

## 4. How it fits (integration + dependencies)

**Depends on (verified-present, no new prerequisites for this slice):**
- `vox_render::relight::{relight_scene, RelightSettings, IlluminantSpec, derive_intrinsic, forward_band, reilluminate_one}` — the whole kernel (`relight.rs`). **Reused, not rebuilt.**
- `vox_data::spectral_capture::{forward_rgb, LightSpd}` (`spectral_capture.rs:207,14`) — the render-consistent observer.
- `vox_core::types::GaussianSplat::{spectral, spectral_f32, spectral_mut}` (`types.rs:142–160`).
- Editor seams: `ShellRequest` (`mod.rs:148`), `plant_asset` (`mod.rs:1117`), `push_undo`/`UndoEntry::PlacedAsset` (`mod.rs:731`/`1141`), `status_bar` (`mod.rs:1329`), `viewport_tex` invalidation (`mod.rs:1140`), `cpu_render` (`shell/cpu_render.rs`), the `ochroma_editor --frames/--shot` headless harness (`bin/ochroma_editor.rs:54`).
- rhai host: `register_fn` registration discipline (`rhai_runtime.rs:64`) — though the *World-aware* bindings live in `vox_app`, not `vox_script`. **(See Verification corrections — the shell does not currently own the rhai Engine the original plan assumed.)**

**Depended on by / composes with (named roadmap gaps):**
- **#5 GPU runtime spectral-relight kernel** (Wedge, L): this spec is the *reachability* half. The seam here (`active_illuminant` → composite) is exactly where a future `OCHROMA_RELIGHT=gpu` WGSL twin plugs in — Step 2's `relight_scene` call becomes a `GpuGi`-style resident pass on the shared device. We deliberately stay on the CPU oracle now so the GPU twin (gap #5) has a *bit-exact reference already wired into a product surface*.
- **#34 Reflectance/emission data-model split** (Wedge, XL): when `GaussianSplat` gains a parallel `reflectance:[u16;16]`, `srgb_distance_under` drops its `derive_intrinsic` division and reads reflectance directly — the divergence becomes physically exact. This spec is forward-compatible: the reference-illuminant arg is the seam where that swap lands.
- **#11 Script host API** (Engine API, L): the `metamer_divergence`/`set_illuminant` bindings are the *first World-aware rhai functions* — they motivate the `trait ScriptHost` over World that #11 generalizes. Today scripts only emit `ScriptCommand`s; this is the first script→World *query*.
- **#13 Runtime AI content generation** (Wedge, M): "build me a forgery puzzle" becomes a generatable intent once the primitive exists — the AI emits a script calling `metamer_divergence`.

**Must NOT break:**
- **The green-gate invariant.** All new tests are headless, state/pixel-asserted, computed-outcome (no `assert!(x.is_some())`). New code adds to the streak; it does not gate behind GPU presence (the CPU path always runs).
- **Both-config builds.** `vox_render` change is a pure addition; `vox_app` additions follow existing `ShellRequest`/`plant_asset` patterns. No new required features.
- **The no-panic shell rule.** Every binding clamps: unknown entity → `0.0`, unparseable illuminant → `set_illuminant` returns `false` with no state change, empty group → zeroed base. `relight_scene` is already documented "Never panics" (`relight.rs:464`).
- **Engine game-agnosticism.** `vox_render` gains only generic spectral functions; all "forgery"/"canvas" naming is confined to `vox_app`.

**Sequencing:** Phase 3 ("the wedge made playable"). It is the lightest entry into the Phase-3 spine — it needs nothing from Phase 2's GPU loop (it rides the CPU oracle), so it can land *in parallel* with #2/#7/#3 and be waiting, wired, when #5's GPU twin arrives. Cross-gap seam: the `active_illuminant` field is the single integration point #5 (GPU kernel), #34 (data model), and #13 (AI generation) all attach to.

---

## Surprises & advantages

Grounded discoveries that make this cheaper / stronger than the roadmap seed implied:

1. **The metamer proof already exists as runnable, asserted code — not a claim.** `relight_breaks_metamers` (`relight.rs:869`) and `metamer_pair()` (`relight.rs:828`) already compute the exact 0.012/0.03 sRGB bounds the Done-When demands, via `forward_rgb`. Step 1 is literally *promoting a test body to a public function* and pointing it at splat groups. The hard part (constructing a real metamer pair that diverges under cool_led, verified against XYZ) is **done and committed**. This collapses the kernel-math risk of the whole spec to near zero.

2. **`metamer_pair()` is a ready-made demo asset generator.** The forgery demo's two surfaces don't need authoring — `metamer_pair()` already searches for a sharp single/double-band metamer pair that is invisible under neutral and maximally divergent under cool_led (`relight.rs:828–867`). The "two canvases" are a function call away; the demo seeder reuses it directly.

3. **The illuminant string vocabulary the seed wanted already parses.** `IlluminantSpec::parse` (`relight.rs:105`) already accepts `"tungsten"|"daylight"|"cool_led"|"neutral"|"d65"|"d50"|"a"|"f11"|"sun@<hour>"`. The rhai string args resolve through shipped code — no new parser, and the designer gets CIE references and physical-sun illuminants *for free* beyond the seed's two presets.

4. **The CPU-oracle-first choice is a strategic gift, not a compromise.** By wiring the *CPU* `relight_scene` into a product surface now, gap #5's GPU twin inherits a bit-exact reference that is *already validated inside the editor frame loop and a smoke gate* — exactly the "GPU mirrors a CPU oracle, validates on RADV" pattern the engine enforces (`GpuGi` mirrors CPU GI, `spectral_gi.rs:582`). We're building the oracle's harness ahead of the GPU work, which de-risks #5.

5. **First-mover framing is real and unforgeable here.** The HUD receipt `daylight: ΔsRGB 0.009 (metamer) · cool_led: ΔsRGB 0.041 (forgery)` is a screenshot that *no RGB engine can produce a non-zero second number for* — an RGB pipeline that stored one triple at capture yields exactly `0.000` under any relight (`relight.rs:897` comment, design Appendix B). The demo *is* the competitive proof, and it's a one-key vertical slice a non-engineer can watch.

6. **rhai binding lands in the right crate by accident of the existing design.** Because `vox_script`'s rhai runtime is command-emitting with *no World handle* (`rhai_runtime.rs` — `set_position` just queues a `ScriptCommand`), the World-aware `metamer_divergence` *must* live in `vox_app` where the entity table and overlay are. The architecture's game-agnosticism rule is satisfied not by discipline but by where the data already sits — zero friction.

---

## Verification corrections

The skeptic flagged `sound=false` on a load-bearing false premise. Surfaced honestly, not silently fixed:

- **The false premise:** §2b and Step 3 claim "the shell already runs rhai via `script_gen` compile-verification" and propose registering host functions "on the shell's own rhai engine instance." Verification of `EditorShell` (`vox_app/src/shell/mod.rs:182-276`) found **no rhai `Engine` field and no Rhai runtime owned by the shell**. The shell's `script_gen` path generates rhai *source* and compile-checks it, but it does not hold a live `rhai::Engine` on which `register_fn` could attach a World-aware `metamer_divergence`. There is no `engine.register_fn(...)` call site in the shell to extend.
- **Why it matters:** Step 3's entire wiring story ("two new registrations on the shell's own rhai engine instance") has no host. As written, the step is not launchable — an implementer would discover there is nothing to register on.
- **The corrected binding home (re-grounding required before Step 3 is started):** the World-aware bindings must attach to whichever live `rhai::Engine` actually evaluates designer scripts. Candidates the implementer must verify: (a) `vox_script`'s `RhaiRuntime` (`rhai_runtime.rs`) — but it is command-emitting with no World handle and lives in a near-engine crate, so a `metamer_divergence` query that reads overlay splat groups would have to push a *query command* and read a result channel, not return a value inline; or (b) a **new** `rhai::Engine` that the shell constructs and owns explicitly (the cleanest fit for §2b's intent) — this is additive and keeps the World-aware closure in `vox_app`, but it is **new infrastructure the spec did not budget**, not a registration on an existing instance. Decision and the resulting Done-When shape must be re-derived once (a) vs (b) is chosen.
- **Scope of the impact:** Steps 1 (the `vox_render` math) and 2 (the editor `SetIlluminant` composite) and 4 (the demo seeder + HUD) are **unaffected** — they do not depend on the rhai host. Only Step 3's rhai-query reachability rests on the false premise. The spec's headline capability (a designer script branching on metamer divergence) remains achievable; the integration path to it needs one correction the spec did not make.
