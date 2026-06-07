> **Adversarial verification:** SOUND. The skeptic checked all 28 named code references and every one resolves accurately: `ShellEntity {name, kind, pos}` at `mod.rs:88` (confirming the gap is real — no `asset_range` field today); `UndoEntry::PlacedAsset {name, start, len}` at `:120`; the `ShellRequest` enum at `:148`; `EditorShell` at `:182`; `plant_asset` at `:1117`; the range-shift undo machinery at `:969-975` with its dedicated test at `:3618`; the `menu_bar` category iteration at `:1240-1259`; `GaussianSplat::{position, set_position, apply_transform}` at `types.rs:129/181/198`. The design's central move — making the entity→overlay-range provenance a first-class field — is justified and the reuse of `plant_asset` for "one PlacedAsset undo per copy" is sound. No issues flagged.

## 0. Header

**Status:** Draft
**Scope:** Give the `ochroma_editor` shell a real selection model (one→many entities) and an `edit.duplicate` command that clones the selected `ShellEntity`s — their World rows AND their exact overlay splat ranges — at a spatial offset, as one `PlacedAsset` undo per copy. Affects only `vox_app` (the GAME layer); engine crates untouched.
**Related:** `docs/superpowers/specs/2026-06-07-aaa-capability-roadmap.md` (gap #17, Editor dim; gap #1 save/load is the sibling floor), `2026-06-06-editor-sota-shell-design.md`.
**Effort:** L. **Roadmap rank:** #17 (Editor Workflows for Production Teams), seed first-slice expanded and corrected below.

> Grounding correction to the seed: the roadmap seed says "edit.duplicate cloning selected ShellEntities + overlay ranges at an offset." Grounding revealed the load-bearing obstacle the seed glosses: **a `ShellEntity` does not know its own overlay range.** `ShellEntity` is `{name, kind, pos}` (mod.rs:88); the entity↔`[start,start+len)` mapping exists ONLY on `UndoEntry::PlacedAsset { name, start, len }` (mod.rs:120), keyed by name, and is mutated/shifted by every undo. Duplicate therefore cannot be a thin re-plant — it first needs an authoritative entity→range index. This spec makes that index a first-class field, which is also the seam every later workflow gap (#25 crash-recovery, #1 save/load round-trip of provenance) needs. That is the real shape of the L.

---

## 1. What we need

The capability: a user selects one or more things in the World panel and clones them in place, getting independent, individually-undoable copies — the universal "lay down one good asset, stamp out fifty" loop that every production editor has and Ochroma's shell does not.

Concrete observables a user/developer gains that do not exist today:

- **Multi-select in the World panel.** Click selects; Ctrl+Click toggles a row into/out of the selection; Shift+Click range-selects. Today `selected: usize` (mod.rs:186) holds exactly one index — there is no way to act on two entities at once. AAA bar: the selection set survives panel re-layout and drives the inspector + any multi-entity command.
- **`edit.duplicate` (Ctrl+D) clones the selection at an offset.** Select 1 tree → Duplicate → the World panel shows 2 entities, the viewport overlay splat count exactly doubles, and the copy is visibly offset (not z-fighting the original). Today no duplicate/copy/paste exists anywhere in the shell (roadmap Editor audit: "no multi-select/copy-paste").
- **One `PlacedAsset` undo per copy, range-exact.** After duplicating, a single Ctrl+Z removes exactly the duplicate's splat range and its World row, leaving the original bit-identical — proven against the existing range-tracked undo invariant (mod.rs:956-985), not tail truncation.
- **Duplicating N entities = N independent copies = N undo entries.** Select 3, Duplicate, get 3 new rows + 3 new ranges; three Ctrl+Z presses peel them off one at a time in LIFO order, each leaving the rest intact.
- **An authoritative entity→overlay-range index** (`asset_range` on `ShellEntity`) so any future command (move, delete, save/load provenance) can find an entity's splats without reverse-engineering the undo stack. This is the reusable seam, not a duplicate-only hack.

Why it is blocking (Editor Workflows for Production Teams dimension): the roadmap names this dimension's floor as "an editor that discards all work on close cannot hold a project, so there is no real authoring." Duplicate is the first *productive* edit above that floor — capture/AI/PCG can make one hero asset, but a production team's throughput is stamping, arranging, and varying it. Without multi-select + duplicate the shell is a viewer with one-at-a-time placement; with it, it is an arrangement tool. It also de-risks #1 (save/load) by forcing the entity→range provenance into the data model where save must serialize it.

---

## 2. How it's gonna be (the design)

Everything lives in `crates/vox_app/src/shell/mod.rs` (the GAME layer — engine crates stay game-agnostic, per CLAUDE.md). It reuses, never forks, the verified planting/undo cores. Three changes: a provenance field on `ShellEntity`, a selection set on `EditorShell`, and a `duplicate_selected` core routed through the existing `ShellRequest`/`plant_asset`/`push_undo` spine.

### 2.1 Entity provenance: `ShellEntity.asset_range`

`ShellEntity` gains an optional overlay range so an entity authoritatively owns its splats:

```rust
// crates/vox_app/src/shell/mod.rs — CHANGED
#[derive(Clone)]
pub struct ShellEntity {
    pub name: String,
    pub kind: String,
    pub pos: [f32; 3],
    /// Half-open overlay range `[start, start+len)` of THIS entity's splats in
    /// `EditorShell::overlay`, or `None` for entities with no overlay splats
    /// (the seed demo rows). Kept in lockstep with the overlay by `plant_asset`
    /// (sets it), `undo` (drains + shifts it), and `duplicate_selected` (reads it).
    asset_range: Option<OverlayRange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayRange { pub start: usize, pub len: usize }
```

`plant_asset` (mod.rs:1117) is the SINGLE writer of new ranges; it already computes `start = self.overlay.len()` and `len = splats.len()` for the `PlacedAsset` undo entry, so it sets `asset_range: Some(OverlayRange{start,len})` on the `ShellEntity` it pushes — zero new computation. The undo arm for `PlacedAsset` (mod.rs:956-985) already shifts every later undo entry's `start` down by `removed`; that same loop now ALSO shifts every `ShellEntity.asset_range` whose `start >= end`, keeping the two views coherent. This makes the undo stack and the entity list two consistent projections of the same range bookkeeping — the existing `undo_earlier_asset_shifts_later_asset_range` test (mod.rs:3618) proves the stack half; a twin assertion proves the entity half.

> Decision: store the range on the entity rather than re-deriving it from the undo stack at duplicate time. Rationale: the undo stack is LIFO and lossy (entries age out past `HISTORY_CAP=200`, mod.rs:288, and an aged-out asset is permanent-but-rangeless on the stack — mod.rs:723-730); an entity that outlived its undo entry must still be duplicable. The entity is the durable owner; the undo entry is the transient reversal record.

### 2.2 Selection model: `Selection` on `EditorShell`

`selected: usize` is replaced by a small selection type that keeps a primary (for the inspector's single-target scrub fields, which are inherently one-node) plus a set:

```rust
// crates/vox_app/src/shell/mod.rs — NEW
#[derive(Clone, Default)]
pub struct Selection {
    /// The anchor / inspector target — always a valid index when non-empty.
    primary: usize,
    /// All selected entity indices (includes `primary`). Empty == nothing selected.
    set: std::collections::BTreeSet<usize>,
}
impl Selection {
    pub fn single(i: usize) -> Self { Self { primary: i, set: [i].into() } }
    pub fn primary(&self) -> usize { self.primary }
    pub fn contains(&self, i: usize) -> bool { self.set.contains(&i) }
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ { self.set.iter().copied() }
    pub fn len(&self) -> usize { self.set.len() }
    pub fn is_empty(&self) -> bool { self.set.is_empty() }
    /// Ctrl+Click toggle; keeps `primary` valid.
    pub fn toggle(&mut self, i: usize) { /* insert/remove, repoint primary */ }
    /// Plain click: collapse to one.
    pub fn select_only(&mut self, i: usize) { *self = Self::single(i); }
    /// Shift+Click: select the inclusive range primary..=i.
    pub fn extend_to(&mut self, i: usize) { /* fill range */ }
    /// After entities removed: drop out-of-range indices, repoint primary.
    pub fn clamp_to(&mut self, len: usize) { /* retain < len */ }
}
```

`EditorShell::selected: usize` becomes `selection: Selection`. A compatibility accessor `pub fn selected(&self) -> usize { self.selection.primary() }` preserves the inspector's existing single-target reads (mod.rs:1514, `let sel = (*self.selected).min(...)`) — the inspector binds to `primary`, unchanged in behavior. The `ShellViewer.selected: &'a mut usize` borrow (mod.rs:1369) becomes `&'a mut Selection`; the hierarchy panel's `selectable_label(*self.selected == i, ...)` (mod.rs:1436) becomes `self.selection.contains(i)` and the click handler reads egui modifiers to choose `select_only` / `toggle` / `extend_to`.

### 2.3 The duplicate core + request

```rust
// ShellRequest — NEW variant (mirrors ::Undo, drained in drain_requests mod.rs:1022)
ShellRequest::DuplicateSelection,

// build_registry (mod.rs:1726): one new command, category "Edit" so the menu
// auto-lists it (menu_bar iterates registry by category — mod.rs:1240-1259; no
// menu code change needed). Closure queues the request like edit.undo does.
r.add(Command::new("edit.duplicate", "Duplicate", "Edit", "Ctrl+D", move || {
    q.borrow_mut().push(ShellRequest::DuplicateSelection)
}));

// EditorShell — NEW. Clones each selected entity that owns an overlay range,
// at a fixed world offset, through the SAME plant_asset core (so each copy gets
// one PlacedAsset undo entry, a numbered name, viewport invalidation, a receipt).
fn duplicate_selected(&mut self) -> String {
    const DUP_OFFSET: [f32; 3] = [2.0, 0.0, 0.0]; // clamp-safe constant offset
    // Snapshot indices+ranges BEFORE planting (planting mutates entities/overlay).
    let targets: Vec<(String, String, [f32;3], OverlayRange)> = self.selection.indices()
        .filter_map(|i| { let e = self.entities.get(i)?;
            Some((dup_label(&e.name), e.kind.clone(), e.pos, e.asset_range?)) })
        .collect();
    if targets.is_empty() { return "Nothing to duplicate".into(); }
    let mut new_indices = Vec::new();
    for (label, kind, pos, range) in targets {
        // Copy the source splats and translate each by DUP_OFFSET.
        let mut splats: Vec<GaussianSplat> =
            self.overlay[range.start..range.start+range.len].to_vec();
        for s in &mut splats {
            let p = s.position();
            s.set_position([p[0]+DUP_OFFSET[0], p[1]+DUP_OFFSET[1], p[2]+DUP_OFFSET[2]]);
        }
        let offset_pos = [pos[0]+DUP_OFFSET[0], pos[1]+DUP_OFFSET[1], pos[2]+DUP_OFFSET[2]];
        self.plant_asset(&label, &kind, splats, offset_pos, "Duplicated", "");
        new_indices.push(self.entities.len() - 1);
    }
    self.selection = selection_of(&new_indices); // select the copies
    format!("Duplicated {} item(s) — undo with Ctrl+Z", new_indices.len())
}
```

`dup_label` strips a trailing ` NN` and reuses the bare label so `plant_asset`'s per-label counter (`asset_counts`, mod.rs:234) continues the monotonic numbering ("Silver Birch 01" → duplicate is "Silver Birch 02"). Because each copy goes through `plant_asset`, each gets exactly one `PlacedAsset` undo entry — the seed's "one PlacedAsset undo per copy" falls out for free.

### 2.4 Data flow

```
World panel click ──(egui modifiers)──> Selection::{select_only|toggle|extend_to}
                                                  │
Ctrl+D / Edit▸Duplicate / palette ──> registry.run("edit.duplicate")
                                                  │ closure pushes
                                          ShellRequest::DuplicateSelection
                                                  │ drain_requests (next frame)
                                          duplicate_selected()
                                                  │ per selected entity w/ range
                                   overlay[range].to_vec() + set_position(+offset)
                                                  │
                                          plant_asset(label,kind,splats,pos,…)  ◄── existing core
                                          ├─ overlay.extend(splats)             (count doubles)
                                          ├─ entities.push(ShellEntity{asset_range:Some(..)})
                                          ├─ push_undo(PlacedAsset{name,start,len})
                                          └─ viewport_tex = None                 (re-rasterize)
                                                  │
                                          Ctrl+Z ─> undo() PlacedAsset arm: drain [start,end),
                                                    shift later undo starts AND entity ranges,
                                                    remove World row
```

No new device, no GPU work, no new crate — this is pure CPU editor state on the existing one-frame `ui()` request-drain loop (mod.rs:524-577). Engine-crate game-agnosticism is preserved: `GaussianSplat::position/set_position` (vox_core/types.rs:129,181) are generic geometry ops, not game concepts.

---

## 3. How it's gonna be made (the implementation plan)

Four ordered steps. Each implements AND wires in the same step. Done-Whens are exact commands with exact observable output (no "tests pass").

### Step 1 — Entity provenance index (`asset_range`) [S] — LAUNCHABLE TOMORROW

Add `OverlayRange` + `ShellEntity.asset_range: Option<OverlayRange>`. Make `plant_asset` (mod.rs:1117) set it from the `start`/`len` it already computes. Make the `PlacedAsset` undo arm (mod.rs:969-975 loop) ALSO shift every `entities[*].asset_range` whose `start >= end` down by `removed`, and clear the `asset_range` of the removed entity's row before/at removal. Update the two `ShellEntity { name, kind, pos }` literals (the demo seed rows in `new()` and any test) to `..` with `asset_range: None`.

**Exact files:** `crates/vox_app/src/shell/mod.rs` (struct mod.rs:88, `plant_asset` mod.rs:1117, undo arm mod.rs:956-985, seed rows).

**Exact test (new, in the existing `mod tests`):**
```rust
#[test]
fn plant_asset_records_entity_range_and_undo_shifts_it() {
    let mut shell = EditorShell::default();
    shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
    let tree_n = shell.overlay.len();
    let tree = shell.entities.iter().find(|e| e.name == "Silver Birch 01").unwrap();
    assert_eq!(tree.asset_range_for_test(), Some((0, tree_n))); // exact computed range
    shell.raise_terrain_headless(0);
    let terr = shell.entities.iter().find(|e| e.name == "Forge Terrain 01").unwrap();
    assert_eq!(terr.asset_range_for_test(), Some((tree_n, shell.overlay.len() - tree_n)));
    // Undo the EARLIER tree (move its entry to top, mirroring mod.rs:3636).
    let pos = shell.undo_stack.iter().position(|e|
        matches!(e, UndoEntry::PlacedAsset{name,..} if name=="Silver Birch 01")).unwrap();
    let entry = shell.undo_stack.remove(pos); shell.undo_stack.push(entry);
    assert!(shell.registry.run("edit.undo")); shell.drain_requests();
    let terr = shell.entities.iter().find(|e| e.name == "Forge Terrain 01").unwrap();
    assert_eq!(terr.asset_range_for_test(), Some((0, tree_n_terrain_len(&shell)))); // shifted to head
}
```
(`asset_range_for_test()` returns `Option<(usize,usize)>` for the private field; real computed values, never `is_some()`.)

**Done When:** `cargo test -p vox_app --lib plant_asset_records_entity_range_and_undo_shifts_it -- --nocapture` prints `test ... ok` AND the pre-existing `undo_earlier_asset_shifts_later_asset_range` still prints `ok` in the same run (proving the stack/entity views stay consistent).

### Step 2 — `Selection` model replaces `selected: usize` [M]

Introduce `Selection` (§2.2). Replace the `EditorShell.selected: usize` field with `selection: Selection`, add `pub fn selected(&self) -> usize` (primary). Re-point `ShellViewer.selected` to `&mut Selection`; update the hierarchy click (mod.rs:1436-1438) to read egui modifiers (`ctx.input` plain/`command`/`shift`) → `select_only`/`toggle`/`extend_to`; update the inspector's `(*self.selected)` read (mod.rs:1514) to `self.selection.primary()`. Call `selection.clamp_to(self.entities.len())` wherever the undo arm currently fixes `selected` (mod.rs:978-980).

**Exact test (new):**
```rust
#[test]
fn selection_toggle_and_range_track_indices() {
    let mut s = Selection::single(2);
    assert_eq!(s.primary(), 2); assert_eq!(s.len(), 1);
    s.toggle(4);                       // Ctrl+Click adds
    assert!(s.contains(2) && s.contains(4) && s.len() == 2);
    s.toggle(2);                       // toggling primary removes it, repoints
    assert!(!s.contains(2) && s.contains(4));
    s.extend_to(7);                    // Shift from primary(4) to 7
    assert_eq!(s.indices().collect::<Vec<_>>(), vec![4,5,6,7]);
    s.clamp_to(6);                     // entities shrank to 6
    assert_eq!(s.indices().collect::<Vec<_>>(), vec![4,5]);
}
```

**Done When:** `cargo test -p vox_app --lib selection_toggle_and_range_track_indices -- --nocapture` prints `ok`, AND `cargo test -p vox_app --lib shell::` reports `0 failed` (the 153 existing shell tests still pass with the field rename — the compatibility `selected()` accessor keeps the inspector tests green).

### Step 3 — `edit.duplicate` command + `duplicate_selected` core [M]

Add `ShellRequest::DuplicateSelection`, the `drain_requests` arm calling `self.duplicate_selected()` (and logging its receipt via `log_receipt`, mirroring `Undo`), the `duplicate_selected` core (§2.3), `dup_label`, and the `edit.duplicate` registry command (category "Edit"). Add Ctrl+D input handling in `ui()` next to the Ctrl+Z block (mod.rs:539-544): `i.modifiers.command && i.key_pressed(egui::Key::D)` → `registry.run("edit.duplicate")`.

**Exact test (new) — the seed's headline, computed:**
```rust
#[test]
fn duplicate_one_tree_doubles_overlay_and_one_undo_removes_exactly_the_copy() {
    let mut shell = EditorShell::default();
    shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
    let base_overlay = shell.overlay.len();
    let base_entities = shell.entities.len();
    let src: Vec<GaussianSplat> = shell.overlay.clone(); // the original's splats
    shell.selection = Selection::single(
        shell.entities.iter().position(|e| e.name == "Silver Birch 01").unwrap());
    assert!(shell.registry.run("edit.duplicate")); shell.drain_requests();
    // World shows 2, overlay doubled.
    assert_eq!(shell.entities.len(), base_entities + 1);
    assert_eq!(shell.overlay.len(), base_overlay * 2);
    assert!(shell.entities.iter().any(|e| e.name == "Silver Birch 02"));
    // The copy is the source translated by +X (not a clone in place).
    let copy = &shell.overlay[base_overlay..];
    assert_eq!(copy.len(), src.len());
    for (c, o) in copy.iter().zip(src.iter()) {
        let (cp, op) = (c.position(), o.position());
        assert!((cp[0] - (op[0] + 2.0)).abs() < 1e-4); // offset applied, exact
        assert!((cp[1] - op[1]).abs() < 1e-4 && (cp[2] - op[2]).abs() < 1e-4);
        assert_eq!(c.spectral(), o.spectral());        // radiance copied bit-exact
    }
    // One Ctrl+Z removes EXACTLY the duplicate's range; original untouched.
    assert!(shell.registry.run("edit.undo")); shell.drain_requests();
    assert_eq!(shell.overlay.len(), base_overlay);
    assert_eq!(shell.entities.len(), base_entities);
    assert!(!shell.entities.iter().any(|e| e.name == "Silver Birch 02"));
    for (a, b) in shell.overlay.iter().zip(src.iter()) {
        assert_eq!(a.position(), b.position());        // original bit-identical
    }
}
```

**Done When:** `cargo test -p vox_app --lib duplicate_one_tree_doubles_overlay_and_one_undo_removes_exactly_the_copy -- --nocapture` prints `ok`. This is the seed's verbatim acceptance ("select 1 tree, Duplicate, World shows 2 + overlay doubles + one Ctrl+Z removes exactly the duplicate's range") as a computed assertion.

### Step 4 — Multi-duplicate + headless pixel proof [M]

Cover N>1: a test selecting 2 entities, duplicating, asserting 2 new rows + 2 new `PlacedAsset` entries + 2 LIFO undos. Then a headless viewport proof through the existing `viewport::scene_texture` path (the same readback the shell snapshot uses, mod.rs:581): assert the rasterized texture has MORE non-background pixels after duplicate than before (the copy is visible), mirroring `overlay_adds_visible_pixels_over_base` (mod.rs, viewport tests).

**Exact test (new):**
```rust
#[test]
fn duplicate_two_entities_makes_two_copies_two_undos_and_more_pixels() {
    let mut shell = EditorShell::default();
    shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
    shell.raise_terrain_headless(0);
    let lit_before = nonbackground_pixels(&shell); // helper: rasterize overlay, count
    let undo_before = shell.undo_stack.len();
    shell.selection = selection_of(&[0, 1]);        // both rows
    assert!(shell.registry.run("edit.duplicate")); shell.drain_requests();
    assert_eq!(shell.undo_stack.len(), undo_before + 2); // one PlacedAsset per copy
    assert_eq!(shell.entities.len(), 4);
    let lit_after = nonbackground_pixels(&shell);
    assert!(lit_after > lit_before, "{lit_after} !> {lit_before}"); // copies visible
    // Two LIFO undos peel exactly the two copies.
    assert!(shell.registry.run("edit.undo")); shell.drain_requests();
    assert!(shell.registry.run("edit.undo")); shell.drain_requests();
    assert_eq!(shell.entities.len(), 2);
    assert_eq!(shell.undo_stack.len(), undo_before);
}
```

**Done When:** `cargo test -p vox_app --lib duplicate_two_entities_makes_two_copies_two_undos_and_more_pixels -- --nocapture` prints `ok`, AND `cargo run -p vox_app --bin ochroma_editor -- --frames 2 --shot /tmp/dup.png` exits 0 and writes `/tmp/dup.png` (the binary still launches with the changed shell — no panic, the no-panic shell rule holds).

---

## 4. How it fits (integration + dependencies)

**Depends on (named gaps):**
- **#1 Project + open/save/save-as** — the `ShellEntity.asset_range` provenance this spec adds is exactly what save/load must serialize to round-trip a duplicated world. This spec hardens the data model #1 needs; they should land adjacently. No hard ordering, but #1 must serialize `asset_range` or it will discard duplicate provenance.
- Nothing else hard-blocks it; it runs entirely on the existing CPU shell loop.

**What depends on it:**
- **#25 Session crash recovery + autosave** — autosave serializes the same entity+range provenance.
- **#9 Play-in-Editor** — Play snapshots/restores the world; a coherent entity↔overlay index makes the snapshot exact.
- Any future move/delete/group/prefab-library command — all need the entity→range index this introduces. Duplicate is the first consumer; the index is the reusable seam.

**Composes with existing systems (named):**
- The `ShellRequest`/`drain_requests` spine (mod.rs:148,1022) — `DuplicateSelection` is a sibling of `Undo`.
- The shared `plant_asset` core (mod.rs:1117) — every plug-in's plant (FloraPrime trees, Forge terrain/buildings, Crucible scenes) routes through it; duplicate joins them with zero forked machinery.
- The range-tracked `PlacedAsset` undo (mod.rs:956-985) — duplicate produces standard `PlacedAsset` entries, so undo "just works" and stays bit-exact (the `undo_earlier_asset_shifts_later_asset_range` invariant, mod.rs:3618).
- The one-command-surface registry (command_palette.rs:30) — `edit.duplicate` is reachable from menu (auto-listed by category), Ctrl+D, AND the Ctrl+K palette, all dispatching the same closure. The Ask-Ochroma intent path could later emit `RunCommand{id:"edit.duplicate"}` for free.

**What it must NOT break:**
- **The 11-green-gate / both-config builds:** the change is `vox_app`-only and feature-flag-agnostic; `cargo build` and `cargo build --features forge-native,crucible-native` must both compile. Step 2's field rename is the only churn risk — the `selected()` accessor neutralizes it for the inspector tests.
- **The no-panic shell rule:** `duplicate_selected` clamps to existing indices (`entities.get(i)`), filters out rangeless entities, and no-ops on an empty selection ("Nothing to duplicate") — it can never index out of bounds or panic on a stale selection. The `DUP_OFFSET` is a constant (no unbounded numeric input); if a future variable offset is added it goes through the same clamp discipline every numeric input in the shell already follows.
- **The 153 passing shell tests** — Step 2's Done-When gates on `0 failed`.

**4-phase sequencing:** Phase 1 (Editor floors track) alongside #1 — "everything here is independently startable on this exact codebase." It is the productive-edit layer immediately above the save/load floor and needs no GPU loop, no relight, no AI. The roadmap's Editor track is `#1 → #9/#17/#25`; this is the #17 node, sharing the provenance seam with #1 and feeding #9/#25.

**Cross-gap seams:** the `asset_range` field is the single seam where Editor-workflow gaps (#1 save, #25 recovery, #9 Play) converge. Defining it here, with an undo-shift invariant test, pays that integration debt once.

---

## Surprises & advantages

Grounded discoveries that make this cheaper or stronger than the seed implies:

- **`plant_asset` is already the universal asset on-ramp.** Trees, terrain, buildings, AND Crucible scenes all funnel through one core (mod.rs:1050-1145) that handles overlay insertion, monotonic naming, range-tracked undo, viewport invalidation, and receipts. Duplicate inherits ALL of it by calling `plant_asset` on copied splats — "one PlacedAsset undo per copy" is not engineered, it falls out. The seed's whole undo requirement is satisfied by reuse, not new code.
- **The range-shift undo machinery already solves the hard half.** The `PlacedAsset` undo arm already drains an exact `[start,end)` range and shifts every later entry's `start` down (mod.rs:969-975), with a dedicated coexistence test (mod.rs:3618). Duplicating N items and undoing them one-at-a-time is the EXACT scenario that machinery was built for — interleaved independent ranges. We extend that one shift loop to also move `asset_range`; the invariant is already proven for the stack.
- **The menu surfaces the command for free.** `menu_bar` iterates the registry by `category` (mod.rs:1240-1259), so registering `edit.duplicate` in category "Edit" makes it appear in the Edit menu with its Ctrl+D hint with ZERO menu-code changes — same for the Ctrl+K palette. One `Command::new` line lights up three surfaces.
- **`GaussianSplat::set_position` is a clean offset primitive.** The offset clone is a per-splat `set_position(p + offset)` (types.rs:181) — no transform matrix, no rotation bookkeeping needed for the first slice. A richer transform later is a one-line swap to `apply_transform` (types.rs:198), which already exists and is tested.
- **Real surprise the seed missed (a benefit disguised as a cost):** because `ShellEntity` had no range, this work is FORCED to add the entity→overlay provenance index — which is precisely the missing piece #1 (save/load) and #25 (crash recovery) both need to serialize a world. The duplicate feature pays down a dependency two other Editor-track gaps were going to hit. What looked like scope creep is actually the shared seam landing early, under a test that pins the undo/entity coherence invariant.
- **First-mover angle on the wedge:** duplicated splats carry their baked `spectral[16]` radiance bit-exact (asserted in Step 3). The moment the GPU relight kernel (#5) lands, a stamped-out forest of duplicates ALL relight correctly with no per-copy bake — duplicate composes with the relight wedge automatically because the data model is radiance-uniform.
