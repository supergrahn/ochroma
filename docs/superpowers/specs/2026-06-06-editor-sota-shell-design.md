# Design: Ochroma Editor — SOTA Shell + Host-Plugin Design System (2026-06-06)

**Status:** Draft
**Scope:** The editor face: egui-pro shell (docking, tokens, icons, type ramp, node canvas, command palette) hosted in vox_app's existing egui editor, a vox_ui token design system both UI stacks consume, and the host-plugin contract making Crucible/FloraPrime/Forge visual editors inside Ochroma. UiTree+Vello remains the in-game/viewport overlay stack with a defined convergence line.
**Related:** [Competitive research](./2026-06-06-engine-competitive-research.json), [retained UI tree slice](../../FEATURES.md), the SOTA checklist embedded below as the acceptance rubric

> Produced by a researched judge-panel workflow (wf_f3a4b51f-d8f): a weighted 100-point SOTA checklist distilled from UE 5.7 / Unity 6 / Godot 4.4 / Rerun / Zed / Blender 4.x (current frames calibrate at ~3-6/100), three mandated designs, two adversarial judges who VERIFIED codebase claims line-by-line. Winner: egui-pro (83.5) with grafts from hybrid-staged (shared tokens, stack seam) — uitree-native (70, feasibility 3.5) stays the long-game convergence track. The user's brief, verbatim, is the Problem statement.

---

**Status:** Proposed
**Author:** Synthesis (egui-pro spine + Hybrid token-spine/Crucible-adapter grafts)
**Date:** 2026-06-06

## UX Principle (user feedback on mockup v1, 2026-06-06)

> "Maybe a bit more modernized, more informative, comprehensive and easy to use for a **non-game developer**."

This is a STRATEGIC differentiator, not a nice-to-have: Unreal is expert-hostile; nobody owns
"approachable pro engine." Binding consequences for every editor surface:
- **Plain language first**: "World", "Properties", "Material & Light — how it looks under real light",
  "View: Real light", "Detail budget" — jargon lives in tooltips, not labels.
- **Labeled primary actions**: the Canva rule — "＋ Add to world" is a big labeled button, not a bare icon.
- **Guided, not empty**: contextual assistant suggestions in the viewport (tip chips with one-click "Do it"),
  wired to the in-editor AI assistant candidate (#16). "Ask Ochroma" (Ctrl+K) is the universal entry point —
  command palette and assistant are ONE surface.
- **Progressive disclosure**: friendly defaults on top, expert depth one foldout down. Status bar reads
  "All systems healthy", details on hover/click.
- Mockup v2 (committed alongside) is the calibrated reference for this principle.

## UX Principle 2 — AI-native interaction (user direction, 2026-06-06)

> "With AI heavily involved in both generation, creation, assembly, game play, development of code and
> everything, that probably changes how users use the software as well."

The editor is designed AI-NATIVE, not AI-decorated. Binding consequences:
- **One command surface**: every editor action (host or plugin) is a registered `Command` — the palette,
  keyboard, menus, AND the AI assistant all invoke the same registry. The plugin contract's command
  registration IS the AI tool-call surface; nothing is operable by hand that an agent can't drive, and
  vice versa.
- **Intent-first creation**: "Ask Ochroma" (Ctrl+K) generates, not just navigates — scene edits, cook
  graphs (Crucible), vegetation (FloraPrime), environments (Forge), spectral materials, Rhai scripts.
  Existing substrate: vox_nn text_to_city/nl_commands/llm_client (library-only today), the deterministic
  LLM stub, the tip-chip pattern from mockup v2. Manual tools are the refinement layer.
- **Provenance + reversibility**: AI-generated content enters through the same undo stack, diffable like
  any edit (the node graphs make this natural: a generated cook graph IS inspectable structure, not a
  black-box blob).
- **Agent-operable by construction**: the headless pixel-provable surfaces (smokes, shell_snapshot) are
  the same affordances that let agents run, verify, and iterate the editor autonomously — the development
  model that built this engine becomes a product feature.
Phase 1 (tokens/dock/icons) is unaffected; the Command registry lands with the plugin contract phase and
"Ask Ochroma" upgrades from palette to generative assistant in the assistant phase (#16).

## Problem

- The user's verdict on the shipped editor frames, verbatim: **"not what you expect from a 2026 SOTA game engine editor."** They are right.
- The frames the user reacted to come from `crates/vox_app/src/bin/engine_runner.rs` — a CPU **software compositor** drawing fixed panels (inspector / node-graph strip / HUD) with a **3x5 / 5x7 bitmap font** (`burn_text`). A bitmap-font UI alone caps any SOTA score below 50 (checklist item 3).
- There is **no tokenized theme**: `crates/vox_ui/src/theme.rs` writes literal `Color32::from_rgb` into egui `Visuals` from a flat `OchromaColors` const struct (verified: `apply_ochroma_theme` at L4, `OchromaColors` at L126, no cascade/override/swap).
- There is **no icon family**: `entity_icon()` returns single placeholder Unicode codepoints (◆ ☀ ♪ ⚡), not a vector set.
- There is **no docking, no chrome shell, no command palette, no perf graph**; node-graph wires render as flat software primitives below SOTA fidelity.
- The editor LOGIC is strong and largely already exists (verified: `crates/vox_app/src/editor.rs` is 1252 lines with a working menu bar L493, toolbar L591, status bar L702, hierarchy SidePanel L735, inspector with `DragValue` scrub L819-830; `node_graph_widget.rs::show_egui` L467 already draws **cubic bezier** wires L512; `vox_editor` holds registry/search/live_cook/templates/subgraphs/content_browser). The FACE is the problem, not the brains.
- Two editor faces exist simultaneously — the software-compositor `engine_runner` and the egui editor in `editor.rs`/`main.rs` — and the team risks shipping both.
- The ecosystem mandate is unmet: Crucible (PCG cook graphs), Forge (terrain/road/water/scatter → `ForgeVolume`), and FloraPrime (vegetation) have no host to plug into, and nothing forces them to share Ochroma's look.

## Solution Summary

Promote the existing egui editor (`editor.rs`, egui 0.31 / egui-wgpu 0.31 / wgpu 24) to **THE Ochroma editor**, and retire `engine_runner`'s software-compositor face. This is the **Rerun playbook (egui done seriously)**: it reaches the full SOTA checklist without a wgpu-vs-vello version war, because egui already runs on the engine's wgpu 24.

Onto that single coherent stack we add: a **JSON-backed token design system** as the single source of truth (`vox_ui::tokens::Tokens` ← `assets/ui/ochroma.theme.json`); **egui_dock** docking; **egui-phosphor** icons; an egui-native **NodeCanvas** (bezier/type-colored/grid/minimap) over the existing `OchromaNodeGraph`; a fuzzy **command palette**; and the splat **viewport-as-GPU-texture** via `register_native_texture` (no CPU readback of the frame).

The **in-game / viewport-overlay UI stays on the owned UiTree + Vello GPU path** (`vox_ui`, the rank-1 investment) — SpectralHUD, the perf unitgraph, gizmo labels, and on-wire value chips composite over the live 3D framebuffer at game framerate **without an egui repaint**. The egui editor and the Vello overlay **read the exact same `Tokens` bytes** (egui via `egui_theme::apply`, UiTree via `StyleSheet::from_tokens`), proven pixel-equal in CI. This is the one non-negotiable seam, made explicit and token-unified rather than hidden.

Ochroma is THE editor application. Crucible/Forge/FloraPrime ship as **`EditorPlugin`s** whose `PluginCtx` exposes **only** `&Tokens`, the shared widget kit, and the shared `NodeCanvas` — there is **no API to set a `Visuals` or push a raw `Color32` panel**, so a plugin physically cannot look different from the host.

Headless provability is first-class: every scored item gets a pixel/behavior assertion via **egui_kittest** (editor face) and **`render_to_rgba`** (Vello overlay, the path `spectral_hud.rs` already asserts).

## Done When

Each gate is one exact command with exact human-visible / asserted output. All headless. `--bin shell_snapshot` builds an `egui::Context`, runs the real `EditorShell`, renders RGBA, and writes a PNG.

1. **Tokens are the single source, both stacks agree.**
   `cargo test -p vox_ui tokens::tests::egui_and_uitree_resolve_same_accent`
   → PASS, prints `accent egui=[60,130,255,255] uitree=[60,130,255,255] EQUAL`. Asserts `egui_theme::apply` Visuals widget-active fill bytes == `StyleSheet::from_tokens` resolved accent bytes == `Tokens::load("assets/ui/ochroma.theme.json").color("accent.base")`. (items 1, 19-seam)

2. **Theme swap is a file edit.**
   `cargo test -p vox_ui tokens::tests::light_theme_swaps_window_bg`
   → PASS; loads `ochroma_light.theme.json`, asserts window bg changes from `[18,18,24]` to the light value by >40 in a channel. (item 1)

3. **AA vector type ramp — the bitmap font is gone.**
   `cargo test -p vox_ui widgets::tests::type_ramp_is_aa_not_bitmap`
   → PASS; egui_kittest renders title(20)/body(13)/caption(11), asserts >2 distinct glyph heights AND >16 distinct grayscale luminance levels along a glyph edge (continuous AA, not a 7px grid). (item 3)

4. **Bitmap-font eradication is a hard CI gate.**
   `cargo run -p vox_app --bin shell_snapshot -- --shot /tmp/shell.png && cargo test -p vox_app shell::tests::no_burn_text_signature`
   → writes `/tmp/shell.png`; scans every panel region for the exact `burn_text` 5x7 glyph bitmap and asserts **zero matches**. (item 3, regression lock)

5. **Coherent icon family.**
   `cargo test -p vox_ui icons::tests::phosphor_glyph_present`
   → PASS; renders `icon::TERRAIN` (phosphor `mountains`), asserts the rasterized bbox has >120 non-background vector pixels and >=6 role icons produce distinct pixel signatures. (item 4)

6. **Full chrome + drag-drop docking.**
   `cargo test -p vox_app shell::tests::dock_tabs_present_and_movable`
   → PASS; asserts the kittest tree contains tab labels `Hierarchy, Inspector, Viewport, Node Graph, Content`, a draggable tab strip rect, menu+toolbar+status bands at expected y-ranges; calls `dock.move_tab("Inspector","Hierarchy")` and asserts the Inspector pane x-origin moves by > pane_w/2; layout round-trips through `editor_layout.ron`. `/tmp/shell.png` visibly shows a dark tokenized editor with icon toolbar, dock tabs, menu+status bars. (items 5, 6)

7. **Viewport is a GPU texture, no CPU readback.**
   `cargo test -p vox_app shell::viewport::tests::viewport_is_gpu_texture`
   → PASS; asserts the Viewport tab paints an `egui::Image` whose `TextureId` is a `User` id from `Renderer::register_native_texture`, not an epaint-managed texture, and that no CPU readback of the splat frame occurs on the display path. A resize test asserts the `TextureId` is re-registered when the texture view changes. (item 19)

8. **Bezier, type-colored, grid wires.**
   `cargo test -p vox_ui node_canvas::tests::bezier_typed_wires`
   → PASS; renders a `Terrain(Splats out) -> Splatize` graph, asserts the wire midpoint y differs from the straight-chord y by > thickness (curved) AND the wire color at the source equals `tokens.wire_color(PortType::Splats)=[100,150,255,255]` AND a dot grid renders at zoom 0.5. (items 10, 11, 12)

9. **Minimap.**
   `cargo test -p vox_ui node_canvas::tests::minimap_drawn`
   → PASS; asserts a 240x160 minimap rect at ~0.65 opacity, bottom-right, containing scaled node rects. (item 13)

10. **Inspector foldout + search + drag-scrub.**
    `cargo test -p vox_app inspector::tests::scrub_and_foldout`
    → PASS; simulates a label-drag on the X field, asserts the bound `f32` changed by drag·speed; asserts a collapsed foldout hides its body rect; typing in the search box narrows visible rows. (items 15, 16)

11. **Fuzzy command palette.**
    `cargo test -p vox_app shell::tests::command_palette_fuzzy`
    → PASS; opens palette (Ctrl+Shift+P), types `tern`, asserts the top hit is `Forge: Terrain` via the existing registry prefix>substring>subsequence ranker. (item 17)

12. **Content browser rendered.**
    `cargo test -p vox_app content::tests::thumbnail_grid`
    → PASS; asserts the Content tab paints thumbnail grid cells + a type-filter row + a breadcrumb. (item 18)

13. **Live perf overlay on the Vello GPU path.**
    `cargo run -p vox_app --bin engine_runner -- --headless --frames 120 --shot /tmp/overlay.png && cargo test -p vox_app overlay::tests::perf_unitgraph_is_a_graph`
    → PASS; the perf overlay renders through `VelloCtx::render_to_rgba` (asserts `vello_hud_px_last_frame > 5000` on a GPU host; self-skips to the CPU `VelloCtxCpu` oracle with a printed note when no adapter); asserts the rolling line has >=30 distinct heights over 120 frames (a time-series, not a number). Gizmo axis labels match `axis.x/y/z` = `#E63C3C/#3CC83C/#3C64E6`. (items 9, 8, 19-overlay)

14. **Plugin inherits the host design system (the contract proof — Crucible, the cheap example).**
    `cargo test -p vox_app plugins::crucible_editor::tests::panel_styled_by_host`
    → PASS; installs `CruciblePlugin` into `EditorShell`, renders, asserts the Crucible panel background pixel == `tokens.color("surface.2")=[30,30,40,255]` (the plugin set no color) AND its node headers use `category_header(Spatial)=[60,130,80,255]`. (directive C)

15. **Hover motion.**
    `cargo test -p vox_app shell::tests::hover_changes_button_fill`
    → PASS; asserts a button fill differs between hover and non-hover frames (egui `animation_time` from `motion.fast`). (item 20)

16. **Full-shell regression lock.**
    `cargo test -p vox_app --test sota_snapshot`
    → PASS; egui_kittest snapshot diff of the full shell against `tests/snapshots/shell_dark.png` within tolerance.

**Aggregate human-visible proof:** open `/tmp/shell.png` — a dark, tokenized, icon-dense, dockable editor with a GPU viewport image, a bezier type-colored node graph with minimap, a searchable foldout inspector with drag-scrub fields, a command palette, and a Crucible plugin panel **indistinguishable in styling from host panels**. Open `/tmp/overlay.png` — a rolling perf unitgraph and color-coded gizmo labels composited over the live splat frame, all anti-aliased proportional text, zero bitmap glyphs.

## Capabilities

| capability (checklist item) | real pixel/behavior test | forbidden stub |
|---|---|---|
| Tokenized theme, JSON source (1) | `egui_and_uitree_resolve_same_accent` asserts both stacks resolve `accent.base` to identical bytes from the JSON | `assert!(tokens.color("accent").is_some())`; hardcoded `Color32` in `apply`; a flat const palette with no `Tokens`/JSON |
| Theme swap (1) | `light_theme_swaps_window_bg` asserts bg byte-changes on file swap | `assert!(load("light").is_ok())` |
| 4px/rem scaling (2) | set `rem` 14→18, assert `button_padding`/panel rects scale proportionally | pixel-fixed magic numbers |
| AA vector type ramp (3) | `type_ramp_is_aa_not_bitmap` asserts >16 grayscale levels on a glyph edge + >2 sizes | any `burn_text`/3x5/5x7 use in the editor path |
| Bitmap-font eradication (3) | `no_burn_text_signature` scans panels for the 5x7 glyph bitmap, asserts 0 matches | leaving a software-compositor panel in the editor face |
| Icon family (4) | `phosphor_glyph_present` asserts >120 vector px for `icon::TERRAIN`, 6 distinct icons | a single Unicode codepoint as "the icon" |
| Drag-drop docking (5) | `dock_tabs_present_and_movable` asserts named tabs, draggable strip, `move_tab` repositions, ron round-trip | fixed `SidePanel`-only layout |
| Full chrome (6) | same test asserts menu+toolbar+tab strip+status bands by rect | missing any of the four bands |
| Floating viewport toolbar (7) | asserts an `Area` over the viewport `Image` holds gizmo/space/snap icon buttons | side-panel-only viewport controls |
| In-viewport gizmos (8) | overlay test asserts `axis.x/y/z` colored handles at screen-constant size | flat non-axis handles |
| Live perf graph (9) | `perf_unitgraph_is_a_graph` asserts >=30 distinct line heights over 120 frames | a single FPS text label only |
| Bezier/typed/grid wires (10/11/12) | `bezier_typed_wires` asserts curve + `wire_color(Splats)` match + dot grid | 1px straight monochrome lines |
| Minimap (13) | `minimap_drawn` asserts 240x160 @0.65 rect with scaled node rects | no minimap |
| Node styling/org (14) | asserts category header colors, rounded body alpha, muted-red outline, comment-frame rect | logic-only with no rendered fidelity (<=0.3) |
| Inspector foldout+search (15) | `scrub_and_foldout` asserts collapsed body hidden + live filter narrows rows | flat label list |
| Drag-scrub fields (16) | `scrub_and_foldout` asserts label-drag mutates bound `f32` by delta | static `property_row` label+spacer |
| Command palette (17) | `command_palette_fuzzy` asserts `tern` → `Forge: Terrain` | no palette |
| Content browser rendered (18) | `thumbnail_grid` asserts cells + type filter + breadcrumb | logic without thumbnails |
| GPU viewport (19) | `viewport_is_gpu_texture` asserts `User` `TextureId` via `register_native_texture`, no frame readback | reading the splat frame back to CPU to blit |
| GPU overlay (19) | `perf_unitgraph_is_a_graph` asserts `vello_hud_px_last_frame>5000` via real `render_to_rgba` | `assert!(vello_ctx.is_some())` |
| Motion (20) | `hover_changes_button_fill` asserts fill differs hover vs non-hover | static frames |
| Plugin inherits design system | `panel_styled_by_host` asserts panel bg == host token + headers == category token | a plugin that sets its own `Visuals`/`Color32` (the API makes this impossible) |

## Architecture

### Stack division (the one explicit seam)

| Surface | Stack | Why |
|---|---|---|
| Editor shell: chrome, docking, menus, toolbars, inspector, content browser, command palette, asset editors, **node-graph editor canvas** | **egui 0.31 + egui_dock + egui-phosphor**, on egui-wgpu (**wgpu 24**) | egui already runs on the engine's wgpu 24; the near-SOTA editor already exists in `editor.rs`; no version war |
| Splat **viewport** inside the shell | egui `Image` over a **GPU texture** via `register_native_texture` | the splat frame is already a wgpu 24 texture — stop reading it to CPU |
| **In-game UI + viewport overlays**: SpectralHUD, perf unitgraph, gizmo axis labels, on-wire value chips | **vox_ui UiTree + Vello GPU** (`render_to_rgba`) | composites over the 3D framebuffer at game framerate without an egui repaint; preserves the rank-1 Vello investment |
| **Convergence spine** | `vox_ui::tokens::Tokens` ← `assets/ui/ochroma.theme.json`, consumed by BOTH stacks | one JSON edit reskins the whole ecosystem; proven pixel-equal in CI |

**Documented end-state (deferred, not abandoned):** egui (wgpu 24) and Vello (wgpu 23.0.1, vello 0.4.1) currently cross at the framebuffer composite. When **vello 0.5+ lands on wgpu 24** the two paths unify on one device; tracked in `vello_ctx.rs`. Until then the seam is real, named, and token-unified — it does not block any checklist item (the egui face is fully GPU on wgpu 24 today; the Vello overlay is GPU on wgpu 23 today; only their *composite* crosses CPU).

### Token design system — `vox_ui` (host, engine-agnostic)

- **`vox_ui/src/tokens.rs`** (NEW). `struct Tokens { color: HashMap<String,[u8;4]>, space: [f32;7], radius: [f32;3], type_ramp: TypeRamp, motion_ms: HashMap<String,f32>, rem: f32 }`. `Tokens::load(path) -> io::Result<Tokens>` (serde JSON), `Tokens::default()` (== `ochroma.theme.json`, kept in sync by a round-trip test), `color(&self, dotted: &str) -> [u8;4]`, `wire_color(&self, PortType) -> [u8;4]`, `category_header(&self, NodeCategory) -> [u8;4]`. The flat `OchromaColors` consts become `#[deprecated]` thin readers off `Tokens::default()`; `AXIS_X/Y/Z` are carried verbatim into the JSON.
- **`vox_ui/src/egui_theme.rs`** (NEW). `apply(ctx: &egui::Context, t: &Tokens)` writes egui `Visuals` + `Spacing` + `text_styles` derived FROM tokens. Replaces the literal-RGB `apply_ochroma_theme`, which becomes a one-line shim `apply(ctx, &Tokens::default())` so `editor.rs`/`main.rs` build unchanged through migration.
- **`vox_ui/src/ui_tree.rs`** (EXTEND). Add `StyleSheet::from_tokens(&Tokens) -> StyleSheet` so the Vello overlay path resolves classes to the same RGBA. (Verified current state: `UiKind` has 6 variants, no token indirection — `from_tokens` is the minimal addition needed for the overlay to share the spine; the overlay does **not** need the full editor widget kit.)
- **`vox_ui/src/design/icons.rs`** (NEW). Re-export egui-phosphor; `install(ctx)` merges the Phosphor font family; `mod icon { pub const TERRAIN: &str = …; }` maps roles to codepoints (replaces `entity_icon()`).

### Widget kit — `vox_ui/src/widgets.rs` (NEW, the kit plugins MUST use)

`scrub_drag(ui, &mut f32, ScrubOpts)` (label-drag scrub + direct entry + optional slider), `foldout(ui, id, title, body)` (collapsible section), `search_box(ui, &mut String)`, `icon_button(ui, icon, tip)`, `tab_label(icon, text, closeable)`. All styled from `Tokens` only.

### NodeCanvas — `vox_ui/src/node_canvas/` (NEW, replaces `node_graph_widget.rs`'s render path)

A `GraphModel` trait + one egui-native renderer so host and every plugin share one canvas:

- Infinite dot/line grid (zoom 0.23..2.07, step 1.2, pan); cubic **bezier** wires (curvature 0.5, thickness 4.0, AA) whose color is a **gradient between the two socket `PortType` colors** (Blender colored-noodle); exec/Flow wires drawn white with an L→R arrowhead (UE K2); colored port sockets; rounded node bodies (radius `md`) with per-category colored headers from `category_header(NodeCategory)`; muted/error red outline; reroute knots; tintable comment frames (C key); a 240x160 minimap at 0.65 opacity, bottom-right.
- The existing `VisualNode/VisualConnection/CommentBox/NodeThumbnail` data structs move here unchanged as the `GraphModel` adapter so construction sites keep compiling. `node_graph_widget.rs::render_to_pixels` (the CPU rasterizer) is retired from the editor path but may remain `#[doc(hidden)]` for headless thumbnails.

### Host shell + plugin contract — `vox_app/src/shell/`

- **`shell/mod.rs`** — `EditorShell` owns an egui_dock `DockState<PanelId>` + a `TabViewer` dispatching each tab to a built-in panel or a plugin panel. The fixed `SidePanel`/`TopBottomPanel` bodies in `editor.rs::show` are **moved verbatim** into `TabViewer` arms (hierarchy, inspector, viewport, content, node graph, console); menu_bar + toolbar + status_bar stay as `TopBottomPanel`s around the dock area. Layout persists to `editor_layout.ron`.
- **`shell/host.rs`** — the plugin contract (below).
- **`shell/viewport.rs`** — registers the splat wgpu texture via `Renderer::register_native_texture`, shows it as an `egui::Image` in the Viewport tab, paints the floating viewport toolbar (`Area`) and gizmo overlay over it, re-registers the `TextureId` on resize.
- **`shell/command_palette.rs`** — fuzzy palette (Ctrl+Shift+P) over the registered `Command`s + assets, reusing `vox_editor::registry`'s prefix>substring>subsequence ranker.

### Plugin contract (the UE host-plugin model)

```rust
// vox_app/src/shell/host.rs
pub trait EditorPlugin { fn id(&self) -> &str; fn register(&mut self, host: &mut EditorHost); }

pub struct EditorHost { /* panel/graph/asset/command/menu/toolbar registries */ }
impl EditorHost {
    pub fn add_panel(&mut self, desc: PanelDesc);
    pub fn add_graph_editor(&mut self, desc: GraphEditorDesc);
    pub fn add_asset_editor(&mut self, desc: AssetEditorDesc);
    pub fn add_command(&mut self, cmd: Command);
    pub fn add_menu(&mut self, item: MenuItem);
    pub fn add_toolbar(&mut self, item: ToolItem);
}
pub struct PanelDesc { pub id: String, pub title: String, pub icon: &'static str,
    pub default_dock: Dock, pub build: Box<dyn FnMut(&mut egui::Ui, &mut PluginCtx)> }
pub struct GraphEditorDesc { pub id: String, pub title: String,
    pub registry: Box<dyn NodeRegistryLike>, pub model: Box<dyn vox_ui::node_canvas::GraphModel> }

// ENFORCEMENT: a plugin receives ONLY the design system — no Visuals setter, no Color32, no top-level panel.
pub struct PluginCtx<'a> {
    pub tokens: &'a vox_ui::tokens::Tokens,
    pub widgets: &'a vox_ui::widgets::WidgetKit,
    pub canvas:  &'a mut vox_ui::node_canvas::NodeCanvas,
}
```

A plugin node-graph editor is just `add_graph_editor` with the plugin's own registry; it inherits curved/type-colored wires, grid, and minimap from the host with zero plugin rendering code. A plugin node colors its header by passing a `NodeCategory` enum (Generator/Spatial/Field/Sink/Math) — **never an RGB** — so even node coloring is tokenized. Ochroma's own inspector/content-browser/node-graph panels are implemented through this same contract (dogfood). ENGINE crates (`vox_core`/`vox_data`/`vox_render`) gain **no** game/tool concepts; all plugin code lives in `vox_app/src/plugins/`.

### Data flow (editor frame)

winit → egui_winit → egui pass: `EditorShell::ui` lays out the egui_dock tree (built-in + plugin panels) → egui-wgpu paints to the wgpu 24 surface; the Viewport tab draws the splat `register_native_texture` `Image`; the Vello overlay (HUD/perf/gizmo-labels) composites over the viewport texture. Param edits in the egui inspector or NodeCanvas route through the **existing** `OchromaNodeGraph::request_recook` / `live_cook` path — the live PCG loop is unchanged, only its face changes.

## Design Language (concrete token values)

The shared ecosystem design system. All values live in `vox_ui/src/tokens.rs` + `assets/ui/ochroma.theme.json`; egui consumes via `egui_theme::apply`, UiTree via `StyleSheet::from_tokens`. Plugins consume `Tokens` only; none may hardcode color/size.

**COLOR (dark, default; `ochroma_light.theme.json` proves the swap):**
- `surface.bg.0` #121218 (window) · `.1` #18181C (panel) · `.2` #1E1E28 (raised/cards) · `.3` #26262F (widget)
- `surface.hover` #303041 · `surface.active` #37374B · `surface.border` #2D2D3C · `border.strong` #3A3A4C
- `accent.base` #3C82FF (60,130,255) · `accent.hover` #5096FF · `accent.dim` #285AB4
- `text.primary` #DCDEE6 · `text.secondary` #8C91A0 · `text.disabled` #50555F
- `status.success` #32C878 · `status.warning` #FFB432 · `status.error` #FF4646
- `axis.x` #E63C3C · `axis.y` #3CC83C · `axis.z` #3C64E6 (carried verbatim — the one already-SOTA convention)

**PORT-TYPE → SOCKET COLOR** (drives type-colored gradient wires; keyed by `vox_editor::node_graph::PortType`): `Splats` #4CC2FF · `SpectralField` #B478FF · `Terrain` #8CC85A · `Mesh` #FFB45A · `LodMesh` #FF8C5A · `Instances` #FFE164 · `Scalar` #B0B6C8 · `BiomeMap` #5AC8A0 · `SplatWeights` #C896FF · exec/Flow #FFFFFF (arrowed L→R).

**NODE-HEADER by `NodeCategory`:** Generator #6C5A3C · Spatial/Terrain #3C8250 · Field #6E5A96 · Sink/Output #3C5A82 · Math #5A6478. Wire color gradients between the two endpoint socket colors.

**SPACE (4px grid, rem-scalable):** `s0`=0 `s1`=4 `s2`=8 `s3`=12 `s4`=16 `s5`=24 `s6`=32. `rem = ui_font_size` (default 14); egui `Spacing` derived as multiples of `rem` (item_spacing `(s2,s1)`, button_padding `(s2,s1)`, window_margin `s2`, indent `s3`), so one Settings knob scales the whole UI (GPUI rule).

**RADIUS:** `sm`=4 (widgets/tabs) · `md`=6 (cards/windows/nodes) · `lg`=8 · pill=999. Tabs `sm` top-only.

**TYPE RAMP (real AA vector via egui glyph atlas — the bitmap font is gone from the editor):** caption 11 / body 13 / body-strong 13 bold / heading 16 / title 20; mono 12 for values. Phosphor merged as a font family so icons sit inline with text.

**ICONS (egui-phosphor, regular+fill):** move=arrows-out-cardinal, rotate=arrows-clockwise, scale=corners-out, world/local=globe/crosshair, snap=magnet, show-flags=eye, perf=speedometer, search=magnifying-glass, mesh=cube, light=lightbulb, audio=speaker-high, script=code, camera=video-camera, terrain=mountains, particle=sparkle, folder=folder, play/pause/stop.

**MOTION:** `motion.fast`=0.12 (hover/active), foldout expand 0.15s ease-out, tab-drag ghost via egui_dock. egui `style.animation_time` from the token.

**ENFORCEMENT:** `PluginCtx` hands a plugin only `tokens`, `widgets`, `canvas`. There is no API to set `Visuals`, push a raw `Color32` panel fill, or create a top-level panel — so a Crucible or Forge panel is physically unable to diverge. Swapping `ochroma_dark.theme.json` → `ochroma_light.theme.json` restyles host and every plugin in one reload (proven by test 2).

## Migration Order

Each step ships AND wires (no "wire later") and is headless-provable; the bar never drops, phasing only orders the build. **The bitmap-font cap breaks at step 3.**

1. **Tokens + JSON spine + dual-consumer skin.** Add `tokens.rs` + `assets/ui/ochroma.theme.json` + `ochroma_light.theme.json`; add `egui_theme::apply` and `StyleSheet::from_tokens`; rewrite `apply_ochroma_theme` as a shim; deprecate `OchromaColors`. `editor.rs`/`main.rs`/UiTree all keep building and immediately gain the token theme. Proof: test 1, test 2.
2. **Icons + widget kit.** `icons::install` in the egui ctx; swap `entity_icon()` Unicode for phosphor; add `widgets.rs`. Inspector in `editor.rs` switches its `DragValue`s to `scrub_drag` and its flat label list to `foldout` sections + a search filter. Proof: tests 5, 10.
3. **Docking shell + bitmap-font eradication.** Introduce egui_dock; `editor.rs::show`'s fixed panel bodies move verbatim into `TabViewer` arms; chrome stays as surrounding bars; layout → `editor_layout.ron`. **Delete `engine_runner`'s software-compositor editor panels** (`burn_text` inspector/HUD, the node-graph blit). After this step the editor face has no bitmap font. Proof: tests 4, 6.
4. **Viewport-as-texture.** `shell/viewport.rs` registers the splat texture via `register_native_texture`, shows it as `Image`, adds the floating toolbar + gizmo overlay; the dock CentralPanel owns the frame. Proof: test 7.
5. **NodeCanvas.** `node_canvas/` replaces `node_graph_widget.rs`'s render path (data structs preserved as the `GraphModel` adapter); the Node Graph tab renders the existing `OchromaNodeGraph` with bezier/type-colored wires/grid/minimap. Proof: tests 8, 9.
6. **Command palette + Vello overlay.** `command_palette.rs` (Ctrl+Shift+P) over the Command registry; the **perf unitgraph, gizmo labels, and SpectralHUD stay on the Vello `render_to_rgba` path** reading the same `Tokens`. Proof: tests 11, 13.
7. **Plugin host + Crucible example (the worked end-to-end).** `shell/host.rs` defines `EditorPlugin`/`EditorHost`/`PluginCtx`. `plugins/crucible_editor.rs` wraps `crucible-nodes` (verified: `CrucibleNode` with `descriptor()` + `cook(PortMap) -> Result<PortMap, CookError>`) via a `ForeignNodeAdapter` carrying an explicit `PortType ↔ PortDataType` mapping table (any unmapped foreign type is a hard compile-visible TODO, never a silent `Any`); registers a dockable panel + a `GraphEditorDesc`. Proof: test 14.
8. **Forge plugin (second; corrected).** Forge has **no node trait** — verified `forge-cli` exposes `run(json) -> Result<ForgeVolume, String>` command fns (terrain/water/road/scatter/building/vegetation). `plugins/forge_editor.rs` adds a **generator→node synthesis layer**: one `OchromaNode` per Forge command, params from the command's JSON schema, `cook` calling `forge_cli::cmd::<domain>::run(json)` and emitting `ForgeVolume` + spectral reflectance on a typed `Terrain`/`Splats` output port; nodes color by `NodeCategory::Spatial`. FloraPrime (verified: a Python PyTorch model, not Rust) adopts later via a cook/inference bridge node, not a node-trait wrapper. Proof: a `forge_panel_styled_by_host` test mirroring test 14.

## API

```rust
// vox_ui::tokens
pub struct Tokens { /* color/space/radius/type_ramp/motion_ms/rem */ }
impl Tokens {
    pub fn default() -> Self;                          // == assets/ui/ochroma.theme.json
    pub fn load(path: &std::path::Path) -> std::io::Result<Tokens>;
    pub fn color(&self, dotted: &str) -> [u8; 4];
    pub fn wire_color(&self, t: vox_editor::node_graph::PortType) -> [u8; 4];
    pub fn category_header(&self, cat: NodeCategory) -> [u8; 4];
    pub fn rem(&self) -> f32;
}
pub enum NodeCategory { Generator, Spatial, Field, Sink, Math }

// vox_ui::egui_theme
pub fn apply(ctx: &egui::Context, t: &Tokens); // writes Visuals+Spacing+text_styles from tokens

// vox_ui::ui_tree (added — lets the Vello overlay share the spine)
impl StyleSheet { pub fn from_tokens(t: &Tokens) -> StyleSheet; }

// vox_ui::widgets (the shared kit plugins MUST use)
pub struct ScrubOpts { pub speed: f32, pub range: Option<std::ops::RangeInclusive<f32>>, pub suffix: &'static str }
pub fn scrub_drag(ui: &mut egui::Ui, value: &mut f32, opts: ScrubOpts) -> egui::Response;
pub fn foldout<R>(ui: &mut egui::Ui, id: egui::Id, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) -> Option<R>;
pub fn search_box(ui: &mut egui::Ui, query: &mut String) -> egui::Response;
pub fn icon_button(ui: &mut egui::Ui, icon: &str, tip: &str) -> egui::Response;

// vox_ui::node_canvas
pub trait GraphModel {
    fn nodes(&self) -> Vec<NodeView>;            // id, title, category, pos, size, ports
    fn wires(&self) -> Vec<WireView>;            // from(node,port)->to(node,port), exec flag
    fn port_type(&self, node: u64, port: &str) -> Option<vox_editor::node_graph::PortType>;
    fn move_node(&mut self, node: u64, pos: egui::Pos2);
    fn connect(&mut self, from: (u64,&str), to: (u64,&str)) -> Result<(), String>;
    fn wire_value_label(&self, from: (u64,&str)) -> Option<String>;
}
pub struct NodeCanvas { /* zoom, pan, grid, minimap state */ }
impl NodeCanvas {
    pub fn new() -> Self;
    pub fn ui(&mut self, ui: &mut egui::Ui, t: &Tokens, model: &mut dyn GraphModel) -> CanvasResponse;
    pub fn set_curvature(&mut self, c: f32);      // default 0.5
    pub fn set_wire_thickness(&mut self, t: f32); // default 4.0
}

// vox_ui::vello_ctx (overlay path; extended beyond FillRect for the perf graph + gizmo labels)
pub enum DrawCmd { FillRect{..}, StrokePath{ pts: Vec<[f32;2]>, width: f32, color: [f32;4] },
    FillPath{..}, Gradient{..}, Glyph{ x:f32, y:f32, text:String, size:f32, color:[u8;3] }, PushClip([f32;4]), PopClip }
impl VelloCtx {
    pub fn stroke_path(&mut self, pts: &[[f32;2]], width: f32, color: [f32;4]);
    pub fn render_to_rgba(&mut self) -> Result<Vec<[u8;4]>, String>; // the headless-proof path spectral_hud already uses
}

// vox_app::shell::host — THE PLUGIN CONTRACT (see Architecture for full signatures)
pub trait EditorPlugin { fn id(&self) -> &str; fn register(&mut self, host: &mut EditorHost); }

// CONCRETE PLUGIN — Crucible (the cheap worked example; CrucibleNode IS a node trait)
pub struct CruciblePlugin;
impl EditorPlugin for CruciblePlugin {
    fn id(&self) -> &str { "crucible" }
    fn register(&mut self, host: &mut EditorHost) {
        host.add_graph_editor(GraphEditorDesc {
            id: "crucible.cook".into(), title: "Crucible".into(),
            registry: Box::new(CrucibleRegistry::default()),       // wraps crucible-nodes via ForeignNodeAdapter
            model:    Box::new(CrucibleGraphModel::new()),         // CrucibleNode.cook(PortMap)->PortMap, PortDataType<->PortType table
        });
        host.add_panel(PanelDesc {
            id: "crucible.inspector".into(), title: "Crucible Cook".into(),
            icon: icon::TERRAIN, default_dock: Dock::RightBottom,
            build: Box::new(|ui, cx| {                              // styled ENTIRELY by host tokens
                vox_ui::widgets::foldout(ui, egui::Id::new("cr_field"), "Field", |ui| {
                    let mut amp = 80.0;
                    vox_ui::widgets::scrub_drag(ui, &mut amp,
                        vox_ui::widgets::ScrubOpts { speed: 0.5, range: Some(0.0..=500.0), suffix: "" });
                });
            }),
        });
        host.add_command(Command { id: "crucible.recook".into(), title: "Crucible: Recook".into(),
            icon: icon::TERRAIN, run: Box::new(|| {}) });
    }
}
// Crucible nodes color by NodeCategory::Spatial -> headers/wires match host terrain nodes automatically.
// Forge plugin (step 8) uses the SAME shape but wraps forge-cli run(json)->ForgeVolume command fns as one OchromaNode per domain.
```

## Wiring Table

| New/changed surface | Wired into | At |
|---|---|---|
| `Tokens` + `ochroma.theme.json` | `egui_theme::apply` + `StyleSheet::from_tokens` | step 1 |
| `apply_ochroma_theme` shim | called by `editor.rs`/`main.rs` (unchanged) | step 1 |
| `icons::install` + `widgets.rs` | `editor.rs` inspector + toolbar | step 2 |
| `EditorShell` (egui_dock) | replaces `editor.rs::show` fixed panel layout; panel bodies moved into `TabViewer` arms | step 3 |
| `engine_runner` software panels | **deleted**; `engine_runner` becomes a thin launcher of `EditorShell` + the Vello overlay | steps 3, 6 |
| `shell/viewport.rs` `register_native_texture` | the existing splat wgpu 24 texture → Viewport tab `Image` | step 4 |
| `NodeCanvas` over `GraphModel` | the Node Graph tab, driven by existing `OchromaNodeGraph` + live_cook | step 5 |
| `command_palette.rs` | the Command/menu/toolbar registry | step 6 |
| Vello overlay (perf/gizmo/HUD) | `render_to_rgba`, composited over the viewport texture, reading `Tokens` | step 6 |
| `EditorHost` / `EditorPlugin` / `PluginCtx` | `EditorShell::install_plugin` | step 7 |
| `CruciblePlugin` / `ForgePlugin` | `vox_app/src/plugins/`, behind a `--plugin` flag + cargo feature (isolates sibling-repo build state) | steps 7, 8 |

## Out of Scope

- Vello 0.5 / wgpu 24 unification of the two stacks (tracked end-state in `vello_ctx.rs`; the seam is documented and token-unified, not closed here).
- Text input / caret / IME beyond what egui provides natively (the command palette and inspector entry use egui's text widgets, not a new UiTree text-input engine).
- FloraPrime as a node-trait plugin (it is a Python PyTorch model; adoption is a later cook/inference bridge node).
- Retiring `vox_app/src/main.rs`'s separate egui editor binary (it converges onto `EditorShell` after step 3 but final deletion is a follow-on cleanup).
- Multi-viewport / floating OS windows for docked panels (egui_dock supports it; not gated by any checklist item).

## IMPORTANT NOTES (verified codebase constraints)

- **wgpu split is REAL (verified in Cargo.lock):** `wgpu 24.0.5` (engine, `vox_render/Cargo.toml: wgpu = "24"`) coexists with `wgpu 23.0.1` pulled by `vello 0.4.1`. egui/egui-wgpu are **0.31.1**, on wgpu 24 — so the editor shell has no version war. The Vello overlay is on wgpu 23. Their composite crosses CPU until vello 0.5 lands on wgpu 24. **Do not claim a unified GPU device before that bump.**
- **`register_native_texture` for item 19 is honest but not "GPUI-class":** the splat frame stays on GPU (no CPU readback for display), but the egui *composite* is egui-on-wgpu-24, not a diffed scene graph. Test 7 asserts the no-readback fact; do not overclaim.
- **NEW DEPS are NOT in Cargo.lock:** `egui_dock`, `egui-phosphor`, `egui_kittest` are absent. **Day-0 gate:** verify the exact egui-0.31-compatible versions compile *before* step 1. If `egui_dock 0.31` is unavailable, `egui_tiles` is the fallback (the `shell/mod.rs` API is small enough to swap). If `egui_kittest`'s wgpu feature won't align to wgpu 24, fall back to its CPU-tessellation snapshot for items 1-18 and prove item 19 with a standalone wgpu readback test.
- **Existing egui editor head-start (verified `crates/vox_app/src/editor.rs`, 1252 lines):** menu_bar L493, toolbar L591, status_bar L702, hierarchy SidePanel L735, inspector SidePanel L796 with `DragValue` scrub L819-830. `node_graph_widget.rs::show_egui` L467 **already** draws cubic bezier wires L512. Items 6/10/15/16 are **upgrades, not green-field**.
- **Theme baseline (verified `vox_ui/src/theme.rs`):** `apply_ochroma_theme` L4 writes literal RGB; `OchromaColors` L126; `AXIS_X/Y/Z` L142-144 = `(230,60,60)/(60,200,60)/(60,100,230)` → carried verbatim. `entity_icon()` returns single Unicode codepoints (placeholders).
- **UiTree baseline (verified `vox_ui/src/ui_tree.rs`):** `UiKind` L226 has exactly 6 variants (Panel/Label/Button/Slider/ProgressBar/Image) — NO Icon/TextInput/ScrollView/Foldout/NumericField. The overlay path needs only `StyleSheet::from_tokens`, NOT a full editor widget kit; the editor face is egui, not UiTree.
- **VelloCtx baseline (verified `vox_ui/src/vello_ctx.rs`):** GPU path currently renders ONLY `DrawCmd::FillRect` (L10). `StrokePath/FillPath/Glyph/Gradient/Clip` are net-new for the perf unitgraph and gizmo labels. `render_to_rgba` is the headless-proof path `spectral_hud.rs` (L242) already asserts via `vello_hud_px_last_frame`.
- **Node-trait parity (verified):** Ochroma `OchromaNode` (`vox_editor/src/node_graph.rs` L144): `descriptor()` L145, `set_param()` L146, `cook(NodeInputs) -> Result<NodeOutputs, NodeError>` L147. Crucible `CrucibleNode` (`~/src/crucible/rust/crates/crucible-nodes/src/*.rs`): `descriptor()` + `cook(PortMap) -> Result<PortMap, CookError>`. Near-exact mirror; the divergences are the enum (`PortType` vs `PortDataType`) and the cook signature (`NodeInputs/NodeOutputs/NodeError` vs `PortMap/CookError`) — handled by an explicit mapping table in `ForeignNodeAdapter`, with unmapped types as compile-visible TODOs.
- **Forge has NO node trait (verified — corrects all three input designs):** `~/src/aetherspectra/forge/crates/forge-cli/src/cmd/{terrain,water,road,scatter,building,vegetation}.rs` expose `pub fn run(json: &str) -> Result<ForgeVolume, String>`. The Forge plugin needs a **generator→node synthesis layer** (one `OchromaNode` per command), not a thin descriptor adapter. Use **Crucible as the worked example**; Forge is the second plugin.
- **FloraPrime is Python (verified):** `~/src/aetherspectra/floraprime_gen/` is a PyTorch model (`model/`, `train.py`, `spectral/`, `foliage/`) — not Rust, no node trait. Out of scope for the first plugin wave; later cook/inference bridge.
- **CLAUDE.md engine purity:** the plugin contract lives in `vox_app` (game/host layer); `vox_core`/`vox_data`/`vox_render` gain no game/tool concepts. `EditorPlugin` takes only host types, holding the line. Crucible/Forge plugins are separate crates behind a `--plugin` flag + cargo feature so the core editor builds even when a sibling repo's Slang/Spectra build prereq is broken.
- **Live-cook must not regress:** step 3 relocates panel bodies but keeps `OchromaNodeGraph` live-cook logic byte-identical (only render call sites move); the existing live-recook smoke line stays as the regression guard.

## Phase 1 (first agent-night) — the before/after frame the user will SEE

Scope: **steps 1-3** (Tokens + JSON spine, icons + widget kit, docking shell + bitmap-font deletion). Phasing orders the build; the bar stays full-SOTA — this wave clears checklist items 1, 2, 3, 4, 5, 6, 15, 16 and the bitmap-font cap.

Deliverable, provable in one night:
- `assets/ui/ochroma.theme.json` + `tokens.rs` + `egui_theme::apply`; `apply_ochroma_theme` shim. (tests 1, 2)
- egui-phosphor installed; `entity_icon()` replaced; `widgets.rs` `scrub_drag`/`foldout`/`search_box`; inspector upgraded. (tests 5, 10)
- egui_dock shell; `editor.rs` panels relocated into dock tabs with menu/toolbar/status chrome; **`engine_runner`'s `burn_text` software panels deleted**. (tests 4, 6)

Human-visible before/after: the user runs `cargo run -p vox_app --bin shell_snapshot -- --shot /tmp/shell.png` and sees a **dark, tokenized, icon-led, drag-dockable editor** — menu bar, Phosphor icon toolbar, draggable tabbed panels (Hierarchy / Inspector / Viewport / Node Graph / Content), foldout searchable inspector with drag-scrub fields, status bar — with **zero bitmap glyphs** (test 4 enforces it). That single frame is the answer to "not what you expect from a 2026 SOTA game engine editor." The GPU viewport texture, NodeCanvas beziers, command palette, Vello perf overlay, and plugins land in subsequent waves at the same full bar.

## Ecosystem Adoption

**Shared design-system crate/module:** `vox_ui::tokens` (the `Tokens` struct) backed by `assets/ui/ochroma.theme.json`, plus `vox_ui::egui_theme`, `vox_ui::widgets`, `vox_ui::node_canvas`, and `vox_ui::design::icons`. This is the single token surface every ecosystem app and plugin consumes.

**Token surface (what an ecosystem app reads):** `Tokens::color("surface.2")`, `Tokens::wire_color(PortType::Terrain)`, `Tokens::category_header(NodeCategory::Spatial)`, the `space`/`radius`/`type_ramp`/`motion` scales, and `icon::*` codepoints. egui apps call `egui_theme::apply(ctx, &tokens)`; UiTree/Vello apps call `StyleSheet::from_tokens(&tokens)`. One JSON edit reskins all of them.

**Concrete Crucible adoption (real paths):** `vox_app/src/plugins/crucible_editor.rs` (NEW) implements `EditorPlugin`. Its `CrucibleRegistry` wraps each `CrucibleNode` from `~/src/crucible/rust/crates/crucible-nodes/src/{terrain,scatter,camera,usd_export,input_nodes}.rs` in a `ForeignNodeAdapter` mapping `crucible::PortDataType` ↔ `vox_editor::node_graph::PortType`; its `CrucibleGraphModel` drives the Crucible cook graph (`~/src/crucible/rust/crates/cook`) through the host `NodeCanvas`. The panel is built with `cx.widgets.foldout` + `cx.widgets.scrub_drag`, styled entirely by `cx.tokens` — so the Crucible editor is **byte-for-byte the same look as the host** (test 14: panel bg == `surface.2`, headers == `category_header(Spatial)`).

**Concrete Forge adoption (corrected, real paths):** `vox_app/src/plugins/forge_editor.rs` (NEW) synthesizes one `OchromaNode` per Forge command function in `~/src/aetherspectra/forge/crates/forge-cli/src/cmd/{terrain,water,road,scatter,building,vegetation}.rs` (each `run(json) -> Result<ForgeVolume, String>`), params from the command's JSON schema, `cook` emitting `ForgeVolume` + spectral reflectance on a typed output port → `Splatize`. Nodes color by `NodeCategory::Spatial`, so Forge wires/headers match host terrain nodes automatically. FloraPrime (`~/src/aetherspectra/floraprime_gen`, Python/PyTorch) adopts later as a cook/inference bridge node, consuming the same tokens.

---

Files cited (absolute): `/home/tom-espen/src/ochroma/crates/vox_ui/src/theme.rs`, `/home/tom-espen/src/ochroma/crates/vox_ui/src/ui_tree.rs`, `/home/tom-espen/src/ochroma/crates/vox_ui/src/vello_ctx.rs`, `/home/tom-espen/src/ochroma/crates/vox_ui/src/node_graph_widget.rs`, `/home/tom-espen/src/ochroma/crates/vox_app/src/editor.rs`, `/home/tom-espen/src/ochroma/crates/vox_app/src/bin/engine_runner.rs`, `/home/tom-espen/src/ochroma/crates/vox_editor/src/node_graph.rs`, `/home/tom-espen/src/ochroma/Cargo.lock`, `/home/tom-espen/src/crucible/rust/crates/crucible-nodes/src/`, `/home/tom-espen/src/aetherspectra/forge/crates/forge-cli/src/cmd/`, `/home/tom-espen/src/aetherspectra/floraprime_gen/`.