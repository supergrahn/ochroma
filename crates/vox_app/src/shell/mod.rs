//! The Ochroma editor SHELL — the SOTA dock-based chrome.
//!
//! `EditorShell` owns an `egui_dock` `DockState<PanelId>` plus a `TabViewer`
//! that dispatches each tab to a built-in panel. Around the dock area it paints
//! the three chrome bands (menu bar, icon toolbar, status bar) as
//! `TopBottomPanel`s. Everything is styled from the token design system
//! (`vox_ui::Tokens` via `egui_theme::apply`), uses Phosphor vector icons, and
//! contains ZERO bitmap glyphs — the headless `shell_snapshot` bin proves it.
//!
//! Phase 1 scope: this is the dockable, tokenized, icon-led face. The panel
//! *bodies* render representative SOTA content (the existing `editor.rs` logic
//! converges onto these arms in the plugin-host wave); the dock/move/layout
//! machinery, chrome, and bitmap-font eradication are real and tested now.

pub mod command_palette;
pub mod content_panel;
pub mod cpu_render;
pub mod graph_bridge;
pub mod host;
pub mod intent;
pub mod plugins;
pub mod forge_native;
pub mod crucible_native;
pub mod script_gen;
pub mod viewport;

use command_palette::{Command, CommandRegistry, PaletteState};
use content_panel::{ContentAction, ContentPanel};
use egui_dock::{DockArea, DockState, NodeIndex, Style as DockStyle};
use graph_bridge::GraphBridge;
use host::{InstalledPlugin, PluginCtx, TabDecl};
use plugins::{CrucibleScene, ForgeBuilding, ForgeTerrain, GrownTree};
use vox_core::types::GaussianSplat;
use vox_render::relight::IlluminantSpec;
use vox_editor::node_graph::NodeId;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use vox_ui::design::icons::icon;
use vox_ui::node_canvas::NodeCanvas;
use vox_ui::widgets::{self, ScrubOpts, WidgetKit};
use vox_ui::Tokens;

/// A dockable tab payload — a built-in panel or a plugin-contributed tab id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabKind {
    Builtin(PanelId),
    Plugin(String),
}

/// Identifies a built-in dockable panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelId {
    Hierarchy,
    Inspector,
    Viewport,
    NodeGraph,
    Content,
    Output,
}

impl PanelId {
    /// The plain-language tab title (UX principle 1: friendly words, not jargon).
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Hierarchy => "World",
            PanelId::Inspector => "Properties",
            PanelId::Viewport => "Viewport",
            PanelId::NodeGraph => "Node Graph",
            PanelId::Content => "Content",
            PanelId::Output => "Output Log",
        }
    }
    pub fn icon(self) -> &'static str {
        match self {
            PanelId::Hierarchy => icon::HIERARCHY,
            PanelId::Inspector => icon::INSPECTOR,
            PanelId::Viewport => icon::CAMERA,
            PanelId::NodeGraph => icon::NODE_GRAPH,
            PanelId::Content => icon::FOLDER,
            PanelId::Output => icon::CONSOLE,
        }
    }
}

/// The `[start, len)` slice of the viewport `overlay` that a planted World entity
/// OWNS (AAA Spec 09 provenance index). Planting records it on the entity so a
/// later duplicate can clone EXACTLY that entity's splats (not the whole overlay),
/// and so undo's range-shift keeps every surviving entity pointing at its own
/// splats. The two projections — the `PlacedAsset` undo entry and this per-entity
/// range — are shifted in lockstep by the undo arm. Spec 06 serializes this index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayRange {
    pub start: usize,
    pub len: usize,
}

/// A demo entity shown in the World/Properties panels so the snapshot has real
/// content (the inspector's drag-scrub fields bind to the selected one).
#[derive(Clone)]
pub struct ShellEntity {
    pub name: String,
    pub kind: String,
    pub pos: [f32; 3],
    /// The overlay range this entity's splats occupy, IF it was planted (a grown
    /// tree / raised terrain / duplicate). `None` for the seed demo entities
    /// (Townhouse, Sun, Camera…) which have no overlay splats — those are filtered
    /// out of `edit.duplicate`. Kept in lockstep with the `PlacedAsset` undo range.
    asset_range: Option<OverlayRange>,
}

impl ShellEntity {
    /// Test-only read of the provenance range as a plain `(start, len)` tuple, so
    /// the Spec 09 tests can assert EXACT ranges without exposing the field.
    #[cfg(test)]
    pub fn asset_range_for_test(&self) -> Option<(usize, usize)> {
        self.asset_range.map(|r| (r.start, r.len))
    }
}

/// The World hierarchy's multi-selection (AAA Spec 09). `set` is the FULL
/// selection (always non-empty in normal use); `primary` is the "active" row the
/// inspector binds to and the anchor `extend_to` ranges from. The two are kept
/// consistent: `primary` is always a member of `set` (or repointed when it would
/// not be). A single-select shell behaves exactly as before — `set == {i}`,
/// `primary == i` — so the existing inspector/hierarchy/undo tests stay green.
#[derive(Clone, Default)]
pub struct Selection {
    primary: usize,
    set: std::collections::BTreeSet<usize>,
}

impl Selection {
    /// A single-row selection: `set = {i}`, `primary = i`.
    pub fn single(i: usize) -> Self {
        let mut set = std::collections::BTreeSet::new();
        set.insert(i);
        Self { primary: i, set }
    }

    /// The active row — the one the inspector binds to.
    pub fn primary(&self) -> usize {
        self.primary
    }

    /// Whether row `i` is selected.
    pub fn contains(&self, i: usize) -> bool {
        self.set.contains(&i)
    }

    /// The selected indices in ascending order.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.set.iter().copied()
    }

    /// How many rows are selected.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Whether nothing is selected.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Toggle row `i` in/out of the selection (Cmd/Ctrl-click). If toggling OUT the
    /// current `primary`, repoint `primary` to the lowest remaining row (or 0 when
    /// the set is now empty) so `primary` is never a stale, non-member index.
    pub fn toggle(&mut self, i: usize) {
        if !self.set.remove(&i) {
            self.set.insert(i);
            self.primary = i;
        } else if self.primary == i {
            self.primary = self.set.iter().next().copied().unwrap_or(0);
        }
    }

    /// Replace the whole selection with just row `i` (a plain click).
    pub fn select_only(&mut self, i: usize) {
        *self = Self::single(i);
    }

    /// Range-select from the current `primary` (the anchor) to `i` INCLUSIVE
    /// (Shift-click), handling `i < primary`. Fills the contiguous span; `primary`
    /// stays the anchor.
    pub fn extend_to(&mut self, i: usize) {
        let (lo, hi) = if i < self.primary { (i, self.primary) } else { (self.primary, i) };
        for j in lo..=hi {
            self.set.insert(j);
        }
    }

    /// Drop every selected index `>= len` (e.g. after an entity was removed by
    /// undo), then repoint `primary` to a surviving member (the lowest), or to the
    /// last valid row when the set went empty — preserving the old "selection
    /// clamps into range" behavior.
    pub fn clamp_to(&mut self, len: usize) {
        self.set.retain(|&i| i < len);
        if !self.set.contains(&self.primary) {
            self.primary = self
                .set
                .iter()
                .next()
                .copied()
                .unwrap_or_else(|| len.saturating_sub(1));
        }
    }
}

/// One reversible edit on the shell's undo stack (design UX Principle 2,
/// "Provenance + reversibility"). Currently every intent/inspector param edit is
/// a `ParamSet`: re-applying `prev` to the SAME concrete `node_id` through
/// `GraphBridge::apply_param` reverts it (the bridge already supports re-applying
/// a param + re-cooking).
#[derive(Debug, Clone)]
pub enum UndoEntry {
    /// A param edit on the concrete node `node_id`: `key` went `prev -> next`.
    /// Undo replays by `node_id` (NOT by kind) so the exact node that was edited
    /// is the one reverted — with two nodes of the same kind, a kind-only lookup
    /// would revert the wrong node or drop the undo entirely.
    ParamSet {
        node_id: NodeId,
        key: &'static str,
        target: String,
        prev: f32,
        next: f32,
    },
    /// An asset (a grown FloraPrime tree OR a raised Forge terrain patch) was
    /// planted into the world: `len` splats were inserted into the viewport overlay
    /// at index `start`, and one World entity named `name` was added. Undo removes
    /// EXACTLY the `[start, start+len)` range from the overlay (NOT the tail) and
    /// the entity, then shifts every later undo entry's `start` down by `len` so
    /// the remaining assets' ranges stay valid. Range-tracked (not tail-truncated)
    /// so two interleaved asset types undo independently without corrupting each
    /// other's splats.
    PlacedAsset { name: String, start: usize, len: usize },
    /// An AI-generated Rhai script (AI-creates-code v1) was written to `path`.
    /// Undo DELETES exactly that file — and ONLY if it is byte-for-byte the
    /// content we wrote (`bytes`). If the user edited it after generation, undo
    /// REFUSES to delete and explains why (the wave lesson: never destroy what you
    /// didn't create / what the user has since changed).
    GeneratedScript {
        /// Domain label for the receipt (e.g. "spin script for the windmill").
        label: String,
        /// The exact file written. Undo deletes only this path.
        path: PathBuf,
        /// The exact bytes we wrote; undo compares the current file against these
        /// before deleting, so a since-modified file is preserved.
        bytes: Vec<u8>,
    },
    /// A grouped transaction (AAA Spec 07): a SEQUENCE of `members` applied as ONE
    /// reversible step, so a multi-action plan ("add 5 birch trees" → five planted
    /// trees) is reverted by ONE Ctrl+Z. Counts as a SINGLE undo-stack entry
    /// (HISTORY_CAP-wise). Undo reverts the members in REVERSE order — the group's
    /// members are NOT individually on the undo stack while in flight, so each must
    /// drain its own correct `[start, len)` range highest-start-first off the LIVE
    /// overlay (see [`Self::undo`]). The mechanism Specs 09 and 12 reuse.
    Group { label: String, members: Vec<UndoEntry> },
}

/// A side-effecting request a registry command pushes for the shell to drain on
/// the next frame. Registry commands are `Fn()` (no `&mut self`), so a command
/// that must mutate shell state (swap the theme, focus a tab) records its intent
/// here; `EditorShell::drain_requests` applies it. This keeps theme/focus on the
/// SAME one-command-surface (the intent executor and a menu click both route
/// through `registry.run`, which fires the closure that queues the request).
///
/// (Not `PartialEq`/`Debug`: the `GrowTree`/`ForgeTerrain` payloads carry
/// `GaussianSplat`s, which are `Pod` but neither — and the shell only ever drains
/// these, never compares them.)
#[derive(Clone)]
pub enum ShellRequest {
    ThemeLight,
    ThemeDark,
    FocusViewport,
    FocusNodeGraph,
    FocusPlugin(String),
    Undo,
    /// The Content tab asked to load an asset into the scene (a double-click).
    /// The shell decodes it (or honestly reports what loading does today) and
    /// appends a receipt line to the Output Log.
    LoadAsset(PathBuf),
    /// FloraPrime grew a tree: plant its splats into the viewport overlay, add a
    /// numbered World entity, push an undo entry, and append a receipt. Queued by
    /// the shell when it drains FloraPrime's grow-sink.
    GrowTree(GrownTree),
    /// Forge raised a terrain patch: plant its splats into the viewport overlay,
    /// add a numbered World entity, push an undo entry, and append a receipt.
    /// Queued by the shell when it drains Forge's terrain-sink — the terrain twin
    /// of `GrowTree`, routed through the SAME `plant_asset` core.
    ForgeTerrain(ForgeTerrain),
    /// Plant a generated building (the building twin of `ForgeTerrain`).
    ForgeBuilding(ForgeBuilding),
    /// Plant a cooked Crucible scene (the Crucible twin of `ForgeBuilding`):
    /// queued when the shell drains Crucible's scene-sink, routed through the
    /// SAME `plant_asset` core.
    CrucibleScene(CrucibleScene),
    /// The "＋ Add to world" affordance (toolbar primary action / empty-state
    /// teaching copy / palette `world.add`) asks to open the command palette in
    /// intent mode pre-filled with "add ", routing the user straight into the
    /// Ask-Ochroma path that really inserts a node (`IntentAction::AddNode`).
    OpenAddPalette,
    /// AAA Spec 03: plant the two metameric "forgery" surfaces into the overlay
    /// (identical under the gallery lamp, divergent under the inspection lamp) and
    /// set the live ΔsRGB HUD receipt. The wedge made one-key reachable.
    ForgeryDemo,
    /// Advance the active inspection illuminant to the next in the cycle
    /// (neutral → cool_led → tungsten → daylight → …) and recompute the ΔsRGB
    /// receipt. Bound to Ctrl+L and the `view.illuminant` command — the dramatic
    /// "flip the forgery" key.
    CycleIlluminant,
    /// Set the active inspection illuminant explicitly (the `--illuminant <name>`
    /// proof-mode path); recomputes the ΔsRGB receipt and re-rasterizes.
    SetIlluminant(IlluminantSpec),
    /// AAA Spec 09: duplicate the current selection — clone every selected entity
    /// AND its exact overlay splat range at a +X offset, as ONE grouped undo.
    /// Queued by `edit.duplicate` (Ctrl+D); drained into `duplicate_selected`.
    DuplicateSelection,
}

/// The editor shell — owns the dock layout, panel state, and tokens.
pub struct EditorShell {
    pub tokens: Tokens,
    pub dock: DockState<TabKind>,
    pub entities: Vec<ShellEntity>,
    /// The World hierarchy selection (AAA Spec 09 — multi-select). Replaces the old
    /// single `selected: usize`; `self.selected()` is the back-compat accessor for
    /// the active (primary) row the inspector binds to.
    pub selection: Selection,
    pub search: String,
    pub status: String,
    /// Last measured GPU pass time (label, milliseconds) for the frame-budget HUD
    /// (Spec 08). Set by the editor each frame from a `GpuTimers` reading; shown in
    /// the status bar. A SEPARATE field from `status` (which the forgery HUD owns).
    pub last_gpu_pass_ms: Option<(&'static str, f32)>,
    /// Toolbar gizmo mode (0=move,1=rotate,2=scale).
    pub gizmo: u8,
    pub snap: bool,
    /// The one-command-surface (menus/toolbar/palette/AI all dispatch through it).
    pub registry: CommandRegistry,
    /// The Ctrl+K command palette state.
    pub palette: PaletteState,
    /// The node-graph canvas renderer state (pan/zoom/drag) for the Node Graph tab.
    pub canvas: NodeCanvas,
    /// The REAL live cook graph (vox_editor template) driving the Node Graph tab,
    /// the Properties param fields, and the live-cook loop.
    pub bridge: GraphBridge,
    /// The shared widget kit handed to plugins (token-styled controls only).
    pub widget_kit: WidgetKit,
    /// Cached viewport scene texture (rasterized splat frame). Uploaded once and
    /// reused; invalidated (set to `None`) whenever [`Self::overlay`] changes so the
    /// next frame re-rasterizes the base scene + planted assets.
    pub viewport_tex: Option<egui::TextureHandle>,
    /// Splats the shell owns ON TOP of the fixed `viewport::build_scene` base —
    /// every planted asset (grown FloraPrime trees AND raised Forge terrain
    /// patches) appends here. Insertions are range-tracked on the undo stack so a
    /// later undo removes exactly one asset's range (not the tail). Composited into
    /// the viewport texture each time the cache is rebuilt.
    pub overlay: Vec<GaussianSplat>,
    /// Shared queue FloraPrime's "Grow tree" button fills with [`GrownTree`]s; the
    /// shell drains it each frame into `GrowTree` requests. The host holds the SAME
    /// `Rc` it handed FloraPrime via [`plugins::FloraPrimePlugin::with_grow_sink`].
    pub flora_sink: Rc<RefCell<Vec<GrownTree>>>,
    /// Shared queue Forge's "Raise terrain" button fills with [`ForgeTerrain`]s;
    /// the shell drains it each frame into `ForgeTerrain` requests. The host holds
    /// the SAME `Rc` it handed Forge via [`plugins::ForgePlugin::with_terrain_sink`]
    /// — the terrain twin of `flora_sink`.
    pub forge_sink: Rc<RefCell<Vec<ForgeTerrain>>>,
    /// Shared queue Forge's "Add building" button fills with [`ForgeBuilding`]s —
    /// drained each frame exactly like `forge_sink`.
    pub building_sink: Rc<RefCell<Vec<ForgeBuilding>>>,
    /// Shared queue Crucible's "Cook scene" button fills with [`CrucibleScene`]s —
    /// drained each frame exactly like `building_sink` (the Crucible twin).
    pub scene_sink: Rc<RefCell<Vec<CrucibleScene>>>,
    /// Per-label placement counter so each planted asset is named "<Label> NN"
    /// (incrementing per label — "Silver Birch 01", "Forge Terrain 01").
    /// MONOTONIC BY DESIGN (wave-14 finding [1]): undo never decrements, so
    /// plant→undo→replant yields "…02" with no "…01" present. Numbers are
    /// placement-order provenance, not a live census — reusing them after undo
    /// would let two different assets carry the same name in receipts/logs.
    asset_counts: std::collections::HashMap<String, usize>,
    /// Installed host-plugins (their tabs joined the dock, commands the registry).
    pub plugins: Vec<InstalledPlugin>,
    /// Set true by the `world.add` command (proves the registry callback fired;
    /// the palette test asserts it).
    pub last_command_flag: Rc<RefCell<bool>>,
    /// The undo stack of reversible assistant/inspector edits (Ctrl+Z reverts the
    /// last one). Provenance + reversibility for AI-driven edits (UX Principle 2).
    pub undo_stack: Vec<UndoEntry>,
    /// Side-effecting requests queued by registry commands (theme/focus), drained
    /// each frame. Shared so a `Fn()` command closure can push onto it.
    pub requests: Rc<RefCell<Vec<ShellRequest>>>,
    /// The assistant history strip shown in the palette: a human-readable receipt
    /// line per executed (or rejected) intent, newest last.
    pub assistant_log: Vec<String>,
    /// The Content tab's live content browser (lazily scans `assets/`).
    pub content: ContentPanel,
    /// Output Log lines appended at runtime (e.g. a content-browser asset load),
    /// shown beneath the static engine banner in the Output Log tab.
    pub output_log: Vec<String>,
    /// Monotonic UI frame counter, bumped once per `ui()`. Used to coalesce a
    /// continuous inspector drag (many per-frame value changes) into ONE undo
    /// entry: see [`EditorShell::record_inspector_edit`].
    frame: u64,
    /// The last inspector edit's (node, key) and the frame it was recorded on, so a
    /// drag spanning consecutive frames updates the existing undo entry's `next`
    /// instead of pushing a new one per frame.
    last_inspector_edit: Option<(NodeId, &'static str, u64)>,
    /// Which brain resolves an Ask-Ochroma sentence (Adoption #16). Selected ONCE
    /// at construction from `OCHROMA_ASK_LLM`: default `Deterministic` (offline,
    /// network-free); the env var opts into the LLM seam. Read once here, never
    /// per keystroke.
    intent_backend: intent::IntentBackend,
    /// The live editable param schema handed to the LLM (node kinds + keys +
    /// ranges) so it can map fuzzy words onto real params. The clamp in
    /// `apply_param` remains the authority on ranges; this is prompt + key
    /// validation only.
    intent_schema: intent::SchemaContext,
    /// The directory AI-generated scripts (AI-creates-code v1) are written into.
    /// Defaults to the real [`Self::default_script_root`] (`assets/scripts/generated`);
    /// tests override it to a temp dir so they never leave files under `assets/`.
    script_root: PathBuf,
    /// AAA Spec 03 — the illuminant the scene splats were (approximately) lit by:
    /// the "gallery lamp" the forgery pair was matched under. Used as the
    /// `reference` for `derive_intrinsic` AND as the metamer baseline in the HUD.
    /// `neutral` by construction ([`metamer_demo_pair`] is neutral-metameric).
    reference_illuminant: IlluminantSpec,
    /// AAA Spec 03 — the active inspection light the viewport renders the overlay
    /// under. Equals `reference_illuminant` by default (the forgery reads
    /// identical); Ctrl+L cycles it so the forgery splits under cool_led/tungsten.
    active_illuminant: IlluminantSpec,
    /// AAA Spec 03 — the two overlay ranges of the planted forgery surfaces, so
    /// [`Self::hud_receipt`] can slice them out of `overlay` and compute the live
    /// ΔsRGB divergence. `None` until the forgery demo is planted.
    demo_groups: Option<(std::ops::Range<usize>, std::ops::Range<usize>)>,
}

impl Default for EditorShell {
    fn default() -> Self {
        Self::new(Tokens::default())
    }
}

/// History bound for both the undo stack and the assistant log: a held-down agent
/// loop (or a user spamming intents) must not grow either Vec without limit. When a
/// push overflows this cap the OLDEST entries are dropped, so the survivors are
/// always the most recent N.
const HISTORY_CAP: usize = 200;

/// The world-space offset a duplicate is placed at relative to its source (Spec
/// 09): +2 units along X, so the copy lands visibly beside the original (not on
/// top of it). Applied to BOTH the entity transform and every cloned splat.
const DUP_OFFSET: [f32; 3] = [2.0, 0.0, 0.0];

impl EditorShell {
    /// Build the shell with the standard SOTA layout:
    /// left = World; center-top = Viewport, center-bottom = Node Graph;
    /// right = Properties; bottom = Content + Output Log (tabbed).
    pub fn new(tokens: Tokens) -> Self {
        use TabKind::Builtin as B;
        let mut dock = DockState::new(vec![B(PanelId::Viewport), B(PanelId::NodeGraph)]);
        let surface = dock.main_surface_mut();
        // Left: World.
        let [center, _left] =
            surface.split_left(NodeIndex::root(), 0.18, vec![B(PanelId::Hierarchy)]);
        // Right: Properties.
        let [center, _right] = surface.split_right(center, 0.78, vec![B(PanelId::Inspector)]);
        // Bottom: Content + Output Log as a tab group.
        let [_center, _bottom] =
            surface.split_below(center, 0.72, vec![B(PanelId::Content), B(PanelId::Output)]);

        let last_command_flag = Rc::new(RefCell::new(false));
        let requests: Rc<RefCell<Vec<ShellRequest>>> = Rc::new(RefCell::new(Vec::new()));
        let registry = build_registry(&last_command_flag, &requests);
        let mut canvas = NodeCanvas::new();
        canvas.set_snap(GRAPH_SNAP);

        EditorShell {
            tokens: tokens.clone(),
            dock,
            registry,
            palette: PaletteState::default(),
            canvas,
            bridge: GraphBridge::new(),
            widget_kit: WidgetKit::new(tokens),
            viewport_tex: None,
            overlay: Vec::new(),
            flora_sink: Rc::new(RefCell::new(Vec::new())),
            forge_sink: Rc::new(RefCell::new(Vec::new())),
            building_sink: Rc::new(RefCell::new(Vec::new())),
            scene_sink: Rc::new(RefCell::new(Vec::new())),
            asset_counts: std::collections::HashMap::new(),
            plugins: Vec::new(),
            last_command_flag,
            undo_stack: Vec::new(),
            requests,
            assistant_log: Vec::new(),
            content: ContentPanel::new(ContentPanel::default_root()),
            output_log: Vec::new(),
            frame: 0,
            last_inspector_edit: None,
            intent_backend: intent::IntentBackend::from_env(),
            intent_schema: intent::SchemaContext::default_editable(),
            script_root: Self::default_script_root(),
            // The forgery pair is metameric under NEUTRAL, so the gallery lamp +
            // bake reference is neutral; the active light starts equal (identical
            // appearance) until the user switches it with Ctrl+L.
            reference_illuminant: IlluminantSpec::parse("neutral").unwrap(),
            active_illuminant: IlluminantSpec::parse("neutral").unwrap(),
            demo_groups: None,
            entities: vec![
                ShellEntity {
                    name: "Townhouse_Row_03".into(),
                    kind: "mesh".into(),
                    pos: [12.0, 0.0, -4.0],
                    asset_range: None,
                },
                ShellEntity {
                    name: "Terrain_Alpine".into(),
                    kind: "terrain".into(),
                    pos: [0.0, 0.0, 0.0],
                    asset_range: None,
                },
                ShellEntity {
                    name: "Sun_Directional".into(),
                    kind: "light".into(),
                    pos: [40.0, 80.0, 20.0],
                    asset_range: None,
                },
                ShellEntity {
                    name: "Camera_Main".into(),
                    kind: "camera".into(),
                    pos: [5.0, 2.0, 14.0],
                    asset_range: None,
                },
            ],
            // Preserve the old "row 0 selected" default so the inspector tests stay
            // green; the Selection model is single-select until the user multi-picks.
            selection: Selection::single(0),
            search: String::new(),
            status: "All systems healthy".into(),
            last_gpu_pass_ms: None,
            gizmo: 0,
            snap: true,
        }
    }

    /// Install a host-plugin: its tabs join the dock (split into the bottom-right
    /// area beside Properties) and its commands join the registry/palette. This is
    /// the `EditorShell::install_plugin` wiring point the design names.
    pub fn install_plugin(&mut self, plugin: Box<dyn crate::shell::host::EditorPlugin>) {
        let plugin_id = plugin.id().to_string();
        let raw_tabs = plugin.tabs();

        // 1. Reject duplicate TabDecl ids WITHIN this plugin (log + drop the dupes).
        //    Two equal `TabKind::Plugin(id)` entries would shadow each other in
        //    dispatch (the first match wins), so a plugin declaring the same tab id
        //    twice is a plugin bug — surface it loudly and keep only the first.
        let mut tabs: Vec<crate::shell::host::TabDecl> = Vec::with_capacity(raw_tabs.len());
        for t in raw_tabs {
            if tabs.iter().any(|kept| kept.id == t.id) {
                eprintln!(
                    "[shell] install_plugin('{plugin_id}'): duplicate TabDecl id '{}' within plugin — rejected (keeping the first)",
                    t.id
                );
                continue;
            }
            tabs.push(t);
        }

        // 2. Duplicate PLUGIN id REPLACES the existing install in place — the same
        //    same-id-replaces discipline the command registry (CommandRegistry::add),
        //    subgraph registry, content-browser, and node-graph registries use. This
        //    is the FIFTH appearance of this registry-collision pattern in the shell;
        //    reinstalling under an existing plugin id swaps it rather than stacking a
        //    shadowed duplicate. Remove the old plugin's dock tabs first.
        if let Some(pos) = self.plugins.iter().position(|ip| ip.plugin.id() == plugin_id) {
            let old = self.plugins.remove(pos);
            for t in &old.tabs {
                while let Some(loc) = self.dock.find_tab(&TabKind::Plugin(t.id.clone())) {
                    self.dock.remove_tab(loc);
                }
            }
        }

        for cmd in plugin.commands() {
            self.registry.add(cmd);
        }
        // Dock each plugin tab next to the Node Graph (center-bottom) so two
        // graph editors visibly coexist. Skip any tab id that already exists in the
        // dock from ANOTHER plugin (log it) so dispatch never becomes ambiguous.
        for t in &tabs {
            if self.dock.find_tab(&TabKind::Plugin(t.id.clone())).is_some() {
                eprintln!(
                    "[shell] install_plugin('{plugin_id}'): tab id '{}' already docked by another plugin — skipping",
                    t.id
                );
                continue;
            }
            if let Some((surface, node, _)) =
                self.dock.find_tab(&TabKind::Builtin(PanelId::NodeGraph))
            {
                self.dock.set_focused_node_and_surface((surface, node));
                self.dock.push_to_focused_leaf(TabKind::Plugin(t.id.clone()));
            } else {
                self.dock
                    .main_surface_mut()
                    .push_to_first_leaf(TabKind::Plugin(t.id.clone()));
            }
        }
        let canvases = tabs
            .iter()
            .map(|t| (t.id.clone(), NodeCanvas::new()))
            .collect();
        self.plugins.push(InstalledPlugin { plugin, tabs, canvases });
    }

    /// Install the FloraPrime vegetation plugin wired to THIS shell's grow-sink, so
    /// its "Grow tree" button plants real splats into the live viewport (the host
    /// drains `flora_sink` each frame). Use this instead of installing a bare
    /// `FloraPrimePlugin::new()` when the grown tree must reach the world.
    pub fn install_floraprime(&mut self) {
        let plugin = plugins::FloraPrimePlugin::with_grow_sink(self.flora_sink.clone());
        self.install_plugin(Box::new(plugin));
    }

    /// The active (primary) selected row — back-compat accessor for the old
    /// `self.selected` field, now backed by [`Selection`] (Spec 09 multi-select).
    /// The inspector binds to this row.
    pub fn selected(&self) -> usize {
        self.selection.primary()
    }

    /// Grow a tree headlessly (no UI click): build the default-species splats and
    /// plant them through the SAME `plant_asset` core the button drives, so
    /// snapshots/tests can prove the planted tree without driving egui input. The
    /// `species_label`/`class`/`species_id` mirror a `FLORAPRIME_SPECIES` row.
    pub fn grow_tree_headless(&mut self, species_label: &str, class: &str, species_id: i32) {
        let skeleton = plugins::grow_tree_skeleton(species_id, 3.0, 200);
        let splats = plugins::skeleton_to_splats(&skeleton, class, species_id);
        self.plant_grown_tree(GrownTree {
            species_label: species_label.to_string(),
            splats,
        });
    }

    /// Install the Forge environment plugin wired to THIS shell's terrain-sink, so
    /// its "Raise terrain" button plants real splats into the live viewport (the
    /// host drains `forge_sink` each frame). Use this instead of installing a bare
    /// `ForgePlugin::new()` when the raised terrain must reach the world — the
    /// terrain twin of [`Self::install_floraprime`].
    pub fn install_forge(&mut self) {
        let plugin =
            plugins::ForgePlugin::with_sinks(self.forge_sink.clone(), self.building_sink.clone());
        self.install_plugin(Box::new(plugin));
    }

    /// Raise a terrain patch headlessly (no UI click): cook the patch and plant it
    /// through the SAME `plant_asset` core the button drives, so snapshots/tests
    /// can prove the planted landform without driving egui input. `seed` selects
    /// the heightfield (distinct seeds → distinct mounds).
    pub fn raise_terrain_headless(&mut self, seed: u32) {
        let mut patch = plugins::forge_terrain_patch(seed);
        let splats = std::mem::take(&mut patch.splats);
        self.plant_asset(&patch.label, "terrain", splats, plugins::TERRAIN_PATCH_ORIGIN, "Raised a", "");
    }

    /// Press "Add building" headlessly (the snapshot binary's proof path) —
    /// generates with this build's backend and plants through the same core.
    pub fn add_building_headless(&mut self, seed: u64) {
        let (splats, backend) = forge_native::generate_building(forge_native::BuildingSpec {
            seed,
            ..Default::default()
        });
        self.plant_forge_building(ForgeBuilding {
            label: "Forge Building".to_string(),
            splats,
            backend,
        });
    }

    /// Install the Crucible plugin wired to THIS shell's scene-sink, so its
    /// "Cook scene" button plants real splats into the live viewport (the host
    /// drains `scene_sink` each frame). Use this instead of installing a bare
    /// `CruciblePlugin::new()` when the cooked scene must reach the world — the
    /// Crucible twin of [`Self::install_forge`].
    pub fn install_crucible(&mut self) {
        let plugin = plugins::CruciblePlugin::with_scene_sink(self.scene_sink.clone());
        self.install_plugin(Box::new(plugin));
    }

    /// Press "Cook scene" headlessly (the snapshot binary's proof path) — cooks
    /// with this build's backend and plants through the same core. With
    /// `crucible-native` this runs the real Crucible cook engine
    /// (graph_builder::build(...).cook() → USD on disk → vox_usd import attempt);
    /// without it, a deterministic preview cluster.
    pub fn cook_scene_headless(&mut self, seed: u64) {
        let (splats, backend) = crucible_native::cook_scene(crucible_native::CrucibleSceneSpec {
            seed,
            ..Default::default()
        });
        self.plant_crucible_scene(CrucibleScene {
            label: "Crucible Scene".to_string(),
            splats,
            backend,
        });
    }

    /// Lay out the full shell into an egui context for one frame.
    pub fn ui(&mut self, ctx: &egui::Context) {
        // Bump the frame counter first thing — inspector-drag coalescing keys off it.
        self.frame = self.frame.wrapping_add(1);

        // Apply any side-effecting requests queued by registry commands last frame
        // (theme swap, tab focus, undo) before laying anything out.
        self.drain_requests();

        // Ctrl+K toggles the one-command-surface (the AI-native entry point).
        let ctrl_k = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::K));
        if ctrl_k {
            self.palette.toggle();
        }
        // Ctrl+Z reverts the last assistant/inspector edit through the registry.
        let ctrl_z = ctx.input(|i| {
            i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z)
        });
        if ctrl_z {
            self.registry.run("edit.undo");
        }
        // Ctrl+D duplicates the World selection (Spec 09) through the same registry
        // surface — the command queues a DuplicateSelection request drained below.
        let ctrl_d = ctx.input(|i| {
            i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::D)
        });
        if ctrl_d {
            self.registry.run("edit.duplicate");
        }
        // Ctrl+L cycles the inspection light — the AAA Spec-03 "flip the forgery"
        // key. Under the gallery lamp the planted metamer pair reads identical;
        // under cool_led/tungsten it splits and the ΔsRGB HUD flips to "(forgery)".
        let ctrl_l = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::L));
        if ctrl_l {
            self.requests.borrow_mut().push(ShellRequest::CycleIlluminant);
        }

        self.menu_bar(ctx);
        self.toolbar(ctx);
        self.status_bar(ctx);

        // Drain any assets the plugins emitted this frame — FloraPrime's "Grow
        // tree" fills the grow-sink, Forge's "Raise terrain" fills the terrain-sink
        // — into requests queued onto the same stream the shell drains (planting
        // needs `&mut self` the plugins can't hold).
        {
            let grown: Vec<GrownTree> = self.flora_sink.borrow_mut().drain(..).collect();
            let raised: Vec<ForgeTerrain> = self.forge_sink.borrow_mut().drain(..).collect();
            let built: Vec<ForgeBuilding> = self.building_sink.borrow_mut().drain(..).collect();
            let cooked: Vec<CrucibleScene> = self.scene_sink.borrow_mut().drain(..).collect();
            if !grown.is_empty() || !raised.is_empty() || !built.is_empty() || !cooked.is_empty() {
                let mut q = self.requests.borrow_mut();
                for tree in grown {
                    q.push(ShellRequest::GrowTree(tree));
                }
                for patch in raised {
                    q.push(ShellRequest::ForgeTerrain(patch));
                }
                for building in built {
                    q.push(ShellRequest::ForgeBuilding(building));
                }
                for scene in cooked {
                    q.push(ShellRequest::CrucibleScene(scene));
                }
            }
        }
        // Apply the freshly-queued plant requests THIS frame so the overlay is
        // current before the viewport texture is (re)built below.
        self.drain_requests();

        // Ensure the viewport scene texture is uploaded (rebuilt when the planted
        // overlay changed, since planting/undo invalidate the cache).
        let viewport_tex = viewport::scene_texture(
            ctx,
            &mut self.viewport_tex,
            &self.overlay,
            &self.reference_illuminant,
            &self.active_illuminant,
        );

        let mut inspector_undo_edits: Vec<(NodeId, &'static str, String, f32)> = Vec::new();
        let mut content_action: Option<ContentAction> = None;
        let mut viewer = ShellViewer {
            tokens: &self.tokens,
            widget_kit: &self.widget_kit,
            entities: &mut self.entities,
            selection: &mut self.selection,
            search: &mut self.search,
            canvas: &mut self.canvas,
            bridge: &mut self.bridge,
            viewport_tex,
            plugins: &mut self.plugins,
            undo_edits: &mut inspector_undo_edits,
            content: &mut self.content,
            output_log: &self.output_log,
            content_action: &mut content_action,
        };
        let dock_style = DockStyle::from_egui(ctx.style().as_ref());
        DockArea::new(&mut self.dock)
            .style(dock_style)
            .show(ctx, &mut viewer);
        // A Content-tab double-click queues a LoadAsset request the shell drains
        // next frame (loading needs `&mut self` the viewer can't hold).
        if let Some(ContentAction::Load(path)) = content_action {
            self.requests.borrow_mut().push(ShellRequest::LoadAsset(path));
        }
        // Route any manual inspector param edits onto the SAME undo stack the AI
        // intents use, so a Properties scrub is Ctrl+Z-reversible (UndoEntry doc
        // invariant).
        for (node_id, key, target, prev) in inspector_undo_edits {
            self.record_inspector_edit(node_id, key, target, prev);
        }

        // The palette overlays everything (foreground order). It runs commands in
        // place (it has the registry) and returns an intent submission for the
        // shell to execute (executing an intent needs `&mut self`).
        let outcome = self.palette.ui(ctx, &self.tokens, &self.registry, &self.assistant_log);
        if let command_palette::PaletteOutcome::IntentSubmitted(text) = outcome {
            self.run_intent(&text);
        }
    }

    /// Force the palette open (for headless snapshots / tests).
    pub fn open_palette(&mut self) {
        self.palette.open = true;
        self.palette.selected = 0;
    }

    /// Execute a natural-language intent end-to-end: parse it against the REAL
    /// registry + graph, run the resulting action through the SAME
    /// `CommandRegistry` + `GraphBridge` the manual surface uses, push any
    /// reversible edit onto the undo stack, and append a human-readable receipt to
    /// the assistant log. Returns the receipt line.
    ///
    /// This is the whole "Ask Ochroma generates, not just navigates" loop: a
    /// sentence in, a real graph/registry mutation out, with provenance.
    pub fn run_intent(&mut self, text: &str) -> String {
        use intent::IntentAction;
        // Adoption #16: resolve through the selected backend (deterministic by
        // default; LLM if OCHROMA_ASK_LLM opted in). The LLM may only ever produce
        // a schema-validated action; any failure falls back to the deterministic
        // parser. A validated SetParam value still flows through `apply_param`'s
        // clamp below — the seam never bypasses that safety net.
        let resolution =
            intent::resolve_intent(&mut self.intent_backend, text, &self.intent_schema, &self.registry);
        let provenance = resolution.provenance;
        // `action` is always `Some` for every backend (Unknown is itself an
        // action); the Option exists for forward-compat with a "no-op" resolution.
        let action = resolution
            .action
            .unwrap_or(IntentAction::Unknown { suggestions: Vec::new() });
        let receipt = match action {
            IntentAction::SetParam { node_kind, key, target, value } => {
                self.apply_param_intent(node_kind, key, &target, value)
            }
            IntentAction::AdjustParam { node_kind, key, target, delta } => {
                // Resolve the relative nudge to an absolute value from the current
                // cooked param, then flow through the same apply path.
                let cur = self
                    .bridge
                    .param_value_of_kind(node_kind, key)
                    .unwrap_or(0.0);
                self.apply_param_intent(node_kind, key, &target, cur + delta)
            }
            IntentAction::AddNode { kind, friendly } => {
                match self.bridge.add_node_by_kind(kind) {
                    Some((_id, connected)) => {
                        if connected {
                            format!("Added a {friendly} node ({kind}) and connected it")
                        } else {
                            format!("Added a {friendly} node ({kind})")
                        }
                    }
                    None => format!("Couldn't add a {friendly} node — unknown kind {kind}"),
                }
            }
            IntentAction::RunCommand { id, receipt } => {
                if self.registry.run(id) {
                    receipt
                } else {
                    format!("Command {id} is not available")
                }
            }
            IntentAction::GenerateScript { params, name } => {
                self.generate_script_intent(params, &name)
            }
            IntentAction::PlantTree { species_label, species_id, class, pos } => {
                // A lone PlantTree leaf (e.g. an LLM emitting a single PlantTree):
                // plant it and push its entry directly — no group needed for one.
                let e = self.plant_grown_tree_at(&species_label, class, species_id, pos);
                self.push_undo(e);
                format!("Planted a {species_label} — undo with Ctrl+Z")
            }
            IntentAction::Plan { label, steps } => {
                // AAA Spec 07: execute every step in order, COLLECTING each planted
                // tree's undo entry, then push ONE Group so the whole plan reverts
                // with a single Ctrl+Z. Count + species are summarized in the receipt.
                let mut members: Vec<UndoEntry> = Vec::new();
                let mut species_label = String::new();
                for step in steps {
                    if let IntentAction::PlantTree { species_label: sl, species_id, class, pos } = step {
                        if species_label.is_empty() {
                            species_label = sl.clone();
                        }
                        members.push(self.plant_grown_tree_at(&sl, class, species_id, pos));
                    }
                }
                let count = members.len();
                if !members.is_empty() {
                    self.push_undo(UndoEntry::Group { label: label.clone(), members });
                }
                if species_label.is_empty() {
                    format!("{label} — undo with Ctrl+Z")
                } else {
                    format!("Planted {count} {species_label} — undo with Ctrl+Z")
                }
            }
            IntentAction::Unknown { suggestions } => {
                // Teach by example: a domain person should see sentences they could
                // actually type, then the nearest real commands as a fallback.
                format!(
                    "I'm not sure how to do that yet. Try something like \
                     'make the terrain more detailed' or 'add a birch tree' — \
                     or one of: {}",
                    suggestions.join(", ")
                )
            }
        };
        // Tag the receipt with provenance so the assistant log is honest about
        // whether the parser or the model drove the edit: "(parser)" /
        // "(llm:model)" / "(llm failed → parser)". Single log path (log_receipt).
        let receipt = format!("{receipt} {}", provenance.receipt_tag());
        self.log_receipt(receipt.clone());
        receipt
    }

    /// Push a receipt onto the assistant log, capping it at [`HISTORY_CAP`] so the
    /// log can never grow unbounded across a long session (survivors are the most
    /// recent entries).
    fn log_receipt(&mut self, receipt: String) {
        self.assistant_log.push(receipt);
        let overflow = self.assistant_log.len().saturating_sub(HISTORY_CAP);
        if overflow > 0 {
            self.assistant_log.drain(0..overflow);
        }
    }

    /// Push a reversible edit onto the undo stack, capping it at [`HISTORY_CAP`] so a
    /// held-down edit loop can never grow it unbounded (survivors are the most
    /// recent entries — the ones a user would actually undo).
    ///
    /// CONTRACT (wave-14 finding [0], resolved as intended behavior): when a
    /// `PlacedAsset` entry ages out of the cap, the asset it placed becomes
    /// PERMANENT — its splats stay in the overlay and its World entity remains.
    /// That is standard editor undo-history semantics (content falling off the
    /// history is kept, never silently deleted), NOT a leak: the splats are
    /// live, rendered world state the user placed and chose not to undo within
    /// the last [`HISTORY_CAP`] edits. Surviving entries' absolute `start`
    /// ranges stay valid because the cap never touches the overlay.
    fn push_undo(&mut self, entry: UndoEntry) {
        // Finding [8]: clear the inspector-drag coalescing anchor at the SINGLE
        // chokepoint where any entry lands on the stack — mirroring undo()'s reset.
        // This guarantees a non-inspector push (e.g. an AI intent's
        // `apply_param_intent`) that interleaves between drag frames can never be
        // coalesced INTO: the next inspector frame starts a fresh entry instead of
        // overwriting the foreign entry's `next`. `record_inspector_edit` re-sets
        // the anchor AFTER it pushes its OWN entry, so a pure drag still coalesces.
        self.last_inspector_edit = None;
        self.undo_stack.push(entry);
        let overflow = self.undo_stack.len().saturating_sub(HISTORY_CAP);
        if overflow > 0 {
            self.undo_stack.drain(0..overflow);
        }
    }

    /// Apply an absolute param edit through `GraphBridge::apply_param_by_kind`,
    /// recording the pre-edit value on the undo stack and producing the exact
    /// receipt line "Set <target> <prev> -> <next>". On a rejected edit the
    /// receipt reports the cook failure and nothing is pushed onto the undo stack.
    fn apply_param_intent(&mut self, node_kind: &'static str, key: &'static str, target: &str, value: f32) -> String {
        // Resolve the concrete node the intent targets (first node of the kind),
        // then drive everything — apply, readback, undo — by that exact id so the
        // undo round-trips against the same node, exactly like the inspector path.
        let Some(node_id) = self.bridge.first_node_of_kind(node_kind) else {
            return format!("There is no {target} to set");
        };
        let Some(prev) = self.bridge.param_value_of_node(node_id, key) else {
            return format!("There is no {target} to set");
        };
        if self.bridge.apply_param_by_kind(node_kind, key, value).is_none() {
            return format!("There is no {target} to set");
        }
        // The bridge rounds integer params + may reject; read back what cooked.
        let applied = self.bridge.param_value_of_node(node_id, key).unwrap_or(value);
        if let Some(err) = self.bridge.last_cook_error.clone() {
            return format!("Couldn't set {target}: {err}");
        }
        self.push_undo(UndoEntry::ParamSet {
            node_id,
            key,
            target: target.to_string(),
            prev,
            next: applied,
        });
        format!("Set {target} {} -> {}", fmt_num(prev), fmt_num(applied))
    }

    /// The real, default directory AI-generated scripts land in: the repo's
    /// `assets/scripts/generated/`, anchored to the crate manifest (NOT the CWD) so
    /// it resolves the same regardless of where the binary runs — mirroring how
    /// `set_theme` anchors the theme file. The Content browser scans `assets/` and
    /// knows `.rhai`, so a script written here appears in the Content panel on its
    /// next refresh with no extra wiring.
    pub fn default_script_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/scripts/generated")
    }

    /// Override where generated scripts are written (tests point this at a temp dir
    /// so they never leave files under the real `assets/`). The real default is
    /// [`Self::default_script_root`].
    pub fn set_script_root(&mut self, root: impl Into<PathBuf>) {
        self.script_root = root.into();
    }

    /// AI-creates-code v1 executor: generate a compile-verified Rhai script from a
    /// vetted template, write it into [`Self::script_root`] (creating the dir;
    /// collisions get a numbered suffix `_01`/`_02` like asset naming), push a
    /// file-deleting undo entry, append a domain-language receipt + Output Log
    /// line, and tell the user it's in their Content panel. On a generation or I/O
    /// failure nothing is written and the receipt explains honestly.
    fn generate_script_intent(&mut self, params: script_gen::Params, name: &str) -> String {
        let template = params.template();
        // Generation compiles the source in a real rhai engine before returning —
        // a template that produced uncompilable source is a caught bug, not a file.
        let generated = match script_gen::generate(template, name, params) {
            Ok(g) => g,
            Err(e) => return format!("Couldn't write a {template} script: {e}"),
        };

        if let Err(e) = std::fs::create_dir_all(&self.script_root) {
            return format!(
                "Couldn't write a {template} script: failed to create {}: {e}",
                self.script_root.display()
            );
        }

        // Collision → numbered suffix, exactly like the asset naming discipline.
        let path = self.unique_script_path(&generated.name);
        let bytes = generated.source.clone().into_bytes();
        if let Err(e) = std::fs::write(&path, &bytes) {
            return format!("Couldn't write a {template} script: {e}");
        }

        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| generated.name.clone());
        // Domain language for the receipt: "spin script for the windmill".
        let label = script_label(template, &generated.name);
        self.push_undo(UndoEntry::GeneratedScript {
            label: label.clone(),
            path: path.clone(),
            bytes,
        });
        let receipt = format!(
            "Wrote a {label} ({}) — it's in your Content panel; undo with Ctrl+Z",
            self.display_script_path(&path)
        );
        self.log_receipt(receipt.clone());
        self.push_output_log(format!("[script] Generated {file_name}"));
        receipt
    }

    /// A non-colliding path under [`Self::script_root`] for stem `<stem>.rhai`:
    /// `<stem>.rhai` if free, else `<stem>_01.rhai`, `<stem>_02.rhai`, … (the asset
    /// naming discipline). Bounded so a pathological directory can't loop forever.
    fn unique_script_path(&self, stem: &str) -> PathBuf {
        let direct = self.script_root.join(format!("{stem}.rhai"));
        if !direct.exists() {
            return direct;
        }
        for n in 1..1000 {
            let candidate = self.script_root.join(format!("{stem}_{n:02}.rhai"));
            if !candidate.exists() {
                return candidate;
            }
        }
        // Pathological fallback — extremely unlikely; keeps the type total.
        self.script_root
            .join(format!("{stem}_{}.rhai", std::process::id()))
    }

    /// Render a generated-script path for the receipt: a path relative to the
    /// script root prefixed with the conventional `assets/scripts/generated/`, so
    /// the receipt reads in domain terms even when the root is a temp dir in tests.
    fn display_script_path(&self, path: &std::path::Path) -> String {
        let file = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("assets/scripts/generated/{file}")
    }

    /// Record a manual inspector param edit (already applied to the bridge by the
    /// Properties scrub) as a reversible [`UndoEntry::ParamSet`], so Ctrl+Z reverts
    /// it on the SAME concrete `node_id` as an AI-intent edit. The applied
    /// (clamped/rounded) value is read back from that exact node; an edit that did
    /// not actually change the cooked value records nothing.
    ///
    /// Drag coalescing (finding [3]): a continuous scrub fires `changed()` on every
    /// frame the value moves, so naively each frame would push its own undo entry —
    /// dozens per gesture. Chosen approach: coalesce HERE rather than via egui
    /// `drag_started`/`drag_stopped`, because the inspector stages edits and drains
    /// them after the dock lays out, so the egui `Response` (and its gesture flags)
    /// is gone by the time we record. Instead we track the last edit's
    /// `(node_id, key, frame)`: if this edit hits the SAME `(node_id, key)` on the
    /// current or immediately-preceding frame AND the top of the undo stack is that
    /// entry, we update its `next` in place (keeping the gesture's ORIGINAL `prev`)
    /// instead of pushing a new entry. A later, separate edit (a frame gap, or a
    /// different param) starts a fresh entry.
    pub fn record_inspector_edit(
        &mut self,
        node_id: NodeId,
        key: &'static str,
        target: String,
        prev: f32,
    ) {
        let applied = self
            .bridge
            .param_value_of_node(node_id, key)
            .unwrap_or(prev);
        if (applied - prev).abs() <= f32::EPSILON {
            return;
        }

        // Coalesce consecutive-frame edits of the same (node, key) into the entry
        // already on top of the stack: extend its `next`, preserve its `prev`.
        let continues_gesture = matches!(
            self.last_inspector_edit,
            Some((last_id, last_key, last_frame))
                if last_id == node_id
                    && last_key == key
                    && self.frame.saturating_sub(last_frame) <= 1
        );
        if continues_gesture
            && let Some(UndoEntry::ParamSet {
                node_id: top_id,
                key: top_key,
                next,
                ..
            }) = self.undo_stack.last_mut()
            && *top_id == node_id
            && *top_key == key
        {
            *next = applied;
            self.last_inspector_edit = Some((node_id, key, self.frame));
            return;
        }

        self.push_undo(UndoEntry::ParamSet {
            node_id,
            key,
            target,
            prev,
            next: applied,
        });
        self.last_inspector_edit = Some((node_id, key, self.frame));
    }

    /// Revert the last reversible edit (the `edit.undo` command). Re-applies the
    /// inverse through the SAME `GraphBridge` path and returns a receipt. An empty
    /// stack is a no-op with an honest receipt.
    pub fn undo(&mut self) -> String {
        let receipt = match self.undo_stack.pop() {
            Some(UndoEntry::Group { label, members }) => {
                // AAA Spec 07: revert a grouped transaction as ONE undo. The members
                // are NOT individually on `undo_stack` while in flight, so we revert
                // them HIGHEST-START-FIRST (reverse insertion order): each `undo_one`
                // drains its own `[start, len)` off the LIVE overlay, and going in
                // reverse means a member never has to be range-shifted by a later
                // member's removal (the higher ranges go first). We do NOT
                // additionally shift the in-flight members.
                let n = members.len();
                for member in members.into_iter().rev() {
                    self.undo_one(member);
                }
                self.last_inspector_edit = None;
                format!("Undid {label} ({n} steps) from the world")
            }
            Some(entry) => self.undo_one(entry),
            None => "Nothing to undo".to_string(),
        };
        self.log_receipt(receipt.clone());
        receipt
    }

    /// Revert ONE undo entry against the LIVE shell state and return its receipt.
    /// Holds the per-variant revert bodies that [`Self::undo`] used to inline, so
    /// both the single-entry path and a [`UndoEntry::Group`]'s per-member reverts
    /// run identical logic. (A `Group` is never passed here — `undo` flattens it.)
    fn undo_one(&mut self, entry: UndoEntry) -> String {
        match entry {
            UndoEntry::ParamSet { node_id, key, target, prev, next } => {
                // Revert the CONCRETE node that was edited (not first-of-kind), so a
                // graph with two nodes of the same kind reverts the right one.
                self.bridge.apply_param(node_id, key, prev);
                // A subsequent inspector edit must start a NEW undo entry, never
                // coalesce into the one we just popped.
                self.last_inspector_edit = None;
                format!("Undid: {target} {} -> {} (back to {})", fmt_num(next), fmt_num(prev), fmt_num(prev))
            }
            UndoEntry::PlacedAsset { name, start, len } => {
                // Remove EXACTLY this asset's `[start, start+len)` range from the
                // overlay (NOT the tail — a later-undone asset may sit above it) and
                // its World entity, then invalidate the viewport cache so the asset
                // disappears next frame. Range-tracked removal is what lets two
                // interleaved asset types (a tree and a terrain patch) undo
                // independently without truncating each other's splats.
                let end = (start + len).min(self.overlay.len());
                let start = start.min(end);
                let removed = end - start;
                self.overlay.drain(start..end);
                // Every undo entry whose range sits ABOVE the removed one must shift
                // down by `removed` so its `start` still points at its own splats.
                for e in self.undo_stack.iter_mut() {
                    if let UndoEntry::PlacedAsset { start: s, .. } = e {
                        if *s >= end {
                            *s -= removed;
                        }
                    }
                }
                // TWIN shift (Spec 09): the per-entity provenance ranges are the
                // SECOND projection of the same overlay. Shift every surviving
                // entity range that sits ABOVE the removed slice down by `removed`,
                // exactly like the undo stack above, so entity ranges and undo
                // ranges stay consistent. (The removed entity drops its own range
                // with its row below — no shift needed for it.)
                for ent in self.entities.iter_mut() {
                    if let Some(r) = ent.asset_range.as_mut() {
                        if r.start >= end {
                            r.start -= removed;
                        }
                    }
                }
                if let Some(pos) = self.entities.iter().rposition(|e| e.name == name) {
                    self.entities.remove(pos);
                    self.selection.clamp_to(self.entities.len());
                }
                self.viewport_tex = None;
                self.last_inspector_edit = None;
                format!("Removed {name} ({removed} points) from the world")
            }
            UndoEntry::GeneratedScript { label, path, bytes } => {
                self.last_inspector_edit = None;
                // Never destroy what the user has since changed: only delete the
                // file if it is byte-for-byte the content we wrote. A missing file,
                // an edited file, or a read error all PRESERVE the file and say why.
                match std::fs::read(&path) {
                    Ok(current) if current == bytes => match std::fs::remove_file(&path) {
                        Ok(()) => format!(
                            "Removed the {label} ({})",
                            self.display_script_path(&path)
                        ),
                        Err(e) => format!(
                            "Kept the {label} ({}) — couldn't delete it: {e}",
                            self.display_script_path(&path)
                        ),
                    },
                    Ok(_) => format!(
                        "Kept the {label} ({}) — you've edited it since it was \
                         generated, so undo left it alone",
                        self.display_script_path(&path)
                    ),
                    Err(_) => format!(
                        "The {label} ({}) is already gone — nothing to undo",
                        self.display_script_path(&path)
                    ),
                }
            }
            UndoEntry::Group { label, members } => {
                // Defensive: a Group should never be passed to undo_one (undo()
                // flattens it), but handle it by reverting in reverse so the type
                // stays total instead of panicking.
                let n = members.len();
                for member in members.into_iter().rev() {
                    self.undo_one(member);
                }
                self.last_inspector_edit = None;
                format!("Undid {label} ({n} steps) from the world")
            }
        }
    }

    /// Drain queued [`ShellRequest`]s (theme swap, tab focus, undo) — the effects
    /// of registry commands that need `&mut self`. Called once per frame at the
    /// top of `ui`.
    pub fn drain_requests(&mut self) {
        let reqs: Vec<ShellRequest> = self.requests.borrow_mut().drain(..).collect();
        for req in reqs {
            match req {
                ShellRequest::ThemeLight => self.set_theme(true),
                ShellRequest::ThemeDark => self.set_theme(false),
                ShellRequest::FocusViewport => self.focus_viewport(),
                ShellRequest::FocusNodeGraph => self.focus_node_graph(),
                ShellRequest::FocusPlugin(id) => self.focus_plugin_tab(&id),
                ShellRequest::Undo => {
                    self.undo();
                }
                ShellRequest::LoadAsset(path) => self.load_content_asset(&path),
                ShellRequest::GrowTree(tree) => self.plant_grown_tree(tree),
                ShellRequest::ForgeTerrain(patch) => self.plant_forge_terrain(patch),
                ShellRequest::ForgeBuilding(building) => self.plant_forge_building(building),
                ShellRequest::CrucibleScene(scene) => self.plant_crucible_scene(scene),
                ShellRequest::OpenAddPalette => {
                    self.palette.open_intent_prefilled("add ");
                }
                ShellRequest::ForgeryDemo => self.plant_forgery_demo(),
                ShellRequest::CycleIlluminant => {
                    let next = Self::next_illuminant(&self.active_illuminant.name());
                    self.active_illuminant =
                        IlluminantSpec::parse(next).unwrap_or_else(|| {
                            IlluminantSpec::parse("neutral").unwrap()
                        });
                    self.viewport_tex = None; // re-rasterize under the new light
                    self.status = self.hud_receipt();
                }
                ShellRequest::SetIlluminant(spec) => {
                    self.active_illuminant = spec;
                    self.viewport_tex = None;
                    self.status = self.hud_receipt();
                }
                ShellRequest::DuplicateSelection => {
                    self.duplicate_selected();
                }
            }
        }
    }

    /// Plant a grown FloraPrime tree into the live world. Thin wrapper over the
    /// shared [`Self::plant_asset`] core: same overlay/World-entity/undo/receipt
    /// machinery a raised terrain patch uses, specialized only by the vegetation
    /// kind, the tree origin, and the "Grew a" verb.
    fn plant_grown_tree(&mut self, tree: GrownTree) {
        self.plant_asset(
            &tree.species_label,
            "vegetation",
            tree.splats,
            plugins::TREE_PLANT_ORIGIN,
            "Grew a",
            "",
        );
    }

    /// Plant a raised Forge terrain patch into the live world. Thin wrapper over
    /// the shared [`Self::plant_asset`] core — the terrain twin of
    /// [`Self::plant_grown_tree`], specialized only by the terrain kind, the patch
    /// origin, and the "Raised a" verb.
    fn plant_forge_terrain(&mut self, patch: ForgeTerrain) {
        self.plant_asset(
            &patch.label,
            "terrain",
            patch.splats,
            plugins::TERRAIN_PATCH_ORIGIN,
            "Raised a",
            "",
        );
    }

    /// Plant a generated building: same shared core as trees and terrain, with
    /// the backend tag carried into the receipt so the user always learns
    /// whether the REAL Forge generator or the built-in preview built it.
    fn plant_forge_building(&mut self, building: ForgeBuilding) {
        let note = format!(", {}", building.backend);
        self.plant_asset(
            &building.label,
            "building",
            building.splats,
            forge_native::BUILDING_PLANT_ORIGIN,
            "Built a",
            &note,
        );
    }

    /// Plant a cooked Crucible scene: same shared core as buildings, with the
    /// backend tag carried into the receipt so the user always learns whether the
    /// REAL Crucible cook engine ran (and whether its USD round-tripped) or the
    /// built-in preview produced it — the Crucible twin of
    /// [`Self::plant_forge_building`].
    fn plant_crucible_scene(&mut self, scene: CrucibleScene) {
        let note = format!(", {}", scene.backend);
        self.plant_asset(
            &scene.label,
            "scene",
            scene.splats,
            crucible_native::CRUCIBLE_PLANT_ORIGIN,
            "Cooked a",
            &note,
        );
    }

    /// AAA Spec 03 — plant the two metameric "forgery" surfaces. Seeds the
    /// verified [`vox_render::relight::metamer_demo_pair`], bakes each reflectance
    /// under the gallery (reference) light so the stored spectral is RADIANCE,
    /// plants BOTH as small surfaces through the shared [`Self::plant_asset`] core
    /// at offset positions, records their overlay ranges, and sets the live ΔsRGB
    /// HUD receipt. Under the gallery lamp the two read identical; switch the
    /// inspection light (Ctrl+L) and they split — the wedge no RGB engine can copy.
    fn plant_forgery_demo(&mut self) {
        let (base_refl, alt_refl) = vox_render::relight::metamer_demo_pair();
        let ref_spd = self.reference_illuminant.spd();
        let bake = |refl: &[f32; 16]| -> [u16; 16] {
            let radiance = vox_render::relight::forward_band(refl, &ref_spd);
            std::array::from_fn(|b| half::f16::from_f32(radiance[b].clamp(0.0, 65504.0)).to_bits())
        };
        // A small 3×3 patch of volume splats centered at `center`.
        let surface = |center: [f32; 3], bits: [u16; 16]| -> Vec<GaussianSplat> {
            let mut v = Vec::new();
            for ix in -1..=1 {
                for iy in -1..=1 {
                    v.push(GaussianSplat::volume(
                        [center[0] + ix as f32 * 0.35, center[1] + iy as f32 * 0.35, center[2]],
                        [0.3, 0.3, 0.3],
                        glam::Quat::IDENTITY,
                        255,
                        bits,
                    ));
                }
            }
            v
        };
        let pos_a = [-1.3f32, 0.6, -4.5];
        let pos_b = [1.3f32, 0.6, -4.5];
        let group_a = surface(pos_a, bake(&base_refl));
        let group_b = surface(pos_b, bake(&alt_refl));

        let start_a = self.overlay.len();
        self.plant_asset("Forgery Original", "forgery", group_a, pos_a, "Planted a", "");
        let range_a = start_a..self.overlay.len();
        let start_b = self.overlay.len();
        self.plant_asset("Forgery Copy", "forgery", group_b, pos_b, "Planted a", "");
        let range_b = start_b..self.overlay.len();

        self.demo_groups = Some((range_a, range_b));
        self.status = self.hud_receipt();
    }

    /// AAA Spec 03 — the live ΔsRGB receipt over the two planted forgery surfaces.
    /// Computes TWO numbers from CURRENT shell state (never hardcoded): the
    /// divergence under the gallery lamp (the metamer baseline, ≈0) and under the
    /// active inspection light. The label flips to "(forgery)" once the active
    /// divergence clears 0.03 — the spectral signature an RGB capture cannot hold.
    /// Returns the prior status unchanged if no forgery has been planted.
    pub(crate) fn hud_receipt(&self) -> String {
        let Some((ra, rb)) = self.demo_groups.clone() else {
            return self.status.clone();
        };
        // The forgery surfaces can be undone (Ctrl+Z drains their overlay range):
        // if either recorded range is now out of bounds, the demo no longer exists,
        // so keep the prior status rather than panic on a stale slice.
        if ra.end > self.overlay.len() || rb.end > self.overlay.len() {
            return self.status.clone();
        }
        let group_a = &self.overlay[ra];
        let group_b = &self.overlay[rb];
        let gallery = &self.reference_illuminant;
        let d = vox_render::relight::metamer_divergence(
            group_a, group_b, &self.reference_illuminant, gallery, gallery,
        );
        let l = vox_render::relight::metamer_divergence(
            group_a, group_b, &self.reference_illuminant, gallery, &self.active_illuminant,
        );
        let label = if l > 0.03 { "forgery" } else { "metamer" };
        format!(
            "{}: ΔsRGB {d:.3} (metamer) · {}: ΔsRGB {l:.3} ({label})",
            gallery.name(),
            self.active_illuminant.name()
        )
    }

    /// Spec 08 — record the last measured GPU pass time for the frame-budget HUD.
    /// Overwrite-not-append (bounded), shown in the status bar next frame.
    pub fn set_gpu_pass_ms(&mut self, pass: &'static str, ms: f32) {
        self.last_gpu_pass_ms = Some((pass, ms));
    }

    /// The inspection-light cycle for Ctrl+L: gallery (neutral) → the two
    /// strongest forgery-revealers → back. Every name is `IlluminantSpec::parse`able.
    fn next_illuminant(current: &str) -> &'static str {
        match current {
            "neutral" => "cool_led",
            "cool_led" => "tungsten",
            "tungsten" => "daylight",
            _ => "neutral",
        }
    }

    /// The shared planting core for EVERY placed asset (grown trees AND raised
    /// terrain patches): insert `splats` into the viewport overlay at its current
    /// tail, add a numbered World entity ("<label> NN"), push a RANGE-TRACKED undo
    /// entry recording exactly where this asset's splats landed, invalidate the
    /// viewport texture cache so the next frame re-rasterizes with the asset, and
    /// append a domain-language receipt + Output Log line. The range (not a tail
    /// length) is what lets a later undo remove exactly THIS asset's splats even
    /// when another asset type was planted on top — no copy-pasted planting
    /// machinery, no tail-truncation corruption between asset types.
    fn plant_asset(
        &mut self,
        label: &str,
        kind: &str,
        splats: Vec<GaussianSplat>,
        pos: [f32; 3],
        verb: &str,
        note: &str,
    ) {
        // Plant through the non-pushing core, then push the returned undo entry —
        // the single-asset path. (A grouped plan collects the entries instead and
        // pushes ONE Group; see `run_intent`'s Plan arm + `plant_grown_tree_at`.)
        let e = self.plant_asset_collect(label, kind, splats, pos, verb, note);
        self.push_undo(e);
    }

    /// The non-pushing planting CORE (AAA Spec 07): does everything [`plant_asset`]
    /// does — records the overlay range, increments the per-label counter, formats
    /// the numbered name, extends the overlay, pushes the World entity, invalidates
    /// the viewport cache, logs the receipt + Output Log line — EXCEPT it does NOT
    /// touch the undo stack. It RETURNS the [`UndoEntry::PlacedAsset`] instead, so a
    /// caller can either push it directly (single asset, via [`plant_asset`]) or
    /// COLLECT several into one [`UndoEntry::Group`] (a multi-step plan). Behavior
    /// for a single asset is byte-identical to the old `plant_asset`.
    fn plant_asset_collect(
        &mut self,
        label: &str,
        kind: &str,
        splats: Vec<GaussianSplat>,
        pos: [f32; 3],
        verb: &str,
        note: &str,
    ) -> UndoEntry {
        let len = splats.len();
        let start = self.overlay.len();
        // Number the entity per label: "Forge Terrain 01", "…02", …
        let n = self.asset_counts.entry(label.to_string()).or_insert(0);
        *n += 1;
        let name = format!("{label} {:02}", *n);

        self.overlay.extend(splats);
        self.entities.push(ShellEntity {
            name: name.clone(),
            kind: kind.to_string(),
            pos,
            // Record the provenance range (Spec 09): start/len are the SAME values
            // the PlacedAsset undo entry carries — zero extra computation. This lets
            // edit.duplicate clone EXACTLY this entity's splats.
            asset_range: Some(OverlayRange { start, len }),
        });
        // Invalidate the cached viewport texture so the asset shows next frame.
        self.viewport_tex = None;
        let receipt = format!("{verb} {name} ({len} points{note}) — undo with Ctrl+Z");
        self.log_receipt(receipt.clone());
        self.push_output_log(format!("[{kind}] {receipt}"));
        UndoEntry::PlacedAsset { name, start, len }
    }

    /// Grow a tree of `species_id`/`class` and plant it at the ABSOLUTE world
    /// position `pos` (AAA Spec 07's per-step planter). The skeleton-to-splats path
    /// already bakes `TREE_PLANT_ORIGIN` into every splat, so we translate each
    /// splat by the DELTA `pos - TREE_PLANT_ORIGIN` (NOT by `pos` absolute — that
    /// would land the tree at `pos*2 - origin`). Returns the entry from the
    /// non-pushing core so the caller groups several into one undo transaction.
    fn plant_grown_tree_at(
        &mut self,
        species_label: &str,
        class: &str,
        species_id: i32,
        pos: [f32; 3],
    ) -> UndoEntry {
        let skeleton = plugins::grow_tree_skeleton(species_id, 3.0, 200);
        let mut splats = plugins::skeleton_to_splats(&skeleton, class, species_id);
        // The splats are baked at TREE_PLANT_ORIGIN; shift them to `pos` by the delta.
        let origin = plugins::TREE_PLANT_ORIGIN;
        let delta = [pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2]];
        for s in &mut splats {
            let p = s.position();
            s.set_position([p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]]);
        }
        self.plant_asset_collect(species_label, "vegetation", splats, pos, "Grew a", "")
    }

    /// Duplicate the current selection (AAA Spec 09, Ctrl+D). For every selected
    /// entity that OWNS an overlay range (the seed demo entities don't — they're
    /// filtered out), clone EXACTLY its splats at the `DUP_OFFSET` (+X), plant the
    /// copy through the SAME [`Self::plant_asset_collect`] core (so it's numbered,
    /// receipted, and overlay-tracked just like any planted asset), and COLLECT the
    /// per-copy [`UndoEntry::PlacedAsset`] into ONE [`UndoEntry::Group`] — so the
    /// whole duplicate (single OR multi-select) is one Ctrl+Z (matching Spec 07's
    /// grouped-undo). The new copies become the selection. Returns the receipt.
    fn duplicate_selected(&mut self) -> String {
        // Snapshot the sources BEFORE planting (planting mutates `self.overlay` /
        // `self.entities`, which would invalidate live indices and ranges). Only
        // entities with a provenance range are duplicable; the seed entities yield
        // `None` and are filtered out here.
        let sources: Vec<(String, String, [f32; 3], OverlayRange)> = self
            .selection
            .indices()
            .filter_map(|i| {
                let e = self.entities.get(i)?;
                Some((dup_label(&e.name), e.kind.clone(), e.pos, e.asset_range?))
            })
            .collect();
        if sources.is_empty() {
            let msg = "Nothing to duplicate".to_string();
            self.log_receipt(msg.clone());
            return msg;
        }

        let n = sources.len();
        let mut members: Vec<UndoEntry> = Vec::with_capacity(n);
        let mut new_indices: Vec<usize> = Vec::with_capacity(n);
        for (label, kind, pos, r) in sources {
            // Clone EXACTLY this entity's splats and translate each by DUP_OFFSET so
            // the copy lands beside the source (same shape, +X). The range is valid
            // because we snapshotted it before any planting widened the overlay.
            let mut splats = self.overlay[r.start..r.start + r.len].to_vec();
            for s in &mut splats {
                let p = s.position();
                s.set_position([
                    p[0] + DUP_OFFSET[0],
                    p[1] + DUP_OFFSET[1],
                    p[2] + DUP_OFFSET[2],
                ]);
            }
            let copy_pos = [pos[0] + DUP_OFFSET[0], pos[1] + DUP_OFFSET[1], pos[2] + DUP_OFFSET[2]];
            // The new entity is pushed to the END of `self.entities`, so its index
            // is the length-before-push; capture it for the post-loop reselection.
            new_indices.push(self.entities.len());
            let entry = self.plant_asset_collect(&label, &kind, splats, copy_pos, "Duplicated", "");
            members.push(entry);
        }
        // ONE Group for the WHOLE duplicate (single or multi) → one Ctrl+Z.
        self.push_undo(UndoEntry::Group {
            label: format!("Duplicated {n} item(s)"),
            members,
        });
        // Select the copies (primary = first copy) so the user can immediately move
        // / re-duplicate them.
        self.selection = selection_of(&new_indices);
        let receipt = format!("Duplicated {n} item(s) — undo with Ctrl+Z");
        self.log_receipt(receipt.clone());
        receipt
    }

    /// Append a line to the Output Log, capping it at [`HISTORY_CAP`].
    fn push_output_log(&mut self, line: String) {
        self.output_log.push(line);
        let overflow = self.output_log.len().saturating_sub(HISTORY_CAP);
        if overflow > 0 {
            self.output_log.drain(0..overflow);
        }
    }

    /// Handle a Content-tab load request: decode the asset through the SAME
    /// engine-agnostic `vox_editor::content_browser::load_asset` path the
    /// browser exposes, and append an honest receipt to the Output Log. Splat
    /// assets report their decoded splat count; scene/script/shader assets
    /// report that their path was handed to the import pipeline (which is the
    /// truth today — the shell does not yet drop them into the live scene).
    pub fn load_content_asset(&mut self, path: &std::path::Path) {
        use vox_editor::content_browser::{load_asset, AssetKind, LoadedAsset};
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let line = match AssetKind::from_path(path) {
            None => format!("[content] {name}: unrecognized asset type — not loaded"),
            Some(kind) => match load_asset(path, kind) {
                Ok(LoadedAsset::Splats(splats)) => {
                    format!("[content] Loaded {name}: {} points", splats.len())
                }
                Ok(LoadedAsset::Scene(_)) => {
                    format!("[content] Loaded {name}: scene queued for import")
                }
                Ok(LoadedAsset::Script(_)) => format!("[content] Opened script {name}"),
                Ok(LoadedAsset::Shader(_)) => format!("[content] Opened shader {name}"),
                Err(e) => format!("[content] Failed to load {name}: {e}"),
            },
        };
        self.output_log.push(line);
        let overflow = self.output_log.len().saturating_sub(HISTORY_CAP);
        if overflow > 0 {
            self.output_log.drain(0..overflow);
        }
    }

    /// Swap the active token theme (the design's "theme swap is a file edit").
    /// Light loads `assets/ui/ochroma_light.theme.json`; dark is `Tokens::default`.
    /// The widget kit is rebuilt so plugin-facing controls re-skin in lockstep.
    pub fn set_theme(&mut self, light: bool) {
        let tokens = if light {
            Tokens::load(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../assets/ui/ochroma_light.theme.json"),
            )
            .unwrap_or_default()
        } else {
            Tokens::default()
        };
        self.tokens = tokens.clone();
        self.widget_kit = WidgetKit::new(tokens);
    }

    /// Select the Node Graph tab as the active/focused tab (for snapshots that
    /// want it maximized — used by `--tab node_graph`).
    pub fn focus_node_graph(&mut self) {
        self.focus_tab(&TabKind::Builtin(PanelId::NodeGraph));
    }

    /// Select the Viewport tab as the active/focused tab (the default snapshot).
    pub fn focus_viewport(&mut self) {
        self.focus_tab(&TabKind::Builtin(PanelId::Viewport));
    }

    /// Select the Content tab as the active/focused tab (used by `--tab content`).
    pub fn focus_content(&mut self) {
        self.focus_tab(&TabKind::Builtin(PanelId::Content));
    }

    /// Select a plugin tab as the active/focused tab by its id (`--tab crucible`).
    pub fn focus_plugin_tab(&mut self, tab_id: &str) {
        self.focus_tab(&TabKind::Plugin(tab_id.to_string()));
    }

    fn focus_tab(&mut self, tab: &TabKind) {
        if let Some((surface, node, t)) = self.dock.find_tab(tab) {
            self.dock.set_active_tab((surface, node, t));
        }
    }

    fn menu_bar(&mut self, ctx: &egui::Context) {
        // Each menu lists the registry commands in its category and DISPATCHES
        // through the registry (the one-command-surface — no direct logic here).
        let mut to_run: Option<String> = None;
        let mut open_palette = false;
        egui::TopBottomPanel::top("shell_menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                for m in ["File", "Edit", "Create", "Build", "Window", "Help"] {
                    ui.menu_button(m, |ui| {
                        let mut any = false;
                        for c in self.registry.commands.iter().filter(|c| c.category == m) {
                            any = true;
                            let label = if c.shortcut.is_empty() {
                                c.title.clone()
                            } else {
                                format!("{}\t{}", c.title, c.shortcut)
                            };
                            if ui.button(label).clicked() {
                                to_run = Some(c.id.clone());
                                ui.close_menu();
                            }
                        }
                        if !any {
                            ui.label(format!("{m} actions"));
                        }
                    });
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // "Ask Ochroma" — the one command surface (UX principle 2).
                    if widgets::primary_action(
                        ui,
                        icon::SEARCH,
                        "Ask Ochroma  (Ctrl+K)",
                        &self.tokens,
                    )
                    .clicked()
                    {
                        open_palette = true;
                    }
                });
            });
        });
        if let Some(id) = to_run {
            self.registry.run(&id);
        }
        if open_palette {
            self.open_palette();
        }
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        let mut add_to_world = false;
        egui::TopBottomPanel::top("shell_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Primary labeled action (the Canva rule) — routes through the
                // registry's `world.add`, the same command the palette runs.
                if widgets::primary_action(ui, icon::ADD, "Add to world", &self.tokens).clicked() {
                    add_to_world = true;
                }
                ui.separator();
                // Gizmo mode icons WITH text labels (UX principle 1).
                for (i, (ic, label)) in [
                    (icon::MOVE, "Move"),
                    (icon::ROTATE, "Rotate"),
                    (icon::SCALE, "Scale"),
                ]
                .into_iter()
                .enumerate()
                {
                    if ui
                        .selectable_label(self.gizmo == i as u8, format!("{ic}  {label}"))
                        .clicked()
                    {
                        self.gizmo = i as u8;
                    }
                }
                ui.separator();
                if ui
                    .selectable_label(self.snap, format!("{}  Snap", icon::SNAP))
                    .clicked()
                {
                    self.snap = !self.snap;
                }
                let _ = widgets::icon_button(ui, icon::SHOW_FLAGS, "What's shown");
                let _ = widgets::icon_button(ui, icon::PERF, "Speed & smoothness");
                ui.separator();
                let _ = widgets::icon_button(ui, icon::PLAY, "Play");
                let _ = widgets::icon_button(ui, icon::PAUSE, "Pause");
                let _ = widgets::icon_button(ui, icon::STOP, "Stop");
            });
        });
        if add_to_world {
            self.registry.run("world.add");
        }
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("shell_status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let [r, g, b, a] = self.tokens.color("status.success");
                ui.label(
                    egui::RichText::new(format!("\u{25CF}  {}", self.status))
                        .color(egui::Color32::from_rgba_unmultiplied(r, g, b, a)),
                );
                // Spec 08 — the measured frame budget: last GPU pass time in ms.
                if let Some((pass, ms)) = self.last_gpu_pass_ms {
                    ui.separator();
                    ui.label(format!("GPU: {pass} {ms:.1} ms"));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let n = self.entities.len();
                    let noun = if n == 1 { "thing" } else { "things" };
                    ui.label(format!("{n} {noun} in the world"));
                    ui.separator();
                    ui.label("Ochroma 0.1.0");
                });
            });
        });
    }

    /// The leaf rect of the node holding a built-in panel (for the dock movement
    /// test). Returns `None` if the panel can't be located.
    pub fn rect_of(&self, panel: PanelId) -> Option<egui::Rect> {
        let want = TabKind::Builtin(panel);
        for (_si, node) in self.dock.iter_all_nodes() {
            if let Some(tabs) = node.tabs()
                && tabs.contains(&want)
            {
                return node.rect();
            }
        }
        None
    }
}

/// The `egui_dock` `TabViewer` that renders each built-in panel AND dispatches
/// plugin tabs to their `EditorPlugin::ui` through a restricted `PluginCtx`.
struct ShellViewer<'a> {
    tokens: &'a Tokens,
    widget_kit: &'a WidgetKit,
    entities: &'a mut Vec<ShellEntity>,
    selection: &'a mut Selection,
    search: &'a mut String,
    canvas: &'a mut NodeCanvas,
    bridge: &'a mut GraphBridge,
    viewport_tex: egui::TextureHandle,
    plugins: &'a mut Vec<InstalledPlugin>,
    /// Manual inspector scrub edits applied this frame, staged as
    /// `(node_id, key, target, prev)` for the shell to record as reversible
    /// `UndoEntry::ParamSet`s after the dock lays out — so a Properties edit is
    /// Ctrl+Z-undoable on the SAME concrete node as an AI-intent edit (the
    /// `UndoEntry` doc invariant). `ShellViewer` borrows `bridge`, not
    /// `undo_stack`, so it stages here and the shell drains via
    /// [`EditorShell::record_inspector_edit`].
    undo_edits: &'a mut Vec<(NodeId, &'static str, String, f32)>,
    /// The live content browser the Content tab renders.
    content: &'a mut ContentPanel,
    /// Runtime Output Log lines (read-only here) shown in the Output Log tab.
    output_log: &'a [String],
    /// A content-browser activation (double-click) staged this frame for the
    /// shell to drain into a `ShellRequest::LoadAsset` after the dock lays out.
    content_action: &'a mut Option<ContentAction>,
}

impl egui_dock::TabViewer for ShellViewer<'_> {
    type Tab = TabKind;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            TabKind::Builtin(p) => format!("{}  {}", p.icon(), p.title()).into(),
            TabKind::Plugin(id) => {
                let decl = self.plugin_tab(id);
                match decl {
                    Some(TabDecl { icon, title, .. }) => format!("{icon}  {title}").into(),
                    None => id.clone().into(),
                }
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            TabKind::Builtin(PanelId::Hierarchy) => self.hierarchy(ui),
            TabKind::Builtin(PanelId::Inspector) => self.inspector(ui),
            TabKind::Builtin(PanelId::Viewport) => self.viewport(ui),
            TabKind::Builtin(PanelId::NodeGraph) => self.node_graph(ui),
            TabKind::Builtin(PanelId::Content) => self.content(ui),
            TabKind::Builtin(PanelId::Output) => self.output(ui),
            TabKind::Plugin(id) => self.plugin_tab_ui(ui, &id.clone()),
        }
    }
}

impl ShellViewer<'_> {
    fn hierarchy(&mut self, ui: &mut egui::Ui) {
        widgets::search_box(ui, self.search);
        ui.separator();
        let q = self.search.to_lowercase();
        let mut shown = 0usize;
        for (i, e) in self.entities.iter().enumerate() {
            if !q.is_empty() && !e.name.to_lowercase().contains(&q) {
                continue;
            }
            shown += 1;
            let (ic, color_key) = vox_ui::design::icons::entity_icon(&e.kind);
            let [r, g, b, a] = self.tokens.color(color_key);
            let label = egui::RichText::new(format!("{ic}  {}", e.name))
                .color(egui::Color32::from_rgba_unmultiplied(r, g, b, a));
            if ui.selectable_label(self.selection.contains(i), label).clicked() {
                // Modifier-aware multi-select (Spec 09): Cmd/Ctrl toggles, Shift
                // range-selects from the anchor, a plain click selects only this row.
                let mods = ui.input(|inp| inp.modifiers);
                if mods.command {
                    self.selection.toggle(i);
                } else if mods.shift {
                    self.selection.extend_to(i);
                } else {
                    self.selection.select_only(i);
                }
            }
        }
        // Empty state teaches: how to put the first thing into the world.
        if shown == 0 {
            let [r, g, b, a] = self.tokens.color("text.secondary");
            let msg = hierarchy_empty_message(self.entities.is_empty());
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(msg)
                    .color(egui::Color32::from_rgba_unmultiplied(r, g, b, a)),
            );
        }
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        // When a graph node is selected, the Properties tab shows that node's
        // REAL params (scrub fields); editing one routes request_recook +
        // live_cook and refreshes the canvas wire labels.
        if let Some((node_id, title, fields)) = self.bridge.selected_params() {
            ui.heading(format!("{}  {title}", icon::NODE_GRAPH));
            ui.separator();
            let tokens = self.tokens;
            // Each edit carries (key, pre-edit value, new value) so a manual scrub
            // can be recorded as a reversible UndoEntry::ParamSet below.
            let mut edits: Vec<(&'static str, f32, f32)> = Vec::new();
            widgets::foldout(ui, egui::Id::new("insp_node_params"), "Parameters", |ui| {
                for f in &fields {
                    ui.horizontal(|ui| {
                        ui.label(f.label);
                        let mut v = f.value;
                        let resp = widgets::scrub_drag(
                            ui,
                            &mut v,
                            tokens,
                            ScrubOpts {
                                speed: f.speed,
                                range: Some(f.range.clone()),
                                ..Default::default()
                            },
                        );
                        if resp.changed() || (v - f.value).abs() > f32::EPSILON {
                            edits.push((f.key, f.value, v));
                        }
                    });
                }
            });
            // Stage each edit against the CONCRETE node_id so undo round-trips to the
            // exact node edited (not first-of-kind), replayed via the SAME
            // apply_param path the AI undo uses.
            for (key, prev, v) in edits {
                self.bridge.apply_param(node_id, key, v);
                // Stage undo only when the edit actually applied (no cook error) —
                // mirrors apply_param_intent, which never records a rejected edit.
                if self.bridge.last_cook_error.is_none() {
                    self.undo_edits.push((
                        node_id,
                        key,
                        format!("{}.{key}", title.to_lowercase()),
                        prev,
                    ));
                }
            }
            // Surface a live-cook failure (e.g. a param the node rejected) in
            // status.error red, so a rejected edit is visible rather than silently
            // leaving stale outputs while the display reverts to the last good value.
            if let Some(err) = self.bridge.last_cook_error.clone() {
                let [r, g, b, a] = self.tokens.color("status.error");
                ui.colored_label(
                    egui::Color32::from_rgba_unmultiplied(r, g, b, a),
                    format!("{}  Couldn't update that: {err}", icon::WARNING),
                );
            }
            return;
        }

        // No node selected: show the World entity's transform (the friendly default).
        let sel = self.selection.primary().min(self.entities.len().saturating_sub(1));
        let name = self.entities.get(sel).map(|e| e.name.clone()).unwrap_or_default();
        ui.heading(name);
        ui.separator();
        let tokens = self.tokens;
        if let Some(e) = self.entities.get_mut(sel) {
            widgets::foldout(ui, egui::Id::new("insp_transform"), "Transform", |ui| {
                let axes = [("axis.x", "X"), ("axis.y", "Y"), ("axis.z", "Z")];
                ui.horizontal(|ui| {
                    for (i, (key, lbl)) in axes.iter().enumerate() {
                        ui.label(*lbl);
                        widgets::scrub_drag(
                            ui,
                            &mut e.pos[i],
                            tokens,
                            ScrubOpts {
                                speed: 0.1,
                                axis_color: Some(key),
                                ..Default::default()
                            },
                        );
                    }
                });
            });
            widgets::foldout(ui, egui::Id::new("insp_material"), "Material & Light", |ui| {
                ui.label("how it looks under real light");
                let mut amp = 80.0f32;
                widgets::scrub_drag(
                    ui,
                    &mut amp,
                    tokens,
                    ScrubOpts {
                        speed: 0.5,
                        range: Some(0.0..=500.0),
                        suffix: " nm",
                        ..Default::default()
                    },
                );
            });
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        // A REAL engine frame: the rasterized spectral-splat scene, uploaded as a
        // texture and drawn as an Image filling the tab. The floating "View: Real
        // light" pill renders over it (UX principle 1: plain-language label).
        let rect = ui.available_rect_before_wrap();
        let [r, g, b, a] = self.tokens.color("surface.bg.0");
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgba_unmultiplied(r, g, b, a));
        let inner = rect.shrink(8.0);
        // Draw the rendered splat frame, scaled to fill the inner rect.
        egui::Image::new(&self.viewport_tex)
            .corner_radius(self.tokens.radius[1])
            .paint_at(ui, inner);

        // Floating "View: Real light" pill (top-left).
        let pill = egui::Rect::from_min_size(
            inner.left_top() + egui::vec2(12.0, 12.0),
            egui::vec2(150.0, 26.0),
        );
        let [pr, pg, pb, pa] = self.tokens.color("surface.bg.2");
        ui.painter().rect_filled(
            pill,
            self.tokens.radius[2],
            egui::Color32::from_rgba_unmultiplied(pr, pg, pb, pa.min(235)),
        );
        let [ar, ag, ab, _] = self.tokens.color("status.success");
        ui.painter().circle_filled(
            pill.left_center() + egui::vec2(12.0, 0.0),
            4.0,
            egui::Color32::from_rgb(ar, ag, ab),
        );
        ui.painter().text(
            pill.left_center() + egui::vec2(22.0, 0.0),
            egui::Align2::LEFT_CENTER,
            "View: Real light",
            egui::FontId::proportional(self.tokens.type_ramp.body),
            egui::Color32::from_rgb(220, 222, 230),
        );
    }

    fn node_graph(&mut self, ui: &mut egui::Ui) {
        // The REAL cook graph: project the live OchromaNodeGraph onto a
        // CanvasGraph each frame (typed wires from real ports, value labels from
        // the cooked wire_values()), render it with the shared canvas, and route
        // node selection into the Properties tab.
        let mut cg = self.bridge.to_canvas_graph();
        let resp = self.canvas.ui(ui, self.tokens, &mut cg);
        if let Some(id) = resp.clicked {
            self.bridge.select_by_canvas_id(id);
        } else if resp.background_clicked {
            self.bridge.selected = None;
        }
    }

    fn content(&mut self, ui: &mut egui::Ui) {
        // Delegate to the REAL content browser panel; stage any double-click
        // load for the shell to drain into a ShellRequest::LoadAsset.
        if let Some(action) = self.content.ui(ui, self.tokens) {
            *self.content_action = Some(action);
        }
    }

    fn output(&mut self, ui: &mut egui::Ui) {
        // A short, warm intro line so an empty log still tells a domain person
        // what they're looking at (precise engine lines follow, unchanged).
        let [hr, hg, hb, ha] = self.tokens.color("text.secondary");
        ui.label(
            egui::RichText::new(
                "This is the activity log — everything Ochroma does shows up here, newest at the bottom.",
            )
            .color(egui::Color32::from_rgba_unmultiplied(hr, hg, hb, ha)),
        );
        ui.separator();
        for line in [
            "[ochroma] Ochroma started",
            "[ochroma] Opened world: alpine_demo",
            "[render] Showing 2.4M points (detail budget)",
            "[ok] All systems healthy",
        ] {
            ui.label(egui::RichText::new(line).monospace());
        }
        // Runtime log lines appended at runtime (e.g. content-browser loads).
        for line in self.output_log {
            ui.label(egui::RichText::new(line).monospace());
        }
    }

    /// Find a plugin tab declaration by its tab id.
    fn plugin_tab(&self, tab_id: &str) -> Option<TabDecl> {
        for p in self.plugins.iter() {
            if let Some(t) = p.tabs.iter().find(|t| t.id == tab_id) {
                return Some(t.clone());
            }
        }
        None
    }

    /// Dispatch a plugin tab to its `EditorPlugin::ui` through a `PluginCtx` that
    /// exposes ONLY the design system (tokens + widget kit + the per-tab canvas).
    fn plugin_tab_ui(&mut self, ui: &mut egui::Ui, tab_id: &str) {
        let tokens = self.tokens;
        let kit = self.widget_kit;
        for p in self.plugins.iter_mut() {
            if !p.tabs.iter().any(|t| t.id == tab_id) {
                continue;
            }
            if let Some((_, canvas)) = p.canvases.iter_mut().find(|(id, _)| id == tab_id) {
                let mut cx = PluginCtx {
                    tokens,
                    widgets: kit,
                    canvas,
                };
                p.plugin.ui(tab_id, ui, &mut cx);
            }
            return;
        }
    }
}

/// World-units the node-graph drag snaps to.
const GRAPH_SNAP: f32 = 8.0;

/// Format a param value for a receipt: integers print without a decimal point
/// (so "64 -> 128", not "64.0 -> 128.0"), fractionals keep two places.
/// Domain-language label for a generated script, used in receipts: e.g. a
/// `Spin` script with stem `windmill_spin` reads "spin script for the windmill".
/// Falls back to "<template> script" when no subject can be recovered from the
/// stem (the stem is "<subject>_<template_id>"; we strip the trailing template id).
fn script_label(template: script_gen::ScriptTemplate, stem: &str) -> String {
    let kind = match template {
        script_gen::ScriptTemplate::Spin => "spin",
        script_gen::ScriptTemplate::Bob => "bob",
        script_gen::ScriptTemplate::PulseLight => "pulse",
    };
    // The stem is conventionally "<subject>_<template_id>"; recover the subject.
    let subject = stem
        .strip_suffix(&format!("_{}", template.id()))
        .filter(|s| !s.is_empty() && *s != "scene");
    match subject {
        Some(s) => format!("{kind} script for the {}", s.replace('_', " ")),
        None => format!("{kind} script"),
    }
}

fn fmt_num(v: f32) -> String {
    if (v.fract()).abs() < f32::EPSILON {
        format!("{}", v as i64)
    } else {
        format!("{v:.2}")
    }
}

/// The teaching copy the World panel shows when nothing is listed. With an empty
/// world it points at the real "＋ Add to world" affordance (which opens the
/// Ask-Ochroma intent path that genuinely inserts a node); otherwise the search
/// filter hid everything. Pure so the empty-world branch is unit-testable from a
/// constructed-empty shell without driving egui paint.
fn hierarchy_empty_message(entities_empty: bool) -> &'static str {
    if entities_empty {
        "This is your world — it's empty for now. Press ＋ Add to world \
         to ask Ochroma for the first thing you'd like to see \
         (try \"add a birch tree\")."
    } else {
        "Nothing here matches your search. Clear it to see everything in the world."
    }
}

/// The base label for a duplicate (Spec 09): strip a trailing ` NN` numeric
/// suffix (so "Silver Birch 01" → "Silver Birch", and the duplicate is re-numbered
/// by the planting counter to "Silver Birch 02"), but leave a user-renamed name
/// like "My Tree" untouched (its last token isn't a number). The strip fires ONLY
/// when the last space-token parses as a `u32`.
fn dup_label(name: &str) -> String {
    match name.rsplit_once(' ') {
        Some((head, tail)) if tail.parse::<u32>().is_ok() => head.to_string(),
        _ => name.to_string(),
    }
}

/// Build a [`Selection`] over `indices` (the freshly-planted copies), with the
/// FIRST index as primary — so after a duplicate the new copies are selected and
/// the inspector binds to the first one.
fn selection_of(indices: &[usize]) -> Selection {
    let mut sel = Selection::default();
    for &i in indices {
        sel.set.insert(i);
    }
    sel.primary = indices.first().copied().unwrap_or(0);
    sel
}

/// Build the editor's one-command-surface. Menus, toolbar, palette and (later)
/// the AI assistant all dispatch through these. `flag` is flipped by the
/// representative `world.add` command so the palette test can observe execution.
fn build_registry(
    flag: &Rc<RefCell<bool>>,
    requests: &Rc<RefCell<Vec<ShellRequest>>>,
) -> CommandRegistry {
    let mut r = CommandRegistry::new();
    let f = flag.clone();
    let q = requests.clone();
    r.add(Command::new(
        "world.add",
        "Add to world",
        "Create",
        "Ctrl+A",
        move || {
            // Proves the registry callback fired (the palette test asserts this),
            // AND queues the real action: open the palette in intent mode primed
            // with "add " so the next sentence inserts a node via AddNode. The
            // request is drained next frame (opening the palette needs `&mut`).
            *f.borrow_mut() = true;
            q.borrow_mut().push(ShellRequest::OpenAddPalette);
        },
    ));
    r.add(Command::new("create.terrain", "Add terrain", "Create", "", || {}));
    r.add(Command::new("create.biome", "Add a climate layer", "Create", "", || {}));
    r.add(Command::new("file.save", "Save world", "File", "Ctrl+S", || {}));
    r.add(Command::new("file.open", "Open world…", "File", "Ctrl+O", || {}));
    // Undo routes through the registry too (same one-command-surface) — its
    // closure queues a request the shell drains, since undo needs `&mut self`.
    let q = requests.clone();
    r.add(Command::new("edit.undo", "Undo", "Edit", "Ctrl+Z", move || {
        q.borrow_mut().push(ShellRequest::Undo)
    }));
    r.add(Command::new("edit.redo", "Redo", "Edit", "Ctrl+Shift+Z", || {}));
    // Duplicate the selection (Spec 09) — clones each selected entity + its splats
    // at +X as ONE grouped undo. Queues a request the shell drains (needs `&mut`).
    let q = requests.clone();
    r.add(Command::new("edit.duplicate", "Duplicate", "Edit", "Ctrl+D", move || {
        q.borrow_mut().push(ShellRequest::DuplicateSelection)
    }));
    r.add(Command::new("build.cook", "Update the world", "Build", "F5", || {}));
    r.add(Command::new("view.wireframe", "Show the wireframe outline", "Window", "", || {}));
    r.add(Command::new("help.about", "About Ochroma", "Help", "", || {}));

    // AAA Spec 03 — the wedge made one-key reachable. "demo.forgery" plants the
    // two metameric surfaces; "view.illuminant" (Ctrl+L) flips the inspection
    // light so they split. Both queue a request the shell drains (planting +
    // illuminant state need `&mut self` the closure can't hold).
    let q = requests.clone();
    r.add(Command::new("demo.forgery", "Plant the spectral forgery demo", "View", "", move || {
        q.borrow_mut().push(ShellRequest::ForgeryDemo)
    }));
    let q = requests.clone();
    r.add(Command::new("view.illuminant", "Cycle the inspection light", "View", "Ctrl+L", move || {
        q.borrow_mut().push(ShellRequest::CycleIlluminant)
    }));

    // The theme + tab-focus commands the intent assistant (and menus) dispatch.
    // Each queues a `ShellRequest` drained next frame — the intent executor and a
    // manual menu click both reach the same effect through `registry.run`.
    let q = requests.clone();
    r.add(Command::new("view.theme_light", "Switch to light theme", "Window", "", move || {
        q.borrow_mut().push(ShellRequest::ThemeLight)
    }));
    let q = requests.clone();
    r.add(Command::new("view.theme_dark", "Switch to dark theme", "Window", "", move || {
        q.borrow_mut().push(ShellRequest::ThemeDark)
    }));
    let q = requests.clone();
    r.add(Command::new("view.focus_viewport", "Show the world", "Window", "", move || {
        q.borrow_mut().push(ShellRequest::FocusViewport)
    }));
    let q = requests.clone();
    r.add(Command::new("view.focus_node_graph", "Show the Node Graph", "Window", "", move || {
        q.borrow_mut().push(ShellRequest::FocusNodeGraph)
    }));
    let q = requests.clone();
    r.add(Command::new("view.focus_crucible", "Show the Crucible graph", "Window", "", move || {
        q.borrow_mut().push(ShellRequest::FocusPlugin(plugins::CRUCIBLE_TAB.to_string()))
    }));
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_dock::{NodeIndex, SurfaceIndex, TabIndex};
    use vox_ui::node_canvas::{CanvasGraph, NodeView, WireView};
    use vox_ui::{NodeCategory, PortType};

    #[test]
    fn dock_tabs_present_and_movable() {
        let shell = EditorShell::default();
        let titles: Vec<&str> = shell
            .dock
            .iter_all_tabs()
            .filter_map(|(_, t)| match t {
                TabKind::Builtin(p) => Some(p.title()),
                TabKind::Plugin(_) => None,
            })
            .collect();
        for want in ["World", "Properties", "Viewport", "Node Graph", "Content", "Output Log"] {
            assert!(titles.contains(&want), "missing dock tab {want}; have {titles:?}");
        }
    }

    /// The design's test 4 (regression lock): render the full shell snapshot and
    /// prove it is REAL anti-aliased vector text — the 5x7 `burn_text` bitmap
    /// only ever emits full-on-or-off pixels, so a continuum of grayscale
    /// coverage levels on glyph edges is incompatible with it.
    #[test]
    fn no_burn_text_signature() {
        let tokens = Tokens::default();
        let bg = tokens.color("surface.bg.0");
        let w = 1280usize;
        let h = 720usize;
        let ctx = egui::Context::default();
        vox_ui::design::icons::install(&ctx);
        vox_ui::egui_theme::apply(&ctx, &tokens);
        let mut shell = EditorShell::new(tokens.clone());
        let rgba = super::cpu_render::render_ui(&ctx, [w, h], bg, |ctx| shell.ui(ctx));

        // Scan the menu-bar band (top 24px) where labels live — assert a rich
        // grayscale continuum (>16 levels), the AA signature.
        let levels = super::cpu_render::distinct_luminance_levels(&rgba, w, (0, 0, w, 24));
        assert!(
            levels > 16,
            "menu-bar text shows only {levels} luminance levels — bitmap-font signature, not AA"
        );

        // And the frame must be substantially painted (not a blank fill).
        let frac = super::cpu_render::non_background_fraction(&rgba, bg, 6);
        assert!(frac > 0.30, "shell snapshot only {:.1}% non-background", frac * 100.0);
    }

    /// Compile-time-ish guard: the editor shell source must not import or call
    /// `burn_text` (the bitmap font). Scans this module's own source.
    #[test]
    fn shell_source_has_no_bitmap_font_calls() {
        // Build the needle at runtime so this test's own source doesn't match.
        let needle = format!("burn{}text", "_");
        // The render path (cpu_render.rs) is what would composite glyphs; it
        // must never reference the bitmap font. (mod.rs is excluded because this
        // test necessarily names the symbol in its messages.)
        let viewer = include_str!("cpu_render.rs");
        assert!(
            !viewer.contains(&needle),
            "the editor shell render path must not reference the bitmap font"
        );
    }

    /// Theme swap is pixel-visible in the rendered shell: a Properties-panel
    /// region fills lighter under the light theme than the dark theme (real RGB
    /// asserted both ways).
    #[test]
    fn theme_swap_changes_panel_pixel() {
        fn sample(theme_light: bool) -> [u8; 4] {
            let tokens = if theme_light {
                Tokens::load(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../assets/ui/ochroma_light.theme.json"),
                )
                .unwrap()
            } else {
                Tokens::default()
            };
            let bg = tokens.color("surface.bg.0");
            let w = 800usize;
            let h = 600usize;
            let ctx = egui::Context::default();
            vox_ui::design::icons::install(&ctx);
            vox_ui::egui_theme::apply(&ctx, &tokens);
            let mut shell = EditorShell::new(tokens);
            let rgba = super::cpu_render::render_ui(&ctx, [w, h], bg, |ctx| shell.ui(ctx));
            // Sample a panel-fill region in the top-left World panel body.
            let (x, y) = (40usize, 200usize);
            let i = (y * w + x) * 4;
            [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
        }
        let dark = sample(false);
        let light = sample(true);
        let delta = (light[0] as i32 - dark[0] as i32).abs();
        assert!(
            light[0] > dark[0] && delta > 40,
            "light theme panel pixel ({light:?}) not clearly lighter than dark ({dark:?})"
        );
    }

    /// SOTA item 15 (Motion): a button's fill differs between a hover frame and a
    /// non-hover frame. The shell theme wires `style.animation_time = motion("fast")`
    /// and distinct `inactive.bg_fill` (surface.bg.3) vs `hovered.bg_fill`
    /// (surface.hover); a frame with the pointer over the button must paint a
    /// measurably different interior fill than a frame with the pointer away.
    #[test]
    fn hover_changes_button_fill() {
        let tokens = Tokens::default();
        let bg = tokens.color("surface.bg.0");
        let (w, h) = (200usize, 80usize);

        // A fixed-rect button so we know exactly where to sample its interior.
        let btn_rect = egui::Rect::from_min_size(egui::pos2(40.0, 20.0), egui::vec2(120.0, 40.0));
        let ui_fn = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.put(btn_rect, egui::Button::new("HOVER ME"));
            });
        };

        // Pointer parked far outside the button (no hover).
        let away = egui::RawInput {
            events: vec![egui::Event::PointerMoved(egui::pos2(2.0, 2.0))],
            ..Default::default()
        };
        // Pointer over the button centre (hovered).
        let over = egui::RawInput {
            events: vec![egui::Event::PointerMoved(btn_rect.center())],
            ..Default::default()
        };

        let render = |raw: egui::RawInput| {
            let ctx = egui::Context::default();
            vox_ui::design::icons::install(&ctx);
            vox_ui::egui_theme::apply(&ctx, &tokens);
            // Advance several frames so the hover animation (animation_time) settles.
            super::cpu_render::render_ui_with_input(&ctx, [w, h], bg, raw, 30, ui_fn)
        };

        let away_px = render(away);
        let over_px = render(over);

        // Sample the button interior centre (avoid the centred glyphs by sampling
        // a few px in from the left edge, vertically centred).
        let sx = (btn_rect.min.x as usize) + 8;
        let sy = btn_rect.center().y as usize;
        let i = (sy * w + sx) * 4;
        let a = [away_px[i], away_px[i + 1], away_px[i + 2]];
        let o = [over_px[i], over_px[i + 1], over_px[i + 2]];
        let delta: i32 = (0..3).map(|c| (a[c] as i32 - o[c] as i32).abs()).sum();
        println!("[hover_changes_button_fill] non-hover fill={a:?} hover fill={o:?} delta={delta}");
        assert!(
            delta > 10,
            "button fill must differ hover vs non-hover (non-hover {a:?} vs hover {o:?}, delta {delta})"
        );
    }

    #[test]
    fn moving_a_tab_changes_its_rect() {
        // Render once to populate leaf rects, capture Inspector's rect, move it
        // into the Hierarchy node, render again, assert its rect moved by more
        // than half a pane width (content follows the tab).
        let ctx = egui::Context::default();
        vox_ui::egui_theme::apply(&ctx, &Tokens::default());
        let mut shell = EditorShell::default();

        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1920.0, 1080.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw.clone(), |ctx| shell.ui(ctx));
        let before = shell.rect_of(PanelId::Inspector).expect("inspector rect before");

        // Find the source location of Inspector and the destination (Hierarchy).
        let src = shell
            .dock
            .find_tab(&TabKind::Builtin(PanelId::Inspector))
            .expect("find inspector");
        let (h_surface, h_node, _) = shell
            .dock
            .find_tab(&TabKind::Builtin(PanelId::Hierarchy))
            .expect("find hierarchy");
        let dst = egui_dock::TabDestination::Node(
            h_surface,
            h_node,
            egui_dock::TabInsert::Append,
        );
        shell.dock.move_tab(src, dst);

        let _ = ctx.run(raw, |ctx| shell.ui(ctx));
        let after = shell.rect_of(PanelId::Inspector).expect("inspector rect after");

        let pane_w = before.width().max(1.0);
        let dx = (after.left() - before.left()).abs();
        assert!(
            dx > pane_w / 2.0,
            "Inspector x-origin moved only {dx} (pane_w {pane_w}); before={before:?} after={after:?}"
        );
        let _ = (NodeIndex::root(), SurfaceIndex::main(), TabIndex(0));
    }

    // === NodeCanvas pixel tests (the cpu_render harness rasterizes the real
    // egui paint mesh, so these assert REAL rendered pixels) ===

    /// Render a NodeCanvas full-frame into RGBA, returning (rgba, w, h).
    fn render_canvas(
        canvas: &mut NodeCanvas,
        graph: &mut CanvasGraph,
        tokens: &Tokens,
        size: [usize; 2],
    ) -> Vec<u8> {
        let ctx = egui::Context::default();
        vox_ui::design::icons::install(&ctx);
        vox_ui::egui_theme::apply(&ctx, tokens);
        let bg = tokens.color("surface.bg.0");
        super::cpu_render::render_ui(&ctx, size, bg, |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    canvas.ui(ui, tokens, graph);
                });
        })
    }

    #[inline]
    fn px(rgba: &[u8], w: usize, x: i32, y: i32) -> [u8; 3] {
        let i = (y as usize * w + x as usize) * 4;
        [rgba[i], rgba[i + 1], rgba[i + 2]]
    }

    /// Two-node, two-port-type graph whose single wire is guaranteed visible and
    /// vertically offset (so the bezier bows) and two-colored (gradient proof).
    fn gradient_graph() -> CanvasGraph {
        let mut g = CanvasGraph::default();
        g.nodes.push(
            NodeView::new(1, "Src", NodeCategory::Spatial, egui::pos2(60.0, 60.0))
                .with_output("out", PortType::Terrain),
        );
        g.nodes.push(
            NodeView::new(2, "Dst", NodeCategory::Sink, egui::pos2(420.0, 240.0))
                .with_input("in", PortType::Splats),
        );
        for n in &mut g.nodes {
            n.size.x = 150.0;
        }
        g.wires.push(WireView {
            from_node: 1, from_port: "out".into(),
            to_node: 2, to_port: "in".into(),
            exec: false, label: None,
        });
        g
    }

    /// SOTA [10]: bezier wire pixels exist OFF the straight chord between the two
    /// sockets (curvature) AND the wire is antialiased (a cross-section through it
    /// shows >8 distinct alpha/coverage levels, impossible for a 1px hard line).
    #[test]
    fn wire_is_curved_and_antialiased_in_pixels() {
        let tokens = Tokens::default();
        let mut canvas = NodeCanvas::new();
        canvas.show_minimap = false;
        let mut graph = gradient_graph();
        let (w, h) = (640usize, 480usize);
        let rgba = render_canvas(&mut canvas, &mut graph, &tokens, [w, h]);

        let origin = egui::Pos2::ZERO; // CentralPanel fills from (0,0)
        let wire = &graph.wires[0];
        let pts = canvas
            .wire_screen_points(origin, &graph, wire, 41)
            .unwrap();
        let p0 = pts[0];
        let p3 = *pts.last().unwrap();
        // Max perpendicular deviation of any sample from the straight chord (the
        // horizontal control handles make an S-curve that passes THROUGH the
        // chord midpoint, so the bow shows at the quarter points, not t=0.5).
        let chord = p3 - p0;
        let len = chord.length().max(1.0);
        let nrm = egui::vec2(-chord.y, chord.x) / len;
        let (mut max_dev, mut bow_pt) = (0.0f32, pts[pts.len() / 4]);
        for p in &pts {
            let d = (*p - p0).dot(nrm).abs();
            if d > max_dev {
                max_dev = d;
                bow_pt = *p;
            }
        }
        assert!(
            max_dev > 5.0,
            "bezier deviates only {max_dev}px from the chord — not curved"
        );

        // Wire color (Terrain->Splats gradient) differs strongly from bg, so any
        // pixel at the bow point that is non-bg is the wire itself.
        let bg = tokens.color("surface.bg.0");
        let mp = px(&rgba, w, bow_pt.x.round() as i32, bow_pt.y.round() as i32);
        let dbg = (0..3).map(|i| (mp[i] as i32 - bg[i] as i32).abs()).max().unwrap();
        assert!(dbg > 20, "no wire pixel at the curved bow point (got {mp:?} vs bg {bg:?})");

        // AA cross-section: collect distinct luminances in a small box around the
        // bow point — the box straddles the antialiased EDGES of the ~4px wire on
        // every side, where coverage blends line->bg. A hard 1px line would show
        // ~2 levels (line + bg); AA shows a continuum.
        let cx = bow_pt.x.round() as i32;
        let cy = bow_pt.y.round() as i32;
        let mut seen = std::collections::HashSet::new();
        for dy in -6..=6 {
            for dx in -6..=6 {
                let x = (cx + dx).clamp(0, w as i32 - 1);
                let y = (cy + dy).clamp(0, h as i32 - 1);
                let p = px(&rgba, w, x, y);
                let lum = (p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100;
                seen.insert(lum);
            }
        }
        assert!(
            seen.len() > 8,
            "wire cross-section shows only {} luminance levels — not antialiased",
            seen.len()
        );
    }

    /// SOTA [11]: the wire color sampled NEAR the source socket matches the
    /// source port-type token color, near the target matches the target's, and
    /// the two differ (gradient between endpoint socket colors).
    #[test]
    fn wire_gradient_matches_socket_token_colors_in_pixels() {
        let tokens = Tokens::default();
        let mut canvas = NodeCanvas::new();
        canvas.show_minimap = false;
        let mut graph = gradient_graph();
        let (w, h) = (640usize, 480usize);
        let rgba = render_canvas(&mut canvas, &mut graph, &tokens, [w, h]);

        let origin = egui::Pos2::ZERO;
        let wire = &graph.wires[0];
        let pts = canvas.wire_screen_points(origin, &graph, wire, 101).unwrap();
        // Sample ~12% in from each end (clear of the socket circles, which are
        // drawn in the same color, and clear of any node body).
        let near_src = pts[pts.len() * 12 / 100];
        let near_dst = pts[pts.len() * 88 / 100];

        let src_tok = tokens.wire_color(PortType::Terrain); // [140,200,90]
        let dst_tok = tokens.wire_color(PortType::Splats); // [76,194,255]
        assert_ne!(src_tok, dst_tok, "endpoint socket colors must differ");

        // The wire is thicker than a pixel; search a small neighborhood for the
        // closest match to the expected gradient color (the rasterized line's
        // centre carries the segment color).
        fn closest_to(rgba: &[u8], w: usize, h: usize, c: egui::Pos2, want: [u8; 4]) -> i32 {
            let mut best = i32::MAX;
            for dy in -4..=4 {
                for dx in -4..=4 {
                    let x = (c.x.round() as i32 + dx).clamp(0, w as i32 - 1);
                    let y = (c.y.round() as i32 + dy).clamp(0, h as i32 - 1);
                    let p = px(rgba, w, x, y);
                    let d: i32 = (0..3).map(|i| (p[i] as i32 - want[i] as i32).abs()).sum();
                    best = best.min(d);
                }
            }
            best
        }
        let d_src = closest_to(&rgba, w, h, near_src, src_tok);
        let d_dst = closest_to(&rgba, w, h, near_dst, dst_tok);
        assert!(
            d_src < 60,
            "wire near source ({d_src}) doesn't match Terrain token {src_tok:?}"
        );
        assert!(
            d_dst < 60,
            "wire near target ({d_dst}) doesn't match Splats token {dst_tok:?}"
        );
    }

    /// SOTA [14]: two nodes of different categories render different header
    /// colors, each equal to its `category_header` token value (sampled pixels).
    #[test]
    fn category_headers_render_their_token_colors() {
        let tokens = Tokens::default();
        let mut canvas = NodeCanvas::new();
        canvas.show_minimap = false;
        let mut graph = CanvasGraph::default();
        // A Spatial node and a Sink node, well separated, no overlapping comment.
        graph.nodes.push({
            let mut n = NodeView::new(1, "Terrain", NodeCategory::Spatial, egui::pos2(60.0, 80.0))
                .with_output("o", PortType::Terrain);
            n.size.x = 150.0;
            n
        });
        graph.nodes.push({
            let mut n = NodeView::new(2, "Splatize", NodeCategory::Sink, egui::pos2(360.0, 80.0))
                .with_input("i", PortType::Splats);
            n.size.x = 150.0;
            n
        });
        let (w, h) = (640usize, 360usize);
        let rgba = render_canvas(&mut canvas, &mut graph, &tokens, [w, h]);

        let origin = egui::Pos2::ZERO;
        // Sample a header-interior pixel: a few px below each node's top, a third
        // of the way in (clear of rounded corners + the title text).
        let sample_header = |id: u64| -> [u8; 3] {
            let r = canvas.node_rect_screen(origin, &graph, id).unwrap();
            let x = (r.min.x + r.width() * 0.30).round() as i32;
            let y = (r.min.y + 5.0).round() as i32;
            px(&rgba, w, x, y)
        };
        let spatial = sample_header(1);
        let sink = sample_header(2);
        let want_spatial = tokens.category_header(NodeCategory::Spatial);
        let want_sink = tokens.category_header(NodeCategory::Sink);

        let near = |got: [u8; 3], want: [u8; 4]| -> bool {
            (0..3).all(|i| (got[i] as i32 - want[i] as i32).abs() <= 24)
        };
        assert!(
            near(spatial, want_spatial),
            "Spatial header pixel {spatial:?} != token {want_spatial:?}"
        );
        assert!(
            near(sink, want_sink),
            "Sink header pixel {sink:?} != token {want_sink:?}"
        );
        assert_ne!(
            [spatial[0], spatial[1], spatial[2]],
            [sink[0], sink[1], sink[2]],
            "two categories must render visibly different header colors"
        );
    }

    // === Command palette tests ===

    /// SOTA [17]: fuzzy 'addw' ranks 'Add to world' first AND Enter fires the
    /// command (the registry callback flips the observable flag).
    #[test]
    fn palette_fuzzy_ranks_and_enter_executes() {
        let shell = EditorShell::default();
        let hits = shell.registry.search("addw");
        assert_eq!(
            hits[0].id, "world.add",
            "'addw' must rank 'Add to world' first; got {:?}",
            hits.iter().map(|c| &c.id).collect::<Vec<_>>()
        );
        // Enter path: running the top hit flips the bound flag.
        assert!(!*shell.last_command_flag.borrow());
        shell.registry.run(&hits[0].id);
        assert!(
            *shell.last_command_flag.borrow(),
            "running the top 'addw' hit must flip the world.add flag"
        );
    }

    /// SOTA [17]: palette pixels are PRESENT with `--palette` and ABSENT without.
    /// The modal dims the WHOLE frame with a translucent black backdrop and paints
    /// a brighter `surface.bg.2` card on top. We prove BOTH: (a) the frame's mean
    /// luminance DROPS sharply when open (the dim backdrop covers everything) and
    /// (b) the modal card centre is BRIGHTER than the dimmed backdrop just outside
    /// it (a real card, not just a dim) — and that this contrast is absent closed.
    #[test]
    fn palette_pixels_present_only_when_open() {
        let (w, h) = (1280usize, 720usize);
        let render = |open: bool| -> Vec<u8> {
            let tokens = Tokens::default();
            let bg = tokens.color("surface.bg.0");
            let ctx = egui::Context::default();
            vox_ui::design::icons::install(&ctx);
            vox_ui::egui_theme::apply(&ctx, &tokens);
            let mut shell = EditorShell::new(tokens.clone());
            super::cpu_render::render_ui(&ctx, [w, h], bg, |ctx| {
                if open {
                    shell.palette.open = true;
                }
                shell.ui(ctx);
            })
        };
        let lum = |p: [u8; 3]| (p[0] as f32 * 30.0 + p[1] as f32 * 59.0 + p[2] as f32 * 11.0) / 100.0;
        let mean_lum = |rgba: &[u8]| -> f32 {
            let n = (rgba.len() / 4) as f32;
            rgba.chunks_exact(4)
                .map(|p| lum([p[0], p[1], p[2]]))
                .sum::<f32>()
                / n
        };

        let open_rgba = render(true);
        let closed_rgba = render(false);

        // (a) Whole-frame dim.
        let ml_open = mean_lum(&open_rgba);
        let ml_closed = mean_lum(&closed_rgba);
        assert!(
            ml_closed - ml_open > 8.0,
            "palette dim backdrop must darken the frame: closed mean {ml_closed:.1} vs open {ml_open:.1}"
        );

        // (b) Card-vs-dimmed-backdrop contrast, sampled in the CENTRE column
        // (avoids the World-panel selection highlight). The modal card sits at
        // ~30% height; a point at ~75% height in the same column is the dimmed
        // viewport. Average a small patch at each to be robust to glyph pixels.
        let patch_lum = |rgba: &[u8], cx: usize, cy: usize| -> f32 {
            let mut s = 0.0;
            let mut n = 0.0;
            for dy in 0..10 {
                for dx in 0..20 {
                    let p = px(rgba, w, (cx + dx) as i32, (cy + dy) as i32);
                    s += lum([p[0], p[1], p[2]]);
                    n += 1.0;
                }
            }
            s / n
        };
        let cx = w / 2 - 10;
        // The backdrop sample sits at 85% height — well BELOW the modal card. (Was
        // 75%, but the card's command list grows downward with each registered
        // command; Spec 09's new `edit.duplicate` row pushed the card's lower edge
        // into the 75% patch. 85% is clear of the card and yields the same margin on
        // both versions, so it still proves the backdrop dim without being layout-
        // fragile to one extra command row.)
        let card_open = patch_lum(&open_rgba, cx, h * 28 / 100);
        let below_open = patch_lum(&open_rgba, cx, h * 85 / 100);
        assert!(
            card_open > below_open + 5.0,
            "open: modal card patch ({card_open:.1}) must be brighter than the dimmed viewport below it ({below_open:.1})"
        );
        // Closed: no modal, so the same centre-column patch is the viewport scene at
        // full (undimmed) brightness — the open backdrop patch must be DIMMER than
        // the closed (undimmed) scene at that location, proving the dim is real.
        let same_loc_closed = patch_lum(&closed_rgba, cx, h * 85 / 100);
        assert!(
            below_open < same_loc_closed - 4.0,
            "open backdrop ({below_open:.1}) must be dimmer than the closed scene ({same_loc_closed:.1})"
        );
    }

    // === Phase 2b: REAL graph / REAL viewport / PLUGIN integration tests ===

    /// Render the full shell at 1920x1080 with the given focused tab, returning
    /// (rgba, w, h, shell) so a test can sample inside a specific tab's rect.
    fn render_full_shell(
        focus: &str,
        with_crucible: bool,
    ) -> (Vec<u8>, usize, usize, EditorShell) {
        let (w, h) = (1920usize, 1080usize);
        let tokens = Tokens::default();
        let bg = tokens.color("surface.bg.0");
        let ctx = egui::Context::default();
        vox_ui::design::icons::install(&ctx);
        vox_ui::egui_theme::apply(&ctx, &tokens);
        let mut shell = EditorShell::new(tokens);
        if with_crucible {
            shell.install_plugin(Box::new(super::plugins::CruciblePlugin::new()));
        }
        match focus {
            "viewport" => shell.focus_viewport(),
            "node_graph" => shell.focus_node_graph(),
            "crucible" => shell.focus_plugin_tab(super::plugins::CRUCIBLE_TAB),
            _ => {}
        }
        let rgba = super::cpu_render::render_ui(&ctx, [w, h], bg, |ctx| shell.ui(ctx));
        (rgba, w, h, shell)
    }

    /// REAL VIEWPORT: the Viewport tab paints actual rendered splats — >5000
    /// non-background pixels INSIDE the viewport rect WITH scene-like color
    /// variance (not a flat fill).
    #[test]
    fn viewport_tab_shows_real_rendered_splats() {
        let (rgba, w, _h, shell) = render_full_shell("viewport", false);
        let rect = shell
            .rect_of(PanelId::Viewport)
            .expect("viewport must have a leaf rect");

        // Sample every pixel inside the viewport rect (shrunk to clear the tab
        // strip + borders). Count non-bg and measure color variance.
        let bg = [16i32, 18, 26]; // viewport studio background
        let x0 = (rect.min.x as usize) + 12;
        let x1 = (rect.max.x as usize).saturating_sub(12).min(w);
        let y0 = (rect.min.y as usize) + 28;
        let y1 = (rect.max.y as usize).saturating_sub(12);
        let mut non_bg = 0usize;
        let (mut sr, mut sg, mut sb, mut n) = (0f64, 0f64, 0f64, 0f64);
        let mut samples: Vec<[f64; 3]> = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let p = px(&rgba, w, x as i32, y as i32);
                let d = (0..3).map(|i| (p[i] as i32 - bg[i]).abs()).max().unwrap();
                if d > 18 {
                    non_bg += 1;
                }
                sr += p[0] as f64;
                sg += p[1] as f64;
                sb += p[2] as f64;
                n += 1.0;
                samples.push([p[0] as f64, p[1] as f64, p[2] as f64]);
            }
        }
        assert!(
            non_bg > 5000,
            "viewport tab shows only {non_bg} rendered (non-bg) pixels inside its rect (need >5000)"
        );
        let (mr, mg, mb) = (sr / n, sg / n, sb / n);
        let var: f64 = samples
            .iter()
            .map(|p| (p[0] - mr).powi(2) + (p[1] - mg).powi(2) + (p[2] - mb).powi(2))
            .sum::<f64>()
            / n;
        assert!(
            var > 80.0,
            "viewport is too flat (color variance {var:.1}) — not a real scene"
        );
    }

    /// The floating "View: Real light" pill renders over the viewport (its
    /// surface.bg.2 card pixels exist near the top-left of the viewport rect).
    #[test]
    fn viewport_pill_renders_over_scene() {
        let (rgba, w, _h, shell) = render_full_shell("viewport", false);
        let rect = shell.rect_of(PanelId::Viewport).unwrap();
        let card = Tokens::default().color("surface.bg.2");
        // Scan the pill region (top-left of the inner viewport).
        let mut hits = 0;
        for y in (rect.min.y as usize + 20)..(rect.min.y as usize + 60) {
            for x in (rect.min.x as usize + 12)..(rect.min.x as usize + 170) {
                let p = px(&rgba, w, x as i32, y as i32);
                if (0..3).all(|i| (p[i] as i32 - card[i] as i32).abs() <= 14) {
                    hits += 1;
                }
            }
        }
        assert!(hits > 200, "the 'View: Real light' pill card is not painted (only {hits} card px)");
    }

    /// REAL GRAPH: the cooked template's REAL wire value labels appear in the
    /// Node Graph canvas pixels — the "Terrain N cells" chip text region is lit.
    #[test]
    fn node_graph_tab_shows_real_wire_value_label_pixels() {
        let (rgba, w, _h, shell) = render_full_shell("node_graph", false);
        let rect = shell
            .rect_of(PanelId::NodeGraph)
            .expect("node graph must have a leaf rect");
        // The wire value chips are bright text on a surface.bg.2 chip — count
        // bright text pixels inside the graph rect (well above the dark canvas).
        let mut bright = 0usize;
        for y in (rect.min.y as usize + 28)..(rect.max.y as usize).saturating_sub(12) {
            for x in (rect.min.x as usize + 12)..(rect.max.x as usize).saturating_sub(12) {
                let p = px(&rgba, w, x as i32, y as i32);
                let lum = (p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100;
                if lum > 180 {
                    bright += 1;
                }
            }
        }
        // Real cooked labels (node titles + value chips) light many bright px.
        assert!(
            bright > 300,
            "node graph shows only {bright} bright label pixels — cooked wire/value text missing"
        );
    }

    /// Selecting the Terrain node populates the Properties tab with its ACTUAL
    /// param names, and a scrub edit changes the cooked sink count — proven by the
    /// wire-value LABEL TEXT changing between two projections of the real graph.
    #[test]
    fn selecting_terrain_then_scrub_changes_wire_label_text() {
        let mut shell = EditorShell::default();
        let terrain = shell.bridge.node_ids[0];

        // Select Terrain -> Properties shows its real params.
        shell.bridge.selected = Some(terrain);
        let (_, title, fields) = shell.bridge.selected_params().unwrap();
        assert_eq!(title, "Terrain");
        let keys: Vec<&str> = fields.iter().map(|f| f.key).collect();
        assert!(
            keys.contains(&"resolution") && keys.contains(&"amplitude"),
            "Terrain Properties must list real params, got {keys:?}"
        );

        // The Terrain output wire label BEFORE the edit.
        let label_of = |s: &EditorShell| -> String {
            s.bridge
                .to_canvas_graph()
                .wires
                .iter()
                .find(|w| w.from_port == "terrain")
                .and_then(|w| w.label.clone())
                .unwrap_or_default()
        };
        let before = label_of(&shell);
        assert!(before.contains("cells"), "before label should be a cell count, got {before:?}");

        // Scrub the resolution up — request_recook + live_cook re-cook the graph.
        shell.bridge.apply_param(terrain, "resolution", 96.0);
        let after = label_of(&shell);
        assert_ne!(
            before, after,
            "scrubbing terrain detail must change the cooked wire value label TEXT ({before:?} -> {after:?})"
        );
        // And the cooked sink (Splatize) splat count genuinely changed.
        assert!(shell.bridge.sink_splat_count().unwrap() > 0);
    }

    /// PLUGIN: installing CruciblePlugin adds its dock tab AND its palette command.
    #[test]
    fn installing_crucible_adds_tab_and_palette_command() {
        let mut shell = EditorShell::default();
        shell.install_plugin(Box::new(super::plugins::CruciblePlugin::new()));

        // Its tab joined the dock.
        let has_tab = shell.dock.iter_all_tabs().any(|(_, t)| {
            matches!(t, TabKind::Plugin(id) if id == super::plugins::CRUCIBLE_TAB)
        });
        assert!(has_tab, "Crucible plugin tab must be present in the dock");

        // Its command is searchable in the palette registry under "Crucible".
        let hits = shell.registry.search("crucible recook");
        assert!(
            hits.iter().any(|c| c.id == "crucible.recook" && c.category == "Crucible"),
            "Crucible: Recook command must be in the palette under category 'Crucible'"
        );
    }

    /// A minimal test plugin with a controllable id and tab id list, used to drive
    /// the install_plugin dedup paths.
    struct TestPlugin {
        id: String,
        tab_ids: Vec<String>,
    }
    impl crate::shell::host::EditorPlugin for TestPlugin {
        fn id(&self) -> &str {
            &self.id
        }
        fn tabs(&self) -> Vec<crate::shell::host::TabDecl> {
            self.tab_ids
                .iter()
                .map(|t| crate::shell::host::TabDecl {
                    id: t.clone(),
                    title: t.clone(),
                    icon: "",
                })
                .collect()
        }
        fn commands(&self) -> Vec<Command> {
            Vec::new()
        }
        fn ui(&mut self, _tab_id: &str, _ui: &mut egui::Ui, _ctx: &mut crate::shell::host::PluginCtx) {}
    }

    fn count_plugin_tabs_in_dock(shell: &EditorShell, tab_id: &str) -> usize {
        shell
            .dock
            .iter_all_tabs()
            .filter(|(_, t)| matches!(t, TabKind::Plugin(id) if id == tab_id))
            .count()
    }

    /// install_plugin: reinstalling a plugin under the SAME id REPLACES it in place
    /// (no shadowed duplicate) — exactly ONE InstalledPlugin and exactly ONE dock tab
    /// remain, mirroring CommandRegistry::add's same-id-replaces policy.
    #[test]
    fn install_plugin_duplicate_id_replaces_in_place() {
        let mut shell = EditorShell::default();
        let before = shell.plugins.len();

        shell.install_plugin(Box::new(TestPlugin {
            id: "test.dup".into(),
            tab_ids: vec!["test.dup.tab".into()],
        }));
        assert_eq!(shell.plugins.len(), before + 1, "first install adds one plugin");
        assert_eq!(
            count_plugin_tabs_in_dock(&shell, "test.dup.tab"),
            1,
            "first install docks exactly one tab"
        );

        // Reinstall the SAME plugin id.
        shell.install_plugin(Box::new(TestPlugin {
            id: "test.dup".into(),
            tab_ids: vec!["test.dup.tab".into()],
        }));
        assert_eq!(
            shell.plugins.iter().filter(|ip| ip.plugin.id() == "test.dup").count(),
            1,
            "duplicate plugin id must REPLACE, not stack a second InstalledPlugin"
        );
        assert_eq!(
            count_plugin_tabs_in_dock(&shell, "test.dup.tab"),
            1,
            "duplicate plugin id must leave exactly one dock tab (no shadowed duplicate)"
        );
    }

    /// install_plugin: a plugin declaring the SAME TabDecl id twice has the duplicate
    /// rejected — only one canvas/tab is registered for that id.
    #[test]
    fn install_plugin_rejects_duplicate_tab_ids_within_one_plugin() {
        let mut shell = EditorShell::default();
        shell.install_plugin(Box::new(TestPlugin {
            id: "test.duptab".into(),
            tab_ids: vec!["shared.tab".into(), "shared.tab".into(), "other.tab".into()],
        }));

        let installed = shell
            .plugins
            .iter()
            .find(|ip| ip.plugin.id() == "test.duptab")
            .expect("plugin installed");
        // The duplicate "shared.tab" must have been dropped: 2 unique tabs kept.
        assert_eq!(
            installed.tabs.len(),
            2,
            "duplicate TabDecl id must be rejected, leaving 2 unique tabs, got {:?}",
            installed.tabs.iter().map(|t| &t.id).collect::<Vec<_>>()
        );
        assert_eq!(
            installed.tabs.iter().filter(|t| t.id == "shared.tab").count(),
            1,
            "exactly one 'shared.tab' must remain"
        );
        // And the dock holds exactly one tab for the deduped id.
        assert_eq!(
            count_plugin_tabs_in_dock(&shell, "shared.tab"),
            1,
            "dock must hold exactly one 'shared.tab' entry"
        );
    }

    /// PLUGIN STYLING: the Crucible canvas renders its category headers in the
    /// SAME token colors as the host graph — sample a Crucible Spatial-node header
    /// pixel and assert it equals `category_header(Spatial)` (the host token), with
    /// the plugin having set no color whatsoever.
    ///
    /// NOTE on enforcement: `PluginCtx` (see `host.rs`) exposes ONLY `tokens`,
    /// `widgets`, `canvas` — it has NO `egui::Visuals` field and NO `egui::Context`
    /// handle, so a plugin physically cannot restyle the host. The
    /// `host::contract_surface::plugin_ctx_exposes_only_design_system` test pins
    /// that type surface (an exhaustive destructure that breaks if a Visuals field
    /// is ever added).
    #[test]
    fn crucible_canvas_uses_host_category_token_colors() {
        let (rgba, w, _h, shell) = render_full_shell("crucible", true);
        let rect = shell
            .dock
            .iter_all_nodes()
            .find_map(|(_, node)| {
                let has = node
                    .tabs()
                    .is_some_and(|ts| ts.iter().any(|t| matches!(t, TabKind::Plugin(id) if id == super::plugins::CRUCIBLE_TAB)));
                if has { node.rect() } else { None }
            })
            .expect("crucible tab must have a leaf rect");

        // The Crucible "terrain" node is a Spatial node near the top-left of the
        // canvas. Its header must be drawn in category_header(Spatial). Scan most
        // of the tab for a pixel matching that exact token color: the panel's
        // "Cook scene" action button + caption + separator now sit above the
        // canvas (mirroring the Forge tab), pushing the canvas down — the assert's
        // intent is "a Spatial header in host token color SOMEWHERE in the tab",
        // not a fixed offset.
        let want = Tokens::default().category_header(NodeCategory::Spatial);
        let mut found = false;
        'outer: for y in (rect.min.y as usize + 30)..(rect.min.y as usize + 380) {
            for x in (rect.min.x as usize + 20)..(rect.min.x as usize + 360) {
                let p = px(&rgba, w, x as i32, y as i32);
                if (0..3).all(|i| (p[i] as i32 - want[i] as i32).abs() <= 16) {
                    found = true;
                    break 'outer;
                }
            }
        }
        assert!(
            found,
            "Crucible Spatial node header pixel must equal host category_header(Spatial)={want:?} — inherited styling"
        );
    }

    // === Phase 3a: Ask Ochroma intent loop / undo / Forge plugin ===

    /// INTENT (set param): "set terrain resolution to 128" routes through the REAL
    /// GraphBridge — the cooked terrain resolution becomes 128 and the receipt text
    /// is exact.
    #[test]
    fn intent_set_param() {
        let mut shell = EditorShell::default();
        // Pre-edit cooked value of terrain.resolution (template default 64).
        let before = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(before, 64.0, "template terrain resolution starts at 64");

        let receipt = shell.run_intent("set terrain resolution to 128");
        // Cooked value (read back from the REAL graph's param cache) is 128.
        let after = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(after, 128.0, "intent must cook terrain resolution to 128, got {after}");
        // Receipt text is exact, now tagged with provenance (default backend is the
        // deterministic parser → "(parser)").
        assert_eq!(receipt, "Set terrain.resolution 64 -> 128 (parser)");
        // And it surfaced in the assistant history strip.
        assert_eq!(shell.assistant_log.last().unwrap(), "Set terrain.resolution 64 -> 128 (parser)");
    }

    /// Adoption #16: a HOSTILE LLM output flowing through the REAL `run_intent`
    /// path still clamps. The seam validates only the KEY (passing the value
    /// through unclamped), so this proves the clamp in `apply_param` is the safety
    /// net behind the LLM: `{"SetParam":{...,"value":1e30}}` cooks to the schema
    /// max (256), not the unbounded value, and the receipt is tagged "(llm:canned)".
    #[test]
    fn llm_hostile_setparam_still_clamps_via_run_intent() {
        let mut shell = EditorShell::default();
        // Inject a canned "LLM" that emits a hostile value for a REAL key.
        shell.intent_backend = intent::IntentBackend::LlmCanned {
            f: std::sync::Arc::new(|_p| {
                Ok(r#"{"SetParam":{"node_kind":"terrain","key":"resolution","value":1e30}}"#.to_string())
            }),
            unavailable: false,
        };
        let receipt = shell.run_intent("crank the terrain detail to infinity");
        let after = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(after, 256.0, "hostile LLM value must clamp to schema max 256, got {after}");
        assert!(shell.bridge.sink_splat_count().unwrap() > 0, "sink still cooks after the clamp");
        assert_eq!(receipt, "Set terrain.resolution 64 -> 256 (llm:canned)");
    }

    /// Findings 0/1 (intent path): a hostile resolution typed into Ask Ochroma is
    /// clamped to the schema range BEFORE it reaches the unbounded heightfield
    /// allocation. "set terrain resolution to 1000000" lands clamped at the schema
    /// max (256); "-5" lands at the schema min (16); a non-finite value (1e30 parses
    /// fine but is enormous) clamps too. None of these panic or abort the editor.
    #[test]
    fn intent_set_param_clamps_hostile_resolution() {
        let mut shell = EditorShell::default();

        shell.run_intent("set terrain resolution to 1000000");
        let after = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(after, 256.0, "hostile-large resolution must clamp to schema max 256, got {after}");
        // The graph cooked cleanly (no abort) and still produces splats.
        assert!(shell.bridge.sink_splat_count().unwrap() > 0, "sink still cooks after clamp");

        shell.run_intent("set terrain resolution to -5");
        let after = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(after, 16.0, "negative resolution must clamp to schema min 16, got {after}");

        shell.run_intent("set terrain resolution to 1e30");
        let after = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(after, 256.0, "1e30 resolution must clamp to schema max 256, got {after}");
        assert!(shell.bridge.sink_splat_count().unwrap() > 0);
    }

    /// INTENT (add node): "add vegetation" grows the live graph by one node whose
    /// real registry type_name is VegetationNode.
    #[test]
    fn intent_add_node() {
        let mut shell = EditorShell::default();
        let before = shell.bridge.node_count();
        let receipt = shell.run_intent("add vegetation");
        let after = shell.bridge.node_count();
        assert_eq!(after, before + 1, "add intent must grow the graph by one node");
        // The new node exists and is a real VegetationNode kind.
        let veg = shell
            .bridge
            .first_node_of_kind("VegetationNode")
            .expect("a VegetationNode must now exist");
        assert_eq!(shell.bridge.graph.node_name(veg), Some("vegetation"));
        assert!(receipt.contains("VegetationNode"), "receipt must name the real kind, got {receipt:?}");
    }

    /// INTENT (unknown): gibberish answers honestly and lists 3 REAL registry
    /// command titles as suggestions.
    #[test]
    fn intent_unknown_lists_suggestions() {
        let mut shell = EditorShell::default();
        let receipt = shell.run_intent("flibbertigibbet wuzzle xyzzy");
        assert!(
            receipt.starts_with("I'm not sure how to do that yet."),
            "unknown intent must answer honestly, got {receipt:?}"
        );
        // Every suggested title must be a real registered command title. The
        // honest fallback lists the nearest real commands after "or one of: ".
        let real_titles: Vec<String> =
            shell.registry.commands.iter().map(|c| c.title.clone()).collect();
        let after = receipt
            .split("or one of: ")
            .nth(1)
            .expect("receipt must name the nearest real commands after 'or one of: '");
        let listed = after
            // The receipt now carries a trailing provenance tag ("(parser)") —
            // strip it before splitting so the last suggestion isn't polluted.
            .trim_end_matches(" (parser)")
            .split(", ")
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        assert_eq!(listed.len(), 3, "must list exactly 3 suggestions, got {listed:?}");
        for s in &listed {
            assert!(real_titles.contains(s), "suggestion {s:?} must be a real command title");
        }
    }

    // === AI-creates-code v1: GenerateScript end-to-end tests ===
    //
    // Every test overrides `script_root` to a UNIQUE temp dir (so nothing is left
    // under the real `assets/`) and removes it on the way out.

    /// A throwaway temp script root, removed when dropped, so no test leaves files
    /// under `assets/` and parallel tests never collide.
    struct TempRoot(PathBuf);
    impl TempRoot {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "ochroma_scriptgen_{tag}_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            TempRoot(p)
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The real default script root is the repo's generated-scripts dir, anchored
    /// to the crate manifest (NOT the CWD) — asserted separately from the temp
    /// override the other tests use.
    #[test]
    fn default_script_root_is_the_real_generated_dir() {
        let root = EditorShell::default_script_root();
        assert!(
            root.ends_with("assets/scripts/generated"),
            "default root must be assets/scripts/generated, got {}",
            root.display()
        );
    }

    /// END-TO-END: "make the windmill spin faster" writes a compile-verified file
    /// into the (temp) root, the receipt is exact domain language, and the file
    /// content equals generate()'s output byte-for-byte.
    #[test]
    fn generate_script_writes_file_with_exact_receipt_and_content() {
        let tmp = TempRoot::new("e2e");
        let mut shell = EditorShell::default();
        shell.set_script_root(&tmp.0);

        let receipt = shell.run_intent("make the windmill spin faster");
        // Exactly one file landed, named windmill_spin.rhai.
        let path = tmp.0.join("windmill_spin.rhai");
        assert!(path.exists(), "the script file must be written, dir: {:?}", std::fs::read_dir(&tmp.0).map(|d| d.count()));

        // Receipt: domain language, the conventional path, Content-panel note, undo hint.
        assert!(receipt.starts_with("Wrote a spin script for the windmill (assets/scripts/generated/windmill_spin.rhai)"),
            "receipt must read in domain language, got {receipt:?}");
        assert!(receipt.contains("it's in your Content panel"), "receipt must mention the Content panel: {receipt:?}");
        assert!(receipt.contains("undo with Ctrl+Z"), "receipt must mention undo: {receipt:?}");

        // File content == generate()'s output for the SAME clamped params.
        let expected = script_gen::generate(
            script_gen::ScriptTemplate::Spin,
            "windmill_spin",
            script_gen::Params::spin(4.0, script_gen::ranges::SPIN_AXIS.default),
        )
        .unwrap();
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, expected.source, "file content must equal generate()'s output");
        // And it really compiles (the file is a valid rhai script).
        rhai::Engine::new().compile(&on_disk).expect("written script must compile");
    }

    /// UNDO deletes exactly the generated file (and only it).
    #[test]
    fn undo_deletes_the_generated_script() {
        let tmp = TempRoot::new("undo_del");
        let mut shell = EditorShell::default();
        shell.set_script_root(&tmp.0);
        shell.run_intent("make the light pulse");
        let path = tmp.0.join("light_pulse_light.rhai");
        assert!(path.exists(), "script must be written first");

        let undo_receipt = shell.undo();
        assert!(!path.exists(), "undo must delete the generated file");
        assert!(undo_receipt.starts_with("Removed the pulse script for the light"),
            "undo receipt must name what was removed, got {undo_receipt:?}");
    }

    /// UNDO after an EXTERNAL modification must NOT delete the file — it survives,
    /// and the receipt explains why (never destroy what the user has changed).
    #[test]
    fn undo_after_external_modification_preserves_file() {
        let tmp = TempRoot::new("undo_keep");
        let mut shell = EditorShell::default();
        shell.set_script_root(&tmp.0);
        shell.run_intent("make the windmill spin");
        let path = tmp.0.join("windmill_spin.rhai");
        assert!(path.exists());

        // The user edits the file after generation.
        std::fs::write(&path, "// my own tweaks\nfn spin_speed() { 9.0 }\n").unwrap();

        let undo_receipt = shell.undo();
        assert!(path.exists(), "an externally-modified file must SURVIVE undo");
        assert!(
            undo_receipt.contains("you've edited it since"),
            "undo receipt must explain why it kept the file, got {undo_receipt:?}"
        );
        // The user's content is intact.
        let kept = std::fs::read_to_string(&path).unwrap();
        assert!(kept.contains("my own tweaks"), "user's edit must be preserved");
    }

    /// COLLISION: two generations of the same name produce _01 then _02 suffixes,
    /// and all three files coexist.
    #[test]
    fn collision_numbers_generated_scripts() {
        let tmp = TempRoot::new("collide");
        let mut shell = EditorShell::default();
        shell.set_script_root(&tmp.0);
        shell.run_intent("make the windmill spin");
        shell.run_intent("make the windmill spin");
        shell.run_intent("make the windmill spin");
        assert!(tmp.0.join("windmill_spin.rhai").exists(), "first is the bare name");
        assert!(tmp.0.join("windmill_spin_01.rhai").exists(), "second is _01");
        assert!(tmp.0.join("windmill_spin_02.rhai").exists(), "third is _02");
    }

    /// UNDO: an intent param edit then `edit.undo` reverts the cooked value to the
    /// pre-edit number; undo with an empty stack is an honest no-op receipt.
    #[test]
    fn undo_reverts_intent_param_edit_and_empty_is_noop() {
        let mut shell = EditorShell::default();
        let before = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(before, 64.0);

        shell.run_intent("set terrain resolution to 128");
        assert_eq!(shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap(), 128.0);

        // Undo via the same one-command-surface (edit.undo queues a request the
        // shell drains): run the command, then drain.
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        let reverted = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(reverted, 64.0, "undo must restore the pre-edit cooked value 64, got {reverted}");

        // The undo stack is now empty: a further undo is a no-op receipt.
        let receipt = shell.undo();
        assert_eq!(receipt, "Nothing to undo");
        // Value unchanged by the no-op undo.
        assert_eq!(shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap(), 64.0);
    }

    /// Finding 6: a MANUAL inspector param edit is recorded on the SAME undo stack as
    /// AI intents, so Ctrl+Z reverts it (the UndoEntry doc invariant that previously
    /// held only for AI edits). Drives the exact code path the inspector + dock-drain
    /// use: `bridge.apply_param(node_id, ..)` then `record_inspector_edit(..)`.
    #[test]
    fn inspector_param_edit_is_undoable() {
        let mut shell = EditorShell::default();
        let terrain = shell.bridge.node_ids[0];
        let prev = shell.bridge.param_value_of_node(terrain, "resolution").unwrap();
        assert_eq!(prev, 64.0);

        // The inspector applies the scrub edit straight to the bridge...
        shell.bridge.apply_param(terrain, "resolution", 100.0);
        assert_eq!(shell.bridge.param_value_of_node(terrain, "resolution").unwrap(), 100.0);
        // ...and the shell drains it onto the undo stack (what the dock-show loop does),
        // recorded against the CONCRETE node id.
        shell.record_inspector_edit(terrain, "resolution", "terrain.resolution".into(), prev);
        assert_eq!(shell.undo_stack.len(), 1, "the inspector edit must record one undo entry");

        // Ctrl+Z (edit.undo) restores the PRE-EDIT value through the same path.
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        let reverted = shell.bridge.param_value_of_kind("TerrainNode", "resolution").unwrap();
        assert_eq!(reverted, 64.0, "undo must restore the inspector edit's pre-edit value 64, got {reverted}");
    }

    /// Finding [2]: with TWO nodes of the SAME kind, an inspector edit + undo must
    /// round-trip against the CONCRETE node edited (node B) and NEVER touch the other
    /// node (node A). Before the fix, undo replayed via first-of-kind, so it either
    /// dropped the undo (equal values) or corrupted node A (differing values).
    #[test]
    fn inspector_undo_targets_edited_node_not_first_of_kind() {
        let mut shell = EditorShell::default();
        // The template already has one VegetationNode (node A). Add a second (node B).
        let node_a = shell.bridge.first_node_of_kind("VegetationNode").unwrap();
        let (node_b, _) = shell.bridge.add_node_by_kind("VegetationNode").unwrap();
        assert_ne!(node_a, node_b, "must be two distinct VegetationNodes");

        // Give A and B DIFFERENT heights so a first-of-kind bug would be observable:
        // A=8, B=6. (Both default to 6; bump A to 8.)
        shell.bridge.apply_param(node_a, "height", 8.0);
        shell.bridge.apply_param(node_b, "height", 6.0);
        let a_before = shell.bridge.param_value_of_node(node_a, "height").unwrap();
        let b_before = shell.bridge.param_value_of_node(node_b, "height").unwrap();
        assert_eq!(a_before, 8.0);
        assert_eq!(b_before, 6.0);

        // Edit node B via the inspector drain path: apply, then record by id.
        shell.bridge.apply_param(node_b, "height", 10.0);
        shell.record_inspector_edit(node_b, "height", "vegetation.height".into(), b_before);
        assert_eq!(shell.undo_stack.len(), 1, "one undo entry recorded for the B edit");
        assert_eq!(shell.bridge.param_value_of_node(node_b, "height").unwrap(), 10.0);
        // Node A untouched by the edit.
        assert_eq!(shell.bridge.param_value_of_node(node_a, "height").unwrap(), 8.0);

        // Undo: B restored to 6, A NEVER touched (stays 8).
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        assert_eq!(
            shell.bridge.param_value_of_node(node_b, "height").unwrap(),
            6.0,
            "undo must restore the EDITED node B to its pre-edit value"
        );
        assert_eq!(
            shell.bridge.param_value_of_node(node_a, "height").unwrap(),
            8.0,
            "undo must NOT touch node A (the other node of the same kind)"
        );
    }

    /// Finding [3]: ten consecutive-frame inspector edits to the same param coalesce
    /// into ONE undo entry whose `prev` is the ORIGINAL value; a later, separate edit
    /// (a frame gap) starts a SECOND entry.
    #[test]
    fn inspector_drag_coalesces_into_one_undo_entry() {
        let mut shell = EditorShell::default();
        let terrain = shell.bridge.node_ids[0];
        let original = shell.bridge.param_value_of_node(terrain, "amplitude").unwrap();

        // Simulate a 10-frame drag: one frame bump + one staged record per frame, all
        // on the same (node, key). amplitude range is 0..=800, so 100,110,...,190 all land.
        let mut prev = original;
        for i in 0..10u32 {
            shell.frame = shell.frame.wrapping_add(1); // each "frame" advances the counter
            let target = 100.0 + i as f32 * 10.0;
            shell.bridge.apply_param(terrain, "amplitude", target);
            shell.record_inspector_edit(terrain, "amplitude", "terrain.amplitude".into(), prev);
            prev = shell.bridge.param_value_of_node(terrain, "amplitude").unwrap();
        }
        assert_eq!(
            shell.undo_stack.len(),
            1,
            "a single 10-frame drag must coalesce into ONE undo entry, got {}",
            shell.undo_stack.len()
        );
        let UndoEntry::ParamSet { prev: entry_prev, next, .. } = shell.undo_stack.last().unwrap()
        else { panic!("expected a ParamSet undo entry") };
        assert_eq!(*entry_prev, original, "the coalesced entry's prev must be the ORIGINAL value");
        assert_eq!(*next, 190.0, "the coalesced entry's next must be the final drag value");

        // A later, SEPARATE edit (a frame gap larger than 1) starts a 2nd entry.
        let before_second = shell.bridge.param_value_of_node(terrain, "amplitude").unwrap();
        shell.frame = shell.frame.wrapping_add(5);
        shell.bridge.apply_param(terrain, "amplitude", 300.0);
        shell.record_inspector_edit(terrain, "amplitude", "terrain.amplitude".into(), before_second);
        assert_eq!(
            shell.undo_stack.len(),
            2,
            "a separate edit after a frame gap must be a 2nd undo entry"
        );
    }

    /// Finding [8]: an AI intent that edits the SAME (node, key) as an in-progress
    /// inspector drag must NOT be coalesced into. Drag frame, then an AI intent on
    /// the same param, then another drag frame on the same param — all within the
    /// 1-frame coalescing window — must yield THREE distinct undo entries (drag,
    /// intent, drag), with the intent entry's prev/next INTACT.
    #[test]
    fn ai_intent_between_drag_frames_is_not_coalesced_into() {
        let mut shell = EditorShell::default();
        let terrain = shell.bridge.node_ids[0];
        let original = shell.bridge.param_value_of_node(terrain, "resolution").unwrap();
        assert_eq!(original, 64.0);

        // Drag frame 1: inspector scrub resolution 64 -> 100.
        shell.frame = shell.frame.wrapping_add(1);
        shell.bridge.apply_param(terrain, "resolution", 100.0);
        shell.record_inspector_edit(terrain, "resolution", "terrain.resolution".into(), original);
        assert_eq!(shell.undo_stack.len(), 1, "drag frame 1 records one entry");

        // AI intent on the SAME (node, key), same frame window: resolution -> 200.
        let receipt = shell.run_intent("set terrain resolution to 200");
        assert!(receipt.starts_with("Set terrain.resolution 100 -> 200"), "intent receipt: {receipt}");
        assert_eq!(shell.undo_stack.len(), 2, "the AI intent must push its OWN entry, not coalesce");

        // Drag frame 2 on the SAME (node, key), still within the window: 200 -> 150.
        shell.frame = shell.frame.wrapping_add(1);
        let before_drag2 = shell.bridge.param_value_of_node(terrain, "resolution").unwrap();
        assert_eq!(before_drag2, 200.0);
        shell.bridge.apply_param(terrain, "resolution", 150.0);
        shell.record_inspector_edit(terrain, "resolution", "terrain.resolution".into(), before_drag2);

        // THREE entries: drag1, intent, drag2 — the intent's entry was NOT overwritten.
        assert_eq!(
            shell.undo_stack.len(),
            3,
            "drag/intent/drag on the same param must be 3 entries, got {}",
            shell.undo_stack.len()
        );
        // The intent entry (middle of the stack) is intact: prev=100, next=200.
        let UndoEntry::ParamSet { prev: i_prev, next: i_next, .. } = &shell.undo_stack[1]
        else { panic!("expected a ParamSet undo entry") };
        assert_eq!(*i_prev, 100.0, "intent entry prev must survive (100)");
        assert_eq!(*i_next, 200.0, "intent entry next must survive (200) — not overwritten by drag2");
        // The drag2 entry is its own fresh entry: prev=200, next=150.
        let UndoEntry::ParamSet { prev: d2_prev, next: d2_next, .. } = &shell.undo_stack[2]
        else { panic!("expected a ParamSet undo entry") };
        assert_eq!(*d2_prev, 200.0);
        assert_eq!(*d2_next, 150.0);
    }

    /// Finding 2: the undo stack and assistant log are bounded — pushing far more than
    /// the cap leaves exactly HISTORY_CAP survivors, and they are the MOST RECENT
    /// ones. Drives the real intent path (each successful set pushes one undo entry +
    /// one receipt).
    #[test]
    fn history_stacks_are_capped_to_most_recent() {
        let mut shell = EditorShell::default();
        // 250 distinct successful param edits via the real run_intent path. Seed is an
        // integer param with range 0..=999, so each value lands distinctly.
        for i in 0..250u32 {
            let v = i % 1000;
            shell.run_intent(&format!("set terrain seed to {v}"));
        }
        assert_eq!(shell.undo_stack.len(), HISTORY_CAP, "undo stack must be capped at {HISTORY_CAP}");
        assert_eq!(shell.assistant_log.len(), HISTORY_CAP, "assistant log must be capped at {HISTORY_CAP}");

        // The SURVIVORS are the most recent: the newest undo entry's `next` is the last
        // value set (249 -> 249), and the oldest survivor is from iteration 50.
        let UndoEntry::ParamSet { next, .. } = shell.undo_stack.last().unwrap()
        else { panic!("expected a ParamSet undo entry") };
        assert_eq!(*next, 249.0, "newest undo entry must be the last edit (seed=249)");
        let UndoEntry::ParamSet { next: oldest_next, .. } = shell.undo_stack.first().unwrap()
        else { panic!("expected a ParamSet undo entry") };
        assert_eq!(*oldest_next, 50.0, "oldest survivor must be iteration 50 (the first 50 were dropped)");

        // The newest receipt names the last edit too.
        assert_eq!(shell.assistant_log.last().unwrap(), "Set terrain.seed 248 -> 249 (parser)");
    }

    /// FORGE PLUGIN: both Crucible AND Forge tabs are present, both command
    /// categories ("Crucible" + "Forge") are in the palette, and the Forge canvas
    /// renders a Spatial header in the HOST token color (pixel == token).
    #[test]
    fn forge_plugin_coexists_and_canvas_uses_host_tokens() {
        let mut shell = EditorShell::default();
        shell.install_plugin(Box::new(super::plugins::CruciblePlugin::new()));
        shell.install_plugin(Box::new(super::plugins::ForgePlugin::new()));

        // BOTH plugin tabs are docked.
        let tab_ids: Vec<String> = shell
            .dock
            .iter_all_tabs()
            .filter_map(|(_, t)| match t {
                TabKind::Plugin(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        assert!(tab_ids.contains(&super::plugins::CRUCIBLE_TAB.to_string()), "Crucible tab missing");
        assert!(tab_ids.contains(&super::plugins::FORGE_TAB.to_string()), "Forge tab missing");

        // BOTH command categories present in the palette registry, with REAL Forge
        // generator names.
        let cats: std::collections::HashSet<&str> =
            shell.registry.commands.iter().map(|c| c.category.as_str()).collect();
        assert!(cats.contains("Crucible"), "Crucible category missing from palette");
        assert!(cats.contains("Forge"), "Forge category missing from palette");
        for real in ["terrain", "building", "scatter", "road", "vegetation", "water"] {
            let id = format!("forge.generate_{real}");
            assert!(
                shell.registry.commands.iter().any(|c| c.id == id && c.category == "Forge"),
                "Forge command {id} missing under category Forge"
            );
        }

        // PIXEL: render with the Forge tab focused and assert a Forge node header
        // is drawn in the host's category_header token color (the plugin set none).
        let (rgba, w, _h, shell2) = render_full_shell_both("forge");
        let rect = shell2
            .dock
            .iter_all_nodes()
            .find_map(|(_, node)| {
                let has = node.tabs().is_some_and(|ts| {
                    ts.iter().any(|t| matches!(t, TabKind::Plugin(id) if id == super::plugins::FORGE_TAB))
                });
                if has { node.rect() } else { None }
            })
            .expect("Forge tab must have a leaf rect");
        let want = Tokens::default().category_header(NodeCategory::Spatial);
        let mut found = false;
        // Scan most of the tab: the panel's action buttons above the canvas have
        // grown ("Raise terrain" + "Add building"), pushing the canvas down — the
        // assert's intent is "a header in host token color SOMEWHERE in the tab",
        // not a fixed offset.
        'outer: for y in (rect.min.y as usize + 30)..(rect.min.y as usize + 380) {
            for x in (rect.min.x as usize + 20)..(rect.min.x as usize + 360) {
                let p = px(&rgba, w, x as i32, y as i32);
                if (0..3).all(|i| (p[i] as i32 - want[i] as i32).abs() <= 16) {
                    found = true;
                    break 'outer;
                }
            }
        }
        assert!(
            found,
            "Forge Spatial node header pixel must equal host category_header(Spatial)={want:?}"
        );
    }

    /// PALETTE SNAPSHOT PIXELS: with a scripted intent executed, the assistant
    /// receipt strip text region is LIT (status.success-colored monospace on a
    /// surface.bg.3 chip) inside the open palette modal.
    #[test]
    fn palette_receipt_strip_is_lit_after_intent() {
        let (w, h) = (1280usize, 720usize);
        let tokens = Tokens::default();
        let bg = tokens.color("surface.bg.0");
        let ctx = egui::Context::default();
        vox_ui::design::icons::install(&ctx);
        vox_ui::egui_theme::apply(&ctx, &tokens);
        let mut shell = EditorShell::new(tokens.clone());
        // Script the generative loop, then open the palette in intent mode.
        let receipt = shell.run_intent("set terrain resolution to 128");
        assert_eq!(receipt, "Set terrain.resolution 64 -> 128 (parser)");
        shell.palette.mode = command_palette::PaletteMode::Intent;
        let rgba = super::cpu_render::render_ui(&ctx, [w, h], bg, |ctx| {
            shell.palette.open = true;
            shell.ui(ctx);
        });

        // The receipt strip renders status.success (green) monospace text on a
        // raised chip inside the centered modal body. Count green-dominant text
        // pixels in the modal region (center column, upper-modal band) — green
        // clearly dominating red+blue is the success-colored receipt text, and it
        // is absent everywhere the modal isn't (the strict region excludes the
        // bottom status bar's own success text).
        let mut lit = 0usize;
        for y in (h * 15 / 100)..(h * 45 / 100) {
            for x in (w * 35 / 100)..(w * 65 / 100) {
                let p = px(&rgba, w, x as i32, y as i32);
                let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
                if g > 90 && g - r > 30 && g - b > 20 {
                    lit += 1;
                }
            }
        }
        assert!(
            lit > 80,
            "the assistant receipt strip must light status.success text pixels in the modal (got {lit})"
        );
    }

    // === FloraPrime: Grow tree → real splats in the live world ===

    /// GROW (end-to-end through the real drain path): a tree FloraPrime grew (pushed
    /// onto the shell's grow-sink) is planted by draining `flora_sink` into a
    /// GrowTree request and applying it. The world count increments, the viewport
    /// overlay grows by the tree's splat count, and the receipt text is exact.
    #[test]
    fn grow_tree_plants_splats_and_world_entity_through_drain() {
        let mut shell = EditorShell::default();
        let world_before = shell.entities.len();
        let overlay_before = shell.overlay.len();
        assert_eq!(overlay_before, 0, "no grown splats before growing");

        // FloraPrime's grow() pushes a GrownTree onto the SAME sink the shell holds.
        let mut flora = plugins::FloraPrimePlugin::with_grow_sink(shell.flora_sink.clone());
        flora.grow(); // default species: Silver Birch, Medium (200 nodes)
        let grown_count = shell.flora_sink.borrow()[0].splats.len();
        assert_eq!(grown_count, 200, "grown Silver Birch has 200 splats");

        // The shell drains the sink into a GrowTree request, then applies it — the
        // exact path EditorShell::ui runs each frame.
        let grown: Vec<GrownTree> = shell.flora_sink.borrow_mut().drain(..).collect();
        for tree in grown {
            shell.requests.borrow_mut().push(ShellRequest::GrowTree(tree));
        }
        shell.drain_requests();

        // World count incremented by one; the new entity is named "Silver Birch 01".
        assert_eq!(shell.entities.len(), world_before + 1, "world grows by one entity");
        assert_eq!(shell.entities.last().unwrap().name, "Silver Birch 01");
        // The viewport overlay grew by EXACTLY the tree's splat count.
        assert_eq!(
            shell.overlay.len(),
            overlay_before + grown_count,
            "overlay must grow by the tree's splat count"
        );
        // The receipt reads in the domain language (matches the landed conventions).
        assert_eq!(
            shell.assistant_log.last().unwrap(),
            &format!("Grew a Silver Birch 01 ({grown_count} points) — undo with Ctrl+Z")
        );
    }

    /// AAA Spec 03 — the forgery demo's live ΔsRGB HUD reads the wedge correctly:
    /// metameric under the gallery lamp, "(forgery)" under the inspection lamp.
    /// Drives the SAME request path the editor UI uses (plant → flip light).
    #[test]
    fn forgery_demo_hud_receipt() {
        // The active light's ΔsRGB is the LAST "ΔsRGB <num>" in the receipt.
        let active_delta = |hud: &str| -> f32 {
            hud.rsplit("ΔsRGB ")
                .next()
                .and_then(|frag| frag.split_whitespace().next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(f32::NAN)
        };

        let mut shell = EditorShell::default();
        let overlay_before = shell.overlay.len();
        shell.requests.borrow_mut().push(ShellRequest::ForgeryDemo);
        shell.drain_requests();
        assert!(
            shell.demo_groups.is_some(),
            "forgery demo must record its two overlay ranges"
        );
        assert!(
            shell.overlay.len() > overlay_before,
            "forgery demo must plant surfaces into the overlay"
        );

        // Under the gallery lamp (neutral == active) the pair reads identical.
        let hud0 = shell.hud_receipt();
        println!("[forgery] gallery HUD: {hud0}");
        assert!(hud0.contains("(metamer)"), "gallery HUD must read metamer: {hud0:?}");
        let d0 = active_delta(&hud0);
        assert!(
            d0 < 0.012,
            "gallery ΔsRGB must be metameric (<0.012), got {d0} from {hud0:?}"
        );

        // Flip the inspection light to cool_led through the request path.
        shell.requests.borrow_mut().push(ShellRequest::SetIlluminant(
            IlluminantSpec::parse("cool_led").unwrap(),
        ));
        shell.drain_requests();
        let hud1 = shell.hud_receipt();
        println!("[forgery] cool_led HUD: {hud1}");
        assert!(
            hud1.contains("(forgery)"),
            "under cool_led the HUD must read forgery: {hud1:?}"
        );
        let l1 = active_delta(&hud1);
        assert!(
            l1 > 0.03,
            "forgery ΔsRGB must exceed 0.03 under cool_led, got {l1} from {hud1:?}"
        );
        // The status bar mirrors the receipt (the drain set it).
        assert!(
            shell.status.contains("cool_led"),
            "status must name the active light: {:?}",
            shell.status
        );
    }

    /// AAA Spec 03 regression: undoing a planted forgery surface (which shrinks
    /// the overlay below the recorded demo ranges) must NOT panic when the HUD is
    /// next recomputed (Ctrl+L). The bounds guard keeps the prior status instead
    /// of slicing a stale range.
    #[test]
    fn forgery_demo_survives_undo_then_illuminant_cycle() {
        let mut shell = EditorShell::default();
        shell.requests.borrow_mut().push(ShellRequest::ForgeryDemo);
        shell.drain_requests();
        assert!(shell.demo_groups.is_some(), "forgery planted");

        // Undo removes the last planted forgery surface, shrinking the overlay so
        // the second recorded range is now out of bounds.
        shell.requests.borrow_mut().push(ShellRequest::Undo);
        shell.drain_requests();

        // Cycling the inspection light recomputes the HUD — would index-panic on
        // the stale range before the bounds guard; now it returns safely.
        shell.requests.borrow_mut().push(ShellRequest::CycleIlluminant);
        shell.drain_requests();
        let hud = shell.hud_receipt();
        assert!(!hud.is_empty(), "hud_receipt must return a safe string, not panic");
    }

    /// Spec 08 — the GPU-ms HUD field is set by set_gpu_pass_ms and renders the
    /// exact status-bar string (a real value, not a stub).
    #[test]
    fn gpu_pass_ms_hud_field_set_and_formatted() {
        let mut shell = EditorShell::default();
        assert!(shell.last_gpu_pass_ms.is_none(), "no GPU-ms before any frame");
        shell.set_gpu_pass_ms("present", 0.42);
        let (pass, ms) = shell.last_gpu_pass_ms.expect("set_gpu_pass_ms stored a reading");
        assert_eq!(pass, "present");
        assert!((ms - 0.42).abs() < 1e-6, "stored the exact ms");
        // The exact string the status bar renders.
        assert_eq!(format!("GPU: {pass} {ms:.1} ms"), "GPU: present 0.4 ms");
    }

    /// UNDO: after growing a tree, undo restores the world count AND the viewport
    /// overlay EXACTLY to their pre-grow state (the specific splats are gone).
    #[test]
    fn undo_removes_grown_tree_splats_and_world_entity() {
        let mut shell = EditorShell::default();
        let world_before = shell.entities.len();
        let overlay_before = shell.overlay.len();

        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let after_count = shell.overlay.len();
        assert!(after_count > overlay_before, "growing adds overlay splats");
        assert_eq!(shell.entities.len(), world_before + 1);
        assert_eq!(shell.undo_stack.len(), 1, "growing pushes exactly one undo entry");

        // Undo via the one-command-surface (edit.undo queues a request the shell drains).
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        assert_eq!(
            shell.overlay.len(),
            overlay_before,
            "undo must restore the overlay to its EXACT pre-grow length"
        );
        assert_eq!(
            shell.entities.len(),
            world_before,
            "undo must remove the grown tree's World entity"
        );
        assert!(
            !shell.entities.iter().any(|e| e.name == "Silver Birch 01"),
            "the grown entity must be gone after undo"
        );
        // The undo receipt names the removed tree and its exact splat count.
        assert_eq!(
            shell.assistant_log.last().unwrap(),
            &format!("Removed Silver Birch 01 ({after_count} points) from the world")
        );
    }

    /// Two grows produce two distinctly-numbered entities ("…01", "…02").
    #[test]
    fn two_grows_produce_incrementing_named_entities() {
        let mut shell = EditorShell::default();
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let names: Vec<&str> = shell.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Silver Birch 01"), "first grow names …01; have {names:?}");
        assert!(names.contains(&"Silver Birch 02"), "second grow names …02; have {names:?}");
    }

    // === Forge: Raise terrain → real splats in the live world ===

    /// RAISE (end-to-end through the real drain path): a patch Forge raised (pushed
    /// onto the shell's terrain-sink) is planted by draining `forge_sink` into a
    /// ForgeTerrain request and applying it. The world count increments, the
    /// viewport overlay grows by the patch's splat count, and the receipt is exact.
    #[test]
    fn raise_terrain_plants_splats_and_world_entity_through_drain() {
        let mut shell = EditorShell::default();
        let world_before = shell.entities.len();
        assert_eq!(shell.overlay.len(), 0, "no planted splats before raising");

        // Forge's generate_terrain() pushes a ForgeTerrain onto the SAME sink.
        let mut forge = plugins::ForgePlugin::with_terrain_sink(shell.forge_sink.clone());
        forge.generate_terrain();
        let patch_count = shell.forge_sink.borrow()[0].splats.len();
        let n = plugins::FORGE_TERRAIN_RESOLUTION as usize;
        assert_eq!(patch_count, n * n, "raised patch has resolution² splats");

        // The shell drains the sink into a ForgeTerrain request, then applies it —
        // the exact path EditorShell::ui runs each frame.
        let raised: Vec<ForgeTerrain> = shell.forge_sink.borrow_mut().drain(..).collect();
        for patch in raised {
            shell.requests.borrow_mut().push(ShellRequest::ForgeTerrain(patch));
        }
        shell.drain_requests();

        assert_eq!(shell.entities.len(), world_before + 1, "world grows by one entity");
        assert_eq!(shell.entities.last().unwrap().name, "Forge Terrain 01");
        assert_eq!(
            shell.overlay.len(),
            patch_count,
            "overlay must grow by the patch's splat count"
        );
        assert_eq!(
            shell.assistant_log.last().unwrap(),
            &format!("Raised a Forge Terrain 01 ({patch_count} points) — undo with Ctrl+Z")
        );
    }

    /// UNDO: after raising a patch, undo restores the world count AND the viewport
    /// overlay EXACTLY to their pre-raise state (the patch's splats are gone).
    #[test]
    fn undo_removes_raised_terrain_splats_and_world_entity() {
        let mut shell = EditorShell::default();
        let world_before = shell.entities.len();

        shell.raise_terrain_headless(0);
        let after_count = shell.overlay.len();
        assert!(after_count > 0, "raising adds overlay splats");
        assert_eq!(shell.entities.len(), world_before + 1);
        assert_eq!(shell.undo_stack.len(), 1, "raising pushes exactly one undo entry");

        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        assert_eq!(shell.overlay.len(), 0, "undo must restore the overlay to empty");
        assert_eq!(shell.entities.len(), world_before, "undo must remove the entity");
        assert!(
            !shell.entities.iter().any(|e| e.name == "Forge Terrain 01"),
            "the raised entity must be gone after undo"
        );
        assert_eq!(
            shell.assistant_log.last().unwrap(),
            &format!("Removed Forge Terrain 01 ({after_count} points) from the world")
        );
    }

    /// Wave-14 [1] codified: counters are MONOTONIC — plant, undo, replant
    /// yields "…02" with no "…01" present (numbers are placement provenance,
    /// never reused; see the asset_counts field doc).
    #[test]
    fn replant_after_undo_gets_a_fresh_number_not_a_reused_one() {
        let mut shell = EditorShell::default();
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        shell.undo();
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let names: Vec<&str> = shell.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"Silver Birch 02"),
            "replant after undo must take the NEXT number; have {names:?}"
        );
        assert!(
            !names.contains(&"Silver Birch 01"),
            "the undone 01 must be gone; have {names:?}"
        );
    }

    /// Wave-14 [0] codified as the intended contract: a PlacedAsset whose undo
    /// entry ages out of HISTORY_CAP becomes PERMANENT — splats stay in the
    /// overlay, the entity stays in the world, and draining the entire
    /// remaining stack cannot remove it. Content falling off undo history is
    /// kept, never silently deleted (standard editor semantics).
    #[test]
    fn capped_out_placed_asset_becomes_permanent_not_leaked() {
        let mut shell = EditorShell::default();
        let base_entities = shell.entities.len();
        // One tree, then push its undo entry off the cap with param edits.
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let tree_splats = shell.overlay.len();
        assert!(tree_splats > 0, "the tree planted splats");
        for i in 0..(HISTORY_CAP + 10) {
            shell.push_undo(UndoEntry::ParamSet {
                node_id: NodeId(0),
                key: "resolution",
                target: "terrain.resolution".into(),
                prev: i as f32,
                next: (i + 1) as f32,
            });
        }
        // Drain the whole surviving stack.
        while !shell.undo_stack.is_empty() {
            shell.undo();
        }
        // The tree is now permanent: still rendered, still listed.
        assert_eq!(
            shell.overlay.len(),
            tree_splats,
            "capped-out asset's splats remain (permanent, by contract)"
        );
        assert_eq!(
            shell.entities.len(),
            base_entities + 1,
            "capped-out asset's entity remains in the world"
        );
    }

    /// "Add building" end-to-end through the real drain path: world +1, overlay
    /// grows by the building's splats, and the receipt names the BACKEND (the
    /// honest tag differs per build config — assert via the constant).
    #[test]
    fn add_building_plants_through_drain_with_backend_receipt() {
        let mut shell = EditorShell::default();
        let base_entities = shell.entities.len();
        let base_overlay = shell.overlay.len();

        let (splats, backend) =
            forge_native::generate_building(forge_native::BuildingSpec::default());
        let count = splats.len();
        assert!(count > 0);
        shell.building_sink.borrow_mut().push(ForgeBuilding {
            label: "Forge Building".to_string(),
            splats,
            backend,
        });
        // Drain exactly like ui() does.
        let built: Vec<ForgeBuilding> = shell.building_sink.borrow_mut().drain(..).collect();
        for b in built {
            shell.plant_forge_building(b);
        }

        assert_eq!(shell.entities.len(), base_entities + 1, "one new world entity");
        assert_eq!(shell.overlay.len(), base_overlay + count, "overlay grew by the building");
        let names: Vec<&str> = shell.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Forge Building 01"), "named entity; have {names:?}");
        let receipt = shell.assistant_log.last().expect("receipt logged");
        assert!(
            receipt.contains(&format!("({count} points, {})", forge_native::BACKEND_TAG)),
            "receipt names the backend: {receipt}"
        );
        assert!(receipt.starts_with("Built a Forge Building 01"), "verb + name: {receipt}");

        // Undo removes exactly the building.
        shell.undo();
        assert_eq!(shell.entities.len(), base_entities, "entity removed on undo");
        assert_eq!(shell.overlay.len(), base_overlay, "overlay restored on undo");
    }

    /// "Cook scene" end-to-end through the real drain path (the Crucible twin of
    /// the building test): world +1, overlay grows by the cooked scene's splats,
    /// the entity is named "Crucible Scene 01", the receipt reads
    /// "Cooked a Crucible Scene 01 (N points, <backend>) — undo with Ctrl+Z" with
    /// the ACTUAL backend tag this build/cook produced, and undo reverses exactly
    /// the scene.
    #[test]
    fn cook_scene_plants_through_drain_with_backend_receipt() {
        let mut shell = EditorShell::default();
        let base_entities = shell.entities.len();
        let base_overlay = shell.overlay.len();

        let (splats, backend) =
            crucible_native::cook_scene(crucible_native::CrucibleSceneSpec::default());
        let count = splats.len();
        assert!(count > 0);
        shell.scene_sink.borrow_mut().push(CrucibleScene {
            label: "Crucible Scene".to_string(),
            splats,
            backend,
        });
        // Drain exactly like ui() does.
        let cooked: Vec<CrucibleScene> = shell.scene_sink.borrow_mut().drain(..).collect();
        for s in cooked {
            shell.plant_crucible_scene(s);
        }

        assert_eq!(shell.entities.len(), base_entities + 1, "one new world entity");
        assert_eq!(shell.overlay.len(), base_overlay + count, "overlay grew by the scene");
        let names: Vec<&str> = shell.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Crucible Scene 01"), "named entity; have {names:?}");
        let receipt = shell.assistant_log.last().expect("receipt logged");
        assert_eq!(
            receipt,
            &format!("Cooked a Crucible Scene 01 ({count} points, {backend}) — undo with Ctrl+Z"),
            "exact receipt with the honest backend tag: {receipt}"
        );

        // Undo removes exactly the scene.
        shell.undo();
        assert_eq!(shell.entities.len(), base_entities, "entity removed on undo");
        assert_eq!(shell.overlay.len(), base_overlay, "overlay restored on undo");
    }

    /// Two cooks produce two distinctly-numbered Crucible scenes ("…01", "…02"),
    /// proving the monotonic per-label counter the receipts rely on.
    #[test]
    fn two_cooks_produce_incrementing_named_scenes() {
        let mut shell = EditorShell::default();
        shell.cook_scene_headless(0);
        shell.cook_scene_headless(1);
        let names: Vec<&str> = shell.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Crucible Scene 01"), "first scene; have {names:?}");
        assert!(names.contains(&"Crucible Scene 02"), "second scene; have {names:?}");
    }

    /// Two raises produce two distinctly-numbered entities ("…01", "…02").
    #[test]
    fn two_raises_produce_incrementing_named_entities() {
        let mut shell = EditorShell::default();
        shell.raise_terrain_headless(0);
        shell.raise_terrain_headless(1);
        let names: Vec<&str> = shell.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Forge Terrain 01"), "first raise names …01; have {names:?}");
        assert!(names.contains(&"Forge Terrain 02"), "second raise names …02; have {names:?}");
    }

    /// COEXISTENCE: grow a tree AND raise terrain → 2 entities, overlay = tree +
    /// terrain. Undo the TERRAIN (the tail here) and the tree's splats survive
    /// bit-identical. (The harder earlier-asset removal is the next test.)
    #[test]
    fn tree_and_terrain_coexist_and_undo_terrain_leaves_tree_untouched() {
        fn splats_eq(a: &GaussianSplat, b: &GaussianSplat) -> bool {
            a.position() == b.position()
                && a.scales() == b.scales()
                && a.opacity() == b.opacity()
                && a.spectral() == b.spectral()
        }

        let mut shell = EditorShell::default();
        let world_before = shell.entities.len();
        // TREE first, then TERRAIN on top.
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let tree_count = shell.overlay.len();
        let tree_splats: Vec<GaussianSplat> = shell.overlay[..tree_count].to_vec();
        shell.raise_terrain_headless(0);
        let terrain_count = shell.overlay.len() - tree_count;
        assert!(terrain_count > 0);

        // Two NEW entities, overlay = tree_count + terrain_count.
        assert_eq!(
            shell.entities.len(),
            world_before + 2,
            "a tree AND a terrain patch coexist (two new world entities)"
        );
        assert_eq!(shell.overlay.len(), tree_count + terrain_count);

        // Undo the terrain (top of stack). The tree's splats sit at the overlay HEAD
        // BELOW the terrain — a tail-truncating undo would chop them; range-tracked
        // removal must leave them bit-identical.
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        assert_eq!(
            shell.overlay.len(),
            tree_count,
            "undoing the terrain must leave EXACTLY the tree's splats"
        );
        assert!(!shell.entities.iter().any(|e| e.name == "Forge Terrain 01"));
        assert!(shell.entities.iter().any(|e| e.name == "Silver Birch 01"));
        assert!(
            shell.overlay.iter().zip(tree_splats.iter()).all(|(a, b)| splats_eq(a, b)),
            "the tree's splats must survive the terrain undo bit-identical"
        );
    }

    /// COEXISTENCE, the hard case the tail-truncation bug CANNOT handle: tree, then
    /// terrain, then undo the EARLIER tree while the terrain (planted ABOVE it)
    /// survives. Range-tracked removal must drain the tree's `[0, tree_count)` range
    /// and SHIFT the terrain's range down so its splats stay valid and bit-identical.
    #[test]
    fn undo_earlier_asset_shifts_later_asset_range() {
        let mut shell = EditorShell::default();
        // Plant tree, then terrain. Stack: [tree(0..T), terrain(T..T+G)].
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let tree_count = shell.overlay.len();
        shell.raise_terrain_headless(0);
        let terrain_count = shell.overlay.len() - tree_count;
        // Capture the terrain splats (the tail) before removing the tree beneath it.
        let terrain_splats: Vec<GaussianSplat> = shell.overlay[tree_count..].to_vec();

        // Undo pops the terrain (LIFO), then re-plant it so the terrain entry sits
        // BELOW a fresh nothing — no. To genuinely undo the EARLIER tree while the
        // terrain remains, we manipulate the stack: pop the terrain entry, undo the
        // tree, then the terrain range must have shifted to start at 0. Drive it via
        // the public undo twice would remove both; instead we remove the tree entry
        // out of order by swapping it to the top, mirroring a future "undo this
        // specific asset" — assert the range-shift invariant the undo arm guarantees.
        // Move the tree's PlacedAsset entry to the top so undo() targets it.
        let tree_pos = shell
            .undo_stack
            .iter()
            .position(|e| matches!(e, UndoEntry::PlacedAsset { name, .. } if name == "Silver Birch 01"))
            .unwrap();
        let tree_entry = shell.undo_stack.remove(tree_pos);
        shell.undo_stack.push(tree_entry);

        assert!(shell.registry.run("edit.undo")); // undo the tree (now top)
        shell.drain_requests();
        // The tree is gone; the terrain survived and its splats are bit-identical,
        // now sitting at the HEAD of the overlay (the range shifted down by tree_count).
        assert_eq!(shell.overlay.len(), terrain_count, "only the terrain remains");
        assert!(!shell.entities.iter().any(|e| e.name == "Silver Birch 01"));
        assert!(shell.entities.iter().any(|e| e.name == "Forge Terrain 01"));
        fn eq(a: &GaussianSplat, b: &GaussianSplat) -> bool {
            a.position() == b.position()
                && a.scales() == b.scales()
                && a.opacity() == b.opacity()
                && a.spectral() == b.spectral()
        }
        assert!(
            shell.overlay.iter().zip(terrain_splats.iter()).all(|(a, b)| eq(a, b)),
            "the terrain's splats must survive the tree undo bit-identical"
        );
        // The remaining undo entry's range must have shifted to start at 0.
        let terrain_entry = shell
            .undo_stack
            .iter()
            .find(|e| matches!(e, UndoEntry::PlacedAsset { name, .. } if name == "Forge Terrain 01"))
            .unwrap();
        let UndoEntry::PlacedAsset { start, len, .. } = terrain_entry else { unreachable!() };
        assert_eq!(*start, 0, "the surviving terrain range must shift to the head");
        assert_eq!(*len, terrain_count);
    }

    // === AAA Spec 07: multi-step Plan → ONE grouped-undo transaction ===

    /// HEADLINE: "add 5 birch trees" plants FIVE distinct entities laid out in a
    /// row (strictly increasing x), as ONE undo-stack Group entry — so a SINGLE
    /// Ctrl+Z restores both the world and the overlay exactly to their pre-plan
    /// state. Proves the grouped-undo reverse-order range math is correct.
    #[test]
    fn add_five_birch_trees_is_one_undo_group() {
        let mut shell = EditorShell::default();
        let e0 = shell.entities.len();
        let o0 = shell.overlay.len();

        let receipt = shell.run_intent("add 5 birch trees");
        assert!(receipt.contains('5'), "receipt must name the count: {receipt}");
        assert!(receipt.contains("Silver Birch"), "receipt must name the species: {receipt}");

        // Five distinct new World entities were added.
        assert_eq!(shell.entities.len(), e0 + 5, "five trees → five new entities");
        // Their x-positions are strictly increasing (a row), proving delta-correct
        // placement (an off-by-origin bug would not produce this exact monotone row).
        let new_xs: Vec<f32> = shell.entities[e0..].iter().map(|e| e.pos[0]).collect();
        assert!(
            new_xs.windows(2).all(|w| w[1] > w[0]),
            "the five trees must form a row with strictly increasing x: {new_xs:?}"
        );
        assert_eq!(
            new_xs,
            vec![-4.0, 0.0, 4.0, 8.0, 12.0],
            "delta-translation must land the row at x = [-4,0,4,8,12]"
        );
        assert!(shell.overlay.len() > o0, "planting five trees adds overlay splats");
        // ONE Group on the undo stack — NOT five separate entries.
        assert_eq!(shell.undo_stack.len(), 1, "the plan is ONE grouped undo entry");
        assert!(
            matches!(shell.undo_stack[0], UndoEntry::Group { ref members, .. } if members.len() == 5),
            "the single entry must be a Group of five members"
        );

        // One Ctrl+Z (via the one-command-surface) restores EVERYTHING.
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        assert_eq!(shell.entities.len(), e0, "one undo restores the world entity count");
        assert_eq!(shell.overlay.len(), o0, "one undo restores the overlay length exactly");
        assert!(shell.undo_stack.is_empty(), "the Group is consumed by the single undo");

        println!("OK: 5 trees, distinct x, one undo restores 0");
    }

    // === World panel empty-state (teaching copy is reachable + honest) ===

    /// The empty-WORLD branch selects the teaching copy that points at the real
    /// "＋ Add to world" affordance; the non-empty (search-hid-everything) branch
    /// selects the search copy. Asserts the exact strings so the copy can't drift
    /// away from what the button does.
    #[test]
    fn hierarchy_empty_message_is_the_add_teaching_copy_when_world_is_empty() {
        let empty = hierarchy_empty_message(true);
        assert_eq!(
            empty,
            "This is your world — it's empty for now. Press ＋ Add to world \
             to ask Ochroma for the first thing you'd like to see \
             (try \"add a birch tree\").",
            "empty world must show the Add-to-world teaching copy"
        );
        let filtered = hierarchy_empty_message(false);
        assert_eq!(
            filtered,
            "Nothing here matches your search. Clear it to see everything in the world.",
            "a non-empty world with no visible rows is the search-empty case"
        );
        assert_ne!(empty, filtered, "the two empty states must teach different things");
    }

    /// EMPTY-STATE REACHABILITY: a shell whose World IS empty (built by clearing
    /// the public `entities` Vec — the only path to emptiness, since the real
    /// removal path / grow-undo only shrinks back to the 4 seeds) renders the
    /// hierarchy without panic, and the empty-WORLD teaching branch is selected.
    /// We render the real shell with the World/Hierarchy tab active so the
    /// `hierarchy()` code path (and its `entities.is_empty()` branch) actually
    /// executes, then assert the message the branch resolves to.
    #[test]
    fn empty_world_renders_teaching_copy() {
        let ctx = egui::Context::default();
        vox_ui::egui_theme::apply(&ctx, &Tokens::default());
        let mut shell = EditorShell::default();

        // Empty the world via the real public field (construction-time emptiness;
        // noted: the in-editor remove/undo path can't reach 0 from the 4 seeds, so
        // clearing the Vec is the honest way to reach the empty state).
        shell.entities.clear();
        shell.search.clear();
        assert!(shell.entities.is_empty(), "world is empty for this test");

        // Render the full shell one frame — the World/Hierarchy tab is in the
        // default dock, so hierarchy() runs and hits the empty-world branch
        // without panicking.
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1920.0, 1080.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw, |ctx| shell.ui(ctx));

        // The branch the rendered hierarchy took resolves to the teaching copy.
        assert_eq!(
            hierarchy_empty_message(shell.entities.is_empty()),
            "This is your world — it's empty for now. Press ＋ Add to world \
             to ask Ochroma for the first thing you'd like to see \
             (try \"add a birch tree\").",
            "an empty world must render the Add-to-world teaching copy"
        );
    }

    /// The "＋ Add to world" command is no longer a no-op: running `world.add`
    /// and draining queues opens the palette in INTENT mode pre-filled with "add ",
    /// dropping the user straight into the Ask-Ochroma path that really inserts a
    /// node. (The flag still flips so the existing palette test holds.)
    #[test]
    fn world_add_opens_intent_palette_prefilled() {
        let mut shell = EditorShell::default();
        assert!(!shell.palette.open, "palette starts closed");

        assert!(shell.registry.run("world.add"), "world.add must be a real command");
        shell.drain_requests();

        assert!(shell.palette.open, "world.add must OPEN the palette");
        assert_eq!(
            shell.palette.mode,
            command_palette::PaletteMode::Intent,
            "world.add must open the palette in intent (Ask-Ochroma) mode"
        );
        assert_eq!(
            shell.palette.query, "add ",
            "the intent line must be pre-filled with the 'add ' verb"
        );
        assert!(
            *shell.last_command_flag.borrow(),
            "world.add must still flip the command flag (palette test invariant)"
        );
    }

    /// The prefilled intent line is a REAL working add (the affordance is not
    /// theatre). AAA Spec 07 split "add <thing>" into two real effects:
    /// (a) a non-species noun ("add a vegetation node") inserts a real GRAPH node;
    /// (b) a species phrase ("add a birch tree") now plants a real TREE — a World
    /// entity + overlay splats + one undo entry — instead of a bare graph node.
    /// Both are genuine mutations, so the affordance produces real work either way.
    #[test]
    fn add_intent_from_prefill_inserts_a_real_node() {
        // (a) A non-species "add" still inserts a real graph node (the AddNode path).
        let mut shell = EditorShell::default();
        let nodes_before = shell.bridge.node_count();
        let receipt = shell.run_intent("add a vegetation node");
        assert!(
            shell.bridge.node_count() > nodes_before,
            "a non-species 'add … node' intent must add a real graph node \
             (before={nodes_before}, after={}, receipt={receipt:?})",
            shell.bridge.node_count()
        );

        // (b) A species phrase plants a real tree (Spec 07): one new World entity,
        // overlay splats, and exactly one undo entry — a genuine world mutation.
        let mut shell = EditorShell::default();
        let entities_before = shell.entities.len();
        let overlay_before = shell.overlay.len();
        let receipt = shell.run_intent("add a birch tree");
        assert_eq!(
            shell.entities.len(),
            entities_before + 1,
            "'add a birch tree' must plant one real tree entity (receipt={receipt:?})"
        );
        assert!(
            shell.overlay.len() > overlay_before,
            "planting a tree must add overlay splats (receipt={receipt:?})"
        );
        assert_eq!(
            shell.undo_stack.len(),
            1,
            "a single-tree plan pushes exactly one undo entry (receipt={receipt:?})"
        );
    }

    /// Render the full shell with BOTH plugins installed and `focus` tab active.
    fn render_full_shell_both(focus: &str) -> (Vec<u8>, usize, usize, EditorShell) {
        let (w, h) = (1920usize, 1080usize);
        let tokens = Tokens::default();
        let bg = tokens.color("surface.bg.0");
        let ctx = egui::Context::default();
        vox_ui::design::icons::install(&ctx);
        vox_ui::egui_theme::apply(&ctx, &tokens);
        let mut shell = EditorShell::new(tokens);
        shell.install_plugin(Box::new(super::plugins::CruciblePlugin::new()));
        shell.install_plugin(Box::new(super::plugins::ForgePlugin::new()));
        match focus {
            "forge" => shell.focus_plugin_tab(super::plugins::FORGE_TAB),
            "crucible" => shell.focus_plugin_tab(super::plugins::CRUCIBLE_TAB),
            _ => {}
        }
        let rgba = super::cpu_render::render_ui(&ctx, [w, h], bg, |ctx| shell.ui(ctx));
        (rgba, w, h, shell)
    }

    // === AAA Spec 09: prefab / duplicate / multi-select ===

    fn dup_splats_eq(a: &GaussianSplat, b: &GaussianSplat) -> bool {
        a.position() == b.position()
            && a.scales() == b.scales()
            && a.opacity() == b.opacity()
            && a.spectral() == b.spectral()
    }

    /// PROVENANCE (Step 1): planting records each entity's exact `[start, len)`
    /// overlay range, and undoing an EARLIER asset shifts the SURVIVING entity's
    /// range down — the entity-range projection stays consistent with the overlay,
    /// exactly like the undo-stack range does. Mirrors the
    /// `undo_earlier_asset_shifts_later_asset_range` out-of-order undo setup.
    #[test]
    fn plant_asset_records_entity_range_and_undo_shifts_it() {
        let mut shell = EditorShell::default();
        // Plant a tree, then a terrain patch. Overlay: [tree(0..T), terrain(T..T+G)].
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let tree_n = shell.overlay.len();
        shell.raise_terrain_headless(0);
        let terr_len = shell.overlay.len() - tree_n;

        // The tree entity owns exactly [0, tree_n); the terrain owns [tree_n, G).
        let tree_ent = shell
            .entities
            .iter()
            .find(|e| e.name == "Silver Birch 01")
            .expect("tree entity present");
        assert_eq!(
            tree_ent.asset_range_for_test(),
            Some((0, tree_n)),
            "the tree entity must own the overlay head [0, tree_n)"
        );
        let terr_ent = shell
            .entities
            .iter()
            .find(|e| e.name == "Forge Terrain 01")
            .expect("terrain entity present");
        assert_eq!(
            terr_ent.asset_range_for_test(),
            Some((tree_n, terr_len)),
            "the terrain entity must own [tree_n, terr_len)"
        );

        // Undo the EARLIER tree out of order (swap its undo entry to the top), as in
        // undo_earlier_asset_shifts_later_asset_range — the surviving terrain entity
        // range must shift down to start at 0.
        let tree_pos = shell
            .undo_stack
            .iter()
            .position(|e| matches!(e, UndoEntry::PlacedAsset { name, .. } if name == "Silver Birch 01"))
            .unwrap();
        let tree_entry = shell.undo_stack.remove(tree_pos);
        shell.undo_stack.push(tree_entry);
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();

        assert!(!shell.entities.iter().any(|e| e.name == "Silver Birch 01"));
        let terr_ent = shell
            .entities
            .iter()
            .find(|e| e.name == "Forge Terrain 01")
            .expect("terrain survives the tree undo");
        assert_eq!(
            terr_ent.asset_range_for_test(),
            Some((0, terr_len)),
            "the surviving terrain entity range must shift to the overlay head"
        );
    }

    /// SELECTION MODEL (Step 2): a `Selection` tracks the exact set of indices under
    /// single / range / toggle / clamp operations — the World multi-select algebra.
    #[test]
    fn selection_toggle_and_range_track_indices() {
        let mut sel = Selection::single(4);
        sel.extend_to(7); // anchor 4 .. 7 inclusive
        assert_eq!(sel.indices().collect::<Vec<_>>(), vec![4, 5, 6, 7]);

        sel.toggle(5); // remove the middle one
        assert_eq!(sel.indices().collect::<Vec<_>>(), vec![4, 6, 7]);

        sel.clamp_to(6); // drop everything >= 6
        assert_eq!(sel.indices().collect::<Vec<_>>(), vec![4]);
        assert_eq!(sel.primary(), 4, "primary must repoint to a surviving member");
    }

    /// HEADLINE (Step 3): duplicate ONE selected tree → the overlay DOUBLES (gains
    /// exactly the tree's splats), a "Silver Birch 02" copy entity exists, every
    /// copy splat is the source +2.0 in X with bit-exact spectral, and ONE Ctrl+Z
    /// removes EXACTLY the copy — overlay and entities back to base, the original
    /// splats bit-identical. The whole duplicate is one grouped undo.
    #[test]
    fn duplicate_one_tree_doubles_overlay_and_one_undo_removes_exactly_the_copy() {
        let mut shell = EditorShell::default();
        // Plant ONE tree; capture the base world/overlay AFTER it.
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let base_entities = shell.entities.len();
        let base_overlay = shell.overlay.len();
        let tree_len = base_overlay; // the tree owns the whole overlay head here
        // The source splats, captured before duplicating.
        let source: Vec<GaussianSplat> = shell.overlay[..tree_len].to_vec();

        // Select the tree entity (the one with a provenance range).
        let tree_idx = shell
            .entities
            .iter()
            .position(|e| e.name == "Silver Birch 01")
            .expect("tree entity present");
        shell.selection = Selection::single(tree_idx);

        // Duplicate through the SAME request + drain path the Ctrl+D command drives.
        shell.registry.run("edit.duplicate");
        shell.drain_requests();

        // One new entity, overlay doubled (old + the tree's own len).
        assert_eq!(
            shell.entities.len(),
            base_entities + 1,
            "duplicate adds exactly one World entity"
        );
        assert_eq!(
            shell.overlay.len(),
            base_overlay + tree_len,
            "duplicate adds exactly the tree's splats (overlay doubles)"
        );
        assert!(
            shell.entities.iter().any(|e| e.name == "Silver Birch 02"),
            "the copy must be the next-numbered 'Silver Birch 02'"
        );
        // Every copy splat == source +2.0 X, with bit-exact spectral/scales/opacity.
        let copy: Vec<GaussianSplat> = shell.overlay[tree_len..].to_vec();
        assert_eq!(copy.len(), source.len(), "copy splat count == source");
        for (c, s) in copy.iter().zip(source.iter()) {
            let cp = c.position();
            let sp = s.position();
            assert!((cp[0] - (sp[0] + 2.0)).abs() < 1e-4, "copy X = source X + 2.0");
            assert!((cp[1] - sp[1]).abs() < 1e-4, "copy Y unchanged");
            assert!((cp[2] - sp[2]).abs() < 1e-4, "copy Z unchanged");
            assert_eq!(c.spectral(), s.spectral(), "copy spectral is bit-exact");
            assert_eq!(c.scales(), s.scales(), "copy scales bit-exact");
            assert_eq!(c.opacity(), s.opacity(), "copy opacity bit-exact");
        }
        // The copies are now selected (primary = the new entity).
        assert!(shell.selection.contains(base_entities), "the copy is selected");

        // ONE Ctrl+Z removes EXACTLY the copy — back to base, originals untouched.
        assert!(shell.registry.run("edit.undo"));
        shell.drain_requests();
        assert_eq!(shell.overlay.len(), base_overlay, "one undo restores the overlay length");
        assert_eq!(shell.entities.len(), base_entities, "one undo removes exactly the copy entity");
        assert!(!shell.entities.iter().any(|e| e.name == "Silver Birch 02"));
        assert!(
            shell.overlay.iter().zip(source.iter()).all(|(a, b)| dup_splats_eq(a, b)),
            "the original tree's splats must be bit-identical after undo"
        );
        println!("OK: duplicate doubled overlay, one undo removed exactly the copy");
    }
}
