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
pub mod forge_process;
pub mod crucible_native;
pub mod play;
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
    /// AAA Spec 06: persist the whole world (entities + their lossless overlay
    /// splat geometry) to `path`. Queued by `file.save` (Ctrl+S); the drain arm
    /// calls [`EditorShell::save_world`] and pushes an Output Log receipt.
    SaveWorld(PathBuf),
    /// AAA Spec 06: load a saved world from `path`, resetting the scene and
    /// REPLAYING each entity's splats through `plant_asset`. Queued by `file.open`
    /// (Ctrl+O); the drain arm calls [`EditorShell::load_world`] + a receipt.
    OpenWorld(PathBuf),
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
                ShellRequest::SaveWorld(path) => {
                    let line = match self.save_world(&path) {
                        Ok((entities, splats)) => format!(
                            "[save] Saved {entities} entities and {splats} splats to {}",
                            path.display()
                        ),
                        Err(e) => format!("[save] Couldn't save to {}: {e}", path.display()),
                    };
                    self.push_output_log(line);
                }
                ShellRequest::OpenWorld(path) => {
                    let line = match self.load_world(&path) {
                        Ok((entities, splats)) => format!(
                            "[open] Loaded {entities} entities and {splats} splats from {}",
                            path.display()
                        ),
                        Err(e) => format!("[open] Couldn't open {}: {e}", path.display()),
                    };
                    self.push_output_log(line);
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

    /// AAA Spec 06 — persist the WHOLE world to `path`: one [`SavedEntity`] per
    /// [`ShellEntity`], and for every entity that owns an overlay range (recorded
    /// by Spec 09's `asset_range`) its EXACT splat slice mapped LOSSLESSLY through
    /// [`vox_data::to_saved_geom`] into `geom_splats`. Rangeless entities (the seed
    /// rows: Townhouse, Sun, Camera…) save as bare records with no splats. The
    /// 16-band `u16` spectral and `i16` rotation are copied verbatim, so a later
    /// [`Self::load_world`] reproduces the overlay byte-identically. Returns
    /// `(entity_count, total_splats_written)`. NO-PANIC: an I/O / serialize failure
    /// is an `Err(String)`.
    pub fn save_world(&self, path: &std::path::Path) -> Result<(usize, usize), String> {
        let mut save = vox_data::world_save::WorldSave::new("project");
        let mut total_splats = 0usize;
        for entity in &self.entities {
            let mut saved = vox_data::world_save::SavedEntity::new(&entity.name, entity.pos);
            saved.tags = vec![entity.kind.clone()];
            // Spec 09 provenance: this entity OWNS overlay[start..start+len]. Slice
            // it and map each splat through the lossless codec. A rangeless entity
            // (seed rows) leaves geom_splats empty.
            if let Some(r) = entity.asset_range {
                let end = (r.start + r.len).min(self.overlay.len());
                let start = r.start.min(end);
                let slice = &self.overlay[start..end];
                saved.geom_splats = slice.iter().map(vox_data::to_saved_geom).collect();
                total_splats += saved.geom_splats.len();
            }
            save.add_entity(saved);
        }
        save.save_to_file(path)?;
        Ok((self.entities.len(), total_splats))
    }

    /// AAA Spec 06 — load a saved world from `path`, REPLACING the current scene.
    /// Resets every piece of scene state (entities, overlay, undo stack,
    /// per-label counters, selection), then for each [`SavedEntity`]: if it carries
    /// `geom_splats`, reconstruct them through [`vox_data::from_saved_geom`] and
    /// REPLAY them through [`Self::plant_asset`] — so `asset_counts`, the undo
    /// stack, the per-entity `asset_range`, and the viewport cache all stay
    /// coherent (NEVER poke `self.overlay` directly). A rangeless saved entity
    /// becomes a bare [`ShellEntity`]. The trailing " NN" placement number is
    /// stripped from the saved name to recover the plant label (re-numbering on
    /// replay is expected — geometry is what must be bit-exact). Returns
    /// `(entity_count, total_splats_loaded)`. NO-PANIC: a missing / corrupt file is
    /// an `Err(String)`.
    pub fn load_world(&mut self, path: &std::path::Path) -> Result<(usize, usize), String> {
        let save = vox_data::world_save::WorldSave::load_from_file(path)?;
        // Reset the scene state so the load is a clean replace, not an append.
        self.entities.clear();
        self.overlay.clear();
        self.undo_stack.clear();
        self.asset_counts.clear();
        self.selection = Selection::single(0);
        self.viewport_tex = None;
        self.last_inspector_edit = None;

        let mut total_splats = 0usize;
        for saved in &save.entities {
            let kind = saved.tags.first().cloned().unwrap_or_else(|| "mesh".to_string());
            if saved.geom_splats.is_empty() {
                // A rangeless entity (a seed row) — bare ShellEntity, no replay.
                self.entities.push(ShellEntity {
                    name: saved.name.clone(),
                    kind,
                    pos: saved.position,
                    asset_range: None,
                });
            } else {
                // Reconstruct the splats losslessly and REPLAY through plant_asset so
                // all Spec 07/09 bookkeeping (counts, undo, asset_range, cache) stays
                // coherent. Strip the trailing " NN" to recover the plant label.
                let splats: Vec<GaussianSplat> =
                    saved.geom_splats.iter().map(vox_data::from_saved_geom).collect();
                total_splats += splats.len();
                let label = dup_label(&saved.name);
                self.plant_asset(&label, &kind, splats, saved.position, "Loaded", "");
            }
        }
        Ok((self.entities.len(), total_splats))
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
    // Save / open the project world (AAA Spec 06) route through the registry like
    // undo: each closure queues a request the shell drains (real file I/O needs
    // `&mut self`). No file-dialog dependency — a fixed CWD path for now.
    let q = requests.clone();
    r.add(Command::new("file.save", "Save world", "File", "Ctrl+S", move || {
        q.borrow_mut().push(ShellRequest::SaveWorld(PathBuf::from("project.ochroma_world")))
    }));
    let q = requests.clone();
    r.add(Command::new("file.open", "Open world…", "File", "Ctrl+O", move || {
        q.borrow_mut().push(ShellRequest::OpenWorld(PathBuf::from("project.ochroma_world")))
    }));
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
#[path = "tests/mod.rs"]
mod tests;
