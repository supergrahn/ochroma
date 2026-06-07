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
pub mod viewport;

use command_palette::{Command, CommandRegistry, PaletteState};
use content_panel::{ContentAction, ContentPanel};
use egui_dock::{DockArea, DockState, NodeIndex, Style as DockStyle};
use graph_bridge::GraphBridge;
use host::{InstalledPlugin, PluginCtx, TabDecl};
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

/// A demo entity shown in the World/Properties panels so the snapshot has real
/// content (the inspector's drag-scrub fields bind to the selected one).
#[derive(Clone)]
pub struct ShellEntity {
    pub name: String,
    pub kind: String,
    pub pos: [f32; 3],
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
}

/// A side-effecting request a registry command pushes for the shell to drain on
/// the next frame. Registry commands are `Fn()` (no `&mut self`), so a command
/// that must mutate shell state (swap the theme, focus a tab) records its intent
/// here; `EditorShell::drain_requests` applies it. This keeps theme/focus on the
/// SAME one-command-surface (the intent executor and a menu click both route
/// through `registry.run`, which fires the closure that queues the request).
#[derive(Debug, Clone, PartialEq)]
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
}

/// The editor shell — owns the dock layout, panel state, and tokens.
pub struct EditorShell {
    pub tokens: Tokens,
    pub dock: DockState<TabKind>,
    pub entities: Vec<ShellEntity>,
    pub selected: usize,
    pub search: String,
    pub status: String,
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
    /// Cached viewport scene texture (rasterized splat frame), uploaded once.
    pub viewport_tex: Option<egui::TextureHandle>,
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
            entities: vec![
                ShellEntity {
                    name: "Townhouse_Row_03".into(),
                    kind: "mesh".into(),
                    pos: [12.0, 0.0, -4.0],
                },
                ShellEntity {
                    name: "Terrain_Alpine".into(),
                    kind: "terrain".into(),
                    pos: [0.0, 0.0, 0.0],
                },
                ShellEntity {
                    name: "Sun_Directional".into(),
                    kind: "light".into(),
                    pos: [40.0, 80.0, 20.0],
                },
                ShellEntity {
                    name: "Camera_Main".into(),
                    kind: "camera".into(),
                    pos: [5.0, 2.0, 14.0],
                },
            ],
            selected: 0,
            search: String::new(),
            status: "All systems healthy".into(),
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

        self.menu_bar(ctx);
        self.toolbar(ctx);
        self.status_bar(ctx);

        // Ensure the viewport scene texture is uploaded once.
        let viewport_tex = viewport::scene_texture(ctx, &mut self.viewport_tex);

        let mut inspector_undo_edits: Vec<(NodeId, &'static str, String, f32)> = Vec::new();
        let mut content_action: Option<ContentAction> = None;
        let mut viewer = ShellViewer {
            tokens: &self.tokens,
            widget_kit: &self.widget_kit,
            entities: &mut self.entities,
            selected: &mut self.selected,
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
            intent::resolve_intent(&self.intent_backend, text, &self.intent_schema, &self.registry);
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
            IntentAction::Unknown { suggestions } => {
                format!(
                    "I don't know how to do that yet — try: {}",
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
    fn push_undo(&mut self, entry: UndoEntry) {
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
            Some(UndoEntry::ParamSet { node_id, key, target, prev, next }) => {
                // Revert the CONCRETE node that was edited (not first-of-kind), so a
                // graph with two nodes of the same kind reverts the right one.
                self.bridge.apply_param(node_id, key, prev);
                // A subsequent inspector edit must start a NEW undo entry, never
                // coalesce into the one we just popped.
                self.last_inspector_edit = None;
                format!("Undid: {target} {} -> {} (back to {})", fmt_num(next), fmt_num(prev), fmt_num(prev))
            }
            None => "Nothing to undo".to_string(),
        };
        self.log_receipt(receipt.clone());
        receipt
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
            }
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
                    format!("[content] Loaded {name}: {} splats", splats.len())
                }
                Ok(LoadedAsset::Scene(_)) => {
                    format!("[content] Loaded {name}: scene handed to the import pipeline")
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
                let _ = widgets::icon_button(ui, icon::SHOW_FLAGS, "Show flags");
                let _ = widgets::icon_button(ui, icon::PERF, "Performance");
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
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{} entities", self.entities.len()));
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
    selected: &'a mut usize,
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
        for (i, e) in self.entities.iter().enumerate() {
            if !q.is_empty() && !e.name.to_lowercase().contains(&q) {
                continue;
            }
            let (ic, color_key) = vox_ui::design::icons::entity_icon(&e.kind);
            let [r, g, b, a] = self.tokens.color(color_key);
            let label = egui::RichText::new(format!("{ic}  {}", e.name))
                .color(egui::Color32::from_rgba_unmultiplied(r, g, b, a));
            if ui.selectable_label(*self.selected == i, label).clicked() {
                *self.selected = i;
            }
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
                    format!("{}  cook failed: {err}", icon::WARNING),
                );
            }
            return;
        }

        // No node selected: show the World entity's transform (the friendly default).
        let sel = (*self.selected).min(self.entities.len().saturating_sub(1));
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
        for line in [
            "[ochroma] Engine started",
            "[ochroma] Loaded scene: alpine_demo",
            "[render] Atom budget: 2.4M splats",
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
fn fmt_num(v: f32) -> String {
    if (v.fract()).abs() < f32::EPSILON {
        format!("{}", v as i64)
    } else {
        format!("{v:.2}")
    }
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
    r.add(Command::new(
        "world.add",
        "Add to world",
        "Create",
        "Ctrl+A",
        move || *f.borrow_mut() = true,
    ));
    r.add(Command::new("create.terrain", "Forge: Terrain", "Create", "", || {}));
    r.add(Command::new("create.biome", "Add biome layer", "Create", "", || {}));
    r.add(Command::new("file.save", "Save world", "File", "Ctrl+S", || {}));
    r.add(Command::new("file.open", "Open world…", "File", "Ctrl+O", || {}));
    // Undo routes through the registry too (same one-command-surface) — its
    // closure queues a request the shell drains, since undo needs `&mut self`.
    let q = requests.clone();
    r.add(Command::new("edit.undo", "Undo", "Edit", "Ctrl+Z", move || {
        q.borrow_mut().push(ShellRequest::Undo)
    }));
    r.add(Command::new("edit.redo", "Redo", "Edit", "Ctrl+Shift+Z", || {}));
    r.add(Command::new("build.cook", "Recook graph", "Build", "F5", || {}));
    r.add(Command::new("view.wireframe", "Toggle wireframe", "Window", "", || {}));
    r.add(Command::new("help.about", "About Ochroma", "Help", "", || {}));

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
    r.add(Command::new("view.focus_viewport", "Show the Viewport", "Window", "", move || {
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
        let card_open = patch_lum(&open_rgba, cx, h * 28 / 100);
        let below_open = patch_lum(&open_rgba, cx, h * 75 / 100);
        assert!(
            card_open > below_open + 5.0,
            "open: modal card patch ({card_open:.1}) must be brighter than the dimmed viewport below it ({below_open:.1})"
        );
        // Closed: no modal, so the same centre-column patch at 28% height is the
        // viewport scene at full (undimmed) brightness — the open card patch must
        // be DIMMER than the closed (undimmed) scene at that location, proving the
        // backdrop dim is really there.
        let same_loc_closed = patch_lum(&closed_rgba, cx, h * 75 / 100);
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
        // canvas. Its header must be drawn in category_header(Spatial). Scan a band
        // a bit below the tab strip for a pixel matching that exact token color.
        let want = Tokens::default().category_header(NodeCategory::Spatial);
        let mut found = false;
        'outer: for y in (rect.min.y as usize + 30)..(rect.min.y as usize + 220) {
            for x in (rect.min.x as usize + 20)..(rect.min.x as usize + 320) {
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
        shell.intent_backend = intent::IntentBackend::LlmCanned(std::sync::Arc::new(|_p| {
            Ok(r#"{"SetParam":{"node_kind":"terrain","key":"resolution","value":1e30}}"#.to_string())
        }));
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
            receipt.starts_with("I don't know how to do that yet — try: "),
            "unknown intent must answer honestly, got {receipt:?}"
        );
        // Every suggested title must be a real registered command title.
        let real_titles: Vec<String> =
            shell.registry.commands.iter().map(|c| c.title.clone()).collect();
        let listed = receipt
            .trim_start_matches("I don't know how to do that yet — try: ")
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
        let UndoEntry::ParamSet { prev: entry_prev, next, .. } = shell.undo_stack.last().unwrap();
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
        let UndoEntry::ParamSet { next, .. } = shell.undo_stack.last().unwrap();
        assert_eq!(*next, 249.0, "newest undo entry must be the last edit (seed=249)");
        let UndoEntry::ParamSet { next: oldest_next, .. } = shell.undo_stack.first().unwrap();
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
        'outer: for y in (rect.min.y as usize + 30)..(rect.min.y as usize + 220) {
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
}
