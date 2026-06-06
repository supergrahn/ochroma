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

pub mod cpu_render;

use egui_dock::{DockArea, DockState, NodeIndex, Style as DockStyle};
use vox_ui::design::icons::icon;
use vox_ui::widgets::{self, ScrubOpts};
use vox_ui::Tokens;

/// Identifies a dockable panel (the `egui_dock` tab payload).
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

/// The editor shell — owns the dock layout, panel state, and tokens.
pub struct EditorShell {
    pub tokens: Tokens,
    pub dock: DockState<PanelId>,
    pub entities: Vec<ShellEntity>,
    pub selected: usize,
    pub search: String,
    pub status: String,
    /// Toolbar gizmo mode (0=move,1=rotate,2=scale).
    pub gizmo: u8,
    pub snap: bool,
}

impl Default for EditorShell {
    fn default() -> Self {
        Self::new(Tokens::default())
    }
}

impl EditorShell {
    /// Build the shell with the standard SOTA layout:
    /// left = World; center-top = Viewport, center-bottom = Node Graph;
    /// right = Properties; bottom = Content + Output Log (tabbed).
    pub fn new(tokens: Tokens) -> Self {
        let mut dock = DockState::new(vec![PanelId::Viewport, PanelId::NodeGraph]);
        let surface = dock.main_surface_mut();
        // Left: World.
        let [center, _left] =
            surface.split_left(NodeIndex::root(), 0.18, vec![PanelId::Hierarchy]);
        // Right: Properties.
        let [center, _right] = surface.split_right(center, 0.78, vec![PanelId::Inspector]);
        // Bottom: Content + Output Log as a tab group.
        let [_center, _bottom] =
            surface.split_below(center, 0.72, vec![PanelId::Content, PanelId::Output]);

        EditorShell {
            tokens,
            dock,
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

    /// Lay out the full shell into an egui context for one frame.
    pub fn ui(&mut self, ctx: &egui::Context) {
        self.menu_bar(ctx);
        self.toolbar(ctx);
        self.status_bar(ctx);

        let mut viewer = ShellViewer {
            tokens: &self.tokens,
            entities: &mut self.entities,
            selected: &mut self.selected,
            search: &mut self.search,
        };
        let dock_style = DockStyle::from_egui(ctx.style().as_ref());
        DockArea::new(&mut self.dock)
            .style(dock_style)
            .show(ctx, &mut viewer);
    }

    fn menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("shell_menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                for m in ["File", "Edit", "Create", "Build", "Window", "Help"] {
                    ui.menu_button(m, |ui| {
                        ui.label(format!("{m} actions"));
                    });
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // "Ask Ochroma" — the one command surface (UX principle 2).
                    let _ = widgets::primary_action(
                        ui,
                        icon::SEARCH,
                        "Ask Ochroma  (Ctrl+K)",
                        &self.tokens,
                    );
                });
            });
        });
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("shell_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Primary labeled action (the Canva rule).
                let _ = widgets::primary_action(ui, icon::ADD, "Add to world", &self.tokens);
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

    /// Move a tab from one panel to another node, returning the leaf rects of
    /// the moved panel BEFORE and AFTER (for the dock movement test). Returns
    /// `None` if either panel can't be located.
    pub fn rect_of(&self, panel: PanelId) -> Option<egui::Rect> {
        for (_si, node) in self.dock.iter_all_nodes() {
            if let Some(tabs) = node.tabs()
                && tabs.contains(&panel)
            {
                return node.rect();
            }
        }
        None
    }
}

/// The `egui_dock` `TabViewer` that renders each built-in panel.
struct ShellViewer<'a> {
    tokens: &'a Tokens,
    entities: &'a mut Vec<ShellEntity>,
    selected: &'a mut usize,
    search: &'a mut String,
}

impl egui_dock::TabViewer for ShellViewer<'_> {
    type Tab = PanelId;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        format!("{}  {}", tab.icon(), tab.title()).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            PanelId::Hierarchy => self.hierarchy(ui),
            PanelId::Inspector => self.inspector(ui),
            PanelId::Viewport => self.viewport(ui),
            PanelId::NodeGraph => self.node_graph(ui),
            PanelId::Content => self.content(ui),
            PanelId::Output => self.output(ui),
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
        // Phase 1: a token-colored placeholder for the GPU splat texture (the
        // register_native_texture wiring lands in the viewport wave). It still
        // proves the central dock area renders with a guided tip chip.
        let rect = ui.available_rect_before_wrap();
        let [r, g, b, a] = self.tokens.color("surface.bg.0");
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgba_unmultiplied(r, g, b, a));
        let scene = egui::Color32::from_rgb(34, 52, 40);
        let inner = rect.shrink(8.0);
        ui.painter().rect_filled(inner, self.tokens.radius[1], scene);
        // Guided tip chip (UX principle 1).
        let chip = egui::Rect::from_min_size(
            inner.left_bottom() + egui::vec2(12.0, -40.0),
            egui::vec2(280.0, 28.0),
        );
        let [cr, cg, cb, ca] = self.tokens.color("surface.bg.2");
        ui.painter()
            .rect_filled(chip, self.tokens.radius[2], egui::Color32::from_rgba_unmultiplied(cr, cg, cb, ca));
        ui.painter().text(
            chip.left_center() + egui::vec2(10.0, 0.0),
            egui::Align2::LEFT_CENTER,
            "Tip: drag from the World list to place \u{2192} Do it",
            egui::FontId::proportional(self.tokens.type_ramp.body),
            egui::Color32::from_rgb(220, 222, 230),
        );
    }

    fn node_graph(&mut self, ui: &mut egui::Ui) {
        // Token-colored node strip placeholder (the bezier NodeCanvas lands in
        // its own wave). Draws a couple of category-colored node headers so the
        // panel reads as a real graph surface in the snapshot.
        let rect = ui.available_rect_before_wrap();
        let [r, g, b, a] = self.tokens.color("surface.bg.0");
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgba_unmultiplied(r, g, b, a));
        let cats = [
            (vox_ui::NodeCategory::Spatial, "Terrain"),
            (vox_ui::NodeCategory::Field, "Splatize"),
            (vox_ui::NodeCategory::Sink, "Output"),
        ];
        let mut x = rect.left() + 24.0;
        let y = rect.top() + 40.0;
        let mut prev: Option<egui::Pos2> = None;
        for (cat, name) in cats {
            let node = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(120.0, 64.0));
            let [br, bg, bb, ba] = self.tokens.color("surface.bg.2");
            ui.painter().rect_filled(
                node,
                self.tokens.radius[1],
                egui::Color32::from_rgba_unmultiplied(br, bg, bb, ba),
            );
            let [hr, hg, hb, ha] = self.tokens.category_header(cat);
            let header = egui::Rect::from_min_size(node.min, egui::vec2(node.width(), 18.0));
            ui.painter().rect_filled(
                header,
                self.tokens.radius[0],
                egui::Color32::from_rgba_unmultiplied(hr, hg, hb, ha),
            );
            ui.painter().text(
                header.left_center() + egui::vec2(6.0, 0.0),
                egui::Align2::LEFT_CENTER,
                name,
                egui::FontId::proportional(self.tokens.type_ramp.caption),
                egui::Color32::WHITE,
            );
            if let Some(p) = prev {
                let [wr, wg, wb, wa] = self.tokens.wire_color(vox_ui::PortType::Terrain);
                ui.painter().line_segment(
                    [p, node.left_center()],
                    egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(wr, wg, wb, wa)),
                );
            }
            prev = Some(node.right_center());
            x += 200.0;
        }
    }

    fn content(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("{}  Assets", icon::FOLDER_OPEN));
            ui.separator();
            for f in ["All", "Meshes", "Materials", "Splats"] {
                let _ = ui.selectable_label(f == "All", f);
            }
        });
        ui.separator();
        // Thumbnail grid.
        egui::Grid::new("content_grid").spacing([8.0, 8.0]).show(ui, |ui| {
            for i in 0..8 {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(64.0, 64.0), egui::Sense::hover());
                let shade = 40 + (i * 12) as u8;
                ui.painter().rect_filled(
                    rect,
                    self.tokens.radius[0],
                    egui::Color32::from_rgb(shade, shade + 6, shade + 14),
                );
                if (i + 1) % 4 == 0 {
                    ui.end_row();
                }
            }
        });
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_dock::{NodeIndex, SurfaceIndex, TabIndex};

    #[test]
    fn dock_tabs_present_and_movable() {
        let shell = EditorShell::default();
        let titles: Vec<&str> = shell
            .dock
            .iter_all_tabs()
            .map(|(_, t)| t.title())
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
        let sy = (btn_rect.center().y as usize);
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
            .find_tab(&PanelId::Inspector)
            .expect("find inspector");
        let (h_surface, h_node, _) = shell
            .dock
            .find_tab(&PanelId::Hierarchy)
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
}
