//! Test harness for the editor shell, extracted verbatim from the
//! in-file `#[cfg(test)] mod tests` block (same module tree: `super`
//! resolves to the `shell` module, so privates remain reachable).

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
        let viewer = include_str!("../cpu_render.rs");
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

    /// AAA Spec 06 (HEADLINE): a grown tree saved to disk and reloaded into a FRESH
    /// editor reproduces its overlay BYTE-IDENTICALLY — the 16-band `u16` spectral
    /// and `i16` rotation match bit-for-bit, the entity comes back, and the splat
    /// counts are exact. This is the floor under every editor workflow.
    #[test]
    fn save_then_fresh_open_round_trips_tree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("project.ochroma_world");

        // --- Shell A: grow a tree, then save the world ---
        let mut a = EditorShell::default();
        let entities_before = a.entities.len();
        a.grow_tree_headless("Silver Birch", "broadleaf", 0);
        let grown = a.overlay.len();
        assert!(grown >= 200, "the grown tree must have >= 200 splats, got {grown}");
        // Bit-exact probes captured from the LIVE overlay before saving.
        let band7 = a.overlay[0].spectral_f32(6);
        let rot0 = a.overlay[0].rotation_raw();

        let (we, ws) = a.save_world(&path).unwrap();
        assert_eq!(we, a.entities.len(), "save reports one SavedEntity per ShellEntity");
        assert_eq!(ws, grown, "save reports exactly the grown splat count");

        // --- Shell B (FRESH): load the saved world ---
        let mut b = EditorShell::default();
        let (le, lsp) = b.load_world(&path).unwrap();
        assert_eq!(le, entities_before + 1, "load yields the seed rows + the one tree entity");
        assert_eq!(lsp, grown, "load reports exactly the grown splat count");
        assert_eq!(b.overlay.len(), grown, "the reloaded overlay has exactly the grown splats");

        // The lossless guarantee: spectral (u16/f16) and rotation (i16) are bit-exact.
        assert_eq!(
            b.overlay[0].spectral_f32(6), band7,
            "band 7 of splat 0 must be bit-identical after reload"
        );
        assert_eq!(
            b.overlay[0].rotation_raw(), rot0,
            "rotation i16 of splat 0 must be bit-identical after reload"
        );
        // Whole-overlay bit-exactness, not just splat 0.
        for (i, (loaded, orig)) in b.overlay.iter().zip(a.overlay.iter()).enumerate() {
            assert_eq!(loaded.spectral(), orig.spectral(), "splat {i} spectral bit-identical");
            assert_eq!(loaded.rotation_raw(), orig.rotation_raw(), "splat {i} rotation bit-identical");
            assert_eq!(loaded.position(), orig.position(), "splat {i} position match");
            assert_eq!(loaded.kind(), orig.kind(), "splat {i} kind match");
        }
        // The tree entity is back (re-numbered on replay, but the label survives).
        assert!(
            b.entities.iter().any(|e| e.name.contains("Silver Birch")),
            "a Silver Birch entity must exist after reload"
        );
        // load REPLAYS through plant_asset → the per-entity asset_range is coherent.
        let tree = b.entities.iter().find(|e| e.name.contains("Silver Birch")).unwrap();
        assert_eq!(
            tree.asset_range_for_test(),
            Some((0, grown)),
            "the reloaded tree owns overlay[0..grown] (asset_range replayed coherently)"
        );

        println!("OK: tree saved and reloaded bit-identical ({grown} splats)");
    }
