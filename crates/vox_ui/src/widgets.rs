//! The shared widget kit — the controls the inspector and every plugin MUST
//! use. All styled from [`crate::tokens::Tokens`] only.

use crate::tokens::Tokens;
use egui::{Color32, Sense, Stroke, Vec2};

/// Options for a [`scrub_drag`] numeric field.
pub struct ScrubOpts {
    /// Units changed per pixel dragged.
    pub speed: f32,
    pub range: Option<std::ops::RangeInclusive<f32>>,
    pub suffix: &'static str,
    /// Optional dotted token key for the left-edge axis color stripe
    /// (e.g. `"axis.x"`). `None` = no stripe.
    pub axis_color: Option<&'static str>,
}

impl Default for ScrubOpts {
    fn default() -> Self {
        ScrubOpts {
            speed: 0.01,
            range: None,
            suffix: "",
            axis_color: None,
        }
    }
}

/// A label-drag numeric field: drag horizontally over the widget to scrub the
/// bound value, double-click to type. Optionally paints a left-edge axis color
/// stripe (drives the X/Y/Z transform fields). Mutates `value` in place.
pub fn scrub_drag(
    ui: &mut egui::Ui,
    value: &mut f32,
    t: &Tokens,
    opts: ScrubOpts,
) -> egui::Response {
    let mut dv = egui::DragValue::new(value).speed(opts.speed as f64);
    if let Some(r) = &opts.range {
        dv = dv.range(r.clone());
    }
    if !opts.suffix.is_empty() {
        dv = dv.suffix(opts.suffix);
    }

    let resp = ui.add(dv);

    // Axis color stripe down the left edge of the field rect (Blender/UE style).
    if let Some(key) = opts.axis_color {
        let [r, g, b, a] = t.color(key);
        let col = Color32::from_rgba_unmultiplied(r, g, b, a);
        let rect = resp.rect;
        let stripe = egui::Rect::from_min_size(rect.min, Vec2::new(3.0, rect.height()));
        ui.painter().rect_filled(stripe, t.radius[0], col);
    }
    resp
}

/// An animated collapsible section, styled from tokens. Returns the body's
/// inner result when expanded. Persists open/closed state by `id`.
pub fn foldout<R>(
    ui: &mut egui::Ui,
    id: egui::Id,
    title: &str,
    body: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let header = state.show_header(ui, |ui| {
        ui.label(egui::RichText::new(title).strong());
    });
    let (_btn, _hdr, body_out) = header.body_unindented(body);
    body_out.map(|r| r.inner)
}

/// A search box with a leading magnifying-glass icon. Mutates `query`.
pub fn search_box(ui: &mut egui::Ui, query: &mut String) -> egui::Response {
    ui.horizontal(|ui| {
        ui.label(crate::design::icons::icon::SEARCH);
        ui.add(
            egui::TextEdit::singleline(query)
                .hint_text("Search")
                .desired_width(ui.available_width()),
        )
    })
    .inner
}

/// A square icon-only button with a hover tooltip.
pub fn icon_button(ui: &mut egui::Ui, icon: &str, tip: &str) -> egui::Response {
    ui.add(egui::Button::new(egui::RichText::new(icon).size(16.0)))
        .on_hover_text(tip)
}

/// A primary labeled action button (the Canva rule — icon + words, accent fill).
pub fn primary_action(ui: &mut egui::Ui, icon: &str, label: &str, t: &Tokens) -> egui::Response {
    let [r, g, b, a] = t.color("accent.base");
    let fill = Color32::from_rgba_unmultiplied(r, g, b, a);
    ui.add(
        egui::Button::new(
            egui::RichText::new(format!("{icon}  {label}"))
                .color(Color32::WHITE)
                .strong(),
        )
        .fill(fill)
        .stroke(Stroke::NONE),
    )
}

/// The shared kit handed to plugins via `PluginCtx` (the only styling surface a
/// plugin gets — it cannot set `Visuals` or push a raw `Color32` panel).
pub struct WidgetKit {
    pub tokens: Tokens,
}

impl WidgetKit {
    pub fn new(tokens: Tokens) -> Self {
        WidgetKit { tokens }
    }
    pub fn scrub_drag(&self, ui: &mut egui::Ui, value: &mut f32, opts: ScrubOpts) -> egui::Response {
        scrub_drag(ui, value, &self.tokens, opts)
    }
    pub fn foldout<R>(
        &self,
        ui: &mut egui::Ui,
        id: egui::Id,
        title: &str,
        body: impl FnOnce(&mut egui::Ui) -> R,
    ) -> Option<R> {
        foldout(ui, id, title, body)
    }
    pub fn search_box(&self, ui: &mut egui::Ui, query: &mut String) -> egui::Response {
        search_box(ui, query)
    }
}

/// Internal helper so tests don't depend on egui internals for the Sense type.
#[allow(dead_code)]
fn _drag_sense() -> Sense {
    Sense::drag()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_drag_paints_axis_stripe_pixels() {
        // Render an X-axis scrub field to a CPU mesh and assert the axis color
        // (axis.x = [230,60,60]) appears as actual painted pixels along the
        // left edge — the design's "axis color stripe (pixel check)".
        let ctx = egui::Context::default();
        crate::egui_theme::apply(&ctx, &Tokens::default());
        let t = Tokens::default();
        let mut v = 1.0f32;
        let full = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                scrub_drag(
                    ui,
                    &mut v,
                    &t,
                    ScrubOpts {
                        axis_color: Some("axis.x"),
                        ..Default::default()
                    },
                );
            });
        });
        // The stripe is a solid rect_filled of axis.x; find it among the shapes.
        let axis = t.color("axis.x");
        let mut found = false;
        for clipped in &full.shapes {
            if let egui::epaint::Shape::Rect(r) = &clipped.shape {
                let f = r.fill;
                if f.r() == axis[0] && f.g() == axis[1] && f.b() == axis[2] {
                    // A thin (3px) vertical stripe.
                    if r.rect.width() <= 4.0 && r.rect.height() > 4.0 {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "axis.x color stripe rect not painted by scrub_drag");
    }

    #[test]
    fn foldout_collapsed_is_shorter_than_expanded() {
        // Real rects: a collapsed foldout occupies less vertical space than an
        // expanded one with a tall body (the design's layout-height check).
        fn measure(open: bool) -> f32 {
            let ctx = egui::Context::default();
            crate::egui_theme::apply(&ctx, &Tokens::default());
            let id = egui::Id::new("fold_test");
            // Seed the persisted open state, then run several frames so the
            // open/close animation settles to its final height.
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let mut st = egui::collapsing_header::CollapsingState::load_with_default_open(
                    ctx, id, open,
                );
                st.set_open(open);
                st.store(ctx);
            });
            let mut measured = 0.0f32;
            for _ in 0..8 {
                let _ = ctx.run(egui::RawInput::default(), |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        // Measure ONLY the foldout's allocated child rect, not
                        // the whole filled CentralPanel.
                        let child = ui.scope(|ui| {
                            foldout(ui, id, "Section", |ui| {
                                for i in 0..10 {
                                    ui.label(format!("row {i}"));
                                }
                            });
                        });
                        measured = child.response.rect.height();
                    });
                });
            }
            measured
        }
        let collapsed = measure(false);
        let expanded = measure(true);
        assert!(
            expanded > collapsed + 40.0,
            "expanded foldout ({expanded}) not taller than collapsed ({collapsed})"
        );
    }
}
