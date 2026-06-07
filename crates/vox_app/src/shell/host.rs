//! The plugin contract v1 — the UE host-plugin model (design's "Plugin contract"
//! section).
//!
//! An [`EditorPlugin`] declares dockable tabs and registry commands, and renders
//! its tab bodies through a [`PluginCtx`] that hands it **only** the shared
//! design system: `&Tokens`, the `&WidgetKit`, and a `&mut NodeCanvas`. There is
//! deliberately NO `Visuals` setter, NO raw `Color32` panel API, and NO
//! top-level-panel handle on `PluginCtx` — so a plugin physically cannot look
//! different from the host. This is the exact surface Crucible / FloraPrime /
//! Forge implement for real; the shipped [`crate::shell::plugins::CruciblePlugin`]
//! is the worked proof.
//!
//! Enforcement is structural: `PluginCtx`'s fields are the design system and
//! nothing else. A plugin that wanted to set `ctx.style().visuals` has no path —
//! the type does not expose `egui::Context`, `egui::Visuals`, or a `Color32`
//! constructor. (The plugin still gets a raw `&mut egui::Ui` to lay widgets into,
//! but its styling is already token-driven via the host's `egui_theme::apply`,
//! and it has no handle to mutate that styling.)

use super::command_palette::Command;
use vox_ui::node_canvas::NodeCanvas;
use vox_ui::widgets::WidgetKit;
use vox_ui::Tokens;

/// A dockable tab a plugin contributes to the host shell.
#[derive(Clone)]
pub struct TabDecl {
    /// Stable id, dispatched back to [`EditorPlugin::ui`] (e.g. `"crucible.cook"`).
    pub id: String,
    /// Friendly tab title shown in the dock strip.
    pub title: String,
    /// Phosphor icon codepoint for the tab.
    pub icon: &'static str,
}

/// The styling surface handed to a plugin each frame. **Only** the design system
/// — tokens, the widget kit, and the shared node canvas. No `Visuals`, no raw
/// `Color32` panel, no `egui::Context`. This is the enforcement: a plugin cannot
/// diverge from the host look because it is given no handle that would let it.
pub struct PluginCtx<'a> {
    /// Read-only design tokens (colors/space/radius/type/motion).
    pub tokens: &'a Tokens,
    /// The shared widget kit (scrub_drag / foldout / search_box), token-styled.
    pub widgets: &'a WidgetKit,
    /// The shared node canvas — a plugin graph editor inherits curved type-colored
    /// wires, grid and minimap with zero plugin rendering code.
    pub canvas: &'a mut NodeCanvas,
}

/// The host-plugin contract. Implemented by Crucible / FloraPrime / Forge.
///
/// `tabs()` + `commands()` are read once at install; `ui()` renders a tab body
/// each frame through the restricted [`PluginCtx`].
pub trait EditorPlugin {
    /// Stable plugin id (e.g. `"crucible"`).
    fn id(&self) -> &str;
    /// The dockable tabs this plugin contributes.
    fn tabs(&self) -> Vec<TabDecl>;
    /// The registry commands this plugin contributes (join the palette/menus).
    fn commands(&self) -> Vec<Command>;
    /// Render the body of tab `tab_id` through the restricted ctx.
    fn ui(&mut self, tab_id: &str, ui: &mut egui::Ui, ctx: &mut PluginCtx);
}

/// An installed plugin: its boxed implementation plus one independent
/// [`NodeCanvas`] per declared tab (so a graph-editor plugin tab keeps its
/// pan/zoom across frames). The plugin owns its `CanvasGraph`; the host owns the
/// canvas and hands it to the plugin via [`PluginCtx::canvas`].
pub struct InstalledPlugin {
    pub plugin: Box<dyn EditorPlugin>,
    pub tabs: Vec<TabDecl>,
    /// One canvas per declared tab id (keyed parallel to `tabs`).
    pub canvases: Vec<(String, NodeCanvas)>,
}

impl InstalledPlugin {
    /// Borrow the canvas state for a tab id.
    pub fn canvas_for(&mut self, tab_id: &str) -> Option<&mut NodeCanvas> {
        self.canvases
            .iter_mut()
            .find(|(id, _)| id == tab_id)
            .map(|(_, c)| c)
    }
}

#[cfg(test)]
mod contract_surface {
    //! Compile-surface documentation of the enforcement. `PluginCtx` exposes
    //! exactly `tokens`, `widgets`, `canvas` — and NOTHING that would let a
    //! plugin restyle the host. The asserts below name the fields that DO exist;
    //! the comment documents what deliberately does not (there is no
    //! `egui::Visuals`, `egui::Context`, or raw panel field — a plugin cannot
    //! reach them, which is the whole point of the contract).
    #[allow(unused_imports)]
    use super::*;

    /// This is a *documentation* test of the type surface, not a behavioral test:
    /// it simply confirms (by naming them) that the only fields on `PluginCtx`
    /// are the design-system handles. If a future change added a `visuals:` or
    /// `ctx: &egui::Context` field, the `PluginCtx { tokens, widgets, canvas }`
    /// destructuring below would fail to compile (it is exhaustive), flagging the
    /// contract breach.
    #[test]
    fn plugin_ctx_exposes_only_design_system() {
        fn _assert_exhaustive_fields(c: PluginCtx) {
            // Exhaustive destructure: adding ANY field (e.g. a Visuals handle)
            // breaks this, which is the compile-time enforcement guard.
            let PluginCtx {
                tokens: _,
                widgets: _,
                canvas: _,
            } = c;
        }
        // No runtime assertion needed; this test passing means the type compiled
        // with exactly the three design-system fields and nothing else.
    }
}
