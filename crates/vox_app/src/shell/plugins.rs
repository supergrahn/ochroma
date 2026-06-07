//! Shipped editor plugins. Currently one: [`CruciblePlugin`] — the worked proof
//! of the host-plugin contract.
//!
//! `CruciblePlugin` hosts a SECOND, independent [`vox_ui::node_canvas`] graph
//! editor inside a dockable tab, labeled with REAL Crucible cook-node names
//! (read from `~/src/crucible/rust/crates/crucible-nodes`: `terrain`, `scatter`,
//! `sun_light`, `usd_export`). It renders entirely through the restricted
//! [`PluginCtx`] — `cx.canvas` + `cx.tokens` — so it inherits the host's curved
//! type-colored wires, dot grid, minimap, and category-colored headers with zero
//! plugin rendering code. Two plugin-hosted graph editors (this and the host's
//! own Node Graph) coexist with identical, token-inherited styling.
//!
//! Its commands register under the palette category `"Crucible"`.

use super::command_palette::Command;
use super::host::{EditorPlugin, PluginCtx, TabDecl};
use std::cell::RefCell;
use std::rc::Rc;
use vox_ui::design::icons::icon;
use vox_ui::node_canvas::{CanvasGraph, NodeView, WireView};
use vox_ui::{NodeCategory, PortType};

/// The tab id of the Crucible cook-graph editor.
pub const CRUCIBLE_TAB: &str = "crucible.cook";

/// The Crucible PCG plugin. Carries an observable flag flipped by its `recook`
/// command so the install test can prove the command actually executes, and owns
/// its OWN cook graph (a second, independent graph from the host's Node Graph).
pub struct CruciblePlugin {
    /// Flipped true when `crucible.recook` runs (the install/palette test reads it).
    pub recooked: Rc<RefCell<bool>>,
    /// This plugin's independent cook graph (real Crucible node names).
    graph: CanvasGraph,
}

impl Default for CruciblePlugin {
    fn default() -> Self {
        CruciblePlugin {
            recooked: Rc::new(RefCell::new(false)),
            graph: build_crucible_graph(),
        }
    }
}

impl CruciblePlugin {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Build the Crucible cook graph with REAL node names. Categories are passed (not
/// RGB) so headers/wires match the host's terrain/field/sink nodes automatically.
///
/// Shape (real crucible-nodes type_names):
///   terrain(Spatial) -> scatter(Generator) -> usd_export(Sink)
///   sun_light(Field) feeds the scatter context.
pub fn build_crucible_graph() -> CanvasGraph {
    let mut g = CanvasGraph::default();
    g.nodes.push({
        let mut n = NodeView::new(1, "terrain", NodeCategory::Spatial, egui::pos2(40.0, 110.0))
            .with_output("terrain", PortType::Terrain);
        n.size.x = 150.0;
        n
    });
    g.nodes.push({
        let mut n = NodeView::new(2, "sun_light", NodeCategory::Field, egui::pos2(40.0, 250.0))
            .with_output("light", PortType::SpectralField);
        n.size.x = 150.0;
        n
    });
    g.nodes.push({
        let mut n = NodeView::new(3, "scatter", NodeCategory::Generator, egui::pos2(260.0, 150.0))
            .with_input("terrain", PortType::Terrain)
            .with_input("light", PortType::SpectralField)
            .with_output("instances", PortType::Instances);
        n.size.x = 150.0;
        n
    });
    g.nodes.push({
        let mut n = NodeView::new(4, "usd_export", NodeCategory::Sink, egui::pos2(480.0, 160.0))
            .with_input("instances", PortType::Instances);
        n.size.x = 150.0;
        n
    });
    g.wires.push(WireView {
        from_node: 1, from_port: "terrain".into(),
        to_node: 3, to_port: "terrain".into(),
        exec: false, label: Some("Terrain 4096 cells".into()),
    });
    g.wires.push(WireView {
        from_node: 2, from_port: "light".into(),
        to_node: 3, to_port: "light".into(),
        exec: false, label: None,
    });
    g.wires.push(WireView {
        from_node: 3, from_port: "instances".into(),
        to_node: 4, to_port: "instances".into(),
        exec: false, label: Some("Instances 512".into()),
    });
    g
}

impl EditorPlugin for CruciblePlugin {
    fn id(&self) -> &str {
        "crucible"
    }

    fn tabs(&self) -> Vec<TabDecl> {
        vec![TabDecl {
            id: CRUCIBLE_TAB.to_string(),
            title: "Crucible".to_string(),
            icon: icon::TERRAIN,
        }]
    }

    fn commands(&self) -> Vec<Command> {
        let f = self.recooked.clone();
        vec![
            Command::new(
                "crucible.recook",
                "Crucible: Recook",
                "Crucible",
                "",
                move || *f.borrow_mut() = true,
            ),
            Command::new("crucible.export_usd", "Crucible: Export USD", "Crucible", "", || {}),
        ]
    }

    fn ui(&mut self, tab_id: &str, ui: &mut egui::Ui, cx: &mut PluginCtx) {
        if tab_id != CRUCIBLE_TAB {
            return;
        }
        // The plugin renders its OWN graph through the SHARED canvas + host
        // tokens — it sets not a single color. `cx.canvas` is the per-tab
        // NodeCanvas the host owns on its behalf; `cx.tokens` is the host design
        // system. The Crucible editor therefore inherits curved type-colored
        // wires, the dot grid, the minimap and category-colored headers with zero
        // plugin rendering code — identical styling to the host's Node Graph.
        let _ = cx.canvas.ui(ui, cx.tokens, &mut self.graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_declares_tab_and_commands() {
        let p = CruciblePlugin::new();
        assert_eq!(p.id(), "crucible");
        let tabs = p.tabs();
        assert_eq!(tabs.len(), 1);
        assert_eq!(tabs[0].title, "Crucible");

        let cmds = p.commands();
        assert!(cmds.iter().any(|c| c.id == "crucible.recook" && c.category == "Crucible"));
    }

    #[test]
    fn recook_command_flips_observable_flag() {
        let p = CruciblePlugin::new();
        let cmds = p.commands();
        let recook = cmds.iter().find(|c| c.id == "crucible.recook").unwrap();
        assert!(!*p.recooked.borrow());
        (recook.run)();
        assert!(*p.recooked.borrow(), "running crucible.recook must flip the flag");
    }

    #[test]
    fn crucible_graph_uses_real_node_names() {
        let g = build_crucible_graph();
        let names: Vec<&str> = g.nodes.iter().map(|n| n.title.as_str()).collect();
        for real in ["terrain", "scatter", "sun_light", "usd_export"] {
            assert!(names.contains(&real), "missing real Crucible node {real}; have {names:?}");
        }
    }
}
