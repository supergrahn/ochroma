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

// ============================================================================
// Forge plugin — the SECOND real plugin. Coexists with Crucible: two plugin tabs
// + two palette command categories ("Crucible" and "Forge").
// ============================================================================

/// The tab id of the Forge generator-graph editor.
pub const FORGE_TAB: &str = "forge.canvas";

/// The REAL Forge generator command names, read from
/// `~/src/aetherspectra/forge/crates/forge-cli/src/cmd/` (each exposes
/// `run(json) -> Result<ForgeVolume, String>`). The design's Forge wave wraps
/// these as one synthesized node per domain; here they drive the canvas labels +
/// the `forge.generate_*` palette commands. Categories are passed (not RGB) so
/// headers/wires color by role through the host tokens.
///
/// `(command_name, friendly_title, NodeCategory)`. Forge generators are spatial
/// world-builders, so they color `Spatial` (matching host terrain nodes) —
/// except `scatter`, which generates instances (`Generator`).
pub const FORGE_GENERATORS: &[(&str, &str, NodeCategory)] = &[
    ("terrain", "Terrain", NodeCategory::Spatial),
    ("building", "Building", NodeCategory::Spatial),
    ("scatter", "Scatter", NodeCategory::Generator),
    ("road", "Road", NodeCategory::Spatial),
    ("vegetation", "Vegetation", NodeCategory::Generator),
    ("water", "Water", NodeCategory::Field),
];

/// The Forge environment-generator plugin. Hosts a canvas of the real Forge
/// generators and registers `forge.generate_<domain>` commands under category
/// "Forge". Renders entirely through the restricted `PluginCtx` — it sets not a
/// single color, inheriting the host's tokens (the contract proof, mirroring
/// Crucible).
pub struct ForgePlugin {
    /// Records the last `forge.generate_*` command that ran (the install test
    /// reads it to prove the command executes).
    pub last_generated: Rc<RefCell<Option<String>>>,
    /// The Forge generator graph (real generator names as nodes).
    graph: CanvasGraph,
}

impl Default for ForgePlugin {
    fn default() -> Self {
        ForgePlugin {
            last_generated: Rc::new(RefCell::new(None)),
            graph: build_forge_graph(),
        }
    }
}

impl ForgePlugin {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Build the Forge generator graph with the REAL generator names. A terrain ->
/// road -> building -> scatter -> vegetation chain plus a water field, so the
/// canvas shows the real Forge domain pipeline with host-styled headers/wires.
pub fn build_forge_graph() -> CanvasGraph {
    let mut g = CanvasGraph::default();
    // Lay the six generators out in two rows.
    for (i, (name, title, cat)) in FORGE_GENERATORS.iter().enumerate() {
        let col = (i % 3) as f32;
        let row = (i / 3) as f32;
        let pos = egui::pos2(40.0 + col * 200.0, 90.0 + row * 150.0);
        let mut n = NodeView::new((i as u64) + 1, *title, *cat, pos)
            .with_output(*name, PortType::Terrain);
        n.size.x = 150.0;
        g.nodes.push(n);
    }
    // Wire terrain(1) -> road(4) and terrain(1) -> building(2) to show typed
    // wires between Forge generators (Terrain port type).
    g.wires.push(WireView {
        from_node: 1, from_port: "terrain".into(),
        to_node: 2, to_port: "terrain".into(),
        exec: false, label: Some("ForgeVolume".into()),
    });
    g.wires.push(WireView {
        from_node: 1, from_port: "terrain".into(),
        to_node: 4, to_port: "terrain".into(),
        exec: false, label: None,
    });
    g
}

impl EditorPlugin for ForgePlugin {
    fn id(&self) -> &str {
        "forge"
    }

    fn tabs(&self) -> Vec<TabDecl> {
        vec![TabDecl {
            id: FORGE_TAB.to_string(),
            title: "Forge".to_string(),
            icon: icon::TERRAIN,
        }]
    }

    fn commands(&self) -> Vec<Command> {
        FORGE_GENERATORS
            .iter()
            .map(|(name, title, _)| {
                let log = self.last_generated.clone();
                let domain = name.to_string();
                Command::new(
                    format!("forge.generate_{name}"),
                    format!("Forge: Generate {title}"),
                    "Forge",
                    "",
                    move || *log.borrow_mut() = Some(domain.clone()),
                )
            })
            .collect()
    }

    fn ui(&mut self, tab_id: &str, ui: &mut egui::Ui, cx: &mut PluginCtx) {
        if tab_id != FORGE_TAB {
            return;
        }
        // Render the Forge generator graph through the SHARED canvas + host tokens
        // — no plugin-set color. Inherits curved type-colored wires, grid, minimap,
        // category-colored headers (the contract proof, identical to Crucible).
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

    #[test]
    fn forge_plugin_declares_tab_and_real_generator_commands() {
        let p = ForgePlugin::new();
        assert_eq!(p.id(), "forge");
        let tabs = p.tabs();
        assert_eq!(tabs.len(), 1);
        assert_eq!(tabs[0].title, "Forge");
        assert_eq!(tabs[0].id, FORGE_TAB);

        // Every command is under category "Forge" and names a REAL Forge domain.
        let cmds = p.commands();
        for (name, _, _) in FORGE_GENERATORS {
            let id = format!("forge.generate_{name}");
            let c = cmds
                .iter()
                .find(|c| c.id == id)
                .unwrap_or_else(|| panic!("missing Forge command {id}"));
            assert_eq!(c.category, "Forge", "Forge command {id} must be in category 'Forge'");
        }
        // The real Forge generator names (read from forge-cli/src/cmd/) are present.
        let names: Vec<&str> = FORGE_GENERATORS.iter().map(|(n, _, _)| *n).collect();
        for real in ["terrain", "building", "scatter", "road", "vegetation", "water"] {
            assert!(names.contains(&real), "Forge generator {real} missing; have {names:?}");
        }
    }

    #[test]
    fn forge_generate_command_records_domain() {
        let p = ForgePlugin::new();
        let cmds = p.commands();
        let gen_terrain = cmds.iter().find(|c| c.id == "forge.generate_terrain").unwrap();
        assert!(p.last_generated.borrow().is_none());
        (gen_terrain.run)();
        assert_eq!(
            p.last_generated.borrow().as_deref(),
            Some("terrain"),
            "running forge.generate_terrain must record the terrain domain"
        );
    }

    #[test]
    fn forge_graph_uses_real_generator_names_and_categories() {
        let g = build_forge_graph();
        let titles: Vec<&str> = g.nodes.iter().map(|n| n.title.as_str()).collect();
        for real in ["Terrain", "Building", "Scatter", "Road", "Vegetation", "Water"] {
            assert!(titles.contains(&real), "Forge graph missing {real}; have {titles:?}");
        }
        // Scatter colors Generator; Terrain colors Spatial — categories drive the
        // host header tokens (never RGB).
        let scatter = g.nodes.iter().find(|n| n.title == "Scatter").unwrap();
        assert_eq!(scatter.category, NodeCategory::Generator);
        let terrain = g.nodes.iter().find(|n| n.title == "Terrain").unwrap();
        assert_eq!(terrain.category, NodeCategory::Spatial);
    }
}
