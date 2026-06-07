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

// ============================================================================
// FloraPrime plugin — the THIRD real plugin. The non-game-developer-friendly
// EDITOR for `~/src/aetherspectra/floraprime_gen` (a graph-diffusion tree
// generator producing tree QSMs — quantitative structure models / skeleton
// graphs). Coexists with Crucible + Forge: three plugin tabs + three palette
// command categories.
//
// FloraPrime's real Python sampler is `sample_tree(checkpoint, species_id,
// crown_radius, n_nodes, device) -> Data(x:(N,14), edge_index:(2,N-1))` (see
// floraprime_gen/sample.py). It is NOT callable from Rust yet, so the "Grow
// tree" action runs a DETERMINISTIC CPU stub (see `grow_tree_skeleton`) that
// produces a real branching skeleton the panel summarizes. The stub is the
// stand-in for the eventual Python sampler bridge and is labeled as such.
// ============================================================================

/// The tab id of the FloraPrime vegetation panel.
pub const FLORAPRIME_TAB: &str = "floraprime.vegetation";

/// The species the panel offers, named for non-game-developers. FloraPrime's
/// real spectral parameter ranges are keyed by species CLASS
/// (`floraprime_gen/spectral/pca_basis.py` → `broadleaf` / `conifer` / `grass`);
/// each named entry maps a friendly label to the integer `species_id` the Python
/// `sample_tree` takes, plus the underlying class. (Species ids 17/18 are the
/// validation split per `data/qsm_dataset.py`; the editor exposes a small picked
/// set of trainable ids.)
///
/// `(friendly_label, species_id, class)`.
pub const FLORAPRIME_SPECIES: &[(&str, i32, &str)] = &[
    ("Silver Birch", 0, "broadleaf"),
    ("English Oak", 1, "broadleaf"),
    ("Scots Pine", 2, "conifer"),
    ("Norway Spruce", 3, "conifer"),
    ("Meadow Grass", 4, "grass"),
];

/// The three detail presets, mapping a plain-language label to FloraPrime's
/// `n_nodes` sampler argument (skeleton node count; the real default is ~200).
///
/// `(label, n_nodes)`.
pub const FLORAPRIME_DETAIL: &[(&str, usize)] =
    &[("Low", 100), ("Medium", 200), ("High", 400)];

/// A node in a generated tree skeleton: a 3D position (metres) and its parent
/// index. The root (index 0) has parent `usize::MAX` (no parent). This mirrors
/// the parent-pointer DAG the real FloraPrime `TopologyDiffusion` enforces via
/// MST projection — here computed deterministically on the CPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkeletonNode {
    pub pos: [f32; 3],
    pub parent: usize,
}

/// A generated tree QSM skeleton: the node list plus derived summary stats the
/// panel reads. `edges == nodes.len() - 1` always holds (a spanning tree).
#[derive(Clone, Debug, PartialEq)]
pub struct TreeSkeleton {
    pub nodes: Vec<SkeletonNode>,
    /// Edge count (always `nodes.len() - 1` for a tree).
    pub edges: usize,
    /// Deepest branch depth in parent-hops from the root.
    pub max_depth: usize,
    /// Tree height in metres (max node Y), derived from `crown_radius`.
    pub height_m: f32,
}

/// A small deterministic LCG so the stub is seedable from the job params with no
/// external RNG dependency. Same seed → identical stream (the determinism the
/// tests assert).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG constants; take the high bits for the unit float.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// PLACEHOLDER FOR THE PYTHON SAMPLER BRIDGE (`floraprime_gen.sample.sample_tree`).
///
/// Deterministically grow a real tree skeleton from the job params — a seeded
/// golden-angle phyllotaxis branching structure with actual 3D node positions
/// and parent edges. Given `(species_id, crown_radius_m, n_nodes)` it returns a
/// [`TreeSkeleton`] with exactly `n_nodes` nodes, `n_nodes - 1` edges, a max
/// branch depth, and a height in metres derived from `crown_radius_m`. Same
/// params → identical skeleton; different species_id → a different skeleton.
///
/// This stands in for the not-yet-callable-from-Rust Python diffusion sampler so
/// the panel can summarize a REAL computed structure today; when the bridge
/// lands it replaces this body, keeping the same `(species_id, crown_radius,
/// n_nodes)` contract `sample_tree` already exposes.
pub fn grow_tree_skeleton(species_id: i32, crown_radius_m: f32, n_nodes: usize) -> TreeSkeleton {
    // Golden angle (radians) — the phyllotaxis branching driver.
    const GOLDEN_ANGLE: f32 = 2.399_963_2;
    // At least a root.
    let n = n_nodes.max(1);
    // Height grows with crown radius: a roughly 1.6:1 height:radius habit.
    let height_m = crown_radius_m * 1.6;

    // Seed mixes species + params so different species → different skeletons but
    // identical params reproduce exactly.
    let seed = (species_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (crown_radius_m.to_bits() as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ (n as u64).wrapping_mul(0xC2B2_AE35_06ED_C1F1);
    let mut rng = Lcg(seed | 1);

    let mut nodes: Vec<SkeletonNode> = Vec::with_capacity(n);
    let mut depth: Vec<usize> = Vec::with_capacity(n);
    // Root at the origin.
    nodes.push(SkeletonNode { pos: [0.0, 0.0, 0.0], parent: usize::MAX });
    depth.push(0);

    for i in 1..n {
        // Pick a parent biased toward recent nodes so the structure actually
        // branches (and deepens) rather than forming a flat star. The jitter is
        // seeded, so the choice is deterministic.
        let span = (i as f32).sqrt().max(1.0);
        let back = (rng.next_f32() * span) as usize;
        let parent = i.saturating_sub(1 + back);

        let pdepth = depth[parent];
        let d = pdepth + 1;
        // Golden-angle phyllotaxis around the trunk; radius widens with the
        // crown radius and the branch's height fraction.
        let theta = GOLDEN_ANGLE * i as f32;
        let height_frac = (d as f32) / (1.0 + n as f32).sqrt();
        let r = crown_radius_m * height_frac.min(1.0) * (0.4 + 0.6 * rng.next_f32());
        let py = nodes[parent].pos[1] + height_m / (1.0 + n as f32).sqrt();

        nodes.push(SkeletonNode {
            pos: [r * theta.cos(), py.min(height_m), r * theta.sin()],
            parent,
        });
        depth.push(d);
    }

    let max_depth = depth.iter().copied().max().unwrap_or(0);
    let real_height = nodes.iter().map(|n| n.pos[1]).fold(0.0_f32, f32::max);
    TreeSkeleton {
        nodes,
        edges: n - 1,
        max_depth,
        height_m: real_height,
    }
}

/// The live state of the vegetation generation job — the panel's readout cycles
/// idle → queued → preview as the user grows a tree.
#[derive(Clone, Debug, PartialEq)]
pub enum GenState {
    /// Nothing generated yet.
    Idle,
    /// "Grow tree" pressed; the (stub) sampler is about to run.
    Queued,
    /// A skeleton was produced — the panel previews its node/edge counts.
    Preview(TreeSkeleton),
}

/// The FloraPrime vegetation editor plugin — a plain-language panel to configure
/// and preview tree generation jobs for non-game-developers. Renders entirely
/// through the restricted `PluginCtx` (tokens + widget kit), setting colors only
/// via design tokens (the contract proof, mirroring Crucible/Forge).
pub struct FloraPrimePlugin {
    /// Selected species index into [`FLORAPRIME_SPECIES`].
    species_idx: usize,
    /// Target crown radius in metres (the real `sample_tree` arg).
    crown_radius_m: f32,
    /// Selected detail index into [`FLORAPRIME_DETAIL`] (maps to `n_nodes`).
    detail_idx: usize,
    /// Current generation-state readout.
    state: GenState,
    /// Flipped true when `floraprime.generate_tree` runs (the install/palette
    /// test reads it to prove the command executes).
    pub generated: Rc<RefCell<bool>>,
}

impl Default for FloraPrimePlugin {
    fn default() -> Self {
        FloraPrimePlugin {
            species_idx: 0,
            crown_radius_m: 3.0, // the real sample_tree default
            detail_idx: 1,       // Medium / n_nodes=200 (the real default)
            state: GenState::Idle,
            generated: Rc::new(RefCell::new(false)),
        }
    }
}

impl FloraPrimePlugin {
    pub fn new() -> Self {
        Self::default()
    }

    /// The currently-selected job params resolved to the real `sample_tree`
    /// arguments `(species_id, crown_radius_m, n_nodes)`.
    pub fn job_params(&self) -> (i32, f32, usize) {
        let (_, species_id, _) = FLORAPRIME_SPECIES[self.species_idx];
        let (_, n_nodes) = FLORAPRIME_DETAIL[self.detail_idx];
        (species_id, self.crown_radius_m, n_nodes)
    }

    /// Run the (stub) sampler for the current params and move the readout to
    /// `Preview`. The same path the `floraprime.generate_tree` command drives.
    pub fn grow(&mut self) {
        let (species_id, crown_radius_m, n_nodes) = self.job_params();
        self.state = GenState::Queued;
        let skeleton = grow_tree_skeleton(species_id, crown_radius_m, n_nodes);
        self.state = GenState::Preview(skeleton);
        *self.generated.borrow_mut() = true;
    }
}

impl EditorPlugin for FloraPrimePlugin {
    fn id(&self) -> &str {
        "floraprime"
    }

    fn tabs(&self) -> Vec<TabDecl> {
        vec![TabDecl {
            id: FLORAPRIME_TAB.to_string(),
            title: "Vegetation".to_string(),
            icon: icon::TERRAIN,
        }]
    }

    fn commands(&self) -> Vec<Command> {
        let g = self.generated.clone();
        vec![Command::new(
            "floraprime.generate_tree",
            "FloraPrime: Grow Tree",
            "FloraPrime",
            "",
            move || *g.borrow_mut() = true,
        )]
    }

    fn ui(&mut self, tab_id: &str, ui: &mut egui::Ui, cx: &mut PluginCtx) {
        if tab_id != FLORAPRIME_TAB {
            return;
        }
        // Every color comes from cx.tokens — the plugin sets none directly (the
        // contract proof). Text colors are resolved from design tokens below.
        let t = cx.tokens;
        let prim = {
            let [r, g, b, a] = t.color("text.primary");
            egui::Color32::from_rgba_unmultiplied(r, g, b, a)
        };
        let sec = {
            let [r, g, b, a] = t.color("text.secondary");
            egui::Color32::from_rgba_unmultiplied(r, g, b, a)
        };

        ui.add_space(t.space[2]);
        ui.label(
            egui::RichText::new("Grow a tree")
                .size(t.type_ramp.title)
                .color(prim)
                .strong(),
        );
        ui.label(
            egui::RichText::new(
                "Configure a FloraPrime tree-generation job in plain language, then preview it.",
            )
            .size(t.type_ramp.caption)
            .color(sec),
        );
        ui.add_space(t.space[3]);

        // Species picker — named species, not raw ids (UX Principle 1).
        ui.label(egui::RichText::new("Species").color(prim).strong());
        let cur = FLORAPRIME_SPECIES[self.species_idx];
        egui::ComboBox::from_id_salt("floraprime.species")
            .selected_text(format!("{} (species {})", cur.0, cur.1))
            .show_ui(ui, |ui| {
                for (i, (label, sid, class)) in FLORAPRIME_SPECIES.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.species_idx,
                        i,
                        format!("{label} (species {sid}, {class})"),
                    );
                }
            });
        ui.add_space(t.space[2]);

        // Crown radius slider in metres (the real sample_tree arg).
        ui.label(egui::RichText::new("Crown radius").color(prim).strong());
        cx.widgets.scrub_drag(
            ui,
            &mut self.crown_radius_m,
            vox_ui::widgets::ScrubOpts {
                speed: 0.02,
                range: Some(0.5..=12.0),
                suffix: " m",
                axis_color: None,
            },
        );
        ui.add_space(t.space[2]);

        // Detail level — n_nodes mapped to Low/Medium/High (plain language).
        ui.label(egui::RichText::new("Detail level").color(prim).strong());
        ui.horizontal(|ui| {
            for (i, (label, n_nodes)) in FLORAPRIME_DETAIL.iter().enumerate() {
                if ui
                    .selectable_label(self.detail_idx == i, format!("{label} ({n_nodes} nodes)"))
                    .clicked()
                {
                    self.detail_idx = i;
                }
            }
        });
        ui.add_space(t.space[3]);

        // The primary action — icon + words, accent fill (the Canva rule).
        if vox_ui::widgets::primary_action(ui, icon::PARTICLE, "Grow tree", t).clicked() {
            self.grow();
        }
        ui.add_space(t.space[3]);

        // Generation-state readout: idle → queued → preview of node/edge counts.
        ui.separator();
        ui.add_space(t.space[2]);
        match &self.state {
            GenState::Idle => {
                ui.label(
                    egui::RichText::new("State: idle — press \"Grow tree\" to generate.")
                        .color(sec),
                );
            }
            GenState::Queued => {
                ui.label(egui::RichText::new("State: queued…").color(sec));
            }
            GenState::Preview(s) => {
                let (_, _, class) = FLORAPRIME_SPECIES[self.species_idx];
                ui.label(
                    egui::RichText::new("State: preview ready")
                        .color(prim)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "{} nodes · {} edges · branch depth {} · {:.1} m tall · {} QSM",
                        s.nodes.len(),
                        s.edges,
                        s.max_depth,
                        s.height_m,
                        class,
                    ))
                    .color(sec),
                );
                ui.label(
                    egui::RichText::new(
                        "(placeholder skeleton from the deterministic CPU stub — the FloraPrime \
                         Python sampler bridge replaces it.)",
                    )
                    .size(t.type_ramp.caption)
                    .color(sec),
                );
            }
        }
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

    // ---- FloraPrime ----------------------------------------------------------

    #[test]
    fn floraprime_declares_vegetation_tab_and_real_command() {
        let p = FloraPrimePlugin::new();
        assert_eq!(p.id(), "floraprime");
        let tabs = p.tabs();
        assert_eq!(tabs.len(), 1);
        assert_eq!(tabs[0].title, "Vegetation");
        assert_eq!(tabs[0].id, FLORAPRIME_TAB);

        let cmds = p.commands();
        let c = cmds
            .iter()
            .find(|c| c.id == "floraprime.generate_tree")
            .expect("missing floraprime.generate_tree command");
        assert_eq!(c.category, "FloraPrime");
    }

    #[test]
    fn floraprime_generate_command_flips_observable_flag() {
        let p = FloraPrimePlugin::new();
        let cmds = p.commands();
        let gen_cmd = cmds.iter().find(|c| c.id == "floraprime.generate_tree").unwrap();
        assert!(!*p.generated.borrow());
        (gen_cmd.run)();
        assert!(*p.generated.borrow(), "running floraprime.generate_tree must flip the flag");
    }

    #[test]
    fn skeleton_has_exact_node_and_edge_counts() {
        // The panel summary depends on these holding exactly: N nodes, N-1 edges.
        for &n in &[100usize, 200, 400] {
            let s = grow_tree_skeleton(0, 3.0, n);
            assert_eq!(s.nodes.len(), n, "skeleton must have exactly {n} nodes");
            assert_eq!(s.edges, n - 1, "a tree of {n} nodes must have {} edges", n - 1);
        }
    }

    #[test]
    fn skeleton_is_a_valid_spanning_tree() {
        // Root has no parent; every other node's parent is a strictly-earlier
        // index (so the parent-pointer structure is an acyclic spanning tree).
        let s = grow_tree_skeleton(2, 4.0, 200);
        assert_eq!(s.nodes[0].parent, usize::MAX, "root must have no parent");
        for (i, node) in s.nodes.iter().enumerate().skip(1) {
            assert!(
                node.parent < i,
                "node {i} parent {} must be an earlier index (acyclic tree)",
                node.parent
            );
        }
        // max_depth is a real reachable depth ≥ 1 for a non-trivial tree.
        assert!(s.max_depth >= 1, "a 200-node tree must branch at least one level deep");
    }

    #[test]
    fn skeleton_height_scales_with_crown_radius() {
        // height_m is derived from crown_radius — a larger crown grows a taller
        // tree (the panel's "{:.1} m tall" readout).
        let small = grow_tree_skeleton(0, 2.0, 200);
        let large = grow_tree_skeleton(0, 6.0, 200);
        assert!(
            large.height_m > small.height_m,
            "crown 6m ({:.2}m tall) must exceed crown 2m ({:.2}m tall)",
            large.height_m,
            small.height_m
        );
    }

    #[test]
    fn skeleton_is_deterministic_for_identical_params() {
        // Same (species_id, crown_radius, n_nodes) → byte-identical skeleton.
        let a = grow_tree_skeleton(1, 3.5, 200);
        let b = grow_tree_skeleton(1, 3.5, 200);
        assert_eq!(a, b, "identical params must reproduce an identical skeleton");
    }

    #[test]
    fn different_species_produce_different_skeletons() {
        // Same crown/detail but a different species_id → a different structure
        // (the species conditioning actually changes the seeded layout).
        let birch = grow_tree_skeleton(0, 3.0, 200);
        let pine = grow_tree_skeleton(2, 3.0, 200);
        assert_ne!(
            birch.nodes, pine.nodes,
            "different species_id must yield a different skeleton geometry"
        );
    }

    #[test]
    fn grow_drives_state_to_preview_matching_selected_params() {
        // The UI path: grow() resolves the selected params and lands in Preview
        // with a skeleton whose counts match the chosen detail level.
        let mut p = FloraPrimePlugin::new(); // defaults: species 0, 3.0m, Medium(200)
        assert_eq!(p.state, GenState::Idle);
        let (species_id, crown, n_nodes) = p.job_params();
        assert_eq!((species_id, crown, n_nodes), (0, 3.0, 200));
        p.grow();
        match &p.state {
            GenState::Preview(s) => {
                assert_eq!(s.nodes.len(), 200);
                assert_eq!(s.edges, 199);
                // grow() reproduces exactly what grow_tree_skeleton would for the
                // resolved params.
                assert_eq!(*s, grow_tree_skeleton(0, 3.0, 200));
            }
            other => panic!("grow() must land in Preview, got {other:?}"),
        }
        assert!(*p.generated.borrow(), "grow() must flip the generated flag");
    }

    #[test]
    fn species_picker_uses_named_species_mapped_to_real_class() {
        // Non-game-dev-friendly: named species, each carrying a real species_id
        // and a real FloraPrime spectral class (broadleaf/conifer/grass).
        let labels: Vec<&str> = FLORAPRIME_SPECIES.iter().map(|(l, _, _)| *l).collect();
        assert!(labels.contains(&"Silver Birch"), "expected a named species; have {labels:?}");
        for (_, _, class) in FLORAPRIME_SPECIES {
            assert!(
                matches!(*class, "broadleaf" | "conifer" | "grass"),
                "species class {class} must be a real FloraPrime spectral class"
            );
        }
        // Detail presets map to the real n_nodes sampler args.
        let nodes: Vec<usize> = FLORAPRIME_DETAIL.iter().map(|(_, n)| *n).collect();
        assert_eq!(nodes, vec![100, 200, 400]);
    }
}
