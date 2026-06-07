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
use glam::Quat;
use std::cell::RefCell;
use std::rc::Rc;
use vox_core::types::GaussianSplat;
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

/// A tree the user grew, ready to plant into the live world: its display-species
/// label (e.g. "Silver Birch", which the shell numbers per-grow) and the REAL
/// `GaussianSplat`s built from the skeleton. The shell drains these from
/// [`FloraPrimePlugin`]'s grow-sink and routes them into the viewport scene
/// overlay + World panel + undo stack.
///
/// (`GaussianSplat` is `Pod` but not `PartialEq`/`Debug`, so this carries only
/// `Clone`; tests compare splats bytewise via `bytemuck`.)
#[derive(Clone)]
pub struct GrownTree {
    /// The friendly species label (the shell appends an incrementing number).
    pub species_label: String,
    /// The deterministic splats built from the grown skeleton.
    pub splats: Vec<GaussianSplat>,
}

/// Where a grown tree plants on the demo terrain. v1 uses a FIXED clear spot to
/// the left of the demo structures (`build_scene` places its amber "buildings"
/// around x∈[-3,2.5], z∈[-4,-9]); this sits the tree to the LEFT of them at the
/// scene's focal depth (z=-6), where its crown rises against the dark studio
/// background above the green ground band, reading as a clear tree silhouette. A
/// later slice adds "＋ Add to world" click-to-place. The Y offset drops the trunk
/// base onto the ground band (`build_scene`'s ground sits at y=-1).
pub const TREE_PLANT_ORIGIN: [f32; 3] = [-4.0, -1.0, -6.0];

/// Convert a grown [`TreeSkeleton`] into REAL `GaussianSplat`s — one volume splat
/// per skeleton node. Trunk/branch nodes (shallow depth) get a brown-bark
/// spectrum; crown-tip nodes (the deepest third of the branch depth) get a leaf
/// spectrum whose green band rises for broadleaf and darkens for conifer. Splat
/// scale follows the node's branch radius (derived from its distance off the
/// trunk axis), and every node is translated by [`TREE_PLANT_ORIGIN`]. Fully
/// deterministic: same skeleton + class → bit-identical splats.
///
/// Spectra are 16-band f16 reflectance, derived the way `viewport::build_scene`
/// derives its colored splats (a band window high, the rest low), so the tree
/// reads in the SAME spectral pipeline as the rest of the scene.
pub fn skeleton_to_splats(skeleton: &TreeSkeleton, class: &str, species_id: i32) -> Vec<GaussianSplat> {
    // Band window helper: value `hi` inside `window`, `lo` outside → a colored
    // 16-band reflectance (identical construction to build_scene's `spd`).
    let spd = |window: std::ops::RangeInclusive<usize>, hi: f32, lo: f32| -> [u16; 16] {
        std::array::from_fn(|i| {
            let v = if window.contains(&i) { hi } else { lo };
            half::f16::from_f32(v).to_bits()
        })
    };

    // Brown bark: warm long-wavelength bias, muted (low overall reflectance).
    let bark = spd(10..=13, 0.42, 0.12);
    // Leaf spectra by class. Green = mid bands high. The window is NARROWER and
    // the off-band floor LOWER than `build_scene`'s broad, muted ground green
    // (spd(5..=9, 0.85, 0.18)), so the canopy reads as a saturated emerald that
    // separates from the ground rather than blending into it. Broadleaf is a
    // brighter, greener canopy; conifer is darker, slightly blue-green; grass
    // (the meadow species) is the most vivid green.
    let leaf = match class {
        "conifer" => spd(6..=8, 0.6, 0.04),
        "grass" => spd(6..=8, 0.98, 0.05),
        _ => spd(6..=8, 0.92, 0.04), // broadleaf (default)
    };

    // The crown is the deepest third of branch depth — those tips get leaves.
    let max_depth = skeleton.max_depth.max(1);
    let crown_from = (max_depth * 2) / 3; // depth >= this → crown/leaf

    // Re-derive per-node depth from the parent pointers (root depth 0). The
    // skeleton stores parents in strictly-increasing index order (a spanning
    // tree), so a single forward pass resolves every depth.
    let mut depth = vec![0usize; skeleton.nodes.len()];
    for (i, node) in skeleton.nodes.iter().enumerate() {
        if node.parent != usize::MAX {
            depth[i] = depth[node.parent] + 1;
        }
    }

    let mut splats = Vec::with_capacity(skeleton.nodes.len());
    for (i, node) in skeleton.nodes.iter().enumerate() {
        let d = depth[i];
        let is_crown = d >= crown_from && node.parent != usize::MAX;
        // Branch radius proxy: distance from the trunk axis (x,z) sets how thick
        // the splat is. Trunk/branch splats are slim cylinders-ish; crown tips
        // are puffy leaf clusters. species_id nudges the leaf puffiness so two
        // species of the same class still differ slightly in canopy density.
        let off = (node.pos[0] * node.pos[0] + node.pos[2] * node.pos[2]).sqrt();
        let (scale, opacity, spectral) = if is_crown {
            // Crown tips are puffy leaf clusters — large enough to merge into a
            // readable canopy silhouette in the splat rasterizer. species_id nudges
            // density so two species of the same class still differ.
            let puff = 0.34 + 0.05 * ((species_id.unsigned_abs() % 4) as f32) + 0.04 * off;
            ([puff, puff, puff], 245u8, leaf)
        } else {
            // Trunk/branch: a solid bark column. Thickness tapers slightly toward
            // the tips (off-axis distance) but keeps a generous floor so the trunk
            // reads as a continuous stem, not scattered dots.
            let thick = (0.30 - 0.02 * off).max(0.14);
            ([thick, thick * 1.5, thick], 255u8, bark)
        };
        let pos = [
            node.pos[0] + TREE_PLANT_ORIGIN[0],
            node.pos[1] + TREE_PLANT_ORIGIN[1],
            node.pos[2] + TREE_PLANT_ORIGIN[2],
        ];
        splats.push(GaussianSplat::volume(pos, scale, Quat::IDENTITY, opacity, spectral));
    }
    splats
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
    /// Grown trees waiting for the host to plant them in the live world. `grow()`
    /// pushes one `GrownTree` (species label + real splats) here; the shell drains
    /// it each frame into the viewport overlay + World panel + undo stack. Shared
    /// (cloned) so the host can hold the SAME queue this plugin writes to.
    pub grow_sink: Rc<RefCell<Vec<GrownTree>>>,
}

impl Default for FloraPrimePlugin {
    fn default() -> Self {
        FloraPrimePlugin {
            species_idx: 0,
            crown_radius_m: 3.0, // the real sample_tree default
            detail_idx: 1,       // Medium / n_nodes=200 (the real default)
            state: GenState::Idle,
            generated: Rc::new(RefCell::new(false)),
            grow_sink: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl FloraPrimePlugin {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a FloraPrime plugin that writes grown trees into the SHARED `sink`,
    /// so the host can drain the SAME queue this plugin's "Grow tree" button fills.
    /// The host clones its own handle to `sink` before constructing this.
    pub fn with_grow_sink(sink: Rc<RefCell<Vec<GrownTree>>>) -> Self {
        FloraPrimePlugin {
            grow_sink: sink,
            ..Self::default()
        }
    }

    /// The currently-selected job params resolved to the real `sample_tree`
    /// arguments `(species_id, crown_radius_m, n_nodes)`.
    pub fn job_params(&self) -> (i32, f32, usize) {
        let (_, species_id, _) = FLORAPRIME_SPECIES[self.species_idx];
        let (_, n_nodes) = FLORAPRIME_DETAIL[self.detail_idx];
        (species_id, self.crown_radius_m, n_nodes)
    }

    /// Run the (stub) sampler for the current params, move the readout to
    /// `Preview`, AND emit a [`GrownTree`] (species label + real splats) onto the
    /// grow-sink for the host to plant into the live world. The same path the
    /// `floraprime.generate_tree` command drives.
    pub fn grow(&mut self) {
        let (species_id, crown_radius_m, n_nodes) = self.job_params();
        let (label, _, class) = FLORAPRIME_SPECIES[self.species_idx];
        self.state = GenState::Queued;
        let skeleton = grow_tree_skeleton(species_id, crown_radius_m, n_nodes);
        // Build the REAL splats from the skeleton and hand them to the host to
        // plant — this is the close-the-loop step: a tree the user can SEE.
        let splats = skeleton_to_splats(&skeleton, class, species_id);
        self.grow_sink.borrow_mut().push(GrownTree {
            species_label: label.to_string(),
            splats,
        });
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
                "Pick a species, set how wide the crown should be, choose how detailed, then grow it.",
            )
            .size(t.type_ramp.caption)
            .color(sec),
        );
        ui.add_space(t.space[3]);

        // Species picker — named species, not raw ids (UX Principle 1). The
        // selected text reads as a forester would name it (species + leaf type);
        // the internal id is kept on each dropdown row so nothing is lost.
        ui.label(egui::RichText::new("Species").color(prim).strong());
        let cur = FLORAPRIME_SPECIES[self.species_idx];
        egui::ComboBox::from_id_salt("floraprime.species")
            .selected_text(format!("{} ({})", cur.0, cur.2))
            .show_ui(ui, |ui| {
                for (i, (label, sid, class)) in FLORAPRIME_SPECIES.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.species_idx,
                        i,
                        format!("{label} ({class}, species {sid})"),
                    );
                }
            });
        ui.add_space(t.space[2]);

        // Crown radius slider in metres (the real sample_tree arg).
        ui.label(egui::RichText::new("Crown radius").color(prim).strong());
        cx.widgets
            .scrub_drag(
                ui,
                &mut self.crown_radius_m,
                vox_ui::widgets::ScrubOpts {
                    speed: 0.02,
                    range: Some(0.5..=12.0),
                    suffix: " m",
                    axis_color: None,
                },
            )
            .on_hover_text(
                "How far the leafy crown spreads, in metres. A wider crown also grows a taller tree.",
            );
        ui.add_space(t.space[2]);

        // Detail level — n_nodes mapped to Low/Medium/High (plain language). The
        // raw branch-point count rides along in parentheses (precision retained).
        ui.label(egui::RichText::new("Detail level").color(prim).strong());
        ui.horizontal(|ui| {
            for (i, (label, n_nodes)) in FLORAPRIME_DETAIL.iter().enumerate() {
                if ui
                    .selectable_label(self.detail_idx == i, format!("{label} ({n_nodes} branch points)"))
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
                    egui::RichText::new("Nothing grown yet — press \"Grow tree\" to see one.")
                        .color(sec),
                );
            }
            GenState::Queued => {
                ui.label(egui::RichText::new("Growing…").color(sec));
            }
            GenState::Preview(s) => {
                let (_, _, class) = FLORAPRIME_SPECIES[self.species_idx];
                ui.label(
                    egui::RichText::new("Here's your tree")
                        .color(prim)
                        .strong(),
                );
                // Domain-readable summary: height and how bushy it branches, in
                // plain words. Exact structural counts ride in parentheses so a
                // technical user still sees them (precision retained, not deleted).
                ui.label(
                    egui::RichText::new(format!(
                        "{:.1} m tall · branches {} levels deep · {} leaf type \
                         ({} branch points, {} connections)",
                        s.height_m,
                        s.max_depth,
                        class,
                        s.nodes.len(),
                        s.edges,
                    ))
                    .color(sec),
                );
                ui.label(
                    egui::RichText::new(
                        "(Preview built locally on the CPU — the full FloraPrime grower replaces it later.)",
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

    // ---- skeleton → splats ---------------------------------------------------

    /// Field-by-field splat equality (GaussianSplat is Pod but not PartialEq):
    /// position, scales, opacity and every spectral band must match.
    fn splats_eq(a: &GaussianSplat, b: &GaussianSplat) -> bool {
        a.position() == b.position()
            && a.scales() == b.scales()
            && a.opacity() == b.opacity()
            && a.spectral() == b.spectral()
    }

    #[test]
    fn skeleton_to_splats_has_one_splat_per_node() {
        // Every skeleton node becomes exactly one splat — a Medium tree (200 nodes)
        // yields 200 splats, which reads as a tree silhouette in the viewport.
        let s = grow_tree_skeleton(0, 3.0, 200);
        let splats = skeleton_to_splats(&s, "broadleaf", 0);
        assert_eq!(splats.len(), 200, "one splat per skeleton node");
        assert_eq!(splats.len(), s.nodes.len());
    }

    #[test]
    fn crown_splats_are_greener_than_trunk_splats() {
        // The leaf (crown) spectra must read GREENER than the brown-bark trunk
        // spectra: the crown's mid (green) band exceeds the trunk's, AND the trunk's
        // long-wavelength (red/bark) band exceeds the crown's. Derive depth to split
        // trunk vs crown the same way the converter does.
        let s = grow_tree_skeleton(0, 3.0, 200);
        let splats = skeleton_to_splats(&s, "broadleaf", 0);
        let mut depth = vec![0usize; s.nodes.len()];
        for (i, n) in s.nodes.iter().enumerate() {
            if n.parent != usize::MAX {
                depth[i] = depth[n.parent] + 1;
            }
        }
        let crown_from = (s.max_depth.max(1) * 2) / 3;
        // Average the green band (7) and a bark band (12) over trunk vs crown.
        let (mut trunk_green, mut trunk_bark, mut nt) = (0f32, 0f32, 0f32);
        let (mut crown_green, mut crown_bark, mut nc) = (0f32, 0f32, 0f32);
        for (i, sp) in splats.iter().enumerate() {
            let g = sp.spectral_f32(7);
            let bark = sp.spectral_f32(12);
            if depth[i] >= crown_from && s.nodes[i].parent != usize::MAX {
                crown_green += g;
                crown_bark += bark;
                nc += 1.0;
            } else {
                trunk_green += g;
                trunk_bark += bark;
                nt += 1.0;
            }
        }
        assert!(nt > 0.0 && nc > 0.0, "tree must have both trunk and crown splats");
        let (cg, tg) = (crown_green / nc, trunk_green / nt);
        let (cb, tb) = (crown_bark / nc, trunk_bark / nt);
        assert!(cg > tg, "crown green band ({cg:.3}) must exceed trunk green band ({tg:.3})");
        assert!(tb > cb, "trunk bark band ({tb:.3}) must exceed crown bark band ({cb:.3})");
    }

    #[test]
    fn broadleaf_crown_greener_than_conifer_crown() {
        // Class drives the leaf spectrum: a broadleaf canopy's green band is
        // brighter than a conifer's darker needles (same skeleton).
        let s = grow_tree_skeleton(0, 3.0, 200);
        let broad = skeleton_to_splats(&s, "broadleaf", 0);
        let conifer = skeleton_to_splats(&s, "conifer", 2);
        // Compare the max green-band reflectance among each tree's splats (the crown
        // tips). Broadleaf's leaf green (0.8) must exceed conifer's (0.5).
        let max_green = |v: &[GaussianSplat]| {
            v.iter().map(|sp| sp.spectral_f32(7)).fold(0.0f32, f32::max)
        };
        let bg = max_green(&broad);
        let cg = max_green(&conifer);
        assert!(bg > cg, "broadleaf crown green ({bg:.3}) must exceed conifer crown green ({cg:.3})");
    }

    #[test]
    fn splat_scale_follows_branch_radius() {
        // A wider crown radius → larger off-axis spread → larger crown splat scales.
        // The widest splat of a 6m-crown tree must exceed the widest of a 2m-crown.
        let small = skeleton_to_splats(&grow_tree_skeleton(0, 2.0, 200), "broadleaf", 0);
        let large = skeleton_to_splats(&grow_tree_skeleton(0, 6.0, 200), "broadleaf", 0);
        let max_scale = |v: &[GaussianSplat]| {
            v.iter().map(|sp| sp.scales()[0]).fold(0.0f32, f32::max)
        };
        let ms = max_scale(&small);
        let ml = max_scale(&large);
        assert!(ml > ms, "6m-crown max splat scale ({ml:.3}) must exceed 2m-crown ({ms:.3})");
    }

    #[test]
    fn skeleton_to_splats_is_deterministic() {
        // Same skeleton + class + species → bit-identical splats (no rand).
        let s = grow_tree_skeleton(1, 3.5, 200);
        let a = skeleton_to_splats(&s, "broadleaf", 1);
        let b = skeleton_to_splats(&s, "broadleaf", 1);
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b.iter()).all(|(x, y)| splats_eq(x, y)),
            "identical inputs must produce bit-identical splats"
        );
    }

    #[test]
    fn grow_emits_a_grown_tree_onto_the_sink() {
        // The UI grow() path must push a GrownTree (species label + real splats)
        // onto the shared grow-sink so the host can plant it.
        let mut p = FloraPrimePlugin::new(); // defaults: Silver Birch, 3.0m, Medium(200)
        assert!(p.grow_sink.borrow().is_empty());
        p.grow();
        let sink = p.grow_sink.borrow();
        assert_eq!(sink.len(), 1, "grow() must emit exactly one GrownTree");
        assert_eq!(sink[0].species_label, "Silver Birch");
        assert_eq!(sink[0].splats.len(), 200, "the grown tree carries one splat per node");
    }
}
