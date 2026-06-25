//! NodeRegistry — typed enumeration of every node type the editor can author,
//! powering search-driven insertion and UE-style context-filtered wire dragging.
//!
//! Each entry carries:
//!   - a stable `name` (the node's `descriptor().type_name`),
//!   - a `category` for menu grouping,
//!   - the typed input/output ports (read straight from the node's real
//!     `descriptor()`, so the registry can never drift from the actual node),
//!   - a `constructor` returning a fresh boxed node that `evaluate()` accepts.
//!
//! Search ranking is deterministic: prefix > substring > fuzzy-subsequence, and
//! within a tier ties break on shorter name then alphabetical, so results are
//! stable across runs.

use crate::node_graph::{OchromaNode, PortType};

use crate::nodes::biome_node::BiomeNode;
use crate::nodes::inhabitation_node::{CatenaryNode, PropPlacementNode};
use crate::nodes::moisture_node::MoistureNode;
use crate::nodes::plot_node::PlotNode;
use crate::nodes::splat_weight_node::SplatWeightNode;
use crate::nodes::splatize_node::SplatizeNode;
use crate::nodes::terrain_node::TerrainNode;
use crate::nodes::urban_sim_node::UrbanSimNode;
use crate::nodes::vegetation_node::VegetationNode;

/// A single typed port (name + type) on a registered node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryPort {
    pub name: String,
    pub port_type: PortType,
}

/// Boxed constructor for a node kind. A closure (rather than a bare `fn` pointer)
/// so dynamically-registered kinds — notably subgraphs — can capture their def.
pub type NodeConstructor = Box<dyn Fn() -> Box<dyn OchromaNode> + Send + Sync>;

/// One entry in the [`NodeRegistry`]: everything the UI needs to list, filter,
/// and instantiate a node type.
pub struct NodeKind {
    pub name: &'static str,
    pub category: &'static str,
    pub inputs: Vec<RegistryPort>,
    pub outputs: Vec<RegistryPort>,
    constructor: NodeConstructor,
}

impl NodeKind {
    /// Instantiate a fresh, working node of this kind.
    pub fn create(&self) -> Box<dyn OchromaNode> {
        (self.constructor)()
    }

    /// Does this node accept the given port type on any of its inputs? Used by
    /// the "drag a wire out, see only nodes that accept this type" context filter.
    pub fn accepts_input(&self, port_type: PortType) -> bool {
        self.inputs.iter().any(|p| p.port_type == port_type)
    }

    /// Does this node produce the given port type on any of its outputs?
    pub fn produces_output(&self, port_type: PortType) -> bool {
        self.outputs.iter().any(|p| p.port_type == port_type)
    }
}

/// Tier of a search match — lower is better (ranked first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchTier {
    Prefix = 0,
    Substring = 1,
    Subsequence = 2,
}

/// A search hit: the matched kind plus its rank components, exposed for tests.
pub struct SearchHit<'a> {
    pub kind: &'a NodeKind,
    pub tier: MatchTier,
}

impl SearchHit<'_> {
    pub fn name(&self) -> &str {
        self.kind.name
    }
}

/// The registry of all authorable node types.
pub struct NodeRegistry {
    kinds: Vec<NodeKind>,
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`NodeKind`] from a node's real descriptor so registry metadata can
/// never drift from the node's actual ports.
fn kind_of(category: &'static str, constructor: fn() -> Box<dyn OchromaNode>) -> NodeKind {
    let probe = constructor();
    let desc = probe.descriptor();
    let type_name = desc.type_name;
    let inputs = desc
        .inputs
        .iter()
        .map(|p| RegistryPort { name: p.name.to_string(), port_type: p.port_type })
        .collect();
    let outputs = desc
        .outputs
        .iter()
        .map(|p| RegistryPort { name: p.name.to_string(), port_type: p.port_type })
        .collect();
    NodeKind { name: type_name, category, inputs, outputs, constructor: Box::new(constructor) }
}

/// Build a [`NodeKind`] from any boxed closure constructor (used for subgraphs).
fn kind_of_boxed(category: &'static str, constructor: NodeConstructor) -> NodeKind {
    let probe = constructor();
    let desc = probe.descriptor();
    let inputs = desc
        .inputs
        .iter()
        .map(|p| RegistryPort { name: p.name.to_string(), port_type: p.port_type })
        .collect();
    let outputs = desc
        .outputs
        .iter()
        .map(|p| RegistryPort { name: p.name.to_string(), port_type: p.port_type })
        .collect();
    NodeKind { name: desc.type_name, category, inputs, outputs, constructor }
}

impl NodeRegistry {
    /// Build the registry with every node type the editor ships.
    pub fn new() -> Self {
        let kinds = vec![
            kind_of("Terrain", || Box::new(TerrainNode::default())),
            kind_of("Terrain", || Box::new(BiomeNode::default())),
            kind_of("Terrain", || Box::new(MoistureNode::default())),
            kind_of("Vegetation", || Box::new(VegetationNode::default())),
            kind_of("Content", || Box::new(PlotNode::default())),
            kind_of("Splatting", || Box::new(SplatizeNode::default())),
            kind_of("Splatting", || Box::new(SplatWeightNode)),
            kind_of("Urban", || Box::new(UrbanSimNode::default())),
            kind_of("Urban", || Box::new(CatenaryNode::default())),
            kind_of("Urban", || Box::new(PropPlacementNode::default())),
        ];
        Self { kinds }
    }

    /// Register a [`SubgraphDef`] as a creatable node kind. After this call the
    /// subgraph appears in [`search`](Self::search), [`get`](Self::get) and
    /// [`create`](Self::create) exactly like a built-in node: its `name` becomes the
    /// kind name and its interface becomes the kind's typed ports, and `create`
    /// yields a fresh [`crate::subgraph::SubgraphNode`] wrapping a deep clone of the
    /// def.
    ///
    /// Returns the `&'static str` kind name under which it was registered (the def's
    /// name, leaked to `'static`).
    ///
    /// **Same-name registration REPLACES the existing kind in place** (it does not
    /// append a shadowed duplicate). Re-registering an edited subgraph under a name
    /// that already exists swaps the constructor and ports to the new def, so
    /// [`create`](Self::create) builds the new version and [`search`](Self::search)
    /// returns exactly one hit. The old kind's leaked name is dropped on the floor
    /// (an unavoidable, bounded one-time leak); the canonical handle stays stable.
    pub fn register_subgraph(&mut self, def: crate::subgraph::SubgraphDef) -> &'static str {
        let name: &'static str = Box::leak(def.name.clone().into_boxed_str());
        // The constructor owns the def and hands out deep clones, so every created
        // instance is independent and the registry copy is never mutated.
        let ctor: NodeConstructor = Box::new(move || {
            Box::new(crate::subgraph::SubgraphNode::new(def.deep_clone()))
        });
        let mut kind = kind_of_boxed("Subgraph", ctor);
        // Keep `name` as the canonical leaked handle (kind_of_boxed copied it from the
        // descriptor's already-leaked type_name, identical content).
        kind.name = name;
        // Replace in place if a kind with this exact name already exists, so a
        // re-registration of an edited subgraph supersedes the stale version rather
        // than being shadowed behind it by `get`/`create`/`search`.
        if let Some(existing) = self.kinds.iter_mut().find(|k| k.name == name) {
            *existing = kind;
        } else {
            self.kinds.push(kind);
        }
        name
    }

    /// Number of registered node kinds.
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// All registered kinds.
    pub fn kinds(&self) -> &[NodeKind] {
        &self.kinds
    }

    /// Look up a kind by its exact `type_name`.
    pub fn get(&self, name: &str) -> Option<&NodeKind> {
        self.kinds.iter().find(|k| k.name == name)
    }

    /// Create a fresh node by exact `type_name`. Returns `None` for unknown names.
    /// The returned node is fully constructed and accepted by `graph.add_node` /
    /// `evaluate()`.
    pub fn create(&self, name: &str) -> Option<Box<dyn OchromaNode>> {
        self.get(name).map(|k| k.create())
    }

    /// Search node names for `query`, returning matches ranked best-first.
    ///
    /// Ranking, all case-insensitive:
    ///   1. Prefix match (name starts with query)
    ///   2. Substring match (query appears contiguously)
    ///   3. Fuzzy subsequence (query chars appear in order, possibly gapped)
    ///
    /// Ties within a tier break on shorter name, then alphabetical — so the order
    /// is fully deterministic. An empty query returns every kind alphabetically.
    pub fn search(&self, query: &str) -> Vec<SearchHit<'_>> {
        let q = query.to_ascii_lowercase();

        if q.is_empty() {
            let mut hits: Vec<SearchHit<'_>> = self
                .kinds
                .iter()
                .map(|k| SearchHit { kind: k, tier: MatchTier::Prefix })
                .collect();
            hits.sort_by(|a, b| a.kind.name.cmp(b.kind.name));
            return hits;
        }

        let mut hits: Vec<SearchHit<'_>> = self
            .kinds
            .iter()
            .filter_map(|k| {
                let name_lc = k.name.to_ascii_lowercase();
                let tier = if name_lc.starts_with(&q) {
                    MatchTier::Prefix
                } else if name_lc.contains(&q) {
                    MatchTier::Substring
                } else if is_subsequence(&q, &name_lc) {
                    MatchTier::Subsequence
                } else {
                    return None;
                };
                Some(SearchHit { kind: k, tier })
            })
            .collect();

        hits.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then_with(|| a.kind.name.len().cmp(&b.kind.name.len()))
                .then_with(|| a.kind.name.cmp(b.kind.name))
        });
        hits
    }

    /// UE-style context filter: every node that can accept `port_type` on one of
    /// its inputs (i.e. valid drop targets when dragging a wire OUT of an output
    /// of that type). Type-mismatched nodes are excluded. Result is ordered like
    /// [`search`] with an empty query (alphabetical) for stability.
    pub fn compatible_with(&self, port_type: PortType) -> Vec<&NodeKind> {
        let mut out: Vec<&NodeKind> = self
            .kinds
            .iter()
            .filter(|k| k.accepts_input(port_type))
            .collect();
        out.sort_by(|a, b| a.name.cmp(b.name));
        out
    }

    /// Counterpart filter: nodes that PRODUCE `port_type` on some output (valid
    /// sources when dragging a wire INTO an input of that type).
    pub fn producing(&self, port_type: PortType) -> Vec<&NodeKind> {
        let mut out: Vec<&NodeKind> = self
            .kinds
            .iter()
            .filter(|k| k.produces_output(port_type))
            .collect();
        out.sort_by(|a, b| a.name.cmp(b.name));
        out
    }
}

/// Is `needle` a subsequence of `haystack` (chars in order, gaps allowed)?
fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut hay = haystack.chars();
    for nc in needle.chars() {
        loop {
            match hay.next() {
                Some(hc) if hc == nc => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}
