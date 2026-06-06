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
use crate::nodes::building_node::BuildingNode;
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
enum MatchTier {
    Prefix = 0,
    Substring = 1,
    Subsequence = 2,
}

/// A search hit: the matched kind plus its rank components, exposed for tests.
pub struct SearchHit<'a> {
    pub kind: &'a NodeKind,
    tier: MatchTier,
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
            kind_of("Building", || Box::new(BuildingNode::default())),
            kind_of("Building", || Box::new(PlotNode::default())),
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
    pub fn register_subgraph(&mut self, def: crate::subgraph::SubgraphDef) -> &'static str {
        let name: &'static str = Box::leak(def.name.clone().into_boxed_str());
        // The constructor owns the def and hands out deep clones, so every created
        // instance is independent and the registry copy is never mutated.
        let ctor: NodeConstructor = Box::new(move || {
            Box::new(crate::subgraph::SubgraphNode::new(def.deep_clone()))
        });
        self.kinds.push(kind_of_boxed("Subgraph", ctor));
        // Patch the freshly-pushed kind's name to the stable leaked name (kind_of_boxed
        // copied it from the descriptor's already-leaked type_name, which is identical
        // content; we keep `name` as the canonical leaked handle).
        let last = self.kinds.last_mut().expect("just pushed");
        last.name = name;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::OchromaNodeGraph;

    #[test]
    fn registry_enumerates_all_shipped_nodes() {
        let reg = NodeRegistry::new();
        // Every node module is represented.
        assert!(!reg.is_empty());
        for expected in [
            "TerrainNode", "BiomeNode", "MoistureNode", "VegetationNode",
            "BuildingNode", "PlotNode", "SplatizeNode", "SplatWeightNode",
            "UrbanSimNode", "CatenaryNode", "PropPlacementNode",
        ] {
            assert!(reg.get(expected).is_some(), "registry missing {expected}");
        }
        assert_eq!(reg.len(), 11);
    }

    #[test]
    fn registry_ports_match_real_descriptor() {
        let reg = NodeRegistry::new();
        // BiomeNode: input "terrain" : Terrain ; output "biome_map" : BiomeMap.
        let biome = reg.get("BiomeNode").unwrap();
        assert_eq!(biome.inputs.len(), 1);
        assert_eq!(biome.inputs[0].name, "terrain");
        assert_eq!(biome.inputs[0].port_type, PortType::Terrain);
        assert_eq!(biome.outputs[0].name, "biome_map");
        assert_eq!(biome.outputs[0].port_type, PortType::BiomeMap);
    }

    #[test]
    fn search_prefix_ranks_above_substring() {
        let reg = NodeRegistry::new();
        // "bio" is a prefix of BiomeNode -> it must be the first hit.
        let hits = reg.search("bio");
        assert!(!hits.is_empty(), "search('bio') found nothing");
        assert_eq!(hits[0].name(), "BiomeNode", "BiomeNode should rank first for 'bio'");
        assert_eq!(hits[0].tier, MatchTier::Prefix);
    }

    #[test]
    fn search_ter_finds_terrain_first() {
        let reg = NodeRegistry::new();
        let hits = reg.search("ter");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].name(), "TerrainNode", "TerrainNode should rank first for 'ter'");
    }

    #[test]
    fn search_substring_beats_subsequence() {
        let reg = NodeRegistry::new();
        // "lat" appears as a contiguous substring in "SplatizeNode" / "SplatWeightNode"
        // (Sp-lat-...) -> Substring tier. It is also a (gapped) subsequence of e.g.
        // "PropPlacementNode" (p..l..a..t? no) — at minimum the substring hits must
        // out-rank any pure subsequence hit.
        let hits = reg.search("lat");
        assert!(!hits.is_empty());
        let first_tier = hits[0].tier;
        assert_eq!(first_tier, MatchTier::Substring, "contiguous 'lat' should be a substring match");
        // Tiers are non-decreasing through the result list.
        for w in hits.windows(2) {
            assert!(w[0].tier <= w[1].tier, "search results must be ordered by tier");
        }
    }

    #[test]
    fn search_is_deterministic_and_stable() {
        let reg = NodeRegistry::new();
        let a: Vec<String> = reg.search("node").iter().map(|h| h.name().to_string()).collect();
        let b: Vec<String> = reg.search("node").iter().map(|h| h.name().to_string()).collect();
        assert_eq!(a, b, "identical queries must return identical ordering");
    }

    #[test]
    fn empty_query_returns_all_alphabetically() {
        let reg = NodeRegistry::new();
        let hits = reg.search("");
        assert_eq!(hits.len(), reg.len());
        let names: Vec<&str> = hits.iter().map(|h| h.name()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "empty query must be alphabetical");
    }

    #[test]
    fn compatible_with_terrain_includes_biome_excludes_mismatched() {
        let reg = NodeRegistry::new();
        // Dragging a Terrain output: only nodes with a Terrain input are valid.
        let compat = reg.compatible_with(PortType::Terrain);
        let names: Vec<&str> = compat.iter().map(|k| k.name).collect();
        assert!(names.contains(&"BiomeNode"), "BiomeNode accepts Terrain input");
        // TerrainNode has NO inputs -> must be excluded.
        assert!(!names.contains(&"TerrainNode"), "TerrainNode has no Terrain input, must be filtered out");
        // SplatizeNode only accepts Mesh -> excluded for a Terrain wire.
        assert!(!names.contains(&"SplatizeNode"), "SplatizeNode accepts Mesh, not Terrain");
    }

    #[test]
    fn compatible_with_mesh_targets_splatize() {
        let reg = NodeRegistry::new();
        let names: Vec<&str> = reg.compatible_with(PortType::Mesh).iter().map(|k| k.name).collect();
        assert!(names.contains(&"SplatizeNode"), "SplatizeNode accepts a Mesh input");
        assert!(!names.contains(&"BiomeNode"), "BiomeNode does not accept Mesh");
    }

    #[test]
    fn producing_biomemap_finds_biome() {
        let reg = NodeRegistry::new();
        let names: Vec<&str> = reg.producing(PortType::BiomeMap).iter().map(|k| k.name).collect();
        assert!(names.contains(&"BiomeNode"), "BiomeNode outputs a BiomeMap");
    }

    #[test]
    fn created_node_evaluates_in_a_graph() {
        let reg = NodeRegistry::new();
        let node = reg.create("TerrainNode").expect("TerrainNode is registered");
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("terrain", node);
        // A registry-created node must produce a real output through evaluate().
        let result = graph.evaluate().unwrap();
        let terrain = result.get(id, "terrain").unwrap().as_terrain().unwrap();
        // Default resolution 256 -> a genuinely computed heightfield.
        assert_eq!(terrain.heights.len(), 256 * 256);
        assert_eq!(terrain.resolution, 256);
    }

    #[test]
    fn created_terrain_into_created_biome_connects_and_flows() {
        let reg = NodeRegistry::new();
        let mut graph = OchromaNodeGraph::new();
        let t = graph.add_node("t", reg.create("TerrainNode").unwrap());
        let b = graph.add_node("b", reg.create("BiomeNode").unwrap());
        // The registry's typed ports are the real ones, so this connect type-checks.
        graph.connect(t, "terrain", b, "terrain").unwrap();
        let result = graph.evaluate().unwrap();
        let biome = result.get(b, "biome_map").unwrap().as_biome_map().unwrap();
        let terrain = result.get(t, "terrain").unwrap().as_terrain().unwrap();
        assert_eq!(biome.len(), terrain.heights.len(), "one biome byte per terrain cell");
    }

    #[test]
    fn unknown_name_creates_nothing() {
        let reg = NodeRegistry::new();
        assert!(reg.create("NoSuchNode").is_none());
        assert!(reg.search("zzzqqq").is_empty());
    }
}
