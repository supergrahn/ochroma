//! Tests for [`crate::registry`].

#[cfg(test)]
mod tests {
    use crate::node_graph::{OchromaNode, OchromaNodeGraph, PortType};
    use crate::registry::{MatchTier, NodeRegistry};

    #[test]
    fn registry_enumerates_all_shipped_nodes() {
        let reg = NodeRegistry::new();
        // Every node module is represented.
        assert!(!reg.is_empty());
        for expected in [
            "TerrainNode", "BiomeNode", "MoistureNode", "VegetationNode",
            "SplatizeNode", "SplatWeightNode",
            "CatenaryNode", "PropPlacementNode",
        ] {
            assert!(reg.get(expected).is_some(), "registry missing {expected}");
        }
        assert_eq!(reg.len(), 8);
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

    /// Registering two distinct subgraph defs under the SAME name replaces the first:
    /// create() builds v2, and search() returns exactly one hit (no shadow).
    #[test]
    fn register_subgraph_same_name_replaces_and_searches_once() {
        use crate::node_graph::NodeInputs;
        use crate::nodes::terrain_node::TerrainNode;
        use crate::subgraph::{ExposedPort, SubgraphDef, SubgraphNode};

        // Build a def named "Same" exposing its inner terrain's "terrain" output,
        // parameterised by amplitude so v1 and v2 cook to different heights.
        fn terrain_def(name: &str, amplitude: f32) -> SubgraphDef {
            let mut inner = OchromaNodeGraph::new();
            let id = inner.add_node(
                "terrain",
                Box::new(TerrainNode {
                    resolution: 16,
                    amplitude,
                    droplet_count: 0,
                    seed: 5,
                    ..Default::default()
                }),
            );
            SubgraphDef {
                name: name.to_string(),
                inner,
                inputs: vec![],
                outputs: vec![ExposedPort {
                    outer_name: "out".into(),
                    inner_node: id,
                    inner_port: "terrain".into(),
                    port_type: PortType::Terrain,
                }],
            }
        }

        // Reference height for v2 (amplitude 999) by cooking the def directly.
        let v2_ref = {
            let node = SubgraphNode::new(terrain_def("Same", 999.0));
            let out = node.cook(NodeInputs::new()).unwrap();
            out["out"].as_terrain().unwrap().heights[100]
        };

        let mut reg = NodeRegistry::new();
        let base_len = reg.len();
        reg.register_subgraph(terrain_def("Same", 100.0)); // v1
        assert_eq!(reg.len(), base_len + 1, "first registration adds one kind");
        reg.register_subgraph(terrain_def("Same", 999.0)); // v2, same name
        assert_eq!(reg.len(), base_len + 1, "same-name re-registration must NOT add a second kind");

        // search returns exactly one "Same".
        let search = reg.search("Same");
        let hits: Vec<&str> = search.iter().map(|h| h.name()).collect();
        assert_eq!(
            hits.iter().filter(|n| **n == "Same").count(),
            1,
            "exactly one Same hit, got {hits:?}"
        );

        // create("Same") builds v2 (amplitude 999), not the stale v1.
        let inst = reg.create("Same").expect("creatable");
        let produced = inst.cook(NodeInputs::new()).unwrap();
        let h = produced["out"].as_terrain().unwrap().heights[100];
        assert_eq!(h, v2_ref, "create() must build the replacement (v2), not the stale v1");
    }
}
