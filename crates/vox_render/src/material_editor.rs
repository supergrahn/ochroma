//! Node-based material editor data model.
//!
//! A visual material editor graph that evaluates to spectral material properties.

use serde::{Deserialize, Serialize};

/// A material editor graph — nodes with connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialGraph {
    pub name: String,
    pub nodes: Vec<MaterialEditorNode>,
    pub connections: Vec<MaterialConnection>,
    pub output_node_id: u32,
    next_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialEditorNode {
    pub id: u32,
    pub node_type: MaterialNodeType,
    pub position: [f32; 2],
    pub preview_color: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialNodeType {
    // Output (final node)
    MaterialOutput,
    // Constants
    SpectralConstant { bands: [f32; 8] },
    FloatConstant { value: f32 },
    ColorConstant { r: f32, g: f32, b: f32 },
    // Textures
    TextureCoordinate,
    TextureSample { path: String },
    // Math
    Add,
    Subtract,
    Multiply,
    Divide,
    Lerp,
    Power,
    Sqrt,
    Abs,
    OneMinus,
    // Spectral
    SpectralBlend { factor: f32 },
    SpectralShift { wavelength_offset: f32 },
    WearBlend { wear_factor: f32 },
    // Surface
    FresnelEffect { power: f32 },
    Roughness { value: f32 },
    Metallic { value: f32 },
    Emission { intensity: f32 },
    Opacity { value: f32 },
    // Procedural
    PerlinNoise { scale: f32, octaves: u32 },
    VoronoiNoise { scale: f32 },
    Checker { scale: f32 },
    Gradient { direction: [f32; 3] },
    // Utility
    Remap {
        from_min: f32,
        from_max: f32,
        to_min: f32,
        to_max: f32,
    },
    SmoothStep { edge0: f32, edge1: f32 },
    Time,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialConnection {
    pub from_node: u32,
    pub from_output: String,
    pub to_node: u32,
    pub to_input: String,
}

impl MaterialGraph {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            connections: Vec::new(),
            output_node_id: 0,
            next_id: 0,
        }
    }

    pub fn add_node(&mut self, node_type: MaterialNodeType, pos: [f32; 2]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(MaterialEditorNode {
            id,
            node_type,
            position: pos,
            preview_color: None,
        });
        id
    }

    pub fn connect(&mut self, from: u32, from_pin: &str, to: u32, to_pin: &str) {
        self.connections.push(MaterialConnection {
            from_node: from,
            from_output: from_pin.to_string(),
            to_node: to,
            to_input: to_pin.to_string(),
        });
    }

    pub fn remove_node(&mut self, id: u32) {
        self.nodes.retain(|n| n.id != id);
        self.connections
            .retain(|c| c.from_node != id && c.to_node != id);
    }

    /// Export the graph as a TOML material definition.
    pub fn compile_to_toml(&self) -> Result<String, String> {
        if self.nodes.is_empty() {
            return Err("Graph has no nodes".to_string());
        }

        let mut toml = String::new();
        toml.push_str(&format!("[material]\nname = \"{}\"\n\n", self.name));

        for node in &self.nodes {
            toml.push_str(&format!("[[nodes]]\nid = {}\n", node.id));
            toml.push_str(&format!("position = [{}, {}]\n", node.position[0], node.position[1]));

            match &node.node_type {
                MaterialNodeType::MaterialOutput => {
                    toml.push_str("type = \"MaterialOutput\"\n");
                }
                MaterialNodeType::SpectralConstant { bands } => {
                    toml.push_str("type = \"SpectralConstant\"\n");
                    toml.push_str(&format!(
                        "bands = [{}, {}, {}, {}, {}, {}, {}, {}]\n",
                        bands[0], bands[1], bands[2], bands[3],
                        bands[4], bands[5], bands[6], bands[7]
                    ));
                }
                MaterialNodeType::FloatConstant { value } => {
                    toml.push_str("type = \"FloatConstant\"\n");
                    toml.push_str(&format!("value = {}\n", value));
                }
                MaterialNodeType::ColorConstant { r, g, b } => {
                    toml.push_str("type = \"ColorConstant\"\n");
                    toml.push_str(&format!("color = [{}, {}, {}]\n", r, g, b));
                }
                MaterialNodeType::Roughness { value } => {
                    toml.push_str("type = \"Roughness\"\n");
                    toml.push_str(&format!("value = {}\n", value));
                }
                MaterialNodeType::Metallic { value } => {
                    toml.push_str("type = \"Metallic\"\n");
                    toml.push_str(&format!("value = {}\n", value));
                }
                MaterialNodeType::Emission { intensity } => {
                    toml.push_str("type = \"Emission\"\n");
                    toml.push_str(&format!("intensity = {}\n", intensity));
                }
                MaterialNodeType::Opacity { value } => {
                    toml.push_str("type = \"Opacity\"\n");
                    toml.push_str(&format!("value = {}\n", value));
                }
                other => {
                    toml.push_str(&format!("type = \"{:?}\"\n", std::mem::discriminant(other)));
                }
            }
            toml.push('\n');
        }

        for conn in &self.connections {
            toml.push_str("[[connections]]\n");
            toml.push_str(&format!(
                "from = {{ node = {}, pin = \"{}\" }}\n",
                conn.from_node, conn.from_output
            ));
            toml.push_str(&format!(
                "to = {{ node = {}, pin = \"{}\" }}\n\n",
                conn.to_node, conn.to_input
            ));
        }

        Ok(toml)
    }

    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn load_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph() {
        let graph = MaterialGraph::new("test_material");
        assert_eq!(graph.name, "test_material");
        assert_eq!(graph.node_count(), 0);
        assert!(graph.connections.is_empty());
    }

    #[test]
    fn test_add_nodes_and_connections() {
        let mut graph = MaterialGraph::new("brick");
        let output = graph.add_node(MaterialNodeType::MaterialOutput, [400.0, 200.0]);
        let color = graph.add_node(
            MaterialNodeType::SpectralConstant {
                bands: [0.1, 0.15, 0.2, 0.3, 0.25, 0.2, 0.15, 0.1],
            },
            [100.0, 100.0],
        );
        let roughness = graph.add_node(MaterialNodeType::Roughness { value: 0.7 }, [100.0, 300.0]);

        graph.connect(color, "out", output, "base_color");
        graph.connect(roughness, "out", output, "roughness");

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.connections.len(), 2);
        assert_eq!(graph.connections[0].from_node, color);
        assert_eq!(graph.connections[0].to_node, output);
    }

    #[test]
    fn test_compile_to_toml() {
        let mut graph = MaterialGraph::new("metal");
        let output = graph.add_node(MaterialNodeType::MaterialOutput, [400.0, 200.0]);
        graph.output_node_id = output;
        let spectral = graph.add_node(
            MaterialNodeType::SpectralConstant {
                bands: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            },
            [100.0, 200.0],
        );
        graph.connect(spectral, "out", output, "base_color");

        let toml = graph.compile_to_toml().unwrap();
        assert!(toml.contains("name = \"metal\""));
        assert!(toml.contains("type = \"MaterialOutput\""));
        assert!(toml.contains("type = \"SpectralConstant\""));
        assert!(toml.contains("[[connections]]"));
    }

    #[test]
    fn test_json_round_trip() {
        let mut graph = MaterialGraph::new("glass");
        let output = graph.add_node(MaterialNodeType::MaterialOutput, [400.0, 200.0]);
        graph.output_node_id = output;
        let fresnel = graph.add_node(MaterialNodeType::FresnelEffect { power: 5.0 }, [100.0, 200.0]);
        graph.connect(fresnel, "out", output, "opacity");

        let json = graph.save_json().unwrap();
        let loaded = MaterialGraph::load_json(&json).unwrap();

        assert_eq!(loaded.name, "glass");
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.connections.len(), 1);
        assert_eq!(loaded.output_node_id, output);
    }

    #[test]
    fn test_remove_node_removes_connections() {
        let mut graph = MaterialGraph::new("test");
        let output = graph.add_node(MaterialNodeType::MaterialOutput, [400.0, 200.0]);
        let color = graph.add_node(
            MaterialNodeType::ColorConstant {
                r: 1.0,
                g: 0.0,
                b: 0.0,
            },
            [100.0, 200.0],
        );
        let roughness = graph.add_node(MaterialNodeType::Roughness { value: 0.5 }, [100.0, 300.0]);

        graph.connect(color, "out", output, "base_color");
        graph.connect(roughness, "out", output, "roughness");

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.connections.len(), 2);

        graph.remove_node(color);

        assert_eq!(graph.node_count(), 2);
        // The connection from color should be removed
        assert_eq!(graph.connections.len(), 1);
        assert_eq!(graph.connections[0].from_node, roughness);
    }
}
