//! SplatWeightNode — BiomeMap → per-cell splat blend weights.

use crate::node_graph::{
    NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};
use crate::nodes::biome_node::{biome_to_splat_weights, BiomeKind};

pub struct SplatWeightNode;

impl Default for SplatWeightNode {
    fn default() -> Self { Self }
}

impl OchromaNode for SplatWeightNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "SplatWeightNode",
            inputs: vec![
                PortSpec { name: "biome_map", port_type: PortType::BiomeMap,   optional: false },
                PortSpec { name: "moisture",  port_type: PortType::ScalarVec,  optional: true  },
            ],
            outputs: vec![
                PortSpec { name: "splat_weights", port_type: PortType::SplatWeights, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), NodeError> {
        Err(NodeError::UnknownParam(key.into()))
    }

    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let biome_map = inputs.get("biome_map")
            .ok_or_else(|| NodeError::MissingInput("biome_map".into()))?
            .as_biome_map()
            .ok_or_else(|| NodeError::TypeMismatch("biome_map".into()))?;

        let moisture = inputs.get("moisture").and_then(|d| d.as_scalar_vec());

        let weights: Vec<[f32; 4]> = biome_map.iter().enumerate().map(|(i, &b)| {
            let kind = BiomeKind::from_u8(b);
            let m = moisture.map(|mv| mv.get(i).cloned().unwrap_or(0.5)).unwrap_or(0.5);
            let mut w = biome_to_splat_weights(kind, 0.0, m);
            // If moisture data provided, blend some weight toward water slot
            if moisture.is_some() && m > 0.5 {
                let blend = (m - 0.5) * 0.4;
                let drain = blend.min(w[1] + w[2] + w[3]);
                let ratio_1 = w[1] / (w[1] + w[2] + w[3] + 1e-8);
                let ratio_2 = w[2] / (w[1] + w[2] + w[3] + 1e-8);
                let ratio_3 = w[3] / (w[1] + w[2] + w[3] + 1e-8);
                w[0] += drain;
                w[1] -= drain * ratio_1;
                w[2] -= drain * ratio_2;
                w[3] -= drain * ratio_3;
            }
            // Normalize
            let sum: f32 = w.iter().sum();
            if sum > 1e-8 { w.iter_mut().for_each(|v| *v /= sum); }
            w
        }).collect();

        let mut out = NodeOutputs::new();
        out.insert("splat_weights".into(), PortData::SplatWeights(weights));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::biome_node::BiomeKind;

    #[test]
    fn test_splat_weight_node_forest_dominant_veg() {
        // Forest biome → vegetation channel (2) > 0.5 for all cells
        let biome_map = vec![BiomeKind::Forest as u8; 4];
        let mut inputs = NodeInputs::new();
        inputs.insert("biome_map".into(), PortData::BiomeMap(biome_map));
        let node = SplatWeightNode;
        let out = node.cook(inputs).unwrap();
        let weights = out["splat_weights"].as_splat_weights().unwrap();
        for w in weights {
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "weights must sum to 1, got {sum}");
            assert!(w[2] > 0.5, "Forest veg channel should dominate: {:?}", w);
        }
    }

    #[test]
    fn test_splat_weight_node_custom_biome_map() {
        // Desert → ground (1) dominant; Wetland → water (0) significant
        let node = SplatWeightNode;
        let desert_map = vec![BiomeKind::Desert as u8];
        let mut inputs = NodeInputs::new();
        inputs.insert("biome_map".into(), PortData::BiomeMap(desert_map));
        let out = node.cook(inputs).unwrap();
        let w = &out["splat_weights"].as_splat_weights().unwrap()[0];
        assert!(w[1] > w[2], "Desert: ground channel should be > vegetation");

        let wetland_map = vec![BiomeKind::Wetland as u8];
        let mut inputs2 = NodeInputs::new();
        inputs2.insert("biome_map".into(), PortData::BiomeMap(wetland_map));
        let out2 = node.cook(inputs2).unwrap();
        let w2 = &out2["splat_weights"].as_splat_weights().unwrap()[0];
        assert!(w2[0] > 0.2, "Wetland: water channel should be significant");
    }
}
