//! MoistureNode — drip + urban moisture → per-cell scalar.

use crate::node_graph::{
    NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

#[derive(Clone)]
pub struct MoistureNode {
    pub urban_scale: f32,
}

impl Default for MoistureNode {
    fn default() -> Self { Self { urban_scale: 1.0 } }
}

impl MoistureNode {
    /// Combine drip and optional urban moisture per cell — per-cell max.
    pub fn combine(&self, drip: &[f32], urban: Option<&[f32]>) -> Vec<f32> {
        drip.iter().enumerate().map(|(i, &d)| {
            let u = urban.and_then(|u| u.get(i)).cloned().unwrap_or(0.0) * self.urban_scale;
            d.max(u).clamp(0.0, 1.0)
        }).collect()
    }

    /// Linear blend between base spectral and water spectral by moisture amount.
    pub fn blend_moisture(base: &[f32; 16], water: &[f32; 16], moisture: f32) -> [f32; 16] {
        let t = moisture.clamp(0.0, 1.0);
        std::array::from_fn(|i| base[i] * (1.0 - t) + water[i] * t)
    }
}

impl OchromaNode for MoistureNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "MoistureNode",
            inputs: vec![
                PortSpec { name: "drip",  port_type: PortType::ScalarVec, optional: false },
                PortSpec { name: "urban", port_type: PortType::ScalarVec, optional: true  },
            ],
            outputs: vec![
                PortSpec { name: "moisture", port_type: PortType::ScalarVec, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("urban_scale", ParamValue::Float(v)) => { self.urban_scale = v as f32; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(self.clone()) }

    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let drip = inputs.get("drip")
            .ok_or_else(|| NodeError::MissingInput("drip".into()))?
            .as_scalar_vec()
            .ok_or_else(|| NodeError::TypeMismatch("drip".into()))?;

        let urban = inputs.get("urban").and_then(|d| d.as_scalar_vec());
        let combined = self.combine(drip, urban.map(|v| v.as_slice()));

        let mut out = NodeOutputs::new();
        out.insert("moisture".into(), PortData::ScalarVec(combined));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moisture_node_combines_drip_and_urban() {
        let node = MoistureNode::default();
        let drip  = vec![0.8f32, 0.1, 0.0, 0.5];
        let urban = vec![0.2f32, 0.0, 0.9, 0.1];
        let result = node.combine(&drip, Some(&urban));
        assert!((result[0] - 0.8).abs() < 0.01); // max(0.8, 0.2) = 0.8
        assert!((result[2] - 0.9).abs() < 0.01); // max(0.0, 0.9) = 0.9
    }

    #[test]
    fn test_moisture_node_drip_only() {
        let node = MoistureNode::default();
        let drip = vec![0.3f32, 0.7, 0.0];
        let result = node.combine(&drip, None);
        assert!((result[0] - 0.3).abs() < 0.01);
        assert!((result[1] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_splat_weight_node_moisture_darkens_alpine() {
        use crate::nodes::biome_node::SpectralTerrainMaterials;
        let mats = SpectralTerrainMaterials::default();
        let dry  = mats.slots[3]; // Dirt
        let wet  = mats.slots[0]; // Water
        let blend = MoistureNode::blend_moisture(&dry, &wet, 0.3);
        // Wet blend at band 0 should be less than dry (water is darker)
        assert!(blend[0] < dry[0] + 0.01, "wet blend should be darker at band 0: blend={} dry={}", blend[0], dry[0]);
    }
}
