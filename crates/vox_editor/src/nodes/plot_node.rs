//! PlotNode — land parcel geometry (ground, driveway, fence, props).

use crate::node_graph::{
    EditorMesh, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

pub struct PlotNode {
    pub archetype:   String,
    pub footprint_w: f32,
    pub footprint_d: f32,
    pub building_w:  f32,
    pub building_d:  f32,
    pub condition:   String,
    pub fence_style: String,
    pub driveway:    bool,
    pub seed:        u64,
}

impl Default for PlotNode {
    fn default() -> Self {
        Self {
            archetype:   "residential_suburban".into(),
            footprint_w: 20.0,
            footprint_d: 30.0,
            building_w:  10.0,
            building_d:  12.0,
            condition:   "new".into(),
            fence_style: "picket".into(),
            driveway:    true,
            seed:        0,
        }
    }
}

fn make_ground_quad(w: f32, d: f32) -> EditorMesh {
    let mut m = EditorMesh::new();
    m.material_id = 3; // dirt
    m.positions = vec![
        [0.0, 0.0, 0.0],
        [w,   0.0, 0.0],
        [w,   0.0, d  ],
        [0.0, 0.0, d  ],
    ];
    m.normals = vec![[0.0, 1.0, 0.0]; 4];
    m.indices  = vec![[0, 1, 2], [0, 2, 3]];
    m
}

impl OchromaNode for PlotNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "PlotNode",
            inputs: vec![],
            outputs: vec![
                PortSpec { name: "ground_mesh", port_type: PortType::Mesh,   optional: false },
                PortSpec { name: "fence_count", port_type: PortType::Scalar, optional: false },
                PortSpec { name: "prop_count",  port_type: PortType::Scalar, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("archetype",   ParamValue::Str(v))   => { self.archetype   = v; Ok(()) }
            ("footprint_w", ParamValue::Float(v)) => { self.footprint_w = v as f32; Ok(()) }
            ("footprint_d", ParamValue::Float(v)) => { self.footprint_d = v as f32; Ok(()) }
            ("building_w",  ParamValue::Float(v)) => { self.building_w  = v as f32; Ok(()) }
            ("building_d",  ParamValue::Float(v)) => { self.building_d  = v as f32; Ok(()) }
            ("condition",   ParamValue::Str(v))   => { self.condition   = v; Ok(()) }
            ("fence_style", ParamValue::Str(v))   => { self.fence_style = v; Ok(()) }
            ("driveway",    ParamValue::Bool(v))  => { self.driveway    = v; Ok(()) }
            ("seed",        ParamValue::Int(v))   => { self.seed        = v as u64; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let ground_mesh = make_ground_quad(self.footprint_w, self.footprint_d);

        // Fence count: perimeter / 1.5m panels, 4 sides
        let perimeter = 2.0 * (self.footprint_w + self.footprint_d);
        let fence_count = (perimeter / 1.5).floor() as f64;

        // Prop count: random garden items
        let prop_count = if self.archetype.contains("residential") { 3.0 } else { 0.0 };

        let mut out = NodeOutputs::new();
        out.insert("ground_mesh".into(), PortData::Mesh(ground_mesh));
        out.insert("fence_count".into(), PortData::Scalar(fence_count));
        out.insert("prop_count".into(),  PortData::Scalar(prop_count));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_node_residential_suburban() {
        let mut node = PlotNode::default();
        node.set_param("archetype",   ParamValue::Str("residential_suburban".into())).unwrap();
        node.set_param("footprint_w", ParamValue::Float(20.0)).unwrap();
        node.set_param("footprint_d", ParamValue::Float(30.0)).unwrap();
        node.set_param("building_w",  ParamValue::Float(10.0)).unwrap();
        node.set_param("building_d",  ParamValue::Float(12.0)).unwrap();
        node.set_param("fence_style", ParamValue::Str("picket".into())).unwrap();
        node.set_param("seed",        ParamValue::Int(42)).unwrap();
        let outputs = node.cook(NodeInputs::new()).unwrap();
        let ground = outputs.get("ground_mesh").expect("ground_mesh output");
        assert!(ground.as_mesh().unwrap().positions.len() > 3);
        let fences = outputs.get("fence_count").expect("fence_count");
        assert!(fences.as_scalar().unwrap() >= 0.0);
    }
}
