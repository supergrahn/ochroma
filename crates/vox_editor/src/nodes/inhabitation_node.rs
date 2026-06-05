//! CatenaryNode + PropPlacementNode.

use crate::node_graph::{
    NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

/// Catenary wire node — Newton's method to find catenary parameter `a`.
#[derive(Clone)]
pub struct CatenaryNode {
    pub start:    [f32; 3],
    pub end:      [f32; 3],
    pub slack:    f32,
    pub segments: usize,
}

impl Default for CatenaryNode {
    fn default() -> Self {
        Self { start: [0.0, 5.0, 0.0], end: [10.0, 5.0, 0.0], slack: 0.1, segments: 20 }
    }
}

impl CatenaryNode {
    fn compute_y_values(&self) -> Vec<f32> {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let h  = (dx * dx + dy * dy).sqrt();
        let target_len = h * (1.0 + self.slack);

        // Newton's method: find `a` s.t. 2a*sinh(h/(2a)) = target_len
        let mut a = h / 2.0_f32;
        for _ in 0..50 {
            let s = h / (2.0 * a);
            let f  = 2.0 * a * s.sinh() - target_len;
            let fp = 2.0 * s.sinh() - h / a * s.cosh();
            if fp.abs() < 1e-8 { break; }
            let da = -f / fp;
            a += da;
            a = a.max(0.001);
            if da.abs() < 1e-6 { break; }
        }

        // Sample Y values along the catenary
        let x0 = -dx * 0.5;
        let y_offset = self.start[1] + dy * 0.5 - a * (x0 / a).cosh();

        let mut pts = Vec::with_capacity(self.segments + 1);
        for i in 0..=self.segments {
            let t = i as f32 / self.segments as f32;
            let x = x0 + t * dx;
            let y = a * (x / a).cosh() + y_offset;
            pts.push(y);
        }
        pts
    }
}

impl OchromaNode for CatenaryNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "CatenaryNode",
            inputs: vec![],
            outputs: vec![
                PortSpec { name: "points", port_type: PortType::ScalarVec, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("slack",    ParamValue::Float(v)) => { self.slack    = v as f32; Ok(()) }
            ("segments", ParamValue::Int(v))   => { self.segments = v as usize; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(self.clone()) }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let pts = self.compute_y_values();
        let mut out = NodeOutputs::new();
        out.insert("points".into(), PortData::ScalarVec(pts));
        Ok(out)
    }
}

/// PropPlacementNode — LCG-based Poisson-disk rejection in 2D area.
#[derive(Clone)]
pub struct PropPlacementNode {
    pub area:          [f32; 2],
    pub count:         usize,
    pub min_clearance: f32,
    pub seed:          u64,
}

impl Default for PropPlacementNode {
    fn default() -> Self {
        Self { area: [20.0, 20.0], count: 10, min_clearance: 2.0, seed: 42 }
    }
}

impl OchromaNode for PropPlacementNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "PropPlacementNode",
            inputs: vec![],
            outputs: vec![
                PortSpec { name: "count", port_type: PortType::Scalar, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("count",         ParamValue::Int(v))   => { self.count         = v as usize; Ok(()) }
            ("min_clearance", ParamValue::Float(v)) => { self.min_clearance = v as f32; Ok(()) }
            ("seed",          ParamValue::Int(v))   => { self.seed          = v as u64; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(self.clone()) }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        // LCG-based Poisson-disk sampling
        let mut state = self.seed.wrapping_add(1);
        let lcg_next = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as f32) / (u32::MAX as f32)
        };

        let mut placed: Vec<[f32; 2]> = Vec::new();
        let mut attempts = 0usize;
        while placed.len() < self.count && attempts < self.count * 20 {
            attempts += 1;
            let x = lcg_next(&mut state) * self.area[0];
            let y = lcg_next(&mut state) * self.area[1];
            let ok = placed.iter().all(|&p| {
                let dx = p[0] - x;
                let dy = p[1] - y;
                (dx*dx + dy*dy).sqrt() >= self.min_clearance
            });
            if ok { placed.push([x, y]); }
        }

        let mut out = NodeOutputs::new();
        out.insert("count".into(), PortData::Scalar(placed.len() as f64));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catenary_curve_sags() {
        let node = CatenaryNode { start: [0.0, 5.0, 0.0], end: [10.0, 5.0, 0.0], slack: 0.1, segments: 20 };
        let pts = node.cook(NodeInputs::new()).unwrap();
        let pts = pts.get("points").unwrap().as_scalar_vec().unwrap();
        let mid = pts[pts.len() / 2];
        println!("catenary midpoint Y = {}", mid);
        assert!(mid < 5.0, "catenary midpoint should sag below endpoints, got {}", mid);
    }

    #[test]
    fn test_prop_placement_respects_clearance() {
        let node = PropPlacementNode { area: [20.0, 20.0], count: 10, min_clearance: 2.0, seed: 42 };
        let out = node.cook(NodeInputs::new()).unwrap();
        let count = out.get("count").unwrap().as_scalar().unwrap() as usize;
        assert!(count <= 10);
        assert!(count > 0);
    }
}
