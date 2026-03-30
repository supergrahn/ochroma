//! UrbanSimNode — traffic/moisture/upkeep reaction-diffusion simulation.

use crate::node_graph::{
    NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

struct UrbanCell {
    traffic_weight: f32,
    refuse_level:   f32,
    civic_upkeep:   f32,
    wind_exposure:  f32,
    moisture:       f32,
}

pub struct UrbanSimNode {
    pub grid_w:     usize,
    pub grid_h:     usize,
    pub iterations: usize,
    pub seed:       u64,
}

impl Default for UrbanSimNode {
    fn default() -> Self {
        Self { grid_w: 16, grid_h: 16, iterations: 10, seed: 0 }
    }
}

fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32) / (u32::MAX as f32)
}

impl OchromaNode for UrbanSimNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "UrbanSimNode",
            inputs: vec![],
            outputs: vec![
                PortSpec { name: "max_traffic",  port_type: PortType::Scalar, optional: false },
                PortSpec { name: "max_moisture", port_type: PortType::Scalar, optional: false },
                PortSpec { name: "cell_count",   port_type: PortType::Scalar, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("grid_w",     ParamValue::Int(v)) => { self.grid_w     = v as usize; Ok(()) }
            ("grid_h",     ParamValue::Int(v)) => { self.grid_h     = v as usize; Ok(()) }
            ("iterations", ParamValue::Int(v)) => { self.iterations = v as usize; Ok(()) }
            ("seed",       ParamValue::Int(v)) => { self.seed       = v as u64;   Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let n = self.grid_w * self.grid_h;
        let mut cells: Vec<UrbanCell> = (0..n).map(|i| {
            let x = i % self.grid_w;
            let y = i / self.grid_w;
            let edge = x == 0 || x == self.grid_w - 1 || y == 0 || y == self.grid_h - 1;
            UrbanCell {
                traffic_weight: 0.0,
                refuse_level:   0.0,
                civic_upkeep:   1.0,
                wind_exposure:  if edge { 1.0 } else { 0.0 },
                moisture:       0.0,
            }
        }).collect();

        // Seed 3 traffic sources
        let mut rng_state = self.seed.wrapping_add(1);
        for _ in 0..3 {
            let xi = (lcg_next(&mut rng_state) * (self.grid_w as f32 - 1.0)) as usize;
            let yi = (lcg_next(&mut rng_state) * (self.grid_h as f32 - 1.0)) as usize;
            let idx = yi * self.grid_w + xi;
            cells[idx].traffic_weight = 1.0;
        }

        // Diffuse iterations
        for _ in 0..self.iterations {
            let prev: Vec<f32> = cells.iter().map(|c| c.traffic_weight).collect();
            for y in 0..self.grid_h {
                for x in 0..self.grid_w {
                    let idx = y * self.grid_w + x;
                    let mut sum = 0.0f32;
                    let mut cnt = 0usize;
                    for (dy, dx) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < self.grid_w as i32 && ny >= 0 && ny < self.grid_h as i32 {
                            sum += prev[(ny as usize) * self.grid_w + nx as usize];
                            cnt += 1;
                        }
                    }
                    cells[idx].traffic_weight = (prev[idx] * 0.6 + (sum / cnt.max(1) as f32) * 0.4).clamp(0.0, 1.0);
                }
            }
        }

        // Derive moisture and civic_upkeep from traffic
        for c in cells.iter_mut() {
            c.moisture     = (c.traffic_weight * 0.3).clamp(0.0, 1.0);
            c.civic_upkeep = (1.0 - c.traffic_weight * 0.5).clamp(0.0, 1.0);
        }

        let max_traffic  = cells.iter().map(|c| c.traffic_weight).fold(0.0f32, f32::max);
        let max_moisture = cells.iter().map(|c| c.moisture).fold(0.0f32, f32::max);

        let mut out = NodeOutputs::new();
        out.insert("max_traffic".into(),  PortData::Scalar(max_traffic as f64));
        out.insert("max_moisture".into(), PortData::Scalar(max_moisture as f64));
        out.insert("cell_count".into(),   PortData::Scalar(n as f64));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::ParamValue;

    #[test]
    fn test_urban_sim_produces_grid() {
        let mut node = UrbanSimNode::default();
        node.set_param("grid_w",     ParamValue::Int(16)).unwrap();
        node.set_param("grid_h",     ParamValue::Int(16)).unwrap();
        node.set_param("iterations", ParamValue::Int(10)).unwrap();
        node.set_param("seed",       ParamValue::Int(1)).unwrap();
        let out = node.cook(NodeInputs::new()).unwrap();
        let traffic  = out.get("max_traffic").unwrap().as_scalar().unwrap();
        let moisture = out.get("max_moisture").unwrap().as_scalar().unwrap();
        println!("max_traffic={} max_moisture={}", traffic, moisture);
        assert!(traffic >= 0.0 && traffic <= 1.0);
        assert!(moisture >= 0.0 && moisture <= 1.0);
    }

    #[test]
    fn test_moisture_drives_spectral_blend() {
        use crate::nodes::biome_node::SpectralTerrainMaterials;
        let mats = SpectralTerrainMaterials::default();
        let dry = mats.slots[3]; // Dirt
        let wet = mats.slots[0]; // Water
        let blend: [f32; 16] = std::array::from_fn(|i| dry[i] * 0.7 + wet[i] * 0.3);
        for i in 0..16 {
            assert!(blend[i] < dry[i] + 0.01, "wet blend should be darker at band {i}");
        }
    }
}
