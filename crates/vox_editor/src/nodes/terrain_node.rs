//! TerrainNode — fBm heightfield + hydraulic erosion.
//! Adapted from aetherspectra/forge/crates/terrain/src/{generate.rs, hydraulic.rs}.

use noise::{Fbm, NoiseFn, Perlin};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::node_graph::{
    HeightfieldSpatial, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

#[derive(Clone)]
pub struct TerrainNode {
    pub resolution:    u32,
    pub world_size:    f32,
    pub amplitude:     f32,
    pub octaves:       usize,
    pub frequency:     f64,
    pub seed:          u32,
    pub droplet_count: u32,
}

impl Default for TerrainNode {
    fn default() -> Self {
        Self { resolution: 256, world_size: 1000.0, amplitude: 200.0, octaves: 6, frequency: 1.0, seed: 0, droplet_count: 80_000 }
    }
}

fn generate_heightfield(resolution: u32, amplitude: f32, octaves: usize, frequency: f64, seed: u32) -> Vec<f32> {
    let mut fbm: Fbm<Perlin> = Fbm::new(seed);
    fbm.octaves = octaves;
    fbm.frequency = frequency;
    fbm.persistence = 0.5;
    fbm.lacunarity = 2.0;

    let n = resolution as usize;
    let mut heights = Vec::with_capacity(n * n);

    // First pass: gather raw values to find range for normalization
    let mut raw = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let nx = x as f64 / n as f64;
            let ny = y as f64 / n as f64;
            raw.push(fbm.get([nx, ny]) as f32);
        }
    }
    let min = raw.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);

    for v in raw {
        heights.push((v - min) / range * amplitude);
    }
    heights
}

fn hydraulic_erode(heights: &mut Vec<f32>, resolution: u32, droplet_count: u32, seed: u32) {
    let n = resolution as usize;
    let mut rng = Pcg64::seed_from_u64(seed as u64 ^ 0xdeadbeef);

    const INERTIA:     f32 = 0.05;
    const CAPACITY:    f32 = 4.0;
    const DEPOSITION:  f32 = 0.1;
    const EROSION:     f32 = 0.3;
    const EVAPORATION: f32 = 0.02;
    const MIN_SLOPE:   f32 = 0.01;
    const GRAVITY:     f32 = 4.0;

    for _ in 0..droplet_count {
        let mut px: f32 = rng.gen::<f32>() * (n as f32 - 2.0);
        let mut py: f32 = rng.gen::<f32>() * (n as f32 - 2.0);
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        let mut speed = 1.0f32;
        let mut water = 1.0f32;
        let mut sediment = 0.0f32;

        for _ in 0..30 {
            let ix = px as usize;
            let iy = py as usize;
            if ix + 1 >= n || iy + 1 >= n { break; }

            let fx = px - ix as f32;
            let fy = py - iy as f32;

            let h00 = heights[iy * n + ix];
            let h10 = heights[iy * n + ix + 1];
            let h01 = heights[(iy + 1) * n + ix];
            let h11 = heights[(iy + 1) * n + ix + 1];

            // Bilinear height and gradient
            let h = h00 * (1.0-fx)*(1.0-fy) + h10 * fx*(1.0-fy) + h01*(1.0-fx)*fy + h11*fx*fy;
            let gx = (h10 - h00) * (1.0-fy) + (h11 - h01) * fy;
            let gy = (h01 - h00) * (1.0-fx) + (h11 - h10) * fx;

            dx = dx * INERTIA - gx * (1.0 - INERTIA);
            dy = dy * INERTIA - gy * (1.0 - INERTIA);
            let len = (dx * dx + dy * dy).sqrt().max(1e-8);
            dx /= len;
            dy /= len;

            let new_px = px + dx;
            let new_py = py + dy;

            if new_px < 0.0 || new_px >= (n-1) as f32 || new_py < 0.0 || new_py >= (n-1) as f32 { break; }

            let ix2 = new_px as usize;
            let iy2 = new_py as usize;
            if ix2 + 1 >= n || iy2 + 1 >= n { break; }

            let fx2 = new_px - ix2 as f32;
            let fy2 = new_py - iy2 as f32;
            let h_new = heights[iy2 * n + ix2] * (1.0-fx2)*(1.0-fy2)
                      + heights[iy2 * n + ix2+1] * fx2*(1.0-fy2)
                      + heights[(iy2+1) * n + ix2] * (1.0-fx2)*fy2
                      + heights[(iy2+1) * n + ix2+1] * fx2*fy2;

            let dh = h_new - h;
            let cap = (-dh).max(MIN_SLOPE) * speed * water * CAPACITY;

            if dh > 0.0 || sediment > cap {
                // Deposit
                let deposit = if dh > 0.0 { sediment.min(dh) } else { (sediment - cap) * DEPOSITION };
                sediment -= deposit;
                let d = deposit;
                heights[iy * n + ix]       += d * (1.0-fx) * (1.0-fy);
                heights[iy * n + ix+1]     += d * fx       * (1.0-fy);
                heights[(iy+1) * n + ix]   += d * (1.0-fx) * fy;
                heights[(iy+1) * n + ix+1] += d * fx       * fy;
            } else {
                // Erode
                let erode = ((cap - sediment) * EROSION).min(-dh);
                sediment += erode;
                heights[iy * n + ix]       -= erode * (1.0-fx) * (1.0-fy);
                heights[iy * n + ix+1]     -= erode * fx       * (1.0-fy);
                heights[(iy+1) * n + ix]   -= erode * (1.0-fx) * fy;
                heights[(iy+1) * n + ix+1] -= erode * fx       * fy;
                // Clamp to 0
                heights[iy * n + ix] = heights[iy * n + ix].max(0.0);
                heights[iy * n + ix+1] = heights[iy * n + ix+1].max(0.0);
                heights[(iy+1) * n + ix] = heights[(iy+1) * n + ix].max(0.0);
                heights[(iy+1) * n + ix+1] = heights[(iy+1) * n + ix+1].max(0.0);
            }

            speed = (speed * speed + dh * GRAVITY).abs().sqrt();
            water *= 1.0 - EVAPORATION;
            px = new_px;
            py = new_py;
        }
    }
}

impl OchromaNode for TerrainNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "TerrainNode",
            inputs: vec![],
            outputs: vec![PortSpec { name: "terrain", port_type: PortType::Terrain, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("resolution",    ParamValue::Int(v))   => { self.resolution    = v as u32; Ok(()) }
            ("world_size",    ParamValue::Float(v)) => { self.world_size    = v as f32; Ok(()) }
            ("amplitude",     ParamValue::Float(v)) => { self.amplitude     = v as f32; Ok(()) }
            ("octaves",       ParamValue::Int(v))   => { self.octaves       = v as usize; Ok(()) }
            ("frequency",     ParamValue::Float(v)) => { self.frequency     = v; Ok(()) }
            ("seed",          ParamValue::Int(v))   => { self.seed          = v as u32; Ok(()) }
            ("droplet_count", ParamValue::Int(v))   => { self.droplet_count = v as u32; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(self.clone()) }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        if self.resolution < 16 {
            return Err(NodeError::CookFailed(format!("resolution must be >= 16, got {}", self.resolution)));
        }
        let mut heights = generate_heightfield(self.resolution, self.amplitude, self.octaves, self.frequency, self.seed);
        if self.droplet_count > 0 {
            hydraulic_erode(&mut heights, self.resolution, self.droplet_count, self.seed);
        }
        let hf = HeightfieldSpatial { heights, resolution: self.resolution, world_size: self.world_size };
        let mut out = NodeOutputs::new();
        out.insert("terrain".into(), PortData::Terrain(hf));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_node_cooks_without_error() {
        let node = TerrainNode { resolution: 64, droplet_count: 0, ..Default::default() };
        let out = node.cook(NodeInputs::new()).unwrap();
        assert!(out.contains_key("terrain"));
    }

    #[test]
    fn output_height_count_matches_resolution() {
        let node = TerrainNode { resolution: 64, droplet_count: 0, ..Default::default() };
        let out = node.cook(NodeInputs::new()).unwrap();
        let hf = out["terrain"].as_terrain().unwrap();
        assert_eq!(hf.heights.len(), 64 * 64);
        assert_eq!(hf.resolution, 64);
    }

    #[test]
    fn heights_are_in_valid_range() {
        let node = TerrainNode { resolution: 64, amplitude: 100.0, droplet_count: 0, ..Default::default() };
        let out = node.cook(NodeInputs::new()).unwrap();
        let hf = out["terrain"].as_terrain().unwrap();
        for &h in &hf.heights {
            assert!(!h.is_nan(), "height is NaN");
            assert!(h >= 0.0 && h <= 200.0, "height out of [0, amplitude*2]: {}", h);
        }
    }

    #[test]
    fn different_seeds_produce_different_terrain() {
        let a = TerrainNode { resolution: 32, seed: 1, droplet_count: 0, ..Default::default() };
        let b = TerrainNode { resolution: 32, seed: 2, droplet_count: 0, ..Default::default() };
        let ha = a.cook(NodeInputs::new()).unwrap();
        let hb = b.cook(NodeInputs::new()).unwrap();
        let ha = ha["terrain"].as_terrain().unwrap();
        let hb = hb["terrain"].as_terrain().unwrap();
        let diff: f32 = ha.heights.iter().zip(hb.heights.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "different seeds should produce different terrain, total diff={}", diff);
    }

    #[test]
    fn erosion_modifies_terrain() {
        let no_erosion   = TerrainNode { resolution: 32, droplet_count: 0,    seed: 42, ..Default::default() };
        let with_erosion = TerrainNode { resolution: 32, droplet_count: 1000, seed: 42, ..Default::default() };
        let ha = no_erosion.cook(NodeInputs::new()).unwrap();
        let hb = with_erosion.cook(NodeInputs::new()).unwrap();
        let ha = ha["terrain"].as_terrain().unwrap();
        let hb = hb["terrain"].as_terrain().unwrap();
        let diff: f32 = ha.heights.iter().zip(hb.heights.iter()).map(|(a, b)| (a - b).abs()).sum();
        println!("erosion diff = {}", diff);
        assert!(diff > 0.01, "erosion should modify terrain heights, diff={}", diff);
    }

    #[test]
    fn invalid_resolution_returns_error() {
        let node = TerrainNode { resolution: 8, ..Default::default() };
        let err = node.cook(NodeInputs::new()).unwrap_err();
        assert!(matches!(err, NodeError::CookFailed(_)));
    }

    #[test]
    fn terrain_node_in_graph_cooks_end_to_end() {
        use crate::node_graph::OchromaNodeGraph;
        let mut graph = OchromaNodeGraph::new();
        let terrain_id = graph.add_node("terrain", Box::new(TerrainNode { resolution: 32, droplet_count: 0, ..Default::default() }));
        graph.cook().unwrap();
        assert!(graph.get_output(terrain_id, "terrain").is_some());
    }
}
