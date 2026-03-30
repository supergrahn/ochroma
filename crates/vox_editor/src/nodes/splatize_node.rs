//! SplatizeNode — converts EditorMesh to GaussianSplat[] with spectral assignment.
//! Area-weighted random triangle sampling + Smits-style material→spectral upsampling.

use half::f16;
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use glam::Quat;

use vox_core::types::GaussianSplat;
use crate::node_graph::{
    EditorMesh, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

pub struct SplatizeNode {
    pub splats_per_sqm: f64,
    pub min_splats:     usize,
    pub max_splats:     usize,
    pub seed:           u64,
    pub splat_scale:    f32,
}

impl Default for SplatizeNode {
    fn default() -> Self {
        Self { splats_per_sqm: 10.0, min_splats: 100, max_splats: 100_000, seed: 0, splat_scale: 0.1 }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn triangle_area(p0: [f32;3], p1: [f32;3], p2: [f32;3]) -> f32 {
    let ab = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
    let ac = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
    let c = cross(ab, ac);
    (c[0]*c[0]+c[1]*c[1]+c[2]*c[2]).sqrt() * 0.5
}

/// Smits-style spectral upsampling — 8 material profiles.
/// Band layout: 380–755nm at 25nm steps (16 bands).
/// Material IDs:
///   0=gray/concrete  1=wood  2=foliage  3=stone  4=metal  5=brick  6=glass  7=fire
pub fn spectral_from_material(material_id: u32) -> [u16; 16] {
    // Each profile is 16 f32 values → converted to f16 bits
    // Bands roughly: [UV-near, UV, VioBlue, Blue, CyanBlue, Cyan, GreenCyan, Green,
    //                  YellowGreen, Yellow, OrangeYellow, Orange, Red, DeepRed, NIR-near, NIR]
    let profile: [f32; 16] = match material_id {
        0 => [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43], // gray concrete
        1 => [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.28, 0.35, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28], // wood (warm brown)
        2 => [0.04, 0.04, 0.05, 0.05, 0.06, 0.08, 0.30, 0.45, 0.42, 0.30, 0.10, 0.06, 0.04, 0.04, 0.40, 0.50], // foliage (green + NIR)
        3 => [0.15, 0.18, 0.20, 0.22, 0.23, 0.24, 0.25, 0.25, 0.26, 0.26, 0.25, 0.24, 0.23, 0.22, 0.22, 0.21], // stone
        4 => [0.55, 0.58, 0.60, 0.62, 0.63, 0.64, 0.65, 0.65, 0.65, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.59], // metal
        5 => [0.04, 0.05, 0.07, 0.09, 0.10, 0.12, 0.14, 0.16, 0.22, 0.32, 0.42, 0.50, 0.52, 0.48, 0.45, 0.43], // brick (strong red)
        6 => [0.08, 0.09, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09], // glass
        7 => [0.10, 0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.55, 0.65, 0.75, 0.80, 0.78, 0.70, 0.60, 0.50, 0.40], // fire/emissive
        _ => [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40], // fallback
    };
    let mut out = [0u16; 16];
    for (i, &v) in profile.iter().enumerate() {
        out[i] = f16::from_f32(v).to_bits();
    }
    out
}

fn splatize(mesh: &EditorMesh, splats_per_sqm: f64, min_splats: usize, max_splats: usize, seed: u64, splat_scale: f32) -> Vec<GaussianSplat> {
    if mesh.indices.is_empty() || mesh.positions.is_empty() {
        return Vec::new();
    }

    // Compute triangle areas
    let areas: Vec<f32> = mesh.indices.iter().map(|tri| {
        let p0 = mesh.positions[tri[0] as usize];
        let p1 = mesh.positions[tri[1] as usize];
        let p2 = mesh.positions[tri[2] as usize];
        triangle_area(p0, p1, p2)
    }).collect();

    let total_area: f32 = areas.iter().sum();
    if total_area < 1e-10 { return Vec::new(); }

    let desired = ((total_area as f64 * splats_per_sqm) as usize).clamp(min_splats, max_splats);

    let mut rng = Pcg64::seed_from_u64(seed);

    // Build CDF for area-weighted sampling
    let mut cdf = Vec::with_capacity(areas.len());
    let mut cumsum = 0.0f32;
    for &a in &areas {
        cumsum += a;
        cdf.push(cumsum / total_area);
    }

    let spectral = spectral_from_material(mesh.material_id);
    let mut splats = Vec::with_capacity(desired);

    for _ in 0..desired {
        // Sample triangle by CDF
        let r: f32 = rng.gen();
        let tri_idx = cdf.partition_point(|&v| v < r).min(mesh.indices.len() - 1);
        let tri = mesh.indices[tri_idx];

        let p0 = mesh.positions[tri[0] as usize];
        let p1 = mesh.positions[tri[1] as usize];
        let p2 = mesh.positions[tri[2] as usize];

        // Barycentric coordinates
        let u: f32 = rng.gen();
        let v: f32 = rng.gen();
        let (su, sv) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };
        let sw = 1.0 - su - sv;

        let pos = [
            p0[0]*sw + p1[0]*su + p2[0]*sv,
            p0[1]*sw + p1[1]*su + p2[1]*sv,
            p0[2]*sw + p1[2]*su + p2[2]*sv,
        ];

        let splat = GaussianSplat::volume(
            pos,
            [splat_scale, splat_scale, splat_scale],
            Quat::IDENTITY,
            200u8,
            spectral,
        );
        splats.push(splat);
    }
    splats
}

impl OchromaNode for SplatizeNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "SplatizeNode",
            inputs: vec![PortSpec { name: "mesh", port_type: PortType::Mesh, optional: false }],
            outputs: vec![PortSpec { name: "splats", port_type: PortType::Splats, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("splats_per_sqm", ParamValue::Float(v)) => { self.splats_per_sqm = v; Ok(()) }
            ("min_splats",     ParamValue::Int(v))   => { self.min_splats     = v as usize; Ok(()) }
            ("max_splats",     ParamValue::Int(v))   => { self.max_splats     = v as usize; Ok(()) }
            ("seed",           ParamValue::Int(v))   => { self.seed           = v as u64; Ok(()) }
            ("splat_scale",    ParamValue::Float(v)) => { self.splat_scale    = v as f32; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let mesh = inputs.get("mesh")
            .ok_or_else(|| NodeError::MissingInput("mesh".into()))?
            .as_mesh()
            .ok_or_else(|| NodeError::TypeMismatch("mesh".into()))?;

        let splats = splatize(mesh, self.splats_per_sqm, self.min_splats, self.max_splats, self.seed, self.splat_scale);
        let mut out = NodeOutputs::new();
        out.insert("splats".into(), PortData::Splats(splats));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> EditorMesh {
        EditorMesh {
            positions: vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,1.0],[0.0,0.0,1.0]],
            normals: vec![[0.0,1.0,0.0]; 4],
            indices: vec![[0,1,2],[0,2,3]],
            material_id: 0,
        }
    }

    #[test]
    fn splatize_produces_splats() {
        let node = SplatizeNode { min_splats: 10, max_splats: 100, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(!splats.is_empty(), "should produce splats from quad mesh");
    }

    #[test]
    fn splat_count_respects_min() {
        let node = SplatizeNode { splats_per_sqm: 0.001, min_splats: 50, max_splats: 1000, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(splats.len() >= 50, "should respect min_splats, got {}", splats.len());
    }

    #[test]
    fn splat_count_respects_max() {
        let node = SplatizeNode { splats_per_sqm: 1e9, min_splats: 0, max_splats: 200, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(splats.len() <= 200, "should respect max_splats, got {}", splats.len());
    }

    #[test]
    fn splat_positions_lie_within_mesh_bounds() {
        let node = SplatizeNode { min_splats: 20, max_splats: 100, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        for s in splats {
            assert!((0.0..=1.0).contains(&s.position()[0]), "x out of [0,1]: {}", s.position()[0]);
            assert!((0.0..=1.0).contains(&s.position()[2]), "z out of [0,1]: {}", s.position()[2]);
        }
    }

    #[test]
    fn splat_spectral_is_nonzero() {
        let node = SplatizeNode { min_splats: 10, max_splats: 50, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        for s in splats {
            let any_nonzero = s.spectral().iter().any(|&v| v != 0);
            assert!(any_nonzero, "splat spectral should be non-zero from material assignment");
        }
    }

    #[test]
    fn foliage_material_has_green_bias() {
        let spectral = spectral_from_material(2);
        let green_sum = half::f16::from_bits(spectral[6]).to_f32()
                      + half::f16::from_bits(spectral[7]).to_f32()
                      + half::f16::from_bits(spectral[8]).to_f32();
        let red_sum   = half::f16::from_bits(spectral[12]).to_f32()
                      + half::f16::from_bits(spectral[13]).to_f32();
        println!("foliage green_sum={} red_sum={}", green_sum, red_sum);
        assert!(green_sum > red_sum, "foliage should have green-band bias, got green={} red={}", green_sum, red_sum);
    }

    #[test]
    fn brick_material_has_red_bias() {
        let spectral = spectral_from_material(5);
        let red_sum = half::f16::from_bits(spectral[10]).to_f32()
                    + half::f16::from_bits(spectral[11]).to_f32()
                    + half::f16::from_bits(spectral[12]).to_f32();
        let uv_sum  = half::f16::from_bits(spectral[0]).to_f32()
                    + half::f16::from_bits(spectral[1]).to_f32();
        assert!(red_sum > uv_sum * 2.0, "brick should have strong red bias, red={} uv={}", red_sum, uv_sum);
    }

    #[test]
    fn splatize_node_in_graph_end_to_end() {
        use crate::node_graph::OchromaNodeGraph;
        use crate::nodes::building_node::BuildingNode;
        let mut graph = OchromaNodeGraph::new();
        let building_id = graph.add_node("building", Box::new(BuildingNode { grid_w: 3, grid_h: 2, grid_d: 3, ..Default::default() }));
        let splat_id = graph.add_node("splatize", Box::new(SplatizeNode { min_splats: 10, max_splats: 500, ..Default::default() }));
        graph.connect(building_id, "mesh", splat_id, "mesh").unwrap();
        graph.cook().unwrap();
        let splats = graph.get_output(splat_id, "splats").unwrap().as_splats().unwrap();
        println!("splats.len() = {}", splats.len());
        assert!(!splats.is_empty(), "splatize should produce splats from building mesh");
    }
}
