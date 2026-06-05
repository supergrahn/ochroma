// stub — implemented below
use crate::node_graph::{NodeDescriptor, NodeError, NodeInputs, NodeOutputs, OchromaNode, ParamValue, PortData, PortSpec, PortType, EditorMesh};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;

#[derive(Clone)]
pub struct VegetationNode {
    pub seed:          u64,
    pub branch_levels: u32,
    pub trunk_radius:  f32,
    pub height:        f32,
}

impl Default for VegetationNode {
    fn default() -> Self {
        Self { seed: 0, branch_levels: 4, trunk_radius: 0.3, height: 8.0 }
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt().max(1e-8);
    [v[0]/len, v[1]/len, v[2]/len]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
fn scale3(v: [f32; 3], s: f32) -> [f32; 3] { [v[0]*s, v[1]*s, v[2]*s] }

fn rotate_around_axis(v: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let dot = v[0]*axis[0]+v[1]*axis[1]+v[2]*axis[2];
    let cr = cross3(axis, v);
    [
        v[0]*cos_a + cr[0]*sin_a + axis[0]*dot*(1.0-cos_a),
        v[1]*cos_a + cr[1]*sin_a + axis[1]*dot*(1.0-cos_a),
        v[2]*cos_a + cr[2]*sin_a + axis[2]*dot*(1.0-cos_a),
    ]
}

fn emit_cylinder(mesh: &mut EditorMesh, base: [f32;3], top: [f32;3], radius: f32) {
    let sides = 6usize;
    let base_idx = mesh.positions.len() as u32;
    let dir = norm3([top[0]-base[0], top[1]-base[1], top[2]-base[2]]);
    // Build perpendicular
    let perp = if dir[1].abs() < 0.9 { norm3(cross3(dir, [0.0,1.0,0.0])) } else { norm3(cross3(dir, [1.0,0.0,0.0])) };
    for ring in [&base, &top] {
        for i in 0..sides {
            let angle = i as f32 * std::f32::consts::TAU / sides as f32;
            let tangent = rotate_around_axis(perp, dir, angle);
            let pos = add3(*ring, scale3(tangent, radius));
            mesh.positions.push(pos);
            mesh.normals.push(norm3(tangent));
        }
    }
    // 12 triangles for 6 sides
    for i in 0..sides as u32 {
        let next = (i + 1) % sides as u32;
        // bottom ring = base_idx..base_idx+sides; top ring = base_idx+sides..
        let b0 = base_idx + i;
        let b1 = base_idx + next;
        let t0 = base_idx + sides as u32 + i;
        let t1 = base_idx + sides as u32 + next;
        mesh.indices.push([b0, b1, t0]);
        mesh.indices.push([b1, t1, t0]);
    }
}

fn emit_leaf_quad(mesh: &mut EditorMesh, pos: [f32;3], size: f32) {
    let base_idx = mesh.positions.len() as u32;
    let h = size * 0.5;
    mesh.positions.extend_from_slice(&[
        [pos[0]-h, pos[1],   pos[2]-h],
        [pos[0]+h, pos[1],   pos[2]-h],
        [pos[0]+h, pos[1],   pos[2]+h],
        [pos[0]-h, pos[1],   pos[2]+h],
    ]);
    mesh.normals.extend_from_slice(&[[0.0,1.0,0.0]; 4]);
    mesh.indices.push([base_idx, base_idx+1, base_idx+2]);
    mesh.indices.push([base_idx, base_idx+2, base_idx+3]);
}

fn grow_segment(
    mesh: &mut EditorMesh,
    origin: [f32;3],
    direction: [f32;3],
    length: f32,
    radius: f32,
    levels_left: u32,
    rng: &mut Pcg64,
) {
    if mesh.positions.len() >= 20_000 { return; }
    let top = add3(origin, scale3(norm3(direction), length));
    emit_cylinder(mesh, origin, top, radius);

    if levels_left == 0 {
        emit_leaf_quad(mesh, top, radius * 4.0);
        return;
    }

    // Two child branches
    let branch_count = 2;
    for _ in 0..branch_count {
        // Random deviation angle
        let deviation_angle = 0.4 + rng.gen::<f32>() * 0.4;
        let rot_axis = norm3([rng.gen::<f32>()-0.5, rng.gen::<f32>()*0.1, rng.gen::<f32>()-0.5]);
        let new_dir = rotate_around_axis(direction, rot_axis, deviation_angle);
        grow_segment(
            mesh, top, new_dir,
            length * 0.7,
            radius * 0.65,
            levels_left - 1,
            rng,
        );
    }
}

fn build_tree(seed: u64, branch_levels: u32, height: f32, trunk_radius: f32) -> EditorMesh {
    let mut mesh = EditorMesh::new();
    mesh.material_id = 2; // foliage
    let mut rng = Pcg64::seed_from_u64(seed);
    grow_segment(&mut mesh, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], height * 0.5, trunk_radius, branch_levels, &mut rng);
    mesh
}

fn decimate_mesh(mesh: &EditorMesh, stride: usize) -> EditorMesh {
    if stride <= 1 || mesh.indices.is_empty() {
        return mesh.clone();
    }
    let mut out = EditorMesh::new();
    out.material_id = mesh.material_id;
    // Re-index: keep every stride-th triangle
    let kept_tris: Vec<[u32; 3]> = mesh.indices.iter().step_by(stride).cloned().collect();
    // Collect used vertex indices
    let mut used: Vec<u32> = kept_tris.iter().flat_map(|t| t.iter().cloned()).collect();
    used.sort_unstable();
    used.dedup();

    let mut remap = vec![0u32; mesh.positions.len()];
    for (new_idx, &old_idx) in used.iter().enumerate() {
        remap[old_idx as usize] = new_idx as u32;
        out.positions.push(mesh.positions[old_idx as usize]);
        out.normals.push(mesh.normals[old_idx as usize]);
    }
    for tri in &kept_tris {
        out.indices.push([remap[tri[0] as usize], remap[tri[1] as usize], remap[tri[2] as usize]]);
    }
    out
}

fn billboard_from_mesh(mesh: &EditorMesh) -> EditorMesh {
    // Compute AABB
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in &mesh.positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    // Clamp to something sensible
    if min[0].is_infinite() { min = [-1.0, 0.0, -1.0]; max = [1.0, 2.0, 1.0]; }

    let cx = (min[0] + max[0]) * 0.5;
    let w  = max[0] - min[0];
    let h  = max[1] - min[1];

    let mut out = EditorMesh::new();
    out.material_id = mesh.material_id;
    // 4 vertices, 2 triangles
    out.positions = vec![
        [cx - w*0.5, min[1], min[2]],
        [cx + w*0.5, min[1], min[2]],
        [cx + w*0.5, min[1] + h, min[2]],
        [cx - w*0.5, min[1] + h, min[2]],
    ];
    out.normals = vec![[0.0, 0.0, 1.0]; 4];
    out.indices = vec![[0, 1, 2], [0, 2, 3]];
    out
}

fn build_lod_set(full: &EditorMesh) -> Vec<EditorMesh> {
    vec![
        full.clone(),               // LOD0 = full
        decimate_mesh(full, 2),     // LOD1 = 50%
        decimate_mesh(full, 5),     // LOD2 = 20%
        billboard_from_mesh(full),  // LOD3 = billboard (exactly 2 triangles)
    ]
}

impl OchromaNode for VegetationNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "VegetationNode",
            inputs: vec![],
            outputs: vec![
                PortSpec { name: "mesh",     port_type: PortType::Mesh,    optional: false },
                PortSpec { name: "lod_mesh", port_type: PortType::LodMesh, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("seed",          ParamValue::Int(v))   => { self.seed          = v as u64; Ok(()) }
            ("branch_levels", ParamValue::Int(v))   => { self.branch_levels = v as u32; Ok(()) }
            ("trunk_radius",  ParamValue::Float(v)) => { self.trunk_radius  = v as f32; Ok(()) }
            ("height",        ParamValue::Float(v)) => { self.height        = v as f32; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(self.clone()) }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let mesh = build_tree(self.seed, self.branch_levels, self.height, self.trunk_radius);
        let lods = build_lod_set(&mesh);
        let mut out = NodeOutputs::new();
        out.insert("mesh".into(),     PortData::Mesh(mesh));
        out.insert("lod_mesh".into(), PortData::LodMesh(lods));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_node() -> VegetationNode { VegetationNode::default() }

    #[test]
    fn vegetation_node_cooks_without_error() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        assert!(out.contains_key("mesh"));
        assert!(out.contains_key("lod_mesh"));
    }

    #[test]
    fn lod_set_has_four_levels() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let lods = out["lod_mesh"].as_lod_mesh().unwrap();
        assert_eq!(lods.len(), 4, "LOD0..LOD3 expected");
    }

    #[test]
    fn lod0_has_more_triangles_than_lod2() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let lods = out["lod_mesh"].as_lod_mesh().unwrap();
        let tri0 = lods[0].indices.len();
        let tri2 = lods[2].indices.len();
        assert!(tri0 > tri2, "LOD0 ({} tris) should have more than LOD2 ({} tris)", tri0, tri2);
    }

    #[test]
    fn lod3_billboard_has_exactly_two_triangles() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let lods = out["lod_mesh"].as_lod_mesh().unwrap();
        assert_eq!(lods[3].indices.len(), 2, "LOD3 billboard must be exactly 2 triangles");
    }

    #[test]
    fn mesh_stays_within_vertex_budget() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let mesh = out["mesh"].as_mesh().unwrap();
        assert!(mesh.positions.len() <= 20_000, "vertex budget guard: {} > 20_000", mesh.positions.len());
    }

    #[test]
    fn different_seeds_produce_different_trees() {
        let a = VegetationNode { seed: 1, ..Default::default() }.cook(NodeInputs::new()).unwrap();
        let b = VegetationNode { seed: 2, ..Default::default() }.cook(NodeInputs::new()).unwrap();
        let pa = &a["mesh"].as_mesh().unwrap().positions;
        let pb = &b["mesh"].as_mesh().unwrap().positions;
        assert!(!pa.is_empty() && !pb.is_empty());
    }

    #[test]
    fn vegetation_node_in_graph() {
        use crate::node_graph::OchromaNodeGraph;
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("tree", Box::new(default_node()));
        graph.cook().unwrap();
        assert!(graph.get_output(id, "lod_mesh").is_some());
    }
}
