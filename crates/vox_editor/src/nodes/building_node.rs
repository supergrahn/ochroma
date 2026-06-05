// stub — implemented in Task 3
use crate::node_graph::{NodeDescriptor, NodeError, NodeInputs, NodeOutputs, OchromaNode, ParamValue, PortData, PortSpec, PortType, EditorMesh};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum BuildingStyle { Plain, Modern, Gothic }

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Program { Residential, Agricultural, Civic, Religious, Commercial, Industrial, Utility }

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Setting { Urban, Suburban, Rural, Industrial, Waterfront, HistoricalOldTown }

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum BuildingCondition { New, Aged, Weathered, Derelict }

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BuildingDescription {
    pub program:       Program,
    pub setting:       Setting,
    pub style_key:     String,
    pub era:           String,
    pub condition:     BuildingCondition,
    pub floors:        u32,
    pub floor_height:  f32,
    pub seed:          u64,
    pub detail_atoms:  Option<Vec<String>>,
    pub organic_atoms: Option<Vec<String>>,
}

impl Default for BuildingDescription {
    fn default() -> Self {
        Self {
            program: Program::Residential,
            setting: Setting::Suburban,
            style_key: "plain".into(),
            era: "modern".into(),
            condition: BuildingCondition::New,
            floors: 2,
            floor_height: 3.0,
            seed: 0,
            detail_atoms: None,
            organic_atoms: None,
        }
    }
}

impl BuildingDescription {
    pub fn to_building_params(&self) -> BuildingStyle {
        if self.style_key.starts_with("gothic") { BuildingStyle::Gothic }
        else if self.style_key.starts_with("modern") { BuildingStyle::Modern }
        else { BuildingStyle::Plain }
    }
}

#[derive(Clone)]
pub struct BuildingNode {
    pub grid_w:      usize,
    pub grid_h:      usize,
    pub grid_d:      usize,
    pub seed:        u64,
    pub style:       BuildingStyle,
    pub max_attempts: u32,
}

impl Default for BuildingNode {
    fn default() -> Self {
        Self { grid_w: 5, grid_h: 3, grid_d: 5, seed: 0, style: BuildingStyle::Plain, max_attempts: 10 }
    }
}

// Tile type bitmask values
const TILE_EMPTY:  u8 = 0b00000001;
const TILE_WALL:   u8 = 0b00000010;
const TILE_WINDOW: u8 = 0b00000100;
const TILE_CORNER: u8 = 0b00001000;
const TILE_DOOR:   u8 = 0b00010000;
const ALL_TILES:   u8 = TILE_EMPTY | TILE_WALL | TILE_WINDOW | TILE_CORNER | TILE_DOOR;

fn try_solve(grid_w: usize, grid_h: usize, grid_d: usize, seed: u64, attempt: u32) -> Option<Vec<u8>> {
    let mut rng = Pcg64::seed_from_u64(seed ^ (attempt as u64).wrapping_mul(0x9e3779b9));
    let n = grid_w * grid_h * grid_d;
    let mut superposition = vec![ALL_TILES; n];

    // Collapse in random order
    let mut order: Vec<usize> = (0..n).collect();
    // Simple shuffle
    for i in (1..n).rev() {
        let j = rng.gen::<usize>() % (i + 1);
        order.swap(i, j);
    }

    for &idx in &order {
        if superposition[idx].count_ones() == 1 { continue; }
        let mask = superposition[idx];
        // Pick a tile from the available options
        let mut choices = Vec::new();
        for bit in [TILE_EMPTY, TILE_WALL, TILE_WINDOW, TILE_CORNER, TILE_DOOR] {
            if mask & bit != 0 { choices.push(bit); }
        }
        if choices.is_empty() { return None; }
        let chosen = choices[rng.gen::<usize>() % choices.len()];
        superposition[idx] = chosen;
        // Propagate: BFS
        propagate(&mut superposition, idx, grid_w, grid_h, grid_d);
    }

    // Verify no zero-entropy cells
    if superposition.contains(&0) { return None; }
    Some(superposition)
}

fn propagate(sup: &mut [u8], start: usize, grid_w: usize, grid_h: usize, grid_d: usize) {
    use std::collections::VecDeque;
    let mut queue = VecDeque::new();
    queue.push_back(start);
    while let Some(idx) = queue.pop_front() {
        let x = (idx % (grid_w * grid_h)) % grid_w;
        let y = (idx % (grid_w * grid_h)) / grid_w;
        let z = idx / (grid_w * grid_h);

        // Neighbors
        let neighbors = [
            if x > 0 { Some(z * grid_w * grid_h + y * grid_w + x - 1) } else { None },
            if x + 1 < grid_w { Some(z * grid_w * grid_h + y * grid_w + x + 1) } else { None },
            if y > 0 { Some(z * grid_w * grid_h + (y-1) * grid_w + x) } else { None },
            if y + 1 < grid_h { Some(z * grid_w * grid_h + (y+1) * grid_w + x) } else { None },
            if z > 0 { Some((z-1) * grid_w * grid_h + y * grid_w + x) } else { None },
            if z + 1 < grid_d { Some((z+1) * grid_w * grid_h + y * grid_w + x) } else { None },
        ];

        for maybe_nb in &neighbors {
            let Some(nb) = *maybe_nb else { continue };
            let old = sup[nb];
            // Adjacency constraint: every tile (empty or wall) may currently neighbour any
            // tile, so the allowed mask is ALL_TILES.
            let allowed = ALL_TILES;
            let new_mask = old & allowed;
            if new_mask != old && new_mask != 0 {
                sup[nb] = new_mask;
                queue.push_back(nb);
            }
        }
    }
}

fn solve_wfc(grid_w: usize, grid_h: usize, grid_d: usize, seed: u64, max_attempts: u32) -> Vec<u8> {
    for attempt in 0..max_attempts {
        if let Some(grid) = try_solve(grid_w, grid_h, grid_d, seed, attempt) {
            return grid;
        }
    }
    // Fallback: all walls
    vec![TILE_WALL; grid_w * grid_h * grid_d]
}

fn emit_box(mesh: &mut EditorMesh, x: f32, y: f32, z: f32, w: f32, h: f32, d: f32) {
    let base = mesh.positions.len() as u32;
    // 8 vertices of the box
    let verts = [
        [x,   y,   z  ], [x+w, y,   z  ], [x+w, y+h, z  ], [x,   y+h, z  ], // front
        [x,   y,   z+d], [x+w, y,   z+d], [x+w, y+h, z+d], [x,   y+h, z+d], // back
    ];
    for v in &verts { mesh.positions.push(*v); mesh.normals.push([0.0, 1.0, 0.0]); }
    // 12 triangles (6 faces, 2 tris each)
    let faces: [[u32; 3]; 12] = [
        [0,1,2],[0,2,3], // front
        [4,6,5],[4,7,6], // back
        [0,4,5],[0,5,1], // bottom
        [3,2,6],[3,6,7], // top
        [0,3,7],[0,7,4], // left
        [1,5,6],[1,6,2], // right
    ];
    for f in &faces { mesh.indices.push([base+f[0], base+f[1], base+f[2]]); }
}

fn tiles_to_mesh(grid: &[u8], grid_w: usize, grid_h: usize, grid_d: usize) -> EditorMesh {
    let mut mesh = EditorMesh::new();
    for z in 0..grid_d {
        for y in 0..grid_h {
            for x in 0..grid_w {
                let tile = grid[z * grid_w * grid_h + y * grid_w + x];
                if tile == TILE_EMPTY { continue; }
                let mat_id = match tile {
                    TILE_WALL   => 0,
                    TILE_WINDOW => 6, // glass
                    TILE_CORNER => 0,
                    TILE_DOOR   => 3, // stone
                    _           => 0,
                };
                let fx = x as f32 * 2.0;
                let fy = y as f32 * 2.5;
                let fz = z as f32 * 2.0;
                let h = if tile == TILE_DOOR { 2.5 } else { 2.0 };
                let mut seg = EditorMesh::new();
                seg.material_id = mat_id;
                emit_box(&mut seg, fx, fy, fz, 2.0, h, 2.0);
                let base = mesh.positions.len() as u32;
                for p in seg.positions { mesh.positions.push(p); }
                for n in seg.normals   { mesh.normals.push(n); }
                for tri in seg.indices { mesh.indices.push([base+tri[0], base+tri[1], base+tri[2]]); }
            }
        }
    }
    mesh
}

impl OchromaNode for BuildingNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "BuildingNode",
            inputs: vec![],
            outputs: vec![PortSpec { name: "mesh", port_type: PortType::Mesh, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match (key, value) {
            ("grid_w",       ParamValue::Int(v)) => { self.grid_w = v as usize; Ok(()) }
            ("grid_h",       ParamValue::Int(v)) => { self.grid_h = v as usize; Ok(()) }
            ("grid_d",       ParamValue::Int(v)) => { self.grid_d = v as usize; Ok(()) }
            ("seed",         ParamValue::Int(v)) => { self.seed = v as u64; Ok(()) }
            ("max_attempts", ParamValue::Int(v)) => { self.max_attempts = v as u32; Ok(()) }
            (k, _) => Err(NodeError::UnknownParam(k.into())),
        }
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(self.clone()) }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let grid = solve_wfc(self.grid_w, self.grid_h, self.grid_d, self.seed, self.max_attempts);
        let mesh = tiles_to_mesh(&grid, self.grid_w, self.grid_h, self.grid_d);
        let mut out = NodeOutputs::new();
        out.insert("mesh".into(), PortData::Mesh(mesh));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> BuildingNode {
        BuildingNode { grid_w: 3, grid_h: 2, grid_d: 3, ..Default::default() }
    }

    #[test]
    fn building_node_cooks_without_error() {
        let out = small().cook(NodeInputs::new()).unwrap();
        assert!(out.contains_key("mesh"));
    }

    #[test]
    fn mesh_has_vertices() {
        let out = small().cook(NodeInputs::new()).unwrap();
        let mesh = out["mesh"].as_mesh().unwrap();
        assert!(!mesh.positions.is_empty(), "mesh should have vertices");
    }

    #[test]
    fn wfc_is_deterministic() {
        let a = small().cook(NodeInputs::new()).unwrap();
        let b = small().cook(NodeInputs::new()).unwrap();
        let pa = &a["mesh"].as_mesh().unwrap().positions;
        let pb = &b["mesh"].as_mesh().unwrap().positions;
        assert_eq!(pa.len(), pb.len(), "same seed → same vertex count");
    }

    #[test]
    fn different_seeds_may_produce_different_buildings() {
        let a = BuildingNode { seed: 1, ..small() }.cook(NodeInputs::new()).unwrap();
        let b = BuildingNode { seed: 2, ..small() }.cook(NodeInputs::new()).unwrap();
        assert!(a.contains_key("mesh"));
        assert!(b.contains_key("mesh"));
    }

    #[test]
    fn building_node_in_graph() {
        use crate::node_graph::OchromaNodeGraph;
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("building", Box::new(small()));
        graph.cook().unwrap();
        assert!(graph.get_output(id, "mesh").is_some());
    }

    #[test]
    fn test_building_description_json_roundtrip() {
        let desc = BuildingDescription {
            program: Program::Residential,
            setting: Setting::Suburban,
            style_key: "craftsman".into(),
            era: "1920s".into(),
            condition: BuildingCondition::Aged,
            floors: 2,
            floor_height: 3.0,
            seed: 42,
            detail_atoms: Some(vec!["exposed_rafter_tails".into(), "tapered_columns".into()]),
            organic_atoms: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&desc).unwrap();
        let restored: BuildingDescription = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.style_key, "craftsman");
        assert_eq!(restored.detail_atoms.unwrap().len(), 2);
    }
}
