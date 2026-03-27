use glam::Vec3;
use rand::prelude::*;
use rand::SeedableRng;

/// A node in the road graph (intersection or endpoint).
#[derive(Debug, Clone)]
pub struct RoadNode {
    pub id: u32,
    pub position: Vec3,
}

/// An edge in the road graph (road segment).
#[derive(Debug, Clone)]
pub struct RoadEdge {
    pub from: u32,
    pub to: u32,
    pub width: f32,
    pub road_type: RoadType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadType {
    MainStreet,
    SideStreet,
    Alley,
}

/// A generated road graph for a district.
#[derive(Debug, Clone)]
pub struct RoadGraph {
    pub nodes: Vec<RoadNode>,
    pub edges: Vec<RoadEdge>,
}

/// A building plot generated along road edges.
#[derive(Debug, Clone)]
pub struct BuildingPlot {
    pub position: Vec3,
    pub width: f32,
    pub depth: f32,
    pub facing_road_edge: u32, // index into edges
    pub facing_angle: f32,     // radians, facing the road
}

/// Generate a road graph for a district.
/// Creates a main street with perpendicular side streets.
pub fn generate_road_graph(
    street_length: f32,
    street_width: f32,
    side_street_spacing: f32,
    seed: u64,
) -> RoadGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut next_id = 0u32;

    // Main street along X axis
    let main_start_id = next_id;
    nodes.push(RoadNode { id: next_id, position: Vec3::new(0.0, 0.0, 0.0) });
    next_id += 1;

    let main_end_id = next_id;
    nodes.push(RoadNode { id: next_id, position: Vec3::new(street_length, 0.0, 0.0) });
    next_id += 1;

    edges.push(RoadEdge {
        from: main_start_id,
        to: main_end_id,
        width: street_width,
        road_type: RoadType::MainStreet,
    });

    // Side streets perpendicular to main street
    let num_side_streets = (street_length / side_street_spacing).floor() as i32;
    let side_street_length = 30.0 + rng.random::<f32>() * 20.0;

    for i in 1..num_side_streets {
        let x = i as f32 * side_street_spacing + rng.random::<f32>() * 5.0 - 2.5;
        if x <= 0.0 || x >= street_length {
            continue;
        }

        // Intersection node on main street
        let intersection_id = next_id;
        nodes.push(RoadNode { id: next_id, position: Vec3::new(x, 0.0, 0.0) });
        next_id += 1;

        // Side street going positive Z
        let side_end_id = next_id;
        let side_len = side_street_length + rng.random::<f32>() * 10.0;
        nodes.push(RoadNode { id: next_id, position: Vec3::new(x, 0.0, side_len) });
        next_id += 1;

        edges.push(RoadEdge {
            from: intersection_id,
            to: side_end_id,
            width: street_width * 0.7,
            road_type: RoadType::SideStreet,
        });

        // Also go negative Z sometimes
        if rng.random::<f32>() > 0.3 {
            let side_neg_id = next_id;
            let neg_len = side_street_length + rng.random::<f32>() * 10.0;
            nodes.push(RoadNode { id: next_id, position: Vec3::new(x, 0.0, -neg_len) });
            next_id += 1;

            edges.push(RoadEdge {
                from: intersection_id,
                to: side_neg_id,
                width: street_width * 0.7,
                road_type: RoadType::SideStreet,
            });
        }
    }

    RoadGraph { nodes, edges }
}

/// Generate building plots along the edges of a road graph.
pub fn generate_building_plots(graph: &RoadGraph, plot_width: f32, plot_depth: f32) -> Vec<BuildingPlot> {
    let mut plots = Vec::new();

    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        let from = graph.nodes.iter().find(|n| n.id == edge.from).unwrap();
        let to = graph.nodes.iter().find(|n| n.id == edge.to).unwrap();

        let dir = (to.position - from.position).normalize();
        let perp = Vec3::new(-dir.z, 0.0, dir.x); // perpendicular on xz plane
        let length = (to.position - from.position).length();
        let facing_angle = dir.z.atan2(dir.x);

        let num_plots = (length / plot_width).floor() as i32;
        let offset = edge.width * 0.5 + plot_depth * 0.5 + 1.0; // setback from road

        for i in 0..num_plots {
            let t = (i as f32 + 0.5) * plot_width;
            let base = from.position + dir * t;

            // Plot on positive side
            plots.push(BuildingPlot {
                position: base + perp * offset,
                width: plot_width,
                depth: plot_depth,
                facing_road_edge: edge_idx as u32,
                facing_angle,
            });

            // Plot on negative side
            plots.push(BuildingPlot {
                position: base - perp * offset,
                width: plot_width,
                depth: plot_depth,
                facing_road_edge: edge_idx as u32,
                facing_angle: facing_angle + std::f32::consts::PI,
            });
        }
    }

    plots
}
