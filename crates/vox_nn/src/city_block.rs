use half::f16;
use rand::prelude::*;
use rand::SeedableRng;
use vox_core::types::GaussianSplat;
use vox_data::proc_gs_advanced::*;

use crate::street_layout::{generate_building_plots, generate_road_graph, RoadGraph};

/// A generated city block with all assets.
pub struct CityBlock {
    pub road_splats: Vec<GaussianSplat>,
    pub building_splats: Vec<(glam::Vec3, Vec<GaussianSplat>)>,
    pub tree_splats: Vec<(glam::Vec3, Vec<GaussianSplat>)>,
    pub prop_splats: Vec<(glam::Vec3, Vec<GaussianSplat>)>,
    pub terrain_splats: Vec<GaussianSplat>,
}

/// Generate a complete city block from a seed.
pub fn generate_city_block(seed: u64, size: f32) -> CityBlock {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate road network
    let graph = generate_road_graph(size, 8.0, 25.0, seed);
    let plots = generate_building_plots(&graph, 6.0, 12.0);

    // Generate road surface splats
    let road_splats = generate_road_surface(&graph);

    // Generate buildings on plots
    let building_splats: Vec<(glam::Vec3, Vec<GaussianSplat>)> = plots
        .iter()
        .enumerate()
        .map(|(i, plot)| {
            let building_seed = seed.wrapping_add(i as u64 * 1000);
            let splats =
                vox_data::proc_gs::emit_splats_simple(building_seed, plot.width, plot.depth);
            (glam::Vec3::new(plot.position.x, 0.0, plot.position.z), splats)
        })
        .collect();

    // Place trees along roads
    let tree_splats: Vec<(glam::Vec3, Vec<GaussianSplat>)> = (0..graph.nodes.len())
        .filter_map(|i| {
            if rng.random::<f32>() > 0.3 {
                return None;
            } // 30% chance of tree at each node
            let node = &graph.nodes[i];
            let tree_seed = seed.wrapping_add(10000 + i as u64);
            let height = 5.0 + rng.random::<f32>() * 5.0;
            let canopy = 2.0 + rng.random::<f32>() * 2.0;
            let offset = glam::Vec3::new(
                node.position.x + (rng.random::<f32>() - 0.5) * 5.0,
                0.0,
                node.position.z + (rng.random::<f32>() - 0.5) * 5.0,
            );
            Some((offset, generate_tree(tree_seed, height, canopy)))
        })
        .collect();

    // Place benches and lamps
    let prop_splats: Vec<(glam::Vec3, Vec<GaussianSplat>)> = graph
        .edges
        .iter()
        .enumerate()
        .flat_map(|(i, _edge)| {
            let mut props = Vec::new();
            // Lamp every ~30m
            if i % 2 == 0 {
                let node = &graph.nodes[i.min(graph.nodes.len() - 1)];
                props.push((
                    glam::Vec3::new(node.position.x + 3.0, 0.0, node.position.z),
                    generate_lamp_post(seed.wrapping_add(20000 + i as u64), 4.5),
                ));
            }
            // Bench occasionally
            if rng.random::<f32>() > 0.7 {
                let node = &graph.nodes[i.min(graph.nodes.len() - 1)];
                props.push((
                    glam::Vec3::new(node.position.x - 3.0, 0.0, node.position.z),
                    generate_bench(seed.wrapping_add(30000 + i as u64)),
                ));
            }
            props
        })
        .collect();

    // Terrain: grass patch covering the block
    let terrain_splats = generate_grass_patch(seed.wrapping_add(50000), size, 2.0);

    CityBlock {
        road_splats,
        building_splats,
        tree_splats,
        prop_splats,
        terrain_splats,
    }
}

/// Generate road surface splats from a road graph.
fn generate_road_surface(graph: &RoadGraph) -> Vec<GaussianSplat> {
    let asphalt_spd: [u16; 8] = std::array::from_fn(|_| f16::from_f32(0.05).to_bits());
    let mut splats = Vec::new();

    for edge in &graph.edges {
        let from = graph.nodes.iter().find(|n| n.id == edge.from).unwrap();
        let to = graph.nodes.iter().find(|n| n.id == edge.to).unwrap();
        let dir = to.position - from.position;
        let length = dir.length();
        if length < 0.01 {
            continue;
        }
        let dir = dir / length;
        let perp = glam::Vec3::new(-dir.z, 0.0, dir.x);
        let half_width = edge.width * 0.5;

        let steps = (length / 0.5).ceil() as usize;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let center = from.position + dir * (t * length);
            let width_steps = (half_width * 2.0 / 0.5).ceil() as i32;
            for w in 0..width_steps {
                let offset = (w as f32 / width_steps as f32 - 0.5) * half_width * 2.0;
                let pos = center + perp * offset;
                splats.push(GaussianSplat {
                    position: [pos.x, 0.02, pos.z],
                    scale: [0.25, 0.01, 0.25],
                    rotation: [0, 0, 0, 32767],
                    opacity: 240,
                    _pad: [0; 3],
                    spectral: asphalt_spd,
                });
            }
        }
    }
    splats
}
