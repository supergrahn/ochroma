use vox_core::lwc::{TileCoord, TILE_SIZE};

use crate::scene_graph::SceneGraph;

/// Expand a scene graph across a multi-tile area.
/// Phase 2: up to 2×2 tiles (4km × 4km).
pub fn expand_to_multi_tile(
    base_scene: &SceneGraph,
    tiles_x: u32,
    tiles_z: u32,
) -> Vec<(TileCoord, SceneGraph)> {
    let mut result = Vec::new();

    for tx in 0..tiles_x {
        for tz in 0..tiles_z {
            let tile = TileCoord { x: tx as i32, z: tz as i32 };
            let offset_x = tx as f32 * TILE_SIZE as f32;
            let offset_z = tz as f32 * TILE_SIZE as f32;

            let mut tile_scene = base_scene.clone();

            // Offset all positions
            for building in &mut tile_scene.street.buildings {
                building.position[0] += offset_x;
                building.position[2] += offset_z;
            }
            for prop in &mut tile_scene.street.props {
                prop.position[0] += offset_x;
                prop.position[2] += offset_z;
            }
            for veg in &mut tile_scene.street.vegetation {
                veg.position[0] += offset_x;
                veg.position[2] += offset_z;
            }

            result.push((tile, tile_scene));
        }
    }

    result
}
