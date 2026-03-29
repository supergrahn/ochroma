use bevy_ecs::prelude::*;
use glam::Vec3;
use vox_terrain::brushes::{BrushType, TerrainBrush};
use vox_terrain::volume::TerrainVolume;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveBrush {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Paint,
    Erode,
}

pub struct TerrainEditorState {
    pub active_brush: ActiveBrush,
    pub brush: TerrainBrush,
    pub flatten_height: f32,
    pub paint_material: u8,
    pub foliage_scatter_pending: bool,
    pub foliage_density: f32,
    pub is_open: bool,
}

impl Default for TerrainEditorState {
    fn default() -> Self {
        Self {
            active_brush: ActiveBrush::Raise,
            brush: TerrainBrush::new(BrushType::Raise, 5.0, 0.5),
            flatten_height: 0.0,
            paint_material: 0,
            foliage_scatter_pending: false,
            foliage_density: 0.5,
            is_open: false,
        }
    }
}

impl TerrainEditorState {
    pub fn set_brush_type(&mut self, bt: BrushType) {
        self.brush.brush_type = bt;
    }

    /// Sync `brush.brush_type` from `active_brush` enum.
    pub fn sync_brush(&mut self) {
        self.brush.brush_type = match self.active_brush {
            ActiveBrush::Raise   => BrushType::Raise,
            ActiveBrush::Lower   => BrushType::Lower,
            ActiveBrush::Smooth  => BrushType::Smooth,
            ActiveBrush::Flatten => BrushType::Flatten { target_height: self.flatten_height },
            ActiveBrush::Paint   => BrushType::Paint { material: self.paint_material },
            ActiveBrush::Erode   => BrushType::Erode,
        };
    }
}

/// Apply a single brush stroke to the TerrainVolume resource.
pub fn apply_brush_stroke(
    world: &mut World,
    center: Vec3,
    brush_type: BrushType,
    radius: f32,
    strength: f32,
    dt: f32,
) {
    if let Some(mut vol) = world.get_resource_mut::<TerrainVolume>() {
        let brush = TerrainBrush::new(brush_type, radius, strength);
        brush.apply(&mut *vol, center, dt);
    }
}
