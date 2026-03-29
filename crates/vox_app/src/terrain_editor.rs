use bevy_ecs::prelude::*;
use egui;
use glam::{Quat, Vec3};
use uuid::Uuid;
use vox_core::ecs::{LodLevel, SplatAssetComponent, SplatInstanceComponent};
use vox_terrain::brushes::{BrushType, TerrainBrush};
use vox_terrain::foliage::{scatter_foliage, FoliageRule};
use vox_terrain::heightmap::Heightmap;
use vox_terrain::volume::{default_volume_materials, volume_to_splats, TerrainVolume};

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

/// Regenerate terrain splats from the current TerrainVolume and update the ECS asset.
pub fn resplat_terrain(world: &mut World, terrain_entity: Entity) {
    let splats = {
        let vol = world.resource::<TerrainVolume>();
        let materials = default_volume_materials();
        volume_to_splats(vol, &materials, 0)
    };
    let splat_count = splats.len() as u32;
    if let Some(mut asset) = world.entity_mut(terrain_entity).get_mut::<SplatAssetComponent>() {
        asset.splats = splats;
        asset.splat_count = splat_count;
    }
}

/// Show the terrain editor egui panel.
pub fn show_terrain_editor_panel(ui: &mut egui::Ui, state: &mut TerrainEditorState) {
    ui.heading("Brush");
    let brush_options: &[(&str, ActiveBrush)] = &[
        ("Raise",   ActiveBrush::Raise),
        ("Lower",   ActiveBrush::Lower),
        ("Smooth",  ActiveBrush::Smooth),
        ("Flatten", ActiveBrush::Flatten),
        ("Paint",   ActiveBrush::Paint),
        ("Erode",   ActiveBrush::Erode),
    ];
    ui.horizontal(|ui| {
        for (label, variant) in brush_options {
            if ui.selectable_label(state.active_brush == *variant, *label).clicked() {
                state.active_brush = *variant;
                state.sync_brush();
            }
        }
    });
    ui.add(egui::Slider::new(&mut state.brush.radius, 0.5..=50.0).text("Radius"));
    ui.add(egui::Slider::new(&mut state.brush.strength, 0.0..=2.0).text("Strength"));
    if state.active_brush == ActiveBrush::Flatten {
        if ui.add(egui::Slider::new(&mut state.flatten_height, -10.0..=50.0).text("Height")).changed() {
            state.sync_brush();
        }
    }
    ui.separator();
    ui.heading("Foliage");
    ui.add(egui::Slider::new(&mut state.foliage_density, 0.0..=1.0).text("Density"));
    if ui.button("Scatter Foliage").clicked() {
        state.foliage_scatter_pending = true;
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

static FOLIAGE_INSTANCE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(5000);

/// Build a `Heightmap` from a `TerrainVolume` by scanning each XZ column for
/// the topmost solid voxel (SDF <= 0).
fn heightmap_from_volume(vol: &TerrainVolume) -> Heightmap {
    let w = vol.size_x;
    let d = vol.size_z;
    let mut data = vec![vol.origin[1]; w * d];

    for z in 0..d {
        for x in 0..w {
            // Scan downward from top
            let mut surface_y = vol.origin[1];
            for y in (0..vol.size_y).rev() {
                if vol.get(x, y, z) <= 0.0 {
                    surface_y = vol.origin[1] + y as f32 * vol.voxel_size;
                    break;
                }
            }
            data[z * w + x] = surface_y;
        }
    }

    let mut hm = Heightmap::from_data(w, d, data, vol.voxel_size);
    hm.origin = [vol.origin[0], vol.origin[2]];
    hm
}

/// Scatter foliage on the terrain volume and spawn instances as ECS entities.
pub fn scatter_foliage_on_terrain(
    world: &mut World,
    rules: &[FoliageRule],
    density_scale: f32,
) {
    let instances = {
        let vol = world.resource::<TerrainVolume>();
        let hm = heightmap_from_volume(vol);

        // Scale density by density_scale: build scaled rules
        let scaled_rules: Vec<FoliageRule> = rules
            .iter()
            .map(|r| {
                let mut scaled = r.clone();
                scaled.density *= density_scale;
                scaled
            })
            .collect();

        scatter_foliage(&hm, &scaled_rules, 0)
    };

    for inst in instances {
        let id = FOLIAGE_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        world.spawn(SplatInstanceComponent {
            asset_uuid: Uuid::nil(),
            position: Vec3::new(inst.position[0], inst.position[1], inst.position[2]),
            rotation: Quat::from_rotation_y(inst.rotation_y),
            scale: inst.scale,
            instance_id: id,
            lod: LodLevel::Full,
        });
    }
}
