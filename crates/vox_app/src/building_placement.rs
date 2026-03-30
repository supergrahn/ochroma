//! Building placement: ghost preview, grid snapping, slope validation, confirm.

use glam::Vec3;

/// Defines a placeable building type.
#[derive(Clone, Debug)]
pub struct BuildingTemplate {
    pub name: String,
    /// Width × depth in metres.
    pub footprint: [f32; 2],
    /// Maximum terrain slope in degrees before placement is invalid.
    pub max_slope_deg: f32,
    pub asset_path: String,
}

/// A building that has been confirmed and should be spawned.
#[derive(Clone, Debug)]
pub struct PlacedBuilding {
    pub position: Vec3,
    pub template: BuildingTemplate,
}

/// Manages the ghost building preview during placement mode.
pub struct BuildingPlacer {
    pub template: BuildingTemplate,
    pub ghost_pos: Vec3,
    pub valid: bool,
    pub slope_deg: f32,
}

impl BuildingPlacer {
    pub fn new(template: BuildingTemplate) -> Self {
        Self {
            template,
            ghost_pos: Vec3::ZERO,
            valid: false,
            slope_deg: 0.0,
        }
    }

    pub fn update(
        &mut self,
        ray_origin: Vec3,
        ray_dir: Vec3,
        height_fn: &dyn Fn(f32, f32) -> f32,
    ) {
        let t = if ray_dir.y.abs() > 1e-6 {
            -ray_origin.y / ray_dir.y
        } else {
            20.0
        };
        let hit = ray_origin + ray_dir * t;

        let grid = self.template.footprint[0].min(self.template.footprint[1]) * 0.5;
        let snapped_x = (hit.x / grid).round() * grid;
        let snapped_z = (hit.z / grid).round() * grid;
        let snapped_y = height_fn(snapped_x, snapped_z);
        self.ghost_pos = Vec3::new(snapped_x, snapped_y, snapped_z);

        let hw = self.template.footprint[0] * 0.5;
        let hd = self.template.footprint[1] * 0.5;
        let corners = [
            height_fn(snapped_x - hw, snapped_z - hd),
            height_fn(snapped_x + hw, snapped_z - hd),
            height_fn(snapped_x - hw, snapped_z + hd),
            height_fn(snapped_x + hw, snapped_z + hd),
        ];
        let h_min = corners.iter().cloned().fold(f32::MAX, f32::min);
        let h_max = corners.iter().cloned().fold(f32::MIN, f32::max);
        let diagonal = (self.template.footprint[0].powi(2) + self.template.footprint[1].powi(2)).sqrt();
        let rise_over_run = (h_max - h_min) / diagonal.max(0.001);
        self.slope_deg = rise_over_run.atan().to_degrees();
        self.valid = self.slope_deg <= self.template.max_slope_deg;
    }

    pub fn confirm(&self) -> Option<PlacedBuilding> {
        if !self.valid {
            return None;
        }
        Some(PlacedBuilding {
            position: self.ghost_pos,
            template: self.template.clone(),
        })
    }
}
