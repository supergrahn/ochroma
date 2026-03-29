/// Minimap data for the UI -- a small overview of the city.
pub struct Minimap {
    pub size: u32,
    pub pixels: Vec<[u8; 4]>,
    pub camera_x: f32,
    pub camera_z: f32,
    pub world_size: f32,
}

impl Minimap {
    pub fn new(size: u32, world_size: f32) -> Self {
        Self {
            size,
            pixels: vec![[30, 60, 30, 255]; (size * size) as usize], // dark green default
            camera_x: 0.0,
            camera_z: 0.0,
            world_size,
        }
    }

    /// Update a pixel on the minimap based on world content.
    pub fn set_pixel(&mut self, world_x: f32, world_z: f32, color: [u8; 4]) {
        let px = ((world_x / self.world_size + 0.5) * self.size as f32) as u32;
        let pz = ((world_z / self.world_size + 0.5) * self.size as f32) as u32;
        if px < self.size && pz < self.size {
            self.pixels[(pz * self.size + px) as usize] = color;
        }
    }

    /// Mark roads on the minimap.
    pub fn mark_road(&mut self, x: f32, z: f32) {
        self.set_pixel(x, z, [80, 80, 80, 255]); // grey for roads
    }

    /// Mark buildings on the minimap.
    pub fn mark_building(&mut self, x: f32, z: f32) {
        self.set_pixel(x, z, [200, 150, 100, 255]); // brown for buildings
    }

    /// Mark zones on the minimap.
    pub fn mark_zone(&mut self, x: f32, z: f32, zone_type: &str) {
        let color = match zone_type {
            "residential" => [100, 150, 255, 255],
            "commercial" => [255, 220, 100, 255],
            "industrial" => [180, 100, 220, 255],
            _ => [128, 128, 128, 255],
        };
        self.set_pixel(x, z, color);
    }

    /// Update camera position for the viewport indicator.
    pub fn update_camera(&mut self, x: f32, z: f32) {
        self.camera_x = x;
        self.camera_z = z;
    }
}

// ── MiniMap (egui painter-based) ──────────────────────────────────────────

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct MiniMapEntity {
    pub position: Vec3,
    pub color: egui::Color32,
}

pub struct MiniMap {
    pub radius: f32,
    pub open: bool,
    pub widget_size: f32,
}

impl MiniMap {
    pub fn new(radius: f32) -> Self {
        Self { radius, open: true, widget_size: 200.0 }
    }

    pub(crate) fn world_to_map(&self, world_pos: Vec3, camera_pos: Vec3, rect: egui::Rect) -> egui::Pos2 {
        let center = rect.center();
        let scale = self.widget_size / (2.0 * self.radius);
        let dx = (world_pos.x - camera_pos.x) * scale;
        let dz = (world_pos.z - camera_pos.z) * scale;
        egui::pos2(center.x + dx, center.y + dz)
    }

    pub fn show(&mut self, ctx: &egui::Context, entities: &[MiniMapEntity], camera_pos: Vec3) {
        if !self.open { return; }
        egui::Window::new("Mini Map")
            .resizable(false)
            .default_size([self.widget_size, self.widget_size])
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(self.widget_size, self.widget_size),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(20, 30, 20));
                for entity in entities {
                    let pos = self.world_to_map(entity.position, camera_pos, rect);
                    if rect.contains(pos) {
                        painter.circle_filled(pos, 2.0, entity.color);
                    }
                }
                let cam_pos = self.world_to_map(camera_pos, camera_pos, rect);
                painter.circle_filled(cam_pos, 4.0, egui::Color32::from_rgb(255, 255, 0));
            });
    }
}

impl Default for MiniMap {
    fn default() -> Self { Self::new(500.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimap_egui_default_radius() {
        let mm = MiniMap::default();
        assert_eq!(mm.radius, 500.0);
        assert!(mm.open);
    }

    #[test]
    fn minimap_world_to_map_center() {
        let mm = MiniMap::new(100.0);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let pos = mm.world_to_map(Vec3::new(10.0, 0.0, 10.0), Vec3::new(10.0, 0.0, 10.0), rect);
        assert!((pos.x - 100.0).abs() < 1.0);
        assert!((pos.y - 100.0).abs() < 1.0);
    }

    #[test]
    fn minimap_world_to_map_offset() {
        let mm = MiniMap::new(100.0);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let pos = mm.world_to_map(Vec3::new(50.0, 0.0, 0.0), Vec3::ZERO, rect);
        assert!((pos.x - 150.0).abs() < 1.0);
    }

    #[test]
    fn minimap_toggle_open() {
        let mut mm = MiniMap::default();
        assert!(mm.open);
        mm.open = false;
        assert!(!mm.open);
    }
}
