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
