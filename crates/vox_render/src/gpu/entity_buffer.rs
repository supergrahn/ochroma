pub struct EntityIdBuffer {
    width: u32,
    height: u32,
    data: Vec<u16>,
}

impl EntityIdBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u16; (width * height) as usize],
        }
    }

    pub fn write(&mut self, x: u32, y: u32, entity_id: u16) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = entity_id;
        }
    }

    /// Returns the entity ID at (x, y), or 0 if out of bounds.
    pub fn pick(&self, x: u32, y: u32) -> u16 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.data[(y * self.width + x) as usize]
    }

    pub fn clear(&mut self) {
        for v in &mut self.data {
            *v = 0;
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.data = vec![0u16; (width * height) as usize];
    }
}
