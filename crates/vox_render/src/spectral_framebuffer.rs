/// An 8-channel spectral framebuffer with auxiliary buffers.
/// This is the core data structure of the rendering pipeline.
pub struct SpectralFramebuffer {
    pub width: u32,
    pub height: u32,
    /// 8 spectral bands per pixel (380-660nm, f32).
    pub spectral: Vec<[f32; 8]>,
    /// Depth buffer (camera-space Z, f32).
    pub depth: Vec<f32>,
    /// World-space normals (3 × f32).
    pub normals: Vec<[f32; 3]>,
    /// Screen-space motion vectors (2 × f32, pixels).
    pub motion: Vec<[f32; 2]>,
    /// Object/entity ID per pixel.
    pub object_id: Vec<u32>,
    /// Spectral albedo (for denoiser — separates material from lighting).
    pub albedo: Vec<[f32; 8]>,
    /// Sample count per pixel (for adaptive sampling).
    pub sample_count: Vec<u16>,
}

impl SpectralFramebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let count = (width * height) as usize;
        Self {
            width,
            height,
            spectral: vec![[0.0; 8]; count],
            depth: vec![f32::MAX; count],
            normals: vec![[0.0, 1.0, 0.0]; count],
            motion: vec![[0.0; 2]; count],
            object_id: vec![0; count],
            albedo: vec![[0.0; 8]; count],
            sample_count: vec![0; count],
        }
    }

    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// Write a spectral sample to a pixel (accumulative — averages with existing samples).
    pub fn write_sample(
        &mut self,
        x: u32,
        y: u32,
        spectral: [f32; 8],
        depth: f32,
        normal: [f32; 3],
        object_id: u32,
        albedo: [f32; 8],
    ) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = self.idx(x, y);
        let n = self.sample_count[i] as f32;

        // Running average
        if n == 0.0 {
            self.spectral[i] = spectral;
            self.depth[i] = depth;
            self.normals[i] = normal;
            self.albedo[i] = albedo;
        } else {
            for b in 0..8 {
                self.spectral[i][b] = (self.spectral[i][b] * n + spectral[b]) / (n + 1.0);
                self.albedo[i][b] = (self.albedo[i][b] * n + albedo[b]) / (n + 1.0);
            }
            // Keep nearest depth
            if depth < self.depth[i] {
                self.depth[i] = depth;
            }
            // Average normals
            for c in 0..3 {
                self.normals[i][c] = (self.normals[i][c] * n + normal[c]) / (n + 1.0);
            }
        }
        self.object_id[i] = object_id;
        self.sample_count[i] += 1;
    }

    /// Set motion vector for temporal reprojection.
    pub fn write_motion(&mut self, x: u32, y: u32, mv: [f32; 2]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = self.idx(x, y);
        self.motion[i] = mv;
    }

    /// Clear all buffers for a new frame.
    pub fn clear(&mut self) {
        for s in &mut self.spectral {
            *s = [0.0; 8];
        }
        for d in &mut self.depth {
            *d = f32::MAX;
        }
        for n in &mut self.normals {
            *n = [0.0, 1.0, 0.0];
        }
        for m in &mut self.motion {
            *m = [0.0; 2];
        }
        for id in &mut self.object_id {
            *id = 0;
        }
        for a in &mut self.albedo {
            *a = [0.0; 8];
        }
        for s in &mut self.sample_count {
            *s = 0;
        }
    }

    /// Average samples per pixel.
    pub fn avg_samples(&self) -> f32 {
        let total: u64 = self.sample_count.iter().map(|&s| s as u64).sum();
        total as f32 / self.pixel_count() as f32
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let count = self.pixel_count();
        count * (8 * 4 // spectral: 8 f32
            + 4        // depth: f32
            + 3 * 4    // normals: 3 f32
            + 2 * 4    // motion: 2 f32
            + 4        // object_id: u32
            + 8 * 4    // albedo: 8 f32
            + 2)       // sample_count: u16
    }
}
