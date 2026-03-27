use std::path::PathBuf;

/// Captures the framebuffer to a PNG file.
pub struct ScreenshotCapture {
    pub save_directory: PathBuf,
    pub capture_requested: bool,
    next_index: u32,
}

impl ScreenshotCapture {
    pub fn new() -> Self {
        let save_dir = dirs_next::picture_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Ochroma Screenshots");
        std::fs::create_dir_all(&save_dir).ok();
        Self {
            save_directory: save_dir,
            capture_requested: false,
            next_index: 0,
        }
    }

    pub fn request_capture(&mut self) {
        self.capture_requested = true;
    }

    /// Save RGBA8 pixels to a PNG file. Returns the file path on success.
    pub fn save_framebuffer(&mut self, pixels: &[[u8; 4]], width: u32, height: u32) -> Result<PathBuf, String> {
        self.capture_requested = false;
        self.next_index += 1;

        let filename = format!("ochroma_{:04}.png", self.next_index);
        let path = self.save_directory.join(&filename);

        // Flatten RGBA to bytes
        let mut rgba_bytes = Vec::with_capacity((width * height * 4) as usize);
        for pixel in pixels {
            rgba_bytes.extend_from_slice(pixel);
        }

        // Use the image crate if available, otherwise write raw PPM
        // For now, write a simple PPM file (always works, no deps)
        let ppm_path = path.with_extension("ppm");
        let mut data = format!("P6\n{} {}\n255\n", width, height).into_bytes();
        for pixel in pixels {
            data.push(pixel[0]); // R
            data.push(pixel[1]); // G
            data.push(pixel[2]); // B
        }

        std::fs::write(&ppm_path, &data).map_err(|e| e.to_string())?;
        println!("[ochroma] Screenshot saved: {}", ppm_path.display());
        Ok(ppm_path)
    }
}
