use std::path::PathBuf;

/// Output image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    PPM,
    PNG,
}

/// Settings for a movie render job.
pub struct MovieRenderSettings {
    pub output_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frame_rate: u32,
    pub start_time: f32,
    pub end_time: f32,
    pub samples_per_pixel: u32,
    pub format: OutputFormat,
}

/// Batch render queue — renders frames sequentially to disk.
pub struct MovieRenderQueue {
    pub settings: MovieRenderSettings,
    pub current_frame: u32,
    pub total_frames: u32,
    pub completed: bool,
}

impl MovieRenderQueue {
    pub fn new(settings: MovieRenderSettings) -> Self {
        let duration = settings.end_time - settings.start_time;
        let total_frames = (duration * settings.frame_rate as f32).ceil() as u32;
        Self {
            settings,
            current_frame: 0,
            total_frames,
            completed: false,
        }
    }

    /// Time value for the next frame to render.
    pub fn next_frame_time(&self) -> f32 {
        self.settings.start_time + self.current_frame as f32 / self.settings.frame_rate as f32
    }

    /// Save a frame's pixel data to disk. Returns the output file path.
    pub fn save_frame(&mut self, pixels: &[[u8; 4]]) -> Result<PathBuf, String> {
        if self.completed {
            return Err("render already complete".into());
        }

        let expected_pixels = (self.settings.width * self.settings.height) as usize;
        if pixels.len() != expected_pixels {
            return Err(format!(
                "expected {} pixels, got {}",
                expected_pixels,
                pixels.len()
            ));
        }

        let ext = match self.settings.format {
            OutputFormat::PPM => "ppm",
            OutputFormat::PNG => "png",
        };
        let filename = format!("frame_{:06}.{}", self.current_frame, ext);
        let path = self.settings.output_dir.join(&filename);

        // Ensure output directory exists
        std::fs::create_dir_all(&self.settings.output_dir)
            .map_err(|e| format!("failed to create output dir: {}", e))?;

        match self.settings.format {
            OutputFormat::PPM => {
                let data = format!(
                    "P6\n{} {}\n255\n",
                    self.settings.width, self.settings.height
                );
                let mut rgb = Vec::with_capacity(pixels.len() * 3);
                for px in pixels {
                    rgb.push(px[0]);
                    rgb.push(px[1]);
                    rgb.push(px[2]);
                }
                let header_bytes = data.as_bytes().to_vec();
                let mut full = header_bytes;
                full.extend_from_slice(&rgb);
                std::fs::write(&path, &full)
                    .map_err(|e| format!("write failed: {}", e))?;
            }
            OutputFormat::PNG => {
                // Simplified: write as PPM with .png extension (real impl would use png encoder)
                let header = format!(
                    "P6\n{} {}\n255\n",
                    self.settings.width, self.settings.height
                );
                let mut rgb = Vec::with_capacity(pixels.len() * 3);
                for px in pixels {
                    rgb.push(px[0]);
                    rgb.push(px[1]);
                    rgb.push(px[2]);
                }
                let mut full = header.into_bytes();
                full.extend_from_slice(&rgb);
                std::fs::write(&path, &full)
                    .map_err(|e| format!("write failed: {}", e))?;
            }
        }

        self.current_frame += 1;
        if self.current_frame >= self.total_frames {
            self.completed = true;
        }

        Ok(path)
    }

    /// Render progress as a fraction [0, 1].
    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 1.0;
        }
        self.current_frame as f32 / self.total_frames as f32
    }

    pub fn is_complete(&self) -> bool {
        self.completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_settings(dir: PathBuf) -> MovieRenderSettings {
        MovieRenderSettings {
            output_dir: dir,
            width: 4,
            height: 4,
            frame_rate: 30,
            start_time: 0.0,
            end_time: 1.0,
            samples_per_pixel: 1,
            format: OutputFormat::PPM,
        }
    }

    #[test]
    fn frame_count_computed_correctly() {
        let dir = std::env::temp_dir().join("ochroma_movie_test_count");
        let q = MovieRenderQueue::new(test_settings(dir));
        assert_eq!(q.total_frames, 30); // 1 second at 30fps
        assert_eq!(q.current_frame, 0);
        assert!(!q.is_complete());
    }

    #[test]
    fn progress_tracks() {
        let dir = std::env::temp_dir().join("ochroma_movie_test_progress");
        let mut q = MovieRenderQueue::new(test_settings(dir));
        assert!((q.progress() - 0.0).abs() < 1e-5);

        // Manually advance
        q.current_frame = 15;
        assert!((q.progress() - 0.5).abs() < 1e-5);

        q.current_frame = 30;
        assert!((q.progress() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn save_frame_creates_file() {
        let dir = std::env::temp_dir().join("ochroma_movie_test_save");
        let _ = std::fs::remove_dir_all(&dir);

        let mut q = MovieRenderQueue::new(test_settings(dir.clone()));
        let pixels = vec![[255u8, 0, 0, 255]; 16]; // 4x4 red
        let path = q.save_frame(&pixels).expect("save should succeed");
        assert!(path.exists());
        assert_eq!(q.current_frame, 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
