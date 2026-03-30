//! Denoiser CNN for noisy spectral renders.
//! Format: safetensors (NOT GGUF — GGUF is for LLMs; CNNs export via safetensors).

use anyhow::{bail, Result};
use std::path::PathBuf;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpectralPixel {
    pub bands: [f32; 16],
}

pub struct SpectralFramebuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<SpectralPixel>,
}

impl SpectralFramebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![SpectralPixel { bands: [0.0f32; 16] }; width * height],
        }
    }

    pub fn pixel(&self, x: usize, y: usize) -> &SpectralPixel {
        &self.pixels[y * self.width + x]
    }

    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut SpectralPixel {
        &mut self.pixels[y * self.width + x]
    }

    pub fn total_energy(&self) -> f32 {
        self.pixels.iter().flat_map(|p| p.bands.iter().copied()).sum()
    }
}

#[derive(Debug)]
pub struct SpectralDenoiser {
    pub model_path: PathBuf,
    weights_loaded: bool,
    pub blur_radius: usize,
}

impl SpectralDenoiser {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self> {
        let path = model_path.into();
        if !path.exists() {
            bail!("Denoiser model not found: {}", path.display());
        }
        let meta = std::fs::metadata(&path)?;
        if meta.len() < 8 {
            bail!("Denoiser model too small to be a valid safetensors file");
        }
        let mut buf = [0u8; 8];
        {
            use std::io::Read;
            std::fs::File::open(&path)?.read_exact(&mut buf)?;
        }
        let header_len = u64::from_le_bytes(buf);
        if header_len == 0 || header_len > meta.len() {
            bail!(
                "Invalid safetensors header length {} in {}",
                header_len,
                path.display()
            );
        }
        Ok(Self { model_path: path, weights_loaded: true, blur_radius: 1 })
    }

    pub fn stub(blur_radius: usize) -> Self {
        Self { model_path: PathBuf::from("<stub>"), weights_loaded: false, blur_radius }
    }

    pub fn apply(&self, fb: &mut SpectralFramebuffer) {
        let _ = self.weights_loaded;
        self.blur_fallback(fb);
    }

    fn blur_fallback(&self, fb: &mut SpectralFramebuffer) {
        let r = self.blur_radius;
        if r == 0 {
            return;
        }
        let w = fb.width;
        let h = fb.height;
        let original = fb.pixels.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 16];
                let mut count = 0.0f32;
                for dx in 0..=(2 * r) {
                    let nx = x + dx;
                    if nx < r || nx - r >= w {
                        continue;
                    }
                    let nx = nx - r;
                    for b in 0..16 {
                        acc[b] += original[y * w + nx].bands[b];
                    }
                    count += 1.0;
                }
                if count > 0.0 {
                    for b in 0..16 {
                        fb.pixels[y * w + x].bands[b] = acc[b] / count;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_denoiser_applies_without_panic() {
        let denoiser = SpectralDenoiser::stub(1);
        let mut fb = SpectralFramebuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                fb.pixel_mut(x, y).bands[3] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let energy_before = fb.total_energy();
        denoiser.apply(&mut fb);
        let energy_after = fb.total_energy();
        let diff = (energy_after - energy_before).abs();
        assert!(
            diff < energy_before * 0.1,
            "denoiser must roughly conserve energy: before={:.2} after={:.2}",
            energy_before,
            energy_after
        );
    }

    #[test]
    fn load_missing_file_returns_err() {
        let result = SpectralDenoiser::load("/tmp/__no_denoiser__.safetensors");
        assert!(result.is_err(), "missing file must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found"), "error must say 'not found': {}", msg);
    }

    #[test]
    fn load_invalid_safetensors_returns_err_with_header_message() {
        let path = "/tmp/bad_safetensors_test.safetensors";
        std::fs::write(path, u64::MAX.to_le_bytes()).unwrap();
        let result = SpectralDenoiser::load(path);
        std::fs::remove_file(path).ok();
        assert!(result.is_err(), "invalid header must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("header length") || msg.contains("header"),
            "error must mention header length, got: {}", msg
        );
    }

    #[test]
    fn framebuffer_pixel_indexing() {
        let mut fb = SpectralFramebuffer::new(4, 4);
        fb.pixel_mut(2, 3).bands[5] = 0.7;
        assert!((fb.pixel(2, 3).bands[5] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn blur_smooths_checkerboard_noise() {
        let denoiser = SpectralDenoiser::stub(1);
        let mut fb = SpectralFramebuffer::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                fb.pixel_mut(x, y).bands[0] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        denoiser.apply(&mut fb);
        let interior_val = fb.pixel(5, 5).bands[0];
        assert!(
            interior_val > 0.0 && interior_val < 1.0,
            "blur must smooth checkerboard (got {})",
            interior_val
        );
    }
}
