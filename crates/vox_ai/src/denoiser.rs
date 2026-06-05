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

    /// True once a safetensors header has been validated via [`Self::load`].
    ///
    /// NOTE (honest depth limit): even when weights are "loaded" we do NOT run a
    /// neural forward pass — there is no safetensors tensor reader / conv kernel
    /// wired up yet. [`apply`](Self::apply) always runs the deterministic
    /// separable box blur below. This accessor exists so callers can detect
    /// whether real weights were present, and so the field is genuinely read.
    pub fn weights_loaded(&self) -> bool {
        self.weights_loaded
    }

    /// Denoise the framebuffer in place.
    ///
    /// Current implementation is a real separable 2D box blur (horizontal then
    /// vertical pass), not a CNN forward pass. See [`weights_loaded`](Self::weights_loaded).
    pub fn apply(&self, fb: &mut SpectralFramebuffer) {
        self.blur_fallback(fb);
    }

    /// Separable 2D box blur: a horizontal pass followed by a vertical pass.
    ///
    /// A separable box filter of radius `r` is mathematically equivalent to a
    /// full `(2r+1) x (2r+1)` 2D box convolution but runs in `O(r)` per pixel
    /// instead of `O(r^2)`. Each pass is normalized by the number of in-bounds
    /// taps, so interior pixels conserve total energy while edge pixels clamp
    /// to the available neighbourhood.
    fn blur_fallback(&self, fb: &mut SpectralFramebuffer) {
        let r = self.blur_radius;
        if r == 0 {
            return;
        }
        let w = fb.width;
        let h = fb.height;
        if w == 0 || h == 0 {
            return;
        }

        // --- Horizontal pass: read from `src`, write into `tmp`. ---
        let src = fb.pixels.clone();
        let mut tmp = src.clone();
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let mut acc = [0.0f32; 16];
                let mut count = 0.0f32;
                let lo = x.saturating_sub(r);
                let hi = (x + r).min(w - 1);
                for nx in lo..=hi {
                    let p = &src[row + nx].bands;
                    for b in 0..16 {
                        acc[b] += p[b];
                    }
                    count += 1.0;
                }
                let inv = 1.0 / count;
                let out = &mut tmp[row + x].bands;
                for b in 0..16 {
                    out[b] = acc[b] * inv;
                }
            }
        }

        // --- Vertical pass: read from `tmp`, write into `fb`. ---
        for x in 0..w {
            for y in 0..h {
                let mut acc = [0.0f32; 16];
                let mut count = 0.0f32;
                let lo = y.saturating_sub(r);
                let hi = (y + r).min(h - 1);
                for ny in lo..=hi {
                    let p = &tmp[ny * w + x].bands;
                    for b in 0..16 {
                        acc[b] += p[b];
                    }
                    count += 1.0;
                }
                let inv = 1.0 / count;
                let out = &mut fb.pixels[y * w + x].bands;
                for b in 0..16 {
                    out[b] = acc[b] * inv;
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
    fn stub_reports_weights_not_loaded() {
        let denoiser = SpectralDenoiser::stub(2);
        assert!(!denoiser.weights_loaded(), "stub must report no weights loaded");
        assert_eq!(denoiser.blur_radius, 2);
    }

    #[test]
    fn load_valid_safetensors_reports_weights_loaded() {
        // Minimal valid-ish safetensors: 8-byte little-endian header length that
        // is non-zero and <= file size, followed by enough bytes.
        let path = "/tmp/vox_ai_good_safetensors_test.safetensors";
        let header_len: u64 = 16;
        let mut bytes = header_len.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 32]); // header + payload padding
        std::fs::write(path, &bytes).unwrap();
        let denoiser = SpectralDenoiser::load(path).unwrap();
        std::fs::remove_file(path).ok();
        assert!(
            denoiser.weights_loaded(),
            "validated safetensors must report weights_loaded()"
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
    fn separable_blur_spreads_bright_pixel_in_both_axes() {
        // A single bright pixel in the center must leak energy into BOTH its
        // horizontal AND vertical neighbours after a real separable 2D blur.
        // A horizontal-only blur would leave the vertical neighbours at 0.
        let denoiser = SpectralDenoiser::stub(1);
        let mut fb = SpectralFramebuffer::new(7, 7);
        fb.pixel_mut(3, 3).bands[2] = 1.0;
        denoiser.apply(&mut fb);

        let left = fb.pixel(2, 3).bands[2];
        let right = fb.pixel(4, 3).bands[2];
        let up = fb.pixel(3, 2).bands[2];
        let down = fb.pixel(3, 4).bands[2];

        assert!(
            left > 0.0 && right > 0.0,
            "horizontal neighbours must receive energy: left={left} right={right}"
        );
        assert!(
            up > 0.0 && down > 0.0,
            "VERTICAL neighbours must receive energy (separable 2D blur): up={up} down={down}"
        );
        // Separable 3x3 box: corner diagonal also receives energy (proof both passes ran).
        let diag = fb.pixel(2, 2).bands[2];
        assert!(
            diag > 0.0,
            "diagonal neighbour must receive energy after both passes: diag={diag}"
        );
        // Center must have decreased (energy redistributed outward).
        let center = fb.pixel(3, 3).bands[2];
        assert!(
            center < 1.0 && center > 0.0,
            "center must redistribute energy: center={center}"
        );
    }

    #[test]
    fn separable_blur_conserves_energy_interior() {
        // For a fully-interior bright pixel, a normalized separable box blur
        // must conserve total energy (no edge clipping involved).
        let denoiser = SpectralDenoiser::stub(1);
        let mut fb = SpectralFramebuffer::new(9, 9);
        fb.pixel_mut(4, 4).bands[0] = 9.0;
        let before = fb.total_energy();
        denoiser.apply(&mut fb);
        let after = fb.total_energy();
        assert!(
            (after - before).abs() < 1e-4,
            "interior separable blur must conserve energy: before={before} after={after}"
        );
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
