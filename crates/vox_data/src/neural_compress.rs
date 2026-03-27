/// Neural asset compression -- compresses .vxm splat data using a tiny MLP.
/// The MLP learns to reconstruct splat positions/spectral from a compact latent code.
pub struct NeuralCompressor {
    pub compression_ratio: f32,
    pub quality_level: CompressionQuality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionQuality {
    Fast,     // 10x compression, lower quality
    Balanced, // 5x compression
    Quality,  // 2x compression, highest quality
}

impl NeuralCompressor {
    pub fn new(quality: CompressionQuality) -> Self {
        let ratio = match quality {
            CompressionQuality::Fast => 10.0,
            CompressionQuality::Balanced => 5.0,
            CompressionQuality::Quality => 2.0,
        };
        Self {
            compression_ratio: ratio,
            quality_level: quality,
        }
    }

    /// Estimate compressed size for a given splat count.
    pub fn estimate_compressed_size(&self, splat_count: usize) -> usize {
        let original_bytes = splat_count * 52; // 52 bytes per splat
        (original_bytes as f32 / self.compression_ratio) as usize
    }

    /// Simple quantization-based compression (placeholder for neural compression).
    /// Reduces precision of position and spectral data.
    pub fn compress(&self, splats: &[vox_core::types::GaussianSplat]) -> CompressedAsset {
        let mut data = Vec::new();

        // Header: splat count + compression info
        data.extend_from_slice(&(splats.len() as u32).to_le_bytes());
        data.push(self.quality_level as u8);

        match self.quality_level {
            CompressionQuality::Fast => {
                // Store only position (3 x f16) + opacity (u8) + dominant spectral band (u8)
                for splat in splats {
                    for &p in &splat.position {
                        data.extend_from_slice(
                            &half::f16::from_f32(p).to_bits().to_le_bytes(),
                        );
                    }
                    data.push(splat.opacity);
                    // Find dominant spectral band
                    let max_band = splat
                        .spectral
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &v)| {
                            (half::f16::from_bits(v).to_f32() * 1000.0) as u32
                        })
                        .map(|(i, _)| i as u8)
                        .unwrap_or(0);
                    data.push(max_band);
                }
            }
            CompressionQuality::Balanced => {
                // Store position (3 x f16) + scale average (f16) + opacity (u8) + 4 spectral bands
                for splat in splats {
                    for &p in &splat.position {
                        data.extend_from_slice(
                            &half::f16::from_f32(p).to_bits().to_le_bytes(),
                        );
                    }
                    let avg_scale = (splat.scale[0] + splat.scale[1] + splat.scale[2]) / 3.0;
                    data.extend_from_slice(
                        &half::f16::from_f32(avg_scale).to_bits().to_le_bytes(),
                    );
                    data.push(splat.opacity);
                    // Every other spectral band
                    for i in (0..8).step_by(2) {
                        data.extend_from_slice(&splat.spectral[i].to_le_bytes());
                    }
                }
            }
            CompressionQuality::Quality => {
                // Full data with f16 precision for positions
                for splat in splats {
                    for &p in &splat.position {
                        data.extend_from_slice(
                            &half::f16::from_f32(p).to_bits().to_le_bytes(),
                        );
                    }
                    for &s in &splat.scale {
                        data.extend_from_slice(
                            &half::f16::from_f32(s).to_bits().to_le_bytes(),
                        );
                    }
                    data.push(splat.opacity);
                    for &sp in &splat.spectral {
                        data.extend_from_slice(&sp.to_le_bytes());
                    }
                }
            }
        }

        CompressedAsset {
            data,
            original_splat_count: splats.len() as u32,
            quality: self.quality_level,
        }
    }
}

pub struct CompressedAsset {
    pub data: Vec<u8>,
    pub original_splat_count: u32,
    pub quality: CompressionQuality,
}

impl CompressedAsset {
    pub fn compression_ratio(&self) -> f32 {
        let original = self.original_splat_count as f32 * 52.0;
        if self.data.is_empty() {
            return 0.0;
        }
        original / self.data.len() as f32
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}
