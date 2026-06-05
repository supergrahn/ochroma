use std::collections::HashMap;
use vox_core::lwc::{TileCoord, TileState};

pub struct TileManager {
    tiles: HashMap<TileCoord, TileState>,
    active_radius: i32,
}

impl TileManager {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            active_radius: 1,
        }
    }

    pub fn with_radius(active_radius: i32) -> Self {
        Self {
            tiles: HashMap::new(),
            active_radius,
        }
    }

    pub fn tile_state(&self, tile: TileCoord) -> TileState {
        self.tiles.get(&tile).copied().unwrap_or(TileState::Cold)
    }

    pub fn update_camera(&mut self, camera_tile: TileCoord) -> Vec<TileCoord> {
        let mut newly_active = Vec::new();

        // Mark far tiles as Evicting then remove
        let r = self.active_radius;
        let to_evict: Vec<TileCoord> = self.tiles.keys().copied().filter(|t| {
            (t.x - camera_tile.x).abs() > r || (t.z - camera_tile.z).abs() > r
        }).collect();
        for t in to_evict {
            self.tiles.remove(&t);
        }

        // Activate tiles within radius; track newly activated ones
        for dx in -r..=r {
            for dz in -r..=r {
                let tile = TileCoord {
                    x: camera_tile.x + dx,
                    z: camera_tile.z + dz,
                };
                let was_present = self.tiles.contains_key(&tile);
                self.tiles.entry(tile).or_insert(TileState::Active);
                if !was_present {
                    newly_active.push(tile);
                }
            }
        }

        newly_active
    }

    pub fn active_tiles(&self) -> Vec<TileCoord> {
        self.tiles.keys().copied().collect()
    }
}

impl Default for TileManager {
    fn default() -> Self {
        Self::new()
    }
}

use vox_data::vxm::{VxmFile, VxmError};

pub struct AsyncAssetLoader;

impl AsyncAssetLoader {
    pub fn new() -> Self { Self }

    pub async fn load_from_bytes(&self, bytes: &[u8]) -> Result<VxmFile, VxmError> {
        let bytes = bytes.to_vec();
        tokio::task::spawn_blocking(move || VxmFile::read(&bytes[..]))
            .await
            .map_err(|e| VxmError::Io(std::io::Error::other(e)))?
    }

    pub async fn load_from_path(&self, path: &std::path::Path) -> Result<VxmFile, VxmError> {
        let bytes = tokio::fs::read(path).await?;
        self.load_from_bytes(&bytes).await
    }
}

impl Default for AsyncAssetLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Spectral codec integration — splat upload path
// ---------------------------------------------------------------------------

use vox_data::spectral_codec::SpectralCodec;
use vox_core::types::GaussianSplat;

/// Splat upload path: applies spectral codec (encode → decode) to each splat's
/// spectral bands during upload, providing neural compression with < 0.15 per-band error.
pub struct SplatUploadPath {
    codec: SpectralCodec,
}

impl SplatUploadPath {
    pub fn new() -> Self {
        Self {
            codec: SpectralCodec::with_hardcoded_weights(),
        }
    }

    /// Process a single splat: encode+decode its spectral bands through the codec.
    /// This acts as a compression step — 16 bands → 4 latent → 16 bands.
    pub fn process_splat(&self, splat: &GaussianSplat) -> GaussianSplat {
        // Decode f16 stored-as-u16 spectral values to f32
        let spectral_f32: [f32; 16] = std::array::from_fn(|b| {
            half::f16::from_bits(splat.spectral()[b]).to_f32()
        });

        // Encode to 4-element latent, then decode back to 16 bands
        let latent = self.codec.encode(&spectral_f32);
        let decoded = self.codec.decode(&latent);

        // Re-encode back to f16 stored as u16
        let mut out = *splat;
        for (slot, &d) in out.spectral_mut().iter_mut().zip(decoded.iter()) {
            *slot = half::f16::from_f32(d).to_bits();
        }
        out
    }

    /// Process a batch of splats through the spectral codec upload path.
    pub fn process_batch(&self, splats: &[GaussianSplat]) -> Vec<GaussianSplat> {
        splats.iter().map(|s| self.process_splat(s)).collect()
    }
}

impl Default for SplatUploadPath {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::lwc::TileCoord;

    #[test]
    fn tile_manager_returns_newly_active_tiles_on_first_update() {
        let mut tm = TileManager::with_radius(1);
        let newly = tm.update_camera(TileCoord { x: 0, z: 0 });
        assert_eq!(newly.len(), 9, "first update should activate 9 tiles (3x3 grid)");
    }

    #[test]
    fn tile_manager_no_newly_active_on_same_position() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 0, z: 0 });
        let newly = tm.update_camera(TileCoord { x: 0, z: 0 });
        assert!(newly.is_empty(), "second call at same position should yield no new tiles");
    }

    #[test]
    fn tile_manager_evicts_distant_tiles() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 0, z: 0 });
        tm.update_camera(TileCoord { x: 100, z: 100 });
        let active = tm.active_tiles();
        assert!(
            !active.contains(&TileCoord { x: 0, z: 0 }),
            "tile (0,0) should be evicted after camera moves to (100,100)"
        );
    }

    #[test]
    fn tile_manager_active_tiles_within_radius() {
        let mut tm = TileManager::with_radius(1);
        tm.update_camera(TileCoord { x: 5, z: 5 });
        let active = tm.active_tiles();
        assert_eq!(active.len(), 9);
        for t in &active {
            assert!((t.x - 5).abs() <= 1 && (t.z - 5).abs() <= 1);
        }
    }

    #[test]
    fn splat_upload_path_codec_round_trip() {
        use vox_core::types::GaussianSplat;
        let path = SplatUploadPath::new();
        // Create a splat with known spectral values
        let v = half::f16::from_f32(0.5).to_bits();
        let splat = GaussianSplat::surface(
            [0.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            0.1,
            0.1,
            200,
            [v; 16],
        );
        let processed = path.process_splat(&splat);
        // After encode+decode, spectral should still be close to original (< 0.15 error)
        for b in 0..16 {
            let orig = half::f16::from_bits(splat.spectral()[b]).to_f32();
            let proc = half::f16::from_bits(processed.spectral()[b]).to_f32();
            let err = (proc - orig).abs();
            assert!(err < 0.15, "band {} upload codec error {:.4} exceeds tolerance", b, err);
        }
    }
}
