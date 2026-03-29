//! Binary serialisation for baked GI data (.vxgi format).
//!
//! Format: 4-byte magic "VXGI", 4-byte u32 splat count,
//! then splat_count * 8 * 4 bytes of f32 irradiance values.

use std::path::Path;

const MAGIC: &[u8; 4] = b"VXGI";

/// Save baked GI irradiance to a `.vxgi` binary file.
pub fn save_vxgi(irradiance: &[[f32; 8]], path: &Path) -> Result<(), String> {
    let mut buf = Vec::with_capacity(8 + irradiance.len() * 32);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(irradiance.len() as u32).to_le_bytes());
    for entry in irradiance {
        for &v in entry {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &buf).map_err(|e| e.to_string())
}

/// Load baked GI irradiance from a `.vxgi` binary file.
pub fn load_vxgi(path: &Path) -> Result<Vec<[f32; 8]>, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 8 { return Err("File too short".into()); }
    if &data[0..4] != MAGIC { return Err("Invalid magic bytes".into()); }
    let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let expected = 8 + count * 32;
    if data.len() < expected {
        return Err(format!("Truncated: expected {} bytes, got {}", expected, data.len()));
    }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let base = 8 + i * 32;
        let mut entry = [0.0f32; 8];
        for (j, v) in entry.iter_mut().enumerate() {
            let off = base + j * 4;
            *v = f32::from_le_bytes(data[off..off+4].try_into().unwrap());
        }
        result.push(entry);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let irr: Vec<[f32; 8]> = vec![
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ];
        let path = std::env::temp_dir().join("test_gi.vxgi");
        save_vxgi(&irr, &path).unwrap();
        let loaded = load_vxgi(&path).unwrap();
        assert_eq!(irr.len(), loaded.len());
        for (a, b) in irr.iter().zip(loaded.iter()) {
            for (&x, &y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_invalid_magic_returns_error() {
        let path = std::env::temp_dir().join("test_gi_bad.vxgi");
        std::fs::write(&path, b"NOPE0000").unwrap();
        assert!(load_vxgi(&path).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_missing_file_returns_error() {
        assert!(load_vxgi(Path::new("/nonexistent/gi.vxgi")).is_err());
    }
}
