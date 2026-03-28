//! Import helpers — high-level functions for importing assets and caching as .vxm.
//!
//! Used by the editor content browser: when a user selects a .glb file,
//! `import_and_cache` auto-converts to splats and saves a .vxm for fast loading.

use std::path::Path;

use crate::import_pipeline::{import_asset, ImportSettings};

/// A successfully imported and cached asset.
pub struct ImportedAsset {
    pub source_path: std::path::PathBuf,
    pub cached_path: std::path::PathBuf,
    pub splat_count: usize,
    pub collision_box: Option<([f32; 3], [f32; 3])>,
    pub warnings: Vec<String>,
}

/// Import any supported asset file and save as .vxm for fast loading.
///
/// The resulting `.vxm` is written into `cache_dir` with the same stem as the source.
pub fn import_and_cache(
    source_path: &Path,
    cache_dir: &Path,
) -> Result<ImportedAsset, String> {
    let settings = ImportSettings::default();
    let result = import_asset(source_path, &settings)?;

    // Ensure cache directory exists
    std::fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("asset");
    let cache_path = cache_dir.join(format!("{}.vxm", stem));

    let file = crate::vxm::VxmFile {
        header: crate::vxm::VxmHeader::new(
            uuid::Uuid::new_v4(),
            result.splats.len() as u32,
            crate::vxm::MaterialType::Generic,
        ),
        splats: result.splats.clone(),
    };
    let mut out = std::fs::File::create(&cache_path).map_err(|e| e.to_string())?;
    file.write(&mut out).map_err(|e| e.to_string())?;

    Ok(ImportedAsset {
        source_path: source_path.to_path_buf(),
        cached_path: cache_path,
        splat_count: result.splats.len(),
        collision_box: result.collision_box,
        warnings: result.warnings,
    })
}

/// Check if a cached version exists and is newer than the source.
pub fn is_cached(source_path: &Path, cache_dir: &Path) -> bool {
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("asset");
    let cache_path = cache_dir.join(format!("{}.vxm", stem));
    if !cache_path.exists() {
        return false;
    }

    let source_time = std::fs::metadata(source_path)
        .and_then(|m| m.modified())
        .ok();
    let cache_time = std::fs::metadata(&cache_path)
        .and_then(|m| m.modified())
        .ok();

    match (source_time, cache_time) {
        (Some(s), Some(c)) => c >= s,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_ply(dir: &Path) -> std::path::PathBuf {
        let ply_path = dir.join("test_import.ply");
        let mut f = std::fs::File::create(&ply_path).unwrap();
        write!(
            f,
            "ply\nformat ascii 1.0\nelement vertex 50\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        )
        .unwrap();
        for i in 0..50 {
            writeln!(f, "{} {} {}", i as f32 * 0.1, 0.0, 0.0).unwrap();
        }
        ply_path
    }

    #[test]
    fn import_and_cache_creates_vxm() {
        let dir = std::env::temp_dir().join("ochroma_test_import_helpers");
        std::fs::create_dir_all(&dir).unwrap();
        let cache_dir = dir.join("cache");

        let ply_path = create_test_ply(&dir);
        let result = import_and_cache(&ply_path, &cache_dir).unwrap();

        assert!(result.cached_path.exists(), ".vxm should be created");
        assert!(result.splat_count > 0, "should have imported splats");
        assert!(result.cached_path.extension().unwrap() == "vxm");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn is_cached_returns_false_for_missing_cache() {
        let dir = std::env::temp_dir().join("ochroma_test_cache_miss");
        std::fs::create_dir_all(&dir).unwrap();
        let ply_path = create_test_ply(&dir);
        let cache_dir = dir.join("empty_cache");

        assert!(!is_cached(&ply_path, &cache_dir));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn is_cached_returns_true_after_import() {
        let dir = std::env::temp_dir().join("ochroma_test_cache_hit");
        std::fs::create_dir_all(&dir).unwrap();
        let cache_dir = dir.join("cache");

        let ply_path = create_test_ply(&dir);
        import_and_cache(&ply_path, &cache_dir).unwrap();

        assert!(is_cached(&ply_path, &cache_dir));

        std::fs::remove_dir_all(&dir).ok();
    }
}
