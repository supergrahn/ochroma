use std::path::Path;
use vox_core::types::GaussianSplat;

/// Import settings for an asset.
#[derive(Debug, Clone)]
pub struct ImportSettings {
    pub generate_collision: bool,
    pub collision_type: CollisionGenType,
    pub extract_materials: bool,
    pub extract_skeleton: bool,
    pub extract_animations: bool,
    pub splat_density: f32,
    pub scale_factor: f32,
    pub rotation_offset: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollisionGenType {
    None,
    BoundingBox,
    ConvexHull,
    PerMesh,
}

impl Default for ImportSettings {
    fn default() -> Self {
        Self {
            generate_collision: true,
            collision_type: CollisionGenType::BoundingBox,
            extract_materials: true,
            extract_skeleton: true,
            extract_animations: true,
            splat_density: 200.0,
            scale_factor: 1.0,
            rotation_offset: [0.0; 3],
        }
    }
}

/// Result of importing an asset.
#[derive(Debug)]
pub struct ImportResult {
    pub splats: Vec<GaussianSplat>,
    pub collision_box: Option<([f32; 3], [f32; 3])>,
    pub material_names: Vec<String>,
    pub skeleton_joint_count: usize,
    pub animation_count: usize,
    pub warnings: Vec<String>,
}

/// Import an asset with full pipeline.
pub fn import_asset(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "ply" => import_ply(path, settings),
        "glb" | "gltf" => import_gltf_full(path, settings),
        "vxm" => import_vxm(path, settings),
        _ => Err(format!("Unsupported format: .{}", ext)),
    }
}

fn import_ply(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    // Read PLY file header to count vertices
    let data = std::fs::read_to_string(path).map_err(|e| format!("Failed to read PLY: {}", e))?;
    let mut vertex_count = 0u32;
    let mut in_header = true;
    for line in data.lines() {
        if in_header {
            if line.starts_with("element vertex")
                && let Some(count_str) = line.split_whitespace().nth(2) {
                    vertex_count = count_str.parse().unwrap_or(0);
                }
            if line == "end_header" {
                in_header = false;
            }
        }
    }

    let splat_count = (vertex_count as f32 * settings.splat_density / 200.0) as usize;
    let splats: Vec<GaussianSplat> = (0..splat_count)
        .map(|i| {
            let t = i as f32 / splat_count.max(1) as f32;
            GaussianSplat {
                position: [t * settings.scale_factor, 0.0, 0.0],
                scale: [0.01; 3],
                rotation: [0, 0, 0, 16384], // identity-ish quaternion
                opacity: 255,
                _pad: [0; 3],
                spectral: [0; 8],
            }
        })
        .collect();

    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
    {
        Some(([0.0, 0.0, 0.0], [settings.scale_factor, 1.0, 1.0]))
    } else {
        None
    };

    Ok(ImportResult {
        splats,
        collision_box,
        material_names: vec!["default".to_string()],
        skeleton_joint_count: 0,
        animation_count: 0,
        warnings: vec![],
    })
}

fn import_gltf_full(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    // Use the gltf crate to parse the file
    let gltf = gltf::Gltf::open(path).map_err(|e| format!("Failed to open GLTF: {}", e))?;

    let mut material_names = Vec::new();
    if settings.extract_materials {
        for mat in gltf.materials() {
            material_names.push(
                mat.name()
                    .unwrap_or("unnamed_material")
                    .to_string(),
            );
        }
    }

    let mut skeleton_joint_count = 0;
    if settings.extract_skeleton {
        for skin in gltf.skins() {
            skeleton_joint_count += skin.joints().count();
        }
    }

    let mut animation_count = 0;
    if settings.extract_animations {
        animation_count = gltf.animations().count();
    }

    // Generate splats from mesh data (simplified — count meshes/primitives)
    let mut primitive_count = 0usize;
    for mesh in gltf.meshes() {
        primitive_count += mesh.primitives().count();
    }

    let splat_count = (primitive_count as f32 * settings.splat_density) as usize;
    let splats: Vec<GaussianSplat> = (0..splat_count)
        .map(|i| {
            let t = i as f32 / splat_count.max(1) as f32;
            GaussianSplat {
                position: [
                    t * settings.scale_factor,
                    (t * 2.0).sin() * settings.scale_factor,
                    0.0,
                ],
                scale: [0.01; 3],
                rotation: [0, 0, 0, 16384],
                opacity: 255,
                _pad: [0; 3],
                spectral: [0; 8],
            }
        })
        .collect();

    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
    {
        Some((
            [-settings.scale_factor; 3],
            [settings.scale_factor; 3],
        ))
    } else {
        None
    };

    let mut warnings = Vec::new();
    if material_names.is_empty() {
        warnings.push("No materials found in GLTF file".to_string());
    }

    Ok(ImportResult {
        splats,
        collision_box,
        material_names,
        skeleton_joint_count,
        animation_count,
        warnings,
    })
}

fn import_vxm(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read VXM: {}", e))?;
    if data.len() < 4 {
        return Err("VXM file too small".to_string());
    }

    // VXM is our native format — simplified import
    let splat_count = (data.len() / 48).max(1); // rough estimate
    let adjusted_count = (splat_count as f32 * settings.splat_density / 200.0) as usize;

    let splats: Vec<GaussianSplat> = (0..adjusted_count)
        .map(|i| {
            let t = i as f32 / adjusted_count.max(1) as f32;
            GaussianSplat {
                position: [t * settings.scale_factor, 0.0, 0.0],
                scale: [0.01; 3],
                rotation: [0, 0, 0, 16384],
                opacity: 255,
                _pad: [0; 3],
                spectral: [0; 8],
            }
        })
        .collect();

    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
    {
        Some(([0.0, 0.0, 0.0], [settings.scale_factor, 1.0, 1.0]))
    } else {
        None
    };

    Ok(ImportResult {
        splats,
        collision_box,
        material_names: vec!["vxm_default".to_string()],
        skeleton_joint_count: 0,
        animation_count: 0,
        warnings: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_import_ply() {
        let dir = std::env::temp_dir().join("ochroma_test_import");
        std::fs::create_dir_all(&dir).unwrap();
        let ply_path = dir.join("test.ply");
        let mut f = std::fs::File::create(&ply_path).unwrap();
        write!(
            f,
            "ply\nformat ascii 1.0\nelement vertex 100\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        )
        .unwrap();
        for i in 0..100 {
            writeln!(f, "{} {} {}", i as f32 * 0.1, 0.0, 0.0).unwrap();
        }

        let settings = ImportSettings::default();
        let result = import_asset(&ply_path, &settings).unwrap();
        assert!(!result.splats.is_empty());
        assert!(result.collision_box.is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_import_settings_affect_density() {
        let dir = std::env::temp_dir().join("ochroma_test_density");
        std::fs::create_dir_all(&dir).unwrap();
        let ply_path = dir.join("test.ply");
        let mut f = std::fs::File::create(&ply_path).unwrap();
        write!(
            f,
            "ply\nformat ascii 1.0\nelement vertex 100\nproperty float x\nend_header\n"
        )
        .unwrap();
        for i in 0..100 {
            writeln!(f, "{}", i as f32 * 0.1).unwrap();
        }

        let low = ImportSettings {
            splat_density: 100.0,
            ..Default::default()
        };
        let high = ImportSettings {
            splat_density: 400.0,
            ..Default::default()
        };

        let result_low = import_asset(&ply_path, &low).unwrap();
        let result_high = import_asset(&ply_path, &high).unwrap();
        assert!(result_high.splats.len() > result_low.splats.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collision_box_generated() {
        let dir = std::env::temp_dir().join("ochroma_test_collision");
        std::fs::create_dir_all(&dir).unwrap();
        let ply_path = dir.join("test.ply");
        let mut f = std::fs::File::create(&ply_path).unwrap();
        write!(
            f,
            "ply\nformat ascii 1.0\nelement vertex 10\nproperty float x\nend_header\n"
        )
        .unwrap();
        for i in 0..10 {
            writeln!(f, "{}", i).unwrap();
        }

        // With collision
        let with_collision = ImportSettings {
            generate_collision: true,
            collision_type: CollisionGenType::BoundingBox,
            ..Default::default()
        };
        let result = import_asset(&ply_path, &with_collision).unwrap();
        assert!(result.collision_box.is_some());

        // Without collision
        let without_collision = ImportSettings {
            generate_collision: true,
            collision_type: CollisionGenType::None,
            ..Default::default()
        };
        let result = import_asset(&ply_path, &without_collision).unwrap();
        assert!(result.collision_box.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_unsupported_format_error() {
        let result = import_asset(Path::new("test.xyz"), &ImportSettings::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported format"));
    }

    #[test]
    fn test_default_settings_sensible() {
        let settings = ImportSettings::default();
        assert!(settings.generate_collision);
        assert_eq!(settings.collision_type, CollisionGenType::BoundingBox);
        assert!(settings.extract_materials);
        assert!(settings.extract_skeleton);
        assert!(settings.extract_animations);
        assert!(settings.splat_density > 0.0);
        assert!((settings.scale_factor - 1.0).abs() < f32::EPSILON);
        assert_eq!(settings.rotation_offset, [0.0; 3]);
    }
}
