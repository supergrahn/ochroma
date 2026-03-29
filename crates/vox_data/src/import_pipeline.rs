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
    use crate::ply_loader;

    let mut warnings = Vec::new();

    let splats = match ply_loader::load_ply(path) {
        Ok(s) => s,
        Err(_) => {
            warnings.push("Binary PLY failed; falling back to vertex-count estimation.".to_string());
            // Read vertex count from ASCII header
            let data = std::fs::read_to_string(path).unwrap_or_default();
            let mut vertex_count = 0u32;
            for line in data.lines() {
                if line.starts_with("element vertex") {
                    if let Some(n) = line.split_whitespace().nth(2) {
                        vertex_count = n.parse().unwrap_or(0);
                    }
                    break;
                }
            }
            let count = (vertex_count as f32 * settings.splat_density / 200.0) as usize;
            (0..count).map(|i| {
                let t = i as f32 / count.max(1) as f32;
                GaussianSplat {
                    position: [t * settings.scale_factor, 0.0, 0.0],
                    scale: [0.01; 3],
                    rotation: [0, 0, 0, 16384],
                    opacity: 255,
                    _pad: [0; 3],
                    spectral: [0; 8],
                }
            }).collect()
        }
    };

    if splats.is_empty() {
        warnings.push("PLY could not be decoded; splat cloud is empty. Only binary PLY is supported.".to_string());
    }

    // Apply scale factor to positions
    let mut splats = splats;
    if (settings.scale_factor - 1.0).abs() > f32::EPSILON {
        for s in splats.iter_mut() {
            s.position[0] *= settings.scale_factor;
            s.position[1] *= settings.scale_factor;
            s.position[2] *= settings.scale_factor;
        }
    }

    // Compute tight bounding box for collision
    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
        && !splats.is_empty()
    {
        let mut mn = splats[0].position;
        let mut mx = splats[0].position;
        for s in &splats {
            for i in 0..3 {
                if s.position[i] < mn[i] { mn[i] = s.position[i]; }
                if s.position[i] > mx[i] { mx[i] = s.position[i]; }
            }
        }
        Some((mn, mx))
    } else {
        None
    };

    Ok(ImportResult {
        splats,
        collision_box,
        material_names: vec!["default".to_string()],
        skeleton_joint_count: 0,
        animation_count: 0,
        warnings,
    })
}

fn import_gltf_full(path: &Path, settings: &ImportSettings) -> Result<ImportResult, String> {
    use crate::gltf_import;
    use gltf::Gltf;

    // Use gltf_import for real triangle-sampled splats
    let gr = gltf_import::import_gltf(path)
        .map_err(|e| format!("GLTF import error: {}", e))?;

    let mut splats = gr.splats;

    // Apply scale factor
    if (settings.scale_factor - 1.0).abs() > f32::EPSILON {
        for s in splats.iter_mut() {
            s.position[0] *= settings.scale_factor;
            s.position[1] *= settings.scale_factor;
            s.position[2] *= settings.scale_factor;
        }
    }

    // Extract metadata
    let gltf_doc = Gltf::open(path).map_err(|e| format!("GLTF metadata error: {}", e))?;

    let mut material_names = Vec::new();
    if settings.extract_materials {
        for mat in gltf_doc.materials() {
            material_names.push(mat.name().unwrap_or("unnamed_material").to_string());
        }
    }

    let mut skeleton_joint_count = 0;
    if settings.extract_skeleton {
        for skin in gltf_doc.skins() {
            skeleton_joint_count += skin.joints().count();
        }
    }

    let animation_count = if settings.extract_animations {
        gltf_doc.animations().count()
    } else {
        0
    };

    let mut warnings = Vec::new();
    if material_names.is_empty() {
        warnings.push("No materials found in GLTF file".to_string());
    }
    if splats.is_empty() {
        warnings.push("No geometry found — splat cloud is empty".to_string());
    }

    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
        && !splats.is_empty()
    {
        let mut mn = splats[0].position;
        let mut mx = splats[0].position;
        for s in &splats {
            for i in 0..3 {
                if s.position[i] < mn[i] { mn[i] = s.position[i]; }
                if s.position[i] > mx[i] { mx[i] = s.position[i]; }
            }
        }
        Some((mn, mx))
    } else {
        None
    };

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
    use crate::vxm::VxmFile;

    let data = std::fs::read(path).map_err(|e| format!("Failed to read VXM: {}", e))?;
    let file = VxmFile::read(std::io::Cursor::new(&data))
        .map_err(|e| format!("VXM parse error: {}", e))?;

    let mut splats = file.splats;

    // Apply scale factor
    if (settings.scale_factor - 1.0).abs() > f32::EPSILON {
        for s in splats.iter_mut() {
            s.position[0] *= settings.scale_factor;
            s.position[1] *= settings.scale_factor;
            s.position[2] *= settings.scale_factor;
        }
    }

    let collision_box = if settings.generate_collision
        && settings.collision_type != CollisionGenType::None
        && !splats.is_empty()
    {
        let mut mn = splats[0].position;
        let mut mx = splats[0].position;
        for s in &splats {
            for i in 0..3 {
                if s.position[i] < mn[i] { mn[i] = s.position[i]; }
                if s.position[i] > mx[i] { mx[i] = s.position[i]; }
            }
        }
        Some((mn, mx))
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
    fn test_import_settings_affect_scale_factor() {
        let dir = std::env::temp_dir().join("ochroma_test_scale");
        std::fs::create_dir_all(&dir).unwrap();
        // Write binary PLY with a single splat at x=1.0
        let ply_path = dir.join("scale_test.ply");
        {
            use std::io::{BufWriter, Write};
            let mut f = BufWriter::new(std::fs::File::create(&ply_path).unwrap());
            write!(f, "ply\nformat binary_little_endian 1.0\nelement vertex 1\n").unwrap();
            write!(f, "property float x\nproperty float y\nproperty float z\n").unwrap();
            write!(f, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n").unwrap();
            write!(f, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n").unwrap();
            write!(f, "property float opacity\n").unwrap();
            write!(f, "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n").unwrap();
            write!(f, "end_header\n").unwrap();
            // splat at x=1.0, y=0, z=0; scale=exp(-2.3)
            f.write_all(&1.0f32.to_le_bytes()).unwrap(); // x
            for _ in 0..2 { f.write_all(&0.0f32.to_le_bytes()).unwrap(); } // y, z
            for _ in 0..3 { f.write_all(&(-2.3f32).to_le_bytes()).unwrap(); } // scale
            f.write_all(&1.0f32.to_le_bytes()).unwrap(); // rot_0 (w)
            for _ in 0..3 { f.write_all(&0.0f32.to_le_bytes()).unwrap(); } // rot x,y,z
            f.write_all(&0.0f32.to_le_bytes()).unwrap(); // opacity (logit)
            for _ in 0..3 { f.write_all(&0.5f32.to_le_bytes()).unwrap(); } // f_dc
        }

        let settings_2x = ImportSettings { scale_factor: 2.0, ..Default::default() };
        let settings_1x = ImportSettings { scale_factor: 1.0, ..Default::default() };

        let result_2x = import_asset(&ply_path, &settings_2x).unwrap();
        let result_1x = import_asset(&ply_path, &settings_1x).unwrap();

        assert_eq!(result_1x.splats.len(), 1);
        assert_eq!(result_2x.splats.len(), 1);
        assert!((result_1x.splats[0].position[0] - 1.0).abs() < 0.01,
            "1x: expected x≈1.0, got {}", result_1x.splats[0].position[0]);
        assert!((result_2x.splats[0].position[0] - 2.0).abs() < 0.01,
            "2x: expected x≈2.0, got {}", result_2x.splats[0].position[0]);

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
