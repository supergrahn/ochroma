//! GLTF/GLB mesh import — converts triangle meshes to Gaussian splat clouds.
//!
//! **NOTE: This produces REFERENCE QUALITY splat clouds, not production quality.**
//! For production assets, train proper 3DGS from multi-view captures.

use std::path::Path;

use glam::Vec3;
use vox_core::spectral::rgb_to_spectral;
use vox_core::types::GaussianSplat;

/// Result of importing a GLTF/GLB file.
#[derive(Debug)]
pub struct ImportResult {
    pub splats: Vec<GaussianSplat>,
    pub mesh_count: usize,
    pub triangle_count: usize,
    pub vertex_count: usize,
}

/// Errors that can occur during GLTF import.
#[derive(Debug)]
pub enum ImportError {
    IoError(String),
    ParseError(String),
    NoMeshes,
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::NoMeshes => write!(f, "No meshes found in GLTF file"),
        }
    }
}

impl std::error::Error for ImportError {}

/// Import a GLTF/GLB file and convert meshes to Gaussian splats.
///
/// **NOTE: This produces REFERENCE QUALITY splat clouds, not production quality.**
/// For production assets, train proper 3DGS from multi-view captures.
///
/// Each triangle in the mesh is sampled with a number of splats proportional to
/// its surface area (~200 splats per square metre, capped at 50 per triangle).
/// Material base colour is converted to approximate spectral coefficients.
pub fn import_gltf(path: &Path) -> Result<ImportResult, ImportError> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| ImportError::ParseError(e.to_string()))?;

    let mut all_splats = Vec::new();
    let mut mesh_count = 0;
    let mut triangle_count = 0;
    let mut vertex_count = 0;

    for mesh in document.meshes() {
        mesh_count += 1;
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            vertex_count += positions.len();

            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => {
                    // No indices — treat as sequential triangle list
                    (0..positions.len() as u32).collect()
                }
            };

            // Get material colour if available
            let base_color = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();
            let r = base_color[0];
            let g = base_color[1];
            let b = base_color[2];

            // Convert to approximate spectral coefficients
            let spectral = rgb_to_spectral(r, g, b);

            // For each triangle, place splats on the surface
            for tri in indices.chunks(3) {
                if tri.len() < 3 {
                    continue;
                }
                let i0 = tri[0] as usize;
                let i1 = tri[1] as usize;
                let i2 = tri[2] as usize;
                if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                    continue;
                }

                let v0 = Vec3::from(positions[i0]);
                let v1 = Vec3::from(positions[i1]);
                let v2 = Vec3::from(positions[i2]);

                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let normal = edge1.cross(edge2);
                let area = normal.length() * 0.5;
                let _normal = if normal.length() > 1e-8 {
                    normal.normalize()
                } else {
                    Vec3::Y
                };

                triangle_count += 1;

                // Number of splats proportional to triangle area
                // ~200 splats per square metre, minimum 1, maximum 50
                let splat_count = ((area * 200.0).ceil() as usize).max(1).min(50);

                for si in 0..splat_count {
                    // Deterministic barycentric coordinates from index
                    let t = si as f32 / splat_count as f32;
                    let u = ((t * 7.3 + 0.1).fract()).min(0.999);
                    let v = ((t * 13.7 + 0.2).fract()).min(0.999);
                    let (u, v) = if u + v > 1.0 {
                        (1.0 - u, 1.0 - v)
                    } else {
                        (u, v)
                    };

                    let pos = v0 * (1.0 - u - v) + v1 * u + v2 * v;
                    let scale = (area / splat_count as f32).sqrt().max(0.001).min(0.1);

                    all_splats.push(GaussianSplat {
                        position: [pos.x, pos.y, pos.z],
                        scale: [scale, scale * 0.3, scale], // flatten along normal
                        rotation: [0, 0, 0, 32767],         // identity quaternion (simplified)
                        opacity: 240,
                        _pad: [0; 3],
                        spectral,
                    });
                }
            }
        }
    }

    if mesh_count == 0 {
        return Err(ImportError::NoMeshes);
    }

    println!(
        "[gltf] Imported: {} meshes, {} triangles, {} vertices -> {} splats (REFERENCE QUALITY)",
        mesh_count,
        triangle_count,
        vertex_count,
        all_splats.len()
    );

    Ok(ImportResult {
        splats: all_splats,
        mesh_count,
        triangle_count,
        vertex_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_rgb_to_spectral_nonzero_for_colour() {
        let spectral = rgb_to_spectral(0.8, 0.5, 0.3);
        let any_nonzero = spectral.iter().any(|&v| v != 0);
        assert!(any_nonzero, "spectral bands should be non-zero for coloured input");
    }

    #[test]
    fn test_rgb_to_spectral_red_has_high_bands_6_7() {
        let spectral = rgb_to_spectral(1.0, 0.0, 0.0);
        let band6 = f16::from_bits(spectral[6]).to_f32();
        let band7 = f16::from_bits(spectral[7]).to_f32();
        assert!(band6 > 0.9, "red input should have high band 6, got {}", band6);
        assert!(band7 > 0.5, "red input should have high band 7, got {}", band7);
        // Blue bands should be zero for pure red
        let band0 = f16::from_bits(spectral[0]).to_f32();
        let band2 = f16::from_bits(spectral[2]).to_f32();
        assert!(band0 < 0.01, "red input should have ~zero band 0, got {}", band0);
        assert!(band2 < 0.01, "red input should have ~zero band 2, got {}", band2);
    }

    #[test]
    fn test_rgb_to_spectral_green_has_high_band_4() {
        let spectral = rgb_to_spectral(0.0, 1.0, 0.0);
        let band4 = f16::from_bits(spectral[4]).to_f32();
        assert!(band4 > 0.9, "green input should have high band 4, got {}", band4);
        // Red bands should be low
        let band6 = f16::from_bits(spectral[6]).to_f32();
        assert!(band6 < 0.01, "green input should have ~zero band 6, got {}", band6);
    }

    #[test]
    fn test_rgb_to_spectral_black_is_all_zero() {
        let spectral = rgb_to_spectral(0.0, 0.0, 0.0);
        for (i, &v) in spectral.iter().enumerate() {
            assert_eq!(v, 0, "black should produce zero band {}", i);
        }
    }

    #[test]
    fn test_import_error_display() {
        let io_err = ImportError::IoError("file not found".to_string());
        assert_eq!(format!("{}", io_err), "IO error: file not found");

        let parse_err = ImportError::ParseError("bad json".to_string());
        assert_eq!(format!("{}", parse_err), "Parse error: bad json");

        let no_mesh = ImportError::NoMeshes;
        assert_eq!(format!("{}", no_mesh), "No meshes found in GLTF file");
    }

    #[test]
    fn test_import_nonexistent_file_returns_error() {
        let result = import_gltf(Path::new("/tmp/nonexistent_file_12345.glb"));
        assert!(result.is_err(), "importing nonexistent file should error");
        match result.unwrap_err() {
            ImportError::ParseError(_) => {} // gltf crate reports this as a parse/io error
            ImportError::IoError(_) => {}
            other => panic!("expected ParseError or IoError, got: {:?}", other),
        }
    }
}
