use std::path::Path;
use thiserror::Error;
use uuid::Uuid;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{VxmFile, VxmHeader, MaterialType};

#[derive(Debug, Error)]
pub enum TurnaroundError {
    #[error("views path does not exist: {0}")]
    ViewsNotFound(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("vxm write error: {0}")]
    Vxm(#[from] vox_data::vxm::VxmError),
}

/// Run a turnaround pass over a set of input views, producing a .vxm output.
///
/// Phase 1 placeholder: generates synthetic splats since real 3DGS reconstruction
/// is not yet available.
pub fn run_turnaround(
    views: &Path,
    output: &Path,
    material_map: Option<&str>,
) -> Result<usize, TurnaroundError> {
    if !views.exists() {
        return Err(TurnaroundError::ViewsNotFound(views.display().to_string()));
    }

    let material = material_map.unwrap_or("generic");
    let mat_type = match material {
        "concrete" => MaterialType::Concrete,
        "glass" => MaterialType::Glass,
        "vegetation" => MaterialType::Vegetation,
        "metal" => MaterialType::Metal,
        "water" => MaterialType::Water,
        _ => MaterialType::Generic,
    };

    // Phase 1 placeholder: generate synthetic unit-cube splats
    let splat_count = 64usize;
    let splats: Vec<GaussianSplat> = (0..splat_count)
        .map(|i| {
            let t = i as f32 / splat_count as f32;
            GaussianSplat {
                position: [t - 0.5, 0.0, (t * 2.0 - 1.0).sin() * 0.5],
                scale: [0.05, 0.05, 0.05],
                rotation: [0, 0, 0, 32767],
                opacity: 200,
                _pad: [0; 3],
                spectral: [0; 8],
            }
        })
        .collect();

    let uuid = Uuid::new_v4();
    let header = VxmHeader::new(uuid, splats.len() as u32, mat_type);
    let vxm = VxmFile { header, splats };

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = std::fs::File::create(output)?;
    vxm.write(file)?;

    Ok(splat_count)
}
