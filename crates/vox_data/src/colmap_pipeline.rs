//! COLMAP photogrammetry subprocess wrapper.
//!
//! Spawns the `colmap` binary, runs sparse reconstruction from an image directory,
//! reads the resulting points3D.txt point cloud, applies Smits RGB→spectral upsampling,
//! and produces a VXM v3 file with per-splat spectral material IDs.
//!
//! Requires `colmap` to be installed and on PATH. Returns Err if not found.

use std::path::Path;
use std::process::Command;
use half::f16;
use thiserror::Error;
use vox_core::types::GaussianSplat;

use crate::spectral_upsampler::{SpectralMaterialDb, SpectralUpsampler};

#[derive(Debug, Error)]
pub enum ColmapError {
    #[error("colmap not found on PATH — install COLMAP: https://colmap.github.io")]
    NotFound,
    #[error("colmap subprocess failed (exit {code}): {stderr}")]
    SubprocessFailed { code: i32, stderr: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("points3D.txt parse error at line {line}: {msg}")]
    ParseError { line: usize, msg: String },
    #[error("vxm write error: {0}")]
    Vxm(#[from] crate::vxm::VxmError),
}

/// A point from the COLMAP sparse reconstruction.
#[derive(Debug, Clone)]
pub struct ColmapPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

pub struct ColmapPipeline;

impl ColmapPipeline {
    /// Run the full COLMAP sparse reconstruction pipeline.
    ///
    /// - `image_dir`: directory containing the input photographs.
    /// - `work_dir`: temporary working directory (created if absent).
    /// - `output_vxm`: path to write the resulting VXM v3 file.
    pub fn run(image_dir: &Path, work_dir: &Path, output_vxm: &Path) -> Result<(), ColmapError> {
        Self::check_colmap_available()?;

        std::fs::create_dir_all(work_dir)?;
        let db_path = work_dir.join("colmap.db");
        let sparse_dir = work_dir.join("sparse");
        let txt_dir = work_dir.join("sparse_txt");
        std::fs::create_dir_all(&sparse_dir)?;
        std::fs::create_dir_all(&txt_dir)?;

        Self::run_colmap(&[
            "feature_extractor",
            "--database_path", db_path.to_str().unwrap(),
            "--image_path", image_dir.to_str().unwrap(),
        ])?;
        Self::run_colmap(&[
            "exhaustive_matcher",
            "--database_path", db_path.to_str().unwrap(),
        ])?;
        Self::run_colmap(&[
            "mapper",
            "--database_path", db_path.to_str().unwrap(),
            "--image_path", image_dir.to_str().unwrap(),
            "--output_path", sparse_dir.to_str().unwrap(),
        ])?;
        let model_dir = sparse_dir.join("0");
        Self::run_colmap(&[
            "model_converter",
            "--input_path", model_dir.to_str().unwrap(),
            "--output_path", txt_dir.to_str().unwrap(),
            "--output_type", "TXT",
        ])?;

        let points3d_path = txt_dir.join("points3D.txt");
        let points = Self::parse_points3d(&points3d_path)?;
        let (splats, material_ids) = Self::points_to_splats(&points);
        let vxm = crate::vxm::VxmFileV3 { splats, material_ids, spectral_level: 1 };
        let file = std::fs::File::create(output_vxm)?;
        vxm.write(std::io::BufWriter::new(file))?;

        Ok(())
    }

    /// Parse COLMAP points3D.txt format.
    /// Expected line format: POINT3D_ID X Y Z R G B ERROR TRACK[]
    pub fn parse_points3d(path: &Path) -> Result<Vec<ColmapPoint>, ColmapError> {
        let text = std::fs::read_to_string(path)?;
        let mut points = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 7 {
                return Err(ColmapError::ParseError {
                    line: line_no,
                    msg: format!("expected ≥7 columns, got {}", cols.len()),
                });
            }
            let parse_f32 = |s: &str, field: &str| -> Result<f32, ColmapError> {
                s.parse::<f32>().map_err(|_| ColmapError::ParseError {
                    line: line_no,
                    msg: format!("cannot parse {} as f32: {}", field, s),
                })
            };
            let parse_u8 = |s: &str, field: &str| -> Result<u8, ColmapError> {
                s.parse::<u8>().map_err(|_| ColmapError::ParseError {
                    line: line_no,
                    msg: format!("cannot parse {} as u8: {}", field, s),
                })
            };
            points.push(ColmapPoint {
                x: parse_f32(cols[1], "X")?,
                y: parse_f32(cols[2], "Y")?,
                z: parse_f32(cols[3], "Z")?,
                r: parse_u8(cols[4], "R")?,
                g: parse_u8(cols[5], "G")?,
                b: parse_u8(cols[6], "B")?,
            });
        }
        Ok(points)
    }

    /// Convert point cloud to GaussianSplats with Smits spectral upsampling.
    pub fn points_to_splats(points: &[ColmapPoint]) -> (Vec<GaussianSplat>, Vec<u16>) {
        let mut splats = Vec::with_capacity(points.len());
        let mut material_ids = Vec::with_capacity(points.len());

        for p in points {
            let r = p.r as f32 / 255.0;
            let g = p.g as f32 / 255.0;
            let b = p.b as f32 / 255.0;
            let spectral_f32 = SpectralUpsampler::from_rgb(r, g, b);

            // Classify to nearest material
            let mat = SpectralMaterialDb::classify(&spectral_f32);
            let mat_id = SpectralMaterialDb::MATERIALS
                .iter()
                .position(|m| m.name == mat.name)
                .map_or(0u16, |i| (i + 1) as u16);

            // Store spectral as f16 bits (matches GaussianSplat internal format)
            let spectral: [u16; GaussianSplat::BANDS] =
                std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits());
            let splat = GaussianSplat::surface(
                [p.x, p.y, p.z],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                0.01, 0.01,
                200,
                spectral,
            );

            splats.push(splat);
            material_ids.push(mat_id);
        }

        (splats, material_ids)
    }

    fn check_colmap_available() -> Result<(), ColmapError> {
        Command::new("colmap")
            .arg("--version")
            .output()
            .map_err(|_| ColmapError::NotFound)?;
        Ok(())
    }

    fn run_colmap(args: &[&str]) -> Result<(), ColmapError> {
        let output = Command::new("colmap").args(args).output()?;
        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(ColmapError::SubprocessFailed { code, stderr });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_POINTS3D: &str = "# 3D point list with one line of data per point:\n\
        #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n\
        1 0.5 1.0 2.0 120 80 40 0.5 1 0 2 1\n\
        2 -1.0 0.5 0.3 60 120 60 0.3 1 2 2 3\n\
        3 0.0 0.0 0.0 200 200 200 0.1 1 4\n";

    #[test]
    fn parse_points3d_extracts_positions() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_points3D.txt");
        std::fs::write(&path, SAMPLE_POINTS3D).unwrap();
        let points = ColmapPipeline::parse_points3d(&path).unwrap();
        assert_eq!(points.len(), 3);
        println!("points[0].x = {}, r = {}", points[0].x, points[0].r);
        assert!((points[0].x - 0.5).abs() < 1e-5, "points[0].x = {}, expected 0.5", points[0].x);
        assert!((points[1].y - 0.5).abs() < 1e-5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parse_points3d_extracts_rgb() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_points3D_rgb.txt");
        std::fs::write(&path, SAMPLE_POINTS3D).unwrap();
        let points = ColmapPipeline::parse_points3d(&path).unwrap();
        assert_eq!(points[0].r, 120, "r = {}, expected 120", points[0].r);
        assert_eq!(points[0].g, 80);
        assert_eq!(points[0].b, 40);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parse_points3d_skips_comment_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_points3D_comments.txt");
        std::fs::write(&path, SAMPLE_POINTS3D).unwrap();
        let points = ColmapPipeline::parse_points3d(&path).unwrap();
        assert_eq!(points.len(), 3, "comment lines should be skipped, got {} points", points.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn points_to_splats_assigns_spectral() {
        let pts = vec![ColmapPoint { x: 0.0, y: 0.0, z: 0.0, r: 200, g: 80, b: 40 }];
        let (splats, mat_ids) = ColmapPipeline::points_to_splats(&pts);
        assert_eq!(splats.len(), 1);
        assert_eq!(mat_ids.len(), 1);
        // All 16 spectral bands should be set (stored as f16 bits — nonzero means != 0u16)
        let any_nonzero = splats[0].spectral().iter().any(|&v| v != 0);
        assert!(any_nonzero, "spectral bands must be populated from Smits upsampling");
    }

    #[test]
    fn points_to_splats_assigns_valid_material_id() {
        let pts = vec![ColmapPoint { x: 0.0, y: 0.0, z: 0.0, r: 30, g: 140, b: 30 }];
        let (_, mat_ids) = ColmapPipeline::points_to_splats(&pts);
        assert!(mat_ids[0] > 0, "material ID should be nonzero — unclassified means wrong");
        assert!(
            mat_ids[0] <= SpectralMaterialDb::MATERIALS.len() as u16,
            "material ID {} out of database range",
            mat_ids[0]
        );
    }

    #[test]
    fn points_to_splats_white_point_classifies_as_snow_or_concrete() {
        let pts = vec![ColmapPoint { x: 0.0, y: 0.0, z: 0.0, r: 230, g: 230, b: 235 }];
        let (_, mat_ids) = ColmapPipeline::points_to_splats(&pts);
        let name = SpectralMaterialDb::find_by_id(mat_ids[0]).unwrap().name;
        assert!(
            name == "snow" || name == "concrete" || name == "glass",
            "bright white point should classify as snow, concrete, or glass, got {}",
            name
        );
    }
}
