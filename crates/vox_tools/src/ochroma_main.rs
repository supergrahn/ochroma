//! ochroma-tools — Ochroma engine asset pipeline CLI.
//!
//! Usage:
//!   ochroma-tools import --images <dir> --out scene.vxm
//!   ochroma-tools import --gltf model.glb --out scene.vxm

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ochroma-tools", about = "Ochroma engine asset pipeline tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Import assets into VXM format with spectral annotation.
    Import {
        /// Input image directory for COLMAP photogrammetry.
        #[arg(long)]
        images: Option<PathBuf>,
        /// Input GLTF/GLB file.
        #[arg(long)]
        gltf: Option<PathBuf>,
        /// Output .vxm file path.
        #[arg(long)]
        out: PathBuf,
        /// Working directory for COLMAP intermediate files.
        #[arg(long, default_value = "/tmp/ochroma_colmap")]
        work_dir: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Import { images, gltf, out, work_dir } => {
            if let Some(img_dir) = images {
                println!("Running COLMAP photogrammetry pipeline...");
                println!("  Image directory : {}", img_dir.display());
                println!("  Work directory  : {}", work_dir.display());
                println!("  Output          : {}", out.display());
                vox_data::ColmapPipeline::run(&img_dir, &work_dir, &out)
                    .map_err(|e| anyhow::anyhow!("COLMAP pipeline failed: {}", e))?;
                println!("Done. Wrote {}", out.display());
            } else if let Some(gltf_path) = gltf {
                println!("Importing GLTF asset...");
                println!("  Input  : {}", gltf_path.display());
                println!("  Output : {}", out.display());
                let settings = vox_data::ImportSettings::default();
                let result = vox_data::import_asset(&gltf_path, &settings)
                    .map_err(|e| anyhow::anyhow!("GLTF import failed: {}", e))?;
                // Auto-classify spectral material IDs from existing spectral data
                let material_ids: Vec<u16> = result.splats.iter().map(|s| {
                    let spectral: [f32; 16] = std::array::from_fn(|b| s.spectral_f32(b));
                    let mat = vox_data::SpectralMaterialDb::classify(&spectral);
                    vox_data::SpectralMaterialDb::MATERIALS
                        .iter()
                        .position(|m| m.name == mat.name)
                        .map_or(0u16, |i| (i + 1) as u16)
                }).collect();
                let splat_count = result.splats.len();
                let vxm = vox_data::VxmFileV3 {
                    splats: result.splats,
                    material_ids,
                    spectral_level: 1,
                };
                let file = std::fs::File::create(&out)?;
                vxm.write(std::io::BufWriter::new(file))?;
                println!("Done. Wrote {} splats to {}", splat_count, out.display());
            } else {
                anyhow::bail!("Provide either --images or --gltf");
            }
        }
    }
    Ok(())
}
