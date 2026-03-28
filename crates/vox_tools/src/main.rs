use std::path::PathBuf;
use clap::{Parser, Subcommand};
use vox_tools::turnaround::run_turnaround;
use vox_tools::build::{BuildTarget, BuildConfig, BuildManifest};

#[derive(Parser)]
#[command(name = "vox_tools", about = "Ochroma engine asset pipeline tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a turnaround capture pipeline to produce a .vxm asset
    Turnaround {
        /// Path to the directory containing input views (images)
        #[arg(long)]
        views: PathBuf,

        /// Output .vxm file path
        #[arg(long)]
        output: PathBuf,

        /// Optional material map hint (e.g. concrete, glass, vegetation, metal, water)
        #[arg(long)]
        material_map: Option<String>,
    },

    /// Import a GLTF/GLB file and convert to .vxm Gaussian splats (REFERENCE QUALITY).
    Import {
        /// Path to the input .gltf or .glb file
        #[arg(long)]
        input: String,
        /// Path to the output .vxm file
        #[arg(long)]
        output: String,
    },

    /// Build the game for a target platform.
    Build {
        #[arg(long, default_value = "linux")]
        target: String,
        #[arg(long, default_value = "release")]
        config: String,
        #[arg(long, default_value = "OchromaCity")]
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Turnaround { views, output, material_map } => {
            match run_turnaround(&views, &output, material_map.as_deref()) {
                Ok(count) => {
                    println!("Turnaround complete: {} splats written to {}", count, output.display());
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Import { input, output } => {
            use std::path::Path;
            match vox_data::gltf_import::import_gltf(Path::new(&input)) {
                Ok(result) => {
                    println!(
                        "Imported: {} splats from {} meshes ({} triangles, {} vertices)",
                        result.splats.len(),
                        result.mesh_count,
                        result.triangle_count,
                        result.vertex_count,
                    );
                    println!("NOTE: This is REFERENCE QUALITY. For production, train proper 3DGS from multi-view captures.");
                    let file = vox_data::vxm::VxmFile {
                        header: vox_data::vxm::VxmHeader::new(
                            uuid::Uuid::new_v4(),
                            result.splats.len() as u32,
                            vox_data::vxm::MaterialType::Generic,
                        ),
                        splats: result.splats,
                    };
                    let mut out = std::fs::File::create(&output).expect("failed to create output file");
                    file.write(&mut out).expect("failed to write VXM file");
                    println!("Saved to: {}", output);
                }
                Err(e) => {
                    eprintln!("Import failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Build { target, config, name } => {
            let target = match target.as_str() {
                "windows" => BuildTarget::Windows,
                "macos" => BuildTarget::MacOS,
                "steamos" => BuildTarget::SteamOS,
                _ => BuildTarget::Linux,
            };
            let config = match config.as_str() {
                "debug" => BuildConfig::Debug,
                "shipping" => BuildConfig::Shipping,
                _ => BuildConfig::Release,
            };
            let manifest = BuildManifest::new(&name, target, config);
            println!("Build command: {}", manifest.build_command());
            println!("Output: {}", manifest.output_binary_name());
        }
    }
}
