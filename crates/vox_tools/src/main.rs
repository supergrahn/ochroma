mod turnaround;
pub mod build;

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use turnaround::run_turnaround;
use build::{BuildTarget, BuildConfig, BuildManifest};

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
