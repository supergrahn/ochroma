mod turnaround;

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use turnaround::run_turnaround;

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
    }
}
