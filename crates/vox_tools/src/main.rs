use std::path::PathBuf;
use clap::{Parser, Subcommand};
use vox_tools::turnaround::run_turnaround;
use vox_tools::build::{BuildTarget, BuildConfig, BuildManifest};
use vox_tools::gltf2splat::{gltf2splat, Gltf2SplatConfig};
use vox_tools::splats2gltf::{splats2gltf, gltf2splats_import};

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

    /// Convert a GLB/GLTF mesh into surfel-style 2DGS splats (.vxm) by
    /// area-weighted surface sampling. Color from material base-color factor
    /// and vertex colors (textures skipped); RGB→16-band spectral via Smits.
    Gltf2splat {
        /// Input .glb / .gltf file.
        input: PathBuf,
        /// Output .vxm file.
        output: PathBuf,
        /// Target splat density in splats per square unit (m^-2).
        #[arg(long, default_value_t = 256.0)]
        density: f32,
    },

    /// Export a .vxm or .ply to a glTF carrying the KHR_gaussian_splatting
    /// extension (splat attributes on a POINTS primitive).
    Splats2gltf {
        /// Input .vxm or .ply file.
        input: PathBuf,
        /// Output .gltf file.
        output: PathBuf,
    },

    /// Import a glTF carrying the KHR_gaussian_splatting extension (POINTS
    /// primitive) back into a .vxm splat asset.
    Gltf2splats {
        /// Input .gltf / .glb file carrying KHR_gaussian_splatting.
        input: PathBuf,
        /// Output .vxm file.
        output: PathBuf,
    },

    /// Import a composed USD scene (.usdc/.usd) and print its stats:
    /// meshes→2DGS splats, PointInstancer→3DGS splats, lights, camera.
    #[command(name = "usd-import")]
    UsdImport {
        /// Path to the input .usdc / .usd (binary) or .usda (text) file.
        file: PathBuf,
    },

    /// Importance-prune a splat asset (any of .vxm/.ply/.spz) into a smaller
    /// .vxm by dropping low-importance Gaussians. A render guard backs off if
    /// pruning visibly hollows the scene. OFFLINE/asset-time optimization.
    Prune {
        /// Input splat asset (.vxm, .ply, or .spz).
        input: PathBuf,
        /// Output .vxm file.
        output: PathBuf,
        /// Target keep-fraction in [0,1] (e.g. 0.5 keeps the most-important half).
        #[arg(long, default_value_t = 0.5)]
        keep: f32,
        /// Render-guard bound: max mean per-pixel abs RGB diff in [0,1].
        #[arg(long, default_value_t = 0.05)]
        max_pixel_diff: f32,
        /// Disable the render guard (apply the requested keep-fraction exactly).
        #[arg(long, default_value_t = false)]
        no_guard: bool,
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

fn light_kind_name(kind: vox_usd::UsdLightKind) -> &'static str {
    match kind {
        vox_usd::UsdLightKind::Sphere => "SphereLight",
        vox_usd::UsdLightKind::Rect => "RectLight",
        vox_usd::UsdLightKind::Disk => "DiskLight",
        vox_usd::UsdLightKind::Distant => "DistantLight",
        vox_usd::UsdLightKind::Dome => "DomeLight",
    }
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
        Commands::Gltf2splat { input, output, density } => {
            let config = Gltf2SplatConfig { density, ..Gltf2SplatConfig::default() };
            match gltf2splat(&input, &output, config) {
                Ok(count) => {
                    println!(
                        "gltf2splat: wrote {} splats to {}",
                        count,
                        output.display()
                    );
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Splats2gltf { input, output } => {
            match splats2gltf(&input, &output) {
                Ok(count) => {
                    println!(
                        "splats2gltf: wrote {} splats (KHR_gaussian_splatting) to {}",
                        count,
                        output.display()
                    );
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Gltf2splats { input, output } => {
            match gltf2splats_import(&input, &output) {
                Ok(count) => {
                    println!(
                        "gltf2splats: imported {} splats (KHR_gaussian_splatting) to {}",
                        count,
                        output.display()
                    );
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::UsdImport { file } => {
            match vox_usd::import_usd(&file) {
                Ok(imp) => {
                    let fname = file
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?");
                    // metersPerUnit printed without trailing zeros when integral.
                    let mpu = imp.meters_per_unit;
                    let mpu_str = if (mpu.fract()).abs() < 1e-9 {
                        format!("{}", mpu as i64)
                    } else {
                        format!("{mpu}")
                    };
                    println!(
                        "[usd] opened stage: {fname} (upAxis={}, metersPerUnit={mpu_str})",
                        imp.up_axis,
                    );
                    for g in &imp.geom_log {
                        println!(
                            "[usd] {:<8} {:<12} -> {} splats",
                            g.path, g.type_name, g.splats,
                        );
                    }
                    for l in &imp.lights {
                        let intensity = if l.intensity.fract().abs() < 1e-6 {
                            format!("{}", l.intensity as i64)
                        } else {
                            format!("{}", l.intensity)
                        };
                        println!(
                            "[usd] /{:<7} {:<12} color=({:.2},{:.2},{:.2}) intensity={intensity}",
                            l.name,
                            light_kind_name(l.kind),
                            l.color[0],
                            l.color[1],
                            l.color[2],
                        );
                    }
                    if let Some(c) = &imp.camera {
                        println!(
                            "[usd] /{:<7} {:<12} fovY={:.1}deg pos=({:.1},{:.1},{:.1})",
                            c.name, "Camera", c.fov_y_deg, c.position.x, c.position.y, c.position.z,
                        );
                    }
                    println!(
                        "[usd] import OK: {} mesh, {} light, {} camera, {} splats, {} warnings",
                        imp.stats.meshes,
                        imp.lights.len(),
                        imp.camera.is_some() as usize,
                        imp.splats.len(),
                        imp.warnings.len(),
                    );
                    for w in &imp.warnings {
                        eprintln!("[usd] warning: {w}");
                    }
                }
                Err(e) => {
                    eprintln!("[usd] import failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Prune { input, output, keep, max_pixel_diff, no_guard } => {
            match vox_tools::prune::run_prune(&input, &output, keep, max_pixel_diff, no_guard) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error: {e:#}");
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
