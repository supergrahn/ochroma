//! Compose independently captured CUDA and Vulkan MegaGeometry benchmark
//! artifacts. Raw producers are forbidden from asserting cross-backend
//! equality; this join is the sole authority for that verdict.

#[path = "support/mega_geometry_bench_logic.rs"]
mod logic;

use logic::{BenchReport, compose_backend_conformance};
use std::path::{Path, PathBuf};

struct Args {
    cuda: PathBuf,
    vulkan: PathBuf,
    json_out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut cuda = None;
    let mut vulkan = None;
    let mut json_out = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a path"))?;
        match flag.as_str() {
            "--cuda" => cuda = Some(PathBuf::from(value)),
            "--vulkan" => vulkan = Some(PathBuf::from(value)),
            "--json-out" => json_out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(Args {
        cuda: cuda.ok_or("--cuda is required")?,
        vulkan: vulkan.ok_or("--vulkan is required")?,
        json_out: json_out.ok_or("--json-out is required")?,
    })
}

fn read_report(path: &Path) -> Result<BenchReport, String> {
    let bytes = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    BenchReport::from_json(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let cuda = read_report(&args.cuda)?;
    let vulkan = read_report(&args.vulkan)?;
    let composed = compose_backend_conformance(&cuda, &vulkan);
    let json = serde_json::to_vec_pretty(&composed)
        .map_err(|error| format!("serialize conformance: {error}"))?;
    if let Some(parent) = args.json_out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let temporary = args.json_out.with_extension("json.tmp");
    std::fs::write(&temporary, json)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, &args.json_out).map_err(|error| {
        format!(
            "publish {} -> {}: {error}",
            temporary.display(),
            args.json_out.display()
        )
    })?;
    println!("{}", composed.line());
    for reason in &composed.reasons {
        eprintln!("[mega_geometry_compose] {reason}");
    }
    Ok(composed.passes())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("[mega_geometry_compose] error: {error}");
            std::process::exit(2);
        }
    }
}
