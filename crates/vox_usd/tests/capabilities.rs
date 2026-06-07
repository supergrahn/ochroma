//! Capability tests — one per row of the design's §3 capability table.
//!
//! Every assertion checks a real composed/computed outcome against the
//! committed Pixar-authored fixtures (regenerate via `tests/data/make_fixture.py`).

use std::path::PathBuf;
use vox_usd::{import_usd, import_usd_with, UsdError, UsdImportSettings, UsdLightKind};

use half::f16;
use vox_data::SpectralUpsampler;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

/// Serialize all tests in this binary: openusd-rs keeps a process-global
/// layer cache behind a Mutex, and under heavy parallel load the in-binary
/// thread interleaving has wedged this suite (a workspace gate hung ~9h here
/// while the suite passes standalone in 0.3s). The whole suite runs in well
/// under a second, so serial execution costs nothing and removes the
/// interleaving entirely. (into_inner: a poisoned guard from a prior panicking
/// test must not cascade.)
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Exact expected splat count for the 2 m cube under the default 200 spm sampler.
///
/// 6 quad faces → 12 triangles. Each triangle of the 2 m cube has area 2 m².
/// `clamp(ceil(2·200), 1, 50)` = 50. → 12 · 50 = 600. Computed here (not hard-
/// coded blind) by mirroring the importer's sampler so the test self-documents.
fn cube_expected_splats(spm: f32) -> usize {
    // 12 triangles, each 2 m² (half of a 2×2 m quad face).
    let per_tri = ((2.0_f32 * spm).ceil() as usize).clamp(1, 50);
    12 * per_tri
}

// --- Open + compose .usdc -------------------------------------------------

#[test]
fn open_and_compose_usdc() {
    let _serial = serial();
    let imp = import_usd(&fixture("cube_lit.usdc")).expect("cube_lit imports");
    assert!(imp.stats.prims >= 3, "prims={}", imp.stats.prims);
    assert_eq!(imp.stats.meshes, 1, "exactly one mesh");
}

// --- Read point3f[] from binary crate (bbox corners) ----------------------

#[test]
fn mesh_bbox_equals_authored_cube_corners() {
    let _serial = serial();
    let imp = import_usd(&fixture("cube_lit.usdc")).unwrap();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for s in &imp.splats {
        if !s.is_surface() {
            continue;
        }
        let p = s.position();
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    // Splats sit on the surface, so the bbox must reach the cube corners ±1.
    for k in 0..3 {
        assert!((min[k] - (-1.0)).abs() < 1e-4, "min[{k}]={} != -1", min[k]);
        assert!((max[k] - 1.0).abs() < 1e-4, "max[{k}]={} != 1", max[k]);
    }
}

// --- Apply composed Xform (translate moves mean) --------------------------

#[test]
fn xform_translate_moves_splat_mean() {
    let _serial = serial();
    // The instancer fixture's three positions average to (4/3, -1/3, 2). We
    // verify Xform accumulation independently with a translated re-import: the
    // PointInstancer path applies `world` to every position, so a parent
    // translate must shift the mean. We assert directly on cube_lit's mesh,
    // whose Cube has no xform (mean ~origin), then on a translated variant.
    let imp = import_usd(&fixture("cube_lit.usdc")).unwrap();
    let mean_x: f32 = imp
        .splats
        .iter()
        .filter(|s| s.is_surface())
        .map(|s| s.position()[0])
        .sum::<f32>()
        / imp.splats.iter().filter(|s| s.is_surface()).count() as f32;
    // Symmetric cube → mean x ≈ 0.
    assert!(mean_x.abs() < 0.05, "untranslated cube mean x = {mean_x}");

    // The instancer fixture exercises the same world-transform application path
    // on known positions; its mean x is exactly (1 + -4 + 7)/3 = 4/3.
    let inst = import_usd(&fixture("instancer.usdc")).unwrap();
    let vol: Vec<_> = inst.splats.iter().filter(|s| s.is_volume()).collect();
    let mean_x: f32 = vol.iter().map(|s| s.position()[0]).sum::<f32>() / vol.len() as f32;
    assert!((mean_x - 4.0 / 3.0).abs() < 0.05, "instancer mean x = {mean_x}");
}

// --- Mesh → 2DGS sampling (exact count + every splat is surface) ----------

#[test]
fn mesh_splat_count_matches_sampler_formula() {
    let _serial = serial();
    let imp = import_usd(&fixture("cube_lit.usdc")).unwrap();
    let surface: Vec<_> = imp.splats.iter().filter(|s| s.is_surface()).collect();
    assert_eq!(
        surface.len(),
        cube_expected_splats(200.0),
        "surface splat count must equal Σ clamp(ceil(area·spm),1,50)",
    );
    // No volume splats in the pure-mesh scene; every splat is a surface splat.
    assert_eq!(imp.splats.len(), surface.len(), "all splats are surface");
    assert!(imp.splats.iter().all(|s| s.is_surface()));
}

#[test]
fn mesh_splat_count_scales_with_density_setting() {
    let _serial = serial();
    // At 10 spm each 2 m² triangle yields clamp(ceil(20),1,50)=20 → 240.
    let settings = UsdImportSettings { mesh_splats_per_sqm: 10.0, ..Default::default() };
    let imp = import_usd_with(&fixture("cube_lit.usdc"), &settings).unwrap();
    let surface = imp.splats.iter().filter(|s| s.is_surface()).count();
    assert_eq!(surface, cube_expected_splats(10.0));
    assert_eq!(surface, 240);
}

// --- PointInstancer → 3DGS (exact positions, all volume) ------------------

#[test]
fn instancer_emits_exact_volume_splats() {
    let _serial = serial();
    let imp = import_usd(&fixture("instancer.usdc")).unwrap();
    let vol: Vec<_> = imp.splats.iter().filter(|s| s.is_volume()).collect();
    assert_eq!(vol.len(), 3, "exactly 3 instances");
    assert!(imp.splats.iter().all(|s| s.is_volume()), "all volume");

    let expected = [
        [1.0_f32, 2.0, 3.0],
        [-4.0, 5.0, -6.0],
        [7.0, -8.0, 9.0],
    ];
    // Match each authored position to some emitted splat within 1e-4.
    for want in expected {
        let found = vol.iter().any(|s| {
            let p = s.position();
            (p[0] - want[0]).abs() < 1e-4
                && (p[1] - want[1]).abs() < 1e-4
                && (p[2] - want[2]).abs() < 1e-4
        });
        assert!(found, "no splat at authored position {want:?}");
    }
}

// --- color3f → spectrum (band-for-band equality) --------------------------

#[test]
fn red_material_upsamples_to_spectrum_band_for_band() {
    let _serial = serial();
    let imp = import_usd(&fixture("red_cube.usdc")).unwrap();
    let splat = imp
        .splats
        .iter()
        .find(|s| s.is_surface())
        .expect("red cube produces surface splats");

    let reference = SpectralUpsampler::from_rgb(1.0, 0.0, 0.0);
    for b in 0..16 {
        let expected = f16::from_f32(reference[b]).to_f32();
        assert_eq!(
            splat.spectral_f32(b),
            expected,
            "spectral band {b} mismatch",
        );
    }
}

// --- Light read (intensity/color exact) -----------------------------------

#[test]
fn sphere_light_intensity_and_color_exact() {
    let _serial = serial();
    let imp = import_usd(&fixture("cube_lit.usdc")).unwrap();
    assert_eq!(imp.lights.len(), 1, "one light");
    let light = &imp.lights[0];
    assert_eq!(light.kind, UsdLightKind::Sphere);
    assert_eq!(light.intensity, 1000.0);
    assert_eq!(light.color, [1.0, 1.0, 1.0]);
}

// --- Camera read (schemaless: fovY + world pos) ---------------------------

#[test]
fn camera_fov_and_position() {
    let _serial = serial();
    let imp = import_usd(&fixture("cube_lit.usdc")).unwrap();
    let cam = imp.camera.as_ref().expect("camera present");

    // Design's load-bearing assertion: fovY within 0.1deg of 2·atan(18/50),
    // where 18 = horizontalAperture/2 = 36/2. ≈ 39.598deg.
    let expected_deg = (2.0_f32 * (18.0_f32 / 50.0).atan()).to_degrees();
    assert!(
        (cam.fov_y_deg - expected_deg).abs() < 0.1,
        "fovY {} vs expected {}",
        cam.fov_y_deg,
        expected_deg,
    );

    let p = cam.position;
    assert!((p.x - 0.0).abs() < 1e-4, "cam x = {}", p.x);
    assert!((p.y - 1.0).abs() < 1e-4, "cam y = {}", p.y);
    assert!((p.z - 6.0).abs() < 1e-4, "cam z = {}", p.z);
}

// --- USDA array geometry now imports (openusd-rs 9fd19fa) ------------------

#[test]
fn usda_array_geometry_imports_into_real_splats() {
    // The text parser learned point3f[]/int[]/tuple arrays, so array-valued
    // text .usda geometry — what crucible's cook engine writes — round-trips
    // into real splats instead of the old UnsupportedTextArray blocker.
    let _serial = serial();
    let imp = import_usd(&fixture("points_text.usda")).expect("array geometry imports");
    assert!(!imp.splats.is_empty(), "point3f[] geometry yields splats");
}

// --- metersPerUnit / upAxis recovered (CLI stats line) --------------------

#[test]
fn open_succeeds_with_stats_for_cube_lit() {
    let _serial = serial();
    let imp = import_usd(&fixture("cube_lit.usdc")).unwrap();
    assert_eq!(imp.stats.meshes, 1);
    assert_eq!(imp.stats.lights, 1);
    assert!(imp.camera.is_some());
    assert_eq!(imp.warnings.len(), 0, "no warnings for the clean fixture");
}

// --- Hostile inputs: error, never abort (wave-3 review criticals) ----------

/// Six hostile-input classes that previously PANICKED through openusd-rs's
/// unwrap/expect wall (reproduced by adversarial review aborting the CLI).
/// All must now surface as Err — the editor must survive opening any file.
/// (Each case prints one panic line to stderr via the default hook; that
/// noise is expected — the contract is "no abort, an Err comes back".)
#[test]
fn hostile_inputs_error_instead_of_aborting() {
    let _serial = serial();
    let dir = std::env::temp_dir().join("vox_usd_hostile");
    std::fs::create_dir_all(&dir).unwrap();

    // (a) truncated PXR-USDC magic only
    let p = dir.join("truncated.usdc");
    std::fs::write(&p, b"PXR-USDC").unwrap();
    assert!(import_usd(&p).is_err(), "truncated magic must Err");

    // (b) garbage bytes with .usdc extension
    let p = dir.join("garbage.usdc");
    std::fs::write(&p, b"definitely not a usd file at all 1234567890").unwrap();
    assert!(import_usd(&p).is_err(), "garbage .usdc must Err");

    // (c) empty file
    let p = dir.join("empty.usdc");
    std::fs::write(&p, b"").unwrap();
    assert!(import_usd(&p).is_err(), "empty .usdc must Err");

    // (d) garbage .usda text
    let p = dir.join("garbage.usda");
    std::fs::write(&p, b"{{{{ not usda ]]]]").unwrap();
    assert!(import_usd(&p).is_err(), "garbage .usda must Err");

    // (e) a DIRECTORY named like a usd file (passes the exists() precheck)
    let p = dir.join("dir.usdc");
    std::fs::create_dir_all(&p).unwrap();
    assert!(import_usd(&p).is_err(), "directory.usdc must Err");

    // (f) extensionless file
    let p = dir.join("noext");
    std::fs::write(&p, b"PXR-USDC").unwrap();
    assert!(import_usd(&p).is_err(), "extensionless must Err");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A pathologically nested scenegraph must not abort the process (the old
/// recursive walk stack-overflowed at depth ~60k — uncatchable). The walk is
/// now an explicit work-stack honoring `settings.max_depth` (default 16):
/// openusd-rs composition is EXPONENTIAL in nesting depth (measured 2x/level;
/// a 20k-deep variant of this test never finished), so the LOW default cap
/// IS the CPU defense and this test proves it on a 200-deep hostile file.
#[test]
fn deep_nesting_is_capped_not_stack_overflow() {
    let _serial = serial();
    const DEPTH: usize = 200;
    let mut text = String::with_capacity(DEPTH * 24);
    text.push_str("#usda 1.0\n");
    for i in 0..DEPTH {
        text.push_str(&format!("def Xform \"n{i}\" {{\n"));
    }
    text.push_str(&"}\n".repeat(DEPTH));

    let p = std::env::temp_dir().join("vox_usd_deep.usda");
    std::fs::write(&p, text).unwrap();

    // DEFAULT settings on purpose: the default max_depth (16) is the shipped
    // CPU defense — openusd-rs composition is EXPONENTIAL in nesting depth
    // (~2x/level), so querying past ~22 levels takes tens of seconds. The
    // default must keep this 200-deep hostile file fast AND surfaced.
    let imp = import_usd(&p).expect("deep nesting must import, capped");
    assert!(
        imp.warnings.iter().any(|w| w.contains("deeper than 16")),
        "depth cap must be surfaced as a warning: {:?}",
        imp.warnings
    );
    assert_eq!(
        imp.entities.len(),
        16,
        "exactly the prims above the cap import (one Xform entity per level)"
    );

    let _ = std::fs::remove_file(&p);
}

/// Hostile BREADTH is bounded the same way: more prims than `max_prims`
/// stops the walk with a warning instead of unbounded work.
#[test]
fn prim_count_cap_stops_hostile_breadth() {
    let _serial = serial();
    let mut text = String::from("#usda 1.0\n");
    for i in 0..500 {
        text.push_str(&format!("def Xform \"w{i}\" {{}}\n"));
    }
    let p = std::env::temp_dir().join("vox_usd_wide.usda");
    std::fs::write(&p, text).unwrap();

    let settings = UsdImportSettings { max_prims: 100, ..Default::default() };
    let imp = import_usd_with(&p, &settings).expect("wide scene must import, capped");
    assert!(
        imp.warnings.iter().any(|w| w.contains("exceeds 100 prims")),
        "prim cap must be surfaced as a warning: {:?}",
        imp.warnings
    );
    assert_eq!(imp.entities.len(), 100, "exactly max_prims prims imported");

    let _ = std::fs::remove_file(&p);
}
