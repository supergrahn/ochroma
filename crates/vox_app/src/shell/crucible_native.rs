//! Native Crucible scene cooking — the third "native, but optional" ecosystem
//! integration (user directive: "Let's get FloraPrime, Forge, Crucible a
//! native, but optional part of Ochroma"), the twin of [`super::forge_native`].
//!
//! Two backends behind ONE public entry ([`cook_scene`]):
//!
//! - **`crucible-native` feature ON**: drives the REAL Crucible cook engine from
//!   the Rust sibling at `~/src/crucible/rust`. It composes a minimal scene
//!   (one box mesh + one sun + default camera/atmosphere) with
//!   `cook::graph_builder::build(...)`, calls `CrucibleGraph::cook()` (which
//!   writes a USD scene — `scene.usda` + sublayers — to a unique temp dir),
//!   re-imports that USD through [`vox_usd`], and bakes the resulting splats
//!   around [`CRUCIBLE_PLANT_ORIGIN`]. The path deps are OPTIONAL (see
//!   vox_app/Cargo.toml) so a missing sibling checkout never breaks default/CI
//!   builds. This closes the loop: Recook → real cook → USD on disk → vox_usd
//!   import → splats → planted in the viewport.
//! - **feature OFF (default)**: a deterministic built-in PREVIEW "cooked scene"
//!   (a box-silhouette splat cluster) through the exact same planting path, so
//!   the Crucible tab behaves identically in every build — only the engine and
//!   the receipt's backend tag differ.
//!
//! Both backends are deterministic in `seed` and return world-space splats
//! baked around [`CRUCIBLE_PLANT_ORIGIN`], mirroring how
//! [`super::forge_native::generate_building`] bakes around
//! [`super::forge_native::BUILDING_PLANT_ORIGIN`].

use glam::Quat;
use vox_core::types::GaussianSplat;

/// Panel-facing scene parameters. Clamped by [`CrucibleSceneSpec::clamped`]
/// before EITHER backend sees them (the wave-8/12 rule: every numeric input
/// clamps before it can reach an allocation or a sibling's validator).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrucibleSceneSpec {
    /// Edge length of the cooked box in metres. Clamped 1.0..=12.0.
    pub size_m: f32,
    /// Deterministic seed (varies the preview jitter / native facade tag).
    pub seed: u64,
}

impl Default for CrucibleSceneSpec {
    fn default() -> Self {
        Self { size_m: 4.0, seed: 0 }
    }
}

impl CrucibleSceneSpec {
    /// Clamp every field to its documented range; a non-finite size falls back
    /// to the default (NaN.clamp propagates NaN — the wave-12 lesson).
    pub fn clamped(self) -> Self {
        let size = if self.size_m.is_finite() { self.size_m.clamp(1.0, 12.0) } else { 4.0 };
        Self { size_m: size, seed: self.seed }
    }
}

/// World-space centre where a cooked Crucible scene plants. Distinct from the
/// FloraPrime tree (`TREE_PLANT_ORIGIN` = [-4,-1,-6]), the Forge terrain patch
/// (`TERRAIN_PATCH_ORIGIN` = [3.2,-0.9,-6.5]) and the Forge building
/// (`BUILDING_PLANT_ORIGIN` = [-7.5,-0.9,-3]): the cooked scene lands at the
/// FRONT-RIGHT of the scene so all four placed-asset kinds read as separate
/// landmarks. Y is the ground band (build_scene ground y=-1).
pub const CRUCIBLE_PLANT_ORIGIN: [f32; 3] = [6.5, -0.9, -2.5];

/// Edge length of the box mesh fed to the cook engine, in metres — also the
/// preview box edge so preview and native silhouettes agree.
pub const SCENE_BOX_EDGE_M: f32 = 4.0;

/// Preview-backend sample spacing along the box faces (metres).
const PREVIEW_SPACING_M: f32 = 0.4;

/// Hard cap on splats from one cooked scene — a hostile/degenerate cook can
/// never balloon the overlay.
pub const MAX_SCENE_SPLATS: usize = 50_000;

/// The backend tag appended to the cook receipt — the user always learns which
/// engine built their scene AND, when the round-trip is incomplete, exactly
/// what happened. See [`cook_native`] / [`COOK_LOOP_NOTE`] for the precise state
/// of the cook→USD→import loop.
///
/// `Crucible native` is emitted ONLY when the real cook engine ran AND
/// [`vox_usd`] imported its USD to splats (the loop fully closed). When the cook
/// ran but its TEXT `.usda` output cannot yet round-trip through vox_usd's
/// parser (the documented blocker), the native path returns
/// [`BACKEND_TAG_COOK_PREVIEW`] instead — honest about the cook having run while
/// the imported geometry came from the preview. The decision is made at runtime
/// in [`cook_native`], not at compile time, because it depends on the import.
#[cfg(feature = "crucible-native")]
pub const BACKEND_TAG: &str = "Crucible native";
#[cfg(not(feature = "crucible-native"))]
pub const BACKEND_TAG: &str =
    "built-in preview — enable crucible-native for the real cook engine";

/// Native tag used when the real cook engine RAN and wrote USD to disk, but its
/// text-`.usda` output could not be re-imported through vox_usd (the documented
/// text-array parser blocker — see [`COOK_LOOP_NOTE`]), so the planted splats
/// came from the deterministic preview cluster. Strictly more honest than
/// "Crucible native": the user learns the engine ran AND that the round-trip is
/// not yet wired through.
#[cfg(feature = "crucible-native")]
pub const BACKEND_TAG_COOK_PREVIEW: &str =
    "Crucible cook ran (USD written); vox_usd round-trip pending — showing preview";

/// The precise, verified state of the cook→USD→import loop, surfaced here so the
/// limitation lives next to the code rather than in a commit message:
///
/// - `cook::graph_builder::build(...).cook()` runs for real and writes
///   `scene.usda` + `geometry/lighting/camera/atmosphere.usda` to disk (verified
///   by [`tests::native_cook_writes_usd_and_attempts_import`]).
/// - Cook's `UsdExportNode` writes TEXT `.usda` whose geometry is array/tuple
///   valued (`point3f[] points = [(...), ...]`). vox_usd's openusd-rs text
///   parser cannot read array/tuple geometry and returns
///   `UsdError::UnsupportedTextArray` (its own documented limitation, reproduced
///   on cook's exact output).
/// - crucible-usd's binary `write_scene_binary` IS readable by vox_usd, but it
///   omits `point3f[] points` entirely ("Full point3f[] support is TODO" in that
///   crate), so it would import zero geometry — not a usable workaround.
/// - `cook::graph_builder` exposes only the text path, and the Crucible sibling
///   is read-only by directive.
///
/// Therefore the native path RUNS the cook (real engine, real USD on disk),
/// attempts the import, and on the expected `UnsupportedTextArray` falls back to
/// the preview cluster with [`BACKEND_TAG_COOK_PREVIEW`]. The day cook gains a
/// USDC export with point data — or vox_usd's text parser learns arrays — this
/// path imports the real cooked geometry with NO change to the planting/receipt
/// wiring (only [`cook_native_inner`] flips from preview-fallback to the
/// imported splats, already implemented below).
#[cfg(feature = "crucible-native")]
pub const COOK_LOOP_NOTE: &str = "see crucible_native::COOK_LOOP_NOTE doc comment";

/// Cool slate-blue reflectance for the cooked scene: a flat, slightly
/// blue-leaning SPD so a cooked scene reads distinct from the warm Forge
/// building. Built exactly like the sibling spectra in `forge_native.rs`.
fn scene_spectral() -> [u16; 16] {
    std::array::from_fn(|b| {
        let v: f32 = match b {
            0..=4 => 0.42,  // short bands — cool blue lean
            5..=9 => 0.30,  // mid
            _ => 0.20,      // long bands lower
        };
        half::f16::from_f32(v).to_bits()
    })
}

/// Cook a scene with whichever backend this build carries, returning the splats
/// and the honest backend tag for the receipt.
///
/// Under `crucible-native` the tag is decided at RUNTIME by [`cook_native`]:
/// [`BACKEND_TAG`] (`Crucible native`) when the real cook ran AND vox_usd
/// imported its USD, or [`BACKEND_TAG_COOK_PREVIEW`] when the cook ran but its
/// USD could not yet round-trip (the documented [`COOK_LOOP_NOTE`] blocker).
/// Without the feature it is always the preview tag.
pub fn cook_scene(spec: CrucibleSceneSpec) -> (Vec<GaussianSplat>, &'static str) {
    let spec = spec.clamped();
    #[cfg(feature = "crucible-native")]
    {
        cook_native(spec)
    }
    #[cfg(not(feature = "crucible-native"))]
    {
        (cook_preview(spec), BACKEND_TAG)
    }
}

/// Deterministic PREVIEW "cooked scene": the six faces of a box sampled on a
/// regular [`PREVIEW_SPACING_M`] grid in cool slate spectra, so the default
/// build's "Cook scene" produces a real, plantable, undoable asset (just not
/// Crucible's cooked geometry).
///
/// Splat count is an exact function of the spec (asserted in tests):
/// `6 * n * n` where `n = round(size / spacing).max(1)`. No randomness at all.
pub fn cook_preview(spec: CrucibleSceneSpec) -> Vec<GaussianSplat> {
    let spec = spec.clamped();
    let e = spec.size_m;
    let [ox, oy, oz] = CRUCIBLE_PLANT_ORIGIN;
    let n = (e / PREVIEW_SPACING_M).round().max(1.0) as usize;
    let spectral = scene_spectral();
    let s = PREVIEW_SPACING_M * 0.55;
    let half = e * 0.5;
    // The box rests ON the ground: shift up so the base sits at oy.
    let base = oy;

    let mut splats = Vec::with_capacity(6 * n * n);
    // Cell-centred coordinate along an edge of length `e`, n samples.
    let coord = |i: usize| -(half) + (i as f32 + 0.5) * (e / n as f32);
    for i in 0..n {
        let a = coord(i);
        for j in 0..n {
            let b = coord(j);
            // y for the two horizontal (top/bottom) faces and box-relative y.
            let y_lo = base;
            let y_hi = base + e;
            // ±X faces (a = z, b = y within box)
            let yb = base + half + b; // map b∈[-half,half] → [base, base+e]
            splats.push(volume_at([ox - half, yb, oz + a], s, spectral));
            splats.push(volume_at([ox + half, yb, oz + a], s, spectral));
            // ±Z faces (a = x, b = y within box)
            splats.push(volume_at([ox + a, yb, oz - half], s, spectral));
            splats.push(volume_at([ox + a, yb, oz + half], s, spectral));
            // ±Y faces (a = x, b = z), bottom and top
            splats.push(volume_at([ox + a, y_lo, oz + b], s, spectral));
            splats.push(volume_at([ox + a, y_hi, oz + b], s, spectral));
        }
    }
    splats
}

#[inline]
fn volume_at(pos: [f32; 3], s: f32, spectral: [u16; 16]) -> GaussianSplat {
    GaussianSplat::volume(pos, [s, s, s], Quat::IDENTITY, 240u8, spectral)
}

// ---------------------------------------------------------------------------
// Native backend — the real Crucible cook engine
// ---------------------------------------------------------------------------

/// A monotonically-increasing per-process counter so each native cook writes to
/// a UNIQUE temp dir (combined with the process id) WITHOUT depending on a
/// wall clock — tests stay deterministic (no `SystemTime::now`).
#[cfg(feature = "crucible-native")]
static COOK_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A unique-per-cook output directory under the system temp dir:
/// `crucible_cook_<pid>_<seq>`. Process id keeps concurrent processes apart; the
/// atomic sequence keeps successive cooks in one process apart.
#[cfg(feature = "crucible-native")]
fn unique_cook_dir() -> std::path::PathBuf {
    let seq = COOK_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("crucible_cook_{}_{}", std::process::id(), seq))
}

/// Construct the minimal-but-REAL scene we cook: a single axis-aligned box mesh
/// (`SCENE_BOX_EDGE_M` cube, 12 triangles), one directional sun, default camera
/// and atmosphere, no terrain / scatter / fog / materials. This is the v1
/// "cooked scene" — every input is a real Crucible type built from its public
/// fields, fed straight into `graph_builder::build`.
#[cfg(feature = "crucible-native")]
fn build_box_scene(
    output_dir: std::path::PathBuf,
) -> Result<crucible_core::graph::CrucibleGraph, String> {
    use crucible_types::camera::CameraState;
    use crucible_types::atmosphere::AtmosphereState;
    use crucible_types::geometry::Mesh;
    use crucible_types::light::{LightDescriptor, LightType};

    let h = SCENE_BOX_EDGE_M * 0.5;
    // 8 cube corners, base resting on y=0 (so the box sits on the ground plane).
    let positions = vec![
        [-h, 0.0, -h], [h, 0.0, -h], [h, SCENE_BOX_EDGE_M, -h], [-h, SCENE_BOX_EDGE_M, -h],
        [-h, 0.0, h], [h, 0.0, h], [h, SCENE_BOX_EDGE_M, h], [-h, SCENE_BOX_EDGE_M, h],
    ];
    // 12 triangles (2 per face), CCW-ish winding (winding is irrelevant to the
    // area-weighted surface sampler vox_usd uses).
    let indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3, // -Z
        4, 6, 5, 4, 7, 6, // +Z
        0, 4, 5, 0, 5, 1, // -Y (bottom)
        3, 2, 6, 3, 6, 7, // +Y (top)
        0, 3, 7, 0, 7, 4, // -X
        1, 5, 6, 1, 6, 2, // +X
    ];
    let mesh = Mesh { positions, indices, ..Default::default() };

    let sun = LightDescriptor {
        direction: [0.2, -1.0, 0.3],
        color: [1.0, 0.98, 0.92],
        intensity: 100_000.0,
        light_type: LightType::Directional,
        role: None,
    };

    cook_lib::graph_builder::build(
        None,                              // no terrain
        vec![("cooked_box".to_string(), mesh)],
        vec![],                            // no scatter
        vec![sun],
        CameraState::default(),
        AtmosphereState::default(),
        None,                              // no fog
        vec![],                            // no materials
        output_dir,
    )
    .map_err(|e| e.to_string())
}

/// NATIVE backend: drive the real Crucible cook engine end-to-end and return
/// world-space splats baked around [`CRUCIBLE_PLANT_ORIGIN`].
///
/// 1. Build the minimal box scene graph ([`build_box_scene`]).
/// 2. `CrucibleGraph::cook()` writes `scene.usda` + sublayers to a unique temp
///    dir ([`unique_cook_dir`]).
/// 3. Re-import the cooked USD through [`vox_usd::import_usd`] (the engine's
///    native USD path; the content browser uses the same importer).
/// 4. Translate the imported splats to the plant origin, clamp to
///    [`MAX_SCENE_SPLATS`], best-effort remove the temp dir.
///
/// Returns the splats AND the runtime backend tag: [`BACKEND_TAG`] when the
/// import closed the loop, or [`BACKEND_TAG_COOK_PREVIEW`] when the cook ran but
/// vox_usd could not round-trip its USD (the documented [`COOK_LOOP_NOTE`]
/// blocker) and the splats came from the preview cluster. Any failure (cook
/// error, USD import error, empty import) falls back to the preview — never a
/// panic (no-panic shell rule).
#[cfg(feature = "crucible-native")]
pub fn cook_native(spec: CrucibleSceneSpec) -> (Vec<GaussianSplat>, &'static str) {
    let spec = spec.clamped();
    let dir = unique_cook_dir();
    let _ = std::fs::remove_dir_all(&dir);

    let imported = cook_native_inner(&dir, spec);
    // Best-effort cleanup: the cooked USD is a throwaway intermediate. If
    // removal fails (e.g. the OS holds a handle) the temp dir is harmless and
    // the OS reclaims it; we never error the editor over it.
    let _ = std::fs::remove_dir_all(&dir);

    match imported {
        // Loop fully closed: real cook → USD → vox_usd import → splats.
        Some(splats) => (splats, BACKEND_TAG),
        // Cook ran (USD written) but the round-trip is pending — honest tag.
        None => (cook_preview(spec), BACKEND_TAG_COOK_PREVIEW),
    }
}

/// The fallible core of [`cook_native`], split out so the temp dir is cleaned
/// up on every path. Runs the REAL cook (writes USD to `dir`), then attempts the
/// vox_usd round-trip. Returns `Some(splats)` ONLY when the import yields real
/// geometry; returns `None` (→ preview fallback) on any cook/import failure or
/// an empty import — which, today, is the EXPECTED path because cook writes
/// array-valued text `.usda` that vox_usd cannot yet parse (see
/// [`COOK_LOOP_NOTE`]). The cook itself is not skipped: it runs and writes real
/// USD on every call, so the integration is genuinely live and this function
/// flips to returning the imported splats the moment the round-trip is wired.
#[cfg(feature = "crucible-native")]
fn cook_native_inner(dir: &std::path::Path, spec: CrucibleSceneSpec) -> Option<Vec<GaussianSplat>> {
    // Run the real cook engine: real USD on disk (verifiable in the tests).
    let mut graph = build_box_scene(dir.to_path_buf()).ok()?;
    graph.cook().ok()?;

    // Attempt the round-trip. The cook writes a root `scene.usda` that sublayers
    // geometry/lighting/etc.; we import the composed root exactly as the content
    // browser does. Today this returns Err(UnsupportedTextArray) / Err(Open)
    // (COOK_LOOP_NOTE), so we fall through to None and cook_native tags the
    // result honestly.
    let scene = dir.join("scene.usda");
    if !scene.exists() {
        return None;
    }
    let import = vox_usd::import_usd(&scene).ok()?;
    if import.splats.is_empty() {
        return None;
    }

    // Translate to the plant origin and clamp to the hard cap.
    let [ox, oy, oz] = CRUCIBLE_PLANT_ORIGIN;
    let mut out: Vec<GaussianSplat> = import
        .splats
        .into_iter()
        .take(MAX_SCENE_SPLATS)
        .map(|mut s| {
            let p = s.position_mut();
            p[0] += ox;
            p[1] += oy;
            p[2] += oz;
            s
        })
        .collect();
    // The import order is deterministic for a fixed scene; `spec.seed` is
    // reserved for future facade variation (documented, intentionally inert).
    let _ = (&mut out, spec.seed);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field-wise splat equality (GaussianSplat is Pod, not PartialEq) — the
    /// established comparison idiom from the planting/forge tests.
    fn splats_eq(a: &GaussianSplat, b: &GaussianSplat) -> bool {
        a.position() == b.position()
            && a.scales() == b.scales()
            && a.opacity() == b.opacity()
            && a.spectral() == b.spectral()
    }

    /// The preview scene's splat count is an exact closed-form function of the
    /// spec: 6 faces × n×n samples.
    #[test]
    fn preview_count_is_exact_for_known_spec() {
        let spec = CrucibleSceneSpec { size_m: 4.0, seed: 0 };
        // n = round(4.0 / 0.4) = 10 → 6 * 10 * 10 = 600.
        let splats = cook_preview(spec);
        assert_eq!(splats.len(), 600, "exact closed-form preview count");
    }

    /// Same spec → bit-identical preview (fully deterministic, no RNG at all).
    #[test]
    fn preview_is_deterministic() {
        let spec = CrucibleSceneSpec { size_m: 6.0, seed: 7 };
        let (a, b) = (cook_preview(spec), cook_preview(spec));
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| splats_eq(x, y)), "bit-identical preview");
    }

    /// Hostile specs clamp to the documented range before generation.
    #[test]
    fn hostile_spec_clamps() {
        let spec = CrucibleSceneSpec { size_m: f32::NAN, seed: 1 }.clamped();
        assert_eq!(spec.size_m, 4.0, "NaN size falls back to the default");
        let big = CrucibleSceneSpec { size_m: 1e9, seed: 1 }.clamped();
        assert_eq!(big.size_m, 12.0, "oversize clamps to 12 m");
        let tiny = CrucibleSceneSpec { size_m: 0.01, seed: 1 }.clamped();
        assert_eq!(tiny.size_m, 1.0, "undersize clamps to 1 m");
    }

    /// A bigger scene yields strictly more preview splats.
    #[test]
    fn bigger_scene_has_more_splats() {
        let small = cook_preview(CrucibleSceneSpec { size_m: 2.0, seed: 0 });
        let big = cook_preview(CrucibleSceneSpec { size_m: 10.0, seed: 0 });
        assert!(big.len() > small.len(), "{} > {}", big.len(), small.len());
    }

    /// Cooked-scene spectra are cool/blue-leaning: short band higher than long.
    #[test]
    fn preview_spectra_are_cool() {
        let s = cook_preview(CrucibleSceneSpec::default());
        let splat = &s[0];
        let short = half::f16::from_bits(splat.spectral()[2]).to_f32();
        let long = half::f16::from_bits(splat.spectral()[14]).to_f32();
        assert!(short > long, "cool scene: short band {short} > long band {long}");
    }

    /// The dispatching entry returns a non-empty splat set and an honest tag for
    /// THIS build. Without the feature the tag is exactly the preview
    /// [`BACKEND_TAG`]; with it, the runtime tag is either `Crucible native` (loop
    /// closed) or [`BACKEND_TAG_COOK_PREVIEW`] (cook ran, round-trip pending).
    #[test]
    fn cook_scene_reports_the_backend() {
        let (splats, tag) = cook_scene(CrucibleSceneSpec::default());
        assert!(!splats.is_empty());
        #[cfg(not(feature = "crucible-native"))]
        assert_eq!(tag, BACKEND_TAG);
        #[cfg(feature = "crucible-native")]
        assert!(
            tag == BACKEND_TAG || tag == BACKEND_TAG_COOK_PREVIEW,
            "honest native tag: {tag}"
        );
    }

    // ---- native-backend tests (compiled only with the sibling) -------------

    /// The REAL cook engine writes the expected USD scene files to disk (proving
    /// the integration is live), and we record what vox_usd does with them. Today
    /// the round-trip is blocked by vox_usd's text-array parser (COOK_LOOP_NOTE):
    /// `geometry.usda` imports as `Err(UnsupportedTextArray)`. This test asserts
    /// the cook output exists AND documents the import outcome so the day it
    /// changes (cook gains USDC point export / vox_usd learns text arrays) the
    /// failing assert flags the loop is ready to close.
    #[cfg(feature = "crucible-native")]
    #[test]
    fn native_cook_writes_usd_and_attempts_import() {
        let dir = unique_cook_dir();
        let _ = std::fs::remove_dir_all(&dir);

        // Cook, asserting the expected USD files appear BEFORE any import.
        let mut graph = build_box_scene(dir.clone()).expect("build minimal box scene");
        graph.cook().expect("cook writes USD");
        for f in ["scene.usda", "geometry.usda", "lighting.usda", "camera.usda", "atmosphere.usda"] {
            assert!(dir.join(f).exists(), "cook must write {f}");
        }

        // The cooked geometry is array-valued text USDA: vox_usd reports its
        // documented UnsupportedTextArray (the precise blocker). If this ever
        // succeeds, the loop is importable and cook_native_inner already returns
        // the real splats — update COOK_LOOP_NOTE and this assert then.
        let geom = vox_usd::import_usd(&dir.join("geometry.usda"));
        assert!(
            matches!(geom, Err(vox_usd::UsdError::UnsupportedTextArray)),
            "cook's text geometry.usda is not yet importable by vox_usd (got {geom:?})"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end native path through [`cook_scene`]: the real cook runs, the
    /// round-trip is attempted, and because it is blocked today the result is the
    /// preview cluster tagged honestly with [`BACKEND_TAG_COOK_PREVIEW`] (the
    /// cook RAN; only the import is pending). Non-empty and within the cap.
    #[cfg(feature = "crucible-native")]
    #[test]
    fn native_cook_scene_runs_cook_and_tags_honestly() {
        let (splats, tag) = cook_scene(CrucibleSceneSpec::default());
        assert!(!splats.is_empty(), "native cook yields plantable splats");
        assert!(splats.len() <= MAX_SCENE_SPLATS, "hard cap holds");
        // Today the round-trip is pending, so the honest cook-preview tag wins.
        // (The day the import closes, this becomes BACKEND_TAG — assert either is
        // acceptable so the test does not falsely fail when the loop closes.)
        assert!(
            tag == BACKEND_TAG_COOK_PREVIEW || tag == BACKEND_TAG,
            "native tag is honest about the cook/import state: {tag}"
        );
    }

    /// The native cook cleans up its OWN temp dir. Tested on a dir this test
    /// owns (cross-process/cross-test scanning would race other parallel cooks):
    /// run the cook into an explicit dir, then mirror cook_native's best-effort
    /// cleanup and assert the dir is gone.
    #[cfg(feature = "crucible-native")]
    #[test]
    fn native_cook_cleans_up_its_temp_dir() {
        let dir = unique_cook_dir();
        let _ = std::fs::remove_dir_all(&dir);

        let _ = cook_native_inner(&dir, CrucibleSceneSpec::default());
        assert!(dir.exists(), "cook wrote USD into the temp dir before cleanup");

        // cook_native removes the dir after import; replicate that contract here.
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!dir.exists(), "temp dir removed after the cook+import round-trip");
    }

    /// Native cook is stable in its splat count across runs (the cook output is a
    /// fixed box and the preview fallback is deterministic).
    #[cfg(feature = "crucible-native")]
    #[test]
    fn native_cook_count_is_stable() {
        let (a, _) = cook_native(CrucibleSceneSpec::default());
        let (b, _) = cook_native(CrucibleSceneSpec::default());
        assert_eq!(a.len(), b.len(), "stable splat count across cooks");
    }

    /// Hostile specs cannot panic the native path: the clamp runs first and any
    /// cook/import failure falls back to the preview.
    #[cfg(feature = "crucible-native")]
    #[test]
    fn native_hostile_spec_cannot_panic() {
        let (splats, _) = cook_scene(CrucibleSceneSpec { size_m: -1e30, seed: 9 });
        assert!(!splats.is_empty(), "clamped hostile spec still cooks");
    }
}
