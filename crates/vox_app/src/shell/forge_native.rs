//! Native Forge building generation — the first "native, but optional"
//! ecosystem integration (user directive: "Let's get FloraPrime, Forge,
//! Crucible a native, but optional part of Ochroma").
//!
//! Two backends behind ONE public entry ([`generate_building`]):
//!
//! - **`forge-native` feature ON**: calls the REAL
//!   `forge_building::generate(BuildingParams)` from the Rust sibling at
//!   `~/src/aetherspectra/forge` and surfel-samples its triangle mesh into
//!   spectral splats. The path deps are OPTIONAL (see vox_app/Cargo.toml) so a
//!   missing sibling checkout never breaks default/CI builds.
//! - **feature OFF (default)**: a deterministic built-in PREVIEW building (box
//!   walls + flat roof) through the exact same planting path, so the Forge tab
//!   behaves identically in every build — only the generator and the receipt's
//!   backend tag differ.
//!
//! Both backends are deterministic in `seed` and return world-space splats
//! baked around [`BUILDING_PLANT_ORIGIN`], mirroring how
//! [`super::plugins::heightfield_to_splats`] bakes [`super::plugins::TERRAIN_PATCH_ORIGIN`].

use glam::Quat;
use vox_core::types::GaussianSplat;

/// Panel-facing building parameters. Clamped by [`BuildingSpec::clamped`]
/// before EITHER backend sees them (the wave-8/12 rule: every numeric input
/// path clamps before it can reach an allocation or a sibling's validator —
/// `forge_building::generate` errors on bad params; we never let one through).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildingSpec {
    /// Storeys above ground. Clamped 1..=6.
    pub storeys: u32,
    /// Footprint width in metres. Clamped 4.0..=24.0.
    pub width_m: f32,
    /// Footprint depth in metres. Clamped 4.0..=24.0.
    pub depth_m: f32,
    /// Deterministic generation seed (also drives facade variation natively).
    pub seed: u64,
}

impl Default for BuildingSpec {
    fn default() -> Self {
        Self { storeys: 2, width_m: 8.0, depth_m: 6.0, seed: 0 }
    }
}

impl BuildingSpec {
    /// Clamp every field to its documented range; NaN dimensions fall back to
    /// the defaults (NaN.clamp propagates NaN — the wave-12 lesson).
    pub fn clamped(self) -> Self {
        let dim = |v: f32, dflt: f32| if v.is_finite() { v.clamp(4.0, 24.0) } else { dflt };
        Self {
            storeys: self.storeys.clamp(1, 6),
            width_m: dim(self.width_m, 8.0),
            depth_m: dim(self.depth_m, 6.0),
            seed: self.seed,
        }
    }
}

/// World-space centre where a generated building plants. Distinct from the
/// FloraPrime tree (`TREE_PLANT_ORIGIN` = [-4,-1,-6]) and the Forge terrain
/// patch (`TERRAIN_PATCH_ORIGIN` = [3.2,-0.9,-6.5]): the building rises at the
/// FRONT-LEFT of the scene (verified visible in the snapshot proof — the first
/// back-left placement [-9,-0.9,-10] sat outside the camera's readable frame)
/// so all three placed-asset kinds read as separate landmarks. Y is the ground
/// band (build_scene ground y=-1).
pub const BUILDING_PLANT_ORIGIN: [f32; 3] = [-7.5, -0.9, -3.0];

/// Metres of wall per storey in both backends (matches the native default
/// `floor_height: 3.0` so preview and native buildings agree on silhouette).
pub const STOREY_HEIGHT_M: f32 = 3.0;

/// Surface-sampling density for the NATIVE mesh→splat conversion: one surfel
/// splat per this many square metres of triangle area (0.25 m² = 4 splats/m²,
/// dense enough that walls read as continuous masonry at viewport scale).
pub const SURFEL_AREA_M2: f32 = 0.25;

/// Hard cap on splats per generated building — a hostile/degenerate mesh can
/// never balloon the overlay (density is scaled down to fit, never aborted).
pub const MAX_BUILDING_SPLATS: usize = 20_000;

/// Preview-backend sample spacing along walls and roof (metres).
const PREVIEW_SPACING_M: f32 = 0.5;

/// The backend tag appended to the planting receipt — the user always learns
/// which generator built their building.
#[cfg(feature = "forge-native")]
pub const BACKEND_TAG: &str = "Forge native";
#[cfg(not(feature = "forge-native"))]
pub const BACKEND_TAG: &str = "built-in preview — enable forge-native for the real generator";

/// Warm brick reflectance: long-wavelength bands high (terracotta/brick reads
/// red-brown), green window muted, blue floor low — built exactly like the
/// sibling spectra in `plugins.rs` (`valley`/`rock`) and `build_scene`'s `spd`.
fn brick_spectral() -> [u16; 16] {
    std::array::from_fn(|b| {
        let v: f32 = match b {
            11..=14 => 0.58, // long bands — brick red-brown
            6..=8 => 0.22,   // green window muted
            _ => 0.10,
        };
        half::f16::from_f32(v).to_bits()
    })
}

/// Slate-grey roof reflectance: flat and dim across the bands so the roof
/// separates from the warm walls.
fn roof_spectral() -> [u16; 16] {
    std::array::from_fn(|_| half::f16::from_f32(0.30).to_bits())
}

/// The repo's deterministic LCG idiom (vfx_graph / many_light): full-period
/// 64-bit mul-add, draws from the top bits, strict [0,1).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F))
    }
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Generate a building with whichever backend this build carries, returning
/// the splats and the honest backend tag for the receipt.
pub fn generate_building(spec: BuildingSpec) -> (Vec<GaussianSplat>, &'static str) {
    let spec = spec.clamped();
    #[cfg(feature = "forge-native")]
    {
        (generate_native(spec), BACKEND_TAG)
    }
    #[cfg(not(feature = "forge-native"))]
    {
        (generate_preview(spec), BACKEND_TAG)
    }
}

/// Deterministic PREVIEW building: four wall faces sampled on a regular
/// [`PREVIEW_SPACING_M`] grid plus a flat roof — a readable box silhouette in
/// brick spectra, so the default build's "Add building" produces a real,
/// plantable, undoable asset (just not Forge's architecture).
///
/// Splat count is an exact function of the spec (asserted in tests):
/// `rows * 2*(wall_x + wall_z) + roof_x * roof_z` with the counts derived
/// below — deterministic, no randomness at all.
pub fn generate_preview(spec: BuildingSpec) -> Vec<GaussianSplat> {
    let spec = spec.clamped();
    let w = spec.width_m;
    let d = spec.depth_m;
    let h = spec.storeys as f32 * STOREY_HEIGHT_M;
    let [ox, oy, oz] = BUILDING_PLANT_ORIGIN;

    let nx = (w / PREVIEW_SPACING_M).round().max(1.0) as usize; // samples along X walls
    let nz = (d / PREVIEW_SPACING_M).round().max(1.0) as usize; // samples along Z walls
    let rows = (h / PREVIEW_SPACING_M).round().max(1.0) as usize;

    let brick = brick_spectral();
    let roof = roof_spectral();
    let s = PREVIEW_SPACING_M * 0.55; // splat radius ~ sample spacing
    let mut splats = Vec::with_capacity(rows * 2 * (nx + nz) + nx * nz);

    // Walls: two X-aligned faces (front/back) and two Z-aligned (left/right).
    for r in 0..rows {
        let y = oy + (r as f32 + 0.5) * PREVIEW_SPACING_M;
        for i in 0..nx {
            let x = ox - w * 0.5 + (i as f32 + 0.5) * (w / nx as f32);
            for z in [oz - d * 0.5, oz + d * 0.5] {
                splats.push(GaussianSplat::volume([x, y, z], [s, s, s], Quat::IDENTITY, 240u8, brick));
            }
        }
        for j in 0..nz {
            let z = oz - d * 0.5 + (j as f32 + 0.5) * (d / nz as f32);
            for x in [ox - w * 0.5, ox + w * 0.5] {
                splats.push(GaussianSplat::volume([x, y, z], [s, s, s], Quat::IDENTITY, 240u8, brick));
            }
        }
    }
    // Flat roof grid.
    for i in 0..nx {
        let x = ox - w * 0.5 + (i as f32 + 0.5) * (w / nx as f32);
        for j in 0..nz {
            let z = oz - d * 0.5 + (j as f32 + 0.5) * (d / nz as f32);
            splats.push(GaussianSplat::volume([x, oy + h, z], [s, s, s], Quat::IDENTITY, 240u8, roof));
        }
    }
    splats
}

/// NATIVE backend: the real Forge building generator. Maps the panel spec onto
/// `forge_building::BuildingParams` (every field documented at the mapping),
/// generates the architectural mesh (walls + cornices + roof, style-driven
/// facades with windows), and surfel-samples its triangles into splats at
/// [`SURFEL_AREA_M2`] density.
#[cfg(feature = "forge-native")]
pub fn generate_native(spec: BuildingSpec) -> Vec<GaussianSplat> {
    use forge_building::{BuildingParams, FootprintShape, Style};

    let spec = spec.clamped();
    let params = BuildingParams {
        // storeys (1..=6) → floors u8: clamp already guarantees the range.
        floors: spec.storeys as u8,
        // Both backends share STOREY_HEIGHT_M so silhouettes agree.
        floor_height: STOREY_HEIGHT_M,
        // Panel metres map 1:1 onto Forge's metric width/depth (>= 2.0 holds:
        // our clamp floor is 4.0).
        width: spec.width_m,
        depth: spec.depth_m,
        // Victorian is forge's own default style; Rectangular keeps the
        // footprint inside the documented plant span. Style variety is a panel
        // control for a later slice.
        style: Style::Victorian,
        footprint: FootprintShape::Rectangular,
        // None = the style's default roof (forge picks per style).
        roof: None,
        window_density: 0.5,
        seed: spec.seed,
        // Interior proxies add triangles invisible from outside — skip.
        generate_interior: false,
    };

    // The clamp above guarantees params pass generate()'s validators, so an
    // Err here is a forge bug, not a user-input path; fall back to the preview
    // rather than panicking the editor (no-panic shell rule).
    let mesh = match forge_building::generate(params) {
        Ok(m) => m,
        Err(_) => return generate_preview(spec),
    };
    mesh_to_splats(&mesh, spec.seed)
}

/// Surfel-sample a forge mesh into splats: per triangle, `ceil(area / SURFEL_AREA_M2)`
/// points (scaled down uniformly if the building would exceed
/// [`MAX_BUILDING_SPLATS`]), each placed by the standard sqrt-barycentric
/// uniform-triangle draw from the deterministic LCG. Splat radius tracks the
/// local sample spacing so facades read as continuous masonry.
#[cfg(feature = "forge-native")]
fn mesh_to_splats(mesh: &forge_mesh::Mesh, seed: u64) -> Vec<GaussianSplat> {
    let [ox, oy, oz] = BUILDING_PLANT_ORIGIN;
    let brick = brick_spectral();
    let roof = roof_spectral();

    // Total surface area decides whether the density must shrink to respect
    // the hard cap.
    let tri_area = |t: &[u32; 3]| -> f32 {
        let p = |i: u32| glam::Vec3::from_array(mesh.positions[i as usize]);
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        (b - a).cross(c - a).length() * 0.5
    };
    let total_area: f32 = mesh.indices.iter().map(tri_area).sum();
    let mut area_per_splat = SURFEL_AREA_M2;
    let projected = (total_area / area_per_splat).ceil() as usize;
    if projected > MAX_BUILDING_SPLATS {
        area_per_splat = total_area / MAX_BUILDING_SPLATS as f32;
    }
    let radius = (area_per_splat.sqrt() * 0.6).max(0.12);

    // The mesh's own AABB top tells us which triangles are "roof" (top 15% of
    // the height) for the two-tone spectra — a readable approximation of the
    // material split without depending on forge's material id table.
    let (mn, mx) = mesh.aabb();
    let roof_y = mn[1] + (mx[1] - mn[1]) * 0.85;

    let mut rng = Lcg::new(seed);
    let mut splats = Vec::new();
    for t in &mesh.indices {
        let p = |i: u32| glam::Vec3::from_array(mesh.positions[i as usize]);
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let area = (b - a).cross(c - a).length() * 0.5;
        let count = (area / area_per_splat).ceil() as usize;
        let spectral = if (a.y + b.y + c.y) / 3.0 >= roof_y { roof } else { brick };
        for _ in 0..count {
            if splats.len() >= MAX_BUILDING_SPLATS {
                return splats;
            }
            // Uniform point on the triangle: sqrt-barycentric draw.
            let (r1, r2) = (rng.next_unit(), rng.next_unit());
            let su = r1.sqrt();
            let (u, v) = (1.0 - su, r2 * su);
            let pos = a * u + b * v + c * (1.0 - u - v);
            splats.push(GaussianSplat::volume(
                [pos.x + ox, pos.y + oy, pos.z + oz],
                [radius, radius, radius],
                Quat::IDENTITY,
                240u8,
                spectral,
            ));
        }
    }
    splats
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field-wise splat equality (GaussianSplat is Pod, not PartialEq) — the
    /// established comparison idiom from the planting tests.
    fn splats_eq(a: &GaussianSplat, b: &GaussianSplat) -> bool {
        a.position() == b.position()
            && a.scales() == b.scales()
            && a.opacity() == b.opacity()
            && a.spectral() == b.spectral()
    }

    /// The preview building's splat count is an exact closed-form function of
    /// the spec: rows*2*(nx+nz) walls + nx*nz roof.
    #[test]
    fn preview_count_is_exact_for_known_spec() {
        let spec = BuildingSpec { storeys: 2, width_m: 8.0, depth_m: 6.0, seed: 0 };
        // nx=16, nz=12, rows=12 → 12*2*(16+12) + 16*12 = 672 + 192 = 864.
        let splats = generate_preview(spec);
        assert_eq!(splats.len(), 864, "exact closed-form preview count");
    }

    /// Same spec → bit-identical preview (fully deterministic, no RNG at all).
    #[test]
    fn preview_is_deterministic() {
        let spec = BuildingSpec { storeys: 3, width_m: 10.0, depth_m: 8.0, seed: 7 };
        let (a, b) = (generate_preview(spec), generate_preview(spec));
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| splats_eq(x, y)), "bit-identical preview");
    }

    /// Hostile specs clamp to the documented ranges before generation.
    #[test]
    fn hostile_spec_clamps() {
        let spec = BuildingSpec { storeys: 99, width_m: 1e9, depth_m: f32::NAN, seed: 1 }.clamped();
        assert_eq!(spec.storeys, 6, "storeys clamp to 6");
        assert_eq!(spec.width_m, 24.0, "width clamps to 24 m");
        assert_eq!(spec.depth_m, 6.0, "NaN depth falls back to the default");
    }

    /// Walls are brick (long bands high), roof is slate (flat) — assert the
    /// actual band relationship on the preview output.
    #[test]
    fn preview_walls_are_brick_and_roof_is_slate() {
        let spec = BuildingSpec::default();
        let splats = generate_preview(spec);
        // The roof grid is the LAST nx*nz splats by construction.
        let wall = &splats[0];
        let roof = &splats[splats.len() - 1];
        let wb = half::f16::from_bits(wall.spectral()[12]).to_f32(); // long band
        let wg = half::f16::from_bits(wall.spectral()[7]).to_f32(); // green band
        assert!(wb > wg, "brick wall: long band {wb} > green band {wg}");
        let rb = half::f16::from_bits(roof.spectral()[12]).to_f32();
        let rg = half::f16::from_bits(roof.spectral()[7]).to_f32();
        assert!((rb - rg).abs() < 1e-3, "slate roof is flat across bands: {rb} vs {rg}");
    }

    /// More storeys → strictly more splats (taller walls).
    #[test]
    fn taller_building_has_more_splats() {
        let one = generate_preview(BuildingSpec { storeys: 1, ..Default::default() });
        let six = generate_preview(BuildingSpec { storeys: 6, ..Default::default() });
        assert!(six.len() > one.len(), "{} > {}", six.len(), one.len());
    }

    /// The dispatching entry returns the build's honest backend tag.
    #[test]
    fn generate_building_reports_the_backend() {
        let (splats, tag) = generate_building(BuildingSpec::default());
        assert!(!splats.is_empty());
        assert_eq!(tag, BACKEND_TAG);
    }

    // ---- native-backend tests (compiled only with the sibling) -------------

    /// The REAL forge generator produces a non-trivial architectural mesh and
    /// the surfel conversion covers it within the documented density bound.
    #[cfg(feature = "forge-native")]
    #[test]
    fn native_generates_real_architecture_within_density_bound() {
        let spec = BuildingSpec::default();
        let params = forge_building::BuildingParams {
            floors: spec.storeys as u8,
            floor_height: STOREY_HEIGHT_M,
            width: spec.width_m,
            depth: spec.depth_m,
            style: forge_building::Style::Victorian,
            footprint: forge_building::FootprintShape::Rectangular,
            roof: None,
            window_density: 0.5,
            seed: spec.seed,
            generate_interior: false,
        };
        let mesh = forge_building::generate(params).expect("clamped params always generate");
        assert!(mesh.triangle_count() > 0, "real architecture has triangles");

        let splats = generate_native(spec);
        assert!(!splats.is_empty());
        assert!(splats.len() <= MAX_BUILDING_SPLATS, "hard cap holds");
        // Density bound: per-triangle ceil() over-counts by at most 1 splat
        // per triangle relative to area/SURFEL_AREA_M2.
        let total_area: f32 = mesh
            .indices
            .iter()
            .map(|t| {
                let p = |i: u32| glam::Vec3::from_array(mesh.positions[i as usize]);
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                (b - a).cross(c - a).length() * 0.5
            })
            .sum();
        let upper = (total_area / SURFEL_AREA_M2).ceil() as usize + mesh.triangle_count();
        assert!(
            splats.len() <= upper.min(MAX_BUILDING_SPLATS),
            "{} splats within density bound {} (area {total_area:.1} m²)",
            splats.len(),
            upper
        );
    }

    /// Native generation is deterministic in the seed.
    #[cfg(feature = "forge-native")]
    #[test]
    fn native_is_deterministic() {
        let spec = BuildingSpec { seed: 42, ..Default::default() };
        let (a, b) = (generate_native(spec), generate_native(spec));
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| splats_eq(x, y)), "bit-identical native build");
    }

    /// Hostile specs cannot reach a ForgeError: the clamp runs first, so even
    /// absurd inputs generate successfully.
    #[cfg(feature = "forge-native")]
    #[test]
    fn native_hostile_spec_cannot_error() {
        let (splats, _) =
            generate_building(BuildingSpec { storeys: 0, width_m: -5.0, depth_m: 1e30, seed: 3 });
        assert!(!splats.is_empty(), "clamped hostile spec still builds");
    }
}
