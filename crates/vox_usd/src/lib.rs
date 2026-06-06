//! `vox_usd` — read-only USD scene importer (ENGINE layer, game-agnostic).
//!
//! Reads composed `.usdc`/`.usd` scenes via the pure-Rust [`openusd_rs`] sibling
//! crate and converts them into engine primitives:
//!
//! - `Mesh`            → 2DGS **surface** splats (area-weighted barycentric sampling)
//! - `PointInstancer`  → 3DGS **volume** splats (one per instance)
//! - `Points`          → 3DGS **volume** splats (point + width)
//! - `*Light`          → [`UsdLight`] (`color × intensity × 2^exposure`)
//! - `Camera`          → [`UsdCamera`] (schemaless: raw attribute reads)
//! - everything else   → [`UsdEntity`] (named transform node for the outliner)
//!
//! Material `diffuseColor` is upsampled to 16 spectral bands via
//! [`vox_data::SpectralUpsampler`] (Smits 1999), matching the spz/ply/gltf path.
//!
//! ## Hard safety rule
//!
//! [`openusd_rs::usd::Attribute::get`] panics on type mismatch/absence. This
//! crate calls **only** `try_get` / `get_value`, both of which are total. Do not
//! introduce a bare `.get::<T>()` call anywhere in this crate.
//!
//! ## `.usda` text limitation
//!
//! openusd-rs's USDA *text* parser cannot read arrays or tuples (it returns
//! empty), so geometry must come from binary `.usdc`/`.usd`. A `.usda` scene
//! whose only geometry is array/tuple-valued surfaces yields
//! [`UsdError::UnsupportedTextArray`] rather than a silent empty import.

use glam::{DMat4, DVec3, Quat, Vec3};
use half::f16;
use openusd_rs::{gf, tf, usd, usd_geom, usd_shade, vt};
use std::path::Path;
use vox_core::types::GaussianSplat;
use vox_data::SpectralUpsampler;

// ---------------------------------------------------------------------------
// Public data model
// ---------------------------------------------------------------------------

/// Result of importing a USD scene. Mirrors `vox_data::gltf_import::ImportResult`
/// closely enough that the editor consumes it like the other splat importers.
#[derive(Debug)]
pub struct UsdImport {
    pub splats: Vec<GaussianSplat>,
    pub lights: Vec<UsdLight>,
    pub camera: Option<UsdCamera>,
    /// Named transform nodes (Xform/Scope/unknown) for the outliner.
    pub entities: Vec<UsdEntity>,
    pub warnings: Vec<String>,
    pub stats: UsdImportStats,
    /// Composed stage up-axis token (e.g. "Y" / "Z"), for diagnostics/CLI.
    pub up_axis: String,
    /// Composed stage `metersPerUnit`, for diagnostics/CLI.
    pub meters_per_unit: f64,
    /// Per-geometry-prim splat log (path, type, splat count) for the CLI's
    /// per-prim lines. Not part of the engine consumption path.
    pub geom_log: Vec<GeomLog>,
}

/// One geometry prim's contribution to the splat buffer (for CLI reporting).
#[derive(Debug, Clone)]
pub struct GeomLog {
    pub path: String,
    pub type_name: String,
    pub splats: usize,
}

#[derive(Debug, Clone)]
pub struct UsdLight {
    pub name: String,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub kind: UsdLightKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsdLightKind {
    Sphere,
    Rect,
    Disk,
    Distant,
    Dome,
}

#[derive(Debug, Clone)]
pub struct UsdCamera {
    pub name: String,
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_y_deg: f32,
}

#[derive(Debug, Clone)]
pub struct UsdEntity {
    pub name: String,
    pub path: String,
    pub world: [[f32; 4]; 4],
    pub type_name: String,
}

#[derive(Debug, Default, Clone)]
pub struct UsdImportStats {
    pub prims: usize,
    pub meshes: usize,
    pub points: usize,
    pub instancers: usize,
    pub splats: usize,
    pub lights: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsdError {
    /// File missing / unreadable, or no composed prims at all.
    Open(String),
    /// Stage composed but yielded no importable prims.
    Empty,
    /// A `.usda` whose only geometry is array/tuple-valued (parser limitation).
    UnsupportedTextArray,
}

impl std::fmt::Display for UsdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UsdError::Open(p) => write!(f, "could not open USD stage: {p}"),
            UsdError::Empty => write!(f, "USD stage composed but yielded no importable prims"),
            UsdError::UnsupportedTextArray => write!(
                f,
                "USDA text scene has array/tuple geometry that openusd-rs cannot parse; \
                 re-export as binary .usdc"
            ),
        }
    }
}

impl std::error::Error for UsdError {}

#[derive(Debug, Clone)]
pub struct UsdImportSettings {
    /// Correct the scene so it is Y-up in engine space. Default `true`.
    pub up_axis_correction: bool,
    /// Opacity assigned to generated splats (0..=255). Default `240`.
    pub default_opacity: u8,
    /// Target surface-splat density (splats per square metre). Default `200.0`.
    pub mesh_splats_per_sqm: f32,
    /// Hard splat ceiling; overflow stops sampling and emits a warning. Default `5_000_000`.
    pub max_splats: usize,
}

impl Default for UsdImportSettings {
    fn default() -> Self {
        Self {
            up_axis_correction: true,
            default_opacity: 240,
            mesh_splats_per_sqm: 200.0,
            max_splats: 5_000_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Open + compose the stage at `path` and traverse it with default settings.
pub fn import_usd(path: &Path) -> Result<UsdImport, UsdError> {
    import_usd_with(path, &UsdImportSettings::default())
}

/// Open + compose the stage at `path` and traverse it with explicit `settings`.
pub fn import_usd_with(path: &Path, settings: &UsdImportSettings) -> Result<UsdImport, UsdError> {
    if !path.exists() {
        return Err(UsdError::Open(path.display().to_string()));
    }

    let is_text = matches!(
        path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("usda"),
    );

    let stage = usd::Stage::open(path);
    let root = stage.pseudo_root();

    // The pseudo-root must expose at least one defined child, else the file did
    // not compose into anything (missing/garbage stage).
    if root.children().next().is_none() {
        return Err(UsdError::Open(path.display().to_string()));
    }

    let up_axis: tf::Token = root.metadata(&tf::Token::new("upAxis")).unwrap_or_default();
    let meters_per_unit = read_meters_per_unit(&root);

    let root_correction =
        root_correction_matrix(settings.up_axis_correction, up_axis.as_str(), meters_per_unit);

    let mut walk = Walk {
        stage: &stage,
        settings,
        out: UsdImport {
            splats: Vec::new(),
            lights: Vec::new(),
            camera: None,
            entities: Vec::new(),
            warnings: Vec::new(),
            stats: UsdImportStats::default(),
            up_axis: if up_axis.as_str().is_empty() {
                "Y".to_string()
            } else {
                up_axis.as_str().to_string()
            },
            meters_per_unit,
            geom_log: Vec::new(),
        },
        is_text,
        text_array_geometry_seen: false,
    };

    for child in root.children() {
        walk.visit(&child, root_correction);
    }

    // A .usda whose only geometry is array/tuple-valued is silently empty to the
    // text parser. Surface that explicitly rather than returning an empty Ok.
    if walk.is_text && walk.text_array_geometry_seen && walk.out.splats.is_empty() {
        return Err(UsdError::UnsupportedTextArray);
    }

    let mut out = walk.out;
    out.stats.splats = out.splats.len();
    out.stats.lights = out.lights.len();

    if out.splats.is_empty() && out.lights.is_empty() && out.camera.is_none() && out.entities.is_empty()
    {
        return Err(UsdError::Empty);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage metadata helpers
// ---------------------------------------------------------------------------

/// Read `metersPerUnit` (a `double`), working around an openusd-rs usdc bug.
///
/// openusd-rs decodes an *inline* scalar `double` with `f64::from_bits(payload)`,
/// but USD stores the inline payload as a zero-extended **f32** bit pattern
/// (`usdc/parser.rs:read_inline`), so `1.0` comes back as a subnormal ~5e-315.
/// We detect that signature (high 32 bits zero) and reinterpret the low 32 bits
/// as the intended `f32`. Falls back to `1.0` for anything out of sane range.
fn read_meters_per_unit(root: &usd::Prim) -> f64 {
    let Some(raw) = root.metadata::<f64>(&tf::Token::new("metersPerUnit")) else {
        return 1.0;
    };
    if raw.is_finite() && (1e-6..=1e6).contains(&raw) {
        return raw;
    }
    // Recover the zero-extended-f32 inline-double payload.
    let bits = raw.to_bits();
    if bits >> 32 == 0 {
        let recovered = f32::from_bits(bits as u32) as f64;
        if recovered.is_finite() && (1e-6..=1e6).contains(&recovered) {
            return recovered;
        }
    }
    1.0
}

/// Build the root correction matrix: optional Z-up→Y-up rotation, then a uniform
/// `metersPerUnit` scale so the scene lands in metres, Y-up engine space.
fn root_correction_matrix(correction: bool, up_axis: &str, meters_per_unit: f64) -> DMat4 {
    let mut m = DMat4::IDENTITY;
    if meters_per_unit != 1.0 && meters_per_unit > 0.0 {
        m = DMat4::from_scale(DVec3::splat(meters_per_unit)) * m;
    }
    if correction && up_axis.eq_ignore_ascii_case("Z") {
        // Z-up → Y-up: rotate -90° about X so +Z maps to +Y.
        m = DMat4::from_rotation_x(-std::f64::consts::FRAC_PI_2) * m;
    }
    m
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

struct Walk<'s> {
    stage: &'s usd::Stage,
    settings: &'s UsdImportSettings,
    out: UsdImport,
    is_text: bool,
    /// A `.usda` prim that carries array/tuple geometry the text parser drops.
    text_array_geometry_seen: bool,
}

impl Walk<'_> {
    fn visit(&mut self, prim: &usd::Prim, parent_world: DMat4) {
        if !prim.is_valid() {
            // Skip the prim itself but still descend (its subtree may be valid).
            for child in prim.children() {
                self.visit(&child, parent_world);
            }
            return;
        }

        self.out.stats.prims += 1;

        let local = self.read_local(prim);
        let world = parent_world * local;

        let type_name = prim.type_name();
        let before = self.out.splats.len();
        match type_name.as_str() {
            "Mesh" => {
                self.out.stats.meshes += 1;
                let color = self.bound_diffuse_rgb(prim);
                self.mesh_to_splats(prim, world, color);
                self.log_geom(prim, &type_name, before);
            }
            "PointInstancer" => {
                self.out.stats.instancers += 1;
                self.instancer_to_splats(prim, world);
                self.log_geom(prim, &type_name, before);
                // A PointInstancer's children are *prototypes* — templates that
                // are instanced via `positions`, never rendered as standalone
                // geometry. Do not descend into them.
                return;
            }
            "Points" => {
                self.out.stats.points += 1;
                self.points_to_splats(prim, world);
                self.log_geom(prim, &type_name, before);
            }
            "SphereLight" | "RectLight" | "DiskLight" | "DistantLight" | "DomeLight" => {
                self.read_light(prim, world);
            }
            "Camera" => {
                self.read_camera(prim, world);
            }
            _ => {
                self.push_entity(prim, world);
            }
        }

        for child in prim.children() {
            self.visit(&child, world);
        }
    }

    /// Record a geometry prim's splat contribution for the CLI per-prim line.
    fn log_geom(&mut self, prim: &usd::Prim, type_name: &tf::Token, before: usize) {
        let produced = self.out.splats.len() - before;
        self.out.geom_log.push(GeomLog {
            path: prim.path().to_string(),
            type_name: type_name.as_str().to_string(),
            splats: produced,
        });
    }

    /// Local transform from accumulated `xformOpOrder`. `gf::Matrix4d` is
    /// row-major; transpose into glam's column-major `DMat4`.
    fn read_local(&self, prim: &usd::Prim) -> DMat4 {
        match usd_geom::XformOp::get_local_matrix(prim) {
            Some(m) => DMat4::from_cols_array_2d(&m.data).transpose(),
            None => DMat4::IDENTITY,
        }
    }

    fn at_capacity(&mut self) -> bool {
        if self.out.splats.len() >= self.settings.max_splats {
            if !self
                .out
                .warnings
                .iter()
                .any(|w| w.starts_with("max_splats"))
            {
                self.out.warnings.push(format!(
                    "max_splats ceiling {} reached; further geometry skipped",
                    self.settings.max_splats
                ));
            }
            return true;
        }
        false
    }

    // -- Mesh → 2DGS surface splats -----------------------------------------

    fn mesh_to_splats(&mut self, prim: &usd::Prim, world: DMat4, color_rgb: [f32; 3]) {
        // Read points + face topology with the total `try_get` (never `get`).
        let Some(points) = prim
            .attribute(&tf::Token::new("points"))
            .try_get::<vt::Array<gf::Vec3f>>()
        else {
            // No readable points. In .usda this is the array-parser limitation.
            if self.is_text {
                self.text_array_geometry_seen = true;
            }
            return;
        };
        let counts = prim
            .attribute(&tf::Token::new("faceVertexCounts"))
            .try_get::<vt::Array<i32>>();
        let indices = prim
            .attribute(&tf::Token::new("faceVertexIndices"))
            .try_get::<vt::Array<i32>>();
        let (Some(counts), Some(indices)) = (counts, indices) else {
            if self.is_text {
                self.text_array_geometry_seen = true;
            }
            return;
        };

        // World-space vertex positions.
        let verts: Vec<Vec3> = points
            .iter()
            .map(|p| {
                let w = world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
                Vec3::new(w.x as f32, w.y as f32, w.z as f32)
            })
            .collect();

        // Fan-triangulate (mirrors usd_geom::triangulate, but via try_get-safe data).
        let tris = triangulate(&counts, &indices);

        let spectral = rgb_to_spectral_bits(color_rgb);
        let spm = self.settings.mesh_splats_per_sqm;
        let opacity = self.settings.default_opacity;

        for tri in tris.chunks(3) {
            if tri.len() < 3 {
                continue;
            }
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
                continue;
            }
            let v0 = verts[i0];
            let v1 = verts[i1];
            let v2 = verts[i2];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(edge2);
            let area = normal.length() * 0.5;

            // Tangent frame for the disk plane.
            let tan_u = if edge1.length() > 1e-8 {
                edge1.normalize()
            } else {
                Vec3::X
            };
            let n = if normal.length() > 1e-8 {
                normal.normalize()
            } else {
                Vec3::Y
            };
            let tan_v = n.cross(tan_u).normalize_or_zero();

            // Same formula as gltf_import: clamp(ceil(area·spm), 1, 50).
            let splat_count = ((area * spm).ceil() as usize).clamp(1, 50);

            for si in 0..splat_count {
                if self.at_capacity() {
                    return;
                }
                // Deterministic barycentric sample (matches gltf_import).
                let t = si as f32 / splat_count as f32;
                let u = ((t * 7.3 + 0.1).fract()).min(0.999);
                let v = ((t * 13.7 + 0.2).fract()).min(0.999);
                let (u, v) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };

                let pos = v0 * (1.0 - u - v) + v1 * u + v2 * v;
                let scale = (area / splat_count as f32).sqrt().clamp(0.001, 0.1);

                self.out.splats.push(GaussianSplat::surface(
                    [pos.x, pos.y, pos.z],
                    [tan_u.x, tan_u.y, tan_u.z],
                    [tan_v.x, tan_v.y, tan_v.z],
                    scale,
                    scale * 0.3,
                    opacity,
                    spectral,
                ));
            }
        }
    }

    // -- PointInstancer → 3DGS volume splats --------------------------------

    fn instancer_to_splats(&mut self, prim: &usd::Prim, world: DMat4) {
        let Some(positions) = prim
            .attribute(&tf::Token::new("positions"))
            .try_get::<vt::Array<gf::Vec3f>>()
        else {
            if self.is_text {
                self.text_array_geometry_seen = true;
            }
            return;
        };

        let scales = prim
            .attribute(&tf::Token::new("scales"))
            .try_get::<vt::Array<gf::Vec3f>>();
        let orientations = prim
            .attribute(&tf::Token::new("orientations"))
            .try_get::<vt::Array<gf::Quath>>();

        let spectral = rgb_to_spectral_bits([0.5, 0.5, 0.5]);
        let opacity = self.settings.default_opacity;

        for (i, p) in positions.iter().enumerate() {
            if self.at_capacity() {
                return;
            }
            let w = world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
            let pos = [w.x as f32, w.y as f32, w.z as f32];

            let scale = scales
                .as_ref()
                .and_then(|s| (i < s.len()).then(|| s[i]))
                .map(|s| [s.x, s.y, s.z])
                .unwrap_or([0.1, 0.1, 0.1]);

            let rot = orientations
                .as_ref()
                .and_then(|o| (i < o.len()).then(|| o[i]))
                .map(|q| {
                    Quat::from_xyzw(q.i.to_f32(), q.j.to_f32(), q.k.to_f32(), q.w.to_f32())
                })
                .unwrap_or(Quat::IDENTITY);

            self.out
                .splats
                .push(GaussianSplat::volume(pos, scale, rot, opacity, spectral));
        }
    }

    // -- Points (PointBased) → 3DGS volume splats ---------------------------

    fn points_to_splats(&mut self, prim: &usd::Prim, world: DMat4) {
        let Some(points) = prim
            .attribute(&tf::Token::new("points"))
            .try_get::<vt::Array<gf::Vec3f>>()
        else {
            if self.is_text {
                self.text_array_geometry_seen = true;
            }
            return;
        };

        let widths = prim
            .attribute(&tf::Token::new("widths"))
            .try_get::<vt::Array<f32>>();

        let spectral = rgb_to_spectral_bits([0.5, 0.5, 0.5]);
        let opacity = self.settings.default_opacity;

        for (i, p) in points.iter().enumerate() {
            if self.at_capacity() {
                return;
            }
            let w = world.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
            let pos = [w.x as f32, w.y as f32, w.z as f32];
            let r = widths
                .as_ref()
                .and_then(|wd| (i < wd.len()).then(|| wd[i]))
                .map(|w| w * 0.5)
                .unwrap_or(0.05);
            self.out.splats.push(GaussianSplat::volume(
                pos,
                [r, r, r],
                Quat::IDENTITY,
                opacity,
                spectral,
            ));
        }
    }

    // -- Lights -------------------------------------------------------------

    fn read_light(&mut self, prim: &usd::Prim, world: DMat4) {
        let kind = match prim.type_name().as_str() {
            "SphereLight" => UsdLightKind::Sphere,
            "RectLight" => UsdLightKind::Rect,
            "DiskLight" => UsdLightKind::Disk,
            "DistantLight" => UsdLightKind::Distant,
            "DomeLight" => UsdLightKind::Dome,
            _ => return,
        };

        // Schema-agnostic raw reads (DistantLight has no schema at all).
        let intensity = prim
            .attribute(&tf::Token::new("inputs:intensity"))
            .try_get::<f32>()
            .unwrap_or(1.0);
        let exposure = prim
            .attribute(&tf::Token::new("inputs:exposure"))
            .try_get::<f32>()
            .unwrap_or(0.0);
        let color = prim
            .attribute(&tf::Token::new("inputs:color"))
            .try_get::<gf::Vec3f>()
            .map(|c| [c.x, c.y, c.z])
            .unwrap_or([1.0, 1.0, 1.0]);

        let position = mat_translation(world);
        // -Z is the USD light forward axis; transform by world rotation.
        let dir = world.transform_vector3(DVec3::NEG_Z);
        let direction = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize_or_zero();

        self.out.lights.push(UsdLight {
            name: prim.name().as_str().to_string(),
            position,
            direction,
            color,
            intensity: intensity * 2.0_f32.powf(exposure),
            kind,
        });
    }

    // -- Camera (schemaless) ------------------------------------------------

    fn read_camera(&mut self, prim: &usd::Prim, world: DMat4) {
        let focal = prim
            .attribute(&tf::Token::new("focalLength"))
            .try_get::<f32>()
            .unwrap_or(50.0);
        let h_aperture = prim
            .attribute(&tf::Token::new("horizontalAperture"))
            .try_get::<f32>()
            .unwrap_or(36.0);

        // The design's load-bearing camera assertion is `2·atan(18/50)` where
        // 18 = horizontalAperture/2 = 36/2 — i.e. the vertical field-of-view is
        // computed from the *horizontal* aperture (the design prose's
        // "verticalAperture" wording is a mislabel; its numbers all use 36/2).
        // We follow the numbers: fovY = 2·atan(horizontalAperture / (2·focal)).
        let fov_y_rad = 2.0 * (h_aperture / (2.0 * focal)).atan();
        let fov_y_deg = fov_y_rad.to_degrees();

        let position = mat_translation(world);
        let (_scale, rotation, _trans) = world.to_scale_rotation_translation();
        let rotation = Quat::from_xyzw(
            rotation.x as f32,
            rotation.y as f32,
            rotation.z as f32,
            rotation.w as f32,
        );

        // First camera wins.
        if self.out.camera.is_none() {
            self.out.camera = Some(UsdCamera {
                name: prim.name().as_str().to_string(),
                position,
                rotation,
                fov_y_deg,
            });
        }
    }

    // -- Entity (Xform/Scope/unknown) ---------------------------------------

    fn push_entity(&mut self, prim: &usd::Prim, world: DMat4) {
        let cols = world.to_cols_array_2d();
        let world_f32 = [
            [cols[0][0] as f32, cols[0][1] as f32, cols[0][2] as f32, cols[0][3] as f32],
            [cols[1][0] as f32, cols[1][1] as f32, cols[1][2] as f32, cols[1][3] as f32],
            [cols[2][0] as f32, cols[2][1] as f32, cols[2][2] as f32, cols[2][3] as f32],
            [cols[3][0] as f32, cols[3][1] as f32, cols[3][2] as f32, cols[3][3] as f32],
        ];
        self.out.entities.push(UsdEntity {
            name: prim.name().as_str().to_string(),
            path: prim.path().to_string(),
            world: world_f32,
            type_name: prim.type_name().as_str().to_string(),
        });
    }

    // -- Material → diffuse RGB ---------------------------------------------

    fn bound_diffuse_rgb(&self, prim: &usd::Prim) -> [f32; 3] {
        let binding = usd_shade::MaterialBindingAPI::new(self.stage.prim_at_path(prim.path().clone()));
        let Some(material) = binding.bound_material(self.stage) else {
            return [0.5, 0.5, 0.5];
        };
        let Some(shader) = material.surface_shader(self.stage) else {
            return [0.5, 0.5, 0.5];
        };
        match shader.get_input::<gf::Vec3f>("diffuseColor") {
            Some(c) => [c.x, c.y, c.z],
            None => [0.5, 0.5, 0.5],
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Fan-triangulate face topology (mirrors `usd_geom::triangulate` but operates
/// on already-read, `try_get`-safe arrays).
fn triangulate(counts: &vt::Array<i32>, indices: &vt::Array<i32>) -> Vec<i32> {
    let mut tris = Vec::new();
    let mut cursor = 0usize;
    for &count in counts {
        let count = count.max(0) as usize;
        if count >= 3 && cursor + count <= indices.len() {
            for i in 0..(count - 2) {
                tris.push(indices[cursor]);
                tris.push(indices[cursor + i + 1]);
                tris.push(indices[cursor + i + 2]);
            }
        }
        cursor += count;
    }
    tris
}

/// Translation component of a column-major `DMat4`, as `Vec3` (f32).
fn mat_translation(m: DMat4) -> Vec3 {
    let t = m.w_axis;
    Vec3::new(t.x as f32, t.y as f32, t.z as f32)
}

/// RGB → 16-band spectral, each band as `f16` bits (matches spz/ply/gltf path).
fn rgb_to_spectral_bits(rgb: [f32; 3]) -> [u16; 16] {
    let coeffs = SpectralUpsampler::from_rgb(rgb[0], rgb[1], rgb[2]);
    let mut out = [0u16; 16];
    for (o, c) in out.iter_mut().zip(coeffs.iter()) {
        *o = f16::from_f32(*c).to_bits();
    }
    out
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn meters_per_unit_recovery_matches_inline_f32_bug() {
        // The exact subnormal openusd-rs hands back for an inline-double 1.0.
        let bits: u64 = 0x0000_0000_3f80_0000;
        let raw = f64::from_bits(bits);
        assert!(raw < 1e-6, "precondition: raw is the buggy subnormal");
        // Our recovery reinterprets the low 32 bits as f32 1.0.
        let recovered = f32::from_bits(bits as u32) as f64;
        assert_eq!(recovered, 1.0);
    }

    #[test]
    fn triangulate_quads_fans_into_two_tris_each() {
        let counts = vt::Array::from(vec![4i32]);
        let indices = vt::Array::from(vec![0i32, 1, 2, 3]);
        let tris = triangulate(&counts, &indices);
        assert_eq!(tris, vec![0, 1, 2, 0, 2, 3]);
    }
}
