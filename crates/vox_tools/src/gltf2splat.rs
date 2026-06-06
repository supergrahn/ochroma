//! GLTF/GLB → Gaussian-splat (.vxm) converter.
//!
//! Surface-samples each mesh primitive's triangles (area-weighted) into
//! surfel-style 2D Gaussian splats:
//!   - position: a random barycentric point on the triangle
//!   - tangent_u / tangent_v: an orthonormal basis in the triangle plane
//!     (so the splat is *flattened along the surface normal* — standard 2DGS)
//!   - scale_u / scale_v: derived from the local triangle edge length
//!   - color: material base-color factor × interpolated vertex color,
//!     converted to a 16-band spectral reflectance via the SAME
//!     [`vox_data::SpectralUpsampler`] (Smits 1999) the PLY loader uses.
//!
//! HONEST SCOPE NOTES:
//!   - **Texture sampling is skipped.** Color comes from the material
//!     base-color *factor* and (if present) vertex colors only. Sampling the
//!     base-color texture image at the surface UV is deliberately not done to
//!     keep the converter dependency-light; the factor is a faithful flat-color
//!     approximation for untextured/factor-driven materials.
//!   - **Skinning / morph targets / animation are ignored.** Only the static
//!     bind-pose geometry of each primitive is sampled, transformed by the
//!     node's world matrix.
//!   - Only `TRIANGLES` primitives are sampled (the common case); other
//!     topologies are skipped.

use std::path::Path;

use glam::{Mat4, Vec2, Vec3};
use half::f16;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{MaterialType, VxmFile, VxmHeader};
use vox_data::SpectralUpsampler;

/// Conversion configuration.
#[derive(Debug, Clone, Copy)]
pub struct Gltf2SplatConfig {
    /// Target splat density in splats per unit world area (m^-2).
    ///
    /// A triangle of area `A` yields `ceil(A * density)` samples (at least 1).
    pub density: f32,
    /// Multiplier applied to the per-splat disk radius relative to the local
    /// triangle scale. Larger values overlap splats; smaller leaves gaps.
    pub scale_factor: f32,
    /// Base opacity for every emitted splat (0..=255).
    pub opacity: u8,
}

impl Default for Gltf2SplatConfig {
    fn default() -> Self {
        Self {
            // ~256 splats per m^2 → hundreds of splats for a unit (1m) cube.
            density: 256.0,
            scale_factor: 1.0,
            opacity: 230,
        }
    }
}

/// Result of a conversion: the splats plus some provenance counters.
pub struct ConversionResult {
    pub splats: Vec<GaussianSplat>,
    pub mesh_primitive_count: usize,
    pub triangle_count: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum Gltf2SplatError {
    #[error("gltf import error: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("vxm write error: {0}")]
    Vxm(#[from] vox_data::vxm::VxmError),
    #[error("primitive {0} is missing required POSITION attribute")]
    MissingPositions(usize),
}

/// A small deterministic xorshift RNG so sampling is reproducible (tests rely
/// on this) without pulling in an external `rand` dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // top 24 bits → mantissa
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Encode a linear-RGB triple into a 16-band spectral `[u16; 16]` (f16 bits),
/// matching `vox_data::ply_loader`'s encoding exactly.
fn rgb_to_spectral(r: f32, g: f32, b: f32) -> [u16; 16] {
    let spectral_f32 = SpectralUpsampler::from_rgb(r, g, b);
    std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits())
}

/// Convert a GLB/GLTF file on disk into Gaussian splats.
pub fn convert_file(
    path: &Path,
    config: Gltf2SplatConfig,
) -> Result<ConversionResult, Gltf2SplatError> {
    let (document, buffers, _images) = gltf::import(path)?;
    convert_document(&document, &buffers, config)
}

/// Core conversion over an already-loaded glTF document + buffers.
pub fn convert_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    config: Gltf2SplatConfig,
) -> Result<ConversionResult, Gltf2SplatError> {
    let mut splats = Vec::new();
    let mut mesh_primitive_count = 0usize;
    let mut triangle_count = 0usize;
    let mut rng = Rng::new(0x9E3779B97F4A7C15);

    // Walk the default scene's node hierarchy so node transforms are applied;
    // fall back to all scenes if there is no default scene.
    let scenes: Vec<gltf::Scene> = match document.default_scene() {
        Some(s) => vec![s],
        None => document.scenes().collect(),
    };

    for scene in scenes {
        for node in scene.nodes() {
            visit_node(
                &node,
                Mat4::IDENTITY,
                buffers,
                config,
                &mut rng,
                &mut splats,
                &mut mesh_primitive_count,
                &mut triangle_count,
            )?;
        }
    }

    Ok(ConversionResult {
        splats,
        mesh_primitive_count,
        triangle_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn visit_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    config: Gltf2SplatConfig,
    rng: &mut Rng,
    splats: &mut Vec<GaussianSplat>,
    mesh_primitive_count: &mut usize,
    triangle_count: &mut usize,
) -> Result<(), Gltf2SplatError> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            sample_primitive(
                &primitive,
                world,
                buffers,
                config,
                rng,
                splats,
                triangle_count,
            )?;
            *mesh_primitive_count += 1;
        }
    }

    for child in node.children() {
        visit_node(
            &child,
            world,
            buffers,
            config,
            rng,
            splats,
            mesh_primitive_count,
            triangle_count,
        )?;
    }
    Ok(())
}

fn sample_primitive(
    primitive: &gltf::Primitive,
    world: Mat4,
    buffers: &[gltf::buffer::Data],
    config: Gltf2SplatConfig,
    rng: &mut Rng,
    splats: &mut Vec<GaussianSplat>,
    triangle_count: &mut usize,
) -> Result<(), Gltf2SplatError> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Ok(());
    }

    let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

    let positions: Vec<Vec3> = match reader.read_positions() {
        Some(iter) => iter.map(Vec3::from).collect(),
        None => return Err(Gltf2SplatError::MissingPositions(primitive.index())),
    };

    let normals: Option<Vec<Vec3>> = reader
        .read_normals()
        .map(|iter| iter.map(Vec3::from).collect());

    // Optional per-vertex colors (RGBA, normalized to f32).
    let vcolors: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|c| c.into_rgba_f32().collect());

    // Material base-color factor (linear RGBA).
    let base_color = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_factor();

    // Build the index list (explicit indices, or implicit 0..n).
    let indices: Vec<u32> = match reader.read_indices() {
        Some(i) => i.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    // Normal matrix (inverse-transpose of the upper 3x3) for transforming
    // directions correctly under non-uniform scale.
    let normal_mat = Mat4::from_mat3(glam::Mat3::from_mat4(world).inverse().transpose());

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        *triangle_count += 1;

        let p0 = world.transform_point3(positions[i0]);
        let p1 = world.transform_point3(positions[i1]);
        let p2 = world.transform_point3(positions[i2]);

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let cross = e1.cross(e2);
        let area = 0.5 * cross.length();
        if area <= 1e-12 {
            continue; // degenerate triangle
        }

        // Geometric normal (consistent winding); prefer averaged vertex normals
        // when available for a smoother surface orientation.
        let geo_normal = cross.normalize();
        let normal = if let Some(ns) = &normals {
            let avg = normal_mat.transform_vector3(ns[i0] + ns[i1] + ns[i2]);
            if avg.length_squared() > 1e-12 {
                let n = avg.normalize();
                // keep it consistent with geometric winding
                if n.dot(geo_normal) < 0.0 { -n } else { n }
            } else {
                geo_normal
            }
        } else {
            geo_normal
        };

        // Orthonormal tangent basis in the triangle plane (perpendicular to the
        // normal): this is what flattens the splat *along* the normal.
        let tangent_u = if e1.length_squared() > 1e-12 {
            (e1 - normal * e1.dot(normal)).normalize_or_zero()
        } else {
            Vec3::ZERO
        };
        let tangent_u = if tangent_u.length_squared() > 1e-12 {
            tangent_u
        } else {
            // fallback: any vector perpendicular to the normal
            let helper = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            normal.cross(helper).normalize()
        };
        let tangent_v = normal.cross(tangent_u).normalize();

        // Splat disk radius from local triangle size. Use sqrt(area) as a
        // characteristic length; spread it over the number of samples so denser
        // sampling yields proportionally smaller (overlapping) disks.
        let n_samples = ((area * config.density).ceil() as usize).max(1);
        let char_len = area.sqrt();
        let radius = (char_len / (n_samples as f32).sqrt()) * config.scale_factor;

        for _ in 0..n_samples {
            // Uniform barycentric sample on the triangle.
            let mut u = rng.next_f32();
            let mut v = rng.next_f32();
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let pos = p0 + e1 * u + e2 * v;

            // Interpolated vertex color (if any) × base-color factor.
            let (cr, cg, cb) = if let Some(vc) = &vcolors {
                let w = 1.0 - u - v;
                let c0 = Vec3::new(vc[i0][0], vc[i0][1], vc[i0][2]);
                let c1 = Vec3::new(vc[i1][0], vc[i1][1], vc[i1][2]);
                let c2 = Vec3::new(vc[i2][0], vc[i2][1], vc[i2][2]);
                let c = c0 * w + c1 * u + c2 * v;
                (
                    c.x * base_color[0],
                    c.y * base_color[1],
                    c.z * base_color[2],
                )
            } else {
                (base_color[0], base_color[1], base_color[2])
            };

            let spectral = rgb_to_spectral(cr, cg, cb);

            splats.push(GaussianSplat::surface(
                pos.into(),
                tangent_u.into(),
                tangent_v.into(),
                radius,
                radius,
                config.opacity,
                spectral,
            ));
        }
    }

    // Silence unused-import lint when no UVs are consumed (textures skipped).
    let _ = Vec2::ZERO;
    Ok(())
}

/// Convert a GLB/GLTF file and write a `.vxm` to `output`.
///
/// Returns the number of splats written.
pub fn gltf2splat(
    input: &Path,
    output: &Path,
    config: Gltf2SplatConfig,
) -> Result<usize, Gltf2SplatError> {
    let result = convert_file(input, config)?;
    let count = result.splats.len();

    let header = VxmHeader::new(
        uuid::Uuid::new_v4(),
        count as u32,
        MaterialType::Generic,
    );
    let vxm = VxmFile {
        header,
        splats: result.splats,
    };

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(output)?;
    vxm.write(file)?;

    Ok(count)
}

// ---------------------------------------------------------------------------
// Minimal GLB generator — used to produce a runnable test asset (assets/cube.glb).
// The `gltf` crate cannot *write* glTF, so we hand-roll a binary GLB for a unit
// cube (positions + normals + indices, single base-color material).
// ---------------------------------------------------------------------------

/// Build the bytes of a binary GLB for a unit cube centered at the origin
/// (extent -0.5..0.5 on each axis) with a solid base color (linear RGBA).
///
/// The cube has 24 vertices (4 per face, so each face has its own flat normal)
/// and 36 indices (12 triangles).
pub fn unit_cube_glb(base_color: [f32; 4]) -> Vec<u8> {
    // 6 faces, each: normal + 4 corner positions (CCW) + 2 triangles.
    // Corners ordered so the face winds CCW when viewed from outside.
    let faces: [( [f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-0.5,-0.5,0.5],[-0.5,0.5,0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,-0.5]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,-0.5]]),
        // -Y
        ([0.0,-1.0, 0.0], [[-0.5,-0.5,0.5],[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,-0.5,0.5]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5]]),
        // -Z
        ([0.0, 0.0,-1.0], [[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5]]),
    ];

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(24);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(24);
    let mut indices: Vec<u16> = Vec::with_capacity(36);

    for (normal, corners) in faces.iter() {
        let base = positions.len() as u16;
        for c in corners.iter() {
            positions.push(*c);
            normals.push(*normal);
        }
        // two triangles: (0,1,2) (0,2,3)
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    // ---- Build the binary buffer: [positions][normals][indices], 4-byte aligned.
    let mut bin: Vec<u8> = Vec::new();
    let pos_offset = bin.len();
    for p in &positions {
        for &f in p {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let nrm_offset = bin.len();
    for n in &normals {
        for &f in n {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let idx_offset = bin.len();
    for &i in &indices {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    // pad bin to 4-byte boundary
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    // Compute position min/max (required by the spec for the POSITION accessor).
    let mut pmin = [f32::INFINITY; 3];
    let mut pmax = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for k in 0..3 {
            pmin[k] = pmin[k].min(p[k]);
            pmax[k] = pmax[k].max(p[k]);
        }
    }

    let pos_len = positions.len() * 12;
    let nrm_len = normals.len() * 12;
    let idx_len = indices.len() * 2;

    // ---- JSON chunk.
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"vox_tools::gltf2splat"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2,"material":0,"mode":4}}]}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[{},{},{},{}]}}}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34963}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}},{{"bufferView":1,"componentType":5126,"count":{},"type":"VEC3"}},{{"bufferView":2,"componentType":5123,"count":{},"type":"SCALAR"}}]}}"#,
        base_color[0], base_color[1], base_color[2], base_color[3],
        bin.len(),
        pos_offset, pos_len,
        nrm_offset, nrm_len,
        idx_offset, idx_len,
        positions.len(), pmin[0], pmin[1], pmin[2], pmax[0], pmax[1], pmax[2],
        normals.len(),
        indices.len(),
    );

    let mut json_bytes = json.into_bytes();
    // pad JSON chunk with spaces to 4-byte boundary
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    // ---- Assemble GLB container.
    // Header(12) + JSON chunk header(8) + json + BIN chunk header(8) + bin.
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb: Vec<u8> = Vec::with_capacity(total_len);
    glb.extend_from_slice(b"glTF"); // magic
    glb.extend_from_slice(&2u32.to_le_bytes()); // version
    glb.extend_from_slice(&(total_len as u32).to_le_bytes()); // total length

    // JSON chunk
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);

    // BIN chunk
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);

    glb
}

/// Write a unit-cube GLB to `path` (used to materialize `assets/cube.glb`).
pub fn write_unit_cube_glb(path: &Path, base_color: [f32; 4]) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, unit_cube_glb(base_color))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    /// Build the cube GLB in memory, convert it, and assert real outcomes.
    #[test]
    fn gltf2splat_cube() {
        // A distinctly *red* cube so we can check spectral bands react to color.
        let base_color = [0.8f32, 0.1, 0.1, 1.0];
        let glb = unit_cube_glb(base_color);

        // Sanity: GLB header is well-formed.
        assert_eq!(&glb[0..4], b"glTF");

        // Parse from the in-memory GLB and convert.
        let (doc, buffers, _images) =
            gltf::import_slice(&glb).expect("cube GLB should parse");
        let config = Gltf2SplatConfig::default();
        let result =
            convert_document(&doc, &buffers, config).expect("conversion should succeed");

        // --- splat_count > 0, and "hundreds" for a unit cube at default density.
        let count = result.splats.len();
        assert!(
            count > 100,
            "expected hundreds of splats for a unit cube, got {count}"
        );
        assert_eq!(result.triangle_count, 12, "unit cube has 12 triangles");
        assert_eq!(result.mesh_primitive_count, 1);

        // --- all positions within the cube bounds ± a splat radius slop.
        let slop = 0.1f32;
        for s in &result.splats {
            let p = s.position();
            for k in 0..3 {
                assert!(
                    p[k] >= -0.5 - slop && p[k] <= 0.5 + slop,
                    "position {p:?} component {k} out of cube bounds"
                );
            }
            // splats must be 2DGS surfels
            assert!(s.is_surface(), "expected surface (2DGS) splats");
        }

        // --- orientation check: splats sitting on the +Z face must have their
        //     surfel normal ≈ +Z (the disk is flattened along the normal).
        let mut checked_pz = 0;
        for s in &result.splats {
            let p = s.position();
            // Restrict to the +Z face *interior* so corner samples shared with
            // the side faces (which also reach z≈0.5) don't pollute the check.
            if p[2] > 0.49 && p[0].abs() < 0.45 && p[1].abs() < 0.45 {
                let n = s.normal();
                // the minimal-extent axis (normal) should align with Z
                assert!(
                    n[2].abs() > 0.9,
                    "splat on +Z face should have Z-aligned normal, got {n:?}"
                );
                // tangents should be perpendicular to Z (lie in the face plane)
                assert!(
                    s.tangent_u()[2].abs() < 0.1 && s.tangent_v()[2].abs() < 0.1,
                    "tangents on +Z face should lie in the XY plane"
                );
                checked_pz += 1;
            }
        }
        assert!(checked_pz > 0, "expected to find splats on the +Z face");

        // --- spectral bands must be non-zero for a colored material.
        let s0 = &result.splats[0];
        let spectral_sum: f32 = (0..GaussianSplat::BANDS).map(|b| s0.spectral_f32(b)).sum();
        assert!(
            spectral_sum > 0.0,
            "colored material must produce non-zero spectral energy"
        );
        // A red material should weight the long-wavelength (red) bands clearly.
        // Bands ~10-12 are ~630-680nm. Compare red end vs blue end.
        let blue_end: f32 = (0..3).map(|b| s0.spectral_f32(b)).sum();
        let red_end: f32 = (10..13).map(|b| s0.spectral_f32(b)).sum();
        assert!(
            red_end > blue_end,
            "red material should have more red-band energy ({red_end}) than blue-band ({blue_end})"
        );
    }

    /// Round-trip the produced .vxm through vox_data's loader.
    #[test]
    fn gltf2splat_cube_roundtrip_vxm() {
        let glb = unit_cube_glb([0.2, 0.6, 0.9, 1.0]);
        let dir = std::env::temp_dir();
        let glb_path = dir.join("vox_tools_test_cube.glb");
        let vxm_path = dir.join("vox_tools_test_cube.vxm");
        std::fs::write(&glb_path, &glb).unwrap();

        let written = gltf2splat(&glb_path, &vxm_path, Gltf2SplatConfig::default())
            .expect("gltf2splat should succeed");
        assert!(written > 0);

        // Load it back via vox_data's VxmFile reader.
        let file = std::fs::File::open(&vxm_path).unwrap();
        let loaded = VxmFile::read(file).expect("vxm should round-trip");
        assert_eq!(
            loaded.splats.len(),
            written,
            "loaded splat count must match what we wrote"
        );
        assert_eq!(loaded.header.splat_count as usize, written);

        // spectral bands survive the round-trip and are non-zero
        let s = &loaded.splats[0];
        let sum: f32 = (0..GaussianSplat::BANDS).map(|b| s.spectral_f32(b)).sum();
        assert!(sum > 0.0, "round-tripped spectral energy must be non-zero");

        let _ = std::fs::remove_file(&glb_path);
        let _ = std::fs::remove_file(&vxm_path);
    }

    /// Materialize the committed `assets/cube.glb` test asset so the CLI
    /// acceptance command is runnable verbatim. Run with:
    ///   cargo test -p ochroma-tools -- --ignored generate_asset_cube_glb
    #[test]
    #[ignore = "asset generator: run explicitly to (re)write assets/cube.glb"]
    fn generate_asset_cube_glb() {
        // workspace-relative: CARGO_MANIFEST_DIR is crates/vox_tools
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/cube.glb");
        write_unit_cube_glb(&path, [0.8, 0.1, 0.1, 1.0]).unwrap();
        // verify it parses
        let (doc, _b, _i) = gltf::import(&path).expect("written asset must parse");
        assert_eq!(doc.meshes().count(), 1);
        println!("wrote {}", path.display());
    }

    /// The on-disk cube generator must produce a parseable GLB.
    #[test]
    fn write_cube_glb_is_parseable() {
        let dir = std::env::temp_dir();
        let path = dir.join("vox_tools_gen_cube.glb");
        write_unit_cube_glb(&path, [1.0, 1.0, 1.0, 1.0]).unwrap();
        let (doc, _b, _i) = gltf::import(&path).expect("generated GLB must parse");
        assert_eq!(doc.meshes().count(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
