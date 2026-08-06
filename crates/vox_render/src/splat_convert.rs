//! Conversion from Ochroma's `GaussianSplat` to a Spectra [`SceneState`].
//!
//! The current `spectra-renderer` is a triangle-mesh spectral path tracer driven
//! by `spectra_scene_state::SceneState` (it no longer ingests Gaussian primitives
//! directly — the old `SplatScene` / `GaussianDisc` / `GaussianBlob` API was
//! removed and the Gaussian rasteriser now lives in the separate
//! `spectra-gaussian-render` crate, which vox_render does not depend on).
//!
//! To render Ochroma splats through the native path tracer we tessellate each
//! splat into a small camera-agnostic quad (2 triangles) placed in world space:
//!
//! * `kind == 0` (surface / 2DGS) → a flat quad spanned by `tangent_u`/`tangent_v`,
//!   scaled by `scale_u`/`scale_v`, with the quad normal = `cross(tu, tv)`.
//! * `kind == 1` (volume / 3DGS) → an axis-aligned quad on the XY plane sized by
//!   the splat's `scale_u`/`scale_v`, facing +Z. (A full ellipsoid tessellation is
//!   future work; a single quad gives a renderable proxy footprint.)
//!
//! Spectral: Ochroma carries 16 bands (380–755 nm). The path tracer's per-material
//! spectral SPD is 16 bands as well, but wiring a unique material per splat requires
//! packing the 132-float `MaterialData` struct, which is out of scope here. All
//! tessellated triangles therefore use the uploader's default material (index 0);
//! `material_ids` is left empty so the uploader assigns material 0 to every triangle.
//! Per-splat spectral colour is preserved on this struct's API surface (see the
//! returned vertex normals / positions) and can be promoted to per-material SPDs
//! once the material packing helper lands.

use spectra_scene_state::{CameraLayer, SceneState};
use vox_core::types::GaussianSplat;

#[cfg(feature = "spectra-native")]
use crate::splat_backend::{
    BlasDesc, InstanceRecordGpu, MATERIAL_FLOATS, PbrMaterial, WEATHERING_SCALE_NEUTRAL,
    pack_mesh_material,
};

/// One quad = 4 vertices, 2 triangles (6 indices).
const VERTS_PER_SPLAT: usize = 4;
const INDICES_PER_SPLAT: usize = 6;

/// Compute the four corner positions and the normal of a single splat's quad.
///
/// Returns `([p0, p1, p2, p3], normal)` where the corners wind CCW:
/// `p0 = c - u - v`, `p1 = c + u - v`, `p2 = c + u + v`, `p3 = c - u + v`,
/// with `u = half_u * tangent_u` and `v = half_v * tangent_v`.
fn splat_quad(s: &GaussianSplat) -> ([[f32; 3]; 4], [f32; 3]) {
    let c = s.position();

    // Choose the in-plane axes. Surface splats carry an authored tangent frame;
    // volume splats have no surface, so we fall back to an XY billboard.
    let (tu, tv) = if s.is_surface() {
        (s.tangent_u(), s.tangent_v())
    } else {
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    };

    let hu = s.scale_u().max(1e-6);
    let hv = s.scale_v().max(1e-6);

    let u = [tu[0] * hu, tu[1] * hu, tu[2] * hu];
    let v = [tv[0] * hv, tv[1] * hv, tv[2] * hv];

    let p0 = [c[0] - u[0] - v[0], c[1] - u[1] - v[1], c[2] - u[2] - v[2]];
    let p1 = [c[0] + u[0] - v[0], c[1] + u[1] - v[1], c[2] + u[2] - v[2]];
    let p2 = [c[0] + u[0] + v[0], c[1] + u[1] + v[1], c[2] + u[2] + v[2]];
    let p3 = [c[0] - u[0] + v[0], c[1] - u[1] + v[1], c[2] - u[2] + v[2]];

    // Normal = normalize(cross(tu, tv)).
    let nx = tu[1] * tv[2] - tu[2] * tv[1];
    let ny = tu[2] * tv[0] - tu[0] * tv[2];
    let nz = tu[0] * tv[1] - tu[1] * tv[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-8);
    let normal = [nx / len, ny / len, nz / len];

    ([p0, p1, p2, p3], normal)
}

/// Append a single splat's quad (4 verts, 2 tris) to the flat geometry arrays.
///
/// `base` is the current vertex count (== positions.len() / 3) before appending,
/// used to offset the triangle indices.
fn push_splat_quad(
    s: &GaussianSplat,
    base: u32,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) {
    let (corners, n) = splat_quad(s);

    for p in &corners {
        positions.extend_from_slice(p);
        normals.extend_from_slice(&n);
    }
    // UVs for the four corners (unit square).
    uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

    // Two triangles: (0,1,2) and (0,2,3), offset by `base`.
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Convert a slice of [`GaussianSplat`] into a renderable [`SceneState`].
///
/// Every splat is tessellated into a quad and packed into the scene's
/// [`spectra_scene_state::GeometryLayer`]. `width`/`height` set the render target
/// and camera resolution; the caller fills in the real camera via
/// [`SceneState::camera`] before rendering.
///
/// Splats with an unrecognised `kind` are skipped.
pub fn splats_to_scene(splats: &[GaussianSplat], width: u32, height: u32) -> SceneState {
    let renderable: Vec<&GaussianSplat> = splats
        .iter()
        .filter(|s| s.is_surface() || s.is_volume())
        .collect();

    let mut positions: Vec<f32> = Vec::with_capacity(renderable.len() * VERTS_PER_SPLAT * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(renderable.len() * VERTS_PER_SPLAT * 3);
    let mut uvs: Vec<f32> = Vec::with_capacity(renderable.len() * VERTS_PER_SPLAT * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(renderable.len() * INDICES_PER_SPLAT);

    for (i, s) in renderable.iter().enumerate() {
        let base = (i * VERTS_PER_SPLAT) as u32;
        push_splat_quad(
            s,
            base,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut indices,
        );
    }

    let mut scene = SceneState::new(width, height);
    scene.geometry.vertex_count = positions.len() / 3;
    scene.geometry.triangle_count = indices.len() / 3;
    scene.geometry.positions = positions;
    scene.geometry.normals = normals;
    scene.geometry.uvs = uvs;
    scene.geometry.indices = indices;
    // material_ids left empty → uploader assigns the default material (index 0).
    scene.mark_geometry_changed();
    scene
}

/// Build a [`CameraLayer`] from a column-major view matrix and vertical FOV.
///
/// This is the camera type the native renderer consumes (`Renderer::set_camera_view_matrix`
/// reads `CameraLayer::view_matrix`). Width/height set the target resolution and aspect.
///
/// `lens_radius` and `focus_distance` drive the path tracer's **real** depth of
/// field (`lens_radius == 0.0` is a pinhole; pass `1.0` focus for back-compat).
/// They are threaded here so the cinematic [`crate::cine::CinePose`] adapter and
/// gameplay callers share one builder rather than zeroing DoF downstream.
pub fn camera_layer(
    view_matrix: [f32; 16],
    fov_y_radians: f32,
    width: u32,
    height: u32,
    lens_radius: f32,
    focus_distance: f32,
) -> CameraLayer {
    let mut cam = CameraLayer::new_default();
    cam.view_matrix = view_matrix;
    cam.fov_y_radians = fov_y_radians;
    cam.width = width;
    cam.height = height;
    cam.lens_radius = lens_radius;
    cam.focus_distance = focus_distance;
    cam
}

/// Resolve a splat's 16-band spectral reflectance to a linear sRGB albedo.
fn splat_albedo(s: &GaussianSplat) -> [f32; 3] {
    use vox_core::spectral::{Illuminant, SpectralBands, spectral_to_xyz, xyz_to_srgb};
    let bands = SpectralBands(std::array::from_fn(|i| {
        half::f16::from_bits(s.spectral()[i]).to_f32()
    }));
    let xyz = spectral_to_xyz(&bands, &Illuminant::d65());
    let lin = xyz_to_srgb(xyz);
    [
        lin[0].clamp(0.0, 1.0),
        lin[1].clamp(0.0, 1.0),
        lin[2].clamp(0.0, 1.0),
    ]
}

/// Convert splats into a **lit, coloured** [`SceneState`] for the path tracer.
///
/// Unlike [`splats_to_scene`] (axis-aligned grey billboards), each splat here
/// becomes a **camera-facing** quad sized by its scale — so volume splats stay
/// solid from any view — carrying a per-colour Lambert [`MaterialData`] built
/// from the splat's spectral reflectance (resolved to sRGB albedo). Materials are
/// deduplicated by quantised colour to keep the count small. `eye` is the camera
/// position the quads face. **Additive**: `splats_to_scene` is unchanged.
pub fn splats_to_lit_scene(
    splats: &[GaussianSplat],
    width: u32,
    height: u32,
    eye: [f32; 3],
) -> SceneState {
    use glam::Vec3;
    use spectra_scene_data::MaterialData;
    use spectra_scene_state::{MATERIAL_FLOATS, MaterialLayer};

    let eye = Vec3::from(eye);
    let renderable: Vec<&GaussianSplat> = splats
        .iter()
        .filter(|s| s.is_surface() || s.is_volume())
        .collect();

    let mut positions = Vec::with_capacity(renderable.len() * 12);
    let mut normals = Vec::with_capacity(renderable.len() * 12);
    let mut uvs = Vec::with_capacity(renderable.len() * 8);
    let mut indices = Vec::with_capacity(renderable.len() * 6);
    let mut material_ids = Vec::with_capacity(renderable.len() * 2);
    let mut params: Vec<f32> = Vec::new();
    let mut palette: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for (i, s) in renderable.iter().enumerate() {
        let c = Vec3::from(s.position());
        let to_eye = {
            let d = eye - c;
            if d.length_squared() < 1e-8 {
                Vec3::Z
            } else {
                d.normalize()
            }
        };
        let up = if to_eye.y.abs() < 0.99 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let right = up.cross(to_eye).normalize();
        let realup = to_eye.cross(right);
        let r = if s.is_volume() {
            s.scales().iter().copied().fold(0.0_f32, f32::max).max(1e-3)
        } else {
            s.scale_u().max(s.scale_v()).max(1e-3)
        };
        let (u, v) = (right * r, realup * r);
        let corners = [c - u - v, c + u - v, c + u + v, c - u + v];
        for p in &corners {
            positions.extend_from_slice(&[p.x, p.y, p.z]);
            normals.extend_from_slice(&[to_eye.x, to_eye.y, to_eye.z]);
        }
        uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Per-colour material (deduped by 5-bit-per-channel quantised albedo).
        let rgb = splat_albedo(s);
        let key = (((rgb[0] * 31.0) as u32) << 10)
            | (((rgb[1] * 31.0) as u32) << 5)
            | ((rgb[2] * 31.0) as u32);
        let mid = *palette.entry(key).or_insert_with(|| {
            let idx = (params.len() / MATERIAL_FLOATS) as u32;
            let mut mat = MaterialData::default();
            mat.base_color = [rgb[0], rgb[1], rgb[2], 1.0];
            mat.roughness = 0.85;
            mat.metallic = 0.0;
            let mut arr = mat.to_f32_array();
            arr.resize(MATERIAL_FLOATS, 0.0);
            params.extend_from_slice(&arr);
            idx
        });
        material_ids.push(mid);
        material_ids.push(mid);
    }

    let material_count = params.len() / MATERIAL_FLOATS;
    let mut scene = SceneState::new(width, height);
    scene.geometry.vertex_count = positions.len() / 3;
    scene.geometry.triangle_count = indices.len() / 3;
    scene.geometry.positions = positions;
    scene.geometry.normals = normals;
    scene.geometry.uvs = uvs;
    scene.geometry.indices = indices;
    scene.geometry.material_ids = material_ids;
    scene.materials = MaterialLayer {
        params,
        spectral_spd: std::collections::HashMap::new(),
        material_count,
    };
    scene.mark_geometry_changed();
    scene.mark_materials_changed();
    scene
}

/// Forge meshes + per-instance transforms/materials → an instanced
/// [`SceneState`] for the resident path tracer (Render Keystone T3).
///
/// Replaces the splat-quad/material-0 live path. Each [`BlasDesc`] is one
/// archetype prototype; each [`InstanceRecordGpu`] names a prototype and
/// carries its own world transform + per-instance material BASE. The base
/// reaches shading via the TLAS instance custom index (uploaded as
/// `instance_material_base`) where the closest-hit ADDS it to each triangle's
/// relative material id, so two instances of the SAME BLAS shade with DIFFERENT
/// materials WHILE per-triangle multi-material is preserved (design §4.1).
///
/// Materials use Spectra's canonical 132-float scalar-layout ABI on every
/// backend. `spectral_spd` populates `MaterialLayer::spectral_spd[material_id]`
/// with the real 16-band reflectance (empty ⇒ no entry ⇒ white, spectral off).
///
/// Geometry: prototype meshes are laid out contiguously into one SceneState
/// vertex/triangle soup (object-space); the per-instance world transform is
/// applied by the TLAS at trace time. `width`/`height` set the render target;
/// the caller fills the real camera before rendering.
/// Synthesize the geometry-anchored weathering PATTERN for a merged scene:
/// 7 floats per vertex `[moss, water_stain, paint_chip, rust, soot,
/// efflorescence, edge_wear]` parallel to `positions` (flat `[x,y,z]`/vertex)
/// and `normals` (flat `[nx,ny,nz]`/vertex).
///
/// Pure + deterministic (a fixed function of the id-sorted merged geometry — no
/// HashMap/RNG iteration, no time), so it never perturbs the determinism
/// artifact. Amplitudes are deliberately SUBTLE (≤ ~0.3 at reference intensity)
/// so a clean city still reads clean; the sim scales them per-instance later.
///
/// The pattern (gravity-directional aging, per [[dynamic-weathering-directive]]):
/// - **soot** rises from the street: strongest near y≈0, fading with height.
/// - **efflorescence** wicks up from the very base (a tighter band than soot).
/// - **water_stain** weeps on down-facing ledges/sills (normal.y < 0).
/// - **edge_wear** on near-vertical wall faces (|normal.y| small) — arris/wall wear.
/// Returns an empty vec when there is no geometry (weathering then stays off).
#[cfg(feature = "spectra-native")]
#[cfg(any(not(feature = "aot-shaders"), test))]
fn build_weathering_pattern(positions: &[f32], normals: &[f32]) -> Vec<f32> {
    let vc = positions.len() / 3;
    if vc == 0 {
        return Vec::new();
    }
    // Street-level reference: soot/efflorescence are anchored to world y≈0 (the
    // ground plane the city sits on). Heights are in metres.
    // CONFIG-FIRST: the weathering-pattern falloffs/caps are `config/ochroma.ron`
    // `materials.weather_*`; defaults equal the old literals (14/4/0.28/0.18/0.22/0.12).
    let mcfg = &vox_config::config().materials;
    let soot_falloff_m = mcfg.weather_soot_falloff_m;
    let efflor_falloff_m = mcfg.weather_efflor_falloff_m;
    let soot_max = mcfg.weather_soot_max;
    let efflor_max = mcfg.weather_efflor_max;
    let stain_max = mcfg.weather_stain_max;
    let edge_max = mcfg.weather_edge_max;

    let mut masks = vec![0.0f32; vc * 7];
    for v in 0..vc {
        let y = positions[v * 3 + 1];
        let ny = normals.get(v * 3 + 1).copied().unwrap_or(0.0);
        let base = masks.get_mut(v * 7..v * 7 + 7).unwrap();

        // Near-vertical WALL factor — soot/efflorescence/edge-wear are FACADE
        // phenomena, so gate them on verticality. This also keeps the pattern off
        // flat up-facing terrain/roads/roofs (ny≈+1 → wall≈0), so the merged
        // scene's ground geometry stays clean even though it shares the buffer.
        let wall = (1.0 - ny.abs()).clamp(0.0, 1.0);
        // Height factor, 1 at the street, decaying with height.
        let h = y.max(0.0);
        let soot = soot_max * (-h / soot_falloff_m).exp() * wall;
        let efflor = efflor_max * (-h / efflor_falloff_m).exp() * wall;
        // Down-facing surfaces (sills, ledge undersides, cornice soffits) weep.
        let down = (-ny).max(0.0); // 1 for a fully down-facing surface
        let stain = stain_max * down * down;
        // Near-vertical wall faces take edge/arris wear.
        let edge = edge_max * wall;

        base[1] = stain; // water_stain
        base[4] = soot; // soot
        base[5] = efflor; // efflorescence
        base[6] = edge; // edge_wear
        // moss/paint_chip/rust left at 0 — those need material/sim context the
        // pattern alone shouldn't assume (a brand-new steel/glass tower has no
        // moss); the sim/cook supplies them per instance later.
    }
    masks
}

#[cfg(feature = "spectra-native")]
pub fn meshes_to_instanced_scene(
    blas: &[BlasDesc],
    instances: &[InstanceRecordGpu],
    materials: &[PbrMaterial],
    spectral_spd: &[(u32, [f32; 16])],
    width: u32,
    height: u32,
) -> SceneState {
    meshes_to_instanced_scene_with_weathering(
        blas,
        instances,
        materials,
        spectral_spd,
        &[],
        width,
        height,
    )
}

/// [`meshes_to_instanced_scene`] plus the SIM's per-instance dynamic weathering.
///
/// `instance_weathering` is a SPARSE, ASCENDING-instance-index side table:
/// `(instance_index, per-channel scale)`. Only instances the sim actually owns
/// (buildings, which know their own age and maintenance) appear in it; every
/// other instance is materialised as [`WEATHERING_SCALE_NEUTRAL`], which
/// reproduces the previous global-only render exactly. An EMPTY table emits no
/// buffer at all, so the GPU keeps its zero-cost legacy path.
///
/// Sparse rather than a field on [`InstanceRecordGpu`] because buildings are a
/// small minority of a city scene's instances (a few hundred among tens of
/// thousands of terrain chunks, cims, vehicles and scattered plants), and
/// because it keeps every unrelated push site untouched.
///
/// # Panics
/// If the table is not strictly ascending, or names an instance that does not
/// exist. Both are host bugs that would otherwise silently weather the wrong
/// building — the exact defect class this feature exists to fix.
#[cfg(feature = "spectra-native")]
pub fn meshes_to_instanced_scene_with_weathering(
    blas: &[BlasDesc],
    instances: &[InstanceRecordGpu],
    materials: &[PbrMaterial],
    spectral_spd: &[(u32, [f32; 16])],
    instance_weathering: &[(u32, [f32; 7])],
    width: u32,
    height: u32,
) -> SceneState {
    // Compatibility entry point for callers that still borrow their prototype
    // table. The live game path uses the consuming entry point below so source
    // BLAS storage is released as each prototype is packed.
    meshes_to_instanced_scene_with_weathering_owned(
        blas.to_vec(),
        instances,
        materials,
        spectral_spd,
        instance_weathering,
        width,
        height,
    )
}

// The consuming scene packer replaces large per-prototype vectors with one
// contiguous soup. glibc otherwise retains the freed source arenas until the
// entire conversion ends, making dead BLAS pages overlap the growing CLAS scene
// in RSS. Return wholly free arenas between prototypes; this is load-time only
// (never a frame-path allocator operation).
#[cfg(target_os = "linux")]
fn release_consumed_blas_pages() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: process-global allocator maintenance takes no application pointer.
    unsafe {
        malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
fn release_consumed_blas_pages() {}

/// Consuming variant of [`meshes_to_instanced_scene_with_weathering`].
///
/// A full city previously retained every [`BlasDesc`] while the contiguous
/// `SceneState` soup was built. That makes source and packed geometry coexist at
/// the exact host-RSS peak. Owning the table lets each completed prototype drop
/// immediately after its streams are appended; no pixel or ABI contract changes.
#[cfg(feature = "spectra-native")]
pub fn meshes_to_instanced_scene_with_weathering_owned(
    blas: Vec<BlasDesc>,
    instances: &[InstanceRecordGpu],
    materials: &[PbrMaterial],
    spectral_spd: &[(u32, [f32; 16])],
    instance_weathering: &[(u32, [f32; 7])],
    width: u32,
    height: u32,
) -> SceneState {
    use spectra_scene_state::MaterialLayer;

    // SLOT-0 OPAQUE CONTRACT: the megakernel clamps every out-of-range resolved
    // material id (instance base + per-triangle relative id) to slot 0. A
    // cutout/vegetation/glass slot 0 turns any id-resolution bug into INVISIBLE
    // geometry (alpha-cutout passes the ray through). The game's MaterialTable
    // reserves an opaque slot 0; enforce the contract here for every producer.
    if let Some(m0) = materials.first() {
        let cutout_armed = m0.opacity_tex >= 0 || m0.vegetation_bsdf;
        let transmissive = m0.transmission > 0.0;
        debug_assert!(
            !cutout_armed && !transmissive,
            "material slot 0 must be guaranteed-opaque (it is the kernel's OOB clamp target); \
             got opacity_tex={} vegetation_bsdf={} transmission={}",
            m0.opacity_tex,
            m0.vegetation_bsdf,
            m0.transmission
        );
        if cutout_armed || transmissive {
            eprintln!(
                "[splat_convert] WARNING: material slot 0 is cutout/glass \
                 (opacity_tex={} vegetation_bsdf={} transmission={}); the kernel's OOB \
                 clamp target must be opaque — reserve an opaque slot 0 \
                 (MaterialTable::reserve_opaque_fallback_slot)",
                m0.opacity_tex, m0.vegetation_bsdf, m0.transmission
            );
        }
    }

    // --- Geometry soup: prototypes laid out contiguously ---
    // Track each prototype's vertex base so a future per-archetype-BLAS uploader
    // can recover sub-ranges; today the HW soup proto shares the buffer and the
    // instance custom index selects the material.
    let total_verts: usize = blas.iter().map(|b| b.positions.len()).sum();
    let total_tris: usize = blas.iter().map(|b| b.indices.len()).sum();

    let mut positions: Vec<f32> = Vec::with_capacity(total_verts * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(total_verts * 3);
    let mut uvs: Vec<f32> = Vec::with_capacity(total_verts * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(total_tris * 3);
    let mut tri_material_ids: Vec<u32> = Vec::with_capacity(total_tris);
    let mut construction_group_ids: Vec<u32> = Vec::with_capacity(total_tris);
    // Capture small per-prototype bounds before consuming the source streams.
    let proto_aabbs: Vec<([f32; 3], [f32; 3])> =
        blas.iter().map(|b| (b.aabb_min, b.aabb_max)).collect();

    let mut vbase: u32 = 0;
    let mut tbase: u32 = 0;
    // K1: per-proto sub-ranges into the shared soup, so the uploader can build one
    // BLAS per prototype (the multi-proto TLAS that makes 100K buildings + 1M cims
    // representable instead of one merged soup).
    let mut proto_ranges: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(blas.len());
    let mut source_triangle_order = Vec::with_capacity(total_tris);
    let mut source_clusters = Vec::new();
    // In non-AOT diagnostics most meshes have no authored weathering at all;
    // constructing the legacy seven-float pattern for those neutral streams is
    // pure zero residency. Keep the exact streams that do exist while each BLAS
    // is consumed, then build the compatibility pattern only when it is needed.
    #[cfg(not(feature = "aot-shaders"))]
    let mut authored_weathering: Vec<(usize, Vec<f32>)> = Vec::new();
    #[cfg(feature = "aot-shaders")]
    let mut weathering_masks = Vec::with_capacity(total_verts * 7);
    for (proto_index, b) in blas.into_iter().enumerate() {
        // The prior iteration's source vectors are out of scope now.
        release_consumed_blas_pages();
        let v_start = vbase;
        let t_start = tbase;
        if !b.indices.is_empty() {
            let partition = vox_data::geometry_clusters::ReadyGeometryClusters::from_source_mesh(
                &b.positions,
                &b.indices,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "prototype {} cannot enter Ochroma MegaGeometry source partition: {error}",
                    b.proto_id
                )
            });
            let order_start = source_triangle_order.len() as u32;
            source_triangle_order.extend(
                partition
                    .triangle_order()
                    .iter()
                    .map(|triangle| t_start + triangle),
            );
            for cluster in partition.clusters() {
                source_clusters.push(spectra_scene_state::SourceGeometryCluster {
                    cluster_id: source_clusters.len() as u32,
                    proto_index: proto_index as u32,
                    triangle_order_start: order_start + cluster.triangle_order_start(),
                    triangle_count: cluster.triangle_count(),
                    bounds_min: cluster.bounds_min(),
                    bounds_max: cluster.bounds_max(),
                });
            }
        }
        #[cfg(feature = "aot-shaders")]
        assert_eq!(
            b.positions.len(),
            b.normals.len(),
            "product BLAS {} has {} positions but {} normals; runtime geometry repair/substitution is forbidden",
            b.proto_id,
            b.positions.len(),
            b.normals.len()
        );
        debug_assert_eq!(
            b.positions.len(),
            b.normals.len(),
            "BlasDesc positions/normals length mismatch"
        );
        #[cfg(feature = "aot-shaders")]
        {
            let expected = b.positions.len() * 7;
            assert_eq!(
                b.weathering_masks.len(),
                expected,
                "product BLAS {} has {} weathering floats for {} vertices; expected exactly {} (7 per vertex); runtime synthesis/substitution is forbidden",
                b.proto_id,
                b.weathering_masks.len(),
                b.positions.len(),
                expected,
            );
            weathering_masks.extend_from_slice(&b.weathering_masks);
        }
        #[cfg(not(feature = "aot-shaders"))]
        if !b.weathering_masks.is_empty() {
            authored_weathering.push((v_start as usize, b.weathering_masks));
        }
        for p in &b.positions {
            positions.extend_from_slice(p);
        }
        for n in &b.normals {
            normals.extend_from_slice(n);
        }
        #[cfg(feature = "aot-shaders")]
        {
            assert_eq!(
                b.uvs.len(),
                b.positions.len(),
                "product BLAS {} has {} positions but {} UVs; runtime zero-fill/substitution is forbidden",
                b.proto_id,
                b.positions.len(),
                b.uvs.len()
            );
            for t in &b.uvs {
                uvs.extend_from_slice(t);
            }
        }
        // Old developer diagnostics may still exercise geometry that predates
        // authored UV streams. This compatibility branch is absent from the AOT
        // product runtime.
        #[cfg(not(feature = "aot-shaders"))]
        if b.uvs.len() == b.positions.len() {
            for t in &b.uvs {
                uvs.extend_from_slice(t);
            }
        } else {
            uvs.extend(std::iter::repeat(0.0).take(b.positions.len() * 2));
        }
        #[cfg(feature = "aot-shaders")]
        assert_eq!(
            b.material_ids.len(),
            b.indices.len(),
            "product BLAS {} has {} triangles but {} triangle material ids; runtime slot-0 substitution is forbidden",
            b.proto_id,
            b.indices.len(),
            b.material_ids.len()
        );
        for (ti, tri) in b.indices.iter().enumerate() {
            #[cfg(feature = "aot-shaders")]
            assert!(
                tri.iter().all(|&index| index < b.positions.len() as u32),
                "product BLAS {} triangle {} references {:?} but has only {} vertices; runtime geometry repair is forbidden",
                b.proto_id,
                ti,
                tri,
                b.positions.len()
            );
            indices.extend_from_slice(&[vbase + tri[0], vbase + tri[1], vbase + tri[2]]);
            // Product reads the exact authored entry. The slot-0 compatibility
            // fallback is compiled only for non-AOT developer diagnostics.
            #[cfg(feature = "aot-shaders")]
            let mid = b.material_ids[ti];
            #[cfg(not(feature = "aot-shaders"))]
            let mid = b.material_ids.get(ti).copied().unwrap_or(0);
            tri_material_ids.push(mid);
            construction_group_ids.push(
                b.construction_group_ids
                    .get(ti)
                    .copied()
                    .unwrap_or(u32::MAX),
            );
        }
        let vcount = b.positions.len() as u32;
        let tcount = b.indices.len() as u32;
        proto_ranges.push((v_start, vcount, t_start, tcount));
        vbase += vcount;
        tbase += tcount;
    }
    release_consumed_blas_pages();

    // --- Instances: transforms SoA + per-instance material BASE + proto index ---
    let mut instance_transforms: Vec<f32> = Vec::with_capacity(instances.len() * 16);
    let mut instance_material_base: Vec<u32> = Vec::with_capacity(instances.len());
    let mut instance_proto_index: Vec<u32> = Vec::with_capacity(instances.len());
    let mut instance_dynamic: Vec<u32> = Vec::with_capacity(instances.len());
    for inst in instances {
        instance_transforms.extend_from_slice(&inst.transform);
        instance_material_base.push(inst.material_base);
        instance_proto_index.push(inst.proto_index);
        instance_dynamic.push(u32::from(inst.dynamic));
    }

    // --- Materials: one backend-independent scalar-layout ABI ---
    let mut params = Vec::with_capacity(materials.len() * MATERIAL_FLOATS);
    for material in materials {
        params.extend_from_slice(&pack_mesh_material(*material));
    }

    // --- Spectral SPD keyed by stable material_id (empty ⇒ white) ---
    let mut spd_map = std::collections::HashMap::with_capacity(spectral_spd.len());
    for (mat_id, spd) in spectral_spd {
        spd_map.insert(*mat_id, *spd);
    }

    // --- Weathering stream (7 floats / scene vertex) ---
    #[cfg(feature = "aot-shaders")]
    assert_eq!(
        weathering_masks.len(),
        positions.len() / 3 * 7,
        "product scene assembled {} authored weathering floats for {} vertices",
        weathering_masks.len(),
        positions.len() / 3,
    );
    #[cfg(not(feature = "aot-shaders"))]
    let mut weathering_masks = if authored_weathering.is_empty() {
        Vec::new()
    } else {
        build_weathering_pattern(&positions, &normals)
    };
    #[cfg(not(feature = "aot-shaders"))]
    for (v_start, authored) in authored_weathering {
        let dst = v_start * 7;
        weathering_masks[dst..dst + authored.len()].copy_from_slice(&authored);
    }

    let mut scene = SceneState::new(width, height);
    scene.geometry.vertex_count = positions.len() / 3;
    scene.geometry.triangle_count = indices.len() / 3;
    scene.geometry.weathering_masks = weathering_masks;
    scene.geometry.positions = positions;
    scene.geometry.normals = normals;
    scene.geometry.uvs = uvs;
    scene.geometry.indices = indices;
    scene.geometry.material_ids = tri_material_ids;
    scene.geometry.construction_group_ids = construction_group_ids;
    scene.geometry.instance_count = instances.len();
    scene.geometry.instance_transforms = instance_transforms;
    scene.geometry.instance_material_base = instance_material_base;
    scene.geometry.instance_proto_index = instance_proto_index;
    scene.geometry.instance_dynamic = instance_dynamic;
    // --- Per-instance DYNAMIC weathering (7 floats / instance, or empty) ---
    // Materialise the sparse sim table into the dense buffer the kernel indexes
    // by committed TLAS instance. Ascending order is asserted, not assumed:
    // an out-of-order or out-of-range entry means the host mapped a building to
    // the wrong instance, which would show up as a randomly grimy neighbour and
    // is exactly what this feature must never do.
    scene.geometry.instance_weathering = if instance_weathering.is_empty() {
        Vec::new()
    } else {
        let mut dense = vec![1.0f32; instances.len() * WEATHERING_SCALE_NEUTRAL.len()];
        let mut previous: Option<u32> = None;
        for (index, scale) in instance_weathering {
            assert!(
                previous.is_none_or(|p| *index > p),
                "instance_weathering must be strictly ascending by instance index \
                 (saw {index} after {previous:?}); a re-ordered table would weather \
                 the wrong buildings"
            );
            assert!(
                (*index as usize) < instances.len(),
                "instance_weathering names instance {index} but the scene has only {} \
                 instances",
                instances.len()
            );
            previous = Some(*index);
            let base = *index as usize * WEATHERING_SCALE_NEUTRAL.len();
            dense[base..base + WEATHERING_SCALE_NEUTRAL.len()].copy_from_slice(scale);
        }
        dense
    };
    scene.geometry.proto_aabbs = proto_aabbs;
    scene.geometry.proto_ranges = proto_ranges;
    scene.geometry.source_triangle_order = source_triangle_order;
    scene.geometry.source_clusters = source_clusters;
    scene.materials = MaterialLayer {
        params,
        spectral_spd: spd_map,
        material_count: materials.len(),
    };
    scene.mark_geometry_changed();
    scene.mark_materials_changed();
    scene
}

/// Append one already-CLAS-partitioned geometry chunk to another without
/// rebuilding either partition. All chunk-local indices and source-cluster
/// offsets are rebased into the destination's contiguous ABI.
#[cfg(feature = "spectra-native")]
pub fn append_instanced_scene_geometry(dst: &mut SceneState, mut src: SceneState) {
    let vertex_base = dst.geometry.vertex_count as u32;
    let triangle_base = dst.geometry.triangle_count as u32;
    let proto_base = dst.geometry.proto_ranges.len() as u32;
    let order_base = dst.geometry.source_triangle_order.len() as u32;

    dst.geometry.positions.append(&mut src.geometry.positions);
    dst.geometry.normals.append(&mut src.geometry.normals);
    dst.geometry.uvs.append(&mut src.geometry.uvs);
    dst.geometry.indices.extend(
        src.geometry
            .indices
            .drain(..)
            .map(|index| vertex_base + index),
    );
    dst.geometry
        .material_ids
        .append(&mut src.geometry.material_ids);
    dst.geometry
        .construction_group_ids
        .append(&mut src.geometry.construction_group_ids);
    dst.geometry
        .weathering_masks
        .append(&mut src.geometry.weathering_masks);
    dst.geometry.proto_aabbs.append(&mut src.geometry.proto_aabbs);
    dst.geometry
        .proto_ranges
        .extend(src.geometry.proto_ranges.drain(..).map(
            |(vertex_start, vertex_count, triangle_start, triangle_count)| {
                (
                    vertex_base + vertex_start,
                    vertex_count,
                    triangle_base + triangle_start,
                    triangle_count,
                )
            },
        ));
    dst.geometry.source_triangle_order.extend(
        src.geometry
            .source_triangle_order
            .drain(..)
            .map(|triangle| triangle_base + triangle),
    );
    for mut cluster in src.geometry.source_clusters.drain(..) {
        cluster.cluster_id = dst.geometry.source_clusters.len() as u32;
        cluster.proto_index += proto_base;
        cluster.triangle_order_start += order_base;
        dst.geometry.source_clusters.push(cluster);
    }
    dst.geometry.vertex_count += src.geometry.vertex_count;
    dst.geometry.triangle_count += src.geometry.triangle_count;
    dst.mark_geometry_changed();
}

/// Partition and append exact, unweathered prototype BLASes directly into an
/// existing CLAS geometry destination.
///
/// Unlike `meshes_to_instanced_scene_with_weathering_owned` followed by
/// [`append_instanced_scene_geometry`], this never constructs a second complete
/// `SceneState` for the incoming page. Source attribute streams are consumed and
/// released one at a time as their flat resident representation is appended.
/// This is the runtime admission primitive for streamed terrain/scatter/prop
/// pages whose exact weathering representation is the absent stream.
#[cfg(feature = "spectra-native")]
pub fn append_unweathered_blas_geometry_owned(dst: &mut SceneState, blas: Vec<BlasDesc>) {
    for b in blas {
        let BlasDesc {
            proto_id,
            positions,
            normals,
            uvs,
            indices,
            material_ids,
            construction_group_ids,
            aabb_min,
            aabb_max,
            weathering_masks,
        } = b;
        assert!(
            weathering_masks.is_empty(),
            "streamed unweathered BLAS {proto_id} carries {} weathering floats",
            weathering_masks.len()
        );
        assert_eq!(
            positions.len(),
            normals.len(),
            "streamed BLAS {proto_id} positions/normals mismatch"
        );
        assert_eq!(
            positions.len(),
            uvs.len(),
            "streamed BLAS {proto_id} positions/UV mismatch"
        );
        assert_eq!(
            indices.len(),
            material_ids.len(),
            "streamed BLAS {proto_id} triangles/material ids mismatch"
        );
        assert!(
            construction_group_ids.is_empty(),
            "streamed unweathered BLAS {proto_id} unexpectedly carries construction groups"
        );

        let vertex_base = dst.geometry.vertex_count as u32;
        let triangle_base = dst.geometry.triangle_count as u32;
        let proto_index = dst.geometry.proto_ranges.len() as u32;
        let order_base = dst.geometry.source_triangle_order.len() as u32;
        let vertex_count = positions.len() as u32;
        let triangle_count = indices.len() as u32;

        let partition = vox_data::geometry_clusters::ReadyGeometryClusters::from_source_mesh(
            &positions,
            &indices,
        )
        .unwrap_or_else(|error| {
            panic!(
                "prototype {proto_id} cannot enter Ochroma MegaGeometry source partition: {error}"
            )
        });
        let (triangle_order, clusters) = partition.into_partition();
        dst.geometry.source_triangle_order.extend(
            triangle_order
                .into_iter()
                .map(|triangle| triangle_base + triangle),
        );
        for cluster in clusters {
            dst.geometry
                .source_clusters
                .push(spectra_scene_state::SourceGeometryCluster {
                    cluster_id: dst.geometry.source_clusters.len() as u32,
                    proto_index,
                    triangle_order_start: order_base + cluster.triangle_order_start(),
                    triangle_count: cluster.triangle_count(),
                    bounds_min: cluster.bounds_min(),
                    bounds_max: cluster.bounds_max(),
                });
        }

        for position in positions {
            dst.geometry.positions.extend_from_slice(&position);
        }
        release_consumed_blas_pages();
        for normal in normals {
            dst.geometry.normals.extend_from_slice(&normal);
        }
        release_consumed_blas_pages();
        for uv in uvs {
            dst.geometry.uvs.extend_from_slice(&uv);
        }
        release_consumed_blas_pages();
        for triangle in indices {
            dst.geometry.indices.extend_from_slice(&[
                vertex_base + triangle[0],
                vertex_base + triangle[1],
                vertex_base + triangle[2],
            ]);
        }
        release_consumed_blas_pages();
        dst.geometry.material_ids.extend(material_ids);
        dst.geometry
            .construction_group_ids
            .extend(std::iter::repeat_n(u32::MAX, triangle_count as usize));
        release_consumed_blas_pages();

        dst.geometry.proto_aabbs.push((aabb_min, aabb_max));
        dst.geometry.proto_ranges.push((
            vertex_base,
            vertex_count,
            triangle_base,
            triangle_count,
        ));
        dst.geometry.vertex_count += vertex_count as usize;
        dst.geometry.triangle_count += triangle_count as usize;
    }
    dst.mark_geometry_changed();
}

/// Attach the global instance/material tables after geometry has been admitted
/// in independently streamed CLAS chunks.
#[cfg(feature = "spectra-native")]
pub fn finalize_instanced_scene_with_weathering(
    mut scene: SceneState,
    instances: &[InstanceRecordGpu],
    materials: &[PbrMaterial],
    spectral_spd: &[(u32, [f32; 16])],
    instance_weathering: &[(u32, [f32; 7])],
) -> SceneState {
    use spectra_scene_state::MaterialLayer;

    if let Some(m0) = materials.first() {
        assert!(
            m0.opacity_tex < 0 && !m0.vegetation_bsdf && m0.transmission <= 0.0,
            "material slot 0 must be guaranteed opaque"
        );
    }
    scene.geometry.instance_count = instances.len();
    scene.geometry.instance_transforms = Vec::with_capacity(instances.len() * 16);
    scene.geometry.instance_material_base = Vec::with_capacity(instances.len());
    scene.geometry.instance_proto_index = Vec::with_capacity(instances.len());
    scene.geometry.instance_dynamic = Vec::with_capacity(instances.len());
    for instance in instances {
        scene
            .geometry
            .instance_transforms
            .extend_from_slice(&instance.transform);
        scene
            .geometry
            .instance_material_base
            .push(instance.material_base);
        scene
            .geometry
            .instance_proto_index
            .push(instance.proto_index);
        scene
            .geometry
            .instance_dynamic
            .push(u32::from(instance.dynamic));
    }
    scene.geometry.instance_weathering = if instance_weathering.is_empty() {
        Vec::new()
    } else {
        let mut dense = vec![1.0f32; instances.len() * WEATHERING_SCALE_NEUTRAL.len()];
        let mut previous = None;
        for (index, scale) in instance_weathering {
            assert!(
                previous.is_none_or(|prior| *index > prior),
                "instance_weathering must be strictly ascending"
            );
            assert!((*index as usize) < instances.len());
            previous = Some(*index);
            let base = *index as usize * WEATHERING_SCALE_NEUTRAL.len();
            dense[base..base + WEATHERING_SCALE_NEUTRAL.len()].copy_from_slice(scale);
        }
        dense
    };

    let mut params = Vec::with_capacity(materials.len() * MATERIAL_FLOATS);
    for material in materials {
        params.extend_from_slice(&pack_mesh_material(*material));
    }
    let mut spd_map = std::collections::HashMap::with_capacity(spectral_spd.len());
    for (material_id, spd) in spectral_spd {
        spd_map.insert(*material_id, *spd);
    }
    scene.materials = MaterialLayer {
        params,
        spectral_spd: spd_map,
        material_count: materials.len(),
    };
    scene.mark_geometry_changed();
    scene.mark_materials_changed();
    scene
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn zero_spectral() -> [u16; 16] {
        [0u16; 16]
    }

    #[cfg(feature = "spectra-native")]
    fn weathering_triangle_blas(proto_id: u64, weathering_masks: Vec<f32>) -> BlasDesc {
        BlasDesc {
            proto_id,
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![[0, 1, 2]],
            material_ids: vec![0],
            construction_group_ids: vec![0],
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [1.0, 1.0, 0.0],
            weathering_masks,
        }
    }

    #[cfg(all(feature = "spectra-native", feature = "aot-shaders"))]
    #[test]
    fn weathering_masks_aot_scene_copies_exact_authored_stream() {
        let authored: Vec<f32> = (0..21).map(|i| i as f32 / 20.0).collect();
        let blas = [weathering_triangle_blas(101, authored.clone())];
        let scene = meshes_to_instanced_scene(&blas, &[], &[PbrMaterial::default()], &[], 64, 48);
        assert_eq!(scene.geometry.weathering_masks, authored);
    }

    #[cfg(all(feature = "spectra-native", feature = "aot-shaders"))]
    #[test]
    #[should_panic(expected = "runtime synthesis/substitution is forbidden")]
    fn weathering_masks_aot_scene_rejects_missing_proto_stream() {
        let blas = [weathering_triangle_blas(102, Vec::new())];
        let _ = meshes_to_instanced_scene(&blas, &[], &[PbrMaterial::default()], &[], 64, 48);
    }

    #[cfg(all(feature = "spectra-native", feature = "aot-shaders"))]
    #[test]
    #[should_panic(expected = "runtime zero-fill/substitution is forbidden")]
    fn aot_scene_rejects_missing_authored_uvs() {
        let mut proto = weathering_triangle_blas(104, vec![0.0; 3 * 7]);
        proto.uvs.clear();
        let _ = meshes_to_instanced_scene(&[proto], &[], &[PbrMaterial::default()], &[], 64, 48);
    }

    #[cfg(all(feature = "spectra-native", feature = "aot-shaders"))]
    #[test]
    #[should_panic(expected = "runtime slot-0 substitution is forbidden")]
    fn aot_scene_rejects_missing_triangle_material_ids() {
        let mut proto = weathering_triangle_blas(105, vec![0.0; 3 * 7]);
        proto.material_ids.clear();
        let _ = meshes_to_instanced_scene(&[proto], &[], &[PbrMaterial::default()], &[], 64, 48);
    }

    #[cfg(all(feature = "spectra-native", not(feature = "aot-shaders")))]
    #[test]
    fn weathering_masks_non_aot_scene_retains_legacy_diagnostic_pattern() {
        let blas = [weathering_triangle_blas(103, Vec::new())];
        let scene = meshes_to_instanced_scene(&blas, &[], &[PbrMaterial::default()], &[], 64, 48);
        assert_eq!(scene.geometry.weathering_masks.len(), 21);
        assert!(
            scene
                .geometry
                .weathering_masks
                .iter()
                .any(|&mask| mask > 0.0),
            "legacy non-AOT diagnostic synthesis should remain observable"
        );
    }

    #[test]
    fn weathering_pattern_is_gravity_directional_and_facade_only() {
        // Three vertices: a street-level WALL face (vertical normal, y=0), the
        // SAME wall higher up (y=40), and a flat GROUND vertex (up-facing, y=0).
        // positions [x,y,z] / vertex; normals [nx,ny,nz] / vertex.
        let positions = [
            0.0, 0.0, 0.0, // wall base
            0.0, 40.0, 0.0, // wall high
            5.0, 0.0, 5.0, // ground
        ];
        let normals = [
            1.0, 0.0, 0.0, // wall base: vertical face
            1.0, 0.0, 0.0, // wall high: vertical face
            0.0, 1.0, 0.0, // ground: up-facing
        ];
        let m = build_weathering_pattern(&positions, &normals);
        assert_eq!(m.len(), 3 * 7, "7 floats per vertex");
        // Channel layout: [moss, water_stain, paint_chip, rust, soot, efflor, edge].
        let soot = |v: usize| m[v * 7 + 4];
        let efflor = |v: usize| m[v * 7 + 5];
        let edge = |v: usize| m[v * 7 + 6];

        // Soot rises from the street: base wall soot >> high wall soot > 0.
        assert!(soot(0) > 0.1, "street-level wall has soot, got {}", soot(0));
        assert!(
            soot(0) > soot(1) * 2.0,
            "soot decays with height: base {} vs high {}",
            soot(0),
            soot(1)
        );
        // Ground (up-facing) gets ~no facade weathering — the pattern is wall-only.
        assert!(soot(2) < 1e-4, "flat ground has no soot, got {}", soot(2));
        assert!(efflor(2) < 1e-4, "flat ground has no efflorescence");
        assert!(edge(2) < 1e-4, "flat ground has no edge-wear");
        // The wall DOES get edge-wear; the ground does not.
        assert!(
            edge(0) > 0.01,
            "vertical wall has edge-wear, got {}",
            edge(0)
        );
    }

    #[test]
    fn weathering_pattern_empty_for_no_geometry() {
        assert!(build_weathering_pattern(&[], &[]).is_empty());
    }

    #[test]
    fn surface_quad_normal_is_z_axis() {
        // tangent_u = X, tangent_v = Y  →  normal = +Z.
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let (_corners, n) = splat_quad(&s);
        assert!(n[2] > 0.99, "nz = {} expected ~1.0", n[2]);
        assert!(n[0].abs() < 1e-5);
        assert!(n[1].abs() < 1e-5);
    }

    #[test]
    fn surface_quad_corner_spread_matches_scale() {
        // scale_u = 2, scale_v = 0.5, axes = X/Y, centre at origin.
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            2.0,
            0.5,
            255,
            zero_spectral(),
        );
        let (corners, _n) = splat_quad(&s);
        // p2 = c + u + v = (+2, +0.5, 0); p0 = (-2, -0.5, 0).
        assert_eq!(corners[2], [2.0, 0.5, 0.0]);
        assert_eq!(corners[0], [-2.0, -0.5, 0.0]);
    }

    #[test]
    fn scene_has_two_triangles_per_splat() {
        let surf = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let vol = GaussianSplat::volume(
            [5.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            glam::Quat::IDENTITY,
            128,
            zero_spectral(),
        );
        let scene = splats_to_scene(&[surf, vol], 64, 48);
        // 2 splats → 8 vertices, 4 triangles, 12 indices.
        assert_eq!(scene.geometry.vertex_count, 8);
        assert_eq!(scene.geometry.triangle_count, 4);
        assert_eq!(scene.geometry.indices.len(), 12);
        assert_eq!(scene.geometry.positions.len(), 24);
        assert_eq!(scene.width, 64);
        assert_eq!(scene.height, 48);
    }

    #[test]
    fn scene_indices_offset_per_splat() {
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let scene = splats_to_scene(&[s.clone(), s], 32, 32);
        // Second splat's triangles must reference vertices 4..7, not 0..3.
        // Last index is the last corner of the second quad → base(4) + 3 = 7.
        assert_eq!(*scene.geometry.indices.last().unwrap(), 7);
        // Max index must equal vertex_count - 1.
        let max_idx = *scene.geometry.indices.iter().max().unwrap();
        assert_eq!(max_idx as usize, scene.geometry.vertex_count - 1);
    }

    #[test]
    fn camera_layer_carries_view_and_fov() {
        let view = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -5.0, 1.0,
        ];
        let cam = camera_layer(view, 0.8, 320, 240, 0.0, 1.0);
        assert_eq!(cam.view_matrix, view);
        assert_eq!(cam.fov_y_radians, 0.8);
        assert_eq!(cam.width, 320);
        assert_eq!(cam.height, 240);
    }

    /// K0 witness: `meshes_to_instanced_scene` records REAL per-proto AABBs and
    /// per-instance proto indices while keeping the geometry soup byte-identical
    /// (vertex/tri totals == sum of inputs ⇒ image unchanged). This is the shared
    /// instancing-keystone foundation that terrain chunks + building archetypes
    /// ride on (replaces the scene-wide-AABB-per-BLAS soup collapse).
    #[cfg(feature = "spectra-native")]
    #[test]
    fn meshes_to_instanced_scene_emits_per_proto_aabbs() {
        // Proto 0: a single triangle, bound [0,0,0]..[1,1,0].
        let proto_a = BlasDesc {
            proto_id: 1,
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![[0, 1, 2]],
            material_ids: vec![0],
            construction_group_ids: vec![0],
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [1.0, 1.0, 0.0],
            weathering_masks: vec![0.0; 3 * 7],
        };
        // Proto 1: a quad (2 triangles), bound [10,0,0]..[12,2,0] — DISTINCT.
        let proto_b = BlasDesc {
            proto_id: 2,
            positions: vec![
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [12.0, 2.0, 0.0],
                [10.0, 2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0]; 4],
            indices: vec![[0, 1, 2], [0, 2, 3]],
            material_ids: vec![0, 0],
            construction_group_ids: vec![0, 0],
            aabb_min: [10.0, 0.0, 0.0],
            aabb_max: [12.0, 2.0, 0.0],
            weathering_masks: vec![0.0; 4 * 7],
        };
        let blas = [proto_a, proto_b];
        // 3 instances referencing protos 0, 1, 0.
        let ident = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let instances = [
            InstanceRecordGpu {
                proto_index: 0,
                transform: ident,
                material_base: 0,
                dynamic: false,
            },
            InstanceRecordGpu {
                proto_index: 1,
                transform: ident,
                material_base: 0,
                dynamic: true,
            },
            InstanceRecordGpu {
                proto_index: 0,
                transform: ident,
                material_base: 0,
                dynamic: false,
            },
        ];
        let materials = [PbrMaterial::default()];
        let scene = meshes_to_instanced_scene(&blas, &instances, &materials, &[], 64, 48);

        let sum_verts: usize = blas.iter().map(|b| b.positions.len()).sum();
        let sum_tris: usize = blas.iter().map(|b| b.indices.len()).sum();

        // One REAL AABB per proto, and the two are DISTINCT (not the merged box).
        assert_eq!(scene.geometry.proto_aabbs.len(), 2, "one AABB per proto");
        let aabbs_distinct = scene.geometry.proto_aabbs[0] != scene.geometry.proto_aabbs[1];
        assert!(
            aabbs_distinct,
            "protos must have distinct AABBs, not the scene-wide box"
        );
        assert_eq!(
            scene.geometry.proto_aabbs[0],
            ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
        );
        assert_eq!(
            scene.geometry.proto_aabbs[1],
            ([10.0, 0.0, 0.0], [12.0, 2.0, 0.0])
        );
        // Per-instance proto index preserved in order.
        assert_eq!(scene.geometry.instance_proto_index, vec![0u32, 1, 0]);
        assert_eq!(scene.geometry.instance_dynamic, vec![0u32, 1, 0]);
        // K1: per-proto soup sub-ranges (vbase, vcount, tbase, tcount), contiguous
        // + non-overlapping (proto 1 starts exactly where proto 0 ends — no soup
        // corruption; this is what the multi-proto uploader slices BLASes from).
        assert_eq!(scene.geometry.proto_ranges.len(), 2);
        assert_eq!(scene.geometry.proto_ranges[0], (0, 3, 0, 1));
        assert_eq!(scene.geometry.proto_ranges[1], (3, 4, 1, 2));
        assert_eq!(
            scene.geometry.proto_ranges[1].0,
            scene.geometry.proto_ranges[0].0 + scene.geometry.proto_ranges[0].1,
            "proto vertex ranges must be contiguous"
        );
        assert_eq!(
            scene.geometry.proto_ranges[1].2,
            scene.geometry.proto_ranges[0].2 + scene.geometry.proto_ranges[0].3,
            "proto triangle ranges must be contiguous"
        );
        // Geometry soup byte-identical: totals == sum of inputs (image unchanged).
        assert_eq!(scene.geometry.vertex_count, sum_verts);
        assert_eq!(scene.geometry.triangle_count, sum_tris);
        scene
            .geometry
            .validate_source_clusters()
            .expect("Ochroma source clusters bind exactly to the assembled prototypes");
        assert_eq!(scene.geometry.source_triangle_order.len(), sum_tris);
        assert_eq!(scene.geometry.source_clusters.len(), 2);
        assert_eq!(scene.geometry.source_clusters[0].proto_index, 0);
        assert_eq!(scene.geometry.source_clusters[1].proto_index, 1);

        println!(
            "protos={} aabbs_distinct={} soup_verts={}(==sum) soup_tris={}(==sum)",
            scene.geometry.proto_aabbs.len(),
            aabbs_distinct,
            scene.geometry.vertex_count,
            scene.geometry.triangle_count,
        );
    }
}
