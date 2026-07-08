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
    BlasDesc, InstanceRecordGpu, PbrMaterial, VULKAN_MATERIAL_FLOATS, pack_cuda_mesh_material,
    pack_vulkan_mesh_material,
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
/// Materials are packed with [`pack_vulkan_mesh_material`] — the proven
/// **156-float Vulkan stride** (`materials.params.len() == materials.len() *
/// 156`), NEVER the 132-float CUDA `MaterialData` (the "black silhouettes"
/// landmine). `spectral_spd` populates `MaterialLayer::spectral_spd[material_id]`
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
    use spectra_scene_state::MaterialLayer;

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

    let mut vbase: u32 = 0;
    let mut tbase: u32 = 0;
    // K1: per-proto sub-ranges into the shared soup, so the uploader can build one
    // BLAS per prototype (the multi-proto TLAS that makes 100K buildings + 1M cims
    // representable instead of one merged soup).
    let mut proto_ranges: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(blas.len());
    for b in blas {
        let v_start = vbase;
        let t_start = tbase;
        debug_assert_eq!(
            b.positions.len(),
            b.normals.len(),
            "BlasDesc positions/normals length mismatch"
        );
        for p in &b.positions {
            positions.extend_from_slice(p);
        }
        for n in &b.normals {
            normals.extend_from_slice(n);
        }
        if b.uvs.len() == b.positions.len() {
            for t in &b.uvs {
                uvs.extend_from_slice(t);
            }
        } else {
            // Missing UVs → unit zeros (kept length-consistent with positions).
            uvs.extend(std::iter::repeat(0.0).take(b.positions.len() * 2));
        }
        for (ti, tri) in b.indices.iter().enumerate() {
            indices.extend_from_slice(&[vbase + tri[0], vbase + tri[1], vbase + tri[2]]);
            // Per-triangle material id (forge channel id ordered); default 0.
            // BlasDesc.material_ids is already u32 — no cast.
            let mid = b.material_ids.get(ti).copied().unwrap_or(0);
            tri_material_ids.push(mid);
        }
        let vcount = b.positions.len() as u32;
        let tcount = b.indices.len() as u32;
        proto_ranges.push((v_start, vcount, t_start, tcount));
        vbase += vcount;
        tbase += tcount;
    }

    // --- Per-prototype object-space AABBs (K0: real per-proto bounds) ---
    // One (min,max) per BlasDesc, parallel to proto order, so the TLAS can give
    // each prototype its REAL bound instead of the merged scene-wide box. This is
    // the shared instancing-keystone foundation that terrain chunks ride on.
    let proto_aabbs: Vec<([f32; 3], [f32; 3])> =
        blas.iter().map(|b| (b.aabb_min, b.aabb_max)).collect();

    // --- Instances: transforms SoA + per-instance material BASE + proto index ---
    let mut instance_transforms: Vec<f32> = Vec::with_capacity(instances.len() * 16);
    let mut instance_material_base: Vec<u32> = Vec::with_capacity(instances.len());
    let mut instance_proto_index: Vec<u32> = Vec::with_capacity(instances.len());
    for inst in instances {
        instance_transforms.extend_from_slice(&inst.transform);
        instance_material_base.push(inst.material_base);
        instance_proto_index.push(inst.proto_index);
    }

    // --- Materials: per-backend stride ---
    // The Slang `MaterialData` struct compiles to DIFFERENT memory layouts on the
    // two backends: CUDA (NVRTC) packs tight (132 floats / 528 bytes); Vulkan
    // (SPIR-V std430) pads every float3 to 16 bytes (156 floats / 624 bytes).
    // Pack the layout that matches the backend the renderer will actually select.
    // Resident rendering selects CUDA on Windows at the type level
    // (`resident_renderer::ResidentBackend = CudarcSlangBackend`). Do not depend
    // on launcher env for this ABI: a missing `SPECTRA_BACKEND=cuda` would pack
    // Vulkan/std430 material rows for CUDA kernels and shift every field after
    // the first float3. Keep an explicit env override for diagnostic runs.
    let cuda_layout = std::env::var("SPECTRA_BACKEND")
        .ok()
        .map(|s| s.eq_ignore_ascii_case("cuda"))
        .unwrap_or(cfg!(target_os = "windows"));
    let params: Vec<f32> = if cuda_layout {
        let mut p = Vec::with_capacity(materials.len() * 132);
        for m in materials {
            p.extend_from_slice(&pack_cuda_mesh_material(*m));
        }
        p
    } else {
        let mut p = Vec::with_capacity(materials.len() * VULKAN_MATERIAL_FLOATS);
        for m in materials {
            p.extend_from_slice(&pack_vulkan_mesh_material(*m));
        }
        p
    };

    // --- Spectral SPD keyed by stable material_id (empty ⇒ white) ---
    let mut spd_map = std::collections::HashMap::with_capacity(spectral_spd.len());
    for (mat_id, spd) in spectral_spd {
        spd_map.insert(*mat_id, *spd);
    }

    // --- Geometry-anchored weathering PATTERN (7 floats / scene vertex) ---
    // The cook bakes per-vertex masks, but they are not carried through the
    // HybridMesh→BlasDesc seam yet, so synthesize the PATTERN here from the
    // merged vertex geometry (position + normal). This is the cook-pattern layer
    // of [[dynamic-weathering-directive]]: WHERE aging appears (soot rising from
    // the street, water-stain weeping below sills/ledges, edge-wear on arrises,
    // efflorescence wicking up from the base). It is a pure, deterministic
    // function of the id-sorted merged geometry (no HashMap/RNG iteration) so it
    // never perturbs the determinism artifact. The sim drives per-instance
    // INTENSITY on top via `set_weathering_intensity` (future); at reference
    // intensity 1.0 this is a SUBTLE pattern (amplitudes ≤ ~0.3) so a clean city
    // still reads clean. Empty when there is no geometry.
    //
    // COOKED-MASK OVERLAY: a proto that carries its cook-baked per-vertex masks
    // (`BlasDesc::weathering_masks`, 7 floats/vertex — the HybridMesh→BlasDesc
    // seam is now carried) REPLACES the synthesized pattern over its own vertex
    // range; every other proto keeps the synthesis (byte-identical to before).
    // Iterates `blas` in the same fixed order as `proto_ranges` — deterministic.
    let mut weathering_masks = build_weathering_pattern(&positions, &normals);
    if !weathering_masks.is_empty() {
        for (b, &(v_start, vcount, _, _)) in blas.iter().zip(proto_ranges.iter()) {
            if b.weathering_masks.len() == vcount as usize * 7 {
                let dst = v_start as usize * 7;
                weathering_masks[dst..dst + b.weathering_masks.len()]
                    .copy_from_slice(&b.weathering_masks);
            }
        }
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
    scene.geometry.instance_count = instances.len();
    scene.geometry.instance_transforms = instance_transforms;
    scene.geometry.instance_material_base = instance_material_base;
    scene.geometry.instance_proto_index = instance_proto_index;
    scene.geometry.proto_aabbs = proto_aabbs;
    scene.geometry.proto_ranges = proto_ranges;
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
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [1.0, 1.0, 0.0],
            weathering_masks: Vec::new(),
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
            aabb_min: [10.0, 0.0, 0.0],
            aabb_max: [12.0, 2.0, 0.0],
            weathering_masks: Vec::new(),
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
            },
            InstanceRecordGpu {
                proto_index: 1,
                transform: ident,
                material_base: 0,
            },
            InstanceRecordGpu {
                proto_index: 0,
                transform: ident,
                material_base: 0,
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

        println!(
            "protos={} aabbs_distinct={} soup_verts={}(==sum) soup_tris={}(==sum)",
            scene.geometry.proto_aabbs.len(),
            aabbs_distinct,
            scene.geometry.vertex_count,
            scene.geometry.triangle_count,
        );
    }
}
