//! Loader for finished assets from the ordinary shipping `.vxp` packs into the
//! bench's `BlasDesc` + `PbrMaterial` types. The benchmark uses the same
//! authoritative binary geometry, surface data, and metadata that Urban
//! Horizon loads; legacy authoring/cook JSON is never a product input.
//!
//! Metadata and surface JSON are parsed generically so `vox_render` does not
//! depend on Urban Horizon's game types. Geometry is decoded by `vox_data`'s
//! canonical VXP reader. Root comes from `MEGAGEOMETRY_ASSET_ROOT` (default
//! `assets`), resolving either `<root>/official/packs` or `<root>/packs`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use vox_data::mega_geometry::ReadyMegaGeometry;
use vox_data::vxp::{VxpAssetIndex, VxpGeometry, VxpReader, content_hash};
use vox_render::MegaGeometryLayer;
use vox_render::splat_backend::{BlasDesc, PbrMaterial};

/// An ordinary finished game asset resolved for the benchmark.
///
/// The asset supplies exactly one authoritative full-detail mesh. Ochroma
/// derives the disposable runtime visibility representation below. This type
/// has no asset-authoring counterpart.
pub struct FinishedAsset {
    /// The one authoritative full-detail mesh shipped in `.vxp`.
    pub base: BlasDesc,
    /// Material palette (`materials[*]`), indexed by each triangle's material id.
    pub materials: Vec<PbrMaterial>,
    /// SHA-256 of the authoritative finished `.Geometry` bytes.
    pub content_hash: String,
    /// Ochroma-derived visibility-program payload. No authoring or metadata
    /// representation is accepted.
    pub mega: Option<Arc<ReadyMegaGeometry>>,
    pub exact_quad_count: u32,
    pub derived_program_count: u32,
    pub rejected_quad_candidates: u32,
    pub source_geometry_bytes: u64,
    pub covered_source_geometry_bytes: u64,
    pub residual_geometry_bytes: u64,
    pub runtime_program_bytes: u64,
}

/// Assemble a `MegaGeometryLayer` (the `SceneState.mega_geometry` payload the
/// renderer's native visibility-program trace consumes) from one
/// Ochroma-derived runtime `ReadyMegaGeometry` payload instanced by
/// `transforms`. Mirrors the game's
/// `assemble_mega_geometry_layer`; the single-payload case needs no offset
/// patching (all program/deformation/template/selector offsets are already
/// 0-based). Each instance references the whole program range; material remap is
/// empty (the bench's per-triangle materials carry through the base mesh).
pub fn assemble_mega_layer(
    payload: &ReadyMegaGeometry,
    transforms: &[[f32; 16]],
    material_count: u32,
) -> MegaGeometryLayer {
    let mut layer = MegaGeometryLayer::default();
    layer
        .program_words
        .extend_from_slice(payload.program_words());
    layer
        .deformation_words
        .extend_from_slice(payload.deformation_words());
    layer
        .template_words
        .extend_from_slice(payload.template_words());
    layer.selectors.extend_from_slice(payload.selectors());
    layer
        .surface_correspondence_words
        .extend_from_slice(payload.surface_correspondence_words());
    layer
        .surface_correspondence_slots
        .extend(0..payload.program_count());
    layer
        .visibility_cell_words
        .extend_from_slice(payload.visibility_cell_words());
    layer
        .visibility_cell_children
        .extend_from_slice(payload.visibility_cell_children());
    layer
        .visibility_coefficient_words
        .extend_from_slice(payload.visibility_coefficient_words());
    layer
        .visibility_coefficient_parameters
        .extend_from_slice(payload.visibility_coefficient_parameters());
    layer.aabbs.extend_from_slice(payload.aabbs());
    layer.source_patch_count = payload.source_patch_count() as u64;
    layer.source_triangle_count = payload.source_triangle_count() as u64;

    // ONE shared identity material-remap block (authored id N -> resident id N):
    // the instances share the building's material palette, so every instance's
    // range references this block. `MegaGeometryLayer::validate` requires each
    // range have count > 0 and stay within `material_remap`.
    let mc = material_count.max(1);
    for m in 0..mc {
        layer.material_remap.push((m, m));
    }

    // MegaGeometry visibility programs are derived in the finished mesh's
    // SOURCE space; the admitted mesh is in OBJECT space. The two differ by the runtime representation's
    // `source_to_object` transform (e.g. a Z-flip + origin shift). Compose it
    // into each mega instance transform exactly as the game does
    // (`mega_proto_instance_transform`, asset_placement.rs) so the realized
    // visibility surface lands ON the mesh instead of a z-flipped, offset ghost.
    //
    // Convention bridge: the bench's instance transforms are the KHR row-major
    // layout (3×3 in indices [0,1,2 / 4,5,6 / 8,9,10], translation at 12/13/14),
    // so reconstruct the mathematical `object_to_world` before composing. The
    // mega layer is consumed column-major (the uploader reads row 0 as
    // [m0,m4,m8,m12]), which is exactly `glam::Mat4::to_cols_array`.
    let source_to_object = glam::Mat4::from_cols_array(&payload.source_to_object());
    let program_count = payload.program_count();
    let visibility_cell_count =
        (payload.visibility_cell_words().len() / MegaGeometryLayer::VISIBILITY_CELL_WORDS) as u32;
    for t in transforms {
        let object_to_world = glam::Mat4::from_cols_array(&[
            t[0], t[4], t[8], 0.0, //
            t[1], t[5], t[9], 0.0, //
            t[2], t[6], t[10], 0.0, //
            t[12], t[13], t[14], 1.0,
        ]);
        let mega = (object_to_world * source_to_object).to_cols_array();
        layer.instance_transforms.extend_from_slice(&mega);
        layer.instance_ranges.push((0, program_count, 0));
        layer
            .instance_visibility_cell_ranges
            .push((0, visibility_cell_count));
        layer.material_remap_ranges.push((0, mc));
    }
    layer
}

impl FinishedAsset {
    /// Triangle count of the full-detail base mesh.
    pub fn base_triangles(&self) -> usize {
        self.base.indices.len()
    }

    /// The base mesh with the MegaGeometry-COVERED triangles removed — the
    /// `city_hybrid` residual mesh. The cooked `covered_triangle_indices` are the
    /// source triangles the visibility programs reproduce exactly, so a mega
    /// scene must trace them via the programs (realized/procedural), NOT ALSO as
    /// materialized mesh. Leaving them in the soup double-covers every covered
    /// surface: the coincident mesh + realized triangles z-fight, and the KHR
    /// tie-break reports the mesh prim id with the realized triangle's
    /// barycentrics (a spurious UV divergence against the triangle oracle). With
    /// them removed, covered pixels resolve ONLY through mega (a different prim
    /// id at the same surface — forgiven by the oracle's depth gate) and the
    /// residual mesh reproduces the rest exactly.
    ///
    /// Returns the base unchanged when there is no mega payload or no covered
    /// set. Unreferenced vertices are left in place (harmless for tracing); only
    /// per-triangle `indices` / `material_ids` are filtered.
    pub fn base_without_covered(&self) -> BlasDesc {
        let Some(mega) = self.mega.as_ref() else {
            return self.base.clone();
        };
        let covered: std::collections::HashSet<u32> =
            mega.covered_triangle_indices().iter().copied().collect();
        if covered.is_empty() {
            return self.base.clone();
        }
        let mut base = self.base.clone();
        let has_mat = base.material_ids.len() == base.indices.len();
        let mut kept_indices = Vec::with_capacity(base.indices.len());
        let mut kept_mats = Vec::with_capacity(base.material_ids.len());
        for (tri, idx) in base.indices.iter().enumerate() {
            if covered.contains(&(tri as u32)) {
                continue;
            }
            kept_indices.push(*idx);
            if has_mat {
                kept_mats.push(base.material_ids[tri]);
            }
        }
        base.indices = kept_indices;
        base.material_ids = if has_mat {
            kept_mats
        } else {
            base.material_ids
        };
        base
    }
}

fn geometry_subset_bytes(base: &BlasDesc, selected: impl Fn(usize) -> bool) -> u64 {
    let mut vertices = std::collections::BTreeSet::<u32>::new();
    let mut triangles = 0_u64;
    for (triangle_index, triangle) in base.indices.iter().enumerate() {
        if selected(triangle_index) {
            triangles = triangles.saturating_add(1);
            vertices.extend(triangle);
        }
    }
    let vertex_count = vertices.len() as u64;
    let mut vertex_stride = 12_u64;
    if base.normals.len() == base.positions.len() {
        vertex_stride = vertex_stride.saturating_add(12);
    }
    if base.uvs.len() == base.positions.len() {
        vertex_stride = vertex_stride.saturating_add(8);
    }
    if base.weathering_masks.len() == base.positions.len() * 7 {
        vertex_stride = vertex_stride.saturating_add(28);
    }
    let triangle_stride = 12_u64
        + if base.material_ids.len() == base.indices.len() {
            4
        } else {
            0
        };
    vertex_count
        .saturating_mul(vertex_stride)
        .saturating_add(triangles.saturating_mul(triangle_stride))
}

fn runtime_program_bytes(payload: &ReadyMegaGeometry) -> u64 {
    let words = payload
        .program_words()
        .len()
        .saturating_add(payload.deformation_words().len())
        .saturating_add(payload.template_words().len())
        .saturating_add(payload.selectors().len())
        .saturating_add(payload.aabbs().len())
        .saturating_add(payload.surface_correspondence_words().len())
        .saturating_add(payload.page_hierarchy_words().len())
        .saturating_add(payload.visibility_cell_words().len())
        .saturating_add(payload.visibility_cell_children().len())
        .saturating_add(payload.visibility_coefficient_words().len())
        .saturating_add(payload.visibility_coefficient_parameters().len());
    (words as u64).saturating_mul(4)
}

fn parse_geometry(geometry: VxpGeometry, proto_id: u64) -> Result<BlasDesc, String> {
    let positions = geometry.positions;
    if positions.is_empty() {
        return Err("mesh has no positions".into());
    }
    let indices = geometry.indices;
    if indices.is_empty() {
        return Err("mesh has no indices".into());
    }
    let normals = (!geometry.normals.is_empty())
        .then_some(geometry.normals)
        .filter(|n| n.len() == positions.len())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs = (!geometry.uvs.is_empty())
        .then_some(geometry.uvs)
        .filter(|u| u.len() == positions.len())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    // `.vxp` stores the per-triangle material palette index in the named
    // `material_indices` u8 stream.
    let material_ids = geometry
        .aux_u8
        .get("material_indices")
        .map(|ids| ids.iter().map(|&id| u32::from(id)).collect::<Vec<_>>())
        .filter(|m| m.len() == indices.len())
        .unwrap_or_default();
    let weathering_masks = geometry
        .aux_f32
        .get("weathering_masks")
        .filter(|(components, values)| *components == 7 && values.len() == positions.len() * 7)
        .map(|(_, values)| values.clone())
        .unwrap_or_default();

    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    Ok(BlasDesc {
        proto_id,
        positions,
        normals,
        uvs,
        indices,
        material_ids,
        aabb_min: lo,
        aabb_max: hi,
        weathering_masks,
    })
}

fn parse_materials(v: &serde_json::Value) -> Vec<PbrMaterial> {
    let Some(arr) = v.as_array() else {
        return vec![PbrMaterial {
            base_color: [0.6, 0.6, 0.6],
            roughness: 0.6,
            ..Default::default()
        }];
    };
    let out: Vec<PbrMaterial> = arr
        .iter()
        .map(|m| {
            let bc = m.get("base_color_factor").and_then(|v| v.as_array());
            let g = |i: usize| {
                bc.and_then(|a| a.get(i))
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.6) as f32
            };
            PbrMaterial {
                base_color: [g(0), g(1), g(2)],
                roughness: m
                    .get("roughness_factor")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.6) as f32,
                metallic: m
                    .get("metallic_factor")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0) as f32,
                ..Default::default()
            }
        })
        .collect();
    if out.is_empty() {
        vec![PbrMaterial {
            base_color: [0.6, 0.6, 0.6],
            roughness: 0.6,
            ..Default::default()
        }]
    } else {
        out
    }
}

/// Root directory for finished assets (`MEGAGEOMETRY_ASSET_ROOT`, default
/// `assets`).
fn asset_root() -> PathBuf {
    std::env::var_os("MEGAGEOMETRY_ASSET_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets"))
}

fn pack_root() -> PathBuf {
    let root = asset_root();
    if root.join("packs").is_dir() {
        root.join("packs")
    } else {
        root.join("official").join("packs")
    }
}

fn pack_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = std::fs::read_dir(root)
        .map_err(|error| format!("read finished pack directory {}: {error}", root.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("vxp"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "finished pack directory {} contains no .vxp files",
            root.display()
        ));
    }
    Ok(paths)
}

fn find_asset(asset_id: &str) -> Result<(VxpReader, VxpAssetIndex), String> {
    let root = pack_root();
    for path in pack_paths(&root)? {
        let reader = VxpReader::open(&path)
            .map_err(|error| format!("open finished pack {}: {error}", path.display()))?;
        if let Some(asset) = reader
            .index()
            .assets
            .iter()
            .find(|asset| asset.asset_id == asset_id)
            .cloned()
        {
            return Ok((reader, asset));
        }
    }
    Err(format!(
        "asset {asset_id:?} is absent from ordinary finished packs in {}",
        root.display()
    ))
}

fn parse_finished_asset(
    mut reader: VxpReader,
    record: &VxpAssetIndex,
    proto_id: u64,
) -> Result<FinishedAsset, String> {
    // Index v2 predates Ochroma-owned continuous detail and may carry legacy
    // camera-distance meshes. They are deliberately not read or admitted:
    // only the authoritative base geometry enters MegaGeometry. Any current
    // pack (v3+) carrying them violates the finished-object contract.
    if reader.index().version >= 3 && !record.geometry_lods.is_empty() {
        return Err(format!(
            "{} carries retired authored LOD geometry",
            record.asset_id
        ));
    }
    let geometry_entry = record
        .geometry
        .as_deref()
        .ok_or_else(|| format!("{} has no finished .Geometry entry", record.asset_id))?;
    let geometry_bytes = reader
        .read_entry(geometry_entry)
        .map_err(|error| format!("read {geometry_entry}: {error}"))?;
    let geometry = VxpGeometry::decode(&geometry_bytes)
        .map_err(|error| format!("decode {geometry_entry}: {error}"))?;
    let has_source_uvs = !geometry.uvs.is_empty();
    let base = parse_geometry(geometry, proto_id)?;
    let empty_uvs = Vec::new();
    let derived = vox_render::mesh_program::prepare_mesh_programs(
        vox_render::mesh_program::MeshProgramInput {
            positions: &base.positions,
            indices: &base.indices,
            uvs: if has_source_uvs {
                &base.uvs
            } else {
                &empty_uvs
            },
            material_ids: &base.material_ids,
        },
    )
    .map_err(|error| {
        format!(
            "derive Ochroma MegaGeometry from {}: {error}",
            record.asset_id
        )
    })?
    .derivation;

    let surface_entry = record
        .surface
        .as_deref()
        .ok_or_else(|| format!("{} has no finished .Surface entry", record.asset_id))?;
    let surface_bytes = reader
        .read_entry(surface_entry)
        .map_err(|error| format!("read {surface_entry}: {error}"))?;
    let surface: serde_json::Value = serde_json::from_slice(&surface_bytes)
        .map_err(|error| format!("parse {surface_entry}: {error}"))?;
    let materials = parse_materials(surface.get("materials").unwrap_or(&serde_json::Value::Null));

    let metadata_entry = record
        .metadata
        .as_deref()
        .ok_or_else(|| format!("{} has no finished .Metadata entry", record.asset_id))?;
    let metadata_bytes = reader
        .read_entry(metadata_entry)
        .map_err(|error| format!("read {metadata_entry}: {error}"))?;
    let _metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)
        .map_err(|error| format!("parse {metadata_entry}: {error}"))?;
    let mega = derived.payload.map(Arc::new);
    let covered = mega
        .as_ref()
        .map(|payload| {
            payload
                .covered_triangle_indices()
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    let source_geometry_bytes = geometry_subset_bytes(&base, |_| true);
    let covered_source_geometry_bytes =
        geometry_subset_bytes(&base, |triangle| covered.contains(&(triangle as u32)));
    let residual_geometry_bytes =
        geometry_subset_bytes(&base, |triangle| !covered.contains(&(triangle as u32)));
    let runtime_program_bytes = mega
        .as_deref()
        .map(runtime_program_bytes)
        .unwrap_or_default();
    let content_hash = content_hash(&geometry_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    Ok(FinishedAsset {
        base,
        materials,
        content_hash,
        mega,
        exact_quad_count: derived.exact_quad_count,
        derived_program_count: derived.program_count,
        rejected_quad_candidates: derived.rejected_quad_candidates,
        source_geometry_bytes,
        covered_source_geometry_bytes,
        residual_geometry_bytes,
        runtime_program_bytes,
    })
}

fn cache() -> &'static Mutex<HashMap<String, Arc<FinishedAsset>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<FinishedAsset>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load (and cache) a finished asset for a fixture `asset_id`. The `.vxp`
/// entries are parsed once per id and shared across modes and instances.
pub fn load_fixture_asset(asset_id: &str, proto_id: u64) -> Result<Arc<FinishedAsset>, String> {
    if let Some(hit) = cache().lock().unwrap().get(asset_id).cloned() {
        return Ok(hit);
    }
    let (reader, record) = find_asset(asset_id)?;
    let asset = Arc::new(parse_finished_asset(reader, &record, proto_id)?);
    cache()
        .lock()
        .unwrap()
        .insert(asset_id.to_string(), asset.clone());
    Ok(asset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_data::vxp::VxpWriter;

    fn fixture_geometry() -> VxpGeometry {
        let mut geometry = VxpGeometry {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![[0, 1, 2]],
            ..Default::default()
        };
        geometry.aux_u8.insert("material_indices".into(), vec![2]);
        geometry
            .aux_f32
            .insert("weathering_masks".into(), (7, vec![0.25; 21]));
        geometry
    }

    fn fixture_pack() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.vxp");
        let mut writer = VxpWriter::new("fixture");
        writer.add_geometry("city.fixture", fixture_geometry().encode());
        writer.add_surface(
            "city.fixture",
            serde_json::to_vec(&serde_json::json!({
                "materials": [{
                    "base_color_factor": [0.2, 0.3, 0.4],
                    "roughness_factor": 0.7,
                    "metallic_factor": 0.1
                }],
                "surfaces": [],
                "material_zones": []
            }))
            .unwrap(),
        );
        writer.add_metadata(
            "city.fixture",
            serde_json::to_vec(&serde_json::json!({
                "asset_id": "city.fixture"
            }))
            .unwrap(),
        );
        writer.finish(&path, false).unwrap();
        (dir, path)
    }

    #[test]
    fn finished_pack_loader_uses_binary_geometry_surface_and_metadata() {
        let (_dir, path) = fixture_pack();
        let reader = VxpReader::open(&path).unwrap();
        let record = reader.index().assets[0].clone();
        let asset = parse_finished_asset(reader, &record, 41).unwrap();
        assert_eq!(asset.base.proto_id, 41);
        assert_eq!(asset.base.indices, vec![[0, 1, 2]]);
        assert_eq!(asset.base.material_ids, vec![2]);
        assert_eq!(asset.base.weathering_masks, vec![0.25; 21]);
        assert_eq!(asset.materials[0].base_color, [0.2, 0.3, 0.4]);
        assert_eq!(asset.content_hash.len(), 64);
        assert!(asset.mega.is_none());
        assert_eq!(asset.source_geometry_bytes, 196);
        assert_eq!(asset.covered_source_geometry_bytes, 0);
        assert_eq!(asset.residual_geometry_bytes, 196);
        assert_eq!(asset.runtime_program_bytes, 0);
    }

    #[test]
    fn finished_pack_loader_rejects_asset_authored_lods() {
        let (_dir, path) = fixture_pack();
        let reader = VxpReader::open(&path).unwrap();
        let mut record = reader.index().assets[0].clone();
        // Simulate a legacy pack index. The v4 writer has no API capable of
        // producing this state.
        record
            .geometry_lods
            .push((1, "legacy_LOD1.Geometry".to_string()));
        let error = match parse_finished_asset(reader, &record, 41) {
            Ok(_) => panic!("authored LOD geometry was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("retired authored LOD geometry"), "{error}");
    }

    /// Explicit product-pack admission probe. It is ignored in ordinary CI
    /// because CI does not carry Urban Horizon's multi-gigabyte official
    /// packs; invoke it with `MEGAGEOMETRY_ASSET_ROOT=<urban>/assets`.
    #[test]
    #[ignore = "requires installed Urban Horizon official .vxp packs"]
    fn installed_product_fixtures_load_only_from_finished_packs() {
        for (asset_id, proto_id) in [
            ("city.res_high.l5.8x8.meridian_house_01", 1),
            ("city.res_low.l1.2x2.cottage_01", 2),
        ] {
            let asset = load_fixture_asset(asset_id, proto_id)
                .unwrap_or_else(|error| panic!("{asset_id}: {error}"));
            assert!(!asset.base.positions.is_empty(), "{asset_id}");
            assert!(!asset.base.indices.is_empty(), "{asset_id}");
            assert_eq!(asset.content_hash.len(), 64, "{asset_id}");
            eprintln!(
                "FINISHED_OBJECT_DERIVATION asset={} triangles={} exact_quads={} programs={} rejected_candidates={} covered_triangles={} source_bytes={} covered_source_bytes={} residual_bytes={} runtime_program_bytes={}",
                asset_id,
                asset.base.indices.len(),
                asset.exact_quad_count,
                asset.derived_program_count,
                asset.rejected_quad_candidates,
                asset
                    .mega
                    .as_ref()
                    .map_or(0, |mega| mega.source_triangle_count()),
                asset.source_geometry_bytes,
                asset.covered_source_geometry_bytes,
                asset.residual_geometry_bytes,
                asset.runtime_program_bytes,
            );
            assert!(
                asset
                    .mega
                    .as_ref()
                    .is_some_and(|mega| mega.validate().is_ok()),
                "{asset_id} has no valid finished-object MegaGeometry: exact_quads={} programs={} rejected_candidates={}",
                asset.exact_quad_count,
                asset.derived_program_count,
                asset.rejected_quad_candidates,
            );
        }
    }
}
