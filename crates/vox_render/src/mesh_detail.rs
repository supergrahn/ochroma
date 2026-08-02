//! Engine-owned deterministic triangle-detail derivation.
//!
//! One authoritative mesh enters Ochroma. This module derives bounded
//! coarse-to-fine triangle representations without accepting authored distance
//! meshes. Material/UV/hard-normal seams are split before simplification, and
//! every child triangle receives an affine barycentric map onto one
//! orientation-compatible parent triangle of the same material.

use crate::mesh_simplify::{MeshInput, MeshOutput, simplify_mesh};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-local structural-admission count. Derivation is engine-owned, keyed
/// by authoritative mesh bytes, and never runs per frame.
static DETAIL_DERIVATION_CALLS: AtomicU64 = AtomicU64::new(0);
static CACHE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static RUNTIME_DETAIL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

pub fn detail_derivation_call_count() -> u64 {
    DETAIL_DERIVATION_CALLS.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleSurfaceCorrespondence {
    pub child_triangle: u32,
    pub parent_triangle: u32,
    /// Parent barycentrics at each child vertex. Interpolating these three
    /// records by a child hit's barycentrics maps every point affinely.
    pub parent_barycentric: [[f32; 3]; 3],
    /// Conservative object-space deviation for the complete child triangle.
    pub max_position_error: f32,
    /// Conservative UV deviation for the complete child triangle.
    pub max_uv_error: f32,
    pub orientation_dot: f32,
    pub material_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DerivedDetailLevel {
    pub group_id: u32,
    pub parent_group_id: Option<u32>,
    pub mesh: MeshOutput,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub weathering_masks: Vec<[f32; 7]>,
    pub accumulated_world_error: f32,
    /// Primitive count of the hierarchy's stable identity root. Carried by every
    /// independently decodable page so direct-root maps can be range-checked
    /// without requiring that root page to be resident.
    pub root_triangle_count: u32,
    /// Empty for the identity root; level `n` maps to level `n - 1`.
    pub correspondence_to_parent: Vec<TriangleSurfaceCorrespondence>,
    /// Empty for the identity root; every other level maps directly to the
    /// identity root. Other hierarchy pages therefore need not be
    /// resident to recover stable temporal surface identity.
    pub correspondence_to_root: Vec<TriangleSurfaceCorrespondence>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DerivedDetailHierarchy {
    /// Coarsest identity parent first, authoritative full detail last.
    pub levels: Vec<DerivedDetailLevel>,
}

#[derive(Clone, Debug)]
pub struct RuntimeDetailPreparation {
    pub clusters: vox_data::geometry_clusters::ReadyGeometryClusters,
    pub cache_path: Option<PathBuf>,
    pub cache_hit: bool,
    pub page_count: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeDetailError {
    #[error(transparent)]
    Derivation(#[from] DetailDerivationError),
    #[error("source cluster derivation failed: {0}")]
    Clusters(#[from] vox_data::geometry_clusters::ReadyGeometryClustersError),
    #[error("runtime detail cache I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("runtime detail cache address space overflow")]
    AddressOverflow,
    #[error("runtime detail cache content failed deterministic verification")]
    CacheCorrupt,
}

/// Derive the residual continuous-detail hierarchy from a finished mesh and
/// publish it as an engine-owned immutable cache file for Spectra's existing
/// asynchronous page transport. The cache is not an asset sidecar: it is
/// content-addressed runtime state and can be deleted/rebuilt without changing
/// the game object.
pub fn prepare_runtime_detail(
    full_positions: &[[f32; 3]],
    full_indices: &[[u32; 3]],
    residual_input: DetailMeshInput<'_>,
    residual_source_triangles: &[u32],
    ratios: &[f32],
) -> Result<RuntimeDetailPreparation, RuntimeDetailError> {
    let mut clusters = vox_data::geometry_clusters::ReadyGeometryClusters::from_source_mesh(
        full_positions,
        full_indices,
    )?;
    if residual_input.indices.is_empty() {
        return Ok(RuntimeDetailPreparation {
            clusters,
            cache_path: None,
            cache_hit: true,
            page_count: 0,
        });
    }
    validate_runtime_detail_request(
        full_positions,
        full_indices,
        residual_input,
        residual_source_triangles,
        ratios,
    )?;
    let cache_key = runtime_detail_source_key(
        full_positions,
        full_indices,
        residual_input,
        residual_source_triangles,
        ratios,
    );
    let cache_root = runtime_detail_cache_root();
    std::fs::create_dir_all(&cache_root).map_err(|source| RuntimeDetailError::Io {
        path: cache_root.clone(),
        source,
    })?;
    let cache_path = cache_root.join(format!("{}.mgc", hex_hash(cache_key)));
    if let Some(runtime_pages) = load_cache_runtime_pages(&cache_path, cache_key)? {
        let page_count = runtime_pages.len() as u32;
        clusters.set_runtime_pages(runtime_pages);
        return Ok(RuntimeDetailPreparation {
            clusters,
            cache_path: Some(cache_path),
            cache_hit: true,
            page_count,
        });
    }
    RUNTIME_DETAIL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    let levels = derive_clustered_detail_levels(
        &clusters,
        residual_input,
        residual_source_triangles,
        full_indices.len(),
        ratios,
    )?;
    let page_count = levels.len() as u32;
    let mut encoded = Vec::with_capacity(levels.len());
    // Consume each decoded level as soon as its independently streamable page
    // exists. Retaining both complete hierarchy forms doubled a cold-cache job.
    for level in levels {
        encoded.push(encode_detail_page(&level)?);
    }
    write_cache_pages_atomically(&cache_path, cache_key, &encoded)?;
    drop(encoded);
    let runtime_pages = load_cache_runtime_pages(&cache_path, cache_key)?
        .ok_or(RuntimeDetailError::CacheCorrupt)?;
    clusters.set_runtime_pages(runtime_pages);
    Ok(RuntimeDetailPreparation {
        clusters,
        cache_path: Some(cache_path),
        cache_hit: false,
        page_count,
    })
}

fn validate_runtime_detail_request(
    full_positions: &[[f32; 3]],
    full_indices: &[[u32; 3]],
    residual_input: DetailMeshInput<'_>,
    residual_source_triangles: &[u32],
    ratios: &[f32],
) -> Result<(), RuntimeDetailError> {
    if full_positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
        || full_indices
            .iter()
            .flatten()
            .any(|index| *index as usize >= full_positions.len())
        || residual_source_triangles.len() != residual_input.indices.len()
        || residual_source_triangles
            .iter()
            .any(|triangle| *triangle as usize >= full_indices.len())
    {
        return Err(RuntimeDetailError::Derivation(
            DetailDerivationError::InvalidSource,
        ));
    }
    validate_input(residual_input)?;
    validate_ratios(ratios)?;
    Ok(())
}

fn remap_source_triangles(
    hierarchy: &mut DerivedDetailHierarchy,
    residual_source_triangles: &[u32],
    full_triangle_count: usize,
) -> Result<(), RuntimeDetailError> {
    for level in &mut hierarchy.levels {
        for source in &mut level.mesh.source_triangle_indices {
            *source = *residual_source_triangles
                .get(*source as usize)
                .ok_or(DetailDerivationError::InvalidSource)?;
            if *source as usize >= full_triangle_count {
                return Err(DetailDerivationError::InvalidSource.into());
            }
        }
    }
    Ok(())
}

/// Build one independently streamable hierarchy per canonical source cluster.
///
/// A whole-building page makes the largest building dictate every fixed GPU
/// slot and forces unrelated city blocks to become resident together. Keeping
/// the existing 256-triangle source partition as the page domain bounds decode
/// storage, native rebuild work, and eviction granularity without changing the
/// finished asset or accepting an authored LOD.
fn derive_clustered_detail_levels(
    clusters: &vox_data::geometry_clusters::ReadyGeometryClusters,
    residual_input: DetailMeshInput<'_>,
    residual_source_triangles: &[u32],
    full_triangle_count: usize,
    ratios: &[f32],
) -> Result<Vec<DerivedDetailLevel>, RuntimeDetailError> {
    let residual_by_source = residual_source_triangles
        .iter()
        .copied()
        .enumerate()
        .map(|(local, source)| (source, local))
        .collect::<BTreeMap<_, _>>();
    let mut levels = Vec::new();
    for cluster in clusters.clusters() {
        let start = cluster.triangle_order_start() as usize;
        let end = start
            .checked_add(cluster.triangle_count() as usize)
            .ok_or(RuntimeDetailError::AddressOverflow)?;
        let mut local_indices = Vec::new();
        let mut local_materials = Vec::new();
        let mut local_sources = Vec::new();
        for &source_triangle in clusters
            .triangle_order()
            .get(start..end)
            .ok_or(DetailDerivationError::InvalidSource)?
        {
            let Some(&local) = residual_by_source.get(&source_triangle) else {
                continue;
            };
            local_indices.push(residual_input.indices[local]);
            local_materials.push(residual_input.material_ids[local]);
            local_sources.push(source_triangle);
        }
        if local_indices.is_empty() {
            continue;
        }
        let mut hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: residual_input.positions,
                normals: residual_input.normals,
                uvs: residual_input.uvs,
                indices: &local_indices,
                material_ids: &local_materials,
                weathering_masks: residual_input.weathering_masks,
            },
            ratios,
        )?;
        remap_source_triangles(&mut hierarchy, &local_sources, full_triangle_count)?;
        let group_base =
            u32::try_from(levels.len()).map_err(|_| RuntimeDetailError::AddressOverflow)?;
        for level in &mut hierarchy.levels {
            level.group_id = level
                .group_id
                .checked_add(group_base)
                .ok_or(RuntimeDetailError::AddressOverflow)?;
            level.parent_group_id = match level.parent_group_id {
                Some(parent) => Some(
                    parent
                        .checked_add(group_base)
                        .ok_or(RuntimeDetailError::AddressOverflow)?,
                ),
                None => None,
            };
        }
        levels.extend(hierarchy.levels);
    }
    if levels.is_empty() {
        return Err(DetailDerivationError::EmptyLevel.into());
    }
    Ok(levels)
}

fn runtime_detail_cache_root() -> PathBuf {
    dirs_next::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("ochroma")
        .join("mega_geometry")
}

fn runtime_detail_source_key(
    full_positions: &[[f32; 3]],
    full_indices: &[[u32; 3]],
    residual_input: DetailMeshInput<'_>,
    residual_source_triangles: &[u32],
    ratios: &[f32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"OCHROMA_RUNTIME_MEGAGEOMETRY_SOURCE");
    hash.update(DETAIL_CACHE_SCHEMA.to_le_bytes());
    hash.update(DETAIL_PAGE_SCHEMA.to_le_bytes());
    hash_f32_arrays(&mut hash, full_positions);
    hash_u32_arrays(&mut hash, full_indices);
    hash_f32_arrays(&mut hash, residual_input.positions);
    hash_f32_arrays(&mut hash, residual_input.normals);
    hash_f32_arrays(&mut hash, residual_input.uvs);
    hash_u32_arrays(&mut hash, residual_input.indices);
    hash_u32s(&mut hash, residual_input.material_ids);
    hash_f32_arrays(&mut hash, residual_input.weathering_masks);
    hash_u32s(&mut hash, residual_source_triangles);
    hash.update((ratios.len() as u64).to_le_bytes());
    for ratio in ratios {
        hash.update(ratio.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}

fn hash_f32_arrays<const N: usize>(hash: &mut Sha256, values: &[[f32; N]]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values.iter().flatten() {
        hash.update(value.to_bits().to_le_bytes());
    }
}

fn hash_u32_arrays<const N: usize>(hash: &mut Sha256, values: &[[u32; N]]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values.iter().flatten() {
        hash.update(value.to_le_bytes());
    }
}

fn hash_u32s(hash: &mut Sha256, values: &[u32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
}

const DETAIL_CACHE_MAGIC: [u8; 4] = *b"MGCF";
const DETAIL_CACHE_SCHEMA: u32 = 2;
const DETAIL_CACHE_FIXED_HEADER: usize = 4 + 4 + 32 + 4;
const DETAIL_CACHE_TABLE_ENTRY: usize = 8 + 32;

#[cfg(test)]
struct DecodedDetailCache {
    levels: Vec<DerivedDetailLevel>,
    pages: Vec<Vec<u8>>,
    page_offsets: Vec<u64>,
}

#[derive(Clone, Copy)]
struct CachedPageSummary {
    group_id: u32,
    parent_group_id: Option<u32>,
    root_index: usize,
    vertex_count: u32,
    primitive_count: u32,
    geometric_error_q: u32,
    source_offset: u64,
    byte_count: u32,
    decoded_byte_count: u32,
    content_hash: [u8; 32],
}

/// Validate a detail cache one independently decodable page at a time and emit
/// only the runtime page table. A cache hit used to hold three complete forms at
/// once: `std::fs::read` of the whole file, copied encoded pages, and every
/// decoded hierarchy level. Runtime streaming needs none of those bodies after
/// their descriptor has been checked; the file itself remains the page source.
fn load_cache_runtime_pages(
    path: &Path,
    expected_key: [u8; 32],
) -> Result<Option<Vec<vox_data::mega_geometry::RuntimeGeometryPage>>, RuntimeDetailError> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(RuntimeDetailError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let file_len = file
        .metadata()
        .map_err(|source| RuntimeDetailError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut reader = std::io::BufReader::new(file);
    let mut fixed = [0u8; DETAIL_CACHE_FIXED_HEADER];
    if let Err(error) = reader.read_exact(&mut fixed) {
        return if error.kind() == std::io::ErrorKind::UnexpectedEof {
            Ok(None)
        } else {
            Err(RuntimeDetailError::Io {
                path: path.to_path_buf(),
                source: error,
            })
        };
    }
    if fixed[..4] != DETAIL_CACHE_MAGIC
        || u32::from_le_bytes(fixed[4..8].try_into().expect("four-byte schema"))
            != DETAIL_CACHE_SCHEMA
        || fixed[8..40] != expected_key
    {
        return Ok(None);
    }
    let page_count =
        u32::from_le_bytes(fixed[40..44].try_into().expect("four-byte page count")) as usize;
    if page_count == 0 {
        return Ok(None);
    }
    let table_bytes = page_count
        .checked_mul(DETAIL_CACHE_TABLE_ENTRY)
        .ok_or(RuntimeDetailError::AddressOverflow)?;
    let mut records = Vec::with_capacity(page_count);
    let mut table_entry = [0u8; DETAIL_CACHE_TABLE_ENTRY];
    for _ in 0..page_count {
        if reader.read_exact(&mut table_entry).is_err() {
            return Ok(None);
        }
        let len = u64::from_le_bytes(table_entry[..8].try_into().expect("eight-byte page len"));
        let hash = table_entry[8..]
            .try_into()
            .expect("thirty-two-byte page hash");
        records.push((len, hash));
    }
    let first_page_offset = DETAIL_CACHE_FIXED_HEADER
        .checked_add(table_bytes)
        .ok_or(RuntimeDetailError::AddressOverflow)?;
    let mut source_offset =
        u64::try_from(first_page_offset).map_err(|_| RuntimeDetailError::AddressOverflow)?;
    let mut summaries: Vec<CachedPageSummary> = Vec::with_capacity(page_count);
    let mut root_bounds = vec![([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]); page_count];
    for (index, (len, expected_hash)) in records.into_iter().enumerate() {
        let len = usize::try_from(len).map_err(|_| RuntimeDetailError::AddressOverflow)?;
        let mut page = Vec::new();
        page.try_reserve_exact(len)
            .map_err(|_| RuntimeDetailError::AddressOverflow)?;
        page.resize(len, 0);
        if reader.read_exact(&mut page).is_err()
            || <[u8; 32]>::from(Sha256::digest(&page)) != expected_hash
        {
            return Ok(None);
        }
        let level = match decode_detail_page(&page) {
            Ok(level) => level,
            Err(_) => return Ok(None),
        };
        if level.group_id as usize != index
            || level
                .parent_group_id
                .is_some_and(|parent| parent as usize >= index)
        {
            return Ok(None);
        }
        let root_index = level
            .parent_group_id
            .map(|parent| summaries[parent as usize].root_index)
            .unwrap_or(index);
        for position in &level.mesh.positions {
            for axis in 0..3 {
                root_bounds[root_index].0[axis] =
                    root_bounds[root_index].0[axis].min(position[axis]);
                root_bounds[root_index].1[axis] =
                    root_bounds[root_index].1[axis].max(position[axis]);
            }
        }
        let vertex_count = u32::try_from(level.mesh.positions.len())
            .map_err(|_| RuntimeDetailError::AddressOverflow)?;
        let primitive_count = u32::try_from(level.mesh.indices.len())
            .map_err(|_| RuntimeDetailError::AddressOverflow)?;
        let decoded_byte_count = u64::from(vertex_count)
            .checked_mul(24 * 4)
            .and_then(|vertex_bytes| {
                u64::from(primitive_count)
                    .checked_mul((3 + 1 + 1 + 15) * 4)
                    .and_then(|triangle_bytes| vertex_bytes.checked_add(triangle_bytes))
            })
            .and_then(|bytes| u32::try_from(bytes).ok())
            .ok_or(RuntimeDetailError::AddressOverflow)?;
        summaries.push(CachedPageSummary {
            group_id: level.group_id,
            parent_group_id: level.parent_group_id,
            root_index,
            vertex_count,
            primitive_count,
            geometric_error_q: (level.accumulated_world_error.max(0.0) * 1024.0)
                .ceil()
                .min(u32::MAX as f32) as u32,
            source_offset,
            byte_count: u32::try_from(len).map_err(|_| RuntimeDetailError::AddressOverflow)?,
            decoded_byte_count,
            content_hash: expected_hash,
        });
        source_offset = source_offset
            .checked_add(u64::try_from(len).map_err(|_| RuntimeDetailError::AddressOverflow)?)
            .ok_or(RuntimeDetailError::AddressOverflow)?;
    }
    if source_offset != file_len {
        return Ok(None);
    }
    let runtime_pages = summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| {
            let bounds = root_bounds[summary.root_index];
            vox_data::mega_geometry::RuntimeGeometryPage {
                page_id: index as u32,
                kind: vox_data::mega_geometry::GEOMETRY_PAGE_KIND_TRIANGLE_DETAIL,
                group_id: summary.group_id,
                vertex_count: summary.vertex_count,
                primitive_count: summary.primitive_count,
                bounds_min: bounds.0,
                bounds_max: bounds.1,
                geometric_error_q: summary.geometric_error_q,
                parent_page_id: summary.parent_group_id,
                source_pack_id: 0,
                source_path: path.to_path_buf(),
                source_offset: summary.source_offset,
                byte_count: summary.byte_count,
                decoded_byte_count: summary.decoded_byte_count,
                codec: vox_data::mega_geometry::GEOMETRY_PAGE_CODEC_TRIANGLE_DETAIL_MGTL_V3,
                codec_version: DETAIL_PAGE_SCHEMA,
                flags: 0,
                content_hash: summary.content_hash,
            }
        })
        .collect();
    Ok(Some(runtime_pages))
}

fn write_cache_pages_atomically(
    path: &Path,
    source_key: [u8; 32],
    pages: &[Vec<u8>],
) -> Result<(), RuntimeDetailError> {
    let parent = path.parent().ok_or(RuntimeDetailError::CacheCorrupt)?;
    let sequence = CACHE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("detail"),
        std::process::id(),
        sequence,
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|source| RuntimeDetailError::Io {
            path: temp.clone(),
            source,
        })?;
    let write = |file: &mut std::fs::File, bytes: &[u8]| {
        file.write_all(bytes).map_err(|source| RuntimeDetailError::Io {
            path: temp.clone(),
            source,
        })
    };
    write(&mut file, &DETAIL_CACHE_MAGIC)?;
    write(&mut file, &DETAIL_CACHE_SCHEMA.to_le_bytes())?;
    write(&mut file, &source_key)?;
    write(
        &mut file,
        &u32::try_from(pages.len())
            .map_err(|_| RuntimeDetailError::AddressOverflow)?
            .to_le_bytes(),
    )?;
    for page in pages {
        write(
            &mut file,
            &u64::try_from(page.len())
                .map_err(|_| RuntimeDetailError::AddressOverflow)?
                .to_le_bytes(),
        )?;
        write(&mut file, &<[u8; 32]>::from(Sha256::digest(page)))?;
    }
    for page in pages {
        write(&mut file, page)?;
    }
    file.sync_all().map_err(|source| RuntimeDetailError::Io {
        path: temp.clone(),
        source,
    })?;
    drop(file);
    if let Err(first_error) = std::fs::rename(&temp, path) {
        // Windows cannot atomically replace an existing destination. Another
        // process may have published the same content-addressed cache first;
        // validate it through the bounded reader before replacing anything.
        if load_cache_runtime_pages(path, source_key)?.is_some() {
            let _ = std::fs::remove_file(&temp);
        } else {
            if path.exists() {
                std::fs::remove_file(path).map_err(|source| RuntimeDetailError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
            }
            std::fs::rename(&temp, path).map_err(|source| RuntimeDetailError::Io {
                path: path.to_path_buf(),
                source: if source.kind() == std::io::ErrorKind::AlreadyExists {
                    first_error
                } else {
                    source
                },
            })?;
        }
    }
    #[cfg(unix)]
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| RuntimeDetailError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    Ok(())
}

#[cfg(test)]
fn encode_cache_file(
    source_key: [u8; 32],
    pages: &[Vec<u8>],
) -> Result<Vec<u8>, RuntimeDetailError> {
    let table_bytes = pages
        .len()
        .checked_mul(DETAIL_CACHE_TABLE_ENTRY)
        .ok_or(RuntimeDetailError::AddressOverflow)?;
    let page_bytes = pages.iter().try_fold(0_usize, |total, page| {
        total
            .checked_add(page.len())
            .ok_or(RuntimeDetailError::AddressOverflow)
    })?;
    let capacity = DETAIL_CACHE_FIXED_HEADER
        .checked_add(table_bytes)
        .and_then(|value| value.checked_add(page_bytes))
        .ok_or(RuntimeDetailError::AddressOverflow)?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&DETAIL_CACHE_MAGIC);
    out.extend_from_slice(&DETAIL_CACHE_SCHEMA.to_le_bytes());
    out.extend_from_slice(&source_key);
    out.extend_from_slice(
        &u32::try_from(pages.len())
            .map_err(|_| RuntimeDetailError::AddressOverflow)?
            .to_le_bytes(),
    );
    for page in pages {
        out.extend_from_slice(
            &u64::try_from(page.len())
                .map_err(|_| RuntimeDetailError::AddressOverflow)?
                .to_le_bytes(),
        );
        out.extend_from_slice(&<[u8; 32]>::from(Sha256::digest(page)));
    }
    for page in pages {
        out.extend_from_slice(page);
    }
    Ok(out)
}

#[cfg(test)]
fn decode_cache_file(
    bytes: &[u8],
    expected_key: [u8; 32],
) -> Result<DecodedDetailCache, RuntimeDetailError> {
    if bytes.len() < DETAIL_CACHE_FIXED_HEADER || bytes[..4] != DETAIL_CACHE_MAGIC {
        return Err(RuntimeDetailError::CacheCorrupt);
    }
    let mut cursor = 4;
    if read_cache_u32(bytes, &mut cursor)? != DETAIL_CACHE_SCHEMA {
        return Err(RuntimeDetailError::CacheCorrupt);
    }
    let key_end = cursor
        .checked_add(32)
        .ok_or(RuntimeDetailError::CacheCorrupt)?;
    if bytes.get(cursor..key_end) != Some(expected_key.as_slice()) {
        return Err(RuntimeDetailError::CacheCorrupt);
    }
    cursor = key_end;
    let page_count = read_cache_u32(bytes, &mut cursor)? as usize;
    if page_count == 0 {
        return Err(RuntimeDetailError::CacheCorrupt);
    }
    let mut records = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        let len = read_cache_u64(bytes, &mut cursor)?;
        let hash_end = cursor
            .checked_add(32)
            .ok_or(RuntimeDetailError::CacheCorrupt)?;
        let hash: [u8; 32] = bytes
            .get(cursor..hash_end)
            .ok_or(RuntimeDetailError::CacheCorrupt)?
            .try_into()
            .map_err(|_| RuntimeDetailError::CacheCorrupt)?;
        cursor = hash_end;
        records.push((len, hash));
    }
    let mut levels = Vec::with_capacity(page_count);
    let mut pages = Vec::with_capacity(page_count);
    let mut page_offsets = Vec::with_capacity(page_count);
    for (index, (len, expected_hash)) in records.into_iter().enumerate() {
        let len = usize::try_from(len).map_err(|_| RuntimeDetailError::AddressOverflow)?;
        let end = cursor
            .checked_add(len)
            .ok_or(RuntimeDetailError::AddressOverflow)?;
        let page = bytes
            .get(cursor..end)
            .ok_or(RuntimeDetailError::CacheCorrupt)?;
        if <[u8; 32]>::from(Sha256::digest(page)) != expected_hash {
            return Err(RuntimeDetailError::CacheCorrupt);
        }
        let level = decode_detail_page(page).map_err(|_| RuntimeDetailError::CacheCorrupt)?;
        if level.group_id as usize != index
            || level
                .parent_group_id
                .is_some_and(|parent| parent as usize >= index)
        {
            return Err(RuntimeDetailError::CacheCorrupt);
        }
        page_offsets.push(u64::try_from(cursor).map_err(|_| RuntimeDetailError::AddressOverflow)?);
        levels.push(level);
        pages.push(page.to_vec());
        cursor = end;
    }
    if cursor != bytes.len() {
        return Err(RuntimeDetailError::CacheCorrupt);
    }
    Ok(DecodedDetailCache {
        levels,
        pages,
        page_offsets,
    })
}

#[cfg(test)]
fn read_cache_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, RuntimeDetailError> {
    let end = cursor
        .checked_add(4)
        .ok_or(RuntimeDetailError::CacheCorrupt)?;
    let value = bytes
        .get(*cursor..end)
        .ok_or(RuntimeDetailError::CacheCorrupt)?;
    *cursor = end;
    Ok(u32::from_le_bytes(
        value
            .try_into()
            .map_err(|_| RuntimeDetailError::CacheCorrupt)?,
    ))
}

#[cfg(test)]
fn read_cache_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, RuntimeDetailError> {
    let end = cursor
        .checked_add(8)
        .ok_or(RuntimeDetailError::CacheCorrupt)?;
    let value = bytes
        .get(*cursor..end)
        .ok_or(RuntimeDetailError::CacheCorrupt)?;
    *cursor = end;
    Ok(u64::from_le_bytes(
        value
            .try_into()
            .map_err(|_| RuntimeDetailError::CacheCorrupt)?,
    ))
}

fn hex_hash(hash: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in hash {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

const DETAIL_PAGE_MAGIC: [u8; 4] = *b"MGTL";
const DETAIL_PAGE_SCHEMA: u32 = 3;
const DETAIL_PAGE_HEADER_WORDS: usize = 12;
const DETAIL_CORRESPONDENCE_WORDS: usize = 15;

/// Encode one independently decodable triangle-detail page. All values needed
/// for native triangle/cluster lowering and temporal surface remap travel with
/// the page; no source asset or authored LOD is consulted at runtime.
pub fn encode_detail_page(level: &DerivedDetailLevel) -> Result<Vec<u8>, DetailDerivationError> {
    if level.mesh.positions.len() != level.normals.len()
        || level.mesh.positions.len() != level.tangents.len()
        || (!level.weathering_masks.is_empty()
            && level.mesh.positions.len() != level.weathering_masks.len())
        || (!level.mesh.uvs.is_empty() && level.mesh.uvs.len() != level.mesh.positions.len())
        || level.mesh.indices.len() != level.mesh.material_ids.len()
        || level.mesh.indices.len() != level.mesh.source_triangle_indices.len()
        || !level.accumulated_world_error.is_finite()
    {
        return Err(DetailDerivationError::InvalidSource);
    }
    validate_decoded_level(level)?;
    let vertex_count = u32::try_from(level.mesh.positions.len())
        .map_err(|_| DetailDerivationError::InvalidSource)?;
    let triangle_count = u32::try_from(level.mesh.indices.len())
        .map_err(|_| DetailDerivationError::InvalidSource)?;
    let correspondence_count = u32::try_from(level.correspondence_to_parent.len())
        .map_err(|_| DetailDerivationError::InvalidSource)?;
    let root_correspondence_count = u32::try_from(level.correspondence_to_root.len())
        .map_err(|_| DetailDerivationError::InvalidSource)?;
    let has_uv = !level.mesh.uvs.is_empty();
    let has_weathering = !level.weathering_masks.is_empty();
    let word_count = DETAIL_PAGE_HEADER_WORDS
        + level.mesh.positions.len()
            * (3 + 3 + 4 + usize::from(has_uv) * 2 + usize::from(has_weathering) * 7)
        + level.mesh.indices.len() * (3 + 1 + 1)
        + (level.correspondence_to_parent.len() + level.correspondence_to_root.len())
            * DETAIL_CORRESPONDENCE_WORDS;
    let mut out = Vec::with_capacity(word_count * 4);
    out.extend_from_slice(&DETAIL_PAGE_MAGIC);
    for word in [
        DETAIL_PAGE_SCHEMA,
        level.group_id,
        level.parent_group_id.unwrap_or(u32::MAX),
        level.accumulated_world_error.to_bits(),
        vertex_count,
        triangle_count,
        correspondence_count,
        u32::from(has_uv),
        u32::from(has_weathering),
        root_correspondence_count,
        level.root_triangle_count,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    for position in &level.mesh.positions {
        push_f32s(&mut out, position);
    }
    for normal in &level.normals {
        push_f32s(&mut out, normal);
    }
    for tangent in &level.tangents {
        push_f32s(&mut out, tangent);
    }
    if has_uv {
        for uv in &level.mesh.uvs {
            push_f32s(&mut out, uv);
        }
    }
    if has_weathering {
        for masks in &level.weathering_masks {
            push_f32s(&mut out, masks);
        }
    }
    for triangle in &level.mesh.indices {
        for index in triangle {
            out.extend_from_slice(&index.to_le_bytes());
        }
    }
    for material in &level.mesh.material_ids {
        out.extend_from_slice(&material.to_le_bytes());
    }
    for source in &level.mesh.source_triangle_indices {
        out.extend_from_slice(&source.to_le_bytes());
    }
    for correspondence in &level.correspondence_to_parent {
        encode_correspondence(&mut out, correspondence);
    }
    for correspondence in &level.correspondence_to_root {
        encode_correspondence(&mut out, correspondence);
    }
    Ok(out)
}

fn encode_correspondence(out: &mut Vec<u8>, correspondence: &TriangleSurfaceCorrespondence) {
    for word in [
        correspondence.child_triangle,
        correspondence.parent_triangle,
        correspondence.material_id,
        correspondence.orientation_dot.to_bits(),
        correspondence.max_position_error.to_bits(),
        correspondence.max_uv_error.to_bits(),
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    for barycentric in correspondence.parent_barycentric {
        push_f32s(out, &barycentric);
    }
}

pub fn decode_detail_page(bytes: &[u8]) -> Result<DerivedDetailLevel, DetailDerivationError> {
    if bytes.len() < DETAIL_PAGE_HEADER_WORDS * 4 || bytes[..4] != DETAIL_PAGE_MAGIC {
        return Err(DetailDerivationError::InvalidSource);
    }
    let mut cursor = 4;
    let schema = read_u32(bytes, &mut cursor)?;
    if schema != DETAIL_PAGE_SCHEMA {
        return Err(DetailDerivationError::InvalidSource);
    }
    let group_id = read_u32(bytes, &mut cursor)?;
    let parent = read_u32(bytes, &mut cursor)?;
    let accumulated_world_error = f32::from_bits(read_u32(bytes, &mut cursor)?);
    let vertex_count = read_u32(bytes, &mut cursor)? as usize;
    let triangle_count = read_u32(bytes, &mut cursor)? as usize;
    let correspondence_count = read_u32(bytes, &mut cursor)? as usize;
    let has_uv = read_u32(bytes, &mut cursor)? != 0;
    let has_weathering = read_u32(bytes, &mut cursor)? != 0;
    let root_correspondence_count = read_u32(bytes, &mut cursor)? as usize;
    let root_triangle_count = read_u32(bytes, &mut cursor)?;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(if has_uv { vertex_count } else { 0 });
    let mut tangents = Vec::with_capacity(vertex_count);
    let mut weathering_masks = Vec::with_capacity(if has_weathering { vertex_count } else { 0 });
    for _ in 0..vertex_count {
        positions.push(read_f32_array::<3>(bytes, &mut cursor)?);
    }
    for _ in 0..vertex_count {
        normals.push(read_f32_array::<3>(bytes, &mut cursor)?);
    }
    for _ in 0..vertex_count {
        tangents.push(read_f32_array::<4>(bytes, &mut cursor)?);
    }
    if has_uv {
        for _ in 0..vertex_count {
            uvs.push(read_f32_array::<2>(bytes, &mut cursor)?);
        }
    }
    if has_weathering {
        for _ in 0..vertex_count {
            weathering_masks.push(read_f32_array::<7>(bytes, &mut cursor)?);
        }
    }
    let mut indices = Vec::with_capacity(triangle_count);
    for _ in 0..triangle_count {
        indices.push([
            read_u32(bytes, &mut cursor)?,
            read_u32(bytes, &mut cursor)?,
            read_u32(bytes, &mut cursor)?,
        ]);
    }
    let mut material_ids = Vec::with_capacity(triangle_count);
    let mut source_triangle_indices = Vec::with_capacity(triangle_count);
    for _ in 0..triangle_count {
        material_ids.push(read_u32(bytes, &mut cursor)?);
    }
    for _ in 0..triangle_count {
        source_triangle_indices.push(read_u32(bytes, &mut cursor)?);
    }
    let mut correspondence_to_parent = Vec::with_capacity(correspondence_count);
    for _ in 0..correspondence_count {
        correspondence_to_parent.push(decode_correspondence(bytes, &mut cursor)?);
    }
    let mut correspondence_to_root = Vec::with_capacity(root_correspondence_count);
    for _ in 0..root_correspondence_count {
        correspondence_to_root.push(decode_correspondence(bytes, &mut cursor)?);
    }
    if cursor != bytes.len()
        || !accumulated_world_error.is_finite()
        || indices
            .iter()
            .flatten()
            .any(|&index| index as usize >= vertex_count)
    {
        return Err(DetailDerivationError::InvalidSource);
    }
    let level = DerivedDetailLevel {
        group_id,
        parent_group_id: (parent != u32::MAX).then_some(parent),
        mesh: MeshOutput {
            positions,
            uvs,
            indices,
            material_ids,
            source_triangle_indices,
        },
        normals,
        tangents,
        weathering_masks,
        accumulated_world_error,
        root_triangle_count,
        correspondence_to_parent,
        correspondence_to_root,
    };
    validate_decoded_level(&level)?;
    Ok(level)
}

fn decode_correspondence(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<TriangleSurfaceCorrespondence, DetailDerivationError> {
    Ok(TriangleSurfaceCorrespondence {
        child_triangle: read_u32(bytes, cursor)?,
        parent_triangle: read_u32(bytes, cursor)?,
        material_id: read_u32(bytes, cursor)?,
        orientation_dot: f32::from_bits(read_u32(bytes, cursor)?),
        max_position_error: f32::from_bits(read_u32(bytes, cursor)?),
        max_uv_error: f32::from_bits(read_u32(bytes, cursor)?),
        parent_barycentric: [
            read_f32_array::<3>(bytes, cursor)?,
            read_f32_array::<3>(bytes, cursor)?,
            read_f32_array::<3>(bytes, cursor)?,
        ],
    })
}

fn validate_decoded_level(level: &DerivedDetailLevel) -> Result<(), DetailDerivationError> {
    let triangle_count = level.mesh.indices.len();
    let invalid_correspondence = |index: usize, entry: &TriangleSurfaceCorrespondence| {
        !entry.max_position_error.is_finite()
            || !entry.max_uv_error.is_finite()
            || !entry.orientation_dot.is_finite()
            || entry.max_position_error < 0.0
            || entry.max_uv_error < 0.0
            || entry.orientation_dot <= 0.0
            || entry.child_triangle as usize != index
            || level.mesh.material_ids[index] != entry.material_id
            || entry.parent_barycentric.iter().any(|barycentric| {
                barycentric
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                    || (barycentric.iter().sum::<f32>() - 1.0).abs() > 1.0e-4
            })
    };
    let correspondence_shape_valid = if level.parent_group_id.is_some() {
        level.correspondence_to_parent.len() == triangle_count
            && level.correspondence_to_root.len() == triangle_count
    } else {
        level.correspondence_to_parent.is_empty() && level.correspondence_to_root.is_empty()
    };
    if !correspondence_shape_valid
        || level.normals.len() != level.mesh.positions.len()
        || level.tangents.len() != level.mesh.positions.len()
        || (!level.weathering_masks.is_empty()
            && level.weathering_masks.len() != level.mesh.positions.len())
        || level.accumulated_world_error < 0.0
        || level.root_triangle_count == 0
        || (level.parent_group_id.is_none() && level.root_triangle_count as usize != triangle_count)
        || level
            .mesh
            .positions
            .iter()
            .flatten()
            .chain(level.normals.iter().flatten())
            .chain(level.tangents.iter().flatten())
            .chain(level.weathering_masks.iter().flatten())
            .chain(level.mesh.uvs.iter().flatten())
            .any(|value| !value.is_finite())
        || level
            .correspondence_to_parent
            .iter()
            .enumerate()
            .any(|(index, entry)| invalid_correspondence(index, entry))
        || level
            .correspondence_to_root
            .iter()
            .enumerate()
            .any(|(index, entry)| {
                invalid_correspondence(index, entry)
                    || entry.parent_triangle >= level.root_triangle_count
            })
    {
        return Err(DetailDerivationError::NonFinite);
    }
    Ok(())
}

fn push_f32s<const N: usize>(out: &mut Vec<u8>, values: &[f32; N]) {
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, DetailDerivationError> {
    let end = cursor
        .checked_add(4)
        .ok_or(DetailDerivationError::InvalidSource)?;
    let word = bytes
        .get(*cursor..end)
        .ok_or(DetailDerivationError::InvalidSource)?;
    *cursor = end;
    Ok(u32::from_le_bytes(word.try_into().unwrap()))
}

fn read_f32_array<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[f32; N], DetailDerivationError> {
    let mut out = [0.0; N];
    for value in &mut out {
        *value = f32::from_bits(read_u32(bytes, cursor)?);
    }
    Ok(out)
}

#[derive(Clone, Copy)]
pub struct DetailMeshInput<'a> {
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub uvs: &'a [[f32; 2]],
    pub indices: &'a [[u32; 3]],
    pub material_ids: &'a [u32],
    /// Optional seven-channel authored weathering masks, parallel to positions.
    pub weathering_masks: &'a [[f32; 7]],
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DetailDerivationError {
    #[error("detail source streams are not parallel or contain invalid indices")]
    InvalidSource,
    #[error("detail ratios must be strictly increasing, finite, in (0,1], and end at 1")]
    InvalidRatios,
    #[error("simplification removed every triangle from a detail level")]
    EmptyLevel,
    #[error("child triangle {child} has no orientation-compatible parent in material {material}")]
    MissingCorrespondence { child: u32, material: u32 },
    #[error("derived detail contains a non-finite value")]
    NonFinite,
}

pub fn derive_detail_hierarchy(
    input: DetailMeshInput<'_>,
    ratios: &[f32],
) -> Result<DerivedDetailHierarchy, DetailDerivationError> {
    DETAIL_DERIVATION_CALLS.fetch_add(1, Ordering::Relaxed);
    validate_input(input)?;
    validate_ratios(ratios)?;
    let split = split_material_seams(input)?;
    let mut levels = Vec::with_capacity(ratios.len());
    for (level_index, &ratio) in ratios.iter().enumerate() {
        let mesh = simplify_mesh(
            &MeshInput {
                positions: &split.positions,
                uvs: &split.uvs,
                indices: &split.indices,
                material_ids: &split.material_ids,
            },
            ratio,
        );
        if mesh.indices.is_empty() {
            return Err(DetailDerivationError::EmptyLevel);
        }
        let normals = derive_normals_from_source(&mesh, &split);
        let tangents = derive_tangents(&mesh, &normals);
        let weathering_masks = derive_weathering_from_source(&mesh, &split);
        levels.push(DerivedDetailLevel {
            group_id: level_index as u32,
            parent_group_id: level_index.checked_sub(1).map(|id| id as u32),
            mesh,
            normals,
            tangents,
            weathering_masks,
            accumulated_world_error: 0.0,
            root_triangle_count: 0,
            correspondence_to_parent: Vec::new(),
            correspondence_to_root: Vec::new(),
        });
    }
    // A global target ratio must never erase a small material/seam domain.
    // Promote only the exact child triangles that have no admissible parent,
    // working fine-to-coarse so every promoted parent is itself proven against
    // the next coarser level. This preserves complete source correspondence
    // without abandoning simplification for unrelated surfaces.
    for child_index in (1..levels.len()).rev() {
        loop {
            let missing =
                match derive_correspondence(&levels[child_index].mesh, &levels[child_index - 1]) {
                    Ok(_) => break,
                    Err(DetailDerivationError::MissingCorrespondence { child, .. }) => child,
                    Err(error) => return Err(error),
                };
            let (parents, children) = levels.split_at_mut(child_index);
            promote_child_triangle(
                &children[0],
                missing as usize,
                &mut parents[child_index - 1],
            )?;
        }
    }
    let root_triangle_count = u32::try_from(levels[0].mesh.indices.len())
        .map_err(|_| DetailDerivationError::InvalidSource)?;
    for level in &mut levels {
        level.root_triangle_count = root_triangle_count;
    }
    for child_index in 1..levels.len() {
        let (parents, children) = levels.split_at_mut(child_index);
        children[0].correspondence_to_parent =
            derive_correspondence(&children[0].mesh, &parents[child_index - 1])?;
        children[0].correspondence_to_root = if child_index == 1 {
            children[0].correspondence_to_parent.clone()
        } else {
            compose_root_correspondence(
                &children[0].correspondence_to_parent,
                &parents[child_index - 1].correspondence_to_root,
            )?
        };
    }
    // Error is the conservative distance from this representation to the
    // authoritative finest surface. It therefore accumulates from fine to
    // coarse, while the authoritative leaf remains exactly zero.
    for child_index in (1..levels.len()).rev() {
        let local_error = levels[child_index]
            .correspondence_to_parent
            .iter()
            .map(|entry| entry.max_position_error)
            .fold(0.0_f32, f32::max);
        levels[child_index - 1].accumulated_world_error =
            levels[child_index].accumulated_world_error + local_error;
    }
    Ok(DerivedDetailHierarchy { levels })
}

fn promote_child_triangle(
    child: &DerivedDetailLevel,
    child_triangle: usize,
    parent: &mut DerivedDetailLevel,
) -> Result<(), DetailDerivationError> {
    let triangle = *child
        .mesh
        .indices
        .get(child_triangle)
        .ok_or(DetailDerivationError::InvalidSource)?;
    let first_vertex = u32::try_from(parent.mesh.positions.len())
        .map_err(|_| DetailDerivationError::InvalidSource)?;
    for vertex in triangle {
        let vertex = vertex as usize;
        parent.mesh.positions.push(
            *child
                .mesh
                .positions
                .get(vertex)
                .ok_or(DetailDerivationError::InvalidSource)?,
        );
        parent.normals.push(
            *child
                .normals
                .get(vertex)
                .ok_or(DetailDerivationError::InvalidSource)?,
        );
        parent.tangents.push(
            *child
                .tangents
                .get(vertex)
                .ok_or(DetailDerivationError::InvalidSource)?,
        );
        if !child.mesh.uvs.is_empty() {
            parent.mesh.uvs.push(
                *child
                    .mesh
                    .uvs
                    .get(vertex)
                    .ok_or(DetailDerivationError::InvalidSource)?,
            );
        }
        if !child.weathering_masks.is_empty() {
            parent.weathering_masks.push(
                *child
                    .weathering_masks
                    .get(vertex)
                    .ok_or(DetailDerivationError::InvalidSource)?,
            );
        }
    }
    parent
        .mesh
        .indices
        .push([first_vertex, first_vertex + 1, first_vertex + 2]);
    parent.mesh.material_ids.push(
        *child
            .mesh
            .material_ids
            .get(child_triangle)
            .ok_or(DetailDerivationError::InvalidSource)?,
    );
    parent.mesh.source_triangle_indices.push(
        *child
            .mesh
            .source_triangle_indices
            .get(child_triangle)
            .ok_or(DetailDerivationError::InvalidSource)?,
    );
    Ok(())
}

struct SplitMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<[u32; 3]>,
    material_ids: Vec<u32>,
    weathering_masks: Vec<[f32; 7]>,
}

fn validate_input(input: DetailMeshInput<'_>) -> Result<(), DetailDerivationError> {
    if input.positions.is_empty()
        || input.indices.is_empty()
        || input.normals.len() != input.positions.len()
        || (!input.uvs.is_empty() && input.uvs.len() != input.positions.len())
        || input.material_ids.len() != input.indices.len()
        || (!input.weathering_masks.is_empty()
            && input.weathering_masks.len() != input.positions.len())
        || input
            .positions
            .iter()
            .flatten()
            .chain(input.normals.iter().flatten())
            .chain(input.uvs.iter().flatten())
            .chain(input.weathering_masks.iter().flatten())
            .any(|value| !value.is_finite())
        || input
            .indices
            .iter()
            .flatten()
            .any(|&vertex| vertex as usize >= input.positions.len())
    {
        return Err(DetailDerivationError::InvalidSource);
    }
    Ok(())
}

fn validate_ratios(ratios: &[f32]) -> Result<(), DetailDerivationError> {
    if ratios.len() < 2
        || ratios.last().copied() != Some(1.0)
        || ratios
            .iter()
            .any(|ratio| !ratio.is_finite() || *ratio <= 0.0 || *ratio > 1.0)
        || ratios.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(DetailDerivationError::InvalidRatios);
    }
    Ok(())
}

fn split_material_seams(input: DetailMeshInput<'_>) -> Result<SplitMesh, DetailDerivationError> {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut weathering_masks = Vec::new();
    let mut remap = BTreeMap::<(u32, u32), u32>::new();
    let mut indices = Vec::with_capacity(input.indices.len());
    for (triangle_index, triangle) in input.indices.iter().enumerate() {
        let material = input.material_ids[triangle_index];
        let mut out = [0_u32; 3];
        for corner in 0..3 {
            let source = triangle[corner];
            out[corner] = *remap.entry((source, material)).or_insert_with(|| {
                let id = positions.len() as u32;
                positions.push(input.positions[source as usize]);
                normals.push(input.normals[source as usize]);
                if !input.uvs.is_empty() {
                    uvs.push(input.uvs[source as usize]);
                }
                if !input.weathering_masks.is_empty() {
                    weathering_masks.push(input.weathering_masks[source as usize]);
                }
                id
            });
        }
        if out[0] != out[1] && out[1] != out[2] && out[0] != out[2] {
            indices.push(out);
        }
    }
    if indices.len() != input.indices.len() {
        return Err(DetailDerivationError::InvalidSource);
    }
    Ok(SplitMesh {
        positions,
        normals,
        uvs,
        indices,
        material_ids: input.material_ids.to_vec(),
        weathering_masks,
    })
}

fn derive_normals_from_source(mesh: &MeshOutput, source: &SplitMesh) -> Vec<[f32; 3]> {
    let mut first_triangle = vec![None; mesh.positions.len()];
    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        for &vertex in triangle {
            first_triangle[vertex as usize].get_or_insert(triangle_index);
        }
    }
    mesh.positions
        .iter()
        .enumerate()
        .map(|(vertex, &position)| {
            let triangle_index = first_triangle[vertex].expect("used output vertex");
            let source_triangle = mesh.source_triangle_indices[triangle_index] as usize;
            let source_indices = source.indices[source_triangle];
            let source_positions = source_indices.map(|index| source.positions[index as usize]);
            let (_, barycentric) = closest_point_barycentric(position, source_positions);
            let source_normals = source_indices.map(|index| source.normals[index as usize]);
            normalize(add(
                add(
                    mul(source_normals[0], barycentric[0]),
                    mul(source_normals[1], barycentric[1]),
                ),
                mul(source_normals[2], barycentric[2]),
            ))
        })
        .collect()
}

fn output_vertex_source_samples(
    mesh: &MeshOutput,
    source: &SplitMesh,
) -> Vec<([u32; 3], [f32; 3])> {
    let mut first_triangle = vec![None; mesh.positions.len()];
    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        for &vertex in triangle {
            first_triangle[vertex as usize].get_or_insert(triangle_index);
        }
    }
    mesh.positions
        .iter()
        .enumerate()
        .map(|(vertex, &position)| {
            let triangle_index = first_triangle[vertex].expect("used output vertex");
            let source_triangle = mesh.source_triangle_indices[triangle_index] as usize;
            let source_indices = source.indices[source_triangle];
            let source_positions = source_indices.map(|index| source.positions[index as usize]);
            let (_, barycentric) = closest_point_barycentric(position, source_positions);
            (source_indices, barycentric)
        })
        .collect()
}

fn derive_weathering_from_source(mesh: &MeshOutput, source: &SplitMesh) -> Vec<[f32; 7]> {
    if source.weathering_masks.is_empty() {
        return Vec::new();
    }
    output_vertex_source_samples(mesh, source)
        .into_iter()
        .map(|(indices, barycentric)| {
            let mut out = [0.0; 7];
            for channel in 0..7 {
                out[channel] = source.weathering_masks[indices[0] as usize][channel]
                    * barycentric[0]
                    + source.weathering_masks[indices[1] as usize][channel] * barycentric[1]
                    + source.weathering_masks[indices[2] as usize][channel] * barycentric[2];
            }
            out
        })
        .collect()
}

fn derive_tangents(mesh: &MeshOutput, normals: &[[f32; 3]]) -> Vec<[f32; 4]> {
    let mut accumulated = vec![[0.0_f32; 3]; mesh.positions.len()];
    if !mesh.uvs.is_empty() {
        for triangle in &mesh.indices {
            let [i0, i1, i2] = triangle.map(|index| index as usize);
            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];
            let t0 = mesh.uvs[i0];
            let t1 = mesh.uvs[i1];
            let t2 = mesh.uvs[i2];
            let e1 = sub(p1, p0);
            let e2 = sub(p2, p0);
            let d1 = [t1[0] - t0[0], t1[1] - t0[1]];
            let d2 = [t2[0] - t0[0], t2[1] - t0[1]];
            let det = d1[0] * d2[1] - d2[0] * d1[1];
            if det.abs() < 1.0e-12 {
                continue;
            }
            let reciprocal = 1.0 / det;
            let tangent = [
                (e1[0] * d2[1] - e2[0] * d1[1]) * reciprocal,
                (e1[1] * d2[1] - e2[1] * d1[1]) * reciprocal,
                (e1[2] * d2[1] - e2[2] * d1[1]) * reciprocal,
            ];
            for index in [i0, i1, i2] {
                accumulated[index] = add(accumulated[index], tangent);
            }
        }
    }
    normals
        .iter()
        .zip(accumulated)
        .map(|(&normal, tangent)| {
            let projected = sub(tangent, mul(normal, dot(normal, tangent)));
            let tangent = if length(projected) > 1.0e-6 {
                normalize(projected)
            } else {
                let up = if normal[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                normalize(cross(up, normal))
            };
            [tangent[0], tangent[1], tangent[2], 1.0]
        })
        .collect()
}

fn derive_correspondence(
    child: &MeshOutput,
    parent: &DerivedDetailLevel,
) -> Result<Vec<TriangleSurfaceCorrespondence>, DetailDerivationError> {
    let mut by_material_indices = BTreeMap::<u32, Vec<usize>>::new();
    for (index, &material) in parent.mesh.material_ids.iter().enumerate() {
        by_material_indices.entry(material).or_default().push(index);
    }
    let by_material = by_material_indices
        .into_iter()
        .map(|(material, indices)| (material, MaterialTriangleBvh::new(&parent.mesh, indices)))
        .collect::<BTreeMap<_, _>>();
    let mut out = Vec::with_capacity(child.indices.len());
    for (child_index, triangle) in child.indices.iter().enumerate() {
        let material = child.material_ids[child_index];
        let child_pos = triangle.map(|vertex| child.positions[vertex as usize]);
        let child_normal = normalize(cross(
            sub(child_pos[1], child_pos[0]),
            sub(child_pos[2], child_pos[0]),
        ));
        let mut best: Option<(f32, usize, [[f32; 3]; 3], f32, f32)> = None;
        if let Some(bvh) = by_material.get(&material) {
            bvh.find_best(
                &parent.mesh,
                child,
                *triangle,
                child_pos,
                child_normal,
                &mut best,
            );
        }
        let Some((error, parent_triangle, barycentric, uv_error, orientation_dot)) = best else {
            return Err(DetailDerivationError::MissingCorrespondence {
                child: child_index as u32,
                material,
            });
        };
        if !error.is_finite() || !uv_error.is_finite() || !orientation_dot.is_finite() {
            return Err(DetailDerivationError::NonFinite);
        }
        out.push(TriangleSurfaceCorrespondence {
            child_triangle: child_index as u32,
            parent_triangle: parent_triangle as u32,
            parent_barycentric: barycentric,
            max_position_error: error,
            max_uv_error: uv_error,
            orientation_dot,
            material_id: material,
        });
    }
    Ok(out)
}

fn compose_root_correspondence(
    child_to_parent: &[TriangleSurfaceCorrespondence],
    parent_to_root: &[TriangleSurfaceCorrespondence],
) -> Result<Vec<TriangleSurfaceCorrespondence>, DetailDerivationError> {
    let mut out = Vec::with_capacity(child_to_parent.len());
    for child in child_to_parent {
        let parent = parent_to_root
            .get(child.parent_triangle as usize)
            .filter(|entry| entry.child_triangle == child.parent_triangle)
            .ok_or(DetailDerivationError::InvalidSource)?;
        if parent.material_id != child.material_id {
            return Err(DetailDerivationError::InvalidSource);
        }
        let mut root_barycentric = [[0.0_f32; 3]; 3];
        for child_vertex in 0..3 {
            for parent_vertex in 0..3 {
                let weight = child.parent_barycentric[child_vertex][parent_vertex];
                for root_corner in 0..3 {
                    root_barycentric[child_vertex][root_corner] +=
                        weight * parent.parent_barycentric[parent_vertex][root_corner];
                }
            }
            let sum = root_barycentric[child_vertex].iter().sum::<f32>();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(DetailDerivationError::NonFinite);
            }
            for weight in &mut root_barycentric[child_vertex] {
                *weight = (*weight / sum).max(0.0);
            }
            let renormalize = root_barycentric[child_vertex].iter().sum::<f32>();
            for weight in &mut root_barycentric[child_vertex] {
                *weight /= renormalize;
            }
        }
        let max_position_error = child.max_position_error + parent.max_position_error;
        let max_uv_error = child.max_uv_error + parent.max_uv_error;
        let orientation_dot = child.orientation_dot.min(parent.orientation_dot);
        if !max_position_error.is_finite()
            || !max_uv_error.is_finite()
            || !orientation_dot.is_finite()
        {
            return Err(DetailDerivationError::NonFinite);
        }
        out.push(TriangleSurfaceCorrespondence {
            child_triangle: child.child_triangle,
            parent_triangle: parent.parent_triangle,
            parent_barycentric: root_barycentric,
            max_position_error,
            max_uv_error,
            orientation_dot,
            material_id: child.material_id,
        });
    }
    Ok(out)
}

const BVH_NONE: u32 = u32::MAX;
const BVH_LEAF_TRIANGLES: usize = 8;

#[derive(Clone, Copy)]
struct TriangleBvhNode {
    min: [f32; 3],
    max: [f32; 3],
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

struct MaterialTriangleBvh {
    nodes: Vec<TriangleBvhNode>,
    order: Vec<usize>,
}

impl MaterialTriangleBvh {
    fn new(mesh: &MeshOutput, mut order: Vec<usize>) -> Self {
        let mut nodes = Vec::with_capacity(order.len().saturating_mul(2));
        if !order.is_empty() {
            let len = order.len();
            build_triangle_bvh_node(mesh, &mut order, &mut nodes, 0, len);
        }
        Self { nodes, order }
    }

    fn find_best(
        &self,
        parent: &MeshOutput,
        child: &MeshOutput,
        child_triangle: [u32; 3],
        child_pos: [[f32; 3]; 3],
        child_normal: [f32; 3],
        best: &mut Option<(f32, usize, [[f32; 3]; 3], f32, f32)>,
    ) {
        if !self.nodes.is_empty() {
            self.search_node(
                0,
                parent,
                child,
                child_triangle,
                child_pos,
                child_normal,
                best,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search_node(
        &self,
        node_index: usize,
        parent: &MeshOutput,
        child: &MeshOutput,
        child_triangle: [u32; 3],
        child_pos: [[f32; 3]; 3],
        child_normal: [f32; 3],
        best: &mut Option<(f32, usize, [[f32; 3]; 3], f32, f32)>,
    ) {
        let node = self.nodes[node_index];
        let lower_bound = child_pos
            .iter()
            .map(|&point| distance_to_aabb(point, node.min, node.max))
            .fold(0.0_f32, f32::max);
        if best.as_ref().is_some_and(|current| lower_bound > current.0) {
            return;
        }
        if node.left == BVH_NONE {
            for &parent_index in
                &self.order[node.start as usize..(node.start + node.count) as usize]
            {
                evaluate_parent_candidate(
                    parent,
                    child,
                    child_triangle,
                    child_pos,
                    child_normal,
                    parent_index,
                    best,
                );
            }
            return;
        }
        let left = node.left as usize;
        let right = node.right as usize;
        let left_bound = node_child_lower_bound(self.nodes[left], child_pos);
        let right_bound = node_child_lower_bound(self.nodes[right], child_pos);
        let (first, second) =
            if left_bound < right_bound || (left_bound == right_bound && left < right) {
                (left, right)
            } else {
                (right, left)
            };
        self.search_node(
            first,
            parent,
            child,
            child_triangle,
            child_pos,
            child_normal,
            best,
        );
        self.search_node(
            second,
            parent,
            child,
            child_triangle,
            child_pos,
            child_normal,
            best,
        );
    }
}

fn build_triangle_bvh_node(
    mesh: &MeshOutput,
    order: &mut Vec<usize>,
    nodes: &mut Vec<TriangleBvhNode>,
    start: usize,
    end: usize,
) -> u32 {
    let (min, max, centroid_min, centroid_max) = triangle_range_bounds(mesh, &order[start..end]);
    let id = nodes.len() as u32;
    nodes.push(TriangleBvhNode {
        min,
        max,
        left: BVH_NONE,
        right: BVH_NONE,
        start: start as u32,
        count: (end - start) as u32,
    });
    if end - start <= BVH_LEAF_TRIANGLES {
        return id;
    }
    let extent = [
        centroid_max[0] - centroid_min[0],
        centroid_max[1] - centroid_min[1],
        centroid_max[2] - centroid_min[2],
    ];
    let axis = if extent[1] > extent[0] && extent[1] >= extent[2] {
        1
    } else if extent[2] > extent[0] && extent[2] > extent[1] {
        2
    } else {
        0
    };
    order[start..end].sort_by(|&a, &b| {
        triangle_centroid(mesh, a)[axis]
            .total_cmp(&triangle_centroid(mesh, b)[axis])
            .then(a.cmp(&b))
    });
    let middle = start + (end - start) / 2;
    let left = build_triangle_bvh_node(mesh, order, nodes, start, middle);
    let right = build_triangle_bvh_node(mesh, order, nodes, middle, end);
    nodes[id as usize].left = left;
    nodes[id as usize].right = right;
    id
}

fn triangle_range_bounds(
    mesh: &MeshOutput,
    triangles: &[usize],
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut centroid_min = [f32::INFINITY; 3];
    let mut centroid_max = [f32::NEG_INFINITY; 3];
    for &triangle_index in triangles {
        let triangle = mesh.indices[triangle_index];
        let centroid = triangle_centroid(mesh, triangle_index);
        for axis in 0..3 {
            centroid_min[axis] = centroid_min[axis].min(centroid[axis]);
            centroid_max[axis] = centroid_max[axis].max(centroid[axis]);
        }
        for vertex in triangle {
            let position = mesh.positions[vertex as usize];
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
        }
    }
    (min, max, centroid_min, centroid_max)
}

fn triangle_centroid(mesh: &MeshOutput, index: usize) -> [f32; 3] {
    let triangle = mesh.indices[index];
    let a = mesh.positions[triangle[0] as usize];
    let b = mesh.positions[triangle[1] as usize];
    let c = mesh.positions[triangle[2] as usize];
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

fn node_child_lower_bound(node: TriangleBvhNode, child: [[f32; 3]; 3]) -> f32 {
    child
        .iter()
        .map(|&point| distance_to_aabb(point, node.min, node.max))
        .fold(0.0_f32, f32::max)
}

fn distance_to_aabb(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut squared = 0.0;
    for axis in 0..3 {
        let delta = if point[axis] < min[axis] {
            min[axis] - point[axis]
        } else if point[axis] > max[axis] {
            point[axis] - max[axis]
        } else {
            0.0
        };
        squared += delta * delta;
    }
    squared.sqrt()
}

fn evaluate_parent_candidate(
    parent: &MeshOutput,
    child: &MeshOutput,
    child_triangle: [u32; 3],
    child_pos: [[f32; 3]; 3],
    child_normal: [f32; 3],
    parent_index: usize,
    best: &mut Option<(f32, usize, [[f32; 3]; 3], f32, f32)>,
) {
    let parent_tri = parent.indices[parent_index];
    let parent_pos = parent_tri.map(|vertex| parent.positions[vertex as usize]);
    let parent_normal = normalize(cross(
        sub(parent_pos[1], parent_pos[0]),
        sub(parent_pos[2], parent_pos[0]),
    ));
    let orientation = dot(child_normal, parent_normal);
    if orientation <= 0.0 {
        return;
    }
    let mut bary = [[0.0; 3]; 3];
    let mut max_position_error = 0.0_f32;
    let mut max_uv_error = 0.0_f32;
    for corner in 0..3 {
        let (_, raw_weights) = closest_point_barycentric(child_pos[corner], parent_pos);
        let Some(weights) = normalized_nonnegative_barycentric(raw_weights) else {
            return;
        };
        let mapped = add(
            add(
                mul(parent_pos[0], weights[0]),
                mul(parent_pos[1], weights[1]),
            ),
            mul(parent_pos[2], weights[2]),
        );
        bary[corner] = weights;
        max_position_error = max_position_error.max(length(sub(child_pos[corner], mapped)));
        if !child.uvs.is_empty() && !parent.uvs.is_empty() {
            let puv = parent_tri.map(|vertex| parent.uvs[vertex as usize]);
            let mapped_uv = [
                puv[0][0] * weights[0] + puv[1][0] * weights[1] + puv[2][0] * weights[2],
                puv[0][1] * weights[0] + puv[1][1] * weights[1] + puv[2][1] * weights[2],
            ];
            let cuv = child.uvs[child_triangle[corner] as usize];
            max_uv_error = max_uv_error
                .max(((cuv[0] - mapped_uv[0]).powi(2) + (cuv[1] - mapped_uv[1]).powi(2)).sqrt());
        }
    }
    if best.as_ref().is_none_or(|current| {
        max_position_error < current.0
            || (max_position_error == current.0 && parent_index < current.1)
    }) {
        *best = Some((
            max_position_error,
            parent_index,
            bary,
            max_uv_error,
            orientation,
        ));
    }
}

/// Floating-point closest-point region tests can return a tiny negative
/// barycentric on large, thin architectural triangles. The mathematical result
/// is still on the triangle, so canonicalise it before it becomes persistent
/// surface correspondence. Truly non-finite/empty weights still reject the
/// candidate instead of being hidden by a fallback.
fn normalized_nonnegative_barycentric(weights: [f32; 3]) -> Option<[f32; 3]> {
    if weights.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut normalized = weights.map(|value| value.max(0.0));
    let sum = normalized.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 1.0e-20 {
        return None;
    }
    for value in &mut normalized {
        *value /= sum;
    }
    Some(normalized)
}

fn closest_point_barycentric(p: [f32; 3], t: [[f32; 3]; 3]) -> ([f32; 3], [f32; 3]) {
    // Ericson's closest-point-on-triangle regions, with barycentrics returned.
    let ab = sub(t[1], t[0]);
    let ac = sub(t[2], t[0]);
    let ap = sub(p, t[0]);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (t[0], [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, t[1]);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (t[1], [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(t[0], mul(ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = sub(p, t[2]);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (t[2], [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(t[0], mul(ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(t[1], mul(sub(t[2], t[1]), w)), [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let weights = [1.0 - v - w, v, w];
    (
        add(
            add(mul(t[0], weights[0]), mul(t[1], weights[1])),
            mul(t[2], weights[2]),
        ),
        weights,
    )
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    if len > 1.0e-20 {
        mul(v, 1.0 / len)
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> (
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<[f32; 2]>,
        Vec<[u32; 3]>,
        Vec<u32>,
    ) {
        let p = vec![
            [-1., -1., -1.],
            [1., -1., -1.],
            [1., 1., -1.],
            [-1., 1., -1.],
            [-1., -1., 1.],
            [1., -1., 1.],
            [1., 1., 1.],
            [-1., 1., 1.],
        ];
        let n = p.iter().map(|&v| normalize(v)).collect();
        let uv = p
            .iter()
            .map(|v| [(v[0] + 1.) * 0.5, (v[1] + 1.) * 0.5])
            .collect();
        let i = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let m = vec![0; i.len()];
        (p, n, uv, i, m)
    }

    #[test]
    fn hierarchy_is_deterministic_and_ends_at_authoritative_detail() {
        let (p, n, uv, i, m) = cube();
        let input = DetailMeshInput {
            positions: &p,
            normals: &n,
            uvs: &uv,
            indices: &i,
            material_ids: &m,
            weathering_masks: &[],
        };
        let a = derive_detail_hierarchy(input, &[0.5, 1.0]).unwrap();
        let b = derive_detail_hierarchy(input, &[0.5, 1.0]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.levels.len(), 2);
        assert_eq!(a.levels[0].parent_group_id, None);
        assert_eq!(a.levels[1].parent_group_id, Some(0));
        assert_eq!(a.levels[1].mesh.indices.len(), i.len());
        assert_eq!(a.levels[1].mesh.material_ids, m);
        assert!(a.levels[0].accumulated_world_error >= 0.0);
        assert_eq!(a.levels[1].accumulated_world_error, 0.0);
        for (&position, &normal) in a.levels[1].mesh.positions.iter().zip(&a.levels[1].normals) {
            assert!(dot(normalize(position), normal) > 0.999_999);
        }
    }

    #[test]
    fn every_fine_triangle_has_a_bounded_affine_parent_map() {
        let (p, n, uv, i, m) = cube();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.5, 1.0],
        )
        .unwrap();
        let maps = &hierarchy.levels[1].correspondence_to_parent;
        assert_eq!(maps.len(), hierarchy.levels[1].mesh.indices.len());
        for (index, map) in maps.iter().enumerate() {
            assert_eq!(map.child_triangle, index as u32);
            assert_eq!(map.material_id, 0);
            assert!(map.orientation_dot > 0.0);
            assert!(map.max_position_error.is_finite());
            assert!(map.max_uv_error.is_finite());
            for barycentric in map.parent_barycentric {
                assert!((barycentric.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
                assert!(barycentric.iter().all(|weight| *weight >= 0.0));
            }
        }
    }

    #[test]
    fn every_optional_level_has_an_independent_direct_root_map() {
        let (p, n, uv, i, m) = cube();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.25, 0.5, 0.75, 1.0],
        )
        .unwrap();
        let root_count = hierarchy.levels[0].mesh.indices.len() as u32;
        for (level_index, level) in hierarchy.levels.iter().enumerate() {
            assert_eq!(level.root_triangle_count, root_count);
            if level_index == 0 {
                assert!(level.correspondence_to_root.is_empty());
                continue;
            }
            assert_eq!(level.correspondence_to_root.len(), level.mesh.indices.len());
            for (triangle, map) in level.correspondence_to_root.iter().enumerate() {
                assert_eq!(map.child_triangle, triangle as u32);
                assert!(map.parent_triangle < root_count);
                assert_eq!(
                    map.material_id,
                    hierarchy.levels[0].mesh.material_ids[map.parent_triangle as usize]
                );
                for barycentric in map.parent_barycentric {
                    assert!((barycentric.iter().sum::<f32>() - 1.0).abs() <= 1.0e-4);
                }
            }
            assert_eq!(
                decode_detail_page(&encode_detail_page(level).unwrap()).unwrap(),
                *level
            );
        }
    }

    #[test]
    fn material_domains_never_cross_correspondence() {
        let (p, n, uv, i, mut m) = cube();
        for material in m.iter_mut().skip(6) {
            *material = 1;
        }
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.75, 1.0],
        )
        .unwrap();
        for map in &hierarchy.levels[1].correspondence_to_parent {
            let parent_material =
                hierarchy.levels[0].mesh.material_ids[map.parent_triangle as usize];
            assert_eq!(map.material_id, parent_material);
        }
    }

    #[test]
    fn detail_pages_are_independently_decodable_and_bit_exact() {
        let (p, n, uv, i, m) = cube();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.5, 1.0],
        )
        .unwrap();
        for level in &hierarchy.levels {
            let encoded = encode_detail_page(level).unwrap();
            assert_eq!(decode_detail_page(&encoded).unwrap(), *level);

            let mut truncated = encoded.clone();
            truncated.pop();
            assert_eq!(
                decode_detail_page(&truncated),
                Err(DetailDerivationError::InvalidSource)
            );

            let mut trailing = encoded.clone();
            trailing.push(0);
            assert_eq!(
                decode_detail_page(&trailing),
                Err(DetailDerivationError::InvalidSource)
            );
        }
    }

    #[test]
    fn detail_pages_preserve_tangent_frames_and_authored_weathering() {
        let (p, n, uv, i, m) = cube();
        let weathering = (0..p.len())
            .map(|vertex| {
                std::array::from_fn(|channel| (vertex * 7 + channel) as f32 / (p.len() * 7) as f32)
            })
            .collect::<Vec<[f32; 7]>>();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &weathering,
            },
            &[0.5, 1.0],
        )
        .unwrap();
        for level in &hierarchy.levels {
            assert_eq!(level.tangents.len(), level.mesh.positions.len());
            assert_eq!(level.weathering_masks.len(), level.mesh.positions.len());
            assert!(
                level
                    .tangents
                    .iter()
                    .flatten()
                    .chain(level.weathering_masks.iter().flatten())
                    .all(|value| value.is_finite())
            );
            let encoded = encode_detail_page(level).unwrap();
            assert_eq!(decode_detail_page(&encoded).unwrap(), *level);
        }
    }

    #[test]
    fn detail_page_rejects_non_finite_and_invalid_correspondence() {
        let (p, n, uv, i, m) = cube();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.5, 1.0],
        )
        .unwrap();
        let mut non_finite = encode_detail_page(&hierarchy.levels[0]).unwrap();
        non_finite[16..20].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert_eq!(
            decode_detail_page(&non_finite),
            Err(DetailDerivationError::InvalidSource)
        );

        let mut invalid = hierarchy.levels[1].clone();
        invalid.correspondence_to_parent[0].orientation_dot = 0.0;
        assert_eq!(
            encode_detail_page(&invalid),
            Err(DetailDerivationError::NonFinite)
        );
    }

    #[test]
    fn correspondence_bvh_matches_exhaustive_search_exactly() {
        let (p, n, uv, i, m) = cube();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.5, 1.0],
        )
        .unwrap();
        let parent = &hierarchy.levels[0].mesh;
        let child = &hierarchy.levels[1].mesh;
        for (child_index, &child_triangle) in child.indices.iter().enumerate() {
            let child_pos = child_triangle.map(|vertex| child.positions[vertex as usize]);
            let child_normal = normalize(cross(
                sub(child_pos[1], child_pos[0]),
                sub(child_pos[2], child_pos[0]),
            ));
            let material = child.material_ids[child_index];
            let mut exhaustive = None;
            for (parent_index, &parent_material) in parent.material_ids.iter().enumerate() {
                if parent_material == material {
                    evaluate_parent_candidate(
                        parent,
                        child,
                        child_triangle,
                        child_pos,
                        child_normal,
                        parent_index,
                        &mut exhaustive,
                    );
                }
            }
            let expected = exhaustive.unwrap();
            let actual = hierarchy.levels[1].correspondence_to_parent[child_index];
            assert_eq!(actual.parent_triangle, expected.1 as u32);
            assert_eq!(actual.max_position_error.to_bits(), expected.0.to_bits());
            assert_eq!(actual.parent_barycentric, expected.2);
            assert_eq!(actual.max_uv_error.to_bits(), expected.3.to_bits());
            assert_eq!(actual.orientation_dot.to_bits(), expected.4.to_bits());
        }
    }

    #[test]
    fn runtime_cache_hit_does_not_repeat_detail_derivation() {
        let (mut p, n, uv, i, m) = cube();
        // Keep this fixture's cache identity isolated from other tests and
        // product assets while remaining deterministic across processes.
        p[0][0] = -1.031_25;
        let input = DetailMeshInput {
            positions: &p,
            normals: &n,
            uvs: &uv,
            indices: &i,
            material_ids: &m,
            weathering_masks: &[],
        };
        let sources = (0..i.len() as u32).collect::<Vec<_>>();
        let ratios = [0.5, 1.0];
        let key = runtime_detail_source_key(&p, &i, input, &sources, &ratios);
        let path = runtime_detail_cache_root().join(format!("{}.mgc", hex_hash(key)));
        let _ = std::fs::remove_file(&path);

        let first = prepare_runtime_detail(&p, &i, input, &sources, &ratios).unwrap();
        let second = prepare_runtime_detail(&p, &i, input, &sources, &ratios).unwrap();

        assert!(!first.cache_hit);
        assert!(second.cache_hit);
        assert_eq!(first.cache_path, second.cache_path);
        assert_eq!(first.page_count, second.page_count);
        assert_eq!(
            first.clusters.runtime_pages(),
            second.clusters.runtime_pages()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn runtime_pages_follow_bounded_source_clusters_not_whole_buildings() {
        let width = 20_u32;
        let height = 15_u32;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for z in 0..=height {
            for x in 0..=width {
                positions.push([x as f32, 0.0, z as f32]);
                normals.push([0.0, 1.0, 0.0]);
                uvs.push([x as f32 / width as f32, z as f32 / height as f32]);
            }
        }
        let mut indices = Vec::new();
        for z in 0..height {
            for x in 0..width {
                let a = z * (width + 1) + x;
                let b = a + 1;
                let c = a + width + 1;
                let d = c + 1;
                indices.extend_from_slice(&[[a, c, b], [b, c, d]]);
            }
        }
        assert_eq!(indices.len(), 600);
        let materials = vec![0; indices.len()];
        let sources = (0..indices.len() as u32).collect::<Vec<_>>();
        let input = DetailMeshInput {
            positions: &positions,
            normals: &normals,
            uvs: &uvs,
            indices: &indices,
            material_ids: &materials,
            weathering_masks: &[],
        };
        let prepared =
            prepare_runtime_detail(&positions, &indices, input, &sources, &[0.5, 1.0]).unwrap();
        let pages = prepared.clusters.runtime_pages();
        let roots = pages
            .iter()
            .filter(|page| page.parent_page_id.is_none())
            .count();
        assert_eq!(roots, prepared.clusters.clusters().len());
        assert_eq!(pages.len(), roots * 2);
        assert!(pages.iter().all(|page| page.flags & 1 == 0));
        assert!(
            pages.iter().all(|page| page.primitive_count
                <= vox_data::geometry_clusters::MAX_TRIANGLES_PER_SOURCE_CLUSTER)
        );
        assert!(
            pages
                .iter()
                .all(|page| page.decoded_byte_count <= 256 * 3 * 24 * 4 + 256 * 20 * 4)
        );
        if let Some(path) = prepared.cache_path {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn runtime_cache_detects_corruption_and_preserves_page_offsets() {
        let (p, n, uv, i, m) = cube();
        let hierarchy = derive_detail_hierarchy(
            DetailMeshInput {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &i,
                material_ids: &m,
                weathering_masks: &[],
            },
            &[0.5, 1.0],
        )
        .unwrap();
        let pages = hierarchy
            .levels
            .iter()
            .map(|level| encode_detail_page(level).unwrap())
            .collect::<Vec<_>>();
        let key = [0x5a; 32];
        let bytes = encode_cache_file(key, &pages).unwrap();
        let decoded = decode_cache_file(&bytes, key).unwrap();
        assert_eq!(decoded.levels, hierarchy.levels);
        for ((offset, page), expected) in
            decoded.page_offsets.iter().zip(&decoded.pages).zip(&pages)
        {
            let start = *offset as usize;
            assert_eq!(page, expected);
            assert_eq!(&bytes[start..start + page.len()], page);
        }

        let mut corrupt = bytes;
        *corrupt.last_mut().unwrap() ^= 0x80;
        assert!(matches!(
            decode_cache_file(&corrupt, key),
            Err(RuntimeDetailError::CacheCorrupt)
        ));
        assert!(matches!(
            decode_cache_file(&corrupt, [0x33; 32]),
            Err(RuntimeDetailError::CacheCorrupt)
        ));
    }
}
