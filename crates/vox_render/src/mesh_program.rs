//! Engine-owned exact visibility-program derivation from finished meshes.
//!
//! Authoring tools provide ordinary indexed geometry. Ochroma may replace a
//! subset of that geometry with ray-native programs only when the finished
//! vertices, winding, UVs, and material ids prove an exact vertical rectangle
//! lattice. Everything else remains authoritative triangle detail.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use vox_data::mega_geometry::{
    READY_VISIBILITY_CELL_WORDS, READY_VISIBILITY_COEFFICIENT_WORDS,
    READY_VISIBILITY_DEFORMATION_WORDS, READY_VISIBILITY_PROGRAM_WORDS,
    READY_VISIBILITY_TEMPLATE_WORDS, ReadyMegaGeometry, ReadyMegaGeometryError,
};

const QUANTIZATION: f64 = 100_000.0;
const UV_EPSILON: f32 = 2.0e-4;
const NORMAL_EPSILON: f32 = 1.0e-4;
const COORDINATE_SCALE: u32 = 1_000_000;
const PROGRAM_CACHE_SCHEMA: u32 = 2;
/// Runtime deformation values inside this interval are guaranteed to remain
/// inside every program's admitted native procedural AABB.
const DEFORMATION_SCALE_MIN: f32 = 0.975;
const DEFORMATION_SCALE_MAX: f32 = 1.025;
static PROGRAM_CACHE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
pub struct MeshProgramInput<'a> {
    pub positions: &'a [[f32; 3]],
    pub indices: &'a [[u32; 3]],
    pub uvs: &'a [[f32; 2]],
    pub material_ids: &'a [u32],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshProgramDerivation {
    pub payload: Option<ReadyMegaGeometry>,
    pub exact_quad_count: u32,
    pub program_count: u32,
    pub covered_triangle_count: u32,
    pub residual_triangle_count: u32,
    pub rejected_quad_candidates: u32,
}

#[derive(Debug, Error)]
pub enum MeshProgramError {
    #[error("finished mesh has inconsistent or invalid streams")]
    InvalidMesh,
    #[error("derived MegaGeometry payload is invalid: {0}")]
    InvalidPayload(#[from] ReadyMegaGeometryError),
    #[error("derived visibility representation exceeds its bounded u32 address space")]
    AddressOverflow,
    #[error("runtime geometry-program cache I/O at {path}: {source}")]
    CacheIo {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("runtime geometry-program cache encoding failed: {0}")]
    CacheEncoding(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
pub struct MeshProgramPreparation {
    pub derivation: MeshProgramDerivation,
    pub cache_path: PathBuf,
    pub cache_hit: bool,
}

#[derive(Serialize, Deserialize)]
struct ProgramCache {
    schema: u32,
    source_hash: [u8; 32],
    derivation: MeshProgramDerivation,
}

/// Load or derive Ochroma's exact mesh programs from the ordinary finished
/// mesh. The cache is engine-owned disposable runtime state, never an asset
/// sidecar and never an authoring product.
pub fn prepare_mesh_programs(
    input: MeshProgramInput<'_>,
) -> Result<MeshProgramPreparation, MeshProgramError> {
    validate_input(input)?;
    let source_hash = finished_mesh_hash(input);
    let cache_root = dirs_next::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("ochroma")
        .join("mega_geometry");
    std::fs::create_dir_all(&cache_root).map_err(|source| MeshProgramError::CacheIo {
        path: cache_root.clone(),
        source,
    })?;
    let cache_path = cache_root.join(format!("{}.mgp", hex_hash(program_cache_key(source_hash))));
    if let Some(derivation) = load_program_cache(&cache_path, source_hash)? {
        if validate_cached_derivation(input, &derivation).is_ok() {
            return Ok(MeshProgramPreparation {
                derivation,
                cache_path,
                cache_hit: true,
            });
        }
        let _ = std::fs::remove_file(&cache_path);
    }
    let derivation = derive_mesh_programs(input)?;
    validate_cached_derivation(input, &derivation)?;
    let bytes = serde_json::to_vec(&ProgramCache {
        schema: PROGRAM_CACHE_SCHEMA,
        source_hash,
        derivation: derivation.clone(),
    })?;
    write_program_cache_atomically(&cache_path, &bytes)?;
    Ok(MeshProgramPreparation {
        derivation,
        cache_path,
        cache_hit: false,
    })
}

fn program_cache_key(source_hash: [u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"OCHROMA_RUNTIME_MESH_PROGRAM");
    hash.update(PROGRAM_CACHE_SCHEMA.to_le_bytes());
    hash.update(source_hash);
    hash.finalize().into()
}

fn load_program_cache(
    path: &Path,
    source_hash: [u8; 32],
) -> Result<Option<MeshProgramDerivation>, MeshProgramError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(MeshProgramError::CacheIo {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let Ok(cache) = serde_json::from_slice::<ProgramCache>(&bytes) else {
        return Ok(None);
    };
    if cache.schema != PROGRAM_CACHE_SCHEMA || cache.source_hash != source_hash {
        return Ok(None);
    }
    if validate_cached_derivation_counts(&cache.derivation).is_err() {
        return Ok(None);
    }
    Ok(Some(cache.derivation))
}

fn validate_cached_derivation(
    input: MeshProgramInput<'_>,
    derivation: &MeshProgramDerivation,
) -> Result<(), MeshProgramError> {
    validate_cached_derivation_counts(derivation)?;
    if derivation.covered_triangle_count + derivation.residual_triangle_count
        != input.indices.len() as u32
        || derivation.payload.as_ref().is_some_and(|payload| {
            payload
                .covered_triangle_indices()
                .iter()
                .any(|triangle| *triangle as usize >= input.indices.len())
        })
    {
        return Err(MeshProgramError::InvalidMesh);
    }
    Ok(())
}

fn validate_cached_derivation_counts(
    derivation: &MeshProgramDerivation,
) -> Result<(), MeshProgramError> {
    match &derivation.payload {
        Some(payload) => {
            payload.validate()?;
            if payload.program_count() != derivation.program_count
                || payload.source_triangle_count() != derivation.covered_triangle_count
                || payload.covered_triangle_indices().len()
                    != derivation.covered_triangle_count as usize
            {
                return Err(MeshProgramError::InvalidMesh);
            }
        }
        None if derivation.program_count != 0 || derivation.covered_triangle_count != 0 => {
            return Err(MeshProgramError::InvalidMesh);
        }
        None => {}
    }
    Ok(())
}

fn write_program_cache_atomically(path: &Path, bytes: &[u8]) -> Result<(), MeshProgramError> {
    let parent = path.parent().ok_or(MeshProgramError::InvalidMesh)?;
    let sequence = PROGRAM_CACHE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("program"),
        std::process::id(),
        sequence,
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|source| MeshProgramError::CacheIo {
            path: temp.clone(),
            source,
        })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| MeshProgramError::CacheIo {
            path: temp.clone(),
            source,
        })?;
    drop(file);
    match std::fs::rename(&temp, path) {
        Ok(()) => {}
        Err(_) if std::fs::read(path).is_ok_and(|existing| existing == bytes) => {
            let _ = std::fs::remove_file(&temp);
        }
        Err(_) => {
            if path.exists() {
                std::fs::remove_file(path).map_err(|source| MeshProgramError::CacheIo {
                    path: path.to_path_buf(),
                    source,
                })?;
            }
            std::fs::rename(&temp, path).map_err(|source| MeshProgramError::CacheIo {
                path: path.to_path_buf(),
                source,
            })?;
        }
    }
    Ok(())
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct QPoint(i64, i64, i64);

impl QPoint {
    fn from_position(value: [f32; 3]) -> Option<Self> {
        value
            .iter()
            .all(|v| v.is_finite())
            .then(|| Self(quantize(value[0]), quantize(value[1]), quantize(value[2])))
    }
}

#[derive(Clone, Copy, Debug)]
struct Quad {
    edge_a: [f32; 2],
    y0: f32,
    width: f32,
    height: f32,
    tangent: [f32; 2],
    normal: [f32; 2],
    uv_mode: u32,
    uv_scale: f32,
    material: u32,
    triangles: [u32; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct LatticeKey {
    tangent_x: i64,
    tangent_z: i64,
    plane_offset: i64,
    width: i64,
    height: i64,
    uv_mode: u32,
    uv_scale: i64,
}

#[derive(Clone, Debug)]
struct Cell {
    x: i32,
    y: i32,
    quad: Quad,
}

#[derive(Clone, Debug)]
struct Program {
    edge_a: [f32; 2],
    edge_b: [f32; 2],
    y_base: f32,
    floor_height: f32,
    bays: u32,
    floors: u32,
    uv_mode: u32,
    uv_scale: f32,
    patch_u1: f32,
    patch_y1: f32,
    cells: Vec<Option<Quad>>,
}

pub fn derive_mesh_programs(
    input: MeshProgramInput<'_>,
) -> Result<MeshProgramDerivation, MeshProgramError> {
    validate_input(input)?;
    let (quads, rejected_quad_candidates) = extract_exact_quads(input);
    let programs = build_lattices(&quads)?;
    if programs.is_empty() {
        return Ok(MeshProgramDerivation {
            payload: None,
            exact_quad_count: quads.len() as u32,
            program_count: 0,
            covered_triangle_count: 0,
            residual_triangle_count: input.indices.len() as u32,
            rejected_quad_candidates,
        });
    }

    let mut covered = programs
        .iter()
        .flat_map(|program| program.cells.iter().flatten())
        .flat_map(|quad| quad.triangles)
        .collect::<Vec<_>>();
    covered.sort_unstable();
    covered.dedup();
    if covered.len() % 2 != 0 {
        return Err(MeshProgramError::InvalidMesh);
    }

    let source_patch_count =
        u32::try_from(covered.len() / 2).map_err(|_| MeshProgramError::AddressOverflow)?;
    let source_triangle_count =
        u32::try_from(covered.len()).map_err(|_| MeshProgramError::AddressOverflow)?;
    let mut program_words = Vec::new();
    let deformation_words = identity_deformation_words(&programs);
    let mut template_words = Vec::new();
    let mut selectors = Vec::new();
    let mut aabbs = Vec::new();
    let mut realized_patch_offset = 0_u32;
    let source_hash = finished_mesh_hash(input);

    for (program_index, program) in programs.iter().enumerate() {
        let template_offset = u32::try_from(template_words.len() / READY_VISIBILITY_TEMPLATE_WORDS)
            .map_err(|_| MeshProgramError::AddressOverflow)?;
        let mut material_templates = BTreeMap::<Option<u32>, u32>::new();
        for material in std::iter::once(None).chain(
            program
                .cells
                .iter()
                .filter_map(|cell| cell.as_ref().map(|quad| Some(quad.material))),
        ) {
            if material_templates.contains_key(&material) {
                continue;
            }
            let local = u32::try_from(material_templates.len())
                .map_err(|_| MeshProgramError::AddressOverflow)?;
            material_templates.insert(material, local);
            encode_template(
                material,
                program.patch_u1,
                program.patch_y1,
                &mut template_words,
            )?;
        }
        let selector_offset =
            u32::try_from(selectors.len()).map_err(|_| MeshProgramError::AddressOverflow)?;
        for cell in &program.cells {
            selectors.push(material_templates[&cell.as_ref().map(|quad| quad.material)]);
        }
        let realized_patch_count = u32::try_from(program.cells.iter().flatten().count())
            .map_err(|_| MeshProgramError::AddressOverflow)?;
        let key = program_surface_key(source_hash, program_index as u32);
        encode_program(
            program,
            key,
            program_index as u32,
            template_offset,
            u32::try_from(material_templates.len())
                .map_err(|_| MeshProgramError::AddressOverflow)?,
            selector_offset,
            u32::try_from(program.cells.len()).map_err(|_| MeshProgramError::AddressOverflow)?,
            realized_patch_offset,
            realized_patch_count,
            &mut program_words,
        );
        aabbs.extend(program_aabb(program));
        realized_patch_offset = realized_patch_offset
            .checked_add(realized_patch_count)
            .ok_or(MeshProgramError::AddressOverflow)?;
    }
    if realized_patch_offset != source_patch_count {
        return Err(MeshProgramError::InvalidMesh);
    }
    let (cells, children, coefficients, parameters) =
        encode_visibility_fabric(&programs, &aabbs, covered.len() != input.indices.len())?;
    let payload = ReadyMegaGeometry::from_certified_programs(
        source_patch_count,
        source_triangle_count,
        covered,
        identity_transform(),
        program_words,
        deformation_words,
        template_words,
        selectors,
        aabbs,
        cells,
        children,
        coefficients,
        parameters,
    )?;
    Ok(MeshProgramDerivation {
        payload: Some(payload),
        exact_quad_count: quads.len() as u32,
        program_count: programs.len() as u32,
        covered_triangle_count: source_triangle_count,
        residual_triangle_count: u32::try_from(input.indices.len())
            .map_err(|_| MeshProgramError::AddressOverflow)?
            .saturating_sub(source_triangle_count),
        rejected_quad_candidates,
    })
}

fn validate_input(input: MeshProgramInput<'_>) -> Result<(), MeshProgramError> {
    if input.positions.is_empty()
        || input.indices.is_empty()
        || input.material_ids.len() != input.indices.len()
        || (!input.uvs.is_empty() && input.uvs.len() != input.positions.len())
        || input
            .positions
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || input.uvs.iter().flatten().any(|value| !value.is_finite())
        || input
            .indices
            .iter()
            .flatten()
            .any(|index| *index as usize >= input.positions.len())
    {
        return Err(MeshProgramError::InvalidMesh);
    }
    Ok(())
}

fn extract_exact_quads(input: MeshProgramInput<'_>) -> (Vec<Quad>, u32) {
    let mut edges = BTreeMap::<(QPoint, QPoint), Vec<u32>>::new();
    for (triangle_index, triangle) in input.indices.iter().enumerate() {
        for [a, b] in [
            [triangle[0], triangle[1]],
            [triangle[1], triangle[2]],
            [triangle[2], triangle[0]],
        ] {
            let Some(qa) = QPoint::from_position(input.positions[a as usize]) else {
                continue;
            };
            let Some(qb) = QPoint::from_position(input.positions[b as usize]) else {
                continue;
            };
            let edge = if qa <= qb { (qa, qb) } else { (qb, qa) };
            edges.entry(edge).or_default().push(triangle_index as u32);
        }
    }
    let mut used = BTreeSet::new();
    let mut quads = Vec::new();
    let mut rejected = 0_u32;
    for ((diagonal_a, diagonal_b), triangles) in edges {
        if triangles.len() != 2
            || used.contains(&triangles[0])
            || used.contains(&triangles[1])
            || input.material_ids[triangles[0] as usize]
                != input.material_ids[triangles[1] as usize]
        {
            continue;
        }
        match exact_quad(
            input,
            [triangles[0], triangles[1]],
            [diagonal_a, diagonal_b],
        ) {
            Some(quad) => {
                used.extend(quad.triangles);
                quads.push(quad);
            }
            None => rejected = rejected.saturating_add(1),
        }
    }
    quads.sort_by_key(|quad| (quad.triangles[0], quad.triangles[1]));
    (quads, rejected)
}

fn exact_quad(
    input: MeshProgramInput<'_>,
    mut triangles: [u32; 2],
    diagonal: [QPoint; 2],
) -> Option<Quad> {
    triangles.sort_unstable();
    let mut corners = BTreeMap::<QPoint, ([f32; 3], Vec<u32>)>::new();
    for &triangle in &triangles {
        for &vertex in &input.indices[triangle as usize] {
            let position = input.positions[vertex as usize];
            corners
                .entry(QPoint::from_position(position)?)
                .or_insert_with(|| (position, Vec::new()))
                .1
                .push(vertex);
        }
    }
    if corners.len() != 4 {
        return None;
    }
    let mut by_y = BTreeMap::<i64, Vec<(QPoint, [f32; 3])>>::new();
    for (&point, &(position, _)) in &corners {
        by_y.entry(point.1).or_default().push((point, position));
    }
    if by_y.len() != 2 || by_y.values().any(|values| values.len() != 2) {
        return None;
    }
    let mut rows = by_y.into_values();
    let lower = rows.next()?;
    let upper = rows.next()?;
    let lower_by_xz = lower
        .into_iter()
        .map(|(q, p)| ((q.0, q.2), (q, p)))
        .collect::<BTreeMap<_, _>>();
    let upper_by_xz = upper
        .into_iter()
        .map(|(q, p)| ((q.0, q.2), (q, p)))
        .collect::<BTreeMap<_, _>>();
    if lower_by_xz.keys().ne(upper_by_xz.keys()) {
        return None;
    }
    let endpoints = lower_by_xz.keys().copied().collect::<Vec<_>>();
    let mut p0 = lower_by_xz[&endpoints[0]].1;
    let mut p1 = lower_by_xz[&endpoints[1]].1;
    let mut q0 = lower_by_xz[&endpoints[0]].0;
    let mut q1 = lower_by_xz[&endpoints[1]].0;
    let mut upper0 = upper_by_xz[&endpoints[0]].1;
    let mut upper1 = upper_by_xz[&endpoints[1]].1;
    let diagonal_matches = |a: QPoint, b: QPoint| {
        (diagonal[0] == a && diagonal[1] == b) || (diagonal[0] == b && diagonal[1] == a)
    };
    if !diagonal_matches(q0, upper_by_xz[&endpoints[1]].0) {
        if !diagonal_matches(q1, upper_by_xz[&endpoints[0]].0) {
            return None;
        }
        std::mem::swap(&mut p0, &mut p1);
        std::mem::swap(&mut q0, &mut q1);
        std::mem::swap(&mut upper0, &mut upper1);
    }
    let y0 = p0[1];
    let y1 = upper0[1];
    let dx = p1[0] - p0[0];
    let dz = p1[2] - p0[2];
    let width = dx.hypot(dz);
    let height = y1 - y0;
    if width <= 1.0e-5 || height <= 1.0e-5 {
        return None;
    }
    let tangent = [dx / width, dz / width];
    let normal = [tangent[1], -tangent[0]];
    let source_normal = triangle_normal(
        input.positions[input.indices[triangles[0] as usize][0] as usize],
        input.positions[input.indices[triangles[0] as usize][1] as usize],
        input.positions[input.indices[triangles[0] as usize][2] as usize],
    )?;
    if source_normal[0] * normal[0] + source_normal[2] * normal[1] < 1.0 - NORMAL_EPSILON
        || source_normal[1].abs() > NORMAL_EPSILON
    {
        return None;
    }
    let (uv_mode, uv_scale) = prove_uvs(input, &corners, [p0, p1, upper1, upper0])?;
    Some(Quad {
        edge_a: [p0[0], p0[2]],
        y0,
        width,
        height,
        tangent,
        normal,
        uv_mode,
        uv_scale,
        material: input.material_ids[triangles[0] as usize],
        triangles,
    })
}

fn prove_uvs(
    input: MeshProgramInput<'_>,
    corners: &BTreeMap<QPoint, ([f32; 3], Vec<u32>)>,
    ordered: [[f32; 3]; 4],
) -> Option<(u32, f32)> {
    if input.uvs.is_empty() {
        let dx = (ordered[1][0] - ordered[0][0]).abs();
        let dz = (ordered[1][2] - ordered[0][2]).abs();
        return Some((u32::from(dz > dx), 1.0));
    }
    let mut uvs = Vec::with_capacity(4);
    for position in ordered {
        let (_, vertices) = corners.get(&QPoint::from_position(position)?)?;
        let first = input.uvs[*vertices.first()? as usize];
        if vertices
            .iter()
            .any(|vertex| !near2(input.uvs[*vertex as usize], first, UV_EPSILON))
        {
            return None;
        }
        uvs.push(first);
    }
    for mode in 0..=1 {
        let projected = ordered.map(|p| {
            if mode == 0 {
                [p[0], p[1]]
            } else {
                [p[2], p[1]]
            }
        });
        let mut scales = Vec::new();
        for axis in 0..2 {
            for i in 1..4 {
                let delta = projected[i][axis] - projected[0][axis];
                if delta.abs() > 1.0e-5 {
                    scales.push((uvs[i][axis] - uvs[0][axis]) / delta);
                }
            }
        }
        let Some(&scale) = scales.first() else {
            continue;
        };
        if scale <= 0.0
            || scales
                .iter()
                .any(|value| (*value - scale).abs() > UV_EPSILON)
            || (0..4).any(|i| {
                !near2(
                    uvs[i],
                    [projected[i][0] * scale, projected[i][1] * scale],
                    UV_EPSILON,
                )
            })
        {
            continue;
        }
        return Some((mode, scale));
    }
    None
}

fn build_lattices(quads: &[Quad]) -> Result<Vec<Program>, MeshProgramError> {
    let mut groups = BTreeMap::<LatticeKey, Vec<Quad>>::new();
    for &quad in quads {
        let plane_offset = quad.normal[0] * quad.edge_a[0] + quad.normal[1] * quad.edge_a[1];
        groups
            .entry(LatticeKey {
                tangent_x: quantize(quad.tangent[0]),
                tangent_z: quantize(quad.tangent[1]),
                plane_offset: quantize(plane_offset),
                width: quantize(quad.width),
                height: quantize(quad.height),
                uv_mode: quad.uv_mode,
                uv_scale: quantize(quad.uv_scale),
            })
            .or_default()
            .push(quad);
    }
    let mut programs = Vec::new();
    for (_, group) in groups {
        let s_values = group
            .iter()
            .map(|quad| dot2(quad.edge_a, quad.tangent))
            .collect::<Vec<_>>();
        let y_values = group.iter().map(|quad| quad.y0).collect::<Vec<_>>();
        let origin_s = s_values.iter().copied().fold(f32::INFINITY, f32::min);
        let origin_y = y_values.iter().copied().fold(f32::INFINITY, f32::min);
        let period_s = infer_period(&s_values, group[0].width);
        let period_y = infer_period(&y_values, group[0].height);
        let mut grid = BTreeMap::<(i32, i32), Quad>::new();
        for quad in group {
            let x = ((dot2(quad.edge_a, quad.tangent) - origin_s) / period_s).round() as i32;
            let y = ((quad.y0 - origin_y) / period_y).round() as i32;
            if ((dot2(quad.edge_a, quad.tangent) - origin_s) - x as f32 * period_s).abs()
                > 4.0 / QUANTIZATION as f32
                || ((quad.y0 - origin_y) - y as f32 * period_y).abs() > 4.0 / QUANTIZATION as f32
                || grid.insert((x, y), quad).is_some()
            {
                continue;
            }
        }
        let mut remaining = grid.keys().copied().collect::<BTreeSet<_>>();
        while let Some(seed) = remaining.iter().next().copied() {
            let mut queue = VecDeque::from([seed]);
            let mut component = Vec::new();
            remaining.remove(&seed);
            while let Some(cell) = queue.pop_front() {
                component.push(Cell {
                    x: cell.0,
                    y: cell.1,
                    quad: grid[&cell],
                });
                for neighbor in [
                    (cell.0 - 1, cell.1),
                    (cell.0 + 1, cell.1),
                    (cell.0, cell.1 - 1),
                    (cell.0, cell.1 + 1),
                ] {
                    if remaining.remove(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
            // One isolated quad costs more than its two source triangles and
            // proves no reusable topology. It stays on the exact residual path.
            if component.len() < 2 {
                continue;
            }
            component.sort_by_key(|cell| (cell.y, cell.x));
            programs.push(program_from_component(&component, period_s, period_y)?);
        }
    }
    programs.sort_by(|a, b| {
        a.y_base
            .total_cmp(&b.y_base)
            .then_with(|| a.edge_a[0].total_cmp(&b.edge_a[0]))
            .then_with(|| a.edge_a[1].total_cmp(&b.edge_a[1]))
    });
    Ok(programs)
}

fn program_from_component(
    component: &[Cell],
    period_s: f32,
    period_y: f32,
) -> Result<Program, MeshProgramError> {
    let min_x = component.iter().map(|cell| cell.x).min().unwrap_or(0);
    let max_x = component.iter().map(|cell| cell.x).max().unwrap_or(0);
    let min_y = component.iter().map(|cell| cell.y).min().unwrap_or(0);
    let max_y = component.iter().map(|cell| cell.y).max().unwrap_or(0);
    let bays = u32::try_from(max_x - min_x + 1).map_err(|_| MeshProgramError::AddressOverflow)?;
    let floors = u32::try_from(max_y - min_y + 1).map_err(|_| MeshProgramError::AddressOverflow)?;
    let first = component[0].quad;
    let base = [
        first.edge_a[0] + first.tangent[0] * (min_x - component[0].x) as f32 * period_s,
        first.edge_a[1] + first.tangent[1] * (min_x - component[0].x) as f32 * period_s,
    ];
    let end = [
        base[0] + first.tangent[0] * bays as f32 * period_s,
        base[1] + first.tangent[1] * bays as f32 * period_s,
    ];
    let cell_count = usize::try_from(bays)
        .ok()
        .and_then(|b| usize::try_from(floors).ok().and_then(|f| b.checked_mul(f)))
        .ok_or(MeshProgramError::AddressOverflow)?;
    let mut cells = vec![None; cell_count];
    for cell in component {
        let x = (cell.x - min_x) as usize;
        let y = (cell.y - min_y) as usize;
        cells[y * bays as usize + x] = Some(cell.quad);
    }
    Ok(Program {
        edge_a: base,
        edge_b: end,
        y_base: first.y0 + (min_y - component[0].y) as f32 * period_y,
        floor_height: period_y,
        bays,
        floors,
        uv_mode: first.uv_mode,
        uv_scale: first.uv_scale,
        patch_u1: (first.width / period_s).clamp(1.0e-6, 1.0),
        patch_y1: (first.height / period_y).clamp(1.0e-6, 1.0),
        cells,
    })
}

fn encode_program(
    program: &Program,
    key: [u32; 4],
    index: u32,
    template_offset: u32,
    template_count: u32,
    selector_offset: u32,
    selector_count: u32,
    realized_patch_offset: u32,
    realized_patch_count: u32,
    out: &mut Vec<u32>,
) {
    out.extend([
        1,
        (READY_VISIBILITY_PROGRAM_WORDS * 4) as u32,
        0,
        4,
        key[0],
        key[1],
        key[2],
        key[3],
        index,
        index,
        0,
        program.uv_mode,
        program.bays,
        program.floors,
        template_offset,
        template_count,
        selector_offset,
        selector_count,
        program.edge_a[0].to_bits(),
        program.edge_a[1].to_bits(),
        program.edge_b[0].to_bits(),
        program.edge_b[1].to_bits(),
        program.y_base.to_bits(),
        program.floor_height.to_bits(),
        program.uv_scale.to_bits(),
        index,
        0,
        COORDINATE_SCALE,
        8,
        3,
        realized_patch_offset,
        realized_patch_count,
    ]);
}

fn encode_template(
    material: Option<u32>,
    patch_u1: f32,
    patch_y1: f32,
    out: &mut Vec<u32>,
) -> Result<(), MeshProgramError> {
    let start = out.len();
    out.resize(start + READY_VISIBILITY_TEMPLATE_WORDS, 0);
    out[start] = 1;
    out[start + 1] = (READY_VISIBILITY_TEMPLATE_WORDS * 4) as u32;
    if let Some(material) = material {
        if material > u8::MAX as u32 {
            return Err(MeshProgramError::InvalidMesh);
        }
        out[start + 2] = 1;
        out[start + 4] = 0;
        out[start + 5] = material;
        out[start + 6] = 0;
        out[start + 7] = quantized_unit(patch_u1);
        out[start + 8] = 0;
        out[start + 9] = quantized_unit(patch_y1);
    }
    Ok(())
}

fn encode_visibility_fabric(
    programs: &[Program],
    aabbs: &[f32],
    has_residual: bool,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>), MeshProgramError> {
    let mut cells = Vec::<[u32; READY_VISIBILITY_CELL_WORDS]>::new();
    let mut children = Vec::new();
    let mut coefficients = Vec::new();
    let mut parameters = Vec::new();
    let mut active = Vec::new();
    for (index, program) in programs.iter().enumerate() {
        let bounds: [f32; 6] = aabbs[index * 6..index * 6 + 6]
            .try_into()
            .expect("one program has exactly one six-float AABB");
        let tangent = normalized2([
            program.edge_b[0] - program.edge_a[0],
            program.edge_b[1] - program.edge_a[1],
        ]);
        let normal = [tangent[1], -tangent[0]];
        let parameter_offset =
            u32::try_from(parameters.len()).map_err(|_| MeshProgramError::AddressOverflow)?;
        parameters.extend_from_slice(&bounds);
        parameters.extend([
            normal[0],
            normal[1],
            -(normal[0] * program.edge_a[0] + normal[1] * program.edge_a[1]),
            tangent[0],
            tangent[1],
            distance2(program.edge_a, program.edge_b),
            program.y_base,
            program.y_base + program.floors as f32 * program.floor_height,
        ]);
        coefficients.extend([
            1,
            (READY_VISIBILITY_COEFFICIENT_WORDS * 4) as u32,
            0,
            index as u32,
            parameter_offset,
            14,
            1.0e-6_f32.to_bits(),
            0,
        ]);
        let id = cells.len() as u32;
        cells.push(cell_record(
            id,
            u32::from(has_residual) << 1 | 1,
            0,
            0,
            index as u32,
            1,
            index as u32,
            1,
            &bounds,
        ));
        active.push((id, index as u32, 1_u32, bounds));
    }
    while active.len() > 1 {
        let mut next = Vec::new();
        for pair in active.chunks(2) {
            if pair.len() == 1 {
                next.push(pair[0]);
                continue;
            }
            let child_offset =
                u32::try_from(children.len()).map_err(|_| MeshProgramError::AddressOverflow)?;
            children.extend([pair[0].0, pair[1].0]);
            let first = pair[0].1;
            let count = pair[0]
                .2
                .checked_add(pair[1].2)
                .ok_or(MeshProgramError::AddressOverflow)?;
            let bounds = union_bounds(pair[0].3, pair[1].3);
            let id = cells.len() as u32;
            cells.push(cell_record(
                id,
                u32::from(has_residual) << 1,
                child_offset,
                2,
                first,
                count,
                first,
                count,
                &bounds,
            ));
            next.push((id, first, count, bounds));
        }
        active = next;
    }
    Ok((
        cells.into_iter().flatten().collect(),
        children,
        coefficients,
        parameters,
    ))
}

#[allow(clippy::too_many_arguments)]
fn cell_record(
    id: u32,
    flags: u32,
    child_offset: u32,
    child_count: u32,
    program_offset: u32,
    program_count: u32,
    coefficient_offset: u32,
    coefficient_count: u32,
    bounds: &[f32; 6],
) -> [u32; READY_VISIBILITY_CELL_WORDS] {
    [
        1,
        (READY_VISIBILITY_CELL_WORDS * 4) as u32,
        id,
        flags | (8 << 16),
        child_offset,
        child_count,
        program_offset,
        program_count,
        coefficient_offset,
        coefficient_count,
        bounds[0].to_bits(),
        bounds[1].to_bits(),
        bounds[2].to_bits(),
        bounds[3].to_bits(),
        bounds[4].to_bits(),
        bounds[5].to_bits(),
    ]
}

fn identity_deformation_words(programs: &[Program]) -> Vec<u32> {
    let mut words = Vec::with_capacity(programs.len() * READY_VISIBILITY_DEFORMATION_WORDS);
    for program in programs {
        let center = [
            (program.edge_a[0] + program.edge_b[0]) * 0.5,
            (program.edge_a[1] + program.edge_b[1]) * 0.5,
        ];
        let height = (program.floors as f32 * program.floor_height).max(1.0e-4);
        words.extend([
            1,
            (READY_VISIBILITY_DEFORMATION_WORDS * 4) as u32,
            0,
            0,
            center[0].to_bits(),
            center[1].to_bits(),
            1.0_f32.to_bits(),
            height.to_bits(),
            center[0].to_bits(),
            center[1].to_bits(),
            program.y_base.to_bits(),
            height.to_bits(),
            0.0_f32.to_bits(),
            0.0_f32.to_bits(),
            1.0_f32.to_bits(),
            0,
        ]);
    }
    words
}

fn identity_transform() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn program_aabb(program: &Program) -> [f32; 6] {
    let padding = 1.0e-4_f32.max(distance2(program.edge_a, program.edge_b) * 1.0e-6);
    let center = [
        (program.edge_a[0] + program.edge_b[0]) * 0.5,
        (program.edge_a[1] + program.edge_b[1]) * 0.5,
    ];
    let scaled_a = [
        center[0] + (program.edge_a[0] - center[0]) * DEFORMATION_SCALE_MAX,
        center[1] + (program.edge_a[1] - center[1]) * DEFORMATION_SCALE_MAX,
    ];
    let scaled_b = [
        center[0] + (program.edge_b[0] - center[0]) * DEFORMATION_SCALE_MAX,
        center[1] + (program.edge_b[1] - center[1]) * DEFORMATION_SCALE_MAX,
    ];
    debug_assert!(DEFORMATION_SCALE_MIN > 0.0);
    [
        scaled_a[0].min(scaled_b[0]) - padding,
        program.y_base - padding,
        scaled_a[1].min(scaled_b[1]) - padding,
        scaled_a[0].max(scaled_b[0]) + padding,
        program.y_base + program.floors as f32 * program.floor_height + padding,
        scaled_a[1].max(scaled_b[1]) + padding,
    ]
}

fn finished_mesh_hash(input: MeshProgramInput<'_>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"OCHROMA_FINISHED_MESH_PROGRAM_SOURCE");
    hash.update((input.positions.len() as u64).to_le_bytes());
    for value in input.positions.iter().flatten() {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.update((input.indices.len() as u64).to_le_bytes());
    for index in input.indices.iter().flatten() {
        hash.update(index.to_le_bytes());
    }
    hash.update((input.uvs.len() as u64).to_le_bytes());
    for value in input.uvs.iter().flatten() {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.update((input.material_ids.len() as u64).to_le_bytes());
    for material in input.material_ids {
        hash.update(material.to_le_bytes());
    }
    hash.finalize().into()
}

fn program_surface_key(source_hash: [u8; 32], index: u32) -> [u32; 4] {
    let mut hash = Sha256::new();
    hash.update(source_hash);
    hash.update(index.to_le_bytes());
    let bytes: [u8; 32] = hash.finalize().into();
    std::array::from_fn(|word| {
        u32::from_le_bytes(bytes[word * 4..word * 4 + 4].try_into().unwrap())
    })
}

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<[f32; 3]> {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let value = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    (length > 1.0e-8).then(|| [value[0] / length, value[1] / length, value[2] / length])
}

fn quantize(value: f32) -> i64 {
    (value as f64 * QUANTIZATION).round() as i64
}

fn infer_period(values: &[f32], minimum: f32) -> f32 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f32::total_cmp);
    ordered.dedup_by(|a, b| (*a - *b).abs() <= 4.0 / QUANTIZATION as f32);
    ordered
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|delta| *delta >= minimum - 4.0 / QUANTIZATION as f32)
        .fold(f32::INFINITY, f32::min)
        .is_finite()
        .then(|| {
            ordered
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .filter(|delta| *delta >= minimum - 4.0 / QUANTIZATION as f32)
                .fold(f32::INFINITY, f32::min)
                .max(minimum)
        })
        .unwrap_or(minimum)
}

fn quantized_unit(value: f32) -> u32 {
    (value.clamp(1.0 / COORDINATE_SCALE as f32, 1.0) * COORDINATE_SCALE as f32).round() as u32
}

fn near2(a: [f32; 2], b: [f32; 2], epsilon: f32) -> bool {
    (a[0] - b[0]).abs() <= epsilon && (a[1] - b[1]).abs() <= epsilon
}

fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn distance2(a: [f32; 2], b: [f32; 2]) -> f32 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn normalized2(value: [f32; 2]) -> [f32; 2] {
    let length = value[0].hypot(value[1]).max(1.0e-8);
    [value[0] / length, value[1] / length]
}

fn union_bounds(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].max(b[3]),
        a[4].max(b[4]),
        a[5].max(b[5]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_panel_wall() -> (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        // edge direction +X produces -Z, matching this winding.
        let indices = vec![[0, 2, 1], [0, 3, 2], [1, 5, 4], [1, 2, 5]];
        let uvs = positions.iter().map(|p| [p[0], p[1]]).collect();
        let materials = vec![3; 4];
        (positions, indices, uvs, materials)
    }

    #[test]
    fn finished_mesh_derives_exact_repeated_program_without_authoring_receipt() {
        let (positions, indices, uvs, material_ids) = two_panel_wall();
        let result = derive_mesh_programs(MeshProgramInput {
            positions: &positions,
            indices: &indices,
            uvs: &uvs,
            material_ids: &material_ids,
        })
        .unwrap();
        let payload = result.payload.unwrap();
        assert_eq!(payload.program_count(), 1);
        assert_eq!(payload.source_patch_count(), 2);
        assert_eq!(payload.covered_triangle_indices(), &[0, 1, 2, 3]);
        assert_eq!(result.residual_triangle_count, 0);
        payload.validate().unwrap();
    }

    #[test]
    fn uv_mismatch_stays_on_authoritative_triangle_path() {
        let (positions, indices, mut uvs, material_ids) = two_panel_wall();
        uvs[5][0] += 0.25;
        let result = derive_mesh_programs(MeshProgramInput {
            positions: &positions,
            indices: &indices,
            uvs: &uvs,
            material_ids: &material_ids,
        })
        .unwrap();
        assert!(result.payload.is_none());
        assert_eq!(result.residual_triangle_count, 4);
    }

    #[test]
    fn each_program_has_a_nontrivial_conservative_taper_envelope() {
        let (positions, indices, uvs, material_ids) = two_panel_wall();
        let payload = derive_mesh_programs(MeshProgramInput {
            positions: &positions,
            indices: &indices,
            uvs: &uvs,
            material_ids: &material_ids,
        })
        .unwrap()
        .payload
        .unwrap();

        assert_eq!(payload.deformation_count(), payload.program_count());
        let deformation = payload.deformation_words();
        assert_eq!(f32::from_bits(deformation[4]), 1.0);
        assert_eq!(f32::from_bits(deformation[6]), 1.0);
        let bounds = payload.aabbs();
        assert!(bounds[0] <= -0.025);
        assert!(bounds[3] >= 2.025);
    }

    #[test]
    fn runtime_program_cache_hit_does_not_repeat_derivation() {
        let (mut positions, indices, uvs, material_ids) = two_panel_wall();
        positions[0][2] = 0.031_25;
        positions[1][2] = 0.031_25;
        positions[2][2] = 0.031_25;
        positions[3][2] = 0.031_25;
        positions[4][2] = 0.031_25;
        positions[5][2] = 0.031_25;
        let input = MeshProgramInput {
            positions: &positions,
            indices: &indices,
            uvs: &uvs,
            material_ids: &material_ids,
        };
        let source_hash = finished_mesh_hash(input);
        let path = dirs_next::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("ochroma")
            .join("mega_geometry")
            .join(format!("{}.mgp", hex_hash(program_cache_key(source_hash))));
        let _ = std::fs::remove_file(&path);

        let first = prepare_mesh_programs(input).unwrap();
        let second = prepare_mesh_programs(input).unwrap();

        assert!(!first.cache_hit);
        assert!(second.cache_hit);
        assert_eq!(first.derivation, second.derivation);
        let _ = std::fs::remove_file(path);
    }
}
