//! Game-agnostic Ochroma runtime/cache representation for MegaGeometry.
//!
//! A finished game object supplies only authoritative ordinary source geometry.
//! Ochroma derives runtime surface correspondence, programs, and page schedules
//! from that geometry. Spectra consumes only Ochroma data; neither layer
//! receives an authoring graph, receipt, exporter product, or authoring-tool
//! dependency. Serialization exists only for Ochroma's disposable
//! content-addressed cache and tests; this type is never an asset or `.vxp`
//! product.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use thiserror::Error;

/// Schema 8 fixes the visibility-coefficient payload to one normalized,
/// executable 14-float record per exact surface program. Schema 7 was an
/// unreleased development shape with different coefficient semantics and must
/// be re-derived rather than interpreted ambiguously.
pub const READY_MEGA_GEOMETRY_SCHEMA: u32 = 8;
pub const READY_VISIBILITY_PROGRAM_WORDS: usize = 32;
pub const READY_VISIBILITY_DEFORMATION_WORDS: usize = 16;
pub const READY_VISIBILITY_TEMPLATE_WORDS: usize = 28;
pub const READY_VISIBILITY_PROGRAM_BYTES: u32 = 128;
pub const READY_VISIBILITY_TEMPLATE_BYTES: u32 = 112;
/// `child_group, child_prim, parent_group, parent_prim, error_q, valid`.
pub const READY_SURFACE_CORRESPONDENCE_WORDS: usize = 6;
/// `parent, left, right, first_program, program_count, min.xyz, max.xyz`.
pub const READY_VISIBILITY_PAGE_NODE_WORDS: usize = 11;
pub const READY_VISIBILITY_CELL_WORDS: usize = 16;
pub const READY_VISIBILITY_COEFFICIENT_WORDS: usize = 8;
/// Page carries one independently decodable ray-native program record.
pub const GEOMETRY_PAGE_KIND_RAY_NATIVE: u32 = 0;
/// Page carries one independently decodable Ochroma-derived triangle detail level.
pub const GEOMETRY_PAGE_KIND_TRIANGLE_DETAIL: u32 = 1;
/// Spectra codec id for canonical independently-decodable `MGTL` schema-3
/// triangle detail. Kept in the game-agnostic carrier so Ochroma can describe
/// its runtime cache without depending on Spectra.
pub const GEOMETRY_PAGE_CODEC_TRIANGLE_DETAIL_MGTL_V3: u32 = 3;

/// Runtime-only provenance for one immutable Ochroma cache page. The runtime
/// geometry cache resolves the byte range once so render code can service GPU
/// requests without reopening asset-container metadata or inventing a source
/// address. This is deliberately not serialized inside `.Metadata` or `.vxp`.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeGeometryPage {
    pub page_id: u32,
    pub kind: u32,
    pub group_id: u32,
    pub vertex_count: u32,
    pub primitive_count: u32,
    /// Conservative object-space bounds shared by every level in this
    /// independently selectable cluster hierarchy.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub geometric_error_q: u32,
    pub parent_page_id: Option<u32>,
    pub source_pack_id: u32,
    pub source_path: PathBuf,
    pub source_offset: u64,
    pub byte_count: u32,
    pub decoded_byte_count: u32,
    pub codec: u32,
    pub codec_version: u32,
    pub flags: u32,
    pub content_hash: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyMegaGeometry {
    schema: u32,
    content_hash: [u8; 32],
    source_patch_count: u32,
    source_triangle_count: u32,
    /// Sorted source-mesh triangle indices replaced exactly by the programs.
    covered_triangle_indices: Vec<u32>,
    /// Finished-mesh source coordinates to canonical object-local coordinates.
    /// Runtime adapters must never infer this frame from a parcel or catalog
    /// footprint.
    source_to_object: [f32; 16],
    program_count: u32,
    deformation_count: u32,
    template_count: u32,
    selector_count: u32,
    program_words: Vec<u32>,
    deformation_words: Vec<u32>,
    template_words: Vec<u32>,
    selectors: Vec<u32>,
    /// Conservative object-space AABBs, six floats per program.
    aabbs: Vec<f32>,
    /// Ochroma-derived stable correspondence, one entry per exact program. The
    /// initial full-detail representation uses certified identity parents;
    /// future Ochroma-derived detail cuts replace only the parent pair.
    surface_correspondence_words: Vec<u32>,
    /// Ochroma-derived, full-detail page hierarchy. It is a residency/visibility
    /// structure, never a runtime-generated distance mesh.
    page_hierarchy_words: Vec<u32>,
    /// Tool-neutral sparse visibility-cell DAG. Records are fixed-width and
    /// backend neutral.
    #[serde(default)]
    visibility_cell_words: Vec<u32>,
    #[serde(default)]
    visibility_cell_children: Vec<u32>,
    /// Factorized plane/transform/polynomial coefficient descriptors and their
    /// immutable f32 parameter arena.
    #[serde(default)]
    visibility_coefficient_words: Vec<u32>,
    #[serde(default)]
    visibility_coefficient_parameters: Vec<f32>,
    /// Resolved by the runtime pack loader after metadata deserialization.
    #[serde(skip)]
    runtime_pages: Vec<RuntimeGeometryPage>,
}

impl ReadyMegaGeometry {
    pub fn schema(&self) -> u32 {
        self.schema
    }

    /// Construct runtime/cache state from mesh-certified programs.
    ///
    /// Runtime grouping, correspondence, hierarchy, and the content address are
    /// deliberately derived here in Ochroma rather than accepted from an
    /// authoring tool. The finished object's validated source contract
    /// establishes exactness; Ochroma owns how exact programs become stable
    /// runtime work.
    #[allow(clippy::too_many_arguments)]
    pub fn from_certified_programs(
        source_patch_count: u32,
        source_triangle_count: u32,
        covered_triangle_indices: Vec<u32>,
        source_to_object: [f32; 16],
        program_words: Vec<u32>,
        deformation_words: Vec<u32>,
        template_words: Vec<u32>,
        selectors: Vec<u32>,
        aabbs: Vec<f32>,
        visibility_cell_words: Vec<u32>,
        visibility_cell_children: Vec<u32>,
        visibility_coefficient_words: Vec<u32>,
        visibility_coefficient_parameters: Vec<f32>,
    ) -> Result<Self, ReadyMegaGeometryError> {
        let program_count = exact_record_count(
            "program",
            program_words.len(),
            READY_VISIBILITY_PROGRAM_WORDS,
        )?;
        if aabbs.len() != program_count as usize * 6 {
            return Err(ReadyMegaGeometryError::InvalidBufferLength("aabb"));
        }
        let surface_correspondence_words = derive_identity_surface_correspondence(&program_words);
        let page_hierarchy_words = derive_visibility_page_hierarchy(&aabbs)?;

        let mut hasher = Sha256::new();
        hasher.update(READY_MEGA_GEOMETRY_SCHEMA.to_le_bytes());
        hasher.update(source_patch_count.to_le_bytes());
        hasher.update(source_triangle_count.to_le_bytes());
        for triangle in &covered_triangle_indices {
            hasher.update(triangle.to_le_bytes());
        }
        for value in source_to_object {
            hasher.update(value.to_bits().to_le_bytes());
        }
        for word in program_words
            .iter()
            .chain(deformation_words.iter())
            .chain(template_words.iter())
            .chain(selectors.iter())
            .chain(surface_correspondence_words.iter())
            .chain(page_hierarchy_words.iter())
            .chain(visibility_cell_words.iter())
            .chain(visibility_cell_children.iter())
            .chain(visibility_coefficient_words.iter())
        {
            hasher.update(word.to_le_bytes());
        }
        for value in &aabbs {
            hasher.update(value.to_bits().to_le_bytes());
        }
        for value in &visibility_coefficient_parameters {
            hasher.update(value.to_bits().to_le_bytes());
        }
        Self::new(
            hasher.finalize().into(),
            source_patch_count,
            source_triangle_count,
            covered_triangle_indices,
            source_to_object,
            program_words,
            deformation_words,
            template_words,
            selectors,
            aabbs,
            surface_correspondence_words,
            page_hierarchy_words,
            visibility_cell_words,
            visibility_cell_children,
            visibility_coefficient_words,
            visibility_coefficient_parameters,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        content_hash: [u8; 32],
        source_patch_count: u32,
        source_triangle_count: u32,
        covered_triangle_indices: Vec<u32>,
        source_to_object: [f32; 16],
        program_words: Vec<u32>,
        deformation_words: Vec<u32>,
        template_words: Vec<u32>,
        selectors: Vec<u32>,
        aabbs: Vec<f32>,
        surface_correspondence_words: Vec<u32>,
        page_hierarchy_words: Vec<u32>,
        visibility_cell_words: Vec<u32>,
        visibility_cell_children: Vec<u32>,
        visibility_coefficient_words: Vec<u32>,
        visibility_coefficient_parameters: Vec<f32>,
    ) -> Result<Self, ReadyMegaGeometryError> {
        let program_count = exact_record_count(
            "program",
            program_words.len(),
            READY_VISIBILITY_PROGRAM_WORDS,
        )?;
        let deformation_count = exact_record_count(
            "deformation",
            deformation_words.len(),
            READY_VISIBILITY_DEFORMATION_WORDS,
        )?;
        let template_count = exact_record_count(
            "template",
            template_words.len(),
            READY_VISIBILITY_TEMPLATE_WORDS,
        )?;
        let payload = Self {
            schema: READY_MEGA_GEOMETRY_SCHEMA,
            content_hash,
            source_patch_count,
            source_triangle_count,
            covered_triangle_indices,
            source_to_object,
            program_count,
            deformation_count,
            template_count,
            selector_count: u32::try_from(selectors.len())
                .map_err(|_| ReadyMegaGeometryError::CountOverflow)?,
            program_words,
            deformation_words,
            template_words,
            selectors,
            aabbs,
            surface_correspondence_words,
            page_hierarchy_words,
            visibility_cell_words,
            visibility_cell_children,
            visibility_coefficient_words,
            visibility_coefficient_parameters,
            runtime_pages: Vec::new(),
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), ReadyMegaGeometryError> {
        if self.schema != READY_MEGA_GEOMETRY_SCHEMA && self.schema != 6 {
            return Err(ReadyMegaGeometryError::UnsupportedSchema(self.schema));
        }
        if self.program_count == 0
            || self.deformation_count == 0
            || self.template_count == 0
            || self.selector_count == 0
            || self.source_patch_count == 0
            || self.source_triangle_count != self.source_patch_count.saturating_mul(2)
            || self.covered_triangle_indices.len() != self.source_triangle_count as usize
        {
            return Err(ReadyMegaGeometryError::InvalidCoverage);
        }
        if self.source_to_object.iter().any(|value| !value.is_finite())
            || self.source_to_object[3] != 0.0
            || self.source_to_object[7] != 0.0
            || self.source_to_object[11] != 0.0
            || self.source_to_object[15] != 1.0
        {
            return Err(ReadyMegaGeometryError::InvalidSourceTransform);
        }
        if self
            .covered_triangle_indices
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(ReadyMegaGeometryError::InvalidCoverage);
        }
        self.validate_visibility_fabric()?;
        validate_flat_count(
            "program",
            &self.program_words,
            self.program_count,
            READY_VISIBILITY_PROGRAM_WORDS,
        )?;
        validate_flat_count(
            "deformation",
            &self.deformation_words,
            self.deformation_count,
            READY_VISIBILITY_DEFORMATION_WORDS,
        )?;
        validate_flat_count(
            "template",
            &self.template_words,
            self.template_count,
            READY_VISIBILITY_TEMPLATE_WORDS,
        )?;
        if self.selectors.len() != self.selector_count as usize
            || self.aabbs.len() != self.program_count as usize * 6
            || self.surface_correspondence_words.len()
                != self.program_count as usize * READY_SURFACE_CORRESPONDENCE_WORDS
            || self.page_hierarchy_words.len() % READY_VISIBILITY_PAGE_NODE_WORDS != 0
            || self.aabbs.iter().any(|value| !value.is_finite())
        {
            return Err(ReadyMegaGeometryError::InvalidBufferLength("selector/aabb"));
        }
        for record in self
            .surface_correspondence_words
            .chunks_exact(READY_SURFACE_CORRESPONDENCE_WORDS)
        {
            if record[0] >= self.program_count || record[2] >= self.program_count || record[5] > 1 {
                return Err(ReadyMegaGeometryError::InvalidSurfaceCorrespondence);
            }
        }
        let page_nodes = self
            .page_hierarchy_words
            .chunks_exact(READY_VISIBILITY_PAGE_NODE_WORDS);
        let page_node_count = page_nodes.len() as u32;
        if page_node_count == 0
            || page_nodes.clone().any(|record| {
                record[4] == 0
                    || record[3]
                        .checked_add(record[4])
                        .is_none_or(|end| end > self.program_count)
                    || record[5..11]
                        .iter()
                        .any(|word| !f32::from_bits(*word).is_finite())
                    || (0..3).any(|axis| {
                        f32::from_bits(record[5 + axis]) > f32::from_bits(record[8 + axis])
                    })
            })
        {
            return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
        }
        let page_nodes = self
            .page_hierarchy_words
            .chunks_exact(READY_VISIBILITY_PAGE_NODE_WORDS)
            .collect::<Vec<_>>();
        let roots = page_nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node[0] == u32::MAX)
            .collect::<Vec<_>>();
        if roots.len() != 1 || roots[0].1[3] != 0 || roots[0].1[4] != self.program_count {
            return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
        }
        for (index, node) in page_nodes.iter().enumerate() {
            let index = index as u32;
            let left = node[1];
            let right = node[2];
            if left == u32::MAX {
                if right != u32::MAX {
                    return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
                }
                continue;
            }
            if right == u32::MAX
                || left >= index
                || right >= index
                || page_nodes[left as usize][0] != index
                || page_nodes[right as usize][0] != index
            {
                return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
            }
        }
        for (index, record) in self
            .program_words
            .chunks_exact(READY_VISIBILITY_PROGRAM_WORDS)
            .enumerate()
        {
            if record[0] != 1
                || record[1] != READY_VISIBILITY_PROGRAM_BYTES
                || record[2] & !1 != 0
                || record[31] == 0
            {
                return Err(ReadyMegaGeometryError::InvalidRecordHeader {
                    kind: "program",
                    index,
                });
            }
            let deformation = record[9];
            let template_end = record[14].checked_add(record[15]);
            let selector_end = record[16].checked_add(record[17]);
            if deformation >= self.deformation_count
                || template_end.is_none_or(|end| end > self.template_count)
                || selector_end.is_none_or(|end| end > self.selector_count)
            {
                return Err(ReadyMegaGeometryError::RangeEscapesPayload(index));
            }
        }
        let mut realized_patch_end = 0_u32;
        for record in self
            .program_words
            .chunks_exact(READY_VISIBILITY_PROGRAM_WORDS)
        {
            if record[30] != realized_patch_end {
                return Err(ReadyMegaGeometryError::InvalidCoverage);
            }
            realized_patch_end = realized_patch_end
                .checked_add(record[31])
                .ok_or(ReadyMegaGeometryError::InvalidCoverage)?;
        }
        if realized_patch_end != self.source_patch_count {
            return Err(ReadyMegaGeometryError::InvalidCoverage);
        }
        for (index, record) in self
            .template_words
            .chunks_exact(READY_VISIBILITY_TEMPLATE_WORDS)
            .enumerate()
        {
            if record[0] != 1 || record[1] != READY_VISIBILITY_TEMPLATE_BYTES || record[2] > 4 {
                return Err(ReadyMegaGeometryError::InvalidRecordHeader {
                    kind: "template",
                    index,
                });
            }
        }
        for (program_index, record) in self
            .program_words
            .chunks_exact(READY_VISIBILITY_PROGRAM_WORDS)
            .enumerate()
        {
            let selector_offset = record[16] as usize;
            let selector_count = record[17] as usize;
            let template_count = record[15];
            if self.selectors[selector_offset..selector_offset + selector_count]
                .iter()
                .any(|selector| *selector >= template_count)
            {
                return Err(ReadyMegaGeometryError::InvalidSelector(program_index));
            }
        }
        Ok(())
    }

    fn validate_visibility_fabric(&self) -> Result<(), ReadyMegaGeometryError> {
        if self.schema == 6 {
            return if self.visibility_cell_words.is_empty()
                && self.visibility_cell_children.is_empty()
                && self.visibility_coefficient_words.is_empty()
                && self.visibility_coefficient_parameters.is_empty()
            {
                Ok(())
            } else {
                Err(ReadyMegaGeometryError::InvalidVisibilityFabric)
            };
        }
        if self.visibility_cell_words.is_empty()
            || self.visibility_cell_words.len() % READY_VISIBILITY_CELL_WORDS != 0
            || self.visibility_coefficient_words.is_empty()
            || self.visibility_coefficient_words.len() % READY_VISIBILITY_COEFFICIENT_WORDS != 0
            || self
                .visibility_coefficient_parameters
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
        }
        let cells = self
            .visibility_cell_words
            .chunks_exact(READY_VISIBILITY_CELL_WORDS)
            .collect::<Vec<_>>();
        let coefficient_count =
            (self.visibility_coefficient_words.len() / READY_VISIBILITY_COEFFICIENT_WORDS) as u32;
        let mut referenced = vec![false; cells.len()];
        for (index, cell) in cells.iter().enumerate() {
            let flags = cell[3] & 0xffff;
            let max_subdivisions = cell[3] >> 16;
            let child_end = cell[4].checked_add(cell[5]);
            let program_end = cell[6].checked_add(cell[7]);
            let coefficient_end = cell[8].checked_add(cell[9]);
            if cell[0] != 1
                || cell[1] != (READY_VISIBILITY_CELL_WORDS * 4) as u32
                || cell[2] != index as u32
                || flags & !0b11 != 0
                || max_subdivisions == 0
                || child_end.is_none_or(|end| end > self.visibility_cell_children.len() as u32)
                || program_end.is_none_or(|end| end > self.program_count)
                || coefficient_end.is_none_or(|end| end > coefficient_count)
                || cell[10..16]
                    .iter()
                    .any(|word| !f32::from_bits(*word).is_finite())
                || (0..3)
                    .any(|axis| f32::from_bits(cell[10 + axis]) > f32::from_bits(cell[13 + axis]))
            {
                return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
            }
            let children = &self.visibility_cell_children
                [cell[4] as usize..child_end.unwrap_or(cell[4]) as usize];
            if children
                .iter()
                .any(|child| *child >= index as u32 || referenced[*child as usize])
            {
                return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
            }
            for child in children {
                referenced[*child as usize] = true;
            }
            let leaf = flags & 1 != 0;
            if leaf != (children.is_empty() && cell[9] > 0) {
                return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
            }
        }
        if referenced.iter().filter(|value| !**value).count() != 1 {
            return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
        }
        for (coefficient_index, coefficient) in self
            .visibility_coefficient_words
            .chunks_exact(READY_VISIBILITY_COEFFICIENT_WORDS)
            .enumerate()
        {
            let parameter_end = coefficient[4].checked_add(coefficient[5]);
            if coefficient[0] != 1
                || coefficient[1] != (READY_VISIBILITY_COEFFICIENT_WORDS * 4) as u32
                || coefficient[2] > 1
                || coefficient[3] >= self.program_count
                || coefficient[3] != coefficient_index as u32
                || coefficient[5] != 14
                || parameter_end
                    .is_none_or(|end| end > self.visibility_coefficient_parameters.len() as u32)
                || !f32::from_bits(coefficient[6]).is_finite()
                || f32::from_bits(coefficient[6]) <= 0.0
            {
                return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
            }
            let parameters = &self.visibility_coefficient_parameters
                [coefficient[4] as usize..parameter_end.expect("checked above") as usize];
            let normal_length =
                (parameters[6] * parameters[6] + parameters[7] * parameters[7]).sqrt();
            let tangent_length =
                (parameters[9] * parameters[9] + parameters[10] * parameters[10]).sqrt();
            if (0..3).any(|axis| parameters[axis] > parameters[axis + 3])
                || (normal_length - 1.0).abs() > 1.0e-4
                || (tangent_length - 1.0).abs() > 1.0e-4
                || (parameters[6] * parameters[9] + parameters[7] * parameters[10]).abs() > 1.0e-4
                || parameters[11] <= 1.0e-5
                || parameters[13] <= parameters[12]
            {
                return Err(ReadyMegaGeometryError::InvalidVisibilityFabric);
            }
        }
        Ok(())
    }

    pub fn content_hash(&self) -> [u8; 32] {
        self.content_hash
    }
    pub fn source_patch_count(&self) -> u32 {
        self.source_patch_count
    }
    pub fn source_triangle_count(&self) -> u32 {
        self.source_triangle_count
    }
    pub fn covered_triangle_indices(&self) -> &[u32] {
        &self.covered_triangle_indices
    }
    pub fn source_to_object(&self) -> [f32; 16] {
        self.source_to_object
    }
    pub fn program_count(&self) -> u32 {
        self.program_count
    }
    pub fn deformation_count(&self) -> u32 {
        self.deformation_count
    }
    pub fn template_count(&self) -> u32 {
        self.template_count
    }
    pub fn selector_count(&self) -> u32 {
        self.selector_count
    }
    pub fn program_words(&self) -> &[u32] {
        &self.program_words
    }
    pub fn deformation_words(&self) -> &[u32] {
        &self.deformation_words
    }
    pub fn template_words(&self) -> &[u32] {
        &self.template_words
    }
    pub fn selectors(&self) -> &[u32] {
        &self.selectors
    }
    pub fn aabbs(&self) -> &[f32] {
        &self.aabbs
    }
    pub fn surface_correspondence_words(&self) -> &[u32] {
        &self.surface_correspondence_words
    }
    pub fn page_hierarchy_words(&self) -> &[u32] {
        &self.page_hierarchy_words
    }
    pub fn visibility_cell_words(&self) -> &[u32] {
        &self.visibility_cell_words
    }
    pub fn visibility_cell_children(&self) -> &[u32] {
        &self.visibility_cell_children
    }
    pub fn visibility_coefficient_words(&self) -> &[u32] {
        &self.visibility_coefficient_words
    }
    pub fn visibility_coefficient_parameters(&self) -> &[f32] {
        &self.visibility_coefficient_parameters
    }

    pub fn runtime_pages(&self) -> &[RuntimeGeometryPage] {
        &self.runtime_pages
    }

    /// Attach pack-resolved page ranges. The pack index has already validated
    /// ordering, parent closure, byte counts and SHA-256 syntax.
    pub fn set_runtime_pages(&mut self, pages: Vec<RuntimeGeometryPage>) {
        self.runtime_pages = pages;
    }
}

fn derive_identity_surface_correspondence(program_words: &[u32]) -> Vec<u32> {
    program_words
        .chunks_exact(READY_VISIBILITY_PROGRAM_WORDS)
        .enumerate()
        .flat_map(|(program_index, record)| {
            let group = program_index as u32;
            [group, record[26], group, record[26], 0, 1]
        })
        .collect()
}

/// Deterministic bottom-up hierarchy over the authoritative program order.
/// Nodes are emitted leaves-first so every child id is lower than its parent.
fn derive_visibility_page_hierarchy(aabbs: &[f32]) -> Result<Vec<u32>, ReadyMegaGeometryError> {
    if aabbs.is_empty() || aabbs.len() % 6 != 0 {
        return Err(ReadyMegaGeometryError::InvalidBufferLength("aabb"));
    }
    #[derive(Clone, Copy)]
    struct Node {
        parent: u32,
        left: u32,
        right: u32,
        first: u32,
        count: u32,
        min: [f32; 3],
        max: [f32; 3],
    }
    const NONE: u32 = u32::MAX;
    let mut nodes = Vec::<Node>::with_capacity(aabbs.len() / 3);
    for (index, bounds) in aabbs.chunks_exact(6).enumerate() {
        let min = [bounds[0], bounds[1], bounds[2]];
        let max = [bounds[3], bounds[4], bounds[5]];
        if min
            .iter()
            .zip(max)
            .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > &hi)
        {
            return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
        }
        nodes.push(Node {
            parent: NONE,
            left: NONE,
            right: NONE,
            first: index as u32,
            count: 1,
            min,
            max,
        });
    }
    let mut level = (0..nodes.len() as u32).collect::<Vec<_>>();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let left = pair[0];
            let right = *pair.get(1).unwrap_or(&left);
            let a = nodes[left as usize];
            let b = nodes[right as usize];
            let id = nodes.len() as u32;
            nodes[left as usize].parent = id;
            if right != left {
                nodes[right as usize].parent = id;
            }
            nodes.push(Node {
                parent: NONE,
                left,
                right,
                first: a.first,
                count: a.count + if right == left { 0 } else { b.count },
                min: [
                    a.min[0].min(b.min[0]),
                    a.min[1].min(b.min[1]),
                    a.min[2].min(b.min[2]),
                ],
                max: [
                    a.max[0].max(b.max[0]),
                    a.max[1].max(b.max[1]),
                    a.max[2].max(b.max[2]),
                ],
            });
            next.push(id);
        }
        level = next;
    }
    Ok(nodes
        .into_iter()
        .flat_map(|node| {
            [
                node.parent,
                node.left,
                node.right,
                node.first,
                node.count,
                node.min[0].to_bits(),
                node.min[1].to_bits(),
                node.min[2].to_bits(),
                node.max[0].to_bits(),
                node.max[1].to_bits(),
                node.max[2].to_bits(),
            ]
        })
        .collect())
}

fn exact_record_count(
    kind: &'static str,
    words: usize,
    stride: usize,
) -> Result<u32, ReadyMegaGeometryError> {
    if words == 0 || words % stride != 0 {
        return Err(ReadyMegaGeometryError::InvalidBufferLength(kind));
    }
    u32::try_from(words / stride).map_err(|_| ReadyMegaGeometryError::CountOverflow)
}

fn validate_flat_count(
    kind: &'static str,
    words: &[u32],
    count: u32,
    stride: usize,
) -> Result<(), ReadyMegaGeometryError> {
    if words.len() != count as usize * stride {
        return Err(ReadyMegaGeometryError::InvalidBufferLength(kind));
    }
    Ok(())
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ReadyMegaGeometryError {
    #[error("unsupported ReadyMegaGeometry schema {0}")]
    UnsupportedSchema(u32),
    #[error("invalid MegaGeometry source coverage")]
    InvalidCoverage,
    #[error("invalid MegaGeometry derived source-to-object transform")]
    InvalidSourceTransform,
    #[error("invalid {0} buffer length")]
    InvalidBufferLength(&'static str),
    #[error("MegaGeometry count exceeds u32")]
    CountOverflow,
    #[error("invalid {kind} record header at {index}")]
    InvalidRecordHeader { kind: &'static str, index: usize },
    #[error("program {0} references data outside its payload")]
    RangeEscapesPayload(usize),
    #[error("program {0} contains an invalid local template selector")]
    InvalidSelector(usize),
    #[error("invalid MegaGeometry surface correspondence")]
    InvalidSurfaceCorrespondence,
    #[error("invalid MegaGeometry page hierarchy")]
    InvalidPageHierarchy,
    #[error("invalid MegaGeometry visibility-cell/coefficient fabric")]
    InvalidVisibilityFabric,
}
