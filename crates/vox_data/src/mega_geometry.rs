//! Serialized, game-agnostic MegaGeometry visibility payload.
//!
//! Forge cooks these backend-neutral words once. Games carry the validated
//! payload unchanged; Spectra uploads it without interpreting BlueprintGraph
//! or regenerating triangles.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const READY_MEGA_GEOMETRY_SCHEMA: u32 = 6;
pub const READY_VISIBILITY_PROGRAM_WORDS: usize = 32;
pub const READY_VISIBILITY_DEFORMATION_WORDS: usize = 16;
pub const READY_VISIBILITY_TEMPLATE_WORDS: usize = 28;
pub const READY_VISIBILITY_PROGRAM_BYTES: u32 = 128;
pub const READY_VISIBILITY_TEMPLATE_BYTES: u32 = 112;
/// `child_group, child_prim, parent_group, parent_prim, error_q, valid`.
pub const READY_SURFACE_CORRESPONDENCE_WORDS: usize = 6;
/// `parent, left, right, first_program, program_count, min.xyz, max.xyz`.
pub const READY_VISIBILITY_PAGE_NODE_WORDS: usize = 11;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyMegaGeometry {
    schema: u32,
    content_hash: [u8; 32],
    source_patch_count: u32,
    source_triangle_count: u32,
    /// Sorted source-mesh triangle indices replaced exactly by the programs.
    covered_triangle_indices: Vec<u32>,
    /// Authored source coordinates to the canonical object-local coordinates
    /// used by the unchanged cooked mesh. Runtime adapters must never infer
    /// this frame from a parcel or catalog footprint.
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
    /// Forge-cooked stable correspondence, one entry per exact program. The
    /// initial full-detail representation uses certified identity parents;
    /// future cooked LOD cuts replace only the parent pair, never invent data
    /// at runtime.
    surface_correspondence_words: Vec<u32>,
    /// Forge-cooked, full-detail page hierarchy. It is a residency/visibility
    /// structure, never a runtime-generated distance mesh.
    page_hierarchy_words: Vec<u32>,
}

impl ReadyMegaGeometry {
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
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), ReadyMegaGeometryError> {
        if self.schema != READY_MEGA_GEOMETRY_SCHEMA {
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
        let page_nodes = self.page_hierarchy_words.chunks_exact(READY_VISIBILITY_PAGE_NODE_WORDS);
        let page_node_count = page_nodes.len() as u32;
        if page_node_count == 0 || page_nodes.clone().any(|record| {
            record[4] == 0
                || record[3].checked_add(record[4]).is_none_or(|end| end > self.program_count)
                || record[5..11].iter().any(|word| !f32::from_bits(*word).is_finite())
                || (0..3).any(|axis| f32::from_bits(record[5 + axis]) > f32::from_bits(record[8 + axis]))
        }) {
            return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
        }
        let page_nodes = self.page_hierarchy_words.chunks_exact(READY_VISIBILITY_PAGE_NODE_WORDS).collect::<Vec<_>>();
        let roots = page_nodes.iter().enumerate().filter(|(_, node)| node[0] == u32::MAX).collect::<Vec<_>>();
        if roots.len() != 1 || roots[0].1[3] != 0 || roots[0].1[4] != self.program_count {
            return Err(ReadyMegaGeometryError::InvalidPageHierarchy);
        }
        for (index, node) in page_nodes.iter().enumerate() {
            let index = index as u32;
            let left = node[1]; let right = node[2];
            if left == u32::MAX {
                if right != u32::MAX { return Err(ReadyMegaGeometryError::InvalidPageHierarchy); }
                continue;
            }
            if right == u32::MAX || left >= index || right >= index
                || page_nodes[left as usize][0] != index || page_nodes[right as usize][0] != index {
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
            if record[0] != 1
                || record[1] != READY_VISIBILITY_TEMPLATE_BYTES
                || record[2] > 4
            {
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
    pub fn page_hierarchy_words(&self) -> &[u32] { &self.page_hierarchy_words }
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
    #[error("invalid MegaGeometry authored source-to-object transform")]
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
}
