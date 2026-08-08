//! Validated, backend-neutral partition of one authoritative source mesh.
//!
//! This is an Ochroma MegaGeometry input contract, not an authored mesh-LOD
//! chain. Every source triangle appears exactly once and retains its original
//! identity. Native backends may lower the partition to triangle GAS, CLAS,
//! procedural acceleration, resident pages, or another measured
//! representation without changing the game object.

use crate::mega_geometry::RuntimeGeometryPage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const READY_GEOMETRY_CLUSTERS_SCHEMA: u32 = 2;
/// Native acceleration partition, not a meshlet or I/O-sector size. A 256
/// triangle ceiling multiplied a city into tens of thousands of BLASes on
/// Vulkan/Metal and made acceleration metadata dominate memory and rays.
/// Four thousand triangles remains a bounded, independently rebuildable RT
/// unit while amortising BLAS/TLAS metadata and preserving useful QEM interiors.
pub const MAX_TRIANGLES_PER_SOURCE_CLUSTER: u32 = 4096;
pub const MIN_TRIANGLES_PER_SOURCE_CLUSTER: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyGeometryCluster {
    id: u32,
    triangle_order_start: u32,
    triangle_count: u32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

impl ReadyGeometryCluster {
    pub fn new(
        id: u32,
        triangle_order_start: u32,
        triangle_count: u32,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) -> Result<Self, ReadyGeometryClustersError> {
        let cluster = Self {
            id,
            triangle_order_start,
            triangle_count,
            bounds_min,
            bounds_max,
        };
        cluster.validate()?;
        Ok(cluster)
    }

    pub fn id(self) -> u32 {
        self.id
    }

    pub fn triangle_order_start(self) -> u32 {
        self.triangle_order_start
    }

    pub fn triangle_count(self) -> u32 {
        self.triangle_count
    }

    pub fn bounds_min(self) -> [f32; 3] {
        self.bounds_min
    }

    pub fn bounds_max(self) -> [f32; 3] {
        self.bounds_max
    }

    fn validate(&self) -> Result<(), ReadyGeometryClustersError> {
        if self.triangle_count == 0 || self.triangle_count > MAX_TRIANGLES_PER_SOURCE_CLUSTER {
            return Err(ReadyGeometryClustersError::InvalidClusterSize {
                id: self.id,
                triangle_count: self.triangle_count,
            });
        }
        if self
            .bounds_min
            .iter()
            .chain(self.bounds_max.iter())
            .any(|value| !value.is_finite())
            || (0..3).any(|axis| self.bounds_min[axis] > self.bounds_max[axis])
        {
            return Err(ReadyGeometryClustersError::InvalidBounds(self.id));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyGeometryClusters {
    schema: u32,
    content_hash: [u8; 32],
    source_triangle_count: u32,
    triangle_order: Vec<u32>,
    clusters: Vec<ReadyGeometryCluster>,
    /// Resolved by the runtime pack loader. These immutable pages are not part
    /// of authored metadata or the source-partition content hash.
    #[serde(skip)]
    runtime_pages: Vec<RuntimeGeometryPage>,
}

impl ReadyGeometryClusters {
    /// Derive Ochroma's canonical full-detail source partition directly from
    /// the authoritative mesh. This is structural asset preparation, not a
    /// distance LOD or an authoring-tool schedule.
    pub fn from_source_mesh(
        positions: &[[f32; 3]],
        indices: &[[u32; 3]],
    ) -> Result<Self, ReadyGeometryClustersError> {
        if indices.is_empty() {
            return Err(ReadyGeometryClustersError::InvalidCoverage);
        }
        let source_triangle_count = u32::try_from(indices.len())
            .map_err(|_| ReadyGeometryClustersError::TriangleCountOverflow)?;
        let mut centroid_min = [f32::INFINITY; 3];
        let mut centroid_max = [f32::NEG_INFINITY; 3];
        for (triangle, source_indices) in indices.iter().enumerate() {
            let mut centroid = [0.0_f32; 3];
            for &vertex in source_indices {
                let position = positions.get(vertex as usize).ok_or(
                    ReadyGeometryClustersError::VertexOutOfRange {
                        triangle: triangle as u32,
                        vertex,
                    },
                )?;
                if position.iter().any(|value| !value.is_finite()) {
                    return Err(ReadyGeometryClustersError::NonFinitePosition(vertex));
                }
                for axis in 0..3 {
                    centroid[axis] += position[axis] / 3.0;
                }
            }
            for axis in 0..3 {
                centroid_min[axis] = centroid_min[axis].min(centroid[axis]);
                centroid_max[axis] = centroid_max[axis].max(centroid[axis]);
            }
        }
        let triangle_order =
            canonical_morton_order(positions, indices, centroid_min, centroid_max);
        let ranges = canonical_cluster_ranges(indices.len());
        let mut clusters = Vec::with_capacity(ranges.len());
        for (id, (start, count)) in ranges.into_iter().enumerate() {
            let mut bounds_min = [f32::INFINITY; 3];
            let mut bounds_max = [f32::NEG_INFINITY; 3];
            for &source_triangle in &triangle_order[start..start + count] {
                for &vertex in &indices[source_triangle as usize] {
                    let position = positions[vertex as usize];
                    for axis in 0..3 {
                        bounds_min[axis] = bounds_min[axis].min(position[axis]);
                        bounds_max[axis] = bounds_max[axis].max(position[axis]);
                    }
                }
            }
            clusters.push(ReadyGeometryCluster::new(
                id as u32,
                start as u32,
                count as u32,
                bounds_min,
                bounds_max,
            )?);
        }
        Self::new(source_triangle_count, triangle_order, clusters)
    }

    pub fn new(
        source_triangle_count: u32,
        triangle_order: Vec<u32>,
        clusters: Vec<ReadyGeometryCluster>,
    ) -> Result<Self, ReadyGeometryClustersError> {
        let content_hash = content_hash(source_triangle_count, &triangle_order, &clusters);
        let ready = Self {
            schema: READY_GEOMETRY_CLUSTERS_SCHEMA,
            content_hash,
            source_triangle_count,
            triangle_order,
            clusters,
            runtime_pages: Vec::new(),
        };
        ready.validate()?;
        Ok(ready)
    }

    pub fn validate(&self) -> Result<(), ReadyGeometryClustersError> {
        if self.schema != READY_GEOMETRY_CLUSTERS_SCHEMA {
            return Err(ReadyGeometryClustersError::UnsupportedSchema(self.schema));
        }
        if self.source_triangle_count == 0
            || self.triangle_order.len() != self.source_triangle_count as usize
            || self.clusters.is_empty()
        {
            return Err(ReadyGeometryClustersError::InvalidCoverage);
        }
        if self.content_hash
            != content_hash(
                self.source_triangle_count,
                &self.triangle_order,
                &self.clusters,
            )
        {
            return Err(ReadyGeometryClustersError::ContentHashMismatch);
        }

        let mut seen = vec![false; self.source_triangle_count as usize];
        for &triangle in &self.triangle_order {
            let Some(slot) = seen.get_mut(triangle as usize) else {
                return Err(ReadyGeometryClustersError::TriangleOutOfRange(triangle));
            };
            if std::mem::replace(slot, true) {
                return Err(ReadyGeometryClustersError::DuplicateTriangle(triangle));
            }
        }

        let mut expected_start = 0_u32;
        for (index, cluster) in self.clusters.iter().enumerate() {
            cluster.validate()?;
            if cluster.id != index as u32 {
                return Err(ReadyGeometryClustersError::NonCanonicalClusterId {
                    expected: index as u32,
                    actual: cluster.id,
                });
            }
            if cluster.triangle_order_start != expected_start {
                return Err(ReadyGeometryClustersError::NonContiguousClusterRange {
                    id: cluster.id,
                    expected_start,
                    actual_start: cluster.triangle_order_start,
                });
            }
            if self.clusters.len() > 1 && cluster.triangle_count < MIN_TRIANGLES_PER_SOURCE_CLUSTER
            {
                return Err(ReadyGeometryClustersError::InvalidClusterSize {
                    id: cluster.id,
                    triangle_count: cluster.triangle_count,
                });
            }
            expected_start = expected_start
                .checked_add(cluster.triangle_count)
                .ok_or(ReadyGeometryClustersError::InvalidCoverage)?;
        }
        if expected_start != self.source_triangle_count {
            return Err(ReadyGeometryClustersError::InvalidCoverage);
        }
        Ok(())
    }

    pub fn schema(&self) -> u32 {
        self.schema
    }

    pub fn content_hash(&self) -> [u8; 32] {
        self.content_hash
    }

    pub fn source_triangle_count(&self) -> u32 {
        self.source_triangle_count
    }

    pub fn triangle_order(&self) -> &[u32] {
        &self.triangle_order
    }

    pub fn clusters(&self) -> &[ReadyGeometryCluster] {
        &self.clusters
    }

    /// Consume the validated partition into its two runtime streams.
    ///
    /// Scene assembly uses this after validation so a large prototype's Morton
    /// order can move into the resident CLAS table instead of being copied from
    /// a temporary [`ReadyGeometryClusters`] allocation.
    pub fn into_partition(self) -> (Vec<u32>, Vec<ReadyGeometryCluster>) {
        (self.triangle_order, self.clusters)
    }

    pub fn runtime_pages(&self) -> &[RuntimeGeometryPage] {
        &self.runtime_pages
    }

    pub fn set_runtime_pages(&mut self, pages: Vec<RuntimeGeometryPage>) {
        self.runtime_pages = pages;
    }
}

fn canonical_cluster_ranges(triangle_count: usize) -> Vec<(usize, usize)> {
    let maximum = MAX_TRIANGLES_PER_SOURCE_CLUSTER as usize;
    let minimum = MIN_TRIANGLES_PER_SOURCE_CLUSTER as usize;
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < triangle_count {
        let count = (triangle_count - start).min(maximum);
        ranges.push((start, count));
        start += count;
    }
    if ranges.len() > 1 && ranges.last().is_some_and(|range| range.1 < minimum) {
        let tail = ranges.pop().expect("tail exists");
        let previous = ranges.pop().expect("previous range exists");
        let total = previous.1 + tail.1;
        let first = total / 2;
        ranges.push((previous.0, first));
        ranges.push((previous.0 + first, total - first));
    }
    ranges
}

fn canonical_morton_order(
    positions: &[[f32; 3]],
    indices: &[[u32; 3]],
    minimum: [f32; 3],
    maximum: [f32; 3],
) -> Vec<u32> {
    let extent = [
        (maximum[0] - minimum[0]).max(1.0e-6),
        (maximum[1] - minimum[1]).max(1.0e-6),
        (maximum[2] - minimum[2]).max(1.0e-6),
    ];
    // One packed u64 per triangle replaces the old 12-byte centroid array plus
    // an 8-byte keyed array. The first validation pass above already established
    // finite/in-range source data; recomputing three adds here is cheaper than
    // retaining hundreds of MiB solely across the sort.
    let mut keyed = indices
        .iter()
        .enumerate()
        .map(|(triangle, source_indices)| {
            let mut centroid = [0.0_f32; 3];
            for &vertex in source_indices {
                let position = positions[vertex as usize];
                for axis in 0..3 {
                    centroid[axis] += position[axis] / 3.0;
                }
            }
            let quantized = [0, 1, 2].map(|axis| {
                (((centroid[axis] - minimum[axis]) / extent[axis]).clamp(0.0, 1.0) * 1023.0).round()
                    as u32
            });
            (u64::from(morton_3d_10bit(
                quantized[0],
                quantized[1],
                quantized[2],
            )) << 32)
                | triangle as u64
        })
        .collect::<Vec<_>>();
    keyed.sort_unstable();
    keyed.into_iter().map(|entry| entry as u32).collect()
}

fn morton_3d_10bit(x: u32, y: u32, z: u32) -> u32 {
    fn spread_10(mut value: u32) -> u32 {
        value &= 0x0000_03ff;
        value = (value | value << 16) & 0x0300_00ff;
        value = (value | value << 8) & 0x0300_f00f;
        value = (value | value << 4) & 0x030c_30c3;
        value = (value | value << 2) & 0x0924_9249;
        value
    }
    spread_10(x) | (spread_10(y) << 1) | (spread_10(z) << 2)
}

fn content_hash(
    source_triangle_count: u32,
    triangle_order: &[u32],
    clusters: &[ReadyGeometryCluster],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(READY_GEOMETRY_CLUSTERS_SCHEMA.to_le_bytes());
    hash.update(source_triangle_count.to_le_bytes());
    hash.update((triangle_order.len() as u64).to_le_bytes());
    for triangle in triangle_order {
        hash.update(triangle.to_le_bytes());
    }
    hash.update((clusters.len() as u64).to_le_bytes());
    for cluster in clusters {
        hash.update(cluster.id.to_le_bytes());
        hash.update(cluster.triangle_order_start.to_le_bytes());
        hash.update(cluster.triangle_count.to_le_bytes());
        for value in cluster.bounds_min.iter().chain(cluster.bounds_max.iter()) {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    hash.finalize().into()
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ReadyGeometryClustersError {
    #[error("unsupported ReadyGeometryClusters schema {0}")]
    UnsupportedSchema(u32),
    #[error("source-cluster coverage is not an exact source-triangle partition")]
    InvalidCoverage,
    #[error("source-cluster content hash does not match its payload")]
    ContentHashMismatch,
    #[error("source triangle {0} is out of range")]
    TriangleOutOfRange(u32),
    #[error("source triangle {0} appears more than once")]
    DuplicateTriangle(u32),
    #[error("cluster id is not canonical: expected {expected}, got {actual}")]
    NonCanonicalClusterId { expected: u32, actual: u32 },
    #[error(
        "cluster {id} range is not contiguous: expected start {expected_start}, got {actual_start}"
    )]
    NonContiguousClusterRange {
        id: u32,
        expected_start: u32,
        actual_start: u32,
    },
    #[error("cluster {id} has invalid triangle count {triangle_count}")]
    InvalidClusterSize { id: u32, triangle_count: u32 },
    #[error("cluster {0} has invalid bounds")]
    InvalidBounds(u32),
    #[error("source mesh has more triangles than the schema can address")]
    TriangleCountOverflow,
    #[error("source triangle {triangle} references absent vertex {vertex}")]
    VertexOutOfRange { triangle: u32, vertex: u32 },
    #[error("source vertex {0} has a non-finite position")]
    NonFinitePosition(u32),
}
