//! Game-agnostic MegaGeometry runtime contracts.
//!
//! Geometry decisions belong to resident GPU work. The CPU may submit a fixed
//! native acceleration-structure call or transport a bounded opaque page packet,
//! but it must not walk scene geometry, select LOD, schedule pages, or inspect a
//! GPU-written work count to decide what happens next.

use std::sync::atomic::{AtomicU64, Ordering};

pub const OPAQUE_GEOMETRY_HOST_PACKET_SCHEMA: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum ForbiddenGeometryCpuActivity {
    RenderThreadInstanceVisit = 0,
    RenderThreadPrototypeVisit = 1,
    RenderThreadVisibilityNodeVisit = 2,
    RenderThreadClusterVisit = 3,
    RenderThreadPageVisit = 4,
    ConstructionEvaluation = 5,
    LodOrToleranceDecision = 6,
    PagePriorityDecision = 7,
    EvictionDecision = 8,
    BlockingReadback = 9,
    FrameAllocation = 10,
    /// §6.0 crime: sorting/deduplicating dirty partitions on the CPU. The
    /// hierarchical-IAS compaction is GPU-owned; this mirrors Spectra's
    /// `geometry_cpu_contract::ForbiddenGeometryCpuActivity::DirtyPartitionCompaction`
    /// (same discriminant) so the live-path counter bridge is index-aligned and
    /// lossless.
    DirtyPartitionCompaction = 11,
}

impl ForbiddenGeometryCpuActivity {
    pub const ALL: [Self; 12] = [
        Self::RenderThreadInstanceVisit,
        Self::RenderThreadPrototypeVisit,
        Self::RenderThreadVisibilityNodeVisit,
        Self::RenderThreadClusterVisit,
        Self::RenderThreadPageVisit,
        Self::ConstructionEvaluation,
        Self::LodOrToleranceDecision,
        Self::PagePriorityDecision,
        Self::EvictionDecision,
        Self::BlockingReadback,
        Self::FrameAllocation,
        Self::DirtyPartitionCompaction,
    ];

    const COUNT: usize = Self::ALL.len();

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryHostServiceKind {
    NativeAccelerationSubmission,
    OpaquePageTransport,
}

#[derive(Debug)]
pub struct GeometryCpuOwnershipCounters {
    forbidden: [AtomicU64; ForbiddenGeometryCpuActivity::COUNT],
    native_submission_calls: AtomicU64,
    native_submission_ns: AtomicU64,
    page_transport_calls: AtomicU64,
    page_transport_ns: AtomicU64,
}

impl Default for GeometryCpuOwnershipCounters {
    fn default() -> Self {
        Self {
            forbidden: std::array::from_fn(|_| AtomicU64::new(0)),
            native_submission_calls: AtomicU64::new(0),
            native_submission_ns: AtomicU64::new(0),
            page_transport_calls: AtomicU64::new(0),
            page_transport_ns: AtomicU64::new(0),
        }
    }
}

impl GeometryCpuOwnershipCounters {
    pub fn record_forbidden(&self, activity: ForbiddenGeometryCpuActivity) {
        self.forbidden[activity.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_host_service(&self, service: GeometryHostServiceKind, elapsed_ns: u64) {
        self.add_host_service(service, 1, elapsed_ns);
    }

    /// Fold an already-aggregated delta of forbidden activity into the witness.
    /// Used by the live-path bridge that mirrors Spectra's probed
    /// `geometry_cpu_contract` counters into this engine-side witness: the bridge
    /// computes the positive per-activity delta since the last frame and adds it
    /// here, so `record_forbidden` semantics are preserved without replaying the
    /// call one-at-a-time. A no-op when `count == 0`.
    pub fn add_forbidden(&self, activity: ForbiddenGeometryCpuActivity, count: u64) {
        if count != 0 {
            self.forbidden[activity.index()].fetch_add(count, Ordering::Relaxed);
        }
    }

    /// Fold an already-aggregated delta of host-service calls/time into the
    /// witness (the call/time counterpart of [`add_forbidden`] for the live-path
    /// bridge). A no-op when `calls == 0`.
    pub fn add_host_service(&self, service: GeometryHostServiceKind, calls: u64, elapsed_ns: u64) {
        if calls == 0 {
            return;
        }
        let (call_ctr, time_ctr) = match service {
            GeometryHostServiceKind::NativeAccelerationSubmission => {
                (&self.native_submission_calls, &self.native_submission_ns)
            }
            GeometryHostServiceKind::OpaquePageTransport => {
                (&self.page_transport_calls, &self.page_transport_ns)
            }
        };
        call_ctr.fetch_add(calls, Ordering::Relaxed);
        time_ctr.fetch_add(elapsed_ns, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> GeometryCpuOwnershipSnapshot {
        GeometryCpuOwnershipSnapshot {
            forbidden: std::array::from_fn(|index| self.forbidden[index].load(Ordering::Relaxed)),
            native_submission_calls: self.native_submission_calls.load(Ordering::Relaxed),
            native_submission_ns: self.native_submission_ns.load(Ordering::Relaxed),
            page_transport_calls: self.page_transport_calls.load(Ordering::Relaxed),
            page_transport_ns: self.page_transport_ns.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryCpuOwnershipSnapshot {
    forbidden: [u64; ForbiddenGeometryCpuActivity::COUNT],
    native_submission_calls: u64,
    native_submission_ns: u64,
    page_transport_calls: u64,
    page_transport_ns: u64,
}

impl GeometryCpuOwnershipSnapshot {
    pub fn forbidden_count(&self, activity: ForbiddenGeometryCpuActivity) -> u64 {
        self.forbidden[activity.index()]
    }

    pub fn forbidden_total(&self) -> u64 {
        self.forbidden.iter().sum()
    }

    pub fn verdict_eligible(&self) -> bool {
        self.forbidden_total() == 0
    }

    pub fn host_service_calls(&self, service: GeometryHostServiceKind) -> u64 {
        match service {
            GeometryHostServiceKind::NativeAccelerationSubmission => self.native_submission_calls,
            GeometryHostServiceKind::OpaquePageTransport => self.page_transport_calls,
        }
    }

    pub fn host_service_ns(&self, service: GeometryHostServiceKind) -> u64 {
        match service {
            GeometryHostServiceKind::NativeAccelerationSubmission => self.native_submission_ns,
            GeometryHostServiceKind::OpaquePageTransport => self.page_transport_ns,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpaqueGeometryHostPacketHeader {
    schema: u32,
    byte_len: u32,
    capacity_bytes: u32,
    record_count: u32,
    record_capacity: u32,
    generation: u32,
    content_hash: [u8; 32],
}

impl OpaqueGeometryHostPacketHeader {
    pub fn new(
        byte_len: u32,
        capacity_bytes: u32,
        record_count: u32,
        record_capacity: u32,
        generation: u32,
        content_hash: [u8; 32],
    ) -> Result<Self, OpaqueGeometryHostPacketError> {
        let header = Self {
            schema: OPAQUE_GEOMETRY_HOST_PACKET_SCHEMA,
            byte_len,
            capacity_bytes,
            record_count,
            record_capacity,
            generation,
            content_hash,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn schema(&self) -> u32 {
        self.schema
    }

    pub fn byte_len(&self) -> u32 {
        self.byte_len
    }

    pub fn capacity_bytes(&self) -> u32 {
        self.capacity_bytes
    }

    pub fn record_count(&self) -> u32 {
        self.record_count
    }

    pub fn record_capacity(&self) -> u32 {
        self.record_capacity
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn content_hash(&self) -> [u8; 32] {
        self.content_hash
    }

    fn validate(&self) -> Result<(), OpaqueGeometryHostPacketError> {
        if self.schema != OPAQUE_GEOMETRY_HOST_PACKET_SCHEMA {
            return Err(OpaqueGeometryHostPacketError::UnsupportedSchema {
                actual: self.schema,
            });
        }
        if self.byte_len > self.capacity_bytes {
            return Err(OpaqueGeometryHostPacketError::ByteCapacityExceeded {
                byte_len: self.byte_len,
                capacity_bytes: self.capacity_bytes,
            });
        }
        if self.record_count > self.record_capacity {
            return Err(OpaqueGeometryHostPacketError::RecordCapacityExceeded {
                record_count: self.record_count,
                record_capacity: self.record_capacity,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpaqueGeometryHostPacket<'a> {
    header: OpaqueGeometryHostPacketHeader,
    bytes: &'a [u8],
}

impl<'a> OpaqueGeometryHostPacket<'a> {
    pub fn new(
        header: OpaqueGeometryHostPacketHeader,
        bytes: &'a [u8],
    ) -> Result<Self, OpaqueGeometryHostPacketError> {
        header.validate()?;
        if bytes.len() != header.byte_len as usize {
            return Err(OpaqueGeometryHostPacketError::PayloadLengthMismatch {
                header_bytes: header.byte_len,
                payload_bytes: bytes.len(),
            });
        }
        Ok(Self { header, bytes })
    }

    pub fn header(&self) -> OpaqueGeometryHostPacketHeader {
        self.header
    }

    pub fn opaque_bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpaqueGeometryHostPacketError {
    UnsupportedSchema {
        actual: u32,
    },
    ByteCapacityExceeded {
        byte_len: u32,
        capacity_bytes: u32,
    },
    RecordCapacityExceeded {
        record_count: u32,
        record_capacity: u32,
    },
    PayloadLengthMismatch {
        header_bytes: u32,
        payload_bytes: usize,
    },
}
