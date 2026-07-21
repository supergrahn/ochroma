use vox_render::mega_geometry::{
    ForbiddenGeometryCpuActivity, GeometryCpuOwnershipCounters, GeometryHostServiceKind,
    OPAQUE_GEOMETRY_HOST_PACKET_SCHEMA, OpaqueGeometryHostPacket, OpaqueGeometryHostPacketError,
    OpaqueGeometryHostPacketHeader,
};

#[test]
fn default_cpu_contract_is_verdict_eligible() {
    let counters = GeometryCpuOwnershipCounters::default();
    let snapshot = counters.snapshot();

    assert_eq!(snapshot.forbidden_total(), 0);
    assert!(snapshot.verdict_eligible());
}

#[test]
fn every_forbidden_cpu_activity_suppresses_the_verdict() {
    for activity in ForbiddenGeometryCpuActivity::ALL {
        let counters = GeometryCpuOwnershipCounters::default();
        counters.record_forbidden(activity);
        let snapshot = counters.snapshot();

        assert_eq!(snapshot.forbidden_count(activity), 1, "{activity:?}");
        assert_eq!(snapshot.forbidden_total(), 1, "{activity:?}");
        assert!(!snapshot.verdict_eligible(), "{activity:?}");
    }
}

#[test]
fn allowed_host_services_are_measured_without_becoming_cpu_decisions() {
    let counters = GeometryCpuOwnershipCounters::default();
    counters.record_host_service(
        GeometryHostServiceKind::NativeAccelerationSubmission,
        41_000,
    );
    counters.record_host_service(GeometryHostServiceKind::OpaquePageTransport, 73_000);
    let snapshot = counters.snapshot();

    assert_eq!(
        snapshot.host_service_calls(GeometryHostServiceKind::NativeAccelerationSubmission),
        1
    );
    assert_eq!(
        snapshot.host_service_ns(GeometryHostServiceKind::NativeAccelerationSubmission),
        41_000
    );
    assert_eq!(
        snapshot.host_service_calls(GeometryHostServiceKind::OpaquePageTransport),
        1
    );
    assert_eq!(snapshot.forbidden_total(), 0);
    assert!(snapshot.verdict_eligible());
}

#[test]
fn bridge_delta_folds_reflect_spectra_probe_activity() {
    // The live-path bridge (`bridge_geometry_cpu_ownership`) folds the positive
    // per-activity / per-service delta of Spectra's probe into these engine-side
    // counters via `add_forbidden` / `add_host_service`. Exercise those exact
    // primitives: a forbidden fold suppresses the verdict, a call/ns fold is
    // measured, and a zero-call fold is a no-op (never a spurious 0-call event).
    let counters = GeometryCpuOwnershipCounters::default();
    counters.add_forbidden(ForbiddenGeometryCpuActivity::DirtyPartitionCompaction, 3);
    counters.add_host_service(GeometryHostServiceKind::OpaquePageTransport, 2, 5_000);
    counters.add_host_service(GeometryHostServiceKind::NativeAccelerationSubmission, 0, 999);
    let snapshot = counters.snapshot();

    assert_eq!(
        snapshot.forbidden_count(ForbiddenGeometryCpuActivity::DirtyPartitionCompaction),
        3
    );
    assert_eq!(snapshot.forbidden_total(), 3);
    assert!(!snapshot.verdict_eligible());
    assert_eq!(
        snapshot.host_service_calls(GeometryHostServiceKind::OpaquePageTransport),
        2
    );
    assert_eq!(
        snapshot.host_service_ns(GeometryHostServiceKind::OpaquePageTransport),
        5_000
    );
    // calls == 0 is a no-op: no phantom native submission is recorded.
    assert_eq!(
        snapshot.host_service_calls(GeometryHostServiceKind::NativeAccelerationSubmission),
        0
    );
    assert_eq!(
        snapshot.host_service_ns(GeometryHostServiceKind::NativeAccelerationSubmission),
        0
    );
}

#[test]
fn host_packet_is_bounded_and_opaque() {
    let bytes = [7_u8; 32];
    let header = OpaqueGeometryHostPacketHeader::new(32, 64, 2, 4, 9, [3; 32]).unwrap();
    let packet = OpaqueGeometryHostPacket::new(header, &bytes).unwrap();

    assert_eq!(packet.header().schema(), OPAQUE_GEOMETRY_HOST_PACKET_SCHEMA);
    assert_eq!(packet.header().record_count(), 2);
    assert_eq!(packet.header().record_capacity(), 4);
    assert_eq!(packet.header().generation(), 9);
    assert_eq!(packet.opaque_bytes(), bytes);
}

#[test]
fn host_packet_rejects_capacity_and_length_mismatches() {
    assert_eq!(
        OpaqueGeometryHostPacketHeader::new(65, 64, 2, 4, 9, [3; 32]),
        Err(OpaqueGeometryHostPacketError::ByteCapacityExceeded {
            byte_len: 65,
            capacity_bytes: 64,
        })
    );
    assert_eq!(
        OpaqueGeometryHostPacketHeader::new(32, 64, 5, 4, 9, [3; 32]),
        Err(OpaqueGeometryHostPacketError::RecordCapacityExceeded {
            record_count: 5,
            record_capacity: 4,
        })
    );

    let header = OpaqueGeometryHostPacketHeader::new(32, 64, 2, 4, 9, [3; 32]).unwrap();
    assert_eq!(
        OpaqueGeometryHostPacket::new(header, &[0; 31]),
        Err(OpaqueGeometryHostPacketError::PayloadLengthMismatch {
            header_bytes: 32,
            payload_bytes: 31,
        })
    );
}
