use thiserror::Error;

pub mod crdt;
pub mod lobby;
pub mod net_walk_demo;
pub mod quic_transport;
pub mod replication;
pub mod replication_loop;
pub mod replication_packet;
pub mod replication_system;
pub mod rollback;
pub mod spectral_relevance;
pub mod splat_replication;
pub mod transport;
pub mod world_hosting;
pub mod world_replication;
pub use net_walk_demo::{
    RollbackQuicConfig, RollbackQuicReport, WalkDemoConfig, WalkDemoError, WalkDemoReport,
    run_loopback_walk_demo, run_rollback_quic_demo,
};
pub use quic_transport::{
    QuicClient, QuicConnection, QuicServer, QuicTransport, TransportError, TransportRole,
};
pub use replication::{
    CommandPayload, EntityDelta, NetMessage, ReplicationClient, ReplicationServer,
};
pub use replication_packet::{PlayerStatePacket, ReplicationPacket};
pub use transport::{GameClient, GameServer};

#[derive(Debug, Error)]
pub enum NetError {
    #[error("connection refused")]
    ConnectionRefused,
    #[error("timeout")]
    Timeout,
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("deserialization error: {0}")]
    Deserialization(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
