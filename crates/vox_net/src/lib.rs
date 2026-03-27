use thiserror::Error;

pub mod crdt;
pub mod lobby;
pub mod replication;
pub mod transport;
pub use replication::{EntityDelta, NetMessage, PlayerAction, ReplicationClient, ReplicationServer};
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
