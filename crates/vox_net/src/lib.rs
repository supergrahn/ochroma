use thiserror::Error;

pub mod replication;
pub use replication::{EntityDelta, ReplicationClient, ReplicationServer};

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
}
