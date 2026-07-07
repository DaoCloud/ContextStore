//! Unified error types

use thiserror::Error;

pub type Result<T> = std::result::Result<T, KVError>;

#[derive(Debug, Error)]
pub enum KVError {
    #[error("config error: {0}")]
    Config(String),

    #[error("Key not found: {0}")]
    NotFound(String),

    #[error("Key already exists: {0}")]
    AlreadyExists(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("metadata error: {0}")]
    Metadata(String),

    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl From<KVError> for tonic::Status {
    fn from(e: KVError) -> Self {
        match e {
            KVError::NotFound(m) => tonic::Status::not_found(m),
            KVError::AlreadyExists(m) => tonic::Status::already_exists(m),
            KVError::InvalidArgument(m) => tonic::Status::invalid_argument(m),
            other => tonic::Status::internal(other.to_string()),
        }
    }
}
