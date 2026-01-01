//! Error types for the vector search engine.

use thiserror::Error;

/// Main error type for the engine
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Segment error: {0}")]
    Segment(String),

    #[error("Invalid segment magic: expected SVSEGM")]
    InvalidMagic,

    #[error("Unsupported segment version: {0}")]
    UnsupportedVersion(u32),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },

    #[error("Vector ID out of range: {0}")]
    VectorIdOutOfRange(u32),

    #[error("Segment not found: {0}")]
    SegmentNotFound(u64),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Catalog error: {0}")]
    Catalog(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Kernel dispatch error: {0}")]
    KernelDispatch(String),

    #[error("Compaction error: {0}")]
    Compaction(String),

    #[error("Filter error: {0}")]
    Filter(String),

    #[error("Index is empty")]
    EmptyIndex,
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

impl From<bincode::Error> for Error {
    fn from(e: bincode::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}
