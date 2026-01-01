// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Error types for toondb-tools

use thiserror::Error;

/// Errors that can occur during bulk operations
#[derive(Error, Debug)]
pub enum ToolsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("File too small: need {needed} bytes, got {actual}")]
    FileTooSmall { needed: usize, actual: usize },
    
    #[error("Invalid NPY header: {0}")]
    InvalidNpyHeader(String),
    
    #[error("HNSW error: {0}")]
    Hnsw(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<ToolsError> for std::io::Error {
    fn from(err: ToolsError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
    }
}
