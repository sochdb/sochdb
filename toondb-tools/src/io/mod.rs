// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Zero-copy I/O for vector data
//!
//! This module provides memory-mapped readers for various vector formats,
//! enabling bulk ingestion without per-vector allocations.

pub mod npy;
pub mod raw;
pub mod residency;
pub mod telemetry;
pub mod direct;

pub use npy::NpyReader;
pub use raw::RawF32Reader;
pub use raw::write_raw_f32;
pub use residency::{prefault_region, ensure_resident_for_hnsw, ResidencyStats, MemAdvice, madvise};
pub use telemetry::{FaultTelemetry, FaultStats, FaultGate, with_telemetry};
pub use direct::{load_vectors_bulk, load_npy_bulk, load_bulk, OwnedVectors};

use crate::error::ToolsError;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Supported vector input formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorFormat {
    /// Raw row-major float32 binary (N × D × 4 bytes)
    RawF32,
    /// NumPy .npy format (C-order, float32, 2D)
    Npy,
    /// ToonDB native format (future)
    ToonNative,
}

impl VectorFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "npy" => Some(Self::Npy),
            "bin" | "f32" | "raw" => Some(Self::RawF32),
            "toon" | "tdb" => Some(Self::ToonNative),
            _ => None,
        }
    }
    
    /// Parse format from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "npy" | "numpy" => Some(Self::Npy),
            "raw" | "raw_f32" | "f32" | "bin" => Some(Self::RawF32),
            "toon" | "native" => Some(Self::ToonNative),
            _ => None,
        }
    }
}

/// Metadata for a vector file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMeta {
    /// Number of vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Optional: distance metric
    pub metric: Option<String>,
    /// File format
    pub format: String,
}

/// Unified vector reader that wraps different format readers
/// 
/// Using an enum instead of trait objects for:
/// - Better performance (no dynamic dispatch)
/// - Simpler API (no boxing)
/// - Support for `impl Iterator` in chunks()
pub enum VectorReader {
    Npy(NpyReader),
    RawF32(RawF32Reader),
}

impl VectorReader {
    /// Get the number of vectors
    pub fn num_vectors(&self) -> usize {
        match self {
            Self::Npy(r) => r.num_vectors(),
            Self::RawF32(r) => r.num_vectors(),
        }
    }
    
    /// Get the vector dimension
    pub fn dimension(&self) -> usize {
        match self {
            Self::Npy(r) => r.dimension(),
            Self::RawF32(r) => r.dimension(),
        }
    }
    
    /// Get a slice of all vectors (zero-copy)
    pub fn vectors(&self) -> &[f32] {
        match self {
            Self::Npy(r) => r.vectors(),
            Self::RawF32(r) => r.vectors(),
        }
    }
    
    /// Get metadata
    pub fn meta(&self) -> VectorMeta {
        match self {
            Self::Npy(r) => r.meta(),
            Self::RawF32(r) => r.meta(),
        }
    }
    
    /// Iterate over vectors by chunk
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[f32]> {
        self.vectors().chunks(chunk_size * self.dimension())
    }
}

/// Open a vector file with automatic format detection
pub fn open_vectors(
    path: &Path,
    format: Option<VectorFormat>,
    dimension: Option<usize>,
) -> Result<VectorReader, ToolsError> {
    let format = format
        .or_else(|| VectorFormat::from_extension(path))
        .ok_or_else(|| ToolsError::InvalidFormat(
            "Could not detect format from extension. Use --format".to_string()
        ))?;
    
    match format {
        VectorFormat::Npy => {
            let reader = NpyReader::open(path)?;
            // Validate dimension if provided
            if let Some(d) = dimension {
                if reader.dimension() != d {
                    return Err(ToolsError::DimensionMismatch {
                        expected: d,
                        actual: reader.dimension(),
                    });
                }
            }
            Ok(VectorReader::Npy(reader))
        }
        VectorFormat::RawF32 => {
            let dim = dimension.ok_or_else(|| ToolsError::InvalidArgument(
                "Dimension required for raw_f32 format".to_string()
            ))?;
            let reader = RawF32Reader::open(path, dim)?;
            Ok(VectorReader::RawF32(reader))
        }
        VectorFormat::ToonNative => {
            Err(ToolsError::InvalidFormat(
                "ToonNative format not yet implemented".to_string()
            ))
        }
    }
}

/// Open an optional ID file (raw u64)
pub fn open_ids(path: &Path) -> Result<(Mmap, usize), ToolsError> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let num_ids = mmap.len() / 8;
    Ok((mmap, num_ids))
}

/// Get u64 IDs from memory-mapped file
pub fn ids_from_mmap(mmap: &Mmap) -> &[u64] {
    unsafe {
        std::slice::from_raw_parts(
            mmap.as_ptr() as *const u64,
            mmap.len() / 8,
        )
    }
}
