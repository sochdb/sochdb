// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Raw float32 binary file reader
//!
//! Format: N × D × 4 bytes of row-major float32 data
//! 
//! This is the simplest and fastest format - just raw bytes.
//! Use when you control the data pipeline and don't need metadata.
//!
//! ## Companion Files (optional)
//!
//! - `vectors.f32` - Main vector data
//! - `meta.json` - Metadata: `{"n": 10000, "dim": 768, "metric": "cosine"}`
//! - `ids.u64` - Optional ID file (N × 8 bytes of u64)

use crate::error::ToolsError;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

/// Zero-copy raw float32 file reader
pub struct RawF32Reader {
    mmap: Mmap,
    num_vectors: usize,
    dimension: usize,
}

impl RawF32Reader {
    /// Open a raw f32 file with known dimension
    pub fn open(path: &Path, dimension: usize) -> Result<Self, ToolsError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        Self::from_mmap(mmap, dimension)
    }
    
    /// Open with companion meta.json file
    pub fn open_with_meta(path: &Path) -> Result<Self, ToolsError> {
        // Look for meta.json in same directory
        let meta_path = path.with_extension("json");
        let alt_meta_path = path.parent()
            .map(|p| p.join("meta.json"))
            .ok_or_else(|| ToolsError::InvalidArgument("Invalid path".to_string()))?;
        
        let meta = if meta_path.exists() {
            Self::read_meta(&meta_path)?
        } else if alt_meta_path.exists() {
            Self::read_meta(&alt_meta_path)?
        } else {
            return Err(ToolsError::InvalidArgument(
                "No meta.json found. Use open() with explicit dimension".to_string()
            ));
        };
        
        Self::open(path, meta.dim)
    }
    
    /// Create reader from memory-mapped data
    pub fn from_mmap(mmap: Mmap, dimension: usize) -> Result<Self, ToolsError> {
        if dimension == 0 {
            return Err(ToolsError::InvalidArgument("Dimension cannot be 0".to_string()));
        }
        
        let n_floats = mmap.len() / 4;
        let num_vectors = n_floats / dimension;
        
        if num_vectors == 0 {
            return Err(ToolsError::FileTooSmall {
                needed: dimension * 4,
                actual: mmap.len(),
            });
        }
        
        // Check alignment
        if (mmap.as_ptr() as usize) % 4 != 0 {
            // This is rare with mmap but we should warn
            tracing::warn!("Memory map not 4-byte aligned, may impact performance");
        }
        
        Ok(Self {
            mmap,
            num_vectors,
            dimension,
        })
    }
    
    /// Read companion meta.json file
    fn read_meta(path: &Path) -> Result<RawMeta, ToolsError> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| ToolsError::Serialization(e.to_string()))
    }
    
    /// Get the number of vectors
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
    
    /// Get the vector dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Get a slice of all vectors (zero-copy)
    pub fn vectors(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr() as *const f32,
                self.num_vectors * self.dimension,
            )
        }
    }
    
    /// Get metadata
    pub fn meta(&self) -> super::VectorMeta {
        super::VectorMeta {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            metric: None,
            format: "raw_f32".to_string(),
        }
    }
}

/// Metadata for raw f32 format
#[derive(Debug, Serialize, Deserialize)]
struct RawMeta {
    /// Number of vectors
    n: usize,
    /// Dimension
    dim: usize,
    /// Distance metric
    #[serde(default)]
    metric: Option<String>,
}

/// Write vectors to raw f32 format with companion meta.json
pub fn write_raw_f32(
    path: &Path,
    vectors: &[f32],
    dimension: usize,
    metric: Option<&str>,
) -> Result<(), ToolsError> {
    use std::io::Write;
    
    let num_vectors = vectors.len() / dimension;
    
    // Write vectors
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4)
    };
    std::fs::write(path, bytes)?;
    
    // Write meta.json
    let meta = RawMeta {
        n: num_vectors,
        dim: dimension,
        metric: metric.map(|s| s.to_string()),
    };
    let meta_path = path.with_extension("json");
    let mut file = File::create(meta_path)?;
    write!(file, "{}", serde_json::to_string_pretty(&meta)
        .map_err(|e| ToolsError::Serialization(e.to_string()))?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_raw_reader() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.f32");
        
        // Create test data
        let n = 100;
        let d = 768;
        let vectors: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.01).collect();
        
        write_raw_f32(&path, &vectors, d, Some("cosine")).unwrap();
        
        let reader = RawF32Reader::open(&path, d).unwrap();
        assert_eq!(reader.num_vectors(), n);
        assert_eq!(reader.dimension(), d);
        assert_eq!(reader.vectors().len(), n * d);
    }
}
