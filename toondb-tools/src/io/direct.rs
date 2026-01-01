// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Direct Read for Bulk Builds (Task 4)
//!
//! Replaces mmap with direct read() for bulk index construction where
//! all N vectors are required. Eliminates mmap-related failure modes
//! (residency management, TLB pressure, page faults) for this use case.
//!
//! ## Why Not Mmap for Bulk Builds?
//!
//! For bulk construction where every vector is touched:
//! - Mmap's lazy loading provides no benefit
//! - Mmap introduces complexity (residency, TLB pressure)
//! - Direct read() is simpler and equally fast for one-time I/O
//! - Guarantees memory residency without prefaulting
//!
//! ## When to Use Each
//!
//! | Use Case | Best Approach |
//! |----------|---------------|
//! | Bulk build (all vectors) | Direct read() |
//! | Query-time (partial access) | Mmap |
//! | Streaming build (incremental) | Mmap with prefault |

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::ToolsError;

/// Owned vector data from direct read
pub struct OwnedVectors {
    /// Vector data (owned)
    pub data: Vec<f32>,
    /// Number of vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
}

impl OwnedVectors {
    /// Get vectors as a slice
    pub fn vectors(&self) -> &[f32] {
        &self.data
    }
    
    /// Get a single vector
    pub fn get(&self, idx: usize) -> Option<&[f32]> {
        if idx < self.num_vectors {
            let start = idx * self.dimension;
            Some(&self.data[start..start + self.dimension])
        } else {
            None
        }
    }
    
    /// Get memory usage
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Load vectors from raw f32 file via direct read (Task 4)
///
/// This function:
/// 1. Opens the file
/// 2. Allocates a Vec<f32> of exactly the right size
/// 3. Reads the entire file into the Vec
/// 4. Returns owned data that is guaranteed resident
///
/// # Performance
///
/// I/O cost: O(N · D · 4) bytes read sequentially
/// Allocation cost: O(N · D · 4) bytes
/// Total: ~Same as mmap + prefault, but simpler and more reliable
///
/// For 1M × 768D vectors (3GB):
/// - Read time: ~0.5s at 6 GB/s
/// - Allocation: instant (virtual memory)
/// - No page faults during subsequent use
pub fn load_vectors_bulk(
    path: &Path,
    dimension: usize,
) -> Result<OwnedVectors, ToolsError> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len() as usize;
    
    // Validate
    if file_size % 4 != 0 {
        return Err(ToolsError::InvalidFormat(
            "File size not a multiple of 4 bytes".to_string()
        ));
    }
    
    let n_floats = file_size / 4;
    if n_floats % dimension != 0 {
        return Err(ToolsError::InvalidFormat(format!(
            "File contains {} floats, not divisible by dimension {}",
            n_floats, dimension
        )));
    }
    
    let num_vectors = n_floats / dimension;
    
    // Allocate buffer
    let mut data = vec![0f32; n_floats];
    
    // Read entire file
    // Safety: Vec<f32> and &[u8] have compatible layouts for aligned data
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            file_size,
        )
    };
    
    file.read_exact(bytes)?;
    
    Ok(OwnedVectors {
        data,
        num_vectors,
        dimension,
    })
}

/// Load vectors with explicit count (for files without header)
pub fn load_vectors_bulk_exact(
    path: &Path,
    num_vectors: usize,
    dimension: usize,
) -> Result<OwnedVectors, ToolsError> {
    let mut file = File::open(path)?;
    let expected_size = num_vectors * dimension * 4;
    
    // Allocate buffer
    let mut data = vec![0f32; num_vectors * dimension];
    
    // Read exact amount
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            expected_size,
        )
    };
    
    file.read_exact(bytes)?;
    
    Ok(OwnedVectors {
        data,
        num_vectors,
        dimension,
    })
}

/// Load NPY format via direct read
pub fn load_npy_bulk(path: &Path) -> Result<OwnedVectors, ToolsError> {
    let mut file = File::open(path)?;
    
    // Read and parse NPY header
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    
    if &magic != b"\x93NUMPY" {
        return Err(ToolsError::InvalidFormat(
            "Invalid NPY magic bytes".to_string()
        ));
    }
    
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    
    let (header_len, header_size) = match version[0] {
        1 => {
            let mut len_bytes = [0u8; 2];
            file.read_exact(&mut len_bytes)?;
            (u16::from_le_bytes(len_bytes) as usize, 10)
        }
        2 | 3 => {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            (u32::from_le_bytes(len_bytes) as usize, 12)
        }
        _ => return Err(ToolsError::InvalidFormat(
            format!("Unsupported NPY version: {}.{}", version[0], version[1])
        )),
    };
    
    // Read header
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);
    
    // Parse shape from header (simple parsing)
    let (num_vectors, dimension) = parse_npy_shape(&header)?;
    
    // Seek to data start (might already be there, but be explicit)
    file.seek(SeekFrom::Start((header_size + header_len) as u64))?;
    
    // Read all vector data
    let n_floats = num_vectors * dimension;
    let mut data = vec![0f32; n_floats];
    
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            n_floats * 4,
        )
    };
    
    file.read_exact(bytes)?;
    
    Ok(OwnedVectors {
        data,
        num_vectors,
        dimension,
    })
}

/// Parse shape tuple from NPY header
fn parse_npy_shape(header: &str) -> Result<(usize, usize), ToolsError> {
    // Find 'shape': (N, D)
    let shape_start = header.find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| ToolsError::InvalidFormat("No shape in NPY header".to_string()))?;
    
    let paren_start = header[shape_start..].find('(')
        .ok_or_else(|| ToolsError::InvalidFormat("No shape tuple".to_string()))?
        + shape_start;
    
    let paren_end = header[paren_start..].find(')')
        .ok_or_else(|| ToolsError::InvalidFormat("Unclosed shape tuple".to_string()))?
        + paren_start;
    
    let shape_str = &header[paren_start + 1..paren_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    if dims.len() != 2 {
        return Err(ToolsError::InvalidFormat(format!(
            "Expected 2D shape, got {:?}", dims
        )));
    }
    
    Ok((dims[0], dims[1]))
}

/// Unified bulk loader with automatic format detection
pub fn load_bulk(path: &Path, dimension: Option<usize>) -> Result<OwnedVectors, ToolsError> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    match ext {
        "npy" => load_npy_bulk(path),
        "f32" | "bin" | "raw" => {
            let dim = dimension.ok_or_else(|| ToolsError::InvalidArgument(
                "Dimension required for raw format".to_string()
            ))?;
            load_vectors_bulk(path, dim)
        }
        _ => Err(ToolsError::InvalidFormat(format!(
            "Unknown extension: {}", ext
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;
    
    #[test]
    fn test_load_raw() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.f32");
        
        // Write test data
        let n = 100;
        let d = 64;
        let vectors: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4)
        };
        std::fs::write(&path, bytes).unwrap();
        
        // Load
        let owned = load_vectors_bulk(&path, d).unwrap();
        assert_eq!(owned.num_vectors, n);
        assert_eq!(owned.dimension, d);
        assert_eq!(owned.vectors()[0], 0.0);
        assert_eq!(owned.vectors()[d], d as f32);
    }
    
    #[test]
    fn test_load_npy() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.npy");
        
        // Write minimal NPY file
        let n = 10usize;
        let d = 4usize;
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {})}}", n, d);
        let padding = 64 - ((10 + header.len()) % 64);
        let padded_header = format!("{}{}", header, " ".repeat(padding - 1));
        
        let mut file = File::create(&path).unwrap();
        file.write_all(b"\x93NUMPY").unwrap();
        file.write_all(&[1, 0]).unwrap(); // version 1.0
        file.write_all(&(padded_header.len() as u16).to_le_bytes()).unwrap();
        file.write_all(padded_header.as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
        
        // Write data
        let vectors: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4)
        };
        file.write_all(bytes).unwrap();
        
        // Load
        let owned = load_npy_bulk(&path).unwrap();
        assert_eq!(owned.num_vectors, n);
        assert_eq!(owned.dimension, d);
    }
}
