// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! NumPy .npy file reader with zero-copy memory mapping
//!
//! NPY format specification:
//! - 6-byte magic: \x93NUMPY
//! - 2-byte version (major, minor)
//! - Header length: 2 bytes (v1) or 4 bytes (v2+)
//! - ASCII header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (N, D)}
//! - Data: N × D × sizeof(dtype) bytes
//!
//! This reader only supports:
//! - float32 dtype ('<f4' or '|f4')
//! - C-order (fortran_order: False)
//! - 2D arrays (shape: (N, D))

use crate::error::ToolsError;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Zero-copy NumPy .npy file reader
pub struct NpyReader {
    mmap: Mmap,
    num_vectors: usize,
    dimension: usize,
    data_offset: usize,
}

impl NpyReader {
    /// Open a .npy file for reading
    pub fn open(path: &Path) -> Result<Self, ToolsError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        Self::from_mmap(mmap)
    }
    
    /// Create reader from memory-mapped data
    pub fn from_mmap(mmap: Mmap) -> Result<Self, ToolsError> {
        // Check minimum size
        if mmap.len() < 10 {
            return Err(ToolsError::InvalidNpyHeader("File too small".to_string()));
        }
        
        // Check magic bytes
        if &mmap[0..6] != b"\x93NUMPY" {
            return Err(ToolsError::InvalidNpyHeader(
                "Missing magic bytes '\\x93NUMPY'".to_string()
            ));
        }
        
        let version_major = mmap[6];
        let version_minor = mmap[7];
        
        // Parse header length based on version
        let (header_len, header_start) = match version_major {
            1 => {
                let len = u16::from_le_bytes([mmap[8], mmap[9]]) as usize;
                (len, 10)
            }
            2 | 3 => {
                if mmap.len() < 12 {
                    return Err(ToolsError::InvalidNpyHeader("File too small for v2 header".to_string()));
                }
                let len = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
                (len, 12)
            }
            _ => {
                return Err(ToolsError::InvalidNpyHeader(
                    format!("Unsupported version: {}.{}", version_major, version_minor)
                ));
            }
        };
        
        let data_offset = header_start + header_len;
        
        if mmap.len() < data_offset {
            return Err(ToolsError::InvalidNpyHeader("Header extends past file end".to_string()));
        }
        
        // Parse header
        let header = std::str::from_utf8(&mmap[header_start..data_offset])
            .map_err(|e| ToolsError::InvalidNpyHeader(format!("Invalid UTF-8: {}", e)))?;
        
        // Extract dtype
        let descr = Self::extract_field(header, "descr")?;
        if descr != "<f4" && descr != "|f4" && descr != "float32" {
            return Err(ToolsError::InvalidNpyHeader(
                format!("Only float32 dtype supported, got '{}'", descr)
            ));
        }
        
        // Check fortran order
        let fortran = Self::extract_field(header, "fortran_order")?;
        if fortran.to_lowercase() != "false" {
            return Err(ToolsError::InvalidNpyHeader(
                "Only C-order (fortran_order: False) supported".to_string()
            ));
        }
        
        // Extract shape
        let (num_vectors, dimension) = Self::extract_shape(header)?;
        
        // Validate data size
        let expected_bytes = num_vectors * dimension * 4;
        let actual_bytes = mmap.len() - data_offset;
        if actual_bytes < expected_bytes {
            return Err(ToolsError::FileTooSmall {
                needed: expected_bytes,
                actual: actual_bytes,
            });
        }
        
        Ok(Self {
            mmap,
            num_vectors,
            dimension,
            data_offset,
        })
    }
    
    /// Extract a field value from the header dict
    fn extract_field(header: &str, field: &str) -> Result<String, ToolsError> {
        // Look for 'field': value or "field": value
        let patterns = [
            format!("'{}': ", field),
            format!("'{}' : ", field),
            format!("\"{}\":", field),
            format!("\"{}\": ", field),
        ];
        
        for pattern in &patterns {
            if let Some(start) = header.find(pattern.as_str()) {
                let value_start = start + pattern.len();
                let rest = &header[value_start..];
                
                // Find end of value (comma or closing brace)
                let end = rest.find(|c: char| c == ',' || c == '}' || c == '\n')
                    .unwrap_or(rest.len());
                
                let value = rest[..end].trim().trim_matches('\'').trim_matches('"');
                return Ok(value.to_string());
            }
        }
        
        Err(ToolsError::InvalidNpyHeader(
            format!("Could not find '{}' in header", field)
        ))
    }
    
    /// Extract shape tuple from header
    fn extract_shape(header: &str) -> Result<(usize, usize), ToolsError> {
        let shape_start = header.find("'shape':")
            .or_else(|| header.find("\"shape\":"))
            .ok_or_else(|| ToolsError::InvalidNpyHeader("Could not find 'shape'".to_string()))?;
        
        let rest = &header[shape_start..];
        let paren_start = rest.find('(')
            .ok_or_else(|| ToolsError::InvalidNpyHeader("Could not find shape tuple".to_string()))?;
        let paren_end = rest.find(')')
            .ok_or_else(|| ToolsError::InvalidNpyHeader("Could not find shape tuple end".to_string()))?;
        
        let shape_str = &rest[paren_start + 1..paren_end];
        let dims: Vec<usize> = shape_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        
        match dims.len() {
            1 => {
                // 1D array: (N,) - treat as N vectors of dimension 1
                Ok((dims[0], 1))
            }
            2 => Ok((dims[0], dims[1])),
            _ => Err(ToolsError::InvalidNpyHeader(
                format!("Expected 1D or 2D array, got {} dimensions", dims.len())
            )),
        }
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
                self.mmap[self.data_offset..].as_ptr() as *const f32,
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
            format: "npy".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    /// Create a minimal valid NPY file
    fn create_test_npy(n: usize, d: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        
        // Magic
        file.write_all(b"\x93NUMPY").unwrap();
        // Version 1.0
        file.write_all(&[1, 0]).unwrap();
        
        // Header
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            n, d
        );
        // Pad to 64-byte boundary
        let padding = 64 - ((10 + header.len()) % 64);
        let padded_header = format!("{}{}", header, " ".repeat(padding - 1));
        
        // Header length
        let header_len = padded_header.len() as u16;
        file.write_all(&header_len.to_le_bytes()).unwrap();
        // Header content + newline
        file.write_all(padded_header.as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
        
        // Data: n*d floats
        let data: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.01).collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        file.write_all(bytes).unwrap();
        
        file
    }
    
    #[test]
    fn test_npy_reader() {
        let file = create_test_npy(100, 768);
        let reader = NpyReader::open(file.path()).unwrap();
        
        assert_eq!(reader.num_vectors(), 100);
        assert_eq!(reader.dimension(), 768);
        assert_eq!(reader.vectors().len(), 100 * 768);
    }
}
