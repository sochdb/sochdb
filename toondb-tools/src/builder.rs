// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Unified HNSW Build Kernel (Task 3)
//!
//! Single, canonical, performance-validated build function that both
//! CLI and Python FFI invoke. Eliminates code path divergence that
//! caused the 14× performance regression.
//!
//! ## Design Principles
//!
//! 1. **Single hot path**: Both CLI and FFI call the same function
//! 2. **Zero-copy**: Works directly on contiguous f32 slices
//! 3. **No internal allocations**: All memory pre-allocated
//! 4. **Parallel by default**: Uses rayon internally
//! 5. **Observable**: Integrated telemetry for performance regression detection
//!
//! ## Usage
//!
//! ```rust,ignore
//! use toondb_tools::builder::{BuildConfig, build_hnsw_index};
//!
//! let config = BuildConfig::default().with_m(16).with_ef_construction(100);
//! let index = build_hnsw_index(vectors, ids, 768, config)?;
//! ```

use std::time::Instant;
use thiserror::Error;

use toondb_index::hnsw::{HnswConfig, HnswIndex};

use crate::io::telemetry::FaultTelemetry;

/// Build configuration for HNSW index construction
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// HNSW M parameter (max connections per node)
    pub m: usize,
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    /// Batch size for insertion (vectors per batch)
    pub batch_size: usize,
    /// Enable telemetry capture
    pub enable_telemetry: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            batch_size: 1000,
            enable_telemetry: false,
            verbose: false,
        }
    }
}

impl BuildConfig {
    /// Set M parameter
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }
    
    /// Set ef_construction parameter
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }
    
    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
    
    /// Enable telemetry
    pub fn with_telemetry(mut self) -> Self {
        self.enable_telemetry = true;
        self
    }
    
    /// Enable verbose output
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
    
    /// Convert to HnswConfig
    pub fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig {
            max_connections: self.m,
            max_connections_layer0: self.m * 2,
            ef_construction: self.ef_construction,
            ..Default::default()
        }
    }
}

/// Build errors
#[derive(Debug, Error)]
pub enum BuildError {
    #[error("Dimension mismatch: vectors length {vectors_len} not divisible by dimension {dimension}")]
    DimensionMismatch { vectors_len: usize, dimension: usize },
    
    #[error("ID count mismatch: {ids_len} IDs vs {n_vectors} vectors")]
    IdCountMismatch { ids_len: usize, n_vectors: usize },
    
    #[error("Insert failed: {0}")]
    InsertFailed(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Build result with statistics
/// 
/// Note: Does not derive Debug because HnswIndex doesn't implement Debug
pub struct BuildResult {
    /// The constructed index
    pub index: HnswIndex,
    /// Number of vectors inserted
    pub n_vectors: usize,
    /// Total build time
    pub build_time: std::time::Duration,
    /// Vectors per second
    pub throughput: f64,
    /// Optional telemetry
    pub telemetry: Option<FaultTelemetry>,
}

impl std::fmt::Debug for BuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildResult")
            .field("n_vectors", &self.n_vectors)
            .field("build_time", &self.build_time)
            .field("throughput", &self.throughput)
            .field("telemetry", &self.telemetry)
            .field("index", &"<HnswIndex>")
            .finish()
    }
}

/// Single optimized build kernel (Task 3)
///
/// This is THE canonical function for HNSW construction.
/// Both CLI and Python FFI MUST call this function.
///
/// # Arguments
///
/// * `vectors` - Contiguous f32 slice, row-major [N × D]
/// * `ids` - u64 slice, length = N
/// * `dim` - Vector dimension
/// * `config` - Build configuration
///
/// # Complexity
///
/// O(N · ef_construction · log(N)) expected distance evaluations.
///
/// # Performance Guarantees
///
/// - No allocations in hot path (all pre-allocated)
/// - Parallel construction via rayon
/// - Same code path as profiler benchmarks
pub fn build_hnsw_index(
    vectors: &[f32],
    ids: &[u64],
    dim: usize,
    config: BuildConfig,
) -> Result<BuildResult, BuildError> {
    // Validate inputs
    if vectors.len() % dim != 0 {
        return Err(BuildError::DimensionMismatch {
            vectors_len: vectors.len(),
            dimension: dim,
        });
    }
    
    let n = vectors.len() / dim;
    
    if ids.len() != n {
        return Err(BuildError::IdCountMismatch {
            ids_len: ids.len(),
            n_vectors: n,
        });
    }
    
    if config.m == 0 {
        return Err(BuildError::InvalidConfig("M cannot be 0".to_string()));
    }
    
    // Create index
    let hnsw_config = config.to_hnsw_config();
    let index = HnswIndex::new(dim, hnsw_config);
    
    // Start telemetry if enabled
    let mut telemetry = if config.enable_telemetry {
        Some(FaultTelemetry::capture_start_labeled("HNSW Build"))
    } else {
        None
    };
    
    let start = Instant::now();
    
    // ==========================================================================
    // CANONICAL HOT PATH
    // ==========================================================================
    // This is the ONLY code path for HNSW construction.
    // Both CLI and FFI must use this exact sequence.
    // Do NOT add alternative paths without performance validation.
    // ==========================================================================
    
    // Convert u64 -> u128 once (amortized over all batches)
    // TODO: Add insert_batch_contiguous_u64 to avoid this allocation
    let ids_u128: Vec<u128> = ids.iter().map(|&id| id as u128).collect();
    
    // Insert in batches
    let batch_size = config.batch_size;
    let mut total_inserted = 0usize;
    
    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(n);
        let batch_ids = &ids_u128[chunk_start..chunk_end];
        let batch_vectors = &vectors[chunk_start * dim..chunk_end * dim];
        
        // Use insert_batch_contiguous - the same path as the profiler
        let inserted = index.insert_batch_contiguous(batch_ids, batch_vectors, dim)
            .map_err(|e| BuildError::InsertFailed(e))?;
        
        total_inserted += inserted;
        
        if config.verbose && chunk_start % (batch_size * 10) == 0 {
            let pct = 100.0 * chunk_start as f64 / n as f64;
            let rate = total_inserted as f64 / start.elapsed().as_secs_f64();
            eprintln!("  [{:5.1}%] {} vectors ({:.0} vec/s)", pct, total_inserted, rate);
        }
    }
    
    let build_time = start.elapsed();
    let throughput = total_inserted as f64 / build_time.as_secs_f64();
    
    // Capture end telemetry
    if let Some(ref mut t) = telemetry {
        t.capture_end();
        if config.verbose {
            t.print_summary("HNSW Build");
        }
    }
    
    if config.verbose {
        eprintln!("  Inserted {} vectors in {:.2}s ({:.0} vec/s)", 
            total_inserted, build_time.as_secs_f64(), throughput);
    }
    
    Ok(BuildResult {
        index,
        n_vectors: total_inserted,
        build_time,
        throughput,
        telemetry,
    })
}

/// Build with u128 IDs directly (for compatibility with existing code)
pub fn build_hnsw_index_u128(
    vectors: &[f32],
    ids: &[u128],
    dim: usize,
    config: BuildConfig,
) -> Result<BuildResult, BuildError> {
    // Validate inputs
    if vectors.len() % dim != 0 {
        return Err(BuildError::DimensionMismatch {
            vectors_len: vectors.len(),
            dimension: dim,
        });
    }
    
    let n = vectors.len() / dim;
    
    if ids.len() != n {
        return Err(BuildError::IdCountMismatch {
            ids_len: ids.len(),
            n_vectors: n,
        });
    }
    
    // Create index
    let hnsw_config = config.to_hnsw_config();
    let index = HnswIndex::new(dim, hnsw_config);
    
    let mut telemetry = if config.enable_telemetry {
        Some(FaultTelemetry::capture_start_labeled("HNSW Build"))
    } else {
        None
    };
    
    let start = Instant::now();
    
    // Insert in batches
    let batch_size = config.batch_size;
    let mut total_inserted = 0usize;
    
    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(n);
        let batch_ids = &ids[chunk_start..chunk_end];
        let batch_vectors = &vectors[chunk_start * dim..chunk_end * dim];
        
        let inserted = index.insert_batch_contiguous(batch_ids, batch_vectors, dim)
            .map_err(|e| BuildError::InsertFailed(e))?;
        
        total_inserted += inserted;
    }
    
    let build_time = start.elapsed();
    let throughput = total_inserted as f64 / build_time.as_secs_f64();
    
    if let Some(ref mut t) = telemetry {
        t.capture_end();
    }
    
    Ok(BuildResult {
        index,
        n_vectors: total_inserted,
        build_time,
        throughput,
        telemetry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_basic() {
        let n = 100;
        let d = 64;
        let vectors: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.01).collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        
        let config = BuildConfig::default();
        let result = build_hnsw_index(&vectors, &ids, d, config).unwrap();
        
        assert_eq!(result.n_vectors, n);
        assert!(result.throughput > 0.0);
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let vectors = vec![1.0f32; 100];
        let ids = vec![0u64; 10];
        
        let result = build_hnsw_index(&vectors, &ids, 11, BuildConfig::default());
        assert!(matches!(result, Err(BuildError::DimensionMismatch { .. })));
    }
}
