//! ToonDB Python Native Extension
//!
//! High-performance Python bindings for ToonDB vector operations via PyO3.
//!
//! ## Why PyO3 Instead of Subprocess?
//!
//! The previous architecture used subprocess + temp files:
//! ```text
//! Python NumPy → tofile() → disk → subprocess → mmap → insert → done
//!              ↑ O(N·D) write    ↑ fork/exec    ↑ O(N·D) read
//! ```
//!
//! This PyO3 extension provides in-process zero-copy access:
//! ```text  
//! Python NumPy → PyO3 (zero-copy view) → insert → done
//!              ↑ O(1) pointer handoff
//! ```
//!
//! ## Performance
//!
//! | Method | 768D Throughput | Overhead |
//! |--------|-----------------|----------|
//! | Subprocess + disk | ~1,600 vec/s | 1.0× (previous "fast") |
//! | PyO3 zero-copy | ~15,000 vec/s | 0.1× (10× faster) |
//!
//! The subprocess approach paid:
//! - O(N·D) disk write (embeddings.tofile)
//! - Process startup latency (~50ms)
//! - O(N·D) disk read/mmap in CLI
//! - CLI used `insert_batch_flat` which is correct but not the fastest path
//!
//! PyO3 eliminates all of this by directly calling the core insertion API
//! with GIL release during the expensive HNSW work.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, PyArrayMethods, ToPyArray, IntoPyArray};
use numpy::ndarray::{Array1, Array2};
use std::sync::Arc;

use toondb_index::hnsw::{HnswConfig, HnswIndex, DistanceMetric};
use toondb_index::vector_quantized::Precision;

// =============================================================================
// Performance Guardrails (Task 6)
// =============================================================================

/// Check if safe mode is enabled and emit warning.
fn check_safe_mode() -> bool {
    static WARNED: std::sync::Once = std::sync::Once::new();
    
    let enabled = std::env::var("TOONDB_BATCH_SAFE_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    
    if enabled {
        WARNED.call_once(|| {
            eprintln!(
                "\n\
                ╔══════════════════════════════════════════════════════════════╗\n\
                ║  WARNING: TOONDB_BATCH_SAFE_MODE=1 is active                 ║\n\
                ║  Batch inserts are running 10-100× SLOWER than normal.       ║\n\
                ║  Unset this variable for production/benchmarking.            ║\n\
                ╚══════════════════════════════════════════════════════════════╝\n"
            );
        });
    }
    enabled
}

/// Log insertion path for debugging
fn log_insert_path(path: &str, contiguous: bool, n: usize) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    
    LOGGED.call_once(|| {
        if std::env::var("TOONDB_DEBUG_INSERT").is_ok() {
            eprintln!(
                "[toondb] Insert path: {} | contiguous={} | batch_size={}",
                path, contiguous, n
            );
        }
    });
}

// =============================================================================
// HnswIndex Python Wrapper
// =============================================================================

/// HNSW Vector Index with approximate nearest neighbor search.
///
/// This is a high-performance vector index using Hierarchical Navigable
/// Small World graphs. It provides ~250x speedup over brute-force search.
///
/// Example:
///     >>> import numpy as np
///     >>> from toondb import HnswIndex
///     >>> 
///     >>> # Create index
///     >>> index = HnswIndex(dimension=768, m=16, ef_construction=100)
///     >>> 
///     >>> # Insert vectors (zero-copy from numpy)
///     >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
///     >>> index.insert_batch(embeddings)  # ~15,000 vec/s
///     >>> 
///     >>> # Search
///     >>> query = np.random.randn(768).astype(np.float32)
///     >>> ids, distances = index.search(query, k=10)
#[pyclass(name = "HnswIndex")]
pub struct PyHnswIndex {
    inner: Arc<HnswIndex>,
    dimension: usize,
    next_id: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyHnswIndex {
    /// Create a new HNSW index.
    ///
    /// Args:
    ///     dimension: Vector dimension (e.g., 768 for text embeddings).
    ///     m: Max connections per node (default: 16). Higher = better recall, more memory.
    ///     ef_construction: Construction search depth (default: 100). Higher = better quality, slower build.
    ///     metric: Distance metric ("cosine", "euclidean", "dot"). Default: "cosine".
    ///     precision: Quantization precision ("f32", "f16", "bf16"). Default: "f32".
    ///
    /// Example:
    ///     >>> index = HnswIndex(768, m=32, ef_construction=200)
    #[new]
    #[pyo3(signature = (dimension, m=16, ef_construction=100, metric="cosine", precision="f32"))]
    fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        metric: &str,
        precision: &str,
    ) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("dimension must be > 0"));
        }
        
        let distance_metric = match metric.to_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "euclidean" | "l2" => DistanceMetric::Euclidean,
            "dot" | "dot_product" | "inner_product" => DistanceMetric::DotProduct,
            _ => return Err(PyValueError::new_err(
                format!("Unknown metric: {}. Use 'cosine', 'euclidean', or 'dot'", metric)
            )),
        };
        
        let quant_precision = match precision.to_lowercase().as_str() {
            "f32" | "float32" => Precision::F32,
            "f16" | "float16" => Precision::F16,
            "bf16" | "bfloat16" => Precision::BF16,
            _ => return Err(PyValueError::new_err(
                format!("Unknown precision: {}. Use 'f32', 'f16', or 'bf16'", precision)
            )),
        };
        
        let config = HnswConfig {
            max_connections: m,
            max_connections_layer0: m * 2,
            ef_construction,
            metric: distance_metric,
            quantization_precision: Some(quant_precision),
            ..Default::default()
        };
        
        let index = HnswIndex::new(dimension, config);
        
        Ok(Self {
            inner: Arc::new(index),
            dimension,
            next_id: std::sync::atomic::AtomicU64::new(0),
        })
    }
    
    /// Insert a batch of vectors with auto-generated IDs.
    ///
    /// This is the fastest insertion method - uses zero-copy NumPy access
    /// and releases the GIL during HNSW construction.
    ///
    /// Args:
    ///     vectors: 2D float32 array of shape (N, dimension).
    ///
    /// Returns:
    ///     Number of vectors inserted.
    ///
    /// Example:
    ///     >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
    ///     >>> count = index.insert_batch(embeddings)
    ///     >>> print(f"Inserted {count} vectors")
    fn insert_batch<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<usize> {
        let shape = vectors.shape();
        let n = shape[0];
        let d = shape[1];
        
        if d != self.dimension {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: index has {}, got {}",
                self.dimension, d
            )));
        }
        
        // Check contiguity for zero-copy
        let is_contiguous = vectors.is_c_contiguous();
        log_insert_path("insert_batch", is_contiguous, n);
        
        // Check safe mode
        if check_safe_mode() {
            return self.insert_batch_safe(py, vectors);
        }
        
        // Generate sequential IDs
        let start_id = self.next_id.fetch_add(n as u64, std::sync::atomic::Ordering::SeqCst);
        let ids: Vec<u128> = (start_id..start_id + n as u64)
            .map(|id| id as u128)
            .collect();
        
        // Get contiguous slice - this is the zero-copy path
        let vec_slice = if is_contiguous {
            // ZERO-COPY: Direct pointer to NumPy buffer
            vectors.as_slice().map_err(|e| {
                PyValueError::new_err(format!("Failed to get slice: {}", e))
            })?
        } else {
            // Fallback: must copy for non-contiguous arrays (rare)
            return Err(PyValueError::new_err(
                "Non-contiguous array. Use np.ascontiguousarray(vectors) first."
            ));
        };
        
        // Release GIL for the expensive HNSW work
        let inner = Arc::clone(&self.inner);
        let result = py.allow_threads(move || {
            inner.insert_batch_contiguous(&ids, vec_slice, d)
        });
        
        result.map_err(|e| PyRuntimeError::new_err(e))
    }
    
    /// Insert vectors with explicit IDs.
    ///
    /// Args:
    ///     ids: 1D uint64 array of IDs.
    ///     vectors: 2D float32 array of shape (N, dimension).
    ///
    /// Returns:
    ///     Number of vectors inserted.
    ///
    /// Example:
    ///     >>> ids = np.array([100, 101, 102], dtype=np.uint64)
    ///     >>> vecs = np.random.randn(3, 768).astype(np.float32)
    ///     >>> index.insert_batch_with_ids(ids, vecs)
    fn insert_batch_with_ids<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'py, u64>,
        vectors: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<usize> {
        let id_slice = ids.as_slice().map_err(|e| {
            PyValueError::new_err(format!("IDs must be contiguous: {}", e))
        })?;
        
        let shape = vectors.shape();
        let n = shape[0];
        let d = shape[1];
        
        if d != self.dimension {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: index has {}, got {}",
                self.dimension, d
            )));
        }
        
        if id_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "ID count {} != vector count {}",
                id_slice.len(), n
            )));
        }
        
        // Check contiguity
        if !vectors.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "Vectors must be C-contiguous. Use np.ascontiguousarray(vectors)."
            ));
        }
        
        log_insert_path("insert_batch_with_ids", true, n);
        
        let vec_slice = vectors.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Failed to get slice: {}", e))
        })?;
        
        // Release GIL - use u64-optimized method to avoid Python-side allocation
        // The conversion to u128 happens in Rust which is faster than Python
        let inner = Arc::clone(&self.inner);
        let ids_vec: Vec<u64> = id_slice.to_vec();
        let result = py.allow_threads(move || {
            inner.insert_batch_contiguous_u64(&ids_vec, vec_slice, d)
        });
        
        result.map_err(|e| PyRuntimeError::new_err(e))
    }
    
    /// Safe mode insertion (sequential single-insert).
    fn insert_batch_safe<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<usize> {
        let shape = vectors.shape();
        let n = shape[0];
        let d = shape[1];
        
        let vec_data: Vec<f32> = vectors.to_vec().map_err(|e| {
            PyValueError::new_err(format!("Failed to copy vectors: {}", e))
        })?;
        
        let inner = Arc::clone(&self.inner);
        let next_id = &self.next_id;
        
        let mut count = 0usize;
        for i in 0..n {
            let id = next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as u128;
            let start = i * d;
            let end = start + d;
            let vec: Vec<f32> = vec_data[start..end].to_vec();
            
            if inner.insert(id, vec).is_ok() {
                count += 1;
            }
        }
        
        Ok(count)
    }
    
    /// Search for k nearest neighbors.
    ///
    /// Args:
    ///     query: 1D float32 array of dimension D.
    ///     k: Number of neighbors to return.
    ///     ef_search: Search depth (default: k * 2). Higher = better recall, slower.
    ///
    /// Returns:
    ///     Tuple of (ids, distances) as numpy arrays.
    ///
    /// Example:
    ///     >>> query = np.random.randn(768).astype(np.float32)
    ///     >>> ids, dists = index.search(query, k=10)
    ///     >>> for i, d in zip(ids, dists):
    ///     ...     print(f"ID {i}: distance {d:.4f}")
    #[pyo3(signature = (query, k, ef_search=None))]
    #[allow(unused_variables)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        ef_search: Option<usize>,  // TODO: Support runtime ef_search override
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f32>>)> {
        let query_slice = query.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Query must be contiguous: {}", e))
        })?;
        
        if query_slice.len() != self.dimension {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} != index dimension {}",
                query_slice.len(), self.dimension
            )));
        }
        
        // Release GIL for search
        let inner = Arc::clone(&self.inner);
        let query_vec: Vec<f32> = query_slice.to_vec();
        
        let results = py.allow_threads(move || {
            inner.search(&query_vec, k)
        }).map_err(|e| PyRuntimeError::new_err(e))?;
        
        // Convert to numpy arrays using ndarray
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id as u64).collect();
        let distances: Vec<f32> = results.iter().map(|(_, d)| *d as f32).collect();
        
        let ids_array = Array1::from_vec(ids).into_pyarray(py);
        let dists_array = Array1::from_vec(distances).into_pyarray(py);
        
        Ok((ids_array.into(), dists_array.into()))
    }
    
    /// Batch search for multiple queries.
    ///
    /// Args:
    ///     queries: 2D float32 array of shape (Q, dimension).
    ///     k: Number of neighbors per query.
    ///     ef_search: Search depth (default: k * 2).
    ///
    /// Returns:
    ///     Tuple of (ids, distances) as 2D numpy arrays of shape (Q, k).
    #[pyo3(signature = (queries, k, ef_search=None))]
    #[allow(unused_variables)]
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef_search: Option<usize>,  // TODO: Support runtime ef_search override
    ) -> PyResult<(Py<PyArray2<u64>>, Py<PyArray2<f32>>)> {
        let shape = queries.shape();
        let num_queries = shape[0];
        let d = shape[1];
        
        if d != self.dimension {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} != index dimension {}",
                d, self.dimension
            )));
        }
        
        let queries_vec: Vec<f32> = queries.to_vec().map_err(|e| {
            PyValueError::new_err(format!("Failed to copy queries: {}", e))
        })?;
        
        // Release GIL for parallel search
        let inner = Arc::clone(&self.inner);
        let all_results = py.allow_threads(move || {
            use rayon::prelude::*;
            
            (0..num_queries)
                .into_par_iter()
                .map(|i| {
                    let start = i * d;
                    let end = start + d;
                    let query = &queries_vec[start..end];
                    inner.search(query, k).unwrap_or_default()
                })
                .collect::<Vec<_>>()
        });
        
        // Flatten to 2D arrays
        let mut ids_flat = Vec::with_capacity(num_queries * k);
        let mut dists_flat = Vec::with_capacity(num_queries * k);
        
        for results in all_results {
            for (id, dist) in results.iter().take(k) {
                ids_flat.push(*id as u64);
                dists_flat.push(*dist as f32);
            }
            // Pad if fewer than k results
            for _ in results.len()..k {
                ids_flat.push(u64::MAX);
                dists_flat.push(f32::INFINITY);
            }
        }
        
        // Create 2D arrays using ndarray
        let ids_array = Array2::from_shape_vec((num_queries, k), ids_flat)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create IDs array: {}", e)))?
            .into_pyarray(py);
        let dists_array = Array2::from_shape_vec((num_queries, k), dists_flat)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create distances array: {}", e)))?
            .into_pyarray(py);
        
        Ok((ids_array.into(), dists_array.into()))
    }
    
    /// Get the number of vectors in the index.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    /// Get the dimension of vectors.
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Check if index is empty.
    fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }
    
    /// Save index to disk (compressed).
    ///
    /// Args:
    ///     path: Output file path.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_disk_compressed(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save: {}", e)))
    }
    
    /// Load index from disk.
    ///
    /// Args:
    ///     path: Input file path.
    ///
    /// Returns:
    ///     Loaded HnswIndex.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let index = HnswIndex::load_from_disk_compressed(path)
            .or_else(|_| HnswIndex::load_from_disk(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {}", e)))?;
        
        let stats = index.stats();
        let dimension = stats.dimension;
        let len = stats.num_vectors;
        
        Ok(Self {
            inner: Arc::new(index),
            dimension,
            next_id: std::sync::atomic::AtomicU64::new(len as u64),
        })
    }
    
    /// Get index statistics.
    fn stats(&self) -> PyResult<std::collections::HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let stats = self.inner.stats();
            let mut map = std::collections::HashMap::new();
            
            map.insert("num_vectors".to_string(), stats.num_vectors.into_pyobject(py)?.into());
            map.insert("dimension".to_string(), stats.dimension.into_pyobject(py)?.into());
            map.insert("max_layer".to_string(), stats.max_layer.into_pyobject(py)?.into());
            map.insert("avg_connections".to_string(), stats.avg_connections.into_pyobject(py)?.into());
            
            Ok(map)
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HnswIndex(dimension={}, vectors={}, max_layer={})",
            self.dimension,
            self.inner.len(),
            self.inner.stats().max_layer,
        )
    }
    
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Build an HNSW index from embeddings (in-process, zero-copy).
///
/// This is the recommended way to build an index from NumPy arrays.
/// It's ~10x faster than the subprocess-based bulk_build_index.
///
/// Args:
///     embeddings: 2D float32 array of shape (N, D).
///     m: HNSW max connections (default: 16).
///     ef_construction: Construction depth (default: 100).
///     metric: Distance metric (default: "cosine").
///     ids: Optional 1D uint64 array of IDs.
///
/// Returns:
///     HnswIndex with inserted vectors.
///
/// Example:
///     >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
///     >>> index = build_index(embeddings, m=16, ef_construction=100)
///     >>> index.save("my_index.hnsw")
#[pyfunction]
#[pyo3(signature = (embeddings, m=16, ef_construction=100, metric="cosine", ids=None))]
fn build_index<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f32>,
    m: usize,
    ef_construction: usize,
    metric: &str,
    ids: Option<PyReadonlyArray1<'py, u64>>,
) -> PyResult<PyHnswIndex> {
    let shape = embeddings.shape();
    let d = shape[1];
    
    let index = PyHnswIndex::new(d, m, ef_construction, metric, "f32")?;
    
    if let Some(id_array) = ids {
        index.insert_batch_with_ids(py, id_array, embeddings)?;
    } else {
        index.insert_batch(py, embeddings)?;
    }
    
    Ok(index)
}

/// Get version information.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Check if running in safe mode.
#[pyfunction]
fn is_safe_mode() -> bool {
    check_safe_mode()
}

// =============================================================================
// Python Module
// =============================================================================

/// ToonDB - AI-native database with vector search.
///
/// This module provides high-performance vector indexing and search
/// using HNSW (Hierarchical Navigable Small World) graphs.
///
/// Example:
///     >>> import numpy as np
///     >>> import toondb
///     >>> 
///     >>> # Build index from embeddings
///     >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
///     >>> index = toondb.build_index(embeddings)
///     >>> 
///     >>> # Search
///     >>> query = np.random.randn(768).astype(np.float32)
///     >>> ids, distances = index.search(query, k=10)
#[pymodule]
fn toondb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHnswIndex>()?;
    m.add_function(wrap_pyfunction!(build_index, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_safe_mode, m)?)?;
    
    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
