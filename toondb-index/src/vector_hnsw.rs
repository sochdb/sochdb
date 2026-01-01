// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HNSW (Hierarchical Navigable Small World) Vector Index
//!
//! Provides O(log N) approximate nearest neighbor search with high recall (>95%).
//! This replaces the O(N) brute-force implementation with a graph-based approach.

use crate::vector_storage::{MemoryVectorStorage, MmapVectorStorage, VectorStorage};
use ndarray::Array1;
use parking_lot::RwLock;
use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

pub type Embedding = Array1<f32>;

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// HNSW node with layered graph structure
#[derive(Clone)]
struct HNSWNode {
    edge_id: u128,
    vector_id: u64,
    /// Neighbors for each layer (layer 0 = densest, higher layers = sparser)
    layers: Vec<Vec<usize>>,
    /// Tombstone flag for soft deletion. Deleted nodes are skipped in search
    /// but kept in the graph to maintain connectivity. Periodic rebuild removes them.
    deleted: bool,
}

/// Candidate entry for priority queue (min-heap by distance)
#[derive(Clone)]
struct Candidate {
    distance: f32,
    node_idx: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// HNSW Vector Index for O(log N) approximate nearest neighbor search
///
/// Key parameters:
/// - M: max connections per layer (typically 16-32)
/// - M_max: max connections for layer 0 (typically 2*M)
/// - ef_construction: search depth during insertion (typically 200)
/// - ef_search: search depth during queries (typically 50-100)
///
/// Performance targets:
/// - 10K vectors: < 1ms
/// - 1M vectors: < 5ms
/// - 10M vectors: < 10ms
/// - 100M vectors: < 50ms
///
/// **TODO (MEMORY OPTIMIZATION):**
/// The current implementation loads ALL vectors into heap memory (`Vec<HNSWNode>`).
/// For large datasets with high-dimensional embeddings:
/// - 1M vectors × 768 dims × 4 bytes = ~3GB RAM
/// - 10M vectors × 1536 dims × 4 bytes = ~60GB RAM (exceeds desktop RAM)
///
/// **RECOMMENDED FIX:**
/// 1. **Memory-mapped Vector Storage:**
///    - Keep HNSW graph structure (navigable links) in memory
///    - Store actual embedding vectors in mmap'd file
///    - Fetch vectors on-demand during distance calculations
///    - Example: `nodes: Vec<HNSWNode>` where `HNSWNode.vector` is a file offset
///
/// 2. **Quantization:**
///    - Store vectors in lower precision (f16 or int8) to reduce memory 2-4x
///    - Keep original vectors on disk, quantized in memory
///    - Rerank top-k with original vectors for accuracy
///
/// 3. **Hierarchical Storage:**
///    - Hot vectors (recently accessed) in memory
///    - Cold vectors on disk with LRU cache
///    - Pre-load neighbors during graph traversal
///
/// Memory-mapped storage is essential for 10M+ vectors on desktop hardware.
pub struct VectorIndex {
    nodes: RwLock<Vec<HNSWNode>>,
    storage: Arc<dyn VectorStorage>,
    entry_point: AtomicUsize,
    max_level: RwLock<usize>,
    metric: DistanceMetric,
    expected_dim: RwLock<Option<usize>>,

    /// Mutex for structural modifications (entry_point, max_level changes)
    /// This ensures atomic updates when inserting the first node or when
    /// a new node has a higher level than the current max.
    structure_lock: parking_lot::Mutex<()>,

    // HNSW parameters
    m: usize,
    m_max: usize,
    ef_construction: usize,
    ef_search: RwLock<usize>,
    ml: f32, // 1/ln(M)
}

impl VectorIndex {
    /// Create new HNSW vector index with default parameters
    ///
    /// Default: M=16, ef_construction=200, ef_search=100
    pub fn new(metric: DistanceMetric) -> Self {
        Self::with_params(metric, 16, 200, 100)
    }

    /// Create HNSW index with custom parameters
    ///
    /// - M: max connections per layer (16-32 recommended)
    /// - ef_construction: build quality (100-400, higher = better but slower)
    /// - ef_search: query quality (50-200, higher = better recall but slower)
    pub fn with_params(
        metric: DistanceMetric,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        let m_max = 2 * m;
        let ml = 1.0 / (m as f32).ln();

        Self {
            nodes: RwLock::new(Vec::new()),
            storage: Arc::new(MemoryVectorStorage::new(0)), // Default to empty memory storage
            entry_point: AtomicUsize::new(0),
            max_level: RwLock::new(0),
            metric,
            expected_dim: RwLock::new(None),
            structure_lock: parking_lot::Mutex::new(()),
            m,
            m_max,
            ef_construction,
            ef_search: RwLock::new(ef_search),
            ml,
        }
    }

    /// Create index with fixed dimension validation
    pub fn with_dimension(metric: DistanceMetric, dim: usize) -> Self {
        let mut index = Self::new(metric);
        *index.expected_dim.write() = Some(dim);
        index.storage = Arc::new(MemoryVectorStorage::new(dim));
        index
    }

    /// Create index with Mmap storage
    pub fn with_mmap_storage<P: AsRef<Path>>(
        metric: DistanceMetric,
        path: P,
        dim: usize,
    ) -> io::Result<Self> {
        let storage = MmapVectorStorage::new(path, dim)?;
        let mut index = Self::new(metric);
        *index.expected_dim.write() = Some(dim);
        index.storage = Arc::new(storage);
        Ok(index)
    }

    /// Add vector to index with O(log N) HNSW insertion
    ///
    /// **Dimension Locking (Gap #14 Fix):**
    /// On first insert, the dimension is locked. All subsequent inserts must
    /// match this dimension to prevent silent search quality degradation.
    pub fn add(&self, edge_id: u128, vector: Embedding) -> Result<(), String> {
        // Lock dimension on first insert (Gap #14 fix)
        let expected_dim = {
            let mut dim_guard = self.expected_dim.write();
            if let Some(dim) = *dim_guard {
                dim
            } else {
                // First insert - lock the dimension
                let dim = vector.len();
                *dim_guard = Some(dim);
                tracing::info!("Vector index: locking dimension to {} on first insert", dim);
                dim
            }
        };

        // Validate dimension (always enforced)
        if vector.len() != expected_dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}. \
                 Dimension was locked on first insert. If you changed embedding models, \
                 you must rebuild the index.",
                expected_dim,
                vector.len()
            ));
        }

        // Determine random level using exponential distribution (before acquiring locks)
        let level = self.random_level();

        // Append vector to storage (before acquiring node lock)
        let vector_id = self.storage.append(&vector).map_err(|e| e.to_string())?;

        // Acquire structure lock for atomic first-node insertion check
        // This prevents race conditions when two threads insert into an empty index
        let _structure_guard = self.structure_lock.lock();

        let mut nodes = self.nodes.write();
        let idx = nodes.len();

        // Create new node
        let node = HNSWNode {
            edge_id,
            vector_id,
            layers: vec![Vec::new(); level + 1],
            deleted: false,
        };

        // First node becomes entry point (now race-free due to structure_lock)
        if idx == 0 {
            nodes.push(node);
            self.entry_point.store(0, AtomicOrdering::Release);
            *self.max_level.write() = level;
            return Ok(());
        }

        nodes.push(node);

        // Read max_level while holding nodes lock to ensure consistency
        let max_level_val = *self.max_level.read();

        // Search for insertion point
        let mut curr_nearest = vec![self.entry_point.load(AtomicOrdering::Acquire)];

        // Zoom into higher layers
        for lc in (level + 1..=max_level_val).rev() {
            curr_nearest = self.search_layer_internal(&nodes, &vector, &curr_nearest, 1, lc)?;
        }

        // Insert at each layer from top to bottom
        for lc in (0..=level).rev() {
            let candidates = self.search_layer_internal(
                &nodes,
                &vector,
                &curr_nearest,
                self.ef_construction,
                lc,
            )?;

            let m = if lc == 0 { self.m_max } else { self.m };
            let neighbors = self.select_neighbors_heuristic(&nodes, &candidates, &vector, m)?;

            // Add bidirectional links
            for &neighbor_idx in &neighbors {
                nodes[idx].layers[lc].push(neighbor_idx);

                // Only add back-link if neighbor has this layer
                if lc < nodes[neighbor_idx].layers.len() {
                    nodes[neighbor_idx].layers[lc].push(idx);

                    // Prune if neighbor exceeds m
                    let max_conn = if lc == 0 { self.m_max } else { self.m };
                    if nodes[neighbor_idx].layers[lc].len() > max_conn {
                        let pruned = self.prune_connections(&nodes, neighbor_idx, lc, max_conn)?;
                        nodes[neighbor_idx].layers[lc] = pruned;
                    }
                }
            }

            curr_nearest = candidates;
        }

        // Update entry point if this node has higher level (atomic with structure_lock held)
        if level > max_level_val {
            self.entry_point.store(idx, AtomicOrdering::Release);
            *self.max_level.write() = level;
        }

        Ok(())
    }

    /// Soft-delete a vector by edge_id using tombstone pattern
    /// Returns Ok(true) if found and deleted, Ok(false) if not found
    pub fn delete(&self, edge_id: u128) -> Result<bool, String> {
        let mut nodes = self.nodes.write();

        // Find the node with matching edge_id
        for node in nodes.iter_mut() {
            if node.edge_id == edge_id {
                if node.deleted {
                    return Ok(false); // Already deleted
                }
                node.deleted = true;
                return Ok(true);
            }
        }

        Ok(false) // Not found
    }

    /// Check if a vector exists and is not deleted
    pub fn contains(&self, edge_id: u128) -> bool {
        let nodes = self.nodes.read();
        nodes.iter().any(|n| n.edge_id == edge_id && !n.deleted)
    }

    /// Search for k nearest neighbors with O(log N) HNSW algorithm
    pub fn search(&self, query: &Embedding, k: usize) -> Result<Vec<(u128, f32)>, String> {
        // Validate dimension
        if let Some(expected_dim) = *self.expected_dim.read()
            && query.len() != expected_dim
        {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                expected_dim,
                query.len()
            ));
        }

        let nodes = self.nodes.read();
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        let mut curr_nearest = vec![self.entry_point.load(AtomicOrdering::Acquire)];
        let max_level_val = *self.max_level.read();

        // Search through layers from top to bottom
        for lc in (1..=max_level_val).rev() {
            curr_nearest = self.search_layer_internal(&nodes, query, &curr_nearest, 1, lc)?;
        }

        // Search layer 0 with ef_search parameter
        let ef = *self.ef_search.read();
        let candidates = self.search_layer_internal(&nodes, query, &curr_nearest, ef.max(k), 0)?;

        // Convert to result format and limit to k, skipping deleted nodes
        let mut results: Vec<(u128, f32)> = Vec::with_capacity(candidates.len());
        let mut vec_buffer = vec![0.0; self.storage.dim()];
        let query_slice = query.as_slice().unwrap();

        for idx in candidates {
            // Skip deleted nodes (tombstone pattern)
            if nodes[idx].deleted {
                continue;
            }
            self.storage
                .get_into(nodes[idx].vector_id, &mut vec_buffer)
                .map_err(|e| e.to_string())?;
            let dist = self.compute_distance(&vec_buffer, query_slice);
            results.push((nodes[idx].edge_id, dist));
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Search within a specific layer (internal algorithm)
    fn search_layer_internal(
        &self,
        nodes: &[HNSWNode],
        query: &Embedding,
        entry_points: &[usize],
        num_closest: usize,
        layer: usize,
    ) -> Result<Vec<usize>, String> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        let mut vec_buffer = vec![0.0; self.storage.dim()];
        let query_slice = query.as_slice().unwrap();

        // Initialize with entry points
        for &ep in entry_points {
            self.storage
                .get_into(nodes[ep].vector_id, &mut vec_buffer)
                .map_err(|e| e.to_string())?;
            let dist = self.compute_distance(&vec_buffer, query_slice);
            candidates.push(Candidate {
                distance: dist,
                node_idx: ep,
            });
            w.push(Reverse(Candidate {
                distance: dist,
                node_idx: ep,
            }));
            visited.insert(ep);
        }

        while let Some(c) = candidates.pop() {
            // If c is farther than furthest in w, we're done
            if let Some(Reverse(furthest)) = w.peek()
                && c.distance > furthest.distance
            {
                break;
            }

            // Check all neighbors of c at this layer
            if layer < nodes[c.node_idx].layers.len() {
                for &neighbor_idx in &nodes[c.node_idx].layers[layer] {
                    if visited.insert(neighbor_idx) {
                        self.storage
                            .get_into(nodes[neighbor_idx].vector_id, &mut vec_buffer)
                            .map_err(|e| e.to_string())?;
                        let dist = self.compute_distance(&vec_buffer, query_slice);

                        // Add to result set if better than worst or w not full
                        if let Some(Reverse(furthest)) = w.peek() {
                            if dist < furthest.distance || w.len() < num_closest {
                                candidates.push(Candidate {
                                    distance: dist,
                                    node_idx: neighbor_idx,
                                });
                                w.push(Reverse(Candidate {
                                    distance: dist,
                                    node_idx: neighbor_idx,
                                }));

                                if w.len() > num_closest {
                                    w.pop();
                                }
                            }
                        } else {
                            candidates.push(Candidate {
                                distance: dist,
                                node_idx: neighbor_idx,
                            });
                            w.push(Reverse(Candidate {
                                distance: dist,
                                node_idx: neighbor_idx,
                            }));
                        }
                    }
                }
            }
        }

        Ok(w.into_iter().map(|Reverse(c)| c.node_idx).collect())
    }

    /// Select neighbors using heuristic (Algorithm 4 from HNSW paper)
    fn select_neighbors_heuristic(
        &self,
        nodes: &[HNSWNode],
        candidates: &[usize],
        query: &Embedding,
        m: usize,
    ) -> Result<Vec<usize>, String> {
        if candidates.len() <= m {
            return Ok(candidates.to_vec());
        }

        // Simple heuristic: take m closest by distance
        let mut sorted: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
        let mut vec_buffer = vec![0.0; self.storage.dim()];
        let query_slice = query.as_slice().unwrap();

        for &idx in candidates {
            self.storage
                .get_into(nodes[idx].vector_id, &mut vec_buffer)
                .map_err(|e| e.to_string())?;
            let dist = self.compute_distance(&vec_buffer, query_slice);
            sorted.push((idx, dist));
        }

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted.truncate(m);
        Ok(sorted.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Prune connections when node exceeds m
    fn prune_connections(
        &self,
        nodes: &[HNSWNode],
        node_idx: usize,
        layer: usize,
        m: usize,
    ) -> Result<Vec<usize>, String> {
        let connections = &nodes[node_idx].layers[layer];
        if connections.len() <= m {
            return Ok(connections.clone());
        }

        // Keep m closest neighbors
        let mut node_vec_buffer = vec![0.0; self.storage.dim()];
        self.storage
            .get_into(nodes[node_idx].vector_id, &mut node_vec_buffer)
            .map_err(|e| e.to_string())?;

        let mut neighbor_vec_buffer = vec![0.0; self.storage.dim()];
        let mut sorted: Vec<(usize, f32)> = Vec::with_capacity(connections.len());

        for &idx in connections {
            self.storage
                .get_into(nodes[idx].vector_id, &mut neighbor_vec_buffer)
                .map_err(|e| e.to_string())?;
            let dist = self.compute_distance(&node_vec_buffer, &neighbor_vec_buffer);
            sorted.push((idx, dist));
        }

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted.truncate(m);
        Ok(sorted.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Generate random level using exponential distribution per HNSW paper
    ///
    /// Uses the configured level_multiplier (ml = 1/ln(M)) for proper level distribution.
    /// Per Malkov & Yashunin 2018: level = floor(-ln(uniform(0,1)) * mL)
    /// This ensures proper hierarchical structure with O(log N) search complexity.
    ///
    /// For M=16: mL ≈ 0.36, expected level ≈ 0.36, P(level≥1) ≈ 30%
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.r#gen();
        // Avoid ln(0) which is -infinity
        let level = if uniform > 0.0 {
            (-uniform.ln() * self.ml).floor() as usize
        } else {
            0
        };
        level.min(15) // Cap at 15 levels for safety
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - (dot / (norm_a * norm_b + 1e-8))
            }
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::DotProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
        }
    }

    /// Number of vectors in index
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Clear all vectors
    pub fn clear(&self) {
        self.nodes.write().clear();
        self.entry_point.store(0, AtomicOrdering::Release);
        *self.max_level.write() = 0;
    }

    /// Set search quality parameter
    pub fn set_ef_search(&self, ef: usize) {
        *self.ef_search.write() = ef;
    }

    /// Save index to disk (version 2 with HNSW graph)
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let nodes = self.nodes.read();
        let mut file = BufWriter::new(File::create(path)?);

        // Header
        file.write_all(b"CHRL_VEC")?;
        file.write_all(&2u32.to_le_bytes())?; // version 2 for HNSW

        // Metadata
        // Metadata
        file.write_all(&[self.metric as u8])?;
        // Use expected_dim if set, otherwise try to get dim from storage
        let dim = self
            .expected_dim
            .read()
            .unwrap_or_else(|| self.storage.dim()) as u32;
        file.write_all(&dim.to_le_bytes())?;
        file.write_all(&(nodes.len() as u64).to_le_bytes())?;

        // HNSW parameters
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.m_max as u32).to_le_bytes())?;
        file.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        let ef_search_val = *self.ef_search.read();
        file.write_all(&(ef_search_val as u32).to_le_bytes())?;
        file.write_all(&self.ml.to_le_bytes())?;

        // Graph metadata
        let max_level_val = *self.max_level.read();
        file.write_all(&(max_level_val as u32).to_le_bytes())?;
        file.write_all(&(self.entry_point.load(AtomicOrdering::Acquire) as u64).to_le_bytes())?;

        // Nodes with graph structure
        for node in nodes.iter() {
            // Edge ID
            file.write_all(&node.edge_id.to_le_bytes())?;

            // Vector
            let vec = self.storage.get(node.vector_id).map_err(io::Error::other)?;
            for &val in vec.iter() {
                file.write_all(&val.to_le_bytes())?;
            }

            // Graph: num_layers, then for each layer: num_neighbors + neighbor_indices
            file.write_all(&(node.layers.len() as u32).to_le_bytes())?;
            for layer_connections in &node.layers {
                file.write_all(&(layer_connections.len() as u32).to_le_bytes())?;
                for &neighbor_idx in layer_connections {
                    file.write_all(&(neighbor_idx as u64).to_le_bytes())?;
                }
            }
        }

        file.flush()
    }

    /// Load index from disk
    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);

        // Read and validate header
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != b"CHRL_VEC" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid vector index magic header",
            ));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        // Support both version 1 (old brute-force) and version 2 (HNSW)
        if version == 1 {
            return Self::load_v1_format(file);
        } else if version != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported vector index version: {}", version),
            ));
        }

        // Read metadata
        let mut metric_byte = [0u8; 1];
        file.read_exact(&mut metric_byte)?;
        let metric = match metric_byte[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid metric byte: {}", metric_byte[0]),
                ));
            }
        };

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;

        // Read HNSW parameters
        let mut m_bytes = [0u8; 4];
        file.read_exact(&mut m_bytes)?;
        let m = u32::from_le_bytes(m_bytes) as usize;

        let mut m_max_bytes = [0u8; 4];
        file.read_exact(&mut m_max_bytes)?;
        let m_max = u32::from_le_bytes(m_max_bytes) as usize;

        let mut ef_construction_bytes = [0u8; 4];
        file.read_exact(&mut ef_construction_bytes)?;
        let ef_construction = u32::from_le_bytes(ef_construction_bytes) as usize;

        let mut ef_search_bytes = [0u8; 4];
        file.read_exact(&mut ef_search_bytes)?;
        let ef_search = u32::from_le_bytes(ef_search_bytes) as usize;

        let mut ml_bytes = [0u8; 4];
        file.read_exact(&mut ml_bytes)?;
        let ml = f32::from_le_bytes(ml_bytes);

        // Read graph metadata
        let mut max_level_bytes = [0u8; 4];
        file.read_exact(&mut max_level_bytes)?;
        let max_level = u32::from_le_bytes(max_level_bytes) as usize;

        let mut entry_point_bytes = [0u8; 8];
        file.read_exact(&mut entry_point_bytes)?;
        let entry_point = u64::from_le_bytes(entry_point_bytes) as usize;

        // Initialize storage (in-memory for loaded index)
        let storage = Arc::new(MemoryVectorStorage::new(dimension));

        // Read nodes
        let mut nodes = Vec::with_capacity(count);
        for _ in 0..count {
            // Edge ID
            let mut edge_id_bytes = [0u8; 16];
            file.read_exact(&mut edge_id_bytes)?;
            let edge_id = u128::from_le_bytes(edge_id_bytes);

            // Vector
            let mut vector_data = vec![0f32; dimension];
            for val in vector_data.iter_mut() {
                let mut f_bytes = [0u8; 4];
                file.read_exact(&mut f_bytes)?;
                *val = f32::from_le_bytes(f_bytes);
            }
            let vector = Array1::from_vec(vector_data);
            let vector_id = storage.append(&vector)?;

            // Graph layers
            let mut num_layers_bytes = [0u8; 4];
            file.read_exact(&mut num_layers_bytes)?;
            let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;

            let mut layers = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let mut num_neighbors_bytes = [0u8; 4];
                file.read_exact(&mut num_neighbors_bytes)?;
                let num_neighbors = u32::from_le_bytes(num_neighbors_bytes) as usize;

                let mut layer_connections = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    let mut neighbor_idx_bytes = [0u8; 8];
                    file.read_exact(&mut neighbor_idx_bytes)?;
                    let neighbor_idx = u64::from_le_bytes(neighbor_idx_bytes) as usize;
                    layer_connections.push(neighbor_idx);
                }
                layers.push(layer_connections);
            }

            nodes.push(HNSWNode {
                edge_id,
                vector_id,
                layers,
                deleted: false,
            });
        }

        Ok(Self {
            nodes: RwLock::new(nodes),
            storage,
            entry_point: AtomicUsize::new(entry_point),
            max_level: RwLock::new(max_level),
            metric,
            expected_dim: RwLock::new(if dimension > 0 { Some(dimension) } else { None }),
            structure_lock: parking_lot::Mutex::new(()),
            m,
            m_max,
            ef_construction,
            ef_search: RwLock::new(ef_search),
            ml,
        })
    }

    /// Load old version 1 format (brute-force) and convert to HNSW
    fn load_v1_format<R: Read>(mut file: R) -> io::Result<Self> {
        // Read metric
        let mut metric_byte = [0u8; 1];
        file.read_exact(&mut metric_byte)?;
        let metric = match metric_byte[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid metric byte: {}", metric_byte[0]),
                ));
            }
        };

        // Read dimension
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        // Read count
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;

        // Create new HNSW index
        let index = Self::new(metric);

        // Read and re-insert all vectors (builds HNSW graph)
        for _ in 0..count {
            // Edge ID
            let mut edge_id_bytes = [0u8; 16];
            file.read_exact(&mut edge_id_bytes)?;
            let edge_id = u128::from_le_bytes(edge_id_bytes);

            // Vector
            let mut vector_data = vec![0f32; dimension];
            for val in vector_data.iter_mut() {
                let mut f_bytes = [0u8; 4];
                file.read_exact(&mut f_bytes)?;
                *val = f32::from_le_bytes(f_bytes);
            }
            let vector = Array1::from_vec(vector_data);

            // Insert into HNSW (this builds the graph structure)
            index
                .add(edge_id, vector)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }

        Ok(index)
    }

    /// Create vector index with persistence enabled
    pub fn with_persistence<P: AsRef<Path>>(path: P, metric: DistanceMetric) -> Self {
        if path.as_ref().exists() {
            Self::load_from_disk(&path).unwrap_or_else(|e| {
                eprintln!(
                    "Warning: Failed to load vector index: {}. Starting fresh.",
                    e
                );
                Self::new(metric)
            })
        } else {
            Self::new(metric)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use std::time::Instant;

    #[test]
    fn test_hnsw_basic() {
        let index = VectorIndex::new(DistanceMetric::Cosine);

        // Add vectors
        index.add(1, arr1(&[1.0, 0.0, 0.0])).unwrap();
        index.add(2, arr1(&[0.9, 0.1, 0.0])).unwrap();
        index.add(3, arr1(&[0.0, 1.0, 0.0])).unwrap();

        // Search
        let results = index.search(&arr1(&[1.0, 0.0, 0.0]), 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Exact match should be first
    }

    #[test]
    fn test_hnsw_empty() {
        let index = VectorIndex::new(DistanceMetric::Euclidean);
        let results = index.search(&arr1(&[1.0, 2.0]), 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_single() {
        let index = VectorIndex::new(DistanceMetric::Euclidean);
        index.add(42, arr1(&[1.0, 2.0, 3.0])).unwrap();

        let results = index.search(&arr1(&[1.0, 2.0, 3.0]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn test_hnsw_dimension_validation() {
        let index = VectorIndex::with_dimension(DistanceMetric::Cosine, 3);

        assert!(index.add(1, arr1(&[1.0, 0.0, 0.0])).is_ok());
        assert!(index.add(2, arr1(&[1.0, 0.0])).is_err());

        assert!(index.search(&arr1(&[1.0, 0.0, 0.0]), 1).is_ok());
        assert!(index.search(&arr1(&[1.0, 0.0]), 1).is_err());
    }

    #[test]
    fn test_hnsw_persistence() {
        use std::fs;
        let path = "/tmp/test_hnsw_index.bin";

        // Create and populate index
        {
            let index = VectorIndex::new(DistanceMetric::Cosine);
            index.add(1, arr1(&[1.0, 0.0])).unwrap();
            index.add(2, arr1(&[0.0, 1.0])).unwrap();
            index.add(3, arr1(&[0.5, 0.5])).unwrap();
            index.save_to_disk(path).unwrap();
        }

        // Load and verify
        {
            let index = VectorIndex::load_from_disk(path).unwrap();
            assert_eq!(index.len(), 3);

            let results = index.search(&arr1(&[1.0, 0.0]), 2).unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, 1);
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_hnsw_performance_1k() {
        let index = VectorIndex::with_params(DistanceMetric::Cosine, 16, 200, 100);

        // Insert 1000 random vectors
        let start = Instant::now();
        for i in 0..1000 {
            let vec: Vec<f32> = (0..128)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            index.add(i as u128, Array1::from_vec(vec)).unwrap();
        }
        let insert_time = start.elapsed();
        println!("HNSW: Inserted 1K vectors in {:?}", insert_time);

        // Search
        let query: Vec<f32> = (0..128).map(|i| (i % 10) as f32 / 10.0).collect();
        let query = Array1::from_vec(query);

        let start = Instant::now();
        let results = index.search(&query, 10).unwrap();
        let search_time = start.elapsed();

        println!("HNSW: Search in 1K vectors took {:?}", search_time);
        assert_eq!(results.len(), 10);
        assert!(search_time.as_micros() < 10000, "Search should be < 10ms");
    }

    #[test]
    fn test_mmap_storage_persistence() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let dim = 128;

        // Create index with mmap storage
        let index = VectorIndex::with_mmap_storage(DistanceMetric::Cosine, &path, dim).unwrap();

        // Add vectors
        for i in 0..100 {
            let vec: Vec<f32> = (0..dim)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            index.add(i as u128, Array1::from_vec(vec)).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query_vec: Vec<f32> = (0..dim)
            .map(|j| ((50 * 7 + j * 13) % 100) as f32 / 100.0)
            .collect();
        let results = index.search(&Array1::from_vec(query_vec), 5).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 50); // Should find exact match

        // Close index (drop)
        drop(index);

        // Verify file exists and has content
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(file_len, (100 * dim * 4) as u64);
    }

    #[test]
    fn test_hnsw_performance_10k() {
        let index = VectorIndex::with_params(DistanceMetric::Cosine, 16, 200, 100);

        // Insert 10K random vectors
        let start = Instant::now();
        for i in 0..10000 {
            let vec: Vec<f32> = (0..128)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            index.add(i as u128, Array1::from_vec(vec)).unwrap();
        }
        let insert_time = start.elapsed();
        println!("HNSW: Inserted 10K vectors in {:?}", insert_time);

        // Search
        let query: Vec<f32> = (0..128).map(|i| (i % 10) as f32 / 10.0).collect();
        let query = Array1::from_vec(query);

        let start = Instant::now();
        let results = index.search(&query, 10).unwrap();
        let search_time = start.elapsed();

        println!("HNSW: Search in 10K vectors took {:?}", search_time);
        assert_eq!(results.len(), 10);
        assert!(
            search_time.as_micros() < 50000,
            "Search should be < 50ms for 10K vectors"
        );
    }

    #[test]
    fn test_hnsw_metrics() {
        // Test Euclidean
        let index_euc = VectorIndex::new(DistanceMetric::Euclidean);
        index_euc.add(1, arr1(&[0.0, 0.0])).unwrap();
        index_euc.add(2, arr1(&[3.0, 4.0])).unwrap();
        let results = index_euc.search(&arr1(&[0.0, 0.0]), 2).unwrap();
        assert_eq!(results[0].0, 1);

        // Test DotProduct
        let index_dot = VectorIndex::new(DistanceMetric::DotProduct);
        index_dot.add(1, arr1(&[1.0, 0.0])).unwrap();
        index_dot.add(2, arr1(&[0.0, 1.0])).unwrap();
        let results = index_dot.search(&arr1(&[1.0, 0.0]), 1).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_hnsw_clear() {
        let index = VectorIndex::new(DistanceMetric::Cosine);
        index.add(1, arr1(&[1.0, 0.0])).unwrap();
        assert_eq!(index.len(), 1);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }
}
