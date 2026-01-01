// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Locality-Aware Insertion Ordering (Task 5)
//!
//! Optimizes cache/TLB efficiency during HNSW construction by reordering
//! vectors to maximize spatial locality. Consecutive insertions are likely
//! to have similar vectors, increasing cache hit probability during graph
//! traversal.
//!
//! ## Why Ordering Matters
//!
//! HNSW build quality and cache efficiency are insertion-order sensitive:
//! - Random order: each traversal touches vectors scattered across memory
//! - Locality-aware: neighbor candidates are recently-touched and cache-hot
//!
//! ## Expected Improvements
//!
//! - 10-30% throughput improvement in compute-bound regime
//! - 0.5-2% better recall at same ef_construction
//! - More predictable build times
//!
//! ## Algorithms
//!
//! 1. **Random Projection Sort** (O(N·D) + O(N log N))
//!    - Project vectors to 1D via random unit vector
//!    - Sort by projection value
//!    - Simple, fast, effective
//!
//! 2. **Coarse K-Means** (O(N·D·iters))
//!    - Cluster into k = √N groups
//!    - Insert cluster-by-cluster
//!    - Better locality but slower preprocessing

use rand::Rng;
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

/// Reordering strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingStrategy {
    /// No reordering (original order)
    None,
    /// Random projection to 1D, then sort
    RandomProjection,
    /// Coarse k-means clustering
    CoarseKMeans { k: usize },
    /// Block shuffle (simple locality improvement)
    BlockShuffle { block_size: usize },
}

impl Default for OrderingStrategy {
    fn default() -> Self {
        Self::RandomProjection
    }
}

/// Result of reordering operation
#[derive(Debug)]
pub struct ReorderResult {
    /// Permutation indices (new_position[i] = original_index)
    pub permutation: Vec<usize>,
    /// Time spent computing order
    pub compute_time: std::time::Duration,
    /// Strategy used
    pub strategy: OrderingStrategy,
}

impl ReorderResult {
    /// Apply permutation to a slice, returning reordered data
    pub fn apply_to_vectors(&self, vectors: &[f32], dim: usize) -> Vec<f32> {
        let n = self.permutation.len();
        let mut result = vec![0.0f32; n * dim];
        
        for (new_idx, &old_idx) in self.permutation.iter().enumerate() {
            let src_start = old_idx * dim;
            let dst_start = new_idx * dim;
            result[dst_start..dst_start + dim].copy_from_slice(&vectors[src_start..src_start + dim]);
        }
        
        result
    }
    
    /// Apply permutation to IDs
    pub fn apply_to_ids<T: Copy>(&self, ids: &[T]) -> Vec<T> {
        self.permutation.iter().map(|&i| ids[i]).collect()
    }
}

/// Compute locality-aware ordering for vectors
///
/// # Arguments
/// * `vectors` - Contiguous f32 slice [N × D]
/// * `dim` - Vector dimension
/// * `strategy` - Ordering strategy
///
/// # Returns
/// Permutation that should be applied before insertion
pub fn compute_ordering(
    vectors: &[f32],
    dim: usize,
    strategy: OrderingStrategy,
) -> ReorderResult {
    let start = Instant::now();
    let n = vectors.len() / dim;
    
    let permutation = match strategy {
        OrderingStrategy::None => {
            (0..n).collect()
        }
        OrderingStrategy::RandomProjection => {
            random_projection_order(vectors, dim)
        }
        OrderingStrategy::CoarseKMeans { k } => {
            coarse_kmeans_order(vectors, dim, k)
        }
        OrderingStrategy::BlockShuffle { block_size } => {
            block_shuffle_order(n, block_size)
        }
    };
    
    ReorderResult {
        permutation,
        compute_time: start.elapsed(),
        strategy,
    }
}

/// Random projection ordering
///
/// 1. Generate random unit vector
/// 2. Project all vectors onto it
/// 3. Sort by projection value
///
/// This places similar vectors nearby in the insertion order,
/// improving cache locality during graph construction.
fn random_projection_order(vectors: &[f32], dim: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    
    // Generate random unit vector
    let mut rng = rand::thread_rng();
    let random_vec: Vec<f32> = (0..dim)
        .map(|_| {
            // Use random::<f32>() instead of gen() which is a reserved keyword
            use rand::distributions::{Distribution, Uniform};
            let dist = Uniform::new(-0.5_f32, 0.5_f32);
            dist.sample(&mut rng)
        })
        .collect();
    
    // Normalize
    let norm: f32 = random_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let unit_vec: Vec<f32> = random_vec.iter().map(|x| x / norm).collect();
    
    // Project all vectors (parallel)
    let projections: Vec<(usize, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            let vec = &vectors[start..start + dim];
            let proj: f32 = vec.iter()
                .zip(unit_vec.iter())
                .map(|(a, b)| a * b)
                .sum();
            (i, proj)
        })
        .collect();
    
    // Sort by projection
    let mut sorted = projections;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    sorted.iter().map(|(i, _)| *i).collect()
}

/// Coarse k-means ordering
///
/// 1. Run k-means with k = √N
/// 2. Assign vectors to clusters
/// 3. Insert cluster-by-cluster
fn coarse_kmeans_order(vectors: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    let k = k.min(n).max(1);
    
    // Initialize centroids with k-means++ style (simplified: random sample)
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    
    for _ in 0..k {
        let idx = rng.gen_range(0..n);
        let start = idx * dim;
        centroids.push(vectors[start..start + dim].to_vec());
    }
    
    // Run 3 iterations of k-means (enough for coarse clustering)
    let mut assignments = vec![0usize; n];
    
    for _iter in 0..3 {
        // Assign points to nearest centroid (parallel)
        assignments = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let vec = &vectors[start..start + dim];
                
                let mut best_cluster = 0;
                let mut best_dist = f32::MAX;
                
                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f32 = vec.iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                
                best_cluster
            })
            .collect();
        
        // Update centroids
        let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        
        for (i, &c) in assignments.iter().enumerate() {
            let start = i * dim;
            for d in 0..dim {
                new_centroids[c][d] += vectors[start + d];
            }
            counts[c] += 1;
        }
        
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = new_centroids[c][d] / counts[c] as f32;
                }
            }
        }
    }
    
    // Group indices by cluster, then flatten
    let mut cluster_groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in assignments.iter().enumerate() {
        cluster_groups[c].push(i);
    }
    
    cluster_groups.into_iter().flatten().collect()
}

/// Block shuffle ordering
///
/// Divide into blocks, shuffle block order.
/// Simple but provides some locality improvement.
fn block_shuffle_order(n: usize, block_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let block_size = block_size.max(1);
    
    // Create blocks
    let mut blocks: Vec<Vec<usize>> = (0..n)
        .collect::<Vec<_>>()
        .chunks(block_size)
        .map(|c| c.to_vec())
        .collect();
    
    // Shuffle blocks (but keep elements within blocks in order)
    // SliceRandom is already imported via prelude
    blocks.shuffle(&mut rng);
    
    blocks.into_iter().flatten().collect()
}

/// Convenience function: reorder vectors in-place
pub fn reorder_vectors_inplace(
    vectors: &mut [f32],
    dim: usize,
    strategy: OrderingStrategy,
) -> ReorderResult {
    let result = compute_ordering(vectors, dim, strategy);
    let reordered = result.apply_to_vectors(vectors, dim);
    vectors.copy_from_slice(&reordered);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_projection() {
        let n = 100;
        let d = 16;
        let vectors: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        
        let result = compute_ordering(&vectors, d, OrderingStrategy::RandomProjection);
        
        assert_eq!(result.permutation.len(), n);
        // All indices should be present
        let mut sorted = result.permutation.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }
    
    #[test]
    fn test_coarse_kmeans() {
        let n = 100;
        let d = 16;
        let vectors: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.01).collect();
        
        let k = (n as f64).sqrt() as usize;
        let result = compute_ordering(&vectors, d, OrderingStrategy::CoarseKMeans { k });
        
        assert_eq!(result.permutation.len(), n);
    }
    
    #[test]
    fn test_apply_permutation() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 vectors, dim=2
        let result = ReorderResult {
            permutation: vec![2, 0, 1], // reverse-ish order
            compute_time: std::time::Duration::ZERO,
            strategy: OrderingStrategy::None,
        };
        
        let reordered = result.apply_to_vectors(&vectors, 2);
        assert_eq!(reordered, vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
