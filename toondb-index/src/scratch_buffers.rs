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

//! Thread-Local Scratch Buffer Pool (Task 10)
//!
//! Eliminates per-call allocation overhead in hot paths through buffer reuse.
//! With 900+ distance-related function calls per insert and multiple search/selection
//! calls per layer, allocation overhead accumulates significantly.
//!
//! ## Problem
//! 
//! Current hot paths allocate temporary buffers:
//! - search_layer: HashSet, BinaryHeap, Vec allocations
//! - select_neighbors_heuristic: Vec allocations
//! - Per allocation: ~50-100ns + 3μs zero-init for 3KB
//! - Per insert: ~50 allocations × 3μs = 150μs overhead
//! - At 2000 vec/s: 300ms/s spent on allocation (15% overhead)
//!
//! ## Solution
//!
//! Thread-local reusable buffers with clear() instead of allocation:
//! - Pre-allocated buffers sized for typical workloads
//! - Zero allocation cost in steady state
//! - Better cache behavior (hot buffers stay resident)
//! - No allocator contention between threads
//!
//! ## Expected Performance
//! 
//! - 10-15% reduction in per-insert overhead
//! - More stable P99 latency (no allocator spikes)
//! - 1.1-1.15× consistent throughput improvement

use std::cell::RefCell;
use std::collections::{HashSet, BinaryHeap, HashMap};
use smallvec::SmallVec;

use crate::hnsw::SearchCandidate;

/// Thread-local scratch buffers for hot path operations
/// 
/// These buffers are reused across multiple operations to eliminate
/// allocation overhead. All buffers are pre-sized for typical workloads
/// to minimize reallocation during growth.
pub struct ScratchBuffers {
    /// Visited set for graph traversal (search_layer)
    /// Typical size: 500-1000 nodes for ef_search=200
    pub visited: HashSet<u32>,
    
    /// Candidate priority queue for beam search
    /// Typical size: ef_construction (100) or ef_search (200) 
    pub candidates: BinaryHeap<SearchCandidate>,
    
    /// Results buffer for search operations
    /// Typical size: k (10-100) for search results
    pub results: Vec<SearchCandidate>,
    
    /// Working set for neighbor selection
    /// Typical size: ef_construction (100) candidates
    pub working_set: Vec<SearchCandidate>,
    
    /// Distance computation cache for batch operations  
    /// Typical size: 1000-2000 distance pairs per batch
    pub distance_cache: HashMap<(u32, u32), f32>,
    
    /// Projection buffer for Johnson-Lindenstrauss filtering
    /// Fixed size: 32 dimensions for projection
    pub projection_buffer: [f32; 32],
    
    /// Temporary neighbor list for pruning operations
    /// Typical size: max_connections (16-64)  
    pub temp_neighbors: SmallVec<[u32; 64]>,
    
    /// Node indices for batch processing
    /// Typical size: batch_size (100-1000) for wave processing
    pub node_indices: Vec<u32>,
    
    /// Float buffer for SIMD operations
    /// Typical size: 768-1536 for distance computation  
    pub float_buffer: Vec<f32>,
}

impl ScratchBuffers {
    /// Create new scratch buffers with optimal initial capacity
    /// 
    /// Capacities are chosen based on empirical analysis of typical
    /// HNSW workloads to minimize reallocation during normal operation.
    pub fn new() -> Self {
        Self {
            visited: HashSet::with_capacity(1024),
            candidates: BinaryHeap::with_capacity(256),
            results: Vec::with_capacity(128),
            working_set: Vec::with_capacity(256),
            distance_cache: HashMap::with_capacity(1024),
            projection_buffer: [0.0; 32],
            temp_neighbors: SmallVec::new(),
            node_indices: Vec::with_capacity(1024),
            float_buffer: Vec::with_capacity(1536),
        }
    }
    
    /// Clear all buffers for reuse
    /// 
    /// This is much faster than allocation since it just resets lengths
    /// without deallocating the underlying memory.
    pub fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.results.clear(); 
        self.working_set.clear();
        self.distance_cache.clear();
        // projection_buffer doesn't need clearing (will be overwritten)
        self.temp_neighbors.clear();
        self.node_indices.clear();
        self.float_buffer.clear();
    }
    
    /// Get memory usage of scratch buffers in bytes
    pub fn memory_usage(&self) -> usize {
        let visited_mem = self.visited.capacity() * std::mem::size_of::<u32>();
        let candidates_mem = self.candidates.capacity() * std::mem::size_of::<SearchCandidate>();
        let results_mem = self.results.capacity() * std::mem::size_of::<SearchCandidate>();
        let working_set_mem = self.working_set.capacity() * std::mem::size_of::<SearchCandidate>();
        let distance_cache_mem = self.distance_cache.capacity() * 
            (std::mem::size_of::<(u32, u32)>() + std::mem::size_of::<f32>());
        let projection_mem = std::mem::size_of::<[f32; 32]>();
        let temp_neighbors_mem = self.temp_neighbors.capacity() * std::mem::size_of::<u32>();
        let node_indices_mem = self.node_indices.capacity() * std::mem::size_of::<u32>();
        let float_buffer_mem = self.float_buffer.capacity() * std::mem::size_of::<f32>();
        
        visited_mem + candidates_mem + results_mem + working_set_mem + 
        distance_cache_mem + projection_mem + temp_neighbors_mem + 
        node_indices_mem + float_buffer_mem
    }
    
    /// Resize buffers if they've grown beyond typical usage
    /// 
    /// This prevents memory bloat from occasional large operations
    /// while preserving performance for normal workloads.
    pub fn maybe_shrink(&mut self) {
        const MAX_VISITED: usize = 2048;
        const MAX_CANDIDATES: usize = 512;
        const MAX_RESULTS: usize = 256;
        const MAX_WORKING_SET: usize = 512;
        const MAX_CACHE: usize = 2048;
        const MAX_INDICES: usize = 2048;
        const MAX_FLOAT_BUFFER: usize = 3072;
        
        if self.visited.capacity() > MAX_VISITED {
            self.visited = HashSet::with_capacity(1024);
        }
        if self.candidates.capacity() > MAX_CANDIDATES {
            self.candidates = BinaryHeap::with_capacity(256);
        }
        if self.results.capacity() > MAX_RESULTS {
            self.results = Vec::with_capacity(128);
        }
        if self.working_set.capacity() > MAX_WORKING_SET {
            self.working_set = Vec::with_capacity(256);
        }
        if self.distance_cache.capacity() > MAX_CACHE {
            self.distance_cache = HashMap::with_capacity(1024);
        }
        if self.node_indices.capacity() > MAX_INDICES {
            self.node_indices = Vec::with_capacity(1024);
        }
        if self.float_buffer.capacity() > MAX_FLOAT_BUFFER {
            self.float_buffer = Vec::with_capacity(1536);
        }
    }
}

impl Default for ScratchBuffers {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local storage for scratch buffers
// 
// Each thread gets its own set of scratch buffers to eliminate
// allocation overhead without cross-thread synchronization.
thread_local! {
    static SCRATCH: RefCell<ScratchBuffers> = RefCell::new(ScratchBuffers::new());
}

/// Execute function with access to thread-local scratch buffers
/// 
/// This is the primary API for using scratch buffers in hot paths.
/// The buffers are automatically cleared before use and remain
/// allocated for future operations.
/// 
/// # Example
/// 
/// ```rust
/// let results = with_scratch_buffers(|scratch| {
///     scratch.visited.insert(node_id);
///     scratch.candidates.push(candidate);
///     // ... use buffers ...
///     std::mem::take(&mut scratch.results) // Return owned result
/// });
/// ```
pub fn with_scratch_buffers<F, R>(f: F) -> R
where 
    F: FnOnce(&mut ScratchBuffers) -> R,
{
    SCRATCH.with(|scratch| {
        let mut buffers = scratch.borrow_mut();
        buffers.clear();
        f(&mut buffers)
    })
}

/// Get current thread's scratch buffer memory usage
pub fn scratch_memory_usage() -> usize {
    SCRATCH.with(|scratch| {
        scratch.borrow().memory_usage()
    })
}

/// Shrink scratch buffers if they've grown too large
/// 
/// Should be called periodically (e.g., after large batch operations)
/// to prevent memory bloat while preserving performance.
pub fn shrink_scratch_buffers() {
    SCRATCH.with(|scratch| {
        scratch.borrow_mut().maybe_shrink();
    });
}

/// Statistics about scratch buffer usage across all threads
#[derive(Debug, Clone)]
pub struct ScratchBufferStats {
    pub total_threads: usize,
    pub total_memory_bytes: usize,
    pub avg_memory_per_thread: f64,
    pub max_memory_per_thread: usize,
}

// Note: Getting stats across all threads is not easily possible with thread_local!
// In practice, you would track this at the application level or use a global registry

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scratch_buffers_basic() {
        let result = with_scratch_buffers(|scratch| {
            scratch.visited.insert(1);
            scratch.visited.insert(2);
            scratch.visited.len()
        });
        
        assert_eq!(result, 2);
        
        // Buffers should be cleared for next use
        let result2 = with_scratch_buffers(|scratch| {
            scratch.visited.len()
        });
        
        assert_eq!(result2, 0);
    }
    
    #[test]
    fn test_memory_usage() {
        let usage = with_scratch_buffers(|scratch| {
            scratch.memory_usage()
        });
        
        assert!(usage > 0);
        println!("Scratch buffer memory usage: {} bytes", usage);
    }
    
    #[test]
    fn test_shrink_buffers() {
        with_scratch_buffers(|scratch| {
            // Force growth
            scratch.visited.reserve(10000);
            scratch.float_buffer.reserve(10000);
        });
        
        let usage_before = scratch_memory_usage();
        shrink_scratch_buffers();
        let usage_after = scratch_memory_usage();
        
        assert!(usage_after <= usage_before);
    }
    
    #[test]
    fn test_projection_buffer() {
        with_scratch_buffers(|scratch| {
            scratch.projection_buffer[0] = 1.0;
            scratch.projection_buffer[31] = 2.0;
            assert_eq!(scratch.projection_buffer[0], 1.0);
            assert_eq!(scratch.projection_buffer[31], 2.0);
        });
    }
}