// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! AoSoA (Array of Structures of Arrays) Vector Tiles
//!
//! This module provides cache-line aligned vector storage with a tiled layout
//! optimized for SIMD operations and minimal cache misses.
//!
//! # Layout
//!
//! Traditional AoS (Array of Structures):
//! ```text
//! [v0_d0, v0_d1, ..., v0_dN, v1_d0, v1_d1, ..., v1_dN, ...]
//! ```
//!
//! AoSoA with tile size T=4:
//! ```text
//! Tile 0: [v0_d0, v1_d0, v2_d0, v3_d0, v0_d1, v1_d1, v2_d1, v3_d1, ...]
//! Tile 1: [v4_d0, v5_d0, v6_d0, v7_d0, v4_d1, v5_d1, v6_d1, v7_d1, ...]
//! ```
//!
//! # Benefits
//!
//! 1. **SIMD Efficiency**: Process 4/8 vectors simultaneously without gather
//! 2. **Cache Alignment**: Tiles are 64-byte aligned for cache line access
//! 3. **Prefetch Friendly**: Sequential tile access enables hardware prefetch
//! 4. **Batch Distance**: Compute distance to 4 vectors in one pass
//!
//! # Memory Layout
//!
//! For dimension D and tile size T:
//! - Each tile contains T vectors
//! - Tile size in bytes: T × D × sizeof(element)
//! - Tiles are padded to 64-byte boundaries

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::marker::PhantomData;

/// Cache line size (64 bytes on most modern CPUs).
pub const CACHE_LINE_SIZE: usize = 64;

/// Default tile size (number of vectors per tile).
/// 4 works well for AVX2 (256-bit = 8 × f32), NEON (128-bit = 4 × f32).
pub const DEFAULT_TILE_SIZE: usize = 4;

/// Maximum tile size for very wide SIMD (AVX-512).
pub const MAX_TILE_SIZE: usize = 16;

/// AoSoA tile containing a fixed number of vectors.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VectorTile<const T: usize = DEFAULT_TILE_SIZE> {
    /// Interleaved vector data: [v0_d0, v1_d0, ..., vT_d0, v0_d1, v1_d1, ...].
    data: Vec<f32>,
    /// Vector dimension.
    dimension: usize,
}

impl<const T: usize> VectorTile<T> {
    /// Create a new tile for vectors of given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            data: vec![0.0; T * dimension],
            dimension,
        }
    }

    /// Set vector at position within tile.
    #[inline]
    pub fn set_vector(&mut self, pos: usize, vector: &[f32]) {
        debug_assert!(pos < T);
        debug_assert_eq!(vector.len(), self.dimension);
        
        for (d, &v) in vector.iter().enumerate() {
            self.data[d * T + pos] = v;
        }
    }

    /// Get vector at position within tile.
    pub fn get_vector(&self, pos: usize) -> Vec<f32> {
        debug_assert!(pos < T);
        
        (0..self.dimension)
            .map(|d| self.data[d * T + pos])
            .collect()
    }

    /// Get pointer to dimension d values (T consecutive f32s).
    #[inline]
    pub fn dim_ptr(&self, d: usize) -> *const f32 {
        debug_assert!(d < self.dimension);
        self.data[d * T..].as_ptr()
    }

    /// Compute squared L2 distances to all T vectors in tile.
    #[inline]
    pub fn batch_l2_squared(&self, query: &[f32]) -> [f32; T] {
        debug_assert_eq!(query.len(), self.dimension);
        
        let mut dists = [0.0f32; T];
        
        for (d, &q) in query.iter().enumerate() {
            let base = d * T;
            for i in 0..T {
                let diff = q - self.data[base + i];
                dists[i] += diff * diff;
            }
        }
        
        dists
    }

    /// Compute dot products with all T vectors in tile.
    #[inline]
    pub fn batch_dot(&self, query: &[f32]) -> [f32; T] {
        debug_assert_eq!(query.len(), self.dimension);
        
        let mut dots = [0.0f32; T];
        
        for (d, &q) in query.iter().enumerate() {
            let base = d * T;
            for i in 0..T {
                dots[i] += q * self.data[base + i];
            }
        }
        
        dots
    }

    /// Get tile size.
    pub const fn tile_size() -> usize {
        T
    }

    /// Get dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Aligned memory block for cache-line efficiency.
pub struct AlignedBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedBlock {
    /// Allocate a new aligned block.
    pub fn new(size: usize, alignment: usize) -> Option<Self> {
        if size == 0 {
            return None;
        }

        let layout = Layout::from_size_align(size, alignment).ok()?;
        
        // SAFETY: layout is valid and non-zero
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)?;
        
        Some(Self { ptr, layout, len: size })
    }

    /// Get pointer to data.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get mutable pointer to data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for AlignedBlock {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// SAFETY: AlignedBlock is just a raw pointer with size info
unsafe impl Send for AlignedBlock {}
unsafe impl Sync for AlignedBlock {}

/// AoSoA vector storage with tiled layout.
pub struct TiledVectorStore<const T: usize = DEFAULT_TILE_SIZE> {
    /// Aligned storage for tiles.
    storage: Option<AlignedBlock>,
    /// Vector dimension.
    dimension: usize,
    /// Number of vectors.
    count: usize,
    /// Number of tiles.
    num_tiles: usize,
    /// Bytes per tile.
    tile_bytes: usize,
    /// Phantom for const generic.
    _phantom: PhantomData<[(); T]>,
}

impl<const T: usize> TiledVectorStore<T> {
    /// Create a new tiled vector store.
    pub fn new(dimension: usize, capacity: usize) -> Self {
        if capacity == 0 || dimension == 0 {
            return Self {
                storage: None,
                dimension,
                count: 0,
                num_tiles: 0,
                tile_bytes: 0,
                _phantom: PhantomData,
            };
        }

        let num_tiles = (capacity + T - 1) / T;
        let tile_floats = T * dimension;
        let tile_bytes = tile_floats * std::mem::size_of::<f32>();
        
        // Align tile size to cache line
        let aligned_tile_bytes = (tile_bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;
        let total_bytes = num_tiles * aligned_tile_bytes;

        let storage = AlignedBlock::new(total_bytes, CACHE_LINE_SIZE);

        Self {
            storage,
            dimension,
            count: 0,
            num_tiles,
            tile_bytes: aligned_tile_bytes,
            _phantom: PhantomData,
        }
    }

    /// Add a vector, returns its index.
    pub fn push(&mut self, vector: &[f32]) -> Option<usize> {
        if vector.len() != self.dimension {
            return None;
        }

        let idx = self.count;
        let tile_idx = idx / T;
        let pos_in_tile = idx % T;

        if tile_idx >= self.num_tiles {
            return None; // At capacity
        }

        // Write vector in AoSoA layout
        if let Some(ref mut storage) = self.storage {
            let tile_ptr = unsafe { 
                storage.as_mut_ptr().add(tile_idx * self.tile_bytes) as *mut f32 
            };

            for (d, &v) in vector.iter().enumerate() {
                unsafe {
                    *tile_ptr.add(d * T + pos_in_tile) = v;
                }
            }
        }

        self.count += 1;
        Some(idx)
    }

    /// Get a vector by index.
    pub fn get(&self, idx: usize) -> Option<Vec<f32>> {
        if idx >= self.count {
            return None;
        }

        let tile_idx = idx / T;
        let pos_in_tile = idx % T;

        let storage = self.storage.as_ref()?;
        let tile_ptr = unsafe { 
            storage.as_ptr().add(tile_idx * self.tile_bytes) as *const f32 
        };

        let vector: Vec<f32> = (0..self.dimension)
            .map(|d| unsafe { *tile_ptr.add(d * T + pos_in_tile) })
            .collect();

        Some(vector)
    }

    /// Get pointer to a specific tile.
    #[inline]
    pub fn tile_ptr(&self, tile_idx: usize) -> Option<*const f32> {
        if tile_idx >= self.num_tiles {
            return None;
        }

        let storage = self.storage.as_ref()?;
        Some(unsafe { 
            storage.as_ptr().add(tile_idx * self.tile_bytes) as *const f32 
        })
    }

    /// Compute batch distances to vectors at given indices.
    pub fn batch_l2_squared(&self, query: &[f32], indices: &[usize]) -> Vec<f32> {
        if query.len() != self.dimension {
            return vec![];
        }

        indices.iter().map(|&idx| {
            if idx >= self.count {
                return f32::MAX;
            }

            let tile_idx = idx / T;
            let pos_in_tile = idx % T;
            
            let Some(tile_ptr) = self.tile_ptr(tile_idx) else {
                return f32::MAX;
            };

            let mut dist = 0.0f32;
            for (d, &q) in query.iter().enumerate() {
                let v = unsafe { *tile_ptr.add(d * T + pos_in_tile) };
                let diff = q - v;
                dist += diff * diff;
            }
            dist
        }).collect()
    }

    /// Compute distances to all vectors in a tile.
    #[inline]
    pub fn tile_l2_squared(&self, query: &[f32], tile_idx: usize) -> [f32; T] {
        let mut dists = [f32::MAX; T];
        
        if query.len() != self.dimension || tile_idx >= self.num_tiles {
            return dists;
        }

        let Some(tile_ptr) = self.tile_ptr(tile_idx) else {
            return dists;
        };

        // Reset to zero for valid slots
        let valid_count = (self.count.saturating_sub(tile_idx * T)).min(T);
        for i in 0..valid_count {
            dists[i] = 0.0;
        }

        for (d, &q) in query.iter().enumerate() {
            let base_ptr = unsafe { tile_ptr.add(d * T) };
            for i in 0..valid_count {
                let v = unsafe { *base_ptr.add(i) };
                let diff = q - v;
                dists[i] += diff * diff;
            }
        }

        dists
    }

    /// SIMD-optimized tile distance computation (AVX2).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn tile_l2_squared_avx2(&self, query: &[f32], tile_idx: usize) -> [f32; T] {
        use std::arch::x86_64::*;

        let mut dists = [f32::MAX; T];
        
        if query.len() != self.dimension || tile_idx >= self.num_tiles {
            return dists;
        }

        let Some(tile_ptr) = self.tile_ptr(tile_idx) else {
            return dists;
        };

        // Initialize accumulators (assume T=4 for AVX2 with 128-bit chunks)
        let mut acc = _mm_setzero_ps();

        for (d, &q) in query.iter().enumerate() {
            let q_vec = _mm_set1_ps(q);
            let v_vec = _mm_loadu_ps(tile_ptr.add(d * T));
            let diff = _mm_sub_ps(q_vec, v_vec);
            acc = _mm_fmadd_ps(diff, diff, acc);
        }

        // Store results
        _mm_storeu_ps(dists.as_mut_ptr(), acc);
        
        // Mark invalid slots
        let valid_count = (self.count.saturating_sub(tile_idx * T)).min(T);
        for i in valid_count..T {
            dists[i] = f32::MAX;
        }

        dists
    }

    /// SIMD-optimized tile distance computation (NEON).
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn tile_l2_squared_neon(&self, query: &[f32], tile_idx: usize) -> [f32; T] {
        use std::arch::aarch64::*;

        let mut dists = [f32::MAX; T];
        
        if query.len() != self.dimension || tile_idx >= self.num_tiles {
            return dists;
        }

        let Some(tile_ptr) = self.tile_ptr(tile_idx) else {
            return dists;
        };

        // Initialize accumulator (NEON: 128-bit = 4 × f32)
        // SAFETY: We're in an unsafe function with NEON available on aarch64
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for (d, &q) in query.iter().enumerate() {
            // SAFETY: NEON intrinsics are safe with valid pointers
            unsafe {
                let q_vec = vdupq_n_f32(q);
                let v_vec = vld1q_f32(tile_ptr.add(d * T));
                let diff = vsubq_f32(q_vec, v_vec);
                acc = vfmaq_f32(acc, diff, diff);
            }
        }

        // Store results
        // SAFETY: dists is a valid array with enough space
        unsafe { vst1q_f32(dists.as_mut_ptr(), acc) };
        
        // Mark invalid slots
        let valid_count = (self.count.saturating_sub(tile_idx * T)).min(T);
        for i in valid_count..T {
            dists[i] = f32::MAX;
        }

        dists
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of tiles.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Tile size.
    pub const fn tile_size() -> usize {
        T
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.storage.as_ref().map_or(0, |s| s.len())
    }

    /// Capacity (max vectors).
    pub fn capacity(&self) -> usize {
        self.num_tiles * T
    }
}

/// Statistics about tiled storage.
#[derive(Debug, Clone)]
pub struct TiledStoreStats {
    /// Number of vectors.
    pub count: usize,
    /// Number of tiles.
    pub num_tiles: usize,
    /// Bytes per tile.
    pub tile_bytes: usize,
    /// Total memory bytes.
    pub total_bytes: usize,
    /// Tile utilization (0.0 to 1.0).
    pub utilization: f64,
    /// Bytes per vector (effective).
    pub bytes_per_vector: f64,
}

impl<const T: usize> TiledVectorStore<T> {
    /// Get storage statistics.
    pub fn stats(&self) -> TiledStoreStats {
        let total_bytes = self.memory_bytes();
        let utilization = if self.num_tiles > 0 {
            self.count as f64 / (self.num_tiles * T) as f64
        } else {
            0.0
        };
        let bytes_per_vector = if self.count > 0 {
            total_bytes as f64 / self.count as f64
        } else {
            0.0
        };

        TiledStoreStats {
            count: self.count,
            num_tiles: self.num_tiles,
            tile_bytes: self.tile_bytes,
            total_bytes,
            utilization,
            bytes_per_vector,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_tile_basic() {
        let mut tile: VectorTile<4> = VectorTile::new(3);
        
        tile.set_vector(0, &[1.0, 2.0, 3.0]);
        tile.set_vector(1, &[4.0, 5.0, 6.0]);
        tile.set_vector(2, &[7.0, 8.0, 9.0]);
        tile.set_vector(3, &[10.0, 11.0, 12.0]);
        
        assert_eq!(tile.get_vector(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(tile.get_vector(1), vec![4.0, 5.0, 6.0]);
        assert_eq!(tile.get_vector(2), vec![7.0, 8.0, 9.0]);
        assert_eq!(tile.get_vector(3), vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_tile_batch_distance() {
        let mut tile: VectorTile<4> = VectorTile::new(3);
        
        tile.set_vector(0, &[1.0, 0.0, 0.0]);
        tile.set_vector(1, &[0.0, 1.0, 0.0]);
        tile.set_vector(2, &[0.0, 0.0, 1.0]);
        tile.set_vector(3, &[1.0, 1.0, 1.0]);
        
        let query = [0.0, 0.0, 0.0];
        let dists = tile.batch_l2_squared(&query);
        
        assert!((dists[0] - 1.0).abs() < 1e-6);
        assert!((dists[1] - 1.0).abs() < 1e-6);
        assert!((dists[2] - 1.0).abs() < 1e-6);
        assert!((dists[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_tile_batch_dot() {
        let mut tile: VectorTile<4> = VectorTile::new(3);
        
        tile.set_vector(0, &[1.0, 0.0, 0.0]);
        tile.set_vector(1, &[0.0, 1.0, 0.0]);
        tile.set_vector(2, &[0.0, 0.0, 1.0]);
        tile.set_vector(3, &[1.0, 1.0, 1.0]);
        
        let query = [1.0, 1.0, 1.0];
        let dots = tile.batch_dot(&query);
        
        assert!((dots[0] - 1.0).abs() < 1e-6);
        assert!((dots[1] - 1.0).abs() < 1e-6);
        assert!((dots[2] - 1.0).abs() < 1e-6);
        assert!((dots[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_tiled_store_basic() {
        let mut store: TiledVectorStore<4> = TiledVectorStore::new(128, 100);
        
        // Add some vectors
        for i in 0..50 {
            let v: Vec<f32> = (0..128).map(|d| (i * 128 + d) as f32).collect();
            assert_eq!(store.push(&v), Some(i));
        }
        
        assert_eq!(store.len(), 50);
        assert_eq!(store.dimension(), 128);
        
        // Verify retrieval
        let v0 = store.get(0).unwrap();
        assert_eq!(v0.len(), 128);
        assert!((v0[0] - 0.0).abs() < 1e-6);
        assert!((v0[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tiled_store_distance() {
        let mut store: TiledVectorStore<4> = TiledVectorStore::new(3, 8);
        
        store.push(&[1.0, 0.0, 0.0]).unwrap();
        store.push(&[0.0, 1.0, 0.0]).unwrap();
        store.push(&[0.0, 0.0, 1.0]).unwrap();
        store.push(&[1.0, 1.0, 1.0]).unwrap();
        
        let query = vec![0.0, 0.0, 0.0];
        let dists = store.batch_l2_squared(&query, &[0, 1, 2, 3]);
        
        assert!((dists[0] - 1.0).abs() < 1e-6);
        assert!((dists[1] - 1.0).abs() < 1e-6);
        assert!((dists[2] - 1.0).abs() < 1e-6);
        assert!((dists[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_tiled_store_tile_distance() {
        let mut store: TiledVectorStore<4> = TiledVectorStore::new(3, 8);
        
        store.push(&[1.0, 0.0, 0.0]).unwrap();
        store.push(&[0.0, 1.0, 0.0]).unwrap();
        store.push(&[0.0, 0.0, 1.0]).unwrap();
        store.push(&[1.0, 1.0, 1.0]).unwrap();
        
        let query = vec![0.0, 0.0, 0.0];
        let dists = store.tile_l2_squared(&query, 0);
        
        assert!((dists[0] - 1.0).abs() < 1e-6);
        assert!((dists[1] - 1.0).abs() < 1e-6);
        assert!((dists[2] - 1.0).abs() < 1e-6);
        assert!((dists[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aligned_block() {
        let block = AlignedBlock::new(256, 64).unwrap();
        
        assert_eq!(block.len(), 256);
        assert_eq!(block.as_ptr() as usize % 64, 0); // Check alignment
    }

    #[test]
    fn test_store_stats() {
        let mut store: TiledVectorStore<4> = TiledVectorStore::new(128, 100);
        
        for i in 0..50 {
            let v: Vec<f32> = (0..128).map(|d| (i * 128 + d) as f32).collect();
            store.push(&v).unwrap();
        }
        
        let stats = store.stats();
        assert_eq!(stats.count, 50);
        assert!(stats.num_tiles > 0);
        assert!(stats.total_bytes > 0);
        assert!(stats.utilization > 0.0 && stats.utilization <= 1.0);
    }

    #[test]
    fn test_empty_store() {
        let store: TiledVectorStore<4> = TiledVectorStore::new(128, 0);
        
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.memory_bytes(), 0);
    }

    #[test]
    fn test_partial_tile() {
        let mut store: TiledVectorStore<4> = TiledVectorStore::new(3, 10);
        
        // Add only 2 vectors (partial first tile)
        store.push(&[1.0, 2.0, 3.0]).unwrap();
        store.push(&[4.0, 5.0, 6.0]).unwrap();
        
        let query = vec![0.0, 0.0, 0.0];
        let dists = store.tile_l2_squared(&query, 0);
        
        // First two should be valid
        assert!((dists[0] - 14.0).abs() < 1e-6); // 1^2 + 2^2 + 3^2 = 14
        assert!((dists[1] - 77.0).abs() < 1e-6); // 4^2 + 5^2 + 6^2 = 77
        
        // Last two should be MAX (invalid)
        assert_eq!(dists[2], f32::MAX);
        assert_eq!(dists[3], f32::MAX);
    }
}
