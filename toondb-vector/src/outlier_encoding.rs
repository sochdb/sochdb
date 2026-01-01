// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Optimized Outlier Encoding
//!
//! This module provides efficient outlier representation that avoids
//! `.contains()` calls in hot loops by using:
//!
//! - **Bitvector**: O(1) membership for dense outliers (k/D > threshold)
//! - **Sorted list + binary search**: O(log k) for sparse outliers
//!
//! # Problem
//!
//! Original approach:
//! ```text
//! for dim in 0..D {
//!     if outliers.contains(dim) {  // O(k) linear scan!
//!         use_outlier_value(dim);
//!     } else {
//!         use_quantized_value(dim);
//!     }
//! }
//! ```
//!
//! This is poison in tight loops: O(D × k) per vector decode.
//!
//! # Solution
//!
//! Hybrid representation with automatic crossover:
//!
//! ```text
//! if k/D > BITVEC_THRESHOLD {
//!     // Dense: use bitvector, O(1) per check
//!     bitvec.test(dim)
//! } else {
//!     // Sparse: sorted list + binary search, O(log k) per check
//!     sorted_dims.binary_search(&dim).is_ok()
//! }
//! ```
//!
//! # Crossover Analysis
//!
//! For dimension D and k outliers:
//! - Bitvector: D/8 bytes storage, O(1) access
//! - Sorted list: k × 2 bytes storage (u16), O(log k) access
//!
//! Crossover when: k × 2 ≈ D/8, i.e., k ≈ D/16
//! For D=768, crossover at k ≈ 48 outliers.

/// Threshold for switching from sorted list to bitvector.
/// When k/D > this ratio, use bitvector.
pub const BITVEC_THRESHOLD: f32 = 0.0625; // 1/16 = 6.25%

/// Maximum dimension supported (u16 indices).
pub const MAX_DIMENSION: usize = 65536;

/// Optimized outlier set with hybrid representation.
#[derive(Debug, Clone)]
pub enum OutlierSet {
    /// Empty set (no outliers).
    Empty,
    /// Sparse representation: sorted list of dimension indices.
    Sparse(SparseOutliers),
    /// Dense representation: bitvector membership.
    Dense(DenseOutliers),
}

impl OutlierSet {
    /// Create empty outlier set.
    pub fn empty() -> Self {
        OutlierSet::Empty
    }

    /// Create from dimension indices, automatically choosing representation.
    pub fn from_dims(dims: &[u16], dimension: usize) -> Self {
        if dims.is_empty() {
            return OutlierSet::Empty;
        }

        let density = dims.len() as f32 / dimension as f32;
        
        if density > BITVEC_THRESHOLD {
            OutlierSet::Dense(DenseOutliers::from_dims(dims, dimension))
        } else {
            OutlierSet::Sparse(SparseOutliers::from_dims(dims))
        }
    }

    /// Create from iterator of dimension indices.
    pub fn from_iter<I: IntoIterator<Item = u16>>(iter: I, dimension: usize) -> Self {
        let dims: Vec<u16> = iter.into_iter().collect();
        Self::from_dims(&dims, dimension)
    }

    /// Check if dimension is an outlier. O(1) for dense, O(log k) for sparse.
    #[inline]
    pub fn contains(&self, dim: u16) -> bool {
        match self {
            OutlierSet::Empty => false,
            OutlierSet::Sparse(s) => s.contains(dim),
            OutlierSet::Dense(d) => d.contains(dim),
        }
    }

    /// Number of outliers.
    pub fn len(&self) -> usize {
        match self {
            OutlierSet::Empty => 0,
            OutlierSet::Sparse(s) => s.len(),
            OutlierSet::Dense(d) => d.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        matches!(self, OutlierSet::Empty)
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            OutlierSet::Empty => 0,
            OutlierSet::Sparse(s) => s.memory_bytes(),
            OutlierSet::Dense(d) => d.memory_bytes(),
        }
    }

    /// Iterate over outlier dimensions.
    pub fn iter(&self) -> OutlierIterator<'_> {
        match self {
            OutlierSet::Empty => OutlierIterator::Empty,
            OutlierSet::Sparse(s) => OutlierIterator::Sparse(s.dims.iter()),
            OutlierSet::Dense(d) => OutlierIterator::Dense(DenseIterator::new(d)),
        }
    }

    /// Get density ratio.
    pub fn density(&self, dimension: usize) -> f32 {
        self.len() as f32 / dimension as f32
    }

    /// Check representation type.
    pub fn is_dense(&self) -> bool {
        matches!(self, OutlierSet::Dense(_))
    }
}

/// Sparse outlier representation using sorted list.
#[derive(Debug, Clone)]
pub struct SparseOutliers {
    /// Sorted dimension indices.
    dims: Vec<u16>,
}

impl SparseOutliers {
    /// Create from dimension indices (will sort).
    pub fn from_dims(dims: &[u16]) -> Self {
        let mut sorted = dims.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self { dims: sorted }
    }

    /// Check membership using binary search. O(log k).
    #[inline]
    pub fn contains(&self, dim: u16) -> bool {
        self.dims.binary_search(&dim).is_ok()
    }

    /// Get position of dimension in sorted list (for value lookup).
    #[inline]
    pub fn position(&self, dim: u16) -> Option<usize> {
        self.dims.binary_search(&dim).ok()
    }

    /// Number of outliers.
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.dims.len() * std::mem::size_of::<u16>()
    }

    /// Get raw dimensions slice.
    pub fn dims(&self) -> &[u16] {
        &self.dims
    }
}

/// Dense outlier representation using bitvector.
#[derive(Debug, Clone)]
pub struct DenseOutliers {
    /// Bitvector (1 bit per dimension).
    bits: Vec<u64>,
    /// Number of set bits (outliers).
    count: usize,
}

impl DenseOutliers {
    /// Create from dimension indices.
    pub fn from_dims(dims: &[u16], dimension: usize) -> Self {
        let num_words = (dimension + 63) / 64;
        let mut bits = vec![0u64; num_words];
        
        for &dim in dims {
            let word_idx = dim as usize / 64;
            let bit_idx = dim as usize % 64;
            if word_idx < bits.len() {
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }

        // Count unique set bits
        let count = bits.iter().map(|w| w.count_ones() as usize).sum();

        Self { bits, count }
    }

    /// Check membership. O(1).
    #[inline]
    pub fn contains(&self, dim: u16) -> bool {
        let word_idx = dim as usize / 64;
        let bit_idx = dim as usize % 64;
        
        if word_idx >= self.bits.len() {
            return false;
        }
        
        (self.bits[word_idx] >> bit_idx) & 1 == 1
    }

    /// Get the rank (count of set bits before this position).
    /// Useful for value lookup in compressed storage.
    #[inline]
    pub fn rank(&self, dim: u16) -> usize {
        let word_idx = dim as usize / 64;
        let bit_idx = dim as usize % 64;

        let mut count = 0usize;
        
        // Count all bits in previous words
        for i in 0..word_idx.min(self.bits.len()) {
            count += self.bits[i].count_ones() as usize;
        }

        // Count bits before position in current word
        if word_idx < self.bits.len() {
            let mask = (1u64 << bit_idx) - 1;
            count += (self.bits[word_idx] & mask).count_ones() as usize;
        }

        count
    }

    /// Number of outliers.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>()
    }

    /// Get dimension capacity.
    pub fn capacity(&self) -> usize {
        self.bits.len() * 64
    }
}

/// Iterator over outlier dimensions.
pub enum OutlierIterator<'a> {
    Empty,
    Sparse(std::slice::Iter<'a, u16>),
    Dense(DenseIterator<'a>),
}

impl<'a> Iterator for OutlierIterator<'a> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            OutlierIterator::Empty => None,
            OutlierIterator::Sparse(iter) => iter.next().copied(),
            OutlierIterator::Dense(iter) => iter.next(),
        }
    }
}

/// Iterator over dense bitvector.
pub struct DenseIterator<'a> {
    bits: &'a [u64],
    word_idx: usize,
    #[allow(dead_code)]
    bit_idx: u32,
    remaining: u64,
}

impl<'a> DenseIterator<'a> {
    fn new(dense: &'a DenseOutliers) -> Self {
        let remaining = dense.bits.first().copied().unwrap_or(0);
        Self {
            bits: &dense.bits,
            word_idx: 0,
            bit_idx: 0,
            remaining,
        }
    }
}

impl Iterator for DenseIterator<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining != 0 {
                let tz = self.remaining.trailing_zeros();
                self.remaining &= self.remaining - 1; // Clear lowest set bit
                return Some((self.word_idx * 64 + tz as usize) as u16);
            }

            self.word_idx += 1;
            if self.word_idx >= self.bits.len() {
                return None;
            }
            self.remaining = self.bits[self.word_idx];
        }
    }
}

/// Outlier entry with dimension and value.
#[derive(Debug, Clone, Copy)]
pub struct OutlierValue {
    /// Dimension index.
    pub dim: u16,
    /// Original f32 value.
    pub value: f32,
}

/// Complete outlier storage for a vector.
#[derive(Debug, Clone)]
pub struct OutlierStorage {
    /// Outlier set for membership testing.
    pub set: OutlierSet,
    /// Outlier values (same order as set iteration).
    pub values: Vec<f32>,
}

impl OutlierStorage {
    /// Create empty storage.
    pub fn empty() -> Self {
        Self {
            set: OutlierSet::Empty,
            values: Vec::new(),
        }
    }

    /// Create from entries.
    pub fn from_entries(entries: &[OutlierValue], dimension: usize) -> Self {
        if entries.is_empty() {
            return Self::empty();
        }

        // Sort by dimension
        let mut sorted: Vec<_> = entries.to_vec();
        sorted.sort_by_key(|e| e.dim);

        let dims: Vec<u16> = sorted.iter().map(|e| e.dim).collect();
        let values: Vec<f32> = sorted.iter().map(|e| e.value).collect();

        Self {
            set: OutlierSet::from_dims(&dims, dimension),
            values,
        }
    }

    /// Get outlier value for dimension, if it exists.
    #[inline]
    pub fn get(&self, dim: u16) -> Option<f32> {
        match &self.set {
            OutlierSet::Empty => None,
            OutlierSet::Sparse(s) => {
                s.position(dim).map(|pos| self.values[pos])
            }
            OutlierSet::Dense(d) => {
                if d.contains(dim) {
                    Some(self.values[d.rank(dim)])
                } else {
                    None
                }
            }
        }
    }

    /// Check if dimension is an outlier.
    #[inline]
    pub fn contains(&self, dim: u16) -> bool {
        self.set.contains(dim)
    }

    /// Number of outliers.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.set.memory_bytes() + self.values.len() * std::mem::size_of::<f32>()
    }

    /// Iterate over (dimension, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u16, f32)> + '_ {
        self.set.iter().zip(self.values.iter().copied())
    }
}

/// Batch outlier lookup for SIMD-friendly decode.
pub struct BatchOutlierLookup<'a> {
    storage: &'a OutlierStorage,
}

impl<'a> BatchOutlierLookup<'a> {
    /// Create a batch lookup helper.
    pub fn new(storage: &'a OutlierStorage) -> Self {
        Self { storage }
    }

    /// Lookup multiple dimensions at once, returning mask and values.
    /// For dimensions that are not outliers, value is 0.0.
    pub fn lookup_batch(&self, dims: &[u16]) -> (Vec<bool>, Vec<f32>) {
        let mut is_outlier = Vec::with_capacity(dims.len());
        let mut values = Vec::with_capacity(dims.len());

        for &dim in dims {
            if let Some(v) = self.storage.get(dim) {
                is_outlier.push(true);
                values.push(v);
            } else {
                is_outlier.push(false);
                values.push(0.0);
            }
        }

        (is_outlier, values)
    }

    /// Lookup 4 dimensions at once (SIMD-friendly).
    #[inline]
    pub fn lookup_4(&self, dims: [u16; 4]) -> ([bool; 4], [f32; 4]) {
        let mut is_outlier = [false; 4];
        let mut values = [0.0f32; 4];

        for i in 0..4 {
            if let Some(v) = self.storage.get(dims[i]) {
                is_outlier[i] = true;
                values[i] = v;
            }
        }

        (is_outlier, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_outlier_set() {
        let set = OutlierSet::empty();
        
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(0));
        assert!(!set.contains(100));
    }

    #[test]
    fn test_sparse_outliers() {
        let dims = vec![5, 10, 100, 200];
        let set = OutlierSet::from_dims(&dims, 768);
        
        assert!(!set.is_dense());
        assert_eq!(set.len(), 4);
        
        assert!(set.contains(5));
        assert!(set.contains(10));
        assert!(set.contains(100));
        assert!(set.contains(200));
        
        assert!(!set.contains(0));
        assert!(!set.contains(6));
        assert!(!set.contains(99));
    }

    #[test]
    fn test_dense_outliers() {
        // Create enough outliers to trigger dense representation
        let dims: Vec<u16> = (0..100).collect(); // 100 outliers for D=768 → ~13%
        let set = OutlierSet::from_dims(&dims, 768);
        
        assert!(set.is_dense());
        assert_eq!(set.len(), 100);
        
        for d in 0..100 {
            assert!(set.contains(d));
        }
        assert!(!set.contains(100));
        assert!(!set.contains(500));
    }

    #[test]
    fn test_sparse_binary_search() {
        let sparse = SparseOutliers::from_dims(&[10, 20, 30, 40, 50]);
        
        assert_eq!(sparse.position(10), Some(0));
        assert_eq!(sparse.position(30), Some(2));
        assert_eq!(sparse.position(50), Some(4));
        assert_eq!(sparse.position(15), None);
    }

    #[test]
    fn test_dense_rank() {
        let dense = DenseOutliers::from_dims(&[0, 1, 2, 10, 20], 768);
        
        assert_eq!(dense.rank(0), 0); // No bits before position 0
        assert_eq!(dense.rank(1), 1); // One bit (position 0)
        assert_eq!(dense.rank(2), 2); // Two bits (0, 1)
        assert_eq!(dense.rank(10), 3); // Three bits (0, 1, 2)
        assert_eq!(dense.rank(20), 4); // Four bits (0, 1, 2, 10)
    }

    #[test]
    fn test_outlier_storage() {
        let entries = vec![
            OutlierValue { dim: 10, value: 1.5 },
            OutlierValue { dim: 20, value: -2.0 },
            OutlierValue { dim: 5, value: 0.5 },
        ];
        
        let storage = OutlierStorage::from_entries(&entries, 768);
        
        assert_eq!(storage.len(), 3);
        assert!(storage.contains(5));
        assert!(storage.contains(10));
        assert!(storage.contains(20));
        assert!(!storage.contains(15));
        
        assert_eq!(storage.get(5), Some(0.5));
        assert_eq!(storage.get(10), Some(1.5));
        assert_eq!(storage.get(20), Some(-2.0));
        assert_eq!(storage.get(15), None);
    }

    #[test]
    fn test_batch_lookup() {
        let entries = vec![
            OutlierValue { dim: 0, value: 1.0 },
            OutlierValue { dim: 2, value: 2.0 },
        ];
        let storage = OutlierStorage::from_entries(&entries, 768);
        let lookup = BatchOutlierLookup::new(&storage);
        
        let (is_outlier, values) = lookup.lookup_4([0, 1, 2, 3]);
        
        assert_eq!(is_outlier, [true, false, true, false]);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 2.0);
        assert_eq!(values[3], 0.0);
    }

    #[test]
    fn test_outlier_iteration() {
        let dims = vec![5, 10, 100];
        let set = OutlierSet::from_dims(&dims, 768);
        
        let collected: Vec<u16> = set.iter().collect();
        assert_eq!(collected, vec![5, 10, 100]);
    }

    #[test]
    fn test_dense_iteration() {
        let dims: Vec<u16> = (0..100).collect();
        let set = OutlierSet::from_dims(&dims, 768);
        
        let collected: Vec<u16> = set.iter().collect();
        assert_eq!(collected.len(), 100);
        assert_eq!(collected[0], 0);
        assert_eq!(collected[99], 99);
    }

    #[test]
    fn test_crossover_threshold() {
        // Just below threshold: should be sparse
        let sparse_dims: Vec<u16> = (0..40).collect(); // ~5% for D=768
        let sparse_set = OutlierSet::from_dims(&sparse_dims, 768);
        assert!(!sparse_set.is_dense());
        
        // Above threshold: should be dense
        let dense_dims: Vec<u16> = (0..60).collect(); // ~8% for D=768
        let dense_set = OutlierSet::from_dims(&dense_dims, 768);
        assert!(dense_set.is_dense());
    }

    #[test]
    fn test_memory_efficiency() {
        let dimension = 768;
        
        // Sparse: 10 outliers
        let sparse_dims: Vec<u16> = (0..10).collect();
        let sparse_set = OutlierSet::from_dims(&sparse_dims, dimension);
        assert!(sparse_set.memory_bytes() < 100); // ~20 bytes
        
        // Dense: 100 outliers
        let dense_dims: Vec<u16> = (0..100).collect();
        let dense_set = OutlierSet::from_dims(&dense_dims, dimension);
        // Bitvector: 768/64 = 12 words × 8 bytes = 96 bytes
        assert!(dense_set.memory_bytes() <= 100);
    }

    #[test]
    fn test_unsorted_input() {
        let dims = vec![100, 5, 50, 10, 200];
        let set = OutlierSet::from_dims(&dims, 768);
        
        // Should work regardless of input order
        for &d in &dims {
            assert!(set.contains(d));
        }
    }

    #[test]
    fn test_duplicate_dims() {
        let dims = vec![5, 5, 10, 10, 10];
        let set = OutlierSet::from_dims(&dims, 768);
        
        // Should deduplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(5));
        assert!(set.contains(10));
    }
}
