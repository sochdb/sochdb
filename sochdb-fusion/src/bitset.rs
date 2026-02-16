// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Dense Bit-Vector for Zero-Allocation Candidate Masks
//!
//! The `BitSet` is the key intermediate representation in fused query execution.
//! It flows between index stages (ART → HNSW → CSR) without heap allocation,
//! serialization, or cache flushes.
//!
//! ## Why BitSet, not HashSet or Vec?
//!
//! | Operation          | BitSet (64-bit words) | HashSet<u64>    | Vec<u64>      |
//! |--------------------|----------------------|-----------------|---------------|
//! | Insert             | O(1) — bit set       | O(1) amortized  | O(n) sorted   |
//! | Contains           | O(1) — bit test      | O(1) amortized  | O(log n)      |
//! | Intersection (AND) | O(n/64) — SIMD-able  | O(min(a,b))     | O(a+b) merge  |
//! | Union (OR)         | O(n/64) — SIMD-able  | O(a+b)          | O(a+b) merge  |
//! | Memory (1M items)  | 128 KB               | ~40 MB          | ~8 MB         |
//!
//! The BitSet enables SIMD-parallel set intersection: a single AVX-512 instruction
//! processes 512 candidates simultaneously.
//!
//! ## Cache Efficiency
//!
//! For 1M candidates, the BitSet occupies 128 KB — fits entirely in L2 cache.
//! This means the intersection of an ART filter result with an HNSW candidate
//! set costs ~0 cache misses beyond the initial load.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Dense bit-vector for O(1) membership test and O(n/64) set operations.
///
/// Internally stored as `Vec<u64>` — each u64 word holds 64 bits.
/// The canonical representation keeps trailing zero words but truncates none,
/// enabling O(1) `contains()` without bounds checking the common case.
#[derive(Clone, Serialize, Deserialize)]
pub struct BitSet {
    /// The bit storage. Bit `i` is at `words[i / 64] & (1 << (i % 64))`.
    words: Vec<u64>,
    /// The total capacity in bits (not the number of set bits).
    capacity: usize,
    /// Cached count of set bits (updated lazily).
    count: usize,
}

impl BitSet {
    /// Create an empty BitSet with at least `capacity` bits.
    pub fn with_capacity(capacity: usize) -> Self {
        let num_words = (capacity + 63) / 64;
        Self {
            words: vec![0u64; num_words],
            capacity,
            count: 0,
        }
    }

    /// Create a BitSet with all bits set (universe).
    pub fn all(capacity: usize) -> Self {
        let num_words = (capacity + 63) / 64;
        let mut words = vec![u64::MAX; num_words];
        // Clear any trailing bits beyond capacity
        let trailing = capacity % 64;
        if trailing > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] = (1u64 << trailing) - 1;
        }
        Self {
            words,
            capacity,
            count: capacity,
        }
    }

    /// Create a BitSet from an iterator of set bit positions.
    pub fn from_iter(capacity: usize, iter: impl IntoIterator<Item = usize>) -> Self {
        let mut bs = Self::with_capacity(capacity);
        for bit in iter {
            bs.set(bit);
        }
        bs
    }

    /// The capacity of this BitSet in bits.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of set bits.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Is this BitSet empty (no bits set)?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The selectivity of this BitSet (fraction of bits set).
    #[inline]
    pub fn selectivity(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.count as f64 / self.capacity as f64
    }

    /// Set bit at position `pos`.
    #[inline]
    pub fn set(&mut self, pos: usize) {
        debug_assert!(pos < self.capacity, "BitSet::set out of bounds: {} >= {}", pos, self.capacity);
        let word = pos / 64;
        let bit = pos % 64;
        if word < self.words.len() {
            let mask = 1u64 << bit;
            if self.words[word] & mask == 0 {
                self.words[word] |= mask;
                self.count += 1;
            }
        }
    }

    /// Clear bit at position `pos`.
    #[inline]
    pub fn clear(&mut self, pos: usize) {
        debug_assert!(pos < self.capacity, "BitSet::clear out of bounds: {} >= {}", pos, self.capacity);
        let word = pos / 64;
        let bit = pos % 64;
        if word < self.words.len() {
            let mask = 1u64 << bit;
            if self.words[word] & mask != 0 {
                self.words[word] &= !mask;
                self.count -= 1;
            }
        }
    }

    /// Test if bit at position `pos` is set.
    #[inline]
    pub fn contains(&self, pos: usize) -> bool {
        if pos >= self.capacity {
            return false;
        }
        let word = pos / 64;
        let bit = pos % 64;
        self.words.get(word).map_or(false, |w| w & (1u64 << bit) != 0)
    }

    /// In-place intersection (AND) with another BitSet.
    ///
    /// After this operation, only bits set in *both* sets remain.
    /// This is the key operation for fused queries: the ART filter produces
    /// a BitSet, which is ANDed with the HNSW candidate set.
    ///
    /// Performance: O(n/64) — processes 64 candidates per instruction.
    /// With AVX-512, this becomes O(n/512).
    pub fn intersect(&mut self, other: &BitSet) {
        let min_len = self.words.len().min(other.words.len());

        // AND the overlapping words
        for i in 0..min_len {
            self.words[i] &= other.words[i];
        }

        // Clear any words beyond the other's length
        for word in self.words[min_len..].iter_mut() {
            *word = 0;
        }

        // Recount (could be optimized with popcount SIMD)
        self.recount();
    }

    /// In-place union (OR) with another BitSet.
    pub fn union(&mut self, other: &BitSet) {
        // Extend if necessary
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }

        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] |= other.words[i];
        }

        self.recount();
    }

    /// In-place difference (AND NOT) — remove bits that are set in `other`.
    pub fn difference(&mut self, other: &BitSet) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= !other.words[i];
        }

        self.recount();
    }

    /// Create a new BitSet that is the intersection of this and another.
    pub fn and(&self, other: &BitSet) -> BitSet {
        let mut result = self.clone();
        result.intersect(other);
        result
    }

    /// Create a new BitSet that is the union of this and another.
    pub fn or(&self, other: &BitSet) -> BitSet {
        let mut result = self.clone();
        result.union(other);
        result
    }

    /// Iterate over the positions of set bits.
    ///
    /// Uses hardware `trailing_zeros` to skip zero regions efficiently.
    /// For sparse sets, this is much faster than iterating all bits.
    pub fn iter(&self) -> BitSetIter<'_> {
        BitSetIter {
            words: &self.words,
            word_idx: 0,
            current_word: self.words.first().copied().unwrap_or(0),
            base: 0,
        }
    }

    /// Clear all bits.
    pub fn clear_all(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
        self.count = 0;
    }

    /// Recount the number of set bits using popcount.
    fn recount(&mut self) {
        self.count = self.words.iter().map(|w| w.count_ones() as usize).sum();
    }

    /// Memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.words.len() * 8
    }

    /// Access the raw word array (for SIMD operations).
    #[inline]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Mutable access to the raw word array (for SIMD operations).
    #[inline]
    pub fn as_words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

impl fmt::Debug for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BitSet(capacity={}, count={}, selectivity={:.4})",
            self.capacity,
            self.count,
            self.selectivity()
        )
    }
}

impl fmt::Display for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BitSet[{}/{}]",
            self.count, self.capacity
        )
    }
}

/// Iterator over set bit positions, using hardware `trailing_zeros`.
pub struct BitSetIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    current_word: u64,
    base: usize,
}

impl Iterator for BitSetIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_word != 0 {
                let tz = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1; // Clear lowest set bit
                return Some(self.base + tz);
            }

            self.word_idx += 1;
            if self.word_idx >= self.words.len() {
                return None;
            }

            self.current_word = self.words[self.word_idx];
            self.base = self.word_idx * 64;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining: usize = self.current_word.count_ones() as usize
            + self.words[self.word_idx.saturating_add(1)..]
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum::<usize>();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BitSetIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut bs = BitSet::with_capacity(100);
        assert!(bs.is_empty());
        assert_eq!(bs.count(), 0);

        bs.set(0);
        bs.set(63);
        bs.set(64);
        bs.set(99);

        assert_eq!(bs.count(), 4);
        assert!(bs.contains(0));
        assert!(bs.contains(63));
        assert!(bs.contains(64));
        assert!(bs.contains(99));
        assert!(!bs.contains(1));
        assert!(!bs.contains(50));
    }

    #[test]
    fn test_intersection() {
        let mut a = BitSet::with_capacity(100);
        let mut b = BitSet::with_capacity(100);

        a.set(1);
        a.set(5);
        a.set(10);
        a.set(50);

        b.set(5);
        b.set(10);
        b.set(20);

        a.intersect(&b);

        assert_eq!(a.count(), 2);
        assert!(a.contains(5));
        assert!(a.contains(10));
        assert!(!a.contains(1));
        assert!(!a.contains(50));
    }

    #[test]
    fn test_union() {
        let mut a = BitSet::with_capacity(100);
        let mut b = BitSet::with_capacity(100);

        a.set(1);
        a.set(5);
        b.set(5);
        b.set(10);

        a.union(&b);

        assert_eq!(a.count(), 3);
        assert!(a.contains(1));
        assert!(a.contains(5));
        assert!(a.contains(10));
    }

    #[test]
    fn test_iterator() {
        let mut bs = BitSet::with_capacity(200);
        bs.set(0);
        bs.set(63);
        bs.set(64);
        bs.set(127);
        bs.set(128);
        bs.set(199);

        let bits: Vec<usize> = bs.iter().collect();
        assert_eq!(bits, vec![0, 63, 64, 127, 128, 199]);
        assert_eq!(bs.iter().len(), 6);
    }

    #[test]
    fn test_all() {
        let bs = BitSet::all(100);
        assert_eq!(bs.count(), 100);
        assert!(bs.contains(0));
        assert!(bs.contains(99));
        assert!(!bs.contains(100));
    }

    #[test]
    fn test_from_iter() {
        let bs = BitSet::from_iter(100, vec![1, 5, 10, 50]);
        assert_eq!(bs.count(), 4);
        assert!(bs.contains(1));
        assert!(bs.contains(50));
    }

    #[test]
    fn test_difference() {
        let mut a = BitSet::from_iter(100, vec![1, 5, 10, 50]);
        let b = BitSet::from_iter(100, vec![5, 10, 20]);

        a.difference(&b);

        assert_eq!(a.count(), 2);
        assert!(a.contains(1));
        assert!(a.contains(50));
        assert!(!a.contains(5));
    }

    #[test]
    fn test_selectivity() {
        let bs = BitSet::from_iter(1000, 0..100);
        assert!((bs.selectivity() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_memory_usage() {
        let bs = BitSet::with_capacity(1_000_000);
        // 1M bits = 128 KB + struct overhead
        assert!(bs.memory_usage() < 200_000);
    }

    #[test]
    fn test_clear() {
        let mut bs = BitSet::from_iter(100, vec![1, 5, 10]);
        bs.clear(5);
        assert_eq!(bs.count(), 2);
        assert!(!bs.contains(5));
    }
}
