//! Filter support with bitsets and filter-aware widening.

use crate::types::*;

/// Bitset filter for vector IDs
#[derive(Debug, Clone)]
pub struct BitsetFilter {
    /// Bitset words (64 bits each)
    bits: Vec<u64>,
    /// Number of vectors covered
    n_vec: u32,
    /// Cached population count
    popcount: Option<u32>,
}

impl BitsetFilter {
    /// Create an empty filter (all vectors excluded)
    pub fn new_empty(n_vec: u32) -> Self {
        let num_words = (n_vec as usize + 63) / 64;
        Self {
            bits: vec![0; num_words],
            n_vec,
            popcount: Some(0),
        }
    }

    /// Create a full filter (all vectors included)
    pub fn new_full(n_vec: u32) -> Self {
        let num_words = (n_vec as usize + 63) / 64;
        let mut bits = vec![u64::MAX; num_words];
        
        // Clear bits beyond n_vec
        let remainder = n_vec as usize % 64;
        if remainder > 0 && !bits.is_empty() {
            bits[num_words - 1] = (1u64 << remainder) - 1;
        }
        
        Self {
            bits,
            n_vec,
            popcount: Some(n_vec),
        }
    }

    /// Create from a list of included vector IDs
    pub fn from_ids(n_vec: u32, ids: &[VectorId]) -> Self {
        let mut filter = Self::new_empty(n_vec);
        for &id in ids {
            filter.set(id);
        }
        filter.popcount = None; // Invalidate cache
        filter
    }

    /// Set a bit (include vector)
    #[inline]
    pub fn set(&mut self, id: VectorId) {
        if id < self.n_vec {
            let word_idx = id as usize / 64;
            let bit_idx = id as usize % 64;
            self.bits[word_idx] |= 1u64 << bit_idx;
            self.popcount = None;
        }
    }

    /// Clear a bit (exclude vector)
    #[inline]
    pub fn clear(&mut self, id: VectorId) {
        if id < self.n_vec {
            let word_idx = id as usize / 64;
            let bit_idx = id as usize % 64;
            self.bits[word_idx] &= !(1u64 << bit_idx);
            self.popcount = None;
        }
    }

    /// Check if a vector is included
    #[inline]
    pub fn contains(&self, id: VectorId) -> bool {
        if id >= self.n_vec {
            return false;
        }
        let word_idx = id as usize / 64;
        let bit_idx = id as usize % 64;
        (self.bits[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Count of included vectors
    pub fn count(&mut self) -> u32 {
        if let Some(c) = self.popcount {
            return c;
        }
        let c = self.bits.iter().map(|w| w.count_ones()).sum();
        self.popcount = Some(c);
        c
    }

    /// Get selectivity (fraction of vectors included)
    pub fn selectivity(&mut self) -> f32 {
        if self.n_vec == 0 {
            return 0.0;
        }
        self.count() as f32 / self.n_vec as f32
    }

    /// AND with another filter
    pub fn and(&self, other: &BitsetFilter) -> BitsetFilter {
        assert_eq!(self.n_vec, other.n_vec);
        let bits: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| a & b)
            .collect();
        BitsetFilter {
            bits,
            n_vec: self.n_vec,
            popcount: None,
        }
    }

    /// OR with another filter
    pub fn or(&self, other: &BitsetFilter) -> BitsetFilter {
        assert_eq!(self.n_vec, other.n_vec);
        let bits: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| a | b)
            .collect();
        BitsetFilter {
            bits,
            n_vec: self.n_vec,
            popcount: None,
        }
    }

    /// NOT (invert filter)
    pub fn not(&self) -> BitsetFilter {
        let num_words = self.bits.len();
        let mut bits: Vec<u64> = self.bits.iter().map(|&w| !w).collect();
        
        // Clear bits beyond n_vec
        let remainder = self.n_vec as usize % 64;
        if remainder > 0 && !bits.is_empty() {
            bits[num_words - 1] &= (1u64 << remainder) - 1;
        }
        
        BitsetFilter {
            bits,
            n_vec: self.n_vec,
            popcount: None,
        }
    }

    /// Apply filter to candidates, returning only included ones
    pub fn filter_candidates(&self, candidates: &[ScoredCandidate]) -> Vec<ScoredCandidate> {
        candidates.iter()
            .filter(|c| self.contains(c.id))
            .copied()
            .collect()
    }

    /// Get raw bits
    pub fn bits(&self) -> &[u64] {
        &self.bits
    }

    /// Get number of vectors
    pub fn n_vec(&self) -> u32 {
        self.n_vec
    }
}

/// Compute filter-aware widening factor
/// If selectivity is `s`, we need ~1/s more candidates before filtering
pub fn compute_widening_factor(selectivity: f32, max_factor: f32) -> f32 {
    if selectivity <= 0.0 {
        return max_factor;
    }
    (1.0 / selectivity).min(max_factor)
}

/// Apply filter-aware widening to candidate count
pub fn widen_for_filter(base_count: usize, selectivity: f32, max_factor: f32) -> usize {
    let factor = compute_widening_factor(selectivity, max_factor);
    ((base_count as f32) * factor).ceil() as usize
}

/// Combine a tombstone bitset with a filter
pub fn apply_tombstones(filter: &BitsetFilter, tombstones: &[u64]) -> BitsetFilter {
    let mut result = filter.clone();
    for (i, &tombstone_word) in tombstones.iter().enumerate() {
        if i < result.bits.len() {
            // Clear bits that are tombstoned
            result.bits[i] &= !tombstone_word;
        }
    }
    result.popcount = None;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_basic() {
        let mut filter = BitsetFilter::new_empty(100);
        
        assert!(!filter.contains(0));
        filter.set(0);
        assert!(filter.contains(0));
        
        filter.set(50);
        filter.set(99);
        assert_eq!(filter.count(), 3);
        
        filter.clear(50);
        assert!(!filter.contains(50));
        assert_eq!(filter.count(), 2);
    }

    #[test]
    fn test_bitset_full() {
        let mut filter = BitsetFilter::new_full(100);
        assert!(filter.contains(0));
        assert!(filter.contains(99));
        assert!(!filter.contains(100));
        assert_eq!(filter.count(), 100);
    }

    #[test]
    fn test_bitset_and() {
        let mut a = BitsetFilter::new_empty(100);
        a.set(0);
        a.set(1);
        a.set(2);
        
        let mut b = BitsetFilter::new_empty(100);
        b.set(1);
        b.set(2);
        b.set(3);
        
        let mut c = a.and(&b);
        assert!(!c.contains(0));
        assert!(c.contains(1));
        assert!(c.contains(2));
        assert!(!c.contains(3));
        assert_eq!(c.count(), 2);
    }

    #[test]
    fn test_bitset_or() {
        let mut a = BitsetFilter::new_empty(100);
        a.set(0);
        a.set(1);
        
        let mut b = BitsetFilter::new_empty(100);
        b.set(1);
        b.set(2);
        
        let mut c = a.or(&b);
        assert!(c.contains(0));
        assert!(c.contains(1));
        assert!(c.contains(2));
        assert!(!c.contains(3));
        assert_eq!(c.count(), 3);
    }

    #[test]
    fn test_selectivity() {
        let mut filter = BitsetFilter::new_empty(1000);
        for i in 0..100 {
            filter.set(i);
        }
        
        let selectivity = filter.selectivity();
        assert!((selectivity - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_widening_factor() {
        // 10% selectivity -> 10x widening (capped)
        assert!((compute_widening_factor(0.1, 20.0) - 10.0).abs() < 0.001);
        
        // 1% selectivity -> capped at max
        assert!((compute_widening_factor(0.01, 20.0) - 20.0).abs() < 0.001);
        
        // 100% selectivity -> no widening
        assert!((compute_widening_factor(1.0, 20.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_candidates() {
        let mut filter = BitsetFilter::new_empty(100);
        filter.set(1);
        filter.set(3);
        filter.set(5);
        
        let candidates = vec![
            ScoredCandidate { id: 0, score: 1.0 },
            ScoredCandidate { id: 1, score: 2.0 },
            ScoredCandidate { id: 2, score: 3.0 },
            ScoredCandidate { id: 3, score: 4.0 },
        ];
        
        let filtered = filter.filter_candidates(&candidates);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id, 1);
        assert_eq!(filtered[1].id, 3);
    }
}
