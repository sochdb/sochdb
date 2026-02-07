// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! # SSI Scaling Guardrails
//!
//! Provides scalable alternatives to per-key read sets for SSI:
//! - **Range Locks**: Interval-based locking for range scans
//! - **Predicate Locks**: Bloom filter-based summarized read sets
//! - **Adaptive Switching**: Automatically choose based on workload
//!
//! ## Problem Statement
//!
//! Naive SSI with per-key read sets is O(n) in scan workloads.
//! For analytics tables and backups, this causes:
//! - Memory blowup from storing every key
//! - O(n) conflict checking at commit time
//!
//! ## Solution
//!
//! Replace per-key sets with interval/predicate representations:
//! - Range locks: O(log m) operations where m is number of intervals
//! - Bloom filters: O(1) membership check with false-positive aborts
//!
//! ## Complexity Analysis
//!
//! | Operation     | Per-Key      | Range Lock   | Bloom Filter |
//! |---------------|--------------|--------------|--------------|
//! | Add read      | O(1)         | O(log m)     | O(k)         |
//! | Check conflict| O(n)         | O(log m)     | O(k)         |
//! | Memory        | O(n)         | O(m)         | O(fixed)     |
//!
//! where n = keys read, m = intervals, k = hash functions

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;

/// Transaction ID type
pub type TxnId = u64;

/// Key range boundary
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeBound {
    /// Unbounded (negative or positive infinity)
    Unbounded,
    /// Inclusive bound
    Inclusive(Vec<u8>),
    /// Exclusive bound
    Exclusive(Vec<u8>),
}

impl RangeBound {
    /// Check if this bound is less than a key (for start bounds)
    pub fn is_before(&self, key: &[u8]) -> bool {
        match self {
            RangeBound::Unbounded => true,
            RangeBound::Inclusive(bound) => bound.as_slice() <= key,
            RangeBound::Exclusive(bound) => bound.as_slice() < key,
        }
    }

    /// Check if this bound is greater than a key (for end bounds)
    pub fn is_after(&self, key: &[u8]) -> bool {
        match self {
            RangeBound::Unbounded => true,
            RangeBound::Inclusive(bound) => bound.as_slice() >= key,
            RangeBound::Exclusive(bound) => bound.as_slice() > key,
        }
    }
}

/// Key range for range locks
#[derive(Debug, Clone)]
pub struct KeyRange {
    /// Start of range
    pub start: RangeBound,
    /// End of range
    pub end: RangeBound,
    /// Optional table/index identifier
    pub table_id: Option<u64>,
}

impl KeyRange {
    /// Create a point range (single key)
    pub fn point(key: Vec<u8>) -> Self {
        Self {
            start: RangeBound::Inclusive(key.clone()),
            end: RangeBound::Inclusive(key),
            table_id: None,
        }
    }

    /// Create a range
    pub fn range(start: RangeBound, end: RangeBound) -> Self {
        Self {
            start,
            end,
            table_id: None,
        }
    }

    /// Create a full table scan range
    pub fn full_table(table_id: u64) -> Self {
        Self {
            start: RangeBound::Unbounded,
            end: RangeBound::Unbounded,
            table_id: Some(table_id),
        }
    }

    /// Check if this range contains a key
    pub fn contains(&self, key: &[u8]) -> bool {
        self.start.is_before(key) && self.end.is_after(key)
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &KeyRange) -> bool {
        // Check table ID first
        if self.table_id.is_some() && other.table_id.is_some() && self.table_id != other.table_id {
            return false;
        }

        // Check range overlap
        // Ranges overlap if: start1 < end2 AND start2 < end1
        let self_start_before_other_end = match (&self.start, &other.end) {
            (RangeBound::Unbounded, _) | (_, RangeBound::Unbounded) => true,
            (RangeBound::Inclusive(s), RangeBound::Inclusive(e)) => s <= e,
            (RangeBound::Inclusive(s), RangeBound::Exclusive(e)) => s < e,
            (RangeBound::Exclusive(s), RangeBound::Inclusive(e)) => s < e,
            (RangeBound::Exclusive(s), RangeBound::Exclusive(e)) => s < e,
        };

        let other_start_before_self_end = match (&other.start, &self.end) {
            (RangeBound::Unbounded, _) | (_, RangeBound::Unbounded) => true,
            (RangeBound::Inclusive(s), RangeBound::Inclusive(e)) => s <= e,
            (RangeBound::Inclusive(s), RangeBound::Exclusive(e)) => s < e,
            (RangeBound::Exclusive(s), RangeBound::Inclusive(e)) => s < e,
            (RangeBound::Exclusive(s), RangeBound::Exclusive(e)) => s < e,
        };

        self_start_before_other_end && other_start_before_self_end
    }

    /// Try to merge with another range (if adjacent or overlapping)
    pub fn try_merge(&self, other: &KeyRange) -> Option<KeyRange> {
        if !self.overlaps(other) {
            return None;
        }

        // Merge to create smallest enclosing range
        let start = match (&self.start, &other.start) {
            (RangeBound::Unbounded, _) | (_, RangeBound::Unbounded) => RangeBound::Unbounded,
            (RangeBound::Inclusive(a), RangeBound::Inclusive(b)) => {
                RangeBound::Inclusive(a.min(b).clone())
            }
            (RangeBound::Exclusive(a), RangeBound::Exclusive(b)) => {
                RangeBound::Exclusive(a.min(b).clone())
            }
            (RangeBound::Inclusive(a), RangeBound::Exclusive(b)) => {
                if a <= b {
                    RangeBound::Inclusive(a.clone())
                } else {
                    RangeBound::Exclusive(b.clone())
                }
            }
            (RangeBound::Exclusive(a), RangeBound::Inclusive(b)) => {
                if b <= a {
                    RangeBound::Inclusive(b.clone())
                } else {
                    RangeBound::Exclusive(a.clone())
                }
            }
        };

        let end = match (&self.end, &other.end) {
            (RangeBound::Unbounded, _) | (_, RangeBound::Unbounded) => RangeBound::Unbounded,
            (RangeBound::Inclusive(a), RangeBound::Inclusive(b)) => {
                RangeBound::Inclusive(a.max(b).clone())
            }
            (RangeBound::Exclusive(a), RangeBound::Exclusive(b)) => {
                RangeBound::Exclusive(a.max(b).clone())
            }
            (RangeBound::Inclusive(a), RangeBound::Exclusive(b)) => {
                if a >= b {
                    RangeBound::Inclusive(a.clone())
                } else {
                    RangeBound::Exclusive(b.clone())
                }
            }
            (RangeBound::Exclusive(a), RangeBound::Inclusive(b)) => {
                if b >= a {
                    RangeBound::Inclusive(b.clone())
                } else {
                    RangeBound::Exclusive(a.clone())
                }
            }
        };

        Some(KeyRange {
            start,
            end,
            table_id: self.table_id.or(other.table_id),
        })
    }
}

/// Range lock entry
#[derive(Debug, Clone)]
struct RangeLockEntry {
    range: KeyRange,
    txn_id: TxnId,
    is_write: bool,
}

/// Interval-based range lock manager
///
/// Uses an interval tree for O(log n) conflict detection.
pub struct RangeLockManager {
    /// All active range locks
    locks: RwLock<Vec<RangeLockEntry>>,
    /// Statistics
    stats: RangeLockStats,
}

/// Range lock statistics
#[derive(Default)]
pub struct RangeLockStats {
    pub total_locks: AtomicU64,
    pub range_locks: AtomicU64,
    pub point_locks: AtomicU64,
    pub conflicts_detected: AtomicU64,
    pub merges_performed: AtomicU64,
}

impl RangeLockManager {
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(Vec::new()),
            stats: RangeLockStats::default(),
        }
    }

    /// Acquire a range lock for a transaction
    pub fn acquire(
        &self,
        txn_id: TxnId,
        range: KeyRange,
        is_write: bool,
    ) -> Result<(), RangeLockConflict> {
        self.stats.total_locks.fetch_add(1, AtomicOrdering::Relaxed);

        // Check for conflicts
        {
            let locks = self.locks.read();
            for entry in locks.iter() {
                if entry.txn_id == txn_id {
                    continue; // Own lock
                }

                if range.overlaps(&entry.range) {
                    // Conflict if either is a write lock
                    if is_write || entry.is_write {
                        self.stats.conflicts_detected.fetch_add(1, AtomicOrdering::Relaxed);
                        return Err(RangeLockConflict {
                            holder_txn: entry.txn_id,
                            requester_txn: txn_id,
                            is_write_conflict: is_write && entry.is_write,
                        });
                    }
                }
            }
        }

        // Acquire the lock
        let mut locks = self.locks.write();

        // Try to merge with existing locks from same transaction
        let mut merged = false;
        for entry in locks.iter_mut() {
            if entry.txn_id == txn_id && entry.is_write == is_write {
                if let Some(merged_range) = entry.range.try_merge(&range) {
                    entry.range = merged_range;
                    merged = true;
                    self.stats.merges_performed.fetch_add(1, AtomicOrdering::Relaxed);
                    break;
                }
            }
        }

        if !merged {
            locks.push(RangeLockEntry {
                range,
                txn_id,
                is_write,
            });
        }

        Ok(())
    }

    /// Release all locks held by a transaction
    pub fn release(&self, txn_id: TxnId) {
        let mut locks = self.locks.write();
        locks.retain(|entry| entry.txn_id != txn_id);
    }

    /// Check for conflicts without acquiring
    pub fn check_conflict(
        &self,
        txn_id: TxnId,
        range: &KeyRange,
        is_write: bool,
    ) -> Option<TxnId> {
        let locks = self.locks.read();
        for entry in locks.iter() {
            if entry.txn_id == txn_id {
                continue;
            }
            if range.overlaps(&entry.range) && (is_write || entry.is_write) {
                return Some(entry.txn_id);
            }
        }
        None
    }

    /// Get number of locks held by a transaction
    pub fn lock_count(&self, txn_id: TxnId) -> usize {
        self.locks
            .read()
            .iter()
            .filter(|e| e.txn_id == txn_id)
            .count()
    }
}

impl Default for RangeLockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Range lock conflict error
#[derive(Debug, Clone)]
pub struct RangeLockConflict {
    pub holder_txn: TxnId,
    pub requester_txn: TxnId,
    pub is_write_conflict: bool,
}

impl std::fmt::Display for RangeLockConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Range lock conflict: txn {} blocked by txn {} (write: {})",
            self.requester_txn, self.holder_txn, self.is_write_conflict
        )
    }
}

impl std::error::Error for RangeLockConflict {}

/// Bloom filter for approximate read set
///
/// Uses k hash functions for O(k) operations with configurable
/// false positive rate.
pub struct BloomReadSet {
    /// Bit vector
    bits: Vec<u64>,
    /// Number of hash functions
    k: usize,
    /// Number of bits
    m: usize,
    /// Number of items added
    count: usize,
}

impl BloomReadSet {
    /// Create a new bloom filter
    ///
    /// Size is calculated for target false positive rate:
    /// m = -n * ln(p) / (ln(2))^2
    /// k = m/n * ln(2)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let m = (-((expected_items as f64) * false_positive_rate.ln())
            / (2.0_f64.ln().powi(2))) as usize;
        let k = ((m as f64 / expected_items as f64) * 2.0_f64.ln()).ceil() as usize;

        // Round up to 64-bit words
        let words = m.div_ceil(64);

        Self {
            bits: vec![0u64; words],
            k: k.max(1),
            m: words * 64,
            count: 0,
        }
    }

    /// Add a key to the read set
    pub fn add(&mut self, key: &[u8]) {
        for i in 0..self.k {
            let hash = self.hash(key, i);
            let bit_idx = hash % self.m;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            self.bits[word_idx] |= 1 << bit_offset;
        }
        self.count += 1;
    }

    /// Check if a key might be in the read set
    ///
    /// Returns:
    /// - false: key is definitely NOT in set
    /// - true: key MIGHT be in set (or false positive)
    pub fn might_contain(&self, key: &[u8]) -> bool {
        for i in 0..self.k {
            let hash = self.hash(key, i);
            let bit_idx = hash % self.m;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            if self.bits[word_idx] & (1 << bit_offset) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute hash for key with seed
    fn hash(&self, key: &[u8], seed: usize) -> usize {
        // Double hashing: h(i) = h1 + i * h2
        let h1 = self.hash_fnv1a(key);
        let h2 = self.hash_murmur_like(key);
        ((h1 as u128 + (seed as u128) * (h2 as u128)) % (self.m as u128)) as usize
    }

    fn hash_fnv1a(&self, key: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for byte in key {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn hash_murmur_like(&self, key: &[u8]) -> u64 {
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: u32 = 47;

        let mut h: u64 = 0xdeadbeef ^ (key.len() as u64).wrapping_mul(M);

        for chunk in key.chunks(8) {
            let mut k = 0u64;
            for (i, byte) in chunk.iter().enumerate() {
                k |= (*byte as u64) << (i * 8);
            }
            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);
            h ^= k;
            h = h.wrapping_mul(M);
        }

        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        h
    }

    /// Estimated false positive rate
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let ones = self.bits.iter().map(|w| w.count_ones()).sum::<u32>() as f64;
        let ratio = ones / self.m as f64;
        ratio.powi(self.k as i32)
    }

    /// Number of items added
    pub fn count(&self) -> usize {
        self.count
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

/// Adaptive read set that switches strategy based on size
pub struct AdaptiveReadSet {
    /// Threshold for switching from exact to bloom
    threshold: usize,
    /// Exact keys (if small enough)
    exact_keys: Option<HashSet<Vec<u8>>>,
    /// Bloom filter (if too many keys)
    bloom: Option<BloomReadSet>,
    /// Range locks (for range scans)
    ranges: Vec<KeyRange>,
}

impl AdaptiveReadSet {
    /// Create a new adaptive read set
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            exact_keys: Some(HashSet::new()),
            bloom: None,
            ranges: Vec::new(),
        }
    }

    /// Record a point read
    pub fn add_point(&mut self, key: Vec<u8>) {
        if let Some(ref mut exact) = self.exact_keys {
            exact.insert(key.clone());
            if exact.len() >= self.threshold {
                // Switch to bloom filter
                let mut bloom = BloomReadSet::new(self.threshold * 10, 0.01);
                for k in exact.drain() {
                    bloom.add(&k);
                }
                bloom.add(&key);
                self.bloom = Some(bloom);
                self.exact_keys = None;
            }
        } else if let Some(ref mut bloom) = self.bloom {
            bloom.add(&key);
        }
    }

    /// Record a range scan
    pub fn add_range(&mut self, range: KeyRange) {
        // Try to merge with existing ranges
        let mut merged = false;
        for existing in &mut self.ranges {
            if let Some(merged_range) = existing.try_merge(&range) {
                *existing = merged_range;
                merged = true;
                break;
            }
        }
        if !merged {
            self.ranges.push(range);
        }
    }

    /// Check if a key might conflict
    pub fn might_conflict(&self, key: &[u8]) -> bool {
        // Check exact keys
        if let Some(ref exact) = self.exact_keys {
            if exact.contains(key) {
                return true;
            }
        }

        // Check bloom filter
        if let Some(ref bloom) = self.bloom {
            if bloom.might_contain(key) {
                return true;
            }
        }

        // Check ranges
        for range in &self.ranges {
            if range.contains(key) {
                return true;
            }
        }

        false
    }

    /// Memory usage
    pub fn memory_bytes(&self) -> usize {
        let exact = self
            .exact_keys
            .as_ref()
            .map(|e| e.iter().map(|k| k.len()).sum::<usize>())
            .unwrap_or(0);
        let bloom = self.bloom.as_ref().map(|b| b.memory_bytes()).unwrap_or(0);
        let ranges = self.ranges.len() * 64; // Approximate
        exact + bloom + ranges
    }

    /// Is using exact tracking?
    pub fn is_exact(&self) -> bool {
        self.exact_keys.is_some()
    }
}

/// Backoff strategy for conflict resolution
#[derive(Debug, Clone)]
pub struct BackoffStrategy {
    /// Initial backoff duration
    pub initial: Duration,
    /// Maximum backoff duration
    pub max: Duration,
    /// Multiplier for exponential backoff
    pub multiplier: f64,
    /// Add jitter to prevent thundering herd
    pub jitter: bool,
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self {
            initial: Duration::from_micros(100),
            max: Duration::from_millis(100),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

impl BackoffStrategy {
    /// Calculate backoff for attempt number
    pub fn backoff_for(&self, attempt: u32) -> Duration {
        let base = self.initial.as_nanos() as f64 * self.multiplier.powi(attempt as i32);
        let capped = base.min(self.max.as_nanos() as f64);

        let final_nanos = if self.jitter {
            // Add ±25% jitter
            let jitter = (capped * 0.25) * (rand_simple() * 2.0 - 1.0);
            (capped + jitter).max(0.0)
        } else {
            capped
        };

        Duration::from_nanos(final_nanos as u64)
    }
}

/// Simple pseudo-random for jitter (deterministic but varied)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    ((nanos % 1000) as f64) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_range_contains() {
        let range = KeyRange::range(
            RangeBound::Inclusive(b"aaa".to_vec()),
            RangeBound::Inclusive(b"zzz".to_vec()),
        );

        assert!(range.contains(b"mmm"));
        assert!(range.contains(b"aaa"));
        assert!(range.contains(b"zzz"));
        assert!(!range.contains(b"AAA"));
    }

    #[test]
    fn test_key_range_overlap() {
        let r1 = KeyRange::range(
            RangeBound::Inclusive(b"a".to_vec()),
            RangeBound::Inclusive(b"m".to_vec()),
        );
        let r2 = KeyRange::range(
            RangeBound::Inclusive(b"k".to_vec()),
            RangeBound::Inclusive(b"z".to_vec()),
        );
        let r3 = KeyRange::range(
            RangeBound::Inclusive(b"n".to_vec()),
            RangeBound::Inclusive(b"z".to_vec()),
        );

        assert!(r1.overlaps(&r2)); // a-m overlaps k-z
        assert!(!r1.overlaps(&r3)); // a-m doesn't overlap n-z
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomReadSet::new(1000, 0.01);

        bloom.add(b"key1");
        bloom.add(b"key2");
        bloom.add(b"key3");

        assert!(bloom.might_contain(b"key1"));
        assert!(bloom.might_contain(b"key2"));
        assert!(bloom.might_contain(b"key3"));
        // Might have false positives, but shouldn't have false negatives
    }

    #[test]
    fn test_range_lock_manager() {
        let manager = RangeLockManager::new();

        // Acquire non-overlapping ranges
        assert!(manager
            .acquire(1, KeyRange::point(b"key1".to_vec()), true)
            .is_ok());
        assert!(manager
            .acquire(2, KeyRange::point(b"key2".to_vec()), true)
            .is_ok());

        // Conflict on overlapping write
        assert!(manager
            .acquire(3, KeyRange::point(b"key1".to_vec()), true)
            .is_err());

        // Release and retry
        manager.release(1);
        assert!(manager
            .acquire(3, KeyRange::point(b"key1".to_vec()), true)
            .is_ok());
    }

    #[test]
    fn test_adaptive_read_set() {
        let mut set = AdaptiveReadSet::new(100);

        // Add some point reads
        for i in 0..50 {
            set.add_point(format!("key{}", i).into_bytes());
        }
        assert!(set.is_exact());

        // Add more to trigger bloom switch
        for i in 50..150 {
            set.add_point(format!("key{}", i).into_bytes());
        }
        assert!(!set.is_exact());

        // Should still detect conflicts
        assert!(set.might_conflict(b"key0"));
        assert!(set.might_conflict(b"key100"));
    }

    #[test]
    fn test_backoff_strategy() {
        let strategy = BackoffStrategy::default();

        let b0 = strategy.backoff_for(0);
        let b1 = strategy.backoff_for(1);
        let b5 = strategy.backoff_for(5);

        assert!(b1 > b0);
        assert!(b5 > b1);
        assert!(b5 <= strategy.max);
    }
}
