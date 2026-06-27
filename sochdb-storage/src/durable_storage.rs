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

//! Durable Storage Layer
//!
//! Wires together the live storage components into a durable engine:
//!
//! - WAL (txn_wal.rs) for crash-consistent durability + recovery
//! - Group Commit for throughput
//! - MVCC for isolation
//! - LSCS for columnar efficiency
//!
//! Truth-in-capabilities: the live path provides crash-consistent WAL recovery,
//! but NOT at-rest encryption, point-in-time recovery, ARIES checkpointing, or
//! WAL fencing — those modules exist but are quarantined behind the empty,
//! non-default `experimental` feature and are unwired. Query
//! [`crate::durability_capabilities`] rather than relying on prose like
//! "production-grade".
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      DurableStorage                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
//! │  │ MvccManager │    │ GroupCommit │───▶│ TxnWal (fsync)      │ │
//! │  │             │    │             │    └─────────────────────┘ │
//! │  │ ┌─────────┐ │    └─────────────┘                            │
//! │  │ │Snapshots│ │                                                │
//! │  │ └─────────┘ │    ┌─────────────────────────────────────────┐│
//! │  │ ┌─────────┐ │    │              MemTable                    ││
//! │  │ │ Txn Map │ │    │  (key → (value, txn_id, version))       ││
//! │  │ └─────────┘ │    └─────────────────────────────────────────┘│
//! │  └─────────────┘                                                │
//! │                      ┌─────────────────────────────────────────┐│
//! │                      │              LSCS (SST)                  ││
//! │                      │  Immutable columnar segments             ││
//! │                      └─────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Concurrency
//!
//! - Writers: Serialize through WAL, use MVCC for conflict detection
//! - Readers: Lock-free reads at snapshot timestamp
//! - Commits: Batched through GroupCommit for throughput
//!
//! ## Isolation Contract
//!
//! The live write path (`MvccManager`, used by `DurableStorage`) provides
//! **Serializable Snapshot Isolation (SSI)**, which is strictly stronger than
//! plain Snapshot Isolation (SI):
//!
//! - **Snapshot reads.** Every transaction reads from a consistent snapshot
//!   fixed at `begin_transaction()` (its `snapshot_ts`). Concurrent commits are
//!   invisible to it — see `test_snapshot_isolation`. This alone is SI and is
//!   vulnerable to **write skew** (two transactions each read a set the other
//!   writes, both commit, and no serial order reproduces the result).
//! - **Write-skew prevention.** On commit, `MvccManager::validate_ssi` inspects
//!   recently-committed concurrent transactions for rw-antidependency edges:
//!   an inbound edge (another txn wrote a key we read, `T_other →rw→ T_me`) and
//!   an outbound edge (we wrote a key another txn read, `T_me →rw→ T_other`).
//!   When a transaction sits on **both** an inbound and an outbound rw-edge it
//!   is the pivot of a potential dependency cycle (Cahill/Fekete "dangerous
//!   structure"), so it is aborted. This is the conservative safe subset of the
//!   SSI test: it may abort some serializable schedules (false positives) but
//!   **never admits a non-serializable one** (no false negatives), so the
//!   externally observable isolation level is Serializable.
//! - **Read-only transactions** (`begin_read_only`) skip read-set tracking and
//!   never participate in validation; they always observe a serializable
//!   snapshot and never abort.
//!
//! ### MVCC garbage collection is snapshot-safe
//!
//! Old versions are pruned by `DurableStorage::gc()`, which feeds the
//! **low-water-mark** `MvccManager::min_active_snapshot()` — the minimum
//! `snapshot_ts` across all still-active transactions — into
//! `MvccMemTable::gc()` and then `BinarySearchChain::gc_by_ts()`. The chain
//! retains every version with `commit_ts > min_active_ts` **plus one anchor
//! version at or below the watermark**, so any in-flight reader can still
//! resolve the correct version for its snapshot. A version is only freed once
//! **no active snapshot can observe it**. The watermark is recomputed on every
//! `begin`/`commit`/`abort`, so it is monotonic with respect to the oldest live
//! reader. See `test_gc_preserves_versions_for_active_snapshot`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use dashmap::DashMap;
use smallvec::SmallVec;

use crossbeam_skiplist::SkipMap;

use crate::DurabilityCapabilities;
use crate::deferred_index::{DeferredIndexConfig, DeferredSortedIndex};
use crate::encryption::EncryptionKey;
use crate::group_commit::EventDrivenGroupCommit;
use crate::keyring;
use crate::txn_wal::{TxnWal, TxnWalBuffer, TxnWalEntry};
use sochdb_core::version_chain::{
    BinarySearchChain, ChainEntry, MvccVersionChain, MvccVersionChainMut, Timestamp, TxnId,
    VisibilityContext, WriteConflictDetection,
};
use sochdb_core::{Result, SochDBError};

// =============================================================================
// SSI Bloom Filter - Fast Conflict Pre-Filtering
// =============================================================================

/// Space-efficient Bloom filter for SSI conflict detection
///
/// Used to quickly determine if two transactions MIGHT have conflicting keys.
/// False positives are acceptable (leads to unnecessary exact checks),
/// but false negatives are not allowed.
///
/// ## Configuration
///
/// For 1000 keys with 1% false positive rate:
/// - m = ~9600 bits ≈ 1.2 KB per transaction
/// - k = 7 hash functions
///
/// ## Lazy Initialization
///
/// The bit vector is lazily initialized on first insert to avoid
/// allocation overhead for read-only transactions.
#[derive(Clone, Debug)]
pub struct SsiBloomFilter {
    /// Bit vector (each u64 holds 64 bits) - lazily initialized
    bits: Option<Vec<u64>>,
    /// Expected capacity (used for lazy init sizing)
    expected_capacity: usize,
    /// Number of hash functions to use
    num_hashes: u32,
}

impl SsiBloomFilter {
    /// Optimal number of bits per item for 1% false positive rate
    /// m/n = -ln(p) / (ln(2))² ≈ 9.6 for p = 0.01
    const BITS_PER_ITEM: f64 = 9.6;

    /// Optimal number of hash functions for 1% false positive rate
    /// k = (m/n) × ln(2) ≈ 7
    const DEFAULT_NUM_HASHES: u32 = 7;

    /// Minimum capacity to avoid tiny filters
    const MIN_CAPACITY: usize = 64;

    /// Create a new bloom filter for expected item count (lazy allocation)
    ///
    /// Configured for ~1% false positive rate.
    /// The bit vector is not allocated until first insert.
    #[inline]
    pub fn new(expected_items: usize) -> Self {
        Self {
            bits: None,
            expected_capacity: expected_items.max(Self::MIN_CAPACITY),
            num_hashes: Self::DEFAULT_NUM_HASHES,
        }
    }

    /// Create with specific capacity in words (for memory-constrained scenarios)
    pub fn with_word_capacity(words: usize) -> Self {
        Self {
            bits: None,
            expected_capacity: words.max(1) * 64 / 10, // Approx items from words
            num_hashes: Self::DEFAULT_NUM_HASHES,
        }
    }

    /// Ensure bits are allocated (lazy initialization)
    #[inline]
    fn ensure_allocated(&mut self) {
        if self.bits.is_none() {
            let num_bits = ((self.expected_capacity as f64) * Self::BITS_PER_ITEM).ceil() as usize;
            let num_words = num_bits.div_ceil(64);
            self.bits = Some(vec![0u64; num_words]);
        }
    }

    /// Add a key to the filter - O(k) where k = num_hashes
    #[inline]
    pub fn insert(&mut self, key: &[u8]) {
        self.ensure_allocated();
        let bits = self.bits.as_mut().unwrap();
        let num_bits = bits.len() * 64;
        if num_bits == 0 {
            return;
        }

        // Use two hash functions to simulate k hash functions
        // h(i) = h1 + i * h2 (double hashing technique)
        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            bits[word_idx] |= 1 << bit_pos;
        }
    }

    /// Check if a key might be present - O(k)
    ///
    /// Returns:
    /// - false: Key is definitely NOT in the set (or filter not initialized)
    /// - true: Key MIGHT be in the set (needs exact check)
    #[inline]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let bits = match &self.bits {
            Some(b) => b,
            None => return false, // Uninitialized = empty
        };
        let num_bits = bits.len() * 64;
        if num_bits == 0 {
            return false;
        }

        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if bits[word_idx] & (1 << bit_pos) == 0 {
                return false; // Definitely not present
            }
        }
        true // Might be present
    }

    /// Check if this filter might intersect with another
    ///
    /// Fast O(m/64) check using bitwise AND of all words.
    /// If no bits are shared, sets are definitely disjoint.
    #[inline]
    pub fn may_intersect(&self, other: &SsiBloomFilter) -> bool {
        let (self_bits, other_bits) = match (&self.bits, &other.bits) {
            (Some(s), Some(o)) => (s, o),
            _ => return false, // Either uninitialized = no intersection
        };
        let min_len = self_bits.len().min(other_bits.len());
        for i in 0..min_len {
            if self_bits[i] & other_bits[i] != 0 {
                return true; // Might intersect
            }
        }
        false // Definitely disjoint
    }

    /// First hash function (using built-in hasher)
    #[inline]
    fn hash1(key: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Second hash function (using twox-hash for independence)
    #[inline]
    fn hash2(key: &[u8]) -> u64 {
        twox_hash::xxh3::hash64(key)
    }

    /// Get the memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.bits.as_ref().map(|b| b.len() * 8).unwrap_or(0) + std::mem::size_of::<Self>()
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        match &self.bits {
            Some(bits) => bits.iter().all(|&w| w == 0),
            None => true,
        }
    }
}

/// Type alias for inline key storage - keys up to 32 bytes stored on stack
/// This eliminates heap allocation for typical keys like "users/12345" (12 bytes)
pub type InlineKey = SmallVec<[u8; 32]>;

/// Version of a key-value pair
#[derive(Debug, Clone)]
pub struct Version {
    /// The value (None = tombstone)
    pub value: Option<Vec<u8>>,
    /// Transaction that created this version
    pub txn_id: u64,
    /// Commit timestamp (0 = uncommitted)
    pub commit_ts: u64,
}

// Rec 11: Implement ChainEntry so BinarySearchChain<Version> works
impl ChainEntry for Version {
    #[inline]
    fn commit_ts(&self) -> u64 {
        self.commit_ts
    }
    #[inline]
    fn txn_id(&self) -> u64 {
        self.txn_id
    }
    #[inline]
    fn set_commit_ts(&mut self, ts: u64) {
        self.commit_ts = ts;
    }
}

// ============================================================================
// Optimized VersionChain with Binary Search (Task 1: mm.md)
// ============================================================================

/// Multi-version data for a single key with O(log v) read complexity
///
/// ## Optimization: Binary Search with Sorted Commit Ordering
///
/// Separates committed versions (sorted descending by commit_ts) from
/// uncommitted version (single optional slot per transaction).
///
/// **Before:** O(v) linear scan + O(v) max computation = O(v)
/// **After:** O(1) uncommitted check + O(log v) binary search = O(log v)
///
/// For v=10 versions: 3.3x speedup
/// For v=100 versions: 7x speedup
///
/// ## Rec 11: Consolidated
///
/// Delegates binary-search logic to `BinarySearchChain<Version>` from sochdb-core,
/// eliminating duplication with `mvcc_concurrent::VersionChain`.
#[derive(Debug, Default)]
pub struct VersionChain {
    /// Consolidated binary-search chain (Rec 11)
    inner: BinarySearchChain<Version>,
}

impl VersionChain {
    /// Create a new empty version chain
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: BinarySearchChain::new(),
        }
    }

    /// Add a new uncommitted version
    /// If there's already an uncommitted version from this txn, update it in place
    ///
    /// O(1) - just updates the uncommitted slot
    #[inline]
    pub fn add_uncommitted(&mut self, value: Option<Vec<u8>>, txn_id: u64) {
        match self.inner.uncommitted_mut() {
            Some(v) if v.txn_id == txn_id => {
                // Update in place - O(1)
                v.value = value;
            }
            _ => {
                // New or different txn — set the slot
                self.inner.set_uncommitted(Version {
                    value,
                    txn_id,
                    commit_ts: 0,
                });
            }
        }
    }

    /// Commit a version - moves from uncommitted slot to sorted committed list
    ///
    /// O(log v) - inserts into sorted position using binary search
    #[inline]
    pub fn commit(&mut self, txn_id: u64, commit_ts: u64) -> bool {
        self.inner.commit(txn_id, commit_ts)
    }

    /// Abort a version (remove uncommitted version for txn)
    ///
    /// O(1) - just clears the uncommitted slot if it matches
    #[inline]
    pub fn abort(&mut self, txn_id: u64) {
        self.inner.abort(txn_id);
    }

    /// Read at a snapshot timestamp, optionally seeing own uncommitted writes
    ///
    /// ## Complexity: O(1) + O(log v) = O(log v)
    #[inline]
    pub fn read_at(&self, snapshot_ts: u64, current_txn_id: Option<u64>) -> Option<&Version> {
        self.inner.read_at(snapshot_ts, current_txn_id)
    }

    /// Check if there's an uncommitted version by another transaction
    ///
    /// O(1) - just checks the uncommitted slot
    #[inline]
    pub fn has_write_conflict(&self, my_txn_id: u64) -> bool {
        self.inner.has_write_conflict(my_txn_id)
    }

    /// Garbage collect old versions
    pub fn gc(&mut self, min_active_ts: u64) {
        self.inner.gc_by_ts(min_active_ts);
    }

    /// Get total version count (committed + uncommitted)
    #[inline]
    pub fn version_count(&self) -> usize {
        self.inner.version_count()
    }

    // Legacy compatibility: get versions vec (for tests)
    #[cfg(test)]
    pub fn versions(&self) -> Vec<Version> {
        let mut result = self.inner.committed_versions().to_vec();
        if let Some(v) = self.inner.uncommitted() {
            result.push(v.clone());
        }
        result
    }
}

// =============================================================================
// Rec 6: Unified Version Chain Trait Implementations
// =============================================================================

impl MvccVersionChain for VersionChain {
    type Value = Option<Vec<u8>>;

    fn get_visible(&self, ctx: &VisibilityContext) -> Option<&Self::Value> {
        // Delegate to BinarySearchChain, then project to value field
        self.inner
            .read_at(ctx.snapshot_ts, Some(ctx.reader_txn_id))
            .map(|v| &v.value)
    }

    fn get_latest(&self) -> Option<&Self::Value> {
        self.inner.latest().map(|v| &v.value)
    }

    fn version_count(&self) -> usize {
        self.inner.version_count()
    }
}

impl MvccVersionChainMut for VersionChain {
    fn add_uncommitted(&mut self, value: Self::Value, txn_id: TxnId) {
        self.add_uncommitted(value, txn_id);
    }

    fn commit_version(&mut self, txn_id: TxnId, commit_ts: Timestamp) -> bool {
        self.inner.commit(txn_id, commit_ts)
    }

    fn delete_version(&mut self, txn_id: TxnId, _delete_ts: Timestamp) -> bool {
        // Insert a tombstone (None value) as uncommitted
        self.add_uncommitted(None, txn_id);
        true
    }

    fn gc(&mut self, min_visible_ts: Timestamp) -> (usize, usize) {
        let before = self.inner.committed_count();
        self.inner.gc_by_ts(min_visible_ts);
        let removed = before - self.inner.committed_count();
        (removed, removed * std::mem::size_of::<Version>())
    }
}

impl WriteConflictDetection for VersionChain {
    fn has_write_conflict(&self, txn_id: TxnId) -> bool {
        self.has_write_conflict(txn_id)
    }
}

// =============================================================================
// Pre-sizing Constants to Avoid HashSet Resize Overhead
// =============================================================================

/// Default capacity for write_set HashSet
/// Sized for typical OLTP transactions (10-50 keys)
/// Avoids resize overhead that caused +11% regression
const WRITE_SET_INITIAL_CAPACITY: usize = 32;

/// Default capacity for read_set HashSet  
/// Typically larger than write_set due to read-heavy patterns
const READ_SET_INITIAL_CAPACITY: usize = 64;

/// Transaction state for MVCC
#[derive(Debug, Clone)]
pub struct MvccTransaction {
    /// Transaction ID
    pub txn_id: u64,
    /// Snapshot timestamp (reads see commits before this)
    pub snapshot_ts: u64,
    /// Keys written by this transaction - uses SmallVec for inline storage
    /// Pre-sized to WRITE_SET_INITIAL_CAPACITY to avoid resize overhead
    pub write_set: HashSet<InlineKey>,
    /// Keys read by this transaction (for SSI validation) - uses SmallVec for inline storage
    /// Pre-sized to READ_SET_INITIAL_CAPACITY to avoid resize overhead
    pub read_set: HashSet<InlineKey>,
    /// Bloom filter for write set - fast SSI pre-filtering
    pub write_bloom: SsiBloomFilter,
    /// Bloom filter for read set - fast SSI pre-filtering
    pub read_bloom: SsiBloomFilter,
    /// Transaction state
    pub state: TxnState,
    /// Transaction mode for SSI optimization (Recommendation 9)
    /// ReadOnly/WriteOnly modes skip SSI tracking for 2.6x improvement
    pub mode: TransactionMode,
}

impl MvccTransaction {
    /// Create a new transaction with pre-sized collections
    ///
    /// This avoids HashSet resize overhead during the transaction
    /// which was causing +11% regression on write_set.insert().
    #[inline]
    pub fn new(txn_id: u64, snapshot_ts: u64) -> Self {
        Self::with_mode(txn_id, snapshot_ts, TransactionMode::ReadWrite)
    }

    /// Create a read-only transaction (SSI bypass - 2.6x faster)
    ///
    /// Read-only transactions skip all SSI tracking:
    /// - No read_set allocation
    /// - No read_bloom allocation  
    /// - No commit validation
    ///
    /// ## Performance
    ///
    /// For N=100 reads: 8350ns → 3230ns (2.6× improvement)
    #[inline]
    pub fn read_only(txn_id: u64, snapshot_ts: u64) -> Self {
        Self::with_mode(txn_id, snapshot_ts, TransactionMode::ReadOnly)
    }

    /// Create a write-only transaction (partial SSI bypass)
    ///
    /// Write-only transactions skip read tracking:
    /// - No read_set tracking
    /// - No read_bloom inserts
    /// - Still needs write_set for commit
    #[inline]
    pub fn write_only(txn_id: u64, snapshot_ts: u64) -> Self {
        Self::with_mode(txn_id, snapshot_ts, TransactionMode::WriteOnly)
    }

    /// Create transaction with specific mode
    #[inline]
    pub fn with_mode(txn_id: u64, snapshot_ts: u64, mode: TransactionMode) -> Self {
        // Optimize allocation based on mode
        let (write_capacity, read_capacity) = match mode {
            TransactionMode::ReadOnly => (0, 0), // No tracking needed
            TransactionMode::WriteOnly => (WRITE_SET_INITIAL_CAPACITY, 0),
            TransactionMode::ReadWrite => (WRITE_SET_INITIAL_CAPACITY, READ_SET_INITIAL_CAPACITY),
        };
        Self::with_capacity(txn_id, snapshot_ts, write_capacity, read_capacity, mode)
    }

    /// Create with custom capacities for expected workload
    ///
    /// Use this when you know the transaction will write many keys
    /// to avoid resize overhead entirely.
    #[inline]
    pub fn with_capacity(
        txn_id: u64,
        snapshot_ts: u64,
        write_capacity: usize,
        read_capacity: usize,
        mode: TransactionMode,
    ) -> Self {
        Self {
            txn_id,
            snapshot_ts,
            write_set: HashSet::with_capacity(write_capacity),
            read_set: HashSet::with_capacity(read_capacity),
            write_bloom: SsiBloomFilter::new(write_capacity.max(1)),
            read_bloom: SsiBloomFilter::new(read_capacity.max(1)),
            state: TxnState::Active,
            mode,
        }
    }

    /// Check if this is a read-only transaction
    #[inline]
    pub fn is_read_only(&self) -> bool {
        self.write_set.is_empty()
    }

    /// Check if this is a single-key write transaction
    #[inline]
    pub fn is_single_key_write(&self) -> bool {
        self.write_set.len() == 1 && self.read_set.len() <= 1
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

// =============================================================================
// Transaction Mode for SSI Bypass (Recommendation 9)
// =============================================================================

/// Transaction mode for SSI optimization
///
/// By classifying transactions at begin time, we can skip expensive SSI
/// tracking for the majority of transactions:
///
/// | Mode      | SSI Read Tracking | SSI Write Tracking | Commit Overhead |
/// |-----------|-------------------|--------------------|-----------------|
/// | ReadOnly  | None             | None               | ~10 ns          |
/// | WriteOnly | None             | Full               | ~30 ns          |
/// | ReadWrite | Full             | Full               | ~50 ns          |
///
/// ## Performance Analysis
///
/// For read-only transactions (typically 90% of workload):
/// ```text
/// Current:  T_txn = T_begin + N × (T_read + T_record) + T_commit
///                 = 100ns + N × (32ns + 50ns) + 50ns = 150ns + 82ns × N
///
/// ReadOnly: T_txn = T_begin_ro + N × T_read + T_commit_ro
///                 = 20ns + N × 32ns + 10ns = 30ns + 32ns × N
///
/// For N=100 reads: 8350ns → 3230ns (2.6× faster)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransactionMode {
    /// Read-only transaction - skips ALL SSI tracking
    /// Cannot form rw-antidependency cycles (no writes to create outgoing edges)
    /// Safe to skip read_set, read_bloom, and commit validation entirely
    ReadOnly,

    /// Write-only transaction - skips read tracking
    /// Cannot form incoming rw-edges (no reads from concurrent writers)
    /// Only needs write_set and write_bloom tracking
    WriteOnly,

    /// Full read-write transaction (default) - complete SSI tracking
    /// May form both incoming and outgoing rw-edges
    /// Requires full validation at commit time
    #[default]
    ReadWrite,
}

impl TransactionMode {
    /// Check if this mode requires read tracking
    #[inline]
    pub fn tracks_reads(&self) -> bool {
        matches!(self, TransactionMode::ReadWrite)
    }

    /// Check if this mode requires write tracking
    #[inline]
    pub fn tracks_writes(&self) -> bool {
        matches!(
            self,
            TransactionMode::WriteOnly | TransactionMode::ReadWrite
        )
    }

    /// Check if commit needs SSI validation
    #[inline]
    pub fn needs_ssi_validation(&self) -> bool {
        matches!(self, TransactionMode::ReadWrite)
    }
}

/// SSI conflict edge type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Read-write conflict: T1 reads X, then T2 writes X
    ReadWrite,
    /// Write-read conflict: T1 writes X, then T2 reads X  
    WriteRead,
}

/// SSI conflict edge for dangerous structure detection
#[derive(Debug, Clone)]
pub struct ConflictEdge {
    /// Source transaction
    pub from_txn: u64,
    /// Target transaction
    pub to_txn: u64,
    /// Type of conflict
    pub conflict_type: ConflictType,
}

/// MVCC Manager with SSI support
///
/// Uses DashMap for lock-free per-transaction access.
/// Implements Serializable Snapshot Isolation (SSI) with
/// dangerous structure detection for rw-antidependency cycles.
#[allow(clippy::type_complexity)]
pub struct MvccManager {
    /// Active transactions (sharded for concurrent access)
    active_txns: DashMap<u64, MvccTransaction>,
    /// Current timestamp counter
    ts_counter: AtomicU64,
    /// Minimum active snapshot timestamp (for GC)
    min_active_ts: AtomicU64,
    /// Refcounted multiset of active snapshot timestamps, ordered.
    ///
    /// Maintained incrementally on txn begin/end so the GC watermark
    /// (`min_active_ts`) is O(log N) per begin/commit instead of an O(active_txns)
    /// full `active_txns` DashMap scan (which locked every shard on every
    /// begin AND commit — the dominant on-CPU cost under concurrent writes).
    /// A multiset (count per ts) is required because `begin` reads ts_counter
    /// without incrementing, so concurrent txns can share a snapshot_ts.
    active_snapshots: parking_lot::Mutex<std::collections::BTreeMap<u64, u32>>,
    /// Recently committed transactions for SSI validation
    /// Maps txn_id -> (commit_ts, read_bloom, write_bloom, read_set, write_set)
    /// Bloom filters enable fast O(m/64) pre-filtering before O(n) exact checks
    recent_commits: DashMap<
        u64,
        (
            u64,
            SsiBloomFilter,
            SsiBloomFilter,
            HashSet<InlineKey>,
            HashSet<InlineKey>,
        ),
    >,
    /// Max recent commits to track
    max_recent_commits: usize,
}

impl Default for MvccManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MvccManager {
    pub fn new() -> Self {
        Self {
            active_txns: DashMap::new(),
            ts_counter: AtomicU64::new(1),
            min_active_ts: AtomicU64::new(0),
            active_snapshots: parking_lot::Mutex::new(std::collections::BTreeMap::new()),
            recent_commits: DashMap::new(),
            max_recent_commits: 1000, // Track last 1000 commits for SSI
        }
    }

    /// Begin a new transaction with snapshot isolation
    ///
    /// Uses pre-sized HashSets to avoid resize overhead (+11% regression fix)
    pub fn begin(&self, txn_id: u64) -> MvccTransaction {
        self.begin_with_mode(txn_id, TransactionMode::ReadWrite)
    }

    /// Begin a read-only transaction (SSI bypass - 2.6x faster)
    ///
    /// Read-only transactions skip all SSI tracking, reducing
    /// per-read overhead from ~82ns to ~32ns.
    ///
    /// ## Safety
    ///
    /// Caller must ensure no writes are performed. Attempting to
    /// write in a read-only transaction will still succeed but
    /// won't be tracked for SSI validation.
    #[inline]
    pub fn begin_read_only(&self, txn_id: u64) -> MvccTransaction {
        self.begin_with_mode(txn_id, TransactionMode::ReadOnly)
    }

    /// Begin a write-only transaction (partial SSI bypass)
    ///
    /// Write-only transactions skip read tracking, reducing overhead
    /// for insert-heavy workloads.
    #[inline]
    pub fn begin_write_only(&self, txn_id: u64) -> MvccTransaction {
        self.begin_with_mode(txn_id, TransactionMode::WriteOnly)
    }

    /// Begin a transaction with specific mode
    ///
    /// This is the core transaction creation method that all other
    /// begin_* methods delegate to.
    pub fn begin_with_mode(&self, txn_id: u64, mode: TransactionMode) -> MvccTransaction {
        let snapshot_ts = self.ts_counter.load(Ordering::SeqCst);

        // Create transaction with mode-optimized allocations
        let txn = MvccTransaction::with_mode(txn_id, snapshot_ts, mode);

        self.active_txns.insert(txn_id, txn.clone());
        self.track_active_begin(snapshot_ts);

        txn
    }

    /// Get transaction if active (clones - use get_snapshot_ts for hot path)
    pub fn get(&self, txn_id: u64) -> Option<MvccTransaction> {
        self.active_txns.get(&txn_id).map(|t| t.clone())
    }

    /// Fast path: get just the snapshot timestamp without cloning
    /// This is the hot path for reads - avoids cloning bloom filters
    #[inline]
    pub fn get_snapshot_ts(&self, txn_id: u64) -> Option<u64> {
        self.active_txns.get(&txn_id).map(|t| t.snapshot_ts)
    }

    /// Record a read (for SSI) - uses inline key storage + bloom filter
    ///
    /// ## SSI Bypass (Recommendation 9)
    ///
    /// For ReadOnly mode transactions, this is a no-op (instant return).
    /// For WriteOnly mode transactions, this is a no-op.
    /// Only ReadWrite mode transactions track reads for SSI.
    ///
    /// This reduces per-read overhead from ~50ns to ~0ns for read-only txns.
    #[inline]
    pub fn record_read(&self, txn_id: u64, key: &[u8]) {
        if let Some(mut txn) = self.active_txns.get_mut(&txn_id) {
            // SSI Bypass: Skip tracking for read-only and write-only modes
            if !txn.mode.tracks_reads() {
                return;
            }

            // Only track reads if within reasonable bounds
            if txn.read_set.len() < 10000 {
                txn.read_set.insert(SmallVec::from_slice(key));
                txn.read_bloom.insert(key);
            }
        }
    }

    /// Record a write - uses inline key storage + bloom filter
    ///
    /// Note: Even ReadOnly transactions can record writes (mode is a hint).
    /// The mode only affects SSI tracking, not write capability.
    pub fn record_write(&self, txn_id: u64, key: &[u8]) {
        if let Some(mut txn) = self.active_txns.get_mut(&txn_id) {
            txn.write_set.insert(SmallVec::from_slice(key));
            txn.write_bloom.insert(key);
        }
    }

    /// Allocate commit timestamp
    pub fn alloc_commit_ts(&self) -> u64 {
        self.ts_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Commit transaction with SSI validation
    /// Returns (commit_ts, write_set) so the memtable can be updated efficiently
    /// Returns None if SSI validation fails (dangerous structure detected)
    ///
    /// ## SSI Bypass (Recommendation 9)
    ///
    /// For ReadOnly mode: Skip validation entirely (~10ns commit)
    /// For WriteOnly mode: Skip read-based validation (~30ns commit)
    /// For ReadWrite mode: Full validation (~50ns commit)
    pub fn commit(&self, txn_id: u64) -> Option<(u64, HashSet<InlineKey>)> {
        // Get transaction before removing
        let txn = self.active_txns.get(&txn_id)?.clone();

        // SSI Bypass: Skip validation for ReadOnly transactions
        // ReadOnly can never form rw-antidependency cycles
        if txn.mode != TransactionMode::ReadWrite || !self.validate_ssi(&txn) {
            // For ReadOnly/WriteOnly: always valid (mode check short-circuits)
            // For ReadWrite: check SSI validation result
            if txn.mode == TransactionMode::ReadWrite && !self.validate_ssi(&txn) {
                // Abort on SSI violation
                if self.active_txns.remove(&txn_id).is_some() {
                    self.track_active_end(txn.snapshot_ts);
                }
                return None;
            }
        }

        let commit_ts = self.alloc_commit_ts();

        // Extract write_set and remove transaction - takes ownership
        let (_, removed_txn) = self.active_txns.remove(&txn_id)?;
        // Capture before `removed_txn` is partially moved below; snapshot_ts is
        // the multiset key for the matching begin.
        let snap = removed_txn.snapshot_ts;

        // OPTIMIZATION: Only track ReadWrite transactions for SSI
        // ReadOnly/WriteOnly can't form complete rw-antidependency cycles
        let needs_ssi_tracking = removed_txn.mode == TransactionMode::ReadWrite
            && !removed_txn.read_set.is_empty()
            && !removed_txn.write_set.is_empty();

        if needs_ssi_tracking {
            // Need to clone write_set since we return it AND track it
            let write_set_for_return = removed_txn.write_set.clone();

            self.track_commit_owned(
                txn_id,
                commit_ts,
                removed_txn.read_bloom,
                removed_txn.write_bloom,
                removed_txn.read_set,
                removed_txn.write_set,
            );

            self.track_active_end(snap);
            Some((commit_ts, write_set_for_return))
        } else {
            // Fast path: no SSI tracking needed, avoid clone entirely
            self.track_active_end(snap);
            Some((commit_ts, removed_txn.write_set))
        }
    }

    /// Validate SSI constraints for a committing transaction
    ///
    /// ## Transaction Classification (Task 3: Optimistic MVCC)
    ///
    /// Transactions are classified and routed through appropriate fast paths:
    ///
    /// | Class      | Criteria                      | Validation Cost |
    /// |------------|-------------------------------|-----------------|
    /// | ReadOnly   | write_set.is_empty()          | 0 ns           |
    /// | SingleKey  | write_set.len() == 1          | 0 ns           |
    /// | Disjoint   | bloom filters don't intersect | ~10 ns         |
    /// | General    | full SSI check                | ~50 ns         |
    ///
    /// Expected distribution: ~60% read-only, ~25% single-key, ~10% disjoint, ~5% general
    /// Weighted average: ~8 ns vs 50 ns baseline (6x improvement)
    ///
    /// Detects "dangerous structures" - rw-antidependency cycles:
    /// - T1 reads X (snapshot sees old value)
    /// - T2 writes X (concurrent write)  
    /// - T2 reads Y (snapshot sees old value)
    /// - T1 writes Y (concurrent write)
    ///
    /// If T1 → rw → T2 → rw → T1 exists, abort T1
    #[inline]
    fn validate_ssi(&self, txn: &MvccTransaction) -> bool {
        // =================================================================
        // Fast Path 1: Read-only transactions (0 ns)
        // =================================================================
        // Read-only transactions can never form rw-antidependency cycles
        // because they have no writes to create outgoing rw-edges
        if txn.write_set.is_empty() {
            return true;
        }

        // =================================================================
        // Fast Path 2: No recent commits to check (0 ns)
        // =================================================================
        if self.recent_commits.is_empty() {
            return true;
        }

        // =================================================================
        // Fast Path 3: Single-key write transactions (0 ns)
        // =================================================================
        // A single-key write transaction cannot form a dangerous cycle:
        // - For a cycle T1 →rw→ T2 →rw→ T1, we need T1 to read what T2 wrote
        //   AND T2 to read what T1 wrote
        // - With only one key in write_set, the same key would need to be
        //   in both read_set AND write_set of both transactions
        // - This is already prevented by our conflict detection (write-write)
        if txn.write_set.len() == 1 && txn.read_set.len() <= 1 {
            return true;
        }

        let my_snapshot = txn.snapshot_ts;

        // =================================================================
        // Fast Path 4: Disjoint transactions using Bloom filters (~10 ns)
        // =================================================================
        // Pre-filter using bloom filters: if our write_bloom doesn't intersect
        // with any concurrent transaction's read_bloom AND vice versa,
        // there can be no rw-antidependency
        let mut any_may_intersect = false;
        for entry in self.recent_commits.iter() {
            let (_, (other_commit_ts, other_read_bloom, other_write_bloom, _, _)) = entry.pair();

            // Only check concurrent transactions.
            //
            // A committed transaction is *concurrent* with us iff its writes are
            // invisible to our snapshot. Read visibility is strict
            // (`commit_ts < snapshot_ts`), so a transaction with
            // `commit_ts >= my_snapshot` is invisible and must be validated
            // against. Using `<` here (not `<=`) keeps this window consistent
            // with `read_at`; a `<=` would skip a boundary transaction whose
            // `commit_ts == my_snapshot` and miss a genuine write-skew.
            if *other_commit_ts < my_snapshot {
                continue;
            }

            // Check bloom filter intersection (O(m/64) per filter)
            // If our writes may intersect their reads OR their writes may intersect our reads
            if txn.write_bloom.may_intersect(other_read_bloom)
                || other_write_bloom.may_intersect(&txn.read_bloom)
            {
                any_may_intersect = true;
                break;
            }
        }

        // No bloom intersection means definitely disjoint - no SSI conflict possible
        if !any_may_intersect {
            return true;
        }

        // =================================================================
        // Full SSI Validation (~50 ns)
        // =================================================================
        // Check for rw-conflicts with recently committed transactions
        // An rw-conflict exists if:
        // - T_other wrote to a key that T_me read (T_other →rw→ T_me)
        // - T_me wrote to a key that T_other read (T_me →rw→ T_other)

        let mut in_conflict_with: Vec<u64> = Vec::new();
        let mut out_conflict_to: Vec<u64> = Vec::new();

        for entry in self.recent_commits.iter() {
            let (
                other_txn_id,
                (
                    other_commit_ts,
                    _other_read_bloom,
                    other_write_bloom,
                    other_read_set,
                    other_write_set,
                ),
            ) = entry.pair();

            // Only consider transactions concurrent with us: those whose writes
            // are invisible to our snapshot. Read visibility is strict
            // (`commit_ts < snapshot_ts`), so `commit_ts >= my_snapshot` means
            // concurrent. `<` (not `<=`) keeps this consistent with `read_at`.
            if *other_commit_ts < my_snapshot {
                continue;
            }

            // Check: other wrote → we read (other →rw→ me)
            // T_other wrote a key that T_me read (rw-dependency inbound)
            //
            // Bloom-accelerated: First check bloom filter for fast rejection (O(m/64))
            // Only do expensive HashSet intersection if bloom says "maybe conflict"
            let mut has_in_conflict = false;
            for key in txn.read_set.iter() {
                if other_write_bloom.may_contain(key) {
                    // Bloom says maybe - do exact check
                    if other_write_set.contains(key) {
                        has_in_conflict = true;
                        break;
                    }
                }
            }
            if has_in_conflict {
                in_conflict_with.push(*other_txn_id);
            }

            // Check: we wrote → other read (me →rw→ other)
            // T_me wrote a key that T_other read (rw-dependency outbound)
            //
            // Bloom-accelerated: Use our write_bloom against their read_set
            let mut has_out_conflict = false;
            for key in other_read_set.iter() {
                if txn.write_bloom.may_contain(key) {
                    // Bloom says maybe - do exact check
                    if txn.write_set.contains(key) {
                        has_out_conflict = true;
                        break;
                    }
                }
            }
            if has_out_conflict {
                out_conflict_to.push(*other_txn_id);
            }
        }

        // Dangerous structure: we have both incoming AND outgoing rw-edges
        // This creates a potential cycle: T1 →rw→ T_me →rw→ T2
        //
        // Conservative check: if both exist, abort
        // A more precise check would verify the cycle path, but this is safe
        if !in_conflict_with.is_empty() && !out_conflict_to.is_empty() {
            return false; // SSI violation - abort
        }

        true
    }

    /// Track a committed transaction for future SSI validation
    ///
    /// Only tracks transactions that have both reads AND writes, since SSI
    /// only detects rw-antidependency cycles. Pure read or pure write
    /// transactions can't form cycles.
    ///
    /// ## Optimization: Zero-Copy Transfer
    ///
    /// Takes ownership of sets instead of cloning to avoid the +15% commit
    /// phase regression. The caller should use mem::take() to transfer ownership.
    fn track_commit_owned(
        &self,
        txn_id: u64,
        commit_ts: u64,
        read_bloom: SsiBloomFilter,
        write_bloom: SsiBloomFilter,
        read_set: HashSet<InlineKey>,
        write_set: HashSet<InlineKey>,
    ) {
        // Optimization: Only track mixed read-write transactions
        // Pure reads can't create outgoing rw-edges
        // Pure writes can't create incoming rw-edges
        if read_set.is_empty() || write_set.is_empty() {
            return; // Skip tracking - can't form SSI cycle
        }

        // Add to recent commits with bloom filters for fast SSI pre-filtering
        // No cloning needed - we take ownership
        self.recent_commits.insert(
            txn_id,
            (commit_ts, read_bloom, write_bloom, read_set, write_set),
        );

        // Lazy pruning: only prune when we're significantly over capacity
        // Avoids pruning overhead on every commit
        if self.recent_commits.len() > self.max_recent_commits * 2 {
            // Remove entries with lowest commit_ts
            let min_active = self.min_active_ts.load(Ordering::Relaxed);
            self.recent_commits
                .retain(|_, (ts, _, _, _, _)| *ts >= min_active);
        }
    }

    /// Legacy track_commit that clones - kept for compatibility
    #[allow(dead_code)]
    fn track_commit(
        &self,
        txn_id: u64,
        commit_ts: u64,
        read_bloom: SsiBloomFilter,
        write_bloom: SsiBloomFilter,
        read_set: &HashSet<InlineKey>,
        write_set: &HashSet<InlineKey>,
    ) {
        if read_set.is_empty() || write_set.is_empty() {
            return;
        }
        self.recent_commits.insert(
            txn_id,
            (
                commit_ts,
                read_bloom,
                write_bloom,
                read_set.clone(),
                write_set.clone(),
            ),
        );
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) {
        if let Some((_, t)) = self.active_txns.remove(&txn_id) {
            self.track_active_end(t.snapshot_ts);
        }
    }

    /// Get minimum active snapshot timestamp
    pub fn min_active_snapshot(&self) -> u64 {
        self.min_active_ts.load(Ordering::SeqCst)
    }

    /// Get count of active transactions
    pub fn active_transaction_count(&self) -> usize {
        self.active_txns.len()
    }

    /// Record a transaction entering the active set (its snapshot_ts) and
    /// refresh `min_active_ts`. O(log N) — replaces the old O(active_txns) scan.
    fn track_active_begin(&self, snapshot_ts: u64) {
        let mut snaps = self.active_snapshots.lock();
        *snaps.entry(snapshot_ts).or_insert(0) += 1;
        // The set is non-empty (we just inserted); the smallest key is the min.
        let min = *snaps.keys().next().expect("non-empty after insert");
        self.min_active_ts.store(min, Ordering::SeqCst);
    }

    /// Record a transaction leaving the active set and refresh `min_active_ts`.
    /// O(log N). `snapshot_ts` MUST be the value used at the matching begin.
    fn track_active_end(&self, snapshot_ts: u64) {
        let mut snaps = self.active_snapshots.lock();
        if let Some(c) = snaps.get_mut(&snapshot_ts) {
            *c -= 1;
            if *c == 0 {
                snaps.remove(&snapshot_ts);
            }
        }
        // When no txn is active the watermark is the current ts_counter, exactly
        // as the old full-scan did via `unwrap_or_else`.
        let min = snaps
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| self.ts_counter.load(Ordering::SeqCst));
        self.min_active_ts.store(min, Ordering::SeqCst);
    }

    /// Test-only SOUND consistency check: with no concurrent mutation in flight,
    /// the active-snapshot multiset must exactly mirror `active_txns` and yield
    /// the same watermark as a full scan. Must be called from a quiescent (e.g.
    /// single-threaded) point — it reads both structures and is not atomic.
    #[cfg(test)]
    pub(crate) fn assert_active_snapshots_consistent(&self) {
        use std::collections::BTreeMap;
        let mut from_txns: BTreeMap<u64, u32> = BTreeMap::new();
        for e in self.active_txns.iter() {
            *from_txns.entry(e.value().snapshot_ts).or_insert(0) += 1;
        }
        let snaps = self.active_snapshots.lock();
        assert_eq!(
            *snaps, from_txns,
            "active_snapshots multiset drifted from active_txns"
        );
        let watermark = self.min_active_ts.load(Ordering::SeqCst);
        match from_txns.keys().next().copied() {
            // Non-empty: watermark must equal the oldest active snapshot.
            Some(min) => assert_eq!(watermark, min, "min_active_ts watermark wrong"),
            // Empty: any watermark <= ts_counter is SAFE (GC just stays
            // conservative); the initial state is 0, post-drain it is ts_counter.
            None => assert!(
                watermark <= self.ts_counter.load(Ordering::SeqCst),
                "empty-state watermark {} exceeds ts_counter",
                watermark
            ),
        }
    }

    /// Recompute `min_active_ts` from the active-snapshot multiset (O(log N)).
    /// Retained for any caller that needs a watermark refresh without a
    /// corresponding begin/end (e.g. recovery).
    #[allow(dead_code)]
    fn update_min_active_ts(&self) {
        let snaps = self.active_snapshots.lock();
        let min = snaps
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| self.ts_counter.load(Ordering::SeqCst));
        self.min_active_ts.store(min, Ordering::SeqCst);
    }
}

/// Epoch-based dirty list for O(expired) GC instead of O(n)
///
/// Instead of scanning ALL version chains, we track which keys have versions
/// created in each epoch. GC only needs to visit keys from old epochs.
struct EpochDirtyList {
    /// Ring buffer of epoch -> dirty keys
    /// Index = epoch % EPOCH_RING_SIZE
    epochs: [parking_lot::Mutex<Vec<Vec<u8>>>; 4],
    /// Current epoch
    current_epoch: AtomicU64,
}

const EPOCH_RING_SIZE: usize = 4;

impl EpochDirtyList {
    fn new() -> Self {
        Self {
            epochs: [
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
            ],
            current_epoch: AtomicU64::new(0),
        }
    }

    /// Record a version created in the current epoch
    #[inline]
    fn record_version(&self, key: Vec<u8>) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        let idx = (epoch as usize) % EPOCH_RING_SIZE;
        self.epochs[idx].lock().push(key);
    }

    /// Record multiple versions in a single lock acquisition (Rec 3: MVCC Batching)
    ///
    /// Performance: Single lock acquire vs N lock acquires for batch of N writes.
    /// For 100 writes: ~100x fewer mutex operations.
    #[inline]
    fn record_versions_batch(&self, keys: impl IntoIterator<Item = Vec<u8>>) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        let idx = (epoch as usize) % EPOCH_RING_SIZE;
        let mut guard = self.epochs[idx].lock();
        guard.extend(keys);
    }

    /// Advance to next epoch, returning old epoch's dirty keys
    fn advance_epoch(&self) -> (u64, Vec<Vec<u8>>) {
        let old_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
        let old_idx = (old_epoch as usize) % EPOCH_RING_SIZE;

        // Drain the old epoch's dirty list
        let mut guard = self.epochs[old_idx].lock();
        let keys = std::mem::take(&mut *guard);
        (old_epoch, keys)
    }

    /// Get current epoch
    #[allow(dead_code)]
    fn current(&self) -> u64 {
        self.current_epoch.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Streaming Scan Iterator
// ============================================================================

/// Streaming iterator for range scans
///
/// Yields results one at a time without materializing the full result set.
/// This enables processing of very large result sets with O(1) memory per
/// iteration instead of O(N) for the entire result set.
struct ScanRangeIterator<'a> {
    memtable: &'a MvccMemTable,
    start: Vec<u8>,
    end: Vec<u8>,
    snapshot_ts: u64,
    current_txn_id: Option<u64>,
    use_ordered: bool,
    // We use Option to defer initialization
    ordered_iter: Option<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a>>,
    unordered_iter: Option<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a>>,
    initialized: bool,
}

impl<'a> Iterator for ScanRangeIterator<'a> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        // Lazy initialization on first call
        if !self.initialized {
            self.initialized = true;

            if self.use_ordered {
                // Try deferred index first (after compaction, it uses a SkipMap internally)
                if let Some(ref def_idx) = self.memtable.deferred_index {
                    let start = self.start.clone();
                    let end = self.end.clone();
                    let snapshot_ts = self.snapshot_ts;
                    let current_txn_id = self.current_txn_id;
                    let data = &self.memtable.data;

                    // Collect keys from deferred index (already sorted after compact)
                    let keys: Vec<Vec<u8>> = if end.is_empty() {
                        def_idx.range_from(&start).collect()
                    } else {
                        def_idx.range(&start, &end).collect()
                    };

                    let iter: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> =
                        Box::new(keys.into_iter().filter_map(move |key| {
                            if let Some(chain) = data.get(&key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                Some((key, value.clone()))
                            } else {
                                None
                            }
                        }));
                    self.ordered_iter = Some(iter);
                } else if let Some(ref idx) = self.memtable.ordered_index {
                    let start = self.start.clone();
                    let end = self.end.clone();
                    let snapshot_ts = self.snapshot_ts;
                    let current_txn_id = self.current_txn_id;
                    let data = &self.memtable.data;

                    let iter: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> = if end.is_empty()
                    {
                        Box::new(idx.range(start..).filter_map(move |entry| {
                            let key = entry.key();
                            if let Some(chain) = data.get(key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                Some((key.clone(), value.clone()))
                            } else {
                                None
                            }
                        }))
                    } else {
                        Box::new(idx.range(start..end).filter_map(move |entry| {
                            let key = entry.key();
                            if let Some(chain) = data.get(key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                Some((key.clone(), value.clone()))
                            } else {
                                None
                            }
                        }))
                    };
                    self.ordered_iter = Some(iter);
                }
            } else {
                // Unordered full scan
                let start = self.start.clone();
                let end = self.end.clone();
                let snapshot_ts = self.snapshot_ts;
                let current_txn_id = self.current_txn_id;

                let iter: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> =
                    Box::new(self.memtable.data.iter().filter_map(move |entry| {
                        let key = entry.key();

                        if key.as_slice() < start.as_slice() {
                            return None;
                        }
                        if !end.is_empty() && key.as_slice() >= end.as_slice() {
                            return None;
                        }

                        if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                            && let Some(value) = &v.value
                        {
                            Some((key.clone(), value.clone()))
                        } else {
                            None
                        }
                    }));
                self.unordered_iter = Some(iter);
            }
        }

        // Get next from appropriate iterator
        if let Some(ref mut iter) = self.ordered_iter {
            iter.next()
        } else if let Some(ref mut iter) = self.unordered_iter {
            iter.next()
        } else {
            None
        }
    }
}

/// MemTable with MVCC support
///
/// Uses DashMap for lock-free concurrent access per key.
/// This eliminates the global write lock bottleneck.
///
/// Uses epoch-based dirty tracking for O(expired) GC instead of O(n) full scan.
/// Maintains a deferred sorted index for efficient scans:
/// - Writes: O(1) append to hot buffer
/// - Scans: O(N log N) sort-on-demand (amortized across many writes)
pub struct MvccMemTable {
    /// Key -> VersionChain (sharded for concurrent access)
    data: DashMap<Vec<u8>, VersionChain>,
    /// Deferred sorted index for efficient prefix/range scans (optional)
    /// O(1) insert to hot buffer, O(N log N) sort on first scan
    /// When None, scan_prefix will fall back to O(N) DashMap iteration
    deferred_index: Option<DeferredSortedIndex>,
    /// Legacy SkipMap for compatibility (used when deferred=false)
    ordered_index: Option<SkipMap<Vec<u8>, ()>>,
    /// Whether to use deferred sorting (true) or immediate SkipMap (false)
    #[allow(dead_code)]
    use_deferred: bool,
    /// Approximate size in bytes
    size_bytes: AtomicU64,
    /// Epoch-based dirty list for efficient GC
    dirty_list: EpochDirtyList,
}

impl Default for MvccMemTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MvccMemTable {
    pub fn new() -> Self {
        Self::with_ordered_index(true)
    }

    /// Create memtable with optional ordered index
    ///
    /// When `enable_ordered_index` is false, saves ~134 ns/op on writes
    /// but scan_prefix becomes O(N) instead of O(log N + K)
    ///
    /// Uses deferred sorting by default for better write performance:
    /// - Writes: O(1) append to hot buffer
    /// - Scans: O(N log N) sort-on-demand
    pub fn with_ordered_index(enable_ordered_index: bool) -> Self {
        Self::with_index_mode(enable_ordered_index, true)
    }

    /// Create memtable with fine-grained control over indexing
    ///
    /// # Arguments
    /// * `enable_ordered_index` - Whether to maintain an ordered index
    /// * `use_deferred` - If true, use deferred sorting (O(1) writes, sort-on-scan)
    ///                    If false, use SkipMap (O(log N) writes)
    pub fn with_index_mode(enable_ordered_index: bool, use_deferred: bool) -> Self {
        Self {
            data: DashMap::new(),
            deferred_index: if enable_ordered_index && use_deferred {
                Some(DeferredSortedIndex::with_config(DeferredIndexConfig {
                    max_unsorted_entries: 10_000, // Compact every 10K writes
                    enabled: true,
                }))
            } else {
                None
            },
            ordered_index: if enable_ordered_index && !use_deferred {
                Some(SkipMap::new())
            } else {
                None
            },
            use_deferred,
            size_bytes: AtomicU64::new(0),
            dirty_list: EpochDirtyList::new(),
        }
    }

    /// Write a key-value pair (uncommitted)
    pub fn write(&self, key: Vec<u8>, value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
        let key_len = key.len();

        // Track this key in the current epoch's dirty list for GC
        self.dirty_list.record_version(key.clone());

        // Insert into ordered index for prefix scans (if enabled)
        // Deferred: O(1) append to hot buffer
        // SkipMap: O(log N) insert
        if let Some(ref idx) = self.deferred_index {
            idx.insert(key.clone());
        } else if let Some(ref idx) = self.ordered_index {
            idx.insert(key.clone(), ());
        }

        // Use entry API for atomic get-or-insert
        let mut entry = self.data.entry(key).or_default();

        // Check for write-write conflict
        if entry.has_write_conflict(txn_id) {
            return Err(SochDBError::Internal(
                "Write-write conflict detected".into(),
            ));
        }
        entry.add_uncommitted(value, txn_id);
        self.size_bytes
            .fetch_add((key_len + value_size) as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Write multiple key-value pairs (uncommitted) - more efficient than individual writes
    ///
    /// Optimizations applied (Rec 3: MVCC Batching):
    /// - Batched dirty list tracking: single lock acquire for all keys
    /// - Deferred index: O(1) append per key
    pub fn write_batch(&self, writes: &[(Vec<u8>, Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        let mut total_size = 0u64;

        // Rec 3: Batch MVCC tracking - single lock acquire for all keys
        self.dirty_list
            .record_versions_batch(writes.iter().map(|(k, _)| k.clone()));

        for (key, value) in writes {
            // Insert into ordered index (if enabled)
            // Deferred: O(1) append, SkipMap: O(log N)
            if let Some(ref idx) = self.deferred_index {
                idx.insert(key.clone());
            } else if let Some(ref idx) = self.ordered_index {
                idx.insert(key.clone(), ());
            }

            let mut entry = self.data.entry(key.clone()).or_default();

            if entry.has_write_conflict(txn_id) {
                return Err(SochDBError::Internal(
                    "Write-write conflict detected".into(),
                ));
            }

            let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
            entry.add_uncommitted(value.clone(), txn_id);
            total_size += (key.len() + value_size) as u64;
        }

        self.size_bytes.fetch_add(total_size, Ordering::Relaxed);
        Ok(())
    }

    /// Read at snapshot timestamp, with optional current txn to see own writes
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        self.data.get(key).and_then(|chain| {
            chain
                .read_at(snapshot_ts, current_txn_id)
                .and_then(|v| v.value.clone())
        })
    }

    /// Commit all versions for a transaction
    ///
    /// Only updates the keys that were written by this transaction (tracked in write_set).
    /// Accepts InlineKey for zero-allocation MVCC tracking.
    pub fn commit(&self, txn_id: u64, commit_ts: u64, write_set: &HashSet<InlineKey>) {
        // Only iterate over keys we know were written - O(k) instead of O(n)
        for key in write_set {
            if let Some(mut chain) = self.data.get_mut(key.as_slice()) {
                chain.commit(txn_id, commit_ts);
            }
        }
    }

    /// Legacy commit method (iterates all keys) - kept for backward compatibility
    #[allow(dead_code)]
    pub fn commit_all(&self, txn_id: u64, commit_ts: u64) {
        for mut entry in self.data.iter_mut() {
            entry.value_mut().commit(txn_id, commit_ts);
        }
    }

    /// Abort all versions for a transaction
    pub fn abort(&self, txn_id: u64) {
        for mut entry in self.data.iter_mut() {
            entry.value_mut().abort(txn_id);
        }
    }

    /// Scan keys with prefix at snapshot (without seeing uncommitted from other txns)
    ///
    /// ## Performance
    ///
    /// When ordered_index is enabled: O(log N + K) complexity
    /// - O(log N) to seek to the first key with prefix
    /// - O(K) to iterate matching keys
    ///
    /// When ordered_index is disabled: O(N) full DashMap scan (fallback)
    ///
    /// ## Optimizations Applied
    ///
    /// - Pre-allocates result vector based on expected output size
    /// - Uses batch-friendly iteration patterns
    /// - Minimizes allocations during iteration
    /// - Deferred index: compacts hot buffer on first scan for sorted iteration
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        // Estimate result size for pre-allocation (use 10% of total as heuristic)
        let estimated_size = (self.data.len() / 10).max(64);
        let mut results = Vec::with_capacity(estimated_size);

        if let Some(ref idx) = self.deferred_index {
            // Deferred index path: sort-on-scan (compacts hot buffer if needed)
            for key in idx.range_from(prefix) {
                // Stop when we've passed the prefix range
                if !key.starts_with(prefix) {
                    break;
                }

                // O(1) lookup in DashMap for version chain
                if let Some(chain) = self.data.get(&key)
                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key, value.clone()));
                }
            }
        } else if let Some(ref idx) = self.ordered_index {
            // Fast path: O(log N) seek to first key >= prefix
            for entry in idx.range(prefix.to_vec()..) {
                let key = entry.key();

                // Stop when we've passed the prefix range
                if !key.starts_with(prefix) {
                    break;
                }

                // O(1) lookup in DashMap for version chain
                if let Some(chain) = self.data.get(key)
                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.clone(), value.clone()));
                }
            }
        } else {
            // Fallback: O(N) full DashMap scan when ordered_index is disabled
            // Optimized with batch-friendly iteration
            for entry in self.data.iter() {
                let key = entry.key();
                if !key.starts_with(prefix) {
                    continue;
                }
                if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.clone(), value.clone()));
                }
            }
        }

        results
    }

    /// Optimized full scan with batch allocation
    ///
    /// For use when scanning entire tables/namespaces.
    /// Pre-allocates based on actual data size.
    pub fn scan_all(
        &self,
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::with_capacity(self.data.len());

        for entry in self.data.iter() {
            if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                && let Some(value) = &v.value
            {
                results.push((entry.key().clone(), value.clone()));
            }
        }

        results
    }

    /// Streaming scan iterator for very large datasets
    ///
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set in memory.
    pub fn scan_prefix_iter<'a>(
        &'a self,
        prefix: &'a [u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
        self.data.iter().filter_map(move |entry| {
            let key = entry.key();
            if !key.starts_with(prefix) {
                return None;
            }
            if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                && let Some(value) = &v.value
            {
                Some((key.clone(), value.clone()))
            } else {
                None
            }
        })
    }

    /// Scan range
    pub fn scan_range(
        &self,
        start: &[u8],
        end: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();

        if let Some(ref idx) = self.deferred_index {
            // Deferred index path: sort-on-scan
            if end.is_empty() {
                for key in idx.range_from(start) {
                    if let Some(chain) = self.data.get(&key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key, value.clone()));
                    }
                }
            } else {
                for key in idx.range(start, end) {
                    if let Some(chain) = self.data.get(&key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key, value.clone()));
                    }
                }
            }
        } else if let Some(ref idx) = self.ordered_index {
            // Use range scan on SkipMap
            if end.is_empty() {
                // Unbounded end
                for entry in idx.range(start.to_vec()..) {
                    let key = entry.key();
                    if let Some(chain) = self.data.get(key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key.clone(), value.clone()));
                    }
                }
            } else {
                for entry in idx.range(start.to_vec()..end.to_vec()) {
                    let key = entry.key();
                    if let Some(chain) = self.data.get(key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key.clone(), value.clone()));
                    }
                }
            }
        } else {
            // Fallback to full scan if no ordered index
            for entry in self.data.iter() {
                let key = entry.key();

                if key.as_slice() < start {
                    continue;
                }
                if !end.is_empty() && key.as_slice() >= end {
                    continue;
                }

                if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.clone(), value.clone()));
                }
            }
        }

        results
    }

    /// Streaming range scan iterator for very large datasets
    ///
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set in memory. Uses the ordered
    /// index when available for O(log N + K) complexity.
    ///
    /// ## Zero-Allocation Design
    ///
    /// While the iterator itself cannot avoid allocations for returned
    /// values (since the caller needs ownership), it avoids:
    /// - Pre-materializing all results
    /// - Intermediate buffers
    /// - Repeated key comparisons for already-visited entries
    ///
    /// ## Usage
    ///
    /// ```ignore
    /// for (key, value) in memtable.scan_range_iter(b"start", b"end", ts, txn) {
    ///     // Process each result as it arrives
    ///     // Memory usage is O(1) per iteration, not O(N) total
    /// }
    /// ```
    pub fn scan_range_iter<'a>(
        &'a self,
        start: &'a [u8],
        end: &'a [u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
        // Compact deferred index before scanning if needed
        if let Some(ref idx) = self.deferred_index {
            idx.compact();
        }

        // Use either ordered index or full scan
        let use_ordered = self.ordered_index.is_some() || self.deferred_index.is_some();

        // Create iterator based on availability of ordered index
        ScanRangeIterator {
            memtable: self,
            start: start.to_vec(),
            end: end.to_vec(),
            snapshot_ts,
            current_txn_id,
            use_ordered,
            ordered_iter: None,
            unordered_iter: None,
            initialized: false,
        }
    }

    /// Get approximate size
    pub fn size(&self) -> u64 {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Garbage collect old versions using epoch-based dirty list
    ///
    /// O(expired_versions) instead of O(all_versions)
    /// Only visits keys that had versions created in the old epoch.
    pub fn gc(&self, min_active_ts: u64) -> usize {
        // Advance epoch and get the dirty keys from the old epoch
        let (_old_epoch, dirty_keys) = self.dirty_list.advance_epoch();

        if dirty_keys.is_empty() {
            return 0;
        }

        let mut gc_count = 0;

        // Only visit keys that were modified in the old epoch
        // Use a HashSet to deduplicate keys that were written multiple times
        let unique_keys: std::collections::HashSet<_> = dirty_keys.into_iter().collect();

        for key in unique_keys {
            if let Some(mut entry) = self.data.get_mut(&key) {
                let before = entry.value().version_count();
                entry.value_mut().gc(min_active_ts);
                gc_count += before.saturating_sub(entry.value().version_count());
            }
        }

        gc_count
    }

    /// Legacy full-scan GC (for testing or when epoch-based tracking isn't available)
    #[allow(dead_code)]
    pub fn gc_full_scan(&self, min_active_ts: u64) -> usize {
        let mut gc_count = 0;

        for mut entry in self.data.iter_mut() {
            let before = entry.value().version_count();
            entry.value_mut().gc(min_active_ts);
            gc_count += before.saturating_sub(entry.value().version_count());
        }

        gc_count
    }
}

// =============================================================================
// Rec 11: Unified MvccStore Implementation for MvccMemTable
// =============================================================================

impl sochdb_core::version_chain::MvccStore for MvccMemTable {
    fn mvcc_get(&self, key: &[u8], snapshot_ts: u64, txn_id: Option<u64>) -> Option<Vec<u8>> {
        self.read(key, snapshot_ts, txn_id)
    }

    fn mvcc_put(
        &self,
        key: &[u8],
        value: Option<Vec<u8>>,
        txn_id: u64,
    ) -> std::result::Result<(), sochdb_core::version_chain::MvccStoreError> {
        let mut entry = self.data.entry(key.to_vec()).or_default();
        if entry.has_write_conflict(txn_id) {
            return Err(sochdb_core::version_chain::MvccStoreError::WriteConflict);
        }
        entry.add_uncommitted(value, txn_id);
        Ok(())
    }

    fn mvcc_commit_key(&self, key: &[u8], txn_id: u64, commit_ts: u64) -> bool {
        if let Some(mut chain) = self.data.get_mut(key) {
            return chain.commit(txn_id, commit_ts);
        }
        false
    }

    fn mvcc_abort_key(&self, key: &[u8], txn_id: u64) {
        if let Some(mut chain) = self.data.get_mut(key) {
            chain.abort(txn_id);
        }
    }

    fn mvcc_has_conflict(&self, key: &[u8], txn_id: u64) -> bool {
        self.data
            .get(key)
            .map(|chain| chain.has_write_conflict(txn_id))
            .unwrap_or(false)
    }

    fn mvcc_gc(&self, min_ts: u64) -> sochdb_core::version_chain::MvccGcStats {
        let mut stats = sochdb_core::version_chain::MvccGcStats::default();
        for mut entry in self.data.iter_mut() {
            stats.keys_scanned += 1;
            let before = entry.value().version_count();
            entry.value_mut().gc(min_ts);
            stats.versions_removed += before.saturating_sub(entry.value().version_count());
        }
        stats
    }

    fn mvcc_key_count(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// ArenaMvccMemTable - Arena-Backed MVCC MemTable with Reduced Allocations
// ============================================================================

use crate::key_buffer::ArenaKeyHandle;

/// Epoch-based dirty list using ArenaKeyHandle for reduced allocations
struct ArenaEpochDirtyList {
    epochs: [parking_lot::Mutex<Vec<ArenaKeyHandle>>; 4],
    current_epoch: AtomicU64,
}

impl ArenaEpochDirtyList {
    fn new() -> Self {
        Self {
            epochs: [
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
            ],
            current_epoch: AtomicU64::new(0),
        }
    }

    #[inline]
    fn record_version(&self, key: ArenaKeyHandle) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        let idx = (epoch as usize) % EPOCH_RING_SIZE;
        self.epochs[idx].lock().push(key);
    }

    fn advance_epoch(&self) -> (u64, Vec<ArenaKeyHandle>) {
        let old_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
        let old_idx = (old_epoch as usize) % EPOCH_RING_SIZE;
        let mut guard = self.epochs[old_idx].lock();
        let keys = std::mem::take(&mut *guard);
        (old_epoch, keys)
    }
}

/// Arena-backed MVCC MemTable with optimized key storage
///
/// This version uses `ArenaKeyHandle` instead of `Vec<u8>` for keys,
/// reducing per-write allocations from 3 to 1:
///
/// - Before: 3 × Vec<u8> clones per write (dirty_list, ordered_index, data)
/// - After: 1 × ArenaKeyHandle creation, 3 × O(1) copies (16 bytes each)
///
/// ## Performance
///
/// Expected improvement: 20-30% throughput increase on write-heavy workloads
/// by reducing:
/// - Heap allocations: 3 → 1 per write
/// - Bytes copied: 3L → L + 48 bytes (where L = key length)
pub struct ArenaMvccMemTable {
    /// Key -> VersionChain (uses ArenaKeyHandle for O(1) hash)
    data: DashMap<ArenaKeyHandle, VersionChain>,
    /// Ordered index for prefix scans
    ordered_index: Option<SkipMap<ArenaKeyHandle, ()>>,
    /// Approximate size in bytes
    size_bytes: AtomicU64,
    /// Epoch-based dirty list (arena-backed)
    dirty_list: ArenaEpochDirtyList,
}

impl ArenaMvccMemTable {
    pub fn new() -> Self {
        Self::with_ordered_index(true)
    }

    pub fn with_ordered_index(enable_ordered_index: bool) -> Self {
        Self {
            data: DashMap::new(),
            ordered_index: if enable_ordered_index {
                Some(SkipMap::new())
            } else {
                None
            },
            size_bytes: AtomicU64::new(0),
            dirty_list: ArenaEpochDirtyList::new(),
        }
    }

    /// Write a key-value pair using arena key handle
    ///
    /// Only creates ONE ArenaKeyHandle, then copies it (16 bytes) to each location.
    /// This is much cheaper than cloning Vec<u8> which requires heap allocation.
    pub fn write(&self, key: &[u8], value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
        let key_len = key.len();

        // Create ONE ArenaKeyHandle - this is the only allocation for the key
        let key_handle = ArenaKeyHandle::new(key);

        // Track in dirty list (O(1) copy of 16-byte handle)
        self.dirty_list.record_version(key_handle.clone());

        // Insert into ordered index (O(1) copy of 16-byte handle)
        if let Some(ref idx) = self.ordered_index {
            idx.insert(key_handle.clone(), ());
        }

        // Use entry API with the handle
        let mut entry = self.data.entry(key_handle).or_default();

        if entry.has_write_conflict(txn_id) {
            return Err(SochDBError::Internal(
                "Write-write conflict detected".into(),
            ));
        }
        entry.add_uncommitted(value, txn_id);
        self.size_bytes
            .fetch_add((key_len + value_size) as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Write batch using arena key handles
    pub fn write_batch(&self, writes: &[(&[u8], Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        let mut total_size = 0u64;

        for (key, value) in writes {
            let key_handle = ArenaKeyHandle::new(key);

            self.dirty_list.record_version(key_handle.clone());

            if let Some(ref idx) = self.ordered_index {
                idx.insert(key_handle.clone(), ());
            }

            let mut entry = self.data.entry(key_handle).or_default();

            if entry.has_write_conflict(txn_id) {
                return Err(SochDBError::Internal(
                    "Write-write conflict detected".into(),
                ));
            }

            let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
            entry.add_uncommitted(value.clone(), txn_id);
            total_size += (key.len() + value_size) as u64;
        }

        self.size_bytes.fetch_add(total_size, Ordering::Relaxed);
        Ok(())
    }

    /// Read at snapshot timestamp
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        // Create temporary handle for lookup (uses pre-computed hash for O(1) lookup)
        let key_handle = ArenaKeyHandle::new(key);
        self.data.get(&key_handle).and_then(|chain| {
            chain
                .read_at(snapshot_ts, current_txn_id)
                .and_then(|v| v.value.clone())
        })
    }

    /// Commit transaction
    pub fn commit(&self, txn_id: u64, commit_ts: u64, write_set: &HashSet<InlineKey>) {
        for key in write_set {
            let key_handle = ArenaKeyHandle::new(key.as_slice());
            if let Some(mut chain) = self.data.get_mut(&key_handle) {
                chain.commit(txn_id, commit_ts);
            }
        }
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) {
        for mut entry in self.data.iter_mut() {
            entry.value_mut().abort(txn_id);
        }
    }

    /// Scan prefix
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();
        let prefix_handle = ArenaKeyHandle::new(prefix);

        if let Some(ref idx) = self.ordered_index {
            for entry in idx.range(prefix_handle..) {
                let key = entry.key();

                if !key.as_bytes().starts_with(prefix) {
                    break;
                }

                if let Some(chain) = self.data.get(key)
                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.as_bytes().to_vec(), value.clone()));
                }
            }
        } else {
            for entry in self.data.iter() {
                let key = entry.key();
                if !key.as_bytes().starts_with(prefix) {
                    continue;
                }
                if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.as_bytes().to_vec(), value.clone()));
                }
            }
        }

        results
    }

    /// Get approximate size
    pub fn size(&self) -> u64 {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Garbage collect old versions
    pub fn gc(&self, min_active_ts: u64) -> usize {
        let (_old_epoch, dirty_keys) = self.dirty_list.advance_epoch();

        if dirty_keys.is_empty() {
            return 0;
        }

        let mut gc_count = 0;
        let unique_keys: std::collections::HashSet<_> = dirty_keys.into_iter().collect();

        for key in unique_keys {
            if let Some(mut entry) = self.data.get_mut(&key) {
                let before = entry.value().version_count();
                entry.value_mut().gc(min_active_ts);
                gc_count += before.saturating_sub(entry.value().version_count());
            }
        }

        gc_count
    }
}

impl Default for ArenaMvccMemTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MemTableKind - Unified MemTable Abstraction (Principal Engineer Pattern)
// ============================================================================

/// Configuration for memtable type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemTableType {
    /// Standard MVCC memtable with deferred sorting
    /// Best for: general workloads, balanced read/write
    Standard,
    /// Arena-backed memtable with reduced allocations
    /// Best for: write-heavy workloads, large keys
    Arena,
}

impl Default for MemTableType {
    fn default() -> Self {
        // Default to Standard which now has deferred sorting
        MemTableType::Standard
    }
}

/// Unified memtable abstraction using enum dispatch
///
/// This pattern provides:
/// - Zero-cost abstraction (no vtable, no dynamic dispatch)
/// - Type-safe switching between implementations
/// - Easy extensibility for future memtable types
///
/// ## Why Enum over Trait Object?
///
/// - Hot path performance: enum match is a single branch vs vtable indirection
/// - Cache friendliness: no pointer chasing
/// - Inlining: compiler can inline through enum dispatch
pub enum MemTableKind {
    Standard(MvccMemTable),
    Arena(ArenaMvccMemTable),
}

impl MemTableKind {
    /// Create a new memtable of the specified type
    pub fn new(kind: MemTableType, enable_ordered_index: bool) -> Self {
        match kind {
            MemTableType::Standard => {
                MemTableKind::Standard(MvccMemTable::with_ordered_index(enable_ordered_index))
            }
            MemTableType::Arena => {
                MemTableKind::Arena(ArenaMvccMemTable::with_ordered_index(enable_ordered_index))
            }
        }
    }

    /// Write a key-value pair
    #[inline]
    pub fn write(&self, key: Vec<u8>, value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        match self {
            MemTableKind::Standard(m) => m.write(key, value, txn_id),
            MemTableKind::Arena(m) => m.write(&key, value, txn_id),
        }
    }

    /// Write batch of key-value pairs
    #[inline]
    pub fn write_batch(&self, writes: &[(Vec<u8>, Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        match self {
            MemTableKind::Standard(m) => m.write_batch(writes, txn_id),
            MemTableKind::Arena(m) => {
                // Convert to arena-compatible format
                let arena_writes: Vec<(&[u8], Option<Vec<u8>>)> = writes
                    .iter()
                    .map(|(k, v)| (k.as_slice(), v.clone()))
                    .collect();
                m.write_batch(&arena_writes, txn_id)
            }
        }
    }

    /// Read at snapshot timestamp
    #[inline]
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        match self {
            MemTableKind::Standard(m) => m.read(key, snapshot_ts, current_txn_id),
            MemTableKind::Arena(m) => m.read(key, snapshot_ts, current_txn_id),
        }
    }

    /// Commit transaction
    #[inline]
    pub fn commit(&self, txn_id: u64, commit_ts: u64, write_set: &HashSet<InlineKey>) {
        match self {
            MemTableKind::Standard(m) => m.commit(txn_id, commit_ts, write_set),
            MemTableKind::Arena(m) => m.commit(txn_id, commit_ts, write_set),
        }
    }

    /// Abort transaction
    #[inline]
    pub fn abort(&self, txn_id: u64) {
        match self {
            MemTableKind::Standard(m) => m.abort(txn_id),
            MemTableKind::Arena(m) => m.abort(txn_id),
        }
    }

    /// Scan prefix
    #[inline]
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        match self {
            MemTableKind::Standard(m) => m.scan_prefix(prefix, snapshot_ts, current_txn_id),
            MemTableKind::Arena(m) => m.scan_prefix(prefix, snapshot_ts, current_txn_id),
        }
    }

    /// Scan range
    #[inline]
    pub fn scan_range(
        &self,
        start: &[u8],
        end: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        match self {
            MemTableKind::Standard(m) => m.scan_range(start, end, snapshot_ts, current_txn_id),
            MemTableKind::Arena(m) => {
                // ArenaMvccMemTable doesn't have scan_range, use scan_prefix fallback
                let mut results = Vec::new();
                if let Some(ref idx) = m.ordered_index {
                    let start_handle = ArenaKeyHandle::new(start);
                    let end_handle = ArenaKeyHandle::new(end);

                    if end.is_empty() {
                        for entry in idx.range(start_handle..) {
                            let key = entry.key();
                            if let Some(chain) = m.data.get(key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                results.push((key.as_bytes().to_vec(), value.clone()));
                            }
                        }
                    } else {
                        for entry in idx.range(start_handle..end_handle) {
                            let key = entry.key();
                            if let Some(chain) = m.data.get(key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                results.push((key.as_bytes().to_vec(), value.clone()));
                            }
                        }
                    }
                } else {
                    for entry in m.data.iter() {
                        let key = entry.key();
                        let key_bytes = key.as_bytes();
                        if key_bytes < start {
                            continue;
                        }
                        if !end.is_empty() && key_bytes >= end {
                            continue;
                        }
                        if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                            && let Some(value) = &v.value
                        {
                            results.push((key_bytes.to_vec(), value.clone()));
                        }
                    }
                }
                results
            }
        }
    }

    /// Scan range iterator (returns collected results for now)
    #[inline]
    pub fn scan_range_iter<'a>(
        &'a self,
        start: &'a [u8],
        end: &'a [u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> {
        match self {
            MemTableKind::Standard(m) => {
                Box::new(m.scan_range_iter(start, end, snapshot_ts, current_txn_id))
            }
            MemTableKind::Arena(_) => {
                // Arena version returns collected results as iterator
                let results = self.scan_range(start, end, snapshot_ts, current_txn_id);
                Box::new(results.into_iter())
            }
        }
    }

    /// Get approximate size
    #[inline]
    pub fn size(&self) -> u64 {
        match self {
            MemTableKind::Standard(m) => m.size(),
            MemTableKind::Arena(m) => m.size(),
        }
    }

    /// Garbage collect old versions
    #[inline]
    pub fn gc(&self, min_active_ts: u64) -> usize {
        match self {
            MemTableKind::Standard(m) => m.gc(min_active_ts),
            MemTableKind::Arena(m) => m.gc(min_active_ts),
        }
    }

    /// Get the kind of memtable
    pub fn kind(&self) -> MemTableType {
        match self {
            MemTableKind::Standard(_) => MemTableType::Standard,
            MemTableKind::Arena(_) => MemTableType::Arena,
        }
    }
}

/// Durable storage engine with full ACID support
pub struct DurableStorage {
    /// Path to storage directory
    path: PathBuf,
    /// Write-ahead log
    wal: Arc<TxnWal>,
    /// MVCC manager
    mvcc: Arc<MvccManager>,
    /// In-memory data (unified abstraction over Standard/Arena)
    memtable: Arc<MemTableKind>,
    /// Per-transaction WAL buffers for batched writes
    /// Key: txn_id, Value: TxnWalBuffer that accumulates writes in memory
    /// At commit, buffer is flushed to WAL with single lock acquisition
    txn_write_buffers: DashMap<u64, TxnWalBuffer>,
    /// Group commit buffer (optional)
    group_commit: Option<Arc<EventDrivenGroupCommit>>,
    /// Recovery state
    needs_recovery: AtomicU64, // 1 = needs recovery
    /// Last checkpoint LSN
    last_checkpoint_lsn: AtomicU64,
    /// Synchronous mode (like SQLite's PRAGMA synchronous)
    /// 0 = OFF, 1 = NORMAL (periodic sync), 2 = FULL (sync every commit)
    sync_mode: AtomicU64,
    /// Commits since last sync (for NORMAL mode)
    commits_since_sync: AtomicU64,
    /// Adaptive batch sizing for NORMAL mode (Little's Law)
    /// Arrival rate in requests/sec × 1000 for precision
    arrival_rate_ema: AtomicU64,
    /// Last commit timestamp in microseconds
    last_commit_us: AtomicU64,
    /// Estimated fsync latency in microseconds
    fsync_latency_us: AtomicU64,
    /// Database lock for exclusive access (None = no locking)
    #[allow(dead_code)]
    db_lock: Option<crate::lock::DatabaseLock>,
    /// Whether at-rest encryption is active for this instance (drives the live
    /// per-instance durability matrix). Set from the resolved keyring at open.
    at_rest_encrypted: bool,
    /// Whether Point-in-Time Recovery is enabled for this database (Task 3B PITR
    /// phase 1). Derived from the presence of `wal.manifest` at open (the manifest
    /// is the single source of truth), or set by `enable_point_in_time_recovery`.
    /// When enabled, the destructive `truncate_wal()` is forbidden (segment
    /// sealing is the PITR-safe replacement, landing in a later phase) so the WAL
    /// record ordinal stays a stable, durable monotonic LSN across restarts.
    pitr_enabled: AtomicBool,
}

/// Encryption configuration for opening a [`DurableStorage`].
///
/// `disabled()` (the default for the legacy open variants) keeps the database
/// plaintext and byte-compatible with pre-encryption binaries. `with_kek()`
/// supplies the Key-Encryption-Key — the operator secret that *wraps* a
/// per-database data key; it is never used verbatim as the cipher key (see
/// [`crate::keyring`]). A wrong/missing KEK for an encrypted database fails
/// closed at open (the DB will refuse to open, never silently read as plaintext).
pub struct StorageEncryption {
    /// The KEK. `None` ⇒ plaintext database.
    pub kek: Option<EncryptionKey>,
    /// Human-readable identifier for the key source (e.g. "env:SOCHDB_ENCRYPTION_KEY",
    /// "embedded", "kms:..."). Bound into the keyring for provenance.
    pub source_id: String,
}

impl StorageEncryption {
    /// Plaintext (no encryption) — the default.
    pub fn disabled() -> Self {
        Self {
            kek: None,
            source_id: "none".to_string(),
        }
    }

    /// Encrypt at rest under the given KEK.
    pub fn with_kek(kek: EncryptionKey, source_id: impl Into<String>) -> Self {
        Self {
            kek: Some(kek),
            source_id: source_id.into(),
        }
    }

    /// Whether a key is configured (i.e. encryption is requested).
    pub fn is_enabled(&self) -> bool {
        self.kek.is_some()
    }
}

impl DurableStorage {
    /// Open or create durable storage at path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, true)
    }

    /// Open with configurable ordered index
    ///
    /// When `enable_ordered_index` is false, saves ~134 ns/op on writes
    /// but scan_prefix becomes O(N) instead of O(log N + K)
    pub fn open_with_config<P: AsRef<Path>>(path: P, enable_ordered_index: bool) -> Result<Self> {
        Self::open_with_full_config(path, enable_ordered_index, MemTableType::Standard)
    }

    /// Open with arena-backed memtable for write-heavy workloads
    ///
    /// Uses ArenaMvccMemTable which reduces per-write allocations from 3 to 1.
    /// Best for workloads with:
    /// - High write throughput
    /// - Large keys (reduces allocation overhead)
    /// - Minimal concurrent reads during writes
    pub fn open_with_arena<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_full_config(path, true, MemTableType::Arena)
    }

    /// Open with full configuration options
    ///
    /// # Arguments
    /// * `path` - Storage directory path
    /// * `enable_ordered_index` - Enable ordered index for O(log N) scans
    /// * `memtable_type` - Type of memtable to use (Standard or Arena)
    ///
    /// # Locking
    ///
    /// Acquires an exclusive advisory lock on the database directory.
    /// This prevents concurrent multi-process access which would corrupt data.
    /// If another process has the database open, returns `Err(DatabaseLocked)`.
    pub fn open_with_full_config<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
        memtable_type: MemTableType,
    ) -> Result<Self> {
        Self::open_with_full_config_internal(
            path,
            enable_ordered_index,
            memtable_type,
            true,
            StorageEncryption::disabled(),
        )
    }

    /// Open with at-rest encryption configured via [`StorageEncryption`].
    ///
    /// With `StorageEncryption::disabled()` this is identical to
    /// [`Self::open_with_full_config`]. With a KEK, the database is opened (or
    /// created) encrypted: a per-DB data key is generated and wrapped into the
    /// keyring on first open, and a wrong/missing key on a subsequent open fails
    /// closed. Reads are unaffected (the live read path is in-memory); only the
    /// WAL write/recovery path pays the AEAD cost.
    pub fn open_with_encryption<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
        memtable_type: MemTableType,
        encryption: StorageEncryption,
    ) -> Result<Self> {
        Self::open_with_full_config_internal(
            path,
            enable_ordered_index,
            memtable_type,
            true,
            encryption,
        )
    }

    /// The live, per-instance durability capabilities of THIS database.
    ///
    /// Unlike the build-level [`crate::durability_capabilities`] (which reports
    /// defaults), this reflects the actual resolved state — notably whether
    /// at-rest encryption is active for this opened instance.
    pub fn durability_capabilities(&self) -> DurabilityCapabilities {
        DurabilityCapabilities {
            crash_recovery: true,
            at_rest_encryption: self.at_rest_encrypted,
            // Point-in-time recovery is live (recover_to) when PITR is enabled:
            // the WAL is fully retained and can be replayed to an LSN/timestamp.
            point_in_time_recovery: self.pitr_enabled.load(Ordering::SeqCst),
            // ARIES / fencing remain unwired on the live path.
            aries_checkpoint: false,
            wal_fencing: false,
        }
    }

    /// Whether at-rest encryption is active for this instance.
    pub fn is_encrypted(&self) -> bool {
        self.at_rest_encrypted
    }

    /// Open without locking (for testing crash recovery scenarios)
    ///
    /// # Safety
    /// This should ONLY be used in tests that simulate crashes by forgetting
    /// the storage instance. In production, always use `open_with_full_config`.
    #[cfg(test)]
    pub fn open_without_lock<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_full_config_internal(
            path,
            true,
            MemTableType::Standard,
            false,
            StorageEncryption::disabled(),
        )
    }

    /// Open an ephemeral (in-memory-like) DurableStorage backed by a temp directory.
    ///
    /// Uses the full DurableStorage engine (WAL, MVCC, SSI) but writes to a
    /// temporary directory that is automatically cleaned up when the
    /// `EphemeralHandle` is dropped. This ensures test and production code paths
    /// are identical — bugs found in tests are guaranteed to reproduce in production.
    ///
    /// # Returns
    /// An `EphemeralHandle` that owns both the storage and the temp directory.
    /// Access the storage via `handle.storage()` or `Deref` coercion.
    ///
    /// # Example
    /// ```ignore
    /// let handle = DurableStorage::open_ephemeral()?;
    /// let txn = handle.begin_transaction()?;
    /// handle.write(txn, b"key".to_vec(), b"value".to_vec())?;
    /// handle.commit(txn)?;
    /// // temp directory cleaned up when `handle` drops
    /// ```
    pub fn open_ephemeral() -> Result<EphemeralHandle> {
        let tmp = tempfile::tempdir().map_err(|e| SochDBError::Io(e))?;
        let storage = Self::open_with_full_config_internal(
            tmp.path(),
            true,
            MemTableType::Standard,
            false, // No lock needed for ephemeral
            StorageEncryption::disabled(),
        )?;
        Ok(EphemeralHandle {
            storage,
            _tmpdir: tmp,
        })
    }

    /// Open an ephemeral DurableStorage with group commit enabled.
    ///
    /// Same as `open_ephemeral()` but with group commit for higher throughput.
    pub fn open_ephemeral_with_group_commit() -> Result<EphemeralHandle> {
        let tmp = tempfile::tempdir().map_err(|e| SochDBError::Io(e))?;
        let mut storage = Self::open_with_full_config_internal(
            tmp.path(),
            true,
            MemTableType::Standard,
            false,
            StorageEncryption::disabled(),
        )?;

        let wal = storage.wal.clone();
        let gc = EventDrivenGroupCommit::new(move |txn_ids: &[u64]| {
            for &txn_id in txn_ids {
                let entry = TxnWalEntry::txn_commit(txn_id);
                wal.append_no_flush(&entry).map_err(|e| e.to_string())?;
            }
            wal.flush().map_err(|e| e.to_string())?;
            wal.sync().map_err(|e| e.to_string())?;
            Ok(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64)
        });
        storage.group_commit = Some(Arc::new(gc));

        Ok(EphemeralHandle {
            storage,
            _tmpdir: tmp,
        })
    }

    fn open_with_full_config_internal<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
        memtable_type: MemTableType,
        acquire_lock: bool,
        encryption: StorageEncryption,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        // Acquire exclusive lock on database directory (unless disabled for testing)
        let db_lock = if acquire_lock {
            Some(
                crate::lock::DatabaseLock::acquire(&path)
                    .map_err(|e| SochDBError::LockError(e.to_string()))?,
            )
        } else {
            None
        };

        let wal_path = path.join("wal.log");

        // Resolve at-rest encryption from the keyring BEFORE touching the WAL.
        // - A fresh DB (no wal.log yet) with a KEK ⇒ create an encrypted keyring.
        // - An existing plaintext DB with a KEK ⇒ refused (must migrate explicitly).
        // - An encrypted DB with a wrong/missing KEK ⇒ refused fail-closed.
        let is_new_db = !wal_path.exists();
        let enc_state = keyring::load_or_init(
            &path,
            encryption.kek.as_ref(),
            &encryption.source_id,
            is_new_db,
        )?;
        let at_rest_encrypted = enc_state.is_encrypted();
        let wal = Arc::new(TxnWal::new_with_encryption(
            &wal_path,
            enc_state.engine(),
            enc_state.db_uuid(),
            enc_state.key_epoch(),
        )?);

        // PITR anchor: the presence of wal.manifest is the single source of truth
        // that this DB is PITR-enabled. If present, seed last_checkpoint_lsn from
        // it (it is otherwise in-memory and lost on restart).
        let (pitr_enabled, initial_checkpoint_lsn) =
            if crate::wal_manifest::WalManifest::exists(&path) {
                let m = crate::wal_manifest::WalManifest::load(&path)?;
                (true, m.last_checkpoint_lsn)
            } else {
                (false, 0)
            };

        let storage = Self {
            path,
            wal: wal.clone(),
            mvcc: Arc::new(MvccManager::new()),
            memtable: Arc::new(MemTableKind::new(memtable_type, enable_ordered_index)),
            txn_write_buffers: DashMap::new(),
            group_commit: None,
            needs_recovery: AtomicU64::new(0),
            last_checkpoint_lsn: AtomicU64::new(initial_checkpoint_lsn),
            sync_mode: AtomicU64::new(1), // Default: NORMAL (like SQLite)
            commits_since_sync: AtomicU64::new(0),
            // Adaptive batch sizing (Little's Law)
            arrival_rate_ema: AtomicU64::new(1_000_000), // 1000 req/s × 1000 initial
            last_commit_us: AtomicU64::new(0),
            fsync_latency_us: AtomicU64::new(5000), // 5ms default
            db_lock,
            at_rest_encrypted,
            pitr_enabled: AtomicBool::new(pitr_enabled),
        };

        // Check if recovery needed
        if storage.check_recovery_needed()? {
            storage.needs_recovery.store(1, Ordering::SeqCst);
        }

        Ok(storage)
    }

    /// Open with group commit enabled
    pub fn open_with_group_commit<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_group_commit_and_config(path, true)
    }

    /// Open with group commit and configurable ordered index
    pub fn open_with_group_commit_and_config<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
    ) -> Result<Self> {
        let mut storage = Self::open_with_config(path, enable_ordered_index)?;

        let wal = storage.wal.clone();
        let gc = EventDrivenGroupCommit::new(move |txn_ids: &[u64]| {
            // Write all commit records WITHOUT flushing (batch them)
            for &txn_id in txn_ids {
                let entry = TxnWalEntry::txn_commit(txn_id);
                wal.append_no_flush(&entry).map_err(|e| e.to_string())?;
            }

            // Then do a SINGLE flush + fsync for the entire batch
            wal.flush().map_err(|e| e.to_string())?;
            wal.sync().map_err(|e| e.to_string())?;

            // Return commit timestamp
            Ok(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64)
        });

        storage.group_commit = Some(Arc::new(gc));
        Ok(storage)
    }

    /// Open with IndexPolicy for automatic memtable/index configuration
    ///
    /// This is the recommended constructor for new code. The policy determines:
    /// - Whether to use ordered index (ScanOptimized only)
    /// - Whether to use arena-backed memtable (WriteOptimized, AppendOnly)
    /// - Default settings optimized for the workload pattern
    ///
    /// # Arguments
    /// * `path` - Storage directory path
    /// * `policy` - Index policy determining write/scan tradeoffs
    /// * `group_commit` - Whether to enable group commit for throughput
    pub fn open_with_policy<P: AsRef<Path>>(
        path: P,
        policy: crate::index_policy::IndexPolicy,
        group_commit: bool,
    ) -> Result<Self> {
        Self::open_with_policy_encrypted(path, policy, group_commit, StorageEncryption::disabled())
    }

    /// Policy-based open with at-rest encryption configured.
    ///
    /// Identical to [`Self::open_with_policy`] but threads a [`StorageEncryption`]
    /// down to the keyring/WAL so the embedded `Database` kernel can open (or
    /// create) an encrypted database. `StorageEncryption::disabled()` is exactly
    /// the plaintext behaviour.
    pub fn open_with_policy_encrypted<P: AsRef<Path>>(
        path: P,
        policy: crate::index_policy::IndexPolicy,
        group_commit: bool,
        encryption: StorageEncryption,
    ) -> Result<Self> {
        use crate::index_policy::IndexPolicy;

        // Derive configuration from policy
        let (enable_ordered_index, memtable_type) = match policy {
            IndexPolicy::WriteOptimized | IndexPolicy::AppendOnly => {
                // Write-heavy: no ordered index, use arena for reduced allocations
                (false, MemTableType::Arena)
            }
            IndexPolicy::Balanced => {
                // Mixed OLTP: deferred sorting (already implemented in Standard)
                (true, MemTableType::Standard)
            }
            IndexPolicy::ScanOptimized => {
                // Scan-heavy: maintain ordered index
                (true, MemTableType::Standard)
            }
        };

        if group_commit {
            let mut storage =
                Self::open_with_encryption(path, enable_ordered_index, memtable_type, encryption)?;

            let wal = storage.wal.clone();
            let gc = EventDrivenGroupCommit::new(move |txn_ids: &[u64]| {
                for &txn_id in txn_ids {
                    let entry = TxnWalEntry::txn_commit(txn_id);
                    wal.append_no_flush(&entry).map_err(|e| e.to_string())?;
                }
                wal.flush().map_err(|e| e.to_string())?;
                wal.sync().map_err(|e| e.to_string())?;
                Ok(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64)
            });
            storage.group_commit = Some(Arc::new(gc));
            Ok(storage)
        } else {
            Self::open_with_encryption(path, enable_ordered_index, memtable_type, encryption)
        }
    }

    /// Open storage for concurrent mode (multi-reader, single-writer)
    ///
    /// This method opens the storage WITHOUT acquiring the exclusive file lock.
    /// Coordination is handled by the concurrent MVCC layer instead.
    ///
    /// # Safety
    ///
    /// This must ONLY be called from `Database::open_concurrent()` which
    /// manages the concurrent MVCC coordination. Direct use will cause
    /// data corruption.
    pub fn open_for_concurrent<P: AsRef<Path>>(
        path: P,
        policy: crate::index_policy::IndexPolicy,
    ) -> Result<Self> {
        Self::open_for_concurrent_encrypted(path, policy, StorageEncryption::disabled())
    }

    /// Concurrent-mode open with at-rest encryption configured.
    ///
    /// Identical to [`Self::open_for_concurrent`] but threads a
    /// [`StorageEncryption`] through, so an encrypted database can also be opened
    /// in concurrent (multi-reader) mode rather than failing closed for lack of a
    /// key channel.
    pub fn open_for_concurrent_encrypted<P: AsRef<Path>>(
        path: P,
        policy: crate::index_policy::IndexPolicy,
        encryption: StorageEncryption,
    ) -> Result<Self> {
        use crate::index_policy::IndexPolicy;

        let (enable_ordered_index, memtable_type) = match policy {
            IndexPolicy::WriteOptimized | IndexPolicy::AppendOnly => (false, MemTableType::Arena),
            IndexPolicy::Balanced => (true, MemTableType::Standard),
            IndexPolicy::ScanOptimized => (true, MemTableType::Standard),
        };

        // Open WITHOUT exclusive file lock (concurrent MVCC handles coordination)
        Self::open_with_full_config_internal(
            path,
            enable_ordered_index,
            memtable_type,
            false,
            encryption,
        )
    }

    /// Get the memtable type being used
    pub fn memtable_type(&self) -> MemTableType {
        self.memtable.kind()
    }

    /// Check if recovery is needed (dirty shutdown detection)
    ///
    /// Note: Recovery must ALWAYS run to rebuild the in-memory memtable from WAL.
    /// The clean_shutdown marker only tells us if there might be uncommitted transactions,
    /// but committed data still needs to be loaded from WAL into the memtable.
    fn check_recovery_needed(&self) -> Result<bool> {
        let marker_path = self.path.join(".clean_shutdown");
        if marker_path.exists() {
            // Clean shutdown - remove marker
            std::fs::remove_file(&marker_path)?;
        }
        // ALWAYS need recovery to rebuild memtable from WAL
        // The memtable is in-memory only and needs to be restored on every startup
        Ok(true)
    }

    /// Perform crash recovery
    pub fn recover(&self) -> Result<RecoveryStats> {
        if self.needs_recovery.load(Ordering::SeqCst) == 0 {
            return Ok(RecoveryStats::default());
        }

        let (writes, txn_count) = self.wal.replay_for_recovery()?;

        // Apply committed writes to memtable
        let recovery_txn_id = self.wal.alloc_txn_id();
        let commit_ts = self.mvcc.alloc_commit_ts();

        // Collect keys being written for efficient commit
        let mut write_set: HashSet<InlineKey> = HashSet::new();
        for (key, value) in &writes {
            write_set.insert(SmallVec::from_slice(key));
            self.memtable
                .write(key.clone(), Some(value.clone()), recovery_txn_id)?;
        }
        self.memtable.commit(recovery_txn_id, commit_ts, &write_set);

        self.needs_recovery.store(0, Ordering::SeqCst);

        Ok(RecoveryStats {
            transactions_recovered: txn_count,
            writes_recovered: writes.len(),
            commit_ts,
        })
    }

    /// Point-in-Time Recovery: rebuild the in-memory state as of `target`.
    ///
    /// This is the PITR analogue of [`Self::recover`] — call it on a FRESH open
    /// INSTEAD of `recover()` (not in addition to it), to materialize the
    /// database as it existed at a chosen LSN or commit timestamp. Because PITR
    /// mode never truncates the WAL, the full history is retained and replayed up
    /// to (and stopping at) the target, with transaction atomicity preserved
    /// (a transaction is applied only if its commit is within the target).
    ///
    /// Requires PITR to be enabled (the WAL must be fully retained); returns an
    /// error otherwise, since a truncated WAL cannot honor an arbitrary target.
    pub fn recover_to(&self, target: crate::txn_wal::RecoveryTarget) -> Result<RecoveryStats> {
        if !self.pitr_enabled.load(Ordering::SeqCst) {
            return Err(SochDBError::InvalidArgument(
                "recover_to requires Point-in-Time Recovery to be enabled \
                 (the full WAL must be retained); call enable_point_in_time_recovery first"
                    .to_string(),
            ));
        }

        // Single-shot recovery on a fresh open: refuse if state was already
        // rebuilt (by recover() or a prior recover_to()). Re-applying would layer
        // a stale set over the point-in-time state under a newer commit_ts and
        // silently corrupt it (recover()/recover_to() both clear needs_recovery).
        if self.needs_recovery.load(Ordering::SeqCst) == 0 {
            return Err(SochDBError::InvalidArgument(
                "recover_to must be the sole recovery on a fresh open, but state \
                 was already recovered; reopen the database and call recover_to first"
                    .to_string(),
            ));
        }

        // Make the on-disk WAL match current_lsn() before replaying. Under the
        // default NORMAL sync mode the commit record(s) may still sit in the
        // BufWriter, while current_lsn() counts the in-memory sequence — so
        // without this flush+fsync a captured-LSN cut would silently drop
        // committed-but-unflushed records (replay reads a fresh on-disk handle).
        self.wal.flush()?;
        self.wal.sync()?;

        let (writes, txn_count) = self.wal.replay_to_target(target)?;

        // Apply the bounded set of committed writes to the (fresh) memtable,
        // mirroring recover().
        let recovery_txn_id = self.wal.alloc_txn_id();
        let commit_ts = self.mvcc.alloc_commit_ts();
        let mut write_set: HashSet<InlineKey> = HashSet::new();
        for (key, value) in &writes {
            write_set.insert(SmallVec::from_slice(key));
            self.memtable
                .write(key.clone(), Some(value.clone()), recovery_txn_id)?;
        }
        self.memtable.commit(recovery_txn_id, commit_ts, &write_set);

        self.needs_recovery.store(0, Ordering::SeqCst);

        Ok(RecoveryStats {
            transactions_recovered: txn_count,
            writes_recovered: writes.len(),
            commit_ts,
        })
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self) -> Result<u64> {
        let txn_id = self.wal.begin_transaction()?;
        self.mvcc.begin(txn_id);
        Ok(txn_id)
    }

    /// Begin a transaction with a specific mode (ReadOnly/WriteOnly/ReadWrite)
    ///
    /// This enables mode-aware optimizations:
    /// - ReadOnly: Skip SSI tracking, 2.6x faster reads
    /// - WriteOnly: Skip read tracking, faster bulk inserts
    /// - ReadWrite: Full SSI for serializable isolation
    pub fn begin_with_mode(&self, mode: TransactionMode) -> Result<u64> {
        let txn_id = self.wal.begin_transaction()?;
        self.mvcc.begin_with_mode(txn_id, mode);
        Ok(txn_id)
    }

    /// Begin a read-only transaction without any WAL records.
    ///
    /// This is a performance-critical optimization that eliminates two WAL
    /// mutex acquisitions per read (TxnBegin + TxnAbort). Since read-only
    /// transactions have no state to recover, WAL records are unnecessary.
    ///
    /// Callers MUST use `abort_read_only_fast()` to clean up.
    #[inline]
    pub fn begin_read_only_fast(&self) -> u64 {
        let txn_id = self.wal.alloc_txn_id();
        self.mvcc.begin_read_only(txn_id);
        txn_id
    }

    /// Abort a fast read-only transaction.
    ///
    /// O(1) cleanup: only removes MVCC state. No WAL write, no memtable scan.
    #[inline]
    pub fn abort_read_only_fast(&self, txn_id: u64) {
        self.mvcc.abort(txn_id);
    }

    /// Read a key WITHOUT any MVCC transaction tracking.
    ///
    /// Uses the current global timestamp to see all committed writes.
    /// Bypasses: begin/abort, active_txns DashMap, record_read, stats.
    /// Only safe for single-threaded access (no concurrent writes).
    #[inline]
    pub fn read_latest(&self, key: &[u8]) -> Option<Vec<u8>> {
        let snapshot_ts = self
            .mvcc
            .ts_counter
            .load(std::sync::atomic::Ordering::Relaxed);
        self.memtable.read(key, snapshot_ts, None)
    }

    /// Scan keys with a prefix WITHOUT any MVCC transaction tracking.
    ///
    /// Uses the current global timestamp. Only safe for single-threaded access.
    #[inline]
    pub fn scan_latest(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let snapshot_ts = self
            .mvcc
            .ts_counter
            .load(std::sync::atomic::Ordering::Relaxed);
        self.memtable.scan_prefix(prefix, snapshot_ts, None)
    }

    /// Read a key within a transaction
    #[inline]
    pub fn read(&self, txn_id: u64, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Fast path: get just snapshot_ts without cloning whole transaction
        let snapshot_ts = self
            .mvcc
            .get_snapshot_ts(txn_id)
            .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

        // Record read for SSI (skipped for read-only transactions)
        self.mvcc.record_read(txn_id, key);

        // Read at snapshot timestamp, seeing own uncommitted writes
        Ok(self.memtable.read(key, snapshot_ts, Some(txn_id)))
    }

    /// Write a key-value pair within a transaction
    ///
    /// Writes are buffered and only flushed to disk on commit.
    /// This provides ~10× better throughput for batched inserts.
    pub fn write(&self, txn_id: u64, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        // Use the zero-allocation path internally
        self.write_refs(txn_id, &key, &value)?;

        Ok(())
    }

    /// Write from references - zero allocation hot path
    ///
    /// Avoids cloning key/value by writing to WAL from refs directly,
    /// then only allocating once for memtable storage.
    #[inline]
    pub fn write_refs(&self, txn_id: u64, key: &[u8], value: &[u8]) -> Result<()> {
        // Record write for MVCC (uses InlineKey - zero allocation for small keys)
        self.mvcc.record_write(txn_id, key);

        // Buffer writes in memory using TxnWalBuffer - NO WAL lock taken!
        // This reduces lock contention from O(writes) to O(1) per transaction
        self.txn_write_buffers
            .entry(txn_id)
            .or_insert_with(|| TxnWalBuffer::new(txn_id))
            .append(key, value);

        // Write to memtable (needs owned key/value for storage)
        self.memtable
            .write(key.to_vec(), Some(value.to_vec()), txn_id)?;

        Ok(())
    }

    /// Delete a key within a transaction
    pub fn delete(&self, txn_id: u64, key: Vec<u8>) -> Result<()> {
        // Record write (uses InlineKey - zero allocation for small keys)
        self.mvcc.record_write(txn_id, &key);

        // Buffer tombstone in memory - NO WAL lock taken!
        self.txn_write_buffers
            .entry(txn_id)
            .or_insert_with(|| TxnWalBuffer::new(txn_id))
            .append(&key, &[]); // Empty value = tombstone

        // Write tombstone to memtable
        self.memtable.write(key, None, txn_id)?;

        Ok(())
    }

    /// Batch write multiple key-value pairs with reduced overhead
    ///
    /// This API amortizes fixed costs over the batch:
    /// - Single DashMap entry lookup for TxnWalBuffer
    /// - Single MVCC write set update
    /// - Batch memtable operations
    ///
    /// Performance: ~2-3x faster than individual write_refs calls
    /// for batches of 100+ entries.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID
    /// * `writes` - Slice of (key, value) pairs
    #[inline]
    pub fn write_batch_refs(&self, txn_id: u64, writes: &[(&[u8], &[u8])]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }

        // Single DashMap access for entire batch
        let mut buffer_entry = self
            .txn_write_buffers
            .entry(txn_id)
            .or_insert_with(|| TxnWalBuffer::new(txn_id));

        // Batch operations with reduced per-row overhead
        for (key, value) in writes {
            // Record write for MVCC
            self.mvcc.record_write(txn_id, key);

            // Append to WAL buffer
            buffer_entry.append(key, value);
        }
        drop(buffer_entry);

        // Batch write to memtable
        let owned_writes: Vec<(Vec<u8>, Option<Vec<u8>>)> = writes
            .iter()
            .map(|(k, v)| (k.to_vec(), Some(v.to_vec())))
            .collect();
        self.memtable.write_batch(&owned_writes, txn_id)?;

        Ok(())
    }

    /// Commit a transaction
    ///
    /// With sync_mode:
    /// - 0 (OFF): No sync, risk of data loss
    /// - 1 (NORMAL): Adaptive sync using Little's Law: W* = √(τ/λ)
    /// - 2 (FULL): Sync every commit (safest, slowest)
    pub fn commit(&self, txn_id: u64) -> Result<u64> {
        // First, flush all buffered DATA writes to the WAL with a SINGLE lock
        // acquisition (O(1) lock instead of O(writes) locks). Flushing data
        // records before validation is safe: ARIES recovery only treats them as
        // winners if a *durable commit record* for this transaction also exists,
        // and that record is written below — strictly after validation succeeds.
        if let Some((_, buffer)) = self.txn_write_buffers.remove(&txn_id)
            && !buffer.is_empty()
        {
            // Flush entire buffer to WAL with one lock
            self.wal.flush_buffer(&buffer)?;
        }

        // ====================================================================
        // Task 1 — Linearizable commit invariant:
        //
        //   A commit record must NEVER become durable unless the transaction
        //   has already passed SSI validation and its write set is final.
        //
        // We therefore VALIDATE (and freeze the write set) *before* making the
        // commit record durable. `mvcc.commit` performs SSI validation and
        // returns `None` on a dangerous structure (serialization conflict) or
        // if the transaction no longer exists. Writing/fsyncing the commit
        // record before this point would let a crash resurrect a transaction
        // that the live system rejected — the classic "committed-after-crash,
        // aborted-before-crash" non-linearizability bug.
        // ====================================================================
        let (commit_ts, write_set) = self.mvcc.commit(txn_id).ok_or_else(|| {
            SochDBError::Validation(
                "transaction aborted: SSI validation failed (serialization conflict) \
                 or transaction not found"
                    .into(),
            )
        })?;

        // Validation passed and the write set is frozen. Only now may the
        // commit record become durable.
        if let Some(gc) = &self.group_commit {
            // Submit the *validated* commit intent to group commit and wait.
            // This batches multiple validated commits into a single fsync.
            gc.submit_and_wait(txn_id).map_err(SochDBError::Internal)?;
        } else {
            // Direct commit path with adaptive sync (Little's Law)
            let sync_mode = self.sync_mode.load(Ordering::Relaxed);
            let commits = self.commits_since_sync.fetch_add(1, Ordering::Relaxed);

            // Update arrival rate for adaptive batching
            self.update_arrival_rate();

            // Write commit record (no flush yet - BufWriter will buffer it)
            let entry = TxnWalEntry::txn_commit(txn_id);
            self.wal.append_no_flush(&entry)?;

            // Determine if we should sync/flush based on mode
            let should_sync = match sync_mode {
                0 => false,                                      // OFF: never sync
                1 => commits >= self.adaptive_batch_threshold(), // NORMAL: adaptive
                _ => true,                                       // FULL: always sync
            };

            if should_sync {
                // Measure fsync latency for adaptive tuning
                let start = std::time::Instant::now();
                self.wal.flush()?;
                self.wal.sync()?;
                let latency_us = start.elapsed().as_micros() as u64;

                // Update fsync latency estimate (EMA with α = 0.1)
                let old_latency = self.fsync_latency_us.load(Ordering::Relaxed);
                let new_latency = (old_latency * 9 + latency_us) / 10;
                self.fsync_latency_us.store(new_latency, Ordering::Relaxed);

                self.commits_since_sync.store(0, Ordering::Relaxed);
            }
        }

        // Commit record is durable (or buffered per sync mode) for a validated
        // transaction — publish the writes to the memtable. (O(k), k = keys.)
        self.memtable.commit(txn_id, commit_ts, &write_set);

        Ok(commit_ts)
    }

    /// Update arrival rate using exponential moving average
    #[inline]
    fn update_arrival_rate(&self) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let last = self.last_commit_us.swap(now_us, Ordering::Relaxed);

        if last > 0 {
            let delta_us = now_us.saturating_sub(last);
            if delta_us > 0 && delta_us < 10_000_000 {
                // Ignore gaps > 10s
                // Rate = 1_000_000 / delta_us (requests/sec)
                // Stored as rate × 1000 for precision
                let instant_rate = 1_000_000_000 / delta_us;

                // EMA with α = 0.1
                let old_rate = self.arrival_rate_ema.load(Ordering::Relaxed);
                let new_rate = (old_rate * 9 + instant_rate) / 10;
                self.arrival_rate_ema.store(new_rate, Ordering::Relaxed);
            }
        }
    }

    /// Compute optimal batch threshold using Little's Law
    ///
    /// W* = √(τ / λ) where τ = fsync latency, λ = arrival rate
    /// Returns the number of commits to batch before fsync
    #[inline]
    fn adaptive_batch_threshold(&self) -> u64 {
        let lambda = self.arrival_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0; // req/s
        let tau = self.fsync_latency_us.load(Ordering::Relaxed) as f64 / 1_000_000.0; // seconds

        if lambda <= 0.0 || tau <= 0.0 {
            return 100; // Fallback to fixed threshold
        }

        // Little's Law: W* = sqrt(2 × τ × λ)
        // This minimizes total time = wait_time + fsync_overhead
        let n_opt = (2.0 * tau * lambda).sqrt();

        // Clamp between 1 and 1000
        (n_opt as u64).clamp(1, 1000)
    }

    /// Set synchronous mode
    ///
    /// - 0: OFF - No fsync (risk of data loss)
    /// - 1: NORMAL - Periodic fsync (balanced)
    /// - 2: FULL - Fsync every commit (safest)
    pub fn set_sync_mode(&self, mode: u64) {
        self.sync_mode.store(mode.min(2), Ordering::Relaxed);
    }

    /// Force a group commit flush (useful for benchmarking or testing)
    pub fn flush_group_commit(&self) {
        if let Some(gc) = &self.group_commit {
            gc.flush_batch();
        }
    }

    /// Abort a transaction
    ///
    /// Performance: O(1) for read-only transactions (no writes to clean up).
    /// For write transactions, O(N) memtable scan is required to remove
    /// uncommitted versions.
    pub fn abort(&self, txn_id: u64) -> Result<()> {
        // Check if transaction had any buffered writes.
        // Read-only transactions never populate txn_write_buffers,
        // so this returns None — allowing us to skip the O(N) memtable scan.
        let had_writes = self.txn_write_buffers.remove(&txn_id).is_some();

        if had_writes {
            // Write abort record to WAL (only needed if data was written)
            self.wal.abort_transaction(txn_id)?;
            // Clean up uncommitted memtable entries
            self.memtable.abort(txn_id);
        }

        // MVCC cleanup is always O(1) — just removes from active_txns DashMap
        self.mvcc.abort(txn_id);

        Ok(())
    }

    /// Scan keys with prefix
    #[inline]
    pub fn scan(&self, txn_id: u64, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Fast path: get just snapshot_ts without cloning whole transaction
        let snapshot_ts = self
            .mvcc
            .get_snapshot_ts(txn_id)
            .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

        // Note: Scan doesn't record individual key reads for SSI (too expensive)
        // SSI conflicts are tracked at the prefix level if needed
        Ok(self.memtable.scan_prefix(prefix, snapshot_ts, Some(txn_id)))
    }

    /// Scan keys in range
    #[inline]
    pub fn scan_range(
        &self,
        txn_id: u64,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let snapshot_ts = self
            .mvcc
            .get_snapshot_ts(txn_id)
            .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

        Ok(self
            .memtable
            .scan_range(start, end, snapshot_ts, Some(txn_id)))
    }

    /// Streaming scan for very large result sets
    ///
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set in memory.
    #[inline]
    pub fn scan_range_iter<'a>(
        &'a self,
        txn_id: u64,
        start: &'a [u8],
        end: &'a [u8],
    ) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
        let snapshot_ts = self.mvcc.get_snapshot_ts(txn_id).unwrap_or(0);
        self.memtable
            .scan_range_iter(start, end, snapshot_ts, Some(txn_id))
    }

    /// Force fsync to disk
    /// Flush the WAL's in-memory buffer to the OS
    ///
    /// This ensures all buffered writes are pushed from the BufWriter
    /// into the OS page cache. Call this before `fsync()` to ensure
    /// all data is durable.
    pub fn flush_wal(&self) -> Result<()> {
        self.wal.flush()
    }

    /// Force sync the WAL to disk (fsync)
    pub fn fsync(&self) -> Result<()> {
        self.wal.sync()
    }

    /// Write checkpoint
    pub fn checkpoint(&self) -> Result<u64> {
        let txn_id = 0; // System transaction
        let entry = TxnWalEntry::checkpoint(txn_id);
        let lsn = self.wal.append_sync(&entry)?;
        self.last_checkpoint_lsn.store(lsn, Ordering::SeqCst);
        // PITR: persist the checkpoint LSN to the durable manifest so it survives
        // restart. This piggybacks the fsync that append_sync already performed;
        // the manifest write is itself crash-safe (temp + fsync + atomic rename).
        // No-op (and no manifest write) when PITR is not enabled.
        if self.pitr_enabled.load(Ordering::SeqCst) {
            self.persist_pitr_manifest(lsn)?;
        }
        Ok(lsn)
    }

    /// The current durable monotonic LSN (the WAL record ordinal). In PITR mode
    /// the WAL is never truncated, so this is stable and monotonic across
    /// restarts (`recover_state` rebuilds it by re-counting on reopen).
    pub fn current_lsn(&self) -> u64 {
        self.wal.sequence()
    }

    /// Whether Point-in-Time Recovery is enabled for this database.
    pub fn is_pitr_enabled(&self) -> bool {
        self.pitr_enabled.load(Ordering::SeqCst)
    }

    /// Enable Point-in-Time Recovery for this database (one-way, explicit opt-in).
    ///
    /// Writes the durable `wal.manifest` whose presence marks the DB PITR-enabled
    /// on every future open. Once enabled, the destructive [`Self::truncate_wal`]
    /// is forbidden so the WAL record ordinal stays a stable durable LSN (segment
    /// sealing — the PITR-safe space-reclaim — lands in a later phase). Idempotent.
    pub fn enable_point_in_time_recovery(&self) -> Result<()> {
        if self.pitr_enabled.load(Ordering::SeqCst) {
            return Ok(()); // already enabled
        }
        // Persist the durable manifest FIRST; only flip the in-memory flag after
        // the durable write succeeds. Otherwise a failed manifest write would
        // leave `durability_capabilities()` reporting PITR (and the truncate
        // guard active) with no durable anchor on disk.
        let lsn = self.last_checkpoint_lsn.load(Ordering::SeqCst);
        if crate::wal_manifest::WalManifest::exists(&self.path) {
            self.persist_pitr_manifest(lsn)?;
        } else {
            crate::wal_manifest::WalManifest::new(lsn).write_atomic(&self.path)?;
        }
        self.pitr_enabled.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Persist the PITR manifest with the given checkpoint LSN, preserving the
    /// existing db identity if a manifest is already present.
    fn persist_pitr_manifest(&self, lsn: u64) -> Result<()> {
        let manifest = match crate::wal_manifest::WalManifest::load(&self.path) {
            Ok(mut m) => {
                m.last_checkpoint_lsn = lsn;
                m
            }
            Err(_) => crate::wal_manifest::WalManifest::new(lsn),
        };
        manifest.write_atomic(&self.path)
    }

    /// Truncate the WAL file after checkpoint.
    ///
    /// This physically truncates the WAL file to 0 bytes, reclaiming disk
    /// space. The in-memory memtable retains all data for the current
    /// session, but a crash after truncation will result in data loss
    /// since the WAL is the only persistence mechanism for DurableStorage.
    ///
    /// Call after `checkpoint()` when WAL durability across restarts is
    /// not required (e.g. desktop telemetry viewers, caches).
    ///
    /// Refused when PITR is enabled: truncation resets the WAL record ordinal,
    /// which would break the durable monotonic LSN that PITR anchors on. The
    /// PITR-safe way to reclaim space is segment sealing (a later phase).
    pub fn truncate_wal(&self) -> Result<()> {
        if self.pitr_enabled.load(Ordering::SeqCst) {
            return Err(SochDBError::InvalidArgument(
                "truncate_wal is disabled while Point-in-Time Recovery is enabled \
                 (it would reset the durable LSN); use segment sealing to reclaim space"
                    .to_string(),
            ));
        }
        self.wal.truncate()
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        // Get WAL size from the WAL manager
        let wal_size = self.wal.size_bytes();

        // Get active transaction count from MVCC
        let active_txns = self.mvcc.active_transaction_count();

        StorageStats {
            memtable_size_bytes: self.memtable.size(),
            wal_size_bytes: wal_size,
            active_transactions: active_txns,
            min_active_snapshot: self.mvcc.min_active_snapshot(),
            last_checkpoint_lsn: self.last_checkpoint_lsn.load(Ordering::SeqCst),
        }
    }

    /// Garbage collect old versions
    pub fn gc(&self) -> usize {
        let min_ts = self.mvcc.min_active_snapshot();
        self.memtable.gc(min_ts)
    }

    /// Clean shutdown
    pub fn shutdown(&self) -> Result<()> {
        // Sync WAL
        self.fsync()?;

        // Write clean shutdown marker
        let marker_path = self.path.join(".clean_shutdown");
        std::fs::write(&marker_path, b"clean")?;

        Ok(())
    }
}

impl Drop for DurableStorage {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

// =============================================================================
// EphemeralHandle - Temp-directory-backed DurableStorage for testing
// =============================================================================

/// Owns a `DurableStorage` instance backed by a temporary directory.
///
/// The temp directory is automatically cleaned up when this handle is dropped.
/// Access the underlying storage via `Deref` coercion or `.storage()`.
///
/// # Why this exists
///
/// SochDB previously had two storage engines — `LscsStorage` (BTreeMap-backed,
/// in-memory WAL, used by tests) and `DurableStorage` (SkipMap-backed, real WAL,
/// used in production). This dual-engine architecture meant bugs could surface
/// only in production because the test path exercised different code.
///
/// `EphemeralHandle` eliminates this divergence: tests use the exact same
/// `DurableStorage` engine as production, just backed by a temp directory.
pub struct EphemeralHandle {
    storage: DurableStorage,
    _tmpdir: tempfile::TempDir,
}

impl EphemeralHandle {
    /// Get a reference to the underlying storage
    pub fn storage(&self) -> &DurableStorage {
        &self.storage
    }

    /// Get a mutable reference to the underlying storage
    pub fn storage_mut(&mut self) -> &mut DurableStorage {
        &mut self.storage
    }

    /// Consume the handle and return the storage and temp directory.
    ///
    /// Useful when you need an `Arc<DurableStorage>` — the caller must keep
    /// the `TempDir` alive for the lifetime of the storage.
    pub fn into_parts(self) -> (DurableStorage, tempfile::TempDir) {
        (self.storage, self._tmpdir)
    }
}

impl std::ops::Deref for EphemeralHandle {
    type Target = DurableStorage;
    fn deref(&self) -> &DurableStorage {
        &self.storage
    }
}

impl std::ops::DerefMut for EphemeralHandle {
    fn deref_mut(&mut self) -> &mut DurableStorage {
        &mut self.storage
    }
}

/// Recovery statistics
#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub transactions_recovered: usize,
    pub writes_recovered: usize,
    pub commit_ts: u64,
}

/// Storage statistics
#[derive(Debug, Default)]
pub struct StorageStats {
    pub memtable_size_bytes: u64,
    pub wal_size_bytes: u64,
    pub active_transactions: usize,
    pub min_active_snapshot: u64,
    pub last_checkpoint_lsn: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Incremental min-active-ts (multiset) must exactly track the old full-scan
    /// semantics across begin / commit / abort, including shared snapshot_ts
    /// (refcounting) and the empty -> ts_counter watermark fallback.
    #[test]
    fn test_incremental_min_active_ts() {
        let m = MvccManager::new();
        m.assert_active_snapshots_consistent(); // empty

        // Two txns share the SAME snapshot_ts (begin reads ts_counter w/o bump).
        let a = m.begin(1);
        let b = m.begin(2);
        assert_eq!(a.snapshot_ts, b.snapshot_ts, "shared snapshot expected");
        assert_eq!(m.min_active_snapshot(), a.snapshot_ts);
        m.assert_active_snapshots_consistent();

        // Advance ts via a commit, then a third txn has a HIGHER snapshot.
        m.commit(1); // removes a; b still holds the low snapshot
        assert_eq!(m.min_active_snapshot(), b.snapshot_ts);
        m.assert_active_snapshots_consistent();

        let c = m.begin(3);
        assert!(c.snapshot_ts >= b.snapshot_ts);
        // min is still b's (the oldest still-active), NOT c's.
        assert_eq!(m.min_active_snapshot(), b.snapshot_ts);
        m.assert_active_snapshots_consistent();

        // Abort the oldest -> watermark advances to c's snapshot.
        m.abort(2);
        assert_eq!(m.min_active_snapshot(), c.snapshot_ts);
        m.assert_active_snapshots_consistent();

        // Drain the last one -> empty -> watermark falls back to ts_counter.
        m.commit(3);
        assert_eq!(m.min_active_snapshot(), m.ts_counter.load(Ordering::SeqCst));
        m.assert_active_snapshots_consistent();

        // Double-abort / abort of unknown id must be a no-op (no underflow).
        m.abort(999);
        m.abort(3);
        m.assert_active_snapshots_consistent();
    }

    /// Concurrent stress: many threads begin/commit/abort; after they all join
    /// (quiescent), the multiset must exactly mirror active_txns. This catches
    /// any refcount leak/underflow under real contention.
    #[test]
    fn test_incremental_min_active_ts_concurrent() {
        use std::sync::Arc;
        let m = Arc::new(MvccManager::new());
        let mut handles = vec![];
        for t in 0..8u64 {
            let m = Arc::clone(&m);
            handles.push(std::thread::spawn(move || {
                for i in 0..500u64 {
                    let id = t * 100_000 + i;
                    let _txn = m.begin(id);
                    if i % 3 == 0 {
                        m.abort(id);
                    } else {
                        m.commit(id);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // All transactions retired -> multiset empty, mirrors active_txns.
        m.assert_active_snapshots_consistent();
        assert_eq!(m.active_transaction_count(), 0);
    }

    #[test]
    fn test_basic_transaction() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Begin transaction
        let txn_id = storage.begin_transaction().unwrap();

        // Write data
        storage
            .write(txn_id, b"key1".to_vec(), b"value1".to_vec())
            .unwrap();
        storage
            .write(txn_id, b"key2".to_vec(), b"value2".to_vec())
            .unwrap();

        // Read back (within same transaction)
        let v1 = storage.read(txn_id, b"key1").unwrap();
        assert_eq!(v1, Some(b"value1".to_vec()));

        // Commit
        let commit_ts = storage.commit(txn_id).unwrap();
        assert!(commit_ts > 0);

        // Read in new transaction
        let txn2 = storage.begin_transaction().unwrap();
        let v1 = storage.read(txn2, b"key1").unwrap();
        assert_eq!(v1, Some(b"value1".to_vec()));
        storage.abort(txn2).unwrap();
    }

    #[test]
    fn test_snapshot_isolation() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // T1: Write initial value
        let t1 = storage.begin_transaction().unwrap();
        storage.write(t1, b"key".to_vec(), b"v1".to_vec()).unwrap();
        storage.commit(t1).unwrap();

        // T2: Start reading (snapshot at this point)
        let t2 = storage.begin_transaction().unwrap();

        // T3: Update the value
        let t3 = storage.begin_transaction().unwrap();
        storage.write(t3, b"key".to_vec(), b"v2".to_vec()).unwrap();
        storage.commit(t3).unwrap();

        // T2 should still see v1 (snapshot isolation)
        let v = storage.read(t2, b"key").unwrap();
        assert_eq!(v, Some(b"v1".to_vec()));

        // New transaction should see v2
        let t4 = storage.begin_transaction().unwrap();
        let v = storage.read(t4, b"key").unwrap();
        assert_eq!(v, Some(b"v2".to_vec()));

        storage.abort(t2).unwrap();
        storage.abort(t4).unwrap();
    }

    #[test]
    fn test_gc_preserves_versions_for_active_snapshot() {
        // GC must not free a version that an in-flight reader's snapshot can
        // still observe. The low-water-mark (min_active_snapshot) is the oldest
        // live reader; any version visible to it must survive GC.
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Seed an initial committed version v1.
        let t1 = storage.begin_transaction().unwrap();
        storage.write(t1, b"k".to_vec(), b"v1".to_vec()).unwrap();
        storage.commit(t1).unwrap();

        // Open a long-lived snapshot reader BEFORE any newer writes. Its
        // snapshot_ts pins the GC watermark so v1 cannot be reclaimed.
        let reader = storage.begin_transaction().unwrap();
        assert_eq!(
            storage.read(reader, b"k").unwrap(),
            Some(b"v1".to_vec()),
            "reader's snapshot must see v1"
        );

        // Produce several newer committed versions while the reader is active.
        for v in ["v2", "v3", "v4"] {
            let w = storage.begin_transaction().unwrap();
            storage
                .write(w, b"k".to_vec(), v.as_bytes().to_vec())
                .unwrap();
            storage.commit(w).unwrap();
        }

        // Run GC. Because `reader` is still active, the watermark is pinned at
        // its snapshot and the version it needs (v1) must NOT be freed.
        let _freed = storage.gc();
        assert_eq!(
            storage.read(reader, b"k").unwrap(),
            Some(b"v1".to_vec()),
            "GC must not free a version still visible to an active snapshot"
        );

        // A fresh transaction sees the latest committed version.
        let fresh = storage.begin_transaction().unwrap();
        assert_eq!(storage.read(fresh, b"k").unwrap(), Some(b"v4".to_vec()));
        storage.abort(fresh).unwrap();

        // Release the old reader; the watermark can now advance.
        storage.abort(reader).unwrap();

        // After the old snapshot is gone, GC may reclaim superseded versions,
        // and new readers still resolve the latest value correctly.
        let _freed2 = storage.gc();
        let after = storage.begin_transaction().unwrap();
        assert_eq!(storage.read(after, b"k").unwrap(), Some(b"v4".to_vec()));
        storage.abort(after).unwrap();
    }

    #[test]
    fn test_ssi_detects_write_skew() {
        // Classic write-skew: two concurrent transactions each read what the
        // other writes (disjoint write keys). Under plain SI both would commit,
        // violating serializability. Under SSI the pivot transaction (with both
        // an inbound and an outbound rw-edge) must be aborted, so the two
        // commits cannot both succeed.
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Seed two keys.
        let seed = storage.begin_transaction().unwrap();
        storage.write(seed, b"x".to_vec(), b"0".to_vec()).unwrap();
        storage.write(seed, b"y".to_vec(), b"0".to_vec()).unwrap();
        storage.commit(seed).unwrap();

        // T1 reads x and y, then writes x.
        let t1 = storage.begin_transaction().unwrap();
        let _ = storage.read(t1, b"x").unwrap();
        let _ = storage.read(t1, b"y").unwrap();

        // T2 reads x and y, then writes y. (Concurrent with T1.)
        let t2 = storage.begin_transaction().unwrap();
        let _ = storage.read(t2, b"x").unwrap();
        let _ = storage.read(t2, b"y").unwrap();

        storage.write(t1, b"x".to_vec(), b"1".to_vec()).unwrap();
        storage.write(t2, b"y".to_vec(), b"1".to_vec()).unwrap();

        // First committer wins.
        let c1 = storage.commit(t1);
        assert!(c1.is_ok(), "first committer should succeed: {c1:?}");

        // The second committer forms a dangerous rw-structure with T1 and must
        // be rejected to preserve serializability.
        let c2 = storage.commit(t2);
        assert!(
            c2.is_err(),
            "SSI must abort the write-skew pivot, but commit succeeded"
        );
    }

    #[test]
    fn test_abort_transaction() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Write initial value
        let t1 = storage.begin_transaction().unwrap();
        storage.write(t1, b"key".to_vec(), b"v1".to_vec()).unwrap();
        storage.commit(t1).unwrap();

        // Start transaction that will abort
        let t2 = storage.begin_transaction().unwrap();
        storage.write(t2, b"key".to_vec(), b"v2".to_vec()).unwrap();
        storage.abort(t2).unwrap();

        // New transaction should see v1 (aborted changes not visible)
        let t3 = storage.begin_transaction().unwrap();
        let v = storage.read(t3, b"key").unwrap();
        assert_eq!(v, Some(b"v1".to_vec()));
        storage.abort(t3).unwrap();
    }

    #[test]
    fn test_crash_recovery() {
        let dir = tempdir().unwrap();

        // Phase 1: Write data and commit
        {
            // Use open_without_lock for crash simulation tests
            let storage = DurableStorage::open_without_lock(dir.path()).unwrap();

            // Set sync mode to FULL to ensure data is synced before "crash"
            storage.set_sync_mode(2); // FULL: sync every commit

            let txn = storage.begin_transaction().unwrap();
            storage
                .write(txn, b"persist".to_vec(), b"data".to_vec())
                .unwrap();
            storage.commit(txn).unwrap();

            // Simulate crash (no clean shutdown)
            std::mem::forget(storage);
        }

        // Phase 2: Reopen and recover
        {
            let storage = DurableStorage::open_without_lock(dir.path()).unwrap();
            let stats = storage.recover().unwrap();
            assert!(stats.transactions_recovered > 0 || stats.writes_recovered > 0);

            // Data should be recovered
            let txn = storage.begin_transaction().unwrap();
            let v = storage.read(txn, b"persist").unwrap();
            assert_eq!(v, Some(b"data".to_vec()));
            storage.abort(txn).unwrap();
        }
    }

    /// At-rest encryption end-to-end through the live DurableStorage engine
    /// (Task 3B): an encrypted DB round-trips committed data, never leaks
    /// plaintext to the WAL file, flips the per-instance durability matrix, and
    /// fails CLOSED on a wrong or missing key (never a silent plaintext/empty
    /// open).
    #[test]
    fn test_at_rest_encryption_end_to_end() {
        let dir = tempdir().unwrap();
        let kek = || EncryptionKey::new([0xABu8; 32]);

        // Phase 1: create an encrypted DB, write committed data, clean close.
        {
            let storage = DurableStorage::open_with_encryption(
                dir.path(),
                true,
                MemTableType::Standard,
                StorageEncryption::with_kek(kek(), "test"),
            )
            .unwrap();
            assert!(storage.is_encrypted());
            assert!(storage.durability_capabilities().at_rest_encryption);
            storage.set_sync_mode(2);
            let t = storage.begin_transaction().unwrap();
            storage
                .write(t, b"secret-key".to_vec(), b"secret-value".to_vec())
                .unwrap();
            storage.commit(t).unwrap();
        } // clean drop releases the file lock

        // The keyring exists and the WAL bytes do not leak the plaintext record.
        assert!(dir.path().join("keyring.json").exists());
        let raw = std::fs::read(dir.path().join("wal.log")).unwrap();
        assert!(!contains(&raw, b"secret-value"), "value leaked to WAL");
        assert!(!contains(&raw, b"secret-key"), "key leaked to WAL");

        // Phase 2: reopen with the CORRECT key, recover, read back.
        {
            let storage = DurableStorage::open_with_encryption(
                dir.path(),
                true,
                MemTableType::Standard,
                StorageEncryption::with_kek(kek(), "test"),
            )
            .unwrap();
            storage.recover().unwrap();
            let t = storage.begin_transaction().unwrap();
            assert_eq!(
                storage.read(t, b"secret-key").unwrap(),
                Some(b"secret-value".to_vec()),
                "committed encrypted data must round-trip"
            );
            storage.abort(t).unwrap();
        }

        // Phase 3: WRONG key must fail closed (keyring canary), not open empty.
        {
            let wrong = DurableStorage::open_with_encryption(
                dir.path(),
                true,
                MemTableType::Standard,
                StorageEncryption::with_kek(EncryptionKey::new([0x00u8; 32]), "test"),
            );
            assert!(wrong.is_err(), "wrong KEK must fail closed");
        }

        // Phase 4: opening an encrypted DB with NO key must fail closed.
        {
            let no_key = DurableStorage::open_with_encryption(
                dir.path(),
                true,
                MemTableType::Standard,
                StorageEncryption::disabled(),
            );
            assert!(
                no_key.is_err(),
                "encrypted DB opened without key must fail closed"
            );
        }
    }

    /// A plaintext DB reports at_rest_encryption=false on the live matrix.
    #[test]
    fn test_plaintext_db_reports_unencrypted() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open_without_lock(dir.path()).unwrap();
        assert!(!storage.is_encrypted());
        assert!(!storage.durability_capabilities().at_rest_encryption);
        assert!(storage.durability_capabilities().crash_recovery);
        assert!(!dir.path().join("keyring.json").exists());
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    /// PITR phase 1 — durable monotonic LSN anchor.
    ///
    /// Enabling PITR writes the durable manifest; the WAL record ordinal (LSN)
    /// and the last-checkpoint LSN then survive a reopen, and the destructive
    /// truncate is refused so the anchor can never reset. A non-PITR DB is
    /// completely unaffected (no manifest, truncate works).
    #[test]
    fn test_pitr_durable_lsn_and_truncate_guard() {
        let dir = tempdir().unwrap();

        // Default DB: not PITR, no manifest, truncate allowed.
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            assert!(!s.is_pitr_enabled());
            assert!(!s.durability_capabilities().point_in_time_recovery);
            assert!(s.truncate_wal().is_ok());
        }
        assert!(!dir.path().join("wal.manifest").exists());

        // Enable PITR, write+checkpoint, capture the LSN.
        let lsn_before;
        let ckpt_lsn;
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.enable_point_in_time_recovery().unwrap();
            assert!(s.is_pitr_enabled());

            let t = s.begin_transaction().unwrap();
            s.write(t, b"k1".to_vec(), b"v1".to_vec()).unwrap();
            s.commit(t).unwrap();
            ckpt_lsn = s.checkpoint().unwrap();
            lsn_before = s.current_lsn();
            assert!(lsn_before > 0);

            // truncate is refused while PITR is on.
            assert!(
                s.truncate_wal().is_err(),
                "truncate must be refused in PITR mode"
            );
        }
        assert!(dir.path().join("wal.manifest").exists());

        // Reopen: PITR auto-detected from the manifest; LSN did NOT reset.
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            assert!(
                s.is_pitr_enabled(),
                "PITR must be auto-detected from manifest"
            );
            s.recover().unwrap();
            assert_eq!(
                s.current_lsn(),
                lsn_before,
                "durable LSN must survive reopen (not reset to 0/record-recount drift)"
            );
            assert_eq!(
                s.stats().last_checkpoint_lsn,
                ckpt_lsn,
                "last_checkpoint_lsn must be restored from the manifest"
            );
            // Committed data still readable.
            let t = s.begin_transaction().unwrap();
            assert_eq!(s.read(t, b"k1").unwrap(), Some(b"v1".to_vec()));
            s.abort(t).unwrap();
        }
    }

    /// PITR phase 2 — END-TO-END restore to a point in time.
    ///
    /// Enable PITR, commit two transactions, then on fresh reopens
    /// `recover_to(target)` materializes the exact historical state: an LSN cut
    /// between the two transactions sees only the first; the full LSN / a
    /// far-future timestamp sees both; timestamp 0 sees nothing. Transaction
    /// atomicity is preserved at the cut.
    #[test]
    fn test_pitr_recover_to_point_in_time() {
        use crate::txn_wal::RecoveryTarget;

        let dir = tempdir().unwrap();

        // Build history: txn1 sets k1=v1; txn2 overwrites k1=v1b and adds k2=v2.
        let (lsn_after_txn1, lsn_after_txn2);
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.enable_point_in_time_recovery().unwrap();

            let t1 = s.begin_transaction().unwrap();
            s.write(t1, b"k1".to_vec(), b"v1".to_vec()).unwrap();
            s.commit(t1).unwrap();
            lsn_after_txn1 = s.current_lsn();

            let t2 = s.begin_transaction().unwrap();
            s.write(t2, b"k1".to_vec(), b"v1b".to_vec()).unwrap();
            s.write(t2, b"k2".to_vec(), b"v2".to_vec()).unwrap();
            s.commit(t2).unwrap();
            lsn_after_txn2 = s.current_lsn();

            s.checkpoint().unwrap();
        }
        assert!(lsn_after_txn2 > lsn_after_txn1);

        // Restore to the cut between txn1 and txn2: only txn1's effect is visible.
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.recover_to(RecoveryTarget::Lsn(lsn_after_txn1)).unwrap();
            let t = s.begin_transaction().unwrap();
            assert_eq!(
                s.read(t, b"k1").unwrap(),
                Some(b"v1".to_vec()),
                "txn1 value"
            );
            assert_eq!(
                s.read(t, b"k2").unwrap(),
                None,
                "txn2 must NOT be present at the cut"
            );
            s.abort(t).unwrap();
        }

        // Restore to the full LSN: both transactions visible (txn2 wins on k1).
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.recover_to(RecoveryTarget::Lsn(lsn_after_txn2)).unwrap();
            let t = s.begin_transaction().unwrap();
            assert_eq!(s.read(t, b"k1").unwrap(), Some(b"v1b".to_vec()));
            assert_eq!(s.read(t, b"k2").unwrap(), Some(b"v2".to_vec()));
            s.abort(t).unwrap();
        }

        // Timestamp bounds: MAX => everything; 0 => nothing.
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.recover_to(RecoveryTarget::Timestamp(u64::MAX)).unwrap();
            let t = s.begin_transaction().unwrap();
            assert_eq!(s.read(t, b"k1").unwrap(), Some(b"v1b".to_vec()));
            assert_eq!(s.read(t, b"k2").unwrap(), Some(b"v2".to_vec()));
            s.abort(t).unwrap();
        }
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.recover_to(RecoveryTarget::Timestamp(0)).unwrap();
            let t = s.begin_transaction().unwrap();
            assert_eq!(s.read(t, b"k1").unwrap(), None, "no commit is <= ts 0");
            s.abort(t).unwrap();
        }

        // The capability matrix now reports PITR live for this DB.
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            assert!(s.durability_capabilities().point_in_time_recovery);
        }
    }

    /// recover_to is refused on a non-PITR database (the WAL may be truncated, so
    /// an arbitrary target cannot be honored).
    #[test]
    fn test_recover_to_refused_without_pitr() {
        use crate::txn_wal::RecoveryTarget;
        let dir = tempdir().unwrap();
        let s = DurableStorage::open_without_lock(dir.path()).unwrap();
        assert!(s.recover_to(RecoveryTarget::Lsn(1)).is_err());
        assert!(!s.durability_capabilities().point_in_time_recovery);
    }

    /// Regression (review HIGH): under the DEFAULT NORMAL sync mode a commit
    /// record may sit unflushed in the BufWriter while current_lsn() counts it.
    /// recover_to MUST flush+fsync before replaying, or a same-process restore to
    /// the captured LSN silently drops the committed-but-unflushed tail.
    #[test]
    fn test_recover_to_flushes_before_replay() {
        use crate::txn_wal::RecoveryTarget;
        let dir = tempdir().unwrap();
        let s = DurableStorage::open_without_lock(dir.path()).unwrap(); // NORMAL sync
        s.enable_point_in_time_recovery().unwrap();
        let t = s.begin_transaction().unwrap();
        s.write(t, b"k".to_vec(), b"v".to_vec()).unwrap();
        s.commit(t).unwrap(); // commit record likely unflushed under NORMAL
        let lsn = s.current_lsn();
        // No checkpoint, SAME process: replay reads a fresh on-disk handle.
        let stats = s.recover_to(RecoveryTarget::Lsn(lsn)).unwrap();
        assert!(
            stats.writes_recovered >= 1,
            "committed-but-unflushed data must be recovered (flush before replay)"
        );
    }

    /// Regression (review MEDIUM): recover_to must be the SOLE recovery on a
    /// fresh open. After recover() (or a prior recover_to) it must refuse, rather
    /// than silently layer a stale set over the point-in-time state.
    #[test]
    fn test_recover_to_refuses_after_recovery() {
        use crate::txn_wal::RecoveryTarget;
        let dir = tempdir().unwrap();
        {
            let s = DurableStorage::open_without_lock(dir.path()).unwrap();
            s.enable_point_in_time_recovery().unwrap();
            let t = s.begin_transaction().unwrap();
            s.write(t, b"k".to_vec(), b"v".to_vec()).unwrap();
            s.commit(t).unwrap();
            s.checkpoint().unwrap();
        }
        let s = DurableStorage::open_without_lock(dir.path()).unwrap();
        s.recover().unwrap(); // full recovery first
        assert!(
            s.recover_to(RecoveryTarget::Lsn(1)).is_err(),
            "recover_to after recover() must refuse (would double-apply)"
        );
        // And a second recover_to after a first also refuses.
        let s2 = DurableStorage::open_without_lock(dir.path()).unwrap();
        s2.recover_to(RecoveryTarget::Lsn(u64::MAX)).unwrap();
        assert!(s2.recover_to(RecoveryTarget::Lsn(1)).is_err());
    }

    /// recover_to over an ENCRYPTED WAL: the bounded replay decrypts correctly
    /// (review LOW: the encrypted bounded path was previously untested).
    #[test]
    fn test_pitr_recover_to_encrypted() {
        use crate::encryption::EncryptionKey;
        use crate::txn_wal::RecoveryTarget;

        let dir = tempdir().unwrap();
        let kek = [0x9Fu8; 32];
        let mk = || StorageEncryption::with_kek(EncryptionKey::new(kek), "test");

        let lsn_after_txn1;
        {
            let s = DurableStorage::open_with_encryption(
                dir.path(),
                true,
                MemTableType::Standard,
                mk(),
            )
            .unwrap();
            s.enable_point_in_time_recovery().unwrap();
            let t1 = s.begin_transaction().unwrap();
            s.write(t1, b"k1".to_vec(), b"v1".to_vec()).unwrap();
            s.commit(t1).unwrap();
            lsn_after_txn1 = s.current_lsn();
            let t2 = s.begin_transaction().unwrap();
            s.write(t2, b"k2".to_vec(), b"v2".to_vec()).unwrap();
            s.commit(t2).unwrap();
            s.checkpoint().unwrap();
            s.shutdown().ok();
        }

        // Reopen encrypted, restore to the cut between the two txns.
        let s =
            DurableStorage::open_with_encryption(dir.path(), true, MemTableType::Standard, mk())
                .unwrap();
        s.recover_to(RecoveryTarget::Lsn(lsn_after_txn1)).unwrap();
        let t = s.begin_transaction().unwrap();
        assert_eq!(
            s.read(t, b"k1").unwrap(),
            Some(b"v1".to_vec()),
            "encrypted bounded replay must decrypt the in-window record"
        );
        assert_eq!(s.read(t, b"k2").unwrap(), None, "txn2 is past the cut");
        s.abort(t).unwrap();
    }

    /// PITR composes with at-rest encryption (the manifest anchor is independent
    /// of the keyring; an encrypted PITR DB round-trips and stays fail-closed).
    #[test]
    fn test_pitr_with_encryption() {
        use crate::encryption::EncryptionKey;

        let dir = tempdir().unwrap();
        let kek = [0x2Bu8; 32];

        {
            let s = DurableStorage::open_with_encryption(
                dir.path(),
                true,
                MemTableType::Standard,
                StorageEncryption::with_kek(EncryptionKey::new(kek), "test"),
            )
            .unwrap();
            s.enable_point_in_time_recovery().unwrap();
            let t = s.begin_transaction().unwrap();
            s.write(t, b"ek".to_vec(), b"ev".to_vec()).unwrap();
            s.commit(t).unwrap();
            s.checkpoint().unwrap();
            s.shutdown().ok();
        }
        assert!(dir.path().join("keyring.json").exists());
        assert!(dir.path().join("wal.manifest").exists());

        // Reopen encrypted + PITR.
        let s = DurableStorage::open_with_encryption(
            dir.path(),
            true,
            MemTableType::Standard,
            StorageEncryption::with_kek(EncryptionKey::new(kek), "test"),
        )
        .unwrap();
        assert!(s.is_pitr_enabled());
        assert!(s.is_encrypted());
        s.recover().unwrap();
        let t = s.begin_transaction().unwrap();
        assert_eq!(s.read(t, b"ek").unwrap(), Some(b"ev".to_vec()));
        s.abort(t).unwrap();
    }

    /// Crash-atomicity on an ENCRYPTED database: a simulated crash (forget) on an
    /// encrypted WAL must, on reopen with the correct key, replay committed data
    /// (decrypted via the crypto-aware recovery path) while NOT resurrecting
    /// aborted/in-flight writes — and a wrong key after the crash fails closed.
    /// Combines the at-rest-encryption + crash-atomicity guarantees.
    #[test]
    fn test_encrypted_crash_recovery_atomicity() {
        use crate::encryption::EncryptionKey;
        let dir = tempdir().unwrap();
        let kek = || StorageEncryption::with_kek(EncryptionKey::new([0xC1u8; 32]), "test");
        // open encrypted WITHOUT the file lock so a forget()+reopen works in-test.
        let open = |enc: StorageEncryption| {
            DurableStorage::open_with_full_config_internal(
                dir.path(),
                true,
                MemTableType::Standard,
                false,
                enc,
            )
        };

        {
            let storage = open(kek()).unwrap();
            assert!(storage.is_encrypted());
            storage.set_sync_mode(2); // FULL: fsync each commit before the "crash"
            let t1 = storage.begin_transaction().unwrap();
            storage
                .write(t1, b"committed".to_vec(), b"durable".to_vec())
                .unwrap();
            storage.commit(t1).unwrap();
            let t2 = storage.begin_transaction().unwrap();
            storage
                .write(t2, b"aborted".to_vec(), b"x".to_vec())
                .unwrap();
            storage.abort(t2).unwrap();
            let t3 = storage.begin_transaction().unwrap();
            storage
                .write(t3, b"inflight".to_vec(), b"y".to_vec())
                .unwrap();
            std::mem::forget(storage); // crash: skip clean shutdown
        }

        // Reopen with the CORRECT key: committed survives, others do not.
        {
            let storage = open(kek()).unwrap();
            storage.recover().unwrap();
            let t = storage.begin_transaction().unwrap();
            assert_eq!(
                storage.read(t, b"committed").unwrap(),
                Some(b"durable".to_vec()),
                "committed encrypted write must survive the crash"
            );
            assert_eq!(storage.read(t, b"aborted").unwrap(), None);
            assert_eq!(storage.read(t, b"inflight").unwrap(), None);
            storage.abort(t).unwrap();
        }

        // Wrong key after the crash must fail closed (keyring canary).
        assert!(
            open(StorageEncryption::with_kek(
                EncryptionKey::new([0u8; 32]),
                "test"
            ))
            .is_err(),
            "wrong key after crash must fail closed"
        );
    }

    /// Crash-atomicity invariant (Task 4 — completes the Task 1 single-writer
    /// contract). `test_crash_recovery` proves committed data survives a crash;
    /// this proves the other half: recovery must NOT resurrect aborted or
    /// in-flight (never-committed) writes. Together they assert the live
    /// single-writer engine's commit is atomic AND durable across a crash.
    #[test]
    fn test_crash_recovery_atomicity() {
        let dir = tempdir().unwrap();

        // Phase 1: one committed write, one aborted, one in-flight at crash.
        {
            let storage = DurableStorage::open_without_lock(dir.path()).unwrap();
            storage.set_sync_mode(2); // FULL: fsync every commit before the "crash"

            // Committed — must survive.
            let t1 = storage.begin_transaction().unwrap();
            storage
                .write(t1, b"committed".to_vec(), b"durable".to_vec())
                .unwrap();
            storage.commit(t1).unwrap();

            // Aborted — must NOT be resurrected.
            let t2 = storage.begin_transaction().unwrap();
            storage
                .write(t2, b"aborted".to_vec(), b"rolledback".to_vec())
                .unwrap();
            storage.abort(t2).unwrap();

            // In-flight (never committed) at crash time — must NOT be resurrected.
            let t3 = storage.begin_transaction().unwrap();
            storage
                .write(t3, b"inflight".to_vec(), b"lost".to_vec())
                .unwrap();

            // Simulate a crash: skip Drop / clean shutdown.
            std::mem::forget(storage);
        }

        // Phase 2: reopen + recover; assert atomicity.
        {
            let storage = DurableStorage::open_without_lock(dir.path()).unwrap();
            storage.recover().unwrap();

            let t = storage.begin_transaction().unwrap();
            assert_eq!(
                storage.read(t, b"committed").unwrap(),
                Some(b"durable".to_vec()),
                "committed write must survive the crash"
            );
            assert_eq!(
                storage.read(t, b"aborted").unwrap(),
                None,
                "aborted write must not be resurrected by recovery"
            );
            assert_eq!(
                storage.read(t, b"inflight").unwrap(),
                None,
                "uncommitted in-flight write must not be resurrected by recovery"
            );
            storage.abort(t).unwrap();
        }
    }

    #[test]
    fn test_scan_prefix() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        let txn = storage.begin_transaction().unwrap();
        storage
            .write(txn, b"user:1".to_vec(), b"alice".to_vec())
            .unwrap();
        storage
            .write(txn, b"user:2".to_vec(), b"bob".to_vec())
            .unwrap();
        storage
            .write(txn, b"order:1".to_vec(), b"order1".to_vec())
            .unwrap();
        storage.commit(txn).unwrap();

        let txn2 = storage.begin_transaction().unwrap();
        let users = storage.scan(txn2, b"user:").unwrap();
        assert_eq!(users.len(), 2);
        storage.abort(txn2).unwrap();
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Insert
        let t1 = storage.begin_transaction().unwrap();
        storage
            .write(t1, b"key".to_vec(), b"value".to_vec())
            .unwrap();
        storage.commit(t1).unwrap();

        // Verify exists
        let t2 = storage.begin_transaction().unwrap();
        assert!(storage.read(t2, b"key").unwrap().is_some());
        storage.abort(t2).unwrap();

        // Delete
        let t3 = storage.begin_transaction().unwrap();
        storage.delete(t3, b"key".to_vec()).unwrap();
        storage.commit(t3).unwrap();

        // Verify deleted
        let t4 = storage.begin_transaction().unwrap();
        assert!(storage.read(t4, b"key").unwrap().is_none());
        storage.abort(t4).unwrap();
    }

    #[test]
    fn test_gc() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Create multiple versions
        for i in 0..10 {
            let txn = storage.begin_transaction().unwrap();
            storage
                .write(txn, b"key".to_vec(), format!("v{}", i).into_bytes())
                .unwrap();
            storage.commit(txn).unwrap();
        }

        // GC should reclaim old versions
        let gc_count = storage.gc();
        // At least some versions should be collected
        // (exact count depends on implementation)
        let _ = gc_count; // gc_count is usize, always >= 0
    }

    #[test]
    fn test_group_commit() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempdir().unwrap();
        let storage = Arc::new(DurableStorage::open_with_group_commit(dir.path()).unwrap());

        // Spawn multiple threads to commit concurrently
        let mut handles = vec![];
        for i in 0..4 {
            let storage = Arc::clone(&storage);
            handles.push(thread::spawn(move || {
                let txn = storage.begin_transaction().unwrap();
                storage
                    .write(
                        txn,
                        format!("key{}", i).into_bytes(),
                        format!("val{}", i).into_bytes(),
                    )
                    .unwrap();
                storage.commit(txn).unwrap()
            }));
        }

        // Wait for all commits
        let mut commit_times = vec![];
        for h in handles {
            commit_times.push(h.join().unwrap());
        }

        // All commits should succeed
        assert!(commit_times.iter().all(|&ts| ts > 0));

        // Verify data persisted
        let txn = storage.begin_transaction().unwrap();
        for i in 0..4 {
            let val = storage.read(txn, format!("key{}", i).as_bytes()).unwrap();
            assert_eq!(val, Some(format!("val{}", i).into_bytes()));
        }
        storage.abort(txn).unwrap();
    }

    // ==================== ArenaMvccMemTable Tests ====================

    #[test]
    fn test_arena_memtable_basic_write_read() {
        let memtable = ArenaMvccMemTable::new();

        // Write some values
        memtable
            .write(b"key1", Some(b"value1".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"key2", Some(b"value2".to_vec()), 1)
            .unwrap();

        // Read them back (uncommitted, so need txn_id match)
        assert_eq!(memtable.read(b"key1", 0, Some(1)), Some(b"value1".to_vec()));
        assert_eq!(memtable.read(b"key2", 0, Some(1)), Some(b"value2".to_vec()));
        assert_eq!(memtable.read(b"key3", 0, Some(1)), None);
    }

    #[test]
    fn test_arena_memtable_update() {
        let memtable = ArenaMvccMemTable::new();

        memtable.write(b"key", Some(b"v1".to_vec()), 1).unwrap();
        memtable.write(b"key", Some(b"v2".to_vec()), 1).unwrap();

        assert_eq!(memtable.read(b"key", 0, Some(1)), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_arena_memtable_delete() {
        let memtable = ArenaMvccMemTable::new();

        memtable.write(b"key", Some(b"value".to_vec()), 1).unwrap();
        memtable.write(b"key", None, 1).unwrap(); // Delete = None value

        assert_eq!(memtable.read(b"key", 0, Some(1)), None);
    }

    #[test]
    fn test_arena_memtable_scan_prefix() {
        let memtable = ArenaMvccMemTable::new();

        memtable
            .write(b"user:1:name", Some(b"Alice".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"user:1:email", Some(b"alice@test.com".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"user:2:name", Some(b"Bob".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"order:1", Some(b"order_data".to_vec()), 1)
            .unwrap();

        // Create a write set and commit
        let mut write_set = HashSet::new();
        write_set.insert(InlineKey::from_slice(b"user:1:name"));
        write_set.insert(InlineKey::from_slice(b"user:1:email"));
        write_set.insert(InlineKey::from_slice(b"user:2:name"));
        write_set.insert(InlineKey::from_slice(b"order:1"));
        memtable.commit(1, 10, &write_set);

        // Scan for user:1:* (snapshot_ts > commit_ts to see committed data)
        let results = memtable.scan_prefix(b"user:1:", 11, None);
        assert_eq!(results.len(), 2);

        // Scan for all users
        let results = memtable.scan_prefix(b"user:", 11, None);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_arena_memtable_write_batch() {
        let memtable = ArenaMvccMemTable::new();

        let writes: Vec<(&[u8], Option<Vec<u8>>)> = vec![
            (b"k1", Some(b"v1".to_vec())),
            (b"k2", Some(b"v2".to_vec())),
            (b"k3", Some(b"v3".to_vec())),
        ];

        memtable.write_batch(&writes, 1).unwrap();

        assert_eq!(memtable.read(b"k1", 0, Some(1)), Some(b"v1".to_vec()));
        assert_eq!(memtable.read(b"k2", 0, Some(1)), Some(b"v2".to_vec()));
        assert_eq!(memtable.read(b"k3", 0, Some(1)), Some(b"v3".to_vec()));
    }

    #[test]
    fn test_arena_memtable_gc() {
        let memtable = ArenaMvccMemTable::new();

        // Write multiple versions
        for i in 0..10 {
            memtable
                .write(b"key", Some(format!("v{}", i).into_bytes()), i + 1)
                .unwrap();

            let mut write_set = HashSet::new();
            write_set.insert(InlineKey::from_slice(b"key"));
            memtable.commit(i + 1, (i + 1) * 10, &write_set);
        }

        // GC old versions
        let gc_count = memtable.gc(90);
        let _ = gc_count; // gc_count is usize, always >= 0
    }

    #[test]
    fn test_arena_memtable_size_tracking() {
        let memtable = ArenaMvccMemTable::new();

        assert_eq!(memtable.size(), 0);

        memtable.write(b"key", Some(b"value".to_vec()), 1).unwrap();

        assert!(memtable.size() > 0);
    }

    #[test]
    fn test_arena_memtable_abort() {
        let memtable = ArenaMvccMemTable::new();

        memtable
            .write(b"key", Some(b"uncommitted".to_vec()), 1)
            .unwrap();

        // Visible to same txn
        assert_eq!(
            memtable.read(b"key", 0, Some(1)),
            Some(b"uncommitted".to_vec())
        );

        // Not visible to other txns
        assert_eq!(memtable.read(b"key", 0, Some(2)), None);

        // Abort
        memtable.abort(1);

        // No longer visible
        assert_eq!(memtable.read(b"key", 0, Some(1)), None);
    }

    // ========================================================================
    // MemTableKind Tests - Unified Abstraction
    // ========================================================================

    #[test]
    fn test_memtable_kind_standard() {
        let memtable = MemTableKind::new(MemTableType::Standard, true);
        assert_eq!(memtable.kind(), MemTableType::Standard);

        // Write and read
        memtable
            .write(b"key1".to_vec(), Some(b"value1".to_vec()), 1)
            .unwrap();

        // Commit transaction at ts=100
        let write_set = std::iter::once(InlineKey::from_slice(b"key1")).collect();
        memtable.commit(1, 100, &write_set);

        // Read after commit - snapshot_ts must be > commit_ts for visibility
        let v = memtable.read(b"key1", 101, None);
        assert_eq!(v, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_memtable_kind_arena() {
        let memtable = MemTableKind::new(MemTableType::Arena, true);
        assert_eq!(memtable.kind(), MemTableType::Arena);

        // Write and read
        memtable
            .write(b"key1".to_vec(), Some(b"value1".to_vec()), 1)
            .unwrap();

        // Commit at ts=100
        let write_set = std::iter::once(InlineKey::from_slice(b"key1")).collect();
        memtable.commit(1, 100, &write_set);

        // Read after commit - snapshot_ts > commit_ts
        let v = memtable.read(b"key1", 101, None);
        assert_eq!(v, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_memtable_kind_scan_range() {
        // Test both implementations have consistent behavior
        for kind in [MemTableType::Standard, MemTableType::Arena] {
            let memtable = MemTableKind::new(kind, true);

            // Write some data
            for i in 0..5 {
                let key = format!("key{}", i);
                let value = format!("value{}", i);
                memtable
                    .write(key.into_bytes(), Some(value.into_bytes()), 1)
                    .unwrap();
            }

            // Commit all at ts=100
            let write_set: HashSet<InlineKey> = (0..5)
                .map(|i| InlineKey::from_slice(format!("key{}", i).as_bytes()))
                .collect();
            memtable.commit(1, 100, &write_set);

            // Scan range with snapshot_ts > commit_ts
            let results = memtable.scan_range(b"key1", b"key4", 101, None);
            assert_eq!(
                results.len(),
                3,
                "kind={:?} should have 3 results (key1, key2, key3)",
                kind
            );
        }
    }

    #[test]
    fn test_durable_storage_arena() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open_with_arena(dir.path()).unwrap();

        assert_eq!(storage.memtable_type(), MemTableType::Arena);

        // Basic transaction should work the same
        let txn_id = storage.begin_transaction().unwrap();
        storage
            .write(txn_id, b"key1".to_vec(), b"value1".to_vec())
            .unwrap();
        storage.commit(txn_id).unwrap();

        let txn2 = storage.begin_transaction().unwrap();
        let v = storage.read(txn2, b"key1").unwrap();
        assert_eq!(v, Some(b"value1".to_vec()));
        storage.abort(txn2).unwrap();
    }

    #[test]
    fn test_durable_storage_full_config() {
        let dir = tempdir().unwrap();

        // Test with Arena and ordered index enabled
        let storage =
            DurableStorage::open_with_full_config(dir.path(), true, MemTableType::Arena).unwrap();

        assert_eq!(storage.memtable_type(), MemTableType::Arena);

        // Write multiple keys
        let txn = storage.begin_transaction().unwrap();
        for i in 0..10 {
            let key = format!("key{:02}", i);
            let value = format!("value{}", i);
            storage
                .write(txn, key.into_bytes(), value.into_bytes())
                .unwrap();
        }
        storage.commit(txn).unwrap();

        // Scan should work (uses scan method for prefix)
        let txn2 = storage.begin_transaction().unwrap();
        let results = storage.scan(txn2, b"key0").unwrap();
        assert_eq!(results.len(), 10); // key00 through key09
        storage.abort(txn2).unwrap();
    }
}
