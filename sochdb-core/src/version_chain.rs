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

//! Unified MVCC Version Chain Interface
//!
//! This module defines the canonical interface for MVCC version chains across SochDB.
//! Multiple implementations exist for different subsystems, but they share these traits.
//!
//! ## Implementations
//!
//! | Implementation | Location | Use Case |
//! |---------------|----------|----------|
//! | `VersionChain` | `sochdb_core::epoch_gc` | Epoch-based GC with VecDeque |
//! | `VersionChain` | `sochdb_storage::mvcc_snapshot` | Snapshot-based visibility |
//! | `VersionChain` | `sochdb_storage::version_store` | Generic key-value MVCC |
//! | `VersionChain` | `sochdb_storage::durable_storage` | Binary-search optimized |
//!
//! ## Visibility Semantics
//!
//! All implementations follow these MVCC visibility rules:
//!
//! 1. **Read Committed**: A version is visible if its creating transaction has committed
//!    before the reader's start timestamp.
//!
//! 2. **Snapshot Isolation**: A version is visible if:
//!    - It was committed before the reader's snapshot timestamp
//!    - It was not deleted, or deleted after the snapshot timestamp
//!
//! 3. **Serializable (SSI)**: Adds read-write conflict detection on top of SI.

/// Transaction identifier type
pub type TxnId = u64;

/// Logical timestamp type
pub type Timestamp = u64;

/// Version visibility context
/// 
/// Provides the information needed to determine if a version is visible
/// to a particular reader/transaction.
#[derive(Debug, Clone)]
pub struct VisibilityContext {
    /// Reader's transaction ID
    pub reader_txn_id: TxnId,
    /// Reader's snapshot timestamp
    pub snapshot_ts: Timestamp,
    /// Set of transaction IDs that are still active (not committed)
    pub active_txn_ids: std::collections::HashSet<TxnId>,
}

impl VisibilityContext {
    /// Create a new visibility context
    pub fn new(reader_txn_id: TxnId, snapshot_ts: Timestamp) -> Self {
        Self {
            reader_txn_id,
            snapshot_ts,
            active_txn_ids: std::collections::HashSet::new(),
        }
    }

    /// Create with active transaction set
    pub fn with_active_txns(
        reader_txn_id: TxnId,
        snapshot_ts: Timestamp,
        active_txn_ids: std::collections::HashSet<TxnId>,
    ) -> Self {
        Self {
            reader_txn_id,
            snapshot_ts,
            active_txn_ids,
        }
    }

    /// Check if a transaction was committed before this snapshot
    pub fn is_committed_before(&self, txn_id: TxnId, commit_ts: Option<Timestamp>) -> bool {
        match commit_ts {
            Some(ts) => ts < self.snapshot_ts && !self.active_txn_ids.contains(&txn_id),
            None => false,
        }
    }
}

/// Version metadata
/// 
/// Common metadata for all version chain implementations.
#[derive(Debug, Clone)]
pub struct VersionMeta {
    /// Transaction that created this version
    pub created_by: TxnId,
    /// Timestamp when this version was created
    pub created_ts: Timestamp,
    /// Transaction that deleted this version (0 = not deleted)
    pub deleted_by: TxnId,
    /// Timestamp when this version was deleted (0 = not deleted)
    pub deleted_ts: Timestamp,
    /// Commit timestamp (0 = not yet committed)
    pub commit_ts: Timestamp,
}

impl VersionMeta {
    /// Create metadata for a new uncommitted version
    pub fn new_uncommitted(created_by: TxnId, created_ts: Timestamp) -> Self {
        Self {
            created_by,
            created_ts,
            deleted_by: 0,
            deleted_ts: 0,
            commit_ts: 0,
        }
    }

    /// Check if this version is visible according to the context
    pub fn is_visible(&self, ctx: &VisibilityContext) -> bool {
        // Must be committed before snapshot
        if self.commit_ts == 0 {
            // Uncommitted - only visible to creating transaction
            return self.created_by == ctx.reader_txn_id;
        }

        if self.commit_ts >= ctx.snapshot_ts {
            return false;
        }

        // Must not be deleted, or deleted after snapshot
        if self.deleted_by != 0 && self.deleted_ts < ctx.snapshot_ts {
            return false;
        }

        true
    }

    /// Mark as committed
    pub fn commit(&mut self, commit_ts: Timestamp) {
        self.commit_ts = commit_ts;
    }

    /// Mark as deleted
    pub fn delete(&mut self, deleted_by: TxnId, deleted_ts: Timestamp) {
        self.deleted_by = deleted_by;
        self.deleted_ts = deleted_ts;
    }

    /// Check if version is committed
    pub fn is_committed(&self) -> bool {
        self.commit_ts != 0
    }

    /// Check if version is deleted
    pub fn is_deleted(&self) -> bool {
        self.deleted_by != 0
    }
}

/// Trait for MVCC version chain implementations
/// 
/// Implementors store multiple versions of a value and provide
/// visibility-based access according to MVCC semantics.
pub trait MvccVersionChain {
    /// The value type stored in versions
    type Value;

    /// Get the visible version for the given context
    fn get_visible(&self, ctx: &VisibilityContext) -> Option<&Self::Value>;

    /// Get the latest version (regardless of visibility)
    fn get_latest(&self) -> Option<&Self::Value>;

    /// Number of versions in the chain
    fn version_count(&self) -> usize;

    /// Check if the chain is empty
    fn is_empty(&self) -> bool {
        self.version_count() == 0
    }
}

/// Trait for mutable version chain operations
pub trait MvccVersionChainMut: MvccVersionChain {
    /// Add a new uncommitted version
    fn add_uncommitted(&mut self, value: Self::Value, txn_id: TxnId);

    /// Commit a version
    fn commit_version(&mut self, txn_id: TxnId, commit_ts: Timestamp) -> bool;

    /// Mark the latest visible version as deleted
    fn delete_version(&mut self, txn_id: TxnId, delete_ts: Timestamp) -> bool;

    /// Garbage collect versions older than the given timestamp
    /// Returns (versions_removed, bytes_freed)
    fn gc(&mut self, min_visible_ts: Timestamp) -> (usize, usize);
}

/// Trait for detecting write conflicts
pub trait WriteConflictDetection {
    /// Check if there's a write-write conflict with another transaction
    fn has_write_conflict(&self, txn_id: TxnId) -> bool;
}

// =============================================================================
// Rec 6: Compile-Time Concurrency Policy Markers
// =============================================================================

/// Marker trait for version chain concurrency strategy.
///
/// Implementors tag their version chain with the concurrency mechanism used,
/// enabling generic code to select appropriate strategies at compile time.
pub trait ConcurrencyPolicy: Send + Sync {
    /// Human-readable name for diagnostics
    const NAME: &'static str;
}

/// No internal synchronization — caller must hold external lock (DashMap shard, RwLock, etc.)
pub struct ExternalLock;
impl ConcurrencyPolicy for ExternalLock {
    const NAME: &'static str = "ExternalLock";
}

/// Internal RwLock — chain can be shared across threads with &self methods
pub struct InternalRwLock;
impl ConcurrencyPolicy for InternalRwLock {
    const NAME: &'static str = "InternalRwLock";
}

/// Lock-free atomics — CAS-based, fully concurrent &self methods
pub struct LockFreeAtomic;
impl ConcurrencyPolicy for LockFreeAtomic {
    const NAME: &'static str = "LockFreeAtomic";
}

// Safety: marker types are stateless
unsafe impl Send for ExternalLock {}
unsafe impl Sync for ExternalLock {}
unsafe impl Send for InternalRwLock {}
unsafe impl Sync for InternalRwLock {}
unsafe impl Send for LockFreeAtomic {}
unsafe impl Sync for LockFreeAtomic {}

// =============================================================================
// Rec 11: Consolidated Binary-Search Version Chain
// =============================================================================

/// Trait for version entry types used in binary-search chains.
///
/// Implemented by `durable_storage::Version` and `mvcc_concurrent::VersionEntry`
/// to allow a single `BinarySearchChain<E>` to handle both.
pub trait ChainEntry: Sized + std::fmt::Debug {
    /// Get the commit timestamp (0 = uncommitted)
    fn commit_ts(&self) -> u64;
    /// Get the transaction ID that created this version
    fn txn_id(&self) -> u64;
    /// Set the commit timestamp (called during commit)
    fn set_commit_ts(&mut self, ts: u64);
}

/// A binary-search optimized version chain — the consolidated core.
///
/// Stores committed versions sorted descending by `commit_ts`, with at most
/// one uncommitted version slot. Uses `partition_point` for O(log V) lookups.
///
/// This struct captures the duplicated logic previously in:
/// - `durable_storage::VersionChain`
/// - `mvcc_concurrent::VersionChain`
///
/// Both modules now wrap `BinarySearchChain<E>` and delegate the core
/// binary-search / commit / abort / read / conflict-check operations to it.
#[derive(Debug)]
pub struct BinarySearchChain<E: ChainEntry> {
    /// Committed versions sorted by commit_ts DESCENDING (newest first)
    committed: Vec<E>,
    /// Single uncommitted version slot (at most one per transaction writing this key)
    uncommitted: Option<E>,
}

impl<E: ChainEntry> Default for BinarySearchChain<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: ChainEntry> BinarySearchChain<E> {
    /// Create a new empty chain.
    #[inline]
    pub fn new() -> Self {
        Self {
            committed: Vec::new(),
            uncommitted: None,
        }
    }

    // ---- Uncommitted slot management ----

    /// Replace the uncommitted slot. Returns the previous entry if any.
    #[inline]
    pub fn set_uncommitted(&mut self, entry: E) -> Option<E> {
        self.uncommitted.replace(entry)
    }

    /// Reference to the uncommitted entry.
    #[inline]
    pub fn uncommitted(&self) -> Option<&E> {
        self.uncommitted.as_ref()
    }

    /// Mutable reference to the uncommitted entry.
    #[inline]
    pub fn uncommitted_mut(&mut self) -> Option<&mut E> {
        self.uncommitted.as_mut()
    }

    // ---- Core operations ----

    /// Commit the uncommitted version with the given `commit_ts`.
    ///
    /// Moves the entry from the uncommitted slot into the sorted committed
    /// list at the correct position. Returns `true` on success.
    ///
    /// Complexity: O(log V) binary search + O(V) insert (amortised O(1) for newest).
    pub fn commit(&mut self, txn_id: u64, commit_ts: u64) -> bool {
        if let Some(ref mut v) = self.uncommitted {
            if v.txn_id() == txn_id && v.commit_ts() == 0 {
                v.set_commit_ts(commit_ts);
                let committed_version = self.uncommitted.take().unwrap();
                let insert_pos = self
                    .committed
                    .partition_point(|e| e.commit_ts() > commit_ts);
                self.committed.insert(insert_pos, committed_version);
                return true;
            }
        }
        false
    }

    /// Abort a transaction — clear the uncommitted slot if it matches `txn_id`.
    #[inline]
    pub fn abort(&mut self, txn_id: u64) {
        if let Some(ref v) = self.uncommitted {
            if v.txn_id() == txn_id {
                self.uncommitted = None;
            }
        }
    }

    /// Read the visible version at the given snapshot timestamp.
    ///
    /// If `current_txn_id` is provided and matches the uncommitted version's
    /// transaction, the uncommitted version is returned (read-own-writes).
    ///
    /// Otherwise performs O(log V) binary search for the newest committed
    /// version with `commit_ts < snapshot_ts`.
    #[inline]
    pub fn read_at(&self, snapshot_ts: u64, current_txn_id: Option<u64>) -> Option<&E> {
        if let Some(txn_id) = current_txn_id {
            if let Some(ref v) = self.uncommitted {
                if v.txn_id() == txn_id {
                    return Some(v);
                }
            }
        }
        let idx = self
            .committed
            .partition_point(|v| v.commit_ts() >= snapshot_ts);
        self.committed.get(idx)
    }

    /// Check if there's a write-write conflict with another transaction.
    #[inline]
    pub fn has_write_conflict(&self, my_txn_id: u64) -> bool {
        if let Some(ref v) = self.uncommitted {
            return v.txn_id() != my_txn_id;
        }
        false
    }

    /// GC versions older than `min_active_ts`.
    ///
    /// Keeps all versions with `commit_ts > min_active_ts` plus one anchor
    /// version at or below the threshold (for new snapshots).
    pub fn gc_by_ts(&mut self, min_active_ts: u64) {
        if self.committed.len() <= 1 {
            return;
        }
        let split_idx = self
            .committed
            .partition_point(|v| v.commit_ts() > min_active_ts);
        let keep_count = if split_idx < self.committed.len() {
            split_idx + 1
        } else {
            split_idx
        };
        self.committed.truncate(keep_count);
    }

    // ---- Accessors ----

    /// Total version count (committed + uncommitted).
    #[inline]
    pub fn version_count(&self) -> usize {
        self.committed.len() + usize::from(self.uncommitted.is_some())
    }

    /// Number of committed versions.
    #[inline]
    pub fn committed_count(&self) -> usize {
        self.committed.len()
    }

    /// Check if chain is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty() && self.uncommitted.is_none()
    }

    /// Slice of committed versions (newest first).
    #[inline]
    pub fn committed_versions(&self) -> &[E] {
        &self.committed
    }

    /// Mutable access to committed versions (for custom GC).
    #[inline]
    pub fn committed_versions_mut(&mut self) -> &mut Vec<E> {
        &mut self.committed
    }

    /// Latest version: uncommitted if present, else newest committed.
    #[inline]
    pub fn latest(&self) -> Option<&E> {
        self.uncommitted.as_ref().or_else(|| self.committed.first())
    }

    /// Newest committed version only.
    #[inline]
    pub fn latest_committed(&self) -> Option<&E> {
        self.committed.first()
    }
}

// =============================================================================
// Rec 11: Unified MVCC Store Trait
// =============================================================================

/// Error type for MVCC store operations.
#[derive(Debug)]
pub enum MvccStoreError {
    /// Another uncommitted write exists on this key
    WriteConflict,
}

impl std::fmt::Display for MvccStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WriteConflict => write!(f, "write-write conflict"),
        }
    }
}

impl std::error::Error for MvccStoreError {}

/// GC statistics returned by `MvccStore::mvcc_gc`.
#[derive(Debug, Default, Clone)]
pub struct MvccGcStats {
    pub versions_removed: usize,
    pub keys_scanned: usize,
}

/// Unified MVCC key-value store trait.
///
/// Provides a common interface for:
/// - `durable_storage::MvccMemTable`
/// - `mvcc_concurrent::VersionStore`
/// - `epoch_mvcc::EpochMvccStore`
///
/// Callers can program against this trait to be agnostic to the
/// underlying concurrency / storage strategy.
pub trait MvccStore: Send + Sync {
    /// Read the visible value at `snapshot_ts`, optionally seeing own writes
    /// from `txn_id`.
    fn mvcc_get(&self, key: &[u8], snapshot_ts: u64, txn_id: Option<u64>) -> Option<Vec<u8>>;

    /// Write a value (or tombstone `None`) as uncommitted for the given transaction.
    fn mvcc_put(
        &self,
        key: &[u8],
        value: Option<Vec<u8>>,
        txn_id: u64,
    ) -> Result<(), MvccStoreError>;

    /// Commit one key's uncommitted write. Returns `true` if found and committed.
    fn mvcc_commit_key(&self, key: &[u8], txn_id: u64, commit_ts: u64) -> bool;

    /// Abort one key's uncommitted write.
    fn mvcc_abort_key(&self, key: &[u8], txn_id: u64);

    /// Check if there's an uncommitted write conflict on a key.
    fn mvcc_has_conflict(&self, key: &[u8], txn_id: u64) -> bool;

    /// Run garbage collection. Returns statistics.
    fn mvcc_gc(&self, min_ts: u64) -> MvccGcStats;

    /// Number of distinct keys in the store.
    fn mvcc_key_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_meta_visibility() {
        let mut meta = VersionMeta::new_uncommitted(1, 100);
        
        // Uncommitted - only visible to creator
        let ctx = VisibilityContext::new(1, 200);
        assert!(meta.is_visible(&ctx));
        
        let ctx2 = VisibilityContext::new(2, 200);
        assert!(!meta.is_visible(&ctx2));
        
        // After commit - visible to later snapshots
        meta.commit(150);
        assert!(meta.is_visible(&ctx2));
        
        // Not visible to earlier snapshots
        let ctx3 = VisibilityContext::new(3, 100);
        assert!(!meta.is_visible(&ctx3));
    }

    #[test]
    fn test_version_meta_deletion() {
        let mut meta = VersionMeta::new_uncommitted(1, 100);
        meta.commit(150);
        meta.delete(2, 200);
        
        // Visible before deletion
        let ctx = VisibilityContext::new(3, 180);
        assert!(meta.is_visible(&ctx));
        
        // Not visible after deletion
        let ctx2 = VisibilityContext::new(3, 250);
        assert!(!meta.is_visible(&ctx2));
    }

    #[test]
    fn test_visibility_context_committed_before() {
        let mut active = std::collections::HashSet::new();
        active.insert(5);
        
        let ctx = VisibilityContext::with_active_txns(1, 200, active);
        
        // Committed before snapshot
        assert!(ctx.is_committed_before(2, Some(100)));
        
        // Committed after snapshot
        assert!(!ctx.is_committed_before(3, Some(250)));
        
        // Active transaction - not committed
        assert!(!ctx.is_committed_before(5, Some(100)));
        
        // No commit timestamp
        assert!(!ctx.is_committed_before(6, None));
    }

    #[test]
    fn test_concurrency_policy_names() {
        assert_eq!(ExternalLock::NAME, "ExternalLock");
        assert_eq!(InternalRwLock::NAME, "InternalRwLock");
        assert_eq!(LockFreeAtomic::NAME, "LockFreeAtomic");
    }

    // ---- BinarySearchChain tests (Rec 11) ----

    /// Minimal entry type for testing the consolidated chain.
    #[derive(Debug, Clone)]
    struct TestEntry {
        commit_ts: u64,
        txn_id: u64,
        val: i32,
    }

    impl ChainEntry for TestEntry {
        fn commit_ts(&self) -> u64 { self.commit_ts }
        fn txn_id(&self) -> u64 { self.txn_id }
        fn set_commit_ts(&mut self, ts: u64) { self.commit_ts = ts; }
    }

    #[test]
    fn test_binary_search_chain_commit_and_read() {
        let mut chain = BinarySearchChain::<TestEntry>::new();
        assert!(chain.is_empty());

        // Add uncommitted, then commit
        chain.set_uncommitted(TestEntry { commit_ts: 0, txn_id: 1, val: 10 });
        assert_eq!(chain.version_count(), 1);

        // Read own writes
        let v = chain.read_at(100, Some(1)).unwrap();
        assert_eq!(v.val, 10);

        // Other txn can't see it
        assert!(chain.read_at(100, Some(2)).is_none());

        // Commit
        assert!(chain.commit(1, 50));
        assert_eq!(chain.committed_count(), 1);

        // Now visible to snapshot_ts > 50
        let v = chain.read_at(51, None).unwrap();
        assert_eq!(v.val, 10);

        // Not visible to snapshot_ts <= 50
        assert!(chain.read_at(50, None).is_none());
    }

    #[test]
    fn test_binary_search_chain_abort() {
        let mut chain = BinarySearchChain::<TestEntry>::new();
        chain.set_uncommitted(TestEntry { commit_ts: 0, txn_id: 1, val: 10 });
        chain.abort(1);
        assert!(chain.is_empty());
        // Abort wrong txn is a no-op
        chain.set_uncommitted(TestEntry { commit_ts: 0, txn_id: 2, val: 20 });
        chain.abort(1);
        assert_eq!(chain.version_count(), 1);
    }

    #[test]
    fn test_binary_search_chain_write_conflict() {
        let mut chain = BinarySearchChain::<TestEntry>::new();
        assert!(!chain.has_write_conflict(1));

        chain.set_uncommitted(TestEntry { commit_ts: 0, txn_id: 1, val: 10 });
        assert!(!chain.has_write_conflict(1)); // own txn
        assert!(chain.has_write_conflict(2));  // other txn
    }

    #[test]
    fn test_binary_search_chain_gc() {
        let mut chain = BinarySearchChain::<TestEntry>::new();

        // Commit 5 versions at ts 10, 20, 30, 40, 50
        for i in 1..=5u64 {
            chain.set_uncommitted(TestEntry { commit_ts: 0, txn_id: i, val: i as i32 });
            chain.commit(i, i * 10);
        }
        assert_eq!(chain.committed_count(), 5);

        // GC with min_active_ts = 25 → keep ts > 25 (30, 40, 50) + 1 anchor (20)
        chain.gc_by_ts(25);
        assert_eq!(chain.committed_count(), 4); // 50, 40, 30, 20

        // GC with min_active_ts = 45 → keep ts > 45 (50) + 1 anchor (40)
        chain.gc_by_ts(45);
        assert_eq!(chain.committed_count(), 2); // 50, 40
    }

    #[test]
    fn test_binary_search_chain_multiple_versions() {
        let mut chain = BinarySearchChain::<TestEntry>::new();

        // Commit in order: ts=100, ts=200, ts=300
        for (i, ts) in [100u64, 200, 300].iter().enumerate() {
            let txn = (i + 1) as u64;
            chain.set_uncommitted(TestEntry { commit_ts: 0, txn_id: txn, val: *ts as i32 });
            chain.commit(txn, *ts);
        }

        // Snapshot at 150 → sees ts=100
        assert_eq!(chain.read_at(150, None).unwrap().val, 100);
        // Snapshot at 250 → sees ts=200
        assert_eq!(chain.read_at(250, None).unwrap().val, 200);
        // Snapshot at 350 → sees ts=300
        assert_eq!(chain.read_at(350, None).unwrap().val, 300);
        // Snapshot at 50 → sees nothing
        assert!(chain.read_at(50, None).is_none());
    }
}
