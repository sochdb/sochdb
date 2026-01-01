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

//! Unified Transaction Coordinator
//!
//! This module defines the canonical transaction management interface for ToonDB.
//! It consolidates the functionality of multiple transaction managers:
//!
//! ## Implementation Guide
//!
//! There are THREE transaction manager implementations in ToonDB. Here's when to use each:
//!
//! ### 1. `MvccTransactionManager` (wal_integration.rs) - **RECOMMENDED FOR PRODUCTION**
//!
//! Full-featured transaction manager with:
//! - ✅ WAL-based durability (fsync on commit)
//! - ✅ MVCC snapshot isolation
//! - ✅ Serializable Snapshot Isolation (SSI)
//! - ✅ Group commit for high throughput
//! - ✅ Version chains with garbage collection
//!
//! ```ignore
//! use toondb_storage::wal_integration::MvccTransactionManager;
//!
//! let txm = MvccTransactionManager::new("wal.log", |key, value| {
//!     // Apply callback
//!     Ok(())
//! })?;
//!
//! let txn_id = txm.begin(IsolationLevel::Serializable)?;
//! txm.write(txn_id, b"key", b"value")?;
//! txm.commit(txn_id)?; // fsync guarantee
//! ```
//!
//! ### 2. `MvccManager` (durable_storage.rs) - SSI VALIDATION ONLY
//!
//! Provides SSI conflict detection but delegates durability to storage layer.
//! Used internally by `DurableStorage` for transaction tracking.
//!
//! - ✅ SSI rw-antidependency detection
//! - ✅ Bloom filter optimization for read/write sets
//! - ❌ No WAL (relies on caller for durability)
//!
//! ### 3. `TransactionManager` (mvcc_snapshot.rs) - **DEPRECATED FOR NEW CODE**
//!
//! Minimal snapshot isolation without durability. Only suitable for:
//! - Unit testing
//! - Ephemeral in-memory operations
//!
//! **Do not use for production workloads requiring crash recovery.**
//!
//! ## Transaction Coordinator Trait
//!
//! The `TransactionCoordinator` trait unifies the common interface:

use toondb_core::Result;

// Re-export TransactionMode from durable_storage (canonical definition)
pub use crate::durable_storage::TransactionMode;

/// Isolation level for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Read committed - see all committed data
    ReadCommitted,
    /// Snapshot isolation (default) - consistent point-in-time view
    #[default]
    SnapshotIsolation,
    /// Serializable via SSI - full serializability guarantee
    Serializable,
}

/// Transaction handle returned by begin operations
#[derive(Debug, Clone)]
pub struct TransactionHandle {
    /// Unique transaction ID
    pub txn_id: u64,
    /// Snapshot timestamp for MVCC visibility
    pub snapshot_ts: u64,
    /// Transaction mode
    pub mode: TransactionMode,
    /// Isolation level
    pub isolation_level: IsolationLevel,
}

/// Unified transaction coordinator interface
///
/// This trait defines the canonical API for transaction management.
/// All new transaction manager implementations should implement this trait.
///
/// ## Durability Contract
///
/// Implementations MUST document their durability guarantees:
/// - `Durable`: Committed transactions survive crash (WAL + fsync)
/// - `Ephemeral`: No durability guarantee (in-memory only)
pub trait TransactionCoordinator: Send + Sync {
    /// Begin a new transaction with default isolation level
    fn begin(&self) -> Result<TransactionHandle>;

    /// Begin a transaction with specified isolation level
    fn begin_with_isolation(&self, isolation: IsolationLevel) -> Result<TransactionHandle>;

    /// Begin a transaction with specified mode (for optimization)
    fn begin_with_mode(&self, mode: TransactionMode) -> Result<TransactionHandle>;

    /// Commit a transaction
    ///
    /// For durable implementations, this includes fsync.
    /// Returns the commit timestamp on success.
    fn commit(&self, txn_id: u64) -> Result<u64>;

    /// Abort a transaction
    ///
    /// Discards all changes made by the transaction.
    fn abort(&self, txn_id: u64) -> Result<()>;

    /// Get the current snapshot timestamp for a transaction
    fn get_snapshot_ts(&self, txn_id: u64) -> Option<u64>;

    /// Check if a transaction is still active
    fn is_active(&self, txn_id: u64) -> bool;

    /// Record a read for SSI tracking
    fn record_read(&self, txn_id: u64, key: &[u8]);

    /// Record a write for SSI tracking
    fn record_write(&self, txn_id: u64, key: &[u8]);

    /// Get durability guarantee of this implementation
    fn durability(&self) -> DurabilityLevel;
}

/// Durability guarantee level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityLevel {
    /// Fully durable with WAL + fsync
    Durable,
    /// Group commit (durable after batch fsync)
    GroupCommit,
    /// No durability (in-memory only)
    Ephemeral,
}

/// Recovery statistics returned after crash recovery
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of transactions recovered
    pub transactions_recovered: usize,
    /// Number of individual writes recovered
    pub writes_recovered: usize,
    /// The commit timestamp after recovery
    pub commit_ts: u64,
}

// =============================================================================
// Re-exports for convenience
// =============================================================================

// Note: The canonical implementation is MvccTransactionManager from wal_integration
// pub use crate::wal_integration::MvccTransactionManager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_mode_tracking() {
        assert!(TransactionMode::ReadWrite.tracks_reads());
        assert!(TransactionMode::ReadWrite.tracks_writes());
        
        assert!(!TransactionMode::ReadOnly.tracks_reads());
        assert!(!TransactionMode::ReadOnly.tracks_writes());
        
        assert!(!TransactionMode::WriteOnly.tracks_reads());
        assert!(TransactionMode::WriteOnly.tracks_writes());
    }

    #[test]
    fn test_isolation_level_default() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::SnapshotIsolation);
    }
}
