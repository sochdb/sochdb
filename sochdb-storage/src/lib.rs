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

//! SochDB Storage Layer
//!
//! Log-Structured Column Store (LSCS) with transaction-aware WAL for TOON-native data.
//!
//! ## Runtime Modes
//!
//! This crate supports two runtime modes:
//!
//! ### Embedded Sync Mode (like SQLite)
//!
//! For embedded deployments without async runtime:
//!
//! ```toml
//! sochdb-storage = { version = "...", default-features = false, features = ["embedded-sync"] }
//! ```
//!
//! Benefits:
//! - ~500KB smaller binary
//! - No async runtime overhead
//! - Simpler embedded integration
//!
//! ### Async Mode (default, for servers)
//!
//! For server deployments with async I/O:
//!
//! ```toml
//! sochdb-storage = { version = "..." }  # async enabled by default
//! ```
//!
//! Benefits:
//! - Better scalability for concurrent connections
//! - Non-blocking I/O for server workloads
//!
//! ## Novel Components
//!
//! - **LSCS** (`lscs`): Log-Structured Column Store - columnar variant of LSM with
//!   schema-aware compression and column-aware compaction for reduced write amplification.
//!
//! - **Transaction WAL** (`txn_wal`): ACID-compliant Write-Ahead Log with transaction
//!   boundaries, commit/abort markers, and crash recovery.
//!
//! - **StorageEngine Trait** (`storage_engine`): Pluggable storage backend abstraction
//!   enabling 80% I/O reduction for columnar projections (Task 1).
//!
//! - **Page Manager** (`page_manager`): TOON file format with magic header and O(1)
//!   page allocation (Task 8).
//!
//! - **Columnar Compression** (`columnar_compression`): Type-aware encoding with
//!   dictionary, RLE, and delta compression for 2-4× storage reduction (Task 9).
//!
//! ## Utility Components
//!
//! - **Bloom Filters** (`bloom`): Probabilistic existence checks
//! - **Block Checksums** (`block_checksum`): Data integrity validation
//! - **Compression** (`compression`): LZ4/Zstd compression
//! - **Sketches** (`sketches`): Approximate algorithms (HyperLogLog, CountMin, DDSketch)

// New TOON-native storage components
pub mod actor; // Actor-based connection manager (mm.md Task 7.2)
pub mod admission_control; // Admission control with cost model + tenant fairness (Task 6)
#[cfg(feature = "experimental")]
pub mod aries_recovery; // ARIES-style crash recovery (Task 1) [quarantined: unwired]
pub mod cdc; // WAL-derived Change Data Capture (T1)
#[cfg(feature = "experimental")]
pub mod checkpoint; // ARIES-style checkpointing with WAL truncation (mm.md Task 1.4) [quarantined: unwired]
pub mod columnar_compression;
pub mod correctness_testing; // Property-based correctness testing (Task 13)
pub mod database; // Database Kernel (shared by embedded + server)
pub mod durability_contract; // Durability contract hardening (Task 4)
pub mod durable_storage; // Fully wired durable storage with MVCC
pub mod encryption; // Data-at-rest encryption (AES-256-GCM-SIV envelope) — now wired (Task 3B)
pub mod ffi;
pub mod group_commit; // Event-driven Group Commit (Task 4)
pub mod hlc; // Hybrid Logical Clock for commit timestamps (mm.md Task 1.3)
pub mod hybrid_store; // PAX hybrid row-column storage (mm.md Task 4.1)
pub mod io_isolation; // I/O isolation policy with cache partitioning (Task 5)
pub mod ipc; // IPC Protocol with multiplexing (mm.md Task 7.1)
#[cfg(unix)]
pub mod ipc_server; // Unix Socket IPC Server (Task 3)
pub mod keyring; // KEK/DEK envelope: HKDF-derived DEK, wrapped + persisted, fail-closed (Task 3B)
pub mod learned_index_integration;
pub mod lock; // Advisory file locking for database exclusivity
pub mod lscs;
pub mod mvcc_concurrent; // Concurrent MVCC for multi-reader single-writer (Task: Concurrent Embedded)
#[deprecated(
    note = "Unused duplicate; live MVCC is mvcc_concurrent::ConcurrentMvcc + durable_storage::MvccMemTable. Scheduled for removal (Task 2 consolidation)."
)]
pub mod mvcc_new;
pub mod mvcc_snapshot;
pub mod page_manager;
#[cfg(feature = "experimental")]
pub mod pitr; // Point-in-Time Recovery with WAL archiving (Task 11) [quarantined: unwired]
#[cfg(feature = "experimental")]
pub mod production_wal; // Production WAL with ARIES recovery (mm.md Task 3) [quarantined: unwired]
pub mod ssi; // Serializable Snapshot Isolation (Task 2)
#[deprecated(
    note = "Unused; SSI lives in ssi/MvccManager. Scheduled for removal (Task 2 consolidation)."
)]
pub mod ssi_scaling; // SSI scaling guardrails with range locks (Task 7)
pub mod storage_engine;
pub mod streaming_iterator; // Streaming Iterator Architecture (mm.md Task 4)
pub mod supervisor; // Supervised background workers (panic-contained restart) (Task 4)
pub mod transaction; // Unified Transaction Coordinator trait and types
pub mod txn_arena; // Transaction-scoped arena with zero-copy key/value plumbing
pub mod txn_wal;
pub mod upgrade_contract; // Upgrade compatibility contract (Task 12)
#[cfg(feature = "experimental")]
pub mod wal_fencing; // Epoch-based WAL fencing for split-brain detection [quarantined: unwired]
pub mod wal_integration;
pub mod wal_manifest; // Durable PITR anchor (last-checkpoint LSN + DB identity), crash-safe (Task 3B PITR)
pub mod zero_copy_safety; // Zero-Copy Validation Layer (Task 5) // FFI bindings for Python SDK

// Performance optimization modules
#[deprecated(
    note = "Unused duplicate; live learned index is learned_index_integration. Scheduled for removal (Task 2 consolidation)."
)]
pub mod adaptive_learned_index;
#[deprecated(
    note = "Unused duplicate memtable; live memtables are lscs::ColumnarMemtable + durable_storage::MvccMemTable. Scheduled for removal (Task 2 consolidation)."
)]
pub mod adaptive_memtable; // Adaptive memtable sizing with memory pressure (Task 10)
pub mod batch_wal; // Batched WAL with vectored I/O (Task 3)
pub mod deferred_index; // Deferred sorted index with LSM-style compaction (Rec 2)
pub mod dirty_tracking; // Batched dirty tracking with MPSC queue
pub mod index_policy; // Per-table index policy
pub mod key_buffer; // Cache-line aligned key buffer (Task 2)
#[deprecated(
    note = "Unused duplicate memtable; live memtables are lscs::ColumnarMemtable + durable_storage::MvccMemTable. Scheduled for removal (Task 2 consolidation)."
)]
pub mod lockfree_memtable; // Lock-free read path with hazard pointers (Task 4)
pub mod packed_row;
pub mod queue_index; // Queue-optimized index structure (Task: Queue Index Policy) // Unified row storage with delta encoding (Task 1)

// PhD-Level Architectural Optimizations (December 2025)
#[deprecated(
    note = "Unused duplicate; live learned index is learned_index_integration. Scheduled for removal (Task 2 consolidation)."
)]
pub mod clr_learned_index; // CLR Learned Index for sorted runs (Task 3)
#[cfg(feature = "experimental")]
pub mod columnar_wal; // Columnar WAL Layout (Task 4) [quarantined: unwired]
pub mod epoch_arena; // Epoch-Partitioned Key Arena (Task 1)
pub mod generational_slab; // Generational Slab Allocator (Task 5)
#[cfg(feature = "experimental")]
pub mod hierarchical_ts; // Hierarchical Timestamp Oracle (Task 9) [quarantined: unwired]
#[cfg(all(unix, feature = "experimental"))]
pub mod io_uring_wal; // [quarantined: unwired]
pub mod lockfree_epoch; // Lock-Free Epoch Tracking (Task 3)
pub mod polymorphic_value; // Polymorphic Value Encoding (Task 12)
#[cfg(feature = "experimental")]
pub mod rl_workload; // RL Workload Classifier (Task 10) [quarantined: unwired]
pub mod shard_coalesced; // Shard-Coalesced Batch DashMap (Task 6)
#[deprecated(
    note = "Unused duplicate memtable; live memtables are lscs::ColumnarMemtable + durable_storage::MvccMemTable. Scheduled for removal (Task 2 consolidation)."
)]
pub mod stratified_skiplist; // Stratified SkipList with Deferred Promotion (Task 2) // io_uring WAL Submission (Task 11)

// New performance modules (Recommendations 1-9)
pub mod cow_btree; // Copy-on-Write B-Tree for ordered access (Recommendation 5)
pub mod epoch_mvcc; // Epoch-based MVCC for O(log E) version lookup (Recommendation 7)
pub mod page_cache; // Application-level page cache with Clock-Pro (Recommendation 8)
pub mod row_format; // Slot-based columnar row storage (Recommendation 1)
pub mod tiered_memtable; // Tiered MemTable with deferred sorting (Recommendation 3)
pub mod tournament_tree; // K-way merge with tournament tree (Task 2)
pub mod vectorized_scan; // SIMD-accelerated vectorized scan engine (Recommendation 2)
pub mod zero_copy_serde; // Zero-copy serialization for WAL (Recommendation 6)

// Namespace and multi-tenancy support (Task 3)
pub mod lazy_namespace; // Per-namespace lazy hydrate/evict
pub mod namespace; // Namespace routing and on-disk layout
pub mod object_store_tier; // Object-storage cold tier for immutable segments

// Core utilities
pub mod backend;
pub mod backup;
pub mod block_checksum;
pub mod bloom;
pub mod compression;
pub mod dict_compression;
pub mod direct_io;
#[cfg(unix)]
pub mod io_uring;
pub mod manifest;
pub mod memory;
pub mod parallel_merge;
pub mod payload;
pub mod prefetch;
pub mod sketches;
pub mod two_level_index;
pub mod validation;
pub mod version_store;
pub mod zero_copy;

// Re-exports for new components
pub use columnar_compression::{
    ColumnEncoder, DeltaEncoder, DictionaryEncoder, EncodingStats, EncodingType, RleEncoder,
};
pub use learned_index_integration::{
    HybridIndex, IndexManager, IndexType, KeyStats, PointLookupExecutor,
};
pub use lscs::{
    ColumnDef, ColumnGroup, ColumnType, ColumnarMemtable, Lscs, LscsConfig, LscsRecoveryStats,
    LscsStats, TableSchema,
};
#[allow(deprecated)]
pub use mvcc_snapshot::{
    MvccStore, Snapshot as MvccSnapshot, Timestamp, TransactionManager, TxnId, TxnStatus,
    VersionChain, VersionInfo,
};
pub use page_manager::{
    DEFAULT_PAGE_SIZE, DbHeader, FORMAT_VERSION, FreePageHeader, PageId, PageManager,
    PageManagerStats, PageType, SOCHDB_MAGIC,
};
pub use storage_engine::{
    ColumnId, ColumnIterator, Row, RowId, StorageEngine, StorageEngineType, StorageStats,
    TxnHandle, open_storage_engine,
};
pub use transaction::{
    DurabilityLevel, IsolationLevel, RecoveryStats as TxnRecoveryStats, TransactionCoordinator,
    TransactionHandle,
};
pub use txn_wal::{
    CrashRecoveryStats, RecoveryTarget, TxnWal, TxnWalBuffer, TxnWalEntry, TxnWalStats,
};
pub use wal_integration::{
    GroupCommitBuffer, MvccTransactionManager, RecoveryStats, Transaction, TxnState,
    WalStorageManager,
};
pub use wal_manifest::WalManifest;

// Re-exports for performance optimization modules
#[allow(deprecated)]
pub use adaptive_learned_index::{AdaptiveLearnedIndex, LearnedIndexStats, PiecewiseLinearModel};
#[allow(deprecated)]
pub use adaptive_memtable::{
    AdaptiveMemtableConfig, AdaptiveMemtableSizer, AdaptiveMemtableStats, DEFAULT_BASE_SIZE,
    MAX_MEMTABLE_SIZE, MIN_MEMTABLE_SIZE,
};
pub use batch_wal::{
    BatchAccumulator, BatchedWalReader, BatchedWalStats, BatchedWalWriter, ConcurrentBatchedWal,
    DEFAULT_MAX_BATCH_BYTES, DEFAULT_MAX_BATCH_SIZE,
};
#[allow(deprecated)]
pub use clr_learned_index::{ClrIndex, ClrLookupResult, ClrStats, IndexedSortedRun};
pub use key_buffer::{
    ArenaKey,
    ArenaKeyHandle,
    BatchKeyGenerator,
    InternedTablePrefix,
    // Arena allocation for high-throughput key operations
    KeyArena,
    KeyBuffer,
    MAX_KEY_LENGTH,
};
#[allow(deprecated)]
pub use lockfree_memtable::{
    HazardDomain,
    INLINE_VALUE_SIZE,
    LockFreeMemTable,
    LockFreeVersion,
    LockFreeVersionChain,
    // Inline value storage for reduced memory indirection
    ValueStorage,
};
pub use packed_row::{
    PackedColumnDef, PackedColumnType, PackedRow, PackedRowBuilder, PackedTableSchema,
};

// Re-exports for utilities
pub use backend::{LocalFsBackend, ObjectMetadata, StorageBackend};
pub use backup::{BackupManager, BackupMetadata};
pub use block_checksum::{
    BlockChecksumConfig, BlockChecksumStats, BlockType as BlockChecksumType, BlockWriter,
    ChecksummedBlock,
};
pub use bloom::{BlockedBloomFilter, BloomFilter, LevelAdaptiveFPR, UnifiedBloomFilter};
pub use compression::{CompressionEngine, CompressionStats, StorageTier};
pub use manifest::{FileMetadata, LsmState, Manifest, VersionEdit};
pub use memory::{MemoryBudget, MemoryTracker, WriteBufferManager, WriteBufferStats};
#[allow(deprecated)]
pub use mvcc_new::{
    ColumnGroupRef, ReadVersion, Snapshot, SnapshotGuard, VersionGuard, VersionSet,
    VersionSetStats, VersionSetStatsSnapshot,
};
pub use payload::{CompressionType, PayloadStats, PayloadStore};
pub use sketches::{AdaptiveSketch, CountMinSketch, DDSketch, ExponentialHistogram, HyperLogLog};
pub use two_level_index::{
    BlockIndexEntry, BlockIndexReader, FencePointer, TemporalKey, TwoLevelIndex,
};
pub use validation::{SSTableValidator, validate_sstable_file};

// Re-exports for durable storage
pub use durable_storage::{
    ArenaMvccMemTable, DurableStorage, EphemeralHandle, MvccMemTable, StorageEncryption,
    TransactionMode,
};
// At-rest encryption public surface (Task 3B), reachable from the crate root
// alongside DurableStorage::open_with_encryption / Database::open_with_config_and_encryption.
pub use encryption::{EncryptionEngine, EncryptionError, EncryptionKey, generate_key};
pub use keyring::EncryptionState;

// ============================================================================
// Truth-in-capabilities: durability feature matrix (Task 3A)
// ============================================================================

/// Durability features actually wired into THIS build's live storage path.
///
/// Prose like "production-grade" must not be read as implying features that are
/// quarantined behind the empty, non-default `experimental` feature and
/// unreferenced by the live write/recovery path. Query this matrix instead of
/// trusting documentation strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurabilityCapabilities {
    /// Crash-consistent WAL recovery (txn_wal / RecoveryStats / durability_contract). Live.
    pub crash_recovery: bool,
    /// At-rest encryption (AES-256-GCM-SIV envelope). Wired into the live WAL path
    /// (Task 3B): inactive by default, active per-database when a key is configured.
    /// The build-level `durability_capabilities()` reports the DEFAULT (false);
    /// query `DurableStorage::durability_capabilities()` for the live per-instance
    /// state.
    pub at_rest_encryption: bool,
    /// Point-in-time recovery via WAL archiving. `pitr` module — substrate landing
    /// incrementally (Task 3B); reported true per-instance once archiving is active.
    pub point_in_time_recovery: bool,
    /// ARIES-style checkpointing. `aries_recovery` / `checkpoint` modules, quarantined/unwired.
    pub aries_checkpoint: bool,
    /// Epoch-based WAL fencing (split-brain detection). `wal_fencing` module, quarantined/unwired.
    pub wal_fencing: bool,
}

/// The DEFAULT durability capabilities of the current build — a function of what
/// is actually wired, not of documentation. At-rest encryption is now wired into
/// the live WAL path (Task 3B) but is INACTIVE unless a key is configured, so the
/// build default reports it `false`. For the live per-database state (which
/// reflects whether encryption is actually active on that instance), call
/// [`durable_storage::DurableStorage::durability_capabilities`].
pub const fn durability_capabilities() -> DurabilityCapabilities {
    DurabilityCapabilities {
        crash_recovery: true,
        at_rest_encryption: false,
        point_in_time_recovery: false,
        aries_checkpoint: false,
        wal_fencing: false,
    }
}

#[cfg(test)]
mod durability_capabilities_tests {
    use super::durability_capabilities;

    #[test]
    fn live_build_durability_matrix_is_honest() {
        let caps = durability_capabilities();
        // The one durability guarantee the live path actually provides.
        assert!(
            caps.crash_recovery,
            "live path must provide crash-consistent WAL recovery"
        );
        // Quarantined/unwired — must NOT be advertised as present on the live build.
        assert!(!caps.at_rest_encryption);
        assert!(!caps.point_in_time_recovery);
        assert!(!caps.aries_checkpoint);
        assert!(!caps.wal_fencing);
    }
}

// Re-exports for concurrent MVCC (Task: Concurrent Embedded)
pub use mvcc_concurrent::{
    ConcurrentMvcc, ConcurrentVersionChain, ConcurrentVersionEntry, HlcTimestamp, ReaderSlot,
    VersionStore, VersionStoreStats, WriterGuard,
};

// Super Version and Copy-on-Write Version Set (mm.md Task 1)
pub mod compaction_policy;
pub mod concurrent_art;
pub mod optimized_scan;
pub mod sstable;
pub mod version_set;
pub mod wal_segment;

// Re-exports for new performance modules (Recommendations 1-9)
pub use compaction_policy::{
    CompactionConfig, CompactionFile, CompactionJob, CompactionPicker, CompactionPriority,
    CompactionReason, CompactionState, CompactionStats, CompactionStrategy,
    LeveledCompactionPicker, RetentionConfig, UniversalCompactionPicker, VersionPruner,
};
pub use concurrent_art::ConcurrentART;
pub use cow_btree::{BTreeEntry, BTreeSnapshot, CowBTree, Node, SearchResult};
pub use epoch_mvcc::{
    CommitResult, EpochManager, EpochMvccStore, EpochSnapshot, EpochTransaction, EpochVersionChain,
    GcStats, StoreStats, VersionEntry,
};
pub use lazy_namespace::{LazyNamespaceConfig, LazyNamespaceTable};
pub use object_store_tier::{ObjectStoreTier, ObjectStoreTierConfig, SegmentDescriptor};
pub use optimized_scan::{
    EntrySource, FileRange, LevelFiles, RangeScanner, ScanConfig, ScanStats, TournamentTree,
    VersionedEntry,
};
pub use page_cache::{CacheStats, CachedPage, ClockProCache, PageId as CachePageId, PageState};
pub use row_format::{Slot, SlotRow, SlotRowArena, SlotRowFlags, SlotRowHandle};
pub use sstable::{
    BlockBuilder, BlockCache, BlockHandle, BlockIterator, BlockType, BloomFilterPolicy,
    FilterPolicy, FilterReader, Footer, Header, ReadOptions, RibbonFilterPolicy, SSTable,
    SSTableBuilder, SSTableBuilderOptions, SSTableBuilderResult, SSTableFormat, Section,
    SectionType, TableMetadata, XorFilterPolicy,
};
pub use tiered_memtable::{HotEntry, SortedBatch, TieredMemTable};
pub use vectorized_scan::{
    ColumnVector,
    ComparisonOp,
    DEFAULT_BATCH_SIZE,
    Int64Comparison,
    // SoA + Late Materialization (80/20 optimization)
    SimdVisibilityFilter,
    SoaBatch,
    SoaScanIterator,
    SoaScanStats,
    SoaSource,
    StreamingScanIterator,
    ValueHandle,
    VectorBatch,
    VectorPredicate,
    VectorizedScanConfig,
    VectorizedScanStats,
    VersionedSlice,
};
pub use version_set::{
    FileMetadata as VersionFileMetadata, ImmutableMemTable, ImmutableMemTableRef, LevelMetadata,
    SuperVersion, SuperVersionHandle, VersionSet as CowVersionSet,
};
pub use wal_segment::{
    CheckpointRecord, RecoveryIterator, SegmentConfig, SegmentHeader, SegmentMetadata,
    SegmentStats, WalEntry, WalSegmentManager,
};
pub use zero_copy_serde::{
    FORMAT_VERSION as SERDE_FORMAT_VERSION, FieldDescriptor, HEADER_SIZE as SERDE_HEADER_SIZE,
    MmapWalReader, SerdeStats, WalBatchReader, WalBatchWriter, WalEntryBuilder, WalEntryHeader,
    WalEntryReader, WalEntryType, ZERO_COPY_MAGIC, ZeroCopyHeader,
};

// Re-exports for transaction arena and zero-copy plumbing
pub use txn_arena::{ArenaWriteSet, BytesRef, KeyFingerprint, TxnArena, TxnWriteBuffer, WriteOp};

// Re-exports for dirty tracking with batching
pub use dirty_tracking::{BatchedDirtyTracker, DirtyEvent, DirtyTrackingStats, TxnDirtyBuffer};

// Re-exports for per-table index policy
pub use index_policy::{
    BalancedTableIndex, IndexPolicy, SortedRun, TableIndexConfig, TableIndexRegistry,
};

// Re-exports for queue-optimized index structure
pub use queue_index::{
    CompositeQueueKey, QueueIndex, QueueIndexConfig, QueueIndexStats, QueueTableRegistry,
};

// Re-exports for CDC engine
pub use cdc::{CdcConfig, CdcEmitter, CdcError, CdcEvent, CdcLog, CdcOperation, CdcSubscriber};

// Re-exports for database kernel
pub use database::{
    ColumnDef as DbColumnDef,
    ColumnType as DbColumnType,
    ColumnarQueryResult, // SIMD-friendly columnar result format
    Database,
    DatabaseConfig,
    GroupCommitSettings,
    QueryBuilder,
    QueryResult,
    QueryRowIterator,
    RecoveryStats as DbRecoveryStats,
    Stats as DbStats,
    SyncMode,
    TableSchema as DbTableSchema,
    TxnHandle as KernelTxnHandle,
    VectorSearchResult,
};
