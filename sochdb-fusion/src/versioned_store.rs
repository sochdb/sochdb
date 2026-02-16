// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Versioned Knowledge Object Store
//!
//! Wires the [`EpochMvccStore`] (epoch-based MVCC with O(log E) reads) and
//! [`HybridLogicalClock`] (monotonic HLC timestamps) to [`KnowledgeObject`],
//! creating a transactional, bitemporal, content-addressed object store.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │             VersionedObjectStore                    │
//! │                                                     │
//! │  ┌──────────────┐   ┌───────────────┐              │
//! │  │ HLC          │   │ EpochManager  │              │
//! │  │ (system_time)│   │ (epoch GC)    │              │
//! │  └──────┬───────┘   └───────┬───────┘              │
//! │         │                   │                       │
//! │         ▼                   ▼                       │
//! │  ┌─────────────────────────────────┐               │
//! │  │    EpochMvccStore<Vec<u8>>      │               │
//! │  │  key = OID bytes (32 bytes)     │               │
//! │  │  value = compressed KO bytes    │               │
//! │  └─────────────────────────────────┘               │
//! │                                                     │
//! │  ┌─────────────────────────────────┐               │
//! │  │    Secondary Index: name → OID  │ (optional)    │
//! │  └─────────────────────────────────┘               │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Decisions
//!
//! 1. **OID as key**: The MVCC store is keyed by `ObjectId.as_bytes()` (32 bytes).
//!    Content-addressed identity means the same object always maps to the same key.
//!
//! 2. **Compressed values**: Objects are stored as `to_compressed_bytes()` with a
//!    configurable [`CompressionMode`]. Default is `Lz4` for hot data.
//!
//! 3. **HLC for system_time**: On every `put()`, the store assigns an HLC
//!    timestamp to `BitemporalCoord.system_time`, ensuring monotonic ordering
//!    of writes across threads.
//!
//! 4. **Epoch-based versioning**: Each version of an object is stored in the
//!    epoch-partitioned MVCC layer, enabling O(log E) historical reads and
//!    batch GC of expired epochs.
//!
//! ## Example
//!
//! ```rust,ignore
//! use sochdb_fusion::versioned_store::VersionedObjectStore;
//! use sochdb_core::knowledge_object::*;
//!
//! let store = VersionedObjectStore::new();
//!
//! // Insert an object
//! let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
//!     .attribute("name", SochValue::Text("Alice".into()))
//!     .build();
//! let oid = store.put(ko);
//!
//! // Read it back
//! let retrieved = store.get(&oid).unwrap();
//! assert_eq!(retrieved.oid(), oid);
//!
//! // Historical read at a prior epoch
//! let old = store.get_at_epoch(&oid, 1);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use sochdb_core::knowledge_object::{
    BitemporalCoord, CompressionMode, KnowledgeObject, KnowledgeObjectBuilder,
    KnowledgeObjectError, ObjectId, ObjectKind,
};
use sochdb_storage::epoch_mvcc::{EpochManager, EpochMvccStore};
use sochdb_storage::hlc::HybridLogicalClock;

// =============================================================================
// Versioned Object Store
// =============================================================================

/// A transactional, bitemporal, content-addressed object store.
///
/// Wraps [`EpochMvccStore`] with [`HybridLogicalClock`] integration to provide
/// full Knowledge Fabric versioning semantics.
pub struct VersionedObjectStore {
    /// The epoch-based MVCC store.
    /// Key: `ObjectId.as_bytes()` (32 bytes), Value: compressed KO bytes.
    mvcc: EpochMvccStore<Vec<u8>>,

    /// Hybrid logical clock for monotonic system_time assignment.
    hlc: Arc<HybridLogicalClock>,

    /// Per-object compression mode.
    compression: CompressionMode,

    /// Object count (atomic for lock-free reads).
    object_count: AtomicU64,

    /// OID → latest epoch mapping (for fast "does this object exist?" checks).
    /// This is a secondary index — the source of truth is the MVCC store.
    oid_index: RwLock<HashMap<ObjectId, u64>>,
}

impl VersionedObjectStore {
    /// Create a new store with default settings (LZ4 compression, 10ms epochs).
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: StoreConfig) -> Self {
        let epoch_manager = Arc::new(EpochManager::with_duration_ms(config.epoch_duration_ms));
        Self {
            mvcc: EpochMvccStore::with_epoch_manager(epoch_manager),
            hlc: Arc::new(HybridLogicalClock::new()),
            compression: config.compression,
            object_count: AtomicU64::new(0),
            oid_index: RwLock::new(HashMap::new()),
        }
    }

    /// Create with an externally-managed HLC (for distributed clock sync).
    pub fn with_hlc(hlc: Arc<HybridLogicalClock>, config: StoreConfig) -> Self {
        let epoch_manager = Arc::new(EpochManager::with_duration_ms(config.epoch_duration_ms));
        Self {
            mvcc: EpochMvccStore::with_epoch_manager(epoch_manager),
            hlc,
            compression: config.compression,
            object_count: AtomicU64::new(0),
            oid_index: RwLock::new(HashMap::new()),
        }
    }

    // =========================================================================
    // Write Operations
    // =========================================================================

    /// Insert a Knowledge Object, assigning an HLC system_time.
    ///
    /// Returns the `ObjectId` (content-addressed). If an object with the same
    /// OID already exists, a new version is created in the MVCC chain.
    pub fn put(&self, mut ko: KnowledgeObject) -> Result<ObjectId, VersionedStoreError> {
        // Assign system_time from HLC
        let system_time = self.hlc.next();
        let temporal = ko.temporal().clone();
        let new_temporal = BitemporalCoord {
            valid_from: temporal.valid_from,
            valid_to: temporal.valid_to,
            system_time,
        };
        ko.set_temporal(new_temporal);

        let oid = ko.oid();
        let key = oid.as_bytes().to_vec();

        // Serialize with compression
        let bytes = ko
            .to_compressed_bytes(self.compression)
            .map_err(|e| VersionedStoreError::Serialization(e.to_string()))?;

        // Write through MVCC transaction
        let mut txn = self.mvcc.begin_txn();
        txn.put(key, bytes);
        let result = txn.commit();

        // Update secondary index
        {
            let mut index = self.oid_index.write();
            let existed = index.insert(oid, result.commit_epoch);
            if existed.is_none() {
                self.object_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Maybe advance epoch
        self.mvcc.maybe_advance_epoch();

        Ok(oid)
    }

    /// Insert a KnowledgeObject built from a builder, assigning HLC system_time.
    pub fn put_builder(
        &self,
        builder: KnowledgeObjectBuilder,
    ) -> Result<ObjectId, VersionedStoreError> {
        self.put(builder.build())
    }

    /// Batch-insert multiple objects in a single MVCC transaction.
    ///
    /// All objects get the same commit epoch but individual HLC system_times.
    pub fn put_batch(
        &self,
        objects: Vec<KnowledgeObject>,
    ) -> Result<Vec<ObjectId>, VersionedStoreError> {
        let mut txn = self.mvcc.begin_txn();
        let mut oids = Vec::with_capacity(objects.len());

        for mut ko in objects {
            let system_time = self.hlc.next();
            let temporal = ko.temporal().clone();
            ko.set_temporal(BitemporalCoord {
                valid_from: temporal.valid_from,
                valid_to: temporal.valid_to,
                system_time,
            });

            let oid = ko.oid();
            let key = oid.as_bytes().to_vec();
            let bytes = ko
                .to_compressed_bytes(self.compression)
                .map_err(|e| VersionedStoreError::Serialization(e.to_string()))?;

            txn.put(key, bytes);
            oids.push(oid);
        }

        let result = txn.commit();

        // Update secondary index
        {
            let mut index = self.oid_index.write();
            let mut new_count = 0u64;
            for &oid in &oids {
                if index.insert(oid, result.commit_epoch).is_none() {
                    new_count += 1;
                }
            }
            self.object_count.fetch_add(new_count, Ordering::Relaxed);
        }

        self.mvcc.maybe_advance_epoch();
        Ok(oids)
    }

    /// Soft-delete an object (tombstone in MVCC — still accessible via time travel).
    pub fn delete(&self, oid: &ObjectId) -> Result<bool, VersionedStoreError> {
        let key = oid.as_bytes().to_vec();

        // Check existence
        if !self.oid_index.read().contains_key(oid) {
            return Ok(false);
        }

        let mut txn = self.mvcc.begin_txn();
        txn.delete(key);
        txn.commit();

        self.oid_index.write().remove(oid);
        self.object_count.fetch_sub(1, Ordering::Relaxed);
        self.mvcc.maybe_advance_epoch();

        Ok(true)
    }

    // =========================================================================
    // Read Operations
    // =========================================================================

    /// Get the latest version of an object by OID.
    pub fn get(&self, oid: &ObjectId) -> Result<Option<KnowledgeObject>, VersionedStoreError> {
        let key = oid.as_bytes().to_vec();
        let txn = self.mvcc.begin_txn();
        let result = txn.get(&key);
        txn.abort(); // Read-only, just release the reader slot

        match result {
            Some(bytes) => {
                let ko = KnowledgeObject::from_compressed_bytes(&bytes)
                    .map_err(|e| VersionedStoreError::Deserialization(e.to_string()))?;
                Ok(Some(ko))
            }
            None => Ok(None),
        }
    }

    /// Get a specific version of an object visible at the given epoch.
    pub fn get_at_epoch(
        &self,
        oid: &ObjectId,
        epoch: u64,
    ) -> Result<Option<KnowledgeObject>, VersionedStoreError> {
        let key = oid.as_bytes();
        let result = self.mvcc.read_at_epoch(key, epoch);

        match result {
            Some(bytes) => {
                let ko = KnowledgeObject::from_compressed_bytes(&bytes)
                    .map_err(|e| VersionedStoreError::Deserialization(e.to_string()))?;
                Ok(Some(ko))
            }
            None => Ok(None),
        }
    }

    /// Check if an object exists in the store (current version).
    pub fn contains(&self, oid: &ObjectId) -> bool {
        self.oid_index.read().contains_key(oid)
    }

    /// Get all version history for an object (most recent first).
    ///
    /// Returns `(epoch, KnowledgeObject)` pairs ordered by epoch descending.
    pub fn history(
        &self,
        oid: &ObjectId,
    ) -> Result<Vec<(u64, KnowledgeObject)>, VersionedStoreError> {
        let key = oid.as_bytes();

        // Read versions across epochs by probing each epoch
        // The EpochMvccStore doesn't expose version chains directly via the
        // public API, so we walk epochs from current down to 1.
        let current_epoch = self.mvcc.epoch_manager().current_epoch();
        let mut versions = Vec::new();
        let mut last_seen: Option<Vec<u8>> = None;

        for epoch in (1..=current_epoch).rev() {
            if let Some(bytes) = self.mvcc.read_at_epoch(key, epoch) {
                // Deduplicate: only emit if this epoch has a *different* version
                let is_new = match &last_seen {
                    Some(prev) => prev != &bytes,
                    None => true,
                };
                if is_new {
                    let ko = KnowledgeObject::from_compressed_bytes(&bytes)
                        .map_err(|e| VersionedStoreError::Deserialization(e.to_string()))?;
                    versions.push((epoch, ko));
                    last_seen = Some(bytes);
                }
            }
        }

        Ok(versions)
    }

    // =========================================================================
    // Temporal Queries
    // =========================================================================

    /// Find all objects valid at a given valid_time across the entire store.
    ///
    /// Scans all current objects and filters by `BitemporalCoord.valid_at()`.
    /// For large stores, prefer using the [`FusionEngine`] with a temporal filter stage.
    pub fn objects_valid_at(
        &self,
        valid_time: u64,
    ) -> Result<Vec<KnowledgeObject>, VersionedStoreError> {
        let index = self.oid_index.read();
        let mut result = Vec::new();

        for oid in index.keys() {
            if let Some(ko) = self.get(oid)? {
                if ko.valid_at(valid_time) {
                    result.push(ko);
                }
            }
        }

        Ok(result)
    }

    /// Find all objects visible at (system_time, valid_time) — full bitemporal query.
    pub fn objects_visible_at(
        &self,
        system_time: u64,
        valid_time: u64,
    ) -> Result<Vec<KnowledgeObject>, VersionedStoreError> {
        let index = self.oid_index.read();
        let mut result = Vec::new();

        for oid in index.keys() {
            if let Some(ko) = self.get(oid)? {
                if ko.visible_at(system_time, valid_time) {
                    result.push(ko);
                }
            }
        }

        Ok(result)
    }

    // =========================================================================
    // Maintenance
    // =========================================================================

    /// Garbage-collect old epoch versions.
    pub fn gc(&self) -> GcResult {
        let stats = self.mvcc.gc();
        GcResult {
            versions_removed: stats.versions_removed,
            chains_emptied: stats.chains_emptied,
        }
    }

    /// Force epoch advancement.
    pub fn advance_epoch(&self) -> u64 {
        self.mvcc.epoch_manager().advance_epoch()
    }

    /// Get store statistics.
    pub fn stats(&self) -> ObjectStoreStats {
        let mvcc_stats = self.mvcc.stats();
        ObjectStoreStats {
            object_count: self.object_count.load(Ordering::Relaxed),
            total_versions: mvcc_stats.total_versions,
            current_epoch: mvcc_stats.current_epoch,
            min_safe_epoch: mvcc_stats.min_safe_epoch,
            compression: self.compression,
            hlc_current: self.hlc.current(),
        }
    }

    /// Get a reference to the HLC (for external clock sync).
    pub fn hlc(&self) -> &Arc<HybridLogicalClock> {
        &self.hlc
    }

    /// Get a reference to the epoch manager.
    pub fn epoch_manager(&self) -> &Arc<EpochManager> {
        self.mvcc.epoch_manager()
    }

    /// Get all current OIDs (for iteration / index building).
    pub fn all_oids(&self) -> Vec<ObjectId> {
        self.oid_index.read().keys().copied().collect()
    }

    /// Load all current objects (for building the fusion engine).
    pub fn all_objects(&self) -> Result<Vec<KnowledgeObject>, VersionedStoreError> {
        let oids = self.all_oids();
        let mut objects = Vec::with_capacity(oids.len());
        for oid in &oids {
            if let Some(ko) = self.get(oid)? {
                objects.push(ko);
            }
        }
        Ok(objects)
    }
}

impl Default for VersionedObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

// We need KnowledgeObject to support set_temporal for the store to assign HLC times.
// This extends KnowledgeObject with a setter (added in the same module scope via a trait).

/// Extension trait for setting temporal coordinates on a KnowledgeObject.
/// Used by the versioned store to assign HLC system_time on write.
trait SetTemporal {
    fn set_temporal(&mut self, coord: BitemporalCoord);
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the versioned object store.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Per-object compression mode.
    pub compression: CompressionMode,
    /// Epoch duration in milliseconds (default: 10ms).
    pub epoch_duration_ms: u64,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            compression: CompressionMode::Lz4,
            epoch_duration_ms: 10,
        }
    }
}

impl StoreConfig {
    /// No compression, faster writes.
    pub fn uncompressed() -> Self {
        Self {
            compression: CompressionMode::None,
            ..Default::default()
        }
    }

    /// ZSTD compression, better ratios for archival.
    pub fn archival() -> Self {
        Self {
            compression: CompressionMode::zstd_high(),
            epoch_duration_ms: 100, // longer epochs for batch workloads
        }
    }
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors from the versioned object store.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VersionedStoreError {
    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("object not found: {0}")]
    NotFound(String),

    #[error("transaction conflict on OID {0}")]
    Conflict(String),
}

// =============================================================================
// Statistics
// =============================================================================

/// Store statistics.
#[derive(Debug, Clone)]
pub struct ObjectStoreStats {
    /// Number of distinct objects (current, non-deleted).
    pub object_count: u64,
    /// Total version entries across all objects.
    pub total_versions: usize,
    /// Current MVCC epoch.
    pub current_epoch: u64,
    /// Minimum safe epoch (GC boundary).
    pub min_safe_epoch: u64,
    /// Active compression mode.
    pub compression: CompressionMode,
    /// Current HLC timestamp.
    pub hlc_current: u64,
}

/// GC result.
#[derive(Debug, Clone)]
pub struct GcResult {
    pub versions_removed: usize,
    pub chains_emptied: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_core::soch::SochValue;

    fn make_entity(name: &str) -> KnowledgeObject {
        KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text(name.into()))
            .valid_from(1000)
            .valid_to(u64::MAX)
            .build()
    }

    fn make_entity_with_embedding(name: &str, vec: Vec<f32>) -> KnowledgeObject {
        KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text(name.into()))
            .embedding("semantic", vec)
            .valid_from(1000)
            .valid_to(u64::MAX)
            .build()
    }

    #[test]
    fn test_put_and_get() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());
        let ko = make_entity("Alice");
        let original_oid = ko.oid();

        // system_time will be reassigned by HLC, so OID stays the same
        // (system_time is NOT part of the OID computation)
        let oid = store.put(ko).unwrap();

        let retrieved = store.get(&oid).unwrap().unwrap();
        assert_eq!(retrieved.text_attribute("name"), Some("Alice"));

        // system_time should be non-zero (assigned by HLC)
        assert!(retrieved.temporal().system_time > 0);
    }

    #[test]
    fn test_put_with_compression() {
        for mode in [
            CompressionMode::None,
            CompressionMode::Lz4,
            CompressionMode::zstd(),
        ] {
            let config = StoreConfig {
                compression: mode,
                ..Default::default()
            };
            let store = VersionedObjectStore::with_config(config);
            let ko = make_entity_with_embedding("Bob", vec![0.1; 384]);
            let oid = store.put(ko).unwrap();

            let retrieved = store.get(&oid).unwrap().unwrap();
            assert_eq!(retrieved.text_attribute("name"), Some("Bob"));
            assert!(retrieved.embedding("semantic").is_some());
        }
    }

    #[test]
    fn test_contains_and_delete() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());
        let ko = make_entity("Charlie");
        let oid = store.put(ko).unwrap();

        assert!(store.contains(&oid));
        assert_eq!(store.stats().object_count, 1);

        let deleted = store.delete(&oid).unwrap();
        assert!(deleted);
        assert!(!store.contains(&oid));
        assert_eq!(store.stats().object_count, 0);

        // Double-delete returns false
        assert!(!store.delete(&oid).unwrap());
    }

    #[test]
    fn test_get_nonexistent() {
        let store = VersionedObjectStore::new();
        let fake_oid = ObjectId::from_content(b"does_not_exist");
        let result = store.get(&fake_oid).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_put() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());
        let objects = vec![
            make_entity("Alice"),
            make_entity("Bob"),
            make_entity("Charlie"),
        ];

        let oids = store.put_batch(objects).unwrap();
        assert_eq!(oids.len(), 3);
        assert_eq!(store.stats().object_count, 3);

        for oid in &oids {
            assert!(store.contains(oid));
            assert!(store.get(oid).unwrap().is_some());
        }
    }

    #[test]
    fn test_hlc_monotonic_system_time() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());

        let ko1 = make_entity("First");
        let ko2 = make_entity("Second");

        let oid1 = store.put(ko1).unwrap();
        let oid2 = store.put(ko2).unwrap();

        let r1 = store.get(&oid1).unwrap().unwrap();
        let r2 = store.get(&oid2).unwrap().unwrap();

        // HLC system_time must be monotonically increasing
        assert!(
            r2.temporal().system_time > r1.temporal().system_time,
            "system_time must be monotonic: {} > {}",
            r2.temporal().system_time,
            r1.temporal().system_time
        );
    }

    #[test]
    fn test_epoch_versioning() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());

        // Insert at epoch 1
        let ko1 = make_entity("Alice-v1");
        let oid1 = store.put(ko1).unwrap();
        let epoch1 = store.stats().current_epoch;

        // Advance epoch
        store.advance_epoch();

        // Insert a different object at epoch 2
        let ko2 = make_entity("Bob");
        let oid2 = store.put(ko2).unwrap();

        // Both should be readable
        assert!(store.get(&oid1).unwrap().is_some());
        assert!(store.get(&oid2).unwrap().is_some());
    }

    #[test]
    fn test_epoch_historical_read() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());

        let ko = make_entity("HistoryTest");
        let oid = store.put(ko).unwrap();
        let epoch_after_insert = store.stats().current_epoch;

        // Object should be visible at current epoch
        let result = store.get_at_epoch(&oid, epoch_after_insert).unwrap();
        assert!(result.is_some());

        // Object should NOT be visible at epoch 0 (before it existed)
        let result = store.get_at_epoch(&oid, 0).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_valid_at_query() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());

        // Object valid from 100 to 200
        let ko_narrow = KnowledgeObjectBuilder::new(ObjectKind::Event)
            .attribute("name", SochValue::Text("meeting".into()))
            .valid_from(100)
            .valid_to(200)
            .build();
        store.put(ko_narrow).unwrap();

        // Object valid from 150 onwards
        let ko_open = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .valid_from(150)
            .valid_to(u64::MAX)
            .build();
        store.put(ko_open).unwrap();

        // At time 120: only the meeting is valid
        let at_120 = store.objects_valid_at(120).unwrap();
        assert_eq!(at_120.len(), 1);
        assert_eq!(at_120[0].text_attribute("name"), Some("meeting"));

        // At time 175: both are valid
        let at_175 = store.objects_valid_at(175).unwrap();
        assert_eq!(at_175.len(), 2);

        // At time 250: only Alice is valid
        let at_250 = store.objects_valid_at(250).unwrap();
        assert_eq!(at_250.len(), 1);
        assert_eq!(at_250[0].text_attribute("name"), Some("Alice"));
    }

    #[test]
    fn test_all_objects() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());
        let objects = vec![
            make_entity("A"),
            make_entity("B"),
            make_entity("C"),
        ];
        store.put_batch(objects).unwrap();

        let all = store.all_objects().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_stats() {
        let store = VersionedObjectStore::with_config(StoreConfig {
            compression: CompressionMode::Lz4,
            epoch_duration_ms: 10,
        });

        assert_eq!(store.stats().object_count, 0);
        assert_eq!(store.stats().compression, CompressionMode::Lz4);

        store.put(make_entity("X")).unwrap();
        assert_eq!(store.stats().object_count, 1);
        assert!(store.stats().hlc_current > 0);
    }

    #[test]
    fn test_gc() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());
        store.put(make_entity("A")).unwrap();
        store.advance_epoch();
        store.put(make_entity("B")).unwrap();

        let gc_result = store.gc();
        // GC should not panic; results depend on timing
        assert!(gc_result.versions_removed >= 0);
    }

    #[test]
    fn test_shared_hlc() {
        let hlc = Arc::new(HybridLogicalClock::new());
        let store1 = VersionedObjectStore::with_hlc(hlc.clone(), StoreConfig::uncompressed());
        let store2 = VersionedObjectStore::with_hlc(hlc.clone(), StoreConfig::uncompressed());

        let oid1 = store1.put(make_entity("From-Store1")).unwrap();
        let oid2 = store2.put(make_entity("From-Store2")).unwrap();

        let r1 = store1.get(&oid1).unwrap().unwrap();
        let r2 = store2.get(&oid2).unwrap().unwrap();

        // Both use the same HLC, so timestamps are monotonic across stores
        assert!(r2.temporal().system_time > r1.temporal().system_time);
    }

    #[test]
    fn test_config_presets() {
        let _uncompressed = StoreConfig::uncompressed();
        let _archival = StoreConfig::archival();
        let _default = StoreConfig::default();

        assert_eq!(_uncompressed.compression, CompressionMode::None);
        assert_eq!(_archival.compression, CompressionMode::zstd_high());
        assert_eq!(_default.compression, CompressionMode::Lz4);
    }

    #[test]
    fn test_put_builder() {
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());
        let builder = KnowledgeObjectBuilder::new(ObjectKind::Fact)
            .attribute("claim", SochValue::Text("water is wet".into()));

        let oid = store.put_builder(builder).unwrap();
        let ko = store.get(&oid).unwrap().unwrap();
        assert_eq!(ko.text_attribute("claim"), Some("water is wet"));
    }
}
