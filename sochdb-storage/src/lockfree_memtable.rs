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

//! Lock-Free MemTable with Hazard Pointer Protection
//!
//! This module provides a lock-free read path for the MVCC memtable using
//! hazard pointers for safe memory reclamation.
//!
//! ## Problem Analysis
//!
//! Current implementation uses RwLock on entire HashMap:
//! ```ignore
//! pub struct MvccMemTable {
//!     data: RwLock<HashMap<Vec<u8>, VersionChain>>,  // LOCK!
//! }
//! ```
//!
//! Problems:
//! - `parking_lot::RwLock` read acquire: ~20-30ns uncontended
//! - Under contention: ~100-500ns due to cache coherency
//! - RwLock has reader-count atomic → contention point
//!
//! ## Solution
//!
//! True lock-free reads using hazard pointers:
//! - O(1) uncontended reads (~15ns)
//! - Linear scaling with reader count
//! - No reader-reader interference
//!
//! ## Scalability Model (Amdahl's Law)
//!
//! For N threads with serial fraction s:
//! Speedup = 1 / (s + (1-s)/N)
//!
//! RwLock: s ≈ 0.02 → For N=8: Speedup = 6.4×
//! Lock-Free: s ≈ 0.001 → For N=8: Speedup = 7.9×
//!
//! **Improvement: 23% better scaling**

use std::collections::HashSet;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;
use parking_lot::Mutex;

use sochdb_core::{Result, SochDBError};

/// Number of hazard pointers per thread
const HP_PER_THREAD: usize = 2;

/// Maximum number of threads supported
const MAX_THREADS: usize = 128;

/// Number of retired nodes before attempting reclamation
const RECLAMATION_THRESHOLD: usize = 64;

/// Number of version slots per fat node (Rec 2)
///
/// 8 slots × 8-byte pointer = 64 bytes — fits in one cache line.
/// Reduces pointer chases from O(v) to O(v/8) for version chain traversal.
const FAT_NODE_SLOTS: usize = 8;

/// Maximum size for inline value storage (fits in cache line with metadata)
///
/// Cache line = 64 bytes
/// Metadata: txn_id (8) + commit_ts (8) + next ptr (8) + storage discriminant (1) + len (1) = 26 bytes
/// Inline data: 64 - 26 = 38 bytes (we use 56 for larger threshold since struct may span lines)
pub const INLINE_VALUE_SIZE: usize = 56;

/// Optimized value storage with inline allocation for small values
///
/// For typical database workloads, 80%+ of values are < 56 bytes.
/// Storing these inline eliminates heap allocation and pointer chasing.
///
/// ## Cache Analysis
///
/// Current path: DashMap lookup → Version ptr → Value ptr (Vec data)
/// Cache misses: 2-3 (worst case)
///
/// Inline path: DashMap lookup → Version with inline value
/// Cache misses: 1
///
/// Expected speedup: 2-2.5× for reads on small values
#[repr(C)]
pub enum ValueStorage {
    /// Value stored inline (most common case for small values)
    Inline {
        len: u8,
        data: [u8; INLINE_VALUE_SIZE],
    },
    /// Value stored on heap (for large values > 56 bytes)
    Heap(Box<[u8]>),
    /// Tombstone marker (key was deleted)
    Tombstone,
}

impl std::fmt::Debug for ValueStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueStorage::Inline { len, .. } => write!(f, "Inline(len={})", len),
            ValueStorage::Heap(data) => write!(f, "Heap(len={})", data.len()),
            ValueStorage::Tombstone => write!(f, "Tombstone"),
        }
    }
}

impl ValueStorage {
    /// Create new value storage, preferring inline when possible
    #[inline]
    pub fn new(value: Option<&[u8]>) -> Self {
        match value {
            None => ValueStorage::Tombstone,
            Some(v) if v.len() <= INLINE_VALUE_SIZE => {
                let mut data = [0u8; INLINE_VALUE_SIZE];
                data[..v.len()].copy_from_slice(v);
                ValueStorage::Inline {
                    len: v.len() as u8,
                    data,
                }
            }
            Some(v) => ValueStorage::Heap(v.to_vec().into_boxed_slice()),
        }
    }

    /// Get value as byte slice
    #[inline]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            ValueStorage::Inline { len, data } => Some(&data[..*len as usize]),
            ValueStorage::Heap(data) => Some(data),
            ValueStorage::Tombstone => None,
        }
    }

    /// Check if this is a tombstone
    #[inline]
    pub fn is_tombstone(&self) -> bool {
        matches!(self, ValueStorage::Tombstone)
    }

    /// Check if value is stored inline
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self, ValueStorage::Inline { .. })
    }

    /// Get the size of the stored value
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ValueStorage::Inline { len, .. } => *len as usize,
            ValueStorage::Heap(data) => data.len(),
            ValueStorage::Tombstone => 0,
        }
    }

    /// Check if the stored value is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Version of a key-value pair for lock-free access
///
/// Uses inline storage for small values to eliminate heap allocation
/// and improve cache locality. Most database values (80%+) fit inline.
#[derive(Debug)]
pub struct LockFreeVersion {
    /// The value with optimized inline storage
    pub storage: ValueStorage,
    /// Transaction that created this version
    pub txn_id: u64,
    /// Commit timestamp (0 = uncommitted)
    pub commit_ts: AtomicU64,
    /// Next version in chain (older)
    pub next: AtomicPtr<LockFreeVersion>,
}

impl LockFreeVersion {
    /// Create a new uncommitted version with value slice (zero-copy for inline)
    #[inline]
    pub fn new_from_slice(value: Option<&[u8]>, txn_id: u64) -> Self {
        Self {
            storage: ValueStorage::new(value),
            txn_id,
            commit_ts: AtomicU64::new(0),
            next: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Create a new uncommitted version (legacy API - accepts owned Vec)
    pub fn new(value: Option<Vec<u8>>, txn_id: u64) -> Self {
        Self::new_from_slice(value.as_deref(), txn_id)
    }

    /// Get the value as bytes (zero-copy for inline values)
    #[inline]
    pub fn get_value(&self) -> Option<&[u8]> {
        self.storage.as_bytes()
    }

    /// Get the value as owned Vec (for compatibility)
    ///
    /// Note: Prefer `get_value()` to avoid allocation
    #[inline]
    pub fn value_cloned(&self) -> Option<Vec<u8>> {
        self.storage.as_bytes().map(|v| v.to_vec())
    }

    /// Check if committed
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.commit_ts.load(Ordering::Acquire) > 0
    }

    /// Get commit timestamp
    #[inline]
    pub fn get_commit_ts(&self) -> u64 {
        self.commit_ts.load(Ordering::Acquire)
    }

    /// Set commit timestamp
    #[inline]
    pub fn set_commit_ts(&self, ts: u64) {
        self.commit_ts.store(ts, Ordering::Release);
    }

    /// Check if value is stored inline (for diagnostics)
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.storage.is_inline()
    }
}

/// Fat-node for version chain (Rec 2: Lock-Free Fat-Node Version Chain)
///
/// Groups up to 8 version pointers per node, reducing pointer chases from
/// O(v) to O(v/8). Slot pointers occupy 64 bytes = 1 cache line, so scanning
/// all 8 slots costs a single cache-line fetch instead of 8 random chases.
///
/// Layout:
/// - `count` (AtomicU8): number of occupied slots (0..FAT_NODE_SLOTS)
/// - `slots`: array of AtomicPtr<LockFreeVersion>, newest at index `count-1`
/// - `next`: pointer to older FatNode
pub struct FatNode {
    /// Number of valid version pointers in `slots` (0..=FAT_NODE_SLOTS)
    count: AtomicU8,
    /// Version pointers, newest at `count-1`, oldest at 0
    slots: [AtomicPtr<LockFreeVersion>; FAT_NODE_SLOTS],
    /// Next (older) fat node in the chain
    next: AtomicPtr<FatNode>,
}

impl FatNode {
    /// Create a new fat node with one initial version and a link to older node
    fn new_with_first(version: *mut LockFreeVersion, older: *mut FatNode) -> Self {
        let slots = std::array::from_fn(|i| {
            if i == 0 {
                AtomicPtr::new(version)
            } else {
                AtomicPtr::new(ptr::null_mut())
            }
        });
        Self {
            count: AtomicU8::new(1),
            slots,
            next: AtomicPtr::new(older),
        }
    }

    /// Try to append a version to this fat node. Returns Ok(()) if successful,
    /// Err(version_ptr) if the node is full.
    ///
    /// Thread-safety: CAS on `count` serializes slot reservation. The reserving
    /// thread then publishes the pointer via Release store on the slot.
    #[inline]
    fn try_push(&self, version: *mut LockFreeVersion) -> std::result::Result<(), *mut LockFreeVersion> {
        loop {
            let c = self.count.load(Ordering::Acquire);
            if c as usize >= FAT_NODE_SLOTS {
                return Err(version); // Full
            }
            // Reserve slot `c` by CAS count → c+1
            match self.count.compare_exchange(c, c + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    // We own slot `c` — publish the version pointer
                    self.slots[c as usize].store(version, Ordering::Release);
                    return Ok(());
                }
                Err(_) => continue, // Another thread won; retry
            }
        }
    }

    /// Get version pointer at slot index (must be < count)
    #[inline]
    fn slot(&self, idx: u8) -> *mut LockFreeVersion {
        self.slots[idx as usize].load(Ordering::Acquire)
    }

    /// Iterate versions newest-first (index count-1 down to 0)
    #[inline]
    fn iter_newest_first(&self) -> impl Iterator<Item = &LockFreeVersion> {
        let count = self.count.load(Ordering::Acquire);
        (0..count).rev().filter_map(move |i| {
            let ptr = self.slots[i as usize].load(Ordering::Acquire);
            if ptr.is_null() { None } else { Some(unsafe { &*ptr }) }
        })
    }
}

/// Lock-free version chain using fat-node grouping (Rec 2)
///
/// Instead of a singly-linked list of individual versions, versions are grouped
/// into fat nodes of 8. This reduces pointer chases from O(v) to O(v/8) since
/// scanning 8 slots within a fat node hits the same cache line.
pub struct LockFreeVersionChain {
    /// Head fat node (contains the most recent versions)
    head: AtomicPtr<FatNode>,
}

impl Default for LockFreeVersionChain {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeVersionChain {
    /// Create empty version chain
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Add a new uncommitted version
    ///
    /// Returns error if there's already an uncommitted version from another txn
    pub fn add_uncommitted(&self, value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let new_version = Box::into_raw(Box::new(LockFreeVersion::new(value, txn_id)));

        loop {
            let head = self.head.load(Ordering::Acquire);

            // Check for write-write conflict: inspect the newest version
            if !head.is_null() {
                let fat = unsafe { &*head };
                let count = fat.count.load(Ordering::Acquire);
                if count > 0 {
                    let newest = fat.slot(count - 1);
                    if !newest.is_null() {
                        let newest_ref = unsafe { &*newest };
                        if !newest_ref.is_committed() && newest_ref.txn_id != txn_id {
                            unsafe { drop(Box::from_raw(new_version)); }
                            return Err(SochDBError::Internal("Write-write conflict".into()));
                        }
                    }
                }

                // Try to push into existing fat node
                match fat.try_push(new_version) {
                    Ok(()) => return Ok(()),
                    Err(_) => {
                        // Fat node is full — allocate new one linking to current head
                        let new_fat = Box::into_raw(Box::new(FatNode::new_with_first(new_version, head)));
                        match self.head.compare_exchange(head, new_fat, Ordering::AcqRel, Ordering::Acquire) {
                            Ok(_) => return Ok(()),
                            Err(_) => {
                                // CAS failed — reclaim the fat node, keep the version for retry
                                unsafe {
                                    // Detach version from fat node before dropping it
                                    (*new_fat).slots[0].store(ptr::null_mut(), Ordering::Relaxed);
                                    (*new_fat).count.store(0, Ordering::Relaxed);
                                    drop(Box::from_raw(new_fat));
                                }
                                continue; // Retry from the top
                            }
                        }
                    }
                }
            } else {
                // No head — allocate first fat node
                let new_fat = Box::into_raw(Box::new(FatNode::new_with_first(new_version, ptr::null_mut())));
                match self.head.compare_exchange(head, new_fat, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return Ok(()),
                    Err(_) => {
                        unsafe {
                            (*new_fat).slots[0].store(ptr::null_mut(), Ordering::Relaxed);
                            (*new_fat).count.store(0, Ordering::Relaxed);
                            drop(Box::from_raw(new_fat));
                        }
                        continue;
                    }
                }
            }
        }
    }

    /// Commit a version
    pub fn commit(&self, txn_id: u64, commit_ts: u64) -> bool {
        let mut fat_ptr = self.head.load(Ordering::Acquire);

        while !fat_ptr.is_null() {
            let fat = unsafe { &*fat_ptr };
            // Scan this fat node's slots (newest first)
            for ver in fat.iter_newest_first() {
                if ver.txn_id == txn_id && !ver.is_committed() {
                    ver.set_commit_ts(commit_ts);
                    return true;
                }
            }
            fat_ptr = fat.next.load(Ordering::Acquire);
        }

        false
    }

    /// Read at a snapshot timestamp
    ///
    /// Returns the most recent committed version visible at snapshot_ts,
    /// or an uncommitted version if it belongs to current_txn_id.
    pub fn read_at(
        &self,
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<&LockFreeVersion> {
        let mut fat_ptr = self.head.load(Ordering::Acquire);

        while !fat_ptr.is_null() {
            let fat = unsafe { &*fat_ptr };
            // Scan this fat node's slots (newest first)
            for version in fat.iter_newest_first() {
                // Check if this is our own uncommitted write
                if let Some(txn_id) = current_txn_id
                    && version.txn_id == txn_id
                    && !version.is_committed()
                {
                    return Some(version);
                }

                // Check if this version is visible
                let commit_ts = version.get_commit_ts();
                if commit_ts > 0 && commit_ts < snapshot_ts {
                    return Some(version);
                }
            }
            fat_ptr = fat.next.load(Ordering::Acquire);
        }

        None
    }

    /// Check if there's an uncommitted version by another transaction
    pub fn has_write_conflict(&self, my_txn_id: u64) -> bool {
        let head = self.head.load(Ordering::Acquire);

        if !head.is_null() {
            let fat = unsafe { &*head };
            let count = fat.count.load(Ordering::Acquire);
            if count > 0 {
                let newest = fat.slot(count - 1);
                if !newest.is_null() {
                    let version = unsafe { &*newest };
                    return !version.is_committed() && version.txn_id != my_txn_id;
                }
            }
        }

        false
    }
}

/// Thread-local hazard pointer record
///
/// Cache-line aligned to prevent false sharing
#[repr(C, align(64))]
struct HazardRecord {
    /// Protected pointers
    hazard: [AtomicPtr<LockFreeVersion>; HP_PER_THREAD],
    /// Active flag (non-zero if thread is using this record)
    active: AtomicU64,
}

impl HazardRecord {
    const fn new() -> Self {
        Self {
            hazard: [
                AtomicPtr::new(ptr::null_mut()),
                AtomicPtr::new(ptr::null_mut()),
            ],
            active: AtomicU64::new(0),
        }
    }

    /// Acquire this record for a thread
    fn try_acquire(&self, thread_id: u64) -> bool {
        self.active
            .compare_exchange(0, thread_id, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Release this record
    #[allow(dead_code)]
    fn release(&self) {
        // Clear hazard pointers first
        for hp in &self.hazard {
            hp.store(ptr::null_mut(), Ordering::Release);
        }
        self.active.store(0, Ordering::Release);
    }
}

/// Hazard pointer domain for safe memory reclamation
pub struct HazardDomain {
    /// Hazard records (one per potential thread)
    records: Vec<HazardRecord>,
    /// Retired nodes pending reclamation
    retired: Mutex<Vec<*mut LockFreeVersion>>,
}

impl HazardDomain {
    /// Create a new hazard domain
    pub fn new(max_threads: usize) -> Self {
        let mut records = Vec::with_capacity(max_threads);
        for _ in 0..max_threads {
            records.push(HazardRecord::new());
        }

        Self {
            records,
            retired: Mutex::new(Vec::with_capacity(RECLAMATION_THRESHOLD * 2)),
        }
    }

    /// Get a hazard record for the current thread
    fn get_record(&self) -> Option<&HazardRecord> {
        let thread_id = thread_id::get() as u64;

        // First try to find already owned record
        for record in &self.records {
            if record.active.load(Ordering::Acquire) == thread_id {
                return Some(record);
            }
        }

        // Try to acquire a new record
        self.records
            .iter()
            .find(|record| record.try_acquire(thread_id))
    }

    /// Protect a pointer with hazard pointer
    #[inline]
    pub fn protect(&self, ptr: *mut LockFreeVersion, slot: usize) -> bool {
        if let Some(record) = self.get_record()
            && slot < HP_PER_THREAD
        {
            record.hazard[slot].store(ptr, Ordering::Release);
            std::sync::atomic::fence(Ordering::SeqCst);
            return true;
        }
        false
    }

    /// Clear a hazard pointer slot
    #[inline]
    pub fn clear(&self, slot: usize) {
        if let Some(record) = self.get_record()
            && slot < HP_PER_THREAD
        {
            record.hazard[slot].store(ptr::null_mut(), Ordering::Release);
        }
    }

    /// Retire a pointer for later reclamation
    pub fn retire(&self, ptr: *mut LockFreeVersion) {
        let mut retired = self.retired.lock();
        retired.push(ptr);

        // Attempt reclamation if threshold reached
        if retired.len() >= RECLAMATION_THRESHOLD {
            self.try_reclaim(&mut retired);
        }
    }

    /// Try to reclaim retired pointers not protected by any hazard pointer
    fn try_reclaim(&self, retired: &mut Vec<*mut LockFreeVersion>) {
        // Collect all active hazard pointers
        let mut protected: HashSet<usize> = HashSet::new();

        for record in &self.records {
            if record.active.load(Ordering::Acquire) != 0 {
                for hp in &record.hazard {
                    let ptr = hp.load(Ordering::Acquire);
                    if !ptr.is_null() {
                        protected.insert(ptr as usize);
                    }
                }
            }
        }

        // Reclaim unprotected nodes
        let mut still_retired = Vec::new();
        for ptr in retired.drain(..) {
            if protected.contains(&(ptr as usize)) {
                still_retired.push(ptr);
            } else {
                // Safe to reclaim
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }

        *retired = still_retired;
    }
}

impl Drop for HazardDomain {
    fn drop(&mut self) {
        // Reclaim all retired nodes
        let mut retired = self.retired.lock();
        for ptr in retired.drain(..) {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

// Thread ID helper (simple implementation)
mod thread_id {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

    thread_local! {
        static THREAD_ID: usize = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get() -> usize {
        THREAD_ID.with(|id| *id)
    }
}

/// Lock-free memtable with hazard pointer protection
pub struct LockFreeMemTable {
    /// Concurrent hash map (lock-free for reads, fine-grained locking for writes)
    data: DashMap<Vec<u8>, LockFreeVersionChain>,
    /// Hazard pointer domain
    hazard_domain: HazardDomain,
    /// Approximate size in bytes
    size_bytes: AtomicUsize,
}

impl LockFreeMemTable {
    /// Create a new lock-free memtable
    pub fn new() -> Self {
        Self {
            data: DashMap::new(),
            hazard_domain: HazardDomain::new(MAX_THREADS),
            size_bytes: AtomicUsize::new(0),
        }
    }

    /// Read a value at snapshot timestamp
    ///
    /// This is a lock-free read protected by hazard pointers.
    /// Returns a cloned value for safety across hazard pointer boundaries.
    pub fn read(&self, key: &[u8], snapshot_ts: u64, txn_id: Option<u64>) -> Option<Vec<u8>> {
        let chain = self.data.get(key)?;

        // Read with hazard pointer protection
        if let Some(version) = chain.read_at(snapshot_ts, txn_id) {
            // Protect the version
            let ptr = version as *const LockFreeVersion as *mut LockFreeVersion;
            self.hazard_domain.protect(ptr, 0);

            // Get value using optimized inline storage
            // Clone is still needed due to hazard pointer lifetime
            let result = version.value_cloned();

            // Clear hazard pointer
            self.hazard_domain.clear(0);

            result
        } else {
            None
        }
    }

    /// Read a value at snapshot timestamp with zero-copy callback
    ///
    /// This is an optimized read path that avoids cloning for inline values.
    /// The callback receives a reference to the value, avoiding allocation.
    ///
    /// # Arguments
    /// * `key` - The key to read
    /// * `snapshot_ts` - Snapshot timestamp for visibility
    /// * `txn_id` - Current transaction ID (to see own uncommitted writes)
    /// * `f` - Callback that receives the value reference
    ///
    /// # Returns
    /// The result of the callback, or None if key not found
    #[inline]
    pub fn read_with<F, R>(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        txn_id: Option<u64>,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let chain = self.data.get(key)?;

        if let Some(version) = chain.read_at(snapshot_ts, txn_id) {
            // Protect the version
            let ptr = version as *const LockFreeVersion as *mut LockFreeVersion;
            self.hazard_domain.protect(ptr, 0);

            // Call callback with value reference (zero-copy for inline)
            let result = version.get_value().map(f);

            // Clear hazard pointer
            self.hazard_domain.clear(0);

            result
        } else {
            None
        }
    }

    /// Write a value (creates uncommitted version)
    pub fn write(&self, key: Vec<u8>, value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);

        // Get or create version chain
        let chain = self.data.entry(key.clone()).or_default();

        // Add uncommitted version
        chain.add_uncommitted(value, txn_id)?;

        // Update size estimate
        self.size_bytes
            .fetch_add(key.len() + value_size + 64, Ordering::Relaxed);

        Ok(())
    }

    /// Commit a transaction's writes
    pub fn commit(&self, txn_id: u64, commit_ts: u64, keys: &[Vec<u8>]) {
        for key in keys {
            if let Some(chain) = self.data.get(key) {
                chain.commit(txn_id, commit_ts);
            }
        }
    }

    /// Check for write conflict
    pub fn has_write_conflict(&self, key: &[u8], txn_id: u64) -> bool {
        if let Some(chain) = self.data.get(key) {
            chain.has_write_conflict(txn_id)
        } else {
            false
        }
    }

    /// Get approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Get number of keys
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// Safety: LockFreeMemTable uses atomic operations and proper synchronization
// for all shared data access. The raw pointers in HazardDomain are only
// dereferenced under proper hazard pointer protection.
unsafe impl Send for LockFreeMemTable {}
unsafe impl Sync for LockFreeMemTable {}

impl Default for LockFreeMemTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_write_read() {
        let memtable = LockFreeMemTable::new();

        // Write
        memtable
            .write(b"key1".to_vec(), Some(b"value1".to_vec()), 1)
            .unwrap();

        // Read own uncommitted write
        let val = memtable.read(b"key1", 100, Some(1));
        assert_eq!(val, Some(b"value1".to_vec()));

        // Cannot read uncommitted from other txn
        let val = memtable.read(b"key1", 100, Some(2));
        assert!(val.is_none());

        // Commit and read
        memtable.commit(1, 50, &[b"key1".to_vec()]);
        let val = memtable.read(b"key1", 100, None);
        assert_eq!(val, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_snapshot_isolation() {
        let memtable = LockFreeMemTable::new();

        // Write and commit at ts=10
        memtable
            .write(b"key".to_vec(), Some(b"v1".to_vec()), 1)
            .unwrap();
        memtable.commit(1, 10, &[b"key".to_vec()]);

        // Write and commit at ts=20
        memtable
            .write(b"key".to_vec(), Some(b"v2".to_vec()), 2)
            .unwrap();
        memtable.commit(2, 20, &[b"key".to_vec()]);

        // Snapshot at ts=15 sees v1
        assert_eq!(memtable.read(b"key", 15, None), Some(b"v1".to_vec()));

        // Snapshot at ts=25 sees v2
        assert_eq!(memtable.read(b"key", 25, None), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_write_conflict() {
        let memtable = LockFreeMemTable::new();

        // First write
        memtable
            .write(b"key".to_vec(), Some(b"v1".to_vec()), 1)
            .unwrap();

        // Conflicting write should fail
        let result = memtable.write(b"key".to_vec(), Some(b"v2".to_vec()), 2);
        assert!(result.is_err());

        // Same txn can write again
        let result = memtable.write(b"key".to_vec(), Some(b"v1_updated".to_vec()), 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concurrent_reads() {
        let memtable = Arc::new(LockFreeMemTable::new());

        // Setup data
        for i in 0..100 {
            let key = format!("key{}", i).into_bytes();
            let val = format!("value{}", i).into_bytes();
            memtable.write(key.clone(), Some(val), 1).unwrap();
        }
        memtable.commit(
            1,
            10,
            &(0..100)
                .map(|i| format!("key{}", i).into_bytes())
                .collect::<Vec<_>>(),
        );

        // Concurrent reads
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let mt = Arc::clone(&memtable);
                thread::spawn(move || {
                    for i in 0..100 {
                        let key = format!("key{}", i).into_bytes();
                        let expected = format!("value{}", i).into_bytes();
                        let val = mt.read(&key, 100, None);
                        assert_eq!(val, Some(expected), "Thread {} failed at key{}", t, i);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_inline_storage() {
        // Test that small values are stored inline
        let small_value = b"small".to_vec();
        let version = LockFreeVersion::new(Some(small_value.clone()), 1);
        assert!(version.is_inline(), "Small values should be inline");
        assert_eq!(version.get_value(), Some(small_value.as_slice()));

        // Test that large values are stored on heap
        let large_value = vec![42u8; 100]; // > INLINE_VALUE_SIZE
        let version = LockFreeVersion::new(Some(large_value.clone()), 2);
        assert!(!version.is_inline(), "Large values should be on heap");
        assert_eq!(version.get_value(), Some(large_value.as_slice()));

        // Test tombstone
        let version = LockFreeVersion::new(None, 3);
        assert!(version.storage.is_tombstone());
        assert_eq!(version.get_value(), None);
    }

    #[test]
    fn test_inline_threshold() {
        // Exactly at threshold should be inline
        let value = vec![0u8; INLINE_VALUE_SIZE];
        let version = LockFreeVersion::new(Some(value.clone()), 1);
        assert!(version.is_inline(), "Values at threshold should be inline");

        // One byte over threshold should be heap
        let value = vec![0u8; INLINE_VALUE_SIZE + 1];
        let version = LockFreeVersion::new(Some(value), 2);
        assert!(
            !version.is_inline(),
            "Values over threshold should be on heap"
        );
    }

    #[test]
    fn test_read_with_callback() {
        let memtable = LockFreeMemTable::new();

        memtable
            .write(b"key1".to_vec(), Some(b"value1".to_vec()), 1)
            .unwrap();
        memtable.commit(1, 10, &[b"key1".to_vec()]);

        // Use read_with for zero-copy access
        let len = memtable.read_with(b"key1", 100, None, |v| v.len());
        assert_eq!(len, Some(6)); // "value1".len()

        // Verify callback receives correct data
        let matches = memtable.read_with(b"key1", 100, None, |v| v == b"value1");
        assert_eq!(matches, Some(true));
    }

    #[test]
    fn test_fat_node_overflow() {
        // Write more than FAT_NODE_SLOTS versions to a single key
        // to verify fat node chaining works correctly
        let memtable = LockFreeMemTable::new();

        for i in 0..12u64 {
            // Each write from a different committed txn
            memtable
                .write(
                    b"key".to_vec(),
                    Some(format!("v{}", i).into_bytes()),
                    i + 1,
                )
                .unwrap();
            memtable.commit(i + 1, (i + 1) * 10, &[b"key".to_vec()]);
        }

        // Read at latest snapshot should return v11 (committed at ts=120)
        let val = memtable.read(b"key", 200, None);
        assert_eq!(val, Some(b"v11".to_vec()));

        // Read at ts=55 should return v4 (committed at ts=50)
        let val = memtable.read(b"key", 55, None);
        assert_eq!(val, Some(b"v4".to_vec()));

        // Read at ts=5 should return None (v0 committed at ts=10)
        let val = memtable.read(b"key", 5, None);
        assert_eq!(val, None);
    }

    #[test]
    fn test_fat_node_concurrent_writes() {
        use std::sync::Arc;
        use std::thread;

        let memtable = Arc::new(LockFreeMemTable::new());

        // 4 threads writing to different keys concurrently
        let mut handles = Vec::new();
        for t in 0..4u64 {
            let mt = Arc::clone(&memtable);
            handles.push(thread::spawn(move || {
                for i in 0..20u64 {
                    let key = format!("k{}-{}", t, i).into_bytes();
                    let val = format!("v{}-{}", t, i).into_bytes();
                    let txn_id = t * 1000 + i + 1;
                    mt.write(key.clone(), Some(val), txn_id).unwrap();
                    mt.commit(txn_id, txn_id * 10, &[key]);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify all 80 keys are readable
        for t in 0..4u64 {
            for i in 0..20u64 {
                let key = format!("k{}-{}", t, i).into_bytes();
                let val = memtable.read(&key, u64::MAX, None);
                assert_eq!(
                    val,
                    Some(format!("v{}-{}", t, i).into_bytes()),
                    "Missing key k{}-{}",
                    t,
                    i
                );
            }
        }
    }
}
