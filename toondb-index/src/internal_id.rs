// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Dense internal ID remapping for HNSW index.
//!
//! This module provides a bijective mapping between external user IDs (u128)
//! and internal dense IDs (u32) for optimal cache and memory performance.
//!
//! # Motivation
//!
//! External IDs (u128) are 16 bytes each. In the HNSW traversal hot path:
//! - Neighbor lists store IDs directly (16 bytes × M neighbors = 256+ bytes/node)
//! - Visited sets hash/compare 128-bit values
//! - Cache lines wasted on ID padding
//!
//! Dense remapping to u32 provides:
//! - 4× reduction in ID storage (4 bytes vs 16 bytes)
//! - 4× more IDs fit in cache line (16 vs 4)
//! - Faster visited set operations (u32 hash/cmp is single instruction)
//! - Enables bitmap-based visited sets for even faster membership tests
//!
//! # Implementation
//!
//! The mapping is established at build time:
//! - `external_to_internal`: HashMap<u128, u32> for insert-time lookup
//! - `internal_to_external`: Vec<u128> for O(1) reverse lookup
//!
//! Internal IDs are assigned sequentially: φ: external_id → [0..N)
//! This is a "coordinate compression" transform that's O(N) at build time.

use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum number of vectors supported with u32 internal IDs.
/// This gives us ~4 billion vectors which exceeds practical HNSW limits.
pub const MAX_INTERNAL_ID: u32 = u32::MAX - 1;

/// Reserved value for "no neighbor" or "invalid" slots in CSR arrays.
pub const INVALID_INTERNAL_ID: u32 = u32::MAX;

/// Internal ID type for all graph operations.
/// This is a newtype for type safety to prevent mixing with external IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct InternalId(pub u32);

impl InternalId {
    /// Create a new internal ID.
    #[inline(always)]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    #[inline(always)]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Check if this is a valid ID (not the invalid sentinel).
    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        self.0 != INVALID_INTERNAL_ID
    }

    /// Invalid/sentinel value for empty neighbor slots.
    #[inline(always)]
    pub const fn invalid() -> Self {
        Self(INVALID_INTERNAL_ID)
    }
}

impl From<u32> for InternalId {
    #[inline(always)]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<InternalId> for u32 {
    #[inline(always)]
    fn from(id: InternalId) -> Self {
        id.0
    }
}

impl From<InternalId> for usize {
    #[inline(always)]
    fn from(id: InternalId) -> Self {
        id.0 as usize
    }
}

/// Bidirectional mapping between external u128 IDs and internal u32 IDs.
///
/// Thread-safe for concurrent insertions with atomic ID counter.
/// Lookup operations are lock-free after initial insertion.
pub struct IdMapper {
    /// External → Internal mapping (for insertion and external queries)
    external_to_internal: DashMap<u128, InternalId>,

    /// Internal → External mapping (for returning results)
    /// Uses RwLock<Vec> for O(1) indexed access with growable storage
    internal_to_external: RwLock<Vec<u128>>,

    /// Atomic counter for assigning sequential internal IDs
    next_id: AtomicU32,
}

impl IdMapper {
    /// Create a new ID mapper with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create a new ID mapper with specified initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            external_to_internal: DashMap::with_capacity(capacity),
            internal_to_external: RwLock::new(Vec::with_capacity(capacity)),
            next_id: AtomicU32::new(0),
        }
    }

    /// Get the number of mapped IDs.
    #[inline]
    pub fn len(&self) -> usize {
        self.next_id.load(Ordering::Relaxed) as usize
    }

    /// Check if mapper is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Register an external ID and get its internal ID.
    ///
    /// If the external ID already exists, returns the existing internal ID.
    /// If new, assigns the next sequential internal ID.
    ///
    /// # Panics
    /// Panics if we exceed MAX_INTERNAL_ID (4 billion vectors).
    pub fn register(&self, external_id: u128) -> InternalId {
        // Fast path: already registered
        if let Some(internal) = self.external_to_internal.get(&external_id) {
            return *internal;
        }

        // Slow path: register new ID (uses entry API to handle races)
        let entry = self.external_to_internal.entry(external_id);
        
        match entry {
            dashmap::mapref::entry::Entry::Occupied(e) => *e.get(),
            dashmap::mapref::entry::Entry::Vacant(e) => {
                // Allocate new internal ID atomically
                let internal_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                
                if internal_id >= MAX_INTERNAL_ID {
                    panic!("Exceeded maximum internal ID capacity ({})", MAX_INTERNAL_ID);
                }
                
                let internal = InternalId::new(internal_id);
                e.insert(internal);
                
                // Update reverse mapping
                let mut reverse = self.internal_to_external.write();
                if (internal_id as usize) >= reverse.len() {
                    reverse.resize(internal_id as usize + 1, 0);
                }
                reverse[internal_id as usize] = external_id;
                
                internal
            }
        }
    }

    /// Look up internal ID for an external ID.
    #[inline]
    pub fn to_internal(&self, external_id: u128) -> Option<InternalId> {
        self.external_to_internal.get(&external_id).map(|r| *r)
    }

    /// Look up external ID for an internal ID.
    #[inline]
    pub fn to_external(&self, internal_id: InternalId) -> Option<u128> {
        let reverse = self.internal_to_external.read();
        let idx = internal_id.get() as usize;
        if idx < reverse.len() {
            Some(reverse[idx])
        } else {
            None
        }
    }

    /// Bulk convert internal IDs to external IDs.
    /// More efficient than repeated single lookups for result sets.
    pub fn to_external_batch(&self, internal_ids: &[InternalId]) -> Vec<u128> {
        let reverse = self.internal_to_external.read();
        internal_ids
            .iter()
            .filter_map(|&id| {
                let idx = id.get() as usize;
                if idx < reverse.len() {
                    Some(reverse[idx])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clear all mappings (for testing/reset).
    pub fn clear(&self) {
        self.external_to_internal.clear();
        self.internal_to_external.write().clear();
        self.next_id.store(0, Ordering::SeqCst);
    }

    /// Shrink internal storage to fit current size.
    pub fn shrink_to_fit(&self) {
        self.external_to_internal.shrink_to_fit();
        self.internal_to_external.write().shrink_to_fit();
    }

    /// Get memory usage estimate in bytes.
    pub fn memory_usage(&self) -> usize {
        let forward_size = self.external_to_internal.len() * (16 + 4 + 8); // key + value + overhead
        let reverse_size = self.internal_to_external.read().capacity() * 16;
        forward_size + reverse_size
    }
}

impl Default for IdMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for IdMapper {
    fn clone(&self) -> Self {
        let reverse = self.internal_to_external.read();
        let cloned = Self {
            external_to_internal: self.external_to_internal.clone(),
            internal_to_external: RwLock::new(reverse.clone()),
            next_id: AtomicU32::new(self.next_id.load(Ordering::SeqCst)),
        };
        cloned
    }
}

/// Bitmap-based visited set for ultra-fast membership testing.
///
/// For dense internal IDs, a bitmap is more cache-efficient than HashSet:
/// - O(1) insert/lookup with single bit operation
/// - 8× more IDs per cache line vs HashSet<u32>
/// - No hashing overhead
///
/// Memory: N/8 bytes (e.g., 1M vectors = 125 KB)
pub struct VisitedBitmap {
    bits: Vec<u64>,
    len: usize,
}

impl VisitedBitmap {
    /// Create a new visited bitmap for up to `capacity` internal IDs.
    pub fn new(capacity: usize) -> Self {
        let num_words = (capacity + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            len: capacity,
        }
    }

    /// Check if an ID has been visited.
    #[inline(always)]
    pub fn contains(&self, id: InternalId) -> bool {
        let idx = id.get() as usize;
        if idx >= self.len {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    /// Mark an ID as visited. Returns true if it was not already visited.
    #[inline(always)]
    pub fn insert(&mut self, id: InternalId) -> bool {
        let idx = id.get() as usize;
        if idx >= self.len {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        let mask = 1u64 << bit;
        let was_unset = (self.bits[word] & mask) == 0;
        self.bits[word] |= mask;
        was_unset
    }

    /// Clear all visited markers (reuse between searches).
    #[inline]
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// Get capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_id_basic() {
        let id = InternalId::new(42);
        assert_eq!(id.get(), 42);
        assert!(id.is_valid());
        
        let invalid = InternalId::invalid();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_id_mapper_basic() {
        let mapper = IdMapper::new();
        
        // First registration
        let internal1 = mapper.register(1000);
        assert_eq!(internal1.get(), 0);
        
        // Second unique ID
        let internal2 = mapper.register(2000);
        assert_eq!(internal2.get(), 1);
        
        // Duplicate registration returns same ID
        let internal1_again = mapper.register(1000);
        assert_eq!(internal1_again, internal1);
        
        // Verify reverse lookup
        assert_eq!(mapper.to_external(internal1), Some(1000));
        assert_eq!(mapper.to_external(internal2), Some(2000));
        
        // Verify forward lookup
        assert_eq!(mapper.to_internal(1000), Some(internal1));
        assert_eq!(mapper.to_internal(2000), Some(internal2));
        assert_eq!(mapper.to_internal(3000), None);
    }

    #[test]
    fn test_id_mapper_bulk() {
        let mapper = IdMapper::new();
        
        // Register many IDs
        for i in 0..1000u128 {
            let internal = mapper.register(i * 100);
            assert_eq!(internal.get(), i as u32);
        }
        
        // Bulk conversion
        let internals: Vec<_> = (0..10).map(InternalId::new).collect();
        let externals = mapper.to_external_batch(&internals);
        assert_eq!(externals.len(), 10);
        for (i, &ext) in externals.iter().enumerate() {
            assert_eq!(ext, i as u128 * 100);
        }
    }

    #[test]
    fn test_visited_bitmap() {
        let mut visited = VisitedBitmap::new(1000);
        
        // Initially empty
        assert!(!visited.contains(InternalId::new(0)));
        assert!(!visited.contains(InternalId::new(500)));
        
        // Insert returns true for new, false for duplicate
        assert!(visited.insert(InternalId::new(42)));
        assert!(!visited.insert(InternalId::new(42)));
        
        // Contains works
        assert!(visited.contains(InternalId::new(42)));
        assert!(!visited.contains(InternalId::new(43)));
        
        // Insert multiple
        for i in 0..100 {
            visited.insert(InternalId::new(i * 10));
        }
        
        for i in 0..100 {
            assert!(visited.contains(InternalId::new(i * 10)));
        }
        
        // Clear works
        visited.clear();
        assert!(!visited.contains(InternalId::new(42)));
    }

    #[test]
    fn test_visited_bitmap_edge_cases() {
        let mut visited = VisitedBitmap::new(64);
        
        // Test boundary between words
        assert!(visited.insert(InternalId::new(63)));
        assert!(visited.contains(InternalId::new(63)));
        
        // Out of bounds returns false
        assert!(!visited.insert(InternalId::new(64)));
        assert!(!visited.contains(InternalId::new(100)));
    }

    #[test]
    fn test_id_mapper_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let mapper = Arc::new(IdMapper::new());
        let mut handles = vec![];

        // Concurrent insertions from multiple threads
        for t in 0..4 {
            let mapper = Arc::clone(&mapper);
            handles.push(thread::spawn(move || {
                for i in 0..250 {
                    let ext_id = (t * 1000 + i) as u128;
                    mapper.register(ext_id);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 1000 unique IDs
        assert_eq!(mapper.len(), 1000);

        // All should be retrievable
        for t in 0..4u128 {
            for i in 0..250u128 {
                let ext_id = t * 1000 + i;
                assert!(mapper.to_internal(ext_id).is_some());
            }
        }
    }
}
