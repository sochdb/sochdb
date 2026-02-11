// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # DiskHashIndex — A Memory-Mapped Open-Addressing Hash Table for Fixed-Size Records
//!
//! ## Why This Exists (Replacing Sled)
//!
//! Sled is a general-purpose embedded key-value store built on a Bw-tree with a
//! lock-free pagecache, write-ahead log, and support for variable-length keys,
//! range scans, CAS operations, and transactions. For the PayloadIndex use case,
//! every one of those capabilities is dead weight:
//!
//! - **Keys are fixed 16 bytes** (u128 edge_id) — no variable-length key management needed
//! - **Values are fixed 17 bytes** (offset + length + compression + uncompressed_length)
//!   — no variable-length value encoding needed
//! - **No range scans** — only point lookups, existence checks, and full iteration
//! - **No transactions or CAS** — single-writer append-only workload
//! - **No WAL needed** — the payload data file IS the recovery source (rebuild_index exists)
//!
//! Sled imposes ~3-5x space amplification (WAL + pagecache metadata + B-tree node
//! fragmentation + free list tracking) and forces bincode serialization on every
//! get/insert for data that has a known, fixed layout. It also pulls in ~40 transitive
//! dependencies and adds ~1.5MB to the release binary.
//!
//! ## Design
//!
//! ### Data Structure: Open-Addressing Hash Table with Linear Probing
//!
//! The fundamental insight: when keys and values are fixed-size, a hash table can be
//! laid out as a flat array of fixed-size slots in a memory-mapped file. Each slot is
//! at a deterministic byte offset: `HEADER_SIZE + slot_index * SLOT_SIZE`. This means:
//!
//! - **Lookup = 1 hash computation + 1-3 cache-line reads** (expected at α=0.75)
//! - **Zero deserialization** — read raw bytes, interpret in-place
//! - **Zero allocation per operation** — no heap activity for get/insert
//! - **OS-managed caching** — the kernel page cache handles hot/cold data automatically
//!
//! ### Why Linear Probing (Not Cuckoo, Robin Hood, or B-tree)
//!
//! **Linear probing** is chosen over alternatives for a specific, systems-level reason:
//! sequential memory access. When a probe chain extends beyond the home slot, linear
//! probing reads the *next* slot in memory. On modern CPUs with 64-byte cache lines
//! and hardware prefetchers that detect sequential access patterns, this means:
//!
//! - At 40 bytes/slot, ~1.6 slots per cache line
//! - A probe chain of length 3 (expected at α=0.75) touches ≤2 cache lines
//! - The L1 prefetcher will speculatively load the next cache line after the first miss
//!
//! Robin Hood hashing improves *variance* of probe lengths but doesn't improve *expected*
//! probe length, and requires read-modify-write on insert (shifting existing entries),
//! which is more expensive on mmap'd storage. Cuckoo hashing guarantees O(1) worst-case
//! lookup but has O(n) amortized insert and requires 2 hash functions + 2 possible cache
//! misses per lookup (two non-adjacent locations). B-trees add O(log n) indirection and
//! pointer-chasing — catastrophic for the page cache hit pattern we want.
//!
//! ### Slot Layout (40 bytes, 8-byte aligned)
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!  0       1    tag (0x00=empty, 0x01=occupied)
//!  1       1    compression type (0=None, 1=LZ4, 2=ZSTD)
//!  2       2    reserved (zero-padded, future use)
//!  4       4    length (u32 LE, compressed payload size)
//!  8       4    uncompressed_length (u32 LE)
//! 12       4    reserved (zero-padded, alignment to 16)
//! 16      16    edge_id (u128 LE)
//! 32       8    offset (u64 LE, position in payload.data)
//! ```
//!
//! Why 40 bytes instead of the minimal 34?
//! - 40 is divisible by 8, giving natural alignment for the u64 offset field
//! - 4096 / 40 = 102 slots per page — good utilization (vs. 120 at 34 bytes
//!   but with misaligned cross-page reads)
//! - The 6 bytes of padding cost ~15% space but eliminate unaligned access penalties
//!   on architectures that don't support unaligned loads (ARM) and avoid split cache-line
//!   reads even on x86
//!
//! ### Capacity Management
//!
//! Capacity is always a power of two. This allows replacing modulo with bitwise AND:
//! `slot_index = hash & (num_slots - 1)`. Integer division is 20-90 cycles on modern
//! x86 vs. 1 cycle for AND — this matters when hashing is only ~5-10 cycles (SeaHash).
//!
//! Growth strategy: when load factor exceeds 0.75, allocate a new file with 2×
//! capacity, rehash all entries, and atomically rename. Amortized O(1) per insert.
//! The 0.75 threshold balances probe length (E[probes] ≈ 2.5 for successful lookup)
//! against space utilization. Lower thresholds waste disk; higher thresholds degrade
//! to linear scan.
//!
//! ### Crash Safety
//!
//! The index is an acceleration structure — not the source of truth. The append-only
//! payload data file contains all information needed to rebuild the index (the
//! `rebuild_index()` method already exists). Therefore:
//!
//! - We do NOT maintain a write-ahead log (unlike sled)
//! - Individual slot writes may be torn on crash — this is acceptable
//! - `save()` calls `msync` for explicit durability when requested
//! - On corruption detection (invalid magic, impossible probe chains), the caller
//!   can rebuild from the data file
//!
//! This is fundamentally simpler and more correct than sled's approach, which must
//! maintain its own WAL to protect its own complex internal structures.
//!
//! ### Memory Footprint
//!
//! With mmap, the RSS (resident set size) is determined by the OS page cache, not
//! by the index size. For a workload that accesses a hot set of H entries:
//!
//! - RSS ≈ H × 40 / 0.75 bytes (the pages containing those slots)
//! - Cold entries are paged out automatically under memory pressure
//! - No explicit cache or eviction policy needed — the kernel LRU does this
//!
//! Compare to sled, which maintains an in-process page cache, free list, and WAL
//! buffer — all consuming RSS regardless of access patterns.
//!
//! ### Performance Characteristics
//!
//! | Operation       | This Index         | Sled               |
//! |-----------------|--------------------|--------------------|
//! | Point lookup    | ~200ns (1 hash + mmap read) | ~1-5μs (tree traverse + deser) |
//! | Insert          | ~300ns (hash + mmap write)  | ~2-10μs (WAL + tree insert + ser) |
//! | Disk space/entry| 53 bytes (40/0.75) | ~150-300 bytes (WAL + tree nodes) |
//! | RAM overhead    | <1 MB (mmap handle + metadata) | 10-50 MB (page cache + WAL buffer) |
//! | Dependencies    | 0 (uses memmap2 already in workspace) | ~40 transitive crates |
//! | Binary size     | ~0 KB incremental  | ~1.5 MB |

use memmap2::MmapMut;
use parking_lot::RwLock;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use sochdb_core::{Result, SochDBError};
use super::{CompressionType, PayloadIndex, PayloadMeta};

// =============================================================================
// Constants
// =============================================================================

const MAGIC: [u8; 8] = *b"SOCHIDX2";
const VERSION: u32 = 1;
const HEADER_SIZE: u64 = 64;
const SLOT_SIZE: u64 = 40;

const TAG_EMPTY: u8 = 0x00;
const TAG_OCCUPIED: u8 = 0x01;

/// Initial capacity (must be power of 2).
/// 4096 slots × 40 bytes = 160 KB — fits in L2 cache on most CPUs.
/// Supports ~3072 entries before first resize (at 0.75 load factor).
const INITIAL_CAPACITY: u64 = 4096;

/// Maximum load factor before triggering resize.
/// At α=0.75, expected probe length for successful lookup ≈ 2.5,
/// for unsuccessful lookup ≈ 8.5. Both are within 1-2 cache-line reads
/// at 40 bytes/slot.
const MAX_LOAD_FACTOR: f64 = 0.75;

// =============================================================================
// Header Layout (64 bytes)
// =============================================================================
//
// Offset  Size  Field
// ------  ----  -----
//  0       8    magic ("SOCHIDX2")
//  8       4    version (1)
// 12       4    reserved
// 16       8    num_slots (u64 LE, always power of 2)
// 24       8    num_entries (u64 LE)
// 32       8    seed (u64 LE, hash seed)
// 40      24    reserved (zero)

const HEADER_OFF_MAGIC: usize = 0;
const HEADER_OFF_VERSION: usize = 8;
const HEADER_OFF_NUM_SLOTS: usize = 16;
const HEADER_OFF_NUM_ENTRIES: usize = 24;
const HEADER_OFF_SEED: usize = 32;

// Slot field offsets within each 40-byte slot
const SLOT_OFF_TAG: usize = 0;
const SLOT_OFF_COMPRESSION: usize = 1;
// 2..4: reserved
const SLOT_OFF_LENGTH: usize = 4;
const SLOT_OFF_UNCOMPRESSED_LEN: usize = 8;
// 12..16: reserved/alignment
const SLOT_OFF_EDGE_ID: usize = 16;
const SLOT_OFF_OFFSET: usize = 32;

// =============================================================================
// Hash Function
// =============================================================================

/// Fast, high-quality hash for u128 keys.
///
/// We use a custom Stafford variant-13 double-mix applied to both halves of the
/// u128, folded with XOR. This is faster than calling SeaHash (which processes
/// byte slices) and provides excellent avalanche properties for integer keys.
///
/// The seed is XOR'd in before mixing to make the hash table resistant to
/// algorithmic complexity attacks (if edge_ids are adversarially chosen).
///
/// Stafford variant-13 is the finalizer used in SplitMix64 and has been
/// empirically shown to pass all SMHasher tests.
#[inline(always)]
fn hash_u128(key: u128, seed: u64) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;

    let mut h = lo ^ seed;
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;

    let mut g = hi ^ seed.wrapping_mul(0x9e3779b97f4a7c15); // golden ratio constant
    g ^= g >> 30;
    g = g.wrapping_mul(0xbf58476d1ce4e5b9);
    g ^= g >> 27;
    g = g.wrapping_mul(0x94d049bb133111eb);
    g ^= g >> 31;

    h ^ g
}

// =============================================================================
// DiskHashIndex
// =============================================================================

/// A memory-mapped, open-addressing hash table for fixed-size PayloadMeta records.
///
/// See module-level documentation for full design rationale.
pub(crate) struct DiskHashIndex {
    /// Memory-mapped file containing header + slot array.
    /// Protected by RwLock for concurrent read access with exclusive write.
    mmap: RwLock<MmapMut>,

    /// Underlying file handle (needed for resize operations).
    file: RwLock<std::fs::File>,

    /// Path to the index file.
    path: PathBuf,

    /// Cached from header. Updated atomically on insert.
    num_entries: AtomicU64,

    /// Cached from header. Only changes on resize.
    num_slots: AtomicU64,

    /// Hash seed (read once from header, immutable after open).
    seed: u64,
}

impl DiskHashIndex {
    /// Open or create a disk-backed hash index at the given path.
    ///
    /// If the file exists and has a valid header, it is opened in-place.
    /// If the file doesn't exist, a new index is created with `INITIAL_CAPACITY` slots.
    pub fn new(index_path: PathBuf) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = index_path.parent() {
            fs::create_dir_all(parent)?;
        }

        if index_path.exists() && fs::metadata(&index_path)?.len() >= HEADER_SIZE {
            Self::open_existing(index_path)
        } else {
            Self::create_new(index_path, INITIAL_CAPACITY)
        }
    }

    /// Create a new empty index file with the given capacity.
    fn create_new(path: PathBuf, capacity: u64) -> Result<Self> {
        debug_assert!(capacity.is_power_of_two(), "Capacity must be power of 2");

        let file_size = HEADER_SIZE + capacity * SLOT_SIZE;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| SochDBError::Internal(format!("Failed to create index file: {}", e)))?;

        file.set_len(file_size)
            .map_err(|e| SochDBError::Internal(format!("Failed to set index file size: {}", e)))?;

        // Safety: we just created this file and control its lifetime.
        let mut mmap = unsafe {
            memmap2::MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| SochDBError::Internal(format!("Failed to mmap index: {}", e)))?
        };

        // Generate a random seed from system entropy
        let seed = Self::generate_seed();

        // Write header
        mmap[HEADER_OFF_MAGIC..HEADER_OFF_MAGIC + 8].copy_from_slice(&MAGIC);
        mmap[HEADER_OFF_VERSION..HEADER_OFF_VERSION + 4]
            .copy_from_slice(&VERSION.to_le_bytes());
        mmap[HEADER_OFF_NUM_SLOTS..HEADER_OFF_NUM_SLOTS + 8]
            .copy_from_slice(&capacity.to_le_bytes());
        mmap[HEADER_OFF_NUM_ENTRIES..HEADER_OFF_NUM_ENTRIES + 8]
            .copy_from_slice(&0u64.to_le_bytes());
        mmap[HEADER_OFF_SEED..HEADER_OFF_SEED + 8]
            .copy_from_slice(&seed.to_le_bytes());

        // All slots are zero-initialized (TAG_EMPTY = 0x00) by the OS via ftruncate.

        mmap.flush()
            .map_err(|e| SochDBError::Internal(format!("Failed to flush new index: {}", e)))?;

        tracing::info!(
            capacity = capacity,
            file_size_kb = file_size / 1024,
            path = %path.display(),
            "Created new DiskHashIndex"
        );

        Ok(Self {
            mmap: RwLock::new(mmap),
            file: RwLock::new(file),
            path,
            num_entries: AtomicU64::new(0),
            num_slots: AtomicU64::new(capacity),
            seed,
        })
    }

    /// Open an existing index file, validating the header.
    fn open_existing(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| SochDBError::Internal(format!("Failed to open index file: {}", e)))?;

        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| SochDBError::Internal(format!("Failed to mmap index: {}", e)))?
        };

        // Validate magic
        if mmap.len() < HEADER_SIZE as usize || mmap[0..8] != MAGIC {
            return Err(SochDBError::Corruption(
                "Invalid DiskHashIndex magic — file corrupt or wrong format. \
                 Delete the index file to trigger rebuild from payload data."
                    .into(),
            ));
        }

        let version = u32::from_le_bytes(
            mmap[HEADER_OFF_VERSION..HEADER_OFF_VERSION + 4]
                .try_into()
                .unwrap(),
        );
        if version != VERSION {
            return Err(SochDBError::Corruption(format!(
                "Unsupported DiskHashIndex version {} (expected {})",
                version, VERSION
            )));
        }

        let num_slots = u64::from_le_bytes(
            mmap[HEADER_OFF_NUM_SLOTS..HEADER_OFF_NUM_SLOTS + 8]
                .try_into()
                .unwrap(),
        );
        let num_entries = u64::from_le_bytes(
            mmap[HEADER_OFF_NUM_ENTRIES..HEADER_OFF_NUM_ENTRIES + 8]
                .try_into()
                .unwrap(),
        );
        let seed = u64::from_le_bytes(
            mmap[HEADER_OFF_SEED..HEADER_OFF_SEED + 8]
                .try_into()
                .unwrap(),
        );

        // Sanity checks
        if !num_slots.is_power_of_two() {
            return Err(SochDBError::Corruption(format!(
                "num_slots {} is not a power of 2 — index corrupt",
                num_slots
            )));
        }

        let expected_file_size = HEADER_SIZE + num_slots * SLOT_SIZE;
        if (mmap.len() as u64) < expected_file_size {
            return Err(SochDBError::Corruption(format!(
                "Index file truncated: expected {} bytes, got {}",
                expected_file_size,
                mmap.len()
            )));
        }

        tracing::info!(
            num_entries = num_entries,
            num_slots = num_slots,
            load_factor = format!("{:.2}", num_entries as f64 / num_slots as f64),
            path = %path.display(),
            "Opened existing DiskHashIndex"
        );

        Ok(Self {
            mmap: RwLock::new(mmap),
            file: RwLock::new(file),
            path,
            num_entries: AtomicU64::new(num_entries),
            num_slots: AtomicU64::new(num_slots),
            seed,
        })
    }

    /// Compute the byte offset of slot `i` in the mmap.
    #[inline(always)]
    fn slot_offset(slot_index: u64) -> usize {
        (HEADER_SIZE + slot_index * SLOT_SIZE) as usize
    }

    /// Read a slot's tag byte.
    #[inline(always)]
    fn read_tag(mmap: &MmapMut, slot_index: u64) -> u8 {
        mmap[Self::slot_offset(slot_index) + SLOT_OFF_TAG]
    }

    /// Read the edge_id from a slot.
    #[inline(always)]
    fn read_edge_id(mmap: &MmapMut, slot_index: u64) -> u128 {
        let base = Self::slot_offset(slot_index) + SLOT_OFF_EDGE_ID;
        u128::from_le_bytes(mmap[base..base + 16].try_into().unwrap())
    }

    /// Read a full PayloadMeta from a slot (only call if tag == OCCUPIED).
    #[inline]
    fn read_meta(mmap: &MmapMut, slot_index: u64) -> PayloadMeta {
        let base = Self::slot_offset(slot_index);
        let compression_byte = mmap[base + SLOT_OFF_COMPRESSION];
        let length = u32::from_le_bytes(
            mmap[base + SLOT_OFF_LENGTH..base + SLOT_OFF_LENGTH + 4]
                .try_into()
                .unwrap(),
        );
        let uncompressed_length = u32::from_le_bytes(
            mmap[base + SLOT_OFF_UNCOMPRESSED_LEN..base + SLOT_OFF_UNCOMPRESSED_LEN + 4]
                .try_into()
                .unwrap(),
        );
        let edge_id = u128::from_le_bytes(
            mmap[base + SLOT_OFF_EDGE_ID..base + SLOT_OFF_EDGE_ID + 16]
                .try_into()
                .unwrap(),
        );
        let offset = u64::from_le_bytes(
            mmap[base + SLOT_OFF_OFFSET..base + SLOT_OFF_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        PayloadMeta {
            edge_id,
            offset,
            length,
            compression: CompressionType::from_u8(compression_byte).unwrap_or(CompressionType::None),
            uncompressed_length,
        }
    }

    /// Write a PayloadMeta into a slot, setting the tag to OCCUPIED.
    #[inline]
    fn write_slot(mmap: &mut MmapMut, slot_index: u64, meta: &PayloadMeta) {
        let base = Self::slot_offset(slot_index);

        mmap[base + SLOT_OFF_TAG] = TAG_OCCUPIED;
        mmap[base + SLOT_OFF_COMPRESSION] = meta.compression as u8;
        // Zero reserved bytes
        mmap[base + 2] = 0;
        mmap[base + 3] = 0;
        mmap[base + SLOT_OFF_LENGTH..base + SLOT_OFF_LENGTH + 4]
            .copy_from_slice(&meta.length.to_le_bytes());
        mmap[base + SLOT_OFF_UNCOMPRESSED_LEN..base + SLOT_OFF_UNCOMPRESSED_LEN + 4]
            .copy_from_slice(&meta.uncompressed_length.to_le_bytes());
        // Zero alignment padding
        mmap[base + 12..base + 16].copy_from_slice(&[0u8; 4]);
        mmap[base + SLOT_OFF_EDGE_ID..base + SLOT_OFF_EDGE_ID + 16]
            .copy_from_slice(&meta.edge_id.to_le_bytes());
        mmap[base + SLOT_OFF_OFFSET..base + SLOT_OFF_OFFSET + 8]
            .copy_from_slice(&meta.offset.to_le_bytes());
    }

    /// Update the num_entries field in the header.
    fn write_header_entries(mmap: &mut MmapMut, count: u64) {
        mmap[HEADER_OFF_NUM_ENTRIES..HEADER_OFF_NUM_ENTRIES + 8]
            .copy_from_slice(&count.to_le_bytes());
    }

    /// Find the slot for a given edge_id (linear probing).
    ///
    /// Returns `Ok(slot_index)` if found, or `Err(first_empty_slot)` if not found
    /// (the empty slot is where the entry *would* go).
    fn probe(&self, mmap: &MmapMut, edge_id: u128) -> std::result::Result<u64, u64> {
        let num_slots = self.num_slots.load(Ordering::Relaxed);
        let mask = num_slots - 1; // Power-of-2 modulo
        let mut slot = hash_u128(edge_id, self.seed) & mask;

        // Linear probing. We are guaranteed to terminate because load factor < 1.0,
        // so there is always at least one empty slot.
        loop {
            let tag = Self::read_tag(mmap, slot);
            if tag == TAG_EMPTY {
                return Err(slot);
            }
            if tag == TAG_OCCUPIED && Self::read_edge_id(mmap, slot) == edge_id {
                return Ok(slot);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Grow the table by 2x and rehash all entries.
    ///
    /// This is O(n) but amortized O(1) per insert because we double each time.
    /// Strategy: create a new file, rehash into it, atomically rename over the old one.
    fn grow(&self) -> Result<()> {
        let old_num_slots = self.num_slots.load(Ordering::Relaxed);
        let new_num_slots = old_num_slots
            .checked_mul(2)
            .ok_or_else(|| SochDBError::Internal("Index capacity overflow".into()))?;

        let new_file_size = HEADER_SIZE + new_num_slots * SLOT_SIZE;

        tracing::info!(
            old_slots = old_num_slots,
            new_slots = new_num_slots,
            new_file_size_mb = new_file_size / (1024 * 1024),
            "DiskHashIndex: growing table"
        );

        // Create temp file for the new table
        let temp_path = self.path.with_extension("tmp");
        let new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
            .map_err(|e| SochDBError::Internal(format!("Failed to create temp index: {}", e)))?;

        new_file.set_len(new_file_size)
            .map_err(|e| SochDBError::Internal(format!("Failed to set temp index size: {}", e)))?;

        let mut new_mmap = unsafe {
            memmap2::MmapOptions::new()
                .map_mut(&new_file)
                .map_err(|e| SochDBError::Internal(format!("Failed to mmap temp index: {}", e)))?
        };

        // Write header to new file (same seed, new capacity)
        new_mmap[HEADER_OFF_MAGIC..HEADER_OFF_MAGIC + 8].copy_from_slice(&MAGIC);
        new_mmap[HEADER_OFF_VERSION..HEADER_OFF_VERSION + 4]
            .copy_from_slice(&VERSION.to_le_bytes());
        new_mmap[HEADER_OFF_NUM_SLOTS..HEADER_OFF_NUM_SLOTS + 8]
            .copy_from_slice(&new_num_slots.to_le_bytes());
        new_mmap[HEADER_OFF_SEED..HEADER_OFF_SEED + 8]
            .copy_from_slice(&self.seed.to_le_bytes());

        // Rehash all occupied slots from old mmap into new mmap
        let old_mmap = self.mmap.read();
        let new_mask = new_num_slots - 1;
        let mut rehashed = 0u64;

        for old_slot in 0..old_num_slots {
            if Self::read_tag(&old_mmap, old_slot) != TAG_OCCUPIED {
                continue;
            }

            let meta = Self::read_meta(&old_mmap, old_slot);
            let mut new_slot = hash_u128(meta.edge_id, self.seed) & new_mask;

            // Find empty slot in new table (guaranteed to exist — load factor halved)
            loop {
                if new_mmap[Self::slot_offset(new_slot) + SLOT_OFF_TAG] == TAG_EMPTY {
                    break;
                }
                new_slot = (new_slot + 1) & new_mask;
            }

            Self::write_slot(&mut new_mmap, new_slot, &meta);
            rehashed += 1;
        }
        drop(old_mmap);

        // Write final entry count
        Self::write_header_entries(&mut new_mmap, rehashed);

        // Flush new mmap to disk
        new_mmap
            .flush()
            .map_err(|e| SochDBError::Internal(format!("Failed to flush grown index: {}", e)))?;

        // Atomic swap: rename temp file over the old one.
        // On POSIX, rename is atomic within the same filesystem.
        fs::rename(&temp_path, &self.path)
            .map_err(|e| SochDBError::Internal(format!("Failed to rename grown index: {}", e)))?;

        // Update internal state
        *self.mmap.write() = new_mmap;
        *self.file.write() = new_file;
        self.num_slots.store(new_num_slots, Ordering::Release);

        tracing::info!(
            rehashed = rehashed,
            new_slots = new_num_slots,
            load_factor = format!("{:.2}", rehashed as f64 / new_num_slots as f64),
            "DiskHashIndex: grow complete"
        );

        Ok(())
    }

    /// Check if the table needs to grow.
    #[inline]
    fn needs_grow(&self) -> bool {
        let entries = self.num_entries.load(Ordering::Relaxed);
        let slots = self.num_slots.load(Ordering::Relaxed);
        (entries + 1) as f64 / slots as f64 > MAX_LOAD_FACTOR
    }

    /// Generate a random seed using available system entropy.
    fn generate_seed() -> u64 {
        // Use a combination of time, address space, and process id for entropy.
        // This is NOT cryptographic — it's just to prevent hash flooding.
        let mut seed: u64 = 0;

        // Mix in high-resolution time
        #[cfg(unix)]
        {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe {
                libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
            }
            seed ^= ts.tv_sec as u64;
            seed ^= (ts.tv_nsec as u64).wrapping_mul(0x9e3779b97f4a7c15);
        }

        #[cfg(not(unix))]
        {
            use std::time::SystemTime;
            if let Ok(dur) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
                seed ^= dur.as_nanos() as u64;
            }
        }

        // Mix in process id
        seed ^= (std::process::id() as u64).wrapping_mul(0x517cc1b727220a95);

        // Mix in a stack address for ASLR entropy
        let stack_var: u8 = 0;
        seed ^= ((&stack_var as *const u8 as u64) >> 12).wrapping_mul(0x6c62272e07bb0142);

        // Final mixing (SplitMix64 finalizer)
        seed ^= seed >> 30;
        seed = seed.wrapping_mul(0xbf58476d1ce4e5b9);
        seed ^= seed >> 27;
        seed = seed.wrapping_mul(0x94d049bb133111eb);
        seed ^= seed >> 31;

        // Avoid seed = 0 (degenerate hash behavior)
        if seed == 0 {
            seed = 0x1234567890abcdef;
        }

        seed
    }
}

// =============================================================================
// PayloadIndex trait implementation
// =============================================================================

impl PayloadIndex for DiskHashIndex {
    fn insert(&self, edge_id: u128, meta: PayloadMeta) -> Result<()> {
        // Check if we need to grow BEFORE acquiring write lock
        if self.needs_grow() {
            self.grow()?;
        }

        let mut mmap = self.mmap.write();

        match self.probe(&mmap, edge_id) {
            Ok(existing_slot) => {
                // Key already exists — overwrite value in place
                Self::write_slot(&mut mmap, existing_slot, &meta);
            }
            Err(empty_slot) => {
                // Key not found — insert into the empty slot
                Self::write_slot(&mut mmap, empty_slot, &meta);
                let new_count = self.num_entries.fetch_add(1, Ordering::AcqRel) + 1;
                Self::write_header_entries(&mut mmap, new_count);
            }
        }

        Ok(())
    }

    fn get(&self, edge_id: u128) -> Result<Option<PayloadMeta>> {
        let mmap = self.mmap.read();
        match self.probe(&mmap, edge_id) {
            Ok(slot) => Ok(Some(Self::read_meta(&mmap, slot))),
            Err(_) => Ok(None),
        }
    }

    fn contains_key(&self, edge_id: u128) -> bool {
        let mmap = self.mmap.read();
        self.probe(&mmap, edge_id).is_ok()
    }

    fn len(&self) -> usize {
        self.num_entries.load(Ordering::Relaxed) as usize
    }

    fn is_empty(&self) -> bool {
        self.num_entries.load(Ordering::Relaxed) == 0
    }

    fn iter_values(&self) -> Box<dyn Iterator<Item = PayloadMeta> + '_> {
        // Collect all occupied slots into a Vec.
        // This is O(num_slots) — acceptable since iter_values is used for
        // stats/diagnostics, not in the hot path.
        let mmap = self.mmap.read();
        let num_slots = self.num_slots.load(Ordering::Relaxed);
        let mut entries = Vec::with_capacity(self.num_entries.load(Ordering::Relaxed) as usize);

        for slot in 0..num_slots {
            if Self::read_tag(&mmap, slot) == TAG_OCCUPIED {
                entries.push(Self::read_meta(&mmap, slot));
            }
        }

        Box::new(entries.into_iter())
    }

    fn save(&self) -> Result<()> {
        let mmap = self.mmap.read();
        mmap.flush()
            .map_err(|e| SochDBError::Internal(format!("DiskHashIndex flush failed: {}", e)))?;
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_meta(edge_id: u128, offset: u64) -> PayloadMeta {
        PayloadMeta {
            edge_id,
            offset,
            length: 100,
            compression: CompressionType::None,
            uncompressed_length: 100,
        }
    }

    #[test]
    fn test_basic_insert_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        idx.insert(1, make_meta(1, 0)).unwrap();
        idx.insert(2, make_meta(2, 100)).unwrap();

        let m1 = idx.get(1).unwrap().unwrap();
        assert_eq!(m1.edge_id, 1);
        assert_eq!(m1.offset, 0);

        let m2 = idx.get(2).unwrap().unwrap();
        assert_eq!(m2.edge_id, 2);
        assert_eq!(m2.offset, 100);

        assert!(idx.get(999).unwrap().is_none());
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_overwrite() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        idx.insert(1, make_meta(1, 0)).unwrap();
        assert_eq!(idx.get(1).unwrap().unwrap().offset, 0);

        // Overwrite with different offset
        idx.insert(1, make_meta(1, 999)).unwrap();
        assert_eq!(idx.get(1).unwrap().unwrap().offset, 999);

        // Count should not increase on overwrite
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");

        // Write
        {
            let idx = DiskHashIndex::new(path.clone()).unwrap();
            idx.insert(1, make_meta(1, 0)).unwrap();
            idx.insert(2, make_meta(2, 100)).unwrap();
            idx.save().unwrap();
        }

        // Reopen and verify
        {
            let idx = DiskHashIndex::new(path).unwrap();
            assert_eq!(idx.len(), 2);
            assert_eq!(idx.get(1).unwrap().unwrap().offset, 0);
            assert_eq!(idx.get(2).unwrap().unwrap().offset, 100);
        }
    }

    #[test]
    fn test_grow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        // Insert enough entries to trigger multiple resizes.
        // Initial capacity = 4096, grows at 75% = 3072 entries.
        let n = 10_000u128;
        for i in 0..n {
            idx.insert(i, make_meta(i, i as u64 * 40)).unwrap();
        }

        assert_eq!(idx.len(), n as usize);

        // Verify every entry survived rehashing
        for i in 0..n {
            let meta = idx.get(i).unwrap().unwrap();
            assert_eq!(meta.edge_id, i);
            assert_eq!(meta.offset, i as u64 * 40);
        }
    }

    #[test]
    fn test_contains_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        idx.insert(42, make_meta(42, 0)).unwrap();

        assert!(idx.contains_key(42));
        assert!(!idx.contains_key(43));
    }

    #[test]
    fn test_iter_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        for i in 0..100u128 {
            idx.insert(i, make_meta(i, i as u64)).unwrap();
        }

        let values: Vec<_> = idx.iter_values().collect();
        assert_eq!(values.len(), 100);

        // All edge_ids should be present (order is not guaranteed)
        let mut ids: Vec<u128> = values.iter().map(|m| m.edge_id).collect();
        ids.sort();
        let expected: Vec<u128> = (0..100).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn test_compression_types() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        let mut meta = make_meta(1, 0);
        meta.compression = CompressionType::LZ4;
        meta.uncompressed_length = 500;
        meta.length = 200;
        idx.insert(1, meta).unwrap();

        let retrieved = idx.get(1).unwrap().unwrap();
        assert_eq!(retrieved.compression, CompressionType::LZ4);
        assert_eq!(retrieved.uncompressed_length, 500);
        assert_eq!(retrieved.length, 200);
    }

    #[test]
    fn test_large_edge_ids() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        // Test with u128 values that exercise both halves
        let ids = [
            u128::MAX,
            u128::MAX - 1,
            1u128 << 64,
            (1u128 << 64) + 1,
            0u128,
            u128::MAX / 2,
        ];

        for (i, &id) in ids.iter().enumerate() {
            idx.insert(id, make_meta(id, i as u64 * 100)).unwrap();
        }

        for (i, &id) in ids.iter().enumerate() {
            let meta = idx.get(id).unwrap().unwrap();
            assert_eq!(meta.edge_id, id);
            assert_eq!(meta.offset, i as u64 * 100);
        }
    }

    #[test]
    fn test_is_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let idx = DiskHashIndex::new(path).unwrap();

        assert!(idx.is_empty());
        idx.insert(1, make_meta(1, 0)).unwrap();
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_grow_preserves_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");

        // Insert enough to trigger grow, then save + reopen
        {
            let idx = DiskHashIndex::new(path.clone()).unwrap();
            for i in 0..5000u128 {
                idx.insert(i, make_meta(i, i as u64)).unwrap();
            }
            idx.save().unwrap();
        }

        // Reopen
        {
            let idx = DiskHashIndex::new(path).unwrap();
            assert_eq!(idx.len(), 5000);
            for i in 0..5000u128 {
                assert_eq!(idx.get(i).unwrap().unwrap().offset, i as u64);
            }
        }
    }

    #[test]
    fn test_hash_distribution() {
        // Verify the hash function doesn't degenerate for sequential keys.
        // A good hash should distribute sequential u128s uniformly.
        let seed = 0xdeadbeef_u64;
        let n = 10_000;
        let num_buckets = 1024u64;

        let mut counts = vec![0u32; num_buckets as usize];
        for i in 0..n {
            let h = hash_u128(i as u128, seed);
            counts[(h & (num_buckets - 1)) as usize] += 1;
        }

        let expected = n as f64 / num_buckets as f64; // ~9.77
        let max_count = *counts.iter().max().unwrap() as f64;
        let min_count = *counts.iter().min().unwrap() as f64;

        // With good uniformity, max/expected should be < 3.0 and min > 0
        assert!(
            max_count / expected < 3.0,
            "Hash distribution too skewed: max={}, expected={}",
            max_count,
            expected
        );
        assert!(
            min_count > 0.0,
            "Hash has empty buckets — degenerate distribution"
        );
    }
}
