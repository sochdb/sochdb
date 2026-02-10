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

//! SSTable Reader
//!
//! This module provides an SSTable reader with:
//! - Memory-mapped I/O for efficient access
//! - Lazy block loading
//! - Block cache integration
//! - Binary search in index for O(log n) lookups
//! - Filter-based negative lookup optimization

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;

use super::block::{Block, BlockHandle, BlockIterator, BlockType};
use super::filter::FilterReader;
use super::format::{Footer, Header, Section, SectionType, SSTableFormat, HEADER_SIZE};

/// Block cache entry
pub struct CachedBlock {
    /// Raw block data
    pub data: Vec<u8>,
    /// Block type (compression)
    pub block_type: BlockType,
    /// Decompressed data (if applicable)
    pub decompressed: Vec<u8>,
}

/// Simple block cache (HashMap-based for simplicity)
pub struct BlockCache {
    /// Cache entries by (file_id, block_offset)
    entries: RwLock<HashMap<(u64, u64), Arc<CachedBlock>>>,
    /// Maximum capacity
    capacity: usize,
}

impl BlockCache {
    /// Create a new block cache
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
        }
    }

    /// Get a cached block
    pub fn get(&self, file_id: u64, offset: u64) -> Option<Arc<CachedBlock>> {
        self.entries.read().get(&(file_id, offset)).cloned()
    }

    /// Insert a block into cache
    pub fn insert(&self, file_id: u64, offset: u64, block: CachedBlock) -> Arc<CachedBlock> {
        let block = Arc::new(block);
        let mut entries = self.entries.write();
        
        // Simple eviction: clear when full
        if entries.len() >= self.capacity {
            entries.clear();
        }
        
        entries.insert((file_id, offset), block.clone());
        block
    }
}

/// Read options
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Verify checksums when reading blocks
    pub verify_checksums: bool,
    /// Fill block cache
    pub fill_cache: bool,
    /// Use filter to skip blocks
    pub use_filter: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            verify_checksums: true,
            fill_cache: true,
            use_filter: true,
        }
    }
}

/// SSTable reader for reading SSTable files
pub struct SSTable {
    /// File path
    path: PathBuf,
    /// Unique file ID for caching
    file_id: u64,
    /// Memory-mapped file
    mmap: Mmap,
    /// Parsed header
    header: Header,
    /// Parsed footer with sections
    footer: Footer,
    /// Index block (cached)
    index: Vec<u8>,
    /// Parsed index entries
    index_entries: Vec<IndexEntry>,
    /// Filter reader (if filter section exists)
    filter: Option<FilterReader>,
    /// File metadata
    metadata: TableMetadata,
    /// Block cache reference
    cache: Option<Arc<BlockCache>>,
}

/// Index entry
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Largest key in this block (separator)
    largest_key: Vec<u8>,
    /// Block handle
    handle: BlockHandle,
}

/// Table metadata
#[derive(Debug, Clone)]
pub struct TableMetadata {
    /// File size
    pub file_size: u64,
    /// Number of data blocks
    pub num_data_blocks: usize,
    /// Smallest key
    pub smallest_key: Option<Vec<u8>>,
    /// Largest key
    pub largest_key: Option<Vec<u8>>,
}

impl SSTable {
    /// Open an SSTable file
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Self::open_with_cache(path, None)
    }

    /// Open an SSTable file with a block cache
    pub fn open_with_cache<P: AsRef<Path>>(
        path: P,
        cache: Option<Arc<BlockCache>>,
    ) -> std::io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // Memory-map the file
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Generate file ID from path hash
        let file_id = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            path.hash(&mut hasher);
            hasher.finish()
        };

        // Parse header
        if mmap.len() < HEADER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small for SSTable header",
            ));
        }

        let header = Header::decode(&mmap[..HEADER_SIZE]).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid SSTable header")
        })?;

        // Parse footer
        let footer_offset = header.footer_offset as usize;
        if footer_offset >= mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Footer offset beyond file",
            ));
        }

        let footer = Footer::decode(&mmap[footer_offset..], header.num_sections).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid SSTable footer")
        })?;

        // Load index section
        let index_section = footer
            .sections
            .iter()
            .find(|s| s.section_type == SectionType::Index)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing index section")
            })?;

        let index_start = index_section.offset as usize;
        let index_end = index_start + index_section.size as usize;
        let index = mmap[index_start..index_end].to_vec();

        // Parse index entries
        let index_entries = Self::parse_index(&index)?;

        // Load filter section if present
        let filter = footer
            .sections
            .iter()
            .find(|s| s.section_type == SectionType::Filter)
            .and_then(|section| {
                let start = section.offset as usize;
                let end = start + section.size as usize;
                FilterReader::from_bytes(&mmap[start..end])
            });

        // Extract metadata
        let metadata = TableMetadata {
            file_size,
            num_data_blocks: index_entries.len(),
            smallest_key: index_entries.first().map(|e| e.largest_key.clone()),
            largest_key: index_entries.last().map(|e| e.largest_key.clone()),
        };

        Ok(Self {
            path: path.to_path_buf(),
            file_id,
            mmap,
            header,
            footer,
            index,
            index_entries,
            filter,
            metadata,
            cache,
        })
    }

    /// Parse index entries from index block data
    fn parse_index(data: &[u8]) -> std::io::Result<Vec<IndexEntry>> {
        let mut entries = Vec::new();
        let block = Block::new(data.to_vec()).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid index block")
        })?;
        let mut iter = block.iter();

        while iter.valid() {
            let key = iter.key().to_vec();
            let value = iter.value();

            let (handle, _bytes_read) = BlockHandle::decode(value).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid block handle")
            })?;

            entries.push(IndexEntry {
                largest_key: key,
                handle,
            });

            iter.next();
        }

        Ok(entries)
    }

    /// Get a value by key
    pub fn get(&self, key: &[u8], options: &ReadOptions) -> std::io::Result<Option<Vec<u8>>> {
        // Use filter to check if key might exist
        if options.use_filter {
            if let Some(ref filter) = self.filter {
                if !filter.may_contain(key) {
                    return Ok(None);
                }
            }
        }

        // Binary search in index to find the right block
        let block_idx = self.find_block_for_key(key);
        if block_idx >= self.index_entries.len() {
            return Ok(None);
        }

        // Load and search the block
        let block_data = self.read_block(&self.index_entries[block_idx].handle, options)?;
        let block = Block::new(block_data).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid data block")
        })?;

        let iter = block.seek(key);
        if iter.valid() && iter.key() == key {
            Ok(Some(iter.value().to_vec()))
        } else {
            Ok(None)
        }
    }

    /// Binary search to find block that might contain the key
    fn find_block_for_key(&self, key: &[u8]) -> usize {
        // Binary search for first block where largest_key >= key
        self.index_entries
            .binary_search_by(|entry| {
                if entry.largest_key.as_slice() < key {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|i| i)
    }

    /// Read a block from file
    fn read_block(
        &self,
        handle: &BlockHandle,
        options: &ReadOptions,
    ) -> std::io::Result<Vec<u8>> {
        let offset = handle.offset();
        let size = handle.size();

        // Try cache first
        if let Some(ref cache) = self.cache {
            if let Some(block) = cache.get(self.file_id, offset) {
                return Ok(block.decompressed.clone());
            }
        }

        // Read from mmap
        let start = offset as usize;
        let end = start + size as usize;

        if end + 5 > self.mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Block extends beyond file",
            ));
        }

        let block_data = &self.mmap[start..end];
        let block_type = BlockType::from_u8(self.mmap[end]);
        let stored_checksum = u32::from_le_bytes([
            self.mmap[end + 1],
            self.mmap[end + 2],
            self.mmap[end + 3],
            self.mmap[end + 4],
        ]);

        // Verify checksum if requested
        if options.verify_checksums {
            let computed_checksum = crc32fast::hash(block_data);
            if computed_checksum != stored_checksum {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Block checksum mismatch",
                ));
            }
        }

        // Decompress if needed
        let decompressed = match block_type {
            BlockType::Uncompressed => block_data.to_vec(),
            BlockType::Lz4 => lz4_flex::decompress_size_prepended(block_data).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("LZ4 error: {}", e))
            })?,
            BlockType::Zstd => zstd::decode_all(block_data).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Zstd error: {}", e))
            })?,
            BlockType::Snappy => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Snappy not supported",
                ))
            }
        };

        // Cache the block
        if options.fill_cache {
            if let Some(ref cache) = self.cache {
                cache.insert(
                    self.file_id,
                    offset,
                    CachedBlock {
                        data: block_data.to_vec(),
                        block_type,
                        decompressed: decompressed.clone(),
                    },
                );
            }
        }

        Ok(decompressed)
    }

    /// Create an iterator over all entries
    pub fn iter(&self) -> SSTableIterator {
        SSTableIterator::new(self)
    }

    /// Create a range iterator
    pub fn range(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> RangeIterator {
        RangeIterator::new(self, start, end)
    }

    /// Get table metadata
    pub fn metadata(&self) -> &TableMetadata {
        &self.metadata
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get number of data blocks
    pub fn num_blocks(&self) -> usize {
        self.index_entries.len()
    }

    /// Check if key might exist (using filter)
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.filter
            .as_ref()
            .map(|f| f.may_contain(key))
            .unwrap_or(true)
    }
}

/// Iterator over all entries in an SSTable
///
/// Iterates through all data blocks sequentially, yielding every key-value
/// entry. Loads each block, uses the proven `BlockIterator` to collect entries, 
/// then advances through them. This avoids self-referential borrows while 
/// reusing the correct prefix-decompression logic in `BlockIterator`.
pub struct SSTableIterator<'a> {
    table: &'a SSTable,
    /// Current block index into `table.index_entries`
    block_idx: usize,
    /// Collected entries from current block: (key, value)
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    /// Current entry index within `entries`
    entry_idx: usize,
    /// Read options
    options: ReadOptions,
    /// Is iterator positioned on a valid entry?
    valid: bool,
}

impl<'a> SSTableIterator<'a> {
    fn new(table: &'a SSTable) -> Self {
        let mut iter = Self {
            table,
            block_idx: 0,
            entries: Vec::new(),
            entry_idx: 0,
            options: ReadOptions::default(),
            valid: false,
        };
        iter.load_block(0);
        iter
    }

    /// Load block at `block_idx`, collect all entries via `BlockIterator`,
    /// and position on the first entry.
    fn load_block(&mut self, block_idx: usize) {
        self.block_idx = block_idx;
        self.entries.clear();
        self.entry_idx = 0;
        self.valid = false;

        while self.block_idx < self.table.index_entries.len() {
            let handle = &self.table.index_entries[self.block_idx].handle;
            match self.table.read_block(handle, &self.options) {
                Ok(data) => {
                    if let Some(block) = Block::new(data) {
                        // Use BlockIterator to collect all entries
                        let mut bi = block.iter();
                        while bi.valid() {
                            self.entries.push((
                                bi.key().to_vec(),
                                bi.value().to_vec(),
                            ));
                            bi.next();
                        }
                        if !self.entries.is_empty() {
                            self.entry_idx = 0;
                            self.valid = true;
                            return;
                        }
                        // Block had no entries — try next
                    }
                }
                Err(_) => {
                    // I/O error — skip this block
                }
            }
            self.block_idx += 1;
        }
    }

    /// Check if iterator is valid
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Get current key (only valid when `valid() == true`)
    pub fn key(&self) -> Option<&[u8]> {
        if self.valid {
            Some(&self.entries[self.entry_idx].0)
        } else {
            None
        }
    }

    /// Get current value (only valid when `valid() == true`)
    pub fn value(&self) -> Option<&[u8]> {
        if self.valid {
            Some(&self.entries[self.entry_idx].1)
        } else {
            None
        }
    }

    /// Advance to the next entry. If the current block is exhausted,
    /// loads the next block and positions on its first entry.
    pub fn next(&mut self) {
        if !self.valid {
            return;
        }

        self.entry_idx += 1;
        if self.entry_idx < self.entries.len() {
            return; // still within current block
        }

        // Current block exhausted — move to next block
        self.load_block(self.block_idx + 1);
    }

    /// Seek to the first entry with key >= `target`.
    pub fn seek(&mut self, target: &[u8]) {
        // Binary search to find starting block
        let start_block = self.table.find_block_for_key(target);
        self.load_block(start_block);

        // Scan forward until key >= target
        while self.valid {
            if self.entries[self.entry_idx].0.as_slice() >= target {
                return;
            }
            self.next();
        }
    }

    /// Seek to the very first entry in the SSTable.
    pub fn seek_to_first(&mut self) {
        self.load_block(0);
    }
}

/// Range iterator over entries in [start, end) of an SSTable.
///
/// Uses `SSTableIterator` internally, adding upper-bound checking.
pub struct RangeIterator<'a> {
    inner: SSTableIterator<'a>,
    end: Option<Vec<u8>>,
    exhausted: bool,
}

impl<'a> RangeIterator<'a> {
    fn new(table: &'a SSTable, start: Option<&[u8]>, end: Option<&[u8]>) -> Self {
        let mut inner = SSTableIterator::new(table);

        // Seek to start if provided
        if let Some(start_key) = start {
            inner.seek(start_key);
        }

        let mut ri = Self {
            inner,
            end: end.map(|e| e.to_vec()),
            exhausted: false,
        };

        // Check if first entry is already beyond end
        ri.check_end();
        ri
    }

    /// Check if current key >= end bound; if so, mark exhausted.
    fn check_end(&mut self) {
        if self.exhausted {
            return;
        }
        if !self.inner.valid() {
            self.exhausted = true;
            return;
        }
        if let Some(ref end_key) = self.end {
            if let Some(key) = self.inner.key() {
                if key >= end_key.as_slice() {
                    self.exhausted = true;
                }
            }
        }
    }

    /// Check if range is exhausted
    pub fn exhausted(&self) -> bool {
        self.exhausted
    }

    /// Check if iterator is valid (positioned on an entry within bounds)
    pub fn valid(&self) -> bool {
        !self.exhausted && self.inner.valid()
    }

    /// Get current key
    pub fn key(&self) -> Option<&[u8]> {
        if self.exhausted { None } else { self.inner.key() }
    }

    /// Get current value
    pub fn value(&self) -> Option<&[u8]> {
        if self.exhausted { None } else { self.inner.value() }
    }

    /// Advance to next entry within the range
    pub fn next(&mut self) {
        if self.exhausted {
            return;
        }
        self.inner.next();
        self.check_end();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::builder::{SSTableBuilder, SSTableBuilderOptions};
    use tempfile::tempdir;

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        // Build SSTable
        let options = SSTableBuilderOptions {
            block_size: 256,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in 0..100 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        // Read SSTable
        let table = SSTable::open(&path).unwrap();

        assert_eq!(table.num_blocks(), table.metadata.num_data_blocks);
    }

    #[test]
    fn test_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_get.sst");

        let options = SSTableBuilderOptions {
            block_size: 256,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in 0..100 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        let table = SSTable::open(&path).unwrap();
        let read_opts = ReadOptions::default();

        // Test existing key
        let result = table.get(b"key00050", &read_opts).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), b"value00050");

        // Test non-existing key
        let result = table.get(b"nonexistent", &read_opts).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_block_cache() {
        let cache = BlockCache::new(100);

        let block = CachedBlock {
            data: vec![1, 2, 3],
            block_type: BlockType::Uncompressed,
            decompressed: vec![1, 2, 3],
        };

        cache.insert(1, 0, block);

        let cached = cache.get(1, 0);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().data, vec![1, 2, 3]);

        let missing = cache.get(1, 100);
        assert!(missing.is_none());
    }

    #[test]
    fn test_sstable_iterator_full_scan() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_iter.sst");

        // Use small blocks to force multiple blocks
        let options = SSTableBuilderOptions {
            block_size: 64,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        let n = 200;
        for i in 0..n {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        let table = SSTable::open(&path).unwrap();
        eprintln!("num_blocks = {}", table.num_blocks());
        assert!(table.num_blocks() > 1, "Need multiple blocks for this test");

        // Full iteration should return all entries in order
        let mut iter = table.iter();
        let mut count = 0;
        let mut prev_key: Option<Vec<u8>> = None;

        while iter.valid() {
            let key = iter.key().unwrap().to_vec();
            let value = iter.value().unwrap().to_vec();

            let expected_key = format!("key{:05}", count);
            let expected_val = format!("value{:05}", count);
            assert_eq!(
                String::from_utf8_lossy(&key),
                expected_key,
                "key mismatch at entry {}",
                count
            );
            assert_eq!(
                String::from_utf8_lossy(&value),
                expected_val,
                "value mismatch at entry {}",
                count
            );

            // Keys must be strictly increasing
            if let Some(ref pk) = prev_key {
                assert!(key > *pk, "keys not in order at {}", count);
            }
            prev_key = Some(key);

            count += 1;
            iter.next();
        }

        assert_eq!(count, n, "iterator did not return all {} entries", n);
    }

    #[test]
    fn test_sstable_iterator_seek() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_seek.sst");

        let options = SSTableBuilderOptions {
            block_size: 64,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in (0..100).step_by(2) {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        let table = SSTable::open(&path).unwrap();

        // Seek to exact key
        let mut iter = table.iter();
        iter.seek(b"key00010");
        assert!(iter.valid());
        assert_eq!(iter.key().unwrap(), b"key00010");

        // Seek to key between entries (should land on next key)
        iter.seek(b"key00011");
        assert!(iter.valid());
        assert_eq!(iter.key().unwrap(), b"key00012");

        // Seek past end
        iter.seek(b"key99999");
        assert!(!iter.valid());

        // Seek to first
        iter.seek_to_first();
        assert!(iter.valid());
        assert_eq!(iter.key().unwrap(), b"key00000");
    }

    #[test]
    fn test_range_iterator() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_range.sst");

        let options = SSTableBuilderOptions {
            block_size: 64,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in 0..100 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        let table = SSTable::open(&path).unwrap();

        // Range [key00010, key00020)
        let mut range = table.range(Some(b"key00010"), Some(b"key00020"));
        let mut count = 0;

        while range.valid() {
            let key = range.key().unwrap();
            assert!(key >= b"key00010".as_slice());
            assert!(key < b"key00020".as_slice());
            count += 1;
            range.next();
        }

        assert_eq!(count, 10, "expected 10 keys in range [10, 20)");
        assert!(range.exhausted());

        // Full range (no bounds)
        let mut range = table.range(None, None);
        let mut total = 0;
        while range.valid() {
            total += 1;
            range.next();
        }
        assert_eq!(total, 100);
    }
}
