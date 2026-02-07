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

//! # I/O Isolation Policy
//!
//! Implements cache partitioning and I/O isolation to prevent:
//! - Page cache pollution from large scans
//! - p99 cliffs from compaction I/O
//! - Memory fragmentation under pressure
//!
//! ## Design Principles
//!
//! 1. **Workload Classification**: Classify I/O as query, compaction, or backup
//! 2. **Cache Partitioning**: Separate caches for different workloads
//! 3. **Direct I/O Policy**: Use O_DIRECT based on access pattern
//! 4. **Eviction Priority**: Prefer eviction over allocator fragmentation
//!
//! ## Algorithm
//!
//! Cache eviction uses CLOCK-Pro (or segmented LRU) to approximate
//! recency+frequency without LRU's pathological scan sensitivity.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

/// I/O workload type for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoWorkloadType {
    /// User query (point reads, small scans)
    Query,
    /// Background compaction/merge
    Compaction,
    /// Backup/snapshot
    Backup,
    /// WAL writes
    Wal,
    /// Cache warmup/preload
    Warmup,
}

impl IoWorkloadType {
    /// Should this workload use Direct I/O?
    pub fn prefers_direct_io(&self) -> bool {
        match self {
            IoWorkloadType::Query => false,        // Benefit from cache
            IoWorkloadType::Compaction => true,    // One-time sequential
            IoWorkloadType::Backup => true,        // One-time sequential
            IoWorkloadType::Wal => false,          // Small writes, buffered
            IoWorkloadType::Warmup => false,       // Explicitly filling cache
        }
    }

    /// Cache partition weight (higher = more cache share)
    pub fn cache_weight(&self) -> u32 {
        match self {
            IoWorkloadType::Query => 80,       // Most cache to queries
            IoWorkloadType::Compaction => 10,  // Minimal cache
            IoWorkloadType::Backup => 0,       // No cache
            IoWorkloadType::Wal => 5,          // Small buffer
            IoWorkloadType::Warmup => 5,       // Uses query partition
        }
    }
}

/// I/O access pattern for policy decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Random point reads
    RandomRead,
    /// Sequential scan
    SequentialScan,
    /// Random writes
    RandomWrite,
    /// Sequential writes (WAL, compaction output)
    SequentialWrite,
    /// Mixed pattern
    Mixed,
}

impl AccessPattern {
    /// Estimate if this access will benefit from page cache
    pub fn cache_benefit_probability(&self) -> f64 {
        match self {
            AccessPattern::RandomRead => 0.8,     // Likely reused
            AccessPattern::SequentialScan => 0.2, // Low reuse probability
            AccessPattern::RandomWrite => 0.5,    // May be read back
            AccessPattern::SequentialWrite => 0.1,// Rarely re-read immediately
            AccessPattern::Mixed => 0.5,
        }
    }
}

/// Cache partition for workload isolation
pub struct CachePartition {
    /// Partition name
    pub name: String,
    /// Maximum size in bytes
    pub max_bytes: usize,
    /// Current size in bytes
    current_bytes: AtomicUsize,
    /// Hit count
    hits: AtomicU64,
    /// Miss count
    misses: AtomicU64,
    /// Eviction count
    evictions: AtomicU64,
}

impl CachePartition {
    /// Create a new partition
    pub fn new(name: &str, max_bytes: usize) -> Self {
        Self {
            name: name.to_string(),
            max_bytes,
            current_bytes: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Try to allocate space in this partition
    pub fn try_allocate(&self, bytes: usize) -> bool {
        loop {
            let current = self.current_bytes.load(Ordering::Relaxed);
            if current + bytes > self.max_bytes {
                return false;
            }
            if self.current_bytes
                .compare_exchange_weak(
                    current,
                    current + bytes,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Release space from this partition
    pub fn release(&self, bytes: usize) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Record a cache hit
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an eviction
    pub fn record_eviction(&self, bytes: usize) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 1.0;
        }
        hits as f64 / total as f64
    }

    /// Get utilization (0.0 - 1.0)
    pub fn utilization(&self) -> f64 {
        self.current_bytes.load(Ordering::Relaxed) as f64 / self.max_bytes as f64
    }

    /// Get partition stats
    pub fn stats(&self) -> PartitionStats {
        PartitionStats {
            name: self.name.clone(),
            max_bytes: self.max_bytes,
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: self.hit_rate(),
            utilization: self.utilization(),
        }
    }
}

/// Partition statistics
#[derive(Debug, Clone)]
pub struct PartitionStats {
    pub name: String,
    pub max_bytes: usize,
    pub current_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    pub utilization: f64,
}

/// I/O isolation policy configuration
#[derive(Debug, Clone)]
pub struct IoIsolationConfig {
    /// Total cache budget in bytes
    pub total_cache_bytes: usize,
    /// Query partition percentage
    pub query_partition_pct: u8,
    /// Compaction partition percentage
    pub compaction_partition_pct: u8,
    /// WAL partition percentage
    pub wal_partition_pct: u8,
    /// Enable automatic Direct I/O for large scans
    pub auto_direct_io: bool,
    /// Threshold for switching to Direct I/O (bytes)
    pub direct_io_threshold: usize,
    /// Under memory pressure, prefer eviction over OOM
    pub prefer_eviction: bool,
    /// Memory pressure threshold (0.0 - 1.0)
    pub memory_pressure_threshold: f64,
}

impl Default for IoIsolationConfig {
    fn default() -> Self {
        Self {
            total_cache_bytes: 1024 * 1024 * 1024, // 1GB
            query_partition_pct: 70,
            compaction_partition_pct: 20,
            wal_partition_pct: 10,
            auto_direct_io: true,
            direct_io_threshold: 64 * 1024 * 1024, // 64MB
            prefer_eviction: true,
            memory_pressure_threshold: 0.85,
        }
    }
}

/// I/O isolation manager
pub struct IoIsolationManager {
    config: IoIsolationConfig,
    /// Query cache partition
    query_partition: CachePartition,
    /// Compaction cache partition
    compaction_partition: CachePartition,
    /// WAL cache partition
    wal_partition: CachePartition,
    /// Total I/O bytes read
    total_read_bytes: AtomicU64,
    /// Total I/O bytes written
    total_write_bytes: AtomicU64,
    /// Direct I/O bytes
    direct_io_bytes: AtomicU64,
    /// Buffered I/O bytes
    buffered_io_bytes: AtomicU64,
}

impl IoIsolationManager {
    /// Create a new I/O isolation manager
    pub fn new(config: IoIsolationConfig) -> Self {
        let total = config.total_cache_bytes;
        let query_bytes = total * config.query_partition_pct as usize / 100;
        let compaction_bytes = total * config.compaction_partition_pct as usize / 100;
        let wal_bytes = total * config.wal_partition_pct as usize / 100;

        Self {
            config,
            query_partition: CachePartition::new("query", query_bytes),
            compaction_partition: CachePartition::new("compaction", compaction_bytes),
            wal_partition: CachePartition::new("wal", wal_bytes),
            total_read_bytes: AtomicU64::new(0),
            total_write_bytes: AtomicU64::new(0),
            direct_io_bytes: AtomicU64::new(0),
            buffered_io_bytes: AtomicU64::new(0),
        }
    }

    /// Get the appropriate cache partition for a workload
    pub fn partition_for(&self, workload: IoWorkloadType) -> &CachePartition {
        match workload {
            IoWorkloadType::Query | IoWorkloadType::Warmup => &self.query_partition,
            IoWorkloadType::Compaction | IoWorkloadType::Backup => &self.compaction_partition,
            IoWorkloadType::Wal => &self.wal_partition,
        }
    }

    /// Decide whether to use Direct I/O for an operation
    pub fn should_use_direct_io(
        &self,
        workload: IoWorkloadType,
        pattern: AccessPattern,
        size_bytes: usize,
    ) -> bool {
        // Explicit workload preference
        if workload.prefers_direct_io() {
            return true;
        }

        // Auto-detect based on size and pattern
        if self.config.auto_direct_io {
            if size_bytes >= self.config.direct_io_threshold {
                // Large operation
                if pattern.cache_benefit_probability() < 0.3 {
                    return true;
                }
            }
        }

        false
    }

    /// Record I/O operation
    pub fn record_io(&self, bytes: usize, is_write: bool, is_direct: bool) {
        if is_write {
            self.total_write_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        } else {
            self.total_read_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        }

        if is_direct {
            self.direct_io_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        } else {
            self.buffered_io_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }

    /// Check if under memory pressure
    pub fn under_memory_pressure(&self) -> bool {
        let total_util = (self.query_partition.utilization()
            + self.compaction_partition.utilization()
            + self.wal_partition.utilization())
            / 3.0;
        total_util > self.config.memory_pressure_threshold
    }

    /// Trigger emergency eviction if under pressure
    pub fn maybe_evict(&self, target_bytes: usize) -> usize {
        if !self.under_memory_pressure() {
            return 0;
        }

        if !self.config.prefer_eviction {
            return 0;
        }

        // In a real implementation, this would trigger cache eviction
        // For now, just signal how much should be evicted
        target_bytes
    }

    /// Get all partition stats
    pub fn all_stats(&self) -> Vec<PartitionStats> {
        vec![
            self.query_partition.stats(),
            self.compaction_partition.stats(),
            self.wal_partition.stats(),
        ]
    }

    /// Get I/O stats
    pub fn io_stats(&self) -> IoStats {
        IoStats {
            total_read_bytes: self.total_read_bytes.load(Ordering::Relaxed),
            total_write_bytes: self.total_write_bytes.load(Ordering::Relaxed),
            direct_io_bytes: self.direct_io_bytes.load(Ordering::Relaxed),
            buffered_io_bytes: self.buffered_io_bytes.load(Ordering::Relaxed),
            direct_io_ratio: {
                let direct = self.direct_io_bytes.load(Ordering::Relaxed);
                let buffered = self.buffered_io_bytes.load(Ordering::Relaxed);
                let total = direct + buffered;
                if total == 0 { 0.0 } else { direct as f64 / total as f64 }
            },
        }
    }
}

/// I/O statistics
#[derive(Debug, Clone)]
pub struct IoStats {
    pub total_read_bytes: u64,
    pub total_write_bytes: u64,
    pub direct_io_bytes: u64,
    pub buffered_io_bytes: u64,
    pub direct_io_ratio: f64,
}

/// Alignment contract for Direct I/O
pub struct AlignmentContract {
    /// Required buffer alignment
    pub buffer_alignment: usize,
    /// Required offset alignment
    pub offset_alignment: usize,
    /// Required size alignment
    pub size_alignment: usize,
}

impl AlignmentContract {
    /// Platform-specific contract
    #[cfg(target_os = "linux")]
    pub fn platform_default() -> Self {
        Self {
            buffer_alignment: 512,
            offset_alignment: 512,
            size_alignment: 512,
        }
    }

    #[cfg(target_os = "macos")]
    pub fn platform_default() -> Self {
        Self {
            buffer_alignment: 4096,
            offset_alignment: 4096,
            size_alignment: 4096,
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn platform_default() -> Self {
        Self {
            buffer_alignment: 4096,
            offset_alignment: 4096,
            size_alignment: 4096,
        }
    }

    /// Validate buffer alignment
    pub fn validate_buffer(&self, ptr: *const u8) -> Result<(), AlignmentError> {
        if (ptr as usize).is_multiple_of(self.buffer_alignment) {
            Ok(())
        } else {
            Err(AlignmentError::BufferMisaligned {
                actual: ptr as usize % self.buffer_alignment,
                required: self.buffer_alignment,
            })
        }
    }

    /// Validate offset alignment
    pub fn validate_offset(&self, offset: u64) -> Result<(), AlignmentError> {
        if (offset as usize).is_multiple_of(self.offset_alignment) {
            Ok(())
        } else {
            Err(AlignmentError::OffsetMisaligned {
                actual: offset,
                required: self.offset_alignment,
            })
        }
    }

    /// Validate size alignment
    pub fn validate_size(&self, size: usize) -> Result<(), AlignmentError> {
        if size.is_multiple_of(self.size_alignment) {
            Ok(())
        } else {
            Err(AlignmentError::SizeMisaligned {
                actual: size,
                required: self.size_alignment,
            })
        }
    }

    /// Round up size to alignment
    pub fn align_size(&self, size: usize) -> usize {
        size.div_ceil(self.size_alignment) * self.size_alignment
    }
}

/// Alignment error
#[derive(Debug)]
pub enum AlignmentError {
    BufferMisaligned { actual: usize, required: usize },
    OffsetMisaligned { actual: u64, required: usize },
    SizeMisaligned { actual: usize, required: usize },
}

impl std::fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlignmentError::BufferMisaligned { actual, required } => {
                write!(
                    f,
                    "Buffer misaligned: offset {} not multiple of {}",
                    actual, required
                )
            }
            AlignmentError::OffsetMisaligned { actual, required } => {
                write!(
                    f,
                    "Offset {} not aligned to {} bytes",
                    actual, required
                )
            }
            AlignmentError::SizeMisaligned { actual, required } => {
                write!(
                    f,
                    "Size {} not aligned to {} bytes",
                    actual, required
                )
            }
        }
    }
}

impl std::error::Error for AlignmentError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_partition_allocation() {
        let partition = CachePartition::new("test", 1024);

        assert!(partition.try_allocate(512));
        assert_eq!(partition.current_bytes.load(Ordering::Relaxed), 512);

        assert!(partition.try_allocate(512));
        assert_eq!(partition.current_bytes.load(Ordering::Relaxed), 1024);

        // Should fail - over capacity
        assert!(!partition.try_allocate(1));

        // Release and try again
        partition.release(512);
        assert!(partition.try_allocate(512));
    }

    #[test]
    fn test_partition_stats() {
        let partition = CachePartition::new("test", 1000);
        
        partition.try_allocate(500);
        partition.record_hit();
        partition.record_hit();
        partition.record_miss();

        let stats = partition.stats();
        assert_eq!(stats.current_bytes, 500);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
        assert!((stats.utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_direct_io_decision() {
        let manager = IoIsolationManager::new(IoIsolationConfig::default());

        // Compaction always uses direct I/O
        assert!(manager.should_use_direct_io(
            IoWorkloadType::Compaction,
            AccessPattern::SequentialScan,
            1024
        ));

        // Query with small size uses buffered
        assert!(!manager.should_use_direct_io(
            IoWorkloadType::Query,
            AccessPattern::RandomRead,
            4096
        ));

        // Query with large size and low reuse probability uses direct
        assert!(manager.should_use_direct_io(
            IoWorkloadType::Query,
            AccessPattern::SequentialScan,
            100 * 1024 * 1024 // 100MB
        ));
    }

    #[test]
    fn test_alignment_contract() {
        let contract = AlignmentContract::platform_default();

        // Valid alignment
        assert!(contract.validate_offset(4096).is_ok());
        assert!(contract.validate_size(4096).is_ok());

        // Invalid alignment (on most platforms)
        if contract.offset_alignment > 1 {
            assert!(contract.validate_offset(1).is_err());
        }

        // Align size
        let aligned = contract.align_size(5000);
        assert!(aligned >= 5000);
        assert!(aligned.is_multiple_of(contract.size_alignment));
    }
}
