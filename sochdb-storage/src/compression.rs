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

//! Storage compression and optimization module
//!
//! Implements multi-tier compression strategy:
//! - Hot data (recent): LZ4 for speed
//! - Warm data (1-30 days): Zstd level 3 for balance
//! - Cold data (>30 days): Zstd level 19 for maximum compression
//!
//! Also provides:
//! - Deduplication for common patterns (system prompts)
//! - Automatic tiering based on age
//! - Compression ratio tracking

use std::collections::HashMap;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

/// Compression type identifier
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None = 0,
    Lz4 = 1,
    ZstdFast = 2, // Level 3
    ZstdMax = 3,  // Level 19
}

impl CompressionType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => CompressionType::Lz4,
            2 => CompressionType::ZstdFast,
            3 => CompressionType::ZstdMax,
            _ => CompressionType::None,
        }
    }
}

/// Storage tier based on data age
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    Hot,  // < 24 hours
    Warm, // 1-30 days
    Cold, // > 30 days
}

impl StorageTier {
    /// Determine tier based on age
    pub fn from_age(timestamp_us: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let age_us = now.saturating_sub(timestamp_us);
        let age_hours = age_us / 3_600_000_000;

        if age_hours < 24 {
            StorageTier::Hot
        } else if age_hours < 720 {
            // 30 days
            StorageTier::Warm
        } else {
            StorageTier::Cold
        }
    }

    /// Get recommended compression for this tier
    pub fn compression_type(&self) -> CompressionType {
        match self {
            StorageTier::Hot => CompressionType::Lz4, // Fast compression
            StorageTier::Warm => CompressionType::ZstdFast, // Balanced
            StorageTier::Cold => CompressionType::ZstdMax, // Maximum compression
        }
    }
}

/// Compression engine
pub struct CompressionEngine {
    /// Deduplication cache (hash -> compressed data)
    dedup_cache: HashMap<u64, Vec<u8>>,
    /// Compression statistics
    stats: CompressionStats,
}

#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    pub total_uncompressed: u64,
    pub total_compressed: u64,
    pub lz4_count: u64,
    pub zstd_fast_count: u64,
    pub zstd_max_count: u64,
    pub dedup_hits: u64,
}

impl CompressionStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.total_uncompressed == 0 {
            return 1.0;
        }
        self.total_compressed as f64 / self.total_uncompressed as f64
    }

    pub fn space_saved_bytes(&self) -> u64 {
        self.total_uncompressed
            .saturating_sub(self.total_compressed)
    }
}

impl CompressionEngine {
    pub fn new() -> Self {
        Self {
            dedup_cache: HashMap::new(),
            stats: CompressionStats::default(),
        }
    }

    /// Compress data using specified algorithm
    pub fn compress(
        &mut self,
        data: &[u8],
        compression: CompressionType,
    ) -> Result<Vec<u8>, std::io::Error> {
        self.stats.total_uncompressed += data.len() as u64;

        let compressed = match compression {
            CompressionType::None => data.to_vec(),
            CompressionType::Lz4 => self.compress_lz4(data)?,
            CompressionType::ZstdFast => self.compress_zstd(data, 3)?,
            CompressionType::ZstdMax => self.compress_zstd(data, 19)?,
        };

        self.stats.total_compressed += compressed.len() as u64;

        match compression {
            CompressionType::Lz4 => self.stats.lz4_count += 1,
            CompressionType::ZstdFast => self.stats.zstd_fast_count += 1,
            CompressionType::ZstdMax => self.stats.zstd_max_count += 1,
            _ => {}
        }

        Ok(compressed)
    }

    /// Decompress data
    pub fn decompress(
        &self,
        data: &[u8],
        compression: CompressionType,
    ) -> Result<Vec<u8>, std::io::Error> {
        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Lz4 => self.decompress_lz4(data),
            CompressionType::ZstdFast | CompressionType::ZstdMax => self.decompress_zstd(data),
        }
    }

    /// Compress with deduplication
    pub fn compress_with_dedup(
        &mut self,
        data: &[u8],
        compression: CompressionType,
    ) -> Result<Vec<u8>, std::io::Error> {
        // Use xxHash3 for dedup hashing — 5× faster than SipHash, non-adversarial context
        let hash = twox_hash::xxh3::hash64(data);

        // Check dedup cache
        if let Some(cached) = self.dedup_cache.get(&hash) {
            self.stats.dedup_hits += 1;
            return Ok(cached.clone());
        }

        // Compress and cache
        let compressed = self.compress(data, compression)?;

        // Only cache if it's worth it (data > 1KB and compression ratio > 2:1)
        if data.len() > 1024 && compressed.len() > 0 && (data.len() / compressed.len()) >= 2 {
            self.dedup_cache.insert(hash, compressed.clone());
        }

        Ok(compressed)
    }

    /// LZ4 compression using lz4_flex (block mode, ~3 GB/s throughput)
    ///
    /// Wire format: [original_len: u32 LE] [lz4_compressed_payload...]
    /// If compressed output >= original size, falls back to uncompressed with len=0 sentinel.
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let compressed = lz4_flex::compress_prepend_size(data);
        // Fallback: if compressed is larger than original + 4-byte header, store raw
        if compressed.len() >= data.len() + 4 {
            let mut output = Vec::with_capacity(data.len() + 4);
            output.extend_from_slice(&0u32.to_le_bytes()); // 0 = uncompressed sentinel
            output.extend_from_slice(data);
            Ok(output)
        } else {
            Ok(compressed)
        }
    }

    /// LZ4 decompression
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LZ4 data too short (< 4 bytes)",
            ));
        }
        let original_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if original_len == 0 {
            // Uncompressed fallback: sentinel 0 means raw payload follows
            return Ok(data[4..].to_vec());
        }
        lz4_flex::decompress_size_prepended(data).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LZ4 decompression failed: {}", e),
            )
        })
    }

    /// Zstd compression at the specified level
    ///
    /// Level 3: ~500 MB/s, ~3× ratio (warm tier)
    /// Level 19: ~40 MB/s, ~4.5× ratio (cold tier — use from background compaction only)
    ///
    /// Wire format: raw zstd frame (self-describing, includes original size)
    /// If compressed output >= original, falls back with a 4-byte sentinel header.
    fn compress_zstd(&self, data: &[u8], level: i32) -> Result<Vec<u8>, std::io::Error> {
        let compressed = zstd::encode_all(std::io::Cursor::new(data), level)?;
        // Fallback: if compression didn't help, store raw with sentinel
        if compressed.len() >= data.len() {
            let mut output = Vec::with_capacity(data.len() + 4);
            output.extend_from_slice(b"\x00\x00\x00\x00"); // 4 zero bytes = uncompressed sentinel
            output.extend_from_slice(data);
            Ok(output)
        } else {
            Ok(compressed)
        }
    }

    /// Zstd decompression
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Zstd data too short (< 4 bytes)",
            ));
        }
        // Check for uncompressed sentinel (4 zero bytes and NOT a valid zstd magic)
        if &data[0..4] == b"\x00\x00\x00\x00" {
            return Ok(data[4..].to_vec());
        }
        zstd::decode_all(std::io::Cursor::new(data))
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Clear deduplication cache
    pub fn clear_cache(&mut self) {
        self.dedup_cache.clear();
    }

    /// Get cache size in bytes
    pub fn cache_size(&self) -> usize {
        self.dedup_cache.values().map(|v| v.len()).sum()
    }
}

impl Default for CompressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: Determine optimal compression for payload
pub fn choose_compression(size: usize, age_us: u64) -> CompressionType {
    // Small payloads: don't compress (overhead not worth it)
    if size < 512 {
        return CompressionType::None;
    }

    // Use tier-based compression
    let tier = StorageTier::from_age(age_us);
    tier.compression_type()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Recent data -> Hot
        let tier = StorageTier::from_age(now - 3_600_000_000); // 1 hour ago
        assert_eq!(tier, StorageTier::Hot);

        // Week old -> Warm
        let tier = StorageTier::from_age(now - 604_800_000_000); // 7 days ago
        assert_eq!(tier, StorageTier::Warm);

        // Very old -> Cold
        let tier = StorageTier::from_age(now - 3_000_000_000_000); // ~35 days ago
        assert_eq!(tier, StorageTier::Cold);
    }

    #[test]
    fn test_lz4_roundtrip() {
        let mut engine = CompressionEngine::new();
        let data = b"Hello, World! This is test data for LZ4 compression roundtrip.";

        let compressed = engine.compress(data, CompressionType::Lz4).unwrap();
        let decompressed = engine
            .decompress(&compressed, CompressionType::Lz4)
            .unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_zstd_fast_roundtrip() {
        let mut engine = CompressionEngine::new();
        let data = b"Hello, World! This is test data for Zstd level-3 compression roundtrip.";

        let compressed = engine.compress(data, CompressionType::ZstdFast).unwrap();
        let decompressed = engine
            .decompress(&compressed, CompressionType::ZstdFast)
            .unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_zstd_max_roundtrip() {
        let mut engine = CompressionEngine::new();
        let data = b"Hello, World! This is test data for Zstd level-19 maximum compression roundtrip.";

        let compressed = engine.compress(data, CompressionType::ZstdMax).unwrap();
        let decompressed = engine
            .decompress(&compressed, CompressionType::ZstdMax)
            .unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_real_compression_ratio() {
        let mut engine = CompressionEngine::new();
        // Highly compressible data: repeated pattern
        let data: Vec<u8> = "The quick brown fox jumps over the lazy dog. "
            .repeat(100)
            .into_bytes();

        let lz4 = engine.compress(&data, CompressionType::Lz4).unwrap();
        assert!(
            lz4.len() < data.len(),
            "LZ4 should compress repetitive data: {} -> {}",
            data.len(),
            lz4.len()
        );

        let mut engine2 = CompressionEngine::new();
        let zstd_fast = engine2.compress(&data, CompressionType::ZstdFast).unwrap();
        assert!(
            zstd_fast.len() < data.len(),
            "ZstdFast should compress repetitive data: {} -> {}",
            data.len(),
            zstd_fast.len()
        );

        let mut engine3 = CompressionEngine::new();
        let zstd_max = engine3.compress(&data, CompressionType::ZstdMax).unwrap();
        assert!(
            zstd_max.len() < data.len(),
            "ZstdMax should compress repetitive data: {} -> {}",
            data.len(),
            zstd_max.len()
        );

        // ZstdMax should compress at least as well as ZstdFast
        assert!(
            zstd_max.len() <= zstd_fast.len(),
            "ZstdMax ({}) should be <= ZstdFast ({})",
            zstd_max.len(),
            zstd_fast.len()
        );
    }

    #[test]
    fn test_compression_stats() {
        let mut engine = CompressionEngine::new();
        let data: Vec<u8> = "Test data for compression statistics. ".repeat(50).into_bytes();

        engine.compress(&data, CompressionType::Lz4).unwrap();

        let stats = engine.stats();
        assert!(stats.total_uncompressed > 0);
        assert!(stats.total_compressed > 0);
        assert_eq!(stats.lz4_count, 1);
        // Real compression should actually save space on repetitive data
        assert!(
            stats.space_saved_bytes() > 0,
            "Should save space on compressible data"
        );
        assert!(
            stats.compression_ratio() < 1.0,
            "Ratio should be < 1.0 (compressed smaller than original)"
        );
    }

    #[test]
    fn test_deduplication() {
        let mut engine = CompressionEngine::new();
        // Data must be > 1024 bytes AND achieve 2:1 compression for caching
        let data: Vec<u8> = "Repeated system prompt for deduplication testing. "
            .repeat(100)
            .into_bytes();
        assert!(data.len() > 1024);

        // First call: compresses and caches
        let first = engine
            .compress_with_dedup(&data, CompressionType::Lz4)
            .unwrap();
        assert_eq!(engine.stats().dedup_hits, 0);

        // Second call: should hit dedup cache
        let second = engine
            .compress_with_dedup(&data, CompressionType::Lz4)
            .unwrap();
        assert_eq!(engine.stats().dedup_hits, 1);
        assert_eq!(first, second);
    }

    #[test]
    fn test_small_data_fallback() {
        // Data too small to compress effectively — should still roundtrip correctly
        let mut engine = CompressionEngine::new();
        let data = b"tiny";

        let lz4 = engine.compress(data, CompressionType::Lz4).unwrap();
        let rt = engine.decompress(&lz4, CompressionType::Lz4).unwrap();
        assert_eq!(data.as_slice(), rt.as_slice());

        let mut engine2 = CompressionEngine::new();
        let zstd = engine2.compress(data, CompressionType::ZstdFast).unwrap();
        let rt2 = engine2.decompress(&zstd, CompressionType::ZstdFast).unwrap();
        assert_eq!(data.as_slice(), rt2.as_slice());
    }

    #[test]
    fn test_choose_compression() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Small payload -> None
        assert_eq!(choose_compression(100, now), CompressionType::None);

        // Recent large payload -> LZ4
        assert_eq!(choose_compression(10000, now), CompressionType::Lz4);

        // Old large payload -> ZstdMax
        let old = now - 4_000_000_000_000; // ~46 days ago
        assert_eq!(choose_compression(10000, old), CompressionType::ZstdMax);
    }

    #[test]
    fn test_none_compression_passthrough() {
        let mut engine = CompressionEngine::new();
        let data = b"no compression applied";

        let compressed = engine.compress(data, CompressionType::None).unwrap();
        assert_eq!(data.as_slice(), compressed.as_slice());

        let decompressed = engine
            .decompress(&compressed, CompressionType::None)
            .unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_large_payload_compression() {
        let mut engine = CompressionEngine::new();
        // Simulate a large LLM conversation context (JSON-like)
        let data: Vec<u8> = (0..10000)
            .map(|i| format!("{{\"role\":\"user\",\"content\":\"message {}\"}},", i))
            .collect::<String>()
            .into_bytes();

        let compressed = engine.compress(&data, CompressionType::ZstdFast).unwrap();
        let ratio = compressed.len() as f64 / data.len() as f64;
        assert!(
            ratio < 0.5,
            "Large repetitive JSON should compress to < 50%: ratio={:.3}",
            ratio
        );

        let decompressed = engine
            .decompress(&compressed, CompressionType::ZstdFast)
            .unwrap();
        assert_eq!(data, decompressed);
    }
}
