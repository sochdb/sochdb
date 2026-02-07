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

//! # Point-in-Time Recovery (PITR)
//!
//! Enables recovery to any point in time using:
//! - Continuous WAL archiving
//! - Periodic base snapshots
//! - Log shipping to object storage
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │   Live WAL      │────▶│  WAL Archiver    │────▶│  Object Storage │
//! └─────────────────┘     └──────────────────┘     │  (S3/GCS/Azure) │
//!                                                   └─────────────────┘
//!                                                           │
//! ┌─────────────────┐     ┌──────────────────┐              ▼
//! │  Base Snapshot  │────▶│  Snapshot Store  │────▶┌─────────────────┐
//! └─────────────────┘     └──────────────────┘     │    Recovery     │
//!                                                   │    Target       │
//!                                                   └─────────────────┘
//! ```
//!
//! ## Recovery Point Objective (RPO)
//!
//! - With synchronous archiving: RPO = 0 (no data loss)
//! - With async archiving (default): RPO = archive interval (typically 1 minute)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;

/// Log Sequence Number for PITR
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lsn(pub u64);

impl Lsn {
    /// Invalid/zero LSN
    pub const INVALID: Lsn = Lsn(0);

    /// Create from raw value
    pub fn new(value: u64) -> Self {
        Lsn(value)
    }

    /// Get raw value
    pub fn value(&self) -> u64 {
        self.0
    }

    /// Increment by offset
    pub fn advance(&self, offset: u64) -> Self {
        Lsn(self.0 + offset)
    }
}

/// Recovery target specification
#[derive(Debug, Clone)]
pub enum RecoveryTarget {
    /// Recover to latest available state
    Latest,
    /// Recover to specific LSN
    Lsn(Lsn),
    /// Recover to specific timestamp
    Timestamp(u64),
    /// Recover to named restore point
    RestorePoint(String),
    /// Recover to specific transaction ID
    TransactionId(u64),
}

/// WAL segment for archival
#[derive(Debug, Clone)]
pub struct WalSegment {
    /// Segment number
    pub segment_id: u64,
    /// Starting LSN
    pub start_lsn: Lsn,
    /// Ending LSN
    pub end_lsn: Lsn,
    /// File path
    pub path: PathBuf,
    /// Segment size in bytes
    pub size_bytes: u64,
    /// CRC32 checksum
    pub checksum: u32,
    /// Creation timestamp
    pub created_at: u64,
    /// Is segment archived?
    pub archived: bool,
}

/// Archive destination configuration
#[derive(Debug, Clone)]
pub enum ArchiveDestination {
    /// Local filesystem
    LocalPath(PathBuf),
    /// Amazon S3
    S3 {
        bucket: String,
        prefix: String,
        region: String,
    },
    /// Google Cloud Storage
    Gcs {
        bucket: String,
        prefix: String,
    },
    /// Azure Blob Storage
    Azure {
        container: String,
        prefix: String,
    },
}

/// WAL archiver configuration
#[derive(Debug, Clone)]
pub struct WalArchiverConfig {
    /// Archive destination
    pub destination: ArchiveDestination,
    /// Archive interval (for async mode)
    pub archive_interval: Duration,
    /// Whether to use synchronous archiving (impacts write latency)
    pub synchronous: bool,
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    /// Encryption key (if enabled)
    pub encryption_key: Option<Vec<u8>>,
    /// Retention period for archived WAL
    pub retention: Duration,
    /// Maximum concurrent uploads
    pub max_concurrent_uploads: usize,
}

impl Default for WalArchiverConfig {
    fn default() -> Self {
        Self {
            destination: ArchiveDestination::LocalPath(PathBuf::from("/var/sochdb/wal_archive")),
            archive_interval: Duration::from_secs(60),
            synchronous: false,
            compression: CompressionAlgorithm::Zstd,
            encryption_key: None,
            retention: Duration::from_secs(7 * 24 * 3600), // 7 days
            max_concurrent_uploads: 4,
        }
    }
}

/// Compression algorithm for archived WAL
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
    Snappy,
}

/// WAL archiver state
pub struct WalArchiver {
    config: WalArchiverConfig,
    /// Current archive LSN (all WAL up to this LSN is archived)
    archive_lsn: AtomicU64,
    /// Segments pending archive
    pending_segments: RwLock<Vec<WalSegment>>,
    /// Archive history
    archive_history: RwLock<BTreeMap<u64, WalSegment>>,
}

impl WalArchiver {
    /// Create a new WAL archiver
    pub fn new(config: WalArchiverConfig) -> Self {
        Self {
            config,
            archive_lsn: AtomicU64::new(0),
            pending_segments: RwLock::new(Vec::new()),
            archive_history: RwLock::new(BTreeMap::new()),
        }
    }

    /// Queue a segment for archiving
    pub fn queue_segment(&self, segment: WalSegment) {
        self.pending_segments.write().push(segment);
    }

    /// Get current archive LSN
    pub fn archive_lsn(&self) -> Lsn {
        Lsn(self.archive_lsn.load(Ordering::Acquire))
    }

    /// Archive pending segments (called periodically or on demand)
    pub fn archive_pending(&self) -> Result<usize, PitrError> {
        let segments: Vec<_> = {
            let mut pending = self.pending_segments.write();
            std::mem::take(&mut *pending)
        };

        if segments.is_empty() {
            return Ok(0);
        }

        let count = segments.len();

        for segment in segments {
            self.archive_segment(&segment)?;

            // Update archive LSN
            let new_lsn = segment.end_lsn.value();
            self.archive_lsn.fetch_max(new_lsn, Ordering::Release);

            // Add to history
            self.archive_history.write().insert(segment.segment_id, segment);
        }

        Ok(count)
    }

    /// Archive a single segment
    fn archive_segment(&self, segment: &WalSegment) -> Result<(), PitrError> {
        match &self.config.destination {
            ArchiveDestination::LocalPath(path) => {
                let dest = path.join(format!("wal_{:016x}", segment.segment_id));
                // In production, this would copy with compression
                std::fs::copy(&segment.path, &dest)
                    .map_err(|e| PitrError::ArchiveFailed(e.to_string()))?;
            }
            ArchiveDestination::S3 { .. } |
            ArchiveDestination::Gcs { .. } |
            ArchiveDestination::Azure { .. } => {
                // Would use appropriate SDK
                return Err(PitrError::NotImplemented("Cloud storage archiving".into()));
            }
        }

        Ok(())
    }

    /// Get segments needed for recovery to target LSN
    pub fn segments_for_recovery(&self, target: Lsn) -> Vec<WalSegment> {
        self.archive_history
            .read()
            .values()
            .filter(|s| s.end_lsn >= target)
            .cloned()
            .collect()
    }
}

/// Base snapshot for PITR
#[derive(Debug, Clone)]
pub struct BaseSnapshot {
    /// Snapshot ID
    pub id: String,
    /// Creation timestamp
    pub created_at: u64,
    /// LSN at snapshot time
    pub lsn: Lsn,
    /// Snapshot size in bytes
    pub size_bytes: u64,
    /// Storage location
    pub location: PathBuf,
    /// Checksum
    pub checksum: String,
}

/// PITR recovery orchestrator
pub struct PitrRecovery {
    /// Base snapshot to start from
    base_snapshot: Option<BaseSnapshot>,
    /// WAL segments to replay
    wal_segments: Vec<WalSegment>,
    /// Recovery target
    target: RecoveryTarget,
    /// Current recovery LSN
    current_lsn: Lsn,
    /// Recovery statistics
    stats: RecoveryStats,
}

/// Recovery statistics
#[derive(Debug, Default)]
pub struct RecoveryStats {
    /// Bytes restored from base snapshot
    pub base_bytes_restored: u64,
    /// WAL bytes replayed
    pub wal_bytes_replayed: u64,
    /// WAL records replayed
    pub wal_records_replayed: u64,
    /// Time spent on base restore
    pub base_restore_time: Duration,
    /// Time spent on WAL replay
    pub wal_replay_time: Duration,
    /// Target LSN reached
    pub target_lsn_reached: bool,
}

impl PitrRecovery {
    /// Create a new PITR recovery
    pub fn new(target: RecoveryTarget) -> Self {
        Self {
            base_snapshot: None,
            wal_segments: Vec::new(),
            target,
            current_lsn: Lsn::INVALID,
            stats: RecoveryStats::default(),
        }
    }

    /// Set base snapshot
    pub fn with_base_snapshot(mut self, snapshot: BaseSnapshot) -> Self {
        self.current_lsn = snapshot.lsn;
        self.base_snapshot = Some(snapshot);
        self
    }

    /// Add WAL segments for replay
    pub fn with_wal_segments(mut self, segments: Vec<WalSegment>) -> Self {
        self.wal_segments = segments;
        self
    }

    /// Calculate if recovery is feasible
    pub fn validate(&self) -> Result<(), PitrError> {
        // Check if we have a base snapshot
        if self.base_snapshot.is_none() {
            return Err(PitrError::NoBaseSnapshot);
        }

        // Check if WAL chain is complete
        let base = self.base_snapshot.as_ref().unwrap();
        let mut expected_lsn = base.lsn;

        for segment in &self.wal_segments {
            if segment.start_lsn > expected_lsn {
                return Err(PitrError::WalGap {
                    expected: expected_lsn,
                    found: segment.start_lsn,
                });
            }
            expected_lsn = segment.end_lsn;
        }

        // Check if we can reach target
        match &self.target {
            RecoveryTarget::Lsn(target_lsn) => {
                if *target_lsn > expected_lsn {
                    return Err(PitrError::TargetUnreachable {
                        target: *target_lsn,
                        available: expected_lsn,
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Get recovery statistics
    pub fn stats(&self) -> &RecoveryStats {
        &self.stats
    }

    /// Get current recovery LSN
    pub fn current_lsn(&self) -> Lsn {
        self.current_lsn
    }
}

/// Restore point (named bookmark for PITR)
#[derive(Debug, Clone)]
pub struct RestorePoint {
    /// Restore point name
    pub name: String,
    /// LSN at creation
    pub lsn: Lsn,
    /// Creation timestamp
    pub created_at: u64,
    /// Optional description
    pub description: Option<String>,
}

/// PITR error types
#[derive(Debug)]
pub enum PitrError {
    /// Archive operation failed
    ArchiveFailed(String),
    /// No base snapshot available
    NoBaseSnapshot,
    /// Gap in WAL chain
    WalGap { expected: Lsn, found: Lsn },
    /// Target cannot be reached
    TargetUnreachable { target: Lsn, available: Lsn },
    /// Feature not implemented
    NotImplemented(String),
    /// I/O error
    Io(std::io::Error),
}

impl std::fmt::Display for PitrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PitrError::ArchiveFailed(msg) => write!(f, "Archive failed: {}", msg),
            PitrError::NoBaseSnapshot => write!(f, "No base snapshot available for recovery"),
            PitrError::WalGap { expected, found } => {
                write!(f, "WAL gap: expected LSN {:?}, found {:?}", expected, found)
            }
            PitrError::TargetUnreachable { target, available } => {
                write!(
                    f,
                    "Recovery target {:?} unreachable, only have WAL up to {:?}",
                    target, available
                )
            }
            PitrError::NotImplemented(feature) => write!(f, "Not implemented: {}", feature),
            PitrError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for PitrError {}

impl From<std::io::Error> for PitrError {
    fn from(e: std::io::Error) -> Self {
        PitrError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsn_ordering() {
        let lsn1 = Lsn::new(100);
        let lsn2 = Lsn::new(200);

        assert!(lsn1 < lsn2);
        assert_eq!(lsn1.advance(100), lsn2);
    }

    #[test]
    fn test_wal_archiver_queue() {
        let config = WalArchiverConfig::default();
        let archiver = WalArchiver::new(config);

        let segment = WalSegment {
            segment_id: 1,
            start_lsn: Lsn::new(0),
            end_lsn: Lsn::new(1000),
            path: PathBuf::from("/tmp/wal_1"),
            size_bytes: 1024,
            checksum: 0x12345678,
            created_at: 0,
            archived: false,
        };

        archiver.queue_segment(segment);
        assert_eq!(archiver.pending_segments.read().len(), 1);
    }

    #[test]
    fn test_pitr_recovery_validation_no_snapshot() {
        let recovery = PitrRecovery::new(RecoveryTarget::Latest);
        assert!(matches!(recovery.validate(), Err(PitrError::NoBaseSnapshot)));
    }

    #[test]
    fn test_pitr_recovery_validation_wal_gap() {
        let snapshot = BaseSnapshot {
            id: "snap1".to_string(),
            created_at: 0,
            lsn: Lsn::new(100),
            size_bytes: 1024,
            location: PathBuf::from("/tmp/snap1"),
            checksum: "abc".to_string(),
        };

        let segments = vec![
            WalSegment {
                segment_id: 1,
                start_lsn: Lsn::new(200), // Gap! Expected 100
                end_lsn: Lsn::new(300),
                path: PathBuf::from("/tmp/wal_1"),
                size_bytes: 1024,
                checksum: 0,
                created_at: 0,
                archived: true,
            },
        ];

        let recovery = PitrRecovery::new(RecoveryTarget::Latest)
            .with_base_snapshot(snapshot)
            .with_wal_segments(segments);

        assert!(matches!(recovery.validate(), Err(PitrError::WalGap { .. })));
    }

    #[test]
    fn test_pitr_recovery_validation_success() {
        let snapshot = BaseSnapshot {
            id: "snap1".to_string(),
            created_at: 0,
            lsn: Lsn::new(100),
            size_bytes: 1024,
            location: PathBuf::from("/tmp/snap1"),
            checksum: "abc".to_string(),
        };

        let segments = vec![
            WalSegment {
                segment_id: 1,
                start_lsn: Lsn::new(100),
                end_lsn: Lsn::new(200),
                path: PathBuf::from("/tmp/wal_1"),
                size_bytes: 1024,
                checksum: 0,
                created_at: 0,
                archived: true,
            },
        ];

        let recovery = PitrRecovery::new(RecoveryTarget::Lsn(Lsn::new(150)))
            .with_base_snapshot(snapshot)
            .with_wal_segments(segments);

        assert!(recovery.validate().is_ok());
    }
}
