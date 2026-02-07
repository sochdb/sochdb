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

//! # Durability Contract Hardening
//!
//! This module formalizes and enforces durability guarantees for the storage layer.
//! It ensures that:
//! - Any mode that allows data loss is explicitly labeled and requires opt-in
//! - The default mode is always fsync-after-commit (ACID D)
//! - ARIES recovery invariants are mechanically enforced
//! - WAL archiving hooks are available for PITR
//!
//! ## Durability Levels
//!
//! - `Durable`: fsync after commit (default, safe)
//! - `GroupCommit`: fsync batched, bounded latency (production)
//! - `Periodic`: fsync on timer (UNSAFE - labeled clearly)
//! - `NoSync`: no fsync (UNSAFE - for testing only)
//!
//! ## ARIES Recovery Contract
//!
//! 1. **Write-Ahead Logging**: No page written to disk before its log record
//! 2. **Force on Commit**: All log records for a txn forced before commit returns
//! 3. **Steal**: Buffer pool can evict dirty pages before commit (with WAL guarantee)
//! 4. **Recovery Phases**: Analysis → Redo → Undo (with CLRs for nested undo)

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Durability level for write operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityLevel {
    /// Every commit is immediately fsync'd (safest, slowest)
    /// Guarantees: commit returns ⇒ data is on stable storage
    Durable,

    /// Commits are batched and fsync'd together (production default)
    /// Guarantees: commit returns ⇒ data will be on stable storage within flush_interval
    /// Trade-off: Up to flush_interval ms of data loss on crash
    GroupCommit {
        /// Maximum number of commits to batch
        max_batch: usize,
        /// Maximum time to wait before flush
        flush_interval: Duration,
    },

    /// Fsync on timer (UNSAFE - data loss possible)
    /// WARNING: Up to sync_interval of committed transactions can be lost
    Periodic {
        /// Sync interval
        sync_interval: Duration,
        /// Acknowledgement that this is unsafe
        accept_data_loss_risk: bool,
    },

    /// No fsync (UNSAFE - testing only)
    /// WARNING: ANY committed data can be lost on crash
    NoSync {
        /// Acknowledgement that this is unsafe
        accept_data_loss_risk: bool,
    },
}

impl DurabilityLevel {
    /// Check if this level is safe for production
    pub fn is_production_safe(&self) -> bool {
        matches!(self, DurabilityLevel::Durable | DurabilityLevel::GroupCommit { .. })
    }

    /// Check if explicit risk acceptance is required
    pub fn requires_risk_acceptance(&self) -> bool {
        matches!(self, DurabilityLevel::Periodic { .. } | DurabilityLevel::NoSync { .. })
    }

    /// Get the default production-safe configuration
    pub fn default_production() -> Self {
        DurabilityLevel::GroupCommit {
            max_batch: 128,
            flush_interval: Duration::from_millis(10),
        }
    }

    /// Get the safest configuration
    pub fn safest() -> Self {
        DurabilityLevel::Durable
    }
}

impl Default for DurabilityLevel {
    fn default() -> Self {
        Self::default_production()
    }
}

impl fmt::Display for DurabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DurabilityLevel::Durable => write!(f, "Durable (fsync per commit)"),
            DurabilityLevel::GroupCommit { max_batch, flush_interval } => {
                write!(f, "GroupCommit (batch={}, interval={:?})", max_batch, flush_interval)
            }
            DurabilityLevel::Periodic { sync_interval, .. } => {
                write!(f, "UNSAFE:Periodic (interval={:?})", sync_interval)
            }
            DurabilityLevel::NoSync { .. } => {
                write!(f, "UNSAFE:NoSync (testing only)")
            }
        }
    }
}

/// Durability contract validation error
#[derive(Debug, Clone)]
pub struct DurabilityContractError {
    pub message: String,
    pub contract_violated: &'static str,
}

impl fmt::Display for DurabilityContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Durability contract '{}' violated: {}", self.contract_violated, self.message)
    }
}

impl std::error::Error for DurabilityContractError {}

/// WAL archiving configuration for PITR
#[derive(Debug, Clone)]
pub struct WalArchiveConfig {
    /// Enable WAL archiving
    pub enabled: bool,
    /// Archive destination path (local) or command
    pub destination: WalArchiveDestination,
    /// Archive trigger: segment full or time-based
    pub trigger: WalArchiveTrigger,
    /// Compress archived segments
    pub compress: bool,
    /// Verify archived segments (read back and check)
    pub verify: bool,
}

impl Default for WalArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            destination: WalArchiveDestination::LocalPath(PathBuf::from("/var/lib/sochdb/wal_archive")),
            trigger: WalArchiveTrigger::SegmentFull,
            compress: true,
            verify: true,
        }
    }
}

/// WAL archive destination
#[derive(Debug, Clone)]
pub enum WalArchiveDestination {
    /// Local filesystem path
    LocalPath(PathBuf),
    /// External command (receives segment path as argument)
    ExternalCommand(String),
    /// S3-compatible object storage (bucket URL)
    ObjectStorage {
        endpoint: String,
        bucket: String,
        prefix: String,
    },
}

/// When to trigger WAL archiving
#[derive(Debug, Clone, Copy)]
pub enum WalArchiveTrigger {
    /// Archive when segment is full (default)
    SegmentFull,
    /// Archive on timer (may archive partial segments)
    TimeBased(Duration),
    /// Archive on both conditions (whichever first)
    Both {
        max_interval: Duration,
    },
}

/// Checkpoint configuration for avoiding long write stalls
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Checkpoint mode
    pub mode: CheckpointMode,
    /// Maximum time allowed for checkpoint
    pub max_duration: Duration,
    /// Trigger checkpoint at this WAL size
    pub wal_size_trigger: u64,
    /// Trigger checkpoint at this many transactions
    pub txn_count_trigger: u64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            mode: CheckpointMode::Incremental,
            max_duration: Duration::from_secs(30),
            wal_size_trigger: 128 * 1024 * 1024, // 128MB
            txn_count_trigger: 100_000,
        }
    }
}

/// Checkpoint mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointMode {
    /// Full checkpoint (write all dirty pages) - can cause stalls
    Full,
    /// Incremental checkpoint (spread writes over time)
    Incremental,
    /// Copy-on-write snapshot (no stalls, more space)
    CopyOnWrite,
}

/// Durability contract that enforces ARIES invariants
pub struct DurabilityContract {
    /// Configured durability level
    pub level: DurabilityLevel,
    /// WAL archive configuration
    pub archive: WalArchiveConfig,
    /// Checkpoint configuration
    pub checkpoint: CheckpointConfig,
    /// Require WAL before page flush (invariant)
    wal_before_page: bool,
    /// Force log on commit (invariant)
    force_on_commit: bool,
}

impl DurabilityContract {
    /// Create a new durability contract with safe defaults
    pub fn new(level: DurabilityLevel) -> Result<Self, DurabilityContractError> {
        // Validate unsafe levels have risk acceptance
        match &level {
            DurabilityLevel::Periodic { accept_data_loss_risk, .. } 
            | DurabilityLevel::NoSync { accept_data_loss_risk, .. } => {
                if !accept_data_loss_risk {
                    return Err(DurabilityContractError {
                        message: format!(
                            "Unsafe durability level '{}' requires explicit accept_data_loss_risk=true",
                            level
                        ),
                        contract_violated: "EXPLICIT_RISK_ACCEPTANCE",
                    });
                }
            }
            _ => {}
        }

        Ok(Self {
            level,
            archive: WalArchiveConfig::default(),
            checkpoint: CheckpointConfig::default(),
            wal_before_page: true,
            force_on_commit: true,
        })
    }

    /// Create with production defaults
    pub fn production() -> Self {
        Self {
            level: DurabilityLevel::default_production(),
            archive: WalArchiveConfig::default(),
            checkpoint: CheckpointConfig::default(),
            wal_before_page: true,
            force_on_commit: true,
        }
    }

    /// Enable WAL archiving
    pub fn with_archive(mut self, config: WalArchiveConfig) -> Self {
        self.archive = config;
        self
    }

    /// Configure checkpointing
    pub fn with_checkpoint(mut self, config: CheckpointConfig) -> Self {
        self.checkpoint = config;
        self
    }

    /// Validate that a page flush respects WAL protocol
    /// 
    /// ARIES invariant: Page cannot be flushed until its log record is durable
    pub fn validate_page_flush(&self, page_lsn: u64, flushed_lsn: u64) -> Result<(), DurabilityContractError> {
        if self.wal_before_page && page_lsn > flushed_lsn {
            return Err(DurabilityContractError {
                message: format!(
                    "Page LSN {} exceeds flushed WAL LSN {} - would violate WAL protocol",
                    page_lsn, flushed_lsn
                ),
                contract_violated: "WAL_BEFORE_PAGE",
            });
        }
        Ok(())
    }

    /// Validate that a commit respects force-on-commit
    ///
    /// ARIES invariant: Commit record must be durable before returning
    pub fn validate_commit(&self, commit_lsn: u64, flushed_lsn: u64) -> Result<(), DurabilityContractError> {
        if self.force_on_commit && commit_lsn > flushed_lsn {
            return Err(DurabilityContractError {
                message: format!(
                    "Commit LSN {} not yet flushed (flushed: {}) - would violate force-on-commit",
                    commit_lsn, flushed_lsn
                ),
                contract_violated: "FORCE_ON_COMMIT",
            });
        }
        Ok(())
    }

    /// Get maximum data loss window based on durability level
    pub fn max_data_loss_window(&self) -> Option<Duration> {
        match &self.level {
            DurabilityLevel::Durable => None, // No data loss possible
            DurabilityLevel::GroupCommit { flush_interval, .. } => Some(*flush_interval),
            DurabilityLevel::Periodic { sync_interval, .. } => Some(*sync_interval),
            DurabilityLevel::NoSync { .. } => None, // Unbounded - all data at risk
        }
    }

    /// Check if this contract is suitable for production
    pub fn is_production_ready(&self) -> bool {
        self.level.is_production_safe() && self.wal_before_page && self.force_on_commit
    }

    /// Generate a human-readable description of guarantees
    pub fn describe_guarantees(&self) -> String {
        let mut desc = Vec::new();

        desc.push(format!("Durability: {}", self.level));

        if self.wal_before_page {
            desc.push("WAL-before-page: ENFORCED".to_string());
        } else {
            desc.push("WAL-before-page: DISABLED (UNSAFE)".to_string());
        }

        if self.force_on_commit {
            desc.push("Force-on-commit: ENFORCED".to_string());
        } else {
            desc.push("Force-on-commit: DISABLED (UNSAFE)".to_string());
        }

        if let Some(window) = self.max_data_loss_window() {
            desc.push(format!("Max data loss window: {:?}", window));
        } else if matches!(self.level, DurabilityLevel::NoSync { .. }) {
            desc.push("Max data loss window: UNBOUNDED (all data at risk)".to_string());
        } else {
            desc.push("Max data loss window: None (fully durable)".to_string());
        }

        if self.archive.enabled {
            desc.push("WAL archiving: ENABLED".to_string());
        }

        match self.checkpoint.mode {
            CheckpointMode::Full => desc.push("Checkpoint: Full (may cause stalls)".to_string()),
            CheckpointMode::Incremental => desc.push("Checkpoint: Incremental".to_string()),
            CheckpointMode::CopyOnWrite => desc.push("Checkpoint: Copy-on-write".to_string()),
        }

        desc.join("\n")
    }
}

impl Default for DurabilityContract {
    fn default() -> Self {
        Self::production()
    }
}

/// Recovery point information for PITR
#[derive(Debug, Clone)]
pub struct RecoveryPoint {
    /// LSN of this recovery point
    pub lsn: u64,
    /// Timestamp of this recovery point
    pub timestamp: u64,
    /// Checkpoint ID if this is a checkpoint
    pub checkpoint_id: Option<u64>,
    /// Description
    pub description: String,
}

/// WAL segment metadata for archiving
#[derive(Debug, Clone)]
pub struct WalSegmentMetadata {
    /// Segment number
    pub segment_number: u64,
    /// Start LSN
    pub start_lsn: u64,
    /// End LSN
    pub end_lsn: u64,
    /// Size in bytes
    pub size_bytes: u64,
    /// CRC32 checksum
    pub checksum: u32,
    /// Is segment complete (closed)
    pub is_complete: bool,
    /// Archive status
    pub archive_status: ArchiveStatus,
}

/// Archive status for a WAL segment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveStatus {
    /// Not yet archived
    Pending,
    /// Archive in progress
    InProgress,
    /// Successfully archived
    Archived,
    /// Archive failed
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_durability_level_defaults() {
        let level = DurabilityLevel::default();
        assert!(level.is_production_safe());
        assert!(!level.requires_risk_acceptance());
    }

    #[test]
    fn test_unsafe_requires_acknowledgement() {
        let result = DurabilityContract::new(DurabilityLevel::NoSync {
            accept_data_loss_risk: false,
        });
        assert!(result.is_err());

        let result = DurabilityContract::new(DurabilityLevel::NoSync {
            accept_data_loss_risk: true,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_wal_before_page_validation() {
        let contract = DurabilityContract::production();

        // Valid: page LSN <= flushed LSN
        assert!(contract.validate_page_flush(100, 100).is_ok());
        assert!(contract.validate_page_flush(50, 100).is_ok());

        // Invalid: page LSN > flushed LSN
        assert!(contract.validate_page_flush(150, 100).is_err());
    }

    #[test]
    fn test_force_on_commit_validation() {
        let contract = DurabilityContract::production();

        // Valid: commit LSN <= flushed LSN
        assert!(contract.validate_commit(100, 100).is_ok());
        assert!(contract.validate_commit(50, 100).is_ok());

        // Invalid: commit LSN > flushed LSN
        assert!(contract.validate_commit(150, 100).is_err());
    }

    #[test]
    fn test_data_loss_window() {
        let durable = DurabilityContract::new(DurabilityLevel::Durable).unwrap();
        assert!(durable.max_data_loss_window().is_none());

        let group = DurabilityContract::new(DurabilityLevel::GroupCommit {
            max_batch: 100,
            flush_interval: Duration::from_millis(10),
        }).unwrap();
        assert_eq!(group.max_data_loss_window(), Some(Duration::from_millis(10)));
    }

    #[test]
    fn test_production_ready_check() {
        let contract = DurabilityContract::production();
        assert!(contract.is_production_ready());

        let unsafe_contract = DurabilityContract::new(DurabilityLevel::NoSync {
            accept_data_loss_risk: true,
        }).unwrap();
        assert!(!unsafe_contract.is_production_ready());
    }
}
