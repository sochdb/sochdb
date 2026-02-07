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

//! # Upgrade Compatibility Contract
//!
//! Manages versioned file formats and safe upgrade paths:
//! - Versioned magic numbers for all persisted formats
//! - Forward/backward compatibility policies
//! - Migration orchestration (N → N+1 only)
//! - Downgrade behavior specification
//!
//! ## Design Principles
//!
//! 1. **Explicit Versioning**: All formats have magic + version in header
//! 2. **Safe Upgrades**: Migrations are atomic with rollback capability
//! 3. **No Silent Corruption**: Incompatible formats fail loudly
//! 4. **Document Downgrades**: Usually "not supported" but explicit

use std::collections::HashMap;
use std::fmt;

/// Magic number for SochDB files (8 bytes)
pub const SOCHDB_MAGIC: [u8; 8] = *b"SOCHDB\x00\x01";

/// File format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatType {
    /// Write-Ahead Log segment
    WalSegment,
    /// Data page file
    DataPage,
    /// Manifest/catalog
    Manifest,
    /// HNSW vector index
    HnswIndex,
    /// SSTable (sorted string table)
    Sstable,
    /// Checkpoint file
    Checkpoint,
    /// Backup archive
    BackupArchive,
}

impl FormatType {
    /// Get unique identifier for format type
    pub fn type_id(&self) -> u8 {
        match self {
            FormatType::WalSegment => 0x01,
            FormatType::DataPage => 0x02,
            FormatType::Manifest => 0x03,
            FormatType::HnswIndex => 0x04,
            FormatType::Sstable => 0x05,
            FormatType::Checkpoint => 0x06,
            FormatType::BackupArchive => 0x07,
        }
    }

    /// Parse from type ID
    pub fn from_type_id(id: u8) -> Option<Self> {
        match id {
            0x01 => Some(FormatType::WalSegment),
            0x02 => Some(FormatType::DataPage),
            0x03 => Some(FormatType::Manifest),
            0x04 => Some(FormatType::HnswIndex),
            0x05 => Some(FormatType::Sstable),
            0x06 => Some(FormatType::Checkpoint),
            0x07 => Some(FormatType::BackupArchive),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            FormatType::WalSegment => "WAL Segment",
            FormatType::DataPage => "Data Page",
            FormatType::Manifest => "Manifest",
            FormatType::HnswIndex => "HNSW Index",
            FormatType::Sstable => "SSTable",
            FormatType::Checkpoint => "Checkpoint",
            FormatType::BackupArchive => "Backup Archive",
        }
    }
}

/// Format version with major.minor
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
}

impl FormatVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Check if this version is compatible with another
    /// Same major version is backward compatible
    pub fn is_compatible_with(&self, other: &FormatVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }

    /// Check if upgrade from other to self is supported
    pub fn can_upgrade_from(&self, other: &FormatVersion) -> bool {
        // Only N → N+1 upgrades supported (within same major)
        if self.major == other.major {
            return self.minor >= other.minor;
        }
        // Major version upgrade: only N.x → (N+1).0
        if self.major == other.major + 1 && self.minor == 0 {
            return true;
        }
        false
    }

    /// Serialize to bytes (4 bytes)
    pub fn to_bytes(&self) -> [u8; 4] {
        let mut buf = [0u8; 4];
        buf[0..2].copy_from_slice(&self.major.to_le_bytes());
        buf[2..4].copy_from_slice(&self.minor.to_le_bytes());
        buf
    }

    /// Parse from bytes
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 4 {
            return None;
        }
        Some(Self {
            major: u16::from_le_bytes([buf[0], buf[1]]),
            minor: u16::from_le_bytes([buf[2], buf[3]]),
        })
    }
}

impl fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Current format versions
pub mod current_versions {
    use super::*;

    pub const WAL_SEGMENT: FormatVersion = FormatVersion::new(1, 0);
    pub const DATA_PAGE: FormatVersion = FormatVersion::new(1, 0);
    pub const MANIFEST: FormatVersion = FormatVersion::new(1, 0);
    pub const HNSW_INDEX: FormatVersion = FormatVersion::new(1, 0);
    pub const SSTABLE: FormatVersion = FormatVersion::new(1, 0);
    pub const CHECKPOINT: FormatVersion = FormatVersion::new(1, 0);
    pub const BACKUP_ARCHIVE: FormatVersion = FormatVersion::new(1, 0);
}

/// File header with magic and version
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic bytes (8)
    pub magic: [u8; 8],
    /// Format type (1 byte)
    pub format_type: FormatType,
    /// Format version (4 bytes)
    pub version: FormatVersion,
    /// Feature flags (4 bytes)
    pub feature_flags: u32,
    /// Reserved for future use (15 bytes)
    pub reserved: [u8; 15],
}

impl FileHeader {
    /// Header size in bytes
    pub const SIZE: usize = 32;

    /// Create a new header for a format type
    pub fn new(format_type: FormatType, version: FormatVersion) -> Self {
        Self {
            magic: SOCHDB_MAGIC,
            format_type,
            version,
            feature_flags: 0,
            reserved: [0; 15],
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8] = self.format_type.type_id();
        buf[9..13].copy_from_slice(&self.version.to_bytes());
        buf[13..17].copy_from_slice(&self.feature_flags.to_le_bytes());
        // reserved stays zero
        buf
    }

    /// Parse from bytes
    pub fn from_bytes(buf: &[u8]) -> Result<Self, VersionError> {
        if buf.len() < Self::SIZE {
            return Err(VersionError::InvalidHeader("Header too short".to_string()));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);

        if magic != SOCHDB_MAGIC {
            return Err(VersionError::InvalidMagic {
                expected: SOCHDB_MAGIC,
                found: magic,
            });
        }

        let format_type = FormatType::from_type_id(buf[8])
            .ok_or_else(|| VersionError::UnknownFormatType(buf[8]))?;

        let version = FormatVersion::from_bytes(&buf[9..13])
            .ok_or_else(|| VersionError::InvalidHeader("Invalid version bytes".to_string()))?;

        let feature_flags = u32::from_le_bytes([buf[13], buf[14], buf[15], buf[16]]);

        Ok(Self {
            magic,
            format_type,
            version,
            feature_flags,
            reserved: [0; 15],
        })
    }

    /// Check compatibility with expected type and version
    pub fn check_compatibility(
        &self,
        expected_type: FormatType,
        current_version: FormatVersion,
    ) -> Result<CompatibilityResult, VersionError> {
        if self.format_type != expected_type {
            return Err(VersionError::TypeMismatch {
                expected: expected_type,
                found: self.format_type,
            });
        }

        if self.version == current_version {
            Ok(CompatibilityResult::Exact)
        } else if current_version.is_compatible_with(&self.version) {
            Ok(CompatibilityResult::BackwardCompatible {
                file_version: self.version,
                current_version,
            })
        } else if current_version.can_upgrade_from(&self.version) {
            Ok(CompatibilityResult::NeedsMigration {
                from: self.version,
                to: current_version,
            })
        } else {
            Err(VersionError::Incompatible {
                file_version: self.version,
                current_version,
            })
        }
    }
}

/// Compatibility check result
#[derive(Debug, Clone)]
pub enum CompatibilityResult {
    /// Exact version match
    Exact,
    /// File version is older but readable
    BackwardCompatible {
        file_version: FormatVersion,
        current_version: FormatVersion,
    },
    /// Migration required before use
    NeedsMigration {
        from: FormatVersion,
        to: FormatVersion,
    },
}

/// Version-related errors
#[derive(Debug, Clone)]
pub enum VersionError {
    /// Invalid magic bytes
    InvalidMagic {
        expected: [u8; 8],
        found: [u8; 8],
    },
    /// Unknown format type
    UnknownFormatType(u8),
    /// Format type mismatch
    TypeMismatch {
        expected: FormatType,
        found: FormatType,
    },
    /// Version incompatible
    Incompatible {
        file_version: FormatVersion,
        current_version: FormatVersion,
    },
    /// Invalid header
    InvalidHeader(String),
    /// Migration failed
    MigrationFailed {
        from: FormatVersion,
        to: FormatVersion,
        reason: String,
    },
    /// Downgrade not supported
    DowngradeNotSupported {
        from: FormatVersion,
        to: FormatVersion,
    },
}

impl fmt::Display for VersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionError::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "Invalid magic: expected {:?}, found {:?}",
                    expected, found
                )
            }
            VersionError::UnknownFormatType(id) => {
                write!(f, "Unknown format type: 0x{:02x}", id)
            }
            VersionError::TypeMismatch { expected, found } => {
                write!(
                    f,
                    "Format type mismatch: expected {}, found {}",
                    expected.name(),
                    found.name()
                )
            }
            VersionError::Incompatible {
                file_version,
                current_version,
            } => {
                write!(
                    f,
                    "Incompatible version: file is {}, current is {}",
                    file_version, current_version
                )
            }
            VersionError::InvalidHeader(msg) => {
                write!(f, "Invalid header: {}", msg)
            }
            VersionError::MigrationFailed { from, to, reason } => {
                write!(f, "Migration from {} to {} failed: {}", from, to, reason)
            }
            VersionError::DowngradeNotSupported { from, to } => {
                write!(f, "Downgrade from {} to {} is not supported", from, to)
            }
        }
    }
}

impl std::error::Error for VersionError {}

/// Migration step
pub trait Migration: Send + Sync {
    /// Source version
    fn from_version(&self) -> FormatVersion;
    /// Target version
    fn to_version(&self) -> FormatVersion;
    /// Migrate data (returns new data)
    fn migrate(&self, data: &[u8]) -> Result<Vec<u8>, VersionError>;
    /// Check if migration is reversible
    fn is_reversible(&self) -> bool;
    /// Reverse migration (if reversible)
    fn reverse(&self, data: &[u8]) -> Result<Vec<u8>, VersionError>;
}

/// Migration registry
pub struct MigrationRegistry {
    /// Registered migrations by format type
    migrations: HashMap<FormatType, Vec<Box<dyn Migration>>>,
}

impl MigrationRegistry {
    /// Create a new migration registry
    pub fn new() -> Self {
        Self {
            migrations: HashMap::new(),
        }
    }

    /// Register a migration
    pub fn register(&mut self, format_type: FormatType, migration: Box<dyn Migration>) {
        self.migrations
            .entry(format_type)
            .or_insert_with(Vec::new)
            .push(migration);
    }

    /// Find migration path from one version to another
    pub fn find_path(
        &self,
        format_type: FormatType,
        from: FormatVersion,
        to: FormatVersion,
    ) -> Option<Vec<&dyn Migration>> {
        let migrations = self.migrations.get(&format_type)?;

        // Simple linear search for now (could use graph algorithm for complex paths)
        let mut path = Vec::new();
        let mut current = from;

        while current < to {
            let next = migrations
                .iter()
                .find(|m| m.from_version() == current && m.to_version() > current)?;
            path.push(next.as_ref());
            current = next.to_version();
        }

        if current == to {
            Some(path)
        } else {
            None
        }
    }

    /// Execute migration path
    pub fn execute_path(
        &self,
        path: &[&dyn Migration],
        data: &[u8],
    ) -> Result<Vec<u8>, VersionError> {
        let mut current_data = data.to_vec();
        for migration in path {
            current_data = migration.migrate(&current_data)?;
        }
        Ok(current_data)
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Upgrade policy configuration
#[derive(Debug, Clone)]
pub struct UpgradePolicy {
    /// Allow automatic minor version upgrades
    pub auto_minor_upgrade: bool,
    /// Allow automatic major version upgrades
    pub auto_major_upgrade: bool,
    /// Create backup before migration
    pub backup_before_migration: bool,
    /// Supported upgrade paths
    pub supported_paths: Vec<(FormatVersion, FormatVersion)>,
}

impl Default for UpgradePolicy {
    fn default() -> Self {
        Self {
            auto_minor_upgrade: true,
            auto_major_upgrade: false, // Require explicit action
            backup_before_migration: true,
            supported_paths: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_version_compatibility() {
        let v1_0 = FormatVersion::new(1, 0);
        let v1_1 = FormatVersion::new(1, 1);
        let v2_0 = FormatVersion::new(2, 0);

        // Same version is compatible
        assert!(v1_0.is_compatible_with(&v1_0));

        // Newer minor is compatible with older
        assert!(v1_1.is_compatible_with(&v1_0));

        // Older minor is not compatible with newer
        assert!(!v1_0.is_compatible_with(&v1_1));

        // Different major is not compatible
        assert!(!v2_0.is_compatible_with(&v1_0));
    }

    #[test]
    fn test_upgrade_paths() {
        let v1_0 = FormatVersion::new(1, 0);
        let v1_1 = FormatVersion::new(1, 1);
        let v2_0 = FormatVersion::new(2, 0);

        // Can upgrade within same major
        assert!(v1_1.can_upgrade_from(&v1_0));

        // Can upgrade to next major.0
        assert!(v2_0.can_upgrade_from(&v1_1));

        // Cannot skip major versions
        let v3_0 = FormatVersion::new(3, 0);
        assert!(!v3_0.can_upgrade_from(&v1_0));
    }

    #[test]
    fn test_file_header_roundtrip() {
        let header = FileHeader::new(FormatType::WalSegment, FormatVersion::new(1, 2));

        let bytes = header.to_bytes();
        let parsed = FileHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.format_type, FormatType::WalSegment);
        assert_eq!(parsed.version, FormatVersion::new(1, 2));
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut bytes = [0u8; FileHeader::SIZE];
        bytes[0..8].copy_from_slice(b"INVALID!");

        let result = FileHeader::from_bytes(&bytes);
        assert!(matches!(result, Err(VersionError::InvalidMagic { .. })));
    }

    #[test]
    fn test_compatibility_check() {
        let header = FileHeader::new(FormatType::Manifest, FormatVersion::new(1, 0));

        // Exact match
        let result = header
            .check_compatibility(FormatType::Manifest, FormatVersion::new(1, 0))
            .unwrap();
        assert!(matches!(result, CompatibilityResult::Exact));

        // Backward compatible
        let result = header
            .check_compatibility(FormatType::Manifest, FormatVersion::new(1, 1))
            .unwrap();
        assert!(matches!(result, CompatibilityResult::BackwardCompatible { .. }));

        // Needs migration
        let result = header
            .check_compatibility(FormatType::Manifest, FormatVersion::new(2, 0))
            .unwrap();
        assert!(matches!(result, CompatibilityResult::NeedsMigration { .. }));
    }
}
