// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # WAL manifest — durable PITR anchor (Task 3B, PITR phase 1)
//!
//! The manifest is the **single source of truth** for a database's Point-in-Time
//! Recovery state. Its mere PRESENCE marks a database as PITR-enabled (so the
//! anchor is consistent regardless of how a given process opens the DB), and it
//! durably records the last-checkpoint LSN so it survives process restarts.
//!
//! ## Why a manifest at all
//!
//! On the live path `DurableStorage::checkpoint()` does NOT truncate the WAL, so
//! the per-file record counter (`TxnWal::sequence`) is already a monotonic LSN
//! that `recover_state` rebuilds by re-counting on every reopen — i.e. it is
//! durable across restarts *as long as the WAL is never truncated*. PITR mode
//! therefore forbids the destructive `truncate_wal()` (segment **sealing** is the
//! PITR-safe replacement, landing in a later phase), which keeps `sequence` a
//! stable global anchor. The manifest then only needs to persist the
//! last-checkpoint LSN (which is otherwise in-memory and lost on restart) plus a
//! DB identity, written crash-safely.
//!
//! ## On-disk format (`<db_dir>/wal.manifest`, JSON)
//!
//! ```text
//! { format_version, db_uuid (hex 16), last_checkpoint_lsn }
//! ```
//!
//! Persisted with the same atomic temp→fsync→rename→fsync-dir pattern as the
//! keyring, so a crash never leaves a torn manifest.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

use sochdb_core::{Result, SochDBError};

/// Current manifest format version.
const WAL_MANIFEST_FORMAT_VERSION: u32 = 1;
/// Manifest file name within the database directory.
pub const WAL_MANIFEST_FILE: &str = "wal.manifest";

/// On-disk WAL manifest (the durable PITR anchor).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WalManifest {
    /// Format version of this manifest.
    pub format_version: u32,
    /// 16-byte database identity (hex-encoded), bound for future segment-catalog
    /// integrity. Random per database at PITR-enable time.
    pub db_uuid: String,
    /// LSN (monotonic WAL record ordinal) as of the most recent checkpoint.
    pub last_checkpoint_lsn: u64,
}

impl WalManifest {
    /// Create a fresh manifest with a random db_uuid and the given starting LSN.
    pub fn new(last_checkpoint_lsn: u64) -> Self {
        let mut uuid = [0u8; 16];
        {
            use rand::RngCore;
            rand::rngs::OsRng.fill_bytes(&mut uuid);
        }
        Self {
            format_version: WAL_MANIFEST_FORMAT_VERSION,
            db_uuid: hex::encode(uuid),
            last_checkpoint_lsn,
        }
    }

    /// Path to the manifest within `db_dir`.
    pub fn path(db_dir: &Path) -> PathBuf {
        db_dir.join(WAL_MANIFEST_FILE)
    }

    /// Whether a manifest exists in `db_dir` (i.e. the DB is PITR-enabled).
    pub fn exists(db_dir: &Path) -> bool {
        Self::path(db_dir).exists()
    }

    /// Load and validate the manifest from `db_dir`.
    pub fn load(db_dir: &Path) -> Result<Self> {
        let bytes = fs::read(Self::path(db_dir))?;
        let m: WalManifest = serde_json::from_slice(&bytes)
            .map_err(|e| SochDBError::Corruption(format!("malformed wal.manifest: {e}")))?;
        if m.format_version != WAL_MANIFEST_FORMAT_VERSION {
            return Err(SochDBError::Corruption(format!(
                "unsupported wal.manifest version {} (expected {})",
                m.format_version, WAL_MANIFEST_FORMAT_VERSION
            )));
        }
        Ok(m)
    }

    /// Atomically persist the manifest: write temp, fsync, rename, fsync dir.
    /// Crash-safe — a torn write leaves the previous manifest (or none) intact.
    pub fn write_atomic(&self, db_dir: &Path) -> Result<()> {
        fs::create_dir_all(db_dir)?;
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| SochDBError::Internal(format!("serialize wal.manifest: {e}")))?;
        let mut tmp = NamedTempFile::new_in(db_dir)?;
        tmp.write_all(&json)?;
        tmp.as_file().sync_all()?;
        let path = Self::path(db_dir);
        let f = tmp.persist(&path).map_err(|e| SochDBError::Io(e.error))?;
        f.sync_all()?;
        fsync_dir(db_dir);
        Ok(())
    }
}

/// fsync the directory so the rename is durable. Best-effort off Unix (opening a
/// directory handle isn't supported there).
fn fsync_dir(db_dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(dir) = fs::File::open(db_dir) {
            let _ = dir.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = db_dir;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_and_exists() {
        let dir = tempdir().unwrap();
        assert!(!WalManifest::exists(dir.path()));
        let m = WalManifest::new(42);
        m.write_atomic(dir.path()).unwrap();
        assert!(WalManifest::exists(dir.path()));
        let loaded = WalManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.last_checkpoint_lsn, 42);
        assert_eq!(loaded.db_uuid, m.db_uuid);
        assert_eq!(loaded.format_version, WAL_MANIFEST_FORMAT_VERSION);
    }

    #[test]
    fn overwrite_advances_lsn_and_keeps_uuid() {
        let dir = tempdir().unwrap();
        let mut m = WalManifest::new(10);
        m.write_atomic(dir.path()).unwrap();
        let uuid = m.db_uuid.clone();
        m.last_checkpoint_lsn = 100;
        m.write_atomic(dir.path()).unwrap();
        let loaded = WalManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.last_checkpoint_lsn, 100);
        assert_eq!(
            loaded.db_uuid, uuid,
            "db identity must be stable across writes"
        );
    }

    #[test]
    fn a_torn_temp_does_not_corrupt_the_committed_manifest() {
        let dir = tempdir().unwrap();
        WalManifest::new(7).write_atomic(dir.path()).unwrap();
        // Simulate a torn in-progress write: a stray *.tmp left in the dir.
        std::fs::write(dir.path().join("stray.tmp"), b"{ partial").unwrap();
        // The committed manifest is still intact and parseable.
        let loaded = WalManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.last_checkpoint_lsn, 7);
    }

    #[test]
    fn load_rejects_garbage() {
        let dir = tempdir().unwrap();
        std::fs::write(WalManifest::path(dir.path()), b"not json").unwrap();
        assert!(WalManifest::load(dir.path()).is_err());
    }
}
