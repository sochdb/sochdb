//! Fault injection for the durability layer.
//!
//! The existing simulator ([`sochdb-simulation`]) models *performance*: it
//! answers "how fast, at what scale" against analytic component models. It
//! deliberately does not link the real crates, so it cannot answer the question
//! this module exists for — "what does the store do when the disk says no".
//!
//! Durability code is uniquely resistant to ordinary testing because its
//! correctness claims are all about states that only appear when something
//! fails. A checkpoint that works when nothing goes wrong tells you nothing:
//! the interesting states are the ones where the process died between two
//! syscalls, and no amount of happy-path testing visits them. The helpers here
//! produce those states directly, on disk, so a test can assert on what an
//! operator would actually find after a crash.
//!
//! # Why these helpers manipulate files rather than hooking the code
//!
//! A fault-injection hook inside [`crate::durability`] would prove that the
//! code handles the faults *the hook knows how to express*, which is a weaker
//! and more circular claim: the hook and the code are written by the same
//! person with the same mental model, so a state neither of them imagined stays
//! untested. Reconstructing the on-disk state instead means the test asserts
//! against the artifact recovery actually consumes, and stays honest even if
//! the checkpoint implementation is rewritten.
//!
//! Available only under `cfg(test)` or the `fault-injection` feature, so none
//! of it can be reached from a production build.

use std::fs;
use std::path::{Path, PathBuf};

/// Log file name, mirrored from [`crate::durability`].
const WAL_FILE: &str = "memory.wal";
/// Snapshot file name, mirrored from [`crate::durability`].
const SNAPSHOT_FILE: &str = "memory.snapshot";
/// Snapshot staging file name, mirrored from [`crate::durability`].
const SNAPSHOT_TMP: &str = "memory.snapshot.tmp";

/// Where a checkpoint was interrupted.
///
/// A checkpoint is a five-step sequence — write the staging file, `fsync` it,
/// rename it over the live snapshot, `fsync` the directory, truncate the log —
/// and a crash can land in any gap between them. The variants are the gaps that
/// leave *distinguishable* state on disk; the `fsync` steps change durability,
/// not content, so they collapse into their neighbours.
///
/// Each variant must recover to exactly the same live state. That is the whole
/// correctness argument for compaction, and it is what
/// [`crate::tests`] asserts against every variant of this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointCrash {
    /// Died while writing the staging file.
    ///
    /// The live snapshot is whatever it was before, the log is untouched and
    /// complete, and a partial `.tmp` is left behind. Recovery must ignore the
    /// `.tmp` entirely: nothing has committed to it, and it is not the name
    /// recovery reads.
    BeforeRename,
    /// Died after the rename, before the log was truncated.
    ///
    /// The dangerous one, and the *common* one, because the window spans an
    /// `fsync`. The new snapshot is live while the log still holds every record
    /// the snapshot already contains, so recovery sees each of those records
    /// twice and must apply them once.
    AfterRename,
    /// Died after the truncate — i.e. the checkpoint actually completed.
    ///
    /// Present so the "crash" matrix includes the successful case and proves
    /// the other variants are being compared against the right answer.
    AfterTruncate,
}

impl CheckpointCrash {
    /// Every distinguishable interruption, for exhaustive test matrices.
    pub fn all() -> [CheckpointCrash; 3] {
        [Self::BeforeRename, Self::AfterRename, Self::AfterTruncate]
    }
}

/// A data directory whose contents can be captured and rewound.
///
/// Recreating a crash means restoring bytes that a successful checkpoint has
/// already destroyed, so the state has to be captured *before* the checkpoint
/// runs and put back afterwards.
pub struct DurabilityFaults {
    dir: PathBuf,
    saved_log: Option<Vec<u8>>,
    saved_snapshot: Option<Vec<u8>>,
}

impl DurabilityFaults {
    /// Capture the current on-disk state of `dir`.
    pub fn capture(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            saved_log: fs::read(dir.join(WAL_FILE)).ok(),
            saved_snapshot: fs::read(dir.join(SNAPSHOT_FILE)).ok(),
        }
    }

    /// Rewind the directory to the state a checkpoint interrupted at `crash`
    /// would have left.
    ///
    /// Call after running a real, successful checkpoint: the completed
    /// checkpoint supplies the *new* snapshot, and the captured state supplies
    /// the log and old snapshot it destroyed. Assembling the result from both
    /// keeps the snapshot a genuine product of the real implementation rather
    /// than a hand-written forgery that could drift from it.
    pub fn rewind_to(&self, crash: CheckpointCrash) {
        let log = self.dir.join(WAL_FILE);
        let snapshot = self.dir.join(SNAPSHOT_FILE);
        let tmp = self.dir.join(SNAPSHOT_TMP);

        match crash {
            CheckpointCrash::BeforeRename => {
                // The rename never happened: restore the previous snapshot (or
                // remove it, if this was the first checkpoint), put the full log
                // back, and leave behind the half-written staging file a crash
                // mid-write would produce.
                let partial = fs::read(&snapshot)
                    .map(|b| b[..b.len() / 2].to_vec())
                    .unwrap_or_default();
                match &self.saved_snapshot {
                    Some(bytes) => fs::write(&snapshot, bytes).unwrap(),
                    None => {
                        let _ = fs::remove_file(&snapshot);
                    }
                }
                fs::write(&tmp, partial).unwrap();
                self.restore_log(&log);
            }
            CheckpointCrash::AfterRename => {
                // New snapshot stays; the log it subsumed comes back.
                self.restore_log(&log);
            }
            CheckpointCrash::AfterTruncate => {}
        }
    }

    fn restore_log(&self, log: &Path) {
        match &self.saved_log {
            Some(bytes) => fs::write(log, bytes).unwrap(),
            None => {
                let _ = fs::remove_file(log);
            }
        }
    }

    /// Append `garbage` to the log, as a torn write would.
    ///
    /// A process killed mid-append leaves a record whose length header promises
    /// more bytes than the file contains. Recovery must stop cleanly at it
    /// rather than failing: the record was never acknowledged to a client.
    pub fn tear_log_tail(&self, garbage: &[u8]) {
        let log = self.dir.join(WAL_FILE);
        let mut bytes = fs::read(&log).unwrap_or_default();
        bytes.extend_from_slice(garbage);
        fs::write(&log, bytes).unwrap();
    }

    /// Replace the snapshot's last line with something undecodable.
    ///
    /// Unlike a torn log tail this is *not* recoverable by ignoring it. A
    /// snapshot line is the only surviving copy of live state — the log that
    /// carried it has been truncated — so silently skipping it would delete an
    /// episode while reporting a successful recovery. The store must refuse to
    /// open instead.
    pub fn corrupt_snapshot_tail(&self) {
        let path = self.dir.join(SNAPSHOT_FILE);
        let text = fs::read_to_string(&path).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        assert!(!lines.is_empty(), "snapshot must be non-empty to corrupt");
        lines.pop();
        let mut out = lines.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("{\"Episode\":{\"id\":\"not-a-number\"}}\n");
        fs::write(&path, out).unwrap();
    }

    /// Point the log at `/dev/full` so every write to it returns `ENOSPC`.
    ///
    /// The one fault-injection primitive that needs no privileges, no
    /// filesystem setup and no code changes: `/dev/full` accepts `open`, reports
    /// a successful `write` of zero bytes, and fails `flush` with `ENOSPC`,
    /// which is precisely the shape of a full disk. Attempts to inject the same
    /// fault with file permissions are vacuous here, because opening the log
    /// fails outright and the code under test never reaches its write path.
    ///
    /// Returns `false` if the fault cannot be injected on this platform, so a
    /// caller can skip rather than fail. `/dev/full` is Unix-only, and the
    /// symlink call itself does not exist elsewhere.
    #[cfg(unix)]
    pub fn fill_disk(&self) -> bool {
        if !Path::new("/dev/full").exists() {
            return false;
        }
        let log = self.dir.join(WAL_FILE);
        let _ = fs::remove_file(&log);
        std::os::unix::fs::symlink("/dev/full", &log).is_ok()
    }

    /// Always `false` off Unix: there is no `/dev/full` to point at.
    #[cfg(not(unix))]
    pub fn fill_disk(&self) -> bool {
        false
    }

    /// Whether a staging file was left behind.
    pub fn has_staging_file(&self) -> bool {
        self.dir.join(SNAPSHOT_TMP).exists()
    }

    /// Size of the log in bytes; `0` if it does not exist.
    pub fn log_len(&self) -> u64 {
        fs::metadata(self.dir.join(WAL_FILE))
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Size of the snapshot in bytes; `0` if it does not exist.
    pub fn snapshot_len(&self) -> u64 {
        fs::metadata(self.dir.join(SNAPSHOT_FILE))
            .map(|m| m.len())
            .unwrap_or(0)
    }
}
