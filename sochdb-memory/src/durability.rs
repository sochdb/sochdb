//! Write-ahead logging for agent memory.
//!
//! Before this module existed, [`crate::MemoryStore`] accepted a `data_dir`
//! argument and threw it away: every episode, fact and index lived only in RAM,
//! so a restart of the gRPC server silently discarded the entire memory of
//! every agent it served. The crate's own module docs advertised
//! `episode write -> WAL -> lexical index`, and the WAL step did not exist.
//!
//! # What is logged, and what is not
//!
//! Only *source* state is logged: episodes as written, facts as asserted, and
//! fact invalidations. The BM25 postings, trigram postings and embeddings are
//! deterministic functions of episode text, so logging them would multiply
//! write amplification by the size of three indexes and — worse — create a
//! second copy of derived state that can disagree with the indexes it was
//! derived from. Recovery rebuilds them by replaying the source through the
//! *same* code path a live write takes, which makes a recovered store
//! byte-identical to one that never crashed rather than merely similar.
//!
//! Embeddings are the exception worth naming: they are not rebuilt inline,
//! because doing so would block recovery on one model call per episode. They
//! are re-queued for the enrichment daemon instead, which is exactly the state
//! a freshly written episode is in.
//!
//! # Why the log does not grow forever
//!
//! A log that is only ever appended to ends at `ENOSPC`, and until then makes
//! every restart slower than the last, because replay is linear in everything
//! that ever happened rather than in what is currently true. [`MemoryWal::checkpoint`]
//! bounds both: it writes the live state to a snapshot and drops the log that
//! state subsumes, so the log holds only what has happened *since* the last
//! checkpoint and startup cost tracks the size of memory instead of its age.

use crate::episode::Episode;
use crate::fact::{FactEdge, FactId};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sochdb_storage::{TxnWal, TxnWalEntry};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// How much of a crash an acknowledged write is expected to survive.
///
/// The tiers are distinguished by *which* buffer the record has reached when
/// `write_episode` returns, because that is what decides what a given failure
/// mode can destroy. Naming them after the failure they survive keeps callers
/// from assuming a cheap tier protects them from an expensive fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    /// No log is written. Everything is lost on exit, however clean.
    ///
    /// This is the default because it is the behaviour every existing caller
    /// already has, and because agent memory is frequently used as a cache in
    /// front of a system of record that does not need a second one.
    #[default]
    None,

    /// The record reaches the kernel before the write is acknowledged.
    ///
    /// Survives the process dying — panic, `kill -9`, OOM killer — because the
    /// page cache outlives it. Does *not* survive the machine losing power,
    /// since no `fsync` has been issued. Costs one `write` syscall per episode.
    Buffered,

    /// The record reaches the disk before the write is acknowledged.
    ///
    /// Survives power loss and kernel panic. Costs one `fsync` per episode,
    /// which is three to four orders of magnitude more expensive than
    /// [`Durability::Buffered`] on rotational or network storage.
    Sync,
}

impl Durability {
    /// Whether this tier writes a log at all.
    pub fn is_logged(self) -> bool {
        self != Durability::None
    }
}

/// One replayable mutation of durable memory state.
///
/// Tagged by serde so that a log written by an older build, which did not know
/// about a later variant, still deserializes every variant it *did* write —
/// a positional encoding would silently reinterpret the fields instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum MemoryRecord {
    /// An episode exactly as it was admitted, including its assigned id.
    ///
    /// The id is logged rather than re-derived so that replay reproduces the
    /// ids the client was handed. A client that stored `ep:7` must still find
    /// `ep:7` after recovery.
    Episode(Box<Episode>),
    FactAdded {
        namespace: String,
        fact: Box<FactEdge>,
    },
    FactInvalidated {
        namespace: String,
        fact_id: FactId,
        t_invalid: u64,
    },
}

/// The memory store's write-ahead log.
///
/// Wraps [`TxnWal`] to hide two things callers should not have to think about:
/// that memory records are not transactions (they carry no txn id, and each one
/// is independently complete), and that the durability tier decides which of
/// `TxnWal`'s three differently-durable append methods is correct.
pub(crate) struct MemoryWal {
    wal: TxnWal,
    durability: Durability,
    data_dir: PathBuf,
    /// Records appended since the log was last truncated by a checkpoint.
    ///
    /// Drives the decision to checkpoint at all. Counting records rather than
    /// bytes keeps the trigger independent of how large episodes happen to be,
    /// which is what makes a single default meaningful across deployments.
    since_checkpoint: AtomicU64,
    /// The first append error, if one has happened.
    ///
    /// A failed append is *indeterminate*, not failed: [`TxnWal`] frames the
    /// record into a `BufWriter` and flushes separately, and `BufWriter`
    /// retains bytes it could not write, so a record whose flush returned
    /// `ENOSPC` can still reach the disk behind a later successful flush. The
    /// caller has been told the write failed while the log may yet contain it.
    ///
    /// From that instant the log and memory disagree and nothing can reconcile
    /// them in-process, so the store stops accepting writes rather than
    /// layering more divergence on top of it. Failing every subsequent write
    /// with the original error is what turns a silent, unbounded drift into one
    /// loud, diagnosable event.
    poison: Mutex<Option<String>>,
}

/// Memory records are not transactional, so they all share this id.
///
/// [`TxnWal`] frames every record with a transaction id because its primary
/// user is the transactional storage engine. Agent memory has no multi-record
/// atomicity to express — each record is independently meaningful and
/// independently replayable — so inventing per-record ids would imply a
/// grouping guarantee that nothing here provides.
const NON_TRANSACTIONAL: u64 = 0;

/// Log file name inside the caller's data directory.
const WAL_FILE: &str = "memory.wal";

/// Compacted state as of the last checkpoint: newline-delimited [`MemoryRecord`].
///
/// Deliberately the *same* record stream the log carries, rather than a second
/// serialization format describing the same state. Recovery therefore replays
/// snapshot and log through one code path, and a bug in reading back compacted
/// state is a bug in reading back logged state — which the log's own tests
/// already exercise on every run.
const SNAPSHOT_FILE: &str = "memory.snapshot";

/// Where a snapshot is assembled before it is atomically renamed into place.
const SNAPSHOT_TMP: &str = "memory.snapshot.tmp";

/// `fsync` a directory so a rename into it is durable.
///
/// A rename is a directory modification, and on POSIX it is not guaranteed to
/// survive power loss until the *directory* is synced. Skipping this is the one
/// ordering that loses acknowledged data: the log can be truncated while the
/// rename that replaced it is still only in the page cache, leaving neither a
/// current snapshot nor the records it was built from.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> Result<(), String> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| format!("syncing {}: {e}", dir.display()))
}

/// No-op off Unix, because there is no portable way to do it.
///
/// `File::open` on a directory fails unconditionally on Windows — `CreateFileW`
/// rejects a directory handle without `FILE_FLAG_BACKUP_SEMANTICS`, which
/// `std::fs::File` never sets — so calling it here would make *every* checkpoint
/// fail, at the worst possible point: after the snapshot has been renamed into
/// place but before the log is truncated. The log would then grow without bound
/// while every attempt to compact it reported an error, which is precisely the
/// failure checkpointing exists to prevent.
///
/// The cost of the no-op is narrower than that failure: a Windows host that
/// loses power in the window between the rename and the truncate may come back
/// with the previous snapshot instead of the new one. Recovery still succeeds —
/// the log has not been truncated at that point, so the older snapshot plus the
/// full log reconstruct the same state.
#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> Result<(), String> {
    Ok(())
}

impl MemoryWal {
    /// Open (or create) the log under `data_dir`.
    ///
    /// The directory, not the file, is the caller-facing unit so that snapshots
    /// and any future sidecar state land beside the log rather than scattered
    /// wherever the caller happened to point.
    pub(crate) fn open(data_dir: &Path, durability: Durability) -> Result<Self, String> {
        let path = data_dir.join(WAL_FILE);
        let wal = TxnWal::new(&path)
            .map_err(|e| format!("opening memory WAL at {}: {e}", path.display()))?;
        Ok(Self {
            wal,
            durability,
            data_dir: data_dir.to_path_buf(),
            since_checkpoint: AtomicU64::new(0),
            poison: Mutex::new(None),
        })
    }

    /// Whether the log has latched into fail-stop after an append error.
    ///
    /// Exposed so callers can decline expensive work that is certain to fail:
    /// building a checkpoint's record list deep-clones every episode and fact
    /// in the store, and there is no point paying that to reach an error the
    /// log can report immediately.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poison.lock().is_some()
    }

    /// Records appended since the last successful checkpoint.
    pub(crate) fn since_checkpoint(&self) -> u64 {
        self.since_checkpoint.load(Ordering::Relaxed)
    }

    /// Append one record, returning only once it has reached the medium this
    /// log's [`Durability`] tier promises.
    ///
    /// Returns the *first* error this log ever saw, not the latest, once it has
    /// been poisoned — see [`MemoryWal::poison`].
    pub(crate) fn append(&self, record: &MemoryRecord) -> Result<(), String> {
        if let Some(first) = self.poison.lock().as_ref() {
            return Err(format!(
                "memory WAL is no longer accepting writes after an earlier failure: {first}"
            ));
        }

        match self.append_inner(record) {
            Ok(()) => {
                self.since_checkpoint.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                let mut poison = self.poison.lock();
                if poison.is_none() {
                    tracing::error!(
                        error = %e,
                        "memory WAL append failed; refusing further writes because the log \
                         and memory may now disagree"
                    );
                    *poison = Some(e.clone());
                }
                Err(e)
            }
        }
    }

    fn append_inner(&self, record: &MemoryRecord) -> Result<(), String> {
        let value =
            serde_json::to_vec(record).map_err(|e| format!("encoding memory record: {e}"))?;
        let entry = TxnWalEntry::data(NON_TRANSACTIONAL, Vec::new(), value);

        match self.durability {
            // `None` is unreachable by construction — `MemoryStore` builds no
            // `MemoryWal` for it at all — but is handled as `Buffered` rather
            // than dropped, because silently discarding a record that was
            // handed to a log is the worse way to be wrong about this.
            Durability::None | Durability::Buffered => {
                self.wal
                    .append(&entry)
                    .map_err(|e| format!("appending memory record: {e}"))?;
                self.wal
                    .flush()
                    .map_err(|e| format!("flushing memory WAL: {e}"))?;
            }
            Durability::Sync => {
                self.wal
                    .append_sync(&entry)
                    .map_err(|e| format!("syncing memory record: {e}"))?;
            }
        }
        Ok(())
    }

    /// Force everything written so far all the way to disk.
    ///
    /// Lets a [`Durability::Buffered`] store take a durability point on demand —
    /// before a planned shutdown, say — without paying `fsync` per episode.
    pub(crate) fn sync(&self) -> Result<(), String> {
        self.wal
            .sync()
            .map_err(|e| format!("syncing memory WAL: {e}"))
    }

    /// Replay the snapshot, then the log, oldest first.
    ///
    /// Snapshot before log is the only correct order: the snapshot is the
    /// compacted past and the log is what happened after it. Both are fed to
    /// the same `apply`, and both may describe the same record, because a
    /// checkpoint renames the snapshot into place before truncating the log and
    /// a crash in between leaves the overlap on disk. Callers make replay
    /// idempotent rather than trying to compute the overlap, which cannot be
    /// done without a sequence number the records do not carry.
    pub(crate) fn replay<F>(&self, mut apply: F) -> Result<u64, String>
    where
        F: FnMut(MemoryRecord),
    {
        let mut total = self.replay_snapshot(&mut apply)?;
        total += self.replay_log(&mut apply)?;
        Ok(total)
    }

    /// Replay the compacted snapshot, if one exists.
    fn replay_snapshot<F>(&self, apply: &mut F) -> Result<u64, String>
    where
        F: FnMut(MemoryRecord),
    {
        let path = self.data_dir.join(SNAPSHOT_FILE);
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(format!("opening {}: {e}", path.display())),
        };

        let mut count = 0u64;
        for (line_no, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("reading {}: {e}", path.display()))?;
            if line.is_empty() {
                continue;
            }
            let record: MemoryRecord = serde_json::from_str(&line).map_err(|e| {
                format!(
                    "{} line {} could not be decoded ({e}); this build does not \
                     understand the snapshot it was given",
                    path.display(),
                    line_no + 1
                )
            })?;
            apply(record);
            count += 1;
        }
        Ok(count)
    }

    /// Replay every record in the log, oldest first.
    ///
    /// A torn trailing record — the normal result of crashing mid-append — ends
    /// replay cleanly rather than failing recovery, because a record that was
    /// never completely written was also never acknowledged to a client. Every
    /// record before it is protected by a CRC32 that [`TxnWal`] verifies.
    ///
    /// A record that passes that CRC but cannot be *decoded* is a different
    /// thing entirely, and fails recovery loudly. It is intact on disk, so it
    /// is not corruption: it is a schema this build does not understand, which
    /// in practice means a downgrade. Skipping it would leave `next_episode_id`
    /// behind the ids already in the log, so the store would re-issue live ids
    /// — two episodes named `ep:7`, double-counted BM25 corpus statistics, and
    /// a namespace that is quietly wrong rather than loudly unavailable.
    fn replay_log<F>(&self, apply: &mut F) -> Result<u64, String>
    where
        F: FnMut(MemoryRecord),
    {
        let mut decoded = 0u64;
        let mut undecodable: Option<String> = None;

        self.wal
            .replay(|entry| {
                if undecodable.is_some() {
                    return Ok(());
                }
                match serde_json::from_slice::<MemoryRecord>(&entry.value) {
                    Ok(record) => {
                        apply(record);
                        decoded += 1;
                    }
                    Err(e) => {
                        undecodable = Some(format!(
                            "record {} is intact on disk but could not be decoded ({e}); \
                             this build does not understand the log it was given",
                            decoded + 1
                        ));
                    }
                }
                Ok(())
            })
            .map_err(|e| format!("replaying memory WAL: {e}"))?;

        match undecodable {
            Some(why) => Err(why),
            None => Ok(decoded),
        }
    }

    /// Replace the snapshot with `records` and drop the log they subsume.
    ///
    /// The caller must guarantee that no write is between its log append and
    /// its in-memory publish for the duration of this call, or the snapshot can
    /// miss a record that truncation then destroys.
    ///
    /// # Ordering
    ///
    /// The sequence is: write the new snapshot to a temporary file, `fsync` it,
    /// rename it over the old one, `fsync` the directory, and only then
    /// truncate the log. Every prefix of that sequence is recoverable:
    ///
    /// - crash before the rename — old snapshot plus the untouched log; the
    ///   partial temporary file is ignored because nothing reads that name
    /// - crash after the rename, before the truncate — new snapshot plus a log
    ///   that still holds records the snapshot already contains, which is why
    ///   replay must be idempotent
    /// - crash after the truncate — new snapshot plus an empty log
    ///
    /// The directory `fsync` is not optional: without it the rename itself can
    /// be lost on power failure while the truncated log is not, which is the
    /// one ordering that loses acknowledged data.
    pub(crate) fn checkpoint(&self, records: &[MemoryRecord]) -> Result<u64, String> {
        if let Some(first) = self.poison.lock().as_ref() {
            return Err(format!(
                "memory WAL is no longer accepting writes after an earlier failure: {first}"
            ));
        }

        let tmp = self.data_dir.join(SNAPSHOT_TMP);
        let final_path = self.data_dir.join(SNAPSHOT_FILE);

        {
            let file =
                File::create(&tmp).map_err(|e| format!("creating {}: {e}", tmp.display()))?;
            let mut out = BufWriter::new(file);
            for record in records {
                serde_json::to_writer(&mut out, record)
                    .map_err(|e| format!("encoding snapshot record: {e}"))?;
                out.write_all(b"\n")
                    .map_err(|e| format!("writing {}: {e}", tmp.display()))?;
            }
            out.flush()
                .map_err(|e| format!("flushing {}: {e}", tmp.display()))?;
            out.get_ref()
                .sync_all()
                .map_err(|e| format!("syncing {}: {e}", tmp.display()))?;
        }

        std::fs::rename(&tmp, &final_path).map_err(|e| {
            format!(
                "renaming {} to {}: {e}",
                tmp.display(),
                final_path.display()
            )
        })?;

        sync_dir(&self.data_dir)?;
        self.wal
            .truncate()
            .map_err(|e| format!("truncating memory WAL: {e}"))?;
        self.since_checkpoint.store(0, Ordering::Relaxed);

        Ok(records.len() as u64)
    }
}

/// Flush on drop so an orderly shutdown never loses a buffered record.
///
/// Without this, a [`Durability::Buffered`] store that exits cleanly could
/// still lose writes sitting in the log's userspace buffer — the one failure
/// mode a user who enabled durability would never expect to be exposed to.
impl Drop for MemoryWal {
    fn drop(&mut self) {
        if let Err(e) = self.wal.sync() {
            // Nothing can be returned from `drop`, but staying silent here
            // means a shutdown that lost buffered records looks identical to
            // one that did not.
            tracing::error!(error = %e, "final memory WAL sync failed; buffered records may be lost");
        }
    }
}
