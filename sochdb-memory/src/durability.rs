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

use crate::episode::Episode;
use crate::fact::{FactEdge, FactId};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sochdb_storage::{TxnWal, TxnWalEntry};
use std::path::Path;

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
            poison: Mutex::new(None),
        })
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
            Ok(()) => Ok(()),
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
    pub(crate) fn replay<F>(&self, mut apply: F) -> Result<u64, String>
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
