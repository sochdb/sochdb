use crate::durability::{Durability, MemoryRecord, MemoryWal};
use crate::enrichment::{EnrichmentJob, EnrichmentQueue};
use crate::episode::{ConversationTurn, Episode, EpisodeId, EpisodeWrite};
use crate::fact::{FactEdge, FactId};
use parking_lot::RwLock;
use sochdb_query::{EmbeddingProvider, MockEmbeddingProvider, trigram_index::TrigramIndex};
use sochdb_storage::hlc::HybridLogicalClock;
use sochdb_vector::bm25::BM25Config;
use sochdb_vector::inverted_index::InvertedIndex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("namespace not found: {0}")]
    NamespaceNotFound(String),
    #[error("episode not found: {0}")]
    EpisodeNotFound(u64),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("durability error: {0}")]
    Durability(String),
}

pub type MemoryResult<T> = Result<T, MemoryError>;

#[derive(Debug, Clone)]
pub struct MemoryStoreConfig {
    pub max_enrichment_queue: usize,
    /// Run embedding + HNSW insert synchronously on write (bench/tests).
    pub enrich_on_write: bool,
    /// What an acknowledged write is expected to survive. See [`Durability`].
    ///
    /// Has no effect unless the store is opened with a data directory: there is
    /// nowhere to log to otherwise, and [`MemoryStore::new`] rejects that
    /// combination rather than quietly downgrading it.
    pub durability: Durability,
    /// Log records after which a checkpoint is due; `0` disables automatic ones.
    ///
    /// The bound that keeps the log from growing for the lifetime of the
    /// deployment. Too low and the store pays a stop-the-world snapshot of live
    /// state for a handful of episodes; too high and the log — and with it,
    /// restart time — grows further than an operator would accept. The default
    /// is deliberately large enough that a checkpoint is a rare event and small
    /// enough that replay stays in the low seconds.
    ///
    /// Checkpoints are driven by the lifecycle daemon rather than by the write
    /// path, so crossing this threshold costs an arbitrary unlucky request
    /// nothing.
    pub checkpoint_after_records: u64,
}

impl Default for MemoryStoreConfig {
    fn default() -> Self {
        Self {
            max_enrichment_queue: 10_000,
            enrich_on_write: false,
            durability: Durability::None,
            checkpoint_after_records: 50_000,
        }
    }
}

/// What a store found in its log when it started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub episodes: u64,
    pub facts: u64,
    pub invalidations: u64,
    /// Episodes re-queued for embedding because vectors are not logged.
    pub enrichment_requeued: u64,
}

impl RecoveryReport {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct WriteResult {
    pub episode_id: EpisodeId,
    pub t_created: u64,
    pub lexical_indexed: bool,
    pub ingestion_lag_us: u64,
    pub enrichment_queued: bool,
}

pub(crate) struct NamespaceIndexes {
    pub(crate) bm25: InvertedIndex,
    pub(crate) trigram: TrigramIndex,
    pub(crate) vectors: HashMap<u64, Vec<f32>>,
    pub(crate) episodes: HashMap<u64, Episode>,
    facts: Vec<FactEdge>,
    next_episode_id: u64,
    next_fact_id: u64,
}

impl NamespaceIndexes {
    fn new() -> Self {
        Self {
            bm25: InvertedIndex::new(BM25Config::default()),
            trigram: TrigramIndex::new(),
            vectors: HashMap::new(),
            episodes: HashMap::new(),
            facts: Vec::new(),
            next_episode_id: 1,
            next_fact_id: 1,
        }
    }

    /// Make an episode retrievable on the lexical lanes.
    ///
    /// The single place indexes are populated from an episode, so that a
    /// replayed episode is indexed by exactly the code that indexes a live one.
    /// When recovery indexes through a second, parallel implementation, the two
    /// drift, and the resulting store is subtly unlike the one that crashed.
    ///
    /// Idempotent by id. A checkpoint renames a new snapshot into place and
    /// *then* truncates the log, so a crash between those two steps leaves
    /// records present in both — recovery must be able to see an episode twice
    /// and produce the same store. Without this guard the second sighting would
    /// call [`InvertedIndex::add_document_with_id`] again for a doc id it
    /// already holds, which overwrites that document's postings while counting
    /// its length into the corpus a second time: permanently skewed avgdl and
    /// IDF for every query against that namespace.
    ///
    /// Returns whether the episode was newly inserted, so recovery can avoid
    /// queueing a second embedding — and therefore a second HNSW vector — for
    /// an episode it encounters in both the snapshot and the log.
    fn publish_episode(&mut self, episode: Episode) -> bool {
        let doc_id = episode.id.0;
        self.next_episode_id = self.next_episode_id.max(doc_id + 1);
        if self.episodes.contains_key(&doc_id) {
            return false;
        }
        self.bm25.add_document_with_id(doc_id, &episode.text);
        self.trigram.insert(doc_id, &episode.text);
        self.episodes.insert(doc_id, episode);
        true
    }

    /// Restore a fact under the id it was originally given.
    ///
    /// `next_fact_id` is advanced past it so a post-recovery assertion cannot
    /// reuse an id that a client already holds a reference to.
    ///
    /// Idempotent by id, for the same reason as [`Self::publish_episode`] — and
    /// with an extra consequence: a snapshot stores each fact in its *current*
    /// state, so a fact the snapshot already shows as retracted must not be
    /// resurrected by the `FactAdded` record still sitting in the log.
    ///
    /// Returns whether the fact was newly inserted.
    fn publish_fact(&mut self, fact: FactEdge) -> bool {
        self.next_fact_id = self.next_fact_id.max(fact.id.0 + 1);
        if self.facts.iter().any(|f| f.id == fact.id) {
            return false;
        }
        self.facts.push(fact);
        true
    }
}

/// Independently owned mutable state for one namespace.
pub(crate) type NamespaceHandle = Arc<RwLock<NamespaceIndexes>>;

/// Agent memory store: write-time lexical recall + async enrichment queue.
pub struct MemoryStore {
    hlc: HybridLogicalClock,
    /// Directory of namespaces — *lookup and creation only*.
    ///
    /// Each namespace owns its indexes behind its own lock, so this outer lock
    /// is held just long enough for a hash lookup and an `Arc` clone, never
    /// across index mutation. Holding one store-wide write guard while updating
    /// BM25, trigram and episode state made every agent's write serialise
    /// against every other agent's write, and block every reader, over state
    /// they do not share: namespaces have nothing in common but this map.
    pub(crate) namespaces: RwLock<HashMap<String, NamespaceHandle>>,
    pub(crate) enrichment: EnrichmentQueue,
    pub(crate) embedder: Arc<dyn EmbeddingProvider>,
    /// Present exactly when this store was opened with a data directory.
    ///
    /// `Option` rather than a no-op sink so that the un-logged path — which is
    /// the default and the hot one — costs a null check rather than a virtual
    /// call and a discarded serialization.
    wal: Option<MemoryWal>,
    /// Held for reading by writers, for writing by [`Self::checkpoint`].
    ///
    /// A checkpoint snapshots memory and then discards the log that memory
    /// subsumes, so it must not run while any writer sits between its log
    /// append and its in-memory publish: that writer's record is in the log the
    /// checkpoint is about to delete, and its state is not yet in the memory
    /// the checkpoint is reading. Truncation would destroy the only remaining
    /// copy of an acknowledged write.
    ///
    /// Guarding the pair rather than each half is the whole point, so this
    /// cannot be folded into the per-namespace locks — a checkpoint spans every
    /// namespace, and the gap it has to exclude spans two separate critical
    /// sections. Writers contend only with checkpoints, never with each other:
    /// an uncontended `parking_lot` read acquisition is tens of nanoseconds
    /// against a checkpoint measured in milliseconds.
    checkpoint_lock: RwLock<()>,
    /// What replay found at startup. Empty for a store that was not logged.
    recovery: RecoveryReport,
    config: MemoryStoreConfig,
}

fn default_embedder() -> Arc<dyn EmbeddingProvider> {
    Arc::new(MockEmbeddingProvider::new(384))
}

impl MemoryStore {
    /// Open a store, recovering any memory left behind by a previous process.
    ///
    /// Passing `Some(dir)` opens a write-ahead log under `dir` and replays it;
    /// passing `None` builds a purely in-memory store. The two are separated by
    /// an argument rather than by two constructors because the failure this
    /// guards against is a caller who *believes* they passed a directory.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Durability`] if `data_dir` is `None` while
    /// `config.durability` asks for a log. That combination is unsatisfiable,
    /// and the previous behaviour — accepting it and silently keeping
    /// everything in RAM — is precisely how an operator ends up believing their
    /// agents' memory is being persisted when it is not.
    pub fn new(data_dir: Option<&Path>, config: MemoryStoreConfig) -> MemoryResult<Self> {
        Self::with_embedder(data_dir, config, default_embedder())
    }

    pub fn with_embedder(
        data_dir: Option<&Path>,
        config: MemoryStoreConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> MemoryResult<Self> {
        // Durability is the switch; `data_dir` only says where. A directory
        // supplied alongside `Durability::None` is deliberately left unused
        // rather than rejected, so a deployment can toggle logging off through
        // one config field without also having to unset its path. The reverse —
        // asking for durability with nowhere to put it — is rejected, because
        // that is the direction in which silence loses data.
        let wal = match (data_dir, config.durability.is_logged()) {
            (Some(dir), true) => {
                std::fs::create_dir_all(dir)?;
                Some(MemoryWal::open(dir, config.durability).map_err(MemoryError::Durability)?)
            }
            (None, true) => {
                return Err(MemoryError::Durability(format!(
                    "durability {:?} requires a data directory, but none was given",
                    config.durability
                )));
            }
            (_, false) => None,
        };

        let mut store = Self {
            hlc: HybridLogicalClock::new(),
            namespaces: RwLock::new(HashMap::new()),
            enrichment: EnrichmentQueue::new(config.max_enrichment_queue),
            embedder,
            wal,
            checkpoint_lock: RwLock::new(()),
            recovery: RecoveryReport::default(),
            config,
        };
        store.recovery = store.recover()?;
        Ok(store)
    }

    /// In-memory store with default settings. Cannot fail: nothing is opened.
    pub fn with_defaults() -> Self {
        Self::new(None, MemoryStoreConfig::default())
            .expect("an in-memory store with default config opens no files and cannot fail")
    }

    /// Build from the environment: embedder, data directory and durability.
    ///
    /// - `SOCHDB_EMBEDDER` — e.g. `fastembed:bge-small-en`, or `mock`/unset for
    ///   the default mock embedder. See
    ///   [`sochdb_query::embedding_provider::embedder_from_env`].
    /// - `SOCHDB_MEMORY_DIR` — directory for the write-ahead log. Unset means
    ///   memory is not persisted and is lost on restart.
    /// - `SOCHDB_MEMORY_DURABILITY` — `none`, `buffered` (default when a
    ///   directory is set) or `sync`.
    /// - `SOCHDB_MEMORY_CHECKPOINT_RECORDS` — log records after which the
    ///   lifecycle daemon compacts the log; `0` means only on request. Reachable
    ///   from the environment because the right value depends on how large the
    ///   deployment's episodes are and how long a restart is allowed to take,
    ///   neither of which this crate can know.
    ///
    /// # Errors
    ///
    /// Propagates any failure to open or replay the log. A server that cannot
    /// read the memory it was told to persist must not come up pretending to
    /// have none: that silently presents every agent with a blank history.
    pub fn from_env() -> MemoryResult<Self> {
        let data_dir = std::env::var_os("SOCHDB_MEMORY_DIR").map(PathBuf::from);
        let requested = std::env::var("SOCHDB_MEMORY_DURABILITY").unwrap_or_default();
        let durability = match requested.trim().to_ascii_lowercase().as_str() {
            "sync" => Durability::Sync,
            "buffered" => Durability::Buffered,
            "none" => Durability::None,
            // Unset defaults to logging whenever a directory was named: a
            // caller who set `SOCHDB_MEMORY_DIR` has already said they want
            // their memory to survive.
            "" if data_dir.is_some() => Durability::Buffered,
            "" => Durability::None,
            // A value that is set but not understood is refused rather than
            // defaulted. Quietly reading `fsync` or `Sync ` as `Buffered` would
            // hand an operator who asked to survive power loss a tier that does
            // not, and every downstream signal — including the server's own
            // "write-ahead logged" startup line — would agree that they got
            // what they asked for.
            other => {
                return Err(MemoryError::Durability(format!(
                    "SOCHDB_MEMORY_DURABILITY={other:?} is not one of none, buffered, sync"
                )));
            }
        };

        let default_config = MemoryStoreConfig::default();
        let checkpoint_after_records = match std::env::var("SOCHDB_MEMORY_CHECKPOINT_RECORDS") {
            Err(_) => default_config.checkpoint_after_records,
            // Refused rather than defaulted, for the same reason an
            // unrecognised durability tier is: an operator who set a threshold
            // and got the built-in one instead has no way to find out.
            Ok(raw) => raw.trim().parse::<u64>().map_err(|e| {
                MemoryError::Durability(format!(
                    "SOCHDB_MEMORY_CHECKPOINT_RECORDS={raw:?} is not a record count ({e})"
                ))
            })?,
        };

        Self::with_embedder(
            data_dir.as_deref(),
            MemoryStoreConfig {
                durability,
                checkpoint_after_records,
                ..default_config
            },
            sochdb_query::embedding_provider::embedder_from_env(),
        )
    }

    /// Replay the log into the indexes, if there is a log.
    ///
    /// Runs before the store is handed to any caller, so no query can observe
    /// a half-recovered namespace.
    fn recover(&self) -> MemoryResult<RecoveryReport> {
        let Some(wal) = &self.wal else {
            return Ok(RecoveryReport::default());
        };

        let mut report = RecoveryReport::default();
        let mut requeue: Vec<EnrichmentJob> = Vec::new();

        wal.replay(|record| match record {
            MemoryRecord::Episode(episode) => {
                // `enriched` is deliberately reset: the vector that justified
                // it was never logged, so claiming enrichment here would leave
                // the episode permanently invisible to the vector lane while
                // reporting that it is not.
                let mut restored = *episode;
                restored.enriched = false;
                let namespace = restored.namespace.clone();
                let episode_id = restored.id.0;
                let text = restored.text.clone();

                // Counted and re-queued only when it is genuinely new. An
                // episode present in both the snapshot and the log is one
                // episode; queueing it twice would insert two vectors for one
                // id into the namespace's HNSW, and the vector lane would then
                // return the same episode twice for a single query.
                if self
                    .namespace_or_create(&namespace)
                    .write()
                    .publish_episode(restored)
                {
                    report.episodes += 1;
                    requeue.push(EnrichmentJob {
                        namespace,
                        episode_id,
                        text,
                    });
                }
            }
            MemoryRecord::FactAdded { namespace, fact } => {
                if self
                    .namespace_or_create(&namespace)
                    .write()
                    .publish_fact(*fact)
                {
                    report.facts += 1;
                }
            }
            MemoryRecord::FactInvalidated {
                namespace,
                fact_id,
                t_invalid,
            } => {
                report.invalidations += 1;
                if let Some(handle) = self.namespace(&namespace) {
                    let mut ns = handle.write();
                    if let Some(fact) = ns.facts.iter_mut().find(|f| f.id == fact_id) {
                        fact.invalidate(t_invalid);
                    }
                }
            }
        })
        .map_err(MemoryError::Durability)?;

        // Enqueued after replay rather than during it, and without the live
        // depth bound: these jobs are not new work competing for admission,
        // they are work that was already accepted before the crash. See
        // [`EnrichmentQueue::enqueue_recovered`].
        report.enrichment_requeued = self.enrichment.enqueue_recovered(requeue) as u64;

        if !report.is_empty() {
            tracing::info!(
                episodes = report.episodes,
                facts = report.facts,
                invalidations = report.invalidations,
                enrichment_requeued = report.enrichment_requeued,
                "recovered agent memory from write-ahead log"
            );
        }
        Ok(report)
    }

    /// Whether acknowledged writes are being logged.
    pub fn is_durable(&self) -> bool {
        self.wal.is_some()
    }

    /// What this store recovered from its log when it opened.
    ///
    /// Exposed so an operator can assert on it rather than read it out of a log
    /// line: "how much memory came back" is the one question a restart raises.
    pub fn recovery_report(&self) -> RecoveryReport {
        self.recovery
    }

    /// Force every logged write all the way to disk.
    ///
    /// A no-op for an un-logged store. Lets a [`Durability::Buffered`] deployment
    /// take a durability point before a planned restart without paying `fsync`
    /// on every episode.
    pub fn sync(&self) -> MemoryResult<()> {
        match &self.wal {
            Some(wal) => wal.sync().map_err(MemoryError::Durability),
            None => Ok(()),
        }
    }

    /// Log a record, or do nothing if this store keeps no log.
    fn log(&self, record: &MemoryRecord) -> MemoryResult<()> {
        match &self.wal {
            Some(wal) => wal.append(record).map_err(MemoryError::Durability),
            None => Ok(()),
        }
    }

    /// Records appended since the last checkpoint. Zero for an un-logged store.
    pub fn records_since_checkpoint(&self) -> u64 {
        self.wal.as_ref().map_or(0, |w| w.since_checkpoint())
    }

    /// The configured automatic-checkpoint threshold; `0` means manual only.
    pub fn checkpoint_threshold(&self) -> u64 {
        self.config.checkpoint_after_records
    }

    /// Whether enough has been logged to justify the cost of a checkpoint.
    ///
    /// Always false when [`MemoryStoreConfig::checkpoint_after_records`] is
    /// zero, which is how a deployment asks to checkpoint only on demand.
    pub fn checkpoint_due(&self) -> bool {
        let threshold = self.config.checkpoint_after_records;
        threshold > 0 && self.records_since_checkpoint() >= threshold
    }

    /// Compact the log: snapshot live state, then drop the log it subsumes.
    ///
    /// Returns the number of records written to the snapshot, or `Ok(0)` for an
    /// un-logged store, for which there is nothing to compact.
    ///
    /// Without this the log is append-only for the lifetime of the deployment:
    /// it ends at `ENOSPC`, and long before that every restart replays the
    /// entire history of the store rather than its current contents, so startup
    /// time grows without bound even though the memory being restored does not.
    ///
    /// # What a snapshot contains
    ///
    /// Live state, expressed in the same records the log carries: one `Episode`
    /// per episode, and one `FactAdded` per fact *carrying its current temporal
    /// coordinates*. A retracted fact therefore needs no accompanying
    /// `FactInvalidated` — it is snapshotted already closed — which is what
    /// keeps the snapshot proportional to live state instead of to history.
    ///
    /// # Cost
    ///
    /// Writers are excluded for the duration. The snapshot is built into a
    /// `Vec` under the guard and encoded outside it in the sense that matters —
    /// the namespace locks are released as soon as the records are cloned —
    /// but the guard itself spans the `fsync`, because releasing it before the
    /// log is truncated would reintroduce the race it exists to prevent.
    pub fn checkpoint(&self) -> MemoryResult<u64> {
        let Some(wal) = &self.wal else {
            return Ok(0);
        };

        let _exclusive = self.checkpoint_lock.write();

        // Checked before the record list is built, not after. Assembling it
        // deep-clones every episode's text and every fact in every namespace,
        // and a poisoned log will reject the result unconditionally — so doing
        // the work first would charge a failing store the full cost of a
        // snapshot on every attempt, with all writers blocked behind this
        // guard for each one.
        if wal.is_poisoned() {
            return Err(MemoryError::Durability(
                "memory WAL is no longer accepting writes after an earlier failure".to_string(),
            ));
        }

        let handles: Vec<(String, NamespaceHandle)> = self
            .namespaces
            .read()
            .iter()
            .map(|(name, handle)| (name.clone(), Arc::clone(handle)))
            .collect();

        let mut records = Vec::new();
        for (name, handle) in handles {
            let ns = handle.read();
            records.extend(
                ns.episodes
                    .values()
                    .map(|e| MemoryRecord::Episode(Box::new(e.clone()))),
            );
            records.extend(ns.facts.iter().map(|f| MemoryRecord::FactAdded {
                namespace: name.clone(),
                fact: Box::new(f.clone()),
            }));
        }

        wal.checkpoint(&records).map_err(MemoryError::Durability)
    }

    pub fn enrichment_queue(&self) -> &EnrichmentQueue {
        &self.enrichment
    }

    /// Handle for an existing namespace, or `None`.
    ///
    /// The store-wide read guard is dropped before the caller can touch the
    /// namespace, so a long read of one namespace never blocks a write to
    /// another.
    pub(crate) fn namespace(&self, name: &str) -> Option<NamespaceHandle> {
        self.namespaces.read().get(name).cloned()
    }

    /// Handle for a namespace, creating it if this is its first write.
    ///
    /// Takes the cheap read path first: after the first write to a namespace,
    /// which is the overwhelming majority of calls, no writer ever touches the
    /// directory. `entry` on the write path keeps a concurrent creation of the
    /// same namespace from producing two sets of indexes.
    pub(crate) fn namespace_or_create(&self, name: &str) -> NamespaceHandle {
        if let Some(handle) = self.namespaces.read().get(name) {
            return Arc::clone(handle);
        }
        Arc::clone(
            self.namespaces
                .write()
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(RwLock::new(NamespaceIndexes::new()))),
        )
    }

    /// Write episode: lexical lanes indexed synchronously; enrichment queued async.
    pub fn write_episode(&self, write: EpisodeWrite) -> MemoryResult<WriteResult> {
        let start = Instant::now();
        let t_created = self.hlc.next();
        // Default validity-start to wall-clock unix milliseconds, matching the
        // `as_of=<unix_ms>` query contract. `t_created` is a raw HLC tick
        // (`physical_micros << 16 | logical`, ~1e21), so defaulting to it made
        // the bi-temporal filter `t_valid_from <= as_of` always false for any
        // realistic `as_of` — silently returning zero results for every
        // episode written without an explicit validity time (the common case).
        // Callers that pass `t_valid_from` keep their own time domain.
        let t_valid = write
            .t_valid_from
            .unwrap_or_else(|| HybridLogicalClock::physical_time(t_created) / 1000);

        let handle = self.namespace_or_create(&write.namespace);
        let episode = {
            let mut ns = handle.write();

            let episode_id = EpisodeId(ns.next_episode_id);
            ns.next_episode_id += 1;

            let episode = Episode {
                id: episode_id,
                namespace: write.namespace.clone(),
                text: write.text.clone(),
                t_created,
                t_valid_from: t_valid,
                enriched: false,
                metadata: write.metadata.clone(),
            };

            // Un-logged stores publish under the same lock that allocated the
            // id: there is nothing to order against, so splitting the critical
            // section would only double lock traffic on the hot path.
            if self.wal.is_none() {
                ns.publish_episode(episode.clone());
            }
            episode
        };
        let episode_id = episode.id;

        // Write-*ahead*: the record is on the medium its durability tier
        // promises before the episode becomes visible to any reader. Publishing
        // first would let a query observe — and an agent act on — an episode
        // that a crash one instant later erases.
        //
        // The checkpoint guard spans both steps. A checkpoint that ran between
        // them would snapshot a namespace that does not yet contain this
        // episode and then truncate the log record that does, erasing a write
        // this call is about to report as successful.
        if self.wal.is_some() {
            let _checkpoint_guard = self.checkpoint_lock.read();
            self.log(&MemoryRecord::Episode(Box::new(episode.clone())))?;
            handle.write().publish_episode(episode);
        }

        let job = EnrichmentJob {
            namespace: write.namespace.clone(),
            episode_id: episode_id.0,
            text: write.text.clone(),
        };

        let enrichment_queued = self.enrichment.try_enqueue(job.clone()).is_ok();
        let ingestion_lag_us = start.elapsed().as_micros() as u64;

        let result = WriteResult {
            episode_id,
            t_created,
            lexical_indexed: true,
            ingestion_lag_us,
            enrichment_queued,
        };

        if self.config.enrich_on_write {
            let _ = self.enrich_episode(&job);
        }

        Ok(result)
    }

    /// Ingest conversation turns as WINDOWED, speaker-prefixed episodes — the
    /// ingestion shape that maximizes retrieval recall.
    ///
    /// Writing one bare-text episode per turn strips the conversational context
    /// an episode needs to be retrievable: a short, context-dependent turn like
    /// "Yeah, May 7th" cannot be matched to "When did she go?" by any embedder.
    /// Grouping `window` consecutive turns into a single `"speaker: text"`-
    /// formatted episode restores that context, and a `stride` smaller than
    /// `window` produces OVERLAPPING episodes so no turn is stranded near a chunk
    /// boundary without context.
    ///
    /// Measured on LoCoMo (same 384-d embedder) — retrieval *structure*, not
    /// embedder strength, is the dominant recall lever:
    /// - one bare turn per episode:            hit@10 ~0.61
    /// - `window=5,  stride=5` (disjoint):     hit@10 ~0.89
    /// - `window=10, stride=4` (≈40% overlap): hit@10 ~0.91  (best)
    ///
    /// Good defaults for chat: `window≈10`, `stride≈window*0.4`. Too-large
    /// windows over-coarsen (the vector lane loses discrimination); too-small a
    /// stride (e.g. 1) crowds top-k with near-duplicate episodes and *hurts*.
    /// `window` is clamped to ≥1 and `stride` to `1..=window`.
    ///
    /// Returns one [`WriteResult`] per episode written (i.e. per window).
    pub fn write_turns(
        &self,
        namespace: &str,
        turns: &[ConversationTurn],
        window: usize,
        stride: usize,
        t_valid_from: Option<u64>,
    ) -> MemoryResult<Vec<WriteResult>> {
        let w = window.max(1);
        let s = stride.clamp(1, w);
        let mut out = Vec::new();
        let mut start = 0;
        while start < turns.len() {
            let end = (start + w).min(turns.len());
            let group = &turns[start..end];
            let text = group
                .iter()
                .map(|t| {
                    if t.speaker.is_empty() {
                        t.text.clone()
                    } else {
                        format!("{}: {}", t.speaker, t.text)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !text.trim().is_empty() {
                out.push(self.write_episode(EpisodeWrite {
                    namespace: namespace.to_string(),
                    text,
                    t_valid_from,
                    metadata: None,
                })?);
            }
            if end == turns.len() {
                break;
            }
            start += s;
        }
        Ok(out)
    }

    pub fn get_episode(&self, namespace: &str, id: EpisodeId) -> MemoryResult<Episode> {
        let handle = self
            .namespace(namespace)
            .ok_or_else(|| MemoryError::NamespaceNotFound(namespace.to_string()))?;
        let ns = handle.read();
        ns.episodes
            .get(&id.0)
            .cloned()
            .ok_or(MemoryError::EpisodeNotFound(id.0))
    }

    pub fn namespace_bm25(&self, namespace: &str) -> Option<Arc<InvertedIndex>> {
        // BM25 index is behind the namespace lock — expose search via store methods instead
        let _ = namespace;
        None
    }

    pub fn search_bm25(&self, namespace: &str, query: &str, k: usize) -> Vec<(u64, f32)> {
        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        ns.bm25.search(query, k)
    }

    pub fn search_trigram_literal(
        &self,
        namespace: &str,
        literal: &str,
        k: usize,
    ) -> Vec<(u64, f32)> {
        let trigrams = sochdb_query::trigram_index::trigrams_of(literal);
        if trigrams.is_empty() {
            return Vec::new();
        }
        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        ns.trigram
            .candidates(&trigrams)
            .into_iter()
            .take(k)
            .map(|doc_id| (doc_id, 1.0))
            .collect()
    }

    pub fn episode_text(&self, namespace: &str, doc_id: u64) -> Option<String> {
        let handle = self.namespace(namespace)?;
        let ns = handle.read();
        ns.episodes.get(&doc_id).map(|e| e.text.clone())
    }

    pub fn add_fact(&self, namespace: &str, mut fact: FactEdge) -> MemoryResult<FactId> {
        // Acquired before any namespace lock. Every path that takes both must
        // take them in this order or a checkpoint waiting on a namespace read
        // deadlocks against a writer waiting on the checkpoint guard.
        let _checkpoint_guard = self.wal.is_some().then(|| self.checkpoint_lock.read());
        let handle = self.namespace_or_create(namespace);
        let mut ns = handle.write();
        let id = FactId(ns.next_fact_id);
        ns.next_fact_id += 1;
        fact.id = id;

        // Unlike episodes, fact mutations are applied under the namespace lock
        // held across the append. Episode publishing is order-independent —
        // it is keyed by id and advances a counter by `max` — so splitting its
        // critical section is free. Fact state is not: see `invalidate_fact`.
        // Facts are asserted orders of magnitude less often than episodes are
        // written, so the serialization costs nothing measurable.
        if self.wal.is_some() {
            self.log(&MemoryRecord::FactAdded {
                namespace: namespace.to_string(),
                fact: Box::new(fact.clone()),
            })?;
        }
        ns.facts.push(fact);
        Ok(id)
    }

    pub fn facts_valid_at(&self, namespace: &str, tau: u64) -> Vec<FactEdge> {
        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        ns.facts
            .iter()
            .filter(|f| f.is_valid_at(tau))
            .cloned()
            .collect()
    }

    /// Close a fact's valid-time interval at `t_invalid`.
    ///
    /// Returns whether the fact was found. Errors only if the invalidation
    /// could not be logged — in which case the fact is left *valid*, since a
    /// retraction that a restart would undo is worse than one that visibly
    /// failed: the caller can retry, but cannot detect silent resurrection.
    ///
    /// The namespace write lock is held across the append. `close_valid_time`
    /// is a plain overwrite, so this is the one record type whose *application*
    /// order changes the result: two concurrent invalidations of the same fact
    /// could otherwise append as `(100, 200)` — appends are serialized by the
    /// log's own writer lock — and then apply as `(200, 100)`, leaving live
    /// state at `valid_to = 100` while the log replays to `200`. The store
    /// would answer `facts_valid_at(150)` one way before a restart and the
    /// other way after it.
    pub fn invalidate_fact(
        &self,
        namespace: &str,
        fact_id: FactId,
        t_invalid: u64,
    ) -> MemoryResult<bool> {
        let Some(handle) = self.namespace(namespace) else {
            return Ok(false);
        };
        // See `add_fact`: checkpoint guard before namespace lock, always.
        let _checkpoint_guard = self.wal.is_some().then(|| self.checkpoint_lock.read());
        let mut ns = handle.write();

        if !ns.facts.iter().any(|f| f.id == fact_id) {
            return Ok(false);
        }
        if self.wal.is_some() {
            self.log(&MemoryRecord::FactInvalidated {
                namespace: namespace.to_string(),
                fact_id,
                t_invalid,
            })?;
        }

        if let Some(fact) = ns.facts.iter_mut().find(|f| f.id == fact_id) {
            fact.invalidate(t_invalid);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn episode_count(&self, namespace: &str) -> usize {
        self.namespace(namespace)
            .map(|handle| handle.read().episodes.len())
            .unwrap_or(0)
    }
}
